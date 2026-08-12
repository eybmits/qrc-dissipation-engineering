import numpy as np

from qrc import dissipators as dsp
from qrc import liouvillian as dense_lio
from qrc import readout, reservoirs as res, sparse_evolve, tasks
from qrc.reservoirs import LindbladReservoir, ising_xx_hamiltonian, transverse_drive


def _jump_families(J, n, target_strength):
    edges = dsp.graph_edges(J)
    pair_edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if abs(J[i, j]) > 1e-9
    ]
    return [
        dsp.normalize_jump_strength(dsp.local_loss(n, 1.0), target_strength),
        dsp.normalize_jump_strength(dsp.dephasing(n, 1.0), target_strength),
        dsp.normalize_jump_strength(dsp.thermal(n, 1.0, 0.3), target_strength),
        dsp.normalize_jump_strength(dsp.collective_loss(n, 1.0), target_strength),
        dsp.normalize_jump_strength(dsp.exchange(n, 1.0, edges), target_strength),
        dsp.normalize_jump_strength(dsp.pair_loss(n, 1.0, pair_edges), target_strength),
    ]


def _stm_capacity(X, post_inputs, train_len, test_len):
    Xb = readout.add_bias(X)
    tr = slice(0, train_len)
    te = slice(train_len, train_len + test_len)
    total = 0.0
    for delay in (1, 2, 3):
        y = tasks.delayed_target(post_inputs, delay)
        train_mask = np.zeros(len(y), dtype=bool)
        test_mask = np.zeros(len(y), dtype=bool)
        train_mask[tr] = True
        test_mask[te] = True
        train_mask &= ~np.isnan(y)
        test_mask &= ~np.isnan(y)
        w = readout.train_readout(Xb[train_mask], y[train_mask], ridge=1e-8)
        total += readout.capacity(y[test_mask], readout.predict(Xb[test_mask], w))
    return total


def test_sparse_superoperators_equal_dense():
    for n in (3, 4):
        rng = np.random.default_rng(100 + n)
        J = res.random_couplings(n, 1.0, rng)
        H = ising_xx_hamiltonian(J, h=0.5, n_qubits=n) + 0.5 * transverse_drive(n)
        target = dsp.jump_strength(dsp.local_loss(n, 1.0))

        comm_diff = np.max(np.abs(
            sparse_evolve.commutator_super(H).toarray() - dense_lio.commutator_super(H)
        ))
        assert comm_diff < 1e-10

        for jumps in _jump_families(J, n, target):
            dense = dense_lio.lindbladian(H, jumps)
            sparse = sparse_evolve.lindbladian(H, jumps).toarray()
            assert np.max(np.abs(sparse - dense)) < 1e-10

            jump_dense = dense_lio.dissipator_super(jumps[0][0])
            jump_sparse = sparse_evolve.dissipator_super(jumps[0][0]).toarray()
            assert np.max(np.abs(jump_sparse - jump_dense)) < 1e-10


def test_sparse_matrix_free_features_match_dense_lindblad_stm():
    h, dt = 0.5, 0.5
    wash, train, test = 4, 12, 8
    total = wash + train + test

    for n in (4, 5):
        rng = np.random.default_rng(200 + n)
        J = res.random_couplings(n, 1.0, rng)
        H0 = ising_xx_hamiltonian(J, h, n)
        Hx = transverse_drive(n)
        H_static = H0 + h * Hx
        H_drive = h * Hx
        target = dsp.jump_strength(dsp.local_loss(n, 1.0))
        jumps = dsp.normalize_jump_strength(
            dsp.collective_loss(n, 1.0), target
        )
        inputs = tasks.stm_inputs(total, rng)
        obs = readout.pauli_observables(n, max_weight=2)

        dense = LindbladReservoir.from_terms(
            n, H_static, H_drive, jumps, dt, cache_propagators=False, quantize=None
        )
        sparse = sparse_evolve.SparseLindbladReservoir.from_terms(
            n, H_static, H_drive, jumps, dt
        )

        X_dense = dense.run(inputs, obs, washout=wash)
        X_sparse = sparse.run(inputs, obs, washout=wash)
        assert np.max(np.abs(X_sparse - X_dense)) < 1e-8

        post = inputs[wash:]
        dense_mc = _stm_capacity(X_dense, post, train, test)
        sparse_mc = _stm_capacity(X_sparse, post, train, test)
        assert abs(sparse_mc - dense_mc) < 1e-6
