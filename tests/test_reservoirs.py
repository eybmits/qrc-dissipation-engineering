import numpy as np

from qrc import reservoirs as res
from qrc import readout
from qrc import operators as op


def test_fn_preserves_trace_and_runs():
    rng = np.random.default_rng(0)
    N = 3
    J = res.random_couplings(N, 1.0, rng)
    r = res.FujiNakajimaReservoir(N, J, h=1.0, dt=1.0)
    rho = r.initial_state()
    for s in rng.uniform(0, 1, 10):
        rho = r.step(rho, float(s))
        assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)
        assert np.allclose(rho, rho.conj().T, atol=1e-9)


def test_cd_preserves_trace():
    rng = np.random.default_rng(1)
    N = 3
    J = res.random_couplings(N, 1.0, rng)
    r = res.continuous_dissipation(N, J, h=1.0, gamma=0.5, dt=1.0)
    rho = r.initial_state()
    for s in rng.uniform(0, 1, 8):
        rho = r.step(rho, float(s))
        assert np.isclose(np.trace(rho).real, 1.0, atol=1e-8)


def test_run_feature_shapes():
    rng = np.random.default_rng(2)
    N = 3
    J = res.random_couplings(N, 1.0, rng)
    obs = readout.pauli_observables(N, max_weight=2)
    inputs = rng.uniform(0, 1, 20)
    r = res.FujiNakajimaReservoir(N, J, h=1.0, dt=1.0)
    X = r.run(inputs, obs, washout=5, n_virtual=1)
    assert X.shape == (15, len(obs))
    Xv = r.run(inputs, obs, washout=5, n_virtual=4)
    assert Xv.shape == (15, 4 * len(obs))


def test_cd_propagator_cache_matches_stream():
    rng = np.random.default_rng(3)
    N = 2
    J = res.random_couplings(N, 1.0, rng)
    obs = readout.pauli_observables(N, max_weight=1)
    inputs = rng.integers(0, 2, 12).astype(float)
    r1 = res.continuous_dissipation(N, J, h=1.0, gamma=0.5, dt=1.0, cache_propagators=False)
    r2 = res.continuous_dissipation(N, J, h=1.0, gamma=0.5, dt=1.0, cache_propagators=True)
    X1 = r1.run(inputs, obs, washout=2, n_virtual=3)
    X2 = r2.run(inputs, obs, washout=2, n_virtual=3)
    assert np.allclose(X1, X2, atol=1e-7)


def test_heterogeneous_gamma_builds():
    rng = np.random.default_rng(4)
    N = 3
    J = res.random_couplings(N, 1.0, rng)
    gammas = np.array([0.1, 0.5, 1.0])
    r = res.continuous_dissipation(N, J, h=1.0, gamma=gammas, dt=1.0)
    rho = r.step(r.initial_state(), 0.4)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-8)
