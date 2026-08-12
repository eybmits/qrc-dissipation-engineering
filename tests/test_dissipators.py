import numpy as np

from qrc import dissipators as dsp
from qrc import diagnostics as diag
from qrc import reservoirs as res
from qrc.operators import sminus


def _collective_coefficients(jumps, n):
    assert len(jumps) == 1
    jump, rate = jumps[0]
    denominator = np.trace(sminus(0, n).conj().T @ sminus(0, n)).real
    coefficients = np.asarray(
        [
            np.trace(sminus(i, n).conj().T @ jump) / denominator
            for i in range(n)
        ],
        dtype=complex,
    )
    return coefficients, float(rate)


def test_rate_normalization_mean():
    g = dsp.loguniform_rates(8, 0.1, 10, np.random.default_rng(0), mean=0.5)
    assert np.isclose(g.mean(), 0.5)


def test_degree_rates_uniform_on_dense_graph():
    # dense coupling graph -> every node degree N-1 -> degree profile == uniform
    rng = np.random.default_rng(0)
    J = res.random_couplings(5, 1.0, rng)
    g = dsp.degree_rates(J, mean=0.3)
    assert np.allclose(g, 0.3)


def test_jump_strength_matching():
    rng = np.random.default_rng(1)
    J = res.random_couplings(5, 1.0, rng)
    n = 5
    target = dsp.jump_strength(dsp.local_loss(n, 0.3))
    for fam in [dsp.dephasing(n, 0.3), dsp.thermal(n, 0.3, 0.1),
                dsp.collective_loss(n, 0.3), dsp.exchange(n, 0.3, dsp.graph_edges(J))]:
        normed = dsp.normalize_jump_strength(fam, target)
        assert np.isclose(dsp.jump_strength(normed), target)


def test_collective_loss_preserves_complex_relative_phases():
    n = 4
    raw = np.asarray([1.0, 2.0j, -3.0, -4.0j], dtype=complex)
    jumps = dsp.collective_loss(n, 0.75, c=raw)
    actual, rate = _collective_coefficients(jumps, n)
    expected = raw * np.sqrt(n / np.sum(np.abs(raw) ** 2))

    assert rate == 0.75
    assert np.allclose(actual, expected)


def test_collective_loss_preserves_previous_real_normalization():
    n = 4
    raw = np.asarray([1.0, -2.0, 0.5, 3.0])
    jumps = dsp.collective_loss(n, 0.4, c=raw)
    actual, rate = _collective_coefficients(jumps, n)
    expected = raw * np.sqrt(n / np.sum(raw ** 2))

    assert rate == 0.4
    assert np.allclose(actual, expected)
    assert np.max(np.abs(actual.imag)) == 0.0


def test_collective_loss_rejects_invalid_complex_coefficients():
    with np.testing.assert_raises(ValueError):
        dsp.collective_loss(3, 1.0, c=[1.0, 2.0])
    with np.testing.assert_raises(ValueError):
        dsp.collective_loss(3, 1.0, c=[0.0, 0.0, 0.0])
    with np.testing.assert_raises(ValueError):
        dsp.collective_loss(3, 1.0, c=[1.0, np.nan, 1.0])


def test_one_body_lowering_budget_equals_kossakowski_trace():
    n = 4
    rng = np.random.default_rng(20260725)
    A = rng.normal(size=(3, n)) + 1j * rng.normal(size=(3, n))
    jumps = []
    for row in A:
        L = sum((row[i] * sminus(i, n) for i in range(n)), start=0j)
        jumps.append((L, 1.0))

    gamma = A.conj().T @ A
    expected = (2 ** (n - 1)) * np.trace(gamma).real
    assert np.isclose(dsp.jump_strength(jumps), expected)


def test_unitality_predicts_dephasing_exchange():
    # dephasing (sigma^z) and exchange (sigma^- sigma^+) are unital -> defect 0
    n = 4
    edges = [(0, 1), (1, 0), (2, 3), (3, 2)]
    assert diag.unitality_defect(dsp.dephasing(n, 0.5)) < 1e-12
    assert diag.unitality_defect(dsp.exchange(n, 0.5, edges)) < 1e-12
    # amplitude damping and collective loss are non-unital
    assert diag.unitality_defect(dsp.local_loss(n, 0.5)) > 1e-6
    assert diag.unitality_defect(dsp.collective_loss(n, 0.5)) > 1e-6
