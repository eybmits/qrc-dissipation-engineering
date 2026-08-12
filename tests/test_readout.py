import numpy as np

from qrc import readout


def test_observable_count():
    obs = readout.pauli_observables(5, max_weight=2)
    # 3*5 singles + 3*C(5,2)=30 pairs = 45
    assert len(obs) == 45
    singles = [o for o in obs if len(o.name) == 2]
    assert len(singles) == 15


def test_capacity_perfect_and_zero():
    y = np.linspace(0, 1, 100)
    assert np.isclose(readout.capacity(y, 2 * y + 3), 1.0)  # affine -> C=1
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(100)
    assert readout.capacity(y, noise) < 0.3


def test_linear_readout_recovers_weights():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 6))
    w_true = rng.standard_normal(6)
    y = X @ w_true
    Xb = readout.add_bias(X)
    w = readout.train_readout(Xb, y)
    assert np.allclose(w[:-1], w_true, atol=1e-6)
    assert abs(w[-1]) < 1e-6


def test_fit_and_score_split():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 4))
    w_true = rng.standard_normal(4)
    y = X @ w_true + 0.01 * rng.standard_normal(300)
    res = readout.fit_and_score(X[:200], y[:200], X[200:], y[200:], metric="capacity")
    assert res.test_metric > 0.95
