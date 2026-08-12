import numpy as np

from qrc import tasks


def test_stm_delay():
    rng = np.random.default_rng(0)
    s = tasks.stm_inputs(50, rng)
    y = tasks.delayed_target(s, tau=3)
    assert np.all(np.isnan(y[:3]))
    assert np.allclose(y[3:], s[:-3])


def test_parity_target():
    s = np.array([1, 0, 1, 1, 0, 1], dtype=float)
    y = tasks.parity_target(s, tau=2)
    # y_k = (s_{k-1} + s_{k-2}) mod 2
    assert np.isnan(y[0]) and np.isnan(y[1])
    assert y[2] == (s[0] + s[1]) % 2
    assert y[3] == (s[1] + s[2]) % 2
    assert y[5] == (s[3] + s[4]) % 2


def test_narma_bounded():
    rng = np.random.default_rng(1)
    s = tasks.narma_inputs(500, rng)
    y = tasks.narma_target(s, order=10, input_scale=0.2)
    valid = y[~np.isnan(y)]
    assert np.all(np.isfinite(valid))
    assert np.max(np.abs(valid)) < 5.0  # stays bounded


def test_mackey_glass_range():
    series = tasks.mackey_glass_series(300, tau=17, discard=200)
    assert series.shape == (300,)
    scaled = tasks.rescale_unit(series)
    assert scaled.min() >= -1e-9 and scaled.max() <= 1 + 1e-9
    # MG is non-trivial (not constant)
    assert np.std(scaled) > 0.05
