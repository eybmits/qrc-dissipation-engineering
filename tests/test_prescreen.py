from __future__ import annotations

import numpy as np

from qrc.prescreen import (
    empirical_regret,
    feature_remainder_moment_bounds,
    geometric_lag_remainder_bound,
    prescreen_candidates,
    regularized_score_error_bound,
    safe_shortlist_indices,
    selected_regret_bound,
    shortlist_regret,
    top1_is_certified,
    topk_set_is_certified,
)


def test_two_delta_regret_bound_by_random_adversarial_perturbations():
    rng = np.random.default_rng(4)
    for _ in range(500):
        predicted = rng.uniform(0, 1, size=9)
        delta = rng.uniform(0, 0.15)
        true = predicted + rng.uniform(-delta, delta, size=predicted.size)
        chosen = int(np.argmax(predicted))
        assert empirical_regret(true, chosen) <= selected_regret_bound(delta) + 1e-12


def test_margin_certificate_recovers_unique_winner():
    predicted = np.array([0.8, 0.55, 0.3])
    delta = 0.1
    assert top1_is_certified(predicted, delta)
    rng = np.random.default_rng(5)
    for _ in range(1000):
        true = predicted + rng.uniform(-delta, delta, size=3)
        assert int(np.argmax(true)) == 0


def test_safe_shortlist_never_discards_true_winner():
    rng = np.random.default_rng(6)
    for _ in range(500):
        predicted = rng.uniform(0, 1, size=8)
        delta = rng.uniform(0, 0.2)
        keep = safe_shortlist_indices(predicted, delta)
        for _ in range(20):
            true = predicted + rng.uniform(-delta, delta, size=8)
            assert int(np.argmax(true)) in keep


def test_topk_boundary_certificate_is_sufficient():
    predicted = np.array([0.9, 0.8, 0.7, 0.2])
    delta = 0.1
    assert topk_set_is_certified(predicted, 3, delta)
    rng = np.random.default_rng(7)
    for _ in range(1000):
        true = predicted + rng.uniform(-delta, delta, size=4)
        assert set(np.argsort(-true)[:3]) == {0, 1, 2}


def test_regularized_score_error_bound_holds_for_random_psd_matrices():
    rng = np.random.default_rng(8)
    for _ in range(200):
        dimension = 5
        A = rng.normal(size=(dimension, dimension))
        G_hat = A @ A.T
        E = rng.normal(size=(dimension, dimension))
        E = 0.5 * (E + E.T)
        eps_G = 0.02
        E *= eps_G / max(np.linalg.norm(E, 2), 1e-15)
        G = G_hat + E
        # Shift only if needed to preserve PSD, then account for the actual error.
        minimum = np.linalg.eigvalsh(G).min()
        if minimum < 0:
            G += (-minimum + 1e-12) * np.eye(dimension)
        actual_eps_G = np.linalg.norm(G - G_hat, 2)
        g_hat = rng.normal(size=dimension)
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        eps_g = 0.015
        g = g_hat + eps_g * direction
        ridge = 0.3
        score_hat = float(g_hat @ np.linalg.solve(G_hat + ridge * np.eye(dimension), g_hat))
        score = float(g @ np.linalg.solve(G + ridge * np.eye(dimension), g))
        bound = regularized_score_error_bound(g_hat, eps_g, actual_eps_G, ridge)
        assert abs(score - score_hat) <= bound + 1e-10


def test_feature_remainder_bounds_moments():
    rng = np.random.default_rng(9)
    B, r = 1.2, 0.08
    eps_g, eps_G = feature_remainder_moment_bounds(B, r)
    proxy = rng.normal(size=(2000, 4))
    norms = np.linalg.norm(proxy, axis=1)
    proxy *= np.minimum(1.0, B / np.maximum(norms, 1e-15))[:, None]
    error = rng.normal(size=(2000, 4))
    norms = np.linalg.norm(error, axis=1)
    error *= np.minimum(1.0, r / np.maximum(norms, 1e-15))[:, None]
    target = rng.uniform(-1, 1, size=2000)
    true = proxy + error
    g_error = np.linalg.norm(np.mean(true * target[:, None], axis=0) - np.mean(proxy * target[:, None], axis=0))
    G_error = np.linalg.norm(true.T @ true / len(true) - proxy.T @ proxy / len(proxy), 2)
    assert g_error <= eps_g + 1e-12
    assert G_error <= eps_G + 1e-12


def test_geometric_lag_bound_matches_explicit_tails():
    eta, L, epsilon = 0.7, 8, 0.04
    B1, B2 = 1.3, 0.9
    bound = geometric_lag_remainder_bound(eta, L, epsilon, B1, B2)
    first = epsilon * sum(B1 * eta**d for d in range(L + 1, 10000))
    second = epsilon**2 * sum(
        B2 * eta ** (b - 1)
        for b in range(L + 1, 1000)
        for _a in range(b)
    )
    assert np.isclose(bound, first + second, rtol=1e-12, atol=1e-12)


def test_prescreen_certificate_and_shortlist_regret():
    predicted = [0.72, 0.69, 0.2]
    true = [0.68, 0.73, 0.21]
    certificate = prescreen_candidates(
        predicted,
        ["local", "mixed", "collective"],
        uniform_error=0.05,
    )
    assert set(certificate.shortlist) == {"local", "mixed"}
    assert certificate.discarded == ("collective",)
    assert shortlist_regret(true, [0, 1]) == 0.0
