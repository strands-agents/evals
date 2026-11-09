import numpy as np
import pytest
from scipy.stats import multivariate_normal

from strands_evals.optimization.bayesbench import Mixture, Posterior, ScaleWeight, log_evidence, prior_scale_weights


def random_problem(seed, n, p=8):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, p)) < 0.4).astype(float)
    x[:, 0] = 1.0
    prior_mean = rng.normal(0, 0.1, p)
    prior_var = rng.uniform(0.01, 0.3, p) ** 2
    return x, rng.uniform(0, 1, n), rng.uniform(1e-3, 1e-1, n), prior_mean, prior_var


def direct_log_evidence(x, y, noise_var, prior_mean, prior_var):
    """The density of y under N(x mu0, x S0 x' + N), computed directly; a test oracle only."""
    cov = x @ np.diag(prior_var) @ x.T + np.diag(np.broadcast_to(noise_var, y.shape))
    return multivariate_normal(mean=x @ prior_mean, cov=cov).logpdf(y)


def shared_scale_prior(p):
    return lambda scales: (np.zeros(p), np.full(p, scales["effect"] ** 2))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", [3, 30])
def test_the_log_evidence_is_the_marginal_density_of_the_observations(seed, n):
    x, y, noise_var, prior_mean, prior_var = random_problem(seed, n)
    assert log_evidence(x, y, noise_var, prior_mean, prior_var) == pytest.approx(
        direct_log_evidence(x, y, noise_var, prior_mean, prior_var), rel=1e-9
    )


def test_the_log_evidence_holds_when_the_noise_is_tiny():
    x, y, _, prior_mean, prior_var = random_problem(0, 6)
    assert log_evidence(x, y, 1e-6, prior_mean, prior_var) == pytest.approx(
        direct_log_evidence(x, y, 1e-6, prior_mean, prior_var), rel=1e-7
    )


def test_no_observations_have_log_evidence_0():
    assert log_evidence(np.zeros((0, 3)), [], 0.1, np.zeros(3), np.ones(3)) == 0.0


def test_the_weights_are_each_points_normalized_evidence_likeliest_first():
    x, y, noise_var, _, _ = random_problem(1, 25)
    prior = shared_scale_prior(x.shape[1])
    grid = {"effect": [0.01, 0.03, 0.1, 0.3, 1.0]}
    weighted = prior_scale_weights(x, y, noise_var, prior, grid)

    logs = np.array([log_evidence(x, y, noise_var, *prior({"effect": s})) for s in grid["effect"]])
    expected = np.exp(logs - logs.max()) / np.exp(logs - logs.max()).sum()
    assert sum(w.weight for w in weighted) == pytest.approx(1.0)
    assert [w.weight for w in weighted] == sorted((w.weight for w in weighted), reverse=True)
    for point in weighted:
        assert point.weight == pytest.approx(expected[grid["effect"].index(point.scales["effect"])], rel=1e-9)


@pytest.mark.parametrize("effect, likeliest", [(0.0, 0.003), (0.5, 0.3)])
def test_the_likeliest_scale_is_the_size_of_the_effects(effect, likeliest):
    rng = np.random.default_rng(2)
    x = np.column_stack([np.ones(60), rng.random((60, 3)) < 0.5]).astype(float)
    theta = np.array([0.0, effect, -effect, effect])
    y = x @ theta + rng.normal(0, 0.01, 60)

    def prior(scales):  # the intercept's scale is fixed
        return np.zeros(4), np.array([1.0, *[scales["effect"] ** 2] * 3])

    weighted = prior_scale_weights(x, y, 0.01**2, prior, {"effect": [0.003, 0.03, 0.3, 3.0]})
    assert weighted[0].scales == {"effect": likeliest}


def test_the_grid_is_the_product_of_its_groups_and_ties_keep_its_order():
    x, y, noise_var, _, _ = random_problem(3, 10, p=3)
    seen = []

    def prior(scales):
        seen.append(scales)
        return np.zeros(3), np.full(3, scales["main"] ** 2)  # "unused" never changes the prior

    weighted = prior_scale_weights(x, y, noise_var, prior, {"main": [0.1, 1.0], "unused": [1, 2, 3]})
    assert seen == [{"main": m, "unused": u} for m in (0.1, 1.0) for u in (1, 2, 3)]
    tied = [w.scales["unused"] for w in weighted if w.scales["main"] == weighted[0].scales["main"]]
    assert tied == [1, 2, 3]
    assert weighted[0].weight == pytest.approx(weighted[1].weight)


def test_an_empty_grid_is_one_point_with_all_the_weight():
    x, y, noise_var, prior_mean, prior_var = random_problem(4, 5)
    assert prior_scale_weights(x, y, noise_var, lambda s: (prior_mean, prior_var), {}) == [ScaleWeight({}, 1.0)]


def test_a_group_with_no_scale_to_try_is_rejected():
    x, y, noise_var, _, _ = random_problem(4, 5)
    with pytest.raises(ValueError, match="no sd to try"):
        prior_scale_weights(x, y, noise_var, shared_scale_prior(x.shape[1]), {"effect": []})


def test_a_prior_of_the_wrong_length_is_rejected():
    x, y, noise_var, _, _ = random_problem(4, 5)
    with pytest.raises(ValueError, match="needs 8 effects"):
        prior_scale_weights(x, y, noise_var, shared_scale_prior(3), {"effect": [0.1]})


def two_posteriors(seed=0):
    x, y, noise_var, prior_mean, _ = random_problem(seed, 12, p=4)
    narrow = Posterior(prior_mean, np.full(4, 0.01**2)).condition(x, y, noise_var)
    wide = Posterior(prior_mean, np.full(4, 1.0)).condition(x, y, noise_var)
    return narrow, wide


def test_a_mixture_of_one_posterior_is_that_posterior():
    posterior, _ = two_posteriors()
    mixture = Mixture([(1.0, posterior)])
    rows = np.eye(4)
    for got, want in zip(mixture.predictive(rows), posterior.predictive(rows), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-15)
    np.testing.assert_allclose(mixture.mean, posterior.mean, atol=1e-15)
    assert mixture.covariance(rows[0], rows[1]) == pytest.approx(posterior.covariance(rows[0], rows[1]), abs=1e-15)
    assert (mixture.p, mixture.n) == (4, 12)


def test_the_mixtures_moments_follow_the_law_of_total_variance():
    narrow, wide = two_posteriors()
    mixture = Mixture([(1.0, narrow), (3.0, wide)])
    np.testing.assert_allclose(mixture.weights, [0.25, 0.75])
    a, b = np.array([1.0, 1.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0, 1.0])
    (ma_n, mb_n), (va_n, vb_n) = narrow.predictive(np.vstack([a, b]))
    (ma_w, mb_w), (va_w, vb_w) = wide.predictive(np.vstack([a, b]))
    mean_a = 0.25 * ma_n + 0.75 * ma_w
    mean_b = 0.25 * mb_n + 0.75 * mb_w

    means, variances = mixture.predictive(np.vstack([a, b]))
    np.testing.assert_allclose(means, [mean_a, mean_b], atol=1e-14)
    assert variances[0] == pytest.approx(0.25 * (va_n + ma_n**2) + 0.75 * (va_w + ma_w**2) - mean_a**2, rel=1e-9)
    assert mixture.covariance(a, a) == pytest.approx(variances[0], rel=1e-12)
    expected = (
        0.25 * (narrow.covariance(a, b) + ma_n * mb_n) + 0.75 * (wide.covariance(a, b) + ma_w * mb_w) - mean_a * mean_b
    )
    assert mixture.covariance(a, b) == pytest.approx(expected, rel=1e-9)
    np.testing.assert_allclose(mixture.mean, 0.25 * narrow.mean + 0.75 * wide.mean, atol=1e-14)


def test_samples_match_the_mixtures_moments():
    narrow, wide = two_posteriors(1)
    mixture = Mixture([(0.4, narrow), (0.6, wide)])
    a, b = np.array([1.0, 1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0, 0.0])
    draws = mixture.sample(200_000, np.random.default_rng(3))
    assert draws.shape == (200_000, 4)
    values_a, values_b = draws @ a, draws @ b
    (mean_a, _), (var_a, _) = mixture.predictive(np.vstack([a, b]))
    assert values_a.mean() == pytest.approx(mean_a, abs=4 * np.sqrt(var_a / 200_000))
    assert values_a.var() == pytest.approx(var_a, rel=0.02)
    assert np.cov(values_a, values_b)[0, 1] == pytest.approx(mixture.covariance(a, b), rel=0.05, abs=1e-4)


def test_each_draws_component_is_picked_by_weight():
    low = Posterior([0.0], [1e-6])
    high = Posterior([10.0], [1e-6])
    draws = Mixture([(0.3, low), (0.7, high)]).sample(20_000, np.random.default_rng(4))[:, 0]
    assert np.mean(draws > 5) == pytest.approx(0.7, abs=0.02)
    assert Mixture([(1.0, low)]).sample(0, np.random.default_rng(4)).shape == (0, 1)


@pytest.mark.parametrize(
    "components, message",
    [
        ([], "at least one component"),
        ([(-1.0, Posterior([0.0], [1.0]))], "finite, >= 0"),
        ([(0.0, Posterior([0.0], [1.0]))], "not all 0"),
        ([(1.0, Posterior([0.0], [1.0])), (1.0, Posterior([0.0, 0.0], [1.0, 1.0]))], "same p and n"),
        ([(1.0, Posterior([0.0], [1.0])), (1.0, Posterior([0.0], [1.0]).condition([1.0], [1.0], 1.0))], "same p and n"),
    ],
)
def test_a_bad_mixture_is_rejected(components, message):
    with pytest.raises(ValueError, match=message):
        Mixture(components)
