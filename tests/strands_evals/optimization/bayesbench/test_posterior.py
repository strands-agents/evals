import numpy as np
import pytest

from strands_evals.optimization.bayesbench import Posterior


def naive_reference(prior_mean, prior_var, x, y, noise_var):
    """The textbook update with explicit inverses; a test oracle only."""
    prior_precision = np.diag(1.0 / prior_var)
    w = np.diag(1.0 / noise_var)
    cov = np.linalg.inv(prior_precision + x.T @ w @ x)
    return cov @ (prior_precision @ prior_mean + x.T @ w @ y), cov


def random_problem(seed, n, p=52):
    rng = np.random.default_rng(seed)
    prior_mean = np.zeros(p)
    prior_mean[0] = 0.45
    prior_var = rng.uniform(0.02, 0.3, p) ** 2
    x = (rng.random((n, p)) < 0.3).astype(float)
    x[:, 0] = 1.0
    return prior_mean, prior_var, x, rng.uniform(0, 1, n), rng.uniform(1e-4, 1e-2, n)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n", [3, 20, 80])  # fewer observations than effects, too
def test_the_posterior_matches_the_textbook_update(seed, n):
    prior_mean, prior_var, x, y, noise_var = random_problem(seed, n)
    posterior = Posterior(prior_mean, prior_var).condition(x, y, noise_var)
    mean, cov = naive_reference(prior_mean, prior_var, x, y, noise_var)

    np.testing.assert_allclose(posterior.mean, mean, atol=1e-10)
    rows = np.random.default_rng(seed).random((6, len(prior_mean)))
    predicted_mean, predicted_var = posterior.predictive(rows)
    np.testing.assert_allclose(predicted_mean, rows @ mean, atol=1e-10)
    np.testing.assert_allclose(predicted_var, np.diag(rows @ cov @ rows.T), atol=1e-10)
    assert posterior.covariance(rows[0], rows[1]) == pytest.approx(rows[0] @ cov @ rows[1], abs=1e-10)


def test_one_effect_matches_the_normal_normal_answer():
    # Prior N(0.5, 0.2^2): precision 25. Two observations, noise 0.1^2 each: precision 100 each.
    posterior = Posterior([0.5], [0.04]).condition([[1.0], [1.0]], [0.6, 0.8], 0.01)
    assert posterior.mean[0] == pytest.approx((0.5 * 25 + (0.6 + 0.8) * 100) / 225)
    assert posterior.predictive([1.0])[1][0] == pytest.approx(1 / 225)
    assert posterior.n == 2


def test_a_noisier_observation_counts_less():
    posterior = Posterior([0.0], [100.0]).condition([[1.0], [1.0]], [0.0, 1.0], [0.01, 1.0])
    assert posterior.mean[0] == pytest.approx((1.0 / 1.0) / (1 / 100 + 100 + 1))
    assert posterior.mean[0] < 0.01


def test_with_no_observations_the_posterior_is_the_prior():
    prior_mean, prior_var = np.array([0.45, 0.0, 0.0]), np.array([0.25, 0.15, 0.05]) ** 2
    posterior = Posterior(prior_mean, prior_var)
    np.testing.assert_allclose(posterior.mean, prior_mean, atol=1e-12)
    np.testing.assert_allclose(posterior.predictive(np.eye(3))[1], prior_var, atol=1e-12)
    assert posterior.covariance([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0, abs=1e-15)
    assert posterior.n == 0


def test_conditioning_in_batches_equals_conditioning_at_once():
    prior_mean, prior_var, x, y, noise_var = random_problem(3, 40)
    at_once = Posterior(prior_mean, prior_var).condition(x, y, noise_var)
    batched = Posterior(prior_mean, prior_var)
    for i in range(0, 40, 7):
        batched = batched.condition(x[i : i + 7], y[i : i + 7], noise_var[i : i + 7])
    one_by_one = Posterior(prior_mean, prior_var)
    for i in range(40):
        one_by_one = one_by_one.condition(x[i], y[i], noise_var[i])
    for posterior in (batched, one_by_one):
        np.testing.assert_allclose(posterior.mean, at_once.mean, atol=1e-12)
        np.testing.assert_allclose(posterior.predictive(x)[1], at_once.predictive(x)[1], atol=1e-12)
        assert posterior.n == 40


def test_one_noise_variance_applies_to_every_observation():
    prior_mean, prior_var, x, y, _ = random_problem(4, 10)
    shared = Posterior(prior_mean, prior_var).condition(x, y, 1e-3)
    each = Posterior(prior_mean, prior_var).condition(x, y, np.full(10, 1e-3))
    np.testing.assert_array_equal(shared.mean, each.mean)


def test_conditioning_returns_a_new_posterior_and_leaves_the_old_one_unchanged():
    prior = Posterior(np.zeros(2), np.ones(2))
    posterior = prior.condition([[1.0, 1.0]], [1.0], 0.1)
    assert prior.n == 0
    np.testing.assert_array_equal(prior.mean, np.zeros(2))
    assert posterior.n == 1
    with pytest.raises(ValueError):
        posterior.mean[0] = 5.0  # read-only


def test_the_prior_is_copied_from_the_callers_arrays():
    prior_mean = np.zeros(2)
    posterior = Posterior(prior_mean, np.ones(2))
    prior_mean[0] = 9.0
    assert posterior.prior_mean[0] == 0.0
    assert prior_mean.flags.writeable


def test_samples_have_the_prior_moments():
    prior_mean, prior_var = np.array([0.45, 0.0]), np.array([0.04, 0.01])
    samples = Posterior(prior_mean, prior_var).sample(200_000, np.random.default_rng(0))
    assert samples.shape == (200_000, 2)
    np.testing.assert_allclose(samples.mean(axis=0), prior_mean, atol=5e-3)
    np.testing.assert_allclose(samples.var(axis=0), prior_var, rtol=0.02)


def test_samples_have_the_posterior_covariance():
    prior_mean, prior_var, x, y, noise_var = random_problem(5, 30, p=4)
    posterior = Posterior(prior_mean, prior_var).condition(x, y, noise_var)
    mean, cov = naive_reference(prior_mean, prior_var, x, y, noise_var)
    samples = posterior.sample(200_000, np.random.default_rng(1))
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=4 * np.sqrt(np.diag(cov) / 200_000).max())
    np.testing.assert_allclose(np.cov(samples.T), cov, atol=0.02 * np.abs(cov).max())


def test_samples_repeat_with_the_seed():
    posterior = Posterior(np.zeros(3), np.ones(3))
    np.testing.assert_array_equal(
        posterior.sample(5, np.random.default_rng(7)), posterior.sample(5, np.random.default_rng(7))
    )
    assert posterior.sample(0, np.random.default_rng(7)).shape == (0, 3)


def test_no_explicit_inverse_is_formed(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("np.linalg.inv was called")

    monkeypatch.setattr(np.linalg, "inv", boom)
    rng = np.random.default_rng(1)
    x = (rng.random((4, 10)) < 0.5).astype(float)
    posterior = Posterior(np.zeros(10), np.full(10, 0.1)).condition(x, rng.uniform(0, 1, 4), 1e-3)
    posterior.predictive(x)
    posterior.covariance(x[0], x[1])
    posterior.sample(10, rng)


@pytest.mark.parametrize(
    "prior_mean, prior_var, message",
    [
        ([0.0, 0.0], [1.0], "one nonzero length"),
        ([], [], "one nonzero length"),
        ([0.0], [0.0], "must be > 0"),
        ([np.nan], [1.0], "finite"),
    ],
)
def test_a_bad_prior_is_rejected(prior_mean, prior_var, message):
    with pytest.raises(ValueError, match=message):
        Posterior(prior_mean, prior_var)


@pytest.mark.parametrize(
    "x, y, noise_var, message",
    [
        ([[1.0, 0.0, 0.0]], [1.0], 0.1, r"must have 2 columns, got shape \(1, 3\)"),
        ([[1.0, 0.0], [1.0, 1.0]], [1.0], 0.1, "2 rows but y has 1"),
        ([[1.0, 0.0]], [1.0], 0.0, "must be > 0"),
        ([[1.0, 0.0]], [np.inf], 0.1, "finite"),
        ([[1.0, 0.0], [1.0, 1.0]], [1.0, 2.0], [0.1, 0.1, 0.1], "one per value"),
    ],
)
def test_bad_observations_are_rejected(x, y, noise_var, message):
    with pytest.raises(ValueError, match=message):
        Posterior(np.zeros(2), np.ones(2)).condition(x, y, noise_var)


def test_queries_reject_rows_of_the_wrong_length():
    posterior = Posterior(np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="2 columns"):
        posterior.predictive([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="2 finite values"):
        posterior.covariance([1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match=">= 0"):
        posterior.sample(-1, np.random.default_rng(0))
