from statistics import NormalDist

import numpy as np
import pytest

from strands_evals.optimization.bayesbench import (
    Contrast,
    Encoder,
    FactorSpace,
    Mixture,
    Posterior,
    linear_contrast,
)


def grid_encoder():
    return Encoder(FactorSpace({"feature": ["off", "on"], "model": ["small", "large"], "benchmark": ["short", "long"]}))


def effect_row(encoder, level, reference):
    """A feature level's effect over every background: the mean row with it minus the mean row with the reference."""
    cells = encoder.space.cells()
    return encoder.matrix([c for c in cells if c[0] == level]).mean(axis=0) - encoder.matrix(
        [c for c in cells if c[0] == reference]
    ).mean(axis=0)


def interaction_row(encoder, model, reference_model, benchmark="short"):
    """How much more turning the feature on does under `model` than under `reference_model`."""
    row = encoder.row
    return (row(("on", model, benchmark)) - row(("off", model, benchmark))) - (
        row(("on", reference_model, benchmark)) - row(("off", reference_model, benchmark))
    )


def known_posterior(encoder, settings, sd=1e-3):
    """A posterior pinned near the given effects."""
    theta = np.zeros(encoder.p)
    for name, value in settings.items():
        theta[encoder.column_names.index(name)] = value
    return Posterior(theta, np.full(encoder.p, sd**2))


def test_a_big_main_effect_is_not_evidence_of_an_interaction():
    encoder = grid_encoder()
    posterior = known_posterior(encoder, {"feature=on": 0.2})
    assert linear_contrast(posterior, effect_row(encoder, "on", "off")).mean == pytest.approx(0.2)
    interaction = linear_contrast(posterior, interaction_row(encoder, "large", "small"))
    assert interaction.mean == pytest.approx(0.0)
    assert interaction.ci90[0] < 0 < interaction.ci90[1]


def test_another_factors_effect_does_not_leak_into_a_level_contrast():
    encoder = grid_encoder()
    posterior = known_posterior(encoder, {"benchmark=long": 0.3})
    assert linear_contrast(posterior, effect_row(encoder, "on", "off")).mean == pytest.approx(0.0, abs=1e-12)


def test_a_difference_of_effects_cancels_their_shared_uncertainty():
    encoder = grid_encoder()
    theta, var = np.zeros(encoder.p), np.full(encoder.p, 0.2**2)
    column = encoder.column_names.index("feature=on:model=large")
    theta[column], var[column] = 0.08, 0.03**2
    contrast = linear_contrast(Posterior(theta, var), interaction_row(encoder, "large", "small"))
    assert contrast.mean == pytest.approx(0.08)
    assert contrast.sd == pytest.approx(0.03)  # the main effects' large uncertainty cancels


def test_the_interval_and_probability_come_from_the_normal():
    posterior = Posterior([0.1], [0.05**2])
    contrast = linear_contrast(posterior, [1.0])
    assert (contrast.mean, contrast.sd) == (pytest.approx(0.1), pytest.approx(0.05))
    assert contrast.p_positive == pytest.approx(1 - NormalDist(0.1, 0.05).cdf(0))
    low, high = contrast.ci90
    assert low == pytest.approx(NormalDist(0.1, 0.05).inv_cdf(0.05))
    assert high == pytest.approx(NormalDist(0.1, 0.05).inv_cdf(0.95))


def test_a_contrast_with_no_uncertainty_is_positive_only_above_0():
    contrast = linear_contrast(Posterior([0.3, 0.0], [1.0, 1.0]), [0.0, 0.0])  # a row of zeros: exactly 0
    assert contrast == Contrast(mean=0.0, sd=0.0, p_positive=0.0)
    assert contrast.ci90 == (0.0, 0.0)


@pytest.mark.parametrize("mean, p_positive", [(0.3, 1.0), (-0.3, 0.0)])
def test_a_nearly_certain_contrast_has_the_sign_of_its_mean(mean, p_positive):
    assert linear_contrast(Posterior([mean], [1e-30]), [1.0]).p_positive == p_positive


def test_a_mixtures_contrast_is_its_mean_and_sd():
    narrow = Posterior([0.1, 0.0], [1e-4, 1e-4])
    wide = Posterior([0.3, 0.0], [1e-2, 1e-4])
    mixture = Mixture([(0.5, narrow), (0.5, wide)])
    contrast = linear_contrast(mixture, [1.0, 0.0])
    mean, var = mixture.predictive([1.0, 0.0])
    assert contrast.mean == pytest.approx(float(mean[0]))
    assert contrast.sd == pytest.approx(float(np.sqrt(var[0])))
    assert contrast.mean == pytest.approx(0.2)


@pytest.mark.parametrize("row", [[1.0], [1.0, 0.0, 0.0], [[1.0, 0.0]], [np.nan, 0.0]])
def test_a_bad_row_is_rejected(row):
    with pytest.raises(ValueError, match="2 finite values"):
        linear_contrast(Posterior([0.0, 0.0], [1.0, 1.0]), row)
