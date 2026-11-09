import math

import numpy as np
import pytest

from strands_evals.optimization.bayesbench import HeldOutDiagnostics


def record_z(diagnostics, zs):
    """Record observations whose z-scores are `zs` (prediction 0, total variance 1)."""
    for z in zs:
        diagnostics.record(y=z, predicted_mean=0.0, predicted_var=0.5, noise_var=0.5)


def test_the_z_score_uses_the_predictions_and_the_observations_variance():
    diagnostics = HeldOutDiagnostics()
    assert diagnostics.record(y=0.7, predicted_mean=0.5, predicted_var=0.03, noise_var=0.01) == pytest.approx(
        0.2 / math.sqrt(0.04)
    )
    assert diagnostics.z_scores == (pytest.approx(1.0),)


def test_there_is_no_flag_before_the_first_window_completes():
    diagnostics = HeldOutDiagnostics(window=15)
    for _ in range(14):
        diagnostics.record(y=0.9, predicted_mean=0.1, predicted_var=1e-4, noise_var=1e-4)
    assert diagnostics.flagged is False  # wildly wrong, but the window isn't complete
    diagnostics.record(y=0.9, predicted_mean=0.1, predicted_var=1e-4, noise_var=1e-4)
    assert diagnostics.flagged is True


def test_the_flag_is_sticky():
    diagnostics = HeldOutDiagnostics(window=15)
    for _ in range(15):  # one terrible window
        diagnostics.record(y=0.9, predicted_mean=0.1, predicted_var=1e-4, noise_var=1e-4)
    assert diagnostics.flagged is True
    rng = np.random.default_rng(0)
    for _ in range(30):  # then perfectly calibrated observations
        z = rng.standard_normal()
        diagnostics.record(y=0.5 + 0.05 * z, predicted_mean=0.5, predicted_var=0.05**2 / 2, noise_var=0.05**2 / 2)
    assert diagnostics.flagged is True  # a misspecification doesn't un-happen


@pytest.mark.parametrize("z, flagged", [(0.72, False), (0.73, True)])
def test_a_window_biased_to_one_side_flags(z, flagged):
    # |mean z| > norm.ppf(0.9975) / sqrt(15) = 0.7248, while sum z^2 = 15 z^2 stays far below 32.80.
    diagnostics = HeldOutDiagnostics(boundary=None)
    record_z(diagnostics, [z] * 15)
    assert diagnostics.flagged is flagged


@pytest.mark.parametrize("size, flagged", [(1.52, False), (1.54, True)])
def test_an_overdispersed_window_flags(size, flagged):
    # sum z^2 = 14 size^2 > chi2.ppf(0.995, 15) = 32.80 flags, though mean z is 0.
    diagnostics = HeldOutDiagnostics(boundary=None)
    record_z(diagnostics, [size, -size] * 7 + [0.0])
    assert diagnostics.flagged is flagged


def test_windows_are_disjoint():
    # 15 biased z-scores straddling two windows: a rolling window over them would flag; neither disjoint one does.
    diagnostics = HeldOutDiagnostics(boundary=None)
    record_z(diagnostics, [0.0] * 8 + [0.9] * 15 + [0.0] * 7)
    assert diagnostics.n_holdout == 30
    assert diagnostics.flagged is False


def test_well_calibrated_z_scores_rarely_flag():
    """60 observations (4 windows) of iid N(0, 1) z: about 4 x 2 x 0.005 = 4% flag; allow up to 10%."""
    flags = 0
    for rep in range(500):
        rng = np.random.default_rng(rep)
        diagnostics = HeldOutDiagnostics()
        for _ in range(60):
            z = rng.standard_normal()
            diagnostics.record(y=0.5 + 0.05 * z, predicted_mean=0.5, predicted_var=0.05**2 / 2, noise_var=0.05**2 / 2)
        flags += diagnostics.flagged
    assert flags / 500 <= 0.10


def test_observations_outside_the_boundary_are_noted():
    diagnostics = HeldOutDiagnostics()
    diagnostics.record(y=0.98, predicted_mean=0.9, predicted_var=0.01, noise_var=1e-4, label=("m", "h", "b"))
    diagnostics.record(y=0.95, predicted_mean=0.9, predicted_var=0.01, noise_var=1e-4, label="inside")
    snapshot = diagnostics.snapshot()
    assert snapshot.near_boundary == (("m", "h", "b"),)
    assert any("outside [0.05, 0.95]" in message for message in snapshot.messages)


def test_no_boundary_notes_nothing():
    diagnostics = HeldOutDiagnostics(boundary=None)
    diagnostics.record(y=5.0, predicted_mean=4.0, predicted_var=1.0, noise_var=1.0, label="far")
    assert diagnostics.snapshot().near_boundary == ()


def test_the_snapshot_summarizes_every_z_score():
    diagnostics = HeldOutDiagnostics(window=3, boundary=None)
    record_z(diagnostics, [1.0, -2.0, 0.5, 3.0])
    snapshot = diagnostics.snapshot()
    assert snapshot.n_holdout == 4
    assert snapshot.z_mean == pytest.approx(2.5 / 4)
    assert snapshot.z_sq_sum == pytest.approx(1 + 4 + 0.25 + 9)
    assert snapshot.flagged is False
    assert snapshot.messages == ()


def test_the_messages_say_whether_a_window_is_complete_and_whether_it_flagged():
    diagnostics = HeldOutDiagnostics(boundary=None)
    empty = diagnostics.snapshot()
    assert (empty.n_holdout, empty.z_mean, empty.z_sq_sum) == (0, 0.0, 0.0)
    assert empty.messages == ("first window not complete (0/15); no flagging yet",)
    record_z(diagnostics, [3.0] * 15)
    (message,) = diagnostics.snapshot().messages
    assert message.startswith("a held-out z window exceeded its calibrated bound")


@pytest.mark.parametrize(
    "values, message",
    [
        ((np.nan, 0.0, 1.0, 1.0), "finite"),
        ((0.0, 0.0, -1.0, 1.0), ">= 0 and not both 0"),
        ((0.0, 0.0, 0.0, 0.0), ">= 0 and not both 0"),
    ],
)
def test_a_bad_observation_is_rejected(values, message):
    with pytest.raises(ValueError, match=message):
        HeldOutDiagnostics().record(*values)


@pytest.mark.parametrize("kwargs, message", [({"window": 0}, "window"), ({"level": 1.0}, "level")])
def test_bad_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HeldOutDiagnostics(**kwargs)
