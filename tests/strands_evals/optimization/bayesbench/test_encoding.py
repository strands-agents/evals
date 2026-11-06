import numpy as np
import pytest

from strands_evals.optimization.bayesbench import Encoder, FactorSpace


@pytest.fixture
def space545():
    return FactorSpace(
        {
            "model": ["model_a", "model_b", "model_c", "model_d", "model_e"],
            "harness": ["harness_a", "harness_b", "harness_c", "harness_d"],
            "benchmark": ["bench_a", "bench_b", "bench_c", "bench_d", "bench_e"],
        }
    )


def test_a_5x4x5_grid_has_52_columns(space545):
    assert Encoder(space545).p == 52
    assert len(space545.cells()) == 100


def test_the_column_order_is_intercept_then_mains_then_pairs(space545):
    encoder = Encoder(space545)
    assert encoder.column_names[0] == "intercept"
    assert encoder.column_names[1] == "model=model_b"  # the reference, model_a, has no column
    assert encoder.column_names[5] == "harness=harness_b"  # after model's 4 mains
    assert encoder.column_names[8] == "benchmark=bench_b"
    assert encoder.column_names[12] == "model=model_b:harness=harness_b"
    assert encoder.column_names[-1] == "harness=harness_d:benchmark=bench_e"
    assert len(set(encoder.column_names)) == encoder.p


def test_the_column_maps_cover_every_column_once(space545):
    encoder = Encoder(space545)
    assert {factor: len(cols) for factor, cols in encoder.main_columns.items()} == {
        "model": 4,
        "harness": 3,
        "benchmark": 4,
    }
    assert {pair: len(cols) for pair, cols in encoder.interaction_columns.items()} == {
        ("model", "harness"): 12,
        ("model", "benchmark"): 16,
        ("harness", "benchmark"): 12,
    }
    columns = [0, *(c for cols in encoder.main_columns.values() for c in cols)]
    columns += [c for cols in encoder.interaction_columns.values() for c in cols]
    assert sorted(columns) == list(range(encoder.p))


def test_the_reference_cell_is_the_intercept_alone(space545):
    x = Encoder(space545).row(("model_a", "harness_a", "bench_a"))
    assert x[0] == 1.0
    assert x[1:].sum() == 0.0


def test_a_cell_with_no_reference_level_switches_on_its_mains_and_pairs(space545):
    encoder = Encoder(space545)
    x = encoder.row(("model_b", "harness_b", "bench_b"))
    assert [name for name, v in zip(encoder.column_names, x, strict=True) if v == 1.0] == [
        "intercept",
        "model=model_b",
        "harness=harness_b",
        "benchmark=bench_b",
        "model=model_b:harness=harness_b",
        "model=model_b:benchmark=bench_b",
        "harness=harness_b:benchmark=bench_b",
    ]
    assert x.sum() == 7.0


def test_the_full_grid_identifies_every_column(space545):
    encoder = Encoder(space545)
    matrix = encoder.matrix(space545.cells())
    assert matrix.shape == (100, 52)
    assert np.linalg.matrix_rank(matrix) == 52


def test_without_interactions_only_the_mains_are_encoded(space545):
    encoder = Encoder(space545, interactions=False)
    assert encoder.p == 12  # 1 + 4 + 3 + 4
    assert encoder.interaction_columns == {}
    assert encoder.row(("model_b", "harness_b", "bench_b")).sum() == 4.0


def test_one_factor_has_no_interaction_columns():
    encoder = Encoder(FactorSpace({"model": ["small", "large"]}))
    assert encoder.p == 2
    assert encoder.interaction_columns == {}


def test_no_keys_give_an_empty_matrix(space545):
    assert Encoder(space545).matrix([]).shape == (0, 52)


@pytest.mark.parametrize(
    "key, message",
    [
        (("model_a", "harness_a"), "has 3 levels, got 2"),
        (("model_a", "harness_a", "bench_z"), "unknown level 'bench_z'"),
    ],
)
def test_a_row_rejects_a_bad_key(space545, key, message):
    with pytest.raises(ValueError, match=message):
        Encoder(space545).row(key)
