import pytest

from strands_evals.optimization.bayesbench import FactorSpace


def make_space():
    return FactorSpace({"model": ["small", "large"], "tools": ["off", "on", "auto"]})


def test_factors_and_levels_keep_their_order():
    space = make_space()
    assert space.names == ("model", "tools")
    assert space.factors == {"model": ("small", "large"), "tools": ("off", "on", "auto")}


def test_a_key_is_the_levels_in_factor_order():
    assert make_space().key({"tools": "on", "model": "large"}) == ("large", "on")


def test_cells_are_the_product_with_the_last_factor_fastest():
    assert make_space().cells() == [
        ("small", "off"),
        ("small", "on"),
        ("small", "auto"),
        ("large", "off"),
        ("large", "on"),
        ("large", "auto"),
    ]


def test_every_cell_round_trips_through_its_levels():
    space = make_space()
    for cell in space.cells():
        assert space.key(dict(zip(space.names, cell, strict=True))) == cell


def test_a_key_needs_every_factor():
    with pytest.raises(KeyError, match="missing factors"):
        make_space().key({"model": "small"})


@pytest.mark.parametrize(
    "levels, message",
    [
        ({"model": "medium", "tools": "on"}, "unknown level 'medium'"),
        ({"model": "small", "tools": "on", "temperature": "0.7"}, "unknown factors"),
    ],
)
def test_a_key_rejects_an_unknown_level_or_factor(levels, message):
    with pytest.raises(ValueError, match=message):
        make_space().key(levels)


@pytest.mark.parametrize(
    "factors, message",
    [
        ({}, "at least one factor"),
        ({"model": ["small"]}, "at least 2 levels"),
        ({"model": ["small", "small"]}, "repeats a level"),
        ({"model": "ab"}, "sequence of levels"),
    ],
)
def test_the_space_rejects_bad_factors(factors, message):
    with pytest.raises(ValueError, match=message):
        FactorSpace(factors)
