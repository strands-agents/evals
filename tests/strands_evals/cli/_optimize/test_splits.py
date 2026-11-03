import pytest

from strands_evals.cli._optimize.splits import assign_split, parse_shares

NAMES = [f"case-{index:02d}" for index in range(10)]


@pytest.mark.parametrize(
    ("text", "shares"),
    [("0.4,0.3,0.3", (0.4, 0.3, 0.3)), ("0.5,0.5", (0.5, 0.5, 0.0)), ("0.6,0.4,0", (0.6, 0.4, 0.0))],
)
def test_parse_shares_accepts_two_or_three_shares(text, shares):
    assert parse_shares(text) == shares


@pytest.mark.parametrize(
    ("text", "message"),
    [("a,b", "numbers"), ("1", "two or three"), ("0.5,0.6", "sum to 1"), ("1.2,-0.2", ">= 0")],
)
def test_parse_shares_rejects_bad_shares(text, message):
    with pytest.raises(ValueError, match=message):
        parse_shares(text)


def test_assign_split_counts_by_largest_remainder_and_covers_every_case():
    split = assign_split(NAMES, (0.4, 0.3, 0.3), 0)

    assert (len(split.train), len(split.validation), len(split.test)) == (4, 3, 3)
    assert sorted([*split.train, *split.validation, *split.test]) == NAMES


def test_assign_split_is_deterministic_per_seed_and_ignores_input_order():
    first = assign_split(NAMES, (0.4, 0.3, 0.3), 7)

    assert assign_split(list(reversed(NAMES)), (0.4, 0.3, 0.3), 7) == first
    assert assign_split(NAMES, (0.4, 0.3, 0.3), 8) != first


def test_assign_split_allows_an_empty_test_split():
    split = assign_split(NAMES[:4], (0.5, 0.5, 0.0), 0)

    assert split.test == ()


def test_assign_split_gives_the_largest_remainders_to_train_and_validation_first():
    split = assign_split(NAMES[:2], (0.4, 0.3, 0.3), 0)

    assert (len(split.train), len(split.validation), len(split.test)) == (1, 1, 0)


@pytest.mark.parametrize(("names", "shares"), [(NAMES[:1], (0.4, 0.3, 0.3)), (NAMES[:4], (0.9, 0.1, 0.0))])
def test_assign_split_needs_one_train_and_one_validation_case(names, shares):
    with pytest.raises(ValueError, match="at least 1"):
        assign_split(names, shares, 0)
