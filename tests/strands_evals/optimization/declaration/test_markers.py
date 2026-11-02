import pytest
from strands import tool

from strands_evals.optimization import Float, Int, OneOf, TextValue, choice, component, tool_descriptions
from strands_evals.optimization.declaration.markers import _default_size_cap, _ValueMarker


class Manager:
    def __init__(self, window_size: int = 40, ratio: float = 0.5, factory: str = "x"):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.ratio = ratio
        self.factory = factory


class OtherManager:
    def __init__(self, depth: int = 2):
        self.depth = depth


def make_manager(window_size: int = 8) -> Manager:
    return Manager(window_size=window_size)


# --- Int / Float ---


def test_int_keeps_bounds_and_default():
    marker = Int(1, 10, default=3, log=True)
    assert (marker.min_value, marker.max_value, marker.default, marker.log) == (1, 10, 3, True)


def test_int_default_is_optional():
    assert Int(1, 10).default is None


@pytest.mark.parametrize(
    "make",
    [
        lambda: Int(1, 10, default=11),
        lambda: Int(1, 10, default=0),
        lambda: Int(10, 1),
        lambda: Int(0, 10, log=True),
        lambda: Int(1.5, 10),  # type: ignore[arg-type]
        lambda: Int(1, 10, default=True),
        lambda: Int(1, 10, default=2.0),  # type: ignore[arg-type]
    ],
)
def test_int_rejects_bad_declarations(make):
    with pytest.raises(TypeError):
        make()


def test_float_stores_floats_so_an_integer_default_hashes_like_a_proposed_float():
    marker = Float(0, 2, default=1)
    assert marker.default == 1.0 and type(marker.default) is float
    assert type(marker.min_value) is float and type(marker.max_value) is float


@pytest.mark.parametrize(
    "make",
    [
        lambda: Float(0.0, 1.0, default=1.5),
        lambda: Float(1.0, 0.0),
        lambda: Float(-1.0, 1.0, log=True),
        lambda: Float(0.0, 1.0, default=False),
        lambda: Float("0", 1.0),  # type: ignore[arg-type]
        lambda: Float(float("nan"), 1.0),  # NaN compares False, so min_value > max_value alone wouldn't catch it
        lambda: Float(0.0, float("inf")),
        lambda: Float(0.0, 1.0, default=float("nan")),
    ],
)
def test_float_rejects_bad_declarations(make):
    with pytest.raises(TypeError):
        make()


# --- OneOf ---


def test_one_of_default_is_unset_until_filled():
    marker = OneOf("low", "high")
    assert marker.options == ("low", "high")
    assert marker._with_default("high").default == "high"


def test_one_of_accepts_none_as_an_option_and_default():
    assert OneOf(None, "x", default=None).default is None


@pytest.mark.parametrize(
    "make",
    [
        lambda: OneOf(),
        lambda: OneOf("a", "a"),
        lambda: OneOf("a", ["b"]),  # type: ignore[arg-type]
        lambda: OneOf("a", "b", default="c"),
        lambda: OneOf(1, 2, default=True),  # True == 1, but it is not the option 1
        lambda: OneOf(0.5, float("nan")),  # NaN never equals itself, so it could never be chosen
        lambda: OneOf(0.5, float("inf")),
    ],
)
def test_one_of_rejects_bad_declarations(make):
    with pytest.raises(TypeError):
        make()


def test_one_of_tells_true_from_one():
    marker = OneOf(1, True)
    assert marker.options == (1, True)


# --- TextValue ---


def test_text_value_keeps_its_settings():
    marker = TextValue(default="hi", size_cap=10, edit_instructions="be brief")
    assert (marker.default, marker.size_cap, marker.edit_instructions) == ("hi", 10, "be brief")


@pytest.mark.parametrize(
    "make",
    [
        lambda: TextValue(default=3),  # type: ignore[arg-type]
        lambda: TextValue(default="x", size_cap=0),
        lambda: TextValue(default="x", size_cap=True),
    ],
)
def test_text_value_rejects_bad_declarations(make):
    with pytest.raises(TypeError):
        make()


def test_default_size_cap_is_at_least_2000_and_three_times_the_baseline():
    assert _default_size_cap("x" * 10) == 2000
    assert _default_size_cap("x" * 1000) == 3000


# --- component ---


def test_component_splits_tunable_and_fixed_kwargs():
    declared = component(Manager, window_size=Int(1, 100, default=20), factory="kept")
    assert declared.label == "Manager"
    assert set(declared.tunable) == {"window_size"}
    assert declared.fixed == {"factory": "kept"}


def test_component_takes_a_missing_default_from_the_constructor():
    declared = component(Manager, window_size=Int(1, 100), ratio=Float(0.0, 1.0))
    assert declared.tunable["window_size"].default == 40
    assert declared.tunable["ratio"].default == 0.5


def test_component_accepts_a_plain_function():
    declared = component(make_manager, window_size=Int(1, 100))
    assert declared.label == "make_manager"
    assert declared.tunable["window_size"].default == 8


def test_component_builds_fresh_instances():
    declared = component(Manager, window_size=Int(1, 100, default=20))
    first, second = declared.build({"window_size": 5}), declared.build({"window_size": 5})
    assert first is not second
    assert first.window_size == 5


def test_component_whose_constructor_rejects_its_defaults_fails_at_declaration():
    with pytest.raises(TypeError, match="rejects its own defaults") as raised:
        component(Manager, window_size=Int(0, 100, default=0))
    assert isinstance(raised.value.__cause__, ValueError)


def test_component_with_an_unknown_kwarg_fails_at_declaration():
    with pytest.raises(TypeError, match="rejects its own defaults"):
        component(Manager, no_such_arg=1)


def test_component_without_a_default_anywhere_fails():
    with pytest.raises(TypeError, match="has no default"):
        component(Manager, missing=Int(1, 5))


def test_component_constructor_default_outside_the_bounds_fails():
    with pytest.raises(TypeError, match="doesn't fit"):
        component(Manager, window_size=Int(1, 10))


@tool
def lookup(query: str) -> str:
    """Look up a record."""
    return query


def test_component_takes_a_one_of_kwarg_as_tunable():
    declared = component(Manager, factory=OneOf("x", "y"), ratio=0.2)
    assert declared.tunable["factory"].options == ("x", "y")
    assert declared.fixed == {"ratio": 0.2}
    assert declared.build({"factory": "y"}).factory == "y"


def test_component_one_of_without_default_takes_the_constructor_default():
    assert component(Manager, factory=OneOf("y", "x")).tunable["factory"].default == "x"


class Pinned:
    def __init__(self, pin_first):  # no default: a OneOf falls back to its first option
        self.pin_first = pin_first


def test_component_one_of_without_any_default_takes_its_first_option():
    declared = component(Pinned, pin_first=OneOf(0, 1))
    assert declared.tunable["pin_first"].default == 0


class Optional:
    def __init__(self, temperature: float | None = None, pin_first: int | None = None):
        self.temperature = temperature
        self.pin_first = pin_first


@pytest.mark.parametrize("marker", [Int(0, 5), Float(0.0, 1.0)])
def test_component_number_whose_constructor_default_is_none_fails_at_component(marker):
    with pytest.raises(TypeError, match="default for temperature= is None; pass default="):
        component(Optional, temperature=marker)


def test_component_number_with_its_own_default_ignores_a_none_constructor_default():
    assert component(Optional, temperature=Float(0.0, 1.0, default=0.3)).tunable["temperature"].default == 0.3


def test_component_one_of_may_take_a_none_constructor_default_as_an_option():
    assert component(Optional, pin_first=OneOf(None, 1)).tunable["pin_first"].default is None
    with pytest.raises(TypeError, match="doesn't fit"):
        component(Optional, pin_first=OneOf(0, 1))  # None is not an option: pass default=


def test_component_one_of_on_a_real_conversation_manager():
    # strands' SlidingWindowConversationManager declares pin_first: int | None = None.
    from strands.agent.conversation_manager import SlidingWindowConversationManager

    declared = component(SlidingWindowConversationManager, window_size=Int(4, 40), pin_first=OneOf(None, 1))
    assert declared.tunable["window_size"].default == 40
    assert declared.tunable["pin_first"].default is None
    pinned = component(SlidingWindowConversationManager, pin_first=OneOf(0, 1, default=0))
    assert pinned.tunable["pin_first"].default == 0


def test_component_one_of_whose_constructor_default_is_not_an_option_fails():
    with pytest.raises(TypeError, match="doesn't fit"):
        component(Manager, factory=OneOf("y", "z"))


@pytest.mark.parametrize(
    "marker",
    [
        TextValue(default="x"),
        tool_descriptions(lookup),
        choice(component(OtherManager)),
        component(OtherManager),
    ],
)
def test_component_rejects_markers_other_than_int_float_and_one_of(marker):
    with pytest.raises(TypeError, match="only Int, Float, and OneOf"):
        component(Manager, factory=marker)


def test_component_rejects_a_non_callable():
    with pytest.raises(TypeError):
        component("Manager")  # type: ignore[arg-type]


# --- choice ---


def test_choice_labels_default_to_class_names_in_order():
    declared = choice(component(Manager), component(OtherManager))
    assert list(declared.alternatives) == ["Manager", "OtherManager"]
    assert [c.label for c in declared.alternatives.values()] == ["Manager", "OtherManager"]


def test_choice_keyword_labels_keep_their_order_and_reach_the_components():
    declared = choice(summary=component(OtherManager), window=component(Manager))
    assert list(declared.alternatives) == ["summary", "window"]  # the first keyword is the baseline
    assert [c.label for c in declared.alternatives.values()] == ["summary", "window"]


def test_choice_with_the_same_class_twice_needs_keyword_labels():
    with pytest.raises(TypeError, match="keyword"):
        choice(component(Manager, ratio=0.1), component(Manager, ratio=0.9))
    declared = choice(low=component(Manager, ratio=0.1), high=component(Manager, ratio=0.9))
    assert list(declared.alternatives) == ["low", "high"]


def test_one_component_can_sit_in_two_choices_under_different_labels():
    shared = component(Manager)
    first = choice(window=shared, other=component(OtherManager))
    second = choice(sliding=shared, other=component(OtherManager))
    assert first.alternatives["window"].label == "window"
    assert second.alternatives["sliding"].label == "sliding"
    assert shared.label == "Manager"


def test_choice_rejects_mixed_positional_and_keyword_alternatives():
    with pytest.raises(TypeError, match="not both"):
        choice(component(Manager), other=component(OtherManager))


@pytest.mark.parametrize("label", ["b.c", "b/c", ""])
def test_choice_rejects_bad_labels(label):
    with pytest.raises(TypeError):
        choice(**{"a": component(Manager), label: component(OtherManager)})


def test_choice_rejects_non_components():
    with pytest.raises(TypeError):
        choice()
    with pytest.raises(TypeError):
        choice(Manager)  # type: ignore[arg-type]


def test_choice_takes_fixed_values_by_keyword_as_is():
    marker = {"mode": "dense"}
    declared = choice(on=component(Manager), off=None, preset=marker)
    assert list(declared.alternatives) == ["on", "off", "preset"]
    assert declared.alternatives["on"].label == "on"
    assert declared.alternatives["off"] is None
    assert declared.alternatives["preset"] is marker


def test_choice_rejects_a_positional_fixed_value():
    with pytest.raises(TypeError, match="by keyword"):
        choice(component(Manager), None)  # type: ignore[arg-type]


@pytest.mark.parametrize("marker", [Int(1, 5, default=2), OneOf("a", "b"), choice(component(OtherManager))])
def test_choice_rejects_a_marker_as_a_fixed_value(marker):
    with pytest.raises(TypeError, match="component\\(...\\) or a fixed value"):
        choice(on=component(Manager), off=marker)


# --- value markers ---


@pytest.mark.parametrize(
    ("marker", "default"), [(Int(1, 5), 2), (Float(0.0, 1.0), 0.5), (OneOf("a", "b"), "b"), (TextValue(), "x")]
)
def test_value_markers_fill_a_default_into_a_copy(marker, default):
    assert isinstance(marker, _ValueMarker)
    filled = marker._with_default(default)
    assert type(filled) is type(marker) and filled.default == default
    assert marker.default is not default  # the declared marker is left unfilled
