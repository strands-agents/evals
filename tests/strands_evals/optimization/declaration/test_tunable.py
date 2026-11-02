import functools
import inspect

import pytest
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager, SummarizingConversationManager
from strands.hooks import BeforeModelCallEvent
from strands.models.model import Model

from strands_evals.optimization import Float, Int, OneOf, TextValue, TunableFactory, choice, component, tunable
from strands_evals.optimization.declaration.tunable import _FactoryArgumentError


class StubModel(Model):
    """Never called: agents here are only constructed."""

    def update_config(self, **model_config):
        pass

    def get_config(self):
        return {}

    def structured_output(self, *args, **kwargs):
        raise NotImplementedError

    def stream(self, *args, **kwargs):
        raise NotImplementedError


class Window:
    def __init__(self, size: int = 40):
        self.size = size


class Summary:
    def __init__(self, ratio: float = 0.3):
        self.ratio = ratio


MANAGER = choice(component(Window, size=Int(1, 100)), component(Summary, ratio=Float(0.1, 0.9)))


@tunable(manager=MANAGER, prompt=TextValue(), temperature=Float(0.0, 1.0), effort=OneOf("low", "high"))
def make_agent(manager=None, prompt="You help.", temperature=0.7, effort="high", sandbox=None):
    """Build the test agent."""
    return {"manager": manager, "prompt": prompt, "temperature": temperature, "effort": effort, "sandbox": sandbox}


# --- the two mounts ---


def test_both_mounts_return_the_same_factory():
    def build(manager=None, prompt="You help."):
        return manager, prompt

    decorated = tunable(manager=MANAGER, prompt=TextValue())(build)
    injected = tunable(build, inject={"manager": MANAGER, "prompt": TextValue()})
    assert isinstance(decorated, TunableFactory) and isinstance(injected, TunableFactory)
    assert decorated.space.params() == injected.space.params()


def test_the_factory_looks_like_the_wrapped_function():
    assert make_agent.__name__ == "make_agent"
    assert make_agent.__doc__ == "Build the test agent."
    assert "sandbox" in inspect.signature(make_agent).parameters


# --- defaults and the plain call ---


def test_markers_without_default_take_the_parameters_default():
    space = make_agent.space
    assert space.param("prompt").default == "You help."
    assert space.param("temperature").default == 0.7
    assert space.param("effort").default == "high"


def test_the_plain_call_builds_the_baseline_without_an_optimizer():
    built = make_agent()
    assert isinstance(built["manager"], Window) and built["manager"].size == 40
    assert (built["prompt"], built["temperature"], built["effort"]) == ("You help.", 0.7, "high")


def test_the_plain_call_lets_explicit_arguments_win():
    assert make_agent(prompt="Override.")["prompt"] == "Override."


def test_the_plain_call_lets_positional_arguments_win():
    built = make_agent(Summary(), "Override.")
    assert isinstance(built["manager"], Summary) and built["prompt"] == "Override."


def test_the_plain_call_lets_a_name_caught_by_kwargs_win():
    factory = tunable(lambda **kwargs: kwargs, inject={"effort": OneOf("low", "high")})
    assert factory(effort="high") == {"effort": "high"}


def test_one_of_without_any_default_takes_the_first_option():
    factory = tunable(lambda effort: effort, inject={"effort": OneOf("low", "high")})
    assert factory() == "low"


def test_an_explicit_default_wins_over_the_parameters_default():
    factory = tunable(lambda temperature=0.7: temperature, inject={"temperature": Float(0.0, 1.0, default=0.2)})
    assert factory() == 0.2


# --- declaration errors (zero rollouts) ---


def test_a_marker_naming_no_parameter_fails():
    with pytest.raises(TypeError, match="no parameter 'missing'"):
        tunable(missing=Int(1, 5, default=1))(lambda prompt: prompt)


def test_a_function_taking_kwargs_accepts_any_name():
    factory = tunable(lambda **kwargs: kwargs, inject={"extra": Int(1, 5, default=2)})
    assert factory() == {"extra": 2}


def test_stacking_tunable_fails_instead_of_dropping_markers():
    with pytest.raises(TypeError, match="already tunable"):

        @tunable(a=Int(1, 9))
        @tunable(b=Int(1, 9))
        def build(a=1, b=2):
            return a, b


def test_a_positional_only_parameter_fails():
    def build(prompt, /):
        return prompt

    with pytest.raises(TypeError, match="not by keyword"):
        tunable(build, inject={"prompt": TextValue(default="x")})


@pytest.mark.parametrize(
    "make",
    [
        lambda: tunable(lambda prompt: prompt, inject={"prompt": TextValue()}),  # no default anywhere
        lambda: tunable(lambda prompt=None: prompt, inject={"prompt": TextValue()}),  # parameter default None
        lambda: tunable(lambda size=500: size, inject={"size": Int(1, 100)}),  # parameter default out of range
        lambda: tunable(lambda effort="mid": effort, inject={"effort": OneOf("low", "high")}),
        lambda: tunable(lambda prompt="x": prompt, inject={"prompt": "not a marker"}),
        lambda: tunable(lambda prompt="x": prompt),  # no inject=
        lambda: tunable(lambda prompt="x": prompt, inject={"prompt": TextValue()}, other=1),
        lambda: tunable(inject={"prompt": TextValue(default="x")}),  # inject without the function
        lambda: tunable()(lambda: None),  # nothing declared
    ],
)
def test_bad_declarations_fail_at_declaration(make):
    with pytest.raises(TypeError):
        make()


# --- bind and validation ---


def test_bind_fills_missing_keys_with_defaults():
    built = make_agent.bind({"temperature": 0.1})()
    assert built["temperature"] == 0.1 and built["prompt"] == "You help."


def test_bind_builds_fresh_components_on_every_call():
    bound = make_agent.bind({"manager": "Summary", "manager.Summary.ratio": 0.5})
    first, second = bound(), bound()
    assert isinstance(first["manager"], Summary) and first["manager"].ratio == 0.5
    assert first["manager"] is not second["manager"]


def test_a_bad_proposal_dies_at_validation_before_any_agent_is_built():
    calls = []

    def build(temperature=0.5):
        calls.append(temperature)
        return temperature

    factory = tunable(build, inject={"temperature": Float(0.0, 1.0)})
    with pytest.raises(ValueError, match="outside"):
        factory.bind({"temperature": 3.0})
    with pytest.raises(ValueError):
        factory.validate({"temperature": 3.0})
    assert calls == []


def test_bind_rejects_a_key_inactive_under_the_chosen_alternative():
    with pytest.raises(ValueError, match="inactive"):
        make_agent.bind({"manager.Summary.ratio": 0.5})


# --- arguments to the bound factory: call kwargs and resources ---


def test_a_call_kwarg_for_a_non_tunable_parameter_passes_through():
    assert make_agent.bind({})(sandbox="box")["sandbox"] == "box"


def test_a_call_kwarg_the_function_does_not_accept_raises_instead_of_being_dropped():
    with pytest.raises(_FactoryArgumentError, match=r"no arguments named \['sandbx'\]"):
        make_agent.bind({})(sandbx="box")


def test_a_function_taking_kwargs_accepts_any_call_kwarg():
    factory = tunable(lambda effort="low", **kwargs: kwargs, inject={"effort": OneOf("low", "high")})
    assert factory.bind({})(context_manager=False) == {"context_manager": False}


def test_a_call_kwarg_named_like_a_marker_is_a_setup_error():
    with pytest.raises(_FactoryArgumentError, match="prompt"):
        make_agent.bind({})(prompt="clash")
    assert issubclass(_FactoryArgumentError, TypeError)


def test_positional_arguments_to_a_bound_factory_are_a_setup_error():
    with pytest.raises(_FactoryArgumentError, match="keyword arguments only"):
        make_agent.bind({})("box")


def test_resources_reach_the_factory_only_if_it_declares_them():
    built = make_agent.bind({})._with_resources({"sandbox": "box", "unrelated": "dropped"})()
    assert built["sandbox"] == "box"


def test_kwargs_do_not_count_as_declaring_a_resource():
    factory = tunable(lambda effort="low", **kwargs: kwargs, inject={"effort": OneOf("low", "high")})
    assert factory.bind({})._with_resources({"sandbox": "box"})() == {}


def test_a_resource_named_like_a_marker_is_a_setup_error():
    with pytest.raises(_FactoryArgumentError, match="prompt"):
        make_agent.bind({})._with_resources({"prompt": "clash"})


def test_a_call_kwarg_wins_over_a_resource_of_the_same_name():
    bound = make_agent.bind({})._with_resources({"sandbox": "from environment"})
    assert bound(sandbox="from rollout")["sandbox"] == "from rollout"
    assert bound()["sandbox"] == "from environment"


def test_with_resources_leaves_the_bound_factory_unchanged():
    bound = make_agent.bind({})
    bound._with_resources({"sandbox": "box"})
    assert bound()["sandbox"] is None


def test_a_factory_without_a_name_takes_arguments():
    def build(model, prompt="hi", sandbox=None):
        return model, prompt, sandbox

    factory = tunable(functools.partial(build, "model"), inject={"prompt": TextValue()})
    bound = factory.bind({})
    assert bound._with_resources({"sandbox": "box", "unrelated": "dropped"})() == ("model", "hi", "box")
    with pytest.raises(_FactoryArgumentError, match="functools.partial"):
        bound(prompt="clash")


# --- a real component swap ---


@tunable(
    conversation_manager=choice(
        component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8)),
        component(SlidingWindowConversationManager, window_size=Int(4, 200)),
    )
)
def make_strands_agent(conversation_manager=None):
    return Agent(model=StubModel(), conversation_manager=conversation_manager, callback_handler=None)


def test_a_real_component_swap_builds_through_the_constructor_with_its_hooks_registered():
    bound = make_strands_agent.bind(
        {
            "conversation_manager": "SlidingWindowConversationManager",
            "conversation_manager.SlidingWindowConversationManager.window_size": 10,
        }
    )
    agent, other = bound(), bound()
    manager = agent.conversation_manager
    assert isinstance(manager, SlidingWindowConversationManager)
    assert manager.window_size == 10
    assert manager is not other.conversation_manager
    callbacks = agent.hooks.get_callbacks_for(BeforeModelCallEvent(agent=agent, invocation_state={}))
    assert any(getattr(callback, "__self__", None) is manager for callback in callbacks)


@tunable(conversation_manager=choice(on=component(SlidingWindowConversationManager, window_size=Int(4, 200)), off=None))
def make_optional_agent(conversation_manager=None):
    return {"conversation_manager": conversation_manager}


def test_an_optional_component_builds_fresh_when_on_and_passes_none_when_off():
    on = make_optional_agent.bind({})
    first, second = on()["conversation_manager"], on()["conversation_manager"]
    assert isinstance(first, SlidingWindowConversationManager) and first is not second
    assert make_optional_agent.bind({"conversation_manager": "off"})()["conversation_manager"] is None


def test_the_plain_call_builds_the_baseline_component():
    assert isinstance(make_strands_agent().conversation_manager, SummarizingConversationManager)
