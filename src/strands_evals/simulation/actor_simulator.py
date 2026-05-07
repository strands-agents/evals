import logging
import random

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.types.content import Message

from strands_evals.case import Case
from strands_evals.simulation.profiles.actor_profile import DEFAULT_USER_PROFILE_SCHEMA
from strands_evals.simulation.prompt_templates.actor_profile_extraction import ACTOR_PROFILE_PROMPT_TEMPLATE
from strands_evals.simulation.prompt_templates.actor_system_prompt import DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE
from strands_evals.simulation.tools.goal_completion import get_conversation_goal_completion
from strands_evals.types.simulation import ActorOutputBase, ActorProfile, ActorResponse

logger = logging.getLogger(__name__)


class ActorSimulator:
    """Simulates an actor in multi-turn conversations for agent evaluation.

    ActorSimulator wraps a Strands Agent configured to behave as a specific actor
    (typically a user) in conversation scenarios. It maintains conversation history,
    generates contextually appropriate responses, and can assess goal completion.

    Attributes:
        agent: The underlying Strands Agent configured with actor behavior.
        actor_profile: The actor's profile containing traits, context, and goal.
        initial_query: The actor's first query in the conversation.
        conversation_history: List of conversation messages in Strands format.
        model_id: Model identifier for the underlying agent.
    """

    INITIAL_GREETINGS = [
        "hi! how can I help you today?",
        "hello! what can I assist you with?",
        "hi there! how may I help you?",
        "good day! what can I do for you?",
        "hello! what would you like to know?",
    ]

    @classmethod
    def from_case_for_user_simulator(
        cls,
        case: Case,
        system_prompt_template: str | None = None,
        tools: list | None = None,
        model: str | None = None,
        max_turns: int = 10,
    ) -> "ActorSimulator":
        """Create an ActorSimulator configured as a user simulator from a test case.

        Generates a realistic user profile and goal from case.input and optionally
        case.metadata["task_description"], then configures the simulator with
        user-specific defaults. If you already have a profile, use __init__() directly.

        Args:
            case: Test case containing input (initial query) and optional metadata with "task_description".
            system_prompt_template: Custom system prompt template. Uses DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE if None.
            tools: Additional tools available to the user. Defaults to goal completion tool only.
            model: Model identifier for the underlying agent. Uses Strands default if None.
            max_turns: Maximum number of conversation turns before stopping (default: 10).

        Returns:
            ActorSimulator configured for user simulation.

        Example:
            ```python
            from strands_evals import Case, ActorSimulator
            from strands import Agent

            case = Case(
                input="I need to book a flight to Paris",
                metadata={"task_description": "Flight booking confirmed"}
            )

            user_sim = ActorSimulator.from_case_for_user_simulator(case=case, max_turns=5)
            agent = Agent(system_prompt="You are a travel assistant.")

            user_message = case.input
            while user_sim.has_next():
                agent_response = agent(user_message)
                user_result = user_sim.act(str(agent_response))
                user_message = str(user_result.structured_output.message)
            ```
        """
        actor_profile = cls._generate_profile_from_case(case)

        return cls(
            actor_profile=actor_profile,
            initial_query=case.input,
            system_prompt_template=system_prompt_template,
            tools=tools,
            model=model,
            max_turns=max_turns,
        )

    @staticmethod
    def _generate_profile_from_case(case: Case) -> ActorProfile:
        """Generate user profile from case.

        Uses case.input and optionally case.metadata["task_description"] if present.

        Args:
            case: Test case with input and optional task_description in metadata.

        Returns:
            ActorProfile with generated traits, context, and goal.
        """
        initial_query = case.input
        task_description = case.metadata.get("task_description", "") if case.metadata else ""

        profile_prompt = ACTOR_PROFILE_PROMPT_TEMPLATE.format(
            initial_query=initial_query,
            task_description=task_description,
            example=DEFAULT_USER_PROFILE_SCHEMA,
        )
        profile_agent = Agent(callback_handler=None)
        result = profile_agent(profile_prompt, structured_output_model=ActorProfile)
        return result.structured_output

    def __init__(
        self,
        actor_profile: ActorProfile,
        initial_query: str,
        system_prompt_template: str | None = None,
        tools: list | None = None,
        model: str | None = None,
        max_turns: int = 10,
        *,
        structured_output_model: type[ActorOutputBase] | None = None,
    ):
        """Initialize an ActorSimulator with profile and goal.

        Use this constructor when you have a pre-defined ActorProfile. For automatic
        profile generation from test cases, use from_case_for_user_simulator() instead.

        Args:
            actor_profile: ActorProfile object containing traits, context, and actor_goal.
            initial_query: The actor's first query or message.
            system_prompt_template: System prompt for the actor. Accepts two shapes:

                - A template containing the `{actor_profile}` placeholder, which
                  is rendered via `str.format(actor_profile=...)` against the
                  actor's profile.
                - An already-rendered system prompt string with no
                  `{actor_profile}` placeholder, which is used verbatim.

                When `None` (the default), uses
                `DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE` which instructs the LLM
                to set `stop=true` on the structured response when the
                conversation goal is met.

                Pass an explicit template to override.
            tools: Additional tools available to the actor. Defaults to goal completion tool only.
            model: Model identifier for the underlying agent. Uses Strands default if None.
            max_turns: Maximum number of conversation turns before stopping (default: 10).
            structured_output_model: Optional Pydantic model to use for all `act()` calls.
                Must subclass `ActorOutputBase` and include a `message` field.
                When set, `act()` uses this model by default instead of `ActorResponse`.
                Can still be overridden per-call via `act(structured_output_model=...)`.

        Example:
            ```python
            from strands_evals.simulation import ActorSimulator
            from strands_evals.types.simulation import ActorOutputBase, ActorProfile

            class AgentInput(ActorOutputBase):
                message: str | None = None
                urgency: str = "normal"

            profile = ActorProfile(
                traits={"expertise_level": "expert", "communication_style": "technical"},
                context="A software engineer debugging a production issue.",
                actor_goal="Identify and resolve the memory leak."
            )

            simulator = ActorSimulator(
                actor_profile=profile,
                initial_query="Our service is experiencing high memory usage.",
                structured_output_model=AgentInput,
                max_turns=15,
            )

            # act() uses AgentInput automatically
            result = simulator.act(str(agent_response))
            result.structured_output  # AgentInput instance
            ```
        """
        self.actor_profile = actor_profile
        self.initial_query = initial_query
        self.conversation_history: list[Message] = []
        self.model_id = model
        self.stop = False
        self._turn_count = 0
        self._max_turns = max_turns
        self._structured_output_model = structured_output_model

        if structured_output_model is not None:
            self._validate_output_model(structured_output_model)

        if system_prompt_template is None:
            system_prompt_template = DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE

        if "{actor_profile}" in system_prompt_template:
            system_prompt = system_prompt_template.format(actor_profile=actor_profile.model_dump())
        else:
            system_prompt = system_prompt_template

        all_tools = [get_conversation_goal_completion]
        if tools:
            all_tools.extend(tools)

        self._initialize_conversation()

        self.agent = Agent(
            system_prompt=system_prompt,
            messages=self.conversation_history,
            tools=all_tools,
            model=self.model_id,
            callback_handler=None,
        )

    def _initialize_conversation(self):
        """Initialize the conversation history with a greeting and initial query."""
        selected_greeting = random.choice(self.INITIAL_GREETINGS)
        greeting_message = {"role": "user", "content": [{"text": selected_greeting}]}
        self.conversation_history.append(greeting_message)

        initial_query_message = {"role": "assistant", "content": [{"text": self.initial_query.strip()}]}
        self.conversation_history.append(initial_query_message)

    @staticmethod
    def _validate_output_model(model: type[ActorOutputBase]) -> None:
        """Validate that a structured output model is compatible with the simulator."""
        if not issubclass(model, ActorOutputBase):
            raise TypeError(
                f"structured_output_model must be a subclass of ActorOutputBase, got {model.__name__}."
            )
        if "message" not in model.model_fields:
            raise ValueError(
                f"structured_output_model {model.__name__} must have a 'message' field."
            )

    def act(
        self,
        agent_message: str,
        *,
        structured_output_model: type[ActorOutputBase] | None = None,
    ) -> AgentResult:
        """Generate the next actor message in the conversation.

        Processes the agent's message and generates a contextually appropriate
        response from the actor's perspective. The response is returned as an
        `AgentResult` whose `structured_output` is an `ActorResponse` (or the
        caller-provided `structured_output_model`).

        The provided model must subclass `ActorOutputBase` and have a `message`
        field. A `TypeError` is raised if not a subclass, and `ValueError` if
        `message` is missing.

        Args:
            agent_message: The agent's response to react to.
            structured_output_model: Optional Pydantic model to use instead of
                `ActorResponse`. Must subclass `ActorOutputBase` and include a
                `message` field.

        Returns:
            AgentResult with `structured_output` set to either `ActorResponse`
            or the caller-provided model instance.

        Example:
            ```python
            # Default usage
            result = simulator.act(str(agent_response))
            response = result.structured_output  # ActorResponse
            print(response.message)

            # Custom structured output
            result = simulator.act(str(agent_response), structured_output_model=MySchema)
            my_output = result.structured_output  # MySchema instance
            ```
        """
        model = structured_output_model or self._structured_output_model or ActorResponse
        self._validate_output_model(model)

        response = self.agent(agent_message.strip(), structured_output_model=model)
        self._turn_count += 1

        result = response.structured_output

        if result.stop:
            self.stop = True
            if hasattr(result, "stop_reason"):
                result.stop_reason = "goal_completed"
        elif self._turn_count >= self._max_turns:
            result.stop = True
            self.stop = True
            if hasattr(result, "stop_reason"):
                result.stop_reason = "max_turns"

        return response

    def has_next(self) -> bool:
        """Check if the conversation should continue.

        Returns False if the actor signalled stop or if the maximum number of
        turns has been reached.

        Returns:
            True if the conversation should continue, False otherwise.

        Example:
            ```python
            user_message = case.input
            while user_sim.has_next():
                agent_response = agent(user_message)
                user_result = user_sim.act(str(agent_response))
                user_message = str(user_result.structured_output.message)
            ```
        """
        return not self.stop
