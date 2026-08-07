from ...extractors.skills import extract_skill_load_events
from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..evaluator import Evaluator


class SkillInvoked(Evaluator[InputT, OutputT]):
    """Checks if a specific skill was invoked in the trajectory."""

    def __init__(self, skill_name: str, name: str | None = None):
        super().__init__(name=name)
        self.skill_name = skill_name

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        trajectory = evaluation_case.actual_trajectory
        if trajectory is None:
            return [EvaluationOutput(score=0.0, test_pass=False, reason="no trajectory provided")]

        # Read the individual attempts rather than the per-skill summary, so this check applies its
        # own definition of "invoked": a refused load does not count, because the agent never
        # received the skill and an assertion that it was used is false. The summary folds a
        # refusal and a later success into one loaded row, which is right for judging the choice
        # and wrong for asserting the skill was in play.
        attempts = [e for e in extract_skill_load_events(trajectory) if e.name == self.skill_name]
        found = any(e.status == "loaded" for e in attempts)
        refusal = next((e for e in attempts if e.status == "failed"), None)
        if found:
            reason = f"skill '{self.skill_name}' was invoked"
        elif refusal is not None:
            # Report what the harness said: a check that fails on a misspelled skill name and one
            # that fails because nothing was mounted call for different fixes.
            detail = f": {refusal.error}" if refusal.error else ""
            attempted = f" ({len(attempts)} attempts)" if len(attempts) > 1 else ""
            reason = f"skill '{self.skill_name}' was requested but the load failed{attempted}{detail}"
        elif attempts:
            # Every attempt was made and none has a recorded outcome, so the trajectory does not
            # say whether the skill was received. Reported as not invoked, since the check asserts
            # use and use was not observed, but named apart from never asking.
            reason = f"skill '{self.skill_name}' was requested but the trajectory records no outcome"
        else:
            reason = f"skill '{self.skill_name}' was not invoked"
        return [
            EvaluationOutput(
                score=1.0 if found else 0.0,
                test_pass=found,
                reason=reason,
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)
