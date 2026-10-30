import json

import pytest
from pydantic import ValidationError

from strands_evals.types.optimization import Evidence, Observation, Patch


def _canonical(values):
    return json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@pytest.mark.parametrize(
    "value",
    [1, 1.0, 0.5, True, False, "1", None],
    ids=["int", "integral-float", "float", "true", "false", "str", "none"],
)
def test_patch_values_keep_their_exact_type_through_json(value):
    patch = Patch(updates={"key": value}, rationale="")

    restored = Patch.model_validate_json(patch.model_dump_json())

    assert type(restored.updates["key"]) is type(value)
    assert restored.updates["key"] == value


def test_canonical_json_of_values_is_unchanged_by_a_round_trip():
    updates = {"temperature": 0.5, "max_turns": 3, "use_cache": True, "prompt": "Be brief.", "budget": 2.0}
    patch = Patch(updates=updates, rationale="")

    restored = Patch.model_validate_json(patch.model_dump_json())

    assert _canonical(restored.updates) == _canonical(updates)


def test_observation_round_trips_through_json():
    observation = Observation(
        patch=Patch(updates={"max_turns": 3}, rationale="fewer turns", state={"trial": 4}),
        candidate_id="abc123def456",
        values={"max_turns": 3, "system_prompt": "Be brief."},
        objective=0.75,
        objectives=[0.75],
        outcome="infeasible",
        reason="tokens <= 4000: P(met) 0.62 < 0.9",
        violations=[210.0],
    )

    restored = Observation.model_validate_json(observation.model_dump_json())

    assert restored == observation
    assert restored.source_id is None


def test_observation_rejects_a_gate_loop_outcome():
    with pytest.raises(ValidationError):
        Observation(
            patch=Patch(updates={}, rationale=""),
            candidate_id=None,
            values=None,
            objective=None,
            objectives=None,
            outcome="accepted",
            reason="",
            violations=None,
        )


def test_evidence_round_trips_through_json():
    evidence = Evidence(
        candidate_id="abc123def456",
        values={"temperature": 0.2, "use_cache": True},
        origin="numbers",
        objective=None,
        objectives=None,
        violations=None,
        outcome="ineligible",
    )

    restored = Evidence.model_validate_json(evidence.model_dump_json())

    assert restored == evidence
    assert type(restored.values["use_cache"]) is bool


def test_records_are_frozen():
    patch = Patch(updates={"max_turns": 3}, rationale="")

    with pytest.raises(ValidationError):
        patch.rationale = "changed"
