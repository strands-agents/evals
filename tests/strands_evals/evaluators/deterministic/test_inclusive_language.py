import pytest

from strands_evals.evaluators.deterministic.inclusive_language import InclusiveLanguage
from strands_evals.types import EvaluationData
from strands_evals.types.evaluation import EvaluationOutput


class TestInclusiveLanguageDefaults:
    def test_passes_when_no_banned_terms(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="Use the denylist to block bad actors")
        results = evaluator.evaluate(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

    def test_fails_when_blacklist_found(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="Add the IP to the blacklist")
        results = evaluator.evaluate(data)
        assert results == [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="found 1 non-inclusive term(s): 'blacklist' -> 'denylist'",
            )
        ]

    def test_fails_when_whitelist_found(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="Add it to the whitelist")
        results = evaluator.evaluate(data)
        assert results == [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="found 1 non-inclusive term(s): 'whitelist' -> 'allowlist'",
            )
        ]

    def test_fails_when_master_found(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="Push to the master branch")
        results = evaluator.evaluate(data)
        assert results == [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="found 1 non-inclusive term(s): 'master' -> 'primary'",
            )
        ]

    def test_fails_when_slave_found(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="Configure the slave node")
        results = evaluator.evaluate(data)
        assert results == [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="found 1 non-inclusive term(s): 'slave' -> 'replica'",
            )
        ]

    def test_reports_multiple_terms(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="The master sends to the slave")
        results = evaluator.evaluate(data)
        assert results == [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="found 2 non-inclusive term(s): 'master' -> 'primary', 'slave' -> 'replica'",
            )
        ]


class TestInclusiveLanguageCaseSensitivity:
    def test_case_insensitive_uppercase(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="Add to BLACKLIST")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False

    def test_case_insensitive_mixed_case(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="The Whitelist is updated")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False


class TestInclusiveLanguageWordBoundary:
    def test_no_false_positive_on_masterful(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="That was a masterful performance")
        results = evaluator.evaluate(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

    def test_no_false_positive_on_mastering(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="She is mastering the skill")
        results = evaluator.evaluate(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

    def test_no_false_positive_on_slavery(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="The history of slavery is complex")
        results = evaluator.evaluate(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

    def test_matches_term_at_start_of_string(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="master is the default branch")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False

    def test_matches_term_at_end_of_string(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="push to master")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False

    def test_matches_term_with_punctuation_boundary(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="check the blacklist, then proceed")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False


class TestInclusiveLanguageCustomTerms:
    def test_custom_terms_override_defaults(self):
        custom = {"legacy": "historical"}
        evaluator = InclusiveLanguage(terms=custom)
        # Default term should not trigger
        data = EvaluationData(input="q", actual_output="Push to the master branch")
        results = evaluator.evaluate(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

    def test_custom_terms_detected(self):
        custom = {"legacy": "historical", "deprecated": "removed"}
        evaluator = InclusiveLanguage(terms=custom)
        data = EvaluationData(input="q", actual_output="This is a legacy system")
        results = evaluator.evaluate(data)
        assert results == [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="found 1 non-inclusive term(s): 'legacy' -> 'historical'",
            )
        ]

    def test_empty_terms_always_passes(self):
        evaluator = InclusiveLanguage(terms={})
        data = EvaluationData(input="q", actual_output="blacklist whitelist master slave")
        results = evaluator.evaluate(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]


class TestInclusiveLanguageSharedState:
    def test_default_terms_not_shared_between_instances(self):
        evaluator1 = InclusiveLanguage()
        evaluator2 = InclusiveLanguage()
        evaluator1.terms["newterm"] = "replacement"
        assert "newterm" not in evaluator2.terms
        assert "newterm" not in InclusiveLanguage.DEFAULT_TERMS


class TestInclusiveLanguageEdgeCases:
    def test_none_actual_output_passes(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output=None)
        results = evaluator.evaluate(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

    def test_empty_string_passes(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="")
        results = evaluator.evaluate(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

    def test_numeric_output_coerced(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output=42)
        results = evaluator.evaluate(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

    def test_reason_on_pass(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="clean text")
        results = evaluator.evaluate(data)
        assert results[0].reason == "no non-inclusive terms found"

    def test_reason_on_fail_format(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="Add to blacklist")
        results = evaluator.evaluate(data)
        assert results[0].reason == "found 1 non-inclusive term(s): 'blacklist' -> 'denylist'"


class TestInclusiveLanguageAsync:
    @pytest.mark.asyncio
    async def test_evaluate_async_delegates_to_evaluate(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="Use the denylist")
        results = await evaluator.evaluate_async(data)
        assert results == [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

    @pytest.mark.asyncio
    async def test_evaluate_async_detects_terms(self):
        evaluator = InclusiveLanguage()
        data = EvaluationData(input="q", actual_output="Add to blacklist")
        results = await evaluator.evaluate_async(data)
        assert results == [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="found 1 non-inclusive term(s): 'blacklist' -> 'denylist'",
            )
        ]


class TestInclusiveLanguageSerialization:
    def test_to_dict_default_terms(self):
        evaluator = InclusiveLanguage()
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "InclusiveLanguage"
        assert d["terms"] == InclusiveLanguage.DEFAULT_TERMS

    def test_to_dict_custom_terms(self):
        custom = {"foo": "bar"}
        evaluator = InclusiveLanguage(terms=custom)
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "InclusiveLanguage"
        assert d["terms"] == {"foo": "bar"}

    def test_name_parameter(self):
        evaluator = InclusiveLanguage(name="my_scanner")
        assert evaluator.get_name() == "my_scanner"

    def test_default_name(self):
        evaluator = InclusiveLanguage()
        assert evaluator.get_name() == "InclusiveLanguage"
