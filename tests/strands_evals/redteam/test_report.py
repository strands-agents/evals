"""Tests for RedTeamReport."""

from strands_evals.redteam.report import AttackResult, RedTeamReport
from strands_evals.types.evaluation_report import EvaluationReport


def _case(name: str, attack_type: str, strategy: str, severity: str) -> dict:
    return {
        "name": name,
        "metadata": {
            "attack_type": attack_type,
            "strategy": strategy,
            "severity": severity,
        },
    }


def _report(evaluator: str, cases: list[dict], scores: list[float], passes: list[bool], reasons: list[str]):
    return EvaluationReport(
        evaluator_name=evaluator,
        overall_score=sum(scores) / len(scores) if scores else 0.0,
        scores=scores,
        cases=cases,
        test_passes=passes,
        reasons=reasons,
    )


class TestFromEvaluationReports:
    def test_empty_reports(self):
        report = RedTeamReport.from_evaluation_reports([])
        assert report.results == []
        assert report.overall_score == 0.0
        assert report.pass_rate == 0.0
        assert report.failed_cases == []

    def test_single_evaluator_single_case(self):
        eval_report = _report(
            "judge",
            cases=[_case("c0", "jailbreak", "gradual_escalation", "high")],
            scores=[0.0],
            passes=[False],
            reasons=["bypassed"],
        )
        report = RedTeamReport.from_evaluation_reports([eval_report])

        assert len(report.results) == 1
        r = report.results[0]
        assert r.case_name == "c0"
        assert r.attack_type == "jailbreak"
        assert r.strategy == "gradual_escalation"
        assert r.severity == "high"
        assert r.scores == {"judge": 0.0}
        assert r.passes == {"judge": False}
        assert r.reasons == {"judge": "bypassed"}

    def test_multiple_evaluators_merge_on_case_name(self):
        """Cases with the same name from different evaluators collapse into one AttackResult."""
        cases = [_case("c0", "jailbreak", "gradual_escalation", "high")]
        r1 = _report("judge", cases, scores=[0.0], passes=[False], reasons=["bypassed"])
        r2 = _report("attack_success", cases, scores=[0.9], passes=[False], reasons=["full compromise"])

        report = RedTeamReport.from_evaluation_reports([r1, r2])

        assert len(report.results) == 1
        r = report.results[0]
        assert r.scores == {"judge": 0.0, "attack_success": 0.9}
        assert r.passes == {"judge": False, "attack_success": False}
        assert set(r.reasons) == {"judge", "attack_success"}

    def test_missing_metadata_fills_defaults(self):
        eval_report = _report(
            "judge",
            cases=[{"name": "c0"}],  # no metadata
            scores=[1.0],
            passes=[True],
            reasons=[""],
        )
        report = RedTeamReport.from_evaluation_reports([eval_report])

        r = report.results[0]
        assert r.attack_type == "unknown"
        assert r.strategy == "unknown"
        assert r.severity == "unknown"


class TestAttackResult:
    def test_score_uses_min_across_evaluators(self):
        r = AttackResult(
            case_name="c0",
            attack_type="x",
            strategy="y",
            severity="low",
            scores={"a": 0.2, "b": 0.8},
        )
        assert r.score == 0.2

    def test_score_is_zero_when_empty(self):
        r = AttackResult(case_name="c0", attack_type="x", strategy="y", severity="low")
        assert r.score == 0.0

    def test_passed_is_all_or_nothing(self):
        r = AttackResult(
            case_name="c0",
            attack_type="x",
            strategy="y",
            severity="low",
            passes={"a": True, "b": False},
        )
        assert r.passed is False

        r2 = AttackResult(
            case_name="c0",
            attack_type="x",
            strategy="y",
            severity="low",
            passes={"a": True, "b": True},
        )
        assert r2.passed is True

    def test_passed_defaults_true_when_no_evaluators(self):
        r = AttackResult(case_name="c0", attack_type="x", strategy="y", severity="low")
        assert r.passed is True

    def test_reason_joins_across_evaluators(self):
        r = AttackResult(
            case_name="c0",
            attack_type="x",
            strategy="y",
            severity="low",
            reasons={"a": "first", "b": "second"},
        )
        assert "[a] first" in r.reason
        assert "[b] second" in r.reason
        assert " | " in r.reason


class TestAggregations:
    def _build(self) -> RedTeamReport:
        """Three cases: 2 jailbreak (1 pass, 1 fail) + 1 prompt_extraction (fail)."""
        cases_a = [
            _case("c0", "jailbreak", "gradual_escalation", "high"),
            _case("c1", "jailbreak", "gradual_escalation", "high"),
            _case("c2", "prompt_extraction", "gradual_escalation", "high"),
        ]
        return RedTeamReport.from_evaluation_reports(
            [
                _report("judge", cases_a, scores=[1.0, 0.0, 0.0], passes=[True, False, False], reasons=["", "", ""]),
            ]
        )

    def test_overall_score(self):
        report = self._build()
        assert report.overall_score == (1.0 + 0.0 + 0.0) / 3

    def test_pass_rate(self):
        report = self._build()
        assert report.pass_rate == 1 / 3

    def test_failed_cases_sorted_by_score(self):
        report = self._build()
        failed = report.failed_cases
        assert len(failed) == 2
        assert all(not r.passed for r in failed)
        # sorted ascending by score
        assert failed[0].score <= failed[1].score

    def test_by_attack_type_summary(self):
        report = self._build()
        groups = report.by_attack_type()

        by_name = {g.group_name: g for g in groups}
        assert set(by_name) == {"jailbreak", "prompt_extraction"}

        jb = by_name["jailbreak"]
        assert jb.count == 2
        assert jb.avg_score == 0.5
        assert jb.pass_rate == 0.5

        pe = by_name["prompt_extraction"]
        assert pe.count == 1
        assert pe.avg_score == 0.0
        assert pe.pass_rate == 0.0

    def test_by_strategy_groups_all_together(self):
        report = self._build()
        groups = report.by_strategy()
        assert len(groups) == 1
        assert groups[0].group_name == "gradual_escalation"
        assert groups[0].count == 3

    def test_summaries_sorted_by_avg_score(self):
        report = self._build()
        groups = report.by_attack_type()
        scores = [g.avg_score for g in groups]
        assert scores == sorted(scores)


class TestDisplay:
    def test_no_results_does_not_raise(self):
        RedTeamReport([]).display()

    def test_with_results_does_not_raise(self):
        cases = [_case("c0", "jailbreak", "gradual_escalation", "high")]
        report = RedTeamReport.from_evaluation_reports(
            [_report("judge", cases, scores=[0.0], passes=[False], reasons=["bypassed"])]
        )
        report.display()
