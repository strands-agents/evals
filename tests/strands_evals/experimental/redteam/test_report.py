"""Tests for RedTeamReport."""

from strands_evals.experimental.redteam.report import AttackResult, RedTeamReport
from strands_evals.types.evaluation_report import EvaluationReport


def _case(name: str, risk_category: str, strategy: str, severity: str) -> dict:
    return {
        "name": name,
        "metadata": {
            "risk_category": risk_category,
            "strategy": strategy,
            "severity": severity,
        },
    }


def _eval_report(evaluator: str, cases: list[dict], scores: list[float], passes: list[bool], reasons: list[str]):
    return EvaluationReport(
        evaluator_name=evaluator,
        overall_score=sum(scores) / len(scores) if scores else 0.0,
        scores=scores,
        cases=cases,
        test_passes=passes,
        reasons=reasons,
    )


def _empty_report() -> RedTeamReport:
    return RedTeamReport(
        evaluator_name="RedTeam",
        overall_score=0.0,
        scores=[],
        cases=[],
        test_passes=[],
    )


class TestFromEvaluationReports:
    def test_empty_reports(self):
        report = RedTeamReport.from_evaluation_reports([])
        assert report.attack_results() == []
        assert report.overall_score == 0.0
        assert report.failed_cases == []

    def test_single_evaluator_single_case(self):
        eval_report = _eval_report(
            "judge",
            cases=[_case("c0", "guideline_bypass", "gradual_escalation", "high")],
            scores=[0.0],
            passes=[False],
            reasons=["bypassed"],
        )
        report = RedTeamReport.from_evaluation_reports([eval_report])

        results = report.attack_results()
        assert len(results) == 1
        r = results[0]
        assert r.case_name == "c0"
        assert r.risk_category == "guideline_bypass"
        assert r.strategy == "gradual_escalation"
        assert r.severity == "high"
        assert r.scores == {"judge": 0.0}
        assert r.passes == {"judge": False}
        assert r.reasons == {"judge": "bypassed"}

    def test_multiple_evaluators_merge_on_case_name(self):
        cases = [_case("c0", "guideline_bypass", "gradual_escalation", "high")]
        r1 = _eval_report("judge", cases, scores=[0.0], passes=[False], reasons=["bypassed"])
        r2 = _eval_report("attack_success", cases, scores=[0.9], passes=[False], reasons=["full compromise"])

        report = RedTeamReport.from_evaluation_reports([r1, r2])

        results = report.attack_results()
        assert len(results) == 1
        r = results[0]
        assert r.scores == {"judge": 0.0, "attack_success": 0.9}
        assert r.passes == {"judge": False, "attack_success": False}
        assert set(r.reasons) == {"judge", "attack_success"}

    def test_missing_metadata_fills_defaults(self):
        eval_report = _eval_report(
            "judge",
            cases=[{"name": "c0"}],
            scores=[1.0],
            passes=[True],
            reasons=[""],
        )
        report = RedTeamReport.from_evaluation_reports([eval_report])

        r = report.attack_results()[0]
        assert r.risk_category == "unknown"
        assert r.strategy == "unknown"
        assert r.severity == "unknown"


class TestAttackResult:
    def test_score_uses_min_across_evaluators(self):
        r = AttackResult(
            case_name="c0",
            risk_category="x",
            strategy="y",
            severity="low",
            scores={"a": 0.2, "b": 0.8},
        )
        assert r.score == 0.2

    def test_score_is_zero_when_empty(self):
        r = AttackResult(case_name="c0", risk_category="x", strategy="y", severity="low")
        assert r.score == 0.0

    def test_passed_is_all_or_nothing(self):
        r = AttackResult(
            case_name="c0",
            risk_category="x",
            strategy="y",
            severity="low",
            passes={"a": True, "b": False},
        )
        assert r.passed is False

        r2 = AttackResult(
            case_name="c0",
            risk_category="x",
            strategy="y",
            severity="low",
            passes={"a": True, "b": True},
        )
        assert r2.passed is True

    def test_passed_defaults_true_when_no_evaluators(self):
        r = AttackResult(case_name="c0", risk_category="x", strategy="y", severity="low")
        assert r.passed is True

    def test_reason_joins_across_evaluators(self):
        r = AttackResult(
            case_name="c0",
            risk_category="x",
            strategy="y",
            severity="low",
            reasons={"a": "first", "b": "second"},
        )
        assert "[a] first" in r.reason
        assert "[b] second" in r.reason
        assert " | " in r.reason


class TestAggregations:
    def _build(self) -> RedTeamReport:
        cases_a = [
            _case("c0", "guideline_bypass", "gradual_escalation", "high"),
            _case("c1", "guideline_bypass", "gradual_escalation", "high"),
            _case("c2", "system_prompt_leak", "gradual_escalation", "high"),
        ]
        return RedTeamReport.from_evaluation_reports(
            [
                _eval_report(
                    "judge", cases_a, scores=[1.0, 0.0, 0.0], passes=[True, False, False], reasons=["", "", ""]
                ),
            ]
        )

    def test_failed_cases_sorted_by_score(self):
        report = self._build()
        failed = report.failed_cases
        assert len(failed) == 2
        assert all(not r.passed for r in failed)
        assert failed[0].score <= failed[1].score

    def test_by_risk_category_summary(self):
        report = self._build()
        groups = report.by_risk_category()

        by_name = {g.group_name: g for g in groups}
        assert set(by_name) == {"guideline_bypass", "system_prompt_leak"}

        jb = by_name["guideline_bypass"]
        assert jb.count == 2
        assert jb.avg_score == 0.5
        assert jb.pass_rate == 0.5

        pe = by_name["system_prompt_leak"]
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
        groups = report.by_risk_category()
        scores = [g.avg_score for g in groups]
        assert scores == sorted(scores)


class TestDisplay:
    def test_no_results_does_not_raise(self):
        _empty_report().display()

    def test_with_results_does_not_raise(self):
        cases = [_case("c0", "guideline_bypass", "gradual_escalation", "high")]
        report = RedTeamReport.from_evaluation_reports(
            [_eval_report("judge", cases, scores=[0.0], passes=[False], reasons=["bypassed"])]
        )
        report.display()
