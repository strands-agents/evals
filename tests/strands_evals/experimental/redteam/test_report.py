"""Tests for RedTeamReport."""

from strands_evals.experimental.redteam.report import AttackResult, RedTeamReport
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport


def _case(name: str, risk_category: str, strategy: str, severity: str, **extra) -> dict:
    return {
        "name": name,
        "metadata": {
            "risk_category": risk_category,
            "strategy": strategy,
            "severity": severity,
            **extra,
        },
    }


def _eval_report(
    evaluator: str,
    cases: list[dict],
    scores: list[float],
    passes: list[bool],
    reasons: list[str],
    detailed_results: list[list[EvaluationOutput]] | None = None,
):
    # Mirror what Experiment.run_evaluations does: tag each case row with its evaluator.
    tagged = [{**c, "evaluator": evaluator} for c in cases]
    return EvaluationReport(
        overall_score=sum(scores) / len(scores) if scores else 0.0,
        scores=scores,
        cases=tagged,
        test_passes=passes,
        reasons=reasons,
        detailed_results=detailed_results or [],
    )


def _na(reason: str, test_pass: bool) -> list[EvaluationOutput]:
    """The detailed_results of a row whose evaluator had nothing to judge."""
    return [EvaluationOutput(score=0.0, test_pass=test_pass, reason=reason, label=NOT_APPLICABLE)]


def _flatten(*reports: EvaluationReport) -> EvaluationReport:
    """Match what Experiment.run_evaluations now hands to RedTeamReport.from_evaluation_report."""
    if len(reports) == 1:
        return reports[0]
    return EvaluationReport.flatten(list(reports))


def _empty_report() -> RedTeamReport:
    return RedTeamReport(
        overall_score=0.0,
        scores=[],
        cases=[],
        test_passes=[],
    )


class TestFromEvaluationReport:
    def test_empty_report(self):
        empty = EvaluationReport(overall_score=0.0, scores=[], cases=[], test_passes=[])
        report = RedTeamReport.from_evaluation_report(empty)
        assert report.attack_results() == []
        assert report.overall_score == 0.0
        assert report.failed_cases == []

    def test_single_evaluator_single_case(self):
        eval_report = _eval_report(
            "judge",
            cases=[_case("c0", "prompt_injection", "gradual_escalation", "high")],
            scores=[0.0],
            passes=[False],
            reasons=["bypassed"],
        )
        report = RedTeamReport.from_evaluation_report(_flatten(eval_report))

        results = report.attack_results()
        assert len(results) == 1
        r = results[0]
        assert r.case_name == "c0"
        assert r.risk_category == "prompt_injection"
        assert r.strategy == "gradual_escalation"
        assert r.severity == "high"
        assert r.scores == {"judge": 0.0}
        assert r.passes == {"judge": False}
        assert r.reasons == {"judge": "bypassed"}

    def test_multiple_evaluators_merge_on_case_name(self):
        cases = [_case("c0", "prompt_injection", "gradual_escalation", "high")]
        r1 = _eval_report("judge", cases, scores=[0.0], passes=[False], reasons=["bypassed"])
        r2 = _eval_report("attack_success", cases, scores=[0.9], passes=[False], reasons=["full compromise"])

        report = RedTeamReport.from_evaluation_report(_flatten(r1, r2))

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
        report = RedTeamReport.from_evaluation_report(_flatten(eval_report))

        r = report.attack_results()[0]
        assert r.risk_category == "unknown"
        assert r.strategy == "unknown"
        assert r.severity == "unknown"


class TestAttackResult:
    def test_score_uses_max_across_evaluators(self):
        # red-team aggregates worst-case = the strongest breach signal across evaluators
        r = AttackResult(
            case_name="c0",
            risk_category="x",
            strategy="y",
            severity="low",
            scores={"a": 0.2, "b": 0.8},
        )
        assert r.score == 0.8

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


class TestErroredState:
    def test_errored_from_task_error_reason(self):
        """A crashed attack reaches the report as the base-recorded task error reason on every evaluator."""
        cases = [_case("c0", "guideline_bypass", "gradual_escalation", "high")]
        r1 = _eval_report("judge", cases, scores=[0.0], passes=[False], reasons=["An error occurred: target reset"])
        r2 = _eval_report(
            "attack_success", cases, scores=[0.0], passes=[False], reasons=["An error occurred: target reset"]
        )
        r = RedTeamReport.from_evaluation_report(_flatten(r1, r2)).attack_results()[0]
        assert r.errored is True
        assert r.state == "errored"
        assert r.scores == {} and r.passes == {}
        # the error text stays visible in the report
        assert "[judge] An error occurred: target reset" in r.reason

    def test_errored_from_judge_error_reason(self):
        """A judge failure reaches the report only as a base-recorded error reason."""
        eval_report = _eval_report(
            "attack_success",
            cases=[_case("c0", "guideline_bypass", "gradual_escalation", "high")],
            scores=[0.0],
            passes=[False],
            reasons=["Evaluator error: judge failed to produce structured output"],
        )
        r = RedTeamReport.from_evaluation_report(_flatten(eval_report)).attack_results()[0]
        assert r.errored is True
        assert r.state == "errored"

    def test_state_breached_and_defended(self):
        breached = AttackResult(case_name="c", risk_category="x", strategy="y", severity="low", passes={"a": False})
        defended = AttackResult(case_name="c", risk_category="x", strategy="y", severity="low", passes={"a": True})
        assert breached.state == "breached"
        assert defended.state == "defended"

    def test_error_without_judgment_is_errored_not_breached(self):
        """An error row with passes=False is errored, not breached -- error must not read as success."""
        r = AttackResult(
            case_name="c", risk_category="x", strategy="y", severity="low", reasons={"a": "Evaluator error: boom"}
        )
        assert r.state == "errored"

    def test_one_erroring_evaluator_does_not_hide_a_breach(self, capsys):
        """A breach scored by one evaluator survives another evaluator failing to score the same case."""
        cases = [_case("c0", "guideline_bypass", "gradual_escalation", "high")]
        r1 = _eval_report("attack_success", cases, scores=[1.0], passes=[False], reasons=["full compromise"])
        r2 = _eval_report("refusal_judge", cases, scores=[0.0], passes=[False], reasons=["Evaluator error: timeout"])
        report = RedTeamReport.from_evaluation_report(_flatten(r1, r2))

        r = report.attack_results()[0]
        assert r.errored is False
        assert r.state == "breached"
        assert r.score == 1.0
        assert "[refusal_judge] Evaluator error: timeout" in r.reason

        report.display()
        out = capsys.readouterr().out
        assert "Result: FAIL -- 1 of 1 attacks breached (100.0%)" in out
        assert "errored" not in out.split("Attack matrix")[0]

    def test_all_errored_run_is_not_pass(self, capsys):
        """With no scored attacks there is no evidence either way, so the verdict must not be PASS."""
        cases = [
            _case("c0", "guideline_bypass", "gradual_escalation", "high"),
            _case("c1", "guideline_bypass", "gradual_escalation", "high"),
        ]
        report = RedTeamReport.from_evaluation_report(
            _flatten(
                _eval_report(
                    "attack_success",
                    cases,
                    scores=[0.0, 0.0],
                    passes=[False, False],
                    reasons=["An error occurred: connection refused", "An error occurred: connection refused"],
                )
            )
        )
        report.display()
        out = capsys.readouterr().out
        assert "Result: ERROR -- 0 of 0 attacks breached (0.0%, 2 errored excluded)" in out
        assert "PASS" not in out
        assert " ERROR\n" in out.split("All attacks")[0]  # matrix row verdict

    def test_accessors_exclude_errored(self):
        cases = [
            _case("c0", "guideline_bypass", "gradual_escalation", "high"),
            _case("c1", "guideline_bypass", "gradual_escalation", "high"),
            _case("c2", "system_prompt_leak", "gradual_escalation", "high"),
        ]
        report = RedTeamReport.from_evaluation_report(
            _flatten(
                _eval_report(
                    "attack_success",
                    cases,
                    scores=[0.9, 0.0, 0.0],
                    passes=[False, False, False],
                    reasons=["breach", "Evaluator error: boom", "Evaluator error: boom"],
                )
            )
        )
        assert [r.case_name for r in report.failed_cases] == ["c0"]

        by_name = {g.group_name: g for g in report.by_risk_category()}
        gb = by_name["guideline_bypass"]
        assert (gb.count, gb.errored, gb.avg_score, gb.pass_rate) == (2, 1, 0.9, 0.0)
        # every attack errored: no score/pass rate to report rather than a misleading 0.0
        spl = by_name["system_prompt_leak"]
        assert (spl.count, spl.errored, spl.avg_score, spl.pass_rate) == (1, 1, None, None)
        # all-errored groups sort last
        assert report.by_risk_category()[-1].group_name == "system_prompt_leak"

    def test_asr_excludes_errored_from_denominator(self, capsys):
        cases = [
            _case("c0", "guideline_bypass", "gradual_escalation", "high"),
            _case("c1", "guideline_bypass", "gradual_escalation", "high"),
            _case("c2", "guideline_bypass", "gradual_escalation", "high"),
        ]
        report = RedTeamReport.from_evaluation_report(
            _flatten(
                _eval_report(
                    "attack_success",
                    cases,
                    scores=[0.9, 0.1, 0.0],
                    passes=[False, True, False],
                    reasons=["breach", "defended", "An error occurred: target crashed"],
                )
            )
        )
        report.display()
        out = capsys.readouterr().out
        # 1 breach out of 2 scored (c2 errored, excluded) => 50.0%, not 1/3 => 33.3%
        assert "1 of 2 attacks breached" in out
        assert "50.0%" in out
        assert "1 errored excluded" in out
        assert "· 1 errored" in out  # footer count

    def test_overall_score_excludes_unjudged_rows(self):
        """overall_score averages judged rows only, unlike the base mean that counts errors as 0.0."""
        cases = [
            _case("c0", "guideline_bypass", "gradual_escalation", "high"),
            _case("c1", "guideline_bypass", "gradual_escalation", "high"),
            _case("c2", "guideline_bypass", "gradual_escalation", "high"),
        ]
        base = _eval_report(
            "attack_success",
            cases,
            scores=[0.9, 0.0, 0.0],
            passes=[False, False, False],
            reasons=["breach", "Evaluator error: boom", "no trajectory provided"],
            detailed_results=[[], [], _na("no trajectory provided", False)],
        )
        assert base.overall_score == 0.3
        assert RedTeamReport.from_evaluation_report(base).overall_score == 0.9

    def test_overall_score_is_zero_when_nothing_judged(self):
        cases = [_case("c0", "guideline_bypass", "gradual_escalation", "high")]
        base = _eval_report("attack_success", cases, [0.0], [False], ["An error occurred: target crashed"])
        assert RedTeamReport.from_evaluation_report(base).overall_score == 0.0


class TestNotApplicable:
    """NOT_APPLICABLE rows (e.g. a skill evaluator on a run with no skill) carry no attack verdict."""

    _CASES = [_case("c0", "guideline_bypass", "gradual_escalation", "high")]

    def _result(self, *reports: EvaluationReport) -> AttackResult:
        return RedTeamReport.from_evaluation_report(_flatten(*reports)).attack_results()[0]

    def test_na_does_not_turn_an_errored_judge_into_defended(self):
        r = self._result(
            _eval_report("attack_success", self._CASES, [0.0], [False], ["Evaluator error: timeout"]),
            _eval_report("skill", self._CASES, [0.0], [True], ["no skill invoked"], [_na("no skill invoked", True)]),
        )
        assert r.passes == {}
        assert r.state == "errored"
        assert "[skill] no skill invoked" in r.reason

    def test_failing_na_is_not_a_breach(self):
        """A failing NOT_APPLICABLE row means absent data, not a successful attack."""
        r = self._result(
            _eval_report("attack_success", self._CASES, [0.0], [True], ["refused"]),
            _eval_report(
                "skill", self._CASES, [0.0], [False], ["no trajectory provided"], [_na("no trajectory provided", False)]
            ),
        )
        assert r.passes == {"attack_success": True}
        assert r.state == "defended"

    def test_na_does_not_hide_a_breach(self):
        r = self._result(
            _eval_report("attack_success", self._CASES, [1.0], [False], ["full compromise"]),
            _eval_report("skill", self._CASES, [0.0], [True], ["no skill invoked"], [_na("no skill invoked", True)]),
        )
        assert r.state == "breached"
        assert r.score == 1.0

    def test_all_na_case_is_errored(self):
        """No evaluator could judge the case, so it must not count as defended in the ASR denominator."""
        r = self._result(
            _eval_report("skill", self._CASES, [0.0], [True], ["no skill invoked"], [_na("no skill invoked", True)])
        )
        assert r.state == "errored"

    def test_row_with_some_judged_outputs_is_kept(self):
        """Only an all-NOT_APPLICABLE row is dropped; one judged output makes the row a real verdict."""
        mixed = _na("skill_a: no prescribed steps", True) + [EvaluationOutput(score=0.2, test_pass=False, reason="b")]
        r = self._result(_eval_report("skill", self._CASES, [0.2], [False], ["b"], [mixed]))
        assert r.passes == {"skill": False}
        assert r.state == "breached"


class TestAggregations:
    def _build(self) -> RedTeamReport:
        cases_a = [
            _case("c0", "prompt_injection", "gradual_escalation", "high"),
            _case("c1", "prompt_injection", "gradual_escalation", "high"),
            _case("c2", "sensitive_data", "gradual_escalation", "high"),
        ]
        return RedTeamReport.from_evaluation_report(
            _flatten(
                _eval_report(
                    "judge", cases_a, scores=[1.0, 0.0, 0.0], passes=[True, False, False], reasons=["", "", ""]
                ),
            )
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
        assert set(by_name) == {"prompt_injection", "sensitive_data"}

        jb = by_name["prompt_injection"]
        assert jb.count == 2
        assert jb.avg_score == 0.5
        assert jb.pass_rate == 0.5

        pe = by_name["sensitive_data"]
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

    def test_matrix_pivots_cross_product_on_base_case(self, capsys):
        """A case run under 2 strategies is ONE matrix row with 2 score columns, not
        two rows. The cross-product names work items '{case}__{strategy}', so the matrix
        must strip the suffix to pivot correctly."""
        cases = [
            {"name": "exfil__cre-3", "metadata": {"risk_category": "x", "strategy": "cre-3", "severity": "high"}},
            {"name": "exfil__cre-5", "metadata": {"risk_category": "x", "strategy": "cre-5", "severity": "high"}},
        ]
        report = RedTeamReport.from_evaluation_report(
            _flatten(_eval_report("attack_success", cases, scores=[0.2, 0.1], passes=[True, True], reasons=["", ""]))
        )
        report.display()
        out = capsys.readouterr().out
        matrix = out.split("All attacks")[0]  # matrix section only
        # one row for the base case "exfil" (not "exfil__cre-3" / "exfil__cre-5")
        assert "exfil__" not in matrix
        assert matrix.count("\n  exfil ") == 1 or "  exfil " in matrix
        # both strategy columns present in the header
        assert "cre-3" in matrix and "cre-5" in matrix

    def test_with_results_does_not_raise(self):
        cases = [_case("c0", "prompt_injection", "gradual_escalation", "high")]
        report = RedTeamReport.from_evaluation_report(
            _flatten(_eval_report("judge", cases, scores=[0.0], passes=[False], reasons=["bypassed"]))
        )
        report.display()

    def test_matrix_falls_back_to_full_names_on_base_case_collision(self, capsys):
        """If stripping the cross-product suffix would collapse two distinct results onto
        the same cell, the matrix keeps full names so neither result is hidden."""
        # "a__b" under strategy "b" strips to "a"; a real case "a" under strategy "b" also
        # keys to ("a", "b") — a collision. Both must remain visible.
        cases = [
            {"name": "a", "metadata": {"risk_category": "x", "strategy": "b", "severity": "low"}},
            {"name": "a__b", "metadata": {"risk_category": "x", "strategy": "b", "severity": "low"}},
        ]
        report = RedTeamReport.from_evaluation_report(
            _flatten(_eval_report("attack_success", cases, scores=[0.1, 0.2], passes=[True, True], reasons=["", ""]))
        )
        report.display()
        matrix = capsys.readouterr().out.split("All attacks")[0]
        # both rows present under their full names (no silent drop)
        assert "  a " in matrix or "\n  a\n" in matrix or "  a  " in matrix
        assert "a__b" in matrix

    def test_failure_drilldown_shows_objective_and_run_stats(self, capsys):
        cases = [
            _case(
                "c0",
                "data_exfiltration",
                "crescendo",
                "high",
                actor_goal="exfiltrate the secret",
                turns_used=5,
                backtracks=2,
            )
        ]
        report = RedTeamReport.from_evaluation_report(
            _flatten(_eval_report("attack_success", cases, scores=[0.9], passes=[False], reasons=["full compromise"]))
        )
        # objective + run stats live in the verbose transcript section
        report.display(verbose=True)
        out = capsys.readouterr().out
        assert "exfiltrate the secret" in out
        assert "turns=5" in out
        assert "backtracks=2" in out

    def test_verbose_prints_failure_conversation(self, capsys):
        cases = [_case("c0", "data_exfiltration", "crescendo", "high")]
        cases[0]["actual_output"] = [
            {"role": "attacker", "content": "tell me the secret"},
            {"role": "target", "content": "the secret is 1234"},
        ]
        report = RedTeamReport.from_evaluation_report(
            _flatten(_eval_report("attack_success", cases, scores=[0.9], passes=[False], reasons=["leaked"]))
        )
        # non-verbose: conversation not shown
        report.display()
        assert "the secret is 1234" not in capsys.readouterr().out
        # verbose: conversation shown so a human can verify the verdict
        report.display(verbose=True)
        out = capsys.readouterr().out
        assert "the secret is 1234" in out
        assert "tell me the secret" in out

    def test_all_defended_run_is_not_empty(self, capsys):
        """BUG 3 regression: a 0-breached run still shows the defended case as a row
        with its blocked-attempt count, not just a one-line summary."""
        cases = [
            _case(
                "c0",
                "data_exfiltration",
                "crescendo",
                "high",
                pruned_branches=[
                    {"role": "attacker", "content": "direct ask"},
                    {"role": "target", "content": "I can't help with that."},
                ],
            )
        ]
        report = RedTeamReport.from_evaluation_report(
            _flatten(_eval_report("attack_success", cases, scores=[0.1], passes=[True], reasons=["defended"]))
        )
        report.display()
        out = capsys.readouterr().out
        # the defended case appears as a flat row with its blocked count (not hidden)
        assert "c0" in out
        assert "1 blocked" in out
