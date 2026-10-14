"""Tests for strands_evals.analysis.failure_cohorts module."""

from strands_evals.analysis import CohortAnalysis, FailureCohort, analyze_failure_cohorts, print_cohort_summary
from strands_evals.types.evaluation_report import EvaluationReport


class TestFailureCohort:
    """Tests for the FailureCohort model."""

    def test_basic_construction(self):
        cohort = FailureCohort(
            evaluator_name="Faithfulness",
            failed_case_indices=[0, 2, 5],
            failed_case_names=["case_0", "case_2", "case_5"],
            count=3,
        )
        assert cohort.evaluator_name == "Faithfulness"
        assert cohort.failed_case_indices == [0, 2, 5]
        assert cohort.failed_case_names == ["case_0", "case_2", "case_5"]
        assert cohort.count == 3

    def test_is_systemic_with_multiple_failures(self):
        cohort = FailureCohort(
            evaluator_name="Correctness",
            failed_case_indices=[1, 3],
            failed_case_names=["a", "b"],
            count=2,
        )
        assert cohort.is_systemic is True

    def test_is_systemic_with_single_failure(self):
        cohort = FailureCohort(
            evaluator_name="Harmfulness",
            failed_case_indices=[4],
            failed_case_names=["edge_case"],
            count=1,
        )
        assert cohort.is_systemic is False

    def test_is_systemic_boundary(self):
        cohort = FailureCohort(
            evaluator_name="X",
            failed_case_indices=[0, 1],
            failed_case_names=["a", "b"],
            count=2,
        )
        assert cohort.is_systemic is True

    def test_serialization_roundtrip(self):
        cohort = FailureCohort(
            evaluator_name="Faithfulness",
            failed_case_indices=[0, 2],
            failed_case_names=["case_0", "case_2"],
            count=2,
        )
        data = cohort.model_dump()
        restored = FailureCohort.model_validate(data)
        assert restored == cohort


class TestCohortAnalysis:
    """Tests for the CohortAnalysis model."""

    def test_systemic_cohorts_property(self):
        analysis = CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="Faithfulness",
                    failed_case_indices=[0, 1, 2],
                    failed_case_names=["a", "b", "c"],
                    count=3,
                ),
                FailureCohort(
                    evaluator_name="Harmfulness",
                    failed_case_indices=[5],
                    failed_case_names=["e"],
                    count=1,
                ),
            ],
            total_failures=4,
            total_cases=10,
        )
        systemic = analysis.systemic_cohorts
        assert len(systemic) == 1
        assert systemic[0].evaluator_name == "Faithfulness"

    def test_one_off_failures_property(self):
        analysis = CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="Faithfulness",
                    failed_case_indices=[0, 1, 2],
                    failed_case_names=["a", "b", "c"],
                    count=3,
                ),
                FailureCohort(
                    evaluator_name="Harmfulness",
                    failed_case_indices=[5],
                    failed_case_names=["e"],
                    count=1,
                ),
            ],
            total_failures=4,
            total_cases=10,
        )
        one_offs = analysis.one_off_failures
        assert len(one_offs) == 1
        assert one_offs[0].evaluator_name == "Harmfulness"

    def test_empty_cohorts(self):
        analysis = CohortAnalysis(cohorts=[], total_failures=0, total_cases=5)
        assert analysis.systemic_cohorts == []
        assert analysis.one_off_failures == []

    def test_serialization_roundtrip(self):
        analysis = CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="Correctness",
                    failed_case_indices=[1],
                    failed_case_names=["x"],
                    count=1,
                ),
            ],
            total_failures=1,
            total_cases=3,
        )
        data = analysis.model_dump()
        restored = CohortAnalysis.model_validate(data)
        assert restored == analysis


class TestAnalyzeFailureCohorts:
    """Tests for the analyze_failure_cohorts function."""

    def test_all_passing(self):
        report = EvaluationReport(
            overall_score=1.0,
            scores=[1.0, 1.0, 1.0],
            cases=[
                {"name": "case-1", "evaluator": "Correctness"},
                {"name": "case-2", "evaluator": "Correctness"},
                {"name": "case-3", "evaluator": "Faithfulness"},
            ],
            test_passes=[True, True, True],
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis == CohortAnalysis(cohorts=[], total_failures=0, total_cases=3)

    def test_all_failing_single_evaluator(self):
        report = EvaluationReport(
            overall_score=0.0,
            scores=[0.0, 0.0, 0.0],
            cases=[
                {"name": "case-1", "evaluator": "Faithfulness"},
                {"name": "case-2", "evaluator": "Faithfulness"},
                {"name": "case-3", "evaluator": "Faithfulness"},
            ],
            test_passes=[False, False, False],
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis == CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="Faithfulness",
                    failed_case_indices=[0, 1, 2],
                    failed_case_names=["case-1", "case-2", "case-3"],
                    count=3,
                ),
            ],
            total_failures=3,
            total_cases=3,
        )

    def test_multiple_evaluators_sorted_by_count(self):
        report = EvaluationReport(
            overall_score=0.5,
            scores=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            cases=[
                {"name": "c1", "evaluator": "Correctness"},
                {"name": "c2", "evaluator": "Faithfulness"},
                {"name": "c3", "evaluator": "Faithfulness"},
                {"name": "c4", "evaluator": "Faithfulness"},
                {"name": "c5", "evaluator": "Correctness"},
                {"name": "c6", "evaluator": "Faithfulness"},
            ],
            test_passes=[False, False, False, False, True, True],
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis == CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="Faithfulness",
                    failed_case_indices=[1, 2, 3],
                    failed_case_names=["c2", "c3", "c4"],
                    count=3,
                ),
                FailureCohort(
                    evaluator_name="Correctness",
                    failed_case_indices=[0],
                    failed_case_names=["c1"],
                    count=1,
                ),
            ],
            total_failures=4,
            total_cases=6,
        )

    def test_alphabetical_tiebreaker(self):
        report = EvaluationReport(
            overall_score=0.0,
            scores=[0.0, 0.0],
            cases=[
                {"name": "c1", "evaluator": "Zebra"},
                {"name": "c2", "evaluator": "Alpha"},
            ],
            test_passes=[False, False],
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis == CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="Alpha",
                    failed_case_indices=[1],
                    failed_case_names=["c2"],
                    count=1,
                ),
                FailureCohort(
                    evaluator_name="Zebra",
                    failed_case_indices=[0],
                    failed_case_names=["c1"],
                    count=1,
                ),
            ],
            total_failures=2,
            total_cases=2,
        )

    def test_missing_evaluator_key_defaults_to_unknown(self):
        report = EvaluationReport(
            overall_score=0.0,
            scores=[0.0, 0.0],
            cases=[
                {"name": "c1"},
                {"name": "c2"},
            ],
            test_passes=[False, False],
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis == CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="unknown",
                    failed_case_indices=[0, 1],
                    failed_case_names=["c1", "c2"],
                    count=2,
                ),
            ],
            total_failures=2,
            total_cases=2,
        )

    def test_missing_name_key_defaults_to_case_index(self):
        report = EvaluationReport(
            overall_score=0.0,
            scores=[0.0, 0.0],
            cases=[
                {"evaluator": "X"},
                {"evaluator": "X"},
            ],
            test_passes=[False, False],
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis == CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="X",
                    failed_case_indices=[0, 1],
                    failed_case_names=["case_0", "case_1"],
                    count=2,
                ),
            ],
            total_failures=2,
            total_cases=2,
        )

    def test_mixed_pass_fail(self):
        report = EvaluationReport(
            overall_score=0.6,
            scores=[1.0, 0.0, 1.0, 0.0, 0.0],
            cases=[
                {"name": "pass1", "evaluator": "Correctness"},
                {"name": "fail1", "evaluator": "Correctness"},
                {"name": "pass2", "evaluator": "Faithfulness"},
                {"name": "fail2", "evaluator": "Faithfulness"},
                {"name": "fail3", "evaluator": "Harmfulness"},
            ],
            test_passes=[True, False, True, False, False],
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis == CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="Correctness",
                    failed_case_indices=[1],
                    failed_case_names=["fail1"],
                    count=1,
                ),
                FailureCohort(
                    evaluator_name="Faithfulness",
                    failed_case_indices=[3],
                    failed_case_names=["fail2"],
                    count=1,
                ),
                FailureCohort(
                    evaluator_name="Harmfulness",
                    failed_case_indices=[4],
                    failed_case_names=["fail3"],
                    count=1,
                ),
            ],
            total_failures=3,
            total_cases=5,
        )

    def test_empty_report(self):
        report = EvaluationReport(
            overall_score=0.0,
            scores=[],
            cases=[],
            test_passes=[],
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis == CohortAnalysis(cohorts=[], total_failures=0, total_cases=0)

    def test_indices_reflect_original_position(self):
        report = EvaluationReport(
            overall_score=0.5,
            scores=[1.0, 0.0, 1.0, 1.0, 0.0],
            cases=[
                {"name": "a", "evaluator": "E1"},
                {"name": "b", "evaluator": "E1"},
                {"name": "c", "evaluator": "E1"},
                {"name": "d", "evaluator": "E1"},
                {"name": "e", "evaluator": "E1"},
            ],
            test_passes=[True, False, True, True, False],
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis == CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="E1",
                    failed_case_indices=[1, 4],
                    failed_case_names=["b", "e"],
                    count=2,
                ),
            ],
            total_failures=2,
            total_cases=5,
        )

    def test_works_with_report_from_file(self, tmp_path):
        report = EvaluationReport(
            overall_score=0.5,
            scores=[1.0, 0.0, 0.0, 1.0],
            cases=[
                {"name": "c1", "evaluator": "Eval1"},
                {"name": "c2", "evaluator": "Eval1"},
                {"name": "c3", "evaluator": "Eval2"},
                {"name": "c4", "evaluator": "Eval2"},
            ],
            test_passes=[True, False, False, True],
        )
        filepath = str(tmp_path / "report.json")
        report.to_file(filepath)

        loaded = EvaluationReport.from_file(filepath)
        analysis = analyze_failure_cohorts(loaded)
        assert analysis.total_failures == 2
        assert len(analysis.cohorts) == 2

    def test_works_with_flattened_report(self):
        report1 = EvaluationReport(
            overall_score=0.5,
            scores=[1.0, 0.0],
            cases=[
                {"name": "c1", "evaluator": "Correctness"},
                {"name": "c2", "evaluator": "Correctness"},
            ],
            test_passes=[True, False],
        )
        report2 = EvaluationReport(
            overall_score=0.0,
            scores=[0.0, 0.0],
            cases=[
                {"name": "c1", "evaluator": "Faithfulness"},
                {"name": "c2", "evaluator": "Faithfulness"},
            ],
            test_passes=[False, False],
        )
        flattened = EvaluationReport.flatten([report1, report2])
        analysis = analyze_failure_cohorts(flattened)
        assert analysis.total_failures == 3
        assert analysis.total_cases == 4
        assert analysis.cohorts[0].evaluator_name == "Faithfulness"
        assert analysis.cohorts[0].count == 2
        assert analysis.cohorts[1].evaluator_name == "Correctness"
        assert analysis.cohorts[1].count == 1

    def test_large_cohort(self):
        n = 100
        cases = [{"name": f"case_{i}", "evaluator": "BigEval"} for i in range(n)]
        report = EvaluationReport(
            overall_score=0.0,
            scores=[0.0] * n,
            cases=cases,
            test_passes=[False] * n,
        )
        analysis = analyze_failure_cohorts(report)
        assert analysis.total_failures == n
        assert len(analysis.cohorts) == 1
        assert analysis.cohorts[0].count == n
        assert len(analysis.cohorts[0].failed_case_indices) == n


class TestPrintCohortSummary:
    """Tests for the print_cohort_summary display helper."""

    def test_prints_table_without_error(self, capsys):
        analysis = CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="Faithfulness",
                    failed_case_indices=[0, 1, 2],
                    failed_case_names=["a", "b", "c"],
                    count=3,
                ),
                FailureCohort(
                    evaluator_name="Correctness",
                    failed_case_indices=[5],
                    failed_case_names=["f"],
                    count=1,
                ),
            ],
            total_failures=4,
            total_cases=10,
        )
        # Should not raise
        print_cohort_summary(analysis)

    def test_truncates_case_names_beyond_five(self, capsys):
        analysis = CohortAnalysis(
            cohorts=[
                FailureCohort(
                    evaluator_name="BigEval",
                    failed_case_indices=list(range(8)),
                    failed_case_names=[f"case_{i}" for i in range(8)],
                    count=8,
                ),
            ],
            total_failures=8,
            total_cases=20,
        )
        print_cohort_summary(analysis)
        captured = capsys.readouterr()
        assert "+3 more" in captured.out

    def test_empty_analysis(self, capsys):
        analysis = CohortAnalysis(cohorts=[], total_failures=0, total_cases=5)
        # Should not raise on empty
        print_cohort_summary(analysis)


class TestModuleImports:
    """Tests that the module is accessible from the top-level package."""

    def test_import_from_analysis_submodule(self):
        from strands_evals.analysis import analyze_failure_cohorts as fn

        assert callable(fn)

    def test_import_from_top_level(self):
        import strands_evals

        assert hasattr(strands_evals, "analysis")
        assert hasattr(strands_evals.analysis, "analyze_failure_cohorts")
        assert hasattr(strands_evals.analysis, "FailureCohort")
        assert hasattr(strands_evals.analysis, "CohortAnalysis")
        assert hasattr(strands_evals.analysis, "print_cohort_summary")
