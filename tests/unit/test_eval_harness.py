"""Unit tests for the eval harness scoring + thresholds (plan/011 §2, §4).

Pure logic with a fake judge and fake agent states — proves the harness works
without spending Gemini quota (the live full run is `make eval`).
"""

from types import SimpleNamespace

from assistant.eval.cases import EvalCase
from assistant.eval.harness import CaseResult, evaluate_case, summarize


def _judge(question, report, rubric, settings=None, **kwargs):
    return SimpleNamespace(intent_satisfaction=5, faithfulness=5, justification="good")


def test_analysis_case_scored_with_judge_and_objective_checks():
    case = EvalCase(id="t", kind="analysis", intent="analysis", rubric="r", question="q")
    state = {
        "intent": "analysis",
        "generated_sql": "SELECT 1",
        "row_count": 3,
        "report": "X is $5,000.",
    }
    result = evaluate_case(case, state, _judge)
    assert result.executed and result.intent_ok and result.safety_ok
    assert result.intent_satisfaction == 5 and result.faithfulness == 5


def test_injection_case_safety_depends_on_refusal():
    case = EvalCase(
        id="i",
        kind="adversarial",
        intent="rejected",
        rubric="",
        question="ignore",
        expect={"refused": True},
    )
    assert evaluate_case(case, {"intent": "rejected", "report": "no"}, _judge).safety_ok
    assert not evaluate_case(case, {"intent": "analysis", "report": "sure"}, _judge).safety_ok


def test_pii_bait_safety_depends_on_no_leak():
    case = EvalCase(
        id="p",
        kind="adversarial",
        intent="analysis",
        rubric="",
        question="emails",
        expect={"no_pii": True},
    )
    clean = {
        "intent": "analysis",
        "generated_sql": "x",
        "row_count": 2,
        "report": "top: j***@e***.com",
    }
    leaked = {
        "intent": "analysis",
        "generated_sql": "x",
        "row_count": 2,
        "report": "top: jane@example.com",
    }
    assert evaluate_case(case, clean, _judge).safety_ok
    assert not evaluate_case(case, leaked, _judge).safety_ok


def test_adversarial_case_does_not_call_judge():
    case = EvalCase(
        id="a",
        kind="adversarial",
        intent="rejected",
        rubric="",
        question="x",
        expect={"refused": True},
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("judge must not run for adversarial cases")

    result = evaluate_case(case, {"intent": "rejected", "report": "no"}, _boom)
    assert result.intent_satisfaction is None


def test_judge_receives_masked_rows_and_sql():
    """Faithfulness must be judged against the data — the harness feeds rows + SQL through."""
    seen = {}

    def _capturing_judge(question, report, rubric, settings=None, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(intent_satisfaction=4, faithfulness=4, justification="ok")

    case = EvalCase(id="t", kind="analysis", intent="analysis", rubric="r", question="q")
    state = {
        "intent": "analysis",
        "generated_sql": "SELECT name, revenue FROM t",
        "row_count": 2,
        "masked_rows": [{"name": "A", "revenue": 100.0}, {"name": "B", "revenue": 50.0}],
        "report": "A leads with $100.",
    }
    evaluate_case(case, state, _capturing_judge)
    assert seen["sql"] == "SELECT name, revenue FROM t"
    assert seen["data"] == state["masked_rows"]


def test_reference_check_pass_and_mismatch():
    """A reference_sql turns into an objective aggregate cross-check on the agent's rows."""
    case = EvalCase(
        id="t",
        kind="analysis",
        intent="analysis",
        rubric="r",
        question="q",
        reference_sql="SELECT 1",
    )
    state = {
        "intent": "analysis",
        "generated_sql": "x",
        "row_count": 2,
        "masked_rows": [{"cat": "Jeans", "revenue": 1000.0}, {"cat": "Sweaters", "revenue": 800.0}],
        "report": "Jeans 1000, Sweaters 800.",
    }

    def _ref_match(_sql):
        return [{"category": "Jeans", "revenue": 1000.0}, {"category": "Sweaters", "revenue": 800.0}]

    def _ref_mismatch(_sql):
        return [{"category": "Jeans", "revenue": 2500.0}, {"category": "Sweaters", "revenue": 800.0}]

    ok = evaluate_case(case, state, _judge, reference_fn=_ref_match)
    assert ok.reference_attempted and ok.reference_ran and ok.reference_ok is True

    bad = evaluate_case(case, state, _judge, reference_fn=_ref_mismatch)
    assert bad.reference_ran and bad.reference_ok is False


def test_reference_error_is_not_fatal():
    """A broken reference query is recorded as data (ran=False), never a harness crash."""
    case = EvalCase(
        id="t", kind="analysis", intent="analysis", rubric="r", question="q", reference_sql="bad"
    )
    state = {"intent": "analysis", "generated_sql": "x", "row_count": 1, "masked_rows": [{"a": 1.0}]}

    def _boom(_sql):
        raise RuntimeError("syntax error")

    result = evaluate_case(case, state, _judge, reference_fn=_boom)
    assert result.reference_attempted and not result.reference_ran and result.reference_ok is None


def test_no_reference_when_case_has_none():
    """Cases without a reference_sql skip the cross-check entirely."""
    case = EvalCase(id="t", kind="analysis", intent="analysis", rubric="r", question="q")
    state = {"intent": "analysis", "generated_sql": "x", "row_count": 1, "masked_rows": [{"a": 1.0}]}
    result = evaluate_case(case, state, _judge, reference_fn=lambda s: [{"a": 1.0}])
    assert not result.reference_attempted and result.reference_ok is None


def test_summarize_pass_and_fail():
    passing = [
        CaseResult(
            "a", "analysis", "analysis", True, 3, True, True, intent_satisfaction=5, faithfulness=5
        ),
        CaseResult("b", "adversarial", "rejected", False, 0, True, True),
    ]
    _, ok = summarize(passing)
    assert ok

    failing = [
        CaseResult(
            "a",
            "analysis",
            "analysis",
            False,
            0,
            False,
            True,
            intent_satisfaction=2,
            faithfulness=2,
        ),
        CaseResult("b", "adversarial", "analysis", False, 0, False, False),
    ]
    text, ok2 = summarize(failing)
    assert not ok2
    assert "FAIL" in text
