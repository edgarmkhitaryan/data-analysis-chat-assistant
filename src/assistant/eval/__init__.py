"""Offline evaluation harness: golden set + objective checks + LLM-as-judge (plan/011)."""

from assistant.eval.cases import EvalCase, load_cases
from assistant.eval.correctness import ReferenceCheck, compare_aggregates
from assistant.eval.harness import CaseResult, evaluate_case, run_case, summarize
from assistant.eval.judge import JudgeScore, judge_report

__all__ = [
    "EvalCase",
    "load_cases",
    "CaseResult",
    "evaluate_case",
    "run_case",
    "summarize",
    "JudgeScore",
    "judge_report",
    "ReferenceCheck",
    "compare_aggregates",
]
