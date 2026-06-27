"""Run the golden eval set and print a scored report (``make eval``).

Exits non-zero if the release thresholds aren't met, so this can gate CI. Use
``--limit N`` to run a subset (handy when the Gemini free-tier quota is tight) and
``--cases PATH`` to point at a different eval set.
"""

import argparse

from rich.console import Console

from assistant.agent.dependencies import AgentDeps
from assistant.agent.graph import build_graph
from assistant.config import get_settings
from assistant.eval.cases import DEFAULT_CASES_PATH, load_cases
from assistant.eval.harness import CaseResult, run_case, summarize
from assistant.eval.judge import judge_report

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden evaluation set.")
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH, help="path to the eval set JSON")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    console.print(f"Running [cyan]{len(cases)}[/] eval cases (real Gemini + BigQuery) …")
    deps = AgentDeps.create(settings)
    graph = build_graph(deps=deps)

    # Objective correctness cross-check: run a case's reference_sql on BigQuery and compare
    # its aggregates to the agent's result (plan/011 §2). Only fires for cases that supply one.
    def reference_fn(sql: str) -> list[dict]:
        return deps.runner.execute_query(sql).rows

    results: list[CaseResult] = []
    for case in cases:
        try:
            results.append(run_case(case, graph, judge_report, settings, reference_fn=reference_fn))
            console.print(f"  · {case.id} done")
        except Exception as exc:  # noqa: BLE001 — a failing case is data, not a crash
            console.print(f"  [red]· {case.id} errored: {exc}[/]")
            results.append(
                CaseResult(
                    id=case.id,
                    kind=case.kind,
                    intent=None,
                    executed=False,
                    rows=0,
                    intent_ok=False,
                    safety_ok=False,
                    error=str(exc),
                )
            )

    report, passed = summarize(results)
    console.print()
    console.print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
