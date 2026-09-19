"""
pipeline.py

The real LangGraph pipeline for Agent 1 (Data Profiler), wired with your
actual profiling, classification, and fix logic - no more placeholders
for this agent.

Agents 2, 3, 4 are still placeholders since your teammates haven't built
them yet - once they do, swap their placeholder functions the same way
we did for Agent 1.

Usage:
    export GROQ_API_KEY=your_real_key_here
    python pipeline.py data/sample.csv
"""

import sys
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END

from profiler import profile_csv, load_csv
from domain_classifier import classify_domain
from fix_proposer import propose_fixes, apply_approved_fixes
from reporter import reporter_node

class PipelineState(TypedDict):
    csv_path: str
    original_csv_path: str  # kept fixed across retry loops, so the cleaned
                             # filename never stacks up (_cleaned_cleaned...)

    schema: dict
    domain: dict
    profile_report: dict
    proposed_fixes: list

    approved_fixes: list

    is_clean: bool
    recheck_count: int

    chart_paths: list

    report_style: str
    insights: str
    pdf_path: str

    chat_history: list


def profile_node(state: PipelineState) -> PipelineState:
    print("\n[Agent 1a] Profiling CSV...")

    report = profile_csv(state["csv_path"])
    df = load_csv(state["csv_path"])
    sample_rows = df.head(3).to_dict(orient="records")

    domain_result = classify_domain(report["schema"], sample_rows)
    fixes_result = propose_fixes(report)

    state["schema"] = report["schema"]
    state["profile_report"] = report
    state["domain"] = domain_result
    state["proposed_fixes"] = fixes_result.get("fixes", [])

    print(f"  -> Domain: {domain_result['domain']} (confidence: {domain_result.get('confidence')})")
    print(f"  -> {len(state['proposed_fixes'])} fix(es) proposed")

    return state


def human_approval_node(state: PipelineState) -> PipelineState:
    print("\n[Checkpoint] Simulating user approval (auto-approving all fixes for now)...")
    state["approved_fixes"] = state["proposed_fixes"]
    return state


def apply_fixes_node(state: PipelineState) -> PipelineState:
    print("\n[Agent 1b] Applying approved fixes and re-checking...")

    df = load_csv(state["csv_path"])
    cleaned_df, log = apply_approved_fixes(df, state["approved_fixes"])

    for entry in log:
        print("  -", entry)

    # Always derive the cleaned filename from the ORIGINAL path, not the
    # current one, so repeated retry loops don't stack "_cleaned_cleaned...".
    cleaned_path = state["original_csv_path"].replace(".csv", "_cleaned.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    state["csv_path"] = cleaned_path

    recheck_report = profile_csv(cleaned_path)
    remaining_issues = any([
        recheck_report.get("missing_values"),
        recheck_report.get("dtype_inconsistencies"),
        recheck_report.get("outliers")
    ])

    state["is_clean"] = not remaining_issues
    state["recheck_count"] = state.get("recheck_count", 0) + 1

    print(f"  -> Re-check result: {'CLEAN' if state['is_clean'] else 'ISSUES REMAIN'} "
          f"(attempt {state['recheck_count']})")

    return state


def route_after_recheck(state: PipelineState) -> str:
    MAX_RETRIES = 3
    if state["is_clean"]:
        return "proceed"
    elif state["recheck_count"] >= MAX_RETRIES:
        print("  -> Max retries reached, proceeding anyway.")
        return "proceed"
    else:
        return "retry"


def visualizer_node(state: PipelineState) -> PipelineState:
    print("\n[Agent 2 - PLACEHOLDER] Generating charts...")
    state["chart_paths"] = ["placeholder_chart.png"]
    return state


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("profile", profile_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("apply_fixes", apply_fixes_node)
    graph.add_node("visualize", visualizer_node)
    graph.add_node("report", reporter_node)

    graph.set_entry_point("profile")
    graph.add_edge("profile", "human_approval")
    graph.add_edge("human_approval", "apply_fixes")

    graph.add_conditional_edges(
        "apply_fixes",
        route_after_recheck,
        {
            "retry": "profile",
            "proceed": "visualize"
        }
    )

    graph.add_edge("visualize", "report")
    graph.add_edge("report", END)

    return graph.compile()


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    report_style = sys.argv[2] if len(sys.argv) > 2 else "executive"

    pipeline = build_pipeline()

    initial_state: PipelineState = {
        "csv_path": csv_path,
        "original_csv_path": csv_path,
        "schema": {},
        "domain": {},
        "profile_report": {},
        "proposed_fixes": [],
        "approved_fixes": [],
        "is_clean": False,
        "recheck_count": 0,
        "chart_paths": [],
        "report_style": report_style,
        "insights": "",
        "pdf_path": "",
        "chat_history": []
    }

    final_state = pipeline.invoke(initial_state)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Domain: {final_state['domain'].get('domain')}")
    print(f"Cleaned CSV saved at: {final_state['csv_path']}")
    print(f"Chart paths (placeholder): {final_state['chart_paths']}")
    print(f"PDF path (placeholder): {final_state['pdf_path']}")
