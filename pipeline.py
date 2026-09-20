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
import os
from typing import TypedDict

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END

from profiler import profile_csv, load_csv
from domain_classifier import classify_domain
from fix_proposer import propose_fixes, apply_approved_fixes
from reporter import reporter_node

class PipelineState(TypedDict):
    csv_path: str
    original_csv_path: str  

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
    """
    Agent 2 - Data Visualizer

    Reads the cleaned CSV produced by Agent 1, determines suitable
    visualizations based on the classified domain, generates PNG charts,
    and stores their paths in state["chart_paths"].
    """

    print("\n[Agent 2 - Visualizer] Generating charts...")

    
    # 1. Load cleaned data
    
    csv_path = state["csv_path"]
    df = pd.read_csv(csv_path)

    domain = state.get("domain", {}).get("domain", "generic")

    print(f"  -> Reading cleaned data: {csv_path}")
    print(f"  -> Domain: {domain}")

    
    # 2. Create charts folder
    
    os.makedirs("charts", exist_ok=True)

    chart_paths = []

    
    # 3. Identify column types

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Try to identify date/time columns
    date_cols = []

    for col in df.columns:
        if col in numeric_cols:
            continue

        converted = pd.to_datetime(df[col], errors="coerce")

        # Consider it a date column if most non-null values
        # can be converted successfully.
        non_null_count = df[col].notna().sum()

        if non_null_count > 0:
            valid_dates = converted.notna().sum()

            if valid_dates / non_null_count >= 0.7:
                date_cols.append(col)

    
    # 4. SALES domain
    
    if domain == "sales":

        # Revenue / sales trend
        if date_cols and numeric_cols:
            date_col = date_cols[0]
            preferred_value_names = [
               "purchase_amount",
               "sales",
               "revenue",
               "amount",
               "total_sales",
               "total_amount"
            ]
            value_col = next(
                 (
                  col for col in numeric_cols
                  if col.lower() in preferred_value_names
                  or any(name in col.lower() for name in preferred_value_names)
                  ),
                  numeric_cols[0]
            )
            

            temp = df[[date_col, value_col]].copy()
            temp[date_col] = pd.to_datetime(
                temp[date_col], errors="coerce"
            )
            temp = temp.dropna()

            if not temp.empty:
                trend = (
                    temp.groupby(date_col)[value_col]
                    .sum()
                    .reset_index()
                    .sort_values(date_col)
                )

                fig = px.line(
                    trend,
                    x=date_col,
                    y=value_col,
                    title=f"{value_col} Trend Over Time"
                )

                path = f"charts/{value_col}_trend.png"
                fig.write_image(path)
                chart_paths.append(path)

        # Top categories
        non_date_categorical_cols = [
            col for col in categorical_cols
            if col not in date_cols
            and "date" not in col.lower()
            and "time" not in col.lower()
        ]
        if non_date_categorical_cols and numeric_cols:
            category_col = non_date_categorical_cols[0]
            

            value_col = next(
                (
                     col for col in numeric_cols
                     if col.lower() in preferred_value_names
                     or any(name in col.lower() for name in preferred_value_names)
                ),
                numeric_cols[0]
            )

            category_data = (
                df.groupby(category_col)[value_col]
                .sum()
                .reset_index()
                .sort_values(value_col, ascending=False)
                .head(10)
            )

            if not category_data.empty:
                fig = px.bar(
                    category_data,
                    x=category_col,
                    y=value_col,
                    title=f"Top {category_col} by {value_col}"
                )

                path = f"charts/top_{category_col}.png"
                fig.write_image(path)
                chart_paths.append(path)

    
    # 5. TIME-SERIES domain
    
    elif domain == "time-series":

        if date_cols and numeric_cols:
            date_col = date_cols[0]
            value_col = numeric_cols[0]

            temp = df[[date_col, value_col]].copy()
            temp[date_col] = pd.to_datetime(
                temp[date_col], errors="coerce"
            )
            temp = temp.dropna()

            if not temp.empty:
                temp = temp.sort_values(date_col)

                fig = px.line(
                    temp,
                    x=date_col,
                    y=value_col,
                    title=f"{value_col} Over Time"
                )

                path = f"charts/{value_col}_time_series.png"
                fig.write_image(path)
                chart_paths.append(path)

        elif numeric_cols:
            value_col = numeric_cols[0]

            fig = px.line(
                df,
                y=value_col,
                title=f"{value_col} Time Series"
            )

            path = f"charts/{value_col}_time_series.png"
            fig.write_image(path)
            chart_paths.append(path)

    
    # 6. SURVEY domain
    
    elif domain == "survey":

        # Bar chart for categorical responses
        if categorical_cols:
            category_col = categorical_cols[0]

            counts = (
                df[category_col]
                .value_counts()
                .reset_index()
            )

            counts.columns = [category_col, "count"]

            fig = px.bar(
                counts,
                x=category_col,
                y="count",
                title=f"{category_col} Response Counts"
            )

            path = f"charts/{category_col}_responses.png"
            fig.write_image(path)
            chart_paths.append(path)

        # Distribution of first numeric survey variable
        if numeric_cols:
            value_col = numeric_cols[0]

            fig = px.histogram(
                df,
                x=value_col,
                title=f"{value_col} Distribution"
            )

            path = f"charts/{value_col}_distribution.png"
            fig.write_image(path)
            chart_paths.append(path)

    
    # 7. HEALTHCARE domain
    
    elif domain == "healthcare":

        # Distribution of numeric health/lab variable
        if numeric_cols:
            value_col = numeric_cols[0]

            fig = px.histogram(
                df,
                x=value_col,
                title=f"{value_col} Distribution"
            )

            path = f"charts/{value_col}_distribution.png"
            fig.write_image(path)
            chart_paths.append(path)

        # Age-grouped comparison if an age column exists
        age_col = None

        for col in df.columns:
            if col.lower() == "age" or "age" in col.lower():
                age_col = col
                break

        if age_col and len(numeric_cols) >= 2:

            value_col = next(
                (col for col in numeric_cols if col != age_col),
                None
            )

            if value_col:
                fig = px.scatter(
                    df,
                    x=age_col,
                    y=value_col,
                    title=f"{value_col} by Age"
                )

                path = f"charts/{value_col}_by_age.png"
                fig.write_image(path)
                chart_paths.append(path)

    
    # 8. GENERIC fallback
    
    else:

        # Histogram
        if numeric_cols:
            value_col = numeric_cols[0]

            fig = px.histogram(
                df,
                x=value_col,
                title=f"{value_col} Distribution"
            )

            path = f"charts/{value_col}_distribution.png"
            fig.write_image(path)
            chart_paths.append(path)

        # Boxplot
        if len(numeric_cols) >= 1:
            value_col = numeric_cols[0]

            fig = px.box(
                df,
                y=value_col,
                title=f"{value_col} Boxplot"
            )

            path = f"charts/{value_col}_boxplot.png"
            fig.write_image(path)
            chart_paths.append(path)

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            correlation = df[numeric_cols].corr()

            fig = px.imshow(
                correlation,
                text_auto=True,
                title="Correlation Heatmap"
            )

            path = "charts/correlation_heatmap.png"
            fig.write_image(path)
            chart_paths.append(path)


    # 9. Final fallback if no suitable chart was generated
    
    if not chart_paths and numeric_cols:

        value_col = numeric_cols[0]

        fig = px.histogram(
            df,
            x=value_col,
            title=f"{value_col} Distribution"
        )

        path = f"charts/{value_col}_distribution.png"
        fig.write_image(path)
        chart_paths.append(path)

    
    state["chart_paths"] = chart_paths

    print(f"  -> Generated {len(chart_paths)} chart(s)")

    for path in chart_paths:
        print(f"     {path}")

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
        "report_style": "executive",
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
    print(f"Chart paths: {final_state['chart_paths']}")
    print(f"PDF path (placeholder): {final_state['pdf_path']}")
