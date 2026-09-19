import json
from llm_client import call_llm


def build_fix_prompt(profile_report: dict) -> str:
    return f"""You are a data cleaning assistant. Below is a structured data quality report
generated from a dataset. For EACH issue found, propose ONE specific, actionable fix.

Data quality report:
{json.dumps(profile_report, indent=2, default=str)}

Rules for your suggestions:
- For missing values: recommend median imputation if the column is likely skewed,
  mean imputation if likely normally distributed, or "drop rows" if missing % is very low (<2%).
  State your reasoning briefly.
- For dtype inconsistencies: recommend either fixing the bad values (if a clear correction
  exists) or converting them to null/NaN if no correction is obvious.
- For outliers: recommend "cap at boundary", "remove row", or "keep and flag" (if it might
  be a legitimate rare event rather than an error) - use judgment based on how extreme the value is.

Respond with ONLY valid JSON in this exact format, nothing else:
{{
  "fixes": [
    {{
      "column": "<column name>",
      "issue_type": "<missing_value | dtype_inconsistency | outlier>",
      "suggested_action": "<short specific action, e.g. 'impute with median'>",
      "reason": "<one short sentence explaining why>"
    }}
  ]
}}
"""


def propose_fixes(profile_report: dict) -> dict:
    """Main function Agent 1 calls to get fix suggestions.
    Returns a dict with a 'fixes' list - each fix is shown to the user for
    approval before anything is actually applied to the data."""

    has_issues = any([
        profile_report.get("missing_values"),
        profile_report.get("dtype_inconsistencies"),
        profile_report.get("outliers")
    ])
    if not has_issues:
        return {"fixes": []}

    prompt = build_fix_prompt(profile_report)
    raw_response = call_llm(prompt)

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        result = json.loads(raw_response[start:end])

    return result


def apply_approved_fixes(df, approved_fixes: list[dict]):
    """Applies user-approved fixes to the dataframe. This runs AFTER the user
    confirms via the Streamlit UI - nothing here happens automatically.

    IMPORTANT: suggested_action is free-text from an LLM and its exact wording
    can vary between runs, even at temperature=0. We primarily branch on the
    structured `issue_type` field (a fixed enum from our own prompt schema)
    and use suggested_action only to pick BETWEEN known options within that
    type. Anything unrecognized is logged as a warning, never silently
    skipped, so a data quality issue never disappears without a trace.
    """
    import pandas as pd
    import numpy as np

    log = []

    for fix in approved_fixes:
        col = fix["column"]
        action = fix.get("suggested_action", "").lower()
        issue_type = fix.get("issue_type", "")

        if col not in df.columns:
            log.append(f"WARNING: column '{col}' not found in dataframe, skipped")
            continue

        handled = False

        # --- Missing values ---
        if issue_type == "missing_value":
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            if "drop" in action:
                df = df.dropna(subset=[col])
                log.append(f"{col}: dropped rows with missing values")
            elif is_numeric and "mean" in action:
                df[col] = df[col].fillna(df[col].mean())
                log.append(f"{col}: filled missing values with mean")
            elif is_numeric:
                df[col] = df[col].fillna(df[col].median())
                log.append(f"{col}: filled missing values with median"
                            + ("" if "median" in action else " (defaulted - unrecognized phrasing)"))
            else:
                mode_vals = df[col].mode(dropna=True)
                if len(mode_vals) > 0:
                    df[col] = df[col].fillna(mode_vals.iloc[0])
                    log.append(f"{col}: filled missing values with most frequent "
                               f"value (non-numeric column, mean/median not applicable)")
                else:
                    df = df.dropna(subset=[col])
                    log.append(f"{col}: dropped rows with missing values "
                               f"(non-numeric column with no dominant value to impute)")
            handled = True

        # --- Dtype inconsistencies ---
        elif issue_type == "dtype_inconsistency":
            converted_numeric = pd.to_numeric(df[col], errors="coerce")
            converted_date = pd.to_datetime(df[col], errors="coerce")
            if converted_numeric.notnull().sum() >= converted_date.notnull().sum():
                df[col] = converted_numeric
            else:
                df[col] = converted_date
            log.append(f"{col}: converted invalid entries to null")
            handled = True

        # --- Outliers ---
        elif issue_type == "outlier":
            if not pd.api.types.is_numeric_dtype(df[col]):
                log.append(f"WARNING: '{col}' flagged as outlier but is not numeric "
                           f"(likely lost its dtype in a CSV round-trip) - skipped safely")
                handled = True
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            if "remove" in action:
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                log.append(f"{col}: removed outlier rows")
            elif "flag" in action or "keep" in action:
                log.append(f"{col}: kept as-is, flagged as legitimate rare value")
            else:
                df[col] = df[col].clip(lower=lower, upper=upper)
                log.append(f"{col}: capped outliers to [{lower:.2f}, {upper:.2f}]"
                            + ("" if "cap" in action else " (defaulted - unrecognized phrasing)"))
            handled = True

        if not handled:
            log.append(f"WARNING: unrecognized issue_type '{issue_type}' for column "
                       f"'{col}', action '{fix.get('suggested_action')}' - fix NOT applied")

    return df, log
