from reporter import reporter_node

fake_state = {
    "domain": {"domain": "sales", "confidence": 0.95, "reason": "Detected customer and revenue data"},
    "profile_report": {
        "missing_values": {"unit_price": 4, "customer_id": 12},
        "outliers": {"total_spend": 15},
        "num_rows": 1200,
        "num_columns": 8
    },
    "chart_paths": [],
    "report_style": "executive"
}

result = reporter_node(fake_state)
print("Success! PDF created at:", result["pdf_path"])
print("Generated narrative preview:\n", result["insights"][:250])
