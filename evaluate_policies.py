import json, os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "artifacts/synthetic_workloads.csv"
MODEL_PATH = "artifacts/runtime_policy.joblib"
OUT_PATH = "artifacts/policy_comparison.json"
FEATURES = ["request_rate_per_min", "predicted_rate_per_min", "cpu_utilization_pct",
            "memory_utilization_pct", "p95_latency_ms", "cold_start_rate_pct", "workload_type"]

def rule_based_policy(row):
    growth = row["predicted_rate_per_min"] / max(float(row["request_rate_per_min"]), 1.0)
    container = (row["workload_type"] in {"sustained", "batch"}
                 or row["predicted_rate_per_min"] >= 600
                 or row["cpu_utilization_pct"] >= 70
                 or row["p95_latency_ms"] >= 400
                 or growth >= 3)
    return "container" if container else "serverless"

def main():
    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Run `python train_model.py` first; expected artifacts in ./artifacts.")
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    _, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["runtime_label"])
    X, y = test_df[FEATURES], test_df["runtime_label"]
    rf_pred = model.predict(X)
    rule_pred = test_df.apply(rule_based_policy, axis=1)
    result = {
      "dataset": "synthetic workload data; diagnostic only",
      "test_samples": int(len(test_df)),
      "random_forest": {
        "accuracy_vs_synthetic_label": round(float(accuracy_score(y, rf_pred)), 4),
        "macro_f1_vs_synthetic_label": round(float(f1_score(y, rf_pred, average="macro")), 4),
        "classification_report": classification_report(y, rf_pred, output_dict=True, zero_division=0),
        "confusion_matrix_labels": ["container", "serverless"],
        "confusion_matrix": confusion_matrix(y, rf_pred, labels=["container", "serverless"]).tolist()
      },
      "rule_based": {
        "accuracy_vs_synthetic_label": round(float(accuracy_score(y, rule_pred)), 4),
        "macro_f1_vs_synthetic_label": round(float(f1_score(y, rule_pred, average="macro")), 4),
        "classification_report": classification_report(y, rule_pred, output_dict=True, zero_division=0),
        "confusion_matrix_labels": ["container", "serverless"],
        "confusion_matrix": confusion_matrix(y, rule_pred, labels=["container", "serverless"]).tolist()
      },
      "warning": "Synthetic labels only; these metrics do not prove real cloud cost, latency, or SLA improvements."
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved comparison: {OUT_PATH}")
    print(f"Test samples: {len(test_df)}")
    for name in ["random_forest", "rule_based"]:
        print(f"{name}: accuracy={result[name]['accuracy_vs_synthetic_label']}, macro-F1={result[name]['macro_f1_vs_synthetic_label']}")
    print("WARNING: diagnostic metrics on synthetic labels, not real-cloud performance evidence.")

if __name__ == "__main__":
    main()
