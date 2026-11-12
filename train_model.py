"""
Phase 2 training pipeline for the Hybrid ML Resource Orchestrator.

IMPORTANT:
This generates synthetic workload observations and proxy labels using explicit
assumptions. It validates the ML pipeline only. It is NOT evidence that the
learned policy improves real cloud cost or latency. Replace the synthetic
labeling function with measured outcomes before reporting research results.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

SEED = 42
N = 6000
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "runtime_policy.joblib")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "training_metrics.json")
DATA_PATH = os.path.join(ARTIFACT_DIR, "synthetic_workloads.csv")

def make_dataset(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    workload_types = np.array(["bursty", "steady", "sustained", "batch"])
    workload_type = rng.choice(workload_types, size=n, p=[0.30, 0.30, 0.30, 0.10])
    request_rate = np.clip(rng.lognormal(mean=4.7, sigma=1.2, size=n), 0, 5000)
    growth = rng.lognormal(mean=0.0, sigma=0.65, size=n)
    predicted_rate = np.clip(request_rate * growth, 0, 8000)
    cpu = np.clip(8 + request_rate / 22 + rng.normal(0, 13, n), 1, 100)
    memory = np.clip(10 + request_rate / 35 + rng.normal(0, 12, n), 1, 100)
    p95 = np.clip(45 + cpu * 2.5 + rng.normal(0, 65, n), 20, 1500)
    cold_start = np.clip(rng.beta(1.5, 8, n) * 100, 0, 100)

    # Illustrative synthetic proxy costs/latencies. Units are arbitrary.
    # These are deliberately exposed so assumptions can be changed and tested.
    burstiness_bonus = np.isin(workload_type, ["bursty"]).astype(float)
    sustained_bonus = np.isin(workload_type, ["sustained", "batch"]).astype(float)

    serverless_cost = 0.75 + 0.0018 * predicted_rate + 0.012 * cold_start
    serverless_latency = 65 + 2.8 * cold_start + 0.035 * predicted_rate + 0.04 * p95
    container_cost = 1.05 + 0.00055 * predicted_rate + 0.006 * cpu
    container_latency = 45 + 0.9 * cpu + 0.012 * predicted_rate
    # Synthetic SLO penalty: proxy SLO is 500 ms.
    serverless_score = serverless_cost + 0.002 * serverless_latency + 1.5 * (serverless_latency > 500)
    container_score = container_cost + 0.002 * container_latency + 1.5 * (container_latency > 500)

    # Make serverless slightly more attractive for bursty, low-load traffic,
    # and containers more attractive for sustained/batch traffic.
    serverless_score -= 0.12 * burstiness_bonus
    container_score -= 0.18 * sustained_bonus

    label = np.where(serverless_score <= container_score, "serverless", "container")

    df = pd.DataFrame({
        "request_rate_per_min": request_rate,
        "predicted_rate_per_min": predicted_rate,
        "cpu_utilization_pct": cpu,
        "memory_utilization_pct": memory,
        "p95_latency_ms": p95,
        "cold_start_rate_pct": cold_start,
        "workload_type": workload_type,
        "runtime_label": label,
    })
    return df

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = make_dataset()
    df.to_csv(DATA_PATH, index=False)

    features = [
        "request_rate_per_min", "predicted_rate_per_min",
        "cpu_utilization_pct", "memory_utilization_pct",
        "p95_latency_ms", "cold_start_rate_pct", "workload_type"
    ]
    numeric = features[:-1]
    categorical = ["workload_type"]

    X = df[features]
    y = df["runtime_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=14, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "dataset": "synthetic proxy workload dataset",
        "samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix_labels": ["container", "serverless"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["container", "serverless"]).tolist(),
        "warning": "Synthetic-label metrics only measure how well the model imitates the synthetic proxy labels. They are not real-world performance evidence."
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved dataset: {DATA_PATH}")
    print(f"Saved model:   {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Samples: {len(df)} | Accuracy: {metrics['accuracy']} | Macro-F1: {metrics['macro_f1']}")
    print("\nWARNING: These scores are from synthetic proxy labels, not real cloud measurements.")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    main()
