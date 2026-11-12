import os
import joblib
import pandas as pd
from app.schemas import WorkloadState

MODEL_PATH = os.path.join("artifacts", "runtime_policy.joblib")
_model = None

def _get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model

def choose_runtime(state: WorkloadState) -> dict:
    """
    Uses the trained Random Forest when artifacts/runtime_policy.joblib exists.
    Falls back to the original rule-based baseline otherwise.

    The trained model in this phase is trained on synthetic proxy labels only.
    """
    features = {
        "request_rate_per_min": state.request_rate_per_min,
        "predicted_rate_per_min": state.predicted_rate_per_min,
        "cpu_utilization_pct": state.cpu_utilization_pct,
        "memory_utilization_pct": state.memory_utilization_pct,
        "p95_latency_ms": state.p95_latency_ms,
        "cold_start_rate_pct": state.cold_start_rate_pct,
        "workload_type": state.workload_type,
    }

    model = _get_model()
    if model is not None:
        frame = pd.DataFrame([features])
        runtime = str(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(runtime)])
        return {
            "selected_runtime": runtime,
            "policy": "random_forest_synthetic_proxy",
            "explanation": "Runtime selected by the trained Random Forest. Its current training labels are synthetic proxy labels, so this is a prototype decision, not validated cloud optimization.",
            "model_confidence": round(confidence, 4),
            "note": "Validate against measured serverless/container outcomes before claiming improvement."
        }

    projected_growth = state.predicted_rate_per_min / max(state.request_rate_per_min, 1.0)
    reasons = []
    if state.workload_type in {"sustained", "batch"}:
        reasons.append(f"workload type is {state.workload_type}")
    if state.predicted_rate_per_min >= 600:
        reasons.append("predicted request rate is high")
    if state.cpu_utilization_pct >= 70:
        reasons.append("CPU utilization is high")
    if state.p95_latency_ms >= 400:
        reasons.append("P95 latency is approaching/exceeding the example SLO")
    if projected_growth >= 3:
        reasons.append("predicted demand is growing rapidly")

    if reasons:
        runtime = "container"
        explanation = "; ".join(reasons)
    else:
        runtime = "serverless"
        explanation = "low/moderate demand; serverless is selected by the baseline policy"

    return {
        "selected_runtime": runtime,
        "policy": "rule_based_baseline",
        "explanation": explanation,
        "projected_growth_ratio": round(projected_growth, 3),
        "note": "No trained model artifact found. Run `python train_model.py` to train the prototype model."
    }
