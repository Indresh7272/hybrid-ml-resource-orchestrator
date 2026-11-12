from time import perf_counter
from fastapi import FastAPI
from app.schemas import WorkloadState, PredictionInput
from app.orchestrator import choose_runtime
from app.model_service import predict_iris
from app.metrics import record, snapshot, reset

app = FastAPI(
    title="Hybrid ML Resource Orchestrator",
    description="Phase-1 local research prototype with a rule-based baseline.",
    version="0.1.0"
)

# Prototype-only values. These are not provider prices.
SERVERLESS_COST_UNITS = 1.0
CONTAINER_COST_UNITS = 0.35
SERVERLESS_COLD_START_MS = 250.0
CONTAINER_BASE_LATENCY_MS = 35.0
EXAMPLE_SLO_MS = 500.0

@app.get("/health")
def health():
    return {"status": "ok", "phase": "local baseline prototype"}

@app.post("/predict")
def predict(payload: PredictionInput):
    result = predict_iris([
        payload.sepal_length, payload.sepal_width,
        payload.petal_length, payload.petal_width
    ])
    return {"service": "sample_iris_classifier", **result}

@app.post("/orchestrate")
def orchestrate(state: WorkloadState):
    decision = choose_runtime(state)
    start = perf_counter()

    # Simulated runtime characteristics; no actual Lambda or container invocation occurs.
    if decision["selected_runtime"] == "serverless":
        simulated_latency = 70.0 + (SERVERLESS_COLD_START_MS * state.cold_start_rate_pct / 100.0)
        cost_units = SERVERLESS_COST_UNITS
        runtime_key = "serverless"
    else:
        simulated_latency = CONTAINER_BASE_LATENCY_MS + max(0.0, state.cpu_utilization_pct - 60.0) * 3.0
        cost_units = CONTAINER_COST_UNITS
        runtime_key = "container"

    api_latency_ms = round((perf_counter() - start) * 1000.0, 3)
    sla_violation = simulated_latency > EXAMPLE_SLO_MS or state.p95_latency_ms > EXAMPLE_SLO_MS
    record(runtime_key, simulated_latency, cost_units, sla_violation)

    return {
        **decision,
        "simulated_latency_ms": round(simulated_latency, 2),
        "orchestration_api_latency_ms": api_latency_ms,
        "simulated_cost_units": cost_units,
        "example_slo_ms": EXAMPLE_SLO_MS,
        "simulated_sla_violation": sla_violation,
        "warning": "Latency and cost are synthetic simulation values, not real cloud measurements."
    }

@app.get("/metrics")
def metrics():
    return snapshot()

@app.post("/metrics/reset")
def metrics_reset():
    reset()
    return {"status": "metrics reset"}
