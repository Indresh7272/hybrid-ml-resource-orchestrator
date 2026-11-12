
import json
import os
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException

from app.schemas import (
    WorkloadState,
    PredictionInput,
    HybridInferenceRequest,
)
from app.orchestrator import choose_runtime
from app.model_service import predict_iris
from app.metrics import record, record_actual, snapshot, reset


app = FastAPI(
    title="Hybrid ML Resource Orchestrator",
    description="Hybrid serverless and containerized ML orchestration",
    version="0.5.0",
)

# Existing prototype-only simulation values
SERVERLESS_COST_UNITS = 1.0
CONTAINER_COST_UNITS = 0.35
SERVERLESS_COLD_START_MS = 250.0
CONTAINER_BASE_LATENCY_MS = 35.0
EXAMPLE_SLO_MS = 500.0

# Example execution SLO for measured inference requests
ACTUAL_SLO_MS = 500.0

# Permanent Modal inference endpoint, configured through Render
MODAL_INFERENCE_URL = os.getenv(
    "MODAL_INFERENCE_URL", ""
).rstrip("/")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "phase": "hybrid execution integration",
    }


@app.post("/predict")
def predict(payload: PredictionInput):
    result = predict_iris([
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ])

    return {
        "service": "sample_iris_classifier",
        "runtime": "container",
        **result,
    }


def call_modal_inference(payload: PredictionInput) -> dict:
    """Send an Iris prediction request to the Modal serverless endpoint."""

    if not MODAL_INFERENCE_URL:
        raise HTTPException(
            status_code=503,
            detail="MODAL_INFERENCE_URL is not configured.",
        )

    url = f"{MODAL_INFERENCE_URL}/predict"

    request_body = json.dumps(
        payload.model_dump()
    ).encode("utf-8")

    request = Request(
        url=url,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Modal returned HTTP {response.status}"
            )

        return json.loads(
            response.read().decode("utf-8")
        )


@app.post("/orchestrate")
def orchestrate(state: WorkloadState):
    """
    Original prototype endpoint.
    Preserves the simulated runtime comparison.
    """

    decision = choose_runtime(state)
    start = perf_counter()

    if decision["selected_runtime"] == "serverless":
        simulated_latency = (
            70.0
            + SERVERLESS_COLD_START_MS
            * state.cold_start_rate_pct / 100.0
        )
        cost_units = SERVERLESS_COST_UNITS
        runtime_key = "serverless"

    else:
        simulated_latency = (
            CONTAINER_BASE_LATENCY_MS
            + max(
                0.0,
                state.cpu_utilization_pct - 60.0
            ) * 3.0
        )
        cost_units = CONTAINER_COST_UNITS
        runtime_key = "container"

    api_latency_ms = round(
        (perf_counter() - start) * 1000.0, 3
    )

    sla_violation = (
        simulated_latency > EXAMPLE_SLO_MS
        or state.p95_latency_ms > EXAMPLE_SLO_MS
    )

    record(
        runtime_key,
        simulated_latency,
        cost_units,
        sla_violation,
    )

    return {
        **decision,
        "simulated_latency_ms": round(
            simulated_latency, 2
        ),
        "orchestration_api_latency_ms": api_latency_ms,
        "simulated_cost_units": cost_units,
        "example_slo_ms": EXAMPLE_SLO_MS,
        "simulated_sla_violation": sla_violation,
        "warning": (
            "Latency and cost are synthetic simulation values, "
            "not real cloud measurements."
        ),
    }


@app.post("/orchestrate/predict")
def orchestrate_predict(payload: HybridInferenceRequest):
    """
    Select a runtime using Random Forest and execute
    the Iris prediction on Render or Modal.
    """

    decision = choose_runtime(payload.workload)
    runtime = decision["selected_runtime"]

    start = perf_counter()

    try:
        if runtime == "container":
            # Execute inference directly inside Render.
            features = [
                payload.prediction.sepal_length,
                payload.prediction.sepal_width,
                payload.prediction.petal_length,
                payload.prediction.petal_width,
            ]

            prediction_result = predict_iris(features)

        elif runtime == "serverless":
            # Forward inference to the permanent Modal endpoint.
            prediction_result = call_modal_inference(
                payload.prediction
            )

        else:
            raise RuntimeError(
                f"Unsupported runtime selected: {runtime}"
            )

        execution_latency_ms = round(
            (perf_counter() - start) * 1000.0, 3
        )

        record_actual(
            runtime=runtime,
            latency_ms=execution_latency_ms,
            success=True,
            sla_violation=execution_latency_ms > ACTUAL_SLO_MS,
        )

        return {
            **decision,
            "execution": {
                "status": "success",
                "runtime": runtime,
                "measured_latency_ms": execution_latency_ms,
                "actual_slo_ms": ACTUAL_SLO_MS,
                "sla_violation": (
                    execution_latency_ms > ACTUAL_SLO_MS
                ),
                "prediction": prediction_result,
            },
            "measurement_note": (
                "Container latency measures local inference "
                "execution. Serverless latency includes the "
                "Render-to-Modal HTTP round trip. These scopes "
                "must be considered during performance comparison."
            ),
        }

    except HTTPException:
        execution_latency_ms = round(
            (perf_counter() - start) * 1000.0, 3
        )

        record_actual(
            runtime,
            execution_latency_ms,
            success=False,
            sla_violation=False,
        )
        raise

    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        execution_latency_ms = round(
            (perf_counter() - start) * 1000.0, 3
        )

        record_actual(
            runtime,
            execution_latency_ms,
            success=False,
            sla_violation=False,
        )

        raise HTTPException(
            status_code=502,
            detail=f"Hybrid inference execution failed: {exc}",
        ) from exc

    except Exception as exc:
        execution_latency_ms = round(
            (perf_counter() - start) * 1000.0, 3
        )

        record_actual(
            runtime,
            execution_latency_ms,
            success=False,
            sla_violation=False,
        )

        raise HTTPException(
            status_code=502,
            detail=f"Prediction execution failed: {exc}",
        ) from exc


@app.get("/metrics")
def metrics():
    return snapshot()


@app.post("/metrics/reset")
def metrics_reset():
    reset()
    return {"status": "metrics reset"}
