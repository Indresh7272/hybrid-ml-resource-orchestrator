
import os
import modal
import secrets

from functools import lru_cache
from time import perf_counter

from fastapi import FastAPI, HTTPException, Header

from app.schemas import (
    WorkloadState,
    PredictionInput,
    HybridInferenceRequest,
    BenchmarkInferenceRequest
)

from app.orchestrator import choose_runtime
from app.model_service import predict_iris
from app.metrics import record, record_actual, snapshot, reset
from app.phase7_policy import choose_measured_runtime
from app.phase7f_policy import Phase7FRequest, choose_phase7f_runtime
from fastapi.responses import HTMLResponse
# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Hybrid ML Resource Orchestrator",
    description="Hybrid serverless and containerized ML orchestration",
    version="0.5.0",
)


# --------------------------------------------------
# Existing prototype simulation values
# --------------------------------------------------

SERVERLESS_COST_UNITS = 1.0
CONTAINER_COST_UNITS = 0.35

SERVERLESS_COLD_START_MS = 250.0
CONTAINER_BASE_LATENCY_MS = 35.0

EXAMPLE_SLO_MS = 500.0


# --------------------------------------------------
# Actual hybrid execution configuration
# --------------------------------------------------

ACTUAL_SLO_MS = 500.0
BENCHMARK_API_KEY = os.getenv("BENCHMARK_API_KEY", "")

MODAL_APP_NAME = "hybrid-ml-serverless-inference"
MODAL_FUNCTION_NAME = "predict_iris_remote"

MODAL_ENVIRONMENT = os.getenv(
    "MODAL_ENVIRONMENT",
    "main",
)


# --------------------------------------------------
# Modal SDK connection
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_modal_function():
    """
    Retrieve the deployed Modal function.

    The Modal SDK authenticates using the configured
    Modal credentials.
    """

    return modal.Function.from_name(
        MODAL_APP_NAME,
        MODAL_FUNCTION_NAME,
        environment_name=MODAL_ENVIRONMENT,
    )


def call_modal_inference(payload: PredictionInput) -> dict:
    """
    Execute Iris inference remotely through Modal SDK.
    """

    modal_function = get_modal_function()

    prediction_result = modal_function.remote(
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    )

    return prediction_result


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hybrid ML Resource Orchestrator</title>
        <style>
            * { box-sizing: border-box; }

            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f4f8fb;
                color: #14354d;
            }

            nav {
                background: #14354d;
                color: white;
                padding: 22px 8%;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 16px;
            }

            nav a {
                color: white;
                text-decoration: none;
                margin-left: 22px;
                font-size: 14px;
            }

            .hero {
                max-width: 1100px;
                margin: 75px auto 45px;
                padding: 20px;
                text-align: center;
            }

            .tag {
                color: #159d91;
                font-weight: bold;
                letter-spacing: 2px;
                font-size: 12px;
            }

            h1 {
                font-size: clamp(32px, 5vw, 52px);
                line-height: 1.15;
                max-width: 850px;
                margin: 20px auto;
            }

            .description {
                color: #647786;
                max-width: 700px;
                margin: 0 auto;
                line-height: 1.8;
            }

            .button {
                display: inline-block;
                padding: 14px 25px;
                margin: 25px 8px 0;
                border-radius: 8px;
                background: #159d91;
                color: white;
                text-decoration: none;
                font-weight: bold;
            }

            .secondary {
                background: #14354d;
            }

            .cards {
                max-width: 1050px;
                margin: 45px auto;
                padding: 20px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
            }

            .card {
                padding: 28px;
                background: white;
                border: 1px solid #d8e2e9;
                border-radius: 12px;
            }

            .card p {
                color: #647786;
                line-height: 1.6;
                font-size: 14px;
            }

            footer {
                text-align: center;
                padding: 30px;
                color: #647786;
                border-top: 1px solid #d8e2e9;
                font-size: 13px;
            }
        </style>
    </head>
    <body>

        <nav>
            <strong>HYBRID ML ORCHESTRATOR</strong>
            <div>
                <a href="/">Home</a>
                <a href="/docs">API Docs</a>
                <a href="/redoc">ReDoc</a>
                <a href="/health">Health</a>
            </div>
        </nav>

        <section class="hero">
            <div class="tag">CLOUD + ARTIFICIAL INTELLIGENCE</div>

            <h1>
                Smarter ML Inference Through
                Hybrid Cloud Orchestration
            </h1>

            <p class="description">
                A workload-aware orchestration prototype that
                evaluates containerized and serverless machine
                learning execution through ML-driven, empirical
                and adaptive runtime recommendations.
            </p>

            <a class="button" href="/docs">
                Explore API Documentation
            </a>

            <a class="button secondary" href="/health">
                Check API Health
            </a>
        </section>

        <section class="cards">
            <div class="card">
                <h3>Container Inference</h3>
                <p>Local ML inference hosted through the Render container.</p>
            </div>

            <div class="card">
                <h3>Serverless Inference</h3>
                <p>Remote machine learning execution using Modal.</p>
            </div>

            <div class="card">
                <h3>ML Orchestration</h3>
                <p>Random Forest-based workload runtime recommendations.</p>
            </div>

            <div class="card">
                <h3>Adaptive Policies</h3>
                <p>Compare balanced, latency/SLO and throughput objectives.</p>
            </div>
        </section>

        <footer>
            M.Tech CSE — AI & ML | Hybrid ML Resource Orchestrator
            <br><br>
            Research Prototype — Render + Modal
        </footer>

    </body>
    </html>
    """

# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "phase": "hybrid execution integration"
    }

# --------------------------------------------------
# Direct container prediction endpoint
# --------------------------------------------------

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


# --------------------------------------------------
# Original prototype orchestration endpoint
# --------------------------------------------------

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
        (perf_counter() - start) * 1000.0,
        3,
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
            simulated_latency,
            2,
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


# --------------------------------------------------
# New hybrid inference orchestration endpoint
# --------------------------------------------------

@app.post("/orchestrate/predict")
def orchestrate_predict(payload: HybridInferenceRequest):
    """
    Select the execution runtime using Random Forest.

    Container:
        Execute Iris inference directly inside Render.

    Serverless:
        Invoke the deployed Modal function through
        the authenticated Modal Python SDK.

    Record actual execution latency and request metrics.
    """

    # Step 1: Select runtime using the trained policy
    decision = choose_runtime(payload.workload)

    runtime = decision["selected_runtime"]

    # Step 2: Start measuring execution latency
    start = perf_counter()

    try:

        # ------------------------------------------
        # Container execution on Render
        # ------------------------------------------

        if runtime == "container":

            features = [
                payload.prediction.sepal_length,
                payload.prediction.sepal_width,
                payload.prediction.petal_length,
                payload.prediction.petal_width,
            ]

            prediction_result = predict_iris(features)

        # ------------------------------------------
        # Serverless execution on Modal
        # ------------------------------------------

        elif runtime == "serverless":

            prediction_result = call_modal_inference(
                payload.prediction
            )

        else:

            raise RuntimeError(
                f"Unsupported runtime selected: {runtime}"
            )

        # ------------------------------------------
        # Step 3: Calculate measured latency
        # ------------------------------------------

        execution_latency_ms = round(
            (perf_counter() - start) * 1000.0,
            3,
        )

        # ------------------------------------------
        # Step 4: Record successful execution
        # ------------------------------------------

        sla_violation = (
            execution_latency_ms > ACTUAL_SLO_MS
        )

        record_actual(
            runtime=runtime,
            latency_ms=execution_latency_ms,
            success=True,
            sla_violation=sla_violation,
        )

        # ------------------------------------------
        # Step 5: Return execution result
        # ------------------------------------------

        return {
            **decision,

            "execution": {
                "status": "success",

                "runtime": runtime,

                "measured_latency_ms": execution_latency_ms,

                "actual_slo_ms": ACTUAL_SLO_MS,

                "sla_violation": sla_violation,

                "prediction": prediction_result,
            },

            "measurement_note": (
                "Container latency measures inference execution "
                "inside the Render service. Serverless latency "
                "includes the remote Modal invocation, network "
                "overhead, and any startup or queueing delay. "
                "These scopes should be considered during "
                "performance comparison."
            ),
        }

    # ----------------------------------------------
    # Handle explicitly raised FastAPI errors
    # ----------------------------------------------

    except HTTPException:

        execution_latency_ms = round(
            (perf_counter() - start) * 1000.0,
            3,
        )

        record_actual(
            runtime=runtime,
            latency_ms=execution_latency_ms,
            success=False,
            sla_violation=False,
        )

        raise

    # ----------------------------------------------
    # Handle Modal SDK and other execution failures
    # ----------------------------------------------

    except Exception as exc:

        execution_latency_ms = round(
            (perf_counter() - start) * 1000.0,
            3,
        )

        record_actual(
            runtime=runtime,
            latency_ms=execution_latency_ms,
            success=False,
            sla_violation=False,
        )

        raise HTTPException(
            status_code=502,
            detail=f"Hybrid inference execution failed: {exc}",
        ) from exc


# --------------------------------------------------
# Metrics endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():

    return snapshot()


# --------------------------------------------------
# Metrics reset endpoint
# --------------------------------------------------

@app.post("/metrics/reset")
def metrics_reset():

    reset()

    return {
        "status": "metrics reset"
    }

@app.post("/benchmark/predict")
def benchmark_predict(
    payload: BenchmarkInferenceRequest,
    x_benchmark_key: str | None = Header(default=None)
):
    # Keep this endpoint protected from unauthorized requests.
    if not BENCHMARK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Benchmark endpoint is disabled. Configure BENCHMARK_API_KEY."
        )

    if (
        not x_benchmark_key
        or not secrets.compare_digest(x_benchmark_key, BENCHMARK_API_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid benchmark API key."
        )

    # Get the ML model's recommendation separately from the forced runtime.
    recommendation = choose_runtime(payload.workload)
    runtime = payload.execution_runtime

    start_time = perf_counter()

    try:
        features = [
            payload.prediction.sepal_length,
            payload.prediction.sepal_width,
            payload.prediction.petal_length,
            payload.prediction.petal_width
        ]

        # Execute on the specifically requested runtime.
        if runtime == "container":
            prediction_result = predict_iris(features)
        else:
            prediction_result = call_modal_inference(payload.prediction)

        measured_latency = (perf_counter() - start_time) * 1000
        sla_violation = measured_latency > ACTUAL_SLO_MS

        record_actual(
            runtime,
            measured_latency,
            success=True,
            sla_violation=sla_violation
        )

        return {
            "benchmark_mode": "forced_runtime",
            "recommended_runtime": recommendation.get("selected_runtime"),
            "recommended_policy": recommendation.get("policy"),
            "recommendation_confidence": recommendation.get("model_confidence"),
            "execution": {
                "status": "success",
                "runtime": runtime,
                "measured_latency_ms": round(measured_latency, 3),
                "actual_slo_ms": ACTUAL_SLO_MS,
                "sla_violation": sla_violation,
                "prediction": prediction_result
            },
            "measurement_note": (
                "Latency is measured on the Render API server. "
                "Container execution uses local inference in Render, "
                "while serverless execution includes the remote Modal SDK invocation."
            )
        }

    except Exception as exc:
        measured_latency = (perf_counter() - start_time) * 1000

        record_actual(
            runtime,
            measured_latency,
            success=False,
            sla_violation=measured_latency > ACTUAL_SLO_MS
        )

        raise HTTPException(
            status_code=502,
            detail=f"{runtime} inference failed: {type(exc).__name__}"
        ) from exc

@app.post("/orchestrate/compare")
def orchestrate_compare(state: WorkloadState):

    # Original Random Forest recommendation
    random_forest_decision = choose_runtime(state)

    # New Phase 7 empirical recommendation
    measured_decision = choose_measured_runtime(state)

    return {
        "comparison_mode": "policy_recommendations_only",
        "workload_type": state.workload_type,

        "random_forest_policy": random_forest_decision,

        "phase7_empirical_policy": measured_decision,

        "recommendations_agree": (
            random_forest_decision.get("selected_runtime")
            == measured_decision.get("selected_runtime")
        ),

        "execution_performed": False,

        "note": (
            "This endpoint compares policy recommendations only. "
            "It does not execute inference or change the existing "
            "/orchestrate/predict route."
        )
    }

@app.post("/orchestrate/adaptive")
def orchestrate_adaptive(payload: Phase7FRequest):

    # Original Random Forest recommendation
    random_forest_decision = choose_runtime(payload.workload)

    # Phase 7 empirical recommendation
    empirical_decision = choose_measured_runtime(payload.workload)

    # New Phase 7F adaptive recommendation
    adaptive_decision = choose_phase7f_runtime(payload)

    return {
        "comparison_mode": "recommendation_only",

        "random_forest_policy": random_forest_decision,

        "phase7_empirical_policy": empirical_decision,

        "phase7f_adaptive_policy": adaptive_decision,

        "recommendations": {
            "random_forest": random_forest_decision.get("selected_runtime"),
            "phase7_empirical": empirical_decision.get("selected_runtime"),
            "phase7f_adaptive": adaptive_decision.get("selected_runtime")
        },

        "execution_performed": False
    }