# AI-Driven Resource Orchestration for Hybrid Serverless and Containerized Machine Learning Services

**Repository status:** Research prototype / M.Tech project implementation  
**Application:** FastAPI  
**Container-hosted API:** Render  
**Serverless inference provider:** Modal  
**ML example:** Iris classification  

## 1. Project overview

This project evaluates runtime selection for machine-learning inference across two execution paths:

- **Container path:** inference is executed locally inside the FastAPI service hosted on Render.
- **Serverless path:** the Render service invokes a deployed Modal function through the Modal Python SDK.

The application exposes three separate runtime-selection strategies for comparison:

1. **Random Forest policy** — a classifier trained using synthetic proxy labels. The application loads `artifacts/runtime_policy.joblib` when present. If that file is absent, `app/orchestrator.py` falls back to its original rule-based baseline.
2. **Phase 7 empirical policy** — a profile-level policy derived from Phase 6B measurements. It prioritizes observed SLO-violation rate, then median latency, then P95 latency.
3. **Phase 7F adaptive candidate** — a recommendation-only policy that considers the workload category, caller-supplied concurrency level, optimization objective, and explicit acknowledgement of SLO risk.

The primary hybrid inference endpoint, `POST /orchestrate/predict`, continues to use the Random Forest/rule-based `choose_runtime` path. The Phase 7 comparison endpoints return policy recommendations only; they do not execute inference or automatically replace the primary route.

## 2. System architecture

```text
Client / Benchmark Runner
          |
          v
FastAPI service on Render
          |
          +---- /predict ----------------------> Local Iris inference
          |
          +---- /orchestrate/predict ----------> Random Forest / fallback policy
          |                                         |
          |                                         +--> Container: local inference
          |                                         |
          |                                         +--> Serverless: Modal SDK call
          |
          +---- /orchestrate/compare -----------> Random Forest vs Phase 7 empirical
          |
          +---- /orchestrate/adaptive ---------> RF vs Phase 7 vs Phase 7F recommendations
          |
          +---- /benchmark/predict ------------> Authenticated forced-runtime experiments
          |
          +---- /metrics ----------------------> In-process metrics snapshot
```

### Important measurement distinction

Container latency is measured around local model inference within the Render service. Serverless latency includes the remote Modal SDK invocation and can include network, queueing, startup, and runtime overhead. These are application-path measurements on the deployed prototype, not a hardware-equivalent benchmark between identical machines.

## 3. Repository layout

```text
.
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── orchestrator.py
│   ├── model_service.py
│   ├── metrics.py
│   ├── phase7_policy.py
│   └── phase7f_policy.py
├── artifacts/
│   ├── phase7_profile_evidence.json
│   ├── phase7f_concurrency_evidence.json
│   └── runtime_policy.joblib       # required to restore the trained RF model; not present in this package
├── tests/
│   ├── test_phase7_policy.py
│   └── test_phase7f_policy.py
├── results/                        # selected experiment and cloud usage exports
├── docs/                            # policy notes and review documents
├── modal_inference.py
├── requirements.txt
└── Dockerfile
```

## 4. Prerequisites

- Python 3.13 (the supplied Dockerfile and pinned dependency set target Python 3.13).
- pip.
- A Modal account and CLI authentication for deployment/inference integration.
- Render account for hosted API deployment.

## 5. Local setup

From the repository root (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set runtime environment variables in the current PowerShell session as required:

```powershell
$env:MODAL_ENVIRONMENT = "main"
# Configure MODAL_TOKEN_ID and MODAL_TOKEN_SECRET through a secure secret store
# or local environment variables. Do not put tokens in source control.
# BENCHMARK_API_KEY is only required to call POST /benchmark/predict.
```

Start the API locally:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

### Trained policy artifact requirement

This source-review package does **not** include `artifacts/runtime_policy.joblib`, because that binary model artifact was not present in the uploaded source bundle. If the original local project contains the trained file, copy it to `artifacts/runtime_policy.joblib` before claiming that the Random Forest model is restored or before reproducing the deployed classifier's decisions. Without it, `choose_runtime` intentionally falls back to the rule-based baseline. Do not substitute a newly generated or placeholder model and describe it as the original trained artifact.

The two JSON evidence files are included under `artifacts/` because the Phase 7 policy modules load them at import time.

## 6. API endpoint reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Application health check |
| POST | `/predict` | Direct container-hosted Iris prediction |
| POST | `/orchestrate` | Original prototype simulation; synthetic latency and cost units |
| POST | `/orchestrate/predict` | RF/fallback-selected execution on Render or Modal |
| GET | `/metrics` | In-process execution metrics snapshot |
| POST | `/metrics/reset` | Resets in-process metrics |
| POST | `/benchmark/predict` | Protected endpoint forcing `container` or `serverless` for controlled measurement |
| POST | `/orchestrate/compare` | Recommendation-only RF versus Phase 7 empirical policy comparison |
| POST | `/orchestrate/adaptive` | Recommendation-only comparison including Phase 7F |

### Example adaptive-policy request

```json
{
  "workload": {
    "request_rate_per_min": 250,
    "predicted_rate_per_min": 300,
    "cpu_utilization_pct": 35,
    "memory_utilization_pct": 45,
    "p95_latency_ms": 180,
    "cold_start_rate_pct": 1,
    "workload_type": "steady"
  },
  "concurrency_level": 6,
  "optimization_objective": "throughput",
  "allow_slo_risk": true
}
```

Supported `optimization_objective` values are `balanced`, `latency_slo`, and `throughput`. The concurrency value is caller-supplied; the code does not read live in-flight request counts. The throughput exception is narrowly scoped to the measured steady-workload/concurrency-6 condition and requires explicit `allow_slo_risk: true`.

## 7. Modal deployment

The Modal application definition is in `modal_inference.py`. It defines:

- Modal app name: `hybrid-ml-serverless-inference`.
- Remote function: `predict_iris_remote`.
- ASGI app: `fastapi_app` (provides `/health` and `/predict` on the Modal web endpoint).

After authenticating the Modal CLI and ensuring the `app/` source is available:

```powershell
modal deploy .\modal_inference.py
```

The Render API invokes `predict_iris_remote` through `modal.Function.from_name(...)` using `MODAL_ENVIRONMENT` (default `main`). Configure `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `MODAL_ENVIRONMENT` in Render's Environment settings. Configure `BENCHMARK_API_KEY` separately to protect the benchmark endpoint. Never commit credential values, `.modal.toml`, or local `.env` files.

## 8. Render deployment

The included `Dockerfile` uses `python:3.13-slim`, installs `requirements.txt`, copies `app/` and `artifacts/`, and starts Uvicorn on the Render-provided `PORT` (default 10000).

Connect the GitHub repository to a Render Docker web service. Set the health-check path to `/health`. Configure the environment variables described above through Render's dashboard. The `artifacts/` directory must exist because the Dockerfile contains `COPY artifacts ./artifacts`.

## 9. Local verification commands

```powershell
python -m compileall .\app
python -m tests.test_phase7_policy
python -m tests.test_phase7f_policy
```

The tests cover the Phase 7 profile-level selection and the eight Phase 7F policy scenarios. They are policy/unit checks, not performance tests.

## 10. Experimental evidence summary

The `results/` folder contains selected raw CSV exports supplied during the project conversation. Preserve raw files; run future experiments into new timestamped filenames.

| Experiment | Requests | Main observation |
|---|---:|---|
| Phase 6A controlled benchmark | 40 | Container mean 111.14 ms; serverless mean 467.34 ms; server-side SLO violations 0/20 vs 3/20 |
| Phase 6B validation | 160 | Container mean 124.96 ms; serverless mean 549.47 ms; SLO violations 0/80 vs 14/80 |
| Phase 7B holdout feature vectors | 40 | Phase 7 profile choice matched the faster observed runtime in 20/20 matched vectors; this is within known workload categories, not unseen categories |
| Phase 7C concurrency benchmark | 144 | Concurrent execution at levels 1, 3 and 6; initial results motivated repeated high-concurrency studies |
| Phase 7D high-concurrency confirmation | 288 | Container had lower overall mean latency; at concurrency 6, observed throughput was similar across runtimes |
| Phase 7E steady-workload confirmation | 216 | At steady/concurrency 6, serverless showed higher mean wave throughput, but higher mean latency and SLO violation rate |

### Interpretation limitations

- The Random Forest is trained on synthetic proxy labels; its confidence is not the probability that a chosen runtime will be faster or cheaper in production.
- Phase 7 empirical policy is profile-level and derived from the tested Render/Modal Iris-inference measurements.
- Phase 7B changes numeric feature vectors within the same four workload categories; it is not validation on unseen workload categories.
- Phase 7C–7E concurrent tests issue concurrent HTTP inference requests, but the numeric CPU and memory fields in a workload object are descriptors; they do not directly generate CPU or memory load.
- Phase 7F is recommendation-only. It accepts a caller-supplied concurrency value and does not measure live concurrency, automatically switch runtimes, perform admission control, or autoscale resources.
- The 500 ms SLO was frequently exceeded under higher-concurrency conditions; serverless should not be described as satisfying the SLO at concurrency 6.
- Render container and Modal serverless latency scopes differ, as described above.
- Cost reports are account/project-period evidence, not exact per-request cost attribution. The Modal billing export included multiple application IDs; align billing periods and IDs before calculating unit economics.
- Render free-plan CPU and memory graphs were not available in the supplied screenshots (the dashboard indicated that detailed application metrics require a paid plan). The recent 48-hour memory/CPU graph is observational only, not a complete seven-day utilization series.

## 11. Security and operational notes

Before a public or production deployment:

1. Protect `/metrics` and especially `/metrics/reset` with authentication or restrict them to internal access. In the inspected source, these routes have no authentication.
2. Review whether the Modal ASGI `/predict` web endpoint should be public. The separate Render-to-Modal SDK invocation uses Modal credentials, but the ASGI route itself has no application-level authentication in this source.
3. Keep `/benchmark/predict` protected by `BENCHMARK_API_KEY`; use a high-entropy secret and rotate it if exposed.
4. Do not commit `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `BENCHMARK_API_KEY`, `.modal.toml`, `.env`, or local credential files.
5. Metrics are stored in process memory and can reset on restart/redeploy; they are not a persistent observability database.
6. Add rate limiting, structured logs, persistent monitoring, and explicit request-size/time limits before production use.

## 12. Current project status

The prototype implementation, Render/Modal integration, controlled experiments, empirical policy, and Phase 7F recommendation candidate have been implemented and tested in the reported environment. Final academic submission still requires review against institutional formatting requirements, verification of references, final diagrams, and a demonstration script. This repository package is a cleaned source/evidence snapshot; it is not a claim of full production hardening.
