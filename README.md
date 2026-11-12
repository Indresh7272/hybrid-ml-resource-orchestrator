# Hybrid ML Resource Orchestrator — Starter Prototype

This is Phase 1 of the M.Tech project: **AI-Driven Resource Orchestration for Hybrid Serverless and Containerized Machine Learning Services**.

## What this prototype does
- Exposes a small ML prediction API.
- Accepts workload/telemetry information.
- Implements a transparent rule-based hybrid routing baseline.
- Simulates serverless cold-start overhead and per-request cost.
- Tracks request count, simulated latency, estimated cost, and SLA violations.

**Important:** This is a local simulation baseline, not a real AWS Lambda/Kubernetes deployment and not yet the proposed AI model. The next phases will collect traces, train an ML decision model, and integrate real serverless/container runtimes if feasible.

## Requirements
- Python 3.10+
- pip
- Optional: Docker

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open API docs at http://127.0.0.1:8000/docs

## Try it
Call `POST /orchestrate` with:
```json
{
  "request_rate_per_min": 80,
  "predicted_rate_per_min": 100,
  "cpu_utilization_pct": 15,
  "memory_utilization_pct": 30,
  "p95_latency_ms": 120,
  "cold_start_rate_pct": 2,
  "workload_type": "bursty"
}
```
Then try a high-load case:
```json
{
  "request_rate_per_min": 900,
  "predicted_rate_per_min": 1500,
  "cpu_utilization_pct": 82,
  "memory_utilization_pct": 75,
  "p95_latency_ms": 450,
  "cold_start_rate_pct": 1,
  "workload_type": "sustained"
}
```

Other endpoints:
- `GET /health`
- `POST /predict` — sample Iris classifier
- `GET /metrics`
- `POST /metrics/reset`

## Current baseline logic
The initial router is deliberately rule-based so we have a baseline to compare against. Do not describe this phase as AI-driven in research results. Next:
1. collect workload traces and outcomes,
2. define cost/latency/SLA labels or a reward,
3. train and evaluate an ML policy against the baseline,
4. add real Docker/container execution and optionally AWS Lambda,
5. run repeatable experiments.

The simulated cost units and latency are synthetic, not actual cloud prices or measurements.
