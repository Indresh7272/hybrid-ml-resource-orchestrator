# Phase 7F — Adaptive Multi-Objective Orchestration (Candidate)

## Purpose

This is a separate, recommendation-only candidate policy. It does not replace the Random Forest policy, the existing Phase 7 empirical policy, or the current `/orchestrate/predict` endpoint.

It considers:
- workload category;
- caller-supplied concurrency level;
- optimization objective (`balanced`, `latency_slo`, or `throughput`);
- explicit acknowledgement of SLO risk (`allow_slo_risk`).

## Evidence used

The only serverless exception is supported by repeated steady-workload / concurrency-6 waves pooled from:

- Phase 7D run `20261118T052443Z_e8dcc6df`: 4 paired waves per runtime, 24 requests per runtime.
- Phase 7E run `20261118T053631Z_d0cb8c00`: 12 paired waves per runtime, 72 requests per runtime.

Combined at steady / concurrency 6:

| Metric | Container | Serverless |
|---|---:|---:|
| Paired waves | 16 | 16 |
| Requests | 96 | 96 |
| Mean server latency | 580.72 ms | 637.65 ms |
| Median server latency | 507.63 ms | 568.29 ms |
| P95 server latency | 1100.65 ms | 992.61 ms |
| Mean wave throughput | 1.7053 req/s | 1.8729 req/s |
| SLO violations | 55/96 (57.29%) | 68/96 (70.83%) |

Serverless mean wave throughput was about 9.83% higher and request-level P95 was lower, but mean server latency was about 9.80% higher, median latency was higher, and the SLO violation rate was 13.54 percentage points higher. This is a trade-off, not an unconditional serverless advantage.

## Policy behavior

- `balanced`: container.
- `latency_slo`: container.
- `throughput`: serverless only for `steady` and concurrency `6`, and only if `allow_slo_risk=True`.
- All other profile/concurrency combinations: conservative container fallback.
- Unsupported concurrency levels: container fallback with a scope warning.

The serverless route is deliberately gated. At the measured condition, neither runtime consistently meets the 500 ms SLO. The caller must explicitly acknowledge that risk.

## Install

Extract this ZIP into the project root without overwriting existing files. Expected new files:

- `app/phase7f_policy.py`
- `artifacts/phase7f_concurrency_evidence.json`
- `test_phase7f_policy.py`
- `PHASE7F_README.md`

## Local tests

```powershell
python -m py_compile .\app\phase7f_policy.py .\test_phase7f_policy.py
python .\test_phase7f_policy.py
```

## Optional API integration

In `app/main.py`, add:

```python
from app.phase7f_policy import Phase7FRequest, choose_phase7f_runtime
```

Then add this new endpoint without changing existing routes:

```python
@app.post("/orchestrate/adaptive")
def orchestrate_adaptive(payload: Phase7FRequest):
    random_forest_decision = choose_runtime(payload.workload)
    empirical_decision = choose_measured_runtime(payload.workload)
    adaptive_decision = choose_phase7f_runtime(payload)

    return {
        "comparison_mode": "recommendation_only",
        "random_forest_policy": random_forest_decision,
        "phase7_empirical_policy": empirical_decision,
        "phase7f_adaptive_policy": adaptive_decision,
        "execution_performed": False,
    }
```

Keep the existing imports for `choose_runtime` and `choose_measured_runtime`. The example assumes the prior `/orchestrate/compare` integration is present.

## Example request

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

The endpoint is recommendation-only. It does not execute inference or automatically determine live concurrency. A caller or future concurrency controller must supply `concurrency_level`.

## Limitations

- Evidence is specific to the tested Render/Modal deployment and Iris inference.
- The observed 500 ms SLO violation rates at concurrency 6 are high for both runtimes.
- The throughput exception is justified only for steady workload at concurrency 6; do not extrapolate it to other profiles or concurrency levels.
- Cost and resource utilization have not been measured by these experiments.
- No live request counter, admission control, autoscaling or runtime execution integration is included.
- This candidate must pass local tests and an independent evaluation before it is considered for production routing.
