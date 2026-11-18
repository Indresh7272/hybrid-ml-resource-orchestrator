from app.schemas import WorkloadState
from app.phase7_policy import choose_measured_runtime

profiles = ["bursty", "steady", "sustained", "batch"]

for profile in profiles:
    state = WorkloadState(
        request_rate_per_min=100,
        predicted_rate_per_min=120,
        cpu_utilization_pct=30,
        memory_utilization_pct=40,
        p95_latency_ms=150,
        cold_start_rate_pct=2,
        workload_type=profile,
    )
    result = choose_measured_runtime(state)
    print(
        f"{profile}: selected={result['selected_runtime']}; "
        f"policy={result['policy']}; "
        f"n/runtime={result['sample_size_per_runtime']}"
    )
    assert result["selected_runtime"] == "container"

print("Phase 7 empirical policy checks passed.")
