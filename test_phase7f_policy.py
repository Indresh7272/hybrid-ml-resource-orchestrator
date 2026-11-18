"""Local unit tests for the Phase 7F adaptive policy candidate."""
from app.schemas import WorkloadState
from app.phase7f_policy import Phase7FRequest, choose_phase7f_runtime


def workload(kind: str) -> WorkloadState:
    return WorkloadState(
        request_rate_per_min=250,
        predicted_rate_per_min=300,
        cpu_utilization_pct=35,
        memory_utilization_pct=45,
        p95_latency_ms=180,
        cold_start_rate_pct=1,
        workload_type=kind,
    )


def select(kind: str, concurrency: int, objective: str = "balanced", allow_risk: bool = False):
    request = Phase7FRequest(
        workload=workload(kind),
        concurrency_level=concurrency,
        optimization_objective=objective,
        allow_slo_risk=allow_risk,
    )
    return choose_phase7f_runtime(request)


def main() -> None:
    cases = [
        ("steady", 6, "balanced", False, "container", "balanced defaults to container"),
        ("steady", 6, "latency_slo", True, "container", "SLO objective stays on container"),
        ("steady", 6, "throughput", False, "container", "throughput route requires risk acknowledgement"),
        ("steady", 6, "throughput", True, "serverless", "explicit throughput and risk opt-in"),
        ("bursty", 6, "throughput", True, "container", "no repeated throughput exception for bursty"),
        ("steady", 3, "throughput", True, "container", "no throughput exception at concurrency 3"),
        ("sustained", 6, "throughput", True, "container", "no repeated throughput exception for sustained"),
        ("batch", 12, "throughput", True, "container", "unsupported concurrency falls back to container"),
    ]

    for kind, concurrency, objective, risk, expected, description in cases:
        result = select(kind, concurrency, objective, risk)
        actual = result["selected_runtime"]
        assert actual == expected, (
            f"{description}: expected {expected}, received {actual}"
        )
        assert result["execution_performed"] is False
        print(
            f"PASS | workload={kind:9} concurrency={concurrency:2} "
            f"objective={objective:10} allow_slo_risk={str(risk):5} "
            f"selected={actual:10} | {description}"
        )

    # The serverless exception must carry its measured trade-off evidence.
    serverless_case = select("steady", 6, "throughput", True)
    assert serverless_case["evidence"]["paired_waves_per_runtime"] == 16
    assert serverless_case["slo_warning"]

    print(f"\nPhase 7F policy checks passed: {len(cases)} scenarios.")


if __name__ == "__main__":
    main()
