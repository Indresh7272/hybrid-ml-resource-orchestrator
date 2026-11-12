
from threading import Lock

_lock = Lock()

_state = {
    # Existing prototype metrics (simulated)
    "requests_total": 0,
    "serverless_requests": 0,
    "container_requests": 0,
    "estimated_cost_units": 0.0,
    "simulated_sla_violations": 0,
    "latencies_ms": [],

    # New hybrid execution metrics (measured)
    "actual_requests_total": 0,
    "actual_serverless_requests": 0,
    "actual_container_requests": 0,
    "actual_successful_requests": 0,
    "actual_failed_requests": 0,
    "actual_sla_violations": 0,
    "actual_latencies_ms": [],
}


def _percentiles(values):
    latencies = sorted(values)

    def percentile(p):
        if not latencies:
            return None

        idx = min(
            len(latencies) - 1,
            int(round((p / 100) * (len(latencies) - 1)))
        )
        return round(latencies[idx], 2)

    return {
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
    }


# Existing function: retained for prototype simulation metrics
def record(
    runtime: str,
    latency_ms: float,
    cost_units: float,
    sla_violation: bool
):
    with _lock:
        _state["requests_total"] += 1
        _state[f"{runtime}_requests"] += 1
        _state["estimated_cost_units"] += cost_units
        _state["latencies_ms"].append(latency_ms)

        if sla_violation:
            _state["simulated_sla_violations"] += 1


# New function: records actual hybrid execution measurements
def record_actual(
    runtime: str,
    latency_ms: float,
    success: bool,
    sla_violation: bool
):
    if runtime not in {"serverless", "container"}:
        raise ValueError(f"Unsupported runtime: {runtime}")

    with _lock:
        _state["actual_requests_total"] += 1
        _state[f"actual_{runtime}_requests"] += 1
        _state["actual_latencies_ms"].append(latency_ms)

        if success:
            _state["actual_successful_requests"] += 1
        else:
            _state["actual_failed_requests"] += 1

        if sla_violation:
            _state["actual_sla_violations"] += 1


def snapshot():
    with _lock:
        simulated_total = _state["requests_total"]
        actual_total = _state["actual_requests_total"]

        return {
            # Existing simulated metrics
            "requests_total": simulated_total,
            "serverless_requests": _state["serverless_requests"],
            "container_requests": _state["container_requests"],
            "estimated_cost_units_not_real_currency": round(
                _state["estimated_cost_units"], 4
            ),
            "simulated_sla_violations": _state[
                "simulated_sla_violations"
            ],
            "simulated_sla_violation_rate_pct": (
                round(
                    100 * _state["simulated_sla_violations"]
                    / simulated_total, 2
                ) if simulated_total else 0
            ),
            "simulated_latency_ms": _percentiles(
                _state["latencies_ms"]
            ),

            # New measured hybrid execution metrics
            "actual_execution": {
                "requests_total": actual_total,
                "serverless_requests": _state[
                    "actual_serverless_requests"
                ],
                "container_requests": _state[
                    "actual_container_requests"
                ],
                "successful_requests": _state[
                    "actual_successful_requests"
                ],
                "failed_requests": _state[
                    "actual_failed_requests"
                ],
                "actual_sla_violations": _state[
                    "actual_sla_violations"
                ],
                "actual_sla_violation_rate_pct": (
                    round(
                        100 * _state["actual_sla_violations"]
                        / actual_total, 2
                    ) if actual_total else 0
                ),
                "measured_latency_ms": _percentiles(
                    _state["actual_latencies_ms"]
                ),
            },
        }


def reset():
    with _lock:
        _state.update({
            "requests_total": 0,
            "serverless_requests": 0,
            "container_requests": 0,
            "estimated_cost_units": 0.0,
            "simulated_sla_violations": 0,
            "latencies_ms": [],

            "actual_requests_total": 0,
            "actual_serverless_requests": 0,
            "actual_container_requests": 0,
            "actual_successful_requests": 0,
            "actual_failed_requests": 0,
            "actual_sla_violations": 0,
            "actual_latencies_ms": [],
        })
