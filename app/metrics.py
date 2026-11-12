from threading import Lock

_lock = Lock()
_state = {
    "requests_total": 0,
    "serverless_requests": 0,
    "container_requests": 0,
    "estimated_cost_units": 0.0,
    "simulated_sla_violations": 0,
    "latencies_ms": []
}

def record(runtime: str, latency_ms: float, cost_units: float, sla_violation: bool):
    with _lock:
        _state["requests_total"] += 1
        _state[f"{runtime}_requests"] += 1
        _state["estimated_cost_units"] += cost_units
        _state["latencies_ms"].append(latency_ms)
        if sla_violation:
            _state["simulated_sla_violations"] += 1

def snapshot():
    with _lock:
        latencies = sorted(_state["latencies_ms"])
        def percentile(p):
            if not latencies:
                return None
            idx = min(len(latencies) - 1, int(round((p / 100) * (len(latencies) - 1))))
            return round(latencies[idx], 2)
        total = _state["requests_total"]
        return {
            "requests_total": total,
            "serverless_requests": _state["serverless_requests"],
            "container_requests": _state["container_requests"],
            "estimated_cost_units_not_real_currency": round(_state["estimated_cost_units"], 4),
            "simulated_sla_violations": _state["simulated_sla_violations"],
            "simulated_sla_violation_rate_pct": round(100 * _state["simulated_sla_violations"] / total, 2) if total else 0,
            "latency_ms": {"p50": percentile(50), "p95": percentile(95), "p99": percentile(99)}
        }

def reset():
    with _lock:
        _state.update({
            "requests_total": 0,
            "serverless_requests": 0,
            "container_requests": 0,
            "estimated_cost_units": 0.0,
            "simulated_sla_violations": 0,
            "latencies_ms": []
        })
