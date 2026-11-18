"""Empirical profile-level runtime policy calibrated from Phase 6B results.

This is an experimental comparator, not a general production policy. It uses
observed SLO violation rate first, then median server-side latency and P95.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas import WorkloadState

_EVIDENCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "phase7_profile_evidence.json"
)

with _EVIDENCE_PATH.open("r", encoding="utf-8") as _file:
    _EVIDENCE: dict[str, Any] = json.load(_file)


def choose_measured_runtime(state: WorkloadState) -> dict[str, Any]:
    """Select a runtime using measured profile-level SLO and latency evidence.

    The lookup is intentionally limited to the four tested workload profiles.
    It does not extrapolate to unmeasured traffic patterns, model sizes,
    concurrency levels, regions, instance sizes, or provider pricing.
    """
    profile_key = {
        "bursty": "bursty_low_load",
        "steady": "steady_moderate_load",
        "sustained": "sustained_high_load",
        "batch": "batch_heavy_load",
    }.get(state.workload_type)

    if profile_key is None or profile_key not in _EVIDENCE["profiles"]:
        raise ValueError(f"No Phase 6B evidence available for workload type: {state.workload_type}")

    profile = _EVIDENCE["profiles"][profile_key]
    candidates = {
        runtime: profile[runtime]
        for runtime in ("container", "serverless")
    }

    # SLO compliance has priority; latency breaks ties.
    selected_runtime = min(
        candidates,
        key=lambda runtime: (
            candidates[runtime]["sla_violation_rate_pct"],
            candidates[runtime]["median_latency_ms"],
            candidates[runtime]["p95_latency_ms"],
        ),
    )

    selected = candidates[selected_runtime]
    alternative_runtime = "serverless" if selected_runtime == "container" else "container"
    alternative = candidates[alternative_runtime]

    return {
        "selected_runtime": selected_runtime,
        "policy": "phase7_empirical_slo_latency_policy",
        "workload_profile": profile_key,
        "selection_priority": [
            "measured_slo_violation_rate_pct",
            "measured_median_latency_ms",
            "measured_p95_latency_ms",
        ],
        "selected_runtime_evidence": selected,
        "alternative_runtime_evidence": alternative,
        "sample_size_per_runtime": selected["n"],
        "source_run_id": _EVIDENCE["source_run_id"],
        "slo_threshold_ms": _EVIDENCE["slo_threshold_ms"],
        "scope_warning": (
            "Experimental profile-level policy calibrated from one sequential "
            "Render/Modal Iris-inference benchmark. It is not validated for "
            "concurrent load, other models, different regions/instance sizes, "
            "or cost optimization. Preserve the Random Forest policy as a comparator."
        ),
    }
