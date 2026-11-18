"""Phase 7F: conservative multi-objective runtime-selection candidate.

This module is deliberately separate from the existing Random Forest and
Phase 7 empirical policies. It produces a recommendation only; it does not
execute inference or alter the production /orchestrate/predict route.

The only observed exception to the conservative container choice is the
steady workload at concurrency 6 when throughput is explicitly prioritized
AND the caller explicitly accepts the observed SLO risk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas import WorkloadState

_EVIDENCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "phase7f_concurrency_evidence.json"
)
with _EVIDENCE_PATH.open("r", encoding="utf-8") as _handle:
    _EVIDENCE: dict[str, Any] = json.load(_handle)


class Phase7FRequest(BaseModel):
    """Inputs to the Phase 7F recommendation-only policy endpoint."""

    workload: WorkloadState
    concurrency_level: int = Field(ge=1, le=100)
    optimization_objective: Literal["balanced", "latency_slo", "throughput"] = "balanced"
    allow_slo_risk: bool = False


def choose_phase7f_runtime(request: Phase7FRequest) -> dict[str, Any]:
    """Recommend a runtime using objective, concurrency and observed evidence.

    Policy rules:
    - Default to container for balanced and latency/SLO objectives.
    - Use serverless for steady/concurrency-6 only when throughput is the
      explicit objective and allow_slo_risk=True.
    - Fail closed to container for unsupported workload/concurrency regions.

    Concurrency is caller-supplied; this function does not measure live
    in-flight requests. Evidence is specific to the tested Render/Modal
    deployment and Iris inference workload.
    """
    workload_type = request.workload.workload_type
    concurrency = request.concurrency_level
    objective = request.optimization_objective

    selected_runtime = "container"
    selection_reason = "Container is the conservative default for this tested configuration."
    slo_warning = None
    evidence = None
    scope_warning = None

    supported_workloads = {"bursty", "steady", "sustained", "batch"}
    supported_concurrency = {3, 6}

    if workload_type not in supported_workloads:
        scope_warning = (
            "This workload type is outside the four measured profiles; container is a conservative fallback."
        )
    elif concurrency not in supported_concurrency:
        scope_warning = (
            "This concurrency level is outside the Phase 7D/7E confirmation set (3 and 6); "
            "container is a conservative fallback."
        )
    elif objective in {"balanced", "latency_slo"}:
        selection_reason = (
            "Latency/SLO or balanced operation prioritizes the lower observed mean/median latency "
            "and lower aggregate SLO-violation rate; container is selected. The pooled "
            "serverless P95 was lower, so the trade-off is not uniform across every latency statistic."
        )
    elif objective == "throughput" and workload_type == "steady" and concurrency == 6:
        evidence = _EVIDENCE["steady_concurrency_6_pooled"]
        if request.allow_slo_risk:
            selected_runtime = "serverless"
            selection_reason = (
                "For steady workload at concurrency 6, repeated Phase 7D/7E waves showed "
                "higher mean throughput with serverless. This choice knowingly accepts "
                "higher observed latency and SLO-violation rate."
            )
            slo_warning = (
                "Observed pooled SLO violations were 70.83% for serverless versus 57.29% "
                "for container across 16 waves per runtime (96 requests each). "
                "The 500 ms SLO was frequently missed by both runtimes."
            )
        else:
            selection_reason = (
                "Serverless showed a throughput advantage for this condition, but it also "
                "had a higher observed SLO-violation rate. allow_slo_risk must be explicitly "
                "enabled before this policy recommends serverless; container remains selected."
            )
            slo_warning = (
                "Throughput-priority serverless routing is blocked because allow_slo_risk=False."
            )
    elif objective == "throughput":
        selection_reason = (
            "No repeated evidence supports a serverless throughput exception for this "
            "workload/concurrency combination; container is selected conservatively."
        )

    return {
        "selected_runtime": selected_runtime,
        "policy": "phase7f_adaptive_multi_objective_candidate",
        "workload_type": workload_type,
        "concurrency_level": concurrency,
        "optimization_objective": objective,
        "allow_slo_risk": request.allow_slo_risk,
        "selection_reason": selection_reason,
        "evidence": evidence,
        "slo_warning": slo_warning,
        "scope_warning": scope_warning,
        "execution_performed": False,
        "measurement_scope": (
            "Recommendation only. Concurrency is supplied by the caller, not measured live. "
            "Evidence is limited to sequential/concurrent Iris inference tests on the tested "
            "Render and Modal configuration; no cost or resource-utilization optimization "
            "has been established."
        ),
    }
