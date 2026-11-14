from typing import Literal
from pydantic import BaseModel, Field

class WorkloadState(BaseModel):
    request_rate_per_min: float = Field(ge=0, le=1_000_000)
    predicted_rate_per_min: float = Field(ge=0, le=1_000_000)
    cpu_utilization_pct: float = Field(ge=0, le=100)
    memory_utilization_pct: float = Field(ge=0, le=100)
    p95_latency_ms: float = Field(ge=0)
    cold_start_rate_pct: float = Field(ge=0, le=100)
    workload_type: Literal["bursty", "steady", "sustained", "batch"] = "steady"

class PredictionInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


class HybridInferenceRequest(BaseModel):
    workload: WorkloadState
    prediction: PredictionInput


class BenchmarkInferenceRequest(BaseModel):
    workload: WorkloadState
    prediction: PredictionInput
    execution_runtime: Literal["container", "serverless"]
