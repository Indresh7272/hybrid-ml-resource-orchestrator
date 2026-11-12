
import modal
from fastapi import FastAPI

# Create the Modal application
app = modal.App("hybrid-ml-serverless-inference")

# Define the cloud environment for ML inference
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastapi==0.159.1",
        "scikit-learn==1.9.1",
        "numpy==2.5.4",
        "pydantic==2.14.1",
    )
    .add_local_python_source("app")
)


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from app.model_service import predict_iris
    from app.schemas import PredictionInput

    api = FastAPI(
        title="Hybrid ML Serverless Inference",
        description="Serverless Iris ML inference endpoint",
        version="1.0.0",
    )

    @api.get("/health")
    def health():
        return {
            "status": "ok",
            "runtime": "modal_serverless"
        }

    @api.post("/predict")
    def predict(payload: PredictionInput):
        features = [
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]

        result = predict_iris(features)

        return {
            "service": "sample_iris_classifier",
            "runtime": "modal_serverless",
            **result,
        }

    return api
