from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .serving import default_model_path, load_serving_artifacts, predict_from_news_fields, predict_labels


class PredictRequest(BaseModel):
    title: str = Field(default="")
    content: str = Field(default="")
    meta_description: str = Field(default="")
    summary: str = Field(default="")


class BatchPredictRequest(BaseModel):
    texts: list[str]


def create_app(model_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Fake News Detection API", version="1.0.0")

    path = Path(model_path) if model_path else default_model_path()
    artifacts = None

    try:
        artifacts = load_serving_artifacts(path)
    except FileNotFoundError:
        artifacts = None

    @app.get("/health")
    def health() -> dict[str, str]:
        if artifacts is None:
            return {
                "status": "degraded",
                "message": "model not loaded; run scripts/train_serving_model.py",
            }
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict:
        if artifacts is None:
            raise HTTPException(status_code=503, detail="Model is not available")
        return {
            "metadata": artifacts.metadata,
            "metrics": artifacts.metrics,
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, str]:
        if artifacts is None:
            raise HTTPException(status_code=503, detail="Model is not available")
        try:
            label = predict_from_news_fields(
                artifacts,
                title=request.title,
                content=request.content,
                meta_description=request.meta_description,
                summary=request.summary,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"label": label}

    @app.post("/predict-batch")
    def predict_batch(request: BatchPredictRequest) -> dict[str, list[str]]:
        if artifacts is None:
            raise HTTPException(status_code=503, detail="Model is not available")
        labels = predict_labels(artifacts, request.texts)
        return {"labels": labels}

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
