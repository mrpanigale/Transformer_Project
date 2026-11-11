"""FastAPI service for ParsBERT review predictions."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.inference import (
    ArtifactError,
    ArtifactsMissingError,
    ParsBERTPredictor,
    TextTooLongError,
    default_model_dir,
)
from app.normalization import normalize_persian_text


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("digikala_review_api")
MAX_TEXT_CHARACTERS = int(os.environ.get("MAX_TEXT_CHARACTERS", "20000"))


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)

    @field_validator("text")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace")
        return value


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.predictor = None
    application.state.model_load_error = None
    try:
        application.state.predictor = ParsBERTPredictor.from_directory(default_model_dir())
        logger.info("ParsBERT artifacts loaded on CPU")
    except ArtifactsMissingError as exc:
        application.state.model_load_error = "Model artifacts are missing"
        logger.warning("Model artifacts are not mounted; prediction endpoints are not ready: %s", exc)
    except ArtifactError:
        application.state.model_load_error = "Model artifacts could not be loaded"
        logger.exception("Model artifact loading failed")
    except Exception:
        application.state.model_load_error = "Model artifacts could not be loaded"
        logger.exception("Unexpected model initialization error")
    yield


app = FastAPI(
    title="Digikala Review Prediction API",
    description="ParsBERT predictions for recommendation status and rating band.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    """Liveness check; it stays available even before model files are mounted."""
    return {"status": "ok", "service": "digikala-review-api"}


@app.get("/ready", tags=["service"])
def ready(request: Request) -> dict[str, str]:
    """Readiness check for the model-backed prediction route."""
    if request.app.state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.model_load_error or "Model is not ready",
        )
    return {"status": "ready", "model": "parsbert"}


@app.post("/normalize", tags=["inference"])
def normalize(request_body: TextRequest) -> dict[str, str]:
    """Return the original text and the API-side normalized text."""
    return {
        "original_text": request_body.text,
        "normalized_text": normalize_persian_text(request_body.text),
    }


@app.post("/predict", tags=["inference"])
def predict(
    request_body: TextRequest,
    request: Request,
    model: str = Query(default="parsbert"),
    tokenizer: str = Query(default="parsbert"),
) -> dict[str, Any]:
    """Predict both labels; model and tokenizer are selected by server name."""
    if model != "parsbert":
        raise HTTPException(status_code=400, detail="Unknown model. Supported model: parsbert")
    if tokenizer != "parsbert":
        raise HTTPException(status_code=400, detail="Unknown tokenizer. Supported tokenizer: parsbert")

    predictor: ParsBERTPredictor | None = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.model_load_error or "Model is not ready",
        )

    started = time.perf_counter()
    try:
        result = predictor.predict(request_body.text)
    except TextTooLongError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 500 response without logging review text or request bodies."""
    logger.error("Unhandled API error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

