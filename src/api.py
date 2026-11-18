from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import Predictor


predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_models()
    yield


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    text: str
    model: Literal["tfidf", "transformer", "semi_supervised"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predictor.predict(request.text, request.model)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
