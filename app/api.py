import os
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "modelo_sentimento.joblib"))
FRONTEND = ROOT / "frontend" / "index.html"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="API de Sentimento", version="0.1.0")


class PredictRequest(BaseModel):
    textos: list[str] = Field(..., min_length=1, examples=[["Adorei o produto!"]])


class Prediction(BaseModel):
    texto: str
    sentimento: str
    confianca: float | None = None


class PredictResponse(BaseModel):
    predicoes: list[Prediction]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    preds = model.predict(req.textos)
    probas = model.predict_proba(req.textos) if hasattr(model, "predict_proba") else None

    return PredictResponse(
        predicoes=[
            Prediction(
                texto=texto,
                sentimento=str(pred),
                confianca=float(probas[i].max()) if probas is not None else None,
            )
            for i, (texto, pred) in enumerate(zip(req.textos, preds))
        ]
    )
