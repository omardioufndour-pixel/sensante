# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP/UCAD

from pathlib import Path
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np


class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    frissons: bool = Field(...)
    nausee: bool = Field(...)
    region: str = Field(...)


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str


app = FastAPI(title="SenSante API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger
logger = logging.getLogger("sensante")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
LE_SEXE_PATH = MODEL_DIR / "encoder_sexe.pkl"
LE_REGION_PATH = MODEL_DIR / "encoder_region.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.pkl"

# Model artifacts (loaded on startup)
model = None
le_sexe = None
le_region = None
feature_cols = None


@app.on_event("startup")
def load_model():
    """Load model artifacts at application startup."""
    global model, le_sexe, le_region, feature_cols
    logger.info("Chargement des artefacts du modele depuis %s", MODEL_DIR)
    try:
        model = joblib.load(MODEL_PATH)
        le_sexe = joblib.load(LE_SEXE_PATH)
        le_region = joblib.load(LE_REGION_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
    except Exception as e:
        logger.exception("Erreur lors du chargement du modele: %s", e)
        # Raising here will prevent the app from starting correctly.
        raise RuntimeError(f"Impossible de charger les artefacts du modele: {e}")

    logger.info("Modele charge: %s", type(model).__name__)
    if hasattr(model, "classes_"):
        logger.info("Classes: %s", list(model.classes_))


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    if model is None:
        logger.error("Prediction appelee mais le modele n'est pas charge")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0, confiance="aucune", message=f"Sexe invalide : {patient.sexe}")

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0, confiance="aucune", message=f"Region inconnue : {patient.region}")

    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        int(patient.frissons),
        int(patient.nausee),
        region_enc,
    ]])

    try:
        diagnostic = model.predict(features)[0]
        proba_max = float(model.predict_proba(features)[0].max())
    except Exception as e:
        logger.exception("Erreur durant la prediction: %s", e)
        raise HTTPException(status_code=500, detail="Prediction error")

    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    messages = {
        "paludisme": "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typhoide": "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain": "Pas de pathologie detectee. Continuez a surveiller.",
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin."),
    )
