import os

# ===== 1. Importer les bibliothÃ¨ques =====
from typing import Literal
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ===== 2. CrÃ©er l'application =====
app = FastAPI(title="API Churn")

# ===== 3. Charger le champion au dÃ©marrage =====
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")
mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.sklearn.load_model(MODEL_URI)
print(f"Modèle chargé depuis : {MODEL_URI}")

# ===== 4. DÃ©crire les donnÃ©es attendues =====
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float

# ===== 5. Route GET /health =====
@app.get("/health")
def health():
    return {"status": "ok"}

# ===== 6. Route POST /predict =====
@app.post("/predict")
def predict(client: Client):
    # Transformer les donnÃ©es en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # Encoder l'abonnement
    ligne = pd.get_dummies(ligne)

    # Adapter les colonnes au modÃ¨le
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Effectuer la prÃ©diction
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }
