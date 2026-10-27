
# ===== 1. Importer les bibliothèques =====
from typing import Literal
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")

# ===== 3. Charger le champion au démarrage =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")

# ===== 4. Décrire les données attendues =====
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
    # Transformer les données en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # Encoder l'abonnement
    ligne = pd.get_dummies(ligne)

    # Adapter les colonnes au modèle
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Effectuer la prédiction
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }