
# ===== 1. Importer les bibliothèques =====
import mlflow
import mlflow.sklearn

# ===== 2. Se connecter à la base locale du Registry =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Charger le modèle champion =====
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

# ===== 4. Exporter le modèle dans le dossier model/ =====
mlflow.sklearn.save_model(
    model,
    "model",
    serialization_format="cloudpickle"
)

print("Champion exporté dans le dossier model/")
