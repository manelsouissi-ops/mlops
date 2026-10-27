
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 1. Retrouver le run essai-2
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)
run_id = runs.iloc[0]["run_id"]

# 2. Enregistrer essai-2 comme nouvelle version
v = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)
print("Nouvelle version creee :", v.version)

# 3. Identifier le champion actuel
actuel = client.get_model_version_by_alias(
    "churn-model", "champion"
)
print("Champion actuel : version", actuel.version)

# 4. Déplacer l'alias vers la nouvelle version
client.set_registered_model_alias(
    "churn-model", "champion", v.version
)
print(
    "Apres deplacement : champion -> version",
    client.get_model_version_by_alias(
        "churn-model", "champion"
    ).version
)

# 5. Restaurer le champion initial
client.set_registered_model_alias(
    "churn-model", "champion", actuel.version
)
print(
    "Apres retour : champion -> version",
    client.get_model_version_by_alias(
        "churn-model", "champion"
    ).version
)
