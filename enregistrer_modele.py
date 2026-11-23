import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-docker")

df = pd.read_csv("churn_data.csv")

X = pd.get_dummies(df.drop(columns=["churn"]), drop_first=True)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
)

with mlflow.start_run(run_name="churn-docker-random-forest") as run:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_test.iloc[:1],
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    registered = mlflow.register_model(
        f"runs:/{run.info.run_id}/model",
        "churn-model",
    )

client = MlflowClient(tracking_uri="http://localhost:5000")
client.set_registered_model_alias(
    "churn-model",
    "champion",
    registered.version,
)

print("Model name: churn-model")
print("Version:", registered.version)
print("Alias: champion")
print("Run ID:", run.info.run_id)
print("Accuracy:", accuracy)
