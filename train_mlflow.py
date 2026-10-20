import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


df = pd.read_csv("churn_data.csv")


def preparer(colonnes):
    X = pd.get_dummies(df[colonnes], drop_first=True)
    y = df["churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    with mlflow.start_run(run_name=nom):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
        )

        print(f"{nom}: accuracy={accuracy:.4f}, f1={f1:.4f}")


mlflow.set_experiment("churn-prediction")

colonnes_originales = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation",
]

lancer_essai("essai-1", 100, 5, colonnes_originales)
lancer_essai("essai-2", 50, 3, colonnes_originales)
lancer_essai(
    "essai-3",
    100,
    5,
    ["age", "abonnement", "anciennete", "consommation"],
)
