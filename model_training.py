import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)


def train_traffic_model(data, model_dir):
    features = ["avg_speed", "occupancy", "road_capacity"]
    target = "traffic_volume"

    dataset = data[features + [target]].dropna()
    dataset = dataset[dataset[target] >= 0]

    X = dataset[features]
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2 Score": r2_score(y_test, predictions)
    }

    joblib.dump(model, model_dir / "traffic_model.pkl")

    return model, metrics


def train_congestion_model(data, model_dir):
    features = [
        "traffic_volume",
        "avg_speed",
        "occupancy",
        "road_capacity"
    ]

    dataset = data[features + ["congestion_level"]].dropna()

    X = dataset[features]
    y = dataset["congestion_level"].astype(str)

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Insufficient congestion classes for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20,
        random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions)
    }

    joblib.dump(model, model_dir / "congestion_model.pkl")

    return model, metrics


def train_travel_time_model(data, model_dir):
    features = [
        "traffic_volume",
        "avg_speed",
        "occupancy",
        "road_capacity"
    ]

    dataset = data[features + ["travel_time"]].dropna()
    dataset = dataset[dataset["travel_time"] >= 0]

    X = dataset[features]
    y = dataset["travel_time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "R2 Score": r2_score(y_test, predictions)
    }

    joblib.dump(model, model_dir / "travel_time_model.pkl")

    return model, metrics


def train_accident_model(data, model_dir):
    features = [
        "traffic_volume",
        "avg_speed",
        "occupancy",
        "road_capacity"
    ]

    dataset = data[features + ["accident_count"]].dropna()

    dataset["accident_risk"] = (
        dataset["accident_count"] > 0
    ).astype(int)

    X = dataset[features]
    y = dataset["accident_risk"]

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Insufficient accident classes for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20,
        random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions)
    }

    joblib.dump(model, model_dir / "accident_risk_model.pkl")

    return model, metrics


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]

    data_path = (
        project_dir / "data_and_documents"
        / "traffic_sensor_log.csv"
    )

    model_dir = project_dir / "models"
    model_dir.mkdir(exist_ok=True)

    data = pd.read_csv(data_path)

    data.columns = data.columns.str.strip()

    for column in [
        "traffic_volume", "avg_speed", "occupancy",
        "road_capacity", "travel_time", "accident_count"
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    results = {}

    training_functions = {
        "Traffic Volume": train_traffic_model,
        "Congestion": train_congestion_model,
        "Travel Time": train_travel_time_model,
        "Accident Risk": train_accident_model
    }

    for name, function in training_functions.items():
        try:
            _, metrics = function(data, model_dir)
            results[name] = metrics
            print(f"{name} model trained successfully.")
            print(metrics)
        except Exception as error:
            print(f"{name} model could not be trained: {error}")

    print("\nModel training process completed.")

