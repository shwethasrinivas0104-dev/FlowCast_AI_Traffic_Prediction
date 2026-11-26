import joblib
import pandas as pd
from pathlib import Path


class FlowCastPredictor:

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.models = self.load_models()

    def load_models(self):
        model_files = {
            "traffic": "traffic_model.pkl",
            "congestion": "congestion_model.pkl",
            "travel_time": "travel_time_model.pkl",
            "accident_risk": "accident_risk_model.pkl"
        }

        loaded_models = {}

        for name, filename in model_files.items():
            model_path = self.model_dir / filename

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )

            loaded_models[name] = joblib.load(model_path)

        return loaded_models

    def predict(self, avg_speed, occupancy, road_capacity,
                traffic_volume):

        traffic_input = pd.DataFrame({
            "avg_speed": [avg_speed],
            "occupancy": [occupancy],
            "road_capacity": [road_capacity]
        })

        other_input = pd.DataFrame({
            "traffic_volume": [traffic_volume],
            "avg_speed": [avg_speed],
            "occupancy": [occupancy],
            "road_capacity": [road_capacity]
        })

        results = {
            "predicted_traffic_volume": float(
                self.models["traffic"].predict(traffic_input)[0]
            ),
            "predicted_congestion": str(
                self.models["congestion"].predict(other_input)[0]
            ),
            "predicted_travel_time": float(
                self.models["travel_time"].predict(other_input)[0]
            ),
            "accident_risk_class": int(
                self.models["accident_risk"].predict(other_input)[0]
            )
        }

        if hasattr(self.models["accident_risk"], "predict_proba"):
            results["accident_class_probability"] = float(
                self.models["accident_risk"]
                .predict_proba(other_input)[0][1]
            )

        return results


if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parents[1]
    model_dir = project_dir / "models"

    predictor = FlowCastPredictor(model_dir)

    prediction = predictor.predict(
        avg_speed=40,
        occupancy=30,
        road_capacity=1000,
        traffic_volume=500
    )

    print("FlowCast Prediction Results")
    print("----------------------------")

    for name, value in prediction.items():
        print(f"{name}: {value}")

