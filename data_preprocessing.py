import pandas as pd
from pathlib import Path


def load_traffic_data(file_path):
    data = pd.read_csv(file_path)
    return data


def clean_traffic_data(data):
    data = data.copy()

    data.columns = data.columns.str.strip()

    data = data.drop_duplicates()
    data = data.dropna(how="all")

    numeric_columns = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "accident_count",
        "road_capacity"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    if "date" in data.columns:
        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce"
        )

    return data


def create_date_features(data):
    data = data.copy()

    if "date" in data.columns:
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month
        data["day"] = data["date"].dt.day
        data["day_of_week"] = data["date"].dt.dayofweek

    if "time" in data.columns:
        parsed_time = pd.to_datetime(
            data["time"],
            format="mixed",
            errors="coerce"
        )
        data["hour"] = parsed_time.dt.hour

    return data


def prepare_traffic_data(file_path):
    data = load_traffic_data(file_path)
    data = clean_traffic_data(data)
    data = create_date_features(data)

    return data


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]
    file_path = project_dir / "data_and_documents" / "traffic_sensor_log.csv"

    traffic_data = prepare_traffic_data(file_path)

    print("Traffic data preprocessing completed.")
    print("Dataset shape:", traffic_data.shape)
    print(traffic_data.head())

