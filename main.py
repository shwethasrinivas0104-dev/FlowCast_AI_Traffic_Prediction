import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="FlowCast",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 FlowCast")
st.subheader("AI-Powered Smart Traffic Flow Prediction Platform")

st.write(
    "Welcome to FlowCast! This platform will analyse "
    "traffic, weather, and calendar data to predict "
    "traffic conditions."
)
# Load datasets
traffic_data = pd.read_csv("data_and_documents/traffic_sensor_log.csv")
weather_data = pd.read_csv("data_and_documents/weather_observations.csv")
calendar_data = pd.read_csv("data_and_documents/calendar_events.csv")
st.write("Traffic Dataset Columns:", traffic_data.columns.tolist())
st.write("Weather Dataset Columns:", weather_data.columns.tolist())
st.write("Calendar Dataset Columns:", calendar_data.columns.tolist())

# Display dataset information
st.header("Dataset Overview")

st.write("Traffic Sensor Data")
st.dataframe(traffic_data.head())

st.write("Weather Data")
st.dataframe(weather_data.head())

st.write("Calendar Events")
st.dataframe(calendar_data.head())

# Data Quality Check
st.header("Data Quality Check")

datasets = {
    "Traffic Data": traffic_data,
    "Weather Data": weather_data,
    "Calendar Data": calendar_data
}

for name, data in datasets.items():
    st.subheader(name)

    st.write("Missing Values:")
    st.write(data.isnull().sum())

    st.write("Duplicate Rows:")
    st.write(data.duplicated().sum())
st.header("Dataset Summary")

for name, data in datasets.items():
    st.subheader(name)

    st.write("Number of Rows:", data.shape[0])
    st.write("Number of Columns:", data.shape[1])

    st.header("Data Types and Column Information")

for name, data in datasets.items():
    st.subheader(name)

    st.write("Column Names:")
    st.write(data.columns.tolist())

    st.write("Data Types:")
    st.write(data.dtypes.astype(str))

    st.header("Duplicate Rows Analysis")

for name, data in datasets.items():
    st.subheader(name)
    st.write("Duplicate Rows:", data.duplicated().sum())

st.header("Empty Rows Analysis")

for name, data in datasets.items():
    st.subheader(name)
    st.write("Completely Empty Rows:", data.isna().all(axis=1).sum())
    

st.subheader("Traffic Data")
st.dataframe(traffic_data.isnull().sum())

st.subheader("Weather Data")
st.dataframe(weather_data.isnull().sum())

st.subheader("Calendar Data")
st.dataframe(calendar_data.isnull().sum())

# Data Cleaning
st.header("Data Cleaning")

cleaned_datasets = {}

for name, data in datasets.items():
    cleaned_data = data.copy()

    cleaned_data = cleaned_data.drop_duplicates()
    cleaned_data = cleaned_data.dropna(how="all")

    cleaned_datasets[name] = cleaned_data

    st.subheader(name)
    st.write("Original Rows:", len(data))
    st.write("Cleaned Rows:", len(cleaned_data))
    st.dataframe(cleaned_data.head())

  # Data Type Conversion
    st.header("Data Type Conversion")

for name, data in cleaned_datasets.items():
    st.subheader(name)

    st.write("Column Data Types")
    st.dataframe(data.dtypes.astype(str).to_frame("Data Type"))

    st.write("Dataset Information")
    st.write("Rows:", data.shape[0])
    st.write("Columns:", data.shape[1])

# Date Feature Engineering
st.header("Date Feature Engineering")

for name, data in cleaned_datasets.items():
    st.subheader(name)

    date_columns = [
        col for col in data.columns
        if "date" in col.lower()
    ]

    if date_columns:
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
            data[col + "_year"] = data[col].dt.year
            data[col + "_month"] = data[col].dt.month
            data[col + "_day"] = data[col].dt.day
            data[col + "_dayofweek"] = data[col].dt.dayofweek

        st.success("Date features created successfully!")
        st.dataframe(data.head())
    else:
        st.info("No date column found in this dataset.")

# Exploratory Data Analysis (EDA)

st.header("Exploratory Data Analysis")

for name, data in cleaned_datasets.items():
    st.subheader(name)

    st.write("Dataset Shape:")
    st.write("Rows:", data.shape[0])
    st.write("Columns:", data.shape[1])

    st.write("Statistical Summary:")
    st.dataframe(data.describe(include="all").T)

# Traffic Volume Analysis

st.header("Traffic Volume Analysis")

for name, data in cleaned_datasets.items():
    if "traffic" in name.lower():
        if "date" in data.columns and "traffic_volume" in data.columns:

            traffic_trend = data.copy()

            traffic_trend["date"] = pd.to_datetime(
                traffic_trend["date"], errors="coerce"
            )

            traffic_trend["traffic_volume"] = pd.to_numeric(
                traffic_trend["traffic_volume"], errors="coerce"
            )

            traffic_trend = traffic_trend.dropna(
                subset=["date", "traffic_volume"]
            )

            daily_traffic = traffic_trend.groupby(
                "date"
            )["traffic_volume"].mean()

            st.subheader("Daily Average Traffic Volume")
            st.line_chart(daily_traffic)

        else:
            st.warning("Required traffic columns not found.")

# Traffic Volume by Time
st.header("Traffic Volume by Time")

if "time" in traffic_data.columns and "traffic_volume" in traffic_data.columns:

    traffic_data["time"] = traffic_data["time"].astype(str)

    hourly_traffic = traffic_data.groupby("time")[
        "traffic_volume"
    ].mean().sort_index()

    st.subheader("Average Traffic Volume by Time")
    st.line_chart(hourly_traffic)

else:
    st.warning("Required columns not found.")

# Traffic Volume by Day of the Week

st.header("Traffic Volume by Day of the Week")

if "date" in traffic_data.columns and "traffic_volume" in traffic_data.columns:

    weekday_data = traffic_data.copy()

    weekday_data["date"] = pd.to_datetime(
        weekday_data["date"], errors="coerce"
    )

    weekday_data["day_name"] = weekday_data["date"].dt.day_name()

    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    daily_traffic = (
        weekday_data.groupby("day_name")["traffic_volume"]
        .mean()
        .reindex(weekday_order)
    )

    st.subheader("Average Traffic Volume by Weekday")
    st.bar_chart(daily_traffic)

else:
    st.warning("Required date or traffic volume columns not found.")

    # Weather Impact Analysis

st.header("Weather Impact Analysis")

st.subheader("Weather Data Overview")
st.write("Weather Dataset Shape:", weather_data.shape)

st.subheader("Weather Data Summary")
st.dataframe(weather_data.describe(include="all").T)

st.subheader("Weather Missing Values")
st.dataframe(weather_data.isnull().sum())

st.subheader("Weather Correlation Analysis")

weather_numeric = weather_data.select_dtypes(include="number")

if not weather_numeric.empty:
    st.dataframe(weather_numeric.corr())
    st.bar_chart(weather_numeric.corr())
else:
    st.info("No numeric weather columns available.")

# Traffic and Weather Data Integration

st.header("Traffic and Weather Data Integration")

traffic_merge = traffic_data.copy()
weather_merge = weather_data.copy()

traffic_merge["weather_station_id"] = (
    traffic_merge["weather_station_id"].astype(str).str.strip()
)
weather_merge["station_id"] = (
    weather_merge["station_id"].astype(str).str.strip()
)

traffic_merge["date"] = pd.to_datetime(
    traffic_merge["date"], errors="coerce"
).dt.date

weather_merge["date"] = pd.to_datetime(
    weather_merge["date"], errors="coerce"
).dt.date

traffic_merge["time"] = traffic_merge["time"].astype(str).str.strip()
weather_merge["time"] = weather_merge["time"].astype(str).str.strip()

weather_merge = weather_merge.drop_duplicates(
    subset=["station_id", "date", "time"]
)

combined_data = pd.merge(
    traffic_merge,
    weather_merge,
    left_on=["weather_station_id", "date", "time"],
    right_on=["station_id", "date", "time"],
    how="inner"
)

st.subheader("Integrated Dataset")
st.write("Combined Dataset Shape:", combined_data.shape)
st.dataframe(combined_data.head())

st.success("Traffic and weather integration completed!")

# Traffic Pattern Analysis

st.header("Traffic Pattern Analysis")

if "traffic_volume" in combined_data.columns:

    st.subheader("Traffic Volume Overview")

    st.write(
        "Average Traffic Volume:",
        combined_data["traffic_volume"].mean()
    )

    st.write(
        "Maximum Traffic Volume:",
        combined_data["traffic_volume"].max()
    )

    st.subheader("Traffic Volume by Road")

    if "road_name" in combined_data.columns:
        road_traffic = combined_data.groupby(
            "road_name"
        )["traffic_volume"].mean()

        st.bar_chart(road_traffic)

else:
    st.warning("Traffic volume column not found.")

# Congestion Level Analysis

st.header("Congestion Level Analysis")

if "congestion_level" in combined_data.columns:

    st.subheader("Congestion Distribution")

    congestion_counts = combined_data[
        "congestion_level"
    ].value_counts()

    st.dataframe(congestion_counts)

    st.bar_chart(congestion_counts)

else:
    st.warning("Congestion level column not found.")

# Travel Time Analysis

st.header("Travel Time Analysis")

if "travel_time" in combined_data.columns:

    combined_data["travel_time"] = pd.to_numeric(
        combined_data["travel_time"], errors="coerce"
    )

    travel_data = combined_data.dropna(subset=["travel_time"])

    if not travel_data.empty:

        st.subheader("Travel Time Overview")

        st.write(
            "Average Travel Time:",
            round(travel_data["travel_time"].mean(), 2)
        )

        st.write(
            "Maximum Travel Time:",
            round(travel_data["travel_time"].max(), 2)
        )

        if "road_name" in travel_data.columns:

            st.subheader("Average Travel Time by Road")

            road_time = travel_data.groupby(
                "road_name"
            )["travel_time"].mean()

            st.bar_chart(road_time)

    else:
        st.info("No valid travel time data available.")

else:
    st.warning("Travel time column not found.")

# Machine Learning - Traffic Volume Prediction

st.header("Traffic Volume Prediction")

ml_data = combined_data.copy()

required_columns = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "road_capacity"
]

if all(col in ml_data.columns for col in required_columns):

    ml_data = ml_data[required_columns].dropna()

    ml_data = ml_data[ml_data["traffic_volume"] >= 0]
    st.write("ML Dataset Shape:", ml_data.shape)
    st.dataframe(ml_data.head())

    # Split features and target
    X = ml_data[["avg_speed", "occupancy", "road_capacity"]]
    y = ml_data["traffic_volume"]
   
        # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/traffic_model.pkl")
    st.success("Trained model saved successfully!")


    # Feature Importance Analysis
    st.subheader("Feature Importance")

    feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

    feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

    st.dataframe(feature_importance)
    st.bar_chart(feature_importance.set_index("Feature"))


        # Make predictions
    predictions = model.predict(X_test)

    # Prediction Error Analysis
    st.subheader("Prediction Error Analysis")

    error_data = pd.DataFrame({
    "Actual Traffic": y_test.values,
    "Predicted Traffic": predictions
})

    error_data["Prediction Error"] = (
    error_data["Actual Traffic"] - error_data["Predicted Traffic"]
)

    st.dataframe(error_data.head(20))

    st.write(
    "Average Prediction Error:",
    round(error_data["Prediction Error"].mean(), 2)
)

    st.bar_chart(error_data["Prediction Error"].head(20))

    # Residual Distribution
    st.subheader("Prediction Error Distribution")
    st.bar_chart(
    error_data["Prediction Error"].value_counts().sort_index()
)
        # Evaluate model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    st.subheader("Model Performance")
    st.write("Mean Absolute Error:", round(mae, 2))
    st.write("Root Mean Squared Error:", round(rmse, 2))
    st.write("R2 Score:", round(r2, 2))

    st.subheader("Actual vs Predicted Traffic")

    comparison = pd.DataFrame({
    "Actual Traffic": y_test.values[:20],
    "Predicted Traffic": predictions[:20]
})
    st.subheader("Traffic Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Traffic", round(ml_data["traffic_volume"].mean(), 2))

    col2.metric("Highest Traffic", ml_data["traffic_volume"].max())

    col3.metric("Lowest Traffic", ml_data["traffic_volume"].min())


    st.line_chart(comparison)
    # Download Traffic Prediction Results
    csv = comparison.to_csv(index=False).encode("utf-8")

    st.download_button(
    label="Download Traffic Results",
    data=csv,
    file_name="traffic_prediction_results.csv",
    mime="text/csv"
)
    st.subheader("Traffic Volume Distribution")

    st.bar_chart(
    ml_data["traffic_volume"].value_counts().head(10)
)
    st.subheader("Predict Traffic Volume")

    avg_speed_input = st.number_input("Average Speed", min_value=0.0, value=40.0)

    occupancy_input = st.number_input("Occupancy (%)", min_value=0.0, max_value=100.0, value=50.0)

    road_capacity_input = st.number_input("Road Capacity", min_value=1.0, value=2000.0)

    if st.button("Predict Traffic"):
        new_data = pd.DataFrame(
            [[avg_speed_input, occupancy_input, road_capacity_input]],
            columns=["avg_speed", "occupancy", "road_capacity"]
        )

        prediction = model.predict(new_data)
        st.success(f"Predicted Traffic Volume: {round(prediction[0], 2)}")

else:
    st.warning("Required columns are missing for traffic prediction.")



