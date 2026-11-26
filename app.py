import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="FlowCast Analytics",
    layout="wide"
)

@st.cache_resource
def load_models():
    return {
        "traffic": joblib.load(MODEL_DIR / "traffic_model.pkl"),
        "congestion": joblib.load(MODEL_DIR / "congestion_model.pkl"),
        "travel": joblib.load(MODEL_DIR / "travel_time_model.pkl"),
        "accident": joblib.load(MODEL_DIR / "accident_risk_model.pkl")
    }

try:
    models = load_models()
except Exception as error:
    st.error(f"Unable to load models: {error}")
    st.stop()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.title("FlowCast")
st.subheader("Smart Traffic Analytics and Prediction Platform")
st.write(
    "Machine learning based traffic monitoring and prediction "
    "for volume, congestion, travel time and accident risk."
)

st.success("All four prediction models are loaded.")

st.divider()

st.header("Traffic Input Parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_speed = st.number_input(
        "Average Speed",
        min_value=0.0,
        value=40.0,
        step=1.0
    )

with col2:
    occupancy = st.number_input(
        "Road Occupancy",
        min_value=0.0,
        value=30.0,
        step=1.0
    )

with col3:
    road_capacity = st.number_input(
        "Road Capacity",
        min_value=1.0,
        value=1000.0,
        step=10.0
    )

with col4:
    traffic_volume = st.number_input(
        "Current Traffic Volume",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

st.divider()

if st.button("Generate Traffic Predictions", type="primary"):

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

    try:
        predicted_volume = models["traffic"].predict(
            traffic_input
        )[0]

        predicted_congestion = models["congestion"].predict(
            other_input
        )[0]

        predicted_travel_time = models["travel"].predict(
            other_input
        )[0]

        accident_prediction = models["accident"].predict(
            other_input
        )[0]

        if hasattr(models["accident"], "predict_proba"):
            accident_probability = (
                models["accident"].predict_proba(other_input)[0][1]
                * 100
            )
        else:
            accident_probability = None

        result = {
            "Average Speed": avg_speed,
            "Occupancy": occupancy,
            "Road Capacity": road_capacity,
            "Input Traffic Volume": traffic_volume,
            "Predicted Traffic Volume": round(float(predicted_volume), 2),
            "Predicted Congestion": str(predicted_congestion),
            "Predicted Travel Time": round(float(predicted_travel_time), 2),
            "Accident Risk Class": int(accident_prediction)
        }

        if accident_probability is not None:
            result["Accident Class Probability (%)"] = round(
                float(accident_probability), 2
            )

        st.session_state.prediction_history.append(result)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

if st.session_state.prediction_history:

    latest = st.session_state.prediction_history[-1]

    st.header("Traffic Prediction Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Predicted Traffic Volume",
        f"{latest['Predicted Traffic Volume']:.2f}"
    )

    col2.metric(
        "Congestion Level",
        latest["Predicted Congestion"]
    )

    col3.metric(
        "Estimated Travel Time",
        f"{latest['Predicted Travel Time']:.2f}"
    )

    col4.metric(
        "Accident Risk Class",
        "Risk detected" if latest["Accident Risk Class"] == 1
        else "No risk detected"
    )

    st.divider()

    st.header("Traffic Analysis")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Input and Predicted Traffic Volume")

        volume_chart = pd.DataFrame({
            "Traffic Volume": [
                latest["Input Traffic Volume"],
                latest["Predicted Traffic Volume"]
            ]
        }, index=["Input", "Predicted"])

        st.bar_chart(volume_chart)

    with chart_col2:
        st.subheader("Traffic Input Parameters")

        input_chart = pd.DataFrame({
            "Value": [
                latest["Average Speed"],
                latest["Occupancy"],
                latest["Road Capacity"]
            ]
        }, index=[
            "Average Speed",
            "Occupancy",
            "Road Capacity"
        ])

        st.bar_chart(input_chart)

    st.divider()

    st.header("Accident Risk Assessment")

    if latest["Accident Risk Class"] == 1:
        st.warning(
            "The model classified this input as belonging to "
            "the accident-recorded class."
        )
    else:
        st.success(
            "The model classified this input as belonging to "
            "the no-accident-recorded class."
        )

    if "Accident Class Probability (%)" in latest:
        st.write(
            "Model probability for the accident-recorded class:",
            f"{latest['Accident Class Probability (%)']:.2f}%"
        )

    st.caption(
        "This is a model classification based on historical training "
        "data, not a verified forecast of a future accident."
    )

    st.divider()

    st.header("Prediction History")

    history = pd.DataFrame(st.session_state.prediction_history)

    st.dataframe(
        history,
        use_container_width=True
    )

    st.download_button(
        "Download Prediction History",
        data=history.to_csv(index=False),
        file_name="flowcast_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info(
        "Enter traffic parameters above and generate a prediction "
        "to view the analytics dashboard."
    )

st.divider()

st.header("Project Information")

info1, info2, info3 = st.columns(3)

info1.write("**Machine Learning Algorithm**")
info1.write("Random Forest")

info2.write("**Prediction Models**")
info2.write("Four trained models")

info3.write("**Platform**")
info3.write("Python and Streamlit")

st.caption("FlowCast | Smart Traffic Analytics Platform")
