from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Hotel Booking Cancellation",
    page_icon="\U0001F3E1",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model_path = Path(__file__).resolve().parent / "models"

    with open(model_path / "pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open(model_path / "threshold.pkl", "rb") as f:
        threshold = pickle.load(f)

    return pipeline, threshold


pipeline, threshold = load_artifacts()


# Verified results from the untouched test set (notebook, final evaluation cell)
TEST_METRICS = {
    "Accuracy": 0.775,
    "Precision": 0.572,
    "Recall": 0.706,
    "F1": 0.632,
    "ROC-AUC": 0.841,
    "Brier score": 0.136,
}

KEY_FINDINGS = [
    "City Hotel bookings cancel more often than Resort Hotel bookings.",
    "Canceled bookings tend to have longer lead times and higher daily rates.",
    "Cancellation rates peak in July and August and are lowest in November.",
    "Online TA is the largest booking segment and cancels more than other major segments.",
    "More special requests and repeat-guest history are both linked to lower cancellation risk.",
]


st.title("\U0001F3E1 Hotel Booking Cancellation Prediction")
st.write(
    "Estimate the probability that a hotel booking will be canceled, based on the booking details below."
)

st.divider()

# --- Model performance and data insights ---
col_info, col_metrics = st.columns([2, 1])

with col_info:
    st.subheader("What the model does")
    st.write(
        "The model is a trained XGBoost pipeline that predicts whether a booking will be canceled. "
        "It was evaluated once on a test set it had never seen during training or tuning."
    )
    st.subheader("Key findings from the data")
    for finding in KEY_FINDINGS:
        st.write(f"\u2022 {finding}")

with col_metrics:
    st.subheader("Model performance (test set)")
    for name, value in TEST_METRICS.items():
        if name == "Brier score":
            st.metric(name, f"{value:.3f}")
        else:
            st.metric(name, f"{value:.3f}")
    st.write(
        "Threshold: "
        f"{threshold:.1%}  (bookings above this probability are predicted to cancel)"
    )
    st.caption(
        "Recall and F1 are the most useful measures here because most bookings "
        "are not canceled, so accuracy alone can be misleading."
    )

st.divider()

# --- Prediction form ---
st.subheader("Enter booking details")

with st.form("booking_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
        lead_time = st.number_input(
            "Lead Time (days)", min_value=0, max_value=737, value=30
        )
        arrival_date_year = st.selectbox("Arrival Year", [2015, 2016, 2017], index=2)
        stays_in_weekend_nights = st.number_input(
            "Weekend Nights", min_value=0, max_value=19, value=1
        )
        stays_in_week_nights = st.number_input(
            "Week Nights", min_value=0, max_value=50, value=2
        )
        meal = st.selectbox("Meal", ["BB", "HB", "SC", "FB", "Undefined"])

    with col2:
        adults = st.number_input("Adults", min_value=1, max_value=55, value=2)
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        babies = st.number_input("Babies", min_value=0, max_value=10, value=0)
        market_segment = st.selectbox(
            "Market Segment",
            [
                "Online TA",
                "Offline TA/TO",
                "Direct",
                "Groups",
                "Corporate",
                "Complementary",
                "Aviation",
                "Undefined",
            ],
        )
        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
        )
        deposit_type = st.selectbox(
            "Deposit Type", ["No Deposit", "Non Refund", "Refundable"]
        )

    with col3:
        is_repeated_guest = st.selectbox("Repeated Guest", ["No", "Yes"])
        previous_bookings_not_canceled = st.number_input(
            "Previous Bookings Not Canceled", min_value=0, max_value=72, value=0
        )
        booking_changes = st.number_input(
            "Booking Changes", min_value=0, max_value=21, value=0
        )
        adr = st.number_input(
            "Average Daily Rate", min_value=0.0, max_value=510.0, value=100.0
        )
        required_car_parking_spaces = st.number_input(
            "Required Parking Spaces", min_value=0, max_value=4, value=0
        )
        total_of_special_requests = st.number_input(
            "Total Special Requests", min_value=0, max_value=5, value=0
        )
        customer_type = st.selectbox(
            "Customer Type",
            ["Transient", "Transient-Party", "Contract", "Group"],
        )

    predict_button = st.form_submit_button("Predict Cancellation")


if predict_button:
    input_data = pd.DataFrame(
        [
            {
                "hotel": hotel,
                "lead_time": lead_time,
                "arrival_date_year": arrival_date_year,
                "stays_in_weekend_nights": stays_in_weekend_nights,
                "stays_in_week_nights": stays_in_week_nights,
                "adults": adults,
                "children": children,
                "babies": babies,
                "meal": meal,
                "market_segment": market_segment,
                "distribution_channel": distribution_channel,
                "is_repeated_guest": 1 if is_repeated_guest == "Yes" else 0,
                "previous_bookings_not_canceled": previous_bookings_not_canceled,
                "booking_changes": booking_changes,
                "deposit_type": deposit_type,
                "customer_type": customer_type,
                "adr": adr,
                "required_car_parking_spaces": required_car_parking_spaces,
                "total_of_special_requests": total_of_special_requests,
            }
        ]
    )

    input_data["total_nights"] = (
        input_data["stays_in_weekend_nights"] + input_data["stays_in_week_nights"]
    )
    input_data["total_guests"] = (
        input_data["adults"] + input_data["children"] + input_data["babies"]
    )

    cancellation_probability = float(pipeline.predict_proba(input_data)[0, 1])
    prediction = int(cancellation_probability >= threshold)

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Cancellation Probability", f"{cancellation_probability:.2%}")
    result_col2.metric("Decision Threshold", f"{threshold:.2%}")

    if prediction == 1:
        st.error("This booking is likely to be canceled.")
    else:
        st.success("This booking is not likely to be canceled.")

    st.progress(int(np.clip(cancellation_probability * 100, 0, 100)))
