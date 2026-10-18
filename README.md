# Hotel Booking Cancellation Prediction

A data analysis and machine learning project that explores what drives hotel booking cancellations and predicts whether a booking will be canceled, with a Streamlit app for making predictions on individual bookings.

## About the data

The dataset (`data/hotel_bookings.csv`) contains 119,390 hotel bookings from a city hotel and a resort hotel, covering 2015–2017. Each row is one booking with 32 fields, and the target is `is_canceled` (1 = canceled, 0 = not canceled). Raw cancellation rate is about 37%.

Source: Hotel Booking Demand dataset (Austria-based hotels), a widely used public dataset.

## What the project does

1. **Data cleaning** (`notebooks/hotel_booking_demand.ipynb`): removes exact duplicates, handles missing values, drops invalid bookings (zero guests, out-of-range average daily rate), and removes leakage columns (`reservation_status`, `reservation_status_date`) that directly reveal the outcome. Cleaned data saved to `data/cleaned_hotel_bookings.csv` (86,943 rows; 27.31% canceled).
2. **Exploratory analysis**: cancellation rates by hotel type, lead time, ADR, arrival month, market segment, deposit type, special requests, and repeated-guest status, plus numeric correlations.
3. **Modeling**: an XGBoost pipeline (preprocessing, feature selection, scaling, and model all fitted inside cross-validation folds) tuned on development data, with the decision threshold selected on a reserved validation set and final performance measured once on an untouched test set.
4. **Prediction app** (`app.py`): a Streamlit form that takes booking details and returns a cancellation probability and decision.

## Key findings

- City Hotel bookings cancel at a higher rate than Resort Hotel bookings.
- Canceled bookings tend to have longer lead times and higher average daily rates.
- Cancellation rates peak in July and August and are lowest in November.
- Online TA is the largest booking segment and cancels more often than the other major segments.
- Non-refundable deposits are associated with a relatively high cancellation rate despite the deposit.
- More special requests and repeated-guest history are both associated with lower cancellation risk.
- Lead time and previous cancellations show the strongest positive relationships with cancellation; pairwise correlation is used as a screening aid, not a causal claim.

## Model performance

Models were compared with stratified 3-fold cross-validation. XGBoost was selected because it had the strongest cross-validation ROC-AUC while remaining competitive on F1. Hyperparameters (`max_depth`, `min_child_weight`) were tuned on development data; calibration did not improve the Brier score, so raw XGBoost probabilities were kept. The decision threshold (0.303) was chosen on validation data to maximize F1.

Final results on the untouched test set:

| Metric | Score |
|---|---|
| Accuracy | 0.775 |
| Precision | 0.572 |
| Recall | 0.706 |
| F1 | 0.632 |
| ROC-AUC | 0.841 |
| Brier score | 0.136 |

Accuracy is high partly because most bookings are not canceled. Recall and F1 are the more useful measures for catching cancellations.

## Project structure

```
hotel-booking-cancellation-project/
├── app.py                      # Streamlit prediction app
├── README.md
├── requirements.txt            # Pinned dependencies
├── .gitignore
├── PROJECT_WALKTHROUGH.md      # Plain-language guide to the whole project
├── data/
│   ├── hotel_bookings.csv      # Raw dataset
│   └── cleaned_hotel_bookings.csv  # Cleaned dataset produced by the notebook
├── models/
│   ├── pipeline.pkl            # Fitted prediction pipeline
│   └── threshold.pkl           # Decision threshold used by the app
└── notebooks/
    └── hotel_booking_demand.ipynb  # Full analysis, EDA, and model training
```

## How to run

Install dependencies:

```
pip install -r requirements.txt
```

Run the prediction app:

```
streamlit run app.py
```

The app loads `models/pipeline.pkl` and `models/threshold.pkl`, so no retraining is needed to demo it. To retrain the model or reproduce the cleaning and analysis, run the notebook `notebooks/hotel_booking_demand.ipynb` (it writes the cleaned CSV and pickles the pipeline and threshold).

## Limitations

- The data covers two hotels in one region from 2015–2017, so results may not generalize to other markets or time periods.
- The model predicts cancellation probability, not revenue impact; it does not account for booking value, rebooking likelihood, or overbooking costs.
- The threshold of 0.303 prioritizes F1 on validation data; a different operating point may be preferable depending on the cost of false alarms versus missed cancellations.
- Some potentially useful fields (e.g., `country`, `agent`, room types) were excluded to keep the model based only on information available at booking time.
