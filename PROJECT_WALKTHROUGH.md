# Project Walkthrough

This document explains the whole project in very simple words. You can read it to understand what the project does, how it works, and how to explain it to a mentor.

---

## 1. What is this project about?

Imagine you run a hotel. People book rooms online. Sometimes they cancel their booking. Cancellations cost the hotel money because the room might stay empty.

This project does two things:

1. It looks at hotel booking data to find patterns: what kinds of bookings are more likely to be canceled?
2. It builds a computer program that guesses, for a new booking, how likely it is to be canceled.

There is also a simple web app where you can type in booking details and get a prediction.

---

## 2. What files are in the project?

Here is the full list and what each one does:

| File or folder | What it does |
|---|---|
| `data/hotel_bookings.csv` | The raw data. 119,390 hotel bookings with 32 pieces of information each. |
| `data/cleaned_hotel_bookings.csv` | The cleaned data. 86,943 bookings after removing duplicates, bad rows, and missing information. |
| `models/pipeline.pkl` | The trained model saved as a file. It takes booking details and outputs a cancellation probability. |
| `models/threshold.pkl` | A single number (about 0.303). If the model's probability is higher than this, the app says "likely to cancel." |
| `notebooks/hotel_booking_demand.ipynb` | The main analysis notebook. It cleans the data, explores it, trains the model, and checks how well it works. |
| `app.py` | The web app. You open it in a browser, fill in a form, and get a cancellation prediction. |
| `README.md` | A short description of the project for people visiting GitHub. |
| `requirements.txt` | A list of Python packages you need to install before running anything. |
| `.gitignore` | Tells Git which files to ignore (temporary files, caches, etc.). |

---

## 3. How does data move through the project?

Here is the journey of the data, step by step:

1. **Raw data** (`hotel_bookings.csv`) is loaded into the notebook.
2. **Cleaning**: The notebook removes exact duplicate rows, fills in missing values, removes bookings with zero guests, removes bookings with impossible prices, and throws away two columns that would give away the answer (`reservation_status` and `reservation_status_date`).
3. **Cleaned data** is saved to `cleaned_hotel_bookings.csv`.
4. **Feature engineering**: Two new columns are created: `total_nights` (weekend nights + week nights) and `total_guests` (adults + children + babies).
5. **Splitting**: The data is split into training and test sets. The test set is kept completely untouched until the very end.
6. **Training**: The notebook trains several models on the training data, picks XGBoost as the best, tunes it, chooses a decision threshold, and saves the final model to `models/pipeline.pkl`.
7. **Testing**: The saved model is tested once on the untouched test set to see how well it generalizes.
8. **Prediction**: The web app (`app.py`) loads the saved model and threshold, lets a user type in booking details, and shows the predicted cancellation probability.

---

## 4. What happens in the notebook?

The notebook is the heart of the project. It has these main parts:

### A. Understand the data
It loads the raw CSV and checks:
- How many rows and columns?
- What are the column names?
- Are there missing values?
- How many bookings are canceled vs. not canceled?

### B. Clean the data
- Remove exact duplicate rows.
- Drop the `company` column (too much missing data).
- Fill missing `children`, `country`, and `agent` values.
- Remove bookings where no guests are staying.
- Remove bookings with impossible average daily rates (negative or over 1000).
- Remove `reservation_status` and `reservation_status_date` because they directly reveal whether a booking was canceled (this is called data leakage).

### C. Explore the data (EDA)
The notebook makes charts to answer questions like:
- Do city hotels or resort hotels have more cancellations?
- Do canceled bookings have longer lead times?
- Which months have the highest cancellation rate?
- Which market segment cancels most often?
- Does having a non-refundable deposit change the cancellation rate?
- Do repeat guests cancel less often?

### D. Build the model
- Split the data into training and test sets.
- Create a pipeline that: encodes text columns, picks the 25 most useful features, scales numbers, and runs the model.
- Compare 13 different models using cross-validation.
- Pick XGBoost because it had the best ROC-AUC score.
- Tune XGBoost settings (`max_depth`, `min_child_weight`).
- Check if calibrating the probabilities helps (it did not, so raw probabilities are used).
- Choose a decision threshold (0.303) that gives the best F1 score on validation data.
- Test the final model once on the untouched test set.

### E. Save the model
The trained pipeline and threshold are saved to `models/pipeline.pkl` and `models/threshold.pkl`.

---

## 5. What is the model, and why XGBoost?

The model is called **XGBoost**. It is a type of decision tree model that builds many trees one after another, where each new tree tries to fix the mistakes of the previous ones.

The notebook compared 13 different models (including logistic regression, random forest, decision tree, and others) using cross-validation. XGBoost was chosen because:

- It had the highest cross-validation ROC-AUC score (about 0.83), meaning it is good at separating canceled bookings from non-canceled ones.
- Its F1 score was competitive with the best model.

The model does not use every column. It uses only information that would be known at booking time (like hotel type, lead time, deposit type, number of guests). It does not use things like `country` or `agent` because those are not on the booking form in the app.

---

## 6. Important calculations

Here are the key numbers, all verified from the actual test data:

| Metric | Score | What it means |
|---|---|---|
| Accuracy | 0.775 | Out of 100 bookings, the model gets about 77 right. |
| Precision | 0.572 | When the model says "this will cancel," it is right about 57% of the time. |
| Recall | 0.706 | Out of all bookings that actually cancel, the model catches about 71%. |
| F1 | 0.632 | A balance of precision and recall. |
| ROC-AUC | 0.841 | The model is good at ranking bookings by cancellation risk (1.0 would be perfect). |
| Brier score | 0.136 | Measures how accurate the probability guesses are (lower is better). |

**Decision threshold: 0.303.** If the model says a booking has more than a 30.3% chance of canceling, the app predicts "likely to cancel."

Accuracy looks high partly because most bookings are not canceled. That is why recall and F1 are more useful for this problem — we want to catch as many cancellations as possible.

---

## 7. How does the app (`app.py`) work?

The app is built with Streamlit. Here is what happens when you open it:

1. It loads the saved model (`pipeline.pkl`) and threshold (`threshold.pkl`).
2. It shows a short description and the model's test performance.
3. It shows key findings from the data analysis.
4. It shows a form where you can enter booking details: hotel type, lead time, number of guests, deposit type, etc.
5. When you click "Predict Cancellation," the app:
   - Puts your input into a table.
   - Adds the two engineered columns (`total_nights`, `total_guests`).
   - Runs the saved model to get a cancellation probability.
   - Compares the probability to the threshold (0.303).
   - Shows the result: "likely to cancel" or "not likely to cancel," plus a progress bar.

All numbers shown in the app come from the actual trained model and the actual test results — nothing is made up.

---

## 8. How do the files connect?

```
data/hotel_bookings.csv
        |
        v (notebook cleans and analyzes)
data/cleaned_hotel_bookings.csv
        |
        v (notebook trains and saves)
models/pipeline.pkl   models/threshold.pkl
        ^                  ^
        |                  |
        +---- app.py loads these to make predictions
```

The notebook produces the cleaned data and the saved model. The app uses the saved model and threshold to make predictions. The README tells visitors what the project is about.

---

## 9. How do you run the project?

**Step 1:** Install the required Python packages:
```
pip install -r requirements.txt
```

**Step 2:** Run the web app:
```
streamlit run app.py
```

**Step 3:** Open the web address that Streamlit shows (usually `http://localhost:8501`). Fill in the booking form and click predict.

**To retrain the model:** Open `notebooks/hotel_booking_demand.ipynb` and run all the cells. It will recreate the cleaned CSV and save a new pipeline and threshold.

---

## 10. What do the outputs mean?

- **Cancellation probability**: A number from 0 to 1. Closer to 1 means more likely to cancel.
- **Prediction**: "Likely to cancel" if probability is 0.303 or higher, otherwise "not likely to cancel."
- **Progress bar**: A visual bar showing the probability as a percentage.
- **Model metrics**: Accuracy, precision, recall, F1, ROC-AUC, and Brier score from the test set. These tell you how well the model performs overall.

---

## 11. Important design decisions

- **Data leakage prevention**: `reservation_status` and `reservation_status_date` were removed because they directly reveal the answer. If they were kept, the model would "cheat" and look perfect but fail on real new bookings.
- **Train/validation/test split**: The data is split so the test set is never used to make any decisions. This gives an honest estimate of how the model will work in the real world.
- **Threshold selection**: The threshold (0.303) was chosen on validation data to maximize F1, not on the test set. This keeps the test set unbiased.
- **Feature selection**: Only 25 features are selected automatically inside the pipeline, and this selection happens inside each cross-validation fold so no information leaks from the validation fold.
- **No calibration**: Sigmoid calibration was tried but did not improve the Brier score, so raw XGBoost probabilities are used.

---

## 12. Likely mentor questions and simple answers

**Q: What is the dataset?**
A: It is hotel booking data from two hotels (a city hotel and a resort hotel) covering 2015 to 2017. Each row is one booking with details like lead time, deposit type, number of guests, and whether it was canceled.

**Q: Why did you remove `reservation_status` and `reservation_status_date`?**
A: Because they directly tell you whether a booking was canceled. Using them would be data leakage — the model would just look at the answer instead of learning to predict it.

**Q: Why XGBoost?**
A: It was compared against 12 other models using cross-validation. XGBoost had the best ROC-AUC (about 0.83) while keeping a competitive F1 score.

**Q: What is the threshold and why 0.303?**
A: The threshold is the cutoff probability for predicting "will cancel." It was chosen on validation data to maximize F1. It is 0.303, meaning a booking is predicted to cancel if its probability is 30.3% or higher.

**Q: What is the difference between precision and recall?**
A: Precision is "when the model says cancel, how often is it right?" Recall is "out of all real cancellations, how many did the model catch?" For this project, recall matters because missing a cancellation is costly.

**Q: Why is accuracy not the best measure?**
A: Because most bookings are not canceled. A model that always predicts "not canceled" would be right about 73% of the time but useless for catching cancellations.

**Q: Can this model be used in a real hotel?**
A: It is a demonstration project. The data is from 2015–2017 and only two hotels. A real hotel would need recent data from its own bookings, and would need to consider costs like rebooking and overbooking.

**Q: What does the Brier score mean?**
A: It measures how accurate the probability predictions are. Lower is better. A score of 0.136 means the probabilities are reasonably well-calibrated.

**Q: Did you use the test set to pick the model?**
A: No. The test set was used only once, at the very end, to report final performance. All model choices were made using training and validation data.

**Q: What is cross-validation?**
A: It splits the training data into several parts, trains on some and tests on others, rotating which part is tested. This gives a more reliable estimate of model performance than a single split.

**Q: What are `total_nights` and `total_guests`?**
A: New columns created by adding existing ones. `total_nights` = weekend nights + week nights. `total_guests` = adults + children + babies.

---

## 13. Summary

This project takes hotel booking data, cleans it, finds patterns in cancellations, trains an XGBoost model to predict cancellations, and provides a simple web app to try it out. Every number in the app and README comes from the actual trained model and test data — nothing is invented.
