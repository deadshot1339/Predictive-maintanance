# Predictive Maintenance using Machine Learning

## Project Overview

This project predicts whether an industrial machine is likely to fail based on operational parameters such as air temperature, process temperature, rotational speed, torque, tool wear, and machine type.

A Random Forest Classifier is trained on the AI4I 2020 Predictive Maintenance dataset and deployed through an interactive Streamlit dashboard.

---

## Features

* Machine Failure Prediction
* Failure Probability Estimation
* Risk Classification (Low / Medium / High Risk)
* Feature Importance Analysis
* Interactive Streamlit Dashboard

---

## Dataset

Dataset: AI4I 2020 Predictive Maintenance Dataset

Features Used:

* Air Temperature [K]
* Process Temperature [K]
* Rotational Speed [rpm]
* Torque [Nm]
* Tool Wear [min]
* Machine Type

Target:

* Machine Failure (0 = No Failure, 1 = Failure)

---

## Machine Learning Model

Algorithm:

* Random Forest Classifier

Libraries Used:

* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib

---

## Model Performance

Accuracy: 98.4%

Classification Report:

* Precision (Failure Class): 0.85
* Recall (Failure Class): 0.57
* F1 Score (Failure Class): 0.69

---

## Dashboard Features

The Streamlit dashboard allows users to:

1. Enter machine operating parameters
2. Predict machine failure
3. View failure probability
4. View risk level
5. Analyze feature importance

---

## Project Structure

predictive-maintenance/

├── app.py

├── model.pkl

├── columns.pkl

├── feature_importance.csv

├── ai4i2020.csv

├── requirements.txt

└── README.md

---

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the dashboard:

streamlit run app.py

---

## Future Improvements

* Better UI Design
* Real-Time Sensor Integration
* Cloud Deployment
* Multiple ML Model Comparison

---

## Author

Deepanshu
