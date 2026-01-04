
## Customer Churn Prediction System

An end-to-end Machine Learning application that predicts whether a customer is likely to churn(leave) based on their account details.

## Features

* Logistic Regression model trained on telecom churn dataset
* Clean preprocessing pipeline with encoding and scaling
* REST API built using FastAPI
* Interactive web UI built using Streamlit
* Real-time churn prediction with confidence score

## Tech Stack

Python, Pandas, Scikit-Learn, FastAPI, Streamlit

## Project Structure

churn-ml-project/
├── data/
│   └── churn.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── api/
│   └── main.py
├── model/
│   └── churn_model.pkl
├── app.py
└── README.md

## How to Run

1. Train the model :
   python3 -m src.train

2. Start the API:
   uvicorn api.main:app --reload

3. Start the UI:
   streamlit run app.py

Then open: [http://localhost:8501](http://localhost:8501)

## Sample Output

Prediction: Yes
Confidence: 82.4%

## What This Project Demonstrates

* End-to-end ML engineering workflow
* Feature engineering and encoding
* Model training and evaluation
* API deployment of ML model
* Building ML-powered application
