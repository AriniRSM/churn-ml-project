import pandas as pd
from joblib import load

ENCODING_MAP = {
    "gender": {"Male": "gender_Male"},
    "Partner": {"Yes": "Partner_Yes"},
    "Dependents": {"Yes": "Dependents_Yes"},
    "PhoneService": {"Yes": "PhoneService_Yes"},
    "MultipleLines": {
        "Yes": "MultipleLines_Yes",
        "No phone service": "MultipleLines_No phone service"
    },
    "InternetService": {
        "Fiber optic": "InternetService_Fiber optic",
        "No": "InternetService_No"
    },
    "OnlineSecurity": {
        "Yes": "OnlineSecurity_Yes",
        "No internet service": "OnlineSecurity_No internet service"
    },
    "OnlineBackup": {
        "Yes": "OnlineBackup_Yes",
        "No internet service": "OnlineBackup_No internet service"
    },
    "DeviceProtection": {
        "Yes": "DeviceProtection_Yes",
        "No internet service": "DeviceProtection_No internet service"
    },
    "TechSupport": {
        "Yes": "TechSupport_Yes",
        "No internet service": "TechSupport_No internet service"
    },
    "StreamingTV": {
        "Yes": "StreamingTV_Yes",
        "No internet service": "StreamingTV_No internet service"
    },
    "StreamingMovies": {
        "Yes": "StreamingMovies_Yes",
        "No internet service": "StreamingMovies_No internet service"
    },
    "Contract": {
        "One year": "Contract_One year",
        "Two year": "Contract_Two year"
    },
    "PaperlessBilling": {"Yes": "PaperlessBilling_Yes"},
    "PaymentMethod": {
        "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
        "Electronic check": "PaymentMethod_Electronic check",
        "Mailed check": "PaymentMethod_Mailed check"
    }
}

FEATURE_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

def predict_churn(user_input: dict):
    model = load('model/churn_model.pkl')
    scaler = load('model/scaler.pkl')
    data = dict.fromkeys(FEATURE_COLUMNS, 0)

    data['SeniorCitizen'] = user_input['SeniorCitizen']
    data['tenure'] = user_input['tenure']
    data['MonthlyCharges'] = user_input['MonthlyCharges']
    data['TotalCharges'] = user_input['TotalCharges']

    for field, rules in ENCODING_MAP.items():
        value = user_input[field]
        if value in rules:
            data[rules[value]] = 1

    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    proba = model.predict_proba(df_scaled)[0][1]

    prediction = "Yes" if proba >= 0.5 else "No"

    return prediction, round(proba*100, 2)