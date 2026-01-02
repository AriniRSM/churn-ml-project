import pandas as pd
from joblib import load

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

def predict_churn(user_input: dict) -> str:
    model = load('model/churn_model.pkl')
    data = dict.fromkeys(FEATURE_COLUMNS, 0)

    data['SeniorCitizen'] = user_input['SeniorCitizen']
    data['tenure'] = user_input['tenure']
    data['MonthlyCharges'] = user_input['MonthlyCharges']
    data['TotalCharges'] = user_input['TotalCharges']

    if user_input['gender'] == 'Male':
        data['gender_Male'] = 1

    if user_input['Partner'] == 'Yes':
        data['Partner_Yes'] = 1

    if user_input['Dependents'] == 'Yes':
        data['Dependents_Yes'] = 1

    if user_input['PhoneService'] == 'Yes':
        data['PhoneService_Yes'] = 1

    if user_input['MultipleLines'] == 'Yes':
        data['MultipleLines_Yes'] = 1
    elif user_input['MultipleLines'] == 'No phone service':
        data['MultipleLines_No phone service'] = 1

    if user_input['InternetService'] == 'Fiber optic':
        data['InternetService_Fiber optic'] = 1
    elif user_input['InternetService'] == 'No':
        data['InternetService_No'] = 1

    if user_input['OnlineSecurity'] == 'Yes':
        data['OnlineSecurity_Yes'] = 1
    elif user_input['OnlineSecurity'] == 'No internet service':
        data['OnlineSecurity_No internet service'] = 1

    if user_input['OnlineBackup'] == 'Yes':
        data['OnlineBackup_Yes'] = 1
    elif user_input['OnlineBackup'] == 'No internet service':
        data['OnlineBackup_No internet service'] = 1

    if user_input['DeviceProtection'] == 'Yes':
        data['DeviceProtection_Yes'] = 1
    elif user_input['DeviceProtection'] == 'No internet service':
        data['DeviceProtection_No internet service'] = 1

    if user_input['TechSupport'] == 'Yes':
        data['TechSupport_Yes'] = 1
    elif user_input['TechSupport'] == 'No internet service':
        data['TechSupport_No internet service'] = 1

    if user_input['StreamingTV'] == 'Yes':
        data['StreamingTV_Yes'] = 1
    elif user_input['StreamingTV'] == 'No internet service':
        data['StreamingTV_No internet service'] = 1

    if user_input['StreamingMovies'] == 'Yes':
        data['StreamingMovies_Yes'] = 1
    elif user_input['StreamingMovies'] == 'No internet service':
        data['StreamingMovies_No internet service'] = 1

    if user_input['Contract'] == 'One year':
        data['Contract_One year'] = 1
    elif user_input['Contract'] == 'Two year':
        data['Contract_Two year'] = 1

    if user_input['PaperlessBilling'] == 'Yes':
        data['PaperlessBilling_Yes'] = 1

    if user_input['PaymentMethod'] == 'Credit card (automatic)':
        data['PaymentMethod_Credit card (automatic)'] = 1
    elif user_input['PaymentMethod'] == 'Electronic check':
        data['PaymentMethod_Electronic check'] = 1
    elif user_input['PaymentMethod'] == 'Mailed check':
        data['PaymentMethod_Mailed check'] = 1

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return "Yes" if prediction == 1 else "No"