from fastapi import FastAPI
from src.predict import predict_churn
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to Churn Prediction"}

class CustomerInput(BaseModel):
        SeniorCitizen: int
        tenure: int
        MonthlyCharges: float
        TotalCharges: float
        gender: str
        Partner: str
        Dependents: str
        PhoneService: str
        MultipleLines: str
        InternetService: str
        OnlineSecurity: str
        OnlineBackup: str
        DeviceProtection: str
        TechSupport: str
        StreamingTV: str
        StreamingMovies: str
        Contract: str
        PaperlessBilling: str
        PaymentMethod: str

@app.post("/predict")
def predict(input: CustomerInput):
    result = predict_churn(input.dict())
    return {"churn" : result}

