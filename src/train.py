import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df = df.drop('customerID', axis=1)
df = pd.get_dummies(df , drop_first = True)

x = df.drop('Churn_Yes' , axis = 1)
y = df['Churn_Yes']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

preds = model.predict(x_test)
acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)

dump(model, "model/churn_model.pkl")
print("Model saved")
print(x.columns)
