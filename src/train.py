from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.preprocess  import load_and_preprocess

X_train, X_test, y_train, y_test, columns = load_and_preprocess('../data/churn.csv')

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

prediction = model.predict(X_test)
acc = accuracy_score(y_test, prediction)
print("Accuracy:", acc)

dump(model, "../model/churn_model.pkl")
print("Model saved")
print(columns)
