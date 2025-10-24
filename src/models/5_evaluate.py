import pandas as pd
import pickle
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json

gbr = joblib.load("./models/gbr_model.pkl")
X_test = pd.read_csv('./data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('./data/processed_data/y_test.csv').values.ravel()

predictions = gbr.predict(X_test)
pd.DataFrame({"y_pred": predictions, "y_test": y_test}).to_csv("./data/processed_data/predictions.csv")

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"Model Performance sur Test Set")
print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")


scores = {
    'r2_score': r2,
    'mean_squared_error': mse,
}

with open('./metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)