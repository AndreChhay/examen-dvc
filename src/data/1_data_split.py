import pandas as pd
from sklearn.model_selection import train_test_split
import os
os.makedirs("data/processed_data", exist_ok=True)
URL= "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

df = pd.read_csv(URL)

# Export raw.csv
df.to_csv('./data/raw_data/raw.csv', index=False)

# Creqtion X_train, X_test, y_train, y_test au format csv
y = df["silica_concentrate"]
X = df.drop(columns=["silica_concentrate", "date"])

test_size = 0.25
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

X_train.to_csv('./data/processed_data/X_train.csv', index=False)
X_test.to_csv('./data/processed_data/X_test.csv', index=False)
y_train.to_csv('./data/processed_data/y_train.csv', index=False)
y_test.to_csv('./data/processed_data/y_test.csv', index=False)

