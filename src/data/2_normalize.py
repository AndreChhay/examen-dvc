import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('./data/processed_data/X_train.csv')
X_test = pd.read_csv('./data/processed_data/X_test.csv')

print("\n X_train :")
print(X_train.dtypes)
print("\n X_test :")
print(X_test.dtypes)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#en dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(X_train_scaled.head())
print(X_test_scaled.head())


X_train_scaled.to_csv('./data/processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('./data/processed_data/X_test_scaled.csv', index=False)

