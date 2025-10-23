import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

X_train = pd.read_csv('./data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/y_train.csv').values.ravel()

with open('./models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

model = GradientBoostingRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

model_output_path = './models/gbr_model.pkl'
with open(model_output_path, 'wb') as f:
    pickle.dump(model, f)
