import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# Creation pipeline
# pipeline = Pipeline([
#     ('model', GradientBoostingRegressor(random_state=42))
#     ])

model = GradientBoostingRegressor(random_state=42)

#hyperpqrqmetre de gbr
params = [
    {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0]
    }
    ]

scoring_mettrics= "r2"
grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring=scoring_mettrics,
    n_jobs=-1,
    verbose=1
    )

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best parameters:", best_params)

output_path = './models/best_params.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(best_params, f)