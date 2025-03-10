import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.data.copy()
    df['MedHouseVal'] = data.target
    return df

def add_features(X):
    X = X.copy()
    X['RoomsPerBedroom'] = X['AveRooms'] / X['AveBedrms']
    X['PopulationPerBedroom'] = X['Population'] / X['AveBedrms']
    X['MedInc_HouseAge'] = X['MedInc'] * X['HouseAge']
    X['Latitude_Longitude'] = X['Latitude'] * X['Longitude']
    return X

def build_pipeline(model, numerical_features):
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

    return Pipeline(steps=[
        ('add_features', FunctionTransformer(add_features)),
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_regression)),
        ('model', model)
    ])

def evaluate_model(model_name, pipeline, param_grid, X_train, y_train, X_test, y_test):
    if model_name in ['Random Forest', 'XGBoost']:
        grid_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=50, cv=5, scoring='r2', n_jobs=-1, verbose=2, random_state=42)
    else:
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    
    scores = {
        'Best Params': grid_search.best_params_,
        'R²': r2_score(y_test, y_pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    print(f"{model_name} Results:\n", scores, "\n", "-" * 30)
    return grid_search.best_estimator_, scores['R²']

def save_model(model, filename="california_housing_model.joblib"):
    joblib.dump(model, filename)
    print(f"Best model saved as {filename}")

def main():
    df = load_data()
    X, y = df.drop('MedHouseVal', axis=1), df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    engineered_features = ['RoomsPerBedroom', 'PopulationPerBedroom', 'MedInc_HouseAge', 'Latitude_Longitude']
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    numerical_features.extend(engineered_features)
    
    models = {
        'Linear Regression': (LinearRegression(), {
            'selector__k': list(range(3, len(numerical_features) + 1))
        }),
        'Decision Tree': (DecisionTreeRegressor(random_state=42), {
            'selector__k': [4, 8, 'all'],
            'model__max_depth': [None, 5, 10, 15, 20, 30],
            'model__min_samples_split': [2, 5, 10, 20, 50]
        }),
        'Random Forest': (RandomForestRegressor(random_state=42), {
            'selector__k': [4, 8, 'all'],
            'model__n_estimators': [50, 100, 200, 300, 500],
            'model__max_depth': [None, 5, 10, 15, 20, 30],
            'model__min_samples_split': [2, 5, 10, 20, 50],
            'model__min_samples_leaf': [1, 2, 5, 10]
        }),
        'XGBoost': (xgb.XGBRegressor(random_state=42), {
            'selector__k': [4, 8, 'all'],
            'model__n_estimators': [50, 100, 200, 300, 500],
            'model__max_depth': [3, 5, 7, 10, 15],
            'model__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0]
        })
    }

    best_model, best_score = None, float('-inf')
    for name, (model, param_grid) in models.items():
        pipeline = build_pipeline(model, numerical_features)
        estimator, score = evaluate_model(name, pipeline, param_grid, X_train, y_train, X_test, y_test)
        if score > best_score:
            best_model, best_score = estimator, score
    
    print(f"Best Model: {best_model}")
    save_model(best_model)

if __name__ == "__main__":
    main()
