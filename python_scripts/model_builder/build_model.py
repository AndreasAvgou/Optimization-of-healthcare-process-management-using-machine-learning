import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from model_builder.rnn import RNN


def build_model(dataframe):
    X = dataframe.drop(['x_ArrivalDTTM', 'x_ScheduledDTTM', 'x_BeginDTTM', 'Wait'], axis=1)
    y = dataframe['Wait']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    regression_models = [
        {'model': LinearRegression(), 'param_dist': {'fit_intercept': [True, False]}},
        {'model': XGBRegressor(), 'param_dist': {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}},
        {'model': DecisionTreeRegressor(), 'param_dist': {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}},
        {'model': RNN(input_shape=(1, X_train.shape[1]), epochs=30, batch_size=64), 'param_dist': {}}
    ]

    best_models = {}

    for model_info in regression_models:
        model_name = model_info['model'].__class__.__name__
        print(f"Model: {model_name}")

        model = model_info['model']
        param_dist = model_info['param_dist']

        if param_dist:
            n_iter = min(np.prod([len(v) for v in param_dist.values()]), 10)
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, scoring='r2', cv=10, n_iter=n_iter)
            random_search.fit(X_train_scaled, y_train)
            best_model = random_search.best_estimator_
        else:
            best_model = model.fit(X_train_scaled, y_train)

        best_models[model_name] = best_model

    return best_models, X_train_scaled, X_test_scaled, y_train, y_test, X.columns