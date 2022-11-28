#!/usr/bin/env python

import numpy as np
import pandas as pd

def get_sepsis_score(data, model):
    data = data.copy()
    x_mean = np.array([22.99, 85.11, 97.13, 36.86, 122.36, 81.77, 63.23, 18.94, -0.46, 0.49, 7.39, 
        0.63, 23.76, 7.82, 1.59, 130.87, 2.05, 4.15, 31.17, 10.38, 11.08, 196.6, 62.77])
    x_std = np.array([16.86, 16.74, 3.04, 0.72, 23.14, 16.32, 14.02, 5.17, 3.83, 0.19, 0.06, 8.45, 
        19.63, 2.1, 2.02, 47.45, 0.38, 0.61, 5.65, 1.95, 5.87, 100.95, 15.97])
    data_pd = pd.DataFrame(data)
    data_pd = data_pd.fillna(method='ffill')
    data_pd.iloc[:, :-1] = (data_pd.iloc[:, :-1] - x_mean) / x_std

    for i in range(1, len(data)):
        mask = np.isnan(data[i])
        data[i][mask] = data[i - 1][mask]
    data[:,:-1] = (data[:,:-1] - x_mean) / x_std

    processed_features = data[-1].copy()
    processed_features = np.concatenate((add_rolling_std_features_np(data[:, 1:-2], windows=10), processed_features), axis=0)
    processed_features = np.concatenate((add_previous_rows_np(data[:, :-2], windows=3), processed_features), axis=0)
    processed_features = np.concatenate((add_cummax_features_np(data[:, 1:-2]), processed_features), axis=0)
    processed_features = np.concatenate((add_cummin_features_np(data[:, 1:-2]), processed_features), axis=0)
    processed_features = np.nan_to_num(processed_features, posinf=0, neginf=0)

    # predict
    threshold = 0.0361
    score = model.predict_proba(processed_features.reshape(1, -1))[:, 1]
    label = (score >= threshold).astype(int)

    return score, label

def load_sepsis_model():
    model = joblib.load('randomForest.model')
    return model

def add_rolling_std_features(df, windows=10):
    df_roll = df.transform(lambda s: s.rolling(windows, min_periods=1).std(ddof=0))
    df_roll.columns = [str(col) + '_roll_std' for col in df_roll.columns]
    return df_roll

def add_cummax_features_np(data):
    data_cummax = data[:-1].max(axis=0) if data.shape[0] > 1 else np.zeros(data.shape[1])
    return data_cummax

def add_cummin_features_np(data):
    data_cummin = data[:-1].min(axis=0) if data.shape[0] > 1 else np.zeros(data.shape[1])
    return data_cummin

def add_previous_rows_np(data, windows=3):
    data_shift = data[-2] if is_valid_index(data, -2) else np.zeros(data.shape[1])
    for i in range(3, windows+2):
        shift_row = data[-i] if is_valid_index(data, -i) else np.zeros(data.shape[1])
        data_shift = np.concatenate((data_shift, shift_row), axis=0)
    return data_shift

def add_rolling_std_features_np(data, windows=10):
    data_roll = data[-windows:].std(axis=0, ddof=0)
    return data_roll

def is_valid_index(data, index):
    return index in range(-len(data), len(data))