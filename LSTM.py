#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import math


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dropout, LSTM, Dense, Layer, LayerNormalization, BatchNormalization, Conv1D, MaxPooling1D, Reshape, Add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping

import yfinance as yf



# In[ ]:


def apply_ssa_trend(series, window = 15, lookback = 60):
    n_total = len(series)
    result = np.zeros(n_total)

    result[:lookback] = series[:lookback]

    for t in range(lookback, n_total):
        sub_series = series[t - lookback + 1 : t + 1]

        N = len(sub_series)
        L = window
        K = N - L + 1
        X = np.column_stack([sub_series[i:i+L] for i in range(K)])

        U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

        X1 = Sigma[0] * np.outer(U[:, 0], VT[0, :])

        n = N - 1
        vals = [X1[i, n-i] for i in range(max(0, n-K+1), min(n+1, L))]
        result[t] = np.mean(vals)

    return result

def normalize_window(window):
    mean = window.mean(axis = 0)
    std = window.std(axis = 0) + 1e-8
    return (window - mean) / std


# In[ ]:


def get_dataset_columns(ticker, period="1y", window=60):
    df = yf.Tickers(ticker).history(period=period)
    vix_data = yf.Tickers("^VIX").history(period=period, progress=False)

    df.columns = df.columns.droplevel(1)
    vix_data.columns = vix_data.columns.droplevel(1)

    df = df.ffill().dropna()

    vix_data = vix_data.reindex(df.index).ffill()

    df = df.drop(['Stock Splits', 'Dividends'], axis=1)

    open_trend = apply_ssa_trend(df['Open'].values, window=int((window/4)))
    vix_trend  = apply_ssa_trend(vix_data['Open'].values, window=int((window/4)))

    min_len = min(len(df), len(open_trend), len(vix_trend))

    df = df.iloc[:min_len]
    df['Open_Trend'] = open_trend[:min_len]
    df['Vix_Trend'] = vix_trend[:min_len]

    df['Trend_Diff'] = df['Open'] - df['Open_Trend']

    df['Target'] = ((df['Close'] - df['Open'])/ df['Open']) * 100 
    # df['Target'] = np.log(df['Close'] / df['Open']) * 100
    # df['Target'] = df['Close']

    df = df.drop(df.index[0])
    df = df.drop(['High', 'Low', 'Volume', 'Close'], axis=1)

    feature_cols = ["Open_Trend", "Trend_Diff", "Vix_Trend"]
    target_col = ["Target"]

    return df, feature_cols, target_col


# In[ ]:


def get_full_dataset(tickers, period="1y", window_size=60, test_size=0.2):
    X_total, y_total = [], []

    for ticker in tickers:
        df, feature_cols, target_col = get_dataset_columns(ticker, period=period, window=window_size)
        data_x = df[feature_cols].values
        data_y = df[target_col].values

        for i in range(len(df) - window_size):
            window_x = data_x[i : i + window_size]
            window_y = data_y[i + window_size]

            window_x = normalize_window(window_x)

            X_total.append(window_x)
            y_total.append(window_y)

    X_data = np.array(X_total)
    y_data = np.array(y_total)

    nan_mask = np.isnan(X_data).any(axis=(1, 2))
    X_data = X_data[~nan_mask]
    y_data = y_data[~nan_mask] 

    print(f"NaN Deleted: {np.sum(nan_mask)}")

    test_first_index = int(len(X_data) * (1 - test_size))
    X_train, X_test = X_data[:test_first_index], X_data[test_first_index:]
    y_train, y_test = y_data[:test_first_index], y_data[test_first_index:]

    print("""
    X_train Shape : {}
    X_test Shape : {}
    y_train Shape : {}
    y_test Shape : {}
    """.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    return X_train, X_test, y_train, y_test


# In[10]:


def get_data_for_prediction(ticker, period="120d", window_size=60):
    df, feature_cols, _ = get_dataset_columns(ticker, period=period, window=window_size)

    if len(df) < window_size:
        raise ValueError("Not enough rows")

    X = df[feature_cols].iloc[-window_size:].values

    if np.isnan(X).any():
        raise ValueError("NaN detected in prediction window")

    X = normalize_window(X)

    X = X.reshape(1, window_size, len(feature_cols))

    return X


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


def save_model_weights(model, file_name, folder_name):
    os.makedirs(folder_name, exist_ok=True)

    words = file_name.split(".")

    model_name = words[0]

    existing_files = [f for f in os.listdir(folder_name) if f.startswith(model_name)]
    next_number = len(existing_files) + 1
    words[0] = f"{model_name}_{next_number}"

    file_name = ".".join(words)
    save_path = os.path.join(folder_name, file_name)

    model.save_weights(save_path)
    print(f"model saved to: {save_path}")


# In[ ]:





# In[ ]:




