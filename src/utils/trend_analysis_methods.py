import os
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf

from keras import layers
from keras import models

import pandas_ta as ta

import yfinance as yf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

MODEL_WEIGHT_PATH = os.path.join("models", "weights", "march_sixth_4Layers.weights.h5")

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

def get_dataset_columns(ticker, period="1y", window_size=60):
    df = yf.Tickers(ticker).history(period=period, progress=False)
    vix_data = yf.Tickers("^VIX").history(period=period, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.droplevel(1)

    df = df.ffill().dropna()
    vix_data = vix_data.reindex(df.index).ffill()

    open_trend = apply_ssa_trend(df['Open'].values, window=int((window_size/4)))
    vix_trend  = apply_ssa_trend(vix_data['Open'].values, window=int((window_size/4)))

    min_len = min(len(df), len(open_trend), len(vix_trend))
    df = df.iloc[:min_len]

    ########################################################################

    # Features
    
    df['Open_Trend'] = open_trend[:min_len]
    df['Vix_Trend'] = vix_trend[:min_len]
    df['Trend_Diff'] = df['Open'] - df['Open_Trend']
    df['Vix_Trend_Diff'] = vix_data['Open'] - df['Vix_Trend']

    df['RSI'] = ta.rsi(df['Open'], length=14)

    macd = ta.macd(df['Open'])
    df = pd.concat([df, macd], axis=1)

    bbands = ta.bbands(df['Open'], length=20)
    df = pd.concat([df, bbands], axis=1)

    df['ATR'] = ta.atr(df['High'], df['Low'], df['Open'], length=14)

    df['Daily_Range'] = (df['High'] - df['Low']) / df['Open']

    df['VIX_Rel'] = vix_data['Open'] / vix_data['Open'].rolling(20).mean()

    ########################################################################

    # Target

    df['Target'] = (df['Open'].shift(-1) - df['Open']) / df['Open'] * 100
    # df['Target'] = np.log(df['Open'].shift(-1) / df['Open']) * 100
    # df['Target'] = df['Open'].shift(-1)

    ########################################################################

    df = df.iloc[:-1]

    df = df.drop(df.index[0])
    # df = df.drop(['High', 'Low', 'Volume', 'Close', 'Stock Splits', 'Dividends'], axis=1)
    
    df.columns.name = None
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    target_col = ["Target"]

    return df, feature_cols, target_col

def get_full_dataset(tickers, period="1y", window_size=60, test_size=0.2):
    X_total, y_total = [], []
    
    for ticker in tickers:
        df, feature_cols, target_col = get_dataset_columns(ticker, period=period, window_size=window_size)
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

def get_CNN_LSTM(window_size=60, feature_num=3):
    dropout_rate = 0.2
    
    input_layer = layers.Input(shape=(window_size, feature_num), name="Input")
    
    x = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name="Conv1D_1")(input_layer)
    x = layers.LayerNormalization(name="LayerNormalization_1")(x)
    x = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name="Conv1D_2")(x)
    x = layers.LayerNormalization(name="LayerNormalization_2")(x)
    
    x = layers.MaxPooling1D(pool_size=2, name="Max_Pooling")(x)
    
    x = layers.LSTM(64, return_sequences=True, name="LSTM_1")(x)
    x = layers.Dropout(dropout_rate, name="Dropout_1")(x)
    x = layers.BatchNormalization(name="BatchNormalization_1")(x)
    
    x = layers.LSTM(32, return_sequences=False, name="LSTM_2")(x)
    x = layers.Dropout(dropout_rate, name="Dropout_2")(x)
    x = layers.BatchNormalization(name="BatchNormalization_2")(x)
    
    x = layers.Dense(16, activation="relu", name="FullyConnected_1")(x)
    output_layer = layers.Dense(1, activation="linear", name="Output")(x)

    model = models.Model(inputs=input_layer, outputs=output_layer, name='SSA_CNN_LSTM_Hybrid')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss='mse', 
                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    return model

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

def load_TA_model():
    model = get_CNN_LSTM()
    model.load_weights(MODEL_WEIGHT_PATH)

    return model
    