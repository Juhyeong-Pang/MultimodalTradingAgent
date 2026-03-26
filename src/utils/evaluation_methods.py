import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.trend_analysis_methods import get_CNN_LSTM, get_dataset_columns, normalize_window

# ---------------------- Methods for LSTM ----------------------

CNN_LSTM_model = get_CNN_LSTM()

def make_3class_label(y, threshold=0.05):
    labels = np.zeros_like(y, dtype=int)

    labels[y > threshold] = 2
    labels[(y > 0) & (y <= threshold)] = 1 
    labels[y <= 0] = 0
    
    return labels

def evaluate_trend_analysis(X_test, y_test, base_point=0):
    predictions = CNN_LSTM_model.predict(X_test)
    print(pd.DataFrame(predictions).describe())

    base_point = base_point
    class_names = ['Fall', 'Neutral', 'Rise']
    threshold = 0.1

    y_test_class = make_3class_label(y_test, threshold)
    predictions_class = make_3class_label(predictions, threshold)

    unique, counts = np.unique(y_test_class, return_counts=True)
    print("Test set class distribution:")
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples")

    unique, counts = np.unique(predictions_class, return_counts=True)
    print("\nPrediction set class distribution:")
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples")

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_class, predictions_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.title('Market Direction Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test_class, predictions_class, target_names=class_names))

    y_test_flat = y_test.flatten()
    preds_flat = predictions.flatten()

    comparison_df = pd.DataFrame({
        'Actual': y_test_flat,
        'Predicted': preds_flat
    })

    chunk_size = 200
    y_min_limit = -3.5 
    y_max_limit = 3.5 

    total_len = len(comparison_df)
    num_plots = int(np.ceil(total_len / chunk_size))

    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 2.5 * num_plots), sharex=False)

    if num_plots == 1:
        axes = [axes]

    for i in range(num_plots):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_len)
        
        subset = comparison_df.iloc[start:end]
        
        axes[i].plot(subset.index, subset['Actual'], label='Actual Excess Return', alpha=0.7)
        axes[i].plot(subset.index, subset['Predicted'], label='Predicted', alpha=0.9, color='orange')
        
        axes[i].set_ylim(y_min_limit, y_max_limit) 
        
        axes[i].axhline(base_point, label='Base Point', color='red', linestyle='--', alpha=0.3)
        axes[i].set_title(f'Actual vs Predicted (Index {start} to {end-1})')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def get_backtest_data(ticker, period="2y", window_size=60):
    df, feature_cols, target_col = get_dataset_columns(ticker, period=period, window_size=window_size)

    data_x = df[feature_cols].values
    data_y = df[target_col].values
    data_open_close = df[["Open", "Close"]].values

    X_list = []
    y_real = []
    dates_list = []
    actual_opens_list = []

    for i in range(len(df) - window_size):
        window_x = data_x[i : i + window_size]
        window_y = data_y[i + window_size]

        if np.isnan(window_x).any() or np.isnan(window_y).any():
            continue
        
        window_x = normalize_window(window_x)
        current_open = data_open_close[i + window_size]
        
        X_list.append(window_x)
        y_real.append(window_y)
        dates_list.append(df.index[i + window_size])
        actual_opens_list.append(current_open)

    return np.array(X_list), np.array(y_real), dates_list, feature_cols, np.array(actual_opens_list)

def backtest_trend_analysis(model, ticker, period="1y", window_size=60, initial_capital=10000):
    X_bt, y_bt, bt_dates, feature_cols, data_open = get_backtest_data(ticker, period=period, window_size=window_size)
    predictions = model.predict(X_bt, verbose=0).flatten()

    previous_signal = "INIT"
    previous_correct_signal = "INIT"
    
    cash = initial_capital
    cash_when_not_traded = initial_capital
    cash_fixed = 0
    shares = 0
    threshold = 0.1
    portfolio_history = []

    buy_count = 0
    buy_correct_count = 0
    actual_buy_count = 0
    sell_count = 0
    sell_correct_count = 0
    actual_sell_count = 0
    hold_count = 0
    hold_correct_count = 0
    actual_hold_count = 0

    correct_pred_count = 0
    wrong_pred_count = 0 

    current_open_price = data_open[0][0]
    shares_when_not_traded = cash_when_not_traded // current_open_price
    cash_when_not_traded -= shares_when_not_traded * current_open_price
    
    for i in range(len(bt_dates)):
        current_open_price = data_open[i][0]
        y_act = y_bt[i][0]
        cash_when_not_traded = shares_when_not_traded * current_open_price

        if y_act >= threshold:
            correct_signal = "BUY"
            actual_buy_count += 1
        elif y_act <= -0.5 * threshold:
            correct_signal = "SELL"
            actual_sell_count += 1
        else:
            correct_signal ="HOLD"
            actual_hold_count += 1

        correct = False

        if predictions[i] >= threshold:
            signal = "BUY"
            buy_count += 1
            if correct_signal == "BUY":
                correct = True
        elif predictions[i] <= -1 * threshold:
            signal = "SELL"
            sell_count += 1
            if correct_signal == "SELL":
                correct = True
        else:
            signal ="HOLD"
            hold_count += 1
            if correct_signal == "HOLD":
                correct = True

        action = "HOLD" if signal == previous_signal else signal
        previous_signal = previous_signal if signal == "HOLD" else signal

        if signal=="BUY" and cash > current_open_price:
            share_num = cash // current_open_price
            cash -= share_num * current_open_price
            shares += share_num

        elif signal=="SELL" and shares > 0:
            cash += shares * current_open_price
            shares = 0

        if cash > initial_capital:
            rev = cash - initial_capital
            cash -= rev
            cash_fixed += rev

        total_value = cash_fixed + cash + (shares * current_open_price)

        if correct:
            correct_pred_count += 1

            if signal == "BUY":
                buy_correct_count += 1
            elif signal == "SELL":
                sell_correct_count += 1
            elif signal == "HOLD":
                hold_correct_count += 1
        else :
            wrong_pred_count += 1

        portfolio_history.append({
            'Date': bt_dates[i],
            'Signal': signal,
            'Co. Signal': correct_signal,
            'Correct': correct,

            'Open': current_open_price,
            'Actual': y_act,
            'Pred': predictions[i],
            'Action': action,
            'Shares': shares,
            'Cash': cash,
            'Cash Fixed': cash_fixed,
            'Total_Wallet': total_value,
            'Return (%)': ((total_value - initial_capital) / initial_capital) * 100,
            'No Trading : Cash': cash_when_not_traded,
            'No Trading : Return': ((cash_when_not_traded - initial_capital) / initial_capital) * 100,

            'SOLD when BUY':signal=="SELL" and correct_signal=="BUY",
            'BOUGHT when SELL': signal=="BUY" and correct_signal=="SELL",

            'Pred. Buys': buy_count,
            'Act. Buys': actual_buy_count,

            'Pred. Sells': sell_count,
            'Act. Sells': actual_sell_count,

            'Pred. Holds': hold_count,
            'Act. Holds': actual_hold_count,

            'Co. Buys': buy_correct_count,
            'Co. Sells': sell_correct_count,
            'Co. Holds': hold_correct_count,
            
            'Co. Dec': correct_pred_count,
            'Wr. Dec': wrong_pred_count,
        })
    result_df = pd.DataFrame(portfolio_history)
    return result_df

def plot_trend_backtest_result(df, print_detail=False):
    row_num = 5 if print_detail else 2
    figsize = (10, 12) if print_detail else (10, 6)
    fig, axs = plt.subplots(row_num, 1, figsize=figsize, sharex=True)

    other_cols = ["Shares", "Cash", "Cash Fixed"]
    other_colors = ["#DC143C", "#9370DB", "#4169E1"]
    
    buy_signals = df[df['Action'] == 'BUY']
    sell_signals = df[df['Action'] == 'SELL']

    axs[0].plot(df.index, df['Open'], label='Open', color='#4169E1', alpha=0.7)
    axs[0].scatter(buy_signals.index, buy_signals['Open'], 
                    marker='o', color='green', s=20, label='BOUGHT', edgecolor='black', zorder=5)
    axs[0].scatter(sell_signals.index, sell_signals['Open'], 
                    marker='o', color='red', s=20, label='SOLD', edgecolor='black', zorder=5)
    axs[0].set_ylabel('Price (Open)')
    axs[0].set_title('Open')
    axs[0].legend(loc='upper left')

    axs[1].plot(df.index, df['Return (%)'], label='Return using the model', color='#FFA500', alpha=0.7)
    axs[1].plot(df.index, df['No Trading : Return'], label='Return if not traded', color='#2E8B57', alpha=0.7)
    axs[1].axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
    axs[1].set_ylabel('Return %')
    axs[1].set_title('Return Using This Model vs Not Trading')
    axs[1].legend(loc='upper left')

    if (print_detail):
        for i, col_name in enumerate(other_cols):
            ax_idx = i + 2
            axs[ax_idx].plot(df.index, df[col_name], label=col_name, color=other_colors[i])
            axs[ax_idx].set_title(col_name)
            axs[ax_idx].legend(loc='upper left')

    for ax in axs:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ---------------------- Methods for BERT ----------------------