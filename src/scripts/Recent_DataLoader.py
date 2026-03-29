import os
import sys

import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.trend_analysis_methods import get_data_for_prediction
from src.utils.sentiment_analysis_methods import get_news_for_prediction

class Recent_DataLoader:
    def get_trend_dataset(self, ticker, period="120d", window_size=60):
        try:
            return get_data_for_prediction(ticker, period=period, window_size=window_size)
        except:
            print("Ticker Not Found")

    def get_sentiment_dataset(ticker):
        return  get_news_for_prediction(ticker)
