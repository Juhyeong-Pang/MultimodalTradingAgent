import os
import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from pygooglenews import GoogleNews
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.sentiment_analysis_methods import (
    load_SA_model, encode, initialize_le,
    get_news_for_prediction, augment_text, 
)


class SA_Model():
    '''
        Prediction Method: predict(ticker)
         - Insert ticker, then it will output analysis of News Articles
         - Output Format : dictionar with three columns : "Positive", "Neutral", "Negative"
    '''
    def __init__(self):
        self.model = load_SA_model()
        self.le = initialize_le()
        self.gn = GoogleNews(lang='en', country='US')
        self.conn = sqlite3.connect("trading_db.db")
        self.cursor = self.conn.cursor()

    def predict(self, ticker):
        news_articles = get_news_for_prediction(ticker, self.gn)
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        tot = len(news_articles)
        tot = tot if tot > 0 else 1
        for entry in news_articles:
            result = self.predict_single(entry['title'])
            if result is not None:
                if ("positive" in result):
                    positive_count += 1
                elif ("negative" in result):
                    negative_count += 1
                elif ("neutral" in result):
                    neutral_count += 1
        
        result_dict = {
            "Positive": positive_count/tot,
            "Neutral": neutral_count/tot,
            "Negative": negative_count/tot,
        }

        return result_dict

    def predict_single(self, text):
        text_encoded = encode(text)
        # print("Encoded Text: ", text_encoded)
        
        text_encoded = np.expand_dims(text_encoded, axis=0)
        pred = self.model(text_encoded, training=False)

        probabilities = tf.nn.softmax(pred, axis=-1).numpy()[0]

        pred_class = np.argmax(probabilities)
        sentiment = self.le.inverse_transform([pred_class])[0]
        confidence_score = probabilities[pred_class]

        class_names = self.le.classes_
        all_probs = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return sentiment, confidence_score, all_probs
    
    def predict_and_print(self, text):
        new_texts = [text]
        result, confidence, all_probs = self.predict(new_texts)
        
        print(f"Input Headline: \"{text}\"")
        print(f"Top Prediction: {result} ({confidence * 100:.2f}%)")
        print("-" * 30)
        print("Class Probabilities:")
        
        sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
        
        for label, prob in sorted_probs:
            bar = "#" * int(prob * 20) 
            print(f" - {label:<12}: {prob * 100:>6.2f}% {bar}")

        return result