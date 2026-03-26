import os
import sys

import numpy as np

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.sentiment_analysis_methods import load_SA_model, encode

class SA_Model():
    def __init__(self):
        self.model = load_SA_model()
        self.le = LabelEncoder()

    def predict(self, text):
        text_encoded = encode(text)
        # print("Encoded Text: ", text_encoded)
        
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