import os
import sys
import json

import pandas as pd
import numpy as np

from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

base_path = os.path.dirname(os.getcwd()) 
if base_path not in sys.path:
    sys.path.append(base_path)

from src.prompts import SYSTEM_MSG, TEMPLATE, INSTRUCTION, INPUT_STR
from src.models.trend_analysis import TA_Model
from src.models.sentiment_analysis import SA_Model



class Decision_Model:
    def __init__(self):
        self.ta_model = TA_Model()
        self.sa_model = SA_Model()
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def predict(self, ticker, price, cash, share):
        trend_prediction_raw = self.ta_model.predict(ticker)["Prediction"]
        trend_prediction_raw = round(trend_prediction_raw, 2)
        trend_prediction = "+" + str(trend_prediction_raw) if (trend_prediction_raw >= 0) else str(trend_prediction_raw)

        sentiment_prediction_raw = self.sa_model.predict(ticker)
        positive_sen_ratio = round(sentiment_prediction_raw['Positive'], 2)
        neutral_sen_ratio = round(sentiment_prediction_raw['Neutral'], 2)
        negative_sen_ratio = round(sentiment_prediction_raw['Negative'], 2)


        lstm_output = f"LSTM model output : {trend_prediction}%"
        bert_output = f"BERT model output : {positive_sen_ratio}% Positive, {neutral_sen_ratio}% Neutral, {negative_sen_ratio}% Negative"
        macro_data = "Inflation Rate Higher than normal"
        open_price = price
        current_cash = cash
        shares_owned = share

        input_str = INPUT_STR.format(
            lstm_output=lstm_output,
            bert_output=bert_output,
            macro_data=macro_data,
            open_price=open_price,
            current_cash=current_cash,
            shares_owned=shares_owned
        )

        instruction = INSTRUCTION.format(
            input_str=input_str,
        )

        prompt = TEMPLATE.format(
            instruction=instruction,
        )

        response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
        return response.choices[0].message.content




