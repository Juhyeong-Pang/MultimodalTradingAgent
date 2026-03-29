import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.trend_analysis_methods import load_TA_model, get_data_for_prediction

class TA_Model():
    '''
        Prediction Method: predict(ticker, period="120d",window_size=60)
         - Insert ticker, then it wil output predicted Return for the next day
         - Output Format : dictionary containing prediction in "Prediction"
    '''
    def __init__(self):
        self.model = load_TA_model(window_size=60, feature_num=18)

    def predict(self, ticker, period="120d",window_size=60):
        x = get_data_for_prediction(ticker, period=period, window_size=window_size)
        result = self.model.predict(x)[0][0].item() # Expectation: Single Predicted Return Value 

        result_dict = {
            "Prediction": result
        }

        return result_dict