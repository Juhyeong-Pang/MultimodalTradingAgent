import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.trend_analysis_methods import load_TA_model

class TA_Model():
    def __init__(self):
        self.model = load_TA_model()