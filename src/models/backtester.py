import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.trader import trader
# from src.utils.backtesting_methods import 

class backetester:
    def __init__(self):
        self.trader = trader()

    def backtest(self):
        pass

    