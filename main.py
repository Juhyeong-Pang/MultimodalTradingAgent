import logging

from fastapi import FastAPI

from src.models.backtester import backetester
from src.models.trader import trader

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

trader = trader()
backtester = backetester()


@app.post("/trade")
async def trade(ticker: str):
    return await trader.trade(ticker)


@app.get("/status")
async def get_status():
    return trader.get_full_status()


@app.get("/logs")
async def get_logs():
    return trader.get_all_logs()


@app.get("/portfolio/{ticker}")
async def get_ticker_share(ticker: str):
    share = trader.get_share(ticker)
    return {"ticker": ticker, "share": share}


@app.post("/backtest")
async def backtest(tickers: list[str], start_date: str, end_date: str):
    """
    - tickers should be in the format of ["ticker1", "ticker2", "ticker3"]
       - tickers MUST BE A VALID TICKER
    - start_date & end_date should be on the format of "y-m-d" ex: "2024-01-31"

    Example:
     - tickers = [ "V", "MA", "HSBC", "RY", "TD" ]
     - start_date = "2024-01-01"
     - end_date = "2024-01-31"
    """
    return await backtester.run_backtest(tickers, start_date, end_date)


@app.post("/reset")
async def reset(password: str):
    status = trader.reset_db(password)
    return "DB Reset" if status else "Failed to Reset"
