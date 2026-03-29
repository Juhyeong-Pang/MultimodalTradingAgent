import logging

import pandas as pd
import numpy as np
import asyncio
from fastapi import FastAPI
from src.models.trader import trader

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

trader = trader()


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

@app.post("/reset")
async def reset(password: str):
    status = trader.reset_db(password)
    return "DB Reset" if status else "Failed to Reset"