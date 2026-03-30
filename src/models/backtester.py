import os
import sys
import json
import sqlite3
import asyncio
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.decision_maker import Decision_Model

class backetester:
    def __init__(self):
        self.conn = sqlite3.connect("backtesting_trading_db.db")
        self.cursor = self.conn.cursor()
        self.init_db()
        self.trade_history = []

        self.decision_maker = Decision_Model()

    def init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS Account (
                id INTEGER PRIMARY KEY,
                cash REAL
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS Portfolio (
                ticker TEXT PRIMARY KEY,
                share INTEGER DEFAULT 0
            )
        ''')

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            message TEXT                    
        )''')

        self.cursor.execute("SELECT count(*) FROM Account")
        if self.cursor.fetchone()[0] == 0:
            self.cursor.execute("INSERT INTO Account (id, cash) VALUES (1, 100000)")
            self.conn.commit()
    
    async def trade(self, ticker, date, external_price=None):
        if external_price:
            price = external_price
        else:
            hist = yf.Ticker(ticker).history(period="1d")
            price = hist["Open"].iloc[0]

        cash = self.get_cash()
        share = self.get_share(ticker)

        response_raw = await self.decision_maker.predict(ticker, price, cash, share)
        print(response_raw)
        response =  json.loads(str(response_raw))
        action = response.get("action", "SELL")
        share = int(response.get("share", "30"))
        reason = response.get("reason", "-")

        if (action == "BUY"):
            self.buy(ticker, share, price, reason)
            self.trade_history.append({"date": date, "ticker": ticker, "action": "BUY", "price": price})
        elif (action == "SELL"):
            self.sell(ticker, share, price, reason)
            self.trade_history.append({"date": date, "ticker": ticker, "action": "SELL", "price": price})
        else:
            message = f"Decided to hold for {ticker}; Reason: {reason}"
            self.add_log(message)

        return {
            "status": "success",
            "ticker": ticker,
            "decision": response
        }
    
    async def run_backtest(self, tickers, start_date, end_date):
        print(f"--- Starting Backtest ({start_date} ~ {end_date}) ---")
        start_cash = self.get_cash()

        all_data = yf.download(tickers, start=start_date, end=end_date, group_by='column')

        if len(tickers) > 1:
            open_prices = all_data['Open']
        else:
            open_prices = pd.DataFrame({tickers[0]: all_data['Open']})

        cash_per_ticker = start_cash / len(tickers)
        buy_hold_shares = {}
        total_initial_buy_cost = 0

        first_prices = open_prices.iloc[0]
        for ticker in tickers:
            price = first_prices[ticker]
            share = cash_per_ticker // price
            buy_hold_shares[ticker] = share
            total_initial_buy_cost += (share * price)
        bh_remaining_cash = start_cash - total_initial_buy_cost

        for date, row in open_prices.iterrows():
            print(f"\n[Date: {date.date()}]")
            
            for ticker in tickers:
                current_price = row[ticker]
                if pd.isna(current_price):
                    continue
                    
                await self.trade(ticker, date, external_price=float(current_price))
            
            print(f"Cash: {self.get_cash():.2f}")

        print(f"\n--- End ---")

        last_prices = open_prices.iloc[-1]
        self.show_total_performance(start_cash, tickers, last_prices, bh_remaining_cash, buy_hold_shares)
        self.plot_trading_results(tickers, open_prices)

    def show_total_performance(self, start_cash, tickers, last_prices, bh_remaining_cash, bh_shares):
        status = self.get_full_status()
        current_cash = status['current_cash']
        strategy_portfolio_value = 0
        
        for item in status['portfolio']:
            t = list(item.keys())[0]
            if t in last_prices:
                share = item[t]['share']
                strategy_portfolio_value += (share * last_prices[t])
        
        strategy_total = current_cash + strategy_portfolio_value
        strategy_return = (strategy_total / start_cash - 1) * 100

        bh_portfolio_value = 0
        for ticker in tickers:
            bh_portfolio_value += bh_shares[ticker] * last_prices[ticker]

        bh_total = bh_remaining_cash + bh_portfolio_value
        bh_return = (bh_total / start_cash - 1) * 100

        print("\n" + "="*40)
        print(f"{' ':<15} | {'Final Asset':>10} | {'Return':>8}")
        print("-" * 40)
        print(f"{'AI Strategy':<15} | {strategy_total:>10.2f} | {strategy_return:>7.2f}%")
        print(f"{'Buy & Hold':<15} | {bh_total:>10.2f} | {bh_return:>7.2f}%")
        print("-" * 40)
        print(f"How AI performed better compared to BH: {strategy_return - bh_return:.2f}%p")
        print("="*40)

    def buy(self, ticker, share, price, reason):
        if share < 0 : return

        current_cash = self.get_cash()
        current_share = self.get_share(ticker)
        total_cost = share * price

        if (total_cost > current_cash):
            print("Not Enough Cash")
            return

        self.update_cash(current_cash - total_cost)
        self.update_share(ticker, current_share + share)

        updated_cash = self.get_cash()
        updated_share = self.get_share(ticker)
        message = f"""
        Bought {share} share of {ticker} for {price} each ({total_cost} total).
        Status:
         - cash: {updated_cash}
         - share: {updated_share}
        Reason: {reason}
        """

        self.add_log(message)
        

    def sell(self, ticker, share, price, reason):
        if share < 0 : return

        current_cash = self.get_cash()
        current_share = self.get_share(ticker)
        total_cost = share * price

        if (share > current_share):
            print("Not Enough Share")
            return

        self.update_cash(current_cash + total_cost)
        self.update_share(ticker, current_share - share)

        updated_cash = self.get_cash()
        updated_share = self.get_share(ticker)
        message = f"""
        Sold {share} share of {ticker} for {price} each ({total_cost} total).
        Status:
         - cash: {updated_cash}
         - share: {updated_share}
        Reason: {reason}
        """

        self.add_log(message)

    def get_cash(self):
        self.cursor.execute('SELECT cash FROM Account WHERE id = 1')
        row = self.cursor.fetchone()
        return row[0]
    
    def update_cash(self, value):
        self.cursor.execute('UPDATE Account SET cash = ? WHERE id = 1', (value,))
        self.conn.commit()

    def get_share(self, ticker):
        self.cursor.execute('SELECT share FROM Portfolio WHERE ticker = ?', (ticker,))
        row = self.cursor.fetchone()
        return row[0] if row else 0
    
    def update_share(self, ticker, value):
        self.cursor.execute("""
            INSERT INTO Portfolio (ticker, share) 
            VALUES (?, ?)
            ON CONFLICT(ticker) 
            DO UPDATE SET share = excluded.share
            """, (ticker, value)
        )
        self.conn.commit()

    def add_log(self, message):
        self.cursor.execute("INSERT INTO Log (message) VALUES (?)", (message, ))
        self.conn.commit()

    def get_all_logs(self):
        self.cursor.execute("SELECT * FROM Log ORDER BY id DESC")
        rows = self.cursor.fetchall()
        return [{"id": r[0], "timestamp": r[1], "message": r[2]} for r in rows]

    def get_full_status(self):
        cash = self.get_cash()
        
        self.cursor.execute("SELECT ticker, share FROM Portfolio WHERE share > 0")
        rows = self.cursor.fetchall()
        portfolio = [{r[0]: {"ticker": r[0], "share": r[1]}} for r in rows]

        return {
            "current_cash": cash,
            "portfolio": portfolio
        }
    
    def show_performance(self, ticker, start_cash, last_price):
        status = self.get_full_status()
        cash = status['current_cash']

        for portfolio in status['portfolio']:
            if portfolio == ticker:
                share = portfolio[ticker]['share']

        total_cash = cash + (share * last_price)
        total_return = total_cash / start_cash * 100
        
        print("\n" + "="*30)
        print("BACKTESTING RESULT")
        print(f"Cash: {cash:.2f}")
        print(f"Share: {share}")
        print(f"Total Cash: {total_cash:.2f}")
        print(f"Return: {total_return:.2f}")
        print("="*30)


    def plot_trading_results(self, tickers, df_prices):
        n = len(tickers)
        fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n), sharex=True)
        
        if n == 1: axes = [axes]

        for i, ticker in enumerate(tickers):
            ax = axes[i]
            ax.plot(df_prices.index, df_prices[ticker], label=f"{ticker} Price", color='gray', alpha=0.5)
            
            ticker_trades = [t for t in self.trade_history if t["ticker"] == ticker]
            
            buys = [t for t in ticker_trades if t["action"] == "BUY"]
            sells = [t for t in ticker_trades if t["action"] == "SELL"]

            if buys:
                ax.scatter([b["date"] for b in buys], [b["price"] for b in buys], 
                           marker='^', color='green', s=100, label='BUY Signal', zorder=5)

            if sells:
                ax.scatter([s["date"] for s in sells], [s["price"] for s in sells], 
                           marker='v', color='red', s=100, label='SELL Signal', zorder=5)

            ax.set_title(f"Trading Signals for {ticker}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        
        
        save_dir = os.path.join("assets", "graph")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_str = "_".join(tickers[:3]) 
        filename = f"result_{ticker_str}_{now}.png"
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi=300)
        print(f"Chart saved to: {save_path}")

    def reset_db(self, password):    
        if password == "IWANTTORESETMYDATABASE":
            print("Start Deleting...")
            
            try:
                self.cursor.execute("DELETE FROM Account")
                self.cursor.execute("DELETE FROM Portfolio")
                
                self.cursor.execute("INSERT INTO Account (id, cash) VALUES (1, 100000)")
                
                self.add_log("Database was reset")
                self.conn.commit()
                return True
                
            except sqlite3.Error as e:
                print(f"Error: {e}")
                self.conn.rollback() 
                return False
        else:
            print("Wrong Password")
            return False

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    async def main():
        tester = backetester()
        tester.reset_db("IWANTTORESETMYDATABASE")

        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX"
        ]
        
        await tester.run_backtest(
            tickers=tickers, 
            start_date="2025-02-01", 
            end_date="2025-02-28"
        )

    asyncio.run(main())