import json
import sqlite3
import yfinance as yf
from models.decision_maker import Decision_Model

class trader:
    def __init__(self):
        self.conn = sqlite3.connect("trading_db.db")
        self.cursor = self.conn.cursor()
        self.init_db()

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
    
    async def trade(self, ticker):
        hist = yf.Ticker(ticker).history(period="1d")
        if hist.empty:
            print(f"Data not found for {ticker}")
            return
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
        elif (action == "SELL"):
            self.sell(ticker, share, price, reason)
        else:
            message = f"Decided to hold for {ticker}; Reason: {reason}"
            self.add_log(message)

        return {
            "status": "success",
            "ticker": ticker,
            "decision": response
        }
        

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
        portfolio = [{"ticker": r[0], "share": r[1]} for r in rows]

        return {
            "current_cash": cash,
            "portfolio": portfolio
        }

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
