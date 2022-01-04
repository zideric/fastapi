from fastapi import FastAPI


from ticker import Ticker
from TradingSystem import TradingSystem as ts
import pandas as pd
from datetime import datetime, timedelta



app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
   
@app.put("/symbol/{symbol}")
def insert_symbol(symbol: str = None):
    db_engine ,pgconn = ts.create_db_connection()
    symbol_list=[symbol]
    df=pd.DataFrame(symbol_list)
    ts.insert_symbol_to_db(db_engine,df)
    return {"symbol": symbol}