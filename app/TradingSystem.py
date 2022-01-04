from ticker import Ticker
import pandas as pd
from sqlalchemy import create_engine, text #Text to use simple sql statement
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


class TradingSystem:
    @staticmethod
    def create_db_connection():
        #set Connection sqlalchemy
        conn='postgresql+psycopg2://postgres:somePassword@192.168.1.240/TradePy'
        #conn='postgresql+psycopg2://trade_user:password@172.16.5.76/TradePy'
        #create engine
        engine = create_engine(conn)

        #set connection psycopg2
        pgconn= psycopg2.connect(
            host='192.168.1.240',
            database='TradePy',
            user='postgres',
            password='somePassword'
        )

        return engine,pgconn
    @staticmethod
    def insert_symbol_to_db(engine,symbol_list,force=False):
        """
        insert into db the ticker you want to manage, if exists the ticker will be
        delete and insert again with the last_update = 2000-01-01
        """
        if force:
            st = text('delete from stock.symbols where symbol = :s; insert into stock.symbols (symbol,last_update) values(:s,:d)')
        else:
            st = text('insert into stock.symbols (symbol,last_update) values(:s,:d) on conflict (symbol) do nothing;')
        for  symbols in symbol_list.itertuples(index=False):
            engine.execute(st, {"s":symbols[0],"d":"2000-01-01"})

    @staticmethod
    def delete_symbol_from_db(engine,symbol):
        st = text('delete from stock.symbols where symbol = :s; delete from stock.history where symbol = :s;')
        engine.execute(st, {"s":symbol})

    @staticmethod
    def insert_symbol_info_to_db(engine,ticker):
        st = text('update stock.symbols ' +
                    'set long_name = :ln ,'+
                    '    short_name = :sn ,'+
                    '    market = :m ,'+
                    '    quote_type = :qt ,'+
                    '    exchange = :e ,'+
                    '    price_to_book = :pb ,'+
                    '    trailing_pe = :tpe ,'+
                    '    beta = :bt ,'+
                    '    industry = :i ,'+
                    '    sector = :se '+
                    'where symbol = :sy'
                 )
        engine.execute(st, {"ln":ticker.longName,
                            "sn":ticker.shortName,
                            "m":ticker.market,
                            "qt":ticker.quoteType,
                            "e":ticker.exchange,
                            "i":ticker.industry,
                            "pb":ticker.priceToBook,
                            "tpe":ticker.trailingPE,
                            "bt":ticker.beta,
                            "sy":ticker.name,
                            "se":ticker.sector
                            })

    @staticmethod
    def get_symbol_from_db(engine):
        """
        get all the symbol from db using sqlalchemy
        """
        #df_symbol=pd.read_sql_table(table_name='symbols_start_update',schema='stock',con=engine)
        df_symbol=pd.read_sql_query(sql='select * from stock.symbols_start_update order by 2 asc',con=engine)
        return df_symbol
    @staticmethod
    def process_symbols(ticker_list,engine,pgconnection):
        yesterday = datetime.now() - timedelta(1)
        #df_hist=[]
        #for index, symbols in ticker_list.iterrows():
        for symbols in ticker_list.itertuples(index=False):
            start_at = time()
            tick=Ticker(symbols[0])
            df = tick.get_data_yahoo(symbols[2],yesterday) #SYMBOLS[2] = start date and not last_update
            df['RSI'] = tick.computeRSI(df['Adj Close'],14)
            df['volatility'] = tick.computeVolatility(df)
            df['hma'] = tick.computeHMA(df[['Adj Close','Open']])
            df['symbol']=tick.name
            df['last_update']=yesterday
            if df.empty:
                print(symbols[0]+" is empty")
                delete_symbol_from_db(engine,symbols[0])
            else:
                insert_processed_to_db(df,engine,pgconnection)
                if symbols[3] is None:
                    tick.get_info()
                    insert_symbol_info_to_db(engine,tick)
            print(symbols[0]+" processed in "+str(int(time()-start_at))+" seconds")
            #print(df.head())
            #df_hist.append(df)


        #df_hist = pd.concat(df_hist)
        #return df_hist
    @staticmethod
    def insert_processed_to_db(df_symb,engine,pgconnection):
        df_symb.to_sql('stock_history', schema='stage',con = engine, if_exists = 'replace', chunksize = 1000)
        engine.execute(text("CALL stock.insert_history()"))
        pgcursor= pgconnection.cursor()
        pgcursor.execute("CALL stock.insert_history()")
        pgconnection.commit()
