

#%%

"""
https://towardsdatascience.com/free-stock-data-for-python-using-yahoo-finance-api-9dafd96cad2e
occhio soprattutto allo sleep per evitare di essere bloccati
"""

from numpy import index_exp
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import json
from yfinance import ticker
from sqlalchemy import create_engine
import pandas as pd

class Ticker:
    """ Ticker class to get data from stock using yfinance """
    def __init__(self, symbol):
        self.s = yf.Ticker(symbol)
        self.name = symbol


    def get_info(self):
        """ Return general info """
        info=json.loads(json.dumps(self.s.info))

        try:
            self.longName = info['longName']
        except:
            self.longName = ''

        try:
            self.shortName = info['shortName']
        except:
            self.shortName = ''

        try:
            self.market = info['market']
        except:
            self.market = ''

        try:
            self.quoteType = info['quoteType']
        except:
            self.quoteType = ''


        try:
            self.sharesOutstanding = info['sharesOutstanding']
        except:
            self.sharesOutstanding = 0

        try:
            self.priceToBook = info['priceToBook']
        except:
            self.priceToBook = .0

        try:
            self.exchange = info['exchange']
        except:
            self.exchange = ''



        try:
            self.trailingPE = info['trailingPE']
        except:
            self.trailingPE = .0

        try:
            self.beta = info['beta']
        except:
            self.beta = .0

        try:
            self.industry = info['industry']
        except:
            self.industry = ''

        try:
            self.sector = info['sector']
        except:
            self.sector = ''



        return info

    def get_hist_max(self):
        """ historical data in pandas dataframe """
        return self.s.history(period="max")

    def get_data_yahoo(self,date_begin,date_end):
        """ Return data from range of date using pandas datareader """
        yf.pdr_override()
        dfy=pdr.get_data_yahoo(self.name,start=date_begin,end=date_end)
        adj=dfy['Adj Close']/dfy['Close']
        df_out = dfy.copy() #dovrei evitare il warning di sovrascrittura del DataFrame
        df_out['adj_open'] = adj*dfy['Open']
        df_out['adj_low'] = adj*dfy['Low']
        df_out['adj_high'] = adj*dfy['High']
        df_out['MM50'] = dfy['Adj Close'].rolling(50).mean()
        df_out['MM25'] = dfy['Adj Close'].rolling(25).mean() #media mobile semplice
        df_out['MM12'] = dfy['Adj Close'].rolling(12).mean() #media mobile semplice
        df_out['EMA25'] = pd.Series.ewm(dfy['Adj Close'], span=25).mean() #media mobile esponenziale
        df_out['EMA12'] = pd.Series.ewm(dfy['Adj Close'], span=12).mean() #media mobile esponenziale
        df_out['MACD'] = pd.Series.ewm(dfy['Adj Close'], span=25).mean()-pd.Series.ewm(dfy['Adj Close'], span=12).mean()
        df_out['signalMACD'] = pd.Series.ewm(df_out['MACD'], span=9).mean() #signal MACD è il segnale del MACD
        #Stochastic oscillator
        df_out['14-high'] = dfy['High'].rolling(14).max()
        df_out['14-low'] = dfy['Low'].rolling(14).min()
        df_out['perc_K'] = (dfy['Adj Close'] - df_out['14-low'])*100/(df_out['14-high'] - df_out['14-low'])
        df_out['perc_D'] = df_out['perc_K'].rolling(3).mean()
        #Bollinger Band
        df_out['BollingerUp'] = dfy['Adj Close'].rolling(window=20).mean() + (dfy['Adj Close'].rolling(window=20).std() * 2)
        df_out['BollingerDown'] = dfy['Adj Close'].rolling(window=20).mean() - (dfy['Adj Close'].rolling(window=20).std() * 2)
        df_out['tether_line'] = (df_out['adj_high'].rolling(20).max()+df_out['adj_low'].rolling(20).min())/2
        df_out['aroon_up'] = dfy['Adj Close'].rolling(25).apply(lambda x: float(np.argmax(x)+1)/25*100) #25 periods
        df_out['aroon_down'] = dfy['Adj Close'].rolling(25).apply(lambda x: float(np.argmin(x)+1)/25*100) #25 periods
        df_out['typic_price'] = (df_out['adj_high']+df_out['adj_low']+df_out['Adj Close'])/3
        df_out['cci'] = (df_out['typic_price']-df_out['typic_price'].rolling(20).mean())/(0.015*df_out['typic_price'].rolling(20).std()) #20 periods 0.015 is a constant
        #df_out['volatility'] = (np.log(df_out['Adj Close']/df_out['Adj Close'].shift(1))).rolling(30).std() * np.sqrt(252)
        df_out['Result_10_periods'] = dfy['Adj Close'].shift(-10)
        df_out['Result_30_periods'] = dfy['Adj Close'].shift(-30)
        df_out['Result_50_periods'] = dfy['Adj Close'].shift(-50)
        #df_out['RSI'] = Ticker.RSI(dfy['Adj Close'],14)
        return df_out

    def get_dividends(self):
        return self.s.dividends

    def get_splits(self):
        return self.s.splits



    @staticmethod
    def RSI(series,n):
        delta = series.diff()
        u = delta * 0
        d = u.copy()
        i_pos = delta > 0
        i_neg = delta < 0
        u[i_pos] = delta[i_pos]
        d[i_neg] = delta[i_neg]
        rs = pd.Series.ewm(u, span=n).mean() / pd.Series.ewm(d, span=n).mean()
        return 100 - (100 / (1 + rs))

    @staticmethod
    def computeRSI (data, time_window):
        diff = data.diff(1).dropna()        # diff in one field(one day)

        #this preservers dimensions off diff values
        up_chg = 0 * diff
        down_chg = 0 * diff

        # up change is equal to the positive difference, otherwise equal to zero
        up_chg[diff > 0] = diff[ diff>0 ]

        # down change is equal to negative deifference, otherwise equal to zero
        down_chg[diff < 0] = diff[ diff < 0 ]

        # check pandas documentation for ewm
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
        # values are related to exponential decay
        # we set com=time_window-1 so we get decay alpha=1/time_window
        up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
        down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()

        rs = abs(up_chg_avg/down_chg_avg)
        rsi = 100 - 100/(1+rs)
        return rsi
    @staticmethod
    def computeVolatility(df_input):
        df = df_input.copy()
        df['Log_Ret']= np.log(df['Adj Close']/df['Adj Close'].shift(1))
        df['Volatility']=(df['Log_Ret'].rolling(30).std())*np.sqrt(252)
        return df['Volatility']

    @staticmethod
    def computeHMA(df_input):
        #periods
        n=20
        p=int(n/2)
        radq_n=int(np.sqrt(n))

        #weights
        weights=np.linspace(0,n,n)
        weights_p=np.linspace(0,p,p)
        weights_sq=np.linspace(0,radq_n,radq_n)

        df = df_input.copy()
        #WMAs
        df['w_a']=2*df['Adj Close'].rolling(p).apply(lambda x: np.sum(weights_p*x)/ np.sum(weights_p))
        df['w_b']=df['Adj Close'].rolling(n).apply(lambda x: np.sum(weights*x)/ np.sum(weights))
        df['w_c'] = df['w_a']-df['w_b']
        df['hma'] = df['w_c'].rolling(radq_n).apply(lambda x: np.sum(weights_sq*x)/ np.sum(weights_sq))
        return df['hma']
