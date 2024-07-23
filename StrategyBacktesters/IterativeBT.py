import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from bidask import edge
plt.style.use("seaborn-v0_8")

class IterativeBase():

    def __init__(self, symbol, start, end, amount, interval = "1d", use_TC = True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.initial_balance = amount
        self.current_balance  = amount
        self.units = 0 #units of stock held
        self.trades = 0 # number of trades we make
        self.position = 0 #start from neutral position
        self.use_TC = use_TC
        self.last_traded_price = None
        self.get_data()

    def get_data(self):
        df = yf.download(self.symbol, self.start, self.end, interval=self.interval)
        pct_spread = edge(df.Open, df.High, df.Low, df.Close)
        df = df.Close.to_frame()
        df["spread"] = df.Close * pct_spread
        df.rename(columns = {"Close":"price"}, inplace = True)
        self.data = df
    def plot_data(self, cols = None):
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize = (12,8), title = self.symbol)
        plt.legend(fontsize = 13)
    
    def get_bar(self,bar):
        date = str(self.data.index[bar].date())
        price = round(self.data.price.iloc[bar],5)
        spread = round(self.data.spread.iloc[bar],5)
        return date, price, spread
    
    def get_currBalance(self,bar):
        date,price,spread = self.get_bar(bar)
        print(f"{date} | Current Balance: {round(self.current_balance,2)}")
    
    def buy_stock(self,bar,units = None, amount = None):
        date,price,spread = self.get_bar(bar)
        if self.use_TC: #adjust price to ask price
            price += spread/2
        if amount is not None:
            units = int(amount/price)
        self.current_balance -= price * units 
        self.units += units
        self.trades += 1
        print(f"{date} | Buying {units} units for {round(price,5)}")
    
    def sell_stock(self,bar, units = None, amount = None):
        date,price,spread = self.get_bar(bar)
        if self.use_TC: #adjust price to Bid price
            price -= spread/2
        if amount is not None:
            units = int(amount/price)
        self.current_balance += price * units
        self.units -= units
        self.trades += 1
        print(f"{date} | Selling {units} units for {round(price,5)}")

    def print_current_position_value(self,bar):
        date,price,spread = self.get_bar(bar)
        curr_pos_value = self.units * price
        print(f"{date} | Current Position Value = {round(curr_pos_value,2)}")

    def print_current_nav(self,bar): #get the net asset/account value
        date,price,spread = self.get_bar(bar)
        nav = self.current_balance + self.units*price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))

    def close_pos(self,bar):
        date,price,spread = self.get_bar(bar)
        print(80* "#")
        print(f"+++ CLOSING FINAL POSITION AT {date} +++")
        self.current_balance += self.units * price
        if self.units < 0: # if short, say closing short pos
            print("{} | closing short position of {} units for {}".format(date, -1 * self.units, price))
        else: #if long, say closing long pos
            print("{} | closing long position of {} units for {}".format(date, self.units, price))
        self.units = 0
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        self.get_currBalance(bar)
        print("{} | Profit & Loss = {}".format(date, round(self.current_balance - self.initial_balance, 2) ))      
        print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(80 * "#")

class IterativeBacktest(IterativeBase):
    '''
    Backtester class to backtest different trading strategies
    Inherits helper methods and attributes from the IterativeBase class.
    '''

    #helper methods to take long and short positions

    def go_long(self, bar, units = None, amount = None):
        if self.position == -1 : #if currently short go neutral first
            self.buy_stock(bar, units = self.units)
        if amount is not None:
            if amount == "all":
                amount = self.current_balance
            self.buy_stock(bar, amount=amount)
        else:
            self.buy_stock(bar, units=units)
    
    def go_short(self, bar, units = None, amount = None):
        if self.position == 1 : #if currently long go neutral first
            self.sell_stock(bar, units = self.units)
        #once we are neutral, take long position
        if amount is not None:
            if amount == "all":
                amount = self.current_balance
            self.sell_stock(bar, amount=amount)
        else:
            self.sell_stock(bar, units=units)
    
    def reset_data(self):
        self.position = 0
        self.current_balance = self.initial_balance
        self.trades = 0
        self.get_data() #load the dataset
    
    def test_sma(self, sma_s, sma_l):

        print(f"Testing SMA Strategy | Short Window = {sma_s} | Long Window = {sma_l}")

        self.reset_data()

        self.data["SMA_S"] = self.data.price.rolling(sma_s).mean()
        self.data["SMA_L"] = self.data.price.rolling(sma_l).mean()
        self.data.dropna(inplace = True)

        #trade using strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar) --> we close all pos at the last bar!
            if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: # signal to go long
                if self.position in [0, -1]: #if curr position is short/neutral, we go long
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
            elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: # signal to go short
                if self.position in [0, 1]: #if curr position is neutral/long, we go short.
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
        self.close_pos(bar + 1)
    
    def test_contrarian(self, window, quantile_window, quantile):
        print(f"Test Contrarian Strategy | window = {window} | quantile window = {quantile_window} | quantile = {quantile}")

        self.reset_data()

        #calculate required metrics
        self.data["window"] = self.data.price.rolling(window).mean()
        self.data["price_change_dir"] = np.sign(self.data.price.pct_change())
        self.data["mag_price_change"] = self.data.price.pct_change().abs().rolling(window).mean()
        self.data["mag_pcquantile"] = self.data.price.pct_change().abs().rolling(quantile_window).apply(lambda x:x.sort_values().quantile(quantile))
        self.data.dropna(inplace = True)

        for bar in range(len(self.data)-1):
            #we get a large momentum trend
            if self.data.mag_price_change.iloc[bar] > self.data.mag_pcquantile.iloc[bar]:
                if self.data.price_change_dir.iloc[bar] == 1 and self.position != -1: #large price increase, go short
                    self.go_short(bar, amount="all")
                    self.position = -1
                elif self.data.price_change_dir.iloc[bar] == -1 and self.position != 1: #large price fall, go long
                    self.go_long(bar, amount = "all")
                    self.position = 1
        
        self.close_pos(bar+1)