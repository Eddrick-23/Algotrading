import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import yfinance as yf
from bidask import edge

#trading costs too high, consider trading on margin?

class MomentumBT():

    def __init__(self, symbol, window, start, end, interval = "1d"):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        window: int
            time window (number of bars) to be used for strategy
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.window = window
        self.start = start
        self.end = end
        self.interval = interval
        self.get_data()
        self.process_data()
    
    def get_data(self):
        df = yf.download(self.symbol, self.start, self.end, interval=self.interval)
        self.half_spread = edge(df.Open,df.High, df.Low,df.Close)
        df = df.Close.to_frame()
        df.rename(columns = {"Close":"Price"}, inplace = True)
        self.preprocessed_data = df #to be used for changing windows so we don't have to repeatedly call yf.download
        self.data = df
    def process_data(self):
        '''
            Calculates important metrics like returns,rolling averages etc. for further analysis
        '''
        df = self.data.copy()
        df["returns"] = np.log(df.Price/df.Price.shift(1))
        df["window"] = df.Price.rolling(self.window).mean()
        self.data = df
    def set_window(self, window = None):
        if window is not None:
            self.window = window
            self.data = self.preprocessed_data
            self.process_data()
    def backtest(self, print_res = True, return_res = False):
        #look at price changes, if price is increasing, go long, else go short
        df = self.data.copy()
        df["price_diff"] = df.window.diff()
        df["price_pct_change"] = df.window.pct_change()
        df.dropna(inplace = True)
        df["positions"] = np.sign(df.price_pct_change) * np.where(df.price_pct_change > df.price_pct_change.sort_values().quantile(0.9), 1, -1)
        df["trades"] = df.positions.diff().fillna(0).abs()
        df["creturns"] = df.returns.cumsum().apply(np.exp)
        df["strategy"] = df.positions.shift(1) * df.returns
        df["strategy_net"] = df.strategy - df.trades*self.half_spread
        df["cstrategy"] = df.strategy.cumsum().apply(np.exp)
        df["cstrategy_net"] = df.strategy_net.cumsum().apply(np.exp)

        self.result = df
        strategy_perf = round(df.cstrategy_net.iloc[-1],4)
        strategy_outperf = round(df.cstrategy_net.iloc[-1] - df.creturns.iloc[-1], 4)
        if print_res:
            print(f"Strategy Net performance: {strategy_perf} || Strategy outperformance vs Buy and Hold: {strategy_outperf}" )
        if return_res:
            return strategy_perf #returns the absolute performance to be stored/used elsewhere
    def plot_data(self):
        '''
            plots the data as a line graph
        '''
        if self.result is None:
            raise Exception("Run Backtest first, no results to plot!")
        self.result[["creturns","cstrategy","cstrategy_net"]].plot(figsize = (12,8), fontsize = 13)
        tc = round(self.half_spread,4)
        plt.title(f"{self.symbol} || Momentum Window = {self.window} || TC(% of price traded) = {tc}", size = 20)
        plt.legend(fontsize = 13)
        plt.show()
    def optimise_strategy(self, window_range):
        '''
            window_range: int
                window size to be used
        '''
        windows = range(1,window_range+1)
        results = []
        for w in windows:
            self.set_window(w)
            r = self.backtest(print_res = False,return_res=True)
            results.append(r)
        best = max(results)
        optimal_window = windows[np.argmax(results)]
        self.optimisation_results = pd.DataFrame(data = windows, columns=["Window"])
        self.optimisation_results["performance"] = results
        print(f"Optimal Window : {optimal_window} || Performance : {round(best,4)}")

