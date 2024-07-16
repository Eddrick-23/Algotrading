import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import yfinance as yf

class SimpleMovingAverageBT():
    def __init__(self, symbol, sma_s, sma_l, start, end):
        self.symbol = symbol
        self.sma_s = sma_s
        self.sma_l = sma_l
        self.start = start
        self.end = end
        self.get_data()
        self.process_data()
    
    def __repr__(self):
        return "SimpleMovingAverageBT: {}, Short window = {}, Long window = {}".format(self.symbol, self.sma_s, self.sma_l)
    def get_data(self):
        '''
            Loads the stock data from the yfinance library
        '''
        df = yf.download(self.symbol, self.start, self.end)
        df = df.Close.to_frame() #we will work with close prices only
        df.rename(columns = {"Close":"Price"}, inplace = True)
        self.preprocessed_data = df #to be used for changing windows so we don't have to repeatedly call yf.download
        self.data = df
    def process_data(self):
        '''
            Calculates important metrics like returns,rolling averages etc. for further analysis
        '''
        df = self.data.copy() #always make a copy first to prevent overriding data before calculations
        df["returns"] = np.log(df.Price/df.Price.shift(1))
        df["SMA_S"] = df.Price.rolling(self.sma_s).mean()
        df["SMA_L"] = df.Price.rolling(self.sma_l).mean()
        self.data = df
    def set_window(self, sma_s = None, sma_l = None):
        if sma_s != None:
            self.sma_s = sma_s
        if sma_l != None:
            self.sma_l = sma_l
        self.data = self.preprocessed_data
        self.process_data()
        
    def backtest(self, print_res = True, return_res = False):
        df = self.data.copy().dropna()
        df["positions"] = np.where(df.SMA_S>df.SMA_L, 1, -1)
        df["r_strategy"] = df.positions.shift(1) * df.returns
        df.dropna(inplace = True)
        df["creturns"] = df.returns.cumsum().apply(np.exp) #convert the log returns
        df["cr_strategy"] = df.r_strategy.cumsum().apply(np.exp)
        self.result = df
        strategy_perf = round(df.cr_strategy.iloc[-1],4)
        strategy_outperf = round(df.cr_strategy.iloc[-1] - df.creturns.iloc[-1], 4)
        if print_res:
            print(f"SMA absolute performance: {strategy_perf} || SMA outperformance vs Buy and Hold: {strategy_outperf}" )
        if return_res:
            return strategy_perf #returns the absolute performance to be stored/used elsewhere
    
    def plot_data(self):
        if self.result is None:
            raise Exception("Run Backtest first, no results to plot!")
        self.result[["creturns","cr_strategy"]].plot(figsize = (12,8), fontsize = 13)
        plt.title(f"{self.symbol} || SMA(short) = {self.sma_s} || SMA(Long) = {self.sma_l}", size = 20)
        plt.legend(fontsize = 13)
        plt.show()

    def optimise_windows(self, sma_s_range, sma_l_range):
        #find optimal short and long term window combination based on a given input range
        all_combinations = list(product(range(*sma_s_range),range(*sma_l_range)))
        results = []
        for comb in all_combinations:
            sma_s, sma_l = comb[0], comb[1] #get the windows
            self.set_window(sma_s,sma_l)
            res = self.backtest(print_res=False, return_res=True)
            results.append(res)
        best = max(results)
        optimal_window = all_combinations[np.argmax(results)]
        self.optimisation_results = pd.DataFrame(data = all_combinations, columns=["SMA_S", "SMA_L"])
        self.optimisation_results["performance"] = results

        print(f"Optimal Window(short, long) : {optimal_window[0]},{optimal_window[1]} || Performance : {round(best,4)}")
  
        


