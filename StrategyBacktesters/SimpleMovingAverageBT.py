import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import yfinance as yf
from bidask import edge

class SimpleMovingAverageBT():
    ''' class to backtest stocks using Simple Moving Averages
    Attributes
    ----------
    symbol: str
        ticker symbol
    sma_s: int
        short window length
    sma_l: int
        long window length
    start: str
        start date
    end: str
        end date
    preprocessed_data: pandas dataframe
        Contains raw data that has not been backtested
    data: pandas dataframe
        Contains data after backtesting, contains more columns with different metrics
    optimisation_results: pandas dataframe
        Contains performance data of different short and long window combinations

    Methods
    -------
    set_window:
        Sets new window for SMA strategy
    backtest:
        Backtests strategy using simple moving averages
    plot_data:
        plots the data as a line graph
    optimise_windows:
        finds the optimal window based on specified ranges for the short and long window
    '''

    def __init__(self, symbol, sma_s, sma_l, start, end, interval = "1d"):
        '''
        parameters
        ----------
        symbol: str
            ticker symbol
        sma_s: int
            short window length
        sma_l: int
            long window length
        start: str
            start date
        end: str
            end date
        '''
        self.symbol = symbol
        self.sma_s = sma_s
        self.sma_l = sma_l
        self.start = start
        self.end = end
        self.interval = interval
        self.get_data()
        self.process_data()
    
    def __repr__(self):
        return "SimpleMovingAverageBT: {}, Short window = {}, Long window = {}".format(self.symbol, self.sma_s, self.sma_l)
    def get_data(self):
        '''
            Loads the stock data from the yfinance library
        '''
        df = yf.download(self.symbol, self.start, self.end, interval= self.interval)
        self.estimate_ptc(df)
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
    def estimate_ptc(self, df):
        spread = edge(df.Open, df.High, df.Low, df.Close)
        self.ptc = (spread/200) / df.Close.mean()

    def set_window(self, sma_s = None, sma_l = None):
        ''' Sets new window for SMA strategy
        parameters
        ----------
        sma_s : int
            short window
        sma_l: int
            long window
        '''
        if sma_s != None:
            self.sma_s = sma_s
        if sma_l != None:
            self.sma_l = sma_l
        self.data = self.preprocessed_data
        self.process_data()
        
    def backtest(self, print_res = True, return_res = False):
        ''' Backtests strategy using simple moving averages
        parameters
        ----------
        print_res: boolean, optional
            Set to False to remove output
        return_res: boolean, optional
            Set to True to return the absolute performance
        '''
        df = self.data.copy().dropna()
        df["positions"] = np.where(df.SMA_S>df.SMA_L, 1, -1)
        df["r_strategy"] = df.positions.shift(1) * df.returns
        df.dropna(inplace = True)
        df["trades"] = df["positions"].diff().fillna(0).abs()
        df["r_strategy_net"] = df.r_strategy - df.trades * self.ptc
        df["creturns"] = df.returns.cumsum().apply(np.exp) #convert the log returns
        df["cr_strategy"] = df.r_strategy.cumsum().apply(np.exp)
        df["cr_strategy_net"] = df.r_strategy_net.cumsum().apply(np.exp)
        self.result = df
        strategy_perf = round(df.cr_strategy_net.iloc[-1],4)
        strategy_outperf = round(df.cr_strategy_net.iloc[-1] - df.creturns.iloc[-1], 4)
        if print_res:
            print(f"SMA Net performance: {strategy_perf} || SMA outperformance vs Buy and Hold: {strategy_outperf}" )
        if return_res:
            return strategy_perf #returns the absolute performance to be stored/used elsewhere
    
    def plot_data(self):
        '''
            plots the data as a line graph
        '''
        if self.result is None:
            raise Exception("Run Backtest first, no results to plot!")
        self.result[["creturns","cr_strategy","cr_strategy_net"]].plot(figsize = (12,8), fontsize = 13)
        plt.title(f"{self.symbol} || SMA(short) = {self.sma_s} || SMA(Long) = {self.sma_l}", size = 20)
        plt.legend(fontsize = 13)
        plt.show()

    def optimise_windows(self, sma_s_range, sma_l_range):
        ''' finds the optimal window based on specified ranges for the short and long window
        parameters
        ----------
        sma_s_range: tuple
            Pass in a tuple of integers in the following format: (lower bound, upper bound, step)
        sma_l_range: tuple
            Pass in a tuple of integers in the following format: (lower bound, upper bound, step)
        '''
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
  
        


