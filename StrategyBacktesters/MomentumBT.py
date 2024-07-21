import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import yfinance as yf
from bidask import edge

#trading costs too high, consider trading on margin?

class MomentumBT():
    ''' class to backtest stocks using Momentum strategy
    Attributes
    ----------
    symbol: str
            ticker symbol (instrument) to be backtested
        window: int
            time window (number of bars) to be used for strategy
        start: str
            start date for data import
        end: str
            end date for data import
        interval: str, optional
            time intervals for price data
        leverage: int, optional
            leverage that we trade at
        quantile: float, optional
            Value must be in the range (0,1)
            Higher quantiles mean that more significant price changes required before taking a long position.
        half_spread:
            percentage trading cost per dollar traded
        preprocessed_data: pandas dataframe
            Contains raw data that has not been backtested
        data: pandas dataframe
            Contains data after backtesting, contains more columns with different metrics
        optimisation_results: pandas dataframe
            Contains performance data of different short and long window combinations
    Methods
    -------
    set_parameters:
        Sets new parameters for Momentum Strategy
    backtest:
        Backtests strategy
    plot_data:
        plots the data as a line graph
    optimise_parameters:
        finds optimal window size, leverage, quantile based on specified ranges
    '''

    def __init__(self, symbol, window, start, end, interval = "1d",leverage = 1, quantile = 0.5):
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
        interval: str, optional
            time intervals for price data
        leverage: int, optional
            leverage that we trade at
        quantile: float, optional
            Value must be in the range (0,1)
            Higher quantiles mean that more significant price changes required before taking a long position.
        '''
        self.symbol = symbol
        self.window = window
        self.start = start
        self.end = end
        self.interval = interval
        self.leverage = leverage
        self.quantile = quantile
        self.get_data()
        self.process_data()
    def __repr__(self):
        return "MomentumBT: {}, window = {}, quantile = {}, leverage = {}".format(self.symbol,self.window, self.quantile, self.leverage)
    def get_data(self):
        '''
            Loads stock data from the yfinance library
        '''
        df = yf.download(self.symbol, self.start, self.end, interval=self.interval)
        self.half_spread = edge(df.Open,df.High, df.Low,df.Close)/2
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
        df["lev_returns"] =  np.log(df.Price.pct_change()*self.leverage+1)
        df["window"] = df.Price.rolling(self.window).mean()
        self.data = df
    def set_parameters(self, window = None, quantile = None, leverage = None):
        ''' Sets new parameters
        parameters
        ----------
        window: int
            window size to compare price changes
        quantile: float
            Must be in the range (0,1)
            Higher quantiles would mean we only go long for more significant price changes
        leverage: int
            leverage used for trading
        '''
        if window is not None:
            self.window = window
            self.data = self.preprocessed_data
            self.process_data()
        if quantile is not None:
            self.quantile = quantile
        if leverage is not None:
            self.leverage = leverage
    def backtest(self, print_res = True, return_res = False):
        ''' Backtests strategy using price momentum changes
        parameters
        ----------
        print_res: boolean, optional
            Set to False to remove output
        return_res: boolean, optional
            Set to True to return the absolute performance
        '''
        df = self.data.copy()
        df["price_diff"] = df.window.diff()
        df["mag_pricechange"] = df.window.pct_change().abs()
        df.dropna(inplace = True)
        df["positions"] = np.sign(df.mag_pricechange) * np.where(df.mag_pricechange > df.mag_pricechange.sort_values().quantile(self.quantile), 1, -1)
        df["trades"] = df.positions.diff().fillna(0).abs()
        df["creturns"] = df.returns.cumsum().apply(np.exp)
        df["clev_returns"] = df.lev_returns.cumsum().apply(np.exp)
        df["strategy"] = df.positions.shift(1) * df.lev_returns #use levered returns
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
        self.result[["creturns","clev_returns","cstrategy","cstrategy_net"]].plot(figsize = (12,10), fontsize = 13)
        tc = round(self.half_spread,4)
        plt.title(f"{self.symbol} || Leverage = {self.leverage} || Momentum Window = {self.window} ||Quantile = {round(self.quantile,4)} || TC(% of price traded) = {tc}", size = 13)
        plt.legend(fontsize = 13)
        plt.show()
    def optimise_strategy(self, window_range, qrange, leverage_range):
        '''
            window_range: int
                window size to be used
            quantile range: tuple
                (lower limit, upper limit, number of points, endpoint)
            leverage range: tuple
                (lower limit, upper limit)
        '''
        windows = list(range(1,window_range+1))
        quantiles = list(np.linspace(qrange[0],qrange[1],qrange[2],qrange[3]))
        leverages = list(range(leverage_range[0],leverage_range[1]+1))
        comb = list(product(windows,quantiles,leverages))

        results = []
        for c in comb:
            self.set_parameters(window=c[0],quantile=c[1], leverage = c[2])
            r = self.backtest(print_res = False,return_res=True)
            results.append(r)
        best = max(results)
        optimal_window, quantile, leverage = comb[np.argmax(results)]
        self.optimisation_results = pd.DataFrame(data = comb, columns=["Window","Quantile","Leverage"])
        self.optimisation_results["performance"] = results
        print(f"Optimal Window : {optimal_window} ||Quantile : {round(quantile,4)} || Leverage = {leverage} || Performance : {round(best,4)}")

