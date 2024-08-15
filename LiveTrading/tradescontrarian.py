#Scripy for trading using contrarian strategy

from ib_insync import *
import pandas as pd
import numpy as np
import datetime as dt
import os
from tabulate import tabulate

ib = IB()
ib.connect()

#strategy parameters (sma strategy)
window = 5
quantile_window = 10
quantile = 0.8 #closer value to 1 means we would need a more significant price change before we swap positions.
units = 1000 #number of units to trade per long or short pos taken
freq = '15 mins' # we will be accessing bars at 1 min freq
contract = Forex("EURUSD") #create contract for the Forex Ticker
ib.qualifyContracts(contract)
cfd = CFD("EUR", currency="USD") #create a cfd contract for trading
ib.qualifyContracts(cfd) #qualify the cfd contract
contractID = cfd.conId

def onBarUpdate(bars, hasNewBar): #what to do when we receive a new bar
    global df, last_bar #refers to global variables/created variables will be under the global scope

    if bars[-1].date > last_bar: #when there is a new bar, pass new data to strategy
        last_bar = bars[-1].date #update new last bar date

        #preprocess data
        df = pd.DataFrame(bars)[["date","open","high","low","close"]].iloc[:-1] #take everything but the current last i.e. we include the previous last bar as all new data will now be with bars[-1]
        df.set_index("date", inplace=True)

        ## Trading Strategy ##
        df["window"] = df.Price.rolling(window).mean()
        df["pc_direction"] = np.sign(df.window.diff())
        df["mag_pricechange"] = df["Price"].pct_change().abs().rolling(window).mean()
        df["mag_pcquantile"] = df["Price"].pct_change().abs().rolling(quantile_window).apply(lambda x: x.sort_values().quantile(quantile))
        df.dropna(inplace = True)
        df["positions"] =  -1 * np.sign(df.pc_direction)*np.where(df.mag_pricechange > df.mag_pcquantile, 1, 0) #if large price increase, go short, vice versa.
        ## Trading Strategy ##

        #trading
        target = df["position"][-1] * units #number of units to buy, sign inidicates long or short
        execute_trade(target = target)

        #Display 
        os.system('clear')
        trade_reporting()

def execute_trade(target):
    '''
        buy or sell positions based on specified number of units.
    '''
    global current_pos #to access the global current_pos variable which will be 0 when we start(start neutral)
    #1. get the current position
    try:
        current_pos = [pos.position for pos in ib.positions() if pos.contract.conId == contractID][0]
    except: #if currently no positions, will throw an error, we catch using try except
        current_pos = 0 #if no positions, set accordingly

    #2. calculate trade to make (qty and direction)
    trades = target - current_pos # sign >> direction, value >> qty

    
    #3. trade Execution

    if trades > 0:
        side = "BUY"
        order = MarketOrder(side,abs(trades))
        trade = ib.placeOrder(cfd,order)
    elif trades < 0:
        side = "SELL"
        order = MarketOrder(side,abs(trades))
        trade = ib.placeOrder(cfd,order)
    else:
        pass #dont trade if neutral

def trade_reporting():
    global report

    fill_df = util.df([fs.execution for fs in ib.fills()])[["execId","time","side","cumQty","avgPrice"]].set_index("execId")
    profit_df = util.df([fs.commissionReport for fs in ib.fills()])[["execId","realizedPNL","commission"]].set_index("execId")
    report = pd.concat([fill_df,profit_df], axis=1).set_index("time").loc[session_start:] #only take execution data from when we started trading
    report = report.groupby("time").agg({"side":"first","cumQty":"max","avgPrice":"mean","realizedPNL":"sum","commission":"sum"}) # trades may simultaneously occur e.g. going from long to short >> need to  aggregate them
    report["cumPNL"] = report.realizedPNL.cumsum() #track the cumulative PNL
    report["cumTC"] = report.commission.cumsum() #track cumulative trading costs
    time_now = "Time Now(UTC): " + str(dt.datetime.now(dt.UTC).time())
    end_time = "End Time(UTC): " + str(endtime)
    report_tabulate = tabulate(report, headers=["side","cumQTY","avgPrice","realizedPNL","commission","cumPNL","cumTC"],tablefmt="heavy_grid")
    os.system('clear')
    print(df,report_tabulate,time_now,end_time,sep="\n")


if __name__ == "__main__":
    #session start
    session_start = pd.to_datetime(dt.datetime.now(dt.UTC))
    endtime = dt.time(15,0,0)
    print(f"start time: {session_start} || end time: {endtime}")
    bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting=freq,
            whatToShow='MIDPOINT',
            useRTH=True,
            formatDate=2,
            keepUpToDate=True)
    last_bar = bars[-1].date
    bars.updateEvent += onBarUpdate #run function when we get a new bar

    #stop trading session (when requirements are met)
    while True:
        ib.sleep(5) #check every 5 seconds, sleep ib so that we stop receiving new ticks that may cause other codes to run
        if dt.datetime.now(dt.UTC).time() >= endtime: #if stop conditions are met(pass a certain time)
            execute_trade(target=0) #close all positions >> trades = target - current pos >> if long we go short vice versa.
            ib.cancelHistoricalData(bars) #stop stream
            ib.sleep(10)
            try:
                trade_reporting()
            except:
                pass
            print("Session stopped")
            ib.disconnect()
            break
        else: #if stop conditions not met, keep trading
            pass

