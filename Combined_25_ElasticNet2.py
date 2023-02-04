import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats
import requests as request
from bs4 import BeautifulSoup
from datetime import datetime
import time
import csv
import pickle
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from sklearn.model_selection import RepeatedKFold
warnings.simplefilter(action='ignore', category=FutureWarning)


#start runtime
global start
start = datetime.now()

#where to save results
filepath_to_csv = ('C:/Users/Admin/OneDrive/STOCK_PREDICTIONS/')

#models will save to file location!!!

#amount of days to pull and interval
global period
period = "366d"
global interval
interval = "1d"

#minimum days stock has been listed on exchange
datafilter_days = 500

#ticker list to select one, or multiple (tickerlist = ['tsla'] or tickerlist = ['tsla', 'nke'])
#tickerlist = ['tsla', 'nke']

    
def grabstocks():
    ### DATA
    stocks = request.get('https://finance.yahoo.com/losers/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAABwIVuDpYjgE3pZ0XefRHyyhGMQtW_qeJA8STShxLRzxtXwR5Jh9nsKp2ZwDSrveP7Mi61NlBuGGsNCYLWba8d0QIbMve7OMBtzjzI4bbCA84Q6Cm43jCyN57ZzcM2VT9fBioDxjQ5-bmecxgt3_N-S1JRDU5vjxLpz7wgBuAYB1')
    soup = BeautifulSoup(stocks.text, 'lxml')
    table1 = soup.find('div', id='fin-scr-res-table')
    headers = []
    for i in table1.find_all('td'):
        title=i.text
        headers.append(title)

    headers
    len(headers)
    cont = 0
    dato = pd.DataFrame(headers)
    dato2 = dato[::10]
    bist = list(dato2[0])
    global bist2
    bist2 = []
    global bist3
    bist3 =[]
    print("--------------------------------------------")
            
    while cont < len(bist):    
        bist2 = (bist[cont].upper())
        datalist(bist2)
        cont = cont + 1  
        if cont == len(bist):
            print("-------------------------------------------- \nTOTAL STOCKS PASSED:", len(bist3))
            break
                
    global tickerlist
    tickerlist = bist3
    
def datalist(x):
    day = yf.Ticker(x).history(period="1d")
    yearcount = yf.Ticker(x).history(period="max", interval=interval)
    if len(yearcount) > datafilter_days and int(day["Close"]) >= 2:
        bist3.append(x)
        print(x + ': Appended')
    else:
        print(x + ': Too low')
        
def data_collecter():
    print("--------------------------------------------")
    counter = 0
    for i in tickerlist:
        global startfunction
        startfunction = datetime.now()
        ticker = tickerlist[counter].upper()
        stock = yf.Ticker(ticker)
        stock_hist = stock.history(period=period, interval=interval)
        
        #moving data to find out difference in prices between two days
        stock_prev = stock_hist.copy()
        stock_prev = stock_prev.shift(1)

        ###finding actual close
        data = stock_hist[["Close"]]
        data = data.rename(columns = {'Close':'Actual_Close'})

        ##setup out target
        data["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

        ##join data
        predict = ["Close", "Volume", "Open", "High", "Low"]
        data = data.join(stock_prev[predict]).iloc[1:]
        
        #rolling means/more specific data
        weekly_mean = data.rolling(7).mean()["Close"]
        quarterly_mean = data.rolling(90).mean()["Close"]
        annual_mean = data.rolling(365).mean()["Close"]
        weekly_trend = data.shift(1).rolling(7).sum()["Target"]
        spy = yf.Ticker('SPY')
        daysss = len(stock_hist)
        dayyys = str(daysss) + "d"

        #JOINING IN THE S&P
        sp_period = len(data) + 1
        sp = spy.history(period=str(sp_period) + "d", interval=interval)
        data["weekly_mean"] = weekly_mean / data["Close"]
        data["quarterly_mean"] = quarterly_mean / data["Close"]
        data["annual_mean"] = annual_mean / data["Close"]
        data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
        data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
        data["weekly_trend"] = weekly_trend
        data["open_close_ratio"] = data["Open"] / data["Close"]
        data["high_close_ratio"] = data["High"] / data["Close"]
        data["low_close_ratio"] = data["Low"] / data["Close"]
        sp = sp.rename(columns = {'Close':'SP CLOSE'})
        sp = sp["SP CLOSE"]
        data = data.join(sp).iloc[1:]
        sp_weekly_mean = data.rolling(7).mean()["SP CLOSE"]
        data["sp_weekly_mean"] = sp_weekly_mean
        data = data.fillna(0)

        
        ####### LINEAR REG  
        data2 = data.drop(['Actual_Close', 'Target', 'quarterly_mean', 'annual_mean', 'annual_weekly_mean', 'annual_quarterly_mean', 'weekly_trend'], axis=1)
        data2 = data2.drop(['open_close_ratio', 'high_close_ratio', 'low_close_ratio', 'sp_weekly_mean', 'High', 'Low', 'weekly_mean'], axis=1)
        global axisvalues_collect
        axisvalues_collect=list(range(1,len(data2.columns)+1))
        

        
        data2=data2.apply(calc_slope,axis=1)

        data2 = data2.drop(['intercept', 'rvalue', 'pvalue', 'intercept_stderr'], axis=1)
        data = data.join(data2)

        
        optimizer(data, ticker)
        print("\nRuntime for:", ticker, datetime.now()-startfunction, "\n--------------------------------------------")
        counter = counter + 1
    global optiruntime
    optiruntime = (datetime.now() - startfunction)
    
def calc_slope(row):
        a = scipy.stats.linregress(row, y=axisvalues_collect)
        return pd.Series(a._asdict())
    
def optimizer(data_alter, ticker):
    ### OPTIMIZING PARAMETERS FOR MODEL
    model = ElasticNet(fit_intercept=True)
    

    data_alter = data_alter.drop(['Target', 'sp_weekly_mean', 'annual_weekly_mean', 'annual_quarterly_mean', 'weekly_trend', 'slope', 'annual_mean', 'quarterly_mean'], axis=1)
    data_alter = data_alter.drop(['high_close_ratio', 'low_close_ratio', 'stderr'], axis=1)
    y = data_alter['Actual_Close']
    X = data_alter.drop(['Actual_Close'], axis=1)
    
    #make training set - 25% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


    parameters = {
                      'max_iter'    : sp_randInt(5000, 20000),
                      'tol'         : sp_randFloat()

                     }
    print("-------------------------------------------- \nOptimization started for:", ticker)
    
    cross_validation = RepeatedKFold(n_splits=2, n_repeats=4, random_state =1)
    randm_src = RandomizedSearchCV(estimator=model, param_distributions = parameters,
                                   cv = cross_validation, n_iter = 1000, verbose = 1, n_jobs=-1, random_state=1)
    
    randm_src.fit(X_train, y_train)
    print("Optimization complete for:", ticker)
    print("Model fit for: ", ticker + "\n----------------")
    
    
    y_pred = randm_src.predict(X_test)

    print("Model statistics for:" + ticker + "\n----------------")
    print('Model Score:', randm_src.score(X_test, y_test))
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

    #save the model
    with open('model_' + ticker + '.pkl', 'wb') as f:
        pickle.dump(randm_src,f)
    print("Model", ticker, "stored")

def guesser():
    counter = 0
    global startguesser
    startguesser = datetime.now()
    data1 = pd.DataFrame(columns = ['Ticker', 'Yesterdays Close', 'Todays Predicted Close', 'Todays Actual Close', 'Difference', 'Tomorrows Predicted Close'])
    for i in tickerlist:
        ticker = tickerlist[counter].upper()
        stock = yf.Ticker(ticker)
        stock_hist = stock.history(period=period, interval=interval)
        
        #moving data to find out difference in prices between two days
        stock_prev = stock_hist.copy()
        stock_prev = stock_prev.shift(1)

        ###finding actual close
        data = stock_hist[["Close"]]
        data = data.rename(columns = {'Close':'Actual_Close'})

        ##setup out target
        data["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

        ###join the data
        predict = ["Close", "Volume", "Open", "High", "Low"]
        data = data.join(stock_prev[predict]).iloc[1:]
        
        #rolling means/more specific data
        weekly_mean = data.rolling(7).mean()["Close"]
        quarterly_mean = data.rolling(90).mean()["Close"]
        annual_mean = data.rolling(365).mean()["Close"]
        weekly_trend = data.shift(1).rolling(7).sum()["Target"]
        spy = yf.Ticker('SPY')
        daysss = len(stock_hist)
        dayyys = str(daysss) + "d"

        #JOINING IN THE S&P
        sp_period = len(data) + 1
        sp = spy.history(period=str(sp_period) + "d", interval=interval)
        data["weekly_mean"] = weekly_mean / data["Close"]
        data["quarterly_mean"] = quarterly_mean / data["Close"]
        data["annual_mean"] = annual_mean / data["Close"]
        data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
        data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
        data["weekly_trend"] = weekly_trend
        data["open_close_ratio"] = data["Open"] / data["Close"]
        data["high_close_ratio"] = data["High"] / data["Close"]
        data["low_close_ratio"] = data["Low"] / data["Close"]
        sp = sp.rename(columns = {'Close':'SP CLOSE'})
        sp = sp["SP CLOSE"]
        data = data.join(sp).iloc[1:]
        sp_weekly_mean = data.rolling(7).mean()["SP CLOSE"]
        data["sp_weekly_mean"] = sp_weekly_mean
        data = data.fillna(0)
        
        ####### LINEAR REG  
        data2 = data.drop(['Actual_Close', 'Target', 'quarterly_mean', 'annual_mean', 'annual_weekly_mean', 'annual_quarterly_mean', 'weekly_trend'], axis=1)
        data2 = data2.drop(['open_close_ratio', 'high_close_ratio', 'low_close_ratio', 'sp_weekly_mean', 'High', 'Low', 'weekly_mean'], axis=1)

        global axisvalues_predict
        axisvalues_predict=list(range(1,len(data2.columns)+1))
        
        def calc_slope(row):
            a = scipy.stats.linregress(row, y=axisvalues_predict)
            return pd.Series(a._asdict())
        
        data2=data2.apply(calc_slope,axis=1)

        data2 = data2.drop(['intercept', 'rvalue', 'pvalue', 'intercept_stderr'], axis=1)
        
        data = data.join(data2)

    #open model
        data = data.drop(['annual_quarterly_mean', 'annual_weekly_mean', 'slope', 'weekly_trend', 'annual_mean', 'quarterly_mean', 'high_close_ratio'], axis=1)
        data = data.drop(['low_close_ratio', 'stderr'], axis=1)
        with open('model_' + ticker + '.pkl', 'rb') as f:
            model = pickle.load(f)
        data2 = data
        data = data.drop(['Target', 'sp_weekly_mean', 'Actual_Close'], axis=1)
        y_pred = model.predict(data.tail(2))
        y_pred_fixed = np.delete(y_pred, 1)
        y_pred_tmrw = model.predict(data.tail(1))
        

        
        close_prev = data.copy()
        yest_close_fixed = close_prev.tail(1)["Close"]
        
        data1 = data1.append({'Ticker' : ticker, 'Yesterdays Close' : float(yest_close_fixed), 'Todays Predicted Close' : float(y_pred_fixed), 'Todays Actual Close' : float(data2.tail(1)["Actual_Close"]), 'Difference' : float(data2.tail(1)["Actual_Close"]) - float(y_pred_fixed), 'Tomorrows Predicted Close' : float(y_pred_tmrw)}, ignore_index = True)
        print(ticker, "close predicted \n--------------------------------------------")
        counter = counter + 1
        
        

        print(data1)
     
    now_ = time.strftime("%d_%m_%Y_%H_%M_%S")
    data1.to_csv(filepath_to_csv + str(now_) + '_25_loser' + '.csv', index=False)
    global guessertime
    guessertime = (datetime.now()-startguesser) 
    
def results():   
    print("-------------------------------------------- \nResults Printed and Saved to 'filepath'")
    print("Optimization Runtime: ", optiruntime)
    print("Guesser Runtime: ", guessertime)  
    print("Full Runtime: ", datetime.now()-start, "\n --------------------------------------------")

grabstocks()
tickerlist = tickerlist
data_collecter()
guesser()
results()






