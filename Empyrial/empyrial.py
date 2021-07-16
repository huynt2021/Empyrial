import numpy as np
import pandas as pd
import math
import datetime as dt
from empyrical import*
import quantstats as qs
from IPython.display import display
import matplotlib.pyplot as plt
import copy
import yfinance as yf
from fpdf import FPDF
import warnings
from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt, objective_functions, black_litterman, BlackLittermanModel

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------

today = dt.date.today()

# ------------------------------------------------------------------------------------------
class Engine:


  def __init__(self,start_date, portfolio, weights=None, rebalance=None, benchmark=['SPY'], end_date=today, optimizer=None, max_vol=0.15, diversification=1,confidences=None, view=None, min_weights=None, max_weights=None, risk_manager=None):
    self.start_date = start_date
    self.end_date = end_date
    self.portfolio = portfolio
    self.weights = weights
    self.benchmark = benchmark
    self.optimizer = optimizer
    self.rebalance = rebalance
    self.max_vol = max_vol
    self.diversification = diversification
    self.max_weights = max_weights
    self.min_weights = min_weights
    self.view = view
    self.confidences = confidences
    self.risk_manager = risk_manager
  

    if self.weights==None:
      self.weights = [1.0/len(self.portfolio)]*len(self.portfolio)
  
    if self.optimizer=="EF":
      self.weights = efficient_frontier(self, perf="False")

    if self.optimizer=="MEANVAR":
      self.weights = mean_var(self, vol_max=max_vol, perf="False")

    if self.optimizer=="HRP":
      self.weights = hrp(self, perf="False")

    if self.optimizer=="MINVAR":
      self.weights = min_var(self, perf="False")

    if self.optimizer== "BL":
      self.weights = bl(self, perf="False")

    if self.optimizer != None and self.optimizer != "EF" and self.optimizer !="MEANVAR" and self.optimizer !="HRP" and self.optimizer !="MINVAR" and self.optimizer !="BL":
      opt = self.optimizer
      self.weights = opt()
      
    if self.rebalance != None:
      self.rebalance = make_rebalance(self.start_date, self.end_date, self.optimizer, self.portfolio, self.rebalance, self.weights, self.max_vol, self.diversification, self.min_weights, self.max_weights)
#-------------------------------------------------------------------------------------------
def get_returns(stocks,wts, start_date, end_date=today):
    
  if len(stocks) > 1:
    assets = yf.download(stocks, start=start_date, end=end_date, progress=False)['Adj Close']
    assets = assets.filter(stocks)
    ret_data = assets.pct_change()[1:]
    returns = (ret_data * wts).sum(axis = 1)
    return returns
  else:
    df = yf.download(stocks, start=start_date, end=end_date, progress=False)['Adj Close']
    df = pd.DataFrame(df)
    returns = df.pct_change()
    return returns
# ------------------------------------------------------------------------------------------
def information_ratio(returns, benchmark_returns, days=252):
 return_difference = returns - benchmark_returns
 volatility = return_difference.std() * np.sqrt(days)
 information_ratio = return_difference.mean() / volatility
 return information_ratio

def graph_allocation(my_portfolio):
  fig1, ax1 = plt.subplots()
  ax1.pie(my_portfolio.weights, labels=my_portfolio.portfolio, autopct='%1.1f%%',
          shadow=False)
  ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.title("Portfolio's allocation")
  plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------
#initialize a variable that we be set to false
#initialize a variable that we be set to false
def empyrial(my_portfolio, rf=0.0, sigma_value=1, confidence_value=0.95, rebalance=False):
  
  #standard emyrial output
  if rebalance == False:
      
      #standard returns calculation
      returns = get_returns(my_portfolio.portfolio, my_portfolio.weights, start_date=my_portfolio.start_date,end_date=my_portfolio.end_date)
      
  #when we want to do the rebalancing
  if rebalance == True:

      #we want to get the dataframe with the dates and weights
      rebalance_schedule = my_portfolio.rebalance

      columns = []
      for date in rebalance_schedule.columns:
            date = date[0:10]
            columns.append(date)
      rebalance_schedule.columns = columns
      
      #then want to make a list of the dates and start with our first date
      dates = [my_portfolio.start_date]
      
      #then our rebalancing dates into that list 
      dates = dates + rebalance_schedule.columns.to_list()
      
      datess = []
      for date in dates:
          date = date[0:10]
          datess.append(date)
      dates = datess
      #this will hold returns
      returns = pd.Series()
      
  
      #then we want to be able to call the dates like tuples
      for i in range(len(dates) - 1):
          

          #get our weights
          weights = rebalance_schedule[str(dates[i+1])]
          
          #then we want to get the returns
          add_returns = get_returns(my_portfolio.portfolio, weights, start_date = dates[i], end_date = dates[i+1])
          
          #then append those returns
          returns = returns.append(add_returns)

  creturns = (returns + 1).cumprod()

  #risk manager
  try:
    if list(my_portfolio.risk_manager.keys())[0] == "Stop Loss":

      values = []
      for r in creturns:
        if r <= 1+my_portfolio.risk_manager['Stop Loss']:
          values.append(r)
        else:
          pass

      try:
        date = creturns[creturns == values[0]].index[0]
        date = str(date.to_pydatetime())
        my_portfolio.end_date = date[0:10]
        returns = returns[:my_portfolio.end_date]

      except Exception as e:
        pass

    if list(my_portfolio.risk_manager.keys())[0] == "Take Profit":

      values = []
      for r in creturns:
        if r >= 1+my_portfolio.risk_manager['Take Profit']:
          values.append(r)
        else:
          pass
      
      try:
        date = creturns[creturns == values[0]].index[0]
        date = str(date.to_pydatetime())
        my_portfolio.end_date = date[0:10]
        returns = returns[:my_portfolio.end_date]

      except Exception as e:
        pass

    if list(my_portfolio.risk_manager.keys())[0] == "Max Drawdown":

      drawdown = qs.stats.to_drawdown_series(returns)

      values = []
      for r in drawdown:
        if r <= my_portfolio.risk_manager['Max Drawdown']:
          values.append(r)
        else:
          pass

      try:
        date = drawdown[drawdown == values[0]].index[0]
        date = str(date.to_pydatetime())
        my_portfolio.end_date = date[0:10]
        returns = returns[:my_portfolio.end_date]

      except Exception as e:
        pass

  except Exception as e:
      pass
      
  
  print("Start date: " + str(my_portfolio.start_date))
  print("End date: " + str(my_portfolio.end_date))

  benchmark = get_returns(my_portfolio.benchmark, wts=[1], start_date=my_portfolio.start_date,end_date=my_portfolio.end_date)

  CAGR = cagr(returns, period=DAILY, annualization=None)
  CAGR = round(CAGR,2)
  CAGR = CAGR.tolist()
  CAGR = str(round(CAGR*100,2)) + '%'



  CUM = cum_returns(returns, starting_value=0, out=None)*100
  CUM = CUM.iloc[-1]
  CUM = CUM.tolist()
  CUM = str(round(CUM,2)) + '%'


  VOL = qs.stats.volatility(returns, annualize=True, trading_year_days=252)
  VOL = VOL.tolist()
  VOL = str(round(VOL*100,2))+' %'


  SR = qs.stats.sharpe(returns, rf=rf)
  SR = np.round(SR, decimals=2)
  SR = str(SR)

  empyrial.SR = SR

  CR =  qs.stats.calmar(returns)
  CR = CR.tolist()
  CR = str(round(CR,2))

  empyrial.CR = CR

  STABILITY = stability_of_timeseries(returns)
  STABILITY = round(STABILITY,2)
  STABILITY = str(STABILITY)


  MD = max_drawdown(returns, out=None)
  MD = str(round(MD*100,2))+' %'

  
  '''OR = omega_ratio(returns, risk_free=0.0, required_return=0.0)
  OR = round(OR,2)
  OR = str(OR)
  print(OR)'''

  SOR = sortino_ratio(returns, required_return=0, period=DAILY)
  SOR = round(SOR,2)
  SOR = str(SOR)


  SK = qs.stats.skew(returns)
  SK = round(SK,2)
  SK = SK.tolist()
  SK = str(SK)


  KU = qs.stats.kurtosis(returns)
  KU = round(KU,2)
  KU = KU.tolist()
  KU = str(KU)

  TA = tail_ratio(returns)
  TA = round(TA,2)
  TA = str(TA)


  CSR = qs.stats.common_sense_ratio(returns)
  CSR = round(CSR,2)
  CSR = CSR.tolist()
  CSR = str(CSR)


  VAR = qs.stats.value_at_risk(returns, sigma=sigma_value, confidence=confidence_value)
  VAR = np.round(VAR, decimals=2)
  VAR = str(VAR*100)+' %'

  AL = alpha_beta(returns, benchmark, risk_free=rf)
  AL = AL[0]
  AL = round(AL,2)

  BTA = alpha_beta(returns, benchmark, risk_free=rf)
  BTA = BTA[1]
  BTA = round(BTA,2)

  def condition(x):
    return x > 0

  win = sum(condition(x) for x in returns)
  total = len(returns)
  win_ratio = win/total
  win_ratio = win_ratio*100
  win_ratio = round(win_ratio,2)

  IR = information_ratio(returns, benchmark.iloc[:,0])
  IR = round(IR,2)



  data = {'':['Annual return', 'Cumulative return', 'Annual volatility','Winning day ratio', 'Sharpe ratio','Calmar ratio', 'Information ratio', 'Stability', 'Max Drawdown','Sortino ratio','Skew', 'Kurtosis', 'Tail Ratio', 'Common sense ratio', 'Daily value at risk',
              'Alpha', 'Beta'

  ],
        'Backtest':[CAGR, CUM, VOL,f'{win_ratio}%', SR, CR, IR, STABILITY, MD, SOR, SK, KU, TA, CSR, VAR, AL, BTA]}

  # Create DataFrame
  df = pd.DataFrame(data)
  df.set_index('', inplace=True)
  df.style.set_properties(**{'background-color': 'white',
                           'color': 'black',
                          'border-color':'black'})
  display(df)

  empyrial.df = data

  if rebalance == True:
      df

  y = []
  for x in returns:
    y.append(x)

  arr = np.array(y)
  arr
  returns.index
  my_color = np.where(arr>=0, 'blue', 'grey')
  plt.figure(figsize=(30,8))
  plt.vlines(x=returns.index, ymin=0, ymax=arr, color=my_color, alpha=0.4)
  plt.title('Returns')

  empyrial.returns = returns
  empyrial.creturns = creturns
  empyrial.benchmark = benchmark
  empyrial.CAGR = CAGR
  empyrial.CUM = CUM
  empyrial.VOL = VOL
  empyrial.SR = SR
  empyrial.win_ratio = win_ratio
  empyrial.CR = CR
  empyrial.IR = IR
  empyrial.STABILITY = STABILITY
  empyrial.MD = MD
  empyrial.SOR = SOR
  empyrial.SK = SK
  empyrial.KU = KU
  empyrial.TA = TA
  empyrial.CSR = CSR
  empyrial.VAR = VAR
  empyrial.AL = AL
  empyrial.BTA = BTA

  try:
    empyrial.orderbook = make_rebalance.output
  except Exception as e:
    OrderBook = pd.DataFrame(
        {'Assets': my_portfolio.portfolio,
        'Allocation': my_portfolio.weights,
        })

    empyrial.orderbook = OrderBook.T

  return qs.plots.returns(returns,benchmark, cumulative=True), qs.plots.monthly_heatmap(returns), qs.plots.drawdown(returns), qs.plots.drawdowns_periods(returns), qs.plots.rolling_volatility(returns), qs.plots.rolling_sharpe(returns), qs.plots.rolling_beta(returns, benchmark), graph_opt(my_portfolio.portfolio, my_portfolio.weights, pie_size=7, font_size=14)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def flatten(seq):
    l = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):
                l.append(elt2)
        else:
            l.append(elt)
    return l
#-------------------------------------------------------------------------------
def graph_opt(my_portfolio, my_weights, pie_size, font_size):
  fig1, ax1 = plt.subplots()
  fig1.set_size_inches(pie_size, pie_size)
  cs= ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#f6c9ff', '#a6fff6', '#fffeb8', '#ffe1d4', '#cccdff', '#fad6ff']
  ax1.pie(my_weights, labels=my_portfolio, autopct='%1.1f%%',
          shadow=False, colors=cs)
  ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.rcParams['font.size'] = font_size
  plt.show()
#--------------------------------------------------------------------------
def equal_weighting(my_portfolio):
  return [1.0/len(my_portfolio.portfolio)]*len(my_portfolio.portfolio)
#--------------------------------------------------------------------------------
def efficient_frontier(my_portfolio, perf=True):

  #changed to take in desired timeline, the problem is that it would use all historical data
  ohlc = yf.download(my_portfolio.portfolio, start = my_portfolio.start_date, end = my_portfolio.end_date, progress=False)
  prices = ohlc["Adj Close"].dropna(how="all")
  df = prices.filter(my_portfolio.portfolio)
  
  #sometimes we will pick a date range where company isn't public we can't set price to 0 so it has to go to 1
  df = df.fillna(1)

  mu = expected_returns.mean_historical_return(df)
  S = risk_models.sample_cov(df)

  #optimize for max sharpe ratio
  ef = EfficientFrontier(mu, S)

  if my_portfolio.min_weights != None:
    ef.add_constraint(lambda x : x >= my_portfolio.min_weights)
  if my_portfolio.max_weights != None:
    ef.add_constraint(lambda x : x <= my_portfolio.max_weights)
  weights = ef.max_sharpe()
  cleaned_weights = ef.clean_weights()
  wts = cleaned_weights.items()

  result = []
  for val in wts:
    a, b = map(list, zip(*[val]))
    result.append(b)
  

  if perf==True:
    pred = ef.portfolio_performance(verbose=True);
  
  return flatten(result)
#-------------------------------------------------------------------------------
def bl(my_portfolio, perf=True):


  ohlc = yf.download(my_portfolio.portfolio, start = my_portfolio.start_date, end = my_portfolio.end_date, progress=False)
  market_prices = yf.download("SPY",start = my_portfolio.start_date, end = my_portfolio.end_date, progress=False)["Adj Close"]
  prices = ohlc["Adj Close"]

  df = prices.filter(my_portfolio.portfolio)

  mcaps = {}
  for t in my_portfolio.portfolio:
      stock = yf.Ticker(t)
      mcaps[t] = stock.info["marketCap"]
  S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

  delta = black_litterman.market_implied_risk_aversion(market_prices)

  market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

  bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta,
                        absolute_views=my_portfolio.view, omega="idzorek", view_confidences= my_portfolio.confidences)
  
  ret_bl = bl.bl_returns()
  S_bl = bl.bl_cov()
  
  ef = EfficientFrontier(ret_bl, S_bl)
  if my_portfolio.min_weights != None:
    ef.add_constraint(lambda x : x >= my_portfolio.min_weights)
  if my_portfolio.max_weights != None:
    ef.add_constraint(lambda x : x <= my_portfolio.max_weights)
  ef.max_sharpe()
  weights = ef.clean_weights()
  lists = dict(weights)
  wts = weights.items()


  wei = []
  for a in my_portfolio.portfolio:
    value = lists[a]
    wei.append(value)

  if perf==True:
    pred = ef.portfolio_performance(verbose=True);
  
  return wei
#-------------------------------------------------------------------------------
def hrp(my_portfolio, perf=True):

  #changed to take in desired timeline, the problem is that it would use all historical data

  ohlc = yf.download(my_portfolio.portfolio, start = my_portfolio.start_date, end = my_portfolio.end_date, progress=False)
  prices = ohlc["Adj Close"].dropna(how="all")
  prices = prices.filter(my_portfolio.portfolio)
  
  #sometimes we will pick a date range where company isn't public we can't set price to 0 so it has to go to 1
  prices = prices.fillna(1)

  rets = expected_returns.returns_from_prices(prices)
  hrp = HRPOpt(rets)
  hrp.optimize()
  weights = hrp.clean_weights()

  wts = weights.items()

  result = []
  for val in wts:
    a, b = map(list, zip(*[val]))
    result.append(b)
  
  if perf==True:
    hrp.portfolio_performance(verbose=True);

  return flatten(result)
#-----------------------------------------------------------------------------
def mean_var(my_portfolio, vol_max=0.15, perf=True):
  
  #changed to take in desired timeline, the problem is that it would use all historical data

  ohlc = yf.download(my_portfolio.portfolio, start = my_portfolio.start_date, end = my_portfolio.end_date, progress=False)
  prices = ohlc["Adj Close"].dropna(how="all")
  prices = prices.filter(my_portfolio.portfolio)
  
  #sometimes we will pick a date range where company isn't public we can't set price to 0 so it has to go to 1
  prices = prices.fillna(1)

  mu = expected_returns.capm_return(prices)
  S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

  ef = EfficientFrontier(mu, S)
  ef.add_objective(objective_functions.L2_reg, gamma=my_portfolio.diversification)
  if my_portfolio.min_weights != None:
    ef.add_constraint(lambda x : x >= my_portfolio.min_weights)
  if my_portfolio.max_weights != None:
    ef.add_constraint(lambda x : x <= my_portfolio.max_weights)
  ef.efficient_risk(vol_max)
  weights = ef.clean_weights()

  wts = weights.items()

  result = []
  for val in wts:
    a, b = map(list, zip(*[val]))
    result.append(b)

  if perf==True:
    ef.portfolio_performance(verbose=True);

  return flatten(result)
#--------------------------------------------------------------------------------
def min_var(my_portfolio, perf=True):

  ohlc = yf.download(my_portfolio.portfolio, start = my_portfolio.start_date, end = my_portfolio.end_date, progress=False)
  prices = ohlc["Adj Close"].dropna(how="all")
  prices = prices.filter(my_portfolio.portfolio)

  mu = expected_returns.capm_return(prices)
  S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

  ef = EfficientFrontier(mu, S)
  ef.add_objective(objective_functions.L2_reg, gamma=my_portfolio.diversification)
  if my_portfolio.min_weights != None:
    ef.add_constraint(lambda x : x >= my_portfolio.min_weights)
  if my_portfolio.max_weights != None:
    ef.add_constraint(lambda x : x <= my_portfolio.max_weights)
  ef.min_volatility()
  weights = ef.clean_weights()

  wts = weights.items()

  result = []
  for val in wts:
    a, b = map(list, zip(*[val]))
    result.append(b)

  if perf==True:
    ef.portfolio_performance(verbose=True);

  return flatten(result)
#------------------------------------------------------------------------------
def optimizer(my_portfolio, vol_max=25, pie_size=5, font_size=14):
  
  returns1 = get_returns(my_portfolio.portfolio, equal_weighting(my_portfolio), start_date=my_portfolio.start_date,end_date=my_portfolio.end_date)
  creturns1 = (returns1 + 1).cumprod()

  port = copy.deepcopy(my_portfolio.portfolio)

  if my_portfolio.optimizer == "EF":
    wts = efficient_frontier(my_portfolio)
  
  if my_portfolio.optimizer == "HRP":
    wts = hrp(my_portfolio)

  if my_portfolio.optimizer == "MEANVAR":
    wts = mean_var(my_portfolio, my_portfolio.max_vol)

  if my_portfolio.optimizer == "MINVAR":
    wts = min_var(my_portfolio)

  if my_portfolio.optimizer == "EW":
    wts = equal_weighting(my_portfolio)

  if my_portfolio.optimizer == "BL":
    wts = bl(my_portfolio)

  if my_portfolio.optimizer != None and my_portfolio.optimizer != "EF" and my_portfolio.optimizer !="MEANVAR" and my_portfolio.optimizer !="HRP" and my_portfolio.optimizer !="MINVAR" and my_portfolio.optimizer !="BL":
      opt = my_portfolio.optimizer
      my_portfolio.weights = opt()

  print(wts)
  print("\n")

  indices = [i for i, x in enumerate(wts) if x == 0.0]

  while 0.0 in wts: wts.remove(0.0)

  for i in sorted(indices, reverse=True):
    del port[i]

    
  graph_opt(port, wts, pie_size, font_size)

  print("\n")

  returns2 = get_returns(port, wts, start_date=my_portfolio.start_date,end_date=my_portfolio.end_date)
  creturns2 = (returns2 + 1).cumprod()

  plt.rcParams['font.size'] = 13
  plt.figure(figsize=(30,10))
  plt.xlabel("Portfolio vs Benchmark")

  ax1 = creturns1.plot(color='blue', label='Without optimization')
  ax2 = creturns2.plot(color='red', label='With optimization')

  h1, l1 = ax1.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()


  plt.legend(l1+l2, loc=2)
  plt.show()
#-------------------------------------------------------------------------------------------------
def check_schedule(rebalance):
    
    #this is the list of acceptable rebalancing schedules
    acceptable_schedules = ["6mo","1y","2y", "quarterly", "monthly"]
    
    #we want to loop through the acceptable schedules to see if the user inputted the right rebalance
    for i in acceptable_schedules:
        
        #want to make this a try with an error if it is incorrect
        
        #check if the inputted rebalance is an acceptable
        if i == rebalance:
            
            #if the inputted rebalancing is in the acceptable schedule set to true
            valid_schedule = True
            return valid_schedule
            break
        
        #if the inputted value isn't in our accepted rebalancing schedule
        else:
            valid_schedule = False
         
    #return that back
    return valid_schedule
#--------------------------------------------------------------------------------
def valid_range(start_date, end_date, rebalance):
    
    #make the start date to a datetime
    start_date = dt.datetime.strptime(start_date, "%Y-%M-%d")
    
    #have to make end date a datetime because strptime is not supported for date
    end_date = dt.datetime(end_date.year, end_date.month, end_date.day)
    
    #gets the number of days
    days = (end_date - start_date).days
    
    #all of this is for checking the length
    
    if rebalance == "6mo" and days <= (int(365/2)):
        raise KeyError("Date Range does not encompass rebalancing interval")
        
    if rebalance == "1y" and days <= int(365):
        raise KeyError("Date Range does not encompass rebalancing interval")
        
    if rebalance == "2y" and days <= int(365 * 2):
        raise KeyError("Date Range does not encompass rebalancing interval")
        
    if rebalance == "quarterly" and days <= int(365 / 4):
        raise KeyError("Date Range does not encompass rebalancing interval")
        
    if rebalance == "monthyl" and days <= int(30):
        raise KeyError("Date Range does not encompass rebalancing interval")
        
    #we will needs these dates later on so we'll return them back
    return start_date, end_date

#--------------------------------------------------------------------------------
def get_date_range(start_date, end_date, rebalance):
    
    #this will keep track of the rebalancing dates and we want to start on the first date
    rebalance_dates = [start_date]
    input_date = start_date
    
    if rebalance == "6mo":
        
        #run for an arbitrarily large number we'll resolve this by breaking when we break the equality
        for i in range(1000):
        
            #this gives us the future date
            input_date = input_date + dt.timedelta(days = 365/2)
            
            #then we want to compare dates
            if input_date < end_date:
                
                #if the dates are less than the final date we put that into our date list 
                rebalance_dates.append(input_date)
              
            #when the rebalance date is greater than our final date
            else:
                break

    if rebalance == "1y":
        
        #run for an arbitrarily large number we'll resolve this by breaking when we break the equality
        for i in range(1000):
        
            #this gives us the future date
            input_date = input_date + dt.timedelta(days = 365)
            
            #then we want to compare dates
            if input_date < end_date:
                
                #if the dates are less than the final date we put that into our date list 
                rebalance_dates.append(input_date)
              
            #when the rebalance date is greater than our final date
            else:
                break

    if rebalance == "2y":
        
        #run for an arbitrarily large number we'll resolve this by breaking when we break the equality
        for i in range(1000):
        
            #this gives us the future date
            input_date = input_date + dt.timedelta(days = 365 * 2)
            
            #then we want to compare dates
            if input_date < end_date:
                
                #if the dates are less than the final date we put that into our date list 
                rebalance_dates.append(input_date)
              
            #when the rebalance date is greater than our final date
            else:
                break
            
    if rebalance == "quarterly":
        
        #run for an arbitrarily large number we'll resolve this by breaking when we break the equality
        for i in range(1000):
        
            #this gives us the future date
            input_date = input_date + dt.timedelta(days = 365 / 4)
            
            #then we want to compare dates
            if input_date < end_date:
                
                #if the dates are less than the final date we put that into our date list 
                rebalance_dates.append(input_date)
              
            #when the rebalance date is greater than our final date
            else:
                break
            
    if rebalance == "monthly":
        
        #run for an arbitrarily large number we'll resolve this by breaking when we break the equality
        for i in range(1000):
        
            #this gives us the future date
            input_date = input_date + dt.timedelta(days = 30)
            
            #then we want to compare dates
            if input_date < end_date:
                
                #if the dates are less than the final date we put that into our date list 
                rebalance_dates.append(input_date)
              
            #when the rebalance date is greater than our final date
            else:
                break
    
    #then we want to return those dates
    return rebalance_dates
#--------------------------------------------------------------------------------
def make_rebalance(start_date, end_date, optimize, portfolio_input, rebalance, allocation,  vol_max, div, min,max):

    #makes sure that the value passed through for rebalancing is a valid one
    valid_schedule = check_schedule(rebalance)
        
    if valid_schedule == False:
        raise KeyError("Not an accepted rebalancing schedule")
        
    #this checks to make sure that the date range given works for the rebalancing
    start_date, end_date = valid_range(start_date, end_date, rebalance)
    
    #this function will get us the specific dates
    dates = get_date_range(start_date, end_date, rebalance)
    
    #we are going to make columns with the end date and the weights
    columns = ["end_date"] + portfolio_input
    
    #then make a dataframe with the index being the tickers
    output_df = pd.DataFrame(index = portfolio_input)
    
    for i in range(len(dates) - 1):
      
        try:
          portfolio = Engine(
              start_date = dates[0],
              end_date = dates[i+1],
              portfolio = portfolio_input,
              weights = allocation,
              optimizer = "{}".format(optimize),
              max_vol = vol_max,
              diversification = div,
              min_weights = min,
              max_weights = max)
          
        except TypeError:
          portfolio = Engine(
              start_date = dates[0],
              end_date = dates[i+1],
              portfolio = portfolio_input,
              weights = allocation,
              optimizer = optimize,
              max_vol = vol_max,
              diversification = div,
              min_weights = min,
              max_weights = max)


        output_df["{}".format(dates[i+1])] = portfolio.weights
        
    
    #we have to run it one more time to get what the optimization is for up to today's date
    try:
      portfolio = Engine(
        start_date = dates[0],
        portfolio = portfolio_input,
        weights = allocation,
        optimizer = "{}".format(optimize),
        max_vol = vol_max,
        diversification = div,
        min_weights = min,
        max_weights = max)
      
    except TypeError:
      portfolio = Engine(
        start_date = dates[0],
        portfolio = portfolio_input,
        weights = allocation,
        optimizer = optimize,
        max_vol = vol_max,
        diversification = div,
        min_weights = min,
        max_weights = max)

    
    output_df['{}'.format(today)] = portfolio.weights
    
    make_rebalance.output = output_df
    return output_df
#--------------------------------------------------------------------------------------------------------
def get_report(my_portfolio, rf=0.0, sigma_value=1, confidence_value=0.95, rebalance=False):

  #standard emyrial output
  if rebalance == False:
      
      #standard returns calculation
      returns = get_returns(my_portfolio.portfolio, my_portfolio.weights, start_date=my_portfolio.start_date,end_date=my_portfolio.end_date)
      
  #when we want to do the rebalancing
  if rebalance == True:
      
      print("")
      print('rebalance hit')
      
      #we want to get the dataframe with the dates and weights
      rebalance_schedule = my_portfolio.rebalance

      columns = []
      for date in rebalance_schedule.columns:
            date = date[0:10]
            columns.append(date)
      rebalance_schedule.columns = columns
      
      #then want to make a list of the dates and start with our first date
      dates = [my_portfolio.start_date]
      
      #then our rebalancing dates into that list 
      dates = dates + rebalance_schedule.columns.to_list()

      datess = []
      for date in dates:
          date = date[0:10]
          datess.append(date)
      dates = datess
      #this will hold returns
      returns = pd.Series()
      
  
      #then we want to be able to call the dates like tuples
      for i in range(len(dates) - 1):

          #get our weights
          weights = rebalance_schedule[str(dates[i+1])]
          
          #then we want to get the returns
          add_returns = get_returns(my_portfolio.portfolio, weights, start_date = dates[i], end_date = dates[i+1])
          
          #then append those returns
          returns = returns.append(add_returns)

  creturns = (returns + 1).cumprod()

  #risk manager
  try:
    if list(my_portfolio.risk_manager.keys())[0] == "Stop Loss":

      values = []
      for r in creturns:
        if r <= 1+my_portfolio.risk_manager['Stop Loss']:
          values.append(r)
        else:
          pass

      try:
        date = creturns[creturns == values[0]].index[0]
        date = str(date.to_pydatetime())
        my_portfolio.end_date = date[0:10]
        returns = returns[:my_portfolio.end_date]

      except Exception as e:
        pass

    if list(my_portfolio.risk_manager.keys())[0] == "Take Profit":

      values = []
      for r in creturns:
        if r >= 1+my_portfolio.risk_manager['Take Profit']:
          values.append(r)
        else:
          pass
      
      try:
        date = creturns[creturns == values[0]].index[0]
        date = str(date.to_pydatetime())
        my_portfolio.end_date = date[0:10]
        returns = returns[:my_portfolio.end_date]

      except Exception as e:
        pass

    if list(my_portfolio.risk_manager.keys())[0] == "Max Drawdown":

      drawdown = qs.stats.to_drawdown_series(returns)

      values = []
      for r in drawdown:
        if r <= my_portfolio.risk_manager['Max Drawdown']:
          values.append(r)
        else:
          pass

      try:
        date = drawdown[drawdown == values[0]].index[0]
        date = str(date.to_pydatetime())
        my_portfolio.end_date = date[0:10]
        returns = returns[:my_portfolio.end_date]

      except Exception as e:
        pass

  except Exception as e:
      pass

  fig1, ax1 = plt.subplots()
  fig1.set_size_inches(5, 5)
  cs= ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#f6c9ff', '#a6fff6', '#fffeb8', '#ffe1d4', '#cccdff', '#fad6ff']
  ax1.pie(my_portfolio.weights, labels=my_portfolio.portfolio, autopct='%1.1f%%',
          shadow=False, colors=cs)
  ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.rcParams['font.size'] = 12
  plt.close(fig1)
  fig1.savefig('allocation.png')

  pdf = FPDF()
  pdf.add_page()
  pdf.set_font('arial', 'B', 14)
  pdf.image('https://user-images.githubusercontent.com/61618641/120909011-98f8a180-c670-11eb-8844-2d423ba3fa9c.png', x = None, y = None, w = 45, h = 5, type = '', link = 'https://github.com/ssantoshp/Empyrial')
  pdf.cell(20, 15, f"Report", ln=1)
  pdf.set_font('arial', size=11)
  pdf.image('allocation.png', x = 135, y = 0, w = 70, h = 70, type = '', link = '')
  pdf.cell(20, 7, f"Start date: " + str(my_portfolio.start_date), ln=1)
  pdf.cell(20, 7, f"End date: " + str(my_portfolio.end_date), ln=1)


  benchmark = get_returns(my_portfolio.benchmark, wts=[1], start_date=my_portfolio.start_date,end_date=my_portfolio.end_date)

  CAGR = cagr(returns, period=DAILY, annualization=None)
  CAGR = round(CAGR,2)
  CAGR = CAGR.tolist()
  CAGR = str(round(CAGR*100,2)) + '%'



  CUM = cum_returns(returns, starting_value=0, out=None)*100
  CUM = CUM.iloc[-1]
  CUM = CUM.tolist()
  CUM = str(round(CUM,2)) + '%'


  VOL = qs.stats.volatility(returns, annualize=True, trading_year_days=252)
  VOL = VOL.tolist()
  VOL = str(round(VOL*100,2))+' %'


  SR = qs.stats.sharpe(returns, rf=rf)
  SR = np.round(SR, decimals=2)
  SR = str(SR)

  empyrial.SR = SR

  CR =  qs.stats.calmar(returns)
  CR = CR.tolist()
  CR = str(round(CR,2))

  empyrial.CR = CR

  STABILITY = stability_of_timeseries(returns)
  STABILITY = round(STABILITY,2)
  STABILITY = str(STABILITY)


  MD = max_drawdown(returns, out=None)
  MD = str(round(MD*100,2))+' %'

  
  '''OR = omega_ratio(returns, risk_free=0.0, required_return=0.0)
  OR = round(OR,2)
  OR = str(OR)
  print(OR)'''

  SOR = sortino_ratio(returns, required_return=0, period=DAILY)
  SOR = round(SOR,2)
  SOR = str(SOR)


  SK = qs.stats.skew(returns)
  SK = round(SK,2)
  SK = SK.tolist()
  SK = str(SK)


  KU = qs.stats.kurtosis(returns)
  KU = round(KU,2)
  KU = KU.tolist()
  KU = str(KU)

  TA = tail_ratio(returns)
  TA = round(TA,2)
  TA = str(TA)


  CSR = qs.stats.common_sense_ratio(returns)
  CSR = round(CSR,2)
  CSR = CSR.tolist()
  CSR = str(CSR)


  VAR = qs.stats.value_at_risk(returns, sigma=sigma_value, confidence=confidence_value)
  VAR = np.round(VAR, decimals=2)
  VAR = str(VAR*100)+' %'

  AL = alpha_beta(returns, benchmark, risk_free=rf)
  AL = AL[0]
  AL = round(AL,2)

  BTA = alpha_beta(returns, benchmark, risk_free=rf)
  BTA = BTA[1]
  BTA = round(BTA,2)

  def condition(x):
    return x > 0

  win = sum(condition(x) for x in returns)
  total = len(returns)
  win_ratio = win/total
  win_ratio = win_ratio*100
  win_ratio = round(win_ratio,2)

  IR = information_ratio(returns, benchmark.iloc[:,0])
  IR = round(IR,2)



  data = {'':['Annual return', 'Cumulative return', 'Annual volatility','Winning day ratio', 'Sharpe ratio','Calmar ratio', 'Information ratio', 'Stability', 'Max Drawdown','Sortino ratio','Skew', 'Kurtosis', 'Tail Ratio', 'Common sense ratio', 'Daily value at risk',
              'Alpha', 'Beta'

  ],
        'Backtest':[CAGR, CUM, VOL,f'{win_ratio}%', SR, CR, IR, STABILITY, MD, SOR, SK, KU, TA, CSR, VAR, AL, BTA]}

  # Create DataFrame
  df = pd.DataFrame(data)
  df.set_index('', inplace=True)
  df.style.set_properties(**{'background-color': 'white',
                           'color': 'black',
                          'border-color':'black'})

  empyrial.df = data

  if rebalance == True:
      df

  y = []
  for x in returns:
    y.append(x)

  arr = np.array(y)
  arr
  returns.index
  my_color = np.where(arr>=0, 'blue', 'grey')
  ret = plt.figure(figsize=(30,8))
  plt.vlines(x=returns.index, ymin=0, ymax=arr, color=my_color, alpha=0.4)
  plt.title('Returns')
  plt.close(ret)
  ret.savefig('ret.png')
  

  pdf.cell(20, 7, f"", ln=1)
  pdf.cell(20, 7, f"Annual return: " + str(CAGR), ln=1)
  pdf.cell(20, 7, f"Cumulative return: " + str(CUM), ln=1)
  pdf.cell(20, 7, f"Annual volatility: " + str(VOL), ln=1)
  pdf.cell(20, 7, f"Winning day ratio: " + str(win_ratio), ln=1)
  pdf.cell(20, 7, f"Sharpe ratio: " + str(SR), ln=1)
  pdf.cell(20, 7, f"Calmar ratio: " + str(CR), ln=1)
  pdf.cell(20, 7, f"Information ratio: " + str(IR), ln=1)
  pdf.cell(20, 7, f"Stability: " + str(STABILITY), ln=1)
  pdf.cell(20, 7, f"Max drawdown: " + str(MD), ln=1)
  pdf.cell(20, 7, f"Sortino ratio: " + str(SOR), ln=1)
  pdf.cell(20, 7, f"Skew: " + str(SK), ln=1)
  pdf.cell(20, 7, f"Kurtosis: " + str(KU), ln=1)
  pdf.cell(20, 7, f"Tail ratio: " + str(TA), ln=1)
  pdf.cell(20, 7, f"Common sense ratio: " + str(CSR), ln=1)
  pdf.cell(20, 7, f"Daily value at risk: " + str(VAR), ln=1)
  pdf.cell(20, 7, f"Alpha: " + str(AL), ln=1)
  pdf.cell(20, 7, f"Beta: " + str(BTA), ln=1)

  qs.plots.returns(returns,benchmark, cumulative=True, savefig= "retbench.png", show=False)
  qs.plots.monthly_heatmap(returns,savefig= "heatmap.png", show=False)
  qs.plots.drawdown(returns, savefig= "drawdown.png", show=False)
  qs.plots.drawdowns_periods(returns, savefig= "d_periods.png", show=False)
  qs.plots.rolling_volatility(returns,savefig= "rvol.png", show=False)
  qs.plots.rolling_sharpe(returns,savefig= "rsharpe.png", show=False)
  qs.plots.rolling_beta(returns, benchmark,savefig= "rbeta.png", show=False)

  pdf.image('ret.png', x = -20, y = None, w = 250, h = 80, type = '', link = '')
  pdf.cell(20, 7, f"", ln=1)
  pdf.image('retbench.png', x = -20, y = None, w = 200, h = 100, type = '', link = '')
  pdf.cell(20, 7, f"", ln=1)
  pdf.image('heatmap.png', x = None, y = None, w = 200, h = 80, type = '', link = '')
  pdf.cell(20, 7, f"", ln=1)
  pdf.image('drawdown.png', x = None, y = None, w = 200, h = 80, type = '', link = '')
  pdf.cell(20, 7, f"", ln=1)
  pdf.image('d_periods.png', x = None, y = None, w = 200, h = 80, type = '', link = '')
  pdf.cell(20, 7, f"", ln=1)
  pdf.image('rvol.png', x = None, y = None, w = 190, h = 80, type = '', link = '')
  pdf.cell(20, 7, f"", ln=1)
  pdf.image('rsharpe.png', x = None, y = None, w = 190, h = 80, type = '', link = '')
  pdf.cell(20, 7, f"", ln=1)
  pdf.image('rbeta.png', x = None, y = None, w = 190, h = 80, type = '', link = '')

  pdf.output('report.pdf', 'F')
