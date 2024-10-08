import numpy as np
import pandas as pd
import datetime as dt
import quantstats as qs
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib
import copy
import yfinance as yf
from fpdf import FPDF
import warnings
import logging
import random

# Optional: switch to 'Agg' backend for non-interactive plots
matplotlib.use('Agg')

from empyrical import (
    cagr,
    cum_returns,
    stability_of_timeseries,
    max_drawdown,
    sortino_ratio,
    alpha_beta,
    tail_ratio,
)
from pypfopt import (
    EfficientFrontier,
    risk_models,
    expected_returns,
    HRPOpt,
    objective_functions,
    plotting,
    # black_litterman,
    # BlackLittermanModel,
)
from pypfopt.exceptions import OptimizationError

warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.legend').disabled = True
TODAY = dt.date.today()
BENCHMARK = ["SPY"]
DAYS_IN_YEAR = 365

rebalance_periods = {
    "daily": DAYS_IN_YEAR / 365,
    "weekly": DAYS_IN_YEAR / 52,
    "monthly": DAYS_IN_YEAR / 12,
    "month": DAYS_IN_YEAR / 12,
    "m": DAYS_IN_YEAR / 12,
    "quarterly": DAYS_IN_YEAR / 4,
    "quarter": DAYS_IN_YEAR / 4,
    "q": DAYS_IN_YEAR / 4,
    "6m": DAYS_IN_YEAR / 2,
    "2q": DAYS_IN_YEAR / 2,
    "1y": DAYS_IN_YEAR,
    "year": DAYS_IN_YEAR,
    "y": DAYS_IN_YEAR,
    "2y": DAYS_IN_YEAR * 2,
}

#defining colors for the allocation pie
CS = [
          "#ff9999",
          "#66b3ff",
          "#99ff99",
          "#ffcc99",
          "#f6c9ff",
          "#a6fff6",
          "#fffeb8",
          "#ffe1d4",
          "#cccdff",
          "#fad6ff",
      ]

def print_full_pd_df(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(x)

    # Resetting to default
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

class Engine:
    def __init__(
        self,
        start_date,
        portfolio,
        weights=None,
        rebalance=None,
        benchmark=None,
        end_date=TODAY,
        optimizer=None,
        max_vol=0.15,
        diversification=1,
        expected_returns=None,
        risk_model=None,
        # confidences=None,
        # view=None,
        min_weights=None,
        max_weights=None,
        risk_manager=None,
        data=pd.DataFrame(),
        data_all=None,
        cash_ratio_is_dynamic = False,
        cash_ratio=0.30,  # Add a cash_ratio parameter with a default value of 30%
        risk_score_data = None,
        risk_acceptance = 0.95
    ):
        self.cash_ratio_is_dynamic = cash_ratio_is_dynamic
        self.risk_score_data = risk_score_data
        self.risk_acceptance = risk_acceptance
        
        self.cash_ratio = cash_ratio
        # Adding "Cash" to portfolio
        if "Cash" not in portfolio:
            portfolio = ["Cash"] + portfolio

            # Generate synthetic cash prices with small fluctuations
            # Avoid err: ValueError: Quadratic form matrices must be symmetric/Hermitian
            np.random.seed(68)
            num_days = len(data.index)
            cash_returns = np.random.normal(0, 0.005, size=num_days)  # 0 mean return, 0.5% std dev
            cash_prices = 1 + np.cumsum(cash_returns)  # Start at 1 and simulate price changes

            # Add a constant price of 1 for the "Cash" column
            data["Cash"] = cash_prices

        self.portfolio = portfolio

        if benchmark is None:
            benchmark = BENCHMARK

        self.start_date = start_date
        self.end_date = end_date
        # self.portfolio = portfolio
        self.weights = weights
        self.benchmark = benchmark
        self.optimizer = optimizer
        self.rebalance = rebalance
        self.max_vol = max_vol
        self.diversification = diversification
        self.expected_returns = expected_returns
        if expected_returns is not None:
            assert expected_returns in ["mean_historical_return", "ema_historical_return", "capm_return"], f"Expected return method: {expected_returns} not supported yet! \n Set an appropriate expected returns parameter to your portfolio: mean_historical_return, ema_historical_return or capm_return."
        self.risk_model = risk_model
        if risk_model is not None:
            assert risk_model in ["sample_cov", "semicovariance", "exp_cov", "ledoit_wolf", "ledoit_wolf_constant_variance", "ledoit_wolf_single_factor", "ledoit_wolf_constant_correlation", "oracle_approximating"], f"Risk model: {risk_model} not supported yet! \n Set an appropriate risk model to your portfolio: sample_cov, semicovariance, exp_cov, ledoit_wolf, ledoit_wolf_constant_variance, ledoit_wolf_single_factor, ledoit_wolf_constant_correlation, oracle_approximating."
        self.max_weights = max_weights
        self.min_weights = min_weights
        self.risk_manager = risk_manager

        # print(f"ORIG.data:\n")
        # print_full_pd_df(data.head(100))

        # Filter the data to the portfolio and the date range
        # Start portfolio from the first date when all assets not N/A, NaN
        self.data = data.filter(self.portfolio).loc[pd.to_datetime(self.start_date).date():pd.to_datetime(self.end_date).date()]
        # print(f"self.data: {data}")

        # Find the first date with no missing data
        first_valid_date = self.data.dropna().index.min()
        # print(f"first_valid_date: {first_valid_date}")
        # print(f"The first date where all stocks have data is: {first_valid_date.strftime('%Y-%m-%d')}")
        # Update self.start_date with this date in "YYYY-MM-DD" format
        self.start_date = first_valid_date.strftime('%Y-%m-%d')

        # truncate self.data and self.data_all to start from first_valid_date
        self.data = self.data.loc[first_valid_date:]

        # Backup data to data_all
        self.data_all = data_all
        if self.data_all is None:
            self.data_all = data.filter(self.portfolio+self.benchmark).loc[first_valid_date:pd.to_datetime(self.end_date).date()]

        # Initialize weights if not provided
        if weights is None:
            weights = [1.0 / len(portfolio)] * len(portfolio)
        
        # Adjust weights to include the initial cash ratio
        adjusted_weights = [(1 - self.cash_ratio) * w for w in weights]
        self.weights = [self.cash_ratio] + adjusted_weights  # Add cash weight to the start of the weights list

        optimizers = {
            "EF": efficient_frontier,
            "MEANVAR": mean_var,
            "HRP": hrp,
            "MINVAR": min_var,
        }
        if self.optimizer is None and self.weights is None:
            self.weights = [1.0 / len(self.portfolio)] * len(self.portfolio)
        elif self.optimizer in optimizers.keys():
            if self.optimizer == "MEANVAR":
                self.weights = optimizers.get(self.optimizer)(self, vol_max=max_vol, perf=False)
            else:
                self.weights = optimizers.get(self.optimizer)(self, perf=False)

        # Rebalance if required
        if self.rebalance is not None:
            self.rebalance = make_rebalance(
                self.start_date,
                self.end_date,
                self.optimizer,
                self.portfolio,
                self.rebalance,
                self.weights,
                self.max_vol,
                self.diversification,
                self.min_weights,
                self.max_weights,
                self.expected_returns,
                self.risk_model,
                self.data,
                self.data_all,

                self.cash_ratio_is_dynamic,
                self.cash_ratio,  # Pass the cash_ratio to the rebalancing function
                self.risk_score_data,
                self.risk_acceptance,
            )

def calculate_cash_ratio(risk_acceptance = 0.85, risk_score_data = None, start_date = None, end_date = None, last_start_date = None, last_end_date = None):
    """
    Calculate the cash ratio based on the market risk score and risk acceptance.

    :param date: Date of calculating cash ratio.
    :param risk_acceptance: User's risk acceptance level (0.0 to 1.0). But SHOULD BE 0.8 to 0.9
    :param risk_score_data: Dictionary containing date as keys and risk scores as values.
    :return: Calculated cash ratio.
    """
    
    # Convert input date to a datetime object if it's not already
    if isinstance(end_date, str):
        date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    elif isinstance(end_date, dt.datetime):
        date = end_date.date()

    # Filter the dictionary for dates before or equal to the input date
    valid_dates = {k: v for k, v in risk_score_data.items() if k <= date}

    # Step 1: Get the most recent risk score before the given date
    most_recent_score = 35
    if valid_dates:
        most_recent_date = max(valid_dates.keys())
        most_recent_score = valid_dates[most_recent_date]

    # Step 2: Calculate the average risk score for the current rebalancing period
    current_period_scores = [score for d, score in risk_score_data.items() if start_date.date() <= d <= end_date.date()]
    current_avg_score = sum(current_period_scores) / len(current_period_scores) if current_period_scores else 35

    # Step 3: Calculate the average risk score for the last rebalancing period
    last_avg_score = 35
    if last_start_date and last_end_date:
        last_period_scores = [score for d, score in risk_score_data.items() if last_start_date.date() <= d <= last_end_date.date()]
        last_avg_score = sum(last_period_scores) / len(last_period_scores) if last_period_scores else 35

    # Step 4: Calculate the change in the average risk score from the last period to the current period/score
    # score_change = (most_recent_score - last_avg_score) / last_avg_score * 100
    score_change = (most_recent_score - last_avg_score)

    # Step 5: Weighting the factors (you can adjust these weights as necessary)

    # weight_most_recent = 0.55
    # weight_current_avg = 0.2
    # weight_score_change = 0.15
    # weight_last_avg = 0.1

    # weight_most_recent = 0.3
    # weight_current_avg = 0.2
    # weight_score_change = 0.1
    # weight_last_avg = 0.4

    weight_most_recent = 0.7
    weight_current_avg = 0.2
    weight_score_change = 0.0
    weight_last_avg = 0.1

    risk_score = (
        weight_most_recent * most_recent_score +
        weight_current_avg * current_avg_score +
        weight_score_change * score_change +
        weight_last_avg * last_avg_score
    )

    # Find the most recent risk score before the input date
    # if valid_dates:
    #     most_recent_date = max(valid_dates.keys())
    #     risk_score = valid_dates[most_recent_date]
    # else:
    #     risk_score = 40  # Default to score of 40 if no valid dates are found

    # Determine the maximum stock weight based on the risk score
    # Initialize max_stock_weight
    max_stock_weight = 0.5  # Default to the lowest weight if no range is matched

    # DEFAULT
    # risk_ranges = [
    #     (0, 35, 1.0),
    #     (35, 45, 0.7),
    #     (45, 57, 0.6),
    #     (57, 64, 0.5),
    #     (64, 100, 0.35)
    # ]

    risk_ranges = [
        (0, 35, 0.85),   # risk_score: 0-35 => base_stock_weight: 0.75
        (35, 40, 0.65),
        (40, 45, 0.45),
        (45, 55, 0.35),
        (55, 65, 0.20),
        (64, 100, 0.10)
    ]

    for lower_bound, upper_bound, base_stock_weight in risk_ranges:
        if lower_bound <= risk_score < upper_bound:
            # Calculate the position within the current range (0.0 at lower bound to 1.0 at upper bound)
            range_position = (risk_score - lower_bound) / (upper_bound - lower_bound)
            # Adjust max_stock_weight based on where the risk_score falls within the range
            max_stock_weight = base_stock_weight * (1.0 + range_position * (1 - base_stock_weight))
            break

    # Calculate desired stock ratio based on risk acceptance
    desired_stock_ratio = max_stock_weight * risk_acceptance

    # Calculate and return the cash ratio
    cash_ratio = 1.0 - desired_stock_ratio

    return cash_ratio

def calculate_percent_change_from_cagr(cagr, start_date, end_date):
    # Calculate the number of years between the start and end dates
    num_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25

    # Calculate the percent change using the formula
    percent_change = ((1 + cagr)**num_years - 1) * 100
    return percent_change

def get_returns(stocks, wts, start_date, end_date=TODAY):
    pass
    
    # if len(stocks) > 1:
        # assets = yf.download(stocks, start=start_date, end=end_date, progress=False)["Adj Close"]
        # assets = assets.filter(stocks)
        # initial_alloc = wts/assets.iloc[0]
        # if initial_alloc.isna().any():
            # raise ValueError("Some stock is not available at initial state!")
        # portfolio_value = (assets * initial_alloc).sum(axis=1)
        # returns = portfolio_value.pct_change()[1:]
        # return returns
    # else:
        # df = yf.download(stocks, start=start_date, end=end_date, progress=False)["Adj Close"]
        # df = pd.DataFrame(df)
        # returns = df.pct_change()[1:]
        # return returns


def get_returns_from_data(data, wts, stocks):
    assets = data.filter(stocks)
    initial_alloc = wts/assets.iloc[0]
    
    if initial_alloc.isna().any():        
        raise ValueError("Some stock is not available at initial state!")
    portfolio_value = (assets * initial_alloc).sum(axis=1)
    returns = portfolio_value.pct_change()[1:]
    return returns


def calculate_information_ratio(returns, benchmark_returns, days=252) -> float:
    return_difference = returns - benchmark_returns
    volatility = return_difference.std() * np.sqrt(days)
    information_ratio_result = return_difference.mean() / volatility
    return information_ratio_result


def graph_allocation(my_portfolio):
    fig1, ax1 = plt.subplots()
    ax1.pie(
        my_portfolio.weights,
        labels=my_portfolio.portfolio,
        autopct="%1.1f%%",
        shadow=False,
    )
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Portfolio's allocation")
    # plt.show()
    
    

def graph_assets_price_history(data, save=False):
    # Individual asset price
    plt.figure(figsize=(12,6))
    title = "Assets Price History"
    my_stocks = data
    #create and pllot the graph
    for c in my_stocks.columns.values:
        plt.plot(my_stocks[c], label = c)
    plt.title(title)
    plt.xlabel("Date", fontsize =18)
    plt.ylabel("Price", fontsize=18)
    plt.legend(my_stocks.columns.values , loc = "upper left")
    if save:
        plt.savefig("assets_price_history.png")
    # plt.show()



def empyrial(my_portfolio, rf=0.0, sigma_value=1, confidence_value=0.95, report=False, save_pdf=False, filename="empyrial_report.pdf"):

    creturns = None
    if isinstance(my_portfolio.rebalance, pd.DataFrame):
        # we want to get the dataframe with the dates and weights
        rebalance_schedule = my_portfolio.rebalance

        columns = []

        for date in rebalance_schedule.columns:
            date = date[0:10]
            columns.append(date)
        rebalance_schedule.columns = columns

        # then want to make a list of the dates and start with our first date
        dates = [my_portfolio.start_date]

        # then our rebalancing dates into that list
        dates = dates + rebalance_schedule.columns.to_list()

        datess = []
        for date in dates:
            date = date[0:10]
            datess.append(date)
        dates = datess
        # this will hold returns
        returns = pd.Series()
        cumulative_returns_list = []  # To store cumulative returns after each rebalancement

        # then we want to be able to call the dates like tuples
        for i in range(len(dates) - 1):
            # get our weights
            weights = rebalance_schedule[str(dates[i + 1])]

            # then we want to get the returns
            
            # add_returns = get_returns(
                # my_portfolio.portfolio,
                # weights,
                # start_date=dates[i],
                # end_date=dates[i + 1],
            # )
            add_returns = None
            if not my_portfolio.data.empty:
                add_returns = get_returns_from_data(my_portfolio.data, weights, my_portfolio.portfolio)
            else:
                add_returns = get_returns(
                    my_portfolio.portfolio,
                    weights,
                    start_date=dates[i],
                    end_date=dates[i + 1],
                )

            # then append those returns
            # returns = returns.append(add_returns)
            # NOTE: recent versions of pandas >> use concat
            # returns = pd.concat([returns, add_returns])
            
            # Concatenate the Series or DataFrames
            # Then, drop duplicate indices, keeping the first occurrence
            combined = pd.concat([returns, add_returns])
            combined_keep_first = combined[~combined.index.duplicated(keep='first')]
            combined_keep_last = combined[~combined.index.duplicated(keep='last')]
            combined_first_part = combined_keep_first.iloc[:i+1]
            combined_last_part = combined_keep_last.iloc[i+1:]
            
            returns = pd.concat([combined_first_part, combined_last_part])

            # Recalculate cumulative returns after each rebalance
            creturns = (returns + 1).cumprod()
            cumulative_returns_list.append(creturns)
    else:
        if not my_portfolio.data.empty:
            returns = get_returns_from_data(my_portfolio.data, my_portfolio.weights, my_portfolio.portfolio)
        else:
            returns = get_returns(
              my_portfolio.portfolio,
              my_portfolio.weights,
              start_date=my_portfolio.start_date,
              end_date=my_portfolio.end_date,
            )
        
        # Calculate cumulative returns of the portfolio
        creturns = (returns + 1).cumprod()

    ### Orig. calculation of creturns:
    # Calculate cumulative returns of the portfolio
    # creturns = (returns + 1).cumprod()

    # Get benchmark prices (assuming benchmark is a single asset)
    benchmark_prices = my_portfolio.data_all[my_portfolio.benchmark[0]]
    
    ### CALC metrics using PyPortfolioOpt
    data_no_cash = my_portfolio.data.drop(columns=["Cash"])
    cov_matrix = risk_models.sample_cov(data_no_cash) # covariance matrix
    corr_matrix = data_no_cash.corr() # corr matrix

    ### Risk manager
    try:
        if list(my_portfolio.risk_manager.keys())[0] == "Stop Loss":

            values = []
            for r in creturns:
                if r <= 1 + my_portfolio.risk_manager["Stop Loss"]:
                    values.append(r)
                else:
                    pass

            try:
                date = creturns[creturns == values[0]].index[0]
                date = str(date.to_pydatetime())
                my_portfolio.end_date = date[0:10]
                returns = returns[: my_portfolio.end_date]

            except Exception as e:
                pass

        if list(my_portfolio.risk_manager.keys())[0] == "Take Profit":

            values = []
            for r in creturns:
                if r >= 1 + my_portfolio.risk_manager["Take Profit"]:
                    values.append(r)
                else:
                    pass

            try:
                date = creturns[creturns == values[0]].index[0]
                date = str(date.to_pydatetime())
                my_portfolio.end_date = date[0:10]
                returns = returns[: my_portfolio.end_date]

            except Exception as e:
                pass

        if list(my_portfolio.risk_manager.keys())[0] == "Max Drawdown":

            drawdown = qs.stats.to_drawdown_series(returns)

            values = []
            for r in drawdown:
                if r <= my_portfolio.risk_manager["Max Drawdown"]:
                    values.append(r)
                else:
                    pass

            try:
                date = drawdown[drawdown == values[0]].index[0]
                date = str(date.to_pydatetime())
                my_portfolio.end_date = date[0:10]
                returns = returns[: my_portfolio.end_date]

            except Exception as e:
                pass

    except Exception as e:
        pass
    
    benchmark = None
    if not my_portfolio.data.empty:
        benchmark = get_returns_from_data(
            data=my_portfolio.data_all,
            wts=[1],
            stocks=my_portfolio.benchmark)
        benchmark = benchmark.dropna()
    else:
        benchmark = get_returns(
            my_portfolio.benchmark,
            wts=[1],
            start_date=my_portfolio.start_date,
            end_date=my_portfolio.end_date,
        )
        benchmark = benchmark.dropna()
    
    if isinstance(benchmark, pd.DataFrame):
        # print("benchmark is a DataFrame")
        pass
    elif isinstance(benchmark, pd.Series):
        # print("benchmark is a Series")
        # Convert Series to DataFrame
        benchmark = benchmark.to_frame()
    
    # tz_localize
    # Ensure the index of the benchmark is a DatetimeIndex
    if not isinstance(benchmark.index, pd.DatetimeIndex):
        benchmark.index = pd.to_datetime(benchmark.index)
    # Ensure returns has a DatetimeIndex
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
        
    # Ensure that 'returns' has a name
    if returns.name is None:
        returns.name = 'Portfolio Ret.'
    # Ensure that 'benchmark' has a name
    if isinstance(benchmark, pd.Series):
        if benchmark.name is None:
            benchmark.name = 'Benchmark'
    
    first_date = returns.index.min().strftime('%Y-%m-%d')
    last_date = returns.index.max().strftime('%Y-%m-%d')
    num_years_invested = (returns.index.max() - returns.index.min()).days / 365.25
    num_years_invested = str(round(num_years_invested, 2))
    # STATS
    CAGR = cagr(returns, period='daily', annualization=None)
    CAGR_VAL = CAGR
    # CAGR = round(CAGR, 2)
    # CAGR = CAGR.tolist()
    CAGR = str(round(CAGR * 100, 2)) + "%"
    
    # Assuming CAGR is already calculated and is in decimal form (not percentage)
    portfolio_percent_change = calculate_percent_change_from_cagr(CAGR_VAL, first_date, last_date)
    portfolio_percent_change_str = str(round(portfolio_percent_change, 2)) + "%"
    portfolio_percent_change_str = f"{portfolio_percent_change_str}"

    CUM = cum_returns(returns, starting_value=0, out=None) * 100
    CUM = CUM.iloc[-1]
    CUM = CUM.tolist()
    CUM = f"{str(round(CUM, 2))} % over {num_years_invested} yrs"

    VOL = qs.stats.volatility(returns, annualize=True)
    VOL = VOL.tolist()
    VOL = str(round(VOL * 100, 2)) + " %"

    SR = qs.stats.sharpe(returns, rf=rf)
    SR = np.round(SR, decimals=2)
    SR = str(SR)

    empyrial.SR = SR

    CR = qs.stats.calmar(returns)
    CR = CR.tolist()
    CR = str(round(CR, 2))

    empyrial.CR = CR

    STABILITY = stability_of_timeseries(returns)
    STABILITY = round(STABILITY, 2)
    STABILITY = str(STABILITY)

    MD = max_drawdown(returns, out=None)
    MD = str(round(MD * 100, 2)) + " %"

    """OR = omega_ratio(returns, risk_free=0.0, required_return=0.0)
    OR = round(OR,2)
    OR = str(OR)
    print(OR)"""

    SOR = sortino_ratio(returns, required_return=0, period='daily')
    SOR = round(SOR, 2)
    SOR = str(SOR)

    SK = qs.stats.skew(returns)
    SK = round(SK, 2)
    SK = SK.tolist()
    SK = str(SK)

    KU = qs.stats.kurtosis(returns)
    KU = round(KU, 2)
    KU = KU.tolist()
    KU = str(KU)

    TA = tail_ratio(returns)
    TA = round(TA, 2)
    TA = str(TA)

    CSR = qs.stats.common_sense_ratio(returns)
    CSR = round(CSR, 2)
    CSR = CSR.tolist()
    CSR = str(CSR)

    VAR = qs.stats.value_at_risk(
        returns, sigma=sigma_value, confidence=confidence_value
    )
    VAR = np.round(VAR, decimals=2)
    VAR = str(VAR * 100) + " %"

    alpha, beta = alpha_beta(returns, benchmark, risk_free=rf)
    AL = round(alpha, 2)
    BTA = round(beta, 2)

    def condition(x):
        return x > 0

    win = sum(condition(x) for x in returns)
    total = len(returns)
    win_ratio = win / total
    win_ratio = win_ratio * 100
    win_ratio = round(win_ratio, 2)

    IR = calculate_information_ratio(returns, benchmark.iloc[:, 0])
    IR = round(IR, 2)

    data = {
        "": [
            "Annual return CAGR",
            # "Return",
            "Cumulative return",
            "Sharpe ratio",
            "Sortino ratio",
            "Common sense ratio",
            "Annual volatility",
            "Max Drawdown",
            "Calmar ratio",
            "Winning day ratio",
            "Information ratio",
            "Stability",
            "Skew",
            "Kurtosis",
            "Tail Ratio",
            "Daily value at risk",
            "Alpha",
            "Beta",
        ],
        "Portfolio/Backtest": [
            CAGR,
            # portfolio_percent_change_str,
            CUM,
            SR,
            SOR,
            CSR,
            VOL,
            MD,            
            CR,
            f"{win_ratio}%",
            IR,
            STABILITY,
            SK,
            KU,
            TA,
            VAR,
            AL,
            BTA,
        ],
    }

    # Create DataFrame
    df_backtest = pd.DataFrame(data)
    df_backtest.set_index("", inplace=True)
    df_backtest.style.set_properties(
        **{"background-color": "white", "color": "black", "border-color": "grey"}
    )
    display(df_backtest)

    empyrial.df = data

    y = []
    for x in returns:
        y.append(x)

    arr = np.array(y)
    # arr
    # returns.index
    my_color = np.where(arr >= 0, "blue", "grey")
    ret = plt.figure(figsize=(30, 8))
    plt.vlines(x=returns.index, ymin=0, ymax=arr, color=my_color, alpha=0.4)
    plt.title("Returns")

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
    
    # Calculate benchmark metrics
    # benchmark_CAGR = cagr(benchmark, period='daily', annualization=None)
    # benchmark_CAGR = str(round(benchmark_CAGR * 100, 2)) + "%"

    # benchmark_CUM = cum_returns(benchmark, starting_value=0, out=None) * 100
    # benchmark_CUM = benchmark_CUM.iloc[-1]
    # benchmark_CUM = str(round(benchmark_CUM, 2)) + "%"

    # benchmark_VOL = qs.stats.volatility(benchmark, annualize=True)
    # benchmark_VOL = str(round(benchmark_VOL * 100, 2)) + " %"

    # benchmark_SR = qs.stats.sharpe(benchmark, rf=rf)
    # benchmark_SR = str(np.round(benchmark_SR, decimals=2))

    # benchmark_CR = qs.stats.calmar(benchmark)
    # benchmark_CR = str(round(benchmark_CR, 2))

    # benchmark_STABILITY = stability_of_timeseries(benchmark)
    # benchmark_STABILITY = str(round(benchmark_STABILITY, 2))

    # benchmark_MD = max_drawdown(benchmark)
    # benchmark_MD = str(round(benchmark_MD * 100, 2)) + " %"

    # benchmark_SOR = sortino_ratio(benchmark, required_return=0, period='daily')
    # benchmark_SOR = str(round(benchmark_SOR, 2))

    # benchmark_SK = qs.stats.skew(benchmark)
    # benchmark_SK = str(round(benchmark_SK, 2))

    # benchmark_KU = qs.stats.kurtosis(benchmark)
    # benchmark_KU = str(round(benchmark_KU, 2))

    # benchmark_TA = tail_ratio(benchmark)
    # benchmark_TA = str(round(benchmark_TA, 2))

    # benchmark_CSR = qs.stats.common_sense_ratio(benchmark)
    # benchmark_CSR = str(round(benchmark_CSR, 2))

    # benchmark_VAR = qs.stats.value_at_risk(benchmark, sigma=sigma_value, confidence=confidence_value)
    # benchmark_VAR = str(np.round(benchmark_VAR, decimals=2) * 100) + " %"
    
    benchmark_df = pd.DataFrame(),
    if isinstance(benchmark, pd.Series):
        # Convert Series to DataFrame
        benchmark_df = benchmark.to_frame()
    elif isinstance(benchmark, pd.DataFrame):
        benchmark_df = benchmark
    
    # benchmark_df is DataFrame
    # Assuming benchmark_df has a single column; adjust 'benchmark_column' as needed
    benchmark_column = benchmark_df.columns[0]  # Change this to the correct column name if necessary

    # Calculate benchmark metrics
    benchmark_CAGR = cagr(benchmark_df[benchmark_column], period='daily', annualization=None)
    benchmark_CAGR_VAL = benchmark_CAGR
    benchmark_CAGR = str(round(benchmark_CAGR * 100, 2)) + "%"
    
    benchmark_percent_change = calculate_percent_change_from_cagr(benchmark_CAGR_VAL, first_date, last_date)
    benchmark_percent_change_str = str(round(benchmark_percent_change, 2)) + "%"
    benchmark_percent_change_str = f"{benchmark_percent_change_str}"

    benchmark_CUM = cum_returns(benchmark_df[benchmark_column], starting_value=0, out=None) * 100
    benchmark_CUM = benchmark_CUM.iloc[-1]
    benchmark_CUM = f"{str(round(benchmark_CUM, 2))} % over {num_years_invested} yrs"

    benchmark_VOL = qs.stats.volatility(benchmark_df[benchmark_column], annualize=True)
    benchmark_VOL = str(round(benchmark_VOL * 100, 2)) + " %"

    benchmark_SR = qs.stats.sharpe(benchmark_df[benchmark_column], rf=rf)
    benchmark_SR = str(np.round(benchmark_SR, decimals=2))

    benchmark_CR = qs.stats.calmar(benchmark_df[benchmark_column])
    benchmark_CR = str(round(benchmark_CR, 2))

    benchmark_STABILITY = stability_of_timeseries(benchmark_df[benchmark_column])
    benchmark_STABILITY = str(round(benchmark_STABILITY, 2))

    benchmark_MD = max_drawdown(benchmark_df[benchmark_column])
    benchmark_MD = str(round(benchmark_MD * 100, 2)) + " %"

    benchmark_SOR = sortino_ratio(benchmark_df[benchmark_column], required_return=0, period='daily')
    benchmark_SOR = str(round(benchmark_SOR, 2))

    benchmark_SK = qs.stats.skew(benchmark_df[benchmark_column])
    benchmark_SK = str(round(benchmark_SK, 2))

    benchmark_KU = qs.stats.kurtosis(benchmark_df[benchmark_column])
    benchmark_KU = str(round(benchmark_KU, 2))

    benchmark_TA = tail_ratio(benchmark_df[benchmark_column])
    benchmark_TA = str(round(benchmark_TA, 2))

    benchmark_CSR = qs.stats.common_sense_ratio(benchmark_df[benchmark_column])
    benchmark_CSR = str(round(benchmark_CSR, 2))

    benchmark_VAR = qs.stats.value_at_risk(benchmark_df[benchmark_column], sigma=sigma_value, confidence=confidence_value)
    benchmark_VAR = str(np.round(benchmark_VAR, decimals=2) * 100) + " %"
    
    stats_benchmark = {
        "": [
            "Annual return CAGR",
            # "Return",
            "Cumulative return",
            "Sharpe ratio",
            "Sortino ratio",
            "Common sense ratio",
            "Annual volatility",
            "Max Drawdown",
            "Calmar ratio",
            "Winning day ratio",
            "Information ratio",
            "Stability",
            "Skew",
            "Kurtosis",
            "Tail Ratio",
            "Daily value at risk",
            "Alpha",
            "Beta",
        ],
        "Benchmark": [
            benchmark_CAGR,
            # benchmark_percent_change_str,
            benchmark_CUM,
            benchmark_SR,
            benchmark_SOR,
            benchmark_CSR,
            benchmark_VOL,
            benchmark_MD,
            benchmark_CR,
            "N/A",
            "N/A",
            benchmark_STABILITY,
            benchmark_SK,
            benchmark_KU,
            benchmark_TA,
            benchmark_VAR,
            "--",
            "--",
        ],
    }

    # Create DataFrame
    df_benchmark = pd.DataFrame(stats_benchmark)
    df_benchmark.set_index("", inplace=True)
    df_benchmark.style.set_properties(
        **{"background-color": "white", "color": "black", "border-color": "grey"}
    )
    display(df_benchmark)
    # END: Calculate benchmark metrics

    try:
        empyrial.orderbook = make_rebalance.output
    except Exception as e:
        OrderBook = pd.DataFrame(
            {
                "Assets": my_portfolio.portfolio,
                "Allocation": my_portfolio.weights,
            }
        )

        empyrial.orderbook = OrderBook.T

    wts = copy.deepcopy(my_portfolio.weights)
    indices = [i for i, x in enumerate(wts) if x == 0.0]

    # while 0.0 in wts:
    #     wts.remove(0.0)

    # for i in sorted(indices, reverse=True):
    #     del my_portfolio.portfolio[i]

    ### Performance Report ###
    # Portfolio performance metrics
    portfolio_performance = {
        "cagr": empyrial.CAGR,
        "cumulative_return": empyrial.CUM,
        "volatility": empyrial.VOL,
        "sharpe_ratio": empyrial.SR,
        "calmar_ratio": empyrial.CR,
        "max_drawdown": empyrial.MD,
        "sortino_ratio": empyrial.SOR,
        "alpha": empyrial.AL,
        "beta": empyrial.BTA,
    }

    # Benchmark performance metrics
    benchmark_performance = {
        "cagr": benchmark_CAGR,
        "cumulative_return": benchmark_CUM,
        "volatility": benchmark_VOL,
        "sharpe_ratio": benchmark_SR,
        "calmar_ratio": benchmark_CR,
        "max_drawdown": benchmark_MD,
        "sortino_ratio": benchmark_SOR,
    }

    if not report:
      qs.plots.returns(returns, benchmark, cumulative=True)
      qs.plots.yearly_returns(returns, benchmark),
      qs.plots.monthly_heatmap(returns, benchmark)
      qs.plots.drawdown(returns)
      qs.plots.drawdowns_periods(returns)
      qs.plots.rolling_volatility(returns)
      qs.plots.rolling_sharpe(returns)
      qs.plots.rolling_beta(returns, benchmark)
      graph_opt(my_portfolio.portfolio, wts, pie_size=7, font_size=14)

    else:
        ret.savefig("ret.png")
        
        qs.plots.returns(returns, benchmark, cumulative=True, savefig="retbench.png")
        qs.plots.yearly_returns(returns, benchmark, savefig="y_returns.png"),
        qs.plots.monthly_heatmap(returns, benchmark, savefig="heatmap.png")
        qs.plots.drawdown(returns, savefig="drawdown.png")
        qs.plots.drawdowns_periods(returns, savefig="d_periods.png")
        qs.plots.rolling_volatility(returns, savefig="rvol.png")
        qs.plots.rolling_sharpe(returns, savefig="rsharpe.png")
        qs.plots.rolling_beta(returns, benchmark, savefig="rbeta.png")
        graph_opt(my_portfolio.portfolio, wts, pie_size=7, font_size=14, save=True)
        graph_assets_price_history(my_portfolio.data, save=True)
        
        # Create absolute value comparision plot
        # Ensure the initial values are set for both portfolio and benchmark
        initial_portfolio_value = 1000000
        initial_benchmark_value = 1000000
        # Calculate cumulative values
        portfolio_values = (returns + 1).cumprod() * initial_portfolio_value
        benchmark_values = (benchmark + 1).cumprod() * initial_benchmark_value
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values.index, portfolio_values, label="Portfolio", color='blue')
        plt.plot(benchmark_values.index, benchmark_values, label="Benchmark", color='orange')
        # Adding titles and labels
        plt.title("Portfolio vs Benchmark - Abs. Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        # Save plot as image
        plot_port_vs_benchmark_filename = "portfolio_vs_benchmark.png"
        plt.savefig(plot_port_vs_benchmark_filename)
        plt.close()
        
        plt.close('all')
        
        ### PDF
        if save_pdf:
            # else >> save pdf
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("arial", "B", 14)
            pdf.image(
            "https://i.postimg.cc/yN4LPM46/my-portolio.png",
            x=None,
            y=None,
            w=45,
            h=5,
            type="",
            link="https://google.com/",
            )
            pdf.cell(20, 15, f"Report", ln=1)
            pdf.set_font("arial", size=11)
            pdf.image("allocation.png", x=135, y=0, w=70, h=70, type="", link="")
            pdf.cell(20, 7, f"Start date: " + str(first_date), ln=1)
            pdf.cell(20, 7, f"End date: " + str(last_date), ln=1)

            # pdf.cell(20, 7, f"", ln=1)
            # pdf.cell(20, 7, f"Annual return: " + str(CAGR), ln=1)
            # pdf.cell(20, 7, f"Cumulative return: " + str(CUM), ln=1)
            # pdf.cell(20, 7, f"Annual volatility: " + str(VOL), ln=1)
            # pdf.cell(20, 7, f"Winning day ratio: " + str(win_ratio), ln=1)
            # pdf.cell(20, 7, f"Sharpe ratio: " + str(SR), ln=1)
            # pdf.cell(20, 7, f"Calmar ratio: " + str(CR), ln=1)
            # pdf.cell(20, 7, f"Information ratio: " + str(IR), ln=1)
            # pdf.cell(20, 7, f"Stability: " + str(STABILITY), ln=1)
            # pdf.cell(20, 7, f"Max drawdown: " + str(MD), ln=1)
            # pdf.cell(20, 7, f"Sortino ratio: " + str(SOR), ln=1)
            # pdf.cell(20, 7, f"Skew: " + str(SK), ln=1)
            # pdf.cell(20, 7, f"Kurtosis: " + str(KU), ln=1)
            # pdf.cell(20, 7, f"Tail ratio: " + str(TA), ln=1)
            # pdf.cell(20, 7, f"Common sense ratio: " + str(CSR), ln=1)
            # pdf.cell(20, 7, f"Daily value at risk: " + str(VAR), ln=1)
            # pdf.cell(20, 7, f"Alpha: " + str(AL), ln=1)
            # pdf.cell(20, 7, f"Beta: " + str(BTA), ln=1)

            # Printing to PDF side by side
            pdf.cell(20, 7, f"", ln=1)
            pdf.set_font("arial", "B", 11)
            pdf.cell(60, 7, "Metric", ln=0)
            pdf.cell(60, 7, "Portfolio", ln=0)
            pdf.cell(60, 7, "Benchmark", ln=1)
            pdf.set_font("arial", size=11)
            pdf.cell(60, 7, f"Annual return", ln=0)
            pdf.cell(60, 7, f"{CAGR}", ln=0)
            pdf.cell(60, 7, f"{benchmark_CAGR}", ln=1)
            
            pdf.cell(60, 7, f"Return", ln=0)
            pdf.cell(60, 7, f"{portfolio_percent_change_str}", ln=0)
            pdf.cell(60, 7, f"{benchmark_percent_change_str}", ln=1)

            pdf.cell(60, 7, f"Cumulative return", ln=0)
            pdf.cell(60, 7, f"{CUM}", ln=0)
            pdf.cell(60, 7, f"{benchmark_CUM}", ln=1)

            pdf.cell(60, 7, f"Annual volatility", ln=0)
            pdf.cell(60, 7, f"{VOL}", ln=0)
            pdf.cell(60, 7, f"{benchmark_VOL}", ln=1)

            pdf.cell(60, 7, f"Winning day ratio", ln=0)
            pdf.cell(60, 7, f"{win_ratio}%", ln=0)
            pdf.cell(60, 7, f"-", ln=1)  # Assuming no winning day ratio calculation for benchmark

            pdf.cell(60, 7, f"Sharpe ratio", ln=0)
            pdf.cell(60, 7, f"{SR}", ln=0)
            pdf.cell(60, 7, f"{benchmark_SR}", ln=1)

            pdf.cell(60, 7, f"Calmar ratio", ln=0)
            pdf.cell(60, 7, f"{CR}", ln=0)
            pdf.cell(60, 7, f"{benchmark_CR}", ln=1)

            pdf.cell(60, 7, f"Information ratio", ln=0)
            pdf.cell(60, 7, f"{IR}", ln=0)
            pdf.cell(60, 7, f"-", ln=1)  # Information ratio typically not calculated for the benchmark

            pdf.cell(60, 7, f"Stability", ln=0)
            pdf.cell(60, 7, f"{STABILITY}", ln=0)
            pdf.cell(60, 7, f"{benchmark_STABILITY}", ln=1)

            pdf.cell(60, 7, f"Max drawdown", ln=0)
            pdf.cell(60, 7, f"{MD}", ln=0)
            pdf.cell(60, 7, f"{benchmark_MD}", ln=1)

            pdf.cell(60, 7, f"Sortino ratio", ln=0)
            pdf.cell(60, 7, f"{SOR}", ln=0)
            pdf.cell(60, 7, f"{benchmark_SOR}", ln=1)

            pdf.cell(60, 7, f"Skew", ln=0)
            pdf.cell(60, 7, f"{SK}", ln=0)
            pdf.cell(60, 7, f"{benchmark_SK}", ln=1)

            pdf.cell(60, 7, f"Kurtosis", ln=0)
            pdf.cell(60, 7, f"{KU}", ln=0)
            pdf.cell(60, 7, f"{benchmark_KU}", ln=1)

            pdf.cell(60, 7, f"Tail ratio", ln=0)
            pdf.cell(60, 7, f"{TA}", ln=0)
            pdf.cell(60, 7, f"{benchmark_TA}", ln=1)

            pdf.cell(60, 7, f"Common sense ratio", ln=0)
            pdf.cell(60, 7, f"{CSR}", ln=0)
            pdf.cell(60, 7, f"{benchmark_CSR}", ln=1)

            pdf.cell(60, 7, f"Daily value at risk", ln=0)
            pdf.cell(60, 7, f"{VAR}", ln=0)
            pdf.cell(60, 7, f"{benchmark_VAR}", ln=1)

            pdf.cell(60, 7, f"Alpha", ln=0)
            pdf.cell(60, 7, f"{AL}", ln=1)  # Alpha and Beta calculated for portfolio only
            pdf.cell(60, 7, f"Beta", ln=0)
            pdf.cell(60, 7, f"{BTA}", ln=1)
            
            # Add the plot image to the PDF
            pdf.add_page()
            pdf.image(plot_port_vs_benchmark_filename, x=None, y=None, w=200)
            pdf.cell(20, 7, f"", ln=1)
            
            # Other plots
            pdf.image("ret.png", x=-20, y=None, w=250, h=80, type="", link="")
            pdf.cell(20, 7, f"", ln=1)
            pdf.image("y_returns.png", x=-20, y=None, w=200, h=100, type="", link="")
            pdf.cell(20, 7, f"", ln=1)
            pdf.image("retbench.png", x=None, y=None, w=200, h=100, type="", link="")
            pdf.cell(20, 7, f"", ln=1)
            pdf.image("heatmap.png", x=None, y=None, w=200, h=80, type="", link="")
            pdf.cell(20, 7, f"", ln=1)
            pdf.image("drawdown.png", x=None, y=None, w=200, h=80, type="", link="")
            pdf.cell(20, 7, f"", ln=1)
            pdf.image("d_periods.png", x=None, y=None, w=200, h=80, type="", link="")
            pdf.cell(20, 7, f"", ln=1)
            pdf.image("rvol.png", x=None, y=None, w=190, h=80, type="", link="")
            pdf.cell(20, 7, f"", ln=1)
            pdf.image("rsharpe.png", x=None, y=None, w=190, h=80, type="", link="")
            pdf.cell(20, 7, f"", ln=1)
            pdf.image("rbeta.png", x=None, y=None, w=190, h=80, type="", link="")

            pdf.output(dest="F", name=filename)
            print("The PDF was generated successfully!")

    # Return relevant data
    return {
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
        "portfolio_df": df_backtest,
        "stock_prices": my_portfolio.data,
        "benchmark_df": df_benchmark,
        "benchmark_prices": benchmark_prices,
        "cumulative_returns": creturns,
        "weights_orderbook": empyrial.orderbook,
    }

def flatten(subject) -> list:
    muster = []
    for item in subject:
        if isinstance(item, (list, tuple, set)):
            muster.extend(flatten(item))
        else:
            muster.append(item)
    return muster


def graph_opt(my_portfolio, my_weights, pie_size, font_size, save=False):
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(pie_size, pie_size)
    ax1.pie(my_weights, labels=my_portfolio, autopct="%1.1f%%", shadow=False, colors=CS)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.rcParams["font.size"] = font_size
    if save:
      plt.savefig("allocation.png")
    # plt.show()


def equal_weighting(my_portfolio) -> list:
    return [1.0 / len(my_portfolio.portfolio)] * len(my_portfolio.portfolio)

def efficient_frontier(my_portfolio, perf=True) -> list:
    if not my_portfolio.data.empty:
        # print("---data NOT EMPTY >> using custom data")
        df = my_portfolio.data
        pass
    else:
        # print("---data is EMPTY >> yfinance?")
        pass
    
        # ohlc = yf.download(
            # my_portfolio.portfolio,
            # start=my_portfolio.start_date,
            # end=my_portfolio.end_date,
            # progress=False,
        # )
        # prices = ohlc["Adj Close"].dropna(how="all")
        # df = prices.filter(my_portfolio.portfolio)

    # sometimes we will pick a date range where company isn't public we can't set price to 0 so it has to go to 1
    df = df.fillna(1)
    if my_portfolio.expected_returns == None:
        my_portfolio.expected_returns = 'mean_historical_return'
    if my_portfolio.risk_model == None:
        my_portfolio.risk_model = 'sample_cov'
    mu = expected_returns.return_model(df, method=my_portfolio.expected_returns)
    S = risk_models.risk_matrix(df, method=my_portfolio.risk_model)

    # optimize for max sharpe ratio
    ef = EfficientFrontier(mu, S)

    # Add constraints
    ef.add_objective(objective_functions.L2_reg, gamma=my_portfolio.diversification)
    if my_portfolio.min_weights is not None:
        ef.add_constraint(lambda x: x >= my_portfolio.min_weights)
    if my_portfolio.max_weights is not None:
        ef.add_constraint(lambda x: x <= my_portfolio.max_weights)

    # ef.max_sharpe()  # or whatever optimization you're performing

    # Initial diversification setting
    diversification = my_portfolio.diversification
    # Retry loop with a maximum number of retries
    max_retries = 5
    retry_count = 0
    success = False
    while not success and retry_count < max_retries:
        try:
            if retry_count > 0:
                ef = EfficientFrontier(mu, S)
                ef.add_objective(objective_functions.L2_reg, gamma=diversification)

            ef.max_sharpe()  # Try to optimize with current diversification
            success = True
        except OptimizationError as e:
            retry_count += 1
            print(f"Optimization failed on attempt {retry_count}: {e}")
            print("Adjusting diversification and trying again...")
            diversification *= 0.95  # Adjust the diversification parameter (reduce by 10%)
        except ValueError as e:
            retry_count += 1
            print(f"Max Sharpe optimization failed: {e}\n")
            # Fall back to a different strategy or notify the user
            if "must have an expected return exceeding the risk-free rate" in str(e):
                print("Optimization failed: all expected returns are below the risk-free rate.")
                print(f"Max Sharpe optimization failed: {e}\nSwitching to Min Volatility...")
                ef.min_volatility()

    cleaned_weights = ef.clean_weights()
    wts = cleaned_weights.items()

    result = []
    for val in wts:
        a, b = map(list, zip(*[val]))
        result.append(b)

    if perf is True:
        pred = ef.portfolio_performance(verbose=True)

    return flatten(result)


def hrp(my_portfolio, perf=True) -> list:
    if not my_portfolio.data.empty:
        prices = my_portfolio.data
    else:
        pass
        
        # ohlc = yf.download(
            # my_portfolio.portfolio,
            # start=my_portfolio.start_date,
            # end=my_portfolio.end_date,
            # progress=False,
        # )
        # prices = ohlc["Adj Close"].dropna(how="all")
        # prices = prices.filter(my_portfolio.portfolio)

    # sometimes we will pick a date range where company isn't public we can't set price to 0 so it has to go to 1
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

    if perf is True:
        hrp.portfolio_performance(verbose=True)

    return flatten(result)


def mean_var(my_portfolio, vol_max=0.15, perf=True) -> list:
    if not my_portfolio.data.empty:
        prices = my_portfolio.data
    else:
        pass
        
        # ohlc = yf.download(
            # my_portfolio.portfolio,
            # start=my_portfolio.start_date,
            # end=my_portfolio.end_date,
            # progress=False,
        # )
        # prices = ohlc["Adj Close"].dropna(how="all")
        # prices = prices.filter(my_portfolio.portfolio)

    # sometimes we will pick a date range where company isn't public we can't set price to 0 so it has to go to 1
    prices = prices.fillna(1)

    if my_portfolio.expected_returns == None:
        my_portfolio.expected_returns = 'capm_return'
    if my_portfolio.risk_model == None:
        my_portfolio.risk_model = 'ledoit_wolf'
    
    # Calculate expected returns and risk matrix
    mu = expected_returns.return_model(prices, method=my_portfolio.expected_returns)
    S = risk_models.risk_matrix(prices, method=my_portfolio.risk_model)

    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=my_portfolio.diversification)
    
    # Add constraints for min and max weights if provided
    if my_portfolio.min_weights is not None:
        ef.add_constraint(lambda x: x >= my_portfolio.min_weights)
    if my_portfolio.max_weights is not None:
        ef.add_constraint(lambda x: x <= my_portfolio.max_weights)
    
    # ef.efficient_risk(vol_max)
    # weights = ef.clean_weights()
    
    weights = None
    try:
        # Attempt to optimize for the given maximum volatility
        ef.efficient_risk(vol_max)
        weights = ef.clean_weights()
    except ValueError as e:
        # If the error is about the minimum volatility being higher, use the minimum volatility
        if "higher target_volatility" in str(e):
            min_volatility = float(str(e).split(" ")[4].strip("."))
            print(f"Target volatility is too low, using minimum achievable volatility: {min_volatility}")
            ef.efficient_risk(min_volatility)
            weights = ef.clean_weights()
        else:
            raise e  # Re-raise any other errors

    wts = weights.items()

    result = []
    for val in wts:
        a, b = map(list, zip(*[val]))
        result.append(b)

    if perf is True:
        ef.portfolio_performance(verbose=True)

    return flatten(result)


def min_var(my_portfolio, perf=True) -> list:
    if not my_portfolio.data.empty:
        prices = my_portfolio.data
    else:
        pass
        
        # ohlc = yf.download(
            # my_portfolio.portfolio,
            # start=my_portfolio.start_date,
            # end=my_portfolio.end_date,
            # progress=False,
        # )
        # prices = ohlc["Adj Close"].dropna(how="all")
        # prices = prices.filter(my_portfolio.portfolio)

    if my_portfolio.expected_returns == None:
        my_portfolio.expected_returns = 'capm_return'
    if my_portfolio.risk_model == None:
            my_portfolio.risk_model = 'ledoit_wolf'

    mu = expected_returns.return_model(prices, method=my_portfolio.expected_returns)
    S = risk_models.risk_matrix(prices, method=my_portfolio.risk_model)

    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=my_portfolio.diversification)
    if my_portfolio.min_weights is not None:
        ef.add_constraint(lambda x: x >= my_portfolio.min_weights)
    if my_portfolio.max_weights is not None:
        ef.add_constraint(lambda x: x <= my_portfolio.max_weights)
    ef.min_volatility()
    weights = ef.clean_weights()

    wts = weights.items()

    result = []
    for val in wts:
        a, b = map(list, zip(*[val]))
        result.append(b)

    if perf is True:
        ef.portfolio_performance(verbose=True)

    return flatten(result)


def optimize_portfolio(my_portfolio, vol_max=25, pie_size=5, font_size=14):
    if my_portfolio.optimizer == None:
        raise Exception("You didn't define any optimizer in your portfolio!")
    
    # returns1 = get_returns(
        # my_portfolio.portfolio,
        # equal_weighting(my_portfolio),
        # start_date=my_portfolio.start_date,
        # end_date=my_portfolio.end_date,
    # )
    
    returns1 = None
    if not my_portfolio.data.empty:
        returns1 = get_returns_from_data(
            my_portfolio.data, 
            equal_weighting(my_portfolio), 
            my_portfolio.portfolio)
    else:
        returns1 = get_returns(
            my_portfolio.portfolio,
            equal_weighting(my_portfolio),
            start_date=my_portfolio.start_date,
            end_date=my_portfolio.end_date,
        )
    
    creturns1 = (returns1 + 1).cumprod()

    port = copy.deepcopy(my_portfolio.portfolio)

    wts = [1.0 / len(my_portfolio.portfolio)] * len(my_portfolio.portfolio)

    optimizers = {
        "EF": efficient_frontier,
        "MEANVAR": mean_var,
        "HRP": hrp,
        "MINVAR": min_var,
    }
    
    if my_portfolio.optimizer in optimizers.keys():
        if my_portfolio.optimizer == "MEANVAR":
            wts = optimizers.get(my_portfolio.optimizer)(my_portfolio, my_portfolio.max_vol)
        else:
            wts = optimizers.get(my_portfolio.optimizer)(my_portfolio)
    else:
        opt = my_portfolio.optimizer
        my_portfolio.weights = opt()

    print("\n")

    indices = [i for i, x in enumerate(wts) if x == 0.0]

    # while 0.0 in wts:
    #     wts.remove(0.0)

    # for i in sorted(indices, reverse=True):
    #     del port[i]

    graph_opt(port, wts, pie_size, font_size)

    print("\n")

    # returns2 = get_returns(
        # port, wts, start_date=my_portfolio.start_date, end_date=my_portfolio.end_date
    # )
    
    returns2 = None
    if not my_portfolio.data.empty:
        returns2 = get_returns_from_data(
            my_portfolio.data, 
            wts, 
            port)
    else:
        returns2 = get_returns(
            port,
            wts,
            start_date=my_portfolio.start_date, 
            end_date=my_portfolio.end_date
        )
    
    creturns2 = (returns2 + 1).cumprod()

    plt.rcParams["font.size"] = 13
    plt.figure(figsize=(30, 10))
    plt.xlabel("Portfolio vs Benchmark")

    ax1 = creturns1.plot(color="blue", label="Without optimization")
    ax2 = creturns2.plot(color="red", label="With optimization")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    plt.legend(l1 + l2, loc=2)
    # plt.show()
    plt.savefig("equal-weights-vs-optimized-cum-returns.png")
    plt.close('all')


def check_schedule(rebalance) -> bool:
    valid_schedule = False
    if rebalance.lower() in rebalance_periods.keys():
        valid_schedule = True
    return valid_schedule


def valid_range(start_date, end_date, rebalance) -> tuple:

    # make the start date to a datetime
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")

    # custom dates don't need further chekings
    if type(rebalance) is list:
        return start_date, rebalance[-1]
    
    # make the end date to a datetime
    end_date = dt.datetime.strptime(str(end_date), "%Y-%m-%d")

    # gets the number of days
    days = (end_date - start_date).days

    # checking that date range covers rebalance period
    if rebalance in rebalance_periods.keys() and days <= (int(rebalance_periods[rebalance])):
        raise KeyError("Date Range does not encompass rebalancing interval")

    # we will needs these dates later on so we'll return them back
    return start_date, end_date


def get_date_range(start_date, end_date, rebalance) -> list:
    # this will keep track of the rebalancing dates and we want to start on the first date
    rebalance_dates = [start_date]
    input_date = start_date

    if rebalance in rebalance_periods.keys():
        # run for an arbitrarily large number we'll resolve this by breaking when we break the equality
        for i in range(1000):
            # increment the date based on the selected period
            input_date = input_date + dt.timedelta(days=rebalance_periods.get(rebalance))
            if input_date <= end_date:
                # append the new date if it is earlier or equal to the final date
                rebalance_dates.append(input_date)
            else:
                # break when the next rebalance date is later than our end date
                break

    # then we want to return those dates
    return rebalance_dates

def scale_weights_for_cash(weights, cash_ratio):

    # Sum the absolute values of all the weights
    total_weight = sum(abs(weight) for i, weight in enumerate(weights))

    if total_weight == 0:
        raise ValueError("Total weight cannot be zero.")
    weights = [w / total_weight for w in weights]

    # other_weights_sum = total_weight - abs(weights[0])
    # if other_weights_sum == 0:
    #     raise ValueError("Sum of other weights cannot be zero when scaling.")

    # Scale non-cash weights to fit the remaining allocation after cash
    scaled_weights = []
    for i, weight in enumerate(weights):
        if i == 0:
            scaled_weights.append(cash_ratio)
        else:
            # scaled_weights.append((1.0 - cash_ratio) * weight / other_weights_sum)
            if weight < 0:
                scaled_weights.append(0)
            else:
                scaled_weights.append(weight)

    # Ensure the scaled weights sum up to 1.0
    scaled_weights_total = sum(scaled_weights)

    other_weights_sum = scaled_weights_total - abs(scaled_weights[0])

    scaled_weights = weights
    if other_weights_sum == 0:
        print(f"weights (err): \n{weights}")
        # raise ValueError("Sum of other weights cannot be zero when scaling.")
    else:
        scaled_weights = [(1.0 - cash_ratio) * w / other_weights_sum for w in scaled_weights]
        scaled_weights[0] = cash_ratio # restore cash weight

    return scaled_weights

    # ORIG. CODE of mine
    # # Adjust the weights dynamically
    # current_cash_ratio = cash_ratio
    # other_weights_sum = sum(weights[1:])  # Exclude cash weight

    # # Scale the other weights to fit within the remaining allocation after cash
    # scaled_weights = [current_cash_ratio] + [(1.0 - current_cash_ratio) * w / other_weights_sum for w in weights[1:]]
    # return scaled_weights

def make_rebalance(
    start_date,
    end_date,
    optimize,
    portfolio_input,
    rebalance,
    allocation,
    vol_max,
    div,
    min,
    max,
    expected_returns,
    risk_model,
    df,
    data_all,
    cash_ratio_is_dynamic,
    cash_ratio,  # Add cash_ratio parameter
    risk_score_data,
    risk_acceptance,
    
) -> pd.DataFrame:
    sdate = str(start_date)[:10]
    if rebalance[0] != sdate:

        # makes sure the start date matches the first element of the list of custom rebalance dates
        if type(rebalance) is list:
            raise KeyError("the rebalance dates and start date doesn't match")

        # makes sure that the value passed through for rebalancing is a valid one
        valid_schedule = check_schedule(rebalance)

        if valid_schedule is False:
            raise KeyError("Not an accepted rebalancing schedule")

    # this checks to make sure that the date range given works for the rebalancing
    start_date, end_date = valid_range(start_date, end_date, rebalance)

    # this function will get us the specific dates
    if rebalance[0] != sdate:
        dates = get_date_range(start_date, end_date, rebalance)
    else:
        dates = rebalance

    # we are going to make columns with the end date and the weights
    columns = ["end_date"] + portfolio_input

    # then make a dataframe with the index being the tickers
    output_df = pd.DataFrame(index=portfolio_input)

    # Initialize previous period dates
    last_start_date = None
    last_end_date = None

    the_dynamic_cash_ratio = cash_ratio
    for i in range(len(dates) - 1):
        # Current period dates
        current_start_date = dates[i]
        current_end_date = dates[i + 1]

        if cash_ratio_is_dynamic and risk_score_data is not None:
            # the_dynamic_cash_ratio = random.uniform(0.1, 0.6)
            # the_dynamic_cash_ratio = calculate_cash_ratio(dates[i+1], 0.9, risk_score_data)
            the_dynamic_cash_ratio = calculate_cash_ratio(
                risk_acceptance = risk_acceptance, 
                risk_score_data = risk_score_data, 
                start_date = current_start_date, 
                end_date = current_end_date, 
                last_start_date = last_start_date, 
                last_end_date = last_end_date
            )
        else:
            the_dynamic_cash_ratio = cash_ratio

        # Update the last period dates for the next iteration
        last_start_date = current_start_date
        last_end_date = current_end_date
        
        try:
            portfolio = Engine(
                start_date=dates[0],
                end_date=dates[i + 1],
                portfolio=portfolio_input,
                weights=allocation,
                optimizer="{}".format(optimize),
                max_vol=vol_max,
                diversification=div,
                min_weights=min,
                max_weights=max,
                expected_returns=expected_returns,
                risk_model=risk_model,
                data=df, # Ensure custom data is passed here
                data_all=data_all,
                cash_ratio_is_dynamic = cash_ratio_is_dynamic,
                cash_ratio=the_dynamic_cash_ratio,
                risk_score_data = risk_score_data,
                risk_acceptance = risk_acceptance,
            )

        except TypeError:
            portfolio = Engine(
                start_date=dates[0],
                end_date=dates[i + 1],
                portfolio=portfolio_input,
                weights=allocation,
                optimizer=optimize,
                max_vol=vol_max,
                diversification=div,
                min_weights=min,
                max_weights=max,
                expected_returns=expected_returns,
                risk_model=risk_model,
                data=df, # Ensure custom data is passed here
                data_all=data_all,
                cash_ratio_is_dynamic = cash_ratio_is_dynamic,
                cash_ratio=the_dynamic_cash_ratio,
                risk_score_data = risk_score_data,
                risk_acceptance = risk_acceptance,
            )
        
        # Adjust the weights dynamically
        portfolio.weights = scale_weights_for_cash(portfolio.weights, portfolio.cash_ratio)

        # Orig. code
        output_df["{}".format(dates[i + 1])] = portfolio.weights
        
        # # Mod: might not work: ddjust weights to match the index of output_df
        # cleaned_weights = pd.Series(portfolio.weights, index=portfolio_input)
        # cleaned_weights = cleaned_weights.reindex(output_df.index).fillna(0)
        # output_df["{}".format(dates[i + 1])] = cleaned_weights

    # we have to run it one more time to get what the optimization is for up to today's date
    try:
        portfolio = Engine(
            start_date=dates[0],
            portfolio=portfolio_input,
            weights=allocation,
            optimizer="{}".format(optimize),
            max_vol=vol_max,
            diversification=div,
            min_weights=min,
            max_weights=max,
            expected_returns=expected_returns,
            risk_model=risk_model,
            data=df, # Ensure custom data is passed here
            data_all=data_all,
            cash_ratio_is_dynamic = cash_ratio_is_dynamic,
            cash_ratio=the_dynamic_cash_ratio,
            risk_score_data = risk_score_data,
            risk_acceptance = risk_acceptance,
        )

    except TypeError:
        portfolio = Engine(
            start_date=dates[0],
            portfolio=portfolio_input,
            weights=allocation,
            optimizer=optimize,
            max_vol=vol_max,
            diversification=div,
            min_weights=min,
            max_weights=max,
            expected_returns=expected_returns,
            risk_model=risk_model,
            data=df, # Ensure custom data is passed here
            data_all=data_all,
            cash_ratio_is_dynamic = cash_ratio_is_dynamic,
            cash_ratio=the_dynamic_cash_ratio,
            risk_score_data = risk_score_data,
            risk_acceptance = risk_acceptance,
        )
    
    # Adjust the weights dynamically
    portfolio.weights = scale_weights_for_cash(portfolio.weights, portfolio.cash_ratio)

    output_df["{}".format(TODAY)] = portfolio.weights

    make_rebalance.output = output_df
    # print("!!!FINAL Rebalance schedule: ")
    # print(output_df)
    return output_df
