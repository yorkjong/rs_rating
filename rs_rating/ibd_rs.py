"""
ibd.py - Stock Analysis and Ranking Module

This module provides tools for analyzing and ranking stocks based on their
relative strength compared to a benchmark index, inspired by the Investor's
Business Daily (IBD) methodology.

Key Features:
-------------
- Relative strength calculation
- Stock and industry ranking generation
- Percentile-based filtering of rankings

Usage:
------

::

    import ibd

    # Generate rankings for a list of stocks
    tickers = ['MSFT', 'NVDA', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    stock_rankings, industry_rankings = ibd.rankings(tickers)

    # Calculate relative strength for a single stock
    stock_rs = ibd.relative_strength(stock_closes, index_closes)

    # Filter rankings based on a minimum percentile
    min_percentile = 80
    top_stocks = stock_rankings[stock_rankings["Percentile"] >= min_percentile]

See Also:
---------
- `RS Rating — Indicator by Fred6724 — tradingview
  <https://www.tradingview.com/script/pziQwiT2/>`_
- `Relative Strength (IBD Style) — Indicator by Skyte — TradingView
  <https://www.tradingview.com/script/SHE1xOMC-Relative-Strength-IBD-Style/>`_

  - `relative-strength/rs_ranking.py at main · skyte/relative-strength
    <https://github.com/skyte/relative-strength/blob/main/rs_ranking.py>`_

- `Exclusive IBD Ratings | Stock News & Stock Market Analysis - IBD
  <https://www.investors.com/ibd-university/
  find-evaluate-stocks/exclusive-ratings/>`_
"""
__version__ = "4.5"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/05 (initial version) ~ 2024/10/04 (last revision)"

__all__ = [
    'relative_strength',
    'relative_strength_3m',
    'ranking',
    'rankings',
    'ma_window_size',
]

import os

import numpy as np
import pandas as pd
import yfinance as yf

from . import yf_utils as yfu


#------------------------------------------------------------------------------
# IBD RS (Relative Strength) Rating
#------------------------------------------------------------------------------

def relative_strength(closes, closes_ref, interval='1d'):
    """
    Calculate the relative strength of a stock compared to a reference index.

    Relative Strength (RS) is a metric used to evaluate the performance of a
    stock relative to a benchmark index. A higher RS rating indicates that the
    stock has outperformed the index, while a lower RS rating suggests
    underperformance.

    This function calculates the RS rating by comparing the quarter-weighted
    growth of the stock's closing prices to the quarter-weighted growth of
    the reference index's closing prices over the past year. The formula is as
    follows:

    ::

        PR = current/previous = ((current - previous) / previous) + 1
           = return + 1
        relative_rate = PR_stock / PR_index
        relative_strength = relative_rate * 100

    Here PR means 'price ratio' or 'price relative'

    The quarter-weighted growth is calculated using the `weighted_return`
    function.

    Parameters
    ----------
    closes : pd.Series
        Closing prices of the stock.

    closes_ref : pd.Series
        Closing prices of the reference index.

    interval : str, optional
        The frequency of the data points. Must be one of '1d' for daily data,
        '1wk' for weekly data, or '1mo' for monthly data. Defaults to '1d'.

    Returns
    -------
    pd.Series
        Relative strength values for the stock.

    Example
    -------
    >>> stock_closes = pd.Series([100, 102, 105, 103, 107])
    >>> index_closes = pd.Series([1000, 1010, 1015, 1005, 1020])
    >>> rs = relative_strength(stock_closes, index_closes)

    """
    ret_stock = weighted_return(closes, interval)
    ret_ref = weighted_return(closes_ref, interval)
    rs = (1 + ret_stock) / (1 + ret_ref) * 100
    return round(rs, 2)


def weighted_return(closes, interval):
    """
    Calculate the performance of the last year, with the most recent quarter
    weighted double.

    This function calculates returns for each of the last four quarters and
    applies a weighting scheme that emphasizes recent performance. The most
    recent quarter is given a weight of 40%, while each of the three preceding
    quarters are given a weight of 20%.

    Here is the formula for calculating the return:

        RS Return = 40% * P3 + 20% * P6 + 20% * P9 + 20% * P12
        With
        P3 = Performance over the last quarter (3 months)
        P6 = Performance over the last two quarters (6 months)
        P9 = Performance over the last three quarters (9 months)
        P12 = Performance over the last four quarters (12 months)

    Parameters
    ----------
    closes: pd.Series
        Closing prices of the stock/index.
    interval: str, optional
        The frequency of the data points. Must be one of '1d' for daily
        data, '1wk' for weekly data, or '1mo' for monthly data.

    Returns
    -------
    pd.Series: Performance values of the stock/index.

    Example
    -------
    >>> closes = pd.Series([100, 102, 105, 103, 107, 110, 112])
    >>> weighted_perf = weighted_return(closes)
    """
    # Calculate performances over the last quarters
    p1 = quarters_return(closes, 1, interval) # over the last quarter
    p2 = quarters_return(closes, 2, interval) # over the last two quarters
    p3 = quarters_return(closes, 3, interval) # over the last three quarters
    p4 = quarters_return(closes, 4, interval) # over the last four quarters
    return (2 * p1 + p2 + p3 + p4) / 5


def quarters_return(closes, n, interval):
    """
    Calculate the return (percentage change) over the last n quarters.

    This function uses 63 trading days (252 / 4) as an approximation for
    one quarter. This is based on the common assumption of 252 trading
    days in a year.

    Parameters
    ----------
    closes : pd.Series
        Closing prices of the stock or index.

    n : int
        Number of quarters to look back.

    interval : str, optional
        The frequency of the data points. Must be one of '1d' for daily data,
        '1wk' for weekly data, or '1mo' for monthly data.

    Returns
    -------
    pd.Series
        The return (percentage change) over the last n quarters.

    Example
    -------
    >>> closes = pd.Series([100, 102, 105, 103, 107, 110, 112])
    >>> quarterly_return = quarters_return(closes, 1)
    """
    quarter = {
        '1d': 252//4,   # 252 trading days in a year
        '1wk': 52//4,   # 52 weeks in a year
        '1mo': 12//4,   # 12 months in a year
    }[interval]
    periods = min(len(closes) - 1, quarter * n)

    ret = closes.ffill().pct_change(periods=periods, fill_method=None)
    return ret.fillna(0)


#------------------------------------------------------------------------------
# IBD's 3-Month Relative Strength
#------------------------------------------------------------------------------

def relative_strength_3m(closes, closes_ref, interval='1d'):
    """
    Calculate the 3-Month Relative Strength of a stock compared to a reference
    index, based on price performance (returns).

    The 3-Month Relative Strength Rating (RS Rating) measures the stock's
    price performance against a benchmark index over a recent three-month
    period. This rating is designed to help investors quickly gauge the
    strength of a stock's performance relative to the market.

    Parameters
    ----------
    closes : pd.Series
        Closing prices of the stock.

    closes_ref : pd.Series
        Closing prices of the reference index.

    interval : str, optional
        The frequency of the data points. Must be one of '1d' for daily data,
        '1wk' for weekly data, or '1mo' for monthly data. Defaults to '1d'.

    Returns
    -------
    pd.Series
        3-Month relative strength values for the stock, rounded to two decimal
        places. The values represent the stock's performance relative to the
        benchmark index, with 100 indicating parity.
    """
    # Determine the number of trading days for the specified interval
    span = {
        '1d': 252 // 4,  # a 3-month period based on 252 trading days in a year
        '1wk': 52 // 4,  # 13 weeks (3 months) for weekly data
        '1mo': 12 // 4,  # 3 months for monthly data
    }[interval]

    # Calculate daily returns for the stock and reference index
    returns_stock = closes.ffill().pct_change(fill_method=None)
    returns_ref = closes_ref.ffill().pct_change(fill_method=None)

    # Calculate the Exponential Moving Average (EMA) of the returns
    ema_returns_stock = returns_stock.ewm(span=span, adjust=False).mean()
    ema_returns_ref = returns_ref.ewm(span=span, adjust=False).mean()

    # Calculate the relative strength (RS). Add 1 to avoid negative values
    rs = (ema_returns_stock + 1) / (ema_returns_ref + 1) * 100

    return rs.round(2)  # Return the RS values rounded to two decimal places


#------------------------------------------------------------------------------
# IBD RS Ranking (with RS rating)
#------------------------------------------------------------------------------

def ranking(tickers, ticker_ref='^GSPC', period='2y', interval='1d',
            rs_period='12mo'):
    """
    Rank stocks based on their IBD Relative Strength against an index
    benchmark.

    Parameters
    ----------
    tickers : list of str
        List of stock tickers to rank.

    ticker_ref : str, optional
        Ticker symbol of the benchmark. Default to '^GSPC' (S&P 500)

    period : str, optional
        Period for historical data ('6mo', '1y', '2y', '5y', 'ytd', 'max').
        Default to '2y' (two years).

    interval : str, optional
        Interval for historical data ('1d', '1wk', '1mo').
        Default to '1wk' (one week).

    rs_period : str, optional
        Specify the period for Relative Strength calculation ('12mo' or '3mo').
        Default to '12mo'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the ranked stocks.

    Example
    -------
    >>> tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    >>> stock_rankings  = ranking(tickers)
    >>> print(stock_rankings.head())
    """
    # Select the appropriate relative strength function based on the rs_period
    rs_func = {
        '3mo': relative_strength_3m,
        '12mo': relative_strength,
    }[rs_period]

    # Fetch data for stock and index
    df = yf.download([ticker_ref] + tickers, period=period, interval=interval)
    df = df.xs('Close', level='Price', axis=1)

    # Fetch info for stocks
    info = yfu.download_tickers_info(tickers, ['sector', 'industry'])

    rs_data = []
    for ticker in tickers:
        rs = rs_func(df[ticker], df[ticker_ref], interval)
        end_date = rs.index[-1]

        # Calculate max values for the specified time periods
        one_week_ago = end_date - pd.DateOffset(weeks=1)
        one_month_ago = end_date - pd.DateOffset(months=1)
        three_months_ago = end_date - pd.DateOffset(months=3)
        six_months_ago = end_date - pd.DateOffset(months=6)

        rs_data.append({
            'Ticker': ticker,
            'Price': df[ticker].asof(end_date).round(2),
            'Sector': info[ticker]['sector'],
            'Industry': info[ticker]['industry'],
            'Relative Strength': rs.asof(end_date),
            '1wk..end': rs.loc[one_week_ago:end_date].max(),
            '1mo..1wk': rs.loc[one_month_ago:one_week_ago].max(),
            '3mo..1mo': rs.loc[three_months_ago:one_month_ago].max(),
            '6mo..3mo': rs.loc[six_months_ago:three_months_ago].max(),
        })

    # Create DataFrame from RS data
    ranking_df = pd.DataFrame(rs_data)

    # Rank based on Relative Strength
    rank_columns = ['Rank (%)',
                    '1mo..1wk (%)', '3mo..1mo (%)',  '6mo..3mo (%)']
    rs_columns = ['Relative Strength',
                  '1mo..1wk', '3mo..1mo', '6mo..3mo']
    for rank_col, rs_col in zip(rank_columns, rs_columns):
        rank_pct = ranking_df[rs_col].rank(pct=True)
        ranking_df[rank_col] = (rank_pct * 100).round(2)

    # Sort by current rank
    ranking_df = ranking_df.sort_values(by='Rank (%)', ascending=False)

    # Calculate average RS for each industry
    industry_rs = ranking_df.groupby('Industry')[
        'Relative Strength'].mean().round(2).reset_index()
    industry_rs.columns = ['Industry', 'Industry RS']

    # Rank industries based on average Relative Strength
    rank_pct = industry_rs['Industry RS'].rank(pct=True)
    industry_rs['Industry Rank (%)'] = (rank_pct * 100).round(2)

    # Merge industry rankings back into stock DataFrame
    ranking_df = pd.merge(ranking_df, industry_rs, on='Industry', how='left')

    return ranking_df


#------------------------------------------------------------------------------
# IBD RS Rankings (with RS rating)
#------------------------------------------------------------------------------

def rankings(tickers, ticker_ref='^GSPC', period='2y', interval='1d',
             percentile_method='rank', rs_period='12mo'):
    """
    Analyze stocks and generate ranking tables for individual stocks and
    industries.

    This function calculates relative strength (RS) for the given stocks compared
    to a reference index, and then ranks both individual stocks and industries
    based on their RS values. It provides historical RS data and percentile
    rankings.

    Parameters
    ----------
    tickers : List[str]
        A list of stock tickers to analyze.

    ticker_ref : str, optional
        The ticker symbol for the reference index. Defaults to '^GSPC' (S&P 500).

    period : str, optional
        The period for which to fetch historical data. Defaults to '2y' (two years).

    interval : str, optional
        The frequency of the data points. Must be one of '1d' for daily data,
        '1wk' for weekly data, or '1mo' for monthly data. Defaults to '1d'.

    percentile_method : str, optional
        Method to calculate percentiles. Either 'rank' or 'qcut'. Defaults to 'rank'.

    rs_period : str, optional
        Specify the period for Relative Strength calculation ('12mo' or '3mo').
        Defaults to '12mo'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of two Pandas DataFrames:

        1. Stock Rankings DataFrame:
            - Columns: Rank, Ticker, Price, Sector, Industry, RS (current),
              RS (1 month ago), RS (3 months ago), RS (6 months ago),
              Percentile (current), Percentile (1 month ago),
              Percentile (3 months ago), Percentile (6 months ago)

        2. Industry Rankings DataFrame:
            - Columns: Rank, Industry, Sector, RS (current),
              RS (1 month ago), RS (3 months ago), RS (6 months ago),
              Tickers (list of tickers in the industry),
              Percentile (current), Percentile (1 month ago),
              Percentile (3 months ago), Percentile (6 months ago)
    """
    # Select the appropriate relative strength function based on the rs_period
    rs_func = {
        '3mo': relative_strength_3m,
        '12mo': relative_strength,
    }[rs_period]

    # Batch download stock data
    df = yf.download([ticker_ref] + tickers, period=period, interval=interval)
    df = df.xs('Close', level='Price', axis=1)

    # Batch download stock info
    info = yfu.download_tickers_info(tickers, ['sector', 'industry'])
    # Calculate RS values for all stocks
    rs_data = []
    for ticker in tickers:
        rs_series = rs_func(df[ticker], df[ticker_ref], interval)
        end_date = rs_series.index[-1]

        rs_data.append({
            'Ticker': ticker,
            'Price': df[ticker].asof(end_date).round(2),
            'Sector': info[ticker]['sector'],
            'Industry': info[ticker]['industry'],
            'RS': rs_series.asof(end_date),
            '1M': rs_series.asof(end_date - pd.DateOffset(months=1)),
            '3M': rs_series.asof(end_date - pd.DateOffset(months=3)),
            '6M': rs_series.asof(end_date - pd.DateOffset(months=6))
        })

    # Create DataFrame from RS data
    stock_df = pd.DataFrame(rs_data)

    # Calculate percentiles for RS values
    def calc_percentile(series):
        if percentile_method == 'rank':
            return series.rank(pct=True).mul(100).round(2)
        elif percentile_method == 'qcut':
            return pd.qcut(series, 100, labels=False, duplicates='drop')
        else:
            raise ValueError(
                "percentile_method must be either 'rank' or 'qcut'")

    for col in ['RS', '1M', '3M', '6M']:
        stock_df[f'Percentile ({col})'] = calc_percentile(stock_df[col])

    # Sort stocks
    stock_df = stock_df.sort_values('RS',
                                    ascending=False).reset_index(drop=True)

    def get_sorted_tickers(tickers):
        return ','.join(
            sorted(tickers,
                   key=lambda t:
                        stock_df.loc[stock_df['Ticker'] == t, 'RS'].values[0],
                   reverse=True)
        )

    # Calculate industry rankings
    industry_df = stock_df.groupby('Industry').agg({
        'Sector': 'first',
        'RS': lambda x: x.mean().round(2),
        '1M': lambda x: x.mean().round(2),
        '3M': lambda x: x.mean().round(2),
        '6M': lambda x: x.mean().round(2),
        'Ticker': get_sorted_tickers
    }).reset_index()

    # Calculate percentiles for industry RS values
    for col in ['RS', '1M', '3M', '6M']:
        industry_df[f'Percentile ({col})'] = calc_percentile(industry_df[col])

    # Sort industries
    industry_df = industry_df.sort_values(
        'RS', ascending=False).reset_index(drop=True)

    # Rename columns for clarity
    stock_df = stock_df.rename(columns={
        'RS': 'Relative Strength',
        '1M': '1 Month Ago',
        '3M': '3 Months Ago',
        '6M': '6 Months Ago',
        'Percentile (RS)': 'Percentile'
    })
    industry_df = industry_df.rename(columns={
        'RS': 'Relative Strength',
        '1M': '1 Month Ago',
        '3M': '3 Months Ago',
        '6M': '6 Months Ago',
        'Ticker': 'Tickers',
        'Percentile (RS)': 'Percentile'
    })

    return stock_df, industry_df


#------------------------------------------------------------------------------
# Misc Help Functions
#------------------------------------------------------------------------------

def ma_window_size(interval, days):
    """
    Calculate moving average window size based on IBD (Investor's Business
    Daily) convention.

    This function adjusts the window size for weekly data to maintain
    consistency with daily calculations.

    Parameters
    ----------
    interval : str
        The data interval. Must be either '1d' for daily or '1wk' for weekly.

    days : int
        Number of calendar days for the desired moving average period.

    Returns
    -------
    int
        Calculated window size (number of data points) for the moving average.

    Raises
    ------
    ValueError
        If an unsupported interval is provided (not '1d' or '1wk').

    Examples
    --------
    >>> ma_window_size('1d', 50)
    50
    >>> ma_window_size('1wk', 50)
    10
    """
    if interval == '1d':
        return days
    elif interval == '1wk':
        return days // 5  # 1 week = 5 trading days
    else:
        raise ValueError("Unsupported interval")


#------------------------------------------------------------------------------
# Unit Test
#------------------------------------------------------------------------------

def test_ranking(period='2y', rs_period='12mo', out_dir='out'):
    import os
    from datetime import datetime
    from .stock_indices import get_tickers

    #code = 'SPX+DJIA+NDX+SOX'
    code = 'SPX'
    tickers = get_tickers(code)
    remove_tickers = ['HBAN', 'SW', 'BRK.B', 'VLTO', 'ARM', 'SOLV', 'GEV', 'BF.B']
    tickers = [t for t in tickers if t not in remove_tickers]

    rank = ranking(tickers, period=period, interval='1d', rs_period=rs_period)
    print(rank.head(10))

    # Save to CSV
    print("\n\n***")
    os.makedirs(out_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    filename = f'{code}_stocks_{period}_ibd{rs_period}_{today}.csv'
    rank.to_csv(os.path.join(out_dir, filename), index=False)
    print(f'Your "{filename}" is in the "{out_dir}" folder.')
    print("***\n")


def test_rankings(min_percentile=80, percentile_method='qcut',
                  rs_period='12mo', out_dir='out'):
    '''
    Parameters
    ----------
    min_percentile : int, optional
        The minimum percentile for a stock to be included in the rankings.
        Defaults to 80.

    out_dir : str, optional
        The output directory to store CSV tables. Defaults to 'out'.
    '''
    import stock_indices as si

    tickers = si.get_tickers('SOX')
    #tickers = ['2330.TW', '2401.TW']
    rank_stock, rank_indust = rankings(tickers, interval='1d',
                                       percentile_method=percentile_method,
                                       rs_period=rs_period)

    if rank_stock.empty or rank_indust.empty:
        print("Not enough data to generate rankings.")
        return

    print('Stock Rankings:')
    print(rank_stock[rank_stock["Percentile"] >= min_percentile])

    print('\n\nIndustry Rankings:')
    print(rank_indust)

    # Save to CSV
    print("\n\n***")
    os.makedirs(out_dir, exist_ok=True)
    for table, kind in zip([rank_stock, rank_indust],
                           ['stocks', 'industries']):
        filename = f'rs_{kind}.csv'
        table.to_csv(os.path.join(out_dir, filename), index=False)
        print(f'Your "{filename}" is in the "{out_dir}" folder.')
    print("***\n")


if __name__ == "__main__":
    import time

    start_time = time.time()
    test_ranking(rs_period='3mo')
    test_rankings(percentile_method='qcut', rs_period='3mo')
    print(f"Execution time: {time.time() - start_time:.4f} seconds")

