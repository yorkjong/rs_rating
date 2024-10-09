"""
IBD RS (Relative Strength) Rating Module
----------------------------------------

This module provides tools for analyzing and ranking stocks based on their
relative strength compared to a benchmark index, inspired by the Investor's
Business Daily (IBD) methodology.

Key Features:
~~~~~~~~~~~~~
- Relative strength calculation
- Stock and industry ranking generation
- Rating-based filtering of rankings

Usage:
~~~~~~
::

    import ibd_res as ibd

    # Generate rankings for a list of stocks
    tickers = ['MSFT', 'NVDA', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    stock_rankings, industry_rankings = ibd.rankings(tickers)

    # Calculate relative strength for a single stock
    stock_rs = ibd.relative_strength(stock_closes, index_closes)

    # Filter rankings based on a minimum rating
    min_rating = 80
    top_stocks = stock_rankings[stock_rankings["Rating"] >= min_rating]

See Also:
~~~~~~~~~
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
__version__ = "5.1"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/05 (initial version) ~ 2024/10/10 (last revision)"

__all__ = [
    'relative_strength',
    'relative_strength_3m',
    'rankings',
]

import numpy as np
import pandas as pd
import yfinance as yf

from . import yf_utils as yfu
from .ranking_utils import append_ratings, groupby_industry


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

        growth = (current - previous) / previous
        gf = current/previous = growth + 1
        relative_rate = gf_stock / gf_index
        relative_strength = relative_rate * 100

    Here gf means "growth factor", i.e., "price ratio"

    The quarter-weighted growth is calculated using the `weighted_growth`
    function.

    Parameters
    ----------
    closes: pd.Series
        Closing prices of the stock.

    closes_ref: pd.Series
        Closing prices of the reference index.

    interval: str, optional
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
    ret_stock = weighted_growth(closes, interval)
    ret_ref = weighted_growth(closes_ref, interval)
    rs = (1 + ret_stock) / (1 + ret_ref) * 100
    return round(rs, 2)


def weighted_growth(closes, interval):
    """
    Calculate the performance of the last year, with the most recent quarter
    weighted double.

    This function calculates growths (returns) for each of the last four
    quarters and applies a weighting scheme that emphasizes recent performance.
    The most recent quarter is given a weight of 40%, while each of the three
    preceding quarters are given a weight of 20%.

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
    >>> weighted_perf = weighted_growth(closes)
    """
    # Calculate performances over the last quarters
    p1 = quarters_growth(closes, 1, interval) # over the last quarter
    p2 = quarters_growth(closes, 2, interval) # over the last two quarters
    p3 = quarters_growth(closes, 3, interval) # over the last three quarters
    p4 = quarters_growth(closes, 4, interval) # over the last four quarters
    return (2 * p1 + p2 + p3 + p4) / 5


def quarters_growth(closes, n, interval):
    """
    Calculate the return (percentage change) over the last n quarters.

    This function uses 63 trading days (252 / 4) as an approximation for
    one quarter. This is based on the common assumption of 252 trading
    days in a year.

    Parameters
    ----------
    closes: pd.Series
        Closing prices of the stock or index.

    n: int
        Number of quarters to look back.

    interval: str, optional
        The frequency of the data points. Must be one of '1d' for daily data,
        '1wk' for weekly data, or '1mo' for monthly data.

    Returns
    -------
    pd.Series
        The return (percentage change) over the last n quarters.

    Example
    -------
    >>> closes = pd.Series([100, 102, 105, 103, 107, 110, 112])
    >>> quarterly_growth = quarters_growth(closes, 1)
    """
    quarter = {
        '1d': 252//4,   # 252 trading days in a year
        '1wk': 52//4,   # 52 weeks in a year
        '1mo': 12//4,   # 12 months in a year
    }[interval]
    periods = min(len(closes) - 1, quarter * n)

    growth = closes.ffill().pct_change(periods=periods, fill_method=None)
    return growth.fillna(0)


#------------------------------------------------------------------------------
# IBD's 3-Month Relative Strength
#------------------------------------------------------------------------------

def relative_strength_3m(closes, closes_ref, interval='1d'):
    """
    Calculate the 3-Month Relative Strength of a stock compared to a reference
    index, based on price performance (growths).

    The 3-Month Relative Strength Rating (RS Rating) measures the stock's
    price performance against a benchmark index over a recent three-month
    period. This rating is designed to help investors quickly gauge the
    strength of a stock's performance relative to the market.

    Parameters
    ----------
    closes: pd.Series
        Closing prices of the stock.

    closes_ref: pd.Series
        Closing prices of the reference index.

    interval: str, optional
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

    # Calculate daily growths (returns) for the stock and reference index
    growth_stock = closes.pct_change(fill_method=None).fillna(0)
    growth_ref = closes_ref.pct_change(fill_method=None).fillna(0)

    # Calculate the Exponential Moving Average (EMA) of the growths
    ema_growth_stock = growth_stock.ewm(span=span, adjust=False).mean()
    ema_growth_ref = growth_ref.ewm(span=span, adjust=False).mean()

    # Calculate the cumulative sums
    cum_sotck = ema_growth_stock.rolling(window=span, min_periods=1).sum()
    cum_ref = ema_growth_ref.rolling(window=span, min_periods=1).sum()

    # Calculate the relative strength (RS).
    rs = (cum_sotck + 1) / (cum_ref + 1).abs() * 100

    return rs.round(2)  # Return the RS values rounded to two decimal places


#------------------------------------------------------------------------------
# IBD RS Rankings (with RS rating)
#------------------------------------------------------------------------------

def build_stock_rs_df(tickers, ticker_ref='^GSPC', period='2y', interval='1d',
                      rs_window='12mo'):
    """
    Calculates the Relative Strength (RS) of a list of stock tickers compared
    to a reference index and returns a DataFrame of stock rankings.

    Parameters
    ----------
    tickers : list
        List of stock tickers to analyze.

    ticker_ref : str, optional
        The reference index ticker symbol. Default is '^GSPC' (S&P 500).

    period : str, optional
        The duration for which historical stock data is fetched. Default is
        '2y' (two years).

    interval : str, optional
        The time interval between data points. Can be '1d' (daily), '1wk'
        (weekly), or '1mo' (monthly). Default is '1d'.

    rs_window : str, optional
        The period for calculating RS. Either '3mo' or '12mo'. Default is
        '12mo'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing stock information and RS values:
        - 'Ticker': Stock ticker
        - 'Price': Latest stock price
        - 'Sector': Sector of the stock
        - 'Industry': Industry of the stock
        - 'RS': Current RS value
        - '1 Month Ago': RS value one month ago
        - '3 Months Ago': RS value three months ago
        - '6 Months Ago': RS value six months ago
    """
    # Select the appropriate relative strength function based on the rs_window
    rs_func = {
        '3mo': relative_strength_3m,
        '12mo': relative_strength,
    }[rs_window]

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
            '1 Month Ago': rs_series.asof(end_date - pd.DateOffset(months=1)),
            '3 Months Ago': rs_series.asof(end_date - pd.DateOffset(months=3)),
            '6 Months Ago': rs_series.asof(end_date - pd.DateOffset(months=6)),
        })

    # Create DataFrame from RS data
    stock_df = pd.DataFrame(rs_data)

    return stock_df


def rankings(tickers, ticker_ref='^GSPC', period='2y', interval='1d',
             rating_method='rank', rs_window='12mo'):
    """
    Generates stock and industry ranking tables based on Relative Strength (RS)
    compared to a reference index.

    Parameters
    ----------
    tickers : list
        List of stock tickers to analyze.

    ticker_ref : str, optional
        The reference index ticker symbol. Default is '^GSPC' (S&P 500).

    period : str, optional
        Duration for fetching historical stock data. Default is '2y' (two
        years).

    interval : str, optional
        Time interval between data points. Can be '1d' (daily), '1wk'
        (weekly), or '1mo' (monthly). Default is '1d'.

    rating_method : str, optional
        Method for calculating stock ratings. Can be either 'rank' or 'qcut'.
        Default is 'rank'.

    rs_window : str, optional
        Period for calculating RS. Either '3mo' or '12mo'. Default is '12mo'.

    Returns
    -------
    tuple of pd.DataFrame
        1. Stock Rankings DataFrame:
            - 'Rank': Stock rank based on RS
            - 'Ticker': Stock ticker
            - 'Price': Latest stock price
            - 'Sector': Sector of the stock
            - 'Industry': Industry of the stock
            - 'RS': Current RS value
            - '1 Month Ago': RS value one month ago
            - '3 Months Ago': RS value three months ago
            - '6 Months Ago': RS value six months ago
            - 'Rating': Current rating
            - 'Rating (1 Month Ago)': Rating one month ago
            - 'Rating (3 Months Ago)': Rating three months ago
            - 'Rating (6 Months Ago)': Rating six months ago

        2. Industry Rankings DataFrame:
            - 'Rank': Industry rank based on RS
            - 'Industry': Industry name
            - 'Sector': Sector name
            - 'RS': Current RS value
            - '1 Month Ago': RS value one month ago
            - '3 Months Ago': RS value three months ago
            - '6 Months Ago': RS value six months ago
            - 'Tickers': List of stock tickers in the industry
            - 'Rating': Current rating
            - 'Rating (1 Month Ago)': Rating one month ago
            - 'Rating (3 Months Ago)': Rating three months ago
            - 'Rating (6 Months Ago)': Rating six months ago
    """
    stock_df = build_stock_rs_df(tickers, ticker_ref,
                                 period, interval, rs_window)
    stock_df = stock_df.sort_values(by='RS', ascending=False)

    rs_columns = ['RS', '1 Month Ago', '3 Months Ago', '6 Months Ago']
    stock_df = append_ratings(stock_df, rs_columns, method=rating_method)

    columns =  ['Sector', 'Ticker'] + rs_columns
    industry_df = groupby_industry(stock_df, columns, key='RS')

    industry_df = industry_df.sort_values(by='RS', ascending=False)
    industry_df = append_ratings(industry_df, rs_columns, method=rating_method)

    industry_df = industry_df.rename(columns={
        'Ticker': 'Tickers',
    })

    return stock_df, industry_df


#------------------------------------------------------------------------------
# Unit Test
#------------------------------------------------------------------------------

def test_rankings(min_rating=80, rating_method='qcut',
                  rs_window='12mo', out_dir='out'):
    '''
    Parameters
    ----------
    min_rating: int, optional
        The minimum rating for a stock to be included in the rankings.
        Defaults to 80.

    out_dir: str, optional
        The output directory to store CSV tables. Defaults to 'out'.
    '''
    import os
    from datetime import datetime
    from . import stock_indices as si

    code = 'SPX'
    tickers = si.get_tickers(code)
    rank_stock, rank_indust = rankings(tickers, interval='1d',
                                       rating_method=rating_method,
                                       rs_window=rs_window)

    if rank_stock.empty or rank_indust.empty:
        print("Not enough data to generate rankings.")
        return

    print('Stock Rankings:')
    print(rank_stock[rank_stock["Rating (RS)"] >= min_rating])

    print('\n\nIndustry Rankings:')
    print(rank_indust)

    # Save to CSV
    print("\n\n***")
    today = datetime.now().strftime('%Y%m%d')
    os.makedirs(out_dir, exist_ok=True)
    for table, kind in zip([rank_stock, rank_indust],
                           ['stocks', 'industries']):
        filename = f'rs_{kind}_{rs_window}_{rating_method}_{today}.csv'
        table.to_csv(os.path.join(out_dir, filename), index=False)
        print(f'Your "{filename}" is in the "{out_dir}" folder.')
    print("***\n")


if __name__ == "__main__":
    import time

    start_time = time.time()
    test_rankings(rating_method='qcut', rs_window='3mo')
    print(f"Execution time: {time.time() - start_time:.4f} seconds")

