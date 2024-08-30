"""
Mansfield Relative Strength (RSM) Module

This module provides functions for calculating and ranking stocks based on
Mansfield Relative Strength (RSM) and related metrics. It includes methods
to compute Dorsey Relative Strength (RSD), Mansfield Relative Strength (RSM)
using both Simple Moving Average (SMA) and Exponential Moving Average (EMA),
as well as functionality to rank stocks against a benchmark index.

Functions:
----------
- dorsey_relative_strength(closes, closes_index): Computes Dorsey Relative
  Strength (RSD).
- mansfield_relative_strength(closes, closes_index, window): Computes Mansfield
  Relative Strength (RSM) using SMA.
- mansfield_relative_strength_with_ema(closes, closes_index, window, adjust):
  Computes RSM using EMA.
- ranking(tickers, ticker_ref='^GSPC', period='2y', interval='1wk', ma='SMA',
  window=52): Ranks stocks based on their RSM.

Examples:
---------
To calculate RSM for a list of stock tickers and rank them:

>>> tickers = ['AAPL', 'MSFT', 'GOOGL']
>>> rank = ranking(tickers, period='2y', interval='1wk', window=52)
>>> print(rank.head())

To compute Mansfield Relative Strength using SMA for specific close prices:

>>> closes = pd.Series([...])  # Example closing prices
>>> closes_index = pd.Series([...])  # Example index closing prices
>>> rsm = mansfield_relative_strength(closes, closes_index, window=52)
>>> print(rsm)

See Also:
---------
- `Mansfield Relative Strength | ChartMill.com
  <https://www.chartmill.com/documentation/technical-analysis/indicators/
  35-Mansfield-Relative-Strength>`_
- `How to create the Mansfield Relative Performance Indicator - Stage Analysis
  <https://www.stageanalysis.net/blog/4266/
  how-to-create-the-mansfield-relative-performance-indicator>`_

"""
__version__ = "1.7"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/23 (initial version) ~ 2024/08/30 (last revision)"

__all__ = [
    'mansfield_relative_strength',
    'mansfield_relative_strength_with_ema',
    'dorsey_relative_strength',
    'ranking',
]

import numpy as np
import pandas as pd
import yfinance as yf

import vistock.yf_utils as yfu
from .ta import simple_moving_average, exponential_moving_average


def mansfield_relative_strength(closes, closes_index, window):
    """
    Calculate Mansfield Relative Strength (RSM) for given close prices, index
    close prices, and window size using a Simple Moving Average (SMA).

    Parameters
    ----------
    closes : pandas.Series
        Series of closing prices for the stock.

    closes_index : pandas.Series
        Series of closing prices for the benchmark index.

    window : int
        Window size for calculating the Simple Moving Average of the Dorsey
        Relative Strength.

    Returns
    -------
    pandas.Series
        Series containing the calculated Mansfield Relative Strength (RSM)
        values with SMA.
    Examples
    --------
    >>> stock_closes = pd.Series([100, 105, 110],
    ...     index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
    >>> index_closes = pd.Series([2000, 2050, 2100],
    ...     index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
    >>> mansfield_relative_strength(stock_closes, index_closes, window=2)
    2024-01-01    5.000000
    2024-01-02    5.097847
    2024-01-03    5.238095
    dtype: float64
    """
    closes = closes.ffill()
    closes_index = closes_index.ffill()
    rsd = dorsey_relative_strength(closes, closes_index)
    rsm = ((rsd / simple_moving_average(rsd, window)) - 1) * 100
    return round(rsm, 2)


def mansfield_relative_strength_with_ema(closes, closes_index, window,
                                         adjust=False):
    """
    Calculate Mansfield Relative Strength (RSM) using an Exponential Moving
    Average (EMA) for the Dorsey Relative Strength.

    Parameters
    ----------
    closes : pandas.Series
        Series of closing prices for the stock.

    closes_index : pandas.Series
        Series of closing prices for the benchmark index.

    window : int
        Number of periods over which to calculate the Exponential Moving Average
        of the Dorsey Relative Strength.

    adjust : bool, optional
        Whether to adjust the EMA calculation (default is False).

    Returns
    -------
    pandas.Series
        Series containing the calculated Mansfield Relative Strength (RSM)
        values with EMA.

    Examples
    --------
    >>> stock_closes = pd.Series([100, 105, 110],
    ...     index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
    >>> index_closes = pd.Series([2000, 2050, 2100],
    ...     index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
    >>> mansfield_relative_strength_with_ema(stock_closes, index_closes,
    ...     window=2)
    2024-01-01    5.000000
    2024-01-02    5.097847
    2024-01-03    5.238095
    dtype: float64
    """
    closes.ffill(inplace=True)
    closes_index.ffill(inplace=True)
    rsd = dorsey_relative_strength(closes, closes_index)
    rsm = ((rsd / exponential_moving_average(rsd, window, adjust)) - 1) * 100
    return round(rsm, 2)


def dorsey_relative_strength(closes, closes_index):
    """
    Calculate Dorsey Relative Strength (RSD) for given close prices and index
    close prices.

    Parameters
    ----------
    closes : pandas.Series
        Series of closing prices for the stock.

    closes_index : pandas.Series
        Series of closing prices for the benchmark index.

    Returns
    -------
    pandas.Series
        Series containing the calculated Dorsey Relative Strength (RSD) values.
    """
    return (closes / closes_index) * 100



def move_columns_to_end(df, columns_to_move):
    """
    Move specified columns to the end of the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose columns need to be reordered.

    columns_to_move : list of str
        List of column names to move to the end.

    Returns
    -------
    pandas.DataFrame
        DataFrame with specified columns moved to the end.
    """
    # Get the list of columns that are not in columns_to_move
    cols = [col for col in df.columns if col not in columns_to_move] + columns_to_move
    # Reorder DataFrame columns
    df = df[cols]
    return df


def ranking(tickers, ticker_ref='^GSPC', period='2y', interval='1wk', ma="SMA"):
    """
    Rank stocks based on their Mansfield Relative Strength (RSM) against an
    index benchmark.

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
        Interval for historical data ('1d', '1wk').
        Default to '1wk' (one week).

    ma : str, optional
        Moving average type ('SMA', 'EMA'). Default to 'SMA'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the ranked stocks.
    """
    # Fetch data for stocks and index
    df_all = yf.download([ticker_ref] + tickers, period=period, interval=interval)
    df_ref = df_all.xs(ticker_ref, level='Ticker', axis=1)

    # Fetch info for stocks
    info = yfu.download_tickers_info(tickers, ['sector', 'industry'])

    try:
        # Get the function to calculate relative strength
        rs_func = {
            'SMA': mansfield_relative_strength,
            'EMA': mansfield_relative_strength_with_ema,
        }[ma]

        # Select the MA function based on the 'ma' parameter
        ma_func = {
            'SMA': simple_moving_average,
            'EMA': exponential_moving_average,
        }[ma]
    except KeyError:
        raise ValueError("Invalid moving average type. Must be 'SMA' or 'EMA'.")
    try:
        rs_window = { '1d': 260, '1wk': 52}[interval]
        ma_window = { '1d': 150, '1wk': 30}[interval]
        vma_window = { '1d': 50, '1wk': 10}[interval]
    except KeyError:
        raise ValueError("Invalid interval. " "Must be '1d', '1wk', or '1mo'.")

    results = []
    for ticker in tickers:
        df = df_all.xs(ticker, level='Ticker', axis=1)
        rsm = rs_func(df['Close'], df_ref['Close'], rs_window)
        price_div_ma = round(df['Close'] / ma_func(df['Close'], ma_window), 2)
        vol_div_ma = round(df['Volume'] / ma_func(df['Volume'], vma_window), 2)

        # Calculate RSM for different time periods
        end_date = rsm.index[-1]
        one_week_ago = end_date - pd.DateOffset(weeks=1)
        one_month_ago = end_date - pd.DateOffset(months=1)
        three_months_ago = end_date - pd.DateOffset(months=3)
        six_months_ago = end_date - pd.DateOffset(months=6)

        # Construct DataFrame for current stock
        row = {
            'Ticker': ticker,
            'Sector': info[ticker]['sector'],
            'Industry': info[ticker]['industry'],
            'RS (%)': rsm.asof(end_date),
            '1 Week Ago': rsm.asof(one_week_ago),
            '1 Month Ago': rsm.asof(one_month_ago),
            '3 Months Ago': rsm.asof(three_months_ago),
            '6 Months Ago': rsm.asof(six_months_ago),
            'Price': round(df['Close'].iloc[-1], 2),
            'Price / MA': price_div_ma[-1],
            'Volume / VMA': vol_div_ma[-1],
        }
        results.append(row)

    # Combine results into a single DataFrame
    ranking_df = pd.DataFrame(results)

    # Rank based on Relative Strength
    rank_columns = ['RS Rank (%)', ' 1 Week Ago',
                    ' 1 Month Ago', ' 3 Months Ago', ' 6 Months Ago']
    rs_columns = ['RS (%)', '1 Week Ago',
                  '1 Month Ago', '3 Months Ago', '6 Months Ago' ]
    for rank_col, rs_col in zip(rank_columns, rs_columns):
        rank_pct = ranking_df[rs_col].rank(pct=True)
        ranking_df[rank_col] = round(rank_pct * 100, 2)

    # Sort by current rank
    ranking_df = ranking_df.sort_values(by='RS Rank (%)', ascending=False)

    ranking_df = move_columns_to_end(
        ranking_df,
        ['Price', 'Price / MA', 'Volume / VMA']
    )
    return ranking_df


def main(period='2y', ma="EMA", out_dir='out'):
    import os
    from datetime import datetime
    from vistock.stock_indices import get_tickers

    code = 'SPX+DJIA+NDX+SOX'
    code = 'SOX'
    tickers = get_tickers(code)

    rank = ranking(tickers, period=period, interval='1wk', ma=ma)
    print(rank.head(10))

    # Save to CSV
    print("\n\n***")
    os.makedirs(out_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    filename = f'{code}_stocks_{period}_{ma}_{today}.csv'
    rank.to_csv(os.path.join(out_dir, filename), index=False)
    print(f'Your "{filename}" is in the "{out_dir}" folder.')
    print("***\n")


if __name__ == "__main__":
    import time

    start_time = time.time()
    #main()
    main(ma="SMA")
    print(f"Execution time: {time.time() - start_time:.4f} seconds")
