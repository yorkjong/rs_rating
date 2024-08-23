"""
Mansfield Relative Strength (RSM) Module

This module provides functions for calculating and ranking stocks based on
Mansfield Relative Strength (RSM) and related metrics. It includes methods
to compute Dorsey Relative Strength (RSD), Mansfield Relative Strength (RSM)
using both Simple Moving Average (SMA) and Exponential Moving Average (EMA),
as well as functionality to rank stocks against a benchmark index.

Functions
---------
- dorsey_relative_strength(closes, closes_index): Computes Dorsey Relative
  Strength (RSD).
- mansfield_relative_strength(closes, closes_index, window): Computes Mansfield
  Relative Strength (RSM) using SMA.
- mansfield_relative_strength_with_ema(closes, closes_index, window, adjust):
  Computes RSM using EMA.
- ranking(tickers, ticker_ref='^GSPC', period='2y', interval='1wk', ma='SMA',
  window=52): Ranks stocks based on their RSM.

Examples
--------
To calculate RSM for a list of stock tickers and rank them:

>>> tickers = ['AAPL', 'MSFT', 'GOOGL']
>>> rank = ranking(tickers, period='2y', interval='1wk', window=52)
>>> print(rank.head())

To compute Mansfield Relative Strength using SMA for specific close prices:

>>> closes = pd.Series([...])  # Example closing prices
>>> closes_index = pd.Series([...])  # Example index closing prices
>>> rsm = mansfield_relative_strength(closes, closes_index, window=52)
>>> print(rsm)

See Also
--------
- `Mansfield Relative Strength | ChartMill.com
  <https://www.chartmill.com/documentation/technical-analysis/indicators/
  35-Mansfield-Relative-Strength>`_
- `How to create the Mansfield Relative Performance Indicator - Stage Analysis
  <https://www.stageanalysis.net/blog/4266/
  how-to-create-the-mansfield-relative-performance-indicator>`_

"""
__version__ = "1.1"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/23 (initial version) ~ 2024/08/23 (last revision)"

__all__ = [
    'mansfield_relative_strength',
    'mansfield_relative_strength_with_ema',
    'dorsey_relative_strength',
    'ranking',
]

import pandas as pd
import yfinance as yf

from ta import simple_moving_average, exponential_moving_average


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
    ... index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
    >>> index_closes = pd.Series([2000, 2050, 2100],
    ... index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
    >>> mansfield_relative_strength(stock_closes, index_closes, window=2)
    2024-01-01    5.000000
    2024-01-02    5.097847
    2024-01-03    5.238095
    dtype: float64
    """
    rsd = dorsey_relative_strength(closes, closes_index)
    return ((rsd / simple_moving_average(rsd, window)) - 1) * 100


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
    ... index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
    >>> index_closes = pd.Series([2000, 2050, 2100],
    ... index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
    >>> mansfield_relative_strength_with_ema(stock_closes, index_closes,
    ... window=2)
    2024-01-01    5.000000
    2024-01-02    5.097847
    2024-01-03    5.238095
    dtype: float64
    """
    rsd = dorsey_relative_strength(closes, closes_index)
    return ((rsd / exponential_moving_average(rsd, window, adjust)) - 1) * 100


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


def ranking(tickers, ticker_ref='^GSPC', period='2y', interval='1wk',
            window=52, ma="SMA"):
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
        Interval for historical data ('1d', '1wk', '1mo').
        Default to '1wk' (one week).

    window : int, optional
        Window size for moving average calculation. Default to 52.

    ma : str, optional
        Moving average type ('SMA', 'EMA'). Default to 'SMA'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the ranked stocks.
    """
    # Fetch data for stock and index
    df = yf.download([ticker_ref] + tickers, period=period, interval=interval)
    df = df.xs('Close', level='Price', axis=1)

    # Get the function to calculate relative strength
    try:
        rs_func = {
            'SMA': mansfield_relative_strength,
            'EMA': mansfield_relative_strength_with_ema,
        }[ma]
    except KeyError:
        raise ValueError("Invalid moving average type. Must be 'SMA' or 'EMA'.")

    results = []
    for ticker in tickers:
        rs = rs_func(df[ticker], df[ticker_ref], window)
        rsm = pd.Series(rs, name=ticker)

        # Calculate RSM for different time periods
        end_date = rsm.index[-1]
        one_month_ago = end_date - pd.DateOffset(months=1)
        three_months_ago = end_date - pd.DateOffset(months=3)
        six_months_ago = end_date - pd.DateOffset(months=6)

        rsm_1m = rsm[rsm.index <= one_month_ago].iloc[-1]
        rsm_3m = rsm[rsm.index <= three_months_ago].iloc[-1]
        rsm_6m = rsm[rsm.index <= six_months_ago].iloc[-1]

        # Construct DataFrame for current stock
        rank_df = pd.DataFrame({
            'Ticker': [ticker],
            'Price': [df[ticker].iloc[-1]],
            'Relative Strength': [rsm.iloc[-1]],
            '1 Month Ago': [rsm_1m],
            '3 Months Ago': [rsm_3m],
            '6 Months Ago': [rsm_6m]
        })
        results.append(rank_df)

    # Combine results into a single DataFrame
    ranking_df = pd.concat(results, ignore_index=True)

    # Rank based on Relative Strength
    rank_columns = ['Rank', ' 1 Month Ago', ' 3 Months Ago', ' 6 Months Ago']
    rs_columns = ['Relative Strength',
                  '1 Month Ago', '3 Months Ago', '6 Months Ago' ]
    for rank_col, rs_col in zip(rank_columns, rs_columns):
        ranking_df[rank_col] = ranking_df[rs_col].rank(
                ascending=False, method='min').astype(int)

    # Sort by current rank
    ranking_df = ranking_df.sort_values(by='Rank')

    return ranking_df


def main():
    tickers = [
        'AMD', 'ADI', 'AMAT', 'ASML', 'AZTA', 'AVGO', 'COHR', 'ENTG', 'GFS',
        'INTC', 'IPGP', 'KLAC', 'LRCX', 'LSCC', 'MRVL', 'MCHP', 'MU', 'MPWR',
        'NOVT', 'NVDA', 'NXPI', 'ON', 'QRVO', 'QCOM', 'SWKS', 'SYNA', 'TSM',
        'TER', 'TXN', 'WOLF'
    ]

    rank = ranking(tickers, period='2y', interval='1wk', ma="EMA", window=52)
    print(rank.head(10))


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    print(f"Execution time: {time.time() - start_time:.4f} seconds")

