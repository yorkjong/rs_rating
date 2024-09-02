"""
Utility functions for working with Yahoo Finance data.

This module contains various utility functions for retrieving and processing
stock data using the Yahoo Finance API via the `yfinance` library.

Functions
---------
- download_quarterly_financials: Downloads the quarterly financials for multiple
  stocks and returns the specified fields.
- download_tickers_info: Download and return financial information for multiple
  stocks in parallel.

Example
-------
Here's a basic example of how to use the `download_tickers_info` function:

>>> import yfinance as yf
>>> from vistock.yf_utils import download_tickers_info
>>> symbols = ['AAPL', 'MSFT', 'TSLA']
    >>> info = download_tickers_info(symbols) # doctest: +NORMALIZE_WHITESPACE
    ...                                       # doctest: +ELLIPSIS
    [...**********************100%**********************]
    3 of 3 info downloaded
>>> info['AAPL']['longName']
'Apple Inc.'
"""
__version__ = "2.7"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/26 (initial version) ~ 2024/09/01 (last revision)"

__all__ = [
    'calc_cap_weighted_eps',
    'calc_share_weighted_eps',
    'download_quarterly_financials',
    'download_tickers_info',
]

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

#------------------------------------------------------------------------------
# Weighted Average EPS
#------------------------------------------------------------------------------

def calc_cap_weighted_eps(financials, tickers_info):
    """
    Calculate the market-cap-weighted average EPS (Earnings Per Share) for all
    stock symbols in the provided data using NumPy for numerical calculations.

    Parameters
    ----------
    financials : dict
        A dictionary where each key is a stock ticker and the value is a
        DataFrame of the ticker's quarterly financials.
    tickers_info : dict
        A dictionary where each key is a stock ticker and the value is a
        dictionary of the ticker's info, including market cap.

    Returns
    -------
    numpy.ndarray
        The market-cap-weighted average EPS over the specified number of
        quarters.

    Examples
    --------
    >>> tickers = ['AAPL', 'MSFT', 'GOOG']
    >>> financials = download_quarterly_financials(tickers, ['Basic EPS'])
    ...                             # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    [...**********************100%**********************]
    3 of 3 financials downloaded
    >>> tickers_info = {
    ...     'AAPL': {'marketCap': 2500000000},
    ...     'MSFT': {'marketCap': 2000000000},
    ...     'GOOG': {'marketCap': 1800000000},
    ... }
    >>> eps_avg = calc_weighted_average_eps(financials, tickers_info)
    >>> type(eps_avg)
    <class 'numpy.ndarray'>
    """
    # Initialize lists to store EPS and market cap data
    eps_list = []
    market_caps = []

    for symbol, financial_df in financials.items():
        # Retrieve the market cap for each symbol
        market_cap = tickers_info.get(symbol, {}).get('marketCap', 0)

        if (market_cap > 0 and financial_df is not None
                           and 'Basic EPS' in financial_df.columns):
            # Apply forward fill to fill missing EPS values
            eps = financial_df['Basic EPS'].ffill().values
            eps_list.append(eps)
            market_caps.append(market_cap)
        else:
            print("Warning: No valid EPS or "
                  f"market cap data available for {symbol}.")

    if not eps_list:
        print("No valid EPS data found for any symbol.")
        return np.array([])

    # Ensure all EPS arrays have the same length. Use NaN for padding.
    max_length = max(len(eps) for eps in eps_list)
    eps_array = np.full((len(eps_list), max_length), np.nan)

    # Fill EPS data right-aligned
    for i, eps in enumerate(eps_list):
        eps_array[i, -len(eps):] = eps  # Right-align the data

    # Convert market caps to a NumPy array
    market_caps = np.array(market_caps)

    # Calculate weighted EPS using broadcasting
    weighted_eps = eps_array * market_caps[:, np.newaxis]

    # Calculate weighted average EPS
    total_market_cap = market_caps.sum()
    if total_market_cap == 0:
        print("Total market cap is zero. "
              "Cannot calculate weighted average EPS.")
        return np.array([])

    weighted_avg_eps = np.nansum(weighted_eps, axis=0) / total_market_cap

    return weighted_avg_eps


def calc_share_weighted_eps(financials, tickers_info):
    """
    Calculate the share-weighted average EPS (Earnings Per Share) for all
    stock symbols in the provided data using NumPy for numerical calculations.

    Parameters
    ----------
    financials : dict
        A dictionary where each key is a stock ticker and the value is a
        DataFrame of the ticker's quarterly financials.
    tickers_info : dict
        A dictionary where each key is a stock ticker and the value is a
        dictionary of the ticker's info, including market cap and previous
        close price.

    Returns
    -------
    numpy.ndarray
        The share-weighted average EPS over the specified number of
        quarters.
    """
    # Initialize lists to store EPS, shares outstanding, and market cap data
    eps_list = []
    shares_outstanding = []

    for symbol, financial_df in financials.items():
        # Retrieve the previous close price and market cap for each symbol
        previous_close = tickers_info.get(symbol, {}).get('previousClose', 0)
        market_cap = tickers_info.get(symbol, {}).get('marketCap', 0)

        if previous_close > 0:
            # Calculate shares outstanding
            shares = market_cap / previous_close
        else:
            shares = 0

        if (shares > 0 and financial_df is not None
                       and 'Basic EPS' in financial_df.columns):
            # Apply forward fill to fill missing EPS values
            eps = financial_df['Basic EPS'].ffill().values
            eps_list.append(eps)
            shares_outstanding.append(shares)
        else:
            print("Warning: No valid EPS or "
                  f"share data available for {symbol}.")

    if not eps_list:
        print("No valid EPS data found for any symbol.")
        return np.array([])

    # Ensure all EPS arrays have the same length. Use NaN for padding.
    max_length = max(len(eps) for eps in eps_list)
    eps_array = np.full((len(eps_list), max_length), np.nan)

    # Fill EPS data right-aligned
    for i, eps in enumerate(eps_list):
        eps_array[i, -len(eps):] = eps  # Right-align the data

    # Convert shares outstanding to a NumPy array
    shares_outstanding = np.array(shares_outstanding)

    # Calculate weighted EPS using broadcasting
    weighted_eps = eps_array * shares_outstanding[:, np.newaxis]

    # Calculate weighted average EPS
    total_shares = shares_outstanding.sum()
    if total_shares == 0:
        print("Total shares outstanding is zero. "
              "Cannot calculate weighted average EPS.")
        return np.array([])

    weighted_avg_eps = np.nansum(weighted_eps, axis=0) / total_shares

    return weighted_avg_eps


#------------------------------------------------------------------------------
# Stock Data Downloading
#------------------------------------------------------------------------------

def download_quarterly_financials(symbols, fields=None, max_workers=8,
                                  progress=True):
    """
    Downloads the quarterly financials of multiple stocks and returns the
    specified fields.

    Parameters
    ----------
    symbols: list of str
        List of ticker symbols, e.g., ['AAPL', 'MSFT', 'TSLA']
    fields: list, optional
        List of fields to return. If None, all fields will be returned.
    max_workers: int
        Maximum number of threads to use for parallel requests
    progress: bool
        Whether to show a progress bar

    Returns
    -------
    dict:
        A dictionary where each key is a stock ticker, and the value is a
        DataFrame of the specified fields.

    Examples
    --------
    >>> symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'AMZN']
    >>> financials = download_quarterly_financials(symbols)
    ...                             # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    [...*******************100%**********************]
      5 of 5 financials downloaded
    >>> financials['AAPL']['Basic EPS']
    2022-12-31     NaN
    2023-03-31     NaN
    2023-06-30    1.27
    2023-09-30    1.47
    2023-12-31    2.19
    2024-03-31    1.53
    2024-06-30     1.4
    Name: Basic EPS, dtype: object
    """
    def fetch_financials(symbol):
        """
        Fetch the quarterly financials for a single ticker symbol using
        yfinance.

        Parameters
        ----------
        symbol: str
            Ticker symbol as a string

        Returns
        -------
        DataFrame
            DataFrame containing the ticker's quarterly financials
        """
        try:
            # Fetch the quarterly financials and transpose the DataFrame
            financials = yf.Ticker(symbol).quarterly_financials.T
            financials = financials.sort_index(ascending=True)
            if fields:
                # Filter DataFrame to include only specified fields
                financials = financials[fields]
            return financials
        except Exception as e:
            print(f"Error fetching financials for {symbol}: {e}")
            return pd.DataFrame()

    financials_dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit fetch_financials tasks for all symbols to the thread pool
        future_to_symbol = {
            executor.submit(fetch_financials, symbol):
            symbol for symbol in symbols
        }

        iteration = 0

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                financials = future.result()  # Blocking call, waits for the result
                if not financials.empty:
                    financials_dict[symbol] = financials

                if progress:
                    iteration += 1
                    print_progress_bar(iteration, len(symbols),
                                       suffix='financials downloaded')
            except Exception as e:
                print(f"Error fetching financials for {symbol}: {e}")

    return financials_dict


def download_tickers_info(symbols, fields=None, max_workers=8, progress=True):
    """
    Downloads the basic information of multiple stocks and returns the
    specified fields.

    Parameters
    ----------
    symbols: list of str
        List of ticker symbols, e.g., ['AAPL', 'MSFT', 'TSLA']
    fields: list, optional
        List of fields to return. If None, all fields will be returned.
    max_workers: int
        Maximum number of threads to use for parallel requests
    progress: bool
        Whether to show a progress bar

    Returns
    -------
    dict:
        A dictionary where each key is a stock ticker, and the value is a
        dictionary of the specified fields.

    Examples
    --------
    >>> symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'AMZN']
    >>> info = download_tickers_info(symbols) # doctest: +NORMALIZE_WHITESPACE
    ...                                       # doctest: +ELLIPSIS
    [...*******************100%**********************]  5 of 5 info downloaded
    >>> info['AAPL']['longName']
    'Apple Inc.'
    """
    def fetch_info(symbol):
        """
        Fetch the info for a single ticker symbol using yfinance.

        Parameters
        ----------
        symbol: str
            Ticker symbol as a string
        Returns
        -------
        dict
            Dictionary containing the ticker's info
        """
        try:
            info = yf.Ticker(symbol).info
            if fields is None:
                return info
            if 'symbol' not in info or info['symbol'] != symbol:
                return {}
            if 'quoteType' not in info:
                return {}

            inf = {}
            # Filter info dictionary to include only requested fields
            for key in fields:
                try:
                    inf[key] = info[key]
                except KeyError as e:
                    if key in ['previousClose', 'marketCap',
                               'trailingEps', 'forwardEps']:
                        inf[key] = np.NaN
                    elif key in ['sector', 'industry']:
                        inf[key] = ''
                    else:
                        print(f"Error fetching data for {symbol}: {e}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
        return inf

    info_dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit fetch_info tasks for all symbols to the thread pool
        future_to_symbol = {
            executor.submit(fetch_info, symbol): symbol for symbol in symbols
        }

        iteration = 0

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                info = future.result()  # Blocking call, waits for the result
                if info:
                    info_dict[symbol] = info

                if progress:
                    iteration += 1
                    print_progress_bar(iteration, len(symbols),
                                       suffix='info downloaded')
            except Exception as e:
                print(f"Error fetching info for {symbol}: {e}")

    return info_dict


def print_progress_bar(iteration, total, length=48, fill='*', suffix=''):
    """
    Call in a loop to create a terminal progress bar with the percentage in
    the middle.

    Parameters
    ----------
    iteration: int
        Current iteration number
    total: int
        Total number of iterations
    length: int, optional
        Length of the progress bar in characters (default is 48)
    fill: str, optional
        Character used to fill the progress bar (default is '*')
    suffix: str, optional
        String to display at the end of the progress bar (default is '')

    Returns
    -------
    None

    Examples
    --------
    >>> print_progress_bar(3, 10, length=30) # doctest: +NORMALIZE_WHITESPACE
    [*********    30%              ]  3 of 10
    """
    percent = f"{100. * (iteration / total):3.0f}%"
    percent_len = len(percent)

    filled_length = int(length * iteration // total)
    bar = fill * filled_length + ' ' * (length - filled_length)

    # Calculate the start position to place the percentage in the middle
    pos = (length - percent_len) // 2

    # Replace part of the bar with the percentage string
    bar_with_percent = bar[:pos] + percent + bar[pos + percent_len:]

    # The \r moves the cursor back to the start of the line
    sys.stdout.write(f'\r[{bar_with_percent}]  {iteration} of {total}'
                     f' {suffix}')
    sys.stdout.flush()  # to ensure immediate display

    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


#------------------------------------------------------------------------------
# Unit Test
#------------------------------------------------------------------------------

if __name__ == "__main__":
    import doctest, time

    start_time = time.time()
    doctest.testmod()
    print(f"Execution time: {time.time() - start_time:.4f} seconds")
