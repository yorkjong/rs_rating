"""
Utility functions for working with Yahoo Finance data.

This module contains various utility functions for retrieving and processing
stock data using the Yahoo Finance API via the `yfinance` library.

Functions
---------
- download_tickers_info: Download and return financial information for multiple
  tickers in parallel.

Example
-------
Here's a basic example of how to use the `download_tickers_info` function:

>>> import yfinance as yf
>>> from vistock.yf_utils import download_tickers_info
>>> symbols = ['AAPL', 'MSFT', 'TSLA']
>>> info = download_tickers_info(symbols, max_workers=5, progress=False)
>>> info['AAPL']['longName']
'Apple Inc.'
"""
__version__ = "1.9"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/26 (initial version) ~ 2024/08/30 (last revision)"

__all__ = [
    'download_quarterly_financials',
    'download_tickers_info',
]

import sys

import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd


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
    ...            # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
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
            stock = yf.Ticker(symbol)
            # Fetch the quarterly financials and transpose the DataFrame
            financials = stock.quarterly_financials.T
            return financials.sort_index(ascending=True)
        except Exception as e:
            print(f"Error fetching financials for {symbol}: {e}")
            return None

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
                if financials is not None:
                    if fields:
                        # Filter DataFrame to include only specified fields
                        financials = financials[fields]
                    financials_dict[symbol] = financials
                else:
                    financials_dict[symbol] = None
                iteration += 1
                if progress:
                    print_progress_bar(iteration, len(symbols),
                                       suffix='financials downloaded')
            except Exception as e:
                print(f"Error fetching financials for {symbol}: {e}")
        if progress:
            sys.stdout.write('\n')

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
            stock = yf.Ticker(symbol)
            return stock.info
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return {}

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
                info_dict[symbol] = info
                iteration += 1
                if progress:
                    print_progress_bar(iteration, len(symbols),
                                       suffix='info downloaded')
            except Exception as e:
                print(f"Error fetching info for {symbol}: {e}")
        if progress:
            sys.stdout.write('\n')

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
    percent = f"{100. * (iteration / total):2.0f}%"
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


if __name__ == "__main__":
    import doctest, time

    start_time = time.time()
    doctest.testmod()
    print(f"Execution time: {time.time() - start_time:.4f} seconds")
