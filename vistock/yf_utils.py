"""
Utility functions for working with Yahoo Finance data.

This module contains various utility functions for retrieving and processing
stock data using the Yahoo Finance API via the `yfinance` library.

Functions
---------
- download_tickers_info: Download and return financial information for multiple tickers in parallel.

Example
-------
Here's a basic example of how to use the `download_tickers_info` function:

>>> import yfinance as yf
>>> from vistock.yf_utils import download_tickers_info
>>> symbols = ['AAPL', 'MSFT', 'TSLA']
>>> info_df = download_tickers_info(symbols, max_workers=5, progress=False)
>>> isinstance(info_df, pd.DataFrame)
True
>>> 'AAPL' in info_df.index
True
>>> 'longName' in info_df.columns
True
>>> info_df.loc['AAPL', 'longName']
'Apple Inc.'
"""
__version__ = "1.6"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/26 (initial version) ~ 2024/08/27 (last revision)"

__all__ = [
    'download_tickers_info',
]

import sys

import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd


def download_tickers_info(symbols, max_workers=8, progress=True):
    """
    Download info for multiple tickers in parallel and return as a pandas
    DataFrame.

    Parameters
    ----------
    symbols: list of str
        List of ticker symbols, e.g., ['AAPL', 'MSFT', 'TSLA']
    max_workers: int
        Maximum number of threads to use for parallel requests
    progress: bool
        Whether to show a progress bar

    Returns
    -------
    pandas.DataFrame
        pandas DataFrame containing the info for all tickers

    Examples
    --------
    >>> symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'AMZN']
    >>> info_df = download_tickers_info(symbols, max_workers=5, progress=False)
    >>> info_df['longName']['AAPL']
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
        ticker = yf.Ticker(symbol)
        return ticker.info

    def print_progress_bar(iteration, total, length=48, fill='*'):
        """Call in a loop to create a terminal progress bar with the percentage
        in the middle.
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
        sys.stdout.write(f'\r[{bar_with_percent}]  {iteration} of {total} '
                         'info downloaded')
        sys.stdout.flush()  # to ensure immediate display

    info_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit fetch_info tasks for all symbols to the thread pool
        future_to_symbol = {
            executor.submit(fetch_info, symbol): symbol for symbol in symbols
        }

        iteration = 0

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                info = future.result()
                info['symbol'] = symbol # Add the symbol to the info dictionary
                info_list.append(info)
                iteration += 1
                if progress:
                    print_progress_bar(iteration, len(symbols))
            except Exception as e:
                print(f"Error fetching info for {symbol}: {e}")

    # Convert the list of info dictionaries to a DataFrame
    info_df = pd.DataFrame(info_list)

    # Set 'symbol' column as the index of the DataFrame
    info_df.set_index('symbol', inplace=True)

    return info_df


if __name__ == "__main__":
    import doctest, time

    start_time = time.time()
    doctest.testmod()
    print(f"Execution time: {time.time() - start_time:.4f} seconds")
