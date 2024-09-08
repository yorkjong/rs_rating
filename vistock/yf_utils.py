"""
Utility functions for working with Yahoo Finance data.

This module contains various utility functions for retrieving and processing
stock data using the Yahoo Finance API via the `yfinance` library.
"""
__version__ = "3.4"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/26 (initial version) ~ 2024/09/08 (last revision)"

__all__ = [
    'calc_weighted_metric',
    'fetch_financials',
    'download_financials',
    'download_tickers_info',
]

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

#------------------------------------------------------------------------------
# Weighted Average Metric (e.g., EPS, Revenue)
#------------------------------------------------------------------------------

def calc_weighted_metric(financials, tickers_info, metric, weight_field):
    """
    Calculate the weighted average of a specified financial metric for all stock
    symbols in the provided dataset using NumPy. The weights can be based on any
    specified field (e.g., market capitalization or shares outstanding).

    Parameters
    ----------
    financials : dict
        A dictionary where each key is a stock ticker and the value is a
        DataFrame of the ticker's quarterly financials.
    tickers_info : dict
        A dictionary where each key is a stock ticker and the value is a
        dictionary of the ticker's info, including market cap, shares
        outstanding, etc.
    metric : str
        The name of the financial metric to calculate (e.g., 'Basic EPS',
        'Total Revenue', 'Operating Revenue').
    weight_field : str
        The field name to use for weighting (e.g., 'marketCap',
        'sharesOutstanding').

    Returns
    -------
    numpy.ndarray
        The weighted average of the specified metric over the specified
        number of quarters (or years).

    Examples
    --------
    >>> tickers = ['AAPL', 'MSFT', 'GOOG']
    >>> financials = download_financials(tickers, ['Basic EPS'])
    ...                             # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    [...**********************100%**********************]
    3 of 3 financials downloaded
    >>> tickers_info = {
    ...     'AAPL': {'marketCap': 3357369434112,
    ...              'sharesOutstanding': 15204100096},
    ...     'MSFT': {'marketCap': 2985852141568,
    ...              'sharesOutstanding': 7433039872},
    ...     'GOOG': {'marketCap': 1864140193792,
    ...              'sharesOutstanding': 5584999936},
    ... }
    >>> weighted_eps = calc_weighted_metric(financials, tickers_info,
    ...                                     'Basic EPS', 'sharesOutstanding')
    >>> type(weighted_eps)
    <class 'numpy.ndarray'>
    >>> weighted_eps.shape
    (7,)
    """
    # Initialize lists to store metric and weight data
    metric_list = []
    weights = []

    for symbol, financial_df in financials.items():
        # Retrieve the weight based on the provided weight field
        weight = tickers_info.get(symbol, {}).get(weight_field, 0)

        if (weight > 0 and financial_df is not None
                       and metric in financial_df.columns):
            # Apply forward fill to fill missing metric values
            metric_data = financial_df[metric].ffill().values
            metric_list.append(metric_data)
            weights.append(weight)
        else:
            print("Warning: No valid metric or "
                  f"weight data available for {symbol}.")

    if not metric_list:
        print("No valid metric data found for any symbol.")
        return np.array([])

    # Ensure all metric arrays have the same length. Use NaN for padding.
    max_length = max(len(data) for data in metric_list)
    metric_array = np.full((len(metric_list), max_length), np.nan)

    # Fill metric data right-aligned
    for i, data in enumerate(metric_list):
        metric_array[i, -len(data):] = data  # Right-align the data

    # Convert weights to a NumPy array
    weights = np.array(weights)

    # Calculate weighted metric using broadcasting
    weighted_metric = metric_array * weights[:, np.newaxis]

    # Calculate weighted average metric
    total_weight = weights.sum()
    if total_weight == 0:
        print("Total weight is zero. "
              "Cannot calculate weighted average metric.")
        return np.array([])

    weighted_avg_metric = np.nansum(weighted_metric, axis=0) / total_weight

    return weighted_avg_metric


#------------------------------------------------------------------------------
# Stock Data Downloading
#------------------------------------------------------------------------------

def fetch_financials(symbol, fields=None, frequency='quarterly'):
    """
    Fetch the financials for a single ticker symbol using yfinance.

    Parameters
    ----------
    symbol: str
        Ticker symbol as a string
    fields: list, optional
        List of fields to return. If None, all fields will be returned.
        Defaults to None.
    frequency: str
        The frequency of the financial data ('quarterly' or 'annual').

    Returns
    -------
    DataFrame
        DataFrame containing the ticker's financials
    """
    try:
        ticker = yf.Ticker(symbol)
        try:
            financials = {
                'quarterly': ticker.quarterly_financials.T,
                'annual': ticker.financials.T,
            }[frequency]
        except KeyError:
            raise ValueError("Frequency must be 'quarterly' or 'annual'.")

        financials = financials.sort_index(ascending=True)
        if fields:
            financials = financials[fields]
        return financials
    except Exception as e:
        print(f"Error fetching financials for {symbol}: {e}")
        return pd.DataFrame()


def download_financials(symbols, fields=None, frequency='quarterly',
                        max_workers=8, progress=True):
    """
    Downloads the financials (quarterly or annual) of multiple stocks and
    returns the specified fields.

    Parameters
    ----------
    symbols: list of str
        List of ticker symbols, e.g., ['AAPL', 'MSFT', 'TSLA'].
    fields: list, optional
        List of fields to return. If None, all fields will be returned.
        Defaults to None.
    frequency: str, optional
        The frequency of the financial data to fetch. Options are 'quarterly'
        or 'annual'. Defaults to 'quarterly'.
    max_workers: int, optional
        Maximum number of threads to use for parallel requests. Defaults to 8.
    progress: bool, optional
        Whether to show a progress bar. Defaults to True.

    Returns
    -------
    dict:
        A dictionary where each key is a stock ticker, and the value is a
        DataFrame of the specified fields.

    Examples
    --------
    >>> symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'AMZN']
    >>> financials = download_financials(symbols, frequency='annual')
    ...                             # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    [...*******************100%**********************]
      5 of 5 financials downloaded
    >>> epses = financials['AAPL']['Basic EPS']
    >>> type(epses)
    <class 'pandas.core.series.Series'>
    >>> len(epses) >= 4
    True
    """
    financials_dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_financials, symbol, fields, frequency):
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
                except KeyError:
                    if key in ('previousClose', 'trailingEps',
                               'revenuePerShare', 'trailingPE',
                               'marketCap', 'sharesOutstanding'):
                        inf[key] = np.NaN  # Default for numeric fields
                    elif key in ['quoteType', 'sector', 'industry']:
                        inf[key] = ''  # Default for string fields
                    else:
                        inf[key] = None  # Default for other data types
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
