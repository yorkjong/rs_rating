"""
Stock Indices Functions

This module provides functions for retrieving ticker symbols of various stock
market indices and identifying index names from their ticker symbols.

Main functions:
- get_tickers(source): Get tickers for a specified source
- get_name(index_symbol): Get the name of an index from its symbol

Usage:
    from stock_indices import get_tickers, get_name

    spx_tickers = get_tickers('SPX')
    sox_tickers = get_tickers('SOX')
    index_name = get_name('^NDX')
"""
__version__ = "1.4"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/06 (initial version) ~ 2024/08/07 (last revision)"

__all__ = [
    'get_tickers',
    'get_name',
]

from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_spx_tickers():
    """
    Retrieve a list of tickers for companies in the SPX (S&P 500 index).

    Returns:
        list: A list of ticker symbols.

    Examples:
        >>> tickers = get_spx_tickers()
        >>> len(tickers) >= 500
        True
        >>> 'AAPL' in tickers
        True
        >>> 'TSLA' in tickers
        True
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})
    df = pd.read_html(StringIO(str(table)))[0]
    return df['Symbol'].tolist()


def get_ndx_tickers():
    """
    Retrieve a list of tickers for companies in the NDX (NASDAQ-100 Index).

    Returns:
        list: A list of ticker symbols.

    Examples:
        >>> tickers = get_ndx_tickers()
        >>> len(tickers) >= 100
        True
        >>> 'NVDA' in tickers
        True
        >>> 'TSLA' in tickers
        True
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(StringIO(str(table)))[0]
    return df['Ticker'].tolist()


def get_djia_tickers():
    """
    Retrieve a list of tickers for companies in the DJIA (Dow Jones Industrial
    Average) index.

    Returns:
        list: A list of ticker symbols.

    Examples:
        >>> tickers = get_djia_tickers()
        >>> len(tickers) == 30
        True
        >>> 'MSFT' in tickers
        True
        >>> 'AAPL' in tickers
        True
    """
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable', 'id': 'constituents'})
    df = pd.read_html(StringIO(str(table)))[0]
    return df['Symbol'].tolist()


def get_sox_tickers():
    """
    Get a list of tickers for companies in the SOX (PHLX Semiconductor
    Index).

    This function returns a manually maintained list of SOX tickers.
    Note: This list may not be up-to-date and requires periodic updates.

    Returns:
        list: A list of SOX tickers.

    Examples:
        >>> tickers = get_sox_tickers()
        >>> len(tickers) == 30
        True
        >>> 'NVDA' in tickers
        True
        >>> 'AVGO' in tickers
        True
        >>> 'TSM' in tickers
        True
    """
    tickers = [
        'AMD', 'ADI', 'AMAT', 'ASML', 'AZTA', 'AVGO', 'COHR', 'ENTG', 'GFS',
        'INTC', 'IPGP', 'KLAC', 'LRCX', 'LSCC', 'MRVL', 'MCHP', 'MU', 'MPWR',
        'NOVT', 'NVDA', 'NXPI', 'ON', 'QRVO', 'QCOM', 'SWKS', 'SYNA', 'TSM',
        'TER', 'TXN', 'WOLF'
    ]
    return tickers


def get_all_tickers():
    """
    Retrieve a combined list of tickers from multiple major stock market indices.

    This function aggregates ticker symbols from the following indices:
    - S&P 500, SPX
    - Dow Jones Industrial Average, DJIA
    - NASDAQ 100, NDX
    - PHLX Semiconductor, SOX

    Returns:
        list: A list of unique ticker symbols from the specified indices.

    Examples:
        >>> tickers = get_all_tickers()
        >>> 500 <= len(tickers) < 660
        True
        >>> 'AAPL' in tickers
        True
        >>> 'MSFT' in tickers
        True
    """
    return list(set(get_spx_tickers()) |
                set(get_djia_tickers()) |
                set(get_ndx_tickers()) |
                set(get_sox_tickers()))


def get_tickers(source):
    """
    Retrieve a list of tickers for the specified index or combined index.

    Args:
        source (str): The ticker symbol or common abbreviation for the index.
            - Yahoo Finance ticker symbols (e.g., '^GSPC' for S&P 500, '^NDX'
              for NASDAQ-100).
            - Common abbreviations (e.g., 'SPX' for S&P 500, 'NDX' for
              NASDAQ-100).
            - Special keyword 'ALL' to retrieve tickers for all indices
              combined.

            Possible values include:
            - '^GSPC', 'SPX': S&P 500
            - '^DJI', 'DJIA': Dow Jones Industrial Average
            - '^NDX', 'NDX': NASDAQ-100
            - '^SOX', 'SOX': PHLX Semiconductor Index
            - 'ALL': All indices combined

    Returns:
        list: A list of tickers for the specified source.

    Raises:
        KeyError: If the provided source is not recognized or does not
            correspond to a known index.

    Examples:
        >>> len(get_tickers('SPX')) >= 500
        True
        >>> len(get_tickers('^GSPC')) >= 500
        True
        >>> len(get_tickers('^NDX')) >= 100
        True
        >>> get_tickers('^UNKNOWN')
        Traceback (most recent call last):
            ...
        KeyError: "Index symbol '^UNKNOWN' not found."
        >>> len(get_tickers('ALL')) > 500
        True
    """
    dic = {
        '^GSPC': get_spx_tickers,
        '^DJI': get_djia_tickers,
        '^NDX': get_ndx_tickers,
        '^SOX': get_sox_tickers,
        'SPX': get_spx_tickers,
        'DJIA': get_djia_tickers,
        'NDX': get_ndx_tickers,
        'SOX': get_sox_tickers,
        'ALL': get_all_tickers,
    }
    try:
        return dic[source.upper()]()
    except KeyError:
        raise KeyError(f"Index symbol '{source}' not found.")


def get_name(index_symbol):
    """
    Return the name of the index based on the provided symbol.

    Args:
        index_symbol (str): The ticker symbol or common abbreviation for the index.
            - Yahoo Finance ticker symbols (e.g., '^GSPC' for S&P 500, '^NDX' for
              NASDAQ-100).
            - Common abbreviations (e.g., 'SPX' for S&P 500, 'NDX' for
              NASDAQ-100).

            Possible values include:
            - '^GSPC', 'SPX': S&P 500
            - '^DJI', 'DJIA': Dow Jones Industrial Average
            - '^IXIC': NASDAQ
            - '^NDX', 'NDX': NASDAQ 100
            - '^RUT', 'RUT': Russell 2000
            - '^SOX', 'SOX': PHLX Semiconductor Index
            - '^NYA', 'NYA': NYSE Composite
            - '^MID', 'MID': S&P MidCap 400

    Returns:
        str: The name of the index corresponding to the provided symbol, or
            'Unknown' if the symbol is not found.

    Examples:
        >>> get_name('SPX')
        'S&P 500'
        >>> get_name('^GSPC')
        'S&P 500'
        >>> get_name('^DJI')
        'Dow Jones Industrial Average'
        >>> get_name('^IXIC')
        'NASDAQ'
        >>> get_name('^NDX')
        'NASDAQ 100'
        >>> get_name('^RUT')
        'Russell 2000'
        >>> get_name('^SOX')
        'PHLX Semiconductor Index'
        >>> get_name('^HSI')
        'Unknown'
    """
    dic = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones Industrial Average',
        '^IXIC': 'NASDAQ',
        '^NDX': 'NASDAQ 100',
        '^RUT': 'Russell 2000',
        '^SOX': 'PHLX Semiconductor Index',
        '^NYA': 'NYSE Composite',
        '^MID': 'S&P MidCap 400',
        'SPX': 'S&P 500',
        'DJIA': 'Dow Jones Industrial Average',
        'NDX': 'NASDAQ 100',
        'SOX': 'PHLX Semiconductor Index',
        'RUT': 'Russell 2000',
        'NYA': 'NYSE Composite',
        'MID': 'S&P MidCap 400',
    }
    return dic.get(index_symbol, 'Unknown')


if __name__ == "__main__":
    import doctest
    doctest.testmod()

