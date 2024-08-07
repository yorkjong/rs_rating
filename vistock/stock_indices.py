"""
Stock Indices Functions

This module provides functions for retrieving ticker symbols of various stock
market indices and identifying index names from their ticker symbols.

Main functions:
- get_sp500_tickers(): Get S&P 500 tickers
- get_nasdaq100_tickers(): Get NASDAQ-100 tickers
- get_djia_tickers(): Get Dow Jones Industrial Average tickers
- get_sox_tickers(): Get PHLX Semiconductor Index tickers
- get_tickers(index_ticker): Get tickers for a specified index
- get_name(ticker): Get the name of an index from its ticker

Usage:
    from stock_indices import get_tickers, get_name

    sp500_tickers = get_tickers('^GSPC')
    index_name = get_name('^NDX')
"""
__version__ = "1.1"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/06 (initial version) ~ 2024/08/07 (last revision)"

__all__ = [
    'get_sp500_tickers',
    'get_nasdaq100_tickers',
    'get_djia_tickers',
    'get_sox_tickers',
    'get_tickers',
    'get_name',
]

from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_sp500_tickers():
    """
    Retrieves a list of tickers for companies in the S&P 500 index.

    Returns:
        list: A list of ticker symbols.

    Examples:
        >>> tickers = get_sp500_tickers()
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


def get_nasdaq100_tickers():
    """
    Retrieves a list of tickers for companies in the NASDAQ-100 index.

    Returns:
        list: A list of ticker symbols.

    Examples:
        >>> tickers = get_nasdaq100_tickers()
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
    Retrieves a list of tickers for companies in the DJIA (Dow Jones Industrial
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
    Provides a list of SOX (PHLX Semiconductor Index) tickers.

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


def get_tickers(index_ticker):
    """
    Returns a list of tickers for the given index.

    Args:
        index_ticker: The ticker symbol of the index.

    Returns:
        A list of tickers for the given index.

    Raises:
        KeyError: If the index ticker is not found.

    Examples:
        >>> len(get_tickers('^GSPC')) >= 500
        True
        >>> len(get_tickers('^NDX')) >= 100
        True
        >>> get_tickers('^UNKNOWN')
        Traceback (most recent call last):
            ...
        KeyError: "Index ticker '^UNKNOWN' not found."
    """
    dic = {
        '^GSPC': get_sp500_tickers,
        '^DJI': get_djia_tickers,
        '^NDX': get_nasdaq100_tickers,
        '^SOX': get_sox_tickers,
    }
    try:
        return dic[index_ticker]()
    except KeyError:
        raise KeyError(f"Index ticker '{index_ticker}' not found.")


def get_name(ticker):
    """
    Return the name of the index based on the ticker.

    Args:
        ticker: The ticker symbol of the index.

    Returns:
        The name of the index, or 'Unknow' if the ticker is not found.

    Examples:
        >>> get_name('^GSPC')
        'S&P 500'
        >>> get_name('^DJI')
        'DJIA'
        >>> get_name('^IXIC')
        'NASDAQ'
        >>> get_name('^NDX')
        'NASDAQ-100'
        >>> get_name('^RUT')
        'Russell 2000'
        >>> get_name('^SOX')
        'SOX Index'
        >>> get_name('^HSI')
        'Unknown'
    """
    dic = {
        '^GSPC': 'S&P 500',
        '^DJI': 'DJIA',
        '^IXIC': 'NASDAQ',
        '^NDX': 'NASDAQ-100',
        '^RUT': 'Russell 2000',
        '^SOX': 'SOX Index',
        '^NYA': 'NYSE Composite',
        '^MID': 'S&P MidCap 400'
    }
    return dic.get(ticker, 'Unknown')


if __name__ == "__main__":
    import doctest
    doctest.testmod()

