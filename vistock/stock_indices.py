"""
stock_indices.py - Functions for retrieving and managing stock market indices data.

This module provides functions for retrieving ticker symbols for various stock
market indices and identifying index names from their ticker symbols. It
supports querying index tickers from specified sources and obtaining the name
of an index based on its symbol.

Main Functions:
- get_tickers(source): Retrieve ticker symbols for a specified stock market
    index source.
- get_name(index_symbol): Retrieve the name of an index from its ticker symbol.

Usage Examples:
    from stock_indices import get_tickers, get_name

    # Get tickers for the S&P 500 index
    spx_tickers = get_tickers('SPX')

    # Get tickers for the Philadelphia Semiconductor Index
    sox_tickers = get_tickers('SOX')

    # Get tickers for the SPX and the SOX
    tickers = get_tickers('SPX+SOX')

    # Get the name of an index from its symbol
    index_name = get_name('^NDX')
"""
__version__ = "1.6"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/06 (initial version) ~ 2024/08/08 (last revision)"

__all__ = [
    'get_tickers',
    'get_name',
]

from io import StringIO

import requests
import pandas as pd
from bs4 import BeautifulSoup

from . import tw


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


def get_tickers(source):
    """
    Retrieve a list of tickers for the specified index or combined indices.

    Args:
        source (str): The ticker symbol or common abbreviation for the index
            or indices.
            - Yahoo Finance ticker symbols (e.g., '^GSPC' for S&P 500, '^NDX'
                for NASDAQ-100).
            - Common abbreviations (e.g., 'SPX' for S&P 500, 'NDX' for
                NASDAQ-100).
            - Multiple indices can be combined using '+' (e.g., '^GSPC+^NDX').

            Possible values include:
            - '^GSPC', 'SPX': S&P 500
            - '^DJI', 'DJIA': Dow Jones Industrial Average
            - '^NDX', 'NDX': NASDAQ-100
            - '^SOX', 'SOX': PHLX Semiconductor Index

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
        >>> 500 < len(get_tickers('^GSPC+^NDX')) < (500+100)
        True
        >>> 500 < len(get_tickers('SPX+SOX+NDX')) < (500+30+100)
        True
        >>> len(get_tickers('TWSE+TPEX')) >= 2000
        True
        >>> get_tickers('^UNKNOWN')
        Traceback (most recent call last):
            ...
        KeyError: "Index symbol '^UNKNOWN' not found."
    """
    dic = {
        '^GSPC': get_spx_tickers,
        '^DJI': get_djia_tickers,
        '^NDX': get_ndx_tickers,
        '^SOX': get_sox_tickers,
        '^TWII': tw.get_twse_tickers,
        'SPX': get_spx_tickers,
        'DJIA': get_djia_tickers,
        'NDX': get_ndx_tickers,
        'SOX': get_sox_tickers,
        'TWII': tw.get_twse_tickers,
        'TWSE': tw.get_twse_tickers,
        'TPEX': tw.get_tpex_tickers,
        'ESB': tw.get_esb_tickers,
    }

    sources = [s.strip().upper() for s in source.split('+')]
    tickers = set()

    for s in sources:
        if s in dic:
            tickers.update(dic[s]())
        else:
            raise KeyError(f"Index symbol '{s}' not found.")

    return list(tickers)


def get_name(index_symbol):
    """
    Return the name of the index based on the provided symbol.

    Args:
        index_symbol (str): The ticker symbol or common abbreviation for the
            index.
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
            - '^TWII', 'TWII': Taiwan Weighted Index
            - '^STOXX50E': Euro Stoxx 50,
            - '^FTSE': FTSE 100,
            - '^GDAXI': DAX,
            - '^FCHI': CAC 40,
            - '^GSPTSE': S&P/TSX Composite,
            - '^N225': Nikkei 225,
            - '^HSI': Hang Seng Index,

    Returns:
        str: The name of the index if found.
             If not found, returns the original index_symbol.

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
        >>> get_name('^TWII')
        'Taiwan Weighted Index'
        >>> get_name('^HSI')
        'Hang Seng Index'
        >>> get_name('AAPL')
        'AAPL'
    """
    dic = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones Industrial Average',
        '^IXIC': 'NASDAQ',                  # NASDAQ Composite
        '^NDX': 'NASDAQ 100',
        '^RUT': 'Russell 2000',
        '^SOX': 'PHLX Semiconductor Index',
        '^NYA': 'NYSE Composite',
        '^MID': 'S&P MidCap 400',
        '^TWII': 'Taiwan Weighted Index',
        'SPX': 'S&P 500',
        'DJIA': 'Dow Jones Industrial Average',
        'NDX': 'NASDAQ 100',
        'SOX': 'PHLX Semiconductor Index',
        'RUT': 'Russell 2000',
        'NYA': 'NYSE Composite',
        'MID': 'S&P MidCap 400',
        'TWII': 'Taiwan Weighted Index',
        'TWSE': 'Taiwan Stock Exchange',
        'TPEX': 'Taipei Exchange',
        'ESB': 'Emerging Stock Boar',
        '^STOXX50E': 'Euro Stoxx 50',       # Europe
        '^FTSE': 'FTSE 100',                # London, UK
        '^GDAXI': 'DAX',                    # Frankfurt, Germany
        '^FCHI': 'CAC 40',                  # Paris, France
        '^GSPTSE': 'S&P/TSX Composite',     # Canada
        '^N225': 'Nikkei 225',              # Japan
        '^HSI': 'Hang Seng Index',          # Hong Kong
    }
    return dic.get(index_symbol, index_symbol)


#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------

if __name__ == '__main__':
    import doctest, time

    start_time = time.time()
    doctest.testmod()
    print(f"Execution time: {time.time() - start_time:.4f} seconds")

