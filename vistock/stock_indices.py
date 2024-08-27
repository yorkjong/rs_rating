"""
stock_indices.py - Functions for retrieving and managing stock market indices data.

This module provides functions for retrieving ticker symbols for various stock
market indices and identifying index names from their ticker symbols. It
supports querying index tickers from specified sources and obtaining the name
of an index based on its symbol.

Main Functions:
---------------
- get_tickers(source): Retrieve ticker symbols for a specified stock market
  index source.
- get_name(index_symbol): Retrieve the name of an index from its ticker symbol.
- ticker_from_name(name): Get the ticker symbol of an index from its long name.

Usage Examples:
---------------
::

    from stock_indices import get_tickers, get_name

    # Get tickers for the S&P 500 index
    spx_tickers = get_tickers('SPX')

    # Get tickers for the Philadelphia Semiconductor
    sox_tickers = get_tickers('SOX')

    # Get tickers for the SPX and the SOX
    tickers = get_tickers('SPX+SOX')

    # Get the name of an index from its symbol
    index_name = get_name('^NDX')
"""
__version__ = "2.1"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/06 (initial version) ~ 2024/08/26 (last revision)"

__all__ = [
    'get_tickers',
    'get_name',
    'ticker_from_name',
]

import functools
from io import StringIO

import requests
import pandas as pd
from bs4 import BeautifulSoup
import yfinance as yf

from . import tw


def table_from_wikipedia(article, class_, id):
    """
    Fetches a table from a Wikipedia article and returns it as a pandas
    DataFrame.

    Args:
        article (str): The name of the Wikipedia article.
        class_ (str): The class attribute of the table to retrieve.
        id (str): The id attribute of the table to retrieve.

    Returns:
        pandas.DataFrame: The retrieved table.
    """
    url = f"https://en.wikipedia.org/wiki/{article}"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    attrs = {}
    if class_:
        attrs['class'] = class_
    if id:
        attrs['id'] = id
    tag = soup.find('table', attrs=attrs)
    return pd.read_html(StringIO(str(tag)))[0]


def symbols_from_wikipedia_table(article,
                                 class_='wikitable sortable',
                                 id='constituents'):
    """
    Extracts stock symbols from a table in a Wikipedia article.

    Args:
        article (str): The name of the Wikipedia article.
        class_ (str, optional): The class attribute of the table. Defaults to
            'wikitable sortable'.
        id (str, optional): The id attribute of the table. Defaults to
            'constituents'.

    Returns:
        list: A list of stock symbols.
    """
    df = table_from_wikipedia(article, class_, id)
    if 'Symbol' in df.columns:
        return df['Symbol'].tolist()
    elif 'Ticker' in df.columns:
        return df['Ticker'].tolist()
    return []


spx_tickers = functools.partial(
        symbols_from_wikipedia_table, 'List_of_S%26P_500_companies')
djia_tickers = functools.partial(
        symbols_from_wikipedia_table, 'Dow_Jones_Industrial_Average')
ndx_tickers = functools.partial(
        symbols_from_wikipedia_table, 'Nasdaq-100')
rui_tickers = functools.partial(
        symbols_from_wikipedia_table, 'Russell_1000_Index', id=None)


def sox_tickers():
    """
    Get a list of tickers for companies in the SOX (PHLX Semiconductor).

    This function returns a manually maintained list of SOX tickers.

    Note: This list may not be up-to-date and requires periodic updates.

    Returns:
        list: A list of SOX tickers.

    Examples:
        >>> tickers = sox_tickers()
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
            - '^RUI', 'RUI': Russell 1000
            - '^SOX', 'SOX': PHLX Semiconductor
            - '^TWII' 'TWII', 'TWSE': Taiwan Weighted Index
            - 'TPEX': Taipei Exchange
            - 'ESB': Emerging Stock Board

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
        >>> len(get_tickers('^RUI')) >= 1000
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
        '^GSPC': spx_tickers,
        '^DJI': djia_tickers,
        '^NDX': ndx_tickers,
        '^RUI': rui_tickers,
        '^SOX': sox_tickers,
        '^TWII': tw.get_twse_tickers,
        'SPX': spx_tickers,
        'DJIA': djia_tickers,
        'NDX': ndx_tickers,
        'SOX': sox_tickers,
        'RUI': rui_tickers,
        'R1000': rui_tickers,
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


def ticker_from_name(name):
    """
    Get ticker symbol of an index from its long name.

    Args:
        name (str): the long name of a ticker.

    Returns:
        str: the ticker symbol

    Examples:
    >>> ticker_from_name('S&P 500')
    '^GSPC'
    >>> ticker_from_name('Dow Jones Industrial Average')
    '^DJI'
    >>> ticker_from_name('NASDAQ 100')
    '^NDX'
    >>> ticker_from_name('Russell 1000')
    '^RUI'
    >>> ticker_from_name('Taiwan Weighted Index')
    '^TWII'
    """
    dic = {
        "S&P 500": "^GSPC",
        "Dow Jones Industrial Average": "^DJI",
        "NASDAQ 100": "^NDX",
        'NASDAQ Composite': "^IXIC",
        "Russell 1000": "^RUI",
        "Russell 2000": "^RUT",
        "PHLX Semiconductor": "^SOX",
        "Taiwan Weighted Index": "^TWII",
        'Euro Stoxx 50': '^STOXX50E',       # Europe
        'FTSE 100': '^FTSE',                # London, UK
        'DAX': '^GDAXI',                    # Frankfurt, Germany
        'CAC 40': '^FCHI',                  # Paris, France
        'S&P/TSX Composite': '^GSPTSE',     # Canada
        'Nikkei 225': '^N225',              # Japan
        'Hang Seng Index': '^HSI',          # Hong Kong
    }
    return dic[name]


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
            - '^NDX', 'NDX': NASDAQ 100
            - '^IXIC', 'COMP': NASDAQ Composite
            - '^RUI', 'RUI', 'R1000': Russell 1000
            - '^RUT', 'RUT', 'R2000': Russell 2000
            - '^SOX', 'SOX': PHLX Semiconductor
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
        'NASDAQ Composite'
        >>> get_name('^NDX')
        'NASDAQ 100'
        >>> get_name('^RUI')
        'Russell 1000'
        >>> get_name('^RUT')
        'Russell 2000'
        >>> get_name('^SOX')
        'PHLX Semiconductor'
        >>> get_name('^TWII')
        'Taiwan Weighted Index'
        >>> get_name('^HSI')
        'Hang Seng Index'
        >>> get_name('AAPL')
        'Apple Inc.'
        >>> get_name('Unknown')
        'Unknown'
    """
    dic = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones Industrial Average',
        '^NDX': 'NASDAQ 100',
        '^IXIC': 'NASDAQ Composite',
        '^RUI': 'Russell 1000',
        '^RUT': 'Russell 2000',
        '^SOX': 'PHLX Semiconductor',
        '^NYA': 'NYSE Composite',
        '^MID': 'S&P MidCap 400',
        '^TWII': 'Taiwan Weighted Index',
        'SPX': 'S&P 500',
        'DJIA': 'Dow Jones Industrial Average',
        'NDX': 'NASDAQ 100',
        'COMP': 'NASDAQ Composite',
        'RUI': 'Russell 1000',
        'RUT': 'Russell 2000',
        'R1000': 'Russell 1000',
        'R2000': 'Russell 2000',
        'SOX': 'PHLX Semiconductor',
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
    if index_symbol in dic:
        return dic[index_symbol]
    try:
        if tw.is_chinese(index_symbol) or tw.is_taiwan_stock(index_symbol):
            return index_symbol
        return yf.Ticker(index_symbol).info['longName']
    except:
        return index_symbol


#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------

if __name__ == '__main__':
    import doctest, time

    start_time = time.time()
    doctest.testmod()
    print(f"Execution time: {time.time() - start_time:.4f} seconds")

