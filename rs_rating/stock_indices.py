"""
Functions for retrieving and managing stock market indices data.

This module provides functions for retrieving ticker symbols for various stock
market indices and identifying index names from their ticker symbols. It
supports querying index tickers from specified sources and obtaining the name
of an index based on its symbol.

Main Functions:
~~~~~~~~~~~~~~~
- get_tickers(source): Retrieve ticker symbols for a specified stock market
  index source.
- get_name(index_symbol): Retrieve the name of an index from its ticker symbol.
- ticker_from_name(name): Get the ticker symbol of an index from its long name.

Usage Examples:
~~~~~~~~~~~~~~~
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
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/06 (initial version) ~ 2024/10/04 (last revision)"

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


#------------------------------------------------------------------------------
# Wikipedia look-up
#------------------------------------------------------------------------------

def table_from_wikipedia(article, class_, id):
    """
    Fetches a table from a Wikipedia article and returns it as a pandas
    DataFrame.

    Parameters
    ----------
    article : str
        The name of the Wikipedia article.
    class_ : str
        The class attribute of the table to retrieve.
    id : str
        The id attribute of the table to retrieve.

    Returns
    -------
    pandas.DataFrame
        The retrieved table.
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

    Parameters
    ----------
    article : str
        The name of the Wikipedia article.
    class_ : str, optional
        The class attribute of the table. Defaults to 'wikitable sortable'.
    id : str, optional
        The id attribute of the table. Defaults to 'constituents'.

    Returns
    -------
    list
        A list of stock symbols.
    """
    df = table_from_wikipedia(article, class_, id)
    if 'Symbol' in df.columns:
        return df['Symbol'].tolist()
    elif 'Ticker' in df.columns:
        return df['Ticker'].tolist()
    return []


spx_tickers = functools.partial(
        symbols_from_wikipedia_table, 'List_of_S%26P_500_companies',
        class_=None)
djia_tickers = functools.partial(
        symbols_from_wikipedia_table, 'Dow_Jones_Industrial_Average')
ndx_tickers = functools.partial(
        symbols_from_wikipedia_table, 'Nasdaq-100')
rui_tickers = functools.partial(
        symbols_from_wikipedia_table, 'Russell_1000_Index', id=None)

#------------------------------------------------------------------------------
# bullishbears.com look-up
#------------------------------------------------------------------------------

def table_from_bullishbears(article):
    """
    Fetches the first table from a specified Bullish Bears article.

    Parameters
    ----------
    article : str
        The relative path of the article on the Bullish Bears website.

    Returns
    -------
    pd.DataFrame
        The first table found in the specified URL, parsed into a Pandas DataFrame.
    """
    url = f'https://bullishbears.com/{article}'
    return pd.read_html(url)[0]


def symbols_from_bullishbears_table(article):
    """
    Extracts stock symbols from a table in a Bullish Bears article.

    Parameters
    ----------
    article : str
        The relative path of the article on the Bullish Bears website.

    Returns
    -------
    list
        A list of stock symbols from the table's "Symbol" column.

    Examples
    --------
    >>> symbols = symbols_from_bullishbears_table('sp500-stocks-list')
    >>> len(symbols)
    502
    >>> symbols = symbols_from_bullishbears_table('dow-jones-stocks-list')
    >>> len(symbols)
    30
    >>> symbols = symbols_from_bullishbears_table('nasdaq-stocks-list')
    >>> len(symbols)
    100
    >>> symbols = symbols_from_bullishbears_table('russell-2000-stocks-list')
    >>> 'SMCI' in symbols
    True
    >>> len(symbols) > 1990
    True
    """
    df = table_from_bullishbears(article)
    symbols = list(df['Symbol'])
    return [s for s in symbols if isinstance(s, str)]


rut_tickers = functools.partial(
        symbols_from_bullishbears_table, 'russell-2000-stocks-list')


#------------------------------------------------------------------------------
# StatementDog.com look-up
#------------------------------------------------------------------------------

def us_listed_tickers():
    """
    Fetches all symbols of U.S. listed stocks from the StatementDog US stock
    list page and returns them as a list.

    This function mimics the Wilshire 5000 Index by using the StatementDog
    website's US stock list as a proxy for all listed U.S. stocks. The list
    includes stocks from both the New York Stock Exchange (NYSE) and the
    NASDAQ, and serves as a comprehensive dataset of publicly traded companies
    in the U.S.

    Returns
    -------
    list
        A list of stock symbols from the StatementDog US stock list.
        Returns an empty list if the request fails.

    Examples
    --------
    >>> tickers = us_listed_tickers()
    >>> len(tickers) > 5000
    True
    >>> 'AAPL' in tickers
    True
    """
    # URL of the target page
    url = 'https://statementdog.com/us-stock-list'

    # Request the web page content
    response = requests.get(url)

    # Ensure the request was successful
    if response.status_code != 200:
        print("Failed to retrieve the page.")
        return []

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all <span> elements with the class 'us-stock-company-ticker'
    symbols = [span.text for span
               in soup.find_all('span', class_='us-stock-company-ticker')]

    return symbols


#------------------------------------------------------------------------------
# Manually keyed stock list
#------------------------------------------------------------------------------

def sox_tickers():
    """
    Get a list of tickers for companies in the SOX (PHLX Semiconductor).

    This function returns a manually maintained list of SOX tickers.

    Note
    ----
    This list may not be up-to-date and requires periodic updates.

    Returns
    -------
    list
        A list of SOX tickers.

    Examples
    --------
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

#------------------------------------------------------------------------------
# Published Functions
#------------------------------------------------------------------------------

def get_tickers(source):
    """
    Retrieve a list of tickers for the specified index or combined indices.

    Parameters
    ----------
    source : str
        The ticker symbol or common abbreviation for the index or indices.

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
        - '^RUT', 'RUT': Russell 2000
        - '^SOX', 'SOX': PHLX Semiconductor
        - '^W5000', 'W5000': Wilshire 5000 Total Market Index
        - 'U.S. listed': U.S. listed stocks

    Returns
    -------
    list
        A list of tickers for the specified source.

    Raises
    ------
    KeyError
        If the provided source is not recognized or does not
        correspond to a known index.

    Examples
    --------
    >>> len(get_tickers('SPX')) >= 500
    True
    >>> len(get_tickers('^GSPC')) >= 500
    True
    >>> len(get_tickers('^NDX')) >= 100
    True
    >>> len(get_tickers('^RUI')) >= 1000
    True
    >>> len(get_tickers('^RUT')) > 1900
    True
    >>> len(get_tickers('^W5000')) > 5000
    True
    >>> len(get_tickers('U.S.Listed')) > 5000
    True
    >>> len(get_tickers('USLS')) > 5000
    True
    >>> 500 < len(get_tickers('^GSPC+^NDX')) < (500+100)
    True
    >>> 500 < len(get_tickers('SPX+SOX+NDX')) < (500+30+100)
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
        '^RUT': rut_tickers,
        '^SOX': sox_tickers,
        '^W5000': us_listed_tickers,
        'SPX': spx_tickers,
        'DJIA': djia_tickers,
        'NDX': ndx_tickers,
        'SOX': sox_tickers,
        'RUI': rui_tickers,
        'RUT': rut_tickers,
        'R1000': rui_tickers,
        'R2000': rut_tickers,
        'W5000': us_listed_tickers,
        'U.S.LISTED': us_listed_tickers,
        'USLS': us_listed_tickers,
    }

    sources = [s.strip().upper() for s in source.split('+')]
    tickers = set()

    for s in sources:
        if s in dic:
            tickers.update(dic[s]())
        else:
            raise KeyError(f"Index symbol '{s}' not found.")

    symbols = sorted(list(tickers))

    # Filter out symbols containing '.', such as SPAC units or preferred
    # stocks, which may not be supported by yfinance

    # Filter out symbols containing '.', except for Taiwan stocks (.TW, .TWO),
    # as others may not be supported by yfinance
    symbols = [symbol for symbol in symbols
               if '.' not in symbol or symbol.endswith(('.TW', '.TWO'))]

    return symbols


def ticker_from_name(name):
    """
    Get ticker symbol of an index from its long name.

    Parameters
    ----------
    name : str
        The long name of a ticker.

    Returns
    -------
    str
        The ticker symbol.

    Examples
    --------
    >>> ticker_from_name('S&P 500')
    '^GSPC'
    >>> ticker_from_name('Dow Jones Industrial Average')
    '^DJI'
    >>> ticker_from_name('NASDAQ 100')
    '^NDX'
    >>> ticker_from_name('Russell 1000')
    '^RUI'
    >>> ticker_from_name('Wilshire 5000 Total Market Index')
    '^W5000'
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
        "Wilshire 5000 Total Market Index": "^W5000",
        "U.S. Listed Stocks": "USLS",
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

    Parameters
    ----------
    index_symbol : str
        The ticker symbol or common abbreviation for the index.

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
        - '^W5000', 'W5000': Wilshire 5000 Total Market Index
        - 'USLS': U.S. Listed Stocks
        - '^STOXX50E': Euro Stoxx 50
        - '^FTSE': FTSE 100
        - '^GDAXI': DAX
        - '^FCHI': CAC 40
        - '^GSPTSE': S&P/TSX Composite
        - '^N225': Nikkei 225
        - '^HSI': Hang Seng Index

    Returns
    -------
    str
        The name of the index if found. If not found, returns the original index_symbol.

    Examples
    --------
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
    >>> get_name('^W5000')
    'Wilshire 5000 Total Market Index'
    >>> get_name('^TWII')
    'Taiwan Weighted Index'
    >>> get_name('^HSI')
    'Hang Seng Index'
    >>> get_name('SOXX')
    'iShares Semiconductor ETF'
    >>> get_name('AAPL')
    'AAPL'
    """
    dic = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones Industrial Average',
        '^NDX': 'NASDAQ 100',
        '^IXIC': 'NASDAQ Composite',
        '^RUI': 'Russell 1000',
        '^RUT': 'Russell 2000',
        '^SOX': 'PHLX Semiconductor',
        '^W5000': 'Wilshire 5000 Total Market Index',
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
        'W5000': 'Wilshire 5000 Total Market Index',
        'USLS': 'U.S. Listed Stocks',
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
        if yf.Ticker(index_symbol).info['quoteType'] in ('ETF', 'INDEX'):
            return yf.Ticker(index_symbol).info['shortName']
    except:
        pass
    return index_symbol


#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------

if __name__ == '__main__':
    import doctest, time

    start_time = time.time()
    doctest.testmod()
    print(f"Execution time: {time.time() - start_time:.4f} seconds")

