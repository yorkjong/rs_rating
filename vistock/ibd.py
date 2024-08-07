"""
ibd.py - Stock Analysis and Ranking Module

This module provides tools for analyzing and ranking stocks based on their
relative strength compared to a benchmark index, inspired by the Investor's
Business Daily (IBD) methodology.

The module includes functions for calculating relative strength and generating
stock and industry rankings.

Key Features:
- Relative strength calculation
- Stock and industry ranking generation
- Percentile-based filtering of rankings

Public Functions:
- relative_strength: Calculate the relative strength of a stock compared to a
    reference index.
- rankings: Generate comprehensive rankings for both individual stocks and
    industries.

Constants:
- TITLE_PERCENTILE: Column name for the percentile ranking in the output
                    DataFrames. Used for filtering results based on percentile
                    thresholds.

Usage:
    import ibd

    # Generate rankings for a list of stocks
    tickers = ['MSFT', 'NVDA', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    stock_rankings, industry_rankings = ibd.rankings(tickers)

    # Calculate relative strength for a single stock
    stock_rs = ibd.relative_strength(stock_closes, index_closes)

    # Filter rankings based on a minimum percentile
    min_percentile = 80
    top_stocks = stock_rankings[
        stock_rankings[ibd.TITLE_PERCENTILE] >= min_percentile]

Note: This module requires pandas, yfinance, and numpy libraries to be
installed.

For detailed information on each function, please refer to their individual
docstrings.
"""
__version__ = "1.2"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/05 (initial version) ~ 2024/08/07 (last revision)"

__all__ = [
    'relative_strength',
    'rankings',
    'TITLE_PERCENTILE',
]

import os

import pandas as pd
import numpy as np

import yfinance as yf

TITLE_RANK = "Rank"
TITLE_TICKER = "Ticker"
TITLE_TICKERS = "Tickers"
TITLE_SECTOR = "Sector"
TITLE_INDUSTRY = "Industry"
TITLE_PERCENTILE = "Percentile"
TITLE_1M = "1 Month Ago"
TITLE_3M = "3 Months Ago"
TITLE_6M = "6 Months Ago"
TITLE_RS = "Relative Strength"


#------------------------------------------------------------------------------
# IBD RS (Relative Strength) Rating
#------------------------------------------------------------------------------

def relative_strength(closes, closes_ref):
    """
    Calculate the relative strength of a stock compared to a reference index.

    Relative Strength (RS) is a metric used to evaluate the performance of a
    stock relative to a benchmark index. A higher RS rating indicates that the
    stock has outperformed the index, while a lower RS rating suggests
    underperformance.

    This function calculates the RS rating by comparing the quarter-weighted
    growth of the stock's closing prices to the quarter-weighted growth of
    the reference index's closing prices over the past year. The formula is as
    follows:

        growth = a/b =  ((a - b) / b) + 1 = return + 1
        growth_rate = growth_stock / growth_index
        rating = growth_rate * 100

    The quarter-weighted growth is calculated using the `weighted_return`
    function.

    Args:
        closes (pd.Series): Closing prices of the stock.
        closes_ref (pd.Series): Closing prices of the reference index.

    Returns:
        pd.Series: Relative strength values for the stock.

    Example:
        >>> stock_closes = pd.Series([100, 102, 105, 103, 107])
        >>> index_closes = pd.Series([1000, 1010, 1015, 1005, 1020])
        >>> rs = relative_strength(stock_closes, index_closes)
    """
    ret_stock = weighted_return(closes)
    ret_ref = weighted_return(closes_ref)
    rs = (1 + ret_stock) / (1 + ret_ref) * 100
    return round(rs, 2)


def weighted_return(closes):
    """
    Calculate the performance of the last year, with the most recent quarter
    weighted double.

    This function calculates returns for each of the last four quarters and
    applies a weighting scheme that emphasizes recent performance. The most
    recent quarter is given a weight of 40%, while each of the three preceding
    quarters are given a weight of 20%.

    Args:
        closes (pd.Series): Closing prices of the stock/index.

    Returns:
        pd.Series: Performance values of the stock/index.

    Example:
        >>> closes = pd.Series([100, 102, 105, 103, 107, 110, 112])
        >>> weighted_perf = weighted_return(closes)
    """
    p1 = quarters_return(closes, 1) # performance over the last quarter
    p2 = quarters_return(closes, 2) # performance over the last two quarters
    p3 = quarters_return(closes, 3) # performance over the last three quarters
    p4 = quarters_return(closes, 4) # performance over the last four quarters
    return 0.4 * p1 + 0.2 * p2 + 0.2 * p3 + 0.2 * p4


def quarters_return(closes, n):
    """
    Calculate the return (percentage change) over the last n quarters.

    This function uses 63 trading days (252 / 4) as an approximation for
    one quarter. This is based on the common assumption of 252 trading
    days in a year.

    Args:
        closes (pd.Series): Closing prices of the stock/index.
        n (int): Number of quarters to look back.

    Returns:
        pd.Series: the return (percentage change) over the last n quarters.

    Example:
        >>> closes = pd.Series([100, 102, 105, 103, 107, 110, 112])
        >>> quarterly_return = quarters_return(closes, 1)
    """
    length = min(len(closes) - 1, n * int(252 / 4))
    ret = closes.pct_change(periods=length)
    return ret.replace([np.inf, -np.inf], np.nan).fillna(0)


#------------------------------------------------------------------------------
# IBD RS Rankings (with RS rating)
#------------------------------------------------------------------------------

def rankings(tickers, ref_ticker='^GSPC', period='2y'):
    """
    Analyze stocks and generate ranking tables for individual stocks and
    industries.

    This function calculates relative strength (RS) for given stocks compared
    to a reference index, and then ranks both individual stocks and industries
    based on their RS values. It provides historical RS data and percentile
    rankings.

    Args:
        tickers (List[str]): A list of stock tickers to analyze.
        ref_ticker (str, optional): The ticker symbol for the reference index.
            Defaults to '^GSPC' (S&P 500).
        period (str, optional): The period for which to fetch historical data.
            Defaults to '2y' (two years).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two Pandas DataFrames:
            1. Stock Rankings DataFrame:
               - Columns: rank, ticker, sector, industry, RS (current),
                 RS (1 month ago), RS (3 months ago), RS (6 months ago),
                 percentile (current), percentile (1 month ago),
                 percentile (3 months ago), percentile (6 months ago)
               - Sorted by current RS in descending order

            2. Industry Rankings DataFrame:
               - Columns: rank, industry, sector, RS (current),
                 RS (1 month ago), RS (3 months ago), RS (6 months ago),
                 tickers (list of tickers in the industry),
                 percentile (current), percentile (1 month ago),
                 percentile (3 months ago), percentile (6 months ago)
               - Sorted by current RS in descending order

    Example:
        >>> tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        >>> stock_rankings, industry_rankings = rankings(tickers)
        >>> print(stock_rankings.head())
        >>> print(industry_rankings.head())
    """
    def process_stocks(tickers, ref_ticker='^GSPC', period='2y'):
        """
        Processes stock data to extract relevant information for rankings.

        Iterates through the provided tickers, fetches historical data,
        calculates relative strength values, and gathers sector and industry
        information. It also updates industry-specific data for later
        processing.

        Returns:
            tuple: A tuple containing two elements:
                - df_stocks (pd.DataFrame): DataFrame with stock information,
                including ticker, sector, industry, and RS values.
                - industries (dict): Dictionary containing industry-specific
                data, such as RS values and tickers for each industry.
        """
        df_ref = yf.Ticker(ref_ticker).history(period=period)

        data = []
        industries = {}
        for ticker, stock, df in gen_stock_data(tickers, period):
            rs_values = calculate_rs_values(df['Close'], df_ref['Close'])
            info = stock.info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')

            data.append((ticker, sector, industry, *rs_values.values()))
            update_industry_data(industries, industry, sector, rs_values, ticker)

        df_stocks = pd.DataFrame(
            data,
            columns=[TITLE_TICKER, TITLE_SECTOR, TITLE_INDUSTRY,
                     TITLE_RS, TITLE_1M, TITLE_3M, TITLE_6M]
        )
        return df_stocks, industries

    def gen_stock_data(tickers, period):
        """Generate stock data."""
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if len(df) < 6 * 20:  # Ensure at least 6 months of data
                continue
            yield ticker, stock, df

    def calculate_rs_values(prices_stock, prices_ref):
        """Calculate RS values for a single ticker."""
        rs_series = relative_strength(prices_stock, prices_ref)
        rs_latest = rs_series.iloc[-1]
        month = 20

        return {
            "latest": rs_latest,
            "1m": rs_series.iloc[-month],
            "3m": rs_series.iloc[-3*month],
            "6m": rs_series.iloc[-6*month]
        }

    def update_industry_data(industries, industry, sector, rs_values, ticker):
        """Updates industry data with information from a single stock."""
        industry_row = industries.setdefault(industry, {
            "info": (industry, sector),
            TITLE_RS: [],
            TITLE_1M: [], TITLE_3M: [], TITLE_6M: [], TITLE_TICKERS: []
        })
        industry_row[TITLE_RS].append(rs_values['latest'])
        industry_row[TITLE_1M].append(rs_values['1m'])
        industry_row[TITLE_3M].append(rs_values['3m'])
        industry_row[TITLE_6M].append(rs_values['6m'])
        industry_row[TITLE_TICKERS].append(ticker)

    #--------------------------------------------------------------------------

    def calculate_percentiles(df):
        """Calculate percentiles for RS and its historical values."""
        df[TITLE_PERCENTILE] = pd.qcut(df[TITLE_RS], 100, labels=False,
                                       duplicates="drop")
        for title in [TITLE_1M, TITLE_3M, TITLE_6M]:
            df[f" {title}"] = pd.qcut(df[title], 100, labels=False,
                                      duplicates="drop")
        return df

    def rank_by_rs(df):
        """Rank stocks or industries by RS."""
        df = df.sort_values(TITLE_RS, ascending=False)
        df[TITLE_RANK] = range(1, len(df) + 1)
        return df

    #--------------------------------------------------------------------------

    def process_industries(industries, stock_rs):
        """Processes industry data to prepare it for ranking.

        Takes a dictionary of industry data and a dictionary of stock relative
        strengths as input. It extracts relevant information for each industry,
        such as average RS values and a list of tickers, and creates a DataFrame
        suitable for ranking.

        Args:
            industries (dict): Dictionary containing industry-specific data,
                including RS values and tickers for each industry.
            stock_rs (dict): Dictionary mapping stock tickers to their
                relative strength values.

        Returns:
            pd.DataFrame: DataFrame with industry information, including
                industry name, sector, average RS values, and a list of tickers.
        """
        industry_data = [
            get_industry_row(info, stock_rs)
            for info in industries.values() if len(info[TITLE_TICKERS]) > 1
        ]
        return pd.DataFrame(
            industry_data,
            columns=[TITLE_INDUSTRY, TITLE_SECTOR,
                     TITLE_RS,  TITLE_1M, TITLE_3M, TITLE_6M, TITLE_TICKERS]
        )

    def get_industry_row(industry_info, stock_rs):
        """Extracts a single row of data for an industry."""
        industry, sector = industry_info["info"]
        return (
            industry,
            sector,
            *[rs_avg(industry_info, col)
                for col in [TITLE_RS, TITLE_1M, TITLE_3M, TITLE_6M]],
            tickers_str(industry_info, stock_rs)
        )

    def rs_avg(industry_info, column):
        """Calculate average RS value for an industry."""
        return round(np.mean(industry_info[column]), 2)

    def tickers_str(industry_info, stock_rs):
        """Create a comma-separated string of tickers for an industry."""
        tickers = sorted(industry_info[TITLE_TICKERS],
                         key=lambda x: stock_rs[x], reverse=True)
        return ",".join(tickers)

    df_stocks, industries = process_stocks(tickers, ref_ticker, period)
    stock_rankings = rank_by_rs(calculate_percentiles(df_stocks))
    stock_rs = dict(zip(stock_rankings[TITLE_TICKER],
                        stock_rankings[TITLE_RS]))

    industry_df = process_industries(industries, stock_rs)
    industry_rankings = rank_by_rs(calculate_percentiles(industry_df))

    return stock_rankings, industry_rankings


#------------------------------------------------------------------------------
# Unit Test
#------------------------------------------------------------------------------

def main(min_percentile=80, out_dir='out'):
    '''
    Args:
        min_percentile (int, optional): The minimum percentile for a stock to be
            included in the rankings. Defaults to 80.
        out_dir (str, optional): The output directory to store CSV tables.
            Defaults to 'out'
    '''
    from stock_indices import get_sox_tickers
    rank_stock, rank_indust = rankings(get_sox_tickers())

    if rank_stock.empty or rank_indust.empty:
        print("Not enough data to generate rankings.")
        return

    print('Stock Rankings:')
    print(rank_stock[rank_stock[TITLE_PERCENTILE] >= min_percentile])

    print('\n\nIndustry Rankings:')
    print(rank_indust)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save to CSV
    print("\n\n***")
    for table, kind in zip([rank_stock, rank_indust],
                           ['stocks', 'industries']):
        filename = f'rs_{kind}.csv'
        table.to_csv(os.path.join(out_dir, filename), index=False)
        print(f'Your "{filename}" is in the "{out_dir}" folder.')
    print("***\n")


if __name__ == "__main__":
    main()

