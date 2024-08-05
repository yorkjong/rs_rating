"""
Functions for IBD RS and IBD RS Rating
"""
__version__ = "1.0"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/08/05 (initial version) ~ 2024/08/05 (last revision)"

__all__ = [
    'relative_strength',
    'rankings',
]

import pandas as pd
import numpy as np
import yfinance as yf
import os

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


def relative_strength(closes, closes_ref):
    """
    Calculate the relative strength of a stock compared to a reference index.

    Args:
        closes (pd.Series): Closing prices of the stock.
        closes_ref (pd.Series): Closing prices of the reference index.

    Returns:
        pd.Series: Relative strength values for the stock.
    """
    rs_stock = strength(closes)
    rs_ref = strength(closes_ref)

    # ((a - b) / b) + 1 = a/b
    # return + 1 = growth
    rs = (1 + rs_stock) / (1 + rs_ref) * 100
    return round(rs, 2)
    #return rs.round().astype(int)


def strength(closes):
    """
    Calculate the performance of the last year, with the most recent quarter
    weighted double.

    Args:
        closes (pd.Series): Closing prices of the stock/index.

    Returns:
        pd.Series: Performance values of the stock/index.
    """
    q1 = quarters_return(closes, 1)  # the last quarter
    q2 = quarters_return(closes, 2)  # the last two quarters
    q3 = quarters_return(closes, 3)  # the last three quarters
    q4 = quarters_return(closes, 4)  # the last four quarters
    return 0.4 * q1 + 0.2 * q2 + 0.2 * q3 + 0.2 * q4


def quarters_return(closes, n):
    """
    Calculate the return (percentage change) over the last n quarters.

    Args:
        closes (pd.Series): Closing prices of the stock/index.
        n (int): Number of quarters to look back.

    Returns:
        pd.Series: the return (percentage change) over the last n quarters.
    """
    length = min(len(closes) - 1, n * int(252 / 4))
    ret = closes.pct_change(periods=length)
    return ret.replace([np.inf, -np.inf], np.nan).fillna(0)


def rankings(tickers, ref_ticker='^GSPC', period='1y', min_percentile=80):
    """Generate the stock and industry rankings.

    Args:
        tickers (list): A list of stock tickers to analyze.
        ref_ticker (str, optional): The ticker symbol for the reference index.
            Defaults to '^GSPC' (S&P 500).
        period (str, optional): The period for which to fetch historical data.
            Defaults to '1y' (one year).
        min_percentile (int, optional): The minimum percentile for a stock to be
            included in the rankings. Defaults to 80.

    Returns:
        list: A list of two Pandas DataFrames:
            - The first DataFrame contains the stock rankings, sorted by
              relative strength in descending order. It includes columns for
              rank, ticker, sector, industry, relative strength (RS),
              RS values for 1, 3, and 6 months ago, and percentiles for
              each of these RS values.
            - The second DataFrame contains the industry rankings, also
              sorted by relative strength in descending order. It includes
              columns for rank, industry, sector, relative strength (RS),
              RS values for 1, 3, and 6 months ago, a list of tickers in
              the industry, and percentiles for each of the RS values.
    """
    # Initialize variables
    relative_strengths = []
    industries = {}
    stock_rs = {}

    # Get reference index data
    df_ref = yf.Ticker(ref_ticker).history(period=period)

    def calculate_relative_strength(ticker, df):
        """Calculate RS values for a single ticker."""
        rs_series = relative_strength(df['Close'], df_ref['Close'])
        rs_latest = rs_series.iloc[-1]
        month = 20

        return {
            "latest": rs_latest,
            "1m": rs_series.iloc[-month],
            "3m": rs_series.iloc[-3*month],
            "6m": rs_series.iloc[-6*month]
        }

    def process_stock(ticker):
        """Process stock data and update stock_rs and industries."""
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if len(df) < 6 * 20:  # Ensure at least 6 months of data
            return

        rs_values = calculate_relative_strength(ticker, df)
        sector = stock.info.get('sector', 'Unknown')
        industry = stock.info.get('industry', 'Unknown')

        relative_strengths.append((
            ticker, sector, industry,
            rs_values['latest'], rs_values['1m'], rs_values['3m'], rs_values['6m']
        ))
        stock_rs[ticker] = rs_values['latest']

        if industry not in industries:
            industries[industry] = {
                "info": (industry, sector),
                TITLE_RS: [],
                TITLE_1M: [],
                TITLE_3M: [],
                TITLE_6M: [],
                TITLE_TICKERS: []
            }

        industry_row = industries[industry]
        industry_row[TITLE_RS].append(rs_values['latest'])
        industry_row[TITLE_1M].append(rs_values['1m'])
        industry_row[TITLE_3M].append(rs_values['3m'])
        industry_row[TITLE_6M].append(rs_values['6m'])
        industry_row[TITLE_TICKERS].append(ticker)

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

    def rs_avg(industry_info, column):
        """Calculate average RS value for an industry."""
        return round(np.mean(industry_info[column]), 2)

    def tickers_str(industry_info):
        """Create a comma-separated string of tickers for an industry."""
        tickers = sorted(industry_info[TITLE_TICKERS],
                         key=lambda x: stock_rs[x], reverse=True)
        return ",".join(tickers)

    def industry_data(industry_info):
        """Generate a tuple of industry data."""
        industry, sector = industry_info["info"]
        return (
            industry,
            sector,
            *[rs_avg(industry_info, col) for col
              in [TITLE_RS, TITLE_1M, TITLE_3M, TITLE_6M]],
            tickers_str(industry_info)
        )

    # Process all tickers
    for ticker in tickers:
        process_stock(ticker)

    # Create DataFrames for stock and industry rankings
    df_stocks = pd.DataFrame(
        relative_strengths,
        columns=[TITLE_TICKER, TITLE_SECTOR, TITLE_INDUSTRY, TITLE_RS,
                 TITLE_1M, TITLE_3M, TITLE_6M]
    )

    stock_rankings = rank_by_rs(calculate_percentiles(df_stocks))
    stock_rankings = stock_rankings[
        stock_rankings[TITLE_PERCENTILE] >= min_percentile
    ]

    industry_data_list = [
        industry_data(info)
        for info in industries.values() if len(info[TITLE_TICKERS]) > 1
    ]

    df_industries = pd.DataFrame(
        industry_data_list,
        columns=[TITLE_INDUSTRY, TITLE_SECTOR, TITLE_RS, TITLE_1M,
                 TITLE_3M, TITLE_6M, TITLE_TICKERS]
    )

    industry_rankings = rank_by_rs(calculate_percentiles(df_industries))

    return stock_rankings, industry_rankings


#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------

def main(out_dir='output'):
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM",
               "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS"]
    rank_stock, rank_indust = rankings(tickers, min_percentile=80)

    if rank_stock.empty or rank_indust.empty:
        print("Not enough data to generate rankings.")
        return

    print('Stock Rankings:')
    print(rank_stock)

    print('\n\nIndustry Rankings:')
    print(rank_indust)

    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save to CSV
    rank_stock.to_csv(os.path.join(out_dir, 'rs_stocks.csv'), index=False)
    rank_indust.to_csv(os.path.join(out_dir, 'rs_industries.csv'), index=False)

    print(f"\n***\nTables are in the {out_dir} folder.\n***")


if __name__ == "__main__":
    main()

