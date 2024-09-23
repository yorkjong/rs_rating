"""
ibd_fin.py -- IBD Financial Analysis Module

This module provides functions for financial data analysis and comparison, using
methods inspired by those used by Investors Business Daily (IBD).

Key functionalities include:

- Calculation of relative strength for various financial metrics (e.g., EPS,
  Revenue) against benchmarks such as the S&P 500.
- Creation of comparative DataFrames that rank stocks based on their financial
  metrics and performance.
- Sorting and analysis of financial performance across multiple stocks using
  custom ranking methods.

Functions:

- metric_strength_vs_benchmark: Calculates the relative strength of a financial
  metric versus a benchmark.
- weighted_yoy_growth: Computes weighted Year-over-Year (YoY) growth for
  financial data.
- yoy_growth: Calculates Year-over-Year (YoY) growth for a given financial
  metric series.
- financial_metric_ranking: Creates and ranks a DataFrame of financial metrics
  for multiple stocks, based on custom ranking methods.

Usage:

Import this module to access functions for analyzing and ranking financial data.
For example:
::

from ibd_fin import financial_metric_ranking

# Example usage
ranking_df = financial_metric_ranking(stock_data)
"""
__version__ = "1.0"
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/09/15 (initial version) ~ 2024/09/24 (last revision)"

__all__ = [
    'metric_strength_vs_benchmark',
    'financial_metric_ranking',
]

import numpy as np
import pandas as pd

import vistock.yf_utils as yfu


#------------------------------------------------------------------------------
# Financial Metric Relative Strength
#------------------------------------------------------------------------------

def metric_strength_vs_benchmark(quarterly_metric, annual_metric,
                                 quarterly_bench, annual_bench):
    """
    Calculate the relative strength of a financial metric series versus a
    benchmark.

    This function calculates the weighted YoY growth for both the financial
    metric and the benchmark, and then computes the relative strength by
    comparing the two.

    Parameters
    ----------
    quarterly_metric : pd.Series
        Quarterly financial metric series.
    annual_metric : pd.Series
        Annual financial metric series.
    quarterly_bench : pd.Series
        Quarterly benchmark series.
    annual_bench : pd.Series
        Annual benchmark series.

    Returns
    -------
    pd.Series
        Series containing the relative strength values, calculated as the
        difference in weighted YoY growth between the metric and the benchmark,
        multiplied by 100.
    """
    # Calculate weighted YoY growth
    weighted_yoy_growth_metric = weighted_yoy_growth(quarterly_metric,
                                                     annual_metric)
    weighted_yoy_growth_bench = weighted_yoy_growth(quarterly_bench,
                                                    annual_bench)

    # Calculate relative strength
    strength = (weighted_yoy_growth_metric - weighted_yoy_growth_bench) * 100
    return strength


def weighted_yoy_growth(quarterly_data, annual_data):
    """
    Calculate weighted Year-over-Year (YoY) growth for financial data.

    The function combines quarterly and annual YoY growth data, applying a
    weighted moving average to account for the relative importance of each
    frequency.

    Parameters
    ----------
    quarterly_data : pd.Series
        Quarterly financial data series.
    annual_data : pd.Series
        Annual financial data series.

    Returns
    -------
    pd.Series
        Series containing the weighted YoY growth values.
    """
    # Calculate YoY growth for each period
    quarterly_yoy_growth = yoy_growth(quarterly_data, frequency='Q')
    annual_yoy_growth = yoy_growth(annual_data, frequency='A')

    # Weights based on the importance of quarterly vs annual data
    quarterly_weight = 2    # weight for quarterly data
    annual_weight = 1       # weight for annual data

    # Rolling moving average for smoothing YoY growth values
    moving_average = lambda x: x.rolling(window=3, min_periods=1).mean()

    # Combine quarterly and annual YoY growth with weights
    growth = (
        moving_average(quarterly_yoy_growth * quarterly_weight) +
        moving_average(annual_yoy_growth * annual_weight)
    ) / (quarterly_weight + annual_weight)

    return growth


def yoy_growth(data_series, frequency):
    """
    Calculate Year-over-Year (YoY) growth for a financial data series.

    Parameters
    ----------
    data_series : pd.Series
        Series of financial data (e.g., revenue, EPS, RPS).
    frequency : str
        'Q' for quarterly data, 'A' for annual data.

    Returns
    -------
    pd.Series
        Series containing the YoY growth values, where the YoY growth is
        calculated as the percentage change from the corresponding value one
        year prior, adjusted by the minimum absolute value of the current and
        previous values.
    """
    if frequency == 'Q':
        period = 4  # For quarterly data, one year is 4 quarters
    elif frequency == 'A':
        period = 1  # For annual data, one year is 1 year
    else:
        period = 1  # Default case

    # Ensure inputs are pandas Series
    if not isinstance(data_series, pd.Series):
        data_series = pd.Series(data_series)
    data_series = data_series.infer_objects()

    # Shift series to align current and previous values
    shifted_series = data_series.shift(period)

    # Compute minimum absolute values for the current and previous values
    min_abs_values = np.minimum(data_series.abs(), shifted_series.abs())

    # Calculate YoY growth using min abs value
    growth = (data_series - shifted_series) / min_abs_values

    return growth


#------------------------------------------------------------------------------
# Financial Metric Ranking
#------------------------------------------------------------------------------

def financial_metric_ranking(tickers):
    # Fetch info for stocks
    info = yfu.download_tickers_info(
        tickers,
        ['quoteType', 'previousClose',
         'trailingEps', 'revenuePerShare', 'trailingPE',
         'marketCap', 'sharesOutstanding', 'sector', 'industry',]
    )

    # Fetch financials data for stocks
    fins_q = yfu.download_financials(
        tickers, ['Basic EPS', 'Operating Revenue'], 'quarterly')
    fins_a = yfu.download_financials(
        tickers, ['Basic EPS', 'Operating Revenue'], 'annual')

    # weighted EPS of benchmark
    bench_eps_q = yfu.calc_weighted_metric(
        fins_q, info, 'Basic EPS', 'sharesOutstanding')
    bench_eps_a = yfu.calc_weighted_metric(
        fins_a, info, 'Basic EPS', 'sharesOutstanding')

    # weighted RPS of benchmark
    bench_rev_q = yfu.calc_weighted_metric(fins_q, info,
                                           'Operating Revenue', 'marketCap')
    bench_rev_a = yfu.calc_weighted_metric(fins_a, info,
                                           'Operating Revenue', 'marketCap')
    rows = []
    for ticker in tickers:
        eps_q = fins_q[ticker]['Basic EPS']
        eps_a = fins_a[ticker]['Basic EPS']
        eps_rs = metric_strength_vs_benchmark(eps_q, eps_a,
                                              bench_eps_q, bench_eps_a)

        pe = info[ticker]['trailingPE']
        if not isinstance(pe, float):
            print(f"info[{ticker}]['trailingPE']: {pe}")
            pe = np.NaN

        # Construct DataFrame for current stock
        row = {
            'Ticker': ticker,
            'Sector': info[ticker]['sector'],
            'Industry': info[ticker]['industry'],
            'Price': info[ticker]['previousClose'],
            'EPS RS (%)': eps_rs.iloc[-1],
            'TTM EPS': info[ticker]['trailingEps'],
            'Rev RS (%)': rev_rs.iloc[-1],
            'TTM RPS': info[ticker]['revenuePerShare'],
            'TTM PE': round(pe, 2),
        }
        rows.append(row)

    # Combine results into a single DataFrame
    ranking_df = pd.DataFrame(rows)

    # Sort by current rank
    ranking_df = ranking_df.sort_values(by='EPS RS (%)', ascending=False)

    return ranking_df


#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------

def main():
    import os
    from datetime import datetime
    from vistock.stock_indices import get_tickers

    code = 'SOX'
    #code = 'SPX+DJIA+NDX+RUI+SOX'
    tickers = get_tickers(code)

    tickers = ['NVDA', 'TSM']
    rank = financial_metric_ranking(tickers)
    print(rank.head(10))

    # Save to CSV
    print("\n\n***")
    os.makedirs(out_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    filename = f'{code}_ibd_fin_{today}.csv'
    rank.to_csv(os.path.join(out_dir, filename), index=False)
    print(f'Your "{filename}" is in the "{out_dir}" folder.')
    print("***\n")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    print(f"Execution time: {time.time() - start_time:.4f} seconds")

