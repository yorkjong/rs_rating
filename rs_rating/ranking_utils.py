"""
Utilities for Ranking tables
"""
__author__ = "York <york.jong@gmail.com>"
__date__ = "2024/10/06 (initial version) ~ 2024/10/07 (last revision)"

__all__ = [
    'append_percentile',
    'groupby_industry',
]
import pandas as pd


#------------------------------------------------------------------------------

def append_percentile(stock_df, columns, method='rank'):
    """
    Calculates and appends percentile rankings to the stock DataFrame for RS
    values and their historical comparisons.

    Parameters
    ----------
    stock_df: pd.DataFrame
        DataFrame containing stock data with RS values to calculate percentiles.

    columns: list of str
        A list of column names for which to calculate and append percentile
        rankings.

    method: str, optional
        Method to calculate percentiles. Either 'rank' or 'qcut'. Defaults to
        'rank'.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional percentile columns for specified
        RS values.
    """
    for col in columns:
        stock_df[f'Pctl ({col})'] = calc_percentile(stock_df[col],
                                                    method)
    return stock_df


def calc_percentile(series, method='rank'):
    """
    Calculate percentiles for a given Pandas Series.

    Parameters
    ----------
    series: pd.Series
        The input data series for which to calculate percentiles.
    method: str, optional
        The method to use for calculating percentiles.
        Either 'rank' (default) for rank-based percentiles or
        'qcut' for quantile-based percentiles.

    Returns
    -------
    pd.Series
        A series of nullable integer percentile values corresponding to the
        input series, ranging from 1 to 99.

    Raises
    ------
    ValueError
        If the method is not 'rank' or 'qcut'.
    """
    if method == 'rank':
        percentiles = series.rank(pct=True).mul(98) + 1
    elif method == 'qcut':
        percentiles = pd.qcut(series, 99, labels=False, duplicates='drop') + 1
    else:
        raise ValueError("method must be either 'rank' or 'qcut'")
    return percentiles.round().astype('Int64')  # Use Int64 to allow NaN

#------------------------------------------------------------------------------

def groupby_industry(stock_df, columns, key='RS'):
    """
    Groups the stock DataFrame by industry and performs aggregation only on
    specified columns, with automatic handling of column types for aggregation
    (e.g., 'Sector' uses 'first', numeric columns use mean).

    Parameters
    ----------
    stock_df: pd.DataFrame
        DataFrame containing stock data with industry and other columns.

    columns: list of str
        A list of columns to include in the aggregation.

    key: str, optional
        The column name used for sorting items (e.g., 'Ticker' or 'Name').
        Defaults to 'RS'.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame grouped by industry.
    """
    def get_sorted_items(items):
        """Sorts items (e.g., Tickers or Names) based on RS values."""
        return ','.join(
            sorted(items,
                   key=lambda t:
                        stock_df.loc[stock_df['Ticker'] == t, key].values[0],
                   reverse=True)
        )

    agg_funcs = {}

    # Process only the specified columns in columns
    for col in columns:
        if col == 'Ticker' or col == 'Name':
            agg_funcs[col] = get_sorted_items
        elif pd.api.types.is_numeric_dtype(stock_df[col]):
            agg_funcs[col] = lambda x: round(x.mean(), 2)
        else:
            agg_funcs[col] = 'first'

    # Perform aggregation
    industry_df = stock_df.groupby('Industry').agg(agg_funcs).reset_index()

    return industry_df

#------------------------------------------------------------------------------

