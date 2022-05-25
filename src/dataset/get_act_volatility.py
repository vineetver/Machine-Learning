import os
import numpy as np


def get_act_volatility(df, ticker):
    """
    Calculates the annualized volatility of the stock price and saves it to a csv file.
    :param df: Scraped DataFrame
    :param ticker: stock ticker
    :return: annualized volatility
    """
    path = os.path.join('data', ticker + '_weekly_return_volatility.csv')
    df['Return'] = 100.0 * df['Return']
    df['Return'] = df['Return'].round(3)
    df['Date'] = df.index
    df_2 = df[['Year', 'Week_Number', 'Return']]
    df_2.index = range(len(df))
    df_grouped = df_2.groupby(['Year', 'Week_Number'])['Return'].agg([np.mean, np.std])
    df_grouped.reset_index(['Year', 'Week_Number'], inplace=True)
    df_grouped.rename(columns={'mean': 'mean_return',
                               'std' : 'volatility'}, inplace=True)
    df_grouped.fillna(0, inplace=True)
    df_grouped.to_csv(path, index=False)
    print(f'wrote {len(df_grouped)} rows to {path}')

    path = os.path.join('data', ticker + '_weekly_return_volatility_detailed.csv')
    combined_df = df.merge(df_grouped, on=['Year', 'Week_Number'], how='inner')
    combined_df.to_csv(path, index=False)
    print(f'wrote {len(combined_df)} file to {path}')
