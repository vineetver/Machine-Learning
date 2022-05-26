import os
import pandas as pd
import yfinance as yf
import numpy as np


def download_data(ticker: str = 'ETH-USD', start_date: str = '2016-01-01', end_date: str = '2022-05-25') -> pd.DataFrame:
    """
    This function downloads the historical data for the given ticker/stock
    :param ticker: the ticker/stock symbol
    :param start_date: the start date for the data
    :param end_date: the end date for the data
    :return: a dataframe with the historical data
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    return df


def transform_data(df: pd.DataFrame, s_window: int = 14, l_window: int = 50) -> pd.DataFrame:
    """
    This function cleans the dataframe and adds some new features
    :param df: the dataframe to clean
    :param s_window: the short window for the rolling mean
    :param l_window: the long window for the rolling mean
    :return: the cleaned dataframe
    """
    df['Return'] = df['Adj Close'].pct_change(
        periods=1, fill_method='ffill')
    df['Return'].fillna(0, inplace=True)
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        df[col] = df[col].round(2)
    df['Weekday'] = df['Date'].dt.day_name()
    df['Week_Number'] = df['Date'].dt.strftime('%U')
    df['Week_Number'] = df['Week_Number'].astype(int) + 1
    df['Year_Week'] = df['Date'].dt.strftime('%Y-%U')
    df['Short_MA'] = df['Adj Close'].rolling(window=s_window, min_periods=1).mean()
    df['Long_MA'] = df['Adj Close'].rolling(window=l_window, min_periods=1).mean()
    col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday',
                'Week_Number', 'Year_Week', 'Open',
                'High', 'Low', 'Close', 'Volume', 'Adj Close',
                'Return', 'Short_MA', 'Long_MA']
    return df[col_list]


def write_data(df: pd.DataFrame, ticker: str) -> None:
    """
    This function writes the cleaned dataframe to a csv file
    :param ticker: the ticker/stock symbol
    :param df: the cleaned dataframe
    :return: None
    """
    path = os.path.join('data', ticker + '.csv')
    df.to_csv(path, index=False)
    print(f'wrote {len(df)} lines to file: {path}')


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function labels the dataframe for model training.
    For example, if the mean return is less than 0 then the label is 0
    Labels based on mean_return of the week. This is because the mean_return is the average return of the week and I want to trade on the
    weeks that are green.
    :param df: The detailed weekly return volatility dataframe
    :return: the labeled dataframe
    """
    df['label'] = df['mean_return'].apply(lambda x: 1 if x > -0.1 else 0)
    return df


def test_train_split(df: pd.DataFrame) -> None:
    """
    This function splits the dataframe into a training and testing dataframe and saves the train and test to csv. Data from 2018 and 2019 as the training set, and the data from
    2020 and 2021 as the testing set. The columns used are: 'Year', 'Week_Number', 'Return', 'label', 'Adj Close'.
    :param df: the dataframe to split
    :return: None
    """
    df_train = df[(df['Year'] == 2018) | (df['Year'] == 2019)]
    df_test = df[(df['Year'] == 2020) | (df['Year'] == 2021)]

    df_train = df_train[['Year', 'Week_Number', 'Return', 'label', 'Adj Close']]
    df_test = df_test[['Year', 'Week_Number', 'Return', 'label', 'Adj Close']]

    df_train_gp = df_train.groupby(['Year', 'Week_Number', 'label'])[['Return', 'Adj Close']].agg([np.mean, np.std])
    df_train_gp.reset_index(['Year', 'Week_Number', 'label'], inplace=True)
    df_train_gp.columns = ['Year', 'Week_Number', 'label', 'mean_return', 'volatility', 'mean_adj_close', 'std_price']
    df_train_gp.drop(['std_price'], axis=1, inplace=True)
    df_train_gp.fillna(0, inplace=True)

    df_train_gp.to_csv('../data/train.csv', index=False)

    df_test_gp = df_test.groupby(['Year', 'Week_Number', 'label'])[['Return', 'Adj Close']].agg([np.mean, np.std])
    df_test_gp.reset_index(['Year', 'Week_Number', 'label'], inplace=True)
    df_test_gp.columns = ['Year', 'Week_Number', 'label', 'mean_return', 'volatility', 'mean_adj_close', 'std_price']
    df_test_gp.drop(['std_price'], axis=1, inplace=True)
    df_test_gp.fillna(0, inplace=True)

    df_test_gp.to_csv('../data/test.csv', index=False)
