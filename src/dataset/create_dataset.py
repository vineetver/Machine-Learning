import os
import pandas as pd
import yfinance as yf


def download_data(ticker, start_date, end_date):
    """
    This function downloads the historical data for the given ticker/stock
    :param ticker: the ticker/stock symbol
    :param start_date: the start date for the data
    :param end_date: the end date for the data
    :return: a dataframe with the historical data
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    return df


def transform_data(df, s_window, l_window):
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


def write_data(df, ticker):
    """
    This function writes the cleaned dataframe to a csv file
    :param ticker: the ticker/stock symbol
    :param df: the cleaned dataframe
    :return: None
    """
    path = os.path.join('data', ticker + '.csv')
    df.to_csv(path, index=False)
    print(f'wrote {len(df)} lines to file: {path}')


def label_data(df):
    """
    This function labels the dataframe for model training.
    For example, if the mean return is less than 0 then the label is 0
    :param df: The detailed weekly return volatility dataframe
    :return: the labeled dataframe
    """
    df['label'] = df['mean_return'].apply(lambda x: 1 if x > -0.1 else 0)
    return df
