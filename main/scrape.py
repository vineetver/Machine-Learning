"""
Main function to scrape data from yfinance API and save it to a csv file.
:return: None
"""

from src.dataset.create_dataset import download_data, transform_data, write_data
from src.dataset.get_act_volatility import get_act_volatility

TICKER = 'ETH-USD'
START_DATE = '2016-01-01'
END_DATE = '2022-05-25'
S_WINDOW = 14
L_WINDOW = 50


def main():
    # Download data
    df = download_data(TICKER, START_DATE, END_DATE)

    # Clean and Transform data
    df = transform_data(df, S_WINDOW, L_WINDOW)

    # Get actual volatility and write to csv
    get_act_volatility(df, 'ETH-USD')

    # Write data to csv
    write_data(df, TICKER)

    return None


if __name__ == '__main__':
    main()
