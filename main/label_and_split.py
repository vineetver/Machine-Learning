"""
This is the main function to label the dataset and split the dataset into train and test.
"""
from src.dataset.create_dataset import label_data, test_train_split
import pandas as pd


def main():
    """
        We need to label the data, so we can use it for training. There are many ways to do this, but we will use the following:
        Label the datapoint as green(1) if the mean_return is geater than -0.05% and red(0) if the mean_return is smaller than -0.05%.
        The rule is simple and easy to understand. I can also come up with more complex rules, but I will not do that here.
    """
    # Load the data
    df = pd.read_csv('../data/ETH-USD_weekly_return_volatility_detailed.csv')

    # Label the dataset
    df = label_data(df)

    # Save the dataset
    df.to_csv('../data/ETH-USD_weekly_return_volatility_detailed_labeled.csv', index=False)

    # Split the dataset into train and test
    test_train_split(df)

    return None


if __name__ == '__main__':
    main()
