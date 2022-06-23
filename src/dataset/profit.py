import pandas as pd


def my_trading_strategy(df, **kwargs):
    """
    This function calculates the trading strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data.

    Returns
    -------
    profit : pandas.Series
        Series containing the trading strategy profit.
    """

    label = kwargs['labels']
    eval = kwargs['eval']

    df['labels'] = label

    initial: int('Stock') = 0
    cash: int('Starting Money') = 100
    total_profit: int = 0
    total_loss: int = 0
    success: int = 0
    fail: int = 0

    if eval == 'test':
        for i in range(len(df)):
            """
            Start buying 
            """
            if (df['labels'].iloc[i] == 1 and initial == 0):
                initial = cash / df['mean_adj_close'].iloc[i]

            elif (df['labels'].iloc[i] != 1 and initial != 0):
                sell = initial * df['mean_adj_close'].iloc[i]
                if (sell > cash):
                    profit = sell - cash
                    success += 1
                    total_profit += profit
                else:
                    loss = cash - sell
                    fail += 1
                    total_loss += loss
                cash = sell
                initial = 0

        print(f'Number of successful trades: {success}')
        print(f'Number of failed trades: {fail}')
        print(f'My trading strategy profit: {round(total_profit)}')

        return total_profit

    elif eval == 'train':
        for i in range(len(df)):
            """
            Start buying 
            """

            if (df['label'].iloc[i] == 1 and initial == 0):
                initial = cash / df['mean_adj_close'].iloc[i]

            elif (df['label'].iloc[i] != 1 and initial != 0):
                sell = initial * df['mean_adj_close'].iloc[i]
                if (sell > cash):
                    profit = sell - cash
                    success += 1
                    total_profit += profit
                else:
                    loss = cash - sell
                    fail += 1
                    total_loss += loss
                cash = sell
                initial = 0

        print(f'Number of successful trades: {success}')
        print(f'Number of failed trades: {fail}')
        print(f'My trading strategy profit: {round(total_profit)}')
        return total_profit


def buy_and_hold():
    """
    This function calculates the buy and hold strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data.

    Returns
    -------
    profit : pandas.Series
        Series containing the buy and hold strategy profit.
    """
    df = pd.read_csv('../data/test.csv')

    # Buy on the first day of 2020
    initial = 100 / df['mean_adj_close'].loc[df['Year'] == 2020].iloc[0]
    # Sell on the last day of 2021
    cash = df['mean_adj_close'].loc[df['Year'] == 2021].iloc[-1] * initial

    profit = cash
    print(f'Buy and hold strategy profit: {round(profit)}')
    return profit
