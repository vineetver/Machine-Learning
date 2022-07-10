from src.model.neighbors import KNN
from src.dataset.profit import my_trading_strategy, buy_and_hold
import pandas as pd
import numpy as np


def minkowski(a, b, p):
    return np.linalg.norm(a - b, ord=p)


def main():
    """
    Metric:
        1. Minkowski -> lambda a, b: minkowski(a, b, p)
        2. Euclidean -> p = 2
        3. Manhattan -> p = 1
    """
    p = 1.5

    # load test set
    df_test = pd.read_csv('../data/test.csv')

    clf = KNN(params={'n_neighbors': 13, 'metric': lambda a, b: minkowski(a, b, p)})

    # Preprocess the data (split)
    x_train, y_train, x_test, y_test = clf.preprocess()

    # Use Elbow method to find the optimal number of neighbors from the graph
    clf.plot_best_k(x_train, y_train, x_test, y_test)

    # Train the model
    clf.fit(x_train, y_train)

    # Cross validate the model
    clf.cross_validate(x_train, y_train)

    # Predict the labels for the test set
    y_pred = clf.predict(x_test)

    # Evaluate trading strategies
    my_trading_strategy(df_test, labels=y_pred, eval='test')
    buy_and_hold()


if __name__ == '__main__':
    main()
