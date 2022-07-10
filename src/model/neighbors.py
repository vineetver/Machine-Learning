from abc import ABC, abstractmethod
from typing import List, Any
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Model(ABC):
    """Abstract class for models."""

    def __init__(self, params: dict = None):
        """
        Initialize the model with the given features, label and the given parameters.

        Args:
            params: The parameters of the model to use.
        """
        if params is None:
            params = {}
        self.params = params
        self.model = None

    def preprocess(self):
        """ Any model specific preprocessing that needs to be done before training the model."""
        pass

    def split(self, X, Y, test_size: float):
        """Split the data into training and tests sets."""
        pass

    def normalize(self, X):
        """Normalize the data."""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict the labels for the given data."""
        pass

    @abstractmethod
    def evaluate(self, X, Y):
        """Evaluate the model."""
        pass

    def cross_validate(self, X, Y, n_splits: int = 10):
        """Cross validate the model."""
        pass

    def feature_importance(self):
        """Get the feature importance."""
        pass


class KNN(Model, ABC):
    """K-Nearest Neighbors model with Euclidean distance."""

    def __init__(self, params: dict = None):
        super().__init__(params if params is not None else {})

    def preprocess(self) -> tuple[Any, Any, Any, Any]:
        """This function reads the test and train data and returns x_train, y_train, x_test, y_test.

        Args:
            df: The dataframe containing the features and labels.

        Returns:
            X: The features.
            Y: The labels.
        """
        df_train = pd.read_csv('../data/train.csv')
        df_test = pd.read_csv('../data/test.csv')

        x_train = df_train[['mean_return', 'volatility']].values
        y_train = df_train['label'].values

        x_test = df_test[['mean_return', 'volatility']].values
        y_test = df_test['label'].values

        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train) -> KNeighborsClassifier:
        """Fit the model."""
        self.model = KNeighborsClassifier(**self.params)
        self.model.fit(x_train, y_train)
        return self.model

    def predict(self, x_test) -> List[int]:
        """Predict the labels for the given data."""
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test) -> float:
        """Evaluate the model."""
        y_pred = self.predict(x_test)
        print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
        return mean_squared_error(y_test, y_pred)

    def plot_best_k(self, x_train, y_train, x_test, y_test):
        """Plot the best k value."""
        error: list = []
        for k in range(3, 21, 2):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            error.append(mean_squared_error(y_test, y_pred))

        plt.figure(figsize=(10, 4))
        plt.plot(range(3, 21, 2), error, color='red', linestyle='dashed', marker='o', markerfacecolor='black', markersize=4)
        plt.title('Mean squared error vs K')
        plt.xlabel('Number of neighbors: k')
        plt.ylabel('Mean squared error')
        plt.show()


class Centroid(Model, ABC):
    """Nearest Centroid model"""

    def __init__(self, params: dict = None):
        super().__init__(params if params is not None else {})

    def preprocess(self) -> tuple[Any, Any, Any, Any]:
        """This function reads the test and train data and returns x_train, y_train, x_test, y_test.

        Args:
            df: The dataframe containing the features and labels.

        Returns:
            X: The features.
            Y: The labels.
        """
        df_train = pd.read_csv('../data/train.csv')
        df_test = pd.read_csv('../data/test.csv')

        x_train = df_train[['mean_return', 'volatility']].values
        y_train = df_train['label'].values

        x_test = df_test[['mean_return', 'volatility']].values
        y_test = df_test['label'].values

        return x_train, y_train, x_test, y_test

    def fit(self, x_train, y_train) -> NearestCentroid:
        """Fit the model."""
        self.model = NearestCentroid(**self.params)
        self.model.fit(x_train, y_train)
        return self.model

    def predict(self, x_test) -> List[int]:
        """Predict the labels for the given data."""
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test) -> float:
        """Evaluate the model."""
        y_pred = self.predict(x_test)
        print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
        return mean_squared_error(y_test, y_pred)
