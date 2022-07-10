<h2 align='center'>Machine Learning Algorithms on Ethereum Data</h2>

## Task Description

This repository is to implement machine learning algorithms and their variational forms on Ethereum data obtained from
yahoos finance API.

| Algorithm                       |   MSE |
|---------------------------------|------:|
| K-Nearest Neighbors (Minkowski) | 0.037 |
| K-Nearest Neighbors (Manhattan) |     - |
| K-Nearest Neighbors (Euclidean) |     - |
| Nearest Centroid                | 0.066 |
| Domain Transformation          |     - |

## About the Data

The data is obtained from yahoos finance API and is stored in a CSV file.

Attributes:

- Date: Corresponding date of the stock price
- Year: Corresponding year of the stock price
- Month: Corresponding month of the stock price
- Day: Corresponding day of the stock price
- Weekday: Corresponding weekday of the stock price
- Open: Opening price of the stock price
- High: Highest price of the stock price
- Low: Lowest price of the stock price
- Close: Closing price of the stock price
- Volume: Stock volume
- Adj Close: Adjusted closing price of the stock price
- Return: Return of the stock
- Short_MA: Short moving average of the stock price
- Long_MA: Long moving average of the stock price

## Dependencies

  ```sh
$ pip install -r requirements.txt
````

or

  ```sh
$ pip install .
  ```

## Usage

  ```sh
  $ conda create -n "env-name" python=3.x, anaconda
 
  $ conda activate "env-name"
  
  $ cd Neural-Networks-Numpy
  
  $ jupyter notebook
  ```

## Roadmap

- [x] [KNN](https://github.com/vineetver/Machine-Learning/tree/main/Knn_and_variations)
    - [x] Domain Transformation
    - [x] Centroid
    - [x] Predicted Neighbors
    - [x] Original Points
    - [x] Minkowski
- [x] Naive Bayes
    - [ ] Gaussian
    - [ ] Student T
- [ ] Logsitic Regression
- [ ] Decision Tree
- [ ] Random Forest
- [ ] LDA
- [ ] QDA
- [ ] Ada Boost
    - [ ] SVM
    - [ ] Logistic Regression
    - [ ] Naive Bayesian
- [ ] KMeans

## License

Distributed under the MIT License. See `LICENSE.md` for more information.

## Contact

Vineet Verma - vineetver@hotmail.com - [Goodbyeweekend.io](https://www.goodbyeweekend.io/)
