# Stock Price Prediction using Machine Learning

## Overview

This project implements a machine learning-based stock price prediction model using historical stock data and technical indicators. It employs three different algorithms—Linear Regression, Random Forest Regressor, and Support Vector Regression (SVR)—to predict future stock prices based on the input stock data. The goal is to assist traders and investors by providing insights and predictions to inform their investment strategies.

## Features

- **Data Collection:** Stock data is fetched using the `yfinance` API.
- **Feature Engineering:** Technical indicators such as Moving Averages (MA50, MA200), Relative Strength Index (RSI), and Price Change are calculated.
- **Modeling:** 
    - Linear Regression
    - Random Forest Regressor
    - Support Vector Regression (SVR)
- **Evaluation:** The model's performance is assessed using Mean Squared Error (MSE).
- **Visualization:** Actual vs. Predicted stock prices are visualized using `matplotlib`.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/stock-price-prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd stock-price-prediction
    ```

3. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main Python script:
    ```bash
    python stock_price_prediction.py
    ```

2. Input the stock symbol (e.g., AAPL, MSFT, GOOG), start date, and end date when prompted.

3. View the output of the predictions, model performance (MSE), and actual vs. predicted prices plotted on a graph.

## Requirements

- Python 3.x
- `yfinance`
- `sklearn`
- `ta`
- `matplotlib`
- `pandas`
- `numpy`

## Project Structure

```
stock-price-prediction/
├── stock_price_prediction.py  # Main script for running the stock prediction model
├── requirements.txt           # List of required Python libraries
└── README.md                  # Project overview and instructions
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- `yfinance` for easy access to stock data.
- `sklearn` for machine learning models.
- `matplotlib` for data visualization.
- `ta` for technical analysis indicators.
