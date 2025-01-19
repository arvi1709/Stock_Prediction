import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore')  

def validate_date(date_text):
    """Validates date input format."""
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def add_features(data):
    """Adds technical indicators and features to stock data."""
    data['MA50'] = data['Close'].rolling(window=50).mean()  # 50-day moving average
    data['MA200'] = data['Close'].rolling(window=200).mean()  # 200-day moving average
    rsi = RSIIndicator(data['Close'], window=14).rsi()  # Relative Strength Index
    data['RSI'] = rsi
    data['Price_Change'] = data['Close'].pct_change()  # Percentage change in price
    return data.dropna()  # Remove rows with missing values

print("Welcome to the Stock Prediction Model!")
print("Enter a single Company stock symbol (e.g., AAPL, MSFT, GOOG):")

company = input("Enter stock symbol: ").strip().upper()

# Date input and validation
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")

if not (validate_date(start_date) and validate_date(end_date)):
    print("Invalid date format. Please use YYYY-MM-DD.")
    exit()

print(f"\nDownloading stock data for {company} from YFinance...")

# Download stock data
try:
    stock_data = yf.download(company, start=start_date, end=end_date)
except Exception as e:
    print(f"Error downloading data for {company}.")
    exit()

if stock_data.empty:
    print(f"No data available for {company}.")
    exit()

# Feature engineering
company_data = add_features(stock_data)

# Prepare target (shifted closing prices)
company_data['Prediction'] = company_data['Close'].shift(-30)
company_data = company_data.dropna()  # Drop rows with missing values after shifting

# Prepare features and target for modeling
X = company_data[['Close', 'Volume', 'MA50', 'MA200', 'RSI', 'Price_Change']].values
y = company_data['Prediction'].values

if len(X) < 2:
    print(f"Not enough data for {company}, exiting...")
    exit()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Model 3: Support Vector Regression (SVR)
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(X_test)

# Evaluate model performance
lr_mse = mean_squared_error(y_test, lr_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
svr_mse = mean_squared_error(y_test, svr_predictions)

# Display model performance
print(f"\nModel Performance:")
print(f"  Linear Regression MSE: {lr_mse:.2f}")
print(f"  Random Forest MSE: {rf_mse:.2f}")
print(f"  SVR MSE: {svr_mse:.2f}")

# Show first 3 predictions
print(f"\nPredictions for {company} (showing the first 3 predictions):\n")

for i in range(3):
    print(f"Prediction {i+1}:")
    print(f"  Linear Regression Prediction: ${lr_predictions[i]:.2f}")
    print(f"  Random Forest Prediction: ${rf_predictions[i]:.2f}")
    print(f"  SVR Prediction: ${svr_predictions[i]:.2f}")
    print("-" * 40)

# Plot actual vs predicted prices
plt.figure(figsize=(10,  6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(lr_predictions, label='LR Predictions', linestyle='--', color='green')
plt.plot(rf_predictions, label='RF Predictions', linestyle='--', color='orange')
plt.plot(svr_predictions, label='SVR Predictions', linestyle='--', color='red')
plt.title(f"Actual vs Predicted Prices for {company}")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# Compare all three models in a single plot
plt.figure(figsize=(12, 8))
plt.plot(y_test, label='Actual Prices', color='blue', linewidth=2)
plt.plot(lr_predictions, label='Linear Regression Predictions', linestyle='--', color='green', linewidth=2)
plt.plot(rf_predictions, label='Random Forest Predictions', linestyle='--', color='orange', linewidth=2)
plt.plot(svr_predictions, label='SVR Predictions', linestyle='--', color='red', linewidth=2)
plt.title(f"Comparison of Model Predictions for {company}")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.grid()
plt.show()

print("\nThank you!")