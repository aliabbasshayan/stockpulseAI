import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from Phase1 import get_stock_data  # Capital P import

def train_stock_model(ticker="AAPL"):
    # 1. Load ML-ready data from Phase 1
    df = get_stock_data(ticker)
    
    # 2. Features (X) & Target (y)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'Daily_Return']
    target = 'Target_Next_Close'
    
    X = df[features]
    y = df[target]
    
    # 3. Chronological Train-Test Split (80% Train, 20% Test)
    train_size = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"\n--- Data Split Overview ---")
    print(f"Total Rows: {len(df)}")
    print(f"Training Rows (80%): {len(X_train)}")
    print(f"Testing Rows (20%): {len(X_test)}")
    
    # 4. Initialize & Train Linear/Ridge Regression Model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # 5. Model Evaluation
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE) : ${mae:.2f}")
    print(f"Mean Squared Error (MSE)  : {mse:.2f}")
    print(f"Root Mean Sq. Error (RMSE): ${rmse:.2f}")
    print(f"R² Score (Accuracy)       : {r2:.4f}")
    
    # 6. Prediction
    latest_features = X.tail(1)
    next_day_pred = model.predict(latest_features)[0]
    current_close = latest_features['Close'].values[0]
    
    movement = "📈 UP" if next_day_pred > current_close else "📉 DOWN"
    pct_change = ((next_day_pred - current_close) / current_close) * 100
    
    print("\n--- Tomorrow's Prediction ---")
    print(f"Current Price    : ${current_close:.2f}")
    print(f"Predicted Price  : ${next_day_pred:.2f}")
    print(f"Expected Direction: {movement} ({pct_change:+.2f}%)")
    
    # 7. Save Model
    os.makedirs("model", exist_ok=True)
    model_path = "model/stock_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully at '{model_path}'")
    
    return model

if __name__ == "__main__":
    train_stock_model("AAPL")