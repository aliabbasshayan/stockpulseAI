import yfinance as yf
import pandas as pd
import numpy as np

def get_stock_data(ticker="AAPL", period="5y"):
    print(f"Fetching data for {ticker}...")
    
    # 1. Fetch raw historical data
    df = yf.download(ticker, period=period)
    
    # Multi-index column fix (yfinance issue handle karne ke liye)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Basic Data Cleaning & Understanding Checks
    df = df.dropna().drop_duplicates()
    
    print("\n--- Data Health Check ---")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    print(f"Initial Shape: {df.shape}")
    
    # 3. Technical Indicators (Features)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    # RSI Calculation (14 periods)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Target Variable Setup (Next Day Close)
    df['Target_Next_Close'] = df['Close'].shift(-1)
    
    # 5. Clean NaNs created by rolling indicators & shift
    df_clean = df.dropna()
    
    print("\n--- Final ML-Ready Dataset Overview ---")
    print(f"Final Shape: {df_clean.shape}")
    print("\nStatistical Summary:")
    print(df_clean[['Close', 'SMA_5', 'SMA_20', 'RSI', 'Daily_Return']].describe())
    
    print("\nFirst 5 Rows of ML Dataset:")
    print(df_clean[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'Target_Next_Close']].head())
    
    return df_clean

if __name__ == "__main__":
    # Run pipeline for AAPL
    data = get_stock_data("AAPL")