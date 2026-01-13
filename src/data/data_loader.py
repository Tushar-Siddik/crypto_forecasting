# src/data/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class CryptoDataLoader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_crypto_data(self, ticker, start_date, end_date, save=True):
        """
        Fetch cryptocurrency data from Yahoo Finance
        
        Parameters:
        - ticker: str, cryptocurrency ticker (e.g., 'BTC-USD')
        - start_date: str, start date in 'YYYY-MM-DD' format
        - end_date: str, end date in 'YYYY-MM-DD' format
        - save: bool, whether to save the data to disk
        
        Returns:
        - DataFrame with OHLCV data
        """
        tick = yf.Ticker(str(ticker))

        try:
            data = tick.history(
                start=start_date,
                end=end_date,
            )

            if data.empty:
                raise ValueError(f"No data returned for {ticker}")

            if save:
                file_path = os.path.join(
                    self.data_dir,
                    f"{ticker}_{start_date}_{end_date}.csv"
                )
                data.to_csv(file_path)
                print(f"Data saved to {file_path}")

            return data

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def load_saved_data(self, ticker, start_date, end_date):
        """
        Load previously saved data
        
        Parameters:
        - ticker: str, cryptocurrency ticker
        - start_date: str, start date in 'YYYY-MM-DD' format
        - end_date: str, end date in 'YYYY-MM-DD' format
        
        Returns:
        - DataFrame with OHLCV data
        """
        file_path = os.path.join(self.data_dir, f"{ticker}_{start_date}_{end_date}.csv")
        
        if os.path.exists(file_path):
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            print(f"No saved data found for {ticker} from {start_date} to {end_date}")
            return None
    
    def get_latest_data(self, ticker, days=365):
        """
        Get the latest data for a cryptocurrency
        
        Parameters:
        - ticker: str, cryptocurrency ticker
        - days: int, number of days of data to fetch
        
        Returns:
        - DataFrame with OHLCV data
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.fetch_crypto_data(ticker, start_date, end_date)