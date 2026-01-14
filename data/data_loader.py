# data/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
import os
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CryptoDataLoader:
    """Enhanced data loader for cryptocurrency data"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_crypto_data(self, ticker: str, start_date: str, end_date: str, save: bool = True) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Yahoo Finance
        
        Args:
            ticker: Cryptocurrency ticker (e.g., 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            save: Whether to save the data to disk
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {ticker} data from {start_date} to {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Ensure single-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            # Add ticker column
            data['Ticker'] = ticker

            # Ensure Close is a Series, not DataFrame
            if isinstance(data['Close'], pd.DataFrame):
                data['Close'] = data['Close'].iloc[:, 0]
        
            
            if save:
                file_path = os.path.join(self.data_dir, f"{ticker}_{start_date}_{end_date}.csv")
                data.to_csv(file_path)
                logger.info(f"Data saved to {file_path}")
            
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def load_saved_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load previously saved data
        
        Args:
            ticker: Cryptocurrency ticker
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = os.path.join(self.data_dir, f"{ticker}_{start_date}_{end_date}.csv")
        
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded saved data for {ticker} from {file_path}")
            return data
        else:
            logger.warning(f"No saved data found for {ticker} from {start_date} to {end_date}")
            return pd.DataFrame()
    
    def get_latest_data(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        Get the latest data for a cryptocurrency
        
        Args:
            ticker: Cryptocurrency ticker
            days: Number of days of data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Try to load saved data first
        data = self.load_saved_data(ticker, start_date, end_date)
        
        # If no saved data or data is outdated, fetch new data
        if data.empty or (datetime.now() - data.index[-1]).days > 1:
            data = self.fetch_crypto_data(ticker, start_date, end_date)
        
        return data
    
    def fetch_multiple_tickers(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers
        
        Args:
            tickers: List of cryptocurrency tickers
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary with tickers as keys and DataFrames as values
        """
        data_dict = {}
        
        for ticker in tickers:
            data = self.fetch_crypto_data(ticker, start_date, end_date)
            if not data.empty:
                data_dict[ticker] = data
        
        return data_dict
    
    def get_market_data(self, tickers: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Get market data for multiple tickers
        
        Args:
            tickers: List of cryptocurrency tickers
            days: Number of days of data to fetch
            
        Returns:
            Dictionary with tickers as keys and DataFrames as values
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.fetch_multiple_tickers(tickers, start_date, end_date)