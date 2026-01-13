# src/data/feature_engineering.py
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Add momentum indicators
        data['rsi'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
        data['stoch'] = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close']).stoch()
        data['stoch_signal'] = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close']).stoch_signal()
        
        # Add trend indicators
        data['macd'] = ta.trend.MACD(close=data['Close']).macd()
        data['macd_signal'] = ta.trend.MACD(close=data['Close']).macd_signal()
        data['macd_diff'] = ta.trend.MACD(close=data['Close']).macd_diff()
        data['adx'] = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close']).adx()
        data['cci'] = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close']).cci()
        
        # Add volatility indicators
        data['bollinger_hband'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_hband()
        data['bollinger_lband'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_lband()
        data['bollinger_mband'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_mavg()
        data['bollinger_width'] = data['bollinger_hband'] - data['bollinger_lband']
        data['atr'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()
        
        # Add volume indicators
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
        data['vwap'] = ta.volume.VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']).volume_weighted_average_price()
        
        # Add other features
        data['price_change'] = data['Close'].pct_change()
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['log_return'].rolling(window=14).std()
        
        # Add time-based features
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        
        # Drop rows with NaN values
        data = data.dropna()
        
        return data
    
    def create_sequences(self, data, sequence_length, target_col='Close', feature_cols=None):
        """
        Create sequences for time series modeling
        
        Parameters:
        - data: DataFrame with features
        - sequence_length: int, length of input sequences
        - target_col: str, column to predict
        - feature_cols: list, columns to use as features
        
        Returns:
        - X: numpy array of input sequences
        - y: numpy array of target values
        """
        if feature_cols is None:
            feature_cols = data.columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)
        
        # Extract features and target
        features = data[feature_cols].values
        target = data[target_col].values
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, data, feature_cols=None, fit=True):
        """
        Normalize data using MinMaxScaler
        
        Parameters:
        - data: DataFrame or numpy array
        - feature_cols: list, columns to normalize
        - fit: bool, whether to fit the scaler
        
        Returns:
        - normalized_data: numpy array
        """
        if isinstance(data, pd.DataFrame):
            if feature_cols is None:
                feature_cols = data.columns.tolist()
            data_to_normalize = data[feature_cols].values
        else:
            data_to_normalize = data
        
        if fit:
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data_to_normalize)
            self.scalers['default'] = scaler
        else:
            if 'default' not in self.scalers:
                raise ValueError("No scaler found. Please fit the scaler first.")
            normalized_data = self.scalers['default'].transform(data_to_normalize)
        
        return normalized_data
    
    def inverse_transform(self, data):
        """
        Inverse transform normalized data
        
        Parameters:
        - data: numpy array
        
        Returns:
        - original_data: numpy array
        """
        if 'default' not in self.scalers:
            raise ValueError("No scaler found. Please fit the scaler first.")
        
        return self.scalers['default'].inverse_transform(data)