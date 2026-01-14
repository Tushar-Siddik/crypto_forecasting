# data/feature_engineering.py
import pandas as pd
import numpy as np
import ta
from typing import List, Optional, Tuple
import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for cryptocurrency data"""
    
    def __init__(self):
        self.feature_columns = None
        self.target_column = 'Close'
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        try:
            # Price-based features
            data['Price_Change'] = data['Close'].pct_change()
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Close_Open_Ratio'] = data['Close'] / data['Open']
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
                data[f'EMA_{window}'] = data['Close'].ewm(span=window).mean()
            
            # Moving average crossovers
            data['SMA_5_20_Cross'] = data['SMA_5'] / data['SMA_20'] - 1
            data['SMA_20_50_Cross'] = data['SMA_20'] / data['SMA_50'] - 1
            
            # Momentum indicators
            data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
            data['Stoch_K'] = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close']).stoch()
            data['Stoch_D'] = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close']).stoch_signal()
            data['Williams_R'] = ta.momentum.WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close']).williams_r()
            data['UO'] = ta.momentum.UltimateOscillator(high=data['High'], low=data['Low'], close=data['Close']).ultimate_oscillator()
            
            # Trend indicators
            macd = ta.trend.MACD(close=data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Diff'] = macd.macd_diff()
            data['ADX'] = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close']).adx()
            data['CCI'] = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close']).cci()
            data['DMI_Pos'] = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close']).adx_pos()
            data['DMI_Neg'] = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close']).adx_neg()
            data['AROON_Up'] = ta.trend.AroonIndicator(high=data['High'], low=data['Low']).aroon_up()
            data['AROON_Down'] = ta.trend.AroonIndicator(high=data['High'], low=data['Low']).aroon_down()
            
            # Volatility indicators
            bollinger = ta.volatility.BollingerBands(close=data['Close'])
            data['BB_Upper'] = bollinger.bollinger_hband()
            data['BB_Middle'] = bollinger.bollinger_mavg()
            data['BB_Lower'] = bollinger.bollinger_lband()
            data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
            data['ATR'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()
            data['Keltner_High'] = ta.volatility.KeltnerChannel(high=data['High'], low=data['Low'], close=data['Close']).keltner_channel_hband()
            data['Keltner_Low'] = ta.volatility.KeltnerChannel(high=data['High'], low=data['Low'], close=data['Close']).keltner_channel_lband()
            
            # Donchian Channel
            donchian = ta.volatility.DonchianChannel(high=data['High'], low=data['Low'], close=data['Close'])
            data['Donchian_High'] = donchian.donchian_channel_hband()
            data['Donchian_Low'] = donchian.donchian_channel_lband()

            # data['Donchian_High'] = ta.volatility.DonchianChannel(high=data['High'], low=data['Low']).donchian_channel_hband()
            # data['Donchian_Low'] = ta.volatility.DonchianChannel(high=data['High'], low=data['Low']).donchian_channel_lband()
            
            # # Donchian Channel - added close parameter
            # donchian = ta.volatility.DonchianChannel(high=data['High'], low=data['Low'], close=data['Close'])
            # data['Donchian_High'] = donchian.donchian_channel_hband()
            # data['Donchian_Low'] = donchian.donchian_channel_lband()
            
            # Volume indicators (if Volume column exists)
            if 'Volume' in data.columns:
                data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
                data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
                data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']).volume_weighted_average_price()
                data['ADI'] = ta.volume.AccDistIndexIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']).acc_dist_index()
                data['MFI'] = ta.volume.MFIIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']).money_flow_index()
                data['FI'] = ta.volume.ForceIndexIndicator(close=data['Close'], volume=data['Volume']).force_index()
                data['EMV'] = ta.volume.EaseOfMovementIndicator(high=data['High'], low=data['Low'], volume=data['Volume']).ease_of_movement()
                data['VPT'] = ta.volume.VolumePriceTrendIndicator(close=data['Close'], volume=data['Volume']).volume_price_trend()
                data['NVI'] = ta.volume.NegativeVolumeIndexIndicator(close=data['Close'], volume=data['Volume']).negative_volume_index()
                
                # Manual implementation of Positive Volume Index
                def positive_volume_index(close, volume):
                    pvi = pd.Series(index=close.index, dtype=float)
                    pvi.iloc[0] = 100  # Starting value
                    
                    for i in range(1, len(close)):
                        if volume.iloc[i] > volume.iloc[i-1]:
                            pvi.iloc[i] = pvi.iloc[i-1] * (1 + (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1])
                        else:
                            pvi.iloc[i] = pvi.iloc[i-1]
                    
                    return pvi
                                
                data['PVI'] = positive_volume_index(close=data['Close'], volume=data['Volume'])
            
            # Statistical features
            data['Volatility'] = data['Log_Return'].rolling(window=20).std()
            data['Skewness'] = data['Log_Return'].rolling(window=20).skew()
            data['Kurtosis'] = data['Log_Return'].rolling(window=20).kurt()
            
            # Time-based features
            data['Day_of_Week'] = data.index.dayofweek
            data['Month'] = data.index.month
            data['Quarter'] = data.index.quarter
            data['Day_of_Month'] = data.index.day
            data['Week_of_Year'] = data.index.isocalendar().week
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag) if 'Volume' in data.columns else np.nan
            
            # Rolling window statistics
            for window in [5, 10, 20]:
                data[f'Close_Mean_{window}'] = data['Close'].rolling(window=window).mean()
                data[f'Close_Std_{window}'] = data['Close'].rolling(window=window).std()
                data[f'Close_Max_{window}'] = data['Close'].rolling(window=window).max()
                data[f'Close_Min_{window}'] = data['Close'].rolling(window=window).min()
                data[f'Close_Range_{window}'] = data[f'Close_Max_{window}'] - data[f'Close_Min_{window}']
                
                if 'Volume' in data.columns:
                    data[f'Volume_Mean_{window}'] = data['Volume'].rolling(window=window).mean()
                    data[f'Volume_Std_{window}'] = data['Volume'].rolling(window=window).std()
            
            # Drop rows with NaN values
            data = data.dropna()
            
            # Define feature columns (all columns except the target and non-numeric columns)
            self.feature_columns = [col for col in data.columns if col != self.target_column and data[col].dtype != 'object']
            
            logger.info(f"Added {len(self.feature_columns)} features to the dataset")
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            # Return the original data if there's an error
            return df
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int, 
                        prediction_horizon: int = 1, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling
        
        Args:
            data: DataFrame with features
            sequence_length: Length of input sequences
            prediction_horizon: Number of days ahead to predict
            target_col: Column to predict
            
        Returns:
            Tuple of (X, y) arrays
        """
        if self.feature_columns is None:
            raise ValueError("Feature columns not defined. Run add_technical_indicators first.")
        
        # Extract features and target
        features = data[self.feature_columns].values
        target = data[target_col].values
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input: sequence_length days of all features
            X.append(features[i:i+sequence_length])
            
            # Target: prediction_horizon days ahead
            if prediction_horizon == 1:
                y.append(target[i+sequence_length])
            else:
                y.append(target[i+sequence_length:i+sequence_length+prediction_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y
    
    def select_features(self, data: pd.DataFrame, method: str = 'correlation', 
                       top_k: int = 50) -> List[str]:
        """
        Select the most important features
        
        Args:
            data: DataFrame with features
            method: Feature selection method ('correlation', 'mutual_info', 'variance')
            top_k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        if self.feature_columns is None:
            raise ValueError("Feature columns not defined. Run add_technical_indicators first.")
        
        if method == 'correlation':
            # Select features with highest correlation to target
            correlations = data[self.feature_columns].corrwith(data[self.target_column]).abs()
            selected_features = correlations.nlargest(top_k).index.tolist()
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            # Select features with highest mutual information
            mi_scores = mutual_info_regression(data[self.feature_columns], data[self.target_column])
            mi_scores = pd.Series(mi_scores, index=self.feature_columns)
            selected_features = mi_scores.nlargest(top_k).index.tolist()
        
        elif method == 'variance':
            # Select features with highest variance
            variances = data[self.feature_columns].var()
            selected_features = variances.nlargest(top_k).index.tolist()
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.feature_columns = selected_features
        logger.info(f"Selected {len(selected_features)} features using {method} method")
        
        return selected_features