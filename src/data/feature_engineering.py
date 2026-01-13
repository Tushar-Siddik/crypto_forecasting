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
        # Momentum indicators measure the speed or strength of 
        # price movements, helping to identify overbought or oversold conditions.

        data['rsi'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
        # RSI (Relative Strength Index): Measures how strong recent gains/losses are.
        # Range: 0–100.
        # 70 → Overbought (possible sell signal), <30 → Oversold (possible buy signal).

        data['stoch'] = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close']).stoch()
        data['stoch_signal'] = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close']).stoch_signal()
        # Stochastic Oscillator:
        # Compares current price to price range over a period.
        # %K (stoch) → raw oscillator, %D (stoch_signal) → smoothed signal line.
        # Helps identify overbought/oversold conditions similar to RSI.

        # Add trend indicators
        # Trend indicators help detect direction and strength of a trend.
        data['macd'] = ta.trend.MACD(close=data['Close']).macd()
        data['macd_signal'] = ta.trend.MACD(close=data['Close']).macd_signal()
        data['macd_diff'] = ta.trend.MACD(close=data['Close']).macd_diff()
        # MACD (Moving Average Convergence Divergence):
        # Difference between fast EMA and slow EMA.
        # macd_signal → EMA of MACD, used for crossovers.
        # macd_diff → MACD - signal line, shows momentum.
        # Used to spot trend reversals.
        
        data['adx'] = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close']).adx()
        # ADX (Average Directional Index):
        # Measures trend strength, not direction.
        # 25 → strong trend, <20 → weak or sideways market.

        data['cci'] = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close']).cci()
        # CCI (Commodity Channel Index):
        # Measures price deviation from its moving average.
        # Positive → price above mean, Negative → price below mean.
        # Helps identify overbought/oversold or trend extremes.


        # Add volatility indicators
        # Volatility indicators show how much price moves, which helps assess risk and potential breakout points.
        data['bollinger_hband'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_hband()
        data['bollinger_lband'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_lband()
        data['bollinger_mband'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_mavg()
        data['bollinger_width'] = data['bollinger_hband'] - data['bollinger_lband']
        # Bollinger Bands:
        # Upper/Lower bands ± standard deviations from moving average.
        # bollinger_width → measures volatility.
        # Price near upper band → overbought, near lower → oversold.
        
        data['atr'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()
        # ATR (Average True Range):
        # Measures average price range over a period.
        # Higher ATR → more volatility.
        # Used for stop-loss levels or risk management.


        # Add volume indicators
        # Volume indicators incorporate trading activity, which confirms price trends.
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
        # OBV (On-Balance Volume):
        # Cumulative volume that adds/subtracts based on price movement.
        # Rising OBV → buyers dominate, Falling OBV → sellers dominate.
        
        data['vwap'] = ta.volume.VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']).volume_weighted_average_price()
        # VWAP (Volume Weighted Average Price):
        # Average price weighted by volume.
        # Used by traders to gauge fair value intraday.

        # Add other features
        data['price_change'] = data['Close'].pct_change()
        # Price Change %: Day-to-day relative change.

        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        # Log Return: Continuously compounded return, good for statistical analysis.

        data['volatility'] = data['log_return'].rolling(window=14).std()
        # Rolling standard deviation of log returns → historical volatility.

        # Add time-based features
        # Time features help models capture seasonality patterns.
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