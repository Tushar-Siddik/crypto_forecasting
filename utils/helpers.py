# utils/helpers.py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results(results: Dict[str, Any], file_path: str) -> None:
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    with open(file_path, 'w') as f:
        json.dump(results, f, default=convert_numpy, indent=4)

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_timestamp() -> str:
    """Create timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def calculate_prediction_intervals(predictions: np.ndarray, 
                                std_error: float,
                                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals
    
    Args:
        predictions: Predicted values
        std_error: Standard error of predictions
        confidence_level: Confidence level (0-1)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)
    
    margin_of_error = z_score * std_error
    
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    
    return lower_bound, upper_bound

def detect_outliers(data: np.ndarray, method: str = 'iqr', 
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data
    
    Args:
        data: Input data
        method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > threshold
    
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        
        clf = IsolationForest(contamination=0.1, random_state=42)
        outliers = clf.fit_predict(data.reshape(-1, 1)) == -1
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outliers

def smooth_data(data: np.ndarray, method: str = 'moving_average', 
               window_size: int = 5) -> np.ndarray:
    """
    Smooth data using various methods
    
    Args:
        data: Input data
        method: Smoothing method ('moving_average', 'exponential', 'savgol')
        window_size: Window size for smoothing
        
    Returns:
        Smoothed data
    """
    if method == 'moving_average':
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
    elif method == 'exponential':
        alpha = 2 / (window_size + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(data, window_size, 3)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed

def calculate_returns(prices: np.ndarray, method: str = 'simple') -> np.ndarray:
    """
    Calculate returns from prices
    
    Args:
        prices: Price series
        method: Return calculation method ('simple', 'log')
        
    Returns:
        Returns series
    """
    # Ensure prices is 1D
    if prices.ndim > 1:
        prices = prices.flatten()
    
    if method == 'simple':
        returns = np.diff(prices) / prices[:-1]
    elif method == 'log':
        returns = np.log(prices[1:] / prices[:-1])
    else:
        raise ValueError(f"Unknown return calculation method: {method}")
    
    return returns

def calculate_volatility(returns: np.ndarray, window_size: int = 30) -> np.ndarray:
    """
    Calculate rolling volatility
    
    Args:
        returns: Return series
        window_size: Window size for rolling calculation
        
    Returns:
        Rolling volatility
    """
    # Ensure returns is 1D
    if returns.ndim > 1:
        returns = returns.flatten()
    
    # Calculate rolling volatility
    volatility = pd.Series(returns).rolling(window=window_size).std().values
    
    # Remove NaN values at the beginning
    volatility = volatility[window_size-1:]
    
    return volatility

def format_large_number(number: float, precision: int = 2) -> str:
    """Format large numbers with appropriate suffixes"""
    if number >= 1e9:
        return f"{number/1e9:.{precision}f}B"
    elif number >= 1e6:
        return f"{number/1e6:.{precision}f}M"
    elif number >= 1e3:
        return f"{number/1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"

def create_directory_structure(base_path: str, subdirs: List[str]) -> None:
    """Create directory structure"""
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)