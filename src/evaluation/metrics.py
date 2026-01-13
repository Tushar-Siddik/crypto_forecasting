# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using various metrics
    
    Parameters:
    - y_true: numpy array, actual values
    - y_pred: numpy array, predicted values
    
    Returns:
    - metrics: dict, evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    # Mean Absolute Scaled Error (MASE)
    naive_error = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    mase = mae / naive_error if naive_error != 0 else np.inf
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Directional Accuracy': directional_accuracy,
        'MASE': mase
    }