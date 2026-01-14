# evaluation/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class CryptoModelEvaluator:
    """Enhanced evaluation for cryptocurrency forecasting models"""
    
    def __init__(self, y_scaler: Optional[Any] = None):
        self.y_scaler = y_scaler
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model performance using various metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_train: Training values for MASE calculation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Inverse transform if scaler is provided
        if self.y_scaler is not None:
            y_true = self._inverse_transform(y_true)
            y_pred = self._inverse_transform(y_pred)
        
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Financial-specific metrics
        directional_accuracy = self._directional_accuracy(y_true, y_pred)
        
        # Mean Absolute Scaled Error (MASE)
        if y_train is not None:
            if self.y_scaler is not None:
                y_train = self._inverse_transform(y_train)
            mase = self._mase(y_true, y_pred, y_train)
        else:
            mase = None
        
        # Sharpe ratio (based on prediction errors)
        sharpe_ratio = self._sharpe_ratio(y_true, y_pred)
        
        # Maximum drawdown
        max_drawdown = self._max_drawdown(y_true, y_pred)
        
        # Information ratio
        information_ratio = self._information_ratio(y_true, y_pred)
        
        # Hit rate (percentage of correct direction predictions)
        hit_rate = self._hit_rate(y_true, y_pred)
        
        # Average return
        avg_return = self._average_return(y_true, y_pred)
        
        # Volatility of errors
        error_volatility = self._error_volatility(y_true, y_pred)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional Accuracy': directional_accuracy,
            'MASE': mase,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Information Ratio': information_ratio,
            'Hit Rate': hit_rate,
            'Average Return': avg_return,
            'Error Volatility': error_volatility
        }
    
    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data using the scaler"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.y_scaler.inverse_transform(data).flatten()
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy"""
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        return np.mean(direction_true == direction_pred)
    
    def _mase(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error"""
        naive_error = np.mean(np.abs(y_train[1:] - y_train[:-1]))
        mae = mean_absolute_error(y_true, y_pred)
        return mae / naive_error if naive_error != 0 else np.inf
    
    def _sharpe_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Sharpe ratio based on prediction errors"""
        errors = y_true - y_pred
        return np.mean(errors) / np.std(errors) if np.std(errors) != 0 else 0
    
    def _max_drawdown(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate maximum drawdown of the prediction errors"""
        errors = y_true - y_pred
        cumulative = np.cumsum(errors)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return np.max(drawdown)
    
    def _information_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Information Ratio"""
        active_return = y_true - y_pred
        tracking_error = np.std(active_return)
        return np.mean(active_return) / tracking_error if tracking_error != 0 else 0
    
    def _hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate hit rate (percentage of correct predictions)"""
        return np.mean(np.sign(y_true - np.mean(y_true)) == np.sign(y_pred - np.mean(y_pred)))
    
    def _average_return(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate average return"""
        returns = (y_pred[1:] - y_pred[:-1]) / y_pred[:-1]
        return np.mean(returns)
    
    def _error_volatility(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate volatility of prediction errors"""
        errors = y_true - y_pred
        return np.std(errors)
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        n: int = 100, title: str = "Predictions vs Actual",
                        save_path: Optional[str] = None) -> None:
        """
        Plot predictions vs actual values
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            n: Number of points to plot
            title: Plot title
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[:n], label='Actual', color='blue', linewidth=2)
        plt.plot(y_pred[:n], label='Predicted', color='red', linestyle='--', linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """
        Plot residuals to analyze model performance
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True)
        
        # Residuals vs predicted
        axes[1, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Residuals vs Predicted')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, results: Dict[str, Dict[str, float]], 
                      save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            results: Dictionary of model results
            save_path: Path to save the comparison
            
        Returns:
            DataFrame with comparison results
        """
        df = pd.DataFrame(results).T
        
        # Highlight best values for each metric
        def highlight_best(s):
            is_best = s == s.min() if s.name in ['MAE', 'MSE', 'RMSE', 'MAPE', 'MASE', 'Max Drawdown', 'Error Volatility'] else s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_best]
        
        styled_df = df.style.apply(highlight_best)
        
        if save_path:
            df.to_csv(save_path)
            logger.info(f"Model comparison saved to {save_path}")
        
        return styled_df