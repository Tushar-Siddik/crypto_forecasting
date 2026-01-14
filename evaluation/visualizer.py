# evaluation/visualizer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """Advanced visualization for model analysis"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_time_series_with_predictions(self, dates: pd.DatetimeIndex, 
                                        actual: np.ndarray, 
                                        predictions: Dict[str, np.ndarray],
                                        title: str = "Cryptocurrency Price Predictions",
                                        save_path: Optional[str] = None) -> None:
        """
        Plot time series with multiple model predictions
        
        Args:
            dates: Date index
            actual: Actual values
            predictions: Dictionary of model predictions
            title: Plot title
            save_path: Path to save the plot
        """
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Add predictions
        colors = px.colors.qualitative.Set1
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=dates,
                y=pred,
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Time series plot saved to {save_path}")
        
        fig.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: np.ndarray,
                              title: str = "Feature Importance",
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores
            title: Plot title
            save_path: Path to save the plot
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=sorted_scores, y=sorted_features)
        plt.title(title, fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]],
                            metrics: List[str] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Plot model comparison across multiple metrics
        
        Args:
            results: Dictionary of model results
            metrics: List of metrics to plot
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ['MAE', 'RMSE', 'MAPE', 'Directional Accuracy', 'MASE']
        
        # Prepare data
        df = pd.DataFrame(results).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:6]):  # Limit to 6 metrics
            if metric in df.columns:
                df[metric].plot(kind='bar', ax=axes[i])
                axes[i].set_title(metric)
                axes[i].set_ylabel('Value')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(metrics), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_intervals(self, dates: pd.DatetimeIndex,
                               actual: np.ndarray,
                               predictions: np.ndarray,
                               lower_bound: np.ndarray,
                               upper_bound: np.ndarray,
                               title: str = "Predictions with Confidence Intervals",
                               save_path: Optional[str] = None) -> None:
        """
        Plot predictions with confidence intervals
        
        Args:
            dates: Date index
            actual: Actual values
            predictions: Predicted values
            lower_bound: Lower bound of confidence interval
            upper_bound: Upper bound of confidence interval
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(14, 7))
        
        # Plot actual values
        plt.plot(dates, actual, label='Actual', color='black', linewidth=2)
        
        # Plot predictions
        plt.plot(dates, predictions, label='Predicted', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        plt.fill_between(dates, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Confidence Interval')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction intervals plot saved to {save_path}")
        
        plt.show()
    
    def plot_residual_analysis(self, actual: np.ndarray, 
                             predictions: np.ndarray,
                             residuals: np.ndarray = None,
                             save_path: Optional[str] = None) -> None:
        """
        Comprehensive residual analysis
        
        Args:
            actual: Actual values
            predictions: Predicted values
            residuals: Residuals (calculated if not provided)
            save_path: Path to save the plot
        """
        if residuals is None:
            residuals = actual - predictions
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Residuals vs Time
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
        stats.probplot(residuals, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot')
        axes[0, 2].grid(True)
        
        # Residuals vs Predicted
        axes[1, 0].scatter(predictions, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Residuals vs Predicted')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].grid(True)
        
        # Autocorrelation of residuals
        from statsmodels.tsa.stattools import acf
        lag_acf = acf(residuals, nlags=40)
        axes[1, 1].plot(lag_acf)
        axes[1, 1].axhline(y=0, linestyle='--', color='gray')
        axes[1, 1].axhline(y=-1.96/np.sqrt(len(residuals)), linestyle='--', color='gray')
        axes[1, 1].axhline(y=1.96/np.sqrt(len(residuals)), linestyle='--', color='gray')
        axes[1, 1].set_title('Autocorrelation of Residuals')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('ACF')
        axes[1, 1].grid(True)
        
        # Actual vs Predicted
        axes[1, 2].scatter(actual, predictions, alpha=0.5)
        min_val = min(actual.min(), predictions.min())
        max_val = max(actual.max(), predictions.max())
        axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 2].set_title('Actual vs Predicted')
        axes[1, 2].set_xlabel('Actual')
        axes[1, 2].set_ylabel('Predicted')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, history: Dict[str, List[float]],
                           save_path: Optional[str] = None) -> None:
        """
        Plot learning curves
        
        Args:
            history: Training history
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        if 'learning_rate' in history:
            axes[0, 1].plot(history['learning_rate'])
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)
        
        # Epoch time
        if 'epoch_time' in history:
            axes[1, 0].plot(history['epoch_time'])
            axes[1, 0].set_title('Epoch Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].grid(True)
        
        # Loss difference
        if 'train_loss' in history and 'val_loss' in history:
            loss_diff = np.array(history['train_loss']) - np.array(history['val_loss'])
            axes[1, 1].plot(loss_diff)
            axes[1, 1].set_title('Training - Validation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves plot saved to {save_path}")
        
        plt.show()