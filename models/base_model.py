# models/base_model.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    """Base class for all models"""
    
    def __init__(self, input_size: int, output_size: int = 1, **kwargs):
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    
    def save_model(self, path: str) -> None:
        """Save model state dict"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info()
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model state dict"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")