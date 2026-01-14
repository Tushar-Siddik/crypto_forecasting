# models/ensemble.py
import torch
import torch.nn as nn
from typing import List, Dict, Any
import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

from models.base_model import BaseModel

class EnsembleModel(BaseModel):
    """Ensemble of multiple models"""
    
    def __init__(self, models: List[BaseModel], input_size: int, output_size: int = 1, 
                 aggregation_method: str = 'weighted_average', **kwargs):
        super(EnsembleModel, self).__init__(input_size, output_size, **kwargs)
        
        self.models = nn.ModuleList(models)
        self.aggregation_method = aggregation_method
        
        if aggregation_method == 'weighted_average':
            # Learnable weights for each model
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        elif aggregation_method == 'attention':
            # Attention mechanism for model aggregation
            self.attention = nn.Linear(len(models), len(models))
        elif aggregation_method == 'stacking':
            # Stacking with a meta-learner
            self.meta_learner = nn.Sequential(
                nn.Linear(len(models), len(models) * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(len(models) * 2, output_size)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # (batch_size, num_models, output_size)
        
        if self.aggregation_method == 'average':
            # Simple average
            return predictions.mean(dim=1)
        
        elif self.aggregation_method == 'weighted_average':
            # Weighted average with learnable weights
            weights = torch.softmax(self.weights, dim=0)
            return (predictions * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        
        elif self.aggregation_method == 'attention':
            # Attention-based aggregation
            attention_weights = torch.softmax(self.attention(predictions.mean(dim=-1)), dim=1)
            return (predictions * attention_weights.unsqueeze(-1)).sum(dim=1)
        
        elif self.aggregation_method == 'stacking':
            # Stacking with meta-learner
            meta_input = predictions.mean(dim=-1)  # Average across output dimension
            return self.meta_learner(meta_input)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble model information"""
        info = super().get_model_info()
        info['num_models'] = len(self.models)
        info['aggregation_method'] = self.aggregation_method
        info['model_types'] = [type(model).__name__ for model in self.models]
        
        if self.aggregation_method == 'weighted_average':
            info['model_weights'] = torch.softmax(self.weights, dim=0).tolist()
        
        return info