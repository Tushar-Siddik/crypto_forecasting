# src/deployment/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import CryptoDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel

app = FastAPI(title="Crypto Price Forecasting API", description="API for forecasting cryptocurrency prices")

# Global variables for model and scalers
model = None
feature_scaler = None
target_scaler = None
feature_cols = None
sequence_length = 60
model_type = 'lstm'
input_size = None

class PredictionRequest(BaseModel):
    ticker: str
    days: int = 1

class PredictionResponse(BaseModel):
    ticker: str
    predictions: List[float]
    dates: List[str]

class ModelInfo(BaseModel):
    model_type: str
    input_size: int
    sequence_length: int

@app.on_event("startup")
async def load_model():
    """Load the model and scalers on startup"""
    global model, feature_scaler, target_scaler, feature_cols, sequence_length, model_type, input_size
    
    # In a real deployment, these would be loaded from files
    # For this example, we'll initialize them but not load actual weights
    
    # Load model configuration
    model_type = 'lstm'  # or 'transformer'
    sequence_length = 60
    input_size = 20  # This would be determined by the number of features
    
    # Initialize model (without weights)
    if model_type == 'lstm':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.2
        )
    else:  # transformer
        model = TransformerModel(
            input_size=input_size,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            output_size=1
        )
    
    # In a real deployment, load the saved model weights
    # model.load_state_dict(torch.load('path/to/saved/model.pth'))
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize scalers (in a real deployment, these would be loaded from files)
    feature_scaler = None
    target_scaler = None
    
    # Initialize feature columns (in a real deployment, these would be loaded from a file)
    feature_cols = [f'feature_{i}' for i in range(input_size)]

@app.get("/")
async def root():
    return {"message": "Crypto Price Forecasting API"}

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    return ModelInfo(
        model_type=model_type,
        input_size=input_size,
        sequence_length=sequence_length
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions for a cryptocurrency"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Load the latest data for the requested cryptocurrency
        loader = CryptoDataLoader()
        data = loader.get_latest_data(request.ticker, days=sequence_length + 10)
        
        if data is None or len(data) < sequence_length:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {request.ticker}")
        
        # Feature engineering
        engineer = FeatureEngineer()
        data_with_features = engineer.add_technical_indicators(data)
        
        # In a real deployment, we would use the saved feature columns and scalers
        # For this example, we'll just use the last sequence_length rows
        
        # Get the last sequence_length rows
        last_sequence = data_with_features.iloc[-sequence_length:]
        
        # Extract features (in a real deployment, we would use the saved feature columns)
        features = last_sequence.values
        
        # Normalize features (in a real deployment, we would use the saved feature scaler)
        # For this example, we'll skip normalization
        
        # Create input tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.item()
        
        # In a real deployment, we would inverse transform the prediction using the saved target scaler
        # For this example, we'll just return the raw prediction
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=request.days + 1, freq='D')[1:]
        
        # For multi-day prediction, we would need to implement an iterative prediction process
        # For this example, we'll just return the same prediction for each day
        predictions = [prediction] * request.days
        
        return PredictionResponse(
            ticker=request.ticker,
            predictions=predictions,
            dates=future_dates.strftime('%Y-%m-%d').tolist()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)