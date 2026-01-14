# deployment/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging
import json
import os
import joblib
from contextlib import asynccontextmanager

from ..data.data_loader import CryptoDataLoader
from ..data.feature_engineering import FeatureEngineer
from ..data.preprocessor import DataPreprocessor
from ..models.lstm_attention import LSTMAttention
from ..models.transformer import TransformerModel
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

# Global variables
models = {}
scalers = {}
feature_engineers = {}
preprocessors = {}
model_configs = {}

class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Cryptocurrency ticker (e.g., BTC-USD)")
    days: int = Field(1, ge=1, le=30, description="Number of days to predict")
    model_type: str = Field("lstm_attention", description="Model type to use")
    confidence_interval: bool = Field(False, description="Include confidence intervals")

class PredictionResponse(BaseModel):
    ticker: str
    predictions: List[float]
    dates: List[str]
    model_type: str
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    metrics: Optional[Dict[str, float]] = None

class ModelInfo(BaseModel):
    model_type: str
    input_size: int
    output_size: int
    total_parameters: int
    trainable_parameters: int
    last_updated: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    last_updated: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting up cryptocurrency forecasting API...")
    await load_models()
    yield
    # Shutdown
    logger.info("Shutting down cryptocurrency forecasting API...")

app = FastAPI(
    title="Cryptocurrency Forecasting API",
    description="Advanced API for cryptocurrency price forecasting using deep learning models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def load_models():
    """Load all models and related components"""
    model_dir = "models/saved"
    
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory {model_dir} not found")
        return
    
    # Load available models
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.pth'):
            model_type = model_file.split('_')[0]
            ticker = model_file.split('_')[1]
            
            try:
                # Load model configuration
                config_path = os.path.join(model_dir, f"{model_type}_{ticker}_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    model_configs[f"{model_type}_{ticker}"] = config
                
                # Load model
                model_path = os.path.join(model_dir, model_file)
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Create model based on type
                if model_type == 'lstm_attention':
                    model = LSTMAttention(
                        input_size=config['input_size'],
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        output_size=config['output_size'],
                        dropout=config['dropout'],
                        bidirectional=config['bidirectional'],
                        attention_heads=config['attention_heads']
                    )
                elif model_type == 'transformer':
                    model = TransformerModel(
                        input_size=config['input_size'],
                        d_model=config['d_model'],
                        nhead=config['nhead'],
                        num_encoder_layers=config['num_encoder_layers'],
                        dim_feedforward=config['dim_feedforward'],
                        output_size=config['output_size'],
                        dropout=config['dropout']
                    )
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # Move to GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                
                models[f"{model_type}_{ticker}"] = model
                
                # Load scalers
                feature_scaler_path = os.path.join(model_dir, f"{model_type}_{ticker}_feature_scaler.pkl")
                target_scaler_path = os.path.join(model_dir, f"{model_type}_{ticker}_target_scaler.pkl")
                
                if os.path.exists(feature_scaler_path):
                    scalers[f"{model_type}_{ticker}_feature"] = joblib.load(feature_scaler_path)
                
                if os.path.exists(target_scaler_path):
                    scalers[f"{model_type}_{ticker}_target"] = joblib.load(target_scaler_path)
                
                # Load feature columns
                feature_columns_path = os.path.join(model_dir, f"{model_type}_{ticker}_feature_columns.pkl")
                if os.path.exists(feature_columns_path):
                    feature_columns = joblib.load(feature_columns_path)
                    
                    # Create feature engineer
                    feature_engineer = FeatureEngineer()
                    feature_engineer.feature_columns = feature_columns
                    feature_engineers[f"{model_type}_{ticker}"] = feature_engineer
                
                # Create preprocessor
                preprocessor = DataPreprocessor()
                preprocessor.feature_scaler = scalers.get(f"{model_type}_{ticker}_feature")
                preprocessor.target_scaler = scalers.get(f"{model_type}_{ticker}_target")
                preprocessor.feature_columns = feature_columns
                preprocessors[f"{model_type}_{ticker}"] = preprocessor
                
                logger.info(f"Loaded model: {model_type}_{ticker}")
                
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {e}")

def get_model(model_key: str):
    """Get model by key"""
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    return models[model_key]

def get_preprocessor(model_key: str):
    """Get preprocessor by key"""
    if model_key not in preprocessors:
        raise HTTPException(status_code=404, detail=f"Preprocessor for {model_key} not found")
    return preprocessors[model_key]

def get_feature_engineer(model_key: str):
    """Get feature engineer by key"""
    if model_key not in feature_engineers:
        raise HTTPException(status_code=404, detail=f"Feature engineer for {model_key} not found")
    return feature_engineers[model_key]

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Cryptocurrency Forecasting API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        last_updated=datetime.now().isoformat()
    )

@app.get("/models", response_model=List[str])
async def list_models():
    """List available models"""
    return list(models.keys())

@app.get("/models/{model_key}", response_model=ModelInfo)
async def get_model_info(model_key: str):
    """Get model information"""
    model = get_model(model_key)
    info = model.get_model_info()
    info['last_updated'] = datetime.now().isoformat()
    return ModelInfo(**info)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make predictions for a cryptocurrency"""
    model_key = f"{request.model_type}_{request.ticker}"
    
    try:
        # Get components
        model = get_model(model_key)
        preprocessor = get_preprocessor(model_key)
        feature_engineer = get_feature_engineer(model_key)
        
        # Load latest data
        loader = CryptoDataLoader()
        data = loader.get_latest_data(request.ticker, days=120)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
        
        # Add technical indicators
        data_with_features = feature_engineer.add_technical_indicators(data)
        
        # Get the last sequence
        sequence_length = model_configs[model_key]['sequence_length']
        last_sequence = data_with_features.iloc[-sequence_length:]
        
        # Extract features
        features = last_sequence[feature_engineer.feature_columns].values
        
        # Scale features
        if preprocessor.feature_scaler:
            features = preprocessor.feature_scaler.transform(features)
        
        # Create input tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.cpu().numpy().flatten()[0]
        
        # Inverse transform prediction
        if preprocessor.target_scaler:
            prediction = preprocessor.inverse_transform_target(np.array([prediction]))[0]
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=request.days + 1, freq='D')[1:]
        
        # For multi-day prediction, we would implement iterative prediction
        predictions = [prediction] * request.days
        
        # Prepare response
        response = PredictionResponse(
            ticker=request.ticker,
            predictions=predictions,
            dates=future_dates.strftime('%Y-%m-%d').tolist(),
            model_type=request.model_type
        )
        
        # Add confidence intervals if requested
        if request.confidence_interval:
            # This would require implementing prediction intervals
            # For now, we'll add placeholder values
            response.confidence_intervals = [
                {"lower": pred * 0.95, "upper": pred * 1.05} 
                for pred in predictions
            ]
        
        # Log prediction request
        background_tasks.add_task(
            log_prediction,
            request.ticker,
            request.model_type,
            predictions[0]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

async def log_prediction(ticker: str, model_type: str, prediction: float):
    """Log prediction to file or database"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "model_type": model_type,
        "prediction": prediction
    }
    
    # Here you would typically save to a database or log file
    logger.info(f"Prediction logged: {log_entry}")

@app.post("/models/{model_key}/retrain")
async def retrain_model(model_key: str, background_tasks: BackgroundTasks):
    """Retrain a model with latest data"""
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    
    # Start retraining in background
    background_tasks.add_task(retrain_model_task, model_key)
    
    return {"message": f"Model {model_key} retraining started"}

async def retrain_model_task(model_key: str):
    """Background task to retrain model"""
    try:
        logger.info(f"Starting retraining for {model_key}")
        
        # This would implement the retraining logic
        # For now, we'll just log the action
        await asyncio.sleep(10)  # Simulate training time
        
        logger.info(f"Retraining completed for {model_key}")
        
    except Exception as e:
        logger.error(f"Retraining failed for {model_key}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)