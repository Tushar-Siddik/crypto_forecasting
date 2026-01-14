# utils/logger.py
import logging
import os
from datetime import datetime
from typing import Optional
import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

def setup_logger(name: str, log_file: Optional[str] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a') if log_file else logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(name)

# Create the logger instance
logger = setup_logger('crypto_forecast')