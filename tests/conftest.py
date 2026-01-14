# tests/conftest.py
import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

@pytest.fixture
def sample_data():
    """Create sample cryptocurrency data"""
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    np.random.seed(42)
    return pd.DataFrame({
        'Open': np.random.uniform(30000, 60000, len(dates)),
        'High': np.random.uniform(30000, 60000, len(dates)),
        'Low': np.random.uniform(30000, 60000, len(dates)),
        'Close': np.random.uniform(30000, 60000, len(dates)),
        'Volume': np.random.uniform(1000000, 10000000, len(dates))
    }, index=dates)

@pytest.fixture
def sample_sequences():
    """Create sample sequences for testing"""
    return {
        'X': np.random.randn(100, 10, 5),  # 100 samples, 10 timesteps, 5 features
        'y': np.random.randn(100)  # 100 targets
    }

@pytest.fixture
def device():
    """Get device for testing"""
    return torch.device('cpu')  # Use CPU for tests

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    import tempfile
    return tempfile.mkdtemp()