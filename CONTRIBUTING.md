# Contributing to Cryptocurrency Forecasting System

Thank you for your interest in contributing to the Cryptocurrency Forecasting System! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:

- Gender identity and expression
- Sexual orientation
- Disability
- Physical appearance
- Body size
- Race
- Ethnicity
- Age
- Religion
- Nationality

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior include:

- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, or personal/political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Any other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct. Project maintainers who do not follow the Code of Conduct may be removed from the project team.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git installed and configured
- Basic understanding of time series forecasting and deep learning
- Familiarity with PyTorch, pandas, and scikit-learn

### Setting Up Your Development Environment

1. **Fork the repository**:
   ```bash
   # Fork the repository on GitHub
   git clone https://github.com/Tushar-Siddik/crypto_forecasting.git
   cd crypto_forecasting
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Run tests to ensure everything is working**:
   ```bash
   python tests/run_tests.py
   ```

### Project Structure

```
crypto_forecasting/
├── config/                 # Configuration files
├── data/                   # Data handling modules
├── models/                 # Model implementations
├── training/               # Training and tuning
├── evaluation/             # Evaluation and visualization
├── deployment/             # API and deployment
├── utils/                  # Utility functions
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── requirements-dev.txt    # Development dependencies
└── README.md              # Project documentation
```

## Development Workflow

### 1. Create an Issue

Before starting work on a new feature or bug fix:

1. Check if there's an existing issue for your work
2. If not, create a new issue describing:
   - The problem you're solving or feature you're adding
   - Your proposed solution
   - Any questions or concerns

### 2. Create a Branch

Create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Your Changes

- Follow the coding standards outlined below
- Write tests for new functionality
- Update documentation as needed
- Keep commits small and focused

### 4. Test Your Changes

```bash
# Run all tests
python tests/run_tests.py

# Run specific test modules
python tests/run_tests.py test_models

# Run with coverage
pytest --cov=src tests/
```

### 5. Submit a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a pull request with:
   - A clear title
   - A detailed description of changes
   - Reference to any related issues
   - Screenshots or examples if applicable

## Coding Standards

### Python Code Style

We follow PEP 8 and use the following tools:

- **Black** for code formatting
- **Flake8** for linting
- **mypy** for type checking
- **isort** for import sorting

Configure your editor to use these tools automatically.

### Code Formatting

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/
```

### Naming Conventions

- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Prefix with `_`
- **Modules**: `lowercase` with short names

### Documentation

All public functions and classes must have docstrings:

```python
def calculate_returns(prices: np.ndarray, method: str = 'simple') -> np.ndarray:
    """
    Calculate returns from prices.
    
    Args:
        prices: Array of prices
        method: Calculation method ('simple' or 'log')
        
    Returns:
        Array of returns
        
    Raises:
        ValueError: If method is not 'simple' or 'log'
    """
    # Implementation...
```

### Type Hints

Use type hints for all function signatures and class attributes:

```python
from typing import List, Dict, Optional, Union, Tuple

def process_data(
    data: pd.DataFrame,
    config: Dict[str, Any],
    validate: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Implementation...
```

### Error Handling

- Use specific exception types
- Include informative error messages
- Log errors appropriately

```python
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

## Testing Guidelines

### Test Structure

```
tests/
├── test_data.py          # Tests for data modules
├── test_models.py        # Tests for model implementations
├── test_training.py      # Tests for training logic
├── test_evaluation.py    # Tests for evaluation metrics
├── test_integration.py   # Integration tests
└── conftest.py          # Test configuration
```

### Writing Tests

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

```python
import unittest
import numpy as np
from src.data.feature_engineering import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.engineer = FeatureEngineer()
        self.sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        })
    
    def test_add_technical_indicators(self):
        result = self.engineer.add_technical_indicators(self.sample_data)
        self.assertGreater(len(result.columns), len(self.sample_data.columns))
        self.assertIn('RSI', result.columns)
```

### Test Coverage

- Aim for at least 80% code coverage
- Focus on critical paths and edge cases
- Use parameterized tests for multiple scenarios

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html tests/
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and comments
2. **API Documentation**: Auto-generated from docstrings
3. **User Documentation**: Tutorials and guides
4. **Developer Documentation**: Architecture and design

### Writing Documentation

- Use clear, concise language
- Include examples for complex concepts
- Keep documentation up-to-date with code changes

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html
```

## Submitting Changes

### Pull Request Guidelines

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what you changed and why
3. **Testing**: Confirm all tests pass
4. **Documentation**: Update relevant documentation
5. **Breaking Changes**: Clearly indicate any breaking changes

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Review Process

### Reviewer Guidelines

1. **Code Quality**: Check for style, clarity, and correctness
2. **Testing**: Ensure adequate test coverage
3. **Documentation**: Verify documentation is accurate
4. **Performance**: Consider performance implications
5. **Security**: Check for potential security issues

### Review Timeline

- Small changes: 1-2 days
- Medium changes: 3-5 days
- Large changes: 1-2 weeks

### Approval Process

- At least one maintainer must approve
- All automated checks must pass
- Address all reviewer feedback

## Community

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and ideas
- **Email**: For security issues or private matters

### Getting Help

1. Check existing documentation and issues
2. Search GitHub Discussions
3. Create a new issue with appropriate labels

### Contributing Areas

We welcome contributions in:

- **Core Models**: New architectures and improvements
- **Data Sources**: Additional data providers and formats
- **Evaluation**: New metrics and visualization methods
- **Documentation**: Tutorials, examples, and guides
- **Tools**: Utilities and helper functions
- **Performance**: Optimizations and speed improvements

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

<!--
### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number updated
- [ ] Tagged in Git
- [ ] PyPI package published

-->

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:

- **README.md**: In the contributors section
- **CHANGELOG.md**: For specific contributions
- **Release Notes**: For each version
- **Annual Report**: For significant contributions

## Questions?

If you have questions about contributing:

1. Check existing documentation and issues
2. Create a new issue with the `question` label
3. Contact a maintainer directly

Thank you for contributing to the Cryptocurrency Forecasting System!