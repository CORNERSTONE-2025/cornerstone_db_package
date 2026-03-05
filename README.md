# Cornerstone

A comprehensive Python package for solar flare forecasting and active region analysis using machine learning and time series modeling.

## Overview

**Cornerstone** is a machine learning-based framework designed to predict solar flares from SHARP (Space-weather HMI Active Region Patch) magnetogram data. It combines transformer architectures, scikit-learn models, and time series forecasting libraries to provide robust predictions of solar activity.

## Project Structure

### `cornerstone/` - Core Project Code

The main package containing all components for data processing, model training, and prediction:

- **`cornerstone.py`** - Main API and orchestration module
- **`sharp_data.py`** - SHARP data ingestion and preprocessing
- **`config.py`** - Configuration management
- **`ml/`** - Machine learning modules
  - **`models/`** - Model implementations (transformers, LSTM, XGBoost, Rocket, etc.)
  - **`utils/`** - Training utilities, metrics, datasets, and configuration
  - **`Time_Series_Library/`** - Comprehensive time series forecasting models (Autoformer, PatchTST, TimesNet, etc.)
  - **`UnsupervisedScalableRepresentationLearningTimeSeries/`** - Representation learning for time series
- **`configs/`** - Configuration files for databases and models

### `cornerstone_test.py` - Usage Example

A practical example demonstrating how to use the Cornerstone package:
- Load and preprocess SHARP data
- Train models with various configurations
- Generate predictions and evaluate performance
- Visualize results

**Start here** to understand the API and workflow.

## Key Features

- **Multiple Model Architectures**
  - Transformer-based models (Autoformer, Informer, PatchTST)
  - Traditional ML (XGBoost, scikit-learn)
  - RNN/LSTM baselines
  - State-of-the-art time series models (TimesNet, DLinear, Mamba)

- **Comprehensive Time Series Library**
  - 30+ forecasting and classification models
  - Support for multiple prediction tasks
  - Modular architecture for easy extension

- **Data Pipeline**
  - SHARP magnetogram ingestion
  - Feature engineering and preprocessing
  - Automated train/validation/test splitting
  - Dataset configuration management

- **Evaluation & Calibration**
  - Multiple metrics for regression and classification
  - Probability calibration tools
  - Threshold optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/PedemonteDaniele/cornerstone_db_package.git
cd cornerstone_db_package

# Install dependencies
pip install -r cornerstone/ml/Time_Series_Library/requirements.txt

# Additional dependencies
pip install scikit-learn xgboost torch transformers
```

## Quick Start

```python
from cornerstone import Cornerstone
from cornerstone_test import example_workflow

# See cornerstone_test.py for complete examples
# Basic usage pattern:
# 1. Initialize Cornerstone with configuration
# 2. Load SHARP data
# 3. Create train/val/test splits
# 4. Train a model
# 5. Generate predictions and evaluate
```

## Configuration

Configuration is managed via JSON files in `cornerstone/configs/`:
- Database connectivity (`db_config.json`)
- Model hyperparameters
- Data pipeline settings

See example configurations in `cornerstone/configs/` for details.

## Dependencies

### Core Libraries
- **PyTorch** - Deep learning framework
- **Transformers** - State-of-the-art NLP/time series models
- **scikit-learn** - Traditional ML algorithms
- **NumPy, Pandas** - Data manipulation

### Specialized
- **Time Series Library** - Custom implementations of advanced forecasting models
- **TUML** - Time series utilities and models

## Model Categories

### Time Series Forecasting
- Autoformer, Informer, Reformer
- Transformer, Nonstationary Transformer
- PatchTST, DLinear, Linear
- TimesNet, ETSformer, FEDformer
- Mamba, Koopa, and more

### Time Series Classification
- CNN-based (InceptionTime, Rocket, MiniRocket)
- Transformer-based
- Ensemble methods

### Representation Learning
- Unsupervised scalable methods
- Contrastive learning approaches

## Documentation

- See `cornerstone/DOCUMENTATION.md` for detailed API documentation
- Check `cornerstone/ml/` for module-specific documentation
- Review `cornerstone_test.py` for usage examples

## Author

Daniele Pedemonte  
University of Genoa

## License

See LICENSE file for details.

## References

- SHARP magnetogram data: [HMI](http://jsoc.stanford.edu/)
- Time Series Transformer models: [AutoML for Time Series](https://github.com/yuqinie98/PatchTST)
- Advanced forecasting: [Time Series Library](https://github.com/thuml/Time-Series-Library)

---

For issues, questions, or collaboration opportunities, please open an issue on GitHub.
