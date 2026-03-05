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

- **Custom ML Models** (`cornerstone/ml/models/`)
  - Transformer-based: `transformer.py`, `ts_transformer_DP.py`, `transformer_with_attention.py`
  - Ensemble & RNN: `lstm_baseline.py`, `oneFitsAll_DP.py`
  - PatchTST: `patchTST_DP.py`
  - Traditional ML: `XGBoost_DP.py`
  - Time series feature extraction: `embed_DP.py`
  - Rocket-based: `rocket_DP.py`, `rocket_baseline_pytorch_DP.py`, `miniRocket_DP.py`
  - InceptionTime: `inceptionTime_baseline_DP.py`
  - Time Series Library wrapper: `time_series_library_with_mask.py`

- **Comprehensive Time Series Library** (`cornerstone/ml/Time_Series_Library/`)
  - 32 state-of-the-art models:
    - Transformers: Autoformer, Informer, Transformer, Nonstationary_Transformer, iTransformer
    - Attention-based: Crossformer, Pyraformer, Reformer, ETSformer, FEDformer, FiLM
    - Linear & Hybrid: DLinear, TSMixer, TimesNet, TimeMixer, TimeXer, LightTS, TiDE
    - Advanced architectures: PatchTST, Koopa, KANAD, MICN, SegRNN, WPMixer, MultiPatchFormer, Mamba, MambaSimple
    - Specialized: Temporal Fusion Transformer, SCINet, FreTS, PAttn
  - Multiple prediction tasks (forecasting, classification, anomaly detection, imputation)
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
