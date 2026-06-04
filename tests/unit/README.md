# Unit Tests

TODO:
- **11 skipped tests** (4 ROC plot tests requiring sklearn setup, 7 dataloader tests if data missing, 1 MLflow test)

This directory contains unit tests for the Food101 MLOps pipeline. Unit tests verify individual functions and components in isolation, ensuring each part of the system works correctly on its own.

## Overview

The unit test suite covers all core modules of the project:
- **Models**: Architecture definitions and forward passes
- **Training**: Training, validation, and evaluation steps
- **Data Loading**: Dataset loading and preprocessing
- **Evaluation**: Cross-validation analysis and plotting
- **Inference**: Preprocessing, postprocessing, and prediction
- **Utilities**: Helper functions and path management
- **Configuration**: Config file validation

## Test Files

### Core Module Tests

#### `test_utils_helpers.py`
Tests for utility helper functions.
- **Functions tested**: `set_seed()`
- **Coverage**: 
  - Seed setting for Python, NumPy, and PyTorch
  - Reproducibility verification
  - CUDA deterministic mode
  - Environment variable configuration
- **Test count**: 5 tests

#### `test_utils_paths.py`
Tests for path management and directory operations.
- **Functions tested**: `get_paths()`, `clean_outputs_dir()`
- **Coverage**:
  - Path generation with/without folds
  - Model checkpoint path creation
  - Directory creation and cleanup
  - Error handling for missing directories
- **Test count**: 6 tests
- **Note**: Uses temporary project structures for isolation

#### `test_config_validation.py`
Tests for configuration file validation.
- **Coverage**:
  - Config file loading
  - Required parameter validation
  - Data type validation
  - Parameter range validation
- **Test count**: 4 tests

### Model Architecture Tests

#### `test_models_food101.py`
Tests for the Food101 CNN model architecture.
- **Class tested**: `Food101`
- **Coverage**:
  - Model initialization
  - Forward pass shape validation
  - Output dimension correctness
  - Gradient flow verification
  - CPU/GPU device handling
- **Test count**: 5 tests

#### `test_models_food101_0.py`
Tests for the Food101_0 variant architecture.
- **Class tested**: `Food101_0`
- **Coverage**:
  - Model initialization
  - Forward pass mechanics
  - Output shape validation
  - Architecture differences vs Food101
- **Test count**: 4 tests

### Training Pipeline Tests

#### `test_training_engine.py`
Tests for training, validation, and evaluation engine functions.
- **Functions tested**: `train_step()`, `val_step()`, `eval_step()`
- **Coverage**:
  - Training step execution with mock data
  - Validation step execution
  - Evaluation step with probability outputs
  - Loss computation and gradient flow
  - No gradient computation in validation mode
  - Output shape validation
- **Test count**: 8 tests
- **Note**: Uses synthetic data loaders, handles CPU/GPU device compatibility

### Data Loading Tests

#### `test_data_dataloader.py`
Tests for dataset loading and DataLoader creation.
- **Functions tested**: `get_dataloaders()`
- **Coverage**:
  - Dataloader creation with fold parameter
  - Test set dataloader creation (without fold)
  - Image size transformation
  - Batch size handling
  - Class names loading
  - Shuffle flag behavior
  - Error handling for missing directories
- **Test count**: 7 tests
- **Requirements**: Requires `Data/test_dataset/` directory structure
  - Structure: `Data/test_dataset/fold0/train/`, `fold0/val/`, `test/`
  - Needs: `classes.txt` file with class names
  - Needs: Image files organized in class subdirectories

### Evaluation Tests

#### `test_evaluation_crossval_score.py`
Tests for cross-validation scoring and model selection functions.
- **Functions tested**: 
  - `filter_by_loss_discrepancy()`
  - `group_by_arch_and_config()`
  - `score_models()`
  - `select_best_configs()`
  - `select_best_model_from_auc()`
- **Coverage**:
  - DataFrame filtering logic
  - Grouping operations
  - Scoring calculations
  - Best model selection algorithms
  - Error handling
- **Test count**: 6 tests (1 skipped - requires MLflow setup)
- **Note**: Uses mock pandas DataFrames, doesn't require actual MLflow runs

#### `test_evaluation_plots.py`
Tests for plotting functions.
- **Functions tested**: `loss_acc_plot()`, `plot_roc_ovr()`
- **Coverage**:
  - Plot generation with training results
  - Figure properties and structure
  - ROC curve plot creation (skipped - requires sklearn and full evaluation)
  - AUC calculation (skipped)
- **Test count**: 5 tests (3 skipped - require sklearn and model evaluation setup)

### Inference Tests

#### `test_inference_postprocessing.py`
Tests for inference postprocessing functions.
- **Functions tested**: `topk_indices()`, `map_indices_to_labels()`
- **Coverage**:
  - Top-k probability selection
  - Edge cases (k > array size, k=0, negative k)
  - Sorted order verification
  - Label mapping with/without labels
  - Empty input handling
- **Test count**: 6 tests

#### `test_inference_predictor.py`
Tests for image preprocessing and inference prediction functions.
- **Functions tested**: `_preprocess()`, `_softmax()`
- **Coverage**:
  - Image preprocessing pipeline
  - Image resizing
  - Normalization and unit scaling
  - Softmax function correctness
  - Edge cases (zeros, large values, numerical stability)
- **Test count**: 6 tests

## Running Tests

### Run All Unit Tests
```bash
# From project root
pytest tests/unit/ -v

# With coverage report
pytest tests/unit/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models_food101.py -v

# Run specific test function
pytest tests/unit/test_models_food101.py::TestFood101::test_food101_initialization -v
```

### Run Only Non-Skipped Tests
```bash
pytest tests/unit/ -v -m "not skip"
```

### Run Tests in Parallel (if pytest-xdist installed)
```bash
pytest tests/unit/ -n auto
```

## Test Requirements

### Data Requirements
- **`Data/test_dataset/`** directory must exist with proper structure:
  ```
  Data/test_dataset/
  ├── classes.txt              # List of class names (one per line)
  ├── fold0/
  │   ├── train/
  │   │   ├── class1/
  │   │   │   └── *.jpg
  │   │   └── class2/
  │   │       └── *.jpg
  │   └── val/
  │       ├── class1/
  │       └── class2/
  └── test/
      ├── class1/
      └── class2/
  ```

### Dependencies
All dependencies are listed in `pyproject.toml`:
- pytest
- torch, torchvision
- numpy
- pandas
- matplotlib
- sklearn (for ROC plot tests)
- PIL/Pillow (for image tests)

## Test Structure

### Shared Fixtures (`conftest.py`)
The `conftest.py` file provides shared fixtures for all tests:
- `temp_dir`: Temporary directory for file operations
- `sample_image_tensor`: Sample image tensor for model tests
- `sample_batch`: Sample batch of data
- `device`: CPU or CUDA device detection
- `reset_seed`: Fixture to reset random seeds after tests

### Test Organization
Tests are organized by module, following the same structure as `src/`:
```
tests/unit/
├── test_models_*.py          # Model architecture tests
├── test_training_*.py        # Training engine tests
├── test_data_*.py            # Data loading tests
├── test_evaluation_*.py      # Evaluation and plotting tests
├── test_inference_*.py       # Inference pipeline tests
├── test_utils_*.py           # Utility function tests
└── test_config_*.py          # Configuration tests
```

### Areas Covered
- Model architectures and forward passes
- Training, validation, and evaluation loops
- Data loading and preprocessing
- Path management and file operations
- Configuration validation
- Inference preprocessing and postprocessing
- Cross-validation and model selection logic

### Areas Not Yet Covered (Future Work)
- Full MLflow integration (requires running MLflow server)
- End-to-end training pipeline (integration tests)
- ONNX model export and loading (integration tests)
- Full ROC curve generation (requires sklearn setup)
- Actual image dataset processing (requires test images)

## Best Practices

### Writing New Tests
1. **Follow naming conventions**: `test_<function_name>()`
2. **Use descriptive docstrings**: Explain what each test verifies
3. **Test edge cases**: Include boundary conditions and error cases
4. **Keep tests isolated**: Each test should be independent
5. **Use fixtures**: Leverage `conftest.py` for shared setup
6. **Mock external dependencies**: Don't rely on external services in unit tests

### Test Isolation
- Tests use temporary directories when possible
- Random seeds are reset after each test
- Models and data are created fresh for each test
- No shared state between tests

### Device Compatibility
Tests automatically handle CPU/GPU:
- Models are moved to the same device as test data
- Tests work on both CPU-only and CUDA-enabled systems
- Device detection via `conftest.py` fixtures

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` for `Data/test_dataset/`
- **Solution**: Create the test dataset structure (see Data Requirements above)

**Issue**: `RuntimeError: Input type and weight type should be the same`
- **Solution**: Ensure models are moved to the same device as data (use `device` fixture)

**Issue**: Tests pass locally but fail in CI
- **Solution**: Check for hardcoded paths, ensure temporary directories are used

**Issue**: MLflow tests fail
- **Solution**: MLflow tests require a running MLflow server or proper mocking

### Skipped Tests
Some tests are intentionally skipped:
- **ROC plot tests**: Require sklearn and full model evaluation setup
- **MLflow integration tests**: Require MLflow server or complex mocking
- **Data loader tests**: Will skip if `Data/test_dataset/` doesn't exist

## Contributing

When adding new tests:
1. Follow the existing test structure and naming conventions
2. Add tests to the appropriate file or create new file if needed
3. Ensure tests pass both on CPU and GPU (if applicable)
4. Update this README if adding new test categories
5. Add appropriate fixtures to `conftest.py` if needed

## Test Metrics

Run tests with coverage to see current coverage metrics:
```bash
pytest tests/unit/ --cov=src --cov-report=term-missing
```

This will show which lines are covered and which are not, helping identify areas that need more testing.
