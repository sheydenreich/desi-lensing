# DESI Lensing Randoms Analysis Implementation

This document describes the implementation of random data vector analysis functionality in the refactored DESI lensing pipeline.

## Overview

The randoms analysis functionality has been successfully integrated into the refactored DESI lensing codebase, providing capabilities for:

- Generating random realizations of lensing data vectors
- Performing systematic tests with random data
- Source redshift slope analysis
- Data splits analysis
- Statistical validation through p-value computation

## Components

### 1. Core Analysis Module: `randoms.py`

Located at: `desi_lensing/analysis/randoms.py`

**Main Class**: `RandomsAnalyzer`

Key methods:
- `prepare_randoms_datavector()` - Prepares data vectors and covariances
- `generate_randoms_datavectors()` - Generates random realizations using multivariate normal
- `generate_random_source_redshift_slope_test()` - Source redshift slope analysis
- `generate_random_splits_test()` - Data splits analysis
- `compute_pvalue_from_randoms()` - Statistical p-value computation

### 2. CLI Integration

Two main command groups:

**Randoms Generation Commands:**
```bash
# Source redshift slope analysis
python -m desi_lensing.cli.main randoms source-redshift-slope --n-randoms 1000

# Data splits analysis  
python -m desi_lensing.cli.main randoms splits --n-randoms 1000
```

**Plotting Commands:**
```bash
# Plot randoms test results
python -m desi_lensing.cli.main plot randoms --statistic deltasigma
```

### 3. Plotting Integration

The randoms functionality is integrated into the plotting system via:
- Enhanced `plot_randoms_test()` method in `plotting.py`
- Automatic loading of existing randoms results
- Fallback to mock plots for demonstration when no results exist

## Usage Examples

### Generate Random Realizations

```python
from desi_lensing.analysis.randoms import create_randoms_analyzer_from_configs
from desi_lensing.config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig

# Create configurations
comp_config = ComputationConfig(statistics=['deltasigma'], tomography=True)
lens_config = LensGalaxyConfig(galaxy_type='BGS_BRIGHT')
source_config = SourceSurveyConfig(surveys=['DES', 'KiDS'])
output_config = OutputConfig(save_path='./output')

# Create analyzer
randoms_analyzer = create_randoms_analyzer_from_configs(
    comp_config, lens_config, source_config, output_config
)

# Generate source redshift slope test
randoms_analyzer.generate_random_source_redshift_slope_test(n_randoms=1000)
```

### Using CLI

```bash
# Generate randoms for source redshift slope test
python -m desi_lensing.cli.main randoms source-redshift-slope \
    --galaxy-type BGS_BRIGHT \
    --source-surveys DES,KiDS \
    --n-randoms 1000 \
    --datavector-type measured

# Plot the results
python -m desi_lensing.cli.main plot randoms \
    --galaxy-type BGS_BRIGHT \
    --source-surveys DES,KiDS \
    --statistic deltasigma
```

## Key Features

### 1. Flexible Data Vector Preparation
- Supports multiple data vector types: `zero`, `emulator`, `measured`
- Handles both tomographic and non-tomographic analysis
- Accounts for cross-survey covariances when requested
- Supports data splits (RA, Dec, Ntile)

### 2. Robust Random Generation
- Uses numpy's multivariate normal distribution
- Handles singular covariance matrices with diagonal approximation
- Supports both individual and combined survey analysis
- Automatic splitting of combined randoms back to individual surveys

### 3. Statistical Analysis
- P-value computation with and without covariance weighting
- Chi-squared test implementations
- Handles missing data gracefully

### 4. File I/O Integration
- Compatible with existing file naming conventions
- Automatic output directory management
- Support for various file suffixes and split configurations

## File Structure

The implementation adds the following files/modifications:

```
desi_lensing_refactored/
├── desi_lensing/
│   ├── analysis/
│   │   ├── randoms.py          # NEW: Core randoms analysis
│   │   ├── __init__.py         # MODIFIED: Added randoms exports
│   │   └── plotting.py         # MODIFIED: Enhanced plot_randoms_test()
│   └── cli/
│       └── main.py             # MODIFIED: Added randoms CLI commands
├── test_randoms_integration.py # NEW: Integration test script
└── RANDOMS_README.md           # NEW: This documentation
```

## Testing

The implementation includes comprehensive testing:

1. **Unit Tests**: Test individual randoms analyzer functionality
2. **Integration Tests**: Test CLI integration and command structure
3. **End-to-End Tests**: Full workflow from config creation to result generation

Run tests with:
```bash
python test_randoms_integration.py
```

## Compatibility

The implementation is fully compatible with:
- The existing refactored codebase structure
- All configuration classes and path management
- The existing data loading infrastructure
- The modular CLI system
- Matplotlib plotting system

## Future Enhancements

Potential areas for future development:
1. Real data loading integration (currently uses placeholders for missing files)
2. More sophisticated cross-covariance handling
3. Additional systematic test types
4. Parallel processing for large-scale random generation
5. Integration with theoretical prediction modules

## Notes

- The current implementation uses placeholder data when actual measurement files are not available
- File path construction follows the existing naming conventions used throughout the codebase
- All logging and error handling follows the established patterns in the refactored codebase 