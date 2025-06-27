# Plotting Functionality Migration

This document describes the migration of plotting functionality from the legacy `plotting/plot_datavector.py` script to the refactored DESI lensing pipeline architecture.

## Overview

The plotting functionality has been migrated to provide:
- **Modern object-oriented design** using configuration objects
- **Consistent CLI interface** that integrates with the main pipeline
- **Improved modularity** with separate plotting and utility modules
- **Better error handling** and logging
- **Direct CLI arguments** instead of external config files
- **Command-line interface** following the same patterns as compute commands

## Key Changes

### Architecture

| Legacy | Refactored |
|--------|------------|
| `plotting/plot_datavector.py` | `desi_lensing/analysis/plotting.py` |
| `plotting/plotting_utilities.py` | `desi_lensing/analysis/plotting_utils.py` |
| ConfigParser-based config | CLI arguments + Configuration objects |
| Global functions | DataVectorPlotter class |
| Standalone scripts | Integrated CLI commands |

### Configuration

**Legacy approach:**
```python
import configparser
config = configparser.ConfigParser()
config.read("config_plots.conf")
plot_datavector_tomo(config)
```

**Refactored CLI approach:**
```bash
python -m desi_lensing.cli.main plot datavector \
    --galaxy-type LRG \
    --source-surveys HSCY3,KiDS \
    --statistic deltasigma \
    --style paper \
    --output-dir ./plots/
```

**Refactored programmatic approach:**
```python
from desi_lensing.config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig
from desi_lensing.analysis.plotting import DataVectorPlotter

# Create configuration objects (same defaults as CLI)
computation_config = ComputationConfig(statistics=['deltasigma'], tomography=True, ...)
lens_config = LensGalaxyConfig(galaxy_type='LRG', ...)
source_config = SourceSurveyConfig(surveys=['HSCY3'], ...)
output_config = OutputConfig(save_path='./plots/', ...)

# Create plotter and generate plots
plotter = DataVectorPlotter(computation_config, lens_config, source_config, output_config)
plotter.plot_datavector_tomographic()
```

## CLI Usage Examples

### 1. Basic Commands

```bash
# Basic data vector plot
python -m desi_lensing.cli.main plot datavector \
    --galaxy-type LRG \
    --source-surveys HSCY3,KiDS \
    --statistic deltasigma

# B-mode diagnostics
python -m desi_lensing.cli.main plot bmodes \
    --galaxy-type LRG \
    --source-surveys HSCY3 \
    --statistic deltasigma \
    --verbose

# Random lens tests  
python -m desi_lensing.cli.main plot randoms \
    --galaxy-type BGS_BRIGHT \
    --source-surveys DES \
    --statistic deltasigma \
    --style presentation \
    --log-scale

# Cosmology comparison
python -m desi_lensing.cli.main plot compare-cosmologies \
    --galaxy-type LRG \
    --source-surveys HSCY3 \
    --statistic deltasigma \
    --cosmologies planck18,wcdm
```

### 2. Common Options

All plot commands support the same configuration options as compute commands:

```bash
# Galaxy configuration
--galaxy-type {BGS_BRIGHT,LRG,ELG}
--z-bins "0.4,0.6,0.8,1.1"
--bgs-version v1.5
--lrg-version v1.5
--magnitude-cuts / --no-magnitude-cuts

# Source survey configuration  
--source-surveys DES,KiDS,HSCY1,HSCY3
--cut-to-desi / --no-cut-to-desi

# Computation configuration
--cosmology {planck18,wmap9,wcdm}
--tomography / --no-tomography
--rp-min 0.08 --rp-max 80.0 --n-rp-bins 15

# Output configuration
--output-dir ./plots/
--catalogue-path /path/to/catalogues/
```

### 3. Plotting-Specific Options

```bash
# Style options
--style {paper,presentation,notebook}
--log-scale / --linear-scale
--transparent / --opaque
--filename-suffix custom_suffix
--no-save                    # Show only, don't save to file
```

### 4. Help and Defaults

```bash
# Show help for any command
python -m desi_lensing.cli.main plot --help
python -m desi_lensing.cli.main plot datavector --help

# Show default configuration values
python -m desi_lensing.cli.main show-defaults --galaxy-type LRG

# List available source surveys
python -m desi_lensing.cli.main list-surveys
```

## Available Plot Types

### 1. Data Vector Plots (`plot datavector`)
```bash
python -m desi_lensing.cli.main plot datavector --statistic deltasigma --style paper
```
- Plots measured lensing signals for all lens bins and source surveys
- Supports both linear and logarithmic scaling
- Shows tomographic bins with different colors
- Handles both tomographic and non-tomographic analyses

### 2. B-mode Diagnostics (`plot bmodes`)
```bash
python -m desi_lensing.cli.main plot bmodes --statistic deltasigma --verbose
```
- Tests for systematic contamination using B-mode analysis
- Computes and displays p-values and χ² statistics
- Generates LaTeX tables with results
- Essential for validating lensing measurements

### 3. Random Tests (`plot randoms`)
```bash
python -m desi_lensing.cli.main plot randoms --statistic deltasigma --log-scale
```
- Shows lensing signals using random lens positions
- Used to test for systematic effects
- Should be consistent with zero signal

### 4. Cosmology Comparisons (`plot compare-cosmologies`)
```bash
python -m desi_lensing.cli.main plot compare-cosmologies --cosmologies planck18,wcdm
```
- Compares results from different cosmological models
- Useful for testing theoretical systematics
- Shows differences between models

## Programmatic Usage

For advanced use cases, you can still use the classes directly:

```python
from desi_lensing.analysis.plotting import create_plotter_from_configs
from desi_lensing.config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig

# Create configurations (same defaults as CLI)
configs = (
    ComputationConfig(statistics=['deltasigma'], tomography=True),
    LensGalaxyConfig(galaxy_type='LRG'),  # Uses default z-bins
    SourceSurveyConfig(surveys=['HSCY3']),
    OutputConfig(save_path='./plots/')
)

# Create plotter
plotter = create_plotter_from_configs(*configs)

# Generate plots
plotter.plot_datavector_tomographic(statistic='deltasigma', log_scale=False)
pvalues = plotter.plot_bmodes_tomographic(statistic='deltasigma')
```

## Migration Checklist

To migrate from legacy plotting scripts:

1. **Replace standalone scripts with CLI commands:**
   ```bash
   # Old
   python plot_datavector.py config_plots.conf
   
   # New
   python -m desi_lensing.cli.main plot datavector --galaxy-type LRG --source-surveys HSCY3
   ```

2. **Convert configuration files to CLI arguments:**
   - No more `.conf` files needed
   - Use `--help` to see all available options
   - Use `show-defaults` to see default values

3. **Update import statements (for programmatic use):**
   ```python
   # Old
   from plotting_utilities import get_versions, clean_read
   
   # New  
   from desi_lensing.analysis.plotting import DataVectorPlotter
   from desi_lensing.analysis.plotting_utils import setup_matplotlib_style
   ```

4. **Use consistent CLI patterns:**
   - All commands follow the same option structure
   - Same configuration objects used throughout pipeline
   - Consistent logging and error handling

## Integration with Pipeline

The plotting commands are fully integrated with the pipeline:

```bash
# Compute lensing statistics first
python -m desi_lensing.cli.main compute deltasigma \
    --galaxy-type LRG \
    --source-surveys HSCY3 \
    --output-dir ./results/

# Then plot the results using the same configuration
python -m desi_lensing.cli.main plot datavector \
    --galaxy-type LRG \
    --source-surveys HSCY3 \
    --statistic deltasigma \
    --output-dir ./results/
```

## Advanced Features

### Custom Styles
```bash
# Different styles for different purposes
python -m desi_lensing.cli.main plot datavector --style paper        # Publication
python -m desi_lensing.cli.main plot datavector --style presentation # Talks  
python -m desi_lensing.cli.main plot datavector --style notebook     # Jupyter
```

### Output Control
```bash
# Save with custom suffix
python -m desi_lensing.cli.main plot datavector --filename-suffix v2_final

# Show plots without saving
python -m desi_lensing.cli.main plot datavector --no-save

# Transparent backgrounds
python -m desi_lensing.cli.main plot datavector --transparent
```

### Batch Processing
```bash
# Process multiple galaxy types
for galaxy in BGS_BRIGHT LRG; do
    python -m desi_lensing.cli.main plot datavector \
        --galaxy-type $galaxy \
        --filename-suffix $galaxy
done
```

## Troubleshooting

### Common Issues

1. **Command not found**: Install the package in development mode
   ```bash
   pip install -e .
   ```

2. **Missing data files**: The plotter will show warnings for missing files but continue

3. **Configuration errors**: Use `--help` to check available options

4. **Import errors**: Ensure the refactored package is properly installed

### Debug Mode
```bash
python -m desi_lensing.cli.main plot datavector --verbose
python -m desi_lensing.cli.main show-defaults --galaxy-type LRG
```

## Future Enhancements

Planned improvements include:
- Non-tomographic plotting methods
- Interactive plotting with widgets  
- Automated comparison reports
- Enhanced theory prediction loading
- Batch processing utilities

## Notes

- **No external config files needed** - everything via CLI arguments
- **Same defaults as compute commands** for consistency
- **Integrated logging and error handling**
- **Modular design** allows easy extension for new plot types
- **Backward compatibility** maintained through programmatic interface 