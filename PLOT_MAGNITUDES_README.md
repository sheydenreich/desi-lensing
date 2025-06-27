# Plot Magnitudes Functionality

This document describes the new `plot_magnitudes` functionality added to the DESI lensing pipeline plotting framework.

## Overview

The `plot_magnitudes` method creates density plots of lens galaxies in the magnitude-redshift plane, showing:
- Absolute magnitude distributions vs redshift
- Magnitude cuts per redshift bin (with dashed horizontal lines)
- Redshift bin boundaries (with dotted vertical lines)
- Optional extinction correction
- Optional KP3 cut line at -21.5 mag

## Features

### 1. Density Plotting
- Uses `mpl_scatter_density` for high-density galaxy distributions
- Falls back to regular scatter plot if `mpl_scatter_density` is not available
- Custom "white viridis" colormap for clear visualization

### 2. Magnitude Data Handling
- Automatically loads FastSpecFit magnitude data (`ABSMAG01_SDSS_R`)
- Falls back to mock data for testing if FastSpecFit data unavailable
- Supports derived magnitude columns (e.g., `ABSMAG_RP0`)

### 3. Configurable Magnitude Cuts
- Default cuts per galaxy type:
  - BGS_BRIGHT: [-19.5, -20.5, -21.0]
  - LRG/ELG: None (no magnitude cuts)
- Custom cuts can be specified via CLI
- Cuts are NOT part of configuration files (visualization only)

### 4. Extinction Correction
- Optional extinction correction: `ecorr = -0.8*(z-0.1)`
- Toggleable via `--apply-extinction-correction` CLI flag

### 5. Additional Features
- Shows survival fraction after magnitude cuts
- Optional KP3 cut line at -21.5 mag
- Consistent with existing plotting framework styling

## Usage

### Command Line Interface

```bash
# Basic magnitude plot with defaults (iron release)
desi-lensing plot magnitudes --galaxy-type BGS_BRIGHT

# Use future loa release
desi-lensing plot magnitudes --galaxy-type BGS_BRIGHT --release loa

# Custom magnitude cuts
desi-lensing plot magnitudes --galaxy-type BGS_BRIGHT --magnitude-cuts "-19.0,-20.0,-21.5"

# With extinction correction
desi-lensing plot magnitudes --galaxy-type BGS_BRIGHT --apply-extinction-correction

# With KP3 cut line
desi-lensing plot magnitudes --galaxy-type BGS_BRIGHT --add-kp3-cut

# Multiple galaxy types with loa release
desi-lensing plot magnitudes --galaxy-type BGS_BRIGHT,LRG --release loa --filename-suffix "both_types_loa"
```

### Python API

```python
from desi_lensing.analysis.plotting import create_plotter_from_configs
from desi_lensing.config import LensGalaxyConfig, OutputConfig, PlotConfig

# Create plotter
plotter = create_plotter_from_configs(
    computation_config, lens_config, source_config, 
    output_config, plot_config, analysis_config, logger
)

# Basic plot
plotter.plot_magnitudes()

# With custom options
plotter.plot_magnitudes(
    magnitude_cuts=[-19.0, -20.0, -21.5],
    apply_extinction_correction=True,
    add_kp3_cut=True,
    save_plot=True,
    filename_suffix="custom"
)
```

## Dependencies

### Required
- `numpy`
- `matplotlib` 
- `astropy`
- `fitsio`

### Optional
- `mpl_scatter_density` (for high-quality density plots)

## Output Files

Plots are saved with descriptive filenames:
- `absolute_magnitudes_bgs_bright.png` (default)
- `absolute_magnitudes_bgs_bright_ecorr.png` (with extinction correction)
- `absolute_magnitudes_bgs_bright_kp3.png` (with KP3 cut)
- `absolute_magnitudes_lrg_custom.png` (with custom suffix)

## FastSpecFit Integration

The system automatically:
1. Attempts to load real FastSpecFit magnitude data from `/pscratch/sd/i/ioannis/fastspecfit/data/loa/catalogs/`
2. Determines program type (`bright` for BGS, `dark` for LRG/ELG)
3. Uses the specified DESI release (`iron` or `loa`) to select appropriate data
4. Joins magnitude data with lens catalogues via `TARGETID`
5. Falls back to mock data for testing if real data unavailable

## Configuration

All magnitude-related parameters are CLI-only and not part of configuration files:
- `--release`: DESI release to use ('iron' for current, 'loa' for future) 
- `--magnitude-cuts`: Comma-separated magnitude cuts per redshift bin
- `--mag-col`: Magnitude column name (default: `ABSMAG01_SDSS_R`)
- `--apply-extinction-correction`: Enable extinction correction
- `--add-kp3-cut`: Add KP3 reference line

## Example Output

The plots show:
- Dense cloud of galaxies in magnitude-redshift space
- Clear magnitude cuts stepping down with redshift
- Redshift bin boundaries as vertical lines
- Survival fraction text annotation
- Inverted y-axis (brighter magnitudes at top)

## Integration with Existing Framework

The functionality integrates seamlessly with:
- Existing configuration system
- Path management
- Logging framework
- Output directory structure
- CLI argument parsing
- Plot styling system 