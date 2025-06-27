# DESI Galaxy-Galaxy Lensing Analysis Pipeline (v2.0)

A modern, modular pipeline for computing galaxy-galaxy lensing signals using DESI lens galaxies and various source surveys.

## Overview

This is a complete refactoring of the DESI lensing pipeline, moving away from configuration files to a command-line interface with proper argument parsing. The pipeline supports computation of multiple lensing statistics across different source surveys with full tomographic analysis.

## Features

- **Command-line interface**: No more config files! All parameters specified via CLI arguments
- **Modern Python architecture**: Type hints, dataclasses, proper logging, modular design
- **Multiple statistics**: Delta Sigma and Gamma_t with optional B-mode analysis
- **B-mode analysis**: Compute B-modes (45-degree rotated source galaxies) for systematic checks
- **Multiple source surveys**: DES, KiDS, HSC Y1/Y3, SDSS
- **Tomographic analysis**: Full support for tomographic bins
- **Comprehensive validation**: Configuration validation with helpful error messages
- **Flexible cosmology**: Support for Planck18, WMAP9, and custom wCDM
- **HPC-optimized**: Designed for efficient execution on computing clusters

## Installation

```bash
cd desi_lensing_refactored
pip install -e .
```

### Dependencies

- Python ≥ 3.8
- numpy ≥ 1.20.0
- scipy ≥ 1.7.0
- astropy ≥ 5.0.0
- healpy ≥ 1.15.0
- click ≥ 8.0.0
- dsigma (DESI lensing computation library)
- camb (for magnification bias calculations)

## Quick Start

### Basic Delta Sigma computation
```bash
desi-lensing compute deltasigma --galaxy-type BGS_BRIGHT --source-surveys DES,KiDS --z-bins "0.1,0.2,0.3,0.4"
```

### Gamma_t with tomography
```bash
desi-lensing compute gammat --galaxy-type LRG --tomography --source-surveys HSCY1,HSCY3
```

### B-mode analysis for systematics checks
```bash
# Delta Sigma B-modes
desi-lensing compute deltasigma --galaxy-type BGS_BRIGHT --source-surveys DES --bmodes

# Gamma_t B-modes  
desi-lensing compute gammat --galaxy-type ELG --source-surveys KiDS --bmodes
```

### Show available options
```bash
desi-lensing compute deltasigma --help
desi-lensing show-defaults --galaxy-type BGS_BRIGHT
desi-lensing list-surveys
```

## Command Reference

### Main Commands

- `desi-lensing compute deltasigma`: Compute ΔΣ statistic
- `desi-lensing compute gammat`: Compute γₜ statistic  
- `desi-lensing convert-config`: Convert old INI config to new format
- `desi-lensing show-defaults`: Show default parameter values
- `desi-lensing list-surveys`: List available source surveys

### Key Options

#### Galaxy Configuration
- `--galaxy-type`: BGS_BRIGHT, LRG, or ELG
- `--z-bins`: Comma-separated redshift bin edges
- `--randoms`: Which random catalogues to use (e.g., "1,2")
- `--magnitude-cuts/--no-magnitude-cuts`: Apply magnitude cuts

#### Source Surveys
- `--source-surveys`: Comma-separated list (DES,KiDS,HSCY1,HSCY3,SDSS)
- `--cut-to-desi/--no-cut-to-desi`: Cut source catalogues to DESI footprint

#### Computation
- `--cosmology`: planck18, wmap9, or wcdm
- `--n-jobs`: Number of parallel jobs (0=auto)
- `--tomography/--no-tomography`: Enable tomographic analysis
- `--comoving/--physical`: Coordinate system choice
- `--n-jackknife`: Number of jackknife fields
- `--bmodes/--no-bmodes`: Compute B-modes (45-degree rotated shears)

#### Binning
- `--rp-min/--rp-max`: Radial bin range [Mpc/h] for ΔΣ
- `--theta-min/--theta-max`: Angular bin range [arcmin] for γₜ
- `--n-rp-bins/--n-theta-bins`: Number of bins
- `--binning`: log or linear binning

#### Output
- `--output-dir`: Output directory path
- `--save-precomputed/--no-save-precomputed`: Save intermediate results
- `--apply-blinding/--no-blinding`: Apply blinding to results

## B-mode Analysis

B-modes are computed by rotating source galaxy shears by 45 degrees. This is useful for systematic checks, as the B-mode signal should be consistent with zero for a robust analysis. B-modes can be computed for both Delta Sigma and Gamma_t statistics:

```bash
# Delta Sigma B-modes
desi-lensing compute deltasigma --bmodes --galaxy-type BGS_BRIGHT --source-surveys DES

# Gamma_t B-modes  
desi-lensing compute gammat --bmodes --galaxy-type LRG --source-surveys KiDS
```

B-mode results are saved with the `bmodes_` prefix in the filename to distinguish them from regular measurements. Additionally, precomputed tables for B-mode analysis are saved separately with the same `bmodes_` prefix, ensuring that B-mode and normal computations don't interfere with each other's cached results.

## Configuration Migration

Convert old INI configuration files to the new format:

```bash
desi-lensing convert-config old_config.ini -o new_config.yaml
```

## Architecture

### Modular Design

```
desi_lensing/
├── config/          # Configuration management
├── core/            # Core computation classes  
├── data/            # Data loading utilities
├── cli/             # Command-line interface
└── utils/           # Utility functions
```

### Key Classes

- **Configuration Classes**: Type-safe configuration with validation
  - `ComputationConfig`: Computation parameters and cosmology
  - `LensGalaxyConfig`: Lens galaxy selection and binning
  - `SourceSurveyConfig`: Source survey settings and corrections
  - `OutputConfig`: Output paths and file naming

- **Computation Classes**: Modular computation architecture
  - `LensingComputation`: Base computation class with B-mode support
  - `DeltaSigmaComputation`: ΔΣ computation
  - `GammaTComputation`: γₜ computation  

- **Pipeline**: Main orchestrator
  - `LensingPipeline`: Coordinates the entire analysis workflow

### Data Flow

1. **Configuration**: Parse CLI arguments into typed config objects
2. **Validation**: Comprehensive validation with helpful error messages
3. **Data Loading**: Load lens and source catalogues with proper corrections
4. **Computation**: For each (lens_bin, source_survey, statistic):
   - Apply B-mode rotation if requested (45-degree shear rotation)
   - Precompute lensing signals using dsigma
   - Compute jackknife covariance matrices
   - Apply systematic corrections
   - Save results with proper naming
5. **Output**: Save FITS tables and covariance matrices

## Default Parameters

### Lens Galaxy Types
- **BGS_BRIGHT**: z ∈ [0.1, 0.2, 0.3, 0.4]
- **LRG**: z ∈ [0.4, 0.6, 0.8, 1.1]  
- **ELG**: z ∈ [0.8, 1.1, 1.6]

### Source Surveys
- **DES**: 4 tomographic bins, matrix shear response correction
- **KiDS**: 5 tomographic bins, scalar shear response correction
- **HSC Y1**: 4 tomographic bins, full correction suite
- **HSC Y3**: 4 tomographic bins, updated corrections
- **SDSS**: 1 bin, basic corrections

### Computation
- **Cosmology**: Planck18 with H₀=100 km/s/Mpc
- **Binning**: 15 logarithmic bins
- **ΔΣ range**: 0.08 - 80 Mpc/h
- **γₜ range**: 0.3 - 300 arcmin
- **Jackknife**: 100 fields
- **B-modes**: Disabled by default

## Examples

### Reproduce Iron Results
```bash
# BGS Delta Sigma with DES
desi-lensing compute deltasigma \
  --galaxy-type BGS_BRIGHT \
  --source-surveys DES \
  --z-bins "0.1,0.2,0.3,0.4" \
  --bgs-version v1.5 \
  --no-tomography

# LRG with multiple surveys
desi-lensing compute deltasigma \
  --galaxy-type LRG \
  --source-surveys DES,KiDS,HSCY1 \
  --z-bins "0.4,0.6,0.8,1.1" \
  --tomography
```

### Custom Analysis
```bash
# High-resolution gamma_t analysis
desi-lensing compute gammat \
  --galaxy-type ELG \
  --source-surveys HSCY3 \
  --theta-min 0.1 \
  --theta-max 500.0 \
  --n-theta-bins 20 \
  --n-jackknife 200 \
  --cosmology wcdm \
  --w0 -0.9 \
  --wa -0.1
```

### B-mode Validation
```bash
# B-mode systematics check for Delta Sigma
desi-lensing compute deltasigma \
  --galaxy-type BGS_BRIGHT \
  --source-surveys DES,KiDS \
  --bmodes \
  --no-blinding \
  --output-dir ./bmodes_check/

# B-mode systematics check for Gamma_t
desi-lensing compute gammat \
  --galaxy-type LRG \
  --source-surveys HSCY1 \
  --bmodes \
  --no-blinding
```

## File Naming Convention

Output files follow a standardized naming pattern:
```
{statistic}_{galaxy_type}_zmin_{z_min}_zmax_{z_max}_blind{blind}_boost_{boost}.fits
```

For tomographic analysis:
```
{statistic}_{galaxy_type}_zmin_{z_min}_zmax_{z_max}_lenszbin_{lens_bin}_blind{blind}_boost_{boost}.fits
```

For B-mode analysis, files are prefixed with `bmodes_`:
```
bmodes_{statistic}_{galaxy_type}_zmin_{z_min}_zmax_{z_max}_blind{blind}_boost_{boost}.fits
```

## Error Handling

The pipeline includes comprehensive error handling and validation:

- **Configuration validation**: Check parameter compatibility
- **File existence checks**: Verify required catalogues exist
- **Cross-component validation**: Ensure lens-source compatibility
- **Graceful degradation**: Handle missing randoms, empty bins, etc.
- **Detailed logging**: Full trace of computations and any issues

## Performance Considerations

- **Parallelization**: Use `--n-jobs` to control CPU usage
- **Precomputed tables**: Enable `--save-precomputed` to cache intermediate results
- **Memory management**: Handles large catalogues efficiently
- **HPC compatibility**: Designed for SLURM job submission

## Development

### Code Quality
- Type hints throughout
- Comprehensive docstrings
- Modular, testable design
- Proper error handling and logging

### Testing
```bash
pytest tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Migration Guide

### From v1.x (config files)
1. Convert your config file: `desi-lensing convert-config old.ini`
2. Review the generated YAML for any needed adjustments
3. Use the CLI commands with equivalent parameters

### Breaking Changes from v1.x
- Configuration files no longer supported (use CLI args)
- Some parameter names have changed for clarity
- Output directory structure slightly modified
- Requires Python ≥ 3.8

## License

MIT License - see LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

> DESI Collaboration et al. (2024). "Galaxy-Galaxy Lensing in the Dark Energy Spectroscopic Instrument". *In prep*.

## Support

- Documentation: [Read the Docs](https://desi-lensing.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/desihub/desi-lensing/issues)
- Discussions: [DESI Collaboration Slack #lensing channel] 