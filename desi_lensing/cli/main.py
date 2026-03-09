"""Main command-line interface for DESI lensing pipeline."""

# Fix OpenBLAS threading issues on HPC systems before importing numerical libraries
# This prevents "Program is Terminated. Because you tried to allocate too many memory regions"
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "20")
os.environ.setdefault("OMP_NUM_THREADS", "20")

import click
import sys
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np

from ..config import (
    ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig, PlotConfig, AnalysisConfig,
    ConfigValidator
)
from ..core.pipeline import LensingPipeline
from ..utils.logging_utils import setup_logger
from .config_converter import convert_config_file


@click.group()
@click.version_option(version="2.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.pass_context
def cli(ctx, verbose, log_file):
    """DESI Galaxy-Galaxy Lensing Analysis Pipeline."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # Setup logging
    level = logging.INFO if verbose else logging.WARNING
    log_path = Path(log_file) if log_file else None
    logger = setup_logger('desi_lensing', log_path, level)
    ctx.obj['logger'] = logger


@cli.group()
def compute():
    """Compute lensing statistics."""
    pass


@cli.group()
def plot():
    """Plot lensing analysis results."""
    pass


@cli.group()
def randoms():
    """Generate and analyze random data vectors for systematics testing."""
    pass


def add_galaxy_options(func):
    """Add lens galaxy configuration options."""
    func = click.option('--galaxy-type', 
                       default='BGS_BRIGHT,LRG',
                       help='Comma-separated list of lens galaxy types (BGS_BRIGHT, LRG, ELG)')(func)
    func = click.option('--release',
                       type=click.Choice(['iron', 'loa']),
                       default='iron',
                       help='DESI release to use (iron=current, loa=future)')(func)
    func = click.option('--z-bins-bgs-bright', 
                       help='Comma-separated redshift bin edges for BGS_BRIGHT (e.g., "0.1,0.2,0.3,0.4")')(func)
    func = click.option('--z-bins-lrg', 
                       help='Comma-separated redshift bin edges for LRG (e.g., "0.4,0.6,0.8,1.1")')(func)
    func = click.option('--z-bins-elg', 
                       help='Comma-separated redshift bin edges for ELG (e.g., "0.8,1.1,1.6")')(func)
    func = click.option('--bgs-version', default='v1.5', help='BGS catalogue version')(func)
    func = click.option('--lrg-version', default='v1.5', help='LRG catalogue version')(func)
    func = click.option('--elg-version', default='v1.5', help='ELG catalogue version')(func)
    func = click.option('--randoms', default='1,2', help='Comma-separated random indices')(func)
    func = click.option('--randoms-ratio', type=float, default=-1.0,
                       help='Ratio of randoms to lenses. If >= 0, subsample randoms to this ratio * len(lenses)')(func)
    func = click.option('--magnitude-cuts/--no-magnitude-cuts', default=True, 
                       help='Apply magnitude cuts')(func)
    func = click.option('--mstar-complete', is_flag=True, help='Use stellar mass complete sample')(func)
    return func


def add_source_options(func):
    """Add source survey configuration options."""
    func = click.option('--source-surveys', 
                       default='DECADE,DES,KiDS,HSCY3,SDSS',
                       help='Comma-separated list of source surveys (DES, KiDS, HSCY1, HSCY3, SDSS, DECADE, DECADE_NGC, DECADE_SGC)')(func)
    func = click.option('--cut-to-desi/--no-cut-to-desi', default=True,
                       help='Cut source catalogues to DESI footprint')(func)
    return func


def add_computation_options(func):
    """Add computation configuration options."""
    func = click.option('--cosmology', 
                       type=click.Choice(['planck18', 'wmap9', 'wcdm']),
                       default='planck18', help='Cosmology to use')(func)
    func = click.option('--h0', type=float, default=100.0, help='Hubble constant')(func)
    func = click.option('--w0', type=float, help='Dark energy w0 parameter (for wCDM)')(func)
    func = click.option('--wa', type=float, help='Dark energy wa parameter (for wCDM)')(func)
    func = click.option('--n-jobs', type=int, default=0, help='Number of parallel jobs (0=auto)')(func)
    func = click.option('--tomography/--no-tomography', default=True, 
                       help='Use tomographic analysis')(func)
    func = click.option('--comoving/--physical', default=True,
                       help='Use comoving vs physical coordinates')(func)
    func = click.option('--lens-source-cut', type=float, default=None,
                       help='Lens-source separation cut (None for no cut)')(func)
    func = click.option('--n-jackknife', type=int, default=100,
                       help='Number of jackknife fields')(func)
    func = click.option('--bmodes/--no-bmodes', default=False,
                       help='Compute B-modes (45-degree rotated shears)')(func)
    func = click.option('--GPU/--CPU', 'use_gpu', default=True,
                       help='Use GPU vs CPU for computation')(func)
    func = click.option('--force-shared/--do-not-force-shared', 'force_shared_flag', default=None,
                       help='Force shared memory for GPU computation (auto-decided if not specified)')(func)
    func = click.option('--split-by', default=None,
                       help='Comma-separated list of properties to split by (e.g., NTILE,LOGMSTAR)')(func)
    func = click.option('--n-splits', type=int, default=4,
                       help='Number of splits to create (default: 4)')(func)
    return func


def add_binning_options(func):
    """Add binning configuration options."""
    func = click.option('--rp-min', type=float, default=0.08, help='Minimum rp [Mpc/h]')(func)
    func = click.option('--rp-max', type=float, default=80.0, help='Maximum rp [Mpc/h]')(func)
    func = click.option('--n-rp-bins', type=int, default=15, help='Number of rp bins')(func)
    func = click.option('--theta-min', type=float, default=0.3, help='Minimum theta [arcmin]')(func)
    func = click.option('--theta-max', type=float, default=300.0, help='Maximum theta [arcmin]')(func)
    func = click.option('--n-theta-bins', type=int, default=15, help='Number of theta bins')(func)
    func = click.option('--binning', type=click.Choice(['log', 'linear']), default='log',
                       help='Binning type')(func)
    return func


def add_output_options(func):
    """Add output configuration options."""
    func = click.option('--output-dir', type=click.Path(),
                       default='/pscratch/sd/s/sven/lensing_measurements/',
                       help='Output directory')(func)
    func = click.option('--catalogue-path', type=click.Path(),
                       default='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/',
                       help='Catalogue base path for lens galaxies')(func)
    func = click.option('--source-catalogue-path', type=click.Path(),
                       default='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/',
                       help='Catalogue base path for source galaxies')(func)
    func = click.option('--save-precomputed/--no-save-precomputed', default=True,
                       help='Save precomputed tables')(func)
    func = click.option('--apply-blinding/--no-blinding', default=True,
                       help='Apply blinding to results')(func)
    return func


def add_common_options(func):
    """
    Add all common options to a command (comprehensive set).
    
    This is a convenience decorator that applies all option groups.
    For more targeted option sets, use specific decorators like
    @add_galaxy_options, @add_source_options, etc.
    """
    func = add_galaxy_options(func)
    func = add_source_options(func)
    func = add_computation_options(func)
    func = add_binning_options(func)
    func = add_output_options(func)
    return func


def add_basic_options(func):
    """
    Add basic options for typical usage (simplified set).
    
    This is a simpler alternative to @add_common_options that includes
    only the most commonly used options.
    """
    # Basic galaxy options
    func = click.option('--galaxy-type', 
                       default='BGS_BRIGHT,LRG',
                       help='Comma-separated list of lens galaxy types (BGS_BRIGHT, LRG, ELG)')(func)
    func = click.option('--z-bins-bgs-bright', 
                       help='Comma-separated redshift bin edges for BGS_BRIGHT (e.g., "0.1,0.2,0.3,0.4")')(func)
    func = click.option('--z-bins-lrg', 
                       help='Comma-separated redshift bin edges for LRG (e.g., "0.4,0.6,0.8,1.1")')(func)
    func = click.option('--z-bins-elg', 
                       help='Comma-separated redshift bin edges for ELG (e.g., "0.8,1.1,1.6")')(func)
    
    # Basic source options
    func = click.option('--source-surveys', 
                       default='DECADE,DES,KiDS,HSCY3',
                       help='Comma-separated list of source surveys')(func)
    
    # Basic computation options
    func = click.option('--n-jobs', type=int, default=0, help='Number of parallel jobs (0=auto)')(func)
    func = click.option('--GPU/--CPU', 'use_gpu', default=True, help='Use GPU vs CPU')(func)
    
    # Basic output options
    func = click.option('--output-dir', type=click.Path(),
                       default='/pscratch/sd/s/sven/lensing_measurements/',
                       help='Output directory')(func)
    return func


# Keep the old definition for backward compatibility but mark as deprecated
def _add_common_options_legacy(func):
    """DEPRECATED: Use add_common_options or add_basic_options instead."""
    # Galaxy configuration
    func = click.option('--galaxy-type', 
                       default='BGS_BRIGHT,LRG',
                       help='Comma-separated list of lens galaxy types (BGS_BRIGHT, LRG, ELG)')(func)
    func = click.option('--release',
                       type=click.Choice(['iron', 'loa']),
                       default='iron',
                       help='DESI release to use (iron=current, loa=future)')(func)
    func = click.option('--z-bins-bgs-bright', 
                       help='Comma-separated redshift bin edges for BGS_BRIGHT (e.g., "0.1,0.2,0.3,0.4")')(func)
    func = click.option('--z-bins-lrg', 
                       help='Comma-separated redshift bin edges for LRG (e.g., "0.4,0.6,0.8,1.1")')(func)
    func = click.option('--z-bins-elg', 
                       help='Comma-separated redshift bin edges for ELG (e.g., "0.8,1.1,1.6")')(func)
    func = click.option('--bgs-version', default='v1.5', help='BGS catalogue version')(func)
    func = click.option('--lrg-version', default='v1.5', help='LRG catalogue version')(func)
    func = click.option('--elg-version', default='v1.5', help='ELG catalogue version')(func)
    func = click.option('--randoms', default='1,2', help='Comma-separated random indices')(func)
    func = click.option('--randoms-ratio', type=float, default=-1.0,
                       help='Ratio of randoms to lenses. If >= 0, subsample randoms to this ratio * len(lenses)')(func)
    func = click.option('--magnitude-cuts/--no-magnitude-cuts', default=True, 
                       help='Apply magnitude cuts')(func)
    func = click.option('--mstar-complete', is_flag=True, help='Use stellar mass complete sample')(func)
    
    # Source survey configuration
    func = click.option('--source-surveys', 
                       default='DECADE,DES,KiDS,HSCY3,SDSS',
                       help='Comma-separated list of source surveys')(func)
    func = click.option('--cut-to-desi/--no-cut-to-desi', default=True,
                       help='Cut source catalogues to DESI footprint')(func)
    
    # Computation configuration
    func = click.option('--cosmology', 
                       type=click.Choice(['planck18', 'wmap9', 'wcdm']),
                       default='planck18', help='Cosmology to use')(func)
    func = click.option('--h0', type=float, default=100.0, help='Hubble constant')(func)
    func = click.option('--w0', type=float, help='Dark energy w0 parameter (for wCDM)')(func)
    func = click.option('--wa', type=float, help='Dark energy wa parameter (for wCDM)')(func)
    func = click.option('--n-jobs', type=int, default=0, help='Number of parallel jobs (0=auto)')(func)
    func = click.option('--tomography/--no-tomography', default=True, 
                       help='Use tomographic analysis')(func)
    func = click.option('--comoving/--physical', default=True,
                       help='Use comoving vs physical coordinates')(func)
    func = click.option('--lens-source-cut', type=float, default=None,
                       help='Lens-source separation cut (None for no cut)')(func)
    func = click.option('--n-jackknife', type=int, default=100,
                       help='Number of jackknife fields')(func)
    func = click.option('--bmodes/--no-bmodes', default=False,
                       help='Compute B-modes (45-degree rotated shears)')(func)
    
    # GPU computation options
    func = click.option('--GPU/--CPU', 'use_gpu', default=True,
                       help='Use GPU vs CPU for computation')(func)
    func = click.option('--force-shared/--do-not-force-shared', 'force_shared_flag', default=None,
                       help='Force shared memory for GPU computation (auto-decided if not specified)')(func)
    
    # Binning configuration
    func = click.option('--rp-min', type=float, default=0.08, help='Minimum rp [Mpc/h]')(func)
    func = click.option('--rp-max', type=float, default=80.0, help='Maximum rp [Mpc/h]')(func)
    func = click.option('--n-rp-bins', type=int, default=15, help='Number of rp bins')(func)
    func = click.option('--theta-min', type=float, default=0.3, help='Minimum theta [arcmin]')(func)
    func = click.option('--theta-max', type=float, default=300.0, help='Maximum theta [arcmin]')(func)
    func = click.option('--n-theta-bins', type=int, default=15, help='Number of theta bins')(func)
    func = click.option('--binning', type=click.Choice(['log', 'linear']), default='log',
                       help='Binning type')(func)
    
    # Output configuration
    func = click.option('--output-dir', type=click.Path(),
                       default='/pscratch/sd/s/sven/lensing_measurements/',
                       help='Output directory')(func)
    func = click.option('--catalogue-path', type=click.Path(),
                       default='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/',
                       help='Catalogue base path for lens galaxies')(func)
    func = click.option('--source-catalogue-path', type=click.Path(),
                       default='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/',
                       help='Catalogue base path for source galaxies')(func)
    func = click.option('--save-precomputed/--no-save-precomputed', default=True,
                       help='Save precomputed tables')(func)
    func = click.option('--apply-blinding/--no-blinding', default=True,
                       help='Apply blinding to results')(func)
    
    return func


def add_plotting_options(func):
    """Add plotting-specific options."""
    # Plot style options
    func = click.option('--style', 
                       type=click.Choice(['paper', 'presentation', 'notebook']),
                       default='paper', help='Matplotlib style preset')(func)
    func = click.option('--log-scale/--linear-scale', default=False,
                       help='Use logarithmic vs linear scale for y-axis')(func)
    func = click.option('--transparent/--opaque', default=False,
                       help='Use transparent background')(func)
    func = click.option('--filename-suffix', default='',
                       help='Suffix to add to output filenames')(func)
    func = click.option('--no-save', is_flag=True,
                       help="Don't save plots to file (show only)")(func)
    
    return func


def parse_comma_separated_floats(value: str) -> List[float]:
    """Parse comma-separated float values."""
    if not value:
        return []
    return [float(x.strip()) for x in value.split(',')]


def parse_comma_separated_ints(value: str) -> List[int]:
    """Parse comma-separated integer values.""" 
    if not value:
        return []
    return [int(x.strip()) for x in value.split(',')]


def parse_comma_separated_strings(value: str) -> List[str]:
    """Parse comma-separated string values."""
    if not value:
        return []
    return [x.strip() for x in value.split(',')]


def create_configs_from_args(**kwargs) -> Tuple[ComputationConfig, List[LensGalaxyConfig], SourceSurveyConfig, OutputConfig, PlotConfig, AnalysisConfig]:
    """Create configuration objects from CLI arguments."""
    
    # Parse lists from strings
    randoms = parse_comma_separated_ints(kwargs.get('randoms', '1,2'))
    source_surveys = parse_comma_separated_strings(kwargs.get('source_surveys', 'DES,KiDS,HSCY1,HSCY3'))
    galaxy_types = parse_comma_separated_strings(kwargs.get('galaxy_type', 'BGS_BRIGHT,LRG'))
    split_by = parse_comma_separated_strings(kwargs.get('split_by', '')) if kwargs.get('split_by') else None
    
    # Parse per-galaxy-type z-bins overrides
    z_bins_overrides = {}
    if kwargs.get('z_bins_bgs_bright'):
        z_bins_overrides['BGS_BRIGHT'] = parse_comma_separated_floats(kwargs['z_bins_bgs_bright'])
    if kwargs.get('z_bins_lrg'):
        z_bins_overrides['LRG'] = parse_comma_separated_floats(kwargs['z_bins_lrg'])
    if kwargs.get('z_bins_elg'):
        z_bins_overrides['ELG'] = parse_comma_separated_floats(kwargs['z_bins_elg'])
    
    # Validate galaxy types
    valid_galaxy_types = ['BGS_BRIGHT', 'LRG', 'ELG']
    for galaxy_type in galaxy_types:
        if galaxy_type not in valid_galaxy_types:
            raise ValueError(f"Invalid galaxy type '{galaxy_type}'. Valid options: {valid_galaxy_types}")
    
    # Handle GPU/CPU defaults
    use_gpu = kwargs.get('use_gpu', True)
    force_shared_flag = kwargs.get('force_shared_flag')
    
    # Set n_jobs defaults based on GPU usage
    n_jobs = kwargs.get('n_jobs', 0)
    if n_jobs == 0:  # If not explicitly set
        if use_gpu:
            n_jobs = 4
        else:
            n_jobs = 0  # Will be resolved to cpu_count() in computation
    
    # Create configurations
    computation_config = ComputationConfig(
        cosmology=kwargs['cosmology'],
        h0=kwargs['h0'],
        w0=kwargs.get('w0'),
        wa=kwargs.get('wa'),
        n_jobs=n_jobs,
        comoving=kwargs['comoving'],
        lens_source_cut=kwargs.get('lens_source_cut'),
        n_jackknife_fields=kwargs['n_jackknife'],
        rp_min=kwargs['rp_min'],
        rp_max=kwargs['rp_max'],
        n_rp_bins=kwargs['n_rp_bins'],
        theta_min=kwargs['theta_min'],
        theta_max=kwargs['theta_max'],
        n_theta_bins=kwargs['n_theta_bins'],
        binning=kwargs['binning'],
        tomography=kwargs['tomography'],
        bmodes=kwargs.get('bmodes', False),
        use_gpu=use_gpu,
        force_shared=force_shared_flag,
        split_by=split_by,
        n_splits=kwargs.get('n_splits', 4),
    )
    
    # Create lens configurations for each galaxy type
    lens_configs = []
    for galaxy_type in galaxy_types:
        lens_config = LensGalaxyConfig(
            galaxy_type=galaxy_type,
            release=kwargs['release'],
            bgs_catalogue_version=kwargs['bgs_version'],
            lrg_catalogue_version=kwargs['lrg_version'],
            elg_catalogue_version=kwargs['elg_version'],
            which_randoms=randoms,
            randoms_ratio=kwargs['randoms_ratio'],
            magnitude_cuts=kwargs['magnitude_cuts'],
            mstar_complete=kwargs.get('mstar_complete', False),
            weight_type=kwargs.get('weight_type', 'WEIGHT'),
        )
        
        # Apply per-galaxy-type z-bins override, or use defaults
        if galaxy_type in z_bins_overrides:
            lens_config.z_bins = z_bins_overrides[galaxy_type]
        else:
            lens_config.use_default_z_bins()
        
        lens_configs.append(lens_config)
    
    source_config = SourceSurveyConfig(
        surveys=source_surveys,
        cut_catalogues_to_desi=kwargs['cut_to_desi'],
    )
    
    output_config = OutputConfig(
        catalogue_path=kwargs['catalogue_path'],
        source_catalogue_path=kwargs['source_catalogue_path'],
        save_path=kwargs['output_dir'],
        save_precomputed=kwargs['save_precomputed'],
        apply_blinding=kwargs['apply_blinding'],
        verbose=kwargs.get('verbose', True),
    )
    
    plot_config = PlotConfig(
        style=kwargs.get('style', 'paper'),
        transparent_background=kwargs.get('transparent', False),
        log_scale=kwargs.get('log_scale', False),
        filename_suffix=kwargs.get('filename_suffix', ''),
        save_plots=not kwargs.get('no_save', False),
    )
    
    analysis_config = AnalysisConfig()
    
    # Sync n_bins_per_galaxy_type with actual lens configs when user specifies fewer bins
    # (e.g., --z-bins-bgs-bright 0.1,0.4 gives 1 bin instead of default 3)
    # Don't increase bins beyond hardcoded defaults (e.g., LRG computed in 3 bins but only 2 analyzed)
    for lens_config in lens_configs:
        n_bins_from_config = lens_config.get_n_lens_bins()
        n_bins_hardcoded = analysis_config.get_n_bins_for_galaxy_type(lens_config.galaxy_type)
        if n_bins_from_config < n_bins_hardcoded:
            analysis_config.set_n_bins_for_galaxy_type(lens_config.galaxy_type, n_bins_from_config)
    
    return computation_config, lens_configs, source_config, output_config, plot_config, analysis_config


def _execute_computation(ctx, statistic: str, statistic_display_name: str, **kwargs):
    """
    Execute a lensing computation for a given statistic.
    
    This helper function consolidates common logic for all computation commands.
    
    Parameters
    ----------
    ctx : click.Context
        Click context object
    statistic : str
        Statistic name ('deltasigma' or 'gammat')
    statistic_display_name : str
        Human-readable statistic name for logging
    **kwargs : dict
        Command-line arguments
    """
    logger = ctx.obj['logger']
    
    if kwargs.get('bmodes', False):
        logger.info(f"Starting {statistic_display_name} computation with B-modes (45-degree source galaxy rotation)")
    else:
        logger.info(f"Starting {statistic_display_name} computation")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, _, analysis_config = create_configs_from_args(**kwargs)
    comp_config.statistics = [statistic]
    
    # Validate configuration
    validator = ConfigValidator(comp_config, lens_configs[0], source_config, output_config)
    if not validator.is_valid():
        logger.error("Configuration validation failed:")
        logger.error(validator.get_error_summary())
        sys.exit(1)
    
    # Show warnings
    warnings = validator.get_warnings()
    for warning in warnings:
        logger.warning(warning)
    
    # Check if splits analysis is requested
    if comp_config.split_by is not None:
        logger.info(f"Running splits analysis for properties: {comp_config.split_by}")
        from ..analysis.splits import create_splits_analyzer_from_configs
        
        splits_analyzer = create_splits_analyzer_from_configs(
            comp_config, lens_configs[0], source_config, output_config, logger
        )
        splits_analyzer.run()
        
        logger.info(f"Splits computation completed for {statistic_display_name}")
    else:
        # Run standard pipeline
        pipeline = LensingPipeline(comp_config, lens_configs[0], source_config, output_config, logger, analysis_config=analysis_config)
        pipeline.run()
        
        computation_type = f"{statistic_display_name} B-mode" if kwargs.get('bmodes', False) else statistic_display_name
        logger.info(f"{computation_type} computation completed")


@compute.command()
@add_common_options
@click.pass_context
def deltasigma(ctx, **kwargs):
    """Compute Delta Sigma statistic."""
    _execute_computation(ctx, 'deltasigma', 'Delta Sigma', **kwargs)


@compute.command()
@add_common_options
@click.pass_context
def gammat(ctx, **kwargs):
    """Compute Gamma_t statistic."""
    _execute_computation(ctx, 'gammat', 'Gamma_t', **kwargs)


def _setup_plotter_command(ctx, **kwargs):
    """
    Common setup for plotting commands.
    
    This helper consolidates configuration creation, style setup, and plotter
    initialization that is shared across all plotting commands.
    
    Parameters
    ----------
    ctx : click.Context
        Click context object
    **kwargs : dict
        Command-line arguments
        
    Returns
    -------
    tuple
        (logger, comp_config, lens_configs, source_config, output_config, 
         plot_config, analysis_config)
    """
    from ..analysis.plotting_utils import setup_matplotlib_style
    
    logger = ctx.obj['logger']
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = create_configs_from_args(**kwargs)
    
    # Setup matplotlib style
    setup_matplotlib_style(plot_config.style)
    
    return logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config


@plot.command()
@add_common_options
@add_plotting_options
@click.option('--statistic', 
              type=click.Choice(['deltasigma', 'gammat']),
              default='deltasigma',
              help='Lensing statistic to plot')
@click.pass_context
def datavector(ctx, **kwargs):
    """Plot data vectors (tomographic or non-tomographic)."""
    from ..analysis.plotting import create_multi_galaxy_plotter_from_configs
    
    logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = _setup_plotter_command(ctx, **kwargs)
    logger.info(f"Plotting data vectors for {kwargs['statistic']}")
    
    comp_config.statistics = [kwargs['statistic']]
    
    # Create multi-galaxy plotter
    plotter = create_multi_galaxy_plotter_from_configs(
        comp_config, lens_configs, source_config, output_config, plot_config, analysis_config, logger
    )
    
    # Generate plots
    if comp_config.tomography:
        plotter.plot_datavector_tomographic(
            statistic=kwargs['statistic'],
            log_scale=plot_config.log_scale,
            save_plot=plot_config.save_plots,
            filename_suffix=plot_config.filename_suffix
        )
    else:
        logger.warning("Non-tomographic plotting not yet implemented")
    
    logger.info("Data vector plotting completed")


@plot.command(name='covariance-comparison')
@add_common_options
@add_plotting_options
@click.option('--statistic', 
              type=click.Choice(['deltasigma', 'gammat']),
              default='deltasigma',
              help='Lensing statistic to plot')
@click.option('--bmodes/--no-bmodes', default=False,
              help='Compare B-mode (pure noise) covariances instead of signal covariances')
@click.pass_context
def covariance_comparison(ctx, **kwargs):
    """Plot comparison of theory and jackknife covariance uncertainties.
    
    Creates tomographic plots showing:
    - Theoretical uncertainty (sqrt of diagonal) as lines
    - Jackknife uncertainty (sqrt of diagonal) as 'o' markers
    
    This is useful for validating covariance matrices and understanding
    where jackknife and theory estimates agree or differ.
    
    Use --bmodes to compare B-mode (pure noise) covariances, which are used
    for null tests with 45-degree rotated source galaxy shapes.
    
    When multiple galaxy types are specified (e.g., --galaxy-type BGS_BRIGHT,LRG),
    creates a combined plot with columns for each galaxy type's bins.
    
    Example usage:
    
    \b
    # Compare signal covariances
    desi-lensing plot covariance-comparison --galaxy-type BGS_BRIGHT,LRG
    
    \b
    # Compare B-mode covariances  
    desi-lensing plot covariance-comparison --galaxy-type BGS_BRIGHT,LRG --bmodes
    """
    from ..analysis.plotting import create_multi_galaxy_plotter_from_configs
    
    logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = _setup_plotter_command(ctx, **kwargs)
    
    mode_str = "B-mode " if kwargs['bmodes'] else ""
    logger.info(f"Plotting {mode_str}covariance comparison for {kwargs['statistic']}")
    
    comp_config.statistics = [kwargs['statistic']]
    
    # Create multi-galaxy plotter
    plotter = create_multi_galaxy_plotter_from_configs(
        comp_config, lens_configs, source_config, output_config, plot_config, analysis_config, logger
    )
    
    # Generate plots
    if comp_config.tomography:
        plotter.plot_covariance_comparison_tomographic(
            statistic=kwargs['statistic'],
            log_scale=plot_config.log_scale,
            save_plot=plot_config.save_plots,
            filename_suffix=plot_config.filename_suffix,
            bmodes=kwargs['bmodes']
        )
    else:
        logger.warning("Non-tomographic covariance comparison not yet implemented")
    
    logger.info(f"{mode_str.capitalize() if mode_str else ''}Covariance comparison plotting completed")


@plot.command()
@add_common_options
@add_plotting_options
@click.option('--statistic', 
              type=click.Choice(['deltasigma', 'gammat']),
              default='deltasigma',
              help='Lensing statistic to plot')
@click.pass_context
def bmodes(ctx, **kwargs):
    """Plot B-mode diagnostics for systematics testing.
    
    When multiple galaxy types are specified (e.g., --galaxy-type BGS_BRIGHT,LRG),
    creates a combined plot with columns for each galaxy type's bins.
    """
    from ..analysis.plotting import create_plotter_from_configs, create_multi_galaxy_plotter_from_configs
    from ..analysis.plotting_utils import save_p_values_table
    
    logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = _setup_plotter_command(ctx, **kwargs)
    
    galaxy_types = [lc.galaxy_type for lc in lens_configs]
    logger.info(f"Plotting B-mode diagnostics for {kwargs['statistic']} with galaxy types: {galaxy_types}")
    
    comp_config.statistics = [kwargs['statistic']]
    comp_config.bmodes = True  # Force B-modes for this analysis
    
    # Use multi-galaxy mode if more than one galaxy type is requested
    if len(lens_configs) > 1:
        logger.info(f"Using multi-galaxy mode for {len(lens_configs)} galaxy types")
        
        # Create multi-galaxy plotter
        plotter = create_multi_galaxy_plotter_from_configs(
            comp_config, lens_configs, source_config, output_config, plot_config, analysis_config, logger
        )
        
        # Generate combined plots
        if comp_config.tomography:
            pvalues = plotter.plot_bmodes_tomographic(
                statistic=kwargs['statistic'],
                save_plot=plot_config.save_plots,
                filename_suffix=plot_config.filename_suffix
            )
            
            # Save p-values table for combined results
            if pvalues and plot_config.save_plots:
                galaxy_suffix = "_".join(galaxy_types)
                table_path = plotter.plot_dir / f"pvalues_bmodes_{kwargs['statistic']}_{galaxy_suffix}.tex"
                # Save combined p-values table (using first lens_config for basic table formatting)
                save_p_values_table(
                    pvalues, lens_configs[0], source_config, table_path,
                    statistic=kwargs['statistic'],
                    caption=f"B-mode p-values for {kwargs['statistic']} measurements ({', '.join(galaxy_types)})",
                    precision=3
                )
                logger.info(f"Combined B-mode table saved to {table_path}")
        else:
            logger.warning("Non-tomographic B-mode plotting not yet implemented")
    else:
        # Single galaxy type - use standard plotter
        plotter = create_plotter_from_configs(
            comp_config, lens_configs[0], source_config, output_config, plot_config, analysis_config, logger
        )
        
        # Generate plots
        if comp_config.tomography:
            pvalues = plotter.plot_bmodes_tomographic(
                statistic=kwargs['statistic'],
                save_plot=plot_config.save_plots,
                filename_suffix=plot_config.filename_suffix
            )
            
            # Save p-values table
            if pvalues and plot_config.save_plots:
                table_path = plotter.plot_dir / f"pvalues_bmodes_{kwargs['statistic']}.tex"
                save_p_values_table(
                    pvalues, lens_configs[0], source_config, table_path,
                    statistic=kwargs['statistic'],
                    caption=f"B-mode p-values for {kwargs['statistic']} measurements",
                    precision=3
                )
                logger.info(f"B-mode table saved to {table_path}")
        else:
            logger.warning("Non-tomographic B-mode plotting not yet implemented")
    
    logger.info("B-mode plotting completed")


@plot.command(name='randoms')
@add_common_options
@add_plotting_options
@click.option('--statistic', 
              type=click.Choice(['deltasigma', 'gammat']),
              default='deltasigma',
              help='Lensing statistic to plot')
@click.pass_context
def plot_randoms(ctx, **kwargs):
    """Plot random lens tests for systematics testing."""
    from ..analysis.plotting import create_plotter_from_configs
    
    logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = _setup_plotter_command(ctx, **kwargs)
    logger.info(f"Plotting random lens tests for {kwargs['statistic']}")
    
    comp_config.statistics = [kwargs['statistic']]
    
    # Create plotter
    plotter = create_plotter_from_configs(
        comp_config, lens_configs[0], source_config, output_config, plot_config, analysis_config, logger
    )
    
    # Generate plots
    plotter.plot_randoms_test(
        statistic=kwargs['statistic'],
        save_plot=plot_config.save_plots,
        filename_suffix=plot_config.filename_suffix
    )
    
    logger.info("Random lens test plotting completed")


@plot.command()
@add_common_options
@add_plotting_options
@click.option('--statistic', 
              type=click.Choice(['deltasigma', 'gammat']),
              default='deltasigma',
              help='Lensing statistic to plot')
@click.pass_context
def survey_comparison(ctx, **kwargs):
    """Plot survey comparison with tomographic bins as rows."""
    from ..analysis.plotting import create_plotter_from_configs
    
    logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = _setup_plotter_command(ctx, **kwargs)
    logger.info(f"Plotting survey comparison for {kwargs['statistic']}")
    
    comp_config.statistics = [kwargs['statistic']]
    
    # Create plotter for first lens config (supports single galaxy type)
    plotter = create_plotter_from_configs(
        comp_config, lens_configs[0], source_config, output_config, plot_config, analysis_config, logger
    )
    
    # Generate plots
    if comp_config.tomography:
        plotter.plot_survey_comparison_tomographic(
            statistic=kwargs['statistic'],
            log_scale=plot_config.log_scale,
            save_plot=plot_config.save_plots,
            filename_suffix=plot_config.filename_suffix
        )
    else:
        logger.warning("Non-tomographic survey comparison not yet implemented")
    
    logger.info("Survey comparison plotting completed")


@plot.command()
@add_common_options
@add_plotting_options
@click.option('--magnitude-cuts',
              help='Comma-separated magnitude cuts per redshift bin (e.g., "-19.5,-20.5,-21.0")')
@click.option('--mag-col', default='ABSMAG01_SDSS_R',
              help='Magnitude column name')
@click.option('--apply-extinction-correction/--no-extinction-correction', default=False,
              help='Apply extinction correction: ecorr = -0.8*(z-0.1)')
@click.option('--add-kp3-cut/--no-kp3-cut', default=False,
              help='Add KP3 cut at -21.5 mag as horizontal red line')
@click.pass_context
def magnitudes(ctx, **kwargs):
    """Plot absolute magnitude distributions with magnitude cuts and redshift bin boundaries."""
    from ..analysis.plotting import create_plotter_from_configs
    
    logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = _setup_plotter_command(ctx, **kwargs)
    logger.info("Plotting magnitude distributions")
    
    # Parse magnitude cuts if provided
    magnitude_cuts = None
    if kwargs.get('magnitude_cuts'):
        try:
            magnitude_cuts = [float(x.strip()) for x in kwargs['magnitude_cuts'].split(',')]
        except ValueError:
            logger.error("Invalid magnitude cuts format. Use comma-separated floats like '-19.5,-20.5,-21.0'")
            sys.exit(1)
    
    # Plot for each galaxy type
    for lens_config in lens_configs:
        logger.info(f"Plotting magnitudes for {lens_config.galaxy_type}")
        
        # Create plotter for this galaxy type
        plotter = create_plotter_from_configs(
            comp_config, lens_config, source_config, output_config, plot_config, analysis_config, logger
        )
        
        # Generate plot
        plotter.plot_magnitudes(
            magnitude_cuts=magnitude_cuts,
            mag_col=kwargs['mag_col'],
            apply_extinction_correction=kwargs['apply_extinction_correction'],
            add_kp3_cut=kwargs['add_kp3_cut'],
            save_plot=plot_config.save_plots,
            filename_suffix=plot_config.filename_suffix
        )
    
    logger.info("Magnitude plotting completed")


@plot.command()
@add_common_options
@add_plotting_options
@click.option('--statistic', 
              type=click.Choice(['deltasigma', 'gammat']),
              default='deltasigma',
              help='Lensing statistic to plot')
@click.option('--split-by',
              default='NTILE,ra,dec',
              help='Comma-separated list of properties to plot splits for (e.g., "NTILE,ra,dec,LOGMSTAR")')
@click.option('--n-splits', type=int, default=4,
              help='Number of splits')
@click.option('--scale-categories',
              default='small scales,large scales,all scales',
              help='Comma-separated list of scale categories')
@click.option('--use-randoms-uncertainty/--use-covariance-uncertainty', default=False,
              help='Use randoms for slope uncertainty estimation vs covariance')
@click.option('--plot-slope/--no-plot-slope', default=True,
              help='Plot the fitted slope line')
@click.option('--plot-slope-uncertainty/--no-plot-slope-uncertainty', default=True,
              help='Plot slope uncertainty band')
@click.option('--critical-sigma', type=float, default=3.0,
              help='Highlight slopes more significant than this (in sigma)')
@click.option('--boost-correction/--no-boost-correction', default=False,
              help='Whether boost correction was applied in splits computation')
@click.pass_context
def splits_plot(ctx, **kwargs):
    """Plot amplitude vs split property values for systematics testing.
    
    This creates plots showing how lensing signal amplitudes vary with 
    split properties (e.g., NTILE, RA, DEC, LOGMSTAR) for different lens bins 
    and scale categories. A non-zero slope indicates systematic bias.
    
    When multiple galaxy types are specified (e.g., --galaxy-type BGS_BRIGHT,LRG),
    all galaxy types are plotted in a single combined figure with columns for each
    lens bin across all galaxy types.
    """
    from ..analysis.plotting import create_multi_galaxy_plotter_from_configs
    
    logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = _setup_plotter_command(ctx, **kwargs)
    
    galaxy_types = [lc.galaxy_type for lc in lens_configs]
    logger.info(f"Plotting splits for {kwargs['statistic']} with galaxy types: {galaxy_types}")
    
    comp_config.statistics = [kwargs['statistic']]
    
    # Parse split properties and scale categories
    splits_to_plot = parse_comma_separated_strings(kwargs['split_by'])
    scale_categories = parse_comma_separated_strings(kwargs['scale_categories'])
    
    # Create multi-galaxy plotter with all lens configs
    plotter = create_multi_galaxy_plotter_from_configs(
        comp_config, lens_configs, source_config, output_config, plot_config, analysis_config, logger
    )
    
    # Plot all splits in a single combined figure
    results = plotter.plot_all_splits(
        splits_to_consider=splits_to_plot,
        n_splits=kwargs['n_splits'],
        statistic=kwargs['statistic'],
        scale_categories=scale_categories,
        use_randoms_uncertainty=kwargs['use_randoms_uncertainty'],
        plot_slope=kwargs['plot_slope'],
        plot_slope_uncertainty=kwargs['plot_slope_uncertainty'],
        critical_sigma=kwargs['critical_sigma'],
        save_plot=plot_config.save_plots,
        filename_suffix=plot_config.filename_suffix,
        boost_correction=kwargs['boost_correction'],
        verbose=ctx.obj.get('verbose', True)
    )
    
    # Log significant slopes
    for key, result in results.items():
        slope = result.get('slope')
        slope_err = result.get('slope_error')
        if slope is not None and slope_err is not None and slope_err > 0:
            significance = abs(slope) / slope_err
            if significance > kwargs['critical_sigma']:
                logger.warning(f"Significant slope detected: {key} = {slope:.3f} ± {slope_err:.3f} ({significance:.1f}σ)")
    
    logger.info("Splits plotting completed")


@plot.command()
@add_common_options
@add_plotting_options
@click.option('--statistic', 
              type=click.Choice(['deltasigma', 'gammat']),
              default='deltasigma',
              help='Lensing statistic to plot')
@click.option('--cosmologies',
              default='planck18,wcdm',
              help='Comma-separated list of cosmologies to compare')
@click.pass_context
def compare_cosmologies(ctx, **kwargs):
    """Compare results from different cosmological models."""
    from ..analysis.plotting import create_plotter_from_configs
    
    logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = _setup_plotter_command(ctx, **kwargs)
    logger.info(f"Comparing cosmologies for {kwargs['statistic']}")
    
    comp_config.statistics = [kwargs['statistic']]
    
    # Parse cosmologies
    cosmologies = parse_comma_separated_strings(kwargs['cosmologies'])
    
    # Create cosmology configs dict
    cosmology_configs = {}
    for cosmo in cosmologies:
        cosmo_config = ComputationConfig(
            statistics=[kwargs['statistic']],
            cosmology=cosmo,
            h0=kwargs['h0'],
            tomography=kwargs['tomography'],
            rp_min=kwargs['rp_min'],
            rp_max=kwargs['rp_max'],
            n_rp_bins=kwargs['n_rp_bins'],
        )
        cosmology_configs[cosmo] = cosmo_config
    
    # Create plotter
    plotter = create_plotter_from_configs(
        comp_config, lens_configs[0], source_config, output_config, plot_config, analysis_config, logger
    )
    
    # Generate plots
    plotter.compare_different_cosmologies(
        cosmology_configs,
        statistic=kwargs['statistic'],
        save_plot=plot_config.save_plots,
        filename_suffix=plot_config.filename_suffix
    )
    
    logger.info("Cosmology comparison plotting completed")


@plot.command(name='source-redshift-slope')
@add_common_options
@add_plotting_options
@click.option('--statistic', 
              type=click.Choice(['deltasigma', 'gammat']),
              default='deltasigma',
              help='Lensing statistic to plot')
@click.option('--scale-categories',
              default='small scales,large scales,all scales',
              help='Comma-separated list of scale categories')
@click.option('--use-theory-covariance/--use-jackknife-covariance', default=True,
              help='Use theory vs jackknife covariance matrices')
@click.option('--use-all-bins/--use-allowed-bins', default=False,
              help='Use all source bins vs only allowed bins per lens bin')
@click.option('--plot-slope/--no-plot-slope', default=True,
              help='Plot the fitted slope line')
@click.option('--plot-slope-uncertainty/--no-plot-slope-uncertainty', default=True,
              help='Plot slope uncertainty band')
@click.option('--compute-sigma-sys/--no-sigma-sys', default=True,
              help='Compute systematic uncertainty from amplitude scatter')
@click.option('--sigma-sys-method',
              type=click.Choice(['bayesian', 'reduced_chisq']),
              default='reduced_chisq',
              help='Method for sigma_sys calculation')
@click.option('--critical-sigma', type=float, default=3.0,
              help='Highlight slopes more significant than this (in sigma)')
@click.option('--slope-color', default='black',
              help='Color for the fitted slope line')
@click.option('--hscy3-deltaz-shifts',
              help='Comma-separated HSCY3 source redshift shifts per bin (e.g., "0,0,0.115,0.192")')
@click.pass_context
def source_redshift_slope_plot(ctx, **kwargs):
    """
    Plot lensing amplitudes vs source redshift with slope fitting.
    
    This diagnostic plot shows how lensing signal amplitudes vary with source
    redshift for different lens bins and scale categories. A non-zero slope
    could indicate systematic biases related to source redshift estimation
    or calibration issues.
    
    When multiple galaxy types are specified (e.g., --galaxy-type BGS_BRIGHT,LRG),
    creates a combined plot with columns for each galaxy type's bins.
    
    Example usage:
    
    \b
    desi-lensing plot source-redshift-slope --galaxy-type BGS_BRIGHT,LRG \\
        --source-surveys KiDS,DES,HSCY3,DECADE \\
        --use-theory-covariance --scale-categories "small scales,large scales"
    """
    from ..analysis.plotting import create_plotter_from_configs, create_multi_galaxy_plotter_from_configs
    
    logger, comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = _setup_plotter_command(ctx, **kwargs)
    logger.info(f"Plotting source redshift slope for {kwargs['statistic']}")
    
    comp_config.statistics = [kwargs['statistic']]
    
    # Parse scale categories
    scale_categories = parse_comma_separated_strings(kwargs['scale_categories'])
    
    # Parse HSCY3 deltaz shifts if provided
    hscy3_deltaz_shifts = None
    if kwargs.get('hscy3_deltaz_shifts'):
        hscy3_deltaz_shifts = parse_comma_separated_floats(kwargs['hscy3_deltaz_shifts'])
    
    # Use multi-galaxy mode if multiple galaxy types, else single-galaxy mode
    if len(lens_configs) > 1:
        logger.info(f"Using multi-galaxy mode for {len(lens_configs)} galaxy types")
        
        plotter = create_multi_galaxy_plotter_from_configs(
            comp_config, lens_configs, source_config, output_config, plot_config, analysis_config, logger
        )
        
        # Generate combined plot (hscy3_deltaz_shifts not supported for multi-galaxy yet)
        if hscy3_deltaz_shifts is not None:
            logger.warning("hscy3_deltaz_shifts not supported for multi-galaxy plotter, ignoring")
        
        results = plotter.plot_source_redshift_slope_tomographic(
            statistic=kwargs['statistic'],
            scale_categories=scale_categories,
            save_plot=plot_config.save_plots,
            filename_suffix=plot_config.filename_suffix,
            plot_slope=kwargs['plot_slope'],
            plot_slope_uncertainty=kwargs['plot_slope_uncertainty'],
            compute_sigma_sys=kwargs['compute_sigma_sys'],
            sigma_sys_method=kwargs['sigma_sys_method'],
            use_all_bins=kwargs['use_all_bins'],
            use_theory_covariance=kwargs['use_theory_covariance'],
            critical_sigma=kwargs['critical_sigma'],
            slope_color=kwargs['slope_color']
        )
        
        # Log summary of results
        for key, slope in results.get('slopes', {}).items():
            slope_err = results.get('slope_errors', {}).get(key, np.nan)
            if np.isfinite(slope) and np.isfinite(slope_err):
                significance = np.abs(slope) / slope_err if slope_err > 0 else 0
                logger.info(f"  {key}: slope={slope:.3f}±{slope_err:.3f} ({significance:.1f}σ)")
    else:
        # Single galaxy type - create plotter for each lens config
        for lens_config in lens_configs:
            logger.info(f"Processing galaxy type: {lens_config.galaxy_type}")
            
            plotter = create_plotter_from_configs(
                comp_config, lens_config, source_config, output_config, plot_config, analysis_config, logger
            )
            
            # Generate plot
            results = plotter.plot_source_redshift_slope_tomographic(
                statistic=kwargs['statistic'],
                scale_categories=scale_categories,
                save_plot=plot_config.save_plots,
                filename_suffix=plot_config.filename_suffix,
                plot_slope=kwargs['plot_slope'],
                plot_slope_uncertainty=kwargs['plot_slope_uncertainty'],
                compute_sigma_sys=kwargs['compute_sigma_sys'],
                sigma_sys_method=kwargs['sigma_sys_method'],
                use_all_bins=kwargs['use_all_bins'],
                use_theory_covariance=kwargs['use_theory_covariance'],
                hscy3_deltaz_shifts=hscy3_deltaz_shifts,
                critical_sigma=kwargs['critical_sigma'],
                slope_color=kwargs['slope_color']
            )
            
            # Log summary of results
            for key, slope in results.get('slopes', {}).items():
                slope_err = results.get('slope_errors', {}).get(key, np.nan)
                if np.isfinite(slope) and np.isfinite(slope_err):
                    significance = np.abs(slope) / slope_err if slope_err > 0 else 0
                    logger.info(f"  {key}: slope={slope:.3f}±{slope_err:.3f} ({significance:.1f}σ)")
    
    logger.info("Source redshift slope plotting completed")


@plot.command()
@add_galaxy_options
@add_source_options
@add_output_options
@add_plotting_options
@click.option('--surveys-to-overlay',
              help='Comma-separated list of surveys to overlay (e.g., "KiDS,DES,UNIONS"). '
                   'If not specified, uses all configured source surveys plus UNIONS.')
@click.option('--nside', type=int, default=64,
              help='HEALPix nside parameter for map resolution')
@click.option('--smoothing', type=float, default=0.5,
              help='Smoothing scale in degrees for survey boundaries')
@click.option('--sep', type=int, default=30,
              help='Number of graticules per degree for grid')
@click.option('--plot-ratio/--plot-density', default=False,
              help='Plot ratio of observed/target galaxies vs density')
@click.option('--survey-colors',
              help='JSON-formatted dictionary of survey colors (e.g., \'{"KiDS":"blue","DES":"red"}\')')
@click.option('--survey-alphas',
              help='JSON-formatted dictionary of survey alpha values (e.g., \'{"UNIONS":0.9}\')')
@click.option('--survey-label-positions',
              help='JSON-formatted dictionary of survey label positions in degrees (e.g., \'{"KiDS":[0,60]}\')')
@click.pass_context
def footprint(ctx, **kwargs):
    """Plot survey footprints on sky map with lens galaxy density."""
    from ..analysis.plotting import create_footprint_plotter_from_configs
    import json
    
    logger = ctx.obj['logger']
    logger.info("Plotting survey footprints")
    
    # Check if footprint plotting is available
    try:
        from ..analysis.plotting import FOOTPRINT_AVAILABLE
        if not FOOTPRINT_AVAILABLE:
            logger.error(
                "Footprint plotting requires skymapper and healpy packages. "
                "Install them with: pip install skymapper healpy"
            )
            sys.exit(1)
    except ImportError:
        logger.error(
            "Footprint plotting requires skymapper and healpy packages. "
            "Install them with: pip install skymapper healpy"
        )
        sys.exit(1)
    
    # Create minimal configurations for footprint plotting
    # Parse lists from strings
    randoms = parse_comma_separated_ints(kwargs.get('randoms', '1,2'))
    source_surveys = parse_comma_separated_strings(kwargs.get('source_surveys', 'DES,KiDS,HSCY1,HSCY3'))
    galaxy_types = parse_comma_separated_strings(kwargs.get('galaxy_type', 'BGS_BRIGHT'))
    
    # Parse per-galaxy-type z-bins overrides
    z_bins_overrides = {}
    if kwargs.get('z_bins_bgs_bright'):
        z_bins_overrides['BGS_BRIGHT'] = parse_comma_separated_floats(kwargs['z_bins_bgs_bright'])
    if kwargs.get('z_bins_lrg'):
        z_bins_overrides['LRG'] = parse_comma_separated_floats(kwargs['z_bins_lrg'])
    if kwargs.get('z_bins_elg'):
        z_bins_overrides['ELG'] = parse_comma_separated_floats(kwargs['z_bins_elg'])
    
    # Create minimal computation config (not used for footprints but required by plotter)
    computation_config = ComputationConfig(n_jobs=1)
    
    # Create lens configurations for each galaxy type
    lens_configs = []
    for galaxy_type in galaxy_types:
        lens_config = LensGalaxyConfig(
            galaxy_type=galaxy_type,
            release=kwargs.get('release', 'iron'),
            bgs_catalogue_version=kwargs.get('bgs_version', 'v1.5'),
            lrg_catalogue_version=kwargs.get('lrg_version', 'v1.5'),
            elg_catalogue_version=kwargs.get('elg_version', 'v1.5'),
            which_randoms=randoms,
            randoms_ratio=kwargs.get('randoms_ratio', -1.0),
            magnitude_cuts=kwargs.get('magnitude_cuts', True),
            mstar_complete=kwargs.get('mstar_complete', False),
        )
        
        # Apply per-galaxy-type z-bins override, or use defaults
        if galaxy_type in z_bins_overrides:
            lens_config.z_bins = z_bins_overrides[galaxy_type]
        else:
            lens_config.use_default_z_bins()
        
        lens_configs.append(lens_config)
    
    # Create source config
    source_config = SourceSurveyConfig(
        surveys=source_surveys,
        cut_catalogues_to_desi=kwargs.get('cut_to_desi', True),
    )
    
    # Create output config
    output_config = OutputConfig(
        catalogue_path=kwargs.get('catalogue_path', '/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/'),
        source_catalogue_path=kwargs.get('source_catalogue_path', '/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/'),
        save_path=kwargs.get('output_dir', '/pscratch/sd/s/sven/lensing_measurements/'),
        save_precomputed=kwargs.get('save_precomputed', True),
        apply_blinding=kwargs.get('apply_blinding', False),
        verbose=ctx.obj.get('verbose', False),
    )
    
    # Create plot config
    plot_config = PlotConfig(
        style=kwargs.get('style', 'paper'),
        transparent_background=kwargs.get('transparent', False),
        filename_suffix=kwargs.get('filename_suffix', ''),
        save_plots=not kwargs.get('no_save', False),
    )
    
    # Setup matplotlib style
    try:
        from ..analysis.plotting_utils import setup_matplotlib_style
        setup_matplotlib_style(plot_config.style)
    except ImportError:
        logger.warning("Could not load plotting_utils, using default matplotlib style")
    
    # Parse surveys to overlay
    surveys_to_overlay = None
    if kwargs.get('surveys_to_overlay'):
        surveys_to_overlay = parse_comma_separated_strings(kwargs['surveys_to_overlay'])
    
    # Parse JSON parameters if provided
    colors = None
    if kwargs.get('survey_colors'):
        try:
            colors = json.loads(kwargs['survey_colors'])
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for survey-colors")
            sys.exit(1)
    
    alphas = None
    if kwargs.get('survey_alphas'):
        try:
            alphas = json.loads(kwargs['survey_alphas'])
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for survey-alphas")
            sys.exit(1)
    
    label_positions = None
    if kwargs.get('survey_label_positions'):
        try:
            label_positions_raw = json.loads(kwargs['survey_label_positions'])
            # Convert lists to tuples
            label_positions = {k: tuple(v) for k, v in label_positions_raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.error("Invalid JSON format for survey-label-positions")
            sys.exit(1)
    
    # Temporarily disable cut_to_desi to get full footprints
    original_cut_setting = source_config.cut_catalogues_to_desi
    source_config.cut_catalogues_to_desi = False
    
    # Plot for each lens galaxy type
    for lens_config in lens_configs:
        logger.info(f"Plotting footprint for {lens_config.galaxy_type}")
        
        # Create footprint plotter
        plotter = create_footprint_plotter_from_configs(
            computation_config, lens_config, source_config, output_config, plot_config, logger
        )
        
        # Generate footprint plot
        plotter.plot_lens_footprint(
            surveys_to_overlay=surveys_to_overlay,
            nside=kwargs['nside'],
            smoothing=kwargs['smoothing'],
            sep=kwargs['sep'],
            colors=colors,
            alphas=alphas,
            label_positions=label_positions,
            plot_ratio=kwargs['plot_ratio'],
            save_plot=plot_config.save_plots,
            filename_suffix=plot_config.filename_suffix
        )
    
    # Restore original cut_to_desi setting
    source_config.cut_catalogues_to_desi = original_cut_setting
    
    logger.info("Footprint plotting completed")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output YAML file')
@click.pass_context
def convert_config(ctx, config_file, output):
    """Convert old INI config file to new YAML format."""
    logger = ctx.obj['logger']
    logger.info(f"Converting config file: {config_file}")
    
    config_path = Path(config_file)
    output_path = Path(output) if output else config_path.with_suffix('.yaml')
    
    try:
        convert_config_file(config_path, output_path)
        logger.info(f"Converted config saved to: {output_path}")
    except Exception as e:
        logger.error(f"Config conversion failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--galaxy-type', 
             help='Show defaults for specific galaxy types (comma-separated)')
def show_defaults(galaxy_type):
    """Show default configuration values."""
    click.echo("Default Configuration Values:")
    click.echo("=" * 30)
    
    # Show computation defaults
    comp_config = ComputationConfig()
    click.echo("\nComputation:")
    click.echo(f"  Statistics: {comp_config.statistics}")
    click.echo(f"  Cosmology: {comp_config.cosmology}")
    click.echo(f"  Tomography: {comp_config.tomography}")
    click.echo(f"  Comoving: {comp_config.comoving}")
    click.echo(f"  N jobs: {comp_config.n_jobs}")
    click.echo(f"  Binning: {comp_config.binning}")
    click.echo(f"  rp range: {comp_config.rp_min} - {comp_config.rp_max} Mpc/h ({comp_config.n_rp_bins} bins)")
    click.echo(f"  theta range: {comp_config.theta_min} - {comp_config.theta_max} arcmin ({comp_config.n_theta_bins} bins)")
    click.echo(f"  B-modes: {comp_config.bmodes}")
    click.echo(f"  GPU computation: {comp_config.use_gpu}")
    click.echo(f"  Force shared memory: {comp_config.force_shared}")
    
    # Show lens galaxy defaults
    galaxy_types = galaxy_type.split(',') if galaxy_type else ['BGS_BRIGHT', 'LRG']
    
    click.echo("\nLens Galaxies:")
    click.echo(f"  Default galaxy types: {', '.join(galaxy_types)}")
    
    for gtype in galaxy_types:
        lens_config = LensGalaxyConfig()
        lens_config.galaxy_type = gtype
        lens_config.use_default_z_bins()
        
        click.echo(f"\n  {gtype}:")
        click.echo(f"    Release: {lens_config.release}")
        click.echo(f"    Z bins: {lens_config.z_bins}")
        click.echo(f"    Catalogue versions: BGS={lens_config.bgs_catalogue_version}, LRG={lens_config.lrg_catalogue_version}")
        click.echo(f"    Random indices: {lens_config.which_randoms}")
        click.echo(f"    Magnitude cuts: {lens_config.magnitude_cuts}")
    
    # Show analysis configuration
    analysis_config = AnalysisConfig()
    click.echo("\nAnalysis Configuration:")
    for gtype in galaxy_types:
        n_bins = analysis_config.get_n_bins_for_galaxy_type(gtype)
        click.echo(f"  {gtype}: {n_bins} bins")
    
    total_bins = analysis_config.get_total_bins_for_galaxy_types(galaxy_types)
    click.echo(f"  Total bins: {total_bins}")
    
    # Show source survey defaults
    source_config = SourceSurveyConfig()
    click.echo("\nSource Surveys:")
    click.echo(f"  Surveys: {source_config.surveys}")
    click.echo(f"  Cut to DESI: {source_config.cut_catalogues_to_desi}")
    
    # Show output defaults
    output_config = OutputConfig()
    click.echo("\nOutput:")
    click.echo(f"  Save path: {output_config.save_path}")
    click.echo(f"  Save precomputed: {output_config.save_precomputed}")
    click.echo(f"  Apply blinding: {output_config.apply_blinding}")


@cli.command()
def list_surveys():
    """List available source surveys."""
    surveys = ["DES", "KiDS", "HSCY1", "HSCY3", "SDSS", "DECADE", "DECADE_NGC", "DECADE_SGC"]
    click.echo("Available Source Surveys:")
    click.echo("=" * 40)
    for survey in surveys:
        if survey == "DECADE":
            click.echo(f"  {survey} (combines NGC + SGC)")
        elif survey in ["DECADE_NGC", "DECADE_SGC"]:
            click.echo(f"  {survey}")
        else:
            click.echo(f"  {survey}")


@randoms.command()
@add_common_options
@click.option('--n-randoms', type=int, default=1000, help='Number of random realizations')
@click.option('--n-processes', type=int, default=4, help='Number of parallel processes')
@click.option('--use-theory-covariance/--use-jackknife', default=True,
              help='Use theory vs jackknife covariance')
@click.option('--datavector-type', 
              type=click.Choice(['zero', 'emulator', 'chris', 'measured']),
              default='emulator', help='Type of data vector to use')
@click.option('--filename-suffix', default='', help='Suffix for output files')
@click.option('--use-all-bins/--use-allowed-bins', default=False,
              help='Use all source bins vs only allowed bins per lens bin')
@click.pass_context
def source_redshift_slope(ctx, **kwargs):
    """Generate random realizations for source redshift slope analysis.
    
    For source redshift slope analysis, the entire covariance matrix including
    cross-covariances between surveys is used to generate correlated random
    data vectors.
    
    Example:
    
    \b
    desi-lensing randoms source-redshift-slope --galaxy-type BGS_BRIGHT \\
        --source-surveys KiDS,DES,HSCY3 --use-theory-covariance --n-randoms 1000
    """
    from ..analysis.randoms import create_randoms_analyzer_from_configs
    
    logger = ctx.obj['logger']
    logger.info("Starting source redshift slope randoms generation")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, _, analysis_config = create_configs_from_args(**kwargs)
    
    # Process each galaxy type
    for lens_config in lens_configs:
        logger.info(f"Processing galaxy type: {lens_config.galaxy_type}")
        
        # Create randoms analyzer for this galaxy type
        randoms_analyzer = create_randoms_analyzer_from_configs(
            comp_config, lens_config, source_config, output_config, logger
        )
        
        # Add galaxy type to filename suffix to prevent overwriting
        galaxy_suffix = f"_{lens_config.galaxy_type.lower()}"
        full_suffix = f"{kwargs['filename_suffix']}{galaxy_suffix}"
        
        # Generate random realizations
        randoms_analyzer.generate_random_source_redshift_slope_test(
            n_randoms=kwargs['n_randoms'],
            n_processes=kwargs['n_processes'],
            use_theory_covariance=kwargs['use_theory_covariance'],
            datavector_type=kwargs['datavector_type'],
            filename_suffix=full_suffix,
            use_all_bins=kwargs['use_all_bins']
        )
        
        logger.info(f"Completed source redshift slope randoms for {lens_config.galaxy_type}")
    
    logger.info("Source redshift slope randoms generation completed for all galaxy types")


@randoms.command()
@add_common_options
@click.option('--n-randoms', type=int, default=1000, help='Number of random realizations')
@click.option('--n-processes', type=int, default=4, help='Number of parallel processes')
@click.option('--use-theory-covariance/--use-jackknife', default=False,
              help='Use theory vs jackknife covariance')
@click.option('--datavector-type', 
              type=click.Choice(['zero', 'emulator', 'chris', 'measured']),
              default='measured', help='Type of data vector to use')
@click.option('--splits-to-consider',
              default='NTILE',
              help='Comma-separated list of split properties (e.g., "NTILE,ra,dec"). '
                   'Overrides --split-by if provided.')
@click.option('--scale-categories',
              default='small scales,large scales,all scales',
              help='Comma-separated list of scale categories')
@click.option('--boost-correction/--no-boost-correction', default=False,
              help='Whether boost correction was applied in splits computation')
@click.pass_context
def splits(ctx, **kwargs):
    """Generate random realizations for data splits analysis.
    
    This generates random data vectors for each split, computes lensing
    amplitudes, and fits slopes to assess the null distribution of slope
    values. Results are saved with keys in format:
    '{split_by}_{galaxy_type}_{scale_category}_{lens_bin}'
    
    Example:
    
    \b
    desi-lensing randoms splits --galaxy-type BGS_BRIGHT \\
        --splits-to-consider NTILE,ra,dec --n-splits 4 --n-randoms 1000
    """
    from ..analysis.randoms import create_randoms_analyzer_from_configs
    
    logger = ctx.obj['logger']
    logger.info("Starting splits randoms generation")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, _, analysis_config = create_configs_from_args(**kwargs)
    
    # Parse splits to consider - prefer explicit --splits-to-consider, fall back to --split-by
    splits_to_consider = None
    if kwargs.get('splits_to_consider'):
        splits_to_consider = parse_comma_separated_strings(kwargs['splits_to_consider'])
    elif comp_config.split_by:
        splits_to_consider = comp_config.split_by
    
    # Parse scale categories
    scale_categories = None
    if kwargs.get('scale_categories'):
        scale_categories = parse_comma_separated_strings(kwargs['scale_categories'])
    
    if splits_to_consider:
        logger.info(f"Generating randoms for splits: {splits_to_consider}")
    else:
        logger.info("No splits specified, using default: ['NTILE']")
    
    # Process each galaxy type
    for lens_config in lens_configs:
        logger.info(f"Processing galaxy type: {lens_config.galaxy_type}")
        
        # Create randoms analyzer for this galaxy type
        randoms_analyzer = create_randoms_analyzer_from_configs(
            comp_config, lens_config, source_config, output_config, logger
        )
        
        # Add galaxy type to filename suffix to prevent overwriting
        galaxy_suffix = f"_{lens_config.galaxy_type.lower()}"
        
        # Generate random realizations
        randoms_analyzer.generate_random_splits_test(
            n_randoms=kwargs['n_randoms'],
            n_processes=kwargs['n_processes'],
            use_theory_covariance=kwargs['use_theory_covariance'],
            datavector_type=kwargs['datavector_type'],
            splits_to_consider=splits_to_consider,
            scale_categories=scale_categories,
            n_splits=comp_config.n_splits,
            boost_correction=kwargs['boost_correction'],
            filename_suffix=galaxy_suffix
        )
        
        logger.info(f"Completed splits randoms for {lens_config.galaxy_type}")
    
    logger.info("Splits randoms generation completed for all galaxy types")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main() 