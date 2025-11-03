"""Main command-line interface for DESI lensing pipeline."""

import click
import sys
import logging
from pathlib import Path
from typing import List, Tuple

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


def add_common_options(func):
    """Add common options to compute commands."""
    # Galaxy configuration
    func = click.option('--galaxy-type', 
                       default='BGS_BRIGHT,LRG',
                       help='Comma-separated list of lens galaxy types (BGS_BRIGHT, LRG, ELG)')(func)
    func = click.option('--release',
                       type=click.Choice(['iron', 'loa']),
                       default='iron',
                       help='DESI release to use (iron=current, loa=future)')(func)
    func = click.option('--z-bins', 
                       help='Comma-separated redshift bin edges (e.g., "0.1,0.2,0.3,0.4")')(func)
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
    z_bins = parse_comma_separated_floats(kwargs.get('z_bins', '')) if kwargs.get('z_bins') else None
    randoms = parse_comma_separated_ints(kwargs.get('randoms', '1,2'))
    source_surveys = parse_comma_separated_strings(kwargs.get('source_surveys', 'DES,KiDS,HSCY1,HSCY3'))
    galaxy_types = parse_comma_separated_strings(kwargs.get('galaxy_type', 'BGS_BRIGHT,LRG'))
    
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
        
        if z_bins is not None:
            lens_config.z_bins = z_bins
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
    
    return computation_config, lens_configs, source_config, output_config, plot_config, analysis_config


@compute.command()
@add_common_options
@click.pass_context
def deltasigma(ctx, **kwargs):
    """Compute Delta Sigma statistic."""
    logger = ctx.obj['logger']
    
    if kwargs.get('bmodes', False):
        logger.info("Starting Delta Sigma computation with B-modes (45-degree source galaxy rotation)")
    else:
        logger.info("Starting Delta Sigma computation")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, _, analysis_config = create_configs_from_args(**kwargs)
    comp_config.statistics = ['deltasigma']
    
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
    
    # Run pipeline
    pipeline = LensingPipeline(comp_config, lens_configs[0], source_config, output_config, logger)
    pipeline.run()
    
    computation_type = "Delta Sigma B-mode" if kwargs.get('bmodes', False) else "Delta Sigma"
    logger.info(f"{computation_type} computation completed")


@compute.command()
@add_common_options
@click.pass_context
def gammat(ctx, **kwargs):
    """Compute Gamma_t statistic."""
    logger = ctx.obj['logger']
    
    if kwargs.get('bmodes', False):
        logger.info("Starting Gamma_t computation with B-modes (45-degree source galaxy rotation)")
    else:
        logger.info("Starting Gamma_t computation")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, _, analysis_config = create_configs_from_args(**kwargs)
    comp_config.statistics = ['gammat']
    
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
    
    # Run pipeline
    pipeline = LensingPipeline(comp_config, lens_configs[0], source_config, output_config, logger)
    pipeline.run()
    
    computation_type = "Gamma_t B-mode" if kwargs.get('bmodes', False) else "Gamma_t"
    logger.info(f"{computation_type} computation completed")


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
    from ..analysis.plotting_utils import setup_matplotlib_style
    
    logger = ctx.obj['logger']
    logger.info(f"Plotting data vectors for {kwargs['statistic']}")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = create_configs_from_args(**kwargs)
    comp_config.statistics = [kwargs['statistic']]
    
    # Setup matplotlib style
    setup_matplotlib_style(plot_config.style)
    
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


@plot.command()
@add_common_options
@add_plotting_options
@click.option('--statistic', 
              type=click.Choice(['deltasigma', 'gammat']),
              default='deltasigma',
              help='Lensing statistic to plot')
@click.pass_context
def bmodes(ctx, **kwargs):
    """Plot B-mode diagnostics for systematics testing."""
    from ..analysis.plotting import create_plotter_from_configs
    from ..analysis.plotting_utils import setup_matplotlib_style, save_p_values_table
    
    logger = ctx.obj['logger']
    logger.info(f"Plotting B-mode diagnostics for {kwargs['statistic']}")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = create_configs_from_args(**kwargs)
    comp_config.statistics = [kwargs['statistic']]
    comp_config.bmodes = True  # Force B-modes for this analysis
    
    # Setup matplotlib style
    setup_matplotlib_style(plot_config.style)
    
    # Create plotter
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
    from ..analysis.plotting_utils import setup_matplotlib_style
    
    logger = ctx.obj['logger']
    logger.info(f"Plotting random lens tests for {kwargs['statistic']}")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = create_configs_from_args(**kwargs)
    comp_config.statistics = [kwargs['statistic']]
    
    # Setup matplotlib style
    setup_matplotlib_style(plot_config.style)
    
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
    from ..analysis.plotting_utils import setup_matplotlib_style
    
    logger = ctx.obj['logger']
    logger.info("Plotting magnitude distributions")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = create_configs_from_args(**kwargs)
    
    # Parse magnitude cuts if provided
    magnitude_cuts = None
    if kwargs.get('magnitude_cuts'):
        try:
            magnitude_cuts = [float(x.strip()) for x in kwargs['magnitude_cuts'].split(',')]
        except ValueError:
            logger.error("Invalid magnitude cuts format. Use comma-separated floats like '-19.5,-20.5,-21.0'")
            sys.exit(1)
    
    # Setup matplotlib style
    setup_matplotlib_style(plot_config.style)
    
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
@click.option('--cosmologies',
              default='planck18,wcdm',
              help='Comma-separated list of cosmologies to compare')
@click.pass_context
def compare_cosmologies(ctx, **kwargs):
    """Compare results from different cosmological models."""
    from ..analysis.plotting import create_plotter_from_configs
    from ..analysis.plotting_utils import setup_matplotlib_style
    
    logger = ctx.obj['logger']
    logger.info(f"Comparing cosmologies for {kwargs['statistic']}")
    
    # Create base configurations
    comp_config, lens_configs, source_config, output_config, plot_config, analysis_config = create_configs_from_args(**kwargs)
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
    
    # Setup matplotlib style
    setup_matplotlib_style(plot_config.style)
    
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
    surveys = ["DES", "KiDS", "HSCY1", "HSCY3", "SDSS"]
    click.echo("Available Source Surveys:")
    click.echo("=" * 25)
    for survey in surveys:
        click.echo(f"  {survey}")


@randoms.command()
@add_common_options
@click.option('--n-randoms', type=int, default=1000, help='Number of random realizations')
@click.option('--n-jobs', type=int, default=4, help='Number of parallel processes')
@click.option('--use-theory-covariance/--use-jackknife', default=True,
              help='Use theory vs jackknife covariance')
@click.option('--datavector-type', 
              type=click.Choice(['zero', 'emulator', 'measured']),
              default='emulator', help='Type of data vector to use')
@click.option('--filename-suffix', default='', help='Suffix for output files')
@click.pass_context
def source_redshift_slope(ctx, **kwargs):
    """Generate random realizations for source redshift slope analysis."""
    from ..analysis.randoms import create_randoms_analyzer_from_configs
    
    logger = ctx.obj['logger']
    logger.info("Starting source redshift slope randoms generation")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, _, analysis_config = create_configs_from_args(**kwargs)
    
    # Create randoms analyzer
    randoms_analyzer = create_randoms_analyzer_from_configs(
        comp_config, lens_configs[0], source_config, output_config, logger
    )
    
    # Generate random realizations
    randoms_analyzer.generate_random_source_redshift_slope_test(
        n_randoms=kwargs['n_randoms'],
        n_processes=kwargs['n_jobs'],
        use_theory_covariance=kwargs['use_theory_covariance'],
        datavector_type=kwargs['datavector_type'],
        filename_suffix=kwargs['filename_suffix']
    )
    
    logger.info("Source redshift slope randoms generation completed")


@randoms.command()
@add_common_options
@click.option('--n-randoms', type=int, default=1000, help='Number of random realizations')
@click.option('--n-jobs', type=int, default=4, help='Number of parallel processes')
@click.option('--use-theory-covariance/--use-jackknife', default=False,
              help='Use theory vs jackknife covariance')
@click.option('--datavector-type', 
              type=click.Choice(['zero', 'emulator', 'measured']),
              default='measured', help='Type of data vector to use')
@click.pass_context
def splits(ctx, **kwargs):
    """Generate random realizations for data splits analysis."""
    from ..analysis.randoms import create_randoms_analyzer_from_configs
    
    logger = ctx.obj['logger']
    logger.info("Starting splits randoms generation")
    
    # Create configurations
    comp_config, lens_configs, source_config, output_config, _, analysis_config = create_configs_from_args(**kwargs)
    
    # Create randoms analyzer
    randoms_analyzer = create_randoms_analyzer_from_configs(
        comp_config, lens_configs[0], source_config, output_config, logger
    )
    
    # Generate random realizations
    randoms_analyzer.generate_random_splits_test(
        n_randoms=kwargs['n_randoms'],
        n_processes=kwargs['n_jobs'],
        use_theory_covariance=kwargs['use_theory_covariance'],
        datavector_type=kwargs['datavector_type']
    )
    
    logger.info("Splits randoms generation completed")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main() 