#!/usr/bin/env python3
"""
Example script demonstrating how to use the plotting functionality
with the refactored DESI lensing pipeline CLI.

This script shows how to:
1. Use the integrated CLI commands for plotting
2. Generate various types of plots via command-line interface
3. Create and use the DataVectorPlotter programmatically
4. Save results and generate LaTeX tables
"""

import logging
import subprocess
import sys
from pathlib import Path

from desi_lensing.config import (
    ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig, PlotConfig
)
from desi_lensing.analysis.plotting import DataVectorPlotter, create_plotter_from_configs
from desi_lensing.analysis.plotting_utils import setup_matplotlib_style, save_p_values_table
from desi_lensing.utils.logging_utils import setup_logger


def demonstrate_cli_usage():
    """Demonstrate the CLI plotting commands."""
    print("=== CLI Plotting Examples ===\n")
    
    # Basic data vector plot
    cmd1 = [
        "python", "-m", "desi_lensing.cli.main", "plot", "datavector",
        "--galaxy-type", "LRG",
        "--source-surveys", "HSCY3,KiDS", 
        "--statistic", "deltasigma",
        "--style", "paper",
        "--output-dir", "./cli_plots/",
        "--filename-suffix", "example"
    ]
    print("1. Basic data vector plot:")
    print("   " + " ".join(cmd1))
    print()
    
    # B-mode diagnostics
    cmd2 = [
        "python", "-m", "desi_lensing.cli.main", "plot", "bmodes",
        "--galaxy-type", "LRG",
        "--source-surveys", "HSCY3",
        "--statistic", "deltasigma", 
        "--style", "paper",
        "--output-dir", "./cli_plots/",
        "--verbose"
    ]
    print("2. B-mode diagnostics:")
    print("   " + " ".join(cmd2))
    print()
    
    # Random lens tests
    cmd3 = [
        "python", "-m", "desi_lensing.cli.main", "plot", "randoms",
        "--galaxy-type", "BGS_BRIGHT",
        "--source-surveys", "DES",
        "--statistic", "deltasigma",
        "--style", "presentation",
        "--log-scale",
        "--output-dir", "./cli_plots/"
    ]
    print("3. Random lens tests:")
    print("   " + " ".join(cmd3))
    print()
    
    # Cosmology comparison
    cmd4 = [
        "python", "-m", "desi_lensing.cli.main", "plot", "compare-cosmologies",
        "--galaxy-type", "LRG",
        "--source-surveys", "HSCY3",
        "--statistic", "deltasigma",
        "--cosmologies", "planck18,wcdm",
        "--style", "notebook",
        "--output-dir", "./cli_plots/"
    ]
    print("4. Cosmology comparison:")
    print("   " + " ".join(cmd4))
    print()
    
    # Show defaults
    cmd5 = ["python", "-m", "desi_lensing.cli.main", "show-defaults", "--galaxy-type", "LRG"]
    print("5. Show default parameters:")
    print("   " + " ".join(cmd5))
    print()
    
    print("Note: These commands assume you have computed the lensing statistics first.")
    print("To run them, copy and paste into your terminal or use subprocess.run() in Python.")


def run_cli_example():
    """Actually run a CLI example (requires data)."""
    logger = setup_logger('cli_plotting_example', level=logging.INFO)
    logger.info("Running CLI plotting example...")
    
    # Try to run a simple command
    cmd = [
        sys.executable, "-m", "desi_lensing.cli.main", 
        "show-defaults", "--galaxy-type", "LRG"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("CLI Output:")
        print(result.stdout)
        logger.info("CLI command executed successfully")
    except subprocess.CalledProcessError as e:
        logger.warning(f"CLI command failed (expected without package installation): {e}")
        print("CLI execution failed - this is expected if the package isn't installed")
    except FileNotFoundError:
        logger.warning("CLI command not found - package likely not installed")
        print("CLI not available - package needs to be installed first")


def programmatic_example():
    """Example using the plotting classes directly (programmatic approach)."""
    
    # Setup logging
    logger = setup_logger('programmatic_plotting_example', level=logging.INFO)
    logger.info("Starting programmatic plotting example")
    
    # Setup matplotlib style
    setup_matplotlib_style("paper")
    
    # Create configuration objects using the same defaults as CLI
    computation_config = ComputationConfig(
        statistics=['deltasigma'],
        cosmology='planck18',
        h0=100.0,
        n_jobs=4,
        comoving=True,
        lens_source_cut=None,
        n_jackknife_fields=100,
        rp_min=0.08,
        rp_max=80.0,
        n_rp_bins=15,
        binning='log',
        tomography=True,
        bmodes=False,
    )
    
    lens_config = LensGalaxyConfig(
        galaxy_type='LRG',
        lrg_catalogue_version='v1.5',
        which_randoms=[1, 2],
        magnitude_cuts=True,
        mstar_complete=False,
    )
    # Use default z-bins for LRG
    lens_config.use_default_z_bins()
    
    source_config = SourceSurveyConfig(
        surveys=['HSCY3', 'KiDS'],
        cut_catalogues_to_desi=True,
        # HSC Y3 settings
        hscy3_scalar_shear_response_correction=True,
        hscy3_shear_responsivity_correction=True,
        hscy3_hsc_additive_shear_bias_correction=True,
        hscy3_hsc_y3_selection_bias_correction=True,
        # KiDS settings
        kids_scalar_shear_response_correction=True,
        kids_random_subtraction=True,
    )
    
    output_config = OutputConfig(
        catalogue_path='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/',
        save_path='./programmatic_plots/',
        verbose=True,
        save_precomputed=True,
        save_covariance=True,
        apply_blinding=False,  # For plotting examples
    )
    
    plot_config = PlotConfig(
        style='paper',
        transparent_background=False,
        log_scale=False,
        save_plots=True,
        filename_suffix='programmatic',
    )
    
    # Create plotter
    plotter = create_plotter_from_configs(
        computation_config, lens_config, source_config, output_config, plot_config, logger
    )
    
    logger.info("Created DataVectorPlotter successfully")
    
    # Example 1: Plot tomographic data vectors
    logger.info("Plotting tomographic data vectors...")
    try:
        plotter.plot_datavector_tomographic(
            statistic='deltasigma',
            log_scale=plot_config.log_scale,
            save_plot=plot_config.save_plots,
            filename_suffix=plot_config.filename_suffix
        )
        logger.info("✓ Tomographic data vector plot completed")
    except Exception as e:
        logger.warning(f"Plotting failed (expected without real data): {e}")
    
    # Example 2: Plot B-mode diagnostics
    logger.info("Plotting B-mode diagnostics...")
    try:
        pvalues = plotter.plot_bmodes_tomographic(
            statistic='deltasigma',
            save_plot=plot_config.save_plots,
            filename_suffix=plot_config.filename_suffix
        )
        
        # Save p-values to LaTeX table
        if pvalues and plot_config.save_plots:
            table_path = plotter.plot_dir / f"pvalues_bmodes_{plot_config.filename_suffix}.tex"
            save_p_values_table(
                pvalues, lens_config, source_config, table_path,
                statistic='deltasigma',
                caption="B-mode p-values for $\\Delta\\Sigma$ measurements (programmatic example)",
                precision=3
            )
            logger.info(f"✓ B-mode diagnostics completed, table saved to {table_path}")
        
    except Exception as e:
        logger.warning(f"B-mode plotting failed (expected without real data): {e}")
    
    # Show configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Galaxy type: {lens_config.galaxy_type}")
    logger.info(f"  Z bins: {lens_config.z_bins}")
    logger.info(f"  Source surveys: {source_config.surveys}")
    logger.info(f"  Statistics: {computation_config.statistics}")
    logger.info(f"  Tomography: {computation_config.tomography}")
    logger.info(f"  Output path: {output_config.save_path}")


def show_available_commands():
    """Show available CLI commands for plotting."""
    print("=== Available CLI Commands ===\n")
    
    commands = {
        "compute": {
            "deltasigma": "Compute Delta Sigma lensing statistic",
            "gammat": "Compute Gamma_t lensing statistic"
        },
        "plot": {
            "datavector": "Plot data vectors (tomographic or non-tomographic)",
            "bmodes": "Plot B-mode diagnostics for systematics testing", 
            "randoms": "Plot random lens tests for systematics testing",
            "compare-cosmologies": "Compare results from different cosmological models"
        },
        "utility": {
            "show-defaults": "Show default configuration values",
            "list-surveys": "List available source surveys",
            "convert-config": "Convert old INI config to new format"
        }
    }
    
    for group, cmds in commands.items():
        print(f"{group.upper()} COMMANDS:")
        for cmd, desc in cmds.items():
            print(f"  {cmd:<20} {desc}")
        print()
    
    print("For detailed help on any command:")
    print("  python -m desi_lensing.cli.main COMMAND --help")
    print("  python -m desi_lensing.cli.main plot SUBCOMMAND --help")


def main():
    """Main example function."""
    print("DESI Lensing Plotting Examples")
    print("=" * 50)
    print()
    
    show_available_commands()
    print()
    
    demonstrate_cli_usage()
    print()
    
    print("=== Running Examples ===\n")
    
    try:
        # Show CLI defaults
        print("1. Testing CLI defaults command...")
        run_cli_example()
        print()
        
        # Run programmatic example
        print("2. Running programmatic example...")
        programmatic_example()
        print("✓ Programmatic example completed")
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 