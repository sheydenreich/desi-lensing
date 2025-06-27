#!/usr/bin/env python3
"""
Example script demonstrating the plot magnitudes functionality.

This script shows how to use the new plot_magnitudes method in the
DESI lensing pipeline plotting framework.
"""

import sys
import logging
from pathlib import Path

# Add the desi_lensing package to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from desi_lensing.config import (
    ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, 
    OutputConfig, PlotConfig, AnalysisConfig
)
from desi_lensing.analysis.plotting import create_plotter_from_configs
from desi_lensing.analysis.plotting_utils import setup_matplotlib_style
from desi_lensing.utils.logging_utils import setup_logger


def main():
    """Main function to demonstrate magnitude plotting."""
    
    # Setup logging
    logger = setup_logger('plot_magnitudes_example', level=logging.INFO)
    logger.info("Starting magnitude plotting example")
    
    # Create configurations
    comp_config = ComputationConfig()
    
    # BGS configuration
    bgs_config = LensGalaxyConfig(
        galaxy_type="BGS_BRIGHT",
        release="iron",  # or "loa" for future release
        bgs_catalogue_version="v1.5",
        z_bins=[0.1, 0.2, 0.3, 0.4],
        magnitude_cuts=True
    )
    
    source_config = SourceSurveyConfig(
        surveys=["DES"],  # Just use one survey for lens loading
        cut_catalogues_to_desi=True
    )
    
    output_config = OutputConfig(
        catalogue_path="/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/",
        save_path="/tmp/lensing_plots/",
        verbose=True
    )
    
    plot_config = PlotConfig(
        style="paper",
        save_plots=True,
        transparent_background=False
    )
    
    analysis_config = AnalysisConfig()
    
    # Setup matplotlib style
    setup_matplotlib_style(plot_config.style)
    
    # Create plotter
    logger.info("Creating plotter for BGS_BRIGHT")
    plotter = create_plotter_from_configs(
        comp_config, bgs_config, source_config, output_config, 
        plot_config, analysis_config, logger
    )
    
    # Example 1: Default magnitude cuts
    logger.info("Plotting with default magnitude cuts")
    try:
        plotter.plot_magnitudes(
            save_plot=True,
            filename_suffix="default"
        )
    except Exception as e:
        logger.error(f"Failed to plot with default cuts: {e}")
    
    # Example 2: Custom magnitude cuts
    logger.info("Plotting with custom magnitude cuts")
    try:
        plotter.plot_magnitudes(
            magnitude_cuts=[-19.0, -20.0, -21.5],
            save_plot=True,
            filename_suffix="custom_cuts"
        )
    except Exception as e:
        logger.error(f"Failed to plot with custom cuts: {e}")
    
    # Example 3: With extinction correction
    logger.info("Plotting with extinction correction")
    try:
        plotter.plot_magnitudes(
            apply_extinction_correction=True,
            save_plot=True,
            filename_suffix="extinction_corrected"
        )
    except Exception as e:
        logger.error(f"Failed to plot with extinction correction: {e}")
    
    # Example 4: With KP3 cut
    logger.info("Plotting with KP3 cut")
    try:
        plotter.plot_magnitudes(
            add_kp3_cut=True,
            save_plot=True,
            filename_suffix="with_kp3_cut"
        )
    except Exception as e:
        logger.error(f"Failed to plot with KP3 cut: {e}")
    
    # Example 5: LRG configuration (no magnitude cuts)
    logger.info("Creating plotter for LRG")
    lrg_config = LensGalaxyConfig(
        galaxy_type="LRG",
        release="iron",  # Could also use "loa" for future release
        lrg_catalogue_version="v1.5",
        z_bins=[0.4, 0.6, 0.8, 1.1],
        magnitude_cuts=False  # LRGs typically don't have magnitude cuts
    )
    
    lrg_plotter = create_plotter_from_configs(
        comp_config, lrg_config, source_config, output_config, 
        plot_config, analysis_config, logger
    )
    
    try:
        lrg_plotter.plot_magnitudes(
            save_plot=True,
            filename_suffix="lrg_no_cuts"
        )
    except Exception as e:
        logger.error(f"Failed to plot LRG magnitudes: {e}")
    
    logger.info("Magnitude plotting examples completed")


if __name__ == "__main__":
    main() 