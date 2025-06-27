#!/usr/bin/env python3
"""
Example demonstrating the usage of AnalysisConfig for survey-dependent scale cuts and allowed bin combinations.

This script shows how to:
1. Configure survey-specific scale cuts
2. Set allowed source-lens bin combinations
3. Use AnalysisConfig with the plotting routines
4. Update analysis settings programmatically
"""

import numpy as np
from pathlib import Path

from desi_lensing.config import (
    ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, 
    OutputConfig, PlotConfig, AnalysisConfig
)
from desi_lensing.analysis.plotting import create_plotter_from_configs


def create_example_analysis_config():
    """Create an example AnalysisConfig with survey-specific settings."""
    
    # Create AnalysisConfig with default settings
    analysis_config = AnalysisConfig()
    
    # Customize scale cuts for different surveys
    # DES: Use larger scales due to better depth
    analysis_config.set_scale_cuts(
        survey="DES", 
        statistic="deltasigma",
        min_deg=0.002,  # Smaller minimum scale
        max_deg=3.0,    # Larger maximum scale
        rp_pivot=1.5
    )
    
    # SDSS: More conservative scale cuts due to shallower depth
    analysis_config.set_scale_cuts(
        survey="SDSS",
        statistic="deltasigma", 
        min_deg=0.01,   # Larger minimum scale
        max_deg=1.5,    # Smaller maximum scale
        rp_pivot=0.8
    )
    
    # HSC: High-quality survey, use extended range
    analysis_config.set_scale_cuts(
        survey="HSCY3",
        statistic="deltasigma",
        min_deg=0.001,  # Very small minimum scale
        max_deg=4.0,    # Very large maximum scale  
        rp_pivot=2.0
    )
    
    # Different scale cuts for gamma_t
    for survey in ["DES", "KiDS", "HSCY1", "HSCY3"]:
        cuts = analysis_config.get_scale_cuts(survey, "deltasigma")
        analysis_config.set_scale_cuts(
            survey=survey,
            statistic="gammat",
            min_deg=cuts["min_deg"] * 2,  # More conservative for gamma_t
            max_deg=cuts["max_deg"] * 0.8,
            rp_pivot=cuts["rp_pivot"]
        )
    
    # Update allowed bins - make BGS_BRIGHT more conservative for KiDS
    analysis_config.update_allowed_bins(
        galaxy_type="BGS_BRIGHT",
        source_survey="KiDS", 
        lens_bin=0,
        allowed_bins=[4],  # Only highest-z source bin
        conservative=True
    )
    
    # Make LRG analysis more permissive for DES
    analysis_config.update_allowed_bins(
        galaxy_type="LRG",
        source_survey="DES",
        lens_bin=0,
        allowed_bins=[1, 2, 3],  # Allow more source bins
        conservative=False
    )
    
    return analysis_config


def demonstrate_survey_dependent_settings():
    """Demonstrate how to access survey-dependent settings."""
    
    analysis_config = create_example_analysis_config()
    
    print("=== Survey-Dependent Scale Cuts ===")
    surveys = ["DES", "KiDS", "SDSS", "HSCY3"]
    statistics = ["deltasigma", "gammat"]
    
    for statistic in statistics:
        print(f"\n{statistic.upper()} Scale Cuts:")
        print(f"{'Survey':<10} {'min_deg':<12} {'max_deg':<12} {'rp_pivot':<10}")
        print("-" * 50)
        
        for survey in surveys:
            cuts = analysis_config.get_scale_cuts(survey, statistic)
            print(f"{survey:<10} {cuts['min_deg']:<12.6f} {cuts['max_deg']:<12.2f} {cuts['rp_pivot']:<10.1f}")
    
    print("\n=== Allowed Source-Lens Bin Combinations ===")
    galaxy_types = ["BGS_BRIGHT", "LRG"]
    
    for galaxy_type in galaxy_types:
        print(f"\n{galaxy_type}:")
        print(f"{'Survey':<10} {'Lens Bin 0':<15} {'Lens Bin 1':<15} {'Lens Bin 2':<15}")
        print("-" * 70)
        
        for survey in surveys:
            bins_str = []
            for lens_bin in range(3):
                allowed = analysis_config.get_allowed_source_bins(
                    galaxy_type, survey, lens_bin, conservative_cut=True
                )
                bins_str.append(str(allowed) if allowed else "[]")
            
            print(f"{survey:<10} {bins_str[0]:<15} {bins_str[1]:<15} {bins_str[2]:<15}")


def example_plotting_with_analysis_config():
    """Example of using AnalysisConfig with plotting routines."""
    
    # Create configuration objects
    computation_config = ComputationConfig(
        statistics=['deltasigma'],
        tomography=True,
        n_rp_bins=15
    )
    
    lens_config = LensGalaxyConfig(
        galaxy_type='BGS_BRIGHT',
        z_bins=[0.1, 0.2, 0.3, 0.4]
    )
    
    source_config = SourceSurveyConfig(
        surveys=['DES', 'KiDS', 'HSCY3']
    )
    
    output_config = OutputConfig(
        save_path='./analysis_example_output/'
    )
    
    plot_config = PlotConfig(
        save_plots=True,
        style="paper"
    )
    
    # Create custom analysis config
    analysis_config = create_example_analysis_config()
    
    # Create plotter with all configurations
    plotter = create_plotter_from_configs(
        computation_config=computation_config,
        lens_config=lens_config,
        source_config=source_config,
        output_config=output_config,
        plot_config=plot_config,
        analysis_config=analysis_config
    )
    
    print("\n=== Plotter Created Successfully ===")
    print(f"Analysis config supports surveys: {analysis_config.get_supported_surveys()}")
    print(f"Scale categories available: {analysis_config.get_scale_categories()}")
    
    # Example of plotting with survey-specific scale cuts
    # (This would work if you have actual data)
    # plotter.plot_datavector_tomographic(
    #     statistic="deltasigma",
    #     plot_scale_cuts=True,  # Uses survey-specific cuts from AnalysisConfig
    #     save_plot=True
    # )
    
    # Example of overriding scale cuts for all surveys
    custom_cuts = {"min_deg": 0.005, "max_deg": 2.0, "rp_pivot": 1.0}
    # plotter.plot_datavector_tomographic(
    #     statistic="deltasigma", 
    #     scale_cuts_override=custom_cuts,  # Override for all surveys
    #     save_plot=True
    # )


def demonstrate_programmatic_updates():
    """Show how to update analysis settings programmatically."""
    
    analysis_config = AnalysisConfig()
    
    print("\n=== Programmatic Configuration Updates ===")
    
    # Scenario: Need to be more conservative for a publication
    print("Making all surveys more conservative...")
    
    for survey in analysis_config.get_supported_surveys():
        # Get current cuts
        current_cuts = analysis_config.get_scale_cuts(survey, "deltasigma")
        
        # Make more conservative (smaller range)
        new_cuts = {
            "min_deg": current_cuts["min_deg"] * 2,      # Larger minimum scale
            "max_deg": current_cuts["max_deg"] * 0.7,    # Smaller maximum scale
            "rp_pivot": current_cuts["rp_pivot"]
        }
        
        analysis_config.set_scale_cuts(
            survey, "deltasigma", 
            new_cuts["min_deg"], new_cuts["max_deg"], new_cuts["rp_pivot"]
        )
        
        print(f"  {survey}: {current_cuts['min_deg']:.4f}-{current_cuts['max_deg']:.2f} deg "
              f"-> {new_cuts['min_deg']:.4f}-{new_cuts['max_deg']:.2f} deg")
    
    # Scenario: Disable certain source-lens combinations
    print("\nDisabling high-redshift LRG - low-redshift source combinations...")
    
    for survey in ["DES", "KiDS", "HSCY1", "HSCY3"]:
        # Make LRG bin 2 (highest-z) use only highest-z source bins
        current_bins = analysis_config.get_allowed_source_bins("LRG", survey, 2)
        if current_bins:
            # Keep only the highest available source bin
            conservative_bins = [max(current_bins)] if current_bins else []
            analysis_config.update_allowed_bins(
                "LRG", survey, 2, conservative_bins, conservative=True
            )
            print(f"  LRG bin 2 + {survey}: {current_bins} -> {conservative_bins}")


def validate_configuration():
    """Demonstrate configuration validation."""
    
    print("\n=== Configuration Validation ===")
    
    # Create a config with some intentional errors
    analysis_config = AnalysisConfig()
    
    # This should work fine
    errors = analysis_config.validate()
    print(f"Default config validation: {len(errors)} errors")
    
    # Create invalid scale cuts
    analysis_config.set_scale_cuts("DES", "deltasigma", -0.1, 2.0, 1.0)  # Negative min_deg
    analysis_config.set_scale_cuts("KiDS", "deltasigma", 3.0, 2.0, 1.0)   # min_deg > max_deg
    
    errors = analysis_config.validate()
    print(f"After adding errors: {len(errors)} errors")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    print("DESI Lensing AnalysisConfig Example")
    print("=" * 40)
    
    # Demonstrate basic usage
    demonstrate_survey_dependent_settings()
    
    # Show plotting integration
    example_plotting_with_analysis_config()
    
    # Show programmatic updates
    demonstrate_programmatic_updates()
    
    # Show validation
    validate_configuration()
    
    print("\n=== Example Complete ===")
    print("AnalysisConfig provides a centralized way to manage:")
    print("- Survey-specific scale cuts")
    print("- Allowed source-lens bin combinations") 
    print("- Analysis-specific choices")
    print("- Easy programmatic updates")
    print("- Built-in validation") 