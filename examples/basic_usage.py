#!/usr/bin/env python3
"""
Example of using the DESI lensing pipeline programmatically.

This script demonstrates how to set up and run lensing computations
without using the command-line interface.
"""

import logging
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.cosmology import Planck18

from desi_lensing.config import (
    ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig,
    ConfigValidator
)
from desi_lensing.core.pipeline import LensingPipeline
from desi_lensing.utils.logging_utils import setup_logger

# Import required dsigma functions for the calculation example
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density


def main():
    """Example of programmatic pipeline usage."""
    
    # Setup logging
    logger = setup_logger('example', level=logging.INFO)
    logger.info("Starting DESI lensing example")
    
    # Create configuration objects
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
    )
    
    lens_config = LensGalaxyConfig(
        galaxy_type='BGS_BRIGHT',
        bgs_catalogue_version='v1.5',
        z_bins=[0.1, 0.2, 0.3, 0.4],
        which_randoms=[1, 2],
        magnitude_cuts=True,
        mstar_complete=False,
    )
    
    source_config = SourceSurveyConfig(
        surveys=['KiDS'],
        cut_catalogues_to_desi=True,
        # Enable some corrections for DES
        des_scalar_shear_response_correction=True,
        des_matrix_shear_response_correction=True,
        des_random_subtraction=True,
        # Enable some corrections for KiDS
        kids_scalar_shear_response_correction=True,
        kids_random_subtraction=True,
    )
    
    output_config = OutputConfig(
        catalogue_path='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/',
        save_path='/pscratch/sd/s/sven/lensing_measurements/',
        verbose=True,
        save_precomputed=True,
        save_covariance=True,
        apply_blinding=True,
        blinding_label='A',
    )
    
    # Validate configuration
    validator = ConfigValidator(
        computation_config, lens_config, source_config, output_config
    )
    
    if not validator.is_valid():
        logger.error("Configuration validation failed:")
        logger.error(validator.get_error_summary())
        return
    
    # Show any warnings
    warnings = validator.get_warnings()
    for warning in warnings:
        logger.warning(warning)
    
    # Create and run pipeline
    try:
        pipeline = LensingPipeline(
            computation_config, lens_config, source_config, output_config, logger
        )
        
        # Note: This would fail without actual data files
        # pipeline.run()
        
        logger.info("Pipeline setup completed successfully")
        logger.info("Configuration validation passed")
        logger.info("Ready to run with real data")
        
    except Exception as e:
        logger.error(f"Pipeline setup failed: {e}")
        logger.info("This is expected without actual DESI data files")


def example_configuration_only():
    """Example showing just configuration setup and validation."""
    
    print("=== DESI Lensing Configuration Example ===\n")
    
    # Create a more complex configuration
    comp_config = ComputationConfig(
        statistics=['deltasigma', 'gammat'],
        cosmology='wcdm',
        h0=70.0,
        w0=-0.9,
        wa=-0.1,
        tomography=True,
        n_rp_bins=20,
        n_theta_bins=20,
    )
    
    lens_config = LensGalaxyConfig(
        galaxy_type='LRG',
        z_bins=[0.4, 0.6, 0.8, 1.1],
        magnitude_cuts=True,
    )
    
    source_config = SourceSurveyConfig(
        surveys=['HSCY1', 'HSCY3'],
        # HSC Y1 settings
        hscy1_photo_z_dilution_correction=True,
        hscy1_scalar_shear_response_correction=True,
        hscy1_shear_responsivity_correction=True,
        hscy1_hsc_selection_bias_correction=True,
        # HSC Y3 settings  
        hscy3_scalar_shear_response_correction=True,
        hscy3_shear_responsivity_correction=True,
        hscy3_hsc_additive_shear_bias_correction=True,
        hscy3_hsc_y3_selection_bias_correction=True,
    )
    
    output_config = OutputConfig(
        save_path='./lrg_hsc_analysis/',
        apply_blinding=False,  # For testing
    )
    
    # Show configuration details
    print("Computation Configuration:")
    print(f"  Statistics: {comp_config.statistics}")
    print(f"  Cosmology: {comp_config.cosmology} (w0={comp_config.w0}, wa={comp_config.wa})")
    print(f"  Tomography: {comp_config.tomography}")
    print(f"  Binning: {comp_config.n_rp_bins} rp bins, {comp_config.n_theta_bins} theta bins")
    
    print(f"\nLens Configuration:")
    print(f"  Galaxy type: {lens_config.galaxy_type}")
    print(f"  Redshift bins: {lens_config.z_bins}")
    print(f"  Number of lens bins: {lens_config.get_n_lens_bins()}")
    
    print(f"\nSource Configuration:")
    print(f"  Surveys: {source_config.surveys}")
    for survey in source_config.surveys:
        n_tomo = source_config.get_n_tomographic_bins(survey)
        print(f"  {survey}: {n_tomo} tomographic bins")
    
    print(f"\nOutput Configuration:")
    print(f"  Save path: {output_config.save_path}")
    print(f"  Apply blinding: {output_config.apply_blinding}")
    
    # Validate and show results
    validator = ConfigValidator(comp_config, lens_config, source_config, output_config)
    
    print(f"\n=== Validation Results ===")
    if validator.is_valid():
        print("✓ Configuration is valid!")
    else:
        print("✗ Configuration has errors:")
        print(validator.get_error_summary())
    
    warnings = validator.get_warnings()
    if warnings:
        print(f"\nWarnings:")
        for warning in warnings:
            print(f"  ! {warning}")
    
    # Show example filename generation
    print(f"\n=== Example Output Files ===")
    for stat in comp_config.statistics:
        for i, (z_min, z_max) in enumerate(zip(lens_config.z_bins[:-1], lens_config.z_bins[1:])):
            filepath = output_config.get_filepath(
                statistic=stat,
                galaxy_type=lens_config.galaxy_type,
                version="v1.0",
                survey="example",
                z_min=z_min,
                z_max=z_max,
                source_bin=i if comp_config.tomography else None
            )
            print(f"  {filepath.name}")


def example_deltasigma_calculation():
    """
    Example demonstrating actual deltasigma calculation with synthetic data.
    
    This function creates synthetic lens and source catalogs and performs
    a complete deltasigma measurement workflow.
    """
    print("=== DeltaSigma Calculation Example ===\n")
    
    # Set up logging
    logger = setup_logger('deltasigma_example', level=logging.INFO)
    logger.info("Starting deltasigma calculation with synthetic data")
    
    # Create synthetic lens catalog (BGS-like galaxies)
    print("Creating synthetic lens catalog...")
    n_lenses = 5000
    np.random.seed(42)  # For reproducibility
    
    lens_data = {
        'RA': np.random.uniform(140, 170, n_lenses),
        'DEC': np.random.uniform(-5, 25, n_lenses), 
        'Z': np.random.uniform(0.15, 0.35, n_lenses),  # BGS-like redshift range
        'WEIGHT': np.ones(n_lenses)  # Simple unit weights
    }
    
    table_l = Table(lens_data)
    table_l = dsigma_table(table_l, 'lens', z='Z', ra='RA', dec='DEC', w_sys='WEIGHT')
    print(f"  Created {len(table_l)} lens galaxies")
    print(f"  Redshift range: {table_l['z'].min():.3f} - {table_l['z'].max():.3f}")
    
    # Create synthetic source catalog (DES-like)
    print("\nCreating synthetic source catalog...")
    n_sources = 50000
    
    source_data = {
        'RA': np.random.uniform(140, 170, n_sources),
        'DEC': np.random.uniform(-5, 25, n_sources),
        'e_1': np.random.normal(0, 0.3, n_sources),  # Realistic ellipticity dispersion
        'e_2': np.random.normal(0, 0.3, n_sources),
        'w': np.ones(n_sources),  # Unit weights
        'm': np.random.normal(-0.02, 0.01, n_sources),  # Multiplicative bias
        'z_bin': np.random.randint(0, 4, n_sources)  # 4 tomographic bins
    }
    
    table_s = Table(source_data)
    table_s = dsigma_table(table_s, 'source', ra='RA', dec='DEC', 
                          e_1='e_1', e_2='e_2', w='w', m='m', z_bin='z_bin', z=1.0)
    print(f"  Created {len(table_s)} source galaxies")
    print(f"  Tomographic bins: {np.unique(table_s['z_bin'])}")
    
    # Set up cosmology and bins
    cosmology = Planck18.clone(H0=100.0)
    rp_bins = np.logspace(np.log10(0.08), np.log10(80), 16) * u.Mpc  # Comoving bins
    
    print(f"\nUsing cosmology: {cosmology}")
    print(f"Radial bins: {len(rp_bins)-1} bins from {rp_bins[0]:.2f} to {rp_bins[-1]:.1f}")
    
    # Precompute the lensing signal
    print("\nPrecomputing lensing signal...")
    try:
        precompute(table_l, table_s, rp_bins, 
                  comoving=True, 
                  cosmology=cosmology, 
                  n_jobs=2,  # Use fewer jobs for example
                  lens_source_cut=None,
                  progress_bar=True)
        
        # Filter out lenses with no nearby sources
        n_pairs = np.sum(table_l['sum 1'], axis=1)
        good_lenses = n_pairs > 0
        table_l = table_l[good_lenses]
        
        print(f"  Lens galaxies with sources: {np.sum(good_lenses)}/{len(good_lenses)}")
        print(f"  Total lens-source pairs: {np.sum(n_pairs[good_lenses])}")
        
        if len(table_l) == 0:
            print("  No lens-source pairs found. This is expected with random synthetic data.")
            return None
            
    except Exception as e:
        logger.warning(f"Precompute failed: {e}")
        print("  This is expected with random synthetic data (no real clustering)")
        return None
    
    # Compute jackknife fields
    print("\nSetting up jackknife resampling...")
    try:
        n_jackknife = 20
        weights = np.sum(table_l['sum 1'], axis=1)
        centers = compute_jackknife_fields(table_l, n_jackknife, weights=weights)
        print(f"  Created {len(centers)} jackknife regions")
        
    except Exception as e:
        logger.warning(f"Jackknife setup failed: {e}")
        print("  Using simple error estimates instead")
        
    # Calculate deltasigma
    print("\nCalculating deltasigma...")
    try:
        # Stacking arguments
        stacking_kwargs = {
            'return_table': True,
            'random_subtraction': False,  # No randoms in this simple example
        }
        
        # Calculate the excess surface density
        result = excess_surface_density(table_l, **stacking_kwargs)
        
        # Try to calculate covariance with jackknife resampling
        stacking_kwargs['return_table'] = False
        try:
            covariance_matrix = jackknife_resampling(
                excess_surface_density, table_l, **stacking_kwargs
            )
            result['ds_err'] = np.sqrt(np.diag(covariance_matrix))
            print(f"  Calculated covariance matrix: {covariance_matrix.shape}")
        except:
            # Fallback to simple Poisson errors
            result['ds_err'] = np.abs(result['ds']) / np.sqrt(np.maximum(result['n_pairs'], 1))
            print("  Using Poisson error estimates")
        
        # Display results
        print(f"\nDeltaSigma Results:")
        print(f"  Available columns: {result.colnames}")
        print(f"  Number of radial bins: {len(result)}")
        print(f"  Radial range: {result['rp'].min():.2f} - {result['rp'].max():.1f} Mpc")
        print(f"  DeltaSigma range: {result['ds'].min():.2e} - {result['ds'].max():.2e} M☉/pc²")
        
        # Check if n_pairs column exists, otherwise use alternative
        if 'n_pairs' in result.colnames:
            pairs_col = 'n_pairs'
        elif 'N_pairs' in result.colnames:
            pairs_col = 'N_pairs' 
        else:
            # Find any column that might contain pair counts
            pairs_col = None
            for col in result.colnames:
                if 'pair' in col.lower():
                    pairs_col = col
                    break
        
        if pairs_col:
            print(f"  Total pairs: {result[pairs_col].sum()}")
            
            # Show a few example measurements
            print(f"\nSample measurements:")
            print(f"  {'rp [Mpc]':<10} {'ΔΣ [M☉/pc²]':<15} {'Error':<15} {pairs_col:<10}")
            print(f"  {'-'*10:<10} {'-'*15:<15} {'-'*15:<15} {'-'*10:<10}")
            for i in range(0, len(result), max(1, len(result)//5)):
                print(f"  {result['rp'][i]:<10.2f} {result['ds'][i]:<15.2e} "
                      f"{result['ds_err'][i]:<15.2e} {result[pairs_col][i]:<10}")
        else:
            print("  Note: Could not find pair count column")
            print(f"\nSample measurements:")
            print(f"  {'rp [Mpc]':<10} {'ΔΣ [M☉/pc²]':<15} {'Error':<15}")
            print(f"  {'-'*10:<10} {'-'*15:<15} {'-'*15:<15}")
            for i in range(0, len(result), max(1, len(result)//5)):
                print(f"  {result['rp'][i]:<10.2f} {result['ds'][i]:<15.2e} "
                      f"{result['ds_err'][i]:<15.2e}")
        
        return result
        
    except Exception as e:
        logger.error(f"DeltaSigma calculation failed: {e}")
        print("  This is expected with random synthetic data")
        return None


if __name__ == "__main__":
    # Run the configuration-only example first
    example_configuration_only()
    
    print("\n" + "="*60 + "\n")
    
    # Then try the full pipeline example
    main()
    
    print("\n" + "="*60 + "\n")
    
    # Finally run the actual deltasigma calculation example
    example_deltasigma_calculation() 