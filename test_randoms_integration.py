#!/usr/bin/env python
"""
Test script for randoms analysis integration with the refactored DESI lensing pipeline.
"""

import logging
import sys
from pathlib import Path

# Add the desi_lensing module to the path
sys.path.insert(0, str(Path(__file__).parent / "desi_lensing"))

from desi_lensing.config import (
    ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig
)
from desi_lensing.analysis.randoms import create_randoms_analyzer_from_configs
from desi_lensing.utils.logging_utils import setup_logger


def test_randoms_analyzer():
    """Test the RandomsAnalyzer functionality."""
    
    # Setup logging
    logger = setup_logger('test_randoms', level=logging.INFO)
    logger.info("Testing randoms analyzer integration")
    
    # Create test configurations
    computation_config = ComputationConfig(
        statistics=['deltasigma'],
        cosmology='planck18',
        n_rp_bins=10,
        rp_min=0.1,
        rp_max=30.0,
        tomography=True
    )
    
    lens_config = LensGalaxyConfig(
        galaxy_type='BGS_BRIGHT',
        z_bins=[0.1, 0.2, 0.3, 0.4]
    )
    
    source_config = SourceSurveyConfig(
        surveys=['DES', 'KiDS']
    )
    
    output_config = OutputConfig(
        save_path='./test_output'
    )
    
    # Create randoms analyzer
    try:
        randoms_analyzer = create_randoms_analyzer_from_configs(
            computation_config, lens_config, source_config, output_config, logger
        )
        logger.info("Successfully created RandomsAnalyzer")
        
        # Test data preparation
        logger.info("Testing data vector preparation...")
        datavectors, covariances = randoms_analyzer.prepare_randoms_datavector(
            datavector_type="zero",
            tomographic=True
        )
        
        logger.info(f"Prepared {len(datavectors)} data vectors")
        for key, dv in datavectors.items():
            logger.info(f"  {key}: shape {dv.shape}")
        
        # Test random generation (small number for testing)
        logger.info("Testing random generation...")
        randoms = randoms_analyzer.generate_randoms_datavectors(
            datavectors, covariances, n_randoms=10, random_seed=42
        )
        
        logger.info(f"Generated randoms for {len(randoms)} keys")
        for key, rand in randoms.items():
            logger.info(f"  {key}: shape {rand.shape}")
        
        # Test p-value computation
        if len(randoms) > 0:
            key = list(randoms.keys())[0]
            data = datavectors[key]
            random_data = randoms[key]
            cov = covariances[key]
            
            pvalue = randoms_analyzer.compute_pvalue_from_randoms(
                data, random_data, cov
            )
            logger.info(f"Computed p-value: {pvalue:.3f}")
        
        logger.info("All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_integration():
    """Test CLI integration."""
    logger = setup_logger('test_cli', level=logging.INFO)
    logger.info("Testing CLI integration")
    
    try:
        # Import CLI module
        from desi_lensing.cli.main import cli
        
        # Test help command
        from click.testing import CliRunner
        runner = CliRunner()
        
        # Test main help
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'randoms' in result.output
        logger.info("Main CLI help includes randoms command")
        
        # Test randoms help
        result = runner.invoke(cli, ['randoms', '--help'])
        assert result.exit_code == 0
        assert 'source-redshift-slope' in result.output
        assert 'splits' in result.output
        logger.info("Randoms CLI help shows subcommands")
        
        logger.info("CLI integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"CLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing DESI Lensing Randoms Integration")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing RandomsAnalyzer functionality...")
    success &= test_randoms_analyzer()
    
    print("\n2. Testing CLI integration...")
    success &= test_cli_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("All tests PASSED! ✅")
        sys.exit(0)
    else:
        print("Some tests FAILED! ❌")
        sys.exit(1) 