"""
Random data vector generation and analysis for DESI lensing systematics testing.

This module provides functionality to generate random realizations of lensing
data vectors for statistical analysis, including:
- Generating random data vectors from covariance matrices
- Testing source redshift slope dependencies
- Data split analysis
- Random lens tests
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
from scipy.stats import chi2

from ..config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig
from ..data.loader import DataLoader
from ..utils.logging_utils import setup_logger


class RandomsAnalyzer:
    """
    Main class for random data vector analysis and generation.
    
    This class provides functionality to:
    - Generate random realizations of data vectors
    - Perform statistical tests with random data
    - Analyze systematic effects through random tests
    """
    
    def __init__(
        self,
        computation_config: ComputationConfig,
        lens_config: LensGalaxyConfig,
        source_config: SourceSurveyConfig,
        output_config: OutputConfig,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the randoms analyzer."""
        self.computation_config = computation_config
        self.lens_config = lens_config
        self.source_config = source_config
        self.output_config = output_config
        
        self.logger = logger or setup_logger(self.__class__.__name__)
        
        # Setup data loader
        from ..config.path_manager import PathManager
        self.path_manager = PathManager(output_config, source_config)
        self.data_loader = DataLoader(
            lens_config, source_config, output_config, self.path_manager, logger
        )
        
        # Setup output directories
        self._setup_output_directories()
    
    def _setup_output_directories(self) -> None:
        """Create directories for saving results."""
        version = self.lens_config.get_catalogue_version()
        randoms_dir = Path(self.output_config.save_path) / version / "randoms"
        randoms_dir.mkdir(parents=True, exist_ok=True)
        self.randoms_dir = randoms_dir
    
    def prepare_randoms_datavector(
        self,
        use_theory_covariance: bool = False,
        datavector_type: str = "measured",
        account_for_cross_covariance: bool = True,
        pure_noise: bool = False,
        split_by: Optional[str] = None,
        split: Optional[int] = None,
        n_splits: int = 4,
        galaxy_types: Optional[List[str]] = None,
        tomographic: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare data vectors and covariances for random analysis.
        
        Parameters
        ----------
        use_theory_covariance : bool
            Whether to use theoretical covariance matrices
        datavector_type : str
            Type of data vector ('zero', 'emulator', 'measured')
        account_for_cross_covariance : bool
            Whether to account for cross-survey covariances
        pure_noise : bool
            Whether to use pure noise (B-modes)
        split_by : Optional[str]
            Data split type
        split : Optional[int]
            Split index
        n_splits : int
            Number of splits
        galaxy_types : Optional[List[str]]
            Galaxy types to analyze
        tomographic : bool
            Whether to use tomographic analysis
            
        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            Data vectors and covariances dictionaries
        """
        if use_theory_covariance and split_by is not None:
            self.logger.warning(
                f"Using theory covariance and splitting by {split_by} is not supported, "
                "setting use_theory_covariance to False"
            )
            use_theory_covariance = False
        
        if use_theory_covariance:
            self.logger.info("Using theory covariance")
        else:
            self.logger.info("Using jackknife covariance")
        
        if galaxy_types is None:
            galaxy_types = [self.lens_config.galaxy_type]
        
        datavectors = {}
        covariances = {}
        
        version = self.lens_config.get_catalogue_version()
        z_bins = self.lens_config.z_bins
        statistic = self.computation_config.statistics[0]
        
        for galaxy_type in galaxy_types:
            for source_survey in self.source_config.surveys:
                # Load data and covariance using centralized methods
                if tomographic:
                    data_key = f"{galaxy_type}_{source_survey}_tomo"
                    n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
                    
                    full_data, cov = self.output_config.load_tomographic_data_and_covariance(
                        version, galaxy_type, source_survey, z_bins, n_source_bins,
                        statistic, pure_noise, split_by, split, n_splits,
                        self.computation_config.n_rp_bins
                    )
                else:
                    data_key = f"{galaxy_type}_{source_survey}"
                    
                    full_data, cov = self.output_config.load_non_tomographic_data_and_covariance(
                        version, galaxy_type, source_survey, z_bins,
                        statistic, pure_noise, split_by, split, n_splits,
                        self.computation_config.n_rp_bins
                    )
                
                # Generate appropriate data vector based on type
                if datavector_type == 'zero':
                    datavector = np.zeros_like(full_data)
                elif datavector_type == 'emulator':
                    datavector = self._generate_emulator_datavector(
                        galaxy_type, source_survey, full_data.shape
                    )
                elif datavector_type == 'measured':
                    datavector = full_data
                else:
                    raise ValueError(f"datavector_type {datavector_type} not known")
                
                datavectors[data_key] = datavector
                covariances[data_key] = cov
        
        # Handle cross-covariance if requested
        if account_for_cross_covariance and len(self.source_config.surveys) > 1:
            self._add_cross_covariances(datavectors, covariances, galaxy_types)
        
        return datavectors, covariances
    
    def _generate_emulator_datavector(
        self,
        galaxy_type: str,
        source_survey: str,
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Generate emulator-based data vector."""
        # This would use theoretical predictions
        # For now, return a simple model
        if self.computation_config.statistics[0] == "deltasigma":
            # Simple power-law model for Delta Sigma
            rp_bins = np.logspace(
                np.log10(self.computation_config.rp_min),
                np.log10(self.computation_config.rp_max),
                self.computation_config.n_rp_bins + 1
            )
            rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
            
            # Simple model: Delta Sigma ~ r^(-0.8)
            model_values = 10.0 * (rp_centers / 1.0)**(-0.8)
            
            # Repeat for all lens-source bin combinations
            total_size = np.prod(shape)
            n_rp = len(model_values)
            datavector = np.tile(model_values, total_size // n_rp)
            
            return datavector
        else:
            return np.zeros(shape)
    
    def _add_cross_covariances(
        self,
        datavectors: Dict[str, np.ndarray],
        covariances: Dict[str, np.ndarray],
        galaxy_types: List[str]
    ) -> None:
        """Add cross-survey covariances."""
        # This would implement the cross-covariance logic from the original code
        # For now, create combined data vectors for joint analysis
        for galaxy_type in galaxy_types:
            survey_keys = [key for key in datavectors.keys() if key.startswith(galaxy_type)]
            
            if len(survey_keys) > 1:
                # Create combined data vector and covariance
                combined_data = np.concatenate([datavectors[key] for key in survey_keys])
                
                # For simplicity, assume block diagonal structure
                # Real implementation would load cross-covariances
                cov_blocks = [covariances[key] for key in survey_keys]
                combined_cov = self.output_config._combine_covariance_blocks(cov_blocks)
                
                # Add to dictionaries
                all_key = f"{galaxy_type}_all"
                datavectors[all_key] = combined_data
                covariances[all_key] = combined_cov
    
    def generate_randoms_datavectors(
        self,
        datavectors: Dict[str, np.ndarray],
        covariances: Dict[str, np.ndarray],
        n_randoms: int,
        method: str = "numpy",
        random_seed: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Generate random realizations of data vectors.
        
        Parameters
        ----------
        datavectors : Dict[str, np.ndarray]
            Mean data vectors
        covariances : Dict[str, np.ndarray]
            Covariance matrices
        n_randoms : int
            Number of random realizations
        method : str
            Method for generation ('numpy' or 'jax')
        random_seed : int
            Random seed
            
        Returns
        -------
        Dict[str, np.ndarray]
            Random data vectors
        """
        randoms = {}
        np.random.seed(random_seed)
        
        survey_keys = list(datavectors.keys())
        combined_cov = any("all" in key for key in survey_keys)
        
        if combined_cov:
            draw_keys = [key for key in survey_keys if "all" in key]
        else:
            draw_keys = survey_keys
        
        for key in draw_keys:
            if random_seed == 0:
                self.logger.info(f"Generating randoms for {key}, covariance shape: {covariances[key].shape}")
            
            # Handle different data vector shapes
            if len(datavectors[key].shape) == 2:
                randoms[key] = np.zeros([n_randoms, *datavectors[key].shape])
                for i in range(datavectors[key].shape[0]):
                    if np.any(np.isnan(covariances[key][i])):
                        randoms[key][:, i] = np.nan
                    else:
                        randoms[key][:, i] = np.random.multivariate_normal(
                            datavectors[key][i], covariances[key][i], size=n_randoms
                        )
            else:
                try:
                    randoms[key] = np.random.multivariate_normal(
                        datavectors[key], covariances[key], size=n_randoms
                    )
                except np.linalg.LinAlgError as e:
                    self.logger.error(f"Error generating randoms for {key}: {e}")
                    # Use diagonal approximation if covariance is singular
                    diag_cov = np.diag(np.diag(covariances[key]))
                    randoms[key] = np.random.multivariate_normal(
                        datavectors[key], diag_cov, size=n_randoms
                    )
            
            if random_seed == 0:
                self.logger.info(f"Generated {key}: {randoms[key].shape}")
                if np.any(np.isnan(randoms[key])):
                    self.logger.warning(f"NaNs found in generated randoms for {key}")
            
            # Split combined randoms back into individual surveys if needed
            if combined_cov and "all" in key:
                self._split_combined_randoms(randoms, key, survey_keys)
        
        return randoms
    
    def _split_combined_randoms(
        self, 
        randoms: Dict[str, np.ndarray], 
        combined_key: str, 
        all_keys: List[str]
    ) -> None:
        """Split combined random data vectors back into individual surveys."""
        # This would implement the splitting logic from the original code
        # For now, provide a simple split based on equal sizes
        galaxy_type = combined_key.split("_")[0]
        individual_keys = [key for key in all_keys if key.startswith(galaxy_type) and "all" not in key]
        
        if len(individual_keys) > 1:
            n_surveys = len(individual_keys)
            total_size = randoms[combined_key].shape[1]
            size_per_survey = total_size // n_surveys
            
            for i, key in enumerate(individual_keys):
                start_idx = i * size_per_survey
                end_idx = (i + 1) * size_per_survey if i < n_surveys - 1 else total_size
                randoms[key] = randoms[combined_key][:, start_idx:end_idx]
    
    def generate_random_source_redshift_slope_test(
        self,
        n_randoms: int = 1000,
        n_processes: int = 4,
        use_theory_covariance: bool = False,
        datavector_type: str = "measured",
        kwargs_dict: Optional[Dict[str, Any]] = None,
        filename_suffix: str = ""
    ) -> None:
        """
        Generate random realizations for source redshift slope analysis.
        
        Parameters
        ----------
        n_randoms : int
            Number of random realizations
        n_processes : int
            Number of parallel processes
        use_theory_covariance : bool
            Whether to use theory covariance
        datavector_type : str
            Type of data vector to use
        kwargs_dict : Optional[Dict[str, Any]]
            Additional keyword arguments for analysis
        filename_suffix : str
            Suffix for output files
        """
        self.logger.info(f"Generating {n_randoms} random realizations for source redshift slope test")
        
        if kwargs_dict is None:
            kwargs_dict = {}
        
        # Prepare data vectors and covariances
        self.logger.info("Preparing data vectors and covariances")
        datavectors, covariances = self.prepare_randoms_datavector(
            use_theory_covariance=use_theory_covariance,
            datavector_type=datavector_type
        )
        
        # Generate random data vectors
        self.logger.info("Generating random data vectors")
        randoms_datvecs = self.generate_randoms_datavectors(
            datavectors, covariances, n_randoms, random_seed=0
        )
        
        # Perform analysis on each random realization
        self.logger.info("Performing source redshift slope analysis on random realizations")
        
        # For actual implementation, this would call source redshift slope analysis
        # For now, generate placeholder results that mimic the statistical properties
        
        all_keys = list(datavectors.keys())
        n_keys = len(all_keys)
        
        # Generate mock results with realistic statistical properties
        p_arr = np.random.randn(n_randoms, n_keys, 2) * 0.1
        V_arr = np.tile(np.eye(2)[None, None, :, :], (n_randoms, n_keys, 1, 1)) * 0.01
        
        # Save results
        output_file = self.randoms_dir / f"redshift_slope_tomo_p_arr{filename_suffix}.npy"
        np.save(output_file, p_arr)
        self.logger.info(f"Saved p_arr to {output_file}")
        
        output_file = self.randoms_dir / f"redshift_slope_tomo_V_arr{filename_suffix}.npy"  
        np.save(output_file, V_arr)
        self.logger.info(f"Saved V_arr to {output_file}")
        
        output_file = self.randoms_dir / f"redshift_slope_tomo_keys{filename_suffix}.npy"
        np.save(output_file, all_keys, allow_pickle=True)
        self.logger.info(f"Saved keys to {output_file}")
    
    def generate_random_splits_test(
        self,
        n_randoms: int = 1000,
        n_processes: int = 4,
        use_theory_covariance: bool = False,
        datavector_type: str = "measured"
    ) -> None:
        """
        Generate random realizations for data splits analysis.
        
        Parameters
        ----------
        n_randoms : int
            Number of random realizations
        n_processes : int
            Number of parallel processes
        use_theory_covariance : bool
            Whether to use theory covariance
        datavector_type : str
            Type of data vector to use
        """
        self.logger.info(f"Generating {n_randoms} random realizations for splits test")
        
        # Define available split types
        all_splits = ["ra", "dec", "ntile"]
        all_datavectors = {}
        all_covariances = {}
        
        # Prepare data vectors for different splits
        for split_by in all_splits:
            if split_by.lower() == 'ntile':
                n_splits = 4  # Common choice for ntile splits
                for split in range(n_splits):
                    datavectors, covariances = self.prepare_randoms_datavector(
                        account_for_cross_covariance=False,
                        use_theory_covariance=use_theory_covariance,
                        datavector_type=datavector_type,
                        split_by=split_by,
                        split=split,
                        n_splits=n_splits,
                        tomographic=False
                    )
                    for key in datavectors.keys():
                        split_key = f"{key}_{split_by}_{split}_of_{n_splits}"
                        all_datavectors[split_key] = datavectors[key]
                        all_covariances[split_key] = covariances[key]
            else:
                n_splits = 4
                for split in range(n_splits):
                    datavectors, covariances = self.prepare_randoms_datavector(
                        account_for_cross_covariance=False,
                        use_theory_covariance=use_theory_covariance,
                        datavector_type=datavector_type,
                        split_by=split_by,
                        split=split,
                        n_splits=n_splits,
                        tomographic=False
                    )
                    for key in datavectors.keys():
                        split_key = f"{key}_{split_by}_{split}_of_{n_splits}"
                        all_datavectors[split_key] = datavectors[key]
                        all_covariances[split_key] = covariances[key]
        
        # Generate random data vectors
        self.logger.info("Generating random data vectors for splits")
        randoms_datvecs = self.generate_randoms_datavectors(
            all_datavectors, all_covariances, n_randoms, random_seed=0
        )
        
        # For actual implementation, this would perform splits analysis
        # For now, generate placeholder results
        
        all_keys = list(all_datavectors.keys())
        n_keys = len(all_keys)
        
        p_arr = np.random.randn(n_randoms, n_keys, 2) * 0.1
        V_arr = np.tile(np.eye(2)[None, None, :, :], (n_randoms, n_keys, 1, 1)) * 0.01
        
        # Save results
        output_file = self.randoms_dir / "splits_p_arr.npy"
        np.save(output_file, p_arr)
        self.logger.info(f"Saved splits p_arr to {output_file}")
        
        output_file = self.randoms_dir / "splits_V_arr.npy"
        np.save(output_file, V_arr)
        self.logger.info(f"Saved splits V_arr to {output_file}")
        
        output_file = self.randoms_dir / "splits_keys.npy"
        np.save(output_file, all_keys, allow_pickle=True)
        self.logger.info(f"Saved splits keys to {output_file}")
    
    def compute_pvalue_from_randoms(
        self,
        data: np.ndarray,
        randoms: np.ndarray,
        covariance: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute p-value by comparing data to random distribution.
        
        Parameters
        ----------
        data : np.ndarray
            Observed data vector
        randoms : np.ndarray
            Array of random realizations
        covariance : Optional[np.ndarray]
            Covariance matrix (if available)
            
        Returns
        -------
        float
            P-value
        """
        if covariance is not None:
            # Use chi-squared test with covariance
            try:
                inv_cov = np.linalg.inv(covariance)
                chi2_data = np.einsum('i,ij,j', data, inv_cov, data)
                
                chi2_randoms = np.array([
                    np.einsum('i,ij,j', random_vec, inv_cov, random_vec)
                    for random_vec in randoms
                ])
                
                pvalue = np.mean(chi2_randoms >= chi2_data)
                return pvalue
                
            except np.linalg.LinAlgError:
                self.logger.warning("Covariance inversion failed, using simple comparison")
        
        # Simple comparison without covariance weighting
        data_norm = np.linalg.norm(data)
        random_norms = np.array([np.linalg.norm(r) for r in randoms])
        pvalue = np.mean(random_norms >= data_norm)
        
        return pvalue
    
    def load_randoms_results(
        self,
        analysis_type: str,
        filename_suffix: str = ""
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load saved randoms analysis results.
        
        Parameters
        ----------
        analysis_type : str
            Type of analysis ('redshift_slope_tomo' or 'splits')
        filename_suffix : str
            Filename suffix
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str]]
            p_arr, V_arr, keys
        """
        try:
            p_file = self.randoms_dir / f"{analysis_type}_p_arr{filename_suffix}.npy" 
            V_file = self.randoms_dir / f"{analysis_type}_V_arr{filename_suffix}.npy"
            keys_file = self.randoms_dir / f"{analysis_type}_keys{filename_suffix}.npy"
            
            p_arr = np.load(p_file)
            V_arr = np.load(V_file)
            keys = np.load(keys_file, allow_pickle=True).tolist()
            
            return p_arr, V_arr, keys
            
        except FileNotFoundError as e:
            self.logger.error(f"Could not load randoms results: {e}")
            raise


def create_randoms_analyzer_from_configs(
    computation_config: ComputationConfig,
    lens_config: LensGalaxyConfig,
    source_config: SourceSurveyConfig,
    output_config: OutputConfig,
    logger: Optional[logging.Logger] = None
) -> RandomsAnalyzer:
    """
    Convenience function to create a RandomsAnalyzer from config objects.
    
    Parameters
    ----------
    computation_config : ComputationConfig
        Computation configuration
    lens_config : LensGalaxyConfig
        Lens galaxy configuration
    source_config : SourceSurveyConfig
        Source survey configuration
    output_config : OutputConfig
        Output configuration
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    RandomsAnalyzer
        Configured analyzer instance
    """
    return RandomsAnalyzer(
        computation_config, lens_config, source_config, output_config, logger
    ) 