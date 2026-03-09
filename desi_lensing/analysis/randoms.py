"""
Random data vector generation and analysis for DESI lensing systematics testing.

This module provides functionality to generate random realizations of lensing
data vectors for statistical analysis, including:
- Generating random data vectors from covariance matrices
- Testing source redshift slope dependencies
- Data split analysis
- Random lens tests
- Sigma_sys systematic uncertainty estimation
"""

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from multiprocessing import Pool
import traceback

from tqdm import tqdm
import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

from ..config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig, AnalysisConfig
from ..data.loader import DataLoader
from ..utils.logging_utils import setup_logger


# ============================================================================
# Sigma_sys calculation functions
# ============================================================================

def _sigma_sys_minimizing_fnc(sigma_sys: float, amplitudes: np.ndarray, 
                               statistical_errors: np.ndarray, target_value: float) -> float:
    """Objective function for sigma_sys optimization."""
    errors = np.sqrt(statistical_errors**2 + sigma_sys**2)
    mean_amplitude = np.average(amplitudes, weights=1/errors**2)
    chisq = np.sum((amplitudes - mean_amplitude)**2 / errors**2)
    return (chisq - target_value)**2


def calculate_sigma_sys(amplitudes: np.ndarray, errors: np.ndarray, 
                        method: str = "chisqpdf") -> Tuple[float, List[float], Optional[Any]]:
    """
    Calculate systematic uncertainty (sigma_sys) from amplitude scatter.
    
    This function estimates the additional systematic uncertainty needed to
    bring amplitude measurements into statistical consistency.
    
    Parameters
    ----------
    amplitudes : np.ndarray
        Measured lensing amplitudes
    errors : np.ndarray
        Statistical errors on amplitudes
    method : str
        Calculation method: "chisqpdf", "bayesian", or "bayesian_brute"
    
    Returns
    -------
    Tuple[float, List[float], Optional[Any]]
        - reduced_chisq: Reduced chi-squared value
        - sigma_sys_bounds: [2sigma_low, 1sigma_low, median, 1sigma_high, 2sigma_high]
        - samples: MCMC samples if bayesian method, else None
    """
    if method == "chisqpdf":
        return _calculate_sigma_sys_chisqpdf(amplitudes, errors)
    elif method == "bayesian":
        return _calculate_sigma_sys_bayesian(amplitudes, errors, brute=False)
    elif method == "bayesian_brute":
        return _calculate_sigma_sys_bayesian(amplitudes, errors, brute=True)
    else:
        raise ValueError(f"Method {method} not recognized. Use: chisqpdf, bayesian, bayesian_brute")


def _calculate_sigma_sys_chisqpdf(amplitudes: np.ndarray, errors: np.ndarray,
                                   confidence_regions: List[float] = [0.68, 0.95]) -> Tuple[float, List[float], None]:
    """
    Calculate sigma_sys using chi-squared PDF method.
    
    Parameters
    ----------
    amplitudes : np.ndarray
        Measured amplitudes
    errors : np.ndarray
        Statistical errors
    confidence_regions : List[float]
        Confidence levels for bounds (default: 1-sigma and 2-sigma)
    
    Returns
    -------
    Tuple[float, List[float], None]
        reduced_chisq, sigma_sys bounds [2σ_low, 1σ_low, median, 1σ_high, 2σ_high], None
    """
    assert len(amplitudes) == len(errors), "Amplitudes and errors must have the same length"
    n_measurements = len(amplitudes)
    mean_amplitude = np.average(amplitudes, weights=1/errors**2)
    chisq = np.sum((amplitudes - mean_amplitude)**2 / errors**2)
    reduced_chisq = chisq / (n_measurements - 1)
    
    target_chisqs_upper = [chi2.ppf(0.5 + confidence_region/2, n_measurements - 1) 
                          for confidence_region in confidence_regions]
    target_chisqs_lower = [chi2.ppf(0.5 - confidence_region/2, n_measurements - 1) 
                          for confidence_region in confidence_regions[::-1]]
    target_chisqs = np.array(target_chisqs_lower + [n_measurements - 1] + target_chisqs_upper)
    target_values = np.zeros_like(target_chisqs)
    
    for i, target_chisq in enumerate(target_chisqs):
        if (target_chisq >= chisq) or np.isnan(target_chisq):
            target_values[i] = np.nan
        else:
            minval = minimize_scalar(
                _sigma_sys_minimizing_fnc,
                args=(amplitudes, errors, target_chisq),
                bounds=[0, 10],
                tol=1e-8
            )
            target_values[i] = minval.x
            
            assert _sigma_sys_minimizing_fnc(target_values[i], amplitudes, errors, target_chisq) < 1e-4, \
                "Minimization failed!"
    
    return reduced_chisq, target_values[::-1].tolist(), None


def _calculate_sigma_sys_bayesian(amplitudes: np.ndarray, errors: np.ndarray,
                                   brute: bool = False, verbose: bool = False) -> Tuple[float, List[float], List[Any]]:
    """
    Calculate sigma_sys using Bayesian method with nautilus sampler.
    
    Parameters
    ----------
    amplitudes : np.ndarray
        Measured amplitudes
    errors : np.ndarray
        Statistical errors
    brute : bool
        Whether to use brute force method for finding confidence intervals
    verbose : bool
        Whether to print verbose output
    
    Returns
    -------
    Tuple[float, List[float], List[Any]]
        reduced_chisq, sigma_sys bounds, [points, log_w, log_l] samples
    """
    try:
        import nautilus
        from getdist import MCSamples
    except ImportError:
        logging.warning("nautilus or getdist not available, falling back to chisqpdf method")
        return _calculate_sigma_sys_chisqpdf(amplitudes, errors)
    
    if verbose:
        print("Calculating sigma_sys using Bayesian method")
    
    prior = nautilus.Prior()
    parameters = ['Alens', 'sigmasys']
    
    prior.add_parameter('Alens', dist=(-1, 1))
    prior.add_parameter('sigmasys', dist=(0, 2))
    
    def log_likelihood(param_dict):
        total_error = np.sqrt(errors**2 + param_dict['sigmasys']**2)
        chisq = np.sum((amplitudes - param_dict['Alens'])**2 / total_error**2)
        return -0.5 * (chisq + np.sum(np.log(total_error**2)))
    
    sampler = nautilus.Sampler(prior, log_likelihood, pool=40)
    sampler.run(verbose=verbose)
    
    points, log_w, log_l = sampler.posterior()
    samples = MCSamples(samples=points, weights=np.exp(log_w), names=parameters, labels=parameters)
    
    # Get the stats for the chain
    stats = samples.getMargeStats()
    
    if verbose:
        for param in parameters:
            param_stats = stats.parWithName(param)
            mean = param_stats.mean
            lower_1sigma = param_stats.limits[0].lower
            upper_1sigma = param_stats.limits[0].upper
            lower_2sigma = param_stats.limits[1].lower
            upper_2sigma = param_stats.limits[1].upper
            
            print(f"{param}:")
            print(f"  Mean: {mean}")
            print(f"  1-sigma: {mean} -{mean-lower_1sigma} +{upper_1sigma-mean}")
            print(f"  2-sigma: {mean} -{mean-lower_2sigma} +{upper_2sigma-mean}")
    
    meanA = np.average(amplitudes, weights=1/errors**2)
    delta = meanA - amplitudes
    reduced_chisq = np.sum(delta**2 / errors**2) / (len(amplitudes) - 1)
    
    param_stats = stats.parWithName('sigmasys')
    sigma_sys = param_stats.mean
    lower_1sigma = param_stats.limits[0].lower
    upper_1sigma = param_stats.limits[0].upper
    lower_2sigma = param_stats.limits[1].lower
    upper_2sigma = param_stats.limits[1].upper
    
    return reduced_chisq, [lower_2sigma, lower_1sigma, sigma_sys, upper_1sigma, upper_2sigma], [points, log_w, log_l]


def _compute_deltasigma_amplitude(
    data: np.ndarray,
    cov: np.ndarray,
    reference_dv: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Compute lensing signal amplitude by fitting to a reference data vector."""
    if reference_dv is None:
        amp = np.mean(data)
        amp_err = np.sqrt(np.sum(cov)) / len(data)
        return amp, amp_err
    
    try:
        inv_cov = np.linalg.inv(cov)
        numerator = np.einsum('i,ij,j', data, inv_cov, reference_dv)
        denominator = np.einsum('i,ij,j', reference_dv, inv_cov, reference_dv)
        amplitude = numerator / denominator
        amplitude_err = 1.0 / np.sqrt(denominator)
        return amplitude, amplitude_err
    except np.linalg.LinAlgError:
        diag_cov = np.diag(cov)
        weights = reference_dv**2 / diag_cov
        amplitude = np.sum(weights * data / reference_dv) / np.sum(weights)
        amplitude_err = 1.0 / np.sqrt(np.sum(weights))
        return amplitude, amplitude_err


def _get_ntot(galaxy_type: str, source_survey: str, lens_config: LensGalaxyConfig, 
              source_config: SourceSurveyConfig, n_rp_bins: int = 15) -> int:
    """Get total number of data points for a galaxy type and source survey."""
    survey_upper = source_survey.upper()
    if survey_upper in ["ALL_Y1", "ALL_Y3", "ALL"]:
        surveys = ["KIDS", "DES", "HSCY3" if "Y3" in survey_upper or survey_upper == "ALL" else "HSCY1"]
        return sum(_get_ntot(galaxy_type, s, lens_config, source_config, n_rp_bins) for s in surveys)
    n_lens_bins = lens_config.get_n_lens_bins()
    n_source_bins = source_config.get_n_tomographic_bins(source_survey)
    return n_lens_bins * n_source_bins * n_rp_bins


# Worker functions for multiprocessing (must be at module level)
def _source_redshift_slope_worker(args):
    """Worker function for source redshift slope multiprocessing."""
    (i, random_dv, covariances, zsource_dict, scale_categories, 
     galaxy_types, use_all_bins, analyzer) = args
    try:
        p, V = analyzer._compute_source_redshift_slope_pV(
            random_dv, covariances, zsource_dict, scale_categories, galaxy_types, use_all_bins
        )
        return p, V, i
    except Exception as e:
        logging.error(f"Error in job {i}: {e}\n{traceback.format_exc()}")
        return None


def _splits_worker(args):
    """Worker function for splits multiprocessing."""
    (i, random_dv, covariances, splits_to_consider, scale_categories, 
     n_splits, boost_correction, analyzer) = args
    try:
        p, V = analyzer._compute_splits_pV(
            random_dv, covariances, splits_to_consider,
            scale_categories, n_splits, boost_correction
        )
        return p, V, i
    except Exception as e:
        logging.error(f"Error in job {i}: {e}\n{traceback.format_exc()}")
        return None


class RandomsAnalyzer:
    """
    Main class for random data vector analysis and generation.
    
    This class provides functionality to:
    - Generate random realizations of data vectors
    - Perform statistical tests with random data
    - Analyze systematic effects through random tests
    - Calculate sigma_sys systematic uncertainties
    """
    
    def __init__(
        self,
        computation_config: ComputationConfig,
        lens_config: LensGalaxyConfig,
        source_config: SourceSurveyConfig,
        output_config: OutputConfig,
        logger: Optional[logging.Logger] = None,
        chris_path: Optional[str] = None
    ):
        """
        Initialize the randoms analyzer.
        
        Parameters
        ----------
        computation_config : ComputationConfig
            Computation settings
        lens_config : LensGalaxyConfig
            Lens galaxy configuration
        source_config : SourceSurveyConfig
            Source survey configuration  
        output_config : OutputConfig
            Output paths and settings
        logger : Optional[logging.Logger]
            Logger instance
        chris_path : Optional[str]
            Path to Chris Hirata's theory files (for 'chris' mock datavectors)
        """
        self.computation_config = computation_config
        self.lens_config = lens_config
        self.source_config = source_config
        self.output_config = output_config
        self.chris_path = chris_path or "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/model_inputs_desiy3"
        
        self.logger = logger or setup_logger(self.__class__.__name__)
        
        from ..config.path_manager import PathManager
        self.path_manager = PathManager(output_config, source_config)
        self.data_loader = DataLoader(
            lens_config, source_config, output_config, self.path_manager, logger
        )
        
        self.analysis_config = AnalysisConfig()
        self._setup_output_directories()
        
        # Cache for loaded source redshifts from data
        self._zsource_cache = {}
    
    def _setup_output_directories(self) -> None:
        """Create directories for saving results."""
        version = self.lens_config.get_catalogue_version()
        randoms_dir = Path(self.output_config.save_path) / version / "randoms"
        randoms_dir.mkdir(parents=True, exist_ok=True)
        self.randoms_dir = randoms_dir
    
    def prepare_randoms_datavector(
        self,
        use_theory_covariance: bool = True,
        datavector_type: str = "measured",
        account_for_cross_covariance: bool = True,
        pure_noise: bool = False,
        split_by: Optional[str] = None,
        split: Optional[int] = None,
        n_splits: int = 4,
        galaxy_types: Optional[List[str]] = None,
        tomographic: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Prepare data vectors and covariances for random analysis."""
        if use_theory_covariance and split_by is not None:
            self.logger.warning(
                f"Using theory covariance and splitting by {split_by} is not supported, "
                "setting use_theory_covariance to False"
            )
            use_theory_covariance = False
        
        if use_theory_covariance and not tomographic:
            self.logger.warning(
                "Theory covariances only available for tomographic analysis, "
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
                if tomographic:
                    data_key = f"{galaxy_type}_{source_survey}"
                    n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
                    full_data, cov = self.output_config.load_tomographic_data_and_covariance(
                        version, galaxy_type, source_survey, z_bins, n_source_bins,
                        statistic, pure_noise, split_by, split, n_splits,
                        self.computation_config.n_rp_bins, use_theory_covariance
                    )
                else:
                    data_key = f"{galaxy_type}_{source_survey}"
                    full_data, cov = self.output_config.load_non_tomographic_data_and_covariance(
                        version, galaxy_type, source_survey, z_bins,
                        statistic, pure_noise, split_by, split, n_splits,
                        self.computation_config.n_rp_bins
                    )
                
                if datavector_type == 'zero':
                    datavector = np.zeros_like(full_data)
                elif datavector_type == 'emulator':
                    datavector = self._generate_emulator_datavector(
                        galaxy_type, source_survey, full_data.shape
                    )
                elif datavector_type == 'chris':
                    datavector = self._load_chris_mock_datavector(
                        galaxy_type, source_survey, statistic
                    )
                elif datavector_type == 'measured':
                    datavector = full_data
                else:
                    raise ValueError(f"datavector_type {datavector_type} not known. "
                                   "Valid options: 'zero', 'emulator', 'chris', 'measured'")
                
                datavectors[data_key] = datavector
                covariances[data_key] = cov
        
        if account_for_cross_covariance and len(self.source_config.surveys) > 1:
            self._add_cross_covariances(datavectors, covariances, galaxy_types, 
                                       use_theory_covariance, pure_noise)
        
        return datavectors, covariances
    
    
    def _add_cross_covariances(
        self,
        datavectors: Dict[str, np.ndarray],
        covariances: Dict[str, np.ndarray],
        galaxy_types: List[str],
        use_theory_covariance: bool = True,
        pure_noise: bool = False
    ) -> None:
        """Add cross-survey covariances."""
        statistic = self.computation_config.statistics[0]
        
        for galaxy_type in galaxy_types:
            survey_keys = [key for key in datavectors.keys() if key.startswith(galaxy_type)]
            
            if len(survey_keys) > 1:
                hscy3 = any("hscy3" in key.lower() for key in survey_keys)
                hscy1 = any("hscy1" in key.lower() for key in survey_keys)
                
                if hscy3:
                    all_survey_name = "all_y3"
                    survey_order = ['KiDS', 'DES', 'HSCY3']
                elif hscy1:
                    all_survey_name = "all_y1"
                    survey_order = ['KiDS', 'DES', 'HSCY1']
                else:
                    all_survey_name = "all"
                    survey_order = ['KiDS', 'DES', 'HSCY3']
                
                ordered_keys = []
                for survey in survey_order:
                    for key in survey_keys:
                        if survey.lower() in key.lower():
                            ordered_keys.append(key)
                            break
                
                if len(ordered_keys) != len(survey_keys):
                    ordered_keys = survey_keys
                
                combined_data = np.concatenate([datavectors[key] for key in ordered_keys])
                
                combined_cov = None
                if use_theory_covariance:
                    combined_cov = self.output_config.load_theory_covariance(
                        galaxy_type, all_survey_name, statistic, pure_noise,
                        z_bins=self.lens_config.z_bins
                    )
                
                if combined_cov is None:
                    self.logger.info(f"Using block diagonal covariance for {galaxy_type}_all")
                    cov_blocks = [covariances[key] for key in ordered_keys]
                    combined_cov = self._combine_covariance_blocks(cov_blocks)
                
                all_key = f"{galaxy_type}_all"
                datavectors[all_key] = combined_data
                covariances[all_key] = combined_cov
    
    def _combine_covariance_blocks(self, cov_blocks: List[np.ndarray]) -> np.ndarray:
        """Combine covariance blocks into a block diagonal matrix."""
        total_size = sum(cov.shape[0] for cov in cov_blocks)
        combined = np.zeros((total_size, total_size))
        offset = 0
        for cov in cov_blocks:
            size = cov.shape[0]
            combined[offset:offset+size, offset:offset+size] = cov
            offset += size
        return combined
    
    def _generate_emulator_datavector(
        self,
        galaxy_type: str,
        source_survey: str,
        target_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Generate emulator-based reference data vector.
        
        Uses reference data vectors (e.g., from Abacus HOD) for each lens bin,
        replicated across source bins.
        
        Parameters
        ----------
        galaxy_type : str
            Type of lens galaxies
        source_survey : str
            Source survey name
        target_shape : Tuple[int, ...]
            Target shape for the output datavector
            
        Returns
        -------
        np.ndarray
            Emulator-based data vector
        """
        n_lens_bins = self.lens_config.get_n_lens_bins()
        n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
        n_rp_bins = self.computation_config.n_rp_bins
        
        datavector = np.zeros(target_shape[0] if isinstance(target_shape, tuple) else target_shape)
        
        counter = 0
        for lens_bin in range(n_lens_bins):
            # Load reference datavector for this lens bin
            ref_dv = self.output_config.load_reference_datavector(galaxy_type, lens_bin)
            
            if ref_dv is None:
                self.logger.warning(f"No reference datavector for {galaxy_type} lens_bin {lens_bin}, using zeros")
                ref_dv = np.zeros(n_rp_bins)
            elif len(ref_dv) != n_rp_bins:
                self.logger.warning(f"Reference datavector has {len(ref_dv)} bins, expected {n_rp_bins}")
                # Interpolate or truncate as needed
                if len(ref_dv) > n_rp_bins:
                    ref_dv = ref_dv[:n_rp_bins]
                else:
                    ref_dv = np.pad(ref_dv, (0, n_rp_bins - len(ref_dv)), constant_values=0)
            
            # Replicate across source bins
            for source_bin in range(n_source_bins):
                start_idx = counter * n_rp_bins
                end_idx = start_idx + n_rp_bins
                if end_idx <= len(datavector):
                    datavector[start_idx:end_idx] = ref_dv
                counter += 1
        
        # Verify no zeros in emulator datavector (should have signal everywhere)
        if np.any(np.isclose(datavector, 0)):
            self.logger.warning("Emulator datavector contains zeros - some reference DVs may be missing")
        
        return datavector
    
    def _load_chris_mock_datavector(
        self,
        galaxy_type: str,
        source_survey: str,
        statistic: str = "deltasigma"
    ) -> np.ndarray:
        """
        Load mock data vector from Chris Hirata's theory files.
        
        Parameters
        ----------
        galaxy_type : str
            Type of lens galaxies ('BGS_BRIGHT' or 'LRG')
        source_survey : str  
            Source survey name
        statistic : str
            Statistic type ('deltasigma' or 'gammat')
            
        Returns
        -------
        np.ndarray
            Mock data vector from theory prediction
        """
        # Map statistic to file prefix
        if statistic == "deltasigma":
            fstat = "ds"
            append = "_pzwei"
        elif statistic == "gammat":
            fstat = "gt"
            append = ""
        else:
            raise ValueError(f"Invalid statistic {statistic}")
        
        # Map galaxy type to file format
        if galaxy_type.upper() in ["BGS", "BGS_BRIGHT"]:
            fgal = "bgs"
        elif galaxy_type.upper() in ["LRG"]:
            fgal = "lrg"
        else:
            raise ValueError(f"Invalid galaxy type {galaxy_type}")
        
        # Map source survey to file format
        source_lower = source_survey.lower()
        if source_lower == "kids":
            fsurv = "kids1000"
        elif source_lower == "des":
            fsurv = "desy3"
        elif source_lower == "hscy1":
            fsurv = "hscy1"
        elif source_lower == "hscy3":
            fsurv = "hscy3"
        elif source_lower == "decade":
            fsurv = "decade"
        else:
            fsurv = source_lower
        
        # Build filepath
        filename = f"{fstat}modvec_{fsurv}desiy3{fgal}{append}.dat"
        filepath = Path(self.chris_path) / filename
        
        if not filepath.exists():
            # Try in model_inputs_desiy1 subdirectory
            filepath = Path(self.chris_path) / "model_inputs_desiy1" / filename
        
        if not filepath.exists():
            self.logger.warning(f"Chris mock DV file not found: {filepath}")
            # Return zeros as fallback
            n_lens_bins = self.lens_config.get_n_lens_bins()
            n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
            n_rp_bins = self.computation_config.n_rp_bins
            return np.zeros(n_lens_bins * n_source_bins * n_rp_bins)
        
        self.logger.info(f"Loading Chris mock DV from {filepath}")
        data = np.loadtxt(str(filepath))
        
        # File format has multiple columns, we need the signal column (usually column 1)
        if len(data.shape) > 1:
            dv = data[:, 1]
        else:
            dv = data
        
        return dv
    
    def _load_source_redshifts_from_data(
        self,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load actual mean source and lens redshifts from measurement data.
        
        Parameters
        ----------
        galaxy_type : str
            Type of lens galaxies
        source_survey : str
            Source survey name
        lens_bin : int
            Lens bin index
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays of (z_lens, z_source) for each source bin
        """
        cache_key = f"{galaxy_type}_{source_survey}_{lens_bin}"
        
        if cache_key in self._zsource_cache:
            return self._zsource_cache[cache_key]
        
        version = self.lens_config.get_catalogue_version()
        z_bins = self.lens_config.z_bins
        n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
        statistic = self.computation_config.statistics[0]
        
        zlenses = []
        zsources = []
        
        for source_bin in range(n_source_bins):
            results = self.output_config.load_lensing_results(
                version, galaxy_type, source_survey, lens_bin, z_bins,
                statistic, source_bin=source_bin
            )
            
            if results is not None and 'z_l' in results.columns and 'z_s' in results.columns:
                zlenses.append(np.mean(results['z_l']))
                zsources.append(np.mean(results['z_s']))
            else:
                # Fall back to defaults
                zlenses.append(self._get_default_lens_redshift(galaxy_type, lens_bin))
                zsources.append(self._get_default_source_redshift(source_survey, source_bin))
        
        result = (np.array(zlenses), np.array(zsources))
        self._zsource_cache[cache_key] = result
        return result
    
    def _get_default_lens_redshift(self, galaxy_type: str, lens_bin: int) -> float:
        """Get default mean lens redshift for a galaxy type and bin."""
        z_bins = self.lens_config.z_bins
        if lens_bin + 1 < len(z_bins):
            return (z_bins[lens_bin] + z_bins[lens_bin + 1]) / 2
        return 0.5  # fallback
    
    def _apply_delta_z_shifts(
        self,
        full_data: np.ndarray,
        full_zlens: np.ndarray,
        full_zsource: np.ndarray,
        galaxy_type: str,
        source_survey: str,
        delta_z_shifts: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply photo-z systematic shifts to HSC Y3 data.
        
        This implements the same shift correction as the old code,
        recalculating critical surface density with shifted redshifts.
        
        Parameters
        ----------
        full_data : np.ndarray
            Full data vector
        full_zlens : np.ndarray  
            Lens redshifts for each data point
        full_zsource : np.ndarray
            Source redshifts for each data point
        galaxy_type : str
            Type of lens galaxies
        source_survey : str
            Source survey (should be HSCY3)
        delta_z_shifts : List[float]
            Photo-z shifts for each source tomographic bin
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Corrected (data, zlens, zsource) arrays
        """
        if source_survey.upper() != "HSCY3":
            self.logger.warning(f"Delta-z shifts only implemented for HSCY3, got {source_survey}")
            return full_data, full_zlens, full_zsource
        
        try:
            from astropy.cosmology import Planck18 as desicosmo
            cosmo = desicosmo.clone(name='Planck18_h1', H0=100)
            from dsigma.physics import effective_critical_surface_density
        except ImportError:
            self.logger.warning("Cannot apply delta-z shifts without astropy.cosmology and dsigma")
            return full_data, full_zlens, full_zsource
        
        n_data = len(full_data)
        data = np.copy(full_data)
        zsource = np.copy(full_zsource)
        
        # Determine bin structure
        n_lens_bins = self.lens_config.get_n_lens_bins()
        n_source_bins = 4  # HSC Y3 has 4 tomographic bins
        n_rp_bins = self.computation_config.n_rp_bins
        
        total_per_lens = n_source_bins * n_rp_bins
        
        # Load n(z) for HSC Y3
        try:
            nofz_source = self.data_loader._read_nofz("HSCY3")
        except Exception as e:
            self.logger.warning(f"Could not load HSC Y3 n(z): {e}")
            return full_data, full_zlens, full_zsource
        
        version = self.lens_config.get_catalogue_version()
        
        for lens_bin in range(n_lens_bins):
            # Get lens n(z)
            try:
                from ..utils.nofz_utils import get_lens_nofz
                nofz_lens = get_lens_nofz(galaxy_type, "HSCY3", version)
            except (ImportError, Exception) as e:
                self.logger.warning(f"Could not load lens n(z): {e}")
                continue
            
            for source_bin in range(n_source_bins):
                if source_bin >= len(delta_z_shifts):
                    continue
                
                dz = delta_z_shifts[source_bin]
                if np.abs(dz) < 1e-6:
                    continue  # No shift needed
                
                # Calculate fiducial critical surface density
                try:
                    sigmacrit_fiducial = effective_critical_surface_density(
                        nofz_lens['z_mid'],
                        nofz_source['z'],
                        nofz_source['n'][:, source_bin],
                        cosmo
                    )
                    
                    # Shifted redshift distribution
                    z_arr_shifted = nofz_source['z'] + dz
                    mask = (z_arr_shifted >= 0)
                    
                    sigmacrit_shifted = effective_critical_surface_density(
                        nofz_lens['z_mid'],
                        z_arr_shifted[mask],
                        nofz_source['n'][:, source_bin][mask],
                        cosmo
                    )
                    
                    # Compute amplitude shift
                    amplitude_shift = (
                        np.average(sigmacrit_shifted, weights=nofz_lens[f'n_{lens_bin+1}']) /
                        np.average(sigmacrit_fiducial, weights=nofz_lens[f'n_{lens_bin+1}'])
                    )
                except Exception as e:
                    self.logger.warning(f"Error computing delta-z shift for lens {lens_bin} source {source_bin}: {e}")
                    amplitude_shift = 1.0
                
                # Apply shift to data
                start_idx = (lens_bin * n_source_bins + source_bin) * n_rp_bins
                end_idx = start_idx + n_rp_bins
                
                if end_idx <= len(data):
                    data[start_idx:end_idx] *= amplitude_shift
                    zsource[start_idx:end_idx] += dz
                    
                    self.logger.debug(
                        f"{galaxy_type} l{lens_bin+1}, s{source_bin+1}, "
                        f"amplitude shift: {amplitude_shift:.3f}, zshift: {dz:.3f}"
                    )
        
        return data, full_zlens, zsource
    
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
            Dictionary of data vectors keyed by survey/galaxy type
        covariances : Dict[str, np.ndarray]
            Dictionary of covariance matrices
        n_randoms : int
            Number of random realizations to generate
        method : str
            Generation method: "numpy" (default) or "jax" (GPU-accelerated)
        random_seed : int
            Random seed for reproducibility
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of random data vectors with shape (n_randoms, n_data_points)
        """
        randoms = {}
        np.random.seed(random_seed)
        
        survey_keys = list(datavectors.keys())
        combined_cov = any("all" in key for key in survey_keys)
        
        if combined_cov:
            draw_keys = [key for key in survey_keys if "all" in key]
        else:
            draw_keys = survey_keys
        
        if method == "jax":
            randoms = self._generate_randoms_jax(datavectors, covariances, n_randoms, 
                                                 draw_keys, random_seed, combined_cov, survey_keys)
        else:
            randoms = self._generate_randoms_numpy(datavectors, covariances, n_randoms,
                                                   draw_keys, random_seed, combined_cov, survey_keys)
        
        return randoms
    
    def _generate_randoms_numpy(
        self,
        datavectors: Dict[str, np.ndarray],
        covariances: Dict[str, np.ndarray],
        n_randoms: int,
        draw_keys: List[str],
        random_seed: int,
        combined_cov: bool,
        survey_keys: List[str]
    ) -> Dict[str, np.ndarray]:
        """Generate random datavectors using numpy (CPU)."""
        randoms = {}
        
        for key in draw_keys:
            if random_seed == 0:
                self.logger.info(f"Generating randoms for {key}, covariance shape: {covariances[key].shape}, datavector shape: {datavectors[key].shape}")
            
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
                    diag_cov = np.diag(np.diag(covariances[key]))
                    randoms[key] = np.random.multivariate_normal(
                        datavectors[key], diag_cov, size=n_randoms
                    )
            
            if random_seed == 0:
                self.logger.info(f"Generated {key}: {randoms[key].shape}")
                if np.any(np.isnan(randoms[key])):
                    self.logger.warning(f"NaNs found in generated randoms for {key}")
            
            if combined_cov and "all" in key:
                self._split_combined_randoms(randoms, key, survey_keys)
        
        return randoms
    
    def _generate_randoms_jax(
        self,
        datavectors: Dict[str, np.ndarray],
        covariances: Dict[str, np.ndarray],
        n_randoms: int,
        draw_keys: List[str],
        random_seed: int,
        combined_cov: bool,
        survey_keys: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Generate random datavectors using JAX (GPU-accelerated).
        
        Falls back to numpy if JAX is not available.
        """
        try:
            from jax import random
            import jax.numpy as jnp
        except ImportError:
            self.logger.warning("JAX not available, falling back to numpy method")
            return self._generate_randoms_numpy(datavectors, covariances, n_randoms,
                                               draw_keys, random_seed, combined_cov, survey_keys)
        
        randoms = {}
        jnp_key = random.PRNGKey(random_seed)
        
        for key in draw_keys:
            if random_seed == 0:
                self.logger.info(f"Generating randoms for {key} using JAX, covariance shape: {covariances[key].shape}")
            
            try:
                # JAX multivariate normal
                randoms[key] = np.array(random.multivariate_normal(
                    jnp_key, 
                    jnp.array(datavectors[key]),
                    jnp.array(covariances[key]), 
                    shape=(n_randoms,)
                ))
                # Update key for next iteration
                jnp_key, _ = random.split(jnp_key)
                
            except Exception as e:
                self.logger.warning(f"JAX generation failed for {key}: {e}, falling back to numpy")
                randoms[key] = np.random.multivariate_normal(
                    datavectors[key], covariances[key], size=n_randoms
                )
            
            if random_seed == 0:
                self.logger.info(f"Generated {key}: {randoms[key].shape}")
                if np.any(np.isnan(randoms[key])):
                    self.logger.warning(f"NaNs found in generated randoms for {key}")
            
            if combined_cov and "all" in key:
                self._split_combined_randoms(randoms, key, survey_keys)
        
        return randoms
    
    def _split_combined_randoms(self, randoms: Dict[str, np.ndarray], combined_key: str, 
                                 all_keys: List[str]) -> None:
        """Split combined random data vectors back into individual surveys."""
        galaxy_type = combined_key.split("_")[0]
        individual_keys = [key for key in all_keys if key.startswith(galaxy_type) and "all" not in key]
        
        if len(individual_keys) > 1:
            offsets = {}
            offset = 0
            
            for key in individual_keys:
                source_survey = key.split("_")[-1]
                ntot = _get_ntot(galaxy_type, source_survey, self.lens_config, 
                                self.source_config, self.computation_config.n_rp_bins)
                offsets[key] = (offset, offset + ntot)
                offset += ntot
            
            for key in individual_keys:
                start, end = offsets[key]
                randoms[key] = randoms[combined_key][:, start:end]
    
    def _get_scale_mask_for_category(self, scale_category: str, galaxy_type: str, 
                                      lens_bin: int) -> np.ndarray:
        """Get boolean mask for radial bins in a given scale category."""
        n_rp_bins = self.computation_config.n_rp_bins
        rp_bins = np.logspace(
            np.log10(self.computation_config.rp_min),
            np.log10(self.computation_config.rp_max),
            n_rp_bins + 1
        )
        rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
        
        scale_cuts = self.analysis_config.get_scale_cuts(
            self.source_config.surveys[0], self.computation_config.statistics[0]
        )
        rp_pivot = scale_cuts.get("rp_pivot", 3.0)
        rp_min = scale_cuts.get("min", 0.1)
        rp_max = scale_cuts.get("max", 30.0)
        
        if scale_category.lower() == "small scales":
            mask = (rp_centers >= rp_min) & (rp_centers < rp_pivot)
        elif scale_category.lower() == "large scales":
            mask = (rp_centers >= rp_pivot) & (rp_centers <= rp_max)
        else:
            mask = (rp_centers >= rp_min) & (rp_centers <= rp_max)
        return mask
    
    def _get_default_source_redshift(
        self, 
        source_survey: str, 
        source_bin: int,
        galaxy_type: Optional[str] = None,
        lens_bin: Optional[int] = None,
        load_from_data: bool = False
    ) -> float:
        """
        Get mean source redshift for a survey and bin.
        
        Parameters
        ----------
        source_survey : str
            Source survey name
        source_bin : int
            Source tomographic bin index
        galaxy_type : Optional[str]
            Galaxy type (needed for loading from data)
        lens_bin : Optional[int]
            Lens bin index (needed for loading from data)
        load_from_data : bool
            If True, try to load from actual measurement files first
            
        Returns
        -------
        float
            Mean source redshift
        """
        # Try loading from data if requested
        if load_from_data and galaxy_type is not None and lens_bin is not None:
            try:
                _, zsources = self._load_source_redshifts_from_data(
                    galaxy_type, source_survey, lens_bin
                )
                if source_bin < len(zsources) and np.isfinite(zsources[source_bin]):
                    return zsources[source_bin]
            except Exception as e:
                self.logger.debug(f"Could not load z_source from data: {e}")
        
        # Fall back to hardcoded defaults
        default_zsource = {
            "KIDS": [0.1, 0.3, 0.5, 0.7, 0.9],
            "DES": [0.35, 0.63, 0.87, 1.1],
            "HSCY1": [0.4, 0.6, 0.8, 1.0],
            "HSCY3": [0.4, 0.6, 0.8, 1.0],
            "DECADE": [0.4, 0.6, 0.8, 1.0],
            "DECADE_NGC": [0.4, 0.6, 0.8, 1.0],
            "DECADE_SGC": [0.4, 0.6, 0.8, 1.0],
            "SDSS": [0.4],
        }
        survey_upper = source_survey.upper()
        if survey_upper in default_zsource:
            zsources = default_zsource[survey_upper]
            if source_bin < len(zsources):
                return zsources[source_bin]
        return 0.5 + 0.2 * source_bin
    
    def _compute_source_redshifts(
        self, 
        galaxy_types: List[str], 
        scale_categories: List[str],
        use_all_bins: bool = False,
        load_from_data: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute source redshifts for each lens bin and scale category.
        
        Parameters
        ----------
        galaxy_types : List[str]
            List of galaxy types to process
        scale_categories : List[str]
            Scale categories (e.g., 'small scales', 'large scales')
        use_all_bins : bool
            If True, use all source bins instead of allowed bins only
        load_from_data : bool
            If True, try to load z_source from measurement files
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping keys to source redshift arrays
        """
        zsource_dict = {}
        for galaxy_type in galaxy_types:
            n_lens_bins = self.lens_config.get_n_lens_bins()
            for scale_category in scale_categories:
                for lens_bin in range(n_lens_bins):
                    key = f"{galaxy_type}_{scale_category}_{lens_bin}"
                    zsources = []
                    for source_survey in self.source_config.surveys:
                        n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
                        if use_all_bins:
                            allowed_bins = list(range(n_source_bins))
                        else:
                            z_max_bin = self.lens_config.z_bins[lens_bin + 1]
                            allowed_bins = self.analysis_config.get_allowed_source_bins(
                                galaxy_type, source_survey, z_max_bin
                            )
                        for source_bin in allowed_bins:
                            zsource = self._get_default_source_redshift(
                                source_survey, source_bin,
                                galaxy_type=galaxy_type,
                                lens_bin=lens_bin,
                                load_from_data=load_from_data
                            )
                            zsources.append(zsource)
                    zsource_dict[key] = np.array(zsources)
        return zsource_dict
    
    def _compute_source_redshift_slope_pV(
        self,
        datavector: Dict[str, np.ndarray],
        covariances: Dict[str, np.ndarray],
        zsource_dict: Dict[str, np.ndarray],
        scale_categories: List[str],
        galaxy_types: List[str],
        use_all_bins: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute slope fit parameters (p, V) for source redshift slope analysis."""
        p_dict = {}
        V_dict = {}
        n_rp_bins = self.computation_config.n_rp_bins
        
        for galaxy_type in galaxy_types:
            n_lens_bins = self.lens_config.get_n_lens_bins()
            
            for scale_category in scale_categories:
                for lens_bin in range(n_lens_bins):
                    key = f"{galaxy_type}_{scale_category}_{lens_bin}"
                    scale_mask = self._get_scale_mask_for_category(scale_category, galaxy_type, lens_bin)
                    
                    all_amplitudes = []
                    all_amplitude_errors = []
                    all_zsource = []
                    
                    for source_survey in self.source_config.surveys:
                        dv_key = f"{galaxy_type}_{source_survey}"
                        if dv_key not in datavector:
                            continue
                        
                        n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
                        if use_all_bins:
                            allowed_bins = list(range(n_source_bins))
                        else:
                            z_max_bin = self.lens_config.z_bins[lens_bin + 1]
                            allowed_bins = self.analysis_config.get_allowed_source_bins(
                                galaxy_type, source_survey, z_max_bin
                            )
                        
                        for source_bin in allowed_bins:
                            bin_start = (lens_bin * n_source_bins + source_bin) * n_rp_bins
                            bin_end = bin_start + n_rp_bins
                            
                            if bin_end > len(datavector[dv_key]):
                                continue
                            
                            data = datavector[dv_key][bin_start:bin_end][scale_mask]
                            
                            cov_key = dv_key
                            if cov_key in covariances:
                                full_cov = covariances[cov_key]
                                if full_cov.shape[0] > bin_end:
                                    cov = full_cov[bin_start:bin_end, bin_start:bin_end]
                                    cov = cov[np.ix_(scale_mask, scale_mask)]
                                else:
                                    cov = np.eye(len(data)) * 0.01
                            else:
                                cov = np.eye(len(data)) * 0.01
                            
                            ref_dv = self.output_config.load_reference_datavector(galaxy_type, lens_bin)
                            if ref_dv is not None and len(ref_dv) >= len(scale_mask):
                                ref_dv = ref_dv[scale_mask]
                            else:
                                ref_dv = None
                            
                            amplitude, amplitude_err = _compute_deltasigma_amplitude(data, cov, ref_dv)
                            zsource = self._get_default_source_redshift(source_survey, source_bin)
                            
                            if np.isfinite(amplitude) and np.isfinite(amplitude_err):
                                all_amplitudes.append(amplitude)
                                all_amplitude_errors.append(amplitude_err)
                                all_zsource.append(zsource)
                    
                    all_amplitudes = np.array(all_amplitudes)
                    all_amplitude_errors = np.array(all_amplitude_errors)
                    all_zsource = np.array(all_zsource)
                    
                    if len(all_amplitudes) >= 2:
                        try:
                            if len(all_amplitudes) > 2:
                                p, V = np.polyfit(all_zsource, all_amplitudes, 1,
                                                w=1/all_amplitude_errors, cov=True)
                            else:
                                p = np.polyfit(all_zsource, all_amplitudes, 1, w=1/all_amplitude_errors)
                                V = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                        except (np.linalg.LinAlgError, ValueError):
                            p = np.polyfit(all_zsource, all_amplitudes, 1)
                            V = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                        p_dict[key] = p
                        V_dict[key] = V
                    else:
                        p_dict[key] = np.array([np.nan, np.nan])
                        V_dict[key] = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        
        return p_dict, V_dict
    
    def _get_ntile_n_splits(self, galaxy_type: str, computed_only: bool = False) -> int:
        """
        Get number of NTILE splits for a galaxy type.
        
        Parameters
        ----------
        galaxy_type : str
            Type of lens galaxies
        computed_only : bool
            If True, return only the number of computed splits (may be less than total)
            
        Returns
        -------
        int
            Number of NTILE splits
        """
        # Check if analysis_config has NTILE configuration
        if hasattr(self.analysis_config, 'ntile_splits'):
            config = self.analysis_config.ntile_splits
            gtype = galaxy_type.upper()
            
            if computed_only:
                key = f"n_ntile_computed_{gtype[:3].lower()}"
                if key in config:
                    return config[key]
            
            key = f"n_ntile_{gtype[:3].lower()}"
            if key in config:
                return config[key]
        
        # Fall back to defaults
        ntile_splits = {"BGS_BRIGHT": 4, "BGS": 4, "LRG": 3}
        ntile_splits_computed = {"BGS_BRIGHT": 4, "BGS": 4, "LRG": 3}
        
        gtype = galaxy_type.upper()
        if computed_only:
            return ntile_splits_computed.get(gtype, 4)
        return ntile_splits.get(gtype, 4)
    
    def _compute_splits_pV(
        self,
        datavectors: Dict[str, np.ndarray],
        covariances: Dict[str, np.ndarray],
        splits_to_consider: List[str],
        scale_categories: List[str],
        n_splits: int = 4,
        boost_correction: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute slope fit parameters (p, V) for splits analysis."""
        p_dict = {}
        V_dict = {}
        
        galaxy_type = self.lens_config.galaxy_type
        n_lens_bins = self.lens_config.get_n_lens_bins()
        version = self.lens_config.get_catalogue_version()
        z_bins = self.lens_config.z_bins
        statistic = self.computation_config.statistics[0]
        n_rp_bins = self.computation_config.n_rp_bins
        
        for split_by in splits_to_consider:
            if split_by.lower() == "ntile":
                current_n_splits = self._get_ntile_n_splits(galaxy_type)
            else:
                current_n_splits = n_splits
            
            for scale_category in scale_categories:
                for lens_bin in range(n_lens_bins):
                    key = f"{split_by}_{galaxy_type}_{scale_category}_{lens_bin}"
                    scale_mask = self._get_scale_mask_for_category(scale_category, galaxy_type, lens_bin)
                    
                    all_split_values = []
                    all_amplitudes = []
                    all_amplitude_errors = []
                    
                    for source_survey in self.source_config.surveys:
                        for split in range(current_n_splits):
                            dv_key = f"{galaxy_type}_{source_survey}_{split_by}_{split}_of_{n_splits}"
                            
                            if dv_key not in datavectors:
                                continue
                            
                            data = datavectors[dv_key]
                            if len(data.shape) == 2:
                                data = data[lens_bin]
                            elif len(data.shape) == 1:
                                start = lens_bin * n_rp_bins
                                end = start + n_rp_bins
                                if end > len(data):
                                    continue
                                data = data[start:end]
                            
                            data = data[scale_mask]
                            
                            cov_key = dv_key
                            if cov_key in covariances:
                                cov = covariances[cov_key]
                                if len(cov.shape) == 3:
                                    cov = cov[lens_bin]
                                cov = cov[np.ix_(scale_mask, scale_mask)]
                            else:
                                cov = np.eye(len(data)) * 0.01
                            
                            ref_dv = self.output_config.load_reference_datavector(galaxy_type, lens_bin)
                            if ref_dv is not None and len(ref_dv) >= len(scale_mask):
                                ref_dv = ref_dv[scale_mask]
                            else:
                                ref_dv = None
                            
                            amplitude, amplitude_err = _compute_deltasigma_amplitude(data, cov, ref_dv)
                            
                            split_value = self.output_config.load_split_value(
                                version, galaxy_type, source_survey, lens_bin, z_bins,
                                statistic, split_by, split, n_splits, boost_correction
                            )
                            if split_value is not None:
                                split_val = split_value[0]
                            else:
                                split_val = float(split)
                            
                            if np.isfinite(amplitude) and np.isfinite(amplitude_err):
                                all_split_values.append(split_val)
                                all_amplitudes.append(amplitude)
                                all_amplitude_errors.append(amplitude_err)
                    
                    all_amplitudes = np.array(all_amplitudes)
                    all_amplitude_errors = np.array(all_amplitude_errors)
                    all_split_values = np.array(all_split_values)
                    
                    if len(all_amplitudes) >= 2:
                        try:
                            if len(all_amplitudes) > 2:
                                p, V = np.polyfit(all_split_values, all_amplitudes, 1,
                                                w=1/all_amplitude_errors**2, cov=True)
                            else:
                                p = np.polyfit(all_split_values, all_amplitudes, 1)
                                V = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                        except (np.linalg.LinAlgError, ValueError):
                            p = np.polyfit(all_split_values, all_amplitudes, 1)
                            V = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                        p_dict[key] = p
                        V_dict[key] = V
                    else:
                        p_dict[key] = np.array([np.nan, np.nan])
                        V_dict[key] = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        
        return p_dict, V_dict
    
    def generate_random_source_redshift_slope_test(
        self,
        n_randoms: int = 1000,
        n_processes: int = 4,
        use_theory_covariance: bool = True,
        datavector_type: str = "measured",
        kwargs_dict: Optional[Dict[str, Any]] = None,
        filename_suffix: str = "",
        use_all_bins: bool = False
    ) -> None:
        """
        Generate random realizations for source redshift slope analysis.
        
        For source redshift slope analysis, the entire covariance matrix including
        cross-covariances between surveys is used to generate correlated random
        data vectors.
        """
        self.logger.info(f"Generating {n_randoms} random realizations for source redshift slope test")
        self.logger.info(f"Using cross-survey covariances: {use_theory_covariance}")
        
        if kwargs_dict is None:
            kwargs_dict = {}
        
        galaxy_types = [self.lens_config.galaxy_type]
        scale_categories = self.analysis_config.get_scale_categories()
        
        self.logger.info("Preparing data vectors and covariances")
        datavectors, covariances = self.prepare_randoms_datavector(
            use_theory_covariance=use_theory_covariance,
            datavector_type=datavector_type,
            account_for_cross_covariance=True,
            tomographic=True
        )
        
        self.logger.info("Generating random data vectors with full covariance")
        randoms_datvecs = self.generate_randoms_datavectors(
            datavectors, covariances, n_randoms, random_seed=0
        )
        
        zsource_dict = self._compute_source_redshifts(galaxy_types, scale_categories, use_all_bins)
        
        self.logger.info("Testing run with seed 0")
        p_test, V_test = self._compute_source_redshift_slope_pV(
            datavectors, covariances, zsource_dict, scale_categories, galaxy_types, use_all_bins
        )
        all_keys = list(p_test.keys())
        self.logger.info(f"Test run completed. {len(all_keys)} keys")
        
        p_arr = np.zeros((n_randoms, len(all_keys), 2))
        V_arr = np.zeros((n_randoms, len(all_keys), 2, 2))
        
        self.logger.info(f"Running analysis on {n_randoms} randoms with {n_processes} processes")
        
        with Pool(n_processes) as pool:
            with tqdm(total=n_randoms, desc="Source z-slope randoms") as pbar:
                def fill_arrays(result):
                    if result is not None:
                        p, V, i = result
                        for k, key in enumerate(all_keys):
                            p_arr[i, k, :] = p[key]
                            V_arr[i, k, :, :] = V[key]
                    pbar.update(1)
                
                jobs = []
                for i in range(n_randoms):
                    random_dv = {key: randoms_datvecs[key][i] for key in randoms_datvecs.keys()}
                    jobs.append(pool.apply_async(
                        _source_redshift_slope_worker,
                        [(i, random_dv, covariances, zsource_dict, scale_categories, 
                          galaxy_types, use_all_bins, self)],
                        callback=fill_arrays
                    ))
                
                for job in jobs:
                    job.wait()
        
        output_file = self.randoms_dir / f"redshift_slope_tomo_p_arr{filename_suffix}.npy"
        np.save(output_file, p_arr)
        self.logger.info(f"Saved p_arr to {output_file} (shape: {p_arr.shape})")
        
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
        datavector_type: str = "measured",
        splits_to_consider: Optional[List[str]] = None,
        scale_categories: Optional[List[str]] = None,
        n_splits: int = 4,
        boost_correction: bool = False,
        filename_suffix: str = ""
    ) -> None:
        """
        Generate random realizations for data splits analysis.
        
        This generates random data vectors for each split, computes lensing
        amplitudes, and fits slopes to assess the null distribution of slope
        values. The results are saved with keys in format:
        '{split_by}_{galaxy_type}_{scale_category}_{lens_bin}'
        """
        self.logger.info(f"Generating {n_randoms} random realizations for splits test")
        
        if splits_to_consider is None:
            splits_to_consider = ["NTILE"]
        if scale_categories is None:
            scale_categories = self.analysis_config.get_scale_categories()
        
        galaxy_type = self.lens_config.galaxy_type
        galaxy_types = [galaxy_type]
        
        self.logger.info("Preparing datavectors and covariances for all splits")
        all_datavectors = {}
        all_covariances = {}
        
        for split_by in splits_to_consider:
            if split_by.lower() == "ntile":
                current_n_splits = self._get_ntile_n_splits(galaxy_type)
            else:
                current_n_splits = n_splits
            
            for split in range(current_n_splits):
                datavectors, covariances = self.prepare_randoms_datavector(
                    account_for_cross_covariance=False,
                    use_theory_covariance=use_theory_covariance,
                    datavector_type=datavector_type,
                    split_by=split_by,
                    split=split,
                    n_splits=n_splits,
                    galaxy_types=galaxy_types,
                    tomographic=False
                )
                
                for key in datavectors.keys():
                    full_key = f"{key}_{split_by}_{split}_of_{n_splits}"
                    all_datavectors[full_key] = datavectors[key]
                    all_covariances[full_key] = covariances[key]
        
        self.logger.info("Generating random data vectors")
        randoms_datvecs = self.generate_randoms_datavectors(
            all_datavectors, all_covariances, n_randoms, random_seed=0
        )
        
        self.logger.info("Testing run with seed 0")
        p_test, V_test = self._compute_splits_pV(
            all_datavectors, all_covariances, splits_to_consider, 
            scale_categories, n_splits, boost_correction
        )
        all_keys = list(p_test.keys())
        self.logger.info(f"Test run completed. {len(all_keys)} keys")
        
        p_arr = np.zeros((n_randoms, len(all_keys), 2))
        V_arr = np.zeros((n_randoms, len(all_keys), 2, 2))
        
        self.logger.info(f"Running analysis on {n_randoms} randoms with {n_processes} processes")
        
        with Pool(n_processes) as pool:
            with tqdm(total=n_randoms, desc="Splits randoms") as pbar:
                def fill_arrays(result):
                    if result is not None:
                        p, V, i = result
                        for k, key in enumerate(all_keys):
                            p_arr[i, k, :] = p[key]
                            V_arr[i, k, :, :] = V[key]
                    pbar.update(1)
                
                jobs = []
                for i in range(n_randoms):
                    random_dv = {key: randoms_datvecs[key][i] for key in randoms_datvecs.keys()}
                    jobs.append(pool.apply_async(
                        _splits_worker,
                        [(i, random_dv, all_covariances, splits_to_consider,
                          scale_categories, n_splits, boost_correction, self)],
                        callback=fill_arrays
                    ))
                
                for job in jobs:
                    job.wait()
        
        output_file = self.randoms_dir / f"splits_p_arr{filename_suffix}.npy"
        np.save(output_file, p_arr)
        self.logger.info(f"Saved splits p_arr to {output_file} (shape: {p_arr.shape})")
        
        output_file = self.randoms_dir / f"splits_V_arr{filename_suffix}.npy"
        np.save(output_file, V_arr)
        self.logger.info(f"Saved splits V_arr to {output_file}")
        
        output_file = self.randoms_dir / f"splits_keys{filename_suffix}.npy"
        np.save(output_file, all_keys, allow_pickle=True)
        self.logger.info(f"Saved splits keys to {output_file}")
    
    def compute_pvalue_from_randoms(
        self,
        data: np.ndarray,
        randoms: np.ndarray,
        covariance: Optional[np.ndarray] = None
    ) -> float:
        """Compute p-value by comparing data to random distribution."""
        if covariance is not None:
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
        
        data_norm = np.linalg.norm(data)
        random_norms = np.array([np.linalg.norm(r) for r in randoms])
        pvalue = np.mean(random_norms >= data_norm)
        return pvalue
    
    def load_randoms_results(
        self,
        analysis_type: str,
        filename_suffix: str = ""
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load saved randoms analysis results."""
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
    
    def compute_slope_significance(
        self,
        analysis_type: str,
        data_slopes: Dict[str, np.ndarray],
        filename_suffix: str = ""
    ) -> Dict[str, float]:
        """Compute significance of measured slopes using randoms distribution."""
        p_arr, V_arr, keys = self.load_randoms_results(analysis_type, filename_suffix)
        
        pvalues = {}
        for k, key in enumerate(keys):
            if key not in data_slopes:
                continue
            
            data_slope = data_slopes[key][0]
            randoms_slopes = p_arr[:, k, 0]
            pvalue = np.mean(np.abs(randoms_slopes) >= np.abs(data_slope))
            pvalues[key] = pvalue
        
        return pvalues
    
    def load_secondary_effects_slopes(
        self,
        use_all_bins: bool = False,
        effects: Optional[List[str]] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load pre-computed secondary effects slopes.
        
        Secondary effects include intrinsic alignment, source magnification,
        boost factor corrections, and reduced shear effects.
        
        Parameters
        ----------
        use_all_bins : bool
            If True, load allbins version of secondary effects
        effects : Optional[List[str]]
            List of effects to include. Default includes all:
            ["intrinsic_alignment", "source_magnification", "boost", 
             "boost_source", "reduced_shear"]
            
        Returns
        -------
        Optional[Dict[str, np.ndarray]]
            Dictionary mapping keys to secondary effect slope arrays,
            or None if file not found
        """
        if effects is None:
            effects = ["intrinsic_alignment", "source_magnification", "boost", 
                      "boost_source", "reduced_shear"]
        
        effects_str = "_".join(effects)
        
        if use_all_bins:
            filename = f"systematics_{effects_str}_allbins.npz"
        else:
            filename = f"systematics_{effects_str}.npz"
        
        version = self.lens_config.get_catalogue_version()
        filepath = Path(self.output_config.save_path) / version / "secondary_effects" / filename
        
        if not filepath.exists():
            self.logger.warning(f"Secondary effects file not found: {filepath}")
            return None
        
        try:
            with np.load(str(filepath), allow_pickle=True) as data:
                # The file stores a dictionary in arr_0
                slopes_dict = dict(data['arr_0'].item())
            self.logger.info(f"Loaded secondary effects slopes from {filepath}")
            return slopes_dict
        except Exception as e:
            self.logger.error(f"Error loading secondary effects: {e}")
            return None
    
    def add_secondary_effects_to_slopes(
        self,
        measured_slopes: Dict[str, np.ndarray],
        secondary_slopes: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Add secondary effects corrections to measured slopes.
        
        Parameters
        ----------
        measured_slopes : Dict[str, np.ndarray]
            Measured slope values (p[0] is slope, p[1] is intercept)
        secondary_slopes : Dict[str, np.ndarray]
            Secondary effects slope corrections
            
        Returns
        -------
        Dict[str, np.ndarray]
            Corrected slopes with secondary effects added
        """
        corrected = {}
        
        for key, p in measured_slopes.items():
            if key in secondary_slopes:
                p_secondary = secondary_slopes[key]
                # Add secondary effects slope to measured slope
                corrected[key] = np.array([p[0] + p_secondary[0], p[1] + p_secondary[1]])
            else:
                corrected[key] = p.copy()
        
        return corrected
    
    def compute_sigma_sys_for_slopes(
        self,
        amplitudes: np.ndarray,
        errors: np.ndarray,
        method: str = "chisqpdf"
    ) -> Tuple[float, List[float], Optional[Any]]:
        """
        Convenience wrapper for sigma_sys calculation.
        
        Parameters
        ----------
        amplitudes : np.ndarray
            Measured lensing amplitudes
        errors : np.ndarray
            Statistical errors on amplitudes
        method : str
            Calculation method: "chisqpdf", "bayesian", or "bayesian_brute"
            
        Returns
        -------
        Tuple[float, List[float], Optional[Any]]
            reduced_chisq, sigma_sys bounds, samples
        """
        return calculate_sigma_sys(amplitudes, errors, method=method)


def create_randoms_analyzer_from_configs(
    computation_config: ComputationConfig,
    lens_config: LensGalaxyConfig,
    source_config: SourceSurveyConfig,
    output_config: OutputConfig,
    logger: Optional[logging.Logger] = None
) -> RandomsAnalyzer:
    """Convenience function to create a RandomsAnalyzer from config objects."""
    return RandomsAnalyzer(
        computation_config, lens_config, source_config, output_config, logger
    )
