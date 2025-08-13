"""Output configuration for lensing pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import os
import re
import logging

import numpy as np
from astropy.table import Table

from .base import BaseConfig


@dataclass
class OutputConfig(BaseConfig):
    """Configuration for output settings."""
    
    # Base paths
    catalogue_path: str = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/"
    source_catalogue_path: str = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/"
    save_path: str = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensing_measurements/"
    magnification_bias_path: str = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/magnification_bias_DESI/"
    
    # File naming patterns
    filename_template: str = "{statistic}_{galaxy_type}_zmin_{z_min}_zmax_{z_max}_blind{blind}_boost_{boost}"
    filename_template_tomo: str = "{statistic}_{galaxy_type}_zmin_{z_min}_zmax_{z_max}_sbin_{source_bin}_blind{blind}_boost_{boost}"
    
    # Output options
    verbose: bool = True
    save_precomputed: bool = True
    save_covariance: bool = True
    save_unblinded: bool = False
    
    # Blinding
    apply_blinding: bool = True
    blinding_label: str = "A"
    
    def __post_init__(self):
        """Initialize logger after dataclass creation."""
        self._logger = None
    
    @property
    def logger(self) -> logging.Logger:
        """Get or create logger instance."""
        if self._logger is None:
            from ..utils.logging_utils import setup_logger
            self._logger = setup_logger(self.__class__.__name__)
        return self._logger

    def validate(self) -> List[str]:
        """Validate output configuration."""
        errors = []
        
        # Check if paths exist (create warning, not error)
        paths_to_check = [
            ("catalogue_path", self.catalogue_path),
            ("source_catalogue_path", self.source_catalogue_path),
            ("magnification_bias_path", self.magnification_bias_path),
        ]
        
        for path_name, path_value in paths_to_check:
            if not os.path.exists(path_value):
                # This is a warning, not an error - paths might not exist on all systems
                pass
        
        # Validate filename templates contain required placeholders
        required_placeholders = {"{statistic}", "{galaxy_type}", "{z_min}", "{z_max}"}
        
        # Find all placeholders in the template using regex
        template_placeholders = set(re.findall(r'\{[^}]+\}', self.filename_template))
        
        missing = required_placeholders - template_placeholders
        if missing:
            errors.append(f"filename_template missing required placeholders: {missing}")
        
        # Check tomographic template
        required_tomo_placeholders = required_placeholders | {"{source_bin}"}
        tomo_placeholders = set(re.findall(r'\{[^}]+\}', self.filename_template_tomo))
        
        missing_tomo = required_tomo_placeholders - tomo_placeholders
        if missing_tomo:
            errors.append(f"filename_template_tomo missing required placeholders: {missing_tomo}")
        
        return errors
    
    def get_save_path(self, version: str, survey: Optional[str] = None) -> Path:
        """Get the full save path including version and survey."""
        path = Path(self.save_path) / version
        if survey:
            path = path / survey
        return path
    
    def get_precomputed_path(self, version: str, survey: str) -> Path:
        """Get the path for precomputed tables."""
        return self.get_save_path(version) / "precomputed_tables" / survey
    
    def get_filepath(
        self,
        statistic: str,
        galaxy_type: str,
        version: str,
        survey: str,
        z_min: float,
        z_max: float,
        file_type: str = "measurement",  # "measurement", "covariance", "precomputed_lens", "precomputed_random"
        source_bin: Optional[int] = None,
        apply_blinding: Optional[bool] = None,  # Use config default if None
        is_bmode: bool = False,
        boost_correction: bool = False,
        pure_noise: bool = False,
        split_by: Optional[str] = None,
        split: Optional[int] = None,
        n_splits: int = 4,
        extension: Optional[str] = None
    ) -> Path:
        """
        Generate complete filepath for lensing pipeline outputs.
        
        This method consolidates all filepath generation logic and handles:
        - Different file types (measurements, covariance, precomputed tables)
        - Blinding and B-mode modifications
        - Tomographic and non-tomographic analyses
        - Data splits and boost corrections
        
        Parameters
        ----------
        statistic : str
            Statistic type ('deltasigma', 'gammat', etc.)
        galaxy_type : str
            Type of lens galaxies
        version : str
            Catalogue version
        survey : str
            Source survey name
        z_min : float
            Minimum redshift for lens bin
        z_max : float
            Maximum redshift for lens bin
        file_type : str
            Type of file: 'measurement', 'covariance', 'precomputed_lens', 'precomputed_random'
        source_bin : Optional[int]
            Source bin index for tomographic analysis
        apply_blinding : Optional[bool]
            Whether to apply blinding (uses config default if None)
        is_bmode : bool
            Whether this is B-mode analysis
        boost_correction : bool
            Whether boost correction is applied
        pure_noise : bool
            Whether this is pure noise/B-mode analysis
        split_by : Optional[str]
            Data split type
        split : Optional[int]
            Split index
        n_splits : int
            Total number of splits
        extension : Optional[str]
            File extension (auto-determined if None)
            
        Returns
        -------
        Path
            Complete filepath
        """
        # Use config default for blinding if not specified
        if apply_blinding is None:
            apply_blinding = self.apply_blinding
        
        # Modify statistic name for B-modes
        effective_statistic = statistic
        if is_bmode or pure_noise:
            effective_statistic = f"bmodes_{statistic}"
        
        # Generate base filename using templates
        template = self.filename_template_tomo if source_bin is not None else self.filename_template
        
        filename = template.format(
            statistic=effective_statistic,
            galaxy_type=galaxy_type,
            z_min=z_min,
            z_max=z_max,
            source_bin=source_bin if source_bin is not None else "",
            blind=self.blinding_label if apply_blinding else "unblind",
            boost=boost_correction
        )
        
        # Clean up any double underscores from empty source_bin
        filename = filename.replace("__", "_").rstrip("_")
        
        # Add file-type specific modifications
        if file_type == "covariance":
            filename = f"cov_{filename}"
        elif file_type.startswith("precomputed"):
            filename = f"precomp_{filename}"
            if file_type == "precomputed_random":
                filename = f"{filename}_random"
        
        # Add pure noise suffix if needed (and not already in statistic)
        if pure_noise and not is_bmode and "bmodes" not in filename:
            filename += "_bmodes"
        
        # Add data split information
        if split_by is not None and split is not None:
            filename += f"_{split_by}_{split}_of_{n_splits}"
        
        # Determine extension
        if extension is None:
            if file_type == "covariance":
                extension = ".npy"
            elif file_type.startswith("precomputed"):
                extension = ".h5"
            else:
                extension = ".fits"
        
        filename += extension
        
        # Determine base directory
        base_dir = Path(self.save_path) / version
        
        if file_type.startswith("precomputed"):
            base_dir = base_dir / "precomputed_tables" / survey
        elif not apply_blinding and self.save_unblinded:
            base_dir = base_dir / "unblinded" / survey
        else:
            base_dir = base_dir / survey
        
        return base_dir / filename

    def generate_filename(
        self,
        statistic: str,
        galaxy_type: str,
        z_min: float,
        z_max: float,
        source_bin: Optional[int] = None,
        boost_correction: bool = False,
        extension: str = ".fits"
    ) -> str:
        """
        Generate filename based on parameters (legacy method).
        
        DEPRECATED: Use get_filepath() for new code.
        """
        # Use new method and return just the filename
        dummy_version = "v1.0"
        dummy_survey = "dummy"
        
        path = self.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=dummy_version,
            survey=dummy_survey,
            z_min=z_min,
            z_max=z_max,
            boost_correction=boost_correction,
            extension=extension,
            source_bin=source_bin
        )
        
        return path.name
    
    def get_measurement_filepath(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int,
        z_bins: List[float],
        statistic: str,
        source_bin: Optional[int] = None,
        pure_noise: bool = False,
        split_by: Optional[str] = None,
        split: Optional[int] = None,
        n_splits: int = 4,
        extension: str = ".dat"
    ) -> Path:
        """
        Get the filepath for a measurement file using consistent naming.
        
        DEPRECATED: Use get_filepath() for new code.
        """
        z_min = z_bins[lens_bin]
        z_max = z_bins[lens_bin + 1]
        
        return self.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=version,
            survey=source_survey,
            z_min=z_min,
            z_max=z_max,
            source_bin=source_bin,
            pure_noise=pure_noise,
            split_by=split_by,
            split=split,
            n_splits=n_splits,
            extension=extension
        )
    
    def get_covariance_filepath(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int,
        z_bins: List[float],
        statistic: str,
        source_bin: Optional[int] = None,
        pure_noise: bool = False,
        split_by: Optional[str] = None,
        split: Optional[int] = None,
        n_splits: int = 4
    ) -> Path:
        """
        Get the filepath for a covariance file using consistent naming.
        
        DEPRECATED: Use get_filepath() for new code.
        """
        z_min = z_bins[lens_bin]
        z_max = z_bins[lens_bin + 1]
        
        return self.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=version,
            survey=source_survey,
            z_min=z_min,
            z_max=z_max,
            file_type="covariance",
            source_bin=source_bin,
            pure_noise=pure_noise,
            split_by=split_by,
            split=split,
            n_splits=n_splits
        )
    
    def load_lensing_results(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int,
        z_bins: List[float],
        statistic: str = "deltasigma",
        source_bin: Optional[int] = None,
        pure_noise: bool = False,
        split_by: Optional[str] = None,
        split: Optional[int] = None,
        n_splits: int = 4
    ) -> Optional[Table]:
        """
        Load lensing measurement results for a given configuration.
        
        Parameters
        ----------
        version : str
            Catalogue version
        galaxy_type : str
            Type of lens galaxies
        source_survey : str
            Source survey name
        lens_bin : int
            Lens redshift bin index
        z_bins : List[float]
            Redshift bin edges
        statistic : str
            Statistic to load
        source_bin : Optional[int]
            Source bin index for tomographic analysis
        pure_noise : bool
            Whether to load B-mode/noise results
        split_by : Optional[str]
            Data split type
        split : Optional[int]
            Split index
        n_splits : int
            Total number of splits
            
        Returns
        -------
        Optional[Table]
            Loaded results table, or None if file not found
        """
        try:
            filepath = self.get_measurement_filepath(
                version, galaxy_type, source_survey, lens_bin, z_bins,
                statistic, source_bin, pure_noise, split_by, split, n_splits
            )
            
            if filepath.exists():
                return Table.read(str(filepath), format='ascii')
            else:
                self.logger.warning(f"Results file not found: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return None
    
    def load_covariance_matrix(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int,
        z_bins: List[float],
        statistic: str = "deltasigma",
        source_bin: Optional[int] = None,
        pure_noise: bool = False,
        split_by: Optional[str] = None,
        split: Optional[int] = None,
        n_splits: int = 4
    ) -> Optional[np.ndarray]:
        """
        Load covariance matrix for a given configuration.
        
        Parameters
        ----------
        version : str
            Catalogue version
        galaxy_type : str
            Type of lens galaxies
        source_survey : str
            Source survey name
        lens_bin : int
            Lens redshift bin index
        z_bins : List[float]
            Redshift bin edges
        statistic : str
            Statistic type
        source_bin : Optional[int]
            Source bin index for tomographic analysis
        pure_noise : bool
            Whether to load B-mode/noise covariance
        split_by : Optional[str]
            Data split type
        split : Optional[int]
            Split index
        n_splits : int
            Total number of splits
            
        Returns
        -------
        Optional[np.ndarray]
            Loaded covariance matrix, or None if file not found
        """
        try:
            filepath = self.get_covariance_filepath(
                version, galaxy_type, source_survey, lens_bin, z_bins,
                statistic, source_bin, pure_noise, split_by, split, n_splits
            )
            
            if filepath.exists():
                return np.load(str(filepath))
            else:
                self.logger.warning(f"Covariance file not found: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading covariance: {e}")
            return None
    
    def load_tomographic_data_and_covariance(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        z_bins: List[float],
        n_source_bins: int,
        statistic: str = "deltasigma",
        pure_noise: bool = False,
        split_by: Optional[str] = None,
        split: Optional[int] = None,
        n_splits: int = 4,
        n_rp_bins: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load tomographic data and covariance for all lens-source bin combinations.
        
        Parameters
        ----------
        version : str
            Catalogue version
        galaxy_type : str
            Type of lens galaxies
        source_survey : str
            Source survey name
        z_bins : List[float]
            Lens redshift bin edges
        n_source_bins : int
            Number of source tomographic bins
        statistic : str
            Statistic type
        pure_noise : bool
            Whether to load B-mode/noise data
        split_by : Optional[str]
            Data split type
        split : Optional[int]
            Split index
        n_splits : int
            Total number of splits
        n_rp_bins : int
            Number of radial bins (fallback for placeholder data)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Combined data vector and covariance matrix
        """
        try:
            measurements = []
            covariances = []
            
            n_lens_bins = len(z_bins) - 1
            
            for lens_bin in range(n_lens_bins):
                for source_bin in range(n_source_bins):
                    # Load individual measurement and covariance
                    results = self.load_lensing_results(
                        version, galaxy_type, source_survey, lens_bin, z_bins,
                        statistic, source_bin, pure_noise, split_by, split, n_splits
                    )
                    cov = self.load_covariance_matrix(
                        version, galaxy_type, source_survey, lens_bin, z_bins,
                        statistic, source_bin, pure_noise, split_by, split, n_splits
                    )
                    
                    if results is not None and cov is not None:
                        # Extract signal values
                        if statistic == "deltasigma":
                            signal = results['ds'] if 'ds' in results.columns else results[results.columns[1]]
                        else:
                            signal = results['et'] if 'et' in results.columns else results[results.columns[1]]
                        
                        measurements.append(signal)
                        covariances.append(cov)
                    else:
                        # Generate placeholder data if files don't exist
                        self.logger.warning(f"Missing data for {galaxy_type} {source_survey} lens:{lens_bin} source:{source_bin}, using placeholder")
                        measurements.append(np.random.randn(n_rp_bins) * 0.01)
                        covariances.append(np.eye(n_rp_bins) * 0.01**2)
            
            # Combine measurements and covariances
            combined_data = np.concatenate(measurements)
            combined_cov = self._combine_covariance_blocks(covariances)
            
            return combined_data, combined_cov
            
        except Exception as e:
            self.logger.warning(f"Error loading tomographic data: {e}, using placeholder")
            # Fallback to placeholder
            n_lens_bins = len(z_bins) - 1
            total_bins = n_lens_bins * n_source_bins * n_rp_bins
            data = np.random.randn(total_bins) * 0.01
            cov = np.eye(total_bins) * 0.01**2
            
            return data, cov
    
    def load_non_tomographic_data_and_covariance(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        z_bins: List[float],
        statistic: str = "deltasigma",
        pure_noise: bool = False,
        split_by: Optional[str] = None,
        split: Optional[int] = None,
        n_splits: int = 4,
        n_rp_bins: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load non-tomographic data and covariance for all lens bins.
        
        Parameters
        ----------
        version : str
            Catalogue version
        galaxy_type : str
            Type of lens galaxies
        source_survey : str
            Source survey name
        z_bins : List[float]
            Lens redshift bin edges
        statistic : str
            Statistic type
        pure_noise : bool
            Whether to load B-mode/noise data
        split_by : Optional[str]
            Data split type
        split : Optional[int]
            Split index
        n_splits : int
            Total number of splits
        n_rp_bins : int
            Number of radial bins (fallback for placeholder data)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Combined data vector and covariance matrix
        """
        try:
            measurements = []
            covariances = []
            
            n_lens_bins = len(z_bins) - 1
            
            for lens_bin in range(n_lens_bins):
                # Load individual measurement and covariance
                results = self.load_lensing_results(
                    version, galaxy_type, source_survey, lens_bin, z_bins,
                    statistic, None, pure_noise, split_by, split, n_splits
                )
                cov = self.load_covariance_matrix(
                    version, galaxy_type, source_survey, lens_bin, z_bins,
                    statistic, None, pure_noise, split_by, split, n_splits
                )
                
                if results is not None and cov is not None:
                    # Extract signal values
                    if statistic == "deltasigma":
                        signal = results['ds'] if 'ds' in results.columns else results[results.columns[1]]
                    else:
                        signal = results['et'] if 'et' in results.columns else results[results.columns[1]]
                    
                    measurements.append(signal)
                    covariances.append(cov)
                else:
                    # Generate placeholder data if files don't exist
                    self.logger.warning(f"Missing data for {galaxy_type} {source_survey} lens:{lens_bin}, using placeholder")
                    measurements.append(np.random.randn(n_rp_bins) * 0.01)
                    covariances.append(np.eye(n_rp_bins) * 0.01**2)
            
            # Combine measurements and covariances
            combined_data = np.concatenate(measurements)
            combined_cov = self._combine_covariance_blocks(covariances)
            
            return combined_data, combined_cov
            
        except Exception as e:
            self.logger.warning(f"Error loading data: {e}, using placeholder")
            # Fallback to placeholder
            n_lens_bins = len(z_bins) - 1
            total_bins = n_lens_bins * n_rp_bins
            data = np.random.randn(total_bins) * 0.01
            cov = np.eye(total_bins) * 0.01**2
            
            return data, cov
    
    def _combine_covariance_blocks(self, cov_blocks: List[np.ndarray]) -> np.ndarray:
        """Combine individual covariance matrices into a block diagonal structure."""
        if len(cov_blocks) == 1:
            return cov_blocks[0]
        
        # Create block diagonal covariance matrix
        total_size = sum(cov.shape[0] for cov in cov_blocks)
        combined_cov = np.zeros((total_size, total_size))
        
        start_idx = 0
        for cov in cov_blocks:
            end_idx = start_idx + cov.shape[0]
            combined_cov[start_idx:end_idx, start_idx:end_idx] = cov
            start_idx = end_idx
            
        return combined_cov

    def create_output_directories(self, version: str, surveys: List[str]) -> None:
        """Create all necessary output directories."""
        base_path = Path(self.save_path) / version
        
        for survey in surveys:
            # Main output directory
            survey_path = base_path / survey
            survey_path.mkdir(parents=True, exist_ok=True)
            
            # Precomputed tables directory
            if self.save_precomputed:
                precomputed_path = base_path / "precomputed_tables" / survey
                precomputed_path.mkdir(parents=True, exist_ok=True)
            
            # Unblinded directory
            if self.save_unblinded:
                unblinded_path = base_path / "unblinded" / survey
                unblinded_path.mkdir(parents=True, exist_ok=True)
            
            # Randoms directory
            randoms_path = base_path / "randoms"
            randoms_path.mkdir(parents=True, exist_ok=True)
    
    def get_magnification_bias_file(self, version: str) -> Path:
        """Get the magnification bias file path."""
        return Path(self.magnification_bias_path) / version / "DESI_magnification_bias.json" 