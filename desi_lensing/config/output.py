"""Output configuration for lensing pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
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
    
    # Precomputed table naming patterns (without blind/boost since these are applied later)
    precomputed_filename_template: str = "{statistic}_{galaxy_type}_zmin_{z_min}_zmax_{z_max}"
    precomputed_filename_template_tomo: str = "{statistic}_{galaxy_type}_zmin_{z_min}_zmax_{z_max}_sbin_{source_bin}"
    
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
        
        # Validate precomputed templates (same requirements as regular templates, but without blind/boost)
        precomp_placeholders = set(re.findall(r'\{[^}]+\}', self.precomputed_filename_template))
        missing_precomp = required_placeholders - precomp_placeholders
        if missing_precomp:
            errors.append(f"precomputed_filename_template missing required placeholders: {missing_precomp}")
        
        precomp_tomo_placeholders = set(re.findall(r'\{[^}]+\}', self.precomputed_filename_template_tomo))
        missing_precomp_tomo = required_tomo_placeholders - precomp_tomo_placeholders
        if missing_precomp_tomo:
            errors.append(f"precomputed_filename_template_tomo missing required placeholders: {missing_precomp_tomo}")
        
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
        # Select appropriate template based on file type
        if file_type.startswith("precomputed"):
            template = self.precomputed_filename_template_tomo if source_bin is not None else self.precomputed_filename_template
        else:
            template = self.filename_template_tomo if source_bin is not None else self.filename_template
        
        # Format filename with appropriate parameters
        if file_type.startswith("precomputed"):
            filename = template.format(
                statistic=effective_statistic,
                galaxy_type=galaxy_type,
                z_min=z_min,
                z_max=z_max,
                source_bin=source_bin if source_bin is not None else ""
            )
        else:
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

    def get_measurement_path_simple(
        self,
        statistic: str,
        galaxy_type: str,
        version: str,
        survey: str,
        z_min: float,
        z_max: float,
        source_bin: Optional[int] = None,
        is_bmode: bool = False
    ) -> Path:
        """
        Get filepath for a measurement file (simplified API).
        
        This is a user-friendly wrapper around get_filepath() with only the
        most commonly needed parameters for measurements.
        
        Parameters
        ----------
        statistic : str
            Statistic type ('deltasigma', 'gammat')
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
        source_bin : Optional[int]
            Source bin index for tomographic analysis
        is_bmode : bool
            Whether this is B-mode analysis
            
        Returns
        -------
        Path
            Complete filepath for measurement
        """
        return self.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=version,
            survey=survey,
            z_min=z_min,
            z_max=z_max,
            file_type="measurement",
            source_bin=source_bin,
            is_bmode=is_bmode
        )
    
    def get_covariance_path_simple(
        self,
        statistic: str,
        galaxy_type: str,
        version: str,
        survey: str,
        z_min: float,
        z_max: float,
        source_bin: Optional[int] = None,
        is_bmode: bool = False
    ) -> Path:
        """
        Get filepath for a covariance matrix file (simplified API).
        
        Parameters
        ----------
        statistic : str
            Statistic type ('deltasigma', 'gammat')
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
        source_bin : Optional[int]
            Source bin index for tomographic analysis
        is_bmode : bool
            Whether this is B-mode analysis
            
        Returns
        -------
        Path
            Complete filepath for covariance matrix
        """
        return self.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=version,
            survey=survey,
            z_min=z_min,
            z_max=z_max,
            file_type="covariance",
            source_bin=source_bin,
            is_bmode=is_bmode
        )
    
    def get_precomputed_lens_path(
        self,
        statistic: str,
        galaxy_type: str,
        version: str,
        survey: str,
        z_min: float,
        z_max: float,
        source_bin: Optional[int] = None,
        is_bmode: bool = False
    ) -> Path:
        """
        Get filepath for precomputed lens table (simplified API).
        
        Parameters
        ----------
        statistic : str
            Statistic type ('deltasigma', 'gammat')
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
        source_bin : Optional[int]
            Source bin index for tomographic analysis
        is_bmode : bool
            Whether this is B-mode analysis
            
        Returns
        -------
        Path
            Complete filepath for precomputed lens table
        """
        return self.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=version,
            survey=survey,
            z_min=z_min,
            z_max=z_max,
            file_type="precomputed_lens",
            source_bin=source_bin,
            is_bmode=is_bmode
        )
    
    def get_precomputed_random_path(
        self,
        statistic: str,
        galaxy_type: str,
        version: str,
        survey: str,
        z_min: float,
        z_max: float,
        source_bin: Optional[int] = None,
        is_bmode: bool = False
    ) -> Path:
        """
        Get filepath for precomputed random table (simplified API).
        
        Parameters
        ----------
        statistic : str
            Statistic type ('deltasigma', 'gammat')
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
        source_bin : Optional[int]
            Source bin index for tomographic analysis
        is_bmode : bool
            Whether this is B-mode analysis
            
        Returns
        -------
        Path
            Complete filepath for precomputed random table
        """
        return self.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=version,
            survey=survey,
            z_min=z_min,
            z_max=z_max,
            file_type="precomputed_random",
            source_bin=source_bin,
            is_bmode=is_bmode
        )

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
        
        DEPRECATED: Use get_measurement_path_simple() or get_filepath() for new code.
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
        extension: str = None
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
                return Table.read(str(filepath))
            else:
                fallback_path = Path(str(filepath).replace("magnitude_cut/", ""))
                if galaxy_type == "LRG" and fallback_path.exists():
                    self.logger.info("Loading LRG results without magnitude cuts")
                    return Table.read(str(fallback_path))
                else:
                    self.logger.warning(f"Results file not found: {filepath}")
                    return None
                
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_theory_covariance(
        self,
        galaxy_type: str,
        source_survey: str,
        statistic: str = "deltasigma",
        pure_noise: bool = False,
        theory_cov_path: str = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/model_inputs_desiy3",
        z_bins: Optional[List[float]] = None
    ) -> Optional[np.ndarray]:
        """
        Load theory covariance matrix from Chris Hirata's files.
        
        Theory covariances are only available for tomographic measurements,
        not non-tomographic. They combine all lens and source bins.
        
        Parameters
        ----------
        galaxy_type : str
            Type of lens galaxies (e.g., 'BGS_BRIGHT', 'LRG')
        source_survey : str
            Source survey name (e.g., 'KiDS', 'DES', 'HSCY3', 'DECADE')
        statistic : str
            Statistic type ('deltasigma' or 'gammat')
        pure_noise : bool
            Whether to load B-mode/noise covariance
        theory_cov_path : str
            Path to theory covariance files
        z_bins : Optional[List[float]]
            Redshift bin edges. For BGS, if z_bins == [0.1, 0.4] (single k-bin),
            uses 'bkp1' covariance files instead of standard 'bgs' files.
            
        Returns
        -------
        Optional[np.ndarray]
            Theory covariance matrix, or None if not available
        """
        # Map galaxy types
        if galaxy_type.upper() in ["BGS", "BGS_BRIGHT"]:
            # Check if using single k-bin (z_bins = [0.1, 0.4]) vs standard 3-bin
            # Use robust check that works with lists and numpy arrays
            is_kbin = (
                z_bins is not None 
                and len(z_bins) == 2 
                and np.allclose(z_bins, [0.1, 0.4], atol=1e-6)
            )
            if is_kbin:
                fgal = "bkp1"
            else:
                fgal = "bgs"
        elif galaxy_type.upper() in ["LRG"]:
            fgal = "lrg"
        else:
            self.logger.warning(f"Unknown galaxy type {galaxy_type} for theory covariance")
            return None
        
        # Map source surveys (case-insensitive)
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
        elif source_lower in ["decade_ngc", "decade_sgc"]:
            # No theory covariance available for NGC/SGC splits
            self.logger.info(f"No theory covariance available for {source_survey}, will use jackknife")
            return None
        elif source_lower in ["all", "all_y3"]:
            # Combined surveys for Y3
            fsurv = "kids1000desy3hscy3decade_"
        else:
            self.logger.warning(f"Unknown source survey {source_survey} for theory covariance")
            return None
        
        # Map statistics with pure_noise flag
        if statistic == "deltasigma":
            if pure_noise:
                fstat = "dx"
            else:
                fstat = "ds"
            suffix = "_pzwei"
        elif statistic == "gammat":
            if pure_noise:
                fstat = "gx"
            else:
                fstat = "gt"
            suffix = ""
        else:
            self.logger.warning(f"Unknown statistic {statistic} for theory covariance")
            return None
        
        # Build filepath
        filename = f"{fstat}covcorr_{fsurv}desiy3{fgal}{suffix}.dat"
        filepath = Path(theory_cov_path) / filename
        
        try:
            if filepath.exists():
                # Load covariance: format has extra columns, we need the last one reshaped
                data = np.loadtxt(str(filepath), skiprows=1)
                
                # The data format has multiple columns; the last column contains the covariance values
                # We need to determine the size and reshape
                cov_values = data[:, -1]
                n = int(np.sqrt(len(cov_values)))
                
                if n * n != len(cov_values):
                    self.logger.error(f"Covariance matrix from {filepath} is not square: {len(cov_values)} elements")
                    return None
                
                cov_matrix = cov_values.reshape(n, n)
                self.logger.info(f"Loaded theory covariance from {filepath} (shape: {cov_matrix.shape})")
                return cov_matrix
            else:
                self.logger.info(f"Theory covariance file not found: {filepath}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error loading theory covariance from {filepath}: {e}")
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
                return np.loadtxt(str(filepath))
            else:
                fallback_path = Path(str(filepath).replace("magnitude_cut/", ""))
                if galaxy_type == "LRG" and fallback_path.exists():
                    self.logger.info("Loading LRG covariance without magnitude cuts")
                    return np.loadtxt(str(fallback_path))
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
        n_rp_bins: int = 15,
        use_theory_covariance: bool = True
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
        use_theory_covariance : bool
            Whether to try loading theory covariance (default True)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Combined data vector and covariance matrix
        """
        try:
            measurements = []
            covariances = []
            
            n_lens_bins = len(z_bins) - 1
            
            # Try to load theory covariance first (only for tomographic, not splits)
            theory_cov = None
            if use_theory_covariance and split_by is None:
                theory_cov = self.load_theory_covariance(
                    galaxy_type, source_survey, statistic, pure_noise, z_bins=z_bins
                )
            
            for lens_bin in range(n_lens_bins):
                for source_bin in range(n_source_bins):
                    # Load individual measurement
                    results = self.load_lensing_results(
                        version, galaxy_type, source_survey, lens_bin, z_bins,
                        statistic, source_bin, pure_noise, split_by, split, n_splits
                    )
                    
                    # Load covariance (jackknife if theory not available)
                    if theory_cov is None:
                        cov = self.load_covariance_matrix(
                            version, galaxy_type, source_survey, lens_bin, z_bins,
                            statistic, source_bin, pure_noise, split_by, split, n_splits
                        )
                        covariances.append(cov if cov is not None else np.eye(n_rp_bins) * 0.01**2)
                    
                    if results is not None:
                        # Extract signal values
                        if statistic == "deltasigma":
                            signal = results['ds'] if 'ds' in results.columns else results[results.columns[1]]
                        else:
                            signal = results['et'] if 'et' in results.columns else results[results.columns[1]]
                        
                        measurements.append(signal)
                    else:
                        # Generate placeholder data if files don't exist
                        self.logger.warning(f"Missing data for {galaxy_type} {source_survey} lens:{lens_bin} source:{source_bin}, using placeholder")
                        measurements.append(np.random.randn(n_rp_bins) * 0.01)
            
            # Combine measurements and covariances
            combined_data = np.concatenate(measurements)
            
            if theory_cov is not None:
                # Use theory covariance (already combined for all bins)
                combined_cov = theory_cov
            else:
                # Use jackknife covariances (need to combine blocks)
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
    
    def get_splits_directory(
        self,
        version: str,
        source_survey: str
    ) -> Path:
        """
        Get the directory containing split results.
        
        Parameters
        ----------
        version : str
            Catalogue version
        source_survey : str
            Source survey name
            
        Returns
        -------
        Path
            Path to splits directory
        """
        return Path(self.save_path) / version / "splits" / source_survey
    
    def get_splits_filepath(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int,
        z_bins: List[float],
        statistic: str,
        split_by: str,
        split: int,
        n_splits: int,
        boost_correction: bool = False,
        file_type: str = "measurement"
    ) -> Path:
        """
        Get the filepath for split analysis results.
        
        This generates paths consistent with SplitsAnalyzer._save_split_results().
        
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
            Statistic type ('deltasigma' or 'gammat')
        split_by : str
            Property used for splitting (e.g., 'NTILE', 'ra', 'dec')
        split : int
            Split index (0 to n_splits-1)
        n_splits : int
            Total number of splits
        boost_correction : bool
            Whether boost correction was applied
        file_type : str
            Type of file: 'measurement', 'covariance', 'split_value'
            
        Returns
        -------
        Path
            Complete filepath
        """
        z_min = z_bins[lens_bin]
        z_max = z_bins[lens_bin + 1]
        
        # Base filename format (matches SplitsAnalyzer._save_split_results)
        filename_base = (
            f"{statistic}_{galaxy_type}_zmin_{z_min}_zmax_{z_max}_"
            f"blindA_boost_{boost_correction}_split_{split_by}_{split}_of_{n_splits}"
        )
        
        if file_type == "measurement":
            filename = f"{filename_base}.fits"
        elif file_type == "covariance":
            filename = f"covariance_{filename_base}.dat"
        elif file_type == "split_value":
            filename = f"split_value_{filename_base}.txt"
        else:
            raise ValueError(f"Unknown file_type: {file_type}")
        
        splits_dir = self.get_splits_directory(version, source_survey)
        return splits_dir / filename
    
    def load_split_results(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int,
        z_bins: List[float],
        statistic: str,
        split_by: str,
        split: int,
        n_splits: int,
        boost_correction: bool = False
    ) -> Optional[Table]:
        """
        Load split analysis results for a given configuration.
        
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
        split_by : str
            Property used for splitting
        split : int
            Split index
        n_splits : int
            Total number of splits
        boost_correction : bool
            Whether boost correction was applied
            
        Returns
        -------
        Optional[Table]
            Loaded results table, or None if file not found
        """
        try:
            filepath = self.get_splits_filepath(
                version, galaxy_type, source_survey, lens_bin, z_bins,
                statistic, split_by, split, n_splits, boost_correction,
                file_type="measurement"
            )
            
            if filepath.exists():
                return Table.read(str(filepath))
            else:
                self.logger.warning(f"Split results file not found: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading split results: {e}")
            return None
    
    def load_split_covariance(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int,
        z_bins: List[float],
        statistic: str,
        split_by: str,
        split: int,
        n_splits: int,
        boost_correction: bool = False
    ) -> Optional[np.ndarray]:
        """
        Load covariance matrix for split analysis.
        
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
        split_by : str
            Property used for splitting
        split : int
            Split index
        n_splits : int
            Total number of splits
        boost_correction : bool
            Whether boost correction was applied
            
        Returns
        -------
        Optional[np.ndarray]
            Loaded covariance matrix, or None if file not found
        """
        try:
            filepath = self.get_splits_filepath(
                version, galaxy_type, source_survey, lens_bin, z_bins,
                statistic, split_by, split, n_splits, boost_correction,
                file_type="covariance"
            )
            
            if filepath.exists():
                return np.loadtxt(str(filepath))
            else:
                self.logger.warning(f"Split covariance file not found: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading split covariance: {e}")
            return None
    
    def load_split_value(
        self,
        version: str,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int,
        z_bins: List[float],
        statistic: str,
        split_by: str,
        split: int,
        n_splits: int,
        boost_correction: bool = False
    ) -> Optional[Tuple[float, float, float]]:
        """
        Load split value statistics (mean, min, max) for a given split.
        
        The split value file contains the weighted average, minimum, and maximum
        of the split property for galaxies in this split.
        
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
        split_by : str
            Property used for splitting
        split : int
            Split index
        n_splits : int
            Total number of splits
        boost_correction : bool
            Whether boost correction was applied
            
        Returns
        -------
        Optional[Tuple[float, float, float]]
            Tuple of (mean_value, min_value, max_value), or None if file not found
        """
        try:
            filepath = self.get_splits_filepath(
                version, galaxy_type, source_survey, lens_bin, z_bins,
                statistic, split_by, split, n_splits, boost_correction,
                file_type="split_value"
            )
            
            if filepath.exists():
                values = np.loadtxt(str(filepath))
                # File format: [mean, min, max]
                if len(values) >= 3:
                    return (float(values[0]), float(values[1]), float(values[2]))
                elif len(values) >= 1:
                    # Fallback if only mean is stored
                    return (float(values[0]), float(values[0]), float(values[0]))
                else:
                    self.logger.warning(f"Empty split value file: {filepath}")
                    return None
            else:
                self.logger.warning(f"Split value file not found: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading split value: {e}")
            return None
    
    def load_reference_datavector(
        self,
        galaxy_type: str,
        lens_bin: int,
        reference_path: str = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/model_inputs_desiy3",
        datavector_type: str = "abacus",
        rp: Optional[np.ndarray] = None,
        z_lens: Optional[float] = None,
        config: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """
        Load reference data vector for amplitude normalization.
        
        Reference data vectors are theory predictions used to normalize
        lensing amplitude measurements, enabling comparison across different
        lens bins and scale cuts.
        
        Parameters
        ----------
        galaxy_type : str
            Type of lens galaxies ('BGS_BRIGHT' or 'LRG')
        lens_bin : int
            Lens redshift bin index
        reference_path : str
            Path to reference data vector files
        datavector_type : str
            Type of reference data vector:
            - 'default' / 'abacus': Abacus HOD model predictions
            - 'buzzard': Buzzard simulation predictions
            - 'darkemu': Dark emulator predictions (requires rp, z_lens)
            - 'alexie_hod': HOD model from Saito et al. 2016
            - 'tabcorr': Tabulated correlation predictions
        rp : Optional[np.ndarray]
            Projected radii (Mpc/h) for darkemu/alexie_hod
        z_lens : Optional[float]
            Lens redshift for darkemu/alexie_hod
        config : Optional[Any]
            Config object for Mmin lookup (darkemu only)
            
        Returns
        -------
        Optional[np.ndarray]
            Reference data vector, or None if file not found
        """
        try:
            # Map galaxy type to file naming convention
            galaxy_short = galaxy_type[:3].upper()  # 'BGS' or 'LRG'
            
            if datavector_type in ["default", "abacus"]:
                filepath = Path(reference_path) / "reference_datavectors" / "abacus" / f"ds_{galaxy_short}_l{lens_bin}.npy"
                if filepath.exists():
                    ref_dv = np.load(str(filepath))
                    self.logger.info(f"Loaded reference datavector from {filepath} (shape: {ref_dv.shape})")
                    return ref_dv
                    
            elif datavector_type == "buzzard":
                filepath = Path(reference_path) / "reference_datavectors" / f"deltasigma_{galaxy_type}_l{lens_bin}_mean_datavector.npy"
                if filepath.exists():
                    ref_dv = np.load(str(filepath))
                    self.logger.info(f"Loaded reference datavector from {filepath} (shape: {ref_dv.shape})")
                    return ref_dv
                    
            elif datavector_type == "darkemu":
                # Use dark_emulator to generate reference datavector
                if rp is None or z_lens is None:
                    self.logger.warning("darkemu requires rp and z_lens parameters")
                    return None
                return self._generate_darkemu_datavector(rp, z_lens, galaxy_type, lens_bin, config)
                
            elif datavector_type == "alexie_hod":
                # Load HOD model from Saito et al. 2016
                filepath = Path(reference_path) / "reference_datavectors" / "BOSS" / "hod_model_saito2016_mdpl2.txt"
                if filepath.exists() and rp is not None and z_lens is not None:
                    dat = np.loadtxt(str(filepath))
                    # Convert units: rp is comoving, need to apply h and (1+z) factors
                    h = 0.7
                    rp_interp = dat[:, 0] * h / (1 + z_lens)
                    ds_interp = dat[:, 1] / h
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(rp_interp, ds_interp, fill_value=np.nan, bounds_error=False)
                    ref_dv = interp_func(rp)
                    self.logger.info(f"Loaded alexie_hod reference datavector from {filepath}")
                    return ref_dv
                    
            elif datavector_type == "tabcorr":
                filepath = Path(reference_path) / f"lensing_measurements/hod_params/{galaxy_type}_{lens_bin}_ds.npy"
                if filepath.exists():
                    ref_dv = np.load(str(filepath))
                    self.logger.info(f"Loaded tabcorr reference datavector from {filepath}")
                    return ref_dv
            else:
                self.logger.warning(f"Unknown datavector_type: {datavector_type}. "
                                   f"Valid options: default, abacus, buzzard, darkemu, alexie_hod, tabcorr")
                return None
            
            self.logger.warning(f"Reference datavector not found for type {datavector_type}")
            return None
                
        except Exception as e:
            self.logger.error(f"Error loading reference datavector: {e}")
            return None
    
    def _generate_darkemu_datavector(
        self,
        rp: np.ndarray,
        z_lens: float,
        galaxy_type: str,
        lens_bin: int,
        config: Optional[Any] = None,
        Mmin: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Generate reference datavector using dark_emulator.
        
        Parameters
        ----------
        rp : np.ndarray
            Projected radii (Mpc/h)
        z_lens : float
            Lens redshift
        galaxy_type : str
            Galaxy type for Mmin lookup
        lens_bin : int
            Lens bin for Mmin lookup
        config : Optional[Any]
            Config object for Mmin lookup
        Mmin : Optional[float]
            Minimum halo mass (if not provided, use config or default)
            
        Returns
        -------
        Optional[np.ndarray]
            DeltaSigma reference datavector
        """
        try:
            import sys
            sys.path.insert(0, "/global/homes/s/sven/code/dark_emulator_public/")
            from dark_emulator import darkemu
        except ImportError:
            self.logger.warning("dark_emulator not available for reference datavector generation")
            return None
        
        # Cosmology parameters (Planck18)
        omb = 0.0224
        omc = 0.120
        omde = 0.6847
        lnAs = 3.045
        ns = 0.965
        w = -1.
        
        cparam = np.array([omb, omc, omde, lnAs, ns, w])
        
        emu = darkemu.base_class()
        emu.set_cosmology(cparam)
        
        # Calculate h
        omnu = 0.00064
        omm = 1 - omde
        h = np.sqrt((omb + omc + omnu) / omm)
        
        # Get Mmin
        if Mmin is None:
            Mmin = 5e12  # Default value
            # Could load from config if available
        
        dsigma = emu.get_DeltaSigma_massthreshold(rp / h, Mmin, z_lens)
        self.logger.info(f"Generated darkemu reference datavector (Mmin={Mmin:.2e}, z={z_lens})")
        return dsigma