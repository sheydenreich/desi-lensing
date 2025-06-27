"""Base computation class for lensing statistics."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from astropy.table import Table
import numpy as np
import logging

from ..config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig


class BaseComputation(ABC):
    """Base class for all lensing computations."""
    
    def __init__(
        self,
        computation_config: ComputationConfig,
        lens_config: LensGalaxyConfig,
        source_config: SourceSurveyConfig,
        output_config: OutputConfig,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize base computation."""
        self.computation_config = computation_config
        self.lens_config = lens_config
        self.source_config = source_config
        self.output_config = output_config
        
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Initialize cosmology
        self.cosmology = self.computation_config.get_cosmology()
        
        # Will be set during pipeline execution
        self.table_l: Optional[Table] = None
        self.table_r: Optional[Table] = None
        self.table_s: Optional[Table] = None
        
    @abstractmethod
    def precompute(
        self, 
        table_l: Table, 
        table_r: Optional[Table], 
        table_s: Table,
        **kwargs
    ) -> Tuple[Table, Optional[Table]]:
        """Precompute the lensing signal."""
        pass
    
    @abstractmethod
    def compute_statistic(
        self, 
        table_l: Table, 
        table_r: Optional[Table],
        **kwargs
    ) -> Tuple[Table, np.ndarray]:
        """Compute the lensing statistic and covariance."""
        pass
    
    def setup_bins(self) -> np.ndarray:
        """Setup binning for the computation."""
        if self.statistic_name == "deltasigma":
            return self.computation_config.get_rp_bins()
        elif self.statistic_name == "gammat":
            return self.computation_config.get_theta_bins()
        else:
            raise ValueError(f"Unknown statistic: {self.statistic_name}")
    
    def get_weighting(self) -> int:
        """Get weighting parameter for dsigma.precompute."""
        if self.statistic_name == "deltasigma":
            return -2
        elif self.statistic_name == "gammat":
            return 0
        else:
            raise ValueError(f"Unknown statistic: {self.statistic_name}")
    
    def validate_tables(self, table_l: Table, table_r: Optional[Table], table_s: Table) -> None:
        """Validate input tables."""
        required_lens_cols = ["ra", "dec", "z", "w_sys"]
        for col in required_lens_cols:
            if col not in table_l.colnames:
                raise ValueError(f"Lens table missing required column: {col}")
        
        if table_r is not None:
            for col in required_lens_cols:
                if col not in table_r.colnames:
                    raise ValueError(f"Random table missing required column: {col}")
        
        required_source_cols = ["ra", "dec", "e_1", "e_2", "w"]
        for col in required_source_cols:
            if col not in table_s.colnames:
                raise ValueError(f"Source table missing required column: {col}")
    
    def log_computation_info(self, lens_bin: int, source_bin: Optional[int] = None) -> None:
        """Log information about current computation."""
        n_lenses = len(self.table_l) if self.table_l is not None else 0
        n_randoms = len(self.table_r) if self.table_r is not None else 0
        n_sources = len(self.table_s) if self.table_s is not None else 0
        
        if source_bin is not None:
            self.logger.info(
                f"Computing {self.statistic_name} for lens bin {lens_bin+1}, "
                f"source bin {source_bin+1}: {n_lenses} lenses, {n_randoms} randoms, "
                f"{n_sources} sources"
            )
        else:
            self.logger.info(
                f"Computing {self.statistic_name} for lens bin {lens_bin+1}: "
                f"{n_lenses} lenses, {n_randoms} randoms, {n_sources} sources"
            )
    
    @property
    @abstractmethod
    def statistic_name(self) -> str:
        """Name of the statistic being computed."""
        pass
    
    def check_table_overlap(self, table_l: Table, table_s: Table) -> bool:
        """Check if there's sufficient overlap between lens and source samples."""
        if len(table_l) == 0 or len(table_s) == 0:
            return False
        
        # Simple redshift overlap check
        z_l_max = np.max(table_l['z'])
        z_s_min = np.min(table_s.get('z', [0]))  # Some source tables don't have z
        
        # For now, just check that we have some data
        return True  # More sophisticated checks can be added 

    def get_output_filename_prefix(self) -> str:
        """Get the output filename prefix including bmodes if applicable."""
        if self.computation_config.bmodes:
            return f"bmodes_{self.statistic_name}"
        else:
            return self.statistic_name 