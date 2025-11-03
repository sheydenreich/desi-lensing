"""Computation configuration for lensing pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18, WMAP9, w0waCDM, Cosmology

from .base import BaseConfig


@dataclass
class ComputationConfig(BaseConfig):
    """Configuration for lensing computations."""
    
    # Statistics to compute
    statistics: List[Literal["deltasigma", "gammat"]] = field(default_factory=lambda: ["deltasigma"])
    
    # B-modes flag - when True, compute with rotated source galaxies (45 degrees)
    bmodes: bool = False
    
    # Cosmology settings
    cosmology: Literal["planck18", "wmap9", "wcdm"] = "planck18"
    h0: float = 100.0  # For custom cosmology
    om0: Optional[float] = None  # For custom cosmology
    w0: Optional[float] = None  # For wCDM
    wa: Optional[float] = None  # For wCDM
    
    # Computation settings
    n_jobs: int = 0  # 0 means use all available CPUs
    comoving: bool = True
    lens_source_cut: Optional[float] = None  # None for no cut
    n_jackknife_fields: int = 100
    
    # GPU computation settings
    use_gpu: bool = True  # Use GPU for computation by default
    force_shared: Optional[bool] = None  # None means auto-decide based on use_gpu
    
    # Binning configuration
    # Delta Sigma (projected radii in Mpc/h)
    rp_min: float = 0.08
    rp_max: float = 80.0
    n_rp_bins: int = 15
    
    # Gamma_t (angular bins in arcmin)
    theta_min: float = 0.3
    theta_max: float = 300.0
    n_theta_bins: int = 15
    
    # Binning type
    binning: Literal["log", "linear"] = "log"
    
    # Tomography
    tomography: bool = True
    
    def __post_init__(self):
        """Post-initialization to set defaults based on GPU usage."""
        if self.force_shared is None:
            # Set force_shared based on use_gpu
            self.force_shared = self.use_gpu
        
        # Set n_jobs defaults based on GPU usage if not explicitly set
        if self.n_jobs == 0:
            if self.use_gpu:
                self.n_jobs = 4
            else:
                self.n_jobs = 0  # Will be resolved to cpu_count() in computation.py
    
    def validate(self) -> List[str]:
        """Validate computation configuration."""
        errors = []
        
        # Validate statistics
        valid_stats = {"deltasigma", "gammat"}
        for stat in self.statistics:
            if stat not in valid_stats:
                errors.append(f"Invalid statistic '{stat}'. Must be one of {valid_stats}")
        
        # Validate cosmology
        valid_cosmologies = {"planck18", "wmap9", "wcdm"}
        if self.cosmology not in valid_cosmologies:
            errors.append(f"Invalid cosmology '{self.cosmology}'. Must be one of {valid_cosmologies}")
        
        # Validate binning parameters
        if self.rp_min <= 0:
            errors.append("rp_min must be positive")
        if self.rp_max <= self.rp_min:
            errors.append("rp_max must be greater than rp_min")
        if self.n_rp_bins < 1:
            errors.append("n_rp_bins must be at least 1")
            
        if self.theta_min <= 0:
            errors.append("theta_min must be positive")
        if self.theta_max <= self.theta_min:
            errors.append("theta_max must be greater than theta_min")
        if self.n_theta_bins < 1:
            errors.append("n_theta_bins must be at least 1")
        
        # Validate lens_source_cut
        if self.lens_source_cut is not None and self.lens_source_cut < 0:
            errors.append("lens_source_cut must be non-negative or None")
        
        # Validate n_jobs
        if self.n_jobs < 0:
            errors.append("n_jobs must be non-negative")
        
        # Validate n_jackknife_fields
        if self.n_jackknife_fields < 2:
            errors.append("n_jackknife_fields must be at least 2")
        
        return errors
    
    def get_cosmology(self) -> Cosmology:
        """Get the astropy cosmology object."""
        print("H0", self.h0, "cosmology", self.cosmology)
        if self.cosmology == "planck18":
            return Planck18.clone(H0=self.h0)
        elif self.cosmology == "wmap9":
            om0 = self.om0 if self.om0 is not None else 0.22
            return WMAP9.clone(H0=self.h0, Om0=om0)
        elif self.cosmology == "wcdm":
            w0 = self.w0 if self.w0 is not None else -0.48
            wa = self.wa if self.wa is not None else -1.34
            om0 = self.om0 if self.om0 is not None else Planck18.Om0
            return w0waCDM(H0=self.h0, w0=w0, wa=wa, Om0=om0, Ode0=1.0-om0)
        else:
            raise ValueError(f"Unknown cosmology: {self.cosmology}")
    
    def get_rp_bins(self) -> np.ndarray:
        """Get the radial bins for deltasigma."""
        if self.binning == "log":
            return np.geomspace(self.rp_min, self.rp_max, self.n_rp_bins + 1)
        else:
            return np.linspace(self.rp_min, self.rp_max, self.n_rp_bins + 1)
    
    def get_theta_bins(self) -> u.Quantity:
        """Get the angular bins for gammat."""
        if self.binning == "log":
            bins = np.geomspace(self.theta_min, self.theta_max, self.n_theta_bins + 1)
        else:
            bins = np.linspace(self.theta_min, self.theta_max, self.n_theta_bins + 1)
        return bins * u.arcmin 