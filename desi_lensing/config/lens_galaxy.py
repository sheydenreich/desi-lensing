"""Lens galaxy configuration for lensing pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np

from .base import BaseConfig
from .validation_helpers import validate_choice, validate_list_not_empty, validate_increasing_sequence, validate_range


@dataclass
class LensGalaxyConfig(BaseConfig):
    """Configuration for lens galaxies."""
    
    # Galaxy type
    galaxy_type: Literal["BGS_BRIGHT", "LRG", "ELG"] = "BGS_BRIGHT"
    
    # Release selection
    release: Literal["iron", "loa"] = "iron"
    
    # Catalogue versions
    bgs_catalogue_version: str = "v1.5"
    lrg_catalogue_version: str = "v1.5"
    elg_catalogue_version: str = "v1.5"
    
    # Weight type for lenses
    weight_type: Literal["None", "FRACZ_TILELOCID", "PROB_OBS", "WEIGHT"] = "WEIGHT"
    
    # Redshift bins
    z_bins: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    
    # Random catalogues
    which_randoms: List[int] = field(default_factory=lambda: [1, 2])
    randoms_ratio: float = -1.0  # If >= 0, subsample randoms to this ratio * len(lenses)
    
    # Magnitude cuts
    magnitude_cuts: bool = True
    magnitude_column: str = "LOGMSTAR"  # Column to use for magnitude cuts
    
    # Mass completeness
    mstar_complete: bool = False
    
    def validate(self) -> List[str]:
        """Validate lens galaxy configuration with helpful error messages."""
        errors = []
        
        # Validate galaxy type
        error = validate_choice(
            self.galaxy_type,
            ["BGS_BRIGHT", "LRG", "ELG"],
            "galaxy_type",
            context="LensGalaxyConfig"
        )
        if error:
            errors.append(error)
        
        # Validate release
        error = validate_choice(
            self.release,
            ["iron", "loa"],
            "release",
            context="LensGalaxyConfig"
        )
        if error:
            errors.append(error)
        
        # Validate weight type
        error = validate_choice(
            self.weight_type,
            ["None", "FRACZ_TILELOCID", "PROB_OBS", "WEIGHT"],
            "weight_type",
            context="LensGalaxyConfig"
        )
        if error:
            errors.append(error)
        
        # Validate z_bins
        if len(self.z_bins) < 2:
            errors.append("[LensGalaxyConfig.z_bins] Must have at least 2 values to define bins")
        else:
            error = validate_increasing_sequence(self.z_bins, "z_bins", "LensGalaxyConfig")
            if error:
                errors.append(error)
        
        if any(z < 0 for z in self.z_bins):
            errors.append("[LensGalaxyConfig.z_bins] All redshift values must be non-negative")
        
        # Validate randoms
        error = validate_list_not_empty(self.which_randoms, "which_randoms", "LensGalaxyConfig")
        if error:
            errors.append(error)
        
        if any(r < 0 for r in self.which_randoms):
            errors.append("[LensGalaxyConfig.which_randoms] All random indices must be non-negative")
        
        # Validate randoms_ratio
        if self.randoms_ratio == 0:
            errors.append(
                "[LensGalaxyConfig.randoms_ratio] Must be either negative (disabled) or positive. "
                "Use -1.0 to disable subsampling, or a positive value like 2.0 for 2x more randoms than lenses"
            )
        
        return errors
    
    def get_catalogue_version(self) -> str:
        """Get the catalogue version for the current galaxy type."""
        version_map = {
            "BGS_BRIGHT": self.bgs_catalogue_version,
            "LRG": self.lrg_catalogue_version,
            "ELG": self.elg_catalogue_version,
        }
        return version_map[self.galaxy_type]
    
    def get_n_lens_bins(self) -> int:
        """Get the number of lens redshift bins."""
        return len(self.z_bins) - 1
    
    def get_lens_bin_centers(self) -> np.ndarray:
        """Get the centers of the lens redshift bins."""
        bins = np.array(self.z_bins)
        return (bins[:-1] + bins[1:]) / 2
    
    def get_default_z_bins(self) -> List[float]:
        """Get default redshift bins for the galaxy type."""
        defaults = {
            "BGS_BRIGHT": [0.1, 0.2, 0.3, 0.4],
            "LRG": [0.4, 0.6, 0.8, 1.1],
            "ELG": [0.8, 1.1, 1.6],
        }
        return defaults.get(self.galaxy_type, [0.1, 0.2, 0.3, 0.4])
    
    def use_default_z_bins(self) -> None:
        """Set z_bins to the default values for this galaxy type."""
        self.z_bins = self.get_default_z_bins()
    
    @classmethod
    def from_preset(cls, preset: str) -> "LensGalaxyConfig":
        """
        Create LensGalaxyConfig from a named preset.
        
        Available presets:
        - 'bgs_basic': Basic BGS_BRIGHT configuration
        - 'bgs_complete': M* complete BGS_BRIGHT configuration
        - 'lrg_basic': Basic LRG configuration
        - 'elg_basic': Basic ELG configuration
        
        Parameters
        ----------
        preset : str
            Name of the preset configuration
            
        Returns
        -------
        LensGalaxyConfig
            Configuration instance with preset values
            
        Raises
        ------
        ValueError
            If the preset name is not recognized
            
        Examples
        --------
        >>> config = LensGalaxyConfig.from_preset('bgs_basic')
        >>> config = LensGalaxyConfig.from_preset('lrg_basic')
        """
        presets = {
            'bgs_basic': {
                'galaxy_type': 'BGS_BRIGHT',
                'release': 'iron',
                'bgs_catalogue_version': 'v1.5',
                'z_bins': [0.1, 0.2, 0.3, 0.4],
                'which_randoms': [1, 2],
                'magnitude_cuts': True,
                'mstar_complete': False,
            },
            'bgs_complete': {
                'galaxy_type': 'BGS_BRIGHT',
                'release': 'iron',
                'bgs_catalogue_version': 'v1.5',
                'z_bins': [0.1, 0.2, 0.3, 0.4],
                'which_randoms': [1, 2],
                'magnitude_cuts': True,
                'mstar_complete': True,
            },
            'lrg_basic': {
                'galaxy_type': 'LRG',
                'release': 'iron',
                'lrg_catalogue_version': 'v1.5',
                'z_bins': [0.4, 0.6, 0.8, 1.1],
                'which_randoms': [1, 2],
                'magnitude_cuts': False,
                'mstar_complete': False,
            },
            'elg_basic': {
                'galaxy_type': 'ELG',
                'release': 'iron',
                'elg_catalogue_version': 'v1.5',
                'z_bins': [0.8, 1.1, 1.6],
                'which_randoms': [1, 2],
                'magnitude_cuts': False,
                'mstar_complete': False,
            },
        }
        
        if preset not in presets:
            available = ', '.join(presets.keys())
            raise ValueError(
                f"Unknown preset '{preset}'. Available presets: {available}"
            )
        
        return cls(**presets[preset])
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """Return list of available preset names."""
        return ['bgs_basic', 'bgs_complete', 'lrg_basic', 'elg_basic'] 