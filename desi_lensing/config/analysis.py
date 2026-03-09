"""Analysis configuration for lensing pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from .base import BaseConfig


@dataclass
class AnalysisConfig(BaseConfig):
    """Configuration for analysis-specific settings and choices."""
    
    # Number of bins per galaxy type
    n_bins_per_galaxy_type: Dict[str, int] = field(default_factory=lambda: {
        "BGS_BRIGHT": 3,
        "LRG": 2,
        "ELG": 2
    })
    
    # Scale cuts for different surveys and statistics
    # Format: {survey: {statistic: {"min_deg": value, "max_deg": value, "rp_pivot": value}}}
    scale_cuts: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=lambda: {
        "SDSS": {
            "deltasigma": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0},
            "gammat": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        },
        "KiDS": {
            "deltasigma": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0},
            "gammat": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        },
        "DES": {
            "deltasigma": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0},
            "gammat": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        },
        "DECADE": {
            "deltasigma": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0},
            "gammat": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        },
        "DECADE_NGC": {
            "deltasigma": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0},
            "gammat": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        },
        "DECADE_SGC": {
            "deltasigma": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0},
            "gammat": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        },
        "HSCY1": {
            "deltasigma": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0},
            "gammat": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        },
        "HSCY3": {
            "deltasigma": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0},
            "gammat": {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        }
    })
    
    # Allowed source-lens bin combinations
    # Keys use the upper edge of the lens redshift bin (z_max) for stability
    # across different binning schemes (e.g., BGS_BRIGHT 3-bin vs 1-bin).
    # Format: {galaxy_type}_z{z_max:.2f}
    # BGS_BRIGHT bins: [0.1, 0.2, 0.3, 0.4] -> z_max = 0.20, 0.30, 0.40
    # LRG bins: [0.4, 0.6, 0.8, 1.1] -> z_max = 0.60, 0.80, 1.10
    #
    # Conservative cuts (more restrictive)
    allowed_bins_conservative: Dict[str, Dict[str, List[int]]] = field(default_factory=lambda: {
        "KiDS": {
            "BGS_BRIGHT_z0.20": [3, 4], "BGS_BRIGHT_z0.30": [3, 4], "BGS_BRIGHT_z0.40": [3, 4],
            "LRG_z0.60": [3, 4], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "DES": {
            "BGS_BRIGHT_z0.20": [2, 3], "BGS_BRIGHT_z0.30": [2, 3], "BGS_BRIGHT_z0.40": [2, 3],
            "LRG_z0.60": [3], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "DECADE": {
            "BGS_BRIGHT_z0.20": [2, 3], "BGS_BRIGHT_z0.30": [2, 3], "BGS_BRIGHT_z0.40": [2, 3],
            "LRG_z0.60": [3], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "DECADE_NGC": {
            "BGS_BRIGHT_z0.20": [2, 3], "BGS_BRIGHT_z0.30": [2, 3], "BGS_BRIGHT_z0.40": [2, 3],
            "LRG_z0.60": [3], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "DECADE_SGC": {
            "BGS_BRIGHT_z0.20": [2, 3], "BGS_BRIGHT_z0.30": [2, 3], "BGS_BRIGHT_z0.40": [2, 3],
            "LRG_z0.60": [3], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "HSCY1": {
            "BGS_BRIGHT_z0.20": [1, 2, 3], "BGS_BRIGHT_z0.30": [1, 2, 3], "BGS_BRIGHT_z0.40": [1, 2, 3],
            "LRG_z0.60": [1, 2, 3], "LRG_z0.80": [2, 3], "LRG_z1.10": []
        },
        "HSCY3": {
            "BGS_BRIGHT_z0.20": [1, 2, 3], "BGS_BRIGHT_z0.30": [1, 2, 3], "BGS_BRIGHT_z0.40": [1, 2, 3],
            "LRG_z0.60": [2, 3], "LRG_z0.80": [2, 3], "LRG_z1.10": []
        },
        "SDSS": {
            "BGS_BRIGHT_z0.20": [0], "BGS_BRIGHT_z0.30": [0], "BGS_BRIGHT_z0.40": [0],
            "LRG_z0.60": [], "LRG_z0.80": [], "LRG_z1.10": []
        }
    })
    
    # Less conservative cuts (more permissive)
    allowed_bins_less_conservative: Dict[str, Dict[str, List[int]]] = field(default_factory=lambda: {
        "KiDS": {
            "BGS_BRIGHT_z0.20": [1, 2, 3, 4], "BGS_BRIGHT_z0.30": [2, 3, 4], "BGS_BRIGHT_z0.40": [3, 4],
            "LRG_z0.60": [3, 4], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "DES": {
            "BGS_BRIGHT_z0.20": [0, 1, 2, 3], "BGS_BRIGHT_z0.30": [1, 2, 3], "BGS_BRIGHT_z0.40": [1, 2, 3],
            "LRG_z0.60": [2, 3], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "DECADE": {
            "BGS_BRIGHT_z0.20": [0, 1, 2, 3], "BGS_BRIGHT_z0.30": [1, 2, 3], "BGS_BRIGHT_z0.40": [1, 2, 3],
            "LRG_z0.60": [2, 3], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "DECADE_NGC": {
            "BGS_BRIGHT_z0.20": [0, 1, 2, 3], "BGS_BRIGHT_z0.30": [1, 2, 3], "BGS_BRIGHT_z0.40": [1, 2, 3],
            "LRG_z0.60": [2, 3], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "DECADE_SGC": {
            "BGS_BRIGHT_z0.20": [0, 1, 2, 3], "BGS_BRIGHT_z0.30": [1, 2, 3], "BGS_BRIGHT_z0.40": [1, 2, 3],
            "LRG_z0.60": [2, 3], "LRG_z0.80": [], "LRG_z1.10": []
        },
        "HSCY1": {
            "BGS_BRIGHT_z0.20": [0, 1, 2, 3], "BGS_BRIGHT_z0.30": [0, 1, 2, 3], "BGS_BRIGHT_z0.40": [1, 2, 3],
            "LRG_z0.60": [1, 2, 3], "LRG_z0.80": [2, 3], "LRG_z1.10": [3]
        },
        "HSCY3": {
            "BGS_BRIGHT_z0.20": [0, 1, 2, 3], "BGS_BRIGHT_z0.30": [0, 1, 2, 3], "BGS_BRIGHT_z0.40": [1, 2, 3],
            "LRG_z0.60": [1, 2, 3], "LRG_z0.80": [2, 3], "LRG_z1.10": [3]
        },
        "SDSS": {
            "BGS_BRIGHT_z0.20": [0], "BGS_BRIGHT_z0.30": [0], "BGS_BRIGHT_z0.40": [0],
            "LRG_z0.60": [], "LRG_z0.80": [], "LRG_z1.10": []
        }
    })
    
    # Analysis-specific choices
    use_conservative_cuts: bool = True
    apply_scale_cuts: bool = True
    
    # Analyzed scales categories
    analyzed_scales: List[str] = field(default_factory=lambda: [
        "small scales", "large scales", "all scales"
    ])
    
    # NTILE split configuration per galaxy type
    # n_ntile_* is the total number of NTILE splits
    # n_ntile_computed_* is how many were actually computed (may be less)
    ntile_splits: Dict[str, int] = field(default_factory=lambda: {
        "n_ntile_bgs": 4,           # Total NTILE splits for BGS
        "n_ntile_computed_bgs": 4,   # Computed NTILE splits for BGS
        "n_ntile_lrg": 3,           # Total NTILE splits for LRG  
        "n_ntile_computed_lrg": 3,   # Computed NTILE splits for LRG
    })
    
    def validate(self) -> List[str]:
        """Validate analysis configuration."""
        errors = []
        
        # Validate n_bins_per_galaxy_type
        if not isinstance(self.n_bins_per_galaxy_type, dict):
            errors.append("n_bins_per_galaxy_type must be a dictionary")
        else:
            for galaxy_type, n_bins in self.n_bins_per_galaxy_type.items():
                if not isinstance(galaxy_type, str):
                    errors.append(f"Galaxy type keys must be strings, got {type(galaxy_type)}")
                if not isinstance(n_bins, int) or n_bins <= 0:
                    errors.append(f"Number of bins must be a positive integer, got {n_bins} for {galaxy_type}")
        
        # Validate scale cuts structure
        if not isinstance(self.scale_cuts, dict):
            errors.append("scale_cuts must be a dictionary")
        else:
            for survey, survey_data in self.scale_cuts.items():
                if not isinstance(survey_data, dict):
                    errors.append(f"scale_cuts['{survey}'] must be a dictionary")
                    continue
                for statistic, stat_data in survey_data.items():
                    if not isinstance(stat_data, dict):
                        errors.append(f"scale_cuts['{survey}']['{statistic}'] must be a dictionary")
                        continue
                    required_keys = {"min_deg", "max_deg", "rp_pivot"}
                    if not all(key in stat_data for key in required_keys):
                        errors.append(f"scale_cuts['{survey}']['{statistic}'] missing required keys: {required_keys}")
                    
                    # Validate numerical values
                    for key, value in stat_data.items():
                        if key in required_keys and not isinstance(value, (int, float)):
                            errors.append(f"scale_cuts['{survey}']['{statistic}']['{key}'] must be a number")
                        elif key == "min_deg" and value < 0:
                            errors.append(f"min_deg must be non-negative, got {value}")
                        elif key == "max_deg" and value <= 0:
                            errors.append(f"max_deg must be positive, got {value}")
                        elif key == "rp_pivot" and value <= 0:
                            errors.append(f"rp_pivot must be positive, got {value}")
                    
                    # Check that max_deg > min_deg
                    if "min_deg" in stat_data and "max_deg" in stat_data:
                        if stat_data["max_deg"] <= stat_data["min_deg"]:
                            errors.append(f"max_deg must be greater than min_deg for {survey} {statistic}")
        
        # Validate allowed bins structure
        for cuts_name, cuts_dict in [("conservative", self.allowed_bins_conservative), 
                                     ("less_conservative", self.allowed_bins_less_conservative)]:
            if not isinstance(cuts_dict, dict):
                errors.append(f"allowed_bins_{cuts_name} must be a dictionary")
                continue
            
            for survey, survey_data in cuts_dict.items():
                if not isinstance(survey_data, dict):
                    errors.append(f"allowed_bins_{cuts_name}['{survey}'] must be a dictionary")
                    continue
                
                for bin_key, bin_list in survey_data.items():
                    if not isinstance(bin_list, list):
                        errors.append(f"allowed_bins_{cuts_name}['{survey}']['{bin_key}'] must be a list")
                        continue
                    
                    if not all(isinstance(x, int) and x >= 0 for x in bin_list):
                        errors.append(f"allowed_bins_{cuts_name}['{survey}']['{bin_key}'] must contain non-negative integers")
        
        # Validate analyzed_scales
        if not isinstance(self.analyzed_scales, list):
            errors.append("analyzed_scales must be a list")
        elif not all(isinstance(x, str) for x in self.analyzed_scales):
            errors.append("analyzed_scales must be a list of strings")
        
        # Validate NTILE splits configuration
        if not isinstance(self.ntile_splits, dict):
            errors.append("ntile_splits must be a dictionary")
        else:
            for key, value in self.ntile_splits.items():
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"ntile_splits['{key}'] must be a positive integer, got {value}")
                # Check that computed <= total
                if key.startswith("n_ntile_computed_"):
                    gtype = key.replace("n_ntile_computed_", "")
                    total_key = f"n_ntile_{gtype}"
                    if total_key in self.ntile_splits:
                        if value > self.ntile_splits[total_key]:
                            errors.append(f"n_ntile_computed_{gtype} ({value}) cannot exceed n_ntile_{gtype} ({self.ntile_splits[total_key]})")
        
        return errors
    
    def get_scale_cuts(self, survey: str, statistic: str = "deltasigma") -> Dict[str, float]:
        """
        Get scale cuts for a specific survey and statistic.
        
        Parameters
        ----------
        survey : str
            Source survey name (e.g., "DES", "KiDS")
        statistic : str
            Statistic name ("deltasigma" or "gammat")
            
        Returns
        -------
        Dict[str, float]
            Dictionary with keys "min_deg", "max_deg", "rp_pivot"
        """
        survey = survey.upper()
        statistic = statistic.lower()
        
        if survey not in self.scale_cuts:
            # Return default values if survey not found
            return {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        
        if statistic not in self.scale_cuts[survey]:
            # Try deltasigma as fallback
            if "deltasigma" in self.scale_cuts[survey]:
                return self.scale_cuts[survey]["deltasigma"].copy()
            else:
                return {"min_deg": 0.0041666666666, "max_deg": 2.18, "rp_pivot": 1.0}
        
        return self.scale_cuts[survey][statistic].copy()
    
    @staticmethod
    def _make_z_key(galaxy_type: str, z_max: float) -> str:
        """Create a z_max-based key for allowed bins lookup.
        
        Parameters
        ----------
        galaxy_type : str
            Lens galaxy type (e.g., "BGS_BRIGHT", "LRG")
        z_max : float
            Upper edge of the lens redshift bin
            
        Returns
        -------
        str
            Key string, e.g. "BGS_BRIGHT_z0.40"
        """
        return f"{galaxy_type}_z{z_max:.2f}"
    
    def _find_z_key(self, galaxy_type: str, z_max: float, survey_dict: Dict[str, List[int]]) -> Optional[str]:
        """Find the matching z_max key in a survey dictionary, with floating-point tolerance.
        
        First tries an exact formatted match. If that fails, iterates over keys
        for the given galaxy_type and finds one within atol=0.01.
        
        Parameters
        ----------
        galaxy_type : str
            Lens galaxy type
        z_max : float
            Upper edge of the lens redshift bin
        survey_dict : Dict[str, List[int]]
            The per-survey dictionary of allowed bins
            
        Returns
        -------
        Optional[str]
            The matching key, or None if no match found
        """
        # Try exact formatted match first
        bin_key = self._make_z_key(galaxy_type, z_max)
        if bin_key in survey_dict:
            return bin_key
        
        # Fallback: fuzzy match on z_max for floating-point robustness
        prefix = f"{galaxy_type}_z"
        for key in survey_dict:
            if key.startswith(prefix):
                try:
                    key_z = float(key[len(prefix):])
                    if abs(key_z - z_max) < 0.01:
                        return key
                except ValueError:
                    continue
        
        return None
    
    def get_allowed_source_bins(
        self, 
        galaxy_type: str, 
        source_survey: str, 
        z_max: float, 
        conservative_cut: Optional[bool] = None
    ) -> List[int]:
        """
        Get allowed source bins for a given lens redshift bin upper edge, galaxy type, and survey.
        
        Parameters
        ----------
        galaxy_type : str
            Lens galaxy type ("BGS_BRIGHT", "LRG", "ELG")
        source_survey : str
            Source survey name
        z_max : float
            Upper edge of the lens redshift bin (e.g. 0.2 for BGS bin 1,
            0.6 for LRG bin 1). This is stable across different binning
            schemes (e.g. BGS 3-bin vs 1-bin).
        conservative_cut : bool, optional
            Whether to use conservative cuts. If None, uses self.use_conservative_cuts
            
        Returns
        -------
        List[int]
            List of allowed source bin indices
        """
        if conservative_cut is None:
            conservative_cut = self.use_conservative_cuts
        
        # Choose the appropriate cuts dictionary
        cuts_dict = self.allowed_bins_conservative if conservative_cut else self.allowed_bins_less_conservative
        
        # Look up survey
        survey_key = source_survey.upper()
        if survey_key not in cuts_dict:
            return []
        
        # Find matching key with floating-point tolerance
        bin_key = self._find_z_key(galaxy_type, z_max, cuts_dict[survey_key])
        if bin_key is None:
            return []
        
        return cuts_dict[survey_key][bin_key].copy()
    
    def set_scale_cuts(
        self, 
        survey: str, 
        statistic: str, 
        min_deg: float, 
        max_deg: float, 
        rp_pivot: float
    ) -> None:
        """
        Set scale cuts for a specific survey and statistic.
        
        Parameters
        ----------
        survey : str
            Source survey name
        statistic : str
            Statistic name
        min_deg : float
            Minimum angular scale in degrees
        max_deg : float
            Maximum angular scale in degrees
        rp_pivot : float
            Pivot scale in Mpc/h
        """
        survey = survey.upper()
        statistic = statistic.lower()
        
        if survey not in self.scale_cuts:
            self.scale_cuts[survey] = {}
        
        self.scale_cuts[survey][statistic] = {
            "min_deg": min_deg,
            "max_deg": max_deg,
            "rp_pivot": rp_pivot
        }
    
    def update_allowed_bins(
        self,
        galaxy_type: str,
        source_survey: str,
        z_max: float,
        allowed_bins: List[int],
        conservative: bool = True
    ) -> None:
        """
        Update allowed bins for a specific combination.
        
        Parameters
        ----------
        galaxy_type : str
            Lens galaxy type
        source_survey : str
            Source survey name
        z_max : float
            Upper edge of the lens redshift bin
        allowed_bins : List[int]
            List of allowed source bin indices
        conservative : bool
            Whether to update conservative or less conservative cuts
        """
        cuts_dict = self.allowed_bins_conservative if conservative else self.allowed_bins_less_conservative
        
        survey_key = source_survey.upper()
        bin_key = self._make_z_key(galaxy_type, z_max)
        
        if survey_key not in cuts_dict:
            cuts_dict[survey_key] = {}
        
        cuts_dict[survey_key][bin_key] = list(allowed_bins)
    
    def get_all_allowed_bins_combinations(self, conservative_cut: Optional[bool] = None) -> Dict[str, Dict[str, List[int]]]:
        """
        Get all allowed bin combinations.
        
        Parameters
        ----------
        conservative_cut : bool, optional
            Whether to use conservative cuts. If None, uses self.use_conservative_cuts
            
        Returns
        -------
        Dict[str, Dict[str, List[int]]]
            Nested dictionary of all allowed bin combinations
        """
        if conservative_cut is None:
            conservative_cut = self.use_conservative_cuts
        
        return (self.allowed_bins_conservative if conservative_cut 
                else self.allowed_bins_less_conservative).copy()
    
    def supports_survey(self, survey: str) -> bool:
        """Check if a survey is supported in the analysis configuration."""
        return survey.upper() in self.scale_cuts
    
    def get_supported_surveys(self) -> List[str]:
        """Get list of supported surveys."""
        return list(self.scale_cuts.keys())
    
    def get_scale_categories(self) -> List[str]:
        """Get list of available scale categories for analysis."""
        return self.analyzed_scales.copy()
    
    def get_n_bins_for_galaxy_type(self, galaxy_type: str) -> int:
        """
        Get number of bins for a specific galaxy type.
        
        Parameters
        ----------
        galaxy_type : str
            Galaxy type ("BGS_BRIGHT", "LRG", "ELG")
            
        Returns
        -------
        int
            Number of bins for this galaxy type
        """
        return self.n_bins_per_galaxy_type.get(galaxy_type, 2)
    
    def set_n_bins_for_galaxy_type(self, galaxy_type: str, n_bins: int) -> None:
        """
        Set number of bins for a specific galaxy type.
        
        Parameters
        ----------
        galaxy_type : str
            Galaxy type
        n_bins : int
            Number of bins
        """
        self.n_bins_per_galaxy_type[galaxy_type] = n_bins
    
    def get_total_bins_for_galaxy_types(self, galaxy_types: List[str]) -> int:
        """
        Get total number of bins across multiple galaxy types.
        
        Parameters
        ----------
        galaxy_types : List[str]
            List of galaxy types
            
        Returns
        -------
        int
            Total number of bins
        """
        return sum(self.get_n_bins_for_galaxy_type(gt) for gt in galaxy_types)
    
    def get_ntile_n_splits(self, galaxy_type: str, computed_only: bool = False) -> int:
        """
        Get number of NTILE splits for a galaxy type.
        
        Parameters
        ----------
        galaxy_type : str
            Galaxy type ("BGS_BRIGHT", "LRG", etc.)
        computed_only : bool
            If True, return only the number of computed splits
            
        Returns
        -------
        int
            Number of NTILE splits
        """
        gtype_short = galaxy_type[:3].lower()
        
        if computed_only:
            key = f"n_ntile_computed_{gtype_short}"
        else:
            key = f"n_ntile_{gtype_short}"
        
        return self.ntile_splits.get(key, 4)
    
    def set_ntile_n_splits(
        self, 
        galaxy_type: str, 
        n_splits: int, 
        computed_only: bool = False
    ) -> None:
        """
        Set number of NTILE splits for a galaxy type.
        
        Parameters
        ----------
        galaxy_type : str
            Galaxy type
        n_splits : int
            Number of NTILE splits
        computed_only : bool
            If True, set the computed splits count
        """
        gtype_short = galaxy_type[:3].lower()
        
        if computed_only:
            key = f"n_ntile_computed_{gtype_short}"
        else:
            key = f"n_ntile_{gtype_short}"
        
        self.ntile_splits[key] = n_splits
    
    def get_bin_layout_for_galaxy_types(self, galaxy_types: List[str]) -> Dict[str, Tuple[int, int]]:
        """
        Get the bin layout (start_idx, end_idx) for each galaxy type in a combined plot.
        
        Parameters
        ----------
        galaxy_types : List[str]
            List of galaxy types in order they should appear
            
        Returns
        -------
        Dict[str, Tuple[int, int]]
            Dictionary mapping galaxy type to (start_column, end_column) indices
        """
        layout = {}
        current_idx = 0
        
        for galaxy_type in galaxy_types:
            n_bins = self.get_n_bins_for_galaxy_type(galaxy_type)
            layout[galaxy_type] = (current_idx, current_idx + n_bins)
            current_idx += n_bins
        
        return layout 