"""Source survey configuration for lensing pipeline."""

from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional

from .base import BaseConfig


@dataclass
class SourceSurveyConfig(BaseConfig):
    """Configuration for source surveys."""
    
    # Which surveys to use
    surveys: List[Literal["DES", "KiDS", "HSCY1", "HSCY3", "SDSS"]] = field(
        default_factory=lambda: ["DES", "KiDS", "HSCY1", "HSCY3"]
    )
    
    # Cut catalogues to DESI footprint
    cut_catalogues_to_desi: bool = True
    
    # Survey-specific correction settings
    # DES Y3 settings
    des_photo_z_dilution_correction: bool = False
    des_boost_correction: bool = False
    des_scalar_shear_response_correction: bool = True
    des_matrix_shear_response_correction: bool = True
    des_shear_responsivity_correction: bool = False
    des_hsc_selection_bias_correction: bool = False
    des_hsc_additive_shear_bias_correction: bool = False
    des_hsc_y3_selection_bias_correction: bool = False
    des_random_subtraction: bool = True
    
    # KiDS settings
    kids_photo_z_dilution_correction: bool = False
    kids_boost_correction: bool = False
    kids_scalar_shear_response_correction: bool = True
    kids_matrix_shear_response_correction: bool = False
    kids_shear_responsivity_correction: bool = False
    kids_hsc_selection_bias_correction: bool = False
    kids_hsc_additive_shear_bias_correction: bool = False
    kids_hsc_y3_selection_bias_correction: bool = False
    kids_random_subtraction: bool = True
    
    # HSC Y1 settings
    hscy1_photo_z_dilution_correction: bool = True
    hscy1_boost_correction: bool = False
    hscy1_scalar_shear_response_correction: bool = True
    hscy1_matrix_shear_response_correction: bool = False
    hscy1_shear_responsivity_correction: bool = True
    hscy1_hsc_selection_bias_correction: bool = True
    hscy1_hsc_additive_shear_bias_correction: bool = False
    hscy1_hsc_y3_selection_bias_correction: bool = False
    hscy1_random_subtraction: bool = True
    
    # HSC Y3 settings
    hscy3_photo_z_dilution_correction: bool = False
    hscy3_boost_correction: bool = False
    hscy3_scalar_shear_response_correction: bool = True
    hscy3_matrix_shear_response_correction: bool = False
    hscy3_shear_responsivity_correction: bool = True
    hscy3_hsc_selection_bias_correction: bool = False
    hscy3_hsc_additive_shear_bias_correction: bool = True
    hscy3_hsc_y3_selection_bias_correction: bool = True
    hscy3_random_subtraction: bool = True
    
    # SDSS settings
    sdss_photo_z_dilution_correction: bool = True
    sdss_boost_correction: bool = False
    sdss_scalar_shear_response_correction: bool = True
    sdss_matrix_shear_response_correction: bool = False
    sdss_shear_responsivity_correction: bool = True
    sdss_hsc_selection_bias_correction: bool = False
    sdss_hsc_additive_shear_bias_correction: bool = False
    sdss_hsc_y3_selection_bias_correction: bool = False
    sdss_random_subtraction: bool = True
    
    # SDSS-specific defaults
    sdss_mbias: float = -0.04  # 0.96 - 1
    sdss_r: float = 0.87
    
    def validate(self) -> List[str]:
        """Validate source survey configuration."""
        errors = []
        
        # Validate surveys
        valid_surveys = {"DES", "KiDS", "HSCY1", "HSCY3", "SDSS"}
        for survey in self.surveys:
            if survey not in valid_surveys:
                errors.append(f"Invalid survey '{survey}'. Must be one of {valid_surveys}")
        
        if not self.surveys:
            errors.append("At least one survey must be specified")
        
        # Validate SDSS parameters
        if "SDSS" in self.surveys:
            if not -1 <= self.sdss_mbias <= 1:
                errors.append("sdss_mbias should be between -1 and 1")
            if not 0 < self.sdss_r <= 1:
                errors.append("sdss_r should be between 0 and 1")
        
        return errors
    
    def get_survey_settings(self, survey: str) -> Dict[str, bool]:
        """Get correction settings for a specific survey."""
        survey_lower = survey.lower()
        
        settings = {}
        for attr_name in dir(self):
            if attr_name.startswith(f"{survey_lower}_") and not attr_name.startswith("_"):
                # Remove survey prefix and get value
                setting_name = attr_name[len(survey_lower) + 1:]
                settings[setting_name] = getattr(self, attr_name)
        
        return settings
    
    def get_n_tomographic_bins(self, survey: str) -> int:
        """Get number of tomographic bins for a survey."""
        n_tomo_bins = {
            "SDSS": 1,
            "HSCY1": 4,
            "HSCY3": 4,
            "DES": 4,
            "KIDS": 5,
        }
        return n_tomo_bins.get(survey.upper(), 1)
    
    def get_survey_name_mapping(self, survey: str) -> str:
        """Get the internal name mapping for a survey."""
        mapping = {
            "SDSS": "sdss",
            "HSCY1": "hscy1",
            "HSCY3": "hscy3",
            "DES": "desy3",
            "KiDS": "kids1000N",
        }
        return mapping.get(survey.upper(), survey.lower())
    
    def set_survey_defaults(self, survey: str) -> None:
        """Set default correction settings for a specific survey."""
        survey_lower = survey.lower()
        
        # Default settings for each survey
        defaults = {
            "des": {
                "photo_z_dilution_correction": False,
                "boost_correction": False,
                "scalar_shear_response_correction": True,
                "matrix_shear_response_correction": True,
                "shear_responsivity_correction": False,
                "hsc_selection_bias_correction": False,
                "hsc_additive_shear_bias_correction": False,
                "hsc_y3_selection_bias_correction": False,
                "random_subtraction": True,
            },
            "kids": {
                "photo_z_dilution_correction": False,
                "boost_correction": False,
                "scalar_shear_response_correction": True,
                "matrix_shear_response_correction": False,
                "shear_responsivity_correction": False,
                "hsc_selection_bias_correction": False,
                "hsc_additive_shear_bias_correction": False,
                "hsc_y3_selection_bias_correction": False,
                "random_subtraction": True,
            },
            "hscy1": {
                "photo_z_dilution_correction": True,
                "boost_correction": False,
                "scalar_shear_response_correction": True,
                "matrix_shear_response_correction": False,
                "shear_responsivity_correction": True,
                "hsc_selection_bias_correction": True,
                "hsc_additive_shear_bias_correction": False,
                "hsc_y3_selection_bias_correction": False,
                "random_subtraction": True,
            },
            "hscy3": {
                "photo_z_dilution_correction": False,
                "boost_correction": False,
                "scalar_shear_response_correction": True,
                "matrix_shear_response_correction": False,
                "shear_responsivity_correction": True,
                "hsc_selection_bias_correction": False,
                "hsc_additive_shear_bias_correction": True,
                "hsc_y3_selection_bias_correction": True,
                "random_subtraction": True,
            },
            "sdss": {
                "photo_z_dilution_correction": True,
                "boost_correction": False,
                "scalar_shear_response_correction": True,
                "matrix_shear_response_correction": False,
                "shear_responsivity_correction": True,
                "hsc_selection_bias_correction": False,
                "hsc_additive_shear_bias_correction": False,
                "hsc_y3_selection_bias_correction": False,
                "random_subtraction": True,
            },
        }
        
        if survey_lower in defaults:
            for setting, value in defaults[survey_lower].items():
                setattr(self, f"{survey_lower}_{setting}", value) 