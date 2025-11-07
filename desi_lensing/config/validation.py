"""Configuration validation for lensing pipeline."""

from typing import List, Dict, Any
import os
from pathlib import Path

from .computation import ComputationConfig
from .lens_galaxy import LensGalaxyConfig
from .source_survey import SourceSurveyConfig
from .output import OutputConfig


class ConfigValidator:
    """Validates complete pipeline configuration."""
    
    def __init__(
        self,
        computation: ComputationConfig,
        lens_galaxy: LensGalaxyConfig,
        source_survey: SourceSurveyConfig,
        output: OutputConfig
    ):
        """Initialize validator with all configuration components."""
        self.computation = computation
        self.lens_galaxy = lens_galaxy
        self.source_survey = source_survey
        self.output = output
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration components."""
        errors = {
            "computation": self.computation.validate(),
            "lens_galaxy": self.lens_galaxy.validate(),
            "source_survey": self.source_survey.validate(),
            "output": self.output.validate(),
            "cross_validation": self._validate_cross_component()
        }
        return errors
    
    def _validate_cross_component(self) -> List[str]:
        """Validate consistency across configuration components."""
        errors = []
        
        # Check if statistics are supported for surveys
        if "gammat" in self.computation.statistics:
            # Gamma_t requires photo-z dilution correction for some surveys
            for survey in self.source_survey.surveys:
                survey_settings = self.source_survey.get_survey_settings(survey)
                if not survey_settings.get("photo_z_dilution_correction", False):
                    if survey.upper() in ["HSCY1", "SDSS"]:
                        errors.append(
                            f"Survey {survey} typically requires photo_z_dilution_correction=True for gamma_t"
                        )
        
        # Check if lens redshift bins are reasonable for galaxy type
        z_bins = self.lens_galaxy.z_bins
        defaults = self.lens_galaxy.get_default_z_bins()
        
        if len(z_bins) < 2:
            errors.append("Need at least 2 redshift bin edges")
        
        # Warn if using non-standard redshift bins
        if z_bins != defaults:
            # This is just a warning, not an error
            pass
        
        # Check tomography consistency
        if self.computation.tomography:
            for survey in self.source_survey.surveys:
                n_tomo = self.source_survey.get_n_tomographic_bins(survey)
                # if n_tomo < 2:
                    # errors.append(f"Survey {survey} has only {n_tomo} tomographic bin(s), tomography not useful")
        
        # Check cosmology parameters
        if self.computation.cosmology == "wcdm":
            if self.computation.w0 is None or self.computation.wa is None:
                errors.append("wCDM cosmology requires w0 and wa parameters")
        
        # Check file paths exist
        if not os.path.exists(self.output.catalogue_path):
            errors.append(f"Catalogue path does not exist: {self.output.catalogue_path}")
        
        # Check that we have enough randoms for boost correction
        if any(self.source_survey.get_survey_settings(s).get("boost_correction", False) 
               for s in self.source_survey.surveys):
            if len(self.lens_galaxy.which_randoms) < 2:
                errors.append("Boost correction requires at least 2 random catalogues")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if entire configuration is valid."""
        all_errors = self.validate_all()
        return all(len(errors) == 0 for errors in all_errors.values())
    
    def get_error_summary(self) -> str:
        """Get a formatted summary of all validation errors."""
        all_errors = self.validate_all()
        
        if self.is_valid():
            return "Configuration is valid."
        
        summary = "Configuration validation errors:\n\n"
        
        for component, errors in all_errors.items():
            if errors:
                summary += f"{component.replace('_', ' ').title()}:\n"
                for error in errors:
                    summary += f"  - {error}\n"
                summary += "\n"
        
        return summary.rstrip()
    
    def get_warnings(self) -> List[str]:
        """Get configuration warnings (non-fatal issues)."""
        warnings = []
        
        # Check for potential performance issues
        if self.computation.n_jobs == 1:
            warnings.append("Using only 1 CPU core, consider increasing n_jobs for better performance")
        
        if self.computation.n_jackknife_fields > 200:
            warnings.append("Very high number of jackknife fields may be slow")
        
        # Check for unusual settings
        if not self.computation.comoving:
            warnings.append("Using physical coordinates instead of comoving - is this intended?")
        
        if self.computation.lens_source_cut is None:
            warnings.append("No lens-source separation cut applied")
        elif self.computation.lens_source_cut > 0.2:
            warnings.append("Very large lens-source cut may remove too much signal")
        
        # Check output settings
        if not self.output.save_precomputed:
            warnings.append("Not saving precomputed tables - recomputation will be slower")

        if not self.lens_galaxy.z_bins == self.lens_galaxy.get_default_z_bins():
            warnings.append("Using non-standard redshift bins - this may not be what you want")

        # Check magnification bias file paths
        if not os.path.exists(self.output.magnification_bias_path):
            warnings.append(f"Magnification bias path does not exist: {self.output.magnification_bias_path}. Will proceed without magnification bias correction.")
        else:
            # Check lens galaxy version compatibility
            version = self.lens_galaxy.get_catalogue_version()
            mag_bias_file = self.output.get_magnification_bias_file(version)
            if not mag_bias_file.exists():
                warnings.append(f"Magnification bias file not found: {mag_bias_file}. Will proceed without magnification bias correction.")
        
        return warnings 