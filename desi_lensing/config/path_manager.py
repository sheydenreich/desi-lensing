"""Path management for the lensing pipeline."""

from pathlib import Path
from typing import Dict, Optional
import os


class PathManager:
    """Manages all file paths for the lensing pipeline."""
    
    def __init__(self, base_config: "OutputConfig"):
        """Initialize with output configuration."""
        self.base_config = base_config
        self._path_cache: Dict[str, Path] = {}
    
    def get_catalogue_path(self, version: str, survey: Optional[str] = None) -> Path:
        """Get path to catalogue files."""
        cache_key = f"catalogue_{version}_{survey}"
        if cache_key not in self._path_cache:
            path = Path(self.base_config.catalogue_path) / "desi_catalogues" / version
            if survey:
                survey_mapping = {
                    "DES": "desy3",
                    "KIDS": "kids1000N", 
                    "HSCY1": "hscy1",
                    "HSCY3": "hscy3",
                    "SDSS": "sdss"
                }
                survey_name = survey_mapping.get(survey.upper(), survey.lower())
                path = path / survey_name
            self._path_cache[cache_key] = path
        return self._path_cache[cache_key]
    
    def get_source_catalogue_path(self, survey: str, cut_to_desi: bool = True) -> Path:
        """Get path to source survey catalogues."""
        cache_key = f"source_{survey}_{cut_to_desi}"
        if cache_key not in self._path_cache:
            base_path = Path(self.base_config.catalogue_path) / "lensingsurvey_catalogues"
            
            if cut_to_desi:
                if survey.upper() == "SDSS":
                    print("SDSS does not have catalogues cut to DESI, using full catalogue path.")
                else:
                    base_path = base_path / "cut_catalogues"
            
            survey_dirs = {
                "DES": "desy3",
                "KIDS": "kids",
                "HSCY1": "hsc", 
                "HSCY3": "hscy3",
                "SDSS": "sdss"
            }
            
            survey_dir = survey_dirs.get(survey.upper(), survey.lower())
            path = base_path / survey_dir
            self._path_cache[cache_key] = path
        return self._path_cache[cache_key]
    
    def get_lens_catalogue_file(
        self, 
        galaxy_type: str, 
        version: str, 
        survey: Optional[str] = None,
        is_random: bool = False,
        random_index: Optional[int] = None
    ) -> Path:
        """Get specific lens catalogue file path."""
        base_path = self.get_catalogue_path(version, survey)
        
        if is_random:
            if random_index is None:
                raise ValueError("random_index must be provided for random catalogues")
            filename = f"{galaxy_type}_{random_index}_clustering.ran.fits"
        else:
            filename = f"{galaxy_type}_clustering.dat.fits"
        
        return base_path / filename
    
    def get_source_catalogue_file(self, survey: str, galaxy_type: str, cut_to_desi: bool = True) -> Path:
        """Get source catalogue file path."""
        base_path = self.get_source_catalogue_path(survey, cut_to_desi)
        
        galaxy_suffix = ""
        if galaxy_type in ["BGS_BRIGHT"]:
            galaxy_suffix = "_BGS"
        elif galaxy_type in ["LRG", "ELG"]:
            galaxy_suffix = f"_{galaxy_type}"
        
        file_patterns = {
            "DES": f"desy3_cat{galaxy_suffix}.fits",
            "KIDS": f"kids_cat{galaxy_suffix}.fits",
            "HSCY1": f"hsc_cat{galaxy_suffix}.fits",
            "HSCY3": f"hscy3_cat{galaxy_suffix}.fits", 
            "SDSS": "sdss_cat.fits"
        }
        
        filename = file_patterns.get(survey.upper())
        if filename is None:
            raise ValueError(f"Unknown survey: {survey}")
        
        return base_path / filename
    
    def get_calibration_file(self, survey: str) -> Optional[Path]:
        """Get calibration file for survey if needed."""
        if survey.upper() == "HSCY1":
            base_path = Path(self.base_config.catalogue_path) / "lensingsurvey_catalogues" / "hsc"
            return base_path / "hsc_cal.fits"
        elif survey.upper() == "SDSS":
            base_path = Path(self.base_config.catalogue_path) / "lensingsurvey_catalogues" / "sdss"
            return base_path / "table_c_sdss.fits"
        return None
    
    def get_nofz_file(self, survey: str) -> Path:
        """Get n(z) file path for survey."""
        base_path = Path(self.base_config.catalogue_path) / "model_inputs_desiy1"
        
        survey_mapping = {
            "DES": "desy3",
            "KiDS": "kids1000",
            "HSCY1": "hscy1",
            "HSCY3": "hscy3", 
            "SDSS": "sdss"
        }
        
        survey_name = survey_mapping.get(survey.upper(), survey.lower())
        zmax = 3 if survey.upper() != "SDSS" else 2
        
        return base_path / f"pzwei_sources_{survey_name}_tom1_zmax{zmax}.dat"
    
    def get_output_file(
        self,
        statistic: str,
        galaxy_type: str,
        version: str,
        survey: str,
        z_min: float,
        z_max: float,
        lens_bin: Optional[int] = None,
        boost_correction: bool = False,
        is_covariance: bool = False,
        is_bmode: bool = False
    ) -> Path:
        """Get output file path."""
        file_type = "covariance" if is_covariance else "measurement"
        
        return self.base_config.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=version,
            survey=survey,
            z_min=z_min,
            z_max=z_max,
            source_bin=lens_bin,
            boost_correction=boost_correction,
            is_bmode=is_bmode,
            file_type=file_type
        )
    
    def get_precomputed_files(
        self,
        statistic: str,
        galaxy_type: str,
        version: str,
        survey: str,
        z_min: float,
        z_max: float,
        lens_bin: Optional[int] = None,
        boost_correction: bool = False,
        is_bmode: bool = False
    ) -> Dict[str, Path]:
        """Get precomputed file paths."""
        lens_path = self.base_config.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=version,
            survey=survey,
            z_min=z_min,
            z_max=z_max,
            source_bin=lens_bin,
            boost_correction=boost_correction,
            is_bmode=is_bmode,
            file_type="precomputed_lens"
        )
        
        random_path = self.base_config.get_filepath(
            statistic=statistic,
            galaxy_type=galaxy_type,
            version=version,
            survey=survey,
            z_min=z_min,
            z_max=z_max,
            source_bin=lens_bin,
            boost_correction=boost_correction,
            is_bmode=is_bmode,
            file_type="precomputed_random"
        )
        
        # Generate metadata path by changing extension
        metadata_path = lens_path.with_suffix('.pkl')
        
        return {
            "lens": lens_path,
            "random": random_path, 
            "metadata": metadata_path
        }
    
    def validate_paths(self) -> Dict[str, bool]:
        """Validate that required paths exist."""
        paths_to_check = {
            "catalogue_path": self.base_config.catalogue_path,
            "magnification_bias_path": self.base_config.magnification_bias_path,
        }
        
        results = {}
        for name, path in paths_to_check.items():
            results[name] = os.path.exists(path)
        
        return results 