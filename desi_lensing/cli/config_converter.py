"""Convert old INI config files to new YAML format."""

import configparser
from pathlib import Path
from typing import Dict, Any

from ..config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig


def convert_config_file(input_path: Path, output_path: Path) -> None:
    """Convert an INI config file to YAML format."""
    
    config = configparser.ConfigParser()
    config.read(input_path)
    
    # Extract values from old config
    
    # Computation settings
    computation_data = {
        'statistics': config.get('misc', 'statistics', fallback='deltasigma').split(','),
        'cosmology': 'planck18',  # Default, could be inferred from other settings
        'n_jobs': config.getint('misc', 'njobs', fallback=0),
        'comoving': config.getboolean('misc', 'comoving', fallback=True),
        'lens_source_cut': _parse_lens_source_cut(config.get('misc', 'lens_source_cut', fallback='0.1')),
        'n_jackknife_fields': config.getint('misc', 'n_jackknife_fields', fallback=100),
        'tomography': config.getboolean('misc', 'tomography', fallback=False),
    }
    
    # Lens galaxy settings
    lens_data = {
        'galaxy_type': config.get('lens galaxies', 'galaxy_type', fallback='BGS_BRIGHT'),
        'weight_type': config.get('lens galaxies', 'weight_type', fallback='FRACZ_TILELOCID'),
        'bgs_catalogue_version': config.get('lens galaxies', 'BGS_catalogue_version', fallback='v1.5'),
        'lrg_catalogue_version': config.get('lens galaxies', 'LRG_catalogue_version', fallback='v1.5'),
        'z_bins': _parse_list(config.get('lens galaxies', 'z_bins', fallback='0.1,0.2,0.3,0.4')),
        'which_randoms': _parse_int_list(config.get('lens galaxies', 'which_randoms', fallback='1,2')),
        'magnitude_cuts': config.getboolean('lens galaxies', 'magnitude_cuts', fallback=True),
    }
    
    # Add binning information
    if 'deltasigma' in computation_data['statistics']:
        computation_data.update({
            'rp_min': config.getfloat('lens galaxies', 'rpmin', fallback=0.08),
            'rp_max': config.getfloat('lens galaxies', 'rpmax', fallback=80.0),
            'n_rp_bins': config.getint('lens galaxies', 'n_rpbins', fallback=15),
        })
    
    if 'gammat' in computation_data['statistics']:
        computation_data.update({
            'theta_min': config.getfloat('lens galaxies', 'thetamin', fallback=0.3),
            'theta_max': config.getfloat('lens galaxies', 'thetamax', fallback=300.0),
            'n_theta_bins': config.getint('lens galaxies', 'n_thetabins', fallback=15),
        })
    
    computation_data['binning'] = config.get('lens galaxies', 'linlog', fallback='log')
    
    # Source survey settings
    surveys = config.get('source galaxies', 'surveys', fallback='DES,KiDS,HSCY1,HSCY3').split(',')
    surveys = [s.strip() for s in surveys]
    
    source_data = {
        'surveys': surveys,
        'cut_catalogues_to_desi': True,  # Assume this is the default
    }
    
    # Add survey-specific settings
    for survey in ['des', 'kids', 'hscy1', 'hscy3', 'sdss']:
        if config.has_section(survey):
            for key in config[survey]:
                attr_name = f"{survey}_{key}"
                value = config.getboolean(survey, key, fallback=False)
                source_data[attr_name] = value
    
    # Output settings
    output_data = {
        'catalogue_path': config.get('misc', 'catalogue_path', 
                                   fallback='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/'),
        'source_catalogue_path': config.get('misc', 'source_catalogue_path', 
                                           fallback=config.get('misc', 'catalogue_path', 
                                                             fallback='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/')),
        'save_path': config.get('misc', 'savepath',
                              fallback='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensing_measurements/'),
        'magnification_bias_path': config.get('misc', 'magnification_bias_path',
                                            fallback='/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/magnification_bias_DESI/'),
        'verbose': config.getboolean('misc', 'verbose', fallback=True),
        'save_precomputed': True,  # Default assumption
        'apply_blinding': True,    # Default assumption
    }
    
    # Create config objects
    comp_config = ComputationConfig(**computation_data)
    lens_config = LensGalaxyConfig(**lens_data)
    source_config = SourceSurveyConfig(**source_data)
    output_config = OutputConfig(**output_data)
    
    # Create combined config structure
    full_config = {
        'computation': comp_config.to_dict(),
        'lens_galaxy': lens_config.to_dict(),
        'source_survey': source_config.to_dict(),
        'output': output_config.to_dict(),
    }
    
    # Save to YAML
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)


def _parse_lens_source_cut(value: str) -> float:
    """Parse lens_source_cut value."""
    if value.lower() in ['none', 'null']:
        return None
    return float(value)


def _parse_list(value: str) -> list:
    """Parse comma-separated list."""
    return [float(x.strip()) for x in value.split(',')]


def _parse_int_list(value: str) -> list:
    """Parse comma-separated integer list."""
    return [int(x.strip()) for x in value.split(',')] 