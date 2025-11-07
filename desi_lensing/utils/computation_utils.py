"""Computation utility functions."""

import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck18
import camb


def is_table_masked(table: Table) -> bool:
    """Check if any columns in the table are masked."""
    return any(getattr(col, 'mask', None) is not None for col in table.columns.values())


def astropy_to_camb(astropy_cosmo):
    """Convert astropy cosmology to CAMB parameters."""
    H0 = astropy_cosmo.H0.value
    Om0 = astropy_cosmo.Om0
    Ob0 = astropy_cosmo.Ob0
    if Ob0 is None:
        Ob0 = 0.0483  # Default value for Planck 2018

    h = astropy_cosmo.h
    ombh2 = Ob0 * h**2
    omch2 = (Om0 - Ob0) * h**2
    
    # Initialize CAMBparams object
    camb_params = camb.CAMBparams()
    camb_params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    
    # Set dark energy parameters
    w0 = -1  # Default for cosmological constant
    if hasattr(astropy_cosmo, 'w0'):
        w0 = astropy_cosmo.w0
    camb_params.set_dark_energy(w=w0, dark_energy_model='fluid')
    
    return camb_params


def get_camb_results(table_l: Table):
    """Get CAMB results for magnification bias computation."""
    # For now, use Planck18 cosmology
    # In the future, this could use the cosmology from table metadata
    cosmo = Planck18
    camb_params = astropy_to_camb(cosmo)
    camb_params.WantTransfer = True
    camb_params.NonLinear = camb.model.NonLinear_both
    camb_params.set_matter_power(redshifts=np.arange(0, 1.11, 0.01), kmax=1e3)
    camb_results = camb.get_results(camb_params)
    
    return camb_results


def get_blinding_function(
    random_seed: int = None,
    scale: float = 0.75,
    redshift_scale: float = 1.0,
    redshift_slope_center: float = 0.8,
    verbose: bool = False
):
    """Get blinding function for Delta Sigma."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    alpha, beta = (np.random.uniform(0.8, 1.2, size=2) - 1) * scale + 1
    gamma = np.random.normal(loc=0, scale=redshift_scale)

    redshift_mod = lambda zsource: 1 + gamma * (zsource - redshift_slope_center)
    fBlind = lambda r, zsource: 1/4 * ((beta - alpha) * np.log(r/0.15) + 5*alpha - beta) * redshift_mod(zsource)
    
    if verbose:
        print(f"Blinding parameters: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")
    
    return fBlind


def get_blinding_function_gammat(
    random_seed: int = None,
    scale: float = 0.75,
    redshift_scale: float = 1.0,
    redshift_slope_center: float = 0.8,
    verbose: bool = False
):
    """Get blinding function for Gamma_t."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    alpha, beta = (np.random.uniform(0.8, 1.2, size=2) - 1) * scale + 1
    gamma = np.random.normal(loc=0, scale=redshift_scale)

    redshift_mod = lambda zsource: 1 + gamma * (zsource - redshift_slope_center)
    fBlind = lambda r, zsource: 1/4 * ((beta - alpha) * np.log(r/1.5) + 5*alpha - beta) * redshift_mod(zsource)
    
    if verbose:
        print(f"Blinding parameters: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")
    
    return fBlind


def blind_dv(
    result: Table,
    source_survey: str,
    galaxy_type: str,
    lens_bin: int,
    random_seed_galtype_dict: dict = None,
    random_seed_source_survey_dict: dict = None
) -> Table:
    """Apply blinding to the data vector."""
    
    if random_seed_galtype_dict is None:
        random_seed_galtype_dict = {
            "BGS": 4654,
            "BGS_BRIGHT": 4654,
            "LRG": 89753,
            "ELG": 57354
        }
    
    if random_seed_source_survey_dict is None:
        random_seed_source_survey_dict = {
            "des": 98765,
            "hscy1": 98765,
            "kids": 98765,
            "sdss": 98765,
            "hscy3": 98765,
            "decade": 98765,
            "decade_ngc": 98765,
            "decade_sgc": 98765,
        }
    
    random_seed = (random_seed_galtype_dict[galaxy_type] + 
                  random_seed_source_survey_dict[source_survey.lower()] + 
                  int(lens_bin))
    
    result_blinded = result.copy()
    
    if 'ds' in result.keys():
        fBlind = get_blinding_function(random_seed=random_seed)
        result_blinded['ds'] = result['ds'] * fBlind(result['rp'], result['z_s'])
    elif 'et' in result.keys():
        fBlind = get_blinding_function_gammat(random_seed=random_seed)
        result_blinded['et'] = result['et'] * fBlind(result['rp'], result['z_s'])
    else:
        raise ValueError("Data vector table must contain either 'ds' or 'et' column")
    
    return result_blinded 