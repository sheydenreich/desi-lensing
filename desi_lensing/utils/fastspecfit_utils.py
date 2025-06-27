"""
Utilities for handling FastSpecFit magnitude data.

This module provides functionality to add FastSpecFit photometric data
to DESI lens catalogues, particularly absolute magnitudes.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import fitsio
from astropy.table import Table, join

from .logging_utils import setup_logger


def get_fastspecfit_magnitudes(
    lens_table: Table,
    fsf_cols: List[str] = None,
    fsf_dir: str = '/pscratch/sd/i/ioannis/fastspecfit/data/loa/catalogs/',
    prog: str = 'bright',
    release: str = 'iron',
    logger: Optional[logging.Logger] = None
) -> Table:
    """
    Add FastSpecFit columns to lens catalogue based on TARGETID match.
    
    Adapted from get_FSF_loa function in LSS pipeline.
    
    Parameters
    ----------
    lens_table : Table
        Input lens catalogue table
    fsf_cols : List[str], optional
        FastSpecFit columns to add. If None, adds default magnitude columns.
    fsf_dir : str
        Directory containing FastSpecFit files
    prog : str
        Program type ('bright' for BGS, 'dark' for LRG/ELG)
    release : str
        DESI release to use ('iron' for current, 'loa' for future)
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    Table
        Lens table with added FastSpecFit columns
    """
    if logger is None:
        logger = setup_logger(__name__)
    
    if fsf_cols is None:
        # Default columns for magnitudes
        fsf_cols = [
            'TARGETID', 'ABSMAG01_SDSS_R'
        ]
    
    # Ensure TARGETID is included for matching
    if 'TARGETID' not in fsf_cols:
        fsf_cols = ['TARGETID'] + fsf_cols
    
    if release.lower().strip() == 'iron':
        _fsf_dir = fsf_dir.replace('loa', 'iron')
        specphot = 'fastspec'
    elif release.lower().strip() == 'loa':
        _fsf_dir = fsf_dir.replace('iron', 'loa')
        specphot = 'fastphot'
    else:
        raise ValueError(f"Invalid release: {release}, must be 'iron' or 'loa'")

    logger.info(f"Loading FastSpecFit data from {_fsf_dir}")
    
    # Check if directory exists
    fsf_path = Path(_fsf_dir)
    if not fsf_path.exists():
        logger.warning(f"FastSpecFit directory {_fsf_dir} not found, skipping")
        return lens_table
    
    # Load FastSpecFit data from healpix files
    fsf_list = []
    try:
        for hp in range(0, 12):  # Standard nside=1 has 12 healpix pixels
            filename = f'{specphot}-{release}-main-{prog}-nside1-hp{hp:02d}.fits'
            filepath = fsf_path / filename
            

            try:
                fsf_data = Table(fitsio.read(str(filepath), ext='SPECPHOT', columns=fsf_cols))
                fsf_list.append(fsf_data)
                logger.debug(f"Loaded {len(fsf_data)} objects from {filename}")
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
        
        if not fsf_list:
            logger.warning("No FastSpecFit files found or readable")
            return lens_table
        
        # Concatenate all FastSpecFit data
        fsf_combined = Table(np.concatenate([table.as_array() for table in fsf_list]))
        logger.info(f"Combined FastSpecFit data: {len(fsf_combined)} objects")
        
        # Join with lens catalogue on TARGETID
        original_length = len(lens_table)
        lens_table_with_fsf = join(lens_table, fsf_combined, keys=['TARGETID'], join_type='left')
        
        logger.info(f"Length before/after FastSpecFit join: {original_length} -> {len(lens_table_with_fsf)}")
        
        # Add derived magnitude if needed
        lens_table_with_fsf = add_derived_magnitudes(lens_table_with_fsf, logger)
        
        return lens_table_with_fsf
        
    except Exception as e:
        logger.error(f"Error loading FastSpecFit data: {e}")
        return lens_table


def add_derived_magnitudes(lens_table: Table, logger: Optional[logging.Logger] = None) -> Table:
    """
    Add derived magnitude columns to lens catalogue.
    
    Parameters
    ----------
    lens_table : Table
        Input lens catalogue
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    Table
        Table with added derived magnitude columns
    """
    if logger is None:
        logger = setup_logger(__name__)
    
    # Add ABSMAG_RP0 if not present but ABSMAG01_SDSS_R is available
    if 'ABSMAG01_SDSS_R' in lens_table.colnames and 'ABSMAG_RP0' not in lens_table.colnames:
        z_col = 'Z_not4clus' if 'Z_not4clus' in lens_table.colnames else 'Z'
        
        if z_col in lens_table.colnames:
            # Apply redshift evolution correction: ABSMAG_RP0 = ABSMAG01_SDSS_R + 0.97*z - 0.095
            absmag_rp0 = lens_table['ABSMAG01_SDSS_R'] + 0.97 * lens_table[z_col] - 0.095
            lens_table['ABSMAG_RP0'] = absmag_rp0
            logger.info("Added ABSMAG_RP0 column with redshift evolution correction")
        else:
            logger.warning(f"Cannot add ABSMAG_RP0: redshift column '{z_col}' not found")
    
    return lens_table


def create_mock_magnitudes(lens_table: Table, logger: Optional[logging.Logger] = None) -> Table:
    """
    Create mock magnitude data for testing when FastSpecFit data is not available.
    
    Parameters
    ----------
    lens_table : Table
        Input lens catalogue
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    Table
        Table with mock magnitude columns
    """
    if logger is None:
        logger = setup_logger(__name__)
    
    z_col = 'Z_not4clus' if 'Z_not4clus' in lens_table.colnames else 'Z'
    
    if z_col not in lens_table.colnames:
        logger.error("Cannot create mock magnitudes: no redshift column found")
        return lens_table
    
    redshifts = lens_table[z_col]
    n_gals = len(lens_table)
    
    # Create realistic-looking mock magnitudes
    # Base magnitude depends on galaxy type and redshift
    if 'ABSMAG01_SDSS_R' not in lens_table.colnames:
        # For BGS: typically -19 to -22, brighter at higher z
        # For LRG: typically -21 to -24, brighter at higher z
        base_mag = -20.0 - 1.5 * np.log10(1 + redshifts) + np.random.normal(0, 0.8, n_gals)
        lens_table['ABSMAG01_SDSS_R'] = base_mag
        logger.warning("Added mock ABSMAG01_SDSS_R column for testing")
    
    # Add other mock magnitude bands if needed
    for band in ['G', 'Z', 'W1', 'W2']:
        col_name = f'ABSMAG01_SDSS_{band}'
        if col_name not in lens_table.colnames:
            # Simple color relations for mock data
            color_offset = {'G': 0.5, 'Z': -0.3, 'W1': -1.0, 'W2': -1.2}.get(band, 0.0)
            mock_mag = lens_table['ABSMAG01_SDSS_R'] + color_offset + np.random.normal(0, 0.2, n_gals)
            lens_table[col_name] = mock_mag
    
    # Add derived magnitude
    lens_table = add_derived_magnitudes(lens_table, logger)
    
    return lens_table


def apply_extinction_correction(
    magnitudes: np.ndarray, 
    redshifts: np.ndarray, 
    correction_type: str = 'linear'
) -> np.ndarray:
    """
    Apply extinction correction to magnitudes.
    
    Parameters
    ----------
    magnitudes : np.ndarray
        Input magnitudes
    redshifts : np.ndarray
        Redshifts corresponding to magnitudes
    correction_type : str
        Type of correction ('linear' for -0.8*(z-0.1))
        
    Returns
    -------
    np.ndarray
        Corrected magnitudes
    """
    if correction_type == 'linear':
        # Standard correction: ecorr = -0.8*(z-0.1)
        ecorr = -0.8 * (redshifts - 0.1)
        return magnitudes + ecorr
    else:
        raise ValueError(f"Unknown correction type: {correction_type}")


def get_default_magnitude_cuts(galaxy_type: str) -> Optional[np.ndarray]:
    """
    Get default magnitude cuts for a galaxy type.
    
    Parameters
    ----------
    galaxy_type : str
        Galaxy type ('BGS_BRIGHT', 'LRG', 'ELG')
        
    Returns
    -------
    Optional[np.ndarray]
        Magnitude cuts per redshift bin, or None if no cuts
    """
    if galaxy_type == "BGS_BRIGHT":
        return np.array([-19.5, -20.5, -21.0])
    elif galaxy_type in ["LRG", "ELG"]:
        return None
    else:
        return None 