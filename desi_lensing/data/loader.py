"""Data loading functionality for the lensing pipeline."""

import logging
import os
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime
from copy import deepcopy

import numpy as np
import healpy as hp
import fitsio
from astropy.io import fits
from astropy.table import Table, vstack, join

from dsigma.helpers import dsigma_table
from dsigma.surveys import des, kids, hsc, decade

from ..config import LensGalaxyConfig, SourceSurveyConfig, OutputConfig, PathManager


class DataLoader:
    """Handles loading of lens and source catalogues."""
    
    def __init__(
        self,
        lens_config: LensGalaxyConfig,
        source_config: SourceSurveyConfig,
        output_config: OutputConfig,
        path_manager: PathManager,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize data loader."""
        self.lens_config = lens_config
        self.source_config = source_config
        self.output_config = output_config
        self.path_manager = path_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Cache for systematic property assignments
        self._systematic_cache = {}
    
    def load_lens_catalogues(self, source_survey: str) -> Tuple[Table, Optional[Table]]:
        """Load lens and random catalogues."""
        self.logger.info(f"Loading lens catalogues for {self.lens_config.galaxy_type}...")
        
        galaxy_type = self.lens_config.galaxy_type
        version = self.lens_config.get_catalogue_version()
        z_bins = self.lens_config.z_bins
        randoms = self.lens_config.which_randoms
        
        # Check for mass completeness
        if self.lens_config.mstar_complete:
            return self._load_mstar_complete_lens_catalogues(galaxy_type, version, z_bins, randoms)
        
        # Regular loading
        necessary_columns = ['RA', 'DEC', 'Z', 'TARGETID']
        columns_to_add = necessary_columns.copy()
        
        # Add systematic weights if needed
        if self.lens_config.weight_type != "None":
            columns_to_add.append(self.lens_config.weight_type)
        
        # Load main catalogue
        catalogue_path = self.path_manager.get_lens_catalogue_file(
            galaxy_type, version, survey=source_survey, is_random=False
        )
        
        self.logger.info(f"Reading {catalogue_path} from {self._get_last_mtime(catalogue_path)}")
        table_l = self._read_table(catalogue_path, columns=columns_to_add)
        
        # Apply magnitude cuts if requested
        if self.lens_config.magnitude_cuts:
            table_l = self._apply_magnitude_cuts(table_l)
        
        
        # Convert to dsigma table
        table_l = dsigma_table(
            table_l, 'lens', z='Z', ra='RA', dec='DEC', w_sys=self.lens_config.weight_type,TARGETID='TARGETID',
            verbose=self.output_config.verbose
        )
        
        # Load randoms if requested
        table_r = None
        if randoms:
            table_r = self._load_random_catalogues(galaxy_type, source_survey, version, randoms, columns_to_add)
        
        return table_l, table_r
    
    def load_source_catalogue(
        self, 
        survey: str, 
        galaxy_type: str
    ) -> Tuple[Table, Dict[str, Any], Dict[str, Any]]:
        """Load source catalogue and return precompute/stacking kwargs."""
        self.logger.info(f"Loading source catalogue for {survey}...")
        
        cut_to_desi = self.source_config.cut_catalogues_to_desi
        survey_settings = self.source_config.get_survey_settings(survey)
        
        if survey.upper() == "DES":
            return self._load_des_catalogue(galaxy_type, cut_to_desi, survey_settings)
        elif survey.upper() == "KIDS":
            return self._load_kids_catalogue(galaxy_type, cut_to_desi, survey_settings)
        elif survey.upper() == "HSCY1":
            return self._load_hscy1_catalogue(galaxy_type, cut_to_desi, survey_settings)
        elif survey.upper() == "HSCY3":
            return self._load_hscy3_catalogue(galaxy_type, cut_to_desi, survey_settings)
        elif survey.upper() == "SDSS":
            return self._load_sdss_catalogue(galaxy_type, cut_to_desi, survey_settings)
        elif survey.upper() == "DECADE_NGC":
            return self._load_decade_catalogue(galaxy_type, cut_to_desi, "NGC", survey_settings)
        elif survey.upper() == "DECADE_SGC":
            return self._load_decade_catalogue(galaxy_type, cut_to_desi, "SGC", survey_settings)
        else:
            raise ValueError(f"Unsupported survey: {survey}")

    def _load_decade_catalogue(self, galaxy_type: str, cut_to_desi: bool, ngcsgc: str, settings: Dict[str, bool]) -> Tuple[Table, Dict[str, Any], Dict[str, Any]]:
        """Load DECADE DR1 catalogue."""
        # Get file path

        surveystr = f"DECADE_{ngcsgc}"
        catalogue_path = self.path_manager.get_source_catalogue_file(surveystr, galaxy_type, cut_to_desi)
        
        self.logger.info(f"Reading {catalogue_path} from {self._get_last_mtime(catalogue_path)}")
        table_s = Table.read(catalogue_path)
        
        for key in ['R_11', 'R_22', 'R_12', 'R_21']:
            table_s[key] = 0.
        # Apply shear response correction
        for z_bin in range(4):
            select = table_s['DNF_Z'] == z_bin
            R_gamma,R_sel = decade.shear_response(table_s[select])
            if self.output_config.verbose:
                self.logger.info(f"Bin {z_bin + 1}: R_gamma = {100 * 0.5 * np.sum(np.diag(R_gamma)):.1f}% R_sel = {100 * 0.5 * np.sum(np.diag(R_sel)):.1f}%")
                self.logger.info(f"N_gals: {np.sum(select)}")
            for mcal_i in range(2):
                for mcal_j in range(2):
                    table_s[f'R_{mcal_i+1}{mcal_j+1}'][select] = R_gamma[mcal_i,mcal_j] + R_sel[mcal_i,mcal_j]

        table_s = table_s[table_s['DNF_Z'] >= 0]

        table_s = dsigma_table(
            table_s, 'source',
            e_2_convention='standard',
            verbose=self.output_config.verbose,
        )        
        
        # Add multiplicative bias
        table_s['m'] = decade.multiplicative_shear_bias(table_s['z_bin'], gal_cap = ngcsgc)
        
        # Load n(z)
        table_n = self._read_nofz(surveystr)
        
        precompute_kwargs = {'table_n': table_n}
        stacking_kwargs = self._build_stacking_kwargs(settings)
        
        return table_s, precompute_kwargs, stacking_kwargs

    
    def _load_des_catalogue(self, galaxy_type: str, cut_to_desi: bool, settings: Dict[str, bool]) -> Tuple[Table, Dict[str, Any], Dict[str, Any]]:
        """Load DES Y3 catalogue."""
        # Get file path
        catalogue_path = self.path_manager.get_source_catalogue_file("DES", galaxy_type, cut_to_desi)
        
        self.logger.info(f"Reading {catalogue_path} from {self._get_last_mtime(catalogue_path)}")
        table_s = Table.read(catalogue_path)
        
        # DES column mapping
        desy3_keys = {
            'ra': 'RA',
            'dec': 'Dec',
            'z': np.nan,
            'z_bin': 'tombin',
            'e_1': 'e1',
            'e_2': 'e2',
            'w': 'wei',
            'w_1p': 'wei_1p',
            'w_1m': 'wei_1m',
            'w_2p': 'wei_2p',
            'w_2m': 'wei_2m',
            'R_11': 'R11',
            'R_12': 'R12',
            'R_21': 'R21',
            'R_22': 'R22',
            'flags_select': 'mask',
            'flags_select_1p': 'mask_1p',
            'flags_select_1m': 'mask_1m',
            'flags_select_2p': 'mask_2p',
            'flags_select_2m': 'mask_2m'
        }
        
        table_s = dsigma_table(
            table_s, 'source',
            e_2_convention='standard',
            verbose=self.output_config.verbose,
            **desy3_keys
        )
        
        # Convert flags to boolean
        flag_columns = ['flags_select', 'flags_select_1p', 'flags_select_1m', 'flags_select_2p', 'flags_select_2m']
        for col in flag_columns:
            table_s[col] = table_s[col].astype(bool)
        
        table_s['z_bin'] = table_s['z_bin'].astype(int)
        
        # Apply selection response correction
        for z_bin in range(4):
            select = table_s['z_bin'] == z_bin
            R_sel = des.selection_response(table_s[select])
            if self.output_config.verbose:
                self.logger.info(f"Bin {z_bin + 1}: R_sel = {100 * 0.5 * np.sum(np.diag(R_sel)):.1f}%")
                self.logger.info(f"N_gals: {np.sum(select)}")
            table_s['R_11'][select] += 0.5 * np.sum(np.diag(R_sel))
            table_s['R_22'][select] += 0.5 * np.sum(np.diag(R_sel))
        
        # Filter valid z_bins and apply selection
        table_s = table_s[table_s['z_bin'] >= 0]
        table_s = table_s[table_s['flags_select']]
        
        # Add multiplicative bias
        table_s['m'] = des.multiplicative_shear_bias(table_s['z_bin'], version='Y3')
        
        # Set effective redshifts
        table_s['z'] = np.array([0.0, 0.358, 0.631, 0.872])[table_s['z_bin']]
        
        # Load n(z)
        table_n = self._read_nofz("DES")
        
        precompute_kwargs = {'table_n': table_n}
        stacking_kwargs = self._build_stacking_kwargs(settings)
        
        return table_s, precompute_kwargs, stacking_kwargs
    
    def _load_kids_catalogue(self, galaxy_type: str, cut_to_desi: bool, settings: Dict[str, bool]) -> Tuple[Table, Dict[str, Any], Dict[str, Any]]:
        """Load KiDS catalogue."""
        catalogue_path = self.path_manager.get_source_catalogue_file("KiDS", galaxy_type, cut_to_desi)
        
        self.logger.info(f"Reading {catalogue_path} from {self._get_last_mtime(catalogue_path)}")
        table_s = Table.read(catalogue_path)
        
        kids_keys = {
            'ra': 'RA',
            'dec': 'Dec', 
            'z': 'z_phot',
            'e_1': 'e_1',
            'e_2': 'e_2',
            'w': 'weight'
        }
        
        table_s = dsigma_table(
            table_s, 'source',
            e_2_convention='standard',
            verbose=self.output_config.verbose,
            **kids_keys
        )
        
        # Add tomographic bins and multiplicative bias
        table_s['z_bin'] = kids.tomographic_redshift_bin(table_s['z'], version='DR4')
        table_s['m'] = kids.multiplicative_shear_bias(table_s['z_bin'], version='DR4')
        table_s = table_s[table_s['z_bin'] >= 0]
        table_s['z'] = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[table_s['z_bin']]
        
        # Load n(z)
        table_n = self._read_nofz("KiDS")
        
        precompute_kwargs = {'table_n': table_n}
        stacking_kwargs = self._build_stacking_kwargs(settings)
        
        return table_s, precompute_kwargs, stacking_kwargs
    
    def _load_hscy1_catalogue(self, galaxy_type: str, cut_to_desi: bool, settings: Dict[str, bool]) -> Tuple[Table, Dict[str, Any], Dict[str, Any]]:
        """Load HSC Y1 catalogue."""
        catalogue_path = self.path_manager.get_source_catalogue_file("HSCY1", galaxy_type, cut_to_desi)
        
        self.logger.info(f"Reading {catalogue_path} from {self._get_last_mtime(catalogue_path)}")
        table_s = Table.read(catalogue_path)
        
        hsc_keys = {
            'ra': 'RA',
            'dec': 'Dec',
            'z': 'z_phot',
            'z_low': 'z_low',
            'e_1': 'e_1',
            'e_2': 'e_2',
            'w': 'weight',
            'm': 'm_corr',
            'e_rms': 'e_rms',
            'R_2': 'resolution'
        }
        
        table_s = dsigma_table(
            table_s, 'source',
            e_2_convention='standard',
            verbose=self.output_config.verbose,
            **hsc_keys
        )
        
        # Add tomographic bins
        hsc_bins = [0.3, 0.6, 0.9, 1.2, 1.5]
        table_s.add_column(np.digitize(table_s['z'], hsc_bins) - 1, name='z_bin')
        
        # Load calibration table
        cal_path = self.path_manager.get_calibration_file("HSCY1")
        self.logger.info(f"Reading {cal_path} from {self._get_last_mtime(cal_path)}")
        table_c = Table.read(cal_path)
        table_c = dsigma_table(
            table_c, 'calibration', w_sys='weight_som',
            w='weight_source', z_true='z_cosmos', z='zphot_ephor',
            verbose=self.output_config.verbose
        )
        
        precompute_kwargs = {'table_c': table_c}
        stacking_kwargs = self._build_stacking_kwargs(settings)
        
        return table_s, precompute_kwargs, stacking_kwargs
    
    def _load_hscy3_catalogue(self, galaxy_type: str, cut_to_desi: bool, settings: Dict[str, bool]) -> Tuple[Table, Dict[str, Any], Dict[str, Any]]:
        """Load HSC Y3 catalogue."""
        catalogue_path = self.path_manager.get_source_catalogue_file("HSCY3", galaxy_type, cut_to_desi)
        
        self.logger.info(f"Reading {catalogue_path} from {self._get_last_mtime(catalogue_path)}")
        table_s = Table.read(catalogue_path)
        
        hsc_keys = {
            'ra': 'RA',
            'dec': 'Dec',
            'z': 1.,
            'e_1': 'e_1',
            'e_2': 'e_2',
            'w': 'weight',
            'm': 'm_corr',
            'e_rms': 'e_rms',
            'R_2': 'resolution',
            'c_1': 'c_1',
            'c_2': 'c_2',
            'z_bin': 'z_bin',
            'e_psf_1': 'e1_psf',
            'e_psf_2': 'e2_psf',
            'magA': 'aperture_mag'
        }
        
        table_s['z_bin'] = table_s['z_bin'].astype(int) - 1
        
        table_s = dsigma_table(
            table_s, 'source',
            e_2_convention='standard',
            verbose=self.output_config.verbose,
            **hsc_keys
        )
        
        # Load n(z)
        table_n = self._read_nofz("HSCY3")
        
        precompute_kwargs = {'table_n': table_n}
        stacking_kwargs = self._build_stacking_kwargs(settings)
        
        return table_s, precompute_kwargs, stacking_kwargs
    
    def _load_sdss_catalogue(self, galaxy_type: str, cut_to_desi: bool, settings: Dict[str, bool]) -> Tuple[Table, Dict[str, Any], Dict[str, Any]]:
        """Load SDSS catalogue."""
        catalogue_path = self.path_manager.get_source_catalogue_file("SDSS", galaxy_type, cut_to_desi)
        
        self.logger.info(f"Reading {catalogue_path} from {self._get_last_mtime(catalogue_path)}")
        table_s = Table.read(catalogue_path)
        
        # SDSS parameters
        mbias_sdss = self.source_config.sdss_mbias
        r_sdss = self.source_config.sdss_r
        e_rms_sdss = np.sqrt(1 - r_sdss)
        
        sdss_keys = {
            'ra': 'RA',
            'dec': 'Dec',
            'z': 'z_phot',
            'e_1': 'e_1',
            'e_2': 'e_2',
            'w': 'weight'
        }
        
        table_s = dsigma_table(
            table_s, 'source',
            e_2_convention='standard',
            z_bin=0, m=mbias_sdss, e_rms=e_rms_sdss,
            verbose=self.output_config.verbose,
            **sdss_keys
        )
        
        # Load calibration table
        cal_path = self.path_manager.get_calibration_file("SDSS")
        self.logger.info(f"Reading {cal_path} from {self._get_last_mtime(cal_path)}")
        table_c = Table.read(cal_path)
        table_c = dsigma_table(
            table_c, 'calibration', w_sys='w_sys',
            w='w', z_true='z_true', z='z',
            verbose=self.output_config.verbose
        )
        
        precompute_kwargs = {'table_c': table_c}
        stacking_kwargs = self._build_stacking_kwargs(settings)
        
        return table_s, precompute_kwargs, stacking_kwargs
    
    def _load_mstar_complete_lens_catalogues(self, galaxy_type: str, version: str, z_bins: List[float], randoms: List[int]) -> Tuple[Table, Optional[Table]]:
        """Load mass-complete lens catalogues for BGS."""
        if galaxy_type != "BGS_BRIGHT":
            raise ValueError("Mass complete catalogues only available for BGS_BRIGHT")
        
        necessary_columns = ['RA', 'DEC', 'Z', 'WEIGHT']
        
        # File names for mass complete samples
        fnames = [
            'BGS_BRIGHT_1_2_11.0_clustering.dat.fits',
            'BGS_BRIGHT_2_3_11.0_clustering.dat.fits', 
            'BGS_BRIGHT_3_4_11.3_clustering.dat.fits'
        ]
        
        fpath_load = os.path.join(self.output_config.catalogue_path, "desi_catalogues")
        
        tabs_l = []
        tabs_r = []
        
        for lens_bin in range(1, 4):
            fname = fnames[lens_bin - 1]
            file_path = os.path.join(fpath_load, f'matt_cat/{version}pip/{lens_bin}_{lens_bin+1}/{fname}')
            tab_l = Table(fitsio.read(file_path, columns=necessary_columns))
            tabs_l.append(tab_l)
            
            if randoms:
                for rand in randoms:
                    rand_file = os.path.join(fpath_load, f'matt_cat/{version}pip/{lens_bin}_{lens_bin+1}/BGS_BRIGHT-21.5_{rand}_clustering.ran.fits')
                    tab_r = Table(fitsio.read(rand_file, columns=necessary_columns))
                    tabs_r.append(tab_r)
        
        fin_tab_l = vstack(tabs_l)
        fin_tab_r = vstack(tabs_r) if randoms else None
        
        # Convert to dsigma tables
        table_l = dsigma_table(
            fin_tab_l, 'lens', z='Z', ra='RA', dec='DEC', w_sys='WEIGHT',
            verbose=self.output_config.verbose
        )
        
        table_r = None
        if fin_tab_r is not None:
            table_r = dsigma_table(
                fin_tab_r, 'lens', z='Z', ra='RA', dec='DEC', w_sys='WEIGHT',
                verbose=self.output_config.verbose
            )
        
        return table_l, table_r
    
    def _load_random_catalogues(self, galaxy_type: str, survey: str, version: str, randoms: List[int], columns: List[str]) -> Table:
        """Load random catalogues."""
        tables = []
        
        for r in randoms:
            random_path = self.path_manager.get_lens_catalogue_file(
                galaxy_type, version, survey=survey, is_random=True, random_index=r
            )
            
            self.logger.info(f"Reading {random_path} from {self._get_last_mtime(random_path)}")
            
            try:
                table = self._read_table(random_path, columns=columns)
                tables.append(table)
            except Exception as e:
                self.logger.warning(f"Failed to load random catalogue {r}: {e}")
                # Try loading with available columns and assigning missing ones
                table = self._handle_missing_random_columns(random_path, columns, galaxy_type, version)
                tables.append(table)
        
        table_r = vstack(tables)
        
        # Convert to dsigma table
        table_r = dsigma_table(
            table_r, 'lens', z='Z', ra='RA', dec='DEC', w_sys='WEIGHT',
            verbose=self.output_config.verbose
        )
        
        return table_r
    
    def _handle_missing_random_columns(self, random_path: str, needed_columns: List[str], galaxy_type: str, version: str) -> Table:
        """Handle missing columns in random catalogues."""
        # Get available columns
        with fits.open(random_path) as hdul:
            available_columns = hdul[1].columns.names
        
        columns_to_read = [col for col in needed_columns if col in available_columns]
        columns_to_read.append("TARGETID_DATA")
        
        table_r = Table(fitsio.read(random_path, columns=columns_to_read))
        
        # Get missing columns from data catalogue
        missing_columns = [col for col in needed_columns if col not in available_columns]
        if missing_columns:
            self.logger.warning(f"Assigning columns {missing_columns} from data catalogue")
            
            # Load data catalogue with missing columns
            data_path = self.path_manager.get_lens_catalogue_file(
                galaxy_type, version, survey=None, is_random=False
            )
            
            data_columns = missing_columns + ["TARGETID"]
            tab_dat = self._read_table(data_path, columns=data_columns)
            tab_dat.rename_column('TARGETID', 'TARGETID_DATA')
            
            table_r = join(table_r, tab_dat, keys='TARGETID_DATA', join_type='left')
            table_r.remove_column('TARGETID_DATA')
        
        return table_r
    
    def _read_table(self, filename: str, columns: Optional[List[str]] = None) -> Table:
        """Read FITS table with optional column selection."""
        # Ensure filename is a string (in case it's a Path object)
        filename = str(filename)
        
        if columns is None:
            return Table.read(filename)
        
        with fits.open(filename, memmap=True) as hdul:
            available_columns = hdul[1].columns.names
            columns_to_read = [col for col in columns if col in available_columns]
            missing_columns = [col for col in columns if col not in available_columns]
            
            # Always include TARGETID if we have missing columns
            if missing_columns and "TARGETID" not in columns_to_read:
                columns_to_read.append("TARGETID")
            
            # For full HP map cuts, also load RA, DEC, PHOTSYS for systematic assignment
            if "_full_HPmapcut" in os.path.basename(filename):
                for key in ["RA", "DEC", "PHOTSYS"]:
                    if key not in columns_to_read:
                        columns_to_read.append(key)
            
            data = self._hdul_to_table(hdul, columns_to_read)
        
        # Handle missing columns
        if missing_columns:
            self.logger.info(f"Columns {missing_columns} not available in file {filename}")
            data = self._assign_missing_columns(data, missing_columns, filename)
        
        return data
    
    def _hdul_to_table(self, hdul, columns: List[str]) -> Table:
        """Convert HDU to astropy Table."""
        data = Table()
        for col in columns:
            data.add_column(hdul[1].data.field(col), name=col)
        return data
    
    def _assign_missing_columns(self, data: Table, missing_columns: List[str], filename: str) -> Table:
        """Assign missing columns using various methods."""
        # Ensure filename is a string (in case it's a Path object)
        filename = str(filename)
        
        if "_clustering" in os.path.basename(filename):
            # Try to match from full_HPmapcut file
            full_file = filename.replace("_clustering", "_full_HPmapcut")
            if os.path.exists(full_file):
                self.logger.info("Trying to match from full_HPmapcut file")
                tab_full = self._read_table(full_file, columns=["TARGETID"] + missing_columns)
                
                # Ensure unique TARGETIDs
                if len(np.unique(tab_full["TARGETID"])) != len(tab_full["TARGETID"]):
                    self.logger.warning("TARGETID not unique in full_HPmapcut file, removing duplicates")
                    tab_full = self._cut_to_unique_targetid(tab_full)
                
                data = join(data, tab_full, keys="TARGETID", join_type="inner")
            else:
                # Assign via healpix maps
                data = self._assign_from_healpix_maps(data, missing_columns, filename)
        else:
            # Assign via healpix maps
            data = self._assign_from_healpix_maps(data, missing_columns, filename)
        
        return data
    
    def _assign_from_healpix_maps(self, data: Table, missing_columns: List[str], filename: str) -> Table:
        """Assign missing columns from healpix maps."""
        # Ensure filename is a string (in case it's a Path object)
        filename = str(filename)
        
        self.logger.info(f"Assigning {missing_columns} from healpix maps")
        
        # Extract galaxy type and version from filename
        galaxy_type = os.path.basename(filename).split("_")[0]
        if galaxy_type == "BGS":
            galaxy_type = "BGS_BRIGHT"
        
        version_parts = filename.split("/v1.")
        if len(version_parts) > 1:
            version = version_parts[1].split("/")[0]
            version = f"v1.{version}"
        else:
            version = self.lens_config.get_catalogue_version()
        
        for col in missing_columns:
            data[col] = self._assign_systematic_property(data, galaxy_type, col, version)
        
        return data
    
    def _assign_systematic_property(self, table: Table, galaxy_type: str, sysname: str, version: str, nside: int = 256) -> np.ndarray:
        """Assign systematic property from healpix maps."""
        cache_key = f"{galaxy_type}_{sysname}_{version}"
        
        if cache_key in self._systematic_cache:
            systable = self._systematic_cache[cache_key]
        else:
            # Load systematic table
            hpmaps_path = f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version}/hpmaps/"
            fname_template = f"{hpmaps_path}{galaxy_type}_mapprops_healpix_nested_nside256.fits"
            
            try:
                systable = Table.read(fname_template)
                self._systematic_cache[cache_key] = systable
            except Exception:
                # Try north/south split
                result = np.zeros(len(table))
                fname_template_split = fname_template.replace(".fits", "_{}.fits")
                
                for reg in ["N", "S"]:
                    try:
                        systable_reg = Table.read(fname_template_split.format(reg))
                        mask = (table["PHOTSYS"] == reg)
                        phi, theta = np.radians(table['RA'][mask]), np.radians(90. - table['DEC'][mask])
                        ipix = hp.ang2pix(nside, theta, phi, nest=True)
                        result[mask] = systable_reg[sysname][ipix]
                    except Exception as e:
                        self.logger.warning(f"Failed to load systematic map for {reg}: {e}")
                
                return result
        
        phi, theta = np.radians(table['RA']), np.radians(90. - table['DEC'])
        ipix = hp.ang2pix(nside, theta, phi, nest=True)
        return systable[sysname][ipix]
    
    def _cut_to_unique_targetid(self, table: Table) -> Table:
        """Remove duplicate TARGETIDs keeping first occurrence."""
        targetids = table['TARGETID'].data
        _, unique_indices = np.unique(targetids, return_index=True)
        unique_indices.sort()
        return table[unique_indices]
    
    def _read_nofz(self, survey: str) -> Table:
        """Read n(z) file for survey."""
        nofz_path = self.path_manager.get_nofz_file(survey)
        
        n_tomo_bins = self.source_config.get_n_tomographic_bins(survey)
        
        self.logger.info(f"Reading {nofz_path} from {self._get_last_mtime(nofz_path)}")
        
        # Load first tomographic bin to get redshift grid
        dat = np.loadtxt(nofz_path, skiprows=1)
        
        tab = Table()
        tab.add_column(dat[:, 0], name='z')
        
        # Load all tomographic bins
        nofz_data = []
        for i in range(n_tomo_bins):
            bin_path = str(nofz_path).replace("_tom1_", f"_tom{i+1}_")
            bin_data = np.loadtxt(bin_path, skiprows=1)
            nofz_data.append(bin_data[:, 1])
        
        tab.add_column(np.vstack(nofz_data).T, name='n')
        
        return tab
    
    def _build_stacking_kwargs(self, settings: Dict[str, bool]) -> Dict[str, bool]:
        """Build stacking kwargs from survey settings."""
        # Map config keys to stacking kwargs keys
        key_mapping = {
            'photo_z_dilution_correction': 'photo_z_dilution_correction',
            'boost_correction': 'boost_correction', 
            'scalar_shear_response_correction': 'scalar_shear_response_correction',
            'matrix_shear_response_correction': 'matrix_shear_response_correction',
            'shear_responsivity_correction': 'shear_responsivity_correction',
            'hsc_selection_bias_correction': 'hsc_selection_bias_correction',
            'hsc_additive_shear_bias_correction': 'hsc_additive_shear_bias_correction',
            'hsc_y3_selection_bias_correction': 'hsc_y3_selection_bias_correction',
            'random_subtraction': 'random_subtraction',
        }
        
        stacking_kwargs = {}
        for config_key, stack_key in key_mapping.items():
            if config_key in settings:
                stacking_kwargs[stack_key] = settings[config_key]
        
        return stacking_kwargs
    
    def _get_last_mtime(self, path: str) -> datetime:
        """Get last modification time of file."""
        if os.path.exists(path):
            return datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.now()
    
    def _apply_magnitude_cuts(self, table: Table) -> Table:
        """Apply magnitude cuts to catalogue."""
        # Placeholder - implement magnitude cutting logic here
        # This would use the magnitude_column from lens_config
        self.logger.info("Applying magnitude cuts...")
        return table
    