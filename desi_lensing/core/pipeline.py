"""Main lensing pipeline orchestrator."""

# Fix OpenBLAS threading issues on HPC systems before importing numerical libraries
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "20")
os.environ.setdefault("OMP_NUM_THREADS", "20")

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from copy import deepcopy

from astropy.table import Table, join,vstack
import numpy as np

from ..config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig, PathManager
from ..data.loader import DataLoader
from .computation import DeltaSigmaComputation, GammaTComputation
from ..utils.computation_utils import blind_dv


class LensingPipeline:
    """Main pipeline for lensing computations."""
    
    def __init__(
        self,
        computation_config: ComputationConfig,
        lens_config: LensGalaxyConfig,
        source_config: SourceSurveyConfig,
        output_config: OutputConfig,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the lensing pipeline."""
        self.computation_config = computation_config
        self.lens_config = lens_config
        self.source_config = source_config
        self.output_config = output_config
        
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Initialize path manager
        self.path_manager = PathManager(output_config, source_config)
        
        # Initialize data loader
        self.data_loader = DataLoader(
            lens_config, source_config, output_config, self.path_manager, logger
        )
        
        # Load magnification bias data
        self.magnification_bias_data = self._load_magnification_bias()
        
        # Create output directories
        self._setup_output_directories()
    
    def _load_magnification_bias(self) -> Dict[str, Any]:
        """Load magnification bias data."""
        version = self.lens_config.get_catalogue_version()
        mag_bias_file = self.output_config.get_magnification_bias_file(version)
        
        try:
            with open(mag_bias_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Magnification bias file not found: {mag_bias_file}")
            return {}
    
    def _setup_output_directories(self) -> None:
        """Create output directories."""
        version = self.lens_config.get_catalogue_version()
        self.output_config.create_output_directories(version, self.source_config.surveys)
    
    def _create_computation(self, statistic: str):
        """Create computation object for a given statistic."""
        if statistic == "deltasigma":
            return DeltaSigmaComputation(
                self.computation_config, self.lens_config, 
                self.source_config, self.output_config, self.logger
            )
        elif statistic == "gammat":
            return GammaTComputation(
                self.computation_config, self.lens_config,
                self.source_config, self.output_config, self.logger
            )
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    def run(self) -> None:
        """Run the complete lensing pipeline."""
        self.logger.info("Starting DESI lensing pipeline")
        self.logger.info(f"Galaxy type: {self.lens_config.galaxy_type}")
        self.logger.info(f"Source surveys: {self.source_config.surveys}")
        self.logger.info(f"Statistics: {self.computation_config.statistics}")
        self.logger.info(f"Tomography: {self.computation_config.tomography}")
        if self.computation_config.bmodes:
            self.logger.info("B-modes enabled: will rotate source galaxies by 45 degrees")
        
        # Load lens catalogues once
        version = self.lens_config.get_catalogue_version()
        
        # Process each source survey
        for source_survey in self.source_config.surveys:
            self.logger.info(f"Processing source survey: {source_survey}")
            
            # Handle DECADE specially: process NGC and SGC separately then join
            if source_survey.upper() == "DECADE":
                self._process_decade_survey()
                continue
            
            # Regular survey processing
            self._process_single_survey(source_survey)
        
        self.logger.info("DESI lensing pipeline completed")
    
    def _process_single_survey(self, source_survey: str) -> None:
        """Process a single source survey through the full pipeline."""
        # Load lens catalogues
        table_l, table_r = self.data_loader.load_lens_catalogues(source_survey)
        
        # Apply randoms subsampling if requested
        if self.lens_config.randoms_ratio >= 0 and table_r is not None:
            table_r = self._subsample_randoms(table_l, table_r, source_survey)
        
        # Load source catalogue
        table_s, precompute_kwargs, stacking_kwargs = self.data_loader.load_source_catalogue(
            source_survey, self.lens_config.galaxy_type
        )
        
        # Process statistics for this survey
        self._process_survey_statistics(
            source_survey, table_l, table_r, table_s, 
            precompute_kwargs, stacking_kwargs
        )
    
    def _process_survey_statistics(
        self, source_survey: str, table_l: Table, table_r: Optional[Table], 
        table_s: Table, precompute_kwargs: Dict[str, Any], 
        stacking_kwargs: Dict[str, Any]
    ) -> None:
        """Process all statistics and lens bins for a given survey."""
        # Process each statistic
        for statistic in self.computation_config.statistics:
            self.logger.info(f"Computing statistic: {statistic}")
            
            # Process each lens redshift bin
            for lens_bin in range(self.lens_config.get_n_lens_bins()):
                z_min = self.lens_config.z_bins[lens_bin]
                z_max = self.lens_config.z_bins[lens_bin + 1]
                
                # Select lenses in this redshift bin
                mask_l = ((z_min <= table_l['z']) & (table_l['z'] < z_max))
                table_l_part = table_l[mask_l]
                
                if table_r is not None:
                    mask_r = ((z_min <= table_r['z']) & (table_r['z'] < z_max))
                    table_r_part = table_r[mask_r]
                else:
                    table_r_part = None
                
                self.logger.info(f"Lens bin {lens_bin+1}/{self.lens_config.get_n_lens_bins()}: "
                               f"z=[{z_min:.2f}, {z_max:.2f}), {len(table_l_part)} lenses")
                
                # Create computation object to compute final statistic
                computation = self._create_computation(statistic)
                
                # Get magnification bias for this bin
                alpha_l = self._get_magnification_bias(lens_bin)
                
                if self.computation_config.tomography:
                    self._process_tomographic(
                        computation, table_l_part, table_r_part, table_s,
                        source_survey, statistic, lens_bin, z_min, z_max,
                        precompute_kwargs, stacking_kwargs, alpha_l
                    )
                else:
                    self._process_non_tomographic(
                        computation, table_l_part, table_r_part, table_s,
                        source_survey, statistic, lens_bin, z_min, z_max,
                        precompute_kwargs, stacking_kwargs, alpha_l
                    )
    
    def _process_decade_survey(self) -> None:
        """Process DECADE survey by computing NGC and SGC separately, then joining."""
        self.logger.info("Processing DECADE survey (NGC + SGC)")
        
        if not self.computation_config.tomography:
            self.logger.warning("Non-tomographic analysis not yet implemented for DECADE")
            return
        
        # Check if we need to compute anything from scratch
        # If any combination is missing, process both caps completely
        need_computation = self._check_decade_needs_computation()
        
        if need_computation:
            self.logger.info("Some DECADE precomputed tables are missing - computing NGC and SGC")
            for cap in ["NGC", "SGC"]:
                cap_survey = f"DECADE_{cap}"
                self.logger.info(f"Processing {cap_survey}")
                self._process_single_survey(cap_survey)
        
        # Now join precomputed tables and save combined results
        for statistic in self.computation_config.statistics:
            self.logger.info(f"Joining and saving DECADE results for {statistic}")
            
            for lens_bin in range(self.lens_config.get_n_lens_bins()):
                z_min = self.lens_config.z_bins[lens_bin]
                z_max = self.lens_config.z_bins[lens_bin + 1]
                n_source_bins = self.source_config.get_n_tomographic_bins("DECADE")
                
                for source_bin in range(n_source_bins):
                    self.logger.info(f"Lens bin {lens_bin+1}/{self.lens_config.get_n_lens_bins()}: "
                                   f"z=[{z_min:.2f}, {z_max:.2f}), source bin {source_bin+1}/{n_source_bins}")
                    
                    # Try to load precomputed combined tables
                    stacking_kwargs = {'boost_correction': False}  # Default
                    result = self._try_load_precomputed(
                        statistic, "DECADE", lens_bin, z_min, z_max, source_bin, stacking_kwargs
                    )
                    
                    if result is None:
                        # Join NGC and SGC precomputed tables
                        result = self._join_decade_precomputed_tables(
                            statistic, lens_bin, z_min, z_max, source_bin, stacking_kwargs
                        )
                    
                    # Save combined DECADE result
                    if result is not None:
                        self._save_results(
                            result, statistic, "DECADE", lens_bin, z_min, z_max,
                            source_bin, stacking_kwargs.get('boost_correction', False)
                        )
                    else:
                        self.logger.warning(f"Failed to process DECADE for {statistic}, "
                                          f"lens bin {lens_bin}, source bin {source_bin}")
    
    def _check_decade_needs_computation(self) -> bool:
        """Check if any DECADE precomputed tables are missing."""
        version = self.lens_config.get_catalogue_version()
        
        for statistic in self.computation_config.statistics:
            for lens_bin in range(self.lens_config.get_n_lens_bins()):
                z_min = self.lens_config.z_bins[lens_bin]
                z_max = self.lens_config.z_bins[lens_bin + 1]
                n_source_bins = self.source_config.get_n_tomographic_bins("DECADE")
                
                for source_bin in range(n_source_bins):
                    # Check if NGC and SGC precomputed tables exist
                    for cap in ["NGC", "SGC"]:
                        cap_survey = f"DECADE_{cap}"
                        paths = self.path_manager.get_precomputed_files(
                            statistic=statistic,
                            galaxy_type=self.lens_config.galaxy_type,
                            version=version,
                            survey=cap_survey,
                            z_min=z_min,
                            z_max=z_max,
                            lens_bin=source_bin,
                            boost_correction=False,
                            is_bmode=self.computation_config.bmodes
                        )
                        
                        # If any file is missing, we need to compute
                        if not all(p.exists() for p in [paths['lens'], paths['random']]):
                            self.logger.info(f"Missing precomputed tables for {cap_survey}")
                            return True
        
        self.logger.info("All DECADE_NGC and DECADE_SGC precomputed tables exist")
        return False
    
    def _get_magnification_bias(self, lens_bin: int) -> Optional[float]:
        """Get magnification bias for a lens bin."""
        try:
            galaxy_type = self.lens_config.galaxy_type
            return self.magnification_bias_data[galaxy_type]['alphas'][lens_bin]
        except (KeyError, IndexError):
            self.logger.warning(f"Magnification bias not found for {galaxy_type} bin {lens_bin}, only found for {self.magnification_bias_data}")
            return None
    
    def _subsample_randoms(self, table_l, table_r, source_survey):
        """Subsample randoms table to the specified ratio."""
        n_lenses = len(table_l)
        target_n_randoms = int(self.lens_config.randoms_ratio * n_lenses)
        n_randoms = len(table_r)
        
        # Check if we have enough randoms
        if n_randoms < target_n_randoms:
            raise ValueError(
                f"Insufficient randoms for survey {source_survey}: "
                f"need {target_n_randoms} (ratio {self.lens_config.randoms_ratio} * {n_lenses} lenses), "
                f"but only have {n_randoms} randoms"
            )
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Select random subset
        indices = np.random.choice(n_randoms, size=target_n_randoms, replace=False)
        table_r_subsampled = table_r[indices]
        
        self.logger.info(
            f"Subsampled randoms for {source_survey}: "
            f"{n_randoms} -> {target_n_randoms} "
            f"(ratio {self.lens_config.randoms_ratio}, {n_lenses} lenses)"
        )
        
        return table_r_subsampled
    
    def _process_tomographic(
        self, computation, table_l_part, table_r_part, table_s,
        source_survey, statistic, lens_bin, z_min, z_max,
        precompute_kwargs, stacking_kwargs, alpha_l
    ) -> None:
        """Process tomographic analysis."""
        
        n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
        
        for source_bin in range(n_source_bins):
            self.logger.info(f"Source bin {source_bin+1}/{n_source_bins}")
            
            # Select sources in this tomographic bin
            select = (table_s['z_bin'] == source_bin)
            table_s_part = table_s[select]
            
            if len(table_s_part) == 0:
                self.logger.warning(f"No sources in tomographic bin {source_bin}")
                continue
            
            # Handle special case for HSC Y1
            precompute_kwargs_tomo = self._handle_hsc_y1_tomography(
                source_survey, source_bin, precompute_kwargs, table_l_part
            )
            
            if precompute_kwargs_tomo is None:
                continue
            
            # Check if precomputed results exist
            result = self._try_load_precomputed(
                statistic, source_survey, lens_bin, z_min, z_max, source_bin, stacking_kwargs
            )
            
            if result is None:
                # Compute from scratch
                result = self._compute_lensing_signal(
                    computation, table_l_part, table_r_part, table_s_part,
                    precompute_kwargs_tomo, stacking_kwargs, alpha_l,
                    statistic, source_survey, lens_bin, z_min, z_max, source_bin
                )
            
            if result is not None:
                self._save_results(
                    result, statistic, source_survey, lens_bin, z_min, z_max,
                    source_bin, stacking_kwargs.get('boost_correction', False)
                )
    
    def _process_non_tomographic(
        self, computation, table_l_part, table_r_part, table_s,
        source_survey, statistic, lens_bin, z_min, z_max,
        precompute_kwargs, stacking_kwargs, alpha_l
    ) -> None:
        """Process non-tomographic analysis."""
        
        # Get allowed source bins for this lens-source combination
        allowed_bins = self._get_allowed_source_bins(source_survey, lens_bin)
        
        # Select sources from allowed bins
        select = np.zeros(len(table_s), dtype=bool)
        for z_bin in allowed_bins:
            select |= (table_s['z_bin'] == z_bin)
        
        if not np.any(select):
            self.logger.warning(f"No allowed source bins for lens bin {lens_bin}")
            return
        
        table_s_part = table_s[select]
        self.logger.info(f"Using {np.sum(select)}/{len(table_s)} sources from bins {allowed_bins}")
        
        # Handle special case for HSC Y1
        precompute_kwargs_tomo = self._handle_hsc_y1_tomography(
            source_survey, allowed_bins, precompute_kwargs, table_l_part
        )
        
        if precompute_kwargs_tomo is None:
            return
        
        # Check if precomputed results exist
        result = self._try_load_precomputed(
            statistic, source_survey, lens_bin, z_min, z_max, None, stacking_kwargs
        )
        
        if result is None:
            # Compute from scratch
            result = self._compute_lensing_signal(
                computation, table_l_part, table_r_part, table_s_part,
                precompute_kwargs_tomo, stacking_kwargs, alpha_l,
                statistic, source_survey, lens_bin, z_min, z_max, None
            )
        
        if result is not None:
            self._save_results(
                result, statistic, source_survey, lens_bin, z_min, z_max,
                None, stacking_kwargs.get('boost_correction', False)
            )
    
    def _handle_hsc_y1_tomography(self, source_survey, source_bins, precompute_kwargs, table_l_part):
        """Handle special tomography for HSC Y1."""
        if source_survey.upper() != 'HSCY1':
            return deepcopy(precompute_kwargs)
        
        HSC_bins = [0.3, 0.6, 0.9, 1.2, 1.5]
        
        if isinstance(source_bins, int):
            z_s_min = HSC_bins[source_bins]
            z_s_max = HSC_bins[source_bins + 1]
        else:
            z_s_min = HSC_bins[np.min(source_bins)]
            z_s_max = HSC_bins[np.max(source_bins) + 1]
        
        precompute_kwargs_tomo = deepcopy(precompute_kwargs)
        table_c = precompute_kwargs_tomo['table_c']
        table_c_part = table_c[(z_s_min <= table_c['z']) & (table_c['z'] < z_s_max)]
        precompute_kwargs_tomo['table_c'] = table_c_part
        
        if np.amax(table_l_part["z"]) >= np.amax(table_c_part['z']):
            self.logger.warning("HSC: f_bias is undefined for this lens-source combination. Skipping.")
            return None
        
        return precompute_kwargs_tomo
    
    def _get_allowed_source_bins(self, source_survey, lens_bin):
        """Get allowed source bins for non-tomographic analysis."""
        # This would import from the plotting utilities in the original code
        # For now, return all available bins
        n_bins = self.source_config.get_n_tomographic_bins(source_survey)
        return list(range(n_bins))
    
    def _try_load_precomputed(self, statistic, source_survey, lens_bin, z_min, z_max, source_bin, stacking_kwargs):
        """Try to load precomputed results."""
        if not self.output_config.save_precomputed:
            return None
        
        version = self.lens_config.get_catalogue_version()
        
        # Get precomputed file paths with B-mode flag
        precomputed_paths = self.path_manager.get_precomputed_files(
            statistic=statistic,
            galaxy_type=self.lens_config.galaxy_type,
            version=version,
            survey=source_survey,
            z_min=z_min,
            z_max=z_max,
            lens_bin=source_bin,
            boost_correction=stacking_kwargs.get('boost_correction', False),
            is_bmode=self.computation_config.bmodes
        )
        
        # Check if all required files exist
        if all(path.exists() for path in precomputed_paths.values()):
            self.logger.info(f"Loading precomputed results from {precomputed_paths['lens'].parent}")
            
            try:
                # Load precomputed tables
                table_l_precomputed = Table.read(str(precomputed_paths['lens']), format='hdf5')
                
                # Load random table if it exists
                table_r_precomputed = None
                if precomputed_paths['random'].exists():
                    table_r_precomputed = Table.read(str(precomputed_paths['random']), format='hdf5')
                
                # Create computation object to compute final statistic
                computation = self._create_computation(statistic)
                
                # Get magnification bias for this bin
                alpha_l = self._get_magnification_bias(lens_bin)
                
                # Compute final statistic from precomputed tables
                result, covariance_matrix = computation.compute_statistic(
                    table_l_precomputed, table_r_precomputed, alpha_l=alpha_l, **stacking_kwargs
                )
                
                self.logger.info("Successfully loaded and computed from precomputed tables")
                return result, covariance_matrix
                
            except Exception as e:
                self.logger.error(f"Failed to load precomputed results: {e}")
                self.logger.info("Will compute from scratch instead")
                return None
        else:
            self.logger.debug(f"Precomputed files not found for {statistic} lens bin {lens_bin}, source bin {source_bin}")
            return None
    
    def _join_decade_precomputed_tables(
        self, statistic, lens_bin, z_min, z_max, source_bin, stacking_kwargs
    ):
        """Join precomputed tables from DECADE_NGC and DECADE_SGC."""
        self.logger.info("Joining DECADE_NGC and DECADE_SGC precomputed tables")
        
        version = self.lens_config.get_catalogue_version()
        
        # Load NGC precomputed tables
        ngc_paths = self.path_manager.get_precomputed_files(
            statistic=statistic,
            galaxy_type=self.lens_config.galaxy_type,
            version=version,
            survey="DECADE_NGC",
            z_min=z_min,
            z_max=z_max,
            lens_bin=source_bin,
            boost_correction=stacking_kwargs.get('boost_correction', False),
            is_bmode=self.computation_config.bmodes
        )
        
        # Load SGC precomputed tables
        sgc_paths = self.path_manager.get_precomputed_files(
            statistic=statistic,
            galaxy_type=self.lens_config.galaxy_type,
            version=version,
            survey="DECADE_SGC",
            z_min=z_min,
            z_max=z_max,
            lens_bin=source_bin,
            boost_correction=stacking_kwargs.get('boost_correction', False),
            is_bmode=self.computation_config.bmodes
        )
        
        # Check if both sets of files exist
        if not (all(p.exists() for p in ngc_paths.values()) and all(p.exists() for p in sgc_paths.values())):
            self.logger.error("DECADE NGC and/or SGC precomputed tables not found")
            # info about files that are missing
            for path in ngc_paths.values():
                if not path.exists():
                    self.logger.info(f"NGC precomputed table not found: {path}")
            for path in sgc_paths.values():
                if not path.exists():
                    self.logger.info(f"SGC precomputed table not found: {path}")
            return None
        
        try:
            # Load NGC tables
            table_l_ngc = Table.read(str(ngc_paths['lens']), format='hdf5')
            table_r_ngc = None
            if ngc_paths['random'].exists():
                table_r_ngc = Table.read(str(ngc_paths['random']), format='hdf5')
            
            # Load SGC tables
            table_l_sgc = Table.read(str(sgc_paths['lens']), format='hdf5')
            table_r_sgc = None
            if sgc_paths['random'].exists():
                table_r_sgc = Table.read(str(sgc_paths['random']), format='hdf5')
            
            # Join tables using vstack
            # Use metadata_conflicts='silent' and then manually restore NGC metadata
            table_l_combined = vstack([table_l_ngc, table_l_sgc], metadata_conflicts='silent')
            # Restore metadata from NGC table (they should be identical)
            table_l_combined.meta = dict(table_l_ngc.meta)
            
            table_r_combined = None
            if table_r_ngc is not None and table_r_sgc is not None:
                table_r_combined = vstack([table_r_ngc, table_r_sgc], metadata_conflicts='silent')
                # Restore metadata from NGC table
                table_r_combined.meta = dict(table_r_ngc.meta)
            
            self.logger.info(f"Successfully joined DECADE tables: {len(table_l_ngc)} NGC + {len(table_l_sgc)} SGC = {len(table_l_combined)} total lenses")
            
            # Save combined tables with DECADE survey name
            # Do not save precomputed tables for DECADE as they are easy to recreate
            # if self.output_config.save_precomputed:
            #     self._save_precomputed_tables(
            #         table_l_combined, table_r_combined,
            #         statistic, "DECADE", lens_bin, z_min, z_max, source_bin, stacking_kwargs
            #     )
            
            # Compute final statistic from combined tables
            computation = self._create_computation(statistic)
            alpha_l = self._get_magnification_bias(lens_bin)
            
            result, covariance_matrix = computation.compute_statistic(
                table_l_combined, table_r_combined, alpha_l=alpha_l, **stacking_kwargs
            )
            
            self.logger.info("Successfully computed from joined DECADE precomputed tables")
            return result, covariance_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to join DECADE precomputed tables: {e}")
            return None
    
    def _compute_lensing_signal(
        self, computation, table_l_part, table_r_part, table_s_part,
        precompute_kwargs, stacking_kwargs, alpha_l,
        statistic, source_survey, lens_bin, z_min, z_max, source_bin
    ):
        """Compute the lensing signal."""
        try:
            # Precompute
            table_l_precomputed, table_r_precomputed = computation.precompute(
                table_l_part, table_r_part, table_s_part, **precompute_kwargs
            )
            
            if table_l_precomputed is None:
                return None
            
            # Save precomputed tables if requested
            if self.output_config.save_precomputed:
                self._save_precomputed_tables(
                    table_l_precomputed, table_r_precomputed,
                    statistic, source_survey, lens_bin, z_min, z_max, source_bin, stacking_kwargs
                )
            
            # Compute statistic
            result, covariance_matrix = computation.compute_statistic(
                table_l_precomputed, table_r_precomputed, alpha_l=alpha_l, **stacking_kwargs
            )
            
            return result, covariance_matrix
            
        except Exception as e:
            self.logger.error(f"Computation failed: {e}")
            return None
    
    def _save_precomputed_tables(
        self, table_l_precomputed, table_r_precomputed,
        statistic, source_survey, lens_bin, z_min, z_max, source_bin, stacking_kwargs
    ):
        """Save precomputed tables with B-mode-specific naming."""
        version = self.lens_config.get_catalogue_version()
        
        # Get precomputed file paths with B-mode flag
        precomputed_paths = self.path_manager.get_precomputed_files(
            statistic=statistic,
            galaxy_type=self.lens_config.galaxy_type,
            version=version,
            survey=source_survey,
            z_min=z_min,
            z_max=z_max,
            lens_bin=source_bin,
            boost_correction=stacking_kwargs.get('boost_correction', False),
            is_bmode=self.computation_config.bmodes
        )
        
        # Ensure directory exists
        precomputed_paths['lens'].parent.mkdir(parents=True, exist_ok=True)
        
        # Save tables (implementation would save to HDF5/pickle format)
        self.logger.info(f"Saving precomputed tables to {precomputed_paths['lens'].parent}")
        
        table_l_precomputed.write(str(precomputed_paths['lens']), format='hdf5', overwrite=True)
        if table_r_precomputed is not None:
            table_r_precomputed.write(str(precomputed_paths['random']), format='hdf5', overwrite=True)
    
    def _save_results(self, result_tuple, statistic, source_survey, lens_bin, z_min, z_max, source_bin, boost_correction):
        """Save computation results."""
        if result_tuple is None:
            return
        
        result, covariance_matrix = result_tuple
        version = self.lens_config.get_catalogue_version()
        
        # Apply blinding if requested
        if self.output_config.apply_blinding:
            result = blind_dv(result, source_survey, self.lens_config.galaxy_type, lens_bin)
        
        # Generate output filepath using consolidated method
        output_path = self.output_config.get_filepath(
            statistic=statistic,
            galaxy_type=self.lens_config.galaxy_type,
            version=version,
            survey=source_survey,
            z_min=z_min,
            z_max=z_max,
            source_bin=source_bin,
            is_bmode=self.computation_config.bmodes,
            boost_correction=boost_correction,
            file_type="measurement"
        )
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save result
        result.write(str(output_path), overwrite=True)
        
        # Save covariance matrix
        if self.output_config.save_covariance:
            cov_path = self.output_config.get_filepath(
                statistic=statistic,
                galaxy_type=self.lens_config.galaxy_type,
                version=version,
                survey=source_survey,
                z_min=z_min,
                z_max=z_max,
                source_bin=source_bin,
                is_bmode=self.computation_config.bmodes,
                boost_correction=boost_correction,
                file_type="covariance"
            )
            np.savetxt(str(cov_path), covariance_matrix)
        
        self.logger.info(f"Saved results to: {output_path}") 