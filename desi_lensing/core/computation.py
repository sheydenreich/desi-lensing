"""Main computation classes for lensing statistics."""

from typing import Optional, Tuple, Dict, Any
from copy import deepcopy
import numpy as np
from astropy.table import Table
import multiprocessing

from dsigma.precompute import precompute
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling, compress_jackknife_fields
from dsigma.stacking import excess_surface_density, tangential_shear, lens_magnification_bias

from .base import BaseComputation
from ..utils.computation_utils import get_camb_results, is_table_masked


class LensingComputation(BaseComputation):
    """Generic lensing computation class."""
    
    def __init__(self, statistic: str, *args, **kwargs):
        """Initialize with specific statistic."""
        super().__init__(*args, **kwargs)
        self._statistic_name = statistic
    
    @property
    def statistic_name(self) -> str:
        """Return the statistic name."""
        return self._statistic_name
    
    def _rotate_shears_for_bmodes(self, table_s: Table) -> Table:
        """Rotate source galaxies by 45 degrees for B-mode analysis."""
        if not self.computation_config.bmodes:
            return table_s
        
        self.logger.info("Rotating shears by 45 degrees for B-mode analysis")
        
        # Make a copy to avoid modifying original table
        table_s_rotated = table_s.copy()
        
        # Store original shears
        e1_orig = table_s_rotated['e_1'].copy()
        e2_orig = table_s_rotated['e_2'].copy()
        
        self.logger.debug(f"Before rotation: e1={e1_orig[0]:.6f}, e2={e2_orig[0]:.6f}")
        
        # Apply 45-degree rotation: e1_new = -e2, e2_new = e1
        table_s_rotated['e_1'] = -e2_orig
        table_s_rotated['e_2'] = e1_orig
        
        self.logger.debug(f"After rotation: e1={table_s_rotated['e_1'][0]:.6f}, e2={table_s_rotated['e_2'][0]:.6f}")
        
        return table_s_rotated
    
    def precompute(
        self, 
        table_l: Table, 
        table_r: Optional[Table], 
        table_s: Table,
        **kwargs
    ) -> Tuple[Table, Optional[Table]]:
        """Precompute the lensing signal using dsigma."""
        self.validate_tables(table_l, table_r, table_s)
        
        # Check for masked tables
        if is_table_masked(table_l):
            raise ValueError("Lens table is masked")
        if table_r is not None and is_table_masked(table_r):
            raise ValueError("Random table is masked")
        
        # Apply B-mode rotation if requested
        table_s = self._rotate_shears_for_bmodes(table_s)
        
        # Setup bins and weighting
        bins = self.setup_bins()
        weighting = self.get_weighting()
        
        # Prepare precompute kwargs
        precompute_kwargs = {
            'n_jobs': self.computation_config.n_jobs or multiprocessing.cpu_count(),
            'comoving': self.computation_config.comoving,
            'cosmology': self.cosmology,
            'lens_source_cut': self.computation_config.lens_source_cut,
            'use_gpu': self.computation_config.use_gpu,
            'force_shared': self.computation_config.force_shared,
            'progress_bar': False,
            **kwargs
        }
        
        # Precompute for lenses
        if table_s is not None:
            precompute(table_l, table_s, bins, weighting=weighting, **precompute_kwargs)
            
            # Precompute for randoms if available
            if table_r is not None:
                precompute(table_r, table_s, bins, weighting=weighting, **precompute_kwargs)
        
        # Drop lenses/randoms with no nearby sources
        select_l = np.sum(table_l['sum 1'], axis=1) > 0
        
        if not np.all(select_l):
            self.logger.warning(
                f"Only {np.sum(select_l)} lenses are used out of {len(table_l)}"
            )
            table_l = table_l[select_l]
            
            if table_r is not None:
                select_r = np.sum(table_r['sum 1'], axis=1) > 0
                table_r = table_r[select_r]
        
        if not np.any(select_l):
            self.logger.warning("No lens-source pairs found")
            return None, None
        
        return table_l, table_r
    
    def compute_statistic(
        self, 
        table_l: Table, 
        table_r: Optional[Table],
        alpha_l: Optional[float] = None,
        **kwargs
    ) -> Tuple[Table, np.ndarray]:
        """Compute the lensing statistic and covariance matrix."""
        
        # Setup jackknife fields
        try:
            centers = compute_jackknife_fields(
                table_l, 
                self.computation_config.n_jackknife_fields,
                weights=np.sum(table_l['sum 1'], axis=1)
            )
        except (RuntimeError, ValueError) as e:
            self.logger.warning(f"Jackknife field computation failed: {e}")
            self.logger.info("Trying with increased distance threshold")
            centers = compute_jackknife_fields(
                table_l,
                self.computation_config.n_jackknife_fields,
                weights=np.sum(table_l['sum 1'], axis=1),
                distance_threshold=3.0
            )
        
        # Assign jackknife fields to randoms
        if table_r is not None:
            try:
                compute_jackknife_fields(table_r, centers)
            except (RuntimeError, ValueError) as e:
                self.logger.warning(f"Random jackknife assignment failed: {e}")
                compute_jackknife_fields(table_r, centers, distance_threshold=3.0)
        
        # Prepare stacking kwargs
        stacking_kwargs = kwargs.copy()
        if table_r is not None:
            try:
                stacking_kwargs['table_r'] = compress_jackknife_fields(table_r)
            except Exception as e:
                self.logger.warning(f"Jackknife compression failed: {e}")
                stacking_kwargs['table_r'] = table_r
        
        try:
            table_l_compressed = compress_jackknife_fields(table_l)
        except Exception as e:
            self.logger.warning(f"Jackknife compression failed: {e}")
            table_l_compressed = table_l
        
        # Remove photo_z_dilution_correction for gamma_t
        if self.statistic_name == "gammat":
            stacking_kwargs.pop('photo_z_dilution_correction', None)
        
        # Compute the statistic
        stacking_kwargs['return_table'] = True
        
        if self.statistic_name == "deltasigma":
            result = excess_surface_density(table_l_compressed, **stacking_kwargs)
        elif self.statistic_name == "gammat":
            result = tangential_shear(table_l_compressed, **stacking_kwargs)
        else:
            raise ValueError(f"Unknown statistic: {self.statistic_name}")
        
        # Compute covariance matrix
        stacking_kwargs['return_table'] = False
        try:
            if self.statistic_name == "deltasigma":
                covariance_matrix = jackknife_resampling(
                    excess_surface_density, table_l_compressed, **stacking_kwargs
                )
                error_key = 'ds_err'
            elif self.statistic_name == "gammat":
                covariance_matrix = jackknife_resampling(
                    tangential_shear, table_l_compressed, **stacking_kwargs
                )
                error_key = 'et_err'
        except Exception as e:
            self.logger.warning(f"Covariance computation failed: {e}")
            n_bins = len(self.setup_bins()) - 1
            covariance_matrix = np.full((n_bins, n_bins), np.nan)
            error_key = 'ds_err' if self.statistic_name == "deltasigma" else 'et_err'
        
        # Add error column
        result[error_key] = np.sqrt(np.diag(covariance_matrix))
        
        # Add magnification bias if requested
        if alpha_l is not None:
            camb_results = get_camb_results(table_l)
            shear = (self.statistic_name == "gammat")
            magnification_bias = lens_magnification_bias(
                table_l, alpha_l, camb_results, shear=shear
            )
            result.add_column(magnification_bias, name='magnification_bias')
        
        return result, covariance_matrix


class DeltaSigmaComputation(LensingComputation):
    """Delta Sigma computation class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__("deltasigma", *args, **kwargs)


class GammaTComputation(LensingComputation):
    """Gamma_t computation class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__("gammat", *args, **kwargs) 