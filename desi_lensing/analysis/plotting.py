"""
Plotting utilities for DESI lensing data vectors and analysis results.

This module provides plotting functionality migrated from the legacy plotting scripts,
adapted to work with the refactored DESI lensing pipeline architecture.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import matplotlib as mpl
from astropy.table import Table
from astropy import units as u

try:
    import skymapper as skm
    import healpy as hp
    FOOTPRINT_AVAILABLE = True
except ImportError:
    print("WARNING: skymapper and/or healpy not found. Footprint plotting will not be available.")
    FOOTPRINT_AVAILABLE = False

from ..config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig, OutputConfig, PlotConfig, AnalysisConfig
from ..data.loader import DataLoader
from ..utils.logging_utils import setup_logger

# Set matplotlib parameters for consistent plotting
plt.rcParams['errorbar.capsize'] = 1.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['lines.markersize'] = 1.5


class DataVectorPlotter:
    """
    Main class for plotting lensing data vectors and analysis results.
    
    This class provides functionality to plot:
    - Data vectors (tomographic and non-tomographic) for single or multiple galaxy types
    - B-mode diagnostics
    - Comparison plots between different configurations
    - Random tests
    - Source redshift slope analysis (systematic tests)
    - Split analyses for systematics testing
    
    The class supports both single galaxy type analysis (pass a single LensGalaxyConfig)
    and multi-galaxy type analysis (pass a list of LensGalaxyConfig objects).
    """
    
    def __init__(
        self,
        computation_config: ComputationConfig,
        lens_config: Union[LensGalaxyConfig, List[LensGalaxyConfig]],
        source_config: SourceSurveyConfig,
        output_config: OutputConfig,
        plot_config: PlotConfig,
        analysis_config: AnalysisConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the plotter with configuration objects.
        
        Parameters
        ----------
        computation_config : ComputationConfig
            Computation configuration
        lens_config : Union[LensGalaxyConfig, List[LensGalaxyConfig]]
            Single lens galaxy configuration or list of configurations for multi-galaxy plots
        source_config : SourceSurveyConfig
            Source survey configuration
        output_config : OutputConfig
            Output configuration
        plot_config : PlotConfig
            Plot configuration
        analysis_config : AnalysisConfig
            Analysis configuration for scale cuts and bin selections
        logger : Optional[logging.Logger]
            Logger instance
        """
        self.computation_config = computation_config
        self.source_config = source_config
        self.output_config = output_config
        self.plot_config = plot_config
        self.analysis_config = analysis_config
        
        self.logger = logger or setup_logger(self.__class__.__name__)
        
        # Normalize lens_config to list for unified handling
        if isinstance(lens_config, list):
            self._lens_configs = lens_config
        else:
            self._lens_configs = [lens_config]
        
        # Multi-galaxy support attributes
        self.galaxy_types = [cfg.galaxy_type for cfg in self._lens_configs]
        self.bin_layout = analysis_config.get_bin_layout_for_galaxy_types(self.galaxy_types)
        self.total_bins = analysis_config.get_total_bins_for_galaxy_types(self.galaxy_types)
        
        # Setup data loader for accessing results (use first lens config)
        from ..config.path_manager import PathManager
        self.path_manager = PathManager(output_config, source_config)
        self.data_loader = DataLoader(
            self._lens_configs[0], source_config, output_config, self.path_manager, logger
        )
        
        # Common plotting parameters
        self.color_list = ['blue', 'red', 'green', 'orange', 'purple']
        self.survey_names = source_config.surveys
        
        # Setup output directories
        self._setup_output_directories()
    
    @property
    def lens_config(self) -> LensGalaxyConfig:
        """
        Get the lens configuration (for backward compatibility).
        
        Returns the single lens config if only one is present.
        Raises ValueError if multiple configs are present.
        
        Returns
        -------
        LensGalaxyConfig
            The lens galaxy configuration
            
        Raises
        ------
        ValueError
            If multiple lens configs are present
        """
        if len(self._lens_configs) == 1:
            return self._lens_configs[0]
        raise ValueError(
            "Multiple lens configs present. Use _get_lens_config_for_galaxy_type() "
            "or access _lens_configs directly."
        )
    
    @property
    def lens_configs(self) -> List[LensGalaxyConfig]:
        """Get all lens configurations."""
        return self._lens_configs
    
    def _get_lens_config_for_galaxy_type(self, galaxy_type: str) -> Optional[LensGalaxyConfig]:
        """Get the lens config for a specific galaxy type."""
        for config in self._lens_configs:
            if config.galaxy_type == galaxy_type:
                return config
        return None
    
    def _setup_output_directories(self) -> None:
        """Create directories for saving plots."""
        # Use first lens config for versioning
        version = self._lens_configs[0].get_catalogue_version()
        plot_dir = self.plot_config.get_plot_output_dir(self.output_config.save_path) / version
        plot_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = plot_dir
    
    def _load_lensing_results_for_galaxy_type(
        self,
        galaxy_type: str,
        lens_bin: int,
        source_survey: str,
        source_bin: Optional[int] = None,
        statistic: str = "deltasigma"
    ) -> Optional[Table]:
        """Load lensing measurement results for a specific galaxy type and bin."""
        lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
        if lens_config is None:
            return None
        
        version = lens_config.get_catalogue_version()
        z_bins = lens_config.z_bins
        
        return self.output_config.load_lensing_results(
            version, galaxy_type, source_survey, lens_bin, z_bins,
            statistic, source_bin
        )
        
    def _load_lensing_results(
        self, 
        lens_bin: int, 
        source_survey: str,
        source_bin: Optional[int] = None,
        statistic: str = "deltasigma"
    ) -> Optional[Table]:
        """
        Load lensing measurement results for a given configuration.
        
        For backward compatibility with single galaxy type usage.
        For multi-galaxy, use _load_lensing_results_for_galaxy_type instead.
        """
        if len(self._lens_configs) == 1:
            return self._load_lensing_results_for_galaxy_type(
                self._lens_configs[0].galaxy_type, lens_bin, source_survey, source_bin, statistic
            )
        raise ValueError(
            "Multiple lens configs present. Use _load_lensing_results_for_galaxy_type() instead."
        )
    
    def _load_covariance_matrix_for_galaxy_type(
        self,
        galaxy_type: str,
        lens_bin: int,
        source_survey: str,
        source_bin: Optional[int] = None,
        statistic: str = "deltasigma"
    ) -> Optional[np.ndarray]:
        """Load covariance matrix for a specific galaxy type."""
        lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
        if lens_config is None:
            return None
        
        version = lens_config.get_catalogue_version()
        z_bins = lens_config.z_bins
        
        return self.output_config.load_covariance_matrix(
            version, galaxy_type, source_survey, lens_bin, z_bins,
            statistic, source_bin
        )
    
    def _load_covariance_matrix(
        self,
        lens_bin: int,
        source_survey: str, 
        source_bin: Optional[int] = None,
        statistic: str = "deltasigma"
    ) -> Optional[np.ndarray]:
        """
        Load covariance matrix for a given configuration.
        
        For backward compatibility with single galaxy type usage.
        For multi-galaxy, use _load_covariance_matrix_for_galaxy_type instead.
        """
        if len(self._lens_configs) == 1:
            return self._load_covariance_matrix_for_galaxy_type(
                self._lens_configs[0].galaxy_type, lens_bin, source_survey, source_bin, statistic
            )
        raise ValueError(
            "Multiple lens configs present. Use _load_covariance_matrix_for_galaxy_type() instead."
        )
    
    def _load_covariance_for_plotting_for_galaxy_type(
        self,
        galaxy_type: str,
        lens_bin: int,
        source_survey: str,
        source_bin: Optional[int] = None,
        statistic: str = "deltasigma",
        pure_noise: bool = False,
        use_theory: bool = True,
        n_lens_bins: Optional[int] = None,
        n_source_bins: Optional[int] = None,
        n_rp_bins: int = 15
    ) -> Optional[np.ndarray]:
        """
        Load covariance matrix for plotting for a specific galaxy type.
        
        Prefers theory covariance when available, falls back to jackknife.
        """
        lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
        if lens_config is None:
            return None
        
        version = lens_config.get_catalogue_version()
        z_bins = lens_config.z_bins
        
        # Try theory covariance first if requested
        if use_theory and source_bin is not None:
            full_cov = self.output_config.load_theory_covariance(
                galaxy_type, source_survey, statistic, pure_noise=pure_noise,
                z_bins=z_bins
            )
            
            if full_cov is not None:
                if n_lens_bins is None:
                    n_lens_bins = self.analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
                if n_source_bins is None:
                    n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
                
                bin_index = lens_bin * (n_source_bins * n_rp_bins) + source_bin * n_rp_bins
                total_expected = n_lens_bins * n_source_bins * n_rp_bins
                
                if full_cov.shape[0] != total_expected:
                    # Try to infer n_rp_bins from covariance size
                    n_rp_bins_inferred = full_cov.shape[0] // (n_lens_bins * n_source_bins)
                    if n_rp_bins_inferred * n_lens_bins * n_source_bins == full_cov.shape[0]:
                        n_rp_bins = n_rp_bins_inferred
                        bin_index = lens_bin * (n_source_bins * n_rp_bins) + source_bin * n_rp_bins
                    else:
                        self.logger.warning(
                            f"Theory covariance size {full_cov.shape[0]} doesn't match expected "
                            f"{total_expected} ({n_lens_bins}x{n_source_bins}x{n_rp_bins})"
                        )
                        if galaxy_type == "LRG":
                            if np.isclose(full_cov.shape[0]/3, total_expected/2):
                                self.logger.info("Reducing LRG theory covariance to the first 2 lens bins!")
                                full_cov = full_cov[:total_expected, :total_expected]
                        else:
                            full_cov = None
                
                if full_cov is not None:
                    start_idx = bin_index
                    end_idx = bin_index + n_rp_bins
                    
                    if end_idx <= full_cov.shape[0]:
                        return full_cov[start_idx:end_idx, start_idx:end_idx]
                    else:
                        self.logger.warning(f"Index out of bounds: {end_idx} > {full_cov.shape[0]}")
        
        # Fall back to jackknife covariance
        return self.output_config.load_covariance_matrix(
            version, galaxy_type, source_survey, lens_bin, z_bins,
            statistic, source_bin, pure_noise=pure_noise
        )
    
    def _load_covariance_for_plotting(
        self,
        lens_bin: int,
        source_survey: str,
        source_bin: Optional[int] = None,
        statistic: str = "deltasigma",
        pure_noise: bool = False,
        use_theory: bool = True,
        n_lens_bins: Optional[int] = None,
        n_source_bins: Optional[int] = None,
        n_rp_bins: int = 15
    ) -> Optional[np.ndarray]:
        """
        Load covariance matrix for plotting, preferring theory covariance.
        
        For backward compatibility with single galaxy type usage.
        For multi-galaxy, use _load_covariance_for_plotting_for_galaxy_type instead.
        """
        if len(self._lens_configs) == 1:
            return self._load_covariance_for_plotting_for_galaxy_type(
                self._lens_configs[0].galaxy_type, lens_bin, source_survey, source_bin,
                statistic, pure_noise, use_theory, n_lens_bins, n_source_bins, n_rp_bins
            )
        raise ValueError(
            "Multiple lens configs present. Use _load_covariance_for_plotting_for_galaxy_type() instead."
        )
    
    def _add_survey_colorbar_legend(
        self, 
        fig: plt.Figure, 
        axes: np.ndarray, 
        gs: gridspec.GridSpec, 
        color_list: List[str], 
        name_list: List[str],
        start: int = 0,
        skip: int = 1
    ) -> None:
        """
        Add colorbar legend for surveys, adapted from plotting_utilities.
        
        Parameters
        ----------
        fig : plt.Figure
            The figure object
        axes : np.ndarray
            Array of axes
        gs : gridspec.GridSpec
            GridSpec object
        color_list : List[str]
            List of colors for each survey
        name_list : List[str]
            List of survey names
        start : int
            Starting index for adding colorbars
        skip : int
            Skip interval for adding colorbars
        """
        if len(name_list) < len(color_list):
            local_color_list = color_list[:len(name_list)]
            local_name_list = name_list
        elif len(name_list) > len(color_list):
            local_color_list = color_list
            local_name_list = name_list[:len(color_list)]
        else:
            local_color_list = color_list
            local_name_list = name_list

        cmap = mpl.colors.ListedColormap(local_color_list)
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm._A = []

        tick_labels = local_name_list

        ticks = np.linspace(0, 1, len(tick_labels) + 1)
        ticks = 0.5 * (ticks[1:] + ticks[:-1])
        
        for i in range(axes.shape[0])[start::skip]:
            axes[i, -1] = fig.add_subplot(gs[i:i+skip, -1])

            cb = plt.colorbar(sm, cax=axes[i, -1], pad=0.0, ticks=ticks)
            cb.ax.set_yticklabels(tick_labels)
            cb.ax.minorticks_off()
            cb.ax.tick_params(size=0)
    
    def _get_rp_from_deg(
        self, 
        min_deg: float, 
        max_deg: float, 
        galaxy_type: str, 
        lens_bin: int
    ) -> Tuple[float, float]:
        """
        Convert angular scales to physical scales using lens redshift.
        
        Parameters
        ----------
        min_deg : float
            Minimum angle in degrees
        max_deg : float
            Maximum angle in degrees
        galaxy_type : str
            Galaxy type
        lens_bin : int
            Lens bin index
            
        Returns
        -------
        Tuple[float, float]
            (min_rp, max_rp) in Mpc/h
        """
        # Get lens redshift from configuration for this galaxy type
        lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
        if lens_config is None:
            self.logger.warning(f"No lens config found for galaxy type {galaxy_type}")
            return 0.0, 100.0  # Return safe defaults
        
        z_bins = lens_config.z_bins
        zlens = (z_bins[lens_bin] + z_bins[lens_bin + 1]) / 2.0
        
        # Use simple cosmology for scale conversion
        from astropy.cosmology import Planck18
        cosmo = Planck18.clone(H0=100.)
        
        dist = cosmo.comoving_transverse_distance(zlens).to(u.Mpc).value
        min_rp = dist * np.deg2rad(min_deg)
        max_rp = dist * np.deg2rad(max_deg)
        
        return min_rp, max_rp
    
    def _plot_scale_cuts(
        self, 
        axes: np.ndarray, 
        min_deg: float, 
        max_deg: float, 
        rp_pivot: float, 
        statistic: str = "deltasigma",
        shared_axes: bool = True,
        tomographic: bool = False,
        lens_bin_info: Optional[Dict[int, Tuple[str, int]]] = None
    ) -> None:
        """
        Plot scale cuts as gray shaded regions and pivot line.
        
        Parameters
        ----------
        axes : np.ndarray
            Array of matplotlib axes
        min_deg : float
            Minimum angular scale in degrees
        max_deg : float
            Maximum angular scale in degrees
        rp_pivot : float
            Pivot scale in Mpc/h (used for deltasigma only)
        statistic : str
            The statistic being plotted ('deltasigma' or 'gammat')
        shared_axes : bool
            Whether axes share x-limits
        tomographic : bool
            Whether this is a tomographic plot
        lens_bin_info : Optional[Dict[int, Tuple[str, int]]]
            Dictionary mapping column index to (galaxy_type, lens_bin) tuple.
            If None, creates mapping from self.bin_layout for multi-galaxy
            or assumes sequential bins for single galaxy type.
        """
        # Get current axis limits if shared
        if shared_axes:
            if axes.ndim == 1:
                axmin, axmax = axes[0].get_xlim()
            else:
                axmin, axmax = axes[0, 0].get_xlim()
        
        # Build lens_bin_info if not provided
        if lens_bin_info is None:
            lens_bin_info = {}
            for galaxy_type in self.galaxy_types:
                start_col, end_col = self.bin_layout[galaxy_type]
                n_lens_bins = self.analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
                for lens_bin in range(n_lens_bins):
                    col_idx = start_col + lens_bin
                    lens_bin_info[col_idx] = (galaxy_type, lens_bin)
        
        # Handle both 1D and 2D axes arrays
        if axes.ndim == 1:
            axes_list = [(i, axes[i]) for i in range(len(axes))]
        else:
            # For 2D, iterate over columns (excluding colorbar)
            n_rows, n_cols = axes.shape
            axes_list = []
            for col in range(n_cols):
                if axes[0, col] is not None:
                    axes_list.append((col, axes[0, col]))
        
        for col_idx, ax in axes_list:
            if ax is None:  # Skip None axes (e.g., colorbar column)
                continue
            
            # Get galaxy type and lens bin for this column
            if col_idx not in lens_bin_info:
                continue
            
            galaxy_type, lens_bin = lens_bin_info[col_idx]
            
            # Convert scales based on statistic type
            if statistic == "deltasigma":
                # X-axis is in Mpc/h, scale cuts are in degrees
                rpmin, rpmax = self._get_rp_from_deg(min_deg, max_deg, galaxy_type, lens_bin)
                # Add pivot line
                ax.axvline(rp_pivot, color='k', linestyle=':', alpha=0.7)
            else:  # gammat
                # X-axis is in arcmin, scale cuts are in degrees
                rpmin = min_deg * 60.0
                rpmax = max_deg * 60.0
            
            # Get current axis limits if not shared
            if not shared_axes:
                axmin, axmax = ax.get_xlim()
            
            # Add gray shaded regions outside analysis range
            ax.axvspan(axmin, rpmin, color='gray', alpha=0.3)
            ax.axvspan(rpmax, axmax, color='gray', alpha=0.3)
            
            # Restore axis limits if not shared
            if not shared_axes:
                ax.set_xlim(axmin, axmax)
        
        # Set shared axis limits if needed
        if shared_axes:
            if axes.ndim == 1:
                axes[0].set_xlim(axmin, axmax)
            else:
                axes[0, 0].set_xlim(axmin, axmax)
    
    def _initialize_gridspec_figure(
        self, 
        figsize: Tuple[float, float], 
        nrows: int, 
        ncols: int, 
        add_cbar: bool = True, 
        **kwargs
    ) -> Tuple[plt.Figure, np.ndarray, gridspec.GridSpec]:
        """
        Initialize figure with GridSpec layout including colorbar space.
        
        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size (width, height)
        nrows : int
            Number of rows
        ncols : int
            Number of columns
        add_cbar : bool
            Whether to add space for colorbar
            
        Returns
        -------
        Tuple[plt.Figure, np.ndarray, gridspec.GridSpec]
            Figure, axes array, and GridSpec object
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            nrows, ncols + 1 if add_cbar else ncols,
            width_ratios=[20] * ncols + [1] if add_cbar else [20] * ncols,
            **kwargs
        )
        
        axes = np.empty((nrows, ncols + 1 if add_cbar else ncols), dtype=object)
        
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(
                    gs[i, j], 
                    sharey=axes[i, 0] if j > 0 else None,
                    sharex=axes[0, j] if i > 0 else None
                )
                if j > 0:
                    plt.setp(axes[i, j].get_yticklabels(), visible=False)
                if i < nrows - 1:
                    plt.setp(axes[i, j].get_xticklabels(), visible=False)
        
        return fig, axes, gs
    
    def plot_datavector_tomographic(
        self,
        statistic: str = "deltasigma",
        log_scale: bool = False,
        save_plot: bool = True,
        filename_suffix: str = "",
        plot_scale_cuts: Optional[bool] = None,
        scale_cuts_override: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Plot tomographic data vectors for all lens bins and source surveys.
        
        Supports both single galaxy type and multiple galaxy types. For multiple
        galaxy types, creates a plot with columns for each galaxy type's bins.
        
        Parameters
        ----------
        statistic : str
            The statistic to plot ('deltasigma' or 'gammat')
        log_scale : bool
            Whether to use log scale for y-axis
        save_plot : bool
            Whether to save the plot to file
        filename_suffix : str
            Optional suffix for the filename
        plot_scale_cuts : bool, optional
            Whether to plot scale cuts. If None, uses analysis_config.apply_scale_cuts
        scale_cuts_override : Dict[str, float], optional
            Override scale cuts with custom values. Keys: "min_deg", "max_deg", "rp_pivot"
        """
        self.logger.info(f"Plotting tomographic data vectors for {statistic}")
        
        # Determine if scale cuts should be applied
        if plot_scale_cuts is None:
            plot_scale_cuts = self.analysis_config.apply_scale_cuts
        
        n_surveys = len(self.source_config.surveys)
        
        # Setup figure with colorbar space - use total_bins for multi-galaxy support
        fig_width = 7.24
        fig_height = fig_width / self.total_bins * n_surveys
        fig, axes, gs = self._initialize_gridspec_figure(
            (fig_width, fig_height), n_surveys, self.total_bins,
            add_cbar=True, hspace=0, wspace=0
        )
        
        # Build lens_bin_info mapping for scale cuts
        lens_bin_info = {}
        
        # Plot data for each survey and galaxy type
        for survey_idx, source_survey in enumerate(self.source_config.surveys):
            n_tomo_bins = self.source_config.get_n_tomographic_bins(source_survey)
            
            # Iterate over galaxy types and their bins
            for galaxy_type in self.galaxy_types:
                start_col, end_col = self.bin_layout[galaxy_type]
                n_lens_bins = self.analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
                
                for lens_bin in range(n_lens_bins):
                    y_label = ""
                    col_idx = start_col + lens_bin
                    ax = axes[survey_idx, col_idx]
                    
                    # Store lens_bin_info for scale cuts
                    lens_bin_info[col_idx] = (galaxy_type, lens_bin)
                    
                    # Set title for top row
                    if survey_idx == 0:
                        galaxy_type_short = galaxy_type[:3]
                        ax.set_title(f"{galaxy_type_short} Bin {lens_bin + 1}")
                    
                    # Get allowed source bins for this lens bin
                    lens_config_gt = self._get_lens_config_for_galaxy_type(galaxy_type)
                    z_max_bin = lens_config_gt.z_bins[lens_bin + 1] if lens_config_gt else None
                    allowed_source_bins = self.analysis_config.get_allowed_source_bins(
                        galaxy_type, source_survey, z_max_bin
                    ) if z_max_bin is not None else []
                    
                    # Plot each allowed tomographic bin
                    for source_bin in allowed_source_bins:                        
                        results = self._load_lensing_results_for_galaxy_type(
                            galaxy_type, lens_bin, source_survey, source_bin, statistic
                        )
                        
                        if results is None:
                            continue
                        
                        # Load covariance for error bars (use theory when available)
                        cov = self._load_covariance_for_plotting_for_galaxy_type(
                            galaxy_type, lens_bin, source_survey, source_bin, statistic,
                            pure_noise=False, use_theory=True,
                            n_lens_bins=n_lens_bins, n_source_bins=n_tomo_bins
                        )
                        
                        if cov is None:
                            self.logger.warning(
                                f"No covariance for {galaxy_type}, {lens_bin}, {source_survey}, {source_bin}"
                            )
                            continue
                        
                        # Extract data - use covariance diagonal for errors
                        error = np.sqrt(np.diag(cov))
                        if statistic == "deltasigma":
                            rp = results['rp']
                            signal = results['ds']
                            y_label = r"$\Delta\Sigma(r_p)$"
                            if not log_scale:
                                signal = rp * signal
                                error = rp * error
                                y_label = r"$r_p \times \Delta\Sigma(r_p)$"
                        else:  # gammat
                            rp = results['theta']
                            signal = results['et']
                            y_label = r"$\gamma_t(\theta)$"
                            if not log_scale:
                                signal = rp * signal
                                error = rp * error
                                y_label = r"$\theta \times \gamma_t(\theta)$"
                        
                        # Apply small offset for clarity
                        offset = 0.02 * source_bin
                        rp_plot = rp * np.exp(offset)
                        
                        # Plot with error bars
                        ax.errorbar(
                            rp_plot, signal, error,
                            fmt='o', color=self.color_list[source_bin % len(self.color_list)],
                            markersize=2, capsize=1
                        )
                    
                    # Set axis properties
                    ax.set_xscale('log')
                    if log_scale:
                        ax.set_yscale('log')
                    
                    # Labels
                    if survey_idx == n_surveys - 1:  # Bottom row
                        if statistic == "deltasigma":
                            ax.set_xlabel(r"$r_p$ [Mpc/h]")
                        else:
                            ax.set_xlabel(r"$\theta$ [deg]")
                    
                    if col_idx == 0:  # Left column
                        ax.set_ylabel(f"{source_survey}\n{y_label}")
        
        # Add survey colorbar legend for tomographic bins
        max_tomo_bins = max(self.source_config.get_n_tomographic_bins(survey) 
                           for survey in self.source_config.surveys)
        if max_tomo_bins > 1:
            self._add_survey_colorbar_legend(
                fig, axes, gs, 
                self.color_list[:max_tomo_bins],
                [f"Bin {i+1}" for i in range(max_tomo_bins)]
            )
        
        # Plot scale cuts using survey-specific settings
        if plot_scale_cuts:
            for survey_idx, source_survey in enumerate(self.source_config.surveys):
                # Get scale cuts for this survey
                if scale_cuts_override:
                    scale_cuts = scale_cuts_override
                else:
                    scale_cuts = self.analysis_config.get_scale_cuts(source_survey, statistic)
                
                # Plot scale cuts for this survey's row
                survey_axes = axes[survey_idx:survey_idx+1, :-1]  # Exclude colorbar column
                self._plot_scale_cuts(
                    survey_axes, scale_cuts["min_deg"], scale_cuts["max_deg"], 
                    scale_cuts["rp_pivot"], statistic, shared_axes=True, tomographic=True,
                    lens_bin_info=lens_bin_info
                )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            scale_suffix = "_log" if log_scale else ""
            # Include galaxy types in filename for multi-galaxy plots
            galaxy_suffix = "_" + "_".join(self.galaxy_types) if len(self.galaxy_types) > 1 else ""
            filename = f"{statistic}_datavector_tomo{galaxy_suffix}{scale_suffix}{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(
                filepath, dpi=self.plot_config.dpi, 
                transparent=self.plot_config.transparent_background,
                bbox_inches="tight"
            )
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()
    
    def plot_covariance_comparison_tomographic(
        self,
        statistic: str = "deltasigma",
        save_plot: bool = True,
        filename_suffix: str = "",
        plot_scale_cuts: Optional[bool] = None,
        scale_cuts_override: Optional[Dict[str, float]] = None,
        log_scale: bool = False,
        bmodes: bool = False
    ) -> None:
        """
        Plot comparison of theoretical and jackknife covariance uncertainties.
        
        Creates a multi-panel plot showing:
        - Upper panel (2:1 height ratio): Theoretical uncertainty (sqrt of diagonal 
          of theory covariance) as lines and jackknife uncertainty as 'o' markers
        - Lower panel (2:1 height ratio): Ratio of theory to jackknife uncertainty
        
        Supports both single galaxy type and multiple galaxy types. For multiple
        galaxy types, creates a plot with columns for each galaxy type's bins.
        
        Parameters
        ----------
        statistic : str
            The statistic to plot ('deltasigma' or 'gammat')
        save_plot : bool
            Whether to save the plot to file
        filename_suffix : str
            Optional suffix for the filename
        plot_scale_cuts : bool, optional
            Whether to plot scale cuts. If None, uses analysis_config.apply_scale_cuts
        scale_cuts_override : Dict[str, float], optional
            Override scale cuts with custom values. Keys: "min_deg", "max_deg", "rp_pivot"
        log_scale : bool
            Whether to use log scale for y-axis
        bmodes : bool
            Whether to compare B-mode covariances (pure noise covariances).
            If True, loads covariances with pure_noise=True.
        """
        mode_str = "B-mode " if bmodes else ""
        self.logger.info(f"Plotting {mode_str}covariance comparison for {statistic}")
        
        # Determine if scale cuts should be applied
        if plot_scale_cuts is None:
            plot_scale_cuts = self.analysis_config.apply_scale_cuts
        
        n_surveys = len(self.source_config.surveys)
        n_rows = n_surveys * 2  # Two rows per survey: main + ratio
        
        # Setup figure with custom gridspec for 2:1 height ratio panels
        fig_width = 7.24
        fig_height = fig_width / self.total_bins * n_surveys * 1.5  # Extra height for ratio panels
        
        # Create height ratios: [2, 1, 2, 1, ...] for alternating main/ratio panels
        height_ratios = []
        for _ in range(n_surveys):
            height_ratios.extend([2, 1])
        
        # Create figure with custom gridspec
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(
            n_rows, self.total_bins + 1,  # +1 for colorbar
            width_ratios=[20] * self.total_bins + [1],
            height_ratios=height_ratios,
            hspace=0, wspace=0
        )
        
        # Create axes arrays for main and ratio panels
        axes_main = np.empty((n_surveys, self.total_bins + 1), dtype=object)
        axes_ratio = np.empty((n_surveys, self.total_bins + 1), dtype=object)
        
        # Initialize axes with proper sharing
        for survey_idx in range(n_surveys):
            main_row = survey_idx * 2
            ratio_row = survey_idx * 2 + 1
            
            for col_idx in range(self.total_bins):
                # Main panel
                if survey_idx == 0 and col_idx == 0:
                    axes_main[survey_idx, col_idx] = fig.add_subplot(gs[main_row, col_idx])
                elif col_idx == 0:
                    axes_main[survey_idx, col_idx] = fig.add_subplot(
                        gs[main_row, col_idx],
                        sharex=axes_main[0, col_idx]
                    )
                elif survey_idx == 0:
                    axes_main[survey_idx, col_idx] = fig.add_subplot(
                        gs[main_row, col_idx],
                        sharey=axes_main[survey_idx, 0]
                    )
                else:
                    axes_main[survey_idx, col_idx] = fig.add_subplot(
                        gs[main_row, col_idx],
                        sharex=axes_main[0, col_idx],
                        sharey=axes_main[survey_idx, 0]
                    )
                
                # Ratio panel - share x with main panel above
                if col_idx == 0:
                    axes_ratio[survey_idx, col_idx] = fig.add_subplot(
                        gs[ratio_row, col_idx],
                        sharex=axes_main[survey_idx, col_idx]
                    )
                else:
                    axes_ratio[survey_idx, col_idx] = fig.add_subplot(
                        gs[ratio_row, col_idx],
                        sharex=axes_main[survey_idx, col_idx],
                        sharey=axes_ratio[survey_idx, 0]
                    )
                
                # Hide x-tick labels for main panels (ratio panels show them)
                plt.setp(axes_main[survey_idx, col_idx].get_xticklabels(), visible=False)
                
                # Hide y-tick labels for non-leftmost columns
                if col_idx > 0:
                    plt.setp(axes_main[survey_idx, col_idx].get_yticklabels(), visible=False)
                    plt.setp(axes_ratio[survey_idx, col_idx].get_yticklabels(), visible=False)
                
                # Hide x-tick labels for ratio panels except bottom survey
                if survey_idx < n_surveys - 1:
                    plt.setp(axes_ratio[survey_idx, col_idx].get_xticklabels(), visible=False)
        
        # Build lens_bin_info mapping for scale cuts
        lens_bin_info = {}
        
        # Plot data for each survey and galaxy type
        for survey_idx, source_survey in enumerate(self.source_config.surveys):
            n_tomo_bins = self.source_config.get_n_tomographic_bins(source_survey)
            
            # Iterate over galaxy types and their bins
            for galaxy_type in self.galaxy_types:
                start_col, end_col = self.bin_layout[galaxy_type]
                n_lens_bins = self.analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
                lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
                
                if lens_config is None:
                    continue
                
                version = lens_config.get_catalogue_version()
                z_bins = lens_config.z_bins
                
                for lens_bin in range(n_lens_bins):
                    y_label = ""
                    col_idx = start_col + lens_bin
                    ax_main = axes_main[survey_idx, col_idx]
                    ax_ratio = axes_ratio[survey_idx, col_idx]
                    
                    # Store lens_bin_info for scale cuts
                    lens_bin_info[col_idx] = (galaxy_type, lens_bin)
                    
                    # Set title for top row
                    if survey_idx == 0:
                        galaxy_type_short = galaxy_type[:3]
                        ax_main.set_title(f"{galaxy_type_short} Bin {lens_bin + 1}")
                    
                    # Get allowed source bins for this lens bin
                    z_max_bin = lens_config.z_bins[lens_bin + 1]
                    allowed_source_bins = self.analysis_config.get_allowed_source_bins(
                        galaxy_type, source_survey, z_max_bin
                    )
                    
                    # Plot each allowed tomographic bin
                    for source_bin in allowed_source_bins:
                        if source_bin >= n_tomo_bins:
                            continue  # Skip if source bin doesn't exist for this survey
                        
                        # Load lensing results to get rp/theta values
                        results = self._load_lensing_results_for_galaxy_type(
                            galaxy_type, lens_bin, source_survey, source_bin, statistic
                        )
                        
                        if results is None:
                            continue
                        
                        # Load theory covariance
                        theory_cov = self._load_covariance_for_plotting_for_galaxy_type(
                            galaxy_type, lens_bin, source_survey, source_bin, statistic,
                            pure_noise=bmodes, use_theory=True,
                            n_lens_bins=n_lens_bins, n_source_bins=n_tomo_bins
                        )
                        
                        # Load jackknife covariance
                        jackknife_cov = self.output_config.load_covariance_matrix(
                            version, galaxy_type, source_survey, lens_bin, z_bins,
                            statistic, source_bin, pure_noise=bmodes
                        )
                        
                        # Extract x-axis values and set y label
                        if statistic == "deltasigma":
                            rp = results['rp']
                            if bmodes:
                                y_label = r"$\sigma(\Delta\Sigma_\times)$"
                            else:
                                y_label = r"$\sigma(\Delta\Sigma)$"
                        else:  # gammat
                            rp = results['theta']
                            if bmodes:
                                y_label = r"$\sigma(\gamma_\times)$"
                            else:
                                y_label = r"$\sigma(\gamma_t)$"
                        
                        # Apply small offset for clarity
                        offset = 0.02 * source_bin
                        rp_plot = rp * np.exp(offset)
                        color = self.color_list[source_bin % len(self.color_list)]
                        
                        # Extract errors
                        theory_err = None
                        jackknife_err = None
                        
                        # Plot theory covariance as a line in main panel
                        if theory_cov is not None:
                            theory_err = np.sqrt(np.diag(theory_cov))
                            ax_main.plot(
                                rp_plot, theory_err,
                                linestyle='-', color=color, linewidth=1.5, alpha=0.8,
                                label=f"Theory S{source_bin+1}" if lens_bin == 0 and survey_idx == 0 else ""
                            )
                        else:
                            self.logger.warning(
                                f"No theory covariance for {galaxy_type}, {lens_bin}, {source_survey}, {source_bin}"
                            )
                        
                        # Plot jackknife covariance as markers in main panel
                        if jackknife_cov is not None:
                            jackknife_err = np.sqrt(np.diag(jackknife_cov))
                            ax_main.plot(
                                rp_plot, jackknife_err,
                                marker='o', linestyle='', color=color, markersize=3, alpha=0.8,
                                label=f"JK S{source_bin+1}" if lens_bin == 0 and survey_idx == 0 else ""
                            )
                        else:
                            self.logger.warning(
                                f"No jackknife covariance for {galaxy_type}, {lens_bin}, {source_survey}, {source_bin}"
                            )
                        
                        # Plot ratio in ratio panel (theory / jackknife)
                        if theory_err is not None and jackknife_err is not None:
                            ratio = theory_err / jackknife_err
                            ax_ratio.plot(
                                rp_plot, ratio,
                                marker='o', linestyle='-', color=color, 
                                markersize=2, linewidth=1, alpha=0.8
                            )
                    
                    # Set axis properties for main panel
                    ax_main.set_xscale('log')
                    if log_scale:
                        ax_main.set_yscale('log')
                    
                    # Set axis properties for ratio panel
                    ax_ratio.set_xscale('log')
                    ax_ratio.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
                    
                    # Labels for main panel
                    if col_idx == 0:  # Left column
                        ax_main.set_ylabel(f"{source_survey}\n{y_label}")
                        ax_ratio.set_ylabel("Theory/JK")
                    
                    # X-axis labels for ratio panel (bottom row only)
                    if survey_idx == n_surveys - 1:  # Bottom survey
                        if statistic == "deltasigma":
                            ax_ratio.set_xlabel(r"$r_p$ [Mpc/h]")
                        else:
                            ax_ratio.set_xlabel(r"$\theta$ [arcmin]")
        
        # Add survey colorbar legend for tomographic bins
        max_tomo_bins = max(self.source_config.get_n_tomographic_bins(survey) 
                           for survey in self.source_config.surveys)
        if max_tomo_bins > 1:
            # Create colorbar spanning both main and ratio rows for each survey
            local_color_list = self.color_list[:max_tomo_bins]
            tick_labels = [f"Bin {i+1}" for i in range(max_tomo_bins)]
            
            cmap = mpl.colors.ListedColormap(local_color_list)
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm._A = []
            
            ticks = np.linspace(0, 1, len(tick_labels) + 1)
            ticks = 0.5 * (ticks[1:] + ticks[:-1])
            
            # Add one colorbar per survey, spanning both main and ratio rows
            for survey_idx in range(n_surveys):
                main_row = survey_idx * 2
                ratio_row = survey_idx * 2 + 1
                # Create colorbar axis spanning both rows
                cbar_ax = fig.add_subplot(gs[main_row:ratio_row+1, -1])
                cb = plt.colorbar(sm, cax=cbar_ax, pad=0.0, ticks=ticks)
                cb.ax.set_yticklabels(tick_labels)
                cb.ax.minorticks_off()
                cb.ax.tick_params(size=0)
        
        # Plot scale cuts using survey-specific settings
        if plot_scale_cuts:
            for survey_idx, source_survey in enumerate(self.source_config.surveys):
                # Get scale cuts for this survey
                if scale_cuts_override:
                    scale_cuts = scale_cuts_override
                else:
                    scale_cuts = self.analysis_config.get_scale_cuts(source_survey, statistic)
                
                # Plot scale cuts for main panel row
                survey_axes_main = axes_main[survey_idx:survey_idx+1, :-1]  # Exclude colorbar column
                self._plot_scale_cuts(
                    survey_axes_main, scale_cuts["min_deg"], scale_cuts["max_deg"], 
                    scale_cuts["rp_pivot"], statistic, shared_axes=True, tomographic=True,
                    lens_bin_info=lens_bin_info
                )
                
                # Plot scale cuts for ratio panel row
                survey_axes_ratio = axes_ratio[survey_idx:survey_idx+1, :-1]  # Exclude colorbar column
                self._plot_scale_cuts(
                    survey_axes_ratio, scale_cuts["min_deg"], scale_cuts["max_deg"], 
                    scale_cuts["rp_pivot"], statistic, shared_axes=True, tomographic=True,
                    lens_bin_info=lens_bin_info
                )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            scale_suffix = "_log" if log_scale else ""
            bmodes_suffix = "_bmodes" if bmodes else ""
            # Include galaxy types in filename for multi-galaxy plots
            galaxy_suffix = "_" + "_".join(self.galaxy_types) if len(self.galaxy_types) > 1 else ""
            filename = f"{statistic}_covariance_comparison_tomo{galaxy_suffix}{bmodes_suffix}{scale_suffix}{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(
                filepath, dpi=self.plot_config.dpi, 
                transparent=self.plot_config.transparent_background,
                bbox_inches="tight"
            )
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()

    def plot_bmodes_tomographic(
        self,
        statistic: str = "deltasigma", 
        save_plot: bool = True,
        filename_suffix: str = "",
        plot_scale_cuts: Optional[bool] = None,
        scale_cuts_override: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Plot B-mode data vectors and compute p-values.
        
        Supports both single galaxy type and multiple galaxy types.
        
        Parameters
        ----------
        statistic : str
            The statistic to plot ('deltasigma' or 'gammat')
        save_plot : bool
            Whether to save the plot
        filename_suffix : str
            Optional suffix for the filename
        plot_scale_cuts : bool, optional
            Whether to plot scale cuts. If None, uses analysis_config.apply_scale_cuts
        scale_cuts_override : Dict[str, float], optional
            Override scale cuts with custom values. Keys: "min_deg", "max_deg", "rp_pivot"
            
        Returns
        -------
        Dict[str, float]
            Dictionary of p-values for each configuration
        """
        self.logger.info(f"Plotting B-mode diagnostics for {statistic}")
        
        # Determine if scale cuts should be applied
        if plot_scale_cuts is None:
            plot_scale_cuts = self.analysis_config.apply_scale_cuts
        
        # Store the original computation config
        original_bmodes = self.computation_config.bmodes
        
        # Temporarily enable B-modes for this analysis
        self.computation_config.bmodes = True
        
        n_surveys = len(self.source_config.surveys)
        
        # Setup figure with colorbar space - use total_bins for multi-galaxy support
        fig_width = 7.24
        fig_height = fig_width / self.total_bins * n_surveys
        fig, axes, gs = self._initialize_gridspec_figure(
            (fig_width, fig_height), n_surveys, self.total_bins,
            add_cbar=True, hspace=0, wspace=0
        )
        
        pvalues = {}
        lens_bin_info = {}
        
        # Plot B-mode data for each survey and galaxy type
        for survey_idx, source_survey in enumerate(self.source_config.surveys):
            n_tomo_bins = self.source_config.get_n_tomographic_bins(source_survey)
            
            # Get scale cuts for this survey
            if scale_cuts_override:
                scale_cuts = scale_cuts_override
            else:
                scale_cuts = self.analysis_config.get_scale_cuts(source_survey, statistic)
            
            # Iterate over galaxy types and their bins
            for galaxy_type in self.galaxy_types:
                start_col, end_col = self.bin_layout[galaxy_type]
                n_lens_bins = self.analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
                lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
                
                if lens_config is None:
                    continue
                
                version = lens_config.get_catalogue_version()
                z_bins = lens_config.z_bins
                
                for lens_bin in range(n_lens_bins):
                    col_idx = start_col + lens_bin
                    ax = axes[survey_idx, col_idx]
                    
                    # Store lens_bin_info for scale cuts
                    lens_bin_info[col_idx] = (galaxy_type, lens_bin)
                    
                    # Set title for top row
                    if survey_idx == 0:
                        galaxy_type_short = galaxy_type[:3]
                        ax.set_title(f"{galaxy_type_short} Bin {lens_bin + 1}")
                    
                    # Get allowed source bins for this lens bin
                    z_max_bin = z_bins[lens_bin + 1]
                    allowed_source_bins = self.analysis_config.get_allowed_source_bins(
                        galaxy_type, source_survey, z_max_bin
                    )
                    
                    # Combine all allowed tomographic bins for this lens-source combination
                    combined_data = []
                    combined_cov_blocks = []
                    combined_rp = None
                    
                    for source_bin in allowed_source_bins:
                        if source_bin >= n_tomo_bins:
                            continue  # Skip if source bin doesn't exist for this survey
                        
                        # Load B-mode results
                        results = self.output_config.load_lensing_results(
                            version, galaxy_type, source_survey, lens_bin, z_bins,
                            statistic, source_bin, pure_noise=True
                        )
                        # Use theory covariance with pure_noise for B-modes
                        cov = self._load_covariance_for_plotting_for_galaxy_type(
                            galaxy_type, lens_bin, source_survey, source_bin, statistic,
                            pure_noise=True, use_theory=True,
                            n_lens_bins=n_lens_bins, n_source_bins=n_tomo_bins
                        )
                        
                        if results is None or cov is None:
                            self.logger.warning(
                                f"No B-mode results for {galaxy_type} {lens_bin}, {source_survey}, {source_bin}"
                            )
                            continue
                        
                        # Extract B-mode data - use covariance diagonal for errors
                        error = np.sqrt(np.diag(cov))
                        if statistic == "deltasigma":
                            rp = results['rp']
                            signal = results['ds']
                        else:
                            rp = results['theta'] 
                            signal = results['et']
                        
                        # Store rp values for scale cut application
                        if combined_rp is None:
                            combined_rp = rp
                        
                        # Apply scale cuts to data before adding to combined vector
                        if plot_scale_cuts:
                            scale_mask = self._get_scale_cut_mask(
                                rp, scale_cuts, statistic, galaxy_type, lens_bin
                            )
                            signal_cut = signal[scale_mask]
                            cov_cut = cov[np.ix_(scale_mask, scale_mask)]
                        else:
                            signal_cut = signal
                            cov_cut = cov
                        
                        combined_data.append(signal_cut)
                        combined_cov_blocks.append(cov_cut)
                        
                        # Plot individual tomographic bins (all data, not just scale cuts)
                        offset = 0.02 * source_bin
                        rp_plot = rp * np.exp(offset)
                        ax.errorbar(
                            rp_plot, rp * signal, rp * error,
                            fmt='o', color=self.color_list[source_bin % len(self.color_list)],
                            markersize=2, capsize=1
                        )
                    
                    # Compute combined p-value with scale cuts applied
                    if combined_data:
                        combined_data_vec = np.concatenate(combined_data)
                        combined_cov = self._combine_covariance_matrices(combined_cov_blocks)
                        
                        pvalue = self._compute_pvalue(combined_data_vec, combined_cov)
                        chisq = self._compute_chisq(combined_data_vec, combined_cov)
                        
                        key = f"{galaxy_type}_{source_survey}_{lens_bin}"
                        pvalues[key] = pvalue
                        
                        # Add text annotation
                        ax.text(
                            0.05, 0.9, f"p={pvalue:.3f}, $\\chi^2$={chisq:.1f}",
                            transform=ax.transAxes, color='k', fontsize=8,
                            path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')]
                        )
                    
                    # Set axis properties
                    ax.set_xscale('log')
                    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
                    
                    # Labels
                    if survey_idx == n_surveys - 1:  # Bottom row
                        if statistic == "deltasigma":
                            ax.set_xlabel(r"$r_p$ [Mpc/h]")
                        else:
                            ax.set_xlabel(r"$\theta$ [deg]")
                    
                    if col_idx == 0:  # Left column
                        if statistic == "deltasigma":
                            ax.set_ylabel(f"{source_survey}\n$r_p \\times \\Delta\\Sigma_\\times(r_p)$")
                        else:
                            ax.set_ylabel(f"{source_survey}\n$\\theta \\times \\gamma_\\times(\\theta)$")
        
        # Set y-limits appropriate for B-modes
        if statistic == "deltasigma":
            for ax_row in axes:
                for ax in ax_row[:-1]:  # Exclude colorbar column
                    ax.set_ylim(-2, 2)
        else:
            for ax_row in axes:
                for ax in ax_row[:-1]:  # Exclude colorbar column
                    ax.set_ylim(-5e-5, 5e-5)
        
        # Add survey colorbar legend for tomographic bins
        n_tomo_bins_max = max(self.source_config.get_n_tomographic_bins(survey) 
                               for survey in self.source_config.surveys)
        if n_tomo_bins_max > 1:
            self._add_survey_colorbar_legend(
                fig, axes, gs,
                self.color_list[:n_tomo_bins_max],
                [f"Bin {i+1}" for i in range(n_tomo_bins_max)]
            )
        
        # Plot scale cuts using survey-specific settings
        if plot_scale_cuts:
            for survey_idx, source_survey in enumerate(self.source_config.surveys):
                # Get scale cuts for this survey
                if scale_cuts_override:
                    scale_cuts = scale_cuts_override
                else:
                    scale_cuts = self.analysis_config.get_scale_cuts(source_survey, statistic)
                
                # Plot scale cuts for this survey's row
                survey_axes = axes[survey_idx:survey_idx+1, :-1]  # Exclude colorbar column
                self._plot_scale_cuts(
                    survey_axes, scale_cuts["min_deg"], scale_cuts["max_deg"], 
                    scale_cuts["rp_pivot"], statistic, shared_axes=True, tomographic=True,
                    lens_bin_info=lens_bin_info
                )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            # Include galaxy types in filename for multi-galaxy plots
            galaxy_suffix = "_" + "_".join(self.galaxy_types) if len(self.galaxy_types) > 1 else ""
            filename = f"{statistic}_bmodes_tomo{galaxy_suffix}{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(
                filepath, dpi=self.plot_config.dpi,
                transparent=self.plot_config.transparent_background,
                bbox_inches="tight"
            )
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()
        
        # Restore original B-mode setting
        self.computation_config.bmodes = original_bmodes
        
        return pvalues
    
    def plot_survey_comparison_tomographic(
        self,
        statistic: str = "deltasigma",
        log_scale: bool = False,
        save_plot: bool = True,
        filename_suffix: str = "",
        plot_scale_cuts: Optional[bool] = None,
        scale_cuts_override: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Plot tomographic data vectors comparing different surveys.
        
        Each row corresponds to a source tomographic bin, with different colors
        for different surveys. Bins that are not allowed for a given lens-source
        combination are plotted with 'x' markers and smaller size.
        
        Parameters
        ----------
        statistic : str
            The statistic to plot ('deltasigma' or 'gammat')
        log_scale : bool
            Whether to use log scale for y-axis
        save_plot : bool
            Whether to save the plot to file
        filename_suffix : str
            Optional suffix for the filename
        plot_scale_cuts : bool, optional
            Whether to plot scale cuts. If None, uses analysis_config.apply_scale_cuts
        scale_cuts_override : Dict[str, float], optional
            Override scale cuts with custom values. Keys: "min_deg", "max_deg", "rp_pivot"
        """
        self.logger.info(f"Plotting survey comparison for {statistic}")
        
        # Determine if scale cuts should be applied
        if plot_scale_cuts is None:
            plot_scale_cuts = self.analysis_config.apply_scale_cuts
        
        n_lens_bins = self.lens_config.get_n_lens_bins()
        n_surveys = len(self.source_config.surveys)
        
        # Get maximum number of tomographic bins across all surveys
        max_tomo_bins = max(self.source_config.get_n_tomographic_bins(survey) 
                           for survey in self.source_config.surveys)
        
        # Setup figure with colorbar space - rows are tomographic bins
        fig_width = 7.24
        fig_height = fig_width / n_lens_bins * max_tomo_bins
        fig, axes, gs = self._initialize_gridspec_figure(
            (fig_width, fig_height), max_tomo_bins, n_lens_bins,
            add_cbar=True, hspace=0, wspace=0
        )
        
        galaxy_type = self.lens_config.galaxy_type
        
        # Plot data for each combination
        for source_bin in range(max_tomo_bins):
            for lens_bin in range(n_lens_bins):
                y_label = ""
                ax = axes[source_bin, lens_bin]
                
                # Set title for top row
                if source_bin == 0:
                    galaxy_type_short = self.lens_config.galaxy_type[:3]
                    ax.set_title(f"{galaxy_type_short} Bin {lens_bin + 1}")
                
                # Plot each survey with different colors
                for survey_idx, source_survey in enumerate(self.source_config.surveys):
                    n_tomo_bins = self.source_config.get_n_tomographic_bins(source_survey)
                    
                    # Skip if this tomographic bin doesn't exist for this survey
                    if source_bin >= n_tomo_bins:
                        continue
                    
                    # Check if this source bin is allowed for this lens bin
                    z_max_bin = self.lens_config.z_bins[lens_bin + 1]
                    allowed_source_bins = self.analysis_config.get_allowed_source_bins(
                        galaxy_type, source_survey, z_max_bin
                    )
                    is_allowed = source_bin in allowed_source_bins
                    
                    # Load results
                    results = self._load_lensing_results(
                        lens_bin, source_survey, source_bin, statistic
                    )
                    
                    if results is None:
                        self.logger.warning(f"No results found for {lens_bin}, {source_survey}, {source_bin}")
                        continue
                    
                    # Load covariance for error bars
                    cov = self._load_covariance_for_plotting(
                        lens_bin, source_survey, source_bin, statistic,
                        pure_noise=False, use_theory=True,
                        n_source_bins=n_tomo_bins
                    )
                    
                    if cov is None:
                        self.logger.warning(f"No covariance found for {lens_bin}, {source_survey}, {source_bin}")
                        continue
                    
                    # Extract data - use covariance diagonal for errors
                    error = np.sqrt(np.diag(cov))
                    if statistic == "deltasigma":
                        rp = results['rp']
                        signal = results['ds']
                        y_label = r"$\Delta\Sigma(r_p)$"
                        if not log_scale:
                            signal = rp * signal
                            error = rp * error
                            y_label = r"$r_p \times \Delta\Sigma(r_p)$"
                    else:  # gammat
                        rp = results['theta']
                        signal = results['et']
                        y_label = r"$\gamma_t(\theta)$"
                        if not log_scale:
                            signal = rp * signal
                            error = rp * error
                            y_label = r"$\theta \times \gamma_t(\theta)$"
                    
                    # Apply small offset for clarity
                    offset = 0.02 * survey_idx
                    rp_plot = rp * np.exp(offset)
                    
                    # Set marker style based on whether bin is allowed
                    if is_allowed:
                        marker = 'o'
                        markersize = 2
                        alpha = 1.0
                    else:
                        marker = 'x'
                        markersize = 1.5
                        alpha = 0.5
                    
                    # Plot with error bars
                    ax.errorbar(
                        rp_plot, signal, error,
                        fmt=marker, color=self.color_list[survey_idx % len(self.color_list)],
                        markersize=markersize, capsize=1, alpha=alpha,
                        label=source_survey if lens_bin == 0 and source_bin == 0 else ""
                    )
                    
                    # Calculate and display signal-to-noise ratio
                    if is_allowed:
                        try:
                            # Get signal vector (use original signal, not multiplied by rp)
                            if statistic == "deltasigma":
                                signal_vec = results['ds']
                            else:
                                signal_vec = results['et']
                            
                            # Calculate S/N = sqrt(signal^T * Cov^-1 * signal)
                            try:
                                inv_cov = np.linalg.inv(cov)
                                sn_squared = np.einsum('i,ij,j', signal_vec, inv_cov, signal_vec)
                                sn = np.sqrt(np.abs(sn_squared))
                                
                                # Add text annotation with survey color
                                # Position text in upper part of plot, offset by survey index
                                y_pos = 0.95 - 0.08 * survey_idx
                                ax.text(
                                    0.05, y_pos, f"{source_survey}: S/N={sn:.1f}",
                                    transform=ax.transAxes, 
                                    color=self.color_list[survey_idx % len(self.color_list)],
                                    fontsize=7, va='top',
                                    path_effects=[PathEffects.withStroke(linewidth=1.5, foreground='white')]
                                )
                            except np.linalg.LinAlgError:
                                self.logger.warning(f"Could not invert covariance for {source_survey}, bin {source_bin}")
                        except Exception as e:
                            self.logger.warning(f"Could not compute S/N for {source_survey}, bin {source_bin}: {e}")
                
                # Set axis properties
                ax.set_xscale('log')
                if log_scale:
                    ax.set_yscale('log')
                
                # Labels
                if source_bin == max_tomo_bins - 1:  # Bottom row
                    if statistic == "deltasigma":
                        ax.set_xlabel(r"$r_p$ [Mpc/h]")
                    else:
                        ax.set_xlabel(r"$\theta$ [deg]")
                
                if lens_bin == 0:  # Left column
                    ax.set_ylabel(f"Src Bin {source_bin + 1}\n{y_label}")
        
        # Add survey colorbar legend
        if n_surveys > 1:
            self._add_survey_colorbar_legend(
                fig, axes, gs, 
                self.color_list[:n_surveys],
                self.source_config.surveys
            )
        
        # Plot scale cuts using survey-specific settings or override
        if plot_scale_cuts:
            # Create lens bin mapping
            lens_bin_mapping = list(range(n_lens_bins))
            # Use first survey's scale cuts or override
            if scale_cuts_override:
                scale_cuts = scale_cuts_override
            else:
                scale_cuts = self.analysis_config.get_scale_cuts(
                    self.source_config.surveys[0], statistic
                )
            
            # Plot scale cuts for all rows
            self._plot_scale_cuts(
                axes, scale_cuts["min_deg"], scale_cuts["max_deg"], 
                scale_cuts["rp_pivot"], statistic, shared_axes=True, tomographic=True,
                lens_bins=lens_bin_mapping
            )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            scale_suffix = "_log" if log_scale else ""
            filename = f"{statistic}_survey_comparison_tomo{scale_suffix}{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(
                filepath, dpi=self.plot_config.dpi, 
                transparent=self.plot_config.transparent_background,
                bbox_inches="tight"
            )
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()
    
    def plot_randoms_test(
        self,
        statistic: str = "deltasigma",
        save_plot: bool = True,
        filename_suffix: str = "",
        plot_scale_cuts: bool = True,
        min_deg: float = 2.5,
        max_deg: float = 250.0,
        rp_pivot: float = 10.0
    ) -> None:
        """
        Plot random lens tests for systematics testing.
        
        Parameters
        ----------
        statistic : str
            The statistic to plot
        save_plot : bool
            Whether to save the plot
        filename_suffix : str
            Optional suffix for the filename
        plot_scale_cuts : bool
            Whether to plot scale cuts
        min_deg : float
            Minimum angular scale in degrees
        max_deg : float
            Maximum angular scale in degrees
        rp_pivot : float
            Pivot scale in Mpc/h for deltasigma
        """
        self.logger.info(f"Plotting random lens tests for {statistic}")
        
        try:
            # Import randoms analyzer
            from .randoms import create_randoms_analyzer_from_configs
            
            # Create randoms analyzer
            randoms_analyzer = create_randoms_analyzer_from_configs(
                self.computation_config, self.lens_config, 
                self.source_config, self.output_config, self.logger
            )
            
            # Try to load existing randoms results (include galaxy type in suffix)
            try:
                galaxy_suffix = f"_{self.lens_config.galaxy_type.lower()}"
                full_suffix = f"{filename_suffix}{galaxy_suffix}"
                p_arr, V_arr, keys = randoms_analyzer.load_randoms_results(
                    "splits", full_suffix
                )
                self.logger.info(f"Loaded randoms results with {len(keys)} keys")
                
                # Plot the randoms test results
                n_lens_bins = self.lens_config.get_n_lens_bins()
                n_surveys = len(self.source_config.surveys)
                
                fig, axes, gs = self._initialize_gridspec_figure(
                    (7.24, 7.24 / n_lens_bins * n_surveys),
                    n_surveys, n_lens_bins, add_cbar=False,
                    hspace=0, wspace=0
                )
                
                # Plot histograms of random test statistics
                for i, key in enumerate(keys[:min(len(keys), n_surveys * n_lens_bins)]):
                    row = i // n_lens_bins
                    col = i % n_lens_bins
                    ax = axes[row, col] if n_surveys > 1 else axes[col]
                    
                    # Plot histogram of chi-squared values from randoms
                    chi2_values = np.array([np.sum(p_arr[j, i, :]**2) for j in range(p_arr.shape[0])])
                    ax.hist(chi2_values, bins=30, alpha=0.7, density=True, 
                           label='Randoms', color='lightblue')
                    
                    # Add expected chi-squared distribution
                    dof = p_arr.shape[2]  # degrees of freedom
                    x_range = np.linspace(0, np.max(chi2_values), 100)
                    from scipy.stats import chi2 as chi2_dist
                    ax.plot(x_range, chi2_dist.pdf(x_range, dof), 
                           'r--', label=f'χ²({dof})', linewidth=2)
                    
                    # Compute p-value for the null hypothesis test
                    observed_chi2 = np.mean(chi2_values)
                    pvalue = 1 - chi2_dist.cdf(observed_chi2, dof)
                    
                    ax.axvline(observed_chi2, color='red', linestyle='-', 
                              label=f'Observed (p={pvalue:.3f})')
                    ax.set_xlabel('χ² statistic')
                    ax.set_ylabel('Density')
                    ax.set_title(f'{key[:20]}...' if len(key) > 20 else key)
                    ax.legend(fontsize=8)
                
                # Plot scale cuts if requested
                if plot_scale_cuts:
                    # Create lens bin mapping
                    lens_bin_mapping = list(range(n_lens_bins))
                    self._plot_scale_cuts(
                        axes, min_deg, max_deg, rp_pivot, statistic,
                        shared_axes=True, tomographic=True,
                        lens_bins=lens_bin_mapping
                    )
                
                plt.tight_layout()
                
                if save_plot:
                    suffix = f"_{filename_suffix}" if filename_suffix else ""
                    filename = f"{statistic}_randoms_tomo{suffix}.png"
                    filepath = self.plot_dir / filename
                    
                    plt.savefig(filepath, dpi=self.plot_config.dpi, bbox_inches="tight")
                    self.logger.info(f"Saved plot: {filepath}")
                
                plt.show()
                return
                
            except FileNotFoundError:
                self.logger.warning("No existing randoms results found")
        
        except Exception as e:
            self.logger.error(f"Error in randoms test plotting: {e}")
        
        # Fallback: create a placeholder plot showing the structure
        n_lens_bins = self.lens_config.get_n_lens_bins()
        n_surveys = len(self.source_config.surveys)
        
        fig, axes, gs = self._initialize_gridspec_figure(
            (7.24, 7.24 / n_lens_bins * n_surveys),
            n_surveys, n_lens_bins, add_cbar=False,
            hspace=0, wspace=0
        )
        
        # Create placeholder plots
        for survey_idx, source_survey in enumerate(self.source_config.surveys):
            for lens_bin in range(n_lens_bins):
                ax = axes[survey_idx, lens_bin] if n_surveys > 1 else axes[lens_bin]
                
                # Generate mock data for demonstration
                mock_chi2_values = np.random.chisquare(df=10, size=1000)
                ax.hist(mock_chi2_values, bins=30, alpha=0.7, density=True, 
                       label='Mock Randoms', color='lightblue')
                
                # Add theoretical distribution
                x_range = np.linspace(0, np.max(mock_chi2_values), 100)
                from scipy.stats import chi2 as chi2_dist
                ax.plot(x_range, chi2_dist.pdf(x_range, 10), 
                       'r--', label='χ²(10)', linewidth=2)
                
                ax.set_xlabel('χ² statistic')
                ax.set_ylabel('Density')
                ax.set_title(f'{source_survey} Bin {lens_bin+1}')
                ax.legend(fontsize=8)
        
        # Plot scale cuts if requested
        if plot_scale_cuts:
            # Create lens bin mapping
            lens_bin_mapping = list(range(n_lens_bins))
            self._plot_scale_cuts(
                axes, min_deg, max_deg, rp_pivot, statistic,
                shared_axes=True, tomographic=True,
                lens_bins=lens_bin_mapping
            )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            filename = f"{statistic}_randoms_tomo{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(filepath, dpi=self.plot_config.dpi, bbox_inches="tight")
            self.logger.info(f"Saved plot: {filepath}")
        
        self.logger.info("Plotted mock randoms test (run actual randoms analysis first for real results)")
        plt.show()
    
    def _combine_covariance_matrices(self, cov_blocks: List[np.ndarray]) -> np.ndarray:
        """Combine covariance matrices from different tomographic bins."""
        return self.output_config._combine_covariance_blocks(cov_blocks)
    
    def _compute_pvalue(self, data: np.ndarray, covariance: np.ndarray) -> float:
        """Compute p-value for null hypothesis test."""
        try:
            chisq = self._compute_chisq(data, covariance)
            dof = len(data)
            from scipy.stats import chi2
            pvalue = 1 - chi2.cdf(chisq, dof)
            return pvalue
        except Exception as e:
            self.logger.warning(f"P-value computation failed: {e}")
            return np.nan
    
    def _compute_chisq(self, data: np.ndarray, covariance: np.ndarray) -> float:
        """Compute chi-squared statistic."""
        try:
            inv_cov = np.linalg.inv(covariance)
            chisq = np.einsum('i,ij,j', data, inv_cov, data)
            return chisq
        except Exception as e:
            self.logger.warning(f"Chi-squared computation failed: {e}")
            return np.nan
    
    def _get_scale_cut_mask(
        self, 
        rp: np.ndarray, 
        scale_cuts: Dict[str, float], 
        statistic: str,
        galaxy_type: str,
        lens_bin: int
    ) -> np.ndarray:
        """
        Create a boolean mask for applying scale cuts to data.
        
        Parameters
        ----------
        rp : np.ndarray
            Radial or angular separation values
        scale_cuts : Dict[str, float]
            Dictionary with scale cut parameters
        statistic : str
            The statistic type ('deltasigma' or 'gammat')
        galaxy_type : str
            Galaxy type for conversion
        lens_bin : int
            Lens bin index for conversion
            
        Returns
        -------
        np.ndarray
            Boolean mask where True indicates data points within scale cuts
        """
        if statistic == "deltasigma":
            # Convert angular cuts to physical scales
            min_rp, max_rp = self._get_rp_from_deg(
                scale_cuts["min_deg"], scale_cuts["max_deg"], galaxy_type, lens_bin
            )
        else:
            # For gammat, use angular scales directly
            min_rp = scale_cuts["min_deg"]
            max_rp = scale_cuts["max_deg"]
        
        # Create mask for scales within the cut range
        mask = (rp >= min_rp) & (rp <= max_rp)
        return mask
    
    def compare_different_cosmologies(
        self,
        cosmology_configs: Dict[str, ComputationConfig],
        statistic: str = "deltasigma",
        save_plot: bool = True,
        filename_suffix: str = ""
    ) -> None:
        """
        Compare data vectors computed with different cosmological models.
        
        Parameters
        ----------
        cosmology_configs : Dict[str, ComputationConfig]
            Dictionary mapping cosmology names to computation configs
        statistic : str
            The statistic to compare
        save_plot : bool
            Whether to save the plot
        filename_suffix : str
            Optional suffix for the filename
        """
        self.logger.info(f"Comparing different cosmologies for {statistic}")
        
        n_lens_bins = self.lens_config.get_n_lens_bins()
        n_surveys = len(self.source_config.surveys)
        
        fig, axes = plt.subplots(
            n_surveys, n_lens_bins,
            figsize=(7.24, 7.24 / n_lens_bins * n_surveys),
            sharex=True, sharey=True
        )
        
        # Ensure axes is 2D
        if n_surveys == 1:
            axes = axes.reshape(1, -1)
        if n_lens_bins == 1:
            axes = axes.reshape(-1, 1)
        
        # This would require loading results from different cosmology runs
        # Implementation depends on directory structure for different cosmologies
        
        self.logger.warning("Cosmology comparison plotting not yet fully implemented")
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            filename = f"{statistic}_cosmology_comparison{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(filepath, dpi=self.plot_config.dpi, bbox_inches="tight")
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()
    
    def plot_magnitudes(
        self,
        magnitude_cuts: Optional[List[float]] = None,
        mag_col: str = "ABSMAG01_SDSS_R",
        apply_extinction_correction: bool = False,
        add_kp3_cut: bool = False,
        save_plot: bool = True,
        filename_suffix: str = ""
    ) -> None:
        """
        Plot absolute magnitudes versus redshift showing magnitude cuts and redshift bin boundaries.
        
        This function creates a density plot of lens galaxies in the magnitude-redshift plane,
        overlaying the magnitude cuts applied per redshift bin and the redshift bin boundaries.
        
        Parameters
        ----------
        magnitude_cuts : List[float], optional
            Magnitude cuts per redshift bin. If None, uses default values based on galaxy type.
            For BGS_BRIGHT: [-19.5, -20.5, -21.0]
            For LRG: None (no magnitude cuts)
        mag_col : str
            Column name for magnitudes. Default: "ABSMAG01_SDSS_R"
        apply_extinction_correction : bool
            Whether to apply extinction correction: ecorr = -0.8*(z-0.1)
        add_kp3_cut : bool
            Whether to add the KP3 cut at -21.5 mag as a horizontal red line
        save_plot : bool
            Whether to save the plot to file
        filename_suffix : str
            Optional suffix for the filename
        """
        self.logger.info(f"Plotting magnitude distribution for {self.lens_config.galaxy_type}")
        
        galaxy_type = self.lens_config.galaxy_type
        z_bins = self.lens_config.z_bins
        
        # Get default magnitude cuts if not provided
        if magnitude_cuts is None:
            from ..utils import fastspecfit_utils
            magnitude_cuts = fastspecfit_utils.get_default_magnitude_cuts(galaxy_type)
        else:
            magnitude_cuts = np.array(magnitude_cuts) if magnitude_cuts is not None else None
        
        # Load lens catalogue data
        try:
            lens_table = self._load_lens_catalogue_for_magnitudes(mag_col)
        except Exception as e:
            self.logger.error(f"Failed to load lens catalogue: {e}")
            return
        
        if lens_table is None or len(lens_table) == 0:
            self.logger.error("No lens catalogue data available")
            return
        
        # Get redshift and magnitude columns
        z_col = 'Z_not4clus' if 'Z_not4clus' in lens_table.colnames else 'Z'
        redshifts = lens_table[z_col]
        magnitudes = lens_table[mag_col]
        
        # Apply extinction correction if requested
        if apply_extinction_correction:
            ecorr = -0.8 * (redshifts - 0.1)
            magnitudes = magnitudes + ecorr
            mag_label = f"Corrected {mag_col.replace('_', ' ')}"
        else:
            mag_label = mag_col.replace('_', ' ')
        
        # Create figure
        fig = plt.figure(figsize=(5, 3))
        
        # Create density plot
        try:
            ax = self._create_magnitude_density_plot(fig, redshifts, magnitudes)
        except ImportError:
            # Fallback to regular scatter plot if mpl_scatter_density not available
            self.logger.warning("mpl_scatter_density not available, using regular scatter plot")
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(redshifts, magnitudes, s=0.1, alpha=0.5)
        
        # Calculate and display survival fraction
        if magnitude_cuts is not None:
            magnitude_mask = self._get_magnitude_mask(lens_table, magnitude_cuts, z_bins, mag_col, apply_extinction_correction)
            survived_fraction = np.sum(magnitude_mask) / len(lens_table)
            total_galaxies = np.sum(magnitude_mask)
            
            ax.text(0.05, 0.95, f"{survived_fraction*100:.1f}% remain", 
                   transform=ax.transAxes, ha='left', va='top', fontsize=15)
            self.logger.info(f'Total galaxies passing cuts: {total_galaxies}')
        
        # Plot magnitude cuts and redshift bin boundaries
        if magnitude_cuts is not None and len(magnitude_cuts) == len(z_bins) - 1:
            # Plot horizontal magnitude cut lines
            for i in range(len(magnitude_cuts)):
                plt.plot(z_bins[i:i+2], [magnitude_cuts[i], magnitude_cuts[i]], 
                        color='k', linestyle='--', linewidth=2)
            
            # Plot vertical redshift bin boundaries
            for i in range(len(z_bins)):
                if i == 0:
                    ymin_norm = (magnitude_cuts[0] - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
                elif i == len(z_bins) - 1:
                    ymin_norm = (magnitude_cuts[-1] - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
                else:
                    ymin_norm = (magnitude_cuts[i-1] - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
                
                plt.axvline(x=z_bins[i], ymin=ymin_norm, color='k', linestyle=':', linewidth=2)
        
        # Add KP3 cut if requested
        if add_kp3_cut:
            plt.axhline(-21.5, color='r', linestyle='-', linewidth=2, label='KP3 cut')
        
        # Set axis properties
        ax.set_ylim(-25, -17.5)
        ax.set_xlabel('z')
        ax.set_ylabel(f'Absolute {mag_label}')
        ax.invert_yaxis()
        
        # Add legend if KP3 cut is shown
        if add_kp3_cut:
            ax.legend()
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            ecorr_suffix = "_ecorr" if apply_extinction_correction else ""
            kp3_suffix = "_kp3" if add_kp3_cut else ""
            filename = f"absolute_magnitudes_{galaxy_type.lower()}{ecorr_suffix}{kp3_suffix}{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(
                filepath, dpi=self.plot_config.dpi,
                transparent=self.plot_config.transparent_background,
                bbox_inches="tight"
            )
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()
    
    def _load_lens_catalogue_for_magnitudes(self, mag_col: str) -> Optional[Table]:
        """
        Load lens catalogue with magnitude information for plotting.
        
        Parameters
        ----------
        mag_col : str
            Name of magnitude column to ensure is present
            
        Returns
        -------
        Optional[Table]
            Lens catalogue table or None if loading fails
        """
        try:
            # Use the existing data loader to get lens catalogues
            # We'll load for the first source survey (just need the lens data)
            source_survey = self.source_config.surveys[0]
            lens_table, _ = self.data_loader.load_lens_catalogues(source_survey)
            
            # Check if magnitude column exists
            if mag_col not in lens_table.colnames:
                # Try to add it using FastSpecFit data if available
                if mag_col == "ABSMAG01_SDSS_R" and "ABSMAG01_SDSS_R" not in lens_table.colnames:
                    self.logger.info(f"Attempting to add {mag_col} from FastSpecFit data")
                    lens_table = self._add_fastspecfit_magnitudes(lens_table)
                
                if mag_col not in lens_table.colnames:
                    self.logger.error(f"Magnitude column '{mag_col}' not found in catalogue")
                    return None
            
            return lens_table
            
        except Exception as e:
            self.logger.error(f"Failed to load lens catalogue: {e}")
            return None
    
    def _add_fastspecfit_magnitudes(self, lens_table: Table) -> Table:
        """
        Add FastSpecFit magnitude data to lens catalogue.
        
        Parameters
        ----------
        lens_table : Table
            Input lens catalogue
            
        Returns
        -------
        Table
            Catalogue with added magnitude columns
        """
        try:
            from ..utils import fastspecfit_utils
            
            # Determine program type based on galaxy type
            if self.lens_config.galaxy_type == "BGS_BRIGHT":
                prog = 'bright'
            elif self.lens_config.galaxy_type in ["LRG", "ELG"]:
                prog = 'dark'
            else:
                prog = 'bright'  # default
            
            # Try to load real FastSpecFit data
            try:
                lens_table_with_fsf = fastspecfit_utils.get_fastspecfit_magnitudes(
                    lens_table, prog=prog, release=self.lens_config.release, logger=self.logger
                )
                return lens_table_with_fsf
                
            except Exception as e:
                self.logger.error(f"Failed to load FastSpecFit data: {e}")
                return lens_table
                        
        except ImportError:
            self.logger.error("FastSpecFit utilities not available")
            return lens_table
        except Exception as e:
            self.logger.error(f"Failed to add FastSpecFit magnitudes: {e}")
            return lens_table
    
    def _create_magnitude_density_plot(self, fig: plt.Figure, x: np.ndarray, y: np.ndarray) -> plt.Axes:
        """
        Create a density plot using mpl_scatter_density.
        
        Parameters
        ----------
        fig : plt.Figure
            Figure object
        x : np.ndarray
            X coordinates (redshift)
        y : np.ndarray
            Y coordinates (magnitude)
            
        Returns
        -------
        plt.Axes
            Axes object with density plot
        """
        try:
            import mpl_scatter_density
            from matplotlib.colors import LinearSegmentedColormap
            
            # "Viridis-like" colormap with white background
            white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
                (0, '#ffffff'),
                (1e-20, '#440053'),
                (0.2, '#404388'),
                (0.4, '#2a788e'),
                (0.6, '#21a784'),
                (0.8, '#78d151'),
                (1, '#fde624'),
            ], N=256)
            
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            density = ax.scatter_density(x, y, cmap=white_viridis)
            fig.colorbar(density, label=r'$N_{\rm gal}$/pixel')
            return ax
            
        except ImportError:
            raise ImportError("mpl_scatter_density package not available")
    
    def _get_magnitude_mask(
        self, 
        lens_table: Table, 
        magnitude_cuts: np.ndarray, 
        z_bins: List[float], 
        mag_col: str,
        apply_extinction_correction: bool = False
    ) -> np.ndarray:
        """
        Create magnitude mask based on redshift-dependent cuts.
        
        Parameters
        ----------
        lens_table : Table
            Lens catalogue table
        magnitude_cuts : np.ndarray
            Magnitude cuts per redshift bin
        z_bins : List[float]
            Redshift bin edges
        mag_col : str
            Magnitude column name
        apply_extinction_correction : bool
            Whether extinction correction was applied
            
        Returns
        -------
        np.ndarray
            Boolean mask for galaxies passing magnitude cuts
        """
        z_col = 'Z_not4clus' if 'Z_not4clus' in lens_table.colnames else 'Z'
        redshifts = lens_table[z_col]
        magnitudes = lens_table[mag_col].copy()
        
        # Apply extinction correction if requested
        if apply_extinction_correction:
            ecorr = -0.8 * (redshifts - 0.1)
            magnitudes = magnitudes + ecorr
        
        # Create redshift mask (galaxies within redshift range)
        redshift_mask = (redshifts >= z_bins[0]) & (redshifts < z_bins[-1])
        
        # Assign redshift bins
        lens_zbins = np.digitize(redshifts, z_bins) - 1
        lens_zbins = np.clip(lens_zbins, 0, len(magnitude_cuts) - 1)
        
        # Create magnitude mask
        magnitude_mask = np.zeros(len(lens_table), dtype=bool)
        magnitude_mask[redshift_mask] = (magnitudes[redshift_mask] < magnitude_cuts[lens_zbins[redshift_mask]])
        
        return magnitude_mask
    
    def _compute_deltasigma_amplitude(
        self,
        data: np.ndarray,
        cov: np.ndarray,
        reference_dv: Optional[np.ndarray] = None,
        subtract_mean: bool = False
    ) -> Tuple[float, float]:
        """
        Compute lensing signal amplitude by fitting to a reference data vector.
        
        If reference_dv is provided, fits amplitude A such that data ≈ A * reference_dv.
        Otherwise, uses mean of the data vector.
        
        Parameters
        ----------
        data : np.ndarray
            Data vector
        cov : np.ndarray
            Covariance matrix
        reference_dv : Optional[np.ndarray]
            Reference data vector for amplitude fitting
        subtract_mean : bool
            Whether to subtract mean before fitting
            
        Returns
        -------
        Tuple[float, float]
            (amplitude, amplitude_error)
        """
        if reference_dv is None:
            # Just use mean as amplitude
            amplitude = np.mean(data)
            amplitude_err = np.sqrt(np.mean(np.diag(cov))) / np.sqrt(len(data))
            return amplitude, amplitude_err
        
        # Subtract mean if requested
        if subtract_mean:
            data = data - np.mean(data)
            reference_dv = reference_dv - np.mean(reference_dv)
        
        # Compute amplitude via weighted least squares
        # A = (ref^T C^-1 data) / (ref^T C^-1 ref)
        try:
            inv_cov = np.linalg.inv(cov)
            
            numerator = np.einsum('i,ij,j', reference_dv, inv_cov, data)
            denominator = np.einsum('i,ij,j', reference_dv, inv_cov, reference_dv)
            
            amplitude = numerator / denominator
            amplitude_err = 1.0 / np.sqrt(denominator)
            
            return amplitude, amplitude_err
        except np.linalg.LinAlgError:
            # Fall back to unweighted fit
            self.logger.warning("Covariance inversion failed, using unweighted amplitude")
            amplitude = np.sum(data * reference_dv) / np.sum(reference_dv * reference_dv)
            amplitude_err = np.sqrt(np.sum(np.diag(cov))) / np.sum(reference_dv * reference_dv)
            return amplitude, amplitude_err
    
    def _compute_sigma_sys(
        self,
        amplitudes: np.ndarray,
        errors: np.ndarray,
        method: str = "bayesian"
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        """
        Compute systematic uncertainty from scatter in measurements.
        
        Uses Bayesian inference to estimate the systematic uncertainty floor
        needed to make the scatter between measurements consistent with their
        stated uncertainties.
        
        Parameters
        ----------
        amplitudes : np.ndarray
            Array of amplitude measurements
        errors : np.ndarray
            Array of amplitude errors
        method : str
            Method for computing sigma_sys ('bayesian' or 'reduced_chisq')
            
        Returns
        -------
        Tuple[float, np.ndarray, Optional[np.ndarray]]
            (reduced_chisq, sigma_sys_quantiles, posterior_samples)
            where sigma_sys_quantiles is [16th, 50th, 84th, 95th] percentiles
        """
        if method == "bayesian":
            try:
                import pymc as pm
                import arviz as az
            except ImportError:
                self.logger.warning("PyMC not available, falling back to reduced chi-squared method")
                method = "reduced_chisq"
        
        # Compute reduced chi-squared without systematic uncertainty
        mean_amp = np.average(amplitudes, weights=1.0 / errors**2)
        chi2 = np.sum(((amplitudes - mean_amp) / errors)**2)
        dof = len(amplitudes) - 1
        reduced_chisq = chi2 / dof if dof > 0 else np.nan
        
        if method == "reduced_chisq":
            # Estimate sigma_sys from reduced chi-squared
            if reduced_chisq > 1:
                # sigma_sys such that chi2 / dof = 1
                sigma_sys = np.mean(errors) * np.sqrt(reduced_chisq - 1)
                sigma_sys_quantiles = np.array([sigma_sys, sigma_sys, sigma_sys, sigma_sys])
            else:
                sigma_sys_quantiles = np.array([0.0, 0.0, 0.0, 0.0])
            
            return reduced_chisq, sigma_sys_quantiles, None
        
        # Bayesian estimation
        try:
            with pm.Model() as model:
                # Prior for systematic uncertainty (half-normal)
                sigma_sys = pm.HalfNormal('sigma_sys', sigma=np.mean(errors))
                
                # Prior for true amplitude
                amp_true = pm.Normal('amp_true', mu=np.mean(amplitudes), sigma=np.std(amplitudes))
                
                # Likelihood
                total_error = pm.math.sqrt(errors**2 + sigma_sys**2)
                pm.Normal('obs', mu=amp_true, sigma=total_error, observed=amplitudes)
                
                # Sample
                trace = pm.sample(2000, tune=1000, return_inferencedata=True, 
                                 progressbar=False, random_seed=42)
            
            # Extract sigma_sys posterior
            sigma_sys_samples = trace.posterior['sigma_sys'].values.flatten()
            sigma_sys_quantiles = np.percentile(sigma_sys_samples, [16, 50, 84, 95])
            
            return reduced_chisq, sigma_sys_quantiles, sigma_sys_samples
            
        except Exception as e:
            self.logger.warning(f"Bayesian sigma_sys estimation failed: {e}. Using reduced chi-squared.")
            return self._compute_sigma_sys(amplitudes, errors, method="reduced_chisq")
    
    def _load_theory_covariance_for_bin(
        self,
        galaxy_type: str,
        source_survey: str,
        lens_bin: int,
        source_bin: int,
        statistic: str = "deltasigma",
        n_lens_bins: Optional[int] = None,
        n_source_bins: Optional[int] = None,
        n_rp_bins: int = 15
    ) -> Optional[np.ndarray]:
        """
        Load theory covariance and extract sub-matrix for specific lens-source bin combination.
        
        The theory covariance from Chris Hirata combines all lens and source bins.
        This method extracts the appropriate diagonal block for a specific lens-source
        bin combination.
        
        Parameters
        ----------
        galaxy_type : str
            Galaxy type (e.g., 'BGS_BRIGHT', 'LRG')
        source_survey : str
            Source survey name
        lens_bin : int
            Lens bin index
        source_bin : int
            Source bin index
        statistic : str
            Statistic type ('deltasigma' or 'gammat')
        n_lens_bins : int, optional
            Number of lens bins. If None, uses lens_config.
        n_source_bins : int, optional
            Number of source bins. If None, uses source_config.
        n_rp_bins : int
            Number of radial bins (default 15)
            
        Returns
        -------
        Optional[np.ndarray]
            Covariance sub-matrix for the specified bin, or None if not available
        """
        # Get the lens config for this galaxy type
        lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
        
        # Load full theory covariance
        full_cov = self.output_config.load_theory_covariance(
            galaxy_type, source_survey, statistic,
            z_bins=lens_config.z_bins
        )
        
        if full_cov is None:
            return None
        
        # Get number of bins
        if n_lens_bins is None:
            n_lens_bins = lens_config.get_n_lens_bins()
        if n_source_bins is None:
            n_source_bins = self.source_config.get_n_tomographic_bins(source_survey)
        
        # Calculate the index for this lens-source combination
        # Ordering is: lens_bin varies slowest, then source_bin, then rp
        # Combined index = lens_bin * (n_source_bins * n_rp_bins) + source_bin * n_rp_bins
        bin_index = lens_bin * (n_source_bins * n_rp_bins) + source_bin * n_rp_bins
        
        # Check if indices are valid
        total_expected = n_lens_bins * n_source_bins * n_rp_bins
        if full_cov.shape[0] != total_expected:
            self.logger.warning(
                f"Theory covariance size {full_cov.shape[0]} doesn't match expected "
                f"{total_expected} ({n_lens_bins}x{n_source_bins}x{n_rp_bins})"
            )
            # Try to infer n_rp_bins from covariance size
            n_rp_bins_inferred = full_cov.shape[0] // (n_lens_bins * n_source_bins)
            if n_rp_bins_inferred * n_lens_bins * n_source_bins == full_cov.shape[0]:
                n_rp_bins = n_rp_bins_inferred
                bin_index = lens_bin * (n_source_bins * n_rp_bins) + source_bin * n_rp_bins
                self.logger.info(f"Inferred n_rp_bins={n_rp_bins}")
            else:
                return None
        
        # Extract the diagonal block for this bin combination
        start_idx = bin_index
        end_idx = bin_index + n_rp_bins
        
        if end_idx > full_cov.shape[0]:
            self.logger.warning(f"Index out of bounds: {end_idx} > {full_cov.shape[0]}")
            return None
        
        cov_block = full_cov[start_idx:end_idx, start_idx:end_idx]
        return cov_block
    
    def _apply_deltaz_shifts(
        self,
        data: np.ndarray,
        z_lens: np.ndarray,
        z_source: np.ndarray,
        deltaz_shifts: List[float],
        galaxy_type: str,
        source_survey: str = "HSCY3"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply source redshift shifts to correct for photo-z biases.
        
        This adjusts the lensing signal amplitude to account for systematic
        shifts in source redshift distributions (e.g., from photo-z calibration).
        
        Parameters
        ----------
        data : np.ndarray
            Data vector (Delta Sigma or gamma_t values)
        z_lens : np.ndarray
            Lens redshifts corresponding to data
        z_source : np.ndarray
            Source redshifts corresponding to data
        deltaz_shifts : List[float]
            Redshift shifts per source tomographic bin (e.g., [0, 0, 0.115, 0.192])
        galaxy_type : str
            Galaxy type for accessing n(z) information
        source_survey : str
            Source survey name (default 'HSCY3')
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (corrected_data, z_lens, corrected_z_source)
        """
        try:
            from astropy.cosmology import Planck18 as desicosmo
        except ImportError:
            self.logger.warning("Astropy cosmology not available for deltaz shifts")
            return data, z_lens, z_source
        
        # Clone cosmology with h=1
        cosmo = desicosmo.clone(name='Planck18_h1', H0=100)
        
        # We need dsigma.physics for effective_critical_surface_density
        try:
            from dsigma.physics import effective_critical_surface_density
        except ImportError:
            self.logger.warning("dsigma not available for deltaz shifts")
            return data, z_lens, z_source
        
        data_corrected = np.copy(data)
        z_source_corrected = np.copy(z_source)
        
        n_data = len(data)
        n_lens_bins = self.lens_config.get_n_lens_bins()
        n_source_bins = len(deltaz_shifts)
        
        # Validate data size
        if n_data % n_lens_bins != 0:
            self.logger.warning("Data size not divisible by n_lens_bins, skipping deltaz shifts")
            return data, z_lens, z_source
        
        n_per_lens = n_data // n_lens_bins
        if n_per_lens % n_source_bins != 0:
            self.logger.warning("Data structure doesn't match expected source bins, skipping deltaz shifts")
            return data, z_lens, z_source
        
        n_rp = n_per_lens // n_source_bins
        
        # Load n(z) files (this would need to be adapted based on available data)
        # For now, we apply a simpler correction based on mean redshifts
        self.logger.info(f"Applying deltaz shifts: {deltaz_shifts}")
        
        for lens_bin in range(n_lens_bins):
            z_lens_mean = (self.lens_config.z_bins[lens_bin] + 
                          self.lens_config.z_bins[lens_bin + 1]) / 2.0
            
            for source_bin in range(n_source_bins):
                if deltaz_shifts[source_bin] == 0:
                    continue
                
                # Index into data vector
                start_idx = lens_bin * n_per_lens + source_bin * n_rp
                end_idx = start_idx + n_rp
                
                if end_idx > len(data_corrected):
                    continue
                
                # Simple amplitude correction based on critical surface density scaling
                # Sigma_crit ~ D_s / (D_l * D_ls)
                # Shifting z_s affects D_s and D_ls
                z_s_orig = np.mean(z_source[start_idx:end_idx])
                z_s_shifted = z_s_orig + deltaz_shifts[source_bin]
                
                # Skip if shifted redshift would be below lens redshift
                if z_s_shifted <= z_lens_mean:
                    continue
                
                # Compute approximate amplitude correction
                # Using cosmology to compute distance ratios
                D_s_orig = cosmo.angular_diameter_distance(z_s_orig).value
                D_s_shift = cosmo.angular_diameter_distance(z_s_shifted).value
                D_l = cosmo.angular_diameter_distance(z_lens_mean).value
                D_ls_orig = cosmo.angular_diameter_distance_z1z2(z_lens_mean, z_s_orig).value
                D_ls_shift = cosmo.angular_diameter_distance_z1z2(z_lens_mean, z_s_shifted).value
                
                # Sigma_crit ~ D_s / (D_l * D_ls)
                # Amplitude scales as Sigma_crit_new / Sigma_crit_orig
                sigma_crit_ratio = (D_s_shift / D_ls_shift) / (D_s_orig / D_ls_orig)
                
                # Apply correction
                data_corrected[start_idx:end_idx] *= sigma_crit_ratio
                z_source_corrected[start_idx:end_idx] += deltaz_shifts[source_bin]
                
                self.logger.debug(
                    f"L{lens_bin} S{source_bin}: dz={deltaz_shifts[source_bin]:.3f}, "
                    f"amplitude_shift={sigma_crit_ratio:.3f}"
                )
        
        return data_corrected, z_lens, z_source_corrected
    
    def plot_source_redshift_slope_tomographic(
        self,
        statistic: str = "deltasigma",
        scale_categories: Optional[List[str]] = None,
        save_plot: bool = True,
        filename_suffix: str = "",
        plot_slope: bool = True,
        plot_slope_uncertainty: bool = True,
        compute_sigma_sys: bool = True,
        sigma_sys_method: str = "bayesian",
        use_all_bins: bool = False,
        reference_datavector: Optional[Dict[str, np.ndarray]] = None,
        use_theory_covariance: bool = True,
        hscy3_deltaz_shifts: Optional[List[float]] = None,
        critical_sigma: float = 3.0,
        slope_color: str = "black",
        use_randoms_slope_uncertainty: bool = True,
        randoms_filename_suffix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Plot lensing amplitudes vs source redshift and fit slopes.
        
        This creates a multi-panel plot showing how lensing signal amplitudes
        vary with source redshift for different lens bins and scale categories.
        A slope in this plot could indicate systematic biases related to source
        redshift estimation.
        
        Parameters
        ----------
        statistic : str
            The statistic to analyze ('deltasigma' or 'gammat')
        scale_categories : List[str], optional
            List of scale categories to analyze. If None, uses analysis_config.analyzed_scales
        save_plot : bool
            Whether to save the plot to file
        filename_suffix : str
            Optional suffix for the filename
        plot_slope : bool
            Whether to plot the fitted slope line
        plot_slope_uncertainty : bool
            Whether to plot slope uncertainty band
        compute_sigma_sys : bool
            Whether to compute systematic uncertainty
        sigma_sys_method : str
            Method for sigma_sys calculation ('bayesian' or 'reduced_chisq')
        use_all_bins : bool
            Whether to use all source bins (ignoring allowed bins restrictions)
        reference_datavector : Dict[str, np.ndarray], optional
            Reference data vectors for amplitude normalization per lens bin
        use_theory_covariance : bool
            Whether to use theory covariance matrices (default True).
            Theory covariances are from Chris Hirata's model predictions.
        hscy3_deltaz_shifts : List[float], optional
            Source redshift shifts per HSCY3 tomographic bin (e.g., [0, 0, 0.115, 0.192])
            to correct for photo-z biases
        critical_sigma : float
            Highlight slopes more significant than this (default 3.0 sigma)
        slope_color : str
            Color for the fitted slope line (default 'black')
        use_randoms_slope_uncertainty : bool
            Whether to use the standard deviation of slopes from pre-computed
            random realizations as the slope uncertainty (default True).
            Falls back to polyfit uncertainty with a warning if randoms are
            not available.
        randoms_filename_suffix : Optional[str]
            Suffix for the randoms results files. If None, automatically
            determined based on use_all_bins ('_allbins' if True, '' otherwise).
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'slopes': Dict of fitted slopes per lens bin and scale
            - 'slope_errors': Dict of slope uncertainties
            - 'sigma_sys': Dict of systematic uncertainties
            - 'reduced_chisq': Dict of reduced chi-squared values
        """
        self.logger.info(f"Plotting source redshift slope for {statistic}")
        
        if use_theory_covariance:
            self.logger.info("Using theory covariance matrices")
        else:
            self.logger.info("Using jackknife covariance matrices")
        
        if hscy3_deltaz_shifts is not None:
            self.logger.info(f"Will apply HSCY3 deltaz shifts: {hscy3_deltaz_shifts}")
        
        # Use configured scale categories if not provided
        if scale_categories is None:
            scale_categories = self.analysis_config.get_scale_categories()
        
        n_scales = len(scale_categories)
        n_galaxy_types = len(self.galaxy_types)
        
        # Calculate total columns using total_bins (same pattern as plot_datavector_tomographic)
        n_total_cols = self.total_bins
        
        # Build lens_bin_info: Dict[int, Tuple[str, int]] mapping column -> (galaxy_type, lens_bin)
        lens_bin_info: Dict[int, Tuple[str, int]] = {}
        for galaxy_type in self.galaxy_types:
            start_col, end_col = self.bin_layout[galaxy_type]
            n_lens_bins = self.analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
            for lens_bin in range(n_lens_bins):
                col_idx = start_col + lens_bin
                lens_bin_info[col_idx] = (galaxy_type, lens_bin)
        
        # Pre-load theory covariances for each galaxy type and survey
        theory_covs: Dict[str, Dict[str, np.ndarray]] = {}
        if use_theory_covariance:
            for galaxy_type in self.galaxy_types:
                lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
                theory_covs[galaxy_type] = {}
                for source_survey in self.source_config.surveys:
                    theory_cov = self.output_config.load_theory_covariance(
                        galaxy_type, source_survey, statistic,
                        z_bins=lens_config.z_bins
                    )
                    if theory_cov is not None:
                        theory_covs[galaxy_type][source_survey] = theory_cov
                        self.logger.info(f"Loaded theory covariance for {galaxy_type} {source_survey} (shape: {theory_cov.shape})")
                    else:
                        self.logger.warning(f"Theory covariance not available for {galaxy_type} {source_survey}, will use jackknife")
        
        # Load randoms results for slope uncertainty (for each galaxy type)
        randoms_data: Dict[str, Tuple] = {}  # galaxy_type -> (p_arr, keys)
        if use_randoms_slope_uncertainty:
            from .randoms import create_randoms_analyzer_from_configs
            for galaxy_type in self.galaxy_types:
                lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
                try:
                    randoms_analyzer = create_randoms_analyzer_from_configs(
                        self.computation_config, lens_config,
                        self.source_config, self.output_config, self.logger
                    )
                    # Determine suffix based on use_all_bins if not provided, and add galaxy type
                    base_suffix = randoms_filename_suffix if randoms_filename_suffix is not None else ("_allbins" if use_all_bins else "")
                    suffix = f"{base_suffix}_{galaxy_type.lower()}"
                    randoms_p_arr, _, randoms_keys = randoms_analyzer.load_randoms_results(
                        "redshift_slope_tomo", suffix
                    )
                    randoms_keys = list(randoms_keys)  # Convert to list for .index()
                    randoms_data[galaxy_type] = (randoms_p_arr, randoms_keys)
                    self.logger.info(f"Loaded randoms results for {galaxy_type} with {len(randoms_keys)} keys")
                except FileNotFoundError as e:
                    self.logger.warning(f"Randoms results not found for {galaxy_type}, falling back to polyfit uncertainty: {e}")
                except Exception as e:
                    self.logger.warning(f"Error loading randoms results for {galaxy_type}: {e}")
        
        # Setup figure - rows are scale categories, columns are lens bins across all galaxy types
        fig_width = 7.24
        fig_height = fig_width / n_total_cols * n_scales
        fig, axes, gs = self._initialize_gridspec_figure(
            (fig_width, fig_height), n_scales, n_total_cols,
            add_cbar=True, hspace=0, wspace=0
        )
        
        # Storage for results
        results = {
            'slopes': {},
            'slope_errors': {},
            'sigma_sys': {},
            'reduced_chisq': {},
            'amplitudes': {},
            'zsource_mean': {}
        }
        
        # Iterate over scale categories and columns (lens bins across all galaxy types)
        for scale_idx, scale_category in enumerate(scale_categories):
            for col_idx in range(n_total_cols):
                galaxy_type, lens_bin = lens_bin_info[col_idx]
                lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
                n_lens_bins = lens_config.get_n_lens_bins()
                
                ax = axes[scale_idx, col_idx]
                
                # Set titles
                if scale_idx == 0:
                    galaxy_type_short = galaxy_type[:3]
                    ax.set_title(f"{galaxy_type_short} Bin {lens_bin + 1}", fontsize=10)
                
                # Collect amplitudes and source redshifts across surveys
                all_amplitudes = []
                all_amplitude_errors = []
                all_zsource = []
                all_colors = []
                
                for survey_idx, source_survey in enumerate(self.source_config.surveys):
                    n_tomo_bins = self.source_config.get_n_tomographic_bins(source_survey)
                    
                    # Get allowed source bins
                    if use_all_bins:
                        allowed_bins = list(range(n_tomo_bins))
                    else:
                        z_max_bin = lens_config.z_bins[lens_bin + 1]
                        allowed_bins = self.analysis_config.get_allowed_source_bins(
                            galaxy_type, source_survey, z_max_bin
                        )
                    
                    for source_bin in allowed_bins:
                        # Load results
                        results_table = self._load_lensing_results_for_galaxy_type(
                            galaxy_type, lens_bin, source_survey, source_bin, statistic
                        )
                        
                        if results_table is None:
                            self.logger.warning(
                                f"No data for {galaxy_type} {source_survey} l{lens_bin} s{source_bin}"
                            )
                            continue
                        
                        # Load covariance - use theory if available, else jackknife
                        if galaxy_type in theory_covs and source_survey in theory_covs[galaxy_type]:
                            # Extract per-bin covariance from full theory covariance
                            cov = self._load_theory_covariance_for_bin(
                                galaxy_type, source_survey, lens_bin, source_bin,
                                statistic, n_lens_bins, n_tomo_bins
                            )
                            if cov is None:
                                # Fall back to jackknife
                                cov = self._load_covariance_matrix_for_galaxy_type(
                                    galaxy_type, lens_bin, source_survey, source_bin, statistic
                                )
                        else:
                            cov = self._load_covariance_matrix_for_galaxy_type(
                                galaxy_type, lens_bin, source_survey, source_bin, statistic
                            )
                        
                        if cov is None:
                            self.logger.warning(
                                f"No covariance for {galaxy_type} {source_survey} l{lens_bin} s{source_bin}"
                            )
                            continue
                        
                        # Extract data based on statistic
                        if statistic == "deltasigma":
                            rp = results_table['rp']
                            signal = results_table['ds']
                            zsource_col = 'z_source' if 'z_source' in results_table.colnames else None
                        else:
                            rp = results_table['theta']
                            signal = results_table['et']
                            zsource_col = 'z_source' if 'z_source' in results_table.colnames else None
                        
                        # Apply scale cuts for this category
                        scale_mask = self._get_scale_category_mask(
                            rp, scale_category, statistic, galaxy_type, lens_bin
                        )
                        
                        if not np.any(scale_mask):
                            continue
                        
                        signal_cut = signal[scale_mask]
                        cov_cut = cov[np.ix_(scale_mask, scale_mask)]
                        
                        # Get reference data vector - load from output_config or use provided
                        ref_dv = None
                        if reference_datavector is not None:
                            ref_key = f"{galaxy_type}_l{lens_bin}_{scale_category}"
                            if ref_key in reference_datavector:
                                ref_dv = reference_datavector[ref_key][scale_mask]
                        else:
                            # Load from output_config
                            ref_dv_full = self.output_config.load_reference_datavector(
                                galaxy_type, lens_bin
                            )
                            if ref_dv_full is not None and len(ref_dv_full) >= len(scale_mask):
                                ref_dv = ref_dv_full[scale_mask]
                        
                        # Compute amplitude
                        amplitude, amplitude_err = self._compute_deltasigma_amplitude(
                            signal_cut, cov_cut, ref_dv
                        )
                        
                        # Get mean source redshift (from table or use survey defaults)
                        if zsource_col and zsource_col in results_table.colnames:
                            zsource_mean = np.mean(results_table[zsource_col][scale_mask])
                        else:
                            # Use default source redshift for this survey and bin
                            zsource_mean = self._get_default_source_redshift(source_survey, source_bin)
                        
                        all_amplitudes.append(amplitude)
                        all_amplitude_errors.append(amplitude_err)
                        all_zsource.append(zsource_mean)
                        all_colors.append(self.color_list[survey_idx % len(self.color_list)])
                        
                        # Plot point
                        ax.errorbar(
                            zsource_mean, amplitude, amplitude_err,
                            fmt='o', color=self.color_list[survey_idx % len(self.color_list)],
                            markersize=3, capsize=1.5, alpha=0.8
                        )
                
                # Convert to arrays
                all_amplitudes = np.array(all_amplitudes)
                all_amplitude_errors = np.array(all_amplitude_errors)
                all_zsource = np.array(all_zsource)
                
                # Fit slope if we have enough points
                if len(all_amplitudes) >= 2:
                    # Fit linear slope
                    if len(all_amplitudes) > 2:
                        p, V = np.polyfit(all_zsource, all_amplitudes, 1,
                                        w=1/all_amplitude_errors, cov=True)
                        slope, intercept = p
                        polyfit_slope_err = np.sqrt(V[0, 0])
                    else:
                        p = np.polyfit(all_zsource, all_amplitudes, 1,
                                     w=1/all_amplitude_errors)
                        slope, intercept = p
                        V = np.zeros((2, 2))
                        V[0, 0] = np.nan
                        polyfit_slope_err = np.nan
                    
                    # Get slope uncertainty from randoms if available
                    if use_randoms_slope_uncertainty and galaxy_type in randoms_data:
                        randoms_p_arr, randoms_keys = randoms_data[galaxy_type]
                        # Map key format: plotting uses 'l{lens_bin}', randoms uses '{lens_bin}'
                        randoms_key = f"{galaxy_type}_{scale_category}_{lens_bin}"
                        if randoms_key in randoms_keys:
                            key_idx = randoms_keys.index(randoms_key)
                            slope_err = np.std(randoms_p_arr[:, key_idx, 0])
                            # Also get covariance for uncertainty band
                            V = np.cov(randoms_p_arr[:, key_idx, :].T)
                            self.logger.info(f"Loaded randoms results for {randoms_key}, slope error: {slope_err}, covariance: {V}")
                        else:
                            self.logger.warning(
                                f"Key {randoms_key} not found in randoms, using polyfit uncertainty. Available keys: {randoms_keys}"
                            )
                            slope_err = polyfit_slope_err
                    else:
                        slope_err = polyfit_slope_err
                    
                    # Store results
                    key = f"{galaxy_type}_{scale_category}_l{lens_bin}"
                    results['slopes'][key] = slope
                    results['slope_errors'][key] = slope_err
                    results['amplitudes'][key] = all_amplitudes
                    results['zsource_mean'][key] = all_zsource
                    
                    # Compute systematic uncertainty
                    if compute_sigma_sys and len(all_amplitudes) >= 3:
                        reduced_chisq, sigma_sys_quantiles, _ = self._compute_sigma_sys(
                            all_amplitudes, all_amplitude_errors, method=sigma_sys_method
                        )
                        results['sigma_sys'][key] = sigma_sys_quantiles
                        results['reduced_chisq'][key] = reduced_chisq
                        
                        # Add sigma_sys annotation
                        if np.isfinite(sigma_sys_quantiles[1]):
                            sigma_sys_med = sigma_sys_quantiles[1]
                            sigma_sys_upper = sigma_sys_quantiles[2] - sigma_sys_quantiles[1]
                            sigma_sys_lower = sigma_sys_quantiles[1] - sigma_sys_quantiles[0]
                            
                            if sigma_sys_lower > 0:
                                sigma_text = f"$\\sigma_{{\\rm sys}}={sigma_sys_med:.2f}^{{+{sigma_sys_upper:.2f}}}_{{-{sigma_sys_lower:.2f}}}$"
                            else:
                                sigma_text = f"$\\sigma_{{\\rm sys}}\\leq {sigma_sys_med:.2f}+{sigma_sys_upper:.2f}$"
                            
                            ax.text(
                                0.5, 0.3, sigma_text,
                                transform=ax.transAxes, ha='center', fontsize=7,
                                path_effects=[PathEffects.withStroke(linewidth=1, foreground='white')]
                            )
                    
                    # Plot slope
                    if plot_slope and np.isfinite(slope):
                        z_range = np.linspace(np.min(all_zsource), np.max(all_zsource), 100)
                        y_fit = slope * z_range + intercept
                        ax.plot(z_range, y_fit, color=slope_color, linestyle='--', 
                               alpha=0.7, linewidth=1)
                        
                        # Plot uncertainty band
                        if plot_slope_uncertainty and np.isfinite(slope_err):
                            dy = np.sqrt((z_range**2 * V[0, 0]) + V[1, 1] + 2 * z_range * V[0, 1])
                            ax.fill_between(z_range, y_fit - dy, y_fit + dy, 
                                          color='gray', alpha=0.3)
                        
                        # Add slope annotation
                        slope_text = f"$\\beta={slope:.2f}\\pm {slope_err:.2f}$"
                        # Highlight if slope is significant
                        if np.isfinite(slope_err) and np.abs(slope) > critical_sigma * slope_err:
                            bbox = dict(facecolor='gray', edgecolor='red', 
                                      boxstyle='round,pad=0.2', alpha=0.2)
                        else:
                            bbox = None
                        
                        ax.text(
                            0.5, 0.15, slope_text,
                            transform=ax.transAxes, ha='center', fontsize=7,
                            bbox=bbox,
                            path_effects=[PathEffects.withStroke(linewidth=1, foreground='white')]
                        )
                
                # Set axis labels
                if scale_idx == n_scales - 1:  # Bottom row
                    ax.set_xlabel(r"$\langle z_{\rm source}\rangle$", fontsize=9)
                
                if col_idx == 0:  # Left column
                    ylabel = f"{scale_category}\n$A_{{\\Delta\\Sigma}}$"
                    ax.set_ylabel(ylabel, fontsize=9)
        
        # Add survey colorbar legend
        self._add_survey_colorbar_legend(
            fig, axes, gs,
            self.color_list[:len(self.source_config.surveys)],
            self.source_config.surveys
        )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            allbins_suffix = "_allbins" if use_all_bins else ""
            deltaz_suffix = "_hscy3_deltaz" if hscy3_deltaz_shifts is not None else ""
            theory_suffix = "_theory_cov" if use_theory_covariance else "_jackknife_cov"
            filename = f"{statistic}_source_redshift_slope_tomo{allbins_suffix}{deltaz_suffix}{theory_suffix}{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(
                filepath, dpi=self.plot_config.dpi,
                transparent=self.plot_config.transparent_background,
                bbox_inches="tight"
            )
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()
        
        return results
    
    def _get_scale_category_mask(
        self,
        rp: np.ndarray,
        scale_category: str,
        statistic: str,
        galaxy_type: str,
        lens_bin: int
    ) -> np.ndarray:
        """
        Get mask for a specific scale category (small, large, or all scales).
        
        Parameters
        ----------
        rp : np.ndarray
            Radial or angular separations
        scale_category : str
            Scale category name ('small scales', 'large scales', 'all scales')
        statistic : str
            Statistic type
        galaxy_type : str
            Galaxy type
        lens_bin : int
            Lens bin index
            
        Returns
        -------
        np.ndarray
            Boolean mask for scales in this category
        """
        # Get scale cuts from first survey (assuming same for all)
        scale_cuts = self.analysis_config.get_scale_cuts(
            self.source_config.surveys[0], statistic
        )
        
        # Convert scale cuts to same units as rp
        if statistic == "deltasigma":
            rpmin, rpmax = self._get_rp_from_deg(
                scale_cuts["min_deg"], scale_cuts["max_deg"], galaxy_type, lens_bin
            )
            rp_pivot = scale_cuts["rp_pivot"]
        else:
            # For gammat, convert degrees to arcmin
            rpmin = scale_cuts["min_deg"] * 60.0
            rpmax = scale_cuts["max_deg"] * 60.0
            rp_pivot = scale_cuts["rp_pivot"] * 60.0
        
        # Create mask based on category
        if "small" in scale_category.lower():
            mask = (rp >= rpmin) & (rp < rp_pivot)
        elif "large" in scale_category.lower():
            mask = (rp >= rp_pivot) & (rp <= rpmax)
        else:  # "all scales"
            mask = (rp >= rpmin) & (rp <= rpmax)
        
        return mask
    
    def _get_default_source_redshift(self, source_survey: str, source_bin: int) -> float:
        """
        Get default mean source redshift for a survey and tomographic bin.
        
        This uses typical values from the literature for each survey.
        
        Parameters
        ----------
        source_survey : str
            Source survey name
        source_bin : int
            Source tomographic bin index
            
        Returns
        -------
        float
            Mean source redshift
        """
        # Default values based on typical survey properties
        default_redshifts = {
            "SDSS": {0: 0.3},
            "KiDS": {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.9, 4: 1.2},
            "DES": {0: 0.4, 1: 0.6, 2: 0.8, 3: 1.1},
            "DECADE": {0: 0.4, 1: 0.6, 2: 0.8, 3: 1.1},
            "DECADE_NGC": {0: 0.4, 1: 0.6, 2: 0.8, 3: 1.1},
            "DECADE_SGC": {0: 0.4, 1: 0.6, 2: 0.8, 3: 1.1},
            "HSCY1": {0: 0.6, 1: 0.8, 2: 1.0, 3: 1.3},
            "HSCY3": {0: 0.6, 1: 0.8, 2: 1.0, 3: 1.3},
        }
        
        survey_upper = source_survey.upper()
        if survey_upper in default_redshifts:
            if source_bin in default_redshifts[survey_upper]:
                return default_redshifts[survey_upper][source_bin]
        
        # Fallback to simple linear scaling
        return 0.5 + 0.25 * source_bin
    
    def plot_splits(
        self,
        split_by: str,
        n_splits: int = 4,
        statistic: str = "deltasigma",
        scale_categories: Optional[List[str]] = None,
        use_randoms_uncertainty: bool = False,
        plot_slope: bool = True,
        plot_slope_uncertainty: bool = True,
        critical_sigma: float = 3.0,
        slope_color: str = "black",
        save_plot: bool = True,
        filename_suffix: str = "",
        boost_correction: bool = False
    ) -> Dict[str, Any]:
        """
        Plot lensing amplitudes vs split property values with slope fitting.
        
        This creates a multi-panel plot showing how lensing signal amplitudes
        vary with a split property (e.g., NTILE, RA, DEC) for different lens bins 
        and scale categories. A non-zero slope indicates systematic bias.
        
        Supports both single galaxy type and multiple galaxy types.
        
        Parameters
        ----------
        split_by : str
            Property to split by (e.g., 'NTILE', 'ra', 'dec', 'LOGMSTAR')
        n_splits : int
            Number of splits
        statistic : str
            The statistic to analyze ('deltasigma' or 'gammat')
        scale_categories : List[str], optional
            List of scale categories to analyze. If None, uses analysis_config.analyzed_scales
        use_randoms_uncertainty : bool
            Whether to use randoms for slope uncertainty estimation
        plot_slope : bool
            Whether to plot the fitted slope line
        plot_slope_uncertainty : bool
            Whether to plot slope uncertainty band
        critical_sigma : float
            Highlight slopes more significant than this (in units of sigma)
        slope_color : str
            Color for slope line
        save_plot : bool
            Whether to save the plot to file
        filename_suffix : str
            Optional suffix for the filename
        boost_correction : bool
            Whether boost correction was applied in splits computation
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'slopes': Dict of fitted slopes per lens bin and scale
            - 'slope_errors': Dict of slope uncertainties
            - 'p_values': Dict of p-values from randoms (if available)
        """
        self.logger.info(f"Plotting splits for property: {split_by}")
        
        # Use configured scale categories if not provided
        if scale_categories is None:
            scale_categories = self.analysis_config.get_scale_categories()
        
        n_scales = len(scale_categories)
        n_surveys = len(self.source_config.surveys)
        
        # Load randoms results if requested (for each galaxy type)
        randoms_p_std = None
        randoms_V_std = None
        randoms_keys = None
        if use_randoms_uncertainty:
            try:
                from .randoms import create_randoms_analyzer_from_configs
                
                # Collect randoms from all galaxy types
                all_p_arrs = []
                all_keys = []
                
                for lens_config in self._lens_configs:
                    randoms_analyzer = create_randoms_analyzer_from_configs(
                        self.computation_config, lens_config,
                        self.source_config, self.output_config, self.logger
                    )
                    # Include galaxy type in suffix
                    galaxy_suffix = f"_{lens_config.galaxy_type.lower()}"
                    full_suffix = f"{filename_suffix}{galaxy_suffix}"
                    try:
                        p_arr, V_arr, keys = randoms_analyzer.load_randoms_results(
                            "splits", full_suffix
                        )
                        all_p_arrs.append(p_arr)
                        all_keys.extend(keys)
                        self.logger.info(f"Loaded randoms results for {lens_config.galaxy_type} with {len(keys)} keys")
                    except FileNotFoundError as e:
                        self.logger.warning(f"Randoms not found for {lens_config.galaxy_type}: {e}")
                
                if all_p_arrs:
                    # Concatenate results from all galaxy types
                    combined_p_arr = np.concatenate(all_p_arrs, axis=1)
                    randoms_keys = all_keys
                    randoms_p_std = np.std(combined_p_arr, axis=0)
                    # Compute covariance for each key
                    randoms_V_std = np.zeros((*randoms_p_std.shape, randoms_p_std.shape[-1]))
                    for i in range(combined_p_arr.shape[1]):
                        randoms_V_std[i] = np.cov(combined_p_arr[:, i, :].T)
                    self.logger.info(f"Loaded combined randoms results with {len(randoms_keys)} keys")
                else:
                    self.logger.warning("No randoms results found for any galaxy type")
                    use_randoms_uncertainty = False
            except Exception as e:
                self.logger.warning(f"Could not load randoms results: {e}")
                use_randoms_uncertainty = False
        
        # Setup figure - rows are scale categories, columns are total bins across galaxy types
        fig_width = 7.24
        fig_height = fig_width / self.total_bins * n_scales
        fig, axes, gs = self._initialize_gridspec_figure(
            (fig_width, fig_height), n_scales, self.total_bins,
            add_cbar=True, hspace=0, wspace=0
        )
        
        # Storage for results
        results = {
            'slopes': {},
            'slope_errors': {},
            'intercepts': {},
            'slope_covariances': {}
        }
        
        # Iterate over scale categories
        for scale_idx, scale_category in enumerate(scale_categories):
            # Iterate over galaxy types and their bins
            for galaxy_type in self.galaxy_types:
                lens_config = self._get_lens_config_for_galaxy_type(galaxy_type)
                if lens_config is None:
                    continue
                
                start_col, end_col = self.bin_layout[galaxy_type]
                n_lens_bins = self.analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
                version = lens_config.get_catalogue_version()
                z_bins = lens_config.z_bins
                
                for lens_bin in range(n_lens_bins):
                    col_idx = start_col + lens_bin
                    ax = axes[scale_idx, col_idx]
                    
                    # Set titles for top row
                    if scale_idx == 0:
                        galaxy_type_short = galaxy_type[:3]
                        ax.set_title(f"{galaxy_type_short} Bin {lens_bin + 1}", fontsize=10)
                    
                    # Collect data across splits and surveys
                    all_split_values = []
                    all_amplitudes = []
                    all_amplitude_errors = []
                    all_colors = []
                    
                    for survey_idx, source_survey in enumerate(self.source_config.surveys):
                        # Get reference data vector for this lens bin
                        ref_dv = self.output_config.load_reference_datavector(
                            galaxy_type, lens_bin
                        )
                        
                        for split in range(n_splits):
                            # Load split results
                            split_results = self.output_config.load_split_results(
                                version, galaxy_type, source_survey, lens_bin, z_bins,
                                statistic, split_by, split, n_splits, boost_correction
                            )
                            
                            # Load split covariance
                            split_cov = self.output_config.load_split_covariance(
                                version, galaxy_type, source_survey, lens_bin, z_bins,
                                statistic, split_by, split, n_splits, boost_correction
                            )
                            
                            # Load split value
                            split_value_data = self.output_config.load_split_value(
                                version, galaxy_type, source_survey, lens_bin, z_bins,
                                statistic, split_by, split, n_splits, boost_correction
                            )
                            
                            if split_results is None or split_cov is None:
                                self.logger.warning(
                                    f"Missing data for {galaxy_type} {source_survey} "
                                    f"l{lens_bin} {split_by}={split}"
                                )
                                continue
                            
                            # Extract data based on statistic
                            if statistic == "deltasigma":
                                rp = split_results['rp']
                                signal = split_results['ds']
                            else:
                                rp = split_results['theta']
                                signal = split_results['et']
                            
                            # Apply scale cuts for this category
                            scale_mask = self._get_scale_category_mask(
                                rp, scale_category, statistic, galaxy_type, lens_bin
                            )
                            
                            if not np.any(scale_mask):
                                continue
                            
                            signal_cut = signal[scale_mask]
                            cov_cut = split_cov[np.ix_(scale_mask, scale_mask)]
                            
                            # Get reference vector with same scale cuts
                            if ref_dv is not None and len(ref_dv) >= len(scale_mask):
                                ref_dv_cut = ref_dv[scale_mask]
                            else:
                                ref_dv_cut = None
                            
                            # Compute amplitude
                            amplitude, amplitude_err = self._compute_deltasigma_amplitude(
                                signal_cut, cov_cut, ref_dv_cut
                            )
                            
                            # Get split value (use mean)
                            if split_value_data is not None:
                                split_value = split_value_data[0]  # mean value
                            else:
                                # Fallback: use split index
                                split_value = float(split)
                            
                            if np.isfinite(amplitude) and np.isfinite(amplitude_err):
                                all_split_values.append(split_value)
                                all_amplitudes.append(amplitude)
                                all_amplitude_errors.append(amplitude_err)
                                all_colors.append(self.color_list[survey_idx % len(self.color_list)])
                                
                                # Plot point
                                ax.errorbar(
                                    split_value, amplitude, amplitude_err,
                                    fmt='o', color=self.color_list[survey_idx % len(self.color_list)],
                                    markersize=3, capsize=1.5, alpha=0.8
                                )
                    
                    # Convert to arrays
                    all_split_values = np.array(all_split_values)
                    all_amplitudes = np.array(all_amplitudes)
                    all_amplitude_errors = np.array(all_amplitude_errors)
                    
                    # Fit slope if we have enough points
                    if len(all_amplitudes) >= 2:
                        # Fit linear slope with inverse variance weighting
                        if len(all_amplitudes) > 2:
                            try:
                                p, V = np.polyfit(
                                    all_split_values, all_amplitudes, 1,
                                    w=1/all_amplitude_errors**2, cov=True
                                )
                                slope, intercept = p
                                slope_err = np.sqrt(V[0, 0])
                            except (np.linalg.LinAlgError, ValueError):
                                p = np.polyfit(all_split_values, all_amplitudes, 1)
                                slope, intercept = p
                                V = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                                slope_err = np.nan
                        else:
                            p = np.polyfit(all_split_values, all_amplitudes, 1)
                            slope, intercept = p
                            V = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                            slope_err = np.nan
                        
                        # Store results
                        key = f"{galaxy_type}_{scale_category}_l{lens_bin}"
                        results['slopes'][key] = slope
                        results['slope_errors'][key] = slope_err
                        results['intercepts'][key] = intercept
                        results['slope_covariances'][key] = V
                        
                        # Determine slope uncertainty to use
                        if use_randoms_uncertainty and randoms_keys is not None:
                            randoms_key = f"{split_by}_{galaxy_type}_{scale_category}_{lens_bin}"
                            idx = None
                            for i, rk in enumerate(randoms_keys):
                                if rk == randoms_key:
                                    idx = i
                                    break
                            if idx is not None:
                                slope_uncertainty_val = randoms_p_std[idx][0]
                                slope_covmat = randoms_V_std[idx]
                            else:
                                slope_uncertainty_val = slope_err
                                slope_covmat = V
                        else:
                            slope_uncertainty_val = slope_err
                            slope_covmat = V
                        
                        # Plot slope line
                        if plot_slope and np.isfinite(slope):
                            x_range = np.linspace(
                                np.min(all_split_values), np.max(all_split_values), 100
                            )
                            y_fit = slope * x_range + intercept
                            ax.plot(x_range, y_fit, color=slope_color, linestyle='--', 
                                   alpha=0.7, linewidth=1)
                            
                            # Plot uncertainty band
                            if plot_slope_uncertainty and np.all(np.isfinite(slope_covmat)):
                                dy = np.sqrt(
                                    (x_range**2 * slope_covmat[0, 0]) + 
                                    slope_covmat[1, 1] + 
                                    2 * x_range * slope_covmat[0, 1]
                                )
                                ax.fill_between(x_range, y_fit - dy, y_fit + dy,
                                              color='gray', alpha=0.3)
                            
                            # Add slope annotation
                            if np.isfinite(slope_uncertainty_val):
                                slope_text = f"$\\beta={slope:.2f}\\pm {slope_uncertainty_val:.2f}$"
                                # Highlight if slope is significant
                                if np.abs(slope) > critical_sigma * slope_uncertainty_val:
                                    bbox = dict(facecolor='gray', edgecolor='red',
                                              boxstyle='round,pad=0.2', alpha=0.2)
                                else:
                                    bbox = None
                                
                                ax.text(
                                    0.5, 0.85, slope_text,
                                    transform=ax.transAxes, ha='center', fontsize=8,
                                    bbox=bbox
                                )
                    
                    # Set axis labels
                    if scale_idx == n_scales - 1:  # Bottom row
                        ax.set_xlabel(f"{split_by}", fontsize=9)
                    
                    if col_idx == 0:  # Left column
                        ylabel = f"{scale_category}\n$A_{{\\Delta\\Sigma}}$"
                        ax.set_ylabel(ylabel, fontsize=9)
        
        # Add survey colorbar legend
        self._add_survey_colorbar_legend(
            fig, axes, gs,
            self.color_list[:n_surveys],
            self.source_config.surveys
        )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            # Include galaxy types in filename for multi-galaxy plots
            galaxy_suffix = "_" + "_".join(self.galaxy_types) if len(self.galaxy_types) > 1 else ""
            filename = f"{statistic}_split_by_{split_by}{galaxy_suffix}{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(
                filepath, dpi=self.plot_config.dpi,
                transparent=self.plot_config.transparent_background,
                bbox_inches="tight"
            )
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()
        
        return results
    
    def plot_all_splits(
        self,
        splits_to_consider: List[str],
        n_splits: int = 4,
        statistic: str = "deltasigma",
        scale_categories: Optional[List[str]] = None,
        use_randoms_uncertainty: bool = False,
        plot_slope: bool = True,
        plot_slope_uncertainty: bool = True,
        critical_sigma: float = 3.0,
        save_plot: bool = True,
        filename_suffix: str = "",
        boost_correction: bool = False,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Plot amplitude vs split value for multiple split properties.
        
        This is a convenience method that loops over all split types and calls
        plot_splits() for each.
        
        Parameters
        ----------
        splits_to_consider : List[str]
            List of properties to split by (e.g., ['NTILE', 'ra', 'dec', 'LOGMSTAR'])
        n_splits : int
            Number of splits (can be overridden for NTILE)
        statistic : str
            The statistic to analyze ('deltasigma' or 'gammat')
        scale_categories : List[str], optional
            List of scale categories to analyze
        use_randoms_uncertainty : bool
            Whether to use randoms for slope uncertainty estimation
        plot_slope : bool
            Whether to plot the fitted slope line
        plot_slope_uncertainty : bool
            Whether to plot slope uncertainty band
        critical_sigma : float
            Highlight slopes more significant than this (in units of sigma)
        save_plot : bool
            Whether to save the plots to files
        filename_suffix : str
            Optional suffix for the filenames
        boost_correction : bool
            Whether boost correction was applied
        verbose : bool
            Whether to print progress messages
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping split_by to results dictionaries
        """
        self.logger.info(f"Plotting all splits: {splits_to_consider}")
        
        all_results = {}
        
        for split_by in splits_to_consider:
            if verbose:
                print(f"Processing split: {split_by}")
            
            try:
                # Handle NTILE specially if needed (galaxy-type specific n_splits)
                current_n_splits = n_splits
                if split_by.lower() == "ntile":
                    # Could add logic here to use galaxy-type specific n_splits
                    pass
                
                results = self.plot_splits(
                    split_by=split_by,
                    n_splits=current_n_splits,
                    statistic=statistic,
                    scale_categories=scale_categories,
                    use_randoms_uncertainty=use_randoms_uncertainty,
                    plot_slope=plot_slope,
                    plot_slope_uncertainty=plot_slope_uncertainty,
                    critical_sigma=critical_sigma,
                    save_plot=save_plot,
                    filename_suffix=filename_suffix,
                    boost_correction=boost_correction
                )
                
                # Store results with split_by prefix
                for key, value in results.get('slopes', {}).items():
                    all_results[f"{split_by}_{key}"] = {
                        'slope': value,
                        'slope_error': results.get('slope_errors', {}).get(key),
                        'intercept': results.get('intercepts', {}).get(key),
                        'slope_covariance': results.get('slope_covariances', {}).get(key)
                    }
                    
            except Exception as e:
                self.logger.error(f"Could not process split {split_by}: {e}")
                if verbose:
                    print(f"  Error: {e}")
        
        return all_results
    
    def plot_footprint(
        self,
        surveys_to_overlay: Optional[List[str]] = None,
        nside: int = 64,
        smoothing: float = 0.0,
        **kwargs
    ) -> None:
        """
        Plot footprint using FootprintPlotter.
        
        This is a convenience method that creates a FootprintPlotter and calls
        its plot_lens_footprint method.
        
        Parameters
        ----------
        surveys_to_overlay : List[str], optional
            List of surveys to overlay
        nside : int
            HEALPix nside parameter
        smoothing : float
            Smoothing scale in degrees
        **kwargs
            Additional keyword arguments passed to FootprintPlotter.plot_lens_footprint
        """
        if not FOOTPRINT_AVAILABLE:
            self.logger.error(
                "Footprint plotting requires skymapper and healpy packages. "
                "Install them with: pip install skymapper healpy"
            )
            return
        
        # Create footprint plotter
        footprint_plotter = FootprintPlotter(
            self.computation_config,
            self.lens_config,
            self.source_config,
            self.output_config,
            self.plot_config,
            self.logger
        )
        
        # Plot footprint
        footprint_plotter.plot_lens_footprint(
            surveys_to_overlay=surveys_to_overlay,
            nside=nside,
            smoothing=smoothing,
            **kwargs
        )


def create_plotter_from_configs(
    computation_config: ComputationConfig,
    lens_config: LensGalaxyConfig,
    source_config: SourceSurveyConfig,
    output_config: OutputConfig,
    plot_config: PlotConfig,
    analysis_config: AnalysisConfig,
    logger: Optional[logging.Logger] = None
) -> DataVectorPlotter:
    """
    Convenience function to create a DataVectorPlotter from config objects.
    
    Parameters
    ----------
    computation_config : ComputationConfig
        Computation configuration
    lens_config : LensGalaxyConfig
        Lens galaxy configuration  
    source_config : SourceSurveyConfig
        Source survey configuration
    output_config : OutputConfig
        Output configuration
    plot_config : PlotConfig
        Plot configuration
    analysis_config : AnalysisConfig
        Analysis configuration for scale cuts and bin selections
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    DataVectorPlotter
        Configured plotter instance
    """
    return DataVectorPlotter(
        computation_config, lens_config, source_config, output_config, 
        plot_config, analysis_config, logger
    ) 


# MultiGalaxyPlotter has been merged into DataVectorPlotter
# Use DataVectorPlotter with a list of LensGalaxyConfig for multi-galaxy plots


def create_multi_galaxy_plotter_from_configs(
    computation_config: ComputationConfig,
    lens_configs: List[LensGalaxyConfig],
    source_config: SourceSurveyConfig,
    output_config: OutputConfig,
    plot_config: PlotConfig,
    analysis_config: AnalysisConfig,
    logger: Optional[logging.Logger] = None
) -> DataVectorPlotter:
    """
    Convenience function to create a DataVectorPlotter for multiple galaxy types.
    
    This function is provided for backward compatibility. It creates a DataVectorPlotter
    with multiple lens configurations, enabling multi-galaxy plotting functionality.
    
    Parameters
    ----------
    computation_config : ComputationConfig
        Computation configuration
    lens_configs : List[LensGalaxyConfig]
        List of lens galaxy configurations  
    source_config : SourceSurveyConfig
        Source survey configuration
    output_config : OutputConfig
        Output configuration
    plot_config : PlotConfig
        Plot configuration
    analysis_config : AnalysisConfig
        Analysis configuration for scale cuts and bin selections
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    DataVectorPlotter
        Configured plotter instance with multi-galaxy support
    """
    return DataVectorPlotter(
        computation_config, lens_configs, source_config, output_config, 
        plot_config, analysis_config, logger
    )


# Backward compatibility alias
MultiGalaxyPlotter = DataVectorPlotter


# Note: Old MultiGalaxyPlotter class removed here
# Everything now goes through DataVectorPlotter





class FootprintPlotter:
    """
    Plotter class for survey footprints on sky maps.
    
    This class provides functionality to plot:
    - Lens galaxy density on HEALPix maps
    - Source survey footprints overlaid on sky maps
    - Observed vs target galaxy distributions
    - UNIONS and other survey boundaries
    """
    
    def __init__(
        self,
        computation_config: ComputationConfig,
        lens_config: LensGalaxyConfig,
        source_config: SourceSurveyConfig,
        output_config: OutputConfig,
        plot_config: PlotConfig,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the footprint plotter with configuration objects."""
        if not FOOTPRINT_AVAILABLE:
            raise ImportError(
                "Footprint plotting requires skymapper and healpy packages. "
                "Install them with: pip install skymapper healpy"
            )
        
        self.computation_config = computation_config
        self.lens_config = lens_config
        self.source_config = source_config
        self.output_config = output_config
        self.plot_config = plot_config
        
        self.logger = logger or setup_logger(self.__class__.__name__)
        
        # Setup data loader for accessing catalogues
        from ..config.path_manager import PathManager
        self.path_manager = PathManager(output_config, source_config)
        self.logger.info("Setting which_randoms to None, as no randoms are needed for footprint plotting")
        lens_config.which_randoms = None
        self.data_loader = DataLoader(
            lens_config, source_config, output_config, self.path_manager, logger
        )
        
        # Setup output directories
        self._setup_output_directories()
    
    def _setup_output_directories(self) -> None:
        """Create directories for saving plots and vertices."""
        version = self.lens_config.get_catalogue_version()
        plot_dir = self.plot_config.get_plot_output_dir(self.output_config.save_path) / version / "footprints"
        plot_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = plot_dir
        
        # Directory for storing pre-computed vertices
        self.vertices_dir = plot_dir / "vertices_cache"
        self.vertices_dir.mkdir(parents=True, exist_ok=True)
    
    def _place_gals_on_hpmap(self, catalogue: Table, nside: int) -> np.ndarray:
        """
        Place galaxies from catalogue onto HEALPix map.
        
        Parameters
        ----------
        catalogue : Table
            Catalogue with RA and DEC columns
        nside : int
            HEALPix nside parameter
            
        Returns
        -------
        np.ndarray
            HEALPix map with galaxy counts per pixel
        """
        # Get RA/DEC columns (handle different naming conventions)
        ra_col = 'RA' if 'RA' in catalogue.colnames else 'ra'
        dec_col = 'DEC' if 'DEC' in catalogue.colnames else 'dec'
        
        theta = np.radians(90 - catalogue[dec_col])  # HEALPix uses colatitude (90 - Dec)
        phi = np.radians(catalogue[ra_col])  # RA in radians
        pixels = hp.ang2pix(nside, theta, phi, lonlat=False)
        
        # Create a HEALPix map and count galaxies per pixel
        healpix_map = np.zeros(hp.nside2npix(nside), dtype=int)
        np.add.at(healpix_map, pixels, 1)
        
        return healpix_map
    
    def _get_boundary_mask(self, pix: np.ndarray, nside: int, niter: int = 1) -> np.ndarray:
        """
        Get boundary mask for HEALPix pixels.
        
        Parameters
        ----------
        pix : np.ndarray
            Array of HEALPix pixel indices
        nside : int
            HEALPix nside parameter
        niter : int
            Number of iterations for boundary detection
            
        Returns
        -------
        np.ndarray
            Boolean mask for boundary pixels
        """
        # Create full sky mask
        mask = np.zeros(hp.nside2npix(nside), dtype=bool)
        mask[pix] = True
        
        # Find boundaries by checking neighbors
        boundary_mask = np.zeros(len(pix), dtype=bool)
        for i, pixel in enumerate(pix):
            neighbors = hp.get_all_neighbours(nside, pixel)
            # Check if any neighbor is not in the mask
            if not np.all(mask[neighbors]):
                boundary_mask[i] = True
        
        return boundary_mask
    
    def _put_survey_on_grid(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        rap: np.ndarray,
        decp: np.ndarray,
        pixels: np.ndarray,
        vertices: np.ndarray,
        smoothing: u.Quantity = 0.0 * u.deg
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map survey galaxies onto HEALPix grid with optional smoothing.
        
        Parameters
        ----------
        ra : np.ndarray
            Galaxy right ascensions
        dec : np.ndarray
            Galaxy declinations
        rap : np.ndarray
            HEALPix grid RA centers
        decp : np.ndarray
            HEALPix grid DEC centers
        pixels : np.ndarray
            HEALPix pixel indices
        vertices : np.ndarray
            HEALPix pixel vertices
        smoothing : u.Quantity
            Smoothing scale in angular units
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (occupied_pixels, vertices_for_occupied, density_map)
        """
        from scipy.spatial import cKDTree
        
        # Build KDTree for galaxy positions
        galaxy_coords = np.column_stack([ra, dec])
        grid_coords = np.column_stack([rap, decp])
        
        tree = cKDTree(galaxy_coords)
        
        # Find which grid points have galaxies nearby
        smoothing_deg = smoothing.to(u.deg).value
        if smoothing_deg > 0:
            # Query with smoothing radius
            distances, _ = tree.query(grid_coords, k=1, distance_upper_bound=smoothing_deg)
            occupied = distances < smoothing_deg
        else:
            # No smoothing - just check for exact occupation
            occupied = np.zeros(len(pixels), dtype=bool)
            gal_pixels = hp.ang2pix(
                hp.npix2nside(len(pixels)),
                np.radians(90 - dec),
                np.radians(ra),
                lonlat=False
            )
            unique_pixels = np.unique(gal_pixels)
            occupied_idx = np.isin(pixels, unique_pixels)
            occupied[occupied_idx] = True
        
        occupied_pixels = pixels[occupied]
        occupied_vertices = vertices[occupied]
        
        return occupied_pixels, occupied_vertices, occupied
    
    def _load_unions_footprint(self, nside: int) -> Table:
        """
        Load UNIONS footprint from pre-computed pixel file.
        
        Parameters
        ----------
        nside : int
            Target HEALPix nside (will convert from stored nside=4096)
            
        Returns
        -------
        Table
            Table with 'ra' and 'dec' columns for UNIONS pixels
        """
        unions_file = Path("/global/cfs/cdirs/desicollab/users/sven/unions/ipix_mask_UNIONS_fromgalpos.npy")
        
        if not unions_file.exists():
            self.logger.warning(f"UNIONS pixel file not found: {unions_file}")
            return None
        
        try:
            unions_pixel = np.load(unions_file)
            # Convert from nside=4096 to target nside
            theta, phi = hp.pix2ang(nside=2**12, ipix=unions_pixel, lonlat=False)
            ra = np.degrees(phi)  # Right Ascension (in degrees)
            dec = 90 - np.degrees(theta)  # Declination (in degrees)
            
            table = Table()
            table['ra'] = ra
            table['dec'] = dec
            
            self.logger.info(f"Loaded UNIONS footprint with {len(table)} pixels")
            return table
            
        except Exception as e:
            self.logger.error(f"Failed to load UNIONS footprint: {e}")
            return None
    
    def _load_source_catalogue_for_footprint(self, survey: str) -> Optional[Table]:
        """
        Load source catalogue for footprint plotting.
        
        Parameters
        ----------
        survey : str
            Survey name
            
        Returns
        -------
        Optional[Table]
            Source catalogue with ra/dec columns
        """
        try:
            # Temporarily modify source_config to not cut to DESI footprint
            original_cut_setting = self.source_config.cut_catalogues_to_desi
            self.source_config.cut_catalogues_to_desi = False
            
            # Load using pipeline's data loader
            galaxy_type = self.lens_config.galaxy_type
            source_table, _, _ = self.data_loader.load_source_catalogue(survey, galaxy_type)
            
            # Restore original setting
            self.source_config.cut_catalogues_to_desi = original_cut_setting
            
            if source_table is None or len(source_table) == 0:
                self.logger.warning(f"No source catalogue loaded for {survey}")
                return None
            
            # Ensure we have ra/dec columns
            if 'ra' not in source_table.colnames and 'RA' in source_table.colnames:
                source_table['ra'] = source_table['RA']
            if 'dec' not in source_table.colnames and 'DEC' in source_table.colnames:
                source_table['dec'] = source_table['DEC']
            
            # Keep only ra/dec for memory efficiency
            source_table.keep_columns(['ra', 'dec'])
            
            self.logger.info(f"Loaded {len(source_table)} galaxies from {survey}")
            return source_table
            
        except Exception as e:
            self.logger.error(f"Failed to load source catalogue for {survey}: {e}")
            # Restore original setting in case of error
            self.source_config.cut_catalogues_to_desi = original_cut_setting
            return None
    
    def _load_or_compute_vertices(
        self,
        survey: str,
        nside: int,
        smoothing: u.Quantity,
        rap: np.ndarray,
        decp: np.ndarray,
        pixels: np.ndarray,
        vertices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load pre-computed vertices or compute and save them.
        
        Parameters
        ----------
        survey : str
            Survey name
        nside : int
            HEALPix nside parameter
        smoothing : u.Quantity
            Smoothing scale
        rap : np.ndarray
            HEALPix grid RA centers
        decp : np.ndarray
            HEALPix grid DEC centers
        pixels : np.ndarray
            HEALPix pixel indices
        vertices : np.ndarray
            HEALPix pixel vertices
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (occupied_pixels, occupied_vertices)
        """
        smoothing_str = f"{smoothing.value:.2f}".replace(".", "p")
        pix_file = self.vertices_dir / f"pix_{survey}_nside_{nside}_smooth_{smoothing_str}.npy"
        vert_file = self.vertices_dir / f"vert_{survey}_nside_{nside}_smooth_{smoothing_str}.npy"
        
        # Try to load pre-computed vertices
        if pix_file.exists() and vert_file.exists():
            try:
                pix = np.load(pix_file)
                vert = np.load(vert_file)
                self.logger.info(f"Loaded pre-computed vertices for {survey} from {pix_file}")
                return pix, vert
            except Exception as e:
                self.logger.warning(f"Failed to load vertices for {survey}: {e}")
        
        # Compute vertices
        self.logger.info(f"Computing vertices for {survey}")
        
        # Load catalogue based on survey type
        if survey.upper() == "UNIONS":
            catalogue = self._load_unions_footprint(nside)
        else:
            catalogue = self._load_source_catalogue_for_footprint(survey)
        
        if catalogue is None or len(catalogue) == 0:
            self.logger.error(f"Cannot compute vertices for {survey}: no catalogue data")
            return np.array([]), np.array([])
        
        # Map survey to grid
        ra_col = 'ra' if 'ra' in catalogue.colnames else 'RA'
        dec_col = 'dec' if 'dec' in catalogue.colnames else 'DEC'
        
        pix, vert, _ = self._put_survey_on_grid(
            catalogue[ra_col], catalogue[dec_col],
            rap, decp, pixels, vertices,
            smoothing=smoothing
        )
        
        # Save for future use
        try:
            np.save(pix_file, pix)
            np.save(vert_file, vert)
            self.logger.info(f"Saved vertices for {survey} to {pix_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save vertices for {survey}: {e}")
        
        return pix, vert
    
    def plot_lens_footprint(
        self,
        surveys_to_overlay: Optional[List[str]] = None,
        nside: int = 64,
        smoothing: float = 0.0,
        sep: int = 30,
        colors: Optional[Dict[str, str]] = None,
        alphas: Optional[Dict[str, float]] = None,
        label_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        plot_ratio: bool = True,
        save_plot: bool = True,
        filename_suffix: str = ""
    ) -> None:
        """
        Plot lens galaxy footprint with source survey overlays.
        
        Creates a Hammer projection map showing lens galaxy density and
        overlays footprints of source surveys.
        
        Parameters
        ----------
        surveys_to_overlay : List[str], optional
            List of surveys to overlay. If None, uses all configured source surveys plus UNIONS
        nside : int
            HEALPix nside parameter for resolution
        smoothing : float
            Smoothing scale in degrees for survey boundaries
        sep : int
            Number of graticules per degree for grid
        colors : Dict[str, str], optional
            Dictionary mapping survey names to colors
        alphas : Dict[str, float], optional
            Dictionary mapping survey names to alpha values
        label_positions : Dict[str, Tuple[float, float]], optional
            Dictionary mapping survey names to (lon, lat) label positions in degrees
        plot_ratio : bool
            If True, plot ratio of observed to target galaxies (requires target catalogue)
        save_plot : bool
            Whether to save the plot to file
        filename_suffix : str
            Optional suffix for the filename
        """
        self.logger.info("Plotting lens galaxy footprint")
        
        galaxy_type = self.lens_config.galaxy_type
        
        # Default surveys to overlay
        if surveys_to_overlay is None:
            surveys_to_overlay = list(self.source_config.surveys) + ["UNIONS"]
        
        # Default colors and alphas
        if colors is None:
            default_colors = {
                "KiDS": "blue",
                "DES": "red",
                "DECADE": "green",
                "DECADE_NGC": "green",
                "DECADE_SGC": "darkgreen",
                "HSCY1": "orange",
                "HSCY3": "darkorange",
                "UNIONS": "firebrick"
            }
            colors = {s: default_colors.get(s, "gray") for s in surveys_to_overlay}
        
        if alphas is None:
            alphas = {s: 0.7 for s in surveys_to_overlay}
            alphas["UNIONS"] = 0.9  # UNIONS more opaque
        
        # Default label positions (lon, lat in degrees)
        if label_positions is None:
            label_positions = {
                "KiDS": (0, 60),
                "DES": (90, -40),
                "DECADE": (180, 40),
                "DECADE_NGC": (180, 50),
                "DECADE_SGC": (180, -50),
                "HSCY1": (-150, 0),
                "HSCY3": (-150, 10),
                "UNIONS": (-20, 60)
            }
        
        # Create figure
        fig = plt.figure(figsize=(18, 9))
        proj = skm.Hammer()
        footprint = skm.Map(proj, facecolor='white', ax=fig.gca())
        footprint.grid(sep=sep)
        
        # Get HEALPix grid
        pixels, rap, decp, vertices = skm.healpix.getGrid(nside, return_vertices=True)
        
        # Load lens catalogue for the first source survey
        try:
            source_survey = self.source_config.surveys[0]
            print("This is where lens catalogue is loaded")
            lens_table, _ = self.data_loader.load_lens_catalogues(source_survey)
            
            # Apply quality cuts (ZWARN < 100)
            if 'ZWARN' in lens_table.colnames:
                lens_table = lens_table[lens_table['ZWARN'] < 100]
            
            # Keep only RA/DEC
            ra_col = 'RA' if 'RA' in lens_table.colnames else 'ra'
            dec_col = 'DEC' if 'DEC' in lens_table.colnames else 'dec'
            lens_table.keep_columns([ra_col, dec_col])
            
            self.logger.info(f"Loaded {len(lens_table)} lens galaxies")
            
            # Create HEALPix map of observed galaxies
            hpmap_observed = self._place_gals_on_hpmap(lens_table, nside)
            
            # Plot density or ratio
            if plot_ratio:
                # Try to load target catalogue for ratio
                try:
                    # This would need to be adapted based on where target catalogues are stored
                    # For now, just plot density
                    self.logger.warning("Ratio plotting not fully implemented, plotting density instead")
                    mappable = footprint.healpix(hpmap_observed, vmin=0, vmax=np.percentile(hpmap_observed[hpmap_observed > 0], 95), cmap='viridis')
                    cb_label = f"{galaxy_type[:3]} $n_g$ [per pixel]"
                except Exception as e:
                    self.logger.warning(f"Could not plot ratio: {e}")
                    mappable = footprint.healpix(hpmap_observed, vmin=0, vmax=np.percentile(hpmap_observed[hpmap_observed > 0], 95), cmap='viridis')
                    cb_label = f"{galaxy_type[:3]} $n_g$ [per pixel]"
            else:
                mappable = footprint.healpix(hpmap_observed, vmin=0, vmax=np.percentile(hpmap_observed[hpmap_observed > 0], 95), cmap='viridis')
                cb_label = f"{galaxy_type[:3]} $n_g$ [per pixel]"
            
            cb = footprint.colorbar(mappable, cb_label=cb_label)
            
        except Exception as e:
            self.logger.error(f"Failed to load lens catalogue: {e}")
            return
        
        # Overlay survey footprints
        smoothing_qty = smoothing * u.deg
        
        for survey in surveys_to_overlay:
            self.logger.info(f"Overlaying {survey} footprint")
            
            try:
                # Load or compute vertices
                pix, vert = self._load_or_compute_vertices(
                    survey, nside, smoothing_qty,
                    rap, decp, pixels, vertices
                )
                
                if len(pix) == 0:
                    self.logger.warning(f"No vertices computed for {survey}, skipping")
                    continue
                
                # Get boundary vertices
                boundary_mask = self._get_boundary_mask(pix, nside, niter=1)
                myvert = vert[boundary_mask]
                
                self.logger.info(f"Plotting {survey} with {len(myvert)} boundary vertices")
                
                # Plot survey boundary
                footprint.vertex(
                    myvert,
                    facecolors=colors.get(survey, "gray"),
                    alpha=alphas.get(survey, 0.7)
                )
                
                # Add survey label
                if survey in label_positions:
                    posx, posy = label_positions[survey]
                    txt = footprint.ax.text(
                        np.deg2rad(posx), np.deg2rad(posy),
                        survey,
                        size=20,
                        color=colors.get(survey, "gray"),
                        horizontalalignment='center',
                        verticalalignment='bottom'
                    )
                    txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
                
            except Exception as e:
                self.logger.error(f"Failed to overlay {survey}: {e}")
                continue
        
        # Save plot
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            surveys_str = "_".join([s[:3] for s in surveys_to_overlay])
            filename = f"footprint_{galaxy_type.lower()}_{surveys_str}{suffix}.png"
            filepath = self.plot_dir / filename
            
            footprint.savefig(
                filepath,
                dpi=self.plot_config.dpi,
                transparent=self.plot_config.transparent_background,
                bbox_inches="tight",
                pad_inches=0
            )
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()


def create_footprint_plotter_from_configs(
    computation_config: ComputationConfig,
    lens_config: LensGalaxyConfig,
    source_config: SourceSurveyConfig,
    output_config: OutputConfig,
    plot_config: PlotConfig,
    logger: Optional[logging.Logger] = None
) -> FootprintPlotter:
    """
    Convenience function to create a FootprintPlotter from config objects.
    
    Parameters
    ----------
    computation_config : ComputationConfig
        Computation configuration
    lens_config : LensGalaxyConfig
        Lens galaxy configuration
    source_config : SourceSurveyConfig
        Source survey configuration
    output_config : OutputConfig
        Output configuration
    plot_config : PlotConfig
        Plot configuration
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    FootprintPlotter
        Configured footprint plotter instance
    """
    return FootprintPlotter(
        computation_config, lens_config, source_config, 
        output_config, plot_config, logger
    ) 