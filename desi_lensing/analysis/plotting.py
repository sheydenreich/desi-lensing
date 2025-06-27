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
    - Data vectors (tomographic and non-tomographic)
    - B-mode diagnostics
    - Comparison plots between different configurations
    - Random tests
    """
    
    def __init__(
        self,
        computation_config: ComputationConfig,
        lens_config: LensGalaxyConfig,
        source_config: SourceSurveyConfig,
        output_config: OutputConfig,
        plot_config: PlotConfig,
        analysis_config: AnalysisConfig,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the plotter with configuration objects."""
        self.computation_config = computation_config
        self.lens_config = lens_config
        self.source_config = source_config
        self.output_config = output_config
        self.plot_config = plot_config
        self.analysis_config = analysis_config
        
        self.logger = logger or setup_logger(self.__class__.__name__)
        
        # Setup data loader for accessing results
        from ..config.path_manager import PathManager
        self.path_manager = PathManager(output_config)
        self.data_loader = DataLoader(
            lens_config, source_config, output_config, self.path_manager, logger
        )
        
        # Common plotting parameters
        self.color_list = ['blue', 'red', 'green', 'orange', 'purple']
        self.survey_names = source_config.surveys
        
        # Setup output directories
        self._setup_output_directories()
    
    def _setup_output_directories(self) -> None:
        """Create directories for saving plots."""
        version = self.lens_config.get_catalogue_version()
        plot_dir = self.plot_config.get_plot_output_dir(self.output_config.save_path) / version
        plot_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = plot_dir
        
    def _load_lensing_results(
        self, 
        lens_bin: int, 
        source_survey: str,
        source_bin: Optional[int] = None,
        statistic: str = "deltasigma"
    ) -> Optional[Table]:
        """Load lensing measurement results for a given configuration."""
        version = self.lens_config.get_catalogue_version()
        galaxy_type = self.lens_config.galaxy_type
        z_bins = self.lens_config.z_bins
        
        return self.output_config.load_lensing_results(
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
        """Load covariance matrix for a given configuration."""
        version = self.lens_config.get_catalogue_version()
        galaxy_type = self.lens_config.galaxy_type
        z_bins = self.lens_config.z_bins
        
        return self.output_config.load_covariance_matrix(
            version, galaxy_type, source_survey, lens_bin, z_bins,
            statistic, source_bin
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
        # Get lens redshift from configuration
        z_bins = self.lens_config.z_bins
        zlens = (z_bins[lens_bin] + z_bins[lens_bin + 1]) / 2.0
        
        # Use simple cosmology for scale conversion (can be improved with actual cosmology)
        # Approximation: comoving_transverse_distance ≈ c * z / H0 for low z
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
        tomographic: bool = False
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
            Pivot scale in Mpc/h
        statistic : str
            The statistic being plotted ('deltasigma' or 'gammat')
        shared_axes : bool
            Whether axes share x-limits
        tomographic : bool
            Whether this is a tomographic plot
        """
        if shared_axes:
            if axes.ndim == 1:
                axmin, axmax = axes[0].get_xlim()
            else:
                axmin, axmax = axes[0, 0].get_xlim()

        galaxy_type = self.lens_config.galaxy_type
        n_lens_bins = self.lens_config.get_n_lens_bins()
        
        # Handle both 1D and 2D axes arrays
        if axes.ndim == 1:
            axes_to_iterate = [(0, i) for i in range(len(axes))]
        else:
            n_rows, n_cols = axes.shape
            if tomographic:
                axes_to_iterate = [(row, col) for row in range(n_rows) for col in range(n_cols-1)]
            else:
                axes_to_iterate = [(0, col) for col in range(n_cols-1)]
        
        for row_idx, col_idx in axes_to_iterate:
            # Determine lens bin from column index
            if galaxy_type == "BGS_BRIGHT":
                lens_bin = col_idx
            elif galaxy_type == "LRG":
                # Assuming LRG bins come after BGS_BRIGHT bins
                n_bgs_bins = 3 # Typical number of BGS_BRIGHT bins
                if col_idx >= n_bgs_bins:
                    lens_bin = col_idx - n_bgs_bins
                else:
                    continue
                if lens_bin < 0 or lens_bin >= n_lens_bins:
                    continue
            else:
                lens_bin = col_idx
            
            # Get the appropriate axis
            if axes.ndim == 1:
                ax = axes[col_idx]
            else:
                ax = axes[row_idx, col_idx]
            
            # Convert scales if needed
            if statistic == "deltasigma":
                rpmin, rpmax = self._get_rp_from_deg(min_deg, max_deg, galaxy_type, lens_bin)
                # Add pivot line
                ax.axvline(rp_pivot, color='k', linestyle=':', alpha=0.7)
            else:
                rpmin, rpmax = min_deg, max_deg
            
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
        
        n_lens_bins = self.lens_config.get_n_lens_bins()
        n_surveys = len(self.source_config.surveys)
        
        # Setup figure with colorbar space
        fig_width = 7.24
        fig_height = fig_width / n_lens_bins * n_surveys
        fig, axes, gs = self._initialize_gridspec_figure(
            (fig_width, fig_height), n_surveys, n_lens_bins,
            add_cbar=True, hspace=0, wspace=0
        )
        
        # Plot data for each combination
        for survey_idx, source_survey in enumerate(self.source_config.surveys):
            n_tomo_bins = self.source_config.get_n_tomographic_bins(source_survey)
            
            for lens_bin in range(n_lens_bins):
                y_label = ""
                ax = axes[survey_idx, lens_bin]
                
                # Set title for top row
                if survey_idx == 0:
                    galaxy_type_short = self.lens_config.galaxy_type[:3]
                    ax.set_title(f"{galaxy_type_short} Bin {lens_bin + 1}")
                
                # Get allowed source bins for this lens bin
                allowed_source_bins = self.analysis_config.get_allowed_source_bins(
                    self.lens_config.galaxy_type, source_survey, lens_bin
                )
                
                # Plot each allowed tomographic bin
                for source_bin in allowed_source_bins:
                    if source_bin >= n_tomo_bins:
                        continue  # Skip if source bin doesn't exist for this survey
                    
                    results = self._load_lensing_results(
                        lens_bin, source_survey, source_bin, statistic
                    )
                    
                    if results is None:
                        continue
                    
                    # Extract data
                    if statistic == "deltasigma":
                        rp = results['rp']
                        signal = results['ds']
                        error = results['ds_err']
                        y_label = r"$\Delta\Sigma(r_p)$"
                        if not log_scale:
                            signal = rp * signal
                            error = rp * error
                            y_label = r"$r_p \times \Delta\Sigma(r_p)$"
                    else:  # gammat
                        rp = results['theta']
                        signal = results['et']
                        error = results['et_err']
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
                
                if lens_bin == 0:  # Left column
                    ax.set_ylabel(f"{source_survey}\n{y_label}")
        
        # Add survey colorbar legend for tomographic bins
        if n_tomo_bins > 1:
            self._add_survey_colorbar_legend(
                fig, axes, gs, 
                self.color_list[:n_tomo_bins],
                [f"Bin {i+1}" for i in range(n_tomo_bins)]
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
                survey_axes = axes[survey_idx:survey_idx+1, :]
                self._plot_scale_cuts(
                    survey_axes, scale_cuts["min_deg"], scale_cuts["max_deg"], 
                    scale_cuts["rp_pivot"], statistic, shared_axes=True, tomographic=True
                )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            scale_suffix = "_log" if log_scale else ""
            filename = f"{statistic}_datavector_tomo{scale_suffix}{suffix}.png"
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
        
        Parameters
        ----------
        statistic : str
            The statistic to plot ('deltasigma' or 'gammat')
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
            
        Returns
        -------
        Dict[str, float]
            Dictionary of p-values for each configuration
        """
        self.logger.info(f"Plotting B-mode diagnostics for {statistic}")
        
        # Store the original computation config
        original_bmodes = self.computation_config.bmodes
        
        # Temporarily enable B-modes for this analysis
        self.computation_config.bmodes = True
        
        n_lens_bins = self.lens_config.get_n_lens_bins()
        n_surveys = len(self.source_config.surveys)
        
        # Setup figure with colorbar space
        fig_width = 7.24
        fig_height = fig_width / n_lens_bins * n_surveys
        fig, axes, gs = self._initialize_gridspec_figure(
            (fig_width, fig_height), n_surveys, n_lens_bins,
            add_cbar=True, hspace=0, wspace=0
        )
        
        pvalues = {}
        
        # Plot B-mode data
        for survey_idx, source_survey in enumerate(self.source_config.surveys):
            n_tomo_bins = self.source_config.get_n_tomographic_bins(source_survey)
            
            for lens_bin in range(n_lens_bins):
                ax = axes[survey_idx, lens_bin]
                
                # Set title for top row
                if survey_idx == 0:
                    galaxy_type_short = self.lens_config.galaxy_type[:3]
                    ax.set_title(f"{galaxy_type_short} Bin {lens_bin + 1}")
                
                # Combine all tomographic bins for this lens-source combination
                combined_data = []
                combined_cov_blocks = []
                
                for source_bin in range(n_tomo_bins):
                    # Load B-mode results using centralized methods
                    version = self.lens_config.get_catalogue_version()
                    galaxy_type = self.lens_config.galaxy_type
                    z_bins = self.lens_config.z_bins
                    
                    results = self.output_config.load_lensing_results(
                        version, galaxy_type, source_survey, lens_bin, z_bins,
                        statistic, source_bin, pure_noise=True
                    )
                    cov = self.output_config.load_covariance_matrix(
                        version, galaxy_type, source_survey, lens_bin, z_bins,
                        statistic, source_bin, pure_noise=True
                    )
                    
                    if results is None or cov is None:
                        continue
                    
                    # Extract B-mode data  
                    if statistic == "deltasigma":
                        rp = results['rp']
                        signal = results['ds']
                        error = results['ds_err']
                    else:
                        rp = results['theta'] 
                        signal = results['et']
                        error = results['et_err']
                    
                    combined_data.append(signal)
                    combined_cov_blocks.append(cov)
                    
                    # Plot individual tomographic bins
                    offset = 0.02 * source_bin
                    rp_plot = rp * np.exp(offset)
                    ax.errorbar(
                        rp_plot, rp * signal, rp * error,
                        fmt='o', color=self.color_list[source_bin % len(self.color_list)],
                        markersize=2, capsize=1
                    )
                
                # Compute combined p-value
                if combined_data:
                    combined_data_vec = np.concatenate(combined_data)
                    combined_cov = self._combine_covariance_matrices(combined_cov_blocks)
                    
                    pvalue = self._compute_pvalue(combined_data_vec, combined_cov)
                    chisq = self._compute_chisq(combined_data_vec, combined_cov)
                    
                    key = f"{self.lens_config.galaxy_type}_{source_survey}_{lens_bin}"
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
                
                if lens_bin == 0:  # Left column
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
        if n_tomo_bins > 1:
            self._add_survey_colorbar_legend(
                fig, axes, gs,
                self.color_list[:n_tomo_bins],
                [f"Bin {i+1}" for i in range(n_tomo_bins)]
            )
        
        # Plot scale cuts
        if plot_scale_cuts:
            self._plot_scale_cuts(
                axes[:, :-1], min_deg, max_deg, rp_pivot, statistic,
                shared_axes=True, tomographic=True
            )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            filename = f"{statistic}_bmodes_tomo{suffix}.png"
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
            
            # Try to load existing randoms results
            try:
                p_arr, V_arr, keys = randoms_analyzer.load_randoms_results(
                    "splits", filename_suffix
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
                    self._plot_scale_cuts(
                        axes, min_deg, max_deg, rp_pivot, statistic,
                        shared_axes=True, tomographic=True
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
            self._plot_scale_cuts(
                axes, min_deg, max_deg, rp_pivot, statistic,
                shared_axes=True, tomographic=True
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


class MultiGalaxyPlotter:
    """
    Plotter class for handling multiple galaxy types in a single plot.
    
    This class provides functionality to plot:
    - Data vectors with multiple galaxy types (tomographic)
    - Combined plots with configurable column layout (5 columns: 3 BGS + 2 LRG)
    - B-mode diagnostics across galaxy types
    """
    
    def __init__(
        self,
        computation_config: ComputationConfig,
        lens_configs: List[LensGalaxyConfig],
        source_config: SourceSurveyConfig,
        output_config: OutputConfig,
        plot_config: PlotConfig,
        analysis_config: AnalysisConfig,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the multi-galaxy plotter with configuration objects."""
        self.computation_config = computation_config
        self.lens_configs = lens_configs
        self.source_config = source_config
        self.output_config = output_config
        self.plot_config = plot_config
        self.analysis_config = analysis_config
        
        self.logger = logger or setup_logger(self.__class__.__name__)
        
        # Setup data loader for accessing results
        from ..config.path_manager import PathManager
        self.path_manager = PathManager(output_config)
        
        # Common plotting parameters
        self.color_list = ['blue', 'red', 'green', 'orange', 'purple']
        self.survey_names = source_config.surveys
        
        # Get galaxy types and bin layout
        self.galaxy_types = [config.galaxy_type for config in lens_configs]
        self.bin_layout = analysis_config.get_bin_layout_for_galaxy_types(self.galaxy_types)
        self.total_bins = analysis_config.get_total_bins_for_galaxy_types(self.galaxy_types)
        
        # Setup output directories
        self._setup_output_directories()
    
    def _setup_output_directories(self) -> None:
        """Create directories for saving plots."""
        # Use first lens config for versioning
        version = self.lens_configs[0].get_catalogue_version()
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
        # Find the lens config for this galaxy type
        lens_config = None
        for config in self.lens_configs:
            if config.galaxy_type == galaxy_type:
                lens_config = config
                break
        
        if lens_config is None:
            return None
        
        version = lens_config.get_catalogue_version()
        z_bins = lens_config.z_bins
        
        return self.output_config.load_lensing_results(
            version, galaxy_type, source_survey, lens_bin, z_bins,
            statistic, source_bin
        )
    
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
        Plot tomographic data vectors for multiple galaxy types.
        
        Creates a plot with configurable columns per galaxy type:
        - BGS_BRIGHT: 3 columns (bins)
        - LRG: 2 columns (bins)
        - Total: 5 columns + 1 for colorbar
        - Rows: one per source survey
        
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
            Override scale cuts with custom values
        """
        self.logger.info(f"Plotting multi-galaxy tomographic data vectors for {statistic}")
        
        # Determine if scale cuts should be applied
        if plot_scale_cuts is None:
            plot_scale_cuts = self.analysis_config.apply_scale_cuts
        
        n_surveys = len(self.source_config.surveys)
        
        # Setup figure with colorbar space - total_bins columns + 1 for colorbar
        fig_width = 7.24
        fig_height = fig_width / self.total_bins * n_surveys
        fig, axes, gs = self._initialize_gridspec_figure(
            (fig_width, fig_height), n_surveys, self.total_bins,
            add_cbar=True, hspace=0, wspace=0
        )
        
        # Plot data for each galaxy type and bin combination
        for survey_idx, source_survey in enumerate(self.source_config.surveys):
            n_tomo_bins = self.source_config.get_n_tomographic_bins(source_survey)
            
            # Plot each galaxy type in its designated columns
            for galaxy_type in self.galaxy_types:
                start_col, end_col = self.bin_layout[galaxy_type]
                n_lens_bins = self.analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
                
                for lens_bin in range(n_lens_bins):
                    y_label = ""
                    col_idx = start_col + lens_bin
                    ax = axes[survey_idx, col_idx]
                    
                    # Set title for top row
                    if survey_idx == 0:
                        galaxy_type_short = galaxy_type[:3]
                        ax.set_title(f"{galaxy_type_short} Bin {lens_bin + 1}")
                    
                    # Get allowed source bins for this lens bin
                    allowed_source_bins = self.analysis_config.get_allowed_source_bins(
                        galaxy_type, source_survey, lens_bin
                    )
                    
                    # Plot each allowed tomographic bin
                    for source_bin in allowed_source_bins:
                        if source_bin >= n_tomo_bins:
                            continue  # Skip if source bin doesn't exist for this survey
                        
                        results = self._load_lensing_results_for_galaxy_type(
                            galaxy_type, lens_bin, source_survey, source_bin, statistic
                        )
                        
                        if results is None:
                            continue
                        
                        # Extract data
                        if statistic == "deltasigma":
                            rp = results['rp']
                            signal = results['ds']
                            error = results['ds_err']
                            y_label = r"$\Delta\Sigma(r_p)$"
                            if not log_scale:
                                signal = rp * signal
                                error = rp * error
                                y_label = r"$r_p \times \Delta\Sigma(r_p)$"
                        else:  # gammat
                            rp = results['theta']
                            signal = results['et']
                            error = results['et_err']
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
                    scale_cuts["rp_pivot"], statistic, shared_axes=True, tomographic=True
                )
        
        plt.tight_layout()
        
        if save_plot:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            scale_suffix = "_log" if log_scale else ""
            galaxy_suffix = "_".join(self.galaxy_types)
            filename = f"{statistic}_datavector_tomo_{galaxy_suffix}{scale_suffix}{suffix}.png"
            filepath = self.plot_dir / filename
            
            plt.savefig(
                filepath, dpi=self.plot_config.dpi, 
                transparent=self.plot_config.transparent_background,
                bbox_inches="tight"
            )
            self.logger.info(f"Saved plot: {filepath}")
        
        plt.show()
    
    def _initialize_gridspec_figure(
        self, 
        figsize: Tuple[float, float], 
        nrows: int, 
        ncols: int, 
        add_cbar: bool = True, 
        **kwargs
    ) -> Tuple[plt.Figure, np.ndarray, gridspec.GridSpec]:
        """Initialize figure with GridSpec layout including colorbar space."""
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
        """Add colorbar legend for surveys."""
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
    
    def _plot_scale_cuts(
        self, 
        axes: np.ndarray, 
        min_deg: float, 
        max_deg: float, 
        rp_pivot: float, 
        statistic: str = "deltasigma",
        shared_axes: bool = True,
        tomographic: bool = False
    ) -> None:
        """Plot scale cuts as gray shaded regions and pivot line."""
        # Get current axis limits if shared
        if shared_axes:
            if axes.ndim == 1:
                axmin, axmax = axes[0].get_xlim()
            else:
                axmin, axmax = axes[0, 0].get_xlim()
        
        # Handle both 1D and 2D axes arrays
        if axes.ndim == 1:
            axes_to_iterate = axes
        else:
            axes_to_iterate = axes.flatten()
        
        for ax in axes_to_iterate:
            if ax is None:  # Skip colorbar column
                continue
                
            # Convert scales if needed
            if statistic == "deltasigma":
                rpmin, rpmax = min_deg, max_deg  # Simplified - would need proper conversion
                # Add pivot line
                ax.axvline(rp_pivot, color='k', linestyle=':', alpha=0.7)
            else:
                rpmin, rpmax = min_deg, max_deg
            
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


def create_multi_galaxy_plotter_from_configs(
    computation_config: ComputationConfig,
    lens_configs: List[LensGalaxyConfig],
    source_config: SourceSurveyConfig,
    output_config: OutputConfig,
    plot_config: PlotConfig,
    analysis_config: AnalysisConfig,
    logger: Optional[logging.Logger] = None
) -> MultiGalaxyPlotter:
    """
    Convenience function to create a MultiGalaxyPlotter from config objects.
    
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
    MultiGalaxyPlotter
        Configured multi-galaxy plotter instance
    """
    return MultiGalaxyPlotter(
        computation_config, lens_configs, source_config, output_config, 
        plot_config, analysis_config, logger
    ) 