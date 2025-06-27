"""
Utility functions for plotting DESI lensing results.

This module contains common plotting utilities migrated from the legacy
plotting_utilities.py module, adapted for the refactored architecture.
"""

import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

from ..config import ComputationConfig, LensGalaxyConfig, SourceSurveyConfig


def initialize_gridspec_figure(
    figsize: Tuple[float, float],
    nrows: int,
    ncols: int,
    hspace: float = 0.0,
    wspace: float = 0.0,
    add_colorbar: bool = True
) -> Tuple[plt.Figure, np.ndarray, gridspec.GridSpec]:
    """
    Initialize a figure with gridspec layout.
    
    Parameters
    ----------
    figsize : Tuple[float, float]
        Figure size in inches (width, height)
    nrows : int
        Number of rows
    ncols : int  
        Number of columns
    hspace : float
        Height spacing between subplots
    wspace : float
        Width spacing between subplots
    add_colorbar : bool
        Whether to add space for colorbar
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray, gridspec.GridSpec]
        Figure, axes array, and gridspec object
    """
    if add_colorbar:
        # Add extra column for colorbar
        gs = gridspec.GridSpec(
            nrows, ncols + 1,
            width_ratios=[20] * ncols + [1],
            hspace=hspace, wspace=wspace
        )
        fig = plt.figure(figsize=figsize)
        
        # Create main axes
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j])
        
        # Ensure axes is 2D
        if nrows == 1:
            axes = axes.reshape(1, -1)
        if ncols == 1:
            axes = axes.reshape(-1, 1)
            
    else:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize,
            gridspec_kw={'hspace': hspace, 'wspace': wspace}
        )
        gs = None
        
        # Ensure axes is 2D
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
    
    return fig, axes, gs


def add_colorbar_legend(
    fig: plt.Figure,
    axes: np.ndarray,
    gs: gridspec.GridSpec,
    colors: List[str],
    labels: List[str],
    title: str = "",
    location: str = "right"
) -> None:
    """
    Add a colorbar-style legend to the figure.
    
    Parameters
    ----------
    fig : plt.Figure
        The figure object
    axes : np.ndarray
        Array of axes
    gs : gridspec.GridSpec
        GridSpec object
    colors : List[str]
        List of colors for legend entries
    labels : List[str]
        List of labels for legend entries
    title : str
        Title for the legend
    location : str
        Location of the legend ('right' or 'bottom')
    """
    if gs is None:
        # Add legend to existing axes
        legend_elements = []
        for color, label in zip(colors, labels):
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color, markersize=6, label=label)
            )
        
        fig.legend(
            handles=legend_elements, loc='upper right',
            bbox_to_anchor=(0.95, 0.95), title=title
        )
        return
    
    if location == "right":
        # Add colorbar-style legend on the right
        cbar_ax = fig.add_subplot(gs[:, -1])
        
        # Create custom colorbar
        from matplotlib.colors import ListedColormap
        from matplotlib.cm import ScalarMappable
        
        cmap = ListedColormap(colors)
        sm = ScalarMappable(cmap=cmap)
        sm.set_array(np.arange(len(colors)))
        
        cbar = plt.colorbar(sm, cax=cbar_ax, ticks=np.arange(len(colors)))
        cbar.ax.set_yticklabels(labels)
        if title:
            cbar.ax.set_ylabel(title, rotation=270, labelpad=15)


def plot_scale_cuts(
    axes: np.ndarray,
    rp_min: float,
    rp_max: float,
    rp_pivot: Optional[float],
    lens_config: LensGalaxyConfig,
    computation_config: ComputationConfig,
    statistic: str = "deltasigma",
    shared_axes: bool = True
) -> None:
    """
    Add vertical lines and shaded regions to indicate scale cuts.
    
    Parameters
    ----------
    axes : np.ndarray
        Array of matplotlib axes
    rp_min : float
        Minimum scale for analysis
    rp_max : float
        Maximum scale for analysis
    rp_pivot : Optional[float]
        Pivot scale (if any)
    lens_config : LensGalaxyConfig
        Lens galaxy configuration
    computation_config : ComputationConfig
        Computation configuration
    statistic : str
        The statistic being plotted
    shared_axes : bool
        Whether axes share x-limits
    """
    if shared_axes and axes.size > 0:
        axmin, axmax = axes.flat[0].get_xlim()
    
    # Get scale limits for each lens bin
    n_lens_bins = lens_config.get_n_lens_bins()
    
    # Iterate through all axes
    for ax in axes.flat:
        if ax is None:
            continue
            
        if not shared_axes:
            axmin, axmax = ax.get_xlim()
        
        # Add pivot line if specified
        if rp_pivot is not None:
            ax.axvline(rp_pivot, color='k', linestyle=':', alpha=0.7)
        
        # Add shaded regions for excluded scales
        ax.axvspan(axmin, rp_min, color='gray', alpha=0.3)
        ax.axvspan(rp_max, axmax, color='gray', alpha=0.3)
        
        if not shared_axes:
            ax.set_xlim(axmin, axmax)
    
    if shared_axes and axes.size > 0:
        axes.flat[0].set_xlim(axmin, axmax)


def get_theory_prediction(
    rp: np.ndarray,
    galaxy_type: str,
    lens_bin: int,
    computation_config: ComputationConfig,
    statistic: str = "deltasigma"
) -> np.ndarray:
    """
    Get theoretical prediction for comparison.
    
    Parameters
    ----------
    rp : np.ndarray
        Radial distances
    galaxy_type : str
        Type of lens galaxies
    lens_bin : int
        Lens redshift bin
    computation_config : ComputationConfig
        Computation configuration
    statistic : str
        The statistic to compute
        
    Returns
    -------
    np.ndarray
        Theoretical prediction values
    """
    # This is a placeholder - in practice this would load
    # from theory calculations or simulations
    
    # Simple power-law for demonstration
    if statistic == "deltasigma":
        # Typical excess surface density shape
        amplitude = 10.0 * (lens_bin + 1)  # Higher amplitude for higher redshift
        return amplitude * (rp / 1.0) ** (-0.8)
    else:  # gammat
        # Typical tangential shear shape
        amplitude = 1e-3 * (lens_bin + 1)
        return amplitude * (rp / 1.0) ** (-0.8)


def compute_signal_to_noise(
    data: np.ndarray,
    covariance: np.ndarray,
    theory: Optional[np.ndarray] = None
) -> float:
    """
    Compute signal-to-noise ratio.
    
    Parameters
    ----------
    data : np.ndarray
        Data vector
    covariance : np.ndarray
        Covariance matrix
    theory : Optional[np.ndarray]
        Theory prediction (if None, uses data)
        
    Returns
    -------
    float
        Signal-to-noise ratio
    """
    if theory is None:
        theory = data
    
    try:
        inv_cov = np.linalg.inv(covariance)
        sn_squared = np.einsum('i,ij,j', data, inv_cov, theory)
        return np.sqrt(sn_squared)
    except np.linalg.LinAlgError:
        return np.nan


def save_p_values_table(
    pvalues: Dict[str, float],
    lens_config: LensGalaxyConfig,
    source_config: SourceSurveyConfig,
    output_path: Path,
    statistic: str = "deltasigma",
    caption: str = "",
    precision: int = 3
) -> str:
    """
    Generate LaTeX table with p-values.
    
    Parameters
    ----------
    pvalues : Dict[str, float]
        Dictionary of p-values
    lens_config : LensGalaxyConfig
        Lens configuration
    source_config : SourceSurveyConfig
        Source configuration  
    output_path : Path
        Output file path
    statistic : str
        The statistic name
    caption : str
        Table caption
    precision : int
        Decimal precision for p-values
        
    Returns
    -------
    str
        LaTeX table string
    """
    n_lens_bins = lens_config.get_n_lens_bins()
    surveys = source_config.surveys
    
    # Create table header
    header = "\\begin{table}[htbp]\n"
    header += "\\centering\n"
    header += f"\\caption{{{caption}}}\n"
    header += "\\begin{tabular}{l" + "c" * n_lens_bins + "}\n"
    header += "\\hline\n"
    
    # Column headers
    header += "Survey & " + " & ".join([f"Bin {i+1}" for i in range(n_lens_bins)]) + " \\\\\n"
    header += "\\hline\n"
    
    # Data rows
    rows = []
    for survey in surveys:
        row = [survey]
        for lens_bin in range(n_lens_bins):
            key = f"{lens_config.galaxy_type}_{survey}_{lens_bin}"
            if key in pvalues:
                value = f"{pvalues[key]:.{precision}f}"
            else:
                value = "--"
            row.append(value)
        rows.append(" & ".join(row) + " \\\\\n")
    
    # Table footer
    footer = "\\hline\n"
    footer += "\\end{tabular}\n"
    footer += "\\end{table}\n"
    
    # Combine table
    table_str = header + "".join(rows) + footer
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(table_str)
    
    return table_str


def setup_matplotlib_style(style: str = "paper") -> None:
    """
    Setup matplotlib style for consistent plots.
    
    Parameters
    ----------
    style : str
        Style preset ('paper', 'presentation', 'notebook')
    """
    if style == "paper":
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'errorbar.capsize': 1.5,
            'lines.linewidth': 0.8,
            'lines.markersize': 2.0,
        })
    elif style == "presentation":
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'errorbar.capsize': 2.0,
            'lines.linewidth': 1.2,
            'lines.markersize': 3.0,
        })
    elif style == "notebook":
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'errorbar.capsize': 1.5,
            'lines.linewidth': 1.0,
            'lines.markersize': 2.5,
        })


def validate_data_consistency(
    table: Table,
    expected_columns: List[str],
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate that a data table has expected structure.
    
    Parameters
    ----------
    table : Table
        Data table to validate
    expected_columns : List[str]
        List of expected column names
    logger : Optional[logging.Logger]
        Logger for messages
        
    Returns
    -------
    bool
        True if validation passes
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check if table exists
    if table is None:
        logger.error("Table is None")
        return False
    
    # Check columns
    missing_columns = []
    for col in expected_columns:
        if col not in table.colnames:
            missing_columns.append(col)
    
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        return False
    
    # Check for NaN values
    for col in expected_columns:
        if np.any(np.isnan(table[col])):
            logger.warning(f"NaN values found in column: {col}")
    
    # Check for infinite values
    for col in expected_columns:
        if np.any(np.isinf(table[col])):
            logger.warning(f"Infinite values found in column: {col}")
    
    return True 