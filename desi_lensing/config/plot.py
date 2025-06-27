"""Plot configuration for lensing pipeline."""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from .base import BaseConfig


@dataclass
class PlotConfig(BaseConfig):
    """Configuration for plotting settings."""
    
    # Plot output settings
    plot_output_dir: Optional[str] = None  # If None, will use OutputConfig.save_path + "/plots"
    save_plots: bool = True
    filename_suffix: str = ""
    
    # Plot appearance settings
    style: str = "paper"  # paper, presentation, notebook
    transparent_background: bool = False
    log_scale: bool = False
    
    # Plot dimensions and DPI
    dpi: int = 300
    figure_width: float = 7.24  # inches, typical for 2-column papers
    figure_height: Optional[float] = None  # If None, auto-calculated based on aspect ratio
    
    # Color and styling
    color_palette: str = "default"  # default, colorblind, custom
    marker_size: float = 2.0
    line_width: float = 1.0
    error_cap_size: float = 1.0
    
    # Plot-specific settings
    show_legends: bool = True
    show_grid: bool = False
    show_minor_ticks: bool = True
    
    # Text and labels
    font_size: float = 10.0
    label_font_size: Optional[float] = None  # If None, uses font_size
    title_font_size: Optional[float] = None  # If None, uses font_size * 1.2
    
    def validate(self) -> List[str]:
        """Validate plot configuration."""
        errors = []
        
        # Validate style
        valid_styles = ["paper", "presentation", "notebook"]
        if self.style not in valid_styles:
            errors.append(f"style must be one of {valid_styles}, got {self.style}")
        
        # Validate color palette
        valid_palettes = ["default", "colorblind", "custom"]
        if self.color_palette not in valid_palettes:
            errors.append(f"color_palette must be one of {valid_palettes}, got {self.color_palette}")
        
        # Validate numeric values
        if self.dpi <= 0:
            errors.append("dpi must be positive")
        
        if self.figure_width <= 0:
            errors.append("figure_width must be positive")
        
        if self.figure_height is not None and self.figure_height <= 0:
            errors.append("figure_height must be positive if specified")
        
        if self.marker_size <= 0:
            errors.append("marker_size must be positive")
        
        if self.line_width <= 0:
            errors.append("line_width must be positive")
        
        if self.font_size <= 0:
            errors.append("font_size must be positive")
        
        if self.label_font_size is not None and self.label_font_size <= 0:
            errors.append("label_font_size must be positive if specified")
        
        if self.title_font_size is not None and self.title_font_size <= 0:
            errors.append("title_font_size must be positive if specified")
        
        return errors
    
    def get_plot_output_dir(self, output_config_save_path: str) -> Path:
        """Get the plot output directory."""
        if self.plot_output_dir is not None:
            return Path(self.plot_output_dir)
        else:
            return Path(output_config_save_path) / "plots"
    
    def get_label_font_size(self) -> float:
        """Get label font size, using font_size if not specified."""
        return self.label_font_size if self.label_font_size is not None else self.font_size
    
    def get_title_font_size(self) -> float:
        """Get title font size, using font_size * 1.2 if not specified."""
        return self.title_font_size if self.title_font_size is not None else self.font_size * 1.2
    
    def get_figure_height(self, aspect_ratio: float = 0.618) -> float:
        """Get figure height, using golden ratio if not specified."""
        return self.figure_height if self.figure_height is not None else self.figure_width * aspect_ratio
    
    def get_matplotlib_style_dict(self) -> dict:
        """
        Get matplotlib style dictionary for consistent styling.
        
        Returns
        -------
        dict
            Dictionary of matplotlib style parameters
        """
        style_dict = {
            'font.size': self.font_size,
            'axes.labelsize': self.get_label_font_size(),
            'axes.titlesize': self.get_title_font_size(),
            'xtick.labelsize': self.font_size,
            'ytick.labelsize': self.font_size,
            'legend.fontsize': self.font_size,
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            'axes.grid': self.show_grid,
            'xtick.minor.visible': self.show_minor_ticks,
            'ytick.minor.visible': self.show_minor_ticks,
        }
        
        # Style-specific adjustments
        if self.style == "paper":
            style_dict.update({
                'font.family': 'serif',
                'text.usetex': False,  # Can be enabled if LaTeX is available
                'axes.linewidth': 0.8,
                'xtick.major.width': 0.8,
                'ytick.major.width': 0.8,
            })
        elif self.style == "presentation":
            style_dict.update({
                'font.size': self.font_size * 1.2,
                'axes.labelsize': self.get_label_font_size() * 1.2,
                'axes.titlesize': self.get_title_font_size() * 1.2,
                'axes.linewidth': 1.2,
                'lines.linewidth': self.line_width * 1.5,
            })
        elif self.style == "notebook":
            style_dict.update({
                'figure.dpi': 100,  # Lower DPI for notebooks
                'savefig.dpi': self.dpi,
            })
        
        return style_dict 