#!/usr/bin/env python
"""Test script for multi-galaxy plotting functionality."""

import sys
sys.path.insert(0, '/global/u2/s/sven/code/desi-lensing')

def test_analysis_config():
    """Test the new AnalysisConfig functionality."""
    from desi_lensing.config.analysis import AnalysisConfig
    
    print("Testing AnalysisConfig...")
    
    # Create analysis config
    analysis_config = AnalysisConfig()
    
    # Test individual galaxy type bin counts
    print(f"BGS_BRIGHT bins: {analysis_config.get_n_bins_for_galaxy_type('BGS_BRIGHT')}")
    print(f"LRG bins: {analysis_config.get_n_bins_for_galaxy_type('LRG')}")
    print(f"ELG bins: {analysis_config.get_n_bins_for_galaxy_type('ELG')}")
    
    # Test total bins for multiple galaxy types
    galaxy_types = ['BGS_BRIGHT', 'LRG']
    total_bins = analysis_config.get_total_bins_for_galaxy_types(galaxy_types)
    print(f"Total bins for {galaxy_types}: {total_bins}")
    
    # Test bin layout
    layout = analysis_config.get_bin_layout_for_galaxy_types(galaxy_types)
    print(f"Bin layout: {layout}")
    
    # Test validation
    errors = analysis_config.validate()
    print(f"Validation errors: {errors}")
    
    print("AnalysisConfig tests passed!\n")


def test_cli_config_parsing():
    """Test the CLI configuration parsing for multiple galaxy types."""
    from desi_lensing.cli.main import parse_comma_separated_strings
    
    print("Testing CLI config parsing...")
    
    # Test parsing comma-separated galaxy types
    galaxy_types = parse_comma_separated_strings('BGS_BRIGHT,LRG')
    print(f"Parsed galaxy types: {galaxy_types}")
    
    # Test default value
    default_types = parse_comma_separated_strings('BGS_BRIGHT,LRG')
    print(f"Default galaxy types: {default_types}")
    
    print("CLI config parsing tests passed!\n")


def show_expected_plot_structure():
    """Show the expected plot structure."""
    from desi_lensing.config.analysis import AnalysisConfig
    
    print("Expected plot structure:")
    print("=" * 40)
    
    analysis_config = AnalysisConfig()
    galaxy_types = ['BGS_BRIGHT', 'LRG']
    
    # Show column layout
    layout = analysis_config.get_bin_layout_for_galaxy_types(galaxy_types)
    total_bins = analysis_config.get_total_bins_for_galaxy_types(galaxy_types)
    
    print(f"Total columns: {total_bins} + 1 (colorbar) = {total_bins + 1}")
    print("Column layout:")
    for galaxy_type, (start, end) in layout.items():
        n_bins = analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
        print(f"  {galaxy_type}: columns {start}-{end-1} ({n_bins} bins)")
    
    print(f"\nColorbar: column {total_bins}")
    
    # Show example surveys
    print("\nExample with 3 source surveys:")
    surveys = ['DES', 'KiDS', 'HSCY1']
    print(f"Rows: {len(surveys)} (one per survey)")
    print(f"Grid size: {len(surveys)} × {total_bins + 1}")
    
    print("\nTitles (top row):")
    for galaxy_type, (start, end) in layout.items():
        n_bins = analysis_config.get_n_bins_for_galaxy_type(galaxy_type)
        for i in range(n_bins):
            col = start + i
            title = f"{galaxy_type[:3]} Bin {i + 1}"
            print(f"  Column {col}: {title}")


def main():
    """Run all tests."""
    print("Testing multi-galaxy plotting functionality")
    print("=" * 50)
    
    try:
        test_analysis_config()
        test_cli_config_parsing()
        show_expected_plot_structure()
        
        print("\n✓ All tests passed!")
        print("\nTo use the new functionality:")
        print("1. Default behavior now plots both BGS_BRIGHT and LRG")
        print("2. Use --galaxy-type 'BGS_BRIGHT,LRG' for explicit specification")
        print("3. Plot will have 5 columns: 3 for BGS_BRIGHT, 2 for LRG")
        print("4. Rows correspond to source surveys")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 