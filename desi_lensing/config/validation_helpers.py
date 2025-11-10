"""Helper functions for validation error messages."""

from typing import List, Optional
import difflib


def suggest_correction(invalid_value: str, valid_options: List[str], max_suggestions: int = 3) -> str:
    """
    Suggest corrections for an invalid value based on similarity to valid options.
    
    Parameters
    ----------
    invalid_value : str
        The invalid value provided by the user
    valid_options : List[str]
        List of valid options
    max_suggestions : int
        Maximum number of suggestions to return
        
    Returns
    -------
    str
        Formatted string with suggestions, or empty string if no close matches
        
    Examples
    --------
    >>> suggest_correction('BGS', ['BGS_BRIGHT', 'LRG', 'ELG'])
    "Did you mean 'BGS_BRIGHT'?"
    """
    if not valid_options or not invalid_value:
        return ""
    
    # Find close matches
    close_matches = difflib.get_close_matches(
        invalid_value, 
        valid_options, 
        n=max_suggestions, 
        cutoff=0.6
    )
    
    if not close_matches:
        return ""
    
    if len(close_matches) == 1:
        return f"Did you mean '{close_matches[0]}'?"
    else:
        suggestions = "', '".join(close_matches)
        return f"Did you mean one of: '{suggestions}'?"


def format_validation_error(
    error_message: str,
    invalid_value: Optional[str] = None,
    valid_options: Optional[List[str]] = None,
    context: Optional[str] = None
) -> str:
    """
    Format a validation error message with helpful context and suggestions.
    
    Parameters
    ----------
    error_message : str
        Base error message
    invalid_value : str, optional
        The invalid value that triggered the error
    valid_options : List[str], optional
        List of valid options
    context : str, optional
        Additional context (e.g., which config class, which parameter)
        
    Returns
    -------
    str
        Formatted error message with suggestions
        
    Examples
    --------
    >>> format_validation_error(
    ...     "Invalid galaxy type",
    ...     invalid_value="BGS",
    ...     valid_options=["BGS_BRIGHT", "LRG", "ELG"],
    ...     context="LensGalaxyConfig.galaxy_type"
    ... )
    "[LensGalaxyConfig.galaxy_type] Invalid galaxy type 'BGS'. Did you mean 'BGS_BRIGHT'? Valid options: BGS_BRIGHT, LRG, ELG"
    """
    parts = []
    
    # Add context if provided
    if context:
        parts.append(f"[{context}]")
    
    # Add base message with invalid value if provided
    if invalid_value:
        parts.append(f"{error_message} '{invalid_value}'.")
    else:
        parts.append(f"{error_message}.")
    
    # Add suggestion if possible
    if invalid_value and valid_options:
        suggestion = suggest_correction(invalid_value, valid_options)
        if suggestion:
            parts.append(suggestion)
    
    # Add valid options
    if valid_options:
        options_str = ", ".join(str(opt) for opt in valid_options)
        parts.append(f"Valid options: {options_str}")
    
    return " ".join(parts)


def validate_choice(
    value: str,
    valid_options: List[str],
    parameter_name: str,
    context: Optional[str] = None
) -> Optional[str]:
    """
    Validate that a value is in a list of valid options.
    
    Parameters
    ----------
    value : str
        Value to validate
    valid_options : List[str]
        List of valid options
    parameter_name : str
        Name of the parameter being validated
    context : str, optional
        Context string (e.g., class name)
        
    Returns
    -------
    Optional[str]
        Error message if invalid, None if valid
        
    Examples
    --------
    >>> validate_choice('BGS', ['BGS_BRIGHT', 'LRG'], 'galaxy_type')
    "[galaxy_type] Invalid value 'BGS'. Did you mean 'BGS_BRIGHT'? Valid options: BGS_BRIGHT, LRG"
    """
    if value not in valid_options:
        return format_validation_error(
            f"Invalid {parameter_name}",
            invalid_value=value,
            valid_options=valid_options,
            context=context or parameter_name
        )
    return None


def validate_range(
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    parameter_name: str = "value",
    context: Optional[str] = None
) -> Optional[str]:
    """
    Validate that a numeric value is within a specified range.
    
    Parameters
    ----------
    value : float
        Value to validate
    min_value : float, optional
        Minimum allowed value (inclusive)
    max_value : float, optional
        Maximum allowed value (inclusive)
    parameter_name : str
        Name of the parameter being validated
    context : str, optional
        Context string
        
    Returns
    -------
    Optional[str]
        Error message if invalid, None if valid
    """
    errors = []
    
    if min_value is not None and value < min_value:
        msg = f"{parameter_name} must be >= {min_value}, got {value}"
        if context:
            msg = f"[{context}] {msg}"
        errors.append(msg)
    
    if max_value is not None and value > max_value:
        msg = f"{parameter_name} must be <= {max_value}, got {value}"
        if context:
            msg = f"[{context}] {msg}"
        errors.append(msg)
    
    return errors[0] if errors else None


def validate_list_not_empty(
    value: List,
    parameter_name: str,
    context: Optional[str] = None
) -> Optional[str]:
    """
    Validate that a list is not empty.
    
    Parameters
    ----------
    value : List
        List to validate
    parameter_name : str
        Name of the parameter being validated
    context : str, optional
        Context string
        
    Returns
    -------
    Optional[str]
        Error message if empty, None if valid
    """
    if not value:
        msg = f"{parameter_name} cannot be empty"
        if context:
            msg = f"[{context}] {msg}"
        return msg
    return None


def validate_increasing_sequence(
    values: List[float],
    parameter_name: str,
    context: Optional[str] = None
) -> Optional[str]:
    """
    Validate that values are in strictly increasing order.
    
    Parameters
    ----------
    values : List[float]
        Values to validate
    parameter_name : str
        Name of the parameter being validated
    context : str, optional
        Context string
        
    Returns
    -------
    Optional[str]
        Error message if not increasing, None if valid
    """
    for i in range(len(values) - 1):
        if values[i] >= values[i + 1]:
            msg = (f"{parameter_name} must be in strictly increasing order. "
                   f"Found {values[i]} >= {values[i + 1]} at positions {i} and {i + 1}")
            if context:
                msg = f"[{context}] {msg}"
            return msg
    return None

