"""Base configuration class for all configurations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
import json
from pathlib import Path


@dataclass
class BaseConfig(ABC):
    """Base configuration class with common functionality."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, BaseConfig):
                result[key] = value.to_dict()
            elif isinstance(value, list) and value and isinstance(value[0], BaseConfig):
                result[key] = [item.to_dict() for item in value]
            else:
                result[key] = value
        return result
    
    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Convert configuration to YAML string or save to file."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            path.write_text(yaml_str)
        return yaml_str
    
    def to_json(self, path: Optional[Path] = None, indent: int = 2) -> str:
        """Convert configuration to JSON string or save to file."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if path:
            path.write_text(json_str)
        return json_str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "BaseConfig":
        """Load configuration from YAML file."""
        data = yaml.safe_load(path.read_text())
        return cls.from_dict(data)
    
    @classmethod 
    def from_json(cls, path: Path) -> "BaseConfig":
        """Load configuration from JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate configuration and return list of error messages."""
        pass
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    def show_config(self, indent: int = 0) -> str:
        """
        Return a formatted string representation of the configuration.
        
        Parameters
        ----------
        indent : int
            Number of spaces to indent output
            
        Returns
        -------
        str
            Formatted configuration string
        """
        lines = []
        prefix = " " * indent
        class_name = self.__class__.__name__
        lines.append(f"{prefix}{class_name}:")
        
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, BaseConfig):
                # Recursively show nested config
                lines.append(f"{prefix}  {key}:")
                lines.append(value.show_config(indent + 4))
            elif isinstance(value, list) and value and isinstance(value[0], BaseConfig):
                # List of configs
                lines.append(f"{prefix}  {key}:")
                for i, item in enumerate(value):
                    lines.append(f"{prefix}    [{i}]:")
                    lines.append(item.show_config(indent + 6))
            else:
                # Simple value
                lines.append(f"{prefix}  {key}: {value}")
        
        return "\n".join(lines)
    
    def print_config(self) -> None:
        """Print the configuration to stdout."""
        print(self.show_config())
    
    @classmethod
    def from_preset(cls, preset: str) -> "BaseConfig":
        """
        Create configuration from a named preset.
        
        This method should be overridden by subclasses to provide
        specific presets. Base implementation raises NotImplementedError.
        
        Parameters
        ----------
        preset : str
            Name of the preset configuration
            
        Returns
        -------
        BaseConfig
            Configuration instance with preset values
            
        Raises
        ------
        NotImplementedError
            If the subclass doesn't implement presets
        ValueError
            If the preset name is not recognized
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement presets. "
            f"Override from_preset() to add preset support."
        ) 