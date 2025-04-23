"""
MM-Neural-Adjoint - A neural adjoint method implementation
"""

from .models.NA import Network
from .models.base_model import BaseModel

__version__ = "0.1.0"
__all__ = ["Network", "BaseModel"]