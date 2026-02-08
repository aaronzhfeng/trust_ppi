"""TrustPPI: Trustworthy Protein-Protein Interaction Modeling."""

__version__ = "0.1.0"

# Core modules
from . import data
from . import utils
from . import sabdab_data

# Stage 1: Trust Layer
from . import trust

# Stage 2: Geometric Oracle
from . import models

__all__ = [
    'data',
    'utils',
    'sabdab_data',
    'trust',
    'models'
]
