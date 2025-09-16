"""
P3R-Aerospace: Parameter-Efficient Fine-Tuning for Aerospace Vulnerability Detection
"""

__version__ = "1.0.0"
__author__ = "Your Name"

import os
import importlib

# Dynamically import all Python files in this directory (except __init__.py)
_package_dir = os.path.dirname(__file__)
__all__ = []

for filename in os.listdir(_package_dir):
    if filename.endswith(".py") and filename not in {"__init__.py"}:
        module_name = filename[:-3]  # remove ".py"
        module = importlib.import_module(f"{__name__}.{module_name}")

        # Expose all classes with "Model" in their name
        for attr in dir(module):
            if attr.endswith("Model"):
                globals()[attr] = getattr(module, attr)
                __all__.append(attr)
