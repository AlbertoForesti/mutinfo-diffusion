"""
InfoSEDD Synthetic Experiments
================================

This package contains code for running MI estimation experiments on synthetic data
with various estimators (InfoSEDD, MINDE, F-DIME variants).

Main entry point: train.py
"""

__version__ = "0.1.0"

# Import main modules for convenience
try:
    from . import datamodule
    from . import mi_estimator
    from . import model
    from . import infosedd_utils
    from . import fdime_utils
    from . import minde_utils
except ImportError:
    # Allow package to be imported even if dependencies aren't fully installed
    pass
