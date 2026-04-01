"""
InfoSEDD Real Data Experiments
================================

This package contains code for running MI estimation experiments on real datasets
(SummEval, genomic sequences, promoter datasets).

Main entry point: main.py
"""

__version__ = "0.1.0"

# Import main modules for convenience
try:
    from . import dataloader
    from . import diffusion
    from . import utils
    from . import fdime_utils
except ImportError:
    # Allow package to be imported even if dependencies aren't fully installed
    pass
