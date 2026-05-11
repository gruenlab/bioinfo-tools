"""Preprocessing scripts for the Spatial Probe Design pipeline.

This module provides preprocessing utilities for:
- Selection pipeline:  preprocess_for_selection (importable function)
- Analysis scripts:    preprocess_for_analysis  (importable function)
- Evaluation pipeline: preprocess_for_evaluation (CLI script)
                       preprocess_reference_for_evaluation (CLI script)
"""

from __future__ import annotations

from .preprocess_for_selection import preprocess_for_selection, preprocess_for_analysis

__all__ = [
    'preprocess_for_selection',
    'preprocess_for_analysis',
]

__version__ = '2.0.0'
