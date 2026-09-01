"""
Canonical GAPGAN target detector.

This module re-exports the paper-truth MalConv definition from
`binary_malconv.arch` so the model trained in `binary_malconv` is
exactly the model queried by GAPGAN.
"""

from binary_malconv.arch import MalConv

__all__ = ["MalConv"]
