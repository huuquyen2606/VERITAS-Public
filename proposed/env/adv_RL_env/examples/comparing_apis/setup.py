import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Module name must match the PYBIND11_MODULE declaration in the C++ source.
module_name = "fast_sw"

ext_modules = [
    Pybind11Extension(
        module_name,
        ["smith-wanderman.cpp"],
        # Compiler performance optimization.
        cxx_std=11, 
    ),
]

setup(
    name=module_name,
    version="1.0.0",
    author="Security Research Pipeline",
    description="C++ implementation of Smith-Waterman with Affine Gap for API similarity",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)