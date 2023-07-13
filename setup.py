#!/usr/bin/env python
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="amemu",
    version="0.0.1",
    description="amemu",
    url="https://github.com/Harry45/amemu",
    author="Arrykrishna Mootoovaloo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    packages=["amemu"],
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "matplotlib",
        "notebook",
        "scipy",
        "fast-pt",
        "swig",
        "pyccl",
        "classy",
    ],
    python_requires=">=3.9",
)
