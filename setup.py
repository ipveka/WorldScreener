#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="worldscreener",
    version="1.0.0",
    author="WorldScreener Team",
    author_email="contact@worldscreener.com",
    description="A comprehensive stock screening tool for value investors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/worldscreener/worldscreener",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "yfinance>=0.1.70",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
        "tabulate>=0.8.9",
        "pyyaml>=5.4.0",
        "jinja2>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "worldscreener=worldscreener:main",
        ],
    },
    include_package_data=True,
    package_data={
        "worldscreener": [
            "config/*.yaml",
            "templates/*.html",
        ],
    },
)