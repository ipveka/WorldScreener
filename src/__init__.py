#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WorldScreener - A comprehensive stock screening tool for value investors.

This package provides tools to screen, analyze, and visualize value stocks
across European and US markets.
"""

# Suppress urllib3 warnings - must be done before other imports
import warnings
import urllib3
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings('ignore', message=".*OpenSSL.*")
urllib3.disable_warnings()

import logging
from .data_provider import DataProvider
from .screener import Screener
from .analyzer import Analyzer
from .visualizer import Visualizer
from .report_generator import ReportGenerator
from .utils import setup_logging, load_config, save_config

__version__ = '1.0.0'
__author__ = 'WorldScreener Team'

# Set up package logger
logger = logging.getLogger(__name__)

# Initialize logging with default settings
setup_logging()


def create_screener(config_path=None):
    """
    Create a WorldScreener instance with all components.
    
    Args:
        config_path (str, optional): Path to configuration file
        
    Returns:
        tuple: (DataProvider, Screener, Analyzer, Visualizer, ReportGenerator)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize components
    data_provider = DataProvider(config=config)
    screener = Screener(data_provider, config)
    analyzer = Analyzer(data_provider)
    visualizer = Visualizer()
    report_generator = ReportGenerator()
    
    # Add components to Screener for convenience
    screener.analyzer = analyzer
    screener.visualizer = visualizer
    screener.report_generator = report_generator
    screener.data_provider = data_provider
    
    logger.info("WorldScreener instance created")
    return screener