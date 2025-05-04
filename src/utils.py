#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for WorldScreener.

This module provides utility functions used across the application.
"""

import os
import logging
import yaml
import json
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: logging.INFO)
        log_file: Log file path (optional)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    logger.debug("Logging configured")


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        dict: Configuration dictionary
    """
    if not config_path:
        # Use default config path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'default_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load market configuration
        markets_config_path = os.path.join(os.path.dirname(config_path), 'markets_config.yaml')
        if os.path.exists(markets_config_path):
            with open(markets_config_path, 'r') as f:
                markets_config = yaml.safe_load(f)
                
            # Merge configurations
            config.update(markets_config)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        
        # Return default configuration
        return {
            'screening': {
                'default_criteria': {
                    'max_pe_ratio': 15.0,
                    'min_dividend_yield': 3.0,
                    'max_pb_ratio': 1.5,
                    'min_roe': 10.0,
                    'max_debt_to_equity': 1.0,
                    'min_market_cap': 100000000,
                    'min_value_score': 50.0
                },
                'value_score_weights': {
                    'pe_ratio': -0.25,
                    'pb_ratio': -0.20,
                    'dividend_yield': 0.25,
                    'roe': 0.15,
                    'debt_to_equity': -0.15
                }
            },
            'market_indices': {
                'spain': '^IBEX',
                'europe': '^STOXX',
                'eurozone': '^STOXX50E',
                'us': '^GSPC',
                'global': 'ACWI'
            }
        }


def save_config(config, config_path):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to configuration file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")


def format_currency(value, currency='USD'):
    """
    Format a value as currency.
    
    Args:
        value: Numeric value
        currency: Currency code (default: 'USD')
        
    Returns:
        str: Formatted currency string
    """
    if pd.isna(value) or value == 'N/A':
        return 'N/A'
    
    try:
        value = float(value)
        
        # Format large numbers
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B {currency}"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M {currency}"
        elif value >= 1_000:
            return f"{value / 1_000:.2f}K {currency}"
        else:
            return f"{value:.2f} {currency}"
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value):
    """
    Format a value as percentage.
    
    Args:
        value: Numeric value
        
    Returns:
        str: Formatted percentage string
    """
    if pd.isna(value) or value == 'N/A':
        return 'N/A'
    
    try:
        value = float(value)
        return f"{value:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def get_country_name(country_code):
    """
    Get country name from country code.
    
    Args:
        country_code: ISO country code
        
    Returns:
        str: Country name
    """
    country_map = {
        'ES': 'Spain',
        'DE': 'Germany',
        'FR': 'France',
        'IT': 'Italy',
        'GB': 'United Kingdom',
        'NL': 'Netherlands',
        'CH': 'Switzerland',
        'SE': 'Sweden',
        'BE': 'Belgium',
        'DK': 'Denmark',
        'NO': 'Norway',
        'FI': 'Finland',
        'AT': 'Austria',
        'PT': 'Portugal',
        'IE': 'Ireland',
        'GR': 'Greece',
        'PL': 'Poland',
        'US': 'United States'
    }
    
    return country_map.get(country_code, country_code)


def export_to_excel(data_dict, output_file):
    """
    Export multiple DataFrames to Excel file.
    
    Args:
        data_dict: Dictionary of sheet_name: DataFrame pairs
        output_file: Path to output Excel file
    """
    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust columns width
                worksheet = writer.sheets[sheet_name]
                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, max_len)
        
        logger.info(f"Data exported to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")


def create_output_dir(dir_name='output'):
    """
    Create output directory if it doesn't exist.
    
    Args:
        dir_name: Directory name
        
    Returns:
        str: Path to output directory
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, dir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir