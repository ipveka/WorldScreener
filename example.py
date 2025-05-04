#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WorldScreener Demo Script

This script demonstrates the capabilities of the WorldScreener package
with various examples of screening, analyzing, and visualizing value stocks
across international markets.
"""

# Suppress urllib3 warnings - this must be done before any other imports
import warnings
import urllib3
# Directly suppress the specific NotOpenSSLWarning by monkey patching
original_warn = warnings.warn
def custom_warn(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], urllib3.exceptions.NotOpenSSLWarning):
        return
    if len(args) > 0 and "OpenSSL" in str(args[0]):
        return
    return original_warn(*args, **kwargs)
warnings.warn = custom_warn

# Libraries
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tabulate import tabulate

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import WorldScreener modules
from src import create_screener as create_ws, setup_logging, load_config
from src.utils import create_output_dir, format_currency, format_percentage

def demo_multi_region_screening(screener, output_dir):
    """
    Demonstrate multi-region stock screening with custom criteria.
    
    Args:
        screener (Screener): Screener instance
        output_dir (str): Output directory for reports
    
    Returns:
        pandas.DataFrame: Combined DataFrame of all screened stocks
    """
    print("\n================================================================================")
    print("DEMONSTRATION 1: MULTI-REGION SCREENING WITH CUSTOM CRITERIA")
    print("================================================================================\n")
    
    # Custom screening criteria
    criteria = {
        'max_pe_ratio': 15.0,
        'min_dividend_yield': 1.0,
        'max_pb_ratio': 2.0,
        'min_roe': 8.0,
        'max_debt_to_equity': 1.5
    }
    
    # Print criteria
    print("Custom screening criteria:")
    for key, value in criteria.items():
        print(f"- {key}: {value}")
    print()
    
    # Screen stocks in different regions
    regions = ['us', 'europe', 'japan']
    stocks_dict = {}
    
    for region in regions:
        print(f"Screening {region} stocks with custom criteria...")
        try:
            stocks = screener.screen_stocks(
                region=region, 
                criteria=criteria
            )
            stocks_dict[region] = stocks
            
            if not stocks.empty:
                print(f"Found {len(stocks)} {region} value stocks:")
                # Display only essential columns
                display_df = stocks[['ticker', 'name', 'sector', 'country', 'pe_ratio', 
                                     'dividend_yield', 'pb_ratio', 'roe', 'debt_to_equity', 
                                     'market_cap', 'value_score']].copy()
                # Convert market cap to billions for display
                display_df.loc[:, 'market_cap'] = display_df['market_cap'] / 1_000_000_000
                # Reset index and drop the original index column
                display_df = display_df.reset_index(drop=True)
                print(tabulate(display_df, headers='keys', tablefmt='grid'))
                print("Market cap is displayed in billions of US dollars.")
                print()
            else:
                print(f"No {region} stocks found meeting the criteria.\n")
        except Exception as e:
            print(f"Error screening {region} stocks: {e}\n")
            stocks_dict[region] = pd.DataFrame()
    
    # Generate comparative report
    print(f"Generating comparative report to {os.path.join(output_dir, 'comparative_report.html')}...")
    try:
        screener.report_generator.generate_comparative_report(
            stocks_dict,
            output_file=os.path.join(output_dir, 'comparative_report.html')
        )
        print(f"Comparative report saved to {os.path.join(output_dir, 'comparative_report.html')}")
    except Exception as e:
        print(f"Error generating comparative report: {e}")
    
    # Combine all stocks for further analysis
    all_stocks = pd.concat([df for df in stocks_dict.values() if not df.empty])
    return all_stocks

def demo_global_portfolio_creation(screener, stocks_df, output_dir):
    """
    Demonstrate global value portfolio creation and analysis.
    
    Args:
        screener: WorldScreener instance
        stocks_df (pandas.DataFrame): Screened stocks
        output_dir (str): Output directory for reports
    """
    print("\n================================================================================")
    print("DEMONSTRATION 2: GLOBAL VALUE PORTFOLIO CREATION AND ANALYSIS")
    print("================================================================================\n")
    
    print("Using screened stocks for portfolio creation")
    
    if stocks_df.empty:
        print("No stocks found for portfolio.")
        return
    
    # Create portfolio - use .copy() to avoid SettingWithCopyWarning
    portfolio = stocks_df.sort_values('value_score', ascending=False).head(10).copy()
    
    # Display portfolio
    print(f"\nPortfolio created with {len(portfolio)} stocks:")
    # Display only essential columns
    display_df = portfolio[['ticker', 'name', 'sector', 'country', 'pe_ratio', 
                           'dividend_yield', 'pb_ratio', 'roe', 'debt_to_equity', 
                           'market_cap', 'value_score']].copy()
    # Convert market cap to billions for display
    display_df.loc[:, 'market_cap'] = display_df['market_cap'] / 1_000_000_000
    # Reset index and drop the original index column
    display_df = display_df.reset_index(drop=True)
    print(tabulate(display_df, headers='keys', tablefmt='grid'))
    print("Market cap is displayed in billions of US dollars.")
    
    # Generate portfolio report
    print(f"\nGenerating portfolio report to {os.path.join(output_dir, 'portfolio_report.html')}...")
    
    try:
        screener.report_generator.generate_report(
            portfolio,
            output_format='html',
            output_file=os.path.join(output_dir, 'portfolio_report.html')
        )
    except Exception as e:
        print(f"Error generating portfolio report: {e}")
    
    # Analyze top stocks
    print(f"\nPerforming detailed analysis on top stocks: {', '.join(portfolio['ticker'].tolist())}")
    
    for ticker in portfolio['ticker'].tolist():
        try:
            analysis = screener.analyzer.analyze_stock(ticker)
            
            # Generate stock report
            if analysis:
                screener.report_generator.generate_stock_report(
                    analysis,
                    output_file=os.path.join(output_dir, f"{ticker}_report.html")
                )
        except Exception as e:
            print(f"Error in demo_global_portfolio_creation: {e}")

def create_screener():
    """
    Create and configure a WorldScreener instance.
    
    Returns:
        Screener: Configured screener instance
    """
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "default_config.yaml")
    
    # Create WorldScreener instance with real data
    print("Using REAL data from Yahoo Finance")
    
    # Create a screener instance using the module's function
    from src import create_screener as create_ws
    
    # The module's create_screener function returns a Screener instance directly
    screener = create_ws(config_path)
    
    return screener

def create_output_dir(output_dir):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir (str): Path to output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing output directory: {output_dir}")

def main():
    """
    Main function to run all demonstrations.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="WorldScreener Demonstration Script")
    parser.add_argument("--demo-num", type=int, help="Run a specific demo (1-2)")
    parser.add_argument("--output-dir", type=str, help="Output directory for reports and charts")
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    create_output_dir(output_dir)
    print(f"Demo output will be saved to: {output_dir}")
    
    # Initialize WorldScreener components
    print("Initializing WorldScreener components...")
    screener = create_screener()
    
    # Run specific demo or all demos
    if args.demo_num:
        if args.demo_num == 1:
            demo_multi_region_screening(screener, output_dir)
        elif args.demo_num == 2:
            # When running demo 2 directly, pass None for screened_stocks to ensure it works independently
            demo_global_portfolio_creation(screener, pd.DataFrame(), output_dir)
        else:
            print(f"Invalid demo number: {args.demo_num}. Please choose 1-2.")
    else:
        print("Running all demonstrations...")
        # Run multi-region screening first and use those results for portfolio creation
        screened_stocks = demo_multi_region_screening(screener, output_dir)
        demo_global_portfolio_creation(screener, screened_stocks, output_dir)
    
    print("\n" + "="*80)
    print("DEMONSTRATIONS COMPLETED")
    print("="*80 + "\n")
    
    print(f"All demo outputs have been saved to: {output_dir}")

if __name__ == "__main__":
    main()