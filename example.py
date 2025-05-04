#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WorldScreener Demo Script

This script demonstrates the capabilities of the WorldScreener package
with various examples of screening, analyzing, and visualizing value stocks.
"""

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
from src import create_screener, setup_logging, load_config
from src.utils import create_output_dir, format_currency, format_percentage

def demo_spanish_stocks(data_provider, screener, analyzer, visualizer, report_generator, output_dir):
    """
    Demonstrate finding and analyzing Spanish value stocks.
    
    Args:
        data_provider: DataProvider instance
        screener: Screener instance
        analyzer: Analyzer instance
        visualizer: Visualizer instance
        report_generator: ReportGenerator instance
        output_dir: Output directory path
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 1: FINDING VALUE STOCKS IN SPAIN")
    print("="*80)
    
    # Screen Spanish stocks with default criteria
    print("\nScreening Spanish stocks with default criteria...")
    try:
        spanish_stocks = screener.screen_stocks(region='spain', limit=15)
        
        if spanish_stocks.empty:
            print("No Spanish stocks found meeting the criteria.")
            return
        
        # Display results
        print(f"\nFound {len(spanish_stocks)} Spanish value stocks:")
        display_df = spanish_stocks[['ticker', 'name', 'sector', 'pe_ratio', 'dividend_yield', 'value_score']]
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Generate report
        report_file = os.path.join(output_dir, 'spanish_stocks_report.html')
        print(f"\nGenerating HTML report to {report_file}...")
        report_generator.generate_report(spanish_stocks, output_format='html', output_file=report_file)
        
        # Create visualization
        print("\nCreating visualization...")
        chart_file = os.path.join(output_dir, 'spanish_stocks_by_sector.png')
        visualizer.create_sector_breakdown_chart(spanish_stocks, output_file=chart_file)
        print(f"Chart saved to {chart_file}")
        
        # Analyze top stock
        if not spanish_stocks.empty:
            top_stock = spanish_stocks.iloc[0]['ticker']
            print(f"\nAnalyzing top stock: {top_stock}...")
            analysis = analyzer.analyze_stock(top_stock)
            
            # Generate stock analysis report
            analysis_file = os.path.join(output_dir, f'{top_stock}_analysis.html')
            print(f"Generating stock analysis report to {analysis_file}...")
            report_generator.generate_stock_analysis_report(analysis, output_file=analysis_file)
    except Exception as e:
        print(f"Error in demo_spanish_stocks: {e}")

def demo_multi_region_screening(data_provider, screener, analyzer, visualizer, report_generator, output_dir):
    """
    Demonstrate screening across multiple regions with custom criteria.
    
    Args:
        data_provider: DataProvider instance
        screener: Screener instance
        analyzer: Analyzer instance
        visualizer: Visualizer instance
        report_generator: ReportGenerator instance
        output_dir: Output directory path
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 2: MULTI-REGION SCREENING WITH CUSTOM CRITERIA")
    print("="*80)
    
    # Define custom criteria
    custom_criteria = {
        'max_pe_ratio': 12.0,
        'min_dividend_yield': 4.0,
        'max_pb_ratio': 1.2,
        'min_roe': 12.0,
        'max_debt_to_equity': 0.8
    }
    
    print("\nCustom screening criteria:")
    for key, value in custom_criteria.items():
        print(f"- {key}: {value}")
    
    # Screen stocks across regions
    regions = ['europe', 'us']
    results = {}
    
    try:
        for region in regions:
            print(f"\nScreening {region} stocks with custom criteria...")
            stocks = screener.screen_stocks(region=region, criteria=custom_criteria, limit=10)
            results[region] = stocks
            
            if stocks.empty:
                print(f"No {region} stocks found meeting the criteria.")
            else:
                print(f"Found {len(stocks)} {region} value stocks:")
                display_df = stocks[['ticker', 'name', 'sector', 'pe_ratio', 'dividend_yield', 'value_score']]
                print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Generate comparative report
        if any(not df.empty for df in results.values()):
            report_file = os.path.join(output_dir, 'comparative_report.html')
            print(f"\nGenerating comparative report to {report_file}...")
            report_generator.generate_comparative_report(results, output_file=report_file)
            print(f"Report saved to {report_file}")
    except Exception as e:
        print(f"Error in demo_multi_region_screening: {e}")

def demo_portfolio_creation(data_provider, screener, analyzer, visualizer, report_generator, output_dir):
    """
    Demonstrate creating and analyzing a value portfolio.
    
    Args:
        data_provider: DataProvider instance
        screener: Screener instance
        analyzer: Analyzer instance
        visualizer: Visualizer instance
        report_generator: ReportGenerator instance
        output_dir: Output directory path
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 3: VALUE PORTFOLIO CREATION AND ANALYSIS")
    print("="*80)
    
    try:
        # Create a portfolio with top 5 stocks from each region
        regions = ['europe', 'us']
        stocks_per_region = 5
        
        print(f"\nCreating a portfolio with top {stocks_per_region} stocks from each region: {', '.join(regions)}")
        
        portfolio_stocks = pd.DataFrame()
        for region in regions:
            print(f"\nScreening {region} stocks...")
            regional_stocks = screener.screen_stocks(region=region, limit=stocks_per_region)
            if not regional_stocks.empty:
                portfolio_stocks = pd.concat([portfolio_stocks, regional_stocks])
        
        if portfolio_stocks.empty:
            print("No stocks found for portfolio.")
            return
        
        # Display portfolio
        print(f"\nPortfolio created with {len(portfolio_stocks)} stocks:")
        display_df = portfolio_stocks[['ticker', 'name', 'sector', 'country', 'pe_ratio', 'dividend_yield', 'value_score']]
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Generate portfolio report
        report_file = os.path.join(output_dir, 'portfolio_report.html')
        print(f"\nGenerating portfolio report to {report_file}...")
        
        # Create detailed analysis for top 3 stocks
        detailed_analysis = {}
        top_stocks = portfolio_stocks.head(3)['ticker'].tolist()
        print(f"\nPerforming detailed analysis on top stocks: {', '.join(top_stocks)}")
        
        for ticker in top_stocks:
            detailed_analysis[ticker] = analyzer.analyze_stock(ticker)
        
        report_generator.generate_portfolio_report(portfolio_stocks, detailed_analysis, output_file=report_file)
        print(f"Portfolio report saved to {report_file}")
        
        # Create portfolio visualization
        chart_file = os.path.join(output_dir, 'portfolio_allocation.png')
        print(f"\nCreating portfolio allocation chart to {chart_file}...")
        visualizer.create_portfolio_allocation_chart(portfolio_stocks, output_file=chart_file)
        print(f"Chart saved to {chart_file}")
    except Exception as e:
        print(f"Error in demo_portfolio_creation: {e}")

def main():
    """Run all demonstrations."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="WorldScreener Demonstration Script")
    parser.add_argument("--demo-num", type=int, help="Run a specific demo (1-3)")
    parser.add_argument("--use-real-data", action="store_true", help="Use real data instead of mock data")
    parser.add_argument("--output-dir", type=str, help="Output directory for reports and charts")
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    create_output_dir(output_dir)
    print(f"Demo output will be saved to: {output_dir}")
    
    # Initialize components
    print("Initializing WorldScreener components...")
    
    # Load configuration
    config = load_config()
    
    # Create WorldScreener instance
    print("Using mock data for demonstration (Yahoo Finance has rate limits)")
    if args.use_real_data:
        print("WARNING: Using real data may hit API rate limits and cause errors")
    
    # Create a screener instance with mock data enabled by default
    screener = create_screener(use_mock_data=not args.use_real_data)
    
    # Extract components from screener
    data_provider = screener.data_provider
    analyzer = screener.analyzer
    visualizer = screener.visualizer
    report_generator = screener.report_generator
    
    # Run demonstrations
    if args.demo_num:
        if args.demo_num == 1:
            demo_spanish_stocks(data_provider, screener, analyzer, visualizer, report_generator, output_dir)
        elif args.demo_num == 2:
            demo_multi_region_screening(data_provider, screener, analyzer, visualizer, report_generator, output_dir)
        elif args.demo_num == 3:
            demo_portfolio_creation(data_provider, screener, analyzer, visualizer, report_generator, output_dir)
        else:
            print(f"Invalid demo number: {args.demo_num}. Please choose 1-3.")
    else:
        print("Running all demonstrations...")
        demo_spanish_stocks(data_provider, screener, analyzer, visualizer, report_generator, output_dir)
        demo_multi_region_screening(data_provider, screener, analyzer, visualizer, report_generator, output_dir)
        demo_portfolio_creation(data_provider, screener, analyzer, visualizer, report_generator, output_dir)
    
    print("\n" + "="*80)
    print("DEMONSTRATIONS COMPLETED")
    print("="*80)
    print(f"\nAll demo outputs have been saved to: {output_dir}")

if __name__ == "__main__":
    main()