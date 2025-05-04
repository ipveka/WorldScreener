#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WorldScreener - Command Line Interface

This script provides a command-line interface to the WorldScreener package.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import package modules
from src import create_screener, setup_logging, load_config
from src.utils import create_output_dir


def main():
    """Main function for the command-line interface."""
    parser = argparse.ArgumentParser(
        description='WorldScreener - Value Stock Screening Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Screen command
    screen_parser = subparsers.add_parser('screen', help='Screen stocks')
    screen_parser.add_argument('--region', choices=['spain', 'europe', 'eurozone', 'us', 'global'], 
                        default='europe', help='Region to screen for stocks')
    screen_parser.add_argument('--max-pe', type=float, default=None, 
                        help='Maximum P/E ratio')
    screen_parser.add_argument('--min-div', type=float, default=None, 
                        help='Minimum dividend yield (%)')
    screen_parser.add_argument('--max-pb', type=float, default=None, 
                        help='Maximum P/B ratio')
    screen_parser.add_argument('--min-roe', type=float, default=None, 
                        help='Minimum Return on Equity (%)')
    screen_parser.add_argument('--max-debt', type=float, default=None, 
                        help='Maximum Debt-to-Equity ratio')
    screen_parser.add_argument('--min-score', type=float, default=None, 
                        help='Minimum value score (0-100)')
    screen_parser.add_argument('--sectors', type=str, nargs='+',
                        help='Filter by sectors (e.g., "Banking Technology")')
    screen_parser.add_argument('--countries', type=str, nargs='+',
                        help='Filter by countries (e.g., "ES DE FR")')
    screen_parser.add_argument('--limit', type=int, default=None, 
                        help='Maximum number of results')
    screen_parser.add_argument('--output', choices=['text', 'csv', 'json', 'html', 'excel'], 
                        default=None, help='Output format')
    screen_parser.add_argument('--output-file', type=str, 
                        help='Output file path')
    screen_parser.add_argument('--visualize', choices=['value_comparison', 'sector_breakdown', 
                                             'metrics_comparison', 'country_breakdown'], 
                        help='Create visualization')
    screen_parser.add_argument('--plot-file', type=str, 
                        help='Output file path for plot')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a specific stock')
    analyze_parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    analyze_parser.add_argument('--output-file', type=str, help='Output file path')
    analyze_parser.add_argument('--plot', action='store_true', help='Create stock history plot')
    analyze_parser.add_argument('--plot-file', type=str, help='Output file path for plot')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple stocks')
    compare_parser.add_argument('tickers', type=str, nargs='+', help='Stock ticker symbols')
    compare_parser.add_argument('--output', choices=['text', 'csv', 'json', 'html'], 
                        default='text', help='Output format')
    compare_parser.add_argument('--output-file', type=str, help='Output file path')
    compare_parser.add_argument('--plot', action='store_true', help='Create comparative plot')
    compare_parser.add_argument('--plot-file', type=str, help='Output file path for plot')
    compare_parser.add_argument('--normalize', action='store_true', 
                        help='Normalize prices to percentage change')
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Create a value portfolio')
    portfolio_parser.add_argument('--regions', type=str, nargs='+',
                          choices=['spain', 'europe', 'eurozone', 'us', 'global'],
                          default=['europe', 'us'],
                          help='Regions to include in portfolio')
    portfolio_parser.add_argument('--stocks-per-region', type=int, default=5,
                          help='Number of stocks to include from each region')
    portfolio_parser.add_argument('--config', type=str, help='Custom config for regional criteria')
    portfolio_parser.add_argument('--output-dir', type=str, default='output',
                          help='Output directory for portfolio files')
    portfolio_parser.add_argument('--analyze-top', type=int, default=3,
                          help='Number of top stocks to analyze in detail')
    
    # Sector command
    sector_parser = subparsers.add_parser('sector', help='Analyze a specific sector')
    sector_parser.add_argument('sector', type=str, help='Sector name')
    sector_parser.add_argument('--region', choices=['spain', 'europe', 'eurozone', 'us', 'global'],
                        default='europe', help='Region to analyze')
    sector_parser.add_argument('--output-file', type=str, help='Output file path')
    sector_parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    sector_parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for visualization files')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration script')
    demo_parser.add_argument('--demo-num', type=int, choices=[0, 1, 2, 3, 4], default=0,
                      help='Select specific demo (0=all, 1=Spanish stocks, 2=Custom screening, '
                           '3=Portfolio creation, 4=Sector analysis)')
    demo_parser.add_argument('--output-dir', type=str, default='output',
                      help='Output directory for demo files')
    
    # Global options
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to custom configuration file')
    parser.add_argument('--api-key', type=str, 
                        help='API key for financial data provider')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose logging')
    parser.add_argument('--version', action='version', version='WorldScreener 1.0.0',
                        help='Show version')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger('WorldScreener')
    
    # Create output directory if needed
    if hasattr(args, 'output_file') and args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Initialize screener
    data_provider, screener, analyzer, visualizer, report_generator = create_screener(
        config_path=getattr(args, 'config', None),
        api_key=args.api_key
    )
    
    # Load config
    config = load_config(getattr(args, 'config', None))
    
    # Process command
    if args.command == 'screen':
        return handle_screen_command(args, screener, visualizer, report_generator, config)
    elif args.command == 'analyze':
        return handle_analyze_command(args, analyzer, visualizer, report_generator)
    elif args.command == 'compare':
        return handle_compare_command(args, analyzer, visualizer, report_generator)
    elif args.command == 'portfolio':
        return handle_portfolio_command(args, screener, analyzer, visualizer, report_generator, config)
    elif args.command == 'sector':
        return handle_sector_command(args, screener, visualizer, report_generator)
    elif args.command == 'demo':
        # Import demo module
        try:
            from demo import main as run_demo
            sys.argv = [sys.argv[0], f"--demo={args.demo_num}", f"--output-dir={args.output_dir}"]
            if args.verbose:
                sys.argv.append("--verbose")
            return run_demo()
        except ImportError:
            logger.error("Demo module not found. Please make sure demo.py is in the current directory.")
            return 1
    else:
        # No command specified, show help
        parser.print_help()
        return 0


def handle_screen_command(args, screener, visualizer, report_generator, config):
    """Handle the 'screen' command."""
    logger = logging.getLogger('WorldScreener')
    
    # Build custom screening criteria
    criteria = {}
    
    if args.max_pe is not None:
        criteria['max_pe_ratio'] = args.max_pe
    if args.min_div is not None:
        criteria['min_dividend_yield'] = args.min_div
    if args.max_pb is not None:
        criteria['max_pb_ratio'] = args.max_pb
    if args.min_roe is not None:
        criteria['min_roe'] = args.min_roe
    if args.max_debt is not None:
        criteria['max_debt_to_equity'] = args.max_debt
    if args.min_score is not None:
        criteria['min_value_score'] = args.min_score
    
    # Get limit from config if not specified
    limit = args.limit or config['output']['default_limit']
    
    # Screen stocks
    logger.info(f"Screening stocks in {args.region} region")
    results = screener.screen_stocks(region=args.region, criteria=criteria, limit=limit)
    
    if results.empty:
        logger.error("No stocks found meeting the specified criteria")
        return 1
    
    # Apply additional filters if specified
    if args.sectors:
        results = screener.filter_by_sector(results, args.sectors)
    
    if args.countries:
        results = screener.filter_by_country(results, args.countries)
    
    if results.empty:
        logger.error("No stocks found after applying additional filters")
        return 1
    
    # Display results if no output file specified
    if not args.output_file:
        print(f"\nFound {len(results)} stocks meeting criteria:")
        display_df = results[['ticker', 'name', 'sector', 'country', 'pe_ratio', 
                             'dividend_yield', 'value_score']]
        print(tabulate(display_df, headers='keys', tablefmt='psql', showindex=False,
                     floatfmt=('.2f', '', '', '', '.2f', '.2f', '.2f')))
    
    # Get output format from config if not specified
    output_format = args.output or config['output']['default_format']
    
    # Generate report
    if args.output_file:
        if output_format == 'excel':
            # Create Excel report with a single sheet
            excel_data = {args.region: results}
            report = report_generator.generate_excel_report(excel_data, args.output_file)
        else:
            report = report_generator.generate_report(results, output_format=output_format,
                                                    output_file=args.output_file)
        print(report)
    
    # Create visualization if requested
    if args.visualize:
        if args.plot_file:
            getattr(visualizer, f"create_{args.visualize}_chart")(results, output_file=args.plot_file)
            print(f"Visualization saved to {args.plot_file}")
        else:
            chart = getattr(visualizer, f"create_{args.visualize}_chart")(results)
            plt.show()
    
    return 0


def handle_analyze_command(args, analyzer, visualizer, report_generator):
    """Handle the 'analyze' command."""
    logger = logging.getLogger('WorldScreener')
    
    logger.info(f"Analyzing stock: {args.ticker}")
    
    # Perform detailed analysis
    analysis = analyzer.analyze_stock(args.ticker)
    
    if not analysis:
        logger.error(f"No analysis available for {args.ticker}")
        return 1
    
    # Generate report
    if args.output_file:
        report = report_generator.generate_stock_analysis_report(analysis, args.output_file)
        print(report)
    else:
        # Display key metrics
        print(f"\nAnalysis for {analysis.get('Name', args.ticker)} ({args.ticker}):")
        print(f"- Sector: {analysis.get('Sector', 'N/A')}")
        print(f"- Current Price: {analysis.get('Current Price', 'N/A')}")
        print(f"- P/E Ratio: {analysis.get('P/E Ratio', 'N/A')}")
        print(f"- Dividend Yield: {analysis.get('Dividend Yield (%)', 'N/A')}%")
        print(f"- ROE: {analysis.get('ROE (%)', 'N/A')}%")
        print(f"- Debt to Equity: {analysis.get('Debt to Equity', 'N/A')}")
        
        if 'Value Score (0-100)' in analysis:
            print(f"- Value Score: {analysis.get('Value Score (0-100)', 'N/A')}")
            print(f"- Value Assessment: {analysis.get('Value Assessment', 'N/A')}")
    
    # Create stock history chart if requested
    if args.plot:
        if args.plot_file:
            visualizer.create_stock_history_chart(args.ticker, period="1y", output_file=args.plot_file)
            print(f"Stock history chart saved to {args.plot_file}")
        else:
            chart = visualizer.create_stock_history_chart(args.ticker, period="1y")
            plt.show()
    
    return 0


def handle_compare_command(args, analyzer, visualizer, report_generator):
    """Handle the 'compare' command."""
    logger = logging.getLogger('WorldScreener')
    
    logger.info(f"Comparing stocks: {args.tickers}")
    
    # Get comparison data
    comparison = analyzer.compare_stocks(args.tickers)
    
    if comparison.empty:
        logger.error("No comparison data available")
        return 1
    
    # Display results if no output file specified
    if not args.output_file:
        print(f"\nStock Comparison:")
        display_df = comparison[['ticker', 'name', 'sector', 'country', 'pe_ratio', 
                               'dividend_yield', 'value_score']]
        print(tabulate(display_df, headers='keys', tablefmt='psql', showindex=False,
                     floatfmt=('.2f', '', '', '', '.2f', '.2f', '.2f')))
    
    # Generate report
    if args.output_file:
        report = report_generator.generate_report(comparison, output_format=args.output,
                                               output_file=args.output_file)
        print(report)
    
    # Create comparative chart if requested
    if args.plot:
        if args.plot_file:
            visualizer.create_comparative_chart(args.tickers, period="1y", 
                                             output_file=args.plot_file, 
                                             normalize=args.normalize)
            print(f"Comparative chart saved to {args.plot_file}")
        else:
            chart = visualizer.create_comparative_chart(args.tickers, period="1y", 
                                                      normalize=args.normalize)
            plt.show()
    
    return 0


def handle_portfolio_command(args, screener, analyzer, visualizer, report_generator, config):
    """Handle the 'portfolio' command."""
    logger = logging.getLogger('WorldScreener')
    
    logger.info("Creating value portfolio")
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    
    # Define criteria for different regions
    regional_criteria = {}
    
    # Try to load custom config if specified
    if args.config:
        try:
            with open(args.config, 'r') as f:
                import yaml
                custom_config = yaml.safe_load(f)
                if 'regional_criteria' in custom_config:
                    regional_criteria = custom_config['regional_criteria']
        except Exception as e:
            logger.error(f"Error loading custom config: {e}")
            # Use default criteria
            regional_criteria = {}
    
    # Use default criteria if not loaded from custom config
    if not regional_criteria:
        # Default criteria for each region
        regional_criteria = {
            'spain': {
                'max_pe_ratio': 14.0,         
                'min_dividend_yield': 4.0,    
                'max_pb_ratio': 1.2,          
                'min_roe': 10.0,              
                'max_debt_to_equity': 1.0,    
                'min_value_score': 55.0       
            },
            'europe': {
                'max_pe_ratio': 15.0,
                'min_dividend_yield': 3.5,
                'max_pb_ratio': 1.5,
                'min_roe': 12.0,
                'max_debt_to_equity': 1.2,
                'min_value_score': 50.0
            },
            'eurozone': {
                'max_pe_ratio': 15.0,
                'min_dividend_yield': 3.5,
                'max_pb_ratio': 1.5,
                'min_roe': 11.0,
                'max_debt_to_equity': 1.0,
                'min_value_score': 52.0
            },
            'us': {
                'max_pe_ratio': 18.0,
                'min_dividend_yield': 2.0,
                'max_pb_ratio': 2.0,
                'min_roe': 15.0,
                'max_debt_to_equity': 1.5,
                'min_value_score': 45.0
            },
            'global': {
                'max_pe_ratio': 16.0,
                'min_dividend_yield': 2.5,
                'max_pb_ratio': 1.8,
                'min_roe': 12.0,
                'max_debt_to_equity': 1.3,
                'min_value_score': 48.0
            }
        }
    
    # Screen stocks for each region
    portfolio_stocks = {}
    for region in args.regions:
        criteria = regional_criteria.get(region, config['screening']['default_criteria'])
        logger.info(f"Screening {region} stocks with region-specific criteria")
        stocks = screener.screen_stocks(region=region, criteria=criteria, limit=30)
        portfolio_stocks[region] = stocks
        logger.info(f"Found {len(stocks)} {region} stocks meeting criteria")
    
    # Create a consolidated portfolio
    logger.info("Creating consolidated portfolio")
    
    # Take top N stocks from each region
    consolidated = pd.DataFrame()
    for region, stocks in portfolio_stocks.items():
        if not stocks.empty:
            # Add region column
            top_stocks = stocks.sort_values('value_score', ascending=False).head(args.stocks_per_region)
            top_stocks = top_stocks.copy()
            top_stocks['region'] = region
            consolidated = pd.concat([consolidated, top_stocks])
    
    if consolidated.empty:
        logger.error("No stocks found for portfolio")
        return 1
    
    # Reset index and sort by value score
    consolidated = consolidated.reset_index(drop=True)
    consolidated = consolidated.sort_values('value_score', ascending=False)
    
    # Display portfolio
    print(f"\nDiversified Value Portfolio ({len(consolidated)} stocks):")
    display_df = consolidated[['ticker', 'name', 'region', 'sector', 'pe_ratio', 'dividend_yield', 'value_score']]
    print(tabulate(display_df, headers='keys', tablefmt='psql', showindex=False,
                 floatfmt=('.2f', '', '', '', '.2f', '.2f', '.2f')))
    
    # Calculate portfolio metrics
    portfolio_metrics = {
        'Average P/E Ratio': consolidated['pe_ratio'].mean(),
        'Average Dividend Yield': consolidated['dividend_yield'].mean(),
        'Average ROE': consolidated['roe'].mean(),
        'Average Value Score': consolidated['value_score'].mean(),
        'Stock Count': len(consolidated),
        'Regions': len(args.regions)
    }
    
    print("\nPortfolio Metrics:")
    for metric, value in portfolio_metrics.items():
        print(f"- {metric}: {value:.2f}" if isinstance(value, float) else f"- {metric}: {value}")
    
    # Create portfolio visualizations
    logger.info("Creating portfolio visualizations")
    
    # Sector breakdown
    viz_file = os.path.join(output_dir, "portfolio_sector_breakdown.png")
    visualizer.create_sector_breakdown_chart(consolidated, output_file=viz_file,
                                          title="Sector Allocation of Value Portfolio")
    print(f"Sector breakdown chart saved to {viz_file}")
    
    # Country breakdown
    viz_file = os.path.join(output_dir, "portfolio_country_breakdown.png")
    visualizer.create_country_breakdown_chart(consolidated, output_file=viz_file,
                                           title="Country Allocation of Value Portfolio")
    print(f"Country breakdown chart saved to {viz_file}")
    
    # Analyze top stocks
    logger.info(f"Analyzing top {args.analyze_top} stocks in portfolio")
    detailed_analysis = {}
    
    for _, stock in consolidated.head(args.analyze_top).iterrows():
        ticker = stock['ticker']
        logger.info(f"Analyzing {ticker}")
        
        analysis = analyzer.analyze_stock(ticker)
        if analysis:
            detailed_analysis[ticker] = analysis
            
            # Create stock history chart
            viz_file = os.path.join(output_dir, f"{ticker}_history.png")
            visualizer.create_stock_history_chart(ticker, period="1y", output_file=viz_file)
            print(f"Stock history chart saved to {viz_file}")
    
    # Generate portfolio report
    logger.info("Generating portfolio report")
    report_file = os.path.join(output_dir, "value_portfolio.html")
    report_generator.generate_portfolio_report(consolidated, 
                                            detailed_analysis=detailed_analysis,
                                            output_file=report_file)
    print(f"Portfolio report saved to {report_file}")
    
    # Generate Excel report with multiple sheets
    logger.info("Generating Excel report")
    excel_file = os.path.join(output_dir, "value_portfolio.xlsx")
    
    # Prepare data for Excel report
    excel_data = {'Portfolio': consolidated}
    for region, stocks in portfolio_stocks.items():
        if not stocks.empty:
            excel_data[region.capitalize()] = stocks
    
    report_generator.generate_excel_report(excel_data, excel_file)
    print(f"Excel report saved to {excel_file}")
    
    return 0


def handle_sector_command(args, screener, visualizer, report_generator):
    """Handle the 'sector' command."""
    logger = logging.getLogger('WorldScreener')
    
    logger.info(f"Analyzing {args.sector} sector in {args.region} region")
    
    # Screen stocks
    stocks = screener.screen_stocks(region=args.region, limit=50)
    
    if stocks.empty:
        logger.error(f"No stocks found in {args.region} region")
        return 1
    
    # Filter by sector
    sector_stocks = stocks[stocks['sector'] == args.sector]
    
    if sector_stocks.empty:
        logger.error(f"No {args.sector} stocks found in {args.region} region")
        return 1
    
    # Display sector stocks
    print(f"\nFound {len(sector_stocks)} {args.sector} stocks in {args.region} region:")
    display_df = sector_stocks[['ticker', 'name', 'country', 'pe_ratio', 'dividend_yield', 'value_score']]
    print(tabulate(display_df, headers='keys', tablefmt='psql', showindex=False,
                 floatfmt=('.2f', '', '', '.2f', '.2f', '.2f')))
    
    # Calculate sector metrics
    sector_metrics = {
        'Average P/E Ratio': sector_stocks['pe_ratio'].mean(),
        'Average Dividend Yield': sector_stocks['dividend_yield'].mean(),
        'Average ROE': sector_stocks['roe'].mean(),
        'Average Value Score': sector_stocks['value_score'].mean(),
        'Stock Count': len(sector_stocks)
    }
    
    print(f"\n{args.sector} Sector Metrics:")
    for metric, value in sector_metrics.items():
        print(f"- {metric}: {value:.2f}" if isinstance(value, float) else f"- {metric}: {value}")
    
    # Generate sector report
    if args.output_file:
        logger.info(f"Generating {args.sector} sector report")
        report_generator.generate_sector_report(stocks, args.sector, output_file=args.output_file)
        print(f"Sector report saved to {args.output_file}")
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("Creating sector visualizations")
        output_dir = create_output_dir(args.output_dir)
        
        # Value comparison
        viz_file = os.path.join(output_dir, f"{args.sector.lower()}_value_comparison.png")
        visualizer.create_value_comparison_chart(sector_stocks, output_file=viz_file,
                                              title=f"Value Comparison of {args.sector} Stocks")
        print(f"Value comparison chart saved to {viz_file}")
        
        # Country breakdown
        viz_file = os.path.join(output_dir, f"{args.sector.lower()}_country_breakdown.png")
        visualizer.create_country_breakdown_chart(sector_stocks, output_file=viz_file,
                                               title=f"Country Distribution of {args.sector} Stocks")
        print(f"Country breakdown chart saved to {viz_file}")
        
        # Compare top stocks in the sector
        top_stocks = sector_stocks.sort_values('value_score', ascending=False).head(5)
        tickers = top_stocks['ticker'].tolist()
        
        # Create comparative chart
        viz_file = os.path.join(output_dir, f"{args.sector.lower()}_comparison.png")
        visualizer.create_comparative_chart(tickers, period="1y", output_file=viz_file, normalize=True)
        print(f"Comparative chart saved to {viz_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())