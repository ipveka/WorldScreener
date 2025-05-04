#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Report Generator module for WorldScreener.

This module handles generation of reports in various formats.
"""

import logging
import os
import json
from datetime import datetime
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import jinja2
from src.utils import format_currency, format_percentage

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generator for various types of reports from screening results.
    """
    
    def __init__(self, config=None):
        """
        Initialize the ReportGenerator.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        
        # Set up Jinja2 template environment
        templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        logger.info("ReportGenerator initialized")
    
    def generate_report(self, stocks_df, output_format='text', output_file=None):
        """
        Generate a value stocks report in the specified format.
        
        Args:
            stocks_df (pandas.DataFrame): DataFrame with stock data
            output_format (str): Report format ('text', 'csv', 'json', 'html')
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content or file path if saved
        """
        if stocks_df.empty:
            return "No stocks found matching criteria."
        
        # Add price field for compatibility with templates
        if 'current_price' in stocks_df.columns and 'price' not in stocks_df.columns:
            stocks_df = stocks_df.copy()
            stocks_df['price'] = stocks_df['current_price']
        
        # Generate report based on format
        if output_format == 'text':
            report = self._generate_text_report(stocks_df)
        elif output_format == 'csv':
            report = self._generate_csv_report(stocks_df)
        elif output_format == 'json':
            report = self._generate_json_report(stocks_df)
        elif output_format == 'html':
            report = self._generate_html_report(stocks_df)
        else:
            report = f"Unsupported format: {output_format}"
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
            return f"Report saved to {output_file}"
        
        return report
    
    def _generate_text_report(self, df, output_file=None):
        """
        Generate a text report.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content or file path if saved
        """
        # Create a text table using tabulate
        text_report = f"WorldScreener Report\n"
        text_report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        text_report += tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(text_report)
            return f"Report saved to {output_file}"
        return text_report
    
    def _generate_csv_report(self, df, output_file=None):
        """
        Generate a CSV report.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content or file path if saved
        """
        if output_file:
            df.to_csv(output_file, index=False)
            return f"Report saved to {output_file}"
        return df.to_csv(index=False)
    
    def _generate_json_report(self, df, output_file=None):
        """
        Generate a JSON report.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content or file path if saved
        """
        if output_file:
            df.to_json(output_file, orient='records', indent=4)
            return f"Report saved to {output_file}"
        return df.to_json(orient='records', indent=4)
    
    def _generate_html_report(self, df, output_file=None):
        """
        Generate an HTML report.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content or file path if saved
        """
        try:
            # Try to use Jinja template
            template = self.jinja_env.get_template('report_template.html')
            
            html_report = template.render(
                title="WorldScreener Report",
                generated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                table=df.to_html(index=False, classes='data-table'),
                stocks=df.to_dict('records')
            )
        except Exception as e:
            logger.warning(f"Error using Jinja template: {e}. Falling back to basic HTML.")
            # Fallback to basic HTML
            html_report = f"""
            <html>
            <head>
                <title>WorldScreener Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #2c3e50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>WorldScreener Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                {df.to_html(index=False)}
            </body>
            </html>
            """
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            return f"Report saved to {output_file}"
        return html_report
    
    def generate_portfolio_report(self, portfolio_df, detailed_analysis=None, output_file=None):
        """
        Generate a portfolio report with detailed analysis.
        
        Args:
            portfolio_df (pandas.DataFrame): DataFrame with portfolio stocks
            detailed_analysis (dict, optional): Dictionary with detailed stock analysis
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content or file path if saved
        """
        if portfolio_df.empty:
            return "No stocks in portfolio."
        
        try:
            # Try to use Jinja template
            template = self.jinja_env.get_template('portfolio_template.html')
            
            html_report = template.render(
                title="WorldScreener Portfolio Report",
                generated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                portfolio=portfolio_df.to_dict('records'),
                detailed_analysis=detailed_analysis or {},
                avg_pe=portfolio_df['pe_ratio'].mean(),
                avg_div=portfolio_df['dividend_yield'].mean(),
                avg_score=portfolio_df['value_score'].mean()
            )
        except Exception as e:
            logger.warning(f"Error using Jinja template: {e}. Falling back to basic HTML.")
            # Fallback to basic HTML
            html_report = self._generate_basic_portfolio_html(portfolio_df, detailed_analysis)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            return f"Portfolio report saved to {output_file}"
        return html_report
    
    def _generate_basic_portfolio_html(self, portfolio_df, detailed_analysis=None):
        """
        Generate basic portfolio HTML without Jinja.
        
        Args:
            portfolio_df (pandas.DataFrame): DataFrame with portfolio stocks
            detailed_analysis (dict, optional): Dictionary with detailed stock analysis
            
        Returns:
            str: HTML report content
        """
        # Basic HTML template for portfolio report
        html = f"""
        <html>
        <head>
            <title>WorldScreener Portfolio Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ margin-bottom: 30px; }}
                .portfolio-table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
                .portfolio-table th, .portfolio-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .portfolio-table th {{ background-color: #2c3e50; color: white; }}
                .portfolio-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .stock-analysis {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metrics-container {{ display: flex; flex-wrap: wrap; }}
                .metric-group {{ width: 45%; margin-right: 5%; margin-bottom: 20px; }}
                .value-good {{ color: green; }}
                .value-fair {{ color: orange; }}
                .value-poor {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>WorldScreener Portfolio Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="summary">
                    <h2>Portfolio Overview</h2>
                    <p>A portfolio consisting of {len(portfolio_df)} stocks.</p>
                    <p>Average P/E Ratio: {portfolio_df['pe_ratio'].mean():.2f}</p>
                    <p>Average Dividend Yield: {portfolio_df['dividend_yield'].mean():.2f}%</p>
                    <p>Average Value Score: {portfolio_df['value_score'].mean():.2f}</p>
                </div>
                
                <h2>Portfolio Composition</h2>
                <table class="portfolio-table">
                    <tr>
                        <th>Ticker</th>
                        <th>Name</th>
                        <th>Sector</th>
                        <th>Country</th>
                        <th>Price</th>
                        <th>P/E Ratio</th>
                        <th>Dividend Yield</th>
                        <th>Value Score</th>
                    </tr>
        """
        
        # Add each stock to the HTML table
        for _, stock in portfolio_df.iterrows():
            value_class = "value-good" if stock['value_score'] >= 70 else "value-fair" if stock['value_score'] >= 50 else "value-poor"
            
            html += f"""
                    <tr>
                        <td>{stock['ticker']}</td>
                        <td>{stock['name']}</td>
                        <td>{stock['sector'] if 'sector' in stock else 'N/A'}</td>
                        <td>{stock['country'] if 'country' in stock else 'N/A'}</td>
                        <td>{stock['price']:.2f if 'price' in stock and stock['price'] != 'N/A' else 'N/A'}</td>
                        <td>{stock['pe_ratio']:.2f if 'pe_ratio' in stock and stock['pe_ratio'] != 'N/A' else 'N/A'}</td>
                        <td>{stock['dividend_yield']:.2f}% if 'dividend_yield' in stock and stock['dividend_yield'] != 'N/A' else 'N/A'</td>
                        <td class="{value_class}">{stock['value_score']:.2f if 'value_score' in stock and stock['value_score'] != 'N/A' else 'N/A'}</td>
                    </tr>
            """
        
        html += """
                </table>
        """
        
        # Add detailed analysis if available
        if detailed_analysis:
            html += """
                <h2>Top Stocks Analysis</h2>
            """
            
            for ticker, analysis in detailed_analysis.items():
                name = analysis.get('Name', ticker)
                sector = analysis.get('Sector', 'N/A')
                
                value_score = analysis.get('Value Score (0-100)', 'N/A')
                if value_score != 'N/A':
                    value_class = "value-good" if float(value_score) >= 70 else "value-fair" if float(value_score) >= 50 else "value-poor"
                    value_assessment = f'<span class="{value_class}">{analysis.get("Value Assessment", "N/A")}</span>'
                else:
                    value_assessment = 'N/A'
                    
                html += f"""
                <div class="stock-analysis">
                    <h3>{name} ({ticker})</h3>
                    <p>Sector: {sector} | Value Assessment: {value_assessment}</p>
                    
                    <div class="metrics-container">
                        <div class="metric-group">
                            <h4>General Information</h4>
                            <p>Market Cap: {analysis.get('Market Cap', 'N/A')}</p>
                            <p>Current Price: {analysis.get('Current Price', 'N/A')}</p>
                            <p>52-Week Range: {analysis.get('52-Week Low', 'N/A')} - {analysis.get('52-Week High', 'N/A')}</p>
                        </div>
                        
                        <div class="metric-group">
                            <h4>Value Metrics</h4>
                            <p>P/E Ratio: {analysis.get('P/E Ratio', 'N/A')}</p>
                            <p>P/B Ratio: {analysis.get('P/B Ratio', 'N/A')}</p>
                            <p>Dividend Yield: {analysis.get('Dividend Yield (%)', 'N/A')}%</p>
                            <p>ROE: {analysis.get('ROE (%)', 'N/A')}%</p>
                            <p>Debt to Equity: {analysis.get('Debt to Equity', 'N/A')}</p>
                        </div>
                    </div>
                </div>
                """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def generate_comparative_report(self, stocks_dict, output_file=None):
        """
        Generate a comparative report for stocks from different regions.
        
        Args:
            stocks_dict (dict): Dictionary with region names as keys and DataFrames as values
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content
        """
        # Check if we have any stocks
        if not stocks_dict or all(df.empty for df in stocks_dict.values()):
            logger.warning("No stocks available for comparative report")
            return "No stocks available for comparative report"
        
        try:
            # Try to load template
            template = self.jinja_env.get_template('comparative_report.html')
            
            # Prepare data for template
            template_data = {
                'title': 'Value Stocks Comparative Report',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'regions': list(stocks_dict.keys()),
                'stocks_dict': stocks_dict,
                'has_stocks': any(not df.empty for df in stocks_dict.values())
            }
            
            # Render template
            report_content = template.render(**template_data)
            
        except Exception as e:
            logger.warning(f"Error using Jinja template: {e}. Falling back to basic HTML.")
            
            # Create a basic HTML report
            report_content = """
            <html>
            <head>
                <title>Comparative Value Stocks Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                    h1, h2, h3 { color: #2c3e50; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #2c3e50; color: white; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Comparative Value Stocks Report</h1>
                    <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            """
            
            # Add stocks by region
            for region, stocks_df in stocks_dict.items():
                if not stocks_df.empty:
                    # Add price field for compatibility
                    if 'current_price' in stocks_df.columns and 'price' not in stocks_df.columns:
                        stocks_df = stocks_df.copy()
                        stocks_df['price'] = stocks_df['current_price']
                        
                    # Display only essential columns
                    display_df = stocks_df[['ticker', 'name', 'sector', 'country', 'pe_ratio', 
                                           'dividend_yield', 'pb_ratio', 'roe', 'debt_to_equity', 
                                           'market_cap', 'value_score']].copy()
                    # Convert market cap to billions for display
                    display_df.loc[:, 'market_cap'] = display_df['market_cap'] / 1_000_000_000
                    
                    report_content += f"""
                    <h2>{region.title()} Stocks</h2>
                    {display_df.to_html(classes='table table-striped')}
                    <p>Market cap is displayed in billions of US dollars.</p>
                    """
            
            report_content += """
                </div>
            </body>
            </html>
            """
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Comparative report saved to {output_file}")
        
        return report_content
    
    def generate_sector_report(self, stocks_df, sector_name, output_file=None):
        """
        Generate a report focused on a specific sector.
        
        Args:
            stocks_df (pandas.DataFrame): DataFrame with stock data
            sector_name (str): Name of the sector
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content or file path if saved
        """
        if stocks_df.empty:
            return f"No stocks found matching criteria."
        
        # Filter stocks by sector
        sector_stocks = stocks_df[stocks_df['sector'] == sector_name]
        
        if sector_stocks.empty:
            return f"No stocks found in {sector_name} sector."
        
        # Generate report
        try:
            # Try to use Jinja template
            template = self.jinja_env.get_template('sector_template.html')
            
            html_report = template.render(
                title=f"{sector_name} Sector Report",
                generated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                sector_name=sector_name,
                stocks=sector_stocks.to_dict('records'),
                avg_pe=sector_stocks['pe_ratio'].mean(),
                avg_div=sector_stocks['dividend_yield'].mean(),
                avg_score=sector_stocks['value_score'].mean()
            )
        except Exception as e:
            logger.warning(f"Error using Jinja template: {e}. Falling back to basic HTML.")
            # Fallback to basic HTML
            html_report = f"""
            <html>
            <head>
                <title>{sector_name} Sector Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; margin-bottom: 30px; border-radius: 5px; border-left: 5px solid #2c3e50; }}
                    .stock-table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
                    .stock-table th, .stock-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .stock-table th {{ background-color: #2c3e50; color: white; }}
                    .stock-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .value-good {{ color: green; }}
                    .value-fair {{ color: orange; }}
                    .value-poor {{ color: red; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{sector_name} Sector Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="summary">
                        <h2>Sector Summary</h2>
                        <p><strong>Number of Stocks:</strong> {len(sector_stocks)}</p>
                        <p><strong>Average P/E Ratio:</strong> {sector_stocks['pe_ratio'].mean():.2f}</p>
                        <p><strong>Average Dividend Yield:</strong> {sector_stocks['dividend_yield'].mean():.2f}%</p>
                        <p><strong>Average Value Score:</strong> {sector_stocks['value_score'].mean():.2f}</p>
                    </div>
                    
                    <h2>{sector_name} Stocks</h2>
                    <table class="stock-table">
                        <tr>
                            <th>Ticker</th>
                            <th>Name</th>
                            <th>Country</th>
                            <th>Price</th>
                            <th>P/E Ratio</th>
                            <th>Dividend Yield</th>
                            <th>ROE</th>
                            <th>Debt/Equity</th>
                            <th>Value Score</th>
                        </tr>
            """
            
            # Add each stock to the table
            for _, stock in sector_stocks.iterrows():
                value_class = "value-good" if stock['value_score'] >= 70 else "value-fair" if stock['value_score'] >= 50 else "value-poor"
                
                html_report += f"""
                        <tr>
                            <td>{stock['ticker']}</td>
                            <td>{stock['name']}</td>
                            <td>{stock['country'] if 'country' in stock else 'N/A'}</td>
                            <td>{stock['price']:.2f if 'price' in stock and stock['price'] != 'N/A' else 'N/A'}</td>
                            <td>{stock['pe_ratio']:.2f if 'pe_ratio' in stock and stock['pe_ratio'] != 'N/A' else 'N/A'}</td>
                            <td>{stock['dividend_yield']:.2f}% if 'dividend_yield' in stock and stock['dividend_yield'] != 'N/A' else 'N/A'</td>
                            <td>{stock['roe']:.2f}% if 'roe' in stock and stock['roe'] != 'N/A' else 'N/A'</td>
                            <td>{stock['debt_to_equity']:.2f if 'debt_to_equity' in stock and stock['debt_to_equity'] != 'N/A' else 'N/A'}</td>
                            <td class="{value_class}">{stock['value_score']:.2f if 'value_score' in stock and stock['value_score'] != 'N/A' else 'N/A'}</td>
                        </tr>
                """
            
            html_report += """
                    </table>
                </div>
            </body>
            </html>
            """
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            return f"Sector report saved to {output_file}"
        return html_report
    
    def generate_stock_report(self, stock_analysis, output_file=None):
        """
        Generate a detailed report for a single stock.
        
        Args:
            stock_analysis (dict): Dictionary with detailed stock analysis
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content
        """
        # Check if we have any analysis data
        if not stock_analysis:
            return "No analysis data available for stock report"
        
        # Extract basic stock info
        ticker = stock_analysis.get('ticker', 'Unknown')
        name = stock_analysis.get('name', ticker)
        
        # Create a basic HTML report
        html = f"""
        <html>
        <head>
            <title>{name} ({ticker}) - Detailed Stock Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric-group {{ width: 48%; margin-right: 2%; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #2c3e50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .good {{ color: green; }}
                .fair {{ color: orange; }}
                .poor {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{name} ({ticker}) - Detailed Analysis</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="summary">
                    <h2>Stock Summary</h2>
                    <p><strong>Name:</strong> {name}</p>
                    <p><strong>Ticker:</strong> {ticker}</p>
                    <p><strong>Sector:</strong> {stock_analysis.get('sector', stock_analysis.get('Sector', 'N/A'))}</p>
                    <p><strong>Industry:</strong> {stock_analysis.get('industry', stock_analysis.get('Industry', 'N/A'))}</p>
                    <p><strong>Country:</strong> {stock_analysis.get('country', stock_analysis.get('Country', 'N/A'))}</p>
                    <p><strong>Current Price:</strong> {stock_analysis.get('current_price', stock_analysis.get('Current Price', 'N/A'))}</p>
                    <p><strong>Market Cap:</strong> {format_currency(stock_analysis.get('market_cap', stock_analysis.get('Market Cap', 0)) / 1_000_000_000)} B</p>
                </div>
                
                <h2>Valuation Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Assessment</th>
                    </tr>
                    <tr>
                        <td>P/E Ratio</td>
                        <td>{stock_analysis.get('pe_ratio', stock_analysis.get('P/E Ratio', 'N/A'))}</td>
                        <td class="{self._get_assessment_class(stock_analysis.get('pe_ratio_assessment', 'N/A'))}">
                            {stock_analysis.get('pe_ratio_assessment', 'N/A')}
                        </td>
                    </tr>
                    <tr>
                        <td>P/B Ratio</td>
                        <td>{stock_analysis.get('pb_ratio', stock_analysis.get('P/B Ratio', 'N/A'))}</td>
                        <td class="{self._get_assessment_class(stock_analysis.get('pb_ratio_assessment', 'N/A'))}">
                            {stock_analysis.get('pb_ratio_assessment', 'N/A')}
                        </td>
                    </tr>
                    <tr>
                        <td>Dividend Yield</td>
                        <td>{stock_analysis.get('dividend_yield', stock_analysis.get('Dividend Yield (%)', 'N/A'))}%</td>
                        <td class="{self._get_assessment_class(stock_analysis.get('dividend_yield_assessment', 'N/A'))}">
                            {stock_analysis.get('dividend_yield_assessment', 'N/A')}
                        </td>
                    </tr>
                    <tr>
                        <td>ROE</td>
                        <td>{stock_analysis.get('roe', stock_analysis.get('ROE (%)', 'N/A'))}%</td>
                        <td class="{self._get_assessment_class(stock_analysis.get('roe_assessment', 'N/A'))}">
                            {stock_analysis.get('roe_assessment', 'N/A')}
                        </td>
                    </tr>
                    <tr>
                        <td>Debt to Equity</td>
                        <td>{stock_analysis.get('debt_to_equity', stock_analysis.get('Debt to Equity', 'N/A'))}</td>
                        <td class="{self._get_assessment_class(stock_analysis.get('debt_to_equity_assessment', 'N/A'))}">
                            {stock_analysis.get('debt_to_equity_assessment', 'N/A')}
                        </td>
                    </tr>
                </table>
                
                <h2>Overall Value Assessment</h2>
                <div class="summary">
                    <p><strong>Value Score:</strong> {stock_analysis.get('value_score', 'N/A')}</p>
                    <p><strong>Assessment:</strong> <span class="{self._get_assessment_class(stock_analysis.get('overall_assessment', 'N/A'))}">
                        {stock_analysis.get('overall_assessment', 'N/A')}
                    </span></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Stock report for {name} saved to {output_file}")
        
        return html
    
    def _get_assessment_class(self, assessment):
        """
        Get CSS class based on assessment value.
        
        Args:
            assessment (str): Assessment value
            
        Returns:
            str: CSS class
        """
        if not assessment or assessment == 'N/A':
            return ''
            
        assessment = assessment.lower()
        if 'excellent' in assessment or 'good' in assessment or 'strong' in assessment:
            return 'good'
        elif 'fair' in assessment or 'average' in assessment or 'moderate' in assessment:
            return 'fair'
        elif 'poor' in assessment or 'weak' in assessment or 'high risk' in assessment:
            return 'poor'
        else:
            return ''
    
    def generate_stock_analysis_report(self, analysis, output_file=None):
        """
        Generate a detailed stock analysis report.
        
        Args:
            analysis (dict): Dictionary with stock analysis data
            output_file (str, optional): Output file path
            
        Returns:
            str: Report content or file path if saved
        """
        if not analysis:
            return "No analysis data available."
        
        # Basic text report
        text_report = f"\n{'='*50}\n"
        text_report += f"Value Analysis for {analysis.get('Name', 'Unknown')}\n"
        text_report += f"{'='*50}\n\n"
        
        # Print metrics in sections
        text_report += "General Information:\n"
        text_report += f"- Name: {analysis.get('Name', 'N/A')}\n"
        text_report += f"- Ticker: {analysis.get('Ticker', 'N/A')}\n"
        text_report += f"- Sector: {analysis.get('Sector', 'N/A')}\n"
        text_report += f"- Industry: {analysis.get('Industry', 'N/A')}\n"
        text_report += f"- Current Price: {analysis.get('Current Price', 'N/A')}\n"
        text_report += f"- Market Cap: {analysis.get('Market Cap', 'N/A')}\n\n"
        
        # Value Metrics
        text_report += "Value Metrics:\n"
        text_report += f"- P/E Ratio: {analysis.get('P/E Ratio', 'N/A')}\n"
        text_report += f"- Forward P/E: {analysis.get('Forward P/E', 'N/A')}\n"
        text_report += f"- P/B Ratio: {analysis.get('P/B Ratio', 'N/A')}\n"
        text_report += f"- P/S Ratio: {analysis.get('P/S Ratio', 'N/A')}\n"
        text_report += f"- PEG Ratio: {analysis.get('PEG Ratio', 'N/A')}\n"
        text_report += f"- Dividend Yield: {analysis.get('Dividend Yield', 'N/A')}\n"
        text_report += f"- Dividend Payout Ratio: {analysis.get('Dividend Payout Ratio', 'N/A')}\n\n"
        
        # Growth Metrics
        text_report += "Growth Metrics:\n"
        text_report += f"- Revenue Growth (YoY): {analysis.get('Revenue Growth (YoY)', 'N/A')}\n"
        text_report += f"- Earnings Growth (YoY): {analysis.get('Earnings Growth (YoY)', 'N/A')}\n"
        text_report += f"- 5-Year Revenue CAGR: {analysis.get('5-Year Revenue CAGR', 'N/A')}\n"
        text_report += f"- 5-Year Earnings CAGR: {analysis.get('5-Year Earnings CAGR', 'N/A')}\n\n"
        
        # Financial Health
        text_report += "Financial Health:\n"
        text_report += f"- ROE: {analysis.get('ROE', 'N/A')}\n"
        text_report += f"- ROA: {analysis.get('ROA', 'N/A')}\n"
        text_report += f"- ROIC: {analysis.get('ROIC', 'N/A')}\n"
        text_report += f"- Debt to Equity: {analysis.get('Debt to Equity', 'N/A')}\n"
        text_report += f"- Current Ratio: {analysis.get('Current Ratio', 'N/A')}\n"
        text_report += f"- Interest Coverage: {analysis.get('Interest Coverage', 'N/A')}\n\n"
        
        # Value Analysis
        text_report += "Value Analysis:\n"
        text_report += f"- Value Score: {analysis.get('Value Score', 'N/A')}\n"
        text_report += f"- Intrinsic Value Estimate: {analysis.get('Intrinsic Value Estimate', 'N/A')}\n"
        text_report += f"- Margin of Safety: {analysis.get('Margin of Safety', 'N/A')}\n"
        text_report += f"- Recommendation: {analysis.get('Recommendation', 'N/A')}\n\n"
        
        if output_file and output_file.endswith(('.txt', '.text')):
            with open(output_file, 'w') as f:
                f.write(text_report)
            return f"Report saved to {output_file}"
        
        # Generate HTML report if requested
        if output_file and output_file.endswith('.html'):
            try:
                # Try to use Jinja template
                template = self.jinja_env.get_template('analysis_template.html')
                
                html_report = template.render(
                    title=f"Value Analysis for {analysis.get('Name', 'Unknown')}",
                    generated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    analysis=analysis
                )
            except Exception as e:
                logger.warning(f"Error using Jinja template: {e}. Falling back to basic HTML.")
                # Fallback to basic HTML
                html_report = self._generate_basic_analysis_html(analysis)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            return f"Report saved to {output_file}"
        
        return text_report
    
    def _generate_basic_analysis_html(self, analysis):
        """
        Generate basic stock analysis HTML without Jinja.
        
        Args:
            analysis (dict): Dictionary with stock analysis data
            
        Returns:
            str: HTML report content
        """
        # Create HTML sections
        general_info = ""
        for key in ['Name', 'Ticker', 'Sector', 'Industry', 'Current Price', 'Market Cap']:
            if key in analysis:
                general_info += f"<tr><td>{key}</td><td>{analysis[key]}</td></tr>"
        
        value_metrics = ""
        for key in ['P/E Ratio', 'Forward P/E', 'P/B Ratio', 'P/S Ratio', 'PEG Ratio', 
                   'Dividend Yield', 'Dividend Payout Ratio']:
            if key in analysis:
                value_metrics += f"<tr><td>{key}</td><td>{analysis[key]}</td></tr>"
        
        growth_metrics = ""
        for key in ['Revenue Growth (YoY)', 'Earnings Growth (YoY)', 
                   '5-Year Revenue CAGR', '5-Year Earnings CAGR']:
            if key in analysis:
                growth_metrics += f"<tr><td>{key}</td><td>{analysis[key]}</td></tr>"
        
        financial_health = ""
        for key in ['ROE', 'ROA', 'ROIC', 'Debt to Equity', 'Current Ratio', 'Interest Coverage']:
            if key in analysis:
                financial_health += f"<tr><td>{key}</td><td>{analysis[key]}</td></tr>"
        
        value_analysis = ""
        for key in ['Value Score', 'Intrinsic Value Estimate', 'Margin of Safety', 'Recommendation']:
            if key in analysis:
                value_analysis += f"<tr><td>{key}</td><td>{analysis[key]}</td></tr>"
        
        # Create the full HTML report
        html_report = f"""
        <html>
        <head>
            <title>Stock Analysis: {analysis.get('Name', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                h1, h2 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
                h2 {{ margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #2c3e50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .section {{ margin-bottom: 30px; background: #fff; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .recommendation {{ font-weight: bold; font-size: 1.2em; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .neutral {{ color: orange; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Stock Analysis: {analysis.get('Name', 'Unknown')} ({analysis.get('Ticker', 'N/A')})</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>General Information</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {general_info}
                </table>
            </div>
            
            <div class="section">
                <h2>Value Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {value_metrics}
                </table>
            </div>
            
            <div class="section">
                <h2>Growth Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {growth_metrics}
                </table>
            </div>
            
            <div class="section">
                <h2>Financial Health</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {financial_health}
                </table>
            </div>
            
            <div class="section">
                <h2>Value Analysis</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {value_analysis}
                </table>
            </div>
        </body>
        </html>
        """
        
        return html_report