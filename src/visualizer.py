#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualizer module for WorldScreener.

This module provides visualization functionality for stock data.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Visualizer for creating charts and plots from stock data.
    """
    
    def __init__(self, data_provider=None, config=None):
        """
        Initialize the Visualizer.
        
        Args:
            data_provider (DataProvider, optional): DataProvider instance for retrieving stock data
            config (dict, optional): Configuration dictionary
        """
        self.data_provider = data_provider
        self.config = config or {}
        
        # Set up visualization style
        self.setup_style()
        
        logger.info("Visualizer initialized")
    
    def setup_style(self):
        """Set up the visualization style."""
        # Use seaborn style
        sns.set_style('darkgrid')
        
        # Set color palette
        sns.set_palette('viridis')
        
        # Set font size and style
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        # Custom color maps
        self.value_cmap = LinearSegmentedColormap.from_list(
            'value_cmap', ['#ff5e5e', '#ffeb5c', '#52c56f'])
    
    def create_value_comparison_chart(self, stocks_df, output_file=None, title=None):
        """
        Create a horizontal bar chart comparing value scores.
        
        Args:
            stocks_df (pandas.DataFrame): DataFrame with stock data
            output_file (str, optional): Output file path
            title (str, optional): Chart title
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if stocks_df.empty:
            logger.error("No data to visualize")
            return None
        
        plt.figure(figsize=(12, 10))
        
        # Sort by value score and take top 15
        plot_data = stocks_df.sort_values('value_score', ascending=False).head(15)
        
        # Create labels combining ticker and name
        labels = [f"{row['ticker']} ({row['name'][:15]}...)" 
                 if len(row['name']) > 15 else f"{row['ticker']} ({row['name']})" 
                 for _, row in plot_data.iterrows()]
        
        # Create color map based on value score
        colors = plt.cm.viridis(plot_data['value_score'] / 100)
        
        # Plot horizontal bar chart
        bars = plt.barh(labels, plot_data['value_score'], color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}', ha='left', va='center')
        
        plt.xlabel('Value Score (0-100)')
        plt.title(title or 'Top Value Stocks Comparison')
        plt.tight_layout()
        
        # Save or return the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
            plt.close()
        
        return plt.gcf()
    
    def create_sector_breakdown_chart(self, stocks_df, output_file=None, title=None, figsize=(12, 8)):
        """
        Create a sector breakdown chart for a set of stocks.
        
        Args:
            stocks_df (pandas.DataFrame): DataFrame with stock data
            output_file (str, optional): Path to save the chart
            title (str, optional): Chart title
            figsize (tuple, optional): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            if stocks_df.empty:
                logger.warning("No stocks data provided for sector breakdown chart")
                return None
                
            # Count stocks by sector
            sector_counts = stocks_df['sector'].value_counts()
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                sector_counts, 
                labels=sector_counts.index, 
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            
            # Style the chart
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
                
            # Add title
            chart_title = title or "Sector Breakdown"
            ax.set_title(chart_title, fontsize=14, pad=20)
            
            # Add legend
            ax.legend(
                wedges, 
                [f"{sector} ({count})" for sector, count in zip(sector_counts.index, sector_counts)],
                title="Sectors",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            
            plt.tight_layout()
            
            # Save to file if specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Sector breakdown chart saved to {output_file}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sector breakdown chart: {e}")
            return None
            
    def create_region_breakdown_chart(self, stocks_df, output_file=None, title=None, figsize=(12, 8)):
        """
        Create a region breakdown chart for a set of stocks.
        
        Args:
            stocks_df (pandas.DataFrame): DataFrame with stock data
            output_file (str, optional): Path to save the chart
            title (str, optional): Chart title
            figsize (tuple, optional): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            if stocks_df.empty:
                logger.warning("No stocks data provided for region breakdown chart")
                return None
                
            # Count stocks by country
            if 'country' in stocks_df.columns:
                region_counts = stocks_df['country'].value_counts()
            else:
                # If country is not available, try to extract region from ticker
                def get_region(ticker):
                    if ticker.endswith('.T'):
                        return 'Japan'
                    elif any(ticker.endswith(suffix) for suffix in ['.L', '.DE', '.PA', '.MI', '.MC', '.AS', '.SW', '.CO', '.ST', '.HE', '.OL']):
                        return 'Europe'
                    elif '.' not in ticker:
                        return 'US'
                    else:
                        return 'Other'
                
                stocks_df['region'] = stocks_df['ticker'].apply(get_region)
                region_counts = stocks_df['region'].value_counts()
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create pie chart
            colors = plt.cm.tab10.colors
            wedges, texts, autotexts = ax.pie(
                region_counts, 
                labels=region_counts.index, 
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                colors=colors,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            
            # Style the chart
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
                
            # Add title
            chart_title = title or "Regional Breakdown"
            ax.set_title(chart_title, fontsize=14, pad=20)
            
            # Add legend
            ax.legend(
                wedges, 
                [f"{region} ({count})" for region, count in zip(region_counts.index, region_counts)],
                title="Regions",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            
            plt.tight_layout()
            
            # Save to file if specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Region breakdown chart saved to {output_file}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating region breakdown chart: {e}")
            return None
    
    def create_metrics_comparison_chart(self, stocks_df, output_file=None, title=None):
        """
        Create a scatter plot comparing different metrics.
        
        Args:
            stocks_df (pandas.DataFrame): DataFrame with stock data
            output_file (str, optional): Output file path
            title (str, optional): Chart title
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if stocks_df.empty:
            logger.error("No data to visualize")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Use dividend yield and P/E ratio with marker size as value score
        scatter = plt.scatter(stocks_df['pe_ratio'], stocks_df['dividend_yield'], 
                          s=stocks_df['value_score'] * 5, # Size based on value score
                          c=stocks_df['value_score'], # Color based on value score
                          cmap='viridis', alpha=0.7)
        
        plt.xlabel('P/E Ratio')
        plt.ylabel('Dividend Yield (%)')
        plt.title(title or 'Value Metrics Comparison')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Value Score')
        
        # Add annotations for top stocks
        top_stocks = stocks_df.sort_values('value_score', ascending=False).head(5)
        for _, row in top_stocks.iterrows():
            plt.annotate(row['ticker'], 
                        (row['pe_ratio'], row['dividend_yield']),
                        xytext=(5, 5), textcoords='offset points')
        
        # Save or return the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
            plt.close()
        
        return plt.gcf()
    
    def create_country_breakdown_chart(self, stocks_df, output_file=None, title=None):
        """
        Create a bar chart showing country distribution.
        
        Args:
            stocks_df (pandas.DataFrame): DataFrame with stock data
            output_file (str, optional): Output file path
            title (str, optional): Chart title
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if stocks_df.empty:
            logger.error("No data to visualize")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Count stocks by country
        country_counts = stocks_df['country'].value_counts()
        
        # Plot bar chart
        bars = plt.bar(country_counts.index, country_counts.values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}', ha='center', va='bottom')
        
        plt.xlabel('Country')
        plt.ylabel('Number of Stocks')
        plt.title(title or 'Country Distribution of Value Stocks')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or return the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
            plt.close()
        
        return plt.gcf()
    
    def create_stock_history_chart(self, ticker, period="1y", output_file=None):
        """
        Create a stock price history chart.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period (e.g., "1d", "1mo", "1y", "max")
            output_file (str, optional): Output file path
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        # Get historical data
        history = self.data_provider.get_stock_history(ticker, period)
        
        if history.empty:
            logger.error(f"No historical data for {ticker}")
            return None
        
        # Get basic info
        stock_data = self.data_provider.get_stock_data([ticker])
        if not stock_data.empty:
            stock_name = stock_data.iloc[0]['name']
        else:
            stock_name = ticker
        
        # Create figure with 2 subplots (price and volume)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(history.index, history['Close'], label='Close Price')
        
        # Add moving averages
        history['MA50'] = history['Close'].rolling(window=50).mean()
        history['MA200'] = history['Close'].rolling(window=200).mean()
        
        ax1.plot(history.index, history['MA50'], label='50-day MA', linestyle='--')
        if len(history) >= 200:
            ax1.plot(history.index, history['MA200'], label='200-day MA', linestyle='-.')
        
        # Set title and labels
        ax1.set_title(f"{stock_name} ({ticker}) - Price History")
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot volume
        ax2.bar(history.index, history['Volume'], color='gray', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # Format x-axis
        plt.tight_layout()
        
        # Save or return the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
            plt.close()
        
        return fig
    
    def create_comparative_chart(self, tickers, period="1y", output_file=None, normalize=True):
        """
        Create a comparative chart of multiple stocks.
        
        Args:
            tickers (list): List of stock ticker symbols
            period (str): Time period (e.g., "1d", "1mo", "1y", "max")
            output_file (str, optional): Output file path
            normalize (bool): Whether to normalize prices to percentage change
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        plt.figure(figsize=(12, 8))
        
        # Get data for each ticker
        for ticker in tickers:
            history = self.data_provider.get_stock_history(ticker, period)
            
            if history.empty:
                logger.warning(f"No data for {ticker}, skipping")
                continue
            
            # Normalize to percentage change if requested
            if normalize:
                first_price = history['Close'].iloc[0]
                normalized_price = (history['Close'] / first_price - 1) * 100
                plt.plot(history.index, normalized_price, label=ticker)
                ylabel = 'Price Change (%)'
            else:
                plt.plot(history.index, history['Close'], label=ticker)
                ylabel = 'Price'
        
        plt.title('Comparative Performance')
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        
        # Save or return the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
            plt.close()
        
        return plt.gcf()
    
    def create_portfolio_allocation_chart(self, portfolio_df, output_file=None, title=None):
        """
        Create charts showing portfolio allocation by sector and country.
        
        Args:
            portfolio_df (pandas.DataFrame): DataFrame with portfolio stocks
            output_file (str, optional): Output file path
            title (str, optional): Chart title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if portfolio_df.empty:
            logger.warning("No data to visualize")
            return None
        
        # Create figure with two subplots and white background
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        
        # 1. Sector Allocation
        if 'sector' in portfolio_df.columns:
            sector_counts = portfolio_df['sector'].value_counts()
            
            # Create pie chart for sectors
            wedges, texts, autotexts = ax1.pie(
                sector_counts, 
                labels=sector_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            
            # Style the text
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
            
            ax1.set_title('Sector Allocation', fontsize=14, pad=20)
            ax1.axis('equal')  # Equal aspect ratio
        
        # 2. Country Allocation
        if 'country' in portfolio_df.columns:
            country_counts = portfolio_df['country'].value_counts()
            
            # Create pie chart for countries
            wedges, texts, autotexts = ax2.pie(
                country_counts, 
                labels=country_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            
            # Style the text
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
            
            ax2.set_title('Country Allocation', fontsize=14, pad=20)
            ax2.axis('equal')  # Equal aspect ratio
        
        # Set overall title if provided
        if title:
            fig.suptitle(title, fontsize=16, y=1.05)
        else:
            fig.suptitle('Portfolio Allocation', fontsize=16, y=1.05)
        
        # Tight layout
        plt.tight_layout()
        
        # Save to file if specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Portfolio allocation chart saved to {output_file}")
        
        return fig