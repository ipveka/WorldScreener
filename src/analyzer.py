#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyzer module for WorldScreener.

This module provides detailed analysis of individual stocks.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Analyzer:
    """
    Stock analyzer for detailed analysis of individual stocks.
    """
    
    def __init__(self, data_provider, config=None):
        """
        Initialize the Analyzer.
        
        Args:
            data_provider: DataProvider instance
            config (dict, optional): Configuration dictionary
        """
        self.data_provider = data_provider
        self.config = config or {}
        
        # Value score weights
        self.value_score_weights = self.config.get('screening', {}).get('value_score_weights', {})
        
        logger.info("Analyzer initialized")
    
    def analyze_stock(self, ticker):
        """
        Perform a detailed value analysis of a specific stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary with detailed stock analysis
        """
        logger.info(f"Analyzing stock: {ticker}")
        
        # Get basic stock data
        stock = self.data_provider.get_stock_data([ticker])
        if stock.empty:
            logger.error(f"No information available for {ticker}")
            return {}
        
        # Convert DataFrame row to dictionary
        stock_data = stock.iloc[0].to_dict()
        
        # Get historical data
        hist = self.data_provider.get_stock_history(ticker, period="3y")
        
        # Get financial statements
        financials = self.data_provider.get_stock_financials(ticker)
        
        # Create detailed metrics
        metrics = self._create_detailed_metrics(ticker, stock_data, hist, financials)
        
        # Assess stock value
        metrics = self._assess_value(metrics)
        
        return metrics
    
    def _create_detailed_metrics(self, ticker, stock_data, hist, financials):
        """
        Create detailed metrics from stock data.
        
        Args:
            ticker (str): Stock ticker symbol
            stock_data (dict): Basic stock data
            hist (pandas.DataFrame): Historical price data
            financials (dict): Financial statements
            
        Returns:
            dict: Dictionary with detailed metrics
        """
        # Initialize metrics with basic stock data
        metrics = {
            'Name': stock_data.get('name', ticker),
            'Ticker': ticker,
            'Sector': stock_data.get('sector', 'N/A'),
            'Industry': stock_data.get('industry', 'N/A'),
            'Country': stock_data.get('country', 'N/A'),
            'Exchange': stock_data.get('exchange', 'N/A'),
            'Currency': stock_data.get('currency', 'N/A'),
            'Current Price': stock_data.get('price', 'N/A'),
            'Market Cap': stock_data.get('market_cap', 'N/A'),
            'P/E Ratio': stock_data.get('pe_ratio', 'N/A'),
            'P/B Ratio': stock_data.get('pb_ratio', 'N/A'),
            'Dividend Yield (%)': stock_data.get('dividend_yield', 'N/A'),
            'ROE (%)': stock_data.get('roe', 'N/A'),
            'Debt to Equity': stock_data.get('debt_to_equity', 'N/A'),
            '52-Week High': stock_data.get('fifty_two_week_high', 'N/A'),
            '52-Week Low': stock_data.get('fifty_two_week_low', 'N/A')
        }
        
        # Add financial metrics if available
        if not hist.empty:
            # Volatility (standard deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            if len(returns) > 0:
                metrics['Volatility (1-Year)'] = round(returns[-252:].std() * (252 ** 0.5) * 100, 2) if len(returns) >= 252 else 'N/A'
            
            # Maximum Drawdown
            rolling_max = hist['Close'].cummax()
            drawdown = (hist['Close'] - rolling_max) / rolling_max
            metrics['Maximum Drawdown (3-Year) (%)'] = round(drawdown.min() * 100, 2)
            
            # Calculate beta if S&P 500 data is available
            try:
                market = self.data_provider.get_stock_history('^GSPC', period="1y")
                if not market.empty:
                    # Calculate returns
                    stock_returns = hist['Close'][-252:].pct_change().dropna()
                    market_returns = market['Close'][-252:].pct_change().dropna()
                    
                    # Align dates
                    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
                    
                    if aligned.shape[0] > 30:  # Need sufficient data
                        # Calculate beta
                        covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
                        variance = aligned.iloc[:, 1].var()
                        beta = covariance / variance if variance != 0 else 'N/A'
                        metrics['Beta'] = round(beta, 2)
            except Exception as e:
                logger.debug(f"Error calculating beta for {ticker}: {e}")
        
        # Add metrics from financial statements
        income_stmt = financials.get('income_statement', pd.DataFrame())
        balance_sheet = financials.get('balance_sheet', pd.DataFrame())
        cash_flow = financials.get('cash_flow', pd.DataFrame())
        
        if not income_stmt.empty:
            # Get latest fiscal year data
            latest_year = income_stmt.columns[0]
            
            # Profit margins
            revenue = income_stmt.loc['Total Revenue', latest_year] if 'Total Revenue' in income_stmt.index else None
            net_income = income_stmt.loc['Net Income', latest_year] if 'Net Income' in income_stmt.index else None
            
            if revenue and net_income and revenue > 0:
                metrics['Profit Margin (%)'] = round((net_income / revenue) * 100, 2)
        
        if not balance_sheet.empty:
            # Get latest fiscal year data
            latest_year = balance_sheet.columns[0]
            
            # Current ratio
            current_assets = balance_sheet.loc['Total Current Assets', latest_year] if 'Total Current Assets' in balance_sheet.index else None
            current_liabilities = balance_sheet.loc['Total Current Liabilities', latest_year] if 'Total Current Liabilities' in balance_sheet.index else None
            
            if current_assets and current_liabilities and current_liabilities > 0:
                metrics['Current Ratio'] = round(current_assets / current_liabilities, 2)
        
        if not cash_flow.empty:
            # Get latest fiscal year data
            latest_year = cash_flow.columns[0]
            
            # Operating cash flow
            operating_cash_flow = cash_flow.loc['Total Cash From Operating Activities', latest_year] if 'Total Cash From Operating Activities' in cash_flow.index else None
            
            if operating_cash_flow and metrics['Market Cap'] != 'N/A' and metrics['Market Cap'] > 0:
                metrics['Price to Cash Flow'] = round(metrics['Market Cap'] / operating_cash_flow, 2)
        
        # Calculate dividend growth if possible
        if not hist.empty and 'Dividends' in hist.columns:
            dividends = hist['Dividends']
            annual_dividends = dividends.resample('Y').sum()
            
            if len(annual_dividends) >= 5 and annual_dividends.iloc[0] > 0:
                try:
                    cagr = (annual_dividends.iloc[-1] / annual_dividends.iloc[0]) ** (1 / 5) - 1
                    metrics['5-Year Dividend CAGR (%)'] = round(cagr * 100, 2)
                except Exception as e:
                    logger.debug(f"Error calculating dividend growth for {ticker}: {e}")
        
        return metrics
    
    def _assess_value(self, metrics):
        """
        Assess stock value based on metrics.
        
        Args:
            metrics (dict): Stock metrics
            
        Returns:
            dict: Updated metrics with value assessment
        """
        # Calculate value score if we have the required metrics
        required_metrics = ['P/E Ratio', 'P/B Ratio', 'Dividend Yield (%)', 'ROE (%)', 'Debt to Equity']
        
        if all(metrics[k] != 'N/A' for k in required_metrics):
            try:
                # Extract metrics
                pe = float(metrics['P/E Ratio'])
                pb = float(metrics['P/B Ratio'])
                div_yield = float(metrics['Dividend Yield (%)'])
                roe = float(metrics['ROE (%)'])
                debt_to_equity = float(metrics['Debt to Equity'])
                
                # Normalize metrics
                norm_pe = 1 - min(pe / 40, 1) if pe > 0 else 0  # Lower is better
                norm_pb = 1 - min(pb / 5, 1) if pb > 0 else 0  # Lower is better
                norm_div = min(div_yield / 10, 1)  # Higher is better
                norm_roe = min(roe / 30, 1)  # Higher is better
                norm_debt = 1 - min(debt_to_equity / 3, 1)  # Lower is better
                
                # Apply weights
                weighted_pe = norm_pe * abs(self.value_score_weights.get('pe_ratio', 0))
                weighted_pb = norm_pb * abs(self.value_score_weights.get('pb_ratio', 0))
                weighted_div = norm_div * self.value_score_weights.get('dividend_yield', 0)
                weighted_roe = norm_roe * self.value_score_weights.get('roe', 0)
                weighted_debt = norm_debt * abs(self.value_score_weights.get('debt_to_equity', 0))
                
                # Calculate value score (0-100)
                value_score = (weighted_pe + weighted_pb + weighted_div + weighted_roe + weighted_debt) * 100
                
                metrics['Value Score (0-100)'] = round(min(value_score, 100), 2)
                
                # Value assessment
                if value_score >= 80:
                    metrics['Value Assessment'] = "Excellent Value"
                elif value_score >= 65:
                    metrics['Value Assessment'] = "Good Value"
                elif value_score >= 50:
                    metrics['Value Assessment'] = "Fair Value"
                elif value_score >= 35:
                    metrics['Value Assessment'] = "Moderate Value"
                else:
                    metrics['Value Assessment'] = "Poor Value"
                    
            except Exception as e:
                logger.debug(f"Error calculating value score: {e}")
        
        return metrics
    
    def compare_stocks(self, tickers):
        """
        Compare multiple stocks based on key metrics.
        
        Args:
            tickers (list): List of stock ticker symbols
            
        Returns:
            pandas.DataFrame: DataFrame with comparison data
        """
        logger.info(f"Comparing stocks: {tickers}")
        
        # Get data for all tickers
        stock_data = self.data_provider.get_stock_data(tickers)
        
        if stock_data.empty:
            logger.error("No data available for comparison")
            return pd.DataFrame()
        
        # Calculate value score
        stock_data['value_score'] = self._calculate_value_score(stock_data)
        
        # Select key metrics for comparison
        comparison_metrics = [
            'ticker', 'name', 'sector', 'country', 'price', 
            'pe_ratio', 'pb_ratio', 'dividend_yield', 'roe', 
            'debt_to_equity', 'value_score'
        ]
        
        # Ensure all columns are in the DataFrame
        available_metrics = [col for col in comparison_metrics if col in stock_data.columns]
        
        # Sort by value score
        comparison = stock_data[available_metrics].sort_values('value_score', ascending=False)
        
        return comparison
    
    def _calculate_value_score(self, df):
        """
        Calculate value score for multiple stocks.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            
        Returns:
            pandas.Series: Series with value scores
        """
        # Normalize metrics to 0-1 range
        norm_pe = 1 - np.clip(df['pe_ratio'] / 40, 0, 1)  # Lower is better
        norm_pb = 1 - np.clip(df['pb_ratio'] / 5, 0, 1)   # Lower is better
        norm_div = np.clip(df['dividend_yield'] / 10, 0, 1)  # Higher is better
        norm_roe = np.clip(df['roe'] / 30, 0, 1)          # Higher is better
        norm_debt = 1 - np.clip(df['debt_to_equity'] / 3, 0, 1)  # Lower is better
        
        # Apply weights from configuration
        weighted_pe = norm_pe * abs(self.value_score_weights.get('pe_ratio', 0))
        weighted_pb = norm_pb * abs(self.value_score_weights.get('pb_ratio', 0))
        weighted_div = norm_div * self.value_score_weights.get('dividend_yield', 0)
        weighted_roe = norm_roe * self.value_score_weights.get('roe', 0)
        weighted_debt = norm_debt * abs(self.value_score_weights.get('debt_to_equity', 0))
        
        # Calculate the total score (0-100 scale)
        value_score = (weighted_pe + weighted_pb + weighted_div + weighted_roe + weighted_debt) * 100
        
        return value_score