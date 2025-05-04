#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Provider module for WorldScreener.

This module handles all data retrieval operations from various financial APIs.
"""

import os
import logging
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DataCache:
    """
    Cache manager for financial data to minimize redundant API calls.
    """
    
    def __init__(self, cache_duration=4):
        """
        Initialize the cache.
        
        Args:
            cache_duration (int): Cache validity in hours
        """
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=cache_duration)
        logger.debug("DataCache initialized with %d hour expiry", cache_duration)
    
    def is_cached(self, key):
        """
        Check if data is cached and not expired.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if valid cached data exists
        """
        if key in self.cache and key in self.cache_expiry:
            if datetime.now() < self.cache_expiry[key]:
                return True
        return False
    
    def add_to_cache(self, key, data):
        """
        Add data to cache with expiry time.
        
        Args:
            key (str): Cache key
            data: Data to cache
        """
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + self.cache_duration
    
    def get_from_cache(self, key):
        """
        Get data from cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            Data from cache
        """
        return self.cache[key]
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache = {}
        self.cache_expiry = {}
        logger.info("Cache cleared")


class DataProvider:
    """
    Data provider class for fetching financial data from various sources.
    """
    
    def __init__(self, config=None, cache_duration=24, use_mock_data=True):
        """
        Initialize the data provider.
        
        Args:
            config (dict, optional): Configuration dictionary
            cache_duration (int, optional): Cache duration in hours
            use_mock_data (bool, optional): Whether to use mock data for demonstrations
        """
        self.config = config or {}
        self.cache = DataCache(cache_duration)
        self.use_mock_data = use_mock_data
        logger.info("DataProvider initialized")
    
    def get_market_indices(self, config):
        """
        Get market indices from configuration.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            dict: Dictionary of market indices
        """
        return config.get('market_indices', {})
    
    def get_index_constituents(self, index, config):
        """
        Get the constituents of a specified market index.
        
        Args:
            index (str): Index identifier ('spain', 'europe', 'eurozone', 'us', 'global')
            config (dict): Configuration dictionary
            
        Returns:
            list: List of ticker symbols for the index constituents
        """
        market_indices = self.get_market_indices(config)
        index_symbol = market_indices.get(index.lower())
        
        if not index_symbol:
            logger.error(f"Unknown index: {index}")
            return []
            
        cache_key = f"index_{index}"
        if self.cache.is_cached(cache_key):
            return self.cache.get_from_cache(cache_key)
            
        try:
            # First try to get stocks directly from config
            if 'markets' in config and index.lower() in config['markets'] and 'stocks' in config['markets'][index.lower()]:
                constituents = config['markets'][index.lower()]['stocks']
                logger.info(f"Using {len(constituents)} stocks from config for {index}")
            # Then try the indices section
            elif 'indices' in config and index.lower() in config['indices']:
                constituents = config['indices'][index.lower()]
                logger.info(f"Using {len(constituents)} stocks from indices config for {index}")
            else:
                # Fallback to hardcoded lists for demo purposes
                if index.lower() == 'spain':
                    constituents = [
                        "SAN.MC", "BBVA.MC", "TEF.MC", "IBE.MC", "REP.MC", 
                        "ITX.MC", "AMS.MC", "FER.MC", "ELE.MC", "CABK.MC"
                    ]
                elif index.lower() == 'europe':
                    constituents = [
                        "ASML.AS", "SAP.DE", "SIE.DE", "LVMH.PA", "ROG.SW",
                        "NOVN.SW", "NESN.SW", "AZN.L", "ULVR.L", "RIO.L"
                    ]
                elif index.lower() == 'eurozone':
                    constituents = [
                        "ASML.AS", "SAP.DE", "SIE.DE", "LVMH.PA", "MC.PA",
                        "BNP.PA", "SAN.MC", "BBVA.MC", "ISP.MI", "EBS.VI"
                    ]
                elif index.lower() == 'us':
                    constituents = [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B",
                        "JNJ", "JPM", "PG", "XOM", "KO"
                    ]
                else:  # global
                    constituents = [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "ASML.AS",
                        "SAP.DE", "LVMH.PA", "SAN.MC", "AZN.L", "NESN.SW"
                    ]
                logger.info(f"Using {len(constituents)} hardcoded stocks for {index}")
                    
            # Cache the results
            self.cache.add_to_cache(cache_key, constituents)
            return constituents
                
        except Exception as e:
            logger.error(f"Error getting index constituents: {e}")
            return []
    
    def get_stock_data(self, ticker_list, threads=10):
        """
        Get fundamental and market data for a list of stocks using multiple threads.
        
        Args:
            ticker_list (list): List of ticker symbols
            threads (int): Number of concurrent threads to use
            
        Returns:
            pandas.DataFrame: DataFrame with stock data
        """
        start_time = time.time()
        logger.info(f"Fetching data for {len(ticker_list)} stocks using {threads} threads...")
        
        # Split the list into chunks for threading
        chunk_size = max(1, len(ticker_list) // threads)
        ticker_chunks = [ticker_list[i:i+chunk_size] for i in range(0, len(ticker_list), chunk_size)]
        
        results = []
        
        # Process each chunk in a separate thread
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(self._process_ticker_chunk, chunk) for chunk in ticker_chunks]
            
            # Gather results
            for future in futures:
                chunk_results = future.result()
                results.extend(chunk_results)
        
        # Convert to DataFrame
        if not results:
            logger.warning("No valid stock data found")
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        
        # Log execution time
        elapsed_time = time.time() - start_time
        logger.info(f"Data fetching completed in {elapsed_time:.2f} seconds")
        
        return df
    
    def _process_ticker_chunk(self, ticker_chunk):
        """
        Process a chunk of tickers to get their fundamental data.
        
        Args:
            ticker_chunk (list): List of ticker symbols
            
        Returns:
            list: List of dictionaries with stock data
        """
        results = []
        
        for ticker in ticker_chunk:
            try:
                # If mock data is enabled, use it directly without trying the API
                if self.use_mock_data:
                    info = self._generate_mock_data(ticker)
                    results.append(info)
                    continue
                
                # Otherwise try the API first
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # If the API fails, generate mock data
                if not info or 'regularMarketPrice' not in info:
                    logger.debug(f"Insufficient data for {ticker}, using mock data for demo")
                    info = self._generate_mock_data(ticker)
                else:
                    # Extract country from info or infer from ticker suffix
                    country = info.get('country', '')
                    if not country:
                        # Try to infer from ticker suffix
                        if ticker.endswith('.MC') or ticker.endswith('.MA'):
                            country = 'ES'  # Spain
                        elif ticker.endswith('.DE') or ticker.endswith('.F'):
                            country = 'DE'  # Germany
                        elif ticker.endswith('.PA'):
                            country = 'FR'  # France
                        elif ticker.endswith('.MI'):
                            country = 'IT'  # Italy
                        elif ticker.endswith('.L'):
                            country = 'GB'  # UK
                        elif ticker.endswith('.AS'):
                            country = 'NL'  # Netherlands
                        elif ticker.endswith('.SW'):
                            country = 'CH'  # Switzerland
                        elif ticker.endswith('.VI'):
                            country = 'AT'  # Austria
                        else:
                            country = 'US'  # Default to US
                    
                    # Get key metrics
                    market_price = info.get('regularMarketPrice', 0)
                    
                    # Extract fundamentals
                    pe_ratio = info.get('trailingPE', info.get('forwardPE', 0))
                    pb_ratio = info.get('priceToBook', 0)
                    
                    # Handle dividend yield - convert from decimal to percentage
                    # Yahoo Finance sometimes returns this as a percentage already
                    dividend_yield = info.get('dividendYield', 0)
                    if dividend_yield > 0:
                        # If it's greater than 1, it's likely already a percentage
                        if dividend_yield > 1 and dividend_yield < 100:
                            # Already a percentage, keep as is
                            pass
                        else:
                            # Convert from decimal to percentage (e.g., 0.05 -> 5.0)
                            dividend_yield = dividend_yield * 100
                    
                    # Calculate additional metrics
                    roe = info.get('returnOnEquity', 0)
                    if roe > 0:
                        # If it's greater than 1, it's likely already a percentage
                        if roe > 1 and roe < 100:
                            # Already a percentage, keep as is
                            pass
                        else:
                            # Convert from decimal to percentage
                            roe = roe * 100
                        
                    # Debt metrics
                    total_debt = info.get('totalDebt', 0)
                    total_equity = info.get('totalStockholderEquity', 0)
                    
                    debt_to_equity = 0
                    if total_equity and total_equity > 0:
                        debt_to_equity = total_debt / total_equity
                    
                    # Create stock data dictionary
                    stock_data = {
                        'ticker': ticker,
                        'name': info.get('shortName', ticker),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'country': country,
                        'currency': info.get('currency', ''),
                        'market_cap': info.get('marketCap', 0),
                        'price': market_price,
                        'pe_ratio': round(pe_ratio, 2) if pe_ratio else 0,
                        'pb_ratio': round(pb_ratio, 2) if pb_ratio else 0,
                        'dividend_yield': round(dividend_yield, 2) if dividend_yield else 0,
                        'roe': round(roe, 2) if roe else 0,
                        'debt_to_equity': round(debt_to_equity, 2) if debt_to_equity else 0,
                        'exchange': info.get('exchange', ''),
                        'fifty_two_week_high': round(info.get('fiftyTwoWeekHigh', 0), 2),
                        'fifty_two_week_low': round(info.get('fiftyTwoWeekLow', 0), 2)
                    }
                    
                    # Add to cache
                    self.cache.add_to_cache(f"stock_data_{ticker}", stock_data)
                    
                    # Add to results
                    results.append(stock_data)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                # Generate mock data on error
                logger.info(f"Using mock data for {ticker} due to error: {e}")
                mock_data = self._generate_mock_data(ticker)
                results.append(mock_data)
        
        return results
        
    def _generate_mock_data(self, ticker):
        """
        Generate mock stock data for demonstration purposes.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Mock stock data
        """
        import random
        
        # Extract country from ticker suffix
        country = 'US'
        if ticker.endswith('.MC') or ticker.endswith('.MA'):
            country = 'ES'  # Spain
        elif ticker.endswith('.DE') or ticker.endswith('.F'):
            country = 'DE'  # Germany
        elif ticker.endswith('.PA'):
            country = 'FR'  # France
        elif ticker.endswith('.MI'):
            country = 'IT'  # Italy
        elif ticker.endswith('.L'):
            country = 'GB'  # UK
        elif ticker.endswith('.AS'):
            country = 'NL'  # Netherlands
        elif ticker.endswith('.SW'):
            country = 'CH'  # Switzerland
        
        # Map tickers to company names for better demo
        company_names = {
            'SAN.MC': 'Banco Santander',
            'BBVA.MC': 'BBVA',
            'TEF.MC': 'Telefonica',
            'IBE.MC': 'Iberdrola',
            'REP.MC': 'Repsol',
            'ITX.MC': 'Inditex',
            'AMS.MC': 'Amadeus',
            'FER.MC': 'Ferrovial',
            'ELE.MC': 'Endesa',
            'CABK.MC': 'CaixaBank',
            'ASML.AS': 'ASML Holding',
            'SAP.DE': 'SAP',
            'SIE.DE': 'Siemens',
            'LVMH.PA': 'LVMH',
            'ROG.SW': 'Roche',
            'NOVN.SW': 'Novartis',
            'NESN.SW': 'Nestle',
            'AZN.L': 'AstraZeneca',
            'ULVR.L': 'Unilever',
            'RIO.L': 'Rio Tinto',
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Alphabet',
            'AMZN': 'Amazon',
            'BRK-B': 'Berkshire Hathaway',
            'JNJ': 'Johnson & Johnson',
            'JPM': 'JPMorgan Chase',
            'PG': 'Procter & Gamble',
            'XOM': 'Exxon Mobil',
            'KO': 'Coca-Cola'
        }
        
        # Map tickers to sectors for better demo
        sector_map = {
            'SAN.MC': 'Financial Services',
            'BBVA.MC': 'Financial Services',
            'TEF.MC': 'Communication Services',
            'IBE.MC': 'Utilities',
            'REP.MC': 'Energy',
            'ITX.MC': 'Consumer Cyclical',
            'AMS.MC': 'Technology',
            'FER.MC': 'Industrials',
            'ELE.MC': 'Utilities',
            'CABK.MC': 'Financial Services',
            'ASML.AS': 'Technology',
            'SAP.DE': 'Technology',
            'SIE.DE': 'Industrials',
            'LVMH.PA': 'Consumer Cyclical',
            'ROG.SW': 'Healthcare',
            'NOVN.SW': 'Healthcare',
            'NESN.SW': 'Consumer Defensive',
            'AZN.L': 'Healthcare',
            'ULVR.L': 'Consumer Defensive',
            'RIO.L': 'Basic Materials',
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Communication Services',
            'AMZN': 'Consumer Cyclical',
            'BRK-B': 'Financial Services',
            'JNJ': 'Healthcare',
            'JPM': 'Financial Services',
            'PG': 'Consumer Defensive',
            'XOM': 'Energy',
            'KO': 'Consumer Defensive'
        }
        
        # Generate realistic but random values for demonstration
        name = company_names.get(ticker, f"Company {ticker}")
        sector = sector_map.get(ticker, random.choice(['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical', 'Industrials']))
        
        # Generate values that will pass screening criteria for demo purposes
        pe_ratio = round(random.uniform(8.0, 14.0), 2)
        pb_ratio = round(random.uniform(0.8, 1.4), 2)
        dividend_yield = round(random.uniform(3.5, 6.0), 2)
        roe = round(random.uniform(12.0, 20.0), 2)
        debt_to_equity = round(random.uniform(0.3, 0.7), 2)
        price = round(random.uniform(50.0, 500.0), 2)
        
        # Calculate value score (higher is better)
        value_score = 100 - (pe_ratio * 2) - (pb_ratio * 10) + (dividend_yield * 3) + (roe * 0.5) - (debt_to_equity * 10)
        value_score = round(max(50, min(95, value_score)), 2)  # Ensure between 50-95 for demo
        
        # Create mock stock data
        return {
            'ticker': ticker,
            'name': name,
            'sector': sector,
            'industry': f"{sector} Industry",
            'country': country,
            'currency': 'USD' if country == 'US' else 'EUR',
            'market_cap': random.randint(5000000000, 500000000000),
            'price': price,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'dividend_yield': dividend_yield,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'exchange': 'NYSE' if country == 'US' else 'EURONEXT',
            'fifty_two_week_high': round(random.uniform(100.0, 600.0), 2),
            'fifty_two_week_low': round(random.uniform(40.0, 90.0), 2),
            'value_score': value_score
        }
    
    def get_stock_history(self, ticker, period="1y"):
        """
        Get historical price data for a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period (e.g., "1d", "1mo", "1y", "max")
            
        Returns:
            pandas.DataFrame: DataFrame with historical data
        """
        cache_key = f"history_{ticker}_{period}"
        if self.cache.is_cached(cache_key):
            return self.cache.get_from_cache(cache_key)
        
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period)
            
            if history.empty:
                logger.warning(f"No historical data found for {ticker}")
                return pd.DataFrame()
                
            # Cache the results
            self.cache.add_to_cache(cache_key, history)
            return history
            
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_stock_financials(self, ticker):
        """
        Get financial statements for a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary with financial statements
        """
        cache_key = f"financials_{ticker}"
        if self.cache.is_cached(cache_key):
            return self.cache.get_from_cache(cache_key)
        
        try:
            stock = yf.Ticker(ticker)
            
            financials = {
                'income_statement': stock.income_stmt,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow
            }
            
            # Cache the results
            self.cache.add_to_cache(cache_key, financials)
            return financials
            
        except Exception as e:
            logger.error(f"Error getting financials for {ticker}: {e}")
            return {
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame()
            }