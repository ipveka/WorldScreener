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
import yaml

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
    
    def __init__(self, config=None):
        """
        Initialize the data provider.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataProvider initialized")
        self.config = config
        self.data_cache = DataCache()
        self.tickers = self._load_tickers_from_yaml()
    
    def _load_tickers_from_yaml(self):
        """Load tickers from the tickers.yaml file."""
        tickers_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'tickers.yaml')
        try:
            with open(tickers_path, 'r') as file:
                tickers_data = yaml.safe_load(file)
                self.logger.info(f"Loaded tickers from {tickers_path}")
                return tickers_data
        except Exception as e:
            self.logger.error(f"Error loading tickers from {tickers_path}: {e}")
            # Return empty default structure if file can't be loaded
            return {"sp500": [], "eurostoxx": [], "nikkei": []}
    
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
        try:
            # Normalize index to lowercase
            index = index.lower()
            
            # Create cache key
            cache_key = f"constituents_{index}"
            
            # Check cache first
            if self.data_cache.is_cached(cache_key):
                return self.data_cache.get_from_cache(cache_key)
            
            # For predefined regions, use the fetch_real_index_constituents method
            if index in ['us', 'europe', 'japan', 'global']:
                constituents = self.fetch_real_index_constituents(index)
                if constituents:
                    logger.info(f"Using {len(constituents)} real stocks for {index}")
                    self.data_cache.add_to_cache(cache_key, constituents)
                    return constituents
            
            # Get market indices from config as fallback
            market_indices = self.get_market_indices(config)
            
            # Check if index exists in config
            if index in market_indices:
                # Get constituents from config
                constituents = market_indices[index]
                logger.info(f"Using {len(constituents)} stocks from config for {index}")
                self.data_cache.add_to_cache(cache_key, constituents)
                return constituents
            else:
                logger.error(f"Unknown index: {index}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting index constituents: {e}")
            return []
    
    def fetch_real_index_constituents(self, region):
        """Fetch real index constituents for a given region."""
        self.logger.info(f"Fetching constituents for region: {region}")
        
        if region.lower() == 'us':
            tickers = self.tickers.get('sp500', [])
            self.logger.info(f"Using {len(tickers)} real stocks for {region} from tickers.yaml")
            return tickers
        
        elif region.lower() == 'europe':
            tickers = self.tickers.get('eurostoxx', [])
            self.logger.info(f"Using {len(tickers)} real stocks for {region} from tickers.yaml")
            return tickers
        
        elif region.lower() == 'japan':
            tickers = self.tickers.get('nikkei', [])
            self.logger.info(f"Using {len(tickers)} real stocks for {region} from tickers.yaml")
            return tickers
        
        else:
            self.logger.warning(f"Unknown region: {region}. Returning empty list.")
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
                # Create a Ticker object
                yf_ticker = yf.Ticker(ticker)
                
                # Get basic info
                info = {}
                
                # Try to get info from Yahoo Finance
                try:
                    # Get ticker info
                    ticker_info = yf_ticker.info
                    
                    # Extract relevant data
                    info = {
                        'ticker': ticker,
                        'name': ticker_info.get('shortName', ticker_info.get('longName', ticker)),
                        'sector': ticker_info.get('sector', 'Unknown'),
                        'industry': ticker_info.get('industry', 'Unknown'),
                        'country': ticker_info.get('country', self._get_country_from_ticker(ticker)),
                        'market_cap': self._convert_to_usd(ticker_info.get('marketCap', 0), ticker),
                        'pe_ratio': ticker_info.get('trailingPE', 0),
                        'forward_pe': ticker_info.get('forwardPE', 0),
                        'price_to_sales': ticker_info.get('priceToSalesTrailing12Months', 0),
                        'price_to_book': ticker_info.get('priceToBook', 0),
                        'pb_ratio': ticker_info.get('priceToBook', 0),
                        'dividend_yield': ticker_info.get('dividendYield', 0) * 100 if ticker_info.get('dividendYield') else 0,  # Convert to percentage
                        'eps': ticker_info.get('trailingEps', 0),
                        'beta': ticker_info.get('beta', 0),
                        'fifty_two_week_high': ticker_info.get('fiftyTwoWeekHigh', 0),
                        'fifty_two_week_low': ticker_info.get('fiftyTwoWeekLow', 0),
                        'current_price': ticker_info.get('currentPrice', ticker_info.get('regularMarketPrice', 0)),
                        'target_price': ticker_info.get('targetMeanPrice', 0),
                        'recommendation': ticker_info.get('recommendationKey', 'Unknown'),
                        'roe': ticker_info.get('returnOnEquity', 0) * 100 if ticker_info.get('returnOnEquity') else 0,
                        'roa': ticker_info.get('returnOnAssets', 0) * 100 if ticker_info.get('returnOnAssets') else 0,
                        'debt_to_equity': ticker_info.get('debtToEquity', 0) / 100 if ticker_info.get('debtToEquity') else 0,
                        'quick_ratio': ticker_info.get('quickRatio', 0),
                        'current_ratio': ticker_info.get('currentRatio', 0),
                        'peg_ratio': ticker_info.get('pegRatio', 0),
                        'short_ratio': ticker_info.get('shortRatio', 0),
                        'earnings_growth': ticker_info.get('earningsQuarterlyGrowth', 0) * 100 if ticker_info.get('earningsQuarterlyGrowth') else 0,
                        'revenue_growth': ticker_info.get('revenueGrowth', 0) * 100 if ticker_info.get('revenueGrowth') else 0,
                        'gross_margins': ticker_info.get('grossMargins', 0) * 100 if ticker_info.get('grossMargins') else 0,
                        'ebitda_margins': ticker_info.get('ebitdaMargins', 0) * 100 if ticker_info.get('ebitdaMargins') else 0,
                        'profit_margins': ticker_info.get('profitMargins', 0) * 100 if ticker_info.get('profitMargins') else 0
                    }
                    
                    # Add to results
                    results.append(info)
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    # Skip this ticker
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
                
        return results
        
    def _convert_to_usd(self, market_cap, ticker):
        """
        Convert market cap to USD for non-US companies.
        
        Args:
            market_cap (float): Market cap in local currency
            ticker (str): Stock ticker symbol
            
        Returns:
            float: Market cap in USD
        """
        try:
            # If market cap is 0 or None, return 0
            if not market_cap:
                return 0
                
            # For US stocks, no conversion needed
            if '.' not in ticker:
                return market_cap
                
            # Get currency based on ticker suffix
            currency = self._get_currency_from_ticker(ticker)
            
            # If currency is USD, no conversion needed
            if currency == 'USD':
                return market_cap
                
            # Try to get exchange rate from Yahoo Finance
            try:
                # Create currency pair ticker (e.g., EURJPY=X)
                currency_pair = f"{currency}USD=X"
                currency_ticker = yf.Ticker(currency_pair)
                
                # Get current exchange rate
                exchange_rate = currency_ticker.history(period='1d')['Close'].iloc[-1]
                
                # Convert market cap to USD
                return market_cap * exchange_rate
            except Exception as e:
                logger.warning(f"Error getting exchange rate for {currency}: {e}")
                
                # Use approximate exchange rates as fallback
                exchange_rates = {
                    'EUR': 1.10,  # Euro to USD
                    'GBP': 1.30,  # British Pound to USD
                    'JPY': 0.0067,  # Japanese Yen to USD
                    'CHF': 1.12,  # Swiss Franc to USD
                    'SEK': 0.096,  # Swedish Krona to USD
                    'NOK': 0.094,  # Norwegian Krone to USD
                    'DKK': 0.15,  # Danish Krone to USD
                    'AUD': 0.67,  # Australian Dollar to USD
                    'CAD': 0.74,  # Canadian Dollar to USD
                    'HKD': 0.13,  # Hong Kong Dollar to USD
                    'CNY': 0.14,  # Chinese Yuan to USD
                    'KRW': 0.00075  # South Korean Won to USD
                }
                
                # If currency is in the exchange rates dictionary, convert market cap
                if currency in exchange_rates:
                    return market_cap * exchange_rates[currency]
                else:
                    # If currency is not in the dictionary, return market cap as is
                    return market_cap
                
        except Exception as e:
            logger.error(f"Error converting market cap to USD: {e}")
            return market_cap
            
    def _get_currency_from_ticker(self, ticker):
        """
        Get currency from ticker symbol based on suffix.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            str: Currency code
        """
        if ticker.endswith('.T'):
            return 'JPY'  # Japanese Yen
        elif ticker.endswith('.L'):
            return 'GBP'  # British Pound
        elif ticker.endswith('.DE') or ticker.endswith('.F') or ticker.endswith('.PA') or ticker.endswith('.MI') or ticker.endswith('.AS'):
            return 'EUR'  # Euro
        elif ticker.endswith('.SW'):
            return 'CHF'  # Swiss Franc
        elif ticker.endswith('.ST'):
            return 'SEK'  # Swedish Krona
        elif ticker.endswith('.OL'):
            return 'NOK'  # Norwegian Krone
        elif ticker.endswith('.CO'):
            return 'DKK'  # Danish Krone
        elif ticker.endswith('.AX'):
            return 'AUD'  # Australian Dollar
        elif ticker.endswith('.TO'):
            return 'CAD'  # Canadian Dollar
        elif ticker.endswith('.HK'):
            return 'HKD'  # Hong Kong Dollar
        elif ticker.endswith('.SS') or ticker.endswith('.SZ'):
            return 'CNY'  # Chinese Yuan
        elif ticker.endswith('.KS'):
            return 'KRW'  # South Korean Won
        else:
            return 'USD'  # US Dollar
            
    def _get_country_from_ticker(self, ticker):
        """
        Extract country from ticker symbol based on suffix.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            str: Country name
        """
        if ticker.endswith('.MC') or ticker.endswith('.MA'):
            return 'Spain'
        elif ticker.endswith('.DE') or ticker.endswith('.F'):
            return 'Germany'
        elif ticker.endswith('.PA'):
            return 'France'
        elif ticker.endswith('.MI'):
            return 'Italy'
        elif ticker.endswith('.L'):
            return 'United Kingdom'
        elif ticker.endswith('.AS'):
            return 'Netherlands'
        elif ticker.endswith('.SW'):
            return 'Switzerland'
        elif ticker.endswith('.CO'):
            return 'Denmark'
        elif ticker.endswith('.ST'):
            return 'Sweden'
        elif ticker.endswith('.HE'):
            return 'Finland'
        elif ticker.endswith('.OL'):
            return 'Norway'
        elif ticker.endswith('.T'):
            return 'Japan'
        elif ticker.endswith('.KS'):
            return 'South Korea'
        elif ticker.endswith('.SS') or ticker.endswith('.SZ'):
            return 'China'
        elif ticker.endswith('.HK'):
            return 'Hong Kong'
        elif '.' not in ticker:
            return 'United States'
        else:
            return 'Unknown'
    
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
        if self.data_cache.is_cached(cache_key):
            return self.data_cache.get_from_cache(cache_key)
        
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period)
            
            if history.empty:
                logger.warning(f"No historical data found for {ticker}")
                return pd.DataFrame()
                
            # Cache the results
            self.data_cache.add_to_cache(cache_key, history)
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
        if self.data_cache.is_cached(cache_key):
            return self.data_cache.get_from_cache(cache_key)
        
        try:
            stock = yf.Ticker(ticker)
            
            financials = {
                'income_statement': stock.income_stmt,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow
            }
            
            # Cache the results
            self.data_cache.add_to_cache(cache_key, financials)
            return financials
            
        except Exception as e:
            logger.error(f"Error getting financials for {ticker}: {e}")
            return {
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame()
            }