#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Screener module for WorldScreener.

This module contains the main screening logic for filtering stocks.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class Screener:
    """
    Stock screener for filtering stocks based on value criteria.
    """
    
    def __init__(self, data_provider, config):
        """
        Initialize the Screener.
        
        Args:
            data_provider: DataProvider instance for retrieving stock data
            config (dict): Configuration dictionary
        """
        self.data_provider = data_provider
        self.config = config
        
        # Get default criteria from config
        self.default_criteria = config['screening']['default_criteria']
        
        # Weight factors for value score calculation
        self.value_score_weights = config['screening']['value_score_weights']
        
        logger.info("Screener initialized")
    
    def screen_stocks(self, region='europe', criteria=None, limit=500, include_sectors=None, exclude_sectors=None):
        """
        Screen stocks according to value investing criteria.
        
        Args:
            region (str): Region to screen ('spain', 'europe', 'eurozone', 'us', 'global')
            criteria (dict, optional): Screening criteria to override defaults
            limit (int): Maximum number of results to return
            include_sectors (list, optional): List of sectors to include in screening
            exclude_sectors (list, optional): List of sectors to exclude from screening
            
        Returns:
            pandas.DataFrame: DataFrame with screened stock data
        """
        logger.info(f"Screening stocks in {region} region...")
        
        # Merge custom criteria with defaults
        screen_criteria = self.default_criteria.copy()
        if criteria:
            screen_criteria.update(criteria)
        
        # Get constituents for the specified region
        # For Japan, directly use fetch_real_index_constituents
        if region.lower() == 'japan':
            constituents = self.data_provider.fetch_real_index_constituents('japan')
        else:
            constituents = self.data_provider.get_index_constituents(region, self.config)
            
        if not constituents:
            logger.error(f"No constituents found for region: {region}")
            return pd.DataFrame()
        
        # Get stock data
        logger.info(f"Found {len(constituents)} stocks to screen")
        stock_data = self.data_provider.get_stock_data(constituents)
        
        if stock_data.empty:
            logger.error("No stock data available")
            return pd.DataFrame()
        
        # Calculate value score
        stock_data['value_score'] = self._calculate_value_score(stock_data)
        
        # Apply screening criteria
        screened_stocks = self._apply_criteria(stock_data, screen_criteria)
        
        # Apply sector filtering
        if include_sectors or exclude_sectors:
            screened_stocks = self._filter_by_sector(screened_stocks, include_sectors, exclude_sectors)
        
        # Sort by value score (descending)
        sorted_stocks = screened_stocks.sort_values(by='value_score', ascending=False)
        
        # Limit the number of results
        top_stocks = sorted_stocks.head(limit)
        
        logger.info(f"Screening complete. Found {len(top_stocks)} stocks meeting criteria.")
        
        return top_stocks
    
    def _calculate_value_score(self, df):
        """
        Calculate a composite value score based on key value metrics.
        Higher score indicates better value.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock metrics
            
        Returns:
            pandas.Series: Series of value scores
        """
        # Normalize metrics to 0-1 range
        norm_pe = 1 - np.clip(df['pe_ratio'] / 40, 0, 1)  # Lower is better
        norm_pb = 1 - np.clip(df['pb_ratio'] / 5, 0, 1)   # Lower is better
        norm_div = np.clip(df['dividend_yield'] / 10, 0, 1)  # Higher is better
        norm_roe = np.clip(df['roe'] / 30, 0, 1)          # Higher is better
        norm_debt = 1 - np.clip(df['debt_to_equity'] / 3, 0, 1)  # Lower is better
        
        # Apply weights from configuration
        weighted_pe = norm_pe * abs(self.value_score_weights['pe_ratio'])
        weighted_pb = norm_pb * abs(self.value_score_weights['pb_ratio'])
        weighted_div = norm_div * self.value_score_weights['dividend_yield']
        weighted_roe = norm_roe * self.value_score_weights['roe']
        weighted_debt = norm_debt * abs(self.value_score_weights['debt_to_equity'])
        
        # Calculate the total score (0-100 scale)
        value_score = (weighted_pe + weighted_pb + weighted_div + weighted_roe + weighted_debt) * 100
        
        return value_score
    
    def _apply_criteria(self, stock_data, criteria):
        """
        Apply screening criteria to stock data.
        
        Args:
            stock_data (pandas.DataFrame): DataFrame with stock data
            criteria (dict): Screening criteria
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        # Apply filters
        filtered_data = stock_data[
            (stock_data['pe_ratio'] > 0) &  # Positive P/E
            (stock_data['pe_ratio'] <= criteria['max_pe_ratio']) &
            (stock_data['dividend_yield'] >= criteria['min_dividend_yield']) &
            (stock_data['pb_ratio'] > 0) &  # Positive P/B
            (stock_data['pb_ratio'] <= criteria['max_pb_ratio']) &
            (stock_data['roe'] >= criteria['min_roe']) &
            (stock_data['debt_to_equity'] <= criteria['max_debt_to_equity']) &
            (stock_data['market_cap'] >= criteria['min_market_cap'])
        ]
        
        # Add value score filter if specified
        if 'min_value_score' in criteria:
            filtered_data = filtered_data[
                filtered_data['value_score'] >= criteria['min_value_score']
            ]
        
        return filtered_data
    
    def _filter_by_sector(self, stocks_df, include_sectors=None, exclude_sectors=None):
        """
        Filter stocks by sector.
        
        Args:
            stocks_df (pandas.DataFrame): DataFrame of stocks
            include_sectors (list, optional): List of sectors to include
            exclude_sectors (list, optional): List of sectors to exclude
            
        Returns:
            pandas.DataFrame: Filtered stocks
        """
        if stocks_df.empty or 'sector' not in stocks_df.columns:
            return stocks_df
        
        # Make a copy to avoid modifying the original
        filtered_df = stocks_df.copy()
        
        # Apply sector inclusion filter
        if include_sectors:
            include_sectors = [s.lower() for s in include_sectors]
            filtered_df = filtered_df[filtered_df['sector'].str.lower().isin(include_sectors)]
            logger.info(f"Applied sector inclusion filter. {len(filtered_df)} stocks remaining.")
        
        # Apply sector exclusion filter
        if exclude_sectors:
            exclude_sectors = [s.lower() for s in exclude_sectors]
            filtered_df = filtered_df[~filtered_df['sector'].str.lower().isin(exclude_sectors)]
            logger.info(f"Applied sector exclusion filter. {len(filtered_df)} stocks remaining.")
        
        return filtered_df
    
    def filter_by_sector(self, df, sectors):
        """
        Filter stocks by sector.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            sectors (list): List of sectors to include
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        if not sectors:
            return df
        
        return df[df['sector'].isin(sectors)]
    
    def filter_by_country(self, df, countries):
        """
        Filter stocks by country.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            countries (list): List of countries to include
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        if not countries:
            return df
        
        return df[df['country'].isin(countries)]
    
    def get_top_value_stocks(self, region='europe', limit=10):
        """
        Get top value stocks with default criteria.
        
        Args:
            region (str): Region to screen
            limit (int): Maximum number of results
            
        Returns:
            pandas.DataFrame: DataFrame with top value stocks
        """
        return self.screen_stocks(region=region, limit=limit)
    
    def get_high_dividend_stocks(self, region='europe', limit=10):
        """
        Get high dividend stocks.
        
        Args:
            region (str): Region to screen
            limit (int): Maximum number of results
            
        Returns:
            pandas.DataFrame: DataFrame with high dividend stocks
        """
        criteria = self.default_criteria.copy()
        criteria['min_dividend_yield'] = 5.0  # Higher dividend requirement
        
        return self.screen_stocks(region=region, criteria=criteria, limit=limit)
    
    def get_low_pe_stocks(self, region='europe', limit=10):
        """
        Get low P/E stocks.
        
        Args:
            region (str): Region to screen
            limit (int): Maximum number of results
            
        Returns:
            pandas.DataFrame: DataFrame with low P/E stocks
        """
        criteria = self.default_criteria.copy()
        criteria['max_pe_ratio'] = 10.0  # Lower P/E requirement
        
        return self.screen_stocks(region=region, criteria=criteria, limit=limit)