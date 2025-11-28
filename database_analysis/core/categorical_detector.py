#!/usr/bin/env python3
"""
Categorical Detector - Auto-detection and configuration of categorical columns
"""

import pandas as pd
from typing import Dict, List, Any
from .base_analyzer import BaseAnalyzer
from .constants import (
    CONFIG_KEY_COLUMN_MAPPINGS,
    CONFIG_KEY_ANALYSIS_OPTIONS,
)


class CategoricalDetector(BaseAnalyzer):
    """Handles automatic detection and configuration of categorical columns."""
    
    def __init__(
        self,
        config_file: str,
        data_directory: str,
        output_directory: str = None,
        database_name: str = None,
        context=None,
        config_override=None,
    ):
        super().__init__(config_file, data_directory, output_directory, database_name, context=context, config_override=config_override)
    
    def auto_detect_categorical_columns(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Auto-detect categorical columns from all cleaned datasets using comprehensive global defaults."""
        categorical_columns = {}
        
        # Get comprehensive thresholds from global defaults
        thresholds = self._get_detection_thresholds()
        
        # Get skip patterns from global defaults
        skip_patterns = self.global_defaults.get('common_skip_patterns', [])
        
        # Log database-specific configuration if overrides are applied
        db_overrides = self._get_database_overrides()
        if db_overrides:
            self.logger.info(f"Applied database-specific overrides: {list(db_overrides.keys())}")
        
        for dataset_name, df in datasets.items():
            dataset_categoricals = []
            
            for column in df.columns:
                if self._should_analyze_column(column, df[column], thresholds, skip_patterns):
                    dataset_categoricals.append(column)
            
            categorical_columns[dataset_name] = dataset_categoricals
            # Only log if columns were found
            if dataset_categoricals:
                self.logger.info(f"Auto-detected {len(dataset_categoricals)} categorical columns in {dataset_name}")
        
        return categorical_columns
    
    def _get_detection_thresholds(self) -> Dict[str, Any]:
        """Get detection thresholds with database-specific overrides."""
        thresholds = self.global_defaults.get('categorical_detection_thresholds', {})
        
        # Apply database-specific overrides
        db_overrides = self._get_database_overrides()
        for key, value in db_overrides.items():
            thresholds[key] = value
        
        return thresholds
    
    def _get_database_overrides(self) -> Dict[str, Any]:
        """Get database-specific configuration overrides."""
        database_name = self.config.get('database_name', '').lower().replace('-', '_')
        return self.global_defaults.get('database_specific_overrides', {}).get(database_name, {})
    
    def _should_analyze_column(self, column: str, series: pd.Series, thresholds: Dict[str, Any], skip_patterns: List[str]) -> bool:
        """Determine if a column should be analyzed as categorical."""
        # Skip columns matching common patterns
        if any(pattern.lower() in column.lower() for pattern in skip_patterns):
            return False
        
        # Extract thresholds
        max_unique = thresholds.get('max_unique_values', 50)
        max_ratio = thresholds.get('max_uniqueness_ratio', 0.1)
        min_unique = thresholds.get('min_unique_values', 2)
        max_numeric = thresholds.get('max_numeric_categories', 20)
        max_string = thresholds.get('max_string_categories', 100)
        min_samples = thresholds.get('min_samples_per_category', 1)
        max_null_pct = thresholds.get('max_null_percentage', 0.8)
        enable_length_check = thresholds.get('enable_string_length_check', True)
        max_string_length = thresholds.get('max_string_length', 200)
        enable_pattern_match = thresholds.get('enable_pattern_matching', True)
        categorical_patterns = thresholds.get('categorical_patterns', [])
        forced_dtypes = thresholds.get('forced_categorical_dtypes', ['object', 'category', 'string'])
        
        # Continuous numeric exclusion settings
        exclude_numeric = thresholds.get('exclude_numeric_ranges', {})
        enable_numeric_exclusion = exclude_numeric.get('enable', True)
        continuous_threshold = exclude_numeric.get('continuous_threshold', 100)
        
        # Basic statistics
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return False
        
        nunique = non_null_series.nunique()
        total_count = len(series)
        uniqueness_ratio = nunique / total_count if total_count > 0 else 0
        null_percentage = series.isnull().sum() / total_count if total_count > 0 else 0
        
        # Basic filters
        if (nunique < min_unique or 
            nunique > max_unique or 
            uniqueness_ratio > max_ratio or 
            null_percentage > max_null_pct):
            return False
        
        # Check if minimum samples per category is met
        if min_samples > 1:
            value_counts = non_null_series.value_counts()
            if (value_counts < min_samples).any():
                return False
        
        # Pattern matching for column names
        if enable_pattern_match and categorical_patterns:
            if any(pattern.lower() in column.lower() for pattern in categorical_patterns):
                return True
        
        # Force include certain data types
        if str(series.dtype).lower() in [dtype.lower() for dtype in forced_dtypes]:
            return True
        
        # String-specific checks
        if series.dtype == 'object':
            if nunique > max_string:
                return False
            
            # Check string length if enabled
            if enable_length_check:
                try:
                    str_lengths = non_null_series.astype(str).str.len()
                    if str_lengths.max() > max_string_length:
                        return False
                except:
                    pass
        
        # Numeric-specific checks
        elif pd.api.types.is_numeric_dtype(series):
            if nunique > max_numeric:
                return False
            
            # Exclude likely continuous numeric ranges
            if enable_numeric_exclusion and nunique > continuous_threshold:
                return False
        
        return True
    
    def create_dynamic_column_mappings(self, categorical_columns: Dict[str, List[str]]) -> Dict[str, Any]:
        """Create column mappings for auto-detected categorical columns."""
        dynamic_mappings = {}
        
        # First, add existing configured mappings
        column_mappings = self.config.get(CONFIG_KEY_COLUMN_MAPPINGS, {})
        if column_mappings:
            dynamic_mappings.update(column_mappings)
        
        # Check if auto-detection is disabled in config
        auto_detection_enabled = self.config.get(CONFIG_KEY_ANALYSIS_OPTIONS, {}).get('enable_auto_detection', True)
        
        if not auto_detection_enabled:
            self.logger.info("Auto-detection disabled in configuration, using only manual mappings")
            return dynamic_mappings
        
        # Add auto-detected columns
        auto_detected_count = 0
        for dataset_name, columns in categorical_columns.items():
            for column in columns:
                # Create a safe mapping name
                mapping_name = self._create_safe_mapping_name(column)
                
                # Only add if not already configured
                if mapping_name not in dynamic_mappings:
                    dynamic_mappings[mapping_name] = {
                        'dataset': dataset_name,
                        'column': column,
                        'description': f'Auto-detected categorical column: {column}',
                        'use_log_scale': False
                    }
                    auto_detected_count += 1
        
        # Single summary log instead of per-column logs
        if auto_detected_count > 0:
            self.logger.info(f"Added {auto_detected_count} auto-detected columns for analysis")
        
        return dynamic_mappings
    
    def _create_safe_mapping_name(self, column: str) -> str:
        """Create a safe mapping name from column name."""
        mapping_name = column.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
        mapping_name = ''.join(c for c in mapping_name if c.isalnum() or c == '_')
        return mapping_name
    
    def get_column_analysis_summary(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get summary of how columns will be analyzed (configured vs auto-detected)."""
        # Get configured columns
        configured_columns = {}
        column_mappings = self.config.get(CONFIG_KEY_COLUMN_MAPPINGS, {})
        for name, config in column_mappings.items():
            configured_columns[name] = {
                'dataset': config['dataset'],
                'column': config['column'],
                'source': 'manual_configuration'
            }
        
        # Get auto-detected columns
        categorical_columns = self.auto_detect_categorical_columns(datasets)
        auto_detected_columns = {}
        
        for dataset_name, columns in categorical_columns.items():
            for column in columns:
                mapping_name = self._create_safe_mapping_name(column)
                
                # Only include if not already configured
                if mapping_name not in configured_columns:
                    auto_detected_columns[mapping_name] = {
                        'dataset': dataset_name,
                        'column': column,
                        'source': 'auto_detection'
                    }
        
        summary = {
            'configured_columns': configured_columns,
            'auto_detected_columns': auto_detected_columns,
            'total_configured': len(configured_columns),
            'total_auto_detected': len(auto_detected_columns),
            'total_columns_to_analyze': len(configured_columns) + len(auto_detected_columns)
        }
        
        return summary
