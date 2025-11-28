#!/usr/bin/env python3
"""
Data Analyzer - Handles specific data analysis tasks (age, categorical distributions)
"""

import pandas as pd
import ast
from typing import Dict, List, Any, Optional
from .base_analyzer import BaseAnalyzer
from .constants import (
    CONFIG_KEY_COLUMN_MAPPINGS,
    CONFIG_KEY_COMBINED_STATS,
)


class DataAnalyzer(BaseAnalyzer):
    """Handles specific data analysis operations."""
    
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
    
    def extract_categorical_data(self, column_config: Dict[str, Any], datasets: Dict[str, pd.DataFrame]) -> pd.Series:
        """Extract categorical data from the specified cleaned dataset and column."""
        dataset_name = column_config['dataset']
        column_name = column_config['column']
        
        if dataset_name not in datasets:
            self.logger.warning(f"Dataset {dataset_name} not found")
            return pd.Series(dtype=object)
        
        df = datasets[dataset_name]
        if column_name not in df.columns:
            self.logger.warning(f"Column {column_name} not found in dataset {dataset_name}")
            return pd.Series(dtype=object)
        
        # Extract data - handle null values based on configuration
        include_null = column_config.get('include_null', False)
        
        if include_null:
            # Keep null values but replace them with a placeholder for analysis
            data = df[column_name].copy()
            # Use negative indicators from global defaults to identify missing values
            negative_indicators = self.global_defaults.get('negative_indicators', {}).get('standard', [])
            
            # Create mask for null values (pandas NA/NaN)
            mask = data.isnull()
            
            # Also check for negative indicators from global defaults
            for indicator in negative_indicators:
                if indicator == "":
                    # Handle empty strings specifically
                    mask = mask | (data.astype(str).str.strip() == "")
                else:
                    mask = mask | (data.astype(str) == str(indicator))
            
            # Replace all identified null/negative values with "No Finding"
            data.loc[mask] = "No Finding"
        else:
            # Original behavior - drop null values
            data = df[column_name].dropna()
        
        # Handle list-type data (like finding_categories)
        if column_config.get('is_list', False):
            expanded_data = []
            for item in data:
                if isinstance(item, str) and '[' in item:
                    try:
                        parsed_list = ast.literal_eval(item)
                        expanded_data.extend(parsed_list)
                    except (ValueError, SyntaxError):
                        expanded_data.append(item)
                else:
                    expanded_data.append(item)
            data = pd.Series(expanded_data)
        
        return data
    
    def validate_age_data(self, dataset_name: str, datasets: Dict[str, pd.DataFrame]) -> pd.Series:
        """Validate and clean age data from specified cleaned dataset."""
        if dataset_name not in datasets:
            self.logger.warning(f"Dataset {dataset_name} not found")
            return pd.Series(dtype=float)
        
        df = datasets[dataset_name]
        column_mappings = self.config.get(CONFIG_KEY_COLUMN_MAPPINGS, {})
        age_config = column_mappings.get('age', {})
        
        if age_config.get('dataset') != dataset_name or age_config.get('column') not in df.columns:
            self.logger.warning(f"Age column not found in {dataset_name}")
            return pd.Series(dtype=float)
        
        age_column = age_config['column']
        
        # Extract numeric age values
        age_data = df[age_column].astype(str)
        
        # Remove suffix if specified, or use global defaults
        suffix = age_config.get('suffix')
        if not suffix:
            # Try common suffixes from global defaults
            common_suffixes = self.global_defaults.get('age_validation', {}).get('common_suffixes', ['Y', 'years'])
            for common_suffix in common_suffixes:
                if age_data.str.contains(common_suffix, na=False).any():
                    suffix = common_suffix
                    break
        
        if suffix:
            age_data = age_data.str.replace(suffix, '', regex=False)
        
        # Convert to numeric
        age_data = pd.to_numeric(age_data, errors='coerce')
        
        # Apply valid range filter - use global defaults if not specified
        if 'valid_range' in age_config:
            min_age, max_age = age_config['valid_range']
        else:
            default_range = self.global_defaults.get('age_validation', {}).get('default_range', [1, 120])
            min_age, max_age = default_range
            
        valid_ages = age_data[(age_data >= min_age) & (age_data <= max_age)]
        
        # Log validation results only if significant filtering occurred
        total_records = len(df)
        valid_count = len(valid_ages)
        filtered_out = total_records - valid_count
        
        if filtered_out > 0:
            self.logger.info(f"Age validation: {valid_count:,}/{total_records:,} valid ({filtered_out:,} filtered)")
        
        return valid_ages
    
    def analyze_combined_statistics(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze combined statistics across multiple columns based on config definitions."""
        if CONFIG_KEY_COMBINED_STATS not in self.config:
            return {}  # Skip silently - not an error
        
        combined_results = {}
        success_count = 0
        
        for stat_name, stat_config in self.config[CONFIG_KEY_COMBINED_STATS].items():
            try:
                result = self._analyze_single_combined_statistic(stat_name, stat_config, datasets)
                if result is not None and len(result) > 0:
                    combined_results[stat_name] = result
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Error analyzing combined statistic {stat_name}: {e}")
        
        # Single summary log for all combined statistics
        if success_count > 0:
            self.logger.info(f"Analyzed {success_count} combined statistics")
        
        return combined_results
    
    def _analyze_single_combined_statistic(self, stat_name: str, stat_config: Dict[str, Any], datasets: Dict[str, pd.DataFrame]):
        """Analyze a single combined statistic based on its configuration."""
        columns = stat_config.get('columns', [])
        
        # Use global defaults for indicators if not specified
        default_positive = self.global_defaults.get('positive_indicators', {}).get('standard', ['X', 'x', '1', 'Yes', 'yes'])
        positive_indicators = stat_config.get('positive_indicators', default_positive)
        
        analysis_type = stat_config.get('analysis_type', 'presence_count')
        display_name = stat_config.get('display_name', stat_name.replace('_', ' ').title())
        
        if not columns:
            self.logger.warning(f"No columns defined for combined statistic: {stat_name}")
            return None
        
        # Find the dataset containing these columns
        target_df = None
        
        for dataset_name, df in datasets.items():
            available_columns = [col for col in columns if col in df.columns]
            if len(available_columns) == len(columns):
                target_df = df
                break
            elif len(available_columns) > 0:
                self.logger.warning(f"Only {len(available_columns)}/{len(columns)} columns found in {dataset_name}")
        
        if target_df is None:
            self.logger.error(f"Could not find dataset with all required columns for {stat_name}")
            return None
        
        # Perform analysis based on type
        if analysis_type == 'presence_count':
            return self._analyze_presence_count(target_df, columns, positive_indicators, display_name, stat_name)
        elif analysis_type == 'case_count':
            return self._analyze_case_count(target_df, columns, positive_indicators, display_name, stat_name)
        elif analysis_type == 'value_distribution':
            return self._analyze_value_distribution(target_df, columns, display_name, stat_name)
        else:
            self.logger.error(f"Unknown analysis type: {analysis_type}")
            return None
    
    def _analyze_presence_count(self, df: pd.DataFrame, columns: List[str], positive_indicators: List[str], display_name: str, stat_name: str):
        """Count total presence across all specified columns."""
        finding_counts = {}
        
        for column in columns:
            if column in df.columns:
                # Check if this column contains string list data that needs parsing
                sample_data = df[column].dropna().astype(str).head(10)
                contains_lists = any(item.strip().startswith('[') and item.strip().endswith(']') for item in sample_data if item.strip())
                
                if contains_lists:
                    # Parse string list format (e.g., "['Mass']", "['Suspicious Calcification', 'Mass']")
                    finding_counts.update(self._parse_string_list_column(df, column))
                else:
                    # Standard handling for simple categorical data
                    positive_mask = df[column].astype(str).isin(positive_indicators)
                    count = positive_mask.sum()
                    
                    # Clean column name for display
                    clean_name = column.strip().replace('_', ' ').title()
                    if clean_name.endswith(' '):
                        clean_name = clean_name.strip()
                    
                    finding_counts[clean_name] = count
        
        if not finding_counts:
            return None
        
        # Convert to pandas Series for consistent handling
        result_series = pd.Series(finding_counts).sort_values(ascending=False)
        return result_series
    
    def _parse_string_list_column(self, df: pd.DataFrame, column: str):
        """Parse any column containing string list format (e.g., "['item1', 'item2']")."""
        finding_counts = {}
        
        for idx, item_str in df[column].items():
            try:
                # Parse string representation of list
                if pd.isna(item_str) or item_str == '':
                    continue
                    
                # Handle string lists like "['Mass']" or "['Suspicious Calcification', 'Mass']"
                item_str = str(item_str).strip()
                if item_str.startswith('[') and item_str.endswith(']'):
                    items_list = ast.literal_eval(item_str)
                    
                    for item in items_list:
                        item = str(item).strip()
                        # Skip common negative indicators from global defaults
                        negative_indicators = self.global_defaults.get('negative_indicators', {}).get('standard', [])
                        if item and item not in negative_indicators:
                            finding_counts[item] = finding_counts.get(item, 0) + 1
                                
            except (ValueError, SyntaxError):
                # If parsing fails, treat as single item
                item_str = str(item_str).strip()
                negative_indicators = self.global_defaults.get('negative_indicators', {}).get('standard', [])
                if item_str and item_str not in negative_indicators and not item_str.startswith('['):
                    finding_counts[item_str] = finding_counts.get(item_str, 0) + 1
        
        return finding_counts
    
    def _analyze_case_count(self, df: pd.DataFrame, columns: List[str], positive_indicators: List[str], display_name: str, stat_name: str):
        """Count cases with any positive finding across specified columns."""
        # Create a mask for any positive finding
        any_positive_mask = pd.Series([False] * len(df), index=df.index)
        
        finding_details = {}
        for column in columns:
            if column in df.columns:
                # Check if this column contains string list data
                sample_data = df[column].dropna().astype(str).head(10)
                contains_lists = any(item.strip().startswith('[') and item.strip().endswith(']') for item in sample_data if item.strip())
                
                if contains_lists:
                    # For string list format, check if any row contains positive findings
                    for idx, item_str in df[column].items():
                        if pd.isna(item_str):
                            continue
                        item_str = str(item_str).strip()
                        # Mark as positive if it's not a negative indicator from global defaults
                        negative_indicators = self.global_defaults.get('negative_indicators', {}).get('standard', [])
                        # Add list format variants
                        list_negative_indicators = negative_indicators + [f"['{indicator}']" for indicator in negative_indicators if indicator]
                        if item_str not in list_negative_indicators and item_str != '':
                            any_positive_mask.iloc[idx] = True
                    
                    # Get detailed breakdown using the parsing function
                    parsed_findings = self._parse_string_list_column(df, column)
                    finding_details.update(parsed_findings)
                else:
                    # Standard handling for simple categorical data
                    positive_mask = df[column].astype(str).isin(positive_indicators)
                    any_positive_mask |= positive_mask
                    
                    clean_name = column.strip().replace('_', ' ').title()
                    finding_details[clean_name] = positive_mask.sum()
        
        # Summary statistics
        total_cases = len(df)
        cases_with_findings = any_positive_mask.sum()
        cases_without_findings = total_cases - cases_with_findings
        
        summary_data = {
            'Cases with Findings': cases_with_findings,
            'Cases without Findings': cases_without_findings
        }
        
        result_series = pd.Series(summary_data)
        return result_series
    
    def _analyze_value_distribution(self, df: pd.DataFrame, columns: List[str], display_name: str, stat_name: str):
        """Analyze the actual distribution of values in specified columns."""
        if len(columns) != 1:
            self.logger.error(f"Value distribution analysis requires exactly one column, got {len(columns)}")
            return None
        
        column = columns[0]
        if column not in df.columns:
            self.logger.error(f"Column {column} not found in dataset")
            return None
        
        # Get value counts for the column
        value_counts = df[column].value_counts()
        
        # Create meaningful labels for the values using general rules
        labeled_counts = {}
        negative_indicators = self.global_defaults.get('negative_indicators', {}).get('standard', [])
        
        for value, count in value_counts.items():
            value_str = str(value).strip() if not pd.isna(value) else ''
            
            # Handle null/missing/negative values using global defaults
            if pd.isna(value) or value_str == '' or value_str in negative_indicators or value_str.upper() in [ind.upper() for ind in negative_indicators if ind]:
                if 'Unknown/Missing' not in labeled_counts:
                    labeled_counts['Unknown/Missing'] = 0
                labeled_counts['Unknown/Missing'] += count
            else:
                # Use the original value with some cleanup for readability
                clean_label = value_str
                
                # Apply common cleanup patterns
                clean_label = clean_label.replace('_', ' ').replace('-', ' ')
                
                # Capitalize appropriately
                if clean_label.isupper() or clean_label.islower():
                    clean_label = clean_label.title()
                
                # Handle common binary indicators using global defaults
                positive_indicators = self.global_defaults.get('positive_indicators', {}).get('standard', [])
                if clean_label in positive_indicators or clean_label in [str(ind) for ind in positive_indicators]:
                    # Get labeling rules from global defaults
                    labeling_config = self.global_defaults.get('positive_value_labeling', {})
                    cancer_keywords = labeling_config.get('cancer_keywords', [])
                    recall_keywords = labeling_config.get('recall_keywords', [])
                    default_label = labeling_config.get('default_positive_label', 'Positive')
                    
                    # Try to infer meaning from column name using configurable keywords
                    if any(keyword in column.lower() for keyword in cancer_keywords):
                        clean_label = 'Positive Cases'
                    elif any(keyword in column.lower() for keyword in recall_keywords):
                        clean_label = 'Recalled'
                    else:
                        clean_label = default_label
                
                labeled_counts[clean_label] = count
        
        # Convert to pandas Series for consistent handling
        result_series = pd.Series(labeled_counts).sort_values(ascending=False)
        return result_series
    
    def check_data_splits_exist(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """Check if any dataset has split information."""
        for dataset_name, df in datasets.items():
            if 'split' in df.columns:
                return True
        return False
    
    def analyze_stratified_distribution(self, target_column_config: Dict[str, Any], 
                                      stratify_column_config: Dict[str, Any], 
                                      datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze distribution of target column stratified by another column.
        
        Args:
            target_column_config: Configuration for the column to analyze (e.g., breast_density)
            stratify_column_config: Configuration for the column to stratify by (e.g., split)
            datasets: Dictionary of loaded datasets
            
        Returns:
            Dictionary with stratified distribution results
        """
        # Get dataset and column names
        target_dataset = target_column_config['dataset']
        target_column = target_column_config['column']
        stratify_dataset = stratify_column_config['dataset']
        stratify_column = stratify_column_config['column']
        
        # Validate datasets exist
        if target_dataset not in datasets:
            self.logger.warning(f"Target dataset {target_dataset} not found")
            return {}
        
        if stratify_dataset not in datasets:
            self.logger.warning(f"Stratify dataset {stratify_dataset} not found")
            return {}
        
        # Get dataframes
        target_df = datasets[target_dataset]
        stratify_df = datasets[stratify_dataset]
        
        # Validate columns exist
        if target_column not in target_df.columns:
            self.logger.warning(f"Target column {target_column} not found in dataset {target_dataset}")
            return {}
        
        if stratify_column not in stratify_df.columns:
            self.logger.warning(f"Stratify column {stratify_column} not found in dataset {stratify_dataset}")
            return {}
        
        # If datasets are different, we need to merge them
        if target_dataset != stratify_dataset:
            # Try to find common columns for merging
            common_cols = set(target_df.columns) & set(stratify_df.columns)
            if not common_cols:
                self.logger.error(f"No common columns found between {target_dataset} and {stratify_dataset} for merging")
                return {}
            
            # Use the first common column as merge key (you might want to make this configurable)
            merge_key = list(common_cols)[0]
            combined_df = target_df.merge(stratify_df[[merge_key, stratify_column]], on=merge_key, how='inner')
        else:
            combined_df = target_df.copy()
        
        # Handle include_null for target column
        include_null = target_column_config.get('include_null', False)
        
        if include_null:
            # Apply include_null processing to target column
            data = combined_df[target_column].copy()
            negative_indicators = self.global_defaults.get('negative_indicators', {}).get('standard', [])
            
            mask = data.isnull()
            for indicator in negative_indicators:
                if indicator == "":
                    mask = mask | (data.astype(str).str.strip() == "")
                else:
                    mask = mask | (data.astype(str) == str(indicator))
            
            combined_df.loc[mask, target_column] = "No Finding"
        
        # Remove rows where either column is null (after include_null processing)
        clean_df = combined_df[[target_column, stratify_column]].dropna()
        
        if len(clean_df) == 0:
            self.logger.warning("No valid data remaining after cleaning for stratified analysis")
            return {}
        
        # Create stratified distributions
        stratified_results = {}
        stratified_counts = {}
        
        for strata_value in clean_df[stratify_column].unique():
            strata_data = clean_df[clean_df[stratify_column] == strata_value]
            target_counts = strata_data[target_column].value_counts().sort_values(ascending=False)
            
            stratified_results[str(strata_value)] = target_counts.to_dict()
            stratified_counts[str(strata_value)] = len(strata_data)
        
        # Calculate overall distribution for comparison
        overall_counts = clean_df[target_column].value_counts().sort_values(ascending=False)
        
        result = {
            'stratified_distributions': stratified_results,
            'strata_counts': stratified_counts,
            'overall_distribution': overall_counts.to_dict(),
            'total_records': len(clean_df),
            'target_column_name': target_column,
            'stratify_column_name': stratify_column,
            'target_description': target_column_config.get('description', target_column),
            'stratify_description': stratify_column_config.get('description', stratify_column)
        }
        
        self.logger.info(f"Analyzed stratified distribution: {target_column} by {stratify_column} "
                        f"({len(stratified_results)} strata, {len(clean_df)} total records)")
        
        return result
