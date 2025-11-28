#!/usr/bin/env python3
"""
Data Loader - Handles dataset loading, cleaning, and preprocessing
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .base_analyzer import BaseAnalyzer
from .constants import CONFIG_KEY_DATASETS


class DataLoader(BaseAnalyzer):
    """Handles data loading and cleaning operations."""
    
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
    
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load datasets based on configuration with enhanced encoding support."""
        self.logger.info("Loading datasets...")
        
        for dataset_name, dataset_config in self.config[CONFIG_KEY_DATASETS].items():
            filename = dataset_config.get('filename') or dataset_config.get('path')
            if filename is None:
                self.logger.error(f"Dataset {dataset_name} missing filename/path")
                continue

            file_path = Path(filename)
            if not file_path.is_absolute():
                file_path = self.data_dir / filename
            
            if not file_path.exists():
                self.logger.error(f"Dataset file not found: {file_path}")
                continue
            
            # Try multiple encodings for robust loading
            encodings = self.global_defaults.get('data_quality_checks', {}).get('encoding_fallbacks', ['utf-8', 'latin-1', 'cp1252'])
            
            df = None
            # Get loading parameters from dataset config
            raw_loading_params = dataset_config.get('loading_params', {})
            explicit_encoding = raw_loading_params.get('encoding')
            fallback_encoding = raw_loading_params.get('fallback_encoding')
            loading_params = {k: v for k, v in raw_loading_params.items() if k not in ['encoding', 'fallback_encoding']}

            # Respect explicit encoding and optional fallback from config
            if explicit_encoding:
                encodings = [explicit_encoding] + [enc for enc in encodings if enc != explicit_encoding]
            if fallback_encoding and fallback_encoding not in encodings:
                encodings.append(fallback_encoding)
            
            for encoding in encodings:
                try:
                    # Start with encoding and add any additional loading parameters
                    read_params = {'encoding': encoding}
                    read_params.update(loading_params)
                    
                    df = pd.read_csv(file_path, **read_params)
                    self.logger.info(f"Loaded {dataset_name} with {encoding} encoding: {len(df):,} records, {len(df.columns)} columns")
                    if loading_params:
                        self.logger.info(f"Applied loading params for {dataset_name}: {loading_params}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error loading {dataset_name}: {e}")
                    break
            
            if df is not None:
                self.raw_datasets[dataset_name] = df
                # Clean dataset immediately after loading
                self.cleaned_datasets[dataset_name] = self._clean_dataset(df, dataset_name)
            else:
                self.logger.error(f"Failed to load dataset: {dataset_name}")
        
        total_records = sum(len(df) for df in self.cleaned_datasets.values())
        self.logger.info(f"Data loading completed: {len(self.cleaned_datasets)} datasets, {total_records:,} total records")
        
        return self.cleaned_datasets
    
    def _clean_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Clean dataset once during loading for optimal performance."""
        cleaned_df = df.copy()
        
        # Get null indicators from global defaults
        null_indicators = self.global_defaults.get('data_quality_checks', {}).get('null_indicators', 
                                                   ["", "NULL", "null", "NA", "N/A", "nan", "NaN"])
        
        # Replace various null indicators with pd.NA
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].replace(null_indicators, pd.NA)
        
        # Log cleaning results only if significant cleaning occurred
        original_nulls = df.isnull().sum().sum()
        cleaned_nulls = cleaned_df.isnull().sum().sum()
        if cleaned_nulls > original_nulls:
            self.logger.info(f"Data cleaning for {dataset_name}: {cleaned_nulls - original_nulls:,} additional nulls identified")
        
        return cleaned_df
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary information about loaded datasets."""
        summary = {}
        for dataset_name, df in self.cleaned_datasets.items():
            summary[dataset_name] = {
                'total_records': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'null_counts': df.isnull().sum().sum(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        return summary
