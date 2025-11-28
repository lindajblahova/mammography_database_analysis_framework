#!/usr/bin/env python3
"""
Main Mammography Analysis Framework - Orchestrates all components
"""

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .core.base_analyzer import BaseAnalyzer, AnalysisContext
from .core.data_loader import DataLoader
from .core.categorical_detector import CategoricalDetector
from .core.data_analyzer import DataAnalyzer
from .core.visualization_handler import VisualizationHandler
from .core.report_generator import ReportGenerator
from .core.constants import (
    CONFIG_KEY_COLUMN_MAPPINGS,
    CONFIG_KEY_STRATIFIED,
    CONFIG_KEY_COMBINED_STATS,
)


def _resolve_config_path(database_name: str) -> Optional[Path]:
    """
    Locate a config file for the requested database, preferring YAML (new format)
    and falling back to legacy JSON paths.
    """
    candidates = [
        Path("database_analysis/configs") / f"{database_name}.yaml",
        Path("database_analysis/configs") / f"{database_name}.yml",
        Path("database_analysis/configs") / f"{database_name}.json",
        Path("mammography_dataset_creator/configs/databases") / f"{database_name}.yaml",
        Path("mammography_dataset_creator/configs/databases") / f"{database_name}.yml",
        Path("mammography_dataset_creator/configs/databases") / f"{database_name}.json",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return None


def _load_analysis_wrapper(path: Path) -> Optional[Dict[str, Any]]:
    """Load an analysis wrapper config that points to a database config plus overrides."""
    try:
        import yaml  # local import to avoid dependency issues if not needed
    except ImportError:
        yaml = None

    if not path.exists():
        return None

    raw = None
    try:
        if path.suffix.lower() in [".yaml", ".yml"] and yaml:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(raw, dict) or "database_config" not in raw:
        return None

    base_dir = path.parent
    db_cfg = Path(raw["database_config"])
    if not db_cfg.is_absolute():
        db_cfg = (base_dir / db_cfg).resolve()

    config_override = raw.get("analysis_overrides", {})
    for key in ["column_mappings", "stratified_analyses", "analysis_options", "combined_statistics"]:
        if key in raw:
            config_override[key] = raw[key]

    return {
        "config_file": str(db_cfg),
        "data_root": raw.get("data_root", "."),
        "output_dir": raw.get("output_dir"),
        "database_name": raw.get("database_name"),
        "config_override": config_override,
    }


class MammographyAnalysisFramework:
    """
    Main framework class that orchestrates all analysis components.
    
    This class coordinates data loading, categorical detection, analysis,
    visualization, and report generation for mammography datasets.
    """
    
    def __init__(self, config_file: str, data_directory: str, output_directory: str = None, database_name: str = None, config_override: Dict[str, Any] = None):
        """
        Initialize the mammography analysis framework.
        
        Args:
            config_file: Path to the database configuration (YAML or JSON)
            data_directory: Path to directory containing dataset files
            output_directory: Path for output files (default: ./output)
            database_name: Override database name from config
        """
        # Initialize loader first to build shared context (config, logging, paths)
        self.data_loader = DataLoader(config_file, data_directory, output_directory, database_name, config_override=config_override)
        shared_ctx: AnalysisContext = self.data_loader.context
        
        # Initialize dependent components with the shared context
        self.categorical_detector = CategoricalDetector(
            config_file, data_directory, output_directory, database_name, context=shared_ctx, config_override=config_override
        )
        self.data_analyzer = DataAnalyzer(
            config_file, data_directory, output_directory, database_name, context=shared_ctx, config_override=config_override
        )
        self.visualization_handler = VisualizationHandler(
            config_file, data_directory, output_directory, database_name, context=shared_ctx, config_override=config_override
        )
        self.report_generator = ReportGenerator(
            config_file, data_directory, output_directory, database_name, context=shared_ctx, config_override=config_override
        )
        
        # Use data_loader as the primary reference for shared properties
        self.config = self.data_loader.config
        self.logger = self.data_loader.logger  # Use data_loader's logger
        self.global_defaults = self.data_loader.global_defaults
        self.output_dir = self.data_loader.output_dir
        
        # Data containers
        self.datasets = {}
        
        self.logger.info(f"Framework initialized for {self.data_loader.get_database_name()}")
    
    def load_datasets(self) -> Dict[str, Any]:
        """Load all datasets using the data loader."""
        self.datasets = self.data_loader.load_datasets()
        return self.datasets
    
    def analyze_categorical_distribution(self, category_name: str, use_log_scale: bool = False, column_config: Dict[str, Any] = None):
        """
        Analyze and visualize the distribution of a categorical variable.
        
        Args:
            category_name: Name of the category to analyze
            use_log_scale: Whether to use log scale for y-axis
            column_config: Column configuration (for dynamic analysis)
        """
        self.logger.info(f"Analyzing {category_name} distribution")
        
        # Use provided config or get from main config
        if column_config is None:
            if category_name not in self.config.get(CONFIG_KEY_COLUMN_MAPPINGS, {}):
                self.logger.error(f"Category {category_name} not found in configuration")
                return None
            column_config = self.config[CONFIG_KEY_COLUMN_MAPPINGS][category_name]
        
        # Extract data using data analyzer
        data = self.data_analyzer.extract_categorical_data(column_config, self.datasets)
        
        if len(data) == 0:
            self.logger.warning(f"No data found for {category_name}")
            return None
        
        # Calculate value counts
        value_counts = data.value_counts()
        
        # Create visualization using visualization handler
        output_file = self.visualization_handler.create_categorical_distribution_plot(
            category_name, value_counts, use_log_scale
        )
        
        # Log results
        most_common = value_counts.index[0] if len(value_counts) > 0 else "None"
        most_common_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
        self.logger.info(f"Analyzed {category_name}: {len(value_counts)} categories, top: {most_common} ({most_common_count:,})")
        
        return value_counts
    
    def analyze_age_distribution(self):
        """Analyze age distribution with validation (if age data is available)."""
        if 'age' not in self.config.get(CONFIG_KEY_COLUMN_MAPPINGS, {}):
            return  # Skip silently - not an error condition
            
        age_data = self.data_analyzer.validate_age_data(
            self.config[CONFIG_KEY_COLUMN_MAPPINGS]['age']['dataset'], 
            self.datasets
        )
        
        if len(age_data) == 0:
            self.logger.warning("No valid age data found")
            return
        
        # Create visualization
        output_file = self.visualization_handler.create_age_distribution_plot(age_data)
        
        # Log age analysis with summary stats
        mean_age = age_data.mean()
        self.logger.info(f"Age analysis completed: {len(age_data):,} patients, mean age {mean_age:.1f}")
        
        return age_data
    
    def analyze_data_splits(self):
        """Analyze data splits if available in datasets."""
        split_found = False
        for dataset_name, df in self.datasets.items():
            if 'split' in df.columns:
                split_found = True
                split_counts = df['split'].value_counts()
                
                # Create visualization
                output_file = self.visualization_handler.create_data_splits_plot(dataset_name, split_counts)
                
                # Log split summary
                splits_info = ", ".join([f"{k}: {v:,}" for k, v in split_counts.items()])
                self.logger.info(f"Data splits analyzed for {dataset_name}: {splits_info}")
                break
        
        return split_found
    
    def analyze_stratified_distribution(self, target_column: str, stratify_column: str):
        """
        Analyze distribution of target column stratified by another column.
        
        Args:
            target_column: Name of the column to analyze (from column_mappings)
            stratify_column: Name of the column to stratify by (from column_mappings)
            
        Returns:
            Dictionary with stratified distribution results
        """
        # Load datasets if not already loaded
        if not hasattr(self, 'datasets') or not self.datasets:
            self.datasets = self.data_loader.load_datasets()
        
        # Get column configurations
        column_mappings = self.config.get(CONFIG_KEY_COLUMN_MAPPINGS, {})
        
        if target_column not in column_mappings:
            self.logger.error(f"Target column '{target_column}' not found in column_mappings")
            return {}
        
        if stratify_column not in column_mappings:
            self.logger.error(f"Stratify column '{stratify_column}' not found in column_mappings")
            return {}
        
        target_config = column_mappings[target_column]
        stratify_config = column_mappings[stratify_column]
        
        # Perform stratified analysis
        stratified_results = self.data_analyzer.analyze_stratified_distribution(
            target_config, stratify_config, self.datasets
        )
        
        if not stratified_results:
            self.logger.warning(f"No results from stratified analysis: {target_column} by {stratify_column}")
            return {}
        
        # Create visualization
        filename_base = f"{target_column}_by_{stratify_column}"
        output_file = self.visualization_handler.create_stratified_distribution_plot(
            stratified_results, filename_base
        )
        # Also create a grouped comparison chart (counts and percentage)
        self.visualization_handler.create_stratified_comparison_plot(
            stratified_results, filename_base, normalize=False
        )
        self.visualization_handler.create_stratified_comparison_plot(
            stratified_results, filename_base, normalize=True
        )
        
        if output_file:
            self.logger.info(f"Stratified analysis completed: {target_column} by {stratify_column}")
            self.logger.info(f"Visualization saved: {output_file}")
        
        return stratified_results
    
    def run_all_stratified_analyses(self):
        """Run all configured stratified analyses."""
        stratified_config = self.config.get(CONFIG_KEY_STRATIFIED, {})
        
        if not stratified_config:
            self.logger.info("No stratified analyses configured")
            return {}
        
        # Load datasets if not already loaded
        if not hasattr(self, 'datasets') or not self.datasets:
            self.datasets = self.data_loader.load_datasets()
        
        stratified_results = {}
        
        for analysis_name, analysis_config in stratified_config.items():
            if not analysis_config.get('enabled', True):
                continue
                
            target_column = analysis_config['target_column']
            stratify_column = analysis_config['stratify_column']
            description = analysis_config.get('description', f'{target_column} by {stratify_column}')
            
            self.logger.info(f"Running stratified analysis: {description}")
            
            result = self.analyze_stratified_distribution(target_column, stratify_column)
            if result:
                stratified_results[analysis_name] = {
                    'config': analysis_config,
                    'results': result
                }
        
        if stratified_results:
            self.logger.info(f"Completed {len(stratified_results)} stratified analyses")
        
        return stratified_results
    
    def generate_comprehensive_report(self, enable_auto_detection: bool = True):
        """
        Generate comprehensive analysis report with optional auto-detection of categorical columns.
        
        Args:
            enable_auto_detection: Whether to auto-detect categorical columns beyond configured ones
        """
        # Load datasets if not already loaded
        if not self.datasets:
            self.load_datasets()
        
        # Configure analysis scope
        if enable_auto_detection:
            categorical_columns = self.categorical_detector.auto_detect_categorical_columns(self.datasets)
            dynamic_mappings = self.categorical_detector.create_dynamic_column_mappings(categorical_columns)
            detection_mode = "with auto-detection"
        else:
            dynamic_mappings = self.config.get(CONFIG_KEY_COLUMN_MAPPINGS, {})
            detection_mode = "configured columns only"
        
        # Generate analysis summary and basic statistics
        column_summary = self.categorical_detector.get_column_analysis_summary(self.datasets)
        age_data = self.analyze_age_distribution()  # Returns age data or None
        
        stats = self.report_generator.generate_basic_statistics(
            self.datasets, 
            age_data, 
            column_summary
        )
        
        # Save basic statistics
        self.report_generator.save_statistics_json(stats)
        
        self.logger.info(f"Comprehensive analysis started ({detection_mode}): {column_summary['total_columns_to_analyze']} columns to analyze")
        
        # Analyze categorical distributions
        category_results = {}
        categorical_count = 0
        for category_name, category_config in dynamic_mappings.items():
            if category_name in ['age', 'data_split']:
                continue  # These are handled separately
            
            use_log_scale = category_config.get('use_log_scale', False)
            category_results[category_name] = self.analyze_categorical_distribution(
                category_name, use_log_scale, category_config
            )
            categorical_count += 1
        
        # Analyze data splits
        self.analyze_data_splits()
        
        # Run stratified analyses
        stratified_results = self.run_all_stratified_analyses()
        
        # Analyze combined statistics
        combined_results = self.data_analyzer.analyze_combined_statistics(self.datasets)
        
        # Create combined statistics visualizations
        for stat_name, stat_data in combined_results.items():
            if stat_data is not None and len(stat_data) > 0:
                stat_config = self.config.get(CONFIG_KEY_COMBINED_STATS, {}).get(stat_name, {})
                display_name = stat_config.get('display_name', stat_name.replace('_', ' ').title())
                
                # Create main visualization
                self.visualization_handler.create_combined_statistics_plot(stat_data, display_name, stat_name)
                
                # Create detailed breakdown if it's a case count analysis
                analysis_type = stat_config.get('analysis_type', '')
                if analysis_type == 'case_count':
                    # Also create detailed breakdown visualization (this would be implemented in the analyzer)
                    pass
        
        # Generate text report
        self.report_generator.generate_text_report(stats, category_results, combined_results, stratified_results)
        
        self.logger.info(f"Analysis completed: {categorical_count} categorical distributions, age analysis, data splits, {len(stratified_results)} stratified analyses, and {len(combined_results) if combined_results else 0} combined statistics")


# Main execution function for backwards compatibility
def main():
    """Main function for running the analysis framework."""
    
    if len(sys.argv) < 3:
        print("Usage: python mammography_analyzer_new.py <config_file> <data_directory> [output_directory]")
        print("   OR: python mammography_analyzer_new.py --database <database_name> --data-root <workspace_root>")
        sys.exit(1)
    
    # Handle new command format: --database dbname --data-root path
    if sys.argv[1] == '--database' and len(sys.argv) >= 4:
        database_name = sys.argv[2]
        data_root = sys.argv[4] if sys.argv[3] == '--data-root' else '.'
        
        # Find config file based on database name (YAML preferred)
        config_path = _resolve_config_path(database_name)
        if config_path is None:
            # Fallback to legacy path
            config_path = Path(f"mammography-framework/configs/{database_name}.json")
        
        # Initialize and run framework
        framework = MammographyAnalysisFramework(str(config_path), data_root, None, database_name)
        framework.generate_comprehensive_report(enable_auto_detection=True)
        
    else:
        # Handle original format: config_file data_directory [output_directory]
        config_file = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3] if len(sys.argv) > 3 else None

        # Support analysis wrapper configs
        wrapper = _load_analysis_wrapper(Path(config_file))
        if wrapper:
            config_file = wrapper["config_file"]
            data_directory = wrapper["data_root"]
            output_directory = wrapper["output_dir"] or output_directory
            database_name = wrapper.get("database_name")
            config_override = wrapper.get("config_override")
        else:
            config_override = None
        
        # Extract database name from config file name
        config_path = Path(config_file)
        database_name = config_path.stem
        
        # Initialize and run framework
        framework = MammographyAnalysisFramework(config_file, data_directory, output_directory, database_name, config_override=config_override)
        framework.generate_comprehensive_report(enable_auto_detection=True)


if __name__ == "__main__":
    main()
