#!/usr/bin/env python3
"""
Base Analyzer Class - Core functionality and configuration management
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import yaml

from .config_adapter import DatabaseConfigAdapter
from .constants import (
    CONFIG_KEY_DATASETS,
    CONFIG_KEY_COLUMN_MAPPINGS,
    GLOBAL_DEFAULTS_STEM,
    REPORT_TEMPLATE_STEM,
)


@dataclass
class AnalysisContext:
    """Shared context for analysis components to avoid repeated config/loading work."""
    config_file: Path
    data_dir: Path
    output_dir: Path
    database_name: str
    config: Dict[str, Any]
    global_defaults: Dict[str, Any]
    report_template: Dict[str, Any]
    logger: logging.Logger


class BaseAnalyzer:
    """Base class for mammography analysis with configuration management."""
    
    def __init__(
        self,
        config_file: str,
        data_directory: str,
        output_directory: str = None,
        database_name: str = None,
        context: Optional[AnalysisContext] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the base analyzer with configuration and directories."""
        # Minimal logger for early setup; replaced once context is ready
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Reuse existing context when orchestrating multiple components
        if context:
            self._apply_context(context)
            return
        
        self.config_file = Path(config_file)
        self.database_name = database_name
        self._input_data_directory = data_directory
        
        # Load configuration (supports YAML/JSON) and normalize legacy db configs
        self.config = self._load_configuration()
        self.config = self._normalize_config(self.config, data_directory)
        if config_override:
            self._merge_config(self.config, config_override)
        
        # Determine database name
        config_db_name = self.config.get('database_name') or self.config.get('name', '')
        config_db_name = str(config_db_name).lower().replace('-', '_')
        actual_db_name = database_name or config_db_name
        self.database_name = actual_db_name
        
        # Resolve data/output directories with sensible defaults
        self.data_dir, self.output_dir = self._resolve_directories(
            data_directory, output_directory, actual_db_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging once directories are available
        self.logger = self._setup_logging()
        
        # Load auxiliary configs
        self.global_defaults = self._load_global_defaults()
        self.report_template = self._load_report_template()
        
        # Data containers
        self.raw_datasets = {}
        self.cleaned_datasets = {}
        
        # Persist context for reuse by other components
        self.context = AnalysisContext(
            config_file=self.config_file,
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            database_name=self.database_name,
            config=self.config,
            global_defaults=self.global_defaults,
            report_template=self.report_template,
            logger=self.logger,
        )
        
        self.logger.info(f"Initialized analyzer for {self.config.get('database_name', 'Unknown Database')}")
    
    def _apply_context(self, context: AnalysisContext):
        """Attach an existing analysis context (used by orchestrator to share state)."""
        self.context = context
        self.config_file = context.config_file
        self.data_dir = context.data_dir
        self.output_dir = context.output_dir
        self.database_name = context.database_name
        self.config = context.config
        self.global_defaults = context.global_defaults
        self.report_template = context.report_template
        self.logger = context.logger
        self.config_override = None
        
        # Data containers
        self.raw_datasets = {}
        self.cleaned_datasets = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"analysis_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(self.__class__.__name__)
    
    def _load_temp_configuration(self) -> Dict[str, Any]:
        """Load database configuration temporarily to determine output directory."""
        try:
            return self._load_any_config(self.config_file)
        except Exception:
            return {}
    
    def _load_any_config(self, path: Path) -> Dict[str, Any]:
        """Load a YAML or JSON config file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        suffix = path.suffix.lower()
        try:
            if suffix in ['.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            # Default to JSON
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {path}: {e}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file {path}: {e}")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load database configuration from YAML or JSON file."""
        config = self._load_any_config(self.config_file)
        self.logger.info(f"Loaded configuration from {self.config_file}")
        return config

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Deep-merge override dict into base config."""
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _normalize_config(self, config: Dict[str, Any], data_directory: str) -> Dict[str, Any]:
        """
        Normalize configs from the refactored YAML database definitions to the
        analysis-friendly shape (datasets/column_mappings). Acts as an Adapter.
        """
        if CONFIG_KEY_DATASETS in config:
            return config
        
        if 'files' in config or 'field_mappings' in config:
            adapter = DatabaseConfigAdapter(config, self.config_file, data_directory)
            normalized = adapter.to_analysis_config()
            self.logger.info("Adapted database YAML config for analysis usage")
            return normalized
        
        return config
    
    def _resolve_directories(
        self,
        data_directory: str,
        output_directory: Optional[str],
        actual_db_name: str
    ):
        """Determine data/output directories with awareness of repository layout."""
        # Explicit directories from config take precedence
        config_data_dir = self.config.get('data_directory') or self.config.get('data_dir')
        if config_data_dir:
            data_dir = Path(config_data_dir).expanduser().resolve()
            output_dir = Path(output_directory).expanduser() if output_directory else data_dir / "results"
            return data_dir, output_dir
        
        if actual_db_name:
            current_path = Path(data_directory or ".").resolve()
            
            # Navigate to find database_descriptions
            workspace_root = current_path
            while workspace_root.parent != workspace_root:
                if (workspace_root / "database_descriptions").exists():
                    break
                workspace_root = workspace_root.parent
            
            db_folder = workspace_root / "database_descriptions" / actual_db_name.replace('_', '-')
            data_dir = db_folder
            output_dir = Path(output_directory).expanduser() if output_directory else db_folder / "results"
            return data_dir, output_dir
        
        # Fallback to provided paths
        data_dir = Path(data_directory or ".").resolve()
        output_dir = Path(output_directory).expanduser() if output_directory else Path.cwd() / "output"
        return data_dir, output_dir
    
    def _load_global_defaults(self) -> Dict[str, Any]:
        """Load global defaults configuration (supports YAML/JSON)."""
        try:
            defaults = self._load_auxiliary_config(GLOBAL_DEFAULTS_STEM)
            if defaults is None:
                return {}
            return defaults.get(GLOBAL_DEFAULTS_STEM, defaults)
        except Exception as e:
            self.logger.warning(f"Could not load global defaults: {e}")
            return {}
    
    def _load_auxiliary_config(self, stem: str) -> Optional[Dict[str, Any]]:
        """Load auxiliary config (global defaults/report template) from YAML or JSON."""
        for ext in ['.yaml', '.yml', '.json']:
            candidate = self.config_file.parent / f"{stem}{ext}"
            if candidate.exists():
                return self._load_any_config(candidate)
        return None
    
    def _load_report_template(self) -> Dict[str, Any]:
        """Load report template configuration."""
        try:
            template = self._load_auxiliary_config(REPORT_TEMPLATE_STEM)
            if template is None:
                return self._get_default_report_template()
            return template.get(REPORT_TEMPLATE_STEM, template)
        except Exception as e:
            self.logger.warning(f"Could not load report template: {e}")
            return self._get_default_report_template()
    
    def _get_default_report_template(self) -> Dict[str, Any]:
        """Provide a minimal default report template as fallback."""
        return {
            "header": {
                "template": [
                    "Mammography Database Analysis Report",
                    "=============================================================",
                    "Database: {database_name}",
                    "Analysis Date: {analysis_date}",
                    ""
                ]
            },
            "sections": {
                "dataset_overview": {
                    "title": "DATASET OVERVIEW:",
                    "required": True,
                    "template": [
                        "  {dataset_name}:",
                        "    Total records: {total_records:,}",
                        "    Columns: {columns}"
                    ]
                }
            }
        }
    
    def get_database_name(self) -> str:
        """Get the database name from config or provided parameter."""
        return self.database_name or self.config.get('database_name', 'Unknown Database')
