"""
Core constants and identifiers used across the analysis framework.

Centralizing these values avoids hardcoded strings scattered throughout
the codebase and makes it easier to extend with new config keys or chart types.
"""

from enum import Enum

# Configuration keys
CONFIG_KEY_DATASETS = "datasets"
CONFIG_KEY_COLUMN_MAPPINGS = "column_mappings"
CONFIG_KEY_ANALYSIS_OPTIONS = "analysis_options"
CONFIG_KEY_COMBINED_STATS = "combined_statistics"
CONFIG_KEY_STRATIFIED = "stratified_analyses"
CONFIG_KEY_GLOBAL_DEFAULTS = "global_defaults"

# Default dataset naming
DEFAULT_DATASET_NAME = "primary"

# Auxiliary config filenames (without extension)
GLOBAL_DEFAULTS_STEM = "global_defaults"
REPORT_TEMPLATE_STEM = "report_template"

# Output filenames
AGE_DISTRIBUTION_FILE = "age_distribution.png"
BASIC_STATS_FILE = "basic_statistics.json"
TEXT_REPORT_FILE = "analysis_report.txt"


class ChartType(str, Enum):
    """Chart identifiers for registry-based rendering."""
    CATEGORICAL = "categorical_distribution"
    AGE = "age_distribution"
    SPLIT = "data_split"
    COMBINED = "combined_statistics"
    STRATIFIED = "stratified_distribution"
    STRATIFIED_COMPARISON = "stratified_comparison"
