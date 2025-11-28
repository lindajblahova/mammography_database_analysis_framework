"""
Database Analysis Module
Provides tools for analyzing individual mammography databases.
"""

__version__ = "2.0.0"
__author__ = "Mammography Analysis Framework"

# Import main classes for easy access
from .mammography_analyzer_new import MammographyAnalysisFramework
from .core.base_analyzer import BaseAnalyzer
from .core.data_loader import DataLoader
from .core.categorical_detector import CategoricalDetector
from .core.data_analyzer import DataAnalyzer
from .core.visualization_handler import VisualizationHandler
from .core.report_generator import ReportGenerator

__all__ = [
    'MammographyAnalysisFramework',
    'BaseAnalyzer',
    'DataLoader',
    'CategoricalDetector', 
    'DataAnalyzer',
    'VisualizationHandler',
    'ReportGenerator'
]