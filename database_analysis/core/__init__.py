#!/usr/bin/env python3
"""
Core module for mammography analysis framework.
Contains the modular components for data processing, analysis, visualization, and reporting.
"""

from .base_analyzer import BaseAnalyzer, AnalysisContext
from .data_loader import DataLoader
from .categorical_detector import CategoricalDetector
from .data_analyzer import DataAnalyzer
from .visualization_handler import VisualizationHandler
from .report_generator import ReportGenerator
from .chart_registry import ChartRegistry
from .constants import ChartType

__all__ = [
    'BaseAnalyzer',
    'AnalysisContext',
    'DataLoader', 
    'CategoricalDetector',
    'DataAnalyzer',
    'VisualizationHandler',
    'ReportGenerator',
    'ChartRegistry',
    'ChartType'
]
