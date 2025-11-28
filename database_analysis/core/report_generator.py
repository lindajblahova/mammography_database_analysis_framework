#!/usr/bin/env python3
"""
Report Generator - Handles report generation using template system
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base_analyzer import BaseAnalyzer
from .constants import BASIC_STATS_FILE, TEXT_REPORT_FILE


class ReportGenerator(BaseAnalyzer):
    """Handles report generation using template-based system."""
    
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
    
    def generate_basic_statistics(self, datasets: Dict[str, Any], age_data: Optional[Any] = None, column_summary: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate basic statistics for all datasets."""
        
        stats = {
            'database_name': self.get_database_name(),
            'analysis_date': datetime.now().isoformat(),
            'datasets': {}
        }
        
        # Dataset overview
        for dataset_name, df in datasets.items():
            dataset_stats = {
                'total_records': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
            
            # Count unique identifiers if configured
            unique_identifiers = self.config.get('unique_identifiers', {})
            for id_type, id_config in unique_identifiers.items():
                if id_config['dataset'] == dataset_name:
                    unique_count = df[id_config['column']].nunique()
                    dataset_stats[f'unique_{id_type}'] = unique_count
            
            stats['datasets'][dataset_name] = dataset_stats
        
        # Add age statistics if available
        if age_data is not None and len(age_data) > 0:
            stats['age_statistics'] = {
                'mean': float(age_data.mean()),
                'median': float(age_data.median()),
                'std': float(age_data.std()),
                'min': int(age_data.min()),
                'max': int(age_data.max()),
                'count': len(age_data)
            }
        
        # Add column analysis summary if provided
        if column_summary:
            stats['column_analysis_summary'] = column_summary
        
        return stats
    
    def generate_text_report(self, stats: Dict[str, Any], category_results: Dict[str, Any], 
                           combined_results: Dict[str, Any] = None, stratified_results: Dict[str, Any] = None) -> Path:
        """Generate a text-based summary report using template system."""
        # Prepare report data context
        report_context = self._prepare_report_context(stats, category_results, combined_results, stratified_results)
        
        # Generate report using template
        report_lines = []
        
        # Add header
        header_template = self.report_template.get('header', {}).get('template', [])
        for line in header_template:
            try:
                formatted_line = line.format(**report_context)
                report_lines.append(formatted_line)
            except (KeyError, ValueError):
                report_lines.append(line)  # Use line as-is if formatting fails
        
        # Process sections
        sections = self.report_template.get('sections', {})
        for section_name, section_config in sections.items():
            section_lines = self._generate_report_section(section_name, section_config, report_context)
            if section_lines:
                report_lines.extend(section_lines)
        
        # Save report to file
        report_content = "\n".join(report_lines)
        report_file = self.output_dir / TEXT_REPORT_FILE
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Report saved to: {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return None
        
        return report_file
    
    def save_statistics_json(self, stats: Dict[str, Any]) -> Path:
        """Save basic statistics to JSON file."""
        stats_file = self.output_dir / BASIC_STATS_FILE
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            self.logger.info(f"Statistics saved to: {stats_file}")
            return stats_file
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")
            return None
    
    def _prepare_report_context(self, stats: Dict[str, Any], category_results: Dict[str, Any], 
                              combined_results: Dict[str, Any] = None, stratified_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare the data context for report template rendering."""
        context = {
            # Basic stats
            'database_name': stats.get('database_name', 'Unknown'),
            'analysis_date': stats.get('analysis_date', 'Unknown'),
            
            # Datasets info
            'datasets': stats.get('datasets', {}),
            
            # Column analysis summary (might not exist)
            'column_analysis_summary': stats.get('column_analysis_summary'),
            
            # Age statistics (might not exist)
            'age_statistics': stats.get('age_statistics'),
            
            # Category results
            'category_results': category_results,
            
            # Combined statistics (might not exist)
            'combined_results': combined_results,
            
            # Stratified analysis results (might not exist)
            'stratified_results': stratified_results,
            
            # Derived flags for conditional rendering
            'has_age_data': 'age_statistics' in stats,
            'has_column_summary': 'column_analysis_summary' in stats,
            'has_combined_results': bool(combined_results),
            'has_stratified_results': bool(stratified_results),
            'has_data_splits': self._check_for_data_splits_in_context(stats),
        }
        
        # Add column summary details if available
        if context['column_analysis_summary']:
            col_summary = context['column_analysis_summary']
            context.update({
                'total_configured': col_summary.get('total_configured', 0),
                'total_auto_detected': col_summary.get('total_auto_detected', 0),
                'total_columns_to_analyze': col_summary.get('total_columns_to_analyze', 0),
                'configured_columns': col_summary.get('configured_columns', {}),
                'auto_detected_columns': col_summary.get('auto_detected_columns', {})
            })
        
        return context
    
    def _check_for_data_splits_in_context(self, stats: Dict[str, Any]) -> bool:
        """Check if data splits information is available in stats context."""
        # This is a simplified check - in practice, you might want to pass this info explicitly
        return False
    
    def _generate_report_section(self, section_name: str, section_config: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate a single report section based on template configuration."""
        section_lines = []
        
        # Check if section should be included
        if not self._should_include_section(section_config, context):
            return []
        
        # Add section title
        title = section_config.get('title', '')
        if title:
            try:
                formatted_title = title.format(**context)
                section_lines.append("")
                section_lines.append(formatted_title)
            except (KeyError, ValueError):
                section_lines.append("")
                section_lines.append(title)
        
        # Add separator if specified
        separator = section_config.get('separator')
        if separator:
            section_lines.append(separator)
        
        # Handle different section types
        if section_name == 'dataset_overview':
            section_lines.extend(self._generate_dataset_overview_section(section_config, context))
        elif section_name == 'column_analysis_scope':
            section_lines.extend(self._generate_column_analysis_section(section_config, context))
        elif section_name == 'patient_demographics':
            section_lines.extend(self._generate_demographics_section(section_config, context))
        elif section_name == 'category_distributions':
            section_lines.extend(self._generate_category_distributions_section(section_config, context))
        elif section_name == 'combined_statistics':
            section_lines.extend(self._generate_combined_statistics_section(section_config, context))
        elif section_name == 'stratified_analyses':
            section_lines.extend(self._generate_stratified_analyses_section(section_config, context))
        elif section_name == 'generated_files':
            section_lines.extend(self._generate_files_section(section_config, context))
        
        return section_lines
    
    def _should_include_section(self, section_config: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a section should be included based on conditions."""
        # Always include required sections
        if section_config.get('required', False):
            return True
        
        # Check condition if specified
        condition = section_config.get('condition')
        if condition:
            return self._evaluate_condition(condition, context)
        
        return True
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string against the context."""
        try:
            # Simple condition evaluation for common cases
            if condition in context:
                value = context[condition]
                return bool(value) and (not isinstance(value, (dict, list)) or len(value) > 0)
            
            # Handle more complex conditions
            if ' > ' in condition:
                key, threshold = condition.split(' > ')
                return context.get(key.strip(), 0) > int(threshold.strip())
            
            return False
        except:
            return False
    
    # Section generation methods (simplified versions)
    def _generate_dataset_overview_section(self, section_config: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate dataset overview section."""
        lines = []
        template = section_config.get('template', [])
        
        for dataset_name, dataset_info in context.get('datasets', {}).items():
            dataset_context = {
                'dataset_name': dataset_name,
                'total_records': dataset_info.get('total_records', 0),
                'columns': dataset_info.get('columns', 0)
            }
            
            for line_template in template:
                try:
                    formatted_line = line_template.format(**dataset_context)
                    lines.append(formatted_line)
                except (KeyError, ValueError):
                    lines.append(line_template)
        
        return lines
    
    def _generate_column_analysis_section(self, section_config: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate column analysis scope section."""
        lines = []
        template = section_config.get('template', [])
        
        # Add main template lines
        for line_template in template:
            try:
                formatted_line = line_template.format(**context)
                lines.append(formatted_line)
            except (KeyError, ValueError):
                lines.append(line_template)
        
        return lines
    
    def _generate_demographics_section(self, section_config: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate patient demographics section."""
        lines = []
        template = section_config.get('template', [])
        age_stats = context.get('age_statistics', {})
        
        for line_template in template:
            try:
                formatted_line = line_template.format(**age_stats)
                lines.append(formatted_line)
            except (KeyError, ValueError):
                lines.append(line_template)
        
        return lines
    
    def _generate_category_distributions_section(self, section_config: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate category distributions section."""
        lines = []
        category_results = context.get('category_results', {})
        
        for category_name, value_counts in category_results.items():
            if value_counts is not None and len(value_counts) > 0:
                display_name = category_name.replace('_', ' ').title()
                
                # Create context for this specific category
                category_context = {
                    'display_name': display_name,
                    'total_categories': len(value_counts),
                    'most_common': value_counts.index[0],
                    'most_common_count': value_counts.iloc[0],
                    'has_valid_data': True
                }
                
                # Add section header
                title_template = section_config.get('title', '')
                try:
                    formatted_title = title_template.format(**category_context)
                    lines.extend(["", formatted_title])
                except (KeyError, ValueError):
                    lines.extend(["", f"{display_name.upper()} DISTRIBUTION:"])
                
                # Add main template lines
                template = section_config.get('template', [])
                for line_template in template:
                    try:
                        formatted_line = line_template.format(**category_context)
                        lines.append(formatted_line)
                    except (KeyError, ValueError):
                        lines.append(line_template)
                
                # Add top categories using item template
                max_items = section_config.get('max_items', 5)
                item_template = section_config.get('item_template', '')
                
                if item_template:
                    for i, (category, count) in enumerate(value_counts.head(max_items).items(), 1):
                        percentage = (count / value_counts.sum()) * 100
                        item_context = {
                            'index': i,
                            'category': category,
                            'count': count,
                            'percentage': percentage
                        }
                        
                        try:
                            formatted_line = item_template.format(**item_context)
                            lines.append(formatted_line)
                        except (KeyError, ValueError):
                            lines.append(f"    {i}. {category}: {count:,} ({percentage:.1f}%)")
        
        return lines
    
    def _generate_combined_statistics_section(self, section_config: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate combined statistics section."""
        lines = []
        combined_results = context.get('combined_results', {})
        
        if not combined_results:
            return []
        
        for stat_name, stat_data in combined_results.items():
            if stat_data is not None and len(stat_data) > 0:
                display_name = stat_name.replace('_', ' ').title()
                lines.extend(["", f"{display_name.upper()}:"])
                
                total_items = stat_data.sum()
                for category, count in stat_data.items():
                    percentage = (count / total_items) * 100 if total_items > 0 else 0
                    lines.append(f"    • {category}: {count:,} ({percentage:.1f}%)")
        
        return lines
    
    def _generate_files_section(self, section_config: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate generated files section."""
        lines = []
        
        # Add static files
        static_files = section_config.get('static_files', [])
        lines.extend(static_files)
        
        # Add category results files
        category_results = context.get('category_results', {})
        for category_name in category_results.keys():
            visualization_name = category_name.lower().replace(' ', '_')
            display_name = category_name.replace('_', ' ')
            lines.append(f"  {visualization_name}_distribution.png - {display_name} analysis")
        
        # Add combined results files
        combined_results = context.get('combined_results', {})
        for stat_name in combined_results.keys():
            stat_name_display = stat_name.replace('_', ' ')
            lines.append(f"  {stat_name}_combined_distribution.png - {stat_name_display} analysis")
        
        # Add stratified analysis files
        stratified_results = context.get('stratified_results', {})
        for analysis_name in stratified_results.keys():
            analysis_display = analysis_name.replace('_', ' ')
            lines.append(f"  {analysis_name}_stratified_*.png - {analysis_display} analysis")
        
        return lines
    
    def _generate_stratified_analyses_section(self, section_config: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate the stratified analyses section."""
        lines = []
        stratified_results = context.get('stratified_results', {})
        
        if not stratified_results:
            return lines
        
        for analysis_name, analysis_data in stratified_results.items():
            config = analysis_data.get('config', {})
            results = analysis_data.get('results', {})
            
            description = config.get('description', analysis_name)
            target_col = results.get('target_column_name', 'unknown')
            stratify_col = results.get('stratify_column_name', 'unknown')
            total_records = results.get('total_records', 0)
            strata_counts = results.get('strata_counts', {})
            
            lines.append("")
            lines.append(f"{description}:")
            lines.append(f"  Analysis: {target_col} stratified by {stratify_col}")
            lines.append(f"  Total records: {total_records:,}")
            lines.append(f"  Strata found: {len(strata_counts)}")
            
            for strata_name, count in strata_counts.items():
                lines.append(f"    {strata_name}: {count:,} records")
        
        return lines

        
