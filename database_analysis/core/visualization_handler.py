#!/usr/bin/env python3
"""
Visualization Handler - Handles all plotting and visualization tasks
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from .base_analyzer import BaseAnalyzer
from .chart_registry import ChartRegistry
from .constants import ChartType, AGE_DISTRIBUTION_FILE


class VisualizationHandler(BaseAnalyzer):
    """Handles all visualization and plotting operations."""
    
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
        self._setup_plotting()
        self._setup_color_scheme()
        self.chart_registry = ChartRegistry()
        self._register_default_charts()

    def _register_default_charts(self):
        """Register built-in chart renderers."""
        self.chart_registry.register(ChartType.CATEGORICAL, self.create_categorical_distribution_plot)
        self.chart_registry.register(ChartType.AGE, self.create_age_distribution_plot)
        self.chart_registry.register(ChartType.SPLIT, self.create_data_splits_plot)
        self.chart_registry.register(ChartType.COMBINED, self.create_combined_statistics_plot)
        self.chart_registry.register(ChartType.STRATIFIED, self.create_stratified_distribution_plot)
        self.chart_registry.register(ChartType.STRATIFIED_COMPARISON, self.create_stratified_comparison_plot)

    def render_chart(self, chart_type: ChartType, **kwargs):
        """Render a chart via the registry (extensible entry point)."""
        return self.chart_registry.render(chart_type, **kwargs)
    
    def _setup_plotting(self):
        """Setup matplotlib for high-quality plots."""
        # Get visualization defaults
        viz_defaults = self.global_defaults.get('visualization_defaults', {})
        
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = viz_defaults.get('dpi', 300)
        plt.rcParams['savefig.dpi'] = viz_defaults.get('dpi', 300)
        plt.rcParams['font.family'] = viz_defaults.get('font_family', 'serif')
        plt.rcParams['font.size'] = viz_defaults.get('font_size', 12)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def _setup_color_scheme(self):
        """Setup consistent color scheme for all visualizations."""
        self.color_palette = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf',  # cyan
            '#aec7e8',  # light blue
            '#ffbb78',  # light orange
            '#98df8a',  # light green
            '#ff9896',  # light red
            '#c5b0d5',  # light purple
            '#c49c94',  # light brown
            '#f7b6d3',  # light pink
            '#c7c7c7',  # light gray
            '#dbdb8d',  # light olive
            '#9edae5'   # light cyan
        ]
    
    def _get_colors(self, n_colors: int) -> List[str]:
        """Get n colors from the consistent palette."""
        if n_colors <= len(self.color_palette):
            return self.color_palette[:n_colors]
        else:
            # If we need more colors, cycle through the palette
            return [self.color_palette[i % len(self.color_palette)] for i in range(n_colors)]
    
    def _create_readable_pie_chart(self, ax, data: pd.Series, title: str) -> None:
        """Create pie chart with external legend for small percentages."""
        # Get minimum percentage threshold from config
        viz_defaults = self.global_defaults.get('visualization_defaults', {})
        min_percentage = viz_defaults.get('pie_chart_min_percentage', 3.0)
        
        # Calculate percentages
        percentages = (data / data.sum()) * 100
        
        # Separate large and small slices
        large_mask = percentages >= min_percentage
        large_data = data[large_mask]
        small_data = data[~large_mask]
        
        colors = self._get_colors(len(data))
        large_colors = [colors[i] for i in range(len(data)) if large_mask.iloc[i]]
        
        if len(large_data) > 0:
            # Create pie chart for large slices only
            def autopct_format(pct):
                return f'{pct:.1f}%' if pct >= min_percentage else ''
            
            wedges, texts, autotexts = ax.pie(
                large_data.values, 
                labels=large_data.index,
                autopct=autopct_format,
                colors=large_colors,
                startangle=90,
                pctdistance=0.85,
                textprops={'fontsize': 10}
            )
            
            # Style the percentage text to match bar chart style
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('normal')
                autotext.set_fontsize(10)
            
            # Style the labels to match bar chart style
            for text in texts:
                text.set_fontsize(10)
                text.set_fontweight('normal')
        
        # Use consistent title styling
        ax.set_title(title, fontsize=12, fontweight='normal', pad=20)
        
        # Add legend for small percentages if any
        if len(small_data) > 0:
            legend_labels = [f'{label}: {percentages[label]:.1f}%' 
                           for label in small_data.index]
            legend_title = f'Other categories (under {min_percentage:.0f}%):'
            # Position the legend below the pie chart
            ax.text(0.5, -0.15, legend_title + '\n' + '\n'.join(legend_labels),
                   transform=ax.transAxes, va='top', ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
                   fontsize=9)
    
    def create_categorical_distribution_plot(self, category_name: str, value_counts: pd.Series, use_log_scale: bool = False) -> Optional[Path]:
        """Create and save categorical distribution visualization."""
        if len(value_counts) == 0:
            self.logger.warning(f"No data to plot for {category_name}")
            return None
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Get visualization defaults
        viz_defaults = self.global_defaults.get('visualization_defaults', {})
        max_bar_categories = viz_defaults.get('max_bar_categories', 10)
        max_pie_categories = viz_defaults.get('max_pie_categories', 8)
        
        # Bar plot with consistent colors
        top_categories = value_counts.head(max_bar_categories)
        colors = self._get_colors(len(top_categories))
        bars = ax1.bar(range(len(top_categories)), top_categories.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Categories', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.set_title(f'{category_name.replace("_", " ").title()} Distribution', fontsize=12, fontweight='normal')
        ax1.set_xticks(range(len(top_categories)))
        ax1.set_xticklabels(top_categories.index, rotation=45, ha='right', fontsize=10)
        
        if use_log_scale:
            ax1.set_yscale('log')
        
        # Add value labels on bars with consistent styling
        for i, (bar, value) in enumerate(zip(bars, top_categories.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:,}', ha='center', va='bottom', fontsize=10, fontweight='normal')
        
        # Pie chart with improved readability
        if len(value_counts) <= max_pie_categories:
            self._create_readable_pie_chart(
                ax2, 
                value_counts, 
                f'{category_name.replace("_", " ").title()} Proportions'
            )
        else:
            # Show top categories in pie chart
            top_n = value_counts.head(max_pie_categories - 1)
            others = value_counts.iloc[max_pie_categories - 1:].sum()
            
            pie_data = pd.concat([top_n, pd.Series([others], index=['Others'])])
            
            self._create_readable_pie_chart(
                ax2,
                pie_data,
                f'{category_name.replace("_", " ").title()} Proportions (Top {max_pie_categories - 1} + Others)'
            )
        
        plt.tight_layout()
        
        # Save visualization
        visualization_name = category_name.lower().replace(' ', '_')
        output_file = self.output_dir / f"{visualization_name}_distribution.png"
        dpi = viz_defaults.get('dpi', 300)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_age_distribution_plot(self, age_data: pd.Series) -> Optional[Path]:
        """Create age distribution visualization."""
        if len(age_data) == 0:
            self.logger.warning("No age data to plot")
            return None
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Histogram with consistent color
        color = self.color_palette[0]  # Use first color from palette
        axes[0].hist(age_data, bins=20, alpha=0.8, edgecolor='black', color=color, linewidth=0.5)
        axes[0].set_xlabel('Age (years)', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        axes[0].set_title('Age Distribution', fontsize=12, fontweight='normal')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        box_plot = axes[1].boxplot(age_data, patch_artist=True)
        box_plot['boxes'][0].set_facecolor(color)
        box_plot['boxes'][0].set_alpha(0.8)
        axes[1].set_ylabel('Age (years)', fontsize=10)
        axes[1].set_title('Age Distribution (Box Plot)', fontsize=12, fontweight='normal')
        axes[1].grid(True, alpha=0.3)
        
        # Age groups
        analysis_options = self.config.get('analysis_options', {})
        if 'age_groups' in analysis_options:
            age_config = analysis_options['age_groups']
            bins = age_config['bins']
            labels = age_config['labels']
            
            age_groups = pd.cut(age_data, bins=bins, labels=labels, right=False)
            group_counts = age_groups.value_counts()
            
            colors = self._get_colors(len(group_counts))
            bars = axes[2].bar(range(len(group_counts)), group_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[2].set_xlabel('Age Groups', fontsize=10)
            axes[2].set_ylabel('Count', fontsize=10)
            axes[2].set_title('Age Group Distribution', fontsize=12, fontweight='normal')
            axes[2].set_xticks(range(len(group_counts)))
            axes[2].set_xticklabels(group_counts.index, rotation=45, fontsize=10)
            
            # Add value labels on bars
            for bar, value in zip(bars, group_counts.values):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:,}', ha='center', va='bottom', fontsize=10, fontweight='normal')
        
        plt.tight_layout()
        
        # Save plot
        viz_defaults = self.global_defaults.get('visualization_defaults', {})
        dpi = viz_defaults.get('dpi', 300)
        output_file = self.output_dir / AGE_DISTRIBUTION_FILE
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_data_splits_plot(self, dataset_name: str, split_counts: pd.Series) -> Optional[Path]:
        """Create data splits visualization."""
        if len(split_counts) == 0:
            return None
        
        # Visualize split distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = self._get_colors(len(split_counts))
        bars = ax.bar(split_counts.index, split_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Data Split', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'Data Split Distribution - {dataset_name}', fontsize=12, fontweight='normal')
        
        # Add value labels
        for bar, value in zip(bars, split_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{value:,}', ha='center', va='bottom', fontsize=10, fontweight='normal')
        
        plt.tight_layout()
        
        # Save plot
        viz_defaults = self.global_defaults.get('visualization_defaults', {})
        dpi = viz_defaults.get('dpi', 300)
        output_file = self.output_dir / f"data_splits_{dataset_name}.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_combined_statistics_plot(self, data: pd.Series, title: str, filename_base: str) -> Optional[Path]:
        """Create visualization for combined statistics."""
        if len(data) == 0:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Bar chart with consistent colors
        colors = self._get_colors(len(data))
        bars = ax1.bar(range(len(data)), data.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_title(f'{title} - Counts', fontsize=12, fontweight='normal')
        ax1.set_xlabel('Categories', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.set_xticks(range(len(data)))
        ax1.set_xticklabels(data.index, rotation=45, ha='right', fontsize=10)
        
        # Add value labels on bars
        for bar, value in zip(bars, data.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(data.values)*0.01,
                    f'{int(value)}', ha='center', va='bottom', fontweight='normal', fontsize=10)
        
        # Pie chart with improved readability
        viz_defaults = self.global_defaults.get('visualization_defaults', {})
        max_pie_categories = viz_defaults.get('max_pie_categories', 8)
        
        if len(data) <= max_pie_categories and data.sum() > 0:
            self._create_readable_pie_chart(
                ax2,
                data,
                f'{title} - Distribution'
            )
        elif data.sum() > 0:
            # Create pie chart with top categories + Others
            top_n = data.head(max_pie_categories - 1)
            others_sum = data.iloc[max_pie_categories - 1:].sum()
            
            if others_sum > 0:
                pie_data = pd.concat([top_n, pd.Series([others_sum], index=['Others'])])
            else:
                pie_data = top_n
            
            self._create_readable_pie_chart(
                ax2,
                pie_data,
                f'{title} - Distribution (Top {len(top_n)} + Others)'
            )
        else:
            ax2.text(0.5, 0.5, 'No data available\nfor pie chart', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save with high quality
        filename = f"{filename_base}_combined_distribution.png"
        output_file = self.output_dir / filename
        dpi = viz_defaults.get('dpi', 300)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file
    
    def create_stratified_distribution_plot(self, stratified_data: Dict[str, Any], 
                                          filename_base: str) -> Path:
        """
        Create stratified distribution visualization.
        
        Args:
            stratified_data: Results from analyze_stratified_distribution
            filename_base: Base name for output file
            
        Returns:
            Path to the saved plot file
        """
        if not stratified_data or 'stratified_distributions' not in stratified_data:
            self.logger.warning("No stratified data available for plotting")
            return None
        
        viz_defaults = self.global_defaults.get('visualization_defaults', {})
        stratified_distributions = stratified_data['stratified_distributions']
        target_col = stratified_data['target_column_name']
        stratify_col = stratified_data['stratify_column_name']
        target_desc = stratified_data['target_description']
        stratify_desc = stratified_data['stratify_description']
        
        # Prepare data for visualization
        strata_names = list(stratified_distributions.keys())
        n_strata = len(strata_names)
        
        if n_strata == 0:
            self.logger.warning("No strata found in stratified data")
            return None
        
        # Get all unique categories across all strata and determine global order
        all_categories = set()
        for strata_dist in stratified_distributions.values():
            all_categories.update(strata_dist.keys())
        
        # Add categories from overall distribution to ensure consistency
        overall_dist = stratified_data['overall_distribution']
        all_categories.update(overall_dist.keys())
        
        # Sort categories by their overall frequency (descending) to maintain consistent order
        category_order = sorted(all_categories, 
                              key=lambda x: overall_dist.get(x, 0), 
                              reverse=True)
        
        # Create figure with subplots for each stratum + overall comparison
        fig_width = max(12, 4 * min(n_strata + 1, 4))  # Adjust width based on number of strata
        fig_height = max(8, 6 if n_strata <= 3 else 12)
        
        # Calculate subplot layout
        if n_strata <= 3:
            n_cols = n_strata + 1  # +1 for overall distribution
            n_rows = 1
        else:
            n_cols = 3
            n_rows = (n_strata + 2) // 3  # +1 for overall, +2 for ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_strata > 1 else [axes] if n_strata == 1 else [[axes]]
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if n_strata > 1 else [axes]
        
        # Plot each stratum
        for i, (strata_name, strata_dist) in enumerate(stratified_distributions.items()):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            
            # Use the same category order as established above
            categories = [cat for cat in category_order if cat in strata_dist]
            counts = [strata_dist[cat] for cat in categories]
            
            if counts:
                # Create bar chart
                colors = [self.color_palette[j % len(self.color_palette)] for j in range(len(categories))]
                bars = ax.bar(range(len(categories)), counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                           f'{count}', ha='center', va='bottom', fontsize=10)
                
                # Customize chart
                ax.set_title(f'{target_desc}\n({strata_name}: {stratified_data["strata_counts"][strata_name]} records)', 
                           fontsize=11, fontweight='bold')  # Reduced from 12 to 11
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
                ax.set_ylabel('Count', fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # Set y-axis limit to accommodate labels
                ax.set_ylim(0, max(counts) * 1.1)
            else:
                ax.text(0.5, 0.5, f'No data\nfor {strata_name}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        
        # Plot overall distribution in the last subplot
        if len(axes_flat) > n_strata:
            overall_ax = axes_flat[n_strata]
            overall_dist = stratified_data['overall_distribution']
            
            if overall_dist:
                # Use the same category order as established above
                categories = [cat for cat in category_order if cat in overall_dist]
                counts = [overall_dist[cat] for cat in categories]
                
                colors = [self.color_palette[j % len(self.color_palette)] for j in range(len(categories))]
                bars = overall_ax.bar(range(len(categories)), counts, color=colors, alpha=0.8, 
                                    edgecolor='black', linewidth=0.5)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    overall_ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                                  f'{count}', ha='center', va='bottom', fontsize=10)
                
                overall_ax.set_title(f'{target_desc}\n(Overall: {stratified_data["total_records"]} records)', 
                                   fontsize=11, fontweight='bold')  # Reduced from 12 to 11
                overall_ax.set_xticks(range(len(categories)))
                overall_ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
                overall_ax.set_ylabel('Count', fontsize=11)
                overall_ax.grid(True, alpha=0.3)
                overall_ax.set_ylim(0, max(counts) * 1.1)
        
        # Hide any unused subplots
        for i in range(n_strata + 1, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Add main title with dynamic positioning
        title_y_position = 0.95 if n_rows == 1 else 0.97
        top_adjust = 0.85 if n_rows == 1 else 0.90
        
        fig.suptitle(f'Stratified Analysis: {target_desc} by {stratify_desc}', 
                    fontsize=16, fontweight='bold', y=title_y_position)
        
        plt.tight_layout()
        plt.subplots_adjust(top=top_adjust)  # Adjust spacing based on layout
        
        # Save plot
        filename = f"{filename_base}_stratified_{target_col}_by_{stratify_col}.png"
        output_file = self.output_dir / filename
        dpi = viz_defaults.get('dpi', 300)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file

    def create_stratified_comparison_plot(
        self,
        stratified_data: Dict[str, Any],
        filename_base: str,
        normalize: bool = False
    ) -> Optional[Path]:
        """
        Create a single grouped-bar comparison plot for a stratified distribution.
        
        Args:
            stratified_data: Output from analyze_stratified_distribution
            filename_base: Base filename for output
            normalize: If True, convert counts to percentages per stratum
        """
        if not stratified_data or 'stratified_distributions' not in stratified_data:
            self.logger.warning("No stratified data available for comparison plot")
            return None

        distributions = stratified_data['stratified_distributions']
        if not distributions:
            self.logger.warning("Empty stratified distributions for comparison plot")
            return None

        # Build a DataFrame with categories as rows and strata as columns
        categories = set()
        for dist in distributions.values():
            categories.update(dist.keys())
        if not categories:
            self.logger.warning("No categories found for comparison plot")
            return None

        strata = list(distributions.keys())
        data = pd.DataFrame(0, index=sorted(categories), columns=strata)
        for strata_name, dist in distributions.items():
            for cat, count in dist.items():
                data.at[cat, strata_name] = count

        if normalize:
            data = data.div(data.sum(axis=0).replace(0, 1), axis=1) * 100

        fig, ax = plt.subplots(figsize=(max(10, len(strata) * 2.5), 8))
        x = np.arange(len(data.index))
        width = 0.8 / max(len(strata), 1)

        colors = self._get_colors(len(strata))
        for i, strata_name in enumerate(strata):
            offsets = x + (i - (len(strata) - 1) / 2) * width
            bars = ax.bar(offsets, data[strata_name].values, width, label=strata_name, color=colors[i], edgecolor='black', linewidth=0.4, alpha=0.85)
            for bar, val in zip(bars, data[strata_name].values):
                if val > 0:
                    label = f"{val:.1f}%" if normalize else f"{int(val)}"
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom', fontsize=9)

        target_desc = stratified_data.get('target_description', stratified_data.get('target_column_name', 'Target'))
        stratify_desc = stratified_data.get('stratify_description', stratified_data.get('stratify_column_name', 'Strata'))
        ax.set_title(f"{target_desc} by {stratify_desc} (Grouped)", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Percentage' if normalize else 'Count', fontsize=11)
        ax.legend(title=stratify_desc, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        viz_defaults = self.global_defaults.get('visualization_defaults', {})
        dpi = viz_defaults.get('dpi', 300)
        suffix = "_pct" if normalize else ""
        output_file = self.output_dir / f"{filename_base}_grouped{suffix}.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_file
