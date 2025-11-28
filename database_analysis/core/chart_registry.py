"""
Simple chart registry to decouple chart selection from rendering logic.

New chart types can be added by registering a renderer function without
modifying call sites or adding conditional logic.
"""

from typing import Callable, Dict, Any

from .constants import ChartType


class ChartRegistry:
    """Registry mapping chart types to renderer callables."""

    def __init__(self):
        self._renderers: Dict[str, Callable[..., Any]] = {}

    def register(self, chart_type: ChartType, renderer: Callable[..., Any]) -> None:
        """Register a renderer for a chart type."""
        self._renderers[str(chart_type)] = renderer

    def render(self, chart_type: ChartType, **kwargs):
        """Render a chart by delegating to the registered renderer."""
        key = str(chart_type)
        if key not in self._renderers:
            raise ValueError(f"No renderer registered for chart type: {chart_type}")
        return self._renderers[key](**kwargs)
