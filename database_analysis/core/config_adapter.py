"""
Adapters for using refactored YAML database configs with the analysis framework.

The dataset creator refactor moved database definitions to YAML and changed the
shape of the configuration (files/field_mappings/value_mappings). This adapter
translates those configs into the analysis-friendly structure expected by the
legacy analyzers (datasets + column_mappings).
"""

from pathlib import Path
from typing import Dict, Any

from .constants import DEFAULT_DATASET_NAME


class DatabaseConfigAdapter:
    """Convert database YAML configs into analysis-ready configurations."""

    def __init__(self, raw_config: Dict[str, Any], config_path: Path, data_directory: str):
        self.raw_config = raw_config or {}
        self.config_path = config_path
        self.data_directory = Path(data_directory or ".").resolve()

    def to_analysis_config(self) -> Dict[str, Any]:
        """Produce a configuration compatible with the analyzer components."""
        database_name = (
            self.raw_config.get("database_name")
            or self.raw_config.get("name")
            or self.config_path.stem
        )

        base_dir = self._get_base_directory()
        file_entry = self._get_primary_file()
        data_dir, filename = self._resolve_file_path(file_entry, base_dir)

        analysis_config: Dict[str, Any] = {
            "database_name": database_name,
            "data_directory": str(data_dir),
            "datasets": {
                DEFAULT_DATASET_NAME: {
                    "filename": filename,
                    "loading_params": self._build_loading_params(file_entry),
                }
            },
            "column_mappings": self._build_column_mappings(),
        }

        # Preserve human-readable description where possible
        if "description" in self.raw_config:
            analysis_config["description"] = self.raw_config["description"]

        return analysis_config

    def _get_primary_file(self) -> Dict[str, Any]:
        """Pick the first required file (fallback to first) or raise if none exist."""
        files = self.raw_config.get("files") or []
        if not files:
            raise ValueError("Database config is missing 'files' entries required for analysis")

        required_files = [f for f in files if f.get("required", False)]
        return required_files[0] if required_files else files[0]

    def _get_base_directory(self) -> Path:
        """Resolve base_directory from config relative to repo/data dir."""
        base_dir_cfg = self.raw_config.get("base_directory")
        if base_dir_cfg:
            candidate = Path(base_dir_cfg)
            if not candidate.is_absolute():
                # Prefer path relative to provided data_directory, else config file dir
                candidate = (self.data_directory / candidate).resolve()
            return candidate

        # Fallback: use the provided data directory
        return self.data_directory

    def _resolve_file_path(self, file_entry: Dict[str, Any], base_dir: Path):
        """Resolve the file path to a data directory and filename."""
        raw_path = file_entry.get("path") or file_entry.get("filename")
        if not raw_path:
            raise ValueError("File entry must contain 'path' or 'filename'")

        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()

        return candidate.parent, candidate.name

    def _build_loading_params(self, file_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Build loading params section with encoding fallbacks."""
        params: Dict[str, Any] = {}
        if "encoding" in file_entry:
            params["encoding"] = file_entry.get("encoding")
        if "delimiter" in file_entry:
            params["delimiter"] = file_entry.get("delimiter")
        if "fallback_encoding" in file_entry:
            params["fallback_encoding"] = file_entry.get("fallback_encoding")
        return params

    def _build_column_mappings(self) -> Dict[str, Any]:
        """
        Convert field_mappings from the database config into column_mappings.

        The analyzer expects the shape:
        {
            "pathology": {"dataset": "primary", "column": "<source_col>", "description": "..."},
            ...
        }
        """
        mappings = {}
        field_mappings = self.raw_config.get("field_mappings", {})

        for source_col, standard_name in field_mappings.items():
            mapping_name = str(standard_name).lower()
            mappings[mapping_name] = {
                "dataset": DEFAULT_DATASET_NAME,
                "column": source_col,
                "description": f"{standard_name} ({source_col})",
            }

        return mappings
