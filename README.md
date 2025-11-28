# Mammography Database Analysis Framework

Lightweight framework for analyzing mammography datasets: loading, cleaning, auto-detecting categorical columns, producing summary statistics, visualizations and a text report.

**Python version:** 3.10

Quick overview
- Loads datasets defined in database config files (YAML/JSON).
- Auto-detects categorical columns and creates column mappings.
- Runs analyses: categorical distributions, age validation, stratified analyses, and combined statistics.
- Produces high-quality plots (matplotlib) and a plain-text report.

Installation

1. Create a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

Usage

- Analyze a database using a config file:

```powershell
python -m database_analysis.mammography_analyzer <config_file> <data_directory> [output_directory]
```

- Or run by database name (the tool will try to resolve YAML config files):

```powershell
python -m database_analysis.mammography_analyzer --database cbis-ddsm --data-root .
```

Outputs
- Visualizations and reports are saved to the configured `output` folder (defaults to `database_descriptions/<db-name>/results` when using database name resolution).
- Key files: `analysis_report.txt`, `basic_statistics.json`, `age_distribution.png`, and per-category PNG files.

Notes
- The code expects Python 3.10 and uses `pandas`, `numpy`, `matplotlib`, and `PyYAML`.
- If you rely on other dev tools (testing, docs), keep a separate `requirements-dev.txt`.
