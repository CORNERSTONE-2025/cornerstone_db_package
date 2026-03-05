Cornerstone Module
==================

Summary
-------
The `cornerstone.py` module provides a high-level utility class `Cornerstone` and a set of helper functions to manage and preprocess solar physics datasets (SHARP parameters). It includes database synchronization (SQLite/MySQL), extraction of time-series datasets from a database and saving them to HDF5, dataset preprocessing and splitting, model training and testing wrappers that integrate with the repository's ML utilities, and a number of utility helpers for working with Google Drive, SQL dumps and conversions.

This documentation summarizes the public API, key helper functions, usage examples, dependencies, and notes about environment/setup.

Public API
----------

Cornerstone class
~~~~~~~~~~~~~~~~~~
Purpose
: Central orchestration class to manage dataset extraction, preprocessing/splitting, and model training/testing workflows.

Constructor
: __init__(db_config, output_folder="./dataset_compilation/data", project_name="Cornerstone")

- db_config: dict-like config for database access and metadata paths (expected keys include `local_db_path`, `local_metadata_path`, `remote_metadata_url`, and for MySQL `user`, `password`, `host`, `port`, `database`).
- output_folder: where generated datasets, splits and logs will be stored.
- project_name: name used by ClearML tasks (if ClearML is available).

Main methods
- db_update():
  - Synchronize local DB with remote metadata/dumps. Handles first-time full-dump download and incremental updates. Uses helper functions such as `_fetch_remote_metadata`, `_download_file`, `_create_local_db_from_dump`, and `_apply_update_to_local_db`.

- create_hdf5_from_db(label_type, data_version="clean", table_type="DEF", window_hours=24, buffer_minutes=6, sampling_interval=12, test_mode=False, test_size=500, random_seed=42, db_type="sqlite"):
  - Extract event-labelled time-series from a database, package them into consistent time windows (rounding to a sampling interval), and save to an HDF5 file.
  - Parameters control which tables/features to select, window sizes, sampling, and backend (SQLite or MySQL).
  - Output: HDF5 file saved to `output_folder` and a missing-data log CSV if relevant.

- create_dataset_splits(config):
  - Preprocess an HDF5 dataset using `SharpDataset` and related transforms (feature selection, missingness filters, temporal filters, CMASK filter), then produce train/val/test splits (via `SharpDatasetSplitter`).
  - `config` can be a dict or SimpleNamespace with expected attributes (INPUT_H5_FILE, TIMESTEPS, LABEL_TYPE, FEATURE_SELECTION_MODE, etc.).

- train_models(train_config_path=None):
  - High-level wrapper that loads model and training config(s), prepares `PathManager`, `ModelManager`, `MetricsManager`, and runs cross-validation training via the repository training utilities (`run_cv_training`).
  - `train_config_path` allows overriding the default config module.

- test_model(test_config_path=None):
  - High-level wrapper for final test set evaluation. Prepares managers and calls `run_test` from the repo's ML utilities.

Helpers and utilities (selected)
--------------------------------
These functions are module-level helpers used by the class methods. Most are prefixed with `_` and intended for internal use, but they're documented here for clarity.

- _select_db_config(self, data_version, table_type)
  - Returns `(table_name, numeric_features)` depending on `data_version`.

- _save_hdf5(self, path, results, numeric_features, data_version, table_name, label_suffix, all_results)
  - Writes arrays and metadata into an HDF5 file (X, Y, HARPNUM, timestamps, labels, and feature names as attributes).

- _save_missing_log(self, missing_data)
  - Save a CSV describing events with missing data.

- _process_events_parallel(args) and _process_batch_parallel(self, event_rows, batch_data, ...)
  - Internal multiprocessing helpers that split event processing across CPU cores and reassemble results.

- _generate_experiment_name(self, config)
  - Builds a compact, descriptive name for the experiment based on the config (feature selection mode, missingness filtering, CMASK, label type, split strategy).

- _create_output_directories(self, config, experiment_name)
  - Create `splits`, `plots` and `config` folders under the output directory for a given experiment.

- _save_config(self, config, output_dir)
  - Persist text representation of preprocessing/splitting configuration.

- _print_config_summary(self, config)
  - Human-readable summary of important config fields printed during preprocessing.

- _load_local_metadata(self) and _save_local_metadata(self, last_sync_date)
  - Read/write JSON metadata tracking the last synchronization and local DB path. `_fetch_remote_metadata` downloads remote metadata JSON using `gdown`.

- _download_file(self, folder_url, dest_folder="db")
  - Wrapper around `gdown.download_folder` to download a single file and return its path.

- _create_local_db_from_dump(self, dump_path: str, db_type="sqlite", db_path=None)
  - Restore a SQL dump into SQLite or MySQL. For SQLite, the module includes streaming conversion from MySQL SQL -> SQLite-compatible SQL before import. For MySQL, it streams into the server (uses `gunzip -c` if compressed). This function is careful to avoid large memory usage but requires external tools like `sqlite3` and `mysql`/`gunzip` available on PATH.

- _apply_update_to_local_db(self, db_path: str, bin_path: str)
  - Apply an incremental SQL script to the local DB using `executescript()` in SQLite. Raises helpful errors when files are missing.

- _get_or_create_task(self, key)
  - Small wrapper to create or reuse a ClearML `Task` and store it in `self.tasks`.

- _generate_plots(summary_metrics, OUTPUT_CONFIG, path_manager, TASK_CONFIG)
  - Helper to create and save plots if enabled in `OUTPUT_CONFIG`.

Notes and Requirements
----------------------
Dependencies (non-exhaustive):
- Python 3.8+
- pandas, numpy
- h5py
- sklearn
- tqdm
- clearml (optional — only for task logging; code checks for it)
- mysql-connector-python (if using MySQL backend)
- gdown (for Google Drive downloads)
- sqlite3 CLI available on PATH (used by `_create_local_db_from_dump` streaming import)
- gunzip/mysql CLI (for streaming compressed SQL into MySQL)
- Repository-local module `sharp_data` which provides `SharpDataset`, transforms and splitters.

Important behavior notes
- The module supports both SQLite and MySQL; many operations will behave differently depending on `db_type` argument.
- The file contains streaming conversion from MySQL dumps to SQLite SQL. This conversion is heuristic and may require manual inspection for complex schema differences.
- Several helper functions read or write temporary files and call external commands (sqlite3, mysql). Ensure these are available and you have correct permissions.
- Many long code blocks and I/O operations were summarized in the provided file attachment; all public method signatures and the intended behavior are included above, but the implementation details (edge-case handling) are not fully reproduced here.

Usage examples
--------------
Instantiate and run a single pipeline step (very small example):

```python
from cornerstone.cornerstone import Cornerstone

# Minimal db_config example for SQLite
db_config = {
    "local_db_path": "/bigdisk/daniele/cornerstone_module/db/cornerstone_full.db",
    "local_metadata_path": "/bigdisk/daniele/cornerstone_module/db/metadata.json",
    "remote_metadata_url": "https://drive.google.com/drive/folders/..."
}

cs = Cornerstone(db_config=db_config, output_folder="/tmp/cornerstone_out")

# Create HDF5 dataset (this will read the DB and write an HDF5 file)
cs.create_hdf5_from_db(label_type="M", data_version="clean", table_type="DEF", db_type="sqlite")

# Preprocess and create splits from an HDF5 file
config = {
    "INPUT_H5_FILE": "/tmp/cornerstone_out/sharp_dataset_db_SHARPC_clean_DEF_w24h_s12min.h5",
    "TIMESTEPS": 120,
    "LABEL_TYPE": "M",
    "FEATURE_SELECTION_MODE": "direct",
    "DIRECT_FEATURES": [/* feature list */],
    "MISSINGNESS_FILTER_MODE": "single_window",
    "MAX_NAN_RATIO": 0.2,
    "SPLITTING_STRATEGY": "random",
    "TEST_SIZE": 0.2,
    "N_FOLDS": 5,
    "OUTPUT_BASE_DIR": "/tmp/cornerstone_out",
    "SAVE_PLOTS": False,
    "VERBOSE": True
}
cs.create_dataset_splits(config)

# Train models (uses repository model/training utilities)
cs.train_models()

# Test models
cs.test_model()
```

Suggested next steps
--------------------
- If you want inline, complete docstrings added to every method and helper in `cornerstone.py`, I can open the file and insert Google-style docstrings for every public method and main helpers (automatically generated and consistent). Reply `add docstrings` and I'll apply those edits.

- If you'd like HTML documentation (Sphinx / pydoc / MkDocs), tell me which format you prefer and I will:
  - create docstrings (if not present) and
  - generate a small Sphinx config or pdoc output and a README page linking to the generated HTML.

- I can also add a small unit test that imports `Cornerstone` and runs a no-op smoke test to ensure imports and docstrings are valid.

Contact / assumptions
---------------------
- This documentation was generated from the supplied `cornerstone.py` (attachment) which included omitted sections. I focused on the public API and helper functions visible in the file. For highly accurate method parameter descriptions and examples I can insert or refine docstrings directly in the source file upon request.


