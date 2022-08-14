# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [0.1.2] - 2022-08-14

### Added

* More integration tests (though for some reason, there are discrepancies with official results at times) ([#12](https://github.com/sasp-ai/EvalDeT/pull/12))
* Filtering method for tracks (`filter_by_class`, `filter_by_conf`) ([#12](https://github.com/sasp-ai/EvalDeT/pull/12))
* General method for reading CSV file (`from_csv`), reading MOT gt files (`from_mot_gt`) ([#12](https://github.com/sasp-ai/EvalDeT/pull/12))


### Changed

* Version large files for integration tests with `git-lfs` ([#12](https://github.com/sasp-ai/EvalDeT/pull/12))
* All `Track` class methods for reading CSV files become just a wrapper around `from_csv` ([#12](https://github.com/sasp-ai/EvalDeT/pull/12))

## [0.1.1] - 2021-08-14

### Added

* HOTA metrics can now be computed, using the original matching algorithm from the paper (gives similar results, but does not correspond to TrackEval's implementation)

## [0.1.0] - 2021-08-07

### Added

* `Tracks` class, which represents the tracking data (detections), and provides loading functions to read from common MOT data formats
* `compute_mot_metrics` function, which can calculate MOT metrics (for now CLEARMOT and ID metrics, HOTA coming soon, ALTA in the pipeline)
* API documentation hosted on readthedocs
* Pypi package
