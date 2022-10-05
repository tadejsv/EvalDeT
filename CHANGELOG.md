# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [0.1.8] - 2022-10-05

## Changed

* Improve HOTA metrics to use less memory ([#23](https://github.com/tadejsv/EvalDeT/pull/23))
* Improve ID metrics to use less memory ([#23](https://github.com/tadejsv/EvalDeT/pull/23))

## [0.1.7] - 2022-10-02

## Fixed

* Fix typing for compatibility with Python 3.8 ([#22](https://github.com/tadejsv/EvalDeT/pull/22))

## [0.1.6] - 2022-09-29

## Changed

* Fix publishing with `hatch` ([#21](https://github.com/tadejsv/EvalDeT/pull/21))

## [0.1.5] - 2022-09-29

## Changed

* Fix correct 1-index frame handling for MOT-based files ([#20](https://github.com/tadejsv/EvalDeT/pull/20))
* Fix handling of "outside" detections in CVAT format ([#20](https://github.com/tadejsv/EvalDeT/pull/20))

* Switch to `hatch` for packaging ([#20](https://github.com/tadejsv/EvalDeT/pull/20))


## [0.1.4] - 2022-08-29

### Changed
* Performance improvement for HOTA metrics using sparse matrices ([#19](https://github.com/tadejsv/EvalDeT/pull/19))

## [0.1.3] - 2022-08-21

### Changed

* MOT Metrics are now computed using the `MOTMetrics` class, which enables efficient sharing of pre-computed IoU distances across metrics ([#17](https://github.com/tadejsv/EvalDeT/pull/12))
* Increase minimum Python version to 3.9  ([#16](https://github.com/tadejsv/EvalDeT/pull/16))
* Switch packaging system to Poetry  ([#16](https://github.com/tadejsv/EvalDeT/pull/16))

## [0.1.2] - 2022-08-14

### Added

* More integration tests (though for some reason, there are discrepancies with official results at times) ([#12](https://github.com/tadejsv/EvalDeT/pull/12))
* Filtering method for tracks (`filter_by_class`, `filter_by_conf`) ([#12](https://github.com/tadejsv/EvalDeT/pull/12))
* General method for reading CSV file (`from_csv`), reading MOT gt files (`from_mot_gt`) ([#12](https://github.com/tadejsv/EvalDeT/pull/12))


### Changed

* Version large files for integration tests with `git-lfs` ([#12](https://github.com/tadejsv/EvalDeT/pull/12))
* All `Track` class methods for reading CSV files become just a wrapper around `from_csv` ([#12](https://github.com/tadejsv/EvalDeT/pull/12))

## [0.1.1] - 2021-08-14

### Added

* HOTA metrics can now be computed, using the original matching algorithm from the paper (gives similar results, but does not correspond to TrackEval's implementation)

## [0.1.0] - 2021-08-07

### Added

* `Tracks` class, which represents the tracking data (detections), and provides loading functions to read from common MOT data formats
* `compute_mot_metrics` function, which can calculate MOT metrics (for now CLEARMOT and ID metrics, HOTA coming soon, ALTA in the pipeline)
* API documentation hosted on readthedocs
* Pypi package
