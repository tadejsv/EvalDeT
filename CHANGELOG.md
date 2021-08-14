# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.1] - 2021-08-14

### Added

* HOTA metrics can now be computed, using the original matching algorithm from the paper (gives similar results, but does not correspond to TrackEval's implementation)

## [0.1.0] - 2021-08-07

### Added

* `Tracks` class, which represents the tracking data (detections), and provides loading functions to read from common MOT data formats
* `compute_mot_metrics` function, which can calculate MOT metrics (for now CLEARMOT and ID metrics, HOTA coming soon, ALTA in the pipeline)
* API documentation hosted on readthedocs
* Pypi package
