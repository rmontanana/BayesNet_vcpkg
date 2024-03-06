# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] 2024-03-06

### Added

- Change _ascending_ hyperparameter to _order_ with these possible values _{"asc", "desc", "rand"}_, Default is _"desc"_.
- Add the _predict_single_ hyperparameter to control if only the last model created is used to predict in boost training or the whole ensemble (all the models built so far). Default is true.
- sample app to show how to use the library (make sample)

### Changed

- Change the library structure adding folders for each group of classes (classifiers, ensembles, etc).
- The significances of the models generated under the feature selection algorithm are now computed after all the models have been generated and an &alpha;<sub>t</sub> value is computed and assigned to each model.

## [1.0.3] 2024-02-25

### Added

- Voting / probability aggregation in Ensemble classes
- predict_proba method in Classifier
- predict_proba method in BoostAODE
- predict_voting parameter in BoostAODE constructor to use voting or probability to predict (default is voting)
- hyperparameter predict_voting to AODE, AODELd and BoostAODE (Ensemble child classes)
- tests to check predict & predict_proba coherence

## [1.0.2] - 2024-02-20

### Fixed

- Fix bug in BoostAODE: do not include the model if epsilon sub t is greater than 0.5
- Fix bug in BoostAODE: compare accuracy with previous accuracy instead of the first of the ensemble if convergence true

## [1.0.1] - 2024-02-12

### Added

- Notes in Classifier class
- BoostAODE: Add note with used features in initialization with feature selection
- BoostAODE: Add note with the number of models
- BoostAODE: Add note with the number of features used to create models if not all features are used
- Test version number in TestBayesModels
- Add tests with feature_select and notes on BoostAODE

### Fixed

- Network predict test
- Network predict_proba test
- Network score test
