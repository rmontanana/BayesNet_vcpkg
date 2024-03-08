# BayesNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bayesian Network Classifiers using libtorch from scratch

## Installation

### Release

```bash
make release
make buildr
sudo make install
```

### Debug & Tests

```bash
make debug
make test
make coverage
```

### Sample app

After building and installing the release version, you can run the sample app with the following commands:

```bash
make sample
make sample fname=tests/data/glass.arff
```

## Models

### [BoostAODE](docs/BoostAODE.md)
