# <img src="logo.png" alt="logo" width="50"/>  BayesNet

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=flat&logo=c%2B%2B&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](<https://opensource.org/licenses/MIT>)
![Gitea Release](https://img.shields.io/gitea/v/release/rmontanana/bayesnet?gitea_url=https://gitea.rmontanana.es:3000)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/cf3e0ac71d764650b1bf4d8d00d303b1)](https://app.codacy.com/gh/Doctorado-ML/BayesNet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=rmontanana_BayesNet&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=rmontanana_BayesNet)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=rmontanana_BayesNet&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rmontanana_BayesNet)
![Gitea Last Commit](https://img.shields.io/gitea/last-commit/rmontanana/bayesnet?gitea_url=https://gitea.rmontanana.es:3000&logo=gitea)
[![Static Badge](https://img.shields.io/badge/Coverage-97,2%25-green)](html/index.html)

Bayesian Network Classifiers using libtorch from scratch

## Dependencies

The only external dependency is [libtorch](https://pytorch.org/cppdocs/installing.html) which can be installed with the following commands:

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zips
```

## Setup

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

## Diagrams

### UML Class Diagram

![BayesNet UML Class Diagram](diagrams/BayesNet.svg)

### Dependency Diagram

![BayesNet Dependency Diagram](diagrams/dependency.svg)
