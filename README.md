# BayesNet

Bayesian Network Classifier with libtorch from scratch

## 0. Setup

### libxlswriter

Before compiling BayesNet.

```bash
cd lib/libxlsxwriter
make
sudo make install
```

It has to be installed in /usr/local/lib otherwise CMakeLists.txt has to be modified accordingly

Environment variable has to be set:

```bash
 export LD_LIBRARY_PATH=/usr/local/lib
 ```

### Release

```bash
make release
```

### Debug & Tests

```bash
make debug
```

## 1. Introduction
