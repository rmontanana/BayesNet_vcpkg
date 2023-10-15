# BayesNet

Bayesian Network Classifier with libtorch from scratch

## 0. Setup

Before compiling BayesNet.

### boost library

[Getting Started](<https://www.boost.org/doc/libs/1_83_0/more/getting_started/index.html>)

The best option is install the packages that the Linux distribution have in its repository. If this is the case:

```bash
sudo dnf install boost-devel
```

If this is not possible and the compressed packaged is installed, the following environment variable has to be set:

```bash
export BOOST_ROOT=/path/to/library/
```

### libxlswriter

```bash
cd lib/libxlsxwriter
make
make install DESTDIR=/home/rmontanana/Code PREFIX=
```

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
