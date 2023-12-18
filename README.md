# BayesNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bayesian Network Classifier with libtorch from scratch

## 0. Setup

Before compiling BayesNet.

### MPI

In Linux just install openmpi & openmpi-devel packages. Only cmake can't find openmpi install (like in Oracle Linux) set the following variable:

```bash
export MPI_HOME="/usr/lib64/openmpi"
```

In Mac OS X, install mpich with brew and if cmake doesn't find it, edit mpicxx wrapper to remove the ",-commons,use_dylibs" from final_ldflags

```bash
vi /opt/homebrew/bin/mpicx
```

### boost library

[Getting Started](<https://www.boost.org/doc/libs/1_83_0/more/getting_started/index.html>)

The best option is install the packages that the Linux distribution have in its repository. If this is the case:

```bash
sudo dnf install boost-devel
```

If this is not possible and the compressed packaged is installed, the following environment variable has to be set pointing to the folder where it was unzipped to:

```bash
export BOOST_ROOT=/path/to/library/
```

In some cases, it is needed to build the library, to do so:

```bash
cd /path/to/library
mkdir own
./bootstrap.sh --prefix=/path/to/library/own
./b2 install
export BOOST_ROOT=/path/to/library/own/
```

Don't forget to add the export BOOST_ROOT statement to .bashrc or wherever it is meant to be.

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
