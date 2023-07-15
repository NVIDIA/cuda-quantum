# Welcome to the CUDA Quantum Python API

## Installing

Programmers may use pip to handle the installation of the CUDA Quantum Python
API and any of the dependencies needed to do so. This may be done either from
source or through our binary wheel distribution. We currently support the
following operating systems: Linux.

### Pip Wheels

Programmers may install CUDA Quantum via the command line:

```console
pip install cuda-quantum
```

### Pip install from source

You may also install from source as follows:

```console
git clone https://github.com/NVIDIA/cuda-quantum.git
cd cuda-quantum
pip install . --user
```

This will install any dependencies and build the necessary pieces of the
CUDA Quantum repository through cmake. It will then install the
`cuda-quantum` python package through your pip package manager.

### Build entirely from source

If you would like to avoid the use of pip, or you want to build the entire C++ API
with the Python API, follow the [installation instructions][official_install]. You
may then export your python path to point to the installation directory:

```console
export PYTHONPATH=$PYTHONPATH:/path/to/cudaq/install
```

[official_install]: https://nvidia.github.io/cuda-quantum/latest/install.html

## Getting started

You should now be able to import CUDA Quantum and start building quantum programs
in Python!

```console
import cudaq

kernel = cudaq.make_kernel()
```

To see more examples, [go to][python_examples], or check out the
[Python API reference][python_api_reference].

[python_examples]: https://nvidia.github.io/cuda-quantum/latest/using/python.html
[python_api_reference]: https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html
