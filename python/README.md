# Welcome to the CUDA Quantum Python API

## Installing 
Programmers may use pip to handle the installation of the CUDA Quantum Python API 
and any of the dependencies needed to do so. This may be done either from source or
through our binary wheel distribution. We currently support the following operating 
systems:
* Linux

### Pip Wheels
Under Construction! Not yet distributing wheels.
Programmers may install CUDA Quantum via the command line:
```
pip install cuda-quantum --user
```
Note the `--user` flag is required for users on Linux, to ensure the package
is installed in the local `site-packages` folder.

### Pip install from source
You may also install from source as follows:
```
git clone https://github.com/NVIDIA/cuda-quantum.git
cd cuda-quantum
pip install . --user
```
This will install any dependencies and build the necessary pieces of the CUDA Quantum 
repository through cmake. It will then install the 

### Build entirely from source
If you would like to avoid the use of pip, or you want to build the entire C++ API with the
Python API, follow the installation instructions here (TODO). You may then export your python
path to point to the installation directory:
```
export PYTHONPATH=$PYTHONPATH:/path/to/cudaq/install
```