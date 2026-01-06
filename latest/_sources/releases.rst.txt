************************
CUDA-Q Releases
************************

**latest**

The latest version of CUDA-Q is on the main branch of our `GitHub repository <https://github.com/NVIDIA/cuda-quantum>`__ 
and is also available as a Docker image. More information about installing the nightly builds can be found 
:doc:`here <using/install/install>`

- `Docker image (nightly builds) <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nightly/containers/cuda-quantum>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/latest>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples>`__

**0.13.0**

This release adds support for CUDA 13 and Python 3.13 and removes support for 
CUDA 11 and Python 3.10. It adds support for using the CUDA-Q QEC libraries 
for real-time decoding on Quantinuum backends, and adds support for submission 
to QCI backends. Check out the release notes below to learn about additional 
new content.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cudaq/0.13.0>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.13.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.13.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.13.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.13.0>`__.

**0.12.0**

This release contains a range of new features and performance improvements for 
the dynamics simulation and adds more tools for error correction applications. 
It introduces new CUDA-Q API for kernels that return values (`run`), and adds 
support for connecting to Quantum Machines backends. This release also contains
contributions from unitaryHACK. With this release, we now include support for
Python 3.13.

*Note*: Support for Python 3.10 will be removed in future releases.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cudaq/0.12.0>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.12.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.12.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.12.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.12.0>`__.

**0.11.0**

This release contains a range of ergonomic improvements and documentation updates.
It adds support for initializing qubits to have a given state for quantum hardware backends
and exposes a range of new configurations for different simulator backends. This release also
addresses some performance issues with the initial introduction of a general operator framework
in CUDA-Q. This required some breaking changes. We refer to the 
`release notes <https://github.com/NVIDIA/cuda-quantum/releases/0.11.0>`__ for further details.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cudaq/0.11.0>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.11.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.11.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.11.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.11.0>`__.

**0.10.0**

In this release we have added a range of tools for simulating noisy quantum systems.
It includes support for trajectory based noise simulation for our GPU-accelerated statevector 
and tensor network simulators, and a new sampling option for better `stim` performance. We have 
also expanded the range of noise models, and added an `apply_noise` "gate". This release adds 
the C++ support for the CUDA-Q `dynamics` backend (a master equation solver), as well as operator 
classes for fermionic, bosonic and custom operators. This release also includes support for 
submitting to `Pasqal <https://nvidia.github.io/cuda-quantum/0.10.0/using/backends/hardware/neutralatom.html#pasqal>`__ 
backends, and adds C++ support for 
`QuEra <https://nvidia.github.io/cuda-quantum/0.10.0/using/backends/hardware/neutralatom.html#quera-computing>`__ 
backends. Check out our `documentation <https://nvidia.github.io/cuda-quantum/0.10.0>`__, including 
new tutorials and examples, for more information.

*Note*: Support for CUDA 11 will be removed in future releases. Please update to CUDA 12.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cudaq/0.10.0>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.10.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.10.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.10.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.10.0>`__.

**0.9.1**

This release adds support for using 
`Amazon Braket <https://nvidia.github.io/cuda-quantum/0.9.1/using/backends/hardware.html#amazon-braket>`__ and 
`Infeqtion's Superstaq <https://nvidia.github.io/cuda-quantum/0.9.1/using/backends/hardware.html#infleqtion>`__ as backends.

Starting with this release, all C++ quantum kernels will be processed by the `nvq++` compiler regardless of whether 
they run on a simulator or on a quantum hardware backend. This change is largely non-breaking, but language constructs 
that are not officially supported within quantum kernels will now lead to a compilation error whereas previously they 
could be used when executing on a simulator only. The previous behavior can be forced by passing the `--library-mode` 
flag to the compiler. Please note that if you do so, however, the code will never be executable outside of a simulator 
and may not be supported even on simulators.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cudaq/0.9.1>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.9.1>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.9.1>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.9.1/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.9.1>`__.

**0.9.0**

We are very excited to share a new toolset added for modeling and manipulating the dynamics of physical systems. 
The new API allows to define and execute a time evolution under arbitrary operators. For more information, take 
a look at the `docs <https://nvidia.github.io/cuda-quantum/0.9.0/using/backends/dynamics.html>`__.
The 0.9.0 release furthermore includes a range of contribution to add new backends to CUDA-Q, including backends 
from `Anyon Technologies <https://nvidia.github.io/cuda-quantum/0.9.0/using/backends/hardware.html#anyon-technologies-anyon-computing>`__, 
`Ferimioniq <https://nvidia.github.io/cuda-quantum/0.9.0/using/backends/simulators.html#fermioniq>`__, and 
`QuEra Computing <https://nvidia.github.io/cuda-quantum/0.9.0/using/backends/hardware.html#quera-computing>`__, 
as well as updates to existing backends from `ORCA <https://nvidia.github.io/cuda-quantum/0.9.0/using/backends/hardware.html#orca-computing>`__ 
and `OQC <https://nvidia.github.io/cuda-quantum/0.9.0/using/backends/hardware.html#oqc>`__.
We hope you enjoy the new features - also check out our new notebooks and examples to dive into CUDA-Q.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cudaq/0.9.0>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.9.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.9.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.9.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.9.0>`__.

**0.8.0**

The 0.8.0 release adds a range of changes to improve the ease of use and performance with CUDA-Q. 
The changes listed below highlight some of what we think will be the most useful features and changes 
to know about. While the listed changes do not capture all of the great contributions, we would like 
to extend many thanks for every contribution, in particular those from external contributors.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cuda-quantum/0.8.0>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.8.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.8.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.8.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.8.0>`__.

**0.7.1**

The 0.7.1 release adds simulator optimizations with significant performance improvements and 
extends their functionalities. The `nvidia-mgpu` backend now supports user customization of the 
gate fusion level as controlled by the `CUDAQ_MGPU_FUSE` environment variable documented 
`here <https://nvidia.github.io/cuda-quantum/0.7.1/using/backends/simulators.html>`__.
It furthermore adds a range of bug fixes and changes the Python wheel installation instructions.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cuda-quantum/0.7.1>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.7.1>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.7.1>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.7.1/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.7.1>`__.

**0.7.0**

The 0.7.0 release adds support for using NVIDIA Quantum Cloud,
giving you access to our most powerful GPU-accelerated simulators even if you don't have an NVIDIA GPU.
With 0.7.0, we have furthermore greatly increased expressiveness of the Python and C++ language frontends. 
Check out our `documentation <https://nvidia.github.io/cuda-quantum/0.7.0/using/quick_start.html>`__ 
to get started with the new Python syntax support we have added, and `follow our blog <https://developer.nvidia.com/cuda-q>`__
to learn more about the new setup and its performance benefits.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cuda-quantum/0.7.0>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.7.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.7.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.7.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.7.0>`__.

**0.6.0**

The 0.6.0 release contains improved support for various HPC scenarios. We have added a
:ref:`plugin infrastructure <distributed-computing-with-mpi>` for connecting CUDA-Q 
with an existing MPI installation, and we've added a :ref:`new platform target <remote-mqpu-platform>` that distributes workloads across multiple virtual QPUs, 
each simulated by one or more GPUs.

Starting with 0.6.0, we are now also distributing 
:ref:`pre-built binaries <install-prebuilt-binaries>` for using CUDA-Q with C++.
The binaries are built against the `GNU C library <https://www.gnu.org/software/libc/>`__ 
version 2.28.
We've added a detailed :doc:`Building from Source <using/install/data_center_install>` guide to build these binaries for older `glibc` versions.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cuda-quantum/0.6.0>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.6.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.6.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.6.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.6.0>`__.

**0.5.0**

With 0.5.0 we have added support for quantum kernel execution on OQC and IQM backends. For more information, see :doc:`using/backends/hardware`.
CUDA-Q now allows to executing adaptive quantum kernels on quantum hardware backends that support it.
The 0.5.0 release furthermore improves the tensor network simulation tools and adds a matrix product state simulator, see :doc:`using/backends/simulators`.

Additionally, we are now publishing images for experimental features, which currently includes improved Python language support.
Please take a look at :doc:`using/install/install` for more information about how to obtain them.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cuda-quantum/0.5.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.5.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.5.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.5.0>`__.

**0.4.1**

The 0.4.1 release adds support for ARM processors in the form of multi-platform Docker images and `aarch64` Python wheels. Additionally, all GPU-based backends are now included in the Python wheels as well as in the Docker image.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cuda-quantum/0.4.1>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.4.1>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.4.1/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.4.1>`__.

**0.4.0**

CUDA-Q is now available on PyPI!
The 0.4.0 release adds support for quantum kernel execution on Quantinuum and IonQ backends. For more information, see :doc:`using/backends/hardware`.

The 0.4.0 PyPI release does not yet include all of the GPU-based backends.
The fully featured version is available as a Docker image for `linux/amd64` platforms.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cuda-quantum/0.4.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.4.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/0.4.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/tag/0.4.0>`__.

**0.3.0**

The 0.3.0 release of CUDA-Q is available as a Docker image for `linux/amd64` platforms.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.3.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/0.3.0/docs/sphinx/examples>`__
