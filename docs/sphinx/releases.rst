************************
CUDA-Q Releases
************************

**latest**

The latest version of CUDA-Q is on the main branch of our `GitHub repository <https://github.com/NVIDIA/cuda-quantum>`__ and is also available as a Docker image. More information about installing the nightly builds can be found :doc:`here <using/install/install>`

- `Docker image (nightly builds) <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nightly/containers/cuda-quantum>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/latest>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples>`__

**0.8.0**

The 0.8.0 release adds a range of changes to improve the ease of use and performance with CUDA-Q. 
The changes listed below highlight some of what we think will be the most useful features and changes 
to know about. While the listed changes do not capture all of the great contributions, we would like 
to extend many thanks for every contribution, in particular those from external contributors.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum>`__
- `Python wheel <https://pypi.org/project/cuda-quantum/0.8.0>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.8.0>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.8.0>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.8.0/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.8.0>`__.

**0.7.1**

The 0.7.1 release adds simulator optimizations with significant performance improvements and 
extends their functionalities. The `nvidia-mgpu` backend now supports user customization of the 
gate fusion level as controlled by the `CUDAQ_MGPU_FUSE` environment variable documented 
`here <https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html>`__.
It furthermore adds a range of bug fixes and changes the Python wheel installation instructions.

- `Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags>`__
- `Python wheel <https://pypi.org/project/cuda-quantum/0.7.1>`__
- `C++ installer <https://github.com/NVIDIA/cuda-quantum/releases/0.7.1>`__
- `Documentation <https://nvidia.github.io/cuda-quantum/0.7.1>`__
- `Examples <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.7.1/docs/sphinx/examples>`__

The full change log can be found `here <https://github.com/NVIDIA/cuda-quantum/releases/0.7.1>`__.

**0.7.0**

The 0.7.0 release adds support for using :doc:`NVIDIA Quantum Cloud <using/backends/nvqc>`,
giving you access to our most powerful GPU-accelerated simulators even if you don't have an NVIDIA GPU.
With 0.7.0, we have furthermore greatly increased expressiveness of the Python and C++ language frontends. 
Check out our `documentation <https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html>`__ 
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
