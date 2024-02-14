Using CUDA and CUDA Quantum in a Project
****************************************

It may be the case that a project that uses CUDA Quantum kernels may also
want to use CUDA code to do computation on a GPU.  This is possible by using
both the CUDA Toolkit and CUDA Quantum tools. More about programming GPUs
in CUDA can be found in the `Quick Start Guide
<https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html>`__.

If you are using a CUDA Quantum container, the CUDA Toolkit can be installed
using a command such as

.. code:: bash

    apt-get install cuda-toolkit-11-8

Once the :code:`nvcc` compiler is installed, it is possible to write
CUDA kernels and have them execute on the system GPU. See NVIDIA's `An
Easy Introduction to CUDA C and C++
<https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/>`__
for more information on getting started with CUDA.

CUDA code uses a unique syntax and is, typically, saved in a file with
the extension :code:`.cu`. For our example, assume we have written our
CUDA code in the file :code:`my_proj.cu`.

CUDA Quantum code is a library-based extension of C++ and uses
standard conforming C++ syntax. Typically, a quantum kernel would be
saved in a file with the :code:`.cpp` extension. Again for our
example, let's assume that we've written quantum kernels and saved
them in the file :code:`my_proj_quantum.cpp`.

There is a bit of a wrinkle to be aware of before we compile these two
compilation units. Version 11 (and earlier) of CUDA :code:`nvcc`
supports the C++ 11, 14, and 17 standards and the default standard is
determined by the host C++ compiler. The CUDA Quantum compiler,
:code:`nvq++`, defaults to the C++ 20 standard. To get around this
limitation, the project makefiles should select a common C++ standard
version. Fortunately, :code:`nvq++` does allow the use of C++ 17.

Note that starting with version 12 of the CUDA toolkit, the C++ 20
standard is supported.

Our project can then be built with commands such as

.. code:: bash

    nvcc -c -std=c++17 <options> my_proj.cu -o my_proj.o
    nvq++ -std=c++17 <options> my_project_quantum.cpp my_proj.o -L ${CUDA_INSTALL}/lib64 -lcudart -o my_executable

Above, :code:`nvq++` is used for the link step and will make sure the CUDA
Quantum runtime libraries are linked correctly to the executable program.
The CUDA runtime is explicitly added to this command.
