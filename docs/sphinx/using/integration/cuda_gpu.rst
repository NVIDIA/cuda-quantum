Using CUDA and CUDA-Q in a Project
****************************************

It may be the case that a project that uses CUDA-Q kernels may also
want to use CUDA code to do computation on a GPU.  This is possible by using
both the CUDA Toolkit and CUDA-Q tools. More about programming GPUs
in CUDA can be found in the `Quick Start Guide
<https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html>`__.

Once the `nvcc` compiler is installed, it is possible to write
CUDA kernels and have them execute on the system GPU. See NVIDIA's `An
Easy Introduction to CUDA C and C++
<https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/>`__
for more information on getting started with CUDA.

CUDA code uses a unique syntax and is, typically, saved in a file with
the extension `.cu`. For our example, assume we have written our
CUDA code in the file `my_proj.cu`.

CUDA-Q code is a library-based extension of C++ and uses
standard conforming C++ syntax. Typically, a quantum kernel would be
saved in a file with the `.cpp` extension. Again for our
example, let's assume that we've written quantum kernels and saved
them in the file `my_proj_quantum.cpp`.

By default, CUDA-Q uses C++ 20 and builds source code against the 
LLVM C++ standard library (`libc++`). To create a CUDA library that 
can link against CUDA-Q code, make sure to define an API that does 
not rely on C++ data structures that rely on a specific C++ toolchain
for all functions intended to be called from CUDA-Q (see also 
:ref:`clike-shim`). For example, if you define a CUDA kernel in `my_proj.cu`

.. code-block:: cpp

    template <typename CudaDataType>
    __global__ void cudaSetFirstNElements(CudaDataType *sv, const CudaDataType *__restrict__ sv2, int64_t N) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < N) {
            sv[i].x = sv2[i].x;
            sv[i].y = sv2[i].y;
        } else {
            sv[i].x = 0.0;
            sv[i].y = 0.0;
        }
    }

define the following template and all desired template specializations

.. code-block:: cpp

    template <typename CudaDataType>
    void setFirstNElements(uint32_t n_blocks, 
                           int32_t threads_per_block, 
                           void *newDeviceStateVector, 
                           void *deviceStateVector,
                           std::size_t previousStateDimension) {
        cudaSetFirstNElements<<<n_blocks, threads_per_block>>>(
            reinterpret_cast<CudaDataType *>(newDeviceStateVector),
            reinterpret_cast<CudaDataType *>(deviceStateVector),
            previousStateDimension);
    }

    template void
    setFirstNElements<cuFloatComplex>(uint32_t n_blocks, 
                                      int32_t threads_per_block, 
                                      void *newDeviceStateVector, 
                                      void *deviceStateVector,
                                      std::size_t previousStateDimension);

    template void
    setFirstNElements<cuDoubleComplex>(uint32_t n_blocks, 
                                       int32_t threads_per_block, 
                                       void *newDeviceStateVector, 
                                       void *deviceStateVector,
                                       std::size_t previousStateDimension);

Create the corresponding header file `my_proj.h` declaring the `setFirstNElements`
template. You can then call `setFirstNElements` from within `my_proj_quantum.cpp`
by including `my_proj.h`, for example

.. code-block:: cpp

    #include "cuComplex.h"
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #include "my_proj.h"

    using namespace my_kernels; // the namespace where you defined setFirstNElements

    int main () {
        const uint32_t n_blocks = 10;
        const uint32_t threads_per_block = 5;

        void *deviceStateVector;
        cudaMalloc((void **)&deviceStateVector, 2 * sizeof(cuDoubleComplex));
        // ...
        void *newDeviceStateVector;
        cudaMalloc((void **)&newDeviceStateVector, 2 * sizeof(cuDoubleComplex));
        setFirstNElements<cuDoubleComplex>(n_blocks, threads_per_block,
                                           newDeviceStateVector, deviceStateVector,
                                           2);
        return 0;
    }

To get an executable, compile the code with

.. code-block:: bash

    nvcc -c -std=c++17 -Xcompiler -fPIC my_proj.cu -o my_proj.o
    nvq++ my_proj_quantum.cpp my_proj.o -I"${CUDA_HOME}/include/" -L"${CUDA_HOME}/lib64" -lcudart -o my_executable

Above, :code:`nvq++` is used for the link step and will make sure the CUDA-Q
runtime libraries are linked correctly to the executable program.
The CUDA runtime is explicitly added to this command.
