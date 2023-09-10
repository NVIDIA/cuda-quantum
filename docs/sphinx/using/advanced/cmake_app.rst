CUDA Quantum and CMake
**********************

CUDA Quantum provides a full compiler toolchain for building 
out quantum-classical applications. Since most C++ projects 
provide some sort of integration with CMake, it becomes imperative 
for our toolchain to integrate well with the CMake build system. 

CUDA Quantum provides CMake configuration files that allow downstream 
application developers to create hybrid quantum-classical source code 
and build it with the typical CMake workflow, which will natively 
leverage :code:`nvq++` for compilation of CUDA Quantum source files. 

To see how this works, let's create a trivial example. Suppose we have 
a folder named :code:`myCudaQApp` and within that folder we have 
a simple CUDA Quantum source file, like `ghz.cpp`. With a local 
CUDA Quantum installation present on the system, we can create the 
following top-level :code:`CMakeLists.txt` file to build this file with 
:code:`nvq++`:

.. code:: bash 

    cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
    project(myCudaQApp LANGUAGES CXX)
    find_package(CUDAQ REQUIRED)
    add_executable(ghz_exe ghz.cpp)

To configure and build this file, all one needs to do is (from the top level of 
the :code:`myCudaQApp` folder)

.. code:: bash 

    mkdir build && cd build 
    cmake .. -G Ninja -DCUDAQ_DIR="$CUDA_QUANTUM_PATH/lib/cmake/cudaq"
    ninja
    ./ghz_exe 

If you run the :code:`ninja` command with :code:`--verbose` you'll see that 
:code:`nvq++` was used to compile and link the executable. 

