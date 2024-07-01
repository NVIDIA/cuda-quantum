Integrating with Third-Party Libraries
****************************************

CUDA-Q enables easily integrating quantum kernels with existing state-of-the-art tools
for classical computing and high performance computing. While quantum kernels are expressed 
in Python or C++, quantum hardware backends only support a very limited set of classical 
instructions. Correspondingly, only a select set of libraries and functions that are 
directly supported by the CUDA-Q compiler can be called from within quantum kernels.
Code that is not marked as kernel code, on the other hand, can be used to invoke the 
defined quantum kernels as well as any classical tools and libraries. CUDA-Q provides 
a complete toolset to build complex applications running on multi-processor systems involving
CPUs, GPUs and QPUs. 

In Python, integration with any third-party Python package is handled
automatically without any need for additional configuration. For C++, the rest of this
document outlines how to integration with libraries that are compile with a different 
compiler than `nvq++`.


Calling a CUDA-Q library from C++
--------------------------------------

The CUDA-Q compiler is a fully-featured C++ compiler able to process arbitrary C++ code outside
of quantum kernels. In fact, the CUDA-Q installation includes and invokes a custom-built Clang 
compiler under the hood. In general, it hence is not necessary to use a different compiler to 
compile C++ host code that calls into quantum kernels; the host code is compiled along with 
the quantum kernels into a single library or executable by the CUDA-Q toolchain.

If you would still like to create a library that can be included in libraries and executables 
built with a different toolchain, you can configure `nvq++` to include and encapsulate all its
C++ dependencies, such that the CUDA-Q library only depends on C libraries. For example, 
if you have a file `quantum_kernel.cpp` containing 

.. code:: C++

    #include "cudaq.h"
    #include <iostream>
    #include <sstream>

    namespace quantum {

    __qpu__ void bell() {
      cudaq::qvector qs(2);
      h(qs[0]);
      cnot(qs[0], qs[1]);
      mz(qs);
    }

    // Usually, one would probably return something like
    // the most probably value in the distribution rather
    // than the string dump. However, the bell kernel here
    // has a 50/50 chance of returning 00 or 11.
    const char* bell_distribution() {
      auto dist = cudaq::sample(bell);

      std::stringbuf buffer;
      std::ostream os(&buffer);
      dist.dump(os);
      auto sample_results = buffer.str();

      char* ptr = new char[sample_results.size() + 1];
      strcpy(ptr, sample_results.c_str());
      return ptr;
    }

    } // namespace quantum

you can create such a library with the command

.. code:: bash

    nvq++ quantum_kernel.cpp -shared -static-libstdc++ -fPIC -o libquantum_kernel.so

To use this library in a project built with an external toolchain, define a suitable 
header file `quantum_kernel.h` containing

.. code:: C++

    namespace quantum {
      const char* bell_distribution();
    }

and include it in your project. For example, if `external.cpp` contains

.. code:: C++ 

    #include <iostream>
    #include "quantum_kernel.h"

    int main () {
      // some code
      auto bell_dist = quantum::bell_distribution();
      std::cout << bell_dist << std::endl;
      delete[] bell_dist;
      // more code
      return 0;
    }

you can compile it with `g++` and link it with the built `libquantum_kernel.so` library
using the command

.. code:: bash

    g++ external.cpp -lquantum_kernel -L"$(pwd)" -Wl,-rpath="$(pwd)" -I"$(pwd)"

Note that the API defined in `quantum_kernel.h` does not depend on any data types that 
rely on the specific implementation of the C++ standard library. For example, we return a 
`const char*` instead of a `std::string` value. This is important to ensure that the data 
is handled correctly regardless of which C++ standard library is used. If you include
data types that depend on a specific C++ ABI, the linker should detect this incompatibility
and should generate an error. See the section :ref:`clike-shim` below to learn more about 
what to watch out for when defining an API to interface between libraries that are built 
with different compilers. 


Calling an C++ library from CUDA-Q
--------------------------------------

CUDA-Q comes with a complete C++ toolchain, and is configured to use its own C++ standard
library by default. When calling a third-party library from within a CUDA-Q application,
the quantum kernels and the surrounding application code are compiled with the `nvq++` compiler, 
whereas the third-party library may be compiled with a different toolchain. 
For most widely distributed libraries, this will work out of the box. The exception are 
third-party libraries that are pre-compiled (opposed to being header-only libraries) 
and dynamically depend on a different C++ standard library than CUDA-Q uses. 
This is the case for a library `third_party_library.so` if `libstdc++.so` is listed 
in the output of the command

.. code:: bash

    ldd third_party_library.so

To use such a library with CUDA-Q, you have to install the static version of the 
library. The `nvq++` compiler can be configured to link a static library 
`libexternal_library.a` instead of the shared one with the same flags as GCC/Clang:

.. code:: bash

    nvq++ my_code.cpp -Wl,-Bstatic -lexternal_library -Wl,-Bdynamic ...

If the compilation fails due to undefined symbols during linking, then `libexternal_library.a`
does not include the C++ standard library components it depends on. In this case, you can create a 
suitable static library that contains the external library as well as its `libstdc++` dependencies
by combining `libexternal_library.a` and `libstdc++.a` into a single flattened archive.
For example, if the external library is built with GCC and located in `/usr/lib`,
the following commands create a suitable archive using the GNU archiver:

.. code:: bash

    ar cruT libexternal_library_complete.a /usr/lib/libexternal_library.a "$(g++ -print-file-name=libstdc++.a)"
    echo -e 'create libexternal_library_complete.a\naddlib libexternal_library_complete.a\nsave\nend' | ar -M

The above commands create a new static library `libexternal_library_complete.a` in the current directory. 
To use it, adjust the `nvq++` command accordingly:

.. code:: bash

    nvq++ my_code.cpp -L. -Wl,-Bstatic -lexternal_library_complete -Wl,-Bdynamic ...

.. note::

  If the third-party library functions you are calling from your CUDA-Q code 
  consume or produce data types that explicitly depend on the C++ standard library implementation,
  a small shim is needed to properly encapsulate these dependencies, as described in
  :ref:`clike-shim`.


.. _clike-shim:

Interfacing between binaries compiled with a different toolchains
---------------------------------------------------------------------

To avoid incompatibilities between different C++ implementations, 
two binaries built using a different C++ compiler should be careful when passing data types between them
that rely on a specific C++ implementation. While primitive data types can be passed without
issues, attempting to pass a string, for example, will result in a linking failure; a `char*` must 
be passed instead of a string.

Most other data types, including templated types, can be passed as long as their alignment matches.
While different compilers generally align data differently, it is possible to force a particular 
alignment when defining the data type or value. For example, if we define a struct

.. code:: C++

    #pragma pack(push, 4)
    template <typename ScalarType>
    struct Entry {
      bool is_integral;
      ScalarType value;
    };
    #pragma pack(pop)

the `pragma` directives ensure that each item will be 4-byte aligned, regardless of the 
default setting that can differ between compilers.

Even function pointers can usually be passed, as long as their argument and return types can be passed.
When using templated functions and data types, the header file should only include their declaration as well 
as explicit instantiation for all valid template arguments. For example,

.. code:: C++

    template <typename CudaDataType>
    void initializeDeviceStateVector(uint32_t n_blocks, 
                                    int32_t threads_per_block, 
                                    void *deviceStateVector,
                                    std::size_t stateDimension) {
      cudaInitializeDeviceStateVector<<<n_blocks, threads_per_block>>>(
        reinterpret_cast<CudaDataType *>(deviceStateVector), stateDimension);
    }

    template void
    initializeDeviceStateVector<cuFloatComplex>(uint32_t n_blocks, 
                                    int32_t threads_per_block, 
                                    void *deviceStateVector,
                                    std::size_t stateDimension);

    template void
    initializeDeviceStateVector<cuDoubleComplex>(uint32_t n_blocks, 
                                    int32_t threads_per_block, 
                                    void *deviceStateVector,
                                    std::size_t stateDimension);

The most cumbersome aspect when libraries built with different toolchains are combined into a single executable
is exception handling. Since the handling relies on a specific ABI, an thrown exception is not necessarily 
recognized as such in a library that uses a different ABI implementation. While matching the exception type 
may not work as expected, a catch-all will reliably catch any thrown exception. 
