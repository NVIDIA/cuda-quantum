# CUDA-Q NVQLink Library

CUDA-Q NVQLink is a library for describing a tightly integrated Logical QPU. 
Programmers can leverage this library to describe component devices that 
enable real-time coprocessing during qubit coherence times. 

This is very early protoype work. Bugs and issues are likely to be hit. All feedback 
is welcome. This work will continue to rapidly evolve as we learn what works and what doesn't. 

## Getting Started

```bash
# Configure, need cmake 3.28+
cmake -G Ninja .. -DCUDAQ_NVQLINK_INCLUDE_TESTS=ON
# Build
ninja 
# Test
ctest 
```

## Extending the library 

Check out the tests in the `unittests` folder as well as the example codes in `examples`. 

3rd parties can extend this library with new `device` types. The goal is to define 
a subclass of `device_mixin` that allows you specify device traits that your `device` exposes. 
There are a number of traits available, and they are specified in the `device.h` file. There are 
example devices in the `devices/` folder there too. 

3rd parties can also provide custom compiler implementations. Compilers take generic 
code strings and return a `compiled_kernel`. There is one compiler implemented as of 
today, and it is the CUDA-Q compiler. For simplicity, this compiler simply delegates to 
the command line CUDA-Q toolchain. Subclasses should be able to override the `cudaq-opt` 
pass flags. This would allow one to handle CUDA-Q IR operations in a target specific manner 
(e.g., custom lowering of the device_call op).


