
What is a CUDA Quantum Kernel?
------------------------------
A common pattern in the design and implementation of accelerated-node 
programming models is the separation of accelerator-device code from 
existing CPU host code via function-level boundaries. 

.. code-block:: cpp 

    __this_is_device__ void deviceCode(...) { ... do some work on the accelerator ... }
    void hostCode(...) { 
      ... do host work... 
      deviceCode(args...); 
      ... continue host work ... 
    }

This provides a clear delineation between what must be compiled for, and
executed on, an available computational accelerator. The annotation of device
code is common to aid programming model implementations in device code 
discovery, compilation, and runtime-library configuration and setup. 

CUDA Quantum follows a similar pattern. Specifically, in an effort to better enable 
the development of **generic** libraries of quantum algorithmic primitives
and applications, CUDA Quantum defines quantum device code as stand-alone typed 
callables in C++. A typed callable in C++ is any user-defined :code:`struct`
or :code:`class` that provides an :code:`void operator()(Args...) {}` 
overload (an operator-call overload). Implicitly typed callables - C++ 
lambdas, which compiler implementations enable as automated type definitions
- also provide this typed-callable pattern. CUDA Quantum requires that these callable 
definitions be annotated in some way to indicate that this expression is meant 
for compilation and execution on the quantum device. 

CUDA Quantum distinguishes between two separate kinds of kernel expressions - entry-point 
and pure-device quantum kernels. Entry-point kernels are those that can be 
called from host code, while pure-device kernels are those that can only be
called from other quantum kernel code. See the specification for more detail,
but here we note that the "typed" requirement can be relaxed for pure-device kernels:

.. code-block:: cpp 

    __qpu__ void freeFunctionDeviceKernel(cudaq::qspan<> q) { ... }
    // Entry points are those that can be called from host code
    // i.e., can only take classical input and return classical output
    struct myEntryPointKernel1 {
      int operator()(int i, int j) __qpu__ {
        ...
      }
    };
    struct myEntryPointKernel2 {
      // All classical input must be provided by value
      void operator()(std::vector<double> x) __qpu__ {
        ...
      }
    };

    // CUDA Quantum Kernels can be lambdas too
    auto pureDeviceLambda = [](cudaq::qubit& q) __qpu__ {
      ...
    };
    auto entryPointLambda = [](double theta, double phi) __qpu__ {
      ... allocate quantum memory q ... 
      pureDeviceLambda(q);
      ... 
    };