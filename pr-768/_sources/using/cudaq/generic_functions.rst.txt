Generic Library Functions
-------------------------
One of the primary goals of the CUDA Quantum platform is to build up a robust, 
widely-applicable, :code:`cudaq::` namespace of generic, algorithmic primitive
functions. By generic, we mean that these functions are ultimately templated 
on the input CUDA Quantum kernel expression, implying that algorithmic function 
definitions are applicable to a wide-range of input quantum code. This
characteristic of quantum algorithm development is ubiquitous. One often
designs algorithms that are general with regards to a quantum oracle, or a 
state-preparation step, just to name a few examples. CUDA Quantum enables this via 
generic :code:`cudaq::` functions that take any CUDA Quantum kernel expression as input.

Let's take a look at the first couple examples of this already implemented in CUDA Quantum. 
The first function we provide is :code:`cudaq::sample(...)`, which takes any 
kernel as input (with certain characteristics, see the Specification) and
samples the kernel's resultant state over a number of shots, returning a map
of observed measurement bit strings to the corresponding number of times that
configuration was observed. Using this function is straightforward:

.. code-block:: cpp 

    auto myFirstKernel_Toffoli_111_input = [](cudaq::qspan<> threeQubits) __qpu__ {
      // Alias the 3 qubits
      auto& q = threeQubits[0];
      auto& r = threeQubits[1];
      auto& s = threeQubits[2];
      // Create 101
      x (q, s);
      // Manual decomposition of x(q, r, s);
      // i.e., could have also written 
      // x<cudaq::ctrl>(q, r, s); 
      h(s);
      x<cudaq::ctrl>(r, s);
      t<cudaq::adj>(s);
      x<cudaq::ctrl>(q, s);
      t(s);
      x<cudaq::ctrl>(r,s);
      t<cudaq::adj>(s);
      x<cudaq::ctrl>(q,s);
      t(r); t(s);
      x<cudaq::ctrl>(q, r);
      t(q); t<cudaq::adj>(r);
      x<cudaq::ctrl>(q, r);
    };

    // cudaq::sample takes entry point kernels as input
    auto entryPointKernel = [&]() __qpu__ { 
      cudaq::qreg<3> q; 
      myFirstKernel_Toffoli_111_input(q);
      mz(q);
    };

    // Sample the state produced by this kernel
    // dump the counts to stdout
    auto counts = cudaq::sample(entryPointKernel);
    counts.dump();
    // prints { 011:1000 }

If your CUDA Quantum kernel takes classical data as input, then those runtime 
values must be provided to the :code:`cudaq::sample` function as trailing
arguments 

.. code-block:: cpp 

    auto ghz = [](int N) __qpu__ {
      cudaq::qreg q(N);
      h(q[0]);
      for (int i = 0; i < N - 1; i++) {
        x<cudaq::ctrl>(q[i], q[i + 1]);
      }
      mz(q);
    };

    auto counts = cudaq::sample(ghz, 5); // note runtime arguments 
    for (auto& [bits, count] : counts) {
      std::cout << "Observed " << bits << ":" << count "\n";
    }
    // prints 
    // Observed 11111:505
    // Observed 00000:495

Another useful CUDA Quantum function in the variational context is 
:code:`cudaq::observe(...)`. This function takes any kernel expression
(with suitable characteristics noted in the Specification), a user-provided 
:code:`cudaq::spin_op` defining a general quantum mechanical spin operator, and
any kernel runtime arguments to return the expected value of the spin operator 
with respect to the kernel ansatz at the provided runtime parameters. It can be 
used in the following manner:

.. code-block:: cpp 

    auto ansatz = [](double theta) __qpu__ {
      ... Define your parameterized kernel ... 
      ... No measures, as they are dictated by the spin_op ...
    };

    using namespace cudaq::spin;
    cudaq::spin_op H = ...;
    auto exp_val = cudaq::observe(ansatz, H, /* theta */ M_PI / 2.0);
