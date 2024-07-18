Computing Expectation Values
-----------------------------

CUDA-Q provides generic library functions enabling one to compute expectation values 
of quantum spin operators with respect to a parameterized CUDA-Q kernel. Let's take a look 
at an example of this:

.. tab:: Python

   .. literalinclude:: ../../examples/python/expectation_values.py
      :language: python

   Here we define a parameterized CUDA-Q kernel that takes an angle, `theta`, as a single
   input. This angle becomes the argument of a single `ry` rotation.

   We define a Hamiltonian operator via the CUDA-Q `cudaq.SpinOperator` type.

   CUDA-Q provides a generic function `cudaq.observe`. This function takes as input three
   arguments. The first two argument are a parameterized kernel and the `cudaq.SpinOperator` whose
   expectation value we wish to compute. The final arguments are the runtime parameters at which we
   evaluate the parameterized kernel.


.. tab:: C++

   .. literalinclude:: ../../examples/cpp/basics/expectation_values.cpp
      :language: cpp

   Here we define a parameterized CUDA-Q kernel, a callable type named :code:`ansatz` that takes as 
   input a single angle :code:`theta`. This angle becomes the argument of a single :code:`ry` rotation. 

   In host code, we define a Hamiltonian operator via the CUDA-Q :code:`spin_op` type. 
   CUDA-Q provides a generic function :code:`cudaq::observe`. This function takes as input three terms. 
   The first two terms are a parameterized kernel and the :code:`spin_op` whose expectation value we wish to compute.
   The last term contains the runtime parameters at which we evaluate the parameterized kernel. 

   The return type of this function is an :code:`cudaq::observe_result` which contains all the data 
   from the execution, but is trivially convertible to a double, resulting in the expectation value we are interested in. 

   To compile and execute this code, we run the following:

   .. code:: bash 

      nvq++ expectation_values.cpp -o exp_vals.x 
      ./exp_vals.x 


Parallelizing across Multiple Processors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`multi-processor platforms <../backends/platform>` page.


One typical use case of :doc:`multi-processor platforms <../backends/platform>` is to distribute the
expectation value computations of a multi-term Hamiltonian across multiple virtual QPUs.

The following shows an example using the :code:`nvidia-mqpu` platform:

.. tab:: Python

    .. literalinclude:: ../../snippets/python/using/cudaq/platform/observe_mqpu.py
        :language: python
        :start-after: [Begin Documentation]

.. tab:: C++

    .. literalinclude:: ../../snippets/cpp/using/cudaq/platform/observe_mqpu.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]


    One can then target the :code:`nvidia-mqpu` platform by executing the following commands:

    .. code-block:: console

        nvq++ observe_mqpu.cpp -target nvidia-mqpu
        ./a.out

In the above code snippets, since the Hamiltonian contains four non-identity terms, there are four quantum circuits that need to be executed
in order to compute the expectation value of that Hamiltonian and given the quantum state prepared by the ansatz kernel. When the :code:`nvidia-mqpu` platform
is selected, these circuits will be distributed across all available QPUs. The final expectation value result is computed from all QPU execution results.
