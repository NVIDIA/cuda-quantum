
Common Quantum Programming Patterns
***********************************

Compute-Action-Uncompute
------------------------
**[1]** CUDA-Q specifies syntax and semantics to support the compute-action-uncompute
pattern, :math:`W = U V U \dagger` (here :math:`U` is the compute block, and :math:`V` is the action block).

**[2]** Via special syntax for this pattern CUDA-Q enables compiler implementations to 
key on circuit optimizations via programmer intent. Specifically, controlled versions of :math:`W` do not produce controlled
operations of all instructions in :math:`U`, :math:`V`, and :math:`U \dagger`, but instead we are free to only
control :math:`V`. 

**[3]** The CUDA-Q specification requires the following syntax for this pattern:

.. tab:: C++ 

  .. code-block:: cpp

      // Will invoke U V U^dag
      cudaq::compute_action (
          [&](){ 
            /*U_code*/ 
          }, 
          [&]() { 
            /*V_code*/ 
          } 
      ); 

.. tab:: Python 

  .. code-block:: python 

    def computeF():
       ... 
    
    def actionF():
       ...

    # Can take user-defined functions
    cudaq.compute_action(computeF, actionF)

    # Can take Pythonic CUDA-Q lambda kernels
    computeL = lambda : (h(q), x(q), ry(-np.pi, q[0]))
    cudaq.compute_action(computeL, actionF)


**[4]** Compiler implementations must add the uncompute segment
and optimize on any controlled versions of this block of code. 