
Quantum Programming Patterns
****************************
Within quantum kernels, the CUDA Quantum programming model provides simplified
syntax for common quantum programming patterns. The collection of patterns
in this section will grow over time, but we start with two basic features
providing simplified syntax and semantics for common quantum programming
tasks. These tasks may also enable the expressivity of programmer intent
that directly influences compile-time circuit optimizations. 

Compute-Action-Uncompute
------------------------
We require syntax and semantics to support the compute-action-uncompute
pattern, :math:`W = U V U \dagger` (here :math:`U` is the compute block, and :math:`V` is the action block).
This is a very common pattern, and by expressing this pattern with specific
programmer intent, the compiler is able to apply critical resource
optimizations. Specifically, controlled versions of W do not produce controlled
operations of all instructions in :math:`U`, :math:`V`, and :math:`U \dagger`, but instead we are free to only
control :math:`V`. The CUDA Quantum specification requires the following syntax for this pattern:

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

Compiler implementations must add the uncompute segment
and optimize on any controlled versions of this block of code. 