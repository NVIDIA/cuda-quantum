
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

  .. literalinclude:: /../snippets/cpp/patterns/compute_action_example.cpp
     :language: cpp
     :start-after: [Begin Compute Action C++ Snippet]
     :end-before: [End Compute Action C++ Snippet]

.. tab:: Python

  .. literalinclude:: /../snippets/python/patterns/compute_action_example.py
     :language: python
     :start-after: [Begin Compute Action Python Snippet]
     :end-before: [End Compute Action Python Snippet]

**[4]** Compiler implementations must add the uncompute segment
and optimize on any controlled versions of this block of code. 