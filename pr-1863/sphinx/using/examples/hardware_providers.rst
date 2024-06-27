Using Quantum Hardware Providers
-----------------------------------

CUDA-Q contains support for using a set of hardware providers (IonQ, IQM, OQC, and Quantinuum). 
For more information about executing quantum kernels on different hardware backends, please take a look
at :doc:`hardware <../backends/hardware>`.

The following code illustrates how to run kernels on IonQ's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/ionq.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/ionq.cpp
      :language: cpp

The following code illustrates how to run kernels on IQM's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/iqm.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/iqm.cpp
      :language: cpp

The following code illustrates how to run kernels on OQC's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/oqc.py
      :language: python

The following code illustrates how to run kernels on Quantinuum's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/quantinuum.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/quantinuum.cpp
      :language: cpp
