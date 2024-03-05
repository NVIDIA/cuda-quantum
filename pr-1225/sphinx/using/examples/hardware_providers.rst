.. _python-examples-for-hardware-providers:
.. _cpp-examples-for-hardware-providers:

Using Quantum Hardware Providers
-----------------------------------

CUDA Quantum contains support for using a set of hardware providers (Quantinuum, IonQ, and IQM). 
For more information about executing quantum kernels on different hardware backends, please take a look at :ref:`hardware <hardware-landing-page>`.

The following code illustrates how to run kernels on Quantinuum's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/quantinuum.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/quantinuum.cpp
      :language: cpp

The following code illustrates how to run kernels on IonQ's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/ionq.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/ionq.cpp
      :language: cpp

The following code illustrates how to run kernels on IQM's backends.

.. literalinclude:: ../../examples/python/providers/iqm.py
   :language: python