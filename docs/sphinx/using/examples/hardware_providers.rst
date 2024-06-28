Using Quantum Hardware Providers
-----------------------------------

CUDA-Q contains support for using a set of hardware providers (IonQ, IQM, OQC, Quantinuum and ORCA Computing). 
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

The following image shows the schematic of a Time Bin Interferometer (TBI) boson sampling experiment 
that runs on ORCA Computing's backends. A TBI uses optical delay lines with reconfigurable coupling 
parameters. A TBI can be represented by a circuit diagram, like the one below, where this 
illustration example corresponds to 4 photons in 8 modes sent into alternating time-bins in a circuit 
composed of two delay lines in series. 

.. image:: ./images/orca_tbi.png
   :width: 400px
   :align: center

This experiment is performed on ORCA's backends by the code below.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/orca.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/orca.cpp
      :language: cpp