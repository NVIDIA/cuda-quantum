Using Quantum Hardware Providers
-----------------------------------

CUDA-Q contains support for using a set of hardware providers (IonQ, IQM, OQC, ORCA Computing and Quantinuum). 
For more information about executing quantum kernels on different hardware backends, please take a look
at :doc:`hardware <../backends/hardware>`.

IonQ
==================================

The following code illustrates how to run kernels on IonQ's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/ionq.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/ionq.cpp
      :language: cpp

IQM
==================================

The following code illustrates how to run kernels on IQM's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/iqm.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/iqm.cpp
      :language: cpp

OQC
==================================

The following code illustrates how to run kernels on OQC's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/oqc.py
      :language: python

ORCA Computing
==================================

The following code illustrates how to run kernels on ORCA Computing's backends.

ORCA Computing's PT Series implement the boson sampling model of quantum computation, in which 
multiple photons are interfered with each other within a network of beam splitters, and photon 
detectors measure where the photons leave this network.

The following image shows the schematic of a Time Bin Interferometer (TBI) boson sampling experiment 
that runs on ORCA Computing's backends. A TBI uses optical delay lines with reconfigurable coupling 
parameters. A TBI can be represented by a circuit diagram, like the one below, where this 
illustration example corresponds to 4 photons in 8 modes sent into alternating time-bins in a circuit 
composed of two delay lines in series. 

.. image:: ./images/orca_tbi.png
   :width: 400px
   :align: center

The parameters needed to define the time bin interferometer are the the input state, the loop 
lengths, beam splitter angles, and optionally the phase shifter angles, and the number of samples.
The *input state* is the initial state of the photons in the time bin interferometer, 
the left-most entry corresponds to the first mode entering the loop.
The *loop lengths* are the lengths of the different loops in the time bin interferometer.
The *beam splitter angles* and the phase shifter angles are controllable
parameters of the time bin interferometer.

This experiment is performed on ORCA's backends by the code below.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/orca.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/orca.cpp
      :language: cpp
      
Quantinuum
==================================

The following code illustrates how to run kernels on Quantinuum's backends.

.. tab:: Python

   .. literalinclude:: ../../examples/python/providers/quantinuum.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/providers/quantinuum.cpp
      :language: cpp

