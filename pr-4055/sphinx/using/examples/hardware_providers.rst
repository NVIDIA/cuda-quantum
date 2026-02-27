Using Quantum Hardware Providers
-----------------------------------

CUDA-Q contains support for using a set of hardware providers (Amazon Braket, 
Infleqtion, IonQ, IQM, OQC, ORCA Computing, Quantinuum, and QuEra Computing). 
For more information about executing quantum kernels on different hardware 
backends, please take a look at :doc:`hardware <../backends/hardware>`.

.. _amazon-braket-examples:

Amazon Braket
==================================

The following code illustrates how to run kernels on Amazon Braket's backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/braket.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/braket.cpp
      :language: cpp

.. _anyon-examples:

Anyon Technologies
====================

The following code illustrates how to run kernels on Anyon's backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/anyon.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/anyon.cpp
      :language: cpp


.. _infleqtion-examples:

Infleqtion
==================================

The following code illustrates how to run kernels on Infleqtion's backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/infleqtion.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/infleqtion.cpp
      :language: cpp


.. _ionq-examples:

IonQ
==================================

The following code illustrates how to run kernels on IonQ's backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/ionq.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/ionq.cpp
      :language: cpp


.. _iqm-examples:

IQM
==================================

The following code illustrates how to run kernels on IQM's backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/iqm.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/iqm.cpp
      :language: cpp


.. _oqc-examples:

OQC
==================================

The following code illustrates how to run kernels on OQC's backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/oqc.py
      :language: python

.. tab:: C++  
   
   .. literalinclude:: ../../targets/cpp/oqc.cpp
      :language: cpp


.. _orca-examples:

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

   .. literalinclude:: ../../targets/python/orca.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/orca.cpp
      :language: cpp


.. _pasqal-examples:

Pasqal
==================================

The following code illustrates how to run kernels on Pasqal's backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/pasqal.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/pasqal.cpp
      :language: cpp


.. _quantinuum-examples:

Quantinuum
==================================

The following code illustrates how to run kernels on Quantinuum's backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/quantinuum.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/quantinuum.cpp
      :language: cpp


.. _quantum-circuits-examples:

Quantum Circuits, Inc.
========================

The following code illustrates how to run kernels on Quantum Circuits' backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/qci.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/qci.cpp
      :language: cpp

.. _quantum-machines-examples:

Quantum Machines
==================================

The following code illustrates how to run kernels on Quantum Machines' backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/quantum_machines.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/quantum_machines.cpp
      :language: cpp


.. _quera-examples:

QuEra Computing
==================================

The following code illustrates how to run kernels on QuEra's backends.

.. tab:: Python

   .. literalinclude:: ../../targets/python/quera_basic.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../targets/cpp/quera_basic.cpp
      :language: cpp

