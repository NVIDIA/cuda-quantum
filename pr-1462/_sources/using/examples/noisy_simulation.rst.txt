Noisy Simulation
-----------------

CUDA-Q makes it simple to model noise within the simulation of your quantum program.
Let's take a look at the various built-in noise models we support, before concluding with a brief example of a custom noise model constructed from user-defined Kraus Operators.

The following code illustrates how to run a simulation with depolarization noise.

.. tab:: Python
   
   .. literalinclude:: ../../examples/python/noise_depolarization.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/basics/noise_depolarization.cpp
      :language: cpp

The following code illustrates how to run a simulation with amplitude damping noise.

.. tab:: Python

   .. literalinclude:: ../../examples/python/noise_amplitude_damping.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/basics/noise_amplitude_damping.cpp
      :language: cpp

The following code illustrates how to run a simulation with bit-flip noise.

.. tab:: Python

   .. literalinclude:: ../../examples/python/noise_bit_flip.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/basics/noise_bit_flip.cpp
      :language: cpp

The following code illustrates how to run a simulation with phase-flip noise.

.. tab:: Python

   .. literalinclude:: ../../examples/python/noise_phase_flip.py
      :language: python

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/basics/noise_phase_flip.cpp
      :language: cpp

The following code illustrates how to run a simulation with a custom noise model.

.. tab:: Python

   .. literalinclude:: ../../examples/python/noise_kraus_operator.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/basics/noise_modeling.cpp
      :language: cpp