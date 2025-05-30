CUDA-Q Dynamics 
===============
.. _dynamics_examples:


This page contains a number of examples that use CUDA-Q dynamics to simulate a range of fundamental physical systems and specific qubit modalities. All example problems simulate systems of very low dimension so that the code can be run quickly on any device. For small problems, the GPU will not provide a significant performance advantage over the CPU. The GPU will start to outperform the CPU for cases where the total dimension of all subsystems is O(1000).


Cavity QED
---------------
.. _cavity_qed:

.. literalinclude:: ../../examples/python/dynamics/cavity_qed.py
   :language: python

Cross Resonance
-----------------
.. _cross_resonance:

.. literalinclude:: ../../examples/python/dynamics/cross_resonance.py
   :language: python

Gate Calibration
------------------
.. _gate_calibration:

.. literalinclude:: ../../examples/python/dynamics/gate_calibration.py
   :language: python

Heisenberg Model
------------------
.. _heisenberg_model:

.. literalinclude:: ../../examples/python/dynamics/heisenberg_model.py
   :language: python

Landau Zener
-------------------
.. _landau_zener:

.. literalinclude:: ../../examples/python/dynamics/landau_zener.py
   :language: python

Pulse
------
.. _pulse:

.. literalinclude:: ../../examples/python/dynamics/pulse.py
   :language: python

Qubit Control
--------------
.. _qubit_control:

.. literalinclude:: ../../examples/python/dynamics/qubit_control.py
   :language: python

Qubit Dynamics
--------------
.. _qubit_dynamics:

.. literalinclude:: ../../examples/python/dynamics/qubit_dynamics.py
   :language: python

Silicon Spin Qubit
-------------------
.. _silicon_spin_qubit:

.. literalinclude:: ../../examples/python/dynamics/silicon_spin_qubit.py
   :language: python

Tensor Callback
------------------
.. _tensor_callback:

.. literalinclude:: ../../examples/python/dynamics/tensor_callback.py
   :language: python

Transmon Resonator
--------------------
.. _transmon_resonator:

.. literalinclude:: ../../examples/python/dynamics/transmon_resonator.py
   :language: python

Initial State (Multi-GPU Multi-Node)
-------------------------------------
.. _initial_state_mgmn:

.. literalinclude:: ../../examples/python/dynamics/mgmn/initial_state.py
   :language: python

Heisenberg Model (Multi-GPU Multi-Node)
-------------------------------------------
.. _heisenberg_model_mgmn:

.. literalinclude:: ../../examples/python/dynamics/mgmn/multi_gpu.py
   :language: python
