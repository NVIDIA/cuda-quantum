CUDA-Q Dynamics 
===============
.. _dynamics_examples:

Below are a number of examples that use CUDA-Q dynamics to simulate a range of fundamental physical systems and specific qubit modalities. All example problems simulate systems of very low dimension so that the code can be run quickly on any device. For small problems, the GPU will not provide a significant performance advantage over the CPU.  The GPU will start to outperform the CPU for cases where the total dimension of all subsystems is O(1000)

Download the notebooks below here: https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/python/dynamics


.. toctree::
   :maxdepth: 2

      Introduction to CUDA-Q Dynamics (Jaynes-Cummings Model) <../../examples/python/dynamics/dynamics_intro_1.ipynb>
      Introduction to CUDA-Q Dynamics (Time Dependent Hamiltonians) <../../examples/python/dynamics/dynamics_intro_2.ipynb>
      Superconducting Qubits <../../examples/python/dynamics/superconducting.ipynb>
      Spin Qubits <../../examples/python/dynamics/spinqubits.ipynb>
      Trapped Ion Qubits <../../examples/python/dynamics/iontrap.ipynb>
      Control <../../examples/python/dynamics/control.ipynb>
      Multi-GPU Multi-Node <../../examples/python/dynamics/mgmn.ipynb>

