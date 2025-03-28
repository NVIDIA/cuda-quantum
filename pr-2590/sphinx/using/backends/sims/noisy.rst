Noisy Simulators
==================================

Trajectory Noisy Simulation
++++++++++++++++++++++++++++++++++

CUDA-Q GPU simulator backends, :code:`nvidia`, :code:`tensornet`, and :code:`tensornet-mps`,
supports noisy quantum circuit simulations using quantum trajectory method.

When a :code:`noise_model` is provided to CUDA-Q, the backend target 
will incorporate quantum noise into the quantum circuit simulation according 
to the noise model specified, as shown in the below example.

.. tab:: Python

    .. literalinclude:: ../../../snippets/python/using/backends/trajectory.py
        :language: python
        :start-after: [Begin Docs]

    .. code:: bash 
        
        python3 program.py
        { 00:15 01:92 10:81 11:812 }

.. tab:: C++

    .. literalinclude:: ../../../snippets/cpp/using/backends/trajectory.cpp
        :language: cpp
        :start-after: [Begin Documentation]

    .. code:: bash 

        # nvidia target
        nvq++ --target nvidia program.cpp [...] -o program.x
        ./program.x
        { 00:15 01:92 10:81 11:812 }
        # tensornet target
        nvq++ --target tensornet program.cpp [...] -o program.x
        ./program.x
        { 00:10 01:108 10:73 11:809 }
        # tensornet-mps target
        nvq++ --target tensornet-mps program.cpp [...] -o program.x
        ./program.x
        { 00:5 01:86 10:102 11:807 }


In the case of bit-string measurement sampling as in the above example, each measurement 'shot' is executed as a trajectory, 
whereby Kraus operators specified in the noise model are sampled.


Unitary Mixture vs. General Noise Channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantum noise channels can be classified into two categories:

(1) Unitary mixture

The noise channel can be defined by a set of unitary matrices along with list of probabilities associated with those matrices.
The depolarizing channel is an example of unitary mixture, whereby `I` (no noise), `X`, `Y`, or `Z` unitaries may occur to the
quantum state at pre-defined probabilities.

(2) General noise channel

The channel is defined as a set of non-unitary Kraus matrices, satisfying the completely positive and trace preserving (CPTP) condition.
An example of this type of channels is the amplitude damping noise channel.

In trajectory simulation method, simulating unitary mixture noise channels is more efficient than
general noise channels since the trajectory sampling of the latter requires probability calculation based
on the immediate quantum state. 

.. note::
    CUDA-Q noise channel utility automatically detects whether a list of Kraus matrices can be converted to
    the unitary mixture representation for more efficient simulation.

.. list-table:: **Noise Channel Support**
  :widths: 20 30 50

  * - Backend
    - Unitary Mixture
    - General Channel
  * - :code:`nvidia`
    - YES
    - YES
  * - :code:`tensornet`
    - YES
    - NO
  * - :code:`tensornet-mps`
    - YES
    - YES (number of qubits > 1)


Trajectory Expectation Value Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In trajectory simulation method, the statistical error of observable expectation value estimation scales asymptotically 
as :math:`1/\sqrt{N_{trajectories}}`, where :math:`N_{trajectories}` is the number of trajectories.
Hence, depending on the required level of accuracy, the number of trajectories can be specified accordingly.

.. tab:: Python

    .. literalinclude:: ../../../snippets/python/using/backends/trajectory_observe.py
        :language: python
        :start-after: [Begin Docs]

    .. code:: bash 
        
        python3 program.py
        Noisy <Z> with 1024 trajectories = -0.810546875
        Noisy <Z> with 8192 trajectories = -0.800048828125

.. tab:: C++

    .. literalinclude:: ../../../snippets/cpp/using/backends/trajectory_observe.cpp
        :language: cpp
        :start-after: [Begin Documentation]

    .. code:: bash 

        # nvidia target
        nvq++ --target nvidia program.cpp [...] -o program.x
        ./program.x
        Noisy <Z> with 1024 trajectories = -0.810547
        Noisy <Z> with 8192 trajectories = -0.800049

        # tensornet target
        nvq++ --target tensornet program.cpp [...] -o program.x
        ./program.x
        Noisy <Z> with 1024 trajectories = -0.777344
        Noisy <Z> with 8192 trajectories = -0.800537
        
        # tensornet-mps target
        nvq++ --target tensornet-mps program.cpp [...] -o program.x
        ./program.x
        Noisy <Z> with 1024 trajectories = -0.828125
        Noisy <Z> with 8192 trajectories = -0.801758

In the above example, as we increase the number of trajectories, 
the result of CUDA-Q `observe` approaches the true value.

.. note::
    With trajectory noisy simulation, the result of CUDA-Q `observe` is inherently stochastic.  
    For a small number of qubits, the true expectation value can be simulated by the :ref:`density matrix <density-matrix-cpu-backend>` simulator. 

Batched Trajectory Simulation 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the :code:`nvidia` target, when simulating many trajectories with small 
state vectors, the simulation is batched for optimal performance.

.. note::
    
    Batched trajectory simulation is only available on the single-GPU execution mode of the :code:`nvidia` target. 
    
    If batched trajectory simulation is not activated, e.g., due to problem size, number of trajectories, 
    or the nature of the circuit (dynamic circuits with mid-circuit measurements and conditional branching), 
    the required number of trajectories will be executed sequentially.  

The following environment variable options are applicable to the :code:`nvidia` target for batched trajectory noisy simulation. 
Any environment variables must be set prior to setting the target or running "`import cudaq`".

.. list-table:: **Additional environment variable options for trajectory simulation**
  :widths: 20 30 50

  * - Option
    - Value
    - Description
  * - ``CUDAQ_BATCH_SIZE``
    - positive integer or `NONE`
    - The number of state vectors in the batched mode. If `NONE`, the batch size will be calculated based on the available device memory. Default is `NONE`.
  * - ``CUDAQ_BATCHED_SIM_MAX_BRANCHES``
    - positive integer
    - The number of trajectory branches to be tracked simultaneously in the gate fusion. Default is 16. 
  * - ``CUDAQ_BATCHED_SIM_MAX_QUBITS``
    - positive integer
    - The max number of qubits for batching. If the qubit count in the circuit is more than this value, batched trajectory simulation will be disabled. The default value is 20.
  * - ``CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE``
    - positive integer
    - The minimum number of trajectories for batching. If the number of trajectories is less than this value, batched trajectory simulation will be disabled. Default value is 4.

.. note::
    The default batched trajectory simulation parameters have been chosen for optimal performance.

In the below example, we demonstrate the use of these parameters to control trajectory batching.

.. tab:: Python

    .. literalinclude:: ../../../snippets/python/using/backends/trajectory_batching.py
        :language: python
        :start-after: [Begin Docs]

    .. code:: bash 
        
        # Default batching parameter
        python3 program.py
        Simulation elapsed time: 45.75657844543457 ms
        
        # Disable batching by setting batch size to 1
        CUDAQ_BATCH_SIZE=1 python3 program.py
        Simulation elapsed time: 716.090202331543 ms

.. tab:: C++

    .. literalinclude:: ../../../snippets/cpp/using/backends/trajectory_batching.cpp
        :language: cpp
        :start-after: [Begin Documentation]

    .. code:: bash 

        nvq++ --target nvidia program.cpp [...] -o program.x
        # Default batching parameter
        ./program.x
        Simulation elapsed time: 45.47ms
        # Disable batching by setting batch size to 1
        Simulation elapsed time: 558.66ms

.. note::

    The :code:`CUDAQ_LOG_LEVEL` :doc:`environment variable <../../basics/troubleshooting>` can be used to 
    view detailed logs of batched trajectory simulation, e.g., the batch size. 


Density Matrix 
++++++++++++++++

.. _density-matrix-cpu-backend:

Density matrix simulation is helpful for understanding the impact of noise on quantum applications. Unlike state vectors simulation which manipulates the :math:`2^n` state vector, density matrix simulations manipulate the :math:`2^n x 2^n`  density matrix which defines an ensemble of states. To learn how you can leverage the :code:`density-matrix-cpu` backend to study the impact of noise models on your applications, see the  `example here <https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html>`__.

The `Quantum Volume notebook <https://nvidia.github.io/cuda-quantum/latest/applications/python/quantum_volume.html>`__ also demonstrates a full application that leverages the :code:`density-matrix-cpu` backend. 

To execute a program on the :code:`density-matrix-cpu` target, use the following commands:

.. tab:: Python

    .. code:: bash 

        python3 program.py [...] --target density-matrix-cpu

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('density-matrix-cpu')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash 

        nvq++ --target density-matrix-cpu program.cpp [...] -o program.x
        ./program.x


Stim 
++++++

.. _stim-backend:

This backend provides a fast simulator for circuits containing *only* Clifford
gates. Any non-Clifford gates (such as T gates and Toffoli gates) are not
supported. This simulator is based on the `Stim <https://github.com/quantumlib/Stim>`_
library.

To execute a program on the :code:`stim` target, use the following commands:

.. tab:: Python

    .. code:: bash 

        python3 program.py [...] --target stim

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('stim')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash 

        nvq++ --target stim program.cpp [...] -o program.x
        ./program.x

.. note::
    By default CUDA-Q executes kernels using a "shot-by-shot" execution approach.
    This allows for conditional gate execution (i.e. full control flow), but it
    can be slower than executing Stim a single time and generating all the shots
    from that single execution.
    Set the `explicit_measurements` flag with `sample` API for efficient execution.
