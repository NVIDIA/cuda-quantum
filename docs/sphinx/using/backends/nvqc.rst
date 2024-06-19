NVIDIA Quantum Cloud
---------------------
NVIDIA Quantum Cloud (NVQC) offers universal access to the world’s most powerful computing platform, 
for every quantum researcher to do their life’s work.
To learn more about NVQC visit this `link <https://www.nvidia.com/en-us/solutions/quantum-computing/cloud>`__. 

Apply for early access `here <https://developer.nvidia.com/quantum-cloud-early-access-join>`__. 
Access to the Quantum Cloud early access program requires an NVIDIA Developer account.

Quick Start
+++++++++++
Once you have been approved for an early access to NVQC, you will be able to follow these instructions to use it.

1. Follow the instructions in your NVQC Early Access welcome email to obtain an API Key for NVQC. 
You can also find the instructions `here <https://developer.nvidia.com/quantum-cloud-early-access-members>`__ (link available only for approved users)

2. Set the environment variable `NVQC_API_KEY` to the API Key obtained above.

 .. code-block:: console

    export NVQC_API_KEY=<your NVQC API key>

You may wish to persist that environment variable between bash sessions, e.g., by adding it to your `$HOME/.bashrc` file.

3. Run your first NVQC example

The following is a typical CUDA-Q kernel example. 
By selecting the `nvqc` target, the quantum circuit simulation will run on NVQC in the cloud, rather than running locally.


.. tab:: Python
    
    .. literalinclude:: ../../snippets/python/using/cudaq/nvqc/nvqc_intro.py
        :language: python
        :start-after: [Begin Documentation]

    .. code-block:: console
        
        [2024-03-14 19:26:31.438] Submitting jobs to NVQC service with 1 GPU(s). Max execution time: 3600 seconds (excluding queue wait time).

        ================ NVQC Device Info ================
        GPU Device Name: "NVIDIA H100 80GB HBM3"
        CUDA Driver Version / Runtime Version: 12.2 / 11.8
        Total global memory (GB): 79.1
        Memory Clock Rate (MHz): 2619.000
        GPU Clock Rate (MHz): 1980.000
        ==================================================
        { 1111111111111111111111111:486 0000000000000000000000000:514 }

.. tab:: C++

    .. literalinclude:: ../../snippets/cpp/using/cudaq/nvqc/nvqc_intro.cpp
        :language: cpp
        :start-after: [Begin Documentation]

    The code above is saved in `nvqc_intro.cpp` and compiled with the following command, targeting the :code:`nvqc` platform

    .. code-block:: console

        nvq++ nvqc_intro.cpp -o nvqc_intro.x --target nvqc 
        ./nvqc_intro.x

        [2024-03-14 19:25:05.545] Submitting jobs to NVQC service with 1 GPU(s). Max execution time: 3600 seconds (excluding queue wait time).

        ================ NVQC Device Info ================
        GPU Device Name: "NVIDIA H100 80GB HBM3"
        CUDA Driver Version / Runtime Version: 12.2 / 11.8
        Total global memory (GB): 79.1
        Memory Clock Rate (MHz): 2619.000
        GPU Clock Rate (MHz): 1980.000
        ==================================================
        { 
        __global__ : { 1111111111111111111111111:487 0000000000000000000000000:513 }
        result : { 1111111111111111111111111:487 0000000000000000000000000:513 }
        }


Simulator Backend Selection
++++++++++++++++++++++++++++

NVQC hosts all CUDA-Q simulator backends (see :doc:`simulators`). 
You may use the NVQC `backend` (Python) or `--nvqc-backend` (C++) option to select the simulator to be used by the service.

For example, to request the `tensornet` simulator backend, the user can do the following for C++ or Python.

.. tab:: Python

    .. code-block:: python

        cudaq.set_target("nvqc", backend="tensornet")

.. tab:: C++
    
    .. code-block:: console

        nvq++ nvqc_sample.cpp -o nvqc_sample.x --target nvqc --nvqc-backend tensornet


.. note::

  By default, the single-GPU single-precision `custatevec-fp32` simulator backend will be selected if backend information is not specified.

Multiple GPUs
+++++++++++++

Some CUDA-Q simulator backends are capable of multi-GPU distribution as detailed in :doc:`simulators`.
For example, the `nvidia-mgpu` backend can partition and distribute state vector simulation to multiple GPUs to simulate 
a larger number of qubits, whose state vector size grows beyond the memory size of a single GPU.

To select a specific number of GPUs on the NVQC managed service, the following `ngpus` (Python) or `--nvqc-ngpus` (C++) option can be used.


.. tab:: Python

    .. code-block:: python

        cudaq.set_target("nvqc", backend="nvidia-mgpu", ngpus=4)

.. tab:: C++

    .. code-block:: console

        nvq++ nvqc_sample.cpp -o nvqc_sample.x --target nvqc --nvqc-backend nvidia-mgpu --nvqc-ngpus 4


.. note::

    If your NVQC subscription does not contain service instances that have the specified number of GPUs, 
    you may encounter the following error.

    .. code-block:: console
        
        Unable to find NVQC deployment with 16 GPUs.
        Available deployments have {1, 2, 4, 8} GPUs.
        Please check your `ngpus` value (Python) or `--nvqc-ngpus` value (C++).

.. note::

    Not all simulator backends are capable of utilizing multiple GPUs. 
    When requesting a multiple-GPU service with a single-GPU simulator backend, 
    you might encounter the following log message:

    .. code-block:: console
        
        The requested backend simulator (custatevec-fp32) is not capable of using all 4 GPUs requested.
        Only one GPU will be used for simulation.
        Please refer to CUDA-Q documentation for a list of multi-GPU capable simulator backends.

    Consider removing the `ngpus` value (Python) or `--nvqc-ngpus` value (C++) to use the default of 1 GPU 
    if you don't need to use a multi-GPU backend to better utilize NVQC resources.

    Please refer to the table below for a list of backend simulator names along with its multi-GPU capability.

    .. list-table:: Simulator Backends
        :widths: 20 50 10 10
        :header-rows: 1

        *   - Name
            - Description
            - GPU Accelerated 
            - Multi-GPU 
        *   - `qpp`
            - CPU-only state vector simulator
            - no
            - no
        *   - `dm`
            - CPU-only density matrix simulator
            - no
            - no
        *   - `custatevec-fp32`
            - Single-precision `cuStateVec` simulator
            - yes
            - no
        *   - `custatevec-fp64`
            - Double-precision `cuStateVec` simulator
            - yes
            - no
        *   - `tensornet`
            - Double-precision `cuTensorNet` full tensor network contraction simulator
            - yes
            - yes
        *   - `tensornet-mps`
            - Double-precision `cuTensorNet` matrix-product state simulator
            - yes
            - no
        *   - `nvidia-mgpu`
            - Double-precision `cuStateVec` multi-GPU simulator
            - yes
            - yes
    

Multiple QPUs Asynchronous Execution
+++++++++++++++++++++++++++++++++++++

NVQC provides scalable QPU virtualization services, whereby clients
can submit asynchronous jobs simultaneously to NVQC. These jobs are
handled by a pool of service worker instances.

For example, in the following code snippet, using the `nqpus` (Python) or `--nvqc-nqpus` (C++) configuration option,
the user instantiates 3 virtual QPU instances to submit simulation jobs to NVQC
calculating the expectation value along with parameter-shift gradients simultaneously.

.. tab:: Python

    .. literalinclude:: ../../snippets/python/using/cudaq/nvqc/nvqc_mqpu.py
        :language: python
        :start-after: [Begin Documentation]

.. tab:: C++

    .. literalinclude:: ../../snippets/cpp/using/cudaq/nvqc/nvqc_mqpu.cpp
        :language: cpp
        :start-after: [Begin Documentation]

    The code above is saved in `nvqc_vqe.cpp` and compiled with the following command, targeting the :code:`nvqc` platform with 3 virtual QPUs.

    .. code-block:: console

        nvq++ nvqc_vqe.cpp -o nvqc_vqe.x --target nvqc --nvqc-nqpus 3 
        ./nvqc_vqe.x


.. note::

    The NVQC managed-service has a pool of worker instances processing incoming requests on a 
    first-come-first-serve basis. Thus, the attainable speedup using multiple virtual QPUs vs. 
    sequential execution on a single QPU depends on the NVQC service load. For example, 
    if the number of free workers is greater than the number of requested virtual QPUs, a linear
    (ideal) speedup could be achieved. On the other hand, if all the service workers are busy, 
    multi-QPU distribution may not deliver any substantial speedup.  

FAQ
++++

1. How do I get more information about my NVQC API submission?

The environment variable `NVQC_LOG_LEVEL` can be used to turn on and off
certain logs. There are three levels:

- Info (`info`): basic information about NVQC is logged to the console. This is the default.

- Off (`off` or `0`): disable all NVQC logging.

- Trace: (`trace`): log additional information for each NVQC job execution (including timing)

2. I want to persist my API key to a configuration file.

You may persist your NVQC API Key to a credential configuration file in lieu of 
using the `NVQC_API_KEY` environment variable. 
The configuration file can be generated as follows, replacing
the `api_key` value with your NVQC API Key.

.. code:: bash

    echo "key: <api_key>" >> $HOME/.nvqc_config

