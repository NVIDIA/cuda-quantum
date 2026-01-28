Scaleway Quantum as a Service
+++++++++++++++++++++++++++++

.. _scaleway-backend:

`Scaleway Quantum as a Service <https://www.scaleway.com/en/quantum-as-a-service/>`__
is a managed cloud service providing on-demand access to quantum computing
resources, including quantum emulators and quantum processing units (QPUs),
through a unified API.

Scaleway QaaS allows users to submit quantum workloads programmatically and
integrate them into hybrid classical–quantum workflows. The service is designed
to be used either directly through the Scaleway APIs or via higher-level SDKs
and frameworks such as CUDA-Q, Qiskit and Cirq.

To get started, users must have an active Scaleway account and a project with
Quantum Computing enabled. See the
`Scaleway Quantum Computing Quickstart <https://www.scaleway.com/en/docs/quantum-computing/quickstart/>`__
for step-by-step instructions.

Additional information can be found in the
`Scaleway Quantum Computing documentation <https://www.scaleway.com/en/docs/quantum-computing/>`__.

Available emulators and QPUs are listed in the Scaleway Quantum Computing documentation and may evolve over time.

Setting Credentials
```````````````````

.. code:: bash

  export SCW_SECRET_KEY="<secret_key>"
  export SCW_PROJECT_ID="<project_id>"

Submission from C++
```````````````````

To target quantum kernel code for execution on Scaleway QaaS, pass the
--target scaleway flag to the nvq++ compiler.

By default, jobs are submitted to a Scaleway-managed quantum simulator.

.. code:: bash

    nvq++ --target scaleway src.cpp

To execute kernels on a specific Scaleway quantum device, pass the
--scaleway-machine flag to nvq++ and specify the device identifier.

.. code:: bash

    nvq++ --target scaleway --scaleway-machine "<offer_name>" src.cpp

where <offer_name> refers to a Scaleway simulator or QPU available in your
project.

To emulate the target locally without submitting a job to the Scaleway cloud,
use the --emulate flag:

.. code:: bash

    nvq++ --emulate --target scaleway src.cpp

Submission from Python
``````````````````````

The target backend for quantum kernel execution can be selected using the
``cudaq.set_target()`` function.

.. code:: python

   import cudaq
   cudaq.set_target("scaleway")

By default, kernels are executed on a Scaleway quantum simulator.

To select a specific Scaleway device, set the ``machine`` parameter:

.. code:: python

   cudaq.set_target("scaleway", machine="EMU-CUDAQ-H100")

where ``EMU-CUDAQ-H100`` identifies an emulator or QPU offer available through Scaleway
Quantum as a Service.

The number of shots for a kernel execution can be specified via the
``shots_count`` argument to ``cudaq.sample``. The default value is 1000.

.. code:: python

   result = cudaq.sample(kernel, shots_count=100)

Refer to the Scaleway Quantum Computing documentation for details on device
capabilities, execution limits, and pricing.