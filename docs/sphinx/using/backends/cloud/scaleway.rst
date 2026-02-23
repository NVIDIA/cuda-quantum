Scaleway Quantum as a Service
+++++++++++++++++++++++++++++

.. _scaleway-backend:

`Scaleway Quantum as a Service <https://www.scaleway.com/en/quantum-as-a-service/>`__
is a managed cloud service providing on-demand access to quantum computing
resources, including quantum processing units (QPUs) and quantum emulators,
through a unified API.

Scaleway QaaS allows users to submit quantum workloads programmatically and
integrate them into hybrid classical–quantum workflows. The service is designed
to be used either directly through the Scaleway APIs or via higher-level SDK
and frameworks such as CUDA-Q.

To get started, users must have an active Scaleway account and a project with
Quantum Computing enabled. See the
`Quickstart <https://www.scaleway.com/en/docs/quantum-computing/quickstart/>`__
for step-by-step instructions.

Additional information can be found in the
`Scaleway Quantum Computing documentation <https://www.scaleway.com/en/docs/quantum-computing/>`__.

Setting Credentials
```````````````````

.. code:: bash

  export SCW_SECRET_KEY="<secret_key>"
  export SCW_PROJECT_ID="<project_id>"

Submitting
``````````

.. tab:: Python

   The target backend for quantum kernel execution can be selected using the
   ``cudaq.set_target()`` function.

   .. code:: python

      import cudaq
      # Use credentials from environment variables
      cudaq.set_target("scaleway")

      # You can specify manually your credentials
      cudaq.set_target("scaleway", project_id="<project_id>", secret_key="<secret_key>")


   By default, kernels are executed on a Scaleway quantum emulator ``EMU-CUDAQ-H100``.

   To select a specific Scaleway device, set the ``machine`` argument:

   .. code:: python

      machine = "EMU-CUDAQ-H100"
      # machine = EMU-AER-H100 # Access to Aer emulator
      # machine = QPU-EMERALD-54PQ # Access to IQM QPUs (Garnet, Sirius, Emerald)
      # machine = QPU-IBEX-12PQ # Access to AQT IBEX-Q1 QPU
      # machine = EMU-IBEX-12PQ-L4 # Access to AQT IBEX-Q1 emulator
      cudaq.set_target("scaleway", machine=machine)

   where ``EMU-CUDAQ-H100`` identifies an emulator or QPU offer available through Scaleway
   Quantum as a Service. Available emulators and QPUs are listed on the
   `Scaleway Quantum-as-a-Service webpage <https://www.scaleway.com/en/quantum-as-a-service/>`__

   The service will dynamically allocate a dedicated GPU server for your need.

   This allocation can take up to few minutes. To use the same session between different script execution
   or users, you can specify a ``deduplication_id``.

   The session will be created if doesn't exist, else it will retrieve and use the matching one.

   .. code:: python

      machine = "EMU-CUDAQ-H100"
      # The deduplication identifier is a convenient way to keep using the same resource
      # between script calls or user
      # Notes: The target ``machine`` must be the same as well as Scaleway project id
      cudaq.set_target("scaleway", machine=machine, deduplication_id="my-workshop")

   You can specify the maximal duration or the maximal idle duration to limit the billing.

   .. code:: python

      machine = "EMU-CUDAQ-H100"
      # The underlying QPU session will be killed after 30 minutes
      # or after 5 idle minutes without new jobs
      cudaq.set_target("scaleway", machine=machine, max_duration="30m", max_idle_duration="5m")

   The number of shots for a kernel execution can be specified via the
   ``shots_count`` argument to ``cudaq.sample``. The default value is 1000.

   .. code:: python

      result = cudaq.sample(kernel, shots_count=100)

   Refer to the Scaleway Quantum Computing documentation for details on device
   capabilities, execution limits, and pricing.

.. tab:: C++

   To target quantum kernel code for execution on Scaleway QaaS, pass the
   ``--target scaleway`` flag to the ``nvq++`` compiler.

   By default, jobs are submitted to a Scaleway-managed quantum simulator.

   .. code:: bash

      nvq++ --target scaleway src.cpp

   To execute kernels on a specific Scaleway quantum device, pass the
   ``--machine`` flag to ``nvq++`` and specify the device identifier.

   .. code:: bash

      nvq++ --target scaleway --machine "<offer_name>" src.cpp

   where <offer_name> refers to a Scaleway simulator or QPU available in your
   project. Available emulators and QPUs are listed on the
   `Scaleway Quantum-as-a-Service webpage <https://www.scaleway.com/en/quantum-as-a-service/>`__

Manage your QPU session
```````````````````````

If you want to manually shutdown a QPU session, you can do it by calling the Scaleway's QaaS API:

.. code:: bash
   # List active sessions
   curl -X GET \
      -H "X-Auth-Token: $SCW_SECRET_KEY" \
      "https://api.scaleway.com/qaas/v1alpha1/sessions?project_id=<project_id>"

   # Terminate the session
   curl -X POST \
      -H "X-Auth-Token: $SCW_SECRET_KEY" \
      -H "Content-Type: application/json" \
      -d '{}' \
      "https://api.scaleway.com/qaas/v1alpha1/sessions/{session_id}/terminate"
