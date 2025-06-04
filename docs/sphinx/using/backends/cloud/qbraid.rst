QBRAID
+++++++

.. _qbraid-backend:

Setting Credentials
`````````````````````````

Programmers of CUDA-Q may access the `Qbraid Devices
<https://account.qbraid.com//>`__ from either C++ or Python. Generate
an API key from your `Qbraid account <https://account.qbraid.com//>`__ and export
it as an environment variable:

.. code:: bash

  export QBRAID_API_KEY="qbraid_generated_api_key"


Submission from Python
`````````````````````````

    First, set the :code:`qbraid` backend.

    .. code:: python

        cudaq.set_target('qbraid')

    By default, quantum kernel code will be submitted to the IonQ simulator on qBraid.

   To emulate the qbraid's simulator locally, without submitting through the cloud, you can also set the ``emulate`` flag to ``True``. This will emit any target specific compiler diagnostics.

   .. code:: python

       cudaq.set_target('qbraid', emulate=True)

   The number of shots for a kernel execution can be set through the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default, the ``shots_count`` is set to 1000.

   .. code:: python

       cudaq.sample(kernel, shots_count=10000)

   To see a complete example for using Qbraid's backends, take a look at our :doc:`Python examples <../../examples/examples>`.

Submission from C++
`````````````````````````
        To target quantum kernel code for execution using qbraid,
        pass the flag ``--target qbraid`` to the ``nvq++`` compiler.

        .. code:: bash

                nvq++ --target qbraid src.cpp

        This will take the API key and handle all authentication with, and submission to, the Qbraid device. By default, quantum kernel code will be submitted to the Qbraidsimulator.

        To emulate the qbraid's machine locally, without submitting through the cloud, you can also pass the ``--emulate`` flag to ``nvq++``. This will emit any target  specific compiler diagnostics, before running a noise free emulation.

        .. code:: bash

                nvq++ --emulate --target qbraid src.cpp

        To see a complete example for using IonQ's backends, take a look at our :doc:`C++ examples <../../examples/examples>`.
  