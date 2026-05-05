qBraid
++++++

.. _qbraid-backend:

`qBraid <https://www.qbraid.com/>`__ is a cloud platform that brokers access to
quantum simulators and hardware from multiple vendors through a single API.
CUDA-Q can submit OpenQASM 2 jobs to any device exposed by the qBraid service.
See the `qBraid device catalog <https://account.qbraid.com/devices>`__ for the
set of simulators and QPUs currently available.

Setting Credentials
```````````````````

Generate an API key from your `qBraid account <https://account.qbraid.com/>`__
and export it as an environment variable:

.. code:: bash

    export QBRAID_API_KEY="qbraid_generated_api_key"

Alternatively, the API key can be passed directly to ``cudaq.set_target`` via
the ``api_key`` argument (see below).

Submitting
``````````

.. tab:: Python

    The target to which quantum kernels are submitted can be controlled with
    the ``cudaq.set_target()`` function.

    .. code:: python

        cudaq.set_target("qbraid")

    By default, jobs are submitted to the qBraid state vector simulator
    (``qbraid:qbraid:sim:qir-sv``).

    To specify a different qBraid device, set the ``machine`` parameter to its
    qBraid device ID.

    .. code:: python

        cudaq.set_target("qbraid", machine="qbraid:qbraid:sim:qir-sv")

    The API key can also be supplied inline instead of through the
    ``QBRAID_API_KEY`` environment variable.

    .. code:: python

        cudaq.set_target("qbraid", api_key="qbraid_generated_api_key")

    qBraid devices are cloud-hosted, so local emulation via the ``emulate``
    flag is not supported — all jobs are executed on the qBraid service.
    To run without submitting to real hardware, select one of the qBraid
    simulator devices (for example, ``qbraid:qbraid:sim:qir-sv``) via the
    ``machine`` argument.

    The number of shots for a kernel execution can be set through the
    ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. The
    default is 1000.

    .. code:: python

        cudaq.sample(kernel, shots_count=10000)

.. tab:: C++

    To target quantum kernel code for execution on qBraid, pass the flag
    ``--target qbraid`` to the ``nvq++`` compiler. By default jobs are
    submitted to the qBraid state vector simulator
    (``qbraid:qbraid:sim:qir-sv``).

    .. code:: bash

        nvq++ --target qbraid src.cpp

    To execute kernels on a different device, pass ``--qbraid-machine`` with
    the qBraid device ID:

    .. code:: bash

        nvq++ --target qbraid --qbraid-machine "qbraid:qbraid:sim:qir-sv" src.cpp

    The API key can be passed explicitly with ``--qbraid-api_key`` instead of
    being read from ``QBRAID_API_KEY``:

    .. code:: bash

        nvq++ --target qbraid --qbraid-api_key "qbraid_generated_api_key" src.cpp

    qBraid devices are cloud-hosted, so the ``--emulate`` flag is not
    supported for this target — all jobs are executed on the qBraid
    service. To run without submitting to real hardware, pass
    ``--qbraid-machine`` with a qBraid simulator device ID (for example,
    ``qbraid:qbraid:sim:qir-sv``).

To see a complete example for using qBraid's backends, take a look at our
:doc:`Python examples <../../examples/examples>` and
:doc:`C++ examples <../../examples/examples>`.
