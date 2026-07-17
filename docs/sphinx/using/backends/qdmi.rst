QDMI Devices
++++++++++++

The `Quantum Device Management Interface (QDMI)
<https://github.com/Munich-Quantum-Software-Stack/QDMI>`__ provides a common
interface to communicate with quantum resources.
CUDA-Q can load any QDMI device (library) that supports QIR base-string or
OpenQASM 2 programs and submit ``cudaq.sample`` and ``cudaq.observe`` jobs
through it.

Building the QDMI target
````````````````````````

The QDMI target is optional and requires MQT Core. Install it separately and
enable the target explicitly:

.. code:: bash

    python -m pip install mqt-core
    cmake -S . -B build \
      -DCUDAQ_ENABLE_QDMI_BACKEND=ON \
      -DCMAKE_PREFIX_PATH="$(mqt-core-cli --cmake_dir)"
    cmake --build build

Selecting a device
``````````````````

A QDMI device is selected by its shared library and function prefix.

.. tab:: Python

    .. code:: python

        import cudaq

        cudaq.set_target(
            "qdmi",
            library="/path/to/libqdmi-device.so",
            prefix="DEVICE_PREFIX",
        )

.. tab:: C++

    .. code:: bash

        nvq++ --target qdmi \
          --qdmi-library /path/to/libqdmi-device.so \
          --qdmi-prefix DEVICE_PREFIX program.cpp

The available target arguments are:

* Required: ``library`` and ``prefix``.
* Device session: ``base_url``, ``token``, ``auth_file``, ``auth_url``,
  ``username``, ``password``, and ``session_custom1`` through
  ``session_custom5``.
* Device job: ``job_custom1`` through ``job_custom5``.

Python uses the names above. The corresponding ``nvq++`` options use hyphens
and the ``--qdmi-`` prefix, for example ``--qdmi-base-url``.

Current limitations
```````````````````

The CUDA-Q QDMI target requires support for either ``QDMI_PROGRAM_FORMAT_QIRBASESTRING``
or ``QDMI_PROGRAM_FORMAT_QASM2``. The device must make ``QDMI_DEVICE_PROPERTY_OPERATIONS``
(including each operation's ``QDMI_OPERATION_PROPERTY_NAME``), ``QDMI_DEVICE_PROPERTY_SITES``,
and ``QDMI_DEVICE_PROPERTY_COUPLINGMAP`` queryable for compilation. CUDA-Q emulation mode,
``cudaq.run``, and CUDA-Q noise models are not supported
by this target.
