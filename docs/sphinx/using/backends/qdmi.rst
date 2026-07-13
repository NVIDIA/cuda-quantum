QDMI Devices
++++++++++++

The `Quantum Device Management Interface (QDMI)
<https://github.com/Munich-Quantum-Software-Stack/QDMI>`__ provides a common
interface to communicate with quantum resources.
CUDA-Q can load any QDMI device (library) that supports OpenQASM 2 programs and
submit ``cudaq.sample`` and ``cudaq.observe`` jobs through it.

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
and the ``--qdmi-`` prefix, for example ``--qdmi-base-url``. Every argument can
also be supplied through its uppercase ``CUDAQ_QDMI_`` environment variable,
for example ``CUDAQ_QDMI_BASE_URL``.

Current limitations
```````````````````

The current CUDA-Q QDMI target emits OpenQASM 2. The selected device must
advertise support for ``QDMI_PROGRAM_FORMAT_QASM2``. CUDA-Q emulation mode,
``cudaq.run``, and CUDA-Q noise models are not supported by this target.
The target currently exposes the root device from one QDMI device library as
CUDA-Q QPU 0; child devices and multiple QDMI devices are not exposed as
additional CUDA-Q QPUs. The asynchronous ``cudaq.sample_async`` and
``cudaq.observe_async`` APIs are not supported because QDMI does not guarantee
concurrent access to a device session.
