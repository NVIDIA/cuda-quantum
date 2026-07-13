QDMI Devices
++++++++++++

The `Quantum Device Management Interface (QDMI)
<https://github.com/Munich-Quantum-Software-Stack/QDMI>`__ provides a common
interface for local simulators, remote simulators, and quantum hardware.
CUDA-Q can load any QDMI device library that supports OpenQASM 2 programs and
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

Configuration fails if MQT Core is unavailable. Standard CUDA-Q builds and
packages remain independent of MQT Core.

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

The library path and prefix can instead be supplied through
``CUDAQ_QDMI_LIBRARY`` and ``CUDAQ_QDMI_PREFIX``. Device-session parameters can
be supplied with the target arguments ``base_url``, ``token``, ``auth_file``,
``auth_url``, ``username``, ``password``, and ``session_custom1`` through
``session_custom5``. The corresponding environment variables use the
``CUDAQ_QDMI_`` prefix, for example ``CUDAQ_QDMI_TOKEN``.

Current limitations
```````````````````

The current CUDA-Q QDMI target emits OpenQASM 2. The selected device must
advertise support for ``QDMI_PROGRAM_FORMAT_QASM2``. CUDA-Q emulation mode,
``cudaq.run``, and CUDA-Q noise models are not supported by this target.
The asynchronous ``cudaq.sample_async`` and ``cudaq.observe_async`` APIs are
also not supported because QDMI does not guarantee concurrent access to a
device session.
