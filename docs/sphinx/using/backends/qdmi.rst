QDMI devices
++++++++++++

The `Quantum Device Management Interface (QDMI)
<https://github.com/Munich-Quantum-Software-Stack/QDMI>`__ defines a C
interface for quantum devices and simulators. The interface provides two data
flows:

* A device reports information that CUDA-Q uses to configure its compiler.
* CUDA-Q submits programs as jobs and reads their status and results.

CUDA-Q contains one QDMI target for all QDMI devices. Each device package
supplies a separate QDMI device library. The package contains the
provider-specific communication and authentication. You can install a new
device package without adding a provider-specific target to CUDA-Q.

Supported CUDA-Q operations
``````````````````````````

The QDMI target supports these CUDA-Q operations:

* ``cudaq.sample``
* ``cudaq.observe``
* synchronous execution
* asynchronous execution
* persisted asynchronous jobs, if the device supports job retrieval by ID

The QDMI target does not support CUDA-Q emulation mode, ``cudaq.run``, or
CUDA-Q noise models.

Software components
```````````````````

The integration uses these components:

.. list-table:: QDMI integration components
    :header-rows: 1
    :widths: 24 76

    * - Component
      - Function
    * - CUDA-Q frontend and Quake dialect
      - Build the quantum program from CUDA-Q source code.
    * - CUDA-Q compiler passes
      - Convert gates and map logical qubits to device sites.
    * - CUDA-Q QDMI QPU
      - Convert QDMI metadata to CUDA-Q target configuration, select a program
        format, submit jobs, and convert results.
    * - MQT Core QDMI C++ library
      - Provide a typed C++ interface for QDMI devices and jobs.
    * - MQT Core Driver
      - Find device definitions and load QDMI device libraries.
    * - QDMI device library
      - Implement the QDMI C interface, device metadata, job control, provider
        communication, and authentication.
    * - Quantum service or simulator
      - Execute the submitted program.

The components process a program in this order:

.. code:: text

    CUDA-Q source
        -> Quake IR
        -> CUDA-Q basis conversion and qubit mapping
        -> QIR, OpenQASM, or IQM JSON
        -> CUDA-Q QDMI QPU
        -> MQT Core QDMI C++ library and Driver
        -> QDMI device library
        -> quantum service or simulator

Integration boundary
````````````````````

The QDMI C interface is the boundary between CUDA-Q and a device package. The
device library does not implement a CUDA-Q C++ class. The MQT Core Driver loads
the device library and its QDMI symbols. The MQT Core QDMI C++ library gives
the CUDA-Q QDMI QPU an owning, typed interface to these symbols.

In this model, each QDMI device library is a device plugin. The CUDA-Q QDMI
target is the common adapter for these plugins. Device packages can have their
own release and configuration process. CUDA-Q does not need provider-specific
source code for a new QDMI device.

Build the QDMI target
```````````````````

QDMI support is optional and is disabled by default. Enable it with
``CUDAQ_ENABLE_QDMI_BACKEND``:

.. code:: bash

    cmake -S . -B build \
      -DMLIR_DIR=/path/to/lib/cmake/mlir \
      -DCUDAQ_ENABLE_QDMI_BACKEND=ON
    cmake --build build --target cudaq-qdmi-qpu

The build gets a pinned MQT Core revision with CMake ``FetchContent``. A normal
build includes the QDMI C++ library and Driver. A test build also includes the
MQT Core DDSIM and superconducting (SC) QDMI devices.

Install and register a device
````````````````````````````

Install the QDMI device package from the provider. Each device has a stable
registry ID. Examples include:

* ``mqt.ddsim.default`` for the MQT Core DDSIM device
* ``mqt.sc.default`` for the MQT Core SC device
* ``iqm.default`` for the IQM QDMI device
* ``amazon.braket.default`` for the Amazon Braket QDMI device

The MQT Core Driver reads device definitions from JSON configuration. A
definition contains the stable ID, the device library, the QDMI symbol prefix,
and optional session defaults. A device package can install a relocatable
``*.qdmi.json`` definition with its library.

You can also select a complete registry file:

.. code:: bash

    export MQT_CORE_QDMI_CONFIG_FILE=/path/to/qdmi.json

The following registry defines one device:

.. code:: json

    {
      "schema-version": 1,
      "qdmi": {
        "devices": [
          {
            "id": "example.device",
            "library": "/path/to/libexample-qdmi-device.so",
            "prefix": "EXAMPLE",
            "enabled": true
          }
        ]
      }
    }

The Driver also reads an inline registry from
``MQT_CORE_QDMI_CONFIG_JSON``. Refer to the `MQT Core QDMI configuration guide
<https://mqt.readthedocs.io/projects/core/en/latest/qdmi/configuration.html>`__
for the discovery order and the complete schema.

Keep provider credentials in the provider environment or in the device
configuration. The CUDA-Q target does not store credentials in a persisted
future.

Select and use a device
```````````````````````

Use the stable device ID when you select the QDMI target:

.. tab:: Python

    .. code:: python

        import cudaq

        cudaq.set_target("qdmi", device="iqm.default")

.. tab:: C++

    .. code:: bash

        nvq++ --target qdmi --qdmi-device iqm.default program.cpp

The target opens a new QDMI device session for the selected device.

You can then use the normal CUDA-Q API. This example submits a Bell-state
sampling job to the selected QDMI device:

.. tab:: Python

    .. code:: python

        import cudaq

        cudaq.set_target("qdmi", device="example.device")

        @cudaq.kernel
        def bell():
            qubits = cudaq.qvector(2)
            h(qubits[0])
            x.ctrl(qubits[0], qubits[1])

        result = cudaq.sample(bell, shots_count=100)
        print(result)

The target also provides the generic QDMI custom parameters
``session-custom1`` through ``session-custom5`` and ``job-custom1`` through
``job-custom5``. Refer to the device documentation before you set these
parameters. Their meaning is device-specific.

Configure the compiler from device metadata
``````````````````````````````````````

The QDMI QPU reads a metadata snapshot when CUDA-Q selects the target. CUDA-Q
uses this snapshot as follows:

* The number of sites sets the QPU capacity.
* The device operations set the basis for gate conversion.
* The coupling map controls logical-to-physical qubit mapping.
* The supported program formats control program generation and transport.

The QDMI QPU converts this information to a CUDA-Q target configuration. The
existing CUDA-Q basis-conversion and qubit-mapping passes consume the
configuration. The device library does not call the CUDA-Q compiler.

QDMI devices can report more properties, such as operation duration, operation
fidelity, and calibration state. The QDMI target does not use these properties
to configure the CUDA-Q compiler.

CUDA-Q maps established QDMI gate names to the corresponding Quake gates. It
ignores metadata operations that have no equivalent CUDA-Q gate semantics. The
target reports an error if it cannot construct a compiler basis for the device.

CUDA-Q can split an ``observe`` operation into one kernel for each Pauli term.
This split adds observable-basis rotations. The QDMI target applies device
basis conversion again after the split. Each generated kernel therefore uses
the selected device basis.

The metadata snapshot stays valid until you select another target. Select the
target again after a device calibration changes its operations or connectivity.

Select a program format
```````````````````````

The default ``auto`` selection uses the first supported format in this order:

#. QIR Adaptive module
#. QIR Adaptive string
#. QIR Base module
#. QIR Base string
#. OpenQASM 3
#. OpenQASM 2
#. IQM JSON

You can select a format explicitly. Use ``program_format`` in Python. Use
``--qdmi-program-format`` with ``nvq++``. The accepted values are:

* ``auto``
* ``qir-adaptive-module``
* ``qir-adaptive-string``
* ``qir-base-module``
* ``qir-base-string``
* ``qasm3``
* ``qasm2``
* ``iqm-json``

The target transports each format as follows:

.. list-table:: QDMI program transport
    :header-rows: 1
    :widths: 25 35 40

    * - QDMI format
      - CUDA-Q output
      - Submitted data
    * - QIR Adaptive or Base module
      - QIR bitcode for the selected profile
      - Exact bitcode bytes
    * - QIR Adaptive or Base string
      - QIR bitcode for the selected profile
      - Canonical LLVM IR text
    * - OpenQASM 3
      - OpenQASM 2 source
      - The unchanged source with the QDMI OpenQASM 3 format
    * - OpenQASM 2
      - OpenQASM 2 source
      - The unchanged source with the QDMI OpenQASM 2 format
    * - IQM JSON
      - IQM JSON
      - The unchanged JSON document

The QIR module transport preserves all binary data. The QIR string transport
reports Base64 decoding errors and bitcode parsing errors separately.

Use the IQM device
``````````````````

The IQM QDMI device reads the standard IQM environment. Set ``IQM_TOKEN`` and
``IQM_QC_ALIAS`` as the IQM device documentation specifies. Then select
``iqm.default``.

Automatic format selection normally uses QIR Base string for this device. You
can select ``iqm-json`` to use the native IQM program format. The target creates
a logical-to-physical site mapping for IQM JSON when the job parameters do not
provide one.

Submit and reopen asynchronous jobs
``````````````````````````````````

A same-process asynchronous call uses a CUDA-Q future. The future also contains
the provider job IDs and the CUDA-Q result metadata. You can write the future
to a file after job submission.

Use these steps to reopen the future in another process:

#. Configure the same QDMI device and provider environment.
#. Select the QDMI target with the same stable device ID.
#. Create the receiving future. For an ``observe`` job, give it the same
   observable that was used for submission.
#. Read the stored future into the receiving future.
#. Call ``get()``.

For example, reopen an ``observe`` job as follows:

.. code:: cpp

    cudaq::async_observe_result future(&observable);
    std::ifstream("observe-job.json") >> future;
    const auto result = future.get();

CUDA-Q opens a new QDMI device session and asks the device to retrieve each job
by ID. The immediate and reopened paths use the same result conversion.

The persisted data contains the following information:

* a schema version
* the QDMI result-retriever name
* the stable device ID
* the provider job IDs
* the CUDA-Q result and register names
* the output and qubit-reorder metadata
* the execution type

The persisted data does not contain credentials, native handles, device library
paths, or live QDMI client objects.

Job reopening requires QDMI job retrieval by ID. A device that does not support
this operation returns an unsupported-operation error. Its asynchronous jobs
can still complete in the process that submitted them.

Convert results
```````````````

The QDMI QPU requests histogram results and sequential shot data. If a device
does not provide a histogram, the QPU builds one from the shot data.

The QPU then applies the CUDA-Q output and qubit-reorder metadata. This step
removes compiler-only qubits and restores the CUDA-Q qubit order. For
``observe``, each Pauli term has a separate result register. CUDA-Q combines the
term expectation values with the observable coefficients.

Test the integration
````````````````````

CUDA-Q uses the MQT Core DDSIM and SC devices for tests that do not require
provider credentials. The target tests cover these functions:

* automatic and explicit program-format selection
* QIR module and string transport
* OpenQASM 2 and OpenQASM 3 transport
* sample and observe result conversion
* asynchronous execution
* large DDSIM programs
* unsupported program formats
* unsupported persisted job retrieval

Provider tests are not part of the default CI job. A provider test uses the
normal provider environment and authentication configuration.
