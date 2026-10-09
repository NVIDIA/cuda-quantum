********************************************
Extending CUDA-Q with a new Hardware Backend
********************************************

This guide explains how to create a new quantum hardware backend for CUDA-Q.
All external backends are developed as **external plugins** — self-contained packages
that register targets with the CUDA-Q runtime without modifying the core
repository. Plugin authors can distribute these plugins as Python packages so
that end users of the target can install them into their own CUDA-Q
environments.

This guide covers the most common backend shape: a **REST-style backend** that
subclasses ``ServerHelper`` to communicate with a provider's REST API, reusing
the built-in ``remote_rest`` QPU.

All backends use the same plugin package layout and distribution mechanism
described in :doc:`packaging`.


Plugin Directory Structure
==========================

Every backend plugin follows this layout:

.. code-block:: text

    my-backend/
    ├── targets/
    │   └── my-backend.yml       # Target configuration
    ├── lib/
    │   └── libcudaq-serverhelper-my-backend.so   # .dylib on macOS
    └── data/                    # Optional auxiliary files
        └── topology.txt

The ``targets/`` directory contains one or more YAML target configurations.
The ``lib/`` directory contains the shared libraries that implement the backend.
The optional ``data/`` directory holds auxiliary files (device topologies, noise
models, calibration data, etc.).

Shared library filenames are written with the Linux ``.so`` extension
throughout this guide; on macOS the extension is ``.dylib``. Name libraries
in ``plugin-libraries:`` without any extension - CUDA-Q appends the one this
platform uses - so that a single target configuration works everywhere.


REST-Style Backends (Server Helper)
===================================

A REST-style backend communicates with a provider's HTTP API. You implement a
``ServerHelper`` subclass that handles authentication, job submission, polling,
and result processing. The built-in ``remote_rest`` QPU handles the execution
lifecycle.

Server Helper Class
-------------------

The server helper is the core component that handles communication with the
quantum hardware provider's API. It extends the ``ServerHelper`` base class and
implements methods for job submission, result retrieval, and other
provider-specific functionality. The base class definition can be found in the
`CUDA-Q repository <https://github.com/NVIDIA/cuda-quantum/blob/main/runtime/common/ServerHelper.h>`_.

Here's a template for implementing a server helper class (see :download:`server_helper_template.cpp <../../snippets/cpp/using/extending/server_helper_template.cpp>` for the complete file):

.. literalinclude:: ../../snippets/cpp/using/extending/server_helper_template.cpp
   :language: cpp

The ``CUDAQ_REGISTER_TYPE`` macro at the bottom registers the helper so that
the runtime can find it by name when the target is activated.

Target YAML Configuration
-------------------------

Create a YAML file that tells CUDA-Q how to activate your target:

.. code-block:: yaml

    # <provider_name>.yml
    name: "<provider_name>"
    description: "CUDA-Q target for Provider Name."
    cudaq-version: "@CUDA_QUANTUM_VERSION@"

    config:
      platform-qpu: remote_rest
      link-libs: ["-lcudaq-rest-qpu"]
      gen-target-backend: true
      preprocessor-defines: ["-D CUDAQ_QUANTUM_DEVICE"]
      # Optional: customize the JIT lowering pipeline for your gate set.
      # Contact the CUDA-Q team for help setting this up.
      jit-mid-level-pipeline: "lower-to-cfg,func.func(canonicalize,multicontrol-decomposition),decomposition{enable-patterns=U3ToRotations},symbol-dce,<provider_name>-gate-set-mapping"
      # Supported values: qir-base, qir-adaptive, qasm2
      codegen-emission: qir-base
      library-mode: false

    # target-arguments are optional; omit the section if your backend needs none.
    target-arguments:
      - key: api-key
        required: true
        type: string
        platform-arg: api_key
        help-string: "API key for Provider Name."
      - key: url
        required: false
        type: string
        platform-arg: url
        help-string: "Specify Provider Name API server URL."
      - key: device
        required: false
        type: string
        platform-arg: device
        help-string: "Specify the Provider Name quantum device to use."

Key fields:

- ``cudaq-version`` — the CUDA-Q version this plugin was built against. Set via
  CMake's ``@CUDA_QUANTUM_VERSION@`` substitution; checked at target-load time
  for compatibility.
- ``platform-qpu: remote_rest`` — use the built-in REST QPU (no custom QPU
  subclass needed).
- ``link-libs`` — libraries to link when compiling with ``nvq++``.
- ``codegen-emission`` — the IR format sent to the provider (``qir-base``,
  ``qir-adaptive``, or ``qasm2``).
- ``target-arguments`` — declares parameters that surface as
  ``--my-backend-api-key <value>`` on the ``nvq++`` command line and as keyword
  arguments to ``cudaq.set_target("my-backend", api_key=...)`` in Python.

For the full list of recognized YAML fields see the mapping traits in
`TargetConfigYaml.cpp <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/lib/Target/Yaml/TargetConfigYaml.cpp>`_.
For a complete working example of a REST-style plugin, see the
`mock_rest reference plugin <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/plugins/mock_rest>`_.

CMake Build File
----------------

A minimal ``CMakeLists.txt`` for a REST-style plugin:

.. code-block:: cmake

    # No cmake_minimum_required / project() — this file runs as an
    # add_subdirectory() child of the CUDA-Q build via CUDAQ_EXTERNAL_PROJECTS,
    # so it inherits the CUDA-Q project scope and all its CMake targets.

    set(plugin_root ${CMAKE_CURRENT_BINARY_DIR})
    set(plugin_lib_dir ${plugin_root}/lib)
    set(plugin_target_dir ${plugin_root}/targets)
    file(MAKE_DIRECTORY ${plugin_lib_dir} ${plugin_target_dir})

    configure_file(targets/my-backend.yml.in
                   ${plugin_target_dir}/my-backend.yml @ONLY)

    add_library(cudaq-serverhelper-my-backend SHARED
      MyBackendServerHelper.cpp)
    set_target_properties(cudaq-serverhelper-my-backend PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${plugin_lib_dir})
    target_include_directories(cudaq-serverhelper-my-backend
      PRIVATE ${PROJECT_SOURCE_DIR}/runtime)
    target_link_libraries(cudaq-serverhelper-my-backend
      PRIVATE cudaq-common cudaq-logger)

When developing inside the CUDA-Q source tree, build your plugin with
``CUDAQ_EXTERNAL_PROJECTS``:

.. code-block:: bash

    cmake -B build \
      -DCUDAQ_EXTERNAL_PROJECTS="my-backend" \
      -DCUDAQ_EXTERNAL_MY_BACKEND_SOURCE_DIR=$PWD/my-backend

    ninja -C build cudaq-serverhelper-my-backend

See :doc:`packaging` for how to build standalone against an installed CUDA-Q.


Auxiliary Files and ``%PLUGIN_ROOT%``
=====================================

Plugins that ship auxiliary files (device topologies, calibration data, noise
models) place them under their package root — typically in a ``data/``
subdirectory. To reference these files portably in the YAML, use the
``%PLUGIN_ROOT%`` substitution token:

.. code-block:: yaml

    config:
      jit-mid-level-pipeline: "qubit-mapping{device=file(%PLUGIN_ROOT%/data/topology.txt)}"

    target-arguments:
      - key: device
        type: string
        default: "%PLUGIN_ROOT%/data/topology.txt"
        platform-arg: device

When the runtime loads the YAML, every ``%PLUGIN_ROOT%`` is replaced with the
absolute path of the plugin package root. This works regardless of where the
package is installed.


Testing Your Backend
====================

Create a ``tests/`` directory in your plugin with lit tests or standalone test
programs that exercise the full lifecycle:

1. Plugin builds successfully
2. Target YAML is valid and discoverable
3. ``nvq++ --target=my-backend`` compiles a program
4. Python can set the target and run a kernel

For REST-style backends, CUDA-Q provides a mock QPU server framework under
``python/tests/utils/`` that you can use to test without real hardware.

See the reference plugins' ``tests/`` directories for concrete examples:

- `mock_rest/tests/ <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/plugins/mock_rest/tests>`_


Example Usage
=============

After an end user installs the distributed plugin package, they interact with
its target like any built-in target. Here, "installed" refers to installing the
plugin in the target user's CUDA-Q environment, not to the plugin author's
build process. See :doc:`packaging` for how to create and distribute the Python
package and for the commands end users run to install it.

.. tab:: Python

    .. code-block:: python

        import cudaq

        cudaq.set_target('my-backend',
                         api_key='your_api_key',
                         device='your_device')

        @cudaq.kernel
        def bell():
            qubits = cudaq.qvector(2)
            h(qubits[0])
            x.ctrl(qubits[0], qubits[1])
            mz(qubits)

        counts = cudaq.sample(bell)
        print(counts)

.. tab:: C++

    .. code-block:: bash

        nvq++ --target=my-backend --my-backend-api-key=... bell.cpp -o bell
        ./bell


Next Steps
==========

Once you have a working backend implementation, see :doc:`packaging` to learn
how to build platform-specific, installable Python wheels, make the plugin
discoverable by ``nvq++``, and distribute it for the operating systems and
architectures required by your users.
