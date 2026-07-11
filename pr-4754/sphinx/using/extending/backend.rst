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
    │   └── libcudaq-serverhelper-my-backend.so   # (or libcudaq-qpu-my-backend.so)
    └── data/                    # Optional auxiliary files
        └── topology.txt

The ``targets/`` directory contains one or more YAML target configurations.
The ``lib/`` directory contains the shared libraries that implement the backend.
The optional ``data/`` directory holds auxiliary files (device topologies, noise
models, calibration data, etc.).


REST-Style Backends (ServerHelper)
==================================

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

Here's a template for implementing a server helper class:

.. code-block:: cpp

    // ProviderNameServerHelper.cpp
    #include "cudaq/runtime/logger/logger.h"
    #include "common/RestClient.h"
    #include "common/ServerHelper.h"
    #include "cudaq/Support/Version.h"
    #include "cudaq/utils/cudaq_utils.h"
    #include <bitset>
    #include <fstream>
    #include <iostream>
    #include <map>
    #include <regex>
    #include <sstream>
    #include <thread>
    #include <unordered_set>
    
    using json = nlohmann::json;
    
    namespace cudaq {
    
    /// @brief The ProviderNameServerHelper class extends the ServerHelper class
    /// to handle interactions with the Provider Name server for submitting and
    /// retrieving quantum computation jobs.
    class ProviderNameServerHelper : public ServerHelper {
      static constexpr const char *DEFAULT_URL = "https://api.provider-name.com";
      static constexpr const char *DEFAULT_VERSION = "v1.0";
    
    public:
      const std::string name() const override { return "<provider_name>"; }
    
      /// @brief Example implementation of authentication headers.
      RestHeaders getHeaders() override {
        RestHeaders headers;
        headers["Content-Type"] = "application/json";
        
        // Add authentication headers if needed
        if (backendConfig.count("api_key"))
          headers["Authorization"] = "Bearer " + backendConfig["api_key"];
        
        return headers;
      }
    
      /// @brief Example implementation of backend initialization.
      void initialize(BackendConfig config) override {
        CUDAQ_INFO("Initializing Provider Name Backend");
        backendConfig = config;
        
        if (!backendConfig.count("url"))
          backendConfig["url"] = DEFAULT_URL;
        if (!backendConfig.count("version"))
          backendConfig["version"] = DEFAULT_VERSION;

        // Set shots if provided
        if (config.find("shots") != config.end())
          this->setShots(std::stoul(config["shots"]));
      }
    
      /// @brief Example implementation of simple job creation.
      ServerJobPayload createJob(std::vector<KernelExecution> &circuitCodes) override {
        ServerMessage job;
        job["content"] = circuitCodes[0].code;
        job["shots"] = shots;
        
        RestHeaders headers = getHeaders();
        std::string path = "/jobs";
        
        return std::make_tuple(backendConfig["url"] + path, headers, 
                              std::vector<ServerMessage>{job});
      }
    
      /// @brief Example implementation of job ID tracking.
      std::string extractJobId(ServerMessage &postResponse) override {
        if (!postResponse.contains("id"))
          return "";
        
        return postResponse.at("id");
      }
    
      /// @brief Example implementation of job ID tracking.
      std::string constructGetJobPath(ServerMessage &postResponse) override {
        return extractJobId(postResponse);
      }
    
      /// @brief Example implementation of job ID tracking.
      std::string constructGetJobPath(std::string &jobId) override {
        return backendConfig["url"] + "/jobs/" + jobId;
      }
    
      /// @brief Example implementation of job status checking.
      bool jobIsDone(ServerMessage &getJobResponse) override {
        if (!getJobResponse.contains("status"))
          return false;
        
        std::string status = getJobResponse["status"];
        return status == "COMPLETED" || status == "FAILED";
      }
    
      /// @brief Example implementation of result processing.
      ///
      /// The raw results from quantum hardware often need post-processing (bit
      /// reordering, normalization, etc.) to match CUDA-Q's expectations.
      /// This is the place to do that.
      cudaq::sample_result processResults(ServerMessage &getJobResponse,
                                         std::string &jobId) override {
        CUDAQ_INFO("Processing results: {}", getJobResponse.dump());
        
        // Extract measurement results from the response
        auto samplesJson = getJobResponse["results"]["counts"];
        cudaq::CountsDictionary counts;
        
        for (auto &item : samplesJson.items()) {
          std::string bitstring = item.key();
          std::size_t count = item.value();
          counts[bitstring] = count;
        }
        
        // Create an ExecutionResult
        cudaq::ExecutionResult execResult{counts};
        
        // Return the sample_result
        return cudaq::sample_result{execResult};
      }
    
      /// @brief Example implementation of polling configuration.
      std::chrono::microseconds
      nextResultPollingInterval(ServerMessage &postResponse) override {
        return std::chrono::seconds(5);
      }
    };
    
    } // namespace cudaq
    
    // Register the server helper in the CUDA-Q server helper factory
    CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::ProviderNameServerHelper, <provider_name>)

The ``CUDAQ_REGISTER_TYPE`` macro at the bottom registers the helper so that
the runtime can find it by name when the target is activated.

Target YAML Configuration
-------------------------

Create a YAML file that tells CUDA-Q how to activate your target:

.. code-block:: yaml

    # <provider_name>.yml
    name: "<provider_name>"
    description: "CUDA-Q target for Provider Name."
    
    config:
      # Tell DefaultQuantumPlatform what QPU subtype to use
      platform-qpu: remote_rest
      # Add the rest-qpu library to the link list
      link-libs: ["-lcudaq-rest-qpu"]
      # Tell NVQ++ to generate glue code to set the target backend name
      gen-target-backend: true
      # Add preprocessor defines to compilation
      preprocessor-defines: ["-D CUDAQ_QUANTUM_DEVICE"]
      # Define the JIT lowering pipeline
      # This will cover applying hardware-specific constraints since each provider may have different native gate sets, requiring custom mappings and decompositions. You may need assistance from the CUDA-Q team to set this up correctly.
      jit-mid-level-pipeline: "lower-to-cfg,func.func(canonicalize,multicontrol-decomposition),decomposition{enable-patterns=U3ToRotations},symbol-dce,<provider_name>-gate-set-mapping"
      # Tell the rest-qpu that we are generating QIR base profile.
      # As of the time of this writing, qasm2, qir-base and qir-adaptive are supported.
      codegen-emission: qir-base
      library-mode: false

    # Some examples of target arguments are shown below.
    # You do not need to add any arguments for your backend if you do not need them.
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

- ``platform-qpu: remote_rest`` — use the built-in REST QPU (no custom QPU
  subclass needed).
- ``link-libs`` — libraries to link when compiling with ``nvq++``.
- ``codegen-emission`` — the IR format sent to the provider (``qir-base``,
  ``qir-adaptive``, or ``qasm2``).
- ``target-arguments`` — declares parameters that surface as
  ``--my-backend-api-key <value>`` on the ``nvq++`` command line and as keyword
  arguments to ``cudaq.set_target("my-backend", api_key=...)`` in Python.

For a complete working example of a REST-style plugin, see the
`mock_rest reference plugin <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/plugins/mock_rest>`_.

CMakeLists.txt
--------------

A minimal ``CMakeLists.txt`` for a REST-style plugin:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.22)
    project(my-backend-plugin)

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
