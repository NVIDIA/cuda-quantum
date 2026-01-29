********************************************
Extending CUDA-Q with a new Hardware Backend
********************************************

This guide explains how to integrate a new quantum hardware provider with CUDA-Q. The integration process involves creating a server helper that handles communication with the provider's API, defining configuration files, and implementing necessary tests.

Overview
========

CUDA-Q supports various quantum hardware :doc:`backends <../backends/hardware>` through a REST-based architecture. To add a new backend, you'll need to:

1. Create a server helper class that handles API communication
2. Define configuration files for the target
3. Implement tests to verify functionality
4. Add documentation for users

The following sections detail each step of this process.

Server Helper Implementation
============================

The server helper is the core component that handles communication with the quantum hardware provider's API. It extends the ``ServerHelper`` base class and implements methods for job submission, result retrieval, and other provider-specific functionality. The base class definition can be found in the `CUDA-Q repository <https://github.com/NVIDIA/cuda-quantum/blob/main/runtime/common/ServerHelper.h>`_.

Directory Structure
-------------------

Create the following directory structure for your new backend (replace ``<provider_name>`` with your provider's name):

.. code-block:: text

    runtime/cudaq/platform/default/rest/helpers/<provider_name>/
    ├── CMakeLists.txt
    ├── ProviderNameServerHelper.cpp
    └── <provider_name>.yml

Server Helper Class
-------------------

Here's a template for implementing a server helper class:

.. code-block:: cpp

    // ProviderNameServerHelper.cpp
    #include "common/Logger.h"
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

``CMakeLists.txt``
------------------

You will need to configure CUDA-Q's ``cmake`` system for your new server helper. By convention, you should setup your target as optional by adding a CMake flag in the ``CMakeLists.txt`` at the root of the CUDA-Q repository:

.. code-block:: cmake

    # Enable <provider_name> target by default
    if (NOT DEFINED CUDAQ_ENABLE_PROVIDER_NAME_BACKEND)
      set(CUDAQ_ENABLE_PROVIDER_NAME_BACKEND ON CACHE BOOL "Enable building the <Provider Name> target.")
    endif()

Then, create a ``CMakeLists.txt`` file in your server helper's directory and check for this flag:

.. code-block:: cmake

    if(CUDAQ_ENABLE_PROVIDER_NAME_BACKEND)
      target_sources(cudaq-rest-qpu PRIVATE ProviderNameServerHelper.cpp)
      add_target_config(<provider_name>)
      
      add_library(cudaq-serverhelper-<provider_name> SHARED ProviderNameServerHelper.cpp)
      target_link_libraries(cudaq-serverhelper-<provider_name>
        PUBLIC
        cudaq-common
        fmt::fmt-header-only
      )
      install(TARGETS cudaq-serverhelper-<provider_name> DESTINATION lib)
    endif()

Target Configuration
====================

Create a ``YAML`` configuration file for your target:

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
      # Library mode is only for simulators, physical backends must turn this off
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

Update Parent ``CMakeLists.txt``
--------------------------------

Add your provider to the parent ``CMakeLists.txt`` file:

.. code-block:: cmake

    # runtime/cudaq/platform/default/rest/helpers/CMakeLists.txt
    add_subdirectory(<provider_name>)
    add_subdirectory(ionq)
    add_subdirectory(iqm)
    # ... other providers

Testing
=======

Unit Tests
----------

Create unit tests for your server helper:

1. Create a directory structure:

.. code-block:: text

    unittests/backends/<provider_name>/
    ├── CMakeLists.txt
    ├── ProviderNameStartServerAndTest.sh.in
    └── ProviderNameTester.cpp

2. Implement the test files:

.. code-block:: cmake

    # CMakeLists.txt
    add_backend_unittest_executable(ProviderNameTester 
      SOURCES ProviderNameTester.cpp
      BACKEND ProviderName
      BACKEND_CONFIG "<provider_name> url=http://localhost:<PORT>"
    )
    
    configure_file(ProviderNameStartServerAndTest.sh.in
                   ProviderNameStartServerAndTest.sh @ONLY)
    
    add_test(NAME ProviderNameTester COMMAND bash ProviderNameStartServerAndTest.sh)
    set_tests_properties(ProviderNameTester PROPERTIES TIMEOUT 120)

3. Create a shell script to start the mock server and run tests:

.. code-block:: bash

    #!/bin/bash
    
    # Start the mock server
    python3 utils/start_mock_qpu.py <provider_name> &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 2
    
    # Run the test
    @CMAKE_CURRENT_BINARY_DIR@/ProviderNameTester
    TEST_STATUS=$?
    
    # Kill the server
    kill $SERVER_PID
    
    # Return the test status
    exit $TEST_STATUS

4. Implement the C++ test:

.. code-block:: cpp

    // ProviderNameTester.cpp
    #include "common/Logger.h"
    #include "common/RestClient.h"
    #include "common/ServerHelper.h"
    #include "cudaq/platform/quantum_platform.h"
    #include "gtest/gtest.h"
    
    TEST(ProviderNameTester, checkSimpleCircuit) {
      // Create a simple circuit
      auto kernel = cudaq::make_kernel();
      auto qubits = kernel.qalloc(2);
      kernel.h(qubits[0]);
      kernel.cx(qubits[0], qubits[1]);
      kernel.mz(qubits);
      
      // Execute the circuit
      auto counts = cudaq::sample(kernel);
      
      // Check results
      EXPECT_EQ(counts.size(), 2);
      EXPECT_TRUE(counts.has_key("00"));
      EXPECT_TRUE(counts.has_key("11"));
    }

To make sure the C++ tests don't run if your target is not enabled, add the following to ``targettests/lit.site.cfg.py.in``:

.. code-block:: python

    config.cudaq_backends_provider = "@CUDAQ_ENABLE_PROVIDER_NAME_BACKEND@"
    if cmake_boolvar_to_bool(config.cudaq_backends_provider):
        config.available_features.add('provider')
        config.substitutions.append(('%provider_avail', 'true'))
    else:
        config.substitutions.append(('%provider_avail', 'false'))

And add the following to your ``targettests`` ``.cpp`` file:

.. code-block:: cpp

    // RUN: if %provider_avail; then nvq++ --target provider %s -o %t.x; fi

Mock Server
-----------

Create a mock server for testing:

.. code-block:: text

    utils/mock_qpu/<provider_name>/
    └── __init__.py

Implement the mock server:

.. code-block:: python

    # __init__.py
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json
    import sys
    import time
    
    class ProviderNameMockServer(BaseHTTPRequestHandler):
        def _set_headers(self, status_code=200):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
    
        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            if self.path == '/jobs':
                # Create a job
                response = {
                    'id': 'job-123',
                    'status': 'QUEUED'
                }
                self._set_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
        def do_GET(self):
            if self.path.startswith('/jobs/job-123'):
                # Return job status and results
                response = {
                    'id': 'job-123',
                    'status': 'COMPLETED',
                    'results': {
                        'counts': {
                            '00': 500,
                            '11': 500
                        }
                    }
                }
                self._set_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    def startServer(port=8000):
        server_address = ('', port)
        httpd = HTTPServer(server_address, ProviderNameMockServer)
        print(f'Starting mock server on port {port}...')
        httpd.serve_forever()
    
    if __name__ == '__main__':
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
        startServer(port)

Python Tests
------------

Create Python tests for your backend:

.. code-block:: python

    # python/tests/backends/test_<provider_name>.py
    import os
    import sys
    import time
    import pytest
    from multiprocessing import Process
    
    import cudaq
    from cudaq import spin

    skipIf<provider_name>NotInstalled = pytest.mark.skipif(
        not (cudaq.has_target("<provider_name>")),
        reason='Could not find `<provider_name>` in installation')
    
    try:
        from utils.mock_qpu.<provider_name> import startServer
    except:
        print("Mock qpu not available, skipping Provider Name tests.")
        pytest.skip("Mock qpu not available.", allow_module_level=True)
    
    # Define the port for the mock server - make sure this is unique
    # across all tests.
    port = 62444
    
    @pytest.fixture(scope="session", autouse=True)
    def startUpMockServer():
        # Set the targeted QPU
        cudaq.set_target('<provider_name>',
                        url=f'http://localhost:{port}',
                        api_key="test_key")
        
        # Launch the Mock Server
        p = Process(target=startServer, args=(port,))
        p.start()
        time.sleep(1)
        
        yield "Running the tests."
        
        # Kill the server
        p.terminate()
    
    def test_<provider_name>_sample():
        # Create the kernel
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(2)
        kernel.h(qubits[0])
        kernel.cx(qubits[0], qubits[1])
        kernel.mz(qubits)
        
        # Run sample
        counts = cudaq.sample(kernel)
        assert len(counts) == 2
        assert '00' in counts
        assert '11' in counts
        
        # Run sample asynchronously
        future = cudaq.sample_async(kernel)
        counts = future.get()
        assert len(counts) == 2
        assert '00' in counts
        assert '11' in counts

Integration Tests
-----------------

To ensure proper execution on the hardware, a validation backend must be provided that:

1. Consumes the same format that your target will use
2. Validates that circuits passing this validation will execute successfully on the actual hardware
3. Can be accessed by CUDA-Q's GitHub CI/CD pipelines

Your validation backend doesn't need to be publicly available, but it should:

- Accept the same input format as your actual quantum processor
- Return meaningful error messages for invalid circuits
- Provide an API endpoint that can be called from our integration tests

If your validation backend is not publicly available, please coordinate the exchange of necessary credentials for CI/CD with the CUDA-Q team.

Add your target to ``.github/workflows/integration_tests.yml``:

.. code-block:: yaml

    - name: Submit to <provider_name> test server
      if: (success() || failure()) && (inputs.target == '<provider_name>' || github.event_name == 'schedule')
      run: |
        echo "### Submit to <provider_name> server" >> $GITHUB_STEP_SUMMARY
        # Set up any required dependencies        
        # Set up environment variables for authentication
        export PROVIDER_API_KEY='${{ secrets.PROVIDER_API_KEY }}'
        
        # Run the integration tests
        python_tests="docs/sphinx/targets/python/<provider_name>.py"
        cpp_tests="docs/sphinx/targets/cpp/<provider_name>.cpp"
        
        # Execute tests (see other provider examples for implementation details)
        # ...

Documentation
=============

Add documentation for your backend in the appropriate sections of the CUDA-Q documentation. This should include:

1. How to access your server (authentication set up, documentation etc.)
2. How to configure and use the backend
3. Any provider-specific parameters or features
4. Examples of running circuits on the backend
5. Adding your logo to the diagram on :doc:`../backends/hardware`

More specifically, you will need to modify at least the following files:

* ``docs/sphinx/using/examples/hardware_providers.rst``
* ``docs/sphinx/using/backends/hardware.rst``
* ``docs/sphinx/using/backends/hardware/<your-technology>.rst``
* ``docs/sphinx/targets/python/<provider_name>.py``
* ``docs/sphinx/targets/cpp/<provider_name>.cpp``

Example Usage
=============

Once your backend is implemented, users can use it as follows:

.. code-block:: python

    import cudaq
    
    # Set the target to your provider
    cudaq.set_target('<provider_name>', 
                    api_key='your_api_key',
                    device='your_device')
    
    # Create and run a circuit
    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)
    
    # Run the circuit
    counts = cudaq.sample(bell)
    print(counts)

Code Review
===========

Once you have implemented a ``ProviderNameServerHelper``, some basic tests, and documentation, please `create a PR <https://github.com/NVIDIA/cuda-quantum/pulls>`_ with your changes and tag the CUDA-Q team for review.

Maintaining a Backend
=====================

Once your backend is integrated, you will need to maintain it. This includes:

* Fixing bugs (in your integration, tests, or documentation)
* Adding new features
* Integrating with new CUDA-Q features (if additional integration is needed to use them)

This is where having extensive tests against real hardware comes in handy. The benefits are two-fold:

* It allows the CUDA-Q team to roll out new features without breaking your backend integration
* It allows you to validate compatibility with CUDA-Q before rolling out a new version of your backend

Conclusion
==========

By following this guide, you can integrate a new quantum hardware provider with CUDA-Q. The integration involves creating a server helper, defining configuration files, implementing tests, adding documentation, and going through a code review process. Once integrated, users can seamlessly run quantum circuits on your provider's hardware using the CUDA-Q framework.
