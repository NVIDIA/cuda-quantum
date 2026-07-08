# mock_rest Plugin

`mock_rest` is a reference REST-style external QPU plugin. It demonstrates how
to add a new target by providing a `ServerHelper` while reusing CUDA-Q's built-in
`remote_rest` QPU implementation. It also demonstrates how a plugin author can
build the target as a Python package that can be distributed to and installed
by end users.

This example is intentionally small. It does not simulate the submitted quantum
program. Instead, it sends a configured bitstring to a simple mock REST service
and expects that same bitstring as the sample result.

## What It Demonstrates

- A standalone external plugin directory with `targets/`, `lib/`, and Python
  package metadata generated into `build/external/mock-rest`.
- A custom `cudaq::ServerHelper` registered under the target name `mock_rest`.
- A target YAML file that selects `platform-qpu: remote_rest`.
- Required `cudaq-version` metadata that prevents selection by an older CUDA-Q
  installation and warns across newer major or minor release lines.
- Python auto-discovery through the `cudaq.backends` entry point.
- Distribution as a Python package that end users can install with `pip`.
- Optional registration of the installed package in the user plugin scope for
  `nvq++`.

## How It Works

The key files are:

- `targets/mock_rest.yml.in`: declares the `mock_rest` target, selects
  `remote_rest`, records the CUDA-Q version used to build the plugin, links
  `-lcudaq-rest-qpu`, and exposes `url` and `value` target arguments.
- `MockRestServerHelper.cpp`: implements the REST contract for this backend.
  It builds a job payload containing the compiled kernel, shot count, and the
  configured `value` bitstring. When the server returns that value, the helper
  maps it to CUDA-Q sample counts.
- `python/__init__.py.in`: registers the built plugin root with CUDA-Q when the
  package is discovered through the `cudaq.backends` entry point.
- `python/__main__.py.in`: exposes `python3 -m cudaq_example_mock_rest
  --install-nvqpp`, which installs the built plugin into the user plugin scope.

For local testing, this plugin is meant to talk to the generic `mock_qpu` echo
server in CUDA-Q's Python test utilities. It lives in
`python/tests/utils/mock_qpu/echo/` and is launched through the shared test
server driver:

```sh
python3 python/tests/utils/start_mock_qpu.py echo
```

The echo server accepts any POST payload, stores the payload's `value`, and
returns that same value from `GET /jobs/<id>`.

## Build

From the CUDA-Q repository root:

```sh
cmake -B build \
  -DCUDAQ_EXTERNAL_PROJECTS="mock-rest" \
  -DCUDAQ_EXTERNAL_MOCK_REST_SOURCE_DIR=$PWD/examples/plugins/mock_rest

ninja -C build cudaq-example-serverhelper-mock_rest
```

The built plugin root is:

```text
build/external/mock-rest
```

It contains:

- `targets/mock_rest.yml`
- `lib/libcudaq-serverhelper-mock_rest.so` on Linux, or the platform equivalent
- `pyproject.toml`, `__init__.py`, and `__main__.py` for Python packaging

## Test

Run the plugin lit test:

```sh
ninja -C build check-mock-rest
```

The test checks that the plugin build output exists, the YAML target is valid,
the plugin can be installed into a user plugin scope, `nvq++ --list-targets`
finds it, and Python can load the target through direct registration and package
entry-point discovery.

## Run

Start the echo mock QPU in one shell:

```sh
PYTHONPATH=$PWD/build/python \
python3 python/tests/utils/start_mock_qpu.py echo
```

Then run a Python program in another shell:

```python
import cudaq

cudaq.register_backend_path("build/external/mock-rest")
cudaq.set_target("mock_rest",
                 url="http://localhost:62454",
                 value="00101")

kernel = cudaq.make_kernel()
q = kernel.qalloc()
kernel.h(q)
kernel.mz(q)

counts = cudaq.sample(kernel, shots_count=7)
counts.dump()

assert counts["00101"] == 7
```

`cudaq.sample(...)` submits the REST job and waits for the result. Use
`cudaq.sample_async(...).get()` for the same flow when you want the future-based
API.

## Install as an End User

The build produces a standard Python package containing the target YAML and
shared library. A plugin author can build a wheel from this package, publish it
to a package index, or otherwise distribute it to end users. The commands below
show how an end user installs and enables that distributed target. They are
distinct from the plugin-author build steps above.

Because the package contains a native shared library, it is platform-specific.
Plugin authors must build and distribute a separate package for each operating
system and architecture that they intend to support, and ensure that each wheel
has the corresponding platform tag.

For this local example, install the package directly from its build directory
(an end user would normally install the published package name or wheel):

```sh
python3 -m pip install build/external/mock-rest
```

After the Python package is installed in the end user's environment,
`import cudaq` discovers its `cudaq.backends` entry point, so explicit
`cudaq.register_backend_path(...)` is no longer needed.

To make the installed Python package visible to `nvq++` as well:

```sh
python3 -m cudaq_example_mock_rest --install-nvqpp
```

For C++-only workflows, install the built plugin root directly:

```sh
cudaq-install-plugin build/external/mock-rest
```
