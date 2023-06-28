# Getting Started with Developing CUDA Quantum

This document contains guidelines for contributing to the code in this
repository. This document is relevant primarily for contributions to the `nvq++`
compiler, the CUDA Quantum runtime, or to the integrated simulation backends. If
you would like to contribute applications and examples that use the CUDA Quantum
platform, please follow the instructions for [installing CUDA
Quantum][official_install] instead.

[official_install]: https://nvidia.github.io/cuda-quantum/latest/install.html

## Quick start guide

Before getting started with development, please create a fork of this repository
if you haven't done so already and make sure to check out the latest version on
the `main` branch. After following the instruction for [setting up your
development environment](./Dev_Setup.md) and [building CUDA Quantum from
source](Building.md), you should be able to confirm that you can run the tests
and examples using your local build. If you edit [this
file](./runtime/nvqir/CircuitSimulator.h) to add a print statement

```c++
std::cout << "Custom registration of " << #NAME << "\n" << std::endl;
```

to the definition of the `NVQIR_REGISTER_SIMULATOR` macro, you should see this
line printed when you build the code and run an example using the command

```bash
bash "$CUDAQ_REPO_ROOT/scripts/build_cudaq.sh" && \
nvq++ "$CUDAQ_REPO_ROOT/docs/sphinx/examples/cpp/algorithms/grover.cpp" -o grover.out && \
./grover.out
```

When working on compiler internals, it can be useful to look at intermediate
representations for CUDA Quantum kernels.

To see how the kernels in [this
example](./docs/sphinx/examples/cpp/algorithms/grover.cpp) are translated, you
can run

```bash
cudaq-quake $CUDAQ_REPO_ROOT/docs/sphinx/examples/cpp/algorithms/grover.cpp
```

to see its representation in the Quake MLIR dialect. To see its translation to
[QIR](https://www.qir-alliance.org/), you can run

```bash
cudaq-quake $CUDAQ_REPO_ROOT/docs/sphinx/examples/cpp/algorithms/grover.cpp |
cudaq-opt --canonicalize --quake-add-deallocs |
quake-translate --convert-to=qir
```

## Code style

With regards to code format and style, we distinguish public APIs and CUDA
Quantum internals. Public APIs should follow the style guide of the respective
language, specifically [this guide][cpp_style] for C++, and the [this
guide][python_style] for Python. The CUDA Quantum internals on the other hand
follow the [MLIR/LLVM style guide][llvm_style]. Please ensure that your code
includes comprehensive doc comments as well as a comment at the top of the file
to indicating its purpose.

[python_style]: https://google.github.io/styleguide/pyguide.html
[cpp_style]: https://www.gnu.org/prep/standards/standards.html
[llvm_style]: https://llvm.org/docs/CodingStandards.html

## Testing and debugging

CUDA Quantum tests are categorized as unit tests on runtime library code and
`FileCheck` tests on compiler code output. All code added under the runtime
libraries should have an accompanying test added to the appropriate spot in the
`unittests` folder. All code that directly impacts compiler code should have an
accompanying `FileCheck` test. These tests are located in the `test` folder.

When running a CUDA Quantum executable locally, the verbosity of the output can
be configured by setting the `CUDAQ_LOG_LEVEL` environment variable. Setting its
value to `info` will enable printing of informational messages, and setting its
value to `trace` generates additional messages to trace the execution. These
logs can be directed to a file by setting the `CUDAQ_LOG_FILE` variable when
invoking the executable, e.g.

```bash
CUDAQ_LOG_FILE=grover_log.txt CUDAQ_LOG_LEVEL=info grover.out
```
