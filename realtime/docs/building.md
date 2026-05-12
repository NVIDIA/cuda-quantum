# Building CUDA-Q Realtime from Source

To build the CUDA-Q Realtime source code locally,
fork this repository and follow the instructions (from the `realtime` directory)
to build the code:

```bash
mkdir build && cd build
cmake -G Ninja .. -DCUDAQ_REALTIME_BUILD_TESTS=ON
# Build
ninja 
# Install
ninja install
```

## Requirements

- CMake 3.22+

- CUDA toolkit (12+)

In the above `cmake` command, we enabled unit testing with `-DCUDAQ_REALTIME_BUILD_TESTS=ON`.
To execute those unit tests, run the `ctest` command from the `build` directory:

```bash
# Test CUDA-Q realtime
ctest 
```

Please check out the tests in the `unittests` folder
for more information about these tests.

> **_NOTE:_** The above build instructions and tests only cover
the basic CUDA-Q Realtime library, e.g., the dispatch library
and host API; no networking transport layer is included.

## Enable Holoscan Sensor Bridge Support

[Holoscan Sensor Bridge](https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/)
(`HSB`) provides a standard API and open-source software that
streams high-speed data directly to GPU memory through FPGA interfaces.

CUDA-Q Realtime supports `HSB`, enabling users to build applications
for realtime coprocessing between FPGA and GPU systems.

### Hardware Requirements

- NVIDIA ConnectX-7/BlueField

- FPGA

### Software Requirements

- `DOCA` version 3.3 with `gpunetio`

Please refer to [the download page](https://developer.nvidia.com/doca-downloads)
to install `DOCA` for your system.

> **_NOTE:_** Please make sure `doca-sdk-gpunetio` is installed along with `doca-all`.

### Build Holoscan Sensor Bridge

To build CUDA-Q Realtime with `HSB`, first, one needs to compile the `HSB` code.

After cloning `HSB` from [GitHub](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/tree/release-2.6.0-EA),
build it with

```bash
# HOLOLINK_DIR is the top-level directory of HSB source code 
cmake -G Ninja -S "$HOLOLINK_DIR" -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
        -DHOLOLINK_BUILD_PYTHON=OFF \
        -DHOLOLINK_BUILD_TESTS=OFF \
        -DHOLOLINK_BUILD_TOOLS=OFF \
        -DHOLOLINK_BUILD_EXAMPLES=OFF \
        -DHOLOLINK_BUILD_EMULATOR=OFF

cmake --build build \
        --target roce_receiver gpu_roce_transceiver hololink_core
```

> **_NOTE:_**  In order to compile Holoscan Sensor Bridge from source,
we need to install all of its dependencies.
Please refer to `HSB` [documentation](https://docs.nvidia.com/holoscan/sensor-bridge/latest/setup.html)
for more details.

<!--- -->

> **_NOTE:_** One can also use the Holoscan Sensor Bridge Docker [container](https://docs.nvidia.com/holoscan/sensor-bridge/latest/build.html)
to build CUDA-Q Realtime.

### Build CUDA-Q Realtime with `HSB`

To enable `HSB`, we can configure `cmake` when building CUDA-Q Realtime as follows:

```bash
cmake -G Ninja -S "$CUDAQ_REALTIME_DIR" -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS=ON \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HOLOLINK_DIR" \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$HOLOLINK_DIR/build"
cmake --build build
```

The `$CUDAQ_REALTIME_DIR` directory is the `realtime`
sub-directory in CUDA-Q source tree.

### Running the FPGA RPC dispatch test

To run the end-to-end RPC dispatch testing between FPGA and GPU
using CUDA-Q Realtime and Holoscan Sensor Bridge,

- Load the `HSB` bit-file into the FPGA.
The bit-file can be obtained from [here](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/tree/release-2.6.0-EA).

- Run the test script (at `cuda-quantum/realtime/unittests/utils/hololink_test.sh`).
For example,

```bash
bash hololink_test.sh --page-size 512 --device mlx5_0 --gpu 0 --bridge-ip 192.168.0.101 --fpga-ip 192.168.0.2 --unified
```

> **_NOTE:_**
> The command line arguments need to be adjusted based on the system setup:
>
> - `--device` is the `IB` device name that is connected to the FPGA.
> - `--gpu` is the GPU device Id that we want to run the RPC callback on.
> - `--fpga-ip` is the IP address of the `HSB` FPGA.
> - `--bridge-ip` is the IP address of the NIC on the host machine.

Upon successful completion, the above script should print out the following:

```text
=== Verification Summary ===
  ILA samples captured:   100
  tvalid=0 (idle):        0
  RPC responses:          100
  Non-RPC frames:         0
  Unique messages verified: 100 of 100
  Responses matched:    100
  Header errors:        0
  Payload errors:       0

=== PTP Round-Trip Latency ===
  Samples:  100
  Min:      3589 ns
  Max:      6348 ns
  Avg:      3872.0 ns
  CSV written: ptp_latency.csv
  RESULT: PASS

=== Shutting down ===
```

> **_NOTE:_** In the above test script, we execute a simple RPC dispatch tests, whereby
the FPGA sends data (array of bytes) to the GPU; the GPU performs a simple increment
by one calculation on each of the byte in the incoming array and returns the array.
We then validate the data and measure the round-trip latency then output
the report as shown above.

<!--- -->

> **_NOTE:_** One can also execute the whole build and execution using
> the validation script as follows:
>
> ```bash
> bash hololink_test.sh --page-size 512 --device mlx5_0 --gpu 0 --bridge-ip 192.168.0.101 --fpga-ip 192.168.0.2 --unified --build  --hololink-dir $HOLOLINK_DIR --cuda-quantum-dir $CUDAQ_DIR
> ```
>
> `$HOLOLINK_DIR` and `$CUDAQ_DIR` are the top-level source directory of Hololink
> and CUDA-Q accordingly.
> Please note that `$CUDAQ_DIR` here is the parent directory
> that contains the `realtime` sub-directory.
