# CUDA-Q Realtime Installation Guide

The following page describes the installation procedure of
CUDA-Q Realtime, including connectivity to a
[Holoscan Sensor Bridge](https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/)
(`HSB`) FPGA.

## Components

### Hardware Components

- A host system with NVIDIA GPU and ConnectX-7/BlueField NIC.

- A FPGA, programmed with `HSB` IP and connected to the NIC.

> **_NOTE:_** We recommended using NVIDIA ConnectX-7 as prior generations
may not have all the required capabilities.

### Software Components

- CUDA-Q Realtime installer.

- CUDA Runtime (12+)

- [`DOCA` 3.3.0 installation](https://developer.nvidia.com/doca-downloads)
with `gpunetio` support.

> **_NOTE:_** `DOCA` is required to run the end-to-end validation with FPGA
using the builtin `HSB` support of CUDA-Q realtime.

<!--- -->

> **_NOTE:_** Please make sure `doca-sdk-gpunetio` is installed along with `doca-all`.

## Setup

To install CUDA-Q Realtime with Holoscan Sensor Bridge on a host machine
(bare-metal), please follow these steps.

> **_NOTE:_** Alternatively, we can also build and run these steps in a Docker container.
Please refer to this [section](#using-docker) for instructions.

1. Install CUDA-Q Realtime (if not already done so)

    For example,

    ```bash
    ./install_cuda_quantum_realtime_cu13.arm64  --accept
    ```

    > **_NOTE:_** Please verify that CUDA-Q Realtime has been installed to `/opt/nvidia/cudaq/realtime`.

    <!--- -->

    > **_NOTE:_** After the installation, please follow the instructed
    > post-installation step to set the environment variable, e.g.,
    >
    > ```bash
    >  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/cudaq/realtime/lib
    > ```

2. Load `HSB` IP bit-file to the FPGA

    The bit-file for supported FPGA vendors
    can be found [here](https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/QEC/HSB-2.6.0-EA/).

    > **_NOTE:_** Please make sure set up the [host system](https://docs.nvidia.com/holoscan/sensor-bridge/latest/setup.html)
    and the `HSB` FPGA device [IP address](https://docs.nvidia.com/holoscan/sensor-bridge/latest/architecture.html#datachannel-enumeration-and-ip-address-configuration)
    (if not already done so).

3. Run the validation script

    The validation script is located at `/opt/nvidia/cudaq/realtime/validate.sh`.

    ```bash
    bash /opt/nvidia/cudaq/realtime/validate.sh --page-size 512 --device mlx5_0 --gpu 0 --bridge-ip 192.168.0.101 --fpga-ip 192.168.0.2 --unified 
    ```

    > **_NOTE:_**
    > The command line arguments need to be adjusted based on the system setup:
    >
    > - `--device` is the `IB` device name that is connected to the FPGA.
    > - `--gpu` is the GPU device Id that we want to run the RPC callback on.
    > - `--fpga-ip` is the IP address of the `HSB` FPGA.
    > - `--bridge-ip` is the IP address of the NIC on the host machine.
    > - `--page-size` is the ring buffer slot size in bytes.
    > - `--unified` is the flag to enable the unified dispatch mode.
    The other option is `--forward`, which skips the RPC callback.
    Without either of these flags, we will run the test in
    the three-kernel mode with the RPC handler.

    Upon successful completion, the above validation script should
    print out the following:

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

Congratulations! You have successfully validated the CUDA-Q Realtime installation.

> **_NOTE:_** In the above test script, we execute a simple RPC dispatch tests, whereby
the FPGA sends data (array of bytes) to the GPU; the GPU performs
a simple increment by one calculation on each of the byte
in the incoming array (unless in the `--forward` mode) and returns the array.
We validate the data and measure the round-trip latency
then output the report as shown above.

## Using Docker

In the CUDA-Q Realtime installation, the `demo.sh` script will
build a containerized environment containing necessary dependencies
for CUDA-Q Realtime.

For example,

```bash
bash /opt/nvidia/cudaq/realtime/demo.sh 
```

will transfer the local CUDA-Q installation into that containerized environment.

Inside the container, we can then run the validation check, i.e.,

```bash
bash /opt/nvidia/cudaq/realtime/validate.sh --page-size 512 --device mlx5_0 --gpu 0 --bridge-ip 192.168.0.101 --fpga-ip 192.168.0.2 --unified 
```

### Manual Installation in Docker Container

1. Launch your container with networking and GPU support.

    For example, `--net host --gpus all` should be used to launch the container.

    > **_NOTE:_** Depending on the host system configurations, the `--privileged`
    flag may also be required so that the container can access the `IB` devices
    of the host.

2. Install CUDA runtime.

3. Install [`DOCA`](https://developer.nvidia.com/doca-downloads)
with `gpunetio` (`doca-sdk-gpunetio`) support.

4. Download and install CUDA-Q Installer as described in the [setup](#setup) section.

## Using a Custom Networking Implementation

CUDA-Q Realtime allows users to extend the networking layer with
their own hardware and software components,
in addition to one based on Holoscan Sensor Bridge.

Please refer to this [guide](cudaq_realtime_network_interface.md)
for more information about extending the networking interface
for your custom transport protocol.

### Measuring latency

In other to measure the latency with a custom networking implementation,
we need to construct a stimulus (data generation) tool to send data
to CUDA-Q realtime according to the custom networking protocol.

For example, in the `HSB`-based implementation, we use the `ptp_timestamp` field
in the `RPCHeader`/`RPCResponse` (see the message protocol [documentation](cudaq_realtime_message_protocol.md))
to capture the timestamp for latency analysis. Specifically, the stimulus tool (FPGA)
stores the 'send' timestamp in the `RPCHeader` (incoming message),
which will be echoed by the GPU in the outgoing `RPCResponse`
after processing it (e.g., with the RPC handler).
Using the Integrated Logic Analyzer (`ILA`) timestamp when the FPGA receives
the response from the GPU, we can compute the round-trip latency,
i.e., the elapsed time from the timestamp in the header to the `ILA` receive timestamp.

Please refer to `hololink_fpga_playback.cpp` code in the [CUDA-Q repository](https://github.com/NVIDIA/cuda-quantum)
for a sample of data generation tools.

## Troubleshooting

<!-- markdownlint-disable MD013 -->

### Error "`error while loading shared libraries: libcudaq-realtime-bridge-hololink.so cannot open shared object file: No such file or directory`"

This can be resolved by following the post-installation step in the [setup](#setup) section to set the `LD_LIBRARY_PATH`.

### Error "`error while loading shared libraries: libdoca_gpunetio.so.2: cannot open shared object file: No such file or directory`"

This can be resolved by installing the `doca-sdk-gpunetio` package.

### Error "`Cannot find GID for RoCE v2`"

Please make sure the `IB` device name (`--device`) is correct, i.e., the one connected to the `HSB` FPGA.

### Error "`Failed to get remote MAC.`"

Please make sure the IP address of the `HSB` FPGA (`--fpga-ip`) is correct. For example, we can do a quick `ping` test to check the connectivity.
