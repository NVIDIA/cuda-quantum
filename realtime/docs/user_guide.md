# CUDA-Q Realtime Installation Guide

The following page describes the installation procedure of
CUDA-Q Realtime, including connectivity to a
[Holoscan Sensor Bridge](https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/)
(`HSB`) FPGA.

## Components

### Hardware Components

- A host system with NVIDIA GPU and ConnectX-7/BlueField NIC.

- An FPGA, programmed with `HSB` IP and connected to the NIC.

> **_NOTE:_** We recommend using NVIDIA ConnectX-7 as prior generations
may not have all the required capabilities.

### Software Components

- CUDA-Q Realtime installer.

- CUDA Runtime (12.6+ and 13.0+)

- For GPU RoCE: [`DOCA` 3.4.0 installation](https://developer.nvidia.com/doca-downloads)
with `gpunetio` support.

> **_NOTE:_** `DOCA` is required for GPU RoCE validation. CPU RoCE requires
`libibverbs` instead. HSB playback requires the CUDA driver library
(`libcuda.so.1`) with either bridge.

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

    > **_NOTE:_** To install to a different location (which does not require
    > `sudo`), pass `--installpath` to the installer, e.g.,
    >
    > ```bash
    > ./install_cuda_quantum_realtime_cu13.arm64 --accept -- --installpath $HOME/.cudaq_realtime
    > ```
    >
    > The paths used in the remaining steps then need to be adjusted accordingly.

    <!--- -->

    > **_NOTE:_** After the installation, please follow the instructed
    > post-installation step to set the environment variable, e.g.,
    >
    > ```bash
    >  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/cudaq/realtime/lib
    > ```

    The installed validation applications locate their libraries automatically
    and do not require this environment setting.

2. Program the FPGA with `HSB`

    See the [CUDA-Q Realtime installation docs](https://nvidia.github.io/cuda-quantum/latest/using/realtime/installation.html#realtime-hsb-fpga-artifacts)
    for the reusable `nv_hsb_ip` RTL source, the RFSoC PYNQ source example,
    and the pre-built RFSoC bit-file.

    > **_NOTE:_** Please make sure to set up the [host system](https://docs.nvidia.com/holoscan/sensor-bridge/getting-started/host-setup)
    and the `HSB` FPGA device [IP address](https://docs.nvidia.com/holoscan/sensor-bridge/applications/architecture#datachannel-enumeration-and-ip-address-configuration)
    (if not already done so).

3. Run the validation script

    The validation script is located at `/opt/nvidia/cudaq/realtime/validate.sh`.

    ```bash
    bash /opt/nvidia/cudaq/realtime/validate.sh --dispatch=unified \
      --bridge-arg="--device=mlx5_0 --peer-ip=192.168.0.2"
    ```

    > **_NOTE:_**
    > The command line arguments need to be adjusted based on the system setup:
    >
    > - `--device` is the `IB` device name that is connected to the FPGA.
    > - `--peer-ip` is the IP address of the `HSB` FPGA.
    > - `--gpu-id N` selects the GPU device (default: 0); `--cpu` selects CPU dispatch.
    > - `--dispatch=unified` enables unified dispatch.
    > - `--threads N` is the block size (threads per block) that every
    >   DEVICE_CALL handler runs on, in both dispatch modes (default: 32).
    The other option is `--dispatch=forward`, which skips the RPC callback.
    Without a dispatch option, we will run the test in
    the three-kernel mode with the RPC handler.

    RoCE with GPU execution is the default. For a local `UDP` check without an
    FPGA, run:

    ```bash
    bash /opt/nvidia/cudaq/realtime/validate.sh --udp
    ```

    Upon successful completion, the above validation script should
    print out a report such as the following (latency records abbreviated):

    ```text
    REPLIES PASS: 100/100 packets received correctly
    CLOCK tick_period_numerator=1 tick_period_denominator=1000000000 tick_period_seconds=1e-09
    LATENCY request_id=1 ticks=6348 ns=6348
    ...
    LATENCY request_id=100 ticks=3589 ns=3589

    === PTP Round-Trip Latency ===
      Samples:  100
      Min:      3589 ns
      Max:      6348 ns
      Avg:      3872.0 ns
      Median:   3800 ns
    PASS: 100/100 replies; latency_samples=100/100; provider=libcudaq-playback-provider-hsb.so
    VALIDATION PASS execution=gpu dispatch=unified samples=100
    ```

    Latency statistics are in nanoseconds; the clock time base is in seconds per
    tick.
    Unavailable timing is reported separately and does not fail packet validation.
    Use `--num-messages`, `--message-size` (header plus payload bytes), and
    `--message-send-rate` (messages/second) to vary the workload. Defaults are
    100 requests, 32 bytes and 10,000 messages/second (100,000 ns spacing).
    The runner logs the requested rate and converted spacing. For HSB, the
    script rounds accepted intervals up to whole microseconds; intervals
    below 1 µs are rejected.
    Run `validate.sh --help` for all options; see the
    [validation architecture](validation_workflow.md#reading-the-validation-output)
    for output details.

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
bash /opt/nvidia/cudaq/realtime/validate.sh --dispatch=unified \
  --bridge-arg="--device=mlx5_0 --peer-ip=192.168.0.2"
```

### Manual Installation in Docker Container

1. Launch your container with networking and GPU support.

    For example, `--net host --gpus all` should be used to launch the container.

    > **_NOTE:_** Depending on the host system configurations, the `--privileged`
    flag may also be required so that the container can access the `IB` devices
    of the host.

2. Install CUDA runtime.

3. For GPU RoCE, install [`DOCA`](https://developer.nvidia.com/doca-downloads)
with `gpunetio` (`doca-sdk-gpunetio`) support.

4. Download and install CUDA-Q Installer as described in the [setup](#setup) section.

## Using a Custom Networking Implementation

CUDA-Q Realtime allows users to extend the networking layer with
their own hardware and software components,
in addition to one based on Holoscan Sensor Bridge.

Please refer to this [guide](cudaq_realtime_network_interface.md)
for more information about extending the networking interface
for your custom transport protocol. To use the shared validation applications,
see the [validation architecture and playback API](validation_workflow.md).

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

The software `hsb_fpga_emulator` uses its host's monotonic clock for both the
outgoing request timestamp and the response capture. It encodes both as
`{seconds[31:0], nanoseconds[31:0]}`, matching the playback reader's FPGA format.
Request timestamp insertion follows the player's `ptp_bram_ena` enable bit.
Software FPGA latency includes software transport and host scheduling overhead.
The emulator logs its capture clock when playback starts.

Please refer to `hsb_fpga_playback.cpp` code in the [CUDA-Q repository](https://github.com/NVIDIA/cuda-quantum)
for a sample of data generation tools.

## Troubleshooting

<!-- markdownlint-disable MD013 -->

### Error "`error while loading shared libraries: libcudaq-realtime-bridge-gpu-roce.so cannot open shared object file: No such file or directory`"

Check that the GPU RoCE bridge library is present in the installation's `lib/`
directory and that `bin/` and `lib/` remain together. The installed validation
applications locate these libraries automatically.

### Error "`error while loading shared libraries: libdoca_gpunetio.so.2: cannot open shared object file: No such file or directory`"

This can be resolved by installing the `doca-sdk-gpunetio` package.

### Error "`Cannot find GID for RoCE v2`"

Please make sure the `IB` device name (`--device`) is correct, i.e., the one connected to the `HSB` FPGA.

### Error "`Failed to get remote MAC.`"

Please make sure the IP address of the `HSB` FPGA (`--fpga-ip`) is correct. For example, we can do a quick `ping` test to check the connectivity.
