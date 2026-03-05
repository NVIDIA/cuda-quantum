# CUDA-Q Realtime Installation Guide

The following page describes the installation procedure of 
CUDA-Q Realtime, including connectivity to a 
[Holoscan Sensor Bridge](https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/) (`HSB`)
FPGA.


## Components

### Hardware Components

- A host system with NVIDIA GPU and ConnectX-7 NIC.

- A FPGA, programmed with `HSB` IP and connected to the NIC.

> **_NOTE:_** We recommended using NVIDIA ConnectX-7 as prior generations may not have all the required capabilities.


### Software Componets

- CUDA-Q Realtime installer.

- `HSB` source code from [GitHub](<FIXME: LINK TO HSB GitHub>)


## Setup

To install CUDA-Q Realtime with Holoscan Sensor Bridge on a host machine (bare-metal), please follow
these steps.

>  **_NOTE:_** Alternatively, we can also build and run these steps in a Docker container. 
Please refer to this [section](#using-docker) for instructions. 

1. Install CUDA-Q Realtime (if not already done so)

For example, 

```bash
./install_cuda_quantum_realtime_cu13.arm64  --accept
```

>  **_NOTE:_** Please verify that CUDA-Q Realtime has been installed to `/opt/nvidia/cudaq/realtime`.


2. Build Holoscan Sensor Bridge

- Clone the source code from `HSB` source code from [GitHub](<FIXME: LINK TO HSB GitHub>)

- Build `HSB`'s `gpu_roce_transceiver` and `hololink_core` components.

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
        --target gpu_roce_transceiver hololink_core
```

> **_NOTE:_**  In order to compile Holoscan Sensor Bridge from source, we need to install all of
its dependencies (e.g., NVIDIA `nvCOMP`, `DOCA`, `Holoscan SDK`, etc.) 
>
>Please refer to `HSB` [documentation](https://docs.nvidia.com/holoscan/sensor-bridge/latest/setup.html) for more details.

3. Load `HSB` IP bit-file to the FPGA

The bit-file for supported FPGA vendors can be found [here](FIXME:LINK_TO_BITFILE_LOCATION).

4. Run the validation scipt

The validation script is located at `/opt/nvidia/cudaq/realtime/script/validate_hololink.sh`.

```bash
bash hololink_test.sh --page-size 512 --device mlx5_0 --gpu 0 --bridge-ip 192.168.0.101 --fpga-ip 192.168.0.2 --unified --hololink-dir $HOLOLINK_DIR 
```

> **_NOTE:_** 
> The command line arguments need to be adjusted based on the system setup:
> - `--device` is the IB device name that is connected to the FPGA.
> - `--gpu` is the GPU device Id that we want to run the RPC callback on.
> - `--fpga-ip` is the IP address of the `HSB` FPGA.
> - `--bridge-ip` is the IP address of the NIC on the host machine. 
> - `--hololink-dir` is the location of `HSB`, which contains the `build` directory from the above step.

Upon successful completion, the above validation script should print out the following:

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

Congratulations! You have a complete 

> **_NOTE:_** In the above test script, we execute a simple RPC dispatch tests, whereby
the FPGA sends data (array of bytes) to the GPU; the GPU performs a simple increment by one 
calculation on each of the byte in the incoming array and returns the array. 
We then validate the data and measure the round-trip latency then ouput the report as shown above. 


## Using Docker

TODO