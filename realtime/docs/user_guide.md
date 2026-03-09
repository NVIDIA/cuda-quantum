# CUDA-Q Realtime Installation Guide

The following page describes the installation procedure of
CUDA-Q Realtime, including connectivity to a
[Holoscan Sensor Bridge](https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/)
(`HSB`) FPGA.

## Components

### Hardware Components

- A host system with NVIDIA GPU and ConnectX-7 NIC.

- A FPGA, programmed with `HSB` IP and connected to the NIC.

> **_NOTE:_** We recommended using NVIDIA ConnectX-7 as prior generations
may not have all the required capabilities.

### Software Components

- CUDA-Q Realtime installer.

- [`DOCA` 3.3.0 installation](https://developer.nvidia.com/doca-downloads).

> **_NOTE:_** `DOCA` is required to run the end-to-end validation with FPGA
using the builtin `HSB` support of CUDA-Q realtime.

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
    can be found [here](FIXME:LINK_TO_BITFILE_LOCATION).

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
in the incoming array and returns the array.
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
