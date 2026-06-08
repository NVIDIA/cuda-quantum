Installation
^^^^^^^^^^^^^^^^^^^^^^^^

We provide pre-built installers for CUDA-Q Realtime that can be downloaded from our 
`GitHub Releases <https://github.com/NVIDIA/cuda-quantum/releases>`__, starting with the CUDA-Q release tag 0.14.0.
The binaries are available for `x86_64`, and `ARM64` Linux system with CUDA Versions 12.6+ (Driver 560.35.05+) and 13.x (Driver 580.65.06+). Instructions for building CUDA-Q Realtime from source can be found on our `GitHub repository <https://github.com/NVIDIA/cuda-quantum/tree/main/realtime/docs/>`__.

CUDA-Q Realtime has been tested with the following NVIDIA products:
- `NVIDIA IGX Thor <https://www.nvidia.com/en-au/edge-computing/products/igx/>`__
- `NVIDIA GB200 <https://www.nvidia.com/en-us/products/workstations/gb200-developer-kit/>`__
- `NVIDIA DGX Spark <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>`__

Prerequisites
---------------------

.. tab:: Using Holoscan Sensor Bridge

  - CUDA Runtime with version 12.6+ or 13.x
  - `DOCA 3.3.0 installation <https://developer.nvidia.com/doca-downloads>`__ with `gpunetio` support.

  .. note:: 

    Please make sure `doca-sdk-gpunetio` is installed along with `doca-all`.

.. tab:: Using Custom Networking Layer

  - CUDA Runtime with version 12.6+ or 13.x

.. _realtime-hsb-fpga-artifacts:

HSB FPGA IP core and RFSoC bit-file
-----------------------------------

The primary FPGA deliverable is the open-source HSB FPGA IP core, ``nv_hsb_ip``.
Integrate this RTL source into your FPGA design when you want to use HSB with your FPGA target.

The HSB 2.6.0-EA release also provides a fully packaged RFSoC example for the Real Digital RFSoC 4x2 evaluation board, using Vivado part ``xczu48dr-ffvg1517-2-e``.
The `HSB 2.6.0-EA artifact directory <https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/QEC/HSB-2.6.0-EA/>`__ contains the pre-built ``nvqlink_rfsoc_v2603.bit`` bit-file and the ``pynq_rfsoc_2603_EA_release.zip`` RFSoC PYNQ reference-design archive.
The matching ``nv_hsb_ip`` source directory is in the `Holoscan Sensor Bridge release-2.6.0-EA branch <https://github.com/nvidia-holoscan/holoscan-sensor-bridge/tree/release-2.6.0-EA/fpga/nv_hsb_ip>`__.

When building the RFSoC project from the PYNQ archive, place the ``nv_hsb_ip`` directory from that release branch at the same level as the archive's ``pynq`` directory.
Do not mix ``nv_hsb_ip`` from an older HSB release with the HSB 2.6.0-EA RFSoC files.
For another RFSoC part or board, update the Vivado part and constraints in ``pynq/rfsoc-pynq/build/build.tcl`` and rebuild the bit-file.

Setup
---------------------

.. tab:: Using Holoscan Sensor Bridge

  - Install CUDA-Q Realtime. For example,

    .. code-block:: console

        ./install_cuda_quantum_realtime_cu13.arm64  --accept

  - Follow the instructions given by the installer for post-installation steps to set environment variables.

  - Program the FPGA with HSB.
    See :ref:`realtime-hsb-fpga-artifacts` for the reusable ``nv_hsb_ip`` RTL source and the packaged RFSoC example bit-file.

  .. note:: 

    Please make sure to `set up the host system <https://docs.nvidia.com/holoscan/sensor-bridge/latest/setup.html>`__ and the `set up the HSB FPGA device IP address <https://docs.nvidia.com/holoscan/sensor-bridge/latest/architecture.html#datachannel-enumeration-and-ip-address-configuration>`__.

.. tab:: Using Custom Networking Layer
  
  CUDA-Q Realtime defines a Network Provider Interface to build a networking-agnostic application.
  This interface consists of a set of APIs to construct a real-time RPC dispatch solution in a networking-agnostic manner.
  These APIs are backed by a provider plugin (a shared library) implementing the specific transport protocol.

  A guide that explains how to integrate a new networking provider with CUDA-Q Realtime via the Network Provider Interface can be found
  `here <https://github.com/NVIDIA/cuda-quantum/tree/main/realtime/docs/cudaq_realtime_network_interface.md>`__.
  
Latency Measurement
---------------------

The CUDA-Q Realtime installer contains a validation script that includes a latency measurement.
After completing the installation and setup steps, you should find the `validate.sh` script in the CUDA-Q Realtime installation location
(usually in `/opt/nvidia/cudaq/realtime`).
The script executes a simple RPC dispatch tests, whereby
the FPGA sends data (array of bytes) to the GPU, the GPU performs
a simple increment by one calculation on each of the byte
in the incoming array (unless in the `--forward` mode) and returns the array to the FPGA.
The validation includes checking the data correctness and measuring the round-trip latency.

.. tab:: Using Holoscan Sensor Bridge
  
  The validation script has three defined kernel modes, one for a unified kernel (enabled with `--unified`), one for a forward kernel
  (enabled with `--forward`), one for host-dispatch (enabled with `--cpu`), and a 3-kernel setup (used by default).
  The forward kernel skips the RPC callback to measure the raw latency without any compute performed on the GPU.

  .. code-block:: console

      bash validate.sh --page-size 512 --device mlx5_0 --gpu 0 --bridge-ip 192.168.0.101 --fpga-ip 192.168.0.2 --unified 

  .. note:: 

    The command line arguments need to be adjusted based on the system setup:
    - `--device` is the `IB` device name that is connected to the FPGA.
    - `--gpu` is the GPU device Id that we want to run the RPC callback on.
    - `--fpga-ip` is the IP address of the `HSB` FPGA.
    - `--bridge-ip` is the IP address of the NIC on the host machine.
    - `--page-size` is the ring buffer slot size in bytes.
    - `--unified`, `--forward`, or `--cpu` are the mutually exclusive flags to enable the unified, forward, or host-dispatch mode.
  
  Upon successful completion, the above validation script should
  print out something similar to the following:
  
  .. code-block:: console

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

.. tab:: Using Custom Networking Layer

  To measure latency with a custom networking implementation, implement a stimulus (data generation) tool that sends data to CUDA-Q Realtime according to the custom networking protocol.
  
  For example, in the HSB-based implementation, we use the `ptp_timestamp` field in the `RPCHeader` / `RPCResponse` (see the message protocol documentation) to capture the timestamp for latency analysis. Specifically, the stimulus tool (FPGA) stores the 'send' timestamp in the `RPCHeader` (incoming message), which will be echoed by the GPU in the outgoing `RPCResponse` after processing it (e.g., with the RPC handler). Using the Integrated Logic Analyzer timestamp when the FPGA receives the response from the GPU, we can compute the round-trip latency.
  `This file <https://github.com/NVIDIA/cuda-quantum/tree/main/realtime/unittests/utils/hololink_fpga_playback.cpp>`__ contains an example of such a data generation tool.
