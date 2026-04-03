CUDA-Q Realtime
++++++++++++++++++++++++

CUDA-Q Realtime is a library for tightly coupling GPU-accelerated compute to the control system of a quantum processor via a networking layer.
It requires a host system with NVIDIA GPU and ConnectX-7/BlueField NIC, and an FPGA connected to the NIC.

It fulfills two primary responsibilities:

1. It provides the low-level basis of realtime co-processing between FPGA and CPU-GPU systems.
2. It provides the low latency networking layer of the NVQLink architecture, enabling system integrators to achieve few-microsecond data round trips between FPGA and GPU.

The provided networking layer leverages the `Holoscan Sensor Bridge <https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/>`_ to handle high-bandwidth data transfer over Ethernet using the RoCE protocol. This layer can optionally be replaced with a custom implementation.

.. toctree::
  :caption: Contents
  :maxdepth: 1

  Installation <realtime/installation.rst>
  Host API <realtime/host.md>
  Messaging Protocol <realtime/protocol.md>

