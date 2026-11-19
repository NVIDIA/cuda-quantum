# Application-Provided GPU Grid Dispatcher

This example replaces CUDA-Q Realtime's built-in unified dispatch kernel with
an application-provided cooperative grid. The GPU receives requests directly
from the `gpu_roce` bridge, runs the RPC handler across one resident block per
SM, and sends the response without copying the request through the CPU.

The example is part of the existing RoCE validation application and test
harness rather than a separate executable:

- [`grid_dispatcher.cu`](../validation/grid_dispatcher.cu) is the
  application-provided kernel.
- [`roce_app.cpp`](../validation/roce_app.cpp) looks up the kernel and installs
  its launcher when given `--dispatcher=grid`.
- [`hsb_tools.cmake`](../validation/hsb_tools.cmake) embeds the kernel in
  `roce_app` with `cudaq_realtime_add_dispatch_kernel`.
- [`rpc_increment_handler.cu`](../validation/rpc_increment_handler.cu)
  contains the exported grid-stride RPC handler.
- [`roce_test.sh`](../validation/roce_test.sh) runs the same emulator or FPGA
  latency flow used by the built-in dispatcher.

For example, against an FPGA, from a build tree:

```bash
realtime/validation/roce_test.sh --device mlx5_4 --gpu 3 --page-size 512 \
  --bridge-ip 192.168.0.1 --fpga-ip 192.168.0.2 \
  --bin-dir realtime/build/validation --unified --dispatcher grid
```

Add `--blocks N` to use fewer blocks than the GPU has `SMs`. At shutdown,
`roce_app` reports how many distinct `SMs` the blocks ran on.

## Data Path

Exactly one fixed GPU thread, thread 0 of block 0, calls the transport hooks.
As soon as it claims a request, it publishes the request and response frames to
the other blocks, then reads the request header and selects the `DEVICE_CALL`
handler for its own block. Thread 0 of every other block waits for the frames,
reads the header and selects the same handler. Every thread in the cooperative
grid then calls the handler. Each block other than the leader's adds one to an
arrival counter when it finishes; the leader waits for that counter, writes the
response header and publishes the response through the GPU transport.

The frames are published as four 64-bit words, each carrying the request's turn
number next to half of a frame pointer. A block has the whole request once all
four words carry the turn it waits for, so it needs one round trip to the L2
cache rather than a flag read followed by a data read. The leader's own block
gets its work through shared memory and takes no part in the fan-in, so a grid
of one block runs the same path as the built-in kernel.

The request path has no grid-wide barrier. A cooperative grid synchronization
is used only during startup, after each block records its SM placement, copies
the function table and the leader initializes the shared control state.

The launcher reserves just over half of each SM's shared memory per block. This
limits occupancy to one block per SM and makes a grid containing one block per
SM fully resident. Each block keeps a copy of the function table there, so
selecting a handler reads no global memory; a table too large for it is
searched in global memory instead. `--threads` controls how many threads each
of those blocks contains; every launched thread participates in the handler.

## Constraints

- `--dispatcher=grid` requires unified dispatch and cannot be combined with
  forward mode.
- The selected GPU must support cooperative launch and fit the requested grid
  at the requested block size.
- Only the fixed leader may call `cudaq_dev_enter_kernel`,
  `cudaq_dev_rx_poll`, or `cudaq_dev_tx_publish`.
- A handler used with this dispatcher must partition work over the whole grid,
  not only over `threadIdx.x`.
- The grid handles one request at a time. All blocks complete the current
  handler before the leader publishes its response and claims the next request.
