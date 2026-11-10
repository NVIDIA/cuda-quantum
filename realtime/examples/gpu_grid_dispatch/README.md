# gpu_grid_dispatch

Serves the CI's `rpc_increment` RPC (24-byte `RPCHeader` + 8 payload bytes in,
24-byte `RPCResponse` + those bytes each incremented by one out) with a whole
resident cooperative GPU grid, one block on every SM, while the CPU owns the
NIC: CUDA-Q Realtime's single-thread host dispatch loop over the `cpu_roce`
bridge receives each request, its HOST_CALL handler wakes the grid through a
mailbox and waits for the grid's completion in pinned host memory.

- `--notify=auto|gdrcopy|pinned` picks how the CPU writes the mailbox, and
  `--poll=auto|all|leader` which GPU threads poll it; `auto` chooses per
  platform and prints its decision.
- `--request=inline|ring` picks where the bytes travel.  `inline` (default)
  copies the 8 request bytes through the mailbox and the 8 reply bytes back
  through the done line.  `ring` registers the bridge's RX/TX rings with CUDA
  at startup, and the whole grid reads the request in its RX slot and writes
  the reply into the paired TX slot, so payloads up to the slot size minus
  the 24-byte header work; the CPU writes only the response header.  Fast
  where the GPU reaches host memory coherently (GH200, integrated GPUs);
  correct but over PCIe on a discrete GPU.
- `gpu_grid_dispatch.cpp` is the host program, `grid_kernel.cu` the grid,
  `grid_common.h` the protocol between them.
- `gpu_grid_dispatch_test.sh` runs it end to end against the HSB FPGA or, with
  `--emulate`, the FPGA emulator.

Build against an installed CUDA-Q Realtime (gdrcopy is optional):

```bash
cmake -S . -B build -DCUDAQ_REALTIME_ROOT=<realtime install prefix>
cmake --build build
```

The full walkthrough is in
[realtime/docs/gpu_grid_dispatch_example.md](../../docs/gpu_grid_dispatch_example.md).
