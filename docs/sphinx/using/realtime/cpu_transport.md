# CPU RoCE Transport

`CpuRoceTransceiver` is a **pure-CPU RoCEv2 RDMA transport** -- no GPU, no DOCA,
no Hololink/HSB. It moves RPC messages between a quantum control system (FPGA)
and a CPU-based decoder entirely in host memory at microsecond latency. It is a
reference implementation of the `libibverbs` transport option described in the
[CUDA-Q Realtime Host API](host.md): it implements the same ring-buffer slot +
flag protocol, so it plugs into the realtime dispatcher the same way a GPU
transport does.

It ships as a **separate library** (`cudaq-realtime-cpu-transport`,
`realtime/lib/cpu_transport/`) and is *not* part of `libcudaq-realtime`. It
mirrors the `GpuRoceTransceiver` ring-buffer API so the same dispatcher wiring
works with either, and it pairs naturally with the `CUDAQ_DISPATCH_HOST_CALL`
dispatch mode for a fully GPU-free decode loop.

It serves two kinds of consumers:

- **Direct realtime dispatch**: wired to a `libcudaq-realtime` dispatcher (e.g.
  `hsb_bridge_cpu` with `CUDAQ_DISPATCH_HOST_CALL`), as a drop-in CPU
  replacement for the GPU HSB transport.
- **Compiler-lowered `device_call`**: it is the transport behind the
  [`cpu_roce` device-call channel](device_call.md), so a quantum kernel's
  `cudaq::device_call` -- lowered by `nvq++` to the realtime RPC ABI -- can be
  dispatched to a **CPU-based** service over RDMA, with no GPU on either the
  transport or the handler side.

**HSB interoperability.** Although this end runs no HSB software, the on-wire
RDMA framing is **HSB-compatible**: the peer can be a real FPGA containing the
**HSB IP**. The HSB IP receives RDMA Sends and transmits RDMA Writes, so
`CpuRoceTransceiver` interoperates with it directly by selecting the matching
[TX mode](#tx-modes) -- the same FPGA/bridge wire the GPU `GpuRoceTransceiver`
uses. The only HSB-specific piece is the out-of-band control plane used to
exchange QP/rkey at connection time (see below); the data plane is identical.

## C ABI

A C ABI shim (`roce_wrapper.h`) exposes the transceiver so callers need not
include the C++ header:

```c
cpu_roce_transceiver_t cpu_roce_create_transceiver(
    const char *device_name, int ib_port, unsigned tx_ibv_qp,
    size_t frame_size, size_t page_size, unsigned num_pages,
    const char *peer_ip, int forward, int rx_only, int tx_only, int unified,
    cpu_roce_tx_mode_t tx_mode, uint64_t peer_rx_base_addr,
    uint32_t peer_rx_rkey);
int  cpu_roce_setup(cpu_roce_transceiver_t);    // mint local QP/rkey (-> INIT)
int  cpu_roce_connect(cpu_roce_transceiver_t, unsigned peer_qp,
                      const char *peer_ip, uint32_t peer_rx_rkey); // -> RTR/RTS
void cpu_roce_blocking_monitor(cpu_roce_transceiver_t);  // run RX/TX loops
/* ring accessors: cpu_roce_get_{rx,tx}_ring_{data,flag}_addr, get_qp_number,
   get_rkey, get_page_size, get_num_pages, set_local_ip, ... */
```

The RX/TX rings live in host memory; the worker threads poll CPU-accessible
flags and move data with `libibverbs` UC (Unreliable Connected) queue pairs.

## Two-phase bring-up (`setup` / `connect`)

Connected (UC) QPs require each end to know the other's QP number before any
traffic flows, so bring-up is split:

1. `cpu_roce_setup()` opens the device, registers the host rings, creates the
   QP, and transitions it to INIT. After this, `cpu_roce_get_qp_number()` and
   `cpu_roce_get_rkey()` are valid to hand to the peer.
2. The two ends exchange `{qp_number, rkey, roce_ipv4}` out of band. Against a
   real FPGA this is the HSB control plane (`authenticate` / `configure_roce`);
   tests use a minimal socket rendezvous so nothing pulls in an HSB dependency.
3. `cpu_roce_connect()` adopts the peer parameters and transitions the QP to
   RTR/RTS.

Only the rendezvous step changes between deployments; the data-plane wire does
not.

## TX modes

`cpu_roce_tx_mode_t` names the RDMA verb this transceiver's TX path issues. The
name describes the verb the transmitter uses, not a peer role -- whoever sets
the mode is the source:

- `CPU_ROCE_TX_MODE_RDMA_SEND`: TX issues `IBV_WR_SEND` into the peer's
  pre-posted recv WQEs. Use when the peer consumes Sends -- e.g. an FPGA's HSB
  IP (which can only receive Sends), or a transceiver acting as the responder.
- `CPU_ROCE_TX_MODE_RDMA_WRITE_WITH_IMM`: TX issues `IBV_WR_RDMA_WRITE_WITH_IMM`
  into the peer's rx ring using the peer's rkey, with the slot index in the
  immediate. Use when the peer's memory is the write target -- e.g. an FPGA
  pushing syndromes, or a transceiver acting as the requester.

These compose into an asymmetric, FPGA-compatible pattern: a requester Writes
into the responder's ring and the responder Sends results back, mirroring how a
real FPGA RDMA-Writes syndromes and receives Sends. The
[`cpu_roce` device-call channel](device_call.md) is built on exactly this
pattern.

## Testing (`hsb_bridge_cpu`)

`hsb_bridge_cpu` (`realtime/unittests/cpu_transport/`) is a GPU-less bridge that
wires `CpuRoceTransceiver` to a `CUDAQ_DISPATCH_HOST_CALL` dispatcher -- the
CPU analogue of the GPU HSB bridge. `hsb_test_cpu.sh` orchestrates it against
either the HSB FPGA emulator (`--emulate`) or a **real FPGA containing the HSB
IP** (driven over the HSB control plane, exactly as the GPU path is):

```bash
# Emulated loopback (no FPGA):
hsb_test_cpu.sh --emulate --setup-network --build

# Against a real FPGA containing the HSB IP:
hsb_test_cpu.sh --build --fpga-ip 192.168.0.2
```

Like all RoCE tests, this requires a ConnectX NIC (or two RoCE ports / a
loopback) and is therefore **not run in CI**.
