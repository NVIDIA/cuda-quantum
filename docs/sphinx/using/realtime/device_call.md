# CUDA-Q `device_call` Channels

`cudaq::device_call` lets a quantum kernel invoke a classical function on a
remote (or co-located) decoder/service while the program is running. The CUDA-Q
compiler lowers each call to the realtime RPC ABI; a **device-call channel**
carries the request to the service and the response back. This page describes
how channels are selected and documents the **`cpu_roce`** channel (a pure-CPU
RDMA transport). It complements the
[CUDA-Q Realtime Host API](host.md) and
[Messaging Protocol](protocol.md) documents, which describe the dispatcher and
the RPC wire format the channels carry.

## The `device_call` model

A kernel calls a classical function through `cudaq::device_call`:

```cpp
extern "C" int addThem(int a, int b);

__qpu__ int kernel(int a, int b) {
  // 0 is the device id; addThem names the remote function.
  return cudaq::device_call(0, addThem, a, b);
}
```

When compiled with realtime lowering (`nvq++ -frealtime-lowering --enable-mlir`),
the call is rewritten into the realtime
acquire/marshal/dispatch/release ABI. Two compile-time facts are baked in: the
**device id** (the first argument) and the **function id**
(`fnv1a_hash("addThem")`). The channel name is *not* compiled in.

## Selecting a channel

The channel is chosen at runtime, on the application command line, and forwarded
into the runtime by `cudaq::realtime::initialize(argc, argv)`:

```bash
./app --cudaq-device-call=<channel-name> [channel-specific args...]
```

The runtime parses `--cudaq-device-call`, instantiates the channel registered
under that name, and stores it in a session keyed by **device id**. At runtime,
`cudaq::device_call(0, ...)` resolves to the channel registered for device id 0.
The same compiled binary therefore runs over any channel by changing only the
launch flag.

Additional options:

- `--cudaq-device-call-slots=<N>`: ring-buffer slot count
- `--cudaq-device-call-slot-size=<bytes>`: per-slot size
- `--cudaq-device-call-timeout-ms=<ms>`: per-dispatch timeout

The built-in channels (`shared-memory`, `host-dispatch`, `gpu-dispatch`) run an
in-process service and are not covered in depth here. The remainder of this page
documents the `cpu_roce` channel.

## The `cpu_roce` channel

`cpu_roce` carries `device_call` RPCs over a **pure-CPU RoCEv2 RDMA** transport,
landing payloads directly in host memory at microsecond latency. It is an
*external* channel: unlike the in-process channels, its service runs in a
**separate process** (or a real FPGA) reached over the network. The transport is
`CpuRoceTransceiver`, described in [CPU RoCE Transport](cpu_transport.md).

### Wire pattern (FPGA-compatible)

The channel is the **caller** and plays the FPGA role on the wire; the service
plays the bridge/decoder role:

```text
caller (channel) --IBV_WR_RDMA_WRITE_WITH_IMM--> service   (request)
caller (channel) <--IBV_WR_SEND--                service   (response)
```

This mirrors a real FPGA decode loop: the FPGA RDMA-Writes its syndromes into
the decoder's ring and receives the corrections as Sends. A real HSB-enabled
FPGA could therefore replace the software channel unchanged.

### Connection setup

Connected (UC) queue pairs require each end to learn the other's QP number (and,
for the writer, the peer's rkey) before traffic flows. The channel performs a
minimal bidirectional rendezvous between the transceiver's `setup()` and
`connect()` phases. For a real FPGA service this exchange is driven by the HSB
control plane instead; only the rendezvous step differs, not the data plane. The
runtime pulls in no Hololink/HSB dependency.

### Running it

`cpu_roce` requires a ConnectX NIC (or two RoCE ports / a loopback) and a
service process, so it is **not run in CI**. Provide the channel arguments as
bare `key=value` tokens on the same command line:

```bash
./app --cudaq-device-call=cpu_roce \
  --cudaq-device-call-slots=64 --cudaq-device-call-slot-size=384 \
  ib-device=<channel-ib-device> local-ip=<channel-ip> \
  rendezvous-host=<service-ip> rendezvous-port=<port>
```

- `ib-device`: the channel's local IB device
- `local-ip`: the channel's RoCE IPv4
- `rendezvous-host` / `rendezvous-port`: where the service's rendezvous server
  is listening
- the slot count/size must match the service's ring geometry

### Test harness

The repository ships a test-only service daemon and an orchestration script:

- `cpu_roce_test_daemon` runs the `libcudaq-realtime` host dispatcher with
  `CUDAQ_DISPATCH_HOST_CALL` handlers (keyed by `fnv1a_hash`) and a TCP
  rendezvous server. It stands in for a real decoder/bridge.
- `cpu_roce_device_call_test.sh` configures the two RoCE ports and runs either:
  - the default GoogleTest fixture (`CpuRoceDispatchTest`), which drives the
    channel ABI directly, or
  - `--app` mode, which `nvq++`-compiles a real `__qpu__` `device_call`
    application, spawns the daemon, runs the app over `cpu_roce`, and verifies
    the compiler-lowered result.

```bash
# Compiler-driven proof over a two-port loopback:
cpu_roce_device_call_test.sh \
  --channel-device <ibdevA> --channel-ip 10.0.0.1 \
  --daemon-device  <ibdevB> --daemon-ip  10.0.0.2 \
  --setup-network --app
```
