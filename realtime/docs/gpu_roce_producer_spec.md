# GPU RoCE validation transport profile

This reference defines the transport needed by a producer using the GPU RoCE
bridge with `validation_app` or `roce_app --gpu`. The
[validation architecture](validation_workflow.md) describes the shared playback
API and verification. The [message protocol](cudaq_realtime_message_protocol.md)
defines the 24-byte little-endian RPC header and payload encoding.

| Layer | Producer requirement |
| --- | --- |
| Session | Exchange peer IP, queue pair numbers, receiver memory key/address, ring geometry and frame size. |
| Requests | `RoCEv2` over `IPv4`, Unreliable Connection (UC), RDMA Write with Immediate into a receiver ring slot. |
| Replies | Receive RoCE SEND messages on the producer's UC queue pair; a software producer posts receive buffers. |
| Workload | `rpc_increment` (`function_id=0x191b6322`) increments each payload byte modulo 256. Native forwarding echoes the request unchanged. |

HSB supplies its playback controls through FPGA registers, `BRAM` and ILA.
Another producer can use different controls while implementing this transport.
Sending the same RPC bytes over ordinary `UDP`, TCP or another RDMA operation
does not establish compatibility.

## Session setup

Check the network configuration before starting playback:

| Setting | Requirement |
| --- | --- |
| Interface and addressing | Identify the selected RDMA device, its Ethernet interface, and the `RTH` data-plane IP and MAC. The published endpoint contains QP/memory/ring settings; `peer_ip` is the producer's address, not the `RTH` address. HSB playback runs on the `RTH` host and obtains its IP/MAC through SDK discovery or routing on that same interface. |
| `GID` selection | The chosen port must have a `RoCEv2` `IPv4`-mapped global identifier (`GID`). The GPU bridge automatically selects a matching local `GID` table entry; verify that its index corresponds to the intended interface/address, especially with multiple addresses or `VLANs`. The endpoint does not publish this index. |
| Packet size | The QP path `MTU` is 4,096 bytes. Configure the producer consistently and ensure both network interfaces and the intervening network can carry the resulting packets. This is the RDMA path `MTU`, not an Ethernet `MTU` setting; allow room for network headers. |
| Topology | A direct link or a switched path can carry the traffic. With switches, verify forwarding, `VLAN` configuration and frame-size limits along the whole path. Request loss or reply loss fails validation; UC does not retry it. |

1. Create the producer's UC queue pair (QP), selecting the `RoCEv2` `GID` for its
   `IPv4` interface. Configure the receiver with the producer IP and QP number.
   The runner defaults to `192.168.0.2` and QP 2; override these with
   `--bridge-arg="--device=mlx5_0 --peer-ip=IP --remote-qp=N"` as needed.
   The producer QP must be known before the receiver starts, but need not be a
   permanently fixed number.
2. Obtain the receiver's QP, `RKey`, RDMA base address and ring geometry from its
   published endpoint. These values belong to this session; do not reuse values
   from a previous run. Embedded callers can use `cudaq_bridge_get_endpoint_info()`
   and `cudaq_bridge_get_ring_geometry()`.
3. Connect the producer QP to the receiver QP and `GID`. The GPU bridge uses
   send/receive packet sequence numbers (`PSNs`) starting at zero and IB port 1.
   Match these parameters and the network settings above.
4. Post receive buffers for replies and prepare the producer for transmission.
   Wait for `=== App Ready ===`, which follows dispatcher startup. The earlier
   `=== Bridge Ready ===` is insufficient. The shared runner coordinates this.

This preset exchanges connection values out of band; it does not negotiate the
producer QP through RDMA CM or a producer handshake. Use the
[validation runner](user_guide.md#setup) for built-in presets; a custom
playback provider configures its device using the published endpoint.

## Request addressing and slot ownership

Let `S` be the advertised slot stride, `N` the slot count, `B` the RDMA base
address and `K` the `RKey`. For request sequence number `n`, beginning at zero:

```text
slot           = n modulo N
remote address = B + slot * S
remote key     = K
immediate data = slot
operation      = RDMA Write with Immediate
```

With `libibverbs`, use `IBV_WR_RDMA_WRITE_WITH_IMM` and
`wr.imm_data = htonl(slot)`. Immediate data is a network-order 32-bit transport
value; RPC fields are little-endian. The receiver uses the low 12 bits for the
slot. Set upper bits to zero, keep `0 <= slot < N <= 4096`, and ensure the address
and immediate slot agree. Plain RDMA Write provides no completion for this path;
incoming SEND does not write the required ring slot.

The GPU bridge registers its receive ring at I/O virtual address zero, so an
advertised base of `0x0` is valid. This is an RDMA address, not a CPU or CUDA pointer.
Always use the published address and key.

Use a power-of-two slot count and a stride aligned to 128 bytes. GPU RoCE
validation defaults to `S=384`, `N=128`: 49,152 bytes (48 `KiB`). Start at slot
zero and advance sequentially for the separate RX/dispatch/TX path.
The RPC request ID is independent of the slot index.
HSB restarts at slot zero for every batch. Use a fresh receiver unless the
preceding batch filled whole rings, leaving the receiver expecting slot zero.

Do not overwrite a slot until its response arrives and the previous local send
buffer is safe to reuse. Begin with one outstanding request. Pipelining needs
bounded outstanding requests, sufficient posted receives and suitable pacing;
ring capacity is not a receive-credit count. The protocol advertises no credits.

An FPGA generating raw packets must supply valid RoCE headers, destination QP,
`PSN` progression, remote-memory fields, immediate data and invariant `CRC`.
Ordinary `UDP` payloads sent to RoCE port 4791 do not implement RDMA. A verbs NIC
supplies this encapsulation for software producers.

## Frame size and replies

The configured payload capacity `P` sets a fixed reply length `F = 24 + P` bytes,
excluding networking headers. Choose `S >= F`. For validation, set `arg_len=P`
and send a fully initialized frame. The default is `P=8`, `F=32`, `S=384`.
HSB pads requests to 64-byte playback windows; this padding is separate from both
the canonical RPC length and the receiver's slot stride.

If using shorter arguments, initialize the whole frame and keep `arg_len <= P`.
The reply still occupies `F` bytes; only the declared result bytes are meaningful.
Neither slot capacity nor a handler clamp substitutes for checking received
lengths before accessing a payload.

Replies are RoCE SEND messages to the peer QP configured at startup, not RDMA
writes to an address in the request. Replenish posted receive buffers as
completions arrive. Check completion status and byte count, then RPC magic,
status, ID, result length and every result byte. Require `24 + result_len` to fit
within the received frame. Match replies by ID, not arrival order or slot number.

Separate and unified dispatch return an `RPCResponse` with success status and
incremented bytes. Native forwarding returns the request header and payload
unchanged: offset 4 remains `function_id`, not status. The shared playback app
checks the workload header and payload and rejects missing, duplicate or corrupt
replies; trailing transport padding is allowed.

The `ptp_timestamp` field at bytes 16–23 is echoed by the receiver and excluded
from the app's byte comparison; the provider interprets and checks timing.
Other producers can measure round trips with their own clock.
HSB encodes `(seconds << 32) | nanoseconds`, with
nanoseconds below 1,000,000,000; this is not a plain 64-bit nanosecond count.
The playback provider decodes its own timing and returns ticks through the
[common results API](validation_workflow.md#c-api-and-parameter-contract).

## Loss and recovery

UC provides no reliable delivery or transport retries. Local send completion
does not prove that the receiver processed the request. Apply a response deadline;
the RPC protocol provides no retries, duplicate suppression or exactly-once
execution. After a timeout, stop traffic and establish a fresh session before
reusing potentially active slots. A missing reply fails validation, but a timeout
alone cannot locate the loss in the producer, network, receiver or capture path.

Malformed requests have different behavior across dispatch paths: unified
dispatch returns `status=-1` for bad magic or an unknown function ID, while the
separate dispatcher drops them. Incomplete headers and invalid declared lengths
have no consistent error-response contract. Forwarding performs no RPC checks.

For a new producer, verify one request first, then ring wrap, byte overflow
(`255 -> 0`), distinct IDs, bounded pipelining and timeout recovery. Validate each
supported dispatch mode and hardware configuration.

## Implementation references

- [GPU bridge setup and endpoint](../lib/daemon/bridge/gpu_roce/bridge_impl.cpp)
- [Unified slot decoding and response sends](../lib/daemon/bridge/gpu_roce/gpu_roce_unified_transport.cu)
- [Software UC producer](../validation/hsb_fpga_emulator.cpp)
- [Bridge API and execution paths](cudaq_realtime_network_interface.md)

QP setup, memory registration and forwarding also depend on the HSB library
selected by [deps_common.sh](../scripts/deps_common.sh) and the
[HSB patches](../patches/hsb/README.md).
