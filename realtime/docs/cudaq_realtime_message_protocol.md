# [CUDA-Q Realtime Messaging Protocol](#message-protocol)

This document defines the RPC (Remote Procedure Call) payload encoding used by
the realtime dispatch kernel for processing data and returning results. It complements
[cudaq_realtime_host_api.md](cudaq_realtime_host_api.md),
which focuses on wiring and API usage.

## [Scope](#scope)

- RPC header/response wire format
- Payload encoding and type system
- Schema contract and payload interpretation
- Function dispatch semantics

Note: This protocol is hardware-agnostic. While the companion document
[cudaq_realtime_host_api.md](cudaq_realtime_host_api.md) provides
implementation details for both GPU and CPU-based dispatchers,
the wire format and encoding rules specified here apply universally.

## [RPC Header / Response](#rpc-header)

Each ring-buffer slot is interpreted as:

```text
| RPCHeader | payload bytes (arg_len) | unused padding (slot_size - header - payload) |
```

```cpp
struct RPCHeader {
  uint32_t magic;        // RPC_MAGIC_REQUEST
  uint32_t function_id;  // fnv1a_hash("handler_name")
  uint32_t arg_len;      // payload bytes following this header
  uint32_t request_id;   // caller-assigned ID, echoed in the response
};

struct RPCResponse {
  uint32_t magic;        // RPC_MAGIC_RESPONSE
  int32_t  status;       // 0 = success
  uint32_t result_len;   // bytes of response payload
  uint32_t request_id;   // echoed from RPCHeader::request_id
};
```

Both `structs` are 16 bytes, packed with no padding.

Magic values (little-endian 32-bit):

- `RPC_MAGIC_REQUEST = 0x43555152` (`'CUQR'`)
- `RPC_MAGIC_RESPONSE = 0x43555153` (`'CUQS'`)

## [Request ID Semantics](#request-id)

`request_id` is a caller-assigned opaque 32-bit value included in every request.
The dispatch kernel copies it verbatim into the corresponding `RPCResponse`.
The protocol does not interpret or constrain the value; its meaning is defined
by the application.

Typical uses:

- **Shot index**: The sender sets `request_id` to the shot number, enabling
  out-of-order or pipelined verification of responses.
- **Sequence number**: Monotonically increasing counter for detecting lost or
  duplicated messages.
- **Unused**: Set to 0 when not needed. The dispatcher echoes it regardless.

The dispatcher echoes `request_id` in all dispatch paths (cooperative,
regular, and graph-launch).

## [Function ID Semantics](#function-id)

`function_id` selects which handler the dispatcher invokes for a given RPC
message. The dispatcher performs a lookup in the function table (array of
function pointers + IDs) and calls the matching entry.

See [cudaq_realtime_host_api.md](cudaq_realtime_host_api.md) for function ID hashing,
handler naming, and function table registration details.

## [Schema and Payload Interpretation](#schema-interpretation)

The RPC payload is **`typeless` on the wire**. The bytes following `RPCHeader`
are an opaque blob from the protocol's perspective.

**Payload interpretation is defined by the handler schema**, which is registered
in the dispatcher's function table during setup (see [cudaq_realtime_host_api.md](cudaq_realtime_host_api.md)).
The schema specifies:

- Number of arguments
- Type and size of each argument
- Number of return values
- Type and size of each return value

**Out-of-band contract**: The client (e.g., FPGA) firmware and dispatcher function
table must agree on the schema for each `function_id`. Schema mismatches are detected
during integration testing, not at runtime.

For handlers with multiple arguments, the payload is a **concatenation** of
argument data in schema order:

```text
| RPCHeader | arg0_bytes | arg1_bytes | arg2_bytes | ... |
```

The dispatcher uses the schema to determine where each argument begins and ends within
the payload.

### [Type System](#type-system)

Standardized payload type identifiers used in handler schemas:

```cpp
enum PayloadTypeID : uint8_t {
  TYPE_UINT8           = 0x10,
  TYPE_INT32           = 0x11,
  TYPE_INT64           = 0x12,
  TYPE_FLOAT32         = 0x13,
  TYPE_FLOAT64         = 0x14,
  TYPE_ARRAY_UINT8     = 0x20,
  TYPE_ARRAY_INT32     = 0x21,
  TYPE_ARRAY_FLOAT32   = 0x22,
  TYPE_ARRAY_FLOAT64   = 0x23,
  TYPE_BIT_PACKED      = 0x30   // Bit-packed data (LSB-first)
};
```

Schema type descriptor (see [cudaq_realtime_host_api.md](cudaq_realtime_host_api.md)
for full definition):

```cpp
struct cudaq_type_desc_t {
  uint8_t  type_id;       // PayloadTypeID value
  uint8_t  reserved[3];
  uint32_t size_bytes;    // Total size in bytes
  uint32_t num_elements;  // Interpretation depends on type_id
};
```

The `num_elements` field interpretation:

- **Scalar types** (`TYPE_UINT8`, `TYPE_INT32`, etc.): unused, set to 1
- **Array types** (`TYPE_ARRAY_*`): number of array elements
- **TYPE_BIT_PACKED**: number of bits (not bytes)

Note: For arbitrary binary data or vendor-specific formats, use `TYPE_ARRAY_UINT8`.

Encoding rules:

- All multi-byte integers: **little-endian**
- Floating-point: **IEEE 754** format
- Arrays: tightly packed elements (no padding)
- Bit-packed data: LSB-first within each byte,
`size_bytes = ceil(num_elements / 8)`

## [Payload Encoding](#payload-encoding)

The payload contains the argument data for the handler function. The encoding
depends on the argument types specified in the handler schema.

### Single-Argument Payloads

For handlers with one argument, the payload contains the argument data directly:

```text
| RPCHeader | argument_bytes |
```

### Multi-Argument Payloads

For handlers with multiple arguments, arguments are **concatenated in schema order**
with no padding or delimiters:

```text
| RPCHeader | arg0_bytes | arg1_bytes | arg2_bytes | ... |
```

The schema specifies the size of each argument,
allowing the dispatcher to compute offsets.

### Size Constraints

The total payload must fit in a single ring-buffer slot:

```text
total_size = sizeof(RPCHeader) + arg_len ≤ slot_size
max_payload_bytes = slot_size - sizeof(RPCHeader)
```

### Encoding Examples

**Example 1: Handler with signature** `void process(int32_t count, float threshold)`

Schema:

- `arg0`: `TYPE_INT32`, 4 bytes
- `arg1`: `TYPE_FLOAT32`, 4 bytes

Wire encoding:

```text
Offset | Content
-------|--------
0-15   | RPCHeader { magic, function_id, arg_len=8, request_id }
16-19  | count (int32_t, little-endian)
20-23  | threshold (float, IEEE 754)
```

**Example 2: Handler with signature**
`void decode(const uint8_t* bits, uint32_t num_bits)`

Schema:

- `arg0`: `TYPE_BIT_PACKED`, size_bytes=16, num_elements=128
- `arg1`: `TYPE_UINT32`, size_bytes=4, num_elements=1

Wire encoding:

```text
Offset | Content
-------|--------
0-15   | RPCHeader { magic, function_id, arg_len=20, request_id }
16-31  | bits (bit-packed, LSB-first, 128 bits)
32-35  | num_bits=128 (uint32_t, little-endian)
```

### Bit-Packed Data Encoding

For `TYPE_BIT_PACKED` arguments:

- Bits are packed **LSB-first** within each byte
- Payload length: `size_bytes = ceil(num_elements / 8)` bytes
- The schema specifies both `size_bytes` (storage)
and `num_elements` (actual bit count)

Example for 10 bits (size_bytes=2, num_elements=10):

```text
bits:    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9
byte[0]: b0 b1 b2 b3 b4 b5 b6 b7   (LSB-first)
byte[1]: b8 b9 0  0  0  0  0  0    (unused bits set to zero)
```

The handler can use `num_elements` from the schema to determine how many bits
are valid, avoiding the need to pass bit count as a separate argument (though
some handlers may still choose to do so for flexibility).

**Use case**: `TYPE_BIT_PACKED` is suitable for **binary measurements** where
each measurement result is 0 or 1 (1 bit per measurement).

### Multi-Bit Measurement Encoding

For applications requiring richer measurement data (e.g., soft readout, leakage
detection), use array types instead of `TYPE_BIT_PACKED`:

**4-bit soft readout** (confidence values 0-15):

Use `TYPE_ARRAY_UINT8` with custom packing (2 measurements per byte):

- Schema: `TYPE_ARRAY_UINT8`, `size_bytes = ceil(num_measurements / 2)`,
`num_elements = num_measurements`
- Encoding: Low nibble = measurement[0], high nibble = measurement[1], etc.

**8-bit soft readout** (confidence values 0-255):

Use `TYPE_ARRAY_UINT8` with one byte per measurement:

- Schema: `TYPE_ARRAY_UINT8`, `size_bytes = num_measurements`, `num_elements = num_measurements`
- Encoding: byte[i] = measurement[i]

**Floating-point confidence values**:

Use `TYPE_ARRAY_FLOAT32`:

- Schema: `TYPE_ARRAY_FLOAT32`, `size_bytes = num_measurements × 4`,
`num_elements = num_measurements`
- Encoding: IEEE 754 single-precision floats, tightly packed

**Leakage/erasure-resolving readout** (values beyond binary):

Use `TYPE_ARRAY_UINT8` or `TYPE_ARRAY_INT32` depending on
the range of measurement outcomes
(e.g., 0=ground, 1=excited, 2=leakage state).

## [Response Encoding](#response-encoding)

The response is written to the TX ring buffer slot (separate from the RX buffer
that contains the request):

```text
| RPCResponse | result_bytes |
```

Like the request payload, the response payload encoding is **defined by the
handler schema**. The schema's `results[]` array specifies the type and size
of each return value.

### Single-Result Response

For handlers returning one value, the result is written directly after the
response header.

**Example response** for a handler returning a single `uint8_t`:

Schema:

- `result0`: `TYPE_UINT8`, `size_bytes=1`, `num_elements=1`

Wire encoding:

```text
Offset | Content                                    | Value (hex)
-------|--------------------------------------------|--------------
0-3    | magic (RPC_MAGIC_RESPONSE)                 | 53 51 55 43
4-7    | status (0 = success)                       | 00 00 00 00
8-11   | result_len                                 | 01 00 00 00
12-15  | request_id (echoed from request)            | XX XX XX XX
16     | result value (uint8_t)                     | 03
17-... | unused padding                             | XX XX XX XX
```

### Multi-Result Response

For handlers returning multiple values, results are **concatenated in schema order**
(same pattern as multi-argument requests):

```text
| RPCResponse | result0_bytes | result1_bytes | ... |
```

**Example**: Handler returning correction (`uint8_t`) + confidence (`float`)

Schema:

- `result0`: `TYPE_UINT8`, `size_bytes=1`, `num_elements=1`
- `result1`: `TYPE_FLOAT32`, `size_bytes=4`, `num_elements=1`

Wire encoding:

```text
Offset | Content
-------|--------
0-15   | RPCResponse { magic, status=0, result_len=5, request_id }
16     | correction (uint8_t)
17-20  | confidence (float32, IEEE 754)
```

### Status Codes

- `status = 0`: Success
- `status > 0`: Handler-specific error
- `status < 0`: Protocol-level error

## [QEC-Specific Usage Example](#qec-example)

This section shows how the realtime messaging protocol is used for quantum
error correction (QEC) decoding. This is one application of the protocol;
other use cases follow the same pattern.

### QEC Terminology

In QEC applications, the following terminology applies:

- **Measurement result**:
Raw readout value from a QPU measurement (0 or 1 for binary readout)
- **Detection event**:
`XOR`'d measurement results as dictated by the parity check (stabilizer) matrix
- **Syndrome**:
The full history or set of detection events used by the decoder

The decoder consumes detection events (often called "syndrome data" colloquially)
and produces corrections.

### QEC Decoder Handler

Typical QEC decoder signature:

```cpp
void qec_decode(const uint8_t* detection_events, uint32_t num_events, 
                uint8_t* correction);
```

Schema:

- `arg0`: `TYPE_BIT_PACKED`, variable size (detection events, 1 bit per event)
- `arg1`: `TYPE_UINT32`, 4 bytes (number of detection events)
- `result0`: `TYPE_UINT8`, 1 byte (correction bit-packed)

### Decoding Rounds

For QEC applications, one RPC message typically corresponds to one **decoding round**
(one invocation of the decoder with a set of detection events). The boundaries of
each decoding round are determined by the quantum control system (e.g., FPGA) when
building RPC messages.

Note: The term "shot" is often used in quantum computing to mean one full execution
of a quantum program (repeated `num_shots` times for statistics). In the context
of realtime decoding, we use "decoding round" to avoid confusion, as there may be
many RPC invocations during a single quantum program execution.
