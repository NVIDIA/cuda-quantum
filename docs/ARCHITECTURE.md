# NVQLink Architecture (WIP)

## 1. Introduction and Goals

NVQLink is a platform architecture for tightly coupling high-performance
computing (HPC) resources to quantum processing unit (QPU) control systems. It
addresses the critical need for real-time classical coprocessing in
fault-tolerant quantum computing, with quantum error correction (QEC) decoding
as the primary driving workload.

### 1.1 Requirements Overview

#### Functional Requirements

- Provide ultra-low latency communication between QCS (Quantum Control Systems) and RTH (Real-time Hosts)
- Enable classical coprocessing during quantum algorithm execution
- Integrate with existing HPC infrastructure via standard networking
- Support quantum error correction decoding workflows

#### Driving Forces

- Fault-tolerant quantum computing requires real-time classical feedback
- QEC decoding must complete within strict timing budgets to prevent logical errors
- QPU control systems have limited compute resources and must offload heavy processing
- Algorithm researchers need development environments without hardware access

### 1.2 Quality Goals

The following quality goals drive fundamental architectural decisions, ordered
by priority:

| Priority | Quality Goal | Scenario |
|----------|-------------|----------|
| 1 | **Ultra-Low Latency** | A QEC decoder receives syndrome data and must return corrections before the next feedforward event. The round-trip latency (QCS → RTH → QCS) must be < 10 µs to prevent logical errors from accumulating during the wait. |
| 2 | **Extensibility** | A QCS vendor integrates their proprietary PPU firmware without exposing instruction set details. They implement a custom `Channel` and provide a VPPU emulator; no changes to NVQLink core are required. |
| 3 | **Scalability** | A 1000-logical-qubit algorithm runs with lattice surgery. The decoder uses parallel window processing across multiple GPUs, each handling independent spatial/temporal blocks.|
| 4 | **Zero Overhead** | During peak syndrome streaming at 1 MHz, the system processes 1M packets/second. No memory allocation, exception handling, or virtual dispatch occurs in the packet processing path. |
| 5 | **Productivity** | A researcher develops a new AI-based decoder on their laptop using VPPU simulation. Once validated, the same code deploys to physical hardware by changing a config flag—no code modifications needed. |

### 1.3 Stakeholders

| Role | Representative | Expectations |
|------|----------------|--------------|
| **QCS Builder** | Quantum Machines, Qblox, Zurich Instruments, Keysight | Integration requires minimal changes to existing firmware; proprietary PPU instruction sets remain private; clear Network Interface specification |
| **QPU Operator** | National labs, quantum computing centers | System runs reliably under fault-tolerant workloads; latency metrics are measurable and documented; calibration workflows are well-defined |
| **Algorithm Researcher** | University research groups, NVIDIA | Can develop and test QEC protocols without hardware access; documentation explains programming model and device callback semantics |
| **HPC Integrator** | Supercomputing centers (ORNL, LBNL) | Uses standard networking (Ethernet/RDMA); fits into existing HPC infrastructure |
| **Decoder Developer** | QEC research teams | Understands how to register decoder functions; knows GPU execution model; documentation covers real-time constraints and data flow |

## 2. Architecture Constraints

### 2.1 Technical Constraints

| Constraint | Description |
|------------|-------------|
| **RDMA Networking** | Data plane requires RDMA-capable NICs (RoCE v2) for ultra-low latency |
| **Linux OS** | System relies on `libibverbs` API, the Linux kernel RDMA stack, and NVIDIA DOCA GPUNetIO for GPU-direct NIC control |
| **C++17 Minimum** | Codebase uses modern C++ features for template metaprogramming |
| **CUDA Optional** | GPU acceleration requires NVIDIA GPUs with CUDA 12+ |

### 2.2 Organizational Constraints

| Constraint | Description |
|------------|-------------|
| **CUDA-Q Integration** | Must integrate with NVIDIA CUDA-Q quantum compiler ecosystem |
| **Vendor Independence** | Architecture must support multiple QCS vendors without core changes |
| **Standard + NVIDIA APIs** | Data plane uses standard `libibverbs` API; also uses NVIDIA DOCA GPUNetIO for GPU-direct NIC control |

### 2.3 Conventions

| Convention | Description |
|------------|-------------|
| **RAII Resource Management** | All hardware resources use RAII patterns |
| **No Hot Path Allocation** | Memory allocation forbidden in packet processing paths |
| **Header-Only GPU Code** | GPU stream classes are header-only for device compilation |

## 3. System Scope and Context

### 3.1 Business Context

NVQLink connects quantum control systems (which execute quantum programs) to
high-performance computing resources (which provide real-time classical
processing such as QEC decoding).

**External Systems:**

| System | Description | Interface |
|--------|-------------|-----------|
| **Quantum Control System (QCS)** | FPGA/pulse processor executing quantum applications (e.g., Quantum Machines OPX1000, QubiC) | RDMA data plane, UDP control plane |
| **QPU** | Physical quantum processor controlled by QCS | Not directly interfaced by NVQLink |
| **HPC Cluster** | Provides GPU compute resources for decoding | Internal resource allocation |

### 3.2 Technical Context

```text
┌──────────────────────────────────────────────────┐
│            Quantum Control System (QCS)          │
│               FPGA / Pulse Processor             │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │     Quantum Application                    │  │
│  │     (CUDA-Q Quantum Kernels)               │  │
│  │                                            │  │
│  │  • Quantum gate execution                  │  │
│  │  • Issues device_call() for RTH callbacks  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
        │                              │
        │  ▼ RPC requests              │  ▲ Connection params
        │  ▲ RPC responses             │  ▲ START/STOP cmds
        │                              │  ▼ ACK/status
  ══════╪══════════════════════════════╪══════════════════
        │    Data Plane                │  Control Plane
        │    (RDMA WRITE/SEND)         │  (UDP packets)
  ══════╪══════════════════════════════╪══════════════════
        │                              │
┌───────┴──────────────────────────────┴───────────┐
│               Real-time Host (RTH)               │
│                NVQLink Framework                 │
│                                                  │
│  • Daemon: handles device_call() RPC requests    │
│  • Channel: RDMA network transport               │
│  • QCSDevice: UDP control plane management       │
│  • Callbacks: QEC decoding, classical processing │
└──────────────────────────────────────────────────┘

═══════════════════════════════════════════════════
         Network Infrastructure (Medium)

  • RoCE/RDMA NICs (Mellanox ConnectX)
  • RDMA-capable Ethernet switches
  • Quantum Machines Opnic
═══════════════════════════════════════════════════

Legend:
  │       Vertical line (connection)
  ▼       Downward flow (QCS → RTH)
  ▲       Upward flow (RTH → QCS)
  ═════   Network medium (physical infrastructure)
```

**Systems:**

- **QCS (Quantum Control System)**: FPGA/pulse processor executing quantum applications
  - Runs CUDA-Q quantum kernels
  - Issues `device_call()` to invoke RTH processing
  - Examples: Quantum Machines OPX1000, QubiC, custom FPGA implementations

- **RTH (Real-time Host)**: HPC node running NVQLink Framework
  - Implements quantum callback functions (QEC decoding, classical processing)
  - Provides ultra-low latency RPC services
  - Examples: CPU/GPU servers with RDMA NICs

**Communication:**

- **Data Plane** (left arrows): High-performance RPC traffic
  - Protocol: RDMA WRITE (requests) and RDMA SEND (responses)
  - ▼ QCS → RTH: `device_call()` RPC requests
  - ▲ RTH → QCS: RPC responses (results)

- **Control Plane** (right arrows): Out-of-band configuration via UDP
  - ▲ RTH → QCS: Connection parameters (QPN, GID, vaddr, rkey)
  - ▲ RTH → QCS: START/STOP/ABORT commands
  - ▼ QCS → RTH: ACK, status, COMPLETE notifications

**System Boundary:** NVQLink runs on the RTH and provides the communication
layer for QCS to invoke callbacks on RTH compute resources.

## 4. Solution Strategy

The following design principles guide all architectural decisions:

### 4.1 Component-Based Architecture

The system uses composition over inheritance with clear component boundaries:

- **Channel**: Self-sufficient network I/O, owns all hardware resources
- **VerbsContext**: Optional shared InfiniBand context for memory sharing
- **FlowSwitch**: Optional traffic steering coordinator
- **Daemon**: RPC layer that uses Channel for network I/O
- **QCSDevice**: Control plane that configures `Channel` but doesn't own it

Each component can function independently. Sharing is explicit and opt-in.

### 4.2 Zero-Copy Data Flow

```text
NIC → DMA → Buffer → Stream (pointer) → User Code → Stream (pointer) → Buffer → DMA → NIC
       ▲                                                                         ▲
       └──────────────────────── Same memory region ─────────────────────────────┘
```

Techniques:

- Pre-allocated buffer pools (no allocation in hot path)
- Direct DMA buffer access via streams
- Pointer arithmetic instead of memcpy
- Buffer `reset()` for backend-specific memory wrapping

### 4.3 Unified Stream Abstraction

Single stream API works across different contexts:

- **Channel mode**: Persistent streams managing packet lifecycle
- **Buffer mode**: Temporary streams for RPC handlers
- **CPU/GPU variants**: Same conceptual API, different implementations

Non-polymorphic design enables compile-time optimization and zero overhead.

### 4.4 Separation of Concerns

**Control Plane (UDP)**: Configuration, connection setup, execution control

- Uses `QCSDevice` and `ControlServer`
- Out-of-band communication
- Simple UDP protocol

**Data Plane (RoCE)**: High-performance RPC traffic

- Uses `Daemon` and `Channel`
- RDMA for low latency
- Zero-copy packet processing

### 4.5 Type-Safe Serialization

Automatic marshalling via `NVQLINK_RPC_HANDLE` macro:

- Eliminates manual serialization boilerplate
- Compile-time type checking
- No runtime overhead

```cpp
// User writes this
int add(int a, int b) { return a + b; }
daemon.register_function(NVQLINK_RPC_HANDLE(add));

// System automatically generates:
// - Argument deserialization
// - Return value serialization
// - Function dispatch wrapper
```

### 4.6 Explicit Resource Management

- No hidden singletons or global state
- RAII-based lifecycle management
- Clear ownership: `Channel` owns buffers, `Daemon` owns `Channel`
- Optional resource sharing is explicit (`VerbsContext`, `VerbsFlowSwitch`)

## 5. Building Block View

The architecture consists of three distinct layers:

1. **Transport Layer (Channel)**: Hardware-abstracted network I/O with zero-copy buffer management
2. **Service Layer (Daemon)**: RPC protocol with automatic function dispatch and type-safe serialization  
3. **Control Layer (QCS)**: Out-of-band configuration and execution control via UDP

The system supports multiple usage patterns:

- **RPC Mode**: Full-featured server with automatic function registration (via Daemon)
- **Direct I/O Mode**: Low-level packet access for custom protocols (via Channel directly)
- **GPU Mode**: CUDA kernels with direct NIC queue access (via GPUChannel)

All modes provide unified stream abstractions (`InputStream`/`OutputStream`) for
type-safe, zero-copy data handling.

### 5.1 Level 1: Container Diagram (C4)

```text
┌──────────────────────────────────────────────────────────────────┐
│                    NVQLink Framework (RTH)                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Control Plane (out-of-band setup via UDP)               │    │
│  │                                                          │    │
│  │  ┌────────────────────┐    ┌─────────────────────────┐   │    │
│  │  │  QCSDevice         │    │  ControlServer          │   │    │
│  │  │  • trigger()       │───▶│  • UDP socket           │───┼────┼──▶ to QCS
│  │  │  • abort()         │uses│  • send_command()       │   │    │
│  │  │  • upload_program()│    │  • wait_for_response()  │◀──┼────┼─── from QCS
│  │  └────────────────────┘    └────────────┬────────────┘   │    │
│  │                                         │                │    │
│  │    exchange_connection_params(channel): │                │    │
│  │    ┌────────────────────────────────────┼────────────┐   │    │
│  │    │ 1. READ channel params             │            │   │    │
│  │    │ 2. SEND params ────────────────────┼────────────┼───┼────┼──▶ to QCS
│  │    │ 3. RECV QCS params ◀───────────────┼────────────┼───┼────┼─── from QCS
│  │    │ 4. WRITE remote QP to channel      │            │   │    │
│  │    └────────────────────────────────────┼────────────┘   │    │
│  └─────────────────────────────────────────┼────────────────┘    │
│                                            │                     │
│  ┌─────────────────────────────────────────┼────────────────┐    │
│  │  Data Plane (high-performance RPC)      │                │    │
│  │                     read/write          │                │    │
│  │  ┌──────────────────────────────────────▼────────────┐   │    │
│  │  │  Channel (Transport)                              │   │    │
│  │  │  ┌─────────────────────────────────────────────┐  │   │    │
│  │  │  │  RoCEChannel                                │  │   │    │
│  │  │  │  • get_connection_params() → local params   │  │   │    │
│  │  │  │  • set_remote_qp(qpn, gid) ← remote params  │  │   │    │
│  │  │  │  • receive_burst() / send_burst()           │  │   │    │
│  │  │  └─────────────────────────────────────────────┘  │   │    │
│  │  └───────────────────────────┬───────────────────────┘   │    │
│  │                              │ owned by                  │    │
│  │  ┌───────────────────────────▼───────────────────────┐   │    │
│  │  │  Daemon (Service)                                 │   │    │
│  │  │  ┌──────────────┐  ┌──────────────┐               │   │    │
│  │  │  │ Dispatcher   │  │  Function    │               │   │    │
│  │  │  │ (CPU/GPU)    │─▶│  Registry    │               │   │    │
│  │  │  └──────────────┘  └──────────────┘               │   │    │
│  │  └───────────────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                 │                                │
│                                 │ libibverbs API                 │
│                                 ▼                                │
│                    ┌───────────────────────────┐                 │
│                    │   Network Hardware (NIC)  │─────────────────┼──▶ RDMA to QCS
│                    └───────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────┘
```

**Container Relationships:**

1. **Control Plane** (`QCSDevice` + `ControlServer`):
   - **READS** `Channel`'s local RDMA params (QPN, GID, vaddr, rkey)
   - **SENDS** those params **TO external QCS** via UDP
   - **RECEIVES** QCS's params back from QCS
   - **WRITES** remote QP info **INTO Channel** (`set_remote_qp`)
   - Has NO relationship with `Daemon`

2. **Data Plane** (`Daemon` + `Channel`):
   - **Daemon owns Channel** for data plane operations
   - Channel handles network I/O (RDMA WRITE/SEND)
   - Daemon dispatches RPC requests to registered functions

**Key Flow**: `Channel` is created first (with its own RDMA resources). Control
plane extracts `Channel`'s params and sends them to QCS. QCS sends back its
params, which are then written into the `Channel`. Only after this exchange does
`Daemon` take ownership for data plane operations.

> **Note**: With Hololink, the Control Plane, more specifically the
> `ControlServer`, will be slightly different. One big difference is that the
> QCS that uses Hololink won't send back QP information back to the RTH.

### 5.2 Level 2: Component Diagram (C4)

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Daemon (Service Layer)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Daemon Core                            │  │
│  │  • start() / stop()                                       │  │
│  │  • register_function()                                    │  │
│  │  • get_stats()                                            │  │
│  └──────┬─────────────────────────┬──────────────────────────┘  │
│         │ owns                    │ uses                        │
│  ┌──────▼──────────────┐   ┌──────▼───────────────────────────┐ │
│  │    Dispatcher       │   │     FunctionRegistry             │ │
│  │  ┌──────────────┐   │   │  ┌─────────────────────────┐     │ │
│  │  │CPUDispatcher │   │   │  │ • function_id → wrapper │     │ │
│  │  │              │   │   │  │ • Hash-based lookup     │     │ │
│  │  │ while(run) { │   │   │  │ • Type inference        │     │ │
│  │  │   rx_burst() │   │   │  │ • Auto-marshal          │     │ │
│  │  │   dispatch() │   │   │  └─────────────────────────┘     │ │
│  │  │   tx_burst() │   │   │  ┌─────────────────────────┐     │ │
│  │  │ }            │   │   │  │  FunctionWrapper        │     │ │
│  │  └──────────────┘   │   │  │  • Deserialize args     │     │ │
│  │  ┌──────────────┐   │   │  │  • Call user function   │     │ │
│  │  │GPUDispatcher │   │   │  │  • Serialize result     │     │ │
│  │  │              │   │   │  └─────────────────────────┘     │ │
│  │  │ persistent   │   │   └──────────────────────────────────┘ │
│  │  │ kernel       │   │                                        │
│  │  │ polls NIC    │   │   ┌──────────────────────────────────┐ │
│  │  └──────────────┘   │   │     Serialization                │ │
│  └─────────┬───────────┘   │  ┌─────────────────────────────┐ │ │
│            │ uses          │  │ InputStream / OutputStream  │ │ │
│            │               │  │                             │ │ │
│            │               │  │ • read<T>() / write<T>()    │ │ │
│            │               │  │ • Dual-mode construction    │ │ │
│            │               │  │   - Channel (persistent)    │ │ │
│            │               │  │   - Buffer (RPC handler)    │ │ │
│            │               │  │ • Zero-copy via pointers    │ │ │
│            │               │  └─────────────────────────────┘ │ │
│            │               │  ┌─────────────────────────────┐ │ │
│            │               │  │ GPU Streams (device-side)   │ │ │
│            │               │  │                             │ │ │
│            │               │  │ • GPUInputStream            │ │ │
│            │               │  │ • GPUOutputStream           │ │ │
│            │               │  │ • Header-only               │ │ │
│            │               │  └─────────────────────────────┘ │ │
│            │               └──────────────────────────────────┘ │
└────────────┼────────────────────────────────────────────────────┘
             │ uses
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Channel (Transport Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               Channel (Abstract)                          │  │
│  │  • receive_burst(Buffer**, max) → count                   │  │
│  │  • send_burst(Buffer**, count) → count                    │  │
│  │  • acquire_buffer() → Buffer*                             │  │
│  │  • release_buffer(Buffer*)                                │  │
│  │  • register_memory(addr, size)                            │  │
│  └────────────┬──────────────────────────────────────────────┘  │
│               │ implements                                      │
│  ┌────────────▼──────────────────────────────────────────────┐  │
│  │            RoCEChannel                                    │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │  InfiniBand Resources (self-contained)             │   │  │
│  │  │  • ibv_context  • ibv_pd  • ibv_qp  • ibv_cq       │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │  RoCEBufferPool                                    │   │  │
│  │  │  • Pre-allocated DMA buffers                       │   │  │
│  │  │  • Wraps ring buffer slots (zero-copy)             │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │  RoCERingBuffer                                    │   │  │
│  │  │  • RDMA WRITE target                               │   │  │
│  │  │  • Sequence number polling                         │   │  │
│  │  │  • [Header|Slot0|Slot1|...]                        │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Optional Shared Components (composition):                      │
│  ┌──────────────────────┐     ┌─────────────────────────────┐   │
│  │   VerbsContext       │     │   VerbsFlowSwitch           │   │
│  │                      │     │                             │   │
│  │  • Shared ibv_context│     │  • ibv_create_flow()        │   │
│  │  • Shared ibv_pd     │     │  • UDP port → QP routing    │   │
│  │  • For memory sharing│     │  • Hardware steering        │   │
│  │  • Optional          │     │  • Optional                 │   │
│  └──────────────────────┘     └─────────────────────────────┘   │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Buffer (Memory)                              │  │
│  │                                                           │  │
│  │  [Headroom | Data | Tailroom]                             │  │
│  │   ^         ^                                             │  │
│  │   base      data_ptr                                      │  │
│  │                                                           │  │
│  │  • prepend() - add headers                                │  │
│  │  • Zero-copy via pointer manipulation                     │  │
│  │  • Wraps backend-specific memory                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 QCS (Control Layer)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              QCSDevice                                    │  │
│  │  • establish_connection(RoCEChannel*)                     │  │
│  │  • upload_program(binary)                                 │  │
│  │  • trigger() / abort()                                    │  │
│  │  • is_connected() / is_complete()                         │  │
│  └────────────────┬──────────────────────────────────────────┘  │
│                   │ uses                                        │
│  ┌────────────────▼──────────────────────────────────────────┐  │
│  │           ControlServer (UDP)                             │  │
│  │                                                           │  │
│  │  Protocol:                                                │  │
│  │  1. Wait for QCS DISCOVER packet                          │  │
│  │  2. Send Channel params (QPN, GID, vaddr, rkey)           │  │
│  │  3. Receive QCS params (QPN, GID)                         │  │
│  │  4. Configure Channel with remote QP                      │  │
│  │  5. Send START/STOP commands                              │  │
│  │                                                           │  │
│  │  • exchange_connection_params(RoCEChannel*)               │  │
│  │  • send_command(cmd, params)                              │  │
│  │  • wait_for_response(timeout)                             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Component Relationships:**

- **Daemon** owns a **Channel** and uses **FunctionRegistry** for dispatch
- **Dispatcher** (CPU or GPU) polls **Channel** and invokes registered functions
- **FunctionRegistry** contains **FunctionWrappers** with automatic serialization
- **Streams** work over **Channel** (persistent) or **Buffer** (temporary)
- **RoCEChannel** contains **RoCEBufferPool** and **RoCERingBuffer**
- **RoCEChannel** optionally uses **VerbsContext** (shared resources) or **VerbsFlowSwitch** (steering)
- **QCSDevice** uses **ControlServer** for UDP protocol and configures **RoCEChannel**
- **Buffer** provides zero-copy abstraction over backend-specific memory

### 5.3 Level 3: Code Structure

#### 5.3.1 Transport Layer (Channel)

**Location:** `include/cudaq/nvqlink/network/`

The Channel abstraction provides hardware-independent network I/O:

```cpp
class Channel {
public:
  // Lifecycle
  virtual void initialize() = 0;
  virtual void cleanup() = 0;
  
  // Packet I/O (zero-copy)
  virtual uint32_t receive_burst(Buffer** buffers, uint32_t max) = 0;
  virtual uint32_t send_burst(Buffer** buffers, uint32_t count) = 0;
  
  // Buffer management (pre-allocated pool)
  virtual Buffer* acquire_buffer() = 0;
  virtual void release_buffer(Buffer* buffer) = 0;
  
  // Memory registration for RDMA
  virtual void register_memory(void* addr, size_t size) = 0;
  
  // Configuration
  virtual void configure_queues(const std::vector<uint32_t>& queue_ids) = 0;
  
  // Execution model
  virtual ChannelModel get_execution_model() const = 0;
};
```

**Key Characteristics:**

- **Self-sufficient**: Each Channel manages its own hardware resources
- **Zero-copy**: Direct access to NIC DMA buffers
- **Pre-allocated**: Buffer pools created at initialization (no allocation in hot path)
- **Abstract**: Works with RoCE, mock implementations, and future backends (DOCA, Opnic)

**Implemented Channels:**

- `RoCEChannel`: RDMA over Converged Ethernet (libibverbs)
- Mock channels in `examples/mock/` for testing

#### 5.3.2 Service Layer (Daemon)

**Location:** `include/cudaq/nvqlink/daemon/`

The Daemon provides RPC functionality on top of any Channel:

```cpp
class Daemon {
public:
  explicit Daemon(DaemonConfig config, std::unique_ptr<Channel> channel);
  
  // Lifecycle
  void start();
  void stop();
  bool is_running() const;
  
  // Function registration (automatic serialization)
  template<typename F>
  void register_function(F&& fn, std::string_view name);
  
  // Statistics
  struct Stats { uint64_t packets_received, packets_sent, errors; };
  Stats get_stats() const;
};
```

**Sub-components:**

- **FunctionRegistry** (`daemon/registry/`): Automatic type inference, hash-based lookup, compile-time wrapper generation
- **Dispatcher** (`daemon/dispatcher/`): `CPUDispatcher` and `GPUDispatcher` for packet processing
- **Serialization** (`network/serialization/`): `InputStream`/`OutputStream` and GPU variants

#### 5.3.3 Control Layer (QCS)

**Location:** `include/cudaq/nvqlink/qcs/`

```cpp
class QCSDevice {
public:
  explicit QCSDevice(const QCSDeviceConfig& config);
  
  void establish_connection(RoCEChannel* channel);
  bool is_connected() const;
  void disconnect();
  
  void upload_program(const std::vector<std::byte>& binary);
  void trigger();
  bool is_complete() const;
  void abort();
};

class ControlServer {
public:
  explicit ControlServer(uint16_t port);
  
  void start();
  void stop();
  bool exchange_connection_params(RoCEChannel* channel);
  void send_command(const std::string& cmd, const nlohmann::json& params = {});
  nlohmann::json wait_for_response(std::chrono::milliseconds timeout);
};
```

## 6. Runtime View

This section describes the key runtime scenarios based on actual code in
`examples/roce/`, `lib/daemon/`, and `lib/network/channels/roce/`.

### 6.1 Connection Setup

The RTH (server) starts first and controls the connection sequence.

```text
      RTH (Server)                                      QCS (Client)
           │                                                 │
           │  Initialize channel & control server            │
           │  ────────────────────────────────────           │
           │  channel = RoCEChannel(device, port)            │
           │  channel->initialize()                          │
           │  control_server.start()  // UDP:9999            │
           │                                                 │
           │◀─────────────── DISCOVER ───────────────────────│
           │                  (UDP)                          │
           │                                                 │
           │─────────────── CONNECTION_PARAMS ──────────────▶│
           │                 • qpn                           │
           │                 • gid                           │  Create QP
           │                 • vaddr (ring buffer)           │
           │                 • rkey  (RDMA key)              │
           │                 • num_slots, slot_size          │
           │                                                 │
           │◀───────────────── ACK ──────────────────────────│
           │                 • client qpn                    │
           │                 • client gid                    │
           │                                                 │
           │  set_remote_qp(qpn, gid)                        │
           │  daemon->start()                                │
           │                                                 │
           │─────────────────── START ──────────────────────▶│
           │                 {cmd: "START"}                  │  Transition QP
           │                                                 │  to RTS
         ══╪═════════════════════════════════════════════════╪══
           │              Data plane active                  │
         ══╪═════════════════════════════════════════════════╪══
```

**Source:**

- `examples/roce/udp_control_server.cpp`
- `examples/roce/roce_daemon_client.cpp`

### 6.2 RPC Round-Trip

Client sends requests via RDMA WRITE; server responds via RDMA SEND.

```text
        QCS (Client)                           RTH (Server)
             │                                      │
             │                                      │  Dispatcher polling...
             │                                      │
             │  Build packet:                       │
             │  ┌──────────────────────┐            │
             │  │ seq# | len | func_id │            │
             │  │ arg_len | args...    │            │
             │  └──────────────────────┘            │
             │                                      │
             │══════ RDMA WRITE ═══════════════════▶│  Ring buffer
             │       to: vaddr + slot_offset        │  slot updated
             │       rkey: server_rkey              │
             │                                      │
             │                                      │  poll_next() detects seq#
             │                                      │         ▼
             │                                      │  process_packet():
             │                                      │    parse header
             │                                      │    lookup function
             │                                      │    func(in, out)
             │                                      │         ▼
             │                                      │  Prepare response:
             │                                      │  ┌─────────────────┐
             │                                      │  │ status | len    │
             │                                      │  │ result...       │
             │                                      │  └─────────────────┘
             │                                      │
             │◀═══════════════════ RDMA SEND ═══════│
             │                                      │
```

**Source:**

- `lib/daemon/dispatcher/cpu_dispatcher.cpp`,
- `lib/network/channels/roce/roce_channel.cpp`

### 6.3 Dispatcher Loop

```text
┌───────────────────────────────────────────────────────────────────┐
│  CPUDispatcher::polling_worker_thread(core_id)                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│    while (running_) {                                             │
│        Buffer* buffers[32];                                       │
│        n = channel_->receive_burst(buffers, 32);  ◀── poll ring   │
│                                                                   │
│        if (n == 0) { yield(); continue; }                         │
│                                                                   │
│        for (i = 0; i < n; i++) {                                  │
│            process_packet(buffers[i]);                            │
│        }                                                          │
│    }                                                              │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│  process_packet(buffer):                                          │
│                                                                   │
│    header = (RPCHeader*) buffer->get_data()                       │
│    func   = registry_->lookup(header->function_id)                │
│                                                                   │
│    InputStream  in(args_ptr, header->arg_len)    ← zero-copy      │
│    OutputStream out(buffer, capacity)            ← same buffer    │
│                                                                   │
│    status = func->cpu_function(in, out)          ← user code      │
│                                                                   │
│    response = buffer->prepend(sizeof(RPCResponse))                │
│    response->status = status                                      │
│    response->result_len = out.bytes_written()                     │
│                                                                   │
│    channel_->send_burst(&buffer, 1)              ← RDMA SEND      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Source:** `lib/daemon/dispatcher/cpu_dispatcher.cpp`

### 6.4 Ring Buffer Memory Layout

The server exposes a ring buffer for clients to write into via RDMA.

```text
vaddr (exported to client)
    │
    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Header (64 bytes)                                          │
    │    magic, version, num_slots, slot_size                     │
    ├─────────────────────────────────────────────────────────────┤
    │  Slot 0                                                     │
    │  ┌─────────┬─────────┬─────────┬───────────────────────────┐│
    │  │ seq# 8B │ len 4B  │ pad 4B  │ payload (up to 2032B)     ││
    │  └─────────┴─────────┴─────────┴───────────────────────────┘│
    ├─────────────────────────────────────────────────────────────┤
    │  Slot 1                                                     │
    │  ┌─────────┬─────────┬─────────┬───────────────────────────┐│
    │  │ seq# 8B │ len 4B  │ pad 4B  │ payload                   ││
    │  └─────────┴─────────┴─────────┴───────────────────────────┘│
    ├─────────────────────────────────────────────────────────────┤
    │  ...                                                        │
    ├─────────────────────────────────────────────────────────────┤
    │  Slot N-1                                                   │
    └─────────────────────────────────────────────────────────────┘

Detection mechanism:
────────────────────
  Server polls: slot[current].seq# == expected_seq# ?
  
  • No  → no data, return NULL
  • Yes → data ready, return pointer to payload (zero-copy)
          advance: current_slot++, expected_seq++

Why this works:
───────────────
  • Client writes payload first, then seq# (memory barrier)
  • Server sees seq# update → payload is guaranteed visible
  • No interrupts, no CQ polling, no CPU notification needed
```

**Source:**

- `include/cudaq/nvqlink/network/roce/roce_ring_buffer.h`,
- `examples/roce/roce_daemon_client.cpp`

## 7. Deployment View

### 7.1 Infrastructure Overview

WIP

### 7.2 Hardware Requirements

| Component | Production | Development |
|-----------|------------|-------------|
| **NIC** | Mellanox ConnectX (RoCE v2) | Any NIC with Soft-RoCE (rxe) support |
| **CPU** | Multi-core with RT capabilities | Any x86_64 |
| **GPU** | NVIDIA GPU with GPUDirect RDMA | Optional (can use CPU mode) |
| **Network** | Lossless Ethernet (PFC/ECN) | Standard Ethernet |

### 7.3 Software Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (kernel 5.x+) |
| **RDMA Stack** | rdma-core, libibverbs |
| **Compiler** | GCC 9+ or Clang 10+ (C++17) |
| **CUDA** | CUDA 12+ (optional, for GPU mode) |

## 8. Cross-cutting Concepts

### 8.1 Unified Streams

**Location:** `include/cudaq/nvqlink/network/serialization/`

Streams provide type-safe, zero-copy serialization with dual-mode operation:

**CPU Streams:**

```cpp
// Mode 1: Channel (persistent, auto packet management)
InputStream in(channel);
OutputStream out(channel);

// Mode 2: Buffer (temporary, for Daemon RPC handlers)
int rpc_handler(InputStream& in, OutputStream& out) {
  int a = in.read<int>();
  float b = in.read<float>();
  out.write(a * b);
  return 0;
}
```

**GPU Streams:**

```cpp
// Mode 1: GPUChannel (persistent, in CUDA kernel)
__global__ void my_kernel(GPUChannel* gpu_chan) {
  GPUInputStream in(*gpu_chan);
  GPUOutputStream out(*gpu_chan);
  
  while (gpu_chan->is_running()) {
    if (in.available()) {
      int data = in.read<int>();
      out.write(data * 2);
      out.flush();
    }
  }
}

// Mode 2: Buffer (temporary, for GPU Daemon functions)
__global__ void gpu_rpc(char* packet_data, size_t size) {
  GPUInputStream in(packet_data, size);
  int value = in.read<int>();
  // Process...
}
```

**Design:**

- Non-polymorphic (constructor determines mode for zero overhead)
- Zero-copy via pointer arithmetic on DMA buffers
- CPU and GPU variants share conceptual API
- GPU version is header-only for device code

### 8.2 Buffer Management

**Location:** `include/cudaq/nvqlink/network/memory/`

```cpp
class Buffer {
public:
  Buffer(void* base_addr, size_t total_size, size_t headroom, size_t tailroom);
  
  void* get_data() const;
  size_t get_data_length() const;
  void set_data_length(size_t len);
  void* prepend(size_t bytes);  // Add header
  
  void* get_base_address() const;
  size_t get_total_size() const;
};
```

**Memory Layout:**

```text
[Headroom | Data | Tailroom]
 ^         ^
 base      data_ptr
```

**Key Features:**

- Pre-allocated pools (no allocation in hot path)
- Headroom/tailroom for protocol headers
- Direct NIC DMA buffer wrapping
- Zero-copy via `reset()` method for backend-specific memory

### 8.3 RoCE/RDMA Backend

**Location:** `include/cudaq/nvqlink/network/roce/`

The RoCE backend implements RDMA communication using Unreliable Connection (UC) mode with memory polling:

**Components:**

- **RoCEChannel**: UC queue pairs, self-contained IB resources
- **RingBuffer**: Fixed-size slots for RDMA WRITE targets
- **VerbsContext**: Optional shared IB context and PD
- **VerbsFlowSwitch**: Hardware flow steering via `ibv_create_flow()`

**Key Features:**

- UC mode: Connection-oriented, no ACKs (low latency)
- Memory polling: Direct sequence number checks (no CQ overhead)
- Zero-copy: Direct memory access, no packet copies
- Soft-RoCE (rxe) supported for development
- Hardware RoCE NICs supported for production

### 8.4 Configuration

**Location:** `include/cudaq/nvqlink/config.h` (umbrella header)

```cpp
// Network Config (network/config.h)
struct ChannelConfig {
  std::string nic_device;
  uint32_t queue_id;
  size_t pool_size_bytes{64 * 1024 * 1024};
  size_t buffer_size_bytes{2048};
  size_t headroom_bytes{256};
  size_t tailroom_bytes{64};
};

// Daemon Config (daemon/config.h)
enum class DatapathMode { CPU, GPU };

struct DaemonConfig {
  std::string id;
  DatapathMode datapath_mode;
  ComputeConfig compute;
  bool is_valid() const;
};

// QCS Config (qcs/config.h)
struct QCSDeviceConfig {
  std::string name;
  uint16_t control_port{9999};
  bool is_valid() const;
};
```

## 9. Architecture Decisions

### 9.1 UC Mode over RC Mode

**Context:** RDMA offers Reliable Connection (RC) and Unreliable Connection
(UC) modes.

**Decision:** Use UC mode for data plane communication.

**Rationale:** UC mode eliminates ACK overhead, providing lower latency at the
cost of reliability. For our use case, the application-level protocol handles
retries if needed, and the lossless Ethernet fabric prevents packet loss.

### 9.2 Memory Polling over CQ Polling

**Context:** RDMA completions can be detected via Completion Queue (CQ) polling
or direct memory polling.

**Decision:** Use memory polling with sequence numbers in the ring buffer.

**Rationale:** Memory polling reduces latency by avoiding CQ overhead and
enables direct integration with GPU polling loops.

### 9.3 Non-Polymorphic Streams

**Context:** Stream classes could use virtual inheritance for flexibility or
templates for performance.

**Decision:** Use non-polymorphic streams with constructor-based mode selection.

**Rationale:** Eliminates virtual dispatch overhead in the hot path while
maintaining flexibility through dual-mode construction.

## 10. Quality Requirements

### 10.1 Quality Tree

*Note: The (1), (2), (3), (4), (5) references correspond to the top-level quality goals from section 1.2.*

| Quality Category | Quality | Description | Scenario |
|------------------|---------|-------------|----------|
| Performance | Ultra-Low Latency (1) | RPC round-trip must complete within strict timing budgets to prevent logical errors | SC1 |
|  | Throughput | System must sustain high packet processing rates during peak syndrome streaming | SC2 |
|  | Zero Overhead (4) | No memory allocation, exception handling, or virtual dispatch in the packet processing path |  |
| Flexibility | Extensibility (2) | QCS vendors can integrate proprietary hardware without exposing instruction set details or requiring core changes | SC3, SC4 |
|  | Productivity (5) | Researchers can develop and test QEC protocols without physical hardware access | SC5 |
| Maintainability | Testability | Core components must be testable in isolation without hardware dependencies | SC6 |
|  | Code Quality | Codebase follows modern C++ patterns with RAII resource management and no hot path allocations |  |
| Scalability | Linear Scaling (3) | Adding GPUs increases throughput linearly for parallel window processing |  |

### 10.2 Quality Scenarios

| ID | Scenario |
|----|----------|
| SC1 | A QEC decoder receives syndrome data and returns corrections with < 10 µs round-trip latency (p99) under sustained load |
| SC2 | During peak syndrome streaming at 1 MHz, the system processes > 1M packets/second without dropping data |
| SC3 | A developer implements a new Channel backend (e.g., for a new network technology) in less than 1 day |
| SC4 | A QCS vendor integrates their proprietary PPU firmware by implementing a custom Channel; no changes to NVQLink core are required |
| SC5 | A researcher develops a new AI-based decoder on their laptop using VPPU simulation; the same code deploys to physical hardware by changing only a config flag |
| SC6 | Unit test coverage exceeds 80% of core components, with all tests runnable without RDMA hardware via mock channels |

### 10.3 Performance Characteristics

#### Zero-Allocation Hot Path

- Buffer pools pre-allocated at initialization
- No malloc/free during packet processing
- No exception throwing in critical paths
- Lock-free where possible (ring buffer polling)

#### Low Latency Design

- **Memory polling**: Direct sequence number checks (no CQ polling overhead)
- **UC mode RDMA**: No ACKs or flow control
- **Batch processing**: Receive multiple packets per poll cycle
- **Direct dispatch**: Minimal indirection from NIC to user function

#### Scalability

- **Multiple queues**: Each `Channel` owns one queue, create multiple `Channels` for parallelism
- **CPU pinning**: Configure which cores process which queues
- **GPU offload**: Persistent kernels for GPU-side packet processing
- **Hardware flow steering**: NIC-level packet routing to queues

## 11. Risks and Technical Debt

WIP

## 12. Glossary

| Term | Definition |
|------|------------|
| **CQ** | Completion Queue - RDMA queue for work request completions |
| **GID** | Global Identifier - 128-bit address for RDMA communication |
| **HPC** | High-Performance Computing |
| **libibverbs** | Linux RDMA user-space API library |
| **PD** | Protection Domain - RDMA security/isolation boundary |
| **PPU** | Pulse Processing Unit - processor within QCS |
| **QCS** | Quantum Control System - FPGA/processor controlling the QPU |
| **QEC** | Quantum Error Correction |
| **QP** | Queue Pair - RDMA send/receive queue combination |
| **QPN** | Queue Pair Number - identifier for a QP |
| **QPU** | Quantum Processing Unit - the physical quantum hardware |
| **RC** | Reliable Connection - RDMA transport mode with ACKs |
| **RDMA** | Remote Direct Memory Access |
| **rkey** | Remote Key - RDMA memory access permission key |
| **RoCE** | RDMA over Converged Ethernet |
| **RTH** | Real-time Host - HPC node running NVQLink |
| **UC** | Unreliable Connection - RDMA transport mode without ACKs |
| **vaddr** | Virtual Address - memory address for RDMA operations |
| **VPPU** | Virtual PPU - software emulation of PPU for development |

## Appendix A: Usage Patterns

### A.1 RPC Server (Daemon Mode)

```cpp
#include "cudaq/nvqlink/daemon/daemon.h"
#include "cudaq/nvqlink/network/roce/roce_channel.h"
#include "cudaq/nvqlink/qcs/qcs_device.h"

// Application callbacks
int process(int input) { return input * 2; }
void handle_data(const Data& in, Result& out) { /* ... */ }

int main() {
  // 1. Create Channel (data plane)
  auto flow_switch = std::make_shared<VerbsFlowSwitch>();
  auto channel = std::make_unique<RoCEChannel>("mlx5_0", 9000, flow_switch);
  channel->initialize();
  
  // 2. Create QCS control plane (UDP)
  QCSDeviceConfig qcs_config;
  qcs_config.control_port = 9999;
  QCSDevice qcs(qcs_config);
  
  // Exchange RDMA parameters over UDP
  qcs.establish_connection(channel.get());
  
  // 3. Create Daemon (owns channel)
  DaemonConfig daemon_config;
  daemon_config.id = "my_server";
  daemon_config.datapath_mode = DatapathMode::CPU;
  daemon_config.compute.cpu_cores = {0};
  
  Daemon daemon(daemon_config, std::move(channel));
  
  // 4. Register functions
  daemon.register_function(NVQLINK_RPC_HANDLE(process));
  daemon.register_function(NVQLINK_RPC_HANDLE(handle_data));
  
  // 5. Start data plane
  daemon.start();
  
  // 6. Trigger program execution (control plane)
  qcs.trigger();  // Sends START command via UDP
  
  // 7. Run
  while (running) {
    sleep(1);
    auto stats = daemon.get_stats();
    std::cout << "RX: " << stats.packets_received << std::endl;
  }
  
  daemon.stop();
  qcs.disconnect();
}
```

### A.2 Direct I/O (Channel Mode)

```cpp
#include "cudaq/nvqlink/network/roce/roce_channel.h"
#include "cudaq/nvqlink/network/serialization/input_stream.h"
#include "cudaq/nvqlink/network/serialization/output_stream.h"

int main() {
  // Create channel
  auto channel = std::make_unique<RoCEChannel>("mlx5_0", 9000, flow_switch);
  channel->initialize();
  
  // Create persistent streams
  InputStream in(*channel);
  OutputStream out(*channel);
  
  // Custom protocol loop
  while (running) {
    if (in.available()) {
      // Read request
      uint32_t cmd = in.read<uint32_t>();
      float value = in.read<float>();
      
      // Process
      float result = process(cmd, value);
      
      // Write response
      out.write(result);
      out.flush();  // Sends packet
    }
  }
  
  channel->cleanup();
}
```

### A.3 GPU Mode

```cpp
#include "cudaq/nvqlink/network/gpu_channel.h"

__global__ void gpu_processing(GPUChannel* gpu_chan) {
  GPUInputStream in(*gpu_chan);
  GPUOutputStream out(*gpu_chan);
  
  while (gpu_chan->is_running()) {
    if (in.available()) {
      int data = in.read<int>();
      
      // GPU processing
      int result = data * threadIdx.x;
      
      out.write(result);
      out.flush();
    }
  }
}

int main() {
  auto channel = std::make_unique<RoCEChannel>(...);
  channel->initialize();
  
  GPUChannel gpu_chan(std::move(channel));
  
  // Launch persistent kernel
  gpu_chan.start_kernel(gpu_processing, blocks, threads);
  
  // Kernel runs until stopped
  while (running) { sleep(1); }
  
  gpu_chan.stop_kernel();
}
```

## Appendix B: Directory Structure

```text
cudaq-qclink/
├── include/cudaq/nvqlink/
│   ├── nvqlink.h                     # Main public API
│   ├── config.h                      # Umbrella config header
│   ├── compiler.h                    # CUDA-Q compiler integration
│   ├── device.h                      # device_ptr struct (legacy)
│   ├── lqpu.h                        # LQPU configuration
│   │
│   ├── daemon/                       # Layer 2: RPC Service
│   │   ├── daemon.h                  # Main Daemon class
│   │   ├── config.h                  # DaemonConfig, ComputeConfig
│   │   ├── dispatcher/
│   │   │   ├── dispatcher.h          # Abstract dispatcher
│   │   │   ├── cpu_dispatcher.h      # CPU packet processing
│   │   │   └── gpu_dispatcher.h      # GPU persistent kernel
│   │   └── registry/
│   │       ├── function_registry.h   # Function lookup
│   │       ├── function_traits.h     # Type inference
│   │       ├── function_wrapper.h    # Automatic marshalling
│   │       └── rpc_builder.h         # RPC handle macro
│   │
│   ├── network/                      # Layer 1: Transport
│   │   ├── channel.h                 # Abstract Channel interface
│   │   ├── config.h                  # ChannelConfig, MemoryConfig
│   │   ├── gpu_channel.h             # GPU-controlled datapath
│   │   ├── memory/
│   │   │   ├── buffer.h              # Buffer abstraction
│   │   │   ├── buffer_handle.h       # RAII buffer wrapper
│   │   │   └── memory_pool.h         # Pre-allocated buffer pool
│   │   ├── roce/                     # RoCE/RDMA backend
│   │   │   ├── roce_channel.h        # RoCE Channel implementation
│   │   │   ├── roce_buffer_pool.h    # RoCE-specific buffer pool
│   │   │   ├── roce_ring_buffer.h    # RDMA ring buffer
│   │   │   └── verbs_context.h       # Shared ibv resources
│   │   ├── serialization/            # Type-safe streams
│   │   │   ├── input_stream.h        # CPU input stream
│   │   │   ├── output_stream.h       # CPU output stream
│   │   │   ├── gpu_input_stream.h    # GPU input stream
│   │   │   └── gpu_output_stream.h   # GPU output stream
│   │   └── steering/
│   │       ├── flow_switch.h         # Abstract flow steering
│   │       └── verbs_flow_switch.h   # libibverbs flow steering
│   │
│   ├── qcs/                          # Layer 3: Control Plane
│   │   ├── config.h                  # QCSDeviceConfig
│   │   ├── qcs_device.h              # QCS connection management
│   │   └── control_server.h          # UDP control protocol
│   │
│   ├── compilers/                    # CUDA-Q integration
│   │   └── cudaq/
│   │       ├── cudaq.h
│   │       └── cudaq_toolchain.h
│   │
│   └── utils/                        # Utilities
│       ├── extension_point.h
│       └── instrumentation/
│           ├── domains.h
│           ├── logger.h
│           ├── nvtx.h
│           ├── profiler.h
│           ├── quill_logger.h
│           └── tracy.h
│
├── lib/                              # Implementations
│   ├── nvqlink.cpp
│   ├── compiler.cpp
│   ├── daemon/
│   ├── network/
│   ├── qcs/
│   ├── compilers/cudaq/
│   └── utils/
│
├── examples/                         # Example applications
│   ├── mock/                         # Development examples
│   └── roce/                         # Production examples
│
└── unittests/                        # Unit tests
    ├── test_daemon.cpp
    ├── test_streams.cpp
    └── test_utils/
```

## Appendix C: Testing

### Unit Tests

**Location:** `unittests/`

- **test_daemon.cpp**: Daemon lifecycle, function registration, configuration
- **test_streams.cpp**: Serialization/deserialization, type safety
- **LoopbackChannel**: In-memory channel for hardware-independent testing

All tests use GoogleTest framework and run without hardware dependencies.

### Examples

**Location:** `examples/`

- **mock/**: Development examples using in-memory channels
- **roce/**: Production examples using RoCE/RDMA
  - `roce_daemon`: Full RPC server
  - `roce_channel`: Direct I/O mode
  - Clients for both modes
