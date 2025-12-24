# This document is a work in progress, and it is not yet implemented

# End-to-End Example: Quantum Kernel with device_call

This example shows how a QEC decoder callback would work in the new architecture.

## Two Planes: Control vs Data

```text
RTH (HPC Node)
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  CONTROL PLANE (UDP)                DATA PLANE (RoCE)              │
│  ───────────────────                ─────────────────              │
│  ┌────────────────────┐             ┌────────────────────┐         │
│  │  QCSDevice         │             │  Daemon            │         │
│  │  ────────────────  │             │  ────────────────  │         │
│  │  • ControlServer   │             │  • FunctionRegistry│         │
│  │    (UDP socket)    │ configures  │  • Dispatcher      │         │
│  │  • upload_program()│────────────►│  • Callbacks       │         │
│  │  • trigger()       │             │                    │         │
│  └─────────┬──────────┘             └─────────┬──────────┘         │
│            │                                  │                    │
│            │ UDP                              │ owns               │
│            │                                  ▼                    │
│            │                    ┌─────────────────────────────┐    │
│            │                    │     Channel (RoCE)          │    │
│            │                    │  • Ring buffer for packets  │    │
│            │                    │  • RDMA WRITE/SEND          │    │
│            │                    └──────────────┬──────────────┘    │
└────────────┼───────────────────────────────────┼───────────────────┘
             │                                   │
             │ UDP (out-of-band)                 │ RDMA (data path)
             │ • Exchange QPN, GID, rkey         │ • device_call packets
             │ • START/STOP commands             │ • Results
             │ • Program upload (optional)       │
             ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  QCS (FPGA / Pulse Processor)                                        │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐ │
│  │  UDP Client         │    │  RDMA Engine (HoloLink)              │ │
│  │  • Receives params  │    │  • RDMA WRITE for device_call        │ │
│  │  • Receives START   │    │  • Receives RDMA SEND results        │ │
│  └─────────────────────┘    └──────────────────────────────────────┘ │
│                                                                      │
│  Pulse Processor: executes quantum gates, triggers device_call       │
└──────────────────────────────────────────────────────────────────────┘
```

The Control Plane uses simple **UDP sockets** to exchange RDMA connection
parameters (QPN, GID, vaddr, rkey) and send commands. This configures the Data
Plane's **RoCE Channel** before data transfer begins.

## The Quantum Kernel (CUDA-Q side)

```cpp
// qec_kernel.cpp - Compiled for the QSC/FPGA
#include "cudaq.h"

// Declare device callbacks (implemented on RTH)
extern "C" void decoder_enqueue(uint64_t syndrome);
extern "C" bool decoder_get_correction();

__qpu__ void qec_cycle(cudaq::qvector<>& data, cudaq::qvector<>& ancilla, int rounds) {
  for (int r = 0; r < rounds; ++r) {
    // Stabilizer measurement circuit
    // ... (gates on data and ancilla)
    
    auto syndrome = mz(ancilla);
    
    // Send syndrome to RTH for decoding
    cudaq::device_call(/*device_id=*/0, decoder_enqueue, cudaq::to_integer(syndrome));
  }
  
  // Get final correction from decoder
  bool need_correction = cudaq::device_call(/*device_id=*/0, decoder_get_correction);
  
  if (need_correction) {
    x(data[0]);  // Apply correction
  }
}
```

The compiler lowers `cudaq::device_call` to RDMA operations that send packets
to the RTH.

## The RTH Server

```cpp
// rth_qec_server.cpp - Complete RTH setup
#include "cudaq/nvqlink/daemon/daemon.h"
#include "cudaq/nvqlink/channel/roce/roce_channel.h"
#include "cudaq/nvqlink/qcs/qcs_device.h"        // NEW: QCS control plane
#include "cudaq/nvqlink/qcs/control_server.h"    // NEW: RDMA handshake

using namespace cudaq::nvqlink;

//=============================================================================
// CALLBACKS (registered with Daemon - data plane)
//=============================================================================

class Decoder {
  std::vector<uint64_t> syndromes_;
public:
  void enqueue(uint64_t syndrome) { syndromes_.push_back(syndrome); }
  bool get_correction() { return decode_syndromes(syndromes_); }
};

Decoder decoder;

void decoder_enqueue(uint64_t syndrome) { decoder.enqueue(syndrome); }
bool decoder_get_correction() { return decoder.get_correction(); }

//=============================================================================
// MAIN
//=============================================================================

int main() {
  //===========================================================================
  // STEP 1: Create RoCE Channel (DATA PLANE transport)
  // Channel is NOT initialized yet - needs RDMA params from control plane
  //===========================================================================
  auto flow_switch = std::make_shared<VerbsFlowSwitch>();
  auto channel = std::make_unique<RoCEChannel>("mlx5_0", 9000, flow_switch);
  
  // Initialize local resources (QP, buffers) but not connected yet
  channel->initialize();
  
  //===========================================================================
  // STEP 2: Create QCS Device (CONTROL PLANE - uses UDP, not RoCE)
  // This handles: RDMA parameter exchange, program upload, execution trigger
  //===========================================================================
  QCSDeviceConfig qcs_config;
  qcs_config.control_port = 9999;        // UDP port for out-of-band signaling
  
  QCSDevice qcs_device(qcs_config);      // Uses its own UDP socket
  
  // Start UDP control server and wait for QCS to connect
  std::cout << "Waiting for QCS connection on UDP port 9999...\n";
  
  // This does the RDMA handshake over UDP:
  // 1. QCS sends DISCOVER packet
  // 2. RTH sends: {QPN, GID, vaddr, rkey} (channel's connection params)
  // 3. QCS sends: {QPN, GID} (its connection params)
  // 4. RTH configures channel with remote QP info
  qcs_device.establish_connection(channel.get());  // Configures the channel!
  
  std::cout << "QCS connected! RDMA data path ready.\n";
  
  //===========================================================================
  // STEP 3: Create Daemon (DATA PLANE - uses the now-configured RoCE channel)
  //===========================================================================
  DaemonConfig daemon_config;
  daemon_config.id = "qec_decoder";
  daemon_config.mode = DatapathMode::CPU;
  daemon_config.cpu_cores = {0};
  
  // Daemon takes ownership of the configured channel
  Daemon daemon(daemon_config, std::move(channel));
  
  // Register callbacks
  daemon.register_function(NVQLINK_RPC_HANDLE(decoder_enqueue));
  daemon.register_function(NVQLINK_RPC_HANDLE(decoder_get_correction));
  
  // Start data plane (begins polling for RDMA packets)
  daemon.start();
  
  //===========================================================================
  // STEP 4: Upload program and trigger (via UDP control plane)
  //===========================================================================
  
  // Load compiled kernel (from nvq++ compilation)
  auto kernel = load_compiled_kernel("qec_kernel.o");
  
  // Upload to QCS (over UDP or separate channel - NOT the RoCE data path)
  qcs_device.upload_program(kernel);
  
  std::cout << "Program uploaded. Triggering execution...\n";
  
  // Send START command over UDP
  // QCS begins running quantum gates; device_call packets arrive via RDMA
  qcs_device.trigger();
  
  //===========================================================================
  // STEP 5: Run until completion or interrupt
  //===========================================================================
  std::cout << "QEC running. Press Ctrl+C to stop.\n";
  
  while (running && !qcs_device.is_complete()) {
    sleep(1);
    auto stats = daemon.stats();
    std::cout << "Packets: rx=" << stats.rx << " tx=" << stats.tx << "\n";
  }
  
  // Cleanup
  daemon.stop();
  qcs_device.disconnect();  // Sends STOP over UDP
  
  return 0;
}
```

## What Happens Under the Hood

1. **Compilation**: `cudaq::device_call(0, decoder_enqueue, syndrome)` becomes:
   - Serialize: `[function_id=hash("decoder_enqueue"), args=[syndrome]]`
   - Network op: RDMA WRITE to RTH ring buffer

2. **RTH receives packet**:
   - Daemon's dispatcher polls Channel
   - Looks up `function_id` in FunctionRegistry
   - Creates `InputStream` from packet buffer
   - Calls wrapper: `decoder_enqueue_wrapper(in, out)`
   - Wrapper auto-deserializes: `uint64_t syndrome = in.read<uint64_t>()`
   - Calls actual function: `decoder_enqueue(syndrome)`
   - For functions with return value: `out.write(result)`
   - Sends response via RDMA SEND

3. **FPGA receives response**:
   - Polls for RDMA SEND completion
   - Deserializes result
   - Continues quantum execution
