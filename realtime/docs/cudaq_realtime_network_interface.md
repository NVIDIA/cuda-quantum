# CUDA-Q Realtime Network Layer Provider Interface

Using the [CUDA-Q Realtime host API](cudaq_realtime_host_api.md), one can
build an end-to-end RPC dispatch solution, as demonstrated in the
Hololink RDMA example  with a simple increment RPC handler
(`realtime/unittests/utils`).  

In addition to building an end-to-end application based on specific networking software,
e.g., Hololink in the above example, we also provide a networking provider
wrapper interface, allowing one to build a networking-agnostic application.

## Quick Start

CUDA-Q Realtime networking interface consists of a set of APIs to
construct a real-time RPC dispatch solution in a networking-agnostic manner.
These APIs are backed by a provider plugin (as a shared library)
implementing the specific transport protocol.

The basic APIs for the networking interface are:

### Create the networking 'bridge'

```cpp
/// @brief Create and initialize a transport bridge for the specified provider.
/// For the built-in Hololink provider, this loads the Hololink shared library
/// and initializes the transceiver with the provided `args`.  For the EXTERNAL
/// provider, this loads the shared library specified by the
/// CUDAQ_REALTIME_BRIDGE_LIB environment variable and calls its create callback
/// to initialize the bridge.
cudaq_status_t
cudaq_bridge_create(cudaq_realtime_bridge_handle_t *out_bridge_handle,
                    cudaq_realtime_transport_provider_t provider, int argc,
                    char **argv);
```

This will initialize the networking layer context. The `cudaq_realtime_transport_provider_t`
enum specifies whether it is a builtin provider (e.g., Hololink) or an external one.
For the latter, it will perform dynamic loading to retrieve the
networking implementation. Arguments, e.g., networking information, can also be provided
to initialize the networking context.

### Initialize a connection to the remote peer, e.g., a FPGA

```cpp
/// @brief Connect the transport bridge.
cudaq_status_t cudaq_bridge_connect(cudaq_realtime_bridge_handle_t bridge);
```

### Retrieve the transport context information

This context information can be either a ring buffer (for `cudaq_dispatcher_set_ringbuffer`)
or a unified context for (`cudaq_dispatcher_set_unified_launch`).

```cpp
/// @brief Retrieve the transport context for the given bridge.
/// This could be a ring buffer or unified context.
cudaq_status_t cudaq_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t bridge,
    cudaq_realtime_transport_context_t context_type, void *out_context);
```

### Start the transport layer processing loop, i.e., ready to send and receive packages

```cpp
/// @brief Launch the transport bridge's main processing loop (e.g. start
/// Hololink kernels).
cudaq_status_t cudaq_bridge_launch(cudaq_realtime_bridge_handle_t bridge);
```

Depending on the implementation, this could mean launching kernels/functions to
monitor the network stack (e.g., a socket, RDMA data, etc.) and fill up the RPC
header and payload accordingly as specified in the [message protocol](cudaq_realtime_message_protocol.md).

### Terminate the connection to the remote peer

```cpp
/// @brief Disconnect the transport bridge (e.g. stop Hololink kernels and
/// disconnect).
cudaq_status_t cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t bridge);
```

### Destroy the transport context

```cpp
/// @brief Destroy the transport bridge and release all associated resources.
cudaq_status_t cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t bridge);
```

An example of using this wrapper interface can be found at `realtime/unittests/bridge_interface/hololink/hololink_bridge.cpp`.

## Extending CUDA-Q realtime with a custom networking interface

This guide explains how to integrate a new networking provider
with CUDA-Q realtime via this interface.
The integration process involves creating a shared library implementing
the below `cudaq_realtime_bridge_interface_t` and provide a `cudaq_realtime_get_bridge_interface`
function to retrieve a static instance of this interface.

```cpp
/// @brief Interface struct for transport layer providers.  Each provider must
/// implement this interface and provide a `getter` function
/// (`cudaq_realtime_get_bridge_interface`) that returns a pointer to a
/// statically allocated instance of this struct with the function pointers set
/// to the provider's implementation.
typedef struct {
  int version;
  cudaq_status_t (*create)(cudaq_realtime_bridge_handle_t *, int, char **);
  cudaq_status_t (*destroy)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*get_transport_context)(cudaq_realtime_bridge_handle_t,
                                          cudaq_realtime_transport_context_t,
                                          void *);
  cudaq_status_t (*connect)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*launch)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*disconnect)(cudaq_realtime_bridge_handle_t);

} cudaq_realtime_bridge_interface_t;
```

At runtime, when a `CUDAQ_PROVIDER_EXTERNAL` is requested in `cudaq_bridge_create`,
CUDA-Q will retrieve the environment variable `CUDAQ_REALTIME_BRIDGE_LIB`
to locate the shared library implementing this interface
to provide networking functionality.

### Example

Here's a template for implementing a networking interface wrapper class.

```cpp
#include "<your_networking_library.h>"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

// Custom data structure for your networking stack.
// This will be encapsulated as an opaque `cudaq_realtime_bridge_handle_t`. 
struct ProviderNameNetworkContext {
  
};

/// Implementing the cudaq_realtime_bridge_interface_t functions
extern "C" {
static cudaq_status_t
provider_name_bridge_create(cudaq_realtime_bridge_handle_t *handle, int argc,
                    char **argv) {
 
  // Create and initialize the networking handle 
  // This may take into account the arguments.
  ProviderNameNetworkContext *ctx = new ProviderNameNetworkContext(...);
  
  // Set the output handle to the created context (opaque to the caller)
  *handle = ctx;

  return CUDAQ_OK;
}

static cudaq_status_t
provider_name_bridge_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  ProviderNameNetworkContext *ctx = reinterpret_cast<ProviderNameNetworkContext *>(handle);

  // Add any clean-up actions (if not already handled in the `ProviderNameNetworkContext` destructor)  
  
  delete ctx;
  return CUDAQ_OK;
}

static cudaq_status_t
provider_name_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t handle,
    cudaq_realtime_transport_context_t context_type, void *out_context) {

  if (!handle || !out_ringbuffer)
    return CUDAQ_ERR_INVALID_ARG;
  ProviderNameNetworkContext *ctx = reinterpret_cast<ProviderNameNetworkContext *>(handle);

  // Populate the transport context
  if (context_type == RING_BUFFER) {
    out_ringbuffer->rx_flags = ...;
    ...
  }

  return CUDAQ_OK;
}

static cudaq_status_t
provider_name_bridge_connect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  ProviderNameNetworkContext *ctx = reinterpret_cast<ProviderNameNetworkContext *>(handle);

  // Perform any custom actions to initiate a connection: open data stream/socket, etc.

  return CUDAQ_OK;
}

static cudaq_status_t
provider_name_bridge_launch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  ProviderNameNetworkContext *ctx = reinterpret_cast<ProviderNameNetworkContext *>(handle);
  // Launch thread/CUDA kernels/etc. to monitor the networking traffic   

  return CUDAQ_OK;
}

static cudaq_status_t
provider_name_bridge_disconnect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  ProviderNameNetworkContext *ctx = reinterpret_cast<ProviderNameNetworkContext *>(handle);
  
  // Terminate the connection, e.g., stop any network monitoring actions, closing sockets/streams, etc. 
  return CUDAQ_OK;
}


// Add an entry point hook to retrieve the networking interface implementation
cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t cudaq_provider_name_bridge_interface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      provider_name_bridge_create,
      provider_name_bridge_destroy,
      provider_name_bridge_get_transport_context,
      provider_name_bridge_connect,
      provider_name_bridge_launch,
      provider_name_bridge_disconnect,
  };
  return &cudaq_provider_name_bridge_interface;
}
}

```

A sample of a `CMakeLists.txt` configuration is also provided here for reference.

```cmake
find_package(<your_networking_library> REQUIRED)

# Create the networking interface wrapper
add_library(cudaq-realtime-bridge-provider-name SHARED provider_name_bridge_impl.cpp)

target_include_directories(cudaq-realtime-bridge-provider-name
    PRIVATE
      ${CUDAQ_REALTIME_INCLUDE_DIR})

target_link_libraries(cudaq-realtime-bridge-provider-name      
    PRIVATE
      cudaq-realtime
      <your_networking_library target>)
```
