# Device Code for the Unified GPU Dispatcher

A DEVICE + UNIFIED dispatcher runs one kernel that receives a request, calls
its handler and sends the response.  Three binaries contribute to it:

| Code | Comes from |
|------|------------|
| The unified dispatch kernel | `libcudaq-realtime.so`, embedded as `LTO-IR` |
| The transport's device hooks (`cudaq_dev_enter_kernel`, `cudaq_dev_rx_poll`, `cudaq_dev_tx_publish`) | The bridge provider library |
| The `DEVICE_CALL` handlers | The application |

A device function can only call, or take the address of, code in its own CUDA
module, so the three have to end up in one module.  The dispatcher links them
at run time with `nvJitLink`, for the GPU it runs on, when the application looks
up its first handler with `cudaq_dispatcher_get_device_handler`.  This page is
the contract the bridge and the application follow so that this link works.

## Build with `-rdc` and hand the code over

Build the hooks, or the handlers, as relocatable device code: `nvcc -rdc=true`,
or `CUDA_SEPARABLE_COMPILATION ON` in CMake.  Every such binary carries its
device code in its `__nv_relfatbin` section, and
`cudaq_device_code_of_this_binary` (header-only C,
`cudaq/realtime/device_code/device_code_reader.h`) returns it:

```cpp
#include "cudaq/realtime/device_code/device_code_reader.h"

cudaq_device_image_t image;
cudaq_device_code_of_this_binary(&image);
```

It returns the code of the binary the call is compiled into, so a bridge and
an application calling it each get their own.  A binary without `-rdc` code
gets `CUDAQ_ERR_UNSUPPORTED`.  The image is the whole section, one `fatbin`
per translation unit back to back; the dispatcher splits it when it links.
The bytes stay valid for the life of the process.

- **A bridge** returns the image from `get_gpu_dataplane`, together with the
  device-accessible context its hooks read (`cudaq_gpu_dataplane_t`).
- **An application** does not call it: `cudaq_dispatcher_get_device_handler`
  (`cudaq/realtime/daemon/dispatcher/cudaq_realtime.h`) is compiled into the caller's
  binary and makes the call there.  Look up handlers from the binary that
  holds them, which may be a shared library rather than the executable.

## Bridges link no realtime library

`libcudaq-realtime.so` loads bridge providers, never the reverse.  A provider
includes the realtime headers and uses only their header-only code
(`device_code_reader.h` included); it does not link `cudaq-realtime`.

## `nvJitLink`

Building `libcudaq-realtime` needs the `nvJitLink` header and library, from the
`nvJitLink-dev` package (`libnvjitlink-dev`, part of a full CUDA toolkit).  No
public header includes `nvJitLink.h`, so applications and bridges build
without it.  At run time `libcudaq-realtime.so` loads `libnvJitLink.so.<major>`,
which every CUDA runtime install provides.  That `nvJitLink` must be at least as
new as every toolkit that produced the linked code: the one that built
`libcudaq-realtime`, the bridge's and the application's.

## Exporting handlers

CUDA looks up kernels and globals by name, never device functions.  Export a
handler with the macro from `device_code.h`, then ask the dispatcher for its
address in the linked module, with `cudaq_dispatcher_get_device_handler` from
`cudaq/realtime/daemon/dispatcher/cudaq_realtime.h`:

```cpp
// handlers.cu, built with -rdc and compiled into the executable
#include "cudaq/realtime/device_code/device_code.h"

namespace {
__device__ int my_handler(const void *rx_slot, void *tx_slot, size_t slot_size);
}
CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(my_handler)
```

```cpp
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

void *handler;
cudaq_dispatcher_get_device_handler(dispatcher, "my_handler", &handler);
```

- **The first lookup links the module,** with the device code of the binary
  the lookup is compiled into; later lookups reuse it.  After a failed link
  the dispatcher has no module, and the next lookup tries again.
- **All handlers of one dispatcher come from one binary,** and are looked up
  before `cudaq_dispatcher_start`.  A lookup from another binary, or after
  `start` has linked without one, is refused.
- **Every `DEVICE_CALL` entry** must hold an address from
  `cudaq_dispatcher_get_device_handler`.  An address taken in the
  application's own module, for example by an init kernel, points at code the
  linked module does not hold, so `cudaq_dispatcher_start` refuses it.
- **Compile the export into the executable**, not into a static library:
  nothing on the host references it, so the linker would leave an archive
  member holding it out of the binary.  A library can still carry it as an
  INTERFACE source (`target_sources(lib INTERFACE handlers.cu)`), which each
  executable linking the library compiles itself.
- **Handlers have no host-set state yet.**  A `__device__` global the
  application sets from the host lives in its own module, which is not the
  linked module the kernel runs in, so the handler would not see the value.
- **The export macro uses the braced `extern "C"` form**,
  `extern "C" { __device__ T name = value; }`.  Under `-rdc`,
  `extern "C" __device__ T name = value;` only declares `name`, and the host
  link fails with "undefined reference".

## Keep other device symbols private

The link takes all of a binary's `-rdc` code, not only the hooks or handlers.
Two strong definitions of one name fail it with "symbol multiply defined",
naming the symbol.  Keep device symbols that are not part of the contract
`static` or in an anonymous namespace, including those of headers included by
more than one translation unit.  A header that defines non-static device
functions, such as HSB's `gpu_roce_transceiver.cuh`, puts them in every
binary that compiles it with `-rdc`.

The rest of a binary's `-rdc` code is harmless: other kernels, the device
runtime and device-side `cudaGraphLaunch` link and load alongside.

## Optional `LTO`

Plain `-rdc` code links as is, and the hooks stay ordinary calls from the
kernel.  Built with `LTO-IR`, they are inlined into it, which takes the calls
off the per-message path; the in-tree `gpu_roce` bridge does this.  Any of:

- CMake: `INTERPROCEDURAL_OPTIMIZATION ON` on a separable target, which emits
  `code=[compute_XX,lto_XX]`;
- per source, which avoids host `LTO`:
  `--generate-code=arch=compute_XX,code=lto_XX`;
- nvcc directly: `-rdc=true -dlto -arch=sm_XX`.

A bare `-dlto` next to CMake's own `--generate-code` options is rejected by
nvcc.

## Architectures

No architecture list is needed for the link to succeed.  Each image must hold
SASS for the running GPU, or `PTX` or `LTO-IR` for it or an older architecture.
The dispatcher always links with `-lto` for the generic `sm_XY` of the running
GPU.

`__CUDA_ARCH__` is fixed when an image is compiled, not when it is linked:
`LTO-IR` built for `compute_80` keeps its `__CUDA_ARCH__ < 900` branches on any
GPU.  Code that branches on the architecture should ship an entry for each
architecture it cares about; `nvJitLink` takes the newest one that fits the
running GPU.  DOCA's device headers do, for example.

## Toolkit versions

Images must come from a CUDA toolkit no newer than the `nvJitLink` the
application runs with.  Newer code fails in every form, `LTO-IR`, SASS and `PTX`,
and the dispatcher's error then names the `nvJitLink` version.  That
`nvJitLink` must also support the running GPU's architecture: for `sm_100`,
12.8 or newer, even for a build made with CUDA 12.6.

## Wiring sequence

1. Create the bridge, then the dispatcher with `dispatch_path = DEVICE` and
   `kernel_type = UNIFIED`.
2. `cudaq_dispatcher_set_bridge`: takes the bridge's GPU data plane.
3. `cudaq_dispatcher_get_device_handler` (`cudaq_realtime.h`) for each handler.
   The first call links and loads the module.
4. Fill the function table with those addresses, then
   `cudaq_dispatcher_set_function_table` and `cudaq_dispatcher_set_control`.
5. `cudaq_dispatcher_start`.

`realtime/unittests/test_server/cudaq_realtime_test_server.cpp`
(`build_device_table`) and
`realtime/unittests/bridge_interface/gpu_loopback/` are complete examples.
