# cudaq C++ API Code Development Guidelines

**Scope:** This document defines conventions for C++ API boundaries, naming, and
header organization for the `cudaq` project and the `cudaq` compiler
toolchain.

## 1. Goals

1. Make it obvious which APIs are supported for **end users** versus **internal
development**.
2. Preserve long-term stability for the **User API** (backward compatibility +
 deprecation schedule).
3. Allow **Hardware vendors** and **core developers** to access internal
interfaces without accidentally turning them into user-facing commitments.
4. Ensure `cudaq` can discover headers needed to compile user code (including
 internal headers that user headers must include).

---

## 2. Developer Categories (and what they may use)

### 2.1 Users (e.g, Quantum algorithm developers)

- Use only the **User API** shipped with the product.
- Compile with `cudaq` and link to shipped runtime libraries.
- Must not rely on internal headers/namespaces.

### 2.2 Library developers (e.g., `cudaqx`)

- Same restrictions as users: build on top of the **User API** only.
- May consume the API via ``nvq++`` or by importing public headers/libraries into
their own CMake project.

### 2.3 Hardware vendors and core developers

- May use:
  - **User API**
  - **Internal public module APIs**
  - **Module private APIs**
- Contribute changes to the codebase and/or add backends, modules, extensions.

---

## 3. API Layers and Rules

We define a three API layers as illustrated below:

1. **User API**
2. **Internal public module APIs**
3. **Internal private APIs**

```text
┌────────────────────────────────────────────────────────────────────┐
│ Level 1: User API                                                  │
├────────────────────────────────────────────────────────────────────┤
│ Audience:   Users, external libs (e.g., cudaqx)                    │
│ Headers:    "cudaq.h", "cudaq/<subsystem>/<header>.h"              │
│ Namespace:  cudaq::...                                             │
│             cudaq::detail  = explicitly NON-public                 │
│ Naming:     snake_case                                             │
└────────────────────────────────────────────────────────────────────┘
                          ▲
                          │ may include (transitively) when needed
                          │
┌────────────────────────────────────────────────────────────────────┐
│ Level 2: Internal Public Module APIs                               |
├────────────────────────────────────────────────────────────────────┤
│ Audience:   Hardware vendors + core developers                     │
│            (NOT for users / external libs to depend on)            │
│ Headers:    "cudaq_internals/<module>/<hdr>.h"  (or cudaq_dev/...) │
│ Namespace:  cudaq::<module>::...  (module lowercase)               │
│             cudaq::<module>::detail = NON-public                   │
│ Naming:     CamelCase (or consistent module convention)            │
└────────────────────────────────────────────────────────────────────┘
                          ▲
                          │ internal-only use
                          │
┌────────────────────────────────────────────────────────────────────┐
│ Level 3: Internal Private APIs                                     │
├────────────────────────────────────────────────────────────────────┤
│ Audience:   Module implementers only                               │
│ Headers:    module-local (e.g., <module>/src/, include-private/)   │
│ Namespace:  typically in cudaq::<module>::detail (recommended)     │
│ Naming:     unconstrained; keep consistent within module           │
└────────────────────────────────────────────────────────────────────┘
```

Each layer has rules for:

- include paths
- shipping requirements
- naming conventions
- namespace conventions
- compatibility expectations

### 3.1 User API (Public API)

#### 3.1.1 Definition

The **User API** is `cudaq` supported public interface. It is the only API that
users and external libraries (e.g., `cudaqx`) are allowed to depend on.

#### 3.1.2 Headers and includes

- Primary entry header:
  - `#include "cudaq.h"`
- Additional user headers:
  - `#include "cudaq/<subsystem>/<header>.h"`
  - Example: `#include "cudaq/algorithms/run.h"`
- File naming:
  - lowercase file names

#### 3.1.3 Namespaces

- All user-visible declarations live in `namespace cudaq { ... }`.
- Nested namespaces are considered public **except**:
  - `cudaq::detail`

Anything in `detail` is *explicitly non-public* and may change without notice.

#### 3.1.4 Naming style

- User API functions and objects use **snake_case**.
  - Examples: `sample_async`, `sample_results`

#### 3.1.5 Compatibility policy

- The User API must maintain backward compatibility when feasible.
- Breaking changes require:
  - a documented deprecation plan (deprecation schedule),
  - migration guidance,
  - and appropriate versioning/release notes.

---

### 3.2 Internal Public Module APIs (Exported, not user-supported)

#### 3.2.1 Definition

Internal public module APIs are:

- **not supported for end users**, but
- **public to internal developers** because they are exported as part of a
module interface (e.g., via CMake `PUBLIC` headers/libraries).

These APIs often must be shipped because **user headers may include them**, and
`nvq++` must be able to find them.

#### 3.2.2 Header location and include prefix (recommendation)

Internal public headers must **not** live under the `cudaq/` include root to
avoid confusion with the User API.

**Proposed convention:**

- `#include "cudaq_internals/<module_name>/<header_name>.h"`
or
- `#include "cudaq_dev/<module_name>/<header_name>.h"`
or
- `#include "cudaq_<module_name>/<header_name>.h"`

Rationale:

- clearly signals “not for users”
- stable and grep-friendly
- avoids collision with user include hierarchy

#### 3.2.3 Shipping and visibility

- These headers **are shipped** with the product if they are reachable from
shipped user headers or required by `nvq++` compilation.
- They **must be discoverable** by `nvq++` through configured include paths.

#### 3.2.4 Namespaces

- Declarations live under a module namespace nested in `cudaq`:
  - `namespace cudaq::<module_name> { ... }` where `<module_name>` is lowercase
    Examples: `cudaq::compiler`, `cudaq::cudaq_fmt`
- Nested namespaces follow the same visibility convention: they are public
except for the `detail` namespace.

#### 3.2.5 Naming style

- Preferred style for internal module APIs: **`CamelCase`**.
- If a specific module already has a strong existing convention, follow it
consistently; avoid introducing new naming styles within the same module.

#### 3.2.6 Compatibility expectations

- Internal public module APIs are **not** user-stable.
- They may evolve as needed, but changes should still be managed responsibly
because:
  - Hardware vendors and internal developers may depend on them,
  - user headers may indirectly rely on them.

---

### 3.3 Internal Private APIs (Module-only)

#### 3.3.1 Definition

Internal private APIs are implementation details private to a module. They must
not be consumed outside the module.

#### 3.3.2 Header location and visibility

- Private headers live in a physical location separate from public headers
(e.g., a module `<module>/src/` or `include-private/` tree).
- They should **not** be in `nvq++` default/public include search paths.

#### 3.3.3 Shipping

- Private headers are **not shipped** with the product.

#### 3.3.4 Include rules

- Private headers are included using **relative paths** from within the module.
- Exception: template-heavy code sometimes requires headers to be included
transitively.
  - If a shipped header must include a private header, the included declarations
  must be placed the private `detail` namespace.
  - This remains non-public and subject to change.

---

## 4. Practical Conventions and Examples

### 4.1 Include examples

**User code (supported):**

```cpp
#include "cudaq.h"
#include "cudaq/algorithms/run.h"
```

Internal developer code (allowed for hardware vendors/core developers):

```cpp
#include "cudaq_internals/compiler/lower.h"
#include "cudaq_internals/cudaq_fmt/Formatting.h"
```

or

```cpp
#include "cudaq_dev/compiler/lower.h"
#include "cudaq_dev/cudaq_fmt/Formatting.h"
```

Module private include (module-only):

```cpp
#include "LoweringPasses.h"   // relative to module source
```

### 4.2 Namespace examples

User API:

```cpp

namespace cudaq {
  void sample_async();

  namespace detail {
    // not public
  }
}
```

Internal module API:

```cpp
namespace cudaq::compiler {
  class PassPipeline;

  namespace detail {
    // not public
  }
}
```
