# CUDA-Q EDEM

**CUDA-Q EDEM is provided for early experimental scenarios. Its APIs are
evolving quickly and may change incompatibly during v0.x development.**

This component provides caller-owned sparse GF(2) matrices, typed linear maps,
and open EDEM values.
It constructs, validates, packs, reads, and evaluates models independently of
CUDA-Q Logical. The public interface is C11. The private implementation uses
C++20 without exceptions, RTTI, or a C++ runtime dependency. Setup operations
use per-call allocation callbacks, with `malloc`/`free` as the default.
Evaluation uses caller vectors and never allocates.

The API, ABI, and packed format are v0.x and may change between development
releases.

GF2 sources live in `math/gf2`. EDEM owns their combined installation and CMake
package. GF2 has no separate package or release version.

## Standalone build

Building the libraries requires CMake 3.23 or newer, a C11 compiler, and a C++20
compiler. C applications can consume the installed package without enabling C++.

From the CUDA-Q repository root:

```sh
cmake -S edem -B build-edem -DBUILD_SHARED_LIBS=ON
cmake --build build-edem
ctest --test-dir build-edem --output-on-failure
cmake --install build-edem --prefix /path/to/install
```

Set `BUILD_SHARED_LIBS=OFF` for static libraries and
`CUDAQ_EDEM_BUILD_TESTS=OFF` to omit tests. The standalone build brings GF2 in
from the sibling source directory.
GF2 can also build and test independently, without configuring EDEM:

```sh
cmake -S math/gf2 -B build-gf2 -DBUILD_SHARED_LIBS=ON
cmake --build build-gf2
ctest --test-dir build-gf2 --output-on-failure
```

Use `CUDAQ_GF2_BUILD_TESTS=OFF` to omit its sparse-matrix and typed-map tests.
When included by EDEM, their default follows `CUDAQ_EDEM_BUILD_TESTS`.
GF2's standalone build does not create another distribution or install package.

## Python bindings and wheel

Python bindings use nanobind and require Python 3.11 or newer. Enable them in a
native build with `CUDAQ_EDEM_BUILD_PYTHON=ON` after
installing `nanobind>=2.12,<3` into the chosen Python environment. Set
`Python_EXECUTABLE` to select an interpreter. Native builds default to
`CUDAQ_EDEM_BUILD_PYTHON=OFF`.

The `cudaq-edem` wheel includes the `cudaq_edem` Python package, both native
libraries, headers, and CMake exports. The Python module owns aligned buffers
and delegates construction, validation, and evaluation to the C interface.
Python requires no NumPy or Stim dependency. Dependent packages can link to the
standalone wheel's native libraries and use its ownership protocol to exchange
packed EDEM values without bundling another copy of the libraries.

From the repository root, build the source distribution and its wheel:

```sh
python3 -m pip install "build[virtualenv]"
python3 edem/scripts/build_wheel.py --outdir dist/edem
```

The script stages the EDEM and GF2 sources with `pyproject.toml.edem` as the
standard `pyproject.toml`. It leaves the repository's main wheel configuration
unchanged. Scikit-build-core builds the standalone EDEM project and packages
its CMake installation. The wheel version comes from `CUDAQ_EDEM_VERSION`.

Install the resulting wheel with pip. A downstream CMake build can locate its
package configuration through the same Python environment:

```sh
cmake -S /path/to/consumer -B build-consumer \
  -Dcudaq-edem_DIR="$(python3 -c 'import cudaq_edem; print(cudaq_edem.get_cmake_dir())')"
```

Wheels use CPython and platform compatibility tags. CI produces wheel and
source-distribution artifacts; public release publishing is a separate step.

## CUDA-Q build

Append `edem` to `CUDAQ_ENABLE_PROJECTS`, for example:

```sh
cmake -S . -B build -DCUDAQ_ENABLE_PROJECTS="cudaq;runtime;edem"
```

The top-level CUDA-Q configuration retains its usual toolchain and dependency
requirements. EDEM tests default to off in a parent build; enable them with
`CUDAQ_EDEM_BUILD_TESTS=ON`.

## Installed package

Use the same targets in source-tree and installed builds:

```cmake
find_package(cudaq-edem CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE cudaq::edem)
```

Add the installation prefix to `CMAKE_PREFIX_PATH`. The package also exports
`cudaq::gf2`; linking EDEM supplies that dependency transitively. Both targets
share the EDEM distribution version.

Each native shared library exports an explicit list of C symbols. On ELF,
the EDEM and GF2 lists use the `CUDAQ_EDEM_0.1` and `CUDAQ_GF2_0.1` ABI version
nodes. macOS derives its export lists from the same files. Build checks verify
the complete exported surface and, on ELF, each symbol's version node.

Headers, static archives, and package configuration files belong to the
development install component; shared libraries belong to the runtime component.
Parent CUDA-Q builds supply their component names. Standalone builds use
`Development` and `Runtime`. Python bindings and their third-party notices
belong to `EDEMPython`.

## Packaging checks

The tests compile and link C and C++ consumers using the build-tree targets,
then install, relocate, and consume the package from a separate build. They
also exercise a parent that defines GF2 before adding EDEM and installs both
libraries through one `Core` component, and a standalone build with custom
library and header installation directories.

Component checks exhaust the responses of an open syndrome-round fixture,
including incoming boundary errors and faults that survive at the output.
They also check sparse construction, exact fault groups, canonical packing,
relocation, invalid content, large fault distributions, and canonical packed
bytes. GF2's own tests under `math/gf2/test` check sparse matrices, typed maps,
multiwire bundles and the trailing directory without linking EDEM. Native build
checks permit `malloc`/`free` only in GF2's default allocator and reject other
allocator imports, C++ runtime dependencies, and exports outside the C API.
Failure-injection and bounded-arena tests check callback routing, cleanup and
result publication. Relocated package checks also build a consumer
that enables only C, for both shared and static libraries.

When Python bindings are enabled, CTest also checks the build-tree import.
`edem/test/check_wheel.py` installs a wheel into a fresh virtual environment,
imports its extension, loads both native libraries, relocates the installed
package, and compiles C and C++ consumers against its CMake exports. Run it with
the path to the wheel as its argument.

## Native interface and memory

The C headers are the native user surface. `cudaq/gf2/core.h` provides sparse
matrices; `cudaq/gf2/map.h` provides typed linear maps and their packed storage.
`cudaq/edem/core.h` adds semantic construction, presentation and distribution
checks, endpoint translation, and access to fault distributions.

Construction accepts blocks keyed by semantic endpoints such as output syndrome
of code block 2. Producers supply code presentations and coefficients; EDEM
builds the type table and bundles. It returns a checked `cudaq_edem_view`
containing a GF2 map in its `.map` member. EDEM getters accept this distinct
checked type. Use `&value.map` with the GF2 map API for bundles, widths,
offsets, blocks, vector sizes and evaluation.
There is no duplicate EDEM map-getter family or apply function.
`cudaq_edem_resolve_endpoint` and `cudaq_edem_describe_wire` translate vocabulary
between semantic endpoints and generic bundle/wire coordinates.

Construction is one call. The library computes storage requirements internally,
allocates temporary scratch and an exact packed result, and returns a
caller-held `cudaq_gf2_owned_buffer` with a checked, borrowed view:

```c
cudaq_gf2_owned_buffer owner = {0};
cudaq_edem_view value;
cudaq_edem_status status = cudaq_edem_construct(&description, NULL, &owner, &value);
if (status == CUDAQ_EDEM_OK) {
  /* Query or evaluate value.map while owner.data remains alive and immutable. */
  cudaq_gf2_owned_buffer_release(&owner);
}
```

`cudaq_gf2_map_construct` provides the same ownership model for generic maps.
`cudaq/gf2/memory.h` defines GF2 allocation and ownership types shared by both
libraries. The owner stores the packed pointer,
size, deallocation callback and callback context directly; it needs no hidden
header, allocated descriptor, reference count or library handle. It is a unique
owner: move it by copying the record and clearing the source. Release clears the
record and invokes the captured callback once. The callback, its code and its
context must remain alive until release. Moving an owner does not move its bytes
or invalidate existing borrowed views.

Each setup call accepts an optional `cudaq_gf2_memory` with separate `scratch` and
`result` allocator slots. A null policy uses `cudaq_gf2_default_allocator()` for
both. An explicit used slot must provide both callbacks; missing callbacks are
invalid, and there is no fallback to the system allocator. The same provider
may fill both slots. Scratch is released before returning and is never retained;
the result captures only its deallocator and context, not the policy object.
Construct requires an empty result owner. Failures leave output owners and
views unchanged and release all temporary or unpublished allocations.

For bounded operation, `cudaq/gf2/bump_allocator.h` supplies a C11 header-only
provider over caller storage. Its deallocator does nothing. Exhaustion returns
`ALLOCATION_FAILED`, and failed operations do not rewind the arena cursor.
Use separate scratch and result arenas when scratch should be reset after each
call; reset an arena only when none of its allocations remain live. There is no
public exact-size preflight for packed map construction. The low-level CSC
matrix builder retains its caller-buffer interface.

To load an EDEM, call `cudaq_edem_read(buffer, size, memory, &value)`. It validates
GF2 content and EDEM semantics before publishing the checked view. Reading is
zero-copy: the view borrows the supplied aligned, immutable bytes. Read,
description validation and semantic checking use only the scratch slot.

Advanced callers with an already validated GF2 map can use `cudaq_edem_check`
to obtain a checked EDEM view. `cudaq_edem_schema()` declares recognized
sections for an explicit GF2 read; recognition alone does not establish EDEM
semantics. Ordinary loading
uses the EDEM reader so the semantic check cannot accidentally be skipped.
Checked views borrow immutable backing bytes; fabricating a struct or mutating
its backing bytes violates the token's initialization and lifetime rules.

Operations retain no state, caches or allocator policy between calls. Getters
and evaluation allocate nothing. The default provider is the only code in the
native libraries that imports `malloc` and `free`.

Canonical ordering sorts index arrays. Matrix interning uses fingerprints with
full equality checks. Fault coverage uses one bit per indicator. The semantic
constructor also stages its lowered map description and canonical EDEM section
payloads in temporary scratch; its storage is linear in the input size.
Packed values remain independent of this setup storage.

## Component boundary

EDEM supports the map
`(F, V_in, S_in, L_in) -> (D, O, Success, V_out, S_out, L_out)`.
Each code-block bundle has three wires, `(V,S,L)`, with widths `(g,g,2k)`.
Input and output code-block counts are independent.

A type is a nonempty UTF-8 name, a role and a wire count. Names compare by exact
UTF-8 bytes, without normalization. Types are nominal, with no subtyping.
Free bundles do not participate in wiring. Interface bundle compatibility
requires matching type names, roles, wire counts and widths. Local type IDs are
indices, not cross-value identities. The API constructs, validates and applies
individual maps; it does not provide composition operations.

EDEM uses four reserved, single-wire free types: `edem.fault`,
`edem.detector`, `edem.observable`, and `edem.success`. It retains all four in
its type table, including unused types. A presentation named `P` creates the
three-wire interface type `edem.code/P`. The separate, name-sorted presentation
table records `g` and `k` explicitly. Every bundle using that code type must
have the recorded `(g,g,2k)` widths. Presentation names identify generator
ordering and logical representatives; producers supply their code algebra and
coefficients.

GF2 owns type/bundle layout, block validation, canonical storage, matrix
interning, packing, reading and apply. EDEM checks its vocabulary, permitted
input/output roles, presentation consistency, and structural zeros from V_in
to O, S_out and L_out. Fault inputs have one distribution per bundle. Exact
independent and mutually exclusive groups cover original indicators once;
separate bundles retain separate laws. Equal matrix blocks never identify
faults. Original indicators are used directly as fault coordinates.

The component consumes caller-provided coefficients and does not perform
circuit compilation. The open-round fixtures exercise this interface with
explicit coefficients, including boundary inputs and faults that survive at
the output.

## Fault distributions

Each probability must be finite and within `[0, 1]`. An exclusive group's
probabilities are accumulated in binary64 and the total must not exceed one.
This strict check can reject a mathematically normalized group when rounding
makes its computed total greater than one. Construction and packed-value
validation use the same check. Independent probabilities have no group-total
bound.

Packing preserves the supplied probability bits except for normalization of
negative zero to positive zero. It performs no probability renormalization.
Packed-value validation requires positive zero, preserving canonical bytes.

## Python construction

This open boundary-transfer example preserves the incoming coordinates and
adds one late data fault to the outgoing syndrome:

```python
from cudaq_edem import (
    Block, CodePresentation, EDEM, Endpoint, FaultDistribution, PortKind,
    SparseMatrix,
)

identity = SparseMatrix(2, 2, [(0, 0), (1, 1)])
model = EDEM(
    codes=[CodePresentation("repetition-3", generators=2, logical_qubits=1)],
    inputs=[0], outputs=[0],
    faults=[FaultDistribution.independent([0.01])],
    blocks=[
        Block(Endpoint(kind), Endpoint(kind), identity)
        for kind in (PortKind.VIRTUAL, PortKind.SYNDROME, PortKind.LOGICAL)
    ] + [
        Block(Endpoint(PortKind.SYNDROME), Endpoint(PortKind.FAULT),
              SparseMatrix(2, 1, [(0, 0)])),
    ],
)
assert model.apply([1, 0, 0, 0, 0, 0, 0]) == [0, 0, 1, 0, 0, 0]
assert EDEM.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
```

For constructor-created models, input vector order is F partitions followed by
`(V,S,L)` for each input block; output order is D, O, Success partitions followed
by `(V,S,L)` for each output block. Values loaded from packed bytes retain the
GF2 map's bundle order. `model.port(endpoint, output=False)` returns an input
port's width and offset; use `output=True` for output ports.

Properties return independent metadata/matrix copies. Reconstructing a loaded
value from these properties uses the constructor's bundle order. Use
`to_bytes()` and `from_bytes()` to preserve a loaded layout. Python loading
copies bytes once into aligned owner storage and validates them before
publication.

Native producers can instead return a single-use capsule following
`cudaq/edem/python.h`, containing a stable `cudaq_gf2_owned_buffer` record:

```python
model = EDEM.adopt(producer_capsule)
```

Adoption validates the bytes before consuming ownership and keeps their original
address. It retains the producer capsule so callback code and contexts can stay
alive through final release. Reusing a consumed capsule is an error; failed
validation leaves it available to its producer. Native producers can transfer
constructed results without intermediate strings or byte copies.
`from_bytes` remains the convenience entry point for serialized input.

## Packed format

The v0.x storage schemas are declared in `cudaq/gf2/format.h` and
`cudaq/edem/format.h`. These schemas are experimental. Numeric ABI and
format tags identify concrete development revisions, not major releases.
The distribution remains v0.x.
Multibyte integers and IEEE binary64 probabilities are little-endian; indices
are 32-bit, and counts and offsets are 64-bit.

The common 64-byte header contains magic `CQGF2\0\0\0`, format revision, value
kind, total size, checked alignment and index-width constants, directory offset,
entry count, directory byte length, and a zero reserved field. The implemented
kind is a linear map. An EDEM is that map with presentation and distribution
sections, in the same buffer.

The directory is at the end. Each 32-byte entry records a unique section ID,
required/optional flags, element count, payload byte length and payload offset.
Entries sort by ID. Required IDs are below `0x80000000`; optional IDs are at or
above it, and flags must agree. Core map IDs are reserved below `0x10000`.
Extension schemas start at `0x10000`. The five required map sections are:

| ID | Payload |
|---|---|
| 1 | Types and their UTF-8 names |
| 2 | Input bundles, wire widths and vector offsets |
| 3 | Output bundles, wire widths and vector offsets |
| 4 | Sparse block bindings |
| 5 | Interned CSC matrices |
| `0x10000` | EDEM presentation table: type ID, g, k |
| `0x10001` | EDEM distributions associated with input fault bundles |

Required payloads follow the header, optional payloads follow those, and the
directory comes last. Nonempty payloads occur in directory order at aligned,
non-overlapping offsets. Empty payloads have offset zero. There are no unused
ranges other than alignment padding; all padding and reserved fields are zero.
The directory's byte extent ends exactly at total size.

Alignment is the fixed format constant of eight bytes. It is checked in the
header and is not a packing or sizing input. Python owners need only meet that
constant. Nested arrays use schema-defined natural alignment: eight for
64-bit arrays and records, four for 32-bit arrays, and one for strings. Every
reference within a section is relative to that section's start. Changing the
alignment requires a format revision.

Type names sort by UTF-8 bytes and local IDs are remapped. Bundle order is
preserved. Blocks sort by output bundle/wire, then input bundle/wire. Duplicate
endpoint pairs are rejected, zero blocks are omitted, and equal matrices are
stored once in first-use order. Packed wire offsets are checked against widths.
Fault groups sort by kind and first indicator, with sorted members and positive
zero probabilities. EDEM checks their exact coverage with caller scratch.

Appending optional metadata preserves every required payload's bytes and
offset: copy the required region, emit optional payloads in ID order, write a
new trailing directory and update header layout fields. Unknown required
sections are unsupported; unknown optional sections can be skipped after their
bounds and padding are checked. Optional metadata does not change the map's
meaning.
