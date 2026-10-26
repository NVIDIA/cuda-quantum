# Target configuration

This folder contains the CUDA-Q target definitions and configuration system. The
code that relates to the CUDA-Q target system is roughly comprised of:

- Public headers in `cudaq/include/cudaq/Target/`, which define the data types
  required to specify CUDA-Q targets. This is the public C++ API used by the
  rest of CUDA-Q.
- YAML schemas in `./Yaml/`, which document and implement the YAML file format
  supported by CUDA-Q to define, store and load target configurations using text
  files. We use `reflect-cpp` to parse and validate a YAML file against our
  schema. The parsed YAML is then converted to the public C++ API types.
- A list of 'built-in' target definitions in `../Targets` that get bundled and
  shipped as part of any CUDA-Q installation, using `tools/cudaq-target-db-gen`.
- The other files are `.cpp` implementation files, private headers, `cmake`
  files, tests etc.

Target definitions are YAML files. yaml-cpp parses the YAML, reflect-cpp maps
the selected versioned schema in `Yaml/` to C++ types, and that schema is
converted to the canonical `cudaq::config::TargetConfig`. Built-in definitions
are compiled into the target database; the catalog can also load plugin target
definitions.

## API and schema compatibility

The installed headers in `cudaq/include/cudaq/Target/` are the C++ API used by
the rest of CUDA-Q. Breaking ABI changes may break pre-compiled target databases
and thus require a bump of `CUDAQ_TARGET_DB_ABI_VERSION` in `TargetDatabase.h`
at every CUDA-Q release.

YAML schemas are independently versioned by their top-level `version` field;
un-versioned files are permanently treated as v1. Changes to a published schema
must remain backward compatible. Otherwise, add a new schema version under
`Yaml/`, add it to `TargetConfigSchema`, and add the required logic to convert
it to `TargetConfig`. Pre-release tags (e.g. `2-alpha`) on the other hand may
change or be removed without backward compatibility until they are stabilized
under a plain version number.

## Dependencies

This component requires CMake 3.30 and C++20. Its private dependencies are
yaml-cpp 0.9.0 and reflect-cpp 0.25.0. CMake `FetchContent` downloads the exact
supported revisions during configuration, so no separate installation is needed.
Tests additionally need `llvm-lit` and `FileCheck`.

## Build

This component is typically built in-tree as part of the CUDA-Q build. It can
also be built standalone:

```console
cmake -S cudaq/lib/Target -B build/target -G Ninja -DCUDAQ_BUILD_TESTS=ON
cmake --build build/target --target check-target-catalog
```
