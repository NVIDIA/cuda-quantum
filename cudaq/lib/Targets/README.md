# CUDA-Q Target Descriptions

Each subdirectory holds the declarative description of one in-tree CUDA-Q
target, as `<name>/<name>.yml`. These files are the source of truth for target
configuration.

At build time, `cudaq-target-db-gen` reads them and emits the
`CUDAQTargetDatabase` table. Generation is invoked from the top-level
`CMakeLists.txt`, after any `CUDAQ_EXTERNAL_PROJECTS` registrations made with
`add_target_config()` have been processed.
