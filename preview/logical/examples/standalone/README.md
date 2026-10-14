# Standalone CUDA-Q Logical examples

These examples use `cudaq.logical` directly. They do not define
`@cudaq.kernel` functions and do not call `cudaq.estimate`. Instead, they make
the logical authoring, placement, QEC selection, physical lowering, scheduling,
and direct estimation APIs visible.

The sequence progresses from portable logical programs through a complete
physical schedule. The final Gidney--Ekerå lookup-addition example combines the
steps in a small advanced workflow that remains suitable for routine testing.
