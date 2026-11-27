# Third-party license texts

This directory contains the full text of licenses that apply to third-party
libraries redistributed with CUDA-Q in binary form. CUDA-Q itself is licensed
under the Apache License 2.0 (see the `LICENSE` file in the repository root);
the files here apply only to the components listed below. See the `NOTICE` file
for the corresponding copyright notices and source-code locations.

| File                  | License                                 | Applies to                                                                                                     |
| --------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `LICENSE.LGPLv3`      | GNU Lesser General Public License v3    | GMP (`libgmp`), MPFR (`libmpfr`)                                                                               |
| `LICENSE.GPLv3`       | GNU General Public License v3           | Incorporated by reference by the LGPL v3 (the LGPL v3 is a set of additional permissions on top of the GPL v3) |
| `LICENSE.nanobind`    | BSD 3-Clause License                    | nanobind runtime, including the cudaq-edem Python extension                                                   |
| `LICENSE.robin-map`   | MIT License                             | robin-map, header-only dependency bundled by nanobind                                                          |
| `LICENSE.reflect-cpp` | MIT License                             | reflect-cpp, statically linked into `libCUDAQTargetCatalog`                                                    |
| `LICENSE.yaml-cpp`    | MIT License                             | yaml-cpp, statically linked into `libCUDAQTargetCatalog`                                                       |
| `LICENSE.ctre`        | Apache License 2.0 with LLVM Exceptions | `CTRE`, header-only library bundled by reflect-cpp                                                             |
| `LICENSE.enchantum`   | MIT License                             | `enchantum`, header-only library bundled by reflect-cpp                                                        |
| `LICENSE.openmpi`     | BSD 3-Clause License                    | OpenMPI (container images only)                                                                                |
| `LICENSE.ucx`         | BSD 3-Clause License                    | UCX (container images only)                                                                                    |
| `LICENSE.ucc`         | BSD 3-Clause License                    | UCC (container images only)                                                                                    |
| `LICENSE.pmix`        | BSD 3-Clause License                    | PMIx (container images only)                                                                                   |
| `LICENSE.slurm-pmi2`  | Argonne National Laboratory notice      | Slurm PMI2 client library (container images only)                                                              |
| `LICENSE.gdrcopy`     | MIT License                             | GDRCopy user-space library (container images only)                                                             |
| `LICENSE.uv-MIT`      | MIT License                             | uv (container images only), dual licensed with `LICENSE.uv-Apache-2.0`                                         |
| `LICENSE.uv-Apache-2.0` | Apache License 2.0                    | uv (container images only)                                                                                     |

The container images also ship MPI-related software that is built from
unmodified upstream source (OpenMPI, UCX, UCC, PMIx, the Slurm PMI2 library,
MUNGE and GDRCopy) and the `uv` tool. MUNGE is covered by `LICENSE.GPLv3` and
`LICENSE.LGPLv3`. The CUDA-Q package sources image (`cuda-quantum-src`)
contains the source of all of them.

GMP and MPFR are unmodified, dynamically linked shared libraries used by the
Clifford+T rotation synthesis library (`cudaq-synth`). They can be replaced with
compatible versions by substituting the shared library files; see the "Dynamic
linking to GMP and MPFR" section of the installation documentation for details.

The cudaq-edem Python extension statically links the nanobind runtime, which uses
robin-map headers. Its wheel includes both license texts from this directory.
Both native libraries use CUDA-Q's Apache License 2.0.

`reflect-cpp` and `yaml-cpp` are fetched from immutable pinned commits and built
from source at configure time (`FetchContent`, see
`cudaq/lib/Target/cmake/FetchDependencies.cmake`), and are statically linked
into the CUDAQ target catalog library as private dependencies. `reflect-cpp`
bundles the header-only `CTRE` and `enchantum` libraries under
`include/rfl/thirdparty`; `LICENSE.ctre` reproduces the Apache License 2.0 text
with LLVM Exceptions embedded in the bundled `ctre.hpp` header. reflect-cpp also
bundles the `yyjson` header, but that code is not compiled into CUDA-Q binaries
because reflect-cpp's JSON support is disabled; it is therefore not listed
above.
