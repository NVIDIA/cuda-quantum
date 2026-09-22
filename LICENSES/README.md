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
| `LICENSE.reflect-cpp` | MIT License                             | reflect-cpp, statically linked into `libCUDAQTargetCatalog`                                                    |
| `LICENSE.yaml-cpp`    | MIT License                             | yaml-cpp, statically linked into `libCUDAQTargetCatalog`                                                       |
| `LICENSE.ctre`        | Apache License 2.0 with LLVM Exceptions | `CTRE`, header-only library bundled by reflect-cpp                                                             |
| `LICENSE.enchantum`   | MIT License                             | `enchantum`, header-only library bundled by reflect-cpp                                                        |

GMP and MPFR are unmodified, dynamically linked shared libraries used by the
Clifford+T rotation synthesis library (`cudaq-synth`). They can be replaced with
compatible versions by substituting the shared library files; see the "Dynamic
linking to GMP and MPFR" section of the installation documentation for details.

reflect-cpp and yaml-cpp are fetched from immutable pinned commits and built
from source at configure time (FetchContent, see
`cudaq/lib/Target/cmake/Dependencies.cmake`), and are statically linked into the
CUDAQ target catalog library as private dependencies. reflect-cpp bundles the
header-only `CTRE` and `enchantum` libraries under `include/rfl/thirdparty`;
`LICENSE.ctre` reproduces the Apache License 2.0 text with LLVM Exceptions
embedded in the bundled `ctre.hpp` header. reflect-cpp also bundles the `yyjson`
header, but that code is not compiled into CUDA-Q binaries because reflect-cpp's
JSON support is disabled; it is therefore not listed above.
