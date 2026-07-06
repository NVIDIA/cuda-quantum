# Third-party license texts

This directory contains the full text of licenses that apply to third-party
libraries redistributed with CUDA-Q in binary form. CUDA-Q itself is licensed
under the Apache License 2.0 (see the `LICENSE` file in the repository root);
the files here apply only to the components listed below. See the `NOTICE`
file for the corresponding copyright notices and source-code locations.

| File             | License                              | Applies to                                                                 |
| ---------------- | ------------------------------------ | -------------------------------------------------------------------------- |
| `LICENSE.LGPLv3` | GNU Lesser General Public License v3 | GMP (`libgmp`), MPFR (`libmpfr`)                                            |
| `LICENSE.GPLv3`  | GNU General Public License v3        | Incorporated by reference by the LGPL v3 (the LGPL v3 is a set of additional permissions on top of the GPL v3) |

GMP and MPFR are unmodified, dynamically linked shared libraries used by the
Clifford+T rotation synthesis library (`cudaq-synth`). They can be replaced
with compatible versions by substituting the shared library files; see the
"Dynamic linking to GMP and MPFR" section of the installation documentation
for details.
