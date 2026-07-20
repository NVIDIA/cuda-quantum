# LLVM patches

This folder contains changes that deviate from the content of the LLVM
repository. This file contains some comments about why these patches were
created.

## Building `compiler-rt`

Building `compiler-rt` as part of the runtimes, rather than as a project, is
necessary to properly link all runtimes against each other. However, when doing
so, some of the libraries that are successfully installed when building it as a
project are not built/installed when building it as part of the runtimes. The
issue seems to be that the `CAN_TARGET_*` variables are only computed once,
which causes them to be incorrectly set to `FALSE` when building `compiler-rt`
as part of the runtimes. The patch in `CompilerRTUtils.cmake.diff` forces the
variables to be rechecked during the build. This patch may no longer be
necessary after updating to LLVM 19+.
