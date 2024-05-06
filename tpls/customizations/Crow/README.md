# Crow patches

This folder contains changes that deviate from the content of the Crow
repository. This file contains some comments about why these patches were
created.

## Building with libc++

Crow suffers from [this issue](https://github.com/CrowCpp/Crow/issues/576) when
building with `libc++` (the LLVM C++ standard library) instead of `libstdc++`
(the GNU C++ standard library). The necessary fix to enable building with
`libc++` is added by applying the patch in json.h.diff.
