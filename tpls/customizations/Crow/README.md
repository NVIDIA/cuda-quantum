# Crow patches

This folder contains changes that deviate from the content of the Crow
repository. This file contains some comments about why these patches were
created.

## Building with libc++

Crow suffers from [this issue](https://github.com/CrowCpp/Crow/issues/576) when
building with `libc++` (the LLVM C++ standard library) instead of `libstdc++`
(the GNU C++ standard library). The necessary fix to enable building with
`libc++` is added by applying the patch in json.h.diff.
This change has already been contributed to the Crow repository. It has been
merged in commit ad337a8a868d1d6bc61e9803fc82f36c61f706de, after version 1.1.0.
As soon as a new release comes out, the Crow submodule commit should be updated
and the patch here removed.
