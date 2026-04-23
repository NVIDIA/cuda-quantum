<!-- markdownlint-disable MD013 -->
# `remote-mqpu` Debugging Tips

This file contains tips and tricks for when you are performing manual testing/
debugging for `remote-mqpu` target. This file is primarily intended
for **CUDA-Q developers, not end users**. See the user-facing docs here:

- [`remote-mqpu`](https://nvidia.github.io/cuda-quantum/latest/using/backends/platform.html#remote-mqpu-platform)

## Fully local within `cuda-quantum-dev` container

The first step is usually to run the server in a separate window from the
client by disabling any sort of auto-launch capabilities.

1. In one window, launch `cudaq-qpud --port 3030`. You may also
   prefix this command with `CUDAQ_LOG_LEVEL=info` to turn on additional
   logging in the server.
2.
   - If you are using Python, change your `cudaq.set_target` line to be
   something like this: `cudaq.set_target('remote-mqpu', url='localhost:3030')`.
   - If you are using C++, change your `nvq++` command to something like this:
   `nvq++ --target remote-mqpu --remote-mqpu-url localhost:3030`.
