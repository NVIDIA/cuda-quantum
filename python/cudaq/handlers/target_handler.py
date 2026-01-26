# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from .photonics_kernel import PhotonicsHandler


class TargetHandler:
    """Base class for target-specific behavior"""

    def skip_compilation(self):
        # By default, perform compilation on the kernel
        return False

    def call_processed(self, decorator, args):
        # `None` indicates standard call should be used
        return None


class DefaultTargetHandler(TargetHandler):
    """Standard target handler"""
    pass


class PhotonicsTargetHandler(TargetHandler):
    """Handler for `orca-photonics` target"""

    def skip_compilation(self):
        return True

    def call_processed(self, kernel, args):
        if kernel is None:
            raise RuntimeError(
                "The 'orca-photonics' target must be used with a valid function."
            )
        # NOTE: Since this handler does not support MLIR mode (yet), just
        # invoke the kernel. If calling from a bound function, need to
        # unpack the arguments, for example, see `pyGetStateLibraryMode`
        try:
            context_name = cudaq_runtime.getExecutionContextName()
        except RuntimeError:
            context_name = None

        callable_args = args
        if "extract-state" == context_name and len(args) == 1:
            callable_args = args[0]

        PhotonicsHandler(kernel)(*callable_args)
        # `True` indicates call was handled
        return True
