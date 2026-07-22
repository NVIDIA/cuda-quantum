# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import lit.formats

config.name = "cudaq-pulse"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

tools_dir = getattr(config, "cudaq_pulse_tools_dir", "")
llvm_tools = getattr(config, "llvm_tools_dir", "")
config.environment["PATH"] = os.pathsep.join(
    filter(None, [tools_dir, llvm_tools, os.environ.get("PATH", "")])
)
