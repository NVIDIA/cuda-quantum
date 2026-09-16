# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import lit.formats
import lit.util

config.name = "QLX"
config.test_format = lit.formats.ShTest(True)

config.suffixes = [".mlir", ".test"]

config.test_source_root = os.path.dirname(__file__)

# test_exec_root is set by the site config before loading us.

# Retained CLI workflow tests consume the checked-in top-level examples.
config.substitutions.append(("%qlx_src_dir", config.qlx_src_dir))

# Features for REQUIRES lines. Each entry advertises a feature when the
# corresponding tool exists in ${QLX_TOOLS_DIR}.
for _tool in ("qlx-opt", "qlx-translate"):
    if os.path.isfile(os.path.join(config.qlx_tools_dir, _tool)):
        config.available_features.add(_tool)
if getattr(config, "has_cudaq_quake", "").upper() in ("1", "ON", "TRUE", "YES"):
    config.available_features.add("cudaq-quake")

# Build PATH with our tools dir and FileCheck.
path = os.environ.get("PATH", "")
if hasattr(config, "qlx_tools_dir") and config.qlx_tools_dir:
    path = config.qlx_tools_dir + os.pathsep + path
if hasattr(config,
           "filecheck_path") and config.filecheck_path and os.path.isfile(
               config.filecheck_path):
    path = os.path.dirname(config.filecheck_path) + os.pathsep + path
if hasattr(config, "llvm_tools_dir") and config.llvm_tools_dir:
    path = config.llvm_tools_dir + os.pathsep + path
config.environment["PATH"] = path

# Python support for FileCheck tests.
#
# The %python substitution is always available when an interpreter
# was discovered.  The "qlx-python" feature, by contrast, gates tests
# that import the CUDA-Q Logical Python package -- it must
# only be advertised when `import cudaq.logical` actually works from the
# configured PYTHONPATH.  Otherwise a build where the user hasn't built
# the python target yet would run those tests and fail with
# ModuleNotFoundError.
if hasattr(config, "python_executable") and config.python_executable:
    config.substitutions.append(("%python", config.python_executable))

if hasattr(config, "qlx_python_dir") and config.qlx_python_dir:
    config.environment["PYTHONPATH"] = config.qlx_python_dir

if (hasattr(config, "python_executable") and config.python_executable and
        hasattr(config, "qlx_python_dir") and config.qlx_python_dir):
    import subprocess
    _probe_env = dict(os.environ)
    _probe_env["PYTHONPATH"] = config.qlx_python_dir
    _probe = subprocess.run([
        config.python_executable, "-c",
        "import _cudaq_logical_devpath, cudaq.logical"
    ],
                            env=_probe_env,
                            capture_output=True)
    if _probe.returncode == 0:
        config.available_features.add("qlx-python")
