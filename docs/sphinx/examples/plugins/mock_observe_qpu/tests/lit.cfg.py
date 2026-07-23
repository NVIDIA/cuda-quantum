# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Lit configuration for the mock_observe_qpu example plugin test.

import os
import lit.formats
import lit.util
from lit.llvm import llvm_config

config.name = 'mock_observe_qpu plugin'
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = ['.cpp']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.cudaq_obj_root, 'test',
                                     'mock-observe-qpu')

config.substitutions.append(('%cudaq_lib_dir', config.cudaq_lib_dir))
config.substitutions.append(('%cudaq_target_dir', config.cudaq_target_dir))
config.substitutions.append(
    ('%cudaq_example_plugins_dir', config.cudaq_example_plugins_dir))
config.substitutions.append(('%cudaq_plugin_ext', config.cudaq_plugin_ext))

llvm_config.use_default_substitutions()
llvm_config.with_environment('PATH', config.cudaq_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
