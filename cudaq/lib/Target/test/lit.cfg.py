# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import platform
import shlex

import lit.formats

config.name = 'target-catalog'
config.test_format = lit.formats.ShTest()
config.suffixes = ['.test', '.yml']
config.test_source_root = os.path.dirname(__file__)
config.excludes = ['CMakeLists.txt', 'Inputs']

if platform.system() == 'Linux':
    config.available_features.add('system-linux')

config.substitutions.extend([
    ('%target-catalog-driver', shlex.quote(config.target_catalog_driver)),
    ('%cudaq-target-db-gen', shlex.quote(config.target_catalog_db_gen)),
    ('FileCheck', shlex.quote(config.target_catalog_filecheck)),
])
