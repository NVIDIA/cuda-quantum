# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_custom_target(cudaq-install-plugin)
configure_file("cudaq-install-plugin.in"
               "${CUDAQ_BINARY_DIR}/bin/cudaq-install-plugin"
               FILE_PERMISSIONS
                 OWNER_READ OWNER_WRITE OWNER_EXECUTE
                 GROUP_READ GROUP_EXECUTE
                 WORLD_READ WORLD_EXECUTE
               @ONLY)

install(PROGRAMS ${CUDAQ_BINARY_DIR}/bin/cudaq-install-plugin
        DESTINATION bin)
