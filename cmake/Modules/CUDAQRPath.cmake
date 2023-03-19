# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard()

function(cudaq_library_set_rpath LIBRARY_NAME)
  if(APPLE)
    set_target_properties(${LIBRARY_NAME}
      PROPERTIES INSTALL_RPATH "@loader_path")
    set_target_properties(${LIBRARY_NAME}
      PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  else()
    set_target_properties(${LIBRARY_NAME}
      PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

endfunction()
