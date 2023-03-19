# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

file(GLOB CONFIG_FILES CUDAQ*Config.cmake)
file (GLOB LANG_FILES CMake*)
install(FILES NVQIRConfig.cmake DESTINATION lib/cmake/nvqir)
install(FILES ${CONFIG_FILES} DESTINATION lib/cmake/cudaq)
install(FILES ${LANG_FILES} DESTINATION lib/cmake/cudaq)
install(FILES CUDAQConfig.cmake DESTINATION lib/cmake/cudaq)
