# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)
set(LIBRARY_NAME nvqir-nvidia-dynamics)
add_library(${LIBRARY_NAME} SHARED CuSuperOpBackend.cpp)

target_include_directories(${LIBRARY_NAME}
  PRIVATE 
    . .. 
    ${CUDAToolkit_INCLUDE_DIRS} 
    ${CMAKE_SOURCE_DIR}/runtime/common
)

target_link_libraries(${LIBRARY_NAME} 
                      PRIVATE
                        fmt::fmt-header-only
                        cudaq-common
                        CUDA::cudart_static
                     )
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

