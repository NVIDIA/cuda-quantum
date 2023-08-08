# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-partitioner-metis)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/partitioners)
add_library(${LIBRARY_NAME} SHARED MetisGraphPartitioner.cpp)
target_include_directories(${LIBRARY_NAME} PRIVATE ${METIS_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} PRIVATE ${METIS_LIBRARIES} ${METIS_LIBRARY_DIRS}/libGKlib.a cudaq-common LLVMSupport MLIRIR CUDAQSupport QuakeDialect)
add_cudaq_partitioner_plugin_install(${LIBRARY_NAME})
