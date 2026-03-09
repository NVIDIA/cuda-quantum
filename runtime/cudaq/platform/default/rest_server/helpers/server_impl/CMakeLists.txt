# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME rest_server_impl)
add_library(${LIBRARY_NAME} OBJECT RestServer.cpp)

target_link_libraries(${LIBRARY_NAME} PRIVATE Crow::Crow)
target_include_directories(${LIBRARY_NAME} PRIVATE
  ../ ${Crow_SOURCE_DIR}/include ${asio_SOURCE_DIR}/asio/include)
