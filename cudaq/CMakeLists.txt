# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if (NOT DEFINED CUDAQ_ENABLE_CORE)
  set(CUDAQ_ENABLE_CORE ON CACHE BOOL "Enable building CUDA-Q core.")
endif()

if(CUDAQ_ENABLE_CORE)
  add_subdirectory(include)
  add_subdirectory(lib)
  add_subdirectory(tools)
  if(CUDAQ_BUILD_TESTS AND NOT CUDAQ_DISABLE_CPP_FRONTEND)
    add_subdirectory(test)
  endif()

  # The static archives behind libcudaqMLIR, exported under their bare names --
  # QuakeDialect, CCDialect, cudaq-synth, ... -- the same convention LLVM and
  # MLIR use for their component libraries, and the names CUDAQ_MLIR_BUNDLED_LIBS
  # already refers to. A downstream tool that opts into cudaq_use_static_mlir()
  # links these instead of libcudaqMLIR.
  install(EXPORT cudaq-dev-targets
          FILE CUDAQDevTargets.cmake
          DESTINATION lib/cmake/cudaq
          COMPONENT Development)
endif()
