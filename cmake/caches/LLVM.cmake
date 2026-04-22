# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# See also the multi-stage LLVM build: 
# https://github.com/llvm/llvm-project/blob/main/clang/cmake/caches/Release.cmake

# General LLVM build settings
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_OPTIMIZED_TABLEGEN ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
# For reasons that aren't entirely clear to me, we need to build with assertions
# to avoid issues with the remote rest qpu when building with gcc.
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")

set(LLVM_BUILD_TESTS OFF CACHE BOOL "")
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")

if(DEFINED LLVM_ENABLE_RUNTIMES AND LLVM_ENABLE_RUNTIMES MATCHES "libcxx")
    message(STATUS "Setting defaults to use LLVM runtimes.")

    # If we want to build dynamic libraries for the unwinder,
    # we need to build support for exception handling.
    set(LLVM_ENABLE_EH ON CACHE BOOL "")
    set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
    set(LIBCXX_ENABLE_EXCEPTIONS ON CACHE BOOL "")
    set(LIBCXXABI_ENABLE_EXCEPTIONS ON CACHE BOOL "")
    set(LIBCXXABI_ENABLE_FORGIVING_DYNAMIC_CAST ON CACHE BOOL "")

    # Path configurations
    set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE BOOL "")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF CACHE BOOL "")
    set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/${LLVM_DEFAULT_TARGET_TRIPLE}:$ORIGIN/lib/${LLVM_DEFAULT_TARGET_TRIPLE}:$ORIGIN/../lib/${LLVM_DEFAULT_TARGET_TRIPLE}" CACHE STRING "")

    # Default configurations for the built toolchain
    set(CLANG_DEFAULT_OPENMP_RUNTIME libomp CACHE STRING "")
    set(CLANG_DEFAULT_LINKER lld CACHE STRING "")
    set(CLANG_DEFAULT_CXX_STDLIB libc++ CACHE STRING "")
    set(CLANG_DEFAULT_RTLIB compiler-rt CACHE STRING "")
    set(CLANG_DEFAULT_UNWINDLIB libunwind CACHE STRING "")

    # Runtime related build configurations
    set(LLVM_ENABLE_LIBCXX ON CACHE BOOL "")
    set(LIBCXX_CXX_ABI libcxxabi CACHE STRING "")
    set(COMPILER_RT_USE_LIBCXX ON CACHE BOOL "")
    set(COMPILER_RT_USE_LLVM_UNWINDER ON CACHE BOOL "")
    set(LIBUNWIND_USE_COMPILER_RT ON CACHE BOOL "")
    set(LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
    set(LIBCXXABI_USE_COMPILER_RT ON CACHE BOOL "")
    set(LIBCXXABI_USE_LLVM_UNWINDER ON CACHE BOOL "")
    set(LIBUNWIND_HAS_GCC_LIB OFF CACHE BOOL "")
    set(LIBUNWIND_HAS_GCC_S_LIB OFF CACHE BOOL "")
    set(LIBCXXABI_HAS_GCC_LIB OFF CACHE BOOL "")
    set(LIBCXXABI_HAS_GCC_S_LIB OFF CACHE BOOL "")
    set(LIBCXX_HAS_GCC_LIB OFF CACHE BOOL "")
    set(LIBCXX_HAS_GCC_S_LIB OFF CACHE BOOL "")
    set(COMPILER_RT_HAS_GCC_LIB OFF CACHE BOOL "")
    set(COMPILER_RT_HAS_GCC_S_LIB OFF CACHE BOOL "")
    set(COMPILER_RT_BUILD_BUILTINS ON CACHE BOOL "")
    set(COMPILER_RT_BUILD_LIBFUZZER OFF CACHE BOOL "")
    set(COMPILER_RT_BUILD_SANITIZERS OFF CACHE BOOL "")
    set(COMPILER_RT_BUILD_PROFILE OFF CACHE BOOL "")
    set(COMPILER_RT_BUILD_CTX_PROFILE OFF CACHE BOOL "")
    set(COMPILER_RT_BUILD_MEMPROF OFF CACHE BOOL "")
    set(COMPILER_RT_BUILD_XRAY OFF CACHE BOOL "")
    set(COMPILER_RT_USE_BUILTINS_LIBRARY ON CACHE BOOL "")
    # This looks like a configuration that was originally used and
    # probably automatically set to the correct value by the larger
    # LLVM build infrastructure. However, at the time of writing, 
    # this setting was only ever used in one place in the LLVM
    # codebase and never set. Specifically, it is used to determine 
    # whether to initialize the variables that capture what 
    # architectures compiler-rt can target.
    # If this is not set, the check is not performed before 
    # COMPILER_RT_HAS_CRT is set and the dependent option 
    # COMPILER_RT_BUILD_CRT has no effect, causing the build to 
    # not create clang_rt.crtbegin.o and clang_rt.crtend.o.
    set(COMPILER_RT_CRT_STANDALONE_BUILD ON CACHE BOOL "")
    set(COMPILER_RT_BUILD_CRT ON CACHE BOOL "")
    set(COMPILER_RT_BUILD_ORC ON CACHE BOOL "")

    # Build a static libc++ without any dependencies;
    # see also https://libcxx.llvm.org/BuildingLibcxx.html
    set(LIBCXX_HERMETIC_STATIC_LIBRARY ON CACHE BOOL "")
    set(LIBCXXABI_HERMETIC_STATIC_LIBRARY ON CACHE BOOL "")
    set(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY ON CACHE BOOL "")
    set(LIBCXXABI_ENABLE_STATIC_UNWINDER ON CACHE BOOL "")
    set(LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY ON CACHE BOOL "")
    set(COMPILER_RT_ENABLE_STATIC_UNWINDER ON CACHE BOOL "")

endif()
