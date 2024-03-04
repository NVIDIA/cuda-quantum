# General LLVM build settings
set(LLVM_TARGETS_TO_BUILD host CACHE STRING "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_OPTIMIZED_TABLEGEN ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "")

# Path configurations
set(CLANG_RESOURCE_DIR "../" CACHE STRING "")
set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib" CACHE STRING "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR OFF CACHE BOOL "") # see https://github.com/llvm/llvm-project/issues/62114

# Default configurations for the built toolchain
set(CLANG_DEFAULT_CXX_STDLIB libc++ CACHE STRING "")
set(CLANG_DEFAULT_RTLIB compiler-rt CACHE STRING "")
set(CLANG_DEFAULT_UNWINDLIB libunwind CACHE STRING "")
set(CLANG_DEFAULT_OPENMP_RUNTIME libomp CACHE STRING "")
set(CLANG_DEFAULT_LINKER lld CACHE STRING "")

# Runtime related build configurations
set(LLVM_ENABLE_LIBCXX ON CACHE BOOL "")
set(COMPILER_RT_USE_LIBCXX ON CACHE BOOL "")
set(LIBUNWIND_USE_COMPILER_RT ON CACHE BOOL "")
set(LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
set(LIBCXXABI_USE_COMPILER_RT ON CACHE BOOL "")
set(LIBCXXABI_USE_LLVM_UNWINDER ON CACHE BOOL "")
set(LIBCXX_CXX_ABI libcxxabi CACHE STRING "")
set(LIBCXX_HAS_GCC_LIB OFF CACHE BOOL "")
set(LIBCXX_HAS_GCC_S_LIB OFF CACHE BOOL "")
set(LIBCXX_HAS_ATOMIC_LIB OFF CACHE BOOL "")

# If we want to build dynamic libraries for the unwinder,
# we need to build support for exception handling.
# set(LLVM_ENABLE_EH ON CACHE BOOL "")
# set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
# Other option: don't build dynamic libraries...
set(LIBUNWIND_ENABLE_SHARED OFF CACHE BOOL "")
set(LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY ON CACHE BOOL "")
set(LIBCXXABI_ENABLE_SHARED OFF CACHE BOOL "")
set(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY ON CACHE BOOL "")
set(LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")

# See https://libcxx.llvm.org/BuildingLibcxx.html regarding the following settings:
# - LIBCXX_ENABLE_STATIC_ABI_LIBRARY
# - LIBCXX_HERMETIC_STATIC_LIBRARY

# See also the multi-stage LLVM build: 
# https://github.com/llvm/llvm-project/blob/main/clang/cmake/caches/Release.cmake