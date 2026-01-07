# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the CUDA Quantum binaries from scratch such that they can be
# used on a range of Linux systems, provided the requirements documented in 
# the data center installation guide are satisfied.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-assets:amd64-cu12-llvm-main -f docker/build/assets.Dockerfile .

# [Operating System]
ARG base_image=amd64/almalinux:8
FROM ${base_image} AS prereqs
SHELL ["/bin/bash", "-c"]
ARG cuda_version=12.6
ENV CUDA_VERSION=${cuda_version}

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)' && \
    dnf config-manager --enable powertools
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh

# [Prerequisites]
ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON}

# [Build Dependencies]
RUN dnf install -y --nobest --setopt=install_weak_deps=False wget git unzip

## [CUDA]
RUN source /cuda-quantum/scripts/configure_build.sh install-cuda
## [Compiler Toolchain]
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc

# [CUDA-Q Dependencies]
ADD scripts/install_prerequisites.sh /cuda-quantum/scripts/install_prerequisites.sh
ADD scripts/install_toolchain.sh /cuda-quantum/scripts/install_toolchain.sh
ADD scripts/build_llvm.sh /cuda-quantum/scripts/build_llvm.sh
ADD cmake/caches/LLVM.cmake /cuda-quantum/cmake/caches/LLVM.cmake
ADD tpls/customizations/llvm /cuda-quantum/tpls/customizations/llvm
ADD .gitmodules /cuda-quantum/.gitmodules
ADD .git/modules/tpls/pybind11/HEAD /.git_modules/tpls/pybind11/HEAD
ADD .git/modules/tpls/llvm/HEAD /.git_modules/tpls/llvm/HEAD

# This is a hack so that we do not need to rebuild the prerequisites 
# whenever we pick up a new CUDA-Q commit (which is always in CI).
RUN cd /cuda-quantum && git init && \
    git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | \
    while read path_key local_path; do \
        if [ -f "/.git_modules/$local_path/HEAD" ]; then \
            url_key=$(echo $path_key | sed 's/\.path/.url/') && \
            url=$(git config -f .gitmodules --get "$url_key") && \
            git update-index --add --cacheinfo 160000 \
            $(cat /.git_modules/$local_path/HEAD) $local_path; \
        fi; \
    done && git submodule init && git submodule
RUN cd /cuda-quantum && source scripts/configure_build.sh && \
    LLVM_PROJECTS='clang;flang;lld;mlir;openmp;runtimes' \
    bash scripts/install_prerequisites.sh -t llvm

# Validate that the built toolchain and libraries have no GCC dependencies.
RUN source /cuda-quantum/scripts/configure_build.sh && \
    shared_libraries=$(find "${LLVM_INSTALL_PREFIX}" -name '*.so') && \
    executables=$(find "${LLVM_INSTALL_PREFIX}" -executable -type f) && \
    for binary in ${shared_libraries} ${executables}; do \
        if [ -n "$(ldd "${binary}" 2>/dev/null | grep gcc)" ]; then \
            echo -e "\e[01;31mError: ${binary} depends on gcc libraries.\e[0m" >&2; \
        fi \
    done && \
    if [ -n "$(ldd ${shared_libraries} ${executables} | grep gcc)" ]; then \
        echo -e "\e[01;31mThe produced Clang toolchain and libraries depend on GCC libraries.\e[0m" >&2; \
        exit 1; \
    fi

# Checking out a CUDA-Q commit is suboptimal, since the source code
# version must match this file. At the same time, adding the entire current
# directory will always rebuild CUDA-Q, so we instead just add only
# the necessary files for the build to each build stage.
ADD .git/index /cuda-quantum/.git/index
ADD .git/modules/ /cuda-quantum/.git/modules/

## [C++ support]
FROM prereqs AS cpp_build
ADD "cmake" /cuda-quantum/cmake
ADD "docs/CMakeLists.txt" /cuda-quantum/docs/CMakeLists.txt
ADD "docs/sphinx/examples" /cuda-quantum/docs/sphinx/examples
ADD "docs/sphinx/applications" /cuda-quantum/docs/sphinx/applications
ADD "docs/sphinx/targets" /cuda-quantum/docs/sphinx/targets
ADD "docs/sphinx/snippets" /cuda-quantum/docs/sphinx/snippets
ADD "include" /cuda-quantum/include
ADD "lib" /cuda-quantum/lib
ADD "runtime" /cuda-quantum/runtime
ADD "scripts/build_cudaq.sh" /cuda-quantum/scripts/build_cudaq.sh
ADD "scripts/migrate_assets.sh" /cuda-quantum/scripts/migrate_assets.sh
ADD "scripts/cudaq_set_env.sh" /cuda-quantum/scripts/cudaq_set_env.sh
ADD "targettests" /cuda-quantum/targettests
ADD "test" /cuda-quantum/test
ADD "tools" /cuda-quantum/tools
ADD "tpls/customizations" /cuda-quantum/tpls/customizations
ADD "tpls/json" /cuda-quantum/tpls/json
ADD "unittests" /cuda-quantum/unittests
ADD "utils" /cuda-quantum/utils
ADD "CMakeLists.txt" /cuda-quantum/CMakeLists.txt
ADD "LICENSE" /cuda-quantum/LICENSE
ADD "NOTICE" /cuda-quantum/NOTICE

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version

# Note: We statically link libc++ here to make it easy to build shared CUDA-Q libraries with nvq++
# that can then be linked to and called from other C++ code compiled with a different toolchain
# and linked against a different standard library.
RUN cd /cuda-quantum && source scripts/configure_build.sh && \
    LLVM_STAGE1_BUILD="$(find "$(dirname "$(mktemp -d -u)")" -maxdepth 2 -name llvm)" \
    # IMPORTANT:
    # Make sure that the variables and arguments configured here match
    # the ones in the install_prerequisites.sh invocation in the prereqs stage!
    ## [>CUDAQuantumCppBuild]
    CUDAQ_ENABLE_STATIC_LINKING=TRUE \
    CUDAQ_REQUIRE_OPENMP=TRUE \
    CUDAQ_WERROR=TRUE \
    CUDAQ_PYTHON_SUPPORT=OFF \
    LLVM_PROJECTS='clang;flang;lld;mlir;openmp;runtimes' \
    bash scripts/build_cudaq.sh -t llvm -v
    ## [<CUDAQuantumCppBuild]

# Validate that the nvidia backend was built.
RUN source /cuda-quantum/scripts/configure_build.sh && \
    if [ -z "$(ls $CUDAQ_INSTALL_PREFIX/targets/nvidia.yml)" ]; then \
        echo -e "\e[01;31mError: Missing nvidia backend.\e[0m" >&2; \
        exit 1; \
    fi

# Validate that the built toolchain and libraries have no GCC dependencies.
RUN source /cuda-quantum/scripts/configure_build.sh && \
    shared_libraries=$(find "${CUDAQ_INSTALL_PREFIX}" -name '*.so') && \
    executables=$(find "${CUDAQ_INSTALL_PREFIX}" -executable -type f) && \
    has_gcc_dependencies=false && \
    for binary in ${shared_libraries} ${executables}; do \
        libname="$(basename "$binary")" && \
        # Linking cublas dynamically necessarily adds a libgcc_s dependency to the GPU-based simulators.
        # The same holds for a range of CUDA libraries, whereas libcudart.so does not have any GCC dependencies.
        if [ "${libname#libnvqir-custatevec}" != "$libname" ] || [ "${libname#libnvqir-tensornet}" != "$libname" ] || [ "${libname#libnvqir-dynamics}" != "$libname" ]; then \
            echo "Skipping validation of $libname."; \
        elif [ -n "$(ldd "${binary}" 2>/dev/null | grep gcc)" ]; then \
            has_gcc_dependencies=true && \
            echo -e "\e[01;31mError: ${binary} depends on gcc libraries.\e[0m" >&2; \
        fi \
    done && \
    if $has_gcc_dependencies; then \
        echo -e "\e[01;31mThe produced CUDA-Q toolchain and libraries depend on GCC libraries.\e[0m" >&2; \
        exit 1; \
    fi

## [Python support]
FROM prereqs AS python_build
# Bring all possible templates into the image, then pick the exact one
ADD pyproject.toml.cu* /cuda-quantum/
ADD "python" /cuda-quantum/python
ADD "cmake" /cuda-quantum/cmake
ADD "include" /cuda-quantum/include
ADD "lib" /cuda-quantum/lib
ADD "runtime" /cuda-quantum/runtime
ADD "tools" /cuda-quantum/tools
ADD "tpls/customizations" /cuda-quantum/tpls/customizations
ADD "tpls/json" /cuda-quantum/tpls/json
ADD "utils" /cuda-quantum/utils
ADD "CMakeLists.txt" /cuda-quantum/CMakeLists.txt
ADD "LICENSE" /cuda-quantum/LICENSE
ADD "NOTICE" /cuda-quantum/NOTICE

ARG release_version=
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$release_version

ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON}-devel && \
    ${PYTHON} -m ensurepip --upgrade && \
    ${PYTHON} -m pip install numpy build auditwheel patchelf

RUN cd /cuda-quantum && \
    . scripts/configure_build.sh && \
    case "${CUDA_VERSION%%.*}" in \
      12) cp pyproject.toml.cu12 pyproject.toml || true ;; \
      13) cp pyproject.toml.cu13 pyproject.toml || true ;; \
      *)  echo "Unsupported CUDA_VERSION=${CUDA_VERSION}"; exit 1 ;; \
    esac && \
    # Needed to retrigger the LLVM build, since the MLIR Python bindings
    # are not built in the prereqs stage.
    rm -rf "${LLVM_INSTALL_PREFIX}" && \
    LLVM_STAGE1_BUILD="$(find "$(dirname "$(mktemp -d -u)")" -maxdepth 2 -name llvm)" \
    # IMPORTANT:
    # Make sure that the invocation of the install_prerequisites.sh script here matches
    # the ones in the install_prerequisites.sh invocation in the prereqs stage!
    ## [>CUDAQuantumPythonBuild]
    LLVM_PROJECTS='clang;flang;lld;mlir;python-bindings;openmp;runtimes' \
    bash scripts/install_prerequisites.sh -t llvm && \
    CC="$LLVM_INSTALL_PREFIX/bin/clang" \
    CXX="$LLVM_INSTALL_PREFIX/bin/clang++" \
    FC="$LLVM_INSTALL_PREFIX/bin/flang-new" \
    python3 -m build --wheel
    ## [<CUDAQuantumPythonBuild]

# The '[a-z]*linux_[^\.]*' is meant to catch things like:
# - manylinux_2_28_x86_64,
# - manylinux_2_28_aarch64,
# - linux_x86_64, etc.
# If input is linux_<ARCH>, then choose manylinux_2_28_<ARCH> output
RUN echo "Patching up wheel using auditwheel..." && \
    ## [>CUDAQuantumWheel]
    CUDAQ_WHEEL="$(find . -name 'cuda_quantum*.whl')" && \
    MANYLINUX_PLATFORM="$(echo ${CUDAQ_WHEEL} | grep -o '[a-z]*linux_[^\.]*' | sed -re 's/^linux_/manylinux_2_28_/')" && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/_skbuild/lib" \ 
    python3 -m auditwheel -v repair ${CUDAQ_WHEEL} \
        --plat ${MANYLINUX_PLATFORM} \
        --exclude libcublas.so.11 \
        --exclude libcublasLt.so.11 \
        --exclude libcurand.so.10 \
        --exclude libcusolver.so.11 \
        --exclude libcusparse.so.11 \
        --exclude libcutensor.so.2 \
        --exclude libcutensornet.so.2 \
        --exclude libcustatevec.so.1 \
        --exclude libcudensitymat.so.0 \
        --exclude libcudart.so.11.0 \
        --exclude libnvToolsExt.so.1 \
        --exclude libnvidia-ml.so.1 \
        --exclude libcuda.so.1
    ## [<CUDAQuantumWheel]

# Validate that the nvidia backend was built.
RUN if [ -z "$(ls /cuda-quantum/_skbuild/targets/nvidia.yml)" ]; then \
        echo -e "\e[01;31mError: Missing nvidia backend.\e[0m" >&2; \
        exit 1; \
    fi

# Validate that the built toolchain and libraries have no GCC dependencies.
RUN source /cuda-quantum/scripts/configure_build.sh && \
    shared_libraries=$(find "${LLVM_INSTALL_PREFIX}" -name '*.so') && \
    executables=$(find "${LLVM_INSTALL_PREFIX}" -executable -type f) && \
    for binary in ${shared_libraries} ${executables}; do \
        if [ -n "$(ldd "${binary}" 2>/dev/null | grep gcc)" ]; then \
            echo -e "\e[01;31mError: ${binary} depends on gcc libraries.\e[0m" >&2; \
        fi \
    done && \
    if [ -n "$(ldd ${shared_libraries} ${executables} | grep gcc)" ]; then \
        echo -e "\e[01;31mThe produced Clang toolchain and libraries depend on GCC libraries.\e[0m" >&2; \
        exit 1; \
    fi

## [Python Tests]
FROM python_build AS python_tests
RUN gcc_packages=$(dnf list installed "gcc*" | sed '/Installed Packages/d' | cut -d ' ' -f1) && \
    dnf remove -y $gcc_packages && dnf clean all && \
    dnf install -y --nobest --setopt=install_weak_deps=False glibc-devel

## [Python MLIR tests]
RUN cd /cuda-quantum && source scripts/configure_build.sh && \
    python3 -m pip install lit pytest scipy && \
    "${LLVM_INSTALL_PREFIX}/bin/llvm-lit" -v _skbuild/python/tests/mlir \
        --param nvqpp_site_config=_skbuild/python/tests/mlir/lit.site.cfg.py
# The other tests for the Python wheel are run post-installation.

## [C++ Tests]
FROM cpp_build AS cpp_tests
RUN gcc_packages=$(dnf list installed "gcc*" | sed '/Installed Packages/d' | cut -d ' ' -f1) && \
    dnf remove -y $gcc_packages && dnf clean all && \
    dnf install -y --nobest --setopt=install_weak_deps=False glibc-devel

RUN if [ ! -x "$(command -v nvidia-smi)" ] || [ -z "$(nvidia-smi | egrep -o "CUDA Version: ([0-9]{1,}\.)+[0-9]{1,}")" ]; then \
        excludes="--label-exclude gpu_required"; \
    else \
        # Removing gcc packages remove the CUDA toolkit since it depends on them
        source /cuda-quantum/scripts/configure_build.sh install-cudart; \
    fi && cd /cuda-quantum && \
    # FIXME: Tensor unit tests for runtime errors throw a different exception.
    # Issue: https://github.com/NVIDIA/cuda-quantum/issues/2321
    excludes+=" --exclude-regex ctest-nvqpp|ctest-targettests|Tensor.*Error" && \
    ctest --output-on-failure --test-dir build $excludes

ENV PATH="${PATH}:/usr/local/cuda/bin" 
RUN if [ -x "$(command -v nvidia-smi)" ] && [ -n "$(nvidia-smi | egrep -o "CUDA Version: ([0-9]{1,}\.)+[0-9]{1,}")" ]; then \
        source /cuda-quantum/scripts/configure_build.sh install-cudart && \
        # Installing the CUDA compiler will install libstdc++ as well
        dnf install -y --nobest --setopt=install_weak_deps=False \
            cuda-compiler-$(echo ${CUDA_VERSION} | tr . -) \
            cuda-cudart-devel-$(echo ${CUDA_VERSION} | tr . -); \
    fi

RUN python3 -m ensurepip --upgrade && python3 -m pip install lit && \
    dnf install -y --nobest --setopt=install_weak_deps=False file which
RUN cd /cuda-quantum && source scripts/configure_build.sh && \
    if [ ! -x "$(command -v nvcc)" ]; then \
        # The tests is marked correctly as requiring nvcc, but since nvcc
        # is available during the build we need to filter it manually.
        filtered=" --filter-out MixedLanguage/cuda-1"; \
	filtered+="|AST-Quake/calling_convention|test_argument_conversion"; \
    fi && \
    "$LLVM_INSTALL_PREFIX/bin/llvm-lit" -v build/test \
        --param nvqpp_site_config=build/test/lit.site.cfg.py ${filtered} && \
    # FIXME: Some tests are still failing when building against libc++
    # tracked in https://github.com/NVIDIA/cuda-quantum/issues/1712
    filtered=" --filter-out Kernel/inline-qpu-func|execution/vector_bool_parameters" && \
    if [ ! -x "$(command -v nvcc)" ]; then \
        filtered+="|TargetConfig/check_compile"; \
    fi && \
    "$LLVM_INSTALL_PREFIX/bin/llvm-lit" -v build/targettests \
        --param nvqpp_site_config=build/targettests/lit.site.cfg.py ${filtered}

FROM cpp_tests
COPY --from=python_tests /wheelhouse /cuda-quantum/wheelhouse

# Installing a minimal set of CUDA development dependencies.
RUN . /cuda-quantum/scripts/configure_build.sh install-gcc && \
    . /cuda-quantum/scripts/configure_build.sh install-cudart && \
    dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-compiler-$(echo ${CUDA_VERSION} | tr . -) \
        cuda-cudart-devel-$(echo ${CUDA_VERSION} | tr . -) \
        libcublas-devel-$(echo ${CUDA_VERSION} | tr . -) \
        libcurand-devel-$(echo ${CUDA_VERSION} | tr . -) \
        libcusparse-devel-$(echo ${CUDA_VERSION} | tr . -) && \
    if [ $(echo $CUDA_VERSION | cut -d "." -f1) -ge 12 ]; then \
        dnf install -y --nobest --setopt=install_weak_deps=False \
            libnvjitlink-$(echo ${CUDA_VERSION} | tr . -); \
    fi

