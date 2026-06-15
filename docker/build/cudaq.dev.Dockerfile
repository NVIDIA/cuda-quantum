# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Build from the repo root with
#   docker build -t nvidia/cuda-quantum-dev:latest -f docker/build/cudaq.dev.Dockerfile .
#
# If a custom base image is used, then that image (i.e. the build environment) must 
# 1) have all the necessary build dependendencies installed
# 2) define the LLVM_INSTALL_PREFIX environment variable indicating where the 
#    the LLVM binaries that CUDA-Q depends on are installed
# 3) set the CC and CXX environment variable to use the same compiler toolchain
#    as the LLVM dependencies have been built with.

ARG base_image=ghcr.io/nvidia/cuda-quantum-devcontainer:cu12.6-gcc12-main
# Default empty stage for ccache data. CI overrides this with
# --build-context ccache-data=<path> to inject a pre-populated cache,
# while the devcontainer builds get the scratch as a noop.
FROM scratch AS ccache-data
FROM $base_image AS devbuild

ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

# Install MPI before ADD so this layer is cached across source changes.
# mpich or openmpi
ARG mpi=
RUN if [ -n "$mpi" ]; \
    then \
        if [ ! -z "$MPI_PATH" ]; then \
            echo "Using a base image with MPI is not supported when passing a 'mpi' build argument." && exit 1; \
        else \
			apt update && apt install -y lib$mpi-dev ; \
		fi \
    fi

ARG workspace=.
ARG destination="$CUDAQ_REPO_ROOT"
ADD "$workspace" "$destination"
WORKDIR "$destination"

# Configuring a base image that contains the necessary dependencies for GPU
# accelerated components and passing a build argument 
#   install="CMAKE_BUILD_TYPE=Release CUDA_QUANTUM_VERSION=latest"
# creates a dev image that can be used as argument to docker/release/cudaq.Dockerfile
# to create the released cuda-quantum image.
ARG install=
ARG git_source_sha=xxxxxxxx
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_BASEDIR="$CUDAQ_REPO_ROOT"
ENV CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime,time_macros,pch_defines
ENV CCACHE_COMPILERCHECK=content
ENV CCACHE_LOGFILE=/root/.ccache/ccache.log
RUN --mount=from=ccache-data,target=/tmp/ccache-import,rw \
    if [ -d /tmp/ccache-import ] && [ "$(ls -A /tmp/ccache-import 2>/dev/null)" ]; then \
        echo "Importing ccache data..." && \
        mkdir -p /root/.ccache && cp -a /tmp/ccache-import/. /root/.ccache/ && \
        ccache -s 2>/dev/null || true && \
        ccache -z 2>/dev/null || true && \
        find /root/.ccache -type f | wc -l | tr -d ' ' > /root/.ccache/_restore_file_count.txt; \
    else \
        echo "No ccache data injected using empty scratch stage." && \
        mkdir -p /root/.ccache; \
    fi && \
    if [ -n "$install" ]; \
    then \
        expected_prefix=$CUDAQ_INSTALL_PREFIX; \
        install=`echo $install | xargs` && export $install; \
        bash scripts/build_cudaq.sh -v -- -DCUDAQ_TEST_OMP_SLOTS=2; \
        if [ ! "$?" -eq "0" ]; then \
            exit 1; \
        elif [ "$CUDAQ_INSTALL_PREFIX" != "$expected_prefix" ]; then \
            mkdir -p "$expected_prefix"; \
            mv "$CUDAQ_INSTALL_PREFIX"/* "$expected_prefix"; \
            rmdir "$CUDAQ_INSTALL_PREFIX"; \
        fi; \
        echo "source-sha: $git_source_sha" > "$CUDAQ_INSTALL_PREFIX/build_info.txt"; \
    fi && \
    echo "=== ccache stats ===" && (ccache -s 2>/dev/null || true) && \
    (ccache --print-stats 2>/dev/null || ccache -s 2>/dev/null) > /root/.ccache/_build_stats.txt

# CI test stages. Build with --target test or --target test-mpi to run
# tests inside BuildKit without exporting the image. The run_tests arg
# defaults to false so non-CI builds skip these stages entirely.
FROM devbuild AS test
ARG run_tests=false
RUN if [ "$run_tests" = "true" ]; then \
        cd $CUDAQ_REPO_ROOT && \
        python3 -m pip install -r requirements-tests-backend.txt --break-system-packages && \
        bash scripts/run_tests.sh -v; \
    fi

FROM test AS test-mpi
ARG run_tests=false
ARG mpi=
RUN if [ "$run_tests" = "true" ] && [ -n "$mpi" ]; then \
        has_ompiinfo=$(which ompi_info || true) && \
        if [ -n "$has_ompiinfo" ]; then \
            export MPI_PATH="/usr/lib/$(uname -m)-linux-gnu/openmpi/"; \
        else \
            export MPI_PATH="/usr/lib/$(uname -m)-linux-gnu/mpich/"; \
        fi && \
        source $CUDAQ_INSTALL_PREFIX/distributed_interfaces/activate_custom_mpi.sh && \
        cd $CUDAQ_REPO_ROOT && \
        ctest --test-dir build -R MPIApiTest -V; \
    fi

# CI coverage stage. Build with --target coverage to generate code
# coverage inside BuildKit. The run_coverage arg defaults to false
# so non-CI builds skip this stage entirely.
FROM devbuild AS coverage
ARG run_coverage=false
RUN if [ "$run_coverage" = "true" ]; then \
        cd $CUDAQ_REPO_ROOT && \
        bash scripts/generate_cc.sh -v -c -p; \
    fi

FROM scratch AS coverage-export
COPY --from=coverage /workspaces/cuda-quantum/build/ccoverage/coverage.txt /coverage.txt
COPY --from=coverage /workspaces/cuda-quantum/build/pycoverage/coverage.xml /coverage.xml

# Export ccache data so CI can extract it for persistence.
# Tar inside the container to export a single file instead of thousands of
# small ccache entries. This avoids pathological slowness in BuildKit's
# per-file gRPC export through the docker-container driver.
# Build with --target ccache-export --output type=local,dest=/tmp/ccache-export
FROM devbuild AS ccache-tar
RUN tar cf /ccache.tar -C /root/.ccache .

FROM scratch AS ccache-export
COPY --from=ccache-tar /ccache.tar /

FROM devbuild
