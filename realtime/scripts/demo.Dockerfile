ARG DOCA_VERSION=invalid
ARG CUDA_VERSION=invalid
FROM nvcr.io/nvidia/doca/doca:${DOCA_VERSION}-full-rt-cuda${CUDA_VERSION}.0.0

ARG CUDAQ_REALTIME_DIR=/opt/nvidia/cudaq/realtime

# Set LD_LIBRARY_PATH to include the CUDA-Q Realtime library path
ENV LD_LIBRARY_PATH="${CUDAQ_REALTIME_DIR}/lib:$LD_LIBRARY_PATH"
