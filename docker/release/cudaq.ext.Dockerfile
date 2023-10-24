# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ghcr.io/nvidia/cuda-quantum:latest-base
FROM $base_image

USER root

# Install VS Code CLI to facilitate connecting via tunnel to a remote host.
RUN os=$([ "$(uname -m)" == "aarch64" ] && echo cli-alpine-arm64 || echo cli-alpine-x64) \
    && curl -Lk "https://code.visualstudio.com/sha/download?build=stable&os=$os" --output vscode_cli.tar.gz \
    && tar -xf vscode_cli.tar.gz && rm vscode_cli.tar.gz && mv code /usr/bin/

# Copy over additional CUDA Quantum assets.
ARG assets=./assets
COPY "$assets" "$CUDA_QUANTUM_PATH/assets/"
RUN if [ -d "$CUDA_QUANTUM_PATH/assets/documentation" ]; then \
        mkdir -p /home/cudaq/docs && \
        mv "$CUDA_QUANTUM_PATH/assets/documentation"/* /home/cudaq/docs && \
        rmdir "$CUDA_QUANTUM_PATH/assets/documentation"; \
    fi
ADD ./scripts/migrate_assets.sh "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"
RUN for folder in `find "$CUDA_QUANTUM_PATH/assets"/*$(uname -m)/* -maxdepth 0 -type d`; \
    do bash "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh" "$folder" && rm -rf "$folder"; done \
    && rm "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"

# Install additional runtime dependencies.
RUN apt-get install -y --no-install-recommends \
        cuda-nvtx-11-8 libcusolver-11-8 libopenblas-openmp-dev \
        # just here for convenience:
        curl jq 
RUN if [ -x "$(command -v pip)" ]; then \
        apt-get install -y --no-install-recommends gcc \
        && pip install --no-cache-dir jupyterlab matplotlib; \
        if [ -n "$MPI_ROOT" ]; then \
            pip install --no-cache-dir mpi4py~=3.1; \
        fi; \
    fi

# Make sure that apt-get remains updated at the end!;
# If we don't do that, then apt-get will get confused when some CUDA
# components are already installed but not all of them.

USER cudaq

ARG vscode_config=.vscode
COPY "${vscode_config}" $HOME/.vscode
ARG vscode_profile='{\
    "name": "CUDA Quantum",\
    "settings": "{\"settings\":\"{\\r\\n    \\\"workbench.colorTheme\\\": \\\"Default Light+\\\",\\r\\n    \\\"files.exclude\\\": {\\r\\n        \\\".*\\\": true\\r\\n    }\\r\\n}\"}",\
    "extensions": "[{\"identifier\":{\"id\":\"davidanson.vscode-markdownlint\",\"uuid\":\"daf8b44d-8aae-4da2-80c5-1f770219f643\"},\"displayName\":\"markdownlint\"},{\"identifier\":{\"id\":\"jithurjacob.nbpreviewer\",\"uuid\":\"c5a0e020-41e5-4180-aadc-85aa7c4e0517\"},\"displayName\":\"VS Code Jupyter Notebook Previewer\"},{\"identifier\":{\"id\":\"llvm-vs-code-extensions.vscode-mlir\",\"uuid\":\"223e0adc-e5c8-4025-bfb8-b7a1ff223705\"},\"displayName\":\"MLIR\"},{\"identifier\":{\"id\":\"ms-python.python\",\"uuid\":\"f1f59ae4-9318-4f3c-a9b5-81b2eaa5f8a5\"},\"displayName\":\"Python\"},{\"identifier\":{\"id\":\"ms-python.vscode-pylance\",\"uuid\":\"364d2426-116a-433a-a5d8-a5098dc3afbd\"},\"displayName\":\"Pylance\"},{\"identifier\":{\"id\":\"ms-toolsai.jupyter\",\"uuid\":\"6c2f1801-1e7f-45b2-9b5c-7782f1e076e8\"},\"displayName\":\"Jupyter\"},{\"identifier\":{\"id\":\"ms-toolsai.jupyter-keymap\",\"uuid\":\"9f6dc8db-620c-4844-b8c5-e74914f1be27\"},\"displayName\":\"Jupyter Keymap\"},{\"identifier\":{\"id\":\"ms-toolsai.jupyter-renderers\",\"uuid\":\"b15c72f8-d5fe-421a-a4f7-27ed9f6addbf\"},\"displayName\":\"Jupyter Notebook Renderers\"},{\"identifier\":{\"id\":\"ms-toolsai.vscode-jupyter-cell-tags\",\"uuid\":\"ab4fb32a-befb-4102-adf9-1652d0cd6a5e\"},\"displayName\":\"Jupyter Cell Tags\"},{\"identifier\":{\"id\":\"ms-toolsai.vscode-jupyter-slideshow\",\"uuid\":\"e153ca70-b543-4865-b4c5-b31d34185948\"},\"displayName\":\"Jupyter Slide Show\"},{\"identifier\":{\"id\":\"ms-vscode.cmake-tools\",\"uuid\":\"7c889349-8749-43d4-8b5e-08939936d7f4\"},\"displayName\":\"CMake Tools\"},{\"identifier\":{\"id\":\"ms-vscode.cpptools\",\"uuid\":\"690b692e-e8a9-493f-b802-8089d50ac1b2\"},\"displayName\":\"C/C++\"},{\"identifier\":{\"id\":\"ms-vscode.cpptools-extension-pack\",\"uuid\":\"3957b2f6-f086-49b5-a7b4-5da772123130\"},\"displayName\":\"C/C++ Extension Pack\"},{\"identifier\":{\"id\":\"ms-vscode.cpptools-themes\",\"uuid\":\"99b17261-8f6e-45f0-9ad5-a69c6f509a4f\"},\"displayName\":\"C/C++ Themes\"},{\"identifier\":{\"id\":\"ms-vscode.remote-explorer\",\"uuid\":\"11858313-52cc-4e57-b3e4-d7b65281e34b\"},\"displayName\":\"Remote Explorer\"},{\"identifier\":{\"id\":\"ms-vscode.remote-server\",\"uuid\":\"105c0b3c-07a9-4156-a4fc-4141040eb07e\"},\"displayName\":\"Remote - Tunnels\"},{\"identifier\":{\"id\":\"qiu.llvm-ir-language-support\",\"uuid\":\"15b56816-1257-4c77-8f62-681fb363ec70\"},\"displayName\":\"LLVM IR Language Support for VSCode\"},{\"identifier\":{\"id\":\"twxs.cmake\",\"uuid\":\"2be7cf14-2603-402e-9771-fd79df83cdf8\"},\"displayName\":\"CMake\"},{\"identifier\":{\"id\":\"vadimcn.vscode-lldb\",\"uuid\":\"bee31e34-a44b-4a76-9ec2-e9fd1439a0f6\"},\"displayName\":\"CodeLLDB\"}]"\
}'
RUN mkdir -p ~/.vscode && echo "$vscode_profile" > ~/.vscode/cuda-quantum.code-profile
