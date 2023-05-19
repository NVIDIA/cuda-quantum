FROM quay.io/pypa/manylinux_2_28_x86_64 AS buildStage

RUN dnf -y install ninja-build && python3.10 -m pip install cmake lit --user 
ENV DEBIAN_FRONTEND=noninteractive \
    LLVM_VERSION=c0b45fef155fbe3f17f9a6f99074682c69545488 \
    LLVM_INSTALL_PATH=/opt/llvm

RUN mkdir llvm-project && \
    cd llvm-project && \
    git init && \
    git remote add origin https://github.com/llvm/llvm-project && \
    git fetch origin --depth=1 ${LLVM_VERSION} && \
    git reset --hard FETCH_HEAD

RUN cd /llvm-project && mkdir build \
  && cd build \
  && cmake ../llvm -G Ninja -DPython_EXECUTABLE=python3.10 \
                 -DLLVM_TARGETS_TO_BUILD="host" \
                 -DCMAKE_INSTALL_PREFIX=/opt/llvm/ \
                 -DLLVM_ENABLE_PROJECTS="clang;mlir" \
                 -DCMAKE_BUILD_TYPE=Release \
                 -DLLVM_ENABLE_ASSERTIONS=ON \
                 -DLLVM_INSTALL_UTILS=TRUE \
                 -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
                 -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
  && ninja install && cp bin/llvm-lit /opt/llvm/bin/

# Get OpenSSL + CPR + BLAS
RUN dnf install -y openssh-clients \
        perl-core wget \
        glibc-static zlib-static \
    && git clone https://github.com/openssl/openssl \
    && cd openssl && ./config --prefix=/usr/local/ssl --openssldir=/usr/local/ssl -static zlib \
    && make install \
    && cd / && git clone https://github.com/libcpr/cpr \
    && cd cpr && mkdir build && cd build \
    && cmake .. -G Ninja -DCPR_FORCE_USE_SYSTEM_CURL=FALSE \
                  -DCMAKE_INSTALL_LIBDIR=lib \
                  -DOPENSSL_USE_STATIC_LIBS=TRUE \
                  -DBUILD_SHARED_LIBS=FALSE \
                  -DOPENSSL_ROOT_DIR=/usr/local/ssl \
                  -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE \
                  -DCMAKE_INSTALL_PREFIX=/cpr/install \
    && ninja install 
RUN wget http://www.netlib.org/blas/blas-3.11.0.tgz \
    && tar -xzvf blas-3.11.0.tgz && cd BLAS-3.11.0 \
    && make && mv blas_LINUX.a /usr/lib64/libblas.a 