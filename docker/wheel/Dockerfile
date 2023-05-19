# DOCKER_BUILDKIT=1 docker build -t nvidia/cudaq_manylinux_test --network host --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_ed25519)" . --output out
FROM nvidia/cudaq_manylinux_deps as buildStage 

# GET cudaq
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/ \
    && echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_ed25519 \
    && cat /root/.ssh/id_ed25519 && chmod 600 /root/.ssh/id_ed25519 \
    && touch /root/.ssh/known_hosts \
    && ssh-keyscan gitlab-master.nvidia.com >> /root/.ssh/known_hosts \
    && git clone -b wheel https://github.com/nvidia/cuda-quantum \
    cd cuda-quantum && git -c submodule.tpls/llvm.update=none submodule update --init --recursive \
    && rm -rf /root/.ssh 

ADD setup.py cuda-quantum/ 
ADD auditwheel cuda-quantum/
RUN cd cuda-quantum && chmod +x $PWD/auditwheel \ 
    && export LLVM_DIR=/opt/llvm/lib/cmake/llvm \
    && export CUDAQ_CPR_INSTALL=/cpr/install \
    && for i in {10} ; do python3.${i} -m pip install auditwheel scikit-build && python3.${i} setup.py bdist_wheel ; done \
    && for i in $(ls ./dist); do python3.10 $PWD/auditwheel -v repair dist/$i ; done

# Use this with DOCKER_BUILDKIT=1
FROM scratch AS exportStage 
COPY --from=buildStage /cuda-quantum/dist/*.whl . 