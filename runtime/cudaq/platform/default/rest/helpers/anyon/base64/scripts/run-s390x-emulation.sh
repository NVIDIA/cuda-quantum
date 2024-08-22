#!/usr/bin/env sh
#
#docker run --rm --privileged multiarch/qemu-user-static:register --reset
docker run -it multiarch/ubuntu-core:s390x-focal /bin/bash
apt-get update -q -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends make cmake g++ git
#software-properties-common
cd home
git clone https://github.com/tvercaut/base64.git
cd base64
git checkout modpb64xover
cmake -B ./build -DCMAKE_BUILD_TYPE=Debug .
cmake --build ./build --config Debug
cd build
ctest -C Debug --output-on-failure

