# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cat /opt/nvidia/cudaq/build_info.txt

# Launch script: launch cudaq-qpud (nvcf mode) with MPI ranks == Number of NVIDIA GPUs
# IMPORTANT:
# (1) NVCF function must set container environment variable `NUM_GPUS`
# equal to the number of GPUs on the target platform. This will allow clients to query
# the function capability (number of GPUs) by looking at function info. The below
# entry point script helps prevent mis-configuration by checking that functions are
# created and deployed appropriately.
# (2) NVCF function must set container environment variable `NVQC_REST_PAYLOAD_VERSION` equal
# to the RestRequest payload version with which `cudaq-qpud` in the deployment Docker image was compiled.
# Failure to do so will result in early exits of the entry point command, thus deployment failure.
EXPECTED_REST_PAYLOAD_VERSION="$(cudaq-qpud --type nvcf --schema-version | grep -o "CUDA-Q REST API version: \S*" | cut -d ":" -f 2 | tr -d " ")"
if [[ "$NVQC_REST_PAYLOAD_VERSION" !=  "$EXPECTED_REST_PAYLOAD_VERSION" ]]; then
  echo "Invalid Deployment: NVQC_REST_PAYLOAD_VERSION environment variable ($NVQC_REST_PAYLOAD_VERSION) does not match cudaq-qpud (expected $EXPECTED_REST_PAYLOAD_VERSION)."
  exit 1
fi

python3 /nvqc_scripts/nvqc_proxy.py &

NUM_ACTUAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ "$NUM_GPUS" == "$NUM_ACTUAL_GPUS" ]]; then
  cd /tmp
  CMDSTR="mpiexec -np $NUM_ACTUAL_GPUS cudaq-qpud --type nvcf --port 3031"
  while true; do
    echo "export PATH=${PATH}; $CMDSTR" | sudo su -s /bin/bash nobody
  done
else
  echo "Invalid Deployment: Number of GPUs does not match the hardware"
  exit 1
fi
