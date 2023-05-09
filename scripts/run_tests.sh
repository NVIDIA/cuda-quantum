#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cd $CUDAQ_REPO_ROOT
ctest --output-on-failure --test-dir build -E ctest-nvqpp
ctest_status=$?
/opt/llvm/bin/llvm-lit -v --param nvqpp_site_config=build/test/lit.site.cfg.py build/test
lit_status=$?
if [ $ctest_status -eq 0 ] && [ $lit_status -eq 0 ]
then
    exit 0
else 
    echo "ctest failure status = " $ctest_status
    echo "llvm-lit failure status = " $lit_status
    exit 1
fi 
