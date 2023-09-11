#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if [ ! -d $1 ] ; then
    exit 0
fi

for config_file in `find $1/*.config -maxdepth 0 -type f`; do 
    current_dir="$(dirname "${config_file}")" 
    current_dir=$(cd $current_dir; pwd) 
    target_name=${config_file##*/} 
    target_name=${target_name%.config}             
    if [ -n "$RESULT_CONFIG" ]; then 
        RESULT_CONFIG="$RESULT_CONFIG;$current_dir/libnvqir-$target_name.so;$current_dir/$target_name.config"; 
    else 
        RESULT_CONFIG="$current_dir/libnvqir-$target_name.so;$current_dir/$target_name.config"; 
    fi 
done 

echo $RESULT_CONFIG