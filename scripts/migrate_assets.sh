#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This scripts moves CUDA Quantum assets to the correct locations.
#
# Usage:
# bash scripts/migrate_assets.sh "$assets"
# -or-
# bash scripts/migrate_assets.sh "$assets" "$build_config"
# -or-
# sudo -E bash scripts/migrate_assets.sh "$assets"
#
# The assets variable should be set to the path of the directory
# that contains the files to migrate to their correct location.
# The migration target is defined by a the build_config file in 
# the assets folder. This file should be an xml file of the form
#
# <build_config>
# <subfolder_name>target_path</subfolder_name>
# </build_config>
#
# where subfolder_name is the relative path of the folder in assets,
# and target_path is the location to which its content should be 
# moved to.
# Note that existing files are never overwritten, and this script
# does not perform any validation regarding whether the copied
# files are compatible or functional after moving them.

function move_artifacts {
    cd "$1"
    echo "Updating $2 with artifacts in $1:"
    for file in `find . -type f`; 
    do 
        if [ ! -f "$2/$file" ]; 
        then 
            echo -e "\tadding file $2/$file"
            mkdir -p "$(dirname "$2/$file")"
            mv "$file" "$2/$file"
        fi
    done
    for symlink in `find -L . -xtype l`;
    do
        if [ ! -f "$2/$symlink" ]; 
        then
            echo -e "\tadding symbolic link $2/$symlink"
            mkdir -p "$(dirname "$2/$symlink")"
            mv "$symlink" "$2/$symlink"
        fi
    done
    for symlink in `find -L $2 -xtype l`;
    do
        if [ ! -e "$symlink" ] ; then
            echo "Error: broken symbolic link $symlink pointing to $(readlink -f $symlink)." 1>&2;
            exit 1
        fi
    done
    cd - > /dev/null
}

CUDA_QUANTUM_PATH=${CUDA_QUANTUM_PATH:-"$CUDAQ_INSTALL_PREFIX"}
CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-"$CUDA_QUANTUM_PATH"}

assets=${1:-"$CUDAQ_INSTALL_PREFIX"}
build_config=${2:-"$assets/build_config.xml"}
if [ ! -f "$build_config" ]; then 
    build_config="$CUDAQ_INSTALL_PREFIX/build_config.xml"
fi

echo "Migrating assets in $assets."
echo "Using build configuration $build_config."

rdom () { local IFS=\> ; read -d \< E C ;} && \
while rdom; do
    if [ "$E" = "LLVM_INSTALL_PREFIX" ] && [ -d "$assets/llvm" ]; then
        move_artifacts "$assets/llvm" "$C"
    elif [ "$E" = "CUQUANTUM_INSTALL_PREFIX" ] && [ -d "$assets/cuquantum" ]; then
        move_artifacts "$assets/cuquantum" "$C"
    elif [ "$E" = "CUTENSOR_INSTALL_PREFIX" ] && [ -d "$assets/cutensor" ]; then
        move_artifacts "$assets/cutensor" "$C"
    elif [ -d "$assets/$E" ] && [ -n "$(echo $C | tr -d ' ')" ]; then
        move_artifacts "$assets/$E" "$C"
    fi
done < "$build_config"

if [ -d "$assets/cudaq" ]; then
    move_artifacts "$assets/cudaq" "$CUDAQ_INSTALL_PREFIX"
fi

if [ -n "$(find "$assets" -type f)" ]; then
    echo "Warning: not all files in $assets have been migrated." 1>&2;
fi
