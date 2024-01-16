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
# bash scripts/migrate_assets.sh -s "$assets"
# -or-
# bash scripts/migrate_assets.sh -s "$assets" -c "$build_config"
# -or-
# sudo -E bash scripts/migrate_assets.sh -s "$assets"
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

# Process command line arguments
(return 0 2>/dev/null) && is_sourced=true || is_sourced=false
__optind__=$OPTIND
OPTIND=1
while getopts ":c:s:t:" opt; do
  case $opt in
    c) config="$OPTARG"
    ;;
    s) source="$OPTARG"
    ;;
    t) target="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    if $is_sourced; then return 1; else exit 1; fi
    ;;
  esac
done
OPTIND=$__optind__

function move_artifacts {
    mkdir -p "$2" -m 755 && cd "$1"
    echo "Updating $2 with artifacts in $1:"
    for file in `find . -type f`; 
    do 
        if [ ! -f "$2/$file" ]; 
        then 
            echo -e "\tadding file $2/$file"
            mkdir -p "$(dirname "$2/$file")" -m 755 # need x permissions to see content
            mv "$file" "$2/$file"
            chmod a+rX "$2/$file" # add x permissions only for executables
        fi
    done
    for symlink in `find -L . -xtype l -not -path '*/resources/*'`;
    do
        if [ ! -f "$2/$symlink" ]; 
        then
            echo -e "\tadding symbolic link $2/$symlink"
            mkdir -p "$(dirname "$2/$symlink")" -m 755 # need x permissions to see content
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
    find "$2" -type d -exec chmod 755 {} \;
}

if [ -n "$target" ]; then
    CUDA_QUANTUM_PATH="$target"
    CUDAQ_INSTALL_PREFIX=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
elif [ -z "$CUDA_QUANTUM_PATH" ] && [ -z "$CUDAQ_INSTALL_PREFIX" ]; then 
    echo -e "\e[01;31mError: Neither CUDAQ_INSTALL_PREFIX nor CUDA_QUANTUM_PATH are defined.\e[0m" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
else
    CUDA_QUANTUM_PATH=${CUDA_QUANTUM_PATH:-"$CUDAQ_INSTALL_PREFIX"}
    CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-"$CUDA_QUANTUM_PATH"}
fi

assets="${source:-$CUDAQ_INSTALL_PREFIX}"
build_config="${config:-$assets/build_config.xml}"
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
    move_artifacts "$assets/cudaq" "$CUDA_QUANTUM_PATH"
    if [ ! -f "$CUDA_QUANTUM_PATH/build_config.yml" ]; then
        cp "$build_config" "$CUDA_QUANTUM_PATH/build_config.yml"
    fi
    chmod a+rx "$(dirname "$CUDA_QUANTUM_PATH")"
fi

this_file=`readlink -f "${BASH_SOURCE[0]}"`
remaining_files=(`find "$assets" -type f -not -path "$this_file" -not -path "$build_config"`)
if [ ! ${#remaining_files[@]} -eq 0 ]; then
    rel_paths=(${remaining_files[@]##$assets/})
    components=(`echo "${rel_paths[@]%%/*}" | tr ' ' '\n' | uniq`)
    echo -e "\e[01;31mWarning: Some files in $assets have not been migrated since they already exit in their intended destination. To avoid compatibility issues, please make sure the following packages are not already installed on your system: ${components[@]}\e[0m" >&2
fi
