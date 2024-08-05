#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This scripts moves CUDA-Q assets to the correct locations.
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
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
  esac
done
OPTIND=$__optind__
[ -n "$target" ] && install=true || install=false

if $install; then
    CUDA_QUANTUM_PATH="$target" && mkdir -p "$CUDA_QUANTUM_PATH"
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
if [ ! -f "$build_config" ]; then
    echo -e "\e[01;31mError: Missing build configuration.\e[0m" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
fi

remove_assets="$CUDA_QUANTUM_PATH/$($install && echo uninstall.sh || echo remove_assets.sh)"
echo "Migrating assets in $assets."
echo "Using build configuration $build_config."
echo "The script to remove the migrated files can be found in $remove_assets."

read -r -d '' confirmation_prompt << 'EOP'
if [ "$1" == "-y" ]; then continue=true
else
    while true; do
        read -p "Continue (y/n)?" -r choice < /dev/tty
        case "$choice" in
            y|Y ) continue=true && break;;
            n|N ) continue=false && break;;
            * ) echo "Please enter y or n.";;
        esac
    done
fi
EOP

function move_artifacts {
    mkdir -p "$2" -m 755 && cd "$1"
    echo "Updating $2 with artifacts in $1:"

    # Prompting for confirmation to remove the copied files in that folder:
    echo 'echo "Cleaning up '$2':"' >> "$remove_assets"
    echo "$confirmation_prompt" >> "$remove_assets"
    echo 'if $continue; then' >> "$remove_assets"

    find . -type f -print0 | while IFS= read -r -d '' file;
    do 
        if [ ! -f "$2/$file" ]; 
        then 
            echo -e "\tadding file $2/$file"
            mkdir -p "$(dirname "$2/$file")" -m 755 # need x permissions to see content
            mv "$file" "$2/$file"
            echo '  echo -e "\tremoving file '$2/$file'"' >> "$remove_assets"
            echo "  rm $2/$file" >> "$remove_assets"
            echo '  rmdir -p "'$(dirname "$2/$file")'" 2> /dev/null || true' >> "$remove_assets"
            chmod a+rX "$2/$file" # add x permissions only for executables
        fi
    done
    for symlink in `find -L . -xtype l`;
    do
        if [ ! -f "$2/$symlink" ]; 
        then
            echo -e "\tadding symbolic link $2/$symlink"
            mkdir -p "$(dirname "$2/$symlink")" -m 755 # need x permissions to see content
            mv "$symlink" "$2/$symlink"
            echo '  echo -e "\tremoving symbolic link '$2/$symlink'"' >> "$remove_assets"
            echo "  rm $2/$symlink" >> "$remove_assets"
            echo '  rmdir -p "'$(dirname "$2/$symlink")'" 2> /dev/null || true' >> "$remove_assets"
        fi
    done
    for symlink in `find -L $2 -xtype l`;
    do
        if [ ! -e "$symlink" ] ; then
            echo -e "\e[01;31mError: Broken symbolic link $symlink pointing to $(readlink -f $symlink).\e[0m" >&2
            (return 0 2>/dev/null) && return 1 || exit 1
        fi
    done
    echo '  rmdir -p "'$2'" 2> /dev/null || true' >> "$remove_assets"
    echo 'fi' >> "$remove_assets"
    cd - > /dev/null
    find "$2" -type d -exec chmod 755 {} \;
}

rdom () { local IFS=\> ; read -d \< E C ;} && \
while rdom; do
    if [ "$E" = "LLVM_INSTALL_PREFIX" ]; then
        if [ -d "$assets/llvm" ]; then
            move_artifacts "$assets/llvm" "$C"
        elif $install; then
            echo -e "\e[01;31mError: Missing LLVM assets for installation.\e[0m" >&2
            (return 0 2>/dev/null) && return 1 || exit 1
        fi
    elif [ "$E" = "CUQUANTUM_INSTALL_PREFIX" ]; then
        if [ -d "$assets/cuquantum" ]; then
            move_artifacts "$assets/cuquantum" "$C"
        elif $install; then
            echo -e "\e[01;31mError: Missing cuQuantum assets for installation.\e[0m" >&2
            (return 0 2>/dev/null) && return 1 || exit 1
        fi
    elif [ "$E" = "CUTENSOR_INSTALL_PREFIX" ]; then
        if [ -d "$assets/cutensor" ]; then
            move_artifacts "$assets/cutensor" "$C"
        elif $install; then
            echo -e "\e[01;31mError: Missing cuTensor assets for installation.\e[0m" >&2
            (return 0 2>/dev/null) && return 1 || exit 1
        fi
    elif [ -n "$(echo $C | tr -d ' ')" ] && [ -d "$assets/$E" ]; then
        move_artifacts "$assets/$E" "$C"
    fi
done < "$build_config"

if [ -d "$assets/cudaq" ]; then
    move_artifacts "$assets/cudaq" "$CUDA_QUANTUM_PATH"
    if [ ! -f "$CUDA_QUANTUM_PATH/build_config.xml" ]; then
        cp "$build_config" "$CUDA_QUANTUM_PATH/build_config.xml"
    fi
    chmod a+rx "$(dirname "$CUDA_QUANTUM_PATH")"
elif $install; then
    echo -e "\e[01;31mError: Missing CUDA-Q assets for installation.\e[0m" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
fi

function update_profile {
    echo "Configuring CUDA-Q environment variables in $1."
    echo 'CUDAQ_INSTALL_PATH="'${CUDA_QUANTUM_PATH}'"' >> "$1"
    echo '. "${CUDAQ_INSTALL_PATH}/set_env.sh"' >> "$1"
    echo "sed -i '/^CUDAQ_INSTALL_PATH=/d' \"$1\"" >> "$remove_assets"
    echo "sed -i '/"'${CUDAQ_INSTALL_PATH}'"\/set_env.sh/d' \"$1\"" >> "$remove_assets"
}

if $install; then
    . "${CUDA_QUANTUM_PATH}/set_env.sh"
    # Note: Generally, the idea is to set the necessary environment variables
    # to make CUDA-Q discoverable in login shells and for all users. 
    # Non-login shells should inherit them from the original login shell. 
    # If we cannot modify /etc/profile, we instead modify $HOME/.bashrc, which 
    # is always executed by all interactive non-login shells.
    # The reason for this is that bash is a bit particular when it comes to user
    # level profiles for login-shells in the sense that there isn't one specific
    # file that is guaranteed to execute; it first looks for .bash_profile, 
    # then for .bash_login and .profile, and *only* the first file it finds is 
    # executed. Hence, the reliable and non-disruptive way to configure 
    # environment variables at the user level is to instead edit .bashrc.
    if [ -f /etc/profile ] && [ -w /etc/profile ]; then
        update_profile /etc/profile
    else
        update_profile $HOME/.bashrc
    fi
    if [ -f /etc/zprofile ] && [ -w /etc/zprofile ]; then
        update_profile /etc/zprofile
    fi
    if [ -d "${MPI_PATH}" ] && [ -n "$(ls -A "${MPI_PATH}"/* 2> /dev/null)" ] && [ -x "$(command -v "${CUDA_QUANTUM_PATH}/bin/nvq++")" ]; then
        plugin_path="${CUDA_QUANTUM_PATH}/distributed_interfaces"
        bash "${plugin_path}/activate_custom_mpi.sh" && rm -f mpi_comm_impl.o || true
        if [ -f "$plugin_path/libcudaq_distributed_interface_mpi.so" ]; then
            chmod a+rX "$plugin_path/libcudaq_distributed_interface_mpi.so"
        else
            echo -e "\e[01;31mWarning: Failed to build MPI plugin.\e[0m" >&2
            echo -e "Please make sure the necessary libraries and header files are discoverable and then build the MPI plugin by running the script `${plugin_path}/activate_custom_mpi.sh`."
        fi
    fi

    # Final step upon uninstalling is to delete the CUDA_QUANTUM_PATH folder itself.
    # The script will prompt for confirmation before doing that, since this also 
    # removes the build configuration and the script itself.
    echo 'echo "Removing remaining configurationa and scripts in '$CUDA_QUANTUM_PATH' - "' >> "$remove_assets"
    echo "$confirmation_prompt" >> "$remove_assets"
    echo 'if $continue; then' >> "$remove_assets"
    echo "  rm -rf \"$CUDA_QUANTUM_PATH\" && echo Uninstalled CUDA-Q." >> "$remove_assets"
    echo 'fi' >> "$remove_assets"
fi

this_file=`readlink -f "${BASH_SOURCE[0]}"`
remaining_files=(`find "$assets" -type f -not -path "$this_file" -not -path "$build_config"`)
if [ ! ${#remaining_files[@]} -eq 0 ]; then
    rel_paths=(${remaining_files[@]##$assets/})
    components=(`echo "${rel_paths[@]%%/*}" | tr ' ' '\n' | uniq`)
    echo -e "\e[01;31mWarning: Some files in $assets have not been migrated since they already exit in their intended destination. To avoid compatibility issues, please make sure the following packages are not already installed on your system: ${components[@]}\e[0m" >&2
fi
