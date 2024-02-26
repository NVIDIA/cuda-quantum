#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script installs the specified C/C++ toolchain, 
# and exports the CC and CXX environment variables.
#
# Usage:
#   source scripts/install_toolchain.sh -t <toolchain>
# -or-
#   source scripts/install_toolchain.sh -t <toolchain> -e path/to/dir
#
# where <toolchain> can be either llvm, clang16, gcc12, or gcc11. 
# The -e option creates a init_command.sh file in the given directory that 
# can be used to reinstall the same toolchain if needed.

(return 0 2>/dev/null) && is_sourced=true || is_sourced=false
__optind__=$OPTIND
OPTIND=1
toolchain=gcc12
while getopts ":t:e:" opt; do
  case $opt in
    t) toolchain="$OPTARG"
    ;;
    e) export_dir="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    if $is_sourced; then return 1; else exit 1; fi
    ;;
  esac
done
OPTIND=$__optind__

if [ "$(type -t temp_install_if_command_unknown)" != "function" ]; then
    function temp_install_if_command_unknown {
        if [ ! -x "$(command -v $1)" ]; then
            if [ -x "$(command -v apt-get)" ]; then
                if [ -z "$PKG_UNINSTALL" ]; then apt-get update; fi
                apt-get install -y --no-install-recommends $2
            elif [ -x "$(command -v dnf)" ]; then
                dnf install -y --nobest --setopt=install_weak_deps=False $2
            else
                echo "No package manager was found to install $2." >&2
            fi
            PKG_UNINSTALL="$PKG_UNINSTALL $2"
        fi
    }
fi

if [ "$(type -t find_executable)" != "function" ]; then
    function find_executable {
        find "${2:-/}" -path '*/bin*' -name $1 | head -1
    }
fi

if [ "${toolchain#gcc}" != "$toolchain" ]; then

    gcc_version=${toolchain#gcc}
    if [ -x "$(command -v apt-get)" ]; then
        apt-get update && apt-get install -y --no-install-recommends \
            gcc-$gcc_version g++-$gcc_version gfortran-$gcc_version

        CC="$(find_executable gcc-$gcc_version)" 
        CXX="$(find_executable g++-$gcc_version)" 
        FC="$(find_executable gfortran-$gcc_version)"

    elif [ -x "$(command -v dnf)" ]; then
        dnf install -y --nobest --setopt=install_weak_deps=False gcc-toolset-$gcc_version
        enable_script=`find / -path '*gcc*' -path '*'$gcc_version'*' -name enable` && . "$enable_script"
        gcc_root=`dirname "$enable_script"`

        CC="$(find_executable gcc "$gcc_root")"
        CXX="$(find_executable g++ "$gcc_root")"
        FC="$(find_executable gfortran "$gcc_root")"

    else
      echo "No supported package manager detected." >&2
    fi

elif [ "$toolchain" = "clang16" ]; then

    if [ -x "$(command -v apt-get)" ]; then
        temp_install_if_command_unknown wget wget
        temp_install_if_command_unknown gpg gnupg
        temp_install_if_command_unknown add-apt-repository software-properties-common

        wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
        add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main"
        apt-get update && apt-get install -y --no-install-recommends clang-16
    elif [ -x "$(command -v dnf)" ]; then
        dnf install -y --nobest --setopt=install_weak_deps=False clang-16.0.6
    else
        echo "No supported package manager detected." >&2
    fi

    CC="$(find_executable clang-16)" 
    CXX="$(find_executable clang++-16)" 
    FC="$(find_executable flang-new-16)"

elif [ "$toolchain" = "llvm" ]; then

    LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-"$HOME/.llvm"}
    if [ ! -f "$LLVM_INSTALL_PREFIX/bin/clang" ] || [ ! -f "$LLVM_INSTALL_PREFIX/bin/clang++" ] || [ ! -f "$LLVM_INSTALL_PREFIX/bin/ld.lld" ]; then

        this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
        if [ ! -d "$LLVM_SOURCE" ]; then
            mkdir -p "$HOME/.llvm_project"
            llvm_tmp_dir=`mktemp -d -p "$HOME/.llvm_project"` && LLVM_SOURCE="$llvm_tmp_dir"
            temp_install_if_command_unknown git git
            git clone -b main --single-branch --depth 1 https://github.com/llvm/llvm-project "$LLVM_SOURCE"
        fi

        if [ ! -x "$(command -v "$CC")" ] || [ ! -x "$(command -v "$CXX")" ]; then
            if [ -x "$(command -v apt-get)" ]; then
                temp_install_if_command_unknown gcc gcc
                temp_install_if_command_unknown g++ g++
            elif [ -x "$(command -v dnf)" ]; then
                temp_install_if_command_unknown gcc gcc
                temp_install_if_command_unknown g++ gcc-c++
            else
                echo -e "\e[01;31mError: Please define the environment variables CC and CXX.\e[0m" >&2
            fi
        fi

        temp_install_if_command_unknown ninja ninja-build
        temp_install_if_command_unknown cmake cmake
        LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
        LLVM_PROJECTS='clang;flang;lld;compiler-rt' \
        CC="$CC" CXX="$CXX" bash "$this_file_dir/build_llvm.sh" -s "$LLVM_SOURCE" -c Release -v
        if [ -d "$llvm_tmp_dir" ]; then
            echo "The build logs have been moved to $LLVM_INSTALL_PREFIX/logs."
            mkdir -p "$LLVM_INSTALL_PREFIX/logs" && mv "$llvm_tmp_dir/build/logs"/* "$LLVM_INSTALL_PREFIX/logs/"
            rm -rf "$llvm_tmp_dir"
        fi
    fi

    CC="$LLVM_INSTALL_PREFIX/bin/clang"
    CXX="$LLVM_INSTALL_PREFIX/bin/clang++"
    FC="$LLVM_INSTALL_PREFIX/bin/flang-new"

else

    echo "The requested toolchain cannot be installed by this script."
    echo "Supported toolchains: llvm, clang16, gcc12, gcc11."
    if $is_sourced; then return 1; else exit 1; fi

fi

if [ -n "$PKG_UNINSTALL" ]; then
    echo "Uninstalling packages used for bootstrapping: $PKG_UNINSTALL"
    if [ -x "$(command -v apt-get)" ]; then  
        apt-get remove -y $PKG_UNINSTALL
        apt-get autoremove -y --purge
    elif [ -x "$(command -v dnf)" ]; then
        dnf remove -y $PKG_UNINSTALL
        dnf clean all
    else
        echo "No package manager configured for clean up." >&2
    fi
    unset PKG_UNINSTALL
fi

if [ -x "$(command -v "$CC")" ] && [ -x "$(command -v "$CXX")" ]; then 
    export CC="$CC" && export CXX="$CXX" 
    echo "Installed $toolchain toolchain."
    if [ -x "$(command -v "$FC")" ]; then export FC="$FC"
    else unset FC && echo -e "\e[01;31mWarning: No fortran compiler installed.\e[0m" >&2
    fi

    if [ "$export_dir" != "" ]; then 
        mkdir -p "$export_dir"
        this_file=`readlink -f "${BASH_SOURCE[0]}"`
        cat "$this_file" > "$export_dir/install_toolchain.sh"
        env_variables="LLVM_INSTALL_PREFIX=$LLVM_INSTALL_PREFIX LLVM_SOURCE=$LLVM_SOURCE"
        echo "$env_variables source \"$export_dir/install_toolchain.sh\" -t $toolchain" > "$export_dir/init_command.sh"
    fi
else
    echo "Failed to install $toolchain toolchain."
    unset CC && unset CXX
    if $is_sourced; then return 10; else exit 10; fi
fi
