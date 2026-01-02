#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
    (return 0 2>/dev/null) && return 1 || exit 1
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
            elif [ -x "$(command -v brew)" ]; then
                HOMEBREW_NO_AUTO_UPDATE=1 brew install $2
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

# On macOS, always use system Apple Clang regardless of requested toolchain
if [[ "$(uname)" == "Darwin" ]]; then
    # Check if Xcode Command Line Tools are installed
    if ! xcode-select -p &>/dev/null; then
        echo "Xcode Command Line Tools are required on macOS."
        echo "Please run: xcode-select --install"
        echo "Then re-run this script."
        (return 0 2>/dev/null) && return 1 || exit 1
    fi

    if [ "$toolchain" != "llvm" ]; then
        echo "Note: On macOS, using system Apple Clang (requested: $toolchain)."
    fi
    CC="$(xcrun -f clang)"
    CXX="$(xcrun -f clang++)"
    unset FC  # No Fortran compiler by default on macOS

    if [ "$toolchain" = "llvm" ]; then
        # For llvm toolchain, build LLVM using Apple Clang as bootstrap compiler
        LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-"$HOME/.llvm"}
        if [ -f "$LLVM_INSTALL_PREFIX/bin/clang" ] && [ -f "$LLVM_INSTALL_PREFIX/bin/clang++" ]; then
            CC="$LLVM_INSTALL_PREFIX/bin/clang"
            CXX="$LLVM_INSTALL_PREFIX/bin/clang++"
            FC="$LLVM_INSTALL_PREFIX/bin/flang-new"
        else
            temp_install_if_command_unknown ninja ninja
            temp_install_if_command_unknown cmake cmake
            # Note: readlink -f doesn't work on macOS, use alternative
            this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" LLVM_PROJECTS='clang;lld;runtimes' \
            LLVM_SOURCE="$LLVM_SOURCE" LLVM_BUILD_FOLDER="$LLVM_BUILD_FOLDER" \
            CC="$CC" CXX="$CXX" bash "$this_file_dir/build_llvm.sh" -c Release -v
            if [ ! $? -eq 0 ]; then
                echo -e "\e[01;31mError: Failed to build LLVM toolchain from source.\e[0m" >&2
                (return 0 2>/dev/null) && return 3 || exit 3
            fi
            CC="$LLVM_INSTALL_PREFIX/bin/clang"
            CXX="$LLVM_INSTALL_PREFIX/bin/clang++"
            FC="$LLVM_INSTALL_PREFIX/bin/flang-new"
        fi
    fi

elif [ "${toolchain#gcc}" != "$toolchain" ]; then

    gcc_version=${toolchain#gcc}
    if [ -x "$(command -v apt-get)" ]; then
        apt-get update && apt-get install -y --no-install-recommends \
            gcc-$gcc_version g++-$gcc_version gfortran-$gcc_version \
            libstdc++-$gcc_version-dev build-essential

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

    elif [ -x "$(command -v brew)" ]; then
        HOMEBREW_NO_AUTO_UPDATE=1 brew install gcc@$gcc_version
        # For a specific version (e.g., gcc@13)
        CC=$(brew --prefix gcc@$gcc_version)/bin/gcc-$gcc_version
        CXX=$(brew --prefix gcc@$gcc_version)/bin/g++-$gcc_version
        FC=$(brew --prefix gcc@$gcc_version)/bin/gfortran-$gcc_version
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

        CC="$(find_executable clang-16)" 
        CXX="$(find_executable clang++-16)" 
        FC="$(find_executable flang-new-16)"

    elif [ -x "$(command -v dnf)" ]; then
        dnf install -y --nobest --setopt=install_weak_deps=False clang-16.0.6

        CC="$(find_executable clang-16)" 
        CXX="$(find_executable clang++-16)" 
        FC="$(find_executable flang-new-16)"
    elif [ -x "$(command -v brew)" ]; then
        HOMEBREW_NO_AUTO_UPDATE=1 brew install llvm@16
        CC="$(brew --prefix llvm@16)/bin/clang"
        CXX="$(brew --prefix llvm@16)/bin/clang++"
        FC="$(brew --prefix llvm@16)/bin/flang-new"
    else
        echo "No supported package manager detected." >&2
    fi

elif [ "$toolchain" = "llvm" ]; then

    LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-"$HOME/.llvm"}
    if [ ! -f "$LLVM_INSTALL_PREFIX/bin/clang" ] || [ ! -f "$LLVM_INSTALL_PREFIX/bin/clang++" ] || [ ! -f "$LLVM_INSTALL_PREFIX/bin/ld.lld" ]; then

        if [ ! -x "$(command -v "$CC")" ] || [ ! -x "$(command -v "$CXX")" ]; then
            # We use the clang to bootstrap the llvm build since it is faster than gcc.
            source "$(readlink -f "${BASH_SOURCE[0]}")" -t clang16 || \
            echo -e "\e[01;31mError: Failed to install clang compiler for bootstrapping.\e[0m" >&2
            toolchain=llvm
            if [ ! -x "$(command -v "$CC")" ] || [ ! -x "$(command -v "$CXX")" ]; then
                echo -e "\e[01;31mError: No compiler set for bootstrapping. Please define the environment variables CC and CXX.\e[0m" >&2
                (return 0 2>/dev/null) && return 2 || exit 2
            fi
        fi

        temp_install_if_command_unknown ninja ninja-build
        temp_install_if_command_unknown cmake cmake
        this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
        LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" LLVM_PROJECTS='clang;lld;runtimes' \
        LLVM_SOURCE="$LLVM_SOURCE" LLVM_BUILD_FOLDER="$LLVM_BUILD_FOLDER" \
        CC="$CC" CXX="$CXX" bash "$this_file_dir/build_llvm.sh" -c Release -v
        if [ ! $? -eq 0 ]; then 
            echo -e "\e[01;31mError: Failed to build LLVM toolchain from source.\e[0m" >&2
            (return 0 2>/dev/null) && return 3 || exit 3
        fi

        if [ -d "$llvm_tmp_dir" ]; then
            if [ -n "$(ls -A "$llvm_tmp_dir/build/logs"/* 2> /dev/null)" ]; then
                echo "The build logs have been moved to $LLVM_INSTALL_PREFIX/logs."
                mkdir -p "$LLVM_INSTALL_PREFIX/logs" && mv "$llvm_tmp_dir/build/logs"/* "$LLVM_INSTALL_PREFIX/logs/"
            fi
            rm -rf "$llvm_tmp_dir"
        fi
    fi

    CC="$LLVM_INSTALL_PREFIX/bin/clang"
    CXX="$LLVM_INSTALL_PREFIX/bin/clang++"
    FC="$LLVM_INSTALL_PREFIX/bin/flang-new"

else

    echo "The requested toolchain cannot be installed by this script."
    echo "Supported toolchains: llvm, clang16, gcc12, gcc11."
    (return 0 2>/dev/null) && return 1 || exit 1

fi

if [ -n "$PKG_UNINSTALL" ]; then
    echo "Uninstalling packages used for bootstrapping: $PKG_UNINSTALL"
    if [ -x "$(command -v apt-get)" ]; then  
        apt-get remove -y $PKG_UNINSTALL
        apt-get autoremove -y --purge
    elif [ -x "$(command -v dnf)" ]; then
        dnf remove -y $PKG_UNINSTALL
        dnf clean all
    elif [ -x "$(command -v brew)" ]; then
        brew uninstall --force $PKG_UNINSTALL 
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
        # Note: readlink -f doesn't work on macOS, use alternative
        this_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
        cat "$this_file" > "$export_dir/install_toolchain.sh"
        env_variables="LLVM_INSTALL_PREFIX=\"$LLVM_INSTALL_PREFIX\" LLVM_SOURCE=\"$LLVM_SOURCE\" LLVM_BUILD_FOLDER=\"$LLVM_BUILD_FOLDER\""
        echo "$env_variables source \"$export_dir/install_toolchain.sh\" -t $toolchain" > "$export_dir/init_command.sh"
    fi
else
    echo "Failed to install $toolchain toolchain."
    unset CC && unset CXX
    (return 0 2>/dev/null) && return 10 || exit 10
fi
