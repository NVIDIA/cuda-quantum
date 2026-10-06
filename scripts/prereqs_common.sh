# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Shared by install_prerequisites.sh and bootstrap_prerequisites.sh: library
# version/source definitions, command line argument parsing, lock-file
# generation, and common helper functions (retry, download_first,
# temp_install_if_command_unknown, remove_temp_installs, prepare_exit).
#
# Not meant to be run directly -- sourced from one of the two scripts above,
# which set up the per-toolchain installation steps that follow.

# Centralized version / source definitions used by both installation and lockfile
# generation. Keeping these here avoids duplication between code paths.
CMAKE_VERSION=4.0.7
CMAKE_MACOS_TARBALL_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-macos-universal.tar.gz"
CMAKE_LINUX_INSTALLER_URL_BASE="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-"

NINJA_VERSION=1.11.1
NINJA_TARBALL_URL="https://github.com/ninja-build/ninja/archive/refs/tags/v${NINJA_VERSION}.tar.gz"

ZLIB_VERSION=1.3.2
ZLIB_TARBALL_URL="https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz"

BLAS_VERSION=3.11.0
BLAS_TARBALL_URL="https://www.netlib.org/blas/blas-${BLAS_VERSION}.tgz"

# GMP and MPFR back the Clifford+T rotation synthesis library (cudaq-synth).
# Both are LGPL v3 (see https://gmplib.org/ and https://www.mpfr.org/). They
# are built as shared libraries only and linked dynamically.
GMP_VERSION=6.3.0
GMP_TARBALL_URLS="https://ftp.gnu.org/gnu/gmp/gmp-${GMP_VERSION}.tar.xz \
https://gmplib.org/download/gmp/gmp-${GMP_VERSION}.tar.xz"

MPFR_VERSION=4.2.2
MPFR_TARBALL_URLS="https://ftp.gnu.org/gnu/mpfr/mpfr-${MPFR_VERSION}.tar.xz \
https://www.mpfr.org/mpfr-${MPFR_VERSION}/mpfr-${MPFR_VERSION}.tar.xz"

PERL_VERSION=5.38.2
PERL_TARBALL_URL="https://www.cpan.org/src/5.0/perl-${PERL_VERSION}.tar.gz"

OPENSSL_VERSION=3.6.3
OPENSSL_TARBALL_URL="https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz"

CURL_VERSION=8.21.0
CURL_VERSION_UNDERSCORE=curl-8_21_0
CURL_TARBALL_URL="https://github.com/curl/curl/releases/download/${CURL_VERSION_UNDERSCORE}/curl-${CURL_VERSION}.tar.gz"
CACERT_URL="https://curl.se/ca/cacert.pem"
CACERT_SHA256_URL="${CACERT_URL}.sha256"

AWS_SDK_CPP_URL="https://github.com/aws/aws-sdk-cpp"
AWS_SDK_CPP_REF="1.11.454"

# QRMI pre-built C artifacts for Pasqal QRMI connector
QRMI_RELEASE_REPO=${QRMI_RELEASE_REPO:-qiskit-community/qrmi}
QRMI_RELEASE_TAG=${QRMI_RELEASE_TAG:-v0.12.0}
QRMI_RELEASE_VERSION=${QRMI_RELEASE_TAG#v}
QRMI_RELEASE_BASE="https://github.com/${QRMI_RELEASE_REPO}/releases/download/${QRMI_RELEASE_TAG}"
QRMI_ARCHIVE="libqrmi-${QRMI_RELEASE_VERSION}-el8-x86_64.tar.gz"
QRMI_UNPACK_DIR="libqrmi-${QRMI_RELEASE_VERSION}"
# NOTE: This needs to be updated whenever the pre-built artifacts are updated. The SHA-256 can be computed with:
#   wget -O qrmi.tar.gz "${QRMI_RELEASE_BASE}/${QRMI_ARCHIVE}"
#   sha256sum qrmi.tar.gz | awk '{print $1}'
QRMI_ARCHIVE_SHA256=${QRMI_ARCHIVE_SHA256:-2986150d4f55e1f6566bef16d9fb3897ca04dd7eaa681865f7ef244f298a6746}

# Process command line arguments
toolchain=''
exclude_prereq=''
install_all=true
lock_mode=false
prereqs_lock_mode_done=false
__optind__=$OPTIND
OPTIND=1
while getopts ":e:t:ml-:" opt; do
  case $opt in
    e) exclude_prereq="$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')"
    ;;
    t) toolchain="$OPTARG"
    ;;
    m) install_all=false
    ;;
    l) lock_mode=true
    ;;
    :) echo "Option -$OPTARG requires an argument."
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
  esac
done
OPTIND=$__optind__

# Set default install prefix environment variables (only when install_all is true)
if $install_all; then
  source "$(dirname "${BASH_SOURCE[0]}")/set_env_defaults.sh"
fi

# If requested, generate a lock file describing all source archives / repositories
# that would be used to build the prerequisites, then exit without installing.
if $lock_mode; then
  LOCK_FILE="${PREREQS_LOCK_FILE:-cudaq_prereqs.lock}"

  # Helper to append one entry to the lock file in a simple key=value format.
  function add_lock_line {
    local name="$1"; shift
    echo "name=${name} $*" >> "$LOCK_FILE"
  }

  # Initialize / truncate the lock file and add a short header.
  {
    echo "# CUDA-Q prerequisites lockfile"
    echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "# Format: name=<id> key=value ..."
  } > "$LOCK_FILE"

  # [Toolchain] CMake and Ninja sources (compiler toolchain itself is handled
  # via install_toolchain.sh or the system toolchain and is not pinned here).
  # In lockfile mode, always list the toolchain sources regardless of what is
  # currently installed or excluded.
  add_lock_line "cmake-macos" \
    "type=tar" \
    "url=${CMAKE_MACOS_TARBALL_URL}" \
    "version=${CMAKE_VERSION}"
  add_lock_line "cmake" \
    "type=sh" \
    "url=${CMAKE_LINUX_INSTALLER_URL_BASE}$(uname -m).sh" \
    "version=${CMAKE_VERSION}"
  add_lock_line "ninja" \
    "type=tar" \
    "url=${NINJA_TARBALL_URL}" \
    "version=${NINJA_VERSION}"

  # [Zlib]
  add_lock_line "zlib" \
    "type=tar" \
    "url=${ZLIB_TARBALL_URL}" \
    "version=${ZLIB_VERSION}"

  # [BLAS]
  add_lock_line "blas" \
    "type=tar" \
    "url=${BLAS_TARBALL_URL}" \
    "version=${BLAS_VERSION}"

  # [GMP / MPFR]
  add_lock_line "gmp" \
    "type=tar" \
    "url=${GMP_TARBALL_URLS%% *}" \
    "version=${GMP_VERSION}"
  add_lock_line "mpfr" \
    "type=tar" \
    "url=${MPFR_TARBALL_URLS%% *}" \
    "version=${MPFR_VERSION}"

  # [OpenSSL] (and its private Perl used only for the build)
  add_lock_line "perl" \
    "type=tar" \
    "url=${PERL_TARBALL_URL}" \
    "version=${PERL_VERSION}"
  add_lock_line "openssl" \
    "type=tar" \
    "url=${OPENSSL_TARBALL_URL}" \
    "version=${OPENSSL_VERSION}"

  # [CURL] (including CA bundle)
  add_lock_line "cacert" \
    "type=pem" \
    "url=${CACERT_URL}"
  add_lock_line "curl" \
    "type=tar" \
    "url=${CURL_TARBALL_URL}" \
    "version=${CURL_VERSION}"

  # [AWS SDK]
  add_lock_line "aws-sdk-cpp" \
    "type=git" \
    "url=${AWS_SDK_CPP_URL}" \
    "ref=${AWS_SDK_CPP_REF}"

  # [QRMI] Pre-built C artifacts for Pasqal QRMI connector
  # Keep this in sync with the QRMI section in the installation path below.
  add_lock_line "qrmi" \
    "type=tar" \
    "url=${QRMI_RELEASE_BASE}/${QRMI_ARCHIVE}" \
    "version=${QRMI_RELEASE_VERSION}" \
    "sha256=${QRMI_ARCHIVE_SHA256}"

  echo "Prerequisites lockfile written to ${LOCK_FILE}."
  # Don't exit/return here: this file is always sourced, so `return` would
  # only unwind out of this source call, skipping the function definitions
  # below and falling through into the caller's real installation logic.
  # Let the caller stop itself once this file finishes sourcing instead.
  prereqs_lock_mode_done=true
fi

# Create a temporary directory for building source packages
PREREQS_BUILD_DIR=$(mktemp -d)
: "${PREREQS_BUILD_DIR:?ERROR mktemp failed}"
echo "Building prerequisites in $PREREQS_BUILD_DIR"

# Retry a command, clearing package-manager metadata between attempts. The CUDA
# yum repo CDN intermittently serves a stale repomd.xml that points at rotated
# repodata files, producing 404s; clearing metadata forces a fresh fetch.
function retry {
  local n=0 max=5 delay=15
  until "$@"; do
    n=$((n+1))
    if [ "$n" -ge "$max" ]; then
      echo "Command failed after $max attempts: $*" >&2
      return 1
    fi
    echo "Attempt $n/$max failed; clearing repo metadata and retrying in ${delay}s..." >&2
    if [ -x "$(command -v dnf)" ]; then dnf clean all || true
    elif [ -x "$(command -v apt-get)" ]; then apt-get clean || true; fi
    sleep "$delay"
  done
}


function download_first {
  local filename="$1"; shift
  for url in "$@"; do
    echo "Downloading ${url}..."
    if retry wget --tries=1 -O "${filename}" "${url}"; then
      return 0
    fi
    echo "Failed to download from ${url}; trying next mirror..." >&2
  done
  rm -f "${filename}"
  echo "Failed to download from all mirrors: $*" >&2
  return 1
}

function temp_install_if_command_unknown {
  if [ ! -x "$(command -v $1)" ]; then
    if [ -x "$(command -v apt-get)" ]; then
      if [ -z "$PKG_UNINSTALL" ]; then retry apt-get update; fi
      retry apt-get install -y --no-install-recommends $2
    elif [ -x "$(command -v dnf)" ]; then
      retry dnf install -y --nobest --setopt=install_weak_deps=False $2
    elif [ -x "$(command -v brew)" ]; then
      HOMEBREW_NO_AUTO_UPDATE=1 brew install $2
    else
      echo "No package manager was found to install $2." >&2
    fi
    PKG_UNINSTALL="$PKG_UNINSTALL $2"
  fi
}

function remove_temp_installs {
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
}

working_dir=`pwd`
read __errexit__ < <(echo $SHELLOPTS | grep -Eo '(^|:)errexit(:|$)' || echo)
function prepare_exit {
  cd "$working_dir" && remove_temp_installs
  # Comment out to debug pre-req build failures.
  rm -rf "$PREREQS_BUILD_DIR"
  if [ -z "$__errexit__" ]; then set +e; fi
}

set -e
trap 'prepare_exit && ((return 0 2>/dev/null) && return 1 || exit 1)' EXIT
this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(uname)" = "Darwin" ] && [ -x "$(command -v xcrun)" ]; then
  export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"
fi
