# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Lock-file generation for install_prerequisites.sh -l. It lives in its own
# file so that edits to it do not invalidate the Docker layers that build
# LLVM, which need prereqs_common.sh but never run in lock mode.
#
# Not meant to be run directly -- sourced from prereqs_common.sh, which has
# defined the library versions and parsed the arguments by then.

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
# url is the primary; mirrors lists the same fallbacks download_first uses.
add_lock_line "gmp" \
  "type=tar" \
  "url=${GMP_TARBALL_URLS%% *}" \
  "mirrors=$(echo ${GMP_TARBALL_URLS#* } | tr ' ' ',')" \
  "version=${GMP_VERSION}"
add_lock_line "mpfr" \
  "type=tar" \
  "url=${MPFR_TARBALL_URLS%% *}" \
  "mirrors=$(echo ${MPFR_TARBALL_URLS#* } | tr ' ' ',')" \
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
# Keep this in sync with the QRMI section in install_prerequisites.sh.
add_lock_line "qrmi" \
  "type=tar" \
  "url=${QRMI_RELEASE_BASE}/${QRMI_ARCHIVE}" \
  "version=${QRMI_RELEASE_VERSION}" \
  "sha256=${QRMI_ARCHIVE_SHA256}"

echo "Prerequisites lockfile written to ${LOCK_FILE}."
# Don't exit here: prereqs_common.sh still has to define its helper
# functions. install_prerequisites.sh stops itself once it sees this flag.
prereqs_lock_mode_done=true
