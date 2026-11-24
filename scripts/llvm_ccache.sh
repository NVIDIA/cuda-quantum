#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Restores the ccache directory of an LLVM build from a registry before it
# starts, and saves it back when it ends. A build whose Docker layer cache
# misses, whatever the reason, then recompiles only what changed.
#
# Meant to be sourced in the RUN instruction that builds LLVM, with the
# configuration mounted as a secret. A secret takes no part in the layer's
# cache key, and a mount from the build context would, so the layer cache
# keeps working. Nothing is done when the secret is missing (local builds):
#
#   RUN --mount=type=secret,id=llvm_ccache,required=false \
#       source scripts/llvm_ccache.sh && llvm_ccache_begin <flavor> [<compiler check>] && \
#       <build LLVM>
#
# The registry credentials are kept in a file of the temporary directory, never
# in ~/.docker, which would end up in the layer. The file is removed as soon as
# the restore is done, and again when the build ends.
#
# llvm_ccache_begin sets an EXIT trap that calls llvm_ccache_end, so that a
# failed build also saves what it compiled. The exit status of the build is
# never changed, and a problem with the registry never fails the build.
# bash 4 (AlmaLinux 8) replaces itself with the last command of a `-c` string
# and never runs the trap, so end the RUN with `&& true`.
#
# The configuration is a file of key=value lines:
#   user=<registry user>             token=<registry token>
#   pull=<repository:suffix> ...     tried in order, the first one found is used
#   push=<repository:suffix>         where to save; empty to only restore
#   plain_http=true                  for a registry without TLS (tests)
# The tag is <flavor>-<amd64|arm64>-<suffix>, as the ccache of a toolchain
# cannot be used by another.
#
# llvm_ccache_begin also sets the ccache options of the LLVM build, in the
# shell of the RUN instruction only, so that nothing leaks into the image or
# into later stages: the size is limited so that the archive stays well below
# what GHCR accepts, and the compiler is identified by the hash of its binary,
# as the stage-1 compiler of a bootstrapped build is installed at a different
# path each time. A compiler that is a wrapper script needs the second
# argument, e.g. '%compiler% -v', which asks the compiler for its version.
#
# Environment:
#   CCACHE_DIR                       where ccache keeps its data (default ~/.ccache)
#   LLVM_CCACHE_CONFIG               the configuration file (default: the secret)
#   LLVM_CCACHE_MIN_MISSES           save only when the build had at least this
#                                    many cache misses (default: 100)
#   LLVM_CCACHE_MAX_BYTES            do not save a larger archive (default: 5.5 GB,
#                                    GHCR accepts blobs of up to 10 GB)

LLVM_CCACHE_CONFIG=${LLVM_CCACHE_CONFIG:-/run/secrets/llvm_ccache}
LLVM_CCACHE_MIN_MISSES=${LLVM_CCACHE_MIN_MISSES:-100}
LLVM_CCACHE_MAX_BYTES=${LLVM_CCACHE_MAX_BYTES:-5500000000}
LLVM_CCACHE_ORAS_VERSION=1.2.2

_llvm_ccache_get() {
  sed -n "s/^$1=//p" "$LLVM_CCACHE_CONFIG" | head -1
}

_llvm_ccache_download() { # <url> <file>
  local n
  for n in 1 2 3; do
    if command -v curl > /dev/null 2>&1; then
      curl -fsSL --connect-timeout 10 --max-time 300 -o "$2" "$1" && return 0
    else
      wget -q --tries=1 --timeout=60 -O "$2" "$1" && return 0
    fi
    sleep $((n * 5))
  done
  return 1
}

# Puts oras in $LLVM_CCACHE_TMP/bin, checking the release against its checksum.
_llvm_ccache_install_oras() {
  command -v oras > /dev/null 2>&1 && return 0
  local arch sha version=$LLVM_CCACHE_ORAS_VERSION
  case "$(uname -m)" in
    x86_64 | amd64)
      arch=amd64
      sha=bff970346470e5ef888e9f2c0bf7f8ee47283f5a45207d6e7a037da1fb0eae0d ;;
    aarch64 | arm64)
      arch=arm64
      sha=edd7195cbb8ba56c29ede413eefa10c8026201d63326017cd315841b4063aa56 ;;
    *) return 1 ;;
  esac
  local tarball="$LLVM_CCACHE_TMP/oras.tar.gz"
  _llvm_ccache_download \
    "https://github.com/oras-project/oras/releases/download/v${version}/oras_${version}_linux_${arch}.tar.gz" \
    "$tarball" || return 1
  echo "$sha  $tarball" | sha256sum -c - > /dev/null || return 1
  mkdir -p "$LLVM_CCACHE_TMP/bin"
  tar -xzf "$tarball" -C "$LLVM_CCACHE_TMP/bin" oras || return 1
  export PATH="$LLVM_CCACHE_TMP/bin:$PATH"
}

# ORAS options naming the credentials file, set once the temporary directory exists.
_llvm_ccache_auth() {
  echo "--registry-config $LLVM_CCACHE_TMP/auth.json"
}

_llvm_ccache_misses() {
  local n
  n=$(ccache --print-stats 2> /dev/null | awk '$1 == "cache_miss" {print $2; exit}')
  if [ -z "$n" ]; then
    n=$(ccache -s 2> /dev/null | awk '/^cache miss/ {print $NF; exit} /^ *Misses:/ {print $2; exit}')
  fi
  echo "${n:-0}"
}

_llvm_ccache_restore() {
  local ref tag
  mkdir -p "$CCACHE_DIR"
  if [ -n "$(ls -A "$CCACHE_DIR" 2> /dev/null)" ]; then
    echo "llvm-ccache: $CCACHE_DIR is not empty (an earlier step of this builder filled it); not restoring."
    _lc_restored="the cache mount"
    return 0
  fi
  _llvm_ccache_install_oras || { echo "llvm-ccache: could not install oras; building without a restored ccache."; return 1; }
  [ -n "$_lc_token" ] && { echo "$_lc_token" | oras login "${_lc_host}" -u "$_lc_user" --password-stdin $(_llvm_ccache_auth) $_lc_http > /dev/null 2>&1 || echo "llvm-ccache: login to $_lc_host failed."; }
  for ref in $_lc_pull; do
    tag="${ref%:*}:${_lc_flavor}-${_lc_arch}-${ref##*:}"
    rm -rf "$LLVM_CCACHE_TMP/pull"
    if oras pull "$tag" -o "$LLVM_CCACHE_TMP/pull" $(_llvm_ccache_auth) $_lc_http > /dev/null 2>&1 \
      && tar -xf "$LLVM_CCACHE_TMP/pull/ccache-data.tar" -C "$CCACHE_DIR"; then
      echo "llvm-ccache: restored $tag ($(du -sh "$CCACHE_DIR" | cut -f1), $(find "$CCACHE_DIR" -type f | wc -l) files)."
      _lc_restored=$tag
      rm -rf "$LLVM_CCACHE_TMP/pull" "$LLVM_CCACHE_TMP/auth.json"
      return 0
    fi
    echo "llvm-ccache: nothing usable at $tag."
    find "$CCACHE_DIR" -mindepth 1 -delete 2> /dev/null
  done
  rm -f "$LLVM_CCACHE_TMP/auth.json"
  return 0
}

llvm_ccache_begin() { # <flavor> [<compiler check>]
  export CCACHE_DIR=${CCACHE_DIR:-$HOME/.ccache}
  export CCACHE_MAXSIZE=5G
  export CCACHE_COMPRESS=1
  export CCACHE_COMPILERCHECK=${2:-content}
  export CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime
  if [ ! -r "$LLVM_CCACHE_CONFIG" ]; then
    echo "llvm-ccache: disabled, no configuration at $LLVM_CCACHE_CONFIG."
    return 0
  fi
  if ! command -v ccache > /dev/null 2>&1; then
    echo "llvm-ccache: disabled, ccache is not installed."
    return 0
  fi
  _lc_flavor=$1
  case "$(uname -m)" in
    x86_64 | amd64) _lc_arch=amd64 ;;
    aarch64 | arm64) _lc_arch=arm64 ;;
    *) _lc_arch=$(uname -m) ;;
  esac
  _lc_user=$(_llvm_ccache_get user)
  _lc_token=$(_llvm_ccache_get token)
  _lc_pull=$(_llvm_ccache_get pull)
  _lc_push=$(_llvm_ccache_get push)
  _lc_http=
  [ "$(_llvm_ccache_get plain_http)" = true ] && _lc_http=--plain-http
  _lc_host=${_lc_pull%%/*}
  [ -n "$_lc_push" ] && _lc_host=${_lc_push%%/*}
  _lc_restored=
  LLVM_CCACHE_TMP=$(mktemp -d)
  _lc_active=true
  trap 'llvm_ccache_end $?' EXIT
  echo "llvm-ccache: $_lc_flavor-$_lc_arch, ccache $(ccache --version | head -1 | sed 's/.* //'), CCACHE_DIR=$CCACHE_DIR."
  _llvm_ccache_restore || true
  ccache -z > /dev/null 2>&1
  return 0
}

llvm_ccache_end() { # <exit status of the build>
  local status=${1:-0} misses tag
  [ "$_lc_active" = true ] || return "$status"
  _lc_active=false
  trap - EXIT
  echo "llvm-ccache: the build exited with status $status."
  ccache -s 2>&1 | sed 's/^/llvm-ccache:   /'
  misses=$(_llvm_ccache_misses)
  if [ -z "$_lc_push" ]; then
    echo "llvm-ccache: not saving, no destination configured."
  elif [ -n "$_lc_restored" ] && [ "$misses" -lt "$LLVM_CCACHE_MIN_MISSES" ]; then
    echo "llvm-ccache: not saving, $misses cache misses is below $LLVM_CCACHE_MIN_MISSES."
  elif [ "$misses" -eq 0 ]; then
    echo "llvm-ccache: not saving, nothing was compiled."
  else
    tag="${_lc_push%:*}:${_lc_flavor}-${_lc_arch}-${_lc_push##*:}"
    echo "llvm-ccache: saving $misses new results to $tag."
    {
      _llvm_ccache_install_oras \
        && { [ -z "$_lc_token" ] || echo "$_lc_token" | oras login "$_lc_host" -u "$_lc_user" --password-stdin $(_llvm_ccache_auth) $_lc_http > /dev/null 2>&1; } \
        && ccache --cleanup > /dev/null 2>&1 \
        && tar -cf "$LLVM_CCACHE_TMP/ccache-data.tar" -C "$CCACHE_DIR" . \
        && { [ "$(stat -c %s "$LLVM_CCACHE_TMP/ccache-data.tar")" -le "$LLVM_CCACHE_MAX_BYTES" ] \
          || { echo "llvm-ccache: the archive is larger than $LLVM_CCACHE_MAX_BYTES bytes; raise CCACHE_MAXSIZE or LLVM_CCACHE_MAX_BYTES."; false; }; } \
        && (cd "$LLVM_CCACHE_TMP" && oras push "$tag" ccache-data.tar:application/x-tar $(_llvm_ccache_auth) $_lc_http > /dev/null) \
        && echo "llvm-ccache: saved $tag ($(du -h "$LLVM_CCACHE_TMP/ccache-data.tar" | cut -f1))."
    } || echo "llvm-ccache: saving $tag failed; the build is not affected."
  fi
  rm -rf "$LLVM_CCACHE_TMP"
  return "$status"
}
