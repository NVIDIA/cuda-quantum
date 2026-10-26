#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Updates the git submodules, retrying with backoff and falling back to the
# mirrors below when a host such as gitlab.com is down.
#
# Usage: update_submodules.sh [-c <key>=<value>]... [<submodule update args>]
# Env: SUBMODULE_UPDATE_ATTEMPTS (default 5), SUBMODULE_UPDATE_DELAY (default 15s)

# Primary URL and the mirror to use when it is unreachable.
mirrors='
https://gitlab.com/libeigen/eigen.git https://github.com/eigen-mirror/eigen.git
https://gitlab.com/conradsnicta/armadillo-code.git https://github.com/conradsnicta/armadillo-code.git
'

git_options=()
while [ "${1:-}" = "-c" ] && [ $# -ge 2 ]; do
  git_options+=(-c "$2")
  shift 2
done

mirror_options=()
while read -r primary mirror; do
  [ -n "$primary" ] && mirror_options+=(-c "url.$mirror.insteadOf=$primary")
done <<< "$mirrors"

attempts=${SUBMODULE_UPDATE_ATTEMPTS:-5}
delay=${SUBMODULE_UPDATE_DELAY:-15}
update=(submodule update --init --recursive "$@")

for attempt in $(seq 1 "$attempts"); do
  git ${git_options[@]+"${git_options[@]}"} "${update[@]}" && exit 0
  echo "Submodule update failed (attempt $attempt of $attempts), trying the mirrors" >&2
  git ${git_options[@]+"${git_options[@]}"} "${mirror_options[@]}" "${update[@]}" && exit 0
  if [ "$attempt" -lt "$attempts" ]; then
    echo "Submodule update failed (attempt $attempt of $attempts), retrying in ${delay}s" >&2
    sleep "$delay"
    delay=$((delay * 2))
  fi
done

echo "Submodule update failed after $attempts attempts" >&2
exit 1
