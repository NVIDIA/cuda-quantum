#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# cudaq_repro_wrap.sh - wrap a test/build invocation inside a Dockerfile RUN
# step so failures emit a copy-pasteable reproduction block to stderr before
# propagating the exit code.
#
# Usage:
#   cudaq_repro_wrap.sh <image-or-tag-hint> -- <command...>
#
# Example (inside a Dockerfile):
#   RUN bash /usr/local/bin/cudaq_repro_wrap.sh \
#       "ghcr.io/nvidia/cuda-quantum-dev:cu12.6-gcc12-main" \
#       -- ctest --output-on-failure --test-dir build
#
# On non-zero exit, prints a CUDAQ_REPRO_TEST banner block that contains:
#   - the image tag/digest the user can `docker pull` to get the same image
#   - the exact command that failed
#   - instructions for dropping into a shell to debug
#
# Banner format MUST match the one consumed by ci.yml's CI Summary deep-link
# grep: do not change `>>> CUDAQ_REPRO_TEST <<<` / `<<< CUDAQ_REPRO_TEST >>>`
# without also updating the consumer.

set -uo pipefail

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
  cat >&2 <<EOF
Usage: $0 <image-tag-or-digest> -- <command...>
Got args: $*
EOF
  exit 2
fi

image="$1"
shift 2  # drop image + '--'

# Run the command, capture its exit code without crashing this wrapper.
"$@"
status=$?

if [ "$status" -ne 0 ]; then
  # Re-quote the failed command for copy-paste safety. printf %q handles
  # spaces, special chars, and embedded quotes correctly for bash.
  cmd_quoted=""
  for arg in "$@"; do
    cmd_quoted+=" $(printf %q "$arg")"
  done

  {
    echo ''
    echo '>>> CUDAQ_REPRO_TEST <<<'
    echo '## Reproduce this test failure locally'
    echo ''
    echo '```bash'
    echo "# Pull the same image (use the digest above for byte-identical reproduction)"
    echo "docker pull $image"
    echo ''
    echo "# Drop into a shell inside the image"
    echo "docker run --rm -it --entrypoint bash $image"
    echo ''
    echo "# Then, inside the container, re-run the failing command:"
    echo "$cmd_quoted"
    echo '```'
    echo ''
    echo "Failed with exit code: $status"
    echo '<<< CUDAQ_REPRO_TEST >>>'
  } >&2
fi

exit "$status"
