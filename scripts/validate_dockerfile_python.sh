#!/bin/bash
# Validate that the wheel build and auditwheel repair commands in
# docker/build/assets.Dockerfile use the configured PYTHON interpreter.
set -e

DOCKERFILE="docker/build/assets.Dockerfile"

# Strip comments and check for hardcoded python3 in the relevant commands.
if sed 's/#.*//' "$DOCKERFILE" | grep -n 'python3 -m build --wheel'; then
    echo "Error: Found hardcoded python3 in wheel build command" >&2
    exit 1
fi
if sed 's/#.*//' "$DOCKERFILE" | grep -n 'python3 -m auditwheel'; then
    echo "Error: Found hardcoded python3 in auditwheel command" >&2
    exit 1
fi

echo "OK: Dockerfile uses configured PYTHON for wheel build and repair"
