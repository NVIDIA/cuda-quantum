# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Diffs apt and pip packages between a base image and a target image.
# The target image is assumed to be built FROM the base image.
# Outputs:
#   - apt_added: apt package names installed in target but not in base
#   - pip_added: pip package==version lines installed in target but not in base
#
# Usage:
#   ./scripts/diff_image_packages.sh <base_image> <target_image> <output_dir>
#   Writes <output_dir>/apt_packages.txt and <output_dir>/pip_packages.txt

set -euo pipefail

BASE_IMAGE="${1:?Usage: $0 <base_image> <target_image> <output_dir>}"
TARGET_IMAGE="${2:?Usage: $0 <base_image> <target_image> <output_dir>}"
OUTPUT_DIR="${3:?Usage: $0 <base_image> <target_image> <output_dir>}"

mkdir -p "$OUTPUT_DIR"

# Get apt list: manually installed package names (works in minimal images)
get_apt_list() {
  docker run --rm "$1" sh -c \
    'apt-mark showmanual 2>/dev/null | sort -u' || true
}

# Get pip list: package==version (freeze format) for reproducible source download
get_pip_list() {
  docker run --rm "$1" sh -c \
    'pip list --format=freeze 2>/dev/null || python3 -m pip list --format=freeze 2>/dev/null || true' | \
    grep -v '^#' | sort -u || true
}

BASE_APT=$(mktemp)
TARGET_APT=$(mktemp)
BASE_PIP=$(mktemp)
TARGET_PIP=$(mktemp)
trap "rm -f $BASE_APT $TARGET_APT $BASE_PIP $TARGET_PIP" EXIT

echo "Collecting apt list from base image: $BASE_IMAGE"
get_apt_list "$BASE_IMAGE" > "$BASE_APT"
echo "Collecting apt list from target image: $TARGET_IMAGE"
get_apt_list "$TARGET_IMAGE" > "$TARGET_APT"

echo "Collecting pip list from base image: $BASE_IMAGE"
get_pip_list "$BASE_IMAGE" > "$BASE_PIP"
echo "Collecting pip list from target image: $TARGET_IMAGE"
get_pip_list "$TARGET_IMAGE" > "$TARGET_PIP"

# Apt diff: package names in target but not in base
comm -13 <(sort -u "$BASE_APT") <(sort -u "$TARGET_APT") | grep -v '^$' > "$OUTPUT_DIR/apt_packages.txt" || true

# Pip diff: package==version in target but not in base (by package name)
# We compare by package name so we get "added" packages; for versions we use target's version
BASE_PIP_NAMES=$(mktemp)
trap "rm -f $BASE_APT $TARGET_APT $BASE_PIP $TARGET_PIP $BASE_PIP_NAMES" EXIT
awk -F'==' '{print $1}' "$BASE_PIP" | sort -u > "$BASE_PIP_NAMES"
# Lines in TARGET_PIP whose package name is not in BASE_PIP
while IFS= read -r line; do
  pkg="${line%%==*}"
  if ! grep -qxF "$pkg" "$BASE_PIP_NAMES" 2>/dev/null; then
    echo "$line"
  fi
done < "$TARGET_PIP" > "$OUTPUT_DIR/pip_packages.txt" || true

echo "Wrote $OUTPUT_DIR/apt_packages.txt ($(wc -l < "$OUTPUT_DIR/apt_packages.txt" || echo 0) packages)"
echo "Wrote $OUTPUT_DIR/pip_packages.txt ($(wc -l < "$OUTPUT_DIR/pip_packages.txt" || echo 0) packages)"
