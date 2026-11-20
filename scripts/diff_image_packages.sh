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
#
# A package counts as added when the target has it at a version the base image
# does not have: new packages, auto-installed dependencies, and packages the
# build upgraded. Everything the target ships on top of the base needs source.
#
# Outputs in <output_dir>:
#   apt_package_details.tsv  binary<TAB>version<TAB>source<TAB>source_version
#   apt_packages.txt         binary package names (one per line)
#   apt_sources.txt          source=source_version lines, one per source package
#   apt_licenses/<binary>/   (name:arch is name_arch) license, copyright and EULA files the package owns
#   pip_packages.txt         package==version lines
#   pip_licenses/<name>/     license files from the installed distribution
#   images.txt               base and target image references with digests
#
# Usage:
#   ./scripts/diff_image_packages.sh <base_image> <target_image> <output_dir>

set -euo pipefail

BASE_IMAGE="${1:?Usage: $0 <base_image> <target_image> <output_dir>}"
TARGET_IMAGE="${2:?Usage: $0 <base_image> <target_image> <output_dir>}"
OUTPUT_DIR="${3:?Usage: $0 <base_image> <target_image> <output_dir>}"

mkdir -p "$OUTPUT_DIR"

# Override ENTRYPOINT so the image runs exactly "sh -c '...'" (NVIDIA images set ENTRYPOINT).
# -i forwards stdin; every caller either feeds it or redirects from /dev/null.
run_in() {
  docker run --rm -i --entrypoint sh "$1" -c "$2"
}

# binary<TAB>version<TAB>source<TAB>source_version for every installed package.
# An image without dpkg has no apt packages.
APT_DETAILS_SCRIPT='
command -v dpkg-query >/dev/null 2>&1 || exit 0
dpkg-query -W -f="\${db:Status-Abbrev}\t\${binary:Package}\t\${Version}\t\${source:Package}\t\${source:Version}\n" |
  awk -F"\t" "\$1 ~ /^ii/ {print \$2 \"\t\" \$3 \"\t\" \$4 \"\t\" \$5}"
'
get_apt_details() {
  run_in "$1" "$APT_DETAILS_SCRIPT" < /dev/null | sort -u
}

# Copies the files each listed package owns whose names look like a license,
# copyright or EULA. Proprietary packages keep these in /usr/share/doc or in
# their own prefix (e.g. /opt/mellanox/sharp), so ask dpkg what the package
# owns instead of guessing a location. Symlinks are followed.
APT_LICENSES_SCRIPT='
out=$(mktemp -d)
while IFS= read -r pkg; do
  [ -z "$pkg" ] && continue
  dpkg -L "$pkg" 2>/dev/null | grep -iE "(copyright|licen[cs]e|eula|copying|notice)[^/]*$" |
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    # name:arch -> name_arch; artifact paths cannot hold a colon
    d="$out/$(echo "$pkg" | tr : _)"
    mkdir -p "$d"
    cp -L "$f" "$d/$(echo "$f" | tr "/:\"<>|*?" _ | sed "s/^_//")" 2>/dev/null || true
  done
done
tar -C "$out" -cf - .
'

# package==version (freeze format) for reproducible source download. A missing
# Python is a valid empty result; a pip that is present but fails is an error.
PIP_LIST_SCRIPT='
py=$(command -v python3 || command -v python) || exit 0
"$py" -m pip --version >/dev/null 2>&1 || exit 0
"$py" -m pip list --format=freeze
'
get_pip_list() {
  local out
  out=$(run_in "$1" "$PIP_LIST_SCRIPT" < /dev/null) || { echo "pip list failed in image $1" >&2; return 1; }
  { grep -v '^#' <<< "$out" || true; } | { grep . || true; } | sort -u
}

# License files from the installed metadata of each listed distribution.
PIP_LICENSES_SCRIPT='
py=$(command -v python3 || command -v python) || exit 0
exec "$py" /collect_pip_licenses.py
'

image_ref() {
  local ref
  ref=$(docker image inspect --format '{{index .RepoDigests 0}}' "$1" 2>/dev/null) || true
  echo "${ref:-$1 (no repo digest)}"
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Collecting apt list from base image: $BASE_IMAGE"
get_apt_details "$BASE_IMAGE" > "$TMP/base_apt.tsv"
echo "Collecting apt list from target image: $TARGET_IMAGE"
get_apt_details "$TARGET_IMAGE" > "$TMP/target_apt.tsv"

# Added = (binary, version) in the target but not in the base.
awk -F'\t' 'NR==FNR {seen[$1 "\t" $2]=1; next} !(($1 "\t" $2) in seen)' \
  "$TMP/base_apt.tsv" "$TMP/target_apt.tsv" > "$OUTPUT_DIR/apt_package_details.tsv"
cut -f1 "$OUTPUT_DIR/apt_package_details.tsv" | sort -u > "$OUTPUT_DIR/apt_packages.txt"
awk -F'\t' '{print $3 "=" $4}' "$OUTPUT_DIR/apt_package_details.tsv" | sort -u > "$OUTPUT_DIR/apt_sources.txt"

mkdir -p "$OUTPUT_DIR/apt_licenses"
if [ -s "$OUTPUT_DIR/apt_packages.txt" ]; then
  run_in "$TARGET_IMAGE" "$APT_LICENSES_SCRIPT" < "$OUTPUT_DIR/apt_packages.txt" |
    tar -C "$OUTPUT_DIR/apt_licenses" -xf -
fi

echo "Collecting pip list from base image: $BASE_IMAGE"
get_pip_list "$BASE_IMAGE" > "$TMP/base_pip.txt"
echo "Collecting pip list from target image: $TARGET_IMAGE"
get_pip_list "$TARGET_IMAGE" > "$TMP/target_pip.txt"

# Added = package==version in the target but not in the base (new or upgraded).
{ grep -vxFf "$TMP/base_pip.txt" "$TMP/target_pip.txt" || true; } > "$OUTPUT_DIR/pip_packages.txt"

mkdir -p "$OUTPUT_DIR/pip_licenses"
if [ -s "$OUTPUT_DIR/pip_packages.txt" ]; then
  docker run --rm -i -v "$(cd "$(dirname "$0")" && pwd)/collect_pip_licenses.py:/collect_pip_licenses.py:ro" \
    --entrypoint sh "$TARGET_IMAGE" -c "$PIP_LICENSES_SCRIPT" \
    < "$OUTPUT_DIR/pip_packages.txt" | tar -C "$OUTPUT_DIR/pip_licenses" -xf -
fi

{
  echo "base_image: $(image_ref "$BASE_IMAGE")"
  echo "target_image: $(image_ref "$TARGET_IMAGE")"
} > "$OUTPUT_DIR/images.txt"

echo "Wrote $OUTPUT_DIR/apt_packages.txt ($(wc -l < "$OUTPUT_DIR/apt_packages.txt") packages, $(wc -l < "$OUTPUT_DIR/apt_sources.txt") source packages)"
echo "Wrote $OUTPUT_DIR/pip_packages.txt ($(wc -l < "$OUTPUT_DIR/pip_packages.txt") packages)"
