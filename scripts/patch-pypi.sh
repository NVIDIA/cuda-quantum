# copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -euo pipefail

# This script is used to patch the cudaq pypi published packages. For now,
# it is anticipated that one would use this script as a reference and update the
# MODIFY_ME{1,2,3} sections before using it.

# Note: you may need to run "python3 -m pip install -U wheel" first.

# MODIFY_ME1 - review and modify the following variables
PACKAGE_NAME=cudaq
ORIG_VER=0.12.0
NEW_VER=0.12.0.post1

# Ensure required commands are available
missing=()

for cmd in curl jq python3 wget; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    missing+=("$cmd")
  fi
done

if [ ${#missing[@]} -ne 0 ]; then
  echo "[ERROR] the following required commands are missing:"
  echo "  - ${missing[*]}"
  exit 1
fi

# Make a temporary directory to work in
TMP_DIR="/tmp/cudaq-scratch"
mkdir -p ${TMP_DIR}
echo "Building TMP_DIR in $TMP_DIR"
echo "Using temporary directory: $TMP_DIR"
mkdir -p wheels_new

# Be sure to clean up the temporary directory on exit
trap "rm -rf $TMP_DIR" EXIT

### ------------------------------------------------- ###
# Update cudaq metapackage
CUDAQ_METAPACKAGE_ORIG="wheels_orig_${PACKAGE_NAME}"
echo "Downloading the original cudaq wheels into ${CUDAQ_METAPACKAGE_ORIG}..."
mkdir -p ${CUDAQ_METAPACKAGE_ORIG} && \
curl -fsSL "https://pypi.org/pypi/${PACKAGE_NAME}/${ORIG_VER}/json" \
| jq -r '.urls[] | select(.packagetype=="sdist") | .url' \
| xargs -n1 -P4 -I{} wget -c -P ${CUDAQ_METAPACKAGE_ORIG} {}

tar -xvzf ${CUDAQ_METAPACKAGE_ORIG}/*.tar.gz -C ${TMP_DIR}
echo ${NEW_VER} > ${TMP_DIR}/*/_version.txt
cd ${TMP_DIR}/*/

# MODIFY_ME2 - review and modify the source code here
sed -i 's/elif cuda_version < 13000:/elif cuda_version <= 13000:/' setup.py

CUDAQ_META_WHEEL_BUILD=1 python3 -m build . --sdist
cd -

mv -v ${TMP_DIR}/*/dist/cudaq-*.tar.gz wheels_new

# upload cudaq metapackage with:
# python3 -m twine upload --repository testpypi wheels_new/*/dist/cudaq-0.12.0.post2.tar.gz --verbose


### ------------------------------------------------- ###
# modify cuda-quantum-cu* packages

# cuda-quantum-cu* ships actual wheels, so we need to modify them
for package in cuda-quantum-cu12 cuda-quantum-cu11; do
  PACKAGE_NAME_UNDER="${package//-/_}"
  orig_dir=wheels_orig_${PACKAGE_NAME_UNDER}

  # download wheels for this dist/version
  curl -fsSL "https://pypi.org/pypi/${package}/${ORIG_VER}/json" \
    | jq -r '.urls[] | select(.packagetype=="bdist_wheel") | .url' \
    | xargs -n1 -P4 -I{} wget -c -P "$orig_dir" {}

  for f in ${orig_dir}/*.whl; do
    python3 -m wheel unpack $f -d $TMP_DIR

    # --- Begin modifications
    # Update the version
    sed -i "s/^Version: ${ORIG_VER}/Version: ${NEW_VER}/" $TMP_DIR/${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${ORIG_VER}.dist-info/METADATA
    # MODIFY_ME3 - review and modify the METADATA file here
    # ...
    # --- End modifications

    # Re-package into a new whl file now
    cd $TMP_DIR
    mv ${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${ORIG_VER}.dist-info ${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${NEW_VER}.dist-info
    python3 -m wheel pack ${PACKAGE_NAME_UNDER}-${ORIG_VER} -d .
    cd -
    mv $TMP_DIR/${PACKAGE_NAME_UNDER}-${NEW_VER}*.whl wheels_new
    rm -rf $TMP_DIR
  done
done

# python3 -m twine upload --repository testpypi wheels_new/* --verbose
