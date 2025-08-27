# copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script is used to patch the wheel metadata for a PyPI package. For now,
# it is anticipated that one would use this script as a reference and update the
# MODIFY_ME1 and MODIFY_ME2 sections before using it.

# Note: you may need to run "python3 -m pip install -U wheel" first.

# MODIFY_ME1 - review and modify the following variables
PACKAGE_NAME=cudaq
ORIG_VER=0.12.0
NEW_VER=0.12.0.post0

# Make sure that curl, jq, python3, and wget are installed.
if ! command -v curl &> /dev/null; then
  echo "curl could not be found"
  exit 1
fi
if ! command -v jq &> /dev/null; then
  echo "jq could not be found"
  exit 1
fi
if ! command -v python3 &> /dev/null; then
  echo "python3 could not be found"
  exit 1
fi
if ! command -v wget &> /dev/null; then
  echo "wget could not be found"
  exit 1
fi

# Make a temporary directory to work in
TMP_DIR=$(mktemp -d)
echo "Building TMP_DIR in $TMP_DIR"
echo "Using temporary directory: $TMP_DIR"

# # Be sure to clean up the temporary directory on exit
# trap "rm -rf $TMP_DIR" EXIT

echo "Downloading the original wheels into wheels_orig..."
mkdir -p wheels_orig && \
curl -fsSL "https://pypi.org/pypi/${PACKAGE_NAME}/${ORIG_VER}/json" \
| jq -r '.urls[] | select(.packagetype=="sdist") | .url' \
| xargs -n1 -P4 -I{} wget -c -P wheels_orig {}

mkdir -p wheels_new

echo "Placing the patched source into wheels_new..."
tar -xvzf wheels_orig/*.tar.gz -C wheels_new

# at this point, we need to update the version in the setup.py or pyproject.toml
echo ${NEW_VER} > 


# python3 -m wheel unpack $f -d $TMP_DIR

# # --- Begin modifications
# # Update the version
# sed -i "s/^Version: ${ORIG_VER}/Version: ${NEW_VER}/" $TMP_DIR/${PACKAGE_NAME}-${ORIG_VER}/${PACKAGE_NAME}-${ORIG_VER}.dist-info/METADATA
# # MODIFY_ME2 - review and modify the METADATA file here
# # ...
# # --- End modifications

# # Re-package into a new whl file now
# cd $TMP_DIR
# mv cudaq_qec-${ORIG_VER}/${PACKAGE_NAME}-${ORIG_VER}.dist-info ${PACKAGE_NAME}-${ORIG_VER}/${PACKAGE_NAME}-${NEW_VER}.dist-info
# python3 -m wheel pack cudaq_qec-${ORIG_VER} -d .
# cd -
# mv $TMP_DIR/cudaq_qec-${NEW_VER}*.whl wheels_new
# rm -rf $TMP_DIR

# echo "Done!"
# echo "Your original wheels are in wheels_orig, and your patched wheels are in wheels_new."
# echo "You can now upload the patched wheels to PyPI."
