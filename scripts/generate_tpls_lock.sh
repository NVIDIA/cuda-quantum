# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Writes a lock file with one line per submodule: "<commit_sha> <path>".
# Same format as: git config --file .gitmodules --get-regexp '^submodule\..*\.path$' \
#   | awk '{print $2}' | while read p; do printf "%s %s\n" "$(git rev-parse HEAD:$p)" "$p"; done
# Used so the package_sources image (or install_prerequisites.sh -l) can clone each tpl at a pinned commit.
# Must be run from repo root with submodules initialized.
#
# Usage: ./scripts/generate_tpls_lock.sh [output_file]
# Default output: tpls_commits.lock (repo root)
#
# This will produce a file that looks like:
# $ cat tpls_commits.lock
# fc8d07cfe54ba9f5019453dfdb112491246ee017 tpls/fmt
# f8d7d77c06936315286eb55f8de22cd23c188571 tpls/googletest-src
# 7cbf1a2591520c2491aa35339f227775f4d3adf6 tpls/llvm
# 81fe2d424f05e5596772caeaa0e2b7e6518da92c tpls/eigen
# 102a354e0ceded132bb1f38b5d0be90806a3070b tpls/armadillo
# b702f6db50b5f92f4e9a8aeb4b5f985bcbba38f4 tpls/ensmallen
# 287333ee00555aaece5a5cf6acc9040563c6f642 tpls/spdlog
# 67b6156299d22f70e38db3a68fe7ec2a00022739 tpls/xtl
# cccd75769b332f7cc726d702555180d74bd78953 tpls/xtensor
# 8b48ff878c168b51fe5ef7b8c728815b9e1a9857 tpls/pybind11
# 304800611697e8abd8fd424bafa72a45f644f9ed tpls/qpp
# d202b82fbccf897604a18e035c09e1330dffd082 tpls/cpr
# 7609450f71434bdc9fbd9491a9505b423c2a8496 tpls/asio
# 94a011b9f7c0a991e5382927a2dbe5a7d9a056b8 tpls/Crow
# 42e0b9e099180e8570407c33f87b4683cac00d81 tpls/Stim

set -euo pipefail

OUTPUT="${1:-tpls_commits.lock}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

OUTPUT="${OUTPUT:-tpls_commits.lock}"
: > "$OUTPUT"
while read -r p; do
  [ -z "$p" ] && continue
  sha="$(git rev-parse "HEAD:$p" 2>/dev/null)" || continue
  printf "%s %s\n" "$sha" "$p" >> "${OUTPUT:-tpls_commits.lock}"
done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')
