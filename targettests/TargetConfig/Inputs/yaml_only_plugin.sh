# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Shared setup for yaml_only_plugin_*.config tests. Sourced, not run as a test.

setup_yaml_only_plugin() {
	test_root=$1
	lib_ext=${2-}
	cudaq_src=${3-}
	plugin_root="${test_root}/xdg/cudaq/plugins/yaml-only-runtime"
	yml="${plugin_root}/targets/yaml-only-runtime.yml"
	lib="${plugin_root}/targets/yaml-only-runtime${lib_ext}"
	exe="${test_root}/a.out"

	rm -rf "${test_root}"
	mkdir -p "${plugin_root}/targets" "${plugin_root}/lib"

	cat > "${yml}" <<'EOF'
name: yaml-only-runtime
description: YAML-only plugin
config:
  library-mode: true
  gen-target-backend: true
EOF

	cat > "${test_root}/main.cpp" <<'EOF'
#include <cudaq.h>
int main() { return 0; }
EOF
}

compile_yaml_only_plugin_lib() {
	cudaq-target-db-gen --plugin -o "${test_root}/gen.cpp" \
		"yaml-only-runtime=${yml}"
	cxx_flags="-std=c++20 -shared -fPIC"
	if [ "$(uname)" = Darwin ]; then
		cxx_flags="${cxx_flags} -isysroot $(xcrun --show-sdk-path)"
	fi
	# shellcheck disable=SC2086
	c++ ${cxx_flags} -I"${cudaq_src}/cudaq/include" "${test_root}/gen.cpp" \
		-o "${lib}"
}
