# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Generate lit.site.cfg.py from the template.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# The canonical model FileCheck corpus is a source-only subtree.  Materialize
# its execution-root directory so lit can find the parent site configuration
# and map `test/Model` into `build/test/Model` like the CMake-owned subtrees.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Model")

if(LIT_PATH AND FILECHECK_PATH)
  # lit.py needs to be invoked through the Python interpreter; a
  # PATH-found 'lit' / 'llvm-lit' wrapper is directly executable.
  if(LIT_PATH MATCHES "\\.py$")
    set(_qlx_lit_command "${Python3_EXECUTABLE}" "${LIT_PATH}")
  else()
    set(_qlx_lit_command "${LIT_PATH}")
  endif()

  set(QLX_LIT_DEPENDS
    qlx-opt
    qlx-translate
  )
  foreach(_t IN LISTS QLX_FILECHECK_TARGET QLX_NOT_TARGET QLX_COUNT_TARGET)
    if(_t)
      list(APPEND QLX_LIT_DEPENDS ${_t})
    endif()
  endforeach()
  # mlir-translate: only a real target in bundled LLVM mode; installed
  # mode finds it via find_program and there's nothing to add.
  if(QLX_MLIR_TRANSLATE_TARGET)
    list(APPEND QLX_LIT_DEPENDS ${QLX_MLIR_TRANSLATE_TARGET})
  endif()
  # Setup-fixture test: builds the tools / runtime .so's that the lit
  # suite depends on. ctest invokes this before qlx-filecheck-tests
  # via the FIXTURES_REQUIRED wiring below, so a bare
  # `ctest -R qlx-filecheck-tests` on a clean tree does the right
  # thing (previously only `ninja check-qlx` built the deps).
  #
  # Filter out IMPORTED targets — LLVM imports FileCheck / not / count /
  # mlir-translate as pre-built externals in installed-LLVM mode; we
  # can't `cmake --build --target` them.
  set(_qlx_lit_build_targets "")
  foreach(_dep IN LISTS QLX_LIT_DEPENDS)
    if(TARGET ${_dep})
      get_target_property(_imported ${_dep} IMPORTED)
      if(NOT _imported)
        list(APPEND _qlx_lit_build_targets ${_dep})
      endif()
    endif()
  endforeach()

  add_test(NAME qlx-lit-build-tools
    COMMAND ${CMAKE_COMMAND}
      --build ${CMAKE_BINARY_DIR}
      --target ${_qlx_lit_build_targets}
  )
  set_tests_properties(qlx-lit-build-tools PROPERTIES
    FIXTURES_SETUP qlx-lit-tools
    # This fixture runs `cmake --build` on the main build dir. Concurrent
    # Ninja invocations on one build dir collide on the build lock under
    # `ctest -j`, so serialize build fixtures while still parallelizing the
    # FileCheck/pytest tests.
    RESOURCE_LOCK qlx-build-dir
  )

  add_test(NAME qlx-filecheck-tests
    COMMAND ${_qlx_lit_command} -v
      --config-prefix lit.site
      ${CMAKE_CURRENT_BINARY_DIR}/Conversion
      ${CMAKE_CURRENT_BINARY_DIR}/Dialect
      ${CMAKE_CURRENT_BINARY_DIR}/Model
      ${CMAKE_CURRENT_BINARY_DIR}/Target
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(qlx-filecheck-tests PROPERTIES
    FIXTURES_REQUIRED qlx-lit-tools
    LABELS "qlx;conformance"
  )

  # ctest's FIXTURES_REQUIRED automatically runs qlx-lit-build-tools
  # (the fixture setup) before the FileCheck tests when either test is
  # selected, so the regex only needs to name the main tests here.
  add_custom_target(check-qlx
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      # Avoid a trailing `$` here: Ninja treats it as a line-continuation
      # escape in the generated custom command and concatenates `DESC` onto
      # the ctest regular expression.
      -R "^qlx-filecheck-tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running QLX FileCheck tests")
else()
  if(NOT LIT_PATH)
    message(WARNING
      "lit not found — FileCheck tests disabled.  "
      "Bundled mode normally provides it via "
      "third_party/llvm/llvm/utils/lit/lit.py "
      "(LLVM_EXTERNAL_LIT); installed mode expects 'pip install lit'.")
  endif()
  if(NOT FILECHECK_PATH)
    message(WARNING
      "FileCheck not found — FileCheck tests disabled.  "
      "In bundled mode this should resolve automatically via the "
      "FileCheck target; in installed mode build LLVM with "
      "-DLLVM_INSTALL_UTILS=ON or pass -DFILECHECK_PATH=...")
  endif()
endif()
