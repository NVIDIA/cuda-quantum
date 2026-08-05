# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE)
  message(FATAL_ERROR "INPUT_FILE and OUTPUT_FILE are required")
endif()

file(READ "${INPUT_FILE}" GENERATED_MARKDOWN)
string(REPLACE
       "\n[TOC]\n"
       "\n```{contents}\n:local:\n:depth: 2\n```\n"
       GENERATED_MARKDOWN
       "${GENERATED_MARKDOWN}")
file(WRITE "${OUTPUT_FILE}" "${GENERATED_MARKDOWN}")
