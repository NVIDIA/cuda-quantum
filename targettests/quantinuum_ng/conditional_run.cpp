/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if %quantinuum_avail; then nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && echo skipping; fi
// clang-format on
#include "../execution/conditional_run.cpp"
