/*******************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target quantinuum --emulate %s -o %t && echo skipping
#include "../execution/sudoku_2x2-reg_name.cpp"
