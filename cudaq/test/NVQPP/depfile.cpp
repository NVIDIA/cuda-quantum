/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// clang-format off
// Verify nvq++ emits Makefile-syntax depfiles for -MD/-MMD/-MT/-MF on both
// compile paths: the default target (cudaq-quake) and orca-photonics library
// mode (direct clang++ -c).

// RUN: rm -rf %t && mkdir -p %t

// Default target (cudaq-quake path): -MD with a single -MT/-MF.
// RUN: nvq++ -c %s -o %t/default_md.o -MD -MT %t/default_md.o -MF %t/default_md.d -I%S/Inputs 
// RUN: [ -e %t/default_md.o ] && [ -e %t/default_md.d ] 
// RUN: cat %t/default_md.d | FileCheck %s --check-prefix=DEFAULT-MD

// Default target (cudaq-quake path): -MMD with a single -MT/-MF.
// RUN: nvq++ -c %s -o %t/default_mmd.o -MMD -MT %t/default_mmd.o -MF %t/default_mmd.d -I%S/Inputs 
// RUN: [ -e %t/default_mmd.o ] && [ -e %t/default_mmd.d ] 
// RUN: cat %t/default_mmd.d | FileCheck %s --check-prefix=DEFAULT-MMD

// Default target (cudaq-quake path): multiple -MT targets in one depfile.
// RUN: nvq++ -c %s -o %t/default_multi.o -MMD -MT targetA -MT targetB -MF %t/default_multi.d -I%S/Inputs 
// RUN: [ -e %t/default_multi.o ] && [ -e %t/default_multi.d ] 
// RUN: cat %t/default_multi.d | FileCheck %s --check-prefix=MULTI-MT

// Orca-photonics (library mode / direct clang++ path): -MD with a single -MT/-MF. 
// RUN: nvq++ --target orca-photonics -c %s -o %t/orca_md.o -MD -MT %t/orca_md.o -MF %t/orca_md.d -I%S/Inputs 
// RUN: [ -e %t/orca_md.o ] && [ -e %t/orca_md.d ] 
// RUN: cat %t/orca_md.d | FileCheck %s --check-prefix=ORCA-MD

// Orca-photonics (library mode / direct clang++ path): -MMD with a single
// -MT/-MF. 
// RUN: nvq++ --target orca-photonics -c %s -o %t/orca_mmd.o -MMD -MT %t/orca_mmd.o -MF %t/orca_mmd.d -I%S/Inputs 
// RUN: [ -e %t/orca_mmd.o ] && [ -e %t/orca_mmd.d ] 
// RUN: cat %t/orca_mmd.d | FileCheck %s --check-prefix=ORCA-MMD

// Orca-photonics (library mode / direct clang++ path): multiple -MT targets.
// RUN: nvq++ --target orca-photonics -c %s -o %t/orca_multi.o -MMD -MT targetA -MT targetB -MF %t/orca_multi.d -I%S/Inputs 
// RUN: [ -e %t/orca_multi.o ] && [ -e %t/orca_multi.d ] 
// RUN: cat %t/orca_multi.d | FileCheck %s --check-prefix=MULTI-MT

// clang-format on
#include "dep_header.h"
#include <cudaq.h>

int plain_old_function() { return cudaq_test_dep_value(); }

// DEFAULT-MD: default_md.o:
// DEFAULT-MD-DAG: depfile.cpp
// DEFAULT-MD-DAG: __stddef
// DEFAULT-MD-DAG: Inputs/dep_header.h

// DEFAULT-MMD: default_mmd.o:
// DEFAULT-MMD-DAG: depfile.cpp
// DEFAULT-MMD-DAG: Inputs/dep_header.h
// DEFAULT-MMD-NOT: __stddef

// ORCA-MD: orca_md.o:
// ORCA-MD-DAG: depfile.cpp
// ORCA-MD-DAG: __stddef
// ORCA-MD-DAG: Inputs/dep_header.h

// ORCA-MMD: orca_mmd.o:
// ORCA-MMD-DAG: depfile.cpp
// ORCA-MMD-DAG: Inputs/dep_header.h
// ORCA-MMD-NOT: __stddef

// MULTI-MT: targetA targetB:
// MULTI-MT-DAG: depfile.cpp
// MULTI-MT-DAG: Inputs/dep_header.h
