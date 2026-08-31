/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ -c %s -o %t.o > %t.compile.stdout 2> %t.compile.stderr
// RUN: FileCheck %s --check-prefix=NOTICE --input-file=%t.compile.stderr
// RUN: FileCheck %s --check-prefix=NO-NOTICE --allow-empty --input-file=%t.compile.stdout
// RUN: nvq++ %t.o -o %t.exe > %t.link.stdout 2> %t.link.stderr
// RUN: FileCheck %s --check-prefix=NOTICE --input-file=%t.link.stderr
// RUN: FileCheck %s --check-prefix=NO-NOTICE --allow-empty --input-file=%t.link.stdout
// RUN: nvq++ --help > %t.help.stdout 2> %t.help.stderr
// RUN: nvq++ --version > %t.version.stdout 2> %t.version.stderr
// RUN: nvq++ --list-targets > %t.targets.stdout 2> %t.targets.stderr
// RUN: nvq++ -E %s -o %t.ii > %t.preprocess.stdout 2> %t.preprocess.stderr
// RUN: cat %t.help.stdout %t.help.stderr %t.version.stdout %t.version.stderr %t.targets.stdout %t.targets.stderr %t.preprocess.stdout %t.preprocess.stderr | FileCheck %s --check-prefix=NO-NOTICE
// RUN: not nvq++ -c %t.missing.cpp -o %t.missing.o > %t.failure.stdout 2> %t.failure.stderr
// RUN: FileCheck %s --check-prefix=NOTICE --input-file=%t.failure.stderr
// RUN: FileCheck %s --check-prefix=NO-NOTICE --allow-empty --input-file=%t.failure.stdout
// clang-format on

int main() { return 0; }

// clang-format off
// NOTICE: WARNING: The CUDA-Q `sample` and `observe` algorithmic primitives will change in a future release.
// NOTICE-NOT: WARNING: The CUDA-Q
// NO-NOTICE-NOT: WARNING: The CUDA-Q
// clang-format on
