// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// Licensed under the Apache License, Version 2.0.

// REQUIRES: qlx-opt
// RUN: qlx-opt -verify-diagnostics %s --qlx-synthesize-rotations=synthesizer=missing

// expected-error@+1 {{unknown rotation synthesizer 'missing'}}
module {}
