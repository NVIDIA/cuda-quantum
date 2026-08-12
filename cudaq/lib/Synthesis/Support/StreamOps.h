/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/Ellipse.h"
#include "Math/Geometry/GridOp.h"
#include "Math/Geometry/Interval.h"
#include "Math/Geometry/Rectangle.h"
#include "Math/Geometry/ToUpright.h"
#include "Support/DebugScope.h"
#include "cudaq/Synthesis/Math/Integer.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Ring/Domega.h"
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"
#include "cudaq/Synthesis/Math/Ring/Zomega.h"
#include "cudaq/Synthesis/Math/Ring/Zsqrt2.h"
#include "cudaq/Synthesis/Math/Unitary.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
// llvm::raw_ostream operator<< overloads for the synth domain types
//===----------------------------------------------------------------------===//
//
// Included by synth source files that emit LLVM_DEBUG logging so call sites
// can stream domain values directly, e.g.
//
//     cudaq::synth::dbgs() << "k=" << unitary.k() << ", w=" << unitary.w();
//
// without having to spell `.to_string()` at each `<<`. Kept private to
// cudaq/lib/Synthesis: the public type headers should not transitively pull
// in <llvm/Support/raw_ostream.h>.

namespace cudaq::synth {

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Integer &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Real &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ZSqrt2 &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ZOmega &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DSqrt2 &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DOmega &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Interval &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Rectangle &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Ellipse &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const GridOp &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const UprightResult &v) {
  return os << v.to_string();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const DOmegaUnitary &v) {
  return os << v.to_string();
}

} // namespace cudaq::synth
