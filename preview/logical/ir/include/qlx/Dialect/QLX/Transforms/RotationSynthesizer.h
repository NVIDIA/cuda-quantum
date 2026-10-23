// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// Licensed under the Apache License, Version 2.0.

#ifndef QLX_DIALECT_QLX_TRANSFORMS_ROTATIONSYNTHESIZER_H
#define QLX_DIALECT_QLX_TRANSFORMS_ROTATIONSYNTHESIZER_H

#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>
#include <string>

namespace qlx {

/// Positive H/S/T generators in chronological application order.
struct SynthesizedWord {
  llvm::SmallVector<BuiltinAction> actions;
  double achievedError = 0;
  std::string algorithm, implementation, version;
};

class RotationSynthesizer {
public:
  virtual ~RotationSynthesizer() = default;
  virtual llvm::StringRef name() const = 0;
  virtual llvm::StringRef version() const = 0;
  virtual bool supportsU3() const { return false; }
  virtual mlir::FailureOr<SynthesizedWord> synthesizeRz(double angle,
                                                        double precision) = 0;
  virtual mlir::FailureOr<SynthesizedWord>
  synthesizeU3(double theta, double phi, double lam, double precision) {
    return mlir::failure();
  }
  virtual llvm::StringRef lastError() const { return {}; }
};

/// Registration never replaces an existing entry (including gridsynth).
LLVM_EXTERNAL_VISIBILITY bool
registerRotationSynthesizer(std::shared_ptr<RotationSynthesizer> value);
LLVM_EXTERNAL_VISIBILITY std::shared_ptr<RotationSynthesizer>
lookupRotationSynthesizer(llvm::StringRef name);
/// Used to release Python callbacks while the interpreter still holds the GIL.
LLVM_EXTERNAL_VISIBILITY void
unregisterRotationSynthesizer(llvm::StringRef name);

} // namespace qlx
#endif
