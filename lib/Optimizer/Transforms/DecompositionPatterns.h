/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Support/LogicalResult.h>

namespace mlir {
class RewritePatternSet;
}

namespace cudaq {

void populateWithAllDecompositionPatterns(mlir::RewritePatternSet &patterns);

/// Get all names of registered decomposition rewrite patterns.
llvm::SmallVector<llvm::StringRef, 32> getAllDecompositionPatternNames();

/// Get the source and target gates of a registered decomposition rewrite
/// pattern.
///
/// This will throw an assertion if \p patternName is not one of the registered
/// decomposition rewrite patterns as returned by
/// getAllDecompositionPatternNames.
std::pair<llvm::StringRef, llvm::ArrayRef<llvm::StringRef>>
getSourceAndTargetGates(llvm::StringRef patternName);

/// Add a registered decomposition pattern to the given rewrite pattern set.
///
/// This will throw an assertion if \p patternName is not one of the registered
/// decomposition rewrite patterns as returned by
/// getAllDecompositionPatternNames.
void addDecompositionPattern(mlir::RewritePatternSet &patterns,
                             mlir::StringRef patternName);

/// Create a new instance of a registered decomposition pattern.
///
/// This will throw an assertion if \p patternName is not one of the registered
/// decomposition rewrite patterns as returned by
/// getAllDecompositionPatternNames.
std::unique_ptr<mlir::RewritePattern>
createDecompositionPattern(mlir::MLIRContext *context,
                           llvm::StringRef patternName);

} // namespace cudaq
