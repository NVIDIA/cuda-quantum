/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ArgumentWrapper.h"
#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "cudaq/platform/qpu.h"

//===----------------------------------------------------------------------===//
// Translators:
//
// These use some of the same steps as the default platform, so they are placed
// here to encourage code reuse (over block copying).  In Python,
// cudaq.translate can be used to lower a kernel to a particular transport
// layer, such as QIR or Open QASM.  This can be done by the curious user, but
// does not impact the launching of a kernel in any way.
//===----------------------------------------------------------------------===//

namespace cudaq::detail {
/// Lowers \p module to LLVM code. The LLVM code will use QIR as the transport
/// layer. If \p kernelName and \p args are provided, they will specialize the
/// selected entry-point kernel.
std::string lower_to_qir_llvm(const std::string &kernelName,
                              mlir::ModuleOp module, OpaqueArguments &args,
                              const std::string &format);

/// Lowers \p module to `Open QASM 2`. The output will be a string of `Open
/// QASM` code. \p kernelName and \p args should be provided, as they will
/// specialize the selected entry-point kernel.
std::string lower_to_openqasm(const std::string &kernelName,
                              mlir::ModuleOp module, OpaqueArguments &args);
} // namespace cudaq::detail
