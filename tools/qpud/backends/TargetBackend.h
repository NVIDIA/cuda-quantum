/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "common/Registry.h"
#include "cudaq/spin_op.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include <deque>

namespace cudaq {

struct DynamicResult {
  char *ptr;
  std::uint64_t len;
};

// Typedef for the KERNEL.thunk() function
using ThunkFunction = DynamicResult (*)(void *, bool);
class Kernel {
protected:
  ThunkFunction thunk;
  std::string kernelName;
  std::string qirCode;
  std::string quakeCode;

public:
  Kernel(ThunkFunction tf, const std::string name, const std::string qirC,
         const std::string qC)
      : thunk(tf), kernelName(name), qirCode(qirC), quakeCode(qC) {}
  std::string_view getQuakeCode() { return quakeCode; }
  std::string_view getQIRCode() { return qirCode; }
  std::string_view name() const { return kernelName; }
  DynamicResult operator()(void *args, bool isCS) { return thunk(args, isCS); }
};

/// The TargetBackend class provides an extension point for the
/// invocation of auto-generated thunk functions under a variety of
/// execution contexts. Specifically, thunk functions can be invoked for
/// final state sampling (produce counts dictionary), spin_op observation
/// (produce expectation value <psi | H | psi>), and base execution (just
/// invoke the function and get the return value).
class TargetBackend : public cudaq::registry::RegisteredType<TargetBackend> {
protected:
  llvm::LLVMContext llvmContext;

  bool setupTargetTriple(llvm::Module *llvmModule);

  /// @brief Given the kernel, extract the Quake code, synthesize with the
  /// kernel
  /// args, and lower to the Base Profile QIR
  /// @param thunk
  /// @param localLLVMContext
  /// @param kernelArgs
  /// @return
  std::unique_ptr<llvm::Module>
  lowerQuakeToBaseProfile(Kernel &thunk, llvm::LLVMContext &localLLVMContext,
                          cudaq::spin_op *term, void *kernelArgs);

  virtual void applyNativeGateSetPasses(mlir::PassManager &) {}

public:
  /// Compile the quakeCode further to an LLVM Module that can be JIT executed.
  /// Default implementation here lowers the quake code to dynamic full QIR.
  virtual std::unique_ptr<llvm::Module>
  compile(mlir::MLIRContext &context, const std::string_view quakeCode);

  /// Initialize the backend, one time execution
  virtual void initialize() = 0;
  virtual bool isInitialized() { return true; }

  /// @brief Return if this backend is a QPU simulator
  virtual bool isSimulator() { return true; }

  /// @brief Return true if this backend supports conditional feedback
  virtual bool supportsConditionalFeedback() { return true; }

  /// Execute the ThunkFunction with the given args. The return value
  /// (if kernel does not define a void return type) will be packed into
  /// the kernelArgs opaque pointer.
  virtual DynamicResult baseExecute(Kernel &thunk, void *kernelArgs,
                                    bool isClientServer = true) = 0;

  /// Execute the ThunkFunction with the given args, return the histogram
  /// of observed bit strings to number of times observed (the counts
  /// dictionary). Sample the final state `shots` times, return a vector<size_t>
  /// of size 3 * n_unique_bit_strings of the format [bitStringAsLong,
  /// N_MeasuredBitsInString, Count, ... repeat ...]
  virtual std::vector<std::size_t> sample(Kernel &thunk, std::size_t shots,
                                          void *kernelArgs) = 0;

  /// Execute the ThunkFunction with the given args, return the expected value
  /// of the provided spin_op (represented as a vector<double>, with format
  /// [OP0 OP1 OP2 ... COEFF_REAL COEFF_IMAG | OP0 OP1 OP2 ... COEFF_REAL
  /// COEFF_IMAG | ... | NTERMS]).
  virtual std::tuple<double, std::vector<std::size_t>>
  observe(Kernel &thunk, std::vector<double> &spin_op_data,
          const std::size_t shots, void *kernelArgs) = 0;

  /// Execute an Observe task with the given ansatz and spin op data, but
  /// detach and return the job ids and job names for the given task.
  virtual std::tuple<std::vector<std::string>, std::vector<std::string>>
  observeDetach(Kernel &thunk, std::vector<double> &spin_op_data,
                const std::size_t shots, void *kernelArgs) {
    throw std::runtime_error("observeDetach not supported for this backend.");
  }

  /// Given a valid job id, return the observe result.
  virtual std::tuple<double, std::vector<std::size_t>>
  observeFromJobId(const std::string &jobId) {
    throw std::runtime_error(
        "observeFromJobId not supported for this backend.");
  }

  virtual std::tuple<std::string, std::string>
  sampleDetach(Kernel &thunk, const std::size_t shots, void *kernelArgs) {
    throw std::runtime_error("sampleDetach not supported for this backend.");
  }

  /// Given a valid job id, return the sample result.
  virtual std::vector<std::size_t> sampleFromJobId(const std::string &jobId) {
    throw std::runtime_error("sampleFromJobId not supported for this backend.");
  }

  /// @brief Provide a hook for specifying a specific platform backend (e.g.
  /// Quantinuum H1-2)
  virtual void setSpecificBackend(const std::string &backend) {}

  virtual ~TargetBackend() {}
};
} // namespace cudaq
