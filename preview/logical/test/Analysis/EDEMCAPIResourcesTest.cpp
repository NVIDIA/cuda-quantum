/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 *******************************************************************************/

#include "qlx-c/Target/Translations.h"
#include "qlx/Analysis/GadgetEDEM.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>

namespace {
enum class Producer { Success, Failure, BadAlloc, LengthError };
Producer producer = Producer::Success;
unsigned calls = 0;

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

bool empty(const cudaq_gf2_owned_buffer &buffer) {
  return !buffer.data && !buffer.size && !buffer.deallocate && !buffer.context;
}
} // namespace

// Link the production C entry point to a controllable C++ producer so resource
// failures can be exercised without exhausting the machine's real memory.
namespace cudaq::logical {
mlir::FailureOr<PackedGadgetEDEM>
deriveGadgetEDEM(mlir::ModuleOp, llvm::StringRef, llvm::StringRef) {
  ++calls;
  switch (producer) {
  case Producer::BadAlloc:
    throw std::bad_alloc();
  case Producer::LengthError:
    throw std::length_error("test allocation size");
  case Producer::Failure:
    return mlir::failure();
  case Producer::Success:
    break;
  }
  PackedGadgetEDEM result;
  cudaq_edem_description description{};
  require(PackedGadgetEDEM::construct(description, result) == CUDAQ_EDEM_OK,
          "construct an empty packed EDEM");
  return result;
}
} // namespace cudaq::logical

int main() {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  auto invoke = [&](cudaq_gf2_owned_buffer *output) {
    return qlxDeriveGadgetEDEM(wrap(*module), mlirStringRefCreate("", 0),
                               mlirStringRefCreate("", 0), output);
  };

  require(mlirLogicalResultIsFailure(invoke(nullptr)), "reject null output");
  cudaq_gf2_owned_buffer occupied{};
  occupied.size = 1;
  require(mlirLogicalResultIsFailure(invoke(&occupied)),
          "reject occupied output");
  require(occupied.size == 1 && !occupied.data && calls == 0,
          "preserve rejected output without invoking the producer");

  for (auto failure :
       {Producer::Failure, Producer::BadAlloc, Producer::LengthError}) {
    producer = failure;
    cudaq_gf2_owned_buffer output{};
    require(mlirLogicalResultIsFailure(invoke(&output)),
            "translate producer failures into C failure results");
    require(empty(output), "preserve empty ownership on failure");
  }

  producer = Producer::Success;
  cudaq_gf2_owned_buffer output{};
  require(mlirLogicalResultIsSuccess(invoke(&output)),
          "remain usable after resource failures");
  cudaq_edem_view view;
  require(cudaq_edem_read(output.data, output.size, nullptr, &view) ==
              CUDAQ_EDEM_OK,
          "transfer a valid packed EDEM on success");
  cudaq_gf2_owned_buffer_release(&output);
  require(empty(output), "release the transferred ownership");
}
