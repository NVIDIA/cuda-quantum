/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// #include "cudaq/Optimizer/Transforms/SimulationDataStore.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/Types.h"
// #include "mlir/Transforms/DialectConversion.h"

#include "cudaq.h"
// #include "cudaq/algorithms/get_state.h"
// #include "cudaq/Optimizer/Builder/Factory.h"
// #include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
// #include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
// #include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/SimulationDataStore.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Target/LLVMIR/TypeToLLVM.h"

// #include <cstdlib>

#include <utility>
#include <vector>

namespace cudaq::runtime {

// /// Collect simulation state data from all `cudaq::state *` arguments.
// SimulationStateDataStore readSimulationData(
//   mlir::ModuleOp moduleOp, mlir::func::FuncOp func, const void* args,
//   std::size_t startingArgIdx = 0) { SimulationStateDataStore dataStore;

//   auto arguments = func.getArguments();
//   auto argumentLayout = factory::getFunctionArgumentLayout(moduleOp,
//   func.getFunctionType(), startingArgIdx);

//   for (std::size_t argNum = startingArgIdx; argNum < arguments.size();
//   argNum++) {
//     auto offset = argumentLayout.second[argNum - startingArgIdx];
//     auto argument = arguments[argNum];
//     auto type = argument.getType();
//     if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(type)) {
//       if (isa<cudaq::cc::StateType>(ptrTy.getElementType())) {
//         cudaq::state* state;
//         std::memcpy(&state, ((const char *)args) + offset,
//         sizeof(cudaq::state*));

//         void *dataPtr = nullptr;
//         auto stateVector = state->get_tensor();
//         auto precision = state->get_precision();
//         auto numElements = stateVector.get_num_elements();
//         auto elementSize = 0;
//         if (precision == SimulationState::precision::fp32) {
//           elementSize = sizeof(std::complex<float>);
//           auto *hostData = new std::complex<float>[numElements];
//           state->to_host(hostData, numElements);
//           dataPtr = reinterpret_cast<void *>(hostData);
//         } else {
//           elementSize = sizeof(std::complex<double>);
//           auto *hostData = new std::complex<double>[numElements];
//           state->to_host(hostData, numElements);
//           dataPtr = reinterpret_cast<void *>(hostData);
//         }
//         dataStore.setElementSize(elementSize);
//         dataStore.addData(dataPtr, numElements);
//       }
//     }
//   }
//   return dataStore;
// }

cudaq::opt::SimulationStateDataStore readSimulationStateData(
    std::pair<std::size_t, std::vector<std::size_t>> &argumentLayout,
    const void *args);

} // namespace cudaq::runtime
