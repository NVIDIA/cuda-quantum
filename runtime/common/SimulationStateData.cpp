/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/SimulationStateData.h"
#include "common/SimulationState.h"
// #include "cudaq.h"
// #include "cudaq/algorithms/get_state.h"
// #include "cudaq/Optimizer/Builder/Factory.h"
// #include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
// #include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
// #include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/SimulationDataStore.h"
#include "cudaq/qis/state.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include <cstdlib>
#include <iostream>

namespace cudaq::runtime {

// /// Collect simulation state data from all `cudaq::state *` arguments.
// SimulationStateDataStore readSimulationData(
//   mlir::ModuleOp moduleOp, mlir::func::FuncOp func, const void* args,
//   std::size_t startingArgIdx) { SimulationStateDataStore dataStore;

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

// template <typename T>
// static void addStateData(cudaq::state *state,
//                          cudaq::opt::SimulationStateDataStore &store) {
//   std::cout << "Getting number of elements from the state" << std::endl;
//   auto numElements = state->get_tensor().get_num_elements();
//    std::cout << "NumElements: " << numElements << std::endl;
//   std::cout << "Copying simulation data into data store" << std::endl;

//   if (state->is_on_gpu()) {
//     auto *hostData = new T[numElements];
//     state->to_host(hostData, numElements);
//     std::cout << "Copying state:" << std::endl;
//     for (std::size_t i = 0; i < numElements; i++) {
//       std::cout << *(hostData + i) << ",";
//     }
//     std::cout << std::endl;
//     store.addData(hostData, numElements, sizeof(T),
//                   [](void *ptr) { delete static_cast<T *>(ptr); });
//   } else {
//     auto hostData = state->get_tensor().data;
//     store.addData(hostData, numElements, sizeof(T), [](void *ptr) {});
//   }
//   std::cout << "Done Copying simulation data into data store" << std::endl;
// }

/// Collect simulation state data from all `cudaq::state *` arguments.
cudaq::opt::SimulationStateDataStore readSimulationStateData(
    std::pair<std::size_t, std::vector<std::size_t>> &argumentLayout,
    const void *args) {
  cudaq::opt::SimulationStateDataStore dataStore;

  auto offsets = argumentLayout.second;
  std::cout << "Reading simulation data into data store: " << offsets.size()
            << std::endl;
  for (std::size_t argNum = 0; argNum < offsets.size(); argNum++) {
    auto offset = offsets[argNum];
    std::cout << "Offset for state arg: " << offset << std::endl;

    cudaq::state *state;
    std::memcpy(&state, ((const char *)args) + offset, sizeof(cudaq::state *));

    std::cout << "getting precision" << std::endl;
    auto precision = state->get_precision();

    std::cout << "Adding state data: precision is 32 bit: " << (precision == cudaq::SimulationState::precision::fp32) << std::endl;
    // auto precision = state->get_precision();
    // if (precision == cudaq::SimulationState::precision::fp32)
    //   addStateData<std::complex<float>>(state, dataStore);
    // else
    //   addStateData<std::complex<double>>(state, dataStore);

    
    std::cout << "getting tensor" << std::endl;
    auto stateVector = state->get_tensor();
    std::cout << "getting num elements" << std::endl;
    auto numElements = stateVector.get_num_elements();
    std::cout << "numElements: " <<numElements << std::endl;
    auto elementSize = stateVector.element_size();
    std::cout << "elementSize: " << elementSize << std::endl;
    if (state->is_on_gpu()) { 
      std::cout << "State is on gpu, copying... " << std::endl;
      if (precision == cudaq::SimulationState::precision::fp32) {
        assert (elementSize == sizeof(std::complex<float>) && "Incorrect complex<float> element size");
        auto *hostData = new std::complex<float>[numElements];
        state->to_host(hostData, numElements);
        std::cout << "Copying state:" << std::endl;
        for (std::size_t i = 0; i < numElements; i++) {
          std::cout << *(hostData + i) << ",";
        }
        std::cout << std::endl;
        dataStore.addData(hostData, numElements, elementSize,
                      [](void *ptr) { delete static_cast<std::complex<float> *>(ptr); });
      } else {
        assert (elementSize == sizeof(std::complex<double>) && "Incorrect complex<double> element size");
        auto *hostData = new std::complex<double>[numElements];
        state->to_host(hostData, numElements);
        std::cout << "Copying state:" << std::endl;
        for (std::size_t i = 0; i < numElements; i++) {
          std::cout << *(hostData + i) << ",";
        }
        std::cout << std::endl;
        dataStore.addData(hostData, numElements, elementSize,
                      [](void *ptr) { delete static_cast<std::complex<double> *>(ptr); });
      }
    } else {
      std::cout << "Not on gpu, using memory:" << std::endl;
      auto hostData = state->get_tensor().data;
      dataStore.addData(hostData, numElements, elementSize, [](void *ptr) {});
    }
    std::cout << "Done Copying simulation data into data store" << std::endl;
  }
  std::cout << "Done reading simulation data into data store" << std::endl;
  return dataStore;
}
} // namespace cudaq::runtime
