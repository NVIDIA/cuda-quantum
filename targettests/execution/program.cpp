/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include "cudaq/builder/kernels.h"
#include <iostream>

__qpu__ void test1(std::vector<cudaq::complex> inState) {
    cudaq::qvector q1 = inState;
}

//  __qpu__ void test2(cudaq::state *inState) {
//    cudaq::qvector q2(inState);
//    cudaq::x(q2);
// }

// __qpu__ void test3() {
//   auto q3 = cudaq::qvector({M_SQRT1_2, 0., 0., M_SQRT1_2});
// }


// error: /workspaces/cuda-quantum/lib/Frontend/nvqpp/ConvertExpr.cpp:392: not yet implemented: argument type conversion
// __qpu__ void test5(cudaq::state *inState) {
//   test2(inState);
// }



int main() {
    std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
    {
        // Passing state data as argument (vector<complex>)
        std::cout << "test1(vec): "  << "\n";
        auto counts = cudaq::sample(test1, vec);
        counts.dump();
    }

    // {
    //     // Passing state ptr as argument - no support for from_data
    //
    //     std::cout << "test2(state): "  << "\n";
    //     auto state = cudaq::state::from_data(vec);
    //
    //     // 'func.call' op '__nvqpp_cudaq_state_numberOfQubits' does not reference a valid function
    //     auto counts = cudaq::sample(test2, &state);
    //     counts.dump();
    // }

    // {
    //     // Passing a state from another kernel as argument
    //
    //     std::cout << "test2(test3): "  << "\n";
    //     auto state = cudaq::get_state(test3);
    //
    //     // error: 'func.call' op '__nvqpp_cudaq_state_numberOfQubits' does not reference a valid function
    //     auto counts = cudaq::sample(test2, &state);
    //     counts.dump();
    // }

    // {
    //     // Passing a state to another kernel as argument
    //
    //     std::cout << "test4(state): "  << "\n";
    //     
    //     auto state = cudaq::state::from_data(vec);
    //     auto counts = cudaq::sample(test4, &state);
    // }

    // {
    //     // Creating a kernel from state and passing its state to another kernel - is it deprecated?
    //
        std::cout << "test2(kernel): "  << "\n";
        std::vector<std::complex<double>> vec{.70710678, 0., 0., 0.70710678};
        auto kernel = cudaq::make_kernel();
        auto qubits = kernel.qalloc(2);
    
        cudaq::from_state(kernel, qubits, vec);
        auto counts = cudaq::sample(kernel);
    //
    //     // error: 'func.call' op '__nvqpp_cudaq_state_numberOfQubits' does not reference a valid function
    //     //auto state = cudaq::get_state(kernel);
    //     //auto counts = cudaq::sample(test2, &state);
    //
         counts.dump();
    // }

}