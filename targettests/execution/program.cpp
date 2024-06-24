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
    // Should synthesize to
    // h(q1[0]);
    // cx(q1[0], q1[1]);
}

//  __qpu__ void test2(cudaq::state *inState) {
//    cudaq::qvector q2(inState);
//    cudaq::x(q2);
// }

// __qpu__ void test3() {
//   auto q3 = cudaq::qvector({M_SQRT1_2, 0., 0., M_SQRT1_2});
// }

// error: /workspaces/cuda-quantum/lib/Frontend/nvqpp/ConvertExpr.cpp:1938: not yet implemented: unknown function, get_state, in cudaq namespace
// __qpu__ void test4() {
//   cudaq::qvector q(cudaq::get_state(test3));
// }

// error: /workspaces/cuda-quantum/lib/Frontend/nvqpp/ConvertExpr.cpp:392: not yet implemented: argument type conversion
// __qpu__ void test5(cudaq::state *inState) {
//   test2(inState);
// }



int main() {
    std::vector<cudaq::complex> vec{M_SQRT1_2, 0., 0., M_SQRT1_2};

    {
        // Passing state data as argument (vector<complex>)

        // Before synthesis:

        // func.func @__nvqpp__mlirgen__function_test1._Z5test1St6vectorISt7complexIfESaIS1_EE(%arg0: !cc.stdvec<complex<f32>>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
        //     %0 = cc.stdvec_size %arg0 : (!cc.stdvec<complex<f32>>) -> i64
        //     %1 = math.cttz %0 : i64
        //     %2 = cc.stdvec_data %arg0 : (!cc.stdvec<complex<f32>>) -> !cc.ptr<complex<f32>>
        //     %3 = quake.alloca !quake.veq<?>[%1 : i64]
        //     %4 = quake.init_state %3, %2 : (!quake.veq<?>, !cc.ptr<complex<f32>>) -> !quake.veq<?>
        //     return
        // }

        // After synthesis

        // func.func @__nvqpp__mlirgen__function_test1._Z5test1St6vectorISt7complexIfESaIS1_EE() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
        //     %0 = cc.const_array [0.707106769 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32, 0.707106769 : f32, 0.000000e+00 : f32] : !cc.array<complex<f32> x 4>
        //     %1 = cc.alloca !cc.array<complex<f32> x 4>
        //     cc.store %0, %1 : !cc.ptr<!cc.array<complex<f32> x 4>>
        //     %2 = cc.cast %1 : (!cc.ptr<!cc.array<complex<f32> x 4>>) -> !cc.ptr<complex<f32>>
        //     %c4_i64 = arith.constant 4 : i64
        //     %3 = math.cttz %c4_i64 : i64                        // (TODO: replace by a const)
        //     %4 = quake.alloca !quake.veq<?>[%3 : i64]
        //     %5 = quake.init_state %4, %2 : (!quake.veq<?>, !cc.ptr<complex<f32>>) -> !quake.veq<?> // TODO: replace by gates
        //     return
        // }

        // TODO: in StatePreparation pass
        // input - vector<double>, qubits
        // output - MLIR replacing alloca+state_init instructions with gates on qubits

        // %3 = math.cttz %c4_i64 : i64
        // %4 = quake.alloca !quake.veq<?>[%3 : i64]
        // %5 = quake.init_state %4, %2 : (!quake.veq<?>, !cc.ptr<complex<f32>>) -> !quake.veq<?>

        // => (something like)

        // create a function that does the following and call it on qubits
        // %6 = quake.extract_ref %5[0] : (!quake.veq<?>) -> !quake.ref
        // quake.ry (%cst) %6 : (f64, !quake.ref) -> ()
        // ...

        // TODO: Run state preparation pass before synthesis 

        std::cout << "test1(vec): "  << "\n";
        auto counts = cudaq::sample(test1, vec);
        counts.dump();
    }

    // {
    //     // Passing state ptr as argument - no support for from_data

    //     // "func.func"() ({
    //     // ^bb0(%arg0: !cc.ptr<!cc.state>):
    //     //     %0 = "func.call"(%arg0) {callee = @__nvqpp_cudaq_state_numberOfQubits} : (!cc.ptr<!cc.state>) -> i64
    //     //     %1 = "quake.alloca"(%0) : (i64) -> !quake.veq<?>
    //     //     %2 = "quake.init_state"(%1, %arg0) : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>
    //     //     "func.return"() : () -> ()
    //     // }) {"cudaq-entrypoint", "cudaq-kernel", function_type = (!cc.ptr<!cc.state>) -> (), no_this, sym_name = "__nvqpp__mlirgen__function_test2._Z5test2PN5cudaq5stateE"} : () -> ()
        
    //     std::cout << "test2(state): "  << "\n";
    //     auto state = cudaq::state::from_data(vec);

    //     // 'func.call' op '__nvqpp_cudaq_state_numberOfQubits' does not reference a valid function
    //     //auto counts = cudaq::sample(test2, &state);
    //     //counts.dump();
    // }

    // {
    //     // Passing a state from another kernel as argument

    //     // "func.func"() ({
    //     // ^bb0(%arg0: !cc.ptr<!cc.state>):
    //     //     %0 = "func.call"(%arg0) {callee = @__nvqpp_cudaq_state_numberOfQubits} : (!cc.ptr<!cc.state>) -> i64
    //     //     %1 = "quake.alloca"(%0) : (i64) -> !quake.veq<?>
    //     //     %2 = "quake.init_state"(%1, %arg0) : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>
    //     //     "func.return"() : () -> ()
    //     // }) {"cudaq-entrypoint", "cudaq-kernel", function_type = (!cc.ptr<!cc.state>) -> (), no_this, sym_name = "__nvqpp__mlirgen__function_test2._Z5test2PN5cudaq5stateE"} : () -> ()
        
    //     std::cout << "test2(test3): "  << "\n";
    //     auto state = cudaq::get_state(test3);

    //     // error: 'func.call' op '__nvqpp_cudaq_state_numberOfQubits' does not reference a valid function
    //     auto counts = cudaq::sample(test2, &state);
    //     counts.dump();
    // }

    // {
    //     // Passing a state to another kernel as argument
    //     std::cout << "test4(state): "  << "\n";
    //     //auto state = cudaq::state::from_data(vec);
    //     //auto counts = cudaq::sample(test4, &state);
    // }

    // {
    //     // Creating a kernel from state and passing its state to another kernel

    //     // "func.func"() ({
    //     // ^bb0(%arg0: !cc.ptr<!cc.state>):
    //     //     %0 = "func.call"(%arg0) {callee = @__nvqpp_cudaq_state_numberOfQubits} : (!cc.ptr<!cc.state>) -> i64
    //     //     %1 = "quake.alloca"(%0) : (i64) -> !quake.veq<?>
    //     //     %2 = "quake.init_state"(%1, %arg0) : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>
    //     //     "func.return"() : () -> ()
    //     // }) {"cudaq-entrypoint", "cudaq-kernel", function_type = (!cc.ptr<!cc.state>) -> (), no_this, sym_name = "__nvqpp__mlirgen__function_test2._Z5test2PN5cudaq5stateE"} : () -> ()
        
    //     std::cout << "test2(kernel): "  << "\n";
    //     std::vector<std::complex<double>> vec{.70710678, 0., 0., 0.70710678};
    //     auto kernel = cudaq::make_kernel();
    //     auto qubits = kernel.qalloc(2);

    //     cudaq::from_state(kernel, qubits, vec);

    //     // error: 'func.call' op '__nvqpp_cudaq_state_numberOfQubits' does not reference a valid function
    //     //auto state = cudaq::get_state(kernel);
    //     //auto counts = cudaq::sample(test2, &state);

    //     //counts.dump();
    // }

}