# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

def test_kernel_subveqs():
    device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1" 
    #device_arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
    cudaq.set_target("braket",  machine=device_arn)
    #cudaq.set_target("ionq", emulate=True)

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(4)
        #h(qubits[0])
        # for qubit in range(3):
        #     x.ctrl(qubits[qubit], qubits[qubit + 1])
        x(qubits[1])
        x(qubits[2])
        v = qubits[1:3]
        mz(v)
        #mz(qubits)

    counts = cudaq.sample(kernel, shots_count=100)
    print(counts)
    # assert len(counts) == 2
    # assert "0000" in counts
    # assert "1111" in counts

test_kernel_subveqs()

def test_kernel_veqs():
    device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1" 
    cudaq.set_target("braket", emulate=True, machine=device_arn)
    #cudaq.set_target("braket", machine=device_arn)
    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        cx(qubits[0], qubits[1])
        mz(qubits)

    result = cudaq.sample(kernel, shots_count=1000)
    print(result)

#test_kernel_veqs()

# module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__kernel = "__nvqpp__mlirgen__kernel_PyKernelEntryPointRewrite"}} {
#   func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint"} {
#     %c1_i64 = arith.constant 1 : i64
#     %0 = quake.alloca !quake.veq<4>
#     %1 = quake.extract_ref %0[0] : (!quake.veq<4>) -> !quake.ref
#     quake.h %1 : (!quake.ref) -> ()
#     %2 = quake.extract_ref %0[1] : (!quake.veq<4>) -> !quake.ref
#     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
#     %3 = quake.extract_ref %0[2] : (!quake.veq<4>) -> !quake.ref
#     quake.x [%2] %3 : (!quake.ref, !quake.ref) -> ()
#     %4 = quake.extract_ref %0[3] : (!quake.veq<4>) -> !quake.ref
#     quake.x [%3] %4 : (!quake.ref, !quake.ref) -> ()
#     %5 = quake.subveq %0, %c1_i64, %c1_i64 : (!quake.veq<4>, i64, i64) -> !quake.veq<1>
#     %measOut = quake.mz %5 : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
#     return
#   }
# }

# => 

# module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__kernel = "__nvqpp__mlirgen__kernel_PyKernelEntryPointRewrite"}} {
#   func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint"} {
#     %c1_i64 = arith.constant 1 : i64
#     %0 = quake.alloca !quake.veq<4>
#     %1 = quake.extract_ref %0[0] : (!quake.veq<4>) -> !quake.ref
#     quake.h %1 : (!quake.ref) -> ()
#     %2 = quake.extract_ref %0[1] : (!quake.veq<4>) -> !quake.ref
#     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
#     %3 = quake.extract_ref %0[2] : (!quake.veq<4>) -> !quake.ref
#     quake.x [%2] %3 : (!quake.ref, !quake.ref) -> ()
#     %4 = quake.extract_ref %0[3] : (!quake.veq<4>) -> !quake.ref
#     quake.x [%3] %4 : (!quake.ref, !quake.ref) -> ()
#     #%5 = quake.subveq %0, %c1_i64, %c1_i64 : (!quake.veq<4>, i64, i64) -> !quake.veq<1>
#     %measOut = quake.mz %0 : (!quake.veq<4>) -> !cc.stdvec<!quake.measure> # add mapping to output only %0[1]
#     return
#   }
# }

# // -----// IR Dump After ConvertToQIR (quake-to-qir) ('builtin.module' operation) //----- //
# module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__kernel = "__nvqpp__mlirgen__kernel_PyKernelEntryPointRewrite"}} {
#   llvm.func @__quantum__rt__qubit_release_array(!llvm.ptr<struct<"Array", opaque>>)
#   llvm.func @__quantum__qis__mz(!llvm.ptr<struct<"Qubit", opaque>>) -> !llvm.ptr<struct<"Result", opaque>>
#   llvm.func @invokeWithControlQubits(i64, !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>, ...)
#   llvm.func @__quantum__qis__x__ctl(!llvm.ptr<struct<"Array", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>)
#   llvm.func @__quantum__qis__h(!llvm.ptr<struct<"Qubit", opaque>>)
#   llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#   llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr<struct<"Array", opaque>>
#   llvm.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint"} {
#     %0 = llvm.mlir.constant(4 : i64) : i64
#     %1 = llvm.call @__quantum__rt__qubit_allocate_array(%0) : (i64) -> !llvm.ptr<struct<"Array", opaque>>
#     %2 = llvm.mlir.constant(0 : i64) : i64
#     %3 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%1, %2) : (!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#     %4 = llvm.bitcast %3 : !llvm.ptr<i8> to !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %5 = llvm.load %4 : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     llvm.call @__quantum__qis__h(%5) : (!llvm.ptr<struct<"Qubit", opaque>>) -> ()
#     %6 = llvm.mlir.constant(1 : i64) : i64
#     %7 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%1, %6) : (!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#     %8 = llvm.bitcast %7 : !llvm.ptr<i8> to !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %9 = llvm.load %8 : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %10 = llvm.mlir.addressof @__quantum__qis__x__ctl : !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>
#     %11 = llvm.mlir.constant(1 : i64) : i64
#     llvm.call @invokeWithControlQubits(%11, %10, %5, %9) : (i64, !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>) -> ()
#     %12 = llvm.mlir.constant(2 : i64) : i64
#     %13 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%1, %12) : (!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#     %14 = llvm.bitcast %13 : !llvm.ptr<i8> to !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %15 = llvm.load %14 : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %16 = llvm.mlir.addressof @__quantum__qis__x__ctl : !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>
#     %17 = llvm.mlir.constant(1 : i64) : i64
#     llvm.call @invokeWithControlQubits(%17, %16, %9, %15) : (i64, !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>) -> ()
#     %18 = llvm.mlir.constant(3 : i64) : i64
#     %19 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%1, %18) : (!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#     %20 = llvm.bitcast %19 : !llvm.ptr<i8> to !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %21 = llvm.load %20 : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %22 = llvm.mlir.addressof @__quantum__qis__x__ctl : !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>
#     %23 = llvm.mlir.constant(1 : i64) : i64
#     llvm.call @invokeWithControlQubits(%23, %22, %15, %21) : (i64, !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>) -> ()
#     %24 = llvm.mlir.constant(1 : i32) : i32
#     %25 = llvm.alloca %24 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr<array<2 x i8>>
#     %26 = llvm.mlir.addressof @cstr.72303030303000 : !llvm.ptr<array<7 x i8>>
#     %27 = llvm.bitcast %26 : !llvm.ptr<array<7 x i8>> to !llvm.ptr<i8>
#     %28 = llvm.call @__quantum__qis__mz(%9) {registerName = "r00000"} : (!llvm.ptr<struct<"Qubit", opaque>>) -> !llvm.ptr<struct<"Result", opaque>>
#     %29 = llvm.bitcast %28 : !llvm.ptr<struct<"Result", opaque>> to !llvm.ptr<i1>
#     %30 = llvm.load %29 : !llvm.ptr<i1>
#     %31 = llvm.bitcast %25 : !llvm.ptr<array<2 x i8>> to !llvm.ptr<i8>
#     %32 = llvm.zext %30 : i1 to i8
#     llvm.store %32, %31 : !llvm.ptr<i8>
#     %33 = llvm.mlir.addressof @cstr.72303030303100 : !llvm.ptr<array<7 x i8>>
#     %34 = llvm.bitcast %33 : !llvm.ptr<array<7 x i8>> to !llvm.ptr<i8>
#     %35 = llvm.call @__quantum__qis__mz(%15) {registerName = "r00001"} : (!llvm.ptr<struct<"Qubit", opaque>>) -> !llvm.ptr<struct<"Result", opaque>>
#     %36 = llvm.bitcast %35 : !llvm.ptr<struct<"Result", opaque>> to !llvm.ptr<i1>
#     %37 = llvm.load %36 : !llvm.ptr<i1>
#     %38 = llvm.getelementptr %25[0, 1] : (!llvm.ptr<array<2 x i8>>) -> !llvm.ptr<i8>
#     %39 = llvm.zext %37 : i1 to i8
#     llvm.store %39, %38 : !llvm.ptr<i8>
#     llvm.call @__quantum__rt__qubit_release_array(%1) : (!llvm.ptr<struct<"Array", opaque>>) -> ()
#     llvm.return
#   }
#   llvm.mlir.global private constant @cstr.72303030303000("r00000\00") {addr_space = 0 : i32}
#   llvm.mlir.global private constant @cstr.72303030303100("r00001\00") {addr_space = 0 : i32}
# }

# // -----// IR Dump After Canonicalizer (canonicalize) ('func.func' operation: @__nvqpp__mlirgen__kernel) //----- //
# module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__kernel = "__nvqpp__mlirgen__kernel_PyKernelEntryPointRewrite"}} {
#   func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint"} {
#     %c2_i64 = arith.constant 2 : i64
#     %c1_i64 = arith.constant 1 : i64
#     %0 = quake.alloca !quake.veq<4>
#     %1 = quake.extract_ref %0[0] : (!quake.veq<4>) -> !quake.ref
#     quake.h %1 : (!quake.ref) -> ()
#     %2 = quake.extract_ref %0[1] : (!quake.veq<4>) -> !quake.ref
#     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
#     %3 = quake.extract_ref %0[2] : (!quake.veq<4>) -> !quake.ref
#     quake.x [%2] %3 : (!quake.ref, !quake.ref) -> ()
#     %4 = quake.extract_ref %0[3] : (!quake.veq<4>) -> !quake.ref
#     quake.x [%3] %4 : (!quake.ref, !quake.ref) -> ()
#     %5 = quake.subveq %0, %c1_i64, %c2_i64 : (!quake.veq<4>, i64, i64) -> !quake.veq<2>
#     %measOut = quake.mz %5 : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
#     return
#   }
# }

# // IONQ:


# // -----// IR Dump After QIRToQIRProfileFunc (quake-to-qir-func) ('llvm.func' operation: @__nvqpp__mlirgen__kernel) //----- //
# module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__kernel = "__nvqpp__mlirgen__kernel_PyKernelEntryPointRewrite"}} {
#   llvm.func @__quantum__qis__h__body(!llvm.ptr<struct<"Qubit", opaque>>)
#   llvm.func @__quantum__qis__cnot__body(!llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>)
#   llvm.func @__quantum__qis__cz__body(!llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>)
#   llvm.func @__quantum__rt__result_record_output(!llvm.ptr<struct<"Result", opaque>>, !llvm.ptr<i8>)
#   llvm.func @__quantum__qis__read_result__body(!llvm.ptr<struct<"Result", opaque>>) -> i1
#   llvm.func @__quantum__qis__mz__body(!llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Result", opaque>>) attributes {passthrough = ["irreversible"]}
#   llvm.func @__quantum__qis__cz(!llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>)
#   llvm.func @__quantum__qis__cnot(!llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>)
#   llvm.func @__quantum__rt__qubit_release_array(!llvm.ptr<struct<"Array", opaque>>)
#   llvm.func @__quantum__qis__mz(!llvm.ptr<struct<"Qubit", opaque>>) -> !llvm.ptr<struct<"Result", opaque>> attributes {passthrough = ["irreversible"]}
#   llvm.func @invokeWithControlQubits(i64, !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>, ...)
#   llvm.func @__quantum__qis__x__ctl(!llvm.ptr<struct<"Array", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>)
#   llvm.func @__quantum__qis__h(!llvm.ptr<struct<"Qubit", opaque>>)
#   llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#   llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr<struct<"Array", opaque>>
#   llvm.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint", passthrough = ["entry_point", ["qir_profiles", "base_profile"], ["output_labeling_schema", "schema_id"], ["output_names", "[[[0,[1,\22r00000\22]],[1,[2,\22r00001\22]]]]"], ["requiredQubits", "4"], ["requiredResults", "2"]]} {
#     %0 = llvm.mlir.constant(4 : i64) : i64
#     %1 = llvm.call @__quantum__rt__qubit_allocate_array(%0) {StartingOffset = 0 : i64} : (i64) -> !llvm.ptr<struct<"Array", opaque>>
#     %2 = llvm.mlir.constant(0 : i64) : i64
#     %3 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%1, %2) : (!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#     %4 = llvm.bitcast %3 : !llvm.ptr<i8> to !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %5 = llvm.load %4 : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     llvm.call @__quantum__qis__h(%5) : (!llvm.ptr<struct<"Qubit", opaque>>) -> ()
#     %6 = llvm.mlir.constant(1 : i64) : i64
#     %7 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%1, %6) : (!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#     %8 = llvm.bitcast %7 : !llvm.ptr<i8> to !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %9 = llvm.load %8 : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %10 = llvm.mlir.addressof @__quantum__qis__x__ctl : !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>
#     %11 = llvm.mlir.constant(1 : i64) : i64
#     llvm.call @invokeWithControlQubits(%11, %10, %5, %9) : (i64, !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>) -> ()
#     %12 = llvm.mlir.constant(2 : i64) : i64
#     %13 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%1, %12) : (!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#     %14 = llvm.bitcast %13 : !llvm.ptr<i8> to !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %15 = llvm.load %14 : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %16 = llvm.mlir.addressof @__quantum__qis__x__ctl : !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>
#     %17 = llvm.mlir.constant(1 : i64) : i64
#     llvm.call @invokeWithControlQubits(%17, %16, %9, %15) : (i64, !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>) -> ()
#     %18 = llvm.mlir.constant(3 : i64) : i64
#     %19 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%1, %18) : (!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>
#     %20 = llvm.bitcast %19 : !llvm.ptr<i8> to !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %21 = llvm.load %20 : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
#     %22 = llvm.mlir.addressof @__quantum__qis__x__ctl : !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>
#     %23 = llvm.mlir.constant(1 : i64) : i64
#     llvm.call @invokeWithControlQubits(%23, %22, %15, %21) : (i64, !llvm.ptr<func<void (ptr<struct<"Array", opaque>>, ptr<struct<"Qubit", opaque>>)>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>) -> ()
#     %24 = llvm.mlir.constant(1 : i32) : i32
#     %25 = llvm.alloca %24 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr<array<2 x i8>>
#     %26 = llvm.bitcast %25 : !llvm.ptr<array<2 x i8>> to !llvm.ptr<i8>
#     %27 = llvm.getelementptr %25[0, 1] : (!llvm.ptr<array<2 x i8>>) -> !llvm.ptr<i8>
#     %28 = llvm.mlir.addressof @cstr.72303030303000 : !llvm.ptr<array<7 x i8>>
#     %29 = llvm.bitcast %28 : !llvm.ptr<array<7 x i8>> to !llvm.ptr<i8>
#     %30 = llvm.call @__quantum__qis__mz(%9) {registerName = "r00000", result.index = 0 : i64} : (!llvm.ptr<struct<"Qubit", opaque>>) -> !llvm.ptr<struct<"Result", opaque>>
#     %31 = llvm.bitcast %30 : !llvm.ptr<struct<"Result", opaque>> to !llvm.ptr<i1>
#     %32 = llvm.load %31 : !llvm.ptr<i1>
#     %33 = llvm.zext %32 : i1 to i8
#     llvm.store %33, %26 : !llvm.ptr<i8>
#     %34 = llvm.mlir.addressof @cstr.72303030303100 : !llvm.ptr<array<7 x i8>>
#     %35 = llvm.bitcast %34 : !llvm.ptr<array<7 x i8>> to !llvm.ptr<i8>
#     %36 = llvm.call @__quantum__qis__mz(%15) {registerName = "r00001", result.index = 1 : i64} : (!llvm.ptr<struct<"Qubit", opaque>>) -> !llvm.ptr<struct<"Result", opaque>>
#     %37 = llvm.bitcast %36 : !llvm.ptr<struct<"Result", opaque>> to !llvm.ptr<i1>
#     %38 = llvm.load %37 : !llvm.ptr<i1>
#     %39 = llvm.zext %38 : i1 to i8
#     llvm.store %39, %27 : !llvm.ptr<i8>
#     llvm.call @__quantum__rt__qubit_release_array(%1) : (!llvm.ptr<struct<"Array", opaque>>) -> ()
#     %40 = llvm.mlir.constant(0 : i64) : i64
#     %41 = llvm.inttoptr %40 : i64 to !llvm.ptr<struct<"Result", opaque>>
#     %42 = llvm.mlir.addressof @cstr.72303030303000 : !llvm.ptr<array<7 x i8>>
#     %43 = llvm.bitcast %42 : !llvm.ptr<array<7 x i8>> to !llvm.ptr<i8>
#     llvm.call @__quantum__rt__result_record_output(%41, %43) : (!llvm.ptr<struct<"Result", opaque>>, !llvm.ptr<i8>) -> ()
#     %44 = llvm.mlir.constant(1 : i64) : i64
#     %45 = llvm.inttoptr %44 : i64 to !llvm.ptr<struct<"Result", opaque>>
#     %46 = llvm.mlir.addressof @cstr.72303030303100 : !llvm.ptr<array<7 x i8>>
#     %47 = llvm.bitcast %46 : !llvm.ptr<array<7 x i8>> to !llvm.ptr<i8>
#     llvm.call @__quantum__rt__result_record_output(%45, %47) : (!llvm.ptr<struct<"Result", opaque>>, !llvm.ptr<i8>) -> ()
#     llvm.return
#   }
#   llvm.mlir.global private constant @cstr.72303030303000("r00000\00") {addr_space = 0 : i32}
#   llvm.mlir.global private constant @cstr.72303030303100("r00001\00") {addr_space = 0 : i32}
# }