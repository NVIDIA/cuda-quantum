/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// This file provides some common QIR function names for use throughout our
/// MLIR lowering infrastructure.

namespace cudaq::opt {

/// QIS Function name strings
static constexpr const char QIRQISPrefix[] = "__quantum__qis__";
static constexpr const char QIRMeasureBody[] = "__quantum__qis__mz__body";
static constexpr const char QIRMeasure[] = "__quantum__qis__mz";
static constexpr const char QIRMeasureToRegister[] =
    "__quantum__qis__mz__to__register";
static constexpr const char QIRResetBody[] = "__quantum__qis__reset__body";
static constexpr const char QIRReset[] = "__quantum__qis__reset";

static constexpr const char QIRCnot[] = "__quantum__qis__cnot__body";
static constexpr const char QIRCphase[] = "__quantum__qis__cphase";
static constexpr const char QIRCZ[] = "__quantum__qis__cz__body";
static constexpr const char QIRReadResultBody[] = "__quantum__rt__read_result";

static constexpr const char QIRCustomOp[] = "__quantum__qis__custom_unitary";
static constexpr const char QIRCustomAdjOp[] =
    "__quantum__qis__custom_unitary__adj";
static constexpr const char QIRExpPauli[] = "__quantum__qis__exp_pauli";

static constexpr const char NVQIRInvokeWithControlBits[] =
    "invokeWithControlQubits";
static constexpr const char NVQIRInvokeRotationWithControlBits[] =
    "invokeRotationWithControlQubits";
static constexpr const char NVQIRInvokeU3RotationWithControlBits[] =
    "invokeU3RotationWithControlQubits";
static constexpr const char NVQIRInvokeWithControlRegisterOrBits[] =
    "invokeWithControlRegisterOrQubits";
static constexpr const char NVQIRGeneralizedInvokeAny[] =
    "generalizedInvokeWithRotationsControlsTargets";
static constexpr const char NVQIRPackSingleQubitInArray[] =
    "packSingleQubitInArray";
static constexpr const char NVQIRReleasePackedQubitArray[] =
    "releasePackedQubitArray";

/// QIR Array function name strings
static constexpr const char QIRArrayGetElementPtr1d[] =
    "__quantum__rt__array_get_element_ptr_1d";
static constexpr const char QIRArrayQubitAllocateArray[] =
    "__quantum__rt__qubit_allocate_array";
static constexpr const char QIRArrayQubitAllocateArrayWithStateFP64[] =
    "__quantum__rt__qubit_allocate_array_with_state_fp64";
static constexpr const char QIRArrayQubitAllocateArrayWithStateFP32[] =
    "__quantum__rt__qubit_allocate_array_with_state_fp32";
static constexpr const char QIRArrayQubitAllocateArrayWithStateComplex64[] =
    "__quantum__rt__qubit_allocate_array_with_state_complex64";
static constexpr const char QIRArrayQubitAllocateArrayWithStateComplex32[] =
    "__quantum__rt__qubit_allocate_array_with_state_complex32";
static constexpr const char QIRArrayQubitAllocateArrayWithStatePtr[] =
    "__quantum__rt__qubit_allocate_array_with_state_ptr";
static constexpr const char QIRArrayQubitAllocateArrayWithCudaqStatePtr[] =
    "__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr";
static constexpr const char QIRQubitAllocate[] =
    "__quantum__rt__qubit_allocate";
static constexpr const char QIRArrayQubitReleaseArray[] =
    "__quantum__rt__qubit_release_array";
static constexpr const char QIRArrayQubitReleaseQubit[] =
    "__quantum__rt__qubit_release";
static constexpr const char QIRArraySlice[] = "__quantum__rt__array_slice";
static constexpr const char QIRArrayGetSize[] =
    "__quantum__rt__array_get_size_1d";
static constexpr const char QIRArrayConcatArray[] =
    "__quantum__rt__array_concatenate";
static constexpr const char QIRArrayCreateArray[] =
    "__quantum__rt__array_create_1d";

/// Dynamic qubit management helper functions. These are currently only used by
/// the NVQIR simulator.
static constexpr const char QIRisDynamicQubitManagement[] =
    "__quantum__rt__is_dynamic_qubit_management";
static constexpr const char QIRsetDynamicQubitManagement[] =
    "__quantum__rt__set_dynamic_qubit_management";

/// QIR Base/Adaptive Profile record output function names
static constexpr const char QIRRecordOutput[] =
    "__quantum__rt__result_record_output";

/// Custom NVQIR method to cleanup result maps in between consecutive programs.
static constexpr const char QIRClearResultMaps[] =
    "__quantum__rt__clear_result_maps";

// Output logging function names.
static constexpr const char QIRBoolRecordOutput[] =
    "__quantum__rt__bool_record_output";
static constexpr const char QIRIntegerRecordOutput[] =
    "__quantum__rt__int_record_output";
static constexpr const char QIRDoubleRecordOutput[] =
    "__quantum__rt__double_record_output";
static constexpr const char QIRTupleRecordOutput[] =
    "__quantum__rt__tuple_record_output";
static constexpr const char QIRArrayRecordOutput[] =
    "__quantum__rt__array_record_output";

/// Used to specify the type of the data elements in the `QISApplyKrausChannel`
/// call. (`float` or `double`)
enum class KrausChannelDataKind { FloatKind, DoubleKind };

static constexpr const char QISApplyKrausChannel[] =
    "__quantum__qis__apply_kraus_channel_generalized";

static constexpr const char QISTrap[] = "__quantum__qis__trap";

/// Since apply noise is actually a call back to `C++` code, the `QIR` data type
/// `Array` of `Qubit*` must be converted into a `cudaq::qvector`, which is
/// presently a `std::vector<cudaq::qubit>` but with an extremely restricted
/// interface.
static constexpr const char QISConvertArrayToStdvec[] =
    "__quantum__qis__convert_array_to_stdvector";
static constexpr const char QISFreeConvertedStdvec[] =
    "__quantum__qis__free_converted_stdvector";

} // namespace cudaq::opt
