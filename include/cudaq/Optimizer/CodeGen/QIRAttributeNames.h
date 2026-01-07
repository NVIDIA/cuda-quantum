/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// This file provides some common QIR attribute names for use in code gen.

namespace cudaq::opt {

static constexpr const char QIRRegisterNameAttr[] = "registerName";
static constexpr const char QIREntryPointAttrName[] = "entry_point";
static constexpr const char QIRProfilesAttrName[] = "qir_profiles";
static constexpr const char QIROutputLabelingSchemaAttrName[] =
    "output_labeling_schema";
static constexpr const char QIROutputNamesAttrName[] = "output_names";
static constexpr const char QIRMajorVersionFlagName[] = "qir_major_version";
static constexpr const char QIRMinorVersionFlagName[] = "qir_minor_version";

static constexpr const char QIRIrreversibleFlagName[] = "irreversible";
static constexpr const char QIRDynamicQubitsManagementFlagName[] =
    "dynamic_qubit_management";
static constexpr const char QIRDynamicResultManagementFlagName[] =
    "dynamic_result_management";

static constexpr const char StartingOffsetAttrName[] = "StartingOffset";
static constexpr const char ResultIndexAttrName[] = "ResultIndex";
static constexpr const char MzAssignedNameAttrName[] = "MzAssignedName";

namespace qir0_1 {
static constexpr const char RequiredQubitsAttrName[] = "requiredQubits";
static constexpr const char RequiredResultsAttrName[] = "requiredResults";

static constexpr const char QubitResettingFlagName[] = "qubit_resetting";
static constexpr const char ClassicalIntsFlagName[] = "classical_ints";
static constexpr const char ClassicalFloatsFlagName[] = "classical_floats";
static constexpr const char ClassicalFixedPointsFlagName[] =
    "classical_fixed_points";
static constexpr const char UserFunctionsFlagName[] = "user_functions";
static constexpr const char DynamicFloatArgsFlagName[] = "dynamic_float_args";
static constexpr const char ExternFunctionsFlagName[] = "extern_functions";
static constexpr const char BackwardsBranchingFlagName[] =
    "backwards_branching";
} // namespace qir0_1

namespace qir1_0 {
static constexpr const char RequiredQubitsAttrName[] = "required_num_qubits";
static constexpr const char RequiredResultsAttrName[] = "required_num_results";

static constexpr const char IrFunctionsFlagName[] = "ir_functions";
static constexpr const char IntComputationsFlagName[] = "int_computations";
static constexpr const char FloatComputationsFlagName[] = "float_computations";
static constexpr const char BackwardsBranchingFlagName[] =
    "backwards_branching";
static constexpr const char MultipleTargetBranchingFlagName[] =
    "multiple_target_branching";
static constexpr const char MultipleReturnPointsFlagName[] =
    "multiple_return_points";
} // namespace qir1_0

} // namespace cudaq::opt
