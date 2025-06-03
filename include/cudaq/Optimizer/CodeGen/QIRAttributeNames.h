/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
static constexpr const char QIRRequiredQubitsAttrName[] = "required_num_qubits";
static constexpr const char QIRRequiredResultsAttrName[] =
    "required_num_results";
static constexpr const char QIRIrreversibleFlagName[] = "irreversible";
static constexpr const char QIRMajorVersionFlagName[] = "qir_major_version";
static constexpr const char QIRMinorVersionFlagName[] = "qir_minor_version";
static constexpr const char QIRDynamicQubitsManagementFlagName[] =
    "dynamic_qubit_management";
static constexpr const char QIRDynamicResultManagementFlagName[] =
    "dynamic_result_management";
static constexpr const char QIRIrFunctionsFlagName[] = "ir_functions";
static constexpr const char QIRIntComputationsFlagName[] = "int_computations";
static constexpr const char QIRFloatComputationsFlagName[] =
    "float_computations";
static constexpr const char QIRBackwardsBranchingFlagName[] =
    "backwards_branching";
static constexpr const char QIRMultipleTargetBranchingFlagName[] =
    "multiple_target_branching";
static constexpr const char QIRMultipleReturnPointsFlagName[] =
    "multiple_return_points";

static constexpr const char StartingOffsetAttrName[] = "StartingOffset";
static constexpr const char ResultIndexAttrName[] = "ResultIndex";
static constexpr const char MzAssignedNameAttrName[] = "MzAssignedName";

} // namespace cudaq::opt
