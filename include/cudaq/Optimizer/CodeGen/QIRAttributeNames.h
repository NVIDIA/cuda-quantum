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
static constexpr const char QIRRequiredQubitsAttrName[] = "requiredQubits";
static constexpr const char QIRRequiredResultsAttrName[] = "requiredResults";
static constexpr const char QIRIrreversibleFlagName[] = "irreversible";

static constexpr const char StartingOffsetAttrName[] = "StartingOffset";
static constexpr const char ResultIndexAttrName[] = "ResultIndex";
static constexpr const char MzAssignedNameAttrName[] = "MzAssignedName";

} // namespace cudaq::opt
