/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
/// All IR profile definitions, used to define compile targets in terms of the
/// successive compilation stages that a source program must be lowered through.
///
/// An IR profile (short: profile) is an IR 'language' in the CS sense of the
/// word, i.e. a declarative description of what IR is legal and illegal. The IR
/// remains the full source of truth:
///  - it should always be possible to validate that a given IR satisfies a
///  profile;
///  - it should always be possible to construct an IR profile from a given IR
///  that passes validation.

#include "QuakeReferenceSemanticsProfile.h"
#include "QuakeUnmappedProfile.h"
#include "UncheckedProfile.h"
#include <variant>

namespace cudaq::config {

/// All supported IR profiles.
using Profile = std::variant<UncheckedProfile, QuakeReferenceSemanticsProfile,
                             QuakeUnmappedProfile>;

} // namespace cudaq::config
