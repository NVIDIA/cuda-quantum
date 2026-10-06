/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_COMPILER_SPACETIMEPROVIDERS_H
#define QLX_COMPILER_SPACETIMEPROVIDERS_H

#include <optional>

namespace qlx::fabric {
class ProtocolOp;
}

namespace qlx::spacetime {

enum class BuiltinProviderKind {
  SurfaceAutoCCZFactory,
  SurfaceAutoCCZApplication,
  SurfaceSpacelikeCallable,
};

/// Structural summary of one boundary-preserving surface-code callable whose
/// exact P2 body contains authenticated AutoCCZ applications and exact CX
/// connections. An access layer connects an owner outside one reaction box;
/// reaction depth and peak width are derived from shared SSA owners. This
/// description is deliberately independent of helper names and algorithm
/// roles.
struct SurfaceSpacelikeShape {
  unsigned boundaryOwners;
  unsigned accessLayers;
  unsigned reactionDepth;
  unsigned reactionApplications;
  unsigned peakReactionWidth;
};

/// Derive the reusable spacelike-call shape from exact P2 SSA.  Returns no
/// value for ordinary or unsupported callables; callers must then use the
/// ordinary projection path.
std::optional<SurfaceSpacelikeShape>
surfaceSpacelikeShape(fabric::ProtocolOp protocol);

/// Select a built-in provider only after its complete P2 source contract has
/// been recognized. Unknown protocols remain on the ordinary projection path.
std::optional<BuiltinProviderKind> providerFor(fabric::ProtocolOp protocol);

/// Register the built-in compiler providers and their independent P3 plan
/// verifiers. This is intentionally separate from dialect initialization:
/// dialects define general IR, while providers own reusable QEC and physical
/// implementation policies.
void registerBuiltinProviders();

} // namespace qlx::spacetime

#endif // QLX_COMPILER_SPACETIMEPROVIDERS_H
