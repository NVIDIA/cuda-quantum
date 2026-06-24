/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "KernelArgs.h"

namespace cudaq {

KernelArgs::KernelArgs(PackedArgs packed) {
  if (!packed.data.empty())
    store.add("", packed);
}

KernelArgs::KernelArgs(TypeErasedArgs rawArgs) {
  if (!rawArgs.empty())
    store.add("", rawArgs);
}

KernelArgs::KernelArgs(PackedArgs packed, TypeErasedArgs rawArgs)
    : KernelArgs(std::move(packed)) {
  if (!rawArgs.empty())
    store.add("", rawArgs);
}

std::optional<KernelArgs::PackedArgs> KernelArgs::getPacked() const {
  auto ptr = store.get<PackedArgs>("");
  return ptr ? std::optional<PackedArgs>{*ptr} : std::nullopt;
}

std::optional<KernelArgs::TypeErasedArgs> KernelArgs::getTypeErased() const {
  auto ptr = store.get<TypeErasedArgs>("");
  return ptr ? std::optional<TypeErasedArgs>{*ptr} : std::nullopt;
}

} // namespace cudaq
