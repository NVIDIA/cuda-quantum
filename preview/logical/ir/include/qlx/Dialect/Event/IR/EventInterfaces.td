/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef EVENT_INTERFACES_TD
#define EVENT_INTERFACES_TD

include "mlir/IR/OpBase.td"

def Event_UniqueCarryOwnerInterface
    : TypeInterface<"UniqueCarryOwnerInterface"> {
  let cppNamespace = "::qlx::event";
  let description = [{
    Identifies a carry type whose values `event.try_take` must not carry
    more than once by owner: two carries of an interface-implementing type
    that report an equal, non-null `getUniqueOwnerKey()` are rejected. The
    key is typically a property of the type itself rather than of the
    value -- e.g. a type parameterized by a resource symbol would key on
    that symbol, so two carries of the exact same instantiated type are
    mutually exclusive regardless of whether they are the same SSA value.
    A carry type that does not implement this interface is never checked,
    as if it had no such notion of ownership. Expressing the check as an
    interface, rather than a concrete type comparison, lets `event` apply
    it without naming any other dialect's type.
  }];
  let methods = [
    InterfaceMethod<
      "Identity used to detect duplicate ownership among carries. Must "
      "not return a null attribute: a null key from two different "
      "implementers would be indistinguishable from a real collision.",
      "::mlir::Attribute", "getUniqueOwnerKey"
    >
  ];
}

#endif // EVENT_INTERFACES_TD
