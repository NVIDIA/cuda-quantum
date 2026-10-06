/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef TRIVIAL_OPS
#define TRIVIAL_OPS

include "Trivial/TrivialDialect.td"

def Trivial_NopOp : Trivial_Op<"nop"> {
  let summary = "A no-op carrying no operands or results.";
  let description = [{
    A placeholder operation with no operands, results, or regions. It exists so
    the example exercises TableGen op generation (`-gen-op-decls`/`-gen-op-defs`)
    end to end.
  }];
  let assemblyFormat = "attr-dict";
}

#endif // TRIVIAL_OPS
