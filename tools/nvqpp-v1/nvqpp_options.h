
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
namespace llvm {
namespace opt {
class OptTable;
}
} // namespace llvm
namespace cudaq {
namespace nvqpp {
namespace options {
enum NvqppFlags {
  NvqppOption = (1 << 20),
  NvqppCC1Option = (1 << 21),
};

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include "Options.inc"
  LastOption
#undef OPTION
};
const llvm::opt::OptTable &getDriverOptTable();
} // namespace options
} // namespace nvqpp
} // namespace cudaq