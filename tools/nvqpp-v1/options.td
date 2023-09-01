/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

// A short name to show in documentation. The name will be interpreted as rST.
class DocName<string name> { string DocName = name; }

// A brief description to show in documentation, interpreted as rST.
class DocBrief<code descr> { code DocBrief = descr; }

/////////
// Groups
def Quake_Compile_Group : OptionGroup<"<QuakeCompile group>">, DocName<"Quake Compilation flags">,
                   DocBrief<[{Flags controlling the behavior of cudaq-quake compilation.}]>;

def Quake_Opt_Group : OptionGroup<"<QuakeOpt group>">, DocName<"Quake Optimization flags">,
                   DocBrief<[{Flags controlling the behavior of cudaq-opt.}]>;

def IonQ_Group : OptionGroup<"<IonQ group>">, DocName<"IonQ flags">,
                   DocBrief<[{Flags controlling the behavior of IonQ target.}]>;

//===----------------------------------------------------------------------===//
// Quake Options
//===----------------------------------------------------------------------===//
def enable_mlir: Flag<["--"], "enable-mlir">, Group<Quake_Compile_Group>,
  HelpText<"Enable and generate MLIR code for CUDA Quantum kernels.">;


//===----------------------------------------------------------------------===//
// Quake Opt Options
//===----------------------------------------------------------------------===//
def device_code_loading : Flag<["--"], "device-code-loading">,  Group<Quake_Opt_Group>,
  HelpText<"Enable device code loading pass.">;
def no_device_code_loading : Flag<["--"], "no-device-code-loading">, Group<Quake_Opt_Group>,
  HelpText<"Disable device code loading pass.">;
def kernel_execution : Flag<["--"], "kernel-execution">,  Group<Quake_Opt_Group>,
  HelpText<"Enable kernel execution pass.">;
def no_kernel_execution : Flag<["--"], "no-kernel-execution">, Group<Quake_Opt_Group>,
  HelpText<"Disable kernel execution pass.">;
def aggressive_early_inline : Flag<["--"], "aggressive-early-inline">,  Group<Quake_Opt_Group>,
  HelpText<"Enable early inlining pass.">;
def no_aggressive_early_inline : Flag<["--"], "no-aggressive-early-inline">, Group<Quake_Opt_Group>,
  HelpText<"Disable early inlining pass.">;
def quake_apply_specialization : Flag<["--"], "quake-apply-specialization">,  Group<Quake_Opt_Group>,
  HelpText<"Enable specialization of quake.apply ops.">;
def no_quake_apply_specialization : Flag<["--"], "no-quake-apply-specialization">, Group<Quake_Opt_Group>,
  HelpText<"Disable specialization of quake.apply ops.">;
def lambda_lifting : Flag<["--"], "lambda-lifting">,  Group<Quake_Opt_Group>,
  HelpText<"Enable lambda lifting pass.">;
def no_lambda_lifting : Flag<["--"], "no-lambda-lifting">, Group<Quake_Opt_Group>,
  HelpText<"Disable lambda lifting pass.">;

//===----------------------------------------------------------------------===//
// IonQ Options
//===----------------------------------------------------------------------===//

def ionq_machine : Separate<["--"], "ionq-machine">, 
    HelpText<"Select the given IonQ backend target.">, Group<IonQ_Group>;
def ionq_noise_model : Separate<["--"], "ionq-noise-model">, 
    HelpText<"Select the given IonQ noise model when using the simulator backend.">, Group<IonQ_Group>;
