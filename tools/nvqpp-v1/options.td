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

// NvqppOption - Denote nvq++ options
def NvqppOption : OptionFlag;

// NvqppCC1Option - Options that should be forwarded to the CC1 invocation
def NvqppCC1Option : OptionFlag;

/////////
// Groups
def Quake_Compile_Group : OptionGroup<"<QuakeCompile group>">, DocName<"Quake Compilation flags">,
                   DocBrief<[{Flags controlling the behavior of cudaq-quake compilation.}]>;

def Quake_Opt_Group : OptionGroup<"<QuakeOpt group>">, DocName<"Quake Optimization flags">,
                   DocBrief<[{Flags controlling the behavior of cudaq-opt.}]>;

def IonQ_Group : OptionGroup<"<IonQ group>">, DocName<"IonQ flags">,
                   DocBrief<[{Flags controlling the behavior of IonQ target.}]>;

def Quantinuum_Group: OptionGroup<"<Quantinuum group>">, DocName<"Quantinuum flags">,
                   DocBrief<[{Flags controlling the behavior of Quantinuum target.}]>;

//===----------------------------------------------------------------------===//
// General Options
//===----------------------------------------------------------------------===//
def help : Flag<["-", "--"], "help">, HelpText<"Display available options">, Flags<[NvqppOption]>;
def help_host_cxx : Flag<["-", "--"], "help-cxx-host">, HelpText<"Display available options of the host C++ compiler">, Flags<[NvqppOption]>;
def emulate : Flag<["-", "--"], "emulate">, HelpText<"For physical, remote QPU execution, enable local emulation for testing.">, Flags<[NvqppOption]>;
def target : Joined<["-", "--"], "target=">, HelpText<"Set the Target name.">, Flags<[NvqppOption]>;
def list_targets  : Flag<["--"], "list-targets ">, HelpText<"ist the available targets NVQ++ can compile to.">, Flags<[NvqppOption]>;

//===----------------------------------------------------------------------===//
// Quake Options
//===----------------------------------------------------------------------===//
def enable_mlir: Flag<["--"], "enable-mlir">, Group<Quake_Compile_Group>,
  HelpText<"Enable and generate MLIR code for CUDA Quantum kernels.">, Flags<[NvqppOption]>;
// Just emit Quake, no further compilation
def emit_quake: Flag<["-", "--"], "emit-quake">, Group<Quake_Compile_Group>,
  HelpText<"Generate and emit MLIR code for CUDA Quantum kernels.">, Flags<[NvqppOption, NvqppCC1Option]>;

//===----------------------------------------------------------------------===//
// Quake Opt Options
//===----------------------------------------------------------------------===//
def device_code_loading : Flag<["--"], "device-code-loading">,  Group<Quake_Opt_Group>,
  HelpText<"Enable device code loading pass.">, Flags<[NvqppOption, NvqppCC1Option]>;
def no_device_code_loading : Flag<["--"], "no-device-code-loading">, Group<Quake_Opt_Group>,
  HelpText<"Disable device code loading pass.">, Flags<[NvqppOption, NvqppCC1Option]>;
def kernel_execution : Flag<["--"], "kernel-execution">,  Group<Quake_Opt_Group>,
  HelpText<"Enable kernel execution pass.">, Flags<[NvqppOption, NvqppCC1Option]>;
def no_kernel_execution : Flag<["--"], "no-kernel-execution">, Group<Quake_Opt_Group>,
  HelpText<"Disable kernel execution pass.">, Flags<[NvqppOption, NvqppCC1Option]>;
def aggressive_early_inline : Flag<["--"], "aggressive-early-inline">,  Group<Quake_Opt_Group>,
  HelpText<"Enable early inlining pass.">, Flags<[NvqppOption, NvqppCC1Option]>;
def no_aggressive_early_inline : Flag<["--"], "no-aggressive-early-inline">, Group<Quake_Opt_Group>,
  HelpText<"Disable early inlining pass.">, Flags<[NvqppOption, NvqppCC1Option]>;
def quake_apply_specialization : Flag<["--"], "quake-apply-specialization">,  Group<Quake_Opt_Group>,
  HelpText<"Enable specialization of quake.apply ops.">, Flags<[NvqppOption, NvqppCC1Option]>;
def no_quake_apply_specialization : Flag<["--"], "no-quake-apply-specialization">, Group<Quake_Opt_Group>,
  HelpText<"Disable specialization of quake.apply ops.">, Flags<[NvqppOption, NvqppCC1Option]>;
def lambda_lifting : Flag<["--"], "lambda-lifting">,  Group<Quake_Opt_Group>,
  HelpText<"Enable lambda lifting pass.">, Flags<[NvqppOption, NvqppCC1Option]>;
def no_lambda_lifting : Flag<["--"], "no-lambda-lifting">, Group<Quake_Opt_Group>,
  HelpText<"Disable lambda lifting pass.">, Flags<[NvqppOption, NvqppCC1Option]>;
def do_link : Flag<["--", "-"], "opt-do-link">,  Group<Quake_Opt_Group>,
  HelpText<"Internal flag used by driver to notify the frontent tool.">,
  Flags<[NvqppOption]>;
//===----------------------------------------------------------------------===//
// IonQ Options
//===----------------------------------------------------------------------===//

def ionq_machine : Separate<["--"], "ionq-machine">, 
    HelpText<"Select the given IonQ backend target.">, Group<IonQ_Group>, Flags<[NvqppOption]>;
def ionq_noise_model : Separate<["--"], "ionq-noise-model">, 
    HelpText<"Select the given IonQ noise model when using the simulator backend.">, Group<IonQ_Group>, Flags<[NvqppOption]>;


//===----------------------------------------------------------------------===//
// Quantinuum Options
//===----------------------------------------------------------------------===//

def quantinuum_url : Separate<["--"], "quantinuum-url">, 
    HelpText<"Set the Quantinuum server URL.">, Group<Quantinuum_Group>, Flags<[NvqppOption]>;
def quantinuum_machine : Separate<["--"], "quantinuum-machine">, 
    HelpText<"Select the given Quantinuum backend target.">, Group<Quantinuum_Group>, Flags<[NvqppOption]>;
