/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_PULSE_TYPES
#define CUDAQ_PULSE_DIALECT_PULSE_TYPES

include "cudaq-pulse/Dialect/Pulse/PulseDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class PulseType<string name, string typeMnemonic, list<Trait> traits = [],
        string baseCppClass = "mlir::Type">
    : TypeDef<PulseDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Qudit reference (opaque, default dim=2, future-proofed for d>2)
//===----------------------------------------------------------------------===//

def QrefType : PulseType<"Qref", "qref"> {
  let summary = "Opaque pulse-level qudit reference (dim defaults to 2)";
}

//===----------------------------------------------------------------------===//
// Measurement result
//===----------------------------------------------------------------------===//

def MeasurementType : PulseType<"Measurement", "measurement"> {
  let summary = "Classical bit produced by a readout operation";
}

//===----------------------------------------------------------------------===//
// Drive and readout lines (linear resources)
//===----------------------------------------------------------------------===//

def DriveLineType : PulseType<"DriveLine", "drive_line"> {
  let summary = "Physical drive channel (linear resource)";
  let description = [{
    A `drive_line` represents a physical drive channel. It is a linear resource:
    each value is produced exactly once and consumed exactly once. Operations
    that consume a drive line produce a new value with advanced time state.
  }];
}

def ReadoutLineType : PulseType<"ReadoutLine", "readout_line"> {
  let summary = "Physical readout channel (linear resource)";
  let description = [{
    A `readout_line` represents a physical readout channel. Like drive lines,
    readout lines are linear resources with time state.
  }];
}

//===----------------------------------------------------------------------===//
// Tone and timing types
//===----------------------------------------------------------------------===//

def ToneType : PulseType<"Tone", "tone"> {
  let summary = "Carrier signal tone (frequency and phase, linear resource)";
  let description = [{
    A `tone` represents a carrier tone tracking frequency and phase. Tones are
    separate from physical lines, enabling cross-resonance and
    position-independent calibrations.
  }];
}

def TimepointType : PulseType<"Timepoint", "timepoint"> {
  let summary = "Abstract point in time on a line";
}

def DurationType : PulseType<"Duration", "duration"> {
  let summary = "Time interval in virtual time units (VTU)";
}

//===----------------------------------------------------------------------===//
// Waveform (value type, not linear)
//===----------------------------------------------------------------------===//

def WaveformType : PulseType<"Waveform", "waveform"> {
  let summary = "Pulse envelope shape (value type, not linear)";
  let description = [{
    A `waveform` represents a time-domain pulse envelope. Waveforms are value
    types and can be freely passed around and applied to multiple lines.
  }];
}

//===----------------------------------------------------------------------===//
// IQ data
//===----------------------------------------------------------------------===//

def IQDataType : PulseType<"IQData", "iq_data"> {
  let summary = "Raw IQ signal data from readout (value type)";
}

//===----------------------------------------------------------------------===//
// Waveform enum attribute
//===----------------------------------------------------------------------===//

def WaveformAttr : I32EnumAttr<"WaveformKind", "Type of pulse waveform",
  [
    I32EnumAttrCase<"Square", 0, "square">,
    I32EnumAttrCase<"Gaussian", 1, "gaussian">,
    I32EnumAttrCase<"GaussianSquare", 2, "gaussian_square">,
    I32EnumAttrCase<"DRAG", 3, "drag">,
    I32EnumAttrCase<"Cosine", 4, "cosine">,
    I32EnumAttrCase<"TanhRamp", 5, "tanh_ramp">,
    I32EnumAttrCase<"Custom", 6, "custom">
  ]> {
  let cppNamespace = "::pulse";
}

//===----------------------------------------------------------------------===//
// Composite type predicates
//===----------------------------------------------------------------------===//

def AnyLineType : Type<
  Or<[DriveLineType.predicate, ReadoutLineType.predicate]>,
  "any pulse line type"
>;

#endif // CUDAQ_PULSE_DIALECT_PULSE_TYPES
