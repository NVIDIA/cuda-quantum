/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_PULSE_OPS
#define CUDAQ_PULSE_DIALECT_PULSE_OPS

include "cudaq-pulse/Dialect/Pulse/PulseDialect.td"
include "cudaq-pulse/Dialect/Pulse/PulseTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpBase.td"

class PulseOp<string mnemonic, list<Trait> traits = []> :
    Op<PulseDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Qudit allocation
//===----------------------------------------------------------------------===//

def QuditAllocOp : PulseOp<"qudit_alloc", [Pure]> {
  let summary = "Allocate a pulse-level qudit reference";
  let results = (outs QrefType:$qref);
  let assemblyFormat = "attr-dict `:` type($qref)";
}

def QvecAllocOp : PulseOp<"qvec_alloc", [Pure]> {
  let summary = "Allocate a vector of pulse-level qudit references";
  let arguments = (ins I64Attr:$size);
  let results = (outs Variadic<QrefType>:$qrefs);
  let assemblyFormat = "$size attr-dict `:` type($qrefs)";
}

//===----------------------------------------------------------------------===//
// Line operations
//===----------------------------------------------------------------------===//

def GetDriveLineOp : PulseOp<"get_drive_line", [Pure]> {
  let summary = "Get the drive line and tone for a qudit";
  let description = [{
    Returns the drive line and native tone associated with a qudit.
    The returned drive line starts at time t=0.
  }];
  let arguments = (ins QrefType:$qudit);
  let results = (outs DriveLineType:$line, ToneType:$tone);
  let assemblyFormat = "$qudit attr-dict `:` functional-type(operands, results)";
}

def GetReadoutLineOp : PulseOp<"get_readout_line", [Pure]> {
  let summary = "Get the readout line and tone for a qudit";
  let arguments = (ins QrefType:$qudit);
  let results = (outs ReadoutLineType:$line, ToneType:$tone);
  let assemblyFormat = "$qudit attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Tone operations
//===----------------------------------------------------------------------===//

def ToneOp : PulseOp<"tone", [Pure]> {
  let summary = "Create a pulse tone with explicit frequency and phase";
  let arguments = (ins
    AnyFloat:$frequency_hz,
    AnyFloat:$phase_rad
  );
  let results = (outs ToneType:$tone);
  let assemblyFormat = "$frequency_hz `,` $phase_rad attr-dict `:` qualified(type($frequency_hz)) `,` qualified(type($phase_rad)) `->` qualified(type($tone))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Waveform construction
//===----------------------------------------------------------------------===//

def SquarePulseOp : PulseOp<"square", [Pure]> {
  let summary = "Create a constant-amplitude (square) pulse";
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amp_real,
    AnyFloat:$amp_imag
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amp_real `,` $amp_imag attr-dict `:` type($duration) `,` type($amp_real) `,` type($amp_imag) `->` type($pulse)";
  let hasVerifier = 1;
}

def GaussianPulseOp : PulseOp<"gaussian", [Pure]> {
  let summary = "Create a Gaussian pulse";
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude,
    AnyFloat:$sigma
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude `,` $sigma attr-dict `:` type($duration) `,` type($amplitude) `,` type($sigma) `->` type($pulse)";
  let hasVerifier = 1;
}

def GaussianSquarePulseOp : PulseOp<"gaussian_square", [Pure]> {
  let summary = "Create a flat-topped Gaussian pulse";
  let description = [{
    A Gaussian rising edge, a flat top at the given amplitude, and a
    Gaussian falling edge. `risefall` is the duration of each edge in VTU.
  }];
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude,
    AnyFloat:$sigma,
    I64:$risefall
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude `,` $sigma `,` $risefall attr-dict `:` type($duration) `,` type($amplitude) `,` type($sigma) `,` type($risefall) `->` type($pulse)";
  let hasVerifier = 1;
}

def DRAGPulseOp : PulseOp<"drag", [Pure]> {
  let summary = "Create a DRAG pulse for reducing leakage";
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude,
    AnyFloat:$sigma,
    AnyFloat:$beta
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude `,` $sigma `,` $beta attr-dict `:` type($duration) `,` type($amplitude) `,` type($sigma) `,` type($beta) `->` type($pulse)";
  let hasVerifier = 1;
}

def CosinePulseOp : PulseOp<"cosine", [Pure]> {
  let summary = "Create a raised-cosine envelope pulse";
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude attr-dict `:` type($duration) `,` type($amplitude) `->` type($pulse)";
  let hasVerifier = 1;
}

def TanhRampOp : PulseOp<"tanh_ramp", [Pure]> {
  let summary = "Create a hyperbolic-tangent ramp pulse";
  let description = [{
    Smooth ramp from 0 to `amplitude` using tanh shaping. `sigma` controls
    the steepness of the ramp.
  }];
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude,
    AnyFloat:$sigma
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude `,` $sigma attr-dict `:` type($duration) `,` type($amplitude) `,` type($sigma) `->` type($pulse)";
  let hasVerifier = 1;
}

def CustomOp : PulseOp<"custom", [Pure]> {
  let summary = "Create a custom waveform via callback function";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    I64:$duration
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$callee `,` $duration attr-dict `:` type($duration) `->` type($pulse)";
  let hasVerifier = 1;
}

def CustomSamplesOp : PulseOp<"custom_samples", [Pure]> {
  let summary = "Create a waveform from raw sample data";
  let arguments = (ins
    F64ArrayAttr:$samples
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$samples attr-dict `:` type($pulse)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Pulse application operations
//===----------------------------------------------------------------------===//

def DriveOp : PulseOp<"drive", []> {
  let summary = "Apply a waveform to a drive line at a tone";
  let description = [{
    Applies a waveform to a drive line at the specified tone's frequency and
    phase. Advances line time by pulse duration. Returns the updated drive
    line and tone.
  }];
  let arguments = (ins
    DriveLineType:$line,
    WaveformType:$pulse,
    ToneType:$tone
  );
  let results = (outs DriveLineType:$updated_line, ToneType:$updated_tone);
  let assemblyFormat = [{
    $line `,` $pulse `,` $tone attr-dict `:` qualified(type($line))
    `,` qualified(type($pulse)) `,` qualified(type($tone))
    `->` qualified(type($updated_line)) `,` qualified(type($updated_tone))
  }];
}

def ReadoutOp : PulseOp<"readout", []> {
  let summary = "Apply a readout pulse and produce a measurement result";
  let arguments = (ins
    ReadoutLineType:$line,
    WaveformType:$pulse,
    ToneType:$tone,
    StrAttr:$mode
  );
  let results = (outs
    ReadoutLineType:$updated_line,
    ToneType:$updated_tone,
    MeasurementType:$result
  );
  let assemblyFormat = [{
    $line `,` $pulse `,` $tone `,` $mode attr-dict `:` qualified(type($line))
    `,` qualified(type($pulse)) `,` qualified(type($tone))
    `->` qualified(type($updated_line)) `,` qualified(type($updated_tone))
    `,` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// Timing operations
//===----------------------------------------------------------------------===//

def WaitOp : PulseOp<"wait", []> {
  let summary = "Idle a line for a specified duration";
  let arguments = (ins
    AnyLineType:$line,
    DurationType:$duration
  );
  let results = (outs AnyLineType:$updated_line);
  let assemblyFormat = "$line `,` $duration attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def SyncOp : PulseOp<"sync", []> {
  let summary = "Synchronize multiple lines to the same time";
  let description = [{
    Aligns time state of multiple lines. All returned lines will have
    time = max(T1, ..., Tk).
  }];
  let arguments = (ins Variadic<AnyLineType>:$lines);
  let results = (outs Variadic<AnyLineType>:$synced_lines);
  let assemblyFormat = "$lines attr-dict `:` qualified(type($lines)) `->` qualified(type($synced_lines))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def GetCurrentTimeOp : PulseOp<"get_current_time", [Pure]> {
  let summary = "Get the current time of a line as a timepoint";
  let arguments = (ins AnyLineType:$line);
  let results = (outs TimepointType:$time);
  let assemblyFormat = "$line attr-dict `:` qualified(type($line)) `->` qualified(type($time))";
}

def TimepointToIntOp : PulseOp<"timepoint_to_int", [Pure]> {
  let summary = "Convert a timepoint to integer cycles";
  let arguments = (ins TimepointType:$time);
  let results = (outs I64:$time_in_cycles);
  let assemblyFormat = "$time attr-dict `:` qualified(type($time)) `->` qualified(type($time_in_cycles))";
}

def TimepointSubtractOp : PulseOp<"timepoint_subtract", [Pure]> {
  let summary = "Compute the duration between two timepoints";
  let arguments = (ins TimepointType:$lhs, TimepointType:$rhs);
  let results = (outs DurationType:$duration);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def DurationFromIntOp : PulseOp<"duration_from_int", [Pure]> {
  let summary = "Convert an integer to a duration";
  let arguments = (ins I64:$cycles);
  let results = (outs DurationType:$duration);
  let assemblyFormat = "$cycles attr-dict `:` functional-type(operands, results)";
}

def DurationToIntOp : PulseOp<"duration_to_int", [Pure]> {
  let summary = "Convert a duration to an integer";
  let arguments = (ins DurationType:$duration);
  let results = (outs I64:$cycles);
  let assemblyFormat = "$duration attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Tone modulation operations (zero-duration, instantaneous)
//===----------------------------------------------------------------------===//

def ShiftFrequencyOp : PulseOp<"shift_frequency", []> {
  let summary = "Apply a relative frequency shift to a tone";
  let arguments = (ins ToneType:$tone, F64:$frequency_hz);
  let results = (outs ToneType:$updated_tone);
  let assemblyFormat = "$tone `,` $frequency_hz attr-dict `:` qualified(type($tone)) `,` type($frequency_hz) `->` qualified(type($updated_tone))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def ShiftPhaseOp : PulseOp<"shift_phase", []> {
  let summary = "Apply a relative phase shift to a tone (virtual Z gate)";
  let arguments = (ins ToneType:$tone, F64:$phase_rad);
  let results = (outs ToneType:$updated_tone);
  let assemblyFormat = "$tone `,` $phase_rad attr-dict `:` qualified(type($tone)) `,` type($phase_rad) `->` qualified(type($updated_tone))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def SetPhaseOp : PulseOp<"set_phase", []> {
  let summary = "Set absolute phase of a tone";
  let arguments = (ins ToneType:$tone, F64:$phase_rad);
  let results = (outs ToneType:$updated_tone);
  let assemblyFormat = "$tone `,` $phase_rad attr-dict `:` qualified(type($tone)) `,` type($phase_rad) `->` qualified(type($updated_tone))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def SetFrequencyOp : PulseOp<"set_frequency", []> {
  let summary = "Set absolute frequency of a tone";
  let arguments = (ins ToneType:$tone, F64:$frequency_hz);
  let results = (outs ToneType:$updated_tone);
  let assemblyFormat = "$tone `,` $frequency_hz attr-dict `:` qualified(type($tone)) `,` type($frequency_hz) `->` qualified(type($updated_tone))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Waveform algebra
//===----------------------------------------------------------------------===//

def PulseAddOp : PulseOp<"add", [Pure, Commutative]> {
  let summary = "Add two waveforms element-wise";
  let arguments = (ins WaveformType:$lhs, WaveformType:$rhs);
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def PulseSubOp : PulseOp<"sub", [Pure]> {
  let summary = "Subtract two waveforms element-wise";
  let arguments = (ins WaveformType:$lhs, WaveformType:$rhs);
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def PulseMulOp : PulseOp<"mul", [Pure, Commutative]> {
  let summary = "Multiply two waveforms element-wise";
  let arguments = (ins WaveformType:$lhs, WaveformType:$rhs);
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def PulseScaleOp : PulseOp<"scale", [Pure]> {
  let summary = "Scale a waveform by a scalar factor";
  let arguments = (ins
    WaveformType:$pulse,
    AnyTypeOf<[AnyFloat, AnyComplex]>:$scale
  );
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$pulse `,` $scale attr-dict `:` qualified(type($pulse)) `,` type($scale) `->` qualified(type($result))";
  let hasFolder = 1;
}

def PulseNegOp : PulseOp<"neg", [Pure]> {
  let summary = "Negate a waveform";
  let arguments = (ins WaveformType:$pulse);
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$pulse attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Schedule and atomic region
//===----------------------------------------------------------------------===//

def AtomicOp : PulseOp<"atomic", [RecursiveMemoryEffects, NoTerminator]> {
  let summary = "Atomic pulse scheduling region";
  let arguments = (ins Variadic<AnyLineType>:$lines);
  let results = (outs Variadic<AnyLineType>:$updated_lines);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$lines attr-dict `:` functional-type($lines, $updated_lines) $body";
  let hasVerifier = 1;
}

def YieldOp : PulseOp<"yield", [Pure, Terminator]> {
  let summary = "Yield from a pulse region";
  let arguments = (ins Variadic<AnyLineType>:$lines);
  let assemblyFormat = "attr-dict (`(` $lines^ `:` type($lines) `)`)?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IQ operations
//===----------------------------------------------------------------------===//

def IQAcquireOp : PulseOp<"iq_acquire", []> {
  let summary = "Acquire raw IQ data from a readout line";
  let arguments = (ins
    ReadoutLineType:$line,
    WaveformType:$pulse,
    ToneType:$tone
  );
  let results = (outs ReadoutLineType:$updated_line, IQDataType:$iq_data);
  let assemblyFormat = [{
    $line `,` $pulse `,` $tone attr-dict `:` qualified(type($line))
    `,` qualified(type($pulse)) `,` qualified(type($tone))
    `->` qualified(type($updated_line)) `,` qualified(type($iq_data))
  }];
}

def ExtractIQOp : PulseOp<"extract_iq", [Pure]> {
  let summary = "Extract I and Q components from IQ data";
  let arguments = (ins IQDataType:$iq_data);
  let results = (outs F64:$i_value, F64:$q_value);
  let assemblyFormat = "$iq_data attr-dict `:` qualified(type($iq_data)) `->` type($i_value) `,` type($q_value)";
}

def CreateIQOp : PulseOp<"create_iq", [Pure]> {
  let summary = "Create IQ data from I and Q values";
  let arguments = (ins F64:$i_value, F64:$q_value);
  let results = (outs IQDataType:$iq_data);
  let assemblyFormat = "$i_value `,` $q_value attr-dict `:` type($i_value) `,` type($q_value) `->` qualified(type($iq_data))";
}

def IQMagnitudeOp : PulseOp<"iq_magnitude", [Pure]> {
  let summary = "Compute magnitude of IQ data: sqrt(I^2 + Q^2)";
  let arguments = (ins IQDataType:$iq_data);
  let results = (outs F64:$magnitude);
  let assemblyFormat = "$iq_data attr-dict `:` qualified(type($iq_data)) `->` type($magnitude)";
}

def IQPhaseOp : PulseOp<"iq_phase", [Pure]> {
  let summary = "Compute phase of IQ data: atan2(Q, I)";
  let arguments = (ins IQDataType:$iq_data);
  let results = (outs F64:$phase);
  let assemblyFormat = "$iq_data attr-dict `:` qualified(type($iq_data)) `->` type($phase)";
}

#endif // CUDAQ_PULSE_DIALECT_PULSE_OPS
