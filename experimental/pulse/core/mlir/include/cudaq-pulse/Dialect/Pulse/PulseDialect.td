/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_PULSE
#define CUDAQ_PULSE_DIALECT_PULSE

include "mlir/IR/OpBase.td"

def PulseDialect : Dialect {
  let name = "pulse";
  let summary = "Linear-typed MLIR dialect for pulse-level quantum control";
  let description = [{
    The `pulse` dialect provides explicit timing control and pulse-level
    operations on quantum hardware with compile-time verification of
    drive-operation exclusivity, monotone time, and tone-as-linear-resource
    semantics.

    Key concepts:
    - **Lines**: Drive and readout lines connected to qudits (linear resources)
    - **Tones**: Carrier signals with frequency and phase (linear resources)
    - **Waveforms**: Pulse envelopes (value types, freely copyable)
    - **Timing**: Current time tracked per line, explicit waits and syncs
    - **Waveform algebra**: Element-wise add/sub/mul/scale/neg on envelopes

    Example:
    ```mlir
      %q0 = pulse.qudit_alloc : !pulse.qref
      %d0, %tone_q0 = pulse.get_drive_line %q0
          : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
      %wf = pulse.gaussian 40, 0.3, 10.0 : !pulse.waveform
      %d0_out, %tone_out = pulse.drive %d0, %wf, %tone_q0
          : !pulse.drive_line, !pulse.waveform, !pulse.tone
          -> !pulse.drive_line, !pulse.tone
    ```
  }];
  let cppNamespace = "::pulse";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // CUDAQ_PULSE_DIALECT_PULSE
