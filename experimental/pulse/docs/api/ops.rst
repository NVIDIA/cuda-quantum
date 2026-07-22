.. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Pulse Operations API
====================

Channel Access
--------------

.. autofunction:: cudaq_pulse.get_drive_line

.. autofunction:: cudaq_pulse.get_readout_line

Scheduling
----------

.. autofunction:: cudaq_pulse.drive

.. autofunction:: cudaq_pulse.readout

.. autofunction:: cudaq_pulse.wait

.. autofunction:: cudaq_pulse.sync

Phase and Frequency
-------------------

.. autofunction:: cudaq_pulse.shift_phase

.. autofunction:: cudaq_pulse.set_phase

.. autofunction:: cudaq_pulse.shift_frequency

.. autofunction:: cudaq_pulse.set_frequency

Waveform Constructors
---------------------

.. autofunction:: cudaq_pulse.gaussian

.. autofunction:: cudaq_pulse.square

.. autofunction:: cudaq_pulse.drag

.. autofunction:: cudaq_pulse.cosine

.. autofunction:: cudaq_pulse.tanh_ramp

.. autofunction:: cudaq_pulse.gaussian_square

.. autofunction:: cudaq_pulse.custom

.. autofunction:: cudaq_pulse.custom_samples

Waveform Arithmetic
-------------------

.. autofunction:: cudaq_pulse.wf_add

.. autofunction:: cudaq_pulse.wf_sub

.. autofunction:: cudaq_pulse.wf_mul

.. autofunction:: cudaq_pulse.wf_scale

.. autofunction:: cudaq_pulse.wf_neg
