import cudaq

import numpy as np
import os

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`

# To use the ORCA target you will need to set the ORCA_ACCESS_URL environment variable
# or pass a url.
orca_url = os.getenv("ORCA_ACCESS_URL", "http://localhost:8080/sample")

cudaq.set_target("orca", url=orca_url)
# ORCA's PT-Series implement the boson sampling model of quantum computation,
# in which multiple photons are interfered with each other within a
# network of beam splitters, and photon detectors measure where the photons
# leave this network.
# The parameters needed to define the time bin interferometer are the beam
# splitter angles, the phase shifter angles, the input state, the loop lengths
# and optionally the number of samples.
# The input state is the initial state of the photons in the time bin
# interferometer, the left-most entry corresponds to the first mode entering
# the loop.
# The loop lengths are the the lengths of the different loops in the
# time bin interferometer.

# A time-bin boson sampling experiment: An input state of 3 indistinguishable
# photons across 3 time bins (modes) entering the time bin interferometer.
# The interferometer is composed of one loop and beam splitter with its
# correspondent phase shifter. Each photon can either be stored in the loop
# to interfere with the next photon or exit the loop to be measured.
# Since there are 3 time time bins, there are 2 beam splitters and 2 phase
# shifters in the interferometer.

# beam splitter angles
bs_angles = [np.pi / 3, np.pi / 6]

# phase shifter angles
ps_angles = [np.pi / 4, np.pi / 5]

# all time bins are filled with a single photon
input_state = [1, 1, 1]

# The time bin interferometer in this example has only one loop of length 1
loop_lengths = [1]

# By using the synchronous `cudaq.orca.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from ORCA Server.
counts = cudaq.orca.sample(bs_angles, ps_angles, input_state, loop_lengths)

print(counts)
