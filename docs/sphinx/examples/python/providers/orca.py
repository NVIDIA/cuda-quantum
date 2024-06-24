import cudaq

import numpy as np
import os

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`

# To use the ORCA target you will need to set the ORCA_ACCESS_URL environment variable
# or pass a url.
orca_url = os.getenv("ORCA_ACCESS_URL", "http://localhost/sample")

cudaq.set_target("orca", url=orca_url)
# ORCA's PT-Series implement the boson sampling model of quantum computation,
# in which multiple photons are interfered with each other within a network of
# beam splitters, and photon detectors measure where the photons leave this
# network.

# The parameters needed to define the time bin interferometer are the
# the input state, the loop lengths, beam splitter angles, and optionally the
# phase shifter angles, and the number of samples.

# The input state is the initial state of the photons in
#  the time bin interferometer, the left-most entry corresponds to the first
#  mode entering the loop.

# The loop lengths are the the lengths of the different loops in the time bin
# interferometer.

# The beam splitter angles and the phase shifter angles are controllable
# parameters of the time bin interferometer.

# A time-bin boson sampling experiment: An input state of 4 indistinguishable
# photons mixed with 4 vacuum states across 8 time bins (modes) enter the
# time bin interferometer (TBI). The interferometer is composed of two loops
# each with a beam splitter (and optionally with a correspondent phase
# shifter). Each photon can either be stored in a loop to interfere with the
# next photon or exit the loop to be measured. Since there are 8 time time
# bins and 2 loops, there are a total of 14 beam splitters (and optionally 14
# phase shifters in the interferometer), which is the number of controllable
# parameters.

# half of 8 time bins is filled with a single photon and the other half is
# filled with the vacuum state (empty)
input_state = [1, 0, 1, 0, 1, 0, 1, 0]

# The time bin interferometer in this example has two loops, each of length 1
loop_lengths = [1, 1]

# Calculate the number of beam splitters and phase shifters
n_beamsplitters = len(loop_lengths) * len(input_state) - sum(loop_lengths)

# beam splitter angles
bs_angles = np.linspace(np.pi / 8, np.pi / 3, n_beamsplitters)

# Optionally, we can also specify the phase shifter angles, if the system includes phase shifters
# ps_angles = np.linspace(np.pi / 6, np.pi / 3, n_beamsplitters)

# we can also set number of requested samples
n_samples = 10000

# By using the synchronous `cudaq.orca.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from ORCA Server.
counts = cudaq.orca.sample(input_state, loop_lengths, bs_angles, n_samples)

# If the system includes phase shifters, the phase shifter angles can be inluded in the call
# counts = cudaq.orca.sample(input_state, loop_lengths, bs_angles, ps_angles, n_samples)

# Print the results
print(counts)
