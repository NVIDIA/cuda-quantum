import cudaq

import numpy as np
import os
# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`

# To use the ORCA Computing target you will need to set the ORCA_ACCESS_URL
# environment variable or pass a URL.
orca_url = os.getenv("ORCA_ACCESS_URL", "http://localhost/sample")

cudaq.set_target("orca", url=orca_url)

# A time-bin boson sampling experiment: An input state of 4 indistinguishable
# photons mixed with 4 vacuum states across 8 time bins (modes) enter the
# time bin interferometer (TBI). The interferometer is composed of two loops
# each with a beam splitter (and optionally with a corresponding phase
# shifter). Each photon can either be stored in a loop to interfere with the
# next photon or exit the loop to be measured. Since there are 8 time bins
# and 2 loops, there is a total of 14 beam splitters (and optionally 14 phase
# shifters) in the interferometer, which is the number of controllable
# parameters.

# half of 8 time bins is filled with a single photon and the other half is
# filled with the vacuum state (empty)
input_state = [1, 0, 1, 0, 1, 0, 1, 0]

# The time bin interferometer in this example has two loops, each of length 1
loop_lengths = [1, 1]

# Calculate the number of beam splitters and phase shifters
n_beam_splitters = len(loop_lengths) * len(input_state) - sum(loop_lengths)

# beam splitter angles
bs_angles = np.linspace(np.pi / 8, np.pi / 3, n_beam_splitters)

# Optionally, we can also specify the phase shifter angles, if the system
# includes phase shifters
# ```
# ps_angles = np.linspace(np.pi / 6, np.pi / 3, n_beam_splitters)
# ```

# we can also set number of requested samples
n_samples = 10000

# By using the synchronous `cudaq.orca.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from ORCA Server.
counts = cudaq.orca.sample(input_state, loop_lengths, bs_angles, n_samples)

# If the system includes phase shifters, the phase shifter angles can be
# included in the call
# ```
# counts = cudaq.orca.sample(input_state, loop_lengths, bs_angles, ps_angles,
#                            n_samples)
# ```

# Print the results
print(counts)
