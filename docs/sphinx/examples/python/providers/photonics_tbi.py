import cudaq
import numpy as np

from dataclasses import dataclass, field

cudaq.set_target("photonics")

# Global variables
zero = 0
one = 1
n_modes = 4
input_state = [2, 1, 3, 1]
d = sum(input_state)
n_loops = 2
loop_lengths = [1, 2]
sum_loop_lengths = sum(loop_lengths)
n_beamsplitters = n_loops * n_modes - sum_loop_lengths


@dataclass
class TBIParameters:
    bs_angles: list[float]
    ps_angles: list[float]
    input_state: list = field(default_factory=lambda: input_state)
    loop_lengths: list = field(default_factory=lambda: loop_lengths)
    n_samples: int = 1000000


@cudaq.kernel
def TBI(parameters: TBIParameters):
    bs_angles = parameters.bs_angles
    ps_angles = parameters.ps_angles

    quds = [qudit(d) for _ in range(n_modes)]

    for i in range(n_modes):
        for _ in range(input_state[i]):
            plus(quds[i])

    c = 0
    for j in loop_lengths:
        for i in range(n_modes - j):
            beam_splitter(quds[i], quds[i + j], bs_angles[c])
            phase_shift(quds[i], ps_angles[c])
            c += 1

    mz(quds)


bs_angles = np.linspace(np.pi / 3, np.pi / 6, n_beamsplitters)
ps_angles = np.linspace(np.pi / 3, np.pi / 5, n_beamsplitters)

parameters = TBIParameters(bs_angles, ps_angles)

counts = cudaq.sample(TBI, parameters)
counts.dump()
