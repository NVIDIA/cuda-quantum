import cudaq
import numpy as np

cudaq.set_target("orca-photonics")


@cudaq.kernel
def TBI(
    bs_angles: list[float],
    ps_angles: list[float],
    input_state: list[int],
    loop_lengths: list[int],
):
    n_modes = len(input_state)
    level = sum(input_state) + 1  # qudit level

    qumodes = [qudit(level) for _ in range(n_modes)]

    for i in range(n_modes):
        for _ in range(input_state[i]):
            create(qumodes[i])

    counter = 0
    for j in loop_lengths:
        for i in range(n_modes - j):
            beam_splitter(qumodes[i], qumodes[i + j], bs_angles[counter])
            phase_shift(qumodes[i], ps_angles[counter])
            counter += 1

    mz(qumodes)


input_state = [1, 0, 1, 0, 1, 0, 1, 0]
loop_lengths = [1, 1]

n_beam_splitters = len(loop_lengths) * len(input_state) - sum(loop_lengths)
bs_angles = np.linspace(np.pi / 3, np.pi / 6, n_beam_splitters)
ps_angles = np.linspace(np.pi / 3, np.pi / 5, n_beam_splitters)

counts = cudaq.sample(TBI,
                      bs_angles,
                      ps_angles,
                      input_state,
                      loop_lengths,
                      shots_count=1000000)
counts.dump()
