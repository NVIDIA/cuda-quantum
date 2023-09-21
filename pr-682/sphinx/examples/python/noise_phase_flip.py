import cudaq

# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')

# CUDA Quantum supports several different models of noise. In this
# case, we will examine the modeling of decoherence of the qubit phase.
# This will occur from "phase flip" errors, wherein the qubit has a
# user-specified probability of undergoing a Z-180 rotation.

# We will begin by defining an empty noise model that we will add
# our phase flip channel to.
noise = cudaq.NoiseModel()

# Phase flip channel with `1.0` probability of the qubit
# undergoing a phase rotation of 180 degrees (π).
phase_flip = cudaq.PhaseFlipChannel(1.0)
# We will apply this channel to any Z gate on the qubit.
# Meaning, after each Z gate on qubit 0, there will be a
# probability of `1.0` that the qubit undergoes an extra
# Z rotation.
noise.add_channel('z', [0], phase_flip)

kernel = cudaq.make_kernel()
# Single qubit initialized to the |0> state.
qubit = kernel.qalloc()

# Place qubit in superposition state.
kernel.h(qubit)

# Rotate the phase around Z by 180 degrees (π).
kernel.z(qubit)

# Apply another Hadamard and measure.
kernel.h(qubit)
kernel.mz(qubit)

# Without noise, we'd expect the qubit to end in the |1>
# state due to the phase rotation between the two Hadamard
# gates.
counts = cudaq.sample(kernel)
counts.dump()

# With noise, our Z-gate will effectively cancel out due
# to the presence of a phase flip error on the gate with a
# probability of `1.0`. This will put us back in the |0> state.
noisy_counts = cudaq.sample(kernel, noise_model=noise)
noisy_counts.dump()
