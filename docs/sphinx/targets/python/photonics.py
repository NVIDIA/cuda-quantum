import cudaq

cudaq.set_target("photonics")


@cudaq.kernel
def photonicsKernel():
    qutrits = [qudit(3) for _ in range(2)]
    plus(qutrits[0])
    plus(qutrits[1])
    plus(qutrits[1])
    mz(qutrits)


counts = cudaq.sample(photonicsKernel)
print(counts)

state = cudaq.get_state(photonicsKernel)
print(state)
