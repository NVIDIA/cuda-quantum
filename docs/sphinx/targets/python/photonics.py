import cudaq

cudaq.set_target("photonics")


@cudaq.kernel
def photonicsKernel():
    qutrits = [qudit(3) for _ in range(2)]
    create(qutrits[0])
    create(qutrits[1])
    create(qutrits[1])
    mz(qutrits)


counts = cudaq.sample(photonicsKernel)
print(counts)

state = cudaq.get_state(photonicsKernel)
print(state)
