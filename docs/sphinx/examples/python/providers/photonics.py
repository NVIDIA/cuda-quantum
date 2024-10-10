import cudaq

cudaq.set_target("photonics-cpu")


@cudaq.kernel
def photonicsKernel():
    qumodes = [qudit(3) for _ in range(2)]
    create(qumodes[0])
    create(qumodes[1])
    create(qumodes[1])
    mz(qumodes)


counts = cudaq.sample(photonicsKernel)
print(counts)

state = cudaq.get_state(photonicsKernel)
print(state)
