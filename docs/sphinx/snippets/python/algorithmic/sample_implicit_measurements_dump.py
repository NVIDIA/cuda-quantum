import cudaq

# [Begin Kernel Python]
@cudaq.kernel
def kernel():
    a = cudaq.qubit()
    b = cudaq.qubit()
    x(a)
# [End Kernel Python]
# [Begin Sample Python]
cudaq.sample(kernel).dump()
# [End Sample Python]

if __name__ == "__main__":
    pass