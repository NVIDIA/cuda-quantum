import cudaq
from cudaq import x, mz

# [Begin Kernel Python]
@cudaq.kernel
def kernel():
    a = cudaq.qubit()
    b = cudaq.qubit()
    x(a)
    mz(b)
    mz(a)
# [End Kernel Python]

# [Begin Sample Python]
print("Default - no explicit measurements")
cudaq.sample(kernel).dump()

print("\nSetting `explicit_measurements` option")
cudaq.sample(kernel, explicit_measurements=True).dump()
# [End Sample Python]

if __name__ == "__main__":
    pass