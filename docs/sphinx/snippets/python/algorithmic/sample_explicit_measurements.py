import cudaq

# [Begin Kernel Python]
@cudaq.kernel()
def kernel():
   q = cudaq.qubit()
   h(q)
   reg1 = mz(q)
   reset(q)
   x(q)
# [End Kernel Python]

# [Begin Sample Python]
print("Default - no explicit measurements")
cudaq.sample(kernel).dump()

print("\nSetting `explicit_measurements` option")
cudaq.sample(kernel, explicit_measurements=True).dump()
# [End Sample Python]

if __name__ == "__main__":
    pass