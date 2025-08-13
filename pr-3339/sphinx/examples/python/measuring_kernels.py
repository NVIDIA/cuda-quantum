# [Begin Docs]
import cudaq
# [End Docs]


# [Begin Sample1]
@cudaq.kernel
def kernel():
    qubits = cudaq.qvector(2)
    mz(qubits)


# [End Sample1]


# [Begin Sample2]
@cudaq.kernel
def kernel():
    qubits_a = cudaq.qvector(2)
    qubit_b = cudaq.qubit()
    mz(qubits_a)
    mx(qubit_b)


# [End Sample2]


# [Begin Sample3]
@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)

    h(q[0])
    b0 = mz(q[0])
    reset(q[0])
    x(q[0])

    if b0:
        h(q[1])


print(cudaq.sample(kernel))
# [End Sample3]
''' [Begin Sample4]
{ 
  __global__ : { 10:728 11:272 }
   b0 : { 0:505 1:495 }
}
 [End Sample4] '''
