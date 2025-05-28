import cudaq
from cudaq import spin, x, y, z, ry, cx

# [Begin Kernel Python]
# Define a simple kernel for demonstration
@cudaq.kernel
def observe_kernel_demo(theta: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(theta, q[1])
    cx(q[0], q[1])
# [End Kernel Python]

# [Begin Observe Result Usage Python]
# Define a spin_op for demonstration
spinOp = spin.x(0) * spin.x(1) + 0.5 * spin.z(0)
param = 0.23 # Example parameter

expVal_simple = cudaq.observe(observe_kernel_demo, spinOp, param).expectation()
print(f"Simple ExpVal: {expVal_simple}")

result = cudaq.observe(observe_kernel_demo, spinOp, param)
expVal_detailed = result.expectation()
print(f"Detailed ExpVal: {expVal_detailed}")

x0x1_term = spin.x(0) * spin.x(1)
try:
    X0X1Exp = result.expectation(x0x1_term)
    print(f"X0X1 ExpVal: {X0X1Exp}")
    X0X1Data = result.counts(x0x1_term)
    print("X0X1 Counts:")
    X0X1Data.dump()
except Exception as e:
    print(f"Could not get data for X0X1 term directly: {e}")

result.dump()
# [End Observe Result Usage Python]

if __name__ == "__main__":
    pass