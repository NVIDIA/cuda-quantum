import cudaq
from cudaq import spin, x, y, z, ry


@cudaq.kernel()
def ansatz_h2(theta: float):
   q = cudaq.qvector(2)
   x(q[0])
   ry(theta, q[1])
   x.ctrl(q[1], q[0])

h_operator = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(0) * spin.y(1) + \
               .21829 * spin.z(0) - 6.125 * spin.z(1)
energy = cudaq.observe(ansatz_h2, h_operator, 0.59).expectation()
print(f"Energy is {energy}")


if __name__ == "__main__":
    pass