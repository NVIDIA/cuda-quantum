import cudaq
import numpy as np

def main():
    # [Begin Custom Op]
    cudaq.register_operation("custom_h", 1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    cudaq.sample(bell).dump()
    # [End Custom Op]

if __name__ == "__main__":
    main()