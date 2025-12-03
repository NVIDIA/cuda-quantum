import cudaq
import pytest

def test_trace_kernel():
    @cudaq.kernel
    def my_kernel():
        q = cudaq.qubit()
        x(q)
        h(q)
        mz(q)

    trace = cudaq.trace_kernel(my_kernel)
    
    assert trace.get_num_qudits() == 1
    
    instructions = list(trace)
    assert len(instructions) == 3
    
    assert instructions[0].name == "x"
    assert len(instructions[0].controls) == 0
    assert len(instructions[0].targets) == 1
    assert instructions[0].targets[0].id == 0

    assert instructions[1].name == "h"
    
    assert instructions[2].name == "mz"

def test_trace_kernel_with_args():
    @cudaq.kernel
    def my_kernel(angle: float):
        q = cudaq.qubit()
        rx(angle, q)

    trace = cudaq.trace_kernel(my_kernel, 0.5)
    
    instructions = list(trace)
    assert len(instructions) == 1
    assert instructions[0].name == "rx"
    assert len(instructions[0].params) == 1
    assert abs(instructions[0].params[0] - 0.5) < 1e-6

def test_trace_kernel_cnot():
    @cudaq.kernel
    def my_kernel():
        q = cudaq.qvector(2)
        x(q[0])
        cx(q[0], q[1])

    trace = cudaq.trace_kernel(my_kernel)
    
    assert trace.get_num_qudits() == 2
    
    instructions = list(trace)
    assert len(instructions) == 2
    
    assert instructions[0].name == "x"
    
    assert instructions[1].name == "x" # CX is often represented as X with control
    assert len(instructions[1].controls) == 1
    assert instructions[1].controls[0].id == 0
    assert len(instructions[1].targets) == 1
    assert instructions[1].targets[0].id == 1

def test_trace_kernel_loops():
    @cudaq.kernel
    def my_kernel():
        q = cudaq.qvector(2)
        for i in range(2):
            x(q[i])

    trace = cudaq.trace_kernel(my_kernel)
    assert trace.get_num_qudits() == 2

    instructions = list(trace)
    assert len(instructions) == 2
    assert instructions[0].name == "x"
    assert instructions[0].targets[0].id == 0
    assert instructions[1].name == "x"
    assert instructions[1].targets[0].id == 1

def test_trace_kernel_cx_loop():
    @cudaq.kernel
    def my_kernel():
        q = cudaq.qvector(3)
        # 0 -> 1, 1 -> 2
        for i in range(2):
            cx(q[i], q[i+1])

    trace = cudaq.trace_kernel(my_kernel)
    assert trace.get_num_qudits() == 3

    instructions = list(trace)
    assert len(instructions) == 2
    
    # cx(0, 1)
    assert instructions[0].name == "x"
    assert len(instructions[0].controls) == 1
    assert instructions[0].controls[0].id == 0
    assert len(instructions[0].targets) == 1
    assert instructions[0].targets[0].id == 1

    # cx(1, 2)
    assert instructions[1].name == "x"
    assert len(instructions[1].controls) == 1
    assert instructions[1].controls[0].id == 1
    assert len(instructions[1].targets) == 1
    assert instructions[1].targets[0].id == 2

def test_trace_kernel_subkernels():
    @cudaq.kernel
    def subkernel(q: cudaq.qubit):
        x(q)
    
    @cudaq.kernel
    def my_kernel():
        q = cudaq.qubit()
        h(q)
        subkernel(q)

    trace = cudaq.trace_kernel(my_kernel)
    assert trace.get_num_qudits() == 1

    instructions = list(trace)
    assert len(instructions) == 2
    assert instructions[0].name == "h"
    assert instructions[1].name == "x"

def test_trace_kernel_subkernels_with_controls():
    @cudaq.kernel
    def subkernel(q: cudaq.qubit):
        x(q)
    
    @cudaq.kernel
    def my_kernel():
        q = cudaq.qvector(2)
        # Apply subkernel controlled by q[0] on q[1]
        cudaq.control(subkernel, q[0], q[1])

    trace = cudaq.trace_kernel(my_kernel)
    assert trace.get_num_qudits() == 2

    instructions = list(trace)
    assert len(instructions) == 1
    assert instructions[0].name == "x"
    assert len(instructions[0].controls) == 1
    assert instructions[0].controls[0].id == 0
    assert len(instructions[0].targets) == 1
    assert instructions[0].targets[0].id == 1
