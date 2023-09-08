Machine Model
*************

.. role:: raw-html(raw)
    :format: html 

:raw-html:`<a name="mm1"></a>`

CUDA Quantum presumes the existence of one or more classical host processors, zero
or more NVIDIA graphics processing units (GPUs), and zero or more quantum
processing units (QPUs). Each QPU is composed of a classical quantum control
system (distributed FPGAs, GPUs, etc.) and a register of quantum bits
(qubits) whose state evolves via signals from the classical control system.
This model enables potential quantum process parallelism (concurrent execution
of quantum circuits on independent QPUs), dependent quantum parallelism via
quantum message passing and inter-QPU entanglement, and QPU thread-level
parallelism (the ability for concurrent execution of independent quantum
circuits on a single QPU qubit connectivity fabric). The model assumes a
classical memory space for the host processor which inherits the C++ memory
model semantics. The model assumes a classical memory space for each control
system driving the evolution of the multi-qubit state. This control system
memory space should enable primitive arithmetic variable declarations,
stores, and loads, as well as qubit measurement persistence and loading
for fast-feedback and conditional circuit execution. The quantum memory
space on an individual QPU is modeled as an infinite register of logical
qubits and physical connectivity constraints are hidden from the
programmer. Note that compiler implementations of the CUDA Quantum model
are free to enable developer access to the details of the backend's
qubit connectivity for the purpose of developing novel placement strategies. 

CUDA Quantum considers general D-level quantum information systems, qudits. Qudits
are non-copyable, and can be allocated in chunks, called quantum containers.
Quantum containers come in two flavors - memory owning and non-owning (views). 
Moreover, the size of a quantum container can be specified at either compile-time
or dynamically at runtime. As all qudits are non-copyable, qudits and containers 
thereof can only be passed by reference. Each qudit is assigned a unique 
identifying index, and upon deallocation that index is returned and can be 
reused by subsequent allocations. Deallocation occurs implicitly when the qudit goes 
out of scope. Uncomputation of qudit state should occur automatically via 
compiler implementations of the CUDA Quantum model, but programmers should be able 
to specify manual uncomputation at qudit instantiation. 

The CUDA Quantum model considers both remotely hosted QPU execution models as well as
tightly-coupled quantum-classical architectures. Remotely hosted models
support batch circuit execution where each circuit may contain simple
classical-quantum integration (primitive measure-reset and simple
conditional statements). Tightly-coupled execution models provide
streaming instruction execution, measure-reset, and fast-feedback of
qubit measurement results. This multi-modal execution model directly
influences the syntax and semantics of quantum kernel expressions
and their associated host-code context. 