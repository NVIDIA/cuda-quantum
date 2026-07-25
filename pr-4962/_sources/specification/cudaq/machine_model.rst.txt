Machine Model
*************

.. role:: raw-html(raw)
    :format: html 

:raw-html:`<a name="mm1"></a>`

**[1]** CUDA-Q presumes the existence of one or more classical host processors, 
zero or more NVIDIA graphics processing units (GPUs), and zero or more quantum processing units (QPUs). 

**[2]** Each QPU is composed of a classical quantum control system (distributed FPGAs, GPUs, etc.) 
and a register of quantum bits (qubits) whose state evolves via signals from the classical control system.

**[3]** The machine model allows for three modes of quantum process parallelism: concurrent execution
of quantum circuits on independent QPUs, dependent quantum parallelism via
quantum message passing and inter-QPU entanglement, and QPU thread-level
parallelism or the ability for concurrent execution of independent quantum
circuits on a single QPU qubit connectivity fabric. 

**[4]** The model assumes a
classical memory space for the host processor which inherits the native language memory
model semantics (e.g. C++ or Python). 

**[5]** The model assumes a classical memory space for each control
system driving the evolution of the multi-qubit state. This control system
memory space should enable primitive arithmetic variable declarations,
stores, and loads, as well as qubit measurement persistence and loading
for fast-feedback and conditional circuit execution. 

**[6]** The quantum memory
space on an individual QPU is modeled as an infinite register of 
qubits and physical connectivity constraints are hidden from the
programmer. Note that compiler implementations of the CUDA-Q model
are free to enable developer access to the details of the QPU
qubit connectivity for the purpose of developing novel placement strategies. 

**[7]** CUDA-Q considers general :math:`D`-level quantum information systems, e.g. qudits. Qudits
are non-copyable, and can be allocated in chunks, via instantiation of 
user-level quantum container types. Quantum containers come in two flavors - 
quantum memory owning and non-owning (views). Moreover, the size of a quantum container 
can be specified at either compile-time or dynamically at runtime. 
As all qudits are non-copyable, qudits and containers 
thereof can only be passed by reference. 

**[8]** Each allocated qudit is unique and if  
deallocated, is available for subsequent allocation. Deallocation occurs implicitly
when the qudit goes out of scope. Uncomputation of qudit state should occur 
automatically via compiler implementations of the CUDA-Q model.

**[9]** The CUDA-Q model considers both remotely hosted QPU execution models as well as
tightly-coupled quantum-classical architectures. Remotely hosted models
support batch circuit execution where each circuit may contain simple
quantum-classical operation integration. Tightly-coupled execution models provide
streaming instruction execution, measure-reset, and fast-feedback of
qubit measurement results. This multi-modal execution model directly
influences the syntax and semantics of quantum kernel expressions
and their associated host-code context. 

