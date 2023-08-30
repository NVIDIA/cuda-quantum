Control Flow
************
All classical control flow expressions are inherited from the host language
(C++). Programmers can define quantum kernels using standard loop
constructs and conditional statements. Conditional statements on quantum
memory and measured :code:`cudaq::qubit` results are also permitted. Programmers
can define a conditional block given a single unmeasured qubit and compiler
implementations should convert this into a coherent conditional statement
(e.g. :code:`if (q) { x(r); }` should convert to :code:`cnot(q,r)` internally).
Conditional statements on measured qubits results are also permitted to
enable dynamic circuits and fast-feedback
(:code:`auto c = mz(q); if (c) { conditional code...}`).
