::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../index.html){.icon .icon-home}

::: version
pr-5659
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](quick_start.html){.reference .internal}
    -   [Install CUDA-Q](quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q Academic](quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q Kernel?](basics/kernel_intro.html){.reference
        .internal}
    -   [Building your first CUDA-Q
        Program](basics/build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q
        Program](basics/run_kernel.html){.reference .internal}
        -   [Sample](basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](basics/run_kernel.html#run){.reference .internal}
        -   [Observe](basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
        -   [Python
            Stack-Traces](basics/troubleshooting.html#python-stack-traces){.reference
            .internal}
        -   [LLVM-based Python
            packages](basics/troubleshooting.html#llvm-based-python-packages){.reference
            .internal}
-   [Examples](examples/examples.html){.reference .internal}
    -   [Introduction](examples/introduction.html){.reference .internal}
    -   [Building Kernels](examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring Kernels](examples/measuring_kernels.html){.reference
        .internal}
        -   [Measurement
            Handles](examples/measuring_kernels.html#measurement-handles){.reference
            .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](../examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](../examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](../examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing Kernels](examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](examples/expectation_values.html){.reference .internal}
        -   [Parallelizing across Multiple
            Processors](examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
    -   [Optimizers &
        Gradients](../examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [CUDA-Q Optimizer
            Overview](../examples/python/optimizers_gradients.html#CUDA-Q-Optimizer-Overview){.reference
            .internal}
            -   [Gradient-Free Optimizers (no gradients
                required):](../examples/python/optimizers_gradients.html#Gradient-Free-Optimizers-(no-gradients-required):){.reference
                .internal}
            -   [Gradient-Based Optimizers (require
                gradients):](../examples/python/optimizers_gradients.html#Gradient-Based-Optimizers-(require-gradients):){.reference
                .internal}
        -   [1. Built-in CUDA-Q Optimizers and
            Gradients](../examples/python/optimizers_gradients.html#1.-Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
            -   [1.1 Adam Optimizer with Parameter
                Configuration](../examples/python/optimizers_gradients.html#1.1-Adam-Optimizer-with-Parameter-Configuration){.reference
                .internal}
            -   [1.2 SGD (Stochastic Gradient Descent)
                Optimizer](../examples/python/optimizers_gradients.html#1.2-SGD-(Stochastic-Gradient-Descent)-Optimizer){.reference
                .internal}
            -   [1.3 SPSA (Simultaneous Perturbation Stochastic
                Approximation)](../examples/python/optimizers_gradients.html#1.3-SPSA-(Simultaneous-Perturbation-Stochastic-Approximation)){.reference
                .internal}
        -   [2. Third-Party
            Optimizers](../examples/python/optimizers_gradients.html#2.-Third-Party-Optimizers){.reference
            .internal}
        -   [3. Parallel Parameter Shift
            Gradients](../examples/python/optimizers_gradients.html#3.-Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](../examples/python/noisy_simulations.html){.reference
        .internal}
    -   [Pre-Trajectory Sampling with Batch
        Execution](examples/ptsbe.html){.reference .internal}
        -   [Conceptual
            Overview](examples/ptsbe.html#conceptual-overview){.reference
            .internal}
        -   [When to Use
            PTSBE](examples/ptsbe.html#when-to-use-ptsbe){.reference
            .internal}
        -   [Quick Start](examples/ptsbe.html#quick-start){.reference
            .internal}
        -   [Usage
            Tutorial](examples/ptsbe.html#usage-tutorial){.reference
            .internal}
            -   [Controlling the Number of
                Trajectories](examples/ptsbe.html#controlling-the-number-of-trajectories){.reference
                .internal}
            -   [Choosing a Trajectory Sampling
                Strategy](examples/ptsbe.html#choosing-a-trajectory-sampling-strategy){.reference
                .internal}
            -   [Shot Allocation
                Strategies](examples/ptsbe.html#shot-allocation-strategies){.reference
                .internal}
            -   [Inspecting Execution
                Data](examples/ptsbe.html#inspecting-execution-data){.reference
                .internal}
    -   [Detector Error
        Models](examples/dem_from_kernel.html){.reference .internal}
        -   [DEM
            Options](examples/dem_from_kernel.html#dem-options){.reference
            .internal}
        -   [Measurement
            Matrices](examples/dem_from_kernel.html#measurement-matrices){.reference
            .internal}
        -   [Limitations](examples/dem_from_kernel.html#limitations){.reference
            .internal}
    -   [Rotation Synthesis
        (Clifford+T)](examples/rotation_synthesis.html){.reference
        .internal}
        -   [Synthesizing a
            rotation](examples/rotation_synthesis.html#synthesizing-a-rotation){.reference
            .internal}
        -   [Estimating the T count of a
            kernel](examples/rotation_synthesis.html#estimating-the-t-count-of-a-kernel){.reference
            .internal}
        -   [Choosing
            epsilon](examples/rotation_synthesis.html#choosing-epsilon){.reference
            .internal}
        -   [Dependencies](examples/rotation_synthesis.html#dependencies){.reference
            .internal}
    -   [Constructing Operators](examples/operators.html){.reference
        .internal}
        -   [Constructing Spin
            Operators](examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [qBraid](examples/hardware_providers.html#qbraid){.reference
            .internal}
        -   [Quantinuum](examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum
            Machines](examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](examples/hardware_providers.html#quera-computing){.reference
            .internal}
        -   [Scaleway](examples/hardware_providers.html#scaleway){.reference
            .internal}
        -   [TII](examples/hardware_providers.html#tii){.reference
            .internal}
    -   [When to Use sample vs.
        run](examples/sample_vs_run.html){.reference .internal}
        -   [Introduction](examples/sample_vs_run.html#introduction){.reference
            .internal}
        -   [Usage
            Guidelines](examples/sample_vs_run.html#usage-guidelines){.reference
            .internal}
        -   [What Is Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](examples/sample_vs_run.html#what-is-supported-with-sample){.reference
            .internal}
        -   [What Is Not Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](examples/sample_vs_run.html#what-is-not-supported-with-sample){.reference
            .internal}
        -   [How to
            Migrate](examples/sample_vs_run.html#how-to-migrate){.reference
            .internal}
            -   [Step 1: Add a return type to the
                kernel](examples/sample_vs_run.html#step-1-add-a-return-type-to-the-kernel){.reference
                .internal}
            -   [Step 2: Replace [`sample`{.docutils .literal
                .notranslate}]{.pre} with [`run`{.docutils .literal
                .notranslate}]{.pre}](examples/sample_vs_run.html#step-2-replace-sample-with-run){.reference
                .internal}
            -   [Step 3: Update result
                processing](examples/sample_vs_run.html#step-3-update-result-processing){.reference
                .internal}
        -   [Migration
            Examples](examples/sample_vs_run.html#migration-examples){.reference
            .internal}
            -   [Example 1: Simple conditional
                logic](examples/sample_vs_run.html#example-1-simple-conditional-logic){.reference
                .internal}
            -   [Example 2: Returning multiple measurement
                results](examples/sample_vs_run.html#example-2-returning-multiple-measurement-results){.reference
                .internal}
            -   [Example 3: Quantum
                teleportation](examples/sample_vs_run.html#example-3-quantum-teleportation){.reference
                .internal}
        -   [Additional
            Notes](examples/sample_vs_run.html#additional-notes){.reference
            .internal}
    -   [Dynamics Examples](examples/dynamics_examples.html){.reference
        .internal}
        -   [Python Examples (Jupyter
            Notebooks)](examples/dynamics_examples.html#python-examples-jupyter-notebooks){.reference
            .internal}
            -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
                Model)](../examples/python/dynamics/dynamics_intro_1.html){.reference
                .internal}
            -   [Introduction to CUDA-Q Dynamics (Time Dependent
                Hamiltonians)](../examples/python/dynamics/dynamics_intro_2.html){.reference
                .internal}
            -   [Superconducting
                Qubits](../examples/python/dynamics/superconducting.html){.reference
                .internal}
            -   [Spin
                Qubits](../examples/python/dynamics/spinqubits.html){.reference
                .internal}
            -   [Trapped Ion
                Qubits](../examples/python/dynamics/iontrap.html){.reference
                .internal}
            -   [Control](../examples/python/dynamics/control.html){.reference
                .internal}
        -   [C++
            Examples](examples/dynamics_examples.html#c-examples){.reference
            .internal}
            -   [Introduction: Single Qubit
                Dynamics](examples/dynamics_examples.html#introduction-single-qubit-dynamics){.reference
                .internal}
            -   [Introduction: Cavity QED (Jaynes-Cummings
                Model)](examples/dynamics_examples.html#introduction-cavity-qed-jaynes-cummings-model){.reference
                .internal}
            -   [Superconducting Qubits: Cross-Resonance
                Gate](examples/dynamics_examples.html#superconducting-qubits-cross-resonance-gate){.reference
                .internal}
            -   [Spin Qubits: Heisenberg Spin
                Chain](examples/dynamics_examples.html#spin-qubits-heisenberg-spin-chain){.reference
                .internal}
            -   [Control: Driven
                Qubit](examples/dynamics_examples.html#control-driven-qubit){.reference
                .internal}
            -   [State
                Batching](examples/dynamics_examples.html#state-batching){.reference
                .internal}
            -   [Numerical
                Integrators](examples/dynamics_examples.html#numerical-integrators){.reference
                .internal}
-   [Applications](applications.html){.reference .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](../applications/python/krylov.html){.reference
        .internal}
        -   [Setup](../applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](../applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](../applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Quantum-Selected Configuration Interaction
        (QSCI)](../applications/python/qsci.html){.reference .internal}
        -   [0. Problem
            definition](../applications/python/qsci.html#0.-Problem-definition){.reference
            .internal}
        -   [1. Prepare an Approximate Quantum
            State](../applications/python/qsci.html#1.-Prepare-an-Approximate-Quantum-State){.reference
            .internal}
        -   [2 Quantum Sampling to Select
            Configuration](../applications/python/qsci.html#2-Quantum-Sampling-to-Select-Configuration){.reference
            .internal}
        -   [3. Classical Diagonalization on the Selected
            Subspace](../applications/python/qsci.html#3.-Classical-Diagonalization-on-the-Selected-Subspace){.reference
            .internal}
        -   [5. Compare
            results](../applications/python/qsci.html#5.-Compare-results){.reference
            .internal}
        -   [Reference](../applications/python/qsci.html#Reference){.reference
            .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](../applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](../applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using [`Sample`{.docutils .literal .notranslate}]{.pre} to
            perform the Hadamard
            test](../applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](../applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](../applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](../applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](../applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](../applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](../applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](../applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](../applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](../applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](../applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. [`Compute`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`overlap`{.docutils .literal
                .notranslate}]{.pre}](../applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](../applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](../applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](../applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](../applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](../applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](../applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](../applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](../applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](../applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum
        Volume](../applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](../applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](../applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](../applications/python/afqmc.html){.reference .internal}
        -   [Hamiltonian preparation for
            VQE](../applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](../applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](../applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](../applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](../applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](../applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [Factoring Integers With Shor's
        Algorithm](../applications/python/shors.html){.reference
        .internal}
        -   [Shor's
            algorithm](../applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](../applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](../applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](../applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](../applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](../applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](../applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](../applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
            -   [(a) Generate the molecular Hamiltonian using Restricted
                Hartree Fock molecular
                orbitals](../applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Restricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(b) Generate the molecular Hamiltonian using
                Unrestricted Hartree Fock molecular
                orbitals](../applications/python/generate_fermionic_ham.html#(b)-Generate-the-molecular-Hamiltonian-using-Unrestricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(a) Generate the active space hamiltonian using RHF
                molecular
                orbitals.](../applications/python/generate_fermionic_ham.html#(a)-Generate-the-active-space-hamiltonian-using-RHF-molecular-orbitals.){.reference
                .internal}
            -   [(b) Generate the active space Hamiltonian using the
                natural orbitals computed from MP2
                simulation](../applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
                .internal}
            -   [(c) Generate the active space Hamiltonian computed from
                the CASSCF molecular
                orbitals](../applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
                .internal}
            -   [(d) Generate the electronic Hamiltonian using
                ROHF](../applications/python/generate_fermionic_ham.html#(d)-Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
            -   [(e) Generate electronic Hamiltonian using
                UHF](../applications/python/generate_fermionic_ham.html#(e)-Generate-electronic-Hamiltonian-using-UHF){.reference
                .internal}
    -   [The UCCSD Wavefunction
        ansatz](../applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](../applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](../applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](../applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](../applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](../applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](../applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [Sample-Based Krylov Quantum Diagonalization
        (SKQD)](../applications/python/skqd.html){.reference .internal}
        -   [Why
            SKQD?](../applications/python/skqd.html#Why-SKQD?){.reference
            .internal}
        -   [Understanding Krylov
            Subspaces](../applications/python/skqd.html#Understanding-Krylov-Subspaces){.reference
            .internal}
            -   [What is a Krylov
                Subspace?](../applications/python/skqd.html#What-is-a-Krylov-Subspace?){.reference
                .internal}
            -   [The SKQD
                Algorithm](../applications/python/skqd.html#The-SKQD-Algorithm){.reference
                .internal}
        -   [Problem Setup: 22-Qubit Heisenberg
            Model](../applications/python/skqd.html#Problem-Setup:-22-Qubit-Heisenberg-Model){.reference
            .internal}
        -   [Krylov State Generation via Repeated
            Evolution](../applications/python/skqd.html#Krylov-State-Generation-via-Repeated-Evolution){.reference
            .internal}
        -   [Quantum Measurements and
            Sampling](../applications/python/skqd.html#Quantum-Measurements-and-Sampling){.reference
            .internal}
            -   [The Sampling
                Process](../applications/python/skqd.html#The-Sampling-Process){.reference
                .internal}
        -   [Classical Post-Processing and
            Diagonalization](../applications/python/skqd.html#Classical-Post-Processing-and-Diagonalization){.reference
            .internal}
            -   [Matrix Construction
                Details](../applications/python/skqd.html#Matrix-Construction-Details){.reference
                .internal}
            -   [Approach 1: GPU-Vectorized CSR Sparse
                Matrix](../applications/python/skqd.html#Approach-1:-GPU-Vectorized-CSR-Sparse-Matrix){.reference
                .internal}
            -   [Approach 2: Matrix-Free Lanczos via
                [`distributed_eigsh`{.docutils .literal
                .notranslate}]{.pre}](../applications/python/skqd.html#Approach-2:-Matrix-Free-Lanczos-via-distributed_eigsh){.reference
                .internal}
        -   [Results Analysis and
            Convergence](../applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](../applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [Postprocessing Acceleration: CSR matrix approach, single
            GPU vs
            CPU](../applications/python/skqd.html#Postprocessing-Acceleration:-CSR-matrix-approach,-single-GPU-vs-CPU){.reference
            .internal}
        -   [Postprocessing Scale-Up and Scale-Out: Linear Operator
            Approach, Multi-GPU
            Multi-Node](../applications/python/skqd.html#Postprocessing-Scale-Up-and-Scale-Out:-Linear-Operator-Approach,-Multi-GPU-Multi-Node){.reference
            .internal}
            -   [Saving Hamiltonian
                Data](../applications/python/skqd.html#Saving-Hamiltonian-Data){.reference
                .internal}
            -   [Running the Distributed
                Solver](../applications/python/skqd.html#Running-the-Distributed-Solver){.reference
                .internal}
        -   [Summary](../applications/python/skqd.html#Summary){.reference
            .internal}
    -   [Entanglement Accelerates Quantum
        Simulation](../applications/python/entanglement_acc_hamiltonian_simulation.html){.reference
        .internal}
        -   [2. Model
            Definition](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.-Model-Definition){.reference
            .internal}
            -   [2.1 Initial product
                state](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.1-Initial-product-state){.reference
                .internal}
            -   [2.2 QIMF
                Hamiltonian](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.2-QIMF-Hamiltonian){.reference
                .internal}
            -   [2.3 First-Order Trotter Formula
                (PF1)](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.3-First-Order-Trotter-Formula-(PF1)){.reference
                .internal}
            -   [2.4 PF1 step for the QIMF
                partition](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.4-PF1-step-for-the-QIMF-partition){.reference
                .internal}
            -   [2.5 Hamiltonian
                helpers](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.5-Hamiltonian-helpers){.reference
                .internal}
        -   [3. Entanglement
            metrics](../applications/python/entanglement_acc_hamiltonian_simulation.html#3.-Entanglement-metrics){.reference
            .internal}
        -   [4. Simulation
            workflow](../applications/python/entanglement_acc_hamiltonian_simulation.html#4.-Simulation-workflow){.reference
            .internal}
            -   [4.1 Single-step Trotter
                error](../applications/python/entanglement_acc_hamiltonian_simulation.html#4.1-Single-step-Trotter-error){.reference
                .internal}
            -   [4.2 Dual trajectory
                update](../applications/python/entanglement_acc_hamiltonian_simulation.html#4.2-Dual-trajectory-update){.reference
                .internal}
        -   [5. Reproducing the paper's Figure
            1a](../applications/python/entanglement_acc_hamiltonian_simulation.html#5.-Reproducing-the-paper’s-Figure-1a){.reference
            .internal}
            -   [5.1 Visualising the joint
                behaviour](../applications/python/entanglement_acc_hamiltonian_simulation.html#5.1-Visualising-the-joint-behaviour){.reference
                .internal}
            -   [5.2 Interpreting the
                result](../applications/python/entanglement_acc_hamiltonian_simulation.html#5.2-Interpreting-the-result){.reference
                .internal}
        -   [6. References and further
            reading](../applications/python/entanglement_acc_hamiltonian_simulation.html#6.-References-and-further-reading){.reference
            .internal}
    -   [Pre-Trajectory Sampling with Batch Execution
        (PTSBE)](../applications/python/ptsbe.html){.reference
        .internal}
        -   [Set up the
            environment](../applications/python/ptsbe.html#Set-up-the-environment){.reference
            .internal}
        -   [Define the circuit and noise
            model](../applications/python/ptsbe.html#Define-the-circuit-and-noise-model){.reference
            .internal}
            -   [Inline noise with [`apply_noise`{.docutils .literal
                .notranslate}]{.pre}](../applications/python/ptsbe.html#Inline-noise-with-apply_noise){.reference
                .internal}
        -   [Run PTSBE
            sampling](../applications/python/ptsbe.html#Run-PTSBE-sampling){.reference
            .internal}
            -   [Larger circuit for execution
                data](../applications/python/ptsbe.html#Larger-circuit-for-execution-data){.reference
                .internal}
        -   [Inspecting trajectories with execution
            data](../applications/python/ptsbe.html#Inspecting-trajectories-with-execution-data){.reference
            .internal}
        -   [Performance of PTSBE vs standard noisy
            sampling](../applications/python/ptsbe.html#Performance-of-PTSBE-vs-standard-noisy-sampling){.reference
            .internal}
-   [Backends](backends/backends.html){.reference .internal}
    -   [Circuit Simulation](backends/simulators.html){.reference
        .internal}
        -   [State Vector
            Simulators](backends/sims/svsims.html){.reference .internal}
            -   [CPU](backends/sims/svsims.html#cpu){.reference
                .internal}
            -   [Single-GPU](backends/sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](backends/sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network
            Simulators](backends/sims/tnsims.html){.reference .internal}
            -   [Multi-GPU
                multi-node](backends/sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](backends/sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](backends/sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU
            Simulators](backends/sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](backends/sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU with Multi-Node Multi-GPU
                Backends](backends/sims/mqpusims.html#multi-qpu-with-multi-node-multi-gpu-backends){.reference
                .internal}
        -   [Noisy Simulators](backends/sims/noisy.html){.reference
            .internal}
            -   [Trajectory Noisy
                Simulation](backends/sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](backends/sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](backends/sims/noisy.html#stim){.reference
                .internal}
            -   [[`cuStabilizer`{.docutils .literal
                .notranslate}]{.pre}](backends/sims/noisy.html#custabilizer){.reference
                .internal}
        -   [Photonics
            Simulators](backends/sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](backends/sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware (QPUs)](backends/hardware.html){.reference
        .internal}
        -   [Ion Trap QPUs](backends/hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](backends/hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](backends/hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](backends/hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](backends/hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](backends/hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](backends/hardware/superconducting.html#oqc){.reference
                .internal}
            -   [TII](backends/hardware/superconducting.html#tii){.reference
                .internal}
        -   [Neutral Atom
            QPUs](backends/hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](backends/hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](backends/hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](backends/hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic QPUs](backends/hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](backends/hardware/photonic.html#orca-computing){.reference
                .internal}
        -   [Quantum Control
            Systems](backends/hardware/qcontrol.html){.reference
            .internal}
            -   [Quantum
                Machines](backends/hardware/qcontrol.html#quantum-machines){.reference
                .internal}
    -   [Dynamics
        Simulation](backends/dynamics_backends.html){.reference
        .internal}
    -   [Cloud](backends/cloud.html){.reference .internal}
        -   [Amazon Braket
            (braket)](backends/cloud/braket.html){.reference .internal}
            -   [Setting
                Credentials](backends/cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submitting](backends/cloud/braket.html#submitting){.reference
                .internal}
        -   [Scaleway QaaS
            (scaleway)](backends/cloud/scaleway.html){.reference
            .internal}
            -   [Setting
                Credentials](backends/cloud/scaleway.html#setting-credentials){.reference
                .internal}
            -   [Submitting](backends/cloud/scaleway.html#submitting){.reference
                .internal}
            -   [Manage your QPU
                session](backends/cloud/scaleway.html#manage-your-qpu-session){.reference
                .internal}
        -   [qBraid](backends/cloud/qbraid.html){.reference .internal}
            -   [Setting
                Credentials](backends/cloud/qbraid.html#setting-credentials){.reference
                .internal}
            -   [Submitting](backends/cloud/qbraid.html#submitting){.reference
                .internal}
-   [Dynamics](dynamics.html){.reference .internal}
    -   [Quick Start](dynamics.html#quick-start){.reference .internal}
    -   [Operator](dynamics.html#operator){.reference .internal}
    -   [Time-Dependent
        Dynamics](dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch simulation](dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](dynamics.html#examples){.reference .internal}
-   [Realtime](realtime.html){.reference .internal}
    -   [Installation](realtime/installation.html){.reference .internal}
        -   [Prerequisites](realtime/installation.html#prerequisites){.reference
            .internal}
        -   [HSB FPGA IP core and RFSoC
            bit-file](realtime/installation.html#hsb-fpga-ip-core-and-rfsoc-bit-file){.reference
            .internal}
        -   [Setup](realtime/installation.html#setup){.reference
            .internal}
        -   [Latency
            Measurement](realtime/installation.html#latency-measurement){.reference
            .internal}
    -   [Host API](realtime/host.html){.reference .internal}
        -   [What is the
            GpuRoceTransceiver?](realtime/host.html#what-is-the-gpurocetransceiver){.reference
            .internal}
        -   [Transport
            Mechanisms](realtime/host.html#transport-mechanisms){.reference
            .internal}
            -   [Supported Transport
                Options](realtime/host.html#supported-transport-options){.reference
                .internal}
        -   [The 3-Kernel Architecture (GpuRoceTransceiver Example)
            {#three-kernel-architecture}](realtime/host.html#the-3-kernel-architecture-gpurocetransceiver-example-three-kernel-architecture){.reference
            .internal}
            -   [Data Flow
                Summary](realtime/host.html#data-flow-summary){.reference
                .internal}
            -   [Why 3
                Kernels?](realtime/host.html#why-3-kernels){.reference
                .internal}
        -   [Unified Dispatch
            Mode](realtime/host.html#unified-dispatch-mode){.reference
            .internal}
            -   [Architecture](realtime/host.html#architecture){.reference
                .internal}
            -   [Transport-Agnostic
                Design](realtime/host.html#transport-agnostic-design){.reference
                .internal}
            -   [When to Use Which
                Mode](realtime/host.html#when-to-use-which-mode){.reference
                .internal}
            -   [Host API
                Extensions](realtime/host.html#host-api-extensions){.reference
                .internal}
            -   [Wiring Example (Unified Mode with
                GpuRoceTransceiver)](realtime/host.html#wiring-example-unified-mode-with-gpurocetransceiver){.reference
                .internal}
        -   [What This API Does (In One
            Paragraph)](realtime/host.html#what-this-api-does-in-one-paragraph){.reference
            .internal}
        -   [Scope](realtime/host.html#scope){.reference .internal}
        -   [Terms and
            Components](realtime/host.html#terms-and-components){.reference
            .internal}
        -   [Schema Data
            Structures](realtime/host.html#schema-data-structures){.reference
            .internal}
            -   [Type
                Descriptors](realtime/host.html#type-descriptors){.reference
                .internal}
            -   [Handler
                Schema](realtime/host.html#handler-schema){.reference
                .internal}
        -   [RPC Messaging
            Protocol](realtime/host.html#rpc-messaging-protocol){.reference
            .internal}
        -   [Host API
            Overview](realtime/host.html#host-api-overview){.reference
            .internal}
        -   [Manager and Dispatcher
            Topology](realtime/host.html#manager-and-dispatcher-topology){.reference
            .internal}
        -   [Host API
            Functions](realtime/host.html#host-api-functions){.reference
            .internal}
            -   [Occupancy Query and Eager Module
                Loading](realtime/host.html#occupancy-query-and-eager-module-loading){.reference
                .internal}
            -   [Graph-Based Dispatch
                Functions](realtime/host.html#graph-based-dispatch-functions){.reference
                .internal}
            -   [Kernel Launch Helper
                Functions](realtime/host.html#kernel-launch-helper-functions){.reference
                .internal}
        -   [Memory Layout and Ring Buffer
            Wiring](realtime/host.html#memory-layout-and-ring-buffer-wiring){.reference
            .internal}
        -   [Step-by-Step: Wiring the Host API
            (Minimal)](realtime/host.html#step-by-step-wiring-the-host-api-minimal){.reference
            .internal}
        -   [Device Handler and Function
            ID](realtime/host.html#device-handler-and-function-id){.reference
            .internal}
            -   [Multi-Argument Handler
                Example](realtime/host.html#multi-argument-handler-example){.reference
                .internal}
        -   [CUDA Graph Dispatch
            Mode](realtime/host.html#cuda-graph-dispatch-mode){.reference
            .internal}
            -   [Requirements](realtime/host.html#requirements){.reference
                .internal}
            -   [Graph-Based Dispatch
                API](realtime/host.html#graph-based-dispatch-api){.reference
                .internal}
            -   [Graph Handler Setup
                Example](realtime/host.html#graph-handler-setup-example){.reference
                .internal}
            -   [Graph Capture and
                Instantiation](realtime/host.html#graph-capture-and-instantiation){.reference
                .internal}
            -   [When to Use Graph
                Dispatch](realtime/host.html#when-to-use-graph-dispatch){.reference
                .internal}
            -   [Graph vs Device Call
                Dispatch](realtime/host.html#graph-vs-device-call-dispatch){.reference
                .internal}
        -   [Building and Sending an RPC
            Message](realtime/host.html#building-and-sending-an-rpc-message){.reference
            .internal}
        -   [Reading the
            Response](realtime/host.html#reading-the-response){.reference
            .internal}
        -   [Schema-Driven Argument
            Parsing](realtime/host.html#schema-driven-argument-parsing){.reference
            .internal}
        -   [GpuRoceTransceiver 3-Kernel Workflow
            (Primary)](realtime/host.html#gpurocetransceiver-3-kernel-workflow-primary){.reference
            .internal}
        -   [NIC-Free Testing (No GpuRoceTransceiver / No
            ConnectX-7)](realtime/host.html#nic-free-testing-no-gpurocetransceiver-no-connectx-7){.reference
            .internal}
        -   [Troubleshooting](realtime/host.html#troubleshooting){.reference
            .internal}
    -   [Messaging Protocol](realtime/protocol.html){.reference
        .internal}
        -   [Scope](realtime/protocol.html#scope){.reference .internal}
        -   [RPC Header /
            Response](realtime/protocol.html#rpc-header-response){.reference
            .internal}
        -   [Request ID
            Semantics](realtime/protocol.html#request-id-semantics){.reference
            .internal}
        -   [[`PTP`{.docutils .literal .notranslate}]{.pre} Timestamp
            Semantics](realtime/protocol.html#ptp-timestamp-semantics){.reference
            .internal}
        -   [Function ID
            Semantics](realtime/protocol.html#function-id-semantics){.reference
            .internal}
        -   [Schema and Payload
            Interpretation](realtime/protocol.html#schema-and-payload-interpretation){.reference
            .internal}
            -   [Type
                System](realtime/protocol.html#type-system){.reference
                .internal}
        -   [Payload
            Encoding](realtime/protocol.html#payload-encoding){.reference
            .internal}
            -   [Single-Argument
                Payloads](realtime/protocol.html#single-argument-payloads){.reference
                .internal}
            -   [Multi-Argument
                Payloads](realtime/protocol.html#multi-argument-payloads){.reference
                .internal}
            -   [Size
                Constraints](realtime/protocol.html#size-constraints){.reference
                .internal}
            -   [Encoding
                Examples](realtime/protocol.html#encoding-examples){.reference
                .internal}
            -   [Bit-Packed Data
                Encoding](realtime/protocol.html#bit-packed-data-encoding){.reference
                .internal}
            -   [Multi-Bit Measurement
                Encoding](realtime/protocol.html#multi-bit-measurement-encoding){.reference
                .internal}
        -   [Response
            Encoding](realtime/protocol.html#response-encoding){.reference
            .internal}
            -   [Single-Result
                Response](realtime/protocol.html#single-result-response){.reference
                .internal}
            -   [Multi-Result
                Response](realtime/protocol.html#multi-result-response){.reference
                .internal}
            -   [Status
                Codes](realtime/protocol.html#status-codes){.reference
                .internal}
        -   [QEC-Specific Usage
            Example](realtime/protocol.html#qec-specific-usage-example){.reference
            .internal}
            -   [QEC
                Terminology](realtime/protocol.html#qec-terminology){.reference
                .internal}
            -   [QEC Decoder
                Handler](realtime/protocol.html#qec-decoder-handler){.reference
                .internal}
            -   [Decoding
                Rounds](realtime/protocol.html#decoding-rounds){.reference
                .internal}
    -   [CPU RoCE Transport](realtime/cpu_transport.html){.reference
        .internal}
        -   [C ABI](realtime/cpu_transport.html#c-abi){.reference
            .internal}
        -   [Two-phase bring-up ([`setup`{.docutils .literal
            .notranslate}]{.pre} / [`connect`{.docutils .literal
            .notranslate}]{.pre})](realtime/cpu_transport.html#two-phase-bring-up-setup-connect){.reference
            .internal}
        -   [TX modes](realtime/cpu_transport.html#tx-modes){.reference
            .internal}
        -   [Testing ([`hsb_bridge_cpu`{.docutils .literal
            .notranslate}]{.pre})](realtime/cpu_transport.html#testing-hsb-bridge-cpu){.reference
            .internal}
    -   [Device Call Channels](realtime/device_call.html){.reference
        .internal}
        -   [The [`device_call`{.docutils .literal .notranslate}]{.pre}
            model](realtime/device_call.html#the-device-call-model){.reference
            .internal}
        -   [Selecting a
            channel](realtime/device_call.html#selecting-a-channel){.reference
            .internal}
        -   [Extending an in-process
            service](realtime/device_call.html#extending-an-in-process-service){.reference
            .internal}
        -   [The [`cpu_roce`{.docutils .literal .notranslate}]{.pre}
            channel](realtime/device_call.html#the-cpu-roce-channel){.reference
            .internal}
            -   [Wire pattern
                (FPGA-compatible)](realtime/device_call.html#wire-pattern-fpga-compatible){.reference
                .internal}
            -   [Connection
                setup](realtime/device_call.html#connection-setup){.reference
                .internal}
            -   [Running
                it](realtime/device_call.html#running-it){.reference
                .internal}
            -   [Test
                harness](realtime/device_call.html#test-harness){.reference
                .internal}
-   [CUDA-Q QEC](cudaq-qec/cudaq-qec.html){.reference .internal}
-   [CUDA-Q
    Algorithms](cudaq-algorithms/cudaq-algorithms.html){.reference
    .internal}
-   [Installation](install/install.html){.reference .internal}
    -   [Local Installation](install/local_installation.html){.reference
        .internal}
        -   [Introduction](install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
            -   [Dynamic linking to GMP and
                MPFR](install/local_installation.html#dynamic-linking-to-gmp-and-mpfr){.reference
                .internal}
        -   [Next
            Steps](install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](integration/integration.html){.reference .internal}
    -   [Downstream CMake
        Integration](integration/cmake_app.html){.reference .internal}
    -   [Combining CUDA with
        CUDA-Q](integration/cuda_gpu.html){.reference .internal}
    -   [Integrating with Third-Party
        Libraries](integration/libraries.html){.reference .internal}
        -   [Calling a CUDA-Q library from
            C++](integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](extending/extending.html){.reference .internal}
    -   [Compiler development](extending/compiler/index.html){.reference
        .internal}
        -   [Compiler IR](extending/compiler/cudaq_ir.html){.reference
            .internal}
            -   [CUDA-Q
                dialects](extending/compiler/cudaq_ir.html#cuda-q-dialects){.reference
                .internal}
            -   [Source and
                tests](extending/compiler/cudaq_ir.html#source-and-tests){.reference
                .internal}
        -   [Value-semantics IR change
            log](extending/compiler/ir_changelog.html){.reference
            .internal}
            -   [Producing this
                IR](extending/compiler/ir_changelog.html#producing-this-ir){.reference
                .internal}
            -   [Upcoming release ([`main`{.docutils .literal
                .notranslate}]{.pre}
                branch)](extending/compiler/ir_changelog.html#upcoming-release-main-branch){.reference
                .internal}
            -   [0.16.0](extending/compiler/ir_changelog.html#id5){.reference
                .internal}
        -   [External compiler pass
            plugins](extending/compiler/pass_plugins.html){.reference
            .internal}
            -   [Implement and register the
                pass](extending/compiler/pass_plugins.html#implement-and-register-the-pass){.reference
                .internal}
            -   [Build the
                plugin](extending/compiler/pass_plugins.html#build-the-plugin){.reference
                .internal}
            -   [Load and test the
                plugin](extending/compiler/pass_plugins.html#load-and-test-the-plugin){.reference
                .internal}
        -   [Using compile targets to customize
            compilation](extending/compiler/compile_targets.html){.reference
            .internal}
            -   [Constructing single-stage compile
                targets](extending/compiler/compile_targets.html#constructing-single-stage-compile-targets){.reference
                .internal}
            -   [Setting compile targets with [`set_target`{.docutils
                .literal
                .notranslate}]{.pre}](extending/compiler/compile_targets.html#setting-compile-targets-with-set-target){.reference
                .internal}
            -   [Example](extending/compiler/compile_targets.html#example){.reference
                .internal}
    -   [Add a hardware backend](extending/backend.html){.reference
        .internal}
        -   [Plugin Directory
            Structure](extending/backend.html#plugin-directory-structure){.reference
            .internal}
        -   [REST-Style Backends (Server
            Helper)](extending/backend.html#rest-style-backends-server-helper){.reference
            .internal}
            -   [Server Helper
                Class](extending/backend.html#server-helper-class){.reference
                .internal}
            -   [Target YAML
                Configuration](extending/backend.html#target-yaml-configuration){.reference
                .internal}
            -   [CMake Build
                File](extending/backend.html#cmake-build-file){.reference
                .internal}
        -   [Auxiliary Files and [`%PLUGIN_ROOT%`{.docutils .literal
            .notranslate}]{.pre}](extending/backend.html#auxiliary-files-and-plugin-root){.reference
            .internal}
        -   [Testing Your
            Backend](extending/backend.html#testing-your-backend){.reference
            .internal}
        -   [Example
            Usage](extending/backend.html#example-usage){.reference
            .internal}
        -   [Next Steps](extending/backend.html#next-steps){.reference
            .internal}
    -   [Package & distribute a backend
        plugin](extending/packaging.html){.reference .internal}
        -   [Plugin Package
            Layout](extending/packaging.html#plugin-package-layout){.reference
            .internal}
        -   [How to pre-compile a target config
            YAML](extending/packaging.html#how-to-pre-compile-a-target-config-yaml){.reference
            .internal}
        -   [Target YAML Reference (Plugin
            Fields)](extending/packaging.html#target-yaml-reference-plugin-fields){.reference
            .internal}
            -   [[`%PLUGIN_ROOT%`{.docutils .literal
                .notranslate}]{.pre}](extending/packaging.html#plugin-root){.reference
                .internal}
            -   [[`target-arguments`{.docutils .literal
                .notranslate}]{.pre}](extending/packaging.html#target-arguments){.reference
                .internal}
        -   [Building with [`CUDAQ_EXTERNAL_PROJECTS`{.docutils .literal
            .notranslate}]{.pre}](extending/packaging.html#building-with-cudaq-external-projects){.reference
            .internal}
            -   [Baking a target into the pre-compiled database instead
                of shipping a
                plugin](extending/packaging.html#baking-a-target-into-the-pre-compiled-database-instead-of-shipping-a-plugin){.reference
                .internal}
        -   [Python
            Packaging](extending/packaging.html#python-packaging){.reference
            .internal}
            -   [[`pyproject.toml`{.docutils .literal
                .notranslate}]{.pre}](extending/packaging.html#pyproject-toml){.reference
                .internal}
            -   [[`__init__.py`{.docutils .literal
                .notranslate}]{.pre}](extending/packaging.html#init-py){.reference
                .internal}
            -   [[`__main__.py`{.docutils .literal .notranslate}]{.pre}
                ([`--install-nvqpp`{.docutils .literal
                .notranslate}]{.pre}
                hook)](extending/packaging.html#main-py-install-nvqpp-hook){.reference
                .internal}
        -   [Installing the Plugin for End
            Users](extending/packaging.html#installing-the-plugin-for-end-users){.reference
            .internal}
            -   [[`pip`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`install`{.docutils .literal
                .notranslate}]{.pre} (Python --- zero
                config)](extending/packaging.html#pip-install-python-zero-config){.reference
                .internal}
            -   [[`--install-nvqpp`{.docutils .literal
                .notranslate}]{.pre} (make visible to [`nvq++`{.docutils
                .literal
                .notranslate}]{.pre})](extending/packaging.html#install-nvqpp-make-visible-to-nvq){.reference
                .internal}
            -   [[`cudaq-install-plugin`{.docutils .literal
                .notranslate}]{.pre} (C++-only
                workflows)](extending/packaging.html#cudaq-install-plugin-c-only-workflows){.reference
                .internal}
        -   [Discovery
            Mechanics](extending/packaging.html#discovery-mechanics){.reference
            .internal}
            -   [[`nvq++`{.docutils .literal .notranslate}]{.pre} target
                resolution](extending/packaging.html#nvq-target-resolution){.reference
                .internal}
            -   [Python target
                resolution](extending/packaging.html#python-target-resolution){.reference
                .internal}
            -   [Environment
                variables](extending/packaging.html#environment-variables){.reference
                .internal}
        -   [Reference
            Plugins](extending/packaging.html#reference-plugins){.reference
            .internal}
        -   [Quick-Start
            Checklist](extending/packaging.html#quick-start-checklist){.reference
            .internal}
    -   [Create an NVQIR
        simulator](extending/nvqir_simulator.html){.reference .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
-   [Specifications](../specification/index.html){.reference .internal}
    -   [Language Specification](../specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](../specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](../specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](../specification/cudaq/types.html){.reference
            .internal}
            -   [3.1. [`cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. [`cudaq::qubit`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](../specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](../specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. [`cudaq::spin_op`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](../specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on [`cudaq::qubit`{.code .docutils
                .literal
                .notranslate}]{.pre}](../specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](../specification/cudaq/kernels.html){.reference
            .internal}
            -   [6.1. Atomic quantum
                regions](../specification/cudaq/kernels.html#atomic-quantum-regions){.reference
                .internal}
        -   [7. Sub-circuit
            Synthesis](../specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](../specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](../specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](../specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](../specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11.
            Platform](../specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](../specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. [`cudaq::sample`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. [`cudaq::run`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. [`cudaq::observe`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. [`cudaq::optimizer`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. [`cudaq::gradient`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](../specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](../specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](../specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](../specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](../specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](../specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](../specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](../specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](../specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](../specification/quake-dialect.html#motivation){.reference
            .internal}
        -   [Calling between reference and value
            forms](../specification/quake-dialect.html#calling-between-reference-and-value-forms){.reference
            .internal}
-   [API Reference](../api/api.html){.reference .internal}
    -   [C++ API](../api/languages/cpp_api.html){.reference .internal}
        -   [Operators](../api/languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](../api/languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](../api/languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](../api/languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](../api/languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](../api/languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Quantum Error
            Correction](../api/languages/cpp_api.html#quantum-error-correction){.reference
            .internal}
        -   [Platform](../api/languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](../api/languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](../api/languages/cpp_api.html#namespaces){.reference
            .internal}
        -   [PTSBE](../api/languages/cpp_api.html#ptsbe){.reference
            .internal}
            -   [Sampling
                Functions](../api/languages/cpp_api.html#sampling-functions){.reference
                .internal}
            -   [Options](../api/languages/cpp_api.html#options){.reference
                .internal}
            -   [Result
                Type](../api/languages/cpp_api.html#result-type){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](../api/languages/cpp_api.html#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](../api/languages/cpp_api.html#shot-allocation-strategy){.reference
                .internal}
            -   [Execution
                Data](../api/languages/cpp_api.html#execution-data){.reference
                .internal}
            -   [Trajectory and Selection
                Types](../api/languages/cpp_api.html#trajectory-and-selection-types){.reference
                .internal}
    -   [Python API](../api/languages/python_api.html){.reference
        .internal}
        -   [Program
            Construction](../api/languages/python_api.html#program-construction){.reference
            .internal}
            -   [[`make_kernel()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [[`PyKernel`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [[`Kernel`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [[`PyKernelDecorator`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [[`kernel()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](../api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [[`sample_async()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [[`run()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [[`run_async()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [[`observe()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [[`observe_async()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [[`get_state()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [[`get_state_async()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [[`vqe()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [[`draw()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [[`translate()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [[`estimate()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.estimate){.reference
                .internal}
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
            -   [[`dem_from_kernel()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.dem_from_kernel){.reference
                .internal}
        -   [[`cudaq.contrib`{.docutils .literal
            .notranslate}]{.pre}](../api/languages/python_api.html#cudaq-contrib){.reference
            .internal}
            -   [Quantum
                Embeddings](../api/languages/python_api.html#quantum-embeddings){.reference
                .internal}
            -   [Qiskit and OpenQASM
                Interoperability](../api/languages/python_api.html#qiskit-and-openqasm-interoperability){.reference
                .internal}
        -   [Quantum Error
            Correction](../api/languages/python_api.html#quantum-error-correction){.reference
            .internal}
            -   [[`detector()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.detector){.reference
                .internal}
            -   [[`detectors()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.detectors){.reference
                .internal}
            -   [[`logical_observable()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.logical_observable){.reference
                .internal}
            -   [[`to_bools()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.to_bools){.reference
                .internal}
        -   [Backend
            Configuration](../api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [[`parse_args()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.parse_args){.reference
                .internal}
            -   [[`has_target()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [[`get_target()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [[`get_targets()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [[`set_target()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [[`reset_target()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [[`set_noise()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [[`unset_noise()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [[`register_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [[`unregister_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [[`cudaq.apply_noise()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [[`initialize_cudaq()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [[`num_available_gpus()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [[`set_random_seed()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](../api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [[`evolve()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [[`evolve_async()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [[`Schedule`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [[`BaseIntegrator`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [[`InitialState`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [[`InitialStateType`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [[`IntermediateResultSave`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](../api/languages/python_api.html#operators){.reference
            .internal}
            -   [[`OperatorSum`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [[`ProductOperator`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [[`ElementaryOperator`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [[`ScalarOperator`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [[`RydbergHamiltonian`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [[`SuperOperator`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [[`operators.define()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [[`operators.instantiate()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](../api/languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](../api/languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](../api/languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](../api/languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](../api/languages/python_api.html#data-types){.reference
            .internal}
            -   [[`SimulationPrecision`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [[`Target`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [[`State`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [[`Tensor`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [[`QuakeValue`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [[`qubit`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [[`qreg`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [[`qvector`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [[`measure_handle`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.measure_handle){.reference
                .internal}
            -   [[`ComplexMatrix`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [[`SampleResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [[`AsyncSampleResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [[`DEMResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.DEMResult){.reference
                .internal}
            -   [[`ObserveResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [[`AsyncObserveResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [[`AsyncStateResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [[`OptimizationResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [[`EvolveResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [[`AsyncEvolveResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [[`Resources`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.Resources){.reference
                .internal}
            -   [[`EstimateResult`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.EstimateResult){.reference
                .internal}
            -   [Optimizers](../api/languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](../api/languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](../api/languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](../api/languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [[`initialize()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [[`rank()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [[`num_ranks()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [[`all_gather()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [[`broadcast()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [[`is_initialized()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [[`split_communicator()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.mpi.split_communicator){.reference
                .internal}
            -   [[`set_communicator()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.mpi.set_communicator){.reference
                .internal}
            -   [[`finalize()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](../api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
        -   [PTSBE
            Submodule](../api/languages/python_api.html#ptsbe-submodule){.reference
            .internal}
            -   [Sampling
                Functions](../api/languages/python_api.html#sampling-functions){.reference
                .internal}
            -   [Result
                Type](../api/languages/python_api.html#result-type){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](../api/languages/python_api.html#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](../api/languages/python_api.html#shot-allocation-strategy){.reference
                .internal}
            -   [Execution
                Data](../api/languages/python_api.html#execution-data){.reference
                .internal}
            -   [Trajectory and Selection
                Types](../api/languages/python_api.html#trajectory-and-selection-types){.reference
                .internal}
        -   [Synth
            Submodule](../api/languages/python_api.html#synth-submodule){.reference
            .internal}
            -   [[`gridsynth()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.synth.gridsynth){.reference
                .internal}
            -   [[`rz_error()`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.synth.rz_error){.reference
                .internal}
            -   [[`CliffordTSequence`{.docutils .literal
                .notranslate}]{.pre}](../api/languages/python_api.html#cudaq.synth.CliffordTSequence){.reference
                .internal}
    -   [Quantum Operations](../api/default_ops.html){.reference
        .internal}
        -   [Unitary Operations on
            Qubits](../api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [[`x`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#x){.reference
                .internal}
            -   [[`y`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#y){.reference
                .internal}
            -   [[`z`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#z){.reference
                .internal}
            -   [[`h`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#h){.reference
                .internal}
            -   [[`r1`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#r1){.reference
                .internal}
            -   [[`rx`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#rx){.reference
                .internal}
            -   [[`ry`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#ry){.reference
                .internal}
            -   [[`rz`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#rz){.reference
                .internal}
            -   [[`s`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#s){.reference
                .internal}
            -   [[`t`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#t){.reference
                .internal}
            -   [[`swap`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#swap){.reference
                .internal}
            -   [[`u3`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#u3){.reference
                .internal}
            -   [[`exp_pauli`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#exp-pauli){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](../api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](../api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#mz){.reference
                .internal}
            -   [[`mx`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#mx){.reference
                .internal}
            -   [[`my`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](../api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](../api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [[`create`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#create){.reference
                .internal}
            -   [[`annihilate`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#annihilate){.reference
                .internal}
            -   [[`phase_shift`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#phase-shift){.reference
                .internal}
            -   [[`beam_splitter`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](../api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](../versions.html){.reference .internal}

[Preview]{.caption-text}

-   [cudaq-logical](/preview/logical/#http://){.reference .external}
-   [cudaq-edem](#){.current .reference .internal}
    -   [Standalone build](#standalone-build){.reference .internal}
    -   [Python bindings and
        wheel](#python-bindings-and-wheel){.reference .internal}
    -   [CUDA-Q build](#cuda-q-build){.reference .internal}
    -   [Installed package](#installed-package){.reference .internal}
    -   [Packaging checks](#packaging-checks){.reference .internal}
    -   [Native interface and
        memory](#native-interface-and-memory){.reference .internal}
    -   [C++ ownership helper](#c-ownership-helper){.reference
        .internal}
    -   [Component boundary](#component-boundary){.reference .internal}
    -   [Fault distributions](#fault-distributions){.reference
        .internal}
    -   [Python construction](#python-construction){.reference
        .internal}
    -   [Packed format](#packed-format){.reference .internal}
    -   [Serial composition and DEM
        projection](#serial-composition-and-dem-projection){.reference
        .internal}
    -   [Exact noise conversion and column
        coalescing](#exact-noise-conversion-and-column-coalescing){.reference
        .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](../index.html)

::: wy-nav-content
::: rst-content
::: {role="navigation" aria-label="Page navigation"}
-   [](../index.html){.icon .icon-home aria-label="Home"}
-   cudaq-edem
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](../versions.html "CUDA-Q Versions"){.btn .btn-neutral
.float-left accesskey="p"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#cudaq-edem .section}
# cudaq-edem[¶](#cudaq-edem "Permalink to this heading"){.headerlink}

> <div>
>
> \[!NOTE\] This library's API, ABI, and packed format are unstable and
> subject to breaking changes during 0.x development. It is provided
> as-is for the sake of open development and the benefit of developers
> interested in using it on that basis. Feedback and contributions are
> always welcome, just please understand that its purpose is the support
> of other user-facing components in CUDA-Q.
>
> </div>

cudaq-edem provides shared data structures and operations for extended
detector error models (EDEMs). Its packed representation and C API allow
compilers and runtime components to exchange and evaluate EDEMs.

This component provides caller-owned sparse GF(2) matrices, typed linear
maps, and open EDEM values. It constructs, validates, packs, reads, and
evaluates models independently of cudaq-logical. The public interface is
C11. The private implementation uses C++20 without exceptions, RTTI, or
a C++ runtime dependency. Setup operations use per-call allocation
callbacks, with [`malloc`{.docutils .literal
.notranslate}]{.pre}/[`free`{.docutils .literal .notranslate}]{.pre} as
the default. Evaluation uses caller vectors and never allocates.

GF2 sources live in [`math/gf2`{.docutils .literal .notranslate}]{.pre}.
cudaq-edem owns their combined installation and CMake package. GF2 has
no separate package or release version.

::: {#standalone-build .section}
## Standalone build[¶](#standalone-build "Permalink to this heading"){.headerlink}

Building the libraries requires CMake 3.23 or newer, a C11 compiler, and
a C++20 compiler. C applications can consume the installed package
without enabling C++.

From the CUDA-Q repository root:

::: {.highlight-sh .notranslate}
::: highlight
    cmake -S edem -B build/edem -DBUILD_SHARED_LIBS=ON
    cmake --build build/edem
    ctest --test-dir build/edem --output-on-failure
    cmake --install build/edem --prefix /path/to/install
:::
:::

Set [`BUILD_SHARED_LIBS=OFF`{.docutils .literal .notranslate}]{.pre} for
static libraries and [`CUDAQ_EDEM_BUILD_TESTS=OFF`{.docutils .literal
.notranslate}]{.pre} to omit tests. The standalone build brings GF2 in
from the sibling source directory. GF2 can also build and test
independently, without configuring cudaq-edem:

::: {.highlight-sh .notranslate}
::: highlight
    cmake -S math/gf2 -B build/gf2 -DBUILD_SHARED_LIBS=ON
    cmake --build build/gf2
    ctest --test-dir build/gf2 --output-on-failure
:::
:::

Use [`CUDAQ_GF2_BUILD_TESTS=OFF`{.docutils .literal .notranslate}]{.pre}
to omit its sparse-matrix and typed-map tests. When included by
cudaq-edem, their default follows [`CUDAQ_EDEM_BUILD_TESTS`{.docutils
.literal .notranslate}]{.pre}. GF2's standalone build does not create
another distribution or install package.
:::

::: {#python-bindings-and-wheel .section}
## Python bindings and wheel[¶](#python-bindings-and-wheel "Permalink to this heading"){.headerlink}

Python bindings use nanobind and require Python 3.11 or newer. Enable
them in a native build with [`CUDAQ_EDEM_BUILD_PYTHON=ON`{.docutils
.literal .notranslate}]{.pre} after installing
[`nanobind>=2.12,<3`{.docutils .literal .notranslate}]{.pre} into the
chosen Python environment. Set [`Python_EXECUTABLE`{.docutils .literal
.notranslate}]{.pre} to select an interpreter. Standalone native builds
default to [`CUDAQ_EDEM_BUILD_PYTHON=OFF`{.docutils .literal
.notranslate}]{.pre}.

The [`cudaq-edem`{.docutils .literal .notranslate}]{.pre} wheel includes
the [`cudaq_edem`{.docutils .literal .notranslate}]{.pre} Python
package, both native libraries, headers, and CMake exports. The Python
module owns aligned buffers and delegates construction, validation, and
evaluation to the C interface. Python requires no NumPy or Stim
dependency. Dependent packages can link to the standalone wheel's native
libraries and use its ownership protocol to exchange packed EDEM values
without bundling another copy of the libraries.

Python exposes EDEM values, sparse matrices, serial composition, and
checked DEM projection. Generic typed-map construction and section
access are available through the C interface.

From the repository root, build the source distribution and its wheel:

::: {.highlight-sh .notranslate}
::: highlight
    python3 -m pip install "build[virtualenv]"
    python3 edem/scripts/build_wheel.py --outdir dist/edem
:::
:::

The script stages the cudaq-edem and GF2 sources with
[`pyproject.toml.edem`{.docutils .literal .notranslate}]{.pre} as the
standard [`pyproject.toml`{.docutils .literal .notranslate}]{.pre}. It
leaves the repository's main wheel configuration unchanged.
Scikit-build-core builds the standalone cudaq-edem project and packages
its CMake installation. The wheel version comes from
[`CUDAQ_EDEM_VERSION`{.docutils .literal .notranslate}]{.pre}.

Source staging copies only the exact paths in
[`edem/package-sources.txt`{.docutils .literal .notranslate}]{.pre}. Add
new package files to that manifest; missing files and paths outside the
source tree are errors. Build release artifacts from a clean checkout so
listed files also have their committed contents.

Install the resulting wheel with pip. A downstream CMake build can
locate its package configuration through the same Python environment:

::: {.highlight-sh .notranslate}
::: highlight
    cmake -S /path/to/consumer -B build-consumer \
      -Dcudaq-edem_DIR="$(python3 -c 'import cudaq_edem; print(cudaq_edem.get_cmake_dir())')"
:::
:::

Wheels use CPython and platform compatibility tags. CI produces wheel
and source-distribution artifacts; public release publishing is a
separate step.
:::

::: {#cuda-q-build .section}
## CUDA-Q build[¶](#cuda-q-build "Permalink to this heading"){.headerlink}

Append [`edem`{.docutils .literal .notranslate}]{.pre} to
[`CUDAQ_ENABLE_PROJECTS`{.docutils .literal .notranslate}]{.pre}, for
example:

::: {.highlight-sh .notranslate}
::: highlight
    cmake -S . -B build -DCUDAQ_ENABLE_PROJECTS="cudaq;runtime;edem"
:::
:::

The top-level CUDA-Q configuration retains its usual toolchain and
dependency requirements. cudaq-edem tests default to off in a parent
build; enable them with [`CUDAQ_EDEM_BUILD_TESTS=ON`{.docutils .literal
.notranslate}]{.pre}.

When [`CUDAQ_ENABLE_PROJECTS`{.docutils .literal .notranslate}]{.pre}
also includes [`python`{.docutils .literal .notranslate}]{.pre},
cudaq-edem Python bindings default to on. Set
[`CUDAQ_EDEM_BUILD_PYTHON=OFF`{.docutils .literal .notranslate}]{.pre}
to omit them, or set it to [`ON`{.docutils .literal .notranslate}]{.pre}
to build them independently of the parent's Python selection. Explicit
component options take precedence over these defaults.
:::

::: {#installed-package .section}
## Installed package[¶](#installed-package "Permalink to this heading"){.headerlink}

Use the same targets in source-tree and installed builds:

::: {.highlight-cmake .notranslate}
::: highlight
    find_package(cudaq-edem CONFIG REQUIRED)
    target_link_libraries(my_application PRIVATE cudaq::edem)
:::
:::

Add the installation prefix to [`CMAKE_PREFIX_PATH`{.docutils .literal
.notranslate}]{.pre}. The package also exports [`cudaq::gf2`{.docutils
.literal .notranslate}]{.pre}; linking cudaq-edem supplies that
dependency transitively. Both targets share the cudaq-edem distribution
version.

Each native shared library exports an explicit list of C symbols. On
ELF, the cudaq-edem and GF2 lists use the [`CUDAQ_EDEM_0.1`{.docutils
.literal .notranslate}]{.pre} and [`CUDAQ_GF2_0.1`{.docutils .literal
.notranslate}]{.pre} ABI version nodes. macOS derives its export lists
from the same files. Build checks verify the complete exported surface
and, on ELF, each symbol's version node.

Headers, static archives, and package configuration files belong to the
development install component; shared libraries belong to the runtime
component. Parent CUDA-Q builds supply their component names. Standalone
builds use [`Development`{.docutils .literal .notranslate}]{.pre} and
[`Runtime`{.docutils .literal .notranslate}]{.pre}. Python bindings and
their third-party notices belong to [`EDEMPython`{.docutils .literal
.notranslate}]{.pre}.
:::

::: {#packaging-checks .section}
## Packaging checks[¶](#packaging-checks "Permalink to this heading"){.headerlink}

The tests compile and link C and C++ consumers using the build-tree
targets, then install, relocate, and consume the package from a separate
build. They also exercise a parent that defines GF2 before adding
cudaq-edem and installs both libraries through one [`Core`{.docutils
.literal .notranslate}]{.pre} component, and a standalone build with
custom library and header installation directories.

Component checks exhaust the responses of an open syndrome-round
fixture, including incoming boundary errors and faults that survive at
the output. They also check sparse construction, exact fault groups,
canonical packing, relocation, invalid content, large fault
distributions, and canonical packed bytes. GF2's own tests under
[`math/gf2/test`{.docutils .literal .notranslate}]{.pre} check sparse
matrices, typed maps, multiwire bundles and the trailing directory
without linking cudaq-edem. Native build checks permit
[`malloc`{.docutils .literal .notranslate}]{.pre}/[`free`{.docutils
.literal .notranslate}]{.pre} only in GF2's default allocator and reject
other allocator imports, C++ runtime dependencies, and exports outside
the C API. Failure-injection and bounded-arena tests check callback
routing, cleanup and result publication. Relocated package checks also
build a consumer that enables only C, for both shared and static
libraries.

When Python bindings are enabled, CTest also checks the build-tree
import. Its value tests require the native producer helper and exercise
ownership transfer, callback lifetimes, and allocation-failure cleanup.
The helper is absent from installed packages.
[`edem/test/check_wheel.py`{.docutils .literal .notranslate}]{.pre}
installs a wheel into a fresh virtual environment, imports its
extension, loads both native libraries, relocates the installed package,
and compiles C and C++ consumers against its CMake exports. Run it with
the path to the wheel as its argument.

[`edem/test/check_sdist.py`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`--outdir`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`dist/edem`{.docutils .literal .notranslate}]{.pre} builds
from a temporary source tree containing additional unlisted files. It
checks the source archive against the manifest and source contents, then
copies the source archive and its wheel to the requested output
directory. This check requires no Git metadata.
:::

::: {#native-interface-and-memory .section}
## Native interface and memory[¶](#native-interface-and-memory "Permalink to this heading"){.headerlink}

The C headers are the native user surface. [`cudaq/gf2/core.h`{.docutils
.literal .notranslate}]{.pre} provides sparse matrices;
[`cudaq/gf2/map.h`{.docutils .literal .notranslate}]{.pre} provides
typed linear maps and their packed storage.
[`cudaq/edem/core.h`{.docutils .literal .notranslate}]{.pre} adds
semantic construction, presentation and distribution checks, endpoint
translation, and access to fault distributions.

Construction accepts matrix blocks keyed by semantic endpoints such as
output syndrome of code bundle 2. Producers supply code presentations
and coefficients; cudaq-edem builds the type table and bundles. It
returns a checked [`cudaq_edem_view`{.docutils .literal
.notranslate}]{.pre} containing a GF2 map in its [`.map`{.docutils
.literal .notranslate}]{.pre} member. EDEM getters accept this distinct
checked type. Use [`&value.map`{.docutils .literal .notranslate}]{.pre}
with the GF2 map API for bundles, widths, offsets, blocks, vector sizes
and evaluation. There is no duplicate EDEM map-getter family or apply
function. [`cudaq_edem_resolve_endpoint`{.docutils .literal
.notranslate}]{.pre} and [`cudaq_edem_describe_wire`{.docutils .literal
.notranslate}]{.pre} translate vocabulary between semantic endpoints and
generic bundle/wire coordinates.

Construction is one call. The library computes storage requirements
internally, allocates temporary scratch and an exact packed result, and
returns a caller-held [`cudaq_gf2_owned_buffer`{.docutils .literal
.notranslate}]{.pre} with a checked, borrowed view:

::: {.highlight-c .notranslate}
::: highlight
    cudaq_gf2_owned_buffer owner = {0};
    cudaq_edem_view value;
    cudaq_edem_status status = cudaq_edem_construct(&description, NULL, &owner, &value);
    if (status == CUDAQ_EDEM_OK) {
      /* Query or evaluate value.map while owner.data remains alive and immutable. */
      cudaq_gf2_owned_buffer_release(&owner);
    }
:::
:::

[`cudaq_gf2_map_construct`{.docutils .literal .notranslate}]{.pre}
provides the same ownership model for generic maps.
[`cudaq/gf2/memory.h`{.docutils .literal .notranslate}]{.pre} defines
GF2 allocation and ownership types shared by both libraries. The owner
stores the packed pointer, size, deallocation callback and callback
context directly; it needs no hidden header, allocated descriptor,
reference count or library handle. It is a unique owner: move it by
copying the record and clearing the source. Release clears the record
and invokes the captured callback once. The callback, its code and its
context must remain alive until release. Moving an owner does not move
its bytes or invalidate existing borrowed views.

Each setup call accepts an optional [`cudaq_gf2_memory`{.docutils
.literal .notranslate}]{.pre} with separate [`scratch`{.docutils
.literal .notranslate}]{.pre} and [`result`{.docutils .literal
.notranslate}]{.pre} allocator slots. A null policy uses
[`cudaq_gf2_default_allocator()`{.docutils .literal .notranslate}]{.pre}
for both. An explicit used slot must provide both callbacks; missing
callbacks are invalid, and there is no fallback to the system allocator.
The same provider may fill both slots. Scratch is released before
returning and is never retained; the result captures only its
deallocator and context, not the policy object. Construct requires an
empty result owner. Failures leave output owners and views unchanged and
release all temporary or unpublished allocations.

For bounded operation, [`cudaq/gf2/bump_allocator.h`{.docutils .literal
.notranslate}]{.pre} supplies a C11 header-only provider over caller
storage. Its deallocator does nothing. Exhaustion returns
[`ALLOCATION_FAILED`{.docutils .literal .notranslate}]{.pre}, and failed
operations do not rewind the arena cursor. Use separate scratch and
result arenas when scratch should be reset after each call; reset an
arena only when none of its allocations remain live. There is no public
exact-size preflight for packed map construction. The low-level CSC
matrix builder retains its caller-buffer interface.

To load an EDEM, call [`cudaq_edem_read(buffer,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`size,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`memory,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`&value)`{.docutils .literal .notranslate}]{.pre}. It
validates GF2 content and EDEM semantics before publishing the checked
view. Reading is zero-copy: the view borrows the supplied aligned,
immutable bytes. Read, description validation and semantic checking use
only the scratch slot.

Advanced callers with an already validated GF2 map can use
[`cudaq_edem_check`{.docutils .literal .notranslate}]{.pre} to obtain a
checked EDEM view. [`cudaq_edem_schema()`{.docutils .literal
.notranslate}]{.pre} declares recognized sections for an explicit GF2
read; recognition alone does not establish EDEM semantics. Ordinary
loading uses the EDEM reader so the semantic check cannot accidentally
be skipped. Checked views borrow immutable backing bytes; fabricating a
struct or mutating its backing bytes violates the token's initialization
and lifetime rules.

Operations retain no state, caches or allocator policy between calls.
Getters and evaluation allocate nothing. The default provider is the
only code in the native libraries that imports [`malloc`{.docutils
.literal .notranslate}]{.pre} and [`free`{.docutils .literal
.notranslate}]{.pre}.

Canonical ordering sorts index arrays. Matrix interning uses
fingerprints with full equality checks. Fault coverage uses one bit per
indicator. The semantic constructor also stages its lowered map
description and canonical EDEM section payloads in temporary scratch;
its storage is linear in the input size. Packed values remain
independent of this setup storage.
:::

::: {#c-ownership-helper .section}
## C++ ownership helper[¶](#c-ownership-helper "Permalink to this heading"){.headerlink}

The installed header [`cudaq/edem/owned.hpp`{.docutils .literal
.notranslate}]{.pre} provides [`cudaq::edem::v0::PackedEDEM`{.docutils
.literal .notranslate}]{.pre}, a move-only C++17 owner over the C
ownership record and a checked EDEM view. Its operations return C status
codes and work with exceptions and RTTI disabled. Construction takes the
semantic C description.

cudaq-logical can keep this owner in compiler code and translate status
codes into LLVM or MLIR diagnostics. A library such as cudaq-qec can
reuse it inside its model implementation and provide its own public C++
type for decoder-facing operations. An implementation behind a pointer
keeps the helper out of that type's binary layout. This reuses ownership
and validation without duplicating the library's domain API.

Applications may use the installed header directly. It is a source
convenience, compiled in each consumer with hidden visibility and a
versioned inline namespace. Cross-package exchange uses the C ownership
record and packed bytes; each receiving consumer obtains its own checked
view.

For a caller-supplied description and optional memory provider:

::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::edem::PackedEDEM model;
    auto status = cudaq::edem::PackedEDEM::construct(description, model, memory);
    if (status != CUDAQ_EDEM_OK)
      return status;

    const cudaq_edem_view &view = model.view();
    // Use the C EDEM getters and GF2 operations while model owns the bytes.
:::
:::

[`construct`{.docutils .literal .notranslate}]{.pre} replaces its output
on success and preserves it on failure. [`adopt`{.docutils .literal
.notranslate}]{.pre} validates an aligned, uniquely owned C record and
consumes it only on success, without copying the payload.
[`clone`{.docutils .literal .notranslate}]{.pre} makes an independent
copy and validates it through a fresh read. [`release`{.docutils
.literal .notranslate}]{.pre} returns the C record and clears the owner
and cached view; [`reset`{.docutils .literal .notranslate}]{.pre} frees
the payload. A view requires a nonempty owner and cannot be obtained
from a temporary. Copy a C view token before moving an owner if that
token must survive the move; references into the helper object itself
follow that object's lifetime and state.

Setup methods accept the same per-call memory provider as the C API,
including reusable scratch arenas. The helper retains only the result's
C ownership record. Allocator callbacks and contexts must outlive the
owner and any released record. Consumers that retain callback or module
dependencies beside the helper must destroy the payload owner first.
Python producers can release the C record into the capsule protocol in
[`cudaq/edem/python.h`{.docutils .literal .notranslate}]{.pre}.
:::

::: {#component-boundary .section}
## Component boundary[¶](#component-boundary "Permalink to this heading"){.headerlink}

EDEM supports the map [`(F,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`V_in,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`S_in,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`L_in)`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal .notranslate}[`->`{.docutils
.literal .notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`(D,`{.docutils .literal .notranslate}]{.pre}` `{.docutils
.literal .notranslate}[`O,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`Success,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`V_out,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`S_out,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`L_out)`{.docutils .literal .notranslate}]{.pre}. Each
code bundle has three wires, [`(V,S,L)`{.docutils .literal
.notranslate}]{.pre}, with widths [`(v,s,2k)`{.docutils .literal
.notranslate}]{.pre}. Input and output code bundle counts are
independent.

A type is a nonempty UTF-8 name, a role and a wire count. Names compare
by exact UTF-8 bytes, without normalization. Types are nominal, with no
subtyping. Free bundles do not participate in wiring. Interface bundle
compatibility requires matching type names, roles, wire counts and
widths. Local type IDs are indices, not cross-value identities. GF2
constructs, validates and applies typed maps; cudaq-edem also provides
serial composition of its packed values.

EDEM uses four reserved, single-wire free types: [`edem.fault`{.docutils
.literal .notranslate}]{.pre}, [`edem.detector`{.docutils .literal
.notranslate}]{.pre}, [`edem.observable`{.docutils .literal
.notranslate}]{.pre}, and [`edem.success`{.docutils .literal
.notranslate}]{.pre}. It retains all four in its type table, including
unused types. A presentation named [`P`{.docutils .literal
.notranslate}]{.pre} creates the three-wire interface type
[`edem.code/P`{.docutils .literal .notranslate}]{.pre}. The separate,
name-sorted presentation table records [`s`{.docutils .literal
.notranslate}]{.pre}, [`v`{.docutils .literal .notranslate}]{.pre} and
[`k`{.docutils .literal .notranslate}]{.pre} explicitly. Names are
unique and identify a presentation; equal names must have equal counts.
A packed EDEM retains exactly the presentations its code bundles use.
Every bundle using that code type must have the recorded
[`(v,s,2k)`{.docutils .literal .notranslate}]{.pre} widths. Presentation
names identify generator ordering and logical representatives; producers
supply their code algebra and coefficients.

A description carries its presentations in [`code_table`{.docutils
.literal .notranslate}]{.pre}, and code bundles name them by index into
that array. Construction checks the whole table: names are nonempty
UTF-8 and unique.

[`cudaq_edem_shape`{.docutils .literal .notranslate}]{.pre} holds only
port counts, with [`input_code_count`{.docutils .literal
.notranslate}]{.pre} and [`output_code_count`{.docutils .literal
.notranslate}]{.pre} counting code bundles. [`block_count`{.docutils
.literal .notranslate}]{.pre} sits beside [`blocks`{.docutils .literal
.notranslate}]{.pre} ([`cudaq_edem_matrix_block`{.docutils .literal
.notranslate}]{.pre}) in the description. On a checked value,
[`cudaq_edem_get_code_count`{.docutils .literal .notranslate}]{.pre}
returns the number of retained presentations and the GF2 map shape
returns the packed matrix block count.

GF2 owns type/bundle layout, block validation, canonical storage, matrix
interning, packing, reading and apply. cudaq-edem checks its vocabulary,
permitted input/output roles, presentation consistency, and structural
zeros from V_in to S_out and L_out. Fault inputs have one distribution
per bundle. Exact independent and mutually exclusive groups cover
original indicators once; separate bundles retain separate laws. Equal
matrix blocks never identify faults. Original indicators are used
directly as fault coordinates.

The component consumes caller-provided coefficients and does not perform
circuit compilation. The open-round fixtures exercise this interface
with explicit coefficients, including boundary inputs and faults that
survive at the output.
:::

::: {#fault-distributions .section}
## Fault distributions[¶](#fault-distributions "Permalink to this heading"){.headerlink}

Each probability must be finite and within [`[0,`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal .notranslate}[`1]`{.docutils
.literal .notranslate}]{.pre}, with no allowance on individual bounds.
Exclusive-group totals use compensated binary64 summation and must not
exceed [`1`{.docutils .literal .notranslate}]{.pre}` `{.docutils
.literal .notranslate}[`+`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`CUDAQ_EDEM_EXCLUSIVE_TOTAL_TOLERANCE`{.docutils .literal
.notranslate}]{.pre}. The named constant is [`1e-6`{.docutils .literal
.notranslate}]{.pre}, a compatibility allowance on the total.
Construction and packed-value validation use the same check. Independent
probabilities have no group-total bound.

Packing preserves the supplied probability bits except for normalization
of negative zero to positive zero. It performs no probability
renormalization. Packed-value validation requires positive zero,
preserving canonical bytes.
:::

::: {#python-construction .section}
## Python construction[¶](#python-construction "Permalink to this heading"){.headerlink}

This open boundary-transfer example preserves the incoming coordinates
and adds one late data fault to the outgoing syndrome:

::: {.highlight-python .notranslate}
::: highlight
    from cudaq_edem import (
        CodePresentation, EDEM, Endpoint, FaultDistribution, MatrixBlock, PortKind,
        SparseMatrix,
    )

    identity = SparseMatrix(2, 2, [(0, 0), (1, 1)])
    model = EDEM(
        codes=[CodePresentation("repetition-3", syndrome_count=2,
                                virtual_outcome_count=2, logical_qubit_count=1)],
        inputs=[0], outputs=[0],
        faults=[FaultDistribution.independent([0.01])],
        blocks=[
            MatrixBlock(Endpoint(kind), Endpoint(kind), identity)
            for kind in (PortKind.VIRTUAL, PortKind.SYNDROME, PortKind.LOGICAL)
        ] + [
            MatrixBlock(Endpoint(PortKind.SYNDROME), Endpoint(PortKind.FAULT),
                        SparseMatrix(2, 1, [(0, 0)])),
        ],
    )
    assert model.apply([1, 0, 0, 0, 0, 0, 0]) == [0, 0, 1, 0, 0, 0]
    assert EDEM.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
:::
:::

For constructor-created models, input vector order is F ports followed
by [`(V,S,L)`{.docutils .literal .notranslate}]{.pre} for each input
code bundle; output order is D, O, Success ports followed by
[`(V,S,L)`{.docutils .literal .notranslate}]{.pre} for each output code
bundle. Input ports thus partition the columns of the EDEM matrix and
output ports partition its rows. Values loaded from packed bytes retain
the GF2 map's bundle order. An endpoint and an endpoint direction
describe a port. [`model.port(endpoint)`{.docutils .literal
.notranslate}]{.pre} returns an input port's width and offset; pass
[`direction=EndpointDirection.OUTPUT`{.docutils .literal
.notranslate}]{.pre} for output ports.

Properties return independent metadata/matrix copies. Reconstructing a
loaded value from these properties uses the constructor's bundle order.
Use [`to_bytes()`{.docutils .literal .notranslate}]{.pre} and
[`from_bytes()`{.docutils .literal .notranslate}]{.pre} to preserve a
loaded layout. Python loading copies bytes once into aligned owner
storage and validates them before publication.

Native producers can instead return a single-use capsule following
[`cudaq/edem/python.h`{.docutils .literal .notranslate}]{.pre},
containing a stable [`cudaq_gf2_owned_buffer`{.docutils .literal
.notranslate}]{.pre} record:

::: {.highlight-python .notranslate}
::: highlight
    model = EDEM.adopt(producer_capsule)
:::
:::

Adoption validates the bytes before consuming ownership and keeps their
original address. It retains the producer capsule so callback code and
contexts can stay alive through final release. Reusing a consumed
capsule is an error; failed validation leaves it available to its
producer. Native producers can transfer constructed results without
intermediate strings or byte copies. [`from_bytes`{.docutils .literal
.notranslate}]{.pre} remains the convenience entry point for serialized
input.
:::

::: {#packed-format .section}
## Packed format[¶](#packed-format "Permalink to this heading"){.headerlink}

The v0.x storage schemas are declared in [`cudaq/gf2/format.h`{.docutils
.literal .notranslate}]{.pre} and [`cudaq/edem/format.h`{.docutils
.literal .notranslate}]{.pre}. These schemas are unstable during v0.x
development. Numeric ABI and format tags identify concrete development
revisions, not major releases. Multibyte integers and IEEE binary64
probabilities are little-endian; indices are 32-bit, and counts and
offsets are 64-bit.

The common 64-byte header contains magic [`CQGF2\0\0\0`{.docutils
.literal .notranslate}]{.pre}, format revision, value kind, total size,
checked alignment and index-width constants, directory offset, entry
count, directory byte length, and a zero reserved field. The implemented
kind is a linear map. An EDEM is that map with presentation and
distribution sections, in the same buffer.

The directory is at the end. Each 32-byte entry records a unique section
ID, required/optional flags, element count, payload byte length and
payload offset. Entries sort by ID. Required IDs are below
[`0x80000000`{.docutils .literal .notranslate}]{.pre}; optional IDs are
at or above it, and flags must agree. Core map IDs are reserved below
[`0x10000`{.docutils .literal .notranslate}]{.pre}. Extension schemas
start at [`0x10000`{.docutils .literal .notranslate}]{.pre}. The five
required map sections are:

  ID                                                   Payload
  ---------------------------------------------------- --------------------------------------------------------
  1                                                    Types and their UTF-8 names
  2                                                    Input bundles, wire widths and vector offsets
  3                                                    Output bundles, wire widths and vector offsets
  4                                                    Sparse block bindings
  5                                                    Interned CSC matrices
  [`0x10000`{.docutils .literal .notranslate}]{.pre}   EDEM presentation table: type ID, s, v, k
  [`0x10001`{.docutils .literal .notranslate}]{.pre}   EDEM distributions associated with input fault bundles

Required payloads follow the header, optional payloads follow those, and
the directory comes last. Nonempty payloads occur in directory order at
aligned, non-overlapping offsets. Empty payloads have offset zero. There
are no unused ranges other than alignment padding; all padding and
reserved fields are zero. The directory's byte extent ends exactly at
total size.

Alignment is the fixed format constant of eight bytes. It is checked in
the header and is not a packing or sizing input. Python owners need only
meet that constant. Nested arrays use schema-defined natural alignment:
eight for 64-bit arrays and records, four for 32-bit arrays, and one for
strings. Every reference within a section is relative to that section's
start. Changing the alignment requires a format revision.

Type names sort by UTF-8 bytes and local IDs are remapped. Bundle order
is preserved. Blocks sort by output bundle/wire, then input bundle/wire.
Duplicate endpoint pairs are rejected, zero blocks are omitted, and
equal matrices are stored once in first-use order. Packed wire offsets
are checked against widths. Fault groups sort by kind and first
indicator, with sorted members and positive zero probabilities.
cudaq-edem checks their exact coverage with caller scratch.

Appending optional metadata preserves every required payload's bytes and
offset: copy the required region, emit optional payloads in ID order,
write a new trailing directory and update header layout fields. Unknown
required sections are unsupported; unknown optional sections can be
skipped after their bounds and padding are checked. Optional metadata
does not change the map's meaning.
:::

::: {#serial-composition-and-dem-projection .section}
## Serial composition and DEM projection[¶](#serial-composition-and-dem-projection "Permalink to this heading"){.headerlink}

[`cudaq/edem/composition.h`{.docutils .literal .notranslate}]{.pre}
provides [`cudaq_edem_compose_chain`{.docutils .literal
.notranslate}]{.pre}, [`cudaq_edem_is_dem`{.docutils .literal
.notranslate}]{.pre}, and [`cudaq_edem_project_dem`{.docutils .literal
.notranslate}]{.pre}. Serial composition connects every output code
bundle to the same ordinal input bundle of the next occurrence. Counts
and code presentations (name, [`s`{.docutils .literal
.notranslate}]{.pre}, [`v`{.docutils .literal .notranslate}]{.pre} and
[`k`{.docutils .literal .notranslate}]{.pre}) must match. The first
input and last output boundaries remain exposed. Repeated definitions
create distinct occurrence fault ports with their original indicators
and exact independent/exclusive distributions.

The C++17 conveniences in [`cudaq/edem/composition.hpp`{.docutils
.literal .notranslate}]{.pre} use [`PackedEDEM`{.docutils .literal
.notranslate}]{.pre} and return C status codes. For supplied native
values:

::: {.highlight-cpp .notranslate}
::: highlight
    cudaq_edem_view occurrences[] = {
        prep.view(), sec.view(), sec.view(), sec.view(), mz.view()};
    cudaq::edem::PackedEDEM composed, dem;
    auto status = cudaq::edem::compose_chain(occurrences, 5, composed);
    if (status != CUDAQ_EDEM_OK)
      return status;
    return cudaq::edem::project_DEM(composed, dem);
:::
:::

Python calls the same native operations and returns owning values:

::: {.highlight-python .notranslate}
::: highlight
    from cudaq_edem import compose_chain, project_DEM

    composed = compose_chain([prep, sec, sec, sec, mz])
    dem = project_DEM(composed)
:::
:::

Closure is structural, including zero-width code bundles. Success
outputs and exclusive groups do not prevent closure. Strict projection
rejects either exposed boundary. The semantic format already uses
original indicator coordinates, so projection copies the closed packed
EDEM into an independently owned, checked DEM value without changing its
basis or fault laws. Detector, observable and success matrices remain
its fault-to-output blocks. Ports retain occurrence/local-port order;
consumers concatenate columns by fault port and then original indicator.
Projection does not convert exclusive laws to independent errors or
serialize a Stim detector error model.

The serial composer evaluates basis columns through native GF2 typed-map
apply and materializes the complete result. Its cost grows with the
number of source columns and downstream maps. It is an offline reference
operation for serial chains; it does not provide general diagrams or
incremental window updates. Operations retain no state or caches.
Scratch and results use the same per-call memory provider as
construction.
:::

::: {#exact-noise-conversion-and-column-coalescing .section}
## Exact noise conversion and column coalescing[¶](#exact-noise-conversion-and-column-coalescing "Permalink to this heading"){.headerlink}

Noise transformations are separate native operations in
[`cudaq/edem/noise.h`{.docutils .literal .notranslate}]{.pre}. They
return independently owned DEMs and leave their inputs unchanged:

::: {.highlight-python .notranslate}
::: highlight
    from cudaq_edem import factorize_dem, coalesce_dem_columns

    independent = factorize_dem(dem)
    reduced, source_to_result = coalesce_dem_columns(independent)
:::
:::

[`cudaq_edem_into_independent_faults_dem`{.docutils .literal
.notranslate}]{.pre} preserves response columns and ports. It keeps
independent probabilities and exactly factors supported exclusive
channels into independent mechanisms. A supported exclusive group has 3
or 15 equally probable alternatives whose complete response columns,
including identity, form a uniform linear image of the one- or two-qubit
Pauli group. The total probability must be at most [`3/4`{.docutils
.literal .notranslate}]{.pre} or [`15/16`{.docutils .literal
.notranslate}]{.pre}, respectively. Unsupported groups return
[`CUDAQ_EDEM_UNSUPPORTED_DISTRIBUTION`{.docutils .literal
.notranslate}]{.pre}; conversion never approximates or renormalizes
their laws. These conditions are checked from the response columns,
including zero responses and collisions. The joint detector, observable,
and success distribution is preserved. Converted coordinates label
independent mechanisms with the same responses, rather than the original
physical events.

[`cudaq_edem_coalesce_dem_columns`{.docutils .literal
.notranslate}]{.pre} requires independent groups. It merges equal
complete responses using the parity probability [`p`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal .notranslate}[`+`{.docutils
.literal .notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`q`{.docutils .literal .notranslate}]{.pre}` `{.docutils
.literal .notranslate}[`-`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`2*p*q`{.docutils .literal .notranslate}]{.pre}, evaluated
stably for small probabilities and probabilities above one half. It
discards zero responses and zero-probability combined mechanisms. The
result has one fault port and at most one port for each output kind;
global rows retain detector, observable, then success order, each in
original port/row order. Distinct columns sort lexicographically by
their sorted global row IDs. Per-kind row totals and the result column
count must fit [`uint32_t`{.docutils .literal .notranslate}]{.pre}.

The C operation also returns an owned [`uint64_t`{.docutils .literal
.notranslate}]{.pre} array mapping each input response column to its
result column. [`UINT64_MAX`{.docutils .literal .notranslate}]{.pre}
denotes a discarded mechanism. Python returns the mapping as a list,
using [`None`{.docutils .literal .notranslate}]{.pre} for discarded
mechanisms. Factorization preserves column coordinates, so this mapping
can also refer back to the original DEM's response columns. It does not
map sampled independent events back to categorical physical events;
retain the original DEM for its exact source laws and occurrence
bindings.

The C++17 helpers [`factorize_dem`{.docutils .literal
.notranslate}]{.pre} and [`coalesce_dem_columns`{.docutils .literal
.notranslate}]{.pre} in [`cudaq/edem/noise.hpp`{.docutils .literal
.notranslate}]{.pre} use [`PackedEDEM`{.docutils .literal
.notranslate}]{.pre} and the move-only [`ColumnMapping`{.docutils
.literal .notranslate}]{.pre} owner, returning C status codes without
exceptions. Both operations use the same per-call scratch/result
resources as construction. Coalescing uses sparse response columns and
sorting, with memory linear in columns and nonzeros. Formatting adapters
consume the transformed DEM without changing probabilities, merging
columns, or introducing a simulator dependency into the library.
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](../versions.html "CUDA-Q Versions"){.btn .btn-neutral
.float-left accesskey="p" rel="prev"}
:::

------------------------------------------------------------------------

::: {role="contentinfo"}
© Copyright 2026, NVIDIA Corporation & Affiliates.
:::

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme) provided by
[Read the Docs](https://readthedocs.org).
:::
:::
:::
:::
