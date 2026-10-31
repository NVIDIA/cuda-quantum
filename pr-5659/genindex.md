::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-5659
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](using/quick_start.html){.reference .internal}
    -   [Install
        CUDA-Q](using/quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](using/quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q
        Academic](using/quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](using/basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q
        Kernel?](using/basics/kernel_intro.html){.reference .internal}
    -   [Building your first CUDA-Q
        Program](using/basics/build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q
        Program](using/basics/run_kernel.html){.reference .internal}
        -   [Sample](using/basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](using/basics/run_kernel.html#run){.reference
            .internal}
        -   [Observe](using/basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](using/basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](using/basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](using/basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
        -   [Python
            Stack-Traces](using/basics/troubleshooting.html#python-stack-traces){.reference
            .internal}
        -   [LLVM-based Python
            packages](using/basics/troubleshooting.html#llvm-based-python-packages){.reference
            .internal}
-   [Examples](using/examples/examples.html){.reference .internal}
    -   [Introduction](using/examples/introduction.html){.reference
        .internal}
    -   [Building
        Kernels](using/examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](using/examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](using/examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](using/examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](using/examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](using/examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](using/examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](using/examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](using/examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](using/examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](using/examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](using/examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](using/examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](using/examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring
        Kernels](using/examples/measuring_kernels.html){.reference
        .internal}
        -   [Measurement
            Handles](using/examples/measuring_kernels.html#measurement-handles){.reference
            .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](using/examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing
        Kernels](using/examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](using/examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](using/examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](using/examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](using/examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](using/examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](using/examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](using/examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](using/examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](using/examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](using/examples/expectation_values.html){.reference
        .internal}
        -   [Parallelizing across Multiple
            Processors](using/examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](using/examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](using/examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](using/examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](using/examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](using/examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](using/examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
    -   [Optimizers &
        Gradients](examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [CUDA-Q Optimizer
            Overview](examples/python/optimizers_gradients.html#CUDA-Q-Optimizer-Overview){.reference
            .internal}
            -   [Gradient-Free Optimizers (no gradients
                required):](examples/python/optimizers_gradients.html#Gradient-Free-Optimizers-(no-gradients-required):){.reference
                .internal}
            -   [Gradient-Based Optimizers (require
                gradients):](examples/python/optimizers_gradients.html#Gradient-Based-Optimizers-(require-gradients):){.reference
                .internal}
        -   [1. Built-in CUDA-Q Optimizers and
            Gradients](examples/python/optimizers_gradients.html#1.-Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
            -   [1.1 Adam Optimizer with Parameter
                Configuration](examples/python/optimizers_gradients.html#1.1-Adam-Optimizer-with-Parameter-Configuration){.reference
                .internal}
            -   [1.2 SGD (Stochastic Gradient Descent)
                Optimizer](examples/python/optimizers_gradients.html#1.2-SGD-(Stochastic-Gradient-Descent)-Optimizer){.reference
                .internal}
            -   [1.3 SPSA (Simultaneous Perturbation Stochastic
                Approximation)](examples/python/optimizers_gradients.html#1.3-SPSA-(Simultaneous-Perturbation-Stochastic-Approximation)){.reference
                .internal}
        -   [2. Third-Party
            Optimizers](examples/python/optimizers_gradients.html#2.-Third-Party-Optimizers){.reference
            .internal}
        -   [3. Parallel Parameter Shift
            Gradients](examples/python/optimizers_gradients.html#3.-Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](examples/python/noisy_simulations.html){.reference
        .internal}
    -   [Pre-Trajectory Sampling with Batch
        Execution](using/examples/ptsbe.html){.reference .internal}
        -   [Conceptual
            Overview](using/examples/ptsbe.html#conceptual-overview){.reference
            .internal}
        -   [When to Use
            PTSBE](using/examples/ptsbe.html#when-to-use-ptsbe){.reference
            .internal}
        -   [Quick
            Start](using/examples/ptsbe.html#quick-start){.reference
            .internal}
        -   [Usage
            Tutorial](using/examples/ptsbe.html#usage-tutorial){.reference
            .internal}
            -   [Controlling the Number of
                Trajectories](using/examples/ptsbe.html#controlling-the-number-of-trajectories){.reference
                .internal}
            -   [Choosing a Trajectory Sampling
                Strategy](using/examples/ptsbe.html#choosing-a-trajectory-sampling-strategy){.reference
                .internal}
            -   [Shot Allocation
                Strategies](using/examples/ptsbe.html#shot-allocation-strategies){.reference
                .internal}
            -   [Inspecting Execution
                Data](using/examples/ptsbe.html#inspecting-execution-data){.reference
                .internal}
    -   [Detector Error
        Models](using/examples/dem_from_kernel.html){.reference
        .internal}
        -   [DEM
            Options](using/examples/dem_from_kernel.html#dem-options){.reference
            .internal}
        -   [Measurement
            Matrices](using/examples/dem_from_kernel.html#measurement-matrices){.reference
            .internal}
        -   [Limitations](using/examples/dem_from_kernel.html#limitations){.reference
            .internal}
    -   [Rotation Synthesis
        (Clifford+T)](using/examples/rotation_synthesis.html){.reference
        .internal}
        -   [Synthesizing a
            rotation](using/examples/rotation_synthesis.html#synthesizing-a-rotation){.reference
            .internal}
        -   [Estimating the T count of a
            kernel](using/examples/rotation_synthesis.html#estimating-the-t-count-of-a-kernel){.reference
            .internal}
        -   [Choosing
            epsilon](using/examples/rotation_synthesis.html#choosing-epsilon){.reference
            .internal}
        -   [Dependencies](using/examples/rotation_synthesis.html#dependencies){.reference
            .internal}
    -   [Constructing
        Operators](using/examples/operators.html){.reference .internal}
        -   [Constructing Spin
            Operators](using/examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](using/examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](using/examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](using/examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](using/examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](using/examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](using/examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](using/examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](using/examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](using/examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](using/examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [qBraid](using/examples/hardware_providers.html#qbraid){.reference
            .internal}
        -   [Quantinuum](using/examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum
            Machines](using/examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](using/examples/hardware_providers.html#quera-computing){.reference
            .internal}
        -   [Scaleway](using/examples/hardware_providers.html#scaleway){.reference
            .internal}
        -   [TII](using/examples/hardware_providers.html#tii){.reference
            .internal}
    -   [When to Use sample vs.
        run](using/examples/sample_vs_run.html){.reference .internal}
        -   [Introduction](using/examples/sample_vs_run.html#introduction){.reference
            .internal}
        -   [Usage
            Guidelines](using/examples/sample_vs_run.html#usage-guidelines){.reference
            .internal}
        -   [What Is Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](using/examples/sample_vs_run.html#what-is-supported-with-sample){.reference
            .internal}
        -   [What Is Not Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](using/examples/sample_vs_run.html#what-is-not-supported-with-sample){.reference
            .internal}
        -   [How to
            Migrate](using/examples/sample_vs_run.html#how-to-migrate){.reference
            .internal}
            -   [Step 1: Add a return type to the
                kernel](using/examples/sample_vs_run.html#step-1-add-a-return-type-to-the-kernel){.reference
                .internal}
            -   [Step 2: Replace [`sample`{.docutils .literal
                .notranslate}]{.pre} with [`run`{.docutils .literal
                .notranslate}]{.pre}](using/examples/sample_vs_run.html#step-2-replace-sample-with-run){.reference
                .internal}
            -   [Step 3: Update result
                processing](using/examples/sample_vs_run.html#step-3-update-result-processing){.reference
                .internal}
        -   [Migration
            Examples](using/examples/sample_vs_run.html#migration-examples){.reference
            .internal}
            -   [Example 1: Simple conditional
                logic](using/examples/sample_vs_run.html#example-1-simple-conditional-logic){.reference
                .internal}
            -   [Example 2: Returning multiple measurement
                results](using/examples/sample_vs_run.html#example-2-returning-multiple-measurement-results){.reference
                .internal}
            -   [Example 3: Quantum
                teleportation](using/examples/sample_vs_run.html#example-3-quantum-teleportation){.reference
                .internal}
        -   [Additional
            Notes](using/examples/sample_vs_run.html#additional-notes){.reference
            .internal}
    -   [Dynamics
        Examples](using/examples/dynamics_examples.html){.reference
        .internal}
        -   [Python Examples (Jupyter
            Notebooks)](using/examples/dynamics_examples.html#python-examples-jupyter-notebooks){.reference
            .internal}
            -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
                Model)](examples/python/dynamics/dynamics_intro_1.html){.reference
                .internal}
            -   [Introduction to CUDA-Q Dynamics (Time Dependent
                Hamiltonians)](examples/python/dynamics/dynamics_intro_2.html){.reference
                .internal}
            -   [Superconducting
                Qubits](examples/python/dynamics/superconducting.html){.reference
                .internal}
            -   [Spin
                Qubits](examples/python/dynamics/spinqubits.html){.reference
                .internal}
            -   [Trapped Ion
                Qubits](examples/python/dynamics/iontrap.html){.reference
                .internal}
            -   [Control](examples/python/dynamics/control.html){.reference
                .internal}
        -   [C++
            Examples](using/examples/dynamics_examples.html#c-examples){.reference
            .internal}
            -   [Introduction: Single Qubit
                Dynamics](using/examples/dynamics_examples.html#introduction-single-qubit-dynamics){.reference
                .internal}
            -   [Introduction: Cavity QED (Jaynes-Cummings
                Model)](using/examples/dynamics_examples.html#introduction-cavity-qed-jaynes-cummings-model){.reference
                .internal}
            -   [Superconducting Qubits: Cross-Resonance
                Gate](using/examples/dynamics_examples.html#superconducting-qubits-cross-resonance-gate){.reference
                .internal}
            -   [Spin Qubits: Heisenberg Spin
                Chain](using/examples/dynamics_examples.html#spin-qubits-heisenberg-spin-chain){.reference
                .internal}
            -   [Control: Driven
                Qubit](using/examples/dynamics_examples.html#control-driven-qubit){.reference
                .internal}
            -   [State
                Batching](using/examples/dynamics_examples.html#state-batching){.reference
                .internal}
            -   [Numerical
                Integrators](using/examples/dynamics_examples.html#numerical-integrators){.reference
                .internal}
-   [Applications](using/applications.html){.reference .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](applications/python/krylov.html){.reference .internal}
        -   [Setup](applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Quantum-Selected Configuration Interaction
        (QSCI)](applications/python/qsci.html){.reference .internal}
        -   [0. Problem
            definition](applications/python/qsci.html#0.-Problem-definition){.reference
            .internal}
        -   [1. Prepare an Approximate Quantum
            State](applications/python/qsci.html#1.-Prepare-an-Approximate-Quantum-State){.reference
            .internal}
        -   [2 Quantum Sampling to Select
            Configuration](applications/python/qsci.html#2-Quantum-Sampling-to-Select-Configuration){.reference
            .internal}
        -   [3. Classical Diagonalization on the Selected
            Subspace](applications/python/qsci.html#3.-Classical-Diagonalization-on-the-Selected-Subspace){.reference
            .internal}
        -   [5. Compare
            results](applications/python/qsci.html#5.-Compare-results){.reference
            .internal}
        -   [Reference](applications/python/qsci.html#Reference){.reference
            .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using [`Sample`{.docutils .literal .notranslate}]{.pre} to
            perform the Hadamard
            test](applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. [`Compute`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`overlap`{.docutils .literal
                .notranslate}]{.pre}](applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum
        Volume](applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](applications/python/afqmc.html){.reference .internal}
        -   [Hamiltonian preparation for
            VQE](applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [Factoring Integers With Shor's
        Algorithm](applications/python/shors.html){.reference .internal}
        -   [Shor's
            algorithm](applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
            -   [(a) Generate the molecular Hamiltonian using Restricted
                Hartree Fock molecular
                orbitals](applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Restricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(b) Generate the molecular Hamiltonian using
                Unrestricted Hartree Fock molecular
                orbitals](applications/python/generate_fermionic_ham.html#(b)-Generate-the-molecular-Hamiltonian-using-Unrestricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(a) Generate the active space hamiltonian using RHF
                molecular
                orbitals.](applications/python/generate_fermionic_ham.html#(a)-Generate-the-active-space-hamiltonian-using-RHF-molecular-orbitals.){.reference
                .internal}
            -   [(b) Generate the active space Hamiltonian using the
                natural orbitals computed from MP2
                simulation](applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
                .internal}
            -   [(c) Generate the active space Hamiltonian computed from
                the CASSCF molecular
                orbitals](applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
                .internal}
            -   [(d) Generate the electronic Hamiltonian using
                ROHF](applications/python/generate_fermionic_ham.html#(d)-Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
            -   [(e) Generate electronic Hamiltonian using
                UHF](applications/python/generate_fermionic_ham.html#(e)-Generate-electronic-Hamiltonian-using-UHF){.reference
                .internal}
    -   [The UCCSD Wavefunction
        ansatz](applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [Sample-Based Krylov Quantum Diagonalization
        (SKQD)](applications/python/skqd.html){.reference .internal}
        -   [Why
            SKQD?](applications/python/skqd.html#Why-SKQD?){.reference
            .internal}
        -   [Understanding Krylov
            Subspaces](applications/python/skqd.html#Understanding-Krylov-Subspaces){.reference
            .internal}
            -   [What is a Krylov
                Subspace?](applications/python/skqd.html#What-is-a-Krylov-Subspace?){.reference
                .internal}
            -   [The SKQD
                Algorithm](applications/python/skqd.html#The-SKQD-Algorithm){.reference
                .internal}
        -   [Problem Setup: 22-Qubit Heisenberg
            Model](applications/python/skqd.html#Problem-Setup:-22-Qubit-Heisenberg-Model){.reference
            .internal}
        -   [Krylov State Generation via Repeated
            Evolution](applications/python/skqd.html#Krylov-State-Generation-via-Repeated-Evolution){.reference
            .internal}
        -   [Quantum Measurements and
            Sampling](applications/python/skqd.html#Quantum-Measurements-and-Sampling){.reference
            .internal}
            -   [The Sampling
                Process](applications/python/skqd.html#The-Sampling-Process){.reference
                .internal}
        -   [Classical Post-Processing and
            Diagonalization](applications/python/skqd.html#Classical-Post-Processing-and-Diagonalization){.reference
            .internal}
            -   [Matrix Construction
                Details](applications/python/skqd.html#Matrix-Construction-Details){.reference
                .internal}
            -   [Approach 1: GPU-Vectorized CSR Sparse
                Matrix](applications/python/skqd.html#Approach-1:-GPU-Vectorized-CSR-Sparse-Matrix){.reference
                .internal}
            -   [Approach 2: Matrix-Free Lanczos via
                [`distributed_eigsh`{.docutils .literal
                .notranslate}]{.pre}](applications/python/skqd.html#Approach-2:-Matrix-Free-Lanczos-via-distributed_eigsh){.reference
                .internal}
        -   [Results Analysis and
            Convergence](applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [Postprocessing Acceleration: CSR matrix approach, single
            GPU vs
            CPU](applications/python/skqd.html#Postprocessing-Acceleration:-CSR-matrix-approach,-single-GPU-vs-CPU){.reference
            .internal}
        -   [Postprocessing Scale-Up and Scale-Out: Linear Operator
            Approach, Multi-GPU
            Multi-Node](applications/python/skqd.html#Postprocessing-Scale-Up-and-Scale-Out:-Linear-Operator-Approach,-Multi-GPU-Multi-Node){.reference
            .internal}
            -   [Saving Hamiltonian
                Data](applications/python/skqd.html#Saving-Hamiltonian-Data){.reference
                .internal}
            -   [Running the Distributed
                Solver](applications/python/skqd.html#Running-the-Distributed-Solver){.reference
                .internal}
        -   [Summary](applications/python/skqd.html#Summary){.reference
            .internal}
    -   [Entanglement Accelerates Quantum
        Simulation](applications/python/entanglement_acc_hamiltonian_simulation.html){.reference
        .internal}
        -   [2. Model
            Definition](applications/python/entanglement_acc_hamiltonian_simulation.html#2.-Model-Definition){.reference
            .internal}
            -   [2.1 Initial product
                state](applications/python/entanglement_acc_hamiltonian_simulation.html#2.1-Initial-product-state){.reference
                .internal}
            -   [2.2 QIMF
                Hamiltonian](applications/python/entanglement_acc_hamiltonian_simulation.html#2.2-QIMF-Hamiltonian){.reference
                .internal}
            -   [2.3 First-Order Trotter Formula
                (PF1)](applications/python/entanglement_acc_hamiltonian_simulation.html#2.3-First-Order-Trotter-Formula-(PF1)){.reference
                .internal}
            -   [2.4 PF1 step for the QIMF
                partition](applications/python/entanglement_acc_hamiltonian_simulation.html#2.4-PF1-step-for-the-QIMF-partition){.reference
                .internal}
            -   [2.5 Hamiltonian
                helpers](applications/python/entanglement_acc_hamiltonian_simulation.html#2.5-Hamiltonian-helpers){.reference
                .internal}
        -   [3. Entanglement
            metrics](applications/python/entanglement_acc_hamiltonian_simulation.html#3.-Entanglement-metrics){.reference
            .internal}
        -   [4. Simulation
            workflow](applications/python/entanglement_acc_hamiltonian_simulation.html#4.-Simulation-workflow){.reference
            .internal}
            -   [4.1 Single-step Trotter
                error](applications/python/entanglement_acc_hamiltonian_simulation.html#4.1-Single-step-Trotter-error){.reference
                .internal}
            -   [4.2 Dual trajectory
                update](applications/python/entanglement_acc_hamiltonian_simulation.html#4.2-Dual-trajectory-update){.reference
                .internal}
        -   [5. Reproducing the paper's Figure
            1a](applications/python/entanglement_acc_hamiltonian_simulation.html#5.-Reproducing-the-paper’s-Figure-1a){.reference
            .internal}
            -   [5.1 Visualising the joint
                behaviour](applications/python/entanglement_acc_hamiltonian_simulation.html#5.1-Visualising-the-joint-behaviour){.reference
                .internal}
            -   [5.2 Interpreting the
                result](applications/python/entanglement_acc_hamiltonian_simulation.html#5.2-Interpreting-the-result){.reference
                .internal}
        -   [6. References and further
            reading](applications/python/entanglement_acc_hamiltonian_simulation.html#6.-References-and-further-reading){.reference
            .internal}
    -   [Pre-Trajectory Sampling with Batch Execution
        (PTSBE)](applications/python/ptsbe.html){.reference .internal}
        -   [Set up the
            environment](applications/python/ptsbe.html#Set-up-the-environment){.reference
            .internal}
        -   [Define the circuit and noise
            model](applications/python/ptsbe.html#Define-the-circuit-and-noise-model){.reference
            .internal}
            -   [Inline noise with [`apply_noise`{.docutils .literal
                .notranslate}]{.pre}](applications/python/ptsbe.html#Inline-noise-with-apply_noise){.reference
                .internal}
        -   [Run PTSBE
            sampling](applications/python/ptsbe.html#Run-PTSBE-sampling){.reference
            .internal}
            -   [Larger circuit for execution
                data](applications/python/ptsbe.html#Larger-circuit-for-execution-data){.reference
                .internal}
        -   [Inspecting trajectories with execution
            data](applications/python/ptsbe.html#Inspecting-trajectories-with-execution-data){.reference
            .internal}
        -   [Performance of PTSBE vs standard noisy
            sampling](applications/python/ptsbe.html#Performance-of-PTSBE-vs-standard-noisy-sampling){.reference
            .internal}
-   [Backends](using/backends/backends.html){.reference .internal}
    -   [Circuit Simulation](using/backends/simulators.html){.reference
        .internal}
        -   [State Vector
            Simulators](using/backends/sims/svsims.html){.reference
            .internal}
            -   [CPU](using/backends/sims/svsims.html#cpu){.reference
                .internal}
            -   [Single-GPU](using/backends/sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](using/backends/sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network
            Simulators](using/backends/sims/tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](using/backends/sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](using/backends/sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](using/backends/sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU
            Simulators](using/backends/sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](using/backends/sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU with Multi-Node Multi-GPU
                Backends](using/backends/sims/mqpusims.html#multi-qpu-with-multi-node-multi-gpu-backends){.reference
                .internal}
        -   [Noisy
            Simulators](using/backends/sims/noisy.html){.reference
            .internal}
            -   [Trajectory Noisy
                Simulation](using/backends/sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](using/backends/sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](using/backends/sims/noisy.html#stim){.reference
                .internal}
            -   [[`cuStabilizer`{.docutils .literal
                .notranslate}]{.pre}](using/backends/sims/noisy.html#custabilizer){.reference
                .internal}
        -   [Photonics
            Simulators](using/backends/sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](using/backends/sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware
        (QPUs)](using/backends/hardware.html){.reference .internal}
        -   [Ion Trap
            QPUs](using/backends/hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](using/backends/hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](using/backends/hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](using/backends/hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](using/backends/hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](using/backends/hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](using/backends/hardware/superconducting.html#oqc){.reference
                .internal}
            -   [TII](using/backends/hardware/superconducting.html#tii){.reference
                .internal}
        -   [Neutral Atom
            QPUs](using/backends/hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](using/backends/hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](using/backends/hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](using/backends/hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic
            QPUs](using/backends/hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](using/backends/hardware/photonic.html#orca-computing){.reference
                .internal}
        -   [Quantum Control
            Systems](using/backends/hardware/qcontrol.html){.reference
            .internal}
            -   [Quantum
                Machines](using/backends/hardware/qcontrol.html#quantum-machines){.reference
                .internal}
    -   [Dynamics
        Simulation](using/backends/dynamics_backends.html){.reference
        .internal}
    -   [Cloud](using/backends/cloud.html){.reference .internal}
        -   [Amazon Braket
            (braket)](using/backends/cloud/braket.html){.reference
            .internal}
            -   [Setting
                Credentials](using/backends/cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submitting](using/backends/cloud/braket.html#submitting){.reference
                .internal}
        -   [Scaleway QaaS
            (scaleway)](using/backends/cloud/scaleway.html){.reference
            .internal}
            -   [Setting
                Credentials](using/backends/cloud/scaleway.html#setting-credentials){.reference
                .internal}
            -   [Submitting](using/backends/cloud/scaleway.html#submitting){.reference
                .internal}
            -   [Manage your QPU
                session](using/backends/cloud/scaleway.html#manage-your-qpu-session){.reference
                .internal}
        -   [qBraid](using/backends/cloud/qbraid.html){.reference
            .internal}
            -   [Setting
                Credentials](using/backends/cloud/qbraid.html#setting-credentials){.reference
                .internal}
            -   [Submitting](using/backends/cloud/qbraid.html#submitting){.reference
                .internal}
-   [Dynamics](using/dynamics.html){.reference .internal}
    -   [Quick Start](using/dynamics.html#quick-start){.reference
        .internal}
    -   [Operator](using/dynamics.html#operator){.reference .internal}
    -   [Time-Dependent
        Dynamics](using/dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](using/dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](using/dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch
        simulation](using/dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](using/dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](using/dynamics.html#examples){.reference .internal}
-   [Realtime](using/realtime.html){.reference .internal}
    -   [Installation](using/realtime/installation.html){.reference
        .internal}
        -   [Prerequisites](using/realtime/installation.html#prerequisites){.reference
            .internal}
        -   [HSB FPGA IP core and RFSoC
            bit-file](using/realtime/installation.html#hsb-fpga-ip-core-and-rfsoc-bit-file){.reference
            .internal}
        -   [Setup](using/realtime/installation.html#setup){.reference
            .internal}
        -   [Latency
            Measurement](using/realtime/installation.html#latency-measurement){.reference
            .internal}
    -   [Host API](using/realtime/host.html){.reference .internal}
        -   [What is the
            GpuRoceTransceiver?](using/realtime/host.html#what-is-the-gpurocetransceiver){.reference
            .internal}
        -   [Transport
            Mechanisms](using/realtime/host.html#transport-mechanisms){.reference
            .internal}
            -   [Supported Transport
                Options](using/realtime/host.html#supported-transport-options){.reference
                .internal}
        -   [The 3-Kernel Architecture (GpuRoceTransceiver Example)
            {#three-kernel-architecture}](using/realtime/host.html#the-3-kernel-architecture-gpurocetransceiver-example-three-kernel-architecture){.reference
            .internal}
            -   [Data Flow
                Summary](using/realtime/host.html#data-flow-summary){.reference
                .internal}
            -   [Why 3
                Kernels?](using/realtime/host.html#why-3-kernels){.reference
                .internal}
        -   [Unified Dispatch
            Mode](using/realtime/host.html#unified-dispatch-mode){.reference
            .internal}
            -   [Architecture](using/realtime/host.html#architecture){.reference
                .internal}
            -   [Transport-Agnostic
                Design](using/realtime/host.html#transport-agnostic-design){.reference
                .internal}
            -   [When to Use Which
                Mode](using/realtime/host.html#when-to-use-which-mode){.reference
                .internal}
            -   [Host API
                Extensions](using/realtime/host.html#host-api-extensions){.reference
                .internal}
            -   [Wiring Example (Unified Mode with
                GpuRoceTransceiver)](using/realtime/host.html#wiring-example-unified-mode-with-gpurocetransceiver){.reference
                .internal}
        -   [What This API Does (In One
            Paragraph)](using/realtime/host.html#what-this-api-does-in-one-paragraph){.reference
            .internal}
        -   [Scope](using/realtime/host.html#scope){.reference
            .internal}
        -   [Terms and
            Components](using/realtime/host.html#terms-and-components){.reference
            .internal}
        -   [Schema Data
            Structures](using/realtime/host.html#schema-data-structures){.reference
            .internal}
            -   [Type
                Descriptors](using/realtime/host.html#type-descriptors){.reference
                .internal}
            -   [Handler
                Schema](using/realtime/host.html#handler-schema){.reference
                .internal}
        -   [RPC Messaging
            Protocol](using/realtime/host.html#rpc-messaging-protocol){.reference
            .internal}
        -   [Host API
            Overview](using/realtime/host.html#host-api-overview){.reference
            .internal}
        -   [Manager and Dispatcher
            Topology](using/realtime/host.html#manager-and-dispatcher-topology){.reference
            .internal}
        -   [Host API
            Functions](using/realtime/host.html#host-api-functions){.reference
            .internal}
            -   [Occupancy Query and Eager Module
                Loading](using/realtime/host.html#occupancy-query-and-eager-module-loading){.reference
                .internal}
            -   [Graph-Based Dispatch
                Functions](using/realtime/host.html#graph-based-dispatch-functions){.reference
                .internal}
            -   [Kernel Launch Helper
                Functions](using/realtime/host.html#kernel-launch-helper-functions){.reference
                .internal}
        -   [Memory Layout and Ring Buffer
            Wiring](using/realtime/host.html#memory-layout-and-ring-buffer-wiring){.reference
            .internal}
        -   [Step-by-Step: Wiring the Host API
            (Minimal)](using/realtime/host.html#step-by-step-wiring-the-host-api-minimal){.reference
            .internal}
        -   [Device Handler and Function
            ID](using/realtime/host.html#device-handler-and-function-id){.reference
            .internal}
            -   [Multi-Argument Handler
                Example](using/realtime/host.html#multi-argument-handler-example){.reference
                .internal}
        -   [CUDA Graph Dispatch
            Mode](using/realtime/host.html#cuda-graph-dispatch-mode){.reference
            .internal}
            -   [Requirements](using/realtime/host.html#requirements){.reference
                .internal}
            -   [Graph-Based Dispatch
                API](using/realtime/host.html#graph-based-dispatch-api){.reference
                .internal}
            -   [Graph Handler Setup
                Example](using/realtime/host.html#graph-handler-setup-example){.reference
                .internal}
            -   [Graph Capture and
                Instantiation](using/realtime/host.html#graph-capture-and-instantiation){.reference
                .internal}
            -   [When to Use Graph
                Dispatch](using/realtime/host.html#when-to-use-graph-dispatch){.reference
                .internal}
            -   [Graph vs Device Call
                Dispatch](using/realtime/host.html#graph-vs-device-call-dispatch){.reference
                .internal}
        -   [Building and Sending an RPC
            Message](using/realtime/host.html#building-and-sending-an-rpc-message){.reference
            .internal}
        -   [Reading the
            Response](using/realtime/host.html#reading-the-response){.reference
            .internal}
        -   [Schema-Driven Argument
            Parsing](using/realtime/host.html#schema-driven-argument-parsing){.reference
            .internal}
        -   [GpuRoceTransceiver 3-Kernel Workflow
            (Primary)](using/realtime/host.html#gpurocetransceiver-3-kernel-workflow-primary){.reference
            .internal}
        -   [NIC-Free Testing (No GpuRoceTransceiver / No
            ConnectX-7)](using/realtime/host.html#nic-free-testing-no-gpurocetransceiver-no-connectx-7){.reference
            .internal}
        -   [Troubleshooting](using/realtime/host.html#troubleshooting){.reference
            .internal}
    -   [Messaging Protocol](using/realtime/protocol.html){.reference
        .internal}
        -   [Scope](using/realtime/protocol.html#scope){.reference
            .internal}
        -   [RPC Header /
            Response](using/realtime/protocol.html#rpc-header-response){.reference
            .internal}
        -   [Request ID
            Semantics](using/realtime/protocol.html#request-id-semantics){.reference
            .internal}
        -   [[`PTP`{.docutils .literal .notranslate}]{.pre} Timestamp
            Semantics](using/realtime/protocol.html#ptp-timestamp-semantics){.reference
            .internal}
        -   [Function ID
            Semantics](using/realtime/protocol.html#function-id-semantics){.reference
            .internal}
        -   [Schema and Payload
            Interpretation](using/realtime/protocol.html#schema-and-payload-interpretation){.reference
            .internal}
            -   [Type
                System](using/realtime/protocol.html#type-system){.reference
                .internal}
        -   [Payload
            Encoding](using/realtime/protocol.html#payload-encoding){.reference
            .internal}
            -   [Single-Argument
                Payloads](using/realtime/protocol.html#single-argument-payloads){.reference
                .internal}
            -   [Multi-Argument
                Payloads](using/realtime/protocol.html#multi-argument-payloads){.reference
                .internal}
            -   [Size
                Constraints](using/realtime/protocol.html#size-constraints){.reference
                .internal}
            -   [Encoding
                Examples](using/realtime/protocol.html#encoding-examples){.reference
                .internal}
            -   [Bit-Packed Data
                Encoding](using/realtime/protocol.html#bit-packed-data-encoding){.reference
                .internal}
            -   [Multi-Bit Measurement
                Encoding](using/realtime/protocol.html#multi-bit-measurement-encoding){.reference
                .internal}
        -   [Response
            Encoding](using/realtime/protocol.html#response-encoding){.reference
            .internal}
            -   [Single-Result
                Response](using/realtime/protocol.html#single-result-response){.reference
                .internal}
            -   [Multi-Result
                Response](using/realtime/protocol.html#multi-result-response){.reference
                .internal}
            -   [Status
                Codes](using/realtime/protocol.html#status-codes){.reference
                .internal}
        -   [QEC-Specific Usage
            Example](using/realtime/protocol.html#qec-specific-usage-example){.reference
            .internal}
            -   [QEC
                Terminology](using/realtime/protocol.html#qec-terminology){.reference
                .internal}
            -   [QEC Decoder
                Handler](using/realtime/protocol.html#qec-decoder-handler){.reference
                .internal}
            -   [Decoding
                Rounds](using/realtime/protocol.html#decoding-rounds){.reference
                .internal}
    -   [CPU RoCE
        Transport](using/realtime/cpu_transport.html){.reference
        .internal}
        -   [C ABI](using/realtime/cpu_transport.html#c-abi){.reference
            .internal}
        -   [Two-phase bring-up ([`setup`{.docutils .literal
            .notranslate}]{.pre} / [`connect`{.docutils .literal
            .notranslate}]{.pre})](using/realtime/cpu_transport.html#two-phase-bring-up-setup-connect){.reference
            .internal}
        -   [TX
            modes](using/realtime/cpu_transport.html#tx-modes){.reference
            .internal}
        -   [Testing ([`hsb_bridge_cpu`{.docutils .literal
            .notranslate}]{.pre})](using/realtime/cpu_transport.html#testing-hsb-bridge-cpu){.reference
            .internal}
    -   [Device Call
        Channels](using/realtime/device_call.html){.reference .internal}
        -   [The [`device_call`{.docutils .literal .notranslate}]{.pre}
            model](using/realtime/device_call.html#the-device-call-model){.reference
            .internal}
        -   [Selecting a
            channel](using/realtime/device_call.html#selecting-a-channel){.reference
            .internal}
        -   [Extending an in-process
            service](using/realtime/device_call.html#extending-an-in-process-service){.reference
            .internal}
        -   [The [`cpu_roce`{.docutils .literal .notranslate}]{.pre}
            channel](using/realtime/device_call.html#the-cpu-roce-channel){.reference
            .internal}
            -   [Wire pattern
                (FPGA-compatible)](using/realtime/device_call.html#wire-pattern-fpga-compatible){.reference
                .internal}
            -   [Connection
                setup](using/realtime/device_call.html#connection-setup){.reference
                .internal}
            -   [Running
                it](using/realtime/device_call.html#running-it){.reference
                .internal}
            -   [Test
                harness](using/realtime/device_call.html#test-harness){.reference
                .internal}
-   [CUDA-Q QEC](using/cudaq-qec/cudaq-qec.html){.reference .internal}
-   [CUDA-Q
    Algorithms](using/cudaq-algorithms/cudaq-algorithms.html){.reference
    .internal}
-   [Installation](using/install/install.html){.reference .internal}
    -   [Local
        Installation](using/install/local_installation.html){.reference
        .internal}
        -   [Introduction](using/install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](using/install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](using/install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](using/install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](using/install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](using/install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](using/install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](using/install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](using/install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](using/install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](using/install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](using/install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](using/install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](using/install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](using/install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](using/install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](using/install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](using/install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](using/install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](using/install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](using/install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](using/install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](using/install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
            -   [Dynamic linking to GMP and
                MPFR](using/install/local_installation.html#dynamic-linking-to-gmp-and-mpfr){.reference
                .internal}
        -   [Next
            Steps](using/install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](using/install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](using/install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](using/install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](using/install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](using/install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](using/install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](using/install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](using/install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](using/install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](using/install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](using/install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](using/integration/integration.html){.reference
    .internal}
    -   [Downstream CMake
        Integration](using/integration/cmake_app.html){.reference
        .internal}
    -   [Combining CUDA with
        CUDA-Q](using/integration/cuda_gpu.html){.reference .internal}
    -   [Integrating with Third-Party
        Libraries](using/integration/libraries.html){.reference
        .internal}
        -   [Calling a CUDA-Q library from
            C++](using/integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](using/integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](using/integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](using/extending/extending.html){.reference .internal}
    -   [Compiler
        development](using/extending/compiler/index.html){.reference
        .internal}
        -   [Compiler
            IR](using/extending/compiler/cudaq_ir.html){.reference
            .internal}
            -   [CUDA-Q
                dialects](using/extending/compiler/cudaq_ir.html#cuda-q-dialects){.reference
                .internal}
            -   [Source and
                tests](using/extending/compiler/cudaq_ir.html#source-and-tests){.reference
                .internal}
        -   [Value-semantics IR change
            log](using/extending/compiler/ir_changelog.html){.reference
            .internal}
            -   [Producing this
                IR](using/extending/compiler/ir_changelog.html#producing-this-ir){.reference
                .internal}
            -   [Upcoming release ([`main`{.docutils .literal
                .notranslate}]{.pre}
                branch)](using/extending/compiler/ir_changelog.html#upcoming-release-main-branch){.reference
                .internal}
            -   [0.16.0](using/extending/compiler/ir_changelog.html#id5){.reference
                .internal}
        -   [External compiler pass
            plugins](using/extending/compiler/pass_plugins.html){.reference
            .internal}
            -   [Implement and register the
                pass](using/extending/compiler/pass_plugins.html#implement-and-register-the-pass){.reference
                .internal}
            -   [Build the
                plugin](using/extending/compiler/pass_plugins.html#build-the-plugin){.reference
                .internal}
            -   [Load and test the
                plugin](using/extending/compiler/pass_plugins.html#load-and-test-the-plugin){.reference
                .internal}
        -   [Using compile targets to customize
            compilation](using/extending/compiler/compile_targets.html){.reference
            .internal}
            -   [Constructing single-stage compile
                targets](using/extending/compiler/compile_targets.html#constructing-single-stage-compile-targets){.reference
                .internal}
            -   [Setting compile targets with [`set_target`{.docutils
                .literal
                .notranslate}]{.pre}](using/extending/compiler/compile_targets.html#setting-compile-targets-with-set-target){.reference
                .internal}
            -   [Example](using/extending/compiler/compile_targets.html#example){.reference
                .internal}
    -   [Add a hardware
        backend](using/extending/backend.html){.reference .internal}
        -   [Plugin Directory
            Structure](using/extending/backend.html#plugin-directory-structure){.reference
            .internal}
        -   [REST-Style Backends (Server
            Helper)](using/extending/backend.html#rest-style-backends-server-helper){.reference
            .internal}
            -   [Server Helper
                Class](using/extending/backend.html#server-helper-class){.reference
                .internal}
            -   [Target YAML
                Configuration](using/extending/backend.html#target-yaml-configuration){.reference
                .internal}
            -   [CMake Build
                File](using/extending/backend.html#cmake-build-file){.reference
                .internal}
        -   [Auxiliary Files and [`%PLUGIN_ROOT%`{.docutils .literal
            .notranslate}]{.pre}](using/extending/backend.html#auxiliary-files-and-plugin-root){.reference
            .internal}
        -   [Testing Your
            Backend](using/extending/backend.html#testing-your-backend){.reference
            .internal}
        -   [Example
            Usage](using/extending/backend.html#example-usage){.reference
            .internal}
        -   [Next
            Steps](using/extending/backend.html#next-steps){.reference
            .internal}
    -   [Package & distribute a backend
        plugin](using/extending/packaging.html){.reference .internal}
        -   [Plugin Package
            Layout](using/extending/packaging.html#plugin-package-layout){.reference
            .internal}
        -   [How to pre-compile a target config
            YAML](using/extending/packaging.html#how-to-pre-compile-a-target-config-yaml){.reference
            .internal}
        -   [Target YAML Reference (Plugin
            Fields)](using/extending/packaging.html#target-yaml-reference-plugin-fields){.reference
            .internal}
            -   [[`%PLUGIN_ROOT%`{.docutils .literal
                .notranslate}]{.pre}](using/extending/packaging.html#plugin-root){.reference
                .internal}
            -   [[`target-arguments`{.docutils .literal
                .notranslate}]{.pre}](using/extending/packaging.html#target-arguments){.reference
                .internal}
        -   [Building with [`CUDAQ_EXTERNAL_PROJECTS`{.docutils .literal
            .notranslate}]{.pre}](using/extending/packaging.html#building-with-cudaq-external-projects){.reference
            .internal}
            -   [Baking a target into the pre-compiled database instead
                of shipping a
                plugin](using/extending/packaging.html#baking-a-target-into-the-pre-compiled-database-instead-of-shipping-a-plugin){.reference
                .internal}
        -   [Python
            Packaging](using/extending/packaging.html#python-packaging){.reference
            .internal}
            -   [[`pyproject.toml`{.docutils .literal
                .notranslate}]{.pre}](using/extending/packaging.html#pyproject-toml){.reference
                .internal}
            -   [[`__init__.py`{.docutils .literal
                .notranslate}]{.pre}](using/extending/packaging.html#init-py){.reference
                .internal}
            -   [[`__main__.py`{.docutils .literal .notranslate}]{.pre}
                ([`--install-nvqpp`{.docutils .literal
                .notranslate}]{.pre}
                hook)](using/extending/packaging.html#main-py-install-nvqpp-hook){.reference
                .internal}
        -   [Installing the Plugin for End
            Users](using/extending/packaging.html#installing-the-plugin-for-end-users){.reference
            .internal}
            -   [[`pip`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`install`{.docutils .literal
                .notranslate}]{.pre} (Python --- zero
                config)](using/extending/packaging.html#pip-install-python-zero-config){.reference
                .internal}
            -   [[`--install-nvqpp`{.docutils .literal
                .notranslate}]{.pre} (make visible to [`nvq++`{.docutils
                .literal
                .notranslate}]{.pre})](using/extending/packaging.html#install-nvqpp-make-visible-to-nvq){.reference
                .internal}
            -   [[`cudaq-install-plugin`{.docutils .literal
                .notranslate}]{.pre} (C++-only
                workflows)](using/extending/packaging.html#cudaq-install-plugin-c-only-workflows){.reference
                .internal}
        -   [Discovery
            Mechanics](using/extending/packaging.html#discovery-mechanics){.reference
            .internal}
            -   [[`nvq++`{.docutils .literal .notranslate}]{.pre} target
                resolution](using/extending/packaging.html#nvq-target-resolution){.reference
                .internal}
            -   [Python target
                resolution](using/extending/packaging.html#python-target-resolution){.reference
                .internal}
            -   [Environment
                variables](using/extending/packaging.html#environment-variables){.reference
                .internal}
        -   [Reference
            Plugins](using/extending/packaging.html#reference-plugins){.reference
            .internal}
        -   [Quick-Start
            Checklist](using/extending/packaging.html#quick-start-checklist){.reference
            .internal}
    -   [Create an NVQIR
        simulator](using/extending/nvqir_simulator.html){.reference
        .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](using/extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](using/extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
-   [Specifications](specification/index.html){.reference .internal}
    -   [Language Specification](specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](specification/cudaq/types.html){.reference .internal}
            -   [3.1. [`cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. [`cudaq::qubit`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. [`cudaq::spin_op`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on [`cudaq::qubit`{.code .docutils
                .literal
                .notranslate}]{.pre}](specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](specification/cudaq/kernels.html){.reference
            .internal}
            -   [6.1. Atomic quantum
                regions](specification/cudaq/kernels.html#atomic-quantum-regions){.reference
                .internal}
        -   [7. Sub-circuit
            Synthesis](specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11. Platform](specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. [`cudaq::sample`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. [`cudaq::run`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. [`cudaq::observe`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. [`cudaq::optimizer`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. [`cudaq::gradient`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](specification/quake-dialect.html#motivation){.reference
            .internal}
        -   [Calling between reference and value
            forms](specification/quake-dialect.html#calling-between-reference-and-value-forms){.reference
            .internal}
-   [API Reference](api/api.html){.reference .internal}
    -   [C++ API](api/languages/cpp_api.html){.reference .internal}
        -   [Operators](api/languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](api/languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](api/languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](api/languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](api/languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](api/languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Quantum Error
            Correction](api/languages/cpp_api.html#quantum-error-correction){.reference
            .internal}
        -   [Platform](api/languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](api/languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](api/languages/cpp_api.html#namespaces){.reference
            .internal}
        -   [PTSBE](api/languages/cpp_api.html#ptsbe){.reference
            .internal}
            -   [Sampling
                Functions](api/languages/cpp_api.html#sampling-functions){.reference
                .internal}
            -   [Options](api/languages/cpp_api.html#options){.reference
                .internal}
            -   [Result
                Type](api/languages/cpp_api.html#result-type){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](api/languages/cpp_api.html#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](api/languages/cpp_api.html#shot-allocation-strategy){.reference
                .internal}
            -   [Execution
                Data](api/languages/cpp_api.html#execution-data){.reference
                .internal}
            -   [Trajectory and Selection
                Types](api/languages/cpp_api.html#trajectory-and-selection-types){.reference
                .internal}
    -   [Python API](api/languages/python_api.html){.reference
        .internal}
        -   [Program
            Construction](api/languages/python_api.html#program-construction){.reference
            .internal}
            -   [[`make_kernel()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [[`PyKernel`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [[`Kernel`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [[`PyKernelDecorator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [[`kernel()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [[`sample_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [[`run()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [[`run_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [[`observe()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [[`observe_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [[`get_state()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [[`get_state_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [[`vqe()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [[`draw()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [[`translate()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [[`estimate()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.estimate){.reference
                .internal}
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
            -   [[`dem_from_kernel()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.dem_from_kernel){.reference
                .internal}
        -   [[`cudaq.contrib`{.docutils .literal
            .notranslate}]{.pre}](api/languages/python_api.html#cudaq-contrib){.reference
            .internal}
            -   [Quantum
                Embeddings](api/languages/python_api.html#quantum-embeddings){.reference
                .internal}
            -   [Qiskit and OpenQASM
                Interoperability](api/languages/python_api.html#qiskit-and-openqasm-interoperability){.reference
                .internal}
        -   [Quantum Error
            Correction](api/languages/python_api.html#quantum-error-correction){.reference
            .internal}
            -   [[`detector()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.detector){.reference
                .internal}
            -   [[`detectors()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.detectors){.reference
                .internal}
            -   [[`logical_observable()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.logical_observable){.reference
                .internal}
            -   [[`to_bools()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.to_bools){.reference
                .internal}
        -   [Backend
            Configuration](api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [[`parse_args()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.parse_args){.reference
                .internal}
            -   [[`has_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [[`get_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [[`get_targets()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [[`set_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [[`reset_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [[`set_noise()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [[`unset_noise()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [[`register_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [[`unregister_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [[`cudaq.apply_noise()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [[`initialize_cudaq()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [[`num_available_gpus()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [[`set_random_seed()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [[`evolve()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [[`evolve_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [[`Schedule`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [[`BaseIntegrator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [[`InitialState`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [[`InitialStateType`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [[`IntermediateResultSave`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](api/languages/python_api.html#operators){.reference
            .internal}
            -   [[`OperatorSum`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [[`ProductOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [[`ElementaryOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [[`ScalarOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [[`RydbergHamiltonian`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [[`SuperOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [[`operators.define()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [[`operators.instantiate()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](api/languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](api/languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](api/languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](api/languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](api/languages/python_api.html#data-types){.reference
            .internal}
            -   [[`SimulationPrecision`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [[`Target`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [[`State`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [[`Tensor`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [[`QuakeValue`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [[`qubit`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [[`qreg`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [[`qvector`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [[`measure_handle`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.measure_handle){.reference
                .internal}
            -   [[`ComplexMatrix`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [[`SampleResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [[`AsyncSampleResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [[`DEMResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.DEMResult){.reference
                .internal}
            -   [[`ObserveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [[`AsyncObserveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [[`AsyncStateResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [[`OptimizationResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [[`EvolveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [[`AsyncEvolveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [[`Resources`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Resources){.reference
                .internal}
            -   [[`EstimateResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.EstimateResult){.reference
                .internal}
            -   [Optimizers](api/languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](api/languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](api/languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](api/languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [[`initialize()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [[`rank()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [[`num_ranks()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [[`all_gather()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [[`broadcast()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [[`is_initialized()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [[`split_communicator()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.split_communicator){.reference
                .internal}
            -   [[`set_communicator()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.set_communicator){.reference
                .internal}
            -   [[`finalize()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
        -   [PTSBE
            Submodule](api/languages/python_api.html#ptsbe-submodule){.reference
            .internal}
            -   [Sampling
                Functions](api/languages/python_api.html#sampling-functions){.reference
                .internal}
            -   [Result
                Type](api/languages/python_api.html#result-type){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](api/languages/python_api.html#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](api/languages/python_api.html#shot-allocation-strategy){.reference
                .internal}
            -   [Execution
                Data](api/languages/python_api.html#execution-data){.reference
                .internal}
            -   [Trajectory and Selection
                Types](api/languages/python_api.html#trajectory-and-selection-types){.reference
                .internal}
        -   [Synth
            Submodule](api/languages/python_api.html#synth-submodule){.reference
            .internal}
            -   [[`gridsynth()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.synth.gridsynth){.reference
                .internal}
            -   [[`rz_error()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.synth.rz_error){.reference
                .internal}
            -   [[`CliffordTSequence`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.synth.CliffordTSequence){.reference
                .internal}
    -   [Quantum Operations](api/default_ops.html){.reference .internal}
        -   [Unitary Operations on
            Qubits](api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [[`x`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#x){.reference
                .internal}
            -   [[`y`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#y){.reference
                .internal}
            -   [[`z`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#z){.reference
                .internal}
            -   [[`h`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#h){.reference
                .internal}
            -   [[`r1`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#r1){.reference
                .internal}
            -   [[`rx`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#rx){.reference
                .internal}
            -   [[`ry`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#ry){.reference
                .internal}
            -   [[`rz`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#rz){.reference
                .internal}
            -   [[`s`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#s){.reference
                .internal}
            -   [[`t`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#t){.reference
                .internal}
            -   [[`swap`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#swap){.reference
                .internal}
            -   [[`u3`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#u3){.reference
                .internal}
            -   [[`exp_pauli`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#exp-pauli){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#mz){.reference
                .internal}
            -   [[`mx`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#mx){.reference
                .internal}
            -   [[`my`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [[`create`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#create){.reference
                .internal}
            -   [[`annihilate`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#annihilate){.reference
                .internal}
            -   [[`phase_shift`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#phase-shift){.reference
                .internal}
            -   [[`beam_splitter`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](versions.html){.reference .internal}

[Preview]{.caption-text}

-   [cudaq-logical](/preview/logical/#http://){.reference .external}
-   [cudaq-edem](using/edem.html){.reference .internal}
    -   [Standalone build](using/edem.html#standalone-build){.reference
        .internal}
    -   [Python bindings and
        wheel](using/edem.html#python-bindings-and-wheel){.reference
        .internal}
    -   [CUDA-Q build](using/edem.html#cuda-q-build){.reference
        .internal}
    -   [Installed
        package](using/edem.html#installed-package){.reference
        .internal}
    -   [Packaging checks](using/edem.html#packaging-checks){.reference
        .internal}
    -   [Native interface and
        memory](using/edem.html#native-interface-and-memory){.reference
        .internal}
    -   [C++ ownership
        helper](using/edem.html#c-ownership-helper){.reference
        .internal}
    -   [Component
        boundary](using/edem.html#component-boundary){.reference
        .internal}
    -   [Fault
        distributions](using/edem.html#fault-distributions){.reference
        .internal}
    -   [Python
        construction](using/edem.html#python-construction){.reference
        .internal}
    -   [Packed format](using/edem.html#packed-format){.reference
        .internal}
    -   [Serial composition and DEM
        projection](using/edem.html#serial-composition-and-dem-projection){.reference
        .internal}
    -   [Exact noise conversion and column
        coalescing](using/edem.html#exact-noise-conversion-and-column-coalescing){.reference
        .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](index.html)

::: wy-nav-content
::: rst-content
::: {role="navigation" aria-label="Page navigation"}
-   [](index.html){.icon .icon-home aria-label="Home"}
-   Index
-   

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
# Index

::: genindex-jumpbox
[**\_**](#_) \| [**A**](#A) \| [**B**](#B) \| [**C**](#C) \| [**D**](#D)
\| [**E**](#E) \| [**F**](#F) \| [**G**](#G) \| [**H**](#H) \|
[**I**](#I) \| [**K**](#K) \| [**L**](#L) \| [**M**](#M) \| [**N**](#N)
\| [**O**](#O) \| [**P**](#P) \| [**Q**](#Q) \| [**R**](#R) \|
[**S**](#S) \| [**T**](#T) \| [**U**](#U) \| [**V**](#V) \| [**W**](#W)
\| [**X**](#X) \| [**Y**](#Y) \| [**Z**](#Z)
:::

## \_ {#_}

+-----------------------------------+-----------------------------------+
| -   [\_\_add\_\_()                | -   [\_\_init\_\_()               |
|     (cudaq.QuakeValue             |     (c                            |
|                                   | udaq.operators.RydbergHamiltonian |
|   method)](api/languages/python_a |     method)](api/lang             |
| pi.html#cudaq.QuakeValue.__add__) | uages/python_api.html#cudaq.opera |
| -   [\_\_call\_\_()               | tors.RydbergHamiltonian.__init__) |
|     (cudaq.PyKernelDecorator      | -   [\_\_iter\_\_                 |
|     method                        |     (cudaq.SampleResult           |
| )](api/languages/python_api.html# |     attr                          |
| cudaq.PyKernelDecorator.__call__) | ibute)](api/languages/python_api. |
| -   [\_\_getitem\_\_              | html#cudaq.SampleResult.__iter__) |
|     (cudaq.ComplexMatrix          | -   [\_\_len\_\_                  |
|     attribut                      |     (cudaq.SampleResult           |
| e)](api/languages/python_api.html |     att                           |
| #cudaq.ComplexMatrix.__getitem__) | ribute)](api/languages/python_api |
|     -   [(cudaq.KrausChannel      | .html#cudaq.SampleResult.__len__) |
|         attribu                   | -   [\_\_mul\_\_()                |
| te)](api/languages/python_api.htm |     (cudaq.QuakeValue             |
| l#cudaq.KrausChannel.__getitem__) |                                   |
|     -   [(cudaq.SampleResult      |   method)](api/languages/python_a |
|         attribu                   | pi.html#cudaq.QuakeValue.__mul__) |
| te)](api/languages/python_api.htm | -   [\_\_neg\_\_()                |
| l#cudaq.SampleResult.__getitem__) |     (cudaq.QuakeValue             |
| -   [\_\_getitem\_\_()            |                                   |
|     (cudaq.QuakeValue             |   method)](api/languages/python_a |
|     me                            | pi.html#cudaq.QuakeValue.__neg__) |
| thod)](api/languages/python_api.h | -   [\_\_radd\_\_()               |
| tml#cudaq.QuakeValue.__getitem__) |     (cudaq.QuakeValue             |
| -   [\_\_init\_\_                 |                                   |
|                                   |  method)](api/languages/python_ap |
|    (cudaq.AmplitudeDampingChannel | i.html#cudaq.QuakeValue.__radd__) |
|     attribute)](api               | -   [\_\_rmul\_\_()               |
| /languages/python_api.html#cudaq. |     (cudaq.QuakeValue             |
| AmplitudeDampingChannel.__init__) |                                   |
|     -   [(cudaq.BitFlipChannel    |  method)](api/languages/python_ap |
|         attrib                    | i.html#cudaq.QuakeValue.__rmul__) |
| ute)](api/languages/python_api.ht | -   [\_\_rsub\_\_()               |
| ml#cudaq.BitFlipChannel.__init__) |     (cudaq.QuakeValue             |
|                                   |                                   |
| -   [(cudaq.DepolarizationChannel |  method)](api/languages/python_ap |
|         attribute)](a             | i.html#cudaq.QuakeValue.__rsub__) |
| pi/languages/python_api.html#cuda | -   [\_\_str\_\_                  |
| q.DepolarizationChannel.__init__) |     (cudaq.ComplexMatrix          |
|     -   [(cudaq.NoiseModel        |     attr                          |
|         at                        | ibute)](api/languages/python_api. |
| tribute)](api/languages/python_ap | html#cudaq.ComplexMatrix.__str__) |
| i.html#cudaq.NoiseModel.__init__) | -   [\_\_str\_\_()                |
|     -   [(cudaq.PhaseFlipChannel  |     (cudaq.PyKernelDecorator      |
|         attribut                  |     metho                         |
| e)](api/languages/python_api.html | d)](api/languages/python_api.html |
| #cudaq.PhaseFlipChannel.__init__) | #cudaq.PyKernelDecorator.__str__) |
|                                   | -   [\_\_sub\_\_()                |
|                                   |     (cudaq.QuakeValue             |
|                                   |                                   |
|                                   |   method)](api/languages/python_a |
|                                   | pi.html#cudaq.QuakeValue.__sub__) |
+-----------------------------------+-----------------------------------+

## A {#A}

+-----------------------------------+-----------------------------------+
| -   [Adam (class in               | -   [append (cudaq.KrausChannel   |
|     cudaq                         |     at                            |
| .optimizers)](api/languages/pytho | tribute)](api/languages/python_ap |
| n_api.html#cudaq.optimizers.Adam) | i.html#cudaq.KrausChannel.append) |
| -   [add_all_qubit_channel        | -   [argument_count               |
|     (cudaq.NoiseModel             |     (cudaq.PyKernel               |
|     attribute)](api               |     attrib                        |
| /languages/python_api.html#cudaq. | ute)](api/languages/python_api.ht |
| NoiseModel.add_all_qubit_channel) | ml#cudaq.PyKernel.argument_count) |
| -   [add_channel                  | -   [arguments (cudaq.PyKernel    |
|     (cudaq.NoiseModel             |     a                             |
|     attri                         | ttribute)](api/languages/python_a |
| bute)](api/languages/python_api.h | pi.html#cudaq.PyKernel.arguments) |
| tml#cudaq.NoiseModel.add_channel) | -   [as_pauli                     |
| -   [all_gather() (in module      |     (cudaq.o                      |
|                                   | perators.spin.SpinOperatorElement |
|    cudaq.mpi)](api/languages/pyth |     attribute)](api/languages/    |
| on_api.html#cudaq.mpi.all_gather) | python_api.html#cudaq.operators.s |
| -   [amplitude (cudaq.State       | pin.SpinOperatorElement.as_pauli) |
|                                   | -   [AsyncEvolveResult (class in  |
|   attribute)](api/languages/pytho |     cudaq)](api/languages/python_ |
| n_api.html#cudaq.State.amplitude) | api.html#cudaq.AsyncEvolveResult) |
| -   [amplitude_encode() (in       | -   [AsyncObserveResult (class in |
|     module                        |                                   |
|     cudaq.contr                   |    cudaq)](api/languages/python_a |
| ib)](api/languages/python_api.htm | pi.html#cudaq.AsyncObserveResult) |
| l#cudaq.contrib.amplitude_encode) | -   [AsyncSampleResult (class in  |
| -   [AmplitudeDampingChannel      |     cudaq)](api/languages/python_ |
|     (class in                     | api.html#cudaq.AsyncSampleResult) |
|     cu                            | -   [AsyncStateResult (class in   |
| daq)](api/languages/python_api.ht |     cudaq)](api/languages/python  |
| ml#cudaq.AmplitudeDampingChannel) | _api.html#cudaq.AsyncStateResult) |
| -   [amplitudes (cudaq.State      | -   [atomic_quantum_region        |
|                                   |     (cudaq.PyKernelDecorator      |
|  attribute)](api/languages/python |     property)](api/langua         |
| _api.html#cudaq.State.amplitudes) | ges/python_api.html#cudaq.PyKerne |
| -   [angular_encode() (in module  | lDecorator.atomic_quantum_region) |
|     cudaq.con                     | -   [availability_diagnostic      |
| trib)](api/languages/python_api.h |     (cudaq.Target                 |
| tml#cudaq.contrib.angular_encode) |     property)](a                  |
| -   [annotations (cudaq.DEMResult | pi/languages/python_api.html#cuda |
|     pro                           | q.Target.availability_diagnostic) |
| perty)](api/languages/python_api. |                                   |
| html#cudaq.DEMResult.annotations) |                                   |
|     -   [(cudaq.EstimateResult    |                                   |
|         property                  |                                   |
| )](api/languages/python_api.html# |                                   |
| cudaq.EstimateResult.annotations) |                                   |
|     -   [(cudaq.SampleResult      |                                   |
|         proper                    |                                   |
| ty)](api/languages/python_api.htm |                                   |
| l#cudaq.SampleResult.annotations) |                                   |
+-----------------------------------+-----------------------------------+

## B {#B}

+-----------------------------------+-----------------------------------+
| -   [BaseIntegrator (class in     | -   [bias_strength                |
|                                   |     (c                            |
| cudaq.dynamics.integrator)](api/l | udaq.ptsbe.ShotAllocationStrategy |
| anguages/python_api.html#cudaq.dy |     property)](api/languages      |
| namics.integrator.BaseIntegrator) | /python_api.html#cudaq.ptsbe.Shot |
| -   [batch_size                   | AllocationStrategy.bias_strength) |
|     (cudaq.optimizers.Adam        | -   [BitFlipChannel (class in     |
|     property                      |     cudaq)](api/languages/pyth    |
| )](api/languages/python_api.html# | on_api.html#cudaq.BitFlipChannel) |
| cudaq.optimizers.Adam.batch_size) | -   [BosonOperator (class in      |
|     -   [(cudaq.optimizers.SGD    |     cudaq.operators.boson)](      |
|         propert                   | api/languages/python_api.html#cud |
| y)](api/languages/python_api.html | aq.operators.boson.BosonOperator) |
| #cudaq.optimizers.SGD.batch_size) | -   [BosonOperatorElement (class  |
| -   [beta1 (cudaq.optimizers.Adam |     in                            |
|     pro                           |                                   |
| perty)](api/languages/python_api. |   cudaq.operators.boson)](api/lan |
| html#cudaq.optimizers.Adam.beta1) | guages/python_api.html#cudaq.oper |
| -   [beta2 (cudaq.optimizers.Adam | ators.boson.BosonOperatorElement) |
|     pro                           | -   [BosonOperatorTerm (class in  |
| perty)](api/languages/python_api. |     cudaq.operators.boson)](api/  |
| html#cudaq.optimizers.Adam.beta2) | languages/python_api.html#cudaq.o |
| -   [beta_reduction()             | perators.boson.BosonOperatorTerm) |
|     (cudaq.PyKernelDecorator      | -   [broadcast() (in module       |
|     method)](api                  |     cudaq.mpi)](api/languages/pyt |
| /languages/python_api.html#cudaq. | hon_api.html#cudaq.mpi.broadcast) |
| PyKernelDecorator.beta_reduction) |                                   |
+-----------------------------------+-----------------------------------+

## C {#C}

+-----------------------------------+-----------------------------------+
| -   [canonicalize                 | -   [cudaq::phase_damping (C++    |
|     (cu                           |                                   |
| daq.operators.boson.BosonOperator |  class)](api/languages/cpp_api.ht |
|     attribute)](api/languages     | ml#_CPPv4N5cudaq13phase_dampingE) |
| /python_api.html#cudaq.operators. | -   [cud                          |
| boson.BosonOperator.canonicalize) | aq::phase_damping::num_parameters |
|     -   [(cudaq.                  |     (C++                          |
| operators.boson.BosonOperatorTerm |     member)](api/lan              |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|     attribute)](api/languages/pyt | 13phase_damping14num_parametersE) |
| hon_api.html#cudaq.operators.boso | -   [                             |
| n.BosonOperatorTerm.canonicalize) | cudaq::phase_damping::num_targets |
|     -   [(cudaq.                  |     (C++                          |
| operators.fermion.FermionOperator |     member)](api/                 |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|     attribute)](api/languages/pyt | daq13phase_damping11num_targetsE) |
| hon_api.html#cudaq.operators.ferm | -   [cudaq::phase_flip_channel    |
| ion.FermionOperator.canonicalize) |     (C++                          |
|     -   [(cudaq.oper              |     clas                          |
| ators.fermion.FermionOperatorTerm | s)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq18phase_flip_channelE) |
| attribute)](api/languages/python_ | -   [cudaq::p                     |
| api.html#cudaq.operators.fermion. | hase_flip_channel::num_parameters |
| FermionOperatorTerm.canonicalize) |     (C++                          |
|     -                             |     member)](api/language         |
|  [(cudaq.operators.MatrixOperator | s/cpp_api.html#_CPPv4N5cudaq18pha |
|         attribute)](api/lang      | se_flip_channel14num_parametersE) |
| uages/python_api.html#cudaq.opera | -   [cudaq                        |
| tors.MatrixOperator.canonicalize) | ::phase_flip_channel::num_targets |
|     -   [(c                       |     (C++                          |
| udaq.operators.MatrixOperatorTerm |     member)](api/langu            |
|         attribute)](api/language  | ages/cpp_api.html#_CPPv4N5cudaq18 |
| s/python_api.html#cudaq.operators | phase_flip_channel11num_targetsE) |
| .MatrixOperatorTerm.canonicalize) | -   [cudaq::product_op (C++       |
|     -   [(                        |                                   |
| cudaq.operators.spin.SpinOperator |  class)](api/languages/cpp_api.ht |
|         attribute)](api/languag   | ml#_CPPv4I0EN5cudaq10product_opE) |
| es/python_api.html#cudaq.operator | -   [cudaq::product_op::begin     |
| s.spin.SpinOperator.canonicalize) |     (C++                          |
|     -   [(cuda                    |     functio                       |
| q.operators.spin.SpinOperatorTerm | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4NK5cudaq10product_op5beginEv) |
|       attribute)](api/languages/p | -                                 |
| ython_api.html#cudaq.operators.sp |  [cudaq::product_op::canonicalize |
| in.SpinOperatorTerm.canonicalize) |     (C++                          |
| -   [captured_variables()         |     func                          |
|     (cudaq.PyKernelDecorator      | tion)](api/languages/cpp_api.html |
|     method)](api/lan              | #_CPPv4N5cudaq10product_op12canon |
| guages/python_api.html#cudaq.PyKe | icalizeERKNSt3setINSt6size_tEEE), |
| rnelDecorator.captured_variables) |     [\[1\]](api                   |
| -   [CentralDifference (class in  | /languages/cpp_api.html#_CPPv4N5c |
|     cudaq.gradients)              | udaq10product_op12canonicalizeEv) |
| ](api/languages/python_api.html#c | -   [                             |
| udaq.gradients.CentralDifference) | cudaq::product_op::const_iterator |
| -   [channel                      |     (C++                          |
|     (cudaq.ptsbe.TraceInstruction |     struct)](api/                 |
|     property)](a                  | languages/cpp_api.html#_CPPv4N5cu |
| pi/languages/python_api.html#cuda | daq10product_op14const_iteratorE) |
| q.ptsbe.TraceInstruction.channel) | -   [cudaq::product_o             |
| -   [circuit_location             | p::const_iterator::const_iterator |
|     (cudaq.ptsbe.KrausSelection   |     (C++                          |
|     property)](api/lang           |     fu                            |
| uages/python_api.html#cudaq.ptsbe | nction)](api/languages/cpp_api.ht |
| .KrausSelection.circuit_location) | ml#_CPPv4N5cudaq10product_op14con |
| -   [clear (cudaq.Resources       | st_iterator14const_iteratorEPK10p |
|                                   | roduct_opI9HandlerTyENSt6size_tE) |
|   attribute)](api/languages/pytho | -   [cudaq::produ                 |
| n_api.html#cudaq.Resources.clear) | ct_op::const_iterator::operator!= |
|     -   [(cudaq.SampleResult      |     (C++                          |
|         a                         |     fun                           |
| ttribute)](api/languages/python_a | ction)](api/languages/cpp_api.htm |
| pi.html#cudaq.SampleResult.clear) | l#_CPPv4NK5cudaq10product_op14con |
| -   [CliffordTSequence (class in  | st_iteratorneERK14const_iterator) |
|     cudaq.sy                      | -   [cudaq::produ                 |
| nth)](api/languages/python_api.ht | ct_op::const_iterator::operator\* |
| ml#cudaq.synth.CliffordTSequence) |     (C++                          |
| -   [COBYLA (class in             |     function)](api/lang           |
|     cudaq.o                       | uages/cpp_api.html#_CPPv4NK5cudaq |
| ptimizers)](api/languages/python_ | 10product_op14const_iteratormlEv) |
| api.html#cudaq.optimizers.COBYLA) | -   [cudaq::produ                 |
| -   [coefficient                  | ct_op::const_iterator::operator++ |
|     (cudaq.                       |     (C++                          |
| operators.boson.BosonOperatorTerm |     function)](api/lang           |
|     property)](api/languages/py   | uages/cpp_api.html#_CPPv4N5cudaq1 |
| thon_api.html#cudaq.operators.bos | 0product_op14const_iteratorppEi), |
| on.BosonOperatorTerm.coefficient) |     [\[1\]](api/lan               |
|     -   [(cudaq.oper              | guages/cpp_api.html#_CPPv4N5cudaq |
| ators.fermion.FermionOperatorTerm | 10product_op14const_iteratorppEv) |
|                                   | -   [cudaq::produc                |
|   property)](api/languages/python | t_op::const_iterator::operator\-- |
| _api.html#cudaq.operators.fermion |     (C++                          |
| .FermionOperatorTerm.coefficient) |     function)](api/lang           |
|     -   [(c                       | uages/cpp_api.html#_CPPv4N5cudaq1 |
| udaq.operators.MatrixOperatorTerm | 0product_op14const_iteratormmEi), |
|         property)](api/languag    |     [\[1\]](api/lan               |
| es/python_api.html#cudaq.operator | guages/cpp_api.html#_CPPv4N5cudaq |
| s.MatrixOperatorTerm.coefficient) | 10product_op14const_iteratormmEv) |
|     -   [(cuda                    | -   [cudaq::produc                |
| q.operators.spin.SpinOperatorTerm | t_op::const_iterator::operator-\> |
|         property)](api/languages/ |     (C++                          |
| python_api.html#cudaq.operators.s |     function)](api/lan            |
| pin.SpinOperatorTerm.coefficient) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [col_count                    | 10product_op14const_iteratorptEv) |
|     (cudaq.KrausOperator          | -   [cudaq::produ                 |
|     prope                         | ct_op::const_iterator::operator== |
| rty)](api/languages/python_api.ht |     (C++                          |
| ml#cudaq.KrausOperator.col_count) |     fun                           |
| -   [compile()                    | ction)](api/languages/cpp_api.htm |
|     (cudaq.PyKernelDecorator      | l#_CPPv4NK5cudaq10product_op14con |
|     metho                         | st_iteratoreqERK14const_iterator) |
| d)](api/languages/python_api.html | -   [cudaq::product_op::degrees   |
| #cudaq.PyKernelDecorator.compile) |     (C++                          |
| -   [compiledModuleCache()        |     function)                     |
|     (cudaq.PyKernelDecorator      | ](api/languages/cpp_api.html#_CPP |
|     method)](api/lang             | v4NK5cudaq10product_op7degreesEv) |
| uages/python_api.html#cudaq.PyKer | -   [cudaq::product_op::dump (C++ |
| nelDecorator.compiledModuleCache) |     functi                        |
| -   [ComplexMatrix (class in      | on)](api/languages/cpp_api.html#_ |
|     cudaq)](api/languages/pyt     | CPPv4NK5cudaq10product_op4dumpEv) |
| hon_api.html#cudaq.ComplexMatrix) | -   [cudaq::product_op::end (C++  |
| -   [compute                      |     funct                         |
|     (                             | ion)](api/languages/cpp_api.html# |
| cudaq.gradients.CentralDifference | _CPPv4NK5cudaq10product_op3endEv) |
|     attribute)](api/la            | -   [c                            |
| nguages/python_api.html#cudaq.gra | udaq::product_op::get_coefficient |
| dients.CentralDifference.compute) |     (C++                          |
|     -   [(                        |     function)](api/lan            |
| cudaq.gradients.ForwardDifference | guages/cpp_api.html#_CPPv4NK5cuda |
|         attribute)](api/la        | q10product_op15get_coefficientEv) |
| nguages/python_api.html#cudaq.gra | -                                 |
| dients.ForwardDifference.compute) |   [cudaq::product_op::get_term_id |
|     -                             |     (C++                          |
|  [(cudaq.gradients.ParameterShift |     function)](api                |
|         attribute)](api           | /languages/cpp_api.html#_CPPv4NK5 |
| /languages/python_api.html#cudaq. | cudaq10product_op11get_term_idEv) |
| gradients.ParameterShift.compute) | -                                 |
| -   [const()                      |   [cudaq::product_op::is_identity |
|                                   |     (C++                          |
|   (cudaq.operators.ScalarOperator |     function)](api                |
|     class                         | /languages/cpp_api.html#_CPPv4NK5 |
|     method)](a                    | cudaq10product_op11is_identityEv) |
| pi/languages/python_api.html#cuda | -   [cudaq::product_op::num_ops   |
| q.operators.ScalarOperator.const) |     (C++                          |
| -   [controls                     |     function)                     |
|     (cudaq.ptsbe.TraceInstruction | ](api/languages/cpp_api.html#_CPP |
|     property)](ap                 | v4NK5cudaq10product_op7num_opsEv) |
| i/languages/python_api.html#cudaq | -                                 |
| .ptsbe.TraceInstruction.controls) |    [cudaq::product_op::operator\* |
| -   [copy                         |     (C++                          |
|     (cu                           |     function)](api/languages/     |
| daq.operators.boson.BosonOperator | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     attribute)](api/l             | oduct_opmlE10product_opI1TERK15sc |
| anguages/python_api.html#cudaq.op | alar_operatorRK10product_opI1TE), |
| erators.boson.BosonOperator.copy) |     [\[1\]](api/languages/        |
|     -   [(cudaq.                  | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| operators.boson.BosonOperatorTerm | oduct_opmlE10product_opI1TERK15sc |
|         attribute)](api/langu     | alar_operatorRR10product_opI1TE), |
| ages/python_api.html#cudaq.operat |     [\[2\]](api/languages/        |
| ors.boson.BosonOperatorTerm.copy) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(cudaq.                  | oduct_opmlE10product_opI1TERR15sc |
| operators.fermion.FermionOperator | alar_operatorRK10product_opI1TE), |
|         attribute)](api/langu     |     [\[3\]](api/languages/        |
| ages/python_api.html#cudaq.operat | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| ors.fermion.FermionOperator.copy) | oduct_opmlE10product_opI1TERR15sc |
|     -   [(cudaq.oper              | alar_operatorRR10product_opI1TE), |
| ators.fermion.FermionOperatorTerm |     [\[4\]](api/                  |
|         attribute)](api/languages | languages/cpp_api.html#_CPPv4I0EN |
| /python_api.html#cudaq.operators. | 5cudaq10product_opmlE6sum_opI1TER |
| fermion.FermionOperatorTerm.copy) | K15scalar_operatorRK6sum_opI1TE), |
|     -                             |     [\[5\]](api/                  |
|  [(cudaq.operators.MatrixOperator | languages/cpp_api.html#_CPPv4I0EN |
|         attribute)](              | 5cudaq10product_opmlE6sum_opI1TER |
| api/languages/python_api.html#cud | K15scalar_operatorRR6sum_opI1TE), |
| aq.operators.MatrixOperator.copy) |     [\[6\]](api/                  |
|     -   [(c                       | languages/cpp_api.html#_CPPv4I0EN |
| udaq.operators.MatrixOperatorTerm | 5cudaq10product_opmlE6sum_opI1TER |
|         attribute)](api/          | R15scalar_operatorRK6sum_opI1TE), |
| languages/python_api.html#cudaq.o |     [\[7\]](api/                  |
| perators.MatrixOperatorTerm.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(                        | 5cudaq10product_opmlE6sum_opI1TER |
| cudaq.operators.spin.SpinOperator | R15scalar_operatorRR6sum_opI1TE), |
|         attribute)](api           |     [\[8\]](api/languages         |
| /languages/python_api.html#cudaq. | /cpp_api.html#_CPPv4NK5cudaq10pro |
| operators.spin.SpinOperator.copy) | duct_opmlERK6sum_opI9HandlerTyE), |
|     -   [(cuda                    |     [\[9\]](api/languages/cpp_a   |
| q.operators.spin.SpinOperatorTerm | pi.html#_CPPv4NKR5cudaq10product_ |
|         attribute)](api/lan       | opmlERK10product_opI9HandlerTyE), |
| guages/python_api.html#cudaq.oper |     [\[10\]](api/language         |
| ators.spin.SpinOperatorTerm.copy) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [count (cudaq.Resources       | roduct_opmlERK15scalar_operator), |
|                                   |     [\[11\]](api/languages/cpp_a  |
|   attribute)](api/languages/pytho | pi.html#_CPPv4NKR5cudaq10product_ |
| n_api.html#cudaq.Resources.count) | opmlERR10product_opI9HandlerTyE), |
|     -   [(cudaq.SampleResult      |     [\[12\]](api/language         |
|         a                         | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ttribute)](api/languages/python_a | roduct_opmlERR15scalar_operator), |
| pi.html#cudaq.SampleResult.count) |     [\[13\]](api/languages/cpp_   |
| -   [count_controls               | api.html#_CPPv4NO5cudaq10product_ |
|     (cudaq.Resources              | opmlERK10product_opI9HandlerTyE), |
|     attribu                       |     [\[14\]](api/languag          |
| te)](api/languages/python_api.htm | es/cpp_api.html#_CPPv4NO5cudaq10p |
| l#cudaq.Resources.count_controls) | roduct_opmlERK15scalar_operator), |
| -   [count_instructions           |     [\[15\]](api/languages/cpp_   |
|                                   | api.html#_CPPv4NO5cudaq10product_ |
|   (cudaq.ptsbe.PTSBEExecutionData | opmlERR10product_opI9HandlerTyE), |
|     attribute)](api/languages/    |     [\[16\]](api/langua           |
| python_api.html#cudaq.ptsbe.PTSBE | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| ExecutionData.count_instructions) | product_opmlERR15scalar_operator) |
| -   [counts (cudaq.ObserveResult  | -                                 |
|     att                           |   [cudaq::product_op::operator\*= |
| ribute)](api/languages/python_api |     (C++                          |
| .html#cudaq.ObserveResult.counts) |     function)](api/languages/cpp  |
|     -   [(cudaq.SampleResult      | _api.html#_CPPv4N5cudaq10product_ |
|         p                         | opmLERK10product_opI9HandlerTyE), |
| roperty)](api/languages/python_ap |     [\[1\]](api/langua            |
| i.html#cudaq.SampleResult.counts) | ges/cpp_api.html#_CPPv4N5cudaq10p |
| -   [csr_spmatrix (C++            | roduct_opmLERK15scalar_operator), |
|     type)](api/languages/c        |     [\[2\]](api/languages/cp      |
| pp_api.html#_CPPv412csr_spmatrix) | p_api.html#_CPPv4N5cudaq10product |
| -   cudaq                         | _opmLERR10product_opI9HandlerTyE) |
|     -   [module](api/langua       | -   [cudaq::product_op::operator+ |
| ges/python_api.html#module-cudaq) |     (C++                          |
| -   [cudaq (C++                   |     function)](api/langu          |
|     type)](api/lan                | ages/cpp_api.html#_CPPv4I0EN5cuda |
| guages/cpp_api.html#_CPPv45cudaq) | q10product_opplE6sum_opI1TERK15sc |
| -   [cudaq.apply_noise() (in      | alar_operatorRK10product_opI1TE), |
|     module                        |     [\[1\]](api/                  |
|     cudaq)](api/languages/python_ | languages/cpp_api.html#_CPPv4I0EN |
| api.html#cudaq.cudaq.apply_noise) | 5cudaq10product_opplE6sum_opI1TER |
| -   cudaq.boson                   | K15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/languages/py |     [\[2\]](api/langu             |
| thon_api.html#module-cudaq.boson) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq.fermion                 | q10product_opplE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|   -   [module](api/languages/pyth |     [\[3\]](api/                  |
| on_api.html#module-cudaq.fermion) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.operators.custom        | 5cudaq10product_opplE6sum_opI1TER |
|     -   [mo                       | K15scalar_operatorRR6sum_opI1TE), |
| dule](api/languages/python_api.ht |     [\[4\]](api/langu             |
| ml#module-cudaq.operators.custom) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq.spin                    | q10product_opplE6sum_opI1TERR15sc |
|     -   [module](api/languages/p  | alar_operatorRK10product_opI1TE), |
| ython_api.html#module-cudaq.spin) |     [\[5\]](api/                  |
| -   [cudaq::amplitude_damping     | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     cla                           | R15scalar_operatorRK6sum_opI1TE), |
| ss)](api/languages/cpp_api.html#_ |     [\[6\]](api/langu             |
| CPPv4N5cudaq17amplitude_dampingE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -                                 | q10product_opplE6sum_opI1TERR15sc |
| [cudaq::amplitude_damping_channel | alar_operatorRR10product_opI1TE), |
|     (C++                          |     [\[7\]](api/                  |
|     class)](api                   | languages/cpp_api.html#_CPPv4I0EN |
| /languages/cpp_api.html#_CPPv4N5c | 5cudaq10product_opplE6sum_opI1TER |
| udaq25amplitude_damping_channelE) | R15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq::amplitud              |     [\[8\]](api/languages/cpp_a   |
| e_damping_channel::num_parameters | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opplERK10product_opI9HandlerTyE), |
|     member)](api/languages/cpp_a  |     [\[9\]](api/language          |
| pi.html#_CPPv4N5cudaq25amplitude_ | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| damping_channel14num_parametersE) | roduct_opplERK15scalar_operator), |
| -   [cudaq::ampli                 |     [\[10\]](api/languages/       |
| tude_damping_channel::num_targets | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     (C++                          | duct_opplERK6sum_opI9HandlerTyE), |
|     member)](api/languages/cp     |     [\[11\]](api/languages/cpp_a  |
| p_api.html#_CPPv4N5cudaq25amplitu | pi.html#_CPPv4NKR5cudaq10product_ |
| de_damping_channel11num_targetsE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::AnalogRemoteRESTQPU   |     [\[12\]](api/language         |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     class                         | roduct_opplERR15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[13\]](api/languages/       |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::apply_noise (C++      | duct_opplERR6sum_opI9HandlerTyE), |
|     function)](api/               |     [\[                           |
| languages/cpp_api.html#_CPPv4I0Dp | 14\]](api/languages/cpp_api.html# |
| EN5cudaq11apply_noiseEvDpRR4Args) | _CPPv4NKR5cudaq10product_opplEv), |
| -   [cudaq::async_result (C++     |     [\[15\]](api/languages/cpp_   |
|     c                             | api.html#_CPPv4NO5cudaq10product_ |
| lass)](api/languages/cpp_api.html | opplERK10product_opI9HandlerTyE), |
| #_CPPv4I0EN5cudaq12async_resultE) |     [\[16\]](api/languag          |
| -   [cudaq::async_result::get     | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opplERK15scalar_operator), |
|     functi                        |     [\[17\]](api/languages        |
| on)](api/languages/cpp_api.html#_ | /cpp_api.html#_CPPv4NO5cudaq10pro |
| CPPv4N5cudaq12async_result3getEv) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq::async_sample_result   |     [\[18\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     type                          | opplERR10product_opI9HandlerTyE), |
| )](api/languages/cpp_api.html#_CP |     [\[19\]](api/languag          |
| Pv4N5cudaq19async_sample_resultE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::BaseRemoteRESTQPU     | roduct_opplERR15scalar_operator), |
|     (C++                          |     [\[20\]](api/languages        |
|     cla                           | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ss)](api/languages/cpp_api.html#_ | duct_opplERR6sum_opI9HandlerTyE), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [                             |
| -   [cudaq::bit_flip_channel (C++ | \[21\]](api/languages/cpp_api.htm |
|     cl                            | l#_CPPv4NO5cudaq10product_opplEv) |
| ass)](api/languages/cpp_api.html# | -   [cudaq::product_op::operator- |
| _CPPv4N5cudaq16bit_flip_channelE) |     (C++                          |
| -   [cudaq:                       |     function)](api/langu          |
| :bit_flip_channel::num_parameters | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERK15sc |
|     member)](api/langua           | alar_operatorRK10product_opI1TE), |
| ges/cpp_api.html#_CPPv4N5cudaq16b |     [\[1\]](api/                  |
| it_flip_channel14num_parametersE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cud                          | 5cudaq10product_opmiE6sum_opI1TER |
| aq::bit_flip_channel::num_targets | K15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[2\]](api/langu             |
|     member)](api/lan              | ages/cpp_api.html#_CPPv4I0EN5cuda |
| guages/cpp_api.html#_CPPv4N5cudaq | q10product_opmiE6sum_opI1TERK15sc |
| 16bit_flip_channel11num_targetsE) | alar_operatorRR10product_opI1TE), |
| -   [cudaq::boson_handler (C++    |     [\[3\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|  class)](api/languages/cpp_api.ht | 5cudaq10product_opmiE6sum_opI1TER |
| ml#_CPPv4N5cudaq13boson_handlerE) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq::boson_op (C++         |     [\[4\]](api/langu             |
|     type)](api/languages/cpp_     | ages/cpp_api.html#_CPPv4I0EN5cuda |
| api.html#_CPPv4N5cudaq8boson_opE) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq::boson_op_term (C++    | alar_operatorRK10product_opI1TE), |
|                                   |     [\[5\]](api/                  |
|   type)](api/languages/cpp_api.ht | languages/cpp_api.html#_CPPv4I0EN |
| ml#_CPPv4N5cudaq13boson_op_termE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::CodeGenConfig (C++    | R15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[6\]](api/langu             |
| struct)](api/languages/cpp_api.ht | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq::commutation_relations | alar_operatorRR10product_opI1TE), |
|     (C++                          |     [\[7\]](api/                  |
|     struct)]                      | languages/cpp_api.html#_CPPv4I0EN |
| (api/languages/cpp_api.html#_CPPv | 5cudaq10product_opmiE6sum_opI1TER |
| 4N5cudaq21commutation_relationsE) | R15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq::complex (C++          |     [\[8\]](api/languages/cpp_a   |
|     type)](api/languages/cpp      | pi.html#_CPPv4NKR5cudaq10product_ |
| _api.html#_CPPv4N5cudaq7complexE) | opmiERK10product_opI9HandlerTyE), |
| -   [cudaq::complex_matrix (C++   |     [\[9\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| class)](api/languages/cpp_api.htm | roduct_opmiERK15scalar_operator), |
| l#_CPPv4N5cudaq14complex_matrixE) |     [\[10\]](api/languages/       |
| -                                 | cpp_api.html#_CPPv4NKR5cudaq10pro |
|   [cudaq::complex_matrix::adjoint | duct_opmiERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     function)](a                  | pi.html#_CPPv4NKR5cudaq10product_ |
| pi/languages/cpp_api.html#_CPPv4N | opmiERR10product_opI9HandlerTyE), |
| 5cudaq14complex_matrix7adjointEv) |     [\[12\]](api/language         |
| -   [cudaq::                      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| complex_matrix::diagonal_elements | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     function)](api/languages      | cpp_api.html#_CPPv4NKR5cudaq10pro |
| /cpp_api.html#_CPPv4NK5cudaq14com | duct_opmiERR6sum_opI9HandlerTyE), |
| plex_matrix17diagonal_elementsEi) |     [\[                           |
| -   [cudaq::complex_matrix::dump  | 14\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NKR5cudaq10product_opmiEv), |
|     function)](api/language       |     [\[15\]](api/languages/cpp_   |
| s/cpp_api.html#_CPPv4NK5cudaq14co | api.html#_CPPv4NO5cudaq10product_ |
| mplex_matrix4dumpERNSt7ostreamE), | opmiERK10product_opI9HandlerTyE), |
|     [\[1\]]                       |     [\[16\]](api/languag          |
| (api/languages/cpp_api.html#_CPPv | es/cpp_api.html#_CPPv4NO5cudaq10p |
| 4NK5cudaq14complex_matrix4dumpEv) | roduct_opmiERK15scalar_operator), |
| -   [c                            |     [\[17\]](api/languages        |
| udaq::complex_matrix::eigenvalues | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERK6sum_opI9HandlerTyE), |
|     function)](api/lan            |     [\[18\]](api/languages/cpp_   |
| guages/cpp_api.html#_CPPv4NK5cuda | api.html#_CPPv4NO5cudaq10product_ |
| q14complex_matrix11eigenvaluesEv) | opmiERR10product_opI9HandlerTyE), |
| -   [cu                           |     [\[19\]](api/languag          |
| daq::complex_matrix::eigenvectors | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmiERR15scalar_operator), |
|     function)](api/lang           |     [\[20\]](api/languages        |
| uages/cpp_api.html#_CPPv4NK5cudaq | /cpp_api.html#_CPPv4NO5cudaq10pro |
| 14complex_matrix12eigenvectorsEv) | duct_opmiERR6sum_opI9HandlerTyE), |
| -   [c                            |     [                             |
| udaq::complex_matrix::exponential | \[21\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NO5cudaq10product_opmiEv) |
|     function)](api/la             | -   [cudaq::product_op::operator/ |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14complex_matrix11exponentialEv) |     function)](api/language       |
| -                                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|  [cudaq::complex_matrix::identity | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/language          |
|     function)](api/languages      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /cpp_api.html#_CPPv4N5cudaq14comp | roduct_opdvERR15scalar_operator), |
| lex_matrix8identityEKNSt6size_tE) |     [\[2\]](api/languag           |
| -                                 | es/cpp_api.html#_CPPv4NO5cudaq10p |
| [cudaq::complex_matrix::kronecker | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[3\]](api/langua            |
|     function)](api/lang           | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| uages/cpp_api.html#_CPPv4I00EN5cu | product_opdvERR15scalar_operator) |
| daq14complex_matrix9kroneckerE14c | -                                 |
| omplex_matrix8Iterable8Iterable), |    [cudaq::product_op::operator/= |
|     [\[1\]](api/l                 |     (C++                          |
| anguages/cpp_api.html#_CPPv4N5cud |     function)](api/langu          |
| aq14complex_matrix9kroneckerERK14 | ages/cpp_api.html#_CPPv4N5cudaq10 |
| complex_matrixRK14complex_matrix) | product_opdVERK15scalar_operator) |
| -   [cudaq::c                     | -   [cudaq::product_op::operator= |
| omplex_matrix::minimal_eigenvalue |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)](api/languages/     | anguages/cpp_api.html#_CPPv4I00EN |
| cpp_api.html#_CPPv4NK5cudaq14comp | 5cudaq10product_opaSER10product_o |
| lex_matrix18minimal_eigenvalueEv) | pI9HandlerTyERK10product_opI1TE), |
| -   [                             |     [\[1\]](api/languages/cpp     |
| cudaq::complex_matrix::operator() | _api.html#_CPPv4N5cudaq10product_ |
|     (C++                          | opaSERK10product_opI9HandlerTyE), |
|     function)](api/languages/cpp  |     [\[2\]](api/languages/cp      |
| _api.html#_CPPv4N5cudaq14complex_ | p_api.html#_CPPv4N5cudaq10product |
| matrixclENSt6size_tENSt6size_tE), | _opaSERR10product_opI9HandlerTyE) |
|     [\[1\]](api/languages/cpp     | -                                 |
| _api.html#_CPPv4NK5cudaq14complex |    [cudaq::product_op::operator== |
| _matrixclENSt6size_tENSt6size_tE) |     (C++                          |
| -   [                             |     function)](api/languages/cpp  |
| cudaq::complex_matrix::operator\* | _api.html#_CPPv4NK5cudaq10product |
|     (C++                          | _opeqERK10product_opI9HandlerTyE) |
|     function)](api/langua         | -                                 |
| ges/cpp_api.html#_CPPv4N5cudaq14c |  [cudaq::product_op::operator\[\] |
| omplex_matrixmlEN14complex_matrix |     (C++                          |
| 10value_typeERK14complex_matrix), |     function)](ap                 |
|     [\[1\]                        | i/languages/cpp_api.html#_CPPv4NK |
| ](api/languages/cpp_api.html#_CPP | 5cudaq10product_opixENSt6size_tE) |
| v4N5cudaq14complex_matrixmlERK14c | -                                 |
| omplex_matrixRK14complex_matrix), |    [cudaq::product_op::product_op |
|                                   |     (C++                          |
|  [\[2\]](api/languages/cpp_api.ht |     f                             |
| ml#_CPPv4N5cudaq14complex_matrixm | unction)](api/languages/cpp_api.h |
| lERK14complex_matrixRKNSt6vectorI | tml#_CPPv4I00EN5cudaq10product_op |
| N14complex_matrix10value_typeEEE) | 10product_opERK10product_opI1TE), |
| -                                 |     [\[1\]]                       |
| [cudaq::complex_matrix::operator+ | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4I00EN5cudaq10product_op10product |
|     function                      | _opERK10product_opI1TERKN14matrix |
| )](api/languages/cpp_api.html#_CP | _handler20commutation_behaviorE), |
| Pv4N5cudaq14complex_matrixplERK14 |                                   |
| complex_matrixRK14complex_matrix) |   [\[2\]](api/languages/cpp_api.h |
| -                                 | tml#_CPPv4N5cudaq10product_op10pr |
| [cudaq::complex_matrix::operator- | oduct_opENSt6size_tENSt6size_tE), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     function                      | p_api.html#_CPPv4N5cudaq10product |
| )](api/languages/cpp_api.html#_CP | _op10product_opENSt7complexIdEE), |
| Pv4N5cudaq14complex_matrixmiERK14 |     [\[4\]](api/l                 |
| complex_matrixRK14complex_matrix) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cu                           | aq10product_op10product_opERK10pr |
| daq::complex_matrix::operator\[\] | oduct_opI9HandlerTyENSt6size_tE), |
|     (C++                          |     [\[5\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|  function)](api/languages/cpp_api | aq10product_op10product_opERR10pr |
| .html#_CPPv4N5cudaq14complex_matr | oduct_opI9HandlerTyENSt6size_tE), |
| ixixERKNSt6vectorINSt6size_tEEE), |     [\[6\]](api/languages         |
|     [\[1\]](api/languages/cpp_api | /cpp_api.html#_CPPv4N5cudaq10prod |
| .html#_CPPv4NK5cudaq14complex_mat | uct_op10product_opERR9HandlerTy), |
| rixixERKNSt6vectorINSt6size_tEEE) |     [\[7\]](ap                    |
| -   [cudaq::complex_matrix::power | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq10product_op10product_opEd), |
|     function)]                    |     [\[8\]](a                     |
| (api/languages/cpp_api.html#_CPPv | pi/languages/cpp_api.html#_CPPv4N |
| 4N5cudaq14complex_matrix5powerEi) | 5cudaq10product_op10product_opEv) |
| -                                 | -   [cuda                         |
|  [cudaq::complex_matrix::set_zero | q::product_op::to_diagonal_matrix |
|     (C++                          |     (C++                          |
|     function)](ap                 |     function)](api/               |
| i/languages/cpp_api.html#_CPPv4N5 | languages/cpp_api.html#_CPPv4NK5c |
| cudaq14complex_matrix8set_zeroEv) | udaq10product_op18to_diagonal_mat |
| -                                 | rixENSt13unordered_mapINSt6size_t |
| [cudaq::complex_matrix::to_string | ENSt7int64_tEEERKNSt13unordered_m |
|     (C++                          | apINSt6stringENSt7complexIdEEEEb) |
|     function)](api/               | -   [cudaq::product_op::to_matrix |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14complex_matrix9to_stringEv) |     funct                         |
| -   [                             | ion)](api/languages/cpp_api.html# |
| cudaq::complex_matrix::value_type | _CPPv4NK5cudaq10product_op9to_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     type)](api/                   | ENSt7int64_tEEERKNSt13unordered_m |
| languages/cpp_api.html#_CPPv4N5cu | apINSt6stringENSt7complexIdEEEEb) |
| daq14complex_matrix10value_typeE) | -   [cu                           |
| -   [cudaq::contrib (C++          | daq::product_op::to_sparse_matrix |
|     type)](api/languages/cpp      |     (C++                          |
| _api.html#_CPPv4N5cudaq7contribE) |     function)](ap                 |
| -                                 | i/languages/cpp_api.html#_CPPv4NK |
| [cudaq::contrib::amplitude_encode | 5cudaq10product_op16to_sparse_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function)](api/language       | ENSt7int64_tEEERKNSt13unordered_m |
| s/cpp_api.html#_CPPv4N5cudaq7cont | apINSt6stringENSt7complexIdEEEEb) |
| rib16amplitude_encodeENSt4spanIKN | -   [cudaq::product_op::to_string |
| St7complexIdEEEENSt7complexIdEE), |     (C++                          |
|     [\[1\]](api/language          |     function)](                   |
| s/cpp_api.html#_CPPv4N5cudaq7cont | api/languages/cpp_api.html#_CPPv4 |
| rib16amplitude_encodeENSt4spanIKN | NK5cudaq10product_op9to_stringEv) |
| St7complexIfEEEENSt7complexIdEE), | -                                 |
|     [\[2\]                        |  [cudaq::product_op::\~product_op |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq7contrib16amplitude_enco |     fu                            |
| deENSt4spanIKdEENSt7complexIdEE), | nction)](api/languages/cpp_api.ht |
|     [\[3\]                        | ml#_CPPv4N5cudaq10product_opD0Ev) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::ptsbe (C++            |
| v4N5cudaq7contrib16amplitude_enco |     type)](api/languages/c        |
| deENSt4spanIKfEENSt7complexIdEE), | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
|                                   | -   [cudaq::p                     |
| [\[4\]](api/languages/cpp_api.htm | tsbe::ConditionalSamplingStrategy |
| l#_CPPv4N5cudaq7contrib16amplitud |     (C++                          |
| e_encodeERK5stateNSt7complexIdEE) |     class)](api/languag           |
| -                                 | es/cpp_api.html#_CPPv4N5cudaq5pts |
|   [cudaq::contrib::angular_encode | be27ConditionalSamplingStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::C              |
|                                   | onditionalSamplingStrategy::clone |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I0EN5cudaq7contrib14a |                                   |
| ngular_encodeEvRR6KernelR10QuakeV |    function)](api/languages/cpp_a |
| alueNSt4spanIKdEE12RotationAxis), | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
|     [\[1\]](api/languages/cpp_api | ditionalSamplingStrategy5cloneEv) |
| .html#_CPPv4I0EN5cudaq7contrib14a | -   [cuda                         |
| ngular_encodeEvRR6KernelR10QuakeV | q::ptsbe::ConditionalSamplingStra |
| alueR10QuakeValue12RotationAxis), | tegy::ConditionalSamplingStrategy |
|                                   |     (C++                          |
|   [\[2\]](api/languages/cpp_api.h |     function)](api/lang           |
| tml#_CPPv4I0EN5cudaq7contrib14ang | uages/cpp_api.html#_CPPv4N5cudaq5 |
| ular_encodeEvRR6KernelR10QuakeVal | ptsbe27ConditionalSamplingStrateg |
| ueRKNSt6vectorIdEE12RotationAxis) | y27ConditionalSamplingStrategyE19 |
| -   [cudaq::contrib::draw (C++    | TrajectoryPredicateNSt8uint64_tE) |
|     function)                     | -                                 |
| ](api/languages/cpp_api.html#_CPP |   [cudaq::ptsbe::ConditionalSampl |
| v4I0DpEN5cudaq7contrib4drawENSt6s | ingStrategy::generateTrajectories |
| tringERR13QuantumKernelDpRR4Args) |     (C++                          |
| -                                 |     function)](api/language       |
| [cudaq::contrib::get_unitary_cmat | s/cpp_api.html#_CPPv4NK5cudaq5pts |
|     (C++                          | be27ConditionalSamplingStrategy20 |
|     function)](api/languages/cp   | generateTrajectoriesENSt4spanIKN6 |
| p_api.html#_CPPv4I0DpEN5cudaq7con | detail10NoisePointEEENSt6size_tE) |
| trib16get_unitary_cmatE14complex_ | -   [cudaq::ptsbe::               |
| matrixRR13QuantumKernelDpRR4Args) | ConditionalSamplingStrategy::name |
| -   [cudaq::contrib::RotationAxis |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     enum)                         | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| ](api/languages/cpp_api.html#_CPP | nditionalSamplingStrategy4nameEv) |
| v4N5cudaq7contrib12RotationAxisE) | -   [cudaq:                       |
| -                                 | :ptsbe::ConditionalSamplingStrate |
|  [cudaq::contrib::RotationAxis::X | gy::\~ConditionalSamplingStrategy |
|     (C++                          |     (C++                          |
|     enumerator)](                 |     function)](api/languages/     |
| api/languages/cpp_api.html#_CPPv4 | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| N5cudaq7contrib12RotationAxis1XE) | 7ConditionalSamplingStrategyD0Ev) |
| -                                 | -                                 |
|  [cudaq::contrib::RotationAxis::Y | [cudaq::ptsbe::detail::NoisePoint |
|     (C++                          |     (C++                          |
|     enumerator)](                 |     struct)](a                    |
| api/languages/cpp_api.html#_CPPv4 | pi/languages/cpp_api.html#_CPPv4N |
| N5cudaq7contrib12RotationAxis1YE) | 5cudaq5ptsbe6detail10NoisePointE) |
| -                                 | -   [cudaq::p                     |
|  [cudaq::contrib::RotationAxis::Z | tsbe::detail::NoisePoint::channel |
|     (C++                          |     (C++                          |
|     enumerator)](                 |     member)](api/langu            |
| api/languages/cpp_api.html#_CPPv4 | ages/cpp_api.html#_CPPv4N5cudaq5p |
| N5cudaq7contrib12RotationAxis1ZE) | tsbe6detail10NoisePoint7channelE) |
| -   [cudaq::cudaq_json (C++       | -   [cudaq::ptsbe::det            |
|     class)](api/languages/cpp_api | ail::NoisePoint::circuit_location |
| .html#_CPPv4N5cudaq10cudaq_jsonE) |     (C++                          |
| -   [cudaq::DefaultQPU (C++       |     member)](api/languages/cpp_a  |
|     class)](api/languages/cpp_api | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| .html#_CPPv4N5cudaq10DefaultQPUE) | l10NoisePoint16circuit_locationE) |
| -   [cudaq::dem_from_kernel (C++  | -   [cudaq::p                     |
|     function)](api                | tsbe::detail::NoisePoint::op_name |
| /languages/cpp_api.html#_CPPv4I0D |     (C++                          |
| pEN5cudaq15dem_from_kernelENSt6st |     member)](api/langu            |
| ringERR13QuantumKernelDpRR4Args), | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     [                             | tsbe6detail10NoisePoint7op_nameE) |
| \[1\]](api/languages/cpp_api.html | -   [cudaq::                      |
| #_CPPv4I0DpEN5cudaq15dem_from_ker | ptsbe::detail::NoisePoint::qubits |
| nelENSt6stringERR13QuantumKernelP |     (C++                          |
| KN5cudaq11noise_modelEDpRR4Args), |     member)](api/lang             |
|     [\[2\]](api/languages/cp      | uages/cpp_api.html#_CPPv4N5cudaq5 |
| p_api.html#_CPPv4I0DpEN5cudaq15de | ptsbe6detail10NoisePoint6qubitsE) |
| m_from_kernelENSt6stringERR13Quan | -   [cudaq::                      |
| tumKernelPKN5cudaq11noise_modelER | ptsbe::ExhaustiveSamplingStrategy |
| KN5cudaq11dem_optionsEDpRR4Args), |     (C++                          |
|     [\[3\]](ap                    |     class)](api/langua            |
| i/languages/cpp_api.html#_CPPv4I0 | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| DpEN5cudaq15dem_from_kernelENSt6s | sbe26ExhaustiveSamplingStrategyE) |
| tringERR13QuantumKernelPKN5cudaq1 | -   [cudaq::ptsbe::               |
| 1noise_modelERKN5cudaq11dem_optio | ExhaustiveSamplingStrategy::clone |
| nsERN5cudaq15M2DSparseMatrixERN5c |     (C++                          |
| udaq15M2OSparseMatrixEDpRR4Args), |     function)](api/languages/cpp_ |
|     [\[4\]](api/language          | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| s/cpp_api.html#_CPPv4I0DpEN5cudaq | haustiveSamplingStrategy5cloneEv) |
| 15dem_from_kernelENSt6stringERR13 | -   [cu                           |
| QuantumKernelPKN5cudaq11noise_mod | daq::ptsbe::ExhaustiveSamplingStr |
| elERN5cudaq15M2DSparseMatrixERN5c | ategy::ExhaustiveSamplingStrategy |
| udaq15M2OSparseMatrixEDpRR4Args), |     (C++                          |
|     [\[5\]](api/languages/cpp_api |     function)](api/la             |
| .html#_CPPv4I0DpEN5cudaq15dem_fro | nguages/cpp_api.html#_CPPv4N5cuda |
| m_kernelENSt6stringERR13QuantumKe | q5ptsbe26ExhaustiveSamplingStrate |
| rnelRN5cudaq15M2DSparseMatrixERN5 | gy26ExhaustiveSamplingStrategyEv) |
| cudaq15M2OSparseMatrixEDpRR4Args) | -                                 |
| -   [cudaq::dem_options (C++      |    [cudaq::ptsbe::ExhaustiveSampl |
|                                   | ingStrategy::generateTrajectories |
|   struct)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq11dem_optionsE) |     function)](api/languag        |
| -   [cudaq::d                     | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| em_options::allow_gauge_detectors | sbe26ExhaustiveSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/language         | detail10NoisePointEEENSt6size_tE) |
| s/cpp_api.html#_CPPv4N5cudaq11dem | -   [cudaq::ptsbe:                |
| _options21allow_gauge_detectorsE) | :ExhaustiveSamplingStrategy::name |
| -   [cudaq::dem_options::appr     |     (C++                          |
| oximate_disjoint_errors_threshold |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq5ptsbe26E |
|     memb                          | xhaustiveSamplingStrategy4nameEv) |
| er)](api/languages/cpp_api.html#_ | -   [cuda                         |
| CPPv4N5cudaq11dem_options37approx | q::ptsbe::ExhaustiveSamplingStrat |
| imate_disjoint_errors_thresholdE) | egy::\~ExhaustiveSamplingStrategy |
| -   [cuda                         |     (C++                          |
| q::dem_options::block_decompositi |     function)](api/languages      |
| on_from_introducing_remnant_edges | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     (C++                          | 26ExhaustiveSamplingStrategyD0Ev) |
|     member)](api/lang             | -   [cuda                         |
| uages/cpp_api.html#_CPPv4N5cudaq1 | q::ptsbe::OrderedSamplingStrategy |
| 1dem_options50block_decomposition |     (C++                          |
| _from_introducing_remnant_edgesE) |     class)](api/lan               |
| -   [cud                          | guages/cpp_api.html#_CPPv4N5cudaq |
| aq::dem_options::decompose_errors | 5ptsbe23OrderedSamplingStrategyE) |
|     (C++                          | -   [cudaq::ptsb                  |
|     member)](api/lan              | e::OrderedSamplingStrategy::clone |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 11dem_options16decompose_errorsE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
|   [cudaq::dem_options::fold_loops | 3OrderedSamplingStrategy5cloneEv) |
|     (C++                          | -   [cudaq::ptsbe::OrderedSampl   |
|     member)](a                    | ingStrategy::generateTrajectories |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq11dem_options10fold_loopsE) |     function)](api/lang           |
| -   [cudaq::dem_optio             | uages/cpp_api.html#_CPPv4NK5cudaq |
| ns::ignore_decomposition_failures | 5ptsbe23OrderedSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/languages/cpp_ap | detail10NoisePointEEENSt6size_tE) |
| i.html#_CPPv4N5cudaq11dem_options | -   [cudaq::pts                   |
| 29ignore_decomposition_failuresE) | be::OrderedSamplingStrategy::name |
| -   [cudaq::dem_opt               |     (C++                          |
| ions::return_measurement_matrices |     function)](api/languages/     |
|     (C++                          | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     member)](api/languages/cpp_   | 23OrderedSamplingStrategy4nameEv) |
| api.html#_CPPv4N5cudaq11dem_optio | -                                 |
| ns27return_measurement_matricesE) |    [cudaq::ptsbe::OrderedSampling |
| -   [cudaq::depolarization1 (C++  | Strategy::OrderedSamplingStrategy |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     function)](                   |
| #_CPPv4N5cudaq15depolarization1E) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::depolarization2 (C++  | N5cudaq5ptsbe23OrderedSamplingStr |
|     c                             | ategy23OrderedSamplingStrategyEv) |
| lass)](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq15depolarization2E) |  [cudaq::ptsbe::OrderedSamplingSt |
| -   [cudaq:                       | rategy::\~OrderedSamplingStrategy |
| :depolarization2::depolarization2 |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/languages/cp   | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| p_api.html#_CPPv4N5cudaq15depolar | sbe23OrderedSamplingStrategyD0Ev) |
| ization215depolarization2EK4real) | -   [cudaq::pts                   |
| -   [cudaq                        | be::ProbabilisticSamplingStrategy |
| ::depolarization2::num_parameters |     (C++                          |
|     (C++                          |     class)](api/languages         |
|     member)](api/langu            | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| ages/cpp_api.html#_CPPv4N5cudaq15 | 29ProbabilisticSamplingStrategyE) |
| depolarization214num_parametersE) | -   [cudaq::ptsbe::Pro            |
| -   [cu                           | babilisticSamplingStrategy::clone |
| daq::depolarization2::num_targets |     (C++                          |
|     (C++                          |                                   |
|     member)](api/la               |  function)](api/languages/cpp_api |
| nguages/cpp_api.html#_CPPv4N5cuda | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| q15depolarization211num_targetsE) | bilisticSamplingStrategy5cloneEv) |
| -                                 | -                                 |
|    [cudaq::depolarization_channel | [cudaq::ptsbe::ProbabilisticSampl |
|     (C++                          | ingStrategy::generateTrajectories |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languages/     |
| N5cudaq22depolarization_channelE) | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| -   [cudaq::depol                 | 29ProbabilisticSamplingStrategy20 |
| arization_channel::num_parameters | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     member)](api/languages/cp     | -   [cudaq::ptsbe::Pr             |
| p_api.html#_CPPv4N5cudaq22depolar | obabilisticSamplingStrategy::name |
| ization_channel14num_parametersE) |     (C++                          |
| -   [cudaq::de                    |                                   |
| polarization_channel::num_targets |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
|     member)](api/languages        | abilisticSamplingStrategy4nameEv) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -   [cudaq::p                     |
| larization_channel11num_targetsE) | tsbe::ProbabilisticSamplingStrate |
| -   [cudaq::detail (C++           | gy::ProbabilisticSamplingStrategy |
|     type)](api/languages/cp       |     (C++                          |
| p_api.html#_CPPv4N5cudaq6detailE) |     function)]                    |
| -   [cudaq::detail::future (C++   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4N5cudaq5ptsbe29ProbabilisticSamp |
|   class)](api/languages/cpp_api.h | lingStrategy29ProbabilisticSampli |
| tml#_CPPv4N5cudaq6detail6futureE) | ngStrategyENSt8optionalINSt8uint6 |
| -                                 | 4_tEEENSt8optionalINSt6size_tEEE) |
|    [cudaq::detail::future::future | -   [cudaq::pts                   |
|     (C++                          | be::ProbabilisticSamplingStrategy |
|     functi                        | ::\~ProbabilisticSamplingStrategy |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq6detail6future6future |     function)](api/languages/cp   |
| ERNSt6vectorI3JobEERNSt6stringERN | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| St3mapINSt6stringENSt6stringEEE), | robabilisticSamplingStrategyD0Ev) |
|     [\[1\]](api/lan               | -                                 |
| guages/cpp_api.html#_CPPv4N5cudaq | [cudaq::ptsbe::PTSBEExecutionData |
| 6detail6future6futureERR6future), |     (C++                          |
|     [\[2\]                        |     struct)](ap                   |
| ](api/languages/cpp_api.html#_CPP | i/languages/cpp_api.html#_CPPv4N5 |
| v4N5cudaq6detail6future6futureEv) | cudaq5ptsbe18PTSBEExecutionDataE) |
| -   [c                            | -   [cudaq::ptsbe::PTSBE          |
| udaq::detail::kernel_builder_base | ExecutionData::count_instructions |
|     (C++                          |     (C++                          |
|     class)](api/                  |     function)](api/l              |
| languages/cpp_api.html#_CPPv4N5cu | anguages/cpp_api.html#_CPPv4NK5cu |
| daq6detail19kernel_builder_baseE) | daq5ptsbe18PTSBEExecutionData18co |
| -   [cudaq::detail::              | unt_instructionsE20TraceInstructi |
| kernel_builder_base::operator\<\< | onTypeNSt8optionalINSt6stringEEE) |
|     (C++                          | -   [cudaq::ptsbe::P              |
|     function)](api/langu          | TSBEExecutionData::get_trajectory |
| ages/cpp_api.html#_CPPv4N5cudaq6d |     (C++                          |
| etail19kernel_builder_baselsERNSt |     function                      |
| 7ostreamERK19kernel_builder_base) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| [cudaq::detail::KernelBuilderType | Data14get_trajectoryENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe:                |
|     class)](ap                    | :PTSBEExecutionData::instructions |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq6detail17KernelBuilderTypeE) |     member)](api/languages/cp     |
| -   [cudaq::                      | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| detail::KernelBuilderType::create | TSBEExecutionData12instructionsE) |
|     (C++                          | -   [cudaq::ptsbe:                |
|     function                      | :PTSBEExecutionData::trajectories |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq6detail17KernelBuilderT |     member)](api/languages/cp     |
| ype6createEPN4mlir11MLIRContextE) | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| -   [cudaq::detail::Ker           | TSBEExecutionData12trajectoriesE) |
| nelBuilderType::KernelBuilderType | -   [cudaq::ptsbe::PTSBEOptions   |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     struc                         |
| guages/cpp_api.html#_CPPv4N5cudaq | t)](api/languages/cpp_api.html#_C |
| 6detail17KernelBuilderType17Kerne | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| lBuilderTypeERRNSt8functionIFN4ml | -   [cudaq::ptsbe::PTSB           |
| ir4TypeEPN4mlir11MLIRContextEEEE) | EOptions::include_sequential_data |
| -   [cudaq::detector (C++         |     (C++                          |
|     function)](api                |                                   |
| /languages/cpp_api.html#_CPPv4IDp |    member)](api/languages/cpp_api |
| EN5cudaq8detectorEvDpRR8MeasArgs) | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| -   [cudaq::detectors (C++        | ptions23include_sequential_dataE) |
|     function)](api/languages/c    | -   [cudaq::ptsb                  |
| pp_api.html#_CPPv4N5cudaq9detecto | e::PTSBEOptions::max_trajectories |
| rsERKNSt6vectorI14measure_resultE |     (C++                          |
| ERKNSt6vectorI14measure_resultEE) |     member)](api/languages/       |
| -   [cudaq::diag_matrix_callback  | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
|     (C++                          | 2PTSBEOptions16max_trajectoriesE) |
|     class)                        | -   [cudaq::ptsbe::PT             |
| ](api/languages/cpp_api.html#_CPP | SBEOptions::return_execution_data |
| v4N5cudaq20diag_matrix_callbackE) |     (C++                          |
| -   [cudaq::dyn (C++              |     member)](api/languages/cpp_a  |
|     member)](api/languages        | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | EOptions21return_execution_dataE) |
| -   [cudaq::ExecutionContext (C++ | -   [cudaq::pts                   |
|     cl                            | be::PTSBEOptions::shot_allocation |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16ExecutionContextE) |     member)](api/languages        |
| -   [c                            | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| udaq::ExecutionContext::asyncExec | 12PTSBEOptions15shot_allocationE) |
|     (C++                          | -   [cud                          |
|     member)](api/                 | aq::ptsbe::PTSBEOptions::strategy |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9asyncExecE) |     member)](api/l                |
| -   [cud                          | anguages/cpp_api.html#_CPPv4N5cud |
| aq::ExecutionContext::asyncResult | aq5ptsbe12PTSBEOptions8strategyE) |
|     (C++                          | -   [cudaq::ptsbe::PTSBETrace     |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     t                             |
| 16ExecutionContext11asyncResultE) | ype)](api/languages/cpp_api.html# |
| -   [cudaq:                       | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| :ExecutionContext::batchIteration | -   [                             |
|     (C++                          | cudaq::ptsbe::PTSSamplingStrategy |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     class)](api                   |
| xecutionContext14batchIterationE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::E                     | udaq5ptsbe19PTSSamplingStrategyE) |
| xecutionContext::canHandleObserve | -   [cudaq::                      |
|     (C++                          | ptsbe::PTSSamplingStrategy::clone |
|     member)](api/language         |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     function)](api/languag        |
| cutionContext16canHandleObserveE) | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| -   [cudaq::Executio              | sbe19PTSSamplingStrategy5cloneEv) |
| nContext::deferredKernelException | -   [cudaq::ptsbe::PTSSampl       |
|     (C++                          | ingStrategy::generateTrajectories |
|     member)](api/languages/cpp_a  |     (C++                          |
| pi.html#_CPPv4N5cudaq16ExecutionC |     function)](api/               |
| ontext23deferredKernelExceptionE) | languages/cpp_api.html#_CPPv4NK5c |
| -   [cudaq::E                     | udaq5ptsbe19PTSSamplingStrategy20 |
| xecutionContext::ExecutionContext | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     func                          | -   [cudaq:                       |
| tion)](api/languages/cpp_api.html | :ptsbe::PTSSamplingStrategy::name |
| #_CPPv4N5cudaq16ExecutionContext1 |     (C++                          |
| 6ExecutionContextERKNSt6stringE), |     function)](api/langua         |
|     [\[1\]](api/languages/        | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| cpp_api.html#_CPPv4N5cudaq16Execu | tsbe19PTSSamplingStrategy4nameEv) |
| tionContext16ExecutionContextERKN | -   [cudaq::ptsbe::PTSSampli      |
| St6stringENSt6size_tENSt6size_tE) | ngStrategy::\~PTSSamplingStrategy |
| -   [cudaq::Execu                 |     (C++                          |
| tionContext::explicitMeasurements |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     member)](api/languages/cp     | q5ptsbe19PTSSamplingStrategyD0Ev) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::ptsbe::sample (C++    |
| onContext20explicitMeasurementsE) |                                   |
| -   [cuda                         |  function)](api/languages/cpp_api |
| q::ExecutionContext::futureResult | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
|     (C++                          | mpleE13sample_resultRK14sample_op |
|     member)](api/lang             | tionsRR13QuantumKernelDpRR4Args), |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     [\[1\]](api                   |
| 6ExecutionContext12futureResultE) | /languages/cpp_api.html#_CPPv4I0D |
| -   [cudaq::ExecutionContext      | pEN5cudaq5ptsbe6sampleE13sample_r |
| ::hasConditionalsOnMeasureResults | esultRKN5cudaq11noise_modelENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4Args) |
|     mem                           | -   [cudaq::ptsbe::sample_async   |
| ber)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16ExecutionContext31 |     function)](a                  |
| hasConditionalsOnMeasureResultsE) | pi/languages/cpp_api.html#_CPPv4I |
| -   [cudaq:                       | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| :ExecutionContext::inKernelLaunch | 9async_sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](api/langua           |     [\[1\]](api/languages/cp      |
| ges/cpp_api.html#_CPPv4N5cudaq16E | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| xecutionContext14inKernelLaunchE) | be12sample_asyncE19async_sample_r |
| -   [cu                           | esultRKN5cudaq11noise_modelENSt6s |
| daq::ExecutionContext::kernelName | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_options |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     struct)                       |
| q16ExecutionContext10kernelNameE) | ](api/languages/cpp_api.html#_CPP |
| -   [cud                          | v4N5cudaq5ptsbe14sample_optionsE) |
| aq::ExecutionContext::kernelTrace | -   [cudaq::ptsbe::sample_result  |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     class                         |
| guages/cpp_api.html#_CPPv4N5cudaq | )](api/languages/cpp_api.html#_CP |
| 16ExecutionContext11kernelTraceE) | Pv4N5cudaq5ptsbe13sample_resultE) |
| -                                 | -   [cudaq::pts                   |
|    [cudaq::ExecutionContext::name | be::sample_result::execution_data |
|     (C++                          |     (C++                          |
|     member)]                      |     function)](api/languages/c    |
| (api/languages/cpp_api.html#_CPPv | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| 4N5cudaq16ExecutionContext4nameE) | 3sample_result14execution_dataEv) |
| -   [cu                           | -   [cudaq::ptsbe::               |
| daq::ExecutionContext::noiseModel | sample_result::has_execution_data |
|     (C++                          |     (C++                          |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda |    function)](api/languages/cpp_a |
| q16ExecutionContext10noiseModelE) | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| -   [cudaq::Exe                   | ple_result18has_execution_dataEv) |
| cutionContext::numberTrajectories | -   [cudaq::pt                    |
|     (C++                          | sbe::sample_result::sample_result |
|     member)](api/languages/       |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16Execu |     function)](api/l              |
| tionContext18numberTrajectoriesE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [c                            | aq5ptsbe13sample_result13sample_r |
| udaq::ExecutionContext::optResult | esultERRN5cudaq13sample_resultE), |
|     (C++                          |                                   |
|     member)](api/                 |  [\[1\]](api/languages/cpp_api.ht |
| languages/cpp_api.html#_CPPv4N5cu | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| daq16ExecutionContext9optResultE) | sult13sample_resultERRN5cudaq13sa |
| -                                 | mple_resultE18PTSBEExecutionData) |
|   [cudaq::ExecutionContext::qpuId | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::set_execution_data |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/               |
| N5cudaq16ExecutionContext5qpuIdE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq                        | daq5ptsbe13sample_result18set_exe |
| ::ExecutionContext::registerNames | cution_dataE18PTSBEExecutionData) |
|     (C++                          | -   [cud                          |
|     member)](api/langu            | aq::ptsbe::ShotAllocationStrategy |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13registerNamesE) |     struct)](using                |
| -   [cu                           | /examples/ptsbe.html#_CPPv4N5cuda |
| daq::ExecutionContext::reorderIdx | q5ptsbe22ShotAllocationStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAllocatio  |
|     member)](api/la               | nStrategy::ShotAllocationStrategy |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10reorderIdxE) |     function)                     |
| -                                 | ](using/examples/ptsbe.html#_CPPv |
|   [cudaq::ExecutionContext::shots | 4N5cudaq5ptsbe22ShotAllocationStr |
|     (C++                          | ategy22ShotAllocationStrategyE4Ty |
|     member)](                     | pedNSt8optionalINSt8uint64_tEEE), |
| api/languages/cpp_api.html#_CPPv4 |     [\[1\                         |
| N5cudaq16ExecutionContext5shotsE) | ]](using/examples/ptsbe.html#_CPP |
| -   [cudaq::                      | v4N5cudaq5ptsbe22ShotAllocationSt |
| ExecutionContext::simulationState | rategy22ShotAllocationStrategyEv) |
|     (C++                          | -   [cudaq::pt                    |
|     member)](api/languag          | sbe::ShotAllocationStrategy::Type |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15simulationStateE) |     enum)](using/exam             |
| -                                 | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|    [cudaq::ExecutionContext::spin | be22ShotAllocationStrategy4TypeE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAllocatio  |
|     member)]                      | nStrategy::Type::HIGH_WEIGHT_BIAS |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq16ExecutionContext4spinE) |     enumerat                      |
| -   [cudaq::                      | or)](using/examples/ptsbe.html#_C |
| ExecutionContext::totalIterations | PPv4N5cudaq5ptsbe22ShotAllocation |
|     (C++                          | Strategy4Type16HIGH_WEIGHT_BIASE) |
|     member)](api/languag          | -   [cudaq::ptsbe::ShotAllocati   |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | onStrategy::Type::LOW_WEIGHT_BIAS |
| ecutionContext15totalIterationsE) |     (C++                          |
| -   [cudaq::ExecutionResult (C++  |     enumera                       |
|     st                            | tor)](using/examples/ptsbe.html#_ |
| ruct)](api/languages/cpp_api.html | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| #_CPPv4N5cudaq15ExecutionResultE) | nStrategy4Type15LOW_WEIGHT_BIASE) |
| -   [cud                          | -   [cudaq::ptsbe::ShotAlloc      |
| aq::ExecutionResult::appendResult | ationStrategy::Type::PROPORTIONAL |
|     (C++                          |     (C++                          |
|     functio                       |     enum                          |
| n)](api/languages/cpp_api.html#_C | erator)](using/examples/ptsbe.htm |
| PPv4N5cudaq15ExecutionResult12app | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| endResultENSt6stringENSt6size_tE) | tionStrategy4Type12PROPORTIONALE) |
| -   [cu                           | -   [cudaq::ptsbe::Shot           |
| daq::ExecutionResult::deserialize | AllocationStrategy::Type::UNIFORM |
|     (C++                          |     (C++                          |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP |   enumerator)](using/examples/pts |
| v4N5cudaq15ExecutionResult11deser | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| ializeERNSt6vectorINSt6size_tEEE) | AllocationStrategy4Type7UNIFORME) |
| -   [cudaq:                       | -                                 |
| :ExecutionResult::ExecutionResult |   [cudaq::ptsbe::TraceInstruction |
|     (C++                          |     (C++                          |
|     functio                       |     struct)](                     |
| n)](api/languages/cpp_api.html#_C | api/languages/cpp_api.html#_CPPv4 |
| PPv4N5cudaq15ExecutionResult15Exe | N5cudaq5ptsbe16TraceInstructionE) |
| cutionResultE16CountsDictionary), | -   [cudaq:                       |
|     [\[1\]](api/lan               | :ptsbe::TraceInstruction::channel |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult15ExecutionResul |     member)](api/lang             |
| tE16CountsDictionaryNSt6stringE), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     [\[2\                         | ptsbe16TraceInstruction7channelE) |
| ]](api/languages/cpp_api.html#_CP | -   [cudaq::                      |
| Pv4N5cudaq15ExecutionResult15Exec | ptsbe::TraceInstruction::controls |
| utionResultE16CountsDictionaryd), |     (C++                          |
|                                   |     member)](api/langu            |
|    [\[3\]](api/languages/cpp_api. | ages/cpp_api.html#_CPPv4N5cudaq5p |
| html#_CPPv4N5cudaq15ExecutionResu | tsbe16TraceInstruction8controlsE) |
| lt15ExecutionResultENSt6stringE), | -   [cud                          |
|     [\[4\                         | aq::ptsbe::TraceInstruction::name |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     member)](api/l                |
| utionResultERK15ExecutionResult), | anguages/cpp_api.html#_CPPv4N5cud |
|     [\[5\                         | aq5ptsbe16TraceInstruction4nameE) |
| ]](api/languages/cpp_api.html#_CP | -   [cudaq                        |
| Pv4N5cudaq15ExecutionResult15Exec | ::ptsbe::TraceInstruction::params |
| utionResultERR15ExecutionResult), |     (C++                          |
|     [\[6\]](api/language          |     member)](api/lan              |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | guages/cpp_api.html#_CPPv4N5cudaq |
| cutionResult15ExecutionResultEd), | 5ptsbe16TraceInstruction6paramsE) |
|     [\[7\]](api/languag           | -   [cudaq:                       |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | :ptsbe::TraceInstruction::targets |
| ecutionResult15ExecutionResultEv) |     (C++                          |
| -   [                             |     member)](api/lang             |
| cudaq::ExecutionResult::operator= | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     (C++                          | ptsbe16TraceInstruction7targetsE) |
|     function)](api/languages/c    | -   [cudaq::ptsbe::T              |
| pp_api.html#_CPPv4N5cudaq15Execut | raceInstruction::TraceInstruction |
| ionResultaSERK15ExecutionResult), |     (C++                          |
|     [\[1\]](api/languages/        |                                   |
| cpp_api.html#_CPPv4N5cudaq15Execu |   function)](api/languages/cpp_ap |
| tionResultaSERR15ExecutionResult) | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| -   [c                            | Instruction16TraceInstructionE20T |
| udaq::ExecutionResult::operator== | raceInstructionTypeNSt6stringENSt |
|     (C++                          | 6vectorINSt6size_tEEENSt6vectorIN |
|     function)](api/languages/c    | St6size_tEEENSt6vectorIdEENSt8opt |
| pp_api.html#_CPPv4NK5cudaq15Execu | ionalIN5cudaq13kraus_channelEEE), |
| tionResulteqERK15ExecutionResult) |     [\[1\]](api/languages/cpp_a   |
| -   [cud                          | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| aq::ExecutionResult::registerName | eInstruction16TraceInstructionEv) |
|     (C++                          | -   [cud                          |
|     member)](api/lan              | aq::ptsbe::TraceInstruction::type |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult12registerNameE) |     member)](api/l                |
| -   [cudaq                        | anguages/cpp_api.html#_CPPv4N5cud |
| ::ExecutionResult::sequentialData | aq5ptsbe16TraceInstruction4typeE) |
|     (C++                          | -   [c                            |
|     member)](api/langu            | udaq::ptsbe::TraceInstructionType |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| ExecutionResult14sequentialDataE) |     enum)](api/                   |
| -   [                             | languages/cpp_api.html#_CPPv4N5cu |
| cudaq::ExecutionResult::serialize | daq5ptsbe20TraceInstructionTypeE) |
|     (C++                          | -   [cudaq::                      |
|     function)](api/l              | ptsbe::TraceInstructionType::Gate |
| anguages/cpp_api.html#_CPPv4NK5cu |     (C++                          |
| daq15ExecutionResult9serializeEv) |     enumerator)](api/langu        |
| -   [cudaq::fermion_handler (C++  | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     c                             | tsbe20TraceInstructionType4GateE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::ptsbe::               |
| #_CPPv4N5cudaq15fermion_handlerE) | TraceInstructionType::Measurement |
| -   [cudaq::fermion_op (C++       |     (C++                          |
|     type)](api/languages/cpp_api  |                                   |
| .html#_CPPv4N5cudaq10fermion_opE) |    enumerator)](api/languages/cpp |
| -   [cudaq::fermion_op_term (C++  | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
|                                   | aceInstructionType11MeasurementE) |
| type)](api/languages/cpp_api.html | -   [cudaq::p                     |
| #_CPPv4N5cudaq15fermion_op_termE) | tsbe::TraceInstructionType::Noise |
| -   [cudaq::FermioniqQPU (C++     |     (C++                          |
|                                   |     enumerator)](api/langua       |
|   class)](api/languages/cpp_api.h | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | sbe20TraceInstructionType5NoiseE) |
| -   [cudaq::get_state (C++        | -   [                             |
|                                   | cudaq::ptsbe::TrajectoryPredicate |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     type)](api                    |
| ateEDaRR13QuantumKernelDpRR4Args) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::GPUEmulatedQPU (C++   | udaq5ptsbe19TrajectoryPredicateE) |
|                                   | -   [cudaq::QPU (C++              |
| class)](api/languages/cpp_api.htm |     class)](api/languages         |
| l#_CPPv4N5cudaq14GPUEmulatedQPUE) | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| -   [cudaq::gradient (C++         | -   [cudaq::QPU::beginExecution   |
|     class)](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4N5cudaq8gradientE) |     function                      |
| -   [cudaq::gradient::clone (C++  | )](api/languages/cpp_api.html#_CP |
|     fun                           | Pv4N5cudaq3QPU14beginExecutionEv) |
| ction)](api/languages/cpp_api.htm | -   [cuda                         |
| l#_CPPv4N5cudaq8gradient5cloneEv) | q::QPU::configureExecutionContext |
| -   [cudaq::gradient::compute     |     (C++                          |
|     (C++                          |     funct                         |
|     function)](api/language       | ion)](api/languages/cpp_api.html# |
| s/cpp_api.html#_CPPv4N5cudaq8grad | _CPPv4NK5cudaq3QPU25configureExec |
| ient7computeERKNSt6vectorIdEERKNS | utionContextER16ExecutionContext) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::QPU::endExecution     |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     functi                        |
| cudaq8gradient7computeERKNSt6vect | on)](api/languages/cpp_api.html#_ |
| orIdEERNSt6vectorIdEERK7spin_opd) | CPPv4N5cudaq3QPU12endExecutionEv) |
| -   [cudaq::gradient::gradient    | -   [cudaq::QPU::enqueue (C++     |
|     (C++                          |     function)](ap                 |
|     function)](api/lang           | i/languages/cpp_api.html#_CPPv4N5 |
| uages/cpp_api.html#_CPPv4I00EN5cu | cudaq3QPU7enqueueER11QuantumTask) |
| daq8gradient8gradientER7KernelT), | -   [cud                          |
|                                   | aq::QPU::finalizeExecutionContext |
|    [\[1\]](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4I00EN5cudaq8gradient8g |     func                          |
| radientER7KernelTRR10ArgsMapper), | tion)](api/languages/cpp_api.html |
|     [\[2\                         | #_CPPv4NK5cudaq3QPU24finalizeExec |
| ]](api/languages/cpp_api.html#_CP | utionContextER16ExecutionContext) |
| Pv4I00EN5cudaq8gradient8gradientE | -   [cudaq::QPU::getCompileTarget |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3                          |     function)](api/languages/c    |
| \]](api/languages/cpp_api.html#_C | pp_api.html#_CPPv4N5cudaq3QPU16ge |
| PPv4N5cudaq8gradient8gradientERRN | tCompileTargetEPK13RuntimeTarget) |
| St8functionIFvNSt6vectorIdEEEEE), | -                                 |
|     [\[                           | [cudaq::QPU::getExecutionThreadId |
| 4\]](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8gradient8gradientEv) |     function)](api/               |
| -   [cudaq::gradient::setArgs     | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq3QPU20getExecutionThreadIdEv) |
|     fu                            | -   [cudaq::QPU::isEmulated (C++  |
| nction)](api/languages/cpp_api.ht |     func                          |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | tion)](api/languages/cpp_api.html |
| tArgsEvR13QuantumKernelDpRR4Args) | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| -   [cudaq::gradient::setKernel   | -   [cudaq::QPU::isSimulator (C++ |
|     (C++                          |     funct                         |
|     function)](api/languages/c    | ion)](api/languages/cpp_api.html# |
| pp_api.html#_CPPv4I0EN5cudaq8grad | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| ient9setKernelEvR13QuantumKernel) | -   [cudaq::QPU::onRandomSeedSet  |
| -   [cud                          |     (C++                          |
| aq::gradients::central_difference |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq3 |
|     class)](api/la                | QPU15onRandomSeedSetENSt6size_tE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QPU::QPU (C++         |
| q9gradients18central_differenceE) |     functio                       |
| -   [cudaq::gra                   | n)](api/languages/cpp_api.html#_C |
| dients::central_difference::clone | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
|     (C++                          |                                   |
|     function)](api/languages      |  [\[1\]](api/languages/cpp_api.ht |
| /cpp_api.html#_CPPv4N5cudaq9gradi | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| ents18central_difference5cloneEv) |     [\[2\]](api/languages/cpp_    |
| -   [cudaq::gradi                 | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| ents::central_difference::compute | -   [cudaq::QPU::setId (C++       |
|     (C++                          |     function                      |
|     function)](                   | )](api/languages/cpp_api.html#_CP |
| api/languages/cpp_api.html#_CPPv4 | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| N5cudaq9gradients18central_differ | -   [cudaq::QPU::setShots (C++    |
| ence7computeERKNSt6vectorIdEERKNS |     f                             |
| t8functionIFdNSt6vectorIdEEEEEd), | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::QPU::\~QPU (C++       |
| tml#_CPPv4N5cudaq9gradients18cent |     function)](api/languages/cp   |
| ral_difference7computeERKNSt6vect | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QPUState (C++         |
| -   [cudaq::gradie                |     class)](api/languages/cpp_    |
| nts::central_difference::gradient | api.html#_CPPv4N5cudaq8QPUStateE) |
|     (C++                          | -   [cudaq::qreg (C++             |
|     functio                       |     class)](api/lan               |
| n)](api/languages/cpp_api.html#_C | guages/cpp_api.html#_CPPv4I_NSt6s |
| PPv4I00EN5cudaq9gradients18centra | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| l_difference8gradientER7KernelT), | -   [cudaq::qreg::back (C++       |
|     [\[1\]](api/langua            |     function)                     |
| ges/cpp_api.html#_CPPv4I00EN5cuda | ](api/languages/cpp_api.html#_CPP |
| q9gradients18central_difference8g | v4N5cudaq4qreg4backENSt6size_tE), |
| radientER7KernelTRR10ArgsMapper), |     [\[1\]](api/languages/cpp_ap  |
|     [\[2\]](api/languages/cpp_    | i.html#_CPPv4N5cudaq4qreg4backEv) |
| api.html#_CPPv4I00EN5cudaq9gradie | -   [cudaq::qreg::begin (C++      |
| nts18central_difference8gradientE |                                   |
| RR13QuantumKernelRR10ArgsMapper), |  function)](api/languages/cpp_api |
|     [\[3\]](api/languages/cpp     | .html#_CPPv4N5cudaq4qreg5beginEv) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::qreg::clear (C++      |
| 18central_difference8gradientERRN |                                   |
| St8functionIFvNSt6vectorIdEEEEE), |  function)](api/languages/cpp_api |
|     [\[4\]](api/languages/cp      | .html#_CPPv4N5cudaq4qreg5clearEv) |
| p_api.html#_CPPv4N5cudaq9gradient | -   [cudaq::qreg::front (C++      |
| s18central_difference8gradientEv) |     function)]                    |
| -   [cud                          | (api/languages/cpp_api.html#_CPPv |
| aq::gradients::forward_difference | 4N5cudaq4qreg5frontENSt6size_tE), |
|     (C++                          |     [\[1\]](api/languages/cpp_api |
|     class)](api/la                | .html#_CPPv4N5cudaq4qreg5frontEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::qreg::operator\[\]    |
| q9gradients18forward_differenceE) |     (C++                          |
| -   [cudaq::gra                   |     functi                        |
| dients::forward_difference::clone | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq4qregixEKNSt6size_tE) |
|     function)](api/languages      | -   [cudaq::qreg::qreg (C++       |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)                     |
| ents18forward_difference5cloneEv) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::gradi                 | v4N5cudaq4qreg4qregENSt6size_tE), |
| ents::forward_difference::compute |     [\[1\]](api/languages/cpp_ap  |
|     (C++                          | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     function)](                   | -   [cudaq::qreg::size (C++       |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq9gradients18forward_differ |  function)](api/languages/cpp_api |
| ence7computeERKNSt6vectorIdEERKNS | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::qreg::slice (C++      |
|                                   |     function)](api/langu          |
|   [\[1\]](api/languages/cpp_api.h | ages/cpp_api.html#_CPPv4N5cudaq4q |
| tml#_CPPv4N5cudaq9gradients18forw | reg5sliceENSt6size_tENSt6size_tE) |
| ard_difference7computeERKNSt6vect | -   [cudaq::qreg::value_type (C++ |
| orIdEERNSt6vectorIdEERK7spin_opd) |                                   |
| -   [cudaq::gradie                | type)](api/languages/cpp_api.html |
| nts::forward_difference::gradient | #_CPPv4N5cudaq4qreg10value_typeE) |
|     (C++                          | -   [cudaq::qspan (C++            |
|     functio                       |     class)](api/lang              |
| n)](api/languages/cpp_api.html#_C | uages/cpp_api.html#_CPPv4I_NSt6si |
| PPv4I00EN5cudaq9gradients18forwar | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| d_difference8gradientER7KernelT), | -   [cudaq::QuakeValue (C++       |
|     [\[1\]](api/langua            |     class)](api/languages/cpp_api |
| ges/cpp_api.html#_CPPv4I00EN5cuda | .html#_CPPv4N5cudaq10QuakeValueE) |
| q9gradients18forward_difference8g | -   [cudaq::Q                     |
| radientER7KernelTRR10ArgsMapper), | uakeValue::canValidateNumElements |
|     [\[2\]](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4I00EN5cudaq9gradie |     function)](api/languages      |
| nts18forward_difference8gradientE | /cpp_api.html#_CPPv4N5cudaq10Quak |
| RR13QuantumKernelRR10ArgsMapper), | eValue22canValidateNumElementsEv) |
|     [\[3\]](api/languages/cpp     | -                                 |
| _api.html#_CPPv4N5cudaq9gradients |  [cudaq::QuakeValue::constantSize |
| 18forward_difference8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](api                |
|     [\[4\]](api/languages/cp      | /languages/cpp_api.html#_CPPv4N5c |
| p_api.html#_CPPv4N5cudaq9gradient | udaq10QuakeValue12constantSizeEv) |
| s18forward_difference8gradientEv) | -   [cudaq::QuakeValue::dump (C++ |
| -   [                             |     function)](api/lan            |
| cudaq::gradients::parameter_shift | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 10QuakeValue4dumpERNSt7ostreamE), |
|     class)](api                   |     [\                            |
| /languages/cpp_api.html#_CPPv4N5c | [1\]](api/languages/cpp_api.html# |
| udaq9gradients15parameter_shiftE) | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| -   [cudaq::                      | -   [cudaq                        |
| gradients::parameter_shift::clone | ::QuakeValue::getRequiredElements |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/langua         |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| adients15parameter_shift5cloneEv) | uakeValue19getRequiredElementsEv) |
| -   [cudaq::gr                    | -   [cudaq::QuakeValue::getValue  |
| adients::parameter_shift::compute |     (C++                          |
|     (C++                          |     function)]                    |
|     function                      | (api/languages/cpp_api.html#_CPPv |
| )](api/languages/cpp_api.html#_CP | 4NK5cudaq10QuakeValue8getValueEv) |
| Pv4N5cudaq9gradients15parameter_s | -   [cudaq::QuakeValue::inverse   |
| hift7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)                     |
|     [\[1\]](api/languages/cpp_ap  | ](api/languages/cpp_api.html#_CPP |
| i.html#_CPPv4N5cudaq9gradients15p | v4NK5cudaq10QuakeValue7inverseEv) |
| arameter_shift7computeERKNSt6vect | -                                 |
| orIdEERNSt6vectorIdEERK7spin_opd) |    [cudaq::QuakeValue::isSequence |
| -   [cudaq::gra                   |     (C++                          |
| dients::parameter_shift::gradient |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     func                          | 5cudaq10QuakeValue10isSequenceEv) |
| tion)](api/languages/cpp_api.html | -                                 |
| #_CPPv4I00EN5cudaq9gradients15par |    [cudaq::QuakeValue::operator\* |
| ameter_shift8gradientER7KernelT), |     (C++                          |
|     [\[1\]](api/lan               |     function)](api                |
| guages/cpp_api.html#_CPPv4I00EN5c | /languages/cpp_api.html#_CPPv4N5c |
| udaq9gradients15parameter_shift8g | udaq10QuakeValuemlE10QuakeValue), |
| radientER7KernelTRR10ArgsMapper), |                                   |
|     [\[2\]](api/languages/c       | [\[1\]](api/languages/cpp_api.htm |
| pp_api.html#_CPPv4I00EN5cudaq9gra | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| dients15parameter_shift8gradientE | -   [cudaq::QuakeValue::operator+ |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3\]](api/languages/        |     function)](api                |
| cpp_api.html#_CPPv4N5cudaq9gradie | /languages/cpp_api.html#_CPPv4N5c |
| nts15parameter_shift8gradientERRN | udaq10QuakeValueplE10QuakeValue), |
| St8functionIFvNSt6vectorIdEEEEE), |     [                             |
|     [\[4\]](api/languages         | \[1\]](api/languages/cpp_api.html |
| /cpp_api.html#_CPPv4N5cudaq9gradi | #_CPPv4N5cudaq10QuakeValueplEKd), |
| ents15parameter_shift8gradientEv) |                                   |
| -   [cudaq::kernel_builder (C++   | [\[2\]](api/languages/cpp_api.htm |
|     clas                          | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| s)](api/languages/cpp_api.html#_C | -   [cudaq::QuakeValue::operator- |
| PPv4IDpEN5cudaq14kernel_builderE) |     (C++                          |
| -   [c                            |     function)](api                |
| udaq::kernel_builder::constantVal | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuemiE10QuakeValue), |
|     function)](api/la             |     [                             |
| nguages/cpp_api.html#_CPPv4N5cuda | \[1\]](api/languages/cpp_api.html |
| q14kernel_builder11constantValEd) | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| -                                 |     [                             |
|  [cudaq::kernel_builder::detector | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|                                   |                                   |
|    function)](api/languages/cpp_a | [\[3\]](api/languages/cpp_api.htm |
| pi.html#_CPPv4IDpEN5cudaq14kernel | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| _builder8detectorEvDpRR8MeasArgs) | -   [cudaq::QuakeValue::operator/ |
| -                                 |     (C++                          |
| [cudaq::kernel_builder::detectors |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     func                          | udaq10QuakeValuedvE10QuakeValue), |
| tion)](api/languages/cpp_api.html |                                   |
| #_CPPv4N5cudaq14kernel_builder9de | [\[1\]](api/languages/cpp_api.htm |
| tectorsE10QuakeValue10QuakeValue) | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| -   [cu                           | -                                 |
| daq::kernel_builder::getArguments |  [cudaq::QuakeValue::operator\[\] |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)](api                |
| guages/cpp_api.html#_CPPv4N5cudaq | /languages/cpp_api.html#_CPPv4N5c |
| 14kernel_builder12getArgumentsEv) | udaq10QuakeValueixEKNSt6size_tE), |
| -   [cu                           |     [\[1\]](api/                  |
| daq::kernel_builder::getNumParams | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq10QuakeValueixERK10QuakeValue) |
|     function)](api/lan            | -                                 |
| guages/cpp_api.html#_CPPv4N5cudaq |    [cudaq::QuakeValue::QuakeValue |
| 14kernel_builder12getNumParamsEv) |     (C++                          |
| -   [cud                          |     function)](api/languag        |
| aq::kernel_builder::isArgSequence | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     (C++                          | akeValue10QuakeValueERN4mlir20Imp |
|     function)](api/languages/cpp_ | licitLocOpBuilderEN4mlir5ValueE), |
| api.html#_CPPv4N5cudaq14kernel_bu |     [\[1\]                        |
| ilder13isArgSequenceENSt6size_tE) | ](api/languages/cpp_api.html#_CPP |
| -   [cuda                         | v4N5cudaq10QuakeValue10QuakeValue |
| q::kernel_builder::kernel_builder | ERN4mlir20ImplicitLocOpBuilderEd) |
|     (C++                          | -   [cudaq::QuakeValue::size (C++ |
|     function)](api/languages/cpp  |     funct                         |
| _api.html#_CPPv4N5cudaq14kernel_b | ion)](api/languages/cpp_api.html# |
| uilder14kernel_builderERNSt6vecto | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| rIN6detail17KernelBuilderTypeEEE) | -   [cudaq::QuakeValue::slice     |
| -   [cudaq::k                     |     (C++                          |
| ernel_builder::logical_observable |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq10QuakeValu |
|     function)                     | e5sliceEKNSt6size_tEKNSt6size_tE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::quantum_platform (C++ |
| v4IDpEN5cudaq14kernel_builder18lo |     cl                            |
| gical_observableEvDpRR8MeasArgs), | ass)](api/languages/cpp_api.html# |
|     [\[1\]](ap                    | _CPPv4N5cudaq16quantum_platformE) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq:                       |
| cudaq14kernel_builder18logical_ob | :quantum_platform::beginExecution |
| servableE10QuakeValueNSt6size_tE) |     (C++                          |
| -   [cudaq::kernel_builder::name  |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     function)                     | antum_platform14beginExecutionEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::quantum_pl            |
| v4N5cudaq14kernel_builder4nameEv) | atform::configureExecutionContext |
| -                                 |     (C++                          |
|    [cudaq::kernel_builder::qalloc |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4NK5cudaq |
|     function)](api/language       | 16quantum_platform25configureExec |
| s/cpp_api.html#_CPPv4N5cudaq14ker | utionContextER16ExecutionContext) |
| nel_builder6qallocE10QuakeValue), | -   [cuda                         |
|     [\[1\]](api/language          | q::quantum_platform::endExecution |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     (C++                          |
| nel_builder6qallocEKNSt6size_tE), |     function)](api/langu          |
|     [\[2                          | ages/cpp_api.html#_CPPv4N5cudaq16 |
| \]](api/languages/cpp_api.html#_C | quantum_platform12endExecutionEv) |
| PPv4N5cudaq14kernel_builder6qallo | -   [cudaq::q                     |
| cERNSt6vectorINSt7complexIdEEEE), | uantum_platform::enqueueAsyncTask |
|     [\[3\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languages/     |
| N5cudaq14kernel_builder6qallocEv) | cpp_api.html#_CPPv4N5cudaq16quant |
| -   [cudaq::kernel_builder::swap  | um_platform16enqueueAsyncTaskEKNS |
|     (C++                          | t6size_tER19KernelExecutionTask), |
|     function)](api/language       |     [\[1\]](api/languag           |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | es/cpp_api.html#_CPPv4N5cudaq16qu |
| 4kernel_builder4swapEvRK10QuakeVa | antum_platform16enqueueAsyncTaskE |
| lueRK10QuakeValueRK10QuakeValue), | KNSt6size_tERNSt8functionIFvvEEE) |
|                                   | -   [cudaq::quantum_p             |
| [\[1\]](api/languages/cpp_api.htm | latform::finalizeExecutionContext |
| l#_CPPv4I00EN5cudaq14kernel_build |     (C++                          |
| er4swapEvRKNSt6vectorI10QuakeValu |     function)](api/languages/c    |
| eEERK10QuakeValueRK10QuakeValue), | pp_api.html#_CPPv4NK5cudaq16quant |
|                                   | um_platform24finalizeExecutionCon |
| [\[2\]](api/languages/cpp_api.htm | textERN5cudaq16ExecutionContextE) |
| l#_CPPv4N5cudaq14kernel_builder4s | -   [cudaq::qua                   |
| wapERK10QuakeValueRK10QuakeValue) | ntum_platform::get_codegen_config |
| -   [cudaq::KernelExecutionTask   |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     type                          | pp_api.html#_CPPv4N5cudaq16quantu |
| )](api/languages/cpp_api.html#_CP | m_platform18get_codegen_configEv) |
| Pv4N5cudaq19KernelExecutionTaskE) | -   [cuda                         |
| -   [cudaq::KernelThunkResultType | q::quantum_platform::get_exec_ctx |
|     (C++                          |     (C++                          |
|     struct)]                      |     function)](api/langua         |
| (api/languages/cpp_api.html#_CPPv | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| 4N5cudaq21KernelThunkResultTypeE) | quantum_platform12get_exec_ctxEv) |
| -   [cudaq::KernelThunkType (C++  | -   [c                            |
|                                   | udaq::quantum_platform::get_noise |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     function)](api/languages/c    |
| -   [cudaq::kraus_channel (C++    | pp_api.html#_CPPv4N5cudaq16quantu |
|                                   | m_platform9get_noiseENSt6size_tE) |
|  class)](api/languages/cpp_api.ht | -   [cudaq::qua                   |
| ml#_CPPv4N5cudaq13kraus_channelE) | ntum_platform::get_runtime_target |
| -   [cudaq::kraus_channel::empty  |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     function)]                    | p_api.html#_CPPv4NK5cudaq16quantu |
| (api/languages/cpp_api.html#_CPPv | m_platform18get_runtime_targetEv) |
| 4NK5cudaq13kraus_channel5emptyEv) | -   [cud                          |
| -   [cudaq::kraus_c               | aq::quantum_platform::is_emulated |
| hannel::generateUnitaryParameters |     (C++                          |
|     (C++                          |                                   |
|                                   |    function)](api/languages/cpp_a |
|    function)](api/languages/cpp_a | pi.html#_CPPv4NK5cudaq16quantum_p |
| pi.html#_CPPv4N5cudaq13kraus_chan | latform11is_emulatedENSt6size_tE) |
| nel25generateUnitaryParametersEv) | -   [cudaq::                      |
| -                                 | quantum_platform::is_library_mode |
|    [cudaq::kraus_channel::get_ops |     (C++                          |
|     (C++                          |     function)](api/languages      |
|     function)](a                  | /cpp_api.html#_CPPv4NK5cudaq16qua |
| pi/languages/cpp_api.html#_CPPv4N | ntum_platform15is_library_modeEv) |
| K5cudaq13kraus_channel7get_opsEv) | -   [c                            |
| -   [cud                          | udaq::quantum_platform::is_remote |
| aq::kraus_channel::identity_flags |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api/lan              | p_api.html#_CPPv4NK5cudaq16quantu |
| guages/cpp_api.html#_CPPv4N5cudaq | m_platform9is_remoteENSt6size_tE) |
| 13kraus_channel14identity_flagsE) | -   [cuda                         |
| -   [cud                          | q::quantum_platform::is_simulator |
| aq::kraus_channel::is_identity_op |     (C++                          |
|     (C++                          |                                   |
|                                   |   function)](api/languages/cpp_ap |
|    function)](api/languages/cpp_a | i.html#_CPPv4NK5cudaq16quantum_pl |
| pi.html#_CPPv4NK5cudaq13kraus_cha | atform12is_simulatorENSt6size_tE) |
| nnel14is_identity_opENSt6size_tE) | -   [cudaq:                       |
| -   [cudaq::                      | :quantum_platform::list_platforms |
| kraus_channel::is_unitary_mixture |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/languages      | es/cpp_api.html#_CPPv4N5cudaq16qu |
| /cpp_api.html#_CPPv4NK5cudaq13kra | antum_platform14list_platformsEv) |
| us_channel18is_unitary_mixtureEv) | -                                 |
| -   [cu                           |    [cudaq::quantum_platform::name |
| daq::kraus_channel::kraus_channel |     (C++                          |
|     (C++                          |     function)](a                  |
|     function)](api/lang           | pi/languages/cpp_api.html#_CPPv4N |
| uages/cpp_api.html#_CPPv4IDpEN5cu | K5cudaq16quantum_platform4nameEv) |
| daq13kraus_channel13kraus_channel | -   [                             |
| EDpRRNSt16initializer_listI1TEE), | cudaq::quantum_platform::num_qpus |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     function)](api/l              |
| ml#_CPPv4N5cudaq13kraus_channel13 | anguages/cpp_api.html#_CPPv4NK5cu |
| kraus_channelERK13kraus_channel), | daq16quantum_platform8num_qpusEv) |
|     [\[2\]                        | -   [cudaq::                      |
| ](api/languages/cpp_api.html#_CPP | quantum_platform::onRandomSeedSet |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERKNSt6vectorI8kraus_opEE), |                                   |
|     [\[3\]                        | function)](api/languages/cpp_api. |
| ](api/languages/cpp_api.html#_CPP | html#_CPPv4N5cudaq16quantum_platf |
| v4N5cudaq13kraus_channel13kraus_c | orm15onRandomSeedSetENSt6size_tE) |
| hannelERRNSt6vectorI8kraus_opEE), | -   [cudaq::q                     |
|     [\[4\]](api/lan               | uantum_platform::prepareExecution |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 13kraus_channel13kraus_channelEv) |     fu                            |
| -                                 | nction)](api/languages/cpp_api.ht |
| [cudaq::kraus_channel::noise_type | ml#_CPPv4NK5cudaq16quantum_platfo |
|     (C++                          | rm16prepareExecutionENSt6size_tE) |
|     member)](api                  | -   [cudaq:                       |
| /languages/cpp_api.html#_CPPv4N5c | :quantum_platform::reset_exec_ctx |
| udaq13kraus_channel10noise_typeE) |     (C++                          |
| -                                 |     function)](api/languag        |
|   [cudaq::kraus_channel::op_names | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14reset_exec_ctxEv) |
|     member)](                     | -   [cud                          |
| api/languages/cpp_api.html#_CPPv4 | aq::quantum_platform::reset_noise |
| N5cudaq13kraus_channel8op_namesE) |     (C++                          |
| -                                 |     function)](api/languages/cpp_ |
|  [cudaq::kraus_channel::operator= | api.html#_CPPv4N5cudaq16quantum_p |
|     (C++                          | latform11reset_noiseENSt6size_tE) |
|     function)](api/langua         | -   [cuda                         |
| ges/cpp_api.html#_CPPv4N5cudaq13k | q::quantum_platform::set_exec_ctx |
| raus_channelaSERK13kraus_channel) |     (C++                          |
| -   [c                            |     funct                         |
| udaq::kraus_channel::operator\[\] | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq16quantum_platform12 |
|     function)](api/l              | set_exec_ctxEP16ExecutionContext) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [c                            |
| aq13kraus_channelixEKNSt6size_tE) | udaq::quantum_platform::set_noise |
| -                                 |     (C++                          |
| [cudaq::kraus_channel::parameters |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)](api                  | Pv4N5cudaq16quantum_platform9set_ |
| /languages/cpp_api.html#_CPPv4N5c | noiseEPK11noise_modelNSt6size_tE) |
| udaq13kraus_channel10parametersE) | -   [cudaq::quantum_platfor       |
| -   [cudaq::krau                  | m::supports_explicit_measurements |
| s_channel::populateDefaultOpNames |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)](api/languages/cp   | anguages/cpp_api.html#_CPPv4NK5cu |
| p_api.html#_CPPv4N5cudaq13kraus_c | daq16quantum_platform30supports_e |
| hannel22populateDefaultOpNamesEv) | xplicit_measurementsENSt6size_tE) |
| -   [cu                           | -   [cuda                         |
| daq::kraus_channel::probabilities | q::quantum_platform::supports_jit |
|     (C++                          |     (C++                          |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda |   function)](api/languages/cpp_ap |
| q13kraus_channel13probabilitiesE) | i.html#_CPPv4NK5cudaq16quantum_pl |
| -                                 | atform12supports_jitENSt6size_tE) |
|  [cudaq::kraus_channel::push_back | -   [cudaq::quantum_pla           |
|     (C++                          | tform::supports_task_distribution |
|     function)](api                |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     fu                            |
| udaq13kraus_channel9push_backE8kr | nction)](api/languages/cpp_api.ht |
| aus_opNSt8optionalINSt6stringEEE) | ml#_CPPv4NK5cudaq16quantum_platfo |
| -   [cudaq::kraus_channel::size   | rm26supports_task_distributionEv) |
|     (C++                          | -   [cudaq::quantum               |
|     function)                     | _platform::with_execution_context |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4NK5cudaq13kraus_channel4sizeEv) |     function)                     |
| -   [                             | ](api/languages/cpp_api.html#_CPP |
| cudaq::kraus_channel::unitary_ops | v4I0DpEN5cudaq16quantum_platform2 |
|     (C++                          | 2with_execution_contextEDaR16Exec |
|     member)](api/                 | utionContextRR8CallableDpRR4Args) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::QuantumTask (C++      |
| daq13kraus_channel11unitary_opsE) |     type)](api/languages/cpp_api. |
| -   [cudaq::kraus_op (C++         | html#_CPPv4N5cudaq11QuantumTaskE) |
|     struct)](api/languages/cpp_   | -   [cudaq::qubit (C++            |
| api.html#_CPPv4N5cudaq8kraus_opE) |     type)](api/languages/c        |
| -   [cudaq::kraus_op::adjoint     | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     (C++                          | -   [cudaq::qudit (C++            |
|     functi                        |     clas                          |
| on)](api/languages/cpp_api.html#_ | s)](api/languages/cpp_api.html#_C |
| CPPv4NK5cudaq8kraus_op7adjointEv) | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| -   [cudaq::kraus_op::data (C++   | -   [cudaq::qudit::qudit (C++     |
|                                   |                                   |
|  member)](api/languages/cpp_api.h | function)](api/languages/cpp_api. |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | html#_CPPv4N5cudaq5qudit5quditEv) |
| -   [cudaq::kraus_op::kraus_op    | -   [cudaq::QuEraRemoteRESTQPU    |
|     (C++                          |     (C++                          |
|     func                          |     clas                          |
| tion)](api/languages/cpp_api.html | s)](api/languages/cpp_api.html#_C |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | PPv4N5cudaq18QuEraRemoteRESTQPUE) |
| opERRNSt16initializer_listI1TEE), | -   [cudaq::qvector (C++          |
|                                   |     class)                        |
|  [\[1\]](api/languages/cpp_api.ht | ](api/languages/cpp_api.html#_CPP |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| pENSt6vectorIN5cudaq7complexEEE), | -   [cudaq::qvector::back (C++    |
|     [\[2\]](api/l                 |     function)](a                  |
| anguages/cpp_api.html#_CPPv4N5cud | pi/languages/cpp_api.html#_CPPv4N |
| aq8kraus_op8kraus_opERK8kraus_op) | 5cudaq7qvector4backENSt6size_tE), |
| -   [cudaq::kraus_op::nCols (C++  |                                   |
|                                   |   [\[1\]](api/languages/cpp_api.h |
| member)](api/languages/cpp_api.ht | tml#_CPPv4N5cudaq7qvector4backEv) |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | -   [cudaq::qvector::begin (C++   |
| -   [cudaq::kraus_op::nRows (C++  |     fu                            |
|                                   | nction)](api/languages/cpp_api.ht |
| member)](api/languages/cpp_api.ht | ml#_CPPv4N5cudaq7qvector5beginEv) |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | -   [cudaq::qvector::clear (C++   |
| -   [cudaq::kraus_op::operator=   |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function)                     | ml#_CPPv4N5cudaq7qvector5clearEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::qvector::end (C++     |
| v4N5cudaq8kraus_opaSERK8kraus_op) |                                   |
| -   [cudaq::kraus_op::precision   | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq7qvector3endEv) |
|     memb                          | -   [cudaq::qvector::front (C++   |
| er)](api/languages/cpp_api.html#_ |     function)](ap                 |
| CPPv4N5cudaq8kraus_op9precisionE) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [cudaq::KrausSelection (C++   | cudaq7qvector5frontENSt6size_tE), |
|     s                             |                                   |
| truct)](api/languages/cpp_api.htm |  [\[1\]](api/languages/cpp_api.ht |
| l#_CPPv4N5cudaq14KrausSelectionE) | ml#_CPPv4N5cudaq7qvector5frontEv) |
| -   [cudaq:                       | -   [cudaq::qvector::operator=    |
| :KrausSelection::circuit_location |     (C++                          |
|     (C++                          |     functio                       |
|     member)](api/langua           | n)](api/languages/cpp_api.html#_C |
| ges/cpp_api.html#_CPPv4N5cudaq14K | PPv4N5cudaq7qvectoraSERK7qvector) |
| rausSelection16circuit_locationE) | -   [cudaq::qvector::operator\[\] |
| -                                 |     (C++                          |
|  [cudaq::KrausSelection::is_error |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](a                    | v4N5cudaq7qvectorixEKNSt6size_tE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::qvector::qvector (C++ |
| 5cudaq14KrausSelection8is_errorE) |     function)](api/               |
| -   [cudaq::Kra                   | languages/cpp_api.html#_CPPv4N5cu |
| usSelection::kraus_operator_index | daq7qvector7qvectorENSt6size_tE), |
|     (C++                          |     [\[1\]](a                     |
|     member)](api/languages/       | pi/languages/cpp_api.html#_CPPv4N |
| cpp_api.html#_CPPv4N5cudaq14Kraus | 5cudaq7qvector7qvectorERK5state), |
| Selection20kraus_operator_indexE) |     [\[2\]](api                   |
| -   [cuda                         | /languages/cpp_api.html#_CPPv4N5c |
| q::KrausSelection::KrausSelection | udaq7qvector7qvectorERK7qvector), |
|     (C++                          |     [\[3\]](ap                    |
|     function)](a                  | i/languages/cpp_api.html#_CPPv4N5 |
| pi/languages/cpp_api.html#_CPPv4N | cudaq7qvector7qvectorERR7qvector) |
| 5cudaq14KrausSelection14KrausSele | -   [cudaq::qvector::size (C++    |
| ctionENSt6size_tENSt6vectorINSt6s |     fu                            |
| ize_tEEENSt6stringENSt6size_tEb), | nction)](api/languages/cpp_api.ht |
|     [\[1\]](api/langu             | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| ages/cpp_api.html#_CPPv4N5cudaq14 | -   [cudaq::qvector::slice (C++   |
| KrausSelection14KrausSelectionEv) |     function)](api/language       |
| -                                 | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|   [cudaq::KrausSelection::op_name | tor5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qvector::value_type   |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     typ                           |
| N5cudaq14KrausSelection7op_nameE) | e)](api/languages/cpp_api.html#_C |
| -   [                             | PPv4N5cudaq7qvector10value_typeE) |
| cudaq::KrausSelection::operator== | -   [cudaq::qview (C++            |
|     (C++                          |     clas                          |
|     function)](api/languages      | s)](api/languages/cpp_api.html#_C |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| usSelectioneqERK14KrausSelection) | -   [cudaq::qview::back (C++      |
| -                                 |     function)                     |
|    [cudaq::KrausSelection::qubits | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq5qview4backENSt6size_tE) |
|     member)]                      | -   [cudaq::qview::begin (C++     |
| (api/languages/cpp_api.html#_CPPv |                                   |
| 4N5cudaq14KrausSelection6qubitsE) | function)](api/languages/cpp_api. |
| -   [cudaq::KrausTrajectory (C++  | html#_CPPv4N5cudaq5qview5beginEv) |
|     st                            | -   [cudaq::qview::end (C++       |
| ruct)](api/languages/cpp_api.html |                                   |
| #_CPPv4N5cudaq15KrausTrajectoryE) |   function)](api/languages/cpp_ap |
| -                                 | i.html#_CPPv4N5cudaq5qview3endEv) |
|  [cudaq::KrausTrajectory::builder | -   [cudaq::qview::front (C++     |
|     (C++                          |     function)](                   |
|     function)](ap                 | api/languages/cpp_api.html#_CPPv4 |
| i/languages/cpp_api.html#_CPPv4N5 | N5cudaq5qview5frontENSt6size_tE), |
| cudaq15KrausTrajectory7builderEv) |                                   |
| -   [cu                           |    [\[1\]](api/languages/cpp_api. |
| daq::KrausTrajectory::countErrors | html#_CPPv4N5cudaq5qview5frontEv) |
|     (C++                          | -   [cudaq::qview::operator\[\]   |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     functio                       |
| 15KrausTrajectory11countErrorsEv) | n)](api/languages/cpp_api.html#_C |
| -   [                             | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| cudaq::KrausTrajectory::isOrdered | -   [cudaq::qview::qview (C++     |
|     (C++                          |     functio                       |
|     function)](api/l              | n)](api/languages/cpp_api.html#_C |
| anguages/cpp_api.html#_CPPv4NK5cu | PPv4I0EN5cudaq5qview5qviewERR1R), |
| daq15KrausTrajectory9isOrderedEv) |     [\[1                          |
| -   [cudaq::                      | \]](api/languages/cpp_api.html#_C |
| KrausTrajectory::kraus_selections | PPv4N5cudaq5qview5qviewERK5qview) |
|     (C++                          | -   [cudaq::qview::size (C++      |
|     member)](api/languag          |                                   |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | function)](api/languages/cpp_api. |
| ausTrajectory16kraus_selectionsE) | html#_CPPv4NK5cudaq5qview4sizeEv) |
| -   [cudaq:                       | -   [cudaq::qview::slice (C++     |
| :KrausTrajectory::KrausTrajectory |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|     function                      | iew5sliceENSt6size_tENSt6size_tE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qview::value_type     |
| Pv4N5cudaq15KrausTrajectory15Krau |     (C++                          |
| sTrajectoryENSt6size_tENSt6vector |     t                             |
| I14KrausSelectionEEdNSt6size_tE), | ype)](api/languages/cpp_api.html# |
|     [\[1\]](api/languag           | _CPPv4N5cudaq5qview10value_typeE) |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | -   [cudaq::range (C++            |
| ausTrajectory15KrausTrajectoryEv) |     fun                           |
| -   [cudaq::Kr                    | ction)](api/languages/cpp_api.htm |
| ausTrajectory::measurement_counts | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
|     (C++                          | orI11ElementTypeEE11ElementType), |
|     member)](api/languages        |     [\[1\]](api/languages/cpp_    |
| /cpp_api.html#_CPPv4N5cudaq15Krau | api.html#_CPPv4I0EN5cudaq5rangeEN |
| sTrajectory18measurement_countsE) | St6vectorI11ElementTypeEE11Elemen |
| -   [cud                          | tType11ElementType11ElementType), |
| aq::KrausTrajectory::multiplicity |     [                             |
|     (C++                          | \[2\]](api/languages/cpp_api.html |
|     member)](api/lan              | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::real (C++             |
| 15KrausTrajectory12multiplicityE) |     type)](api/languages/         |
| -   [                             | cpp_api.html#_CPPv4N5cudaq4realE) |
| cudaq::KrausTrajectory::num_shots | -   [cudaq::registry (C++         |
|     (C++                          |     type)](api/languages/cpp_     |
|     member)](api                  | api.html#_CPPv4N5cudaq8registryE) |
| /languages/cpp_api.html#_CPPv4N5c | -                                 |
| udaq15KrausTrajectory9num_shotsE) |  [cudaq::registry::RegisteredType |
| -   [c                            |     (C++                          |
| udaq::KrausTrajectory::operator== |     class)](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/languages/c    | 5cudaq8registry14RegisteredTypeE) |
| pp_api.html#_CPPv4NK5cudaq15Kraus | -   [cudaq::RemoteRESTQPU (C++    |
| TrajectoryeqERK15KrausTrajectory) |                                   |
| -   [cu                           |  class)](api/languages/cpp_api.ht |
| daq::KrausTrajectory::probability | ml#_CPPv4N5cudaq13RemoteRESTQPUE) |
|     (C++                          | -   [cudaq::Resources (C++        |
|     member)](api/la               |     class)](api/languages/cpp_a   |
| nguages/cpp_api.html#_CPPv4N5cuda | pi.html#_CPPv4N5cudaq9ResourcesE) |
| q15KrausTrajectory11probabilityE) | -   [cudaq::run (C++              |
| -   [cuda                         |     function)]                    |
| q::KrausTrajectory::trajectory_id | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
|     member)](api/lang             | 5invoke_result_tINSt7decay_tI13Qu |
| uages/cpp_api.html#_CPPv4N5cudaq1 | antumKernelEEDpNSt7decay_tI4ARGSE |
| 5KrausTrajectory13trajectory_idE) | EEEEENSt6size_tERN5cudaq11noise_m |
| -                                 | odelERR13QuantumKernelDpRR4ARGS), |
|   [cudaq::KrausTrajectory::weight |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0DpEN5cu |
|     member)](                     | daq3runENSt6vectorINSt15invoke_re |
| api/languages/cpp_api.html#_CPPv4 | sult_tINSt7decay_tI13QuantumKerne |
| N5cudaq15KrausTrajectory6weightE) | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| -                                 | ize_tERR13QuantumKernelDpRR4ARGS) |
|    [cudaq::KrausTrajectoryBuilder | -   [cudaq::run_async (C++        |
|     (C++                          |     functio                       |
|     class)](                      | n)](api/languages/cpp_api.html#_C |
| api/languages/cpp_api.html#_CPPv4 | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| N5cudaq22KrausTrajectoryBuilderE) | tureINSt6vectorINSt15invoke_resul |
| -   [cud                          | t_tINSt7decay_tI13QuantumKernelEE |
| aq::KrausTrajectoryBuilder::build | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|     (C++                          | ze_tENSt6size_tERN5cudaq11noise_m |
|     function)](api/lang           | odelERR13QuantumKernelDpRR4ARGS), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[1\]](api/la                |
| 22KrausTrajectoryBuilder5buildEv) | nguages/cpp_api.html#_CPPv4I0DpEN |
| -   [cud                          | 5cudaq9run_asyncENSt6futureINSt6v |
| aq::KrausTrajectoryBuilder::setId | ectorINSt15invoke_result_tINSt7de |
|     (C++                          | cay_tI13QuantumKernelEEDpNSt7deca |
|     function)](api/languages/cpp  | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| _api.html#_CPPv4N5cudaq22KrausTra | ize_tERR13QuantumKernelDpRR4ARGS) |
| jectoryBuilder5setIdENSt6size_tE) | -   [cudaq::RuntimeTarget (C++    |
| -   [cudaq::Kraus                 |                                   |
| TrajectoryBuilder::setProbability | struct)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     function)](api/languages/cpp  | -   [cudaq::sample (C++           |
| _api.html#_CPPv4N5cudaq22KrausTra |     function)](api/languages/c    |
| jectoryBuilder14setProbabilityEd) | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| -   [cudaq::Krau                  | mpleE13sample_resultRK14sample_op |
| sTrajectoryBuilder::setSelections | tionsRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\[1\                         |
|     function)](api/languag        | ]](api/languages/cpp_api.html#_CP |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | Pv4I0DpEN5cudaq6sampleE13sample_r |
| ausTrajectoryBuilder13setSelectio | esultRR13QuantumKernelDpRR4Args), |
| nsENSt6vectorI14KrausSelectionEE) |     [\                            |
| -   [cudaq::logical_observable    | [2\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|     function)](api/languages/c    | ize_tERR13QuantumKernelDpRR4Args) |
| pp_api.html#_CPPv4IDpEN5cudaq18lo | -   [cudaq::sample_options (C++   |
| gical_observableEvDpRR8MeasArgs), |     s                             |
|     [\[1\]](api/l                 | truct)](api/languages/cpp_api.htm |
| anguages/cpp_api.html#_CPPv4N5cud | l#_CPPv4N5cudaq14sample_optionsE) |
| aq18logical_observableERKNSt6vect | -   [cudaq::sample_result (C++    |
| orI14measure_resultEENSt6size_tE) |                                   |
| -   [cudaq::M2DSparseMatrix (C++  |  class)](api/languages/cpp_api.ht |
|     st                            | ml#_CPPv4N5cudaq13sample_resultE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::sample_result::append |
| #_CPPv4N5cudaq15M2DSparseMatrixE) |     (C++                          |
| -   [cudaq::M2OSparseMatrix (C++  |                                   |
|     st                            |    function)](api/languages/cpp_a |
| ruct)](api/languages/cpp_api.html | pi.html#_CPPv4N5cudaq13sample_res |
| #_CPPv4N5cudaq15M2OSparseMatrixE) | ult6appendERK15ExecutionResultb), |
| -   [cudaq::matrix_callback (C++  |     [\[1\]](api/languages/cpp_    |
|     c                             | api.html#_CPPv4N5cudaq13sample_re |
| lass)](api/languages/cpp_api.html | sult6appendERR15ExecutionResultb) |
| #_CPPv4N5cudaq15matrix_callbackE) | -   [cudaq::sample_result::begin  |
| -   [cudaq::matrix_handler (C++   |     (C++                          |
|                                   |     function)]                    |
| class)](api/languages/cpp_api.htm | (api/languages/cpp_api.html#_CPPv |
| l#_CPPv4N5cudaq14matrix_handlerE) | 4N5cudaq13sample_result5beginEv), |
| -   [cudaq::mat                   |     [\[1\]]                       |
| rix_handler::commutation_behavior | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NK5cudaq13sample_result5beginEv) |
|     struct)](api/languages/       | -   [cudaq::sample_result::cbegin |
| cpp_api.html#_CPPv4N5cudaq14matri |     (C++                          |
| x_handler20commutation_behaviorE) |     function)](                   |
| -                                 | api/languages/cpp_api.html#_CPPv4 |
|    [cudaq::matrix_handler::define | NK5cudaq13sample_result6cbeginEv) |
|     (C++                          | -   [cudaq::sample_result::cend   |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)                     |
| 5cudaq14matrix_handler6defineENSt | ](api/languages/cpp_api.html#_CPP |
| 6stringENSt6vectorINSt7int64_tEEE | v4NK5cudaq13sample_result4cendEv) |
| RR15matrix_callbackRKNSt13unorder | -   [cudaq::sample_result::clear  |
| ed_mapINSt6stringENSt6stringEEE), |     (C++                          |
|                                   |     function)                     |
| [\[1\]](api/languages/cpp_api.htm | ](api/languages/cpp_api.html#_CPP |
| l#_CPPv4N5cudaq14matrix_handler6d | v4N5cudaq13sample_result5clearEv) |
| efineENSt6stringENSt6vectorINSt7i | -   [cudaq::sample_result::count  |
| nt64_tEEERR15matrix_callbackRR20d |     (C++                          |
| iag_matrix_callbackRKNSt13unorder |     function)](                   |
| ed_mapINSt6stringENSt6stringEEE), | api/languages/cpp_api.html#_CPPv4 |
|     [\[2\]](                      | NK5cudaq13sample_result5countENSt |
| api/languages/cpp_api.html#_CPPv4 | 11string_viewEKNSt11string_viewE) |
| N5cudaq14matrix_handler6defineENS | -   [                             |
| t6stringENSt6vectorINSt7int64_tEE | cudaq::sample_result::deserialize |
| ERR15matrix_callbackRRNSt13unorde |     (C++                          |
| red_mapINSt6stringENSt6stringEEE) |     functio                       |
| -                                 | n)](api/languages/cpp_api.html#_C |
|   [cudaq::matrix_handler::degrees | PPv4N5cudaq13sample_result11deser |
|     (C++                          | ializeERNSt6vectorINSt6size_tEEE) |
|     function)](ap                 | -   [cudaq::sample_result::dump   |
| i/languages/cpp_api.html#_CPPv4NK |     (C++                          |
| 5cudaq14matrix_handler7degreesEv) |     function)](api/languag        |
| -                                 | es/cpp_api.html#_CPPv4NK5cudaq13s |
|  [cudaq::matrix_handler::displace | ample_result4dumpERNSt7ostreamE), |
|     (C++                          |     [\[1\]                        |
|     function)](api/language       | ](api/languages/cpp_api.html#_CPP |
| s/cpp_api.html#_CPPv4N5cudaq14mat | v4NK5cudaq13sample_result4dumpEv) |
| rix_handler8displaceENSt6size_tE) | -   [cudaq::sample_result::end    |
| -   [cudaq::matrix                |     (C++                          |
| _handler::get_expected_dimensions |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq13sample_result3endEv), |
|    function)](api/languages/cpp_a |     [\[1\                         |
| pi.html#_CPPv4NK5cudaq14matrix_ha | ]](api/languages/cpp_api.html#_CP |
| ndler23get_expected_dimensionsEv) | Pv4NK5cudaq13sample_result3endEv) |
| -   [cudaq::matrix_ha             | -   [                             |
| ndler::get_parameter_descriptions | cudaq::sample_result::expectation |
|     (C++                          |     (C++                          |
|                                   |     f                             |
| function)](api/languages/cpp_api. | unction)](api/languages/cpp_api.h |
| html#_CPPv4NK5cudaq14matrix_handl | tml#_CPPv4NK5cudaq13sample_result |
| er26get_parameter_descriptionsEv) | 11expectationEKNSt11string_viewE) |
| -   [c                            | -   [cuda                         |
| udaq::matrix_handler::instantiate | q::sample_result::get_annotations |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](api/langua         |
| pi/languages/cpp_api.html#_CPPv4N | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| 5cudaq14matrix_handler11instantia | sample_result15get_annotationsEv) |
| teENSt6stringERKNSt6vectorINSt6si | -   [c                            |
| ze_tEEERK20commutation_behavior), | udaq::sample_result::get_marginal |
|     [\[1\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languages/cpp_ |
| N5cudaq14matrix_handler11instanti | api.html#_CPPv4NK5cudaq13sample_r |
| ateENSt6stringERRNSt6vectorINSt6s | esult12get_marginalERKNSt6vectorI |
| ize_tEEERK20commutation_behavior) | NSt6size_tEEEKNSt11string_viewE), |
| -   [cuda                         |     [\[1\]](api/languages/cpp_    |
| q::matrix_handler::matrix_handler | api.html#_CPPv4NK5cudaq13sample_r |
|     (C++                          | esult12get_marginalERRKNSt6vector |
|     function)](api/languag        | INSt6size_tEEEKNSt11string_viewE) |
| es/cpp_api.html#_CPPv4I0_NSt11ena | -   [cuda                         |
| ble_if_tINSt12is_base_of_vI16oper | q::sample_result::get_total_shots |
| ator_handler1TEEbEEEN5cudaq14matr |     (C++                          |
| ix_handler14matrix_handlerERK1T), |     function)](api/langua         |
|     [\[1\]](ap                    | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| i/languages/cpp_api.html#_CPPv4I0 | sample_result15get_total_shotsEv) |
| _NSt11enable_if_tINSt12is_base_of | -   [cuda                         |
| _vI16operator_handler1TEEbEEEN5cu | q::sample_result::has_even_parity |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erERK1TRK20commutation_behavior), |     fun                           |
|     [\[2\]](api/languages/cpp_ap  | ction)](api/languages/cpp_api.htm |
| i.html#_CPPv4N5cudaq14matrix_hand | l#_CPPv4N5cudaq13sample_result15h |
| ler14matrix_handlerENSt6size_tE), | as_even_parityENSt11string_viewE) |
|     [\[3\]](api/                  | -   [cuda                         |
| languages/cpp_api.html#_CPPv4N5cu | q::sample_result::has_expectation |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erENSt6stringERKNSt6vectorINSt6si |     funct                         |
| ze_tEEERK20commutation_behavior), | ion)](api/languages/cpp_api.html# |
|     [\[4\]](api/                  | _CPPv4NK5cudaq13sample_result15ha |
| languages/cpp_api.html#_CPPv4N5cu | s_expectationEKNSt11string_viewE) |
| daq14matrix_handler14matrix_handl | -   [cu                           |
| erENSt6stringERRNSt6vectorINSt6si | daq::sample_result::most_probable |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\                            |     fun                           |
| [5\]](api/languages/cpp_api.html# | ction)](api/languages/cpp_api.htm |
| _CPPv4N5cudaq14matrix_handler14ma | l#_CPPv4NK5cudaq13sample_result13 |
| trix_handlerERK14matrix_handler), | most_probableEKNSt11string_viewE) |
|     [                             | -                                 |
| \[6\]](api/languages/cpp_api.html | [cudaq::sample_result::operator+= |
| #_CPPv4N5cudaq14matrix_handler14m |     (C++                          |
| atrix_handlerERR14matrix_handler) |     function)](api/langua         |
| -                                 | ges/cpp_api.html#_CPPv4N5cudaq13s |
|  [cudaq::matrix_handler::momentum | ample_resultpLERK13sample_result) |
|     (C++                          | -                                 |
|     function)](api/language       |  [cudaq::sample_result::operator= |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8momentumENSt6size_tE) |     function)](api/langua         |
| -                                 | ges/cpp_api.html#_CPPv4N5cudaq13s |
|    [cudaq::matrix_handler::number | ample_resultaSERR13sample_result) |
|     (C++                          | -                                 |
|     function)](api/langua         | [cudaq::sample_result::operator== |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6numberENSt6size_tE) |     function)](api/languag        |
| -                                 | es/cpp_api.html#_CPPv4NK5cudaq13s |
| [cudaq::matrix_handler::operator= | ample_resulteqERK13sample_result) |
|     (C++                          | -   [                             |
|     fun                           | cudaq::sample_result::probability |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     function)](api/lan            |
| NSt7is_sameI1T14matrix_handlerE5v | guages/cpp_api.html#_CPPv4NK5cuda |
| alueENSt12is_base_of_vI16operator | q13sample_result11probabilityENSt |
| _handler1TEEEbEEEN5cudaq14matrix_ | 11string_viewEKNSt11string_viewE) |
| handleraSER14matrix_handlerRK1T), | -   [cud                          |
|     [\[1\]](api/languages         | aq::sample_result::register_names |
| /cpp_api.html#_CPPv4N5cudaq14matr |     (C++                          |
| ix_handleraSERK14matrix_handler), |     function)](api/langu          |
|     [\[2\]](api/language          | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| s/cpp_api.html#_CPPv4N5cudaq14mat | 3sample_result14register_namesEv) |
| rix_handleraSERR14matrix_handler) | -                                 |
| -   [                             |    [cudaq::sample_result::reorder |
| cudaq::matrix_handler::operator== |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/languages      | ges/cpp_api.html#_CPPv4N5cudaq13s |
| /cpp_api.html#_CPPv4NK5cudaq14mat | ample_result7reorderERKNSt6vector |
| rix_handlereqERK14matrix_handler) | INSt6size_tEEEKNSt11string_viewE) |
| -                                 | -   [cu                           |
|    [cudaq::matrix_handler::parity | daq::sample_result::sample_result |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/               |
| ges/cpp_api.html#_CPPv4N5cudaq14m | languages/cpp_api.html#_CPPv4N5cu |
| atrix_handler6parityENSt6size_tE) | daq13sample_result13sample_result |
| -                                 | E16CountsDictionary10cudaq_json), |
|  [cudaq::matrix_handler::position |     [                             |
|     (C++                          | \[1\]](api/languages/cpp_api.html |
|     function)](api/language       | #_CPPv4N5cudaq13sample_result13sa |
| s/cpp_api.html#_CPPv4N5cudaq14mat | mple_resultERK15ExecutionResult), |
| rix_handler8positionENSt6size_tE) |     [\[2\]](api/la                |
| -   [cudaq::                      | nguages/cpp_api.html#_CPPv4N5cuda |
| matrix_handler::remove_definition | q13sample_result13sample_resultER |
|     (C++                          | KNSt6vectorI15ExecutionResultEE), |
|     fu                            |                                   |
| nction)](api/languages/cpp_api.ht |  [\[3\]](api/languages/cpp_api.ht |
| ml#_CPPv4N5cudaq14matrix_handler1 | ml#_CPPv4N5cudaq13sample_result13 |
| 7remove_definitionERKNSt6stringE) | sample_resultERR13sample_result), |
| -                                 |     [                             |
|   [cudaq::matrix_handler::squeeze | \[4\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq13sample_result13sa |
|     function)](api/languag        | mple_resultERR15ExecutionResult), |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     [\[5\]](api/lan               |
| trix_handler7squeezeENSt6size_tE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [cudaq::m                     | 13sample_result13sample_resultEdR |
| atrix_handler::to_diagonal_matrix | KNSt6vectorI15ExecutionResultEE), |
|     (C++                          |     [\[6\]](api/lan               |
|     function)](api/lang           | guages/cpp_api.html#_CPPv4N5cudaq |
| uages/cpp_api.html#_CPPv4NK5cudaq | 13sample_result13sample_resultEv) |
| 14matrix_handler18to_diagonal_mat | -                                 |
| rixERNSt13unordered_mapINSt6size_ |  [cudaq::sample_result::serialize |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     function)](api                |
| -                                 | /languages/cpp_api.html#_CPPv4NK5 |
| [cudaq::matrix_handler::to_matrix | cudaq13sample_result9serializeEv) |
|     (C++                          | -   [cudaq::sample_result::size   |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/c    |
| v4NK5cudaq14matrix_handler9to_mat | pp_api.html#_CPPv4NK5cudaq13sampl |
| rixERNSt13unordered_mapINSt6size_ | e_result4sizeEKNSt11string_viewE) |
| tENSt7int64_tEEERKNSt13unordered_ | -   [cudaq::sample_result::to_map |
| mapINSt6stringENSt7complexIdEEEE) |     (C++                          |
| -                                 |     function)](api/languages/cpp  |
| [cudaq::matrix_handler::to_string | _api.html#_CPPv4NK5cudaq13sample_ |
|     (C++                          | result6to_mapEKNSt11string_viewE) |
|     function)](api/               | -   [cuda                         |
| languages/cpp_api.html#_CPPv4NK5c | q::sample_result::\~sample_result |
| udaq14matrix_handler9to_stringEb) |     (C++                          |
| -                                 |     funct                         |
| [cudaq::matrix_handler::unique_id | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq13sample_resultD0Ev) |
|     function)](api/               | -   [cudaq::scalar_callback (C++  |
| languages/cpp_api.html#_CPPv4NK5c |     c                             |
| udaq14matrix_handler9unique_idEv) | lass)](api/languages/cpp_api.html |
| -   [cudaq:                       | #_CPPv4N5cudaq15scalar_callbackE) |
| :matrix_handler::\~matrix_handler | -   [c                            |
|     (C++                          | udaq::scalar_callback::operator() |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)](api/language       |
| CPPv4N5cudaq14matrix_handlerD0Ev) | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| -   [cudaq::matrix_op (C++        | alar_callbackclERKNSt13unordered_ |
|     type)](api/languages/cpp_a    | mapINSt6stringENSt7complexIdEEEE) |
| pi.html#_CPPv4N5cudaq9matrix_opE) | -   [                             |
| -   [cudaq::matrix_op_term (C++   | cudaq::scalar_callback::operator= |
|                                   |     (C++                          |
|  type)](api/languages/cpp_api.htm |     function)](api/languages/c    |
| l#_CPPv4N5cudaq14matrix_op_termE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _callbackaSERK15scalar_callback), |
|    [cudaq::mdiag_operator_handler |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|     class)](                      | r_callbackaSERR15scalar_callback) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq:                       |
| N5cudaq22mdiag_operator_handlerE) | :scalar_callback::scalar_callback |
| -   [cudaq::measure_handle (C++   |     (C++                          |
|                                   |     function)](api/languag        |
| class)](api/languages/cpp_api.htm | es/cpp_api.html#_CPPv4I0_NSt11ena |
| l#_CPPv4N5cudaq14measure_handleE) | ble_if_tINSt16is_invocable_r_vINS |
| -   [cudaq::measure_result (C++   | t7complexIdEE8CallableRKNSt13unor |
|                                   | dered_mapINSt6stringENSt7complexI |
|  type)](api/languages/cpp_api.htm | dEEEEEEbEEEN5cudaq15scalar_callba |
| l#_CPPv4N5cudaq14measure_resultE) | ck15scalar_callbackERR8Callable), |
| -   [cudaq::mpi (C++              |     [\[1\                         |
|     type)](api/languages          | ]](api/languages/cpp_api.html#_CP |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | Pv4N5cudaq15scalar_callback15scal |
| -   [cudaq::mpi::all_gather (C++  | ar_callbackERK15scalar_callback), |
|     fu                            |     [\[2                          |
| nction)](api/languages/cpp_api.ht | \]](api/languages/cpp_api.html#_C |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | PPv4N5cudaq15scalar_callback15sca |
| RNSt6vectorIdEERKNSt6vectorIdEE), | lar_callbackERR15scalar_callback) |
|                                   | -   [cudaq::scalar_operator (C++  |
|   [\[1\]](api/languages/cpp_api.h |     c                             |
| tml#_CPPv4N5cudaq3mpi10all_gather | lass)](api/languages/cpp_api.html |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | #_CPPv4N5cudaq15scalar_operatorE) |
| -   [cudaq::mpi::all_reduce (C++  | -                                 |
|                                   | [cudaq::scalar_operator::evaluate |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |                                   |
| reduceE1TRK1TRK14BinaryFunction), |    function)](api/languages/cpp_a |
|     [\[1\]](api/langu             | pi.html#_CPPv4NK5cudaq15scalar_op |
| ages/cpp_api.html#_CPPv4I00EN5cud | erator8evaluateERKNSt13unordered_ |
| aq3mpi10all_reduceE1TRK1TRK4Func) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::mpi::broadcast (C++   | -   [cudaq::scalar_ope            |
|     function)](api/               | rator::get_parameter_descriptions |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq3mpi9broadcastERNSt6stringEi), |     f                             |
|     [\[1\]](api/la                | unction)](api/languages/cpp_api.h |
| nguages/cpp_api.html#_CPPv4N5cuda | tml#_CPPv4NK5cudaq15scalar_operat |
| q3mpi9broadcastERNSt6vectorIdEEi) | or26get_parameter_descriptionsEv) |
| -   [cudaq::mpi::finalize (C++    | -   [cu                           |
|     f                             | daq::scalar_operator::is_constant |
| unction)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     function)](api/lang           |
| -   [cudaq::mpi::initialize (C++  | uages/cpp_api.html#_CPPv4NK5cudaq |
|     function                      | 15scalar_operator11is_constantEv) |
| )](api/languages/cpp_api.html#_CP | -   [c                            |
| Pv4N5cudaq3mpi10initializeEiPPc), | udaq::scalar_operator::operator\* |
|     [                             |     (C++                          |
| \[1\]](api/languages/cpp_api.html |     function                      |
| #_CPPv4N5cudaq3mpi10initializeEv) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::mpi::is_initialized   | Pv4N5cudaq15scalar_operatormlENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     function                      |     [\[1\                         |
| )](api/languages/cpp_api.html#_CP | ]](api/languages/cpp_api.html#_CP |
| Pv4N5cudaq3mpi14is_initializedEv) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::mpi::num_ranks (C++   | 7complexIdEERR15scalar_operator), |
|     fu                            |     [\[2\]](api/languages/cp      |
| nction)](api/languages/cpp_api.ht | p_api.html#_CPPv4N5cudaq15scalar_ |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | operatormlEdRK15scalar_operator), |
| -   [cudaq::mpi::rank (C++        |     [\[3\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|    function)](api/languages/cpp_a | operatormlEdRR15scalar_operator), |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     [\[4\]](api/languages         |
| -   [cudaq::noise_model (C++      | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|                                   | alar_operatormlENSt7complexIdEE), |
|    class)](api/languages/cpp_api. |     [\[5\]](api/languages/cpp     |
| html#_CPPv4N5cudaq11noise_modelE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::n                     | _operatormlERK15scalar_operator), |
| oise_model::add_all_qubit_channel |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api                | 4NKR5cudaq15scalar_operatormlEd), |
| /languages/cpp_api.html#_CPPv4IDp |     [\[7\]](api/language          |
| EN5cudaq11noise_model21add_all_qu | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| bit_channelEvRK13kraus_channeli), | alar_operatormlENSt7complexIdEE), |
|     [\[1\]](api/langua            |     [\[8\]](api/languages/cp      |
| ges/cpp_api.html#_CPPv4N5cudaq11n | p_api.html#_CPPv4NO5cudaq15scalar |
| oise_model21add_all_qubit_channel | _operatormlERK15scalar_operator), |
| ERKNSt6stringERK13kraus_channeli) |     [\[9\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|  [cudaq::noise_model::add_channel | Pv4NO5cudaq15scalar_operatormlEd) |
|     (C++                          | -   [cu                           |
|     funct                         | daq::scalar_operator::operator\*= |
| ion)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4IDpEN5cudaq11noise_model11a |     function)](api/languag        |
| dd_channelEvRK15PredicateFuncTy), | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     [\[1\]](api/languages/cpp_    | alar_operatormLENSt7complexIdEE), |
| api.html#_CPPv4IDpEN5cudaq11noise |     [\[1\]](api/languages/c       |
| _model11add_channelEvRKNSt6vector | pp_api.html#_CPPv4N5cudaq15scalar |
| INSt6size_tEEERK13kraus_channel), | _operatormLERK15scalar_operator), |
|     [\[2\]](ap                    |     [\[2                          |
| i/languages/cpp_api.html#_CPPv4N5 | \]](api/languages/cpp_api.html#_C |
| cudaq11noise_model11add_channelER | PPv4N5cudaq15scalar_operatormLEd) |
| KNSt6stringERK15PredicateFuncTy), | -   [                             |
|                                   | cudaq::scalar_operator::operator+ |
| [\[3\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq11noise_model11add |     function                      |
| _channelERKNSt6stringERKNSt6vecto | )](api/languages/cpp_api.html#_CP |
| rINSt6size_tEEERK13kraus_channel) | Pv4N5cudaq15scalar_operatorplENSt |
| -   [cudaq::noise_model::empty    | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     function                      | ]](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4N5cudaq15scalar_operatorplENSt |
| Pv4NK5cudaq11noise_model5emptyEv) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
| [cudaq::noise_model::get_channels | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatorplEdRK15scalar_operator), |
|     function)](api/l              |     [\[3\]](api/languages/cp      |
| anguages/cpp_api.html#_CPPv4I0ENK | p_api.html#_CPPv4N5cudaq15scalar_ |
| 5cudaq11noise_model12get_channels | operatorplEdRR15scalar_operator), |
| ENSt6vectorI13kraus_channelEERKNS |     [\[4\]](api/languages         |
| t6vectorINSt6size_tEEERKNSt6vecto | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| rINSt6size_tEEERKNSt6vectorIdEE), | alar_operatorplENSt7complexIdEE), |
|     [\[1\]](api/languages/cpp_a   |     [\[5\]](api/languages/cpp     |
| pi.html#_CPPv4NK5cudaq11noise_mod | _api.html#_CPPv4NKR5cudaq15scalar |
| el12get_channelsERKNSt6stringERKN | _operatorplERK15scalar_operator), |
| St6vectorINSt6size_tEEERKNSt6vect |     [\[6\]]                       |
| orINSt6size_tEEERKNSt6vectorIdEE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatorplEd), |
|  [cudaq::noise_model::noise_model |     [\[7\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api                | 4NKR5cudaq15scalar_operatorplEv), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[8\]](api/language          |
| udaq11noise_model11noise_modelEv) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cu                           | alar_operatorplENSt7complexIdEE), |
| daq::noise_model::PredicateFuncTy |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     type)](api/la                 | _operatorplERK15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[10\]                       |
| q11noise_model15PredicateFuncTyE) | ](api/languages/cpp_api.html#_CPP |
| -   [cud                          | v4NO5cudaq15scalar_operatorplEd), |
| aq::noise_model::register_channel |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/languages      | Pv4NO5cudaq15scalar_operatorplEv) |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | -   [c                            |
| noise_model16register_channelEvv) | udaq::scalar_operator::operator+= |
| -   [cudaq::                      |     (C++                          |
| noise_model::requires_constructor |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     type)](api/languages/cp       | alar_operatorpLENSt7complexIdEE), |
| p_api.html#_CPPv4I0DpEN5cudaq11no |     [\[1\]](api/languages/c       |
| ise_model20requires_constructorE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::noise_model_type (C++ | _operatorpLERK15scalar_operator), |
|     e                             |     [\[2                          |
| num)](api/languages/cpp_api.html# | \]](api/languages/cpp_api.html#_C |
| _CPPv4N5cudaq16noise_model_typeE) | PPv4N5cudaq15scalar_operatorpLEd) |
| -   [cudaq::no                    | -   [                             |
| ise_model_type::amplitude_damping | cudaq::scalar_operator::operator- |
|     (C++                          |     (C++                          |
|     enumerator)](api/languages    |     function                      |
| /cpp_api.html#_CPPv4N5cudaq16nois | )](api/languages/cpp_api.html#_CP |
| e_model_type17amplitude_dampingE) | Pv4N5cudaq15scalar_operatormiENSt |
| -   [cudaq::noise_mode            | 7complexIdEERK15scalar_operator), |
| l_type::amplitude_damping_channel |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     e                             | Pv4N5cudaq15scalar_operatormiENSt |
| numerator)](api/languages/cpp_api | 7complexIdEERR15scalar_operator), |
| .html#_CPPv4N5cudaq16noise_model_ |     [\[2\]](api/languages/cp      |
| type25amplitude_damping_channelE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::n                     | operatormiEdRK15scalar_operator), |
| oise_model_type::bit_flip_channel |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](api/language     | operatormiEdRR15scalar_operator), |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     [\[4\]](api/languages         |
| se_model_type16bit_flip_channelE) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -   [cudaq::                      | alar_operatormiENSt7complexIdEE), |
| noise_model_type::depolarization1 |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|     enumerator)](api/languag      | _operatormiERK15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[6\]]                       |
| ise_model_type15depolarization1E) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::                      | 4NKR5cudaq15scalar_operatormiEd), |
| noise_model_type::depolarization2 |     [\[7\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     enumerator)](api/languag      | 4NKR5cudaq15scalar_operatormiEv), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[8\]](api/language          |
| ise_model_type15depolarization2E) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::noise_m               | alar_operatormiENSt7complexIdEE), |
| odel_type::depolarization_channel |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|                                   | _operatormiERK15scalar_operator), |
|   enumerator)](api/languages/cpp_ |     [\[10\]                       |
| api.html#_CPPv4N5cudaq16noise_mod | ](api/languages/cpp_api.html#_CPP |
| el_type22depolarization_channelE) | v4NO5cudaq15scalar_operatormiEd), |
| -                                 |     [\[11\                        |
|  [cudaq::noise_model_type::pauli1 | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatormiEv) |
|     enumerator)](a                | -   [c                            |
| pi/languages/cpp_api.html#_CPPv4N | udaq::scalar_operator::operator-= |
| 5cudaq16noise_model_type6pauli1E) |     (C++                          |
| -                                 |     function)](api/languag        |
|  [cudaq::noise_model_type::pauli2 | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormIENSt7complexIdEE), |
|     enumerator)](a                |     [\[1\]](api/languages/c       |
| pi/languages/cpp_api.html#_CPPv4N | pp_api.html#_CPPv4N5cudaq15scalar |
| 5cudaq16noise_model_type6pauli2E) | _operatormIERK15scalar_operator), |
| -   [cudaq                        |     [\[2                          |
| ::noise_model_type::phase_damping | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatormIEd) |
|     enumerator)](api/langu        | -   [                             |
| ages/cpp_api.html#_CPPv4N5cudaq16 | cudaq::scalar_operator::operator/ |
| noise_model_type13phase_dampingE) |     (C++                          |
| -   [cudaq::noi                   |     function                      |
| se_model_type::phase_flip_channel | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|     enumerator)](api/languages/   | 7complexIdEERK15scalar_operator), |
| cpp_api.html#_CPPv4N5cudaq16noise |     [\[1\                         |
| _model_type18phase_flip_channelE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatordvENSt |
| [cudaq::noise_model_type::unknown | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     enumerator)](ap               | p_api.html#_CPPv4N5cudaq15scalar_ |
| i/languages/cpp_api.html#_CPPv4N5 | operatordvEdRK15scalar_operator), |
| cudaq16noise_model_type7unknownE) |     [\[3\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4N5cudaq15scalar_ |
| [cudaq::noise_model_type::x_error | operatordvEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     enumerator)](ap               | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| i/languages/cpp_api.html#_CPPv4N5 | alar_operatordvENSt7complexIdEE), |
| cudaq16noise_model_type7x_errorE) |     [\[5\]](api/languages/cpp     |
| -                                 | _api.html#_CPPv4NKR5cudaq15scalar |
| [cudaq::noise_model_type::y_error | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     enumerator)](ap               | (api/languages/cpp_api.html#_CPPv |
| i/languages/cpp_api.html#_CPPv4N5 | 4NKR5cudaq15scalar_operatordvEd), |
| cudaq16noise_model_type7y_errorE) |     [\[7\]](api/language          |
| -                                 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| [cudaq::noise_model_type::z_error | alar_operatordvENSt7complexIdEE), |
|     (C++                          |     [\[8\]](api/languages/cp      |
|     enumerator)](ap               | p_api.html#_CPPv4NO5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatordvERK15scalar_operator), |
| cudaq16noise_model_type7z_errorE) |     [\[9\                         |
| -   [cudaq::num_available_gpus    | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatordvEd) |
|     function                      | -   [c                            |
| )](api/languages/cpp_api.html#_CP | udaq::scalar_operator::operator/= |
| Pv4N5cudaq18num_available_gpusEv) |     (C++                          |
| -   [cudaq::observe (C++          |     function)](api/languag        |
|     function)]                    | es/cpp_api.html#_CPPv4N5cudaq15sc |
| (api/languages/cpp_api.html#_CPPv | alar_operatordVENSt7complexIdEE), |
| 4I00DpEN5cudaq7observeENSt6vector |     [\[1\]](api/languages/c       |
| I14observe_resultEERR13QuantumKer | pp_api.html#_CPPv4N5cudaq15scalar |
| nelRK15SpinOpContainerDpRR4Args), | _operatordVERK15scalar_operator), |
|     [\[1\]](api/languages/cpp_ap  |     [\[2                          |
| i.html#_CPPv4I0DpEN5cudaq7observe | \]](api/languages/cpp_api.html#_C |
| E14observe_resultNSt6size_tERR13Q | PPv4N5cudaq15scalar_operatordVEd) |
| uantumKernelRK7spin_opDpRR4Args), | -   [                             |
|     [\[                           | cudaq::scalar_operator::operator= |
| 2\]](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4I0DpEN5cudaq7observeE14obser |     function)](api/languages/c    |
| ve_resultRK15observe_optionsRR13Q | pp_api.html#_CPPv4N5cudaq15scalar |
| uantumKernelRK7spin_opDpRR4Args), | _operatoraSERK15scalar_operator), |
|     [\[3\]](api/lang              |     [\[1\]](api/languages/        |
| uages/cpp_api.html#_CPPv4I0DpEN5c | cpp_api.html#_CPPv4N5cudaq15scala |
| udaq7observeE14observe_resultRR13 | r_operatoraSERR15scalar_operator) |
| QuantumKernelRK7spin_opDpRR4Args) | -   [c                            |
| -   [cudaq::observe_options (C++  | udaq::scalar_operator::operator== |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |     function)](api/languages/c    |
| #_CPPv4N5cudaq15observe_optionsE) | pp_api.html#_CPPv4NK5cudaq15scala |
| -   [cudaq::observe_result (C++   | r_operatoreqERK15scalar_operator) |
|                                   | -   [cudaq:                       |
| class)](api/languages/cpp_api.htm | :scalar_operator::scalar_operator |
| l#_CPPv4N5cudaq14observe_resultE) |     (C++                          |
| -                                 |     func                          |
|    [cudaq::observe_result::counts | tion)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_operator15 |
|     function)](api/languages/c    | scalar_operatorENSt7complexIdEE), |
| pp_api.html#_CPPv4N5cudaq14observ |     [\[1\]](api/langu             |
| e_result6countsERK12spin_op_term) | ages/cpp_api.html#_CPPv4N5cudaq15 |
| -   [cudaq::observe_result::dump  | scalar_operator15scalar_operatorE |
|     (C++                          | RK15scalar_callbackRRNSt13unorder |
|     function)                     | ed_mapINSt6stringENSt6stringEEE), |
| ](api/languages/cpp_api.html#_CPP |     [\[2\                         |
| v4N5cudaq14observe_result4dumpEv) | ]](api/languages/cpp_api.html#_CP |
| -   [c                            | Pv4N5cudaq15scalar_operator15scal |
| udaq::observe_result::expectation | ar_operatorERK15scalar_operator), |
|     (C++                          |     [\[3\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq15 |
| function)](api/languages/cpp_api. | scalar_operator15scalar_operatorE |
| html#_CPPv4N5cudaq14observe_resul | RR15scalar_callbackRRNSt13unorder |
| t11expectationERK12spin_op_term), | ed_mapINSt6stringENSt6stringEEE), |
|     [\[1\]](api/la                |     [\[4\                         |
| nguages/cpp_api.html#_CPPv4N5cuda | ]](api/languages/cpp_api.html#_CP |
| q14observe_result11expectationEv) | Pv4N5cudaq15scalar_operator15scal |
| -   [cuda                         | ar_operatorERR15scalar_operator), |
| q::observe_result::id_coefficient |     [\[5\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     function)](api/langu          | lar_operator15scalar_operatorEd), |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     [\[6\]](api/languag           |
| observe_result14id_coefficientEv) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cuda                         | alar_operator15scalar_operatorEv) |
| q::observe_result::observe_result | -   [                             |
|     (C++                          | cudaq::scalar_operator::to_matrix |
|                                   |     (C++                          |
|   function)](api/languages/cpp_ap |                                   |
| i.html#_CPPv4N5cudaq14observe_res |   function)](api/languages/cpp_ap |
| ult14observe_resultEdRK7spin_op), | i.html#_CPPv4NK5cudaq15scalar_ope |
|     [\[1\]](a                     | rator9to_matrixERKNSt13unordered_ |
| pi/languages/cpp_api.html#_CPPv4N | mapINSt6stringENSt7complexIdEEEE) |
| 5cudaq14observe_result14observe_r | -   [                             |
| esultEdRK7spin_op13sample_result) | cudaq::scalar_operator::to_string |
| -                                 |     (C++                          |
|  [cudaq::observe_result::operator |     function)](api/l              |
|     double (C++                   | anguages/cpp_api.html#_CPPv4NK5cu |
|     functio                       | daq15scalar_operator9to_stringEv) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::s                     |
| PPv4N5cudaq14observe_resultcvdEv) | calar_operator::\~scalar_operator |
| -                                 |     (C++                          |
|  [cudaq::observe_result::raw_data |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)](ap                 | PPv4N5cudaq15scalar_operatorD0Ev) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq::set_noise (C++        |
| cudaq14observe_result8raw_dataEv) |     function)](api/langu          |
| -   [cudaq::operator_handler (C++ | ages/cpp_api.html#_CPPv4N5cudaq9s |
|     cl                            | et_noiseERKN5cudaq11noise_modelE) |
| ass)](api/languages/cpp_api.html# | -   [cudaq::set_random_seed (C++  |
| _CPPv4N5cudaq16operator_handlerE) |     function)](api/               |
| -   [cudaq::optimizable_function  | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq15set_random_seedENSt6size_tE) |
|     class)                        | -   [cudaq::simulation_precision  |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq20optimizable_functionE) |     enum)                         |
| -   [cudaq::optimization_result   | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq20simulation_precisionE) |
|     type                          | -   [                             |
| )](api/languages/cpp_api.html#_CP | cudaq::simulation_precision::fp32 |
| Pv4N5cudaq19optimization_resultE) |     (C++                          |
| -   [cudaq::optimizer (C++        |     enumerator)](api              |
|     class)](api/languages/cpp_a   | /languages/cpp_api.html#_CPPv4N5c |
| pi.html#_CPPv4N5cudaq9optimizerE) | udaq20simulation_precision4fp32E) |
| -   [cudaq::optimizer::optimize   | -   [                             |
|     (C++                          | cudaq::simulation_precision::fp64 |
|                                   |     (C++                          |
|  function)](api/languages/cpp_api |     enumerator)](api              |
| .html#_CPPv4N5cudaq9optimizer8opt | /languages/cpp_api.html#_CPPv4N5c |
| imizeEKiRR20optimizable_function) | udaq20simulation_precision4fp64E) |
| -   [cu                           | -   [cudaq::SimulationState (C++  |
| daq::optimizer::requiresGradients |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|     function)](api/la             | #_CPPv4N5cudaq15SimulationStateE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [                             |
| q9optimizer17requiresGradientsEv) | cudaq::SimulationState::precision |
| -   [cudaq::orca (C++             |     (C++                          |
|     type)](api/languages/         |     enum)](api                    |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::orca::sample (C++     | udaq15SimulationState9precisionE) |
|     function)](api/languages/c    | -   [cudaq:                       |
| pp_api.html#_CPPv4N5cudaq4orca6sa | :SimulationState::precision::fp32 |
| mpleERNSt6vectorINSt6size_tEEERNS |     (C++                          |
| t6vectorINSt6size_tEEERNSt6vector |     enumerator)](api/lang         |
| IdEERNSt6vectorIdEEiNSt6size_tE), | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     [\[1\]]                       | 5SimulationState9precision4fp32E) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq:                       |
| 4N5cudaq4orca6sampleERNSt6vectorI | :SimulationState::precision::fp64 |
| NSt6size_tEEERNSt6vectorINSt6size |     (C++                          |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     enumerator)](api/lang         |
| -   [cudaq::orca::sample_async    | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     (C++                          | 5SimulationState9precision4fp64E) |
|                                   | -                                 |
| function)](api/languages/cpp_api. |   [cudaq::SimulationState::Tensor |
| html#_CPPv4N5cudaq4orca12sample_a |     (C++                          |
| syncERNSt6vectorINSt6size_tEEERNS |     struct)](                     |
| t6vectorINSt6size_tEEERNSt6vector | api/languages/cpp_api.html#_CPPv4 |
| IdEERNSt6vectorIdEEiNSt6size_tE), | N5cudaq15SimulationState6TensorE) |
|     [\[1\]](api/la                | -   [cudaq::spin_handler (C++     |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q4orca12sample_asyncERNSt6vectorI |   class)](api/languages/cpp_api.h |
| NSt6size_tEEERNSt6vectorINSt6size | tml#_CPPv4N5cudaq12spin_handlerE) |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | -   [cudaq:                       |
| -   [cudaq::OrcaRemoteRESTQPU     | :spin_handler::to_diagonal_matrix |
|     (C++                          |     (C++                          |
|     cla                           |     function)](api/la             |
| ss)](api/languages/cpp_api.html#_ | nguages/cpp_api.html#_CPPv4NK5cud |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | aq12spin_handler18to_diagonal_mat |
| -   [cudaq::other_policies (C++   | rixERNSt13unordered_mapINSt6size_ |
|     s                             | tENSt7int64_tEEERKNSt13unordered_ |
| truct)](api/languages/cpp_api.htm | mapINSt6stringENSt7complexIdEEEE) |
| l#_CPPv4N5cudaq14other_policiesE) | -                                 |
| -   [cudaq::PasqalRemoteRESTQPU   |   [cudaq::spin_handler::to_matrix |
|     (C++                          |     (C++                          |
|     class                         |     function                      |
| )](api/languages/cpp_api.html#_CP | )](api/languages/cpp_api.html#_CP |
| Pv4N5cudaq19PasqalRemoteRESTQPUE) | Pv4N5cudaq12spin_handler9to_matri |
| -   [cudaq::pauli1 (C++           | xERKNSt6stringENSt7complexIdEEb), |
|     class)](api/languages/cp      |     [\[1                          |
| p_api.html#_CPPv4N5cudaq6pauli1E) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4NK5cudaq12spin_handler9to_mat |
|    [cudaq::pauli1::num_parameters | rixERNSt13unordered_mapINSt6size_ |
|     (C++                          | tENSt7int64_tEEERKNSt13unordered_ |
|     member)]                      | mapINSt6stringENSt7complexIdEEEE) |
| (api/languages/cpp_api.html#_CPPv | -   [cuda                         |
| 4N5cudaq6pauli114num_parametersE) | q::spin_handler::to_sparse_matrix |
| -   [cudaq::pauli1::num_targets   |     (C++                          |
|     (C++                          |     function)](api/               |
|     membe                         | languages/cpp_api.html#_CPPv4N5cu |
| r)](api/languages/cpp_api.html#_C | daq12spin_handler16to_sparse_matr |
| PPv4N5cudaq6pauli111num_targetsE) | ixERKNSt6stringENSt7complexIdEEb) |
| -   [cudaq::pauli1::pauli1 (C++   | -                                 |
|     function)](api/languages/cpp_ |   [cudaq::spin_handler::to_string |
| api.html#_CPPv4N5cudaq6pauli16pau |     (C++                          |
| li1ERKNSt6vectorIN5cudaq4realEEE) |     function)](ap                 |
| -   [cudaq::pauli2 (C++           | i/languages/cpp_api.html#_CPPv4NK |
|     class)](api/languages/cp      | 5cudaq12spin_handler9to_stringEb) |
| p_api.html#_CPPv4N5cudaq6pauli2E) | -                                 |
| -                                 |   [cudaq::spin_handler::unique_id |
|    [cudaq::pauli2::num_parameters |     (C++                          |
|     (C++                          |     function)](ap                 |
|     member)]                      | i/languages/cpp_api.html#_CPPv4NK |
| (api/languages/cpp_api.html#_CPPv | 5cudaq12spin_handler9unique_idEv) |
| 4N5cudaq6pauli214num_parametersE) | -   [cudaq::spin_op (C++          |
| -   [cudaq::pauli2::num_targets   |     type)](api/languages/cpp      |
|     (C++                          | _api.html#_CPPv4N5cudaq7spin_opE) |
|     membe                         | -   [cudaq::spin_op_term (C++     |
| r)](api/languages/cpp_api.html#_C |                                   |
| PPv4N5cudaq6pauli211num_targetsE) |    type)](api/languages/cpp_api.h |
| -   [cudaq::pauli2::pauli2 (C++   | tml#_CPPv4N5cudaq12spin_op_termE) |
|     function)](api/languages/cpp_ | -   [cudaq::state (C++            |
| api.html#_CPPv4N5cudaq6pauli26pau |     class)](api/languages/c       |
| li2ERKNSt6vectorIN5cudaq4realEEE) | pp_api.html#_CPPv4N5cudaq5stateE) |
|                                   | -   [cudaq::state::amplitude (C++ |
|                                   |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq5 |
|                                   | state9amplitudeERKNSt6vectorIiEE) |
|                                   | -   [cudaq::state::amplitudes     |
|                                   |     (C++                          |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq5state10amplitud |
|                                   | esERKNSt6vectorINSt6vectorIiEEEE) |
|                                   | -   [cudaq::state::dump (C++      |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq5state4dumpERNSt7ostreamE), |
|                                   |                                   |
|                                   |    [\[1\]](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq5state4dumpEv) |
|                                   | -   [cudaq::state::from_data (C++ |
|                                   |     function)](api/la             |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q5state9from_dataERK10state_data) |
|                                   | -   [cudaq::state::get_num_qubits |
|                                   |     (C++                          |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq5state14get_num_qubitsEv) |
|                                   | -                                 |
|                                   |    [cudaq::state::get_num_tensors |
|                                   |     (C++                          |
|                                   |     function)](a                  |
|                                   | pi/languages/cpp_api.html#_CPPv4N |
|                                   | K5cudaq5state15get_num_tensorsEv) |
|                                   | -   [cudaq::state::get_precision  |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq5state13get_precisionEv) |
|                                   | -   [cudaq::state::get_tensor     |
|                                   |     (C++                          |
|                                   |     function)](api/la             |
|                                   | nguages/cpp_api.html#_CPPv4NK5cud |
|                                   | aq5state10get_tensorENSt6size_tE) |
|                                   | -   [cudaq::state::get_tensors    |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq5state11get_tensorsEv) |
|                                   | -   [cudaq::state::is_on_gpu (C++ |
|                                   |     funct                         |
|                                   | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4NK5cudaq5state9is_on_gpuEv) |
|                                   | -   [cudaq::state::operator()     |
|                                   |     (C++                          |
|                                   |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 5stateclENSt6size_tENSt6size_tE), |
|                                   |     [\[1\]](                      |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq5stateclERKNSt16initializ |
|                                   | er_listINSt6size_tEEENSt6size_tE) |
|                                   | -   [cudaq::state::operator= (C++ |
|                                   |     fun                           |
|                                   | ction)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq5stateaSERR5state) |
|                                   | -   [cudaq::state::operator\[\]   |
|                                   |     (C++                          |
|                                   |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4NK5cudaq5stateixENSt6size_tE) |
|                                   | -   [cudaq::state::overlap (C++   |
|                                   |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq5state7overlapERK5state) |
|                                   | -   [cudaq::state::state (C++     |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 5state5stateEP15SimulationState), |
|                                   |     [\[1\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq5state5stateERK5state), |
|                                   |     [\[2\]](api/languages/cpp_    |
|                                   | api.html#_CPPv4N5cudaq5state5stat |
|                                   | eERKNSt6vectorINSt7complexIdEEEE) |
|                                   | -   [cudaq::state::to_host (C++   |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | I0ENK5cudaq5state7to_hostEvPNSt7c |
|                                   | omplexI10ScalarTypeEENSt6size_tE) |
|                                   | -   [cudaq::state::\~state (C++   |
|                                   |     function)](api/languages/cpp_ |
|                                   | api.html#_CPPv4N5cudaq5stateD0Ev) |
|                                   | -   [cudaq::state_data (C++       |
|                                   |     type)](api/languages/cpp_api  |
|                                   | .html#_CPPv4N5cudaq10state_dataE) |
|                                   | -   [cudaq::sum_op (C++           |
|                                   |     class)](api/languages/cpp_a   |
|                                   | pi.html#_CPPv4I0EN5cudaq6sum_opE) |
|                                   | -   [cudaq::sum_op::begin (C++    |
|                                   |     fu                            |
|                                   | nction)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4NK5cudaq6sum_op5beginEv) |
|                                   | -   [cudaq::sum_op::canonicalize  |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |  function)](api/languages/cpp_api |
|                                   | .html#_CPPv4N5cudaq6sum_op12canon |
|                                   | icalizeERKNSt3setINSt6size_tEEE), |
|                                   |     [\[1\]                        |
|                                   | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq6sum_op12canonicalizeEv) |
|                                   | -                                 |
|                                   |    [cudaq::sum_op::const_iterator |
|                                   |     (C++                          |
|                                   |     struct)]                      |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4N5cudaq6sum_op14const_iteratorE) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator!= |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq6sum_op14con |
|                                   | st_iteratorneERK14const_iterator) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator\* |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratormlEv) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator++ |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratorppEv) |
|                                   | -   [cudaq::su                    |
|                                   | m_op::const_iterator::operator-\> |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratorptEv) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator== |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq6sum_op14con |
|                                   | st_iteratoreqERK14const_iterator) |
|                                   | -   [cudaq::sum_op::degrees (C++  |
|                                   |     func                          |
|                                   | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4NK5cudaq6sum_op7degreesEv) |
|                                   | -                                 |
|                                   |  [cudaq::sum_op::distribute_terms |
|                                   |     (C++                          |
|                                   |     function)](api/languages      |
|                                   | /cpp_api.html#_CPPv4NK5cudaq6sum_ |
|                                   | op16distribute_termsENSt6size_tE) |
|                                   | -   [cudaq::sum_op::dump (C++     |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4NK5cudaq6sum_op4dumpEv) |
|                                   | -   [cudaq::sum_op::empty (C++    |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq6sum_op5emptyEv) |
|                                   | -   [cudaq::sum_op::end (C++      |
|                                   |                                   |
|                                   | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq6sum_op3endEv) |
|                                   | -   [cudaq::sum_op::identity (C++ |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq6sum_op8identityENSt6size_tE), |
|                                   |     [                             |
|                                   | \[1\]](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq6sum_op8identityEv) |
|                                   | -   [cudaq::sum_op::num_terms     |
|                                   |     (C++                          |
|                                   |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq6sum_op9num_termsEv) |
|                                   | -   [cudaq::sum_op::operator\*    |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opmlE6sum_opI1TER |
|                                   | K15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[1\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opmlE6sum_opI1TER |
|                                   | K15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[2\]](api/languages         |
|                                   | /cpp_api.html#_CPPv4NK5cudaq6sum_ |
|                                   | opmlERK10product_opI9HandlerTyE), |
|                                   |     [\[3\]](api/lang              |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 6sum_opmlERK6sum_opI9HandlerTyE), |
|                                   |     [\[4\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opmlERK15scalar_operator), |
|                                   |     [\[5\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4NO5cu |
|                                   | daq6sum_opmlERK15scalar_operator) |
|                                   | -   [cudaq::sum_op::operator\*=   |
|                                   |     (C++                          |
|                                   |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opmLERK10product_opI9HandlerTyE), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_opmLERK15scalar_operator), |
|                                   |     [\[2\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_opmLERK6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator+     |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | K15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[2\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[3\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | K15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[4\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERR15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[5\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | R15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[6\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[7\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | R15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[8\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opplERK10product_opI9HandlerTyE), |
|                                   |     [\[9\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opplERK15scalar_operator), |
|                                   |     [\[10\]](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opplERK6sum_opI9HandlerTyE), |
|                                   |     [\[11\]](api/languages/       |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|                                   |     [\[12\]](api/lan              |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opplERR15scalar_operator), |
|                                   |     [\[13\]](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opplERR6sum_opI9HandlerTyE), |
|                                   |                                   |
|                                   |   [\[14\]](api/languages/cpp_api. |
|                                   | html#_CPPv4NKR5cudaq6sum_opplEv), |
|                                   |     [\[15\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opplERK10product_opI9HandlerTyE), |
|                                   |     [\[16\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opplERK15scalar_operator), |
|                                   |     [\[17\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opplERK6sum_opI9HandlerTyE), |
|                                   |     [\[18\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|                                   |     [\[19\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opplERR15scalar_operator), |
|                                   |     [\[20\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opplERR6sum_opI9HandlerTyE), |
|                                   |     [\[21\]](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NO5cudaq6sum_opplEv) |
|                                   | -   [cudaq::sum_op::operator+=    |
|                                   |     (C++                          |
|                                   |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | oppLERK10product_opI9HandlerTyE), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_oppLERK15scalar_operator), |
|                                   |     [\[2\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 6sum_oppLERK6sum_opI9HandlerTyE), |
|                                   |     [\[3\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | oppLERR10product_opI9HandlerTyE), |
|                                   |     [\[4\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_oppLERR15scalar_operator), |
|                                   |     [\[5\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_oppLERR6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator-     |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[2\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERR15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[3\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opmiE6sum_opI1TER |
|                                   | R15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[4\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[5\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opmiERK10product_opI9HandlerTyE), |
|                                   |     [\[6\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opmiERK15scalar_operator), |
|                                   |     [\[7\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opmiERK6sum_opI9HandlerTyE), |
|                                   |     [\[8\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opmiERR10product_opI9HandlerTyE), |
|                                   |     [\[9\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opmiERR15scalar_operator), |
|                                   |     [\[10\]](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opmiERR6sum_opI9HandlerTyE), |
|                                   |                                   |
|                                   |   [\[11\]](api/languages/cpp_api. |
|                                   | html#_CPPv4NKR5cudaq6sum_opmiEv), |
|                                   |     [\[12\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opmiERK10product_opI9HandlerTyE), |
|                                   |     [\[13\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opmiERK15scalar_operator), |
|                                   |     [\[14\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opmiERK6sum_opI9HandlerTyE), |
|                                   |     [\[15\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opmiERR10product_opI9HandlerTyE), |
|                                   |     [\[16\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opmiERR15scalar_operator), |
|                                   |     [\[17\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opmiERR6sum_opI9HandlerTyE), |
|                                   |     [\[18\]](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NO5cudaq6sum_opmiEv) |
|                                   | -   [cudaq::sum_op::operator-=    |
|                                   |     (C++                          |
|                                   |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opmIERK10product_opI9HandlerTyE), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_opmIERK15scalar_operator), |
|                                   |     [\[2\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 6sum_opmIERK6sum_opI9HandlerTyE), |
|                                   |     [\[3\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opmIERR10product_opI9HandlerTyE), |
|                                   |     [\[4\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_opmIERR15scalar_operator), |
|                                   |     [\[5\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_opmIERR6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator/     |
|                                   |     (C++                          |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opdvERK15scalar_operator), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4NO5cu |
|                                   | daq6sum_opdvERK15scalar_operator) |
|                                   | -   [cudaq::sum_op::operator/=    |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq6sum_opdVERK15scalar_operator) |
|                                   | -   [cudaq::sum_op::operator=     |
|                                   |     (C++                          |
|                                   |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4I00EN5cudaq6sum_opaSER6sum_o |
|                                   | pI9HandlerTyERK10product_opI1TE), |
|                                   |                                   |
|                                   |   [\[1\]](api/languages/cpp_api.h |
|                                   | tml#_CPPv4I00EN5cudaq6sum_opaSER6 |
|                                   | sum_opI9HandlerTyERK6sum_opI1TE), |
|                                   |     [\[2\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opaSERK10product_opI9HandlerTyE), |
|                                   |     [\[3\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 6sum_opaSERK6sum_opI9HandlerTyE), |
|                                   |     [\[4\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opaSERR10product_opI9HandlerTyE), |
|                                   |     [\[5\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_opaSERR6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator==    |
|                                   |     (C++                          |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4NK5cuda |
|                                   | q6sum_opeqERK6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator\[\]  |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq6sum_opixENSt6size_tE) |
|                                   | -   [cudaq::sum_op::sum_op (C++   |
|                                   |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4I00EN5cu |
|                                   | daq6sum_op6sum_opERK6sum_opI1TE), |
|                                   |     [\[1\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4I00EN5cudaq6sum_o |
|                                   | p6sum_opERK6sum_opI1TERKN14matrix |
|                                   | _handler20commutation_behaviorE), |
|                                   |     [\[2\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4IDp0E |
|                                   | N5cudaq6sum_op6sum_opEDpRR4Args), |
|                                   |     [\[3\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq6sum_op6su |
|                                   | m_opERK10product_opI9HandlerTyE), |
|                                   |     [\[4\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | op6sum_opERK6sum_opI9HandlerTyE), |
|                                   |     [\[5\]](api/languag           |
|                                   | es/cpp_api.html#_CPPv4N5cudaq6sum |
|                                   | _op6sum_opERR6sum_opI9HandlerTyE) |
|                                   | -   [                             |
|                                   | cudaq::sum_op::to_diagonal_matrix |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq6sum_op18to_diagonal_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -   [cudaq::sum_op::to_matrix     |
|                                   |     (C++                          |
|                                   |                                   |
|                                   | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq6sum_op9to_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -                                 |
|                                   |  [cudaq::sum_op::to_sparse_matrix |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq6sum_op16to_sparse_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -   [cudaq::sum_op::to_string     |
|                                   |     (C++                          |
|                                   |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq6sum_op9to_stringEv) |
|                                   | -   [cudaq::sum_op::trim (C++     |
|                                   |     function)](api/l              |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_op4trimEdRKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [cudaq::sum_op::\~sum_op (C++ |
|                                   |                                   |
|                                   |    function)](api/languages/cpp_a |
|                                   | pi.html#_CPPv4N5cudaq6sum_opD0Ev) |
|                                   | -   [cudaq::tensor (C++           |
|                                   |     type)](api/languages/cp       |
|                                   | p_api.html#_CPPv4N5cudaq6tensorE) |
|                                   | -   [cudaq::TensorStateData (C++  |
|                                   |                                   |
|                                   | type)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq15TensorStateDataE) |
|                                   | -   [cudaq::to_bools (C++         |
|                                   |     function)](api/languages/cp   |
|                                   | p_api.html#_CPPv4N5cudaq8to_bools |
|                                   | ERKNSt6vectorI14measure_resultEE) |
|                                   | -   [cudaq::to_integer (C++       |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq10to_integerERKNSt6stringE), |
|                                   |     [\[1\]](api/languages/cpp_ap  |
|                                   | i.html#_CPPv4N5cudaq10to_integerE |
|                                   | RKNSt6vectorI14measure_resultEE), |
|                                   |     [\[2\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq10to_integerERKNSt6vectorIbEE) |
|                                   | -   [cudaq::Trace (C++            |
|                                   |     class)](api/languages/c       |
|                                   | pp_api.html#_CPPv4N5cudaq5TraceE) |
|                                   | -   [cudaq::unset_noise (C++      |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq11unset_noiseEv) |
|                                   | -   [cudaq::x_error (C++          |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7x_errorE) |
|                                   | -   [cudaq::y_error (C++          |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7y_errorE) |
|                                   | -                                 |
|                                   |   [cudaq::y_error::num_parameters |
|                                   |     (C++                          |
|                                   |     member)](                     |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq7y_error14num_parametersE) |
|                                   | -   [cudaq::y_error::num_targets  |
|                                   |     (C++                          |
|                                   |     member                        |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq7y_error11num_targetsE) |
|                                   | -   [cudaq::z_error (C++          |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7z_errorE) |
+-----------------------------------+-----------------------------------+

## D {#D}

+-----------------------------------+-----------------------------------+
| -   [define() (cudaq.operators    | -   [depth_for_arity              |
|     method)](api/languages/python |     (cudaq.Resources              |
| _api.html#cudaq.operators.define) |     attribut                      |
|     -   [(cuda                    | e)](api/languages/python_api.html |
| q.operators.MatrixOperatorElement | #cudaq.Resources.depth_for_arity) |
|         class                     | -   [description (cudaq.Target    |
|         method)](api/langu        |                                   |
| ages/python_api.html#cudaq.operat | property)](api/languages/python_a |
| ors.MatrixOperatorElement.define) | pi.html#cudaq.Target.description) |
|     -   [(in module               | -   [deserialize                  |
|         cudaq.operators.cus       |     (cudaq.SampleResult           |
| tom)](api/languages/python_api.ht |     attribu                       |
| ml#cudaq.operators.custom.define) | te)](api/languages/python_api.htm |
| -   [degrees                      | l#cudaq.SampleResult.deserialize) |
|     (cu                           | -   [detector() (in module        |
| daq.operators.boson.BosonOperator |     cudaq)](api/language          |
|     property)](api/lang           | s/python_api.html#cudaq.detector) |
| uages/python_api.html#cudaq.opera | -   [detectors() (in module       |
| tors.boson.BosonOperator.degrees) |     cudaq)](api/languages         |
|     -   [(cudaq.ope               | /python_api.html#cudaq.detectors) |
| rators.boson.BosonOperatorElement | -   [distribute_terms             |
|                                   |     (cu                           |
|        property)](api/languages/p | daq.operators.boson.BosonOperator |
| ython_api.html#cudaq.operators.bo |     attribute)](api/languages/pyt |
| son.BosonOperatorElement.degrees) | hon_api.html#cudaq.operators.boso |
|     -   [(cudaq.                  | n.BosonOperator.distribute_terms) |
| operators.boson.BosonOperatorTerm |     -   [(cudaq.                  |
|         property)](api/language   | operators.fermion.FermionOperator |
| s/python_api.html#cudaq.operators |                                   |
| .boson.BosonOperatorTerm.degrees) | attribute)](api/languages/python_ |
|     -   [(cudaq.                  | api.html#cudaq.operators.fermion. |
| operators.fermion.FermionOperator | FermionOperator.distribute_terms) |
|         property)](api/language   |     -                             |
| s/python_api.html#cudaq.operators |  [(cudaq.operators.MatrixOperator |
| .fermion.FermionOperator.degrees) |         attribute)](api/language  |
|     -   [(cudaq.operato           | s/python_api.html#cudaq.operators |
| rs.fermion.FermionOperatorElement | .MatrixOperator.distribute_terms) |
|                                   |     -   [(                        |
|    property)](api/languages/pytho | cudaq.operators.spin.SpinOperator |
| n_api.html#cudaq.operators.fermio |                                   |
| n.FermionOperatorElement.degrees) |       attribute)](api/languages/p |
|     -   [(cudaq.oper              | ython_api.html#cudaq.operators.sp |
| ators.fermion.FermionOperatorTerm | in.SpinOperator.distribute_terms) |
|                                   | -   [draw() (in module            |
|       property)](api/languages/py |     cudaq)](api/lang              |
| thon_api.html#cudaq.operators.fer | uages/python_api.html#cudaq.draw) |
| mion.FermionOperatorTerm.degrees) | -   [dump (cudaq.ComplexMatrix    |
|     -                             |     a                             |
|  [(cudaq.operators.MatrixOperator | ttribute)](api/languages/python_a |
|         property)](api            | pi.html#cudaq.ComplexMatrix.dump) |
| /languages/python_api.html#cudaq. |     -   [(cudaq.ObserveResult     |
| operators.MatrixOperator.degrees) |         a                         |
|     -   [(cuda                    | ttribute)](api/languages/python_a |
| q.operators.MatrixOperatorElement | pi.html#cudaq.ObserveResult.dump) |
|         property)](api/langua     |     -   [(cu                      |
| ges/python_api.html#cudaq.operato | daq.operators.boson.BosonOperator |
| rs.MatrixOperatorElement.degrees) |         attribute)](api/l         |
|     -   [(c                       | anguages/python_api.html#cudaq.op |
| udaq.operators.MatrixOperatorTerm | erators.boson.BosonOperator.dump) |
|         property)](api/lan        |     -   [(cudaq.                  |
| guages/python_api.html#cudaq.oper | operators.boson.BosonOperatorTerm |
| ators.MatrixOperatorTerm.degrees) |         attribute)](api/langu     |
|     -   [(                        | ages/python_api.html#cudaq.operat |
| cudaq.operators.spin.SpinOperator | ors.boson.BosonOperatorTerm.dump) |
|         property)](api/la         |     -   [(cudaq.                  |
| nguages/python_api.html#cudaq.ope | operators.fermion.FermionOperator |
| rators.spin.SpinOperator.degrees) |         attribute)](api/langu     |
|     -   [(cudaq.o                 | ages/python_api.html#cudaq.operat |
| perators.spin.SpinOperatorElement | ors.fermion.FermionOperator.dump) |
|         property)](api/languages  |     -   [(cudaq.oper              |
| /python_api.html#cudaq.operators. | ators.fermion.FermionOperatorTerm |
| spin.SpinOperatorElement.degrees) |         attribute)](api/languages |
|     -   [(cuda                    | /python_api.html#cudaq.operators. |
| q.operators.spin.SpinOperatorTerm | fermion.FermionOperatorTerm.dump) |
|         property)](api/langua     |     -                             |
| ges/python_api.html#cudaq.operato |  [(cudaq.operators.MatrixOperator |
| rs.spin.SpinOperatorTerm.degrees) |         attribute)](              |
| -   [dem (cudaq.DEMResult         | api/languages/python_api.html#cud |
|     property)](api/languages/pyt  | aq.operators.MatrixOperator.dump) |
| hon_api.html#cudaq.DEMResult.dem) |     -   [(c                       |
| -   [dem_from_kernel() (in module | udaq.operators.MatrixOperatorTerm |
|     cudaq)](api/languages/pytho   |         attribute)](api/          |
| n_api.html#cudaq.dem_from_kernel) | languages/python_api.html#cudaq.o |
| -   [DEMResult (class in          | perators.MatrixOperatorTerm.dump) |
|     cudaq)](api/languages         |     -   [(                        |
| /python_api.html#cudaq.DEMResult) | cudaq.operators.spin.SpinOperator |
| -   [Depolarization1 (class in    |         attribute)](api           |
|     cudaq)](api/languages/pytho   | /languages/python_api.html#cudaq. |
| n_api.html#cudaq.Depolarization1) | operators.spin.SpinOperator.dump) |
| -   [Depolarization2 (class in    |     -   [(cuda                    |
|     cudaq)](api/languages/pytho   | q.operators.spin.SpinOperatorTerm |
| n_api.html#cudaq.Depolarization2) |         attribute)](api/lan       |
| -   [DepolarizationChannel (class | guages/python_api.html#cudaq.oper |
|     in                            | ators.spin.SpinOperatorTerm.dump) |
|                                   |     -   [(cudaq.Resources         |
| cudaq)](api/languages/python_api. |                                   |
| html#cudaq.DepolarizationChannel) |    attribute)](api/languages/pyth |
| -   [depth (cudaq.Resources       | on_api.html#cudaq.Resources.dump) |
|                                   |     -   [(cudaq.SampleResult      |
|    property)](api/languages/pytho |                                   |
| n_api.html#cudaq.Resources.depth) | attribute)](api/languages/python_ |
|                                   | api.html#cudaq.SampleResult.dump) |
|                                   |     -   [(cudaq.State             |
|                                   |                                   |
|                                   |        attribute)](api/languages/ |
|                                   | python_api.html#cudaq.State.dump) |
+-----------------------------------+-----------------------------------+

## E {#E}

+-----------------------------------+-----------------------------------+
| -   [ElementaryOperator (in       | -   [evolve() (in module          |
|     module                        |     cudaq)](api/langua            |
|     cudaq.operators)]             | ges/python_api.html#cudaq.evolve) |
| (api/languages/python_api.html#cu | -   [evolve_async() (in module    |
| daq.operators.ElementaryOperator) |     cudaq)](api/languages/py      |
| -   [empty                        | thon_api.html#cudaq.evolve_async) |
|     (cu                           | -   [EvolveResult (class in       |
| daq.operators.boson.BosonOperator |     cudaq)](api/languages/py      |
|     attribute)](api/la            | thon_api.html#cudaq.EvolveResult) |
| nguages/python_api.html#cudaq.ope | -   [ExhaustiveSamplingStrategy   |
| rators.boson.BosonOperator.empty) |     (class in                     |
|     -   [(cudaq.                  |     cudaq.ptsbe)](api             |
| operators.fermion.FermionOperator | /languages/python_api.html#cudaq. |
|         attribute)](api/langua    | ptsbe.ExhaustiveSamplingStrategy) |
| ges/python_api.html#cudaq.operato | -   [expectation                  |
| rs.fermion.FermionOperator.empty) |     (cudaq.ObserveResult          |
|     -                             |     attribut                      |
|  [(cudaq.operators.MatrixOperator | e)](api/languages/python_api.html |
|         attribute)](a             | #cudaq.ObserveResult.expectation) |
| pi/languages/python_api.html#cuda |     -   [(cudaq.SampleResult      |
| q.operators.MatrixOperator.empty) |         attribu                   |
|     -   [(                        | te)](api/languages/python_api.htm |
| cudaq.operators.spin.SpinOperator | l#cudaq.SampleResult.expectation) |
|         attribute)](api/          | -   [expectation_values           |
| languages/python_api.html#cudaq.o |     (cudaq.EvolveResult           |
| perators.spin.SpinOperator.empty) |     attribute)](ap                |
| -   [enable_return_to_log()       | i/languages/python_api.html#cudaq |
|     (cudaq.PyKernelDecorator      | .EvolveResult.expectation_values) |
|     method)](api/langu            | -   [expectation_z                |
| ages/python_api.html#cudaq.PyKern |     (cudaq.SampleResult           |
| elDecorator.enable_return_to_log) |     attribute                     |
| -   [epsilon                      | )](api/languages/python_api.html# |
|     (cudaq.optimizers.Adam        | cudaq.SampleResult.expectation_z) |
|     prope                         | -   [expected_dimensions          |
| rty)](api/languages/python_api.ht |     (cuda                         |
| ml#cudaq.optimizers.Adam.epsilon) | q.operators.MatrixOperatorElement |
| -   [estimate() (in module        |                                   |
|     cudaq)](api/language          | property)](api/languages/python_a |
| s/python_api.html#cudaq.estimate) | pi.html#cudaq.operators.MatrixOpe |
| -   [estimate_resources() (in     | ratorElement.expected_dimensions) |
|     module                        |                                   |
|                                   |                                   |
|    cudaq)](api/languages/python_a |                                   |
| pi.html#cudaq.estimate_resources) |                                   |
| -   [EstimateResult (class in     |                                   |
|     cudaq)](api/languages/pyth    |                                   |
| on_api.html#cudaq.EstimateResult) |                                   |
| -   [evaluate                     |                                   |
|                                   |                                   |
|   (cudaq.operators.ScalarOperator |                                   |
|     attribute)](api/              |                                   |
| languages/python_api.html#cudaq.o |                                   |
| perators.ScalarOperator.evaluate) |                                   |
| -   [evaluate_coefficient         |                                   |
|     (cudaq.                       |                                   |
| operators.boson.BosonOperatorTerm |                                   |
|     attr                          |                                   |
| ibute)](api/languages/python_api. |                                   |
| html#cudaq.operators.boson.BosonO |                                   |
| peratorTerm.evaluate_coefficient) |                                   |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |                                   |
|         attribut                  |                                   |
| e)](api/languages/python_api.html |                                   |
| #cudaq.operators.fermion.FermionO |                                   |
| peratorTerm.evaluate_coefficient) |                                   |
|     -   [(c                       |                                   |
| udaq.operators.MatrixOperatorTerm |                                   |
|                                   |                                   |
|  attribute)](api/languages/python |                                   |
| _api.html#cudaq.operators.MatrixO |                                   |
| peratorTerm.evaluate_coefficient) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         at                        |                                   |
| tribute)](api/languages/python_ap |                                   |
| i.html#cudaq.operators.spin.SpinO |                                   |
| peratorTerm.evaluate_coefficient) |                                   |
+-----------------------------------+-----------------------------------+

## F {#F}

+-----------------------------------+-----------------------------------+
| -   [f_tol (cudaq.optimizers.Adam | -   [finalize() (in module        |
|     pro                           |     cudaq.mpi)](api/languages/py  |
| perty)](api/languages/python_api. | thon_api.html#cudaq.mpi.finalize) |
| html#cudaq.optimizers.Adam.f_tol) | -   [ForwardDifference (class in  |
|     -   [(cudaq.optimizers.SGD    |     cudaq.gradients)              |
|         pr                        | ](api/languages/python_api.html#c |
| operty)](api/languages/python_api | udaq.gradients.ForwardDifference) |
| .html#cudaq.optimizers.SGD.f_tol) | -   [from_data (cudaq.State       |
| -   [FermionOperator (class in    |                                   |
|                                   |   attribute)](api/languages/pytho |
|    cudaq.operators.fermion)](api/ | n_api.html#cudaq.State.from_data) |
| languages/python_api.html#cudaq.o | -   [from_json                    |
| perators.fermion.FermionOperator) |     (                             |
| -   [FermionOperatorElement       | cudaq.operators.spin.SpinOperator |
|     (class in                     |     attribute)](api/lang          |
|     cuda                          | uages/python_api.html#cudaq.opera |
| q.operators.fermion)](api/languag | tors.spin.SpinOperator.from_json) |
| es/python_api.html#cudaq.operator |     -   [(cuda                    |
| s.fermion.FermionOperatorElement) | q.operators.spin.SpinOperatorTerm |
| -   [FermionOperatorTerm (class   |         attribute)](api/language  |
|     in                            | s/python_api.html#cudaq.operators |
|     c                             | .spin.SpinOperatorTerm.from_json) |
| udaq.operators.fermion)](api/lang | -   [from_matrices                |
| uages/python_api.html#cudaq.opera |     (cudaq.DEMResult              |
| tors.fermion.FermionOperatorTerm) |     attrib                        |
| -   [final_expectation_values     | ute)](api/languages/python_api.ht |
|     (cudaq.EvolveResult           | ml#cudaq.DEMResult.from_matrices) |
|     attribute)](api/lang          | -   [from_word                    |
| uages/python_api.html#cudaq.Evolv |     (                             |
| eResult.final_expectation_values) | cudaq.operators.spin.SpinOperator |
| -   [final_state                  |     attribute)](api/lang          |
|     (cudaq.EvolveResult           | uages/python_api.html#cudaq.opera |
|     attribu                       | tors.spin.SpinOperator.from_word) |
| te)](api/languages/python_api.htm |                                   |
| l#cudaq.EvolveResult.final_state) |                                   |
+-----------------------------------+-----------------------------------+

## G {#G}

+-----------------------------------+-----------------------------------+
| -   [gamma (cudaq.optimizers.SPSA | -   [get_sequential_data          |
|     pro                           |     (cudaq.SampleResult           |
| perty)](api/languages/python_api. |     attribute)](api               |
| html#cudaq.optimizers.SPSA.gamma) | /languages/python_api.html#cudaq. |
| -   [gate_count_by_arity          | SampleResult.get_sequential_data) |
|     (cudaq.Resources              | -   [get_spin                     |
|     property)](                   |     (cudaq.ObserveResult          |
| api/languages/python_api.html#cud |     attri                         |
| aq.Resources.gate_count_by_arity) | bute)](api/languages/python_api.h |
| -   [gate_count_for_arity         | tml#cudaq.ObserveResult.get_spin) |
|     (cudaq.Resources              | -   [get_state() (in module       |
|     attribute)](a                 |     cudaq)](api/languages         |
| pi/languages/python_api.html#cuda | /python_api.html#cudaq.get_state) |
| q.Resources.gate_count_for_arity) | -   [get_state_async() (in module |
| -   [get (cudaq.AsyncEvolveResult |     cudaq)](api/languages/pytho   |
|     attr                          | n_api.html#cudaq.get_state_async) |
| ibute)](api/languages/python_api. | -   [get_state_refval             |
| html#cudaq.AsyncEvolveResult.get) |     (cudaq.State                  |
|                                   |     attri                         |
|    -   [(cudaq.AsyncObserveResult | bute)](api/languages/python_api.h |
|         attri                     | tml#cudaq.State.get_state_refval) |
| bute)](api/languages/python_api.h | -   [get_target() (in module      |
| tml#cudaq.AsyncObserveResult.get) |     cudaq)](api/languages/        |
|     -   [(cudaq.AsyncStateResult  | python_api.html#cudaq.get_target) |
|         att                       | -   [get_targets() (in module     |
| ribute)](api/languages/python_api |     cudaq)](api/languages/p       |
| .html#cudaq.AsyncStateResult.get) | ython_api.html#cudaq.get_targets) |
| -   [get_binary_symplectic_form   | -   [get_total_shots              |
|     (cuda                         |     (cudaq.SampleResult           |
| q.operators.spin.SpinOperatorTerm |     attribute)]                   |
|     attribut                      | (api/languages/python_api.html#cu |
| e)](api/languages/python_api.html | daq.SampleResult.get_total_shots) |
| #cudaq.operators.spin.SpinOperato | -   [get_trajectory               |
| rTerm.get_binary_symplectic_form) |                                   |
| -   [get_channels                 |   (cudaq.ptsbe.PTSBEExecutionData |
|     (cudaq.NoiseModel             |     attribute)](api/langua        |
|     attrib                        | ges/python_api.html#cudaq.ptsbe.P |
| ute)](api/languages/python_api.ht | TSBEExecutionData.get_trajectory) |
| ml#cudaq.NoiseModel.get_channels) | -   [getTensor (cudaq.State       |
| -   [get_marginal_counts          |                                   |
|     (cudaq.SampleResult           |   attribute)](api/languages/pytho |
|     attribute)](api               | n_api.html#cudaq.State.getTensor) |
| /languages/python_api.html#cudaq. | -   [getTensors (cudaq.State      |
| SampleResult.get_marginal_counts) |                                   |
| -   [get_ops (cudaq.KrausChannel  |  attribute)](api/languages/python |
|     att                           | _api.html#cudaq.State.getTensors) |
| ribute)](api/languages/python_api | -   [gradient (class in           |
| .html#cudaq.KrausChannel.get_ops) |     cudaq.g                       |
| -   [get_pauli_word               | radients)](api/languages/python_a |
|     (cuda                         | pi.html#cudaq.gradients.gradient) |
| q.operators.spin.SpinOperatorTerm | -   [GradientDescent (class in    |
|     attribute)](api/languages/pyt |     cudaq.optimizers              |
| hon_api.html#cudaq.operators.spin | )](api/languages/python_api.html# |
| .SpinOperatorTerm.get_pauli_word) | cudaq.optimizers.GradientDescent) |
| -   [get_precision (cudaq.Target  | -   [gridsynth() (in module       |
|     att                           |                                   |
| ribute)](api/languages/python_api | cudaq.synth)](api/languages/pytho |
| .html#cudaq.Target.get_precision) | n_api.html#cudaq.synth.gridsynth) |
| -   [get_register_counts          |                                   |
|     (cudaq.SampleResult           |                                   |
|     attribute)](api               |                                   |
| /languages/python_api.html#cudaq. |                                   |
| SampleResult.get_register_counts) |                                   |
+-----------------------------------+-----------------------------------+

## H {#H}

+-----------------------------------+-----------------------------------+
| -   [has_execution_data           | -   [has_target() (in module      |
|                                   |     cudaq)](api/languages/        |
|    (cudaq.ptsbe.PTSBESampleResult | python_api.html#cudaq.has_target) |
|     attribute)](api/languages     | -   [HIGH_WEIGHT_BIAS             |
| /python_api.html#cudaq.ptsbe.PTSB |                                   |
| ESampleResult.has_execution_data) |   (cudaq.ptsbe.ShotAllocationType |
|                                   |     attribute)](api/language      |
|                                   | s/python_api.html#cudaq.ptsbe.Sho |
|                                   | tAllocationType.HIGH_WEIGHT_BIAS) |
+-----------------------------------+-----------------------------------+

## I {#I}

+-----------------------------------+-----------------------------------+
| -   [I (cudaq.spin.Pauli          | -   [instantiate()                |
|     attribute)](api/languages/py  |     (cudaq.operators              |
| thon_api.html#cudaq.spin.Pauli.I) |     m                             |
| -   [id                           | ethod)](api/languages/python_api. |
|     (cuda                         | html#cudaq.operators.instantiate) |
| q.operators.MatrixOperatorElement |     -   [(in module               |
|     property)](api/l              |         cudaq.operators.custom)]  |
| anguages/python_api.html#cudaq.op | (api/languages/python_api.html#cu |
| erators.MatrixOperatorElement.id) | daq.operators.custom.instantiate) |
| -   [identity                     | -   [instructions                 |
|     (cu                           |                                   |
| daq.operators.boson.BosonOperator |   (cudaq.ptsbe.PTSBEExecutionData |
|     attribute)](api/langu         |     property)](api/lang           |
| ages/python_api.html#cudaq.operat | uages/python_api.html#cudaq.ptsbe |
| ors.boson.BosonOperator.identity) | .PTSBEExecutionData.instructions) |
|     -   [(cudaq.                  | -   [intermediate_states          |
| operators.fermion.FermionOperator |     (cudaq.EvolveResult           |
|         attribute)](api/languages |     attribute)](api               |
| /python_api.html#cudaq.operators. | /languages/python_api.html#cudaq. |
| fermion.FermionOperator.identity) | EvolveResult.intermediate_states) |
|     -                             | -   [IntermediateResultSave       |
|  [(cudaq.operators.MatrixOperator |     (class in                     |
|         attribute)](api/          |     c                             |
| languages/python_api.html#cudaq.o | udaq)](api/languages/python_api.h |
| perators.MatrixOperator.identity) | tml#cudaq.IntermediateResultSave) |
|     -   [(                        | -   [is_compiled()                |
| cudaq.operators.spin.SpinOperator |     (cudaq.PyKernelDecorator      |
|         attribute)](api/lan       |     method)](                     |
| guages/python_api.html#cudaq.oper | api/languages/python_api.html#cud |
| ators.spin.SpinOperator.identity) | aq.PyKernelDecorator.is_compiled) |
| -   [initial_parameters           | -   [is_constant                  |
|     (cudaq.optimizers.Adam        |                                   |
|     property)](api/l              |   (cudaq.operators.ScalarOperator |
| anguages/python_api.html#cudaq.op |     attribute)](api/lan           |
| timizers.Adam.initial_parameters) | guages/python_api.html#cudaq.oper |
|     -   [(cudaq.optimizers.COBYLA | ators.ScalarOperator.is_constant) |
|         property)](api/lan        | -   [is_emulated (cudaq.Target    |
| guages/python_api.html#cudaq.opti |     a                             |
| mizers.COBYLA.initial_parameters) | ttribute)](api/languages/python_a |
|     -   [                         | pi.html#cudaq.Target.is_emulated) |
| (cudaq.optimizers.GradientDescent | -   [is_error                     |
|                                   |     (cudaq.ptsbe.KrausSelection   |
|       property)](api/languages/py |     property)](                   |
| thon_api.html#cudaq.optimizers.Gr | api/languages/python_api.html#cud |
| adientDescent.initial_parameters) | aq.ptsbe.KrausSelection.is_error) |
|     -   [(cudaq.optimizers.LBFGS  | -   [is_identity                  |
|         property)](api/la         |     (cudaq.                       |
| nguages/python_api.html#cudaq.opt | operators.boson.BosonOperatorTerm |
| imizers.LBFGS.initial_parameters) |     attribute)](api/languages/py  |
|                                   | thon_api.html#cudaq.operators.bos |
| -   [(cudaq.optimizers.NelderMead | on.BosonOperatorTerm.is_identity) |
|         property)](api/languag    |     -   [(cudaq.oper              |
| es/python_api.html#cudaq.optimize | ators.fermion.FermionOperatorTerm |
| rs.NelderMead.initial_parameters) |                                   |
|     -   [(cudaq.optimizers.SGD    |  attribute)](api/languages/python |
|         property)](api/           | _api.html#cudaq.operators.fermion |
| languages/python_api.html#cudaq.o | .FermionOperatorTerm.is_identity) |
| ptimizers.SGD.initial_parameters) |     -   [(c                       |
|     -   [(cudaq.optimizers.SPSA   | udaq.operators.MatrixOperatorTerm |
|         property)](api/l          |         attribute)](api/languag   |
| anguages/python_api.html#cudaq.op | es/python_api.html#cudaq.operator |
| timizers.SPSA.initial_parameters) | s.MatrixOperatorTerm.is_identity) |
| -   [initialize() (in module      |     -   [(cuda                    |
|                                   | q.operators.spin.SpinOperatorTerm |
|    cudaq.mpi)](api/languages/pyth |                                   |
| on_api.html#cudaq.mpi.initialize) |        attribute)](api/languages/ |
| -   [initialize_cudaq() (in       | python_api.html#cudaq.operators.s |
|     module                        | pin.SpinOperatorTerm.is_identity) |
|     cudaq)](api/languages/python  | -   [is_initialized() (in module  |
| _api.html#cudaq.initialize_cudaq) |     c                             |
| -   [InitialState (in module      | udaq.mpi)](api/languages/python_a |
|     cudaq.dynamics.helpers)](     | pi.html#cudaq.mpi.is_initialized) |
| api/languages/python_api.html#cud | -   [is_on_gpu (cudaq.State       |
| aq.dynamics.helpers.InitialState) |                                   |
| -   [InitialStateType (class in   |   attribute)](api/languages/pytho |
|     cudaq)](api/languages/python  | n_api.html#cudaq.State.is_on_gpu) |
| _api.html#cudaq.InitialStateType) | -   [is_remote (cudaq.Target      |
|                                   |                                   |
|                                   |  attribute)](api/languages/python |
|                                   | _api.html#cudaq.Target.is_remote) |
|                                   | -   [items (cudaq.SampleResult    |
|                                   |     a                             |
|                                   | ttribute)](api/languages/python_a |
|                                   | pi.html#cudaq.SampleResult.items) |
+-----------------------------------+-----------------------------------+

## K {#K}

+-----------------------------------+-----------------------------------+
| -   [Kernel (in module            | -   [KrausChannel (class in       |
|     cudaq)](api/langua            |     cudaq)](api/languages/py      |
| ges/python_api.html#cudaq.Kernel) | thon_api.html#cudaq.KrausChannel) |
| -   [kernel() (in module          | -   [KrausOperator (class in      |
|     cudaq)](api/langua            |     cudaq)](api/languages/pyt     |
| ges/python_api.html#cudaq.kernel) | hon_api.html#cudaq.KrausOperator) |
| -   [kraus_operator_index         | -   [KrausSelection (class in     |
|     (cudaq.ptsbe.KrausSelection   |     cudaq                         |
|     property)](api/language       | .ptsbe)](api/languages/python_api |
| s/python_api.html#cudaq.ptsbe.Kra | .html#cudaq.ptsbe.KrausSelection) |
| usSelection.kraus_operator_index) | -   [KrausTrajectory (class in    |
| -   [kraus_selections             |     cudaq.                        |
|     (cudaq.ptsbe.KrausTrajectory  | ptsbe)](api/languages/python_api. |
|     property)](api/langu          | html#cudaq.ptsbe.KrausTrajectory) |
| ages/python_api.html#cudaq.ptsbe. |                                   |
| KrausTrajectory.kraus_selections) |                                   |
+-----------------------------------+-----------------------------------+

## L {#L}

+-----------------------------------+-----------------------------------+
| -   [launch_args_required()       | -   [lower_bounds                 |
|     (cudaq.PyKernelDecorator      |     (cudaq.optimizers.Adam        |
|     method)](api/langu            |     property)]                    |
| ages/python_api.html#cudaq.PyKern | (api/languages/python_api.html#cu |
| elDecorator.launch_args_required) | daq.optimizers.Adam.lower_bounds) |
| -   [LBFGS (class in              |     -   [(cudaq.optimizers.COBYLA |
|     cudaq.                        |         property)](a              |
| optimizers)](api/languages/python | pi/languages/python_api.html#cuda |
| _api.html#cudaq.optimizers.LBFGS) | q.optimizers.COBYLA.lower_bounds) |
| -   [left_multiply                |     -   [                         |
|     (cudaq.SuperOperator          | (cudaq.optimizers.GradientDescent |
|     attribute)                    |         property)](api/langua     |
| ](api/languages/python_api.html#c | ges/python_api.html#cudaq.optimiz |
| udaq.SuperOperator.left_multiply) | ers.GradientDescent.lower_bounds) |
| -   [left_right_multiply          |     -   [(cudaq.optimizers.LBFGS  |
|     (cudaq.SuperOperator          |         property)](               |
|     attribute)](api/              | api/languages/python_api.html#cud |
| languages/python_api.html#cudaq.S | aq.optimizers.LBFGS.lower_bounds) |
| uperOperator.left_right_multiply) |                                   |
| -   [logical_observable() (in     | -   [(cudaq.optimizers.NelderMead |
|     module                        |         property)](api/l          |
|                                   | anguages/python_api.html#cudaq.op |
|    cudaq)](api/languages/python_a | timizers.NelderMead.lower_bounds) |
| pi.html#cudaq.logical_observable) |     -   [(cudaq.optimizers.SGD    |
| -   [LOW_WEIGHT_BIAS              |         property)                 |
|                                   | ](api/languages/python_api.html#c |
|   (cudaq.ptsbe.ShotAllocationType | udaq.optimizers.SGD.lower_bounds) |
|     attribute)](api/languag       |     -   [(cudaq.optimizers.SPSA   |
| es/python_api.html#cudaq.ptsbe.Sh |         property)]                |
| otAllocationType.LOW_WEIGHT_BIAS) | (api/languages/python_api.html#cu |
|                                   | daq.optimizers.SPSA.lower_bounds) |
+-----------------------------------+-----------------------------------+

## M {#M}

+-----------------------------------+-----------------------------------+
| -   [m2d (cudaq.DEMResult         | -   [mdiag_sparse_matrix (C++     |
|     property)](api/languages/pyt  |     type)](api/languages/cpp_api. |
| hon_api.html#cudaq.DEMResult.m2d) | html#_CPPv419mdiag_sparse_matrix) |
| -   [m2d_matrix (cudaq.DEMResult  | -   [measure_handle (class in     |
|     pr                            |     cudaq)](api/languages/pyth    |
| operty)](api/languages/python_api | on_api.html#cudaq.measure_handle) |
| .html#cudaq.DEMResult.m2d_matrix) | -   [measurement_counts           |
| -   [m2o (cudaq.DEMResult         |     (cudaq.ptsbe.KrausTrajectory  |
|     property)](api/languages/pyt  |     property)](api/languag        |
| hon_api.html#cudaq.DEMResult.m2o) | es/python_api.html#cudaq.ptsbe.Kr |
| -   [m2o_matrix (cudaq.DEMResult  | ausTrajectory.measurement_counts) |
|     pr                            | -   [merge_kernel()               |
| operty)](api/languages/python_api |     (cudaq.PyKernelDecorator      |
| .html#cudaq.DEMResult.m2o_matrix) |     method)](a                    |
| -   [make_kernel() (in module     | pi/languages/python_api.html#cuda |
|     cudaq)](api/languages/p       | q.PyKernelDecorator.merge_kernel) |
| ython_api.html#cudaq.make_kernel) | -   [merge_quake_source()         |
| -   [matrices_computed            |     (cudaq.PyKernelDecorator      |
|     (cudaq.DEMResult              |     method)](api/lan              |
|     property)                     | guages/python_api.html#cudaq.PyKe |
| ](api/languages/python_api.html#c | rnelDecorator.merge_quake_source) |
| udaq.DEMResult.matrices_computed) | -   [min_degree                   |
| -   [MatrixOperator (class in     |     (cu                           |
|     cudaq.operato                 | daq.operators.boson.BosonOperator |
| rs)](api/languages/python_api.htm |     property)](api/languag        |
| l#cudaq.operators.MatrixOperator) | es/python_api.html#cudaq.operator |
| -   [MatrixOperatorElement (class | s.boson.BosonOperator.min_degree) |
|     in                            |     -   [(cudaq.                  |
|     cudaq.operators)](ap          | operators.boson.BosonOperatorTerm |
| i/languages/python_api.html#cudaq |                                   |
| .operators.MatrixOperatorElement) |        property)](api/languages/p |
| -   [MatrixOperatorTerm (class in | ython_api.html#cudaq.operators.bo |
|     cudaq.operators)]             | son.BosonOperatorTerm.min_degree) |
| (api/languages/python_api.html#cu |     -   [(cudaq.                  |
| daq.operators.MatrixOperatorTerm) | operators.fermion.FermionOperator |
| -   [max_degree                   |                                   |
|     (cu                           |        property)](api/languages/p |
| daq.operators.boson.BosonOperator | ython_api.html#cudaq.operators.fe |
|     property)](api/languag        | rmion.FermionOperator.min_degree) |
| es/python_api.html#cudaq.operator |     -   [(cudaq.oper              |
| s.boson.BosonOperator.max_degree) | ators.fermion.FermionOperatorTerm |
|     -   [(cudaq.                  |                                   |
| operators.boson.BosonOperatorTerm |    property)](api/languages/pytho |
|                                   | n_api.html#cudaq.operators.fermio |
|        property)](api/languages/p | n.FermionOperatorTerm.min_degree) |
| ython_api.html#cudaq.operators.bo |     -                             |
| son.BosonOperatorTerm.max_degree) |  [(cudaq.operators.MatrixOperator |
|     -   [(cudaq.                  |         property)](api/la         |
| operators.fermion.FermionOperator | nguages/python_api.html#cudaq.ope |
|                                   | rators.MatrixOperator.min_degree) |
|        property)](api/languages/p |     -   [(c                       |
| ython_api.html#cudaq.operators.fe | udaq.operators.MatrixOperatorTerm |
| rmion.FermionOperator.max_degree) |         property)](api/langua     |
|     -   [(cudaq.oper              | ges/python_api.html#cudaq.operato |
| ators.fermion.FermionOperatorTerm | rs.MatrixOperatorTerm.min_degree) |
|                                   |     -   [(                        |
|    property)](api/languages/pytho | cudaq.operators.spin.SpinOperator |
| n_api.html#cudaq.operators.fermio |         property)](api/langu      |
| n.FermionOperatorTerm.max_degree) | ages/python_api.html#cudaq.operat |
|     -                             | ors.spin.SpinOperator.min_degree) |
|  [(cudaq.operators.MatrixOperator |     -   [(cuda                    |
|         property)](api/la         | q.operators.spin.SpinOperatorTerm |
| nguages/python_api.html#cudaq.ope |         property)](api/languages  |
| rators.MatrixOperator.max_degree) | /python_api.html#cudaq.operators. |
|     -   [(c                       | spin.SpinOperatorTerm.min_degree) |
| udaq.operators.MatrixOperatorTerm | -   [minimal_eigenvalue           |
|         property)](api/langua     |     (cudaq.ComplexMatrix          |
| ges/python_api.html#cudaq.operato |     attribute)](api               |
| rs.MatrixOperatorTerm.max_degree) | /languages/python_api.html#cudaq. |
|     -   [(                        | ComplexMatrix.minimal_eigenvalue) |
| cudaq.operators.spin.SpinOperator | -   module                        |
|         property)](api/langu      |     -   [cudaq](api/langua        |
| ages/python_api.html#cudaq.operat | ges/python_api.html#module-cudaq) |
| ors.spin.SpinOperator.max_degree) |     -                             |
|     -   [(cuda                    |    [cudaq.boson](api/languages/py |
| q.operators.spin.SpinOperatorTerm | thon_api.html#module-cudaq.boson) |
|         property)](api/languages  |     -   [                         |
| /python_api.html#cudaq.operators. | cudaq.fermion](api/languages/pyth |
| spin.SpinOperatorTerm.max_degree) | on_api.html#module-cudaq.fermion) |
| -   [max_iterations               |     -   [cudaq.operators.cu       |
|     (cudaq.optimizers.Adam        | stom](api/languages/python_api.ht |
|     property)](a                  | ml#module-cudaq.operators.custom) |
| pi/languages/python_api.html#cuda |                                   |
| q.optimizers.Adam.max_iterations) |  -   [cudaq.spin](api/languages/p |
|     -   [(cudaq.optimizers.COBYLA | ython_api.html#module-cudaq.spin) |
|         property)](api            | -   [most_probable                |
| /languages/python_api.html#cudaq. |     (cudaq.SampleResult           |
| optimizers.COBYLA.max_iterations) |     attribute                     |
|     -   [                         | )](api/languages/python_api.html# |
| (cudaq.optimizers.GradientDescent | cudaq.SampleResult.most_probable) |
|         property)](api/language   | -   [multi_qubit_depth            |
| s/python_api.html#cudaq.optimizer |     (cudaq.Resources              |
| s.GradientDescent.max_iterations) |     property)                     |
|     -   [(cudaq.optimizers.LBFGS  | ](api/languages/python_api.html#c |
|         property)](ap             | udaq.Resources.multi_qubit_depth) |
| i/languages/python_api.html#cudaq | -   [multi_qubit_gate_count       |
| .optimizers.LBFGS.max_iterations) |     (cudaq.Resources              |
|                                   |     property)](api                |
| -   [(cudaq.optimizers.NelderMead | /languages/python_api.html#cudaq. |
|         property)](api/lan        | Resources.multi_qubit_gate_count) |
| guages/python_api.html#cudaq.opti | -   [multiplicity                 |
| mizers.NelderMead.max_iterations) |     (cudaq.ptsbe.KrausTrajectory  |
|     -   [(cudaq.optimizers.SGD    |     property)](api/l              |
|         property)](               | anguages/python_api.html#cudaq.pt |
| api/languages/python_api.html#cud | sbe.KrausTrajectory.multiplicity) |
| aq.optimizers.SGD.max_iterations) |                                   |
|     -   [(cudaq.optimizers.SPSA   |                                   |
|         property)](a              |                                   |
| pi/languages/python_api.html#cuda |                                   |
| q.optimizers.SPSA.max_iterations) |                                   |
+-----------------------------------+-----------------------------------+

## N {#N}

+-----------------------------------+-----------------------------------+
| -   [name                         | -   [num_measurements             |
|                                   |     (cudaq.DEMResult              |
|  (cudaq.ptsbe.PTSSamplingStrategy |     property                      |
|     attribute)](a                 | )](api/languages/python_api.html# |
| pi/languages/python_api.html#cuda | cudaq.DEMResult.num_measurements) |
| q.ptsbe.PTSSamplingStrategy.name) | -   [num_observables              |
|     -                             |     (cudaq.DEMResult              |
|    [(cudaq.ptsbe.TraceInstruction |     propert                       |
|         property)                 | y)](api/languages/python_api.html |
| ](api/languages/python_api.html#c | #cudaq.DEMResult.num_observables) |
| udaq.ptsbe.TraceInstruction.name) | -   [num_qpus (cudaq.Target       |
|     -   [(cudaq.PyKernel          |                                   |
|                                   |   attribute)](api/languages/pytho |
|     attribute)](api/languages/pyt | n_api.html#cudaq.Target.num_qpus) |
| hon_api.html#cudaq.PyKernel.name) | -   [num_qubits (cudaq.Resources  |
|     -   [(cudaq.Target            |     pr                            |
|                                   | operty)](api/languages/python_api |
|        property)](api/languages/p | .html#cudaq.Resources.num_qubits) |
| ython_api.html#cudaq.Target.name) |     -   [(cudaq.State             |
| -   [NelderMead (class in         |                                   |
|     cudaq.optim                   |  attribute)](api/languages/python |
| izers)](api/languages/python_api. | _api.html#cudaq.State.num_qubits) |
| html#cudaq.optimizers.NelderMead) | -   [num_ranks() (in module       |
| -   [noise_type                   |     cudaq.mpi)](api/languages/pyt |
|     (cudaq.KrausChannel           | hon_api.html#cudaq.mpi.num_ranks) |
|     prope                         | -   [num_rows                     |
| rty)](api/languages/python_api.ht |     (cudaq.ComplexMatrix          |
| ml#cudaq.KrausChannel.noise_type) |     attri                         |
| -   [NoiseModel (class in         | bute)](api/languages/python_api.h |
|     cudaq)](api/languages/        | tml#cudaq.ComplexMatrix.num_rows) |
| python_api.html#cudaq.NoiseModel) | -   [num_shots                    |
| -   [normalized()                 |     (cudaq.ptsbe.KrausTrajectory  |
|                                   |     property)](ap                 |
|    (cudaq.synth.CliffordTSequence | i/languages/python_api.html#cudaq |
|     method)](api/l                | .ptsbe.KrausTrajectory.num_shots) |
| anguages/python_api.html#cudaq.sy | -   [num_used_qubits              |
| nth.CliffordTSequence.normalized) |     (cudaq.Resources              |
| -   [num_available_gpus() (in     |     propert                       |
|     module                        | y)](api/languages/python_api.html |
|                                   | #cudaq.Resources.num_used_qubits) |
|    cudaq)](api/languages/python_a | -   [nvqir::MPSSimulationState    |
| pi.html#cudaq.num_available_gpus) |     (C++                          |
| -   [num_columns                  |     class)]                       |
|     (cudaq.ComplexMatrix          | (api/languages/cpp_api.html#_CPPv |
|     attribut                      | 4I0EN5nvqir18MPSSimulationStateE) |
| e)](api/languages/python_api.html | -                                 |
| #cudaq.ComplexMatrix.num_columns) |  [nvqir::TensorNetSimulationState |
| -   [num_detectors                |     (C++                          |
|     (cudaq.DEMResult              |     class)](api/l                 |
|     prope                         | anguages/cpp_api.html#_CPPv4I0EN5 |
| rty)](api/languages/python_api.ht | nvqir24TensorNetSimulationStateE) |
| ml#cudaq.DEMResult.num_detectors) |                                   |
+-----------------------------------+-----------------------------------+

## O {#O}

+-----------------------------------+-----------------------------------+
| -   [observe() (in module         | -   [opt_value                    |
|     cudaq)](api/languag           |     (cudaq.OptimizationResult     |
| es/python_api.html#cudaq.observe) |     property)]                    |
| -   [observe_async() (in module   | (api/languages/python_api.html#cu |
|     cudaq)](api/languages/pyt     | daq.OptimizationResult.opt_value) |
| hon_api.html#cudaq.observe_async) | -   [optimal_parameters           |
| -   [ObserveResult (class in      |     (cudaq.OptimizationResult     |
|     cudaq)](api/languages/pyt     |     property)](api/lang           |
| hon_api.html#cudaq.ObserveResult) | uages/python_api.html#cudaq.Optim |
| -   [op_name                      | izationResult.optimal_parameters) |
|     (cudaq.ptsbe.KrausSelection   | -   [OptimizationResult (class in |
|     property)]                    |                                   |
| (api/languages/python_api.html#cu |    cudaq)](api/languages/python_a |
| daq.ptsbe.KrausSelection.op_name) | pi.html#cudaq.OptimizationResult) |
| -   [OperatorSum (in module       | -   [OrderedSamplingStrategy      |
|     cudaq.oper                    |     (class in                     |
| ators)](api/languages/python_api. |     cudaq.ptsbe)](                |
| html#cudaq.operators.OperatorSum) | api/languages/python_api.html#cud |
| -   [ops_count                    | aq.ptsbe.OrderedSamplingStrategy) |
|     (cudaq.                       | -   [overlap (cudaq.State         |
| operators.boson.BosonOperatorTerm |     attribute)](api/languages/pyt |
|     property)](api/languages/     | hon_api.html#cudaq.State.overlap) |
| python_api.html#cudaq.operators.b |                                   |
| oson.BosonOperatorTerm.ops_count) |                                   |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |                                   |
|                                   |                                   |
|     property)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorTerm.ops_count) |                                   |
|     -   [(c                       |                                   |
| udaq.operators.MatrixOperatorTerm |                                   |
|         property)](api/langu      |                                   |
| ages/python_api.html#cudaq.operat |                                   |
| ors.MatrixOperatorTerm.ops_count) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         property)](api/language   |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .spin.SpinOperatorTerm.ops_count) |                                   |
+-----------------------------------+-----------------------------------+

## P {#P}

+-----------------------------------+-----------------------------------+
| -   [parameters                   | -   [per_qubit_depth              |
|     (cudaq.KrausChannel           |     (cudaq.Resources              |
|     prope                         |     propert                       |
| rty)](api/languages/python_api.ht | y)](api/languages/python_api.html |
| ml#cudaq.KrausChannel.parameters) | #cudaq.Resources.per_qubit_depth) |
|     -   [(cu                      | -   [PhaseDamping (class in       |
| daq.operators.boson.BosonOperator |     cudaq)](api/languages/py      |
|         property)](api/languag    | thon_api.html#cudaq.PhaseDamping) |
| es/python_api.html#cudaq.operator | -   [PhaseFlipChannel (class in   |
| s.boson.BosonOperator.parameters) |     cudaq)](api/languages/python  |
|     -   [(cudaq.                  | _api.html#cudaq.PhaseFlipChannel) |
| operators.boson.BosonOperatorTerm | -   [platform (cudaq.Target       |
|                                   |                                   |
|        property)](api/languages/p |    property)](api/languages/pytho |
| ython_api.html#cudaq.operators.bo | n_api.html#cudaq.Target.platform) |
| son.BosonOperatorTerm.parameters) | -   [prepare_call()               |
|     -   [(cudaq.                  |     (cudaq.PyKernelDecorator      |
| operators.fermion.FermionOperator |     method)](a                    |
|                                   | pi/languages/python_api.html#cuda |
|        property)](api/languages/p | q.PyKernelDecorator.prepare_call) |
| ython_api.html#cudaq.operators.fe | -                                 |
| rmion.FermionOperator.parameters) |    [ProbabilisticSamplingStrategy |
|     -   [(cudaq.oper              |     (class in                     |
| ators.fermion.FermionOperatorTerm |     cudaq.ptsbe)](api/la          |
|                                   | nguages/python_api.html#cudaq.pts |
|    property)](api/languages/pytho | be.ProbabilisticSamplingStrategy) |
| n_api.html#cudaq.operators.fermio | -   [probability                  |
| n.FermionOperatorTerm.parameters) |     (cudaq.ptsbe.KrausTrajectory  |
|     -                             |     property)](api/               |
|  [(cudaq.operators.MatrixOperator | languages/python_api.html#cudaq.p |
|         property)](api/la         | tsbe.KrausTrajectory.probability) |
| nguages/python_api.html#cudaq.ope |     -   [(cudaq.SampleResult      |
| rators.MatrixOperator.parameters) |         attribu                   |
|     -   [(cuda                    | te)](api/languages/python_api.htm |
| q.operators.MatrixOperatorElement | l#cudaq.SampleResult.probability) |
|         property)](api/languages  | -   [process_call_arguments()     |
| /python_api.html#cudaq.operators. |     (cudaq.PyKernelDecorator      |
| MatrixOperatorElement.parameters) |     method)](api/languag          |
|     -   [(c                       | es/python_api.html#cudaq.PyKernel |
| udaq.operators.MatrixOperatorTerm | Decorator.process_call_arguments) |
|         property)](api/langua     | -   [ProductOperator (in module   |
| ges/python_api.html#cudaq.operato |     cudaq.operator                |
| rs.MatrixOperatorTerm.parameters) | s)](api/languages/python_api.html |
|     -                             | #cudaq.operators.ProductOperator) |
|  [(cudaq.operators.ScalarOperator | -   [PROPORTIONAL                 |
|         property)](api/la         |                                   |
| nguages/python_api.html#cudaq.ope |   (cudaq.ptsbe.ShotAllocationType |
| rators.ScalarOperator.parameters) |     attribute)](api/lang          |
|     -   [(                        | uages/python_api.html#cudaq.ptsbe |
| cudaq.operators.spin.SpinOperator | .ShotAllocationType.PROPORTIONAL) |
|         property)](api/langu      | -   [ptsbe_execution_data         |
| ages/python_api.html#cudaq.operat |                                   |
| ors.spin.SpinOperator.parameters) |    (cudaq.ptsbe.PTSBESampleResult |
|     -   [(cuda                    |     property)](api/languages/p    |
| q.operators.spin.SpinOperatorTerm | ython_api.html#cudaq.ptsbe.PTSBES |
|         property)](api/languages  | ampleResult.ptsbe_execution_data) |
| /python_api.html#cudaq.operators. | -   [PTSBEExecutionData (class in |
| spin.SpinOperatorTerm.parameters) |     cudaq.pts                     |
| -   [ParameterShift (class in     | be)](api/languages/python_api.htm |
|     cudaq.gradien                 | l#cudaq.ptsbe.PTSBEExecutionData) |
| ts)](api/languages/python_api.htm | -   [PTSBESampleResult (class in  |
| l#cudaq.gradients.ParameterShift) |     cudaq.pt                      |
| -   [params                       | sbe)](api/languages/python_api.ht |
|     (cudaq.ptsbe.TraceInstruction | ml#cudaq.ptsbe.PTSBESampleResult) |
|     property)](                   | -   [PTSSamplingStrategy (class   |
| api/languages/python_api.html#cud |     in                            |
| aq.ptsbe.TraceInstruction.params) |     cudaq.ptsb                    |
| -   [parse_args() (in module      | e)](api/languages/python_api.html |
|     cudaq)](api/languages/        | #cudaq.ptsbe.PTSSamplingStrategy) |
| python_api.html#cudaq.parse_args) | -   [PyKernel (class in           |
| -   [Pauli1 (class in             |     cudaq)](api/language          |
|     cudaq)](api/langua            | s/python_api.html#cudaq.PyKernel) |
| ges/python_api.html#cudaq.Pauli1) | -   [PyKernelDecorator (class in  |
| -   [Pauli2 (class in             |     cudaq)](api/languages/python_ |
|     cudaq)](api/langua            | api.html#cudaq.PyKernelDecorator) |
| ges/python_api.html#cudaq.Pauli2) |                                   |
+-----------------------------------+-----------------------------------+

## Q {#Q}

+-----------------------------------+-----------------------------------+
| -   [qkeModule                    | -   [qubit_count                  |
|     (cudaq.PyKernelDecorator      |     (                             |
|     property)                     | cudaq.operators.spin.SpinOperator |
| ](api/languages/python_api.html#c |     property)](api/langua         |
| udaq.PyKernelDecorator.qkeModule) | ges/python_api.html#cudaq.operato |
| -   [qreg (in module              | rs.spin.SpinOperator.qubit_count) |
|     cudaq)](api/lang              |     -   [(cuda                    |
| uages/python_api.html#cudaq.qreg) | q.operators.spin.SpinOperatorTerm |
| -   [QuakeValue (class in         |         property)](api/languages/ |
|     cudaq)](api/languages/        | python_api.html#cudaq.operators.s |
| python_api.html#cudaq.QuakeValue) | pin.SpinOperatorTerm.qubit_count) |
| -   [qubit (class in              | -   [qubits                       |
|     cudaq)](api/langu             |     (cudaq.ptsbe.KrausSelection   |
| ages/python_api.html#cudaq.qubit) |     property)                     |
|                                   | ](api/languages/python_api.html#c |
|                                   | udaq.ptsbe.KrausSelection.qubits) |
|                                   | -   [qvector (class in            |
|                                   |     cudaq)](api/languag           |
|                                   | es/python_api.html#cudaq.qvector) |
+-----------------------------------+-----------------------------------+

## R {#R}

+-----------------------------------+-----------------------------------+
| -   [random                       | -   [resources                    |
|     (                             |     (cudaq.EstimateResult         |
| cudaq.operators.spin.SpinOperator |     proper                        |
|     attribute)](api/l             | ty)](api/languages/python_api.htm |
| anguages/python_api.html#cudaq.op | l#cudaq.EstimateResult.resources) |
| erators.spin.SpinOperator.random) | -   [right_multiply               |
| -   [rank() (in module            |     (cudaq.SuperOperator          |
|     cudaq.mpi)](api/language      |     attribute)]                   |
| s/python_api.html#cudaq.mpi.rank) | (api/languages/python_api.html#cu |
| -   [register_names               | daq.SuperOperator.right_multiply) |
|     (cudaq.SampleResult           | -   [row_count                    |
|     property)                     |     (cudaq.KrausOperator          |
| ](api/languages/python_api.html#c |     prope                         |
| udaq.SampleResult.register_names) | rty)](api/languages/python_api.ht |
| -                                 | ml#cudaq.KrausOperator.row_count) |
|   [register_set_target_callback() | -   [run() (in module             |
|     (in module                    |     cudaq)](api/lan               |
|     cudaq)]                       | guages/python_api.html#cudaq.run) |
| (api/languages/python_api.html#cu | -   [run_async() (in module       |
| daq.register_set_target_callback) |     cudaq)](api/languages         |
| -   [reset_target() (in module    | /python_api.html#cudaq.run_async) |
|     cudaq)](api/languages/py      | -   [RydbergHamiltonian (class in |
| thon_api.html#cudaq.reset_target) |     cudaq.operators)]             |
| -   [resolve_captured_arguments() | (api/languages/python_api.html#cu |
|     (cudaq.PyKernelDecorator      | daq.operators.RydbergHamiltonian) |
|     method)](api/languages/p      | -   [rz_error() (in module        |
| ython_api.html#cudaq.PyKernelDeco |                                   |
| rator.resolve_captured_arguments) |  cudaq.synth)](api/languages/pyth |
| -   [Resources (class in          | on_api.html#cudaq.synth.rz_error) |
|     cudaq)](api/languages         |                                   |
| /python_api.html#cudaq.Resources) |                                   |
+-----------------------------------+-----------------------------------+

## S {#S}

+-----------------------------------+-----------------------------------+
| -   [sample() (in module          | -   [ShotAllocationStrategy       |
|     cudaq)](api/langua            |     (class in                     |
| ges/python_api.html#cudaq.sample) |     cudaq.ptsbe)]                 |
|     -   [(in module               | (api/languages/python_api.html#cu |
|                                   | daq.ptsbe.ShotAllocationStrategy) |
|      cudaq.orca)](api/languages/p | -   [ShotAllocationType (class in |
| ython_api.html#cudaq.orca.sample) |     cudaq.pts                     |
|     -   [(in module               | be)](api/languages/python_api.htm |
|                                   | l#cudaq.ptsbe.ShotAllocationType) |
|    cudaq.ptsbe)](api/languages/py | -   [signatureWithCallables()     |
| thon_api.html#cudaq.ptsbe.sample) |     (cudaq.PyKernelDecorator      |
| -   [sample_async() (in module    |     method)](api/languag          |
|     cudaq)](api/languages/py      | es/python_api.html#cudaq.PyKernel |
| thon_api.html#cudaq.sample_async) | Decorator.signatureWithCallables) |
|     -   [(in module               | -   [SimulationPrecision (class   |
|         cud                       |     in                            |
| aq.ptsbe)](api/languages/python_a |                                   |
| pi.html#cudaq.ptsbe.sample_async) |   cudaq)](api/languages/python_ap |
| -   [SampleResult (class in       | i.html#cudaq.SimulationPrecision) |
|     cudaq)](api/languages/py      | -   [simulator (cudaq.Target      |
| thon_api.html#cudaq.SampleResult) |                                   |
| -   [scalarCaster()               |   property)](api/languages/python |
|     (cudaq.PyKernelDecorator      | _api.html#cudaq.Target.simulator) |
|     method)](a                    | -   [slice() (cudaq.QuakeValue    |
| pi/languages/python_api.html#cuda |     method)](api/languages/python |
| q.PyKernelDecorator.scalarCaster) | _api.html#cudaq.QuakeValue.slice) |
| -   [ScalarOperator (class in     | -   [SpinOperator (class in       |
|     cudaq.operato                 |     cudaq.operators.spin)         |
| rs)](api/languages/python_api.htm | ](api/languages/python_api.html#c |
| l#cudaq.operators.ScalarOperator) | udaq.operators.spin.SpinOperator) |
| -   [Schedule (class in           | -   [SpinOperatorElement (class   |
|     cudaq)](api/language          |     in                            |
| s/python_api.html#cudaq.Schedule) |     cudaq.operators.spin)](api/l  |
| -   [serialize                    | anguages/python_api.html#cudaq.op |
|     (                             | erators.spin.SpinOperatorElement) |
| cudaq.operators.spin.SpinOperator | -   [SpinOperatorTerm (class in   |
|     attribute)](api/lang          |     cudaq.operators.spin)](ap     |
| uages/python_api.html#cudaq.opera | i/languages/python_api.html#cudaq |
| tors.spin.SpinOperator.serialize) | .operators.spin.SpinOperatorTerm) |
|     -   [(cuda                    | -   [split_communicator() (in     |
| q.operators.spin.SpinOperatorTerm |     module                        |
|         attribute)](api/language  |     cudaq                         |
| s/python_api.html#cudaq.operators | .mpi)](api/languages/python_api.h |
| .spin.SpinOperatorTerm.serialize) | tml#cudaq.mpi.split_communicator) |
|     -   [(cudaq.SampleResult      | -   [SPSA (class in               |
|         attri                     |     cudaq                         |
| bute)](api/languages/python_api.h | .optimizers)](api/languages/pytho |
| tml#cudaq.SampleResult.serialize) | n_api.html#cudaq.optimizers.SPSA) |
| -   [set_communicator() (in       | -   [State (class in              |
|     module                        |     cudaq)](api/langu             |
|     cud                           | ages/python_api.html#cudaq.State) |
| aq.mpi)](api/languages/python_api | -   [step_size                    |
| .html#cudaq.mpi.set_communicator) |     (cudaq.optimizers.Adam        |
| -   [set_noise() (in module       |     propert                       |
|     cudaq)](api/languages         | y)](api/languages/python_api.html |
| /python_api.html#cudaq.set_noise) | #cudaq.optimizers.Adam.step_size) |
| -   [set_random_seed() (in module |     -   [(cudaq.optimizers.SGD    |
|     cudaq)](api/languages/pytho   |         proper                    |
| n_api.html#cudaq.set_random_seed) | ty)](api/languages/python_api.htm |
| -   [set_target() (in module      | l#cudaq.optimizers.SGD.step_size) |
|     cudaq)](api/languages/        |     -   [(cudaq.optimizers.SPSA   |
| python_api.html#cudaq.set_target) |         propert                   |
| -   [SGD (class in                | y)](api/languages/python_api.html |
|     cuda                          | #cudaq.optimizers.SPSA.step_size) |
| q.optimizers)](api/languages/pyth | -   [SuperOperator (class in      |
| on_api.html#cudaq.optimizers.SGD) |     cudaq)](api/languages/pyt     |
|                                   | hon_api.html#cudaq.SuperOperator) |
|                                   | -   [supports_compilation()       |
|                                   |     (cudaq.PyKernelDecorator      |
|                                   |     method)](api/langu            |
|                                   | ages/python_api.html#cudaq.PyKern |
|                                   | elDecorator.supports_compilation) |
+-----------------------------------+-----------------------------------+

## T {#T}

+-----------------------------------+-----------------------------------+
| -   [t_count                      | -   [to_matrix()                  |
|                                   |                                   |
|    (cudaq.synth.CliffordTSequence |   (cudaq.operators.ScalarOperator |
|     property)](ap                 |     method)](api/l                |
| i/languages/python_api.html#cudaq | anguages/python_api.html#cudaq.op |
| .synth.CliffordTSequence.t_count) | erators.ScalarOperator.to_matrix) |
| -   [t_depth (cudaq.Resources     | -   [to_numpy                     |
|                                   |     (cudaq.ComplexMatrix          |
|  property)](api/languages/python_ |     attri                         |
| api.html#cudaq.Resources.t_depth) | bute)](api/languages/python_api.h |
| -   [Target (class in             | tml#cudaq.ComplexMatrix.to_numpy) |
|     cudaq)](api/langua            |     -   [(cudaq.State             |
| ges/python_api.html#cudaq.Target) |                                   |
| -   [target                       |    attribute)](api/languages/pyth |
|     (cudaq.ope                    | on_api.html#cudaq.State.to_numpy) |
| rators.boson.BosonOperatorElement | -   [to_sparse_matrix             |
|     property)](api/languages/     |     (cu                           |
| python_api.html#cudaq.operators.b | daq.operators.boson.BosonOperator |
| oson.BosonOperatorElement.target) |     attribute)](api/languages/pyt |
|     -   [(cudaq.operato           | hon_api.html#cudaq.operators.boso |
| rs.fermion.FermionOperatorElement | n.BosonOperator.to_sparse_matrix) |
|                                   |     -   [(cudaq.                  |
|     property)](api/languages/pyth | operators.boson.BosonOperatorTerm |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorElement.target) | attribute)](api/languages/python_ |
|     -   [(cudaq.o                 | api.html#cudaq.operators.boson.Bo |
| perators.spin.SpinOperatorElement | sonOperatorTerm.to_sparse_matrix) |
|         property)](api/language   |     -   [(cudaq.                  |
| s/python_api.html#cudaq.operators | operators.fermion.FermionOperator |
| .spin.SpinOperatorElement.target) |                                   |
| -   [targets                      | attribute)](api/languages/python_ |
|     (cudaq.ptsbe.TraceInstruction | api.html#cudaq.operators.fermion. |
|     property)](a                  | FermionOperator.to_sparse_matrix) |
| pi/languages/python_api.html#cuda |     -   [(cudaq.oper              |
| q.ptsbe.TraceInstruction.targets) | ators.fermion.FermionOperatorTerm |
| -   [Tensor (class in             |         attr                      |
|     cudaq)](api/langua            | ibute)](api/languages/python_api. |
| ges/python_api.html#cudaq.Tensor) | html#cudaq.operators.fermion.Ferm |
| -   [term_count                   | ionOperatorTerm.to_sparse_matrix) |
|     (cu                           |     -   [(                        |
| daq.operators.boson.BosonOperator | cudaq.operators.spin.SpinOperator |
|     property)](api/languag        |                                   |
| es/python_api.html#cudaq.operator |       attribute)](api/languages/p |
| s.boson.BosonOperator.term_count) | ython_api.html#cudaq.operators.sp |
|     -   [(cudaq.                  | in.SpinOperator.to_sparse_matrix) |
| operators.fermion.FermionOperator |     -   [(cuda                    |
|                                   | q.operators.spin.SpinOperatorTerm |
|        property)](api/languages/p |                                   |
| ython_api.html#cudaq.operators.fe |   attribute)](api/languages/pytho |
| rmion.FermionOperator.term_count) | n_api.html#cudaq.operators.spin.S |
|     -                             | pinOperatorTerm.to_sparse_matrix) |
|  [(cudaq.operators.MatrixOperator | -   [to_string                    |
|         property)](api/la         |     (cudaq.ope                    |
| nguages/python_api.html#cudaq.ope | rators.boson.BosonOperatorElement |
| rators.MatrixOperator.term_count) |     attribute)](api/languages/pyt |
|     -   [(                        | hon_api.html#cudaq.operators.boso |
| cudaq.operators.spin.SpinOperator | n.BosonOperatorElement.to_string) |
|         property)](api/langu      |     -   [(cudaq.operato           |
| ages/python_api.html#cudaq.operat | rs.fermion.FermionOperatorElement |
| ors.spin.SpinOperator.term_count) |                                   |
|     -   [(cuda                    | attribute)](api/languages/python_ |
| q.operators.spin.SpinOperatorTerm | api.html#cudaq.operators.fermion. |
|         property)](api/languages  | FermionOperatorElement.to_string) |
| /python_api.html#cudaq.operators. |     -   [(cuda                    |
| spin.SpinOperatorTerm.term_count) | q.operators.MatrixOperatorElement |
| -   [term_id                      |         attribute)](api/language  |
|     (cudaq.                       | s/python_api.html#cudaq.operators |
| operators.boson.BosonOperatorTerm | .MatrixOperatorElement.to_string) |
|     property)](api/language       |     -   [(cudaq.o                 |
| s/python_api.html#cudaq.operators | perators.spin.SpinOperatorElement |
| .boson.BosonOperatorTerm.term_id) |                                   |
|     -   [(cudaq.oper              |       attribute)](api/languages/p |
| ators.fermion.FermionOperatorTerm | ython_api.html#cudaq.operators.sp |
|                                   | in.SpinOperatorElement.to_string) |
|       property)](api/languages/py | -   [TraceInstruction (class in   |
| thon_api.html#cudaq.operators.fer |     cudaq.p                       |
| mion.FermionOperatorTerm.term_id) | tsbe)](api/languages/python_api.h |
|     -   [(c                       | tml#cudaq.ptsbe.TraceInstruction) |
| udaq.operators.MatrixOperatorTerm | -   [TraceInstructionType (class  |
|         property)](api/lan        |     in                            |
| guages/python_api.html#cudaq.oper |     cudaq.ptsbe                   |
| ators.MatrixOperatorTerm.term_id) | )](api/languages/python_api.html# |
|     -   [(cuda                    | cudaq.ptsbe.TraceInstructionType) |
| q.operators.spin.SpinOperatorTerm | -   [trajectories                 |
|         property)](api/langua     |                                   |
| ges/python_api.html#cudaq.operato |   (cudaq.ptsbe.PTSBEExecutionData |
| rs.spin.SpinOperatorTerm.term_id) |     property)](api/lang           |
| -   [to_bools() (in module        | uages/python_api.html#cudaq.ptsbe |
|     cudaq)](api/language          | .PTSBEExecutionData.trajectories) |
| s/python_api.html#cudaq.to_bools) | -   [trajectory_id                |
| -   [to_dict (cudaq.Resources     |     (cudaq.ptsbe.KrausTrajectory  |
|                                   |     property)](api/la             |
| attribute)](api/languages/python_ | nguages/python_api.html#cudaq.pts |
| api.html#cudaq.Resources.to_dict) | be.KrausTrajectory.trajectory_id) |
| -   [to_json                      | -   [translate() (in module       |
|     (                             |     cudaq)](api/languages         |
| cudaq.operators.spin.SpinOperator | /python_api.html#cudaq.translate) |
|     attribute)](api/la            | -   [trim                         |
| nguages/python_api.html#cudaq.ope |     (cu                           |
| rators.spin.SpinOperator.to_json) | daq.operators.boson.BosonOperator |
|     -   [(cuda                    |     attribute)](api/l             |
| q.operators.spin.SpinOperatorTerm | anguages/python_api.html#cudaq.op |
|         attribute)](api/langua    | erators.boson.BosonOperator.trim) |
| ges/python_api.html#cudaq.operato |     -   [(cudaq.                  |
| rs.spin.SpinOperatorTerm.to_json) | operators.fermion.FermionOperator |
| -   [to_kernel()                  |         attribute)](api/langu     |
|                                   | ages/python_api.html#cudaq.operat |
|    (cudaq.synth.CliffordTSequence | ors.fermion.FermionOperator.trim) |
|     method)](api/                 |     -                             |
| languages/python_api.html#cudaq.s |  [(cudaq.operators.MatrixOperator |
| ynth.CliffordTSequence.to_kernel) |         attribute)](              |
| -   [to_matrix                    | api/languages/python_api.html#cud |
|     (cu                           | aq.operators.MatrixOperator.trim) |
| daq.operators.boson.BosonOperator |     -   [(                        |
|     attribute)](api/langua        | cudaq.operators.spin.SpinOperator |
| ges/python_api.html#cudaq.operato |         attribute)](api           |
| rs.boson.BosonOperator.to_matrix) | /languages/python_api.html#cudaq. |
|     -   [(cudaq.ope               | operators.spin.SpinOperator.trim) |
| rators.boson.BosonOperatorElement | -   [type                         |
|                                   |     (c                            |
|     attribute)](api/languages/pyt | udaq.ptsbe.ShotAllocationStrategy |
| hon_api.html#cudaq.operators.boso |     property)](api/               |
| n.BosonOperatorElement.to_matrix) | languages/python_api.html#cudaq.p |
|     -   [(cudaq.                  | tsbe.ShotAllocationStrategy.type) |
| operators.boson.BosonOperatorTerm |     -                             |
|                                   |    [(cudaq.ptsbe.TraceInstruction |
|        attribute)](api/languages/ |         property)                 |
| python_api.html#cudaq.operators.b | ](api/languages/python_api.html#c |
| oson.BosonOperatorTerm.to_matrix) | udaq.ptsbe.TraceInstruction.type) |
|     -   [(cudaq.                  |                                   |
| operators.fermion.FermionOperator |                                   |
|                                   |                                   |
|        attribute)](api/languages/ |                                   |
| python_api.html#cudaq.operators.f |                                   |
| ermion.FermionOperator.to_matrix) |                                   |
|     -   [(cudaq.operato           |                                   |
| rs.fermion.FermionOperatorElement |                                   |
|                                   |                                   |
| attribute)](api/languages/python_ |                                   |
| api.html#cudaq.operators.fermion. |                                   |
| FermionOperatorElement.to_matrix) |                                   |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |                                   |
|                                   |                                   |
|    attribute)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorTerm.to_matrix) |                                   |
|     -                             |                                   |
|  [(cudaq.operators.MatrixOperator |                                   |
|         attribute)](api/l         |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| erators.MatrixOperator.to_matrix) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.MatrixOperatorElement |                                   |
|         attribute)](api/language  |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .MatrixOperatorElement.to_matrix) |                                   |
|     -   [(c                       |                                   |
| udaq.operators.MatrixOperatorTerm |                                   |
|         attribute)](api/langu     |                                   |
| ages/python_api.html#cudaq.operat |                                   |
| ors.MatrixOperatorTerm.to_matrix) |                                   |
|     -   [(                        |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|         attribute)](api/lang      |                                   |
| uages/python_api.html#cudaq.opera |                                   |
| tors.spin.SpinOperator.to_matrix) |                                   |
|     -   [(cudaq.o                 |                                   |
| perators.spin.SpinOperatorElement |                                   |
|                                   |                                   |
|       attribute)](api/languages/p |                                   |
| ython_api.html#cudaq.operators.sp |                                   |
| in.SpinOperatorElement.to_matrix) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         attribute)](api/language  |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .spin.SpinOperatorTerm.to_matrix) |                                   |
+-----------------------------------+-----------------------------------+

## U {#U}

+-----------------------------------------------------------------------+
| -   [UNIFORM (cudaq.ptsbe.ShotAllocationType                          |
|     attribute)](                                                      |
| api/languages/python_api.html#cudaq.ptsbe.ShotAllocationType.UNIFORM) |
| -   [unregister_set_target_callback() (in module                      |
|     cudaq)                                                            |
| ](api/languages/python_api.html#cudaq.unregister_set_target_callback) |
| -   [unset_noise() (in module                                         |
|     cudaq)](api/languages/python_api.html#cudaq.unset_noise)          |
| -   [upper_bounds (cudaq.optimizers.Adam                              |
|     propert                                                           |
| y)](api/languages/python_api.html#cudaq.optimizers.Adam.upper_bounds) |
|     -   [(cudaq.optimizers.COBYLA                                     |
|         property)                                                     |
| ](api/languages/python_api.html#cudaq.optimizers.COBYLA.upper_bounds) |
|     -   [(cudaq.optimizers.GradientDescent                            |
|         property)](api/lan                                            |
| guages/python_api.html#cudaq.optimizers.GradientDescent.upper_bounds) |
|     -   [(cudaq.optimizers.LBFGS                                      |
|         property                                                      |
| )](api/languages/python_api.html#cudaq.optimizers.LBFGS.upper_bounds) |
|     -   [(cudaq.optimizers.NelderMead                                 |
|         property)](ap                                                 |
| i/languages/python_api.html#cudaq.optimizers.NelderMead.upper_bounds) |
|     -   [(cudaq.optimizers.SGD                                        |
|         proper                                                        |
| ty)](api/languages/python_api.html#cudaq.optimizers.SGD.upper_bounds) |
|     -   [(cudaq.optimizers.SPSA                                       |
|         propert                                                       |
| y)](api/languages/python_api.html#cudaq.optimizers.SPSA.upper_bounds) |
+-----------------------------------------------------------------------+

## V {#V}

+-----------------------------------+-----------------------------------+
| -   [values (cudaq.SampleResult   | -   [vqe() (in module             |
|     at                            |     cudaq)](api/lan               |
| tribute)](api/languages/python_ap | guages/python_api.html#cudaq.vqe) |
| i.html#cudaq.SampleResult.values) |                                   |
+-----------------------------------+-----------------------------------+

## W {#W}

+-----------------------------------------------------------------------+
| -   [weight (cudaq.ptsbe.KrausTrajectory                              |
|     propert                                                           |
| y)](api/languages/python_api.html#cudaq.ptsbe.KrausTrajectory.weight) |
+-----------------------------------------------------------------------+

## X {#X}

+-----------------------------------+-----------------------------------+
| -   [X (cudaq.spin.Pauli          | -   [XError (class in             |
|     attribute)](api/languages/py  |     cudaq)](api/langua            |
| thon_api.html#cudaq.spin.Pauli.X) | ges/python_api.html#cudaq.XError) |
+-----------------------------------+-----------------------------------+

## Y {#Y}

+-----------------------------------+-----------------------------------+
| -   [Y (cudaq.spin.Pauli          | -   [YError (class in             |
|     attribute)](api/languages/py  |     cudaq)](api/langua            |
| thon_api.html#cudaq.spin.Pauli.Y) | ges/python_api.html#cudaq.YError) |
+-----------------------------------+-----------------------------------+

## Z {#Z}

+-----------------------------------+-----------------------------------+
| -   [Z (cudaq.spin.Pauli          | -   [ZError (class in             |
|     attribute)](api/languages/py  |     cudaq)](api/langua            |
| thon_api.html#cudaq.spin.Pauli.Z) | ges/python_api.html#cudaq.ZError) |
+-----------------------------------+-----------------------------------+
:::
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
