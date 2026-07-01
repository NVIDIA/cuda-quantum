::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
0.15.0
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
        -   [Limitations](using/examples/dem_from_kernel.html#limitations){.reference
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
        -   [Quantinuum](using/examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](using/examples/hardware_providers.html#quantum-circuits-inc){.reference
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
            -   [Quantum Circuits,
                Inc.](using/backends/hardware/superconducting.html#quantum-circuits-inc){.reference
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
        -   [Setup](using/realtime/installation.html#setup){.reference
            .internal}
        -   [Latency
            Measurement](using/realtime/installation.html#latency-measurement){.reference
            .internal}
    -   [Host API](using/realtime/host.html){.reference .internal}
        -   [What is
            HSB?](using/realtime/host.html#what-is-hsb){.reference
            .internal}
        -   [Transport
            Mechanisms](using/realtime/host.html#transport-mechanisms){.reference
            .internal}
            -   [Supported Transport
                Options](using/realtime/host.html#supported-transport-options){.reference
                .internal}
        -   [The 3-Kernel Architecture (HSB Example)
            {#three-kernel-architecture}](using/realtime/host.html#the-3-kernel-architecture-hsb-example-three-kernel-architecture){.reference
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
                HSB)](using/realtime/host.html#wiring-example-unified-mode-with-hsb){.reference
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
        -   [HSB 3-Kernel Workflow
            (Primary)](using/realtime/host.html#hsb-3-kernel-workflow-primary){.reference
            .internal}
        -   [NIC-Free Testing (No HSB / No
            ConnectX-7)](using/realtime/host.html#nic-free-testing-no-hsb-no-connectx-7){.reference
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
-   [CUDA-QX](using/cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q
        Solvers](using/cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](using/cudaqx/cudaqx.html#cuda-q-qec){.reference
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
    -   [Add a new Hardware
        Backend](using/extending/backend.html){.reference .internal}
        -   [Overview](using/extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](using/extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](using/extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](using/extending/backend.html#server-helper-class){.reference
                .internal}
            -   [[`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](using/extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](using/extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent [`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](using/extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](using/extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](using/extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](using/extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](using/extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](using/extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](using/extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](using/extending/backend.html#example-usage){.reference
            .internal}
        -   [Code
            Review](using/extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](using/extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](using/extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](using/extending/nvqir_simulator.html){.reference
        .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](using/extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](using/extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q
        IR](using/extending/cudaq_ir.html){.reference .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](using/extending/mlir_pass.html){.reference .internal}
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
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
            -   [[`dem_from_kernel()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.dem_from_kernel){.reference
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
| -   [AmplitudeDampingChannel      | -   [AsyncObserveResult (class in |
|     (class in                     |                                   |
|     cu                            |    cudaq)](api/languages/python_a |
| daq)](api/languages/python_api.ht | pi.html#cudaq.AsyncObserveResult) |
| ml#cudaq.AmplitudeDampingChannel) | -   [AsyncSampleResult (class in  |
| -   [amplitudes (cudaq.State      |     cudaq)](api/languages/python_ |
|                                   | api.html#cudaq.AsyncSampleResult) |
|  attribute)](api/languages/python | -   [AsyncStateResult (class in   |
| _api.html#cudaq.State.amplitudes) |     cudaq)](api/languages/python  |
|                                   | _api.html#cudaq.AsyncStateResult) |
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
| -   [cachedCompiledModule()       | -   [cudaq::produ                 |
|     (cudaq.PyKernelDecorator      | ct_op::const_iterator::operator\* |
|     method)](api/langu            |     (C++                          |
| ages/python_api.html#cudaq.PyKern |     function)](api/lang           |
| elDecorator.cachedCompiledModule) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [canonicalize                 | 10product_op14const_iteratormlEv) |
|     (cu                           | -   [cudaq::produ                 |
| daq.operators.boson.BosonOperator | ct_op::const_iterator::operator++ |
|     attribute)](api/languages     |     (C++                          |
| /python_api.html#cudaq.operators. |     function)](api/lang           |
| boson.BosonOperator.canonicalize) | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     -   [(cudaq.                  | 0product_op14const_iteratorppEi), |
| operators.boson.BosonOperatorTerm |     [\[1\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|     attribute)](api/languages/pyt | 10product_op14const_iteratorppEv) |
| hon_api.html#cudaq.operators.boso | -   [cudaq::produc                |
| n.BosonOperatorTerm.canonicalize) | t_op::const_iterator::operator\-- |
|     -   [(cudaq.                  |     (C++                          |
| operators.fermion.FermionOperator |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     attribute)](api/languages/pyt | 0product_op14const_iteratormmEi), |
| hon_api.html#cudaq.operators.ferm |     [\[1\]](api/lan               |
| ion.FermionOperator.canonicalize) | guages/cpp_api.html#_CPPv4N5cudaq |
|     -   [(cudaq.oper              | 10product_op14const_iteratormmEv) |
| ators.fermion.FermionOperatorTerm | -   [cudaq::produc                |
|                                   | t_op::const_iterator::operator-\> |
| attribute)](api/languages/python_ |     (C++                          |
| api.html#cudaq.operators.fermion. |     function)](api/lan            |
| FermionOperatorTerm.canonicalize) | guages/cpp_api.html#_CPPv4N5cudaq |
|     -                             | 10product_op14const_iteratorptEv) |
|  [(cudaq.operators.MatrixOperator | -   [cudaq::produ                 |
|         attribute)](api/lang      | ct_op::const_iterator::operator== |
| uages/python_api.html#cudaq.opera |     (C++                          |
| tors.MatrixOperator.canonicalize) |     fun                           |
|     -   [(c                       | ction)](api/languages/cpp_api.htm |
| udaq.operators.MatrixOperatorTerm | l#_CPPv4NK5cudaq10product_op14con |
|         attribute)](api/language  | st_iteratoreqERK14const_iterator) |
| s/python_api.html#cudaq.operators | -   [cudaq::product_op::degrees   |
| .MatrixOperatorTerm.canonicalize) |     (C++                          |
|     -   [(                        |     function)                     |
| cudaq.operators.spin.SpinOperator | ](api/languages/cpp_api.html#_CPP |
|         attribute)](api/languag   | v4NK5cudaq10product_op7degreesEv) |
| es/python_api.html#cudaq.operator | -   [cudaq::product_op::dump (C++ |
| s.spin.SpinOperator.canonicalize) |     functi                        |
|     -   [(cuda                    | on)](api/languages/cpp_api.html#_ |
| q.operators.spin.SpinOperatorTerm | CPPv4NK5cudaq10product_op4dumpEv) |
|                                   | -   [cudaq::product_op::end (C++  |
|       attribute)](api/languages/p |     funct                         |
| ython_api.html#cudaq.operators.sp | ion)](api/languages/cpp_api.html# |
| in.SpinOperatorTerm.canonicalize) | _CPPv4NK5cudaq10product_op3endEv) |
| -   [captured_variables()         | -   [c                            |
|     (cudaq.PyKernelDecorator      | udaq::product_op::get_coefficient |
|     method)](api/lan              |     (C++                          |
| guages/python_api.html#cudaq.PyKe |     function)](api/lan            |
| rnelDecorator.captured_variables) | guages/cpp_api.html#_CPPv4NK5cuda |
| -   [CentralDifference (class in  | q10product_op15get_coefficientEv) |
|     cudaq.gradients)              | -                                 |
| ](api/languages/python_api.html#c |   [cudaq::product_op::get_term_id |
| udaq.gradients.CentralDifference) |     (C++                          |
| -   [channel                      |     function)](api                |
|     (cudaq.ptsbe.TraceInstruction | /languages/cpp_api.html#_CPPv4NK5 |
|     property)](a                  | cudaq10product_op11get_term_idEv) |
| pi/languages/python_api.html#cuda | -                                 |
| q.ptsbe.TraceInstruction.channel) |   [cudaq::product_op::is_identity |
| -   [circuit_location             |     (C++                          |
|     (cudaq.ptsbe.KrausSelection   |     function)](api                |
|     property)](api/lang           | /languages/cpp_api.html#_CPPv4NK5 |
| uages/python_api.html#cudaq.ptsbe | cudaq10product_op11is_identityEv) |
| .KrausSelection.circuit_location) | -   [cudaq::product_op::num_ops   |
| -   [clear (cudaq.Resources       |     (C++                          |
|                                   |     function)                     |
|   attribute)](api/languages/pytho | ](api/languages/cpp_api.html#_CPP |
| n_api.html#cudaq.Resources.clear) | v4NK5cudaq10product_op7num_opsEv) |
|     -   [(cudaq.SampleResult      | -                                 |
|         a                         |    [cudaq::product_op::operator\* |
| ttribute)](api/languages/python_a |     (C++                          |
| pi.html#cudaq.SampleResult.clear) |     function)](api/languages/     |
| -   [COBYLA (class in             | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     cudaq.o                       | oduct_opmlE10product_opI1TERK15sc |
| ptimizers)](api/languages/python_ | alar_operatorRK10product_opI1TE), |
| api.html#cudaq.optimizers.COBYLA) |     [\[1\]](api/languages/        |
| -   [coefficient                  | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     (cudaq.                       | oduct_opmlE10product_opI1TERK15sc |
| operators.boson.BosonOperatorTerm | alar_operatorRR10product_opI1TE), |
|     property)](api/languages/py   |     [\[2\]](api/languages/        |
| thon_api.html#cudaq.operators.bos | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| on.BosonOperatorTerm.coefficient) | oduct_opmlE10product_opI1TERR15sc |
|     -   [(cudaq.oper              | alar_operatorRK10product_opI1TE), |
| ators.fermion.FermionOperatorTerm |     [\[3\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|   property)](api/languages/python | oduct_opmlE10product_opI1TERR15sc |
| _api.html#cudaq.operators.fermion | alar_operatorRR10product_opI1TE), |
| .FermionOperatorTerm.coefficient) |     [\[4\]](api/                  |
|     -   [(c                       | languages/cpp_api.html#_CPPv4I0EN |
| udaq.operators.MatrixOperatorTerm | 5cudaq10product_opmlE6sum_opI1TER |
|         property)](api/languag    | K15scalar_operatorRK6sum_opI1TE), |
| es/python_api.html#cudaq.operator |     [\[5\]](api/                  |
| s.MatrixOperatorTerm.coefficient) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cuda                    | 5cudaq10product_opmlE6sum_opI1TER |
| q.operators.spin.SpinOperatorTerm | K15scalar_operatorRR6sum_opI1TE), |
|         property)](api/languages/ |     [\[6\]](api/                  |
| python_api.html#cudaq.operators.s | languages/cpp_api.html#_CPPv4I0EN |
| pin.SpinOperatorTerm.coefficient) | 5cudaq10product_opmlE6sum_opI1TER |
| -   [col_count                    | R15scalar_operatorRK6sum_opI1TE), |
|     (cudaq.KrausOperator          |     [\[7\]](api/                  |
|     prope                         | languages/cpp_api.html#_CPPv4I0EN |
| rty)](api/languages/python_api.ht | 5cudaq10product_opmlE6sum_opI1TER |
| ml#cudaq.KrausOperator.col_count) | R15scalar_operatorRR6sum_opI1TE), |
| -   [compile()                    |     [\[8\]](api/languages         |
|     (cudaq.PyKernelDecorator      | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     metho                         | duct_opmlERK6sum_opI9HandlerTyE), |
| d)](api/languages/python_api.html |     [\[9\]](api/languages/cpp_a   |
| #cudaq.PyKernelDecorator.compile) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [ComplexMatrix (class in      | opmlERK10product_opI9HandlerTyE), |
|     cudaq)](api/languages/pyt     |     [\[10\]](api/language         |
| hon_api.html#cudaq.ComplexMatrix) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [compute                      | roduct_opmlERK15scalar_operator), |
|     (                             |     [\[11\]](api/languages/cpp_a  |
| cudaq.gradients.CentralDifference | pi.html#_CPPv4NKR5cudaq10product_ |
|     attribute)](api/la            | opmlERR10product_opI9HandlerTyE), |
| nguages/python_api.html#cudaq.gra |     [\[12\]](api/language         |
| dients.CentralDifference.compute) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [(                        | roduct_opmlERR15scalar_operator), |
| cudaq.gradients.ForwardDifference |     [\[13\]](api/languages/cpp_   |
|         attribute)](api/la        | api.html#_CPPv4NO5cudaq10product_ |
| nguages/python_api.html#cudaq.gra | opmlERK10product_opI9HandlerTyE), |
| dients.ForwardDifference.compute) |     [\[14\]](api/languag          |
|     -                             | es/cpp_api.html#_CPPv4NO5cudaq10p |
|  [(cudaq.gradients.ParameterShift | roduct_opmlERK15scalar_operator), |
|         attribute)](api           |     [\[15\]](api/languages/cpp_   |
| /languages/python_api.html#cudaq. | api.html#_CPPv4NO5cudaq10product_ |
| gradients.ParameterShift.compute) | opmlERR10product_opI9HandlerTyE), |
| -   [const()                      |     [\[16\]](api/langua           |
|                                   | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|   (cudaq.operators.ScalarOperator | product_opmlERR15scalar_operator) |
|     class                         | -                                 |
|     method)](a                    |   [cudaq::product_op::operator\*= |
| pi/languages/python_api.html#cuda |     (C++                          |
| q.operators.ScalarOperator.const) |     function)](api/languages/cpp  |
| -   [controls                     | _api.html#_CPPv4N5cudaq10product_ |
|     (cudaq.ptsbe.TraceInstruction | opmLERK10product_opI9HandlerTyE), |
|     property)](ap                 |     [\[1\]](api/langua            |
| i/languages/python_api.html#cudaq | ges/cpp_api.html#_CPPv4N5cudaq10p |
| .ptsbe.TraceInstruction.controls) | roduct_opmLERK15scalar_operator), |
| -   [copy                         |     [\[2\]](api/languages/cp      |
|     (cu                           | p_api.html#_CPPv4N5cudaq10product |
| daq.operators.boson.BosonOperator | _opmLERR10product_opI9HandlerTyE) |
|     attribute)](api/l             | -   [cudaq::product_op::operator+ |
| anguages/python_api.html#cudaq.op |     (C++                          |
| erators.boson.BosonOperator.copy) |     function)](api/langu          |
|     -   [(cudaq.                  | ages/cpp_api.html#_CPPv4I0EN5cuda |
| operators.boson.BosonOperatorTerm | q10product_opplE6sum_opI1TERK15sc |
|         attribute)](api/langu     | alar_operatorRK10product_opI1TE), |
| ages/python_api.html#cudaq.operat |     [\[1\]](api/                  |
| ors.boson.BosonOperatorTerm.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cudaq.                  | 5cudaq10product_opplE6sum_opI1TER |
| operators.fermion.FermionOperator | K15scalar_operatorRK6sum_opI1TE), |
|         attribute)](api/langu     |     [\[2\]](api/langu             |
| ages/python_api.html#cudaq.operat | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ors.fermion.FermionOperator.copy) | q10product_opplE6sum_opI1TERK15sc |
|     -   [(cudaq.oper              | alar_operatorRR10product_opI1TE), |
| ators.fermion.FermionOperatorTerm |     [\[3\]](api/                  |
|         attribute)](api/languages | languages/cpp_api.html#_CPPv4I0EN |
| /python_api.html#cudaq.operators. | 5cudaq10product_opplE6sum_opI1TER |
| fermion.FermionOperatorTerm.copy) | K15scalar_operatorRR6sum_opI1TE), |
|     -                             |     [\[4\]](api/langu             |
|  [(cudaq.operators.MatrixOperator | ages/cpp_api.html#_CPPv4I0EN5cuda |
|         attribute)](              | q10product_opplE6sum_opI1TERR15sc |
| api/languages/python_api.html#cud | alar_operatorRK10product_opI1TE), |
| aq.operators.MatrixOperator.copy) |     [\[5\]](api/                  |
|     -   [(c                       | languages/cpp_api.html#_CPPv4I0EN |
| udaq.operators.MatrixOperatorTerm | 5cudaq10product_opplE6sum_opI1TER |
|         attribute)](api/          | R15scalar_operatorRK6sum_opI1TE), |
| languages/python_api.html#cudaq.o |     [\[6\]](api/langu             |
| perators.MatrixOperatorTerm.copy) | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     -   [(                        | q10product_opplE6sum_opI1TERR15sc |
| cudaq.operators.spin.SpinOperator | alar_operatorRR10product_opI1TE), |
|         attribute)](api           |     [\[7\]](api/                  |
| /languages/python_api.html#cudaq. | languages/cpp_api.html#_CPPv4I0EN |
| operators.spin.SpinOperator.copy) | 5cudaq10product_opplE6sum_opI1TER |
|     -   [(cuda                    | R15scalar_operatorRR6sum_opI1TE), |
| q.operators.spin.SpinOperatorTerm |     [\[8\]](api/languages/cpp_a   |
|         attribute)](api/lan       | pi.html#_CPPv4NKR5cudaq10product_ |
| guages/python_api.html#cudaq.oper | opplERK10product_opI9HandlerTyE), |
| ators.spin.SpinOperatorTerm.copy) |     [\[9\]](api/language          |
| -   [count (cudaq.Resources       | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opplERK15scalar_operator), |
|   attribute)](api/languages/pytho |     [\[10\]](api/languages/       |
| n_api.html#cudaq.Resources.count) | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     -   [(cudaq.SampleResult      | duct_opplERK6sum_opI9HandlerTyE), |
|         a                         |     [\[11\]](api/languages/cpp_a  |
| ttribute)](api/languages/python_a | pi.html#_CPPv4NKR5cudaq10product_ |
| pi.html#cudaq.SampleResult.count) | opplERR10product_opI9HandlerTyE), |
| -   [count_controls               |     [\[12\]](api/language         |
|     (cudaq.Resources              | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     attribu                       | roduct_opplERR15scalar_operator), |
| te)](api/languages/python_api.htm |     [\[13\]](api/languages/       |
| l#cudaq.Resources.count_controls) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [count_instructions           | duct_opplERR6sum_opI9HandlerTyE), |
|                                   |     [\[                           |
|   (cudaq.ptsbe.PTSBEExecutionData | 14\]](api/languages/cpp_api.html# |
|     attribute)](api/languages/    | _CPPv4NKR5cudaq10product_opplEv), |
| python_api.html#cudaq.ptsbe.PTSBE |     [\[15\]](api/languages/cpp_   |
| ExecutionData.count_instructions) | api.html#_CPPv4NO5cudaq10product_ |
| -   [counts (cudaq.ObserveResult  | opplERK10product_opI9HandlerTyE), |
|     att                           |     [\[16\]](api/languag          |
| ribute)](api/languages/python_api | es/cpp_api.html#_CPPv4NO5cudaq10p |
| .html#cudaq.ObserveResult.counts) | roduct_opplERK15scalar_operator), |
| -   [csr_spmatrix (C++            |     [\[17\]](api/languages        |
|     type)](api/languages/c        | /cpp_api.html#_CPPv4NO5cudaq10pro |
| pp_api.html#_CPPv412csr_spmatrix) | duct_opplERK6sum_opI9HandlerTyE), |
| -   cudaq                         |     [\[18\]](api/languages/cpp_   |
|     -   [module](api/langua       | api.html#_CPPv4NO5cudaq10product_ |
| ges/python_api.html#module-cudaq) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq (C++                   |     [\[19\]](api/languag          |
|     type)](api/lan                | es/cpp_api.html#_CPPv4NO5cudaq10p |
| guages/cpp_api.html#_CPPv45cudaq) | roduct_opplERR15scalar_operator), |
| -   [cudaq.apply_noise() (in      |     [\[20\]](api/languages        |
|     module                        | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     cudaq)](api/languages/python_ | duct_opplERR6sum_opI9HandlerTyE), |
| api.html#cudaq.cudaq.apply_noise) |     [                             |
| -   cudaq.boson                   | \[21\]](api/languages/cpp_api.htm |
|     -   [module](api/languages/py | l#_CPPv4NO5cudaq10product_opplEv) |
| thon_api.html#module-cudaq.boson) | -   [cudaq::product_op::operator- |
| -   cudaq.fermion                 |     (C++                          |
|                                   |     function)](api/langu          |
|   -   [module](api/languages/pyth | ages/cpp_api.html#_CPPv4I0EN5cuda |
| on_api.html#module-cudaq.fermion) | q10product_opmiE6sum_opI1TERK15sc |
| -   cudaq.operators.custom        | alar_operatorRK10product_opI1TE), |
|     -   [mo                       |     [\[1\]](api/                  |
| dule](api/languages/python_api.ht | languages/cpp_api.html#_CPPv4I0EN |
| ml#module-cudaq.operators.custom) | 5cudaq10product_opmiE6sum_opI1TER |
| -   cudaq.spin                    | K15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/languages/p  |     [\[2\]](api/langu             |
| ython_api.html#module-cudaq.spin) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::amplitude_damping     | q10product_opmiE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     cla                           |     [\[3\]](api/                  |
| ss)](api/languages/cpp_api.html#_ | languages/cpp_api.html#_CPPv4I0EN |
| CPPv4N5cudaq17amplitude_dampingE) | 5cudaq10product_opmiE6sum_opI1TER |
| -                                 | K15scalar_operatorRR6sum_opI1TE), |
| [cudaq::amplitude_damping_channel |     [\[4\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     class)](api                   | q10product_opmiE6sum_opI1TERR15sc |
| /languages/cpp_api.html#_CPPv4N5c | alar_operatorRK10product_opI1TE), |
| udaq25amplitude_damping_channelE) |     [\[5\]](api/                  |
| -   [cudaq::amplitud              | languages/cpp_api.html#_CPPv4I0EN |
| e_damping_channel::num_parameters | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | R15scalar_operatorRK6sum_opI1TE), |
|     member)](api/languages/cpp_a  |     [\[6\]](api/langu             |
| pi.html#_CPPv4N5cudaq25amplitude_ | ages/cpp_api.html#_CPPv4I0EN5cuda |
| damping_channel14num_parametersE) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq::ampli                 | alar_operatorRR10product_opI1TE), |
| tude_damping_channel::num_targets |     [\[7\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     member)](api/languages/cp     | 5cudaq10product_opmiE6sum_opI1TER |
| p_api.html#_CPPv4N5cudaq25amplitu | R15scalar_operatorRR6sum_opI1TE), |
| de_damping_channel11num_targetsE) |     [\[8\]](api/languages/cpp_a   |
| -   [cudaq::AnalogRemoteRESTQPU   | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmiERK10product_opI9HandlerTyE), |
|     class                         |     [\[9\]](api/language          |
| )](api/languages/cpp_api.html#_CP | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | roduct_opmiERK15scalar_operator), |
| -   [cudaq::apply_noise (C++      |     [\[10\]](api/languages/       |
|     function)](api/               | cpp_api.html#_CPPv4NKR5cudaq10pro |
| languages/cpp_api.html#_CPPv4I0Dp | duct_opmiERK6sum_opI9HandlerTyE), |
| EN5cudaq11apply_noiseEvDpRR4Args) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq::async_result (C++     | pi.html#_CPPv4NKR5cudaq10product_ |
|     c                             | opmiERR10product_opI9HandlerTyE), |
| lass)](api/languages/cpp_api.html |     [\[12\]](api/language         |
| #_CPPv4I0EN5cudaq12async_resultE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::async_result::get     | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     functi                        | cpp_api.html#_CPPv4NKR5cudaq10pro |
| on)](api/languages/cpp_api.html#_ | duct_opmiERR6sum_opI9HandlerTyE), |
| CPPv4N5cudaq12async_result3getEv) |     [\[                           |
| -   [cudaq::async_sample_result   | 14\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NKR5cudaq10product_opmiEv), |
|     type                          |     [\[15\]](api/languages/cpp_   |
| )](api/languages/cpp_api.html#_CP | api.html#_CPPv4NO5cudaq10product_ |
| Pv4N5cudaq19async_sample_resultE) | opmiERK10product_opI9HandlerTyE), |
| -   [cudaq::BaseRemoteRESTQPU     |     [\[16\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     cla                           | roduct_opmiERK15scalar_operator), |
| ss)](api/languages/cpp_api.html#_ |     [\[17\]](api/languages        |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::bit_flip_channel (C++ | duct_opmiERK6sum_opI9HandlerTyE), |
|     cl                            |     [\[18\]](api/languages/cpp_   |
| ass)](api/languages/cpp_api.html# | api.html#_CPPv4NO5cudaq10product_ |
| _CPPv4N5cudaq16bit_flip_channelE) | opmiERR10product_opI9HandlerTyE), |
| -   [cudaq:                       |     [\[19\]](api/languag          |
| :bit_flip_channel::num_parameters | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmiERR15scalar_operator), |
|     member)](api/langua           |     [\[20\]](api/languages        |
| ges/cpp_api.html#_CPPv4N5cudaq16b | /cpp_api.html#_CPPv4NO5cudaq10pro |
| it_flip_channel14num_parametersE) | duct_opmiERR6sum_opI9HandlerTyE), |
| -   [cud                          |     [                             |
| aq::bit_flip_channel::num_targets | \[21\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NO5cudaq10product_opmiEv) |
|     member)](api/lan              | -   [cudaq::product_op::operator/ |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16bit_flip_channel11num_targetsE) |     function)](api/language       |
| -   [cudaq::boson_handler (C++    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opdvERK15scalar_operator), |
|  class)](api/languages/cpp_api.ht |     [\[1\]](api/language          |
| ml#_CPPv4N5cudaq13boson_handlerE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::boson_op (C++         | roduct_opdvERR15scalar_operator), |
|     type)](api/languages/cpp_     |     [\[2\]](api/languag           |
| api.html#_CPPv4N5cudaq8boson_opE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::boson_op_term (C++    | roduct_opdvERK15scalar_operator), |
|                                   |     [\[3\]](api/langua            |
|   type)](api/languages/cpp_api.ht | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| ml#_CPPv4N5cudaq13boson_op_termE) | product_opdvERR15scalar_operator) |
| -   [cudaq::CodeGenConfig (C++    | -                                 |
|                                   |    [cudaq::product_op::operator/= |
| struct)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     function)](api/langu          |
| -   [cudaq::commutation_relations | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     (C++                          | product_opdVERK15scalar_operator) |
|     struct)]                      | -   [cudaq::product_op::operator= |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq21commutation_relationsE) |     function)](api/l              |
| -   [cudaq::complex (C++          | anguages/cpp_api.html#_CPPv4I00EN |
|     type)](api/languages/cpp      | 5cudaq10product_opaSER10product_o |
| _api.html#_CPPv4N5cudaq7complexE) | pI9HandlerTyERK10product_opI1TE), |
| -   [cudaq::complex_matrix (C++   |     [\[1\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq10product_ |
| class)](api/languages/cpp_api.htm | opaSERK10product_opI9HandlerTyE), |
| l#_CPPv4N5cudaq14complex_matrixE) |     [\[2\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4N5cudaq10product |
|   [cudaq::complex_matrix::adjoint | _opaSERR10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     function)](a                  |    [cudaq::product_op::operator== |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14complex_matrix7adjointEv) |     function)](api/languages/cpp  |
| -   [cudaq::                      | _api.html#_CPPv4NK5cudaq10product |
| complex_matrix::diagonal_elements | _opeqERK10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     function)](api/languages      |  [cudaq::product_op::operator\[\] |
| /cpp_api.html#_CPPv4NK5cudaq14com |     (C++                          |
| plex_matrix17diagonal_elementsEi) |     function)](ap                 |
| -   [cudaq::complex_matrix::dump  | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq10product_opixENSt6size_tE) |
|     function)](api/language       | -                                 |
| s/cpp_api.html#_CPPv4NK5cudaq14co |    [cudaq::product_op::product_op |
| mplex_matrix4dumpERNSt7ostreamE), |     (C++                          |
|     [\[1\]]                       |     f                             |
| (api/languages/cpp_api.html#_CPPv | unction)](api/languages/cpp_api.h |
| 4NK5cudaq14complex_matrix4dumpEv) | tml#_CPPv4I00EN5cudaq10product_op |
| -   [c                            | 10product_opERK10product_opI1TE), |
| udaq::complex_matrix::eigenvalues |     [\[1\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/lan            | 4I00EN5cudaq10product_op10product |
| guages/cpp_api.html#_CPPv4NK5cuda | _opERK10product_opI1TERKN14matrix |
| q14complex_matrix11eigenvaluesEv) | _handler20commutation_behaviorE), |
| -   [cu                           |                                   |
| daq::complex_matrix::eigenvectors |   [\[2\]](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq10product_op10pr |
|     function)](api/lang           | oduct_opENSt6size_tENSt6size_tE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[3\]](api/languages/cp      |
| 14complex_matrix12eigenvectorsEv) | p_api.html#_CPPv4N5cudaq10product |
| -   [c                            | _op10product_opENSt7complexIdEE), |
| udaq::complex_matrix::exponential |     [\[4\]](api/l                 |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     function)](api/la             | aq10product_op10product_opERK10pr |
| nguages/cpp_api.html#_CPPv4N5cuda | oduct_opI9HandlerTyENSt6size_tE), |
| q14complex_matrix11exponentialEv) |     [\[5\]](api/l                 |
| -                                 | anguages/cpp_api.html#_CPPv4N5cud |
|  [cudaq::complex_matrix::identity | aq10product_op10product_opERR10pr |
|     (C++                          | oduct_opI9HandlerTyENSt6size_tE), |
|     function)](api/languages      |     [\[6\]](api/languages         |
| /cpp_api.html#_CPPv4N5cudaq14comp | /cpp_api.html#_CPPv4N5cudaq10prod |
| lex_matrix8identityEKNSt6size_tE) | uct_op10product_opERR9HandlerTy), |
| -                                 |     [\[7\]](ap                    |
| [cudaq::complex_matrix::kronecker | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq10product_op10product_opEd), |
|     function)](api/lang           |     [\[8\]](a                     |
| uages/cpp_api.html#_CPPv4I00EN5cu | pi/languages/cpp_api.html#_CPPv4N |
| daq14complex_matrix9kroneckerE14c | 5cudaq10product_op10product_opEv) |
| omplex_matrix8Iterable8Iterable), | -   [cuda                         |
|     [\[1\]](api/l                 | q::product_op::to_diagonal_matrix |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq14complex_matrix9kroneckerERK14 |     function)](api/               |
| complex_matrixRK14complex_matrix) | languages/cpp_api.html#_CPPv4NK5c |
| -   [cudaq::c                     | udaq10product_op18to_diagonal_mat |
| omplex_matrix::minimal_eigenvalue | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/languages/     | apINSt6stringENSt7complexIdEEEEb) |
| cpp_api.html#_CPPv4NK5cudaq14comp | -   [cudaq::product_op::to_matrix |
| lex_matrix18minimal_eigenvalueEv) |     (C++                          |
| -   [                             |     funct                         |
| cudaq::complex_matrix::operator() | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NK5cudaq10product_op9to_mat |
|     function)](api/languages/cpp  | rixENSt13unordered_mapINSt6size_t |
| _api.html#_CPPv4N5cudaq14complex_ | ENSt7int64_tEEERKNSt13unordered_m |
| matrixclENSt6size_tENSt6size_tE), | apINSt6stringENSt7complexIdEEEEb) |
|     [\[1\]](api/languages/cpp     | -   [cu                           |
| _api.html#_CPPv4NK5cudaq14complex | daq::product_op::to_sparse_matrix |
| _matrixclENSt6size_tENSt6size_tE) |     (C++                          |
| -   [                             |     function)](ap                 |
| cudaq::complex_matrix::operator\* | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq10product_op16to_sparse_mat |
|     function)](api/langua         | rixENSt13unordered_mapINSt6size_t |
| ges/cpp_api.html#_CPPv4N5cudaq14c | ENSt7int64_tEEERKNSt13unordered_m |
| omplex_matrixmlEN14complex_matrix | apINSt6stringENSt7complexIdEEEEb) |
| 10value_typeERK14complex_matrix), | -   [cudaq::product_op::to_string |
|     [\[1\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](                   |
| v4N5cudaq14complex_matrixmlERK14c | api/languages/cpp_api.html#_CPPv4 |
| omplex_matrixRK14complex_matrix), | NK5cudaq10product_op9to_stringEv) |
|                                   | -                                 |
|  [\[2\]](api/languages/cpp_api.ht |  [cudaq::product_op::\~product_op |
| ml#_CPPv4N5cudaq14complex_matrixm |     (C++                          |
| lERK14complex_matrixRKNSt6vectorI |     fu                            |
| N14complex_matrix10value_typeEEE) | nction)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq10product_opD0Ev) |
| [cudaq::complex_matrix::operator+ | -   [cudaq::ptsbe (C++            |
|     (C++                          |     type)](api/languages/c        |
|     function                      | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::p                     |
| Pv4N5cudaq14complex_matrixplERK14 | tsbe::ConditionalSamplingStrategy |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -                                 |     class)](api/languag           |
| [cudaq::complex_matrix::operator- | es/cpp_api.html#_CPPv4N5cudaq5pts |
|     (C++                          | be27ConditionalSamplingStrategyE) |
|     function                      | -   [cudaq::ptsbe::C              |
| )](api/languages/cpp_api.html#_CP | onditionalSamplingStrategy::clone |
| Pv4N5cudaq14complex_matrixmiERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |                                   |
| -   [cu                           |    function)](api/languages/cpp_a |
| daq::complex_matrix::operator\[\] | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
|     (C++                          | ditionalSamplingStrategy5cloneEv) |
|                                   | -   [cuda                         |
|  function)](api/languages/cpp_api | q::ptsbe::ConditionalSamplingStra |
| .html#_CPPv4N5cudaq14complex_matr | tegy::ConditionalSamplingStrategy |
| ixixERKNSt6vectorINSt6size_tEEE), |     (C++                          |
|     [\[1\]](api/languages/cpp_api |     function)](api/lang           |
| .html#_CPPv4NK5cudaq14complex_mat | uages/cpp_api.html#_CPPv4N5cudaq5 |
| rixixERKNSt6vectorINSt6size_tEEE) | ptsbe27ConditionalSamplingStrateg |
| -   [cudaq::complex_matrix::power | y27ConditionalSamplingStrategyE19 |
|     (C++                          | TrajectoryPredicateNSt8uint64_tE) |
|     function)]                    | -                                 |
| (api/languages/cpp_api.html#_CPPv |   [cudaq::ptsbe::ConditionalSampl |
| 4N5cudaq14complex_matrix5powerEi) | ingStrategy::generateTrajectories |
| -                                 |     (C++                          |
|  [cudaq::complex_matrix::set_zero |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4NK5cudaq5pts |
|     function)](ap                 | be27ConditionalSamplingStrategy20 |
| i/languages/cpp_api.html#_CPPv4N5 | generateTrajectoriesENSt4spanIKN6 |
| cudaq14complex_matrix8set_zeroEv) | detail10NoisePointEEENSt6size_tE) |
| -                                 | -   [cudaq::ptsbe::               |
| [cudaq::complex_matrix::to_string | ConditionalSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     function)](api/               |     function)](api/languages/cpp_ |
| languages/cpp_api.html#_CPPv4NK5c | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| udaq14complex_matrix9to_stringEv) | nditionalSamplingStrategy4nameEv) |
| -   [                             | -   [cudaq:                       |
| cudaq::complex_matrix::value_type | :ptsbe::ConditionalSamplingStrate |
|     (C++                          | gy::\~ConditionalSamplingStrategy |
|     type)](api/                   |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/languages/     |
| daq14complex_matrix10value_typeE) | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| -   [cudaq::contrib (C++          | 7ConditionalSamplingStrategyD0Ev) |
|     type)](api/languages/cpp      | -                                 |
| _api.html#_CPPv4N5cudaq7contribE) | [cudaq::ptsbe::detail::NoisePoint |
| -   [cudaq::contrib::draw (C++    |     (C++                          |
|     function)                     |     struct)](a                    |
| ](api/languages/cpp_api.html#_CPP | pi/languages/cpp_api.html#_CPPv4N |
| v4I0DpEN5cudaq7contrib4drawENSt6s | 5cudaq5ptsbe6detail10NoisePointE) |
| tringERR13QuantumKernelDpRR4Args) | -   [cudaq::p                     |
| -                                 | tsbe::detail::NoisePoint::channel |
| [cudaq::contrib::get_unitary_cmat |     (C++                          |
|     (C++                          |     member)](api/langu            |
|     function)](api/languages/cp   | ages/cpp_api.html#_CPPv4N5cudaq5p |
| p_api.html#_CPPv4I0DpEN5cudaq7con | tsbe6detail10NoisePoint7channelE) |
| trib16get_unitary_cmatE14complex_ | -   [cudaq::ptsbe::det            |
| matrixRR13QuantumKernelDpRR4Args) | ail::NoisePoint::circuit_location |
| -   [cudaq::CusvState (C++        |     (C++                          |
|                                   |     member)](api/languages/cpp_a  |
|    class)](api/languages/cpp_api. | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| html#_CPPv4I0EN5cudaq9CusvStateE) | l10NoisePoint16circuit_locationE) |
| -   [cudaq::dem_from_kernel (C++  | -   [cudaq::p                     |
|     function)](api                | tsbe::detail::NoisePoint::op_name |
| /languages/cpp_api.html#_CPPv4I0D |     (C++                          |
| pEN5cudaq15dem_from_kernelENSt6st |     member)](api/langu            |
| ringERR13QuantumKernelDpRR4Args), | ages/cpp_api.html#_CPPv4N5cudaq5p |
|                                   | tsbe6detail10NoisePoint7op_nameE) |
| [\[1\]](api/languages/cpp_api.htm | -   [cudaq::                      |
| l#_CPPv4I0DpEN5cudaq15dem_from_ke | ptsbe::detail::NoisePoint::qubits |
| rnelENSt6stringERR13QuantumKernel |     (C++                          |
| PKN5cudaq11noise_modelEDpRR4Args) |     member)](api/lang             |
| -   [cudaq::depolarization1 (C++  | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     c                             | ptsbe6detail10NoisePoint6qubitsE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::                      |
| #_CPPv4N5cudaq15depolarization1E) | ptsbe::ExhaustiveSamplingStrategy |
| -   [cudaq::depolarization2 (C++  |     (C++                          |
|     c                             |     class)](api/langua            |
| lass)](api/languages/cpp_api.html | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| #_CPPv4N5cudaq15depolarization2E) | sbe26ExhaustiveSamplingStrategyE) |
| -   [cudaq:                       | -   [cudaq::ptsbe::               |
| :depolarization2::depolarization2 | ExhaustiveSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/languages/cpp_ |
| p_api.html#_CPPv4N5cudaq15depolar | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| ization215depolarization2EK4real) | haustiveSamplingStrategy5cloneEv) |
| -   [cudaq                        | -   [cu                           |
| ::depolarization2::num_parameters | daq::ptsbe::ExhaustiveSamplingStr |
|     (C++                          | ategy::ExhaustiveSamplingStrategy |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     function)](api/la             |
| depolarization214num_parametersE) | nguages/cpp_api.html#_CPPv4N5cuda |
| -   [cu                           | q5ptsbe26ExhaustiveSamplingStrate |
| daq::depolarization2::num_targets | gy26ExhaustiveSamplingStrategyEv) |
|     (C++                          | -                                 |
|     member)](api/la               |    [cudaq::ptsbe::ExhaustiveSampl |
| nguages/cpp_api.html#_CPPv4N5cuda | ingStrategy::generateTrajectories |
| q15depolarization211num_targetsE) |     (C++                          |
| -                                 |     function)](api/languag        |
|    [cudaq::depolarization_channel | es/cpp_api.html#_CPPv4NK5cudaq5pt |
|     (C++                          | sbe26ExhaustiveSamplingStrategy20 |
|     class)](                      | generateTrajectoriesENSt4spanIKN6 |
| api/languages/cpp_api.html#_CPPv4 | detail10NoisePointEEENSt6size_tE) |
| N5cudaq22depolarization_channelE) | -   [cudaq::ptsbe:                |
| -   [cudaq::depol                 | :ExhaustiveSamplingStrategy::name |
| arization_channel::num_parameters |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     member)](api/languages/cp     | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| p_api.html#_CPPv4N5cudaq22depolar | xhaustiveSamplingStrategy4nameEv) |
| ization_channel14num_parametersE) | -   [cuda                         |
| -   [cudaq::de                    | q::ptsbe::ExhaustiveSamplingStrat |
| polarization_channel::num_targets | egy::\~ExhaustiveSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)](api/languages      |
| /cpp_api.html#_CPPv4N5cudaq22depo | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| larization_channel11num_targetsE) | 26ExhaustiveSamplingStrategyD0Ev) |
| -   [cudaq::detail (C++           | -   [cuda                         |
|     type)](api/languages/cp       | q::ptsbe::OrderedSamplingStrategy |
| p_api.html#_CPPv4N5cudaq6detailE) |     (C++                          |
| -   [cudaq::detail::future (C++   |     class)](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|   class)](api/languages/cpp_api.h | 5ptsbe23OrderedSamplingStrategyE) |
| tml#_CPPv4N5cudaq6detail6futureE) | -   [cudaq::ptsb                  |
| -                                 | e::OrderedSamplingStrategy::clone |
|    [cudaq::detail::future::future |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     functi                        | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| on)](api/languages/cpp_api.html#_ | 3OrderedSamplingStrategy5cloneEv) |
| CPPv4N5cudaq6detail6future6future | -   [cudaq::ptsbe::OrderedSampl   |
| ERNSt6vectorI3JobEERNSt6stringERN | ingStrategy::generateTrajectories |
| St3mapINSt6stringENSt6stringEEE), |     (C++                          |
|     [\[1\]](api/lan               |     function)](api/lang           |
| guages/cpp_api.html#_CPPv4N5cudaq | uages/cpp_api.html#_CPPv4NK5cudaq |
| 6detail6future6futureERR6future), | 5ptsbe23OrderedSamplingStrategy20 |
|     [\[2\]                        | generateTrajectoriesENSt4spanIKN6 |
| ](api/languages/cpp_api.html#_CPP | detail10NoisePointEEENSt6size_tE) |
| v4N5cudaq6detail6future6futureEv) | -   [cudaq::pts                   |
| -   [c                            | be::OrderedSamplingStrategy::name |
| udaq::detail::kernel_builder_base |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     class)](api/                  | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| languages/cpp_api.html#_CPPv4N5cu | 23OrderedSamplingStrategy4nameEv) |
| daq6detail19kernel_builder_baseE) | -                                 |
| -   [cudaq::detail::              |    [cudaq::ptsbe::OrderedSampling |
| kernel_builder_base::operator\<\< | Strategy::OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     function)](                   |
| ages/cpp_api.html#_CPPv4N5cudaq6d | api/languages/cpp_api.html#_CPPv4 |
| etail19kernel_builder_baselsERNSt | N5cudaq5ptsbe23OrderedSamplingStr |
| 7ostreamERK19kernel_builder_base) | ategy23OrderedSamplingStrategyEv) |
| -                                 | -                                 |
| [cudaq::detail::KernelBuilderType |  [cudaq::ptsbe::OrderedSamplingSt |
|     (C++                          | rategy::\~OrderedSamplingStrategy |
|     class)](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/langua         |
| cudaq6detail17KernelBuilderTypeE) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq::                      | sbe23OrderedSamplingStrategyD0Ev) |
| detail::KernelBuilderType::create | -   [cudaq::pts                   |
|     (C++                          | be::ProbabilisticSamplingStrategy |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     class)](api/languages         |
| Pv4N5cudaq6detail17KernelBuilderT | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| ype6createEPN4mlir11MLIRContextE) | 29ProbabilisticSamplingStrategyE) |
| -   [cudaq::detail::Ker           | -   [cudaq::ptsbe::Pro            |
| nelBuilderType::KernelBuilderType | babilisticSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     function)](api/lan            |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq |  function)](api/languages/cpp_api |
| 6detail17KernelBuilderType17Kerne | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| lBuilderTypeERRNSt8functionIFN4ml | bilisticSamplingStrategy5cloneEv) |
| ir4TypeEPN4mlir11MLIRContextEEEE) | -                                 |
| -   [cudaq::detector (C++         | [cudaq::ptsbe::ProbabilisticSampl |
|     function)](api                | ingStrategy::generateTrajectories |
| /languages/cpp_api.html#_CPPv4IDp |     (C++                          |
| EN5cudaq8detectorEvDpRR8MeasArgs) |     function)](api/languages/     |
| -   [cudaq::detectors (C++        | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     function)](api/languages/c    | 29ProbabilisticSamplingStrategy20 |
| pp_api.html#_CPPv4N5cudaq9detecto | generateTrajectoriesENSt4spanIKN6 |
| rsERKNSt6vectorI14measure_resultE | detail10NoisePointEEENSt6size_tE) |
| ERKNSt6vectorI14measure_resultEE) | -   [cudaq::ptsbe::Pr             |
| -   [cudaq::diag_matrix_callback  | obabilisticSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     class)                        |                                   |
| ](api/languages/cpp_api.html#_CPP |   function)](api/languages/cpp_ap |
| v4N5cudaq20diag_matrix_callbackE) | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| -   [cudaq::dyn (C++              | abilisticSamplingStrategy4nameEv) |
|     member)](api/languages        | -   [cudaq::p                     |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | tsbe::ProbabilisticSamplingStrate |
| -   [cudaq::ExecutionContext (C++ | gy::ProbabilisticSamplingStrategy |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     function)]                    |
| _CPPv4N5cudaq16ExecutionContextE) | (api/languages/cpp_api.html#_CPPv |
| -   [c                            | 4N5cudaq5ptsbe29ProbabilisticSamp |
| udaq::ExecutionContext::asyncExec | lingStrategy29ProbabilisticSampli |
|     (C++                          | ngStrategyENSt8optionalINSt8uint6 |
|     member)](api/                 | 4_tEEENSt8optionalINSt6size_tEEE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::pts                   |
| daq16ExecutionContext9asyncExecE) | be::ProbabilisticSamplingStrategy |
| -   [cud                          | ::\~ProbabilisticSamplingStrategy |
| aq::ExecutionContext::asyncResult |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api/lan              | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| guages/cpp_api.html#_CPPv4N5cudaq | robabilisticSamplingStrategyD0Ev) |
| 16ExecutionContext11asyncResultE) | -                                 |
| -   [cudaq:                       | [cudaq::ptsbe::PTSBEExecutionData |
| :ExecutionContext::batchIteration |     (C++                          |
|     (C++                          |     struct)](ap                   |
|     member)](api/langua           | i/languages/cpp_api.html#_CPPv4N5 |
| ges/cpp_api.html#_CPPv4N5cudaq16E | cudaq5ptsbe18PTSBEExecutionDataE) |
| xecutionContext14batchIterationE) | -   [cudaq::ptsbe::PTSBE          |
| -   [cudaq::E                     | ExecutionData::count_instructions |
| xecutionContext::canHandleObserve |     (C++                          |
|     (C++                          |     function)](api/l              |
|     member)](api/language         | anguages/cpp_api.html#_CPPv4NK5cu |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | daq5ptsbe18PTSBEExecutionData18co |
| cutionContext16canHandleObserveE) | unt_instructionsE20TraceInstructi |
| -   [cudaq::Executio              | onTypeNSt8optionalINSt6stringEEE) |
| nContext::deferredKernelException | -   [cudaq::ptsbe::P              |
|     (C++                          | TSBEExecutionData::get_trajectory |
|     member)](api/languages/cpp_a  |     (C++                          |
| pi.html#_CPPv4N5cudaq16ExecutionC |     function                      |
| ontext23deferredKernelExceptionE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::E                     | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| xecutionContext::ExecutionContext | Data14get_trajectoryENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe:                |
|     func                          | :PTSBEExecutionData::instructions |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq16ExecutionContext1 |     member)](api/languages/cp     |
| 6ExecutionContextERKNSt6stringE), | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     [\[1\]](api/languages/        | TSBEExecutionData12instructionsE) |
| cpp_api.html#_CPPv4N5cudaq16Execu | -   [cudaq::ptsbe:                |
| tionContext16ExecutionContextERKN | :PTSBEExecutionData::trajectories |
| St6stringENSt6size_tENSt6size_tE) |     (C++                          |
| -   [cudaq::E                     |     member)](api/languages/cp     |
| xecutionContext::expectationValue | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     (C++                          | TSBEExecutionData12trajectoriesE) |
|     member)](api/language         | -   [cudaq::ptsbe::PTSBEOptions   |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     (C++                          |
| cutionContext16expectationValueE) |     struc                         |
| -   [cudaq::Execu                 | t)](api/languages/cpp_api.html#_C |
| tionContext::explicitMeasurements | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
|     (C++                          | -   [cudaq::ptsbe::PTSB           |
|     member)](api/languages/cp     | EOptions::include_sequential_data |
| p_api.html#_CPPv4N5cudaq16Executi |     (C++                          |
| onContext20explicitMeasurementsE) |                                   |
| -   [cuda                         |    member)](api/languages/cpp_api |
| q::ExecutionContext::futureResult | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
|     (C++                          | ptions23include_sequential_dataE) |
|     member)](api/lang             | -   [cudaq::ptsb                  |
| uages/cpp_api.html#_CPPv4N5cudaq1 | e::PTSBEOptions::max_trajectories |
| 6ExecutionContext12futureResultE) |     (C++                          |
| -   [cudaq::ExecutionContext      |     member)](api/languages/       |
| ::hasConditionalsOnMeasureResults | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
|     (C++                          | 2PTSBEOptions16max_trajectoriesE) |
|     mem                           | -   [cudaq::ptsbe::PT             |
| ber)](api/languages/cpp_api.html# | SBEOptions::return_execution_data |
| _CPPv4N5cudaq16ExecutionContext31 |     (C++                          |
| hasConditionalsOnMeasureResultsE) |     member)](api/languages/cpp_a  |
| -   [cudaq:                       | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| :ExecutionContext::inKernelLaunch | EOptions21return_execution_dataE) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/langua           | be::PTSBEOptions::shot_allocation |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14inKernelLaunchE) |     member)](api/languages        |
| -   [cudaq::Executi               | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| onContext::invocationResultBuffer | 12PTSBEOptions15shot_allocationE) |
|     (C++                          | -   [cud                          |
|     member)](api/languages/cpp_   | aq::ptsbe::PTSBEOptions::strategy |
| api.html#_CPPv4N5cudaq16Execution |     (C++                          |
| Context22invocationResultBufferE) |     member)](api/l                |
| -   [cu                           | anguages/cpp_api.html#_CPPv4N5cud |
| daq::ExecutionContext::kernelName | aq5ptsbe12PTSBEOptions8strategyE) |
|     (C++                          | -   [cudaq::ptsbe::PTSBETrace     |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     t                             |
| q16ExecutionContext10kernelNameE) | ype)](api/languages/cpp_api.html# |
| -   [cud                          | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| aq::ExecutionContext::kernelTrace | -   [                             |
|     (C++                          | cudaq::ptsbe::PTSSamplingStrategy |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     class)](api                   |
| 16ExecutionContext11kernelTraceE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq:                       | udaq5ptsbe19PTSSamplingStrategyE) |
| :ExecutionContext::msm_dimensions | -   [cudaq::                      |
|     (C++                          | ptsbe::PTSSamplingStrategy::clone |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     function)](api/languag        |
| xecutionContext14msm_dimensionsE) | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| -   [cudaq::                      | sbe19PTSSamplingStrategy5cloneEv) |
| ExecutionContext::msm_prob_err_id | -   [cudaq::ptsbe::PTSSampl       |
|     (C++                          | ingStrategy::generateTrajectories |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api/               |
| ecutionContext15msm_prob_err_idE) | languages/cpp_api.html#_CPPv4NK5c |
| -   [cudaq::Ex                    | udaq5ptsbe19PTSSamplingStrategy20 |
| ecutionContext::msm_probabilities | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     member)](api/languages        | -   [cudaq:                       |
| /cpp_api.html#_CPPv4N5cudaq16Exec | :ptsbe::PTSSamplingStrategy::name |
| utionContext17msm_probabilitiesE) |     (C++                          |
| -                                 |     function)](api/langua         |
|    [cudaq::ExecutionContext::name | ges/cpp_api.html#_CPPv4NK5cudaq5p |
|     (C++                          | tsbe19PTSSamplingStrategy4nameEv) |
|     member)]                      | -   [cudaq::ptsbe::PTSSampli      |
| (api/languages/cpp_api.html#_CPPv | ngStrategy::\~PTSSamplingStrategy |
| 4N5cudaq16ExecutionContext4nameE) |     (C++                          |
| -   [cu                           |     function)](api/la             |
| daq::ExecutionContext::noiseModel | nguages/cpp_api.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe19PTSSamplingStrategyD0Ev) |
|     member)](api/la               | -   [cudaq::ptsbe::sample (C++    |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q16ExecutionContext10noiseModelE) |  function)](api/languages/cpp_api |
| -   [cudaq::Exe                   | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| cutionContext::numberTrajectories | mpleE13sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](api/languages/       |     [\[1\]](api                   |
| cpp_api.html#_CPPv4N5cudaq16Execu | /languages/cpp_api.html#_CPPv4I0D |
| tionContext18numberTrajectoriesE) | pEN5cudaq5ptsbe6sampleE13sample_r |
| -   [c                            | esultRKN5cudaq11noise_modelENSt6s |
| udaq::ExecutionContext::optResult | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_async   |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](a                  |
| daq16ExecutionContext9optResultE) | pi/languages/cpp_api.html#_CPPv4I |
| -                                 | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
|   [cudaq::ExecutionContext::qpuId | 9async_sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](                     |     [\[1\]](api/languages/cp      |
| api/languages/cpp_api.html#_CPPv4 | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| N5cudaq16ExecutionContext5qpuIdE) | be12sample_asyncE19async_sample_r |
| -   [cudaq                        | esultRKN5cudaq11noise_modelENSt6s |
| ::ExecutionContext::registerNames | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_options |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     struct)                       |
| ExecutionContext13registerNamesE) | ](api/languages/cpp_api.html#_CPP |
| -   [cu                           | v4N5cudaq5ptsbe14sample_optionsE) |
| daq::ExecutionContext::reorderIdx | -   [cudaq::ptsbe::sample_result  |
|     (C++                          |     (C++                          |
|     member)](api/la               |     class                         |
| nguages/cpp_api.html#_CPPv4N5cuda | )](api/languages/cpp_api.html#_CP |
| q16ExecutionContext10reorderIdxE) | Pv4N5cudaq5ptsbe13sample_resultE) |
| -                                 | -   [cudaq::pts                   |
|  [cudaq::ExecutionContext::result | be::sample_result::execution_data |
|     (C++                          |     (C++                          |
|     member)](a                    |     function)](api/languages/c    |
| pi/languages/cpp_api.html#_CPPv4N | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| 5cudaq16ExecutionContext6resultE) | 3sample_result14execution_dataEv) |
| -                                 | -   [cudaq::ptsbe::               |
|   [cudaq::ExecutionContext::shots | sample_result::has_execution_data |
|     (C++                          |     (C++                          |
|     member)](                     |                                   |
| api/languages/cpp_api.html#_CPPv4 |    function)](api/languages/cpp_a |
| N5cudaq16ExecutionContext5shotsE) | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| -   [cudaq::                      | ple_result18has_execution_dataEv) |
| ExecutionContext::simulationState | -   [cudaq::pt                    |
|     (C++                          | sbe::sample_result::sample_result |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api/l              |
| ecutionContext15simulationStateE) | anguages/cpp_api.html#_CPPv4N5cud |
| -                                 | aq5ptsbe13sample_result13sample_r |
|    [cudaq::ExecutionContext::spin | esultERRN5cudaq13sample_resultE), |
|     (C++                          |                                   |
|     member)]                      |  [\[1\]](api/languages/cpp_api.ht |
| (api/languages/cpp_api.html#_CPPv | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| 4N5cudaq16ExecutionContext4spinE) | sult13sample_resultERRN5cudaq13sa |
| -   [cudaq::                      | mple_resultE18PTSBEExecutionData) |
| ExecutionContext::totalIterations | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::set_execution_data |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api/               |
| ecutionContext15totalIterationsE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::Executio              | daq5ptsbe13sample_result18set_exe |
| nContext::warnedNamedMeasurements | cution_dataE18PTSBEExecutionData) |
|     (C++                          | -   [cud                          |
|     member)](api/languages/cpp_a  | aq::ptsbe::ShotAllocationStrategy |
| pi.html#_CPPv4N5cudaq16ExecutionC |     (C++                          |
| ontext23warnedNamedMeasurementsE) |     struct)](using                |
| -   [cudaq::ExecutionResult (C++  | /examples/ptsbe.html#_CPPv4N5cuda |
|     st                            | q5ptsbe22ShotAllocationStrategyE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::ptsbe::ShotAllocatio  |
| #_CPPv4N5cudaq15ExecutionResultE) | nStrategy::ShotAllocationStrategy |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::appendResult |     function)                     |
|     (C++                          | ](using/examples/ptsbe.html#_CPPv |
|     functio                       | 4N5cudaq5ptsbe22ShotAllocationStr |
| n)](api/languages/cpp_api.html#_C | ategy22ShotAllocationStrategyE4Ty |
| PPv4N5cudaq15ExecutionResult12app | pedNSt8optionalINSt8uint64_tEEE), |
| endResultENSt6stringENSt6size_tE) |     [\[1\                         |
| -   [cu                           | ]](using/examples/ptsbe.html#_CPP |
| daq::ExecutionResult::deserialize | v4N5cudaq5ptsbe22ShotAllocationSt |
|     (C++                          | rategy22ShotAllocationStrategyEv) |
|     function)                     | -   [cudaq::pt                    |
| ](api/languages/cpp_api.html#_CPP | sbe::ShotAllocationStrategy::Type |
| v4N5cudaq15ExecutionResult11deser |     (C++                          |
| ializeERNSt6vectorINSt6size_tEEE) |     enum)](using/exam             |
| -   [cudaq:                       | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| :ExecutionResult::ExecutionResult | be22ShotAllocationStrategy4TypeE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAllocatio  |
|     functio                       | nStrategy::Type::HIGH_WEIGHT_BIAS |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq15ExecutionResult15Exe |     enumerat                      |
| cutionResultE16CountsDictionary), | or)](using/examples/ptsbe.html#_C |
|     [\[1\]](api/lan               | PPv4N5cudaq5ptsbe22ShotAllocation |
| guages/cpp_api.html#_CPPv4N5cudaq | Strategy4Type16HIGH_WEIGHT_BIASE) |
| 15ExecutionResult15ExecutionResul | -   [cudaq::ptsbe::ShotAllocati   |
| tE16CountsDictionaryNSt6stringE), | onStrategy::Type::LOW_WEIGHT_BIAS |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     enumera                       |
| Pv4N5cudaq15ExecutionResult15Exec | tor)](using/examples/ptsbe.html#_ |
| utionResultE16CountsDictionaryd), | CPPv4N5cudaq5ptsbe22ShotAllocatio |
|                                   | nStrategy4Type15LOW_WEIGHT_BIASE) |
|    [\[3\]](api/languages/cpp_api. | -   [cudaq::ptsbe::ShotAlloc      |
| html#_CPPv4N5cudaq15ExecutionResu | ationStrategy::Type::PROPORTIONAL |
| lt15ExecutionResultENSt6stringE), |     (C++                          |
|     [\[4\                         |     enum                          |
| ]](api/languages/cpp_api.html#_CP | erator)](using/examples/ptsbe.htm |
| Pv4N5cudaq15ExecutionResult15Exec | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| utionResultERK15ExecutionResult), | tionStrategy4Type12PROPORTIONALE) |
|     [\[5\]](api/language          | -   [cudaq::ptsbe::Shot           |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | AllocationStrategy::Type::UNIFORM |
| cutionResult15ExecutionResultEd), |     (C++                          |
|     [\[6\]](api/languag           |                                   |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |   enumerator)](using/examples/pts |
| ecutionResult15ExecutionResultEv) | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| -   [                             | AllocationStrategy4Type7UNIFORME) |
| cudaq::ExecutionResult::operator= | -                                 |
|     (C++                          |   [cudaq::ptsbe::TraceInstruction |
|     function)](api/languages/     |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq15Execu |     struct)](                     |
| tionResultaSERK15ExecutionResult) | api/languages/cpp_api.html#_CPPv4 |
| -   [c                            | N5cudaq5ptsbe16TraceInstructionE) |
| udaq::ExecutionResult::operator== | -   [cudaq:                       |
|     (C++                          | :ptsbe::TraceInstruction::channel |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4NK5cudaq15Execu |     member)](api/lang             |
| tionResulteqERK15ExecutionResult) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -   [cud                          | ptsbe16TraceInstruction7channelE) |
| aq::ExecutionResult::registerName | -   [cudaq::                      |
|     (C++                          | ptsbe::TraceInstruction::controls |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     member)](api/langu            |
| 15ExecutionResult12registerNameE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [cudaq                        | tsbe16TraceInstruction8controlsE) |
| ::ExecutionResult::sequentialData | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::name |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     member)](api/l                |
| ExecutionResult14sequentialDataE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [                             | aq5ptsbe16TraceInstruction4nameE) |
| cudaq::ExecutionResult::serialize | -   [cudaq                        |
|     (C++                          | ::ptsbe::TraceInstruction::params |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4NK5cu |     member)](api/lan              |
| daq15ExecutionResult9serializeEv) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [cudaq::fermion_handler (C++  | 5ptsbe16TraceInstruction6paramsE) |
|     c                             | -   [cudaq:                       |
| lass)](api/languages/cpp_api.html | :ptsbe::TraceInstruction::targets |
| #_CPPv4N5cudaq15fermion_handlerE) |     (C++                          |
| -   [cudaq::fermion_op (C++       |     member)](api/lang             |
|     type)](api/languages/cpp_api  | uages/cpp_api.html#_CPPv4N5cudaq5 |
| .html#_CPPv4N5cudaq10fermion_opE) | ptsbe16TraceInstruction7targetsE) |
| -   [cudaq::fermion_op_term (C++  | -   [cudaq::ptsbe::T              |
|                                   | raceInstruction::TraceInstruction |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15fermion_op_termE) |                                   |
| -   [cudaq::FermioniqQPU (C++     |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4N5cudaq5ptsbe16Trace |
|   class)](api/languages/cpp_api.h | Instruction16TraceInstructionE20T |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | raceInstructionTypeNSt6stringENSt |
| -   [cudaq::get_state (C++        | 6vectorINSt6size_tEEENSt6vectorIN |
|                                   | St6size_tEEENSt6vectorIdEENSt8opt |
|    function)](api/languages/cpp_a | ionalIN5cudaq13kraus_channelEEE), |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     [\[1\]](api/languages/cpp_a   |
| ateEDaRR13QuantumKernelDpRR4Args) | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| -   [cudaq::gradient (C++         | eInstruction16TraceInstructionEv) |
|     class)](api/languages/cpp_    | -   [cud                          |
| api.html#_CPPv4N5cudaq8gradientE) | aq::ptsbe::TraceInstruction::type |
| -   [cudaq::gradient::clone (C++  |     (C++                          |
|     fun                           |     member)](api/l                |
| ction)](api/languages/cpp_api.htm | anguages/cpp_api.html#_CPPv4N5cud |
| l#_CPPv4N5cudaq8gradient5cloneEv) | aq5ptsbe16TraceInstruction4typeE) |
| -   [cudaq::gradient::compute     | -   [c                            |
|     (C++                          | udaq::ptsbe::TraceInstructionType |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     enum)](api/                   |
| ient7computeERKNSt6vectorIdEERKNS | languages/cpp_api.html#_CPPv4N5cu |
| t8functionIFdNSt6vectorIdEEEEEd), | daq5ptsbe20TraceInstructionTypeE) |
|     [\[1\]](ap                    | -   [cudaq::                      |
| i/languages/cpp_api.html#_CPPv4N5 | ptsbe::TraceInstructionType::Gate |
| cudaq8gradient7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |     enumerator)](api/langu        |
| -   [cudaq::gradient::gradient    | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     (C++                          | tsbe20TraceInstructionType4GateE) |
|     function)](api/lang           | -   [cudaq::ptsbe::               |
| uages/cpp_api.html#_CPPv4I00EN5cu | TraceInstructionType::Measurement |
| daq8gradient8gradientER7KernelT), |     (C++                          |
|                                   |                                   |
|    [\[1\]](api/languages/cpp_api. |    enumerator)](api/languages/cpp |
| html#_CPPv4I00EN5cudaq8gradient8g | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| radientER7KernelTRR10ArgsMapper), | aceInstructionType11MeasurementE) |
|     [\[2\                         | -   [cudaq::p                     |
| ]](api/languages/cpp_api.html#_CP | tsbe::TraceInstructionType::Noise |
| Pv4I00EN5cudaq8gradient8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     enumerator)](api/langua       |
|     [\[3                          | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| \]](api/languages/cpp_api.html#_C | sbe20TraceInstructionType5NoiseE) |
| PPv4N5cudaq8gradient8gradientERRN | -   [                             |
| St8functionIFvNSt6vectorIdEEEEE), | cudaq::ptsbe::TrajectoryPredicate |
|     [\[                           |     (C++                          |
| 4\]](api/languages/cpp_api.html#_ |     type)](api                    |
| CPPv4N5cudaq8gradient8gradientEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gradient::setArgs     | udaq5ptsbe19TrajectoryPredicateE) |
|     (C++                          | -   [cudaq::QPU (C++              |
|     fu                            |     class)](api/languages         |
| nction)](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | -   [cudaq::QPU::beginExecution   |
| tArgsEvR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::gradient::setKernel   |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function)](api/languages/c    | Pv4N5cudaq3QPU14beginExecutionEv) |
| pp_api.html#_CPPv4I0EN5cudaq8grad | -   [cuda                         |
| ient9setKernelEvR13QuantumKernel) | q::QPU::configureExecutionContext |
| -   [cud                          |     (C++                          |
| aq::gradients::central_difference |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     class)](api/la                | _CPPv4NK5cudaq3QPU25configureExec |
| nguages/cpp_api.html#_CPPv4N5cuda | utionContextER16ExecutionContext) |
| q9gradients18central_differenceE) | -   [cudaq::QPU::endExecution     |
| -   [cudaq::gra                   |     (C++                          |
| dients::central_difference::clone |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     function)](api/languages      | CPPv4N5cudaq3QPU12endExecutionEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QPU::enqueue (C++     |
| ents18central_difference5cloneEv) |     function)](ap                 |
| -   [cudaq::gradi                 | i/languages/cpp_api.html#_CPPv4N5 |
| ents::central_difference::compute | cudaq3QPU7enqueueER11QuantumTask) |
|     (C++                          | -   [cud                          |
|     function)](                   | aq::QPU::finalizeExecutionContext |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq9gradients18central_differ |     func                          |
| ence7computeERKNSt6vectorIdEERKNS | tion)](api/languages/cpp_api.html |
| t8functionIFdNSt6vectorIdEEEEEd), | #_CPPv4NK5cudaq3QPU24finalizeExec |
|                                   | utionContextER16ExecutionContext) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::QPU::getCompileTarget |
| tml#_CPPv4N5cudaq9gradients18cent |     (C++                          |
| ral_difference7computeERKNSt6vect |     function)](api/languages/c    |
| orIdEERNSt6vectorIdEERK7spin_opd) | pp_api.html#_CPPv4N5cudaq3QPU16ge |
| -   [cudaq::gradie                | tCompileTargetERK13sample_policy) |
| nts::central_difference::gradient | -   [cudaq::QPU::getConnectivity  |
|     (C++                          |     (C++                          |
|     functio                       |     function)                     |
| n)](api/languages/cpp_api.html#_C | ](api/languages/cpp_api.html#_CPP |
| PPv4I00EN5cudaq9gradients18centra | v4N5cudaq3QPU15getConnectivityEv) |
| l_difference8gradientER7KernelT), | -                                 |
|     [\[1\]](api/langua            | [cudaq::QPU::getExecutionThreadId |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     (C++                          |
| q9gradients18central_difference8g |     function)](api/               |
| radientER7KernelTRR10ArgsMapper), | languages/cpp_api.html#_CPPv4NK5c |
|     [\[2\]](api/languages/cpp_    | udaq3QPU20getExecutionThreadIdEv) |
| api.html#_CPPv4I00EN5cudaq9gradie | -   [cudaq::QPU::getNumQubits     |
| nts18central_difference8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     functi                        |
|     [\[3\]](api/languages/cpp     | on)](api/languages/cpp_api.html#_ |
| _api.html#_CPPv4N5cudaq9gradients | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| 18central_difference8gradientERRN | -   [                             |
| St8functionIFvNSt6vectorIdEEEEE), | cudaq::QPU::getRemoteCapabilities |
|     [\[4\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq9gradient |     function)](api/l              |
| s18central_difference8gradientEv) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cud                          | daq3QPU21getRemoteCapabilitiesEv) |
| aq::gradients::forward_difference | -   [cudaq::QPU::isEmulated (C++  |
|     (C++                          |     func                          |
|     class)](api/la                | tion)](api/languages/cpp_api.html |
| nguages/cpp_api.html#_CPPv4N5cuda | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| q9gradients18forward_differenceE) | -   [cudaq::QPU::isSimulator (C++ |
| -   [cudaq::gra                   |     funct                         |
| dients::forward_difference::clone | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     function)](api/languages      | -   [cudaq::QPU::onRandomSeedSet  |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18forward_difference5cloneEv) |     function)](api/lang           |
| -   [cudaq::gradi                 | uages/cpp_api.html#_CPPv4N5cudaq3 |
| ents::forward_difference::compute | QPU15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::QPU (C++         |
|     function)](                   |     functio                       |
| api/languages/cpp_api.html#_CPPv4 | n)](api/languages/cpp_api.html#_C |
| N5cudaq9gradients18forward_differ | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| ence7computeERKNSt6vectorIdEERKNS |                                   |
| t8functionIFdNSt6vectorIdEEEEEd), |  [\[1\]](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|   [\[1\]](api/languages/cpp_api.h |     [\[2\]](api/languages/cpp_    |
| tml#_CPPv4N5cudaq9gradients18forw | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| ard_difference7computeERKNSt6vect | -   [cudaq::QPU::setId (C++       |
| orIdEERNSt6vectorIdEERK7spin_opd) |     function                      |
| -   [cudaq::gradie                | )](api/languages/cpp_api.html#_CP |
| nts::forward_difference::gradient | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::setShots (C++    |
|     functio                       |     f                             |
| n)](api/languages/cpp_api.html#_C | unction)](api/languages/cpp_api.h |
| PPv4I00EN5cudaq9gradients18forwar | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| d_difference8gradientER7KernelT), | -   [cudaq::                      |
|     [\[1\]](api/langua            | QPU::supportsExplicitMeasurements |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     (C++                          |
| q9gradients18forward_difference8g |     function)](api/languag        |
| radientER7KernelTRR10ArgsMapper), | es/cpp_api.html#_CPPv4N5cudaq3QPU |
|     [\[2\]](api/languages/cpp_    | 28supportsExplicitMeasurementsEv) |
| api.html#_CPPv4I00EN5cudaq9gradie | -   [cudaq::QPU::\~QPU (C++       |
| nts18forward_difference8gradientE |     function)](api/languages/cp   |
| RR13QuantumKernelRR10ArgsMapper), | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::QPUState (C++         |
| _api.html#_CPPv4N5cudaq9gradients |     class)](api/languages/cpp_    |
| 18forward_difference8gradientERRN | api.html#_CPPv4N5cudaq8QPUStateE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::qreg (C++             |
|     [\[4\]](api/languages/cp      |     class)](api/lan               |
| p_api.html#_CPPv4N5cudaq9gradient | guages/cpp_api.html#_CPPv4I_NSt6s |
| s18forward_difference8gradientEv) | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| -   [                             | -   [cudaq::qreg::back (C++       |
| cudaq::gradients::parameter_shift |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     class)](api                   | v4N5cudaq4qreg4backENSt6size_tE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[1\]](api/languages/cpp_ap  |
| udaq9gradients15parameter_shiftE) | i.html#_CPPv4N5cudaq4qreg4backEv) |
| -   [cudaq::                      | -   [cudaq::qreg::begin (C++      |
| gradients::parameter_shift::clone |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     function)](api/langua         | .html#_CPPv4N5cudaq4qreg5beginEv) |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | -   [cudaq::qreg::clear (C++      |
| adients15parameter_shift5cloneEv) |                                   |
| -   [cudaq::gr                    |  function)](api/languages/cpp_api |
| adients::parameter_shift::compute | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     (C++                          | -   [cudaq::qreg::front (C++      |
|     function                      |     function)]                    |
| )](api/languages/cpp_api.html#_CP | (api/languages/cpp_api.html#_CPPv |
| Pv4N5cudaq9gradients15parameter_s | 4N5cudaq4qreg5frontENSt6size_tE), |
| hift7computeERKNSt6vectorIdEERKNS |     [\[1\]](api/languages/cpp_api |
| t8functionIFdNSt6vectorIdEEEEEd), | .html#_CPPv4N5cudaq4qreg5frontEv) |
|     [\[1\]](api/languages/cpp_ap  | -   [cudaq::qreg::operator\[\]    |
| i.html#_CPPv4N5cudaq9gradients15p |     (C++                          |
| arameter_shift7computeERKNSt6vect |     functi                        |
| orIdEERNSt6vectorIdEERK7spin_opd) | on)](api/languages/cpp_api.html#_ |
| -   [cudaq::gra                   | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| dients::parameter_shift::gradient | -   [cudaq::qreg::qreg (C++       |
|     (C++                          |     function)                     |
|     func                          | ](api/languages/cpp_api.html#_CPP |
| tion)](api/languages/cpp_api.html | v4N5cudaq4qreg4qregENSt6size_tE), |
| #_CPPv4I00EN5cudaq9gradients15par |     [\[1\]](api/languages/cpp_ap  |
| ameter_shift8gradientER7KernelT), | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     [\[1\]](api/lan               | -   [cudaq::qreg::size (C++       |
| guages/cpp_api.html#_CPPv4I00EN5c |                                   |
| udaq9gradients15parameter_shift8g |  function)](api/languages/cpp_api |
| radientER7KernelTRR10ArgsMapper), | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     [\[2\]](api/languages/c       | -   [cudaq::qreg::slice (C++      |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     function)](api/langu          |
| dients15parameter_shift8gradientE | ages/cpp_api.html#_CPPv4N5cudaq4q |
| RR13QuantumKernelRR10ArgsMapper), | reg5sliceENSt6size_tENSt6size_tE) |
|     [\[3\]](api/languages/        | -   [cudaq::qreg::value_type (C++ |
| cpp_api.html#_CPPv4N5cudaq9gradie |                                   |
| nts15parameter_shift8gradientERRN | type)](api/languages/cpp_api.html |
| St8functionIFvNSt6vectorIdEEEEE), | #_CPPv4N5cudaq4qreg10value_typeE) |
|     [\[4\]](api/languages         | -   [cudaq::qspan (C++            |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     class)](api/lang              |
| ents15parameter_shift8gradientEv) | uages/cpp_api.html#_CPPv4I_NSt6si |
| -   [cudaq::kernel_builder (C++   | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     clas                          | -   [cudaq::QuakeValue (C++       |
| s)](api/languages/cpp_api.html#_C |     class)](api/languages/cpp_api |
| PPv4IDpEN5cudaq14kernel_builderE) | .html#_CPPv4N5cudaq10QuakeValueE) |
| -   [c                            | -   [cudaq::Q                     |
| udaq::kernel_builder::constantVal | uakeValue::canValidateNumElements |
|     (C++                          |     (C++                          |
|     function)](api/la             |     function)](api/languages      |
| nguages/cpp_api.html#_CPPv4N5cuda | /cpp_api.html#_CPPv4N5cudaq10Quak |
| q14kernel_builder11constantValEd) | eValue22canValidateNumElementsEv) |
| -                                 | -                                 |
|  [cudaq::kernel_builder::detector |  [cudaq::QuakeValue::constantSize |
|     (C++                          |     (C++                          |
|                                   |     function)](api                |
|    function)](api/languages/cpp_a | /languages/cpp_api.html#_CPPv4N5c |
| pi.html#_CPPv4IDpEN5cudaq14kernel | udaq10QuakeValue12constantSizeEv) |
| _builder8detectorEvDpRR8MeasArgs) | -   [cudaq::QuakeValue::dump (C++ |
| -                                 |     function)](api/lan            |
| [cudaq::kernel_builder::detectors | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 10QuakeValue4dumpERNSt7ostreamE), |
|     func                          |     [\                            |
| tion)](api/languages/cpp_api.html | [1\]](api/languages/cpp_api.html# |
| #_CPPv4N5cudaq14kernel_builder9de | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| tectorsE10QuakeValue10QuakeValue) | -   [cudaq                        |
| -   [cu                           | ::QuakeValue::getRequiredElements |
| daq::kernel_builder::getArguments |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/lan            | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| guages/cpp_api.html#_CPPv4N5cudaq | uakeValue19getRequiredElementsEv) |
| 14kernel_builder12getArgumentsEv) | -   [cudaq::QuakeValue::getValue  |
| -   [cu                           |     (C++                          |
| daq::kernel_builder::getNumParams |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/lan            | 4NK5cudaq10QuakeValue8getValueEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::QuakeValue::inverse   |
| 14kernel_builder12getNumParamsEv) |     (C++                          |
| -   [c                            |     function)                     |
| udaq::kernel_builder::isArgStdVec | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq10QuakeValue7inverseEv) |
|     function)](api/languages/cp   | -   [cudaq::QuakeValue::isStdVec  |
| p_api.html#_CPPv4N5cudaq14kernel_ |     (C++                          |
| builder11isArgStdVecENSt6size_tE) |     function)                     |
| -   [cuda                         | ](api/languages/cpp_api.html#_CPP |
| q::kernel_builder::kernel_builder | v4N5cudaq10QuakeValue8isStdVecEv) |
|     (C++                          | -                                 |
|     function)](api/languages/cpp  |    [cudaq::QuakeValue::operator\* |
| _api.html#_CPPv4N5cudaq14kernel_b |     (C++                          |
| uilder14kernel_builderERNSt6vecto |     function)](api                |
| rIN6detail17KernelBuilderTypeEEE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::k                     | udaq10QuakeValuemlE10QuakeValue), |
| ernel_builder::logical_observable |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     function)                     | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::QuakeValue::operator+ |
| v4IDpEN5cudaq14kernel_builder18lo |     (C++                          |
| gical_observableEvDpRR8MeasArgs), |     function)](api                |
|     [\[1\]](ap                    | /languages/cpp_api.html#_CPPv4N5c |
| i/languages/cpp_api.html#_CPPv4N5 | udaq10QuakeValueplE10QuakeValue), |
| cudaq14kernel_builder18logical_ob |     [                             |
| servableE10QuakeValueNSt6size_tE) | \[1\]](api/languages/cpp_api.html |
| -   [cudaq::kernel_builder::name  | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     (C++                          |                                   |
|     function)                     | [\[2\]](api/languages/cpp_api.htm |
| ](api/languages/cpp_api.html#_CPP | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| v4N5cudaq14kernel_builder4nameEv) | -   [cudaq::QuakeValue::operator- |
| -                                 |     (C++                          |
|    [cudaq::kernel_builder::qalloc |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/language       | udaq10QuakeValuemiE10QuakeValue), |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     [                             |
| nel_builder6qallocE10QuakeValue), | \[1\]](api/languages/cpp_api.html |
|     [\[1\]](api/language          | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     [                             |
| nel_builder6qallocEKNSt6size_tE), | \[2\]](api/languages/cpp_api.html |
|     [\[2                          | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| \]](api/languages/cpp_api.html#_C |                                   |
| PPv4N5cudaq14kernel_builder6qallo | [\[3\]](api/languages/cpp_api.htm |
| cERNSt6vectorINSt7complexIdEEEE), | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     [\[3\]](                      | -   [cudaq::QuakeValue::operator/ |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14kernel_builder6qallocEv) |     function)](api                |
| -   [cudaq::kernel_builder::swap  | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuedvE10QuakeValue), |
|     function)](api/language       |                                   |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | [\[1\]](api/languages/cpp_api.htm |
| 4kernel_builder4swapEvRK10QuakeVa | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| lueRK10QuakeValueRK10QuakeValue), | -                                 |
|                                   |  [cudaq::QuakeValue::operator\[\] |
| [\[1\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4I00EN5cudaq14kernel_build |     function)](api                |
| er4swapEvRKNSt6vectorI10QuakeValu | /languages/cpp_api.html#_CPPv4N5c |
| eEERK10QuakeValueRK10QuakeValue), | udaq10QuakeValueixEKNSt6size_tE), |
|                                   |     [\[1\]](api/                  |
| [\[2\]](api/languages/cpp_api.htm | languages/cpp_api.html#_CPPv4N5cu |
| l#_CPPv4N5cudaq14kernel_builder4s | daq10QuakeValueixERK10QuakeValue) |
| wapERK10QuakeValueRK10QuakeValue) | -                                 |
| -   [cudaq::KernelExecutionTask   |    [cudaq::QuakeValue::QuakeValue |
|     (C++                          |     (C++                          |
|     type                          |     function)](api/languag        |
| )](api/languages/cpp_api.html#_CP | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| Pv4N5cudaq19KernelExecutionTaskE) | akeValue10QuakeValueERN4mlir20Imp |
| -   [cudaq::KernelThunkResultType | licitLocOpBuilderEN4mlir5ValueE), |
|     (C++                          |     [\[1\]                        |
|     struct)]                      | ](api/languages/cpp_api.html#_CPP |
| (api/languages/cpp_api.html#_CPPv | v4N5cudaq10QuakeValue10QuakeValue |
| 4N5cudaq21KernelThunkResultTypeE) | ERN4mlir20ImplicitLocOpBuilderEd) |
| -   [cudaq::KernelThunkType (C++  | -   [cudaq::QuakeValue::size (C++ |
|                                   |     funct                         |
| type)](api/languages/cpp_api.html | ion)](api/languages/cpp_api.html# |
| #_CPPv4N5cudaq15KernelThunkTypeE) | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| -   [cudaq::kraus_channel (C++    | -   [cudaq::QuakeValue::slice     |
|                                   |     (C++                          |
|  class)](api/languages/cpp_api.ht |     function)](api/languages/cpp_ |
| ml#_CPPv4N5cudaq13kraus_channelE) | api.html#_CPPv4N5cudaq10QuakeValu |
| -   [cudaq::kraus_channel::empty  | e5sliceEKNSt6size_tEKNSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_platform (C++ |
|     function)]                    |     cl                            |
| (api/languages/cpp_api.html#_CPPv | ass)](api/languages/cpp_api.html# |
| 4NK5cudaq13kraus_channel5emptyEv) | _CPPv4N5cudaq16quantum_platformE) |
| -   [cudaq::kraus_c               | -   [cudaq:                       |
| hannel::generateUnitaryParameters | :quantum_platform::beginExecution |
|     (C++                          |     (C++                          |
|                                   |     function)](api/languag        |
|    function)](api/languages/cpp_a | es/cpp_api.html#_CPPv4N5cudaq16qu |
| pi.html#_CPPv4N5cudaq13kraus_chan | antum_platform14beginExecutionEv) |
| nel25generateUnitaryParametersEv) | -   [cudaq::quantum_pl            |
| -                                 | atform::configureExecutionContext |
|    [cudaq::kraus_channel::get_ops |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     function)](a                  | uages/cpp_api.html#_CPPv4NK5cudaq |
| pi/languages/cpp_api.html#_CPPv4N | 16quantum_platform25configureExec |
| K5cudaq13kraus_channel7get_opsEv) | utionContextER16ExecutionContext) |
| -   [cud                          | -   [cuda                         |
| aq::kraus_channel::identity_flags | q::quantum_platform::connectivity |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/langu          |
| guages/cpp_api.html#_CPPv4N5cudaq | ages/cpp_api.html#_CPPv4N5cudaq16 |
| 13kraus_channel14identity_flagsE) | quantum_platform12connectivityEv) |
| -   [cud                          | -   [cuda                         |
| aq::kraus_channel::is_identity_op | q::quantum_platform::endExecution |
|     (C++                          |     (C++                          |
|                                   |     function)](api/langu          |
|    function)](api/languages/cpp_a | ages/cpp_api.html#_CPPv4N5cudaq16 |
| pi.html#_CPPv4NK5cudaq13kraus_cha | quantum_platform12endExecutionEv) |
| nnel14is_identity_opENSt6size_tE) | -   [cudaq::q                     |
| -   [cudaq::                      | uantum_platform::enqueueAsyncTask |
| kraus_channel::is_unitary_mixture |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     function)](api/languages      | cpp_api.html#_CPPv4N5cudaq16quant |
| /cpp_api.html#_CPPv4NK5cudaq13kra | um_platform16enqueueAsyncTaskEKNS |
| us_channel18is_unitary_mixtureEv) | t6size_tER19KernelExecutionTask), |
| -   [cu                           |     [\[1\]](api/languag           |
| daq::kraus_channel::kraus_channel | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform16enqueueAsyncTaskE |
|     function)](api/lang           | KNSt6size_tERNSt8functionIFvvEEE) |
| uages/cpp_api.html#_CPPv4IDpEN5cu | -   [cudaq::quantum_p             |
| daq13kraus_channel13kraus_channel | latform::finalizeExecutionContext |
| EDpRRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)](api/languages/c    |
|  [\[1\]](api/languages/cpp_api.ht | pp_api.html#_CPPv4NK5cudaq16quant |
| ml#_CPPv4N5cudaq13kraus_channel13 | um_platform24finalizeExecutionCon |
| kraus_channelERK13kraus_channel), | textERN5cudaq16ExecutionContextE) |
|     [\[2\]                        | -   [cudaq::qua                   |
| ](api/languages/cpp_api.html#_CPP | ntum_platform::get_codegen_config |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERKNSt6vectorI8kraus_opEE), |     function)](api/languages/c    |
|     [\[3\]                        | pp_api.html#_CPPv4N5cudaq16quantu |
| ](api/languages/cpp_api.html#_CPP | m_platform18get_codegen_configEv) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cuda                         |
| hannelERRNSt6vectorI8kraus_opEE), | q::quantum_platform::get_exec_ctx |
|     [\[4\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/langua         |
| 13kraus_channel13kraus_channelEv) | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| -                                 | quantum_platform12get_exec_ctxEv) |
| [cudaq::kraus_channel::noise_type | -   [c                            |
|     (C++                          | udaq::quantum_platform::get_noise |
|     member)](api                  |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)](api/languages/c    |
| udaq13kraus_channel10noise_typeE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -                                 | m_platform9get_noiseENSt6size_tE) |
|   [cudaq::kraus_channel::op_names | -   [cudaq:                       |
|     (C++                          | :quantum_platform::get_num_qubits |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq13kraus_channel8op_namesE) | function)](api/languages/cpp_api. |
| -                                 | html#_CPPv4NK5cudaq16quantum_plat |
|  [cudaq::kraus_channel::operator= | form14get_num_qubitsENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_              |
|     function)](api/langua         | platform::get_remote_capabilities |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     (C++                          |
| raus_channelaSERK13kraus_channel) |     function)                     |
| -   [c                            | ](api/languages/cpp_api.html#_CPP |
| udaq::kraus_channel::operator\[\] | v4NK5cudaq16quantum_platform23get |
|     (C++                          | _remote_capabilitiesENSt6size_tE) |
|     function)](api/l              | -   [cudaq::qua                   |
| anguages/cpp_api.html#_CPPv4N5cud | ntum_platform::get_runtime_target |
| aq13kraus_channelixEKNSt6size_tE) |     (C++                          |
| -                                 |     function)](api/languages/cp   |
| [cudaq::kraus_channel::parameters | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform18get_runtime_targetEv) |
|     member)](api                  | -   [cud                          |
| /languages/cpp_api.html#_CPPv4N5c | aq::quantum_platform::is_emulated |
| udaq13kraus_channel10parametersE) |     (C++                          |
| -   [cudaq::krau                  |                                   |
| s_channel::populateDefaultOpNames |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq16quantum_p |
|     function)](api/languages/cp   | latform11is_emulatedENSt6size_tE) |
| p_api.html#_CPPv4N5cudaq13kraus_c | -   [cudaq::                      |
| hannel22populateDefaultOpNamesEv) | quantum_platform::is_library_mode |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::probabilities |     function)](api/languages      |
|     (C++                          | /cpp_api.html#_CPPv4NK5cudaq16qua |
|     member)](api/la               | ntum_platform15is_library_modeEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [c                            |
| q13kraus_channel13probabilitiesE) | udaq::quantum_platform::is_remote |
| -                                 |     (C++                          |
|  [cudaq::kraus_channel::push_back |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4NK5cudaq16quantu |
|     function)](api                | m_platform9is_remoteENSt6size_tE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cuda                         |
| udaq13kraus_channel9push_backE8kr | q::quantum_platform::is_simulator |
| aus_opNSt8optionalINSt6stringEEE) |     (C++                          |
| -   [cudaq::kraus_channel::size   |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)                     | i.html#_CPPv4NK5cudaq16quantum_pl |
| ](api/languages/cpp_api.html#_CPP | atform12is_simulatorENSt6size_tE) |
| v4NK5cudaq13kraus_channel4sizeEv) | -   [c                            |
| -   [                             | udaq::quantum_platform::launchVQE |
| cudaq::kraus_channel::unitary_ops |     (C++                          |
|     (C++                          |     function)](                   |
|     member)](api/                 | api/languages/cpp_api.html#_CPPv4 |
| languages/cpp_api.html#_CPPv4N5cu | N5cudaq16quantum_platform9launchV |
| daq13kraus_channel11unitary_opsE) | QEEKNSt6stringEPKvPN5cudaq8gradie |
| -   [cudaq::kraus_op (C++         | ntERKN5cudaq7spin_opERN5cudaq9opt |
|     struct)](api/languages/cpp_   | imizerEKiKNSt6size_tENSt6size_tE) |
| api.html#_CPPv4N5cudaq8kraus_opE) | -   [cudaq:                       |
| -   [cudaq::kraus_op::adjoint     | :quantum_platform::list_platforms |
|     (C++                          |     (C++                          |
|     functi                        |     function)](api/languag        |
| on)](api/languages/cpp_api.html#_ | es/cpp_api.html#_CPPv4N5cudaq16qu |
| CPPv4NK5cudaq8kraus_op7adjointEv) | antum_platform14list_platformsEv) |
| -   [cudaq::kraus_op::data (C++   | -                                 |
|                                   |    [cudaq::quantum_platform::name |
|  member)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     function)](a                  |
| -   [cudaq::kraus_op::kraus_op    | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | K5cudaq16quantum_platform4nameEv) |
|     func                          | -   [                             |
| tion)](api/languages/cpp_api.html | cudaq::quantum_platform::num_qpus |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |     (C++                          |
| opERRNSt16initializer_listI1TEE), |     function)](api/l              |
|                                   | anguages/cpp_api.html#_CPPv4NK5cu |
|  [\[1\]](api/languages/cpp_api.ht | daq16quantum_platform8num_qpusEv) |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | -   [cudaq::                      |
| pENSt6vectorIN5cudaq7complexEEE), | quantum_platform::onRandomSeedSet |
|     [\[2\]](api/l                 |     (C++                          |
| anguages/cpp_api.html#_CPPv4N5cud |                                   |
| aq8kraus_op8kraus_opERK8kraus_op) | function)](api/languages/cpp_api. |
| -   [cudaq::kraus_op::nCols (C++  | html#_CPPv4N5cudaq16quantum_platf |
|                                   | orm15onRandomSeedSetENSt6size_tE) |
| member)](api/languages/cpp_api.ht | -   [cudaq:                       |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | :quantum_platform::reset_exec_ctx |
| -   [cudaq::kraus_op::nRows (C++  |     (C++                          |
|                                   |     function)](api/languag        |
| member)](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | antum_platform14reset_exec_ctxEv) |
| -   [cudaq::kraus_op::operator=   | -   [cud                          |
|     (C++                          | aq::quantum_platform::reset_noise |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/cpp_ |
| v4N5cudaq8kraus_opaSERK8kraus_op) | api.html#_CPPv4N5cudaq16quantum_p |
| -   [cudaq::kraus_op::precision   | latform11reset_noiseENSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     memb                          | q::quantum_platform::set_exec_ctx |
| er)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8kraus_op9precisionE) |     funct                         |
| -   [cudaq::KrausSelection (C++   | ion)](api/languages/cpp_api.html# |
|     s                             | _CPPv4N5cudaq16quantum_platform12 |
| truct)](api/languages/cpp_api.htm | set_exec_ctxEP16ExecutionContext) |
| l#_CPPv4N5cudaq14KrausSelectionE) | -   [c                            |
| -   [cudaq:                       | udaq::quantum_platform::set_noise |
| :KrausSelection::circuit_location |     (C++                          |
|     (C++                          |     function                      |
|     member)](api/langua           | )](api/languages/cpp_api.html#_CP |
| ges/cpp_api.html#_CPPv4N5cudaq14K | Pv4N5cudaq16quantum_platform9set_ |
| rausSelection16circuit_locationE) | noiseEPK11noise_modelNSt6size_tE) |
| -                                 | -   [cudaq::quantum_platfor       |
|  [cudaq::KrausSelection::is_error | m::supports_explicit_measurements |
|     (C++                          |     (C++                          |
|     member)](a                    |     function)](api/l              |
| pi/languages/cpp_api.html#_CPPv4N | anguages/cpp_api.html#_CPPv4NK5cu |
| 5cudaq14KrausSelection8is_errorE) | daq16quantum_platform30supports_e |
| -   [cudaq::Kra                   | xplicit_measurementsENSt6size_tE) |
| usSelection::kraus_operator_index | -   [cudaq::quantum_pla           |
|     (C++                          | tform::supports_task_distribution |
|     member)](api/languages/       |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq14Kraus |     fu                            |
| Selection20kraus_operator_indexE) | nction)](api/languages/cpp_api.ht |
| -   [cuda                         | ml#_CPPv4NK5cudaq16quantum_platfo |
| q::KrausSelection::KrausSelection | rm26supports_task_distributionEv) |
|     (C++                          | -   [cudaq::quantum               |
|     function)](a                  | _platform::with_execution_context |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14KrausSelection14KrausSele |     function)                     |
| ctionENSt6size_tENSt6vectorINSt6s | ](api/languages/cpp_api.html#_CPP |
| ize_tEEENSt6stringENSt6size_tEb), | v4I0DpEN5cudaq16quantum_platform2 |
|     [\[1\]](api/langu             | 2with_execution_contextEDaR16Exec |
| ages/cpp_api.html#_CPPv4N5cudaq14 | utionContextRR8CallableDpRR4Args) |
| KrausSelection14KrausSelectionEv) | -   [cudaq::QuantumTask (C++      |
| -                                 |     type)](api/languages/cpp_api. |
|   [cudaq::KrausSelection::op_name | html#_CPPv4N5cudaq11QuantumTaskE) |
|     (C++                          | -   [cudaq::qubit (C++            |
|     member)](                     |     type)](api/languages/c        |
| api/languages/cpp_api.html#_CPPv4 | pp_api.html#_CPPv4N5cudaq5qubitE) |
| N5cudaq14KrausSelection7op_nameE) | -   [cudaq::QubitConnectivity     |
| -   [                             |     (C++                          |
| cudaq::KrausSelection::operator== |     ty                            |
|     (C++                          | pe)](api/languages/cpp_api.html#_ |
|     function)](api/languages      | CPPv4N5cudaq17QubitConnectivityE) |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | -   [cudaq::QubitEdge (C++        |
| usSelectioneqERK14KrausSelection) |     type)](api/languages/cpp_a    |
| -                                 | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|    [cudaq::KrausSelection::qubits | -   [cudaq::qudit (C++            |
|     (C++                          |     clas                          |
|     member)]                      | s)](api/languages/cpp_api.html#_C |
| (api/languages/cpp_api.html#_CPPv | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| 4N5cudaq14KrausSelection6qubitsE) | -   [cudaq::qudit::qudit (C++     |
| -   [cudaq::KrausTrajectory (C++  |                                   |
|     st                            | function)](api/languages/cpp_api. |
| ruct)](api/languages/cpp_api.html | html#_CPPv4N5cudaq5qudit5quditEv) |
| #_CPPv4N5cudaq15KrausTrajectoryE) | -   [cudaq::qvector (C++          |
| -                                 |     class)                        |
|  [cudaq::KrausTrajectory::builder | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     function)](ap                 | -   [cudaq::qvector::back (C++    |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](a                  |
| cudaq15KrausTrajectory7builderEv) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cu                           | 5cudaq7qvector4backENSt6size_tE), |
| daq::KrausTrajectory::countErrors |                                   |
|     (C++                          |   [\[1\]](api/languages/cpp_api.h |
|     function)](api/lang           | tml#_CPPv4N5cudaq7qvector4backEv) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::qvector::begin (C++   |
| 15KrausTrajectory11countErrorsEv) |     fu                            |
| -   [                             | nction)](api/languages/cpp_api.ht |
| cudaq::KrausTrajectory::isOrdered | ml#_CPPv4N5cudaq7qvector5beginEv) |
|     (C++                          | -   [cudaq::qvector::clear (C++   |
|     function)](api/l              |     fu                            |
| anguages/cpp_api.html#_CPPv4NK5cu | nction)](api/languages/cpp_api.ht |
| daq15KrausTrajectory9isOrderedEv) | ml#_CPPv4N5cudaq7qvector5clearEv) |
| -   [cudaq::                      | -   [cudaq::qvector::end (C++     |
| KrausTrajectory::kraus_selections |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     member)](api/languag          | html#_CPPv4N5cudaq7qvector3endEv) |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | -   [cudaq::qvector::front (C++   |
| ausTrajectory16kraus_selectionsE) |     function)](ap                 |
| -   [cudaq:                       | i/languages/cpp_api.html#_CPPv4N5 |
| :KrausTrajectory::KrausTrajectory | cudaq7qvector5frontENSt6size_tE), |
|     (C++                          |                                   |
|     function                      |  [\[1\]](api/languages/cpp_api.ht |
| )](api/languages/cpp_api.html#_CP | ml#_CPPv4N5cudaq7qvector5frontEv) |
| Pv4N5cudaq15KrausTrajectory15Krau | -   [cudaq::qvector::operator=    |
| sTrajectoryENSt6size_tENSt6vector |     (C++                          |
| I14KrausSelectionEEdNSt6size_tE), |     functio                       |
|     [\[1\]](api/languag           | n)](api/languages/cpp_api.html#_C |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | PPv4N5cudaq7qvectoraSERK7qvector) |
| ausTrajectory15KrausTrajectoryEv) | -   [cudaq::qvector::operator\[\] |
| -   [cudaq::Kr                    |     (C++                          |
| ausTrajectory::measurement_counts |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/languages        | v4N5cudaq7qvectorixEKNSt6size_tE) |
| /cpp_api.html#_CPPv4N5cudaq15Krau | -   [cudaq::qvector::qvector (C++ |
| sTrajectory18measurement_countsE) |     function)](api/               |
| -   [cud                          | languages/cpp_api.html#_CPPv4N5cu |
| aq::KrausTrajectory::multiplicity | daq7qvector7qvectorENSt6size_tE), |
|     (C++                          |     [\[1\]](a                     |
|     member)](api/lan              | pi/languages/cpp_api.html#_CPPv4N |
| guages/cpp_api.html#_CPPv4N5cudaq | 5cudaq7qvector7qvectorERK5state), |
| 15KrausTrajectory12multiplicityE) |     [\[2\]](api                   |
| -   [                             | /languages/cpp_api.html#_CPPv4N5c |
| cudaq::KrausTrajectory::num_shots | udaq7qvector7qvectorERK7qvector), |
|     (C++                          |     [\[3\]](ap                    |
|     member)](api                  | i/languages/cpp_api.html#_CPPv4N5 |
| /languages/cpp_api.html#_CPPv4N5c | cudaq7qvector7qvectorERR7qvector) |
| udaq15KrausTrajectory9num_shotsE) | -   [cudaq::qvector::size (C++    |
| -   [c                            |     fu                            |
| udaq::KrausTrajectory::operator== | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     function)](api/languages/c    | -   [cudaq::qvector::slice (C++   |
| pp_api.html#_CPPv4NK5cudaq15Kraus |     function)](api/language       |
| TrajectoryeqERK15KrausTrajectory) | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| -   [cu                           | tor5sliceENSt6size_tENSt6size_tE) |
| daq::KrausTrajectory::probability | -   [cudaq::qvector::value_type   |
|     (C++                          |     (C++                          |
|     member)](api/la               |     typ                           |
| nguages/cpp_api.html#_CPPv4N5cuda | e)](api/languages/cpp_api.html#_C |
| q15KrausTrajectory11probabilityE) | PPv4N5cudaq7qvector10value_typeE) |
| -   [cuda                         | -   [cudaq::qview (C++            |
| q::KrausTrajectory::trajectory_id |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     member)](api/lang             | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::qview::back (C++      |
| 5KrausTrajectory13trajectory_idE) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|   [cudaq::KrausTrajectory::weight | v4N5cudaq5qview4backENSt6size_tE) |
|     (C++                          | -   [cudaq::qview::begin (C++     |
|     member)](                     |                                   |
| api/languages/cpp_api.html#_CPPv4 | function)](api/languages/cpp_api. |
| N5cudaq15KrausTrajectory6weightE) | html#_CPPv4N5cudaq5qview5beginEv) |
| -                                 | -   [cudaq::qview::end (C++       |
|    [cudaq::KrausTrajectoryBuilder |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     class)](                      | i.html#_CPPv4N5cudaq5qview3endEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qview::front (C++     |
| N5cudaq22KrausTrajectoryBuilderE) |     function)](                   |
| -   [cud                          | api/languages/cpp_api.html#_CPPv4 |
| aq::KrausTrajectoryBuilder::build | N5cudaq5qview5frontENSt6size_tE), |
|     (C++                          |                                   |
|     function)](api/lang           |    [\[1\]](api/languages/cpp_api. |
| uages/cpp_api.html#_CPPv4NK5cudaq | html#_CPPv4N5cudaq5qview5frontEv) |
| 22KrausTrajectoryBuilder5buildEv) | -   [cudaq::qview::operator\[\]   |
| -   [cud                          |     (C++                          |
| aq::KrausTrajectoryBuilder::setId |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)](api/languages/cpp  | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| _api.html#_CPPv4N5cudaq22KrausTra | -   [cudaq::qview::qview (C++     |
| jectoryBuilder5setIdENSt6size_tE) |     functio                       |
| -   [cudaq::Kraus                 | n)](api/languages/cpp_api.html#_C |
| TrajectoryBuilder::setProbability | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     (C++                          |     [\[1                          |
|     function)](api/languages/cpp  | \]](api/languages/cpp_api.html#_C |
| _api.html#_CPPv4N5cudaq22KrausTra | PPv4N5cudaq5qview5qviewERK5qview) |
| jectoryBuilder14setProbabilityEd) | -   [cudaq::qview::size (C++      |
| -   [cudaq::Krau                  |                                   |
| sTrajectoryBuilder::setSelections | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4NK5cudaq5qview4sizeEv) |
|     function)](api/languag        | -   [cudaq::qview::slice (C++     |
| es/cpp_api.html#_CPPv4N5cudaq22Kr |     function)](api/langua         |
| ausTrajectoryBuilder13setSelectio | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| nsENSt6vectorI14KrausSelectionEE) | iew5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::logical_observable    | -   [cudaq::qview::value_type     |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     t                             |
| pp_api.html#_CPPv4IDpEN5cudaq18lo | ype)](api/languages/cpp_api.html# |
| gical_observableEvDpRR8MeasArgs), | _CPPv4N5cudaq5qview10value_typeE) |
|     [\[1\]](api/l                 | -   [cudaq::range (C++            |
| anguages/cpp_api.html#_CPPv4N5cud |     fun                           |
| aq18logical_observableERKNSt6vect | ction)](api/languages/cpp_api.htm |
| orI14measure_resultEENSt6size_tE) | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| -   [cudaq::matrix_callback (C++  | orI11ElementTypeEE11ElementType), |
|     c                             |     [\[1\]](api/languages/cpp_    |
| lass)](api/languages/cpp_api.html | api.html#_CPPv4I0EN5cudaq5rangeEN |
| #_CPPv4N5cudaq15matrix_callbackE) | St6vectorI11ElementTypeEE11Elemen |
| -   [cudaq::matrix_handler (C++   | tType11ElementType11ElementType), |
|                                   |     [                             |
| class)](api/languages/cpp_api.htm | \[2\]](api/languages/cpp_api.html |
| l#_CPPv4N5cudaq14matrix_handlerE) | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| -   [cudaq::mat                   | -   [cudaq::real (C++             |
| rix_handler::commutation_behavior |     type)](api/languages/         |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq4realE) |
|     struct)](api/languages/       | -   [cudaq::registry (C++         |
| cpp_api.html#_CPPv4N5cudaq14matri |     type)](api/languages/cpp_     |
| x_handler20commutation_behaviorE) | api.html#_CPPv4N5cudaq8registryE) |
| -                                 | -                                 |
|    [cudaq::matrix_handler::define |  [cudaq::registry::RegisteredType |
|     (C++                          |     (C++                          |
|     function)](a                  |     class)](api/                  |
| pi/languages/cpp_api.html#_CPPv4N | languages/cpp_api.html#_CPPv4I0EN |
| 5cudaq14matrix_handler6defineENSt | 5cudaq8registry14RegisteredTypeE) |
| 6stringENSt6vectorINSt7int64_tEEE | -   [cudaq::RemoteCapabilities    |
| RR15matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     struc                         |
|                                   | t)](api/languages/cpp_api.html#_C |
| [\[1\]](api/languages/cpp_api.htm | PPv4N5cudaq18RemoteCapabilitiesE) |
| l#_CPPv4N5cudaq14matrix_handler6d | -   [cudaq::Remot                 |
| efineENSt6stringENSt6vectorINSt7i | eCapabilities::RemoteCapabilities |
| nt64_tEEERR15matrix_callbackRR20d |     (C++                          |
| iag_matrix_callbackRKNSt13unorder |     function)](api/languages/cpp  |
| ed_mapINSt6stringENSt6stringEEE), | _api.html#_CPPv4N5cudaq18RemoteCa |
|     [\[2\]](                      | pabilities18RemoteCapabilitiesEb) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq:                       |
| N5cudaq14matrix_handler6defineENS | :RemoteCapabilities::stateOverlap |
| t6stringENSt6vectorINSt7int64_tEE |     (C++                          |
| ERR15matrix_callbackRRNSt13unorde |     member)](api/langua           |
| red_mapINSt6stringENSt6stringEEE) | ges/cpp_api.html#_CPPv4N5cudaq18R |
| -                                 | emoteCapabilities12stateOverlapE) |
|   [cudaq::matrix_handler::degrees | -                                 |
|     (C++                          |   [cudaq::RemoteCapabilities::vqe |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4NK |     member)](                     |
| 5cudaq14matrix_handler7degreesEv) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | N5cudaq18RemoteCapabilities3vqeE) |
|  [cudaq::matrix_handler::displace | -   [cudaq::Resources (C++        |
|     (C++                          |     class)](api/languages/cpp_a   |
|     function)](api/language       | pi.html#_CPPv4N5cudaq9ResourcesE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::run (C++              |
| rix_handler8displaceENSt6size_tE) |     function)]                    |
| -   [cudaq::matrix                | (api/languages/cpp_api.html#_CPPv |
| _handler::get_expected_dimensions | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
|     (C++                          | 5invoke_result_tINSt7decay_tI13Qu |
|                                   | antumKernelEEDpNSt7decay_tI4ARGSE |
|    function)](api/languages/cpp_a | EEEEENSt6size_tERN5cudaq11noise_m |
| pi.html#_CPPv4NK5cudaq14matrix_ha | odelERR13QuantumKernelDpRR4ARGS), |
| ndler23get_expected_dimensionsEv) |     [\[1\]](api/langu             |
| -   [cudaq::matrix_ha             | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| ndler::get_parameter_descriptions | daq3runENSt6vectorINSt15invoke_re |
|     (C++                          | sult_tINSt7decay_tI13QuantumKerne |
|                                   | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| function)](api/languages/cpp_api. | ize_tERR13QuantumKernelDpRR4ARGS) |
| html#_CPPv4NK5cudaq14matrix_handl | -   [cudaq::run_async (C++        |
| er26get_parameter_descriptionsEv) |     functio                       |
| -   [c                            | n)](api/languages/cpp_api.html#_C |
| udaq::matrix_handler::instantiate | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     (C++                          | tureINSt6vectorINSt15invoke_resul |
|     function)](a                  | t_tINSt7decay_tI13QuantumKernelEE |
| pi/languages/cpp_api.html#_CPPv4N | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| 5cudaq14matrix_handler11instantia | ze_tENSt6size_tERN5cudaq11noise_m |
| teENSt6stringERKNSt6vectorINSt6si | odelERR13QuantumKernelDpRR4ARGS), |
| ze_tEEERK20commutation_behavior), |     [\[1\]](api/la                |
|     [\[1\]](                      | nguages/cpp_api.html#_CPPv4I0DpEN |
| api/languages/cpp_api.html#_CPPv4 | 5cudaq9run_asyncENSt6futureINSt6v |
| N5cudaq14matrix_handler11instanti | ectorINSt15invoke_result_tINSt7de |
| ateENSt6stringERRNSt6vectorINSt6s | cay_tI13QuantumKernelEEDpNSt7deca |
| ize_tEEERK20commutation_behavior) | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| -   [cuda                         | ize_tERR13QuantumKernelDpRR4ARGS) |
| q::matrix_handler::matrix_handler | -   [cudaq::RuntimeTarget (C++    |
|     (C++                          |                                   |
|     function)](api/languag        | struct)](api/languages/cpp_api.ht |
| es/cpp_api.html#_CPPv4I0_NSt11ena | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| ble_if_tINSt12is_base_of_vI16oper | -   [cudaq::sample (C++           |
| ator_handler1TEEbEEEN5cudaq14matr |     function)](api/languages/c    |
| ix_handler14matrix_handlerERK1T), | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     [\[1\]](ap                    | mpleE13sample_resultRK14sample_op |
| i/languages/cpp_api.html#_CPPv4I0 | tionsRR13QuantumKernelDpRR4Args), |
| _NSt11enable_if_tINSt12is_base_of |     [\[1\                         |
| _vI16operator_handler1TEEbEEEN5cu | ]](api/languages/cpp_api.html#_CP |
| daq14matrix_handler14matrix_handl | Pv4I0DpEN5cudaq6sampleE13sample_r |
| erERK1TRK20commutation_behavior), | esultRR13QuantumKernelDpRR4Args), |
|     [\[2\]](api/languages/cpp_ap  |     [\                            |
| i.html#_CPPv4N5cudaq14matrix_hand | [2\]](api/languages/cpp_api.html# |
| ler14matrix_handlerENSt6size_tE), | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|     [\[3\]](api/                  | ize_tERR13QuantumKernelDpRR4Args) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::sample_options (C++   |
| daq14matrix_handler14matrix_handl |     s                             |
| erENSt6stringERKNSt6vectorINSt6si | truct)](api/languages/cpp_api.htm |
| ze_tEEERK20commutation_behavior), | l#_CPPv4N5cudaq14sample_optionsE) |
|     [\[4\]](api/                  | -   [cudaq::sample_result (C++    |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq14matrix_handler14matrix_handl |  class)](api/languages/cpp_api.ht |
| erENSt6stringERRNSt6vectorINSt6si | ml#_CPPv4N5cudaq13sample_resultE) |
| ze_tEEERK20commutation_behavior), | -   [cudaq::sample_result::append |
|     [\                            |     (C++                          |
| [5\]](api/languages/cpp_api.html# |     function)](api/languages/cpp_ |
| _CPPv4N5cudaq14matrix_handler14ma | api.html#_CPPv4N5cudaq13sample_re |
| trix_handlerERK14matrix_handler), | sult6appendERK15ExecutionResultb) |
|     [                             | -   [cudaq::sample_result::begin  |
| \[6\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq14matrix_handler14m |     function)]                    |
| atrix_handlerERR14matrix_handler) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4N5cudaq13sample_result5beginEv), |
|  [cudaq::matrix_handler::momentum |     [\[1\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/language       | 4NK5cudaq13sample_result5beginEv) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::sample_result::cbegin |
| rix_handler8momentumENSt6size_tE) |     (C++                          |
| -                                 |     function)](                   |
|    [cudaq::matrix_handler::number | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq13sample_result6cbeginEv) |
|     function)](api/langua         | -   [cudaq::sample_result::cend   |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6numberENSt6size_tE) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
| [cudaq::matrix_handler::operator= | v4NK5cudaq13sample_result4cendEv) |
|     (C++                          | -   [cudaq::sample_result::clear  |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     function)                     |
| l#_CPPv4I0_NSt11enable_if_tIXaant | ](api/languages/cpp_api.html#_CPP |
| NSt7is_sameI1T14matrix_handlerE5v | v4N5cudaq13sample_result5clearEv) |
| alueENSt12is_base_of_vI16operator | -   [cudaq::sample_result::count  |
| _handler1TEEEbEEEN5cudaq14matrix_ |     (C++                          |
| handleraSER14matrix_handlerRK1T), |     function)](                   |
|     [\[1\]](api/languages         | api/languages/cpp_api.html#_CPPv4 |
| /cpp_api.html#_CPPv4N5cudaq14matr | NK5cudaq13sample_result5countENSt |
| ix_handleraSERK14matrix_handler), | 11string_viewEKNSt11string_viewE) |
|     [\[2\]](api/language          | -   [                             |
| s/cpp_api.html#_CPPv4N5cudaq14mat | cudaq::sample_result::deserialize |
| rix_handleraSERR14matrix_handler) |     (C++                          |
| -   [                             |     functio                       |
| cudaq::matrix_handler::operator== | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq13sample_result11deser |
|     function)](api/languages      | ializeERNSt6vectorINSt6size_tEEE) |
| /cpp_api.html#_CPPv4NK5cudaq14mat | -   [cudaq::sample_result::dump   |
| rix_handlereqERK14matrix_handler) |     (C++                          |
| -                                 |     function)](api/languag        |
|    [cudaq::matrix_handler::parity | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     (C++                          | ample_result4dumpERNSt7ostreamE), |
|     function)](api/langua         |     [\[1\]                        |
| ges/cpp_api.html#_CPPv4N5cudaq14m | ](api/languages/cpp_api.html#_CPP |
| atrix_handler6parityENSt6size_tE) | v4NK5cudaq13sample_result4dumpEv) |
| -                                 | -   [cudaq::sample_result::end    |
|  [cudaq::matrix_handler::position |     (C++                          |
|     (C++                          |     function                      |
|     function)](api/language       | )](api/languages/cpp_api.html#_CP |
| s/cpp_api.html#_CPPv4N5cudaq14mat | Pv4N5cudaq13sample_result3endEv), |
| rix_handler8positionENSt6size_tE) |     [\[1\                         |
| -   [cudaq::                      | ]](api/languages/cpp_api.html#_CP |
| matrix_handler::remove_definition | Pv4NK5cudaq13sample_result3endEv) |
|     (C++                          | -   [                             |
|     fu                            | cudaq::sample_result::expectation |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14matrix_handler1 |     f                             |
| 7remove_definitionERKNSt6stringE) | unction)](api/languages/cpp_api.h |
| -                                 | tml#_CPPv4NK5cudaq13sample_result |
|   [cudaq::matrix_handler::squeeze | 11expectationEKNSt11string_viewE) |
|     (C++                          | -   [c                            |
|     function)](api/languag        | udaq::sample_result::get_marginal |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     (C++                          |
| trix_handler7squeezeENSt6size_tE) |     function)](api/languages/cpp_ |
| -   [cudaq::m                     | api.html#_CPPv4NK5cudaq13sample_r |
| atrix_handler::to_diagonal_matrix | esult12get_marginalERKNSt6vectorI |
|     (C++                          | NSt6size_tEEEKNSt11string_viewE), |
|     function)](api/lang           |     [\[1\]](api/languages/cpp_    |
| uages/cpp_api.html#_CPPv4NK5cudaq | api.html#_CPPv4NK5cudaq13sample_r |
| 14matrix_handler18to_diagonal_mat | esult12get_marginalERRKNSt6vector |
| rixERNSt13unordered_mapINSt6size_ | INSt6size_tEEEKNSt11string_viewE) |
| tENSt7int64_tEEERKNSt13unordered_ | -   [cuda                         |
| mapINSt6stringENSt7complexIdEEEE) | q::sample_result::get_total_shots |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::to_matrix |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq13 |
|     function)                     | sample_result15get_total_shotsEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cuda                         |
| v4NK5cudaq14matrix_handler9to_mat | q::sample_result::has_even_parity |
| rixERNSt13unordered_mapINSt6size_ |     (C++                          |
| tENSt7int64_tEEERKNSt13unordered_ |     fun                           |
| mapINSt6stringENSt7complexIdEEEE) | ction)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq13sample_result15h |
| [cudaq::matrix_handler::to_string | as_even_parityENSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|     function)](api/               | q::sample_result::has_expectation |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14matrix_handler9to_stringEb) |     funct                         |
| -                                 | ion)](api/languages/cpp_api.html# |
| [cudaq::matrix_handler::unique_id | _CPPv4NK5cudaq13sample_result15ha |
|     (C++                          | s_expectationEKNSt11string_viewE) |
|     function)](api/               | -   [cu                           |
| languages/cpp_api.html#_CPPv4NK5c | daq::sample_result::most_probable |
| udaq14matrix_handler9unique_idEv) |     (C++                          |
| -   [cudaq:                       |     fun                           |
| :matrix_handler::\~matrix_handler | ction)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NK5cudaq13sample_result13 |
|     functi                        | most_probableEKNSt11string_viewE) |
| on)](api/languages/cpp_api.html#_ | -                                 |
| CPPv4N5cudaq14matrix_handlerD0Ev) | [cudaq::sample_result::operator+= |
| -   [cudaq::matrix_op (C++        |     (C++                          |
|     type)](api/languages/cpp_a    |     function)](api/langua         |
| pi.html#_CPPv4N5cudaq9matrix_opE) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -   [cudaq::matrix_op_term (C++   | ample_resultpLERK13sample_result) |
|                                   | -                                 |
|  type)](api/languages/cpp_api.htm |  [cudaq::sample_result::operator= |
| l#_CPPv4N5cudaq14matrix_op_termE) |     (C++                          |
| -                                 |     function)](api/langua         |
|    [cudaq::mdiag_operator_handler | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     (C++                          | ample_resultaSERR13sample_result) |
|     class)](                      | -                                 |
| api/languages/cpp_api.html#_CPPv4 | [cudaq::sample_result::operator== |
| N5cudaq22mdiag_operator_handlerE) |     (C++                          |
| -   [cudaq::measure_handle (C++   |     function)](api/languag        |
|                                   | es/cpp_api.html#_CPPv4NK5cudaq13s |
| class)](api/languages/cpp_api.htm | ample_resulteqERK13sample_result) |
| l#_CPPv4N5cudaq14measure_handleE) | -   [                             |
| -   [cudaq::measure_result (C++   | cudaq::sample_result::probability |
|                                   |     (C++                          |
|  type)](api/languages/cpp_api.htm |     function)](api/lan            |
| l#_CPPv4N5cudaq14measure_resultE) | guages/cpp_api.html#_CPPv4NK5cuda |
| -   [cudaq::mpi (C++              | q13sample_result11probabilityENSt |
|     type)](api/languages          | 11string_viewEKNSt11string_viewE) |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | -   [cud                          |
| -   [cudaq::mpi::all_gather (C++  | aq::sample_result::register_names |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/langu          |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| RNSt6vectorIdEERKNSt6vectorIdEE), | 3sample_result14register_namesEv) |
|                                   | -                                 |
|   [\[1\]](api/languages/cpp_api.h |    [cudaq::sample_result::reorder |
| tml#_CPPv4N5cudaq3mpi10all_gather |     (C++                          |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     function)](api/langua         |
| -   [cudaq::mpi::all_reduce (C++  | ges/cpp_api.html#_CPPv4N5cudaq13s |
|                                   | ample_result7reorderERKNSt6vector |
|  function)](api/languages/cpp_api | INSt6size_tEEEKNSt11string_viewE) |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | -   [cu                           |
| reduceE1TRK1TRK14BinaryFunction), | daq::sample_result::sample_result |
|     [\[1\]](api/langu             |     (C++                          |
| ages/cpp_api.html#_CPPv4I00EN5cud |     func                          |
| aq3mpi10all_reduceE1TRK1TRK4Func) | tion)](api/languages/cpp_api.html |
| -   [cudaq::mpi::broadcast (C++   | #_CPPv4N5cudaq13sample_result13sa |
|     function)](api/               | mple_resultERK15ExecutionResult), |
| languages/cpp_api.html#_CPPv4N5cu |     [\[1\]](api/la                |
| daq3mpi9broadcastERNSt6stringEi), | nguages/cpp_api.html#_CPPv4N5cuda |
|     [\[1\]](api/la                | q13sample_result13sample_resultER |
| nguages/cpp_api.html#_CPPv4N5cuda | KNSt6vectorI15ExecutionResultEE), |
| q3mpi9broadcastERNSt6vectorIdEEi) |                                   |
| -   [cudaq::mpi::finalize (C++    |  [\[2\]](api/languages/cpp_api.ht |
|     f                             | ml#_CPPv4N5cudaq13sample_result13 |
| unction)](api/languages/cpp_api.h | sample_resultERR13sample_result), |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     [                             |
| -   [cudaq::mpi::initialize (C++  | \[3\]](api/languages/cpp_api.html |
|     function                      | #_CPPv4N5cudaq13sample_result13sa |
| )](api/languages/cpp_api.html#_CP | mple_resultERR15ExecutionResult), |
| Pv4N5cudaq3mpi10initializeEiPPc), |     [\[4\]](api/lan               |
|     [                             | guages/cpp_api.html#_CPPv4N5cudaq |
| \[1\]](api/languages/cpp_api.html | 13sample_result13sample_resultEdR |
| #_CPPv4N5cudaq3mpi10initializeEv) | KNSt6vectorI15ExecutionResultEE), |
| -   [cudaq::mpi::is_initialized   |     [\[5\]](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     function                      | 13sample_result13sample_resultEv) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq3mpi14is_initializedEv) |  [cudaq::sample_result::serialize |
| -   [cudaq::mpi::num_ranks (C++   |     (C++                          |
|     fu                            |     function)](api                |
| nction)](api/languages/cpp_api.ht | /languages/cpp_api.html#_CPPv4NK5 |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | cudaq13sample_result9serializeEv) |
| -   [cudaq::mpi::rank (C++        | -   [cudaq::sample_result::size   |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](api/languages/c    |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | pp_api.html#_CPPv4NK5cudaq13sampl |
| -   [cudaq::noise_model (C++      | e_result4sizeEKNSt11string_viewE) |
|                                   | -   [cudaq::sample_result::to_map |
|    class)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq11noise_modelE) |     function)](api/languages/cpp  |
| -   [cudaq::n                     | _api.html#_CPPv4NK5cudaq13sample_ |
| oise_model::add_all_qubit_channel | result6to_mapEKNSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|     function)](api                | q::sample_result::\~sample_result |
| /languages/cpp_api.html#_CPPv4IDp |     (C++                          |
| EN5cudaq11noise_model21add_all_qu |     funct                         |
| bit_channelEvRK13kraus_channeli), | ion)](api/languages/cpp_api.html# |
|     [\[1\]](api/langua            | _CPPv4N5cudaq13sample_resultD0Ev) |
| ges/cpp_api.html#_CPPv4N5cudaq11n | -   [cudaq::scalar_callback (C++  |
| oise_model21add_all_qubit_channel |     c                             |
| ERKNSt6stringERK13kraus_channeli) | lass)](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq15scalar_callbackE) |
|  [cudaq::noise_model::add_channel | -   [c                            |
|     (C++                          | udaq::scalar_callback::operator() |
|     funct                         |     (C++                          |
| ion)](api/languages/cpp_api.html# |     function)](api/language       |
| _CPPv4IDpEN5cudaq11noise_model11a | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| dd_channelEvRK15PredicateFuncTy), | alar_callbackclERKNSt13unordered_ |
|     [\[1\]](api/languages/cpp_    | mapINSt6stringENSt7complexIdEEEE) |
| api.html#_CPPv4IDpEN5cudaq11noise | -   [                             |
| _model11add_channelEvRKNSt6vector | cudaq::scalar_callback::operator= |
| INSt6size_tEEERK13kraus_channel), |     (C++                          |
|     [\[2\]](ap                    |     function)](api/languages/c    |
| i/languages/cpp_api.html#_CPPv4N5 | pp_api.html#_CPPv4N5cudaq15scalar |
| cudaq11noise_model11add_channelER | _callbackaSERK15scalar_callback), |
| KNSt6stringERK15PredicateFuncTy), |     [\[1\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4N5cudaq15scala |
| [\[3\]](api/languages/cpp_api.htm | r_callbackaSERR15scalar_callback) |
| l#_CPPv4N5cudaq11noise_model11add | -   [cudaq:                       |
| _channelERKNSt6stringERKNSt6vecto | :scalar_callback::scalar_callback |
| rINSt6size_tEEERK13kraus_channel) |     (C++                          |
| -   [cudaq::noise_model::empty    |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4I0_NSt11ena |
|     function                      | ble_if_tINSt16is_invocable_r_vINS |
| )](api/languages/cpp_api.html#_CP | t7complexIdEE8CallableRKNSt13unor |
| Pv4NK5cudaq11noise_model5emptyEv) | dered_mapINSt6stringENSt7complexI |
| -                                 | dEEEEEEbEEEN5cudaq15scalar_callba |
| [cudaq::noise_model::get_channels | ck15scalar_callbackERR8Callable), |
|     (C++                          |     [\[1\                         |
|     function)](api/l              | ]](api/languages/cpp_api.html#_CP |
| anguages/cpp_api.html#_CPPv4I0ENK | Pv4N5cudaq15scalar_callback15scal |
| 5cudaq11noise_model12get_channels | ar_callbackERK15scalar_callback), |
| ENSt6vectorI13kraus_channelEERKNS |     [\[2                          |
| t6vectorINSt6size_tEEERKNSt6vecto | \]](api/languages/cpp_api.html#_C |
| rINSt6size_tEEERKNSt6vectorIdEE), | PPv4N5cudaq15scalar_callback15sca |
|     [\[1\]](api/languages/cpp_a   | lar_callbackERR15scalar_callback) |
| pi.html#_CPPv4NK5cudaq11noise_mod | -   [cudaq::scalar_operator (C++  |
| el12get_channelsERKNSt6stringERKN |     c                             |
| St6vectorINSt6size_tEEERKNSt6vect | lass)](api/languages/cpp_api.html |
| orINSt6size_tEEERKNSt6vectorIdEE) | #_CPPv4N5cudaq15scalar_operatorE) |
| -                                 | -                                 |
|  [cudaq::noise_model::noise_model | [cudaq::scalar_operator::evaluate |
|     (C++                          |     (C++                          |
|     function)](api                |                                   |
| /languages/cpp_api.html#_CPPv4N5c |    function)](api/languages/cpp_a |
| udaq11noise_model11noise_modelEv) | pi.html#_CPPv4NK5cudaq15scalar_op |
| -   [cu                           | erator8evaluateERKNSt13unordered_ |
| daq::noise_model::PredicateFuncTy | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [cudaq::scalar_ope            |
|     type)](api/la                 | rator::get_parameter_descriptions |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q11noise_model15PredicateFuncTyE) |     f                             |
| -   [cud                          | unction)](api/languages/cpp_api.h |
| aq::noise_model::register_channel | tml#_CPPv4NK5cudaq15scalar_operat |
|     (C++                          | or26get_parameter_descriptionsEv) |
|     function)](api/languages      | -   [cu                           |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | daq::scalar_operator::is_constant |
| noise_model16register_channelEvv) |     (C++                          |
| -   [cudaq::                      |     function)](api/lang           |
| noise_model::requires_constructor | uages/cpp_api.html#_CPPv4NK5cudaq |
|     (C++                          | 15scalar_operator11is_constantEv) |
|     type)](api/languages/cp       | -   [c                            |
| p_api.html#_CPPv4I0DpEN5cudaq11no | udaq::scalar_operator::operator\* |
| ise_model20requires_constructorE) |     (C++                          |
| -   [cudaq::noise_model_type (C++ |     function                      |
|     e                             | )](api/languages/cpp_api.html#_CP |
| num)](api/languages/cpp_api.html# | Pv4N5cudaq15scalar_operatormlENSt |
| _CPPv4N5cudaq16noise_model_typeE) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::no                    |     [\[1\                         |
| ise_model_type::amplitude_damping | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormlENSt |
|     enumerator)](api/languages    | 7complexIdEERR15scalar_operator), |
| /cpp_api.html#_CPPv4N5cudaq16nois |     [\[2\]](api/languages/cp      |
| e_model_type17amplitude_dampingE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::noise_mode            | operatormlEdRK15scalar_operator), |
| l_type::amplitude_damping_channel |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     e                             | operatormlEdRR15scalar_operator), |
| numerator)](api/languages/cpp_api |     [\[4\]](api/languages         |
| .html#_CPPv4N5cudaq16noise_model_ | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| type25amplitude_damping_channelE) | alar_operatormlENSt7complexIdEE), |
| -   [cudaq::n                     |     [\[5\]](api/languages/cpp     |
| oise_model_type::bit_flip_channel | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatormlERK15scalar_operator), |
|     enumerator)](api/language     |     [\[6\]]                       |
| s/cpp_api.html#_CPPv4N5cudaq16noi | (api/languages/cpp_api.html#_CPPv |
| se_model_type16bit_flip_channelE) | 4NKR5cudaq15scalar_operatormlEd), |
| -   [cudaq::                      |     [\[7\]](api/language          |
| noise_model_type::depolarization1 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormlENSt7complexIdEE), |
|     enumerator)](api/languag      |     [\[8\]](api/languages/cp      |
| es/cpp_api.html#_CPPv4N5cudaq16no | p_api.html#_CPPv4NO5cudaq15scalar |
| ise_model_type15depolarization1E) | _operatormlERK15scalar_operator), |
| -   [cudaq::                      |     [\[9\                         |
| noise_model_type::depolarization2 | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatormlEd) |
|     enumerator)](api/languag      | -   [cu                           |
| es/cpp_api.html#_CPPv4N5cudaq16no | daq::scalar_operator::operator\*= |
| ise_model_type15depolarization2E) |     (C++                          |
| -   [cudaq::noise_m               |     function)](api/languag        |
| odel_type::depolarization_channel | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormLENSt7complexIdEE), |
|                                   |     [\[1\]](api/languages/c       |
|   enumerator)](api/languages/cpp_ | pp_api.html#_CPPv4N5cudaq15scalar |
| api.html#_CPPv4N5cudaq16noise_mod | _operatormLERK15scalar_operator), |
| el_type22depolarization_channelE) |     [\[2                          |
| -                                 | \]](api/languages/cpp_api.html#_C |
|  [cudaq::noise_model_type::pauli1 | PPv4N5cudaq15scalar_operatormLEd) |
|     (C++                          | -   [                             |
|     enumerator)](a                | cudaq::scalar_operator::operator+ |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq16noise_model_type6pauli1E) |     function                      |
| -                                 | )](api/languages/cpp_api.html#_CP |
|  [cudaq::noise_model_type::pauli2 | Pv4N5cudaq15scalar_operatorplENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     enumerator)](a                |     [\[1\                         |
| pi/languages/cpp_api.html#_CPPv4N | ]](api/languages/cpp_api.html#_CP |
| 5cudaq16noise_model_type6pauli2E) | Pv4N5cudaq15scalar_operatorplENSt |
| -   [cudaq                        | 7complexIdEERR15scalar_operator), |
| ::noise_model_type::phase_damping |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](api/langu        | operatorplEdRK15scalar_operator), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     [\[3\]](api/languages/cp      |
| noise_model_type13phase_dampingE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::noi                   | operatorplEdRR15scalar_operator), |
| se_model_type::phase_flip_channel |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     enumerator)](api/languages/   | alar_operatorplENSt7complexIdEE), |
| cpp_api.html#_CPPv4N5cudaq16noise |     [\[5\]](api/languages/cpp     |
| _model_type18phase_flip_channelE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -                                 | _operatorplERK15scalar_operator), |
| [cudaq::noise_model_type::unknown |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     enumerator)](ap               | 4NKR5cudaq15scalar_operatorplEd), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[7\]]                       |
| cudaq16noise_model_type7unknownE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatorplEv), |
| [cudaq::noise_model_type::x_error |     [\[8\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     enumerator)](ap               | alar_operatorplENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[9\]](api/languages/cp      |
| cudaq16noise_model_type7x_errorE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -                                 | _operatorplERK15scalar_operator), |
| [cudaq::noise_model_type::y_error |     [\[10\]                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     enumerator)](ap               | v4NO5cudaq15scalar_operatorplEd), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[11\                        |
| cudaq16noise_model_type7y_errorE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4NO5cudaq15scalar_operatorplEv) |
| [cudaq::noise_model_type::z_error | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator+= |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languag        |
| cudaq16noise_model_type7z_errorE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::num_available_gpus    | alar_operatorpLENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     function                      | pp_api.html#_CPPv4N5cudaq15scalar |
| )](api/languages/cpp_api.html#_CP | _operatorpLERK15scalar_operator), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[2                          |
| -   [cudaq::observe (C++          | \]](api/languages/cpp_api.html#_C |
|     function)]                    | PPv4N5cudaq15scalar_operatorpLEd) |
| (api/languages/cpp_api.html#_CPPv | -   [                             |
| 4I00DpEN5cudaq7observeENSt6vector | cudaq::scalar_operator::operator- |
| I14observe_resultEERR13QuantumKer |     (C++                          |
| nelRK15SpinOpContainerDpRR4Args), |     function                      |
|     [\[1\]](api/languages/cpp_ap  | )](api/languages/cpp_api.html#_CP |
| i.html#_CPPv4I0DpEN5cudaq7observe | Pv4N5cudaq15scalar_operatormiENSt |
| E14observe_resultNSt6size_tERR13Q | 7complexIdEERK15scalar_operator), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[1\                         |
|     [\[                           | ]](api/languages/cpp_api.html#_CP |
| 2\]](api/languages/cpp_api.html#_ | Pv4N5cudaq15scalar_operatormiENSt |
| CPPv4I0DpEN5cudaq7observeE14obser | 7complexIdEERR15scalar_operator), |
| ve_resultRK15observe_optionsRR13Q |     [\[2\]](api/languages/cp      |
| uantumKernelRK7spin_opDpRR4Args), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[3\]](api/lang              | operatormiEdRK15scalar_operator), |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     [\[3\]](api/languages/cp      |
| udaq7observeE14observe_resultRR13 | p_api.html#_CPPv4N5cudaq15scalar_ |
| QuantumKernelRK7spin_opDpRR4Args) | operatormiEdRR15scalar_operator), |
| -   [cudaq::observe_options (C++  |     [\[4\]](api/languages         |
|     st                            | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ruct)](api/languages/cpp_api.html | alar_operatormiENSt7complexIdEE), |
| #_CPPv4N5cudaq15observe_optionsE) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::observe_result (C++   | _api.html#_CPPv4NKR5cudaq15scalar |
|                                   | _operatormiERK15scalar_operator), |
| class)](api/languages/cpp_api.htm |     [\[6\]]                       |
| l#_CPPv4N5cudaq14observe_resultE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatormiEd), |
|    [cudaq::observe_result::counts |     [\[7\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/languages/c    | 4NKR5cudaq15scalar_operatormiEv), |
| pp_api.html#_CPPv4N5cudaq14observ |     [\[8\]](api/language          |
| e_result6countsERK12spin_op_term) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::observe_result::dump  | alar_operatormiENSt7complexIdEE), |
|     (C++                          |     [\[9\]](api/languages/cp      |
|     function)                     | p_api.html#_CPPv4NO5cudaq15scalar |
| ](api/languages/cpp_api.html#_CPP | _operatormiERK15scalar_operator), |
| v4N5cudaq14observe_result4dumpEv) |     [\[10\]                       |
| -   [c                            | ](api/languages/cpp_api.html#_CPP |
| udaq::observe_result::expectation | v4NO5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[11\                        |
|                                   | ]](api/languages/cpp_api.html#_CP |
| function)](api/languages/cpp_api. | Pv4NO5cudaq15scalar_operatormiEv) |
| html#_CPPv4N5cudaq14observe_resul | -   [c                            |
| t11expectationERK12spin_op_term), | udaq::scalar_operator::operator-= |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languag        |
| q14observe_result11expectationEv) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cuda                         | alar_operatormIENSt7complexIdEE), |
| q::observe_result::id_coefficient |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function)](api/langu          | _operatormIERK15scalar_operator), |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     [\[2                          |
| observe_result14id_coefficientEv) | \]](api/languages/cpp_api.html#_C |
| -   [cuda                         | PPv4N5cudaq15scalar_operatormIEd) |
| q::observe_result::observe_result | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator/ |
|                                   |     (C++                          |
|   function)](api/languages/cpp_ap |     function                      |
| i.html#_CPPv4N5cudaq14observe_res | )](api/languages/cpp_api.html#_CP |
| ult14observe_resultEdRK7spin_op), | Pv4N5cudaq15scalar_operatordvENSt |
|     [\[1\]](a                     | 7complexIdEERK15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[1\                         |
| 5cudaq14observe_result14observe_r | ]](api/languages/cpp_api.html#_CP |
| esultEdRK7spin_op13sample_result) | Pv4N5cudaq15scalar_operatordvENSt |
| -                                 | 7complexIdEERR15scalar_operator), |
|  [cudaq::observe_result::operator |     [\[2\]](api/languages/cp      |
|     double (C++                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|     functio                       | operatordvEdRK15scalar_operator), |
| n)](api/languages/cpp_api.html#_C |     [\[3\]](api/languages/cp      |
| PPv4N5cudaq14observe_resultcvdEv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatordvEdRR15scalar_operator), |
|  [cudaq::observe_result::raw_data |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     function)](ap                 | alar_operatordvENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[5\]](api/languages/cpp     |
| cudaq14observe_result8raw_dataEv) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::operator_handler (C++ | _operatordvERK15scalar_operator), |
|     cl                            |     [\[6\]]                       |
| ass)](api/languages/cpp_api.html# | (api/languages/cpp_api.html#_CPPv |
| _CPPv4N5cudaq16operator_handlerE) | 4NKR5cudaq15scalar_operatordvEd), |
| -   [cudaq::optimizable_function  |     [\[7\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     class)                        | alar_operatordvENSt7complexIdEE), |
| ](api/languages/cpp_api.html#_CPP |     [\[8\]](api/languages/cp      |
| v4N5cudaq20optimizable_functionE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::optimization_result   | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     type                          | ]](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4NO5cudaq15scalar_operatordvEd) |
| Pv4N5cudaq19optimization_resultE) | -   [c                            |
| -   [cudaq::optimizer (C++        | udaq::scalar_operator::operator/= |
|     class)](api/languages/cpp_a   |     (C++                          |
| pi.html#_CPPv4N5cudaq9optimizerE) |     function)](api/languag        |
| -   [cudaq::optimizer::optimize   | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatordVENSt7complexIdEE), |
|                                   |     [\[1\]](api/languages/c       |
|  function)](api/languages/cpp_api | pp_api.html#_CPPv4N5cudaq15scalar |
| .html#_CPPv4N5cudaq9optimizer8opt | _operatordVERK15scalar_operator), |
| imizeEKiRR20optimizable_function) |     [\[2                          |
| -   [cu                           | \]](api/languages/cpp_api.html#_C |
| daq::optimizer::requiresGradients | PPv4N5cudaq15scalar_operatordVEd) |
|     (C++                          | -   [                             |
|     function)](api/la             | cudaq::scalar_operator::operator= |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9optimizer17requiresGradientsEv) |     function)](api/languages/c    |
| -   [cudaq::orca (C++             | pp_api.html#_CPPv4N5cudaq15scalar |
|     type)](api/languages/         | _operatoraSERK15scalar_operator), |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     [\[1\]](api/languages/        |
| -   [cudaq::orca::sample (C++     | cpp_api.html#_CPPv4N5cudaq15scala |
|     function)](api/languages/c    | r_operatoraSERR15scalar_operator) |
| pp_api.html#_CPPv4N5cudaq4orca6sa | -   [c                            |
| mpleERNSt6vectorINSt6size_tEEERNS | udaq::scalar_operator::operator== |
| t6vectorINSt6size_tEEERNSt6vector |     (C++                          |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     function)](api/languages/c    |
|     [\[1\]]                       | pp_api.html#_CPPv4NK5cudaq15scala |
| (api/languages/cpp_api.html#_CPPv | r_operatoreqERK15scalar_operator) |
| 4N5cudaq4orca6sampleERNSt6vectorI | -   [cudaq:                       |
| NSt6size_tEEERNSt6vectorINSt6size | :scalar_operator::scalar_operator |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     (C++                          |
| -   [cudaq::orca::sample_async    |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq15scalar_operator15 |
| function)](api/languages/cpp_api. | scalar_operatorENSt7complexIdEE), |
| html#_CPPv4N5cudaq4orca12sample_a |     [\[1\]](api/langu             |
| syncERNSt6vectorINSt6size_tEEERNS | ages/cpp_api.html#_CPPv4N5cudaq15 |
| t6vectorINSt6size_tEEERNSt6vector | scalar_operator15scalar_operatorE |
| IdEERNSt6vectorIdEEiNSt6size_tE), | RK15scalar_callbackRRNSt13unorder |
|     [\[1\]](api/la                | ed_mapINSt6stringENSt6stringEEE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[2\                         |
| q4orca12sample_asyncERNSt6vectorI | ]](api/languages/cpp_api.html#_CP |
| NSt6size_tEEERNSt6vectorINSt6size | Pv4N5cudaq15scalar_operator15scal |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | ar_operatorERK15scalar_operator), |
| -   [cudaq::OrcaRemoteRESTQPU     |     [\[3\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     cla                           | scalar_operator15scalar_operatorE |
| ss)](api/languages/cpp_api.html#_ | RR15scalar_callbackRRNSt13unorder |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | ed_mapINSt6stringENSt6stringEEE), |
| -   [cudaq::pauli1 (C++           |     [\[4\                         |
|     class)](api/languages/cp      | ]](api/languages/cpp_api.html#_CP |
| p_api.html#_CPPv4N5cudaq6pauli1E) | Pv4N5cudaq15scalar_operator15scal |
| -                                 | ar_operatorERR15scalar_operator), |
|    [cudaq::pauli1::num_parameters |     [\[5\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     member)]                      | lar_operator15scalar_operatorEd), |
| (api/languages/cpp_api.html#_CPPv |     [\[6\]](api/languag           |
| 4N5cudaq6pauli114num_parametersE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::pauli1::num_targets   | alar_operator15scalar_operatorEv) |
|     (C++                          | -   [                             |
|     membe                         | cudaq::scalar_operator::to_matrix |
| r)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq6pauli111num_targetsE) |                                   |
| -   [cudaq::pauli1::pauli1 (C++   |   function)](api/languages/cpp_ap |
|     function)](api/languages/cpp_ | i.html#_CPPv4NK5cudaq15scalar_ope |
| api.html#_CPPv4N5cudaq6pauli16pau | rator9to_matrixERKNSt13unordered_ |
| li1ERKNSt6vectorIN5cudaq4realEEE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::pauli2 (C++           | -   [                             |
|     class)](api/languages/cp      | cudaq::scalar_operator::to_string |
| p_api.html#_CPPv4N5cudaq6pauli2E) |     (C++                          |
| -                                 |     function)](api/l              |
|    [cudaq::pauli2::num_parameters | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq15scalar_operator9to_stringEv) |
|     member)]                      | -   [cudaq::s                     |
| (api/languages/cpp_api.html#_CPPv | calar_operator::\~scalar_operator |
| 4N5cudaq6pauli214num_parametersE) |     (C++                          |
| -   [cudaq::pauli2::num_targets   |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     membe                         | PPv4N5cudaq15scalar_operatorD0Ev) |
| r)](api/languages/cpp_api.html#_C | -   [cudaq::set_noise (C++        |
| PPv4N5cudaq6pauli211num_targetsE) |     function)](api/langu          |
| -   [cudaq::pauli2::pauli2 (C++   | ages/cpp_api.html#_CPPv4N5cudaq9s |
|     function)](api/languages/cpp_ | et_noiseERKN5cudaq11noise_modelE) |
| api.html#_CPPv4N5cudaq6pauli26pau | -   [cudaq::set_random_seed (C++  |
| li2ERKNSt6vectorIN5cudaq4realEEE) |     function)](api/               |
| -   [cudaq::phase_damping (C++    | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq15set_random_seedENSt6size_tE) |
|  class)](api/languages/cpp_api.ht | -   [cudaq::simulation_precision  |
| ml#_CPPv4N5cudaq13phase_dampingE) |     (C++                          |
| -   [cud                          |     enum)                         |
| aq::phase_damping::num_parameters | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq20simulation_precisionE) |
|     member)](api/lan              | -   [                             |
| guages/cpp_api.html#_CPPv4N5cudaq | cudaq::simulation_precision::fp32 |
| 13phase_damping14num_parametersE) |     (C++                          |
| -   [                             |     enumerator)](api              |
| cudaq::phase_damping::num_targets | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq20simulation_precision4fp32E) |
|     member)](api/                 | -   [                             |
| languages/cpp_api.html#_CPPv4N5cu | cudaq::simulation_precision::fp64 |
| daq13phase_damping11num_targetsE) |     (C++                          |
| -   [cudaq::phase_flip_channel    |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     clas                          | udaq20simulation_precision4fp64E) |
| s)](api/languages/cpp_api.html#_C | -   [cudaq::SimulationState (C++  |
| PPv4N5cudaq18phase_flip_channelE) |     c                             |
| -   [cudaq::p                     | lass)](api/languages/cpp_api.html |
| hase_flip_channel::num_parameters | #_CPPv4N5cudaq15SimulationStateE) |
|     (C++                          | -   [                             |
|     member)](api/language         | cudaq::SimulationState::precision |
| s/cpp_api.html#_CPPv4N5cudaq18pha |     (C++                          |
| se_flip_channel14num_parametersE) |     enum)](api                    |
| -   [cudaq                        | /languages/cpp_api.html#_CPPv4N5c |
| ::phase_flip_channel::num_targets | udaq15SimulationState9precisionE) |
|     (C++                          | -   [cudaq:                       |
|     member)](api/langu            | :SimulationState::precision::fp32 |
| ages/cpp_api.html#_CPPv4N5cudaq18 |     (C++                          |
| phase_flip_channel11num_targetsE) |     enumerator)](api/lang         |
| -   [cudaq::product_op (C++       | uages/cpp_api.html#_CPPv4N5cudaq1 |
|                                   | 5SimulationState9precision4fp32E) |
|  class)](api/languages/cpp_api.ht | -   [cudaq:                       |
| ml#_CPPv4I0EN5cudaq10product_opE) | :SimulationState::precision::fp64 |
| -   [cudaq::product_op::begin     |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     functio                       | uages/cpp_api.html#_CPPv4N5cudaq1 |
| n)](api/languages/cpp_api.html#_C | 5SimulationState9precision4fp64E) |
| PPv4NK5cudaq10product_op5beginEv) | -                                 |
| -                                 |   [cudaq::SimulationState::Tensor |
|  [cudaq::product_op::canonicalize |     (C++                          |
|     (C++                          |     struct)](                     |
|     func                          | api/languages/cpp_api.html#_CPPv4 |
| tion)](api/languages/cpp_api.html | N5cudaq15SimulationState6TensorE) |
| #_CPPv4N5cudaq10product_op12canon | -   [cudaq::spin_handler (C++     |
| icalizeERKNSt3setINSt6size_tEEE), |                                   |
|     [\[1\]](api                   |   class)](api/languages/cpp_api.h |
| /languages/cpp_api.html#_CPPv4N5c | tml#_CPPv4N5cudaq12spin_handlerE) |
| udaq10product_op12canonicalizeEv) | -   [cudaq:                       |
| -   [                             | :spin_handler::to_diagonal_matrix |
| cudaq::product_op::const_iterator |     (C++                          |
|     (C++                          |     function)](api/la             |
|     struct)](api/                 | nguages/cpp_api.html#_CPPv4NK5cud |
| languages/cpp_api.html#_CPPv4N5cu | aq12spin_handler18to_diagonal_mat |
| daq10product_op14const_iteratorE) | rixERNSt13unordered_mapINSt6size_ |
| -   [cudaq::product_o             | tENSt7int64_tEEERKNSt13unordered_ |
| p::const_iterator::const_iterator | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -                                 |
|     fu                            |   [cudaq::spin_handler::to_matrix |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq10product_op14con |     function                      |
| st_iterator14const_iteratorEPK10p | )](api/languages/cpp_api.html#_CP |
| roduct_opI9HandlerTyENSt6size_tE) | Pv4N5cudaq12spin_handler9to_matri |
| -   [cudaq::produ                 | xERKNSt6stringENSt7complexIdEEb), |
| ct_op::const_iterator::operator!= |     [\[1                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     fun                           | PPv4NK5cudaq12spin_handler9to_mat |
| ction)](api/languages/cpp_api.htm | rixERNSt13unordered_mapINSt6size_ |
| l#_CPPv4NK5cudaq10product_op14con | tENSt7int64_tEEERKNSt13unordered_ |
| st_iteratorneERK14const_iterator) | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [cuda                         |
|                                   | q::spin_handler::to_sparse_matrix |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq12spin_handler16to_sparse_matr |
|                                   | ixERKNSt6stringENSt7complexIdEEb) |
|                                   | -                                 |
|                                   |   [cudaq::spin_handler::to_string |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq12spin_handler9to_stringEb) |
|                                   | -                                 |
|                                   |   [cudaq::spin_handler::unique_id |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq12spin_handler9unique_idEv) |
|                                   | -   [cudaq::spin_op (C++          |
|                                   |     type)](api/languages/cpp      |
|                                   | _api.html#_CPPv4N5cudaq7spin_opE) |
|                                   | -   [cudaq::spin_op_term (C++     |
|                                   |                                   |
|                                   |    type)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq12spin_op_termE) |
|                                   | -   [cudaq::state (C++            |
|                                   |     class)](api/languages/c       |
|                                   | pp_api.html#_CPPv4N5cudaq5stateE) |
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
| -   [define() (cudaq.operators    | -   [deserialize                  |
|     method)](api/languages/python |     (cudaq.SampleResult           |
| _api.html#cudaq.operators.define) |     attribu                       |
|     -   [(cuda                    | te)](api/languages/python_api.htm |
| q.operators.MatrixOperatorElement | l#cudaq.SampleResult.deserialize) |
|         class                     | -   [detector() (in module        |
|         method)](api/langu        |     cudaq)](api/language          |
| ages/python_api.html#cudaq.operat | s/python_api.html#cudaq.detector) |
| ors.MatrixOperatorElement.define) | -   [detectors() (in module       |
|     -   [(in module               |     cudaq)](api/languages         |
|         cudaq.operators.cus       | /python_api.html#cudaq.detectors) |
| tom)](api/languages/python_api.ht | -   [distribute_terms             |
| ml#cudaq.operators.custom.define) |     (cu                           |
| -   [degrees                      | daq.operators.boson.BosonOperator |
|     (cu                           |     attribute)](api/languages/pyt |
| daq.operators.boson.BosonOperator | hon_api.html#cudaq.operators.boso |
|     property)](api/lang           | n.BosonOperator.distribute_terms) |
| uages/python_api.html#cudaq.opera |     -   [(cudaq.                  |
| tors.boson.BosonOperator.degrees) | operators.fermion.FermionOperator |
|     -   [(cudaq.ope               |                                   |
| rators.boson.BosonOperatorElement | attribute)](api/languages/python_ |
|                                   | api.html#cudaq.operators.fermion. |
|        property)](api/languages/p | FermionOperator.distribute_terms) |
| ython_api.html#cudaq.operators.bo |     -                             |
| son.BosonOperatorElement.degrees) |  [(cudaq.operators.MatrixOperator |
|     -   [(cudaq.                  |         attribute)](api/language  |
| operators.boson.BosonOperatorTerm | s/python_api.html#cudaq.operators |
|         property)](api/language   | .MatrixOperator.distribute_terms) |
| s/python_api.html#cudaq.operators |     -   [(                        |
| .boson.BosonOperatorTerm.degrees) | cudaq.operators.spin.SpinOperator |
|     -   [(cudaq.                  |                                   |
| operators.fermion.FermionOperator |       attribute)](api/languages/p |
|         property)](api/language   | ython_api.html#cudaq.operators.sp |
| s/python_api.html#cudaq.operators | in.SpinOperator.distribute_terms) |
| .fermion.FermionOperator.degrees) |     -   [(cuda                    |
|     -   [(cudaq.operato           | q.operators.spin.SpinOperatorTerm |
| rs.fermion.FermionOperatorElement |                                   |
|                                   |   attribute)](api/languages/pytho |
|    property)](api/languages/pytho | n_api.html#cudaq.operators.spin.S |
| n_api.html#cudaq.operators.fermio | pinOperatorTerm.distribute_terms) |
| n.FermionOperatorElement.degrees) | -   [draw() (in module            |
|     -   [(cudaq.oper              |     cudaq)](api/lang              |
| ators.fermion.FermionOperatorTerm | uages/python_api.html#cudaq.draw) |
|                                   | -   [dump (cudaq.ComplexMatrix    |
|       property)](api/languages/py |     a                             |
| thon_api.html#cudaq.operators.fer | ttribute)](api/languages/python_a |
| mion.FermionOperatorTerm.degrees) | pi.html#cudaq.ComplexMatrix.dump) |
|     -                             |     -   [(cudaq.ObserveResult     |
|  [(cudaq.operators.MatrixOperator |         a                         |
|         property)](api            | ttribute)](api/languages/python_a |
| /languages/python_api.html#cudaq. | pi.html#cudaq.ObserveResult.dump) |
| operators.MatrixOperator.degrees) |     -   [(cu                      |
|     -   [(cuda                    | daq.operators.boson.BosonOperator |
| q.operators.MatrixOperatorElement |         attribute)](api/l         |
|         property)](api/langua     | anguages/python_api.html#cudaq.op |
| ges/python_api.html#cudaq.operato | erators.boson.BosonOperator.dump) |
| rs.MatrixOperatorElement.degrees) |     -   [(cudaq.                  |
|     -   [(c                       | operators.boson.BosonOperatorTerm |
| udaq.operators.MatrixOperatorTerm |         attribute)](api/langu     |
|         property)](api/lan        | ages/python_api.html#cudaq.operat |
| guages/python_api.html#cudaq.oper | ors.boson.BosonOperatorTerm.dump) |
| ators.MatrixOperatorTerm.degrees) |     -   [(cudaq.                  |
|     -   [(                        | operators.fermion.FermionOperator |
| cudaq.operators.spin.SpinOperator |         attribute)](api/langu     |
|         property)](api/la         | ages/python_api.html#cudaq.operat |
| nguages/python_api.html#cudaq.ope | ors.fermion.FermionOperator.dump) |
| rators.spin.SpinOperator.degrees) |     -   [(cudaq.oper              |
|     -   [(cudaq.o                 | ators.fermion.FermionOperatorTerm |
| perators.spin.SpinOperatorElement |         attribute)](api/languages |
|         property)](api/languages  | /python_api.html#cudaq.operators. |
| /python_api.html#cudaq.operators. | fermion.FermionOperatorTerm.dump) |
| spin.SpinOperatorElement.degrees) |     -                             |
|     -   [(cuda                    |  [(cudaq.operators.MatrixOperator |
| q.operators.spin.SpinOperatorTerm |         attribute)](              |
|         property)](api/langua     | api/languages/python_api.html#cud |
| ges/python_api.html#cudaq.operato | aq.operators.MatrixOperator.dump) |
| rs.spin.SpinOperatorTerm.degrees) |     -   [(c                       |
| -   [dem_from_kernel() (in module | udaq.operators.MatrixOperatorTerm |
|     cudaq)](api/languages/pytho   |         attribute)](api/          |
| n_api.html#cudaq.dem_from_kernel) | languages/python_api.html#cudaq.o |
| -   [Depolarization1 (class in    | perators.MatrixOperatorTerm.dump) |
|     cudaq)](api/languages/pytho   |     -   [(                        |
| n_api.html#cudaq.Depolarization1) | cudaq.operators.spin.SpinOperator |
| -   [Depolarization2 (class in    |         attribute)](api           |
|     cudaq)](api/languages/pytho   | /languages/python_api.html#cudaq. |
| n_api.html#cudaq.Depolarization2) | operators.spin.SpinOperator.dump) |
| -   [DepolarizationChannel (class |     -   [(cuda                    |
|     in                            | q.operators.spin.SpinOperatorTerm |
|                                   |         attribute)](api/lan       |
| cudaq)](api/languages/python_api. | guages/python_api.html#cudaq.oper |
| html#cudaq.DepolarizationChannel) | ators.spin.SpinOperatorTerm.dump) |
| -   [depth (cudaq.Resources       |     -   [(cudaq.Resources         |
|                                   |                                   |
|    property)](api/languages/pytho |    attribute)](api/languages/pyth |
| n_api.html#cudaq.Resources.depth) | on_api.html#cudaq.Resources.dump) |
| -   [depth_for_arity              |     -   [(cudaq.SampleResult      |
|     (cudaq.Resources              |                                   |
|     attribut                      | attribute)](api/languages/python_ |
| e)](api/languages/python_api.html | api.html#cudaq.SampleResult.dump) |
| #cudaq.Resources.depth_for_arity) |     -   [(cudaq.State             |
| -   [description (cudaq.Target    |                                   |
|                                   |        attribute)](api/languages/ |
| property)](api/languages/python_a | python_api.html#cudaq.State.dump) |
| pi.html#cudaq.Target.description) |                                   |
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
| -   [empty_op                     | i/languages/python_api.html#cudaq |
|     (                             | .EvolveResult.expectation_values) |
| cudaq.operators.spin.SpinOperator | -   [expectation_z                |
|     attribute)](api/lan           |     (cudaq.SampleResult           |
| guages/python_api.html#cudaq.oper |     attribute                     |
| ators.spin.SpinOperator.empty_op) | )](api/languages/python_api.html# |
| -   [enable_return_to_log()       | cudaq.SampleResult.expectation_z) |
|     (cudaq.PyKernelDecorator      | -   [expected_dimensions          |
|     method)](api/langu            |     (cuda                         |
| ages/python_api.html#cudaq.PyKern | q.operators.MatrixOperatorElement |
| elDecorator.enable_return_to_log) |                                   |
| -   [epsilon                      | property)](api/languages/python_a |
|     (cudaq.optimizers.Adam        | pi.html#cudaq.operators.MatrixOpe |
|     prope                         | ratorElement.expected_dimensions) |
| rty)](api/languages/python_api.ht |                                   |
| ml#cudaq.optimizers.Adam.epsilon) |                                   |
| -   [estimate_resources() (in     |                                   |
|     module                        |                                   |
|                                   |                                   |
|    cudaq)](api/languages/python_a |                                   |
| pi.html#cudaq.estimate_resources) |                                   |
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
| -   [f_tol (cudaq.optimizers.Adam | -   [for_each_pauli               |
|     pro                           |     (                             |
| perty)](api/languages/python_api. | cudaq.operators.spin.SpinOperator |
| html#cudaq.optimizers.Adam.f_tol) |     attribute)](api/languages     |
|     -   [(cudaq.optimizers.SGD    | /python_api.html#cudaq.operators. |
|         pr                        | spin.SpinOperator.for_each_pauli) |
| operty)](api/languages/python_api |     -   [(cuda                    |
| .html#cudaq.optimizers.SGD.f_tol) | q.operators.spin.SpinOperatorTerm |
| -   [FermionOperator (class in    |                                   |
|                                   |     attribute)](api/languages/pyt |
|    cudaq.operators.fermion)](api/ | hon_api.html#cudaq.operators.spin |
| languages/python_api.html#cudaq.o | .SpinOperatorTerm.for_each_pauli) |
| perators.fermion.FermionOperator) | -   [for_each_term                |
| -   [FermionOperatorElement       |     (                             |
|     (class in                     | cudaq.operators.spin.SpinOperator |
|     cuda                          |     attribute)](api/language      |
| q.operators.fermion)](api/languag | s/python_api.html#cudaq.operators |
| es/python_api.html#cudaq.operator | .spin.SpinOperator.for_each_term) |
| s.fermion.FermionOperatorElement) | -   [ForwardDifference (class in  |
| -   [FermionOperatorTerm (class   |     cudaq.gradients)              |
|     in                            | ](api/languages/python_api.html#c |
|     c                             | udaq.gradients.ForwardDifference) |
| udaq.operators.fermion)](api/lang | -   [from_data (cudaq.State       |
| uages/python_api.html#cudaq.opera |                                   |
| tors.fermion.FermionOperatorTerm) |   attribute)](api/languages/pytho |
| -   [final_expectation_values     | n_api.html#cudaq.State.from_data) |
|     (cudaq.EvolveResult           | -   [from_json                    |
|     attribute)](api/lang          |     (                             |
| uages/python_api.html#cudaq.Evolv | cudaq.operators.spin.SpinOperator |
| eResult.final_expectation_values) |     attribute)](api/lang          |
| -   [final_state                  | uages/python_api.html#cudaq.opera |
|     (cudaq.EvolveResult           | tors.spin.SpinOperator.from_json) |
|     attribu                       |     -   [(cuda                    |
| te)](api/languages/python_api.htm | q.operators.spin.SpinOperatorTerm |
| l#cudaq.EvolveResult.final_state) |         attribute)](api/language  |
| -   [finalize() (in module        | s/python_api.html#cudaq.operators |
|     cudaq.mpi)](api/languages/py  | .spin.SpinOperatorTerm.from_json) |
| thon_api.html#cudaq.mpi.finalize) | -   [from_json()                  |
|                                   |     (cudaq.PyKernelDecorator      |
|                                   |     static                        |
|                                   |     method)                       |
|                                   | ](api/languages/python_api.html#c |
|                                   | udaq.PyKernelDecorator.from_json) |
|                                   | -   [from_word                    |
|                                   |     (                             |
|                                   | cudaq.operators.spin.SpinOperator |
|                                   |     attribute)](api/lang          |
|                                   | uages/python_api.html#cudaq.opera |
|                                   | tors.spin.SpinOperator.from_word) |
+-----------------------------------+-----------------------------------+

## G {#G}

+-----------------------------------+-----------------------------------+
| -   [gamma (cudaq.optimizers.SPSA | -   [get_raw_data                 |
|     pro                           |     (                             |
| perty)](api/languages/python_api. | cudaq.operators.spin.SpinOperator |
| html#cudaq.optimizers.SPSA.gamma) |     attribute)](api/languag       |
| -   [gate_count_by_arity          | es/python_api.html#cudaq.operator |
|     (cudaq.Resources              | s.spin.SpinOperator.get_raw_data) |
|     property)](                   |     -   [(cuda                    |
| api/languages/python_api.html#cud | q.operators.spin.SpinOperatorTerm |
| aq.Resources.gate_count_by_arity) |                                   |
| -   [gate_count_for_arity         |       attribute)](api/languages/p |
|     (cudaq.Resources              | ython_api.html#cudaq.operators.sp |
|     attribute)](a                 | in.SpinOperatorTerm.get_raw_data) |
| pi/languages/python_api.html#cuda | -   [get_register_counts          |
| q.Resources.gate_count_for_arity) |     (cudaq.SampleResult           |
| -   [get (cudaq.AsyncEvolveResult |     attribute)](api               |
|     attr                          | /languages/python_api.html#cudaq. |
| ibute)](api/languages/python_api. | SampleResult.get_register_counts) |
| html#cudaq.AsyncEvolveResult.get) | -   [get_sequential_data          |
|                                   |     (cudaq.SampleResult           |
|    -   [(cudaq.AsyncObserveResult |     attribute)](api               |
|         attri                     | /languages/python_api.html#cudaq. |
| bute)](api/languages/python_api.h | SampleResult.get_sequential_data) |
| tml#cudaq.AsyncObserveResult.get) | -   [get_spin                     |
|     -   [(cudaq.AsyncStateResult  |     (cudaq.ObserveResult          |
|         att                       |     attri                         |
| ribute)](api/languages/python_api | bute)](api/languages/python_api.h |
| .html#cudaq.AsyncStateResult.get) | tml#cudaq.ObserveResult.get_spin) |
| -   [get_binary_symplectic_form   | -   [get_state() (in module       |
|     (cuda                         |     cudaq)](api/languages         |
| q.operators.spin.SpinOperatorTerm | /python_api.html#cudaq.get_state) |
|     attribut                      | -   [get_state_async() (in module |
| e)](api/languages/python_api.html |     cudaq)](api/languages/pytho   |
| #cudaq.operators.spin.SpinOperato | n_api.html#cudaq.get_state_async) |
| rTerm.get_binary_symplectic_form) | -   [get_state_refval             |
| -   [get_channels                 |     (cudaq.State                  |
|     (cudaq.NoiseModel             |     attri                         |
|     attrib                        | bute)](api/languages/python_api.h |
| ute)](api/languages/python_api.ht | tml#cudaq.State.get_state_refval) |
| ml#cudaq.NoiseModel.get_channels) | -   [get_target() (in module      |
| -   [get_coefficient              |     cudaq)](api/languages/        |
|     (                             | python_api.html#cudaq.get_target) |
| cudaq.operators.spin.SpinOperator | -   [get_targets() (in module     |
|     attribute)](api/languages/    |     cudaq)](api/languages/p       |
| python_api.html#cudaq.operators.s | ython_api.html#cudaq.get_targets) |
| pin.SpinOperator.get_coefficient) | -   [get_term_count               |
|     -   [(cuda                    |     (                             |
| q.operators.spin.SpinOperatorTerm | cudaq.operators.spin.SpinOperator |
|                                   |     attribute)](api/languages     |
|    attribute)](api/languages/pyth | /python_api.html#cudaq.operators. |
| on_api.html#cudaq.operators.spin. | spin.SpinOperator.get_term_count) |
| SpinOperatorTerm.get_coefficient) | -   [get_total_shots              |
| -   [get_marginal_counts          |     (cudaq.SampleResult           |
|     (cudaq.SampleResult           |     attribute)]                   |
|     attribute)](api               | (api/languages/python_api.html#cu |
| /languages/python_api.html#cudaq. | daq.SampleResult.get_total_shots) |
| SampleResult.get_marginal_counts) | -   [get_trajectory               |
| -   [get_ops (cudaq.KrausChannel  |                                   |
|     att                           |   (cudaq.ptsbe.PTSBEExecutionData |
| ribute)](api/languages/python_api |     attribute)](api/langua        |
| .html#cudaq.KrausChannel.get_ops) | ges/python_api.html#cudaq.ptsbe.P |
| -   [get_pauli_word               | TSBEExecutionData.get_trajectory) |
|     (cuda                         | -   [getTensor (cudaq.State       |
| q.operators.spin.SpinOperatorTerm |                                   |
|     attribute)](api/languages/pyt |   attribute)](api/languages/pytho |
| hon_api.html#cudaq.operators.spin | n_api.html#cudaq.State.getTensor) |
| .SpinOperatorTerm.get_pauli_word) | -   [getTensors (cudaq.State      |
| -   [get_precision (cudaq.Target  |                                   |
|     att                           |  attribute)](api/languages/python |
| ribute)](api/languages/python_api | _api.html#cudaq.State.getTensors) |
| .html#cudaq.Target.get_precision) | -   [gradient (class in           |
| -   [get_qubit_count              |     cudaq.g                       |
|     (                             | radients)](api/languages/python_a |
| cudaq.operators.spin.SpinOperator | pi.html#cudaq.gradients.gradient) |
|     attribute)](api/languages/    | -   [GradientDescent (class in    |
| python_api.html#cudaq.operators.s |     cudaq.optimizers              |
| pin.SpinOperator.get_qubit_count) | )](api/languages/python_api.html# |
|     -   [(cuda                    | cudaq.optimizers.GradientDescent) |
| q.operators.spin.SpinOperatorTerm |                                   |
|                                   |                                   |
|    attribute)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.spin. |                                   |
| SpinOperatorTerm.get_qubit_count) |                                   |
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
| -   [initialize() (in module      |     -   [(                        |
|                                   | cudaq.operators.spin.SpinOperator |
|    cudaq.mpi)](api/languages/pyth |         attribute)](api/langua    |
| on_api.html#cudaq.mpi.initialize) | ges/python_api.html#cudaq.operato |
| -   [initialize_cudaq() (in       | rs.spin.SpinOperator.is_identity) |
|     module                        |     -   [(cuda                    |
|     cudaq)](api/languages/python  | q.operators.spin.SpinOperatorTerm |
| _api.html#cudaq.initialize_cudaq) |                                   |
| -   [InitialState (in module      |        attribute)](api/languages/ |
|     cudaq.dynamics.helpers)](     | python_api.html#cudaq.operators.s |
| api/languages/python_api.html#cud | pin.SpinOperatorTerm.is_identity) |
| aq.dynamics.helpers.InitialState) | -   [is_initialized() (in module  |
| -   [InitialStateType (class in   |     c                             |
|     cudaq)](api/languages/python  | udaq.mpi)](api/languages/python_a |
| _api.html#cudaq.InitialStateType) | pi.html#cudaq.mpi.is_initialized) |
|                                   | -   [is_on_gpu (cudaq.State       |
|                                   |                                   |
|                                   |   attribute)](api/languages/pytho |
|                                   | n_api.html#cudaq.State.is_on_gpu) |
|                                   | -   [is_remote (cudaq.Target      |
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
| -   [make_kernel() (in module     | -   [measurement_counts           |
|     cudaq)](api/languages/p       |     (cudaq.ptsbe.KrausTrajectory  |
| ython_api.html#cudaq.make_kernel) |     property)](api/languag        |
| -   [MatrixOperator (class in     | es/python_api.html#cudaq.ptsbe.Kr |
|     cudaq.operato                 | ausTrajectory.measurement_counts) |
| rs)](api/languages/python_api.htm | -   [merge_kernel()               |
| l#cudaq.operators.MatrixOperator) |     (cudaq.PyKernelDecorator      |
| -   [MatrixOperatorElement (class |     method)](a                    |
|     in                            | pi/languages/python_api.html#cuda |
|     cudaq.operators)](ap          | q.PyKernelDecorator.merge_kernel) |
| i/languages/python_api.html#cudaq | -   [merge_quake_source()         |
| .operators.MatrixOperatorElement) |     (cudaq.PyKernelDecorator      |
| -   [MatrixOperatorTerm (class in |     method)](api/lan              |
|     cudaq.operators)]             | guages/python_api.html#cudaq.PyKe |
| (api/languages/python_api.html#cu | rnelDecorator.merge_quake_source) |
| daq.operators.MatrixOperatorTerm) | -   [min_degree                   |
| -   [max_degree                   |     (cu                           |
|     (cu                           | daq.operators.boson.BosonOperator |
| daq.operators.boson.BosonOperator |     property)](api/languag        |
|     property)](api/languag        | es/python_api.html#cudaq.operator |
| es/python_api.html#cudaq.operator | s.boson.BosonOperator.min_degree) |
| s.boson.BosonOperator.max_degree) |     -   [(cudaq.                  |
|     -   [(cudaq.                  | operators.boson.BosonOperatorTerm |
| operators.boson.BosonOperatorTerm |                                   |
|                                   |        property)](api/languages/p |
|        property)](api/languages/p | ython_api.html#cudaq.operators.bo |
| ython_api.html#cudaq.operators.bo | son.BosonOperatorTerm.min_degree) |
| son.BosonOperatorTerm.max_degree) |     -   [(cudaq.                  |
|     -   [(cudaq.                  | operators.fermion.FermionOperator |
| operators.fermion.FermionOperator |                                   |
|                                   |        property)](api/languages/p |
|        property)](api/languages/p | ython_api.html#cudaq.operators.fe |
| ython_api.html#cudaq.operators.fe | rmion.FermionOperator.min_degree) |
| rmion.FermionOperator.max_degree) |     -   [(cudaq.oper              |
|     -   [(cudaq.oper              | ators.fermion.FermionOperatorTerm |
| ators.fermion.FermionOperatorTerm |                                   |
|                                   |    property)](api/languages/pytho |
|    property)](api/languages/pytho | n_api.html#cudaq.operators.fermio |
| n_api.html#cudaq.operators.fermio | n.FermionOperatorTerm.min_degree) |
| n.FermionOperatorTerm.max_degree) |     -                             |
|     -                             |  [(cudaq.operators.MatrixOperator |
|  [(cudaq.operators.MatrixOperator |         property)](api/la         |
|         property)](api/la         | nguages/python_api.html#cudaq.ope |
| nguages/python_api.html#cudaq.ope | rators.MatrixOperator.min_degree) |
| rators.MatrixOperator.max_degree) |     -   [(c                       |
|     -   [(c                       | udaq.operators.MatrixOperatorTerm |
| udaq.operators.MatrixOperatorTerm |         property)](api/langua     |
|         property)](api/langua     | ges/python_api.html#cudaq.operato |
| ges/python_api.html#cudaq.operato | rs.MatrixOperatorTerm.min_degree) |
| rs.MatrixOperatorTerm.max_degree) |     -   [(                        |
|     -   [(                        | cudaq.operators.spin.SpinOperator |
| cudaq.operators.spin.SpinOperator |         property)](api/langu      |
|         property)](api/langu      | ages/python_api.html#cudaq.operat |
| ages/python_api.html#cudaq.operat | ors.spin.SpinOperator.min_degree) |
| ors.spin.SpinOperator.max_degree) |     -   [(cuda                    |
|     -   [(cuda                    | q.operators.spin.SpinOperatorTerm |
| q.operators.spin.SpinOperatorTerm |         property)](api/languages  |
|         property)](api/languages  | /python_api.html#cudaq.operators. |
| /python_api.html#cudaq.operators. | spin.SpinOperatorTerm.min_degree) |
| spin.SpinOperatorTerm.max_degree) | -   [minimal_eigenvalue           |
| -   [max_iterations               |     (cudaq.ComplexMatrix          |
|     (cudaq.optimizers.Adam        |     attribute)](api               |
|     property)](a                  | /languages/python_api.html#cudaq. |
| pi/languages/python_api.html#cuda | ComplexMatrix.minimal_eigenvalue) |
| q.optimizers.Adam.max_iterations) | -   module                        |
|     -   [(cudaq.optimizers.COBYLA |     -   [cudaq](api/langua        |
|         property)](api            | ges/python_api.html#module-cudaq) |
| /languages/python_api.html#cudaq. |     -                             |
| optimizers.COBYLA.max_iterations) |    [cudaq.boson](api/languages/py |
|     -   [                         | thon_api.html#module-cudaq.boson) |
| (cudaq.optimizers.GradientDescent |     -   [                         |
|         property)](api/language   | cudaq.fermion](api/languages/pyth |
| s/python_api.html#cudaq.optimizer | on_api.html#module-cudaq.fermion) |
| s.GradientDescent.max_iterations) |     -   [cudaq.operators.cu       |
|     -   [(cudaq.optimizers.LBFGS  | stom](api/languages/python_api.ht |
|         property)](ap             | ml#module-cudaq.operators.custom) |
| i/languages/python_api.html#cudaq |                                   |
| .optimizers.LBFGS.max_iterations) |  -   [cudaq.spin](api/languages/p |
|                                   | ython_api.html#module-cudaq.spin) |
| -   [(cudaq.optimizers.NelderMead | -   [most_probable                |
|         property)](api/lan        |     (cudaq.SampleResult           |
| guages/python_api.html#cudaq.opti |     attribute                     |
| mizers.NelderMead.max_iterations) | )](api/languages/python_api.html# |
|     -   [(cudaq.optimizers.SGD    | cudaq.SampleResult.most_probable) |
|         property)](               | -   [multi_qubit_depth            |
| api/languages/python_api.html#cud |     (cudaq.Resources              |
| aq.optimizers.SGD.max_iterations) |     property)                     |
|     -   [(cudaq.optimizers.SPSA   | ](api/languages/python_api.html#c |
|         property)](a              | udaq.Resources.multi_qubit_depth) |
| pi/languages/python_api.html#cuda | -   [multi_qubit_gate_count       |
| q.optimizers.SPSA.max_iterations) |     (cudaq.Resources              |
| -   [mdiag_sparse_matrix (C++     |     property)](api                |
|     type)](api/languages/cpp_api. | /languages/python_api.html#cudaq. |
| html#_CPPv419mdiag_sparse_matrix) | Resources.multi_qubit_gate_count) |
| -   [measure_handle (class in     | -   [multiplicity                 |
|     cudaq)](api/languages/pyth    |     (cudaq.ptsbe.KrausTrajectory  |
| on_api.html#cudaq.measure_handle) |     property)](api/l              |
|                                   | anguages/python_api.html#cudaq.pt |
|                                   | sbe.KrausTrajectory.multiplicity) |
+-----------------------------------+-----------------------------------+

## N {#N}

+-----------------------------------+-----------------------------------+
| -   [name                         | -   [num_qpus (cudaq.Target       |
|                                   |                                   |
|  (cudaq.ptsbe.PTSSamplingStrategy |   attribute)](api/languages/pytho |
|     attribute)](a                 | n_api.html#cudaq.Target.num_qpus) |
| pi/languages/python_api.html#cuda | -   [num_qubits (cudaq.Resources  |
| q.ptsbe.PTSSamplingStrategy.name) |     pr                            |
|     -                             | operty)](api/languages/python_api |
|    [(cudaq.ptsbe.TraceInstruction | .html#cudaq.Resources.num_qubits) |
|         property)                 |     -   [(cudaq.State             |
| ](api/languages/python_api.html#c |                                   |
| udaq.ptsbe.TraceInstruction.name) |  attribute)](api/languages/python |
|     -   [(cudaq.PyKernel          | _api.html#cudaq.State.num_qubits) |
|                                   | -   [num_ranks() (in module       |
|     attribute)](api/languages/pyt |     cudaq.mpi)](api/languages/pyt |
| hon_api.html#cudaq.PyKernel.name) | hon_api.html#cudaq.mpi.num_ranks) |
|     -   [(cudaq.Target            | -   [num_rows                     |
|                                   |     (cudaq.ComplexMatrix          |
|        property)](api/languages/p |     attri                         |
| ython_api.html#cudaq.Target.name) | bute)](api/languages/python_api.h |
| -   [NelderMead (class in         | tml#cudaq.ComplexMatrix.num_rows) |
|     cudaq.optim                   | -   [num_shots                    |
| izers)](api/languages/python_api. |     (cudaq.ptsbe.KrausTrajectory  |
| html#cudaq.optimizers.NelderMead) |     property)](ap                 |
| -   [noise_type                   | i/languages/python_api.html#cudaq |
|     (cudaq.KrausChannel           | .ptsbe.KrausTrajectory.num_shots) |
|     prope                         | -   [num_used_qubits              |
| rty)](api/languages/python_api.ht |     (cudaq.Resources              |
| ml#cudaq.KrausChannel.noise_type) |     propert                       |
| -   [NoiseModel (class in         | y)](api/languages/python_api.html |
|     cudaq)](api/languages/        | #cudaq.Resources.num_used_qubits) |
| python_api.html#cudaq.NoiseModel) | -   [nvqir::MPSSimulationState    |
| -   [num_available_gpus() (in     |     (C++                          |
|     module                        |     class)]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|    cudaq)](api/languages/python_a | 4I0EN5nvqir18MPSSimulationStateE) |
| pi.html#cudaq.num_available_gpus) | -                                 |
| -   [num_columns                  |  [nvqir::TensorNetSimulationState |
|     (cudaq.ComplexMatrix          |     (C++                          |
|     attribut                      |     class)](api/l                 |
| e)](api/languages/python_api.html | anguages/cpp_api.html#_CPPv4I0EN5 |
| #cudaq.ComplexMatrix.num_columns) | nvqir24TensorNetSimulationStateE) |
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
| -   [random                       | -   [Resources (class in          |
|     (                             |     cudaq)](api/languages         |
| cudaq.operators.spin.SpinOperator | /python_api.html#cudaq.Resources) |
|     attribute)](api/l             | -   [right_multiply               |
| anguages/python_api.html#cudaq.op |     (cudaq.SuperOperator          |
| erators.spin.SpinOperator.random) |     attribute)]                   |
| -   [rank() (in module            | (api/languages/python_api.html#cu |
|     cudaq.mpi)](api/language      | daq.SuperOperator.right_multiply) |
| s/python_api.html#cudaq.mpi.rank) | -   [row_count                    |
| -   [register_names               |     (cudaq.KrausOperator          |
|     (cudaq.SampleResult           |     prope                         |
|     property)                     | rty)](api/languages/python_api.ht |
| ](api/languages/python_api.html#c | ml#cudaq.KrausOperator.row_count) |
| udaq.SampleResult.register_names) | -   [run() (in module             |
| -                                 |     cudaq)](api/lan               |
|   [register_set_target_callback() | guages/python_api.html#cudaq.run) |
|     (in module                    | -   [run_async() (in module       |
|     cudaq)]                       |     cudaq)](api/languages         |
| (api/languages/python_api.html#cu | /python_api.html#cudaq.run_async) |
| daq.register_set_target_callback) | -   [RydbergHamiltonian (class in |
| -   [reset_target() (in module    |     cudaq.operators)]             |
|     cudaq)](api/languages/py      | (api/languages/python_api.html#cu |
| thon_api.html#cudaq.reset_target) | daq.operators.RydbergHamiltonian) |
| -   [resolve_captured_arguments() |                                   |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](api/languages/p      |                                   |
| ython_api.html#cudaq.PyKernelDeco |                                   |
| rator.resolve_captured_arguments) |                                   |
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
| -   [ScalarOperator (class in     |   property)](api/languages/python |
|     cudaq.operato                 | _api.html#cudaq.Target.simulator) |
| rs)](api/languages/python_api.htm | -   [slice() (cudaq.QuakeValue    |
| l#cudaq.operators.ScalarOperator) |     method)](api/languages/python |
| -   [Schedule (class in           | _api.html#cudaq.QuakeValue.slice) |
|     cudaq)](api/language          | -   [SpinOperator (class in       |
| s/python_api.html#cudaq.Schedule) |     cudaq.operators.spin)         |
| -   [serialize                    | ](api/languages/python_api.html#c |
|     (                             | udaq.operators.spin.SpinOperator) |
| cudaq.operators.spin.SpinOperator | -   [SpinOperatorElement (class   |
|     attribute)](api/lang          |     in                            |
| uages/python_api.html#cudaq.opera |     cudaq.operators.spin)](api/l  |
| tors.spin.SpinOperator.serialize) | anguages/python_api.html#cudaq.op |
|     -   [(cuda                    | erators.spin.SpinOperatorElement) |
| q.operators.spin.SpinOperatorTerm | -   [SpinOperatorTerm (class in   |
|         attribute)](api/language  |     cudaq.operators.spin)](ap     |
| s/python_api.html#cudaq.operators | i/languages/python_api.html#cudaq |
| .spin.SpinOperatorTerm.serialize) | .operators.spin.SpinOperatorTerm) |
|     -   [(cudaq.SampleResult      | -   [split_communicator() (in     |
|         attri                     |     module                        |
| bute)](api/languages/python_api.h |     cudaq                         |
| tml#cudaq.SampleResult.serialize) | .mpi)](api/languages/python_api.h |
| -   [set_communicator() (in       | tml#cudaq.mpi.split_communicator) |
|     module                        | -   [SPSA (class in               |
|     cud                           |     cudaq                         |
| aq.mpi)](api/languages/python_api | .optimizers)](api/languages/pytho |
| .html#cudaq.mpi.set_communicator) | n_api.html#cudaq.optimizers.SPSA) |
| -   [set_noise() (in module       | -   [State (class in              |
|     cudaq)](api/languages         |     cudaq)](api/langu             |
| /python_api.html#cudaq.set_noise) | ages/python_api.html#cudaq.State) |
| -   [set_random_seed() (in module | -   [step_size                    |
|     cudaq)](api/languages/pytho   |     (cudaq.optimizers.Adam        |
| n_api.html#cudaq.set_random_seed) |     propert                       |
| -   [set_target() (in module      | y)](api/languages/python_api.html |
|     cudaq)](api/languages/        | #cudaq.optimizers.Adam.step_size) |
| python_api.html#cudaq.set_target) |     -   [(cudaq.optimizers.SGD    |
| -   [SGD (class in                |         proper                    |
|     cuda                          | ty)](api/languages/python_api.htm |
| q.optimizers)](api/languages/pyth | l#cudaq.optimizers.SGD.step_size) |
| on_api.html#cudaq.optimizers.SGD) |     -   [(cudaq.optimizers.SPSA   |
|                                   |         propert                   |
|                                   | y)](api/languages/python_api.html |
|                                   | #cudaq.optimizers.SPSA.step_size) |
|                                   | -   [SuperOperator (class in      |
|                                   |     cudaq)](api/languages/pyt     |
|                                   | hon_api.html#cudaq.SuperOperator) |
|                                   | -   [supports_compilation()       |
|                                   |     (cudaq.PyKernelDecorator      |
|                                   |     method)](api/langu            |
|                                   | ages/python_api.html#cudaq.PyKern |
|                                   | elDecorator.supports_compilation) |
+-----------------------------------+-----------------------------------+

## T {#T}

+-----------------------------------+-----------------------------------+
| -   [Target (class in             | -   [to_matrix()                  |
|     cudaq)](api/langua            |                                   |
| ges/python_api.html#cudaq.Target) |   (cudaq.operators.ScalarOperator |
| -   [target                       |     method)](api/l                |
|     (cudaq.ope                    | anguages/python_api.html#cudaq.op |
| rators.boson.BosonOperatorElement | erators.ScalarOperator.to_matrix) |
|     property)](api/languages/     | -   [to_numpy                     |
| python_api.html#cudaq.operators.b |     (cudaq.ComplexMatrix          |
| oson.BosonOperatorElement.target) |     attri                         |
|     -   [(cudaq.operato           | bute)](api/languages/python_api.h |
| rs.fermion.FermionOperatorElement | tml#cudaq.ComplexMatrix.to_numpy) |
|                                   |     -   [(cudaq.State             |
|     property)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.fermi |    attribute)](api/languages/pyth |
| on.FermionOperatorElement.target) | on_api.html#cudaq.State.to_numpy) |
|     -   [(cudaq.o                 | -   [to_sparse_matrix             |
| perators.spin.SpinOperatorElement |     (cu                           |
|         property)](api/language   | daq.operators.boson.BosonOperator |
| s/python_api.html#cudaq.operators |     attribute)](api/languages/pyt |
| .spin.SpinOperatorElement.target) | hon_api.html#cudaq.operators.boso |
| -   [targets                      | n.BosonOperator.to_sparse_matrix) |
|     (cudaq.ptsbe.TraceInstruction |     -   [(cudaq.                  |
|     property)](a                  | operators.boson.BosonOperatorTerm |
| pi/languages/python_api.html#cuda |                                   |
| q.ptsbe.TraceInstruction.targets) | attribute)](api/languages/python_ |
| -   [Tensor (class in             | api.html#cudaq.operators.boson.Bo |
|     cudaq)](api/langua            | sonOperatorTerm.to_sparse_matrix) |
| ges/python_api.html#cudaq.Tensor) |     -   [(cudaq.                  |
| -   [term_count                   | operators.fermion.FermionOperator |
|     (cu                           |                                   |
| daq.operators.boson.BosonOperator | attribute)](api/languages/python_ |
|     property)](api/languag        | api.html#cudaq.operators.fermion. |
| es/python_api.html#cudaq.operator | FermionOperator.to_sparse_matrix) |
| s.boson.BosonOperator.term_count) |     -   [(cudaq.oper              |
|     -   [(cudaq.                  | ators.fermion.FermionOperatorTerm |
| operators.fermion.FermionOperator |         attr                      |
|                                   | ibute)](api/languages/python_api. |
|        property)](api/languages/p | html#cudaq.operators.fermion.Ferm |
| ython_api.html#cudaq.operators.fe | ionOperatorTerm.to_sparse_matrix) |
| rmion.FermionOperator.term_count) |     -   [(                        |
|     -                             | cudaq.operators.spin.SpinOperator |
|  [(cudaq.operators.MatrixOperator |                                   |
|         property)](api/la         |       attribute)](api/languages/p |
| nguages/python_api.html#cudaq.ope | ython_api.html#cudaq.operators.sp |
| rators.MatrixOperator.term_count) | in.SpinOperator.to_sparse_matrix) |
|     -   [(                        |     -   [(cuda                    |
| cudaq.operators.spin.SpinOperator | q.operators.spin.SpinOperatorTerm |
|         property)](api/langu      |                                   |
| ages/python_api.html#cudaq.operat |   attribute)](api/languages/pytho |
| ors.spin.SpinOperator.term_count) | n_api.html#cudaq.operators.spin.S |
|     -   [(cuda                    | pinOperatorTerm.to_sparse_matrix) |
| q.operators.spin.SpinOperatorTerm | -   [to_string                    |
|         property)](api/languages  |     (cudaq.ope                    |
| /python_api.html#cudaq.operators. | rators.boson.BosonOperatorElement |
| spin.SpinOperatorTerm.term_count) |     attribute)](api/languages/pyt |
| -   [term_id                      | hon_api.html#cudaq.operators.boso |
|     (cudaq.                       | n.BosonOperatorElement.to_string) |
| operators.boson.BosonOperatorTerm |     -   [(cudaq.operato           |
|     property)](api/language       | rs.fermion.FermionOperatorElement |
| s/python_api.html#cudaq.operators |                                   |
| .boson.BosonOperatorTerm.term_id) | attribute)](api/languages/python_ |
|     -   [(cudaq.oper              | api.html#cudaq.operators.fermion. |
| ators.fermion.FermionOperatorTerm | FermionOperatorElement.to_string) |
|                                   |     -   [(cuda                    |
|       property)](api/languages/py | q.operators.MatrixOperatorElement |
| thon_api.html#cudaq.operators.fer |         attribute)](api/language  |
| mion.FermionOperatorTerm.term_id) | s/python_api.html#cudaq.operators |
|     -   [(c                       | .MatrixOperatorElement.to_string) |
| udaq.operators.MatrixOperatorTerm |     -   [(                        |
|         property)](api/lan        | cudaq.operators.spin.SpinOperator |
| guages/python_api.html#cudaq.oper |         attribute)](api/lang      |
| ators.MatrixOperatorTerm.term_id) | uages/python_api.html#cudaq.opera |
|     -   [(cuda                    | tors.spin.SpinOperator.to_string) |
| q.operators.spin.SpinOperatorTerm |     -   [(cudaq.o                 |
|         property)](api/langua     | perators.spin.SpinOperatorElement |
| ges/python_api.html#cudaq.operato |                                   |
| rs.spin.SpinOperatorTerm.term_id) |       attribute)](api/languages/p |
| -   [to_bools() (in module        | ython_api.html#cudaq.operators.sp |
|     cudaq)](api/language          | in.SpinOperatorElement.to_string) |
| s/python_api.html#cudaq.to_bools) |     -   [(cuda                    |
| -   [to_dict (cudaq.Resources     | q.operators.spin.SpinOperatorTerm |
|                                   |         attribute)](api/language  |
| attribute)](api/languages/python_ | s/python_api.html#cudaq.operators |
| api.html#cudaq.Resources.to_dict) | .spin.SpinOperatorTerm.to_string) |
| -   [to_json                      | -   [TraceInstruction (class in   |
|     (                             |     cudaq.p                       |
| cudaq.operators.spin.SpinOperator | tsbe)](api/languages/python_api.h |
|     attribute)](api/la            | tml#cudaq.ptsbe.TraceInstruction) |
| nguages/python_api.html#cudaq.ope | -   [TraceInstructionType (class  |
| rators.spin.SpinOperator.to_json) |     in                            |
|     -   [(cuda                    |     cudaq.ptsbe                   |
| q.operators.spin.SpinOperatorTerm | )](api/languages/python_api.html# |
|         attribute)](api/langua    | cudaq.ptsbe.TraceInstructionType) |
| ges/python_api.html#cudaq.operato | -   [trajectories                 |
| rs.spin.SpinOperatorTerm.to_json) |                                   |
| -   [to_json()                    |   (cudaq.ptsbe.PTSBEExecutionData |
|     (cudaq.PyKernelDecorator      |     property)](api/lang           |
|     metho                         | uages/python_api.html#cudaq.ptsbe |
| d)](api/languages/python_api.html | .PTSBEExecutionData.trajectories) |
| #cudaq.PyKernelDecorator.to_json) | -   [trajectory_id                |
| -   [to_matrix                    |     (cudaq.ptsbe.KrausTrajectory  |
|     (cu                           |     property)](api/la             |
| daq.operators.boson.BosonOperator | nguages/python_api.html#cudaq.pts |
|     attribute)](api/langua        | be.KrausTrajectory.trajectory_id) |
| ges/python_api.html#cudaq.operato | -   [translate() (in module       |
| rs.boson.BosonOperator.to_matrix) |     cudaq)](api/languages         |
|     -   [(cudaq.ope               | /python_api.html#cudaq.translate) |
| rators.boson.BosonOperatorElement | -   [trim                         |
|                                   |     (cu                           |
|     attribute)](api/languages/pyt | daq.operators.boson.BosonOperator |
| hon_api.html#cudaq.operators.boso |     attribute)](api/l             |
| n.BosonOperatorElement.to_matrix) | anguages/python_api.html#cudaq.op |
|     -   [(cudaq.                  | erators.boson.BosonOperator.trim) |
| operators.boson.BosonOperatorTerm |     -   [(cudaq.                  |
|                                   | operators.fermion.FermionOperator |
|        attribute)](api/languages/ |         attribute)](api/langu     |
| python_api.html#cudaq.operators.b | ages/python_api.html#cudaq.operat |
| oson.BosonOperatorTerm.to_matrix) | ors.fermion.FermionOperator.trim) |
|     -   [(cudaq.                  |     -                             |
| operators.fermion.FermionOperator |  [(cudaq.operators.MatrixOperator |
|                                   |         attribute)](              |
|        attribute)](api/languages/ | api/languages/python_api.html#cud |
| python_api.html#cudaq.operators.f | aq.operators.MatrixOperator.trim) |
| ermion.FermionOperator.to_matrix) |     -   [(                        |
|     -   [(cudaq.operato           | cudaq.operators.spin.SpinOperator |
| rs.fermion.FermionOperatorElement |         attribute)](api           |
|                                   | /languages/python_api.html#cudaq. |
| attribute)](api/languages/python_ | operators.spin.SpinOperator.trim) |
| api.html#cudaq.operators.fermion. | -   [type                         |
| FermionOperatorElement.to_matrix) |     (c                            |
|     -   [(cudaq.oper              | udaq.ptsbe.ShotAllocationStrategy |
| ators.fermion.FermionOperatorTerm |     property)](api/               |
|                                   | languages/python_api.html#cudaq.p |
|    attribute)](api/languages/pyth | tsbe.ShotAllocationStrategy.type) |
| on_api.html#cudaq.operators.fermi |     -                             |
| on.FermionOperatorTerm.to_matrix) |    [(cudaq.ptsbe.TraceInstruction |
|     -                             |         property)                 |
|  [(cudaq.operators.MatrixOperator | ](api/languages/python_api.html#c |
|         attribute)](api/l         | udaq.ptsbe.TraceInstruction.type) |
| anguages/python_api.html#cudaq.op | -   [type_to_str()                |
| erators.MatrixOperator.to_matrix) |     (cudaq.PyKernelDecorator      |
|     -   [(cuda                    |     static                        |
| q.operators.MatrixOperatorElement |     method)](                     |
|         attribute)](api/language  | api/languages/python_api.html#cud |
| s/python_api.html#cudaq.operators | aq.PyKernelDecorator.type_to_str) |
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
