::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
latest
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
        -   [HSB FPGA IP core and RFSoC
            bit-file](using/realtime/installation.html#hsb-fpga-ip-core-and-rfsoc-bit-file){.reference
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
        -   [[`cudaq.contrib`{.docutils .literal
            .notranslate}]{.pre}](api/languages/python_api.html#cudaq-contrib){.reference
            .internal}
            -   [Quantum
                Embeddings](api/languages/python_api.html#quantum-embeddings){.reference
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
| -   [Adam (class in               | -   [angular_encode() (in module  |
|     cudaq                         |     cudaq.con                     |
| .optimizers)](api/languages/pytho | trib)](api/languages/python_api.h |
| n_api.html#cudaq.optimizers.Adam) | tml#cudaq.contrib.angular_encode) |
| -   [add_all_qubit_channel        | -   [append (cudaq.KrausChannel   |
|     (cudaq.NoiseModel             |     at                            |
|     attribute)](api               | tribute)](api/languages/python_ap |
| /languages/python_api.html#cudaq. | i.html#cudaq.KrausChannel.append) |
| NoiseModel.add_all_qubit_channel) | -   [argument_count               |
| -   [add_channel                  |     (cudaq.PyKernel               |
|     (cudaq.NoiseModel             |     attrib                        |
|     attri                         | ute)](api/languages/python_api.ht |
| bute)](api/languages/python_api.h | ml#cudaq.PyKernel.argument_count) |
| tml#cudaq.NoiseModel.add_channel) | -   [arguments (cudaq.PyKernel    |
| -   [all_gather() (in module      |     a                             |
|                                   | ttribute)](api/languages/python_a |
|    cudaq.mpi)](api/languages/pyth | pi.html#cudaq.PyKernel.arguments) |
| on_api.html#cudaq.mpi.all_gather) | -   [as_pauli                     |
| -   [amplitude (cudaq.State       |     (cudaq.o                      |
|                                   | perators.spin.SpinOperatorElement |
|   attribute)](api/languages/pytho |     attribute)](api/languages/    |
| n_api.html#cudaq.State.amplitude) | python_api.html#cudaq.operators.s |
| -   [amplitude_encode() (in       | pin.SpinOperatorElement.as_pauli) |
|     module                        | -   [AsyncEvolveResult (class in  |
|     cudaq.contr                   |     cudaq)](api/languages/python_ |
| ib)](api/languages/python_api.htm | api.html#cudaq.AsyncEvolveResult) |
| l#cudaq.contrib.amplitude_encode) | -   [AsyncObserveResult (class in |
| -   [AmplitudeDampingChannel      |                                   |
|     (class in                     |    cudaq)](api/languages/python_a |
|     cu                            | pi.html#cudaq.AsyncObserveResult) |
| daq)](api/languages/python_api.ht | -   [AsyncSampleResult (class in  |
| ml#cudaq.AmplitudeDampingChannel) |     cudaq)](api/languages/python_ |
| -   [amplitudes (cudaq.State      | api.html#cudaq.AsyncSampleResult) |
|                                   | -   [AsyncStateResult (class in   |
|  attribute)](api/languages/python |     cudaq)](api/languages/python  |
| _api.html#cudaq.State.amplitudes) | _api.html#cudaq.AsyncStateResult) |
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
|     (cudaq.PyKernelDecorator      | ct_op::const_iterator::operator!= |
|     method)](api/langu            |     (C++                          |
| ages/python_api.html#cudaq.PyKern |     fun                           |
| elDecorator.cachedCompiledModule) | ction)](api/languages/cpp_api.htm |
| -   [canonicalize                 | l#_CPPv4NK5cudaq10product_op14con |
|     (cu                           | st_iteratorneERK14const_iterator) |
| daq.operators.boson.BosonOperator | -   [cudaq::produ                 |
|     attribute)](api/languages     | ct_op::const_iterator::operator\* |
| /python_api.html#cudaq.operators. |     (C++                          |
| boson.BosonOperator.canonicalize) |     function)](api/lang           |
|     -   [(cudaq.                  | uages/cpp_api.html#_CPPv4NK5cudaq |
| operators.boson.BosonOperatorTerm | 10product_op14const_iteratormlEv) |
|                                   | -   [cudaq::produ                 |
|     attribute)](api/languages/pyt | ct_op::const_iterator::operator++ |
| hon_api.html#cudaq.operators.boso |     (C++                          |
| n.BosonOperatorTerm.canonicalize) |     function)](api/lang           |
|     -   [(cudaq.                  | uages/cpp_api.html#_CPPv4N5cudaq1 |
| operators.fermion.FermionOperator | 0product_op14const_iteratorppEi), |
|                                   |     [\[1\]](api/lan               |
|     attribute)](api/languages/pyt | guages/cpp_api.html#_CPPv4N5cudaq |
| hon_api.html#cudaq.operators.ferm | 10product_op14const_iteratorppEv) |
| ion.FermionOperator.canonicalize) | -   [cudaq::produc                |
|     -   [(cudaq.oper              | t_op::const_iterator::operator\-- |
| ators.fermion.FermionOperatorTerm |     (C++                          |
|                                   |     function)](api/lang           |
| attribute)](api/languages/python_ | uages/cpp_api.html#_CPPv4N5cudaq1 |
| api.html#cudaq.operators.fermion. | 0product_op14const_iteratormmEi), |
| FermionOperatorTerm.canonicalize) |     [\[1\]](api/lan               |
|     -                             | guages/cpp_api.html#_CPPv4N5cudaq |
|  [(cudaq.operators.MatrixOperator | 10product_op14const_iteratormmEv) |
|         attribute)](api/lang      | -   [cudaq::produc                |
| uages/python_api.html#cudaq.opera | t_op::const_iterator::operator-\> |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     function)](api/lan            |
| udaq.operators.MatrixOperatorTerm | guages/cpp_api.html#_CPPv4N5cudaq |
|         attribute)](api/language  | 10product_op14const_iteratorptEv) |
| s/python_api.html#cudaq.operators | -   [cudaq::produ                 |
| .MatrixOperatorTerm.canonicalize) | ct_op::const_iterator::operator== |
|     -   [(                        |     (C++                          |
| cudaq.operators.spin.SpinOperator |     fun                           |
|         attribute)](api/languag   | ction)](api/languages/cpp_api.htm |
| es/python_api.html#cudaq.operator | l#_CPPv4NK5cudaq10product_op14con |
| s.spin.SpinOperator.canonicalize) | st_iteratoreqERK14const_iterator) |
|     -   [(cuda                    | -   [cudaq::product_op::degrees   |
| q.operators.spin.SpinOperatorTerm |     (C++                          |
|                                   |     function)                     |
|       attribute)](api/languages/p | ](api/languages/cpp_api.html#_CPP |
| ython_api.html#cudaq.operators.sp | v4NK5cudaq10product_op7degreesEv) |
| in.SpinOperatorTerm.canonicalize) | -   [cudaq::product_op::dump (C++ |
| -   [captured_variables()         |     functi                        |
|     (cudaq.PyKernelDecorator      | on)](api/languages/cpp_api.html#_ |
|     method)](api/lan              | CPPv4NK5cudaq10product_op4dumpEv) |
| guages/python_api.html#cudaq.PyKe | -   [cudaq::product_op::end (C++  |
| rnelDecorator.captured_variables) |     funct                         |
| -   [CentralDifference (class in  | ion)](api/languages/cpp_api.html# |
|     cudaq.gradients)              | _CPPv4NK5cudaq10product_op3endEv) |
| ](api/languages/python_api.html#c | -   [c                            |
| udaq.gradients.CentralDifference) | udaq::product_op::get_coefficient |
| -   [channel                      |     (C++                          |
|     (cudaq.ptsbe.TraceInstruction |     function)](api/lan            |
|     property)](a                  | guages/cpp_api.html#_CPPv4NK5cuda |
| pi/languages/python_api.html#cuda | q10product_op15get_coefficientEv) |
| q.ptsbe.TraceInstruction.channel) | -                                 |
| -   [circuit_location             |   [cudaq::product_op::get_term_id |
|     (cudaq.ptsbe.KrausSelection   |     (C++                          |
|     property)](api/lang           |     function)](api                |
| uages/python_api.html#cudaq.ptsbe | /languages/cpp_api.html#_CPPv4NK5 |
| .KrausSelection.circuit_location) | cudaq10product_op11get_term_idEv) |
| -   [clear (cudaq.Resources       | -                                 |
|                                   |   [cudaq::product_op::is_identity |
|   attribute)](api/languages/pytho |     (C++                          |
| n_api.html#cudaq.Resources.clear) |     function)](api                |
|     -   [(cudaq.SampleResult      | /languages/cpp_api.html#_CPPv4NK5 |
|         a                         | cudaq10product_op11is_identityEv) |
| ttribute)](api/languages/python_a | -   [cudaq::product_op::num_ops   |
| pi.html#cudaq.SampleResult.clear) |     (C++                          |
| -   [COBYLA (class in             |     function)                     |
|     cudaq.o                       | ](api/languages/cpp_api.html#_CPP |
| ptimizers)](api/languages/python_ | v4NK5cudaq10product_op7num_opsEv) |
| api.html#cudaq.optimizers.COBYLA) | -                                 |
| -   [coefficient                  |    [cudaq::product_op::operator\* |
|     (cudaq.                       |     (C++                          |
| operators.boson.BosonOperatorTerm |     function)](api/languages/     |
|     property)](api/languages/py   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| thon_api.html#cudaq.operators.bos | oduct_opmlE10product_opI1TERK15sc |
| on.BosonOperatorTerm.coefficient) | alar_operatorRK10product_opI1TE), |
|     -   [(cudaq.oper              |     [\[1\]](api/languages/        |
| ators.fermion.FermionOperatorTerm | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|                                   | oduct_opmlE10product_opI1TERK15sc |
|   property)](api/languages/python | alar_operatorRR10product_opI1TE), |
| _api.html#cudaq.operators.fermion |     [\[2\]](api/languages/        |
| .FermionOperatorTerm.coefficient) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(c                       | oduct_opmlE10product_opI1TERR15sc |
| udaq.operators.MatrixOperatorTerm | alar_operatorRK10product_opI1TE), |
|         property)](api/languag    |     [\[3\]](api/languages/        |
| es/python_api.html#cudaq.operator | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| s.MatrixOperatorTerm.coefficient) | oduct_opmlE10product_opI1TERR15sc |
|     -   [(cuda                    | alar_operatorRR10product_opI1TE), |
| q.operators.spin.SpinOperatorTerm |     [\[4\]](api/                  |
|         property)](api/languages/ | languages/cpp_api.html#_CPPv4I0EN |
| python_api.html#cudaq.operators.s | 5cudaq10product_opmlE6sum_opI1TER |
| pin.SpinOperatorTerm.coefficient) | K15scalar_operatorRK6sum_opI1TE), |
| -   [col_count                    |     [\[5\]](api/                  |
|     (cudaq.KrausOperator          | languages/cpp_api.html#_CPPv4I0EN |
|     prope                         | 5cudaq10product_opmlE6sum_opI1TER |
| rty)](api/languages/python_api.ht | K15scalar_operatorRR6sum_opI1TE), |
| ml#cudaq.KrausOperator.col_count) |     [\[6\]](api/                  |
| -   [compile()                    | languages/cpp_api.html#_CPPv4I0EN |
|     (cudaq.PyKernelDecorator      | 5cudaq10product_opmlE6sum_opI1TER |
|     metho                         | R15scalar_operatorRK6sum_opI1TE), |
| d)](api/languages/python_api.html |     [\[7\]](api/                  |
| #cudaq.PyKernelDecorator.compile) | languages/cpp_api.html#_CPPv4I0EN |
| -   [ComplexMatrix (class in      | 5cudaq10product_opmlE6sum_opI1TER |
|     cudaq)](api/languages/pyt     | R15scalar_operatorRR6sum_opI1TE), |
| hon_api.html#cudaq.ComplexMatrix) |     [\[8\]](api/languages         |
| -   [compute                      | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     (                             | duct_opmlERK6sum_opI9HandlerTyE), |
| cudaq.gradients.CentralDifference |     [\[9\]](api/languages/cpp_a   |
|     attribute)](api/la            | pi.html#_CPPv4NKR5cudaq10product_ |
| nguages/python_api.html#cudaq.gra | opmlERK10product_opI9HandlerTyE), |
| dients.CentralDifference.compute) |     [\[10\]](api/language         |
|     -   [(                        | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| cudaq.gradients.ForwardDifference | roduct_opmlERK15scalar_operator), |
|         attribute)](api/la        |     [\[11\]](api/languages/cpp_a  |
| nguages/python_api.html#cudaq.gra | pi.html#_CPPv4NKR5cudaq10product_ |
| dients.ForwardDifference.compute) | opmlERR10product_opI9HandlerTyE), |
|     -                             |     [\[12\]](api/language         |
|  [(cudaq.gradients.ParameterShift | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|         attribute)](api           | roduct_opmlERR15scalar_operator), |
| /languages/python_api.html#cudaq. |     [\[13\]](api/languages/cpp_   |
| gradients.ParameterShift.compute) | api.html#_CPPv4NO5cudaq10product_ |
| -   [const()                      | opmlERK10product_opI9HandlerTyE), |
|                                   |     [\[14\]](api/languag          |
|   (cudaq.operators.ScalarOperator | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     class                         | roduct_opmlERK15scalar_operator), |
|     method)](a                    |     [\[15\]](api/languages/cpp_   |
| pi/languages/python_api.html#cuda | api.html#_CPPv4NO5cudaq10product_ |
| q.operators.ScalarOperator.const) | opmlERR10product_opI9HandlerTyE), |
| -   [controls                     |     [\[16\]](api/langua           |
|     (cudaq.ptsbe.TraceInstruction | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     property)](ap                 | product_opmlERR15scalar_operator) |
| i/languages/python_api.html#cudaq | -                                 |
| .ptsbe.TraceInstruction.controls) |   [cudaq::product_op::operator\*= |
| -   [copy                         |     (C++                          |
|     (cu                           |     function)](api/languages/cpp  |
| daq.operators.boson.BosonOperator | _api.html#_CPPv4N5cudaq10product_ |
|     attribute)](api/l             | opmLERK10product_opI9HandlerTyE), |
| anguages/python_api.html#cudaq.op |     [\[1\]](api/langua            |
| erators.boson.BosonOperator.copy) | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     -   [(cudaq.                  | roduct_opmLERK15scalar_operator), |
| operators.boson.BosonOperatorTerm |     [\[2\]](api/languages/cp      |
|         attribute)](api/langu     | p_api.html#_CPPv4N5cudaq10product |
| ages/python_api.html#cudaq.operat | _opmLERR10product_opI9HandlerTyE) |
| ors.boson.BosonOperatorTerm.copy) | -   [cudaq::product_op::operator+ |
|     -   [(cudaq.                  |     (C++                          |
| operators.fermion.FermionOperator |     function)](api/langu          |
|         attribute)](api/langu     | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ages/python_api.html#cudaq.operat | q10product_opplE6sum_opI1TERK15sc |
| ors.fermion.FermionOperator.copy) | alar_operatorRK10product_opI1TE), |
|     -   [(cudaq.oper              |     [\[1\]](api/                  |
| ators.fermion.FermionOperatorTerm | languages/cpp_api.html#_CPPv4I0EN |
|         attribute)](api/languages | 5cudaq10product_opplE6sum_opI1TER |
| /python_api.html#cudaq.operators. | K15scalar_operatorRK6sum_opI1TE), |
| fermion.FermionOperatorTerm.copy) |     [\[2\]](api/langu             |
|     -                             | ages/cpp_api.html#_CPPv4I0EN5cuda |
|  [(cudaq.operators.MatrixOperator | q10product_opplE6sum_opI1TERK15sc |
|         attribute)](              | alar_operatorRR10product_opI1TE), |
| api/languages/python_api.html#cud |     [\[3\]](api/                  |
| aq.operators.MatrixOperator.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(c                       | 5cudaq10product_opplE6sum_opI1TER |
| udaq.operators.MatrixOperatorTerm | K15scalar_operatorRR6sum_opI1TE), |
|         attribute)](api/          |     [\[4\]](api/langu             |
| languages/python_api.html#cudaq.o | ages/cpp_api.html#_CPPv4I0EN5cuda |
| perators.MatrixOperatorTerm.copy) | q10product_opplE6sum_opI1TERR15sc |
|     -   [(                        | alar_operatorRK10product_opI1TE), |
| cudaq.operators.spin.SpinOperator |     [\[5\]](api/                  |
|         attribute)](api           | languages/cpp_api.html#_CPPv4I0EN |
| /languages/python_api.html#cudaq. | 5cudaq10product_opplE6sum_opI1TER |
| operators.spin.SpinOperator.copy) | R15scalar_operatorRK6sum_opI1TE), |
|     -   [(cuda                    |     [\[6\]](api/langu             |
| q.operators.spin.SpinOperatorTerm | ages/cpp_api.html#_CPPv4I0EN5cuda |
|         attribute)](api/lan       | q10product_opplE6sum_opI1TERR15sc |
| guages/python_api.html#cudaq.oper | alar_operatorRR10product_opI1TE), |
| ators.spin.SpinOperatorTerm.copy) |     [\[7\]](api/                  |
| -   [count (cudaq.Resources       | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opplE6sum_opI1TER |
|   attribute)](api/languages/pytho | R15scalar_operatorRR6sum_opI1TE), |
| n_api.html#cudaq.Resources.count) |     [\[8\]](api/languages/cpp_a   |
|     -   [(cudaq.SampleResult      | pi.html#_CPPv4NKR5cudaq10product_ |
|         a                         | opplERK10product_opI9HandlerTyE), |
| ttribute)](api/languages/python_a |     [\[9\]](api/language          |
| pi.html#cudaq.SampleResult.count) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [count_controls               | roduct_opplERK15scalar_operator), |
|     (cudaq.Resources              |     [\[10\]](api/languages/       |
|     attribu                       | cpp_api.html#_CPPv4NKR5cudaq10pro |
| te)](api/languages/python_api.htm | duct_opplERK6sum_opI9HandlerTyE), |
| l#cudaq.Resources.count_controls) |     [\[11\]](api/languages/cpp_a  |
| -   [count_instructions           | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|   (cudaq.ptsbe.PTSBEExecutionData |     [\[12\]](api/language         |
|     attribute)](api/languages/    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| python_api.html#cudaq.ptsbe.PTSBE | roduct_opplERR15scalar_operator), |
| ExecutionData.count_instructions) |     [\[13\]](api/languages/       |
| -   [counts (cudaq.ObserveResult  | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     att                           | duct_opplERR6sum_opI9HandlerTyE), |
| ribute)](api/languages/python_api |     [\[                           |
| .html#cudaq.ObserveResult.counts) | 14\]](api/languages/cpp_api.html# |
| -   [csr_spmatrix (C++            | _CPPv4NKR5cudaq10product_opplEv), |
|     type)](api/languages/c        |     [\[15\]](api/languages/cpp_   |
| pp_api.html#_CPPv412csr_spmatrix) | api.html#_CPPv4NO5cudaq10product_ |
| -   cudaq                         | opplERK10product_opI9HandlerTyE), |
|     -   [module](api/langua       |     [\[16\]](api/languag          |
| ges/python_api.html#module-cudaq) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq (C++                   | roduct_opplERK15scalar_operator), |
|     type)](api/lan                |     [\[17\]](api/languages        |
| guages/cpp_api.html#_CPPv45cudaq) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq.apply_noise() (in      | duct_opplERK6sum_opI9HandlerTyE), |
|     module                        |     [\[18\]](api/languages/cpp_   |
|     cudaq)](api/languages/python_ | api.html#_CPPv4NO5cudaq10product_ |
| api.html#cudaq.cudaq.apply_noise) | opplERR10product_opI9HandlerTyE), |
| -   cudaq.boson                   |     [\[19\]](api/languag          |
|     -   [module](api/languages/py | es/cpp_api.html#_CPPv4NO5cudaq10p |
| thon_api.html#module-cudaq.boson) | roduct_opplERR15scalar_operator), |
| -   cudaq.fermion                 |     [\[20\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq10pro |
|   -   [module](api/languages/pyth | duct_opplERR6sum_opI9HandlerTyE), |
| on_api.html#module-cudaq.fermion) |     [                             |
| -   cudaq.operators.custom        | \[21\]](api/languages/cpp_api.htm |
|     -   [mo                       | l#_CPPv4NO5cudaq10product_opplEv) |
| dule](api/languages/python_api.ht | -   [cudaq::product_op::operator- |
| ml#module-cudaq.operators.custom) |     (C++                          |
| -   cudaq.spin                    |     function)](api/langu          |
|     -   [module](api/languages/p  | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ython_api.html#module-cudaq.spin) | q10product_opmiE6sum_opI1TERK15sc |
| -   [cudaq::amplitude_damping     | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     cla                           | languages/cpp_api.html#_CPPv4I0EN |
| ss)](api/languages/cpp_api.html#_ | 5cudaq10product_opmiE6sum_opI1TER |
| CPPv4N5cudaq17amplitude_dampingE) | K15scalar_operatorRK6sum_opI1TE), |
| -                                 |     [\[2\]](api/langu             |
| [cudaq::amplitude_damping_channel | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERK15sc |
|     class)](api                   | alar_operatorRR10product_opI1TE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[3\]](api/                  |
| udaq25amplitude_damping_channelE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::amplitud              | 5cudaq10product_opmiE6sum_opI1TER |
| e_damping_channel::num_parameters | K15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[4\]](api/langu             |
|     member)](api/languages/cpp_a  | ages/cpp_api.html#_CPPv4I0EN5cuda |
| pi.html#_CPPv4N5cudaq25amplitude_ | q10product_opmiE6sum_opI1TERR15sc |
| damping_channel14num_parametersE) | alar_operatorRK10product_opI1TE), |
| -   [cudaq::ampli                 |     [\[5\]](api/                  |
| tude_damping_channel::num_targets | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     member)](api/languages/cp     | R15scalar_operatorRK6sum_opI1TE), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [\[6\]](api/langu             |
| de_damping_channel11num_targetsE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::AnalogRemoteRESTQPU   | q10product_opmiE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     class                         |     [\[7\]](api/                  |
| )](api/languages/cpp_api.html#_CP | languages/cpp_api.html#_CPPv4I0EN |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::apply_noise (C++      | R15scalar_operatorRR6sum_opI1TE), |
|     function)](api/               |     [\[8\]](api/languages/cpp_a   |
| languages/cpp_api.html#_CPPv4I0Dp | pi.html#_CPPv4NKR5cudaq10product_ |
| EN5cudaq11apply_noiseEvDpRR4Args) | opmiERK10product_opI9HandlerTyE), |
| -   [cudaq::async_result (C++     |     [\[9\]](api/language          |
|     c                             | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| lass)](api/languages/cpp_api.html | roduct_opmiERK15scalar_operator), |
| #_CPPv4I0EN5cudaq12async_resultE) |     [\[10\]](api/languages/       |
| -   [cudaq::async_result::get     | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     (C++                          | duct_opmiERK6sum_opI9HandlerTyE), |
|     functi                        |     [\[11\]](api/languages/cpp_a  |
| on)](api/languages/cpp_api.html#_ | pi.html#_CPPv4NKR5cudaq10product_ |
| CPPv4N5cudaq12async_result3getEv) | opmiERR10product_opI9HandlerTyE), |
| -   [cudaq::async_sample_result   |     [\[12\]](api/language         |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     type                          | roduct_opmiERR15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[13\]](api/languages/       |
| Pv4N5cudaq19async_sample_resultE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::BaseRemoteRESTQPU     | duct_opmiERR6sum_opI9HandlerTyE), |
|     (C++                          |     [\[                           |
|     cla                           | 14\]](api/languages/cpp_api.html# |
| ss)](api/languages/cpp_api.html#_ | _CPPv4NKR5cudaq10product_opmiEv), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::bit_flip_channel (C++ | api.html#_CPPv4NO5cudaq10product_ |
|     cl                            | opmiERK10product_opI9HandlerTyE), |
| ass)](api/languages/cpp_api.html# |     [\[16\]](api/languag          |
| _CPPv4N5cudaq16bit_flip_channelE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq:                       | roduct_opmiERK15scalar_operator), |
| :bit_flip_channel::num_parameters |     [\[17\]](api/languages        |
|     (C++                          | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     member)](api/langua           | duct_opmiERK6sum_opI9HandlerTyE), |
| ges/cpp_api.html#_CPPv4N5cudaq16b |     [\[18\]](api/languages/cpp_   |
| it_flip_channel14num_parametersE) | api.html#_CPPv4NO5cudaq10product_ |
| -   [cud                          | opmiERR10product_opI9HandlerTyE), |
| aq::bit_flip_channel::num_targets |     [\[19\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     member)](api/lan              | roduct_opmiERR15scalar_operator), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[20\]](api/languages        |
| 16bit_flip_channel11num_targetsE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::boson_handler (C++    | duct_opmiERR6sum_opI9HandlerTyE), |
|                                   |     [                             |
|  class)](api/languages/cpp_api.ht | \[21\]](api/languages/cpp_api.htm |
| ml#_CPPv4N5cudaq13boson_handlerE) | l#_CPPv4NO5cudaq10product_opmiEv) |
| -   [cudaq::boson_op (C++         | -   [cudaq::product_op::operator/ |
|     type)](api/languages/cpp_     |     (C++                          |
| api.html#_CPPv4N5cudaq8boson_opE) |     function)](api/language       |
| -   [cudaq::boson_op_term (C++    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opdvERK15scalar_operator), |
|   type)](api/languages/cpp_api.ht |     [\[1\]](api/language          |
| ml#_CPPv4N5cudaq13boson_op_termE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::CodeGenConfig (C++    | roduct_opdvERR15scalar_operator), |
|                                   |     [\[2\]](api/languag           |
| struct)](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | roduct_opdvERK15scalar_operator), |
| -   [cudaq::commutation_relations |     [\[3\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     struct)]                      | product_opdvERR15scalar_operator) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq21commutation_relationsE) |    [cudaq::product_op::operator/= |
| -   [cudaq::complex (C++          |     (C++                          |
|     type)](api/languages/cpp      |     function)](api/langu          |
| _api.html#_CPPv4N5cudaq7complexE) | ages/cpp_api.html#_CPPv4N5cudaq10 |
| -   [cudaq::complex_matrix (C++   | product_opdVERK15scalar_operator) |
|                                   | -   [cudaq::product_op::operator= |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14complex_matrixE) |     function)](api/l              |
| -                                 | anguages/cpp_api.html#_CPPv4I00EN |
|   [cudaq::complex_matrix::adjoint | 5cudaq10product_opaSER10product_o |
|     (C++                          | pI9HandlerTyERK10product_opI1TE), |
|     function)](a                  |     [\[1\]](api/languages/cpp     |
| pi/languages/cpp_api.html#_CPPv4N | _api.html#_CPPv4N5cudaq10product_ |
| 5cudaq14complex_matrix7adjointEv) | opaSERK10product_opI9HandlerTyE), |
| -   [cudaq::                      |     [\[2\]](api/languages/cp      |
| complex_matrix::diagonal_elements | p_api.html#_CPPv4N5cudaq10product |
|     (C++                          | _opaSERR10product_opI9HandlerTyE) |
|     function)](api/languages      | -                                 |
| /cpp_api.html#_CPPv4NK5cudaq14com |    [cudaq::product_op::operator== |
| plex_matrix17diagonal_elementsEi) |     (C++                          |
| -   [cudaq::complex_matrix::dump  |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq10product |
|     function)](api/language       | _opeqERK10product_opI9HandlerTyE) |
| s/cpp_api.html#_CPPv4NK5cudaq14co | -                                 |
| mplex_matrix4dumpERNSt7ostreamE), |  [cudaq::product_op::operator\[\] |
|     [\[1\]]                       |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](ap                 |
| 4NK5cudaq14complex_matrix4dumpEv) | i/languages/cpp_api.html#_CPPv4NK |
| -   [c                            | 5cudaq10product_opixENSt6size_tE) |
| udaq::complex_matrix::eigenvalues | -                                 |
|     (C++                          |    [cudaq::product_op::product_op |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4NK5cuda |     f                             |
| q14complex_matrix11eigenvaluesEv) | unction)](api/languages/cpp_api.h |
| -   [cu                           | tml#_CPPv4I00EN5cudaq10product_op |
| daq::complex_matrix::eigenvectors | 10product_opERK10product_opI1TE), |
|     (C++                          |     [\[1\]]                       |
|     function)](api/lang           | (api/languages/cpp_api.html#_CPPv |
| uages/cpp_api.html#_CPPv4NK5cudaq | 4I00EN5cudaq10product_op10product |
| 14complex_matrix12eigenvectorsEv) | _opERK10product_opI1TERKN14matrix |
| -   [c                            | _handler20commutation_behaviorE), |
| udaq::complex_matrix::exponential |                                   |
|     (C++                          |   [\[2\]](api/languages/cpp_api.h |
|     function)](api/la             | tml#_CPPv4N5cudaq10product_op10pr |
| nguages/cpp_api.html#_CPPv4N5cuda | oduct_opENSt6size_tENSt6size_tE), |
| q14complex_matrix11exponentialEv) |     [\[3\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4N5cudaq10product |
|  [cudaq::complex_matrix::identity | _op10product_opENSt7complexIdEE), |
|     (C++                          |     [\[4\]](api/l                 |
|     function)](api/languages      | anguages/cpp_api.html#_CPPv4N5cud |
| /cpp_api.html#_CPPv4N5cudaq14comp | aq10product_op10product_opERK10pr |
| lex_matrix8identityEKNSt6size_tE) | oduct_opI9HandlerTyENSt6size_tE), |
| -                                 |     [\[5\]](api/l                 |
| [cudaq::complex_matrix::kronecker | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq10product_op10product_opERR10pr |
|     function)](api/lang           | oduct_opI9HandlerTyENSt6size_tE), |
| uages/cpp_api.html#_CPPv4I00EN5cu |     [\[6\]](api/languages         |
| daq14complex_matrix9kroneckerE14c | /cpp_api.html#_CPPv4N5cudaq10prod |
| omplex_matrix8Iterable8Iterable), | uct_op10product_opERR9HandlerTy), |
|     [\[1\]](api/l                 |     [\[7\]](ap                    |
| anguages/cpp_api.html#_CPPv4N5cud | i/languages/cpp_api.html#_CPPv4N5 |
| aq14complex_matrix9kroneckerERK14 | cudaq10product_op10product_opEd), |
| complex_matrixRK14complex_matrix) |     [\[8\]](a                     |
| -   [cudaq::c                     | pi/languages/cpp_api.html#_CPPv4N |
| omplex_matrix::minimal_eigenvalue | 5cudaq10product_op10product_opEv) |
|     (C++                          | -   [cuda                         |
|     function)](api/languages/     | q::product_op::to_diagonal_matrix |
| cpp_api.html#_CPPv4NK5cudaq14comp |     (C++                          |
| lex_matrix18minimal_eigenvalueEv) |     function)](api/               |
| -   [                             | languages/cpp_api.html#_CPPv4NK5c |
| cudaq::complex_matrix::operator() | udaq10product_op18to_diagonal_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function)](api/languages/cpp  | ENSt7int64_tEEERKNSt13unordered_m |
| _api.html#_CPPv4N5cudaq14complex_ | apINSt6stringENSt7complexIdEEEEb) |
| matrixclENSt6size_tENSt6size_tE), | -   [cudaq::product_op::to_matrix |
|     [\[1\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4NK5cudaq14complex |     funct                         |
| _matrixclENSt6size_tENSt6size_tE) | ion)](api/languages/cpp_api.html# |
| -   [                             | _CPPv4NK5cudaq10product_op9to_mat |
| cudaq::complex_matrix::operator\* | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/langua         | apINSt6stringENSt7complexIdEEEEb) |
| ges/cpp_api.html#_CPPv4N5cudaq14c | -   [cu                           |
| omplex_matrixmlEN14complex_matrix | daq::product_op::to_sparse_matrix |
| 10value_typeERK14complex_matrix), |     (C++                          |
|     [\[1\]                        |     function)](ap                 |
| ](api/languages/cpp_api.html#_CPP | i/languages/cpp_api.html#_CPPv4NK |
| v4N5cudaq14complex_matrixmlERK14c | 5cudaq10product_op16to_sparse_mat |
| omplex_matrixRK14complex_matrix), | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|  [\[2\]](api/languages/cpp_api.ht | apINSt6stringENSt7complexIdEEEEb) |
| ml#_CPPv4N5cudaq14complex_matrixm | -   [cudaq::product_op::to_string |
| lERK14complex_matrixRKNSt6vectorI |     (C++                          |
| N14complex_matrix10value_typeEEE) |     function)](                   |
| -                                 | api/languages/cpp_api.html#_CPPv4 |
| [cudaq::complex_matrix::operator+ | NK5cudaq10product_op9to_stringEv) |
|     (C++                          | -                                 |
|     function                      |  [cudaq::product_op::\~product_op |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq14complex_matrixplERK14 |     fu                            |
| complex_matrixRK14complex_matrix) | nction)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq10product_opD0Ev) |
| [cudaq::complex_matrix::operator- | -   [cudaq::ptsbe (C++            |
|     (C++                          |     type)](api/languages/c        |
|     function                      | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::p                     |
| Pv4N5cudaq14complex_matrixmiERK14 | tsbe::ConditionalSamplingStrategy |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -   [cu                           |     class)](api/languag           |
| daq::complex_matrix::operator\[\] | es/cpp_api.html#_CPPv4N5cudaq5pts |
|     (C++                          | be27ConditionalSamplingStrategyE) |
|                                   | -   [cudaq::ptsbe::C              |
|  function)](api/languages/cpp_api | onditionalSamplingStrategy::clone |
| .html#_CPPv4N5cudaq14complex_matr |     (C++                          |
| ixixERKNSt6vectorINSt6size_tEEE), |                                   |
|     [\[1\]](api/languages/cpp_api |    function)](api/languages/cpp_a |
| .html#_CPPv4NK5cudaq14complex_mat | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| rixixERKNSt6vectorINSt6size_tEEE) | ditionalSamplingStrategy5cloneEv) |
| -   [cudaq::complex_matrix::power | -   [cuda                         |
|     (C++                          | q::ptsbe::ConditionalSamplingStra |
|     function)]                    | tegy::ConditionalSamplingStrategy |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq14complex_matrix5powerEi) |     function)](api/lang           |
| -                                 | uages/cpp_api.html#_CPPv4N5cudaq5 |
|  [cudaq::complex_matrix::set_zero | ptsbe27ConditionalSamplingStrateg |
|     (C++                          | y27ConditionalSamplingStrategyE19 |
|     function)](ap                 | TrajectoryPredicateNSt8uint64_tE) |
| i/languages/cpp_api.html#_CPPv4N5 | -                                 |
| cudaq14complex_matrix8set_zeroEv) |   [cudaq::ptsbe::ConditionalSampl |
| -                                 | ingStrategy::generateTrajectories |
| [cudaq::complex_matrix::to_string |     (C++                          |
|     (C++                          |     function)](api/language       |
|     function)](api/               | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| languages/cpp_api.html#_CPPv4NK5c | be27ConditionalSamplingStrategy20 |
| udaq14complex_matrix9to_stringEv) | generateTrajectoriesENSt4spanIKN6 |
| -   [                             | detail10NoisePointEEENSt6size_tE) |
| cudaq::complex_matrix::value_type | -   [cudaq::ptsbe::               |
|     (C++                          | ConditionalSamplingStrategy::name |
|     type)](api/                   |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/languages/cpp_ |
| daq14complex_matrix10value_typeE) | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| -   [cudaq::contrib (C++          | nditionalSamplingStrategy4nameEv) |
|     type)](api/languages/cpp      | -   [cudaq:                       |
| _api.html#_CPPv4N5cudaq7contribE) | :ptsbe::ConditionalSamplingStrate |
| -                                 | gy::\~ConditionalSamplingStrategy |
| [cudaq::contrib::amplitude_encode |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     function)](api/language       | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| s/cpp_api.html#_CPPv4N5cudaq7cont | 7ConditionalSamplingStrategyD0Ev) |
| rib16amplitude_encodeENSt4spanIKN | -                                 |
| St7complexIdEEEENSt7complexIdEE), | [cudaq::ptsbe::detail::NoisePoint |
|     [\[1\]](api/language          |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq7cont |     struct)](a                    |
| rib16amplitude_encodeENSt4spanIKN | pi/languages/cpp_api.html#_CPPv4N |
| St7complexIfEEEENSt7complexIdEE), | 5cudaq5ptsbe6detail10NoisePointE) |
|     [\[2\]                        | -   [cudaq::p                     |
| ](api/languages/cpp_api.html#_CPP | tsbe::detail::NoisePoint::channel |
| v4N5cudaq7contrib16amplitude_enco |     (C++                          |
| deENSt4spanIKdEENSt7complexIdEE), |     member)](api/langu            |
|     [\[3\]                        | ages/cpp_api.html#_CPPv4N5cudaq5p |
| ](api/languages/cpp_api.html#_CPP | tsbe6detail10NoisePoint7channelE) |
| v4N5cudaq7contrib16amplitude_enco | -   [cudaq::ptsbe::det            |
| deENSt4spanIKfEENSt7complexIdEE), | ail::NoisePoint::circuit_location |
|                                   |     (C++                          |
| [\[4\]](api/languages/cpp_api.htm |     member)](api/languages/cpp_a  |
| l#_CPPv4N5cudaq7contrib16amplitud | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| e_encodeERK5stateNSt7complexIdEE) | l10NoisePoint16circuit_locationE) |
| -                                 | -   [cudaq::p                     |
|   [cudaq::contrib::angular_encode | tsbe::detail::NoisePoint::op_name |
|     (C++                          |     (C++                          |
|                                   |     member)](api/langu            |
|  function)](api/languages/cpp_api | ages/cpp_api.html#_CPPv4N5cudaq5p |
| .html#_CPPv4I0EN5cudaq7contrib14a | tsbe6detail10NoisePoint7op_nameE) |
| ngular_encodeEvRR6KernelR10QuakeV | -   [cudaq::                      |
| alueNSt4spanIKdEE12RotationAxis), | ptsbe::detail::NoisePoint::qubits |
|     [\[1\]](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I0EN5cudaq7contrib14a |     member)](api/lang             |
| ngular_encodeEvRR6KernelR10QuakeV | uages/cpp_api.html#_CPPv4N5cudaq5 |
| alueR10QuakeValue12RotationAxis), | ptsbe6detail10NoisePoint6qubitsE) |
|                                   | -   [cudaq::                      |
|   [\[2\]](api/languages/cpp_api.h | ptsbe::ExhaustiveSamplingStrategy |
| tml#_CPPv4I0EN5cudaq7contrib14ang |     (C++                          |
| ular_encodeEvRR6KernelR10QuakeVal |     class)](api/langua            |
| ueRKNSt6vectorIdEE12RotationAxis) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq::contrib::draw (C++    | sbe26ExhaustiveSamplingStrategyE) |
|     function)                     | -   [cudaq::ptsbe::               |
| ](api/languages/cpp_api.html#_CPP | ExhaustiveSamplingStrategy::clone |
| v4I0DpEN5cudaq7contrib4drawENSt6s |     (C++                          |
| tringERR13QuantumKernelDpRR4Args) |     function)](api/languages/cpp_ |
| -                                 | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| [cudaq::contrib::get_unitary_cmat | haustiveSamplingStrategy5cloneEv) |
|     (C++                          | -   [cu                           |
|     function)](api/languages/cp   | daq::ptsbe::ExhaustiveSamplingStr |
| p_api.html#_CPPv4I0DpEN5cudaq7con | ategy::ExhaustiveSamplingStrategy |
| trib16get_unitary_cmatE14complex_ |     (C++                          |
| matrixRR13QuantumKernelDpRR4Args) |     function)](api/la             |
| -   [cudaq::contrib::RotationAxis | nguages/cpp_api.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe26ExhaustiveSamplingStrate |
|     enum)                         | gy26ExhaustiveSamplingStrategyEv) |
| ](api/languages/cpp_api.html#_CPP | -                                 |
| v4N5cudaq7contrib12RotationAxisE) |    [cudaq::ptsbe::ExhaustiveSampl |
| -                                 | ingStrategy::generateTrajectories |
|  [cudaq::contrib::RotationAxis::X |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](                 | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| api/languages/cpp_api.html#_CPPv4 | sbe26ExhaustiveSamplingStrategy20 |
| N5cudaq7contrib12RotationAxis1XE) | generateTrajectoriesENSt4spanIKN6 |
| -                                 | detail10NoisePointEEENSt6size_tE) |
|  [cudaq::contrib::RotationAxis::Y | -   [cudaq::ptsbe:                |
|     (C++                          | :ExhaustiveSamplingStrategy::name |
|     enumerator)](                 |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languages/cpp  |
| N5cudaq7contrib12RotationAxis1YE) | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| -                                 | xhaustiveSamplingStrategy4nameEv) |
|  [cudaq::contrib::RotationAxis::Z | -   [cuda                         |
|     (C++                          | q::ptsbe::ExhaustiveSamplingStrat |
|     enumerator)](                 | egy::\~ExhaustiveSamplingStrategy |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq7contrib12RotationAxis1ZE) |     function)](api/languages      |
| -   [cudaq::CusvState (C++        | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|                                   | 26ExhaustiveSamplingStrategyD0Ev) |
|    class)](api/languages/cpp_api. | -   [cuda                         |
| html#_CPPv4I0EN5cudaq9CusvStateE) | q::ptsbe::OrderedSamplingStrategy |
| -   [cudaq::dem_from_kernel (C++  |     (C++                          |
|     function)](api                |     class)](api/lan               |
| /languages/cpp_api.html#_CPPv4I0D | guages/cpp_api.html#_CPPv4N5cudaq |
| pEN5cudaq15dem_from_kernelENSt6st | 5ptsbe23OrderedSamplingStrategyE) |
| ringERR13QuantumKernelDpRR4Args), | -   [cudaq::ptsb                  |
|                                   | e::OrderedSamplingStrategy::clone |
| [\[1\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4I0DpEN5cudaq15dem_from_ke |     function)](api/languages/c    |
| rnelENSt6stringERR13QuantumKernel | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| PKN5cudaq11noise_modelEDpRR4Args) | 3OrderedSamplingStrategy5cloneEv) |
| -   [cudaq::depolarization1 (C++  | -   [cudaq::ptsbe::OrderedSampl   |
|     c                             | ingStrategy::generateTrajectories |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15depolarization1E) |     function)](api/lang           |
| -   [cudaq::depolarization2 (C++  | uages/cpp_api.html#_CPPv4NK5cudaq |
|     c                             | 5ptsbe23OrderedSamplingStrategy20 |
| lass)](api/languages/cpp_api.html | generateTrajectoriesENSt4spanIKN6 |
| #_CPPv4N5cudaq15depolarization2E) | detail10NoisePointEEENSt6size_tE) |
| -   [cudaq:                       | -   [cudaq::pts                   |
| :depolarization2::depolarization2 | be::OrderedSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/languages/     |
| p_api.html#_CPPv4N5cudaq15depolar | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| ization215depolarization2EK4real) | 23OrderedSamplingStrategy4nameEv) |
| -   [cudaq                        | -                                 |
| ::depolarization2::num_parameters |    [cudaq::ptsbe::OrderedSampling |
|     (C++                          | Strategy::OrderedSamplingStrategy |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     function)](                   |
| depolarization214num_parametersE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cu                           | N5cudaq5ptsbe23OrderedSamplingStr |
| daq::depolarization2::num_targets | ategy23OrderedSamplingStrategyEv) |
|     (C++                          | -                                 |
|     member)](api/la               |  [cudaq::ptsbe::OrderedSamplingSt |
| nguages/cpp_api.html#_CPPv4N5cuda | rategy::\~OrderedSamplingStrategy |
| q15depolarization211num_targetsE) |     (C++                          |
| -                                 |     function)](api/langua         |
|    [cudaq::depolarization_channel | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     (C++                          | sbe23OrderedSamplingStrategyD0Ev) |
|     class)](                      | -   [cudaq::pts                   |
| api/languages/cpp_api.html#_CPPv4 | be::ProbabilisticSamplingStrategy |
| N5cudaq22depolarization_channelE) |     (C++                          |
| -   [cudaq::depol                 |     class)](api/languages         |
| arization_channel::num_parameters | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     (C++                          | 29ProbabilisticSamplingStrategyE) |
|     member)](api/languages/cp     | -   [cudaq::ptsbe::Pro            |
| p_api.html#_CPPv4N5cudaq22depolar | babilisticSamplingStrategy::clone |
| ization_channel14num_parametersE) |     (C++                          |
| -   [cudaq::de                    |                                   |
| polarization_channel::num_targets |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4NK5cudaq5ptsbe29Proba |
|     member)](api/languages        | bilisticSamplingStrategy5cloneEv) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -                                 |
| larization_channel11num_targetsE) | [cudaq::ptsbe::ProbabilisticSampl |
| -   [cudaq::detail (C++           | ingStrategy::generateTrajectories |
|     type)](api/languages/cp       |     (C++                          |
| p_api.html#_CPPv4N5cudaq6detailE) |     function)](api/languages/     |
| -   [cudaq::detail::future (C++   | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|                                   | 29ProbabilisticSamplingStrategy20 |
|   class)](api/languages/cpp_api.h | generateTrajectoriesENSt4spanIKN6 |
| tml#_CPPv4N5cudaq6detail6futureE) | detail10NoisePointEEENSt6size_tE) |
| -                                 | -   [cudaq::ptsbe::Pr             |
|    [cudaq::detail::future::future | obabilisticSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     functi                        |                                   |
| on)](api/languages/cpp_api.html#_ |   function)](api/languages/cpp_ap |
| CPPv4N5cudaq6detail6future6future | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| ERNSt6vectorI3JobEERNSt6stringERN | abilisticSamplingStrategy4nameEv) |
| St3mapINSt6stringENSt6stringEEE), | -   [cudaq::p                     |
|     [\[1\]](api/lan               | tsbe::ProbabilisticSamplingStrate |
| guages/cpp_api.html#_CPPv4N5cudaq | gy::ProbabilisticSamplingStrategy |
| 6detail6future6futureERR6future), |     (C++                          |
|     [\[2\]                        |     function)]                    |
| ](api/languages/cpp_api.html#_CPP | (api/languages/cpp_api.html#_CPPv |
| v4N5cudaq6detail6future6futureEv) | 4N5cudaq5ptsbe29ProbabilisticSamp |
| -   [c                            | lingStrategy29ProbabilisticSampli |
| udaq::detail::kernel_builder_base | ngStrategyENSt8optionalINSt8uint6 |
|     (C++                          | 4_tEEENSt8optionalINSt6size_tEEE) |
|     class)](api/                  | -   [cudaq::pts                   |
| languages/cpp_api.html#_CPPv4N5cu | be::ProbabilisticSamplingStrategy |
| daq6detail19kernel_builder_baseE) | ::\~ProbabilisticSamplingStrategy |
| -   [cudaq::detail::              |     (C++                          |
| kernel_builder_base::operator\<\< |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4N5cudaq5ptsbe29P |
|     function)](api/langu          | robabilisticSamplingStrategyD0Ev) |
| ages/cpp_api.html#_CPPv4N5cudaq6d | -                                 |
| etail19kernel_builder_baselsERNSt | [cudaq::ptsbe::PTSBEExecutionData |
| 7ostreamERK19kernel_builder_base) |     (C++                          |
| -                                 |     struct)](ap                   |
| [cudaq::detail::KernelBuilderType | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq5ptsbe18PTSBEExecutionDataE) |
|     class)](ap                    | -   [cudaq::ptsbe::PTSBE          |
| i/languages/cpp_api.html#_CPPv4N5 | ExecutionData::count_instructions |
| cudaq6detail17KernelBuilderTypeE) |     (C++                          |
| -   [cudaq::                      |     function)](api/l              |
| detail::KernelBuilderType::create | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq5ptsbe18PTSBEExecutionData18co |
|     function                      | unt_instructionsE20TraceInstructi |
| )](api/languages/cpp_api.html#_CP | onTypeNSt8optionalINSt6stringEEE) |
| Pv4N5cudaq6detail17KernelBuilderT | -   [cudaq::ptsbe::P              |
| ype6createEPN4mlir11MLIRContextE) | TSBEExecutionData::get_trajectory |
| -   [cudaq::detail::Ker           |     (C++                          |
| nelBuilderType::KernelBuilderType |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function)](api/lan            | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| guages/cpp_api.html#_CPPv4N5cudaq | Data14get_trajectoryENSt6size_tE) |
| 6detail17KernelBuilderType17Kerne | -   [cudaq::ptsbe:                |
| lBuilderTypeERRNSt8functionIFN4ml | :PTSBEExecutionData::instructions |
| ir4TypeEPN4mlir11MLIRContextEEEE) |     (C++                          |
| -   [cudaq::detector (C++         |     member)](api/languages/cp     |
|     function)](api                | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| /languages/cpp_api.html#_CPPv4IDp | TSBEExecutionData12instructionsE) |
| EN5cudaq8detectorEvDpRR8MeasArgs) | -   [cudaq::ptsbe:                |
| -   [cudaq::detectors (C++        | :PTSBEExecutionData::trajectories |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq9detecto |     member)](api/languages/cp     |
| rsERKNSt6vectorI14measure_resultE | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| ERKNSt6vectorI14measure_resultEE) | TSBEExecutionData12trajectoriesE) |
| -   [cudaq::diag_matrix_callback  | -   [cudaq::ptsbe::PTSBEOptions   |
|     (C++                          |     (C++                          |
|     class)                        |     struc                         |
| ](api/languages/cpp_api.html#_CPP | t)](api/languages/cpp_api.html#_C |
| v4N5cudaq20diag_matrix_callbackE) | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| -   [cudaq::dyn (C++              | -   [cudaq::ptsbe::PTSB           |
|     member)](api/languages        | EOptions::include_sequential_data |
| /cpp_api.html#_CPPv4N5cudaq3dynE) |     (C++                          |
| -   [cudaq::ExecutionContext (C++ |                                   |
|     cl                            |    member)](api/languages/cpp_api |
| ass)](api/languages/cpp_api.html# | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| _CPPv4N5cudaq16ExecutionContextE) | ptions23include_sequential_dataE) |
| -   [c                            | -   [cudaq::ptsb                  |
| udaq::ExecutionContext::asyncExec | e::PTSBEOptions::max_trajectories |
|     (C++                          |     (C++                          |
|     member)](api/                 |     member)](api/languages/       |
| languages/cpp_api.html#_CPPv4N5cu | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| daq16ExecutionContext9asyncExecE) | 2PTSBEOptions16max_trajectoriesE) |
| -   [cud                          | -   [cudaq::ptsbe::PT             |
| aq::ExecutionContext::asyncResult | SBEOptions::return_execution_data |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     member)](api/languages/cpp_a  |
| guages/cpp_api.html#_CPPv4N5cudaq | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| 16ExecutionContext11asyncResultE) | EOptions21return_execution_dataE) |
| -   [cudaq:                       | -   [cudaq::pts                   |
| :ExecutionContext::batchIteration | be::PTSBEOptions::shot_allocation |
|     (C++                          |     (C++                          |
|     member)](api/langua           |     member)](api/languages        |
| ges/cpp_api.html#_CPPv4N5cudaq16E | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| xecutionContext14batchIterationE) | 12PTSBEOptions15shot_allocationE) |
| -   [cudaq::E                     | -   [cud                          |
| xecutionContext::canHandleObserve | aq::ptsbe::PTSBEOptions::strategy |
|     (C++                          |     (C++                          |
|     member)](api/language         |     member)](api/l                |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | anguages/cpp_api.html#_CPPv4N5cud |
| cutionContext16canHandleObserveE) | aq5ptsbe12PTSBEOptions8strategyE) |
| -   [cudaq::E                     | -   [cudaq::ptsbe::PTSBETrace     |
| xecutionContext::ExecutionContext |     (C++                          |
|     (C++                          |     t                             |
|     func                          | ype)](api/languages/cpp_api.html# |
| tion)](api/languages/cpp_api.html | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| #_CPPv4N5cudaq16ExecutionContext1 | -   [                             |
| 6ExecutionContextERKNSt6stringE), | cudaq::ptsbe::PTSSamplingStrategy |
|     [\[1\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16Execu |     class)](api                   |
| tionContext16ExecutionContextERKN | /languages/cpp_api.html#_CPPv4N5c |
| St6stringENSt6size_tENSt6size_tE) | udaq5ptsbe19PTSSamplingStrategyE) |
| -   [cudaq::E                     | -   [cudaq::                      |
| xecutionContext::expectationValue | ptsbe::PTSSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     member)](api/language         |     function)](api/languag        |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| cutionContext16expectationValueE) | sbe19PTSSamplingStrategy5cloneEv) |
| -   [cudaq::Execu                 | -   [cudaq::ptsbe::PTSSampl       |
| tionContext::explicitMeasurements | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     function)](api/               |
| p_api.html#_CPPv4N5cudaq16Executi | languages/cpp_api.html#_CPPv4NK5c |
| onContext20explicitMeasurementsE) | udaq5ptsbe19PTSSamplingStrategy20 |
| -   [cuda                         | generateTrajectoriesENSt4spanIKN6 |
| q::ExecutionContext::futureResult | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     member)](api/lang             | :ptsbe::PTSSamplingStrategy::name |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     (C++                          |
| 6ExecutionContext12futureResultE) |     function)](api/langua         |
| -   [cudaq::ExecutionContext      | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| ::hasConditionalsOnMeasureResults | tsbe19PTSSamplingStrategy4nameEv) |
|     (C++                          | -   [cudaq::ptsbe::PTSSampli      |
|     mem                           | ngStrategy::\~PTSSamplingStrategy |
| ber)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16ExecutionContext31 |     function)](api/la             |
| hasConditionalsOnMeasureResultsE) | nguages/cpp_api.html#_CPPv4N5cuda |
| -   [cudaq::Executi               | q5ptsbe19PTSSamplingStrategyD0Ev) |
| onContext::invocationResultBuffer | -   [cudaq::ptsbe::sample (C++    |
|     (C++                          |                                   |
|     member)](api/languages/cpp_   |  function)](api/languages/cpp_api |
| api.html#_CPPv4N5cudaq16Execution | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| Context22invocationResultBufferE) | mpleE13sample_resultRK14sample_op |
| -   [cu                           | tionsRR13QuantumKernelDpRR4Args), |
| daq::ExecutionContext::kernelName |     [\[1\]](api                   |
|     (C++                          | /languages/cpp_api.html#_CPPv4I0D |
|     member)](api/la               | pEN5cudaq5ptsbe6sampleE13sample_r |
| nguages/cpp_api.html#_CPPv4N5cuda | esultRKN5cudaq11noise_modelENSt6s |
| q16ExecutionContext10kernelNameE) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [cud                          | -   [cudaq::ptsbe::sample_async   |
| aq::ExecutionContext::kernelTrace |     (C++                          |
|     (C++                          |     function)](a                  |
|     member)](api/lan              | pi/languages/cpp_api.html#_CPPv4I |
| guages/cpp_api.html#_CPPv4N5cudaq | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| 16ExecutionContext11kernelTraceE) | 9async_sample_resultRK14sample_op |
| -   [cudaq:                       | tionsRR13QuantumKernelDpRR4Args), |
| :ExecutionContext::msm_dimensions |     [\[1\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4I0DpEN5cudaq5pts |
|     member)](api/langua           | be12sample_asyncE19async_sample_r |
| ges/cpp_api.html#_CPPv4N5cudaq16E | esultRKN5cudaq11noise_modelENSt6s |
| xecutionContext14msm_dimensionsE) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [cudaq::                      | -   [cudaq::ptsbe::sample_options |
| ExecutionContext::msm_prob_err_id |     (C++                          |
|     (C++                          |     struct)                       |
|     member)](api/languag          | ](api/languages/cpp_api.html#_CPP |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | v4N5cudaq5ptsbe14sample_optionsE) |
| ecutionContext15msm_prob_err_idE) | -   [cudaq::ptsbe::sample_result  |
| -   [cudaq::Ex                    |     (C++                          |
| ecutionContext::msm_probabilities |     class                         |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)](api/languages        | Pv4N5cudaq5ptsbe13sample_resultE) |
| /cpp_api.html#_CPPv4N5cudaq16Exec | -   [cudaq::pts                   |
| utionContext17msm_probabilitiesE) | be::sample_result::execution_data |
| -                                 |     (C++                          |
|    [cudaq::ExecutionContext::name |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
|     member)]                      | 3sample_result14execution_dataEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::ptsbe::               |
| 4N5cudaq16ExecutionContext4nameE) | sample_result::has_execution_data |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::noiseModel |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     member)](api/la               | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| nguages/cpp_api.html#_CPPv4N5cuda | ple_result18has_execution_dataEv) |
| q16ExecutionContext10noiseModelE) | -   [cudaq::pt                    |
| -   [cudaq::Exe                   | sbe::sample_result::sample_result |
| cutionContext::numberTrajectories |     (C++                          |
|     (C++                          |     function)](api/l              |
|     member)](api/languages/       | anguages/cpp_api.html#_CPPv4N5cud |
| cpp_api.html#_CPPv4N5cudaq16Execu | aq5ptsbe13sample_result13sample_r |
| tionContext18numberTrajectoriesE) | esultERRN5cudaq13sample_resultE), |
| -   [c                            |                                   |
| udaq::ExecutionContext::optResult |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq5ptsbe13sample_re |
|     member)](api/                 | sult13sample_resultERRN5cudaq13sa |
| languages/cpp_api.html#_CPPv4N5cu | mple_resultE18PTSBEExecutionData) |
| daq16ExecutionContext9optResultE) | -   [cudaq::ptsbe::               |
| -                                 | sample_result::set_execution_data |
|   [cudaq::ExecutionContext::qpuId |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](                     | languages/cpp_api.html#_CPPv4N5cu |
| api/languages/cpp_api.html#_CPPv4 | daq5ptsbe13sample_result18set_exe |
| N5cudaq16ExecutionContext5qpuIdE) | cution_dataE18PTSBEExecutionData) |
| -   [cudaq                        | -   [cud                          |
| ::ExecutionContext::registerNames | aq::ptsbe::ShotAllocationStrategy |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     struct)](using                |
| ages/cpp_api.html#_CPPv4N5cudaq16 | /examples/ptsbe.html#_CPPv4N5cuda |
| ExecutionContext13registerNamesE) | q5ptsbe22ShotAllocationStrategyE) |
| -   [cu                           | -   [cudaq::ptsbe::ShotAllocatio  |
| daq::ExecutionContext::reorderIdx | nStrategy::ShotAllocationStrategy |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)                     |
| nguages/cpp_api.html#_CPPv4N5cuda | ](using/examples/ptsbe.html#_CPPv |
| q16ExecutionContext10reorderIdxE) | 4N5cudaq5ptsbe22ShotAllocationStr |
| -                                 | ategy22ShotAllocationStrategyE4Ty |
|  [cudaq::ExecutionContext::result | pedNSt8optionalINSt8uint64_tEEE), |
|     (C++                          |     [\[1\                         |
|     member)](a                    | ]](using/examples/ptsbe.html#_CPP |
| pi/languages/cpp_api.html#_CPPv4N | v4N5cudaq5ptsbe22ShotAllocationSt |
| 5cudaq16ExecutionContext6resultE) | rategy22ShotAllocationStrategyEv) |
| -                                 | -   [cudaq::pt                    |
|   [cudaq::ExecutionContext::shots | sbe::ShotAllocationStrategy::Type |
|     (C++                          |     (C++                          |
|     member)](                     |     enum)](using/exam             |
| api/languages/cpp_api.html#_CPPv4 | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| N5cudaq16ExecutionContext5shotsE) | be22ShotAllocationStrategy4TypeE) |
| -   [cudaq::                      | -   [cudaq::ptsbe::ShotAllocatio  |
| ExecutionContext::simulationState | nStrategy::Type::HIGH_WEIGHT_BIAS |
|     (C++                          |     (C++                          |
|     member)](api/languag          |     enumerat                      |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | or)](using/examples/ptsbe.html#_C |
| ecutionContext15simulationStateE) | PPv4N5cudaq5ptsbe22ShotAllocation |
| -                                 | Strategy4Type16HIGH_WEIGHT_BIASE) |
|    [cudaq::ExecutionContext::spin | -   [cudaq::ptsbe::ShotAllocati   |
|     (C++                          | onStrategy::Type::LOW_WEIGHT_BIAS |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     enumera                       |
| 4N5cudaq16ExecutionContext4spinE) | tor)](using/examples/ptsbe.html#_ |
| -   [cudaq::                      | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| ExecutionContext::totalIterations | nStrategy4Type15LOW_WEIGHT_BIASE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAlloc      |
|     member)](api/languag          | ationStrategy::Type::PROPORTIONAL |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15totalIterationsE) |     enum                          |
| -   [cudaq::ExecutionResult (C++  | erator)](using/examples/ptsbe.htm |
|     st                            | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| ruct)](api/languages/cpp_api.html | tionStrategy4Type12PROPORTIONALE) |
| #_CPPv4N5cudaq15ExecutionResultE) | -   [cudaq::ptsbe::Shot           |
| -   [cud                          | AllocationStrategy::Type::UNIFORM |
| aq::ExecutionResult::appendResult |     (C++                          |
|     (C++                          |                                   |
|     functio                       |   enumerator)](using/examples/pts |
| n)](api/languages/cpp_api.html#_C | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| PPv4N5cudaq15ExecutionResult12app | AllocationStrategy4Type7UNIFORME) |
| endResultENSt6stringENSt6size_tE) | -                                 |
| -   [cu                           |   [cudaq::ptsbe::TraceInstruction |
| daq::ExecutionResult::deserialize |     (C++                          |
|     (C++                          |     struct)](                     |
|     function)                     | api/languages/cpp_api.html#_CPPv4 |
| ](api/languages/cpp_api.html#_CPP | N5cudaq5ptsbe16TraceInstructionE) |
| v4N5cudaq15ExecutionResult11deser | -   [cudaq:                       |
| ializeERNSt6vectorINSt6size_tEEE) | :ptsbe::TraceInstruction::channel |
| -   [cudaq:                       |     (C++                          |
| :ExecutionResult::ExecutionResult |     member)](api/lang             |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     functio                       | ptsbe16TraceInstruction7channelE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::                      |
| PPv4N5cudaq15ExecutionResult15Exe | ptsbe::TraceInstruction::controls |
| cutionResultE16CountsDictionary), |     (C++                          |
|     [\[1\]](api/lan               |     member)](api/langu            |
| guages/cpp_api.html#_CPPv4N5cudaq | ages/cpp_api.html#_CPPv4N5cudaq5p |
| 15ExecutionResult15ExecutionResul | tsbe16TraceInstruction8controlsE) |
| tE16CountsDictionaryNSt6stringE), | -   [cud                          |
|     [\[2\                         | aq::ptsbe::TraceInstruction::name |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     member)](api/l                |
| utionResultE16CountsDictionaryd), | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq5ptsbe16TraceInstruction4nameE) |
|    [\[3\]](api/languages/cpp_api. | -   [cudaq                        |
| html#_CPPv4N5cudaq15ExecutionResu | ::ptsbe::TraceInstruction::params |
| lt15ExecutionResultENSt6stringE), |     (C++                          |
|     [\[4\                         |     member)](api/lan              |
| ]](api/languages/cpp_api.html#_CP | guages/cpp_api.html#_CPPv4N5cudaq |
| Pv4N5cudaq15ExecutionResult15Exec | 5ptsbe16TraceInstruction6paramsE) |
| utionResultERK15ExecutionResult), | -   [cudaq:                       |
|     [\[5\]](api/language          | :ptsbe::TraceInstruction::targets |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     (C++                          |
| cutionResult15ExecutionResultEd), |     member)](api/lang             |
|     [\[6\]](api/languag           | uages/cpp_api.html#_CPPv4N5cudaq5 |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | ptsbe16TraceInstruction7targetsE) |
| ecutionResult15ExecutionResultEv) | -   [cudaq::ptsbe::T              |
| -   [                             | raceInstruction::TraceInstruction |
| cudaq::ExecutionResult::operator= |     (C++                          |
|     (C++                          |                                   |
|     function)](api/languages/     |   function)](api/languages/cpp_ap |
| cpp_api.html#_CPPv4N5cudaq15Execu | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| tionResultaSERK15ExecutionResult) | Instruction16TraceInstructionE20T |
| -   [c                            | raceInstructionTypeNSt6stringENSt |
| udaq::ExecutionResult::operator== | 6vectorINSt6size_tEEENSt6vectorIN |
|     (C++                          | St6size_tEEENSt6vectorIdEENSt8opt |
|     function)](api/languages/c    | ionalIN5cudaq13kraus_channelEEE), |
| pp_api.html#_CPPv4NK5cudaq15Execu |     [\[1\]](api/languages/cpp_a   |
| tionResulteqERK15ExecutionResult) | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| -   [cud                          | eInstruction16TraceInstructionEv) |
| aq::ExecutionResult::registerName | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::type |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     member)](api/l                |
| 15ExecutionResult12registerNameE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq                        | aq5ptsbe16TraceInstruction4typeE) |
| ::ExecutionResult::sequentialData | -   [c                            |
|     (C++                          | udaq::ptsbe::TraceInstructionType |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     enum)](api/                   |
| ExecutionResult14sequentialDataE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [                             | daq5ptsbe20TraceInstructionTypeE) |
| cudaq::ExecutionResult::serialize | -   [cudaq::                      |
|     (C++                          | ptsbe::TraceInstructionType::Gate |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4NK5cu |     enumerator)](api/langu        |
| daq15ExecutionResult9serializeEv) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [cudaq::fermion_handler (C++  | tsbe20TraceInstructionType4GateE) |
|     c                             | -   [cudaq::ptsbe::               |
| lass)](api/languages/cpp_api.html | TraceInstructionType::Measurement |
| #_CPPv4N5cudaq15fermion_handlerE) |     (C++                          |
| -   [cudaq::fermion_op (C++       |                                   |
|     type)](api/languages/cpp_api  |    enumerator)](api/languages/cpp |
| .html#_CPPv4N5cudaq10fermion_opE) | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| -   [cudaq::fermion_op_term (C++  | aceInstructionType11MeasurementE) |
|                                   | -   [cudaq::p                     |
| type)](api/languages/cpp_api.html | tsbe::TraceInstructionType::Noise |
| #_CPPv4N5cudaq15fermion_op_termE) |     (C++                          |
| -   [cudaq::FermioniqQPU (C++     |     enumerator)](api/langua       |
|                                   | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|   class)](api/languages/cpp_api.h | sbe20TraceInstructionType5NoiseE) |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | -   [                             |
| -   [cudaq::get_state (C++        | cudaq::ptsbe::TrajectoryPredicate |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     type)](api                    |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | /languages/cpp_api.html#_CPPv4N5c |
| ateEDaRR13QuantumKernelDpRR4Args) | udaq5ptsbe19TrajectoryPredicateE) |
| -   [cudaq::gradient (C++         | -   [cudaq::QPU (C++              |
|     class)](api/languages/cpp_    |     class)](api/languages         |
| api.html#_CPPv4N5cudaq8gradientE) | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| -   [cudaq::gradient::clone (C++  | -   [cudaq::QPU::beginExecution   |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     function                      |
| l#_CPPv4N5cudaq8gradient5cloneEv) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::gradient::compute     | Pv4N5cudaq3QPU14beginExecutionEv) |
|     (C++                          | -   [cuda                         |
|     function)](api/language       | q::QPU::configureExecutionContext |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     (C++                          |
| ient7computeERKNSt6vectorIdEERKNS |     funct                         |
| t8functionIFdNSt6vectorIdEEEEEd), | ion)](api/languages/cpp_api.html# |
|     [\[1\]](ap                    | _CPPv4NK5cudaq3QPU25configureExec |
| i/languages/cpp_api.html#_CPPv4N5 | utionContextER16ExecutionContext) |
| cudaq8gradient7computeERKNSt6vect | -   [cudaq::QPU::endExecution     |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradient::gradient    |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     function)](api/lang           | CPPv4N5cudaq3QPU12endExecutionEv) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -   [cudaq::QPU::enqueue (C++     |
| daq8gradient8gradientER7KernelT), |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|    [\[1\]](api/languages/cpp_api. | cudaq3QPU7enqueueER11QuantumTask) |
| html#_CPPv4I00EN5cudaq8gradient8g | -   [cud                          |
| radientER7KernelTRR10ArgsMapper), | aq::QPU::finalizeExecutionContext |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     func                          |
| Pv4I00EN5cudaq8gradient8gradientE | tion)](api/languages/cpp_api.html |
| RR13QuantumKernelRR10ArgsMapper), | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     [\[3                          | utionContextER16ExecutionContext) |
| \]](api/languages/cpp_api.html#_C | -   [cudaq::QPU::getCompileTarget |
| PPv4N5cudaq8gradient8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](api/languages/c    |
|     [\[                           | pp_api.html#_CPPv4N5cudaq3QPU16ge |
| 4\]](api/languages/cpp_api.html#_ | tCompileTargetERK13sample_policy) |
| CPPv4N5cudaq8gradient8gradientEv) | -   [cudaq::QPU::getConnectivity  |
| -   [cudaq::gradient::setArgs     |     (C++                          |
|     (C++                          |     function)                     |
|     fu                            | ](api/languages/cpp_api.html#_CPP |
| nction)](api/languages/cpp_api.ht | v4N5cudaq3QPU15getConnectivityEv) |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | -                                 |
| tArgsEvR13QuantumKernelDpRR4Args) | [cudaq::QPU::getExecutionThreadId |
| -   [cudaq::gradient::setKernel   |     (C++                          |
|     (C++                          |     function)](api/               |
|     function)](api/languages/c    | languages/cpp_api.html#_CPPv4NK5c |
| pp_api.html#_CPPv4I0EN5cudaq8grad | udaq3QPU20getExecutionThreadIdEv) |
| ient9setKernelEvR13QuantumKernel) | -   [cudaq::QPU::getNumQubits     |
| -   [cud                          |     (C++                          |
| aq::gradients::central_difference |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     class)](api/la                | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [                             |
| q9gradients18central_differenceE) | cudaq::QPU::getRemoteCapabilities |
| -   [cudaq::gra                   |     (C++                          |
| dients::central_difference::clone |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     function)](api/languages      | daq3QPU21getRemoteCapabilitiesEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QPU::isEmulated (C++  |
| ents18central_difference5cloneEv) |     func                          |
| -   [cudaq::gradi                 | tion)](api/languages/cpp_api.html |
| ents::central_difference::compute | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     (C++                          | -   [cudaq::QPU::isSimulator (C++ |
|     function)](                   |     funct                         |
| api/languages/cpp_api.html#_CPPv4 | ion)](api/languages/cpp_api.html# |
| N5cudaq9gradients18central_differ | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| ence7computeERKNSt6vectorIdEERKNS | -   [cudaq::QPU::onRandomSeedSet  |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|                                   |     function)](api/lang           |
|   [\[1\]](api/languages/cpp_api.h | uages/cpp_api.html#_CPPv4N5cudaq3 |
| tml#_CPPv4N5cudaq9gradients18cent | QPU15onRandomSeedSetENSt6size_tE) |
| ral_difference7computeERKNSt6vect | -   [cudaq::QPU::QPU (C++         |
| orIdEERNSt6vectorIdEERK7spin_opd) |     functio                       |
| -   [cudaq::gradie                | n)](api/languages/cpp_api.html#_C |
| nts::central_difference::gradient | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
|     (C++                          |                                   |
|     functio                       |  [\[1\]](api/languages/cpp_api.ht |
| n)](api/languages/cpp_api.html#_C | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| PPv4I00EN5cudaq9gradients18centra |     [\[2\]](api/languages/cpp_    |
| l_difference8gradientER7KernelT), | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     [\[1\]](api/langua            | -   [cudaq::QPU::setId (C++       |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     function                      |
| q9gradients18central_difference8g | )](api/languages/cpp_api.html#_CP |
| radientER7KernelTRR10ArgsMapper), | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::QPU::setShots (C++    |
| api.html#_CPPv4I00EN5cudaq9gradie |     f                             |
| nts18central_difference8gradientE | unction)](api/languages/cpp_api.h |
| RR13QuantumKernelRR10ArgsMapper), | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::                      |
| _api.html#_CPPv4N5cudaq9gradients | QPU::supportsExplicitMeasurements |
| 18central_difference8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](api/languag        |
|     [\[4\]](api/languages/cp      | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| p_api.html#_CPPv4N5cudaq9gradient | 28supportsExplicitMeasurementsEv) |
| s18central_difference8gradientEv) | -   [cudaq::QPU::\~QPU (C++       |
| -   [cud                          |     function)](api/languages/cp   |
| aq::gradients::forward_difference | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     (C++                          | -   [cudaq::QPUState (C++         |
|     class)](api/la                |     class)](api/languages/cpp_    |
| nguages/cpp_api.html#_CPPv4N5cuda | api.html#_CPPv4N5cudaq8QPUStateE) |
| q9gradients18forward_differenceE) | -   [cudaq::qreg (C++             |
| -   [cudaq::gra                   |     class)](api/lan               |
| dients::forward_difference::clone | guages/cpp_api.html#_CPPv4I_NSt6s |
|     (C++                          | ize_tE_NSt6size_tEEN5cudaq4qregE) |
|     function)](api/languages      | -   [cudaq::qreg::back (C++       |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)                     |
| ents18forward_difference5cloneEv) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::gradi                 | v4N5cudaq4qreg4backENSt6size_tE), |
| ents::forward_difference::compute |     [\[1\]](api/languages/cpp_ap  |
|     (C++                          | i.html#_CPPv4N5cudaq4qreg4backEv) |
|     function)](                   | -   [cudaq::qreg::begin (C++      |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq9gradients18forward_differ |  function)](api/languages/cpp_api |
| ence7computeERKNSt6vectorIdEERKNS | .html#_CPPv4N5cudaq4qreg5beginEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::qreg::clear (C++      |
|                                   |                                   |
|   [\[1\]](api/languages/cpp_api.h |  function)](api/languages/cpp_api |
| tml#_CPPv4N5cudaq9gradients18forw | .html#_CPPv4N5cudaq4qreg5clearEv) |
| ard_difference7computeERKNSt6vect | -   [cudaq::qreg::front (C++      |
| orIdEERNSt6vectorIdEERK7spin_opd) |     function)]                    |
| -   [cudaq::gradie                | (api/languages/cpp_api.html#_CPPv |
| nts::forward_difference::gradient | 4N5cudaq4qreg5frontENSt6size_tE), |
|     (C++                          |     [\[1\]](api/languages/cpp_api |
|     functio                       | .html#_CPPv4N5cudaq4qreg5frontEv) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::qreg::operator\[\]    |
| PPv4I00EN5cudaq9gradients18forwar |     (C++                          |
| d_difference8gradientER7KernelT), |     functi                        |
|     [\[1\]](api/langua            | on)](api/languages/cpp_api.html#_ |
| ges/cpp_api.html#_CPPv4I00EN5cuda | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| q9gradients18forward_difference8g | -   [cudaq::qreg::qreg (C++       |
| radientER7KernelTRR10ArgsMapper), |     function)                     |
|     [\[2\]](api/languages/cpp_    | ](api/languages/cpp_api.html#_CPP |
| api.html#_CPPv4I00EN5cudaq9gradie | v4N5cudaq4qreg4qregENSt6size_tE), |
| nts18forward_difference8gradientE |     [\[1\]](api/languages/cpp_ap  |
| RR13QuantumKernelRR10ArgsMapper), | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::qreg::size (C++       |
| _api.html#_CPPv4N5cudaq9gradients |                                   |
| 18forward_difference8gradientERRN |  function)](api/languages/cpp_api |
| St8functionIFvNSt6vectorIdEEEEE), | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qreg::slice (C++      |
| p_api.html#_CPPv4N5cudaq9gradient |     function)](api/langu          |
| s18forward_difference8gradientEv) | ages/cpp_api.html#_CPPv4N5cudaq4q |
| -   [                             | reg5sliceENSt6size_tENSt6size_tE) |
| cudaq::gradients::parameter_shift | -   [cudaq::qreg::value_type (C++ |
|     (C++                          |                                   |
|     class)](api                   | type)](api/languages/cpp_api.html |
| /languages/cpp_api.html#_CPPv4N5c | #_CPPv4N5cudaq4qreg10value_typeE) |
| udaq9gradients15parameter_shiftE) | -   [cudaq::qspan (C++            |
| -   [cudaq::                      |     class)](api/lang              |
| gradients::parameter_shift::clone | uages/cpp_api.html#_CPPv4I_NSt6si |
|     (C++                          | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     function)](api/langua         | -   [cudaq::QuakeValue (C++       |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     class)](api/languages/cpp_api |
| adients15parameter_shift5cloneEv) | .html#_CPPv4N5cudaq10QuakeValueE) |
| -   [cudaq::gr                    | -   [cudaq::Q                     |
| adients::parameter_shift::compute | uakeValue::canValidateNumElements |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/languages      |
| )](api/languages/cpp_api.html#_CP | /cpp_api.html#_CPPv4N5cudaq10Quak |
| Pv4N5cudaq9gradients15parameter_s | eValue22canValidateNumElementsEv) |
| hift7computeERKNSt6vectorIdEERKNS | -                                 |
| t8functionIFdNSt6vectorIdEEEEEd), |  [cudaq::QuakeValue::constantSize |
|     [\[1\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4N5cudaq9gradients15p |     function)](api                |
| arameter_shift7computeERKNSt6vect | /languages/cpp_api.html#_CPPv4N5c |
| orIdEERNSt6vectorIdEERK7spin_opd) | udaq10QuakeValue12constantSizeEv) |
| -   [cudaq::gra                   | -   [cudaq::QuakeValue::dump (C++ |
| dients::parameter_shift::gradient |     function)](api/lan            |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     func                          | 10QuakeValue4dumpERNSt7ostreamE), |
| tion)](api/languages/cpp_api.html |     [\                            |
| #_CPPv4I00EN5cudaq9gradients15par | [1\]](api/languages/cpp_api.html# |
| ameter_shift8gradientER7KernelT), | _CPPv4N5cudaq10QuakeValue4dumpEv) |
|     [\[1\]](api/lan               | -   [cudaq                        |
| guages/cpp_api.html#_CPPv4I00EN5c | ::QuakeValue::getRequiredElements |
| udaq9gradients15parameter_shift8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api/langua         |
|     [\[2\]](api/languages/c       | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| pp_api.html#_CPPv4I00EN5cudaq9gra | uakeValue19getRequiredElementsEv) |
| dients15parameter_shift8gradientE | -   [cudaq::QuakeValue::getValue  |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3\]](api/languages/        |     function)]                    |
| cpp_api.html#_CPPv4N5cudaq9gradie | (api/languages/cpp_api.html#_CPPv |
| nts15parameter_shift8gradientERRN | 4NK5cudaq10QuakeValue8getValueEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QuakeValue::inverse   |
|     [\[4\]](api/languages         |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)                     |
| ents15parameter_shift8gradientEv) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kernel_builder (C++   | v4NK5cudaq10QuakeValue7inverseEv) |
|     clas                          | -   [cudaq::QuakeValue::isStdVec  |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4IDpEN5cudaq14kernel_builderE) |     function)                     |
| -   [c                            | ](api/languages/cpp_api.html#_CPP |
| udaq::kernel_builder::constantVal | v4N5cudaq10QuakeValue8isStdVecEv) |
|     (C++                          | -                                 |
|     function)](api/la             |    [cudaq::QuakeValue::operator\* |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14kernel_builder11constantValEd) |     function)](api                |
| -                                 | /languages/cpp_api.html#_CPPv4N5c |
|  [cudaq::kernel_builder::detector | udaq10QuakeValuemlE10QuakeValue), |
|     (C++                          |                                   |
|                                   | [\[1\]](api/languages/cpp_api.htm |
|    function)](api/languages/cpp_a | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| pi.html#_CPPv4IDpEN5cudaq14kernel | -   [cudaq::QuakeValue::operator+ |
| _builder8detectorEvDpRR8MeasArgs) |     (C++                          |
| -                                 |     function)](api                |
| [cudaq::kernel_builder::detectors | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValueplE10QuakeValue), |
|     func                          |     [                             |
| tion)](api/languages/cpp_api.html | \[1\]](api/languages/cpp_api.html |
| #_CPPv4N5cudaq14kernel_builder9de | #_CPPv4N5cudaq10QuakeValueplEKd), |
| tectorsE10QuakeValue10QuakeValue) |                                   |
| -   [cu                           | [\[2\]](api/languages/cpp_api.htm |
| daq::kernel_builder::getArguments | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|     (C++                          | -   [cudaq::QuakeValue::operator- |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api                |
| 14kernel_builder12getArgumentsEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cu                           | udaq10QuakeValuemiE10QuakeValue), |
| daq::kernel_builder::getNumParams |     [                             |
|     (C++                          | \[1\]](api/languages/cpp_api.html |
|     function)](api/lan            | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [                             |
| 14kernel_builder12getNumParamsEv) | \[2\]](api/languages/cpp_api.html |
| -   [c                            | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| udaq::kernel_builder::isArgStdVec |                                   |
|     (C++                          | [\[3\]](api/languages/cpp_api.htm |
|     function)](api/languages/cp   | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| p_api.html#_CPPv4N5cudaq14kernel_ | -   [cudaq::QuakeValue::operator/ |
| builder11isArgStdVecENSt6size_tE) |     (C++                          |
| -   [cuda                         |     function)](api                |
| q::kernel_builder::kernel_builder | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuedvE10QuakeValue), |
|     function)](api/languages/cpp  |                                   |
| _api.html#_CPPv4N5cudaq14kernel_b | [\[1\]](api/languages/cpp_api.htm |
| uilder14kernel_builderERNSt6vecto | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| rIN6detail17KernelBuilderTypeEEE) | -                                 |
| -   [cudaq::k                     |  [cudaq::QuakeValue::operator\[\] |
| ernel_builder::logical_observable |     (C++                          |
|     (C++                          |     function)](api                |
|     function)                     | /languages/cpp_api.html#_CPPv4N5c |
| ](api/languages/cpp_api.html#_CPP | udaq10QuakeValueixEKNSt6size_tE), |
| v4IDpEN5cudaq14kernel_builder18lo |     [\[1\]](api/                  |
| gical_observableEvDpRR8MeasArgs), | languages/cpp_api.html#_CPPv4N5cu |
|     [\[1\]](ap                    | daq10QuakeValueixERK10QuakeValue) |
| i/languages/cpp_api.html#_CPPv4N5 | -                                 |
| cudaq14kernel_builder18logical_ob |    [cudaq::QuakeValue::QuakeValue |
| servableE10QuakeValueNSt6size_tE) |     (C++                          |
| -   [cudaq::kernel_builder::name  |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     function)                     | akeValue10QuakeValueERN4mlir20Imp |
| ](api/languages/cpp_api.html#_CPP | licitLocOpBuilderEN4mlir5ValueE), |
| v4N5cudaq14kernel_builder4nameEv) |     [\[1\]                        |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|    [cudaq::kernel_builder::qalloc | v4N5cudaq10QuakeValue10QuakeValue |
|     (C++                          | ERN4mlir20ImplicitLocOpBuilderEd) |
|     function)](api/language       | -   [cudaq::QuakeValue::size (C++ |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     funct                         |
| nel_builder6qallocE10QuakeValue), | ion)](api/languages/cpp_api.html# |
|     [\[1\]](api/language          | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::QuakeValue::slice     |
| nel_builder6qallocEKNSt6size_tE), |     (C++                          |
|     [\[2                          |     function)](api/languages/cpp_ |
| \]](api/languages/cpp_api.html#_C | api.html#_CPPv4N5cudaq10QuakeValu |
| PPv4N5cudaq14kernel_builder6qallo | e5sliceEKNSt6size_tEKNSt6size_tE) |
| cERNSt6vectorINSt7complexIdEEEE), | -   [cudaq::quantum_platform (C++ |
|     [\[3\]](                      |     cl                            |
| api/languages/cpp_api.html#_CPPv4 | ass)](api/languages/cpp_api.html# |
| N5cudaq14kernel_builder6qallocEv) | _CPPv4N5cudaq16quantum_platformE) |
| -   [cudaq::kernel_builder::swap  | -   [cudaq:                       |
|     (C++                          | :quantum_platform::beginExecution |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     function)](api/languag        |
| 4kernel_builder4swapEvRK10QuakeVa | es/cpp_api.html#_CPPv4N5cudaq16qu |
| lueRK10QuakeValueRK10QuakeValue), | antum_platform14beginExecutionEv) |
|                                   | -   [cudaq::quantum_pl            |
| [\[1\]](api/languages/cpp_api.htm | atform::configureExecutionContext |
| l#_CPPv4I00EN5cudaq14kernel_build |     (C++                          |
| er4swapEvRKNSt6vectorI10QuakeValu |     function)](api/lang           |
| eEERK10QuakeValueRK10QuakeValue), | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 16quantum_platform25configureExec |
| [\[2\]](api/languages/cpp_api.htm | utionContextER16ExecutionContext) |
| l#_CPPv4N5cudaq14kernel_builder4s | -   [cuda                         |
| wapERK10QuakeValueRK10QuakeValue) | q::quantum_platform::connectivity |
| -   [cudaq::KernelExecutionTask   |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     type                          | ages/cpp_api.html#_CPPv4N5cudaq16 |
| )](api/languages/cpp_api.html#_CP | quantum_platform12connectivityEv) |
| Pv4N5cudaq19KernelExecutionTaskE) | -   [cuda                         |
| -   [cudaq::KernelThunkResultType | q::quantum_platform::endExecution |
|     (C++                          |     (C++                          |
|     struct)]                      |     function)](api/langu          |
| (api/languages/cpp_api.html#_CPPv | ages/cpp_api.html#_CPPv4N5cudaq16 |
| 4N5cudaq21KernelThunkResultTypeE) | quantum_platform12endExecutionEv) |
| -   [cudaq::KernelThunkType (C++  | -   [cudaq::q                     |
|                                   | uantum_platform::enqueueAsyncTask |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     function)](api/languages/     |
| -   [cudaq::kraus_channel (C++    | cpp_api.html#_CPPv4N5cudaq16quant |
|                                   | um_platform16enqueueAsyncTaskEKNS |
|  class)](api/languages/cpp_api.ht | t6size_tER19KernelExecutionTask), |
| ml#_CPPv4N5cudaq13kraus_channelE) |     [\[1\]](api/languag           |
| -   [cudaq::kraus_channel::empty  | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform16enqueueAsyncTaskE |
|     function)]                    | KNSt6size_tERNSt8functionIFvvEEE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::quantum_p             |
| 4NK5cudaq13kraus_channel5emptyEv) | latform::finalizeExecutionContext |
| -   [cudaq::kraus_c               |     (C++                          |
| hannel::generateUnitaryParameters |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq16quant |
|                                   | um_platform24finalizeExecutionCon |
|    function)](api/languages/cpp_a | textERN5cudaq16ExecutionContextE) |
| pi.html#_CPPv4N5cudaq13kraus_chan | -   [cudaq::qua                   |
| nel25generateUnitaryParametersEv) | ntum_platform::get_codegen_config |
| -                                 |     (C++                          |
|    [cudaq::kraus_channel::get_ops |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq16quantu |
|     function)](a                  | m_platform18get_codegen_configEv) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cuda                         |
| K5cudaq13kraus_channel7get_opsEv) | q::quantum_platform::get_exec_ctx |
| -   [cud                          |     (C++                          |
| aq::kraus_channel::identity_flags |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|     member)](api/lan              | quantum_platform12get_exec_ctxEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [c                            |
| 13kraus_channel14identity_flagsE) | udaq::quantum_platform::get_noise |
| -   [cud                          |     (C++                          |
| aq::kraus_channel::is_identity_op |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq16quantu |
|                                   | m_platform9get_noiseENSt6size_tE) |
|    function)](api/languages/cpp_a | -   [cudaq:                       |
| pi.html#_CPPv4NK5cudaq13kraus_cha | :quantum_platform::get_num_qubits |
| nnel14is_identity_opENSt6size_tE) |     (C++                          |
| -   [cudaq::                      |                                   |
| kraus_channel::is_unitary_mixture | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4NK5cudaq16quantum_plat |
|     function)](api/languages      | form14get_num_qubitsENSt6size_tE) |
| /cpp_api.html#_CPPv4NK5cudaq13kra | -   [cudaq::quantum_              |
| us_channel18is_unitary_mixtureEv) | platform::get_remote_capabilities |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::kraus_channel |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/lang           | v4NK5cudaq16quantum_platform23get |
| uages/cpp_api.html#_CPPv4IDpEN5cu | _remote_capabilitiesENSt6size_tE) |
| daq13kraus_channel13kraus_channel | -   [cudaq::qua                   |
| EDpRRNSt16initializer_listI1TEE), | ntum_platform::get_runtime_target |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     function)](api/languages/cp   |
| ml#_CPPv4N5cudaq13kraus_channel13 | p_api.html#_CPPv4NK5cudaq16quantu |
| kraus_channelERK13kraus_channel), | m_platform18get_runtime_targetEv) |
|     [\[2\]                        | -   [cud                          |
| ](api/languages/cpp_api.html#_CPP | aq::quantum_platform::is_emulated |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERKNSt6vectorI8kraus_opEE), |                                   |
|     [\[3\]                        |    function)](api/languages/cpp_a |
| ](api/languages/cpp_api.html#_CPP | pi.html#_CPPv4NK5cudaq16quantum_p |
| v4N5cudaq13kraus_channel13kraus_c | latform11is_emulatedENSt6size_tE) |
| hannelERRNSt6vectorI8kraus_opEE), | -   [cudaq::                      |
|     [\[4\]](api/lan               | quantum_platform::is_library_mode |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 13kraus_channel13kraus_channelEv) |     function)](api/languages      |
| -                                 | /cpp_api.html#_CPPv4NK5cudaq16qua |
| [cudaq::kraus_channel::noise_type | ntum_platform15is_library_modeEv) |
|     (C++                          | -   [c                            |
|     member)](api                  | udaq::quantum_platform::is_remote |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel10noise_typeE) |     function)](api/languages/cp   |
| -                                 | p_api.html#_CPPv4NK5cudaq16quantu |
|   [cudaq::kraus_channel::op_names | m_platform9is_remoteENSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     member)](                     | q::quantum_platform::is_simulator |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq13kraus_channel8op_namesE) |                                   |
| -                                 |   function)](api/languages/cpp_ap |
|  [cudaq::kraus_channel::operator= | i.html#_CPPv4NK5cudaq16quantum_pl |
|     (C++                          | atform12is_simulatorENSt6size_tE) |
|     function)](api/langua         | -   [c                            |
| ges/cpp_api.html#_CPPv4N5cudaq13k | udaq::quantum_platform::launchVQE |
| raus_channelaSERK13kraus_channel) |     (C++                          |
| -   [c                            |     function)](                   |
| udaq::kraus_channel::operator\[\] | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq16quantum_platform9launchV |
|     function)](api/l              | QEEKNSt6stringEPKvPN5cudaq8gradie |
| anguages/cpp_api.html#_CPPv4N5cud | ntERKN5cudaq7spin_opERN5cudaq9opt |
| aq13kraus_channelixEKNSt6size_tE) | imizerEKiKNSt6size_tENSt6size_tE) |
| -                                 | -   [cudaq:                       |
| [cudaq::kraus_channel::parameters | :quantum_platform::list_platforms |
|     (C++                          |     (C++                          |
|     member)](api                  |     function)](api/languag        |
| /languages/cpp_api.html#_CPPv4N5c | es/cpp_api.html#_CPPv4N5cudaq16qu |
| udaq13kraus_channel10parametersE) | antum_platform14list_platformsEv) |
| -   [cudaq::krau                  | -                                 |
| s_channel::populateDefaultOpNames |    [cudaq::quantum_platform::name |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](a                  |
| p_api.html#_CPPv4N5cudaq13kraus_c | pi/languages/cpp_api.html#_CPPv4N |
| hannel22populateDefaultOpNamesEv) | K5cudaq16quantum_platform4nameEv) |
| -   [cu                           | -   [                             |
| daq::kraus_channel::probabilities | cudaq::quantum_platform::num_qpus |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api/l              |
| nguages/cpp_api.html#_CPPv4N5cuda | anguages/cpp_api.html#_CPPv4NK5cu |
| q13kraus_channel13probabilitiesE) | daq16quantum_platform8num_qpusEv) |
| -                                 | -   [cudaq::                      |
|  [cudaq::kraus_channel::push_back | quantum_platform::onRandomSeedSet |
|     (C++                          |     (C++                          |
|     function)](api                |                                   |
| /languages/cpp_api.html#_CPPv4N5c | function)](api/languages/cpp_api. |
| udaq13kraus_channel9push_backE8kr | html#_CPPv4N5cudaq16quantum_platf |
| aus_opNSt8optionalINSt6stringEEE) | orm15onRandomSeedSetENSt6size_tE) |
| -   [cudaq::kraus_channel::size   | -   [cudaq:                       |
|     (C++                          | :quantum_platform::reset_exec_ctx |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languag        |
| v4NK5cudaq13kraus_channel4sizeEv) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [                             | antum_platform14reset_exec_ctxEv) |
| cudaq::kraus_channel::unitary_ops | -   [cud                          |
|     (C++                          | aq::quantum_platform::reset_noise |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/languages/cpp_ |
| daq13kraus_channel11unitary_opsE) | api.html#_CPPv4N5cudaq16quantum_p |
| -   [cudaq::kraus_op (C++         | latform11reset_noiseENSt6size_tE) |
|     struct)](api/languages/cpp_   | -   [cuda                         |
| api.html#_CPPv4N5cudaq8kraus_opE) | q::quantum_platform::set_exec_ctx |
| -   [cudaq::kraus_op::adjoint     |     (C++                          |
|     (C++                          |     funct                         |
|     functi                        | ion)](api/languages/cpp_api.html# |
| on)](api/languages/cpp_api.html#_ | _CPPv4N5cudaq16quantum_platform12 |
| CPPv4NK5cudaq8kraus_op7adjointEv) | set_exec_ctxEP16ExecutionContext) |
| -   [cudaq::kraus_op::data (C++   | -   [c                            |
|                                   | udaq::quantum_platform::set_noise |
|  member)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     function                      |
| -   [cudaq::kraus_op::kraus_op    | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq16quantum_platform9set_ |
|     func                          | noiseEPK11noise_modelNSt6size_tE) |
| tion)](api/languages/cpp_api.html | -   [cudaq::quantum_platfor       |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | m::supports_explicit_measurements |
| opERRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)](api/l              |
|  [\[1\]](api/languages/cpp_api.ht | anguages/cpp_api.html#_CPPv4NK5cu |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | daq16quantum_platform30supports_e |
| pENSt6vectorIN5cudaq7complexEEE), | xplicit_measurementsENSt6size_tE) |
|     [\[2\]](api/l                 | -   [cudaq::quantum_pla           |
| anguages/cpp_api.html#_CPPv4N5cud | tform::supports_task_distribution |
| aq8kraus_op8kraus_opERK8kraus_op) |     (C++                          |
| -   [cudaq::kraus_op::nCols (C++  |     fu                            |
|                                   | nction)](api/languages/cpp_api.ht |
| member)](api/languages/cpp_api.ht | ml#_CPPv4NK5cudaq16quantum_platfo |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | rm26supports_task_distributionEv) |
| -   [cudaq::kraus_op::nRows (C++  | -   [cudaq::quantum               |
|                                   | _platform::with_execution_context |
| member)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) |     function)                     |
| -   [cudaq::kraus_op::operator=   | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4I0DpEN5cudaq16quantum_platform2 |
|     function)                     | 2with_execution_contextEDaR16Exec |
| ](api/languages/cpp_api.html#_CPP | utionContextRR8CallableDpRR4Args) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cudaq::QuantumTask (C++      |
| -   [cudaq::kraus_op::precision   |     type)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq11QuantumTaskE) |
|     memb                          | -   [cudaq::qubit (C++            |
| er)](api/languages/cpp_api.html#_ |     type)](api/languages/c        |
| CPPv4N5cudaq8kraus_op9precisionE) | pp_api.html#_CPPv4N5cudaq5qubitE) |
| -   [cudaq::KrausSelection (C++   | -   [cudaq::QubitConnectivity     |
|     s                             |     (C++                          |
| truct)](api/languages/cpp_api.htm |     ty                            |
| l#_CPPv4N5cudaq14KrausSelectionE) | pe)](api/languages/cpp_api.html#_ |
| -   [cudaq:                       | CPPv4N5cudaq17QubitConnectivityE) |
| :KrausSelection::circuit_location | -   [cudaq::QubitEdge (C++        |
|     (C++                          |     type)](api/languages/cpp_a    |
|     member)](api/langua           | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| ges/cpp_api.html#_CPPv4N5cudaq14K | -   [cudaq::qudit (C++            |
| rausSelection16circuit_locationE) |     clas                          |
| -                                 | s)](api/languages/cpp_api.html#_C |
|  [cudaq::KrausSelection::is_error | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     (C++                          | -   [cudaq::qudit::qudit (C++     |
|     member)](a                    |                                   |
| pi/languages/cpp_api.html#_CPPv4N | function)](api/languages/cpp_api. |
| 5cudaq14KrausSelection8is_errorE) | html#_CPPv4N5cudaq5qudit5quditEv) |
| -   [cudaq::Kra                   | -   [cudaq::qvector (C++          |
| usSelection::kraus_operator_index |     class)                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/languages/       | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| cpp_api.html#_CPPv4N5cudaq14Kraus | -   [cudaq::qvector::back (C++    |
| Selection20kraus_operator_indexE) |     function)](a                  |
| -   [cuda                         | pi/languages/cpp_api.html#_CPPv4N |
| q::KrausSelection::KrausSelection | 5cudaq7qvector4backENSt6size_tE), |
|     (C++                          |                                   |
|     function)](a                  |   [\[1\]](api/languages/cpp_api.h |
| pi/languages/cpp_api.html#_CPPv4N | tml#_CPPv4N5cudaq7qvector4backEv) |
| 5cudaq14KrausSelection14KrausSele | -   [cudaq::qvector::begin (C++   |
| ctionENSt6size_tENSt6vectorINSt6s |     fu                            |
| ize_tEEENSt6stringENSt6size_tEb), | nction)](api/languages/cpp_api.ht |
|     [\[1\]](api/langu             | ml#_CPPv4N5cudaq7qvector5beginEv) |
| ages/cpp_api.html#_CPPv4N5cudaq14 | -   [cudaq::qvector::clear (C++   |
| KrausSelection14KrausSelectionEv) |     fu                            |
| -                                 | nction)](api/languages/cpp_api.ht |
|   [cudaq::KrausSelection::op_name | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     (C++                          | -   [cudaq::qvector::end (C++     |
|     member)](                     |                                   |
| api/languages/cpp_api.html#_CPPv4 | function)](api/languages/cpp_api. |
| N5cudaq14KrausSelection7op_nameE) | html#_CPPv4N5cudaq7qvector3endEv) |
| -   [                             | -   [cudaq::qvector::front (C++   |
| cudaq::KrausSelection::operator== |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/languages      | cudaq7qvector5frontENSt6size_tE), |
| /cpp_api.html#_CPPv4NK5cudaq14Kra |                                   |
| usSelectioneqERK14KrausSelection) |  [\[1\]](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq7qvector5frontEv) |
|    [cudaq::KrausSelection::qubits | -   [cudaq::qvector::operator=    |
|     (C++                          |     (C++                          |
|     member)]                      |     functio                       |
| (api/languages/cpp_api.html#_CPPv | n)](api/languages/cpp_api.html#_C |
| 4N5cudaq14KrausSelection6qubitsE) | PPv4N5cudaq7qvectoraSERK7qvector) |
| -   [cudaq::KrausTrajectory (C++  | -   [cudaq::qvector::operator\[\] |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |     function)                     |
| #_CPPv4N5cudaq15KrausTrajectoryE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq7qvectorixEKNSt6size_tE) |
|  [cudaq::KrausTrajectory::builder | -   [cudaq::qvector::qvector (C++ |
|     (C++                          |     function)](api/               |
|     function)](ap                 | languages/cpp_api.html#_CPPv4N5cu |
| i/languages/cpp_api.html#_CPPv4N5 | daq7qvector7qvectorENSt6size_tE), |
| cudaq15KrausTrajectory7builderEv) |     [\[1\]](a                     |
| -   [cu                           | pi/languages/cpp_api.html#_CPPv4N |
| daq::KrausTrajectory::countErrors | 5cudaq7qvector7qvectorERK5state), |
|     (C++                          |     [\[2\]](api                   |
|     function)](api/lang           | /languages/cpp_api.html#_CPPv4N5c |
| uages/cpp_api.html#_CPPv4NK5cudaq | udaq7qvector7qvectorERK7qvector), |
| 15KrausTrajectory11countErrorsEv) |     [\[3\]](ap                    |
| -   [                             | i/languages/cpp_api.html#_CPPv4N5 |
| cudaq::KrausTrajectory::isOrdered | cudaq7qvector7qvectorERR7qvector) |
|     (C++                          | -   [cudaq::qvector::size (C++    |
|     function)](api/l              |     fu                            |
| anguages/cpp_api.html#_CPPv4NK5cu | nction)](api/languages/cpp_api.ht |
| daq15KrausTrajectory9isOrderedEv) | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| -   [cudaq::                      | -   [cudaq::qvector::slice (C++   |
| KrausTrajectory::kraus_selections |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|     member)](api/languag          | tor5sliceENSt6size_tENSt6size_tE) |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | -   [cudaq::qvector::value_type   |
| ausTrajectory16kraus_selectionsE) |     (C++                          |
| -   [cudaq:                       |     typ                           |
| :KrausTrajectory::KrausTrajectory | e)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq7qvector10value_typeE) |
|     function                      | -   [cudaq::qview (C++            |
| )](api/languages/cpp_api.html#_CP |     clas                          |
| Pv4N5cudaq15KrausTrajectory15Krau | s)](api/languages/cpp_api.html#_C |
| sTrajectoryENSt6size_tENSt6vector | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| I14KrausSelectionEEdNSt6size_tE), | -   [cudaq::qview::back (C++      |
|     [\[1\]](api/languag           |     function)                     |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | ](api/languages/cpp_api.html#_CPP |
| ausTrajectory15KrausTrajectoryEv) | v4N5cudaq5qview4backENSt6size_tE) |
| -   [cudaq::Kr                    | -   [cudaq::qview::begin (C++     |
| ausTrajectory::measurement_counts |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     member)](api/languages        | html#_CPPv4N5cudaq5qview5beginEv) |
| /cpp_api.html#_CPPv4N5cudaq15Krau | -   [cudaq::qview::end (C++       |
| sTrajectory18measurement_countsE) |                                   |
| -   [cud                          |   function)](api/languages/cpp_ap |
| aq::KrausTrajectory::multiplicity | i.html#_CPPv4N5cudaq5qview3endEv) |
|     (C++                          | -   [cudaq::qview::front (C++     |
|     member)](api/lan              |     function)](                   |
| guages/cpp_api.html#_CPPv4N5cudaq | api/languages/cpp_api.html#_CPPv4 |
| 15KrausTrajectory12multiplicityE) | N5cudaq5qview5frontENSt6size_tE), |
| -   [                             |                                   |
| cudaq::KrausTrajectory::num_shots |    [\[1\]](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qview5frontEv) |
|     member)](api                  | -   [cudaq::qview::operator\[\]   |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq15KrausTrajectory9num_shotsE) |     functio                       |
| -   [c                            | n)](api/languages/cpp_api.html#_C |
| udaq::KrausTrajectory::operator== | PPv4N5cudaq5qviewixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qview::qview (C++     |
|     function)](api/languages/c    |     functio                       |
| pp_api.html#_CPPv4NK5cudaq15Kraus | n)](api/languages/cpp_api.html#_C |
| TrajectoryeqERK15KrausTrajectory) | PPv4I0EN5cudaq5qview5qviewERR1R), |
| -   [cu                           |     [\[1                          |
| daq::KrausTrajectory::probability | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq5qview5qviewERK5qview) |
|     member)](api/la               | -   [cudaq::qview::size (C++      |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q15KrausTrajectory11probabilityE) | function)](api/languages/cpp_api. |
| -   [cuda                         | html#_CPPv4NK5cudaq5qview4sizeEv) |
| q::KrausTrajectory::trajectory_id | -   [cudaq::qview::slice (C++     |
|     (C++                          |     function)](api/langua         |
|     member)](api/lang             | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| uages/cpp_api.html#_CPPv4N5cudaq1 | iew5sliceENSt6size_tENSt6size_tE) |
| 5KrausTrajectory13trajectory_idE) | -   [cudaq::qview::value_type     |
| -                                 |     (C++                          |
|   [cudaq::KrausTrajectory::weight |     t                             |
|     (C++                          | ype)](api/languages/cpp_api.html# |
|     member)](                     | _CPPv4N5cudaq5qview10value_typeE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::range (C++            |
| N5cudaq15KrausTrajectory6weightE) |     fun                           |
| -                                 | ction)](api/languages/cpp_api.htm |
|    [cudaq::KrausTrajectoryBuilder | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
|     (C++                          | orI11ElementTypeEE11ElementType), |
|     class)](                      |     [\[1\]](api/languages/cpp_    |
| api/languages/cpp_api.html#_CPPv4 | api.html#_CPPv4I0EN5cudaq5rangeEN |
| N5cudaq22KrausTrajectoryBuilderE) | St6vectorI11ElementTypeEE11Elemen |
| -   [cud                          | tType11ElementType11ElementType), |
| aq::KrausTrajectoryBuilder::build |     [                             |
|     (C++                          | \[2\]](api/languages/cpp_api.html |
|     function)](api/lang           | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::real (C++             |
| 22KrausTrajectoryBuilder5buildEv) |     type)](api/languages/         |
| -   [cud                          | cpp_api.html#_CPPv4N5cudaq4realE) |
| aq::KrausTrajectoryBuilder::setId | -   [cudaq::registry (C++         |
|     (C++                          |     type)](api/languages/cpp_     |
|     function)](api/languages/cpp  | api.html#_CPPv4N5cudaq8registryE) |
| _api.html#_CPPv4N5cudaq22KrausTra | -                                 |
| jectoryBuilder5setIdENSt6size_tE) |  [cudaq::registry::RegisteredType |
| -   [cudaq::Kraus                 |     (C++                          |
| TrajectoryBuilder::setProbability |     class)](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/languages/cpp  | 5cudaq8registry14RegisteredTypeE) |
| _api.html#_CPPv4N5cudaq22KrausTra | -   [cudaq::RemoteCapabilities    |
| jectoryBuilder14setProbabilityEd) |     (C++                          |
| -   [cudaq::Krau                  |     struc                         |
| sTrajectoryBuilder::setSelections | t)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq18RemoteCapabilitiesE) |
|     function)](api/languag        | -   [cudaq::Remot                 |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | eCapabilities::RemoteCapabilities |
| ausTrajectoryBuilder13setSelectio |     (C++                          |
| nsENSt6vectorI14KrausSelectionEE) |     function)](api/languages/cpp  |
| -   [cudaq::logical_observable    | _api.html#_CPPv4N5cudaq18RemoteCa |
|     (C++                          | pabilities18RemoteCapabilitiesEb) |
|     function)](api/languages/c    | -   [cudaq:                       |
| pp_api.html#_CPPv4IDpEN5cudaq18lo | :RemoteCapabilities::stateOverlap |
| gical_observableEvDpRR8MeasArgs), |     (C++                          |
|     [\[1\]](api/l                 |     member)](api/langua           |
| anguages/cpp_api.html#_CPPv4N5cud | ges/cpp_api.html#_CPPv4N5cudaq18R |
| aq18logical_observableERKNSt6vect | emoteCapabilities12stateOverlapE) |
| orI14measure_resultEENSt6size_tE) | -                                 |
| -   [cudaq::matrix_callback (C++  |   [cudaq::RemoteCapabilities::vqe |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     member)](                     |
| #_CPPv4N5cudaq15matrix_callbackE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::matrix_handler (C++   | N5cudaq18RemoteCapabilities3vqeE) |
|                                   | -   [cudaq::Resources (C++        |
| class)](api/languages/cpp_api.htm |     class)](api/languages/cpp_a   |
| l#_CPPv4N5cudaq14matrix_handlerE) | pi.html#_CPPv4N5cudaq9ResourcesE) |
| -   [cudaq::mat                   | -   [cudaq::run (C++              |
| rix_handler::commutation_behavior |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     struct)](api/languages/       | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| cpp_api.html#_CPPv4N5cudaq14matri | 5invoke_result_tINSt7decay_tI13Qu |
| x_handler20commutation_behaviorE) | antumKernelEEDpNSt7decay_tI4ARGSE |
| -                                 | EEEEENSt6size_tERN5cudaq11noise_m |
|    [cudaq::matrix_handler::define | odelERR13QuantumKernelDpRR4ARGS), |
|     (C++                          |     [\[1\]](api/langu             |
|     function)](a                  | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| pi/languages/cpp_api.html#_CPPv4N | daq3runENSt6vectorINSt15invoke_re |
| 5cudaq14matrix_handler6defineENSt | sult_tINSt7decay_tI13QuantumKerne |
| 6stringENSt6vectorINSt7int64_tEEE | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| RR15matrix_callbackRKNSt13unorder | ize_tERR13QuantumKernelDpRR4ARGS) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::run_async (C++        |
|                                   |     functio                       |
| [\[1\]](api/languages/cpp_api.htm | n)](api/languages/cpp_api.html#_C |
| l#_CPPv4N5cudaq14matrix_handler6d | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| efineENSt6stringENSt6vectorINSt7i | tureINSt6vectorINSt15invoke_resul |
| nt64_tEEERR15matrix_callbackRR20d | t_tINSt7decay_tI13QuantumKernelEE |
| iag_matrix_callbackRKNSt13unorder | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| ed_mapINSt6stringENSt6stringEEE), | ze_tENSt6size_tERN5cudaq11noise_m |
|     [\[2\]](                      | odelERR13QuantumKernelDpRR4ARGS), |
| api/languages/cpp_api.html#_CPPv4 |     [\[1\]](api/la                |
| N5cudaq14matrix_handler6defineENS | nguages/cpp_api.html#_CPPv4I0DpEN |
| t6stringENSt6vectorINSt7int64_tEE | 5cudaq9run_asyncENSt6futureINSt6v |
| ERR15matrix_callbackRRNSt13unorde | ectorINSt15invoke_result_tINSt7de |
| red_mapINSt6stringENSt6stringEEE) | cay_tI13QuantumKernelEEDpNSt7deca |
| -                                 | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|   [cudaq::matrix_handler::degrees | ize_tERR13QuantumKernelDpRR4ARGS) |
|     (C++                          | -   [cudaq::RuntimeTarget (C++    |
|     function)](ap                 |                                   |
| i/languages/cpp_api.html#_CPPv4NK | struct)](api/languages/cpp_api.ht |
| 5cudaq14matrix_handler7degreesEv) | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| -                                 | -   [cudaq::sample (C++           |
|  [cudaq::matrix_handler::displace |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     function)](api/language       | mpleE13sample_resultRK14sample_op |
| s/cpp_api.html#_CPPv4N5cudaq14mat | tionsRR13QuantumKernelDpRR4Args), |
| rix_handler8displaceENSt6size_tE) |     [\[1\                         |
| -   [cudaq::matrix                | ]](api/languages/cpp_api.html#_CP |
| _handler::get_expected_dimensions | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     (C++                          | esultRR13QuantumKernelDpRR4Args), |
|                                   |     [\                            |
|    function)](api/languages/cpp_a | [2\]](api/languages/cpp_api.html# |
| pi.html#_CPPv4NK5cudaq14matrix_ha | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| ndler23get_expected_dimensionsEv) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [cudaq::matrix_ha             | -   [cudaq::sample_options (C++   |
| ndler::get_parameter_descriptions |     s                             |
|     (C++                          | truct)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq14sample_optionsE) |
| function)](api/languages/cpp_api. | -   [cudaq::sample_result (C++    |
| html#_CPPv4NK5cudaq14matrix_handl |                                   |
| er26get_parameter_descriptionsEv) |  class)](api/languages/cpp_api.ht |
| -   [c                            | ml#_CPPv4N5cudaq13sample_resultE) |
| udaq::matrix_handler::instantiate | -   [cudaq::sample_result::append |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](api/languages/cpp_ |
| pi/languages/cpp_api.html#_CPPv4N | api.html#_CPPv4N5cudaq13sample_re |
| 5cudaq14matrix_handler11instantia | sult6appendERK15ExecutionResultb) |
| teENSt6stringERKNSt6vectorINSt6si | -   [cudaq::sample_result::begin  |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[1\]](                      |     function)]                    |
| api/languages/cpp_api.html#_CPPv4 | (api/languages/cpp_api.html#_CPPv |
| N5cudaq14matrix_handler11instanti | 4N5cudaq13sample_result5beginEv), |
| ateENSt6stringERRNSt6vectorINSt6s |     [\[1\]]                       |
| ize_tEEERK20commutation_behavior) | (api/languages/cpp_api.html#_CPPv |
| -   [cuda                         | 4NK5cudaq13sample_result5beginEv) |
| q::matrix_handler::matrix_handler | -   [cudaq::sample_result::cbegin |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)](                   |
| es/cpp_api.html#_CPPv4I0_NSt11ena | api/languages/cpp_api.html#_CPPv4 |
| ble_if_tINSt12is_base_of_vI16oper | NK5cudaq13sample_result6cbeginEv) |
| ator_handler1TEEbEEEN5cudaq14matr | -   [cudaq::sample_result::cend   |
| ix_handler14matrix_handlerERK1T), |     (C++                          |
|     [\[1\]](ap                    |     function)                     |
| i/languages/cpp_api.html#_CPPv4I0 | ](api/languages/cpp_api.html#_CPP |
| _NSt11enable_if_tINSt12is_base_of | v4NK5cudaq13sample_result4cendEv) |
| _vI16operator_handler1TEEbEEEN5cu | -   [cudaq::sample_result::clear  |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erERK1TRK20commutation_behavior), |     function)                     |
|     [\[2\]](api/languages/cpp_ap  | ](api/languages/cpp_api.html#_CPP |
| i.html#_CPPv4N5cudaq14matrix_hand | v4N5cudaq13sample_result5clearEv) |
| ler14matrix_handlerENSt6size_tE), | -   [cudaq::sample_result::count  |
|     [\[3\]](api/                  |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](                   |
| daq14matrix_handler14matrix_handl | api/languages/cpp_api.html#_CPPv4 |
| erENSt6stringERKNSt6vectorINSt6si | NK5cudaq13sample_result5countENSt |
| ze_tEEERK20commutation_behavior), | 11string_viewEKNSt11string_viewE) |
|     [\[4\]](api/                  | -   [                             |
| languages/cpp_api.html#_CPPv4N5cu | cudaq::sample_result::deserialize |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erENSt6stringERRNSt6vectorINSt6si |     functio                       |
| ze_tEEERK20commutation_behavior), | n)](api/languages/cpp_api.html#_C |
|     [\                            | PPv4N5cudaq13sample_result11deser |
| [5\]](api/languages/cpp_api.html# | ializeERNSt6vectorINSt6size_tEEE) |
| _CPPv4N5cudaq14matrix_handler14ma | -   [cudaq::sample_result::dump   |
| trix_handlerERK14matrix_handler), |     (C++                          |
|     [                             |     function)](api/languag        |
| \[6\]](api/languages/cpp_api.html | es/cpp_api.html#_CPPv4NK5cudaq13s |
| #_CPPv4N5cudaq14matrix_handler14m | ample_result4dumpERNSt7ostreamE), |
| atrix_handlerERR14matrix_handler) |     [\[1\]                        |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|  [cudaq::matrix_handler::momentum | v4NK5cudaq13sample_result4dumpEv) |
|     (C++                          | -   [cudaq::sample_result::end    |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     function                      |
| rix_handler8momentumENSt6size_tE) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq13sample_result3endEv), |
|    [cudaq::matrix_handler::number |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/langua         | Pv4NK5cudaq13sample_result3endEv) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -   [                             |
| atrix_handler6numberENSt6size_tE) | cudaq::sample_result::expectation |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::operator= |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     fun                           | tml#_CPPv4NK5cudaq13sample_result |
| ction)](api/languages/cpp_api.htm | 11expectationEKNSt11string_viewE) |
| l#_CPPv4I0_NSt11enable_if_tIXaant | -   [c                            |
| NSt7is_sameI1T14matrix_handlerE5v | udaq::sample_result::get_marginal |
| alueENSt12is_base_of_vI16operator |     (C++                          |
| _handler1TEEEbEEEN5cudaq14matrix_ |     function)](api/languages/cpp_ |
| handleraSER14matrix_handlerRK1T), | api.html#_CPPv4NK5cudaq13sample_r |
|     [\[1\]](api/languages         | esult12get_marginalERKNSt6vectorI |
| /cpp_api.html#_CPPv4N5cudaq14matr | NSt6size_tEEEKNSt11string_viewE), |
| ix_handleraSERK14matrix_handler), |     [\[1\]](api/languages/cpp_    |
|     [\[2\]](api/language          | api.html#_CPPv4NK5cudaq13sample_r |
| s/cpp_api.html#_CPPv4N5cudaq14mat | esult12get_marginalERRKNSt6vector |
| rix_handleraSERR14matrix_handler) | INSt6size_tEEEKNSt11string_viewE) |
| -   [                             | -   [cuda                         |
| cudaq::matrix_handler::operator== | q::sample_result::get_total_shots |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/langua         |
| /cpp_api.html#_CPPv4NK5cudaq14mat | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| rix_handlereqERK14matrix_handler) | sample_result15get_total_shotsEv) |
| -                                 | -   [cuda                         |
|    [cudaq::matrix_handler::parity | q::sample_result::has_even_parity |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     fun                           |
| ges/cpp_api.html#_CPPv4N5cudaq14m | ction)](api/languages/cpp_api.htm |
| atrix_handler6parityENSt6size_tE) | l#_CPPv4N5cudaq13sample_result15h |
| -                                 | as_even_parityENSt11string_viewE) |
|  [cudaq::matrix_handler::position | -   [cuda                         |
|     (C++                          | q::sample_result::has_expectation |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     funct                         |
| rix_handler8positionENSt6size_tE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::                      | _CPPv4NK5cudaq13sample_result15ha |
| matrix_handler::remove_definition | s_expectationEKNSt11string_viewE) |
|     (C++                          | -   [cu                           |
|     fu                            | daq::sample_result::most_probable |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14matrix_handler1 |     fun                           |
| 7remove_definitionERKNSt6stringE) | ction)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4NK5cudaq13sample_result13 |
|   [cudaq::matrix_handler::squeeze | most_probableEKNSt11string_viewE) |
|     (C++                          | -                                 |
|     function)](api/languag        | [cudaq::sample_result::operator+= |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     (C++                          |
| trix_handler7squeezeENSt6size_tE) |     function)](api/langua         |
| -   [cudaq::m                     | ges/cpp_api.html#_CPPv4N5cudaq13s |
| atrix_handler::to_diagonal_matrix | ample_resultpLERK13sample_result) |
|     (C++                          | -                                 |
|     function)](api/lang           |  [cudaq::sample_result::operator= |
| uages/cpp_api.html#_CPPv4NK5cudaq |     (C++                          |
| 14matrix_handler18to_diagonal_mat |     function)](api/langua         |
| rixERNSt13unordered_mapINSt6size_ | ges/cpp_api.html#_CPPv4N5cudaq13s |
| tENSt7int64_tEEERKNSt13unordered_ | ample_resultaSERR13sample_result) |
| mapINSt6stringENSt7complexIdEEEE) | -                                 |
| -                                 | [cudaq::sample_result::operator== |
| [cudaq::matrix_handler::to_matrix |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)                     | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ](api/languages/cpp_api.html#_CPP | ample_resulteqERK13sample_result) |
| v4NK5cudaq14matrix_handler9to_mat | -   [                             |
| rixERNSt13unordered_mapINSt6size_ | cudaq::sample_result::probability |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     function)](api/lan            |
| -                                 | guages/cpp_api.html#_CPPv4NK5cuda |
| [cudaq::matrix_handler::to_string | q13sample_result11probabilityENSt |
|     (C++                          | 11string_viewEKNSt11string_viewE) |
|     function)](api/               | -   [cud                          |
| languages/cpp_api.html#_CPPv4NK5c | aq::sample_result::register_names |
| udaq14matrix_handler9to_stringEb) |     (C++                          |
| -                                 |     function)](api/langu          |
| [cudaq::matrix_handler::unique_id | ages/cpp_api.html#_CPPv4NK5cudaq1 |
|     (C++                          | 3sample_result14register_namesEv) |
|     function)](api/               | -                                 |
| languages/cpp_api.html#_CPPv4NK5c |    [cudaq::sample_result::reorder |
| udaq14matrix_handler9unique_idEv) |     (C++                          |
| -   [cudaq:                       |     function)](api/langua         |
| :matrix_handler::\~matrix_handler | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     (C++                          | ample_result7reorderERKNSt6vector |
|     functi                        | INSt6size_tEEEKNSt11string_viewE) |
| on)](api/languages/cpp_api.html#_ | -   [cu                           |
| CPPv4N5cudaq14matrix_handlerD0Ev) | daq::sample_result::sample_result |
| -   [cudaq::matrix_op (C++        |     (C++                          |
|     type)](api/languages/cpp_a    |     func                          |
| pi.html#_CPPv4N5cudaq9matrix_opE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::matrix_op_term (C++   | #_CPPv4N5cudaq13sample_result13sa |
|                                   | mple_resultERK15ExecutionResult), |
|  type)](api/languages/cpp_api.htm |     [\[1\]](api/la                |
| l#_CPPv4N5cudaq14matrix_op_termE) | nguages/cpp_api.html#_CPPv4N5cuda |
| -                                 | q13sample_result13sample_resultER |
|    [cudaq::mdiag_operator_handler | KNSt6vectorI15ExecutionResultEE), |
|     (C++                          |                                   |
|     class)](                      |  [\[2\]](api/languages/cpp_api.ht |
| api/languages/cpp_api.html#_CPPv4 | ml#_CPPv4N5cudaq13sample_result13 |
| N5cudaq22mdiag_operator_handlerE) | sample_resultERR13sample_result), |
| -   [cudaq::measure_handle (C++   |     [                             |
|                                   | \[3\]](api/languages/cpp_api.html |
| class)](api/languages/cpp_api.htm | #_CPPv4N5cudaq13sample_result13sa |
| l#_CPPv4N5cudaq14measure_handleE) | mple_resultERR15ExecutionResult), |
| -   [cudaq::measure_result (C++   |     [\[4\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|  type)](api/languages/cpp_api.htm | 13sample_result13sample_resultEdR |
| l#_CPPv4N5cudaq14measure_resultE) | KNSt6vectorI15ExecutionResultEE), |
| -   [cudaq::mpi (C++              |     [\[5\]](api/lan               |
|     type)](api/languages          | guages/cpp_api.html#_CPPv4N5cudaq |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | 13sample_result13sample_resultEv) |
| -   [cudaq::mpi::all_gather (C++  | -                                 |
|     fu                            |  [cudaq::sample_result::serialize |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     function)](api                |
| RNSt6vectorIdEERKNSt6vectorIdEE), | /languages/cpp_api.html#_CPPv4NK5 |
|                                   | cudaq13sample_result9serializeEv) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::sample_result::size   |
| tml#_CPPv4N5cudaq3mpi10all_gather |     (C++                          |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     function)](api/languages/c    |
| -   [cudaq::mpi::all_reduce (C++  | pp_api.html#_CPPv4NK5cudaq13sampl |
|                                   | e_result4sizeEKNSt11string_viewE) |
|  function)](api/languages/cpp_api | -   [cudaq::sample_result::to_map |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     (C++                          |
| reduceE1TRK1TRK14BinaryFunction), |     function)](api/languages/cpp  |
|     [\[1\]](api/langu             | _api.html#_CPPv4NK5cudaq13sample_ |
| ages/cpp_api.html#_CPPv4I00EN5cud | result6to_mapEKNSt11string_viewE) |
| aq3mpi10all_reduceE1TRK1TRK4Func) | -   [cuda                         |
| -   [cudaq::mpi::broadcast (C++   | q::sample_result::\~sample_result |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     funct                         |
| daq3mpi9broadcastERNSt6stringEi), | ion)](api/languages/cpp_api.html# |
|     [\[1\]](api/la                | _CPPv4N5cudaq13sample_resultD0Ev) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::scalar_callback (C++  |
| q3mpi9broadcastERNSt6vectorIdEEi) |     c                             |
| -   [cudaq::mpi::finalize (C++    | lass)](api/languages/cpp_api.html |
|     f                             | #_CPPv4N5cudaq15scalar_callbackE) |
| unction)](api/languages/cpp_api.h | -   [c                            |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | udaq::scalar_callback::operator() |
| -   [cudaq::mpi::initialize (C++  |     (C++                          |
|     function                      |     function)](api/language       |
| )](api/languages/cpp_api.html#_CP | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| Pv4N5cudaq3mpi10initializeEiPPc), | alar_callbackclERKNSt13unordered_ |
|     [                             | mapINSt6stringENSt7complexIdEEEE) |
| \[1\]](api/languages/cpp_api.html | -   [                             |
| #_CPPv4N5cudaq3mpi10initializeEv) | cudaq::scalar_callback::operator= |
| -   [cudaq::mpi::is_initialized   |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function                      | pp_api.html#_CPPv4N5cudaq15scalar |
| )](api/languages/cpp_api.html#_CP | _callbackaSERK15scalar_callback), |
| Pv4N5cudaq3mpi14is_initializedEv) |     [\[1\]](api/languages/        |
| -   [cudaq::mpi::num_ranks (C++   | cpp_api.html#_CPPv4N5cudaq15scala |
|     fu                            | r_callbackaSERR15scalar_callback) |
| nction)](api/languages/cpp_api.ht | -   [cudaq:                       |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | :scalar_callback::scalar_callback |
| -   [cudaq::mpi::rank (C++        |     (C++                          |
|                                   |     function)](api/languag        |
|    function)](api/languages/cpp_a | es/cpp_api.html#_CPPv4I0_NSt11ena |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | ble_if_tINSt16is_invocable_r_vINS |
| -   [cudaq::noise_model (C++      | t7complexIdEE8CallableRKNSt13unor |
|                                   | dered_mapINSt6stringENSt7complexI |
|    class)](api/languages/cpp_api. | dEEEEEEbEEEN5cudaq15scalar_callba |
| html#_CPPv4N5cudaq11noise_modelE) | ck15scalar_callbackERR8Callable), |
| -   [cudaq::n                     |     [\[1\                         |
| oise_model::add_all_qubit_channel | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_callback15scal |
|     function)](api                | ar_callbackERK15scalar_callback), |
| /languages/cpp_api.html#_CPPv4IDp |     [\[2                          |
| EN5cudaq11noise_model21add_all_qu | \]](api/languages/cpp_api.html#_C |
| bit_channelEvRK13kraus_channeli), | PPv4N5cudaq15scalar_callback15sca |
|     [\[1\]](api/langua            | lar_callbackERR15scalar_callback) |
| ges/cpp_api.html#_CPPv4N5cudaq11n | -   [cudaq::scalar_operator (C++  |
| oise_model21add_all_qubit_channel |     c                             |
| ERKNSt6stringERK13kraus_channeli) | lass)](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq15scalar_operatorE) |
|  [cudaq::noise_model::add_channel | -                                 |
|     (C++                          | [cudaq::scalar_operator::evaluate |
|     funct                         |     (C++                          |
| ion)](api/languages/cpp_api.html# |                                   |
| _CPPv4IDpEN5cudaq11noise_model11a |    function)](api/languages/cpp_a |
| dd_channelEvRK15PredicateFuncTy), | pi.html#_CPPv4NK5cudaq15scalar_op |
|     [\[1\]](api/languages/cpp_    | erator8evaluateERKNSt13unordered_ |
| api.html#_CPPv4IDpEN5cudaq11noise | mapINSt6stringENSt7complexIdEEEE) |
| _model11add_channelEvRKNSt6vector | -   [cudaq::scalar_ope            |
| INSt6size_tEEERK13kraus_channel), | rator::get_parameter_descriptions |
|     [\[2\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     f                             |
| cudaq11noise_model11add_channelER | unction)](api/languages/cpp_api.h |
| KNSt6stringERK15PredicateFuncTy), | tml#_CPPv4NK5cudaq15scalar_operat |
|                                   | or26get_parameter_descriptionsEv) |
| [\[3\]](api/languages/cpp_api.htm | -   [cu                           |
| l#_CPPv4N5cudaq11noise_model11add | daq::scalar_operator::is_constant |
| _channelERKNSt6stringERKNSt6vecto |     (C++                          |
| rINSt6size_tEEERK13kraus_channel) |     function)](api/lang           |
| -   [cudaq::noise_model::empty    | uages/cpp_api.html#_CPPv4NK5cudaq |
|     (C++                          | 15scalar_operator11is_constantEv) |
|     function                      | -   [c                            |
| )](api/languages/cpp_api.html#_CP | udaq::scalar_operator::operator\* |
| Pv4NK5cudaq11noise_model5emptyEv) |     (C++                          |
| -                                 |     function                      |
| [cudaq::noise_model::get_channels | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormlENSt |
|     function)](api/l              | 7complexIdEERK15scalar_operator), |
| anguages/cpp_api.html#_CPPv4I0ENK |     [\[1\                         |
| 5cudaq11noise_model12get_channels | ]](api/languages/cpp_api.html#_CP |
| ENSt6vectorI13kraus_channelEERKNS | Pv4N5cudaq15scalar_operatormlENSt |
| t6vectorINSt6size_tEEERKNSt6vecto | 7complexIdEERR15scalar_operator), |
| rINSt6size_tEEERKNSt6vectorIdEE), |     [\[2\]](api/languages/cp      |
|     [\[1\]](api/languages/cpp_a   | p_api.html#_CPPv4N5cudaq15scalar_ |
| pi.html#_CPPv4NK5cudaq11noise_mod | operatormlEdRK15scalar_operator), |
| el12get_channelsERKNSt6stringERKN |     [\[3\]](api/languages/cp      |
| St6vectorINSt6size_tEEERKNSt6vect | p_api.html#_CPPv4N5cudaq15scalar_ |
| orINSt6size_tEEERKNSt6vectorIdEE) | operatormlEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
|  [cudaq::noise_model::noise_model | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormlENSt7complexIdEE), |
|     function)](api                |     [\[5\]](api/languages/cpp     |
| /languages/cpp_api.html#_CPPv4N5c | _api.html#_CPPv4NKR5cudaq15scalar |
| udaq11noise_model11noise_modelEv) | _operatormlERK15scalar_operator), |
| -   [cu                           |     [\[6\]]                       |
| daq::noise_model::PredicateFuncTy | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormlEd), |
|     type)](api/la                 |     [\[7\]](api/language          |
| nguages/cpp_api.html#_CPPv4N5cuda | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| q11noise_model15PredicateFuncTyE) | alar_operatormlENSt7complexIdEE), |
| -   [cud                          |     [\[8\]](api/languages/cp      |
| aq::noise_model::register_channel | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatormlERK15scalar_operator), |
|     function)](api/languages      |     [\[9\                         |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | ]](api/languages/cpp_api.html#_CP |
| noise_model16register_channelEvv) | Pv4NO5cudaq15scalar_operatormlEd) |
| -   [cudaq::                      | -   [cu                           |
| noise_model::requires_constructor | daq::scalar_operator::operator\*= |
|     (C++                          |     (C++                          |
|     type)](api/languages/cp       |     function)](api/languag        |
| p_api.html#_CPPv4I0DpEN5cudaq11no | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ise_model20requires_constructorE) | alar_operatormLENSt7complexIdEE), |
| -   [cudaq::noise_model_type (C++ |     [\[1\]](api/languages/c       |
|     e                             | pp_api.html#_CPPv4N5cudaq15scalar |
| num)](api/languages/cpp_api.html# | _operatormLERK15scalar_operator), |
| _CPPv4N5cudaq16noise_model_typeE) |     [\[2                          |
| -   [cudaq::no                    | \]](api/languages/cpp_api.html#_C |
| ise_model_type::amplitude_damping | PPv4N5cudaq15scalar_operatormLEd) |
|     (C++                          | -   [                             |
|     enumerator)](api/languages    | cudaq::scalar_operator::operator+ |
| /cpp_api.html#_CPPv4N5cudaq16nois |     (C++                          |
| e_model_type17amplitude_dampingE) |     function                      |
| -   [cudaq::noise_mode            | )](api/languages/cpp_api.html#_CP |
| l_type::amplitude_damping_channel | Pv4N5cudaq15scalar_operatorplENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     e                             |     [\[1\                         |
| numerator)](api/languages/cpp_api | ]](api/languages/cpp_api.html#_CP |
| .html#_CPPv4N5cudaq16noise_model_ | Pv4N5cudaq15scalar_operatorplENSt |
| type25amplitude_damping_channelE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::n                     |     [\[2\]](api/languages/cp      |
| oise_model_type::bit_flip_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatorplEdRK15scalar_operator), |
|     enumerator)](api/language     |     [\[3\]](api/languages/cp      |
| s/cpp_api.html#_CPPv4N5cudaq16noi | p_api.html#_CPPv4N5cudaq15scalar_ |
| se_model_type16bit_flip_channelE) | operatorplEdRR15scalar_operator), |
| -   [cudaq::                      |     [\[4\]](api/languages         |
| noise_model_type::depolarization1 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatorplENSt7complexIdEE), |
|     enumerator)](api/languag      |     [\[5\]](api/languages/cpp     |
| es/cpp_api.html#_CPPv4N5cudaq16no | _api.html#_CPPv4NKR5cudaq15scalar |
| ise_model_type15depolarization1E) | _operatorplERK15scalar_operator), |
| -   [cudaq::                      |     [\[6\]]                       |
| noise_model_type::depolarization2 | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatorplEd), |
|     enumerator)](api/languag      |     [\[7\]]                       |
| es/cpp_api.html#_CPPv4N5cudaq16no | (api/languages/cpp_api.html#_CPPv |
| ise_model_type15depolarization2E) | 4NKR5cudaq15scalar_operatorplEv), |
| -   [cudaq::noise_m               |     [\[8\]](api/language          |
| odel_type::depolarization_channel | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatorplENSt7complexIdEE), |
|                                   |     [\[9\]](api/languages/cp      |
|   enumerator)](api/languages/cpp_ | p_api.html#_CPPv4NO5cudaq15scalar |
| api.html#_CPPv4N5cudaq16noise_mod | _operatorplERK15scalar_operator), |
| el_type22depolarization_channelE) |     [\[10\]                       |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|  [cudaq::noise_model_type::pauli1 | v4NO5cudaq15scalar_operatorplEd), |
|     (C++                          |     [\[11\                        |
|     enumerator)](a                | ]](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4NO5cudaq15scalar_operatorplEv) |
| 5cudaq16noise_model_type6pauli1E) | -   [c                            |
| -                                 | udaq::scalar_operator::operator+= |
|  [cudaq::noise_model_type::pauli2 |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](a                | es/cpp_api.html#_CPPv4N5cudaq15sc |
| pi/languages/cpp_api.html#_CPPv4N | alar_operatorpLENSt7complexIdEE), |
| 5cudaq16noise_model_type6pauli2E) |     [\[1\]](api/languages/c       |
| -   [cudaq                        | pp_api.html#_CPPv4N5cudaq15scalar |
| ::noise_model_type::phase_damping | _operatorpLERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     enumerator)](api/langu        | \]](api/languages/cpp_api.html#_C |
| ages/cpp_api.html#_CPPv4N5cudaq16 | PPv4N5cudaq15scalar_operatorpLEd) |
| noise_model_type13phase_dampingE) | -   [                             |
| -   [cudaq::noi                   | cudaq::scalar_operator::operator- |
| se_model_type::phase_flip_channel |     (C++                          |
|     (C++                          |     function                      |
|     enumerator)](api/languages/   | )](api/languages/cpp_api.html#_CP |
| cpp_api.html#_CPPv4N5cudaq16noise | Pv4N5cudaq15scalar_operatormiENSt |
| _model_type18phase_flip_channelE) | 7complexIdEERK15scalar_operator), |
| -                                 |     [\[1\                         |
| [cudaq::noise_model_type::unknown | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormiENSt |
|     enumerator)](ap               | 7complexIdEERR15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[2\]](api/languages/cp      |
| cudaq16noise_model_type7unknownE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatormiEdRK15scalar_operator), |
| [cudaq::noise_model_type::x_error |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](ap               | operatormiEdRR15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[4\]](api/languages         |
| cudaq16noise_model_type7x_errorE) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -                                 | alar_operatormiENSt7complexIdEE), |
| [cudaq::noise_model_type::y_error |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|     enumerator)](ap               | _operatormiERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[6\]]                       |
| cudaq16noise_model_type7y_errorE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatormiEd), |
| [cudaq::noise_model_type::z_error |     [\[7\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     enumerator)](ap               | 4NKR5cudaq15scalar_operatormiEv), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[8\]](api/language          |
| cudaq16noise_model_type7z_errorE) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::num_available_gpus    | alar_operatormiENSt7complexIdEE), |
|     (C++                          |     [\[9\]](api/languages/cp      |
|     function                      | p_api.html#_CPPv4NO5cudaq15scalar |
| )](api/languages/cpp_api.html#_CP | _operatormiERK15scalar_operator), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[10\]                       |
| -   [cudaq::observe (C++          | ](api/languages/cpp_api.html#_CPP |
|     function)]                    | v4NO5cudaq15scalar_operatormiEd), |
| (api/languages/cpp_api.html#_CPPv |     [\[11\                        |
| 4I00DpEN5cudaq7observeENSt6vector | ]](api/languages/cpp_api.html#_CP |
| I14observe_resultEERR13QuantumKer | Pv4NO5cudaq15scalar_operatormiEv) |
| nelRK15SpinOpContainerDpRR4Args), | -   [c                            |
|     [\[1\]](api/languages/cpp_ap  | udaq::scalar_operator::operator-= |
| i.html#_CPPv4I0DpEN5cudaq7observe |     (C++                          |
| E14observe_resultNSt6size_tERR13Q |     function)](api/languag        |
| uantumKernelRK7spin_opDpRR4Args), | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     [\[                           | alar_operatormIENSt7complexIdEE), |
| 2\]](api/languages/cpp_api.html#_ |     [\[1\]](api/languages/c       |
| CPPv4I0DpEN5cudaq7observeE14obser | pp_api.html#_CPPv4N5cudaq15scalar |
| ve_resultRK15observe_optionsRR13Q | _operatormIERK15scalar_operator), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[2                          |
|     [\[3\]](api/lang              | \]](api/languages/cpp_api.html#_C |
| uages/cpp_api.html#_CPPv4I0DpEN5c | PPv4N5cudaq15scalar_operatormIEd) |
| udaq7observeE14observe_resultRR13 | -   [                             |
| QuantumKernelRK7spin_opDpRR4Args) | cudaq::scalar_operator::operator/ |
| -   [cudaq::observe_options (C++  |     (C++                          |
|     st                            |     function                      |
| ruct)](api/languages/cpp_api.html | )](api/languages/cpp_api.html#_CP |
| #_CPPv4N5cudaq15observe_optionsE) | Pv4N5cudaq15scalar_operatordvENSt |
| -   [cudaq::observe_result (C++   | 7complexIdEERK15scalar_operator), |
|                                   |     [\[1\                         |
| class)](api/languages/cpp_api.htm | ]](api/languages/cpp_api.html#_CP |
| l#_CPPv4N5cudaq14observe_resultE) | Pv4N5cudaq15scalar_operatordvENSt |
| -                                 | 7complexIdEERR15scalar_operator), |
|    [cudaq::observe_result::counts |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     function)](api/languages/c    | operatordvEdRK15scalar_operator), |
| pp_api.html#_CPPv4N5cudaq14observ |     [\[3\]](api/languages/cp      |
| e_result6countsERK12spin_op_term) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::observe_result::dump  | operatordvEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     function)                     | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ](api/languages/cpp_api.html#_CPP | alar_operatordvENSt7complexIdEE), |
| v4N5cudaq14observe_result4dumpEv) |     [\[5\]](api/languages/cpp     |
| -   [c                            | _api.html#_CPPv4NKR5cudaq15scalar |
| udaq::observe_result::expectation | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
| function)](api/languages/cpp_api. | 4NKR5cudaq15scalar_operatordvEd), |
| html#_CPPv4N5cudaq14observe_resul |     [\[7\]](api/language          |
| t11expectationERK12spin_op_term), | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     [\[1\]](api/la                | alar_operatordvENSt7complexIdEE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[8\]](api/languages/cp      |
| q14observe_result11expectationEv) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cuda                         | _operatordvERK15scalar_operator), |
| q::observe_result::id_coefficient |     [\[9\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/langu          | Pv4NO5cudaq15scalar_operatordvEd) |
| ages/cpp_api.html#_CPPv4N5cudaq14 | -   [c                            |
| observe_result14id_coefficientEv) | udaq::scalar_operator::operator/= |
| -   [cuda                         |     (C++                          |
| q::observe_result::observe_result |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|                                   | alar_operatordVENSt7complexIdEE), |
|   function)](api/languages/cpp_ap |     [\[1\]](api/languages/c       |
| i.html#_CPPv4N5cudaq14observe_res | pp_api.html#_CPPv4N5cudaq15scalar |
| ult14observe_resultEdRK7spin_op), | _operatordVERK15scalar_operator), |
|     [\[1\]](a                     |     [\[2                          |
| pi/languages/cpp_api.html#_CPPv4N | \]](api/languages/cpp_api.html#_C |
| 5cudaq14observe_result14observe_r | PPv4N5cudaq15scalar_operatordVEd) |
| esultEdRK7spin_op13sample_result) | -   [                             |
| -                                 | cudaq::scalar_operator::operator= |
|  [cudaq::observe_result::operator |     (C++                          |
|     double (C++                   |     function)](api/languages/c    |
|     functio                       | pp_api.html#_CPPv4N5cudaq15scalar |
| n)](api/languages/cpp_api.html#_C | _operatoraSERK15scalar_operator), |
| PPv4N5cudaq14observe_resultcvdEv) |     [\[1\]](api/languages/        |
| -                                 | cpp_api.html#_CPPv4N5cudaq15scala |
|  [cudaq::observe_result::raw_data | r_operatoraSERR15scalar_operator) |
|     (C++                          | -   [c                            |
|     function)](ap                 | udaq::scalar_operator::operator== |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq14observe_result8raw_dataEv) |     function)](api/languages/c    |
| -   [cudaq::operator_handler (C++ | pp_api.html#_CPPv4NK5cudaq15scala |
|     cl                            | r_operatoreqERK15scalar_operator) |
| ass)](api/languages/cpp_api.html# | -   [cudaq:                       |
| _CPPv4N5cudaq16operator_handlerE) | :scalar_operator::scalar_operator |
| -   [cudaq::optimizable_function  |     (C++                          |
|     (C++                          |     func                          |
|     class)                        | tion)](api/languages/cpp_api.html |
| ](api/languages/cpp_api.html#_CPP | #_CPPv4N5cudaq15scalar_operator15 |
| v4N5cudaq20optimizable_functionE) | scalar_operatorENSt7complexIdEE), |
| -   [cudaq::optimization_result   |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     type                          | scalar_operator15scalar_operatorE |
| )](api/languages/cpp_api.html#_CP | RK15scalar_callbackRRNSt13unorder |
| Pv4N5cudaq19optimization_resultE) | ed_mapINSt6stringENSt6stringEEE), |
| -   [cudaq::optimizer (C++        |     [\[2\                         |
|     class)](api/languages/cpp_a   | ]](api/languages/cpp_api.html#_CP |
| pi.html#_CPPv4N5cudaq9optimizerE) | Pv4N5cudaq15scalar_operator15scal |
| -   [cudaq::optimizer::optimize   | ar_operatorERK15scalar_operator), |
|     (C++                          |     [\[3\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq15 |
|  function)](api/languages/cpp_api | scalar_operator15scalar_operatorE |
| .html#_CPPv4N5cudaq9optimizer8opt | RR15scalar_callbackRRNSt13unorder |
| imizeEKiRR20optimizable_function) | ed_mapINSt6stringENSt6stringEEE), |
| -   [cu                           |     [\[4\                         |
| daq::optimizer::requiresGradients | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operator15scal |
|     function)](api/la             | ar_operatorERR15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[5\]](api/language          |
| q9optimizer17requiresGradientsEv) | s/cpp_api.html#_CPPv4N5cudaq15sca |
| -   [cudaq::orca (C++             | lar_operator15scalar_operatorEd), |
|     type)](api/languages/         |     [\[6\]](api/languag           |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::orca::sample (C++     | alar_operator15scalar_operatorEv) |
|     function)](api/languages/c    | -   [                             |
| pp_api.html#_CPPv4N5cudaq4orca6sa | cudaq::scalar_operator::to_matrix |
| mpleERNSt6vectorINSt6size_tEEERNS |     (C++                          |
| t6vectorINSt6size_tEEERNSt6vector |                                   |
| IdEERNSt6vectorIdEEiNSt6size_tE), |   function)](api/languages/cpp_ap |
|     [\[1\]]                       | i.html#_CPPv4NK5cudaq15scalar_ope |
| (api/languages/cpp_api.html#_CPPv | rator9to_matrixERKNSt13unordered_ |
| 4N5cudaq4orca6sampleERNSt6vectorI | mapINSt6stringENSt7complexIdEEEE) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [                             |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | cudaq::scalar_operator::to_string |
| -   [cudaq::orca::sample_async    |     (C++                          |
|     (C++                          |     function)](api/l              |
|                                   | anguages/cpp_api.html#_CPPv4NK5cu |
| function)](api/languages/cpp_api. | daq15scalar_operator9to_stringEv) |
| html#_CPPv4N5cudaq4orca12sample_a | -   [cudaq::s                     |
| syncERNSt6vectorINSt6size_tEEERNS | calar_operator::\~scalar_operator |
| t6vectorINSt6size_tEEERNSt6vector |     (C++                          |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     functio                       |
|     [\[1\]](api/la                | n)](api/languages/cpp_api.html#_C |
| nguages/cpp_api.html#_CPPv4N5cuda | PPv4N5cudaq15scalar_operatorD0Ev) |
| q4orca12sample_asyncERNSt6vectorI | -   [cudaq::set_noise (C++        |
| NSt6size_tEEERNSt6vectorINSt6size |     function)](api/langu          |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | ages/cpp_api.html#_CPPv4N5cudaq9s |
| -   [cudaq::OrcaRemoteRESTQPU     | et_noiseERKN5cudaq11noise_modelE) |
|     (C++                          | -   [cudaq::set_random_seed (C++  |
|     cla                           |     function)](api/               |
| ss)](api/languages/cpp_api.html#_ | languages/cpp_api.html#_CPPv4N5cu |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | daq15set_random_seedENSt6size_tE) |
| -   [cudaq::pauli1 (C++           | -   [cudaq::simulation_precision  |
|     class)](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq6pauli1E) |     enum)                         |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|    [cudaq::pauli1::num_parameters | v4N5cudaq20simulation_precisionE) |
|     (C++                          | -   [                             |
|     member)]                      | cudaq::simulation_precision::fp32 |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq6pauli114num_parametersE) |     enumerator)](api              |
| -   [cudaq::pauli1::num_targets   | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq20simulation_precision4fp32E) |
|     membe                         | -   [                             |
| r)](api/languages/cpp_api.html#_C | cudaq::simulation_precision::fp64 |
| PPv4N5cudaq6pauli111num_targetsE) |     (C++                          |
| -   [cudaq::pauli1::pauli1 (C++   |     enumerator)](api              |
|     function)](api/languages/cpp_ | /languages/cpp_api.html#_CPPv4N5c |
| api.html#_CPPv4N5cudaq6pauli16pau | udaq20simulation_precision4fp64E) |
| li1ERKNSt6vectorIN5cudaq4realEEE) | -   [cudaq::SimulationState (C++  |
| -   [cudaq::pauli2 (C++           |     c                             |
|     class)](api/languages/cp      | lass)](api/languages/cpp_api.html |
| p_api.html#_CPPv4N5cudaq6pauli2E) | #_CPPv4N5cudaq15SimulationStateE) |
| -                                 | -   [                             |
|    [cudaq::pauli2::num_parameters | cudaq::SimulationState::precision |
|     (C++                          |     (C++                          |
|     member)]                      |     enum)](api                    |
| (api/languages/cpp_api.html#_CPPv | /languages/cpp_api.html#_CPPv4N5c |
| 4N5cudaq6pauli214num_parametersE) | udaq15SimulationState9precisionE) |
| -   [cudaq::pauli2::num_targets   | -   [cudaq:                       |
|     (C++                          | :SimulationState::precision::fp32 |
|     membe                         |     (C++                          |
| r)](api/languages/cpp_api.html#_C |     enumerator)](api/lang         |
| PPv4N5cudaq6pauli211num_targetsE) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [cudaq::pauli2::pauli2 (C++   | 5SimulationState9precision4fp32E) |
|     function)](api/languages/cpp_ | -   [cudaq:                       |
| api.html#_CPPv4N5cudaq6pauli26pau | :SimulationState::precision::fp64 |
| li2ERKNSt6vectorIN5cudaq4realEEE) |     (C++                          |
| -   [cudaq::phase_damping (C++    |     enumerator)](api/lang         |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|  class)](api/languages/cpp_api.ht | 5SimulationState9precision4fp64E) |
| ml#_CPPv4N5cudaq13phase_dampingE) | -                                 |
| -   [cud                          |   [cudaq::SimulationState::Tensor |
| aq::phase_damping::num_parameters |     (C++                          |
|     (C++                          |     struct)](                     |
|     member)](api/lan              | api/languages/cpp_api.html#_CPPv4 |
| guages/cpp_api.html#_CPPv4N5cudaq | N5cudaq15SimulationState6TensorE) |
| 13phase_damping14num_parametersE) | -   [cudaq::spin_handler (C++     |
| -   [                             |                                   |
| cudaq::phase_damping::num_targets |   class)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq12spin_handlerE) |
|     member)](api/                 | -   [cudaq:                       |
| languages/cpp_api.html#_CPPv4N5cu | :spin_handler::to_diagonal_matrix |
| daq13phase_damping11num_targetsE) |     (C++                          |
| -   [cudaq::phase_flip_channel    |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4NK5cud |
|     clas                          | aq12spin_handler18to_diagonal_mat |
| s)](api/languages/cpp_api.html#_C | rixERNSt13unordered_mapINSt6size_ |
| PPv4N5cudaq18phase_flip_channelE) | tENSt7int64_tEEERKNSt13unordered_ |
| -   [cudaq::p                     | mapINSt6stringENSt7complexIdEEEE) |
| hase_flip_channel::num_parameters | -                                 |
|     (C++                          |   [cudaq::spin_handler::to_matrix |
|     member)](api/language         |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq18pha |     function                      |
| se_flip_channel14num_parametersE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq                        | Pv4N5cudaq12spin_handler9to_matri |
| ::phase_flip_channel::num_targets | xERKNSt6stringENSt7complexIdEEb), |
|     (C++                          |     [\[1                          |
|     member)](api/langu            | \]](api/languages/cpp_api.html#_C |
| ages/cpp_api.html#_CPPv4N5cudaq18 | PPv4NK5cudaq12spin_handler9to_mat |
| phase_flip_channel11num_targetsE) | rixERNSt13unordered_mapINSt6size_ |
| -   [cudaq::product_op (C++       | tENSt7int64_tEEERKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|  class)](api/languages/cpp_api.ht | -   [cuda                         |
| ml#_CPPv4I0EN5cudaq10product_opE) | q::spin_handler::to_sparse_matrix |
| -   [cudaq::product_op::begin     |     (C++                          |
|     (C++                          |     function)](api/               |
|     functio                       | languages/cpp_api.html#_CPPv4N5cu |
| n)](api/languages/cpp_api.html#_C | daq12spin_handler16to_sparse_matr |
| PPv4NK5cudaq10product_op5beginEv) | ixERKNSt6stringENSt7complexIdEEb) |
| -                                 | -                                 |
|  [cudaq::product_op::canonicalize |   [cudaq::spin_handler::to_string |
|     (C++                          |     (C++                          |
|     func                          |     function)](ap                 |
| tion)](api/languages/cpp_api.html | i/languages/cpp_api.html#_CPPv4NK |
| #_CPPv4N5cudaq10product_op12canon | 5cudaq12spin_handler9to_stringEb) |
| icalizeERKNSt3setINSt6size_tEEE), | -                                 |
|     [\[1\]](api                   |   [cudaq::spin_handler::unique_id |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq10product_op12canonicalizeEv) |     function)](ap                 |
| -   [                             | i/languages/cpp_api.html#_CPPv4NK |
| cudaq::product_op::const_iterator | 5cudaq12spin_handler9unique_idEv) |
|     (C++                          | -   [cudaq::spin_op (C++          |
|     struct)](api/                 |     type)](api/languages/cpp      |
| languages/cpp_api.html#_CPPv4N5cu | _api.html#_CPPv4N5cudaq7spin_opE) |
| daq10product_op14const_iteratorE) | -   [cudaq::spin_op_term (C++     |
| -   [cudaq::product_o             |                                   |
| p::const_iterator::const_iterator |    type)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq12spin_op_termE) |
|     fu                            | -   [cudaq::state (C++            |
| nction)](api/languages/cpp_api.ht |     class)](api/languages/c       |
| ml#_CPPv4N5cudaq10product_op14con | pp_api.html#_CPPv4N5cudaq5stateE) |
| st_iterator14const_iteratorEPK10p | -   [cudaq::state::amplitude (C++ |
| roduct_opI9HandlerTyENSt6size_tE) |     function)](api/lang           |
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
