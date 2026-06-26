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
| -   [cachedCompiledModule()       | -   [cudaq::product_o             |
|     (cudaq.PyKernelDecorator      | p::const_iterator::const_iterator |
|     method)](api/langu            |     (C++                          |
| ages/python_api.html#cudaq.PyKern |     fu                            |
| elDecorator.cachedCompiledModule) | nction)](api/languages/cpp_api.ht |
| -   [canonicalize                 | ml#_CPPv4N5cudaq10product_op14con |
|     (cu                           | st_iterator14const_iteratorEPK10p |
| daq.operators.boson.BosonOperator | roduct_opI9HandlerTyENSt6size_tE) |
|     attribute)](api/languages     | -   [cudaq::produ                 |
| /python_api.html#cudaq.operators. | ct_op::const_iterator::operator!= |
| boson.BosonOperator.canonicalize) |     (C++                          |
|     -   [(cudaq.                  |     fun                           |
| operators.boson.BosonOperatorTerm | ction)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4NK5cudaq10product_op14con |
|     attribute)](api/languages/pyt | st_iteratorneERK14const_iterator) |
| hon_api.html#cudaq.operators.boso | -   [cudaq::produ                 |
| n.BosonOperatorTerm.canonicalize) | ct_op::const_iterator::operator\* |
|     -   [(cudaq.                  |     (C++                          |
| operators.fermion.FermionOperator |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|     attribute)](api/languages/pyt | 10product_op14const_iteratormlEv) |
| hon_api.html#cudaq.operators.ferm | -   [cudaq::produ                 |
| ion.FermionOperator.canonicalize) | ct_op::const_iterator::operator++ |
|     -   [(cudaq.oper              |     (C++                          |
| ators.fermion.FermionOperatorTerm |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
| attribute)](api/languages/python_ | 0product_op14const_iteratorppEi), |
| api.html#cudaq.operators.fermion. |     [\[1\]](api/lan               |
| FermionOperatorTerm.canonicalize) | guages/cpp_api.html#_CPPv4N5cudaq |
|     -                             | 10product_op14const_iteratorppEv) |
|  [(cudaq.operators.MatrixOperator | -   [cudaq::produc                |
|         attribute)](api/lang      | t_op::const_iterator::operator\-- |
| uages/python_api.html#cudaq.opera |     (C++                          |
| tors.MatrixOperator.canonicalize) |     function)](api/lang           |
|     -   [(c                       | uages/cpp_api.html#_CPPv4N5cudaq1 |
| udaq.operators.MatrixOperatorTerm | 0product_op14const_iteratormmEi), |
|         attribute)](api/language  |     [\[1\]](api/lan               |
| s/python_api.html#cudaq.operators | guages/cpp_api.html#_CPPv4N5cudaq |
| .MatrixOperatorTerm.canonicalize) | 10product_op14const_iteratormmEv) |
|     -   [(                        | -   [cudaq::produc                |
| cudaq.operators.spin.SpinOperator | t_op::const_iterator::operator-\> |
|         attribute)](api/languag   |     (C++                          |
| es/python_api.html#cudaq.operator |     function)](api/lan            |
| s.spin.SpinOperator.canonicalize) | guages/cpp_api.html#_CPPv4N5cudaq |
|     -   [(cuda                    | 10product_op14const_iteratorptEv) |
| q.operators.spin.SpinOperatorTerm | -   [cudaq::produ                 |
|                                   | ct_op::const_iterator::operator== |
|       attribute)](api/languages/p |     (C++                          |
| ython_api.html#cudaq.operators.sp |     fun                           |
| in.SpinOperatorTerm.canonicalize) | ction)](api/languages/cpp_api.htm |
| -   [captured_variables()         | l#_CPPv4NK5cudaq10product_op14con |
|     (cudaq.PyKernelDecorator      | st_iteratoreqERK14const_iterator) |
|     method)](api/lan              | -   [cudaq::product_op::degrees   |
| guages/python_api.html#cudaq.PyKe |     (C++                          |
| rnelDecorator.captured_variables) |     function)                     |
| -   [CentralDifference (class in  | ](api/languages/cpp_api.html#_CPP |
|     cudaq.gradients)              | v4NK5cudaq10product_op7degreesEv) |
| ](api/languages/python_api.html#c | -   [cudaq::product_op::dump (C++ |
| udaq.gradients.CentralDifference) |     functi                        |
| -   [channel                      | on)](api/languages/cpp_api.html#_ |
|     (cudaq.ptsbe.TraceInstruction | CPPv4NK5cudaq10product_op4dumpEv) |
|     property)](a                  | -   [cudaq::product_op::end (C++  |
| pi/languages/python_api.html#cuda |     funct                         |
| q.ptsbe.TraceInstruction.channel) | ion)](api/languages/cpp_api.html# |
| -   [circuit_location             | _CPPv4NK5cudaq10product_op3endEv) |
|     (cudaq.ptsbe.KrausSelection   | -   [c                            |
|     property)](api/lang           | udaq::product_op::get_coefficient |
| uages/python_api.html#cudaq.ptsbe |     (C++                          |
| .KrausSelection.circuit_location) |     function)](api/lan            |
| -   [clear (cudaq.Resources       | guages/cpp_api.html#_CPPv4NK5cuda |
|                                   | q10product_op15get_coefficientEv) |
|   attribute)](api/languages/pytho | -                                 |
| n_api.html#cudaq.Resources.clear) |   [cudaq::product_op::get_term_id |
|     -   [(cudaq.SampleResult      |     (C++                          |
|         a                         |     function)](api                |
| ttribute)](api/languages/python_a | /languages/cpp_api.html#_CPPv4NK5 |
| pi.html#cudaq.SampleResult.clear) | cudaq10product_op11get_term_idEv) |
| -   [COBYLA (class in             | -                                 |
|     cudaq.o                       |   [cudaq::product_op::is_identity |
| ptimizers)](api/languages/python_ |     (C++                          |
| api.html#cudaq.optimizers.COBYLA) |     function)](api                |
| -   [coefficient                  | /languages/cpp_api.html#_CPPv4NK5 |
|     (cudaq.                       | cudaq10product_op11is_identityEv) |
| operators.boson.BosonOperatorTerm | -   [cudaq::product_op::num_ops   |
|     property)](api/languages/py   |     (C++                          |
| thon_api.html#cudaq.operators.bos |     function)                     |
| on.BosonOperatorTerm.coefficient) | ](api/languages/cpp_api.html#_CPP |
|     -   [(cudaq.oper              | v4NK5cudaq10product_op7num_opsEv) |
| ators.fermion.FermionOperatorTerm | -                                 |
|                                   |    [cudaq::product_op::operator\* |
|   property)](api/languages/python |     (C++                          |
| _api.html#cudaq.operators.fermion |     function)](api/languages/     |
| .FermionOperatorTerm.coefficient) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(c                       | oduct_opmlE10product_opI1TERK15sc |
| udaq.operators.MatrixOperatorTerm | alar_operatorRK10product_opI1TE), |
|         property)](api/languag    |     [\[1\]](api/languages/        |
| es/python_api.html#cudaq.operator | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| s.MatrixOperatorTerm.coefficient) | oduct_opmlE10product_opI1TERK15sc |
|     -   [(cuda                    | alar_operatorRR10product_opI1TE), |
| q.operators.spin.SpinOperatorTerm |     [\[2\]](api/languages/        |
|         property)](api/languages/ | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| python_api.html#cudaq.operators.s | oduct_opmlE10product_opI1TERR15sc |
| pin.SpinOperatorTerm.coefficient) | alar_operatorRK10product_opI1TE), |
| -   [col_count                    |     [\[3\]](api/languages/        |
|     (cudaq.KrausOperator          | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     prope                         | oduct_opmlE10product_opI1TERR15sc |
| rty)](api/languages/python_api.ht | alar_operatorRR10product_opI1TE), |
| ml#cudaq.KrausOperator.col_count) |     [\[4\]](api/                  |
| -   [compile()                    | languages/cpp_api.html#_CPPv4I0EN |
|     (cudaq.PyKernelDecorator      | 5cudaq10product_opmlE6sum_opI1TER |
|     metho                         | K15scalar_operatorRK6sum_opI1TE), |
| d)](api/languages/python_api.html |     [\[5\]](api/                  |
| #cudaq.PyKernelDecorator.compile) | languages/cpp_api.html#_CPPv4I0EN |
| -   [ComplexMatrix (class in      | 5cudaq10product_opmlE6sum_opI1TER |
|     cudaq)](api/languages/pyt     | K15scalar_operatorRR6sum_opI1TE), |
| hon_api.html#cudaq.ComplexMatrix) |     [\[6\]](api/                  |
| -   [compute                      | languages/cpp_api.html#_CPPv4I0EN |
|     (                             | 5cudaq10product_opmlE6sum_opI1TER |
| cudaq.gradients.CentralDifference | R15scalar_operatorRK6sum_opI1TE), |
|     attribute)](api/la            |     [\[7\]](api/                  |
| nguages/python_api.html#cudaq.gra | languages/cpp_api.html#_CPPv4I0EN |
| dients.CentralDifference.compute) | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [(                        | R15scalar_operatorRR6sum_opI1TE), |
| cudaq.gradients.ForwardDifference |     [\[8\]](api/languages         |
|         attribute)](api/la        | /cpp_api.html#_CPPv4NK5cudaq10pro |
| nguages/python_api.html#cudaq.gra | duct_opmlERK6sum_opI9HandlerTyE), |
| dients.ForwardDifference.compute) |     [\[9\]](api/languages/cpp_a   |
|     -                             | pi.html#_CPPv4NKR5cudaq10product_ |
|  [(cudaq.gradients.ParameterShift | opmlERK10product_opI9HandlerTyE), |
|         attribute)](api           |     [\[10\]](api/language         |
| /languages/python_api.html#cudaq. | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| gradients.ParameterShift.compute) | roduct_opmlERK15scalar_operator), |
| -   [const()                      |     [\[11\]](api/languages/cpp_a  |
|                                   | pi.html#_CPPv4NKR5cudaq10product_ |
|   (cudaq.operators.ScalarOperator | opmlERR10product_opI9HandlerTyE), |
|     class                         |     [\[12\]](api/language         |
|     method)](a                    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| pi/languages/python_api.html#cuda | roduct_opmlERR15scalar_operator), |
| q.operators.ScalarOperator.const) |     [\[13\]](api/languages/cpp_   |
| -   [controls                     | api.html#_CPPv4NO5cudaq10product_ |
|     (cudaq.ptsbe.TraceInstruction | opmlERK10product_opI9HandlerTyE), |
|     property)](ap                 |     [\[14\]](api/languag          |
| i/languages/python_api.html#cudaq | es/cpp_api.html#_CPPv4NO5cudaq10p |
| .ptsbe.TraceInstruction.controls) | roduct_opmlERK15scalar_operator), |
| -   [copy                         |     [\[15\]](api/languages/cpp_   |
|     (cu                           | api.html#_CPPv4NO5cudaq10product_ |
| daq.operators.boson.BosonOperator | opmlERR10product_opI9HandlerTyE), |
|     attribute)](api/l             |     [\[16\]](api/langua           |
| anguages/python_api.html#cudaq.op | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| erators.boson.BosonOperator.copy) | product_opmlERR15scalar_operator) |
|     -   [(cudaq.                  | -                                 |
| operators.boson.BosonOperatorTerm |   [cudaq::product_op::operator\*= |
|         attribute)](api/langu     |     (C++                          |
| ages/python_api.html#cudaq.operat |     function)](api/languages/cpp  |
| ors.boson.BosonOperatorTerm.copy) | _api.html#_CPPv4N5cudaq10product_ |
|     -   [(cudaq.                  | opmLERK10product_opI9HandlerTyE), |
| operators.fermion.FermionOperator |     [\[1\]](api/langua            |
|         attribute)](api/langu     | ges/cpp_api.html#_CPPv4N5cudaq10p |
| ages/python_api.html#cudaq.operat | roduct_opmLERK15scalar_operator), |
| ors.fermion.FermionOperator.copy) |     [\[2\]](api/languages/cp      |
|     -   [(cudaq.oper              | p_api.html#_CPPv4N5cudaq10product |
| ators.fermion.FermionOperatorTerm | _opmLERR10product_opI9HandlerTyE) |
|         attribute)](api/languages | -   [cudaq::product_op::operator+ |
| /python_api.html#cudaq.operators. |     (C++                          |
| fermion.FermionOperatorTerm.copy) |     function)](api/langu          |
|     -                             | ages/cpp_api.html#_CPPv4I0EN5cuda |
|  [(cudaq.operators.MatrixOperator | q10product_opplE6sum_opI1TERK15sc |
|         attribute)](              | alar_operatorRK10product_opI1TE), |
| api/languages/python_api.html#cud |     [\[1\]](api/                  |
| aq.operators.MatrixOperator.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(c                       | 5cudaq10product_opplE6sum_opI1TER |
| udaq.operators.MatrixOperatorTerm | K15scalar_operatorRK6sum_opI1TE), |
|         attribute)](api/          |     [\[2\]](api/langu             |
| languages/python_api.html#cudaq.o | ages/cpp_api.html#_CPPv4I0EN5cuda |
| perators.MatrixOperatorTerm.copy) | q10product_opplE6sum_opI1TERK15sc |
|     -   [(                        | alar_operatorRR10product_opI1TE), |
| cudaq.operators.spin.SpinOperator |     [\[3\]](api/                  |
|         attribute)](api           | languages/cpp_api.html#_CPPv4I0EN |
| /languages/python_api.html#cudaq. | 5cudaq10product_opplE6sum_opI1TER |
| operators.spin.SpinOperator.copy) | K15scalar_operatorRR6sum_opI1TE), |
|     -   [(cuda                    |     [\[4\]](api/langu             |
| q.operators.spin.SpinOperatorTerm | ages/cpp_api.html#_CPPv4I0EN5cuda |
|         attribute)](api/lan       | q10product_opplE6sum_opI1TERR15sc |
| guages/python_api.html#cudaq.oper | alar_operatorRK10product_opI1TE), |
| ators.spin.SpinOperatorTerm.copy) |     [\[5\]](api/                  |
| -   [count (cudaq.Resources       | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opplE6sum_opI1TER |
|   attribute)](api/languages/pytho | R15scalar_operatorRK6sum_opI1TE), |
| n_api.html#cudaq.Resources.count) |     [\[6\]](api/langu             |
|     -   [(cudaq.SampleResult      | ages/cpp_api.html#_CPPv4I0EN5cuda |
|         a                         | q10product_opplE6sum_opI1TERR15sc |
| ttribute)](api/languages/python_a | alar_operatorRR10product_opI1TE), |
| pi.html#cudaq.SampleResult.count) |     [\[7\]](api/                  |
| -   [count_controls               | languages/cpp_api.html#_CPPv4I0EN |
|     (cudaq.Resources              | 5cudaq10product_opplE6sum_opI1TER |
|     attribu                       | R15scalar_operatorRR6sum_opI1TE), |
| te)](api/languages/python_api.htm |     [\[8\]](api/languages/cpp_a   |
| l#cudaq.Resources.count_controls) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [count_instructions           | opplERK10product_opI9HandlerTyE), |
|                                   |     [\[9\]](api/language          |
|   (cudaq.ptsbe.PTSBEExecutionData | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     attribute)](api/languages/    | roduct_opplERK15scalar_operator), |
| python_api.html#cudaq.ptsbe.PTSBE |     [\[10\]](api/languages/       |
| ExecutionData.count_instructions) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [counts (cudaq.ObserveResult  | duct_opplERK6sum_opI9HandlerTyE), |
|     att                           |     [\[11\]](api/languages/cpp_a  |
| ribute)](api/languages/python_api | pi.html#_CPPv4NKR5cudaq10product_ |
| .html#cudaq.ObserveResult.counts) | opplERR10product_opI9HandlerTyE), |
| -   [csr_spmatrix (C++            |     [\[12\]](api/language         |
|     type)](api/languages/c        | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| pp_api.html#_CPPv412csr_spmatrix) | roduct_opplERR15scalar_operator), |
| -   cudaq                         |     [\[13\]](api/languages/       |
|     -   [module](api/langua       | cpp_api.html#_CPPv4NKR5cudaq10pro |
| ges/python_api.html#module-cudaq) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cudaq (C++                   |     [\[                           |
|     type)](api/lan                | 14\]](api/languages/cpp_api.html# |
| guages/cpp_api.html#_CPPv45cudaq) | _CPPv4NKR5cudaq10product_opplEv), |
| -   [cudaq.apply_noise() (in      |     [\[15\]](api/languages/cpp_   |
|     module                        | api.html#_CPPv4NO5cudaq10product_ |
|     cudaq)](api/languages/python_ | opplERK10product_opI9HandlerTyE), |
| api.html#cudaq.cudaq.apply_noise) |     [\[16\]](api/languag          |
| -   cudaq.boson                   | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     -   [module](api/languages/py | roduct_opplERK15scalar_operator), |
| thon_api.html#module-cudaq.boson) |     [\[17\]](api/languages        |
| -   cudaq.fermion                 | /cpp_api.html#_CPPv4NO5cudaq10pro |
|                                   | duct_opplERK6sum_opI9HandlerTyE), |
|   -   [module](api/languages/pyth |     [\[18\]](api/languages/cpp_   |
| on_api.html#module-cudaq.fermion) | api.html#_CPPv4NO5cudaq10product_ |
| -   cudaq.operators.custom        | opplERR10product_opI9HandlerTyE), |
|     -   [mo                       |     [\[19\]](api/languag          |
| dule](api/languages/python_api.ht | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ml#module-cudaq.operators.custom) | roduct_opplERR15scalar_operator), |
| -   cudaq.spin                    |     [\[20\]](api/languages        |
|     -   [module](api/languages/p  | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ython_api.html#module-cudaq.spin) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cudaq::amplitude_damping     |     [                             |
|     (C++                          | \[21\]](api/languages/cpp_api.htm |
|     cla                           | l#_CPPv4NO5cudaq10product_opplEv) |
| ss)](api/languages/cpp_api.html#_ | -   [cudaq::product_op::operator- |
| CPPv4N5cudaq17amplitude_dampingE) |     (C++                          |
| -                                 |     function)](api/langu          |
| [cudaq::amplitude_damping_channel | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERK15sc |
|     class)](api                   | alar_operatorRK10product_opI1TE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[1\]](api/                  |
| udaq25amplitude_damping_channelE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::amplitud              | 5cudaq10product_opmiE6sum_opI1TER |
| e_damping_channel::num_parameters | K15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[2\]](api/langu             |
|     member)](api/languages/cpp_a  | ages/cpp_api.html#_CPPv4I0EN5cuda |
| pi.html#_CPPv4N5cudaq25amplitude_ | q10product_opmiE6sum_opI1TERK15sc |
| damping_channel14num_parametersE) | alar_operatorRR10product_opI1TE), |
| -   [cudaq::ampli                 |     [\[3\]](api/                  |
| tude_damping_channel::num_targets | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     member)](api/languages/cp     | K15scalar_operatorRR6sum_opI1TE), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [\[4\]](api/langu             |
| de_damping_channel11num_targetsE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::AnalogRemoteRESTQPU   | q10product_opmiE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRK10product_opI1TE), |
|     class                         |     [\[5\]](api/                  |
| )](api/languages/cpp_api.html#_CP | languages/cpp_api.html#_CPPv4I0EN |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::apply_noise (C++      | R15scalar_operatorRK6sum_opI1TE), |
|     function)](api/               |     [\[6\]](api/langu             |
| languages/cpp_api.html#_CPPv4I0Dp | ages/cpp_api.html#_CPPv4I0EN5cuda |
| EN5cudaq11apply_noiseEvDpRR4Args) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq::async_result (C++     | alar_operatorRR10product_opI1TE), |
|     c                             |     [\[7\]](api/                  |
| lass)](api/languages/cpp_api.html | languages/cpp_api.html#_CPPv4I0EN |
| #_CPPv4I0EN5cudaq12async_resultE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::async_result::get     | R15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[8\]](api/languages/cpp_a   |
|     functi                        | pi.html#_CPPv4NKR5cudaq10product_ |
| on)](api/languages/cpp_api.html#_ | opmiERK10product_opI9HandlerTyE), |
| CPPv4N5cudaq12async_result3getEv) |     [\[9\]](api/language          |
| -   [cudaq::async_sample_result   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opmiERK15scalar_operator), |
|     type                          |     [\[10\]](api/languages/       |
| )](api/languages/cpp_api.html#_CP | cpp_api.html#_CPPv4NKR5cudaq10pro |
| Pv4N5cudaq19async_sample_resultE) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::BaseRemoteRESTQPU     |     [\[11\]](api/languages/cpp_a  |
|     (C++                          | pi.html#_CPPv4NKR5cudaq10product_ |
|     cla                           | opmiERR10product_opI9HandlerTyE), |
| ss)](api/languages/cpp_api.html#_ |     [\[12\]](api/language         |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::bit_flip_channel (C++ | roduct_opmiERR15scalar_operator), |
|     cl                            |     [\[13\]](api/languages/       |
| ass)](api/languages/cpp_api.html# | cpp_api.html#_CPPv4NKR5cudaq10pro |
| _CPPv4N5cudaq16bit_flip_channelE) | duct_opmiERR6sum_opI9HandlerTyE), |
| -   [cudaq:                       |     [\[                           |
| :bit_flip_channel::num_parameters | 14\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NKR5cudaq10product_opmiEv), |
|     member)](api/langua           |     [\[15\]](api/languages/cpp_   |
| ges/cpp_api.html#_CPPv4N5cudaq16b | api.html#_CPPv4NO5cudaq10product_ |
| it_flip_channel14num_parametersE) | opmiERK10product_opI9HandlerTyE), |
| -   [cud                          |     [\[16\]](api/languag          |
| aq::bit_flip_channel::num_targets | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmiERK15scalar_operator), |
|     member)](api/lan              |     [\[17\]](api/languages        |
| guages/cpp_api.html#_CPPv4N5cudaq | /cpp_api.html#_CPPv4NO5cudaq10pro |
| 16bit_flip_channel11num_targetsE) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::boson_handler (C++    |     [\[18\]](api/languages/cpp_   |
|                                   | api.html#_CPPv4NO5cudaq10product_ |
|  class)](api/languages/cpp_api.ht | opmiERR10product_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13boson_handlerE) |     [\[19\]](api/languag          |
| -   [cudaq::boson_op (C++         | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     type)](api/languages/cpp_     | roduct_opmiERR15scalar_operator), |
| api.html#_CPPv4N5cudaq8boson_opE) |     [\[20\]](api/languages        |
| -   [cudaq::boson_op_term (C++    | /cpp_api.html#_CPPv4NO5cudaq10pro |
|                                   | duct_opmiERR6sum_opI9HandlerTyE), |
|   type)](api/languages/cpp_api.ht |     [                             |
| ml#_CPPv4N5cudaq13boson_op_termE) | \[21\]](api/languages/cpp_api.htm |
| -   [cudaq::CodeGenConfig (C++    | l#_CPPv4NO5cudaq10product_opmiEv) |
|                                   | -   [cudaq::product_op::operator/ |
| struct)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     function)](api/language       |
| -   [cudaq::commutation_relations | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opdvERK15scalar_operator), |
|     struct)]                      |     [\[1\]](api/language          |
| (api/languages/cpp_api.html#_CPPv | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| 4N5cudaq21commutation_relationsE) | roduct_opdvERR15scalar_operator), |
| -   [cudaq::complex (C++          |     [\[2\]](api/languag           |
|     type)](api/languages/cpp      | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _api.html#_CPPv4N5cudaq7complexE) | roduct_opdvERK15scalar_operator), |
| -   [cudaq::complex_matrix (C++   |     [\[3\]](api/langua            |
|                                   | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| class)](api/languages/cpp_api.htm | product_opdvERR15scalar_operator) |
| l#_CPPv4N5cudaq14complex_matrixE) | -                                 |
| -                                 |    [cudaq::product_op::operator/= |
|   [cudaq::complex_matrix::adjoint |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](a                  | ages/cpp_api.html#_CPPv4N5cudaq10 |
| pi/languages/cpp_api.html#_CPPv4N | product_opdVERK15scalar_operator) |
| 5cudaq14complex_matrix7adjointEv) | -   [cudaq::product_op::operator= |
| -   [cudaq::                      |     (C++                          |
| complex_matrix::diagonal_elements |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4I00EN |
|     function)](api/languages      | 5cudaq10product_opaSER10product_o |
| /cpp_api.html#_CPPv4NK5cudaq14com | pI9HandlerTyERK10product_opI1TE), |
| plex_matrix17diagonal_elementsEi) |     [\[1\]](api/languages/cpp     |
| -   [cudaq::complex_matrix::dump  | _api.html#_CPPv4N5cudaq10product_ |
|     (C++                          | opaSERK10product_opI9HandlerTyE), |
|     function)](api/language       |     [\[2\]](api/languages/cp      |
| s/cpp_api.html#_CPPv4NK5cudaq14co | p_api.html#_CPPv4N5cudaq10product |
| mplex_matrix4dumpERNSt7ostreamE), | _opaSERR10product_opI9HandlerTyE) |
|     [\[1\]]                       | -                                 |
| (api/languages/cpp_api.html#_CPPv |    [cudaq::product_op::operator== |
| 4NK5cudaq14complex_matrix4dumpEv) |     (C++                          |
| -   [c                            |     function)](api/languages/cpp  |
| udaq::complex_matrix::eigenvalues | _api.html#_CPPv4NK5cudaq10product |
|     (C++                          | _opeqERK10product_opI9HandlerTyE) |
|     function)](api/lan            | -                                 |
| guages/cpp_api.html#_CPPv4NK5cuda |  [cudaq::product_op::operator\[\] |
| q14complex_matrix11eigenvaluesEv) |     (C++                          |
| -   [cu                           |     function)](ap                 |
| daq::complex_matrix::eigenvectors | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq10product_opixENSt6size_tE) |
|     function)](api/lang           | -                                 |
| uages/cpp_api.html#_CPPv4NK5cudaq |    [cudaq::product_op::product_op |
| 14complex_matrix12eigenvectorsEv) |     (C++                          |
| -   [c                            |     f                             |
| udaq::complex_matrix::exponential | unction)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4I00EN5cudaq10product_op |
|     function)](api/la             | 10product_opERK10product_opI1TE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[1\]]                       |
| q14complex_matrix11exponentialEv) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4I00EN5cudaq10product_op10product |
|  [cudaq::complex_matrix::identity | _opERK10product_opI1TERKN14matrix |
|     (C++                          | _handler20commutation_behaviorE), |
|     function)](api/languages      |                                   |
| /cpp_api.html#_CPPv4N5cudaq14comp |   [\[2\]](api/languages/cpp_api.h |
| lex_matrix8identityEKNSt6size_tE) | tml#_CPPv4N5cudaq10product_op10pr |
| -                                 | oduct_opENSt6size_tENSt6size_tE), |
| [cudaq::complex_matrix::kronecker |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     function)](api/lang           | _op10product_opENSt7complexIdEE), |
| uages/cpp_api.html#_CPPv4I00EN5cu |     [\[4\]](api/l                 |
| daq14complex_matrix9kroneckerE14c | anguages/cpp_api.html#_CPPv4N5cud |
| omplex_matrix8Iterable8Iterable), | aq10product_op10product_opERK10pr |
|     [\[1\]](api/l                 | oduct_opI9HandlerTyENSt6size_tE), |
| anguages/cpp_api.html#_CPPv4N5cud |     [\[5\]](api/l                 |
| aq14complex_matrix9kroneckerERK14 | anguages/cpp_api.html#_CPPv4N5cud |
| complex_matrixRK14complex_matrix) | aq10product_op10product_opERR10pr |
| -   [cudaq::c                     | oduct_opI9HandlerTyENSt6size_tE), |
| omplex_matrix::minimal_eigenvalue |     [\[6\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq10prod |
|     function)](api/languages/     | uct_op10product_opERR9HandlerTy), |
| cpp_api.html#_CPPv4NK5cudaq14comp |     [\[7\]](ap                    |
| lex_matrix18minimal_eigenvalueEv) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [                             | cudaq10product_op10product_opEd), |
| cudaq::complex_matrix::operator() |     [\[8\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function)](api/languages/cpp  | 5cudaq10product_op10product_opEv) |
| _api.html#_CPPv4N5cudaq14complex_ | -   [cuda                         |
| matrixclENSt6size_tENSt6size_tE), | q::product_op::to_diagonal_matrix |
|     [\[1\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4NK5cudaq14complex |     function)](api/               |
| _matrixclENSt6size_tENSt6size_tE) | languages/cpp_api.html#_CPPv4NK5c |
| -   [                             | udaq10product_op18to_diagonal_mat |
| cudaq::complex_matrix::operator\* | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/langua         | apINSt6stringENSt7complexIdEEEEb) |
| ges/cpp_api.html#_CPPv4N5cudaq14c | -   [cudaq::product_op::to_matrix |
| omplex_matrixmlEN14complex_matrix |     (C++                          |
| 10value_typeERK14complex_matrix), |     funct                         |
|     [\[1\]                        | ion)](api/languages/cpp_api.html# |
| ](api/languages/cpp_api.html#_CPP | _CPPv4NK5cudaq10product_op9to_mat |
| v4N5cudaq14complex_matrixmlERK14c | rixENSt13unordered_mapINSt6size_t |
| omplex_matrixRK14complex_matrix), | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|  [\[2\]](api/languages/cpp_api.ht | -   [cu                           |
| ml#_CPPv4N5cudaq14complex_matrixm | daq::product_op::to_sparse_matrix |
| lERK14complex_matrixRKNSt6vectorI |     (C++                          |
| N14complex_matrix10value_typeEEE) |     function)](ap                 |
| -                                 | i/languages/cpp_api.html#_CPPv4NK |
| [cudaq::complex_matrix::operator+ | 5cudaq10product_op16to_sparse_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function                      | ENSt7int64_tEEERKNSt13unordered_m |
| )](api/languages/cpp_api.html#_CP | apINSt6stringENSt7complexIdEEEEb) |
| Pv4N5cudaq14complex_matrixplERK14 | -   [cudaq::product_op::to_string |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -                                 |     function)](                   |
| [cudaq::complex_matrix::operator- | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq10product_op9to_stringEv) |
|     function                      | -                                 |
| )](api/languages/cpp_api.html#_CP |  [cudaq::product_op::\~product_op |
| Pv4N5cudaq14complex_matrixmiERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     fu                            |
| -   [cu                           | nction)](api/languages/cpp_api.ht |
| daq::complex_matrix::operator\[\] | ml#_CPPv4N5cudaq10product_opD0Ev) |
|     (C++                          | -   [cudaq::ptsbe (C++            |
|                                   |     type)](api/languages/c        |
|  function)](api/languages/cpp_api | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| .html#_CPPv4N5cudaq14complex_matr | -   [cudaq::p                     |
| ixixERKNSt6vectorINSt6size_tEEE), | tsbe::ConditionalSamplingStrategy |
|     [\[1\]](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4NK5cudaq14complex_mat |     class)](api/languag           |
| rixixERKNSt6vectorINSt6size_tEEE) | es/cpp_api.html#_CPPv4N5cudaq5pts |
| -   [cudaq::complex_matrix::power | be27ConditionalSamplingStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::C              |
|     function)]                    | onditionalSamplingStrategy::clone |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq14complex_matrix5powerEi) |                                   |
| -                                 |    function)](api/languages/cpp_a |
|  [cudaq::complex_matrix::set_zero | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
|     (C++                          | ditionalSamplingStrategy5cloneEv) |
|     function)](ap                 | -   [cuda                         |
| i/languages/cpp_api.html#_CPPv4N5 | q::ptsbe::ConditionalSamplingStra |
| cudaq14complex_matrix8set_zeroEv) | tegy::ConditionalSamplingStrategy |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::to_string |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     function)](api/               | ptsbe27ConditionalSamplingStrateg |
| languages/cpp_api.html#_CPPv4NK5c | y27ConditionalSamplingStrategyE19 |
| udaq14complex_matrix9to_stringEv) | TrajectoryPredicateNSt8uint64_tE) |
| -   [                             | -                                 |
| cudaq::complex_matrix::value_type |   [cudaq::ptsbe::ConditionalSampl |
|     (C++                          | ingStrategy::generateTrajectories |
|     type)](api/                   |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/language       |
| daq14complex_matrix10value_typeE) | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| -   [cudaq::contrib (C++          | be27ConditionalSamplingStrategy20 |
|     type)](api/languages/cpp      | generateTrajectoriesENSt4spanIKN6 |
| _api.html#_CPPv4N5cudaq7contribE) | detail10NoisePointEEENSt6size_tE) |
| -                                 | -   [cudaq::ptsbe::               |
| [cudaq::contrib::amplitude_encode | ConditionalSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api/languages/cpp_ |
| s/cpp_api.html#_CPPv4N5cudaq7cont | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| rib16amplitude_encodeENSt4spanIKN | nditionalSamplingStrategy4nameEv) |
| St7complexIdEEEENSt7complexIdEE), | -   [cudaq:                       |
|     [\[1\]](api/language          | :ptsbe::ConditionalSamplingStrate |
| s/cpp_api.html#_CPPv4N5cudaq7cont | gy::\~ConditionalSamplingStrategy |
| rib16amplitude_encodeENSt4spanIKN |     (C++                          |
| St7complexIfEEEENSt7complexIdEE), |     function)](api/languages/     |
|     [\[2\]                        | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| ](api/languages/cpp_api.html#_CPP | 7ConditionalSamplingStrategyD0Ev) |
| v4N5cudaq7contrib16amplitude_enco | -                                 |
| deENSt4spanIKdEENSt7complexIdEE), | [cudaq::ptsbe::detail::NoisePoint |
|     [\[3\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     struct)](a                    |
| v4N5cudaq7contrib16amplitude_enco | pi/languages/cpp_api.html#_CPPv4N |
| deENSt4spanIKfEENSt7complexIdEE), | 5cudaq5ptsbe6detail10NoisePointE) |
|                                   | -   [cudaq::p                     |
| [\[4\]](api/languages/cpp_api.htm | tsbe::detail::NoisePoint::channel |
| l#_CPPv4N5cudaq7contrib16amplitud |     (C++                          |
| e_encodeERK5stateNSt7complexIdEE) |     member)](api/langu            |
| -                                 | ages/cpp_api.html#_CPPv4N5cudaq5p |
|   [cudaq::contrib::angular_encode | tsbe6detail10NoisePoint7channelE) |
|     (C++                          | -   [cudaq::ptsbe::det            |
|                                   | ail::NoisePoint::circuit_location |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I0EN5cudaq7contrib14a |     member)](api/languages/cpp_a  |
| ngular_encodeEvRR6KernelR10QuakeV | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| alueNSt4spanIKdEE12RotationAxis), | l10NoisePoint16circuit_locationE) |
|     [\[1\]](api/languages/cpp_api | -   [cudaq::p                     |
| .html#_CPPv4I0EN5cudaq7contrib14a | tsbe::detail::NoisePoint::op_name |
| ngular_encodeEvRR6KernelR10QuakeV |     (C++                          |
| alueR10QuakeValue12RotationAxis), |     member)](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq5p |
|   [\[2\]](api/languages/cpp_api.h | tsbe6detail10NoisePoint7op_nameE) |
| tml#_CPPv4I0EN5cudaq7contrib14ang | -   [cudaq::                      |
| ular_encodeEvRR6KernelR10QuakeVal | ptsbe::detail::NoisePoint::qubits |
| ueRKNSt6vectorIdEE12RotationAxis) |     (C++                          |
| -   [cudaq::contrib::draw (C++    |     member)](api/lang             |
|     function)                     | uages/cpp_api.html#_CPPv4N5cudaq5 |
| ](api/languages/cpp_api.html#_CPP | ptsbe6detail10NoisePoint6qubitsE) |
| v4I0DpEN5cudaq7contrib4drawENSt6s | -   [cudaq::                      |
| tringERR13QuantumKernelDpRR4Args) | ptsbe::ExhaustiveSamplingStrategy |
| -                                 |     (C++                          |
| [cudaq::contrib::get_unitary_cmat |     class)](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     function)](api/languages/cp   | sbe26ExhaustiveSamplingStrategyE) |
| p_api.html#_CPPv4I0DpEN5cudaq7con | -   [cudaq::ptsbe::               |
| trib16get_unitary_cmatE14complex_ | ExhaustiveSamplingStrategy::clone |
| matrixRR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::contrib::RotationAxis |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
|     enum)                         | haustiveSamplingStrategy5cloneEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cu                           |
| v4N5cudaq7contrib12RotationAxisE) | daq::ptsbe::ExhaustiveSamplingStr |
| -                                 | ategy::ExhaustiveSamplingStrategy |
|  [cudaq::contrib::RotationAxis::X |     (C++                          |
|     (C++                          |     function)](api/la             |
|     enumerator)](                 | nguages/cpp_api.html#_CPPv4N5cuda |
| api/languages/cpp_api.html#_CPPv4 | q5ptsbe26ExhaustiveSamplingStrate |
| N5cudaq7contrib12RotationAxis1XE) | gy26ExhaustiveSamplingStrategyEv) |
| -                                 | -                                 |
|  [cudaq::contrib::RotationAxis::Y |    [cudaq::ptsbe::ExhaustiveSampl |
|     (C++                          | ingStrategy::generateTrajectories |
|     enumerator)](                 |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languag        |
| N5cudaq7contrib12RotationAxis1YE) | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| -                                 | sbe26ExhaustiveSamplingStrategy20 |
|  [cudaq::contrib::RotationAxis::Z | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     enumerator)](                 | -   [cudaq::ptsbe:                |
| api/languages/cpp_api.html#_CPPv4 | :ExhaustiveSamplingStrategy::name |
| N5cudaq7contrib12RotationAxis1ZE) |     (C++                          |
| -   [cudaq::CusvState (C++        |     function)](api/languages/cpp  |
|                                   | _api.html#_CPPv4NK5cudaq5ptsbe26E |
|    class)](api/languages/cpp_api. | xhaustiveSamplingStrategy4nameEv) |
| html#_CPPv4I0EN5cudaq9CusvStateE) | -   [cuda                         |
| -   [cudaq::dem_from_kernel (C++  | q::ptsbe::ExhaustiveSamplingStrat |
|     function)](api                | egy::\~ExhaustiveSamplingStrategy |
| /languages/cpp_api.html#_CPPv4I0D |     (C++                          |
| pEN5cudaq15dem_from_kernelENSt6st |     function)](api/languages      |
| ringERR13QuantumKernelDpRR4Args), | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|                                   | 26ExhaustiveSamplingStrategyD0Ev) |
| [\[1\]](api/languages/cpp_api.htm | -   [cuda                         |
| l#_CPPv4I0DpEN5cudaq15dem_from_ke | q::ptsbe::OrderedSamplingStrategy |
| rnelENSt6stringERR13QuantumKernel |     (C++                          |
| PKN5cudaq11noise_modelEDpRR4Args) |     class)](api/lan               |
| -   [cudaq::depolarization1 (C++  | guages/cpp_api.html#_CPPv4N5cudaq |
|     c                             | 5ptsbe23OrderedSamplingStrategyE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::ptsb                  |
| #_CPPv4N5cudaq15depolarization1E) | e::OrderedSamplingStrategy::clone |
| -   [cudaq::depolarization2 (C++  |     (C++                          |
|     c                             |     function)](api/languages/c    |
| lass)](api/languages/cpp_api.html | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| #_CPPv4N5cudaq15depolarization2E) | 3OrderedSamplingStrategy5cloneEv) |
| -   [cudaq:                       | -   [cudaq::ptsbe::OrderedSampl   |
| :depolarization2::depolarization2 | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/lang           |
| p_api.html#_CPPv4N5cudaq15depolar | uages/cpp_api.html#_CPPv4NK5cudaq |
| ization215depolarization2EK4real) | 5ptsbe23OrderedSamplingStrategy20 |
| -   [cudaq                        | generateTrajectoriesENSt4spanIKN6 |
| ::depolarization2::num_parameters | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/langu            | be::OrderedSamplingStrategy::name |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| depolarization214num_parametersE) |     function)](api/languages/     |
| -   [cu                           | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| daq::depolarization2::num_targets | 23OrderedSamplingStrategy4nameEv) |
|     (C++                          | -                                 |
|     member)](api/la               |    [cudaq::ptsbe::OrderedSampling |
| nguages/cpp_api.html#_CPPv4N5cuda | Strategy::OrderedSamplingStrategy |
| q15depolarization211num_targetsE) |     (C++                          |
| -                                 |     function)](                   |
|    [cudaq::depolarization_channel | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5ptsbe23OrderedSamplingStr |
|     class)](                      | ategy23OrderedSamplingStrategyEv) |
| api/languages/cpp_api.html#_CPPv4 | -                                 |
| N5cudaq22depolarization_channelE) |  [cudaq::ptsbe::OrderedSamplingSt |
| -   [cudaq::depol                 | rategy::\~OrderedSamplingStrategy |
| arization_channel::num_parameters |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     member)](api/languages/cp     | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| p_api.html#_CPPv4N5cudaq22depolar | sbe23OrderedSamplingStrategyD0Ev) |
| ization_channel14num_parametersE) | -   [cudaq::pts                   |
| -   [cudaq::de                    | be::ProbabilisticSamplingStrategy |
| polarization_channel::num_targets |     (C++                          |
|     (C++                          |     class)](api/languages         |
|     member)](api/languages        | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| /cpp_api.html#_CPPv4N5cudaq22depo | 29ProbabilisticSamplingStrategyE) |
| larization_channel11num_targetsE) | -   [cudaq::ptsbe::Pro            |
| -   [cudaq::detail (C++           | babilisticSamplingStrategy::clone |
|     type)](api/languages/cp       |     (C++                          |
| p_api.html#_CPPv4N5cudaq6detailE) |                                   |
| -   [cudaq::detail::future (C++   |  function)](api/languages/cpp_api |
|                                   | .html#_CPPv4NK5cudaq5ptsbe29Proba |
|   class)](api/languages/cpp_api.h | bilisticSamplingStrategy5cloneEv) |
| tml#_CPPv4N5cudaq6detail6futureE) | -                                 |
| -                                 | [cudaq::ptsbe::ProbabilisticSampl |
|    [cudaq::detail::future::future | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     functi                        |     function)](api/languages/     |
| on)](api/languages/cpp_api.html#_ | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| CPPv4N5cudaq6detail6future6future | 29ProbabilisticSamplingStrategy20 |
| ERNSt6vectorI3JobEERNSt6stringERN | generateTrajectoriesENSt4spanIKN6 |
| St3mapINSt6stringENSt6stringEEE), | detail10NoisePointEEENSt6size_tE) |
|     [\[1\]](api/lan               | -   [cudaq::ptsbe::Pr             |
| guages/cpp_api.html#_CPPv4N5cudaq | obabilisticSamplingStrategy::name |
| 6detail6future6futureERR6future), |     (C++                          |
|     [\[2\]                        |                                   |
| ](api/languages/cpp_api.html#_CPP |   function)](api/languages/cpp_ap |
| v4N5cudaq6detail6future6futureEv) | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| -   [c                            | abilisticSamplingStrategy4nameEv) |
| udaq::detail::kernel_builder_base | -   [cudaq::p                     |
|     (C++                          | tsbe::ProbabilisticSamplingStrate |
|     class)](api/                  | gy::ProbabilisticSamplingStrategy |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq6detail19kernel_builder_baseE) |     function)]                    |
| -   [cudaq::detail::              | (api/languages/cpp_api.html#_CPPv |
| kernel_builder_base::operator\<\< | 4N5cudaq5ptsbe29ProbabilisticSamp |
|     (C++                          | lingStrategy29ProbabilisticSampli |
|     function)](api/langu          | ngStrategyENSt8optionalINSt8uint6 |
| ages/cpp_api.html#_CPPv4N5cudaq6d | 4_tEEENSt8optionalINSt6size_tEEE) |
| etail19kernel_builder_baselsERNSt | -   [cudaq::pts                   |
| 7ostreamERK19kernel_builder_base) | be::ProbabilisticSamplingStrategy |
| -                                 | ::\~ProbabilisticSamplingStrategy |
| [cudaq::detail::KernelBuilderType |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     class)](ap                    | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| i/languages/cpp_api.html#_CPPv4N5 | robabilisticSamplingStrategyD0Ev) |
| cudaq6detail17KernelBuilderTypeE) | -                                 |
| -   [cudaq::                      | [cudaq::ptsbe::PTSBEExecutionData |
| detail::KernelBuilderType::create |     (C++                          |
|     (C++                          |     struct)](ap                   |
|     function                      | i/languages/cpp_api.html#_CPPv4N5 |
| )](api/languages/cpp_api.html#_CP | cudaq5ptsbe18PTSBEExecutionDataE) |
| Pv4N5cudaq6detail17KernelBuilderT | -   [cudaq::ptsbe::PTSBE          |
| ype6createEPN4mlir11MLIRContextE) | ExecutionData::count_instructions |
| -   [cudaq::detail::Ker           |     (C++                          |
| nelBuilderType::KernelBuilderType |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     function)](api/lan            | daq5ptsbe18PTSBEExecutionData18co |
| guages/cpp_api.html#_CPPv4N5cudaq | unt_instructionsE20TraceInstructi |
| 6detail17KernelBuilderType17Kerne | onTypeNSt8optionalINSt6stringEEE) |
| lBuilderTypeERRNSt8functionIFN4ml | -   [cudaq::ptsbe::P              |
| ir4TypeEPN4mlir11MLIRContextEEEE) | TSBEExecutionData::get_trajectory |
| -   [cudaq::detector (C++         |     (C++                          |
|     function)](api                |     function                      |
| /languages/cpp_api.html#_CPPv4IDp | )](api/languages/cpp_api.html#_CP |
| EN5cudaq8detectorEvDpRR8MeasArgs) | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| -   [cudaq::detectors (C++        | Data14get_trajectoryENSt6size_tE) |
|     function)](api/languages/c    | -   [cudaq::ptsbe:                |
| pp_api.html#_CPPv4N5cudaq9detecto | :PTSBEExecutionData::instructions |
| rsERKNSt6vectorI14measure_resultE |     (C++                          |
| ERKNSt6vectorI14measure_resultEE) |     member)](api/languages/cp     |
| -   [cudaq::diag_matrix_callback  | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     (C++                          | TSBEExecutionData12instructionsE) |
|     class)                        | -   [cudaq::ptsbe:                |
| ](api/languages/cpp_api.html#_CPP | :PTSBEExecutionData::trajectories |
| v4N5cudaq20diag_matrix_callbackE) |     (C++                          |
| -   [cudaq::dyn (C++              |     member)](api/languages/cp     |
|     member)](api/languages        | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | TSBEExecutionData12trajectoriesE) |
| -   [cudaq::ExecutionContext (C++ | -   [cudaq::ptsbe::PTSBEOptions   |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     struc                         |
| _CPPv4N5cudaq16ExecutionContextE) | t)](api/languages/cpp_api.html#_C |
| -   [c                            | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| udaq::ExecutionContext::asyncExec | -   [cudaq::ptsbe::PTSB           |
|     (C++                          | EOptions::include_sequential_data |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq16ExecutionContext9asyncExecE) |    member)](api/languages/cpp_api |
| -   [cud                          | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| aq::ExecutionContext::asyncResult | ptions23include_sequential_dataE) |
|     (C++                          | -   [cudaq::ptsb                  |
|     member)](api/lan              | e::PTSBEOptions::max_trajectories |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11asyncResultE) |     member)](api/languages/       |
| -   [cudaq:                       | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| :ExecutionContext::batchIteration | 2PTSBEOptions16max_trajectoriesE) |
|     (C++                          | -   [cudaq::ptsbe::PT             |
|     member)](api/langua           | SBEOptions::return_execution_data |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14batchIterationE) |     member)](api/languages/cpp_a  |
| -   [cudaq::E                     | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| xecutionContext::canHandleObserve | EOptions21return_execution_dataE) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/language         | be::PTSBEOptions::shot_allocation |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     (C++                          |
| cutionContext16canHandleObserveE) |     member)](api/languages        |
| -   [cudaq::Executio              | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| nContext::deferredKernelException | 12PTSBEOptions15shot_allocationE) |
|     (C++                          | -   [cud                          |
|     member)](api/languages/cpp_a  | aq::ptsbe::PTSBEOptions::strategy |
| pi.html#_CPPv4N5cudaq16ExecutionC |     (C++                          |
| ontext23deferredKernelExceptionE) |     member)](api/l                |
| -   [cudaq::E                     | anguages/cpp_api.html#_CPPv4N5cud |
| xecutionContext::ExecutionContext | aq5ptsbe12PTSBEOptions8strategyE) |
|     (C++                          | -   [cudaq::ptsbe::PTSBETrace     |
|     func                          |     (C++                          |
| tion)](api/languages/cpp_api.html |     t                             |
| #_CPPv4N5cudaq16ExecutionContext1 | ype)](api/languages/cpp_api.html# |
| 6ExecutionContextERKNSt6stringE), | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
|     [\[1\]](api/languages/        | -   [                             |
| cpp_api.html#_CPPv4N5cudaq16Execu | cudaq::ptsbe::PTSSamplingStrategy |
| tionContext16ExecutionContextERKN |     (C++                          |
| St6stringENSt6size_tENSt6size_tE) |     class)](api                   |
| -   [cudaq::E                     | /languages/cpp_api.html#_CPPv4N5c |
| xecutionContext::expectationValue | udaq5ptsbe19PTSSamplingStrategyE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/language         | ptsbe::PTSSamplingStrategy::clone |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     (C++                          |
| cutionContext16expectationValueE) |     function)](api/languag        |
| -   [cudaq::Execu                 | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| tionContext::explicitMeasurements | sbe19PTSSamplingStrategy5cloneEv) |
|     (C++                          | -   [cudaq::ptsbe::PTSSampl       |
|     member)](api/languages/cp     | ingStrategy::generateTrajectories |
| p_api.html#_CPPv4N5cudaq16Executi |     (C++                          |
| onContext20explicitMeasurementsE) |     function)](api/               |
| -   [cuda                         | languages/cpp_api.html#_CPPv4NK5c |
| q::ExecutionContext::futureResult | udaq5ptsbe19PTSSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/lang             | detail10NoisePointEEENSt6size_tE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq:                       |
| 6ExecutionContext12futureResultE) | :ptsbe::PTSSamplingStrategy::name |
| -   [cudaq::ExecutionContext      |     (C++                          |
| ::hasConditionalsOnMeasureResults |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq5p |
|     mem                           | tsbe19PTSSamplingStrategy4nameEv) |
| ber)](api/languages/cpp_api.html# | -   [cudaq::ptsbe::PTSSampli      |
| _CPPv4N5cudaq16ExecutionContext31 | ngStrategy::\~PTSSamplingStrategy |
| hasConditionalsOnMeasureResultsE) |     (C++                          |
| -   [cudaq:                       |     function)](api/la             |
| :ExecutionContext::inKernelLaunch | nguages/cpp_api.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe19PTSSamplingStrategyD0Ev) |
|     member)](api/langua           | -   [cudaq::ptsbe::sample (C++    |
| ges/cpp_api.html#_CPPv4N5cudaq16E |                                   |
| xecutionContext14inKernelLaunchE) |  function)](api/languages/cpp_api |
| -   [cudaq::Executi               | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| onContext::invocationResultBuffer | mpleE13sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](api/languages/cpp_   |     [\[1\]](api                   |
| api.html#_CPPv4N5cudaq16Execution | /languages/cpp_api.html#_CPPv4I0D |
| Context22invocationResultBufferE) | pEN5cudaq5ptsbe6sampleE13sample_r |
| -   [cu                           | esultRKN5cudaq11noise_modelENSt6s |
| daq::ExecutionContext::kernelName | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_async   |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](a                  |
| q16ExecutionContext10kernelNameE) | pi/languages/cpp_api.html#_CPPv4I |
| -   [cud                          | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| aq::ExecutionContext::kernelTrace | 9async_sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](api/lan              |     [\[1\]](api/languages/cp      |
| guages/cpp_api.html#_CPPv4N5cudaq | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| 16ExecutionContext11kernelTraceE) | be12sample_asyncE19async_sample_r |
| -   [cudaq:                       | esultRKN5cudaq11noise_modelENSt6s |
| :ExecutionContext::msm_dimensions | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_options |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     struct)                       |
| xecutionContext14msm_dimensionsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::                      | v4N5cudaq5ptsbe14sample_optionsE) |
| ExecutionContext::msm_prob_err_id | -   [cudaq::ptsbe::sample_result  |
|     (C++                          |     (C++                          |
|     member)](api/languag          |     class                         |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | )](api/languages/cpp_api.html#_CP |
| ecutionContext15msm_prob_err_idE) | Pv4N5cudaq5ptsbe13sample_resultE) |
| -   [cudaq::Ex                    | -   [cudaq::pts                   |
| ecutionContext::msm_probabilities | be::sample_result::execution_data |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)](api/languages/c    |
| /cpp_api.html#_CPPv4N5cudaq16Exec | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| utionContext17msm_probabilitiesE) | 3sample_result14execution_dataEv) |
| -                                 | -   [cudaq::ptsbe::               |
|    [cudaq::ExecutionContext::name | sample_result::has_execution_data |
|     (C++                          |     (C++                          |
|     member)]                      |                                   |
| (api/languages/cpp_api.html#_CPPv |    function)](api/languages/cpp_a |
| 4N5cudaq16ExecutionContext4nameE) | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| -   [cu                           | ple_result18has_execution_dataEv) |
| daq::ExecutionContext::noiseModel | -   [cudaq::pt                    |
|     (C++                          | sbe::sample_result::sample_result |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/l              |
| q16ExecutionContext10noiseModelE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::Exe                   | aq5ptsbe13sample_result13sample_r |
| cutionContext::numberTrajectories | esultERRN5cudaq13sample_resultE), |
|     (C++                          |                                   |
|     member)](api/languages/       |  [\[1\]](api/languages/cpp_api.ht |
| cpp_api.html#_CPPv4N5cudaq16Execu | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| tionContext18numberTrajectoriesE) | sult13sample_resultERRN5cudaq13sa |
| -   [c                            | mple_resultE18PTSBEExecutionData) |
| udaq::ExecutionContext::optResult | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::set_execution_data |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/               |
| daq16ExecutionContext9optResultE) | languages/cpp_api.html#_CPPv4N5cu |
| -                                 | daq5ptsbe13sample_result18set_exe |
|   [cudaq::ExecutionContext::qpuId | cution_dataE18PTSBEExecutionData) |
|     (C++                          | -   [cud                          |
|     member)](                     | aq::ptsbe::ShotAllocationStrategy |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5qpuIdE) |     struct)](using                |
| -   [cudaq                        | /examples/ptsbe.html#_CPPv4N5cuda |
| ::ExecutionContext::registerNames | q5ptsbe22ShotAllocationStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAllocatio  |
|     member)](api/langu            | nStrategy::ShotAllocationStrategy |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13registerNamesE) |     function)                     |
| -   [cu                           | ](using/examples/ptsbe.html#_CPPv |
| daq::ExecutionContext::reorderIdx | 4N5cudaq5ptsbe22ShotAllocationStr |
|     (C++                          | ategy22ShotAllocationStrategyE4Ty |
|     member)](api/la               | pedNSt8optionalINSt8uint64_tEEE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[1\                         |
| q16ExecutionContext10reorderIdxE) | ]](using/examples/ptsbe.html#_CPP |
| -                                 | v4N5cudaq5ptsbe22ShotAllocationSt |
|  [cudaq::ExecutionContext::result | rategy22ShotAllocationStrategyEv) |
|     (C++                          | -   [cudaq::pt                    |
|     member)](a                    | sbe::ShotAllocationStrategy::Type |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq16ExecutionContext6resultE) |     enum)](using/exam             |
| -                                 | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|   [cudaq::ExecutionContext::shots | be22ShotAllocationStrategy4TypeE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAllocatio  |
|     member)](                     | nStrategy::Type::HIGH_WEIGHT_BIAS |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5shotsE) |     enumerat                      |
| -   [cudaq::                      | or)](using/examples/ptsbe.html#_C |
| ExecutionContext::simulationState | PPv4N5cudaq5ptsbe22ShotAllocation |
|     (C++                          | Strategy4Type16HIGH_WEIGHT_BIASE) |
|     member)](api/languag          | -   [cudaq::ptsbe::ShotAllocati   |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | onStrategy::Type::LOW_WEIGHT_BIAS |
| ecutionContext15simulationStateE) |     (C++                          |
| -                                 |     enumera                       |
|    [cudaq::ExecutionContext::spin | tor)](using/examples/ptsbe.html#_ |
|     (C++                          | CPPv4N5cudaq5ptsbe22ShotAllocatio |
|     member)]                      | nStrategy4Type15LOW_WEIGHT_BIASE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::ptsbe::ShotAlloc      |
| 4N5cudaq16ExecutionContext4spinE) | ationStrategy::Type::PROPORTIONAL |
| -   [cudaq::                      |     (C++                          |
| ExecutionContext::totalIterations |     enum                          |
|     (C++                          | erator)](using/examples/ptsbe.htm |
|     member)](api/languag          | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | tionStrategy4Type12PROPORTIONALE) |
| ecutionContext15totalIterationsE) | -   [cudaq::ptsbe::Shot           |
| -   [cudaq::ExecutionResult (C++  | AllocationStrategy::Type::UNIFORM |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |                                   |
| #_CPPv4N5cudaq15ExecutionResultE) |   enumerator)](using/examples/pts |
| -   [cud                          | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| aq::ExecutionResult::appendResult | AllocationStrategy4Type7UNIFORME) |
|     (C++                          | -                                 |
|     functio                       |   [cudaq::ptsbe::TraceInstruction |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq15ExecutionResult12app |     struct)](                     |
| endResultENSt6stringENSt6size_tE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cu                           | N5cudaq5ptsbe16TraceInstructionE) |
| daq::ExecutionResult::deserialize | -   [cudaq:                       |
|     (C++                          | :ptsbe::TraceInstruction::channel |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     member)](api/lang             |
| v4N5cudaq15ExecutionResult11deser | uages/cpp_api.html#_CPPv4N5cudaq5 |
| ializeERNSt6vectorINSt6size_tEEE) | ptsbe16TraceInstruction7channelE) |
| -   [cudaq:                       | -   [cudaq::                      |
| :ExecutionResult::ExecutionResult | ptsbe::TraceInstruction::controls |
|     (C++                          |     (C++                          |
|     functio                       |     member)](api/langu            |
| n)](api/languages/cpp_api.html#_C | ages/cpp_api.html#_CPPv4N5cudaq5p |
| PPv4N5cudaq15ExecutionResult15Exe | tsbe16TraceInstruction8controlsE) |
| cutionResultE16CountsDictionary), | -   [cud                          |
|     [\[1\]](api/lan               | aq::ptsbe::TraceInstruction::name |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult15ExecutionResul |     member)](api/l                |
| tE16CountsDictionaryNSt6stringE), | anguages/cpp_api.html#_CPPv4N5cud |
|     [\[2\                         | aq5ptsbe16TraceInstruction4nameE) |
| ]](api/languages/cpp_api.html#_CP | -   [cudaq                        |
| Pv4N5cudaq15ExecutionResult15Exec | ::ptsbe::TraceInstruction::params |
| utionResultE16CountsDictionaryd), |     (C++                          |
|                                   |     member)](api/lan              |
|    [\[3\]](api/languages/cpp_api. | guages/cpp_api.html#_CPPv4N5cudaq |
| html#_CPPv4N5cudaq15ExecutionResu | 5ptsbe16TraceInstruction6paramsE) |
| lt15ExecutionResultENSt6stringE), | -   [cudaq:                       |
|     [\[4\                         | :ptsbe::TraceInstruction::targets |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     member)](api/lang             |
| utionResultERK15ExecutionResult), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     [\[5\]](api/language          | ptsbe16TraceInstruction7targetsE) |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | -   [cudaq::ptsbe::T              |
| cutionResult15ExecutionResultEd), | raceInstruction::TraceInstruction |
|     [\[6\]](api/languag           |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |                                   |
| ecutionResult15ExecutionResultEv) |   function)](api/languages/cpp_ap |
| -   [                             | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| cudaq::ExecutionResult::operator= | Instruction16TraceInstructionE20T |
|     (C++                          | raceInstructionTypeNSt6stringENSt |
|     function)](api/languages/     | 6vectorINSt6size_tEEENSt6vectorIN |
| cpp_api.html#_CPPv4N5cudaq15Execu | St6size_tEEENSt6vectorIdEENSt8opt |
| tionResultaSERK15ExecutionResult) | ionalIN5cudaq13kraus_channelEEE), |
| -   [c                            |     [\[1\]](api/languages/cpp_a   |
| udaq::ExecutionResult::operator== | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
|     (C++                          | eInstruction16TraceInstructionEv) |
|     function)](api/languages/c    | -   [cud                          |
| pp_api.html#_CPPv4NK5cudaq15Execu | aq::ptsbe::TraceInstruction::type |
| tionResulteqERK15ExecutionResult) |     (C++                          |
| -   [cud                          |     member)](api/l                |
| aq::ExecutionResult::registerName | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq5ptsbe16TraceInstruction4typeE) |
|     member)](api/lan              | -   [c                            |
| guages/cpp_api.html#_CPPv4N5cudaq | udaq::ptsbe::TraceInstructionType |
| 15ExecutionResult12registerNameE) |     (C++                          |
| -   [cudaq                        |     enum)](api/                   |
| ::ExecutionResult::sequentialData | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq5ptsbe20TraceInstructionTypeE) |
|     member)](api/langu            | -   [cudaq::                      |
| ages/cpp_api.html#_CPPv4N5cudaq15 | ptsbe::TraceInstructionType::Gate |
| ExecutionResult14sequentialDataE) |     (C++                          |
| -   [                             |     enumerator)](api/langu        |
| cudaq::ExecutionResult::serialize | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     (C++                          | tsbe20TraceInstructionType4GateE) |
|     function)](api/l              | -   [cudaq::ptsbe::               |
| anguages/cpp_api.html#_CPPv4NK5cu | TraceInstructionType::Measurement |
| daq15ExecutionResult9serializeEv) |     (C++                          |
| -   [cudaq::fermion_handler (C++  |                                   |
|     c                             |    enumerator)](api/languages/cpp |
| lass)](api/languages/cpp_api.html | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| #_CPPv4N5cudaq15fermion_handlerE) | aceInstructionType11MeasurementE) |
| -   [cudaq::fermion_op (C++       | -   [cudaq::p                     |
|     type)](api/languages/cpp_api  | tsbe::TraceInstructionType::Noise |
| .html#_CPPv4N5cudaq10fermion_opE) |     (C++                          |
| -   [cudaq::fermion_op_term (C++  |     enumerator)](api/langua       |
|                                   | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| type)](api/languages/cpp_api.html | sbe20TraceInstructionType5NoiseE) |
| #_CPPv4N5cudaq15fermion_op_termE) | -   [                             |
| -   [cudaq::FermioniqQPU (C++     | cudaq::ptsbe::TrajectoryPredicate |
|                                   |     (C++                          |
|   class)](api/languages/cpp_api.h |     type)](api                    |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::get_state (C++        | udaq5ptsbe19TrajectoryPredicateE) |
|                                   | -   [cudaq::QPU (C++              |
|    function)](api/languages/cpp_a |     class)](api/languages         |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| ateEDaRR13QuantumKernelDpRR4Args) | -   [cudaq::QPU::beginExecution   |
| -   [cudaq::gradient (C++         |     (C++                          |
|     class)](api/languages/cpp_    |     function                      |
| api.html#_CPPv4N5cudaq8gradientE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::gradient::clone (C++  | Pv4N5cudaq3QPU14beginExecutionEv) |
|     fun                           | -   [cuda                         |
| ction)](api/languages/cpp_api.htm | q::QPU::configureExecutionContext |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     (C++                          |
| -   [cudaq::gradient::compute     |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/language       | _CPPv4NK5cudaq3QPU25configureExec |
| s/cpp_api.html#_CPPv4N5cudaq8grad | utionContextER16ExecutionContext) |
| ient7computeERKNSt6vectorIdEERKNS | -   [cudaq::QPU::endExecution     |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|     [\[1\]](ap                    |     functi                        |
| i/languages/cpp_api.html#_CPPv4N5 | on)](api/languages/cpp_api.html#_ |
| cudaq8gradient7computeERKNSt6vect | CPPv4N5cudaq3QPU12endExecutionEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QPU::enqueue (C++     |
| -   [cudaq::gradient::gradient    |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/lang           | cudaq3QPU7enqueueER11QuantumTask) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -   [cud                          |
| daq8gradient8gradientER7KernelT), | aq::QPU::finalizeExecutionContext |
|                                   |     (C++                          |
|    [\[1\]](api/languages/cpp_api. |     func                          |
| html#_CPPv4I00EN5cudaq8gradient8g | tion)](api/languages/cpp_api.html |
| radientER7KernelTRR10ArgsMapper), | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     [\[2\                         | utionContextER16ExecutionContext) |
| ]](api/languages/cpp_api.html#_CP | -   [cudaq::QPU::getCompileTarget |
| Pv4I00EN5cudaq8gradient8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/languages/c    |
|     [\[3                          | pp_api.html#_CPPv4N5cudaq3QPU16ge |
| \]](api/languages/cpp_api.html#_C | tCompileTargetERK13sample_policy) |
| PPv4N5cudaq8gradient8gradientERRN | -   [cudaq::QPU::getConnectivity  |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[                           |     function)                     |
| 4\]](api/languages/cpp_api.html#_ | ](api/languages/cpp_api.html#_CPP |
| CPPv4N5cudaq8gradient8gradientEv) | v4N5cudaq3QPU15getConnectivityEv) |
| -   [cudaq::gradient::setArgs     | -                                 |
|     (C++                          | [cudaq::QPU::getExecutionThreadId |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/               |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | languages/cpp_api.html#_CPPv4NK5c |
| tArgsEvR13QuantumKernelDpRR4Args) | udaq3QPU20getExecutionThreadIdEv) |
| -   [cudaq::gradient::setKernel   | -   [cudaq::QPU::getNumQubits     |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     functi                        |
| pp_api.html#_CPPv4I0EN5cudaq8grad | on)](api/languages/cpp_api.html#_ |
| ient9setKernelEvR13QuantumKernel) | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| -   [cud                          | -   [                             |
| aq::gradients::central_difference | cudaq::QPU::getRemoteCapabilities |
|     (C++                          |     (C++                          |
|     class)](api/la                |     function)](api/l              |
| nguages/cpp_api.html#_CPPv4N5cuda | anguages/cpp_api.html#_CPPv4NK5cu |
| q9gradients18central_differenceE) | daq3QPU21getRemoteCapabilitiesEv) |
| -   [cudaq::gra                   | -   [cudaq::QPU::isEmulated (C++  |
| dients::central_difference::clone |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     function)](api/languages      | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QPU::isSimulator (C++ |
| ents18central_difference5cloneEv) |     funct                         |
| -   [cudaq::gradi                 | ion)](api/languages/cpp_api.html# |
| ents::central_difference::compute | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     (C++                          | -   [cudaq::QPU::onRandomSeedSet  |
|     function)](                   |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/lang           |
| N5cudaq9gradients18central_differ | uages/cpp_api.html#_CPPv4N5cudaq3 |
| ence7computeERKNSt6vectorIdEERKNS | QPU15onRandomSeedSetENSt6size_tE) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::QPU::QPU (C++         |
|                                   |     functio                       |
|   [\[1\]](api/languages/cpp_api.h | n)](api/languages/cpp_api.html#_C |
| tml#_CPPv4N5cudaq9gradients18cent | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| ral_difference7computeERKNSt6vect |                                   |
| orIdEERNSt6vectorIdEERK7spin_opd) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::gradie                | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| nts::central_difference::gradient |     [\[2\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     functio                       | -   [cudaq::QPU::setId (C++       |
| n)](api/languages/cpp_api.html#_C |     function                      |
| PPv4I00EN5cudaq9gradients18centra | )](api/languages/cpp_api.html#_CP |
| l_difference8gradientER7KernelT), | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     [\[1\]](api/langua            | -   [cudaq::QPU::setShots (C++    |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     f                             |
| q9gradients18central_difference8g | unction)](api/languages/cpp_api.h |
| radientER7KernelTRR10ArgsMapper), | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::                      |
| api.html#_CPPv4I00EN5cudaq9gradie | QPU::supportsExplicitMeasurements |
| nts18central_difference8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/languag        |
|     [\[3\]](api/languages/cpp     | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| _api.html#_CPPv4N5cudaq9gradients | 28supportsExplicitMeasurementsEv) |
| 18central_difference8gradientERRN | -   [cudaq::QPU::\~QPU (C++       |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](api/languages/cp   |
|     [\[4\]](api/languages/cp      | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| p_api.html#_CPPv4N5cudaq9gradient | -   [cudaq::QPUState (C++         |
| s18central_difference8gradientEv) |     class)](api/languages/cpp_    |
| -   [cud                          | api.html#_CPPv4N5cudaq8QPUStateE) |
| aq::gradients::forward_difference | -   [cudaq::qreg (C++             |
|     (C++                          |     class)](api/lan               |
|     class)](api/la                | guages/cpp_api.html#_CPPv4I_NSt6s |
| nguages/cpp_api.html#_CPPv4N5cuda | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| q9gradients18forward_differenceE) | -   [cudaq::qreg::back (C++       |
| -   [cudaq::gra                   |     function)                     |
| dients::forward_difference::clone | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4backENSt6size_tE), |
|     function)](api/languages      |     [\[1\]](api/languages/cpp_ap  |
| /cpp_api.html#_CPPv4N5cudaq9gradi | i.html#_CPPv4N5cudaq4qreg4backEv) |
| ents18forward_difference5cloneEv) | -   [cudaq::qreg::begin (C++      |
| -   [cudaq::gradi                 |                                   |
| ents::forward_difference::compute |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     function)](                   | -   [cudaq::qreg::clear (C++      |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq9gradients18forward_differ |  function)](api/languages/cpp_api |
| ence7computeERKNSt6vectorIdEERKNS | .html#_CPPv4N5cudaq4qreg5clearEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::qreg::front (C++      |
|                                   |     function)]                    |
|   [\[1\]](api/languages/cpp_api.h | (api/languages/cpp_api.html#_CPPv |
| tml#_CPPv4N5cudaq9gradients18forw | 4N5cudaq4qreg5frontENSt6size_tE), |
| ard_difference7computeERKNSt6vect |     [\[1\]](api/languages/cpp_api |
| orIdEERNSt6vectorIdEERK7spin_opd) | .html#_CPPv4N5cudaq4qreg5frontEv) |
| -   [cudaq::gradie                | -   [cudaq::qreg::operator\[\]    |
| nts::forward_difference::gradient |     (C++                          |
|     (C++                          |     functi                        |
|     functio                       | on)](api/languages/cpp_api.html#_ |
| n)](api/languages/cpp_api.html#_C | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| PPv4I00EN5cudaq9gradients18forwar | -   [cudaq::qreg::qreg (C++       |
| d_difference8gradientER7KernelT), |     function)                     |
|     [\[1\]](api/langua            | ](api/languages/cpp_api.html#_CPP |
| ges/cpp_api.html#_CPPv4I00EN5cuda | v4N5cudaq4qreg4qregENSt6size_tE), |
| q9gradients18forward_difference8g |     [\[1\]](api/languages/cpp_ap  |
| radientER7KernelTRR10ArgsMapper), | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::qreg::size (C++       |
| api.html#_CPPv4I00EN5cudaq9gradie |                                   |
| nts18forward_difference8gradientE |  function)](api/languages/cpp_api |
| RR13QuantumKernelRR10ArgsMapper), | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::qreg::slice (C++      |
| _api.html#_CPPv4N5cudaq9gradients |     function)](api/langu          |
| 18forward_difference8gradientERRN | ages/cpp_api.html#_CPPv4N5cudaq4q |
| St8functionIFvNSt6vectorIdEEEEE), | reg5sliceENSt6size_tENSt6size_tE) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qreg::value_type (C++ |
| p_api.html#_CPPv4N5cudaq9gradient |                                   |
| s18forward_difference8gradientEv) | type)](api/languages/cpp_api.html |
| -   [                             | #_CPPv4N5cudaq4qreg10value_typeE) |
| cudaq::gradients::parameter_shift | -   [cudaq::qspan (C++            |
|     (C++                          |     class)](api/lang              |
|     class)](api                   | uages/cpp_api.html#_CPPv4I_NSt6si |
| /languages/cpp_api.html#_CPPv4N5c | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| udaq9gradients15parameter_shiftE) | -   [cudaq::QuakeValue (C++       |
| -   [cudaq::                      |     class)](api/languages/cpp_api |
| gradients::parameter_shift::clone | .html#_CPPv4N5cudaq10QuakeValueE) |
|     (C++                          | -   [cudaq::Q                     |
|     function)](api/langua         | uakeValue::canValidateNumElements |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     (C++                          |
| adients15parameter_shift5cloneEv) |     function)](api/languages      |
| -   [cudaq::gr                    | /cpp_api.html#_CPPv4N5cudaq10Quak |
| adients::parameter_shift::compute | eValue22canValidateNumElementsEv) |
|     (C++                          | -                                 |
|     function                      |  [cudaq::QuakeValue::constantSize |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq9gradients15parameter_s |     function)](api                |
| hift7computeERKNSt6vectorIdEERKNS | /languages/cpp_api.html#_CPPv4N5c |
| t8functionIFdNSt6vectorIdEEEEEd), | udaq10QuakeValue12constantSizeEv) |
|     [\[1\]](api/languages/cpp_ap  | -   [cudaq::QuakeValue::dump (C++ |
| i.html#_CPPv4N5cudaq9gradients15p |     function)](api/lan            |
| arameter_shift7computeERKNSt6vect | guages/cpp_api.html#_CPPv4N5cudaq |
| orIdEERNSt6vectorIdEERK7spin_opd) | 10QuakeValue4dumpERNSt7ostreamE), |
| -   [cudaq::gra                   |     [\                            |
| dients::parameter_shift::gradient | [1\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq10QuakeValue4dumpEv) |
|     func                          | -   [cudaq                        |
| tion)](api/languages/cpp_api.html | ::QuakeValue::getRequiredElements |
| #_CPPv4I00EN5cudaq9gradients15par |     (C++                          |
| ameter_shift8gradientER7KernelT), |     function)](api/langua         |
|     [\[1\]](api/lan               | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| guages/cpp_api.html#_CPPv4I00EN5c | uakeValue19getRequiredElementsEv) |
| udaq9gradients15parameter_shift8g | -   [cudaq::QuakeValue::getValue  |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\]](api/languages/c       |     function)]                    |
| pp_api.html#_CPPv4I00EN5cudaq9gra | (api/languages/cpp_api.html#_CPPv |
| dients15parameter_shift8gradientE | 4NK5cudaq10QuakeValue8getValueEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::QuakeValue::inverse   |
|     [\[3\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq9gradie |     function)                     |
| nts15parameter_shift8gradientERRN | ](api/languages/cpp_api.html#_CPP |
| St8functionIFvNSt6vectorIdEEEEE), | v4NK5cudaq10QuakeValue7inverseEv) |
|     [\[4\]](api/languages         | -   [cudaq::QuakeValue::isStdVec  |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents15parameter_shift8gradientEv) |     function)                     |
| -   [cudaq::kernel_builder (C++   | ](api/languages/cpp_api.html#_CPP |
|     clas                          | v4N5cudaq10QuakeValue8isStdVecEv) |
| s)](api/languages/cpp_api.html#_C | -                                 |
| PPv4IDpEN5cudaq14kernel_builderE) |    [cudaq::QuakeValue::operator\* |
| -   [c                            |     (C++                          |
| udaq::kernel_builder::constantVal |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/la             | udaq10QuakeValuemlE10QuakeValue), |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q14kernel_builder11constantValEd) | [\[1\]](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|  [cudaq::kernel_builder::detector | -   [cudaq::QuakeValue::operator+ |
|     (C++                          |     (C++                          |
|                                   |     function)](api                |
|    function)](api/languages/cpp_a | /languages/cpp_api.html#_CPPv4N5c |
| pi.html#_CPPv4IDpEN5cudaq14kernel | udaq10QuakeValueplE10QuakeValue), |
| _builder8detectorEvDpRR8MeasArgs) |     [                             |
| -                                 | \[1\]](api/languages/cpp_api.html |
| [cudaq::kernel_builder::detectors | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     (C++                          |                                   |
|     func                          | [\[2\]](api/languages/cpp_api.htm |
| tion)](api/languages/cpp_api.html | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| #_CPPv4N5cudaq14kernel_builder9de | -   [cudaq::QuakeValue::operator- |
| tectorsE10QuakeValue10QuakeValue) |     (C++                          |
| -   [cu                           |     function)](api                |
| daq::kernel_builder::getArguments | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuemiE10QuakeValue), |
|     function)](api/lan            |     [                             |
| guages/cpp_api.html#_CPPv4N5cudaq | \[1\]](api/languages/cpp_api.html |
| 14kernel_builder12getArgumentsEv) | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| -   [cu                           |     [                             |
| daq::kernel_builder::getNumParams | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     function)](api/lan            |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq | [\[3\]](api/languages/cpp_api.htm |
| 14kernel_builder12getNumParamsEv) | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| -   [c                            | -   [cudaq::QuakeValue::operator/ |
| udaq::kernel_builder::isArgStdVec |     (C++                          |
|     (C++                          |     function)](api                |
|     function)](api/languages/cp   | /languages/cpp_api.html#_CPPv4N5c |
| p_api.html#_CPPv4N5cudaq14kernel_ | udaq10QuakeValuedvE10QuakeValue), |
| builder11isArgStdVecENSt6size_tE) |                                   |
| -   [cuda                         | [\[1\]](api/languages/cpp_api.htm |
| q::kernel_builder::kernel_builder | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
|     (C++                          | -                                 |
|     function)](api/languages/cpp  |  [cudaq::QuakeValue::operator\[\] |
| _api.html#_CPPv4N5cudaq14kernel_b |     (C++                          |
| uilder14kernel_builderERNSt6vecto |     function)](api                |
| rIN6detail17KernelBuilderTypeEEE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::k                     | udaq10QuakeValueixEKNSt6size_tE), |
| ernel_builder::logical_observable |     [\[1\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     function)                     | daq10QuakeValueixERK10QuakeValue) |
| ](api/languages/cpp_api.html#_CPP | -                                 |
| v4IDpEN5cudaq14kernel_builder18lo |    [cudaq::QuakeValue::QuakeValue |
| gical_observableEvDpRR8MeasArgs), |     (C++                          |
|     [\[1\]](ap                    |     function)](api/languag        |
| i/languages/cpp_api.html#_CPPv4N5 | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| cudaq14kernel_builder18logical_ob | akeValue10QuakeValueERN4mlir20Imp |
| servableE10QuakeValueNSt6size_tE) | licitLocOpBuilderEN4mlir5ValueE), |
| -   [cudaq::kernel_builder::name  |     [\[1\]                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)                     | v4N5cudaq10QuakeValue10QuakeValue |
| ](api/languages/cpp_api.html#_CPP | ERN4mlir20ImplicitLocOpBuilderEd) |
| v4N5cudaq14kernel_builder4nameEv) | -   [cudaq::QuakeValue::size (C++ |
| -                                 |     funct                         |
|    [cudaq::kernel_builder::qalloc | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|     function)](api/language       | -   [cudaq::QuakeValue::slice     |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     (C++                          |
| nel_builder6qallocE10QuakeValue), |     function)](api/languages/cpp_ |
|     [\[1\]](api/language          | api.html#_CPPv4N5cudaq10QuakeValu |
| s/cpp_api.html#_CPPv4N5cudaq14ker | e5sliceEKNSt6size_tEKNSt6size_tE) |
| nel_builder6qallocEKNSt6size_tE), | -   [cudaq::quantum_platform (C++ |
|     [\[2                          |     cl                            |
| \]](api/languages/cpp_api.html#_C | ass)](api/languages/cpp_api.html# |
| PPv4N5cudaq14kernel_builder6qallo | _CPPv4N5cudaq16quantum_platformE) |
| cERNSt6vectorINSt7complexIdEEEE), | -   [cudaq:                       |
|     [\[3\]](                      | :quantum_platform::beginExecution |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14kernel_builder6qallocEv) |     function)](api/languag        |
| -   [cudaq::kernel_builder::swap  | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14beginExecutionEv) |
|     function)](api/language       | -   [cudaq::quantum_pl            |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | atform::configureExecutionContext |
| 4kernel_builder4swapEvRK10QuakeVa |     (C++                          |
| lueRK10QuakeValueRK10QuakeValue), |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
| [\[1\]](api/languages/cpp_api.htm | 16quantum_platform25configureExec |
| l#_CPPv4I00EN5cudaq14kernel_build | utionContextER16ExecutionContext) |
| er4swapEvRKNSt6vectorI10QuakeValu | -   [cuda                         |
| eEERK10QuakeValueRK10QuakeValue), | q::quantum_platform::connectivity |
|                                   |     (C++                          |
| [\[2\]](api/languages/cpp_api.htm |     function)](api/langu          |
| l#_CPPv4N5cudaq14kernel_builder4s | ages/cpp_api.html#_CPPv4N5cudaq16 |
| wapERK10QuakeValueRK10QuakeValue) | quantum_platform12connectivityEv) |
| -   [cudaq::KernelExecutionTask   | -   [cuda                         |
|     (C++                          | q::quantum_platform::endExecution |
|     type                          |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/langu          |
| Pv4N5cudaq19KernelExecutionTaskE) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [cudaq::KernelThunkResultType | quantum_platform12endExecutionEv) |
|     (C++                          | -   [cudaq::q                     |
|     struct)]                      | uantum_platform::enqueueAsyncTask |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq21KernelThunkResultTypeE) |     function)](api/languages/     |
| -   [cudaq::KernelThunkType (C++  | cpp_api.html#_CPPv4N5cudaq16quant |
|                                   | um_platform16enqueueAsyncTaskEKNS |
| type)](api/languages/cpp_api.html | t6size_tER19KernelExecutionTask), |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     [\[1\]](api/languag           |
| -   [cudaq::kraus_channel (C++    | es/cpp_api.html#_CPPv4N5cudaq16qu |
|                                   | antum_platform16enqueueAsyncTaskE |
|  class)](api/languages/cpp_api.ht | KNSt6size_tERNSt8functionIFvvEEE) |
| ml#_CPPv4N5cudaq13kraus_channelE) | -   [cudaq::quantum_p             |
| -   [cudaq::kraus_channel::empty  | latform::finalizeExecutionContext |
|     (C++                          |     (C++                          |
|     function)]                    |     function)](api/languages/c    |
| (api/languages/cpp_api.html#_CPPv | pp_api.html#_CPPv4NK5cudaq16quant |
| 4NK5cudaq13kraus_channel5emptyEv) | um_platform24finalizeExecutionCon |
| -   [cudaq::kraus_c               | textERN5cudaq16ExecutionContextE) |
| hannel::generateUnitaryParameters | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_codegen_config |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](api/languages/c    |
| pi.html#_CPPv4N5cudaq13kraus_chan | pp_api.html#_CPPv4N5cudaq16quantu |
| nel25generateUnitaryParametersEv) | m_platform18get_codegen_configEv) |
| -                                 | -   [cuda                         |
|    [cudaq::kraus_channel::get_ops | q::quantum_platform::get_exec_ctx |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](api/langua         |
| pi/languages/cpp_api.html#_CPPv4N | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| K5cudaq13kraus_channel7get_opsEv) | quantum_platform12get_exec_ctxEv) |
| -   [cud                          | -   [c                            |
| aq::kraus_channel::identity_flags | udaq::quantum_platform::get_noise |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/languages/c    |
| guages/cpp_api.html#_CPPv4N5cudaq | pp_api.html#_CPPv4N5cudaq16quantu |
| 13kraus_channel14identity_flagsE) | m_platform9get_noiseENSt6size_tE) |
| -   [cud                          | -   [cudaq:                       |
| aq::kraus_channel::is_identity_op | :quantum_platform::get_num_qubits |
|     (C++                          |     (C++                          |
|                                   |                                   |
|    function)](api/languages/cpp_a | function)](api/languages/cpp_api. |
| pi.html#_CPPv4NK5cudaq13kraus_cha | html#_CPPv4NK5cudaq16quantum_plat |
| nnel14is_identity_opENSt6size_tE) | form14get_num_qubitsENSt6size_tE) |
| -   [cudaq::                      | -   [cudaq::quantum_              |
| kraus_channel::is_unitary_mixture | platform::get_remote_capabilities |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)                     |
| /cpp_api.html#_CPPv4NK5cudaq13kra | ](api/languages/cpp_api.html#_CPP |
| us_channel18is_unitary_mixtureEv) | v4NK5cudaq16quantum_platform23get |
| -   [cu                           | _remote_capabilitiesENSt6size_tE) |
| daq::kraus_channel::kraus_channel | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_runtime_target |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4IDpEN5cu |     function)](api/languages/cp   |
| daq13kraus_channel13kraus_channel | p_api.html#_CPPv4NK5cudaq16quantu |
| EDpRRNSt16initializer_listI1TEE), | m_platform18get_runtime_targetEv) |
|                                   | -   [cud                          |
|  [\[1\]](api/languages/cpp_api.ht | aq::quantum_platform::is_emulated |
| ml#_CPPv4N5cudaq13kraus_channel13 |     (C++                          |
| kraus_channelERK13kraus_channel), |                                   |
|     [\[2\]                        |    function)](api/languages/cpp_a |
| ](api/languages/cpp_api.html#_CPP | pi.html#_CPPv4NK5cudaq16quantum_p |
| v4N5cudaq13kraus_channel13kraus_c | latform11is_emulatedENSt6size_tE) |
| hannelERKNSt6vectorI8kraus_opEE), | -   [cudaq::                      |
|     [\[3\]                        | quantum_platform::is_library_mode |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq13kraus_channel13kraus_c |     function)](api/languages      |
| hannelERRNSt6vectorI8kraus_opEE), | /cpp_api.html#_CPPv4NK5cudaq16qua |
|     [\[4\]](api/lan               | ntum_platform15is_library_modeEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [c                            |
| 13kraus_channel13kraus_channelEv) | udaq::quantum_platform::is_remote |
| -                                 |     (C++                          |
| [cudaq::kraus_channel::noise_type |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4NK5cudaq16quantu |
|     member)](api                  | m_platform9is_remoteENSt6size_tE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cuda                         |
| udaq13kraus_channel10noise_typeE) | q::quantum_platform::is_simulator |
| -                                 |     (C++                          |
|   [cudaq::kraus_channel::op_names |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     member)](                     | i.html#_CPPv4NK5cudaq16quantum_pl |
| api/languages/cpp_api.html#_CPPv4 | atform12is_simulatorENSt6size_tE) |
| N5cudaq13kraus_channel8op_namesE) | -   [c                            |
| -                                 | udaq::quantum_platform::launchVQE |
|  [cudaq::kraus_channel::operator= |     (C++                          |
|     (C++                          |     function)](                   |
|     function)](api/langua         | api/languages/cpp_api.html#_CPPv4 |
| ges/cpp_api.html#_CPPv4N5cudaq13k | N5cudaq16quantum_platform9launchV |
| raus_channelaSERK13kraus_channel) | QEEKNSt6stringEPKvPN5cudaq8gradie |
| -   [c                            | ntERKN5cudaq7spin_opERN5cudaq9opt |
| udaq::kraus_channel::operator\[\] | imizerEKiKNSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/l              | :quantum_platform::list_platforms |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq13kraus_channelixEKNSt6size_tE) |     function)](api/languag        |
| -                                 | es/cpp_api.html#_CPPv4N5cudaq16qu |
| [cudaq::kraus_channel::parameters | antum_platform14list_platformsEv) |
|     (C++                          | -                                 |
|     member)](api                  |    [cudaq::quantum_platform::name |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel10parametersE) |     function)](a                  |
| -   [cudaq::krau                  | pi/languages/cpp_api.html#_CPPv4N |
| s_channel::populateDefaultOpNames | K5cudaq16quantum_platform4nameEv) |
|     (C++                          | -   [                             |
|     function)](api/languages/cp   | cudaq::quantum_platform::num_qpus |
| p_api.html#_CPPv4N5cudaq13kraus_c |     (C++                          |
| hannel22populateDefaultOpNamesEv) |     function)](api/l              |
| -   [cu                           | anguages/cpp_api.html#_CPPv4NK5cu |
| daq::kraus_channel::probabilities | daq16quantum_platform8num_qpusEv) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/la               | quantum_platform::onRandomSeedSet |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q13kraus_channel13probabilitiesE) |                                   |
| -                                 | function)](api/languages/cpp_api. |
|  [cudaq::kraus_channel::push_back | html#_CPPv4N5cudaq16quantum_platf |
|     (C++                          | orm15onRandomSeedSetENSt6size_tE) |
|     function)](api                | -   [cudaq:                       |
| /languages/cpp_api.html#_CPPv4N5c | :quantum_platform::reset_exec_ctx |
| udaq13kraus_channel9push_backE8kr |     (C++                          |
| aus_opNSt8optionalINSt6stringEEE) |     function)](api/languag        |
| -   [cudaq::kraus_channel::size   | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14reset_exec_ctxEv) |
|     function)                     | -   [cud                          |
| ](api/languages/cpp_api.html#_CPP | aq::quantum_platform::reset_noise |
| v4NK5cudaq13kraus_channel4sizeEv) |     (C++                          |
| -   [                             |     function)](api/languages/cpp_ |
| cudaq::kraus_channel::unitary_ops | api.html#_CPPv4N5cudaq16quantum_p |
|     (C++                          | latform11reset_noiseENSt6size_tE) |
|     member)](api/                 | -   [cuda                         |
| languages/cpp_api.html#_CPPv4N5cu | q::quantum_platform::set_exec_ctx |
| daq13kraus_channel11unitary_opsE) |     (C++                          |
| -   [cudaq::kraus_op (C++         |     funct                         |
|     struct)](api/languages/cpp_   | ion)](api/languages/cpp_api.html# |
| api.html#_CPPv4N5cudaq8kraus_opE) | _CPPv4N5cudaq16quantum_platform12 |
| -   [cudaq::kraus_op::adjoint     | set_exec_ctxEP16ExecutionContext) |
|     (C++                          | -   [c                            |
|     functi                        | udaq::quantum_platform::set_noise |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4NK5cudaq8kraus_op7adjointEv) |     function                      |
| -   [cudaq::kraus_op::data (C++   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq16quantum_platform9set_ |
|  member)](api/languages/cpp_api.h | noiseEPK11noise_modelNSt6size_tE) |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | -   [cudaq::quantum_platfor       |
| -   [cudaq::kraus_op::kraus_op    | m::supports_explicit_measurements |
|     (C++                          |     (C++                          |
|     func                          |     function)](api/l              |
| tion)](api/languages/cpp_api.html | anguages/cpp_api.html#_CPPv4NK5cu |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | daq16quantum_platform30supports_e |
| opERRNSt16initializer_listI1TEE), | xplicit_measurementsENSt6size_tE) |
|                                   | -   [cudaq::quantum_pla           |
|  [\[1\]](api/languages/cpp_api.ht | tform::supports_task_distribution |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o |     (C++                          |
| pENSt6vectorIN5cudaq7complexEEE), |     fu                            |
|     [\[2\]](api/l                 | nction)](api/languages/cpp_api.ht |
| anguages/cpp_api.html#_CPPv4N5cud | ml#_CPPv4NK5cudaq16quantum_platfo |
| aq8kraus_op8kraus_opERK8kraus_op) | rm26supports_task_distributionEv) |
| -   [cudaq::kraus_op::nCols (C++  | -   [cudaq::quantum               |
|                                   | _platform::with_execution_context |
| member)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     function)                     |
| -   [cudaq::kraus_op::nRows (C++  | ](api/languages/cpp_api.html#_CPP |
|                                   | v4I0DpEN5cudaq16quantum_platform2 |
| member)](api/languages/cpp_api.ht | 2with_execution_contextEDaR16Exec |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | utionContextRR8CallableDpRR4Args) |
| -   [cudaq::kraus_op::operator=   | -   [cudaq::QuantumTask (C++      |
|     (C++                          |     type)](api/languages/cpp_api. |
|     function)                     | html#_CPPv4N5cudaq11QuantumTaskE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::qubit (C++            |
| v4N5cudaq8kraus_opaSERK8kraus_op) |     type)](api/languages/c        |
| -   [cudaq::kraus_op::precision   | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     (C++                          | -   [cudaq::QubitConnectivity     |
|     memb                          |     (C++                          |
| er)](api/languages/cpp_api.html#_ |     ty                            |
| CPPv4N5cudaq8kraus_op9precisionE) | pe)](api/languages/cpp_api.html#_ |
| -   [cudaq::KrausSelection (C++   | CPPv4N5cudaq17QubitConnectivityE) |
|     s                             | -   [cudaq::QubitEdge (C++        |
| truct)](api/languages/cpp_api.htm |     type)](api/languages/cpp_a    |
| l#_CPPv4N5cudaq14KrausSelectionE) | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| -   [cudaq:                       | -   [cudaq::qudit (C++            |
| :KrausSelection::circuit_location |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     member)](api/langua           | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| ges/cpp_api.html#_CPPv4N5cudaq14K | -   [cudaq::qudit::qudit (C++     |
| rausSelection16circuit_locationE) |                                   |
| -                                 | function)](api/languages/cpp_api. |
|  [cudaq::KrausSelection::is_error | html#_CPPv4N5cudaq5qudit5quditEv) |
|     (C++                          | -   [cudaq::qvector (C++          |
|     member)](a                    |     class)                        |
| pi/languages/cpp_api.html#_CPPv4N | ](api/languages/cpp_api.html#_CPP |
| 5cudaq14KrausSelection8is_errorE) | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| -   [cudaq::Kra                   | -   [cudaq::qvector::back (C++    |
| usSelection::kraus_operator_index |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     member)](api/languages/       | 5cudaq7qvector4backENSt6size_tE), |
| cpp_api.html#_CPPv4N5cudaq14Kraus |                                   |
| Selection20kraus_operator_indexE) |   [\[1\]](api/languages/cpp_api.h |
| -   [cuda                         | tml#_CPPv4N5cudaq7qvector4backEv) |
| q::KrausSelection::KrausSelection | -   [cudaq::qvector::begin (C++   |
|     (C++                          |     fu                            |
|     function)](a                  | nction)](api/languages/cpp_api.ht |
| pi/languages/cpp_api.html#_CPPv4N | ml#_CPPv4N5cudaq7qvector5beginEv) |
| 5cudaq14KrausSelection14KrausSele | -   [cudaq::qvector::clear (C++   |
| ctionENSt6size_tENSt6vectorINSt6s |     fu                            |
| ize_tEEENSt6stringENSt6size_tEb), | nction)](api/languages/cpp_api.ht |
|     [\[1\]](api/langu             | ml#_CPPv4N5cudaq7qvector5clearEv) |
| ages/cpp_api.html#_CPPv4N5cudaq14 | -   [cudaq::qvector::end (C++     |
| KrausSelection14KrausSelectionEv) |                                   |
| -                                 | function)](api/languages/cpp_api. |
|   [cudaq::KrausSelection::op_name | html#_CPPv4N5cudaq7qvector3endEv) |
|     (C++                          | -   [cudaq::qvector::front (C++   |
|     member)](                     |     function)](ap                 |
| api/languages/cpp_api.html#_CPPv4 | i/languages/cpp_api.html#_CPPv4N5 |
| N5cudaq14KrausSelection7op_nameE) | cudaq7qvector5frontENSt6size_tE), |
| -   [                             |                                   |
| cudaq::KrausSelection::operator== |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     function)](api/languages      | -   [cudaq::qvector::operator=    |
| /cpp_api.html#_CPPv4NK5cudaq14Kra |     (C++                          |
| usSelectioneqERK14KrausSelection) |     functio                       |
| -                                 | n)](api/languages/cpp_api.html#_C |
|    [cudaq::KrausSelection::qubits | PPv4N5cudaq7qvectoraSERK7qvector) |
|     (C++                          | -   [cudaq::qvector::operator\[\] |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)                     |
| 4N5cudaq14KrausSelection6qubitsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::KrausTrajectory (C++  | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     st                            | -   [cudaq::qvector::qvector (C++ |
| ruct)](api/languages/cpp_api.html |     function)](api/               |
| #_CPPv4N5cudaq15KrausTrajectoryE) | languages/cpp_api.html#_CPPv4N5cu |
| -                                 | daq7qvector7qvectorENSt6size_tE), |
|  [cudaq::KrausTrajectory::builder |     [\[1\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function)](ap                 | 5cudaq7qvector7qvectorERK5state), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[2\]](api                   |
| cudaq15KrausTrajectory7builderEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cu                           | udaq7qvector7qvectorERK7qvector), |
| daq::KrausTrajectory::countErrors |     [\[3\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/lang           | cudaq7qvector7qvectorERR7qvector) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::qvector::size (C++    |
| 15KrausTrajectory11countErrorsEv) |     fu                            |
| -   [                             | nction)](api/languages/cpp_api.ht |
| cudaq::KrausTrajectory::isOrdered | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     (C++                          | -   [cudaq::qvector::slice (C++   |
|     function)](api/l              |     function)](api/language       |
| anguages/cpp_api.html#_CPPv4NK5cu | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| daq15KrausTrajectory9isOrderedEv) | tor5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::                      | -   [cudaq::qvector::value_type   |
| KrausTrajectory::kraus_selections |     (C++                          |
|     (C++                          |     typ                           |
|     member)](api/languag          | e)](api/languages/cpp_api.html#_C |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | PPv4N5cudaq7qvector10value_typeE) |
| ausTrajectory16kraus_selectionsE) | -   [cudaq::qview (C++            |
| -   [cudaq:                       |     clas                          |
| :KrausTrajectory::KrausTrajectory | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     function                      | -   [cudaq::qview::back (C++      |
| )](api/languages/cpp_api.html#_CP |     function)                     |
| Pv4N5cudaq15KrausTrajectory15Krau | ](api/languages/cpp_api.html#_CPP |
| sTrajectoryENSt6size_tENSt6vector | v4N5cudaq5qview4backENSt6size_tE) |
| I14KrausSelectionEEdNSt6size_tE), | -   [cudaq::qview::begin (C++     |
|     [\[1\]](api/languag           |                                   |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | function)](api/languages/cpp_api. |
| ausTrajectory15KrausTrajectoryEv) | html#_CPPv4N5cudaq5qview5beginEv) |
| -   [cudaq::Kr                    | -   [cudaq::qview::end (C++       |
| ausTrajectory::measurement_counts |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     member)](api/languages        | i.html#_CPPv4N5cudaq5qview3endEv) |
| /cpp_api.html#_CPPv4N5cudaq15Krau | -   [cudaq::qview::front (C++     |
| sTrajectory18measurement_countsE) |     function)](                   |
| -   [cud                          | api/languages/cpp_api.html#_CPPv4 |
| aq::KrausTrajectory::multiplicity | N5cudaq5qview5frontENSt6size_tE), |
|     (C++                          |                                   |
|     member)](api/lan              |    [\[1\]](api/languages/cpp_api. |
| guages/cpp_api.html#_CPPv4N5cudaq | html#_CPPv4N5cudaq5qview5frontEv) |
| 15KrausTrajectory12multiplicityE) | -   [cudaq::qview::operator\[\]   |
| -   [                             |     (C++                          |
| cudaq::KrausTrajectory::num_shots |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](api                  | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::qview::qview (C++     |
| udaq15KrausTrajectory9num_shotsE) |     functio                       |
| -   [c                            | n)](api/languages/cpp_api.html#_C |
| udaq::KrausTrajectory::operator== | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     (C++                          |     [\[1                          |
|     function)](api/languages/c    | \]](api/languages/cpp_api.html#_C |
| pp_api.html#_CPPv4NK5cudaq15Kraus | PPv4N5cudaq5qview5qviewERK5qview) |
| TrajectoryeqERK15KrausTrajectory) | -   [cudaq::qview::size (C++      |
| -   [cu                           |                                   |
| daq::KrausTrajectory::probability | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4NK5cudaq5qview4sizeEv) |
|     member)](api/la               | -   [cudaq::qview::slice (C++     |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/langua         |
| q15KrausTrajectory11probabilityE) | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| -   [cuda                         | iew5sliceENSt6size_tENSt6size_tE) |
| q::KrausTrajectory::trajectory_id | -   [cudaq::qview::value_type     |
|     (C++                          |     (C++                          |
|     member)](api/lang             |     t                             |
| uages/cpp_api.html#_CPPv4N5cudaq1 | ype)](api/languages/cpp_api.html# |
| 5KrausTrajectory13trajectory_idE) | _CPPv4N5cudaq5qview10value_typeE) |
| -                                 | -   [cudaq::range (C++            |
|   [cudaq::KrausTrajectory::weight |     fun                           |
|     (C++                          | ction)](api/languages/cpp_api.htm |
|     member)](                     | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| api/languages/cpp_api.html#_CPPv4 | orI11ElementTypeEE11ElementType), |
| N5cudaq15KrausTrajectory6weightE) |     [\[1\]](api/languages/cpp_    |
| -                                 | api.html#_CPPv4I0EN5cudaq5rangeEN |
|    [cudaq::KrausTrajectoryBuilder | St6vectorI11ElementTypeEE11Elemen |
|     (C++                          | tType11ElementType11ElementType), |
|     class)](                      |     [                             |
| api/languages/cpp_api.html#_CPPv4 | \[2\]](api/languages/cpp_api.html |
| N5cudaq22KrausTrajectoryBuilderE) | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| -   [cud                          | -   [cudaq::real (C++             |
| aq::KrausTrajectoryBuilder::build |     type)](api/languages/         |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq4realE) |
|     function)](api/lang           | -   [cudaq::registry (C++         |
| uages/cpp_api.html#_CPPv4NK5cudaq |     type)](api/languages/cpp_     |
| 22KrausTrajectoryBuilder5buildEv) | api.html#_CPPv4N5cudaq8registryE) |
| -   [cud                          | -                                 |
| aq::KrausTrajectoryBuilder::setId |  [cudaq::registry::RegisteredType |
|     (C++                          |     (C++                          |
|     function)](api/languages/cpp  |     class)](api/                  |
| _api.html#_CPPv4N5cudaq22KrausTra | languages/cpp_api.html#_CPPv4I0EN |
| jectoryBuilder5setIdENSt6size_tE) | 5cudaq8registry14RegisteredTypeE) |
| -   [cudaq::Kraus                 | -   [cudaq::RemoteCapabilities    |
| TrajectoryBuilder::setProbability |     (C++                          |
|     (C++                          |     struc                         |
|     function)](api/languages/cpp  | t)](api/languages/cpp_api.html#_C |
| _api.html#_CPPv4N5cudaq22KrausTra | PPv4N5cudaq18RemoteCapabilitiesE) |
| jectoryBuilder14setProbabilityEd) | -   [cudaq::Remot                 |
| -   [cudaq::Krau                  | eCapabilities::RemoteCapabilities |
| sTrajectoryBuilder::setSelections |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api/languag        | _api.html#_CPPv4N5cudaq18RemoteCa |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | pabilities18RemoteCapabilitiesEb) |
| ausTrajectoryBuilder13setSelectio | -   [cudaq:                       |
| nsENSt6vectorI14KrausSelectionEE) | :RemoteCapabilities::stateOverlap |
| -   [cudaq::logical_observable    |     (C++                          |
|     (C++                          |     member)](api/langua           |
|     function)](api/languages/c    | ges/cpp_api.html#_CPPv4N5cudaq18R |
| pp_api.html#_CPPv4IDpEN5cudaq18lo | emoteCapabilities12stateOverlapE) |
| gical_observableEvDpRR8MeasArgs), | -                                 |
|     [\[1\]](api/l                 |   [cudaq::RemoteCapabilities::vqe |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq18logical_observableERKNSt6vect |     member)](                     |
| orI14measure_resultEENSt6size_tE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::matrix_callback (C++  | N5cudaq18RemoteCapabilities3vqeE) |
|     c                             | -   [cudaq::Resources (C++        |
| lass)](api/languages/cpp_api.html |     class)](api/languages/cpp_a   |
| #_CPPv4N5cudaq15matrix_callbackE) | pi.html#_CPPv4N5cudaq9ResourcesE) |
| -   [cudaq::matrix_handler (C++   | -   [cudaq::run (C++              |
|                                   |     function)]                    |
| class)](api/languages/cpp_api.htm | (api/languages/cpp_api.html#_CPPv |
| l#_CPPv4N5cudaq14matrix_handlerE) | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| -   [cudaq::mat                   | 5invoke_result_tINSt7decay_tI13Qu |
| rix_handler::commutation_behavior | antumKernelEEDpNSt7decay_tI4ARGSE |
|     (C++                          | EEEEENSt6size_tERN5cudaq11noise_m |
|     struct)](api/languages/       | odelERR13QuantumKernelDpRR4ARGS), |
| cpp_api.html#_CPPv4N5cudaq14matri |     [\[1\]](api/langu             |
| x_handler20commutation_behaviorE) | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| -                                 | daq3runENSt6vectorINSt15invoke_re |
|    [cudaq::matrix_handler::define | sult_tINSt7decay_tI13QuantumKerne |
|     (C++                          | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
|     function)](a                  | ize_tERR13QuantumKernelDpRR4ARGS) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::run_async (C++        |
| 5cudaq14matrix_handler6defineENSt |     functio                       |
| 6stringENSt6vectorINSt7int64_tEEE | n)](api/languages/cpp_api.html#_C |
| RR15matrix_callbackRKNSt13unorder | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| ed_mapINSt6stringENSt6stringEEE), | tureINSt6vectorINSt15invoke_resul |
|                                   | t_tINSt7decay_tI13QuantumKernelEE |
| [\[1\]](api/languages/cpp_api.htm | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| l#_CPPv4N5cudaq14matrix_handler6d | ze_tENSt6size_tERN5cudaq11noise_m |
| efineENSt6stringENSt6vectorINSt7i | odelERR13QuantumKernelDpRR4ARGS), |
| nt64_tEEERR15matrix_callbackRR20d |     [\[1\]](api/la                |
| iag_matrix_callbackRKNSt13unorder | nguages/cpp_api.html#_CPPv4I0DpEN |
| ed_mapINSt6stringENSt6stringEEE), | 5cudaq9run_asyncENSt6futureINSt6v |
|     [\[2\]](                      | ectorINSt15invoke_result_tINSt7de |
| api/languages/cpp_api.html#_CPPv4 | cay_tI13QuantumKernelEEDpNSt7deca |
| N5cudaq14matrix_handler6defineENS | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| t6stringENSt6vectorINSt7int64_tEE | ize_tERR13QuantumKernelDpRR4ARGS) |
| ERR15matrix_callbackRRNSt13unorde | -   [cudaq::RuntimeTarget (C++    |
| red_mapINSt6stringENSt6stringEEE) |                                   |
| -                                 | struct)](api/languages/cpp_api.ht |
|   [cudaq::matrix_handler::degrees | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     (C++                          | -   [cudaq::sample (C++           |
|     function)](ap                 |     function)](api/languages/c    |
| i/languages/cpp_api.html#_CPPv4NK | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| 5cudaq14matrix_handler7degreesEv) | mpleE13sample_resultRK14sample_op |
| -                                 | tionsRR13QuantumKernelDpRR4Args), |
|  [cudaq::matrix_handler::displace |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/language       | Pv4I0DpEN5cudaq6sampleE13sample_r |
| s/cpp_api.html#_CPPv4N5cudaq14mat | esultRR13QuantumKernelDpRR4Args), |
| rix_handler8displaceENSt6size_tE) |     [\                            |
| -   [cudaq::matrix                | [2\]](api/languages/cpp_api.html# |
| _handler::get_expected_dimensions | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4Args) |
|                                   | -   [cudaq::sample_options (C++   |
|    function)](api/languages/cpp_a |     s                             |
| pi.html#_CPPv4NK5cudaq14matrix_ha | truct)](api/languages/cpp_api.htm |
| ndler23get_expected_dimensionsEv) | l#_CPPv4N5cudaq14sample_optionsE) |
| -   [cudaq::matrix_ha             | -   [cudaq::sample_result (C++    |
| ndler::get_parameter_descriptions |                                   |
|     (C++                          |  class)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4N5cudaq13sample_resultE) |
| function)](api/languages/cpp_api. | -   [cudaq::sample_result::append |
| html#_CPPv4NK5cudaq14matrix_handl |     (C++                          |
| er26get_parameter_descriptionsEv) |     function)](api/languages/cpp_ |
| -   [c                            | api.html#_CPPv4N5cudaq13sample_re |
| udaq::matrix_handler::instantiate | sult6appendERK15ExecutionResultb) |
|     (C++                          | -   [cudaq::sample_result::begin  |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)]                    |
| 5cudaq14matrix_handler11instantia | (api/languages/cpp_api.html#_CPPv |
| teENSt6stringERKNSt6vectorINSt6si | 4N5cudaq13sample_result5beginEv), |
| ze_tEEERK20commutation_behavior), |     [\[1\]]                       |
|     [\[1\]](                      | (api/languages/cpp_api.html#_CPPv |
| api/languages/cpp_api.html#_CPPv4 | 4NK5cudaq13sample_result5beginEv) |
| N5cudaq14matrix_handler11instanti | -   [cudaq::sample_result::cbegin |
| ateENSt6stringERRNSt6vectorINSt6s |     (C++                          |
| ize_tEEERK20commutation_behavior) |     function)](                   |
| -   [cuda                         | api/languages/cpp_api.html#_CPPv4 |
| q::matrix_handler::matrix_handler | NK5cudaq13sample_result6cbeginEv) |
|     (C++                          | -   [cudaq::sample_result::cend   |
|     function)](api/languag        |     (C++                          |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     function)                     |
| ble_if_tINSt12is_base_of_vI16oper | ](api/languages/cpp_api.html#_CPP |
| ator_handler1TEEbEEEN5cudaq14matr | v4NK5cudaq13sample_result4cendEv) |
| ix_handler14matrix_handlerERK1T), | -   [cudaq::sample_result::clear  |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4I0 |     function)                     |
| _NSt11enable_if_tINSt12is_base_of | ](api/languages/cpp_api.html#_CPP |
| _vI16operator_handler1TEEbEEEN5cu | v4N5cudaq13sample_result5clearEv) |
| daq14matrix_handler14matrix_handl | -   [cudaq::sample_result::count  |
| erERK1TRK20commutation_behavior), |     (C++                          |
|     [\[2\]](api/languages/cpp_ap  |     function)](                   |
| i.html#_CPPv4N5cudaq14matrix_hand | api/languages/cpp_api.html#_CPPv4 |
| ler14matrix_handlerENSt6size_tE), | NK5cudaq13sample_result5countENSt |
|     [\[3\]](api/                  | 11string_viewEKNSt11string_viewE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [                             |
| daq14matrix_handler14matrix_handl | cudaq::sample_result::deserialize |
| erENSt6stringERKNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     functio                       |
|     [\[4\]](api/                  | n)](api/languages/cpp_api.html#_C |
| languages/cpp_api.html#_CPPv4N5cu | PPv4N5cudaq13sample_result11deser |
| daq14matrix_handler14matrix_handl | ializeERNSt6vectorINSt6size_tEEE) |
| erENSt6stringERRNSt6vectorINSt6si | -   [cudaq::sample_result::dump   |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\                            |     function)](api/languag        |
| [5\]](api/languages/cpp_api.html# | es/cpp_api.html#_CPPv4NK5cudaq13s |
| _CPPv4N5cudaq14matrix_handler14ma | ample_result4dumpERNSt7ostreamE), |
| trix_handlerERK14matrix_handler), |     [\[1\]                        |
|     [                             | ](api/languages/cpp_api.html#_CPP |
| \[6\]](api/languages/cpp_api.html | v4NK5cudaq13sample_result4dumpEv) |
| #_CPPv4N5cudaq14matrix_handler14m | -   [cudaq::sample_result::end    |
| atrix_handlerERR14matrix_handler) |     (C++                          |
| -                                 |     function                      |
|  [cudaq::matrix_handler::momentum | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq13sample_result3endEv), |
|     function)](api/language       |     [\[1\                         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ]](api/languages/cpp_api.html#_CP |
| rix_handler8momentumENSt6size_tE) | Pv4NK5cudaq13sample_result3endEv) |
| -                                 | -   [                             |
|    [cudaq::matrix_handler::number | cudaq::sample_result::expectation |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     f                             |
| ges/cpp_api.html#_CPPv4N5cudaq14m | unction)](api/languages/cpp_api.h |
| atrix_handler6numberENSt6size_tE) | tml#_CPPv4NK5cudaq13sample_result |
| -                                 | 11expectationEKNSt11string_viewE) |
| [cudaq::matrix_handler::operator= | -   [c                            |
|     (C++                          | udaq::sample_result::get_marginal |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     function)](api/languages/cpp_ |
| l#_CPPv4I0_NSt11enable_if_tIXaant | api.html#_CPPv4NK5cudaq13sample_r |
| NSt7is_sameI1T14matrix_handlerE5v | esult12get_marginalERKNSt6vectorI |
| alueENSt12is_base_of_vI16operator | NSt6size_tEEEKNSt11string_viewE), |
| _handler1TEEEbEEEN5cudaq14matrix_ |     [\[1\]](api/languages/cpp_    |
| handleraSER14matrix_handlerRK1T), | api.html#_CPPv4NK5cudaq13sample_r |
|     [\[1\]](api/languages         | esult12get_marginalERRKNSt6vector |
| /cpp_api.html#_CPPv4N5cudaq14matr | INSt6size_tEEEKNSt11string_viewE) |
| ix_handleraSERK14matrix_handler), | -   [cuda                         |
|     [\[2\]](api/language          | q::sample_result::get_total_shots |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handleraSERR14matrix_handler) |     function)](api/langua         |
| -   [                             | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| cudaq::matrix_handler::operator== | sample_result15get_total_shotsEv) |
|     (C++                          | -   [cuda                         |
|     function)](api/languages      | q::sample_result::has_even_parity |
| /cpp_api.html#_CPPv4NK5cudaq14mat |     (C++                          |
| rix_handlereqERK14matrix_handler) |     fun                           |
| -                                 | ction)](api/languages/cpp_api.htm |
|    [cudaq::matrix_handler::parity | l#_CPPv4N5cudaq13sample_result15h |
|     (C++                          | as_even_parityENSt11string_viewE) |
|     function)](api/langua         | -   [cuda                         |
| ges/cpp_api.html#_CPPv4N5cudaq14m | q::sample_result::has_expectation |
| atrix_handler6parityENSt6size_tE) |     (C++                          |
| -                                 |     funct                         |
|  [cudaq::matrix_handler::position | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NK5cudaq13sample_result15ha |
|     function)](api/language       | s_expectationEKNSt11string_viewE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cu                           |
| rix_handler8positionENSt6size_tE) | daq::sample_result::most_probable |
| -   [cudaq::                      |     (C++                          |
| matrix_handler::remove_definition |     fun                           |
|     (C++                          | ction)](api/languages/cpp_api.htm |
|     fu                            | l#_CPPv4NK5cudaq13sample_result13 |
| nction)](api/languages/cpp_api.ht | most_probableEKNSt11string_viewE) |
| ml#_CPPv4N5cudaq14matrix_handler1 | -                                 |
| 7remove_definitionERKNSt6stringE) | [cudaq::sample_result::operator+= |
| -                                 |     (C++                          |
|   [cudaq::matrix_handler::squeeze |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function)](api/languag        | ample_resultpLERK13sample_result) |
| es/cpp_api.html#_CPPv4N5cudaq14ma | -                                 |
| trix_handler7squeezeENSt6size_tE) |  [cudaq::sample_result::operator= |
| -   [cudaq::m                     |     (C++                          |
| atrix_handler::to_diagonal_matrix |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function)](api/lang           | ample_resultaSERR13sample_result) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -                                 |
| 14matrix_handler18to_diagonal_mat | [cudaq::sample_result::operator== |
| rixERNSt13unordered_mapINSt6size_ |     (C++                          |
| tENSt7int64_tEEERKNSt13unordered_ |     function)](api/languag        |
| mapINSt6stringENSt7complexIdEEEE) | es/cpp_api.html#_CPPv4NK5cudaq13s |
| -                                 | ample_resulteqERK13sample_result) |
| [cudaq::matrix_handler::to_matrix | -   [                             |
|     (C++                          | cudaq::sample_result::probability |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/lan            |
| v4NK5cudaq14matrix_handler9to_mat | guages/cpp_api.html#_CPPv4NK5cuda |
| rixERNSt13unordered_mapINSt6size_ | q13sample_result11probabilityENSt |
| tENSt7int64_tEEERKNSt13unordered_ | 11string_viewEKNSt11string_viewE) |
| mapINSt6stringENSt7complexIdEEEE) | -   [cud                          |
| -                                 | aq::sample_result::register_names |
| [cudaq::matrix_handler::to_string |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](api/               | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| languages/cpp_api.html#_CPPv4NK5c | 3sample_result14register_namesEv) |
| udaq14matrix_handler9to_stringEb) | -                                 |
| -                                 |    [cudaq::sample_result::reorder |
| [cudaq::matrix_handler::unique_id |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/               | ges/cpp_api.html#_CPPv4N5cudaq13s |
| languages/cpp_api.html#_CPPv4NK5c | ample_result7reorderERKNSt6vector |
| udaq14matrix_handler9unique_idEv) | INSt6size_tEEEKNSt11string_viewE) |
| -   [cudaq:                       | -   [cu                           |
| :matrix_handler::\~matrix_handler | daq::sample_result::sample_result |
|     (C++                          |     (C++                          |
|     functi                        |     func                          |
| on)](api/languages/cpp_api.html#_ | tion)](api/languages/cpp_api.html |
| CPPv4N5cudaq14matrix_handlerD0Ev) | #_CPPv4N5cudaq13sample_result13sa |
| -   [cudaq::matrix_op (C++        | mple_resultERK15ExecutionResult), |
|     type)](api/languages/cpp_a    |     [\[1\]](api/la                |
| pi.html#_CPPv4N5cudaq9matrix_opE) | nguages/cpp_api.html#_CPPv4N5cuda |
| -   [cudaq::matrix_op_term (C++   | q13sample_result13sample_resultER |
|                                   | KNSt6vectorI15ExecutionResultEE), |
|  type)](api/languages/cpp_api.htm |                                   |
| l#_CPPv4N5cudaq14matrix_op_termE) |  [\[2\]](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq13sample_result13 |
|    [cudaq::mdiag_operator_handler | sample_resultERR13sample_result), |
|     (C++                          |     [                             |
|     class)](                      | \[3\]](api/languages/cpp_api.html |
| api/languages/cpp_api.html#_CPPv4 | #_CPPv4N5cudaq13sample_result13sa |
| N5cudaq22mdiag_operator_handlerE) | mple_resultERR15ExecutionResult), |
| -   [cudaq::measure_handle (C++   |     [\[4\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
| class)](api/languages/cpp_api.htm | 13sample_result13sample_resultEdR |
| l#_CPPv4N5cudaq14measure_handleE) | KNSt6vectorI15ExecutionResultEE), |
| -   [cudaq::measure_result (C++   |     [\[5\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|  type)](api/languages/cpp_api.htm | 13sample_result13sample_resultEv) |
| l#_CPPv4N5cudaq14measure_resultE) | -                                 |
| -   [cudaq::mpi (C++              |  [cudaq::sample_result::serialize |
|     type)](api/languages          |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) |     function)](api                |
| -   [cudaq::mpi::all_gather (C++  | /languages/cpp_api.html#_CPPv4NK5 |
|     fu                            | cudaq13sample_result9serializeEv) |
| nction)](api/languages/cpp_api.ht | -   [cudaq::sample_result::size   |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     (C++                          |
| RNSt6vectorIdEERKNSt6vectorIdEE), |     function)](api/languages/c    |
|                                   | pp_api.html#_CPPv4NK5cudaq13sampl |
|   [\[1\]](api/languages/cpp_api.h | e_result4sizeEKNSt11string_viewE) |
| tml#_CPPv4N5cudaq3mpi10all_gather | -   [cudaq::sample_result::to_map |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     (C++                          |
| -   [cudaq::mpi::all_reduce (C++  |     function)](api/languages/cpp  |
|                                   | _api.html#_CPPv4NK5cudaq13sample_ |
|  function)](api/languages/cpp_api | result6to_mapEKNSt11string_viewE) |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | -   [cuda                         |
| reduceE1TRK1TRK14BinaryFunction), | q::sample_result::\~sample_result |
|     [\[1\]](api/langu             |     (C++                          |
| ages/cpp_api.html#_CPPv4I00EN5cud |     funct                         |
| aq3mpi10all_reduceE1TRK1TRK4Func) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::mpi::broadcast (C++   | _CPPv4N5cudaq13sample_resultD0Ev) |
|     function)](api/               | -   [cudaq::scalar_callback (C++  |
| languages/cpp_api.html#_CPPv4N5cu |     c                             |
| daq3mpi9broadcastERNSt6stringEi), | lass)](api/languages/cpp_api.html |
|     [\[1\]](api/la                | #_CPPv4N5cudaq15scalar_callbackE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [c                            |
| q3mpi9broadcastERNSt6vectorIdEEi) | udaq::scalar_callback::operator() |
| -   [cudaq::mpi::finalize (C++    |     (C++                          |
|     f                             |     function)](api/language       |
| unction)](api/languages/cpp_api.h | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | alar_callbackclERKNSt13unordered_ |
| -   [cudaq::mpi::initialize (C++  | mapINSt6stringENSt7complexIdEEEE) |
|     function                      | -   [                             |
| )](api/languages/cpp_api.html#_CP | cudaq::scalar_callback::operator= |
| Pv4N5cudaq3mpi10initializeEiPPc), |     (C++                          |
|     [                             |     function)](api/languages/c    |
| \[1\]](api/languages/cpp_api.html | pp_api.html#_CPPv4N5cudaq15scalar |
| #_CPPv4N5cudaq3mpi10initializeEv) | _callbackaSERK15scalar_callback), |
| -   [cudaq::mpi::is_initialized   |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|     function                      | r_callbackaSERR15scalar_callback) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq:                       |
| Pv4N5cudaq3mpi14is_initializedEv) | :scalar_callback::scalar_callback |
| -   [cudaq::mpi::num_ranks (C++   |     (C++                          |
|     fu                            |     function)](api/languag        |
| nction)](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4I0_NSt11ena |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | ble_if_tINSt16is_invocable_r_vINS |
| -   [cudaq::mpi::rank (C++        | t7complexIdEE8CallableRKNSt13unor |
|                                   | dered_mapINSt6stringENSt7complexI |
|    function)](api/languages/cpp_a | dEEEEEEbEEEN5cudaq15scalar_callba |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | ck15scalar_callbackERR8Callable), |
| -   [cudaq::noise_model (C++      |     [\[1\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
|    class)](api/languages/cpp_api. | Pv4N5cudaq15scalar_callback15scal |
| html#_CPPv4N5cudaq11noise_modelE) | ar_callbackERK15scalar_callback), |
| -   [cudaq::n                     |     [\[2                          |
| oise_model::add_all_qubit_channel | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_callback15sca |
|     function)](api                | lar_callbackERR15scalar_callback) |
| /languages/cpp_api.html#_CPPv4IDp | -   [cudaq::scalar_operator (C++  |
| EN5cudaq11noise_model21add_all_qu |     c                             |
| bit_channelEvRK13kraus_channeli), | lass)](api/languages/cpp_api.html |
|     [\[1\]](api/langua            | #_CPPv4N5cudaq15scalar_operatorE) |
| ges/cpp_api.html#_CPPv4N5cudaq11n | -                                 |
| oise_model21add_all_qubit_channel | [cudaq::scalar_operator::evaluate |
| ERKNSt6stringERK13kraus_channeli) |     (C++                          |
| -                                 |                                   |
|  [cudaq::noise_model::add_channel |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq15scalar_op |
|     funct                         | erator8evaluateERKNSt13unordered_ |
| ion)](api/languages/cpp_api.html# | mapINSt6stringENSt7complexIdEEEE) |
| _CPPv4IDpEN5cudaq11noise_model11a | -   [cudaq::scalar_ope            |
| dd_channelEvRK15PredicateFuncTy), | rator::get_parameter_descriptions |
|     [\[1\]](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4IDpEN5cudaq11noise |     f                             |
| _model11add_channelEvRKNSt6vector | unction)](api/languages/cpp_api.h |
| INSt6size_tEEERK13kraus_channel), | tml#_CPPv4NK5cudaq15scalar_operat |
|     [\[2\]](ap                    | or26get_parameter_descriptionsEv) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cu                           |
| cudaq11noise_model11add_channelER | daq::scalar_operator::is_constant |
| KNSt6stringERK15PredicateFuncTy), |     (C++                          |
|                                   |     function)](api/lang           |
| [\[3\]](api/languages/cpp_api.htm | uages/cpp_api.html#_CPPv4NK5cudaq |
| l#_CPPv4N5cudaq11noise_model11add | 15scalar_operator11is_constantEv) |
| _channelERKNSt6stringERKNSt6vecto | -   [c                            |
| rINSt6size_tEEERK13kraus_channel) | udaq::scalar_operator::operator\* |
| -   [cudaq::noise_model::empty    |     (C++                          |
|     (C++                          |     function                      |
|     function                      | )](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4N5cudaq15scalar_operatormlENSt |
| Pv4NK5cudaq11noise_model5emptyEv) | 7complexIdEERK15scalar_operator), |
| -                                 |     [\[1\                         |
| [cudaq::noise_model::get_channels | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormlENSt |
|     function)](api/l              | 7complexIdEERR15scalar_operator), |
| anguages/cpp_api.html#_CPPv4I0ENK |     [\[2\]](api/languages/cp      |
| 5cudaq11noise_model12get_channels | p_api.html#_CPPv4N5cudaq15scalar_ |
| ENSt6vectorI13kraus_channelEERKNS | operatormlEdRK15scalar_operator), |
| t6vectorINSt6size_tEEERKNSt6vecto |     [\[3\]](api/languages/cp      |
| rINSt6size_tEEERKNSt6vectorIdEE), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](api/languages/cpp_a   | operatormlEdRR15scalar_operator), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[4\]](api/languages         |
| el12get_channelsERKNSt6stringERKN | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| St6vectorINSt6size_tEEERKNSt6vect | alar_operatormlENSt7complexIdEE), |
| orINSt6size_tEEERKNSt6vectorIdEE) |     [\[5\]](api/languages/cpp     |
| -                                 | _api.html#_CPPv4NKR5cudaq15scalar |
|  [cudaq::noise_model::noise_model | _operatormlERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function)](api                | (api/languages/cpp_api.html#_CPPv |
| /languages/cpp_api.html#_CPPv4N5c | 4NKR5cudaq15scalar_operatormlEd), |
| udaq11noise_model11noise_modelEv) |     [\[7\]](api/language          |
| -   [cu                           | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| daq::noise_model::PredicateFuncTy | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[8\]](api/languages/cp      |
|     type)](api/la                 | p_api.html#_CPPv4NO5cudaq15scalar |
| nguages/cpp_api.html#_CPPv4N5cuda | _operatormlERK15scalar_operator), |
| q11noise_model15PredicateFuncTyE) |     [\[9\                         |
| -   [cud                          | ]](api/languages/cpp_api.html#_CP |
| aq::noise_model::register_channel | Pv4NO5cudaq15scalar_operatormlEd) |
|     (C++                          | -   [cu                           |
|     function)](api/languages      | daq::scalar_operator::operator\*= |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     (C++                          |
| noise_model16register_channelEvv) |     function)](api/languag        |
| -   [cudaq::                      | es/cpp_api.html#_CPPv4N5cudaq15sc |
| noise_model::requires_constructor | alar_operatormLENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     type)](api/languages/cp       | pp_api.html#_CPPv4N5cudaq15scalar |
| p_api.html#_CPPv4I0DpEN5cudaq11no | _operatormLERK15scalar_operator), |
| ise_model20requires_constructorE) |     [\[2                          |
| -   [cudaq::noise_model_type (C++ | \]](api/languages/cpp_api.html#_C |
|     e                             | PPv4N5cudaq15scalar_operatormLEd) |
| num)](api/languages/cpp_api.html# | -   [                             |
| _CPPv4N5cudaq16noise_model_typeE) | cudaq::scalar_operator::operator+ |
| -   [cudaq::no                    |     (C++                          |
| ise_model_type::amplitude_damping |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     enumerator)](api/languages    | Pv4N5cudaq15scalar_operatorplENSt |
| /cpp_api.html#_CPPv4N5cudaq16nois | 7complexIdEERK15scalar_operator), |
| e_model_type17amplitude_dampingE) |     [\[1\                         |
| -   [cudaq::noise_mode            | ]](api/languages/cpp_api.html#_CP |
| l_type::amplitude_damping_channel | Pv4N5cudaq15scalar_operatorplENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     e                             |     [\[2\]](api/languages/cp      |
| numerator)](api/languages/cpp_api | p_api.html#_CPPv4N5cudaq15scalar_ |
| .html#_CPPv4N5cudaq16noise_model_ | operatorplEdRK15scalar_operator), |
| type25amplitude_damping_channelE) |     [\[3\]](api/languages/cp      |
| -   [cudaq::n                     | p_api.html#_CPPv4N5cudaq15scalar_ |
| oise_model_type::bit_flip_channel | operatorplEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     enumerator)](api/language     | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| s/cpp_api.html#_CPPv4N5cudaq16noi | alar_operatorplENSt7complexIdEE), |
| se_model_type16bit_flip_channelE) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::                      | _api.html#_CPPv4NKR5cudaq15scalar |
| noise_model_type::depolarization1 | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     enumerator)](api/languag      | (api/languages/cpp_api.html#_CPPv |
| es/cpp_api.html#_CPPv4N5cudaq16no | 4NKR5cudaq15scalar_operatorplEd), |
| ise_model_type15depolarization1E) |     [\[7\]]                       |
| -   [cudaq::                      | (api/languages/cpp_api.html#_CPPv |
| noise_model_type::depolarization2 | 4NKR5cudaq15scalar_operatorplEv), |
|     (C++                          |     [\[8\]](api/language          |
|     enumerator)](api/languag      | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| es/cpp_api.html#_CPPv4N5cudaq16no | alar_operatorplENSt7complexIdEE), |
| ise_model_type15depolarization2E) |     [\[9\]](api/languages/cp      |
| -   [cudaq::noise_m               | p_api.html#_CPPv4NO5cudaq15scalar |
| odel_type::depolarization_channel | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[10\]                       |
|                                   | ](api/languages/cpp_api.html#_CPP |
|   enumerator)](api/languages/cpp_ | v4NO5cudaq15scalar_operatorplEd), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[11\                        |
| el_type22depolarization_channelE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4NO5cudaq15scalar_operatorplEv) |
|  [cudaq::noise_model_type::pauli1 | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator+= |
|     enumerator)](a                |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/languag        |
| 5cudaq16noise_model_type6pauli1E) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -                                 | alar_operatorpLENSt7complexIdEE), |
|  [cudaq::noise_model_type::pauli2 |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](a                | _operatorpLERK15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[2                          |
| 5cudaq16noise_model_type6pauli2E) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq                        | PPv4N5cudaq15scalar_operatorpLEd) |
| ::noise_model_type::phase_damping | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator- |
|     enumerator)](api/langu        |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     function                      |
| noise_model_type13phase_dampingE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::noi                   | Pv4N5cudaq15scalar_operatormiENSt |
| se_model_type::phase_flip_channel | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](api/languages/   | ]](api/languages/cpp_api.html#_CP |
| cpp_api.html#_CPPv4N5cudaq16noise | Pv4N5cudaq15scalar_operatormiENSt |
| _model_type18phase_flip_channelE) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
| [cudaq::noise_model_type::unknown | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRK15scalar_operator), |
|     enumerator)](ap               |     [\[3\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4N5cudaq15scalar_ |
| cudaq16noise_model_type7unknownE) | operatormiEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
| [cudaq::noise_model_type::x_error | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     enumerator)](ap               |     [\[5\]](api/languages/cpp     |
| i/languages/cpp_api.html#_CPPv4N5 | _api.html#_CPPv4NKR5cudaq15scalar |
| cudaq16noise_model_type7x_errorE) | _operatormiERK15scalar_operator), |
| -                                 |     [\[6\]]                       |
| [cudaq::noise_model_type::y_error | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEd), |
|     enumerator)](ap               |     [\[7\]]                       |
| i/languages/cpp_api.html#_CPPv4N5 | (api/languages/cpp_api.html#_CPPv |
| cudaq16noise_model_type7y_errorE) | 4NKR5cudaq15scalar_operatormiEv), |
| -                                 |     [\[8\]](api/language          |
| [cudaq::noise_model_type::z_error | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     enumerator)](ap               |     [\[9\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4NO5cudaq15scalar |
| cudaq16noise_model_type7z_errorE) | _operatormiERK15scalar_operator), |
| -   [cudaq::num_available_gpus    |     [\[10\]                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function                      | v4NO5cudaq15scalar_operatormiEd), |
| )](api/languages/cpp_api.html#_CP |     [\[11\                        |
| Pv4N5cudaq18num_available_gpusEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::observe (C++          | Pv4NO5cudaq15scalar_operatormiEv) |
|     function)]                    | -   [c                            |
| (api/languages/cpp_api.html#_CPPv | udaq::scalar_operator::operator-= |
| 4I00DpEN5cudaq7observeENSt6vector |     (C++                          |
| I14observe_resultEERR13QuantumKer |     function)](api/languag        |
| nelRK15SpinOpContainerDpRR4Args), | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     [\[1\]](api/languages/cpp_ap  | alar_operatormIENSt7complexIdEE), |
| i.html#_CPPv4I0DpEN5cudaq7observe |     [\[1\]](api/languages/c       |
| E14observe_resultNSt6size_tERR13Q | pp_api.html#_CPPv4N5cudaq15scalar |
| uantumKernelRK7spin_opDpRR4Args), | _operatormIERK15scalar_operator), |
|     [\[                           |     [\[2                          |
| 2\]](api/languages/cpp_api.html#_ | \]](api/languages/cpp_api.html#_C |
| CPPv4I0DpEN5cudaq7observeE14obser | PPv4N5cudaq15scalar_operatormIEd) |
| ve_resultRK15observe_optionsRR13Q | -   [                             |
| uantumKernelRK7spin_opDpRR4Args), | cudaq::scalar_operator::operator/ |
|     [\[3\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     function                      |
| udaq7observeE14observe_resultRR13 | )](api/languages/cpp_api.html#_CP |
| QuantumKernelRK7spin_opDpRR4Args) | Pv4N5cudaq15scalar_operatordvENSt |
| -   [cudaq::observe_options (C++  | 7complexIdEERK15scalar_operator), |
|     st                            |     [\[1\                         |
| ruct)](api/languages/cpp_api.html | ]](api/languages/cpp_api.html#_CP |
| #_CPPv4N5cudaq15observe_optionsE) | Pv4N5cudaq15scalar_operatordvENSt |
| -   [cudaq::observe_result (C++   | 7complexIdEERR15scalar_operator), |
|                                   |     [\[2\]](api/languages/cp      |
| class)](api/languages/cpp_api.htm | p_api.html#_CPPv4N5cudaq15scalar_ |
| l#_CPPv4N5cudaq14observe_resultE) | operatordvEdRK15scalar_operator), |
| -                                 |     [\[3\]](api/languages/cp      |
|    [cudaq::observe_result::counts | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRR15scalar_operator), |
|     function)](api/languages/c    |     [\[4\]](api/languages         |
| pp_api.html#_CPPv4N5cudaq14observ | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| e_result6countsERK12spin_op_term) | alar_operatordvENSt7complexIdEE), |
| -   [cudaq::observe_result::dump  |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|     function)                     | _operatordvERK15scalar_operator), |
| ](api/languages/cpp_api.html#_CPP |     [\[6\]]                       |
| v4N5cudaq14observe_result4dumpEv) | (api/languages/cpp_api.html#_CPPv |
| -   [c                            | 4NKR5cudaq15scalar_operatordvEd), |
| udaq::observe_result::expectation |     [\[7\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|                                   | alar_operatordvENSt7complexIdEE), |
| function)](api/languages/cpp_api. |     [\[8\]](api/languages/cp      |
| html#_CPPv4N5cudaq14observe_resul | p_api.html#_CPPv4NO5cudaq15scalar |
| t11expectationERK12spin_op_term), | _operatordvERK15scalar_operator), |
|     [\[1\]](api/la                |     [\[9\                         |
| nguages/cpp_api.html#_CPPv4N5cuda | ]](api/languages/cpp_api.html#_CP |
| q14observe_result11expectationEv) | Pv4NO5cudaq15scalar_operatordvEd) |
| -   [cuda                         | -   [c                            |
| q::observe_result::id_coefficient | udaq::scalar_operator::operator/= |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     function)](api/languag        |
| ages/cpp_api.html#_CPPv4N5cudaq14 | es/cpp_api.html#_CPPv4N5cudaq15sc |
| observe_result14id_coefficientEv) | alar_operatordVENSt7complexIdEE), |
| -   [cuda                         |     [\[1\]](api/languages/c       |
| q::observe_result::observe_result | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatordVERK15scalar_operator), |
|                                   |     [\[2                          |
|   function)](api/languages/cpp_ap | \]](api/languages/cpp_api.html#_C |
| i.html#_CPPv4N5cudaq14observe_res | PPv4N5cudaq15scalar_operatordVEd) |
| ult14observe_resultEdRK7spin_op), | -   [                             |
|     [\[1\]](a                     | cudaq::scalar_operator::operator= |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14observe_result14observe_r |     function)](api/languages/c    |
| esultEdRK7spin_op13sample_result) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _operatoraSERK15scalar_operator), |
|  [cudaq::observe_result::operator |     [\[1\]](api/languages/        |
|     double (C++                   | cpp_api.html#_CPPv4N5cudaq15scala |
|     functio                       | r_operatoraSERR15scalar_operator) |
| n)](api/languages/cpp_api.html#_C | -   [c                            |
| PPv4N5cudaq14observe_resultcvdEv) | udaq::scalar_operator::operator== |
| -                                 |     (C++                          |
|  [cudaq::observe_result::raw_data |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq15scala |
|     function)](ap                 | r_operatoreqERK15scalar_operator) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq:                       |
| cudaq14observe_result8raw_dataEv) | :scalar_operator::scalar_operator |
| -   [cudaq::operator_handler (C++ |     (C++                          |
|     cl                            |     func                          |
| ass)](api/languages/cpp_api.html# | tion)](api/languages/cpp_api.html |
| _CPPv4N5cudaq16operator_handlerE) | #_CPPv4N5cudaq15scalar_operator15 |
| -   [cudaq::optimizable_function  | scalar_operatorENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/langu             |
|     class)                        | ages/cpp_api.html#_CPPv4N5cudaq15 |
| ](api/languages/cpp_api.html#_CPP | scalar_operator15scalar_operatorE |
| v4N5cudaq20optimizable_functionE) | RK15scalar_callbackRRNSt13unorder |
| -   [cudaq::optimization_result   | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[2\                         |
|     type                          | ]](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4N5cudaq15scalar_operator15scal |
| Pv4N5cudaq19optimization_resultE) | ar_operatorERK15scalar_operator), |
| -   [cudaq::optimizer (C++        |     [\[3\]](api/langu             |
|     class)](api/languages/cpp_a   | ages/cpp_api.html#_CPPv4N5cudaq15 |
| pi.html#_CPPv4N5cudaq9optimizerE) | scalar_operator15scalar_operatorE |
| -   [cudaq::optimizer::optimize   | RR15scalar_callbackRRNSt13unorder |
|     (C++                          | ed_mapINSt6stringENSt6stringEEE), |
|                                   |     [\[4\                         |
|  function)](api/languages/cpp_api | ]](api/languages/cpp_api.html#_CP |
| .html#_CPPv4N5cudaq9optimizer8opt | Pv4N5cudaq15scalar_operator15scal |
| imizeEKiRR20optimizable_function) | ar_operatorERR15scalar_operator), |
| -   [cu                           |     [\[5\]](api/language          |
| daq::optimizer::requiresGradients | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     (C++                          | lar_operator15scalar_operatorEd), |
|     function)](api/la             |     [\[6\]](api/languag           |
| nguages/cpp_api.html#_CPPv4N5cuda | es/cpp_api.html#_CPPv4N5cudaq15sc |
| q9optimizer17requiresGradientsEv) | alar_operator15scalar_operatorEv) |
| -   [cudaq::orca (C++             | -   [                             |
|     type)](api/languages/         | cudaq::scalar_operator::to_matrix |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     (C++                          |
| -   [cudaq::orca::sample (C++     |                                   |
|     function)](api/languages/c    |   function)](api/languages/cpp_ap |
| pp_api.html#_CPPv4N5cudaq4orca6sa | i.html#_CPPv4NK5cudaq15scalar_ope |
| mpleERNSt6vectorINSt6size_tEEERNS | rator9to_matrixERKNSt13unordered_ |
| t6vectorINSt6size_tEEERNSt6vector | mapINSt6stringENSt7complexIdEEEE) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [                             |
|     [\[1\]]                       | cudaq::scalar_operator::to_string |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq4orca6sampleERNSt6vectorI |     function)](api/l              |
| NSt6size_tEEERNSt6vectorINSt6size | anguages/cpp_api.html#_CPPv4NK5cu |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | daq15scalar_operator9to_stringEv) |
| -   [cudaq::orca::sample_async    | -   [cudaq::s                     |
|     (C++                          | calar_operator::\~scalar_operator |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     functio                       |
| html#_CPPv4N5cudaq4orca12sample_a | n)](api/languages/cpp_api.html#_C |
| syncERNSt6vectorINSt6size_tEEERNS | PPv4N5cudaq15scalar_operatorD0Ev) |
| t6vectorINSt6size_tEEERNSt6vector | -   [cudaq::set_noise (C++        |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     function)](api/langu          |
|     [\[1\]](api/la                | ages/cpp_api.html#_CPPv4N5cudaq9s |
| nguages/cpp_api.html#_CPPv4N5cuda | et_noiseERKN5cudaq11noise_modelE) |
| q4orca12sample_asyncERNSt6vectorI | -   [cudaq::set_random_seed (C++  |
| NSt6size_tEEERNSt6vectorINSt6size |     function)](api/               |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::OrcaRemoteRESTQPU     | daq15set_random_seedENSt6size_tE) |
|     (C++                          | -   [cudaq::simulation_precision  |
|     cla                           |     (C++                          |
| ss)](api/languages/cpp_api.html#_ |     enum)                         |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::pauli1 (C++           | v4N5cudaq20simulation_precisionE) |
|     class)](api/languages/cp      | -   [                             |
| p_api.html#_CPPv4N5cudaq6pauli1E) | cudaq::simulation_precision::fp32 |
| -                                 |     (C++                          |
|    [cudaq::pauli1::num_parameters |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)]                      | udaq20simulation_precision4fp32E) |
| (api/languages/cpp_api.html#_CPPv | -   [                             |
| 4N5cudaq6pauli114num_parametersE) | cudaq::simulation_precision::fp64 |
| -   [cudaq::pauli1::num_targets   |     (C++                          |
|     (C++                          |     enumerator)](api              |
|     membe                         | /languages/cpp_api.html#_CPPv4N5c |
| r)](api/languages/cpp_api.html#_C | udaq20simulation_precision4fp64E) |
| PPv4N5cudaq6pauli111num_targetsE) | -   [cudaq::SimulationState (C++  |
| -   [cudaq::pauli1::pauli1 (C++   |     c                             |
|     function)](api/languages/cpp_ | lass)](api/languages/cpp_api.html |
| api.html#_CPPv4N5cudaq6pauli16pau | #_CPPv4N5cudaq15SimulationStateE) |
| li1ERKNSt6vectorIN5cudaq4realEEE) | -   [                             |
| -   [cudaq::pauli2 (C++           | cudaq::SimulationState::precision |
|     class)](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq6pauli2E) |     enum)](api                    |
| -                                 | /languages/cpp_api.html#_CPPv4N5c |
|    [cudaq::pauli2::num_parameters | udaq15SimulationState9precisionE) |
|     (C++                          | -   [cudaq:                       |
|     member)]                      | :SimulationState::precision::fp32 |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq6pauli214num_parametersE) |     enumerator)](api/lang         |
| -   [cudaq::pauli2::num_targets   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     (C++                          | 5SimulationState9precision4fp32E) |
|     membe                         | -   [cudaq:                       |
| r)](api/languages/cpp_api.html#_C | :SimulationState::precision::fp64 |
| PPv4N5cudaq6pauli211num_targetsE) |     (C++                          |
| -   [cudaq::pauli2::pauli2 (C++   |     enumerator)](api/lang         |
|     function)](api/languages/cpp_ | uages/cpp_api.html#_CPPv4N5cudaq1 |
| api.html#_CPPv4N5cudaq6pauli26pau | 5SimulationState9precision4fp64E) |
| li2ERKNSt6vectorIN5cudaq4realEEE) | -                                 |
| -   [cudaq::phase_damping (C++    |   [cudaq::SimulationState::Tensor |
|                                   |     (C++                          |
|  class)](api/languages/cpp_api.ht |     struct)](                     |
| ml#_CPPv4N5cudaq13phase_dampingE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cud                          | N5cudaq15SimulationState6TensorE) |
| aq::phase_damping::num_parameters | -   [cudaq::spin_handler (C++     |
|     (C++                          |                                   |
|     member)](api/lan              |   class)](api/languages/cpp_api.h |
| guages/cpp_api.html#_CPPv4N5cudaq | tml#_CPPv4N5cudaq12spin_handlerE) |
| 13phase_damping14num_parametersE) | -   [cudaq:                       |
| -   [                             | :spin_handler::to_diagonal_matrix |
| cudaq::phase_damping::num_targets |     (C++                          |
|     (C++                          |     function)](api/la             |
|     member)](api/                 | nguages/cpp_api.html#_CPPv4NK5cud |
| languages/cpp_api.html#_CPPv4N5cu | aq12spin_handler18to_diagonal_mat |
| daq13phase_damping11num_targetsE) | rixERNSt13unordered_mapINSt6size_ |
| -   [cudaq::phase_flip_channel    | tENSt7int64_tEEERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|     clas                          | -                                 |
| s)](api/languages/cpp_api.html#_C |   [cudaq::spin_handler::to_matrix |
| PPv4N5cudaq18phase_flip_channelE) |     (C++                          |
| -   [cudaq::p                     |     function                      |
| hase_flip_channel::num_parameters | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq12spin_handler9to_matri |
|     member)](api/language         | xERKNSt6stringENSt7complexIdEEb), |
| s/cpp_api.html#_CPPv4N5cudaq18pha |     [\[1                          |
| se_flip_channel14num_parametersE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq                        | PPv4NK5cudaq12spin_handler9to_mat |
| ::phase_flip_channel::num_targets | rixERNSt13unordered_mapINSt6size_ |
|     (C++                          | tENSt7int64_tEEERKNSt13unordered_ |
|     member)](api/langu            | mapINSt6stringENSt7complexIdEEEE) |
| ages/cpp_api.html#_CPPv4N5cudaq18 | -   [cuda                         |
| phase_flip_channel11num_targetsE) | q::spin_handler::to_sparse_matrix |
| -   [cudaq::product_op (C++       |     (C++                          |
|                                   |     function)](api/               |
|  class)](api/languages/cpp_api.ht | languages/cpp_api.html#_CPPv4N5cu |
| ml#_CPPv4I0EN5cudaq10product_opE) | daq12spin_handler16to_sparse_matr |
| -   [cudaq::product_op::begin     | ixERKNSt6stringENSt7complexIdEEb) |
|     (C++                          | -                                 |
|     functio                       |   [cudaq::spin_handler::to_string |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4NK5cudaq10product_op5beginEv) |     function)](ap                 |
| -                                 | i/languages/cpp_api.html#_CPPv4NK |
|  [cudaq::product_op::canonicalize | 5cudaq12spin_handler9to_stringEb) |
|     (C++                          | -                                 |
|     func                          |   [cudaq::spin_handler::unique_id |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq10product_op12canon |     function)](ap                 |
| icalizeERKNSt3setINSt6size_tEEE), | i/languages/cpp_api.html#_CPPv4NK |
|     [\[1\]](api                   | 5cudaq12spin_handler9unique_idEv) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::spin_op (C++          |
| udaq10product_op12canonicalizeEv) |     type)](api/languages/cpp      |
| -   [                             | _api.html#_CPPv4N5cudaq7spin_opE) |
| cudaq::product_op::const_iterator | -   [cudaq::spin_op_term (C++     |
|     (C++                          |                                   |
|     struct)](api/                 |    type)](api/languages/cpp_api.h |
| languages/cpp_api.html#_CPPv4N5cu | tml#_CPPv4N5cudaq12spin_op_termE) |
| daq10product_op14const_iteratorE) | -   [cudaq::state (C++            |
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
