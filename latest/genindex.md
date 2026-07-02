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
        -   [DEM
            Options](using/examples/dem_from_kernel.html#dem-options){.reference
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
| -   [cachedCompiledModule()       | -   [cudaq::product_op (C++       |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](api/langu            |  class)](api/languages/cpp_api.ht |
| ages/python_api.html#cudaq.PyKern | ml#_CPPv4I0EN5cudaq10product_opE) |
| elDecorator.cachedCompiledModule) | -   [cudaq::product_op::begin     |
| -   [canonicalize                 |     (C++                          |
|     (cu                           |     functio                       |
| daq.operators.boson.BosonOperator | n)](api/languages/cpp_api.html#_C |
|     attribute)](api/languages     | PPv4NK5cudaq10product_op5beginEv) |
| /python_api.html#cudaq.operators. | -                                 |
| boson.BosonOperator.canonicalize) |  [cudaq::product_op::canonicalize |
|     -   [(cudaq.                  |     (C++                          |
| operators.boson.BosonOperatorTerm |     func                          |
|                                   | tion)](api/languages/cpp_api.html |
|     attribute)](api/languages/pyt | #_CPPv4N5cudaq10product_op12canon |
| hon_api.html#cudaq.operators.boso | icalizeERKNSt3setINSt6size_tEEE), |
| n.BosonOperatorTerm.canonicalize) |     [\[1\]](api                   |
|     -   [(cudaq.                  | /languages/cpp_api.html#_CPPv4N5c |
| operators.fermion.FermionOperator | udaq10product_op12canonicalizeEv) |
|                                   | -   [                             |
|     attribute)](api/languages/pyt | cudaq::product_op::const_iterator |
| hon_api.html#cudaq.operators.ferm |     (C++                          |
| ion.FermionOperator.canonicalize) |     struct)](api/                 |
|     -   [(cudaq.oper              | languages/cpp_api.html#_CPPv4N5cu |
| ators.fermion.FermionOperatorTerm | daq10product_op14const_iteratorE) |
|                                   | -   [cudaq::product_o             |
| attribute)](api/languages/python_ | p::const_iterator::const_iterator |
| api.html#cudaq.operators.fermion. |     (C++                          |
| FermionOperatorTerm.canonicalize) |     fu                            |
|     -                             | nction)](api/languages/cpp_api.ht |
|  [(cudaq.operators.MatrixOperator | ml#_CPPv4N5cudaq10product_op14con |
|         attribute)](api/lang      | st_iterator14const_iteratorEPK10p |
| uages/python_api.html#cudaq.opera | roduct_opI9HandlerTyENSt6size_tE) |
| tors.MatrixOperator.canonicalize) | -   [cudaq::produ                 |
|     -   [(c                       | ct_op::const_iterator::operator!= |
| udaq.operators.MatrixOperatorTerm |     (C++                          |
|         attribute)](api/language  |     fun                           |
| s/python_api.html#cudaq.operators | ction)](api/languages/cpp_api.htm |
| .MatrixOperatorTerm.canonicalize) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(                        | st_iteratorneERK14const_iterator) |
| cudaq.operators.spin.SpinOperator | -   [cudaq::produ                 |
|         attribute)](api/languag   | ct_op::const_iterator::operator\* |
| es/python_api.html#cudaq.operator |     (C++                          |
| s.spin.SpinOperator.canonicalize) |     function)](api/lang           |
|     -   [(cuda                    | uages/cpp_api.html#_CPPv4NK5cudaq |
| q.operators.spin.SpinOperatorTerm | 10product_op14const_iteratormlEv) |
|                                   | -   [cudaq::produ                 |
|       attribute)](api/languages/p | ct_op::const_iterator::operator++ |
| ython_api.html#cudaq.operators.sp |     (C++                          |
| in.SpinOperatorTerm.canonicalize) |     function)](api/lang           |
| -   [captured_variables()         | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     (cudaq.PyKernelDecorator      | 0product_op14const_iteratorppEi), |
|     method)](api/lan              |     [\[1\]](api/lan               |
| guages/python_api.html#cudaq.PyKe | guages/cpp_api.html#_CPPv4N5cudaq |
| rnelDecorator.captured_variables) | 10product_op14const_iteratorppEv) |
| -   [CentralDifference (class in  | -   [cudaq::produc                |
|     cudaq.gradients)              | t_op::const_iterator::operator\-- |
| ](api/languages/python_api.html#c |     (C++                          |
| udaq.gradients.CentralDifference) |     function)](api/lang           |
| -   [channel                      | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     (cudaq.ptsbe.TraceInstruction | 0product_op14const_iteratormmEi), |
|     property)](a                  |     [\[1\]](api/lan               |
| pi/languages/python_api.html#cuda | guages/cpp_api.html#_CPPv4N5cudaq |
| q.ptsbe.TraceInstruction.channel) | 10product_op14const_iteratormmEv) |
| -   [circuit_location             | -   [cudaq::produc                |
|     (cudaq.ptsbe.KrausSelection   | t_op::const_iterator::operator-\> |
|     property)](api/lang           |     (C++                          |
| uages/python_api.html#cudaq.ptsbe |     function)](api/lan            |
| .KrausSelection.circuit_location) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [clear (cudaq.Resources       | 10product_op14const_iteratorptEv) |
|                                   | -   [cudaq::produ                 |
|   attribute)](api/languages/pytho | ct_op::const_iterator::operator== |
| n_api.html#cudaq.Resources.clear) |     (C++                          |
|     -   [(cudaq.SampleResult      |     fun                           |
|         a                         | ction)](api/languages/cpp_api.htm |
| ttribute)](api/languages/python_a | l#_CPPv4NK5cudaq10product_op14con |
| pi.html#cudaq.SampleResult.clear) | st_iteratoreqERK14const_iterator) |
| -   [COBYLA (class in             | -   [cudaq::product_op::degrees   |
|     cudaq.o                       |     (C++                          |
| ptimizers)](api/languages/python_ |     function)                     |
| api.html#cudaq.optimizers.COBYLA) | ](api/languages/cpp_api.html#_CPP |
| -   [coefficient                  | v4NK5cudaq10product_op7degreesEv) |
|     (cudaq.                       | -   [cudaq::product_op::dump (C++ |
| operators.boson.BosonOperatorTerm |     functi                        |
|     property)](api/languages/py   | on)](api/languages/cpp_api.html#_ |
| thon_api.html#cudaq.operators.bos | CPPv4NK5cudaq10product_op4dumpEv) |
| on.BosonOperatorTerm.coefficient) | -   [cudaq::product_op::end (C++  |
|     -   [(cudaq.oper              |     funct                         |
| ators.fermion.FermionOperatorTerm | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4NK5cudaq10product_op3endEv) |
|   property)](api/languages/python | -   [c                            |
| _api.html#cudaq.operators.fermion | udaq::product_op::get_coefficient |
| .FermionOperatorTerm.coefficient) |     (C++                          |
|     -   [(c                       |     function)](api/lan            |
| udaq.operators.MatrixOperatorTerm | guages/cpp_api.html#_CPPv4NK5cuda |
|         property)](api/languag    | q10product_op15get_coefficientEv) |
| es/python_api.html#cudaq.operator | -                                 |
| s.MatrixOperatorTerm.coefficient) |   [cudaq::product_op::get_term_id |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     function)](api                |
|         property)](api/languages/ | /languages/cpp_api.html#_CPPv4NK5 |
| python_api.html#cudaq.operators.s | cudaq10product_op11get_term_idEv) |
| pin.SpinOperatorTerm.coefficient) | -                                 |
| -   [col_count                    |   [cudaq::product_op::is_identity |
|     (cudaq.KrausOperator          |     (C++                          |
|     prope                         |     function)](api                |
| rty)](api/languages/python_api.ht | /languages/cpp_api.html#_CPPv4NK5 |
| ml#cudaq.KrausOperator.col_count) | cudaq10product_op11is_identityEv) |
| -   [compile()                    | -   [cudaq::product_op::num_ops   |
|     (cudaq.PyKernelDecorator      |     (C++                          |
|     metho                         |     function)                     |
| d)](api/languages/python_api.html | ](api/languages/cpp_api.html#_CPP |
| #cudaq.PyKernelDecorator.compile) | v4NK5cudaq10product_op7num_opsEv) |
| -   [ComplexMatrix (class in      | -                                 |
|     cudaq)](api/languages/pyt     |    [cudaq::product_op::operator\* |
| hon_api.html#cudaq.ComplexMatrix) |     (C++                          |
| -   [compute                      |     function)](api/languages/     |
|     (                             | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| cudaq.gradients.CentralDifference | oduct_opmlE10product_opI1TERK15sc |
|     attribute)](api/la            | alar_operatorRK10product_opI1TE), |
| nguages/python_api.html#cudaq.gra |     [\[1\]](api/languages/        |
| dients.CentralDifference.compute) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(                        | oduct_opmlE10product_opI1TERK15sc |
| cudaq.gradients.ForwardDifference | alar_operatorRR10product_opI1TE), |
|         attribute)](api/la        |     [\[2\]](api/languages/        |
| nguages/python_api.html#cudaq.gra | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| dients.ForwardDifference.compute) | oduct_opmlE10product_opI1TERR15sc |
|     -                             | alar_operatorRK10product_opI1TE), |
|  [(cudaq.gradients.ParameterShift |     [\[3\]](api/languages/        |
|         attribute)](api           | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| /languages/python_api.html#cudaq. | oduct_opmlE10product_opI1TERR15sc |
| gradients.ParameterShift.compute) | alar_operatorRR10product_opI1TE), |
| -   [const()                      |     [\[4\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|   (cudaq.operators.ScalarOperator | 5cudaq10product_opmlE6sum_opI1TER |
|     class                         | K15scalar_operatorRK6sum_opI1TE), |
|     method)](a                    |     [\[5\]](api/                  |
| pi/languages/python_api.html#cuda | languages/cpp_api.html#_CPPv4I0EN |
| q.operators.ScalarOperator.const) | 5cudaq10product_opmlE6sum_opI1TER |
| -   [controls                     | K15scalar_operatorRR6sum_opI1TE), |
|     (cudaq.ptsbe.TraceInstruction |     [\[6\]](api/                  |
|     property)](ap                 | languages/cpp_api.html#_CPPv4I0EN |
| i/languages/python_api.html#cudaq | 5cudaq10product_opmlE6sum_opI1TER |
| .ptsbe.TraceInstruction.controls) | R15scalar_operatorRK6sum_opI1TE), |
| -   [copy                         |     [\[7\]](api/                  |
|     (cu                           | languages/cpp_api.html#_CPPv4I0EN |
| daq.operators.boson.BosonOperator | 5cudaq10product_opmlE6sum_opI1TER |
|     attribute)](api/l             | R15scalar_operatorRR6sum_opI1TE), |
| anguages/python_api.html#cudaq.op |     [\[8\]](api/languages         |
| erators.boson.BosonOperator.copy) | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     -   [(cudaq.                  | duct_opmlERK6sum_opI9HandlerTyE), |
| operators.boson.BosonOperatorTerm |     [\[9\]](api/languages/cpp_a   |
|         attribute)](api/langu     | pi.html#_CPPv4NKR5cudaq10product_ |
| ages/python_api.html#cudaq.operat | opmlERK10product_opI9HandlerTyE), |
| ors.boson.BosonOperatorTerm.copy) |     [\[10\]](api/language         |
|     -   [(cudaq.                  | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| operators.fermion.FermionOperator | roduct_opmlERK15scalar_operator), |
|         attribute)](api/langu     |     [\[11\]](api/languages/cpp_a  |
| ages/python_api.html#cudaq.operat | pi.html#_CPPv4NKR5cudaq10product_ |
| ors.fermion.FermionOperator.copy) | opmlERR10product_opI9HandlerTyE), |
|     -   [(cudaq.oper              |     [\[12\]](api/language         |
| ators.fermion.FermionOperatorTerm | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|         attribute)](api/languages | roduct_opmlERR15scalar_operator), |
| /python_api.html#cudaq.operators. |     [\[13\]](api/languages/cpp_   |
| fermion.FermionOperatorTerm.copy) | api.html#_CPPv4NO5cudaq10product_ |
|     -                             | opmlERK10product_opI9HandlerTyE), |
|  [(cudaq.operators.MatrixOperator |     [\[14\]](api/languag          |
|         attribute)](              | es/cpp_api.html#_CPPv4NO5cudaq10p |
| api/languages/python_api.html#cud | roduct_opmlERK15scalar_operator), |
| aq.operators.MatrixOperator.copy) |     [\[15\]](api/languages/cpp_   |
|     -   [(c                       | api.html#_CPPv4NO5cudaq10product_ |
| udaq.operators.MatrixOperatorTerm | opmlERR10product_opI9HandlerTyE), |
|         attribute)](api/          |     [\[16\]](api/langua           |
| languages/python_api.html#cudaq.o | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| perators.MatrixOperatorTerm.copy) | product_opmlERR15scalar_operator) |
|     -   [(                        | -                                 |
| cudaq.operators.spin.SpinOperator |   [cudaq::product_op::operator\*= |
|         attribute)](api           |     (C++                          |
| /languages/python_api.html#cudaq. |     function)](api/languages/cpp  |
| operators.spin.SpinOperator.copy) | _api.html#_CPPv4N5cudaq10product_ |
|     -   [(cuda                    | opmLERK10product_opI9HandlerTyE), |
| q.operators.spin.SpinOperatorTerm |     [\[1\]](api/langua            |
|         attribute)](api/lan       | ges/cpp_api.html#_CPPv4N5cudaq10p |
| guages/python_api.html#cudaq.oper | roduct_opmLERK15scalar_operator), |
| ators.spin.SpinOperatorTerm.copy) |     [\[2\]](api/languages/cp      |
| -   [count (cudaq.Resources       | p_api.html#_CPPv4N5cudaq10product |
|                                   | _opmLERR10product_opI9HandlerTyE) |
|   attribute)](api/languages/pytho | -   [cudaq::product_op::operator+ |
| n_api.html#cudaq.Resources.count) |     (C++                          |
|     -   [(cudaq.SampleResult      |     function)](api/langu          |
|         a                         | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ttribute)](api/languages/python_a | q10product_opplE6sum_opI1TERK15sc |
| pi.html#cudaq.SampleResult.count) | alar_operatorRK10product_opI1TE), |
| -   [count_controls               |     [\[1\]](api/                  |
|     (cudaq.Resources              | languages/cpp_api.html#_CPPv4I0EN |
|     attribu                       | 5cudaq10product_opplE6sum_opI1TER |
| te)](api/languages/python_api.htm | K15scalar_operatorRK6sum_opI1TE), |
| l#cudaq.Resources.count_controls) |     [\[2\]](api/langu             |
| -   [count_instructions           | ages/cpp_api.html#_CPPv4I0EN5cuda |
|                                   | q10product_opplE6sum_opI1TERK15sc |
|   (cudaq.ptsbe.PTSBEExecutionData | alar_operatorRR10product_opI1TE), |
|     attribute)](api/languages/    |     [\[3\]](api/                  |
| python_api.html#cudaq.ptsbe.PTSBE | languages/cpp_api.html#_CPPv4I0EN |
| ExecutionData.count_instructions) | 5cudaq10product_opplE6sum_opI1TER |
| -   [counts (cudaq.ObserveResult  | K15scalar_operatorRR6sum_opI1TE), |
|     att                           |     [\[4\]](api/langu             |
| ribute)](api/languages/python_api | ages/cpp_api.html#_CPPv4I0EN5cuda |
| .html#cudaq.ObserveResult.counts) | q10product_opplE6sum_opI1TERR15sc |
| -   [csr_spmatrix (C++            | alar_operatorRK10product_opI1TE), |
|     type)](api/languages/c        |     [\[5\]](api/                  |
| pp_api.html#_CPPv412csr_spmatrix) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq                         | 5cudaq10product_opplE6sum_opI1TER |
|     -   [module](api/langua       | R15scalar_operatorRK6sum_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[6\]](api/langu             |
| -   [cudaq (C++                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     type)](api/lan                | q10product_opplE6sum_opI1TERR15sc |
| guages/cpp_api.html#_CPPv45cudaq) | alar_operatorRR10product_opI1TE), |
| -   [cudaq.apply_noise() (in      |     [\[7\]](api/                  |
|     module                        | languages/cpp_api.html#_CPPv4I0EN |
|     cudaq)](api/languages/python_ | 5cudaq10product_opplE6sum_opI1TER |
| api.html#cudaq.cudaq.apply_noise) | R15scalar_operatorRR6sum_opI1TE), |
| -   cudaq.boson                   |     [\[8\]](api/languages/cpp_a   |
|     -   [module](api/languages/py | pi.html#_CPPv4NKR5cudaq10product_ |
| thon_api.html#module-cudaq.boson) | opplERK10product_opI9HandlerTyE), |
| -   cudaq.fermion                 |     [\[9\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|   -   [module](api/languages/pyth | roduct_opplERK15scalar_operator), |
| on_api.html#module-cudaq.fermion) |     [\[10\]](api/languages/       |
| -   cudaq.operators.custom        | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     -   [mo                       | duct_opplERK6sum_opI9HandlerTyE), |
| dule](api/languages/python_api.ht |     [\[11\]](api/languages/cpp_a  |
| ml#module-cudaq.operators.custom) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.spin                    | opplERR10product_opI9HandlerTyE), |
|     -   [module](api/languages/p  |     [\[12\]](api/language         |
| ython_api.html#module-cudaq.spin) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::amplitude_damping     | roduct_opplERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     cla                           | cpp_api.html#_CPPv4NKR5cudaq10pro |
| ss)](api/languages/cpp_api.html#_ | duct_opplERR6sum_opI9HandlerTyE), |
| CPPv4N5cudaq17amplitude_dampingE) |     [\[                           |
| -                                 | 14\]](api/languages/cpp_api.html# |
| [cudaq::amplitude_damping_channel | _CPPv4NKR5cudaq10product_opplEv), |
|     (C++                          |     [\[15\]](api/languages/cpp_   |
|     class)](api                   | api.html#_CPPv4NO5cudaq10product_ |
| /languages/cpp_api.html#_CPPv4N5c | opplERK10product_opI9HandlerTyE), |
| udaq25amplitude_damping_channelE) |     [\[16\]](api/languag          |
| -   [cudaq::amplitud              | es/cpp_api.html#_CPPv4NO5cudaq10p |
| e_damping_channel::num_parameters | roduct_opplERK15scalar_operator), |
|     (C++                          |     [\[17\]](api/languages        |
|     member)](api/languages/cpp_a  | /cpp_api.html#_CPPv4NO5cudaq10pro |
| pi.html#_CPPv4N5cudaq25amplitude_ | duct_opplERK6sum_opI9HandlerTyE), |
| damping_channel14num_parametersE) |     [\[18\]](api/languages/cpp_   |
| -   [cudaq::ampli                 | api.html#_CPPv4NO5cudaq10product_ |
| tude_damping_channel::num_targets | opplERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[19\]](api/languag          |
|     member)](api/languages/cp     | es/cpp_api.html#_CPPv4NO5cudaq10p |
| p_api.html#_CPPv4N5cudaq25amplitu | roduct_opplERR15scalar_operator), |
| de_damping_channel11num_targetsE) |     [\[20\]](api/languages        |
| -   [cudaq::AnalogRemoteRESTQPU   | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opplERR6sum_opI9HandlerTyE), |
|     class                         |     [                             |
| )](api/languages/cpp_api.html#_CP | \[21\]](api/languages/cpp_api.htm |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | l#_CPPv4NO5cudaq10product_opplEv) |
| -   [cudaq::apply_noise (C++      | -   [cudaq::product_op::operator- |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4I0Dp |     function)](api/langu          |
| EN5cudaq11apply_noiseEvDpRR4Args) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::async_result (C++     | q10product_opmiE6sum_opI1TERK15sc |
|     c                             | alar_operatorRK10product_opI1TE), |
| lass)](api/languages/cpp_api.html |     [\[1\]](api/                  |
| #_CPPv4I0EN5cudaq12async_resultE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::async_result::get     | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRK6sum_opI1TE), |
|     functi                        |     [\[2\]](api/langu             |
| on)](api/languages/cpp_api.html#_ | ages/cpp_api.html#_CPPv4I0EN5cuda |
| CPPv4N5cudaq12async_result3getEv) | q10product_opmiE6sum_opI1TERK15sc |
| -   [cudaq::async_sample_result   | alar_operatorRR10product_opI1TE), |
|     (C++                          |     [\[3\]](api/                  |
|     type                          | languages/cpp_api.html#_CPPv4I0EN |
| )](api/languages/cpp_api.html#_CP | 5cudaq10product_opmiE6sum_opI1TER |
| Pv4N5cudaq19async_sample_resultE) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq::BaseRemoteRESTQPU     |     [\[4\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cla                           | q10product_opmiE6sum_opI1TERR15sc |
| ss)](api/languages/cpp_api.html#_ | alar_operatorRK10product_opI1TE), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [\[5\]](api/                  |
| -   [cudaq::bit_flip_channel (C++ | languages/cpp_api.html#_CPPv4I0EN |
|     cl                            | 5cudaq10product_opmiE6sum_opI1TER |
| ass)](api/languages/cpp_api.html# | R15scalar_operatorRK6sum_opI1TE), |
| _CPPv4N5cudaq16bit_flip_channelE) |     [\[6\]](api/langu             |
| -   [cudaq:                       | ages/cpp_api.html#_CPPv4I0EN5cuda |
| :bit_flip_channel::num_parameters | q10product_opmiE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     member)](api/langua           |     [\[7\]](api/                  |
| ges/cpp_api.html#_CPPv4N5cudaq16b | languages/cpp_api.html#_CPPv4I0EN |
| it_flip_channel14num_parametersE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cud                          | R15scalar_operatorRR6sum_opI1TE), |
| aq::bit_flip_channel::num_targets |     [\[8\]](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4NKR5cudaq10product_ |
|     member)](api/lan              | opmiERK10product_opI9HandlerTyE), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[9\]](api/language          |
| 16bit_flip_channel11num_targetsE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::boson_handler (C++    | roduct_opmiERK15scalar_operator), |
|                                   |     [\[10\]](api/languages/       |
|  class)](api/languages/cpp_api.ht | cpp_api.html#_CPPv4NKR5cudaq10pro |
| ml#_CPPv4N5cudaq13boson_handlerE) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::boson_op (C++         |     [\[11\]](api/languages/cpp_a  |
|     type)](api/languages/cpp_     | pi.html#_CPPv4NKR5cudaq10product_ |
| api.html#_CPPv4N5cudaq8boson_opE) | opmiERR10product_opI9HandlerTyE), |
| -   [cudaq::boson_op_term (C++    |     [\[12\]](api/language         |
|                                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|   type)](api/languages/cpp_api.ht | roduct_opmiERR15scalar_operator), |
| ml#_CPPv4N5cudaq13boson_op_termE) |     [\[13\]](api/languages/       |
| -   [cudaq::CodeGenConfig (C++    | cpp_api.html#_CPPv4NKR5cudaq10pro |
|                                   | duct_opmiERR6sum_opI9HandlerTyE), |
| struct)](api/languages/cpp_api.ht |     [\[                           |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | 14\]](api/languages/cpp_api.html# |
| -   [cudaq::commutation_relations | _CPPv4NKR5cudaq10product_opmiEv), |
|     (C++                          |     [\[15\]](api/languages/cpp_   |
|     struct)]                      | api.html#_CPPv4NO5cudaq10product_ |
| (api/languages/cpp_api.html#_CPPv | opmiERK10product_opI9HandlerTyE), |
| 4N5cudaq21commutation_relationsE) |     [\[16\]](api/languag          |
| -   [cudaq::complex (C++          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     type)](api/languages/cpp      | roduct_opmiERK15scalar_operator), |
| _api.html#_CPPv4N5cudaq7complexE) |     [\[17\]](api/languages        |
| -   [cudaq::complex_matrix (C++   | /cpp_api.html#_CPPv4NO5cudaq10pro |
|                                   | duct_opmiERK6sum_opI9HandlerTyE), |
| class)](api/languages/cpp_api.htm |     [\[18\]](api/languages/cpp_   |
| l#_CPPv4N5cudaq14complex_matrixE) | api.html#_CPPv4NO5cudaq10product_ |
| -                                 | opmiERR10product_opI9HandlerTyE), |
|   [cudaq::complex_matrix::adjoint |     [\[19\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     function)](a                  | roduct_opmiERR15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[20\]](api/languages        |
| 5cudaq14complex_matrix7adjointEv) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::                      | duct_opmiERR6sum_opI9HandlerTyE), |
| complex_matrix::diagonal_elements |     [                             |
|     (C++                          | \[21\]](api/languages/cpp_api.htm |
|     function)](api/languages      | l#_CPPv4NO5cudaq10product_opmiEv) |
| /cpp_api.html#_CPPv4NK5cudaq14com | -   [cudaq::product_op::operator/ |
| plex_matrix17diagonal_elementsEi) |     (C++                          |
| -   [cudaq::complex_matrix::dump  |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/language       | roduct_opdvERK15scalar_operator), |
| s/cpp_api.html#_CPPv4NK5cudaq14co |     [\[1\]](api/language          |
| mplex_matrix4dumpERNSt7ostreamE), | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     [\[1\]]                       | roduct_opdvERR15scalar_operator), |
| (api/languages/cpp_api.html#_CPPv |     [\[2\]](api/languag           |
| 4NK5cudaq14complex_matrix4dumpEv) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [c                            | roduct_opdvERK15scalar_operator), |
| udaq::complex_matrix::eigenvalues |     [\[3\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     function)](api/lan            | product_opdvERR15scalar_operator) |
| guages/cpp_api.html#_CPPv4NK5cuda | -                                 |
| q14complex_matrix11eigenvaluesEv) |    [cudaq::product_op::operator/= |
| -   [cu                           |     (C++                          |
| daq::complex_matrix::eigenvectors |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     function)](api/lang           | product_opdVERK15scalar_operator) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::product_op::operator= |
| 14complex_matrix12eigenvectorsEv) |     (C++                          |
| -   [c                            |     function)](api/l              |
| udaq::complex_matrix::exponential | anguages/cpp_api.html#_CPPv4I00EN |
|     (C++                          | 5cudaq10product_opaSER10product_o |
|     function)](api/la             | pI9HandlerTyERK10product_opI1TE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[1\]](api/languages/cpp     |
| q14complex_matrix11exponentialEv) | _api.html#_CPPv4N5cudaq10product_ |
| -                                 | opaSERK10product_opI9HandlerTyE), |
|  [cudaq::complex_matrix::identity |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     function)](api/languages      | _opaSERR10product_opI9HandlerTyE) |
| /cpp_api.html#_CPPv4N5cudaq14comp | -                                 |
| lex_matrix8identityEKNSt6size_tE) |    [cudaq::product_op::operator== |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::kronecker |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq10product |
|     function)](api/lang           | _opeqERK10product_opI9HandlerTyE) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -                                 |
| daq14complex_matrix9kroneckerE14c |  [cudaq::product_op::operator\[\] |
| omplex_matrix8Iterable8Iterable), |     (C++                          |
|     [\[1\]](api/l                 |     function)](ap                 |
| anguages/cpp_api.html#_CPPv4N5cud | i/languages/cpp_api.html#_CPPv4NK |
| aq14complex_matrix9kroneckerERK14 | 5cudaq10product_opixENSt6size_tE) |
| complex_matrixRK14complex_matrix) | -                                 |
| -   [cudaq::c                     |    [cudaq::product_op::product_op |
| omplex_matrix::minimal_eigenvalue |     (C++                          |
|     (C++                          |     f                             |
|     function)](api/languages/     | unction)](api/languages/cpp_api.h |
| cpp_api.html#_CPPv4NK5cudaq14comp | tml#_CPPv4I00EN5cudaq10product_op |
| lex_matrix18minimal_eigenvalueEv) | 10product_opERK10product_opI1TE), |
| -   [                             |     [\[1\]]                       |
| cudaq::complex_matrix::operator() | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4I00EN5cudaq10product_op10product |
|     function)](api/languages/cpp  | _opERK10product_opI1TERKN14matrix |
| _api.html#_CPPv4N5cudaq14complex_ | _handler20commutation_behaviorE), |
| matrixclENSt6size_tENSt6size_tE), |                                   |
|     [\[1\]](api/languages/cpp     |   [\[2\]](api/languages/cpp_api.h |
| _api.html#_CPPv4NK5cudaq14complex | tml#_CPPv4N5cudaq10product_op10pr |
| _matrixclENSt6size_tENSt6size_tE) | oduct_opENSt6size_tENSt6size_tE), |
| -   [                             |     [\[3\]](api/languages/cp      |
| cudaq::complex_matrix::operator\* | p_api.html#_CPPv4N5cudaq10product |
|     (C++                          | _op10product_opENSt7complexIdEE), |
|     function)](api/langua         |     [\[4\]](api/l                 |
| ges/cpp_api.html#_CPPv4N5cudaq14c | anguages/cpp_api.html#_CPPv4N5cud |
| omplex_matrixmlEN14complex_matrix | aq10product_op10product_opERK10pr |
| 10value_typeERK14complex_matrix), | oduct_opI9HandlerTyENSt6size_tE), |
|     [\[1\]                        |     [\[5\]](api/l                 |
| ](api/languages/cpp_api.html#_CPP | anguages/cpp_api.html#_CPPv4N5cud |
| v4N5cudaq14complex_matrixmlERK14c | aq10product_op10product_opERR10pr |
| omplex_matrixRK14complex_matrix), | oduct_opI9HandlerTyENSt6size_tE), |
|                                   |     [\[6\]](api/languages         |
|  [\[2\]](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4N5cudaq10prod |
| ml#_CPPv4N5cudaq14complex_matrixm | uct_op10product_opERR9HandlerTy), |
| lERK14complex_matrixRKNSt6vectorI |     [\[7\]](ap                    |
| N14complex_matrix10value_typeEEE) | i/languages/cpp_api.html#_CPPv4N5 |
| -                                 | cudaq10product_op10product_opEd), |
| [cudaq::complex_matrix::operator+ |     [\[8\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function                      | 5cudaq10product_op10product_opEv) |
| )](api/languages/cpp_api.html#_CP | -   [cuda                         |
| Pv4N5cudaq14complex_matrixplERK14 | q::product_op::to_diagonal_matrix |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -                                 |     function)](api/               |
| [cudaq::complex_matrix::operator- | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq10product_op18to_diagonal_mat |
|     function                      | rixENSt13unordered_mapINSt6size_t |
| )](api/languages/cpp_api.html#_CP | ENSt7int64_tEEERKNSt13unordered_m |
| Pv4N5cudaq14complex_matrixmiERK14 | apINSt6stringENSt7complexIdEEEEb) |
| complex_matrixRK14complex_matrix) | -   [cudaq::product_op::to_matrix |
| -   [cu                           |     (C++                          |
| daq::complex_matrix::operator\[\] |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4NK5cudaq10product_op9to_mat |
|  function)](api/languages/cpp_api | rixENSt13unordered_mapINSt6size_t |
| .html#_CPPv4N5cudaq14complex_matr | ENSt7int64_tEEERKNSt13unordered_m |
| ixixERKNSt6vectorINSt6size_tEEE), | apINSt6stringENSt7complexIdEEEEb) |
|     [\[1\]](api/languages/cpp_api | -   [cu                           |
| .html#_CPPv4NK5cudaq14complex_mat | daq::product_op::to_sparse_matrix |
| rixixERKNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq::complex_matrix::power |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     function)]                    | 5cudaq10product_op16to_sparse_mat |
| (api/languages/cpp_api.html#_CPPv | rixENSt13unordered_mapINSt6size_t |
| 4N5cudaq14complex_matrix5powerEi) | ENSt7int64_tEEERKNSt13unordered_m |
| -                                 | apINSt6stringENSt7complexIdEEEEb) |
|  [cudaq::complex_matrix::set_zero | -   [cudaq::product_op::to_string |
|     (C++                          |     (C++                          |
|     function)](ap                 |     function)](                   |
| i/languages/cpp_api.html#_CPPv4N5 | api/languages/cpp_api.html#_CPPv4 |
| cudaq14complex_matrix8set_zeroEv) | NK5cudaq10product_op9to_stringEv) |
| -                                 | -                                 |
| [cudaq::complex_matrix::to_string |  [cudaq::product_op::\~product_op |
|     (C++                          |     (C++                          |
|     function)](api/               |     fu                            |
| languages/cpp_api.html#_CPPv4NK5c | nction)](api/languages/cpp_api.ht |
| udaq14complex_matrix9to_stringEv) | ml#_CPPv4N5cudaq10product_opD0Ev) |
| -   [                             | -   [cudaq::ptsbe (C++            |
| cudaq::complex_matrix::value_type |     type)](api/languages/c        |
|     (C++                          | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
|     type)](api/                   | -   [cudaq::p                     |
| languages/cpp_api.html#_CPPv4N5cu | tsbe::ConditionalSamplingStrategy |
| daq14complex_matrix10value_typeE) |     (C++                          |
| -   [cudaq::contrib (C++          |     class)](api/languag           |
|     type)](api/languages/cpp      | es/cpp_api.html#_CPPv4N5cudaq5pts |
| _api.html#_CPPv4N5cudaq7contribE) | be27ConditionalSamplingStrategyE) |
| -                                 | -   [cudaq::ptsbe::C              |
| [cudaq::contrib::amplitude_encode | onditionalSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     function)](api/language       |                                   |
| s/cpp_api.html#_CPPv4N5cudaq7cont |    function)](api/languages/cpp_a |
| rib16amplitude_encodeENSt4spanIKN | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| St7complexIdEEEENSt7complexIdEE), | ditionalSamplingStrategy5cloneEv) |
|     [\[1\]](api/language          | -   [cuda                         |
| s/cpp_api.html#_CPPv4N5cudaq7cont | q::ptsbe::ConditionalSamplingStra |
| rib16amplitude_encodeENSt4spanIKN | tegy::ConditionalSamplingStrategy |
| St7complexIfEEEENSt7complexIdEE), |     (C++                          |
|     [\[2\]                        |     function)](api/lang           |
| ](api/languages/cpp_api.html#_CPP | uages/cpp_api.html#_CPPv4N5cudaq5 |
| v4N5cudaq7contrib16amplitude_enco | ptsbe27ConditionalSamplingStrateg |
| deENSt4spanIKdEENSt7complexIdEE), | y27ConditionalSamplingStrategyE19 |
|     [\[3\]                        | TrajectoryPredicateNSt8uint64_tE) |
| ](api/languages/cpp_api.html#_CPP | -                                 |
| v4N5cudaq7contrib16amplitude_enco |   [cudaq::ptsbe::ConditionalSampl |
| deENSt4spanIKfEENSt7complexIdEE), | ingStrategy::generateTrajectories |
|                                   |     (C++                          |
| [\[4\]](api/languages/cpp_api.htm |     function)](api/language       |
| l#_CPPv4N5cudaq7contrib16amplitud | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| e_encodeERK5stateNSt7complexIdEE) | be27ConditionalSamplingStrategy20 |
| -                                 | generateTrajectoriesENSt4spanIKN6 |
|   [cudaq::contrib::angular_encode | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe::               |
|                                   | ConditionalSamplingStrategy::name |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I0EN5cudaq7contrib14a |     function)](api/languages/cpp_ |
| ngular_encodeEvRR6KernelR10QuakeV | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| alueNSt4spanIKdEE12RotationAxis), | nditionalSamplingStrategy4nameEv) |
|     [\[1\]](api/languages/cpp_api | -   [cudaq:                       |
| .html#_CPPv4I0EN5cudaq7contrib14a | :ptsbe::ConditionalSamplingStrate |
| ngular_encodeEvRR6KernelR10QuakeV | gy::\~ConditionalSamplingStrategy |
| alueR10QuakeValue12RotationAxis), |     (C++                          |
|                                   |     function)](api/languages/     |
|   [\[2\]](api/languages/cpp_api.h | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| tml#_CPPv4I0EN5cudaq7contrib14ang | 7ConditionalSamplingStrategyD0Ev) |
| ular_encodeEvRR6KernelR10QuakeVal | -                                 |
| ueRKNSt6vectorIdEE12RotationAxis) | [cudaq::ptsbe::detail::NoisePoint |
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
| -   [cudaq::contrib::RotationAxis |     (C++                          |
|     (C++                          |     member)](api/languages/cpp_a  |
|     enum)                         | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| ](api/languages/cpp_api.html#_CPP | l10NoisePoint16circuit_locationE) |
| v4N5cudaq7contrib12RotationAxisE) | -   [cudaq::p                     |
| -                                 | tsbe::detail::NoisePoint::op_name |
|  [cudaq::contrib::RotationAxis::X |     (C++                          |
|     (C++                          |     member)](api/langu            |
|     enumerator)](                 | ages/cpp_api.html#_CPPv4N5cudaq5p |
| api/languages/cpp_api.html#_CPPv4 | tsbe6detail10NoisePoint7op_nameE) |
| N5cudaq7contrib12RotationAxis1XE) | -   [cudaq::                      |
| -                                 | ptsbe::detail::NoisePoint::qubits |
|  [cudaq::contrib::RotationAxis::Y |     (C++                          |
|     (C++                          |     member)](api/lang             |
|     enumerator)](                 | uages/cpp_api.html#_CPPv4N5cudaq5 |
| api/languages/cpp_api.html#_CPPv4 | ptsbe6detail10NoisePoint6qubitsE) |
| N5cudaq7contrib12RotationAxis1YE) | -   [cudaq::                      |
| -                                 | ptsbe::ExhaustiveSamplingStrategy |
|  [cudaq::contrib::RotationAxis::Z |     (C++                          |
|     (C++                          |     class)](api/langua            |
|     enumerator)](                 | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| api/languages/cpp_api.html#_CPPv4 | sbe26ExhaustiveSamplingStrategyE) |
| N5cudaq7contrib12RotationAxis1ZE) | -   [cudaq::ptsbe::               |
| -   [cudaq::CusvState (C++        | ExhaustiveSamplingStrategy::clone |
|                                   |     (C++                          |
|    class)](api/languages/cpp_api. |     function)](api/languages/cpp_ |
| html#_CPPv4I0EN5cudaq9CusvStateE) | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| -   [cudaq::dem_from_kernel (C++  | haustiveSamplingStrategy5cloneEv) |
|     function)](api                | -   [cu                           |
| /languages/cpp_api.html#_CPPv4I0D | daq::ptsbe::ExhaustiveSamplingStr |
| pEN5cudaq15dem_from_kernelENSt6st | ategy::ExhaustiveSamplingStrategy |
| ringERR13QuantumKernelDpRR4Args), |     (C++                          |
|     [                             |     function)](api/la             |
| \[1\]](api/languages/cpp_api.html | nguages/cpp_api.html#_CPPv4N5cuda |
| #_CPPv4I0DpEN5cudaq15dem_from_ker | q5ptsbe26ExhaustiveSamplingStrate |
| nelENSt6stringERR13QuantumKernelP | gy26ExhaustiveSamplingStrategyEv) |
| KN5cudaq11noise_modelEDpRR4Args), | -                                 |
|     [\[2\]](api/languages/c       |    [cudaq::ptsbe::ExhaustiveSampl |
| pp_api.html#_CPPv4I0DpEN5cudaq15d | ingStrategy::generateTrajectories |
| em_from_kernelENSt6stringERR13Qua |     (C++                          |
| ntumKernelPKN5cudaq11noise_modelE |     function)](api/languag        |
| RKN5cudaq11dem_optionsEDpRR4Args) | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| -   [cudaq::dem_options (C++      | sbe26ExhaustiveSamplingStrategy20 |
|                                   | generateTrajectoriesENSt4spanIKN6 |
|   struct)](api/languages/cpp_api. | detail10NoisePointEEENSt6size_tE) |
| html#_CPPv4N5cudaq11dem_optionsE) | -   [cudaq::ptsbe:                |
| -   [cudaq::d                     | :ExhaustiveSamplingStrategy::name |
| em_options::allow_gauge_detectors |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     member)](api/language         | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| s/cpp_api.html#_CPPv4N5cudaq11dem | xhaustiveSamplingStrategy4nameEv) |
| _options21allow_gauge_detectorsE) | -   [cuda                         |
| -   [cudaq::dem_options::appr     | q::ptsbe::ExhaustiveSamplingStrat |
| oximate_disjoint_errors_threshold | egy::\~ExhaustiveSamplingStrategy |
|     (C++                          |     (C++                          |
|     memb                          |     function)](api/languages      |
| er)](api/languages/cpp_api.html#_ | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| CPPv4N5cudaq11dem_options37approx | 26ExhaustiveSamplingStrategyD0Ev) |
| imate_disjoint_errors_thresholdE) | -   [cuda                         |
| -   [cuda                         | q::ptsbe::OrderedSamplingStrategy |
| q::dem_options::block_decompositi |     (C++                          |
| on_from_introducing_remnant_edges |     class)](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     member)](api/lang             | 5ptsbe23OrderedSamplingStrategyE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::ptsb                  |
| 1dem_options50block_decomposition | e::OrderedSamplingStrategy::clone |
| _from_introducing_remnant_edgesE) |     (C++                          |
| -   [cud                          |     function)](api/languages/c    |
| aq::dem_options::decompose_errors | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
|     (C++                          | 3OrderedSamplingStrategy5cloneEv) |
|     member)](api/lan              | -   [cudaq::ptsbe::OrderedSampl   |
| guages/cpp_api.html#_CPPv4N5cudaq | ingStrategy::generateTrajectories |
| 11dem_options16decompose_errorsE) |     (C++                          |
| -                                 |     function)](api/lang           |
|   [cudaq::dem_options::fold_loops | uages/cpp_api.html#_CPPv4NK5cudaq |
|     (C++                          | 5ptsbe23OrderedSamplingStrategy20 |
|     member)](a                    | generateTrajectoriesENSt4spanIKN6 |
| pi/languages/cpp_api.html#_CPPv4N | detail10NoisePointEEENSt6size_tE) |
| 5cudaq11dem_options10fold_loopsE) | -   [cudaq::pts                   |
| -   [cudaq::dem_optio             | be::OrderedSamplingStrategy::name |
| ns::ignore_decomposition_failures |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     member)](api/languages/cpp_ap | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| i.html#_CPPv4N5cudaq11dem_options | 23OrderedSamplingStrategy4nameEv) |
| 29ignore_decomposition_failuresE) | -                                 |
| -   [cudaq::dem_opt               |    [cudaq::ptsbe::OrderedSampling |
| ions::return_measurement_matrices | Strategy::OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/languages/cpp_   |     function)](                   |
| api.html#_CPPv4N5cudaq11dem_optio | api/languages/cpp_api.html#_CPPv4 |
| ns27return_measurement_matricesE) | N5cudaq5ptsbe23OrderedSamplingStr |
| -   [cudaq::depolarization1 (C++  | ategy23OrderedSamplingStrategyEv) |
|     c                             | -                                 |
| lass)](api/languages/cpp_api.html |  [cudaq::ptsbe::OrderedSamplingSt |
| #_CPPv4N5cudaq15depolarization1E) | rategy::\~OrderedSamplingStrategy |
| -   [cudaq::depolarization2 (C++  |     (C++                          |
|     c                             |     function)](api/langua         |
| lass)](api/languages/cpp_api.html | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| #_CPPv4N5cudaq15depolarization2E) | sbe23OrderedSamplingStrategyD0Ev) |
| -   [cudaq:                       | -   [cudaq::pts                   |
| :depolarization2::depolarization2 | be::ProbabilisticSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     class)](api/languages         |
| p_api.html#_CPPv4N5cudaq15depolar | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| ization215depolarization2EK4real) | 29ProbabilisticSamplingStrategyE) |
| -   [cudaq                        | -   [cudaq::ptsbe::Pro            |
| ::depolarization2::num_parameters | babilisticSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     member)](api/langu            |                                   |
| ages/cpp_api.html#_CPPv4N5cudaq15 |  function)](api/languages/cpp_api |
| depolarization214num_parametersE) | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| -   [cu                           | bilisticSamplingStrategy5cloneEv) |
| daq::depolarization2::num_targets | -                                 |
|     (C++                          | [cudaq::ptsbe::ProbabilisticSampl |
|     member)](api/la               | ingStrategy::generateTrajectories |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q15depolarization211num_targetsE) |     function)](api/languages/     |
| -                                 | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|    [cudaq::depolarization_channel | 29ProbabilisticSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     class)](                      | detail10NoisePointEEENSt6size_tE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::ptsbe::Pr             |
| N5cudaq22depolarization_channelE) | obabilisticSamplingStrategy::name |
| -   [cudaq::depol                 |     (C++                          |
| arization_channel::num_parameters |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     member)](api/languages/cp     | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| p_api.html#_CPPv4N5cudaq22depolar | abilisticSamplingStrategy4nameEv) |
| ization_channel14num_parametersE) | -   [cudaq::p                     |
| -   [cudaq::de                    | tsbe::ProbabilisticSamplingStrate |
| polarization_channel::num_targets | gy::ProbabilisticSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)]                    |
| /cpp_api.html#_CPPv4N5cudaq22depo | (api/languages/cpp_api.html#_CPPv |
| larization_channel11num_targetsE) | 4N5cudaq5ptsbe29ProbabilisticSamp |
| -   [cudaq::detail (C++           | lingStrategy29ProbabilisticSampli |
|     type)](api/languages/cp       | ngStrategyENSt8optionalINSt8uint6 |
| p_api.html#_CPPv4N5cudaq6detailE) | 4_tEEENSt8optionalINSt6size_tEEE) |
| -   [cudaq::detail::future (C++   | -   [cudaq::pts                   |
|                                   | be::ProbabilisticSamplingStrategy |
|   class)](api/languages/cpp_api.h | ::\~ProbabilisticSamplingStrategy |
| tml#_CPPv4N5cudaq6detail6futureE) |     (C++                          |
| -                                 |     function)](api/languages/cp   |
|    [cudaq::detail::future::future | p_api.html#_CPPv4N5cudaq5ptsbe29P |
|     (C++                          | robabilisticSamplingStrategyD0Ev) |
|     functi                        | -                                 |
| on)](api/languages/cpp_api.html#_ | [cudaq::ptsbe::PTSBEExecutionData |
| CPPv4N5cudaq6detail6future6future |     (C++                          |
| ERNSt6vectorI3JobEERNSt6stringERN |     struct)](ap                   |
| St3mapINSt6stringENSt6stringEEE), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[1\]](api/lan               | cudaq5ptsbe18PTSBEExecutionDataE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::ptsbe::PTSBE          |
| 6detail6future6futureERR6future), | ExecutionData::count_instructions |
|     [\[2\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/l              |
| v4N5cudaq6detail6future6futureEv) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [c                            | daq5ptsbe18PTSBEExecutionData18co |
| udaq::detail::kernel_builder_base | unt_instructionsE20TraceInstructi |
|     (C++                          | onTypeNSt8optionalINSt6stringEEE) |
|     class)](api/                  | -   [cudaq::ptsbe::P              |
| languages/cpp_api.html#_CPPv4N5cu | TSBEExecutionData::get_trajectory |
| daq6detail19kernel_builder_baseE) |     (C++                          |
| -   [cudaq::detail::              |     function                      |
| kernel_builder_base::operator\<\< | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NK5cudaq5ptsbe18PTSBEExecution |
|     function)](api/langu          | Data14get_trajectoryENSt6size_tE) |
| ages/cpp_api.html#_CPPv4N5cudaq6d | -   [cudaq::ptsbe:                |
| etail19kernel_builder_baselsERNSt | :PTSBEExecutionData::instructions |
| 7ostreamERK19kernel_builder_base) |     (C++                          |
| -                                 |     member)](api/languages/cp     |
| [cudaq::detail::KernelBuilderType | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     (C++                          | TSBEExecutionData12instructionsE) |
|     class)](ap                    | -   [cudaq::ptsbe:                |
| i/languages/cpp_api.html#_CPPv4N5 | :PTSBEExecutionData::trajectories |
| cudaq6detail17KernelBuilderTypeE) |     (C++                          |
| -   [cudaq::                      |     member)](api/languages/cp     |
| detail::KernelBuilderType::create | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     (C++                          | TSBEExecutionData12trajectoriesE) |
|     function                      | -   [cudaq::ptsbe::PTSBEOptions   |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq6detail17KernelBuilderT |     struc                         |
| ype6createEPN4mlir11MLIRContextE) | t)](api/languages/cpp_api.html#_C |
| -   [cudaq::detail::Ker           | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| nelBuilderType::KernelBuilderType | -   [cudaq::ptsbe::PTSB           |
|     (C++                          | EOptions::include_sequential_data |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 6detail17KernelBuilderType17Kerne |    member)](api/languages/cpp_api |
| lBuilderTypeERRNSt8functionIFN4ml | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| ir4TypeEPN4mlir11MLIRContextEEEE) | ptions23include_sequential_dataE) |
| -   [cudaq::detector (C++         | -   [cudaq::ptsb                  |
|     function)](api                | e::PTSBEOptions::max_trajectories |
| /languages/cpp_api.html#_CPPv4IDp |     (C++                          |
| EN5cudaq8detectorEvDpRR8MeasArgs) |     member)](api/languages/       |
| -   [cudaq::detectors (C++        | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
|     function)](api/languages/c    | 2PTSBEOptions16max_trajectoriesE) |
| pp_api.html#_CPPv4N5cudaq9detecto | -   [cudaq::ptsbe::PT             |
| rsERKNSt6vectorI14measure_resultE | SBEOptions::return_execution_data |
| ERKNSt6vectorI14measure_resultEE) |     (C++                          |
| -   [cudaq::diag_matrix_callback  |     member)](api/languages/cpp_a  |
|     (C++                          | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
|     class)                        | EOptions21return_execution_dataE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::pts                   |
| v4N5cudaq20diag_matrix_callbackE) | be::PTSBEOptions::shot_allocation |
| -   [cudaq::dyn (C++              |     (C++                          |
|     member)](api/languages        |     member)](api/languages        |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| -   [cudaq::ExecutionContext (C++ | 12PTSBEOptions15shot_allocationE) |
|     cl                            | -   [cud                          |
| ass)](api/languages/cpp_api.html# | aq::ptsbe::PTSBEOptions::strategy |
| _CPPv4N5cudaq16ExecutionContextE) |     (C++                          |
| -   [c                            |     member)](api/l                |
| udaq::ExecutionContext::asyncExec | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq5ptsbe12PTSBEOptions8strategyE) |
|     member)](api/                 | -   [cudaq::ptsbe::PTSBETrace     |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9asyncExecE) |     t                             |
| -   [cud                          | ype)](api/languages/cpp_api.html# |
| aq::ExecutionContext::asyncResult | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
|     (C++                          | -   [                             |
|     member)](api/lan              | cudaq::ptsbe::PTSSamplingStrategy |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11asyncResultE) |     class)](api                   |
| -   [cudaq:                       | /languages/cpp_api.html#_CPPv4N5c |
| :ExecutionContext::batchIteration | udaq5ptsbe19PTSSamplingStrategyE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/langua           | ptsbe::PTSSamplingStrategy::clone |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14batchIterationE) |     function)](api/languag        |
| -   [cudaq::E                     | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| xecutionContext::canHandleObserve | sbe19PTSSamplingStrategy5cloneEv) |
|     (C++                          | -   [cudaq::ptsbe::PTSSampl       |
|     member)](api/language         | ingStrategy::generateTrajectories |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     (C++                          |
| cutionContext16canHandleObserveE) |     function)](api/               |
| -   [cudaq::Executio              | languages/cpp_api.html#_CPPv4NK5c |
| nContext::deferredKernelException | udaq5ptsbe19PTSSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/languages/cpp_a  | detail10NoisePointEEENSt6size_tE) |
| pi.html#_CPPv4N5cudaq16ExecutionC | -   [cudaq:                       |
| ontext23deferredKernelExceptionE) | :ptsbe::PTSSamplingStrategy::name |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::ExecutionContext |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq5p |
|     func                          | tsbe19PTSSamplingStrategy4nameEv) |
| tion)](api/languages/cpp_api.html | -   [cudaq::ptsbe::PTSSampli      |
| #_CPPv4N5cudaq16ExecutionContext1 | ngStrategy::\~PTSSamplingStrategy |
| 6ExecutionContextERKNSt6stringE), |     (C++                          |
|     [\[1\]](api/languages/        |     function)](api/la             |
| cpp_api.html#_CPPv4N5cudaq16Execu | nguages/cpp_api.html#_CPPv4N5cuda |
| tionContext16ExecutionContextERKN | q5ptsbe19PTSSamplingStrategyD0Ev) |
| St6stringENSt6size_tENSt6size_tE) | -   [cudaq::ptsbe::sample (C++    |
| -   [cudaq::E                     |                                   |
| xecutionContext::expectationValue |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
|     member)](api/language         | mpleE13sample_resultRK14sample_op |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | tionsRR13QuantumKernelDpRR4Args), |
| cutionContext16expectationValueE) |     [\[1\]](api                   |
| -   [cudaq::Execu                 | /languages/cpp_api.html#_CPPv4I0D |
| tionContext::explicitMeasurements | pEN5cudaq5ptsbe6sampleE13sample_r |
|     (C++                          | esultRKN5cudaq11noise_modelENSt6s |
|     member)](api/languages/cp     | ize_tERR13QuantumKernelDpRR4Args) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::ptsbe::sample_async   |
| onContext20explicitMeasurementsE) |     (C++                          |
| -   [cuda                         |     function)](a                  |
| q::ExecutionContext::futureResult | pi/languages/cpp_api.html#_CPPv4I |
|     (C++                          | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
|     member)](api/lang             | 9async_sample_resultRK14sample_op |
| uages/cpp_api.html#_CPPv4N5cudaq1 | tionsRR13QuantumKernelDpRR4Args), |
| 6ExecutionContext12futureResultE) |     [\[1\]](api/languages/cp      |
| -   [cudaq::ExecutionContext      | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| ::hasConditionalsOnMeasureResults | be12sample_asyncE19async_sample_r |
|     (C++                          | esultRKN5cudaq11noise_modelENSt6s |
|     mem                           | ize_tERR13QuantumKernelDpRR4Args) |
| ber)](api/languages/cpp_api.html# | -   [cudaq::ptsbe::sample_options |
| _CPPv4N5cudaq16ExecutionContext31 |     (C++                          |
| hasConditionalsOnMeasureResultsE) |     struct)                       |
| -   [cudaq:                       | ](api/languages/cpp_api.html#_CPP |
| :ExecutionContext::inKernelLaunch | v4N5cudaq5ptsbe14sample_optionsE) |
|     (C++                          | -   [cudaq::ptsbe::sample_result  |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     class                         |
| xecutionContext14inKernelLaunchE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::Executi               | Pv4N5cudaq5ptsbe13sample_resultE) |
| onContext::invocationResultBuffer | -   [cudaq::pts                   |
|     (C++                          | be::sample_result::execution_data |
|     member)](api/languages/cpp_   |     (C++                          |
| api.html#_CPPv4N5cudaq16Execution |     function)](api/languages/c    |
| Context22invocationResultBufferE) | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| -   [cu                           | 3sample_result14execution_dataEv) |
| daq::ExecutionContext::kernelName | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::has_execution_data |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q16ExecutionContext10kernelNameE) |    function)](api/languages/cpp_a |
| -   [cud                          | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| aq::ExecutionContext::kernelTrace | ple_result18has_execution_dataEv) |
|     (C++                          | -   [cudaq::pt                    |
|     member)](api/lan              | sbe::sample_result::sample_result |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11kernelTraceE) |     function)](api/l              |
| -   [cudaq:                       | anguages/cpp_api.html#_CPPv4N5cud |
| :ExecutionContext::msm_dimensions | aq5ptsbe13sample_result13sample_r |
|     (C++                          | esultERRN5cudaq13sample_resultE), |
|     member)](api/langua           |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq16E |  [\[1\]](api/languages/cpp_api.ht |
| xecutionContext14msm_dimensionsE) | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| -   [cudaq::                      | sult13sample_resultERRN5cudaq13sa |
| ExecutionContext::msm_prob_err_id | mple_resultE18PTSBEExecutionData) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     member)](api/languag          | sample_result::set_execution_data |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15msm_prob_err_idE) |     function)](api/               |
| -   [cudaq::Ex                    | languages/cpp_api.html#_CPPv4N5cu |
| ecutionContext::msm_probabilities | daq5ptsbe13sample_result18set_exe |
|     (C++                          | cution_dataE18PTSBEExecutionData) |
|     member)](api/languages        | -   [cud                          |
| /cpp_api.html#_CPPv4N5cudaq16Exec | aq::ptsbe::ShotAllocationStrategy |
| utionContext17msm_probabilitiesE) |     (C++                          |
| -                                 |     struct)](using                |
|    [cudaq::ExecutionContext::name | /examples/ptsbe.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe22ShotAllocationStrategyE) |
|     member)]                      | -   [cudaq::ptsbe::ShotAllocatio  |
| (api/languages/cpp_api.html#_CPPv | nStrategy::ShotAllocationStrategy |
| 4N5cudaq16ExecutionContext4nameE) |     (C++                          |
| -   [cu                           |     function)                     |
| daq::ExecutionContext::noiseModel | ](using/examples/ptsbe.html#_CPPv |
|     (C++                          | 4N5cudaq5ptsbe22ShotAllocationStr |
|     member)](api/la               | ategy22ShotAllocationStrategyE4Ty |
| nguages/cpp_api.html#_CPPv4N5cuda | pedNSt8optionalINSt8uint64_tEEE), |
| q16ExecutionContext10noiseModelE) |     [\[1\                         |
| -   [cudaq::Exe                   | ]](using/examples/ptsbe.html#_CPP |
| cutionContext::numberTrajectories | v4N5cudaq5ptsbe22ShotAllocationSt |
|     (C++                          | rategy22ShotAllocationStrategyEv) |
|     member)](api/languages/       | -   [cudaq::pt                    |
| cpp_api.html#_CPPv4N5cudaq16Execu | sbe::ShotAllocationStrategy::Type |
| tionContext18numberTrajectoriesE) |     (C++                          |
| -   [c                            |     enum)](using/exam             |
| udaq::ExecutionContext::optResult | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|     (C++                          | be22ShotAllocationStrategy4TypeE) |
|     member)](api/                 | -   [cudaq::ptsbe::ShotAllocatio  |
| languages/cpp_api.html#_CPPv4N5cu | nStrategy::Type::HIGH_WEIGHT_BIAS |
| daq16ExecutionContext9optResultE) |     (C++                          |
| -                                 |     enumerat                      |
|   [cudaq::ExecutionContext::qpuId | or)](using/examples/ptsbe.html#_C |
|     (C++                          | PPv4N5cudaq5ptsbe22ShotAllocation |
|     member)](                     | Strategy4Type16HIGH_WEIGHT_BIASE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::ptsbe::ShotAllocati   |
| N5cudaq16ExecutionContext5qpuIdE) | onStrategy::Type::LOW_WEIGHT_BIAS |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::registerNames |     enumera                       |
|     (C++                          | tor)](using/examples/ptsbe.html#_ |
|     member)](api/langu            | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| ages/cpp_api.html#_CPPv4N5cudaq16 | nStrategy4Type15LOW_WEIGHT_BIASE) |
| ExecutionContext13registerNamesE) | -   [cudaq::ptsbe::ShotAlloc      |
| -   [cu                           | ationStrategy::Type::PROPORTIONAL |
| daq::ExecutionContext::reorderIdx |     (C++                          |
|     (C++                          |     enum                          |
|     member)](api/la               | erator)](using/examples/ptsbe.htm |
| nguages/cpp_api.html#_CPPv4N5cuda | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| q16ExecutionContext10reorderIdxE) | tionStrategy4Type12PROPORTIONALE) |
| -                                 | -   [cudaq::ptsbe::Shot           |
|  [cudaq::ExecutionContext::result | AllocationStrategy::Type::UNIFORM |
|     (C++                          |     (C++                          |
|     member)](a                    |                                   |
| pi/languages/cpp_api.html#_CPPv4N |   enumerator)](using/examples/pts |
| 5cudaq16ExecutionContext6resultE) | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| -                                 | AllocationStrategy4Type7UNIFORME) |
|   [cudaq::ExecutionContext::shots | -                                 |
|     (C++                          |   [cudaq::ptsbe::TraceInstruction |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     struct)](                     |
| N5cudaq16ExecutionContext5shotsE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::                      | N5cudaq5ptsbe16TraceInstructionE) |
| ExecutionContext::simulationState | -   [cudaq:                       |
|     (C++                          | :ptsbe::TraceInstruction::channel |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     member)](api/lang             |
| ecutionContext15simulationStateE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -                                 | ptsbe16TraceInstruction7channelE) |
|    [cudaq::ExecutionContext::spin | -   [cudaq::                      |
|     (C++                          | ptsbe::TraceInstruction::controls |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     member)](api/langu            |
| 4N5cudaq16ExecutionContext4spinE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [cudaq::                      | tsbe16TraceInstruction8controlsE) |
| ExecutionContext::totalIterations | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::name |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     member)](api/l                |
| ecutionContext15totalIterationsE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::ExecutionResult (C++  | aq5ptsbe16TraceInstruction4nameE) |
|     st                            | -   [cudaq                        |
| ruct)](api/languages/cpp_api.html | ::ptsbe::TraceInstruction::params |
| #_CPPv4N5cudaq15ExecutionResultE) |     (C++                          |
| -   [cud                          |     member)](api/lan              |
| aq::ExecutionResult::appendResult | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 5ptsbe16TraceInstruction6paramsE) |
|     functio                       | -   [cudaq:                       |
| n)](api/languages/cpp_api.html#_C | :ptsbe::TraceInstruction::targets |
| PPv4N5cudaq15ExecutionResult12app |     (C++                          |
| endResultENSt6stringENSt6size_tE) |     member)](api/lang             |
| -   [cu                           | uages/cpp_api.html#_CPPv4N5cudaq5 |
| daq::ExecutionResult::deserialize | ptsbe16TraceInstruction7targetsE) |
|     (C++                          | -   [cudaq::ptsbe::T              |
|     function)                     | raceInstruction::TraceInstruction |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq15ExecutionResult11deser |                                   |
| ializeERNSt6vectorINSt6size_tEEE) |   function)](api/languages/cpp_ap |
| -   [cudaq:                       | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| :ExecutionResult::ExecutionResult | Instruction16TraceInstructionE20T |
|     (C++                          | raceInstructionTypeNSt6stringENSt |
|     functio                       | 6vectorINSt6size_tEEENSt6vectorIN |
| n)](api/languages/cpp_api.html#_C | St6size_tEEENSt6vectorIdEENSt8opt |
| PPv4N5cudaq15ExecutionResult15Exe | ionalIN5cudaq13kraus_channelEEE), |
| cutionResultE16CountsDictionary), |     [\[1\]](api/languages/cpp_a   |
|     [\[1\]](api/lan               | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| guages/cpp_api.html#_CPPv4N5cudaq | eInstruction16TraceInstructionEv) |
| 15ExecutionResult15ExecutionResul | -   [cud                          |
| tE16CountsDictionaryNSt6stringE), | aq::ptsbe::TraceInstruction::type |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     member)](api/l                |
| Pv4N5cudaq15ExecutionResult15Exec | anguages/cpp_api.html#_CPPv4N5cud |
| utionResultE16CountsDictionaryd), | aq5ptsbe16TraceInstruction4typeE) |
|                                   | -   [c                            |
|    [\[3\]](api/languages/cpp_api. | udaq::ptsbe::TraceInstructionType |
| html#_CPPv4N5cudaq15ExecutionResu |     (C++                          |
| lt15ExecutionResultENSt6stringE), |     enum)](api/                   |
|     [\[4\                         | languages/cpp_api.html#_CPPv4N5cu |
| ]](api/languages/cpp_api.html#_CP | daq5ptsbe20TraceInstructionTypeE) |
| Pv4N5cudaq15ExecutionResult15Exec | -   [cudaq::                      |
| utionResultERK15ExecutionResult), | ptsbe::TraceInstructionType::Gate |
|     [\[5\]](api/language          |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     enumerator)](api/langu        |
| cutionResult15ExecutionResultEd), | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     [\[6\]](api/languag           | tsbe20TraceInstructionType4GateE) |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | -   [cudaq::ptsbe::               |
| ecutionResult15ExecutionResultEv) | TraceInstructionType::Measurement |
| -   [                             |     (C++                          |
| cudaq::ExecutionResult::operator= |                                   |
|     (C++                          |    enumerator)](api/languages/cpp |
|     function)](api/languages/     | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| cpp_api.html#_CPPv4N5cudaq15Execu | aceInstructionType11MeasurementE) |
| tionResultaSERK15ExecutionResult) | -   [cudaq::p                     |
| -   [c                            | tsbe::TraceInstructionType::Noise |
| udaq::ExecutionResult::operator== |     (C++                          |
|     (C++                          |     enumerator)](api/langua       |
|     function)](api/languages/c    | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| pp_api.html#_CPPv4NK5cudaq15Execu | sbe20TraceInstructionType5NoiseE) |
| tionResulteqERK15ExecutionResult) | -   [                             |
| -   [cud                          | cudaq::ptsbe::TrajectoryPredicate |
| aq::ExecutionResult::registerName |     (C++                          |
|     (C++                          |     type)](api                    |
|     member)](api/lan              | /languages/cpp_api.html#_CPPv4N5c |
| guages/cpp_api.html#_CPPv4N5cudaq | udaq5ptsbe19TrajectoryPredicateE) |
| 15ExecutionResult12registerNameE) | -   [cudaq::QPU (C++              |
| -   [cudaq                        |     class)](api/languages         |
| ::ExecutionResult::sequentialData | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     (C++                          | -   [cudaq::QPU::beginExecution   |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     function                      |
| ExecutionResult14sequentialDataE) | )](api/languages/cpp_api.html#_CP |
| -   [                             | Pv4N5cudaq3QPU14beginExecutionEv) |
| cudaq::ExecutionResult::serialize | -   [cuda                         |
|     (C++                          | q::QPU::configureExecutionContext |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4NK5cu |     funct                         |
| daq15ExecutionResult9serializeEv) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::fermion_handler (C++  | _CPPv4NK5cudaq3QPU25configureExec |
|     c                             | utionContextER16ExecutionContext) |
| lass)](api/languages/cpp_api.html | -   [cudaq::QPU::endExecution     |
| #_CPPv4N5cudaq15fermion_handlerE) |     (C++                          |
| -   [cudaq::fermion_op (C++       |     functi                        |
|     type)](api/languages/cpp_api  | on)](api/languages/cpp_api.html#_ |
| .html#_CPPv4N5cudaq10fermion_opE) | CPPv4N5cudaq3QPU12endExecutionEv) |
| -   [cudaq::fermion_op_term (C++  | -   [cudaq::QPU::enqueue (C++     |
|                                   |     function)](ap                 |
| type)](api/languages/cpp_api.html | i/languages/cpp_api.html#_CPPv4N5 |
| #_CPPv4N5cudaq15fermion_op_termE) | cudaq3QPU7enqueueER11QuantumTask) |
| -   [cudaq::FermioniqQPU (C++     | -   [cud                          |
|                                   | aq::QPU::finalizeExecutionContext |
|   class)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq12FermioniqQPUE) |     func                          |
| -   [cudaq::get_state (C++        | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4NK5cudaq3QPU24finalizeExec |
|    function)](api/languages/cpp_a | utionContextER16ExecutionContext) |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | -   [cudaq::QPU::getCompileTarget |
| ateEDaRR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::gradient (C++         |     function)](api/languages/c    |
|     class)](api/languages/cpp_    | pp_api.html#_CPPv4N5cudaq3QPU16ge |
| api.html#_CPPv4N5cudaq8gradientE) | tCompileTargetERK13sample_policy) |
| -   [cudaq::gradient::clone (C++  | -   [cudaq::QPU::getConnectivity  |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     function)                     |
| l#_CPPv4N5cudaq8gradient5cloneEv) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::gradient::compute     | v4N5cudaq3QPU15getConnectivityEv) |
|     (C++                          | -                                 |
|     function)](api/language       | [cudaq::QPU::getExecutionThreadId |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     (C++                          |
| ient7computeERKNSt6vectorIdEERKNS |     function)](api/               |
| t8functionIFdNSt6vectorIdEEEEEd), | languages/cpp_api.html#_CPPv4NK5c |
|     [\[1\]](ap                    | udaq3QPU20getExecutionThreadIdEv) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq::QPU::getNumQubits     |
| cudaq8gradient7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |     functi                        |
| -   [cudaq::gradient::gradient    | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     function)](api/lang           | -   [                             |
| uages/cpp_api.html#_CPPv4I00EN5cu | cudaq::QPU::getRemoteCapabilities |
| daq8gradient8gradientER7KernelT), |     (C++                          |
|                                   |     function)](api/l              |
|    [\[1\]](api/languages/cpp_api. | anguages/cpp_api.html#_CPPv4NK5cu |
| html#_CPPv4I00EN5cudaq8gradient8g | daq3QPU21getRemoteCapabilitiesEv) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::QPU::isEmulated (C++  |
|     [\[2\                         |     func                          |
| ]](api/languages/cpp_api.html#_CP | tion)](api/languages/cpp_api.html |
| Pv4I00EN5cudaq8gradient8gradientE | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::QPU::isSimulator (C++ |
|     [\[3                          |     funct                         |
| \]](api/languages/cpp_api.html#_C | ion)](api/languages/cpp_api.html# |
| PPv4N5cudaq8gradient8gradientERRN | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QPU::onRandomSeedSet  |
|     [\[                           |     (C++                          |
| 4\]](api/languages/cpp_api.html#_ |     function)](api/lang           |
| CPPv4N5cudaq8gradient8gradientEv) | uages/cpp_api.html#_CPPv4N5cudaq3 |
| -   [cudaq::gradient::setArgs     | QPU15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::QPU (C++         |
|     fu                            |     functio                       |
| nction)](api/languages/cpp_api.ht | n)](api/languages/cpp_api.html#_C |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| tArgsEvR13QuantumKernelDpRR4Args) |                                   |
| -   [cudaq::gradient::setKernel   |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     function)](api/languages/c    |     [\[2\]](api/languages/cpp_    |
| pp_api.html#_CPPv4I0EN5cudaq8grad | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| ient9setKernelEvR13QuantumKernel) | -   [cudaq::QPU::setId (C++       |
| -   [cud                          |     function                      |
| aq::gradients::central_difference | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     class)](api/la                | -   [cudaq::QPU::setShots (C++    |
| nguages/cpp_api.html#_CPPv4N5cuda |     f                             |
| q9gradients18central_differenceE) | unction)](api/languages/cpp_api.h |
| -   [cudaq::gra                   | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| dients::central_difference::clone | -   [cudaq::                      |
|     (C++                          | QPU::supportsExplicitMeasurements |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)](api/languag        |
| ents18central_difference5cloneEv) | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| -   [cudaq::gradi                 | 28supportsExplicitMeasurementsEv) |
| ents::central_difference::compute | -   [cudaq::QPU::\~QPU (C++       |
|     (C++                          |     function)](api/languages/cp   |
|     function)](                   | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::QPUState (C++         |
| N5cudaq9gradients18central_differ |     class)](api/languages/cpp_    |
| ence7computeERKNSt6vectorIdEERKNS | api.html#_CPPv4N5cudaq8QPUStateE) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::qreg (C++             |
|                                   |     class)](api/lan               |
|   [\[1\]](api/languages/cpp_api.h | guages/cpp_api.html#_CPPv4I_NSt6s |
| tml#_CPPv4N5cudaq9gradients18cent | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| ral_difference7computeERKNSt6vect | -   [cudaq::qreg::back (C++       |
| orIdEERNSt6vectorIdEERK7spin_opd) |     function)                     |
| -   [cudaq::gradie                | ](api/languages/cpp_api.html#_CPP |
| nts::central_difference::gradient | v4N5cudaq4qreg4backENSt6size_tE), |
|     (C++                          |     [\[1\]](api/languages/cpp_ap  |
|     functio                       | i.html#_CPPv4N5cudaq4qreg4backEv) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::qreg::begin (C++      |
| PPv4I00EN5cudaq9gradients18centra |                                   |
| l_difference8gradientER7KernelT), |  function)](api/languages/cpp_api |
|     [\[1\]](api/langua            | .html#_CPPv4N5cudaq4qreg5beginEv) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::qreg::clear (C++      |
| q9gradients18central_difference8g |                                   |
| radientER7KernelTRR10ArgsMapper), |  function)](api/languages/cpp_api |
|     [\[2\]](api/languages/cpp_    | .html#_CPPv4N5cudaq4qreg5clearEv) |
| api.html#_CPPv4I00EN5cudaq9gradie | -   [cudaq::qreg::front (C++      |
| nts18central_difference8gradientE |     function)]                    |
| RR13QuantumKernelRR10ArgsMapper), | (api/languages/cpp_api.html#_CPPv |
|     [\[3\]](api/languages/cpp     | 4N5cudaq4qreg5frontENSt6size_tE), |
| _api.html#_CPPv4N5cudaq9gradients |     [\[1\]](api/languages/cpp_api |
| 18central_difference8gradientERRN | .html#_CPPv4N5cudaq4qreg5frontEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::qreg::operator\[\]    |
|     [\[4\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq9gradient |     functi                        |
| s18central_difference8gradientEv) | on)](api/languages/cpp_api.html#_ |
| -   [cud                          | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| aq::gradients::forward_difference | -   [cudaq::qreg::qreg (C++       |
|     (C++                          |     function)                     |
|     class)](api/la                | ](api/languages/cpp_api.html#_CPP |
| nguages/cpp_api.html#_CPPv4N5cuda | v4N5cudaq4qreg4qregENSt6size_tE), |
| q9gradients18forward_differenceE) |     [\[1\]](api/languages/cpp_ap  |
| -   [cudaq::gra                   | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| dients::forward_difference::clone | -   [cudaq::qreg::size (C++       |
|     (C++                          |                                   |
|     function)](api/languages      |  function)](api/languages/cpp_api |
| /cpp_api.html#_CPPv4N5cudaq9gradi | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| ents18forward_difference5cloneEv) | -   [cudaq::qreg::slice (C++      |
| -   [cudaq::gradi                 |     function)](api/langu          |
| ents::forward_difference::compute | ages/cpp_api.html#_CPPv4N5cudaq4q |
|     (C++                          | reg5sliceENSt6size_tENSt6size_tE) |
|     function)](                   | -   [cudaq::qreg::value_type (C++ |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq9gradients18forward_differ | type)](api/languages/cpp_api.html |
| ence7computeERKNSt6vectorIdEERKNS | #_CPPv4N5cudaq4qreg10value_typeE) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::qspan (C++            |
|                                   |     class)](api/lang              |
|   [\[1\]](api/languages/cpp_api.h | uages/cpp_api.html#_CPPv4I_NSt6si |
| tml#_CPPv4N5cudaq9gradients18forw | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| ard_difference7computeERKNSt6vect | -   [cudaq::QuakeValue (C++       |
| orIdEERNSt6vectorIdEERK7spin_opd) |     class)](api/languages/cpp_api |
| -   [cudaq::gradie                | .html#_CPPv4N5cudaq10QuakeValueE) |
| nts::forward_difference::gradient | -   [cudaq::Q                     |
|     (C++                          | uakeValue::canValidateNumElements |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |     function)](api/languages      |
| PPv4I00EN5cudaq9gradients18forwar | /cpp_api.html#_CPPv4N5cudaq10Quak |
| d_difference8gradientER7KernelT), | eValue22canValidateNumElementsEv) |
|     [\[1\]](api/langua            | -                                 |
| ges/cpp_api.html#_CPPv4I00EN5cuda |  [cudaq::QuakeValue::constantSize |
| q9gradients18forward_difference8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api                |
|     [\[2\]](api/languages/cpp_    | /languages/cpp_api.html#_CPPv4N5c |
| api.html#_CPPv4I00EN5cudaq9gradie | udaq10QuakeValue12constantSizeEv) |
| nts18forward_difference8gradientE | -   [cudaq::QuakeValue::dump (C++ |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/lan            |
|     [\[3\]](api/languages/cpp     | guages/cpp_api.html#_CPPv4N5cudaq |
| _api.html#_CPPv4N5cudaq9gradients | 10QuakeValue4dumpERNSt7ostreamE), |
| 18forward_difference8gradientERRN |     [\                            |
| St8functionIFvNSt6vectorIdEEEEE), | [1\]](api/languages/cpp_api.html# |
|     [\[4\]](api/languages/cp      | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| p_api.html#_CPPv4N5cudaq9gradient | -   [cudaq                        |
| s18forward_difference8gradientEv) | ::QuakeValue::getRequiredElements |
| -   [                             |     (C++                          |
| cudaq::gradients::parameter_shift |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq10Q |
|     class)](api                   | uakeValue19getRequiredElementsEv) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::QuakeValue::getValue  |
| udaq9gradients15parameter_shiftE) |     (C++                          |
| -   [cudaq::                      |     function)]                    |
| gradients::parameter_shift::clone | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NK5cudaq10QuakeValue8getValueEv) |
|     function)](api/langua         | -   [cudaq::QuakeValue::inverse   |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     (C++                          |
| adients15parameter_shift5cloneEv) |     function)                     |
| -   [cudaq::gr                    | ](api/languages/cpp_api.html#_CPP |
| adients::parameter_shift::compute | v4NK5cudaq10QuakeValue7inverseEv) |
|     (C++                          | -   [cudaq::QuakeValue::isStdVec  |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)                     |
| Pv4N5cudaq9gradients15parameter_s | ](api/languages/cpp_api.html#_CPP |
| hift7computeERKNSt6vectorIdEERKNS | v4N5cudaq10QuakeValue8isStdVecEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -                                 |
|     [\[1\]](api/languages/cpp_ap  |    [cudaq::QuakeValue::operator\* |
| i.html#_CPPv4N5cudaq9gradients15p |     (C++                          |
| arameter_shift7computeERKNSt6vect |     function)](api                |
| orIdEERNSt6vectorIdEERK7spin_opd) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gra                   | udaq10QuakeValuemlE10QuakeValue), |
| dients::parameter_shift::gradient |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     func                          | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| tion)](api/languages/cpp_api.html | -   [cudaq::QuakeValue::operator+ |
| #_CPPv4I00EN5cudaq9gradients15par |     (C++                          |
| ameter_shift8gradientER7KernelT), |     function)](api                |
|     [\[1\]](api/lan               | /languages/cpp_api.html#_CPPv4N5c |
| guages/cpp_api.html#_CPPv4I00EN5c | udaq10QuakeValueplE10QuakeValue), |
| udaq9gradients15parameter_shift8g |     [                             |
| radientER7KernelTRR10ArgsMapper), | \[1\]](api/languages/cpp_api.html |
|     [\[2\]](api/languages/c       | #_CPPv4N5cudaq10QuakeValueplEKd), |
| pp_api.html#_CPPv4I00EN5cudaq9gra |                                   |
| dients15parameter_shift8gradientE | [\[2\]](api/languages/cpp_api.htm |
| RR13QuantumKernelRR10ArgsMapper), | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|     [\[3\]](api/languages/        | -   [cudaq::QuakeValue::operator- |
| cpp_api.html#_CPPv4N5cudaq9gradie |     (C++                          |
| nts15parameter_shift8gradientERRN |     function)](api                |
| St8functionIFvNSt6vectorIdEEEEE), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[4\]](api/languages         | udaq10QuakeValuemiE10QuakeValue), |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     [                             |
| ents15parameter_shift8gradientEv) | \[1\]](api/languages/cpp_api.html |
| -   [cudaq::kernel_builder (C++   | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     clas                          |     [                             |
| s)](api/languages/cpp_api.html#_C | \[2\]](api/languages/cpp_api.html |
| PPv4IDpEN5cudaq14kernel_builderE) | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| -   [c                            |                                   |
| udaq::kernel_builder::constantVal | [\[3\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     function)](api/la             | -   [cudaq::QuakeValue::operator/ |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14kernel_builder11constantValEd) |     function)](api                |
| -                                 | /languages/cpp_api.html#_CPPv4N5c |
|  [cudaq::kernel_builder::detector | udaq10QuakeValuedvE10QuakeValue), |
|     (C++                          |                                   |
|                                   | [\[1\]](api/languages/cpp_api.htm |
|    function)](api/languages/cpp_a | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| pi.html#_CPPv4IDpEN5cudaq14kernel | -                                 |
| _builder8detectorEvDpRR8MeasArgs) |  [cudaq::QuakeValue::operator\[\] |
| -                                 |     (C++                          |
| [cudaq::kernel_builder::detectors |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     func                          | udaq10QuakeValueixEKNSt6size_tE), |
| tion)](api/languages/cpp_api.html |     [\[1\]](api/                  |
| #_CPPv4N5cudaq14kernel_builder9de | languages/cpp_api.html#_CPPv4N5cu |
| tectorsE10QuakeValue10QuakeValue) | daq10QuakeValueixERK10QuakeValue) |
| -   [cu                           | -                                 |
| daq::kernel_builder::getArguments |    [cudaq::QuakeValue::QuakeValue |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)](api/languag        |
| guages/cpp_api.html#_CPPv4N5cudaq | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| 14kernel_builder12getArgumentsEv) | akeValue10QuakeValueERN4mlir20Imp |
| -   [cu                           | licitLocOpBuilderEN4mlir5ValueE), |
| daq::kernel_builder::getNumParams |     [\[1\]                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/lan            | v4N5cudaq10QuakeValue10QuakeValue |
| guages/cpp_api.html#_CPPv4N5cudaq | ERN4mlir20ImplicitLocOpBuilderEd) |
| 14kernel_builder12getNumParamsEv) | -   [cudaq::QuakeValue::size (C++ |
| -   [c                            |     funct                         |
| udaq::kernel_builder::isArgStdVec | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|     function)](api/languages/cp   | -   [cudaq::QuakeValue::slice     |
| p_api.html#_CPPv4N5cudaq14kernel_ |     (C++                          |
| builder11isArgStdVecENSt6size_tE) |     function)](api/languages/cpp_ |
| -   [cuda                         | api.html#_CPPv4N5cudaq10QuakeValu |
| q::kernel_builder::kernel_builder | e5sliceEKNSt6size_tEKNSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_platform (C++ |
|     function)](api/languages/cpp  |     cl                            |
| _api.html#_CPPv4N5cudaq14kernel_b | ass)](api/languages/cpp_api.html# |
| uilder14kernel_builderERNSt6vecto | _CPPv4N5cudaq16quantum_platformE) |
| rIN6detail17KernelBuilderTypeEEE) | -   [cudaq:                       |
| -   [cudaq::k                     | :quantum_platform::beginExecution |
| ernel_builder::logical_observable |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)                     | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ](api/languages/cpp_api.html#_CPP | antum_platform14beginExecutionEv) |
| v4IDpEN5cudaq14kernel_builder18lo | -   [cudaq::quantum_pl            |
| gical_observableEvDpRR8MeasArgs), | atform::configureExecutionContext |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/lang           |
| cudaq14kernel_builder18logical_ob | uages/cpp_api.html#_CPPv4NK5cudaq |
| servableE10QuakeValueNSt6size_tE) | 16quantum_platform25configureExec |
| -   [cudaq::kernel_builder::name  | utionContextER16ExecutionContext) |
|     (C++                          | -   [cuda                         |
|     function)                     | q::quantum_platform::connectivity |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq14kernel_builder4nameEv) |     function)](api/langu          |
| -                                 | ages/cpp_api.html#_CPPv4N5cudaq16 |
|    [cudaq::kernel_builder::qalloc | quantum_platform12connectivityEv) |
|     (C++                          | -   [cuda                         |
|     function)](api/language       | q::quantum_platform::endExecution |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     (C++                          |
| nel_builder6qallocE10QuakeValue), |     function)](api/langu          |
|     [\[1\]](api/language          | ages/cpp_api.html#_CPPv4N5cudaq16 |
| s/cpp_api.html#_CPPv4N5cudaq14ker | quantum_platform12endExecutionEv) |
| nel_builder6qallocEKNSt6size_tE), | -   [cudaq::q                     |
|     [\[2                          | uantum_platform::enqueueAsyncTask |
| \]](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq14kernel_builder6qallo |     function)](api/languages/     |
| cERNSt6vectorINSt7complexIdEEEE), | cpp_api.html#_CPPv4N5cudaq16quant |
|     [\[3\]](                      | um_platform16enqueueAsyncTaskEKNS |
| api/languages/cpp_api.html#_CPPv4 | t6size_tER19KernelExecutionTask), |
| N5cudaq14kernel_builder6qallocEv) |     [\[1\]](api/languag           |
| -   [cudaq::kernel_builder::swap  | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform16enqueueAsyncTaskE |
|     function)](api/language       | KNSt6size_tERNSt8functionIFvvEEE) |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | -   [cudaq::quantum_p             |
| 4kernel_builder4swapEvRK10QuakeVa | latform::finalizeExecutionContext |
| lueRK10QuakeValueRK10QuakeValue), |     (C++                          |
|                                   |     function)](api/languages/c    |
| [\[1\]](api/languages/cpp_api.htm | pp_api.html#_CPPv4NK5cudaq16quant |
| l#_CPPv4I00EN5cudaq14kernel_build | um_platform24finalizeExecutionCon |
| er4swapEvRKNSt6vectorI10QuakeValu | textERN5cudaq16ExecutionContextE) |
| eEERK10QuakeValueRK10QuakeValue), | -   [cudaq::qua                   |
|                                   | ntum_platform::get_codegen_config |
| [\[2\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14kernel_builder4s |     function)](api/languages/c    |
| wapERK10QuakeValueRK10QuakeValue) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [cudaq::KernelExecutionTask   | m_platform18get_codegen_configEv) |
|     (C++                          | -   [cuda                         |
|     type                          | q::quantum_platform::get_exec_ctx |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq19KernelExecutionTaskE) |     function)](api/langua         |
| -   [cudaq::KernelThunkResultType | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|     (C++                          | quantum_platform12get_exec_ctxEv) |
|     struct)]                      | -   [c                            |
| (api/languages/cpp_api.html#_CPPv | udaq::quantum_platform::get_noise |
| 4N5cudaq21KernelThunkResultTypeE) |     (C++                          |
| -   [cudaq::KernelThunkType (C++  |     function)](api/languages/c    |
|                                   | pp_api.html#_CPPv4N5cudaq16quantu |
| type)](api/languages/cpp_api.html | m_platform9get_noiseENSt6size_tE) |
| #_CPPv4N5cudaq15KernelThunkTypeE) | -   [cudaq:                       |
| -   [cudaq::kraus_channel (C++    | :quantum_platform::get_num_qubits |
|                                   |     (C++                          |
|  class)](api/languages/cpp_api.ht |                                   |
| ml#_CPPv4N5cudaq13kraus_channelE) | function)](api/languages/cpp_api. |
| -   [cudaq::kraus_channel::empty  | html#_CPPv4NK5cudaq16quantum_plat |
|     (C++                          | form14get_num_qubitsENSt6size_tE) |
|     function)]                    | -   [cudaq::quantum_              |
| (api/languages/cpp_api.html#_CPPv | platform::get_remote_capabilities |
| 4NK5cudaq13kraus_channel5emptyEv) |     (C++                          |
| -   [cudaq::kraus_c               |     function)                     |
| hannel::generateUnitaryParameters | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq16quantum_platform23get |
|                                   | _remote_capabilitiesENSt6size_tE) |
|    function)](api/languages/cpp_a | -   [cudaq::qua                   |
| pi.html#_CPPv4N5cudaq13kraus_chan | ntum_platform::get_runtime_target |
| nel25generateUnitaryParametersEv) |     (C++                          |
| -                                 |     function)](api/languages/cp   |
|    [cudaq::kraus_channel::get_ops | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform18get_runtime_targetEv) |
|     function)](a                  | -   [cud                          |
| pi/languages/cpp_api.html#_CPPv4N | aq::quantum_platform::is_emulated |
| K5cudaq13kraus_channel7get_opsEv) |     (C++                          |
| -   [cud                          |                                   |
| aq::kraus_channel::identity_flags |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq16quantum_p |
|     member)](api/lan              | latform11is_emulatedENSt6size_tE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::                      |
| 13kraus_channel14identity_flagsE) | quantum_platform::is_library_mode |
| -   [cud                          |     (C++                          |
| aq::kraus_channel::is_identity_op |     function)](api/languages      |
|     (C++                          | /cpp_api.html#_CPPv4NK5cudaq16qua |
|                                   | ntum_platform15is_library_modeEv) |
|    function)](api/languages/cpp_a | -   [c                            |
| pi.html#_CPPv4NK5cudaq13kraus_cha | udaq::quantum_platform::is_remote |
| nnel14is_identity_opENSt6size_tE) |     (C++                          |
| -   [cudaq::                      |     function)](api/languages/cp   |
| kraus_channel::is_unitary_mixture | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform9is_remoteENSt6size_tE) |
|     function)](api/languages      | -   [cuda                         |
| /cpp_api.html#_CPPv4NK5cudaq13kra | q::quantum_platform::is_simulator |
| us_channel18is_unitary_mixtureEv) |     (C++                          |
| -   [cu                           |                                   |
| daq::kraus_channel::kraus_channel |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4NK5cudaq16quantum_pl |
|     function)](api/lang           | atform12is_simulatorENSt6size_tE) |
| uages/cpp_api.html#_CPPv4IDpEN5cu | -   [c                            |
| daq13kraus_channel13kraus_channel | udaq::quantum_platform::launchVQE |
| EDpRRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)](                   |
|  [\[1\]](api/languages/cpp_api.ht | api/languages/cpp_api.html#_CPPv4 |
| ml#_CPPv4N5cudaq13kraus_channel13 | N5cudaq16quantum_platform9launchV |
| kraus_channelERK13kraus_channel), | QEEKNSt6stringEPKvPN5cudaq8gradie |
|     [\[2\]                        | ntERKN5cudaq7spin_opERN5cudaq9opt |
| ](api/languages/cpp_api.html#_CPP | imizerEKiKNSt6size_tENSt6size_tE) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq:                       |
| hannelERKNSt6vectorI8kraus_opEE), | :quantum_platform::list_platforms |
|     [\[3\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languag        |
| v4N5cudaq13kraus_channel13kraus_c | es/cpp_api.html#_CPPv4N5cudaq16qu |
| hannelERRNSt6vectorI8kraus_opEE), | antum_platform14list_platformsEv) |
|     [\[4\]](api/lan               | -                                 |
| guages/cpp_api.html#_CPPv4N5cudaq |    [cudaq::quantum_platform::name |
| 13kraus_channel13kraus_channelEv) |     (C++                          |
| -                                 |     function)](a                  |
| [cudaq::kraus_channel::noise_type | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | K5cudaq16quantum_platform4nameEv) |
|     member)](api                  | -   [                             |
| /languages/cpp_api.html#_CPPv4N5c | cudaq::quantum_platform::num_qpus |
| udaq13kraus_channel10noise_typeE) |     (C++                          |
| -                                 |     function)](api/l              |
|   [cudaq::kraus_channel::op_names | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq16quantum_platform8num_qpusEv) |
|     member)](                     | -   [cudaq::                      |
| api/languages/cpp_api.html#_CPPv4 | quantum_platform::onRandomSeedSet |
| N5cudaq13kraus_channel8op_namesE) |     (C++                          |
| -                                 |                                   |
|  [cudaq::kraus_channel::operator= | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq16quantum_platf |
|     function)](api/langua         | orm15onRandomSeedSetENSt6size_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq13k | -   [cudaq:                       |
| raus_channelaSERK13kraus_channel) | :quantum_platform::reset_exec_ctx |
| -   [c                            |     (C++                          |
| udaq::kraus_channel::operator\[\] |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     function)](api/l              | antum_platform14reset_exec_ctxEv) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cud                          |
| aq13kraus_channelixEKNSt6size_tE) | aq::quantum_platform::reset_noise |
| -                                 |     (C++                          |
| [cudaq::kraus_channel::parameters |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq16quantum_p |
|     member)](api                  | latform11reset_noiseENSt6size_tE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cuda                         |
| udaq13kraus_channel10parametersE) | q::quantum_platform::set_exec_ctx |
| -   [cudaq::krau                  |     (C++                          |
| s_channel::populateDefaultOpNames |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/languages/cp   | _CPPv4N5cudaq16quantum_platform12 |
| p_api.html#_CPPv4N5cudaq13kraus_c | set_exec_ctxEP16ExecutionContext) |
| hannel22populateDefaultOpNamesEv) | -   [c                            |
| -   [cu                           | udaq::quantum_platform::set_noise |
| daq::kraus_channel::probabilities |     (C++                          |
|     (C++                          |     function                      |
|     member)](api/la               | )](api/languages/cpp_api.html#_CP |
| nguages/cpp_api.html#_CPPv4N5cuda | Pv4N5cudaq16quantum_platform9set_ |
| q13kraus_channel13probabilitiesE) | noiseEPK11noise_modelNSt6size_tE) |
| -                                 | -   [cudaq::quantum_platfor       |
|  [cudaq::kraus_channel::push_back | m::supports_explicit_measurements |
|     (C++                          |     (C++                          |
|     function)](api                |     function)](api/l              |
| /languages/cpp_api.html#_CPPv4N5c | anguages/cpp_api.html#_CPPv4NK5cu |
| udaq13kraus_channel9push_backE8kr | daq16quantum_platform30supports_e |
| aus_opNSt8optionalINSt6stringEEE) | xplicit_measurementsENSt6size_tE) |
| -   [cudaq::kraus_channel::size   | -   [cudaq::quantum_pla           |
|     (C++                          | tform::supports_task_distribution |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     fu                            |
| v4NK5cudaq13kraus_channel4sizeEv) | nction)](api/languages/cpp_api.ht |
| -   [                             | ml#_CPPv4NK5cudaq16quantum_platfo |
| cudaq::kraus_channel::unitary_ops | rm26supports_task_distributionEv) |
|     (C++                          | -   [cudaq::quantum               |
|     member)](api/                 | _platform::with_execution_context |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq13kraus_channel11unitary_opsE) |     function)                     |
| -   [cudaq::kraus_op (C++         | ](api/languages/cpp_api.html#_CPP |
|     struct)](api/languages/cpp_   | v4I0DpEN5cudaq16quantum_platform2 |
| api.html#_CPPv4N5cudaq8kraus_opE) | 2with_execution_contextEDaR16Exec |
| -   [cudaq::kraus_op::adjoint     | utionContextRR8CallableDpRR4Args) |
|     (C++                          | -   [cudaq::QuantumTask (C++      |
|     functi                        |     type)](api/languages/cpp_api. |
| on)](api/languages/cpp_api.html#_ | html#_CPPv4N5cudaq11QuantumTaskE) |
| CPPv4NK5cudaq8kraus_op7adjointEv) | -   [cudaq::qubit (C++            |
| -   [cudaq::kraus_op::data (C++   |     type)](api/languages/c        |
|                                   | pp_api.html#_CPPv4N5cudaq5qubitE) |
|  member)](api/languages/cpp_api.h | -   [cudaq::QubitConnectivity     |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     (C++                          |
| -   [cudaq::kraus_op::kraus_op    |     ty                            |
|     (C++                          | pe)](api/languages/cpp_api.html#_ |
|     func                          | CPPv4N5cudaq17QubitConnectivityE) |
| tion)](api/languages/cpp_api.html | -   [cudaq::QubitEdge (C++        |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |     type)](api/languages/cpp_a    |
| opERRNSt16initializer_listI1TEE), | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|                                   | -   [cudaq::qudit (C++            |
|  [\[1\]](api/languages/cpp_api.ht |     clas                          |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | s)](api/languages/cpp_api.html#_C |
| pENSt6vectorIN5cudaq7complexEEE), | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     [\[2\]](api/l                 | -   [cudaq::qudit::qudit (C++     |
| anguages/cpp_api.html#_CPPv4N5cud |                                   |
| aq8kraus_op8kraus_opERK8kraus_op) | function)](api/languages/cpp_api. |
| -   [cudaq::kraus_op::nCols (C++  | html#_CPPv4N5cudaq5qudit5quditEv) |
|                                   | -   [cudaq::qvector (C++          |
| member)](api/languages/cpp_api.ht |     class)                        |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus_op::nRows (C++  | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|                                   | -   [cudaq::qvector::back (C++    |
| member)](api/languages/cpp_api.ht |     function)](a                  |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::kraus_op::operator=   | 5cudaq7qvector4backENSt6size_tE), |
|     (C++                          |                                   |
|     function)                     |   [\[1\]](api/languages/cpp_api.h |
| ](api/languages/cpp_api.html#_CPP | tml#_CPPv4N5cudaq7qvector4backEv) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cudaq::qvector::begin (C++   |
| -   [cudaq::kraus_op::precision   |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     memb                          | ml#_CPPv4N5cudaq7qvector5beginEv) |
| er)](api/languages/cpp_api.html#_ | -   [cudaq::qvector::clear (C++   |
| CPPv4N5cudaq8kraus_op9precisionE) |     fu                            |
| -   [cudaq::KrausSelection (C++   | nction)](api/languages/cpp_api.ht |
|     s                             | ml#_CPPv4N5cudaq7qvector5clearEv) |
| truct)](api/languages/cpp_api.htm | -   [cudaq::qvector::end (C++     |
| l#_CPPv4N5cudaq14KrausSelectionE) |                                   |
| -   [cudaq:                       | function)](api/languages/cpp_api. |
| :KrausSelection::circuit_location | html#_CPPv4N5cudaq7qvector3endEv) |
|     (C++                          | -   [cudaq::qvector::front (C++   |
|     member)](api/langua           |     function)](ap                 |
| ges/cpp_api.html#_CPPv4N5cudaq14K | i/languages/cpp_api.html#_CPPv4N5 |
| rausSelection16circuit_locationE) | cudaq7qvector5frontENSt6size_tE), |
| -                                 |                                   |
|  [cudaq::KrausSelection::is_error |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     member)](a                    | -   [cudaq::qvector::operator=    |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14KrausSelection8is_errorE) |     functio                       |
| -   [cudaq::Kra                   | n)](api/languages/cpp_api.html#_C |
| usSelection::kraus_operator_index | PPv4N5cudaq7qvectoraSERK7qvector) |
|     (C++                          | -   [cudaq::qvector::operator\[\] |
|     member)](api/languages/       |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq14Kraus |     function)                     |
| Selection20kraus_operator_indexE) | ](api/languages/cpp_api.html#_CPP |
| -   [cuda                         | v4N5cudaq7qvectorixEKNSt6size_tE) |
| q::KrausSelection::KrausSelection | -   [cudaq::qvector::qvector (C++ |
|     (C++                          |     function)](api/               |
|     function)](a                  | languages/cpp_api.html#_CPPv4N5cu |
| pi/languages/cpp_api.html#_CPPv4N | daq7qvector7qvectorENSt6size_tE), |
| 5cudaq14KrausSelection14KrausSele |     [\[1\]](a                     |
| ctionENSt6size_tENSt6vectorINSt6s | pi/languages/cpp_api.html#_CPPv4N |
| ize_tEEENSt6stringENSt6size_tEb), | 5cudaq7qvector7qvectorERK5state), |
|     [\[1\]](api/langu             |     [\[2\]](api                   |
| ages/cpp_api.html#_CPPv4N5cudaq14 | /languages/cpp_api.html#_CPPv4N5c |
| KrausSelection14KrausSelectionEv) | udaq7qvector7qvectorERK7qvector), |
| -                                 |     [\[3\]](ap                    |
|   [cudaq::KrausSelection::op_name | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq7qvector7qvectorERR7qvector) |
|     member)](                     | -   [cudaq::qvector::size (C++    |
| api/languages/cpp_api.html#_CPPv4 |     fu                            |
| N5cudaq14KrausSelection7op_nameE) | nction)](api/languages/cpp_api.ht |
| -   [                             | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| cudaq::KrausSelection::operator== | -   [cudaq::qvector::slice (C++   |
|     (C++                          |     function)](api/language       |
|     function)](api/languages      | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | tor5sliceENSt6size_tENSt6size_tE) |
| usSelectioneqERK14KrausSelection) | -   [cudaq::qvector::value_type   |
| -                                 |     (C++                          |
|    [cudaq::KrausSelection::qubits |     typ                           |
|     (C++                          | e)](api/languages/cpp_api.html#_C |
|     member)]                      | PPv4N5cudaq7qvector10value_typeE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::qview (C++            |
| 4N5cudaq14KrausSelection6qubitsE) |     clas                          |
| -   [cudaq::KrausTrajectory (C++  | s)](api/languages/cpp_api.html#_C |
|     st                            | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::qview::back (C++      |
| #_CPPv4N5cudaq15KrausTrajectoryE) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|  [cudaq::KrausTrajectory::builder | v4N5cudaq5qview4backENSt6size_tE) |
|     (C++                          | -   [cudaq::qview::begin (C++     |
|     function)](ap                 |                                   |
| i/languages/cpp_api.html#_CPPv4N5 | function)](api/languages/cpp_api. |
| cudaq15KrausTrajectory7builderEv) | html#_CPPv4N5cudaq5qview5beginEv) |
| -   [cu                           | -   [cudaq::qview::end (C++       |
| daq::KrausTrajectory::countErrors |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)](api/lang           | i.html#_CPPv4N5cudaq5qview3endEv) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::qview::front (C++     |
| 15KrausTrajectory11countErrorsEv) |     function)](                   |
| -   [                             | api/languages/cpp_api.html#_CPPv4 |
| cudaq::KrausTrajectory::isOrdered | N5cudaq5qview5frontENSt6size_tE), |
|     (C++                          |                                   |
|     function)](api/l              |    [\[1\]](api/languages/cpp_api. |
| anguages/cpp_api.html#_CPPv4NK5cu | html#_CPPv4N5cudaq5qview5frontEv) |
| daq15KrausTrajectory9isOrderedEv) | -   [cudaq::qview::operator\[\]   |
| -   [cudaq::                      |     (C++                          |
| KrausTrajectory::kraus_selections |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](api/languag          | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | -   [cudaq::qview::qview (C++     |
| ausTrajectory16kraus_selectionsE) |     functio                       |
| -   [cudaq:                       | n)](api/languages/cpp_api.html#_C |
| :KrausTrajectory::KrausTrajectory | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     (C++                          |     [\[1                          |
|     function                      | \]](api/languages/cpp_api.html#_C |
| )](api/languages/cpp_api.html#_CP | PPv4N5cudaq5qview5qviewERK5qview) |
| Pv4N5cudaq15KrausTrajectory15Krau | -   [cudaq::qview::size (C++      |
| sTrajectoryENSt6size_tENSt6vector |                                   |
| I14KrausSelectionEEdNSt6size_tE), | function)](api/languages/cpp_api. |
|     [\[1\]](api/languag           | html#_CPPv4NK5cudaq5qview4sizeEv) |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | -   [cudaq::qview::slice (C++     |
| ausTrajectory15KrausTrajectoryEv) |     function)](api/langua         |
| -   [cudaq::Kr                    | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| ausTrajectory::measurement_counts | iew5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qview::value_type     |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq15Krau |     t                             |
| sTrajectory18measurement_countsE) | ype)](api/languages/cpp_api.html# |
| -   [cud                          | _CPPv4N5cudaq5qview10value_typeE) |
| aq::KrausTrajectory::multiplicity | -   [cudaq::range (C++            |
|     (C++                          |     fun                           |
|     member)](api/lan              | ction)](api/languages/cpp_api.htm |
| guages/cpp_api.html#_CPPv4N5cudaq | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| 15KrausTrajectory12multiplicityE) | orI11ElementTypeEE11ElementType), |
| -   [                             |     [\[1\]](api/languages/cpp_    |
| cudaq::KrausTrajectory::num_shots | api.html#_CPPv4I0EN5cudaq5rangeEN |
|     (C++                          | St6vectorI11ElementTypeEE11Elemen |
|     member)](api                  | tType11ElementType11ElementType), |
| /languages/cpp_api.html#_CPPv4N5c |     [                             |
| udaq15KrausTrajectory9num_shotsE) | \[2\]](api/languages/cpp_api.html |
| -   [c                            | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| udaq::KrausTrajectory::operator== | -   [cudaq::real (C++             |
|     (C++                          |     type)](api/languages/         |
|     function)](api/languages/c    | cpp_api.html#_CPPv4N5cudaq4realE) |
| pp_api.html#_CPPv4NK5cudaq15Kraus | -   [cudaq::registry (C++         |
| TrajectoryeqERK15KrausTrajectory) |     type)](api/languages/cpp_     |
| -   [cu                           | api.html#_CPPv4N5cudaq8registryE) |
| daq::KrausTrajectory::probability | -                                 |
|     (C++                          |  [cudaq::registry::RegisteredType |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     class)](api/                  |
| q15KrausTrajectory11probabilityE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cuda                         | 5cudaq8registry14RegisteredTypeE) |
| q::KrausTrajectory::trajectory_id | -   [cudaq::RemoteCapabilities    |
|     (C++                          |     (C++                          |
|     member)](api/lang             |     struc                         |
| uages/cpp_api.html#_CPPv4N5cudaq1 | t)](api/languages/cpp_api.html#_C |
| 5KrausTrajectory13trajectory_idE) | PPv4N5cudaq18RemoteCapabilitiesE) |
| -                                 | -   [cudaq::Remot                 |
|   [cudaq::KrausTrajectory::weight | eCapabilities::RemoteCapabilities |
|     (C++                          |     (C++                          |
|     member)](                     |     function)](api/languages/cpp  |
| api/languages/cpp_api.html#_CPPv4 | _api.html#_CPPv4N5cudaq18RemoteCa |
| N5cudaq15KrausTrajectory6weightE) | pabilities18RemoteCapabilitiesEb) |
| -                                 | -   [cudaq:                       |
|    [cudaq::KrausTrajectoryBuilder | :RemoteCapabilities::stateOverlap |
|     (C++                          |     (C++                          |
|     class)](                      |     member)](api/langua           |
| api/languages/cpp_api.html#_CPPv4 | ges/cpp_api.html#_CPPv4N5cudaq18R |
| N5cudaq22KrausTrajectoryBuilderE) | emoteCapabilities12stateOverlapE) |
| -   [cud                          | -                                 |
| aq::KrausTrajectoryBuilder::build |   [cudaq::RemoteCapabilities::vqe |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     member)](                     |
| uages/cpp_api.html#_CPPv4NK5cudaq | api/languages/cpp_api.html#_CPPv4 |
| 22KrausTrajectoryBuilder5buildEv) | N5cudaq18RemoteCapabilities3vqeE) |
| -   [cud                          | -   [cudaq::Resources (C++        |
| aq::KrausTrajectoryBuilder::setId |     class)](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     function)](api/languages/cpp  | -   [cudaq::run (C++              |
| _api.html#_CPPv4N5cudaq22KrausTra |     function)]                    |
| jectoryBuilder5setIdENSt6size_tE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::Kraus                 | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| TrajectoryBuilder::setProbability | 5invoke_result_tINSt7decay_tI13Qu |
|     (C++                          | antumKernelEEDpNSt7decay_tI4ARGSE |
|     function)](api/languages/cpp  | EEEEENSt6size_tERN5cudaq11noise_m |
| _api.html#_CPPv4N5cudaq22KrausTra | odelERR13QuantumKernelDpRR4ARGS), |
| jectoryBuilder14setProbabilityEd) |     [\[1\]](api/langu             |
| -   [cudaq::Krau                  | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| sTrajectoryBuilder::setSelections | daq3runENSt6vectorINSt15invoke_re |
|     (C++                          | sult_tINSt7decay_tI13QuantumKerne |
|     function)](api/languag        | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | ize_tERR13QuantumKernelDpRR4ARGS) |
| ausTrajectoryBuilder13setSelectio | -   [cudaq::run_async (C++        |
| nsENSt6vectorI14KrausSelectionEE) |     functio                       |
| -   [cudaq::logical_observable    | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     function)](api/languages/c    | tureINSt6vectorINSt15invoke_resul |
| pp_api.html#_CPPv4IDpEN5cudaq18lo | t_tINSt7decay_tI13QuantumKernelEE |
| gical_observableEvDpRR8MeasArgs), | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|     [\[1\]](api/l                 | ze_tENSt6size_tERN5cudaq11noise_m |
| anguages/cpp_api.html#_CPPv4N5cud | odelERR13QuantumKernelDpRR4ARGS), |
| aq18logical_observableERKNSt6vect |     [\[1\]](api/la                |
| orI14measure_resultEENSt6size_tE) | nguages/cpp_api.html#_CPPv4I0DpEN |
| -   [cudaq::matrix_callback (C++  | 5cudaq9run_asyncENSt6futureINSt6v |
|     c                             | ectorINSt15invoke_result_tINSt7de |
| lass)](api/languages/cpp_api.html | cay_tI13QuantumKernelEEDpNSt7deca |
| #_CPPv4N5cudaq15matrix_callbackE) | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| -   [cudaq::matrix_handler (C++   | ize_tERR13QuantumKernelDpRR4ARGS) |
|                                   | -   [cudaq::RuntimeTarget (C++    |
| class)](api/languages/cpp_api.htm |                                   |
| l#_CPPv4N5cudaq14matrix_handlerE) | struct)](api/languages/cpp_api.ht |
| -   [cudaq::mat                   | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| rix_handler::commutation_behavior | -   [cudaq::sample (C++           |
|     (C++                          |     function)](api/languages/c    |
|     struct)](api/languages/       | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| cpp_api.html#_CPPv4N5cudaq14matri | mpleE13sample_resultRK14sample_op |
| x_handler20commutation_behaviorE) | tionsRR13QuantumKernelDpRR4Args), |
| -                                 |     [\[1\                         |
|    [cudaq::matrix_handler::define | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     function)](a                  | esultRR13QuantumKernelDpRR4Args), |
| pi/languages/cpp_api.html#_CPPv4N |     [\                            |
| 5cudaq14matrix_handler6defineENSt | [2\]](api/languages/cpp_api.html# |
| 6stringENSt6vectorINSt7int64_tEEE | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| RR15matrix_callbackRKNSt13unorder | ize_tERR13QuantumKernelDpRR4Args) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::sample_options (C++   |
|                                   |     s                             |
| [\[1\]](api/languages/cpp_api.htm | truct)](api/languages/cpp_api.htm |
| l#_CPPv4N5cudaq14matrix_handler6d | l#_CPPv4N5cudaq14sample_optionsE) |
| efineENSt6stringENSt6vectorINSt7i | -   [cudaq::sample_result (C++    |
| nt64_tEEERR15matrix_callbackRR20d |                                   |
| iag_matrix_callbackRKNSt13unorder |  class)](api/languages/cpp_api.ht |
| ed_mapINSt6stringENSt6stringEEE), | ml#_CPPv4N5cudaq13sample_resultE) |
|     [\[2\]](                      | -   [cudaq::sample_result::append |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14matrix_handler6defineENS |     function)](api/languages/cpp_ |
| t6stringENSt6vectorINSt7int64_tEE | api.html#_CPPv4N5cudaq13sample_re |
| ERR15matrix_callbackRRNSt13unorde | sult6appendERK15ExecutionResultb) |
| red_mapINSt6stringENSt6stringEEE) | -   [cudaq::sample_result::begin  |
| -                                 |     (C++                          |
|   [cudaq::matrix_handler::degrees |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](ap                 | 4N5cudaq13sample_result5beginEv), |
| i/languages/cpp_api.html#_CPPv4NK |     [\[1\]]                       |
| 5cudaq14matrix_handler7degreesEv) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NK5cudaq13sample_result5beginEv) |
|  [cudaq::matrix_handler::displace | -   [cudaq::sample_result::cbegin |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](                   |
| s/cpp_api.html#_CPPv4N5cudaq14mat | api/languages/cpp_api.html#_CPPv4 |
| rix_handler8displaceENSt6size_tE) | NK5cudaq13sample_result6cbeginEv) |
| -   [cudaq::matrix                | -   [cudaq::sample_result::cend   |
| _handler::get_expected_dimensions |     (C++                          |
|     (C++                          |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
|    function)](api/languages/cpp_a | v4NK5cudaq13sample_result4cendEv) |
| pi.html#_CPPv4NK5cudaq14matrix_ha | -   [cudaq::sample_result::clear  |
| ndler23get_expected_dimensionsEv) |     (C++                          |
| -   [cudaq::matrix_ha             |     function)                     |
| ndler::get_parameter_descriptions | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq13sample_result5clearEv) |
|                                   | -   [cudaq::sample_result::count  |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4NK5cudaq14matrix_handl |     function)](                   |
| er26get_parameter_descriptionsEv) | api/languages/cpp_api.html#_CPPv4 |
| -   [c                            | NK5cudaq13sample_result5countENSt |
| udaq::matrix_handler::instantiate | 11string_viewEKNSt11string_viewE) |
|     (C++                          | -   [                             |
|     function)](a                  | cudaq::sample_result::deserialize |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14matrix_handler11instantia |     functio                       |
| teENSt6stringERKNSt6vectorINSt6si | n)](api/languages/cpp_api.html#_C |
| ze_tEEERK20commutation_behavior), | PPv4N5cudaq13sample_result11deser |
|     [\[1\]](                      | ializeERNSt6vectorINSt6size_tEEE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::sample_result::dump   |
| N5cudaq14matrix_handler11instanti |     (C++                          |
| ateENSt6stringERRNSt6vectorINSt6s |     function)](api/languag        |
| ize_tEEERK20commutation_behavior) | es/cpp_api.html#_CPPv4NK5cudaq13s |
| -   [cuda                         | ample_result4dumpERNSt7ostreamE), |
| q::matrix_handler::matrix_handler |     [\[1\]                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/languag        | v4NK5cudaq13sample_result4dumpEv) |
| es/cpp_api.html#_CPPv4I0_NSt11ena | -   [cudaq::sample_result::end    |
| ble_if_tINSt12is_base_of_vI16oper |     (C++                          |
| ator_handler1TEEbEEEN5cudaq14matr |     function                      |
| ix_handler14matrix_handlerERK1T), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]](ap                    | Pv4N5cudaq13sample_result3endEv), |
| i/languages/cpp_api.html#_CPPv4I0 |     [\[1\                         |
| _NSt11enable_if_tINSt12is_base_of | ]](api/languages/cpp_api.html#_CP |
| _vI16operator_handler1TEEbEEEN5cu | Pv4NK5cudaq13sample_result3endEv) |
| daq14matrix_handler14matrix_handl | -   [                             |
| erERK1TRK20commutation_behavior), | cudaq::sample_result::expectation |
|     [\[2\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4N5cudaq14matrix_hand |     f                             |
| ler14matrix_handlerENSt6size_tE), | unction)](api/languages/cpp_api.h |
|     [\[3\]](api/                  | tml#_CPPv4NK5cudaq13sample_result |
| languages/cpp_api.html#_CPPv4N5cu | 11expectationEKNSt11string_viewE) |
| daq14matrix_handler14matrix_handl | -   [c                            |
| erENSt6stringERKNSt6vectorINSt6si | udaq::sample_result::get_marginal |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[4\]](api/                  |     function)](api/languages/cpp_ |
| languages/cpp_api.html#_CPPv4N5cu | api.html#_CPPv4NK5cudaq13sample_r |
| daq14matrix_handler14matrix_handl | esult12get_marginalERKNSt6vectorI |
| erENSt6stringERRNSt6vectorINSt6si | NSt6size_tEEEKNSt11string_viewE), |
| ze_tEEERK20commutation_behavior), |     [\[1\]](api/languages/cpp_    |
|     [\                            | api.html#_CPPv4NK5cudaq13sample_r |
| [5\]](api/languages/cpp_api.html# | esult12get_marginalERRKNSt6vector |
| _CPPv4N5cudaq14matrix_handler14ma | INSt6size_tEEEKNSt11string_viewE) |
| trix_handlerERK14matrix_handler), | -   [cuda                         |
|     [                             | q::sample_result::get_total_shots |
| \[6\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq14matrix_handler14m |     function)](api/langua         |
| atrix_handlerERR14matrix_handler) | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| -                                 | sample_result15get_total_shotsEv) |
|  [cudaq::matrix_handler::momentum | -   [cuda                         |
|     (C++                          | q::sample_result::has_even_parity |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     fun                           |
| rix_handler8momentumENSt6size_tE) | ction)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq13sample_result15h |
|    [cudaq::matrix_handler::number | as_even_parityENSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|     function)](api/langua         | q::sample_result::has_expectation |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6numberENSt6size_tE) |     funct                         |
| -                                 | ion)](api/languages/cpp_api.html# |
| [cudaq::matrix_handler::operator= | _CPPv4NK5cudaq13sample_result15ha |
|     (C++                          | s_expectationEKNSt11string_viewE) |
|     fun                           | -   [cu                           |
| ction)](api/languages/cpp_api.htm | daq::sample_result::most_probable |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     (C++                          |
| NSt7is_sameI1T14matrix_handlerE5v |     fun                           |
| alueENSt12is_base_of_vI16operator | ction)](api/languages/cpp_api.htm |
| _handler1TEEEbEEEN5cudaq14matrix_ | l#_CPPv4NK5cudaq13sample_result13 |
| handleraSER14matrix_handlerRK1T), | most_probableEKNSt11string_viewE) |
|     [\[1\]](api/languages         | -                                 |
| /cpp_api.html#_CPPv4N5cudaq14matr | [cudaq::sample_result::operator+= |
| ix_handleraSERK14matrix_handler), |     (C++                          |
|     [\[2\]](api/language          |     function)](api/langua         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ges/cpp_api.html#_CPPv4N5cudaq13s |
| rix_handleraSERR14matrix_handler) | ample_resultpLERK13sample_result) |
| -   [                             | -                                 |
| cudaq::matrix_handler::operator== |  [cudaq::sample_result::operator= |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/langua         |
| /cpp_api.html#_CPPv4NK5cudaq14mat | ges/cpp_api.html#_CPPv4N5cudaq13s |
| rix_handlereqERK14matrix_handler) | ample_resultaSERR13sample_result) |
| -                                 | -                                 |
|    [cudaq::matrix_handler::parity | [cudaq::sample_result::operator== |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/languag        |
| ges/cpp_api.html#_CPPv4N5cudaq14m | es/cpp_api.html#_CPPv4NK5cudaq13s |
| atrix_handler6parityENSt6size_tE) | ample_resulteqERK13sample_result) |
| -                                 | -   [                             |
|  [cudaq::matrix_handler::position | cudaq::sample_result::probability |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api/lan            |
| s/cpp_api.html#_CPPv4N5cudaq14mat | guages/cpp_api.html#_CPPv4NK5cuda |
| rix_handler8positionENSt6size_tE) | q13sample_result11probabilityENSt |
| -   [cudaq::                      | 11string_viewEKNSt11string_viewE) |
| matrix_handler::remove_definition | -   [cud                          |
|     (C++                          | aq::sample_result::register_names |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/langu          |
| ml#_CPPv4N5cudaq14matrix_handler1 | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| 7remove_definitionERKNSt6stringE) | 3sample_result14register_namesEv) |
| -                                 | -                                 |
|   [cudaq::matrix_handler::squeeze |    [cudaq::sample_result::reorder |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)](api/langua         |
| es/cpp_api.html#_CPPv4N5cudaq14ma | ges/cpp_api.html#_CPPv4N5cudaq13s |
| trix_handler7squeezeENSt6size_tE) | ample_result7reorderERKNSt6vector |
| -   [cudaq::m                     | INSt6size_tEEEKNSt11string_viewE) |
| atrix_handler::to_diagonal_matrix | -   [cu                           |
|     (C++                          | daq::sample_result::sample_result |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     func                          |
| 14matrix_handler18to_diagonal_mat | tion)](api/languages/cpp_api.html |
| rixERNSt13unordered_mapINSt6size_ | #_CPPv4N5cudaq13sample_result13sa |
| tENSt7int64_tEEERKNSt13unordered_ | mple_resultERK15ExecutionResult), |
| mapINSt6stringENSt7complexIdEEEE) |     [\[1\]](api/la                |
| -                                 | nguages/cpp_api.html#_CPPv4N5cuda |
| [cudaq::matrix_handler::to_matrix | q13sample_result13sample_resultER |
|     (C++                          | KNSt6vectorI15ExecutionResultEE), |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP |  [\[2\]](api/languages/cpp_api.ht |
| v4NK5cudaq14matrix_handler9to_mat | ml#_CPPv4N5cudaq13sample_result13 |
| rixERNSt13unordered_mapINSt6size_ | sample_resultERR13sample_result), |
| tENSt7int64_tEEERKNSt13unordered_ |     [                             |
| mapINSt6stringENSt7complexIdEEEE) | \[3\]](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq13sample_result13sa |
| [cudaq::matrix_handler::to_string | mple_resultERR15ExecutionResult), |
|     (C++                          |     [\[4\]](api/lan               |
|     function)](api/               | guages/cpp_api.html#_CPPv4N5cudaq |
| languages/cpp_api.html#_CPPv4NK5c | 13sample_result13sample_resultEdR |
| udaq14matrix_handler9to_stringEb) | KNSt6vectorI15ExecutionResultEE), |
| -                                 |     [\[5\]](api/lan               |
| [cudaq::matrix_handler::unique_id | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 13sample_result13sample_resultEv) |
|     function)](api/               | -                                 |
| languages/cpp_api.html#_CPPv4NK5c |  [cudaq::sample_result::serialize |
| udaq14matrix_handler9unique_idEv) |     (C++                          |
| -   [cudaq:                       |     function)](api                |
| :matrix_handler::\~matrix_handler | /languages/cpp_api.html#_CPPv4NK5 |
|     (C++                          | cudaq13sample_result9serializeEv) |
|     functi                        | -   [cudaq::sample_result::size   |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     function)](api/languages/c    |
| -   [cudaq::matrix_op (C++        | pp_api.html#_CPPv4NK5cudaq13sampl |
|     type)](api/languages/cpp_a    | e_result4sizeEKNSt11string_viewE) |
| pi.html#_CPPv4N5cudaq9matrix_opE) | -   [cudaq::sample_result::to_map |
| -   [cudaq::matrix_op_term (C++   |     (C++                          |
|                                   |     function)](api/languages/cpp  |
|  type)](api/languages/cpp_api.htm | _api.html#_CPPv4NK5cudaq13sample_ |
| l#_CPPv4N5cudaq14matrix_op_termE) | result6to_mapEKNSt11string_viewE) |
| -                                 | -   [cuda                         |
|    [cudaq::mdiag_operator_handler | q::sample_result::\~sample_result |
|     (C++                          |     (C++                          |
|     class)](                      |     funct                         |
| api/languages/cpp_api.html#_CPPv4 | ion)](api/languages/cpp_api.html# |
| N5cudaq22mdiag_operator_handlerE) | _CPPv4N5cudaq13sample_resultD0Ev) |
| -   [cudaq::measure_handle (C++   | -   [cudaq::scalar_callback (C++  |
|                                   |     c                             |
| class)](api/languages/cpp_api.htm | lass)](api/languages/cpp_api.html |
| l#_CPPv4N5cudaq14measure_handleE) | #_CPPv4N5cudaq15scalar_callbackE) |
| -   [cudaq::measure_result (C++   | -   [c                            |
|                                   | udaq::scalar_callback::operator() |
|  type)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14measure_resultE) |     function)](api/language       |
| -   [cudaq::mpi (C++              | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|     type)](api/languages          | alar_callbackclERKNSt13unordered_ |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::mpi::all_gather (C++  | -   [                             |
|     fu                            | cudaq::scalar_callback::operator= |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     function)](api/languages/c    |
| RNSt6vectorIdEERKNSt6vectorIdEE), | pp_api.html#_CPPv4N5cudaq15scalar |
|                                   | _callbackaSERK15scalar_callback), |
|   [\[1\]](api/languages/cpp_api.h |     [\[1\]](api/languages/        |
| tml#_CPPv4N5cudaq3mpi10all_gather | cpp_api.html#_CPPv4N5cudaq15scala |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | r_callbackaSERR15scalar_callback) |
| -   [cudaq::mpi::all_reduce (C++  | -   [cudaq:                       |
|                                   | :scalar_callback::scalar_callback |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     function)](api/languag        |
| reduceE1TRK1TRK14BinaryFunction), | es/cpp_api.html#_CPPv4I0_NSt11ena |
|     [\[1\]](api/langu             | ble_if_tINSt16is_invocable_r_vINS |
| ages/cpp_api.html#_CPPv4I00EN5cud | t7complexIdEE8CallableRKNSt13unor |
| aq3mpi10all_reduceE1TRK1TRK4Func) | dered_mapINSt6stringENSt7complexI |
| -   [cudaq::mpi::broadcast (C++   | dEEEEEEbEEEN5cudaq15scalar_callba |
|     function)](api/               | ck15scalar_callbackERR8Callable), |
| languages/cpp_api.html#_CPPv4N5cu |     [\[1\                         |
| daq3mpi9broadcastERNSt6stringEi), | ]](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/la                | Pv4N5cudaq15scalar_callback15scal |
| nguages/cpp_api.html#_CPPv4N5cuda | ar_callbackERK15scalar_callback), |
| q3mpi9broadcastERNSt6vectorIdEEi) |     [\[2                          |
| -   [cudaq::mpi::finalize (C++    | \]](api/languages/cpp_api.html#_C |
|     f                             | PPv4N5cudaq15scalar_callback15sca |
| unction)](api/languages/cpp_api.h | lar_callbackERR15scalar_callback) |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | -   [cudaq::scalar_operator (C++  |
| -   [cudaq::mpi::initialize (C++  |     c                             |
|     function                      | lass)](api/languages/cpp_api.html |
| )](api/languages/cpp_api.html#_CP | #_CPPv4N5cudaq15scalar_operatorE) |
| Pv4N5cudaq3mpi10initializeEiPPc), | -                                 |
|     [                             | [cudaq::scalar_operator::evaluate |
| \[1\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq3mpi10initializeEv) |                                   |
| -   [cudaq::mpi::is_initialized   |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq15scalar_op |
|     function                      | erator8evaluateERKNSt13unordered_ |
| )](api/languages/cpp_api.html#_CP | mapINSt6stringENSt7complexIdEEEE) |
| Pv4N5cudaq3mpi14is_initializedEv) | -   [cudaq::scalar_ope            |
| -   [cudaq::mpi::num_ranks (C++   | rator::get_parameter_descriptions |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     f                             |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | unction)](api/languages/cpp_api.h |
| -   [cudaq::mpi::rank (C++        | tml#_CPPv4NK5cudaq15scalar_operat |
|                                   | or26get_parameter_descriptionsEv) |
|    function)](api/languages/cpp_a | -   [cu                           |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | daq::scalar_operator::is_constant |
| -   [cudaq::noise_model (C++      |     (C++                          |
|                                   |     function)](api/lang           |
|    class)](api/languages/cpp_api. | uages/cpp_api.html#_CPPv4NK5cudaq |
| html#_CPPv4N5cudaq11noise_modelE) | 15scalar_operator11is_constantEv) |
| -   [cudaq::n                     | -   [c                            |
| oise_model::add_all_qubit_channel | udaq::scalar_operator::operator\* |
|     (C++                          |     (C++                          |
|     function)](api                |     function                      |
| /languages/cpp_api.html#_CPPv4IDp | )](api/languages/cpp_api.html#_CP |
| EN5cudaq11noise_model21add_all_qu | Pv4N5cudaq15scalar_operatormlENSt |
| bit_channelEvRK13kraus_channeli), | 7complexIdEERK15scalar_operator), |
|     [\[1\]](api/langua            |     [\[1\                         |
| ges/cpp_api.html#_CPPv4N5cudaq11n | ]](api/languages/cpp_api.html#_CP |
| oise_model21add_all_qubit_channel | Pv4N5cudaq15scalar_operatormlENSt |
| ERKNSt6stringERK13kraus_channeli) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
|  [cudaq::noise_model::add_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormlEdRK15scalar_operator), |
|     funct                         |     [\[3\]](api/languages/cp      |
| ion)](api/languages/cpp_api.html# | p_api.html#_CPPv4N5cudaq15scalar_ |
| _CPPv4IDpEN5cudaq11noise_model11a | operatormlEdRR15scalar_operator), |
| dd_channelEvRK15PredicateFuncTy), |     [\[4\]](api/languages         |
|     [\[1\]](api/languages/cpp_    | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| api.html#_CPPv4IDpEN5cudaq11noise | alar_operatormlENSt7complexIdEE), |
| _model11add_channelEvRKNSt6vector |     [\[5\]](api/languages/cpp     |
| INSt6size_tEEERK13kraus_channel), | _api.html#_CPPv4NKR5cudaq15scalar |
|     [\[2\]](ap                    | _operatormlERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[6\]]                       |
| cudaq11noise_model11add_channelER | (api/languages/cpp_api.html#_CPPv |
| KNSt6stringERK15PredicateFuncTy), | 4NKR5cudaq15scalar_operatormlEd), |
|                                   |     [\[7\]](api/language          |
| [\[3\]](api/languages/cpp_api.htm | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| l#_CPPv4N5cudaq11noise_model11add | alar_operatormlENSt7complexIdEE), |
| _channelERKNSt6stringERKNSt6vecto |     [\[8\]](api/languages/cp      |
| rINSt6size_tEEERK13kraus_channel) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::noise_model::empty    | _operatormlERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     function                      | ]](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4NO5cudaq15scalar_operatormlEd) |
| Pv4NK5cudaq11noise_model5emptyEv) | -   [cu                           |
| -                                 | daq::scalar_operator::operator\*= |
| [cudaq::noise_model::get_channels |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/l              | es/cpp_api.html#_CPPv4N5cudaq15sc |
| anguages/cpp_api.html#_CPPv4I0ENK | alar_operatormLENSt7complexIdEE), |
| 5cudaq11noise_model12get_channels |     [\[1\]](api/languages/c       |
| ENSt6vectorI13kraus_channelEERKNS | pp_api.html#_CPPv4N5cudaq15scalar |
| t6vectorINSt6size_tEEERKNSt6vecto | _operatormLERK15scalar_operator), |
| rINSt6size_tEEERKNSt6vectorIdEE), |     [\[2                          |
|     [\[1\]](api/languages/cpp_a   | \]](api/languages/cpp_api.html#_C |
| pi.html#_CPPv4NK5cudaq11noise_mod | PPv4N5cudaq15scalar_operatormLEd) |
| el12get_channelsERKNSt6stringERKN | -   [                             |
| St6vectorINSt6size_tEEERKNSt6vect | cudaq::scalar_operator::operator+ |
| orINSt6size_tEEERKNSt6vectorIdEE) |     (C++                          |
| -                                 |     function                      |
|  [cudaq::noise_model::noise_model | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatorplENSt |
|     function)](api                | 7complexIdEERK15scalar_operator), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[1\                         |
| udaq11noise_model11noise_modelEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cu                           | Pv4N5cudaq15scalar_operatorplENSt |
| daq::noise_model::PredicateFuncTy | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     type)](api/la                 | p_api.html#_CPPv4N5cudaq15scalar_ |
| nguages/cpp_api.html#_CPPv4N5cuda | operatorplEdRK15scalar_operator), |
| q11noise_model15PredicateFuncTyE) |     [\[3\]](api/languages/cp      |
| -   [cud                          | p_api.html#_CPPv4N5cudaq15scalar_ |
| aq::noise_model::register_channel | operatorplEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     function)](api/languages      | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | alar_operatorplENSt7complexIdEE), |
| noise_model16register_channelEvv) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::                      | _api.html#_CPPv4NKR5cudaq15scalar |
| noise_model::requires_constructor | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     type)](api/languages/cp       | (api/languages/cpp_api.html#_CPPv |
| p_api.html#_CPPv4I0DpEN5cudaq11no | 4NKR5cudaq15scalar_operatorplEd), |
| ise_model20requires_constructorE) |     [\[7\]]                       |
| -   [cudaq::noise_model_type (C++ | (api/languages/cpp_api.html#_CPPv |
|     e                             | 4NKR5cudaq15scalar_operatorplEv), |
| num)](api/languages/cpp_api.html# |     [\[8\]](api/language          |
| _CPPv4N5cudaq16noise_model_typeE) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::no                    | alar_operatorplENSt7complexIdEE), |
| ise_model_type::amplitude_damping |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     enumerator)](api/languages    | _operatorplERK15scalar_operator), |
| /cpp_api.html#_CPPv4N5cudaq16nois |     [\[10\]                       |
| e_model_type17amplitude_dampingE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::noise_mode            | v4NO5cudaq15scalar_operatorplEd), |
| l_type::amplitude_damping_channel |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     e                             | Pv4NO5cudaq15scalar_operatorplEv) |
| numerator)](api/languages/cpp_api | -   [c                            |
| .html#_CPPv4N5cudaq16noise_model_ | udaq::scalar_operator::operator+= |
| type25amplitude_damping_channelE) |     (C++                          |
| -   [cudaq::n                     |     function)](api/languag        |
| oise_model_type::bit_flip_channel | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatorpLENSt7complexIdEE), |
|     enumerator)](api/language     |     [\[1\]](api/languages/c       |
| s/cpp_api.html#_CPPv4N5cudaq16noi | pp_api.html#_CPPv4N5cudaq15scalar |
| se_model_type16bit_flip_channelE) | _operatorpLERK15scalar_operator), |
| -   [cudaq::                      |     [\[2                          |
| noise_model_type::depolarization1 | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatorpLEd) |
|     enumerator)](api/languag      | -   [                             |
| es/cpp_api.html#_CPPv4N5cudaq16no | cudaq::scalar_operator::operator- |
| ise_model_type15depolarization1E) |     (C++                          |
| -   [cudaq::                      |     function                      |
| noise_model_type::depolarization2 | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormiENSt |
|     enumerator)](api/languag      | 7complexIdEERK15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[1\                         |
| ise_model_type15depolarization2E) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise_m               | Pv4N5cudaq15scalar_operatormiENSt |
| odel_type::depolarization_channel | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|   enumerator)](api/languages/cpp_ | operatormiEdRK15scalar_operator), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[3\]](api/languages/cp      |
| el_type22depolarization_channelE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatormiEdRR15scalar_operator), |
|  [cudaq::noise_model_type::pauli1 |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     enumerator)](a                | alar_operatormiENSt7complexIdEE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[5\]](api/languages/cpp     |
| 5cudaq16noise_model_type6pauli1E) | _api.html#_CPPv4NKR5cudaq15scalar |
| -                                 | _operatormiERK15scalar_operator), |
|  [cudaq::noise_model_type::pauli2 |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     enumerator)](a                | 4NKR5cudaq15scalar_operatormiEd), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[7\]]                       |
| 5cudaq16noise_model_type6pauli2E) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq                        | 4NKR5cudaq15scalar_operatormiEv), |
| ::noise_model_type::phase_damping |     [\[8\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     enumerator)](api/langu        | alar_operatormiENSt7complexIdEE), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     [\[9\]](api/languages/cp      |
| noise_model_type13phase_dampingE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::noi                   | _operatormiERK15scalar_operator), |
| se_model_type::phase_flip_channel |     [\[10\]                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     enumerator)](api/languages/   | v4NO5cudaq15scalar_operatormiEd), |
| cpp_api.html#_CPPv4N5cudaq16noise |     [\[11\                        |
| _model_type18phase_flip_channelE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4NO5cudaq15scalar_operatormiEv) |
| [cudaq::noise_model_type::unknown | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator-= |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languag        |
| cudaq16noise_model_type7unknownE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -                                 | alar_operatormIENSt7complexIdEE), |
| [cudaq::noise_model_type::x_error |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](ap               | _operatormIERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[2                          |
| cudaq16noise_model_type7x_errorE) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq15scalar_operatormIEd) |
| [cudaq::noise_model_type::y_error | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator/ |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function                      |
| cudaq16noise_model_type7y_errorE) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatordvENSt |
| [cudaq::noise_model_type::z_error | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](ap               | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4N5cudaq15scalar_operatordvENSt |
| cudaq16noise_model_type7z_errorE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::num_available_gpus    |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     function                      | operatordvEdRK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[3\]](api/languages/cp      |
| Pv4N5cudaq18num_available_gpusEv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::observe (C++          | operatordvEdRR15scalar_operator), |
|     function)]                    |     [\[4\]](api/languages         |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| 4I00DpEN5cudaq7observeENSt6vector | alar_operatordvENSt7complexIdEE), |
| I14observe_resultEERR13QuantumKer |     [\[5\]](api/languages/cpp     |
| nelRK15SpinOpContainerDpRR4Args), | _api.html#_CPPv4NKR5cudaq15scalar |
|     [\[1\]](api/languages/cpp_ap  | _operatordvERK15scalar_operator), |
| i.html#_CPPv4I0DpEN5cudaq7observe |     [\[6\]]                       |
| E14observe_resultNSt6size_tERR13Q | (api/languages/cpp_api.html#_CPPv |
| uantumKernelRK7spin_opDpRR4Args), | 4NKR5cudaq15scalar_operatordvEd), |
|     [\[                           |     [\[7\]](api/language          |
| 2\]](api/languages/cpp_api.html#_ | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| CPPv4I0DpEN5cudaq7observeE14obser | alar_operatordvENSt7complexIdEE), |
| ve_resultRK15observe_optionsRR13Q |     [\[8\]](api/languages/cp      |
| uantumKernelRK7spin_opDpRR4Args), | p_api.html#_CPPv4NO5cudaq15scalar |
|     [\[3\]](api/lang              | _operatordvERK15scalar_operator), |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     [\[9\                         |
| udaq7observeE14observe_resultRR13 | ]](api/languages/cpp_api.html#_CP |
| QuantumKernelRK7spin_opDpRR4Args) | Pv4NO5cudaq15scalar_operatordvEd) |
| -   [cudaq::observe_options (C++  | -   [c                            |
|     st                            | udaq::scalar_operator::operator/= |
| ruct)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15observe_optionsE) |     function)](api/languag        |
| -   [cudaq::observe_result (C++   | es/cpp_api.html#_CPPv4N5cudaq15sc |
|                                   | alar_operatordVENSt7complexIdEE), |
| class)](api/languages/cpp_api.htm |     [\[1\]](api/languages/c       |
| l#_CPPv4N5cudaq14observe_resultE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _operatordVERK15scalar_operator), |
|    [cudaq::observe_result::counts |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     function)](api/languages/c    | PPv4N5cudaq15scalar_operatordVEd) |
| pp_api.html#_CPPv4N5cudaq14observ | -   [                             |
| e_result6countsERK12spin_op_term) | cudaq::scalar_operator::operator= |
| -   [cudaq::observe_result::dump  |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)                     | pp_api.html#_CPPv4N5cudaq15scalar |
| ](api/languages/cpp_api.html#_CPP | _operatoraSERK15scalar_operator), |
| v4N5cudaq14observe_result4dumpEv) |     [\[1\]](api/languages/        |
| -   [c                            | cpp_api.html#_CPPv4N5cudaq15scala |
| udaq::observe_result::expectation | r_operatoraSERR15scalar_operator) |
|     (C++                          | -   [c                            |
|                                   | udaq::scalar_operator::operator== |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq14observe_resul |     function)](api/languages/c    |
| t11expectationERK12spin_op_term), | pp_api.html#_CPPv4NK5cudaq15scala |
|     [\[1\]](api/la                | r_operatoreqERK15scalar_operator) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq:                       |
| q14observe_result11expectationEv) | :scalar_operator::scalar_operator |
| -   [cuda                         |     (C++                          |
| q::observe_result::id_coefficient |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     function)](api/langu          | #_CPPv4N5cudaq15scalar_operator15 |
| ages/cpp_api.html#_CPPv4N5cudaq14 | scalar_operatorENSt7complexIdEE), |
| observe_result14id_coefficientEv) |     [\[1\]](api/langu             |
| -   [cuda                         | ages/cpp_api.html#_CPPv4N5cudaq15 |
| q::observe_result::observe_result | scalar_operator15scalar_operatorE |
|     (C++                          | RK15scalar_callbackRRNSt13unorder |
|                                   | ed_mapINSt6stringENSt6stringEEE), |
|   function)](api/languages/cpp_ap |     [\[2\                         |
| i.html#_CPPv4N5cudaq14observe_res | ]](api/languages/cpp_api.html#_CP |
| ult14observe_resultEdRK7spin_op), | Pv4N5cudaq15scalar_operator15scal |
|     [\[1\]](a                     | ar_operatorERK15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[3\]](api/langu             |
| 5cudaq14observe_result14observe_r | ages/cpp_api.html#_CPPv4N5cudaq15 |
| esultEdRK7spin_op13sample_result) | scalar_operator15scalar_operatorE |
| -                                 | RR15scalar_callbackRRNSt13unorder |
|  [cudaq::observe_result::operator | ed_mapINSt6stringENSt6stringEEE), |
|     double (C++                   |     [\[4\                         |
|     functio                       | ]](api/languages/cpp_api.html#_CP |
| n)](api/languages/cpp_api.html#_C | Pv4N5cudaq15scalar_operator15scal |
| PPv4N5cudaq14observe_resultcvdEv) | ar_operatorERR15scalar_operator), |
| -                                 |     [\[5\]](api/language          |
|  [cudaq::observe_result::raw_data | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     (C++                          | lar_operator15scalar_operatorEd), |
|     function)](ap                 |     [\[6\]](api/languag           |
| i/languages/cpp_api.html#_CPPv4N5 | es/cpp_api.html#_CPPv4N5cudaq15sc |
| cudaq14observe_result8raw_dataEv) | alar_operator15scalar_operatorEv) |
| -   [cudaq::operator_handler (C++ | -   [                             |
|     cl                            | cudaq::scalar_operator::to_matrix |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16operator_handlerE) |                                   |
| -   [cudaq::optimizable_function  |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4NK5cudaq15scalar_ope |
|     class)                        | rator9to_matrixERKNSt13unordered_ |
| ](api/languages/cpp_api.html#_CPP | mapINSt6stringENSt7complexIdEEEE) |
| v4N5cudaq20optimizable_functionE) | -   [                             |
| -   [cudaq::optimization_result   | cudaq::scalar_operator::to_string |
|     (C++                          |     (C++                          |
|     type                          |     function)](api/l              |
| )](api/languages/cpp_api.html#_CP | anguages/cpp_api.html#_CPPv4NK5cu |
| Pv4N5cudaq19optimization_resultE) | daq15scalar_operator9to_stringEv) |
| -   [cudaq::optimizer (C++        | -   [cudaq::s                     |
|     class)](api/languages/cpp_a   | calar_operator::\~scalar_operator |
| pi.html#_CPPv4N5cudaq9optimizerE) |     (C++                          |
| -   [cudaq::optimizer::optimize   |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq15scalar_operatorD0Ev) |
|  function)](api/languages/cpp_api | -   [cudaq::set_noise (C++        |
| .html#_CPPv4N5cudaq9optimizer8opt |     function)](api/langu          |
| imizeEKiRR20optimizable_function) | ages/cpp_api.html#_CPPv4N5cudaq9s |
| -   [cu                           | et_noiseERKN5cudaq11noise_modelE) |
| daq::optimizer::requiresGradients | -   [cudaq::set_random_seed (C++  |
|     (C++                          |     function)](api/               |
|     function)](api/la             | languages/cpp_api.html#_CPPv4N5cu |
| nguages/cpp_api.html#_CPPv4N5cuda | daq15set_random_seedENSt6size_tE) |
| q9optimizer17requiresGradientsEv) | -   [cudaq::simulation_precision  |
| -   [cudaq::orca (C++             |     (C++                          |
|     type)](api/languages/         |     enum)                         |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::orca::sample (C++     | v4N5cudaq20simulation_precisionE) |
|     function)](api/languages/c    | -   [                             |
| pp_api.html#_CPPv4N5cudaq4orca6sa | cudaq::simulation_precision::fp32 |
| mpleERNSt6vectorINSt6size_tEEERNS |     (C++                          |
| t6vectorINSt6size_tEEERNSt6vector |     enumerator)](api              |
| IdEERNSt6vectorIdEEiNSt6size_tE), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]]                       | udaq20simulation_precision4fp32E) |
| (api/languages/cpp_api.html#_CPPv | -   [                             |
| 4N5cudaq4orca6sampleERNSt6vectorI | cudaq::simulation_precision::fp64 |
| NSt6size_tEEERNSt6vectorINSt6size |     (C++                          |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     enumerator)](api              |
| -   [cudaq::orca::sample_async    | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq20simulation_precision4fp64E) |
|                                   | -   [cudaq::SimulationState (C++  |
| function)](api/languages/cpp_api. |     c                             |
| html#_CPPv4N5cudaq4orca12sample_a | lass)](api/languages/cpp_api.html |
| syncERNSt6vectorINSt6size_tEEERNS | #_CPPv4N5cudaq15SimulationStateE) |
| t6vectorINSt6size_tEEERNSt6vector | -   [                             |
| IdEERNSt6vectorIdEEiNSt6size_tE), | cudaq::SimulationState::precision |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     enum)](api                    |
| q4orca12sample_asyncERNSt6vectorI | /languages/cpp_api.html#_CPPv4N5c |
| NSt6size_tEEERNSt6vectorINSt6size | udaq15SimulationState9precisionE) |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | -   [cudaq:                       |
| -   [cudaq::OrcaRemoteRESTQPU     | :SimulationState::precision::fp32 |
|     (C++                          |     (C++                          |
|     cla                           |     enumerator)](api/lang         |
| ss)](api/languages/cpp_api.html#_ | uages/cpp_api.html#_CPPv4N5cudaq1 |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | 5SimulationState9precision4fp32E) |
| -   [cudaq::pauli1 (C++           | -   [cudaq:                       |
|     class)](api/languages/cp      | :SimulationState::precision::fp64 |
| p_api.html#_CPPv4N5cudaq6pauli1E) |     (C++                          |
| -                                 |     enumerator)](api/lang         |
|    [cudaq::pauli1::num_parameters | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     (C++                          | 5SimulationState9precision4fp64E) |
|     member)]                      | -                                 |
| (api/languages/cpp_api.html#_CPPv |   [cudaq::SimulationState::Tensor |
| 4N5cudaq6pauli114num_parametersE) |     (C++                          |
| -   [cudaq::pauli1::num_targets   |     struct)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     membe                         | N5cudaq15SimulationState6TensorE) |
| r)](api/languages/cpp_api.html#_C | -   [cudaq::spin_handler (C++     |
| PPv4N5cudaq6pauli111num_targetsE) |                                   |
| -   [cudaq::pauli1::pauli1 (C++   |   class)](api/languages/cpp_api.h |
|     function)](api/languages/cpp_ | tml#_CPPv4N5cudaq12spin_handlerE) |
| api.html#_CPPv4N5cudaq6pauli16pau | -   [cudaq:                       |
| li1ERKNSt6vectorIN5cudaq4realEEE) | :spin_handler::to_diagonal_matrix |
| -   [cudaq::pauli2 (C++           |     (C++                          |
|     class)](api/languages/cp      |     function)](api/la             |
| p_api.html#_CPPv4N5cudaq6pauli2E) | nguages/cpp_api.html#_CPPv4NK5cud |
| -                                 | aq12spin_handler18to_diagonal_mat |
|    [cudaq::pauli2::num_parameters | rixERNSt13unordered_mapINSt6size_ |
|     (C++                          | tENSt7int64_tEEERKNSt13unordered_ |
|     member)]                      | mapINSt6stringENSt7complexIdEEEE) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq6pauli214num_parametersE) |   [cudaq::spin_handler::to_matrix |
| -   [cudaq::pauli2::num_targets   |     (C++                          |
|     (C++                          |     function                      |
|     membe                         | )](api/languages/cpp_api.html#_CP |
| r)](api/languages/cpp_api.html#_C | Pv4N5cudaq12spin_handler9to_matri |
| PPv4N5cudaq6pauli211num_targetsE) | xERKNSt6stringENSt7complexIdEEb), |
| -   [cudaq::pauli2::pauli2 (C++   |     [\[1                          |
|     function)](api/languages/cpp_ | \]](api/languages/cpp_api.html#_C |
| api.html#_CPPv4N5cudaq6pauli26pau | PPv4NK5cudaq12spin_handler9to_mat |
| li2ERKNSt6vectorIN5cudaq4realEEE) | rixERNSt13unordered_mapINSt6size_ |
| -   [cudaq::phase_damping (C++    | tENSt7int64_tEEERKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|  class)](api/languages/cpp_api.ht | -   [cuda                         |
| ml#_CPPv4N5cudaq13phase_dampingE) | q::spin_handler::to_sparse_matrix |
| -   [cud                          |     (C++                          |
| aq::phase_damping::num_parameters |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     member)](api/lan              | daq12spin_handler16to_sparse_matr |
| guages/cpp_api.html#_CPPv4N5cudaq | ixERKNSt6stringENSt7complexIdEEb) |
| 13phase_damping14num_parametersE) | -                                 |
| -   [                             |   [cudaq::spin_handler::to_string |
| cudaq::phase_damping::num_targets |     (C++                          |
|     (C++                          |     function)](ap                 |
|     member)](api/                 | i/languages/cpp_api.html#_CPPv4NK |
| languages/cpp_api.html#_CPPv4N5cu | 5cudaq12spin_handler9to_stringEb) |
| daq13phase_damping11num_targetsE) | -                                 |
| -   [cudaq::phase_flip_channel    |   [cudaq::spin_handler::unique_id |
|     (C++                          |     (C++                          |
|     clas                          |     function)](ap                 |
| s)](api/languages/cpp_api.html#_C | i/languages/cpp_api.html#_CPPv4NK |
| PPv4N5cudaq18phase_flip_channelE) | 5cudaq12spin_handler9unique_idEv) |
| -   [cudaq::p                     | -   [cudaq::spin_op (C++          |
| hase_flip_channel::num_parameters |     type)](api/languages/cpp      |
|     (C++                          | _api.html#_CPPv4N5cudaq7spin_opE) |
|     member)](api/language         | -   [cudaq::spin_op_term (C++     |
| s/cpp_api.html#_CPPv4N5cudaq18pha |                                   |
| se_flip_channel14num_parametersE) |    type)](api/languages/cpp_api.h |
| -   [cudaq                        | tml#_CPPv4N5cudaq12spin_op_termE) |
| ::phase_flip_channel::num_targets | -   [cudaq::state (C++            |
|     (C++                          |     class)](api/languages/c       |
|     member)](api/langu            | pp_api.html#_CPPv4N5cudaq5stateE) |
| ages/cpp_api.html#_CPPv4N5cudaq18 | -   [cudaq::state::amplitude (C++ |
| phase_flip_channel11num_targetsE) |     function)](api/lang           |
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
