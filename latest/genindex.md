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
        -   [Measurement
            Matrices](using/examples/dem_from_kernel.html#measurement-matrices){.reference
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
        -   [qBraid](using/examples/hardware_providers.html#qbraid){.reference
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
| -   [Adam (class in               | -   [annotations (cudaq.DEMResult |
|     cudaq                         |     pro                           |
| .optimizers)](api/languages/pytho | perty)](api/languages/python_api. |
| n_api.html#cudaq.optimizers.Adam) | html#cudaq.DEMResult.annotations) |
| -   [add_all_qubit_channel        |     -   [(cudaq.SampleResult      |
|     (cudaq.NoiseModel             |         proper                    |
|     attribute)](api               | ty)](api/languages/python_api.htm |
| /languages/python_api.html#cudaq. | l#cudaq.SampleResult.annotations) |
| NoiseModel.add_all_qubit_channel) | -   [append (cudaq.KrausChannel   |
| -   [add_channel                  |     at                            |
|     (cudaq.NoiseModel             | tribute)](api/languages/python_ap |
|     attri                         | i.html#cudaq.KrausChannel.append) |
| bute)](api/languages/python_api.h | -   [argument_count               |
| tml#cudaq.NoiseModel.add_channel) |     (cudaq.PyKernel               |
| -   [all_gather() (in module      |     attrib                        |
|                                   | ute)](api/languages/python_api.ht |
|    cudaq.mpi)](api/languages/pyth | ml#cudaq.PyKernel.argument_count) |
| on_api.html#cudaq.mpi.all_gather) | -   [arguments (cudaq.PyKernel    |
| -   [amplitude (cudaq.State       |     a                             |
|                                   | ttribute)](api/languages/python_a |
|   attribute)](api/languages/pytho | pi.html#cudaq.PyKernel.arguments) |
| n_api.html#cudaq.State.amplitude) | -   [as_pauli                     |
| -   [amplitude_encode() (in       |     (cudaq.o                      |
|     module                        | perators.spin.SpinOperatorElement |
|     cudaq.contr                   |     attribute)](api/languages/    |
| ib)](api/languages/python_api.htm | python_api.html#cudaq.operators.s |
| l#cudaq.contrib.amplitude_encode) | pin.SpinOperatorElement.as_pauli) |
| -   [AmplitudeDampingChannel      | -   [AsyncEvolveResult (class in  |
|     (class in                     |     cudaq)](api/languages/python_ |
|     cu                            | api.html#cudaq.AsyncEvolveResult) |
| daq)](api/languages/python_api.ht | -   [AsyncObserveResult (class in |
| ml#cudaq.AmplitudeDampingChannel) |                                   |
| -   [amplitudes (cudaq.State      |    cudaq)](api/languages/python_a |
|                                   | pi.html#cudaq.AsyncObserveResult) |
|  attribute)](api/languages/python | -   [AsyncSampleResult (class in  |
| _api.html#cudaq.State.amplitudes) |     cudaq)](api/languages/python_ |
| -   [angular_encode() (in module  | api.html#cudaq.AsyncSampleResult) |
|     cudaq.con                     | -   [AsyncStateResult (class in   |
| trib)](api/languages/python_api.h |     cudaq)](api/languages/python  |
| tml#cudaq.contrib.angular_encode) | _api.html#cudaq.AsyncStateResult) |
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
| -   [canonicalize                 | -   [cudaq::p                     |
|     (cu                           | hase_flip_channel::num_parameters |
| daq.operators.boson.BosonOperator |     (C++                          |
|     attribute)](api/languages     |     member)](api/language         |
| /python_api.html#cudaq.operators. | s/cpp_api.html#_CPPv4N5cudaq18pha |
| boson.BosonOperator.canonicalize) | se_flip_channel14num_parametersE) |
|     -   [(cudaq.                  | -   [cudaq                        |
| operators.boson.BosonOperatorTerm | ::phase_flip_channel::num_targets |
|                                   |     (C++                          |
|     attribute)](api/languages/pyt |     member)](api/langu            |
| hon_api.html#cudaq.operators.boso | ages/cpp_api.html#_CPPv4N5cudaq18 |
| n.BosonOperatorTerm.canonicalize) | phase_flip_channel11num_targetsE) |
|     -   [(cudaq.                  | -   [cudaq::product_op (C++       |
| operators.fermion.FermionOperator |                                   |
|                                   |  class)](api/languages/cpp_api.ht |
|     attribute)](api/languages/pyt | ml#_CPPv4I0EN5cudaq10product_opE) |
| hon_api.html#cudaq.operators.ferm | -   [cudaq::product_op::begin     |
| ion.FermionOperator.canonicalize) |     (C++                          |
|     -   [(cudaq.oper              |     functio                       |
| ators.fermion.FermionOperatorTerm | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4NK5cudaq10product_op5beginEv) |
| attribute)](api/languages/python_ | -                                 |
| api.html#cudaq.operators.fermion. |  [cudaq::product_op::canonicalize |
| FermionOperatorTerm.canonicalize) |     (C++                          |
|     -                             |     func                          |
|  [(cudaq.operators.MatrixOperator | tion)](api/languages/cpp_api.html |
|         attribute)](api/lang      | #_CPPv4N5cudaq10product_op12canon |
| uages/python_api.html#cudaq.opera | icalizeERKNSt3setINSt6size_tEEE), |
| tors.MatrixOperator.canonicalize) |     [\[1\]](api                   |
|     -   [(c                       | /languages/cpp_api.html#_CPPv4N5c |
| udaq.operators.MatrixOperatorTerm | udaq10product_op12canonicalizeEv) |
|         attribute)](api/language  | -   [                             |
| s/python_api.html#cudaq.operators | cudaq::product_op::const_iterator |
| .MatrixOperatorTerm.canonicalize) |     (C++                          |
|     -   [(                        |     struct)](api/                 |
| cudaq.operators.spin.SpinOperator | languages/cpp_api.html#_CPPv4N5cu |
|         attribute)](api/languag   | daq10product_op14const_iteratorE) |
| es/python_api.html#cudaq.operator | -   [cudaq::product_o             |
| s.spin.SpinOperator.canonicalize) | p::const_iterator::const_iterator |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     fu                            |
|                                   | nction)](api/languages/cpp_api.ht |
|       attribute)](api/languages/p | ml#_CPPv4N5cudaq10product_op14con |
| ython_api.html#cudaq.operators.sp | st_iterator14const_iteratorEPK10p |
| in.SpinOperatorTerm.canonicalize) | roduct_opI9HandlerTyENSt6size_tE) |
| -   [captured_variables()         | -   [cudaq::produ                 |
|     (cudaq.PyKernelDecorator      | ct_op::const_iterator::operator!= |
|     method)](api/lan              |     (C++                          |
| guages/python_api.html#cudaq.PyKe |     fun                           |
| rnelDecorator.captured_variables) | ction)](api/languages/cpp_api.htm |
| -   [CentralDifference (class in  | l#_CPPv4NK5cudaq10product_op14con |
|     cudaq.gradients)              | st_iteratorneERK14const_iterator) |
| ](api/languages/python_api.html#c | -   [cudaq::produ                 |
| udaq.gradients.CentralDifference) | ct_op::const_iterator::operator\* |
| -   [channel                      |     (C++                          |
|     (cudaq.ptsbe.TraceInstruction |     function)](api/lang           |
|     property)](a                  | uages/cpp_api.html#_CPPv4NK5cudaq |
| pi/languages/python_api.html#cuda | 10product_op14const_iteratormlEv) |
| q.ptsbe.TraceInstruction.channel) | -   [cudaq::produ                 |
| -   [circuit_location             | ct_op::const_iterator::operator++ |
|     (cudaq.ptsbe.KrausSelection   |     (C++                          |
|     property)](api/lang           |     function)](api/lang           |
| uages/python_api.html#cudaq.ptsbe | uages/cpp_api.html#_CPPv4N5cudaq1 |
| .KrausSelection.circuit_location) | 0product_op14const_iteratorppEi), |
| -   [clear (cudaq.Resources       |     [\[1\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|   attribute)](api/languages/pytho | 10product_op14const_iteratorppEv) |
| n_api.html#cudaq.Resources.clear) | -   [cudaq::produc                |
|     -   [(cudaq.SampleResult      | t_op::const_iterator::operator\-- |
|         a                         |     (C++                          |
| ttribute)](api/languages/python_a |     function)](api/lang           |
| pi.html#cudaq.SampleResult.clear) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [COBYLA (class in             | 0product_op14const_iteratormmEi), |
|     cudaq.o                       |     [\[1\]](api/lan               |
| ptimizers)](api/languages/python_ | guages/cpp_api.html#_CPPv4N5cudaq |
| api.html#cudaq.optimizers.COBYLA) | 10product_op14const_iteratormmEv) |
| -   [coefficient                  | -   [cudaq::produc                |
|     (cudaq.                       | t_op::const_iterator::operator-\> |
| operators.boson.BosonOperatorTerm |     (C++                          |
|     property)](api/languages/py   |     function)](api/lan            |
| thon_api.html#cudaq.operators.bos | guages/cpp_api.html#_CPPv4N5cudaq |
| on.BosonOperatorTerm.coefficient) | 10product_op14const_iteratorptEv) |
|     -   [(cudaq.oper              | -   [cudaq::produ                 |
| ators.fermion.FermionOperatorTerm | ct_op::const_iterator::operator== |
|                                   |     (C++                          |
|   property)](api/languages/python |     fun                           |
| _api.html#cudaq.operators.fermion | ction)](api/languages/cpp_api.htm |
| .FermionOperatorTerm.coefficient) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(c                       | st_iteratoreqERK14const_iterator) |
| udaq.operators.MatrixOperatorTerm | -   [cudaq::product_op::degrees   |
|         property)](api/languag    |     (C++                          |
| es/python_api.html#cudaq.operator |     function)                     |
| s.MatrixOperatorTerm.coefficient) | ](api/languages/cpp_api.html#_CPP |
|     -   [(cuda                    | v4NK5cudaq10product_op7degreesEv) |
| q.operators.spin.SpinOperatorTerm | -   [cudaq::product_op::dump (C++ |
|         property)](api/languages/ |     functi                        |
| python_api.html#cudaq.operators.s | on)](api/languages/cpp_api.html#_ |
| pin.SpinOperatorTerm.coefficient) | CPPv4NK5cudaq10product_op4dumpEv) |
| -   [col_count                    | -   [cudaq::product_op::end (C++  |
|     (cudaq.KrausOperator          |     funct                         |
|     prope                         | ion)](api/languages/cpp_api.html# |
| rty)](api/languages/python_api.ht | _CPPv4NK5cudaq10product_op3endEv) |
| ml#cudaq.KrausOperator.col_count) | -   [c                            |
| -   [compile()                    | udaq::product_op::get_coefficient |
|     (cudaq.PyKernelDecorator      |     (C++                          |
|     metho                         |     function)](api/lan            |
| d)](api/languages/python_api.html | guages/cpp_api.html#_CPPv4NK5cuda |
| #cudaq.PyKernelDecorator.compile) | q10product_op15get_coefficientEv) |
| -   [compiledModuleCache()        | -                                 |
|     (cudaq.PyKernelDecorator      |   [cudaq::product_op::get_term_id |
|     method)](api/lang             |     (C++                          |
| uages/python_api.html#cudaq.PyKer |     function)](api                |
| nelDecorator.compiledModuleCache) | /languages/cpp_api.html#_CPPv4NK5 |
| -   [ComplexMatrix (class in      | cudaq10product_op11get_term_idEv) |
|     cudaq)](api/languages/pyt     | -                                 |
| hon_api.html#cudaq.ComplexMatrix) |   [cudaq::product_op::is_identity |
| -   [compute                      |     (C++                          |
|     (                             |     function)](api                |
| cudaq.gradients.CentralDifference | /languages/cpp_api.html#_CPPv4NK5 |
|     attribute)](api/la            | cudaq10product_op11is_identityEv) |
| nguages/python_api.html#cudaq.gra | -   [cudaq::product_op::num_ops   |
| dients.CentralDifference.compute) |     (C++                          |
|     -   [(                        |     function)                     |
| cudaq.gradients.ForwardDifference | ](api/languages/cpp_api.html#_CPP |
|         attribute)](api/la        | v4NK5cudaq10product_op7num_opsEv) |
| nguages/python_api.html#cudaq.gra | -                                 |
| dients.ForwardDifference.compute) |    [cudaq::product_op::operator\* |
|     -                             |     (C++                          |
|  [(cudaq.gradients.ParameterShift |     function)](api/languages/     |
|         attribute)](api           | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| /languages/python_api.html#cudaq. | oduct_opmlE10product_opI1TERK15sc |
| gradients.ParameterShift.compute) | alar_operatorRK10product_opI1TE), |
| -   [const()                      |     [\[1\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|   (cudaq.operators.ScalarOperator | oduct_opmlE10product_opI1TERK15sc |
|     class                         | alar_operatorRR10product_opI1TE), |
|     method)](a                    |     [\[2\]](api/languages/        |
| pi/languages/python_api.html#cuda | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| q.operators.ScalarOperator.const) | oduct_opmlE10product_opI1TERR15sc |
| -   [controls                     | alar_operatorRK10product_opI1TE), |
|     (cudaq.ptsbe.TraceInstruction |     [\[3\]](api/languages/        |
|     property)](ap                 | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| i/languages/python_api.html#cudaq | oduct_opmlE10product_opI1TERR15sc |
| .ptsbe.TraceInstruction.controls) | alar_operatorRR10product_opI1TE), |
| -   [copy                         |     [\[4\]](api/                  |
|     (cu                           | languages/cpp_api.html#_CPPv4I0EN |
| daq.operators.boson.BosonOperator | 5cudaq10product_opmlE6sum_opI1TER |
|     attribute)](api/l             | K15scalar_operatorRK6sum_opI1TE), |
| anguages/python_api.html#cudaq.op |     [\[5\]](api/                  |
| erators.boson.BosonOperator.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cudaq.                  | 5cudaq10product_opmlE6sum_opI1TER |
| operators.boson.BosonOperatorTerm | K15scalar_operatorRR6sum_opI1TE), |
|         attribute)](api/langu     |     [\[6\]](api/                  |
| ages/python_api.html#cudaq.operat | languages/cpp_api.html#_CPPv4I0EN |
| ors.boson.BosonOperatorTerm.copy) | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [(cudaq.                  | R15scalar_operatorRK6sum_opI1TE), |
| operators.fermion.FermionOperator |     [\[7\]](api/                  |
|         attribute)](api/langu     | languages/cpp_api.html#_CPPv4I0EN |
| ages/python_api.html#cudaq.operat | 5cudaq10product_opmlE6sum_opI1TER |
| ors.fermion.FermionOperator.copy) | R15scalar_operatorRR6sum_opI1TE), |
|     -   [(cudaq.oper              |     [\[8\]](api/languages         |
| ators.fermion.FermionOperatorTerm | /cpp_api.html#_CPPv4NK5cudaq10pro |
|         attribute)](api/languages | duct_opmlERK6sum_opI9HandlerTyE), |
| /python_api.html#cudaq.operators. |     [\[9\]](api/languages/cpp_a   |
| fermion.FermionOperatorTerm.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -                             | opmlERK10product_opI9HandlerTyE), |
|  [(cudaq.operators.MatrixOperator |     [\[10\]](api/language         |
|         attribute)](              | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| api/languages/python_api.html#cud | roduct_opmlERK15scalar_operator), |
| aq.operators.MatrixOperator.copy) |     [\[11\]](api/languages/cpp_a  |
|     -   [(c                       | pi.html#_CPPv4NKR5cudaq10product_ |
| udaq.operators.MatrixOperatorTerm | opmlERR10product_opI9HandlerTyE), |
|         attribute)](api/          |     [\[12\]](api/language         |
| languages/python_api.html#cudaq.o | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| perators.MatrixOperatorTerm.copy) | roduct_opmlERR15scalar_operator), |
|     -   [(                        |     [\[13\]](api/languages/cpp_   |
| cudaq.operators.spin.SpinOperator | api.html#_CPPv4NO5cudaq10product_ |
|         attribute)](api           | opmlERK10product_opI9HandlerTyE), |
| /languages/python_api.html#cudaq. |     [\[14\]](api/languag          |
| operators.spin.SpinOperator.copy) | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     -   [(cuda                    | roduct_opmlERK15scalar_operator), |
| q.operators.spin.SpinOperatorTerm |     [\[15\]](api/languages/cpp_   |
|         attribute)](api/lan       | api.html#_CPPv4NO5cudaq10product_ |
| guages/python_api.html#cudaq.oper | opmlERR10product_opI9HandlerTyE), |
| ators.spin.SpinOperatorTerm.copy) |     [\[16\]](api/langua           |
| -   [count (cudaq.Resources       | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|                                   | product_opmlERR15scalar_operator) |
|   attribute)](api/languages/pytho | -                                 |
| n_api.html#cudaq.Resources.count) |   [cudaq::product_op::operator\*= |
|     -   [(cudaq.SampleResult      |     (C++                          |
|         a                         |     function)](api/languages/cpp  |
| ttribute)](api/languages/python_a | _api.html#_CPPv4N5cudaq10product_ |
| pi.html#cudaq.SampleResult.count) | opmLERK10product_opI9HandlerTyE), |
| -   [count_controls               |     [\[1\]](api/langua            |
|     (cudaq.Resources              | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     attribu                       | roduct_opmLERK15scalar_operator), |
| te)](api/languages/python_api.htm |     [\[2\]](api/languages/cp      |
| l#cudaq.Resources.count_controls) | p_api.html#_CPPv4N5cudaq10product |
| -   [count_instructions           | _opmLERR10product_opI9HandlerTyE) |
|                                   | -   [cudaq::product_op::operator+ |
|   (cudaq.ptsbe.PTSBEExecutionData |     (C++                          |
|     attribute)](api/languages/    |     function)](api/langu          |
| python_api.html#cudaq.ptsbe.PTSBE | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ExecutionData.count_instructions) | q10product_opplE6sum_opI1TERK15sc |
| -   [counts (cudaq.ObserveResult  | alar_operatorRK10product_opI1TE), |
|     att                           |     [\[1\]](api/                  |
| ribute)](api/languages/python_api | languages/cpp_api.html#_CPPv4I0EN |
| .html#cudaq.ObserveResult.counts) | 5cudaq10product_opplE6sum_opI1TER |
|     -   [(cudaq.SampleResult      | K15scalar_operatorRK6sum_opI1TE), |
|         p                         |     [\[2\]](api/langu             |
| roperty)](api/languages/python_ap | ages/cpp_api.html#_CPPv4I0EN5cuda |
| i.html#cudaq.SampleResult.counts) | q10product_opplE6sum_opI1TERK15sc |
| -   [csr_spmatrix (C++            | alar_operatorRR10product_opI1TE), |
|     type)](api/languages/c        |     [\[3\]](api/                  |
| pp_api.html#_CPPv412csr_spmatrix) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq                         | 5cudaq10product_opplE6sum_opI1TER |
|     -   [module](api/langua       | K15scalar_operatorRR6sum_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[4\]](api/langu             |
| -   [cudaq (C++                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     type)](api/lan                | q10product_opplE6sum_opI1TERR15sc |
| guages/cpp_api.html#_CPPv45cudaq) | alar_operatorRK10product_opI1TE), |
| -   [cudaq.apply_noise() (in      |     [\[5\]](api/                  |
|     module                        | languages/cpp_api.html#_CPPv4I0EN |
|     cudaq)](api/languages/python_ | 5cudaq10product_opplE6sum_opI1TER |
| api.html#cudaq.cudaq.apply_noise) | R15scalar_operatorRK6sum_opI1TE), |
| -   cudaq.boson                   |     [\[6\]](api/langu             |
|     -   [module](api/languages/py | ages/cpp_api.html#_CPPv4I0EN5cuda |
| thon_api.html#module-cudaq.boson) | q10product_opplE6sum_opI1TERR15sc |
| -   cudaq.fermion                 | alar_operatorRR10product_opI1TE), |
|                                   |     [\[7\]](api/                  |
|   -   [module](api/languages/pyth | languages/cpp_api.html#_CPPv4I0EN |
| on_api.html#module-cudaq.fermion) | 5cudaq10product_opplE6sum_opI1TER |
| -   cudaq.operators.custom        | R15scalar_operatorRR6sum_opI1TE), |
|     -   [mo                       |     [\[8\]](api/languages/cpp_a   |
| dule](api/languages/python_api.ht | pi.html#_CPPv4NKR5cudaq10product_ |
| ml#module-cudaq.operators.custom) | opplERK10product_opI9HandlerTyE), |
| -   cudaq.spin                    |     [\[9\]](api/language          |
|     -   [module](api/languages/p  | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ython_api.html#module-cudaq.spin) | roduct_opplERK15scalar_operator), |
| -   [cudaq::amplitude_damping     |     [\[10\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     cla                           | duct_opplERK6sum_opI9HandlerTyE), |
| ss)](api/languages/cpp_api.html#_ |     [\[11\]](api/languages/cpp_a  |
| CPPv4N5cudaq17amplitude_dampingE) | pi.html#_CPPv4NKR5cudaq10product_ |
| -                                 | opplERR10product_opI9HandlerTyE), |
| [cudaq::amplitude_damping_channel |     [\[12\]](api/language         |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     class)](api                   | roduct_opplERR15scalar_operator), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[13\]](api/languages/       |
| udaq25amplitude_damping_channelE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::amplitud              | duct_opplERR6sum_opI9HandlerTyE), |
| e_damping_channel::num_parameters |     [\[                           |
|     (C++                          | 14\]](api/languages/cpp_api.html# |
|     member)](api/languages/cpp_a  | _CPPv4NKR5cudaq10product_opplEv), |
| pi.html#_CPPv4N5cudaq25amplitude_ |     [\[15\]](api/languages/cpp_   |
| damping_channel14num_parametersE) | api.html#_CPPv4NO5cudaq10product_ |
| -   [cudaq::ampli                 | opplERK10product_opI9HandlerTyE), |
| tude_damping_channel::num_targets |     [\[16\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     member)](api/languages/cp     | roduct_opplERK15scalar_operator), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [\[17\]](api/languages        |
| de_damping_channel11num_targetsE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::AnalogRemoteRESTQPU   | duct_opplERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[18\]](api/languages/cpp_   |
|     class                         | api.html#_CPPv4NO5cudaq10product_ |
| )](api/languages/cpp_api.html#_CP | opplERR10product_opI9HandlerTyE), |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) |     [\[19\]](api/languag          |
| -   [cudaq::apply_noise (C++      | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     function)](api/               | roduct_opplERR15scalar_operator), |
| languages/cpp_api.html#_CPPv4I0Dp |     [\[20\]](api/languages        |
| EN5cudaq11apply_noiseEvDpRR4Args) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::async_result (C++     | duct_opplERR6sum_opI9HandlerTyE), |
|     c                             |     [                             |
| lass)](api/languages/cpp_api.html | \[21\]](api/languages/cpp_api.htm |
| #_CPPv4I0EN5cudaq12async_resultE) | l#_CPPv4NO5cudaq10product_opplEv) |
| -   [cudaq::async_result::get     | -   [cudaq::product_op::operator- |
|     (C++                          |     (C++                          |
|     functi                        |     function)](api/langu          |
| on)](api/languages/cpp_api.html#_ | ages/cpp_api.html#_CPPv4I0EN5cuda |
| CPPv4N5cudaq12async_result3getEv) | q10product_opmiE6sum_opI1TERK15sc |
| -   [cudaq::async_sample_result   | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     type                          | languages/cpp_api.html#_CPPv4I0EN |
| )](api/languages/cpp_api.html#_CP | 5cudaq10product_opmiE6sum_opI1TER |
| Pv4N5cudaq19async_sample_resultE) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::BaseRemoteRESTQPU     |     [\[2\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cla                           | q10product_opmiE6sum_opI1TERK15sc |
| ss)](api/languages/cpp_api.html#_ | alar_operatorRR10product_opI1TE), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [\[3\]](api/                  |
| -   [cudaq::bit_flip_channel (C++ | languages/cpp_api.html#_CPPv4I0EN |
|     cl                            | 5cudaq10product_opmiE6sum_opI1TER |
| ass)](api/languages/cpp_api.html# | K15scalar_operatorRR6sum_opI1TE), |
| _CPPv4N5cudaq16bit_flip_channelE) |     [\[4\]](api/langu             |
| -   [cudaq:                       | ages/cpp_api.html#_CPPv4I0EN5cuda |
| :bit_flip_channel::num_parameters | q10product_opmiE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRK10product_opI1TE), |
|     member)](api/langua           |     [\[5\]](api/                  |
| ges/cpp_api.html#_CPPv4N5cudaq16b | languages/cpp_api.html#_CPPv4I0EN |
| it_flip_channel14num_parametersE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cud                          | R15scalar_operatorRK6sum_opI1TE), |
| aq::bit_flip_channel::num_targets |     [\[6\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     member)](api/lan              | q10product_opmiE6sum_opI1TERR15sc |
| guages/cpp_api.html#_CPPv4N5cudaq | alar_operatorRR10product_opI1TE), |
| 16bit_flip_channel11num_targetsE) |     [\[7\]](api/                  |
| -   [cudaq::boson_handler (C++    | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opmiE6sum_opI1TER |
|  class)](api/languages/cpp_api.ht | R15scalar_operatorRR6sum_opI1TE), |
| ml#_CPPv4N5cudaq13boson_handlerE) |     [\[8\]](api/languages/cpp_a   |
| -   [cudaq::boson_op (C++         | pi.html#_CPPv4NKR5cudaq10product_ |
|     type)](api/languages/cpp_     | opmiERK10product_opI9HandlerTyE), |
| api.html#_CPPv4N5cudaq8boson_opE) |     [\[9\]](api/language          |
| -   [cudaq::boson_op_term (C++    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opmiERK15scalar_operator), |
|   type)](api/languages/cpp_api.ht |     [\[10\]](api/languages/       |
| ml#_CPPv4N5cudaq13boson_op_termE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::CodeGenConfig (C++    | duct_opmiERK6sum_opI9HandlerTyE), |
|                                   |     [\[11\]](api/languages/cpp_a  |
| struct)](api/languages/cpp_api.ht | pi.html#_CPPv4NKR5cudaq10product_ |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | opmiERR10product_opI9HandlerTyE), |
| -   [cudaq::commutation_relations |     [\[12\]](api/language         |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     struct)]                      | roduct_opmiERR15scalar_operator), |
| (api/languages/cpp_api.html#_CPPv |     [\[13\]](api/languages/       |
| 4N5cudaq21commutation_relationsE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::complex (C++          | duct_opmiERR6sum_opI9HandlerTyE), |
|     type)](api/languages/cpp      |     [\[                           |
| _api.html#_CPPv4N5cudaq7complexE) | 14\]](api/languages/cpp_api.html# |
| -   [cudaq::complex_matrix (C++   | _CPPv4NKR5cudaq10product_opmiEv), |
|                                   |     [\[15\]](api/languages/cpp_   |
| class)](api/languages/cpp_api.htm | api.html#_CPPv4NO5cudaq10product_ |
| l#_CPPv4N5cudaq14complex_matrixE) | opmiERK10product_opI9HandlerTyE), |
| -                                 |     [\[16\]](api/languag          |
|   [cudaq::complex_matrix::adjoint | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmiERK15scalar_operator), |
|     function)](a                  |     [\[17\]](api/languages        |
| pi/languages/cpp_api.html#_CPPv4N | /cpp_api.html#_CPPv4NO5cudaq10pro |
| 5cudaq14complex_matrix7adjointEv) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::                      |     [\[18\]](api/languages/cpp_   |
| complex_matrix::diagonal_elements | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmiERR10product_opI9HandlerTyE), |
|     function)](api/languages      |     [\[19\]](api/languag          |
| /cpp_api.html#_CPPv4NK5cudaq14com | es/cpp_api.html#_CPPv4NO5cudaq10p |
| plex_matrix17diagonal_elementsEi) | roduct_opmiERR15scalar_operator), |
| -   [cudaq::complex_matrix::dump  |     [\[20\]](api/languages        |
|     (C++                          | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     function)](api/language       | duct_opmiERR6sum_opI9HandlerTyE), |
| s/cpp_api.html#_CPPv4NK5cudaq14co |     [                             |
| mplex_matrix4dumpERNSt7ostreamE), | \[21\]](api/languages/cpp_api.htm |
|     [\[1\]]                       | l#_CPPv4NO5cudaq10product_opmiEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::product_op::operator/ |
| 4NK5cudaq14complex_matrix4dumpEv) |     (C++                          |
| -   [c                            |     function)](api/language       |
| udaq::complex_matrix::eigenvalues | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opdvERK15scalar_operator), |
|     function)](api/lan            |     [\[1\]](api/language          |
| guages/cpp_api.html#_CPPv4NK5cuda | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| q14complex_matrix11eigenvaluesEv) | roduct_opdvERR15scalar_operator), |
| -   [cu                           |     [\[2\]](api/languag           |
| daq::complex_matrix::eigenvectors | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opdvERK15scalar_operator), |
|     function)](api/lang           |     [\[3\]](api/langua            |
| uages/cpp_api.html#_CPPv4NK5cudaq | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| 14complex_matrix12eigenvectorsEv) | product_opdvERR15scalar_operator) |
| -   [c                            | -                                 |
| udaq::complex_matrix::exponential |    [cudaq::product_op::operator/= |
|     (C++                          |     (C++                          |
|     function)](api/la             |     function)](api/langu          |
| nguages/cpp_api.html#_CPPv4N5cuda | ages/cpp_api.html#_CPPv4N5cudaq10 |
| q14complex_matrix11exponentialEv) | product_opdVERK15scalar_operator) |
| -                                 | -   [cudaq::product_op::operator= |
|  [cudaq::complex_matrix::identity |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)](api/languages      | anguages/cpp_api.html#_CPPv4I00EN |
| /cpp_api.html#_CPPv4N5cudaq14comp | 5cudaq10product_opaSER10product_o |
| lex_matrix8identityEKNSt6size_tE) | pI9HandlerTyERK10product_opI1TE), |
| -                                 |     [\[1\]](api/languages/cpp     |
| [cudaq::complex_matrix::kronecker | _api.html#_CPPv4N5cudaq10product_ |
|     (C++                          | opaSERK10product_opI9HandlerTyE), |
|     function)](api/lang           |     [\[2\]](api/languages/cp      |
| uages/cpp_api.html#_CPPv4I00EN5cu | p_api.html#_CPPv4N5cudaq10product |
| daq14complex_matrix9kroneckerE14c | _opaSERR10product_opI9HandlerTyE) |
| omplex_matrix8Iterable8Iterable), | -                                 |
|     [\[1\]](api/l                 |    [cudaq::product_op::operator== |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq14complex_matrix9kroneckerERK14 |     function)](api/languages/cpp  |
| complex_matrixRK14complex_matrix) | _api.html#_CPPv4NK5cudaq10product |
| -   [cudaq::c                     | _opeqERK10product_opI9HandlerTyE) |
| omplex_matrix::minimal_eigenvalue | -                                 |
|     (C++                          |  [cudaq::product_op::operator\[\] |
|     function)](api/languages/     |     (C++                          |
| cpp_api.html#_CPPv4NK5cudaq14comp |     function)](ap                 |
| lex_matrix18minimal_eigenvalueEv) | i/languages/cpp_api.html#_CPPv4NK |
| -   [                             | 5cudaq10product_opixENSt6size_tE) |
| cudaq::complex_matrix::operator() | -                                 |
|     (C++                          |    [cudaq::product_op::product_op |
|     function)](api/languages/cpp  |     (C++                          |
| _api.html#_CPPv4N5cudaq14complex_ |     f                             |
| matrixclENSt6size_tENSt6size_tE), | unction)](api/languages/cpp_api.h |
|     [\[1\]](api/languages/cpp     | tml#_CPPv4I00EN5cudaq10product_op |
| _api.html#_CPPv4NK5cudaq14complex | 10product_opERK10product_opI1TE), |
| _matrixclENSt6size_tENSt6size_tE) |     [\[1\]]                       |
| -   [                             | (api/languages/cpp_api.html#_CPPv |
| cudaq::complex_matrix::operator\* | 4I00EN5cudaq10product_op10product |
|     (C++                          | _opERK10product_opI1TERKN14matrix |
|     function)](api/langua         | _handler20commutation_behaviorE), |
| ges/cpp_api.html#_CPPv4N5cudaq14c |                                   |
| omplex_matrixmlEN14complex_matrix |   [\[2\]](api/languages/cpp_api.h |
| 10value_typeERK14complex_matrix), | tml#_CPPv4N5cudaq10product_op10pr |
|     [\[1\]                        | oduct_opENSt6size_tENSt6size_tE), |
| ](api/languages/cpp_api.html#_CPP |     [\[3\]](api/languages/cp      |
| v4N5cudaq14complex_matrixmlERK14c | p_api.html#_CPPv4N5cudaq10product |
| omplex_matrixRK14complex_matrix), | _op10product_opENSt7complexIdEE), |
|                                   |     [\[4\]](api/l                 |
|  [\[2\]](api/languages/cpp_api.ht | anguages/cpp_api.html#_CPPv4N5cud |
| ml#_CPPv4N5cudaq14complex_matrixm | aq10product_op10product_opERK10pr |
| lERK14complex_matrixRKNSt6vectorI | oduct_opI9HandlerTyENSt6size_tE), |
| N14complex_matrix10value_typeEEE) |     [\[5\]](api/l                 |
| -                                 | anguages/cpp_api.html#_CPPv4N5cud |
| [cudaq::complex_matrix::operator+ | aq10product_op10product_opERR10pr |
|     (C++                          | oduct_opI9HandlerTyENSt6size_tE), |
|     function                      |     [\[6\]](api/languages         |
| )](api/languages/cpp_api.html#_CP | /cpp_api.html#_CPPv4N5cudaq10prod |
| Pv4N5cudaq14complex_matrixplERK14 | uct_op10product_opERR9HandlerTy), |
| complex_matrixRK14complex_matrix) |     [\[7\]](ap                    |
| -                                 | i/languages/cpp_api.html#_CPPv4N5 |
| [cudaq::complex_matrix::operator- | cudaq10product_op10product_opEd), |
|     (C++                          |     [\[8\]](a                     |
|     function                      | pi/languages/cpp_api.html#_CPPv4N |
| )](api/languages/cpp_api.html#_CP | 5cudaq10product_op10product_opEv) |
| Pv4N5cudaq14complex_matrixmiERK14 | -   [cuda                         |
| complex_matrixRK14complex_matrix) | q::product_op::to_diagonal_matrix |
| -   [cu                           |     (C++                          |
| daq::complex_matrix::operator\[\] |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4NK5c |
|                                   | udaq10product_op18to_diagonal_mat |
|  function)](api/languages/cpp_api | rixENSt13unordered_mapINSt6size_t |
| .html#_CPPv4N5cudaq14complex_matr | ENSt7int64_tEEERKNSt13unordered_m |
| ixixERKNSt6vectorINSt6size_tEEE), | apINSt6stringENSt7complexIdEEEEb) |
|     [\[1\]](api/languages/cpp_api | -   [cudaq::product_op::to_matrix |
| .html#_CPPv4NK5cudaq14complex_mat |     (C++                          |
| rixixERKNSt6vectorINSt6size_tEEE) |     funct                         |
| -   [cudaq::complex_matrix::power | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NK5cudaq10product_op9to_mat |
|     function)]                    | rixENSt13unordered_mapINSt6size_t |
| (api/languages/cpp_api.html#_CPPv | ENSt7int64_tEEERKNSt13unordered_m |
| 4N5cudaq14complex_matrix5powerEi) | apINSt6stringENSt7complexIdEEEEb) |
| -                                 | -   [cu                           |
|  [cudaq::complex_matrix::set_zero | daq::product_op::to_sparse_matrix |
|     (C++                          |     (C++                          |
|     function)](ap                 |     function)](ap                 |
| i/languages/cpp_api.html#_CPPv4N5 | i/languages/cpp_api.html#_CPPv4NK |
| cudaq14complex_matrix8set_zeroEv) | 5cudaq10product_op16to_sparse_mat |
| -                                 | rixENSt13unordered_mapINSt6size_t |
| [cudaq::complex_matrix::to_string | ENSt7int64_tEEERKNSt13unordered_m |
|     (C++                          | apINSt6stringENSt7complexIdEEEEb) |
|     function)](api/               | -   [cudaq::product_op::to_string |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14complex_matrix9to_stringEv) |     function)](                   |
| -   [                             | api/languages/cpp_api.html#_CPPv4 |
| cudaq::complex_matrix::value_type | NK5cudaq10product_op9to_stringEv) |
|     (C++                          | -                                 |
|     type)](api/                   |  [cudaq::product_op::\~product_op |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq14complex_matrix10value_typeE) |     fu                            |
| -   [cudaq::contrib (C++          | nction)](api/languages/cpp_api.ht |
|     type)](api/languages/cpp      | ml#_CPPv4N5cudaq10product_opD0Ev) |
| _api.html#_CPPv4N5cudaq7contribE) | -   [cudaq::ptsbe (C++            |
| -                                 |     type)](api/languages/c        |
| [cudaq::contrib::amplitude_encode | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
|     (C++                          | -   [cudaq::p                     |
|     function)](api/language       | tsbe::ConditionalSamplingStrategy |
| s/cpp_api.html#_CPPv4N5cudaq7cont |     (C++                          |
| rib16amplitude_encodeENSt4spanIKN |     class)](api/languag           |
| St7complexIdEEEENSt7complexIdEE), | es/cpp_api.html#_CPPv4N5cudaq5pts |
|     [\[1\]](api/language          | be27ConditionalSamplingStrategyE) |
| s/cpp_api.html#_CPPv4N5cudaq7cont | -   [cudaq::ptsbe::C              |
| rib16amplitude_encodeENSt4spanIKN | onditionalSamplingStrategy::clone |
| St7complexIfEEEENSt7complexIdEE), |     (C++                          |
|     [\[2\]                        |                                   |
| ](api/languages/cpp_api.html#_CPP |    function)](api/languages/cpp_a |
| v4N5cudaq7contrib16amplitude_enco | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| deENSt4spanIKdEENSt7complexIdEE), | ditionalSamplingStrategy5cloneEv) |
|     [\[3\]                        | -   [cuda                         |
| ](api/languages/cpp_api.html#_CPP | q::ptsbe::ConditionalSamplingStra |
| v4N5cudaq7contrib16amplitude_enco | tegy::ConditionalSamplingStrategy |
| deENSt4spanIKfEENSt7complexIdEE), |     (C++                          |
|                                   |     function)](api/lang           |
| [\[4\]](api/languages/cpp_api.htm | uages/cpp_api.html#_CPPv4N5cudaq5 |
| l#_CPPv4N5cudaq7contrib16amplitud | ptsbe27ConditionalSamplingStrateg |
| e_encodeERK5stateNSt7complexIdEE) | y27ConditionalSamplingStrategyE19 |
| -                                 | TrajectoryPredicateNSt8uint64_tE) |
|   [cudaq::contrib::angular_encode | -                                 |
|     (C++                          |   [cudaq::ptsbe::ConditionalSampl |
|                                   | ingStrategy::generateTrajectories |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I0EN5cudaq7contrib14a |     function)](api/language       |
| ngular_encodeEvRR6KernelR10QuakeV | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| alueNSt4spanIKdEE12RotationAxis), | be27ConditionalSamplingStrategy20 |
|     [\[1\]](api/languages/cpp_api | generateTrajectoriesENSt4spanIKN6 |
| .html#_CPPv4I0EN5cudaq7contrib14a | detail10NoisePointEEENSt6size_tE) |
| ngular_encodeEvRR6KernelR10QuakeV | -   [cudaq::ptsbe::               |
| alueR10QuakeValue12RotationAxis), | ConditionalSamplingStrategy::name |
|                                   |     (C++                          |
|   [\[2\]](api/languages/cpp_api.h |     function)](api/languages/cpp_ |
| tml#_CPPv4I0EN5cudaq7contrib14ang | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| ular_encodeEvRR6KernelR10QuakeVal | nditionalSamplingStrategy4nameEv) |
| ueRKNSt6vectorIdEE12RotationAxis) | -   [cudaq:                       |
| -   [cudaq::contrib::draw (C++    | :ptsbe::ConditionalSamplingStrate |
|     function)                     | gy::\~ConditionalSamplingStrategy |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4I0DpEN5cudaq7contrib4drawENSt6s |     function)](api/languages/     |
| tringERR13QuantumKernelDpRR4Args) | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| -                                 | 7ConditionalSamplingStrategyD0Ev) |
| [cudaq::contrib::get_unitary_cmat | -                                 |
|     (C++                          | [cudaq::ptsbe::detail::NoisePoint |
|     function)](api/languages/cp   |     (C++                          |
| p_api.html#_CPPv4I0DpEN5cudaq7con |     struct)](a                    |
| trib16get_unitary_cmatE14complex_ | pi/languages/cpp_api.html#_CPPv4N |
| matrixRR13QuantumKernelDpRR4Args) | 5cudaq5ptsbe6detail10NoisePointE) |
| -   [cudaq::contrib::RotationAxis | -   [cudaq::p                     |
|     (C++                          | tsbe::detail::NoisePoint::channel |
|     enum)                         |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     member)](api/langu            |
| v4N5cudaq7contrib12RotationAxisE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -                                 | tsbe6detail10NoisePoint7channelE) |
|  [cudaq::contrib::RotationAxis::X | -   [cudaq::ptsbe::det            |
|     (C++                          | ail::NoisePoint::circuit_location |
|     enumerator)](                 |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     member)](api/languages/cpp_a  |
| N5cudaq7contrib12RotationAxis1XE) | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| -                                 | l10NoisePoint16circuit_locationE) |
|  [cudaq::contrib::RotationAxis::Y | -   [cudaq::p                     |
|     (C++                          | tsbe::detail::NoisePoint::op_name |
|     enumerator)](                 |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     member)](api/langu            |
| N5cudaq7contrib12RotationAxis1YE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -                                 | tsbe6detail10NoisePoint7op_nameE) |
|  [cudaq::contrib::RotationAxis::Z | -   [cudaq::                      |
|     (C++                          | ptsbe::detail::NoisePoint::qubits |
|     enumerator)](                 |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     member)](api/lang             |
| N5cudaq7contrib12RotationAxis1ZE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -   [cudaq::cudaq_json (C++       | ptsbe6detail10NoisePoint6qubitsE) |
|     class)](api/languages/cpp_api | -   [cudaq::                      |
| .html#_CPPv4N5cudaq10cudaq_jsonE) | ptsbe::ExhaustiveSamplingStrategy |
| -   [cudaq::DefaultQPU (C++       |     (C++                          |
|     class)](api/languages/cpp_api |     class)](api/langua            |
| .html#_CPPv4N5cudaq10DefaultQPUE) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq::dem_from_kernel (C++  | sbe26ExhaustiveSamplingStrategyE) |
|     function)](api                | -   [cudaq::ptsbe::               |
| /languages/cpp_api.html#_CPPv4I0D | ExhaustiveSamplingStrategy::clone |
| pEN5cudaq15dem_from_kernelENSt6st |     (C++                          |
| ringERR13QuantumKernelDpRR4Args), |     function)](api/languages/cpp_ |
|     [                             | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| \[1\]](api/languages/cpp_api.html | haustiveSamplingStrategy5cloneEv) |
| #_CPPv4I0DpEN5cudaq15dem_from_ker | -   [cu                           |
| nelENSt6stringERR13QuantumKernelP | daq::ptsbe::ExhaustiveSamplingStr |
| KN5cudaq11noise_modelEDpRR4Args), | ategy::ExhaustiveSamplingStrategy |
|     [\[2\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4I0DpEN5cudaq15de |     function)](api/la             |
| m_from_kernelENSt6stringERR13Quan | nguages/cpp_api.html#_CPPv4N5cuda |
| tumKernelPKN5cudaq11noise_modelER | q5ptsbe26ExhaustiveSamplingStrate |
| KN5cudaq11dem_optionsEDpRR4Args), | gy26ExhaustiveSamplingStrategyEv) |
|     [\[3\]](ap                    | -                                 |
| i/languages/cpp_api.html#_CPPv4I0 |    [cudaq::ptsbe::ExhaustiveSampl |
| DpEN5cudaq15dem_from_kernelENSt6s | ingStrategy::generateTrajectories |
| tringERR13QuantumKernelPKN5cudaq1 |     (C++                          |
| 1noise_modelERKN5cudaq11dem_optio |     function)](api/languag        |
| nsERN5cudaq15M2DSparseMatrixERN5c | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| udaq15M2OSparseMatrixEDpRR4Args), | sbe26ExhaustiveSamplingStrategy20 |
|     [\[4\]](api/language          | generateTrajectoriesENSt4spanIKN6 |
| s/cpp_api.html#_CPPv4I0DpEN5cudaq | detail10NoisePointEEENSt6size_tE) |
| 15dem_from_kernelENSt6stringERR13 | -   [cudaq::ptsbe:                |
| QuantumKernelPKN5cudaq11noise_mod | :ExhaustiveSamplingStrategy::name |
| elERN5cudaq15M2DSparseMatrixERN5c |     (C++                          |
| udaq15M2OSparseMatrixEDpRR4Args), |     function)](api/languages/cpp  |
|     [\[5\]](api/languages/cpp_api | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| .html#_CPPv4I0DpEN5cudaq15dem_fro | xhaustiveSamplingStrategy4nameEv) |
| m_kernelENSt6stringERR13QuantumKe | -   [cuda                         |
| rnelRN5cudaq15M2DSparseMatrixERN5 | q::ptsbe::ExhaustiveSamplingStrat |
| cudaq15M2OSparseMatrixEDpRR4Args) | egy::\~ExhaustiveSamplingStrategy |
| -   [cudaq::dem_options (C++      |     (C++                          |
|                                   |     function)](api/languages      |
|   struct)](api/languages/cpp_api. | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| html#_CPPv4N5cudaq11dem_optionsE) | 26ExhaustiveSamplingStrategyD0Ev) |
| -   [cudaq::d                     | -   [cuda                         |
| em_options::allow_gauge_detectors | q::ptsbe::OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/language         |     class)](api/lan               |
| s/cpp_api.html#_CPPv4N5cudaq11dem | guages/cpp_api.html#_CPPv4N5cudaq |
| _options21allow_gauge_detectorsE) | 5ptsbe23OrderedSamplingStrategyE) |
| -   [cudaq::dem_options::appr     | -   [cudaq::ptsb                  |
| oximate_disjoint_errors_threshold | e::OrderedSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     memb                          |     function)](api/languages/c    |
| er)](api/languages/cpp_api.html#_ | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| CPPv4N5cudaq11dem_options37approx | 3OrderedSamplingStrategy5cloneEv) |
| imate_disjoint_errors_thresholdE) | -   [cudaq::ptsbe::OrderedSampl   |
| -   [cuda                         | ingStrategy::generateTrajectories |
| q::dem_options::block_decompositi |     (C++                          |
| on_from_introducing_remnant_edges |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4NK5cudaq |
|     member)](api/lang             | 5ptsbe23OrderedSamplingStrategy20 |
| uages/cpp_api.html#_CPPv4N5cudaq1 | generateTrajectoriesENSt4spanIKN6 |
| 1dem_options50block_decomposition | detail10NoisePointEEENSt6size_tE) |
| _from_introducing_remnant_edgesE) | -   [cudaq::pts                   |
| -   [cud                          | be::OrderedSamplingStrategy::name |
| aq::dem_options::decompose_errors |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     member)](api/lan              | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| guages/cpp_api.html#_CPPv4N5cudaq | 23OrderedSamplingStrategy4nameEv) |
| 11dem_options16decompose_errorsE) | -                                 |
| -                                 |    [cudaq::ptsbe::OrderedSampling |
|   [cudaq::dem_options::fold_loops | Strategy::OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](a                    |     function)](                   |
| pi/languages/cpp_api.html#_CPPv4N | api/languages/cpp_api.html#_CPPv4 |
| 5cudaq11dem_options10fold_loopsE) | N5cudaq5ptsbe23OrderedSamplingStr |
| -   [cudaq::dem_optio             | ategy23OrderedSamplingStrategyEv) |
| ns::ignore_decomposition_failures | -                                 |
|     (C++                          |  [cudaq::ptsbe::OrderedSamplingSt |
|     member)](api/languages/cpp_ap | rategy::\~OrderedSamplingStrategy |
| i.html#_CPPv4N5cudaq11dem_options |     (C++                          |
| 29ignore_decomposition_failuresE) |     function)](api/langua         |
| -   [cudaq::dem_opt               | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| ions::return_measurement_matrices | sbe23OrderedSamplingStrategyD0Ev) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/languages/cpp_   | be::ProbabilisticSamplingStrategy |
| api.html#_CPPv4N5cudaq11dem_optio |     (C++                          |
| ns27return_measurement_matricesE) |     class)](api/languages         |
| -   [cudaq::depolarization1 (C++  | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     c                             | 29ProbabilisticSamplingStrategyE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::ptsbe::Pro            |
| #_CPPv4N5cudaq15depolarization1E) | babilisticSamplingStrategy::clone |
| -   [cudaq::depolarization2 (C++  |     (C++                          |
|     c                             |                                   |
| lass)](api/languages/cpp_api.html |  function)](api/languages/cpp_api |
| #_CPPv4N5cudaq15depolarization2E) | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| -   [cudaq:                       | bilisticSamplingStrategy5cloneEv) |
| :depolarization2::depolarization2 | -                                 |
|     (C++                          | [cudaq::ptsbe::ProbabilisticSampl |
|     function)](api/languages/cp   | ingStrategy::generateTrajectories |
| p_api.html#_CPPv4N5cudaq15depolar |     (C++                          |
| ization215depolarization2EK4real) |     function)](api/languages/     |
| -   [cudaq                        | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| ::depolarization2::num_parameters | 29ProbabilisticSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/langu            | detail10NoisePointEEENSt6size_tE) |
| ages/cpp_api.html#_CPPv4N5cudaq15 | -   [cudaq::ptsbe::Pr             |
| depolarization214num_parametersE) | obabilisticSamplingStrategy::name |
| -   [cu                           |     (C++                          |
| daq::depolarization2::num_targets |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     member)](api/la               | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| nguages/cpp_api.html#_CPPv4N5cuda | abilisticSamplingStrategy4nameEv) |
| q15depolarization211num_targetsE) | -   [cudaq::p                     |
| -                                 | tsbe::ProbabilisticSamplingStrate |
|    [cudaq::depolarization_channel | gy::ProbabilisticSamplingStrategy |
|     (C++                          |     (C++                          |
|     class)](                      |     function)]                    |
| api/languages/cpp_api.html#_CPPv4 | (api/languages/cpp_api.html#_CPPv |
| N5cudaq22depolarization_channelE) | 4N5cudaq5ptsbe29ProbabilisticSamp |
| -   [cudaq::depol                 | lingStrategy29ProbabilisticSampli |
| arization_channel::num_parameters | ngStrategyENSt8optionalINSt8uint6 |
|     (C++                          | 4_tEEENSt8optionalINSt6size_tEEE) |
|     member)](api/languages/cp     | -   [cudaq::pts                   |
| p_api.html#_CPPv4N5cudaq22depolar | be::ProbabilisticSamplingStrategy |
| ization_channel14num_parametersE) | ::\~ProbabilisticSamplingStrategy |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4N5cudaq5ptsbe29P |
|     member)](api/languages        | robabilisticSamplingStrategyD0Ev) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -                                 |
| larization_channel11num_targetsE) | [cudaq::ptsbe::PTSBEExecutionData |
| -   [cudaq::detail (C++           |     (C++                          |
|     type)](api/languages/cp       |     struct)](ap                   |
| p_api.html#_CPPv4N5cudaq6detailE) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [cudaq::detail::future (C++   | cudaq5ptsbe18PTSBEExecutionDataE) |
|                                   | -   [cudaq::ptsbe::PTSBE          |
|   class)](api/languages/cpp_api.h | ExecutionData::count_instructions |
| tml#_CPPv4N5cudaq6detail6futureE) |     (C++                          |
| -                                 |     function)](api/l              |
|    [cudaq::detail::future::future | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq5ptsbe18PTSBEExecutionData18co |
|     functi                        | unt_instructionsE20TraceInstructi |
| on)](api/languages/cpp_api.html#_ | onTypeNSt8optionalINSt6stringEEE) |
| CPPv4N5cudaq6detail6future6future | -   [cudaq::ptsbe::P              |
| ERNSt6vectorI3JobEERNSt6stringERN | TSBEExecutionData::get_trajectory |
| St3mapINSt6stringENSt6stringEEE), |     (C++                          |
|     [\[1\]](api/lan               |     function                      |
| guages/cpp_api.html#_CPPv4N5cudaq | )](api/languages/cpp_api.html#_CP |
| 6detail6future6futureERR6future), | Pv4NK5cudaq5ptsbe18PTSBEExecution |
|     [\[2\]                        | Data14get_trajectoryENSt6size_tE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::ptsbe:                |
| v4N5cudaq6detail6future6futureEv) | :PTSBEExecutionData::instructions |
| -   [c                            |     (C++                          |
| udaq::detail::kernel_builder_base |     member)](api/languages/cp     |
|     (C++                          | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     class)](api/                  | TSBEExecutionData12instructionsE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::ptsbe:                |
| daq6detail19kernel_builder_baseE) | :PTSBEExecutionData::trajectories |
| -   [cudaq::detail::              |     (C++                          |
| kernel_builder_base::operator\<\< |     member)](api/languages/cp     |
|     (C++                          | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     function)](api/langu          | TSBEExecutionData12trajectoriesE) |
| ages/cpp_api.html#_CPPv4N5cudaq6d | -   [cudaq::ptsbe::PTSBEOptions   |
| etail19kernel_builder_baselsERNSt |     (C++                          |
| 7ostreamERK19kernel_builder_base) |     struc                         |
| -                                 | t)](api/languages/cpp_api.html#_C |
| [cudaq::detail::KernelBuilderType | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
|     (C++                          | -   [cudaq::ptsbe::PTSB           |
|     class)](ap                    | EOptions::include_sequential_data |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq6detail17KernelBuilderTypeE) |                                   |
| -   [cudaq::                      |    member)](api/languages/cpp_api |
| detail::KernelBuilderType::create | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
|     (C++                          | ptions23include_sequential_dataE) |
|     function                      | -   [cudaq::ptsb                  |
| )](api/languages/cpp_api.html#_CP | e::PTSBEOptions::max_trajectories |
| Pv4N5cudaq6detail17KernelBuilderT |     (C++                          |
| ype6createEPN4mlir11MLIRContextE) |     member)](api/languages/       |
| -   [cudaq::detail::Ker           | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| nelBuilderType::KernelBuilderType | 2PTSBEOptions16max_trajectoriesE) |
|     (C++                          | -   [cudaq::ptsbe::PT             |
|     function)](api/lan            | SBEOptions::return_execution_data |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 6detail17KernelBuilderType17Kerne |     member)](api/languages/cpp_a  |
| lBuilderTypeERRNSt8functionIFN4ml | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| ir4TypeEPN4mlir11MLIRContextEEEE) | EOptions21return_execution_dataE) |
| -   [cudaq::detector (C++         | -   [cudaq::pts                   |
|     function)](api                | be::PTSBEOptions::shot_allocation |
| /languages/cpp_api.html#_CPPv4IDp |     (C++                          |
| EN5cudaq8detectorEvDpRR8MeasArgs) |     member)](api/languages        |
| -   [cudaq::detectors (C++        | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     function)](api/languages/c    | 12PTSBEOptions15shot_allocationE) |
| pp_api.html#_CPPv4N5cudaq9detecto | -   [cud                          |
| rsERKNSt6vectorI14measure_resultE | aq::ptsbe::PTSBEOptions::strategy |
| ERKNSt6vectorI14measure_resultEE) |     (C++                          |
| -   [cudaq::diag_matrix_callback  |     member)](api/l                |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     class)                        | aq5ptsbe12PTSBEOptions8strategyE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::ptsbe::PTSBETrace     |
| v4N5cudaq20diag_matrix_callbackE) |     (C++                          |
| -   [cudaq::dyn (C++              |     t                             |
|     member)](api/languages        | ype)](api/languages/cpp_api.html# |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| -   [cudaq::ExecutionContext (C++ | -   [                             |
|     cl                            | cudaq::ptsbe::PTSSamplingStrategy |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16ExecutionContextE) |     class)](api                   |
| -   [c                            | /languages/cpp_api.html#_CPPv4N5c |
| udaq::ExecutionContext::asyncExec | udaq5ptsbe19PTSSamplingStrategyE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/                 | ptsbe::PTSSamplingStrategy::clone |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9asyncExecE) |     function)](api/languag        |
| -   [cud                          | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| aq::ExecutionContext::asyncResult | sbe19PTSSamplingStrategy5cloneEv) |
|     (C++                          | -   [cudaq::ptsbe::PTSSampl       |
|     member)](api/lan              | ingStrategy::generateTrajectories |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11asyncResultE) |     function)](api/               |
| -   [cudaq:                       | languages/cpp_api.html#_CPPv4NK5c |
| :ExecutionContext::batchIteration | udaq5ptsbe19PTSSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/langua           | detail10NoisePointEEENSt6size_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq16E | -   [cudaq:                       |
| xecutionContext14batchIterationE) | :ptsbe::PTSSamplingStrategy::name |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::canHandleObserve |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq5p |
|     member)](api/language         | tsbe19PTSSamplingStrategy4nameEv) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::ptsbe::PTSSampli      |
| cutionContext16canHandleObserveE) | ngStrategy::\~PTSSamplingStrategy |
| -   [cudaq::Executio              |     (C++                          |
| nContext::deferredKernelException |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     member)](api/languages/cpp_a  | q5ptsbe19PTSSamplingStrategyD0Ev) |
| pi.html#_CPPv4N5cudaq16ExecutionC | -   [cudaq::ptsbe::sample (C++    |
| ontext23deferredKernelExceptionE) |                                   |
| -   [cudaq::E                     |  function)](api/languages/cpp_api |
| xecutionContext::ExecutionContext | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
|     (C++                          | mpleE13sample_resultRK14sample_op |
|     func                          | tionsRR13QuantumKernelDpRR4Args), |
| tion)](api/languages/cpp_api.html |     [\[1\]](api                   |
| #_CPPv4N5cudaq16ExecutionContext1 | /languages/cpp_api.html#_CPPv4I0D |
| 6ExecutionContextERKNSt6stringE), | pEN5cudaq5ptsbe6sampleE13sample_r |
|     [\[1\]](api/languages/        | esultRKN5cudaq11noise_modelENSt6s |
| cpp_api.html#_CPPv4N5cudaq16Execu | ize_tERR13QuantumKernelDpRR4Args) |
| tionContext16ExecutionContextERKN | -   [cudaq::ptsbe::sample_async   |
| St6stringENSt6size_tENSt6size_tE) |     (C++                          |
| -   [cudaq::E                     |     function)](a                  |
| xecutionContext::expectationValue | pi/languages/cpp_api.html#_CPPv4I |
|     (C++                          | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
|     member)](api/language         | 9async_sample_resultRK14sample_op |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | tionsRR13QuantumKernelDpRR4Args), |
| cutionContext16expectationValueE) |     [\[1\]](api/languages/cp      |
| -   [cudaq::Execu                 | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| tionContext::explicitMeasurements | be12sample_asyncE19async_sample_r |
|     (C++                          | esultRKN5cudaq11noise_modelENSt6s |
|     member)](api/languages/cp     | ize_tERR13QuantumKernelDpRR4Args) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::ptsbe::sample_options |
| onContext20explicitMeasurementsE) |     (C++                          |
| -   [cuda                         |     struct)                       |
| q::ExecutionContext::futureResult | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq5ptsbe14sample_optionsE) |
|     member)](api/lang             | -   [cudaq::ptsbe::sample_result  |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     (C++                          |
| 6ExecutionContext12futureResultE) |     class                         |
| -   [cudaq::ExecutionContext      | )](api/languages/cpp_api.html#_CP |
| ::hasConditionalsOnMeasureResults | Pv4N5cudaq5ptsbe13sample_resultE) |
|     (C++                          | -   [cudaq::pts                   |
|     mem                           | be::sample_result::execution_data |
| ber)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16ExecutionContext31 |     function)](api/languages/c    |
| hasConditionalsOnMeasureResultsE) | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| -   [cudaq:                       | 3sample_result14execution_dataEv) |
| :ExecutionContext::inKernelLaunch | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::has_execution_data |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |                                   |
| xecutionContext14inKernelLaunchE) |    function)](api/languages/cpp_a |
| -   [cu                           | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| daq::ExecutionContext::kernelName | ple_result18has_execution_dataEv) |
|     (C++                          | -   [cudaq::pt                    |
|     member)](api/la               | sbe::sample_result::sample_result |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10kernelNameE) |     function)](api/l              |
| -   [cud                          | anguages/cpp_api.html#_CPPv4N5cud |
| aq::ExecutionContext::kernelTrace | aq5ptsbe13sample_result13sample_r |
|     (C++                          | esultERRN5cudaq13sample_resultE), |
|     member)](api/lan              |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq |  [\[1\]](api/languages/cpp_api.ht |
| 16ExecutionContext11kernelTraceE) | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| -   [cudaq:                       | sult13sample_resultERRN5cudaq13sa |
| :ExecutionContext::msm_dimensions | mple_resultE18PTSBEExecutionData) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     member)](api/langua           | sample_result::set_execution_data |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14msm_dimensionsE) |     function)](api/               |
| -   [cudaq::                      | languages/cpp_api.html#_CPPv4N5cu |
| ExecutionContext::msm_prob_err_id | daq5ptsbe13sample_result18set_exe |
|     (C++                          | cution_dataE18PTSBEExecutionData) |
|     member)](api/languag          | -   [cud                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | aq::ptsbe::ShotAllocationStrategy |
| ecutionContext15msm_prob_err_idE) |     (C++                          |
| -   [cudaq::Ex                    |     struct)](using                |
| ecutionContext::msm_probabilities | /examples/ptsbe.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe22ShotAllocationStrategyE) |
|     member)](api/languages        | -   [cudaq::ptsbe::ShotAllocatio  |
| /cpp_api.html#_CPPv4N5cudaq16Exec | nStrategy::ShotAllocationStrategy |
| utionContext17msm_probabilitiesE) |     (C++                          |
| -                                 |     function)                     |
|    [cudaq::ExecutionContext::name | ](using/examples/ptsbe.html#_CPPv |
|     (C++                          | 4N5cudaq5ptsbe22ShotAllocationStr |
|     member)]                      | ategy22ShotAllocationStrategyE4Ty |
| (api/languages/cpp_api.html#_CPPv | pedNSt8optionalINSt8uint64_tEEE), |
| 4N5cudaq16ExecutionContext4nameE) |     [\[1\                         |
| -   [cu                           | ]](using/examples/ptsbe.html#_CPP |
| daq::ExecutionContext::noiseModel | v4N5cudaq5ptsbe22ShotAllocationSt |
|     (C++                          | rategy22ShotAllocationStrategyEv) |
|     member)](api/la               | -   [cudaq::pt                    |
| nguages/cpp_api.html#_CPPv4N5cuda | sbe::ShotAllocationStrategy::Type |
| q16ExecutionContext10noiseModelE) |     (C++                          |
| -   [cudaq::Exe                   |     enum)](using/exam             |
| cutionContext::numberTrajectories | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|     (C++                          | be22ShotAllocationStrategy4TypeE) |
|     member)](api/languages/       | -   [cudaq::ptsbe::ShotAllocatio  |
| cpp_api.html#_CPPv4N5cudaq16Execu | nStrategy::Type::HIGH_WEIGHT_BIAS |
| tionContext18numberTrajectoriesE) |     (C++                          |
| -   [c                            |     enumerat                      |
| udaq::ExecutionContext::optResult | or)](using/examples/ptsbe.html#_C |
|     (C++                          | PPv4N5cudaq5ptsbe22ShotAllocation |
|     member)](api/                 | Strategy4Type16HIGH_WEIGHT_BIASE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::ptsbe::ShotAllocati   |
| daq16ExecutionContext9optResultE) | onStrategy::Type::LOW_WEIGHT_BIAS |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::qpuId |     enumera                       |
|     (C++                          | tor)](using/examples/ptsbe.html#_ |
|     member)](                     | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| api/languages/cpp_api.html#_CPPv4 | nStrategy4Type15LOW_WEIGHT_BIASE) |
| N5cudaq16ExecutionContext5qpuIdE) | -   [cudaq::ptsbe::ShotAlloc      |
| -   [cudaq                        | ationStrategy::Type::PROPORTIONAL |
| ::ExecutionContext::registerNames |     (C++                          |
|     (C++                          |     enum                          |
|     member)](api/langu            | erator)](using/examples/ptsbe.htm |
| ages/cpp_api.html#_CPPv4N5cudaq16 | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| ExecutionContext13registerNamesE) | tionStrategy4Type12PROPORTIONALE) |
| -   [cu                           | -   [cudaq::ptsbe::Shot           |
| daq::ExecutionContext::reorderIdx | AllocationStrategy::Type::UNIFORM |
|     (C++                          |     (C++                          |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda |   enumerator)](using/examples/pts |
| q16ExecutionContext10reorderIdxE) | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| -                                 | AllocationStrategy4Type7UNIFORME) |
|  [cudaq::ExecutionContext::result | -                                 |
|     (C++                          |   [cudaq::ptsbe::TraceInstruction |
|     member)](a                    |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     struct)](                     |
| 5cudaq16ExecutionContext6resultE) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | N5cudaq5ptsbe16TraceInstructionE) |
|   [cudaq::ExecutionContext::shots | -   [cudaq:                       |
|     (C++                          | :ptsbe::TraceInstruction::channel |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     member)](api/lang             |
| N5cudaq16ExecutionContext5shotsE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -   [cudaq::                      | ptsbe16TraceInstruction7channelE) |
| ExecutionContext::simulationState | -   [cudaq::                      |
|     (C++                          | ptsbe::TraceInstruction::controls |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     member)](api/langu            |
| ecutionContext15simulationStateE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -                                 | tsbe16TraceInstruction8controlsE) |
|    [cudaq::ExecutionContext::spin | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::name |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     member)](api/l                |
| 4N5cudaq16ExecutionContext4spinE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::                      | aq5ptsbe16TraceInstruction4nameE) |
| ExecutionContext::totalIterations | -   [cudaq                        |
|     (C++                          | ::ptsbe::TraceInstruction::params |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     member)](api/lan              |
| ecutionContext15totalIterationsE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [cudaq::ExecutionResult (C++  | 5ptsbe16TraceInstruction6paramsE) |
|     st                            | -   [cudaq:                       |
| ruct)](api/languages/cpp_api.html | :ptsbe::TraceInstruction::targets |
| #_CPPv4N5cudaq15ExecutionResultE) |     (C++                          |
| -   [cud                          |     member)](api/lang             |
| aq::ExecutionResult::appendResult | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     (C++                          | ptsbe16TraceInstruction7targetsE) |
|     functio                       | -   [cudaq::ptsbe::T              |
| n)](api/languages/cpp_api.html#_C | raceInstruction::TraceInstruction |
| PPv4N5cudaq15ExecutionResult12app |     (C++                          |
| endResultENSt6stringENSt6size_tE) |                                   |
| -   [cu                           |   function)](api/languages/cpp_ap |
| daq::ExecutionResult::deserialize | i.html#_CPPv4N5cudaq5ptsbe16Trace |
|     (C++                          | Instruction16TraceInstructionE20T |
|     function)                     | raceInstructionTypeNSt6stringENSt |
| ](api/languages/cpp_api.html#_CPP | 6vectorINSt6size_tEEENSt6vectorIN |
| v4N5cudaq15ExecutionResult11deser | St6size_tEEENSt6vectorIdEENSt8opt |
| ializeERNSt6vectorINSt6size_tEEE) | ionalIN5cudaq13kraus_channelEEE), |
| -   [cudaq:                       |     [\[1\]](api/languages/cpp_a   |
| :ExecutionResult::ExecutionResult | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
|     (C++                          | eInstruction16TraceInstructionEv) |
|     functio                       | -   [cud                          |
| n)](api/languages/cpp_api.html#_C | aq::ptsbe::TraceInstruction::type |
| PPv4N5cudaq15ExecutionResult15Exe |     (C++                          |
| cutionResultE16CountsDictionary), |     member)](api/l                |
|     [\[1\]](api/lan               | anguages/cpp_api.html#_CPPv4N5cud |
| guages/cpp_api.html#_CPPv4N5cudaq | aq5ptsbe16TraceInstruction4typeE) |
| 15ExecutionResult15ExecutionResul | -   [c                            |
| tE16CountsDictionaryNSt6stringE), | udaq::ptsbe::TraceInstructionType |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     enum)](api/                   |
| Pv4N5cudaq15ExecutionResult15Exec | languages/cpp_api.html#_CPPv4N5cu |
| utionResultE16CountsDictionaryd), | daq5ptsbe20TraceInstructionTypeE) |
|                                   | -   [cudaq::                      |
|    [\[3\]](api/languages/cpp_api. | ptsbe::TraceInstructionType::Gate |
| html#_CPPv4N5cudaq15ExecutionResu |     (C++                          |
| lt15ExecutionResultENSt6stringE), |     enumerator)](api/langu        |
|     [\[4\                         | ages/cpp_api.html#_CPPv4N5cudaq5p |
| ]](api/languages/cpp_api.html#_CP | tsbe20TraceInstructionType4GateE) |
| Pv4N5cudaq15ExecutionResult15Exec | -   [cudaq::ptsbe::               |
| utionResultERK15ExecutionResult), | TraceInstructionType::Measurement |
|     [\[5\]](api/language          |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |                                   |
| cutionResult15ExecutionResultEd), |    enumerator)](api/languages/cpp |
|     [\[6\]](api/languag           | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | aceInstructionType11MeasurementE) |
| ecutionResult15ExecutionResultEv) | -   [cudaq::p                     |
| -   [                             | tsbe::TraceInstructionType::Noise |
| cudaq::ExecutionResult::operator= |     (C++                          |
|     (C++                          |     enumerator)](api/langua       |
|     function)](api/languages/     | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| cpp_api.html#_CPPv4N5cudaq15Execu | sbe20TraceInstructionType5NoiseE) |
| tionResultaSERK15ExecutionResult) | -   [                             |
| -   [c                            | cudaq::ptsbe::TrajectoryPredicate |
| udaq::ExecutionResult::operator== |     (C++                          |
|     (C++                          |     type)](api                    |
|     function)](api/languages/c    | /languages/cpp_api.html#_CPPv4N5c |
| pp_api.html#_CPPv4NK5cudaq15Execu | udaq5ptsbe19TrajectoryPredicateE) |
| tionResulteqERK15ExecutionResult) | -   [cudaq::QPU (C++              |
| -   [cud                          |     class)](api/languages         |
| aq::ExecutionResult::registerName | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     (C++                          | -   [cudaq::QPU::beginExecution   |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function                      |
| 15ExecutionResult12registerNameE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq                        | Pv4N5cudaq3QPU14beginExecutionEv) |
| ::ExecutionResult::sequentialData | -   [cuda                         |
|     (C++                          | q::QPU::configureExecutionContext |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     funct                         |
| ExecutionResult14sequentialDataE) | ion)](api/languages/cpp_api.html# |
| -   [                             | _CPPv4NK5cudaq3QPU25configureExec |
| cudaq::ExecutionResult::serialize | utionContextER16ExecutionContext) |
|     (C++                          | -   [cudaq::QPU::endExecution     |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4NK5cu |     functi                        |
| daq15ExecutionResult9serializeEv) | on)](api/languages/cpp_api.html#_ |
| -   [cudaq::fermion_handler (C++  | CPPv4N5cudaq3QPU12endExecutionEv) |
|     c                             | -   [cudaq::QPU::enqueue (C++     |
| lass)](api/languages/cpp_api.html |     function)](ap                 |
| #_CPPv4N5cudaq15fermion_handlerE) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [cudaq::fermion_op (C++       | cudaq3QPU7enqueueER11QuantumTask) |
|     type)](api/languages/cpp_api  | -   [cud                          |
| .html#_CPPv4N5cudaq10fermion_opE) | aq::QPU::finalizeExecutionContext |
| -   [cudaq::fermion_op_term (C++  |     (C++                          |
|                                   |     func                          |
| type)](api/languages/cpp_api.html | tion)](api/languages/cpp_api.html |
| #_CPPv4N5cudaq15fermion_op_termE) | #_CPPv4NK5cudaq3QPU24finalizeExec |
| -   [cudaq::FermioniqQPU (C++     | utionContextER16ExecutionContext) |
|                                   | -   [cudaq::QPU::getCompileTarget |
|   class)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq12FermioniqQPUE) |     function)](api/languages/c    |
| -   [cudaq::get_state (C++        | pp_api.html#_CPPv4N5cudaq3QPU16ge |
|                                   | tCompileTargetERK13sample_policy) |
|    function)](api/languages/cpp_a | -   [cudaq::QPU::getConnectivity  |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     (C++                          |
| ateEDaRR13QuantumKernelDpRR4Args) |     function)                     |
| -   [cudaq::GPUEmulatedQPU (C++   | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq3QPU15getConnectivityEv) |
| class)](api/languages/cpp_api.htm | -                                 |
| l#_CPPv4N5cudaq14GPUEmulatedQPUE) | [cudaq::QPU::getExecutionThreadId |
| -   [cudaq::gradient (C++         |     (C++                          |
|     class)](api/languages/cpp_    |     function)](api/               |
| api.html#_CPPv4N5cudaq8gradientE) | languages/cpp_api.html#_CPPv4NK5c |
| -   [cudaq::gradient::clone (C++  | udaq3QPU20getExecutionThreadIdEv) |
|     fun                           | -   [cudaq::QPU::getNumQubits     |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     functi                        |
| -   [cudaq::gradient::compute     | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     function)](api/language       | -   [                             |
| s/cpp_api.html#_CPPv4N5cudaq8grad | cudaq::QPU::getRemoteCapabilities |
| ient7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api/l              |
|     [\[1\]](ap                    | anguages/cpp_api.html#_CPPv4NK5cu |
| i/languages/cpp_api.html#_CPPv4N5 | daq3QPU21getRemoteCapabilitiesEv) |
| cudaq8gradient7computeERKNSt6vect | -   [cudaq::QPU::isEmulated (C++  |
| orIdEERNSt6vectorIdEERK7spin_opd) |     func                          |
| -   [cudaq::gradient::gradient    | tion)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     function)](api/lang           | -   [cudaq::QPU::isSimulator (C++ |
| uages/cpp_api.html#_CPPv4I00EN5cu |     funct                         |
| daq8gradient8gradientER7KernelT), | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|    [\[1\]](api/languages/cpp_api. | -   [cudaq::QPU::onRandomSeedSet  |
| html#_CPPv4I00EN5cudaq8gradient8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api/lang           |
|     [\[2\                         | uages/cpp_api.html#_CPPv4N5cudaq3 |
| ]](api/languages/cpp_api.html#_CP | QPU15onRandomSeedSetENSt6size_tE) |
| Pv4I00EN5cudaq8gradient8gradientE | -   [cudaq::QPU::QPU (C++         |
| RR13QuantumKernelRR10ArgsMapper), |     functio                       |
|     [\[3                          | n)](api/languages/cpp_api.html#_C |
| \]](api/languages/cpp_api.html#_C | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| PPv4N5cudaq8gradient8gradientERRN |                                   |
| St8functionIFvNSt6vectorIdEEEEE), |  [\[1\]](api/languages/cpp_api.ht |
|     [\[                           | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| 4\]](api/languages/cpp_api.html#_ |     [\[2\]](api/languages/cpp_    |
| CPPv4N5cudaq8gradient8gradientEv) | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| -   [cudaq::gradient::setArgs     | -   [cudaq::QPU::setId (C++       |
|     (C++                          |     function                      |
|     fu                            | )](api/languages/cpp_api.html#_CP |
| nction)](api/languages/cpp_api.ht | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | -   [cudaq::QPU::setShots (C++    |
| tArgsEvR13QuantumKernelDpRR4Args) |     f                             |
| -   [cudaq::gradient::setKernel   | unction)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
|     function)](api/languages/c    | -   [cudaq::                      |
| pp_api.html#_CPPv4I0EN5cudaq8grad | QPU::supportsExplicitMeasurements |
| ient9setKernelEvR13QuantumKernel) |     (C++                          |
| -   [cud                          |     function)](api/languag        |
| aq::gradients::central_difference | es/cpp_api.html#_CPPv4N5cudaq3QPU |
|     (C++                          | 28supportsExplicitMeasurementsEv) |
|     class)](api/la                | -   [cudaq::QPU::\~QPU (C++       |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languages/cp   |
| q9gradients18central_differenceE) | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| -   [cudaq::gra                   | -   [cudaq::QPUState (C++         |
| dients::central_difference::clone |     class)](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4N5cudaq8QPUStateE) |
|     function)](api/languages      | -   [cudaq::qreg (C++             |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     class)](api/lan               |
| ents18central_difference5cloneEv) | guages/cpp_api.html#_CPPv4I_NSt6s |
| -   [cudaq::gradi                 | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| ents::central_difference::compute | -   [cudaq::qreg::back (C++       |
|     (C++                          |     function)                     |
|     function)](                   | ](api/languages/cpp_api.html#_CPP |
| api/languages/cpp_api.html#_CPPv4 | v4N5cudaq4qreg4backENSt6size_tE), |
| N5cudaq9gradients18central_differ |     [\[1\]](api/languages/cpp_ap  |
| ence7computeERKNSt6vectorIdEERKNS | i.html#_CPPv4N5cudaq4qreg4backEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::qreg::begin (C++      |
|                                   |                                   |
|   [\[1\]](api/languages/cpp_api.h |  function)](api/languages/cpp_api |
| tml#_CPPv4N5cudaq9gradients18cent | .html#_CPPv4N5cudaq4qreg5beginEv) |
| ral_difference7computeERKNSt6vect | -   [cudaq::qreg::clear (C++      |
| orIdEERNSt6vectorIdEERK7spin_opd) |                                   |
| -   [cudaq::gradie                |  function)](api/languages/cpp_api |
| nts::central_difference::gradient | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     (C++                          | -   [cudaq::qreg::front (C++      |
|     functio                       |     function)]                    |
| n)](api/languages/cpp_api.html#_C | (api/languages/cpp_api.html#_CPPv |
| PPv4I00EN5cudaq9gradients18centra | 4N5cudaq4qreg5frontENSt6size_tE), |
| l_difference8gradientER7KernelT), |     [\[1\]](api/languages/cpp_api |
|     [\[1\]](api/langua            | .html#_CPPv4N5cudaq4qreg5frontEv) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::qreg::operator\[\]    |
| q9gradients18central_difference8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     functi                        |
|     [\[2\]](api/languages/cpp_    | on)](api/languages/cpp_api.html#_ |
| api.html#_CPPv4I00EN5cudaq9gradie | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| nts18central_difference8gradientE | -   [cudaq::qreg::qreg (C++       |
| RR13QuantumKernelRR10ArgsMapper), |     function)                     |
|     [\[3\]](api/languages/cpp     | ](api/languages/cpp_api.html#_CPP |
| _api.html#_CPPv4N5cudaq9gradients | v4N5cudaq4qreg4qregENSt6size_tE), |
| 18central_difference8gradientERRN |     [\[1\]](api/languages/cpp_ap  |
| St8functionIFvNSt6vectorIdEEEEE), | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qreg::size (C++       |
| p_api.html#_CPPv4N5cudaq9gradient |                                   |
| s18central_difference8gradientEv) |  function)](api/languages/cpp_api |
| -   [cud                          | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| aq::gradients::forward_difference | -   [cudaq::qreg::slice (C++      |
|     (C++                          |     function)](api/langu          |
|     class)](api/la                | ages/cpp_api.html#_CPPv4N5cudaq4q |
| nguages/cpp_api.html#_CPPv4N5cuda | reg5sliceENSt6size_tENSt6size_tE) |
| q9gradients18forward_differenceE) | -   [cudaq::qreg::value_type (C++ |
| -   [cudaq::gra                   |                                   |
| dients::forward_difference::clone | type)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq4qreg10value_typeE) |
|     function)](api/languages      | -   [cudaq::qspan (C++            |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     class)](api/lang              |
| ents18forward_difference5cloneEv) | uages/cpp_api.html#_CPPv4I_NSt6si |
| -   [cudaq::gradi                 | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| ents::forward_difference::compute | -   [cudaq::QuakeValue (C++       |
|     (C++                          |     class)](api/languages/cpp_api |
|     function)](                   | .html#_CPPv4N5cudaq10QuakeValueE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::Q                     |
| N5cudaq9gradients18forward_differ | uakeValue::canValidateNumElements |
| ence7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api/languages      |
|                                   | /cpp_api.html#_CPPv4N5cudaq10Quak |
|   [\[1\]](api/languages/cpp_api.h | eValue22canValidateNumElementsEv) |
| tml#_CPPv4N5cudaq9gradients18forw | -                                 |
| ard_difference7computeERKNSt6vect |  [cudaq::QuakeValue::constantSize |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradie                |     function)](api                |
| nts::forward_difference::gradient | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValue12constantSizeEv) |
|     functio                       | -   [cudaq::QuakeValue::dump (C++ |
| n)](api/languages/cpp_api.html#_C |     function)](api/lan            |
| PPv4I00EN5cudaq9gradients18forwar | guages/cpp_api.html#_CPPv4N5cudaq |
| d_difference8gradientER7KernelT), | 10QuakeValue4dumpERNSt7ostreamE), |
|     [\[1\]](api/langua            |     [\                            |
| ges/cpp_api.html#_CPPv4I00EN5cuda | [1\]](api/languages/cpp_api.html# |
| q9gradients18forward_difference8g | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq                        |
|     [\[2\]](api/languages/cpp_    | ::QuakeValue::getRequiredElements |
| api.html#_CPPv4I00EN5cudaq9gradie |     (C++                          |
| nts18forward_difference8gradientE |     function)](api/langua         |
| RR13QuantumKernelRR10ArgsMapper), | ges/cpp_api.html#_CPPv4N5cudaq10Q |
|     [\[3\]](api/languages/cpp     | uakeValue19getRequiredElementsEv) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::QuakeValue::getValue  |
| 18forward_difference8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)]                    |
|     [\[4\]](api/languages/cp      | (api/languages/cpp_api.html#_CPPv |
| p_api.html#_CPPv4N5cudaq9gradient | 4NK5cudaq10QuakeValue8getValueEv) |
| s18forward_difference8gradientEv) | -   [cudaq::QuakeValue::inverse   |
| -   [                             |     (C++                          |
| cudaq::gradients::parameter_shift |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     class)](api                   | v4NK5cudaq10QuakeValue7inverseEv) |
| /languages/cpp_api.html#_CPPv4N5c | -                                 |
| udaq9gradients15parameter_shiftE) |    [cudaq::QuakeValue::isSequence |
| -   [cudaq::                      |     (C++                          |
| gradients::parameter_shift::clone |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function)](api/langua         | 5cudaq10QuakeValue10isSequenceEv) |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | -                                 |
| adients15parameter_shift5cloneEv) |    [cudaq::QuakeValue::operator\* |
| -   [cudaq::gr                    |     (C++                          |
| adients::parameter_shift::compute |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function                      | udaq10QuakeValuemlE10QuakeValue), |
| )](api/languages/cpp_api.html#_CP |                                   |
| Pv4N5cudaq9gradients15parameter_s | [\[1\]](api/languages/cpp_api.htm |
| hift7computeERKNSt6vectorIdEERKNS | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::QuakeValue::operator+ |
|     [\[1\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4N5cudaq9gradients15p |     function)](api                |
| arameter_shift7computeERKNSt6vect | /languages/cpp_api.html#_CPPv4N5c |
| orIdEERNSt6vectorIdEERK7spin_opd) | udaq10QuakeValueplE10QuakeValue), |
| -   [cudaq::gra                   |     [                             |
| dients::parameter_shift::gradient | \[1\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     func                          |                                   |
| tion)](api/languages/cpp_api.html | [\[2\]](api/languages/cpp_api.htm |
| #_CPPv4I00EN5cudaq9gradients15par | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| ameter_shift8gradientER7KernelT), | -   [cudaq::QuakeValue::operator- |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4I00EN5c |     function)](api                |
| udaq9gradients15parameter_shift8g | /languages/cpp_api.html#_CPPv4N5c |
| radientER7KernelTRR10ArgsMapper), | udaq10QuakeValuemiE10QuakeValue), |
|     [\[2\]](api/languages/c       |     [                             |
| pp_api.html#_CPPv4I00EN5cudaq9gra | \[1\]](api/languages/cpp_api.html |
| dients15parameter_shift8gradientE | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| RR13QuantumKernelRR10ArgsMapper), |     [                             |
|     [\[3\]](api/languages/        | \[2\]](api/languages/cpp_api.html |
| cpp_api.html#_CPPv4N5cudaq9gradie | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| nts15parameter_shift8gradientERRN |                                   |
| St8functionIFvNSt6vectorIdEEEEE), | [\[3\]](api/languages/cpp_api.htm |
|     [\[4\]](api/languages         | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QuakeValue::operator/ |
| ents15parameter_shift8gradientEv) |     (C++                          |
| -   [cudaq::kernel_builder (C++   |     function)](api                |
|     clas                          | /languages/cpp_api.html#_CPPv4N5c |
| s)](api/languages/cpp_api.html#_C | udaq10QuakeValuedvE10QuakeValue), |
| PPv4IDpEN5cudaq14kernel_builderE) |                                   |
| -   [c                            | [\[1\]](api/languages/cpp_api.htm |
| udaq::kernel_builder::constantVal | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
|     (C++                          | -                                 |
|     function)](api/la             |  [cudaq::QuakeValue::operator\[\] |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14kernel_builder11constantValEd) |     function)](api                |
| -                                 | /languages/cpp_api.html#_CPPv4N5c |
|  [cudaq::kernel_builder::detector | udaq10QuakeValueixEKNSt6size_tE), |
|     (C++                          |     [\[1\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|    function)](api/languages/cpp_a | daq10QuakeValueixERK10QuakeValue) |
| pi.html#_CPPv4IDpEN5cudaq14kernel | -                                 |
| _builder8detectorEvDpRR8MeasArgs) |    [cudaq::QuakeValue::QuakeValue |
| -                                 |     (C++                          |
| [cudaq::kernel_builder::detectors |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     func                          | akeValue10QuakeValueERN4mlir20Imp |
| tion)](api/languages/cpp_api.html | licitLocOpBuilderEN4mlir5ValueE), |
| #_CPPv4N5cudaq14kernel_builder9de |     [\[1\]                        |
| tectorsE10QuakeValue10QuakeValue) | ](api/languages/cpp_api.html#_CPP |
| -   [cu                           | v4N5cudaq10QuakeValue10QuakeValue |
| daq::kernel_builder::getArguments | ERN4mlir20ImplicitLocOpBuilderEd) |
|     (C++                          | -   [cudaq::QuakeValue::size (C++ |
|     function)](api/lan            |     funct                         |
| guages/cpp_api.html#_CPPv4N5cudaq | ion)](api/languages/cpp_api.html# |
| 14kernel_builder12getArgumentsEv) | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| -   [cu                           | -   [cudaq::QuakeValue::slice     |
| daq::kernel_builder::getNumParams |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     function)](api/lan            | api.html#_CPPv4N5cudaq10QuakeValu |
| guages/cpp_api.html#_CPPv4N5cudaq | e5sliceEKNSt6size_tEKNSt6size_tE) |
| 14kernel_builder12getNumParamsEv) | -   [cudaq::quantum_platform (C++ |
| -   [cud                          |     cl                            |
| aq::kernel_builder::isArgSequence | ass)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq16quantum_platformE) |
|     function)](api/languages/cpp_ | -   [cudaq:                       |
| api.html#_CPPv4N5cudaq14kernel_bu | :quantum_platform::beginExecution |
| ilder13isArgSequenceENSt6size_tE) |     (C++                          |
| -   [cuda                         |     function)](api/languag        |
| q::kernel_builder::kernel_builder | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14beginExecutionEv) |
|     function)](api/languages/cpp  | -   [cudaq::quantum_pl            |
| _api.html#_CPPv4N5cudaq14kernel_b | atform::configureExecutionContext |
| uilder14kernel_builderERNSt6vecto |     (C++                          |
| rIN6detail17KernelBuilderTypeEEE) |     function)](api/lang           |
| -   [cudaq::k                     | uages/cpp_api.html#_CPPv4NK5cudaq |
| ernel_builder::logical_observable | 16quantum_platform25configureExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     function)                     | -   [cuda                         |
| ](api/languages/cpp_api.html#_CPP | q::quantum_platform::connectivity |
| v4IDpEN5cudaq14kernel_builder18lo |     (C++                          |
| gical_observableEvDpRR8MeasArgs), |     function)](api/langu          |
|     [\[1\]](ap                    | ages/cpp_api.html#_CPPv4N5cudaq16 |
| i/languages/cpp_api.html#_CPPv4N5 | quantum_platform12connectivityEv) |
| cudaq14kernel_builder18logical_ob | -   [cuda                         |
| servableE10QuakeValueNSt6size_tE) | q::quantum_platform::endExecution |
| -   [cudaq::kernel_builder::name  |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)                     | ages/cpp_api.html#_CPPv4N5cudaq16 |
| ](api/languages/cpp_api.html#_CPP | quantum_platform12endExecutionEv) |
| v4N5cudaq14kernel_builder4nameEv) | -   [cudaq::q                     |
| -                                 | uantum_platform::enqueueAsyncTask |
|    [cudaq::kernel_builder::qalloc |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     function)](api/language       | cpp_api.html#_CPPv4N5cudaq16quant |
| s/cpp_api.html#_CPPv4N5cudaq14ker | um_platform16enqueueAsyncTaskEKNS |
| nel_builder6qallocE10QuakeValue), | t6size_tER19KernelExecutionTask), |
|     [\[1\]](api/language          |     [\[1\]](api/languag           |
| s/cpp_api.html#_CPPv4N5cudaq14ker | es/cpp_api.html#_CPPv4N5cudaq16qu |
| nel_builder6qallocEKNSt6size_tE), | antum_platform16enqueueAsyncTaskE |
|     [\[2                          | KNSt6size_tERNSt8functionIFvvEEE) |
| \]](api/languages/cpp_api.html#_C | -   [cudaq::quantum_p             |
| PPv4N5cudaq14kernel_builder6qallo | latform::finalizeExecutionContext |
| cERNSt6vectorINSt7complexIdEEEE), |     (C++                          |
|     [\[3\]](                      |     function)](api/languages/c    |
| api/languages/cpp_api.html#_CPPv4 | pp_api.html#_CPPv4NK5cudaq16quant |
| N5cudaq14kernel_builder6qallocEv) | um_platform24finalizeExecutionCon |
| -   [cudaq::kernel_builder::swap  | textERN5cudaq16ExecutionContextE) |
|     (C++                          | -   [cudaq::qua                   |
|     function)](api/language       | ntum_platform::get_codegen_config |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     (C++                          |
| 4kernel_builder4swapEvRK10QuakeVa |     function)](api/languages/c    |
| lueRK10QuakeValueRK10QuakeValue), | pp_api.html#_CPPv4N5cudaq16quantu |
|                                   | m_platform18get_codegen_configEv) |
| [\[1\]](api/languages/cpp_api.htm | -   [cuda                         |
| l#_CPPv4I00EN5cudaq14kernel_build | q::quantum_platform::get_exec_ctx |
| er4swapEvRKNSt6vectorI10QuakeValu |     (C++                          |
| eEERK10QuakeValueRK10QuakeValue), |     function)](api/langua         |
|                                   | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| [\[2\]](api/languages/cpp_api.htm | quantum_platform12get_exec_ctxEv) |
| l#_CPPv4N5cudaq14kernel_builder4s | -   [c                            |
| wapERK10QuakeValueRK10QuakeValue) | udaq::quantum_platform::get_noise |
| -   [cudaq::KernelExecutionTask   |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     type                          | pp_api.html#_CPPv4N5cudaq16quantu |
| )](api/languages/cpp_api.html#_CP | m_platform9get_noiseENSt6size_tE) |
| Pv4N5cudaq19KernelExecutionTaskE) | -   [cudaq:                       |
| -   [cudaq::KernelThunkResultType | :quantum_platform::get_num_qubits |
|     (C++                          |     (C++                          |
|     struct)]                      |                                   |
| (api/languages/cpp_api.html#_CPPv | function)](api/languages/cpp_api. |
| 4N5cudaq21KernelThunkResultTypeE) | html#_CPPv4NK5cudaq16quantum_plat |
| -   [cudaq::KernelThunkType (C++  | form14get_num_qubitsENSt6size_tE) |
|                                   | -   [cudaq::quantum_              |
| type)](api/languages/cpp_api.html | platform::get_remote_capabilities |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     (C++                          |
| -   [cudaq::kraus_channel (C++    |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
|  class)](api/languages/cpp_api.ht | v4NK5cudaq16quantum_platform23get |
| ml#_CPPv4N5cudaq13kraus_channelE) | _remote_capabilitiesENSt6size_tE) |
| -   [cudaq::kraus_channel::empty  | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_runtime_target |
|     function)]                    |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/languages/cp   |
| 4NK5cudaq13kraus_channel5emptyEv) | p_api.html#_CPPv4NK5cudaq16quantu |
| -   [cudaq::kraus_c               | m_platform18get_runtime_targetEv) |
| hannel::generateUnitaryParameters | -   [cud                          |
|     (C++                          | aq::quantum_platform::is_emulated |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |                                   |
| pi.html#_CPPv4N5cudaq13kraus_chan |    function)](api/languages/cpp_a |
| nel25generateUnitaryParametersEv) | pi.html#_CPPv4NK5cudaq16quantum_p |
| -                                 | latform11is_emulatedENSt6size_tE) |
|    [cudaq::kraus_channel::get_ops | -   [cudaq::                      |
|     (C++                          | quantum_platform::is_library_mode |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/languages      |
| K5cudaq13kraus_channel7get_opsEv) | /cpp_api.html#_CPPv4NK5cudaq16qua |
| -   [cud                          | ntum_platform15is_library_modeEv) |
| aq::kraus_channel::identity_flags | -   [c                            |
|     (C++                          | udaq::quantum_platform::is_remote |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languages/cp   |
| 13kraus_channel14identity_flagsE) | p_api.html#_CPPv4NK5cudaq16quantu |
| -   [cud                          | m_platform9is_remoteENSt6size_tE) |
| aq::kraus_channel::is_identity_op | -   [cuda                         |
|     (C++                          | q::quantum_platform::is_simulator |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |                                   |
| pi.html#_CPPv4NK5cudaq13kraus_cha |   function)](api/languages/cpp_ap |
| nnel14is_identity_opENSt6size_tE) | i.html#_CPPv4NK5cudaq16quantum_pl |
| -   [cudaq::                      | atform12is_simulatorENSt6size_tE) |
| kraus_channel::is_unitary_mixture | -   [c                            |
|     (C++                          | udaq::quantum_platform::launchVQE |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4NK5cudaq13kra |     function)](                   |
| us_channel18is_unitary_mixtureEv) | api/languages/cpp_api.html#_CPPv4 |
| -   [cu                           | N5cudaq16quantum_platform9launchV |
| daq::kraus_channel::kraus_channel | QEEKNSt6stringEPKvPN5cudaq8gradie |
|     (C++                          | ntERKN5cudaq7spin_opERN5cudaq9opt |
|     function)](api/lang           | imizerEKiKNSt6size_tENSt6size_tE) |
| uages/cpp_api.html#_CPPv4IDpEN5cu | -   [cudaq:                       |
| daq13kraus_channel13kraus_channel | :quantum_platform::list_platforms |
| EDpRRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)](api/languag        |
|  [\[1\]](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ml#_CPPv4N5cudaq13kraus_channel13 | antum_platform14list_platformsEv) |
| kraus_channelERK13kraus_channel), | -                                 |
|     [\[2\]                        |    [cudaq::quantum_platform::name |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq13kraus_channel13kraus_c |     function)](a                  |
| hannelERKNSt6vectorI8kraus_opEE), | pi/languages/cpp_api.html#_CPPv4N |
|     [\[3\]                        | K5cudaq16quantum_platform4nameEv) |
| ](api/languages/cpp_api.html#_CPP | -   [                             |
| v4N5cudaq13kraus_channel13kraus_c | cudaq::quantum_platform::num_qpus |
| hannelERRNSt6vectorI8kraus_opEE), |     (C++                          |
|     [\[4\]](api/lan               |     function)](api/l              |
| guages/cpp_api.html#_CPPv4N5cudaq | anguages/cpp_api.html#_CPPv4NK5cu |
| 13kraus_channel13kraus_channelEv) | daq16quantum_platform8num_qpusEv) |
| -                                 | -   [cudaq::                      |
| [cudaq::kraus_channel::noise_type | quantum_platform::onRandomSeedSet |
|     (C++                          |     (C++                          |
|     member)](api                  |                                   |
| /languages/cpp_api.html#_CPPv4N5c | function)](api/languages/cpp_api. |
| udaq13kraus_channel10noise_typeE) | html#_CPPv4N5cudaq16quantum_platf |
| -                                 | orm15onRandomSeedSetENSt6size_tE) |
|   [cudaq::kraus_channel::op_names | -   [cudaq:                       |
|     (C++                          | :quantum_platform::reset_exec_ctx |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languag        |
| N5cudaq13kraus_channel8op_namesE) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -                                 | antum_platform14reset_exec_ctxEv) |
|  [cudaq::kraus_channel::operator= | -   [cud                          |
|     (C++                          | aq::quantum_platform::reset_noise |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     function)](api/languages/cpp_ |
| raus_channelaSERK13kraus_channel) | api.html#_CPPv4N5cudaq16quantum_p |
| -   [c                            | latform11reset_noiseENSt6size_tE) |
| udaq::kraus_channel::operator\[\] | -   [cuda                         |
|     (C++                          | q::quantum_platform::set_exec_ctx |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4N5cud |     funct                         |
| aq13kraus_channelixEKNSt6size_tE) | ion)](api/languages/cpp_api.html# |
| -                                 | _CPPv4N5cudaq16quantum_platform12 |
| [cudaq::kraus_channel::parameters | set_exec_ctxEP16ExecutionContext) |
|     (C++                          | -   [c                            |
|     member)](api                  | udaq::quantum_platform::set_noise |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel10parametersE) |     function                      |
| -   [cudaq::krau                  | )](api/languages/cpp_api.html#_CP |
| s_channel::populateDefaultOpNames | Pv4N5cudaq16quantum_platform9set_ |
|     (C++                          | noiseEPK11noise_modelNSt6size_tE) |
|     function)](api/languages/cp   | -   [cudaq::quantum_platfor       |
| p_api.html#_CPPv4N5cudaq13kraus_c | m::supports_explicit_measurements |
| hannel22populateDefaultOpNamesEv) |     (C++                          |
| -   [cu                           |     function)](api/l              |
| daq::kraus_channel::probabilities | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq16quantum_platform30supports_e |
|     member)](api/la               | xplicit_measurementsENSt6size_tE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::quantum_pla           |
| q13kraus_channel13probabilitiesE) | tform::supports_task_distribution |
| -                                 |     (C++                          |
|  [cudaq::kraus_channel::push_back |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function)](api                | ml#_CPPv4NK5cudaq16quantum_platfo |
| /languages/cpp_api.html#_CPPv4N5c | rm26supports_task_distributionEv) |
| udaq13kraus_channel9push_backE8kr | -   [cudaq::quantum               |
| aus_opNSt8optionalINSt6stringEEE) | _platform::with_execution_context |
| -   [cudaq::kraus_channel::size   |     (C++                          |
|     (C++                          |     function)                     |
|     function)                     | ](api/languages/cpp_api.html#_CPP |
| ](api/languages/cpp_api.html#_CPP | v4I0DpEN5cudaq16quantum_platform2 |
| v4NK5cudaq13kraus_channel4sizeEv) | 2with_execution_contextEDaR16Exec |
| -   [                             | utionContextRR8CallableDpRR4Args) |
| cudaq::kraus_channel::unitary_ops | -   [cudaq::QuantumTask (C++      |
|     (C++                          |     type)](api/languages/cpp_api. |
|     member)](api/                 | html#_CPPv4N5cudaq11QuantumTaskE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::qubit (C++            |
| daq13kraus_channel11unitary_opsE) |     type)](api/languages/c        |
| -   [cudaq::kraus_op (C++         | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     struct)](api/languages/cpp_   | -   [cudaq::QubitConnectivity     |
| api.html#_CPPv4N5cudaq8kraus_opE) |     (C++                          |
| -   [cudaq::kraus_op::adjoint     |     ty                            |
|     (C++                          | pe)](api/languages/cpp_api.html#_ |
|     functi                        | CPPv4N5cudaq17QubitConnectivityE) |
| on)](api/languages/cpp_api.html#_ | -   [cudaq::QubitEdge (C++        |
| CPPv4NK5cudaq8kraus_op7adjointEv) |     type)](api/languages/cpp_a    |
| -   [cudaq::kraus_op::data (C++   | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|                                   | -   [cudaq::qudit (C++            |
|  member)](api/languages/cpp_api.h |     clas                          |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | s)](api/languages/cpp_api.html#_C |
| -   [cudaq::kraus_op::kraus_op    | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     (C++                          | -   [cudaq::qudit::qudit (C++     |
|     func                          |                                   |
| tion)](api/languages/cpp_api.html | function)](api/languages/cpp_api. |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | html#_CPPv4N5cudaq5qudit5quditEv) |
| opERRNSt16initializer_listI1TEE), | -   [cudaq::QuEraRemoteRESTQPU    |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     clas                          |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | s)](api/languages/cpp_api.html#_C |
| pENSt6vectorIN5cudaq7complexEEE), | PPv4N5cudaq18QuEraRemoteRESTQPUE) |
|     [\[2\]](api/l                 | -   [cudaq::qvector (C++          |
| anguages/cpp_api.html#_CPPv4N5cud |     class)                        |
| aq8kraus_op8kraus_opERK8kraus_op) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus_op::nCols (C++  | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|                                   | -   [cudaq::qvector::back (C++    |
| member)](api/languages/cpp_api.ht |     function)](a                  |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::kraus_op::nRows (C++  | 5cudaq7qvector4backENSt6size_tE), |
|                                   |                                   |
| member)](api/languages/cpp_api.ht |   [\[1\]](api/languages/cpp_api.h |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | tml#_CPPv4N5cudaq7qvector4backEv) |
| -   [cudaq::kraus_op::operator=   | -   [cudaq::qvector::begin (C++   |
|     (C++                          |     fu                            |
|     function)                     | nction)](api/languages/cpp_api.ht |
| ](api/languages/cpp_api.html#_CPP | ml#_CPPv4N5cudaq7qvector5beginEv) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cudaq::qvector::clear (C++   |
| -   [cudaq::kraus_op::precision   |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     memb                          | ml#_CPPv4N5cudaq7qvector5clearEv) |
| er)](api/languages/cpp_api.html#_ | -   [cudaq::qvector::end (C++     |
| CPPv4N5cudaq8kraus_op9precisionE) |                                   |
| -   [cudaq::KrausSelection (C++   | function)](api/languages/cpp_api. |
|     s                             | html#_CPPv4N5cudaq7qvector3endEv) |
| truct)](api/languages/cpp_api.htm | -   [cudaq::qvector::front (C++   |
| l#_CPPv4N5cudaq14KrausSelectionE) |     function)](ap                 |
| -   [cudaq:                       | i/languages/cpp_api.html#_CPPv4N5 |
| :KrausSelection::circuit_location | cudaq7qvector5frontENSt6size_tE), |
|     (C++                          |                                   |
|     member)](api/langua           |  [\[1\]](api/languages/cpp_api.ht |
| ges/cpp_api.html#_CPPv4N5cudaq14K | ml#_CPPv4N5cudaq7qvector5frontEv) |
| rausSelection16circuit_locationE) | -   [cudaq::qvector::operator=    |
| -                                 |     (C++                          |
|  [cudaq::KrausSelection::is_error |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](a                    | PPv4N5cudaq7qvectoraSERK7qvector) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::qvector::operator\[\] |
| 5cudaq14KrausSelection8is_errorE) |     (C++                          |
| -   [cudaq::Kra                   |     function)                     |
| usSelection::kraus_operator_index | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     member)](api/languages/       | -   [cudaq::qvector::qvector (C++ |
| cpp_api.html#_CPPv4N5cudaq14Kraus |     function)](api/               |
| Selection20kraus_operator_indexE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cuda                         | daq7qvector7qvectorENSt6size_tE), |
| q::KrausSelection::KrausSelection |     [\[1\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function)](a                  | 5cudaq7qvector7qvectorERK5state), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[2\]](api                   |
| 5cudaq14KrausSelection14KrausSele | /languages/cpp_api.html#_CPPv4N5c |
| ctionENSt6size_tENSt6vectorINSt6s | udaq7qvector7qvectorERK7qvector), |
| ize_tEEENSt6stringENSt6size_tEb), |     [\[3\]](ap                    |
|     [\[1\]](api/langu             | i/languages/cpp_api.html#_CPPv4N5 |
| ages/cpp_api.html#_CPPv4N5cudaq14 | cudaq7qvector7qvectorERR7qvector) |
| KrausSelection14KrausSelectionEv) | -   [cudaq::qvector::size (C++    |
| -                                 |     fu                            |
|   [cudaq::KrausSelection::op_name | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     member)](                     | -   [cudaq::qvector::slice (C++   |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/language       |
| N5cudaq14KrausSelection7op_nameE) | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| -   [                             | tor5sliceENSt6size_tENSt6size_tE) |
| cudaq::KrausSelection::operator== | -   [cudaq::qvector::value_type   |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     typ                           |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | e)](api/languages/cpp_api.html#_C |
| usSelectioneqERK14KrausSelection) | PPv4N5cudaq7qvector10value_typeE) |
| -                                 | -   [cudaq::qview (C++            |
|    [cudaq::KrausSelection::qubits |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     member)]                      | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::qview::back (C++      |
| 4N5cudaq14KrausSelection6qubitsE) |     function)                     |
| -   [cudaq::KrausTrajectory (C++  | ](api/languages/cpp_api.html#_CPP |
|     st                            | v4N5cudaq5qview4backENSt6size_tE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::qview::begin (C++     |
| #_CPPv4N5cudaq15KrausTrajectoryE) |                                   |
| -                                 | function)](api/languages/cpp_api. |
|  [cudaq::KrausTrajectory::builder | html#_CPPv4N5cudaq5qview5beginEv) |
|     (C++                          | -   [cudaq::qview::end (C++       |
|     function)](ap                 |                                   |
| i/languages/cpp_api.html#_CPPv4N5 |   function)](api/languages/cpp_ap |
| cudaq15KrausTrajectory7builderEv) | i.html#_CPPv4N5cudaq5qview3endEv) |
| -   [cu                           | -   [cudaq::qview::front (C++     |
| daq::KrausTrajectory::countErrors |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/lang           | N5cudaq5qview5frontENSt6size_tE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |                                   |
| 15KrausTrajectory11countErrorsEv) |    [\[1\]](api/languages/cpp_api. |
| -   [                             | html#_CPPv4N5cudaq5qview5frontEv) |
| cudaq::KrausTrajectory::isOrdered | -   [cudaq::qview::operator\[\]   |
|     (C++                          |     (C++                          |
|     function)](api/l              |     functio                       |
| anguages/cpp_api.html#_CPPv4NK5cu | n)](api/languages/cpp_api.html#_C |
| daq15KrausTrajectory9isOrderedEv) | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| -   [cudaq::                      | -   [cudaq::qview::qview (C++     |
| KrausTrajectory::kraus_selections |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](api/languag          | PPv4I0EN5cudaq5qview5qviewERR1R), |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |     [\[1                          |
| ausTrajectory16kraus_selectionsE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq:                       | PPv4N5cudaq5qview5qviewERK5qview) |
| :KrausTrajectory::KrausTrajectory | -   [cudaq::qview::size (C++      |
|     (C++                          |                                   |
|     function                      | function)](api/languages/cpp_api. |
| )](api/languages/cpp_api.html#_CP | html#_CPPv4NK5cudaq5qview4sizeEv) |
| Pv4N5cudaq15KrausTrajectory15Krau | -   [cudaq::qview::slice (C++     |
| sTrajectoryENSt6size_tENSt6vector |     function)](api/langua         |
| I14KrausSelectionEEdNSt6size_tE), | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|     [\[1\]](api/languag           | iew5sliceENSt6size_tENSt6size_tE) |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | -   [cudaq::qview::value_type     |
| ausTrajectory15KrausTrajectoryEv) |     (C++                          |
| -   [cudaq::Kr                    |     t                             |
| ausTrajectory::measurement_counts | ype)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq5qview10value_typeE) |
|     member)](api/languages        | -   [cudaq::range (C++            |
| /cpp_api.html#_CPPv4N5cudaq15Krau |     fun                           |
| sTrajectory18measurement_countsE) | ction)](api/languages/cpp_api.htm |
| -   [cud                          | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| aq::KrausTrajectory::multiplicity | orI11ElementTypeEE11ElementType), |
|     (C++                          |     [\[1\]](api/languages/cpp_    |
|     member)](api/lan              | api.html#_CPPv4I0EN5cudaq5rangeEN |
| guages/cpp_api.html#_CPPv4N5cudaq | St6vectorI11ElementTypeEE11Elemen |
| 15KrausTrajectory12multiplicityE) | tType11ElementType11ElementType), |
| -   [                             |     [                             |
| cudaq::KrausTrajectory::num_shots | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq5rangeENSt6size_tE) |
|     member)](api                  | -   [cudaq::real (C++             |
| /languages/cpp_api.html#_CPPv4N5c |     type)](api/languages/         |
| udaq15KrausTrajectory9num_shotsE) | cpp_api.html#_CPPv4N5cudaq4realE) |
| -   [c                            | -   [cudaq::registry (C++         |
| udaq::KrausTrajectory::operator== |     type)](api/languages/cpp_     |
|     (C++                          | api.html#_CPPv4N5cudaq8registryE) |
|     function)](api/languages/c    | -                                 |
| pp_api.html#_CPPv4NK5cudaq15Kraus |  [cudaq::registry::RegisteredType |
| TrajectoryeqERK15KrausTrajectory) |     (C++                          |
| -   [cu                           |     class)](api/                  |
| daq::KrausTrajectory::probability | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq8registry14RegisteredTypeE) |
|     member)](api/la               | -   [cudaq::RemoteCapabilities    |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q15KrausTrajectory11probabilityE) |     struc                         |
| -   [cuda                         | t)](api/languages/cpp_api.html#_C |
| q::KrausTrajectory::trajectory_id | PPv4N5cudaq18RemoteCapabilitiesE) |
|     (C++                          | -   [cudaq::Remot                 |
|     member)](api/lang             | eCapabilities::RemoteCapabilities |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     (C++                          |
| 5KrausTrajectory13trajectory_idE) |     function)](api/languages/cpp  |
| -                                 | _api.html#_CPPv4N5cudaq18RemoteCa |
|   [cudaq::KrausTrajectory::weight | pabilities18RemoteCapabilitiesEb) |
|     (C++                          | -   [cudaq:                       |
|     member)](                     | :RemoteCapabilities::stateOverlap |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq15KrausTrajectory6weightE) |     member)](api/langua           |
| -                                 | ges/cpp_api.html#_CPPv4N5cudaq18R |
|    [cudaq::KrausTrajectoryBuilder | emoteCapabilities12stateOverlapE) |
|     (C++                          | -                                 |
|     class)](                      |   [cudaq::RemoteCapabilities::vqe |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22KrausTrajectoryBuilderE) |     member)](                     |
| -   [cud                          | api/languages/cpp_api.html#_CPPv4 |
| aq::KrausTrajectoryBuilder::build | N5cudaq18RemoteCapabilities3vqeE) |
|     (C++                          | -   [cudaq::RemoteRESTQPU (C++    |
|     function)](api/lang           |                                   |
| uages/cpp_api.html#_CPPv4NK5cudaq |  class)](api/languages/cpp_api.ht |
| 22KrausTrajectoryBuilder5buildEv) | ml#_CPPv4N5cudaq13RemoteRESTQPUE) |
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
| -   [cudaq::M2DSparseMatrix (C++  | 5cudaq9run_asyncENSt6futureINSt6v |
|     st                            | ectorINSt15invoke_result_tINSt7de |
| ruct)](api/languages/cpp_api.html | cay_tI13QuantumKernelEEDpNSt7deca |
| #_CPPv4N5cudaq15M2DSparseMatrixE) | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| -   [cudaq::M2OSparseMatrix (C++  | ize_tERR13QuantumKernelDpRR4ARGS) |
|     st                            | -   [cudaq::RuntimeTarget (C++    |
| ruct)](api/languages/cpp_api.html |                                   |
| #_CPPv4N5cudaq15M2OSparseMatrixE) | struct)](api/languages/cpp_api.ht |
| -   [cudaq::matrix_callback (C++  | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     c                             | -   [cudaq::sample (C++           |
| lass)](api/languages/cpp_api.html |     function)](api/languages/c    |
| #_CPPv4N5cudaq15matrix_callbackE) | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| -   [cudaq::matrix_handler (C++   | mpleE13sample_resultRK14sample_op |
|                                   | tionsRR13QuantumKernelDpRR4Args), |
| class)](api/languages/cpp_api.htm |     [\[1\                         |
| l#_CPPv4N5cudaq14matrix_handlerE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::mat                   | Pv4I0DpEN5cudaq6sampleE13sample_r |
| rix_handler::commutation_behavior | esultRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\                            |
|     struct)](api/languages/       | [2\]](api/languages/cpp_api.html# |
| cpp_api.html#_CPPv4N5cudaq14matri | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| x_handler20commutation_behaviorE) | ize_tERR13QuantumKernelDpRR4Args) |
| -                                 | -   [cudaq::sample_options (C++   |
|    [cudaq::matrix_handler::define |     s                             |
|     (C++                          | truct)](api/languages/cpp_api.htm |
|     function)](a                  | l#_CPPv4N5cudaq14sample_optionsE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::sample_result (C++    |
| 5cudaq14matrix_handler6defineENSt |                                   |
| 6stringENSt6vectorINSt7int64_tEEE |  class)](api/languages/cpp_api.ht |
| RR15matrix_callbackRKNSt13unorder | ml#_CPPv4N5cudaq13sample_resultE) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::sample_result::append |
|                                   |     (C++                          |
| [\[1\]](api/languages/cpp_api.htm |     function)](api/languages/cpp_ |
| l#_CPPv4N5cudaq14matrix_handler6d | api.html#_CPPv4N5cudaq13sample_re |
| efineENSt6stringENSt6vectorINSt7i | sult6appendERK15ExecutionResultb) |
| nt64_tEEERR15matrix_callbackRR20d | -   [cudaq::sample_result::begin  |
| iag_matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     function)]                    |
|     [\[2\]](                      | (api/languages/cpp_api.html#_CPPv |
| api/languages/cpp_api.html#_CPPv4 | 4N5cudaq13sample_result5beginEv), |
| N5cudaq14matrix_handler6defineENS |     [\[1\]]                       |
| t6stringENSt6vectorINSt7int64_tEE | (api/languages/cpp_api.html#_CPPv |
| ERR15matrix_callbackRRNSt13unorde | 4NK5cudaq13sample_result5beginEv) |
| red_mapINSt6stringENSt6stringEEE) | -   [cudaq::sample_result::cbegin |
| -                                 |     (C++                          |
|   [cudaq::matrix_handler::degrees |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](ap                 | NK5cudaq13sample_result6cbeginEv) |
| i/languages/cpp_api.html#_CPPv4NK | -   [cudaq::sample_result::cend   |
| 5cudaq14matrix_handler7degreesEv) |     (C++                          |
| -                                 |     function)                     |
|  [cudaq::matrix_handler::displace | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq13sample_result4cendEv) |
|     function)](api/language       | -   [cudaq::sample_result::clear  |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8displaceENSt6size_tE) |     function)                     |
| -   [cudaq::matrix                | ](api/languages/cpp_api.html#_CPP |
| _handler::get_expected_dimensions | v4N5cudaq13sample_result5clearEv) |
|     (C++                          | -   [cudaq::sample_result::count  |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](                   |
| pi.html#_CPPv4NK5cudaq14matrix_ha | api/languages/cpp_api.html#_CPPv4 |
| ndler23get_expected_dimensionsEv) | NK5cudaq13sample_result5countENSt |
| -   [cudaq::matrix_ha             | 11string_viewEKNSt11string_viewE) |
| ndler::get_parameter_descriptions | -   [                             |
|     (C++                          | cudaq::sample_result::deserialize |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     functio                       |
| html#_CPPv4NK5cudaq14matrix_handl | n)](api/languages/cpp_api.html#_C |
| er26get_parameter_descriptionsEv) | PPv4N5cudaq13sample_result11deser |
| -   [c                            | ializeERNSt6vectorINSt6size_tEEE) |
| udaq::matrix_handler::instantiate | -   [cudaq::sample_result::dump   |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](api/languag        |
| pi/languages/cpp_api.html#_CPPv4N | es/cpp_api.html#_CPPv4NK5cudaq13s |
| 5cudaq14matrix_handler11instantia | ample_result4dumpERNSt7ostreamE), |
| teENSt6stringERKNSt6vectorINSt6si |     [\[1\]                        |
| ze_tEEERK20commutation_behavior), | ](api/languages/cpp_api.html#_CPP |
|     [\[1\]](                      | v4NK5cudaq13sample_result4dumpEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::sample_result::end    |
| N5cudaq14matrix_handler11instanti |     (C++                          |
| ateENSt6stringERRNSt6vectorINSt6s |     function                      |
| ize_tEEERK20commutation_behavior) | )](api/languages/cpp_api.html#_CP |
| -   [cuda                         | Pv4N5cudaq13sample_result3endEv), |
| q::matrix_handler::matrix_handler |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/languag        | Pv4NK5cudaq13sample_result3endEv) |
| es/cpp_api.html#_CPPv4I0_NSt11ena | -   [                             |
| ble_if_tINSt12is_base_of_vI16oper | cudaq::sample_result::expectation |
| ator_handler1TEEbEEEN5cudaq14matr |     (C++                          |
| ix_handler14matrix_handlerERK1T), |     f                             |
|     [\[1\]](ap                    | unction)](api/languages/cpp_api.h |
| i/languages/cpp_api.html#_CPPv4I0 | tml#_CPPv4NK5cudaq13sample_result |
| _NSt11enable_if_tINSt12is_base_of | 11expectationEKNSt11string_viewE) |
| _vI16operator_handler1TEEbEEEN5cu | -   [cuda                         |
| daq14matrix_handler14matrix_handl | q::sample_result::get_annotations |
| erERK1TRK20commutation_behavior), |     (C++                          |
|     [\[2\]](api/languages/cpp_ap  |     function)](api/langua         |
| i.html#_CPPv4N5cudaq14matrix_hand | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| ler14matrix_handlerENSt6size_tE), | sample_result15get_annotationsEv) |
|     [\[3\]](api/                  | -   [c                            |
| languages/cpp_api.html#_CPPv4N5cu | udaq::sample_result::get_marginal |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erENSt6stringERKNSt6vectorINSt6si |     function)](api/languages/cpp_ |
| ze_tEEERK20commutation_behavior), | api.html#_CPPv4NK5cudaq13sample_r |
|     [\[4\]](api/                  | esult12get_marginalERKNSt6vectorI |
| languages/cpp_api.html#_CPPv4N5cu | NSt6size_tEEEKNSt11string_viewE), |
| daq14matrix_handler14matrix_handl |     [\[1\]](api/languages/cpp_    |
| erENSt6stringERRNSt6vectorINSt6si | api.html#_CPPv4NK5cudaq13sample_r |
| ze_tEEERK20commutation_behavior), | esult12get_marginalERRKNSt6vector |
|     [\                            | INSt6size_tEEEKNSt11string_viewE) |
| [5\]](api/languages/cpp_api.html# | -   [cuda                         |
| _CPPv4N5cudaq14matrix_handler14ma | q::sample_result::get_total_shots |
| trix_handlerERK14matrix_handler), |     (C++                          |
|     [                             |     function)](api/langua         |
| \[6\]](api/languages/cpp_api.html | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| #_CPPv4N5cudaq14matrix_handler14m | sample_result15get_total_shotsEv) |
| atrix_handlerERR14matrix_handler) | -   [cuda                         |
| -                                 | q::sample_result::has_even_parity |
|  [cudaq::matrix_handler::momentum |     (C++                          |
|     (C++                          |     fun                           |
|     function)](api/language       | ction)](api/languages/cpp_api.htm |
| s/cpp_api.html#_CPPv4N5cudaq14mat | l#_CPPv4N5cudaq13sample_result15h |
| rix_handler8momentumENSt6size_tE) | as_even_parityENSt11string_viewE) |
| -                                 | -   [cuda                         |
|    [cudaq::matrix_handler::number | q::sample_result::has_expectation |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     funct                         |
| ges/cpp_api.html#_CPPv4N5cudaq14m | ion)](api/languages/cpp_api.html# |
| atrix_handler6numberENSt6size_tE) | _CPPv4NK5cudaq13sample_result15ha |
| -                                 | s_expectationEKNSt11string_viewE) |
| [cudaq::matrix_handler::operator= | -   [cu                           |
|     (C++                          | daq::sample_result::most_probable |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     fun                           |
| l#_CPPv4I0_NSt11enable_if_tIXaant | ction)](api/languages/cpp_api.htm |
| NSt7is_sameI1T14matrix_handlerE5v | l#_CPPv4NK5cudaq13sample_result13 |
| alueENSt12is_base_of_vI16operator | most_probableEKNSt11string_viewE) |
| _handler1TEEEbEEEN5cudaq14matrix_ | -                                 |
| handleraSER14matrix_handlerRK1T), | [cudaq::sample_result::operator+= |
|     [\[1\]](api/languages         |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq14matr |     function)](api/langua         |
| ix_handleraSERK14matrix_handler), | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     [\[2\]](api/language          | ample_resultpLERK13sample_result) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -                                 |
| rix_handleraSERR14matrix_handler) |  [cudaq::sample_result::operator= |
| -   [                             |     (C++                          |
| cudaq::matrix_handler::operator== |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function)](api/languages      | ample_resultaSERR13sample_result) |
| /cpp_api.html#_CPPv4NK5cudaq14mat | -                                 |
| rix_handlereqERK14matrix_handler) | [cudaq::sample_result::operator== |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::parity |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     function)](api/langua         | ample_resulteqERK13sample_result) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -   [                             |
| atrix_handler6parityENSt6size_tE) | cudaq::sample_result::probability |
| -                                 |     (C++                          |
|  [cudaq::matrix_handler::position |     function)](api/lan            |
|     (C++                          | guages/cpp_api.html#_CPPv4NK5cuda |
|     function)](api/language       | q13sample_result11probabilityENSt |
| s/cpp_api.html#_CPPv4N5cudaq14mat | 11string_viewEKNSt11string_viewE) |
| rix_handler8positionENSt6size_tE) | -   [cud                          |
| -   [cudaq::                      | aq::sample_result::register_names |
| matrix_handler::remove_definition |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     fu                            | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| nction)](api/languages/cpp_api.ht | 3sample_result14register_namesEv) |
| ml#_CPPv4N5cudaq14matrix_handler1 | -                                 |
| 7remove_definitionERKNSt6stringE) |    [cudaq::sample_result::reorder |
| -                                 |     (C++                          |
|   [cudaq::matrix_handler::squeeze |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function)](api/languag        | ample_result7reorderERKNSt6vector |
| es/cpp_api.html#_CPPv4N5cudaq14ma | INSt6size_tEEEKNSt11string_viewE) |
| trix_handler7squeezeENSt6size_tE) | -   [cu                           |
| -   [cudaq::m                     | daq::sample_result::sample_result |
| atrix_handler::to_diagonal_matrix |     (C++                          |
|     (C++                          |     function)](api/               |
|     function)](api/lang           | languages/cpp_api.html#_CPPv4N5cu |
| uages/cpp_api.html#_CPPv4NK5cudaq | daq13sample_result13sample_result |
| 14matrix_handler18to_diagonal_mat | E16CountsDictionary10cudaq_json), |
| rixERNSt13unordered_mapINSt6size_ |     [                             |
| tENSt7int64_tEEERKNSt13unordered_ | \[1\]](api/languages/cpp_api.html |
| mapINSt6stringENSt7complexIdEEEE) | #_CPPv4N5cudaq13sample_result13sa |
| -                                 | mple_resultERK15ExecutionResult), |
| [cudaq::matrix_handler::to_matrix |     [\[2\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     function)                     | q13sample_result13sample_resultER |
| ](api/languages/cpp_api.html#_CPP | KNSt6vectorI15ExecutionResultEE), |
| v4NK5cudaq14matrix_handler9to_mat |                                   |
| rixERNSt13unordered_mapINSt6size_ |  [\[3\]](api/languages/cpp_api.ht |
| tENSt7int64_tEEERKNSt13unordered_ | ml#_CPPv4N5cudaq13sample_result13 |
| mapINSt6stringENSt7complexIdEEEE) | sample_resultERR13sample_result), |
| -                                 |     [                             |
| [cudaq::matrix_handler::to_string | \[4\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq13sample_result13sa |
|     function)](api/               | mple_resultERR15ExecutionResult), |
| languages/cpp_api.html#_CPPv4NK5c |     [\[5\]](api/lan               |
| udaq14matrix_handler9to_stringEb) | guages/cpp_api.html#_CPPv4N5cudaq |
| -                                 | 13sample_result13sample_resultEdR |
| [cudaq::matrix_handler::unique_id | KNSt6vectorI15ExecutionResultEE), |
|     (C++                          |     [\[6\]](api/lan               |
|     function)](api/               | guages/cpp_api.html#_CPPv4N5cudaq |
| languages/cpp_api.html#_CPPv4NK5c | 13sample_result13sample_resultEv) |
| udaq14matrix_handler9unique_idEv) | -                                 |
| -   [cudaq:                       |  [cudaq::sample_result::serialize |
| :matrix_handler::\~matrix_handler |     (C++                          |
|     (C++                          |     function)](api                |
|     functi                        | /languages/cpp_api.html#_CPPv4NK5 |
| on)](api/languages/cpp_api.html#_ | cudaq13sample_result9serializeEv) |
| CPPv4N5cudaq14matrix_handlerD0Ev) | -   [cudaq::sample_result::size   |
| -   [cudaq::matrix_op (C++        |     (C++                          |
|     type)](api/languages/cpp_a    |     function)](api/languages/c    |
| pi.html#_CPPv4N5cudaq9matrix_opE) | pp_api.html#_CPPv4NK5cudaq13sampl |
| -   [cudaq::matrix_op_term (C++   | e_result4sizeEKNSt11string_viewE) |
|                                   | -   [cudaq::sample_result::to_map |
|  type)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14matrix_op_termE) |     function)](api/languages/cpp  |
| -                                 | _api.html#_CPPv4NK5cudaq13sample_ |
|    [cudaq::mdiag_operator_handler | result6to_mapEKNSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|     class)](                      | q::sample_result::\~sample_result |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22mdiag_operator_handlerE) |     funct                         |
| -   [cudaq::measure_handle (C++   | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq13sample_resultD0Ev) |
| class)](api/languages/cpp_api.htm | -   [cudaq::scalar_callback (C++  |
| l#_CPPv4N5cudaq14measure_handleE) |     c                             |
| -   [cudaq::measure_result (C++   | lass)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq15scalar_callbackE) |
|  type)](api/languages/cpp_api.htm | -   [c                            |
| l#_CPPv4N5cudaq14measure_resultE) | udaq::scalar_callback::operator() |
| -   [cudaq::mpi (C++              |     (C++                          |
|     type)](api/languages          |     function)](api/language       |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| -   [cudaq::mpi::all_gather (C++  | alar_callbackclERKNSt13unordered_ |
|     fu                            | mapINSt6stringENSt7complexIdEEEE) |
| nction)](api/languages/cpp_api.ht | -   [                             |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | cudaq::scalar_callback::operator= |
| RNSt6vectorIdEERKNSt6vectorIdEE), |     (C++                          |
|                                   |     function)](api/languages/c    |
|   [\[1\]](api/languages/cpp_api.h | pp_api.html#_CPPv4N5cudaq15scalar |
| tml#_CPPv4N5cudaq3mpi10all_gather | _callbackaSERK15scalar_callback), |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     [\[1\]](api/languages/        |
| -   [cudaq::mpi::all_reduce (C++  | cpp_api.html#_CPPv4N5cudaq15scala |
|                                   | r_callbackaSERR15scalar_callback) |
|  function)](api/languages/cpp_api | -   [cudaq:                       |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | :scalar_callback::scalar_callback |
| reduceE1TRK1TRK14BinaryFunction), |     (C++                          |
|     [\[1\]](api/langu             |     function)](api/languag        |
| ages/cpp_api.html#_CPPv4I00EN5cud | es/cpp_api.html#_CPPv4I0_NSt11ena |
| aq3mpi10all_reduceE1TRK1TRK4Func) | ble_if_tINSt16is_invocable_r_vINS |
| -   [cudaq::mpi::broadcast (C++   | t7complexIdEE8CallableRKNSt13unor |
|     function)](api/               | dered_mapINSt6stringENSt7complexI |
| languages/cpp_api.html#_CPPv4N5cu | dEEEEEEbEEEN5cudaq15scalar_callba |
| daq3mpi9broadcastERNSt6stringEi), | ck15scalar_callbackERR8Callable), |
|     [\[1\]](api/la                |     [\[1\                         |
| nguages/cpp_api.html#_CPPv4N5cuda | ]](api/languages/cpp_api.html#_CP |
| q3mpi9broadcastERNSt6vectorIdEEi) | Pv4N5cudaq15scalar_callback15scal |
| -   [cudaq::mpi::finalize (C++    | ar_callbackERK15scalar_callback), |
|     f                             |     [\[2                          |
| unction)](api/languages/cpp_api.h | \]](api/languages/cpp_api.html#_C |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | PPv4N5cudaq15scalar_callback15sca |
| -   [cudaq::mpi::initialize (C++  | lar_callbackERR15scalar_callback) |
|     function                      | -   [cudaq::scalar_operator (C++  |
| )](api/languages/cpp_api.html#_CP |     c                             |
| Pv4N5cudaq3mpi10initializeEiPPc), | lass)](api/languages/cpp_api.html |
|     [                             | #_CPPv4N5cudaq15scalar_operatorE) |
| \[1\]](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq3mpi10initializeEv) | [cudaq::scalar_operator::evaluate |
| -   [cudaq::mpi::is_initialized   |     (C++                          |
|     (C++                          |                                   |
|     function                      |    function)](api/languages/cpp_a |
| )](api/languages/cpp_api.html#_CP | pi.html#_CPPv4NK5cudaq15scalar_op |
| Pv4N5cudaq3mpi14is_initializedEv) | erator8evaluateERKNSt13unordered_ |
| -   [cudaq::mpi::num_ranks (C++   | mapINSt6stringENSt7complexIdEEEE) |
|     fu                            | -   [cudaq::scalar_ope            |
| nction)](api/languages/cpp_api.ht | rator::get_parameter_descriptions |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     (C++                          |
| -   [cudaq::mpi::rank (C++        |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|    function)](api/languages/cpp_a | tml#_CPPv4NK5cudaq15scalar_operat |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | or26get_parameter_descriptionsEv) |
| -   [cudaq::noise_model (C++      | -   [cu                           |
|                                   | daq::scalar_operator::is_constant |
|    class)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq11noise_modelE) |     function)](api/lang           |
| -   [cudaq::n                     | uages/cpp_api.html#_CPPv4NK5cudaq |
| oise_model::add_all_qubit_channel | 15scalar_operator11is_constantEv) |
|     (C++                          | -   [c                            |
|     function)](api                | udaq::scalar_operator::operator\* |
| /languages/cpp_api.html#_CPPv4IDp |     (C++                          |
| EN5cudaq11noise_model21add_all_qu |     function                      |
| bit_channelEvRK13kraus_channeli), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/langua            | Pv4N5cudaq15scalar_operatormlENSt |
| ges/cpp_api.html#_CPPv4N5cudaq11n | 7complexIdEERK15scalar_operator), |
| oise_model21add_all_qubit_channel |     [\[1\                         |
| ERKNSt6stringERK13kraus_channeli) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatormlENSt |
|  [cudaq::noise_model::add_channel | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     funct                         | p_api.html#_CPPv4N5cudaq15scalar_ |
| ion)](api/languages/cpp_api.html# | operatormlEdRK15scalar_operator), |
| _CPPv4IDpEN5cudaq11noise_model11a |     [\[3\]](api/languages/cp      |
| dd_channelEvRK15PredicateFuncTy), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](api/languages/cpp_    | operatormlEdRR15scalar_operator), |
| api.html#_CPPv4IDpEN5cudaq11noise |     [\[4\]](api/languages         |
| _model11add_channelEvRKNSt6vector | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| INSt6size_tEEERK13kraus_channel), | alar_operatormlENSt7complexIdEE), |
|     [\[2\]](ap                    |     [\[5\]](api/languages/cpp     |
| i/languages/cpp_api.html#_CPPv4N5 | _api.html#_CPPv4NKR5cudaq15scalar |
| cudaq11noise_model11add_channelER | _operatormlERK15scalar_operator), |
| KNSt6stringERK15PredicateFuncTy), |     [\[6\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
| [\[3\]](api/languages/cpp_api.htm | 4NKR5cudaq15scalar_operatormlEd), |
| l#_CPPv4N5cudaq11noise_model11add |     [\[7\]](api/language          |
| _channelERKNSt6stringERKNSt6vecto | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| rINSt6size_tEEERK13kraus_channel) | alar_operatormlENSt7complexIdEE), |
| -   [cudaq::noise_model::empty    |     [\[8\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     function                      | _operatormlERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[9\                         |
| Pv4NK5cudaq11noise_model5emptyEv) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4NO5cudaq15scalar_operatormlEd) |
| [cudaq::noise_model::get_channels | -   [cu                           |
|     (C++                          | daq::scalar_operator::operator\*= |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4I0ENK |     function)](api/languag        |
| 5cudaq11noise_model12get_channels | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ENSt6vectorI13kraus_channelEERKNS | alar_operatormLENSt7complexIdEE), |
| t6vectorINSt6size_tEEERKNSt6vecto |     [\[1\]](api/languages/c       |
| rINSt6size_tEEERKNSt6vectorIdEE), | pp_api.html#_CPPv4N5cudaq15scalar |
|     [\[1\]](api/languages/cpp_a   | _operatormLERK15scalar_operator), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[2                          |
| el12get_channelsERKNSt6stringERKN | \]](api/languages/cpp_api.html#_C |
| St6vectorINSt6size_tEEERKNSt6vect | PPv4N5cudaq15scalar_operatormLEd) |
| orINSt6size_tEEERKNSt6vectorIdEE) | -   [                             |
| -                                 | cudaq::scalar_operator::operator+ |
|  [cudaq::noise_model::noise_model |     (C++                          |
|     (C++                          |     function                      |
|     function)](api                | )](api/languages/cpp_api.html#_CP |
| /languages/cpp_api.html#_CPPv4N5c | Pv4N5cudaq15scalar_operatorplENSt |
| udaq11noise_model11noise_modelEv) | 7complexIdEERK15scalar_operator), |
| -   [cu                           |     [\[1\                         |
| daq::noise_model::PredicateFuncTy | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatorplENSt |
|     type)](api/la                 | 7complexIdEERR15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[2\]](api/languages/cp      |
| q11noise_model15PredicateFuncTyE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cud                          | operatorplEdRK15scalar_operator), |
| aq::noise_model::register_channel |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     function)](api/languages      | operatorplEdRR15scalar_operator), |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     [\[4\]](api/languages         |
| noise_model16register_channelEvv) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -   [cudaq::                      | alar_operatorplENSt7complexIdEE), |
| noise_model::requires_constructor |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|     type)](api/languages/cp       | _operatorplERK15scalar_operator), |
| p_api.html#_CPPv4I0DpEN5cudaq11no |     [\[6\]]                       |
| ise_model20requires_constructorE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::noise_model_type (C++ | 4NKR5cudaq15scalar_operatorplEd), |
|     e                             |     [\[7\]]                       |
| num)](api/languages/cpp_api.html# | (api/languages/cpp_api.html#_CPPv |
| _CPPv4N5cudaq16noise_model_typeE) | 4NKR5cudaq15scalar_operatorplEv), |
| -   [cudaq::no                    |     [\[8\]](api/language          |
| ise_model_type::amplitude_damping | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatorplENSt7complexIdEE), |
|     enumerator)](api/languages    |     [\[9\]](api/languages/cp      |
| /cpp_api.html#_CPPv4N5cudaq16nois | p_api.html#_CPPv4NO5cudaq15scalar |
| e_model_type17amplitude_dampingE) | _operatorplERK15scalar_operator), |
| -   [cudaq::noise_mode            |     [\[10\]                       |
| l_type::amplitude_damping_channel | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatorplEd), |
|     e                             |     [\[11\                        |
| numerator)](api/languages/cpp_api | ]](api/languages/cpp_api.html#_CP |
| .html#_CPPv4N5cudaq16noise_model_ | Pv4NO5cudaq15scalar_operatorplEv) |
| type25amplitude_damping_channelE) | -   [c                            |
| -   [cudaq::n                     | udaq::scalar_operator::operator+= |
| oise_model_type::bit_flip_channel |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](api/language     | es/cpp_api.html#_CPPv4N5cudaq15sc |
| s/cpp_api.html#_CPPv4N5cudaq16noi | alar_operatorpLENSt7complexIdEE), |
| se_model_type16bit_flip_channelE) |     [\[1\]](api/languages/c       |
| -   [cudaq::                      | pp_api.html#_CPPv4N5cudaq15scalar |
| noise_model_type::depolarization1 | _operatorpLERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     enumerator)](api/languag      | \]](api/languages/cpp_api.html#_C |
| es/cpp_api.html#_CPPv4N5cudaq16no | PPv4N5cudaq15scalar_operatorpLEd) |
| ise_model_type15depolarization1E) | -   [                             |
| -   [cudaq::                      | cudaq::scalar_operator::operator- |
| noise_model_type::depolarization2 |     (C++                          |
|     (C++                          |     function                      |
|     enumerator)](api/languag      | )](api/languages/cpp_api.html#_CP |
| es/cpp_api.html#_CPPv4N5cudaq16no | Pv4N5cudaq15scalar_operatormiENSt |
| ise_model_type15depolarization2E) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::noise_m               |     [\[1\                         |
| odel_type::depolarization_channel | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormiENSt |
|                                   | 7complexIdEERR15scalar_operator), |
|   enumerator)](api/languages/cpp_ |     [\[2\]](api/languages/cp      |
| api.html#_CPPv4N5cudaq16noise_mod | p_api.html#_CPPv4N5cudaq15scalar_ |
| el_type22depolarization_channelE) | operatormiEdRK15scalar_operator), |
| -                                 |     [\[3\]](api/languages/cp      |
|  [cudaq::noise_model_type::pauli1 | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRR15scalar_operator), |
|     enumerator)](a                |     [\[4\]](api/languages         |
| pi/languages/cpp_api.html#_CPPv4N | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| 5cudaq16noise_model_type6pauli1E) | alar_operatormiENSt7complexIdEE), |
| -                                 |     [\[5\]](api/languages/cpp     |
|  [cudaq::noise_model_type::pauli2 | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatormiERK15scalar_operator), |
|     enumerator)](a                |     [\[6\]]                       |
| pi/languages/cpp_api.html#_CPPv4N | (api/languages/cpp_api.html#_CPPv |
| 5cudaq16noise_model_type6pauli2E) | 4NKR5cudaq15scalar_operatormiEd), |
| -   [cudaq                        |     [\[7\]]                       |
| ::noise_model_type::phase_damping | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEv), |
|     enumerator)](api/langu        |     [\[8\]](api/language          |
| ages/cpp_api.html#_CPPv4N5cudaq16 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| noise_model_type13phase_dampingE) | alar_operatormiENSt7complexIdEE), |
| -   [cudaq::noi                   |     [\[9\]](api/languages/cp      |
| se_model_type::phase_flip_channel | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatormiERK15scalar_operator), |
|     enumerator)](api/languages/   |     [\[10\]                       |
| cpp_api.html#_CPPv4N5cudaq16noise | ](api/languages/cpp_api.html#_CPP |
| _model_type18phase_flip_channelE) | v4NO5cudaq15scalar_operatormiEd), |
| -                                 |     [\[11\                        |
| [cudaq::noise_model_type::unknown | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatormiEv) |
|     enumerator)](ap               | -   [c                            |
| i/languages/cpp_api.html#_CPPv4N5 | udaq::scalar_operator::operator-= |
| cudaq16noise_model_type7unknownE) |     (C++                          |
| -                                 |     function)](api/languag        |
| [cudaq::noise_model_type::x_error | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormIENSt7complexIdEE), |
|     enumerator)](ap               |     [\[1\]](api/languages/c       |
| i/languages/cpp_api.html#_CPPv4N5 | pp_api.html#_CPPv4N5cudaq15scalar |
| cudaq16noise_model_type7x_errorE) | _operatormIERK15scalar_operator), |
| -                                 |     [\[2                          |
| [cudaq::noise_model_type::y_error | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatormIEd) |
|     enumerator)](ap               | -   [                             |
| i/languages/cpp_api.html#_CPPv4N5 | cudaq::scalar_operator::operator/ |
| cudaq16noise_model_type7y_errorE) |     (C++                          |
| -                                 |     function                      |
| [cudaq::noise_model_type::z_error | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|     enumerator)](ap               | 7complexIdEERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[1\                         |
| cudaq16noise_model_type7z_errorE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::num_available_gpus    | Pv4N5cudaq15scalar_operatordvENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     function                      |     [\[2\]](api/languages/cp      |
| )](api/languages/cpp_api.html#_CP | p_api.html#_CPPv4N5cudaq15scalar_ |
| Pv4N5cudaq18num_available_gpusEv) | operatordvEdRK15scalar_operator), |
| -   [cudaq::observe (C++          |     [\[3\]](api/languages/cp      |
|     function)]                    | p_api.html#_CPPv4N5cudaq15scalar_ |
| (api/languages/cpp_api.html#_CPPv | operatordvEdRR15scalar_operator), |
| 4I00DpEN5cudaq7observeENSt6vector |     [\[4\]](api/languages         |
| I14observe_resultEERR13QuantumKer | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| nelRK15SpinOpContainerDpRR4Args), | alar_operatordvENSt7complexIdEE), |
|     [\[1\]](api/languages/cpp_ap  |     [\[5\]](api/languages/cpp     |
| i.html#_CPPv4I0DpEN5cudaq7observe | _api.html#_CPPv4NKR5cudaq15scalar |
| E14observe_resultNSt6size_tERR13Q | _operatordvERK15scalar_operator), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[6\]]                       |
|     [\[                           | (api/languages/cpp_api.html#_CPPv |
| 2\]](api/languages/cpp_api.html#_ | 4NKR5cudaq15scalar_operatordvEd), |
| CPPv4I0DpEN5cudaq7observeE14obser |     [\[7\]](api/language          |
| ve_resultRK15observe_optionsRR13Q | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| uantumKernelRK7spin_opDpRR4Args), | alar_operatordvENSt7complexIdEE), |
|     [\[3\]](api/lang              |     [\[8\]](api/languages/cp      |
| uages/cpp_api.html#_CPPv4I0DpEN5c | p_api.html#_CPPv4NO5cudaq15scalar |
| udaq7observeE14observe_resultRR13 | _operatordvERK15scalar_operator), |
| QuantumKernelRK7spin_opDpRR4Args) |     [\[9\                         |
| -   [cudaq::observe_options (C++  | ]](api/languages/cpp_api.html#_CP |
|     st                            | Pv4NO5cudaq15scalar_operatordvEd) |
| ruct)](api/languages/cpp_api.html | -   [c                            |
| #_CPPv4N5cudaq15observe_optionsE) | udaq::scalar_operator::operator/= |
| -   [cudaq::observe_result (C++   |     (C++                          |
|                                   |     function)](api/languag        |
| class)](api/languages/cpp_api.htm | es/cpp_api.html#_CPPv4N5cudaq15sc |
| l#_CPPv4N5cudaq14observe_resultE) | alar_operatordVENSt7complexIdEE), |
| -                                 |     [\[1\]](api/languages/c       |
|    [cudaq::observe_result::counts | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatordVERK15scalar_operator), |
|     function)](api/languages/c    |     [\[2                          |
| pp_api.html#_CPPv4N5cudaq14observ | \]](api/languages/cpp_api.html#_C |
| e_result6countsERK12spin_op_term) | PPv4N5cudaq15scalar_operatordVEd) |
| -   [cudaq::observe_result::dump  | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator= |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/c    |
| v4N5cudaq14observe_result4dumpEv) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [c                            | _operatoraSERK15scalar_operator), |
| udaq::observe_result::expectation |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|                                   | r_operatoraSERR15scalar_operator) |
| function)](api/languages/cpp_api. | -   [c                            |
| html#_CPPv4N5cudaq14observe_resul | udaq::scalar_operator::operator== |
| t11expectationERK12spin_op_term), |     (C++                          |
|     [\[1\]](api/la                |     function)](api/languages/c    |
| nguages/cpp_api.html#_CPPv4N5cuda | pp_api.html#_CPPv4NK5cudaq15scala |
| q14observe_result11expectationEv) | r_operatoreqERK15scalar_operator) |
| -   [cuda                         | -   [cudaq:                       |
| q::observe_result::id_coefficient | :scalar_operator::scalar_operator |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     func                          |
| ages/cpp_api.html#_CPPv4N5cudaq14 | tion)](api/languages/cpp_api.html |
| observe_result14id_coefficientEv) | #_CPPv4N5cudaq15scalar_operator15 |
| -   [cuda                         | scalar_operatorENSt7complexIdEE), |
| q::observe_result::observe_result |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|                                   | scalar_operator15scalar_operatorE |
|   function)](api/languages/cpp_ap | RK15scalar_callbackRRNSt13unorder |
| i.html#_CPPv4N5cudaq14observe_res | ed_mapINSt6stringENSt6stringEEE), |
| ult14observe_resultEdRK7spin_op), |     [\[2\                         |
|     [\[1\]](a                     | ]](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4N5cudaq15scalar_operator15scal |
| 5cudaq14observe_result14observe_r | ar_operatorERK15scalar_operator), |
| esultEdRK7spin_op13sample_result) |     [\[3\]](api/langu             |
| -                                 | ages/cpp_api.html#_CPPv4N5cudaq15 |
|  [cudaq::observe_result::operator | scalar_operator15scalar_operatorE |
|     double (C++                   | RR15scalar_callbackRRNSt13unorder |
|     functio                       | ed_mapINSt6stringENSt6stringEEE), |
| n)](api/languages/cpp_api.html#_C |     [\[4\                         |
| PPv4N5cudaq14observe_resultcvdEv) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operator15scal |
|  [cudaq::observe_result::raw_data | ar_operatorERR15scalar_operator), |
|     (C++                          |     [\[5\]](api/language          |
|     function)](ap                 | s/cpp_api.html#_CPPv4N5cudaq15sca |
| i/languages/cpp_api.html#_CPPv4N5 | lar_operator15scalar_operatorEd), |
| cudaq14observe_result8raw_dataEv) |     [\[6\]](api/languag           |
| -   [cudaq::operator_handler (C++ | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     cl                            | alar_operator15scalar_operatorEv) |
| ass)](api/languages/cpp_api.html# | -   [                             |
| _CPPv4N5cudaq16operator_handlerE) | cudaq::scalar_operator::to_matrix |
| -   [cudaq::optimizable_function  |     (C++                          |
|     (C++                          |                                   |
|     class)                        |   function)](api/languages/cpp_ap |
| ](api/languages/cpp_api.html#_CPP | i.html#_CPPv4NK5cudaq15scalar_ope |
| v4N5cudaq20optimizable_functionE) | rator9to_matrixERKNSt13unordered_ |
| -   [cudaq::optimization_result   | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [                             |
|     type                          | cudaq::scalar_operator::to_string |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq19optimization_resultE) |     function)](api/l              |
| -   [cudaq::optimizer (C++        | anguages/cpp_api.html#_CPPv4NK5cu |
|     class)](api/languages/cpp_a   | daq15scalar_operator9to_stringEv) |
| pi.html#_CPPv4N5cudaq9optimizerE) | -   [cudaq::s                     |
| -   [cudaq::optimizer::optimize   | calar_operator::\~scalar_operator |
|     (C++                          |     (C++                          |
|                                   |     functio                       |
|  function)](api/languages/cpp_api | n)](api/languages/cpp_api.html#_C |
| .html#_CPPv4N5cudaq9optimizer8opt | PPv4N5cudaq15scalar_operatorD0Ev) |
| imizeEKiRR20optimizable_function) | -   [cudaq::set_noise (C++        |
| -   [cu                           |     function)](api/langu          |
| daq::optimizer::requiresGradients | ages/cpp_api.html#_CPPv4N5cudaq9s |
|     (C++                          | et_noiseERKN5cudaq11noise_modelE) |
|     function)](api/la             | -   [cudaq::set_random_seed (C++  |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/               |
| q9optimizer17requiresGradientsEv) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::orca (C++             | daq15set_random_seedENSt6size_tE) |
|     type)](api/languages/         | -   [cudaq::simulation_precision  |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     (C++                          |
| -   [cudaq::orca::sample (C++     |     enum)                         |
|     function)](api/languages/c    | ](api/languages/cpp_api.html#_CPP |
| pp_api.html#_CPPv4N5cudaq4orca6sa | v4N5cudaq20simulation_precisionE) |
| mpleERNSt6vectorINSt6size_tEEERNS | -   [                             |
| t6vectorINSt6size_tEEERNSt6vector | cudaq::simulation_precision::fp32 |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     (C++                          |
|     [\[1\]]                       |     enumerator)](api              |
| (api/languages/cpp_api.html#_CPPv | /languages/cpp_api.html#_CPPv4N5c |
| 4N5cudaq4orca6sampleERNSt6vectorI | udaq20simulation_precision4fp32E) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [                             |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | cudaq::simulation_precision::fp64 |
| -   [cudaq::orca::sample_async    |     (C++                          |
|     (C++                          |     enumerator)](api              |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
| function)](api/languages/cpp_api. | udaq20simulation_precision4fp64E) |
| html#_CPPv4N5cudaq4orca12sample_a | -   [cudaq::SimulationState (C++  |
| syncERNSt6vectorINSt6size_tEEERNS |     c                             |
| t6vectorINSt6size_tEEERNSt6vector | lass)](api/languages/cpp_api.html |
| IdEERNSt6vectorIdEEiNSt6size_tE), | #_CPPv4N5cudaq15SimulationStateE) |
|     [\[1\]](api/la                | -   [                             |
| nguages/cpp_api.html#_CPPv4N5cuda | cudaq::SimulationState::precision |
| q4orca12sample_asyncERNSt6vectorI |     (C++                          |
| NSt6size_tEEERNSt6vectorINSt6size |     enum)](api                    |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::OrcaRemoteRESTQPU     | udaq15SimulationState9precisionE) |
|     (C++                          | -   [cudaq:                       |
|     cla                           | :SimulationState::precision::fp32 |
| ss)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |     enumerator)](api/lang         |
| -   [cudaq::other_policies (C++   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     s                             | 5SimulationState9precision4fp32E) |
| truct)](api/languages/cpp_api.htm | -   [cudaq:                       |
| l#_CPPv4N5cudaq14other_policiesE) | :SimulationState::precision::fp64 |
| -   [cudaq::PasqalRemoteRESTQPU   |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     class                         | uages/cpp_api.html#_CPPv4N5cudaq1 |
| )](api/languages/cpp_api.html#_CP | 5SimulationState9precision4fp64E) |
| Pv4N5cudaq19PasqalRemoteRESTQPUE) | -                                 |
| -   [cudaq::pauli1 (C++           |   [cudaq::SimulationState::Tensor |
|     class)](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq6pauli1E) |     struct)](                     |
| -                                 | api/languages/cpp_api.html#_CPPv4 |
|    [cudaq::pauli1::num_parameters | N5cudaq15SimulationState6TensorE) |
|     (C++                          | -   [cudaq::spin_handler (C++     |
|     member)]                      |                                   |
| (api/languages/cpp_api.html#_CPPv |   class)](api/languages/cpp_api.h |
| 4N5cudaq6pauli114num_parametersE) | tml#_CPPv4N5cudaq12spin_handlerE) |
| -   [cudaq::pauli1::num_targets   | -   [cudaq:                       |
|     (C++                          | :spin_handler::to_diagonal_matrix |
|     membe                         |     (C++                          |
| r)](api/languages/cpp_api.html#_C |     function)](api/la             |
| PPv4N5cudaq6pauli111num_targetsE) | nguages/cpp_api.html#_CPPv4NK5cud |
| -   [cudaq::pauli1::pauli1 (C++   | aq12spin_handler18to_diagonal_mat |
|     function)](api/languages/cpp_ | rixERNSt13unordered_mapINSt6size_ |
| api.html#_CPPv4N5cudaq6pauli16pau | tENSt7int64_tEEERKNSt13unordered_ |
| li1ERKNSt6vectorIN5cudaq4realEEE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::pauli2 (C++           | -                                 |
|     class)](api/languages/cp      |   [cudaq::spin_handler::to_matrix |
| p_api.html#_CPPv4N5cudaq6pauli2E) |     (C++                          |
| -                                 |     function                      |
|    [cudaq::pauli2::num_parameters | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq12spin_handler9to_matri |
|     member)]                      | xERKNSt6stringENSt7complexIdEEb), |
| (api/languages/cpp_api.html#_CPPv |     [\[1                          |
| 4N5cudaq6pauli214num_parametersE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::pauli2::num_targets   | PPv4NK5cudaq12spin_handler9to_mat |
|     (C++                          | rixERNSt13unordered_mapINSt6size_ |
|     membe                         | tENSt7int64_tEEERKNSt13unordered_ |
| r)](api/languages/cpp_api.html#_C | mapINSt6stringENSt7complexIdEEEE) |
| PPv4N5cudaq6pauli211num_targetsE) | -   [cuda                         |
| -   [cudaq::pauli2::pauli2 (C++   | q::spin_handler::to_sparse_matrix |
|     function)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq6pauli26pau |     function)](api/               |
| li2ERKNSt6vectorIN5cudaq4realEEE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::phase_damping (C++    | daq12spin_handler16to_sparse_matr |
|                                   | ixERKNSt6stringENSt7complexIdEEb) |
|  class)](api/languages/cpp_api.ht | -                                 |
| ml#_CPPv4N5cudaq13phase_dampingE) |   [cudaq::spin_handler::to_string |
| -   [cud                          |     (C++                          |
| aq::phase_damping::num_parameters |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     member)](api/lan              | 5cudaq12spin_handler9to_stringEb) |
| guages/cpp_api.html#_CPPv4N5cudaq | -                                 |
| 13phase_damping14num_parametersE) |   [cudaq::spin_handler::unique_id |
| -   [                             |     (C++                          |
| cudaq::phase_damping::num_targets |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     member)](api/                 | 5cudaq12spin_handler9unique_idEv) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::spin_op (C++          |
| daq13phase_damping11num_targetsE) |     type)](api/languages/cpp      |
| -   [cudaq::phase_flip_channel    | _api.html#_CPPv4N5cudaq7spin_opE) |
|     (C++                          | -   [cudaq::spin_op_term (C++     |
|     clas                          |                                   |
| s)](api/languages/cpp_api.html#_C |    type)](api/languages/cpp_api.h |
| PPv4N5cudaq18phase_flip_channelE) | tml#_CPPv4N5cudaq12spin_op_termE) |
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
| -   [estimate_resources() (in     |                                   |
|     module                        | property)](api/languages/python_a |
|                                   | pi.html#cudaq.operators.MatrixOpe |
|    cudaq)](api/languages/python_a | ratorElement.expected_dimensions) |
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
| udaq.operators.fermion)](api/lang | -   [from_json()                  |
| uages/python_api.html#cudaq.opera |     (cudaq.PyKernelDecorator      |
| tors.fermion.FermionOperatorTerm) |     static                        |
| -   [final_expectation_values     |     method)                       |
|     (cudaq.EvolveResult           | ](api/languages/python_api.html#c |
|     attribute)](api/lang          | udaq.PyKernelDecorator.from_json) |
| uages/python_api.html#cudaq.Evolv | -   [from_matrices                |
| eResult.final_expectation_values) |     (cudaq.DEMResult              |
| -   [final_state                  |     attrib                        |
|     (cudaq.EvolveResult           | ute)](api/languages/python_api.ht |
|     attribu                       | ml#cudaq.DEMResult.from_matrices) |
| te)](api/languages/python_api.htm | -   [from_word                    |
| l#cudaq.EvolveResult.final_state) |     (                             |
|                                   | cudaq.operators.spin.SpinOperator |
|                                   |     attribute)](api/lang          |
|                                   | uages/python_api.html#cudaq.opera |
|                                   | tors.spin.SpinOperator.from_word) |
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
| -   [get_precision (cudaq.Target  |                                   |
|     att                           |                                   |
| ribute)](api/languages/python_api |                                   |
| .html#cudaq.Target.get_precision) |                                   |
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
| -   [num_available_gpus() (in     |     (cudaq.ptsbe.KrausTrajectory  |
|     module                        |     property)](ap                 |
|                                   | i/languages/python_api.html#cudaq |
|    cudaq)](api/languages/python_a | .ptsbe.KrausTrajectory.num_shots) |
| pi.html#cudaq.num_available_gpus) | -   [num_used_qubits              |
| -   [num_columns                  |     (cudaq.Resources              |
|     (cudaq.ComplexMatrix          |     propert                       |
|     attribut                      | y)](api/languages/python_api.html |
| e)](api/languages/python_api.html | #cudaq.Resources.num_used_qubits) |
| #cudaq.ComplexMatrix.num_columns) | -   [nvqir::MPSSimulationState    |
| -   [num_detectors                |     (C++                          |
|     (cudaq.DEMResult              |     class)]                       |
|     prope                         | (api/languages/cpp_api.html#_CPPv |
| rty)](api/languages/python_api.ht | 4I0EN5nvqir18MPSSimulationStateE) |
| ml#cudaq.DEMResult.num_detectors) | -                                 |
|                                   |  [nvqir::TensorNetSimulationState |
|                                   |     (C++                          |
|                                   |     class)](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4I0EN5 |
|                                   | nvqir24TensorNetSimulationStateE) |
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
| -   [t_depth (cudaq.Resources     | -   [to_matrix()                  |
|                                   |                                   |
|  property)](api/languages/python_ |   (cudaq.operators.ScalarOperator |
| api.html#cudaq.Resources.t_depth) |     method)](api/l                |
| -   [Target (class in             | anguages/python_api.html#cudaq.op |
|     cudaq)](api/langua            | erators.ScalarOperator.to_matrix) |
| ges/python_api.html#cudaq.Target) | -   [to_numpy                     |
| -   [target                       |     (cudaq.ComplexMatrix          |
|     (cudaq.ope                    |     attri                         |
| rators.boson.BosonOperatorElement | bute)](api/languages/python_api.h |
|     property)](api/languages/     | tml#cudaq.ComplexMatrix.to_numpy) |
| python_api.html#cudaq.operators.b |     -   [(cudaq.State             |
| oson.BosonOperatorElement.target) |                                   |
|     -   [(cudaq.operato           |    attribute)](api/languages/pyth |
| rs.fermion.FermionOperatorElement | on_api.html#cudaq.State.to_numpy) |
|                                   | -   [to_sparse_matrix             |
|     property)](api/languages/pyth |     (cu                           |
| on_api.html#cudaq.operators.fermi | daq.operators.boson.BosonOperator |
| on.FermionOperatorElement.target) |     attribute)](api/languages/pyt |
|     -   [(cudaq.o                 | hon_api.html#cudaq.operators.boso |
| perators.spin.SpinOperatorElement | n.BosonOperator.to_sparse_matrix) |
|         property)](api/language   |     -   [(cudaq.                  |
| s/python_api.html#cudaq.operators | operators.boson.BosonOperatorTerm |
| .spin.SpinOperatorElement.target) |                                   |
| -   [targets                      | attribute)](api/languages/python_ |
|     (cudaq.ptsbe.TraceInstruction | api.html#cudaq.operators.boson.Bo |
|     property)](a                  | sonOperatorTerm.to_sparse_matrix) |
| pi/languages/python_api.html#cuda |     -   [(cudaq.                  |
| q.ptsbe.TraceInstruction.targets) | operators.fermion.FermionOperator |
| -   [Tensor (class in             |                                   |
|     cudaq)](api/langua            | attribute)](api/languages/python_ |
| ges/python_api.html#cudaq.Tensor) | api.html#cudaq.operators.fermion. |
| -   [term_count                   | FermionOperator.to_sparse_matrix) |
|     (cu                           |     -   [(cudaq.oper              |
| daq.operators.boson.BosonOperator | ators.fermion.FermionOperatorTerm |
|     property)](api/languag        |         attr                      |
| es/python_api.html#cudaq.operator | ibute)](api/languages/python_api. |
| s.boson.BosonOperator.term_count) | html#cudaq.operators.fermion.Ferm |
|     -   [(cudaq.                  | ionOperatorTerm.to_sparse_matrix) |
| operators.fermion.FermionOperator |     -   [(                        |
|                                   | cudaq.operators.spin.SpinOperator |
|        property)](api/languages/p |                                   |
| ython_api.html#cudaq.operators.fe |       attribute)](api/languages/p |
| rmion.FermionOperator.term_count) | ython_api.html#cudaq.operators.sp |
|     -                             | in.SpinOperator.to_sparse_matrix) |
|  [(cudaq.operators.MatrixOperator |     -   [(cuda                    |
|         property)](api/la         | q.operators.spin.SpinOperatorTerm |
| nguages/python_api.html#cudaq.ope |                                   |
| rators.MatrixOperator.term_count) |   attribute)](api/languages/pytho |
|     -   [(                        | n_api.html#cudaq.operators.spin.S |
| cudaq.operators.spin.SpinOperator | pinOperatorTerm.to_sparse_matrix) |
|         property)](api/langu      | -   [to_string                    |
| ages/python_api.html#cudaq.operat |     (cudaq.ope                    |
| ors.spin.SpinOperator.term_count) | rators.boson.BosonOperatorElement |
|     -   [(cuda                    |     attribute)](api/languages/pyt |
| q.operators.spin.SpinOperatorTerm | hon_api.html#cudaq.operators.boso |
|         property)](api/languages  | n.BosonOperatorElement.to_string) |
| /python_api.html#cudaq.operators. |     -   [(cudaq.operato           |
| spin.SpinOperatorTerm.term_count) | rs.fermion.FermionOperatorElement |
| -   [term_id                      |                                   |
|     (cudaq.                       | attribute)](api/languages/python_ |
| operators.boson.BosonOperatorTerm | api.html#cudaq.operators.fermion. |
|     property)](api/language       | FermionOperatorElement.to_string) |
| s/python_api.html#cudaq.operators |     -   [(cuda                    |
| .boson.BosonOperatorTerm.term_id) | q.operators.MatrixOperatorElement |
|     -   [(cudaq.oper              |         attribute)](api/language  |
| ators.fermion.FermionOperatorTerm | s/python_api.html#cudaq.operators |
|                                   | .MatrixOperatorElement.to_string) |
|       property)](api/languages/py |     -   [(cudaq.o                 |
| thon_api.html#cudaq.operators.fer | perators.spin.SpinOperatorElement |
| mion.FermionOperatorTerm.term_id) |                                   |
|     -   [(c                       |       attribute)](api/languages/p |
| udaq.operators.MatrixOperatorTerm | ython_api.html#cudaq.operators.sp |
|         property)](api/lan        | in.SpinOperatorElement.to_string) |
| guages/python_api.html#cudaq.oper | -   [TraceInstruction (class in   |
| ators.MatrixOperatorTerm.term_id) |     cudaq.p                       |
|     -   [(cuda                    | tsbe)](api/languages/python_api.h |
| q.operators.spin.SpinOperatorTerm | tml#cudaq.ptsbe.TraceInstruction) |
|         property)](api/langua     | -   [TraceInstructionType (class  |
| ges/python_api.html#cudaq.operato |     in                            |
| rs.spin.SpinOperatorTerm.term_id) |     cudaq.ptsbe                   |
| -   [to_bools() (in module        | )](api/languages/python_api.html# |
|     cudaq)](api/language          | cudaq.ptsbe.TraceInstructionType) |
| s/python_api.html#cudaq.to_bools) | -   [trajectories                 |
| -   [to_dict (cudaq.Resources     |                                   |
|                                   |   (cudaq.ptsbe.PTSBEExecutionData |
| attribute)](api/languages/python_ |     property)](api/lang           |
| api.html#cudaq.Resources.to_dict) | uages/python_api.html#cudaq.ptsbe |
| -   [to_json                      | .PTSBEExecutionData.trajectories) |
|     (                             | -   [trajectory_id                |
| cudaq.operators.spin.SpinOperator |     (cudaq.ptsbe.KrausTrajectory  |
|     attribute)](api/la            |     property)](api/la             |
| nguages/python_api.html#cudaq.ope | nguages/python_api.html#cudaq.pts |
| rators.spin.SpinOperator.to_json) | be.KrausTrajectory.trajectory_id) |
|     -   [(cuda                    | -   [translate() (in module       |
| q.operators.spin.SpinOperatorTerm |     cudaq)](api/languages         |
|         attribute)](api/langua    | /python_api.html#cudaq.translate) |
| ges/python_api.html#cudaq.operato | -   [trim                         |
| rs.spin.SpinOperatorTerm.to_json) |     (cu                           |
| -   [to_json()                    | daq.operators.boson.BosonOperator |
|     (cudaq.PyKernelDecorator      |     attribute)](api/l             |
|     metho                         | anguages/python_api.html#cudaq.op |
| d)](api/languages/python_api.html | erators.boson.BosonOperator.trim) |
| #cudaq.PyKernelDecorator.to_json) |     -   [(cudaq.                  |
| -   [to_matrix                    | operators.fermion.FermionOperator |
|     (cu                           |         attribute)](api/langu     |
| daq.operators.boson.BosonOperator | ages/python_api.html#cudaq.operat |
|     attribute)](api/langua        | ors.fermion.FermionOperator.trim) |
| ges/python_api.html#cudaq.operato |     -                             |
| rs.boson.BosonOperator.to_matrix) |  [(cudaq.operators.MatrixOperator |
|     -   [(cudaq.ope               |         attribute)](              |
| rators.boson.BosonOperatorElement | api/languages/python_api.html#cud |
|                                   | aq.operators.MatrixOperator.trim) |
|     attribute)](api/languages/pyt |     -   [(                        |
| hon_api.html#cudaq.operators.boso | cudaq.operators.spin.SpinOperator |
| n.BosonOperatorElement.to_matrix) |         attribute)](api           |
|     -   [(cudaq.                  | /languages/python_api.html#cudaq. |
| operators.boson.BosonOperatorTerm | operators.spin.SpinOperator.trim) |
|                                   | -   [type                         |
|        attribute)](api/languages/ |     (c                            |
| python_api.html#cudaq.operators.b | udaq.ptsbe.ShotAllocationStrategy |
| oson.BosonOperatorTerm.to_matrix) |     property)](api/               |
|     -   [(cudaq.                  | languages/python_api.html#cudaq.p |
| operators.fermion.FermionOperator | tsbe.ShotAllocationStrategy.type) |
|                                   |     -                             |
|        attribute)](api/languages/ |    [(cudaq.ptsbe.TraceInstruction |
| python_api.html#cudaq.operators.f |         property)                 |
| ermion.FermionOperator.to_matrix) | ](api/languages/python_api.html#c |
|     -   [(cudaq.operato           | udaq.ptsbe.TraceInstruction.type) |
| rs.fermion.FermionOperatorElement | -   [type_to_str()                |
|                                   |     (cudaq.PyKernelDecorator      |
| attribute)](api/languages/python_ |     static                        |
| api.html#cudaq.operators.fermion. |     method)](                     |
| FermionOperatorElement.to_matrix) | api/languages/python_api.html#cud |
|     -   [(cudaq.oper              | aq.PyKernelDecorator.type_to_str) |
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
