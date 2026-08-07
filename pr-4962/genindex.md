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
| -   [canonicalize                 | -   [cudaq                        |
|     (cu                           | ::phase_flip_channel::num_targets |
| daq.operators.boson.BosonOperator |     (C++                          |
|     attribute)](api/languages     |     member)](api/langu            |
| /python_api.html#cudaq.operators. | ages/cpp_api.html#_CPPv4N5cudaq18 |
| boson.BosonOperator.canonicalize) | phase_flip_channel11num_targetsE) |
|     -   [(cudaq.                  | -   [cudaq::product_op (C++       |
| operators.boson.BosonOperatorTerm |                                   |
|                                   |  class)](api/languages/cpp_api.ht |
|     attribute)](api/languages/pyt | ml#_CPPv4I0EN5cudaq10product_opE) |
| hon_api.html#cudaq.operators.boso | -   [cudaq::product_op::begin     |
| n.BosonOperatorTerm.canonicalize) |     (C++                          |
|     -   [(cudaq.                  |     functio                       |
| operators.fermion.FermionOperator | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4NK5cudaq10product_op5beginEv) |
|     attribute)](api/languages/pyt | -                                 |
| hon_api.html#cudaq.operators.ferm |  [cudaq::product_op::canonicalize |
| ion.FermionOperator.canonicalize) |     (C++                          |
|     -   [(cudaq.oper              |     func                          |
| ators.fermion.FermionOperatorTerm | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq10product_op12canon |
| attribute)](api/languages/python_ | icalizeERKNSt3setINSt6size_tEEE), |
| api.html#cudaq.operators.fermion. |     [\[1\]](api                   |
| FermionOperatorTerm.canonicalize) | /languages/cpp_api.html#_CPPv4N5c |
|     -                             | udaq10product_op12canonicalizeEv) |
|  [(cudaq.operators.MatrixOperator | -   [                             |
|         attribute)](api/lang      | cudaq::product_op::const_iterator |
| uages/python_api.html#cudaq.opera |     (C++                          |
| tors.MatrixOperator.canonicalize) |     struct)](api/                 |
|     -   [(c                       | languages/cpp_api.html#_CPPv4N5cu |
| udaq.operators.MatrixOperatorTerm | daq10product_op14const_iteratorE) |
|         attribute)](api/language  | -   [cudaq::product_o             |
| s/python_api.html#cudaq.operators | p::const_iterator::const_iterator |
| .MatrixOperatorTerm.canonicalize) |     (C++                          |
|     -   [(                        |     fu                            |
| cudaq.operators.spin.SpinOperator | nction)](api/languages/cpp_api.ht |
|         attribute)](api/languag   | ml#_CPPv4N5cudaq10product_op14con |
| es/python_api.html#cudaq.operator | st_iterator14const_iteratorEPK10p |
| s.spin.SpinOperator.canonicalize) | roduct_opI9HandlerTyENSt6size_tE) |
|     -   [(cuda                    | -   [cudaq::produ                 |
| q.operators.spin.SpinOperatorTerm | ct_op::const_iterator::operator!= |
|                                   |     (C++                          |
|       attribute)](api/languages/p |     fun                           |
| ython_api.html#cudaq.operators.sp | ction)](api/languages/cpp_api.htm |
| in.SpinOperatorTerm.canonicalize) | l#_CPPv4NK5cudaq10product_op14con |
| -   [captured_variables()         | st_iteratorneERK14const_iterator) |
|     (cudaq.PyKernelDecorator      | -   [cudaq::produ                 |
|     method)](api/lan              | ct_op::const_iterator::operator\* |
| guages/python_api.html#cudaq.PyKe |     (C++                          |
| rnelDecorator.captured_variables) |     function)](api/lang           |
| -   [CentralDifference (class in  | uages/cpp_api.html#_CPPv4NK5cudaq |
|     cudaq.gradients)              | 10product_op14const_iteratormlEv) |
| ](api/languages/python_api.html#c | -   [cudaq::produ                 |
| udaq.gradients.CentralDifference) | ct_op::const_iterator::operator++ |
| -   [channel                      |     (C++                          |
|     (cudaq.ptsbe.TraceInstruction |     function)](api/lang           |
|     property)](a                  | uages/cpp_api.html#_CPPv4N5cudaq1 |
| pi/languages/python_api.html#cuda | 0product_op14const_iteratorppEi), |
| q.ptsbe.TraceInstruction.channel) |     [\[1\]](api/lan               |
| -   [circuit_location             | guages/cpp_api.html#_CPPv4N5cudaq |
|     (cudaq.ptsbe.KrausSelection   | 10product_op14const_iteratorppEv) |
|     property)](api/lang           | -   [cudaq::produc                |
| uages/python_api.html#cudaq.ptsbe | t_op::const_iterator::operator\-- |
| .KrausSelection.circuit_location) |     (C++                          |
| -   [clear (cudaq.Resources       |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|   attribute)](api/languages/pytho | 0product_op14const_iteratormmEi), |
| n_api.html#cudaq.Resources.clear) |     [\[1\]](api/lan               |
|     -   [(cudaq.SampleResult      | guages/cpp_api.html#_CPPv4N5cudaq |
|         a                         | 10product_op14const_iteratormmEv) |
| ttribute)](api/languages/python_a | -   [cudaq::produc                |
| pi.html#cudaq.SampleResult.clear) | t_op::const_iterator::operator-\> |
| -   [COBYLA (class in             |     (C++                          |
|     cudaq.o                       |     function)](api/lan            |
| ptimizers)](api/languages/python_ | guages/cpp_api.html#_CPPv4N5cudaq |
| api.html#cudaq.optimizers.COBYLA) | 10product_op14const_iteratorptEv) |
| -   [coefficient                  | -   [cudaq::produ                 |
|     (cudaq.                       | ct_op::const_iterator::operator== |
| operators.boson.BosonOperatorTerm |     (C++                          |
|     property)](api/languages/py   |     fun                           |
| thon_api.html#cudaq.operators.bos | ction)](api/languages/cpp_api.htm |
| on.BosonOperatorTerm.coefficient) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(cudaq.oper              | st_iteratoreqERK14const_iterator) |
| ators.fermion.FermionOperatorTerm | -   [cudaq::product_op::degrees   |
|                                   |     (C++                          |
|   property)](api/languages/python |     function)                     |
| _api.html#cudaq.operators.fermion | ](api/languages/cpp_api.html#_CPP |
| .FermionOperatorTerm.coefficient) | v4NK5cudaq10product_op7degreesEv) |
|     -   [(c                       | -   [cudaq::product_op::dump (C++ |
| udaq.operators.MatrixOperatorTerm |     functi                        |
|         property)](api/languag    | on)](api/languages/cpp_api.html#_ |
| es/python_api.html#cudaq.operator | CPPv4NK5cudaq10product_op4dumpEv) |
| s.MatrixOperatorTerm.coefficient) | -   [cudaq::product_op::end (C++  |
|     -   [(cuda                    |     funct                         |
| q.operators.spin.SpinOperatorTerm | ion)](api/languages/cpp_api.html# |
|         property)](api/languages/ | _CPPv4NK5cudaq10product_op3endEv) |
| python_api.html#cudaq.operators.s | -   [c                            |
| pin.SpinOperatorTerm.coefficient) | udaq::product_op::get_coefficient |
| -   [col_count                    |     (C++                          |
|     (cudaq.KrausOperator          |     function)](api/lan            |
|     prope                         | guages/cpp_api.html#_CPPv4NK5cuda |
| rty)](api/languages/python_api.ht | q10product_op15get_coefficientEv) |
| ml#cudaq.KrausOperator.col_count) | -                                 |
| -   [compile()                    |   [cudaq::product_op::get_term_id |
|     (cudaq.PyKernelDecorator      |     (C++                          |
|     metho                         |     function)](api                |
| d)](api/languages/python_api.html | /languages/cpp_api.html#_CPPv4NK5 |
| #cudaq.PyKernelDecorator.compile) | cudaq10product_op11get_term_idEv) |
| -   [compiledModuleCache()        | -                                 |
|     (cudaq.PyKernelDecorator      |   [cudaq::product_op::is_identity |
|     method)](api/lang             |     (C++                          |
| uages/python_api.html#cudaq.PyKer |     function)](api                |
| nelDecorator.compiledModuleCache) | /languages/cpp_api.html#_CPPv4NK5 |
| -   [ComplexMatrix (class in      | cudaq10product_op11is_identityEv) |
|     cudaq)](api/languages/pyt     | -   [cudaq::product_op::num_ops   |
| hon_api.html#cudaq.ComplexMatrix) |     (C++                          |
| -   [compute                      |     function)                     |
|     (                             | ](api/languages/cpp_api.html#_CPP |
| cudaq.gradients.CentralDifference | v4NK5cudaq10product_op7num_opsEv) |
|     attribute)](api/la            | -                                 |
| nguages/python_api.html#cudaq.gra |    [cudaq::product_op::operator\* |
| dients.CentralDifference.compute) |     (C++                          |
|     -   [(                        |     function)](api/languages/     |
| cudaq.gradients.ForwardDifference | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|         attribute)](api/la        | oduct_opmlE10product_opI1TERK15sc |
| nguages/python_api.html#cudaq.gra | alar_operatorRK10product_opI1TE), |
| dients.ForwardDifference.compute) |     [\[1\]](api/languages/        |
|     -                             | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|  [(cudaq.gradients.ParameterShift | oduct_opmlE10product_opI1TERK15sc |
|         attribute)](api           | alar_operatorRR10product_opI1TE), |
| /languages/python_api.html#cudaq. |     [\[2\]](api/languages/        |
| gradients.ParameterShift.compute) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [const()                      | oduct_opmlE10product_opI1TERR15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|   (cudaq.operators.ScalarOperator |     [\[3\]](api/languages/        |
|     class                         | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     method)](a                    | oduct_opmlE10product_opI1TERR15sc |
| pi/languages/python_api.html#cuda | alar_operatorRR10product_opI1TE), |
| q.operators.ScalarOperator.const) |     [\[4\]](api/                  |
| -   [controls                     | languages/cpp_api.html#_CPPv4I0EN |
|     (cudaq.ptsbe.TraceInstruction | 5cudaq10product_opmlE6sum_opI1TER |
|     property)](ap                 | K15scalar_operatorRK6sum_opI1TE), |
| i/languages/python_api.html#cudaq |     [\[5\]](api/                  |
| .ptsbe.TraceInstruction.controls) | languages/cpp_api.html#_CPPv4I0EN |
| -   [copy                         | 5cudaq10product_opmlE6sum_opI1TER |
|     (cu                           | K15scalar_operatorRR6sum_opI1TE), |
| daq.operators.boson.BosonOperator |     [\[6\]](api/                  |
|     attribute)](api/l             | languages/cpp_api.html#_CPPv4I0EN |
| anguages/python_api.html#cudaq.op | 5cudaq10product_opmlE6sum_opI1TER |
| erators.boson.BosonOperator.copy) | R15scalar_operatorRK6sum_opI1TE), |
|     -   [(cudaq.                  |     [\[7\]](api/                  |
| operators.boson.BosonOperatorTerm | languages/cpp_api.html#_CPPv4I0EN |
|         attribute)](api/langu     | 5cudaq10product_opmlE6sum_opI1TER |
| ages/python_api.html#cudaq.operat | R15scalar_operatorRR6sum_opI1TE), |
| ors.boson.BosonOperatorTerm.copy) |     [\[8\]](api/languages         |
|     -   [(cudaq.                  | /cpp_api.html#_CPPv4NK5cudaq10pro |
| operators.fermion.FermionOperator | duct_opmlERK6sum_opI9HandlerTyE), |
|         attribute)](api/langu     |     [\[9\]](api/languages/cpp_a   |
| ages/python_api.html#cudaq.operat | pi.html#_CPPv4NKR5cudaq10product_ |
| ors.fermion.FermionOperator.copy) | opmlERK10product_opI9HandlerTyE), |
|     -   [(cudaq.oper              |     [\[10\]](api/language         |
| ators.fermion.FermionOperatorTerm | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|         attribute)](api/languages | roduct_opmlERK15scalar_operator), |
| /python_api.html#cudaq.operators. |     [\[11\]](api/languages/cpp_a  |
| fermion.FermionOperatorTerm.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -                             | opmlERR10product_opI9HandlerTyE), |
|  [(cudaq.operators.MatrixOperator |     [\[12\]](api/language         |
|         attribute)](              | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| api/languages/python_api.html#cud | roduct_opmlERR15scalar_operator), |
| aq.operators.MatrixOperator.copy) |     [\[13\]](api/languages/cpp_   |
|     -   [(c                       | api.html#_CPPv4NO5cudaq10product_ |
| udaq.operators.MatrixOperatorTerm | opmlERK10product_opI9HandlerTyE), |
|         attribute)](api/          |     [\[14\]](api/languag          |
| languages/python_api.html#cudaq.o | es/cpp_api.html#_CPPv4NO5cudaq10p |
| perators.MatrixOperatorTerm.copy) | roduct_opmlERK15scalar_operator), |
|     -   [(                        |     [\[15\]](api/languages/cpp_   |
| cudaq.operators.spin.SpinOperator | api.html#_CPPv4NO5cudaq10product_ |
|         attribute)](api           | opmlERR10product_opI9HandlerTyE), |
| /languages/python_api.html#cudaq. |     [\[16\]](api/langua           |
| operators.spin.SpinOperator.copy) | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     -   [(cuda                    | product_opmlERR15scalar_operator) |
| q.operators.spin.SpinOperatorTerm | -                                 |
|         attribute)](api/lan       |   [cudaq::product_op::operator\*= |
| guages/python_api.html#cudaq.oper |     (C++                          |
| ators.spin.SpinOperatorTerm.copy) |     function)](api/languages/cpp  |
| -   [count (cudaq.Resources       | _api.html#_CPPv4N5cudaq10product_ |
|                                   | opmLERK10product_opI9HandlerTyE), |
|   attribute)](api/languages/pytho |     [\[1\]](api/langua            |
| n_api.html#cudaq.Resources.count) | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     -   [(cudaq.SampleResult      | roduct_opmLERK15scalar_operator), |
|         a                         |     [\[2\]](api/languages/cp      |
| ttribute)](api/languages/python_a | p_api.html#_CPPv4N5cudaq10product |
| pi.html#cudaq.SampleResult.count) | _opmLERR10product_opI9HandlerTyE) |
| -   [count_controls               | -   [cudaq::product_op::operator+ |
|     (cudaq.Resources              |     (C++                          |
|     attribu                       |     function)](api/langu          |
| te)](api/languages/python_api.htm | ages/cpp_api.html#_CPPv4I0EN5cuda |
| l#cudaq.Resources.count_controls) | q10product_opplE6sum_opI1TERK15sc |
| -   [count_instructions           | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]](api/                  |
|   (cudaq.ptsbe.PTSBEExecutionData | languages/cpp_api.html#_CPPv4I0EN |
|     attribute)](api/languages/    | 5cudaq10product_opplE6sum_opI1TER |
| python_api.html#cudaq.ptsbe.PTSBE | K15scalar_operatorRK6sum_opI1TE), |
| ExecutionData.count_instructions) |     [\[2\]](api/langu             |
| -   [counts (cudaq.ObserveResult  | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     att                           | q10product_opplE6sum_opI1TERK15sc |
| ribute)](api/languages/python_api | alar_operatorRR10product_opI1TE), |
| .html#cudaq.ObserveResult.counts) |     [\[3\]](api/                  |
| -   [csr_spmatrix (C++            | languages/cpp_api.html#_CPPv4I0EN |
|     type)](api/languages/c        | 5cudaq10product_opplE6sum_opI1TER |
| pp_api.html#_CPPv412csr_spmatrix) | K15scalar_operatorRR6sum_opI1TE), |
| -   cudaq                         |     [\[4\]](api/langu             |
|     -   [module](api/langua       | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ges/python_api.html#module-cudaq) | q10product_opplE6sum_opI1TERR15sc |
| -   [cudaq (C++                   | alar_operatorRK10product_opI1TE), |
|     type)](api/lan                |     [\[5\]](api/                  |
| guages/cpp_api.html#_CPPv45cudaq) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq.apply_noise() (in      | 5cudaq10product_opplE6sum_opI1TER |
|     module                        | R15scalar_operatorRK6sum_opI1TE), |
|     cudaq)](api/languages/python_ |     [\[6\]](api/langu             |
| api.html#cudaq.cudaq.apply_noise) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq.boson                   | q10product_opplE6sum_opI1TERR15sc |
|     -   [module](api/languages/py | alar_operatorRR10product_opI1TE), |
| thon_api.html#module-cudaq.boson) |     [\[7\]](api/                  |
| -   cudaq.fermion                 | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opplE6sum_opI1TER |
|   -   [module](api/languages/pyth | R15scalar_operatorRR6sum_opI1TE), |
| on_api.html#module-cudaq.fermion) |     [\[8\]](api/languages/cpp_a   |
| -   cudaq.operators.custom        | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [mo                       | opplERK10product_opI9HandlerTyE), |
| dule](api/languages/python_api.ht |     [\[9\]](api/language          |
| ml#module-cudaq.operators.custom) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   cudaq.spin                    | roduct_opplERK15scalar_operator), |
|     -   [module](api/languages/p  |     [\[10\]](api/languages/       |
| ython_api.html#module-cudaq.spin) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::amplitude_damping     | duct_opplERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     cla                           | pi.html#_CPPv4NKR5cudaq10product_ |
| ss)](api/languages/cpp_api.html#_ | opplERR10product_opI9HandlerTyE), |
| CPPv4N5cudaq17amplitude_dampingE) |     [\[12\]](api/language         |
| -                                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| [cudaq::amplitude_damping_channel | roduct_opplERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     class)](api                   | cpp_api.html#_CPPv4NKR5cudaq10pro |
| /languages/cpp_api.html#_CPPv4N5c | duct_opplERR6sum_opI9HandlerTyE), |
| udaq25amplitude_damping_channelE) |     [\[                           |
| -   [cudaq::amplitud              | 14\]](api/languages/cpp_api.html# |
| e_damping_channel::num_parameters | _CPPv4NKR5cudaq10product_opplEv), |
|     (C++                          |     [\[15\]](api/languages/cpp_   |
|     member)](api/languages/cpp_a  | api.html#_CPPv4NO5cudaq10product_ |
| pi.html#_CPPv4N5cudaq25amplitude_ | opplERK10product_opI9HandlerTyE), |
| damping_channel14num_parametersE) |     [\[16\]](api/languag          |
| -   [cudaq::ampli                 | es/cpp_api.html#_CPPv4NO5cudaq10p |
| tude_damping_channel::num_targets | roduct_opplERK15scalar_operator), |
|     (C++                          |     [\[17\]](api/languages        |
|     member)](api/languages/cp     | /cpp_api.html#_CPPv4NO5cudaq10pro |
| p_api.html#_CPPv4N5cudaq25amplitu | duct_opplERK6sum_opI9HandlerTyE), |
| de_damping_channel11num_targetsE) |     [\[18\]](api/languages/cpp_   |
| -   [cudaq::AnalogRemoteRESTQPU   | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opplERR10product_opI9HandlerTyE), |
|     class                         |     [\[19\]](api/languag          |
| )](api/languages/cpp_api.html#_CP | es/cpp_api.html#_CPPv4NO5cudaq10p |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | roduct_opplERR15scalar_operator), |
| -   [cudaq::apply_noise (C++      |     [\[20\]](api/languages        |
|     function)](api/               | /cpp_api.html#_CPPv4NO5cudaq10pro |
| languages/cpp_api.html#_CPPv4I0Dp | duct_opplERR6sum_opI9HandlerTyE), |
| EN5cudaq11apply_noiseEvDpRR4Args) |     [                             |
| -   [cudaq::async_result (C++     | \[21\]](api/languages/cpp_api.htm |
|     c                             | l#_CPPv4NO5cudaq10product_opplEv) |
| lass)](api/languages/cpp_api.html | -   [cudaq::product_op::operator- |
| #_CPPv4I0EN5cudaq12async_resultE) |     (C++                          |
| -   [cudaq::async_result::get     |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     functi                        | q10product_opmiE6sum_opI1TERK15sc |
| on)](api/languages/cpp_api.html#_ | alar_operatorRK10product_opI1TE), |
| CPPv4N5cudaq12async_result3getEv) |     [\[1\]](api/                  |
| -   [cudaq::async_sample_result   | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     type                          | K15scalar_operatorRK6sum_opI1TE), |
| )](api/languages/cpp_api.html#_CP |     [\[2\]](api/langu             |
| Pv4N5cudaq19async_sample_resultE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::BaseRemoteRESTQPU     | q10product_opmiE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     cla                           |     [\[3\]](api/                  |
| ss)](api/languages/cpp_api.html#_ | languages/cpp_api.html#_CPPv4I0EN |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::bit_flip_channel (C++ | K15scalar_operatorRR6sum_opI1TE), |
|     cl                            |     [\[4\]](api/langu             |
| ass)](api/languages/cpp_api.html# | ages/cpp_api.html#_CPPv4I0EN5cuda |
| _CPPv4N5cudaq16bit_flip_channelE) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq:                       | alar_operatorRK10product_opI1TE), |
| :bit_flip_channel::num_parameters |     [\[5\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     member)](api/langua           | 5cudaq10product_opmiE6sum_opI1TER |
| ges/cpp_api.html#_CPPv4N5cudaq16b | R15scalar_operatorRK6sum_opI1TE), |
| it_flip_channel14num_parametersE) |     [\[6\]](api/langu             |
| -   [cud                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
| aq::bit_flip_channel::num_targets | q10product_opmiE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     member)](api/lan              |     [\[7\]](api/                  |
| guages/cpp_api.html#_CPPv4N5cudaq | languages/cpp_api.html#_CPPv4I0EN |
| 16bit_flip_channel11num_targetsE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::boson_handler (C++    | R15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[8\]](api/languages/cpp_a   |
|  class)](api/languages/cpp_api.ht | pi.html#_CPPv4NKR5cudaq10product_ |
| ml#_CPPv4N5cudaq13boson_handlerE) | opmiERK10product_opI9HandlerTyE), |
| -   [cudaq::boson_op (C++         |     [\[9\]](api/language          |
|     type)](api/languages/cpp_     | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| api.html#_CPPv4N5cudaq8boson_opE) | roduct_opmiERK15scalar_operator), |
| -   [cudaq::boson_op_term (C++    |     [\[10\]](api/languages/       |
|                                   | cpp_api.html#_CPPv4NKR5cudaq10pro |
|   type)](api/languages/cpp_api.ht | duct_opmiERK6sum_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13boson_op_termE) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq::CodeGenConfig (C++    | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opmiERR10product_opI9HandlerTyE), |
| struct)](api/languages/cpp_api.ht |     [\[12\]](api/language         |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::commutation_relations | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     struct)]                      | cpp_api.html#_CPPv4NKR5cudaq10pro |
| (api/languages/cpp_api.html#_CPPv | duct_opmiERR6sum_opI9HandlerTyE), |
| 4N5cudaq21commutation_relationsE) |     [\[                           |
| -   [cudaq::complex (C++          | 14\]](api/languages/cpp_api.html# |
|     type)](api/languages/cpp      | _CPPv4NKR5cudaq10product_opmiEv), |
| _api.html#_CPPv4N5cudaq7complexE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::complex_matrix (C++   | api.html#_CPPv4NO5cudaq10product_ |
|                                   | opmiERK10product_opI9HandlerTyE), |
| class)](api/languages/cpp_api.htm |     [\[16\]](api/languag          |
| l#_CPPv4N5cudaq14complex_matrixE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -                                 | roduct_opmiERK15scalar_operator), |
|   [cudaq::complex_matrix::adjoint |     [\[17\]](api/languages        |
|     (C++                          | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     function)](a                  | duct_opmiERK6sum_opI9HandlerTyE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[18\]](api/languages/cpp_   |
| 5cudaq14complex_matrix7adjointEv) | api.html#_CPPv4NO5cudaq10product_ |
| -   [cudaq::                      | opmiERR10product_opI9HandlerTyE), |
| complex_matrix::diagonal_elements |     [\[19\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     function)](api/languages      | roduct_opmiERR15scalar_operator), |
| /cpp_api.html#_CPPv4NK5cudaq14com |     [\[20\]](api/languages        |
| plex_matrix17diagonal_elementsEi) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::complex_matrix::dump  | duct_opmiERR6sum_opI9HandlerTyE), |
|     (C++                          |     [                             |
|     function)](api/language       | \[21\]](api/languages/cpp_api.htm |
| s/cpp_api.html#_CPPv4NK5cudaq14co | l#_CPPv4NO5cudaq10product_opmiEv) |
| mplex_matrix4dumpERNSt7ostreamE), | -   [cudaq::product_op::operator/ |
|     [\[1\]]                       |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/language       |
| 4NK5cudaq14complex_matrix4dumpEv) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [c                            | roduct_opdvERK15scalar_operator), |
| udaq::complex_matrix::eigenvalues |     [\[1\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/lan            | roduct_opdvERR15scalar_operator), |
| guages/cpp_api.html#_CPPv4NK5cuda |     [\[2\]](api/languag           |
| q14complex_matrix11eigenvaluesEv) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cu                           | roduct_opdvERK15scalar_operator), |
| daq::complex_matrix::eigenvectors |     [\[3\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     function)](api/lang           | product_opdvERR15scalar_operator) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -                                 |
| 14complex_matrix12eigenvectorsEv) |    [cudaq::product_op::operator/= |
| -   [c                            |     (C++                          |
| udaq::complex_matrix::exponential |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     function)](api/la             | product_opdVERK15scalar_operator) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::product_op::operator= |
| q14complex_matrix11exponentialEv) |     (C++                          |
| -                                 |     function)](api/l              |
|  [cudaq::complex_matrix::identity | anguages/cpp_api.html#_CPPv4I00EN |
|     (C++                          | 5cudaq10product_opaSER10product_o |
|     function)](api/languages      | pI9HandlerTyERK10product_opI1TE), |
| /cpp_api.html#_CPPv4N5cudaq14comp |     [\[1\]](api/languages/cpp     |
| lex_matrix8identityEKNSt6size_tE) | _api.html#_CPPv4N5cudaq10product_ |
| -                                 | opaSERK10product_opI9HandlerTyE), |
| [cudaq::complex_matrix::kronecker |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     function)](api/lang           | _opaSERR10product_opI9HandlerTyE) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -                                 |
| daq14complex_matrix9kroneckerE14c |    [cudaq::product_op::operator== |
| omplex_matrix8Iterable8Iterable), |     (C++                          |
|     [\[1\]](api/l                 |     function)](api/languages/cpp  |
| anguages/cpp_api.html#_CPPv4N5cud | _api.html#_CPPv4NK5cudaq10product |
| aq14complex_matrix9kroneckerERK14 | _opeqERK10product_opI9HandlerTyE) |
| complex_matrixRK14complex_matrix) | -                                 |
| -   [cudaq::c                     |  [cudaq::product_op::operator\[\] |
| omplex_matrix::minimal_eigenvalue |     (C++                          |
|     (C++                          |     function)](ap                 |
|     function)](api/languages/     | i/languages/cpp_api.html#_CPPv4NK |
| cpp_api.html#_CPPv4NK5cudaq14comp | 5cudaq10product_opixENSt6size_tE) |
| lex_matrix18minimal_eigenvalueEv) | -                                 |
| -   [                             |    [cudaq::product_op::product_op |
| cudaq::complex_matrix::operator() |     (C++                          |
|     (C++                          |     f                             |
|     function)](api/languages/cpp  | unction)](api/languages/cpp_api.h |
| _api.html#_CPPv4N5cudaq14complex_ | tml#_CPPv4I00EN5cudaq10product_op |
| matrixclENSt6size_tENSt6size_tE), | 10product_opERK10product_opI1TE), |
|     [\[1\]](api/languages/cpp     |     [\[1\]]                       |
| _api.html#_CPPv4NK5cudaq14complex | (api/languages/cpp_api.html#_CPPv |
| _matrixclENSt6size_tENSt6size_tE) | 4I00EN5cudaq10product_op10product |
| -   [                             | _opERK10product_opI1TERKN14matrix |
| cudaq::complex_matrix::operator\* | _handler20commutation_behaviorE), |
|     (C++                          |                                   |
|     function)](api/langua         |   [\[2\]](api/languages/cpp_api.h |
| ges/cpp_api.html#_CPPv4N5cudaq14c | tml#_CPPv4N5cudaq10product_op10pr |
| omplex_matrixmlEN14complex_matrix | oduct_opENSt6size_tENSt6size_tE), |
| 10value_typeERK14complex_matrix), |     [\[3\]](api/languages/cp      |
|     [\[1\]                        | p_api.html#_CPPv4N5cudaq10product |
| ](api/languages/cpp_api.html#_CPP | _op10product_opENSt7complexIdEE), |
| v4N5cudaq14complex_matrixmlERK14c |     [\[4\]](api/l                 |
| omplex_matrixRK14complex_matrix), | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq10product_op10product_opERK10pr |
|  [\[2\]](api/languages/cpp_api.ht | oduct_opI9HandlerTyENSt6size_tE), |
| ml#_CPPv4N5cudaq14complex_matrixm |     [\[5\]](api/l                 |
| lERK14complex_matrixRKNSt6vectorI | anguages/cpp_api.html#_CPPv4N5cud |
| N14complex_matrix10value_typeEEE) | aq10product_op10product_opERR10pr |
| -                                 | oduct_opI9HandlerTyENSt6size_tE), |
| [cudaq::complex_matrix::operator+ |     [\[6\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq10prod |
|     function                      | uct_op10product_opERR9HandlerTy), |
| )](api/languages/cpp_api.html#_CP |     [\[7\]](ap                    |
| Pv4N5cudaq14complex_matrixplERK14 | i/languages/cpp_api.html#_CPPv4N5 |
| complex_matrixRK14complex_matrix) | cudaq10product_op10product_opEd), |
| -                                 |     [\[8\]](a                     |
| [cudaq::complex_matrix::operator- | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq10product_op10product_opEv) |
|     function                      | -   [cuda                         |
| )](api/languages/cpp_api.html#_CP | q::product_op::to_diagonal_matrix |
| Pv4N5cudaq14complex_matrixmiERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     function)](api/               |
| -   [cu                           | languages/cpp_api.html#_CPPv4NK5c |
| daq::complex_matrix::operator\[\] | udaq10product_op18to_diagonal_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|  function)](api/languages/cpp_api | apINSt6stringENSt7complexIdEEEEb) |
| .html#_CPPv4N5cudaq14complex_matr | -   [cudaq::product_op::to_matrix |
| ixixERKNSt6vectorINSt6size_tEEE), |     (C++                          |
|     [\[1\]](api/languages/cpp_api |     funct                         |
| .html#_CPPv4NK5cudaq14complex_mat | ion)](api/languages/cpp_api.html# |
| rixixERKNSt6vectorINSt6size_tEEE) | _CPPv4NK5cudaq10product_op9to_mat |
| -   [cudaq::complex_matrix::power | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)]                    | apINSt6stringENSt7complexIdEEEEb) |
| (api/languages/cpp_api.html#_CPPv | -   [cu                           |
| 4N5cudaq14complex_matrix5powerEi) | daq::product_op::to_sparse_matrix |
| -                                 |     (C++                          |
|  [cudaq::complex_matrix::set_zero |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     function)](ap                 | 5cudaq10product_op16to_sparse_mat |
| i/languages/cpp_api.html#_CPPv4N5 | rixENSt13unordered_mapINSt6size_t |
| cudaq14complex_matrix8set_zeroEv) | ENSt7int64_tEEERKNSt13unordered_m |
| -                                 | apINSt6stringENSt7complexIdEEEEb) |
| [cudaq::complex_matrix::to_string | -   [cudaq::product_op::to_string |
|     (C++                          |     (C++                          |
|     function)](api/               |     function)](                   |
| languages/cpp_api.html#_CPPv4NK5c | api/languages/cpp_api.html#_CPPv4 |
| udaq14complex_matrix9to_stringEv) | NK5cudaq10product_op9to_stringEv) |
| -   [                             | -                                 |
| cudaq::complex_matrix::value_type |  [cudaq::product_op::\~product_op |
|     (C++                          |     (C++                          |
|     type)](api/                   |     fu                            |
| languages/cpp_api.html#_CPPv4N5cu | nction)](api/languages/cpp_api.ht |
| daq14complex_matrix10value_typeE) | ml#_CPPv4N5cudaq10product_opD0Ev) |
| -   [cudaq::contrib (C++          | -   [cudaq::ptsbe (C++            |
|     type)](api/languages/cpp      |     type)](api/languages/c        |
| _api.html#_CPPv4N5cudaq7contribE) | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| -                                 | -   [cudaq::p                     |
| [cudaq::contrib::amplitude_encode | tsbe::ConditionalSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/language       |     class)](api/languag           |
| s/cpp_api.html#_CPPv4N5cudaq7cont | es/cpp_api.html#_CPPv4N5cudaq5pts |
| rib16amplitude_encodeENSt4spanIKN | be27ConditionalSamplingStrategyE) |
| St7complexIdEEEENSt7complexIdEE), | -   [cudaq::ptsbe::C              |
|     [\[1\]](api/language          | onditionalSamplingStrategy::clone |
| s/cpp_api.html#_CPPv4N5cudaq7cont |     (C++                          |
| rib16amplitude_encodeENSt4spanIKN |                                   |
| St7complexIfEEEENSt7complexIdEE), |    function)](api/languages/cpp_a |
|     [\[2\]                        | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| ](api/languages/cpp_api.html#_CPP | ditionalSamplingStrategy5cloneEv) |
| v4N5cudaq7contrib16amplitude_enco | -   [cuda                         |
| deENSt4spanIKdEENSt7complexIdEE), | q::ptsbe::ConditionalSamplingStra |
|     [\[3\]                        | tegy::ConditionalSamplingStrategy |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq7contrib16amplitude_enco |     function)](api/lang           |
| deENSt4spanIKfEENSt7complexIdEE), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|                                   | ptsbe27ConditionalSamplingStrateg |
| [\[4\]](api/languages/cpp_api.htm | y27ConditionalSamplingStrategyE19 |
| l#_CPPv4N5cudaq7contrib16amplitud | TrajectoryPredicateNSt8uint64_tE) |
| e_encodeERK5stateNSt7complexIdEE) | -                                 |
| -                                 |   [cudaq::ptsbe::ConditionalSampl |
|   [cudaq::contrib::angular_encode | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|                                   |     function)](api/language       |
|  function)](api/languages/cpp_api | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| .html#_CPPv4I0EN5cudaq7contrib14a | be27ConditionalSamplingStrategy20 |
| ngular_encodeEvRR6KernelR10QuakeV | generateTrajectoriesENSt4spanIKN6 |
| alueNSt4spanIKdEE12RotationAxis), | detail10NoisePointEEENSt6size_tE) |
|     [\[1\]](api/languages/cpp_api | -   [cudaq::ptsbe::               |
| .html#_CPPv4I0EN5cudaq7contrib14a | ConditionalSamplingStrategy::name |
| ngular_encodeEvRR6KernelR10QuakeV |     (C++                          |
| alueR10QuakeValue12RotationAxis), |     function)](api/languages/cpp_ |
|                                   | api.html#_CPPv4NK5cudaq5ptsbe27Co |
|   [\[2\]](api/languages/cpp_api.h | nditionalSamplingStrategy4nameEv) |
| tml#_CPPv4I0EN5cudaq7contrib14ang | -   [cudaq:                       |
| ular_encodeEvRR6KernelR10QuakeVal | :ptsbe::ConditionalSamplingStrate |
| ueRKNSt6vectorIdEE12RotationAxis) | gy::\~ConditionalSamplingStrategy |
| -   [cudaq::contrib::draw (C++    |     (C++                          |
|     function)                     |     function)](api/languages/     |
| ](api/languages/cpp_api.html#_CPP | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| v4I0DpEN5cudaq7contrib4drawENSt6s | 7ConditionalSamplingStrategyD0Ev) |
| tringERR13QuantumKernelDpRR4Args) | -                                 |
| -                                 | [cudaq::ptsbe::detail::NoisePoint |
| [cudaq::contrib::get_unitary_cmat |     (C++                          |
|     (C++                          |     struct)](a                    |
|     function)](api/languages/cp   | pi/languages/cpp_api.html#_CPPv4N |
| p_api.html#_CPPv4I0DpEN5cudaq7con | 5cudaq5ptsbe6detail10NoisePointE) |
| trib16get_unitary_cmatE14complex_ | -   [cudaq::p                     |
| matrixRR13QuantumKernelDpRR4Args) | tsbe::detail::NoisePoint::channel |
| -   [cudaq::contrib::RotationAxis |     (C++                          |
|     (C++                          |     member)](api/langu            |
|     enum)                         | ages/cpp_api.html#_CPPv4N5cudaq5p |
| ](api/languages/cpp_api.html#_CPP | tsbe6detail10NoisePoint7channelE) |
| v4N5cudaq7contrib12RotationAxisE) | -   [cudaq::ptsbe::det            |
| -                                 | ail::NoisePoint::circuit_location |
|  [cudaq::contrib::RotationAxis::X |     (C++                          |
|     (C++                          |     member)](api/languages/cpp_a  |
|     enumerator)](                 | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| api/languages/cpp_api.html#_CPPv4 | l10NoisePoint16circuit_locationE) |
| N5cudaq7contrib12RotationAxis1XE) | -   [cudaq::p                     |
| -                                 | tsbe::detail::NoisePoint::op_name |
|  [cudaq::contrib::RotationAxis::Y |     (C++                          |
|     (C++                          |     member)](api/langu            |
|     enumerator)](                 | ages/cpp_api.html#_CPPv4N5cudaq5p |
| api/languages/cpp_api.html#_CPPv4 | tsbe6detail10NoisePoint7op_nameE) |
| N5cudaq7contrib12RotationAxis1YE) | -   [cudaq::                      |
| -                                 | ptsbe::detail::NoisePoint::qubits |
|  [cudaq::contrib::RotationAxis::Z |     (C++                          |
|     (C++                          |     member)](api/lang             |
|     enumerator)](                 | uages/cpp_api.html#_CPPv4N5cudaq5 |
| api/languages/cpp_api.html#_CPPv4 | ptsbe6detail10NoisePoint6qubitsE) |
| N5cudaq7contrib12RotationAxis1ZE) | -   [cudaq::                      |
| -   [cudaq::DefaultQPU (C++       | ptsbe::ExhaustiveSamplingStrategy |
|     class)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4N5cudaq10DefaultQPUE) |     class)](api/langua            |
| -   [cudaq::dem_from_kernel (C++  | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     function)](api                | sbe26ExhaustiveSamplingStrategyE) |
| /languages/cpp_api.html#_CPPv4I0D | -   [cudaq::ptsbe::               |
| pEN5cudaq15dem_from_kernelENSt6st | ExhaustiveSamplingStrategy::clone |
| ringERR13QuantumKernelDpRR4Args), |     (C++                          |
|     [                             |     function)](api/languages/cpp_ |
| \[1\]](api/languages/cpp_api.html | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| #_CPPv4I0DpEN5cudaq15dem_from_ker | haustiveSamplingStrategy5cloneEv) |
| nelENSt6stringERR13QuantumKernelP | -   [cu                           |
| KN5cudaq11noise_modelEDpRR4Args), | daq::ptsbe::ExhaustiveSamplingStr |
|     [\[2\]](api/languages/cp      | ategy::ExhaustiveSamplingStrategy |
| p_api.html#_CPPv4I0DpEN5cudaq15de |     (C++                          |
| m_from_kernelENSt6stringERR13Quan |     function)](api/la             |
| tumKernelPKN5cudaq11noise_modelER | nguages/cpp_api.html#_CPPv4N5cuda |
| KN5cudaq11dem_optionsEDpRR4Args), | q5ptsbe26ExhaustiveSamplingStrate |
|     [\[3\]](ap                    | gy26ExhaustiveSamplingStrategyEv) |
| i/languages/cpp_api.html#_CPPv4I0 | -                                 |
| DpEN5cudaq15dem_from_kernelENSt6s |    [cudaq::ptsbe::ExhaustiveSampl |
| tringERR13QuantumKernelPKN5cudaq1 | ingStrategy::generateTrajectories |
| 1noise_modelERKN5cudaq11dem_optio |     (C++                          |
| nsERN5cudaq15M2DSparseMatrixERN5c |     function)](api/languag        |
| udaq15M2OSparseMatrixEDpRR4Args), | es/cpp_api.html#_CPPv4NK5cudaq5pt |
|     [\[4\]](api/language          | sbe26ExhaustiveSamplingStrategy20 |
| s/cpp_api.html#_CPPv4I0DpEN5cudaq | generateTrajectoriesENSt4spanIKN6 |
| 15dem_from_kernelENSt6stringERR13 | detail10NoisePointEEENSt6size_tE) |
| QuantumKernelPKN5cudaq11noise_mod | -   [cudaq::ptsbe:                |
| elERN5cudaq15M2DSparseMatrixERN5c | :ExhaustiveSamplingStrategy::name |
| udaq15M2OSparseMatrixEDpRR4Args), |     (C++                          |
|     [\[5\]](api/languages/cpp_api |     function)](api/languages/cpp  |
| .html#_CPPv4I0DpEN5cudaq15dem_fro | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| m_kernelENSt6stringERR13QuantumKe | xhaustiveSamplingStrategy4nameEv) |
| rnelRN5cudaq15M2DSparseMatrixERN5 | -   [cuda                         |
| cudaq15M2OSparseMatrixEDpRR4Args) | q::ptsbe::ExhaustiveSamplingStrat |
| -   [cudaq::dem_options (C++      | egy::\~ExhaustiveSamplingStrategy |
|                                   |     (C++                          |
|   struct)](api/languages/cpp_api. |     function)](api/languages      |
| html#_CPPv4N5cudaq11dem_optionsE) | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| -   [cudaq::d                     | 26ExhaustiveSamplingStrategyD0Ev) |
| em_options::allow_gauge_detectors | -   [cuda                         |
|     (C++                          | q::ptsbe::OrderedSamplingStrategy |
|     member)](api/language         |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq11dem |     class)](api/lan               |
| _options21allow_gauge_detectorsE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [cudaq::dem_options::appr     | 5ptsbe23OrderedSamplingStrategyE) |
| oximate_disjoint_errors_threshold | -   [cudaq::ptsb                  |
|     (C++                          | e::OrderedSamplingStrategy::clone |
|     memb                          |     (C++                          |
| er)](api/languages/cpp_api.html#_ |     function)](api/languages/c    |
| CPPv4N5cudaq11dem_options37approx | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| imate_disjoint_errors_thresholdE) | 3OrderedSamplingStrategy5cloneEv) |
| -   [cuda                         | -   [cudaq::ptsbe::OrderedSampl   |
| q::dem_options::block_decompositi | ingStrategy::generateTrajectories |
| on_from_introducing_remnant_edges |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     member)](api/lang             | uages/cpp_api.html#_CPPv4NK5cudaq |
| uages/cpp_api.html#_CPPv4N5cudaq1 | 5ptsbe23OrderedSamplingStrategy20 |
| 1dem_options50block_decomposition | generateTrajectoriesENSt4spanIKN6 |
| _from_introducing_remnant_edgesE) | detail10NoisePointEEENSt6size_tE) |
| -   [cud                          | -   [cudaq::pts                   |
| aq::dem_options::decompose_errors | be::OrderedSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/languages/     |
| guages/cpp_api.html#_CPPv4N5cudaq | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| 11dem_options16decompose_errorsE) | 23OrderedSamplingStrategy4nameEv) |
| -                                 | -                                 |
|   [cudaq::dem_options::fold_loops |    [cudaq::ptsbe::OrderedSampling |
|     (C++                          | Strategy::OrderedSamplingStrategy |
|     member)](a                    |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](                   |
| 5cudaq11dem_options10fold_loopsE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::dem_optio             | N5cudaq5ptsbe23OrderedSamplingStr |
| ns::ignore_decomposition_failures | ategy23OrderedSamplingStrategyEv) |
|     (C++                          | -                                 |
|     member)](api/languages/cpp_ap |  [cudaq::ptsbe::OrderedSamplingSt |
| i.html#_CPPv4N5cudaq11dem_options | rategy::\~OrderedSamplingStrategy |
| 29ignore_decomposition_failuresE) |     (C++                          |
| -   [cudaq::dem_opt               |     function)](api/langua         |
| ions::return_measurement_matrices | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     (C++                          | sbe23OrderedSamplingStrategyD0Ev) |
|     member)](api/languages/cpp_   | -   [cudaq::pts                   |
| api.html#_CPPv4N5cudaq11dem_optio | be::ProbabilisticSamplingStrategy |
| ns27return_measurement_matricesE) |     (C++                          |
| -   [cudaq::depolarization1 (C++  |     class)](api/languages         |
|     c                             | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| lass)](api/languages/cpp_api.html | 29ProbabilisticSamplingStrategyE) |
| #_CPPv4N5cudaq15depolarization1E) | -   [cudaq::ptsbe::Pro            |
| -   [cudaq::depolarization2 (C++  | babilisticSamplingStrategy::clone |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |                                   |
| #_CPPv4N5cudaq15depolarization2E) |  function)](api/languages/cpp_api |
| -   [cudaq:                       | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| :depolarization2::depolarization2 | bilisticSamplingStrategy5cloneEv) |
|     (C++                          | -                                 |
|     function)](api/languages/cp   | [cudaq::ptsbe::ProbabilisticSampl |
| p_api.html#_CPPv4N5cudaq15depolar | ingStrategy::generateTrajectories |
| ization215depolarization2EK4real) |     (C++                          |
| -   [cudaq                        |     function)](api/languages/     |
| ::depolarization2::num_parameters | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     (C++                          | 29ProbabilisticSamplingStrategy20 |
|     member)](api/langu            | generateTrajectoriesENSt4spanIKN6 |
| ages/cpp_api.html#_CPPv4N5cudaq15 | detail10NoisePointEEENSt6size_tE) |
| depolarization214num_parametersE) | -   [cudaq::ptsbe::Pr             |
| -   [cu                           | obabilisticSamplingStrategy::name |
| daq::depolarization2::num_targets |     (C++                          |
|     (C++                          |                                   |
|     member)](api/la               |   function)](api/languages/cpp_ap |
| nguages/cpp_api.html#_CPPv4N5cuda | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| q15depolarization211num_targetsE) | abilisticSamplingStrategy4nameEv) |
| -                                 | -   [cudaq::p                     |
|    [cudaq::depolarization_channel | tsbe::ProbabilisticSamplingStrate |
|     (C++                          | gy::ProbabilisticSamplingStrategy |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)]                    |
| N5cudaq22depolarization_channelE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::depol                 | 4N5cudaq5ptsbe29ProbabilisticSamp |
| arization_channel::num_parameters | lingStrategy29ProbabilisticSampli |
|     (C++                          | ngStrategyENSt8optionalINSt8uint6 |
|     member)](api/languages/cp     | 4_tEEENSt8optionalINSt6size_tEEE) |
| p_api.html#_CPPv4N5cudaq22depolar | -   [cudaq::pts                   |
| ization_channel14num_parametersE) | be::ProbabilisticSamplingStrategy |
| -   [cudaq::de                    | ::\~ProbabilisticSamplingStrategy |
| polarization_channel::num_targets |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api/languages        | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| /cpp_api.html#_CPPv4N5cudaq22depo | robabilisticSamplingStrategyD0Ev) |
| larization_channel11num_targetsE) | -                                 |
| -   [cudaq::detail (C++           | [cudaq::ptsbe::PTSBEExecutionData |
|     type)](api/languages/cp       |     (C++                          |
| p_api.html#_CPPv4N5cudaq6detailE) |     struct)](ap                   |
| -   [cudaq::detail::future (C++   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq5ptsbe18PTSBEExecutionDataE) |
|   class)](api/languages/cpp_api.h | -   [cudaq::ptsbe::PTSBE          |
| tml#_CPPv4N5cudaq6detail6futureE) | ExecutionData::count_instructions |
| -                                 |     (C++                          |
|    [cudaq::detail::future::future |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     functi                        | daq5ptsbe18PTSBEExecutionData18co |
| on)](api/languages/cpp_api.html#_ | unt_instructionsE20TraceInstructi |
| CPPv4N5cudaq6detail6future6future | onTypeNSt8optionalINSt6stringEEE) |
| ERNSt6vectorI3JobEERNSt6stringERN | -   [cudaq::ptsbe::P              |
| St3mapINSt6stringENSt6stringEEE), | TSBEExecutionData::get_trajectory |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function                      |
| 6detail6future6futureERR6future), | )](api/languages/cpp_api.html#_CP |
|     [\[2\]                        | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| ](api/languages/cpp_api.html#_CPP | Data14get_trajectoryENSt6size_tE) |
| v4N5cudaq6detail6future6futureEv) | -   [cudaq::ptsbe:                |
| -   [c                            | :PTSBEExecutionData::instructions |
| udaq::detail::kernel_builder_base |     (C++                          |
|     (C++                          |     member)](api/languages/cp     |
|     class)](api/                  | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| languages/cpp_api.html#_CPPv4N5cu | TSBEExecutionData12instructionsE) |
| daq6detail19kernel_builder_baseE) | -   [cudaq::ptsbe:                |
| -   [cudaq::detail::              | :PTSBEExecutionData::trajectories |
| kernel_builder_base::operator\<\< |     (C++                          |
|     (C++                          |     member)](api/languages/cp     |
|     function)](api/langu          | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| ages/cpp_api.html#_CPPv4N5cudaq6d | TSBEExecutionData12trajectoriesE) |
| etail19kernel_builder_baselsERNSt | -   [cudaq::ptsbe::PTSBEOptions   |
| 7ostreamERK19kernel_builder_base) |     (C++                          |
| -                                 |     struc                         |
| [cudaq::detail::KernelBuilderType | t)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
|     class)](ap                    | -   [cudaq::ptsbe::PTSB           |
| i/languages/cpp_api.html#_CPPv4N5 | EOptions::include_sequential_data |
| cudaq6detail17KernelBuilderTypeE) |     (C++                          |
| -   [cudaq::                      |                                   |
| detail::KernelBuilderType::create |    member)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
|     function                      | ptions23include_sequential_dataE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::ptsb                  |
| Pv4N5cudaq6detail17KernelBuilderT | e::PTSBEOptions::max_trajectories |
| ype6createEPN4mlir11MLIRContextE) |     (C++                          |
| -   [cudaq::detail::Ker           |     member)](api/languages/       |
| nelBuilderType::KernelBuilderType | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
|     (C++                          | 2PTSBEOptions16max_trajectoriesE) |
|     function)](api/lan            | -   [cudaq::ptsbe::PT             |
| guages/cpp_api.html#_CPPv4N5cudaq | SBEOptions::return_execution_data |
| 6detail17KernelBuilderType17Kerne |     (C++                          |
| lBuilderTypeERRNSt8functionIFN4ml |     member)](api/languages/cpp_a  |
| ir4TypeEPN4mlir11MLIRContextEEEE) | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| -   [cudaq::detector (C++         | EOptions21return_execution_dataE) |
|     function)](api                | -   [cudaq::pts                   |
| /languages/cpp_api.html#_CPPv4IDp | be::PTSBEOptions::shot_allocation |
| EN5cudaq8detectorEvDpRR8MeasArgs) |     (C++                          |
| -   [cudaq::detectors (C++        |     member)](api/languages        |
|     function)](api/languages/c    | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| pp_api.html#_CPPv4N5cudaq9detecto | 12PTSBEOptions15shot_allocationE) |
| rsERKNSt6vectorI14measure_resultE | -   [cud                          |
| ERKNSt6vectorI14measure_resultEE) | aq::ptsbe::PTSBEOptions::strategy |
| -   [cudaq::diag_matrix_callback  |     (C++                          |
|     (C++                          |     member)](api/l                |
|     class)                        | anguages/cpp_api.html#_CPPv4N5cud |
| ](api/languages/cpp_api.html#_CPP | aq5ptsbe12PTSBEOptions8strategyE) |
| v4N5cudaq20diag_matrix_callbackE) | -   [cudaq::ptsbe::PTSBETrace     |
| -   [cudaq::dyn (C++              |     (C++                          |
|     member)](api/languages        |     t                             |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | ype)](api/languages/cpp_api.html# |
| -   [cudaq::ExecutionContext (C++ | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
|     cl                            | -   [                             |
| ass)](api/languages/cpp_api.html# | cudaq::ptsbe::PTSSamplingStrategy |
| _CPPv4N5cudaq16ExecutionContextE) |     (C++                          |
| -   [c                            |     class)](api                   |
| udaq::ExecutionContext::asyncExec | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq5ptsbe19PTSSamplingStrategyE) |
|     member)](api/                 | -   [cudaq::                      |
| languages/cpp_api.html#_CPPv4N5cu | ptsbe::PTSSamplingStrategy::clone |
| daq16ExecutionContext9asyncExecE) |     (C++                          |
| -   [cud                          |     function)](api/languag        |
| aq::ExecutionContext::asyncResult | es/cpp_api.html#_CPPv4NK5cudaq5pt |
|     (C++                          | sbe19PTSSamplingStrategy5cloneEv) |
|     member)](api/lan              | -   [cudaq::ptsbe::PTSSampl       |
| guages/cpp_api.html#_CPPv4N5cudaq | ingStrategy::generateTrajectories |
| 16ExecutionContext11asyncResultE) |     (C++                          |
| -   [cudaq:                       |     function)](api/               |
| :ExecutionContext::batchIteration | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq5ptsbe19PTSSamplingStrategy20 |
|     member)](api/langua           | generateTrajectoriesENSt4spanIKN6 |
| ges/cpp_api.html#_CPPv4N5cudaq16E | detail10NoisePointEEENSt6size_tE) |
| xecutionContext14batchIterationE) | -   [cudaq:                       |
| -   [cudaq::E                     | :ptsbe::PTSSamplingStrategy::name |
| xecutionContext::canHandleObserve |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     member)](api/language         | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | tsbe19PTSSamplingStrategy4nameEv) |
| cutionContext16canHandleObserveE) | -   [cudaq::ptsbe::PTSSampli      |
| -   [cudaq::Executio              | ngStrategy::\~PTSSamplingStrategy |
| nContext::deferredKernelException |     (C++                          |
|     (C++                          |     function)](api/la             |
|     member)](api/languages/cpp_a  | nguages/cpp_api.html#_CPPv4N5cuda |
| pi.html#_CPPv4N5cudaq16ExecutionC | q5ptsbe19PTSSamplingStrategyD0Ev) |
| ontext23deferredKernelExceptionE) | -   [cudaq::ptsbe::sample (C++    |
| -   [cudaq::E                     |                                   |
| xecutionContext::ExecutionContext |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
|     func                          | mpleE13sample_resultRK14sample_op |
| tion)](api/languages/cpp_api.html | tionsRR13QuantumKernelDpRR4Args), |
| #_CPPv4N5cudaq16ExecutionContext1 |     [\[1\]](api                   |
| 6ExecutionContextERKNSt6stringE), | /languages/cpp_api.html#_CPPv4I0D |
|     [\[1\]](api/languages/        | pEN5cudaq5ptsbe6sampleE13sample_r |
| cpp_api.html#_CPPv4N5cudaq16Execu | esultRKN5cudaq11noise_modelENSt6s |
| tionContext16ExecutionContextERKN | ize_tERR13QuantumKernelDpRR4Args) |
| St6stringENSt6size_tENSt6size_tE) | -   [cudaq::ptsbe::sample_async   |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::expectationValue |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4I |
|     member)](api/language         | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | 9async_sample_resultRK14sample_op |
| cutionContext16expectationValueE) | tionsRR13QuantumKernelDpRR4Args), |
| -   [cudaq::Execu                 |     [\[1\]](api/languages/cp      |
| tionContext::explicitMeasurements | p_api.html#_CPPv4I0DpEN5cudaq5pts |
|     (C++                          | be12sample_asyncE19async_sample_r |
|     member)](api/languages/cp     | esultRKN5cudaq11noise_modelENSt6s |
| p_api.html#_CPPv4N5cudaq16Executi | ize_tERR13QuantumKernelDpRR4Args) |
| onContext20explicitMeasurementsE) | -   [cudaq::ptsbe::sample_options |
| -   [cuda                         |     (C++                          |
| q::ExecutionContext::futureResult |     struct)                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/lang             | v4N5cudaq5ptsbe14sample_optionsE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::ptsbe::sample_result  |
| 6ExecutionContext12futureResultE) |     (C++                          |
| -   [cudaq::ExecutionContext      |     class                         |
| ::hasConditionalsOnMeasureResults | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq5ptsbe13sample_resultE) |
|     mem                           | -   [cudaq::pts                   |
| ber)](api/languages/cpp_api.html# | be::sample_result::execution_data |
| _CPPv4N5cudaq16ExecutionContext31 |     (C++                          |
| hasConditionalsOnMeasureResultsE) |     function)](api/languages/c    |
| -   [cudaq:                       | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| :ExecutionContext::inKernelLaunch | 3sample_result14execution_dataEv) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     member)](api/langua           | sample_result::has_execution_data |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14inKernelLaunchE) |                                   |
| -   [cu                           |    function)](api/languages/cpp_a |
| daq::ExecutionContext::kernelName | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
|     (C++                          | ple_result18has_execution_dataEv) |
|     member)](api/la               | -   [cudaq::pt                    |
| nguages/cpp_api.html#_CPPv4N5cuda | sbe::sample_result::sample_result |
| q16ExecutionContext10kernelNameE) |     (C++                          |
| -   [cud                          |     function)](api/l              |
| aq::ExecutionContext::kernelTrace | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq5ptsbe13sample_result13sample_r |
|     member)](api/lan              | esultERRN5cudaq13sample_resultE), |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 16ExecutionContext11kernelTraceE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq:                       | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| :ExecutionContext::msm_dimensions | sult13sample_resultERRN5cudaq13sa |
|     (C++                          | mple_resultE18PTSBEExecutionData) |
|     member)](api/langua           | -   [cudaq::ptsbe::               |
| ges/cpp_api.html#_CPPv4N5cudaq16E | sample_result::set_execution_data |
| xecutionContext14msm_dimensionsE) |     (C++                          |
| -   [cudaq::                      |     function)](api/               |
| ExecutionContext::msm_prob_err_id | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq5ptsbe13sample_result18set_exe |
|     member)](api/languag          | cution_dataE18PTSBEExecutionData) |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | -   [cud                          |
| ecutionContext15msm_prob_err_idE) | aq::ptsbe::ShotAllocationStrategy |
| -   [cudaq::Ex                    |     (C++                          |
| ecutionContext::msm_probabilities |     struct)](using                |
|     (C++                          | /examples/ptsbe.html#_CPPv4N5cuda |
|     member)](api/languages        | q5ptsbe22ShotAllocationStrategyE) |
| /cpp_api.html#_CPPv4N5cudaq16Exec | -   [cudaq::ptsbe::ShotAllocatio  |
| utionContext17msm_probabilitiesE) | nStrategy::ShotAllocationStrategy |
| -                                 |     (C++                          |
|    [cudaq::ExecutionContext::name |     function)                     |
|     (C++                          | ](using/examples/ptsbe.html#_CPPv |
|     member)]                      | 4N5cudaq5ptsbe22ShotAllocationStr |
| (api/languages/cpp_api.html#_CPPv | ategy22ShotAllocationStrategyE4Ty |
| 4N5cudaq16ExecutionContext4nameE) | pedNSt8optionalINSt8uint64_tEEE), |
| -   [cu                           |     [\[1\                         |
| daq::ExecutionContext::noiseModel | ]](using/examples/ptsbe.html#_CPP |
|     (C++                          | v4N5cudaq5ptsbe22ShotAllocationSt |
|     member)](api/la               | rategy22ShotAllocationStrategyEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::pt                    |
| q16ExecutionContext10noiseModelE) | sbe::ShotAllocationStrategy::Type |
| -   [cudaq::Exe                   |     (C++                          |
| cutionContext::numberTrajectories |     enum)](using/exam             |
|     (C++                          | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|     member)](api/languages/       | be22ShotAllocationStrategy4TypeE) |
| cpp_api.html#_CPPv4N5cudaq16Execu | -   [cudaq::ptsbe::ShotAllocatio  |
| tionContext18numberTrajectoriesE) | nStrategy::Type::HIGH_WEIGHT_BIAS |
| -   [c                            |     (C++                          |
| udaq::ExecutionContext::optResult |     enumerat                      |
|     (C++                          | or)](using/examples/ptsbe.html#_C |
|     member)](api/                 | PPv4N5cudaq5ptsbe22ShotAllocation |
| languages/cpp_api.html#_CPPv4N5cu | Strategy4Type16HIGH_WEIGHT_BIASE) |
| daq16ExecutionContext9optResultE) | -   [cudaq::ptsbe::ShotAllocati   |
| -                                 | onStrategy::Type::LOW_WEIGHT_BIAS |
|   [cudaq::ExecutionContext::qpuId |     (C++                          |
|     (C++                          |     enumera                       |
|     member)](                     | tor)](using/examples/ptsbe.html#_ |
| api/languages/cpp_api.html#_CPPv4 | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| N5cudaq16ExecutionContext5qpuIdE) | nStrategy4Type15LOW_WEIGHT_BIASE) |
| -   [cudaq                        | -   [cudaq::ptsbe::ShotAlloc      |
| ::ExecutionContext::registerNames | ationStrategy::Type::PROPORTIONAL |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     enum                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 | erator)](using/examples/ptsbe.htm |
| ExecutionContext13registerNamesE) | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| -   [cu                           | tionStrategy4Type12PROPORTIONALE) |
| daq::ExecutionContext::reorderIdx | -   [cudaq::ptsbe::Shot           |
|     (C++                          | AllocationStrategy::Type::UNIFORM |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q16ExecutionContext10reorderIdxE) |   enumerator)](using/examples/pts |
| -                                 | be.html#_CPPv4N5cudaq5ptsbe22Shot |
|  [cudaq::ExecutionContext::result | AllocationStrategy4Type7UNIFORME) |
|     (C++                          | -                                 |
|     member)](a                    |   [cudaq::ptsbe::TraceInstruction |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq16ExecutionContext6resultE) |     struct)](                     |
| -                                 | api/languages/cpp_api.html#_CPPv4 |
|   [cudaq::ExecutionContext::shots | N5cudaq5ptsbe16TraceInstructionE) |
|     (C++                          | -   [cudaq:                       |
|     member)](                     | :ptsbe::TraceInstruction::channel |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5shotsE) |     member)](api/lang             |
| -   [cudaq::                      | uages/cpp_api.html#_CPPv4N5cudaq5 |
| ExecutionContext::simulationState | ptsbe16TraceInstruction7channelE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/languag          | ptsbe::TraceInstruction::controls |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15simulationStateE) |     member)](api/langu            |
| -                                 | ages/cpp_api.html#_CPPv4N5cudaq5p |
|    [cudaq::ExecutionContext::spin | tsbe16TraceInstruction8controlsE) |
|     (C++                          | -   [cud                          |
|     member)]                      | aq::ptsbe::TraceInstruction::name |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq16ExecutionContext4spinE) |     member)](api/l                |
| -   [cudaq::                      | anguages/cpp_api.html#_CPPv4N5cud |
| ExecutionContext::totalIterations | aq5ptsbe16TraceInstruction4nameE) |
|     (C++                          | -   [cudaq                        |
|     member)](api/languag          | ::ptsbe::TraceInstruction::params |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15totalIterationsE) |     member)](api/lan              |
| -   [cudaq::ExecutionResult (C++  | guages/cpp_api.html#_CPPv4N5cudaq |
|     st                            | 5ptsbe16TraceInstruction6paramsE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq:                       |
| #_CPPv4N5cudaq15ExecutionResultE) | :ptsbe::TraceInstruction::targets |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::appendResult |     member)](api/lang             |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     functio                       | ptsbe16TraceInstruction7targetsE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::ptsbe::T              |
| PPv4N5cudaq15ExecutionResult12app | raceInstruction::TraceInstruction |
| endResultENSt6stringENSt6size_tE) |     (C++                          |
| -   [cu                           |                                   |
| daq::ExecutionResult::deserialize |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4N5cudaq5ptsbe16Trace |
|     function)                     | Instruction16TraceInstructionE20T |
| ](api/languages/cpp_api.html#_CPP | raceInstructionTypeNSt6stringENSt |
| v4N5cudaq15ExecutionResult11deser | 6vectorINSt6size_tEEENSt6vectorIN |
| ializeERNSt6vectorINSt6size_tEEE) | St6size_tEEENSt6vectorIdEENSt8opt |
| -   [cudaq:                       | ionalIN5cudaq13kraus_channelEEE), |
| :ExecutionResult::ExecutionResult |     [\[1\]](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
|     functio                       | eInstruction16TraceInstructionEv) |
| n)](api/languages/cpp_api.html#_C | -   [cud                          |
| PPv4N5cudaq15ExecutionResult15Exe | aq::ptsbe::TraceInstruction::type |
| cutionResultE16CountsDictionary), |     (C++                          |
|     [\[1\]](api/lan               |     member)](api/l                |
| guages/cpp_api.html#_CPPv4N5cudaq | anguages/cpp_api.html#_CPPv4N5cud |
| 15ExecutionResult15ExecutionResul | aq5ptsbe16TraceInstruction4typeE) |
| tE16CountsDictionaryNSt6stringE), | -   [c                            |
|     [\[2\                         | udaq::ptsbe::TraceInstructionType |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     enum)](api/                   |
| utionResultE16CountsDictionaryd), | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq5ptsbe20TraceInstructionTypeE) |
|    [\[3\]](api/languages/cpp_api. | -   [cudaq::                      |
| html#_CPPv4N5cudaq15ExecutionResu | ptsbe::TraceInstructionType::Gate |
| lt15ExecutionResultENSt6stringE), |     (C++                          |
|     [\[4\                         |     enumerator)](api/langu        |
| ]](api/languages/cpp_api.html#_CP | ages/cpp_api.html#_CPPv4N5cudaq5p |
| Pv4N5cudaq15ExecutionResult15Exec | tsbe20TraceInstructionType4GateE) |
| utionResultERK15ExecutionResult), | -   [cudaq::ptsbe::               |
|     [\[5\]](api/language          | TraceInstructionType::Measurement |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     (C++                          |
| cutionResult15ExecutionResultEd), |                                   |
|     [\[6\]](api/languag           |    enumerator)](api/languages/cpp |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| ecutionResult15ExecutionResultEv) | aceInstructionType11MeasurementE) |
| -   [                             | -   [cudaq::p                     |
| cudaq::ExecutionResult::operator= | tsbe::TraceInstructionType::Noise |
|     (C++                          |     (C++                          |
|     function)](api/languages/     |     enumerator)](api/langua       |
| cpp_api.html#_CPPv4N5cudaq15Execu | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| tionResultaSERK15ExecutionResult) | sbe20TraceInstructionType5NoiseE) |
| -   [c                            | -   [                             |
| udaq::ExecutionResult::operator== | cudaq::ptsbe::TrajectoryPredicate |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     type)](api                    |
| pp_api.html#_CPPv4NK5cudaq15Execu | /languages/cpp_api.html#_CPPv4N5c |
| tionResulteqERK15ExecutionResult) | udaq5ptsbe19TrajectoryPredicateE) |
| -   [cud                          | -   [cudaq::QPU (C++              |
| aq::ExecutionResult::registerName |     class)](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     member)](api/lan              | -   [cudaq::QPU::beginExecution   |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult12registerNameE) |     function                      |
| -   [cudaq                        | )](api/languages/cpp_api.html#_CP |
| ::ExecutionResult::sequentialData | Pv4N5cudaq3QPU14beginExecutionEv) |
|     (C++                          | -   [cuda                         |
|     member)](api/langu            | q::QPU::configureExecutionContext |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| ExecutionResult14sequentialDataE) |     funct                         |
| -   [                             | ion)](api/languages/cpp_api.html# |
| cudaq::ExecutionResult::serialize | _CPPv4NK5cudaq3QPU25configureExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     function)](api/l              | -   [cudaq::QPU::endExecution     |
| anguages/cpp_api.html#_CPPv4NK5cu |     (C++                          |
| daq15ExecutionResult9serializeEv) |     functi                        |
| -   [cudaq::fermion_handler (C++  | on)](api/languages/cpp_api.html#_ |
|     c                             | CPPv4N5cudaq3QPU12endExecutionEv) |
| lass)](api/languages/cpp_api.html | -   [cudaq::QPU::enqueue (C++     |
| #_CPPv4N5cudaq15fermion_handlerE) |     function)](ap                 |
| -   [cudaq::fermion_op (C++       | i/languages/cpp_api.html#_CPPv4N5 |
|     type)](api/languages/cpp_api  | cudaq3QPU7enqueueER11QuantumTask) |
| .html#_CPPv4N5cudaq10fermion_opE) | -   [cud                          |
| -   [cudaq::fermion_op_term (C++  | aq::QPU::finalizeExecutionContext |
|                                   |     (C++                          |
| type)](api/languages/cpp_api.html |     func                          |
| #_CPPv4N5cudaq15fermion_op_termE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::FermioniqQPU (C++     | #_CPPv4NK5cudaq3QPU24finalizeExec |
|                                   | utionContextER16ExecutionContext) |
|   class)](api/languages/cpp_api.h | -   [cudaq::QPU::getCompileTarget |
| tml#_CPPv4N5cudaq12FermioniqQPUE) |     (C++                          |
| -   [cudaq::get_state (C++        |     function)](api/languages/c    |
|                                   | pp_api.html#_CPPv4N5cudaq3QPU16ge |
|    function)](api/languages/cpp_a | tCompileTargetERK13sample_policy) |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | -   [cudaq::QPU::getConnectivity  |
| ateEDaRR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::GPUEmulatedQPU (C++   |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
| class)](api/languages/cpp_api.htm | v4N5cudaq3QPU15getConnectivityEv) |
| l#_CPPv4N5cudaq14GPUEmulatedQPUE) | -                                 |
| -   [cudaq::gradient (C++         | [cudaq::QPU::getExecutionThreadId |
|     class)](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4N5cudaq8gradientE) |     function)](api/               |
| -   [cudaq::gradient::clone (C++  | languages/cpp_api.html#_CPPv4NK5c |
|     fun                           | udaq3QPU20getExecutionThreadIdEv) |
| ction)](api/languages/cpp_api.htm | -   [cudaq::QPU::getNumQubits     |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     (C++                          |
| -   [cudaq::gradient::compute     |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     function)](api/language       | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| s/cpp_api.html#_CPPv4N5cudaq8grad | -   [                             |
| ient7computeERKNSt6vectorIdEERKNS | cudaq::QPU::getRemoteCapabilities |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|     [\[1\]](ap                    |     function)](api/l              |
| i/languages/cpp_api.html#_CPPv4N5 | anguages/cpp_api.html#_CPPv4NK5cu |
| cudaq8gradient7computeERKNSt6vect | daq3QPU21getRemoteCapabilitiesEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QPU::isEmulated (C++  |
| -   [cudaq::gradient::gradient    |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     function)](api/lang           | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -   [cudaq::QPU::isSimulator (C++ |
| daq8gradient8gradientER7KernelT), |     funct                         |
|                                   | ion)](api/languages/cpp_api.html# |
|    [\[1\]](api/languages/cpp_api. | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| html#_CPPv4I00EN5cudaq8gradient8g | -   [cudaq::QPU::onRandomSeedSet  |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\                         |     function)](api/lang           |
| ]](api/languages/cpp_api.html#_CP | uages/cpp_api.html#_CPPv4N5cudaq3 |
| Pv4I00EN5cudaq8gradient8gradientE | QPU15onRandomSeedSetENSt6size_tE) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::QPU::QPU (C++         |
|     [\[3                          |     functio                       |
| \]](api/languages/cpp_api.html#_C | n)](api/languages/cpp_api.html#_C |
| PPv4N5cudaq8gradient8gradientERRN | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| St8functionIFvNSt6vectorIdEEEEE), |                                   |
|     [\[                           |  [\[1\]](api/languages/cpp_api.ht |
| 4\]](api/languages/cpp_api.html#_ | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| CPPv4N5cudaq8gradient8gradientEv) |     [\[2\]](api/languages/cpp_    |
| -   [cudaq::gradient::setArgs     | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     (C++                          | -   [cudaq::QPU::setId (C++       |
|     fu                            |     function                      |
| nction)](api/languages/cpp_api.ht | )](api/languages/cpp_api.html#_CP |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| tArgsEvR13QuantumKernelDpRR4Args) | -   [cudaq::QPU::setShots (C++    |
| -   [cudaq::gradient::setKernel   |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     function)](api/languages/c    | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| pp_api.html#_CPPv4I0EN5cudaq8grad | -   [cudaq::                      |
| ient9setKernelEvR13QuantumKernel) | QPU::supportsExplicitMeasurements |
| -   [cud                          |     (C++                          |
| aq::gradients::central_difference |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq3QPU |
|     class)](api/la                | 28supportsExplicitMeasurementsEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QPU::\~QPU (C++       |
| q9gradients18central_differenceE) |     function)](api/languages/cp   |
| -   [cudaq::gra                   | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| dients::central_difference::clone | -   [cudaq::QPUState (C++         |
|     (C++                          |     class)](api/languages/cpp_    |
|     function)](api/languages      | api.html#_CPPv4N5cudaq8QPUStateE) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::qreg (C++             |
| ents18central_difference5cloneEv) |     class)](api/lan               |
| -   [cudaq::gradi                 | guages/cpp_api.html#_CPPv4I_NSt6s |
| ents::central_difference::compute | ize_tE_NSt6size_tEEN5cudaq4qregE) |
|     (C++                          | -   [cudaq::qreg::back (C++       |
|     function)](                   |     function)                     |
| api/languages/cpp_api.html#_CPPv4 | ](api/languages/cpp_api.html#_CPP |
| N5cudaq9gradients18central_differ | v4N5cudaq4qreg4backENSt6size_tE), |
| ence7computeERKNSt6vectorIdEERKNS |     [\[1\]](api/languages/cpp_ap  |
| t8functionIFdNSt6vectorIdEEEEEd), | i.html#_CPPv4N5cudaq4qreg4backEv) |
|                                   | -   [cudaq::qreg::begin (C++      |
|   [\[1\]](api/languages/cpp_api.h |                                   |
| tml#_CPPv4N5cudaq9gradients18cent |  function)](api/languages/cpp_api |
| ral_difference7computeERKNSt6vect | .html#_CPPv4N5cudaq4qreg5beginEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::qreg::clear (C++      |
| -   [cudaq::gradie                |                                   |
| nts::central_difference::gradient |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     functio                       | -   [cudaq::qreg::front (C++      |
| n)](api/languages/cpp_api.html#_C |     function)]                    |
| PPv4I00EN5cudaq9gradients18centra | (api/languages/cpp_api.html#_CPPv |
| l_difference8gradientER7KernelT), | 4N5cudaq4qreg5frontENSt6size_tE), |
|     [\[1\]](api/langua            |     [\[1\]](api/languages/cpp_api |
| ges/cpp_api.html#_CPPv4I00EN5cuda | .html#_CPPv4N5cudaq4qreg5frontEv) |
| q9gradients18central_difference8g | -   [cudaq::qreg::operator\[\]    |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\]](api/languages/cpp_    |     functi                        |
| api.html#_CPPv4I00EN5cudaq9gradie | on)](api/languages/cpp_api.html#_ |
| nts18central_difference8gradientE | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::qreg::qreg (C++       |
|     [\[3\]](api/languages/cpp     |     function)                     |
| _api.html#_CPPv4N5cudaq9gradients | ](api/languages/cpp_api.html#_CPP |
| 18central_difference8gradientERRN | v4N5cudaq4qreg4qregENSt6size_tE), |
| St8functionIFvNSt6vectorIdEEEEE), |     [\[1\]](api/languages/cpp_ap  |
|     [\[4\]](api/languages/cp      | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| p_api.html#_CPPv4N5cudaq9gradient | -   [cudaq::qreg::size (C++       |
| s18central_difference8gradientEv) |                                   |
| -   [cud                          |  function)](api/languages/cpp_api |
| aq::gradients::forward_difference | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     (C++                          | -   [cudaq::qreg::slice (C++      |
|     class)](api/la                |     function)](api/langu          |
| nguages/cpp_api.html#_CPPv4N5cuda | ages/cpp_api.html#_CPPv4N5cudaq4q |
| q9gradients18forward_differenceE) | reg5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::gra                   | -   [cudaq::qreg::value_type (C++ |
| dients::forward_difference::clone |                                   |
|     (C++                          | type)](api/languages/cpp_api.html |
|     function)](api/languages      | #_CPPv4N5cudaq4qreg10value_typeE) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::qspan (C++            |
| ents18forward_difference5cloneEv) |     class)](api/lang              |
| -   [cudaq::gradi                 | uages/cpp_api.html#_CPPv4I_NSt6si |
| ents::forward_difference::compute | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     (C++                          | -   [cudaq::QuakeValue (C++       |
|     function)](                   |     class)](api/languages/cpp_api |
| api/languages/cpp_api.html#_CPPv4 | .html#_CPPv4N5cudaq10QuakeValueE) |
| N5cudaq9gradients18forward_differ | -   [cudaq::Q                     |
| ence7computeERKNSt6vectorIdEERKNS | uakeValue::canValidateNumElements |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|                                   |     function)](api/languages      |
|   [\[1\]](api/languages/cpp_api.h | /cpp_api.html#_CPPv4N5cudaq10Quak |
| tml#_CPPv4N5cudaq9gradients18forw | eValue22canValidateNumElementsEv) |
| ard_difference7computeERKNSt6vect | -                                 |
| orIdEERNSt6vectorIdEERK7spin_opd) |  [cudaq::QuakeValue::constantSize |
| -   [cudaq::gradie                |     (C++                          |
| nts::forward_difference::gradient |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     functio                       | udaq10QuakeValue12constantSizeEv) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::QuakeValue::dump (C++ |
| PPv4I00EN5cudaq9gradients18forwar |     function)](api/lan            |
| d_difference8gradientER7KernelT), | guages/cpp_api.html#_CPPv4N5cudaq |
|     [\[1\]](api/langua            | 10QuakeValue4dumpERNSt7ostreamE), |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     [\                            |
| q9gradients18forward_difference8g | [1\]](api/languages/cpp_api.html# |
| radientER7KernelTRR10ArgsMapper), | _CPPv4N5cudaq10QuakeValue4dumpEv) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq                        |
| api.html#_CPPv4I00EN5cudaq9gradie | ::QuakeValue::getRequiredElements |
| nts18forward_difference8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/langua         |
|     [\[3\]](api/languages/cpp     | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| _api.html#_CPPv4N5cudaq9gradients | uakeValue19getRequiredElementsEv) |
| 18forward_difference8gradientERRN | -   [cudaq::QuakeValue::getValue  |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[4\]](api/languages/cp      |     function)]                    |
| p_api.html#_CPPv4N5cudaq9gradient | (api/languages/cpp_api.html#_CPPv |
| s18forward_difference8gradientEv) | 4NK5cudaq10QuakeValue8getValueEv) |
| -   [                             | -   [cudaq::QuakeValue::inverse   |
| cudaq::gradients::parameter_shift |     (C++                          |
|     (C++                          |     function)                     |
|     class)](api                   | ](api/languages/cpp_api.html#_CPP |
| /languages/cpp_api.html#_CPPv4N5c | v4NK5cudaq10QuakeValue7inverseEv) |
| udaq9gradients15parameter_shiftE) | -   [cudaq::QuakeValue::isStdVec  |
| -   [cudaq::                      |     (C++                          |
| gradients::parameter_shift::clone |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/langua         | v4N5cudaq10QuakeValue8isStdVecEv) |
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
| -   [c                            |     cl                            |
| udaq::kernel_builder::isArgStdVec | ass)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq16quantum_platformE) |
|     function)](api/languages/cp   | -   [cudaq:                       |
| p_api.html#_CPPv4N5cudaq14kernel_ | :quantum_platform::beginExecution |
| builder11isArgStdVecENSt6size_tE) |     (C++                          |
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
| _vI16operator_handler1TEEbEEEN5cu | -   [c                            |
| daq14matrix_handler14matrix_handl | udaq::sample_result::get_marginal |
| erERK1TRK20commutation_behavior), |     (C++                          |
|     [\[2\]](api/languages/cpp_ap  |     function)](api/languages/cpp_ |
| i.html#_CPPv4N5cudaq14matrix_hand | api.html#_CPPv4NK5cudaq13sample_r |
| ler14matrix_handlerENSt6size_tE), | esult12get_marginalERKNSt6vectorI |
|     [\[3\]](api/                  | NSt6size_tEEEKNSt11string_viewE), |
| languages/cpp_api.html#_CPPv4N5cu |     [\[1\]](api/languages/cpp_    |
| daq14matrix_handler14matrix_handl | api.html#_CPPv4NK5cudaq13sample_r |
| erENSt6stringERKNSt6vectorINSt6si | esult12get_marginalERRKNSt6vector |
| ze_tEEERK20commutation_behavior), | INSt6size_tEEEKNSt11string_viewE) |
|     [\[4\]](api/                  | -   [cuda                         |
| languages/cpp_api.html#_CPPv4N5cu | q::sample_result::get_total_shots |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erENSt6stringERRNSt6vectorINSt6si |     function)](api/langua         |
| ze_tEEERK20commutation_behavior), | ges/cpp_api.html#_CPPv4NK5cudaq13 |
|     [\                            | sample_result15get_total_shotsEv) |
| [5\]](api/languages/cpp_api.html# | -   [cuda                         |
| _CPPv4N5cudaq14matrix_handler14ma | q::sample_result::has_even_parity |
| trix_handlerERK14matrix_handler), |     (C++                          |
|     [                             |     fun                           |
| \[6\]](api/languages/cpp_api.html | ction)](api/languages/cpp_api.htm |
| #_CPPv4N5cudaq14matrix_handler14m | l#_CPPv4N5cudaq13sample_result15h |
| atrix_handlerERR14matrix_handler) | as_even_parityENSt11string_viewE) |
| -                                 | -   [cuda                         |
|  [cudaq::matrix_handler::momentum | q::sample_result::has_expectation |
|     (C++                          |     (C++                          |
|     function)](api/language       |     funct                         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ion)](api/languages/cpp_api.html# |
| rix_handler8momentumENSt6size_tE) | _CPPv4NK5cudaq13sample_result15ha |
| -                                 | s_expectationEKNSt11string_viewE) |
|    [cudaq::matrix_handler::number | -   [cu                           |
|     (C++                          | daq::sample_result::most_probable |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     fun                           |
| atrix_handler6numberENSt6size_tE) | ction)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4NK5cudaq13sample_result13 |
| [cudaq::matrix_handler::operator= | most_probableEKNSt11string_viewE) |
|     (C++                          | -                                 |
|     fun                           | [cudaq::sample_result::operator+= |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     function)](api/langua         |
| NSt7is_sameI1T14matrix_handlerE5v | ges/cpp_api.html#_CPPv4N5cudaq13s |
| alueENSt12is_base_of_vI16operator | ample_resultpLERK13sample_result) |
| _handler1TEEEbEEEN5cudaq14matrix_ | -                                 |
| handleraSER14matrix_handlerRK1T), |  [cudaq::sample_result::operator= |
|     [\[1\]](api/languages         |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq14matr |     function)](api/langua         |
| ix_handleraSERK14matrix_handler), | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     [\[2\]](api/language          | ample_resultaSERR13sample_result) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -                                 |
| rix_handleraSERR14matrix_handler) | [cudaq::sample_result::operator== |
| -   [                             |     (C++                          |
| cudaq::matrix_handler::operator== |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     function)](api/languages      | ample_resulteqERK13sample_result) |
| /cpp_api.html#_CPPv4NK5cudaq14mat | -   [                             |
| rix_handlereqERK14matrix_handler) | cudaq::sample_result::probability |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::parity |     function)](api/lan            |
|     (C++                          | guages/cpp_api.html#_CPPv4NK5cuda |
|     function)](api/langua         | q13sample_result11probabilityENSt |
| ges/cpp_api.html#_CPPv4N5cudaq14m | 11string_viewEKNSt11string_viewE) |
| atrix_handler6parityENSt6size_tE) | -   [cud                          |
| -                                 | aq::sample_result::register_names |
|  [cudaq::matrix_handler::position |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](api/language       | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| s/cpp_api.html#_CPPv4N5cudaq14mat | 3sample_result14register_namesEv) |
| rix_handler8positionENSt6size_tE) | -                                 |
| -   [cudaq::                      |    [cudaq::sample_result::reorder |
| matrix_handler::remove_definition |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     fu                            | ges/cpp_api.html#_CPPv4N5cudaq13s |
| nction)](api/languages/cpp_api.ht | ample_result7reorderERKNSt6vector |
| ml#_CPPv4N5cudaq14matrix_handler1 | INSt6size_tEEEKNSt11string_viewE) |
| 7remove_definitionERKNSt6stringE) | -   [cu                           |
| -                                 | daq::sample_result::sample_result |
|   [cudaq::matrix_handler::squeeze |     (C++                          |
|     (C++                          |     func                          |
|     function)](api/languag        | tion)](api/languages/cpp_api.html |
| es/cpp_api.html#_CPPv4N5cudaq14ma | #_CPPv4N5cudaq13sample_result13sa |
| trix_handler7squeezeENSt6size_tE) | mple_resultERK15ExecutionResult), |
| -   [cudaq::m                     |     [\[1\]](api/la                |
| atrix_handler::to_diagonal_matrix | nguages/cpp_api.html#_CPPv4N5cuda |
|     (C++                          | q13sample_result13sample_resultER |
|     function)](api/lang           | KNSt6vectorI15ExecutionResultEE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |                                   |
| 14matrix_handler18to_diagonal_mat |  [\[2\]](api/languages/cpp_api.ht |
| rixERNSt13unordered_mapINSt6size_ | ml#_CPPv4N5cudaq13sample_result13 |
| tENSt7int64_tEEERKNSt13unordered_ | sample_resultERR13sample_result), |
| mapINSt6stringENSt7complexIdEEEE) |     [                             |
| -                                 | \[3\]](api/languages/cpp_api.html |
| [cudaq::matrix_handler::to_matrix | #_CPPv4N5cudaq13sample_result13sa |
|     (C++                          | mple_resultERR15ExecutionResult), |
|     function)                     |     [\[4\]](api/lan               |
| ](api/languages/cpp_api.html#_CPP | guages/cpp_api.html#_CPPv4N5cudaq |
| v4NK5cudaq14matrix_handler9to_mat | 13sample_result13sample_resultEdR |
| rixERNSt13unordered_mapINSt6size_ | KNSt6vectorI15ExecutionResultEE), |
| tENSt7int64_tEEERKNSt13unordered_ |     [\[5\]](api/lan               |
| mapINSt6stringENSt7complexIdEEEE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -                                 | 13sample_result13sample_resultEv) |
| [cudaq::matrix_handler::to_string | -                                 |
|     (C++                          |  [cudaq::sample_result::serialize |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     function)](api                |
| udaq14matrix_handler9to_stringEb) | /languages/cpp_api.html#_CPPv4NK5 |
| -                                 | cudaq13sample_result9serializeEv) |
| [cudaq::matrix_handler::unique_id | -   [cudaq::sample_result::size   |
|     (C++                          |     (C++                          |
|     function)](api/               |     function)](api/languages/c    |
| languages/cpp_api.html#_CPPv4NK5c | pp_api.html#_CPPv4NK5cudaq13sampl |
| udaq14matrix_handler9unique_idEv) | e_result4sizeEKNSt11string_viewE) |
| -   [cudaq:                       | -   [cudaq::sample_result::to_map |
| :matrix_handler::\~matrix_handler |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     functi                        | _api.html#_CPPv4NK5cudaq13sample_ |
| on)](api/languages/cpp_api.html#_ | result6to_mapEKNSt11string_viewE) |
| CPPv4N5cudaq14matrix_handlerD0Ev) | -   [cuda                         |
| -   [cudaq::matrix_op (C++        | q::sample_result::\~sample_result |
|     type)](api/languages/cpp_a    |     (C++                          |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     funct                         |
| -   [cudaq::matrix_op_term (C++   | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq13sample_resultD0Ev) |
|  type)](api/languages/cpp_api.htm | -   [cudaq::scalar_callback (C++  |
| l#_CPPv4N5cudaq14matrix_op_termE) |     c                             |
| -                                 | lass)](api/languages/cpp_api.html |
|    [cudaq::mdiag_operator_handler | #_CPPv4N5cudaq15scalar_callbackE) |
|     (C++                          | -   [c                            |
|     class)](                      | udaq::scalar_callback::operator() |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22mdiag_operator_handlerE) |     function)](api/language       |
| -   [cudaq::measure_handle (C++   | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|                                   | alar_callbackclERKNSt13unordered_ |
| class)](api/languages/cpp_api.htm | mapINSt6stringENSt7complexIdEEEE) |
| l#_CPPv4N5cudaq14measure_handleE) | -   [                             |
| -   [cudaq::measure_result (C++   | cudaq::scalar_callback::operator= |
|                                   |     (C++                          |
|  type)](api/languages/cpp_api.htm |     function)](api/languages/c    |
| l#_CPPv4N5cudaq14measure_resultE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::mpi (C++              | _callbackaSERK15scalar_callback), |
|     type)](api/languages          |     [\[1\]](api/languages/        |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | cpp_api.html#_CPPv4N5cudaq15scala |
| -   [cudaq::mpi::all_gather (C++  | r_callbackaSERR15scalar_callback) |
|     fu                            | -   [cudaq:                       |
| nction)](api/languages/cpp_api.ht | :scalar_callback::scalar_callback |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     (C++                          |
| RNSt6vectorIdEERKNSt6vectorIdEE), |     function)](api/languag        |
|                                   | es/cpp_api.html#_CPPv4I0_NSt11ena |
|   [\[1\]](api/languages/cpp_api.h | ble_if_tINSt16is_invocable_r_vINS |
| tml#_CPPv4N5cudaq3mpi10all_gather | t7complexIdEE8CallableRKNSt13unor |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | dered_mapINSt6stringENSt7complexI |
| -   [cudaq::mpi::all_reduce (C++  | dEEEEEEbEEEN5cudaq15scalar_callba |
|                                   | ck15scalar_callbackERR8Callable), |
|  function)](api/languages/cpp_api |     [\[1\                         |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | ]](api/languages/cpp_api.html#_CP |
| reduceE1TRK1TRK14BinaryFunction), | Pv4N5cudaq15scalar_callback15scal |
|     [\[1\]](api/langu             | ar_callbackERK15scalar_callback), |
| ages/cpp_api.html#_CPPv4I00EN5cud |     [\[2                          |
| aq3mpi10all_reduceE1TRK1TRK4Func) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::mpi::broadcast (C++   | PPv4N5cudaq15scalar_callback15sca |
|     function)](api/               | lar_callbackERR15scalar_callback) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::scalar_operator (C++  |
| daq3mpi9broadcastERNSt6stringEi), |     c                             |
|     [\[1\]](api/la                | lass)](api/languages/cpp_api.html |
| nguages/cpp_api.html#_CPPv4N5cuda | #_CPPv4N5cudaq15scalar_operatorE) |
| q3mpi9broadcastERNSt6vectorIdEEi) | -                                 |
| -   [cudaq::mpi::finalize (C++    | [cudaq::scalar_operator::evaluate |
|     f                             |     (C++                          |
| unction)](api/languages/cpp_api.h |                                   |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |    function)](api/languages/cpp_a |
| -   [cudaq::mpi::initialize (C++  | pi.html#_CPPv4NK5cudaq15scalar_op |
|     function                      | erator8evaluateERKNSt13unordered_ |
| )](api/languages/cpp_api.html#_CP | mapINSt6stringENSt7complexIdEEEE) |
| Pv4N5cudaq3mpi10initializeEiPPc), | -   [cudaq::scalar_ope            |
|     [                             | rator::get_parameter_descriptions |
| \[1\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq3mpi10initializeEv) |     f                             |
| -   [cudaq::mpi::is_initialized   | unction)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4NK5cudaq15scalar_operat |
|     function                      | or26get_parameter_descriptionsEv) |
| )](api/languages/cpp_api.html#_CP | -   [cu                           |
| Pv4N5cudaq3mpi14is_initializedEv) | daq::scalar_operator::is_constant |
| -   [cudaq::mpi::num_ranks (C++   |     (C++                          |
|     fu                            |     function)](api/lang           |
| nction)](api/languages/cpp_api.ht | uages/cpp_api.html#_CPPv4NK5cudaq |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | 15scalar_operator11is_constantEv) |
| -   [cudaq::mpi::rank (C++        | -   [c                            |
|                                   | udaq::scalar_operator::operator\* |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     function                      |
| -   [cudaq::noise_model (C++      | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_operatormlENSt |
|    class)](api/languages/cpp_api. | 7complexIdEERK15scalar_operator), |
| html#_CPPv4N5cudaq11noise_modelE) |     [\[1\                         |
| -   [cudaq::n                     | ]](api/languages/cpp_api.html#_CP |
| oise_model::add_all_qubit_channel | Pv4N5cudaq15scalar_operatormlENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     function)](api                |     [\[2\]](api/languages/cp      |
| /languages/cpp_api.html#_CPPv4IDp | p_api.html#_CPPv4N5cudaq15scalar_ |
| EN5cudaq11noise_model21add_all_qu | operatormlEdRK15scalar_operator), |
| bit_channelEvRK13kraus_channeli), |     [\[3\]](api/languages/cp      |
|     [\[1\]](api/langua            | p_api.html#_CPPv4N5cudaq15scalar_ |
| ges/cpp_api.html#_CPPv4N5cudaq11n | operatormlEdRR15scalar_operator), |
| oise_model21add_all_qubit_channel |     [\[4\]](api/languages         |
| ERKNSt6stringERK13kraus_channeli) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -                                 | alar_operatormlENSt7complexIdEE), |
|  [cudaq::noise_model::add_channel |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|     funct                         | _operatormlERK15scalar_operator), |
| ion)](api/languages/cpp_api.html# |     [\[6\]]                       |
| _CPPv4IDpEN5cudaq11noise_model11a | (api/languages/cpp_api.html#_CPPv |
| dd_channelEvRK15PredicateFuncTy), | 4NKR5cudaq15scalar_operatormlEd), |
|     [\[1\]](api/languages/cpp_    |     [\[7\]](api/language          |
| api.html#_CPPv4IDpEN5cudaq11noise | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| _model11add_channelEvRKNSt6vector | alar_operatormlENSt7complexIdEE), |
| INSt6size_tEEERK13kraus_channel), |     [\[8\]](api/languages/cp      |
|     [\[2\]](ap                    | p_api.html#_CPPv4NO5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatormlERK15scalar_operator), |
| cudaq11noise_model11add_channelER |     [\[9\                         |
| KNSt6stringERK15PredicateFuncTy), | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4NO5cudaq15scalar_operatormlEd) |
| [\[3\]](api/languages/cpp_api.htm | -   [cu                           |
| l#_CPPv4N5cudaq11noise_model11add | daq::scalar_operator::operator\*= |
| _channelERKNSt6stringERKNSt6vecto |     (C++                          |
| rINSt6size_tEEERK13kraus_channel) |     function)](api/languag        |
| -   [cudaq::noise_model::empty    | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormLENSt7complexIdEE), |
|     function                      |     [\[1\]](api/languages/c       |
| )](api/languages/cpp_api.html#_CP | pp_api.html#_CPPv4N5cudaq15scalar |
| Pv4NK5cudaq11noise_model5emptyEv) | _operatormLERK15scalar_operator), |
| -                                 |     [\[2                          |
| [cudaq::noise_model::get_channels | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatormLEd) |
|     function)](api/l              | -   [                             |
| anguages/cpp_api.html#_CPPv4I0ENK | cudaq::scalar_operator::operator+ |
| 5cudaq11noise_model12get_channels |     (C++                          |
| ENSt6vectorI13kraus_channelEERKNS |     function                      |
| t6vectorINSt6size_tEEERKNSt6vecto | )](api/languages/cpp_api.html#_CP |
| rINSt6size_tEEERKNSt6vectorIdEE), | Pv4N5cudaq15scalar_operatorplENSt |
|     [\[1\]](api/languages/cpp_a   | 7complexIdEERK15scalar_operator), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[1\                         |
| el12get_channelsERKNSt6stringERKN | ]](api/languages/cpp_api.html#_CP |
| St6vectorINSt6size_tEEERKNSt6vect | Pv4N5cudaq15scalar_operatorplENSt |
| orINSt6size_tEEERKNSt6vectorIdEE) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
|  [cudaq::noise_model::noise_model | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatorplEdRK15scalar_operator), |
|     function)](api                |     [\[3\]](api/languages/cp      |
| /languages/cpp_api.html#_CPPv4N5c | p_api.html#_CPPv4N5cudaq15scalar_ |
| udaq11noise_model11noise_modelEv) | operatorplEdRR15scalar_operator), |
| -   [cu                           |     [\[4\]](api/languages         |
| daq::noise_model::PredicateFuncTy | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatorplENSt7complexIdEE), |
|     type)](api/la                 |     [\[5\]](api/languages/cpp     |
| nguages/cpp_api.html#_CPPv4N5cuda | _api.html#_CPPv4NKR5cudaq15scalar |
| q11noise_model15PredicateFuncTyE) | _operatorplERK15scalar_operator), |
| -   [cud                          |     [\[6\]]                       |
| aq::noise_model::register_channel | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatorplEd), |
|     function)](api/languages      |     [\[7\]]                       |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | (api/languages/cpp_api.html#_CPPv |
| noise_model16register_channelEvv) | 4NKR5cudaq15scalar_operatorplEv), |
| -   [cudaq::                      |     [\[8\]](api/language          |
| noise_model::requires_constructor | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatorplENSt7complexIdEE), |
|     type)](api/languages/cp       |     [\[9\]](api/languages/cp      |
| p_api.html#_CPPv4I0DpEN5cudaq11no | p_api.html#_CPPv4NO5cudaq15scalar |
| ise_model20requires_constructorE) | _operatorplERK15scalar_operator), |
| -   [cudaq::noise_model_type (C++ |     [\[10\]                       |
|     e                             | ](api/languages/cpp_api.html#_CPP |
| num)](api/languages/cpp_api.html# | v4NO5cudaq15scalar_operatorplEd), |
| _CPPv4N5cudaq16noise_model_typeE) |     [\[11\                        |
| -   [cudaq::no                    | ]](api/languages/cpp_api.html#_CP |
| ise_model_type::amplitude_damping | Pv4NO5cudaq15scalar_operatorplEv) |
|     (C++                          | -   [c                            |
|     enumerator)](api/languages    | udaq::scalar_operator::operator+= |
| /cpp_api.html#_CPPv4N5cudaq16nois |     (C++                          |
| e_model_type17amplitude_dampingE) |     function)](api/languag        |
| -   [cudaq::noise_mode            | es/cpp_api.html#_CPPv4N5cudaq15sc |
| l_type::amplitude_damping_channel | alar_operatorpLENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     e                             | pp_api.html#_CPPv4N5cudaq15scalar |
| numerator)](api/languages/cpp_api | _operatorpLERK15scalar_operator), |
| .html#_CPPv4N5cudaq16noise_model_ |     [\[2                          |
| type25amplitude_damping_channelE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::n                     | PPv4N5cudaq15scalar_operatorpLEd) |
| oise_model_type::bit_flip_channel | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator- |
|     enumerator)](api/language     |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     function                      |
| se_model_type16bit_flip_channelE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::                      | Pv4N5cudaq15scalar_operatormiENSt |
| noise_model_type::depolarization1 | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](api/languag      | ]](api/languages/cpp_api.html#_CP |
| es/cpp_api.html#_CPPv4N5cudaq16no | Pv4N5cudaq15scalar_operatormiENSt |
| ise_model_type15depolarization1E) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::                      |     [\[2\]](api/languages/cp      |
| noise_model_type::depolarization2 | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRK15scalar_operator), |
|     enumerator)](api/languag      |     [\[3\]](api/languages/cp      |
| es/cpp_api.html#_CPPv4N5cudaq16no | p_api.html#_CPPv4N5cudaq15scalar_ |
| ise_model_type15depolarization2E) | operatormiEdRR15scalar_operator), |
| -   [cudaq::noise_m               |     [\[4\]](api/languages         |
| odel_type::depolarization_channel | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|                                   |     [\[5\]](api/languages/cpp     |
|   enumerator)](api/languages/cpp_ | _api.html#_CPPv4NKR5cudaq15scalar |
| api.html#_CPPv4N5cudaq16noise_mod | _operatormiERK15scalar_operator), |
| el_type22depolarization_channelE) |     [\[6\]]                       |
| -                                 | (api/languages/cpp_api.html#_CPPv |
|  [cudaq::noise_model_type::pauli1 | 4NKR5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[7\]]                       |
|     enumerator)](a                | (api/languages/cpp_api.html#_CPPv |
| pi/languages/cpp_api.html#_CPPv4N | 4NKR5cudaq15scalar_operatormiEv), |
| 5cudaq16noise_model_type6pauli1E) |     [\[8\]](api/language          |
| -                                 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|  [cudaq::noise_model_type::pauli2 | alar_operatormiENSt7complexIdEE), |
|     (C++                          |     [\[9\]](api/languages/cp      |
|     enumerator)](a                | p_api.html#_CPPv4NO5cudaq15scalar |
| pi/languages/cpp_api.html#_CPPv4N | _operatormiERK15scalar_operator), |
| 5cudaq16noise_model_type6pauli2E) |     [\[10\]                       |
| -   [cudaq                        | ](api/languages/cpp_api.html#_CPP |
| ::noise_model_type::phase_damping | v4NO5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[11\                        |
|     enumerator)](api/langu        | ]](api/languages/cpp_api.html#_CP |
| ages/cpp_api.html#_CPPv4N5cudaq16 | Pv4NO5cudaq15scalar_operatormiEv) |
| noise_model_type13phase_dampingE) | -   [c                            |
| -   [cudaq::noi                   | udaq::scalar_operator::operator-= |
| se_model_type::phase_flip_channel |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](api/languages/   | es/cpp_api.html#_CPPv4N5cudaq15sc |
| cpp_api.html#_CPPv4N5cudaq16noise | alar_operatormIENSt7complexIdEE), |
| _model_type18phase_flip_channelE) |     [\[1\]](api/languages/c       |
| -                                 | pp_api.html#_CPPv4N5cudaq15scalar |
| [cudaq::noise_model_type::unknown | _operatormIERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     enumerator)](ap               | \]](api/languages/cpp_api.html#_C |
| i/languages/cpp_api.html#_CPPv4N5 | PPv4N5cudaq15scalar_operatormIEd) |
| cudaq16noise_model_type7unknownE) | -   [                             |
| -                                 | cudaq::scalar_operator::operator/ |
| [cudaq::noise_model_type::x_error |     (C++                          |
|     (C++                          |     function                      |
|     enumerator)](ap               | )](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4N5cudaq15scalar_operatordvENSt |
| cudaq16noise_model_type7x_errorE) | 7complexIdEERK15scalar_operator), |
| -                                 |     [\[1\                         |
| [cudaq::noise_model_type::y_error | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|     enumerator)](ap               | 7complexIdEERR15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[2\]](api/languages/cp      |
| cudaq16noise_model_type7y_errorE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatordvEdRK15scalar_operator), |
| [cudaq::noise_model_type::z_error |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](ap               | operatordvEdRR15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[4\]](api/languages         |
| cudaq16noise_model_type7z_errorE) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -   [cudaq::num_available_gpus    | alar_operatordvENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     function                      | _api.html#_CPPv4NKR5cudaq15scalar |
| )](api/languages/cpp_api.html#_CP | _operatordvERK15scalar_operator), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[6\]]                       |
| -   [cudaq::observe (C++          | (api/languages/cpp_api.html#_CPPv |
|     function)]                    | 4NKR5cudaq15scalar_operatordvEd), |
| (api/languages/cpp_api.html#_CPPv |     [\[7\]](api/language          |
| 4I00DpEN5cudaq7observeENSt6vector | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| I14observe_resultEERR13QuantumKer | alar_operatordvENSt7complexIdEE), |
| nelRK15SpinOpContainerDpRR4Args), |     [\[8\]](api/languages/cp      |
|     [\[1\]](api/languages/cpp_ap  | p_api.html#_CPPv4NO5cudaq15scalar |
| i.html#_CPPv4I0DpEN5cudaq7observe | _operatordvERK15scalar_operator), |
| E14observe_resultNSt6size_tERR13Q |     [\[9\                         |
| uantumKernelRK7spin_opDpRR4Args), | ]](api/languages/cpp_api.html#_CP |
|     [\[                           | Pv4NO5cudaq15scalar_operatordvEd) |
| 2\]](api/languages/cpp_api.html#_ | -   [c                            |
| CPPv4I0DpEN5cudaq7observeE14obser | udaq::scalar_operator::operator/= |
| ve_resultRK15observe_optionsRR13Q |     (C++                          |
| uantumKernelRK7spin_opDpRR4Args), |     function)](api/languag        |
|     [\[3\]](api/lang              | es/cpp_api.html#_CPPv4N5cudaq15sc |
| uages/cpp_api.html#_CPPv4I0DpEN5c | alar_operatordVENSt7complexIdEE), |
| udaq7observeE14observe_resultRR13 |     [\[1\]](api/languages/c       |
| QuantumKernelRK7spin_opDpRR4Args) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::observe_options (C++  | _operatordVERK15scalar_operator), |
|     st                            |     [\[2                          |
| ruct)](api/languages/cpp_api.html | \]](api/languages/cpp_api.html#_C |
| #_CPPv4N5cudaq15observe_optionsE) | PPv4N5cudaq15scalar_operatordVEd) |
| -   [cudaq::observe_result (C++   | -   [                             |
|                                   | cudaq::scalar_operator::operator= |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14observe_resultE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4N5cudaq15scalar |
|    [cudaq::observe_result::counts | _operatoraSERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/languages/        |
|     function)](api/languages/c    | cpp_api.html#_CPPv4N5cudaq15scala |
| pp_api.html#_CPPv4N5cudaq14observ | r_operatoraSERR15scalar_operator) |
| e_result6countsERK12spin_op_term) | -   [c                            |
| -   [cudaq::observe_result::dump  | udaq::scalar_operator::operator== |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api/languages/c    |
| ](api/languages/cpp_api.html#_CPP | pp_api.html#_CPPv4NK5cudaq15scala |
| v4N5cudaq14observe_result4dumpEv) | r_operatoreqERK15scalar_operator) |
| -   [c                            | -   [cudaq:                       |
| udaq::observe_result::expectation | :scalar_operator::scalar_operator |
|     (C++                          |     (C++                          |
|                                   |     func                          |
| function)](api/languages/cpp_api. | tion)](api/languages/cpp_api.html |
| html#_CPPv4N5cudaq14observe_resul | #_CPPv4N5cudaq15scalar_operator15 |
| t11expectationERK12spin_op_term), | scalar_operatorENSt7complexIdEE), |
|     [\[1\]](api/la                |     [\[1\]](api/langu             |
| nguages/cpp_api.html#_CPPv4N5cuda | ages/cpp_api.html#_CPPv4N5cudaq15 |
| q14observe_result11expectationEv) | scalar_operator15scalar_operatorE |
| -   [cuda                         | RK15scalar_callbackRRNSt13unorder |
| q::observe_result::id_coefficient | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[2\                         |
|     function)](api/langu          | ]](api/languages/cpp_api.html#_CP |
| ages/cpp_api.html#_CPPv4N5cudaq14 | Pv4N5cudaq15scalar_operator15scal |
| observe_result14id_coefficientEv) | ar_operatorERK15scalar_operator), |
| -   [cuda                         |     [\[3\]](api/langu             |
| q::observe_result::observe_result | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     (C++                          | scalar_operator15scalar_operatorE |
|                                   | RR15scalar_callbackRRNSt13unorder |
|   function)](api/languages/cpp_ap | ed_mapINSt6stringENSt6stringEEE), |
| i.html#_CPPv4N5cudaq14observe_res |     [\[4\                         |
| ult14observe_resultEdRK7spin_op), | ]](api/languages/cpp_api.html#_CP |
|     [\[1\]](a                     | Pv4N5cudaq15scalar_operator15scal |
| pi/languages/cpp_api.html#_CPPv4N | ar_operatorERR15scalar_operator), |
| 5cudaq14observe_result14observe_r |     [\[5\]](api/language          |
| esultEdRK7spin_op13sample_result) | s/cpp_api.html#_CPPv4N5cudaq15sca |
| -                                 | lar_operator15scalar_operatorEd), |
|  [cudaq::observe_result::operator |     [\[6\]](api/languag           |
|     double (C++                   | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     functio                       | alar_operator15scalar_operatorEv) |
| n)](api/languages/cpp_api.html#_C | -   [                             |
| PPv4N5cudaq14observe_resultcvdEv) | cudaq::scalar_operator::to_matrix |
| -                                 |     (C++                          |
|  [cudaq::observe_result::raw_data |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)](ap                 | i.html#_CPPv4NK5cudaq15scalar_ope |
| i/languages/cpp_api.html#_CPPv4N5 | rator9to_matrixERKNSt13unordered_ |
| cudaq14observe_result8raw_dataEv) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::operator_handler (C++ | -   [                             |
|     cl                            | cudaq::scalar_operator::to_string |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16operator_handlerE) |     function)](api/l              |
| -   [cudaq::optimizable_function  | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq15scalar_operator9to_stringEv) |
|     class)                        | -   [cudaq::s                     |
| ](api/languages/cpp_api.html#_CPP | calar_operator::\~scalar_operator |
| v4N5cudaq20optimizable_functionE) |     (C++                          |
| -   [cudaq::optimization_result   |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     type                          | PPv4N5cudaq15scalar_operatorD0Ev) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::set_noise (C++        |
| Pv4N5cudaq19optimization_resultE) |     function)](api/langu          |
| -   [cudaq::optimizer (C++        | ages/cpp_api.html#_CPPv4N5cudaq9s |
|     class)](api/languages/cpp_a   | et_noiseERKN5cudaq11noise_modelE) |
| pi.html#_CPPv4N5cudaq9optimizerE) | -   [cudaq::set_random_seed (C++  |
| -   [cudaq::optimizer::optimize   |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq15set_random_seedENSt6size_tE) |
|  function)](api/languages/cpp_api | -   [cudaq::simulation_precision  |
| .html#_CPPv4N5cudaq9optimizer8opt |     (C++                          |
| imizeEKiRR20optimizable_function) |     enum)                         |
| -   [cu                           | ](api/languages/cpp_api.html#_CPP |
| daq::optimizer::requiresGradients | v4N5cudaq20simulation_precisionE) |
|     (C++                          | -   [                             |
|     function)](api/la             | cudaq::simulation_precision::fp32 |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9optimizer17requiresGradientsEv) |     enumerator)](api              |
| -   [cudaq::orca (C++             | /languages/cpp_api.html#_CPPv4N5c |
|     type)](api/languages/         | udaq20simulation_precision4fp32E) |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | -   [                             |
| -   [cudaq::orca::sample (C++     | cudaq::simulation_precision::fp64 |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     enumerator)](api              |
| mpleERNSt6vectorINSt6size_tEEERNS | /languages/cpp_api.html#_CPPv4N5c |
| t6vectorINSt6size_tEEERNSt6vector | udaq20simulation_precision4fp64E) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [cudaq::SimulationState (C++  |
|     [\[1\]]                       |     c                             |
| (api/languages/cpp_api.html#_CPPv | lass)](api/languages/cpp_api.html |
| 4N5cudaq4orca6sampleERNSt6vectorI | #_CPPv4N5cudaq15SimulationStateE) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [                             |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | cudaq::SimulationState::precision |
| -   [cudaq::orca::sample_async    |     (C++                          |
|     (C++                          |     enum)](api                    |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
| function)](api/languages/cpp_api. | udaq15SimulationState9precisionE) |
| html#_CPPv4N5cudaq4orca12sample_a | -   [cudaq:                       |
| syncERNSt6vectorINSt6size_tEEERNS | :SimulationState::precision::fp32 |
| t6vectorINSt6size_tEEERNSt6vector |     (C++                          |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     enumerator)](api/lang         |
|     [\[1\]](api/la                | uages/cpp_api.html#_CPPv4N5cudaq1 |
| nguages/cpp_api.html#_CPPv4N5cuda | 5SimulationState9precision4fp32E) |
| q4orca12sample_asyncERNSt6vectorI | -   [cudaq:                       |
| NSt6size_tEEERNSt6vectorINSt6size | :SimulationState::precision::fp64 |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     (C++                          |
| -   [cudaq::OrcaRemoteRESTQPU     |     enumerator)](api/lang         |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     cla                           | 5SimulationState9precision4fp64E) |
| ss)](api/languages/cpp_api.html#_ | -                                 |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |   [cudaq::SimulationState::Tensor |
| -   [cudaq::other_policies (C++   |     (C++                          |
|     s                             |     struct)](                     |
| truct)](api/languages/cpp_api.htm | api/languages/cpp_api.html#_CPPv4 |
| l#_CPPv4N5cudaq14other_policiesE) | N5cudaq15SimulationState6TensorE) |
| -   [cudaq::PasqalRemoteRESTQPU   | -   [cudaq::spin_handler (C++     |
|     (C++                          |                                   |
|     class                         |   class)](api/languages/cpp_api.h |
| )](api/languages/cpp_api.html#_CP | tml#_CPPv4N5cudaq12spin_handlerE) |
| Pv4N5cudaq19PasqalRemoteRESTQPUE) | -   [cudaq:                       |
| -   [cudaq::pauli1 (C++           | :spin_handler::to_diagonal_matrix |
|     class)](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq6pauli1E) |     function)](api/la             |
| -                                 | nguages/cpp_api.html#_CPPv4NK5cud |
|    [cudaq::pauli1::num_parameters | aq12spin_handler18to_diagonal_mat |
|     (C++                          | rixERNSt13unordered_mapINSt6size_ |
|     member)]                      | tENSt7int64_tEEERKNSt13unordered_ |
| (api/languages/cpp_api.html#_CPPv | mapINSt6stringENSt7complexIdEEEE) |
| 4N5cudaq6pauli114num_parametersE) | -                                 |
| -   [cudaq::pauli1::num_targets   |   [cudaq::spin_handler::to_matrix |
|     (C++                          |     (C++                          |
|     membe                         |     function                      |
| r)](api/languages/cpp_api.html#_C | )](api/languages/cpp_api.html#_CP |
| PPv4N5cudaq6pauli111num_targetsE) | Pv4N5cudaq12spin_handler9to_matri |
| -   [cudaq::pauli1::pauli1 (C++   | xERKNSt6stringENSt7complexIdEEb), |
|     function)](api/languages/cpp_ |     [\[1                          |
| api.html#_CPPv4N5cudaq6pauli16pau | \]](api/languages/cpp_api.html#_C |
| li1ERKNSt6vectorIN5cudaq4realEEE) | PPv4NK5cudaq12spin_handler9to_mat |
| -   [cudaq::pauli2 (C++           | rixERNSt13unordered_mapINSt6size_ |
|     class)](api/languages/cp      | tENSt7int64_tEEERKNSt13unordered_ |
| p_api.html#_CPPv4N5cudaq6pauli2E) | mapINSt6stringENSt7complexIdEEEE) |
| -                                 | -   [cuda                         |
|    [cudaq::pauli2::num_parameters | q::spin_handler::to_sparse_matrix |
|     (C++                          |     (C++                          |
|     member)]                      |     function)](api/               |
| (api/languages/cpp_api.html#_CPPv | languages/cpp_api.html#_CPPv4N5cu |
| 4N5cudaq6pauli214num_parametersE) | daq12spin_handler16to_sparse_matr |
| -   [cudaq::pauli2::num_targets   | ixERKNSt6stringENSt7complexIdEEb) |
|     (C++                          | -                                 |
|     membe                         |   [cudaq::spin_handler::to_string |
| r)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq6pauli211num_targetsE) |     function)](ap                 |
| -   [cudaq::pauli2::pauli2 (C++   | i/languages/cpp_api.html#_CPPv4NK |
|     function)](api/languages/cpp_ | 5cudaq12spin_handler9to_stringEb) |
| api.html#_CPPv4N5cudaq6pauli26pau | -                                 |
| li2ERKNSt6vectorIN5cudaq4realEEE) |   [cudaq::spin_handler::unique_id |
| -   [cudaq::phase_damping (C++    |     (C++                          |
|                                   |     function)](ap                 |
|  class)](api/languages/cpp_api.ht | i/languages/cpp_api.html#_CPPv4NK |
| ml#_CPPv4N5cudaq13phase_dampingE) | 5cudaq12spin_handler9unique_idEv) |
| -   [cud                          | -   [cudaq::spin_op (C++          |
| aq::phase_damping::num_parameters |     type)](api/languages/cpp      |
|     (C++                          | _api.html#_CPPv4N5cudaq7spin_opE) |
|     member)](api/lan              | -   [cudaq::spin_op_term (C++     |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 13phase_damping14num_parametersE) |    type)](api/languages/cpp_api.h |
| -   [                             | tml#_CPPv4N5cudaq12spin_op_termE) |
| cudaq::phase_damping::num_targets | -   [cudaq::state (C++            |
|     (C++                          |     class)](api/languages/c       |
|     member)](api/                 | pp_api.html#_CPPv4N5cudaq5stateE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::state::amplitude (C++ |
| daq13phase_damping11num_targetsE) |     function)](api/lang           |
| -   [cudaq::phase_flip_channel    | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     (C++                          | state9amplitudeERKNSt6vectorIiEE) |
|     clas                          | -   [cudaq::state::amplitudes     |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq18phase_flip_channelE) |     f                             |
| -   [cudaq::p                     | unction)](api/languages/cpp_api.h |
| hase_flip_channel::num_parameters | tml#_CPPv4N5cudaq5state10amplitud |
|     (C++                          | esERKNSt6vectorINSt6vectorIiEEEE) |
|     member)](api/language         | -   [cudaq::state::dump (C++      |
| s/cpp_api.html#_CPPv4N5cudaq18pha |     function)](ap                 |
| se_flip_channel14num_parametersE) | i/languages/cpp_api.html#_CPPv4NK |
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
| -   [define() (cudaq.operators    | -   [description (cudaq.Target    |
|     method)](api/languages/python |                                   |
| _api.html#cudaq.operators.define) | property)](api/languages/python_a |
|     -   [(cuda                    | pi.html#cudaq.Target.description) |
| q.operators.MatrixOperatorElement | -   [deserialize                  |
|         class                     |     (cudaq.SampleResult           |
|         method)](api/langu        |     attribu                       |
| ages/python_api.html#cudaq.operat | te)](api/languages/python_api.htm |
| ors.MatrixOperatorElement.define) | l#cudaq.SampleResult.deserialize) |
|     -   [(in module               | -   [detector() (in module        |
|         cudaq.operators.cus       |     cudaq)](api/language          |
| tom)](api/languages/python_api.ht | s/python_api.html#cudaq.detector) |
| ml#cudaq.operators.custom.define) | -   [detectors() (in module       |
| -   [degrees                      |     cudaq)](api/languages         |
|     (cu                           | /python_api.html#cudaq.detectors) |
| daq.operators.boson.BosonOperator | -   [distribute_terms             |
|     property)](api/lang           |     (cu                           |
| uages/python_api.html#cudaq.opera | daq.operators.boson.BosonOperator |
| tors.boson.BosonOperator.degrees) |     attribute)](api/languages/pyt |
|     -   [(cudaq.ope               | hon_api.html#cudaq.operators.boso |
| rators.boson.BosonOperatorElement | n.BosonOperator.distribute_terms) |
|                                   |     -   [(cudaq.                  |
|        property)](api/languages/p | operators.fermion.FermionOperator |
| ython_api.html#cudaq.operators.bo |                                   |
| son.BosonOperatorElement.degrees) | attribute)](api/languages/python_ |
|     -   [(cudaq.                  | api.html#cudaq.operators.fermion. |
| operators.boson.BosonOperatorTerm | FermionOperator.distribute_terms) |
|         property)](api/language   |     -                             |
| s/python_api.html#cudaq.operators |  [(cudaq.operators.MatrixOperator |
| .boson.BosonOperatorTerm.degrees) |         attribute)](api/language  |
|     -   [(cudaq.                  | s/python_api.html#cudaq.operators |
| operators.fermion.FermionOperator | .MatrixOperator.distribute_terms) |
|         property)](api/language   |     -   [(                        |
| s/python_api.html#cudaq.operators | cudaq.operators.spin.SpinOperator |
| .fermion.FermionOperator.degrees) |                                   |
|     -   [(cudaq.operato           |       attribute)](api/languages/p |
| rs.fermion.FermionOperatorElement | ython_api.html#cudaq.operators.sp |
|                                   | in.SpinOperator.distribute_terms) |
|    property)](api/languages/pytho | -   [draw() (in module            |
| n_api.html#cudaq.operators.fermio |     cudaq)](api/lang              |
| n.FermionOperatorElement.degrees) | uages/python_api.html#cudaq.draw) |
|     -   [(cudaq.oper              | -   [dump (cudaq.ComplexMatrix    |
| ators.fermion.FermionOperatorTerm |     a                             |
|                                   | ttribute)](api/languages/python_a |
|       property)](api/languages/py | pi.html#cudaq.ComplexMatrix.dump) |
| thon_api.html#cudaq.operators.fer |     -   [(cudaq.ObserveResult     |
| mion.FermionOperatorTerm.degrees) |         a                         |
|     -                             | ttribute)](api/languages/python_a |
|  [(cudaq.operators.MatrixOperator | pi.html#cudaq.ObserveResult.dump) |
|         property)](api            |     -   [(cu                      |
| /languages/python_api.html#cudaq. | daq.operators.boson.BosonOperator |
| operators.MatrixOperator.degrees) |         attribute)](api/l         |
|     -   [(cuda                    | anguages/python_api.html#cudaq.op |
| q.operators.MatrixOperatorElement | erators.boson.BosonOperator.dump) |
|         property)](api/langua     |     -   [(cudaq.                  |
| ges/python_api.html#cudaq.operato | operators.boson.BosonOperatorTerm |
| rs.MatrixOperatorElement.degrees) |         attribute)](api/langu     |
|     -   [(c                       | ages/python_api.html#cudaq.operat |
| udaq.operators.MatrixOperatorTerm | ors.boson.BosonOperatorTerm.dump) |
|         property)](api/lan        |     -   [(cudaq.                  |
| guages/python_api.html#cudaq.oper | operators.fermion.FermionOperator |
| ators.MatrixOperatorTerm.degrees) |         attribute)](api/langu     |
|     -   [(                        | ages/python_api.html#cudaq.operat |
| cudaq.operators.spin.SpinOperator | ors.fermion.FermionOperator.dump) |
|         property)](api/la         |     -   [(cudaq.oper              |
| nguages/python_api.html#cudaq.ope | ators.fermion.FermionOperatorTerm |
| rators.spin.SpinOperator.degrees) |         attribute)](api/languages |
|     -   [(cudaq.o                 | /python_api.html#cudaq.operators. |
| perators.spin.SpinOperatorElement | fermion.FermionOperatorTerm.dump) |
|         property)](api/languages  |     -                             |
| /python_api.html#cudaq.operators. |  [(cudaq.operators.MatrixOperator |
| spin.SpinOperatorElement.degrees) |         attribute)](              |
|     -   [(cuda                    | api/languages/python_api.html#cud |
| q.operators.spin.SpinOperatorTerm | aq.operators.MatrixOperator.dump) |
|         property)](api/langua     |     -   [(c                       |
| ges/python_api.html#cudaq.operato | udaq.operators.MatrixOperatorTerm |
| rs.spin.SpinOperatorTerm.degrees) |         attribute)](api/          |
| -   [dem_from_kernel() (in module | languages/python_api.html#cudaq.o |
|     cudaq)](api/languages/pytho   | perators.MatrixOperatorTerm.dump) |
| n_api.html#cudaq.dem_from_kernel) |     -   [(                        |
| -   [Depolarization1 (class in    | cudaq.operators.spin.SpinOperator |
|     cudaq)](api/languages/pytho   |         attribute)](api           |
| n_api.html#cudaq.Depolarization1) | /languages/python_api.html#cudaq. |
| -   [Depolarization2 (class in    | operators.spin.SpinOperator.dump) |
|     cudaq)](api/languages/pytho   |     -   [(cuda                    |
| n_api.html#cudaq.Depolarization2) | q.operators.spin.SpinOperatorTerm |
| -   [DepolarizationChannel (class |         attribute)](api/lan       |
|     in                            | guages/python_api.html#cudaq.oper |
|                                   | ators.spin.SpinOperatorTerm.dump) |
| cudaq)](api/languages/python_api. |     -   [(cudaq.Resources         |
| html#cudaq.DepolarizationChannel) |                                   |
| -   [depth (cudaq.Resources       |    attribute)](api/languages/pyth |
|                                   | on_api.html#cudaq.Resources.dump) |
|    property)](api/languages/pytho |     -   [(cudaq.SampleResult      |
| n_api.html#cudaq.Resources.depth) |                                   |
| -   [depth_for_arity              | attribute)](api/languages/python_ |
|     (cudaq.Resources              | api.html#cudaq.SampleResult.dump) |
|     attribut                      |     -   [(cudaq.State             |
| e)](api/languages/python_api.html |                                   |
| #cudaq.Resources.depth_for_arity) |        attribute)](api/languages/ |
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
| uages/python_api.html#cudaq.Evolv | -   [from_word                    |
| eResult.final_expectation_values) |     (                             |
| -   [final_state                  | cudaq.operators.spin.SpinOperator |
|     (cudaq.EvolveResult           |     attribute)](api/lang          |
|     attribu                       | uages/python_api.html#cudaq.opera |
| te)](api/languages/python_api.htm | tors.spin.SpinOperator.from_word) |
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
| udaq.operators.MatrixOperatorTerm |     -   [(cudaq.o                 |
|         property)](api/lan        | perators.spin.SpinOperatorElement |
| guages/python_api.html#cudaq.oper |                                   |
| ators.MatrixOperatorTerm.term_id) |       attribute)](api/languages/p |
|     -   [(cuda                    | ython_api.html#cudaq.operators.sp |
| q.operators.spin.SpinOperatorTerm | in.SpinOperatorElement.to_string) |
|         property)](api/langua     | -   [TraceInstruction (class in   |
| ges/python_api.html#cudaq.operato |     cudaq.p                       |
| rs.spin.SpinOperatorTerm.term_id) | tsbe)](api/languages/python_api.h |
| -   [to_bools() (in module        | tml#cudaq.ptsbe.TraceInstruction) |
|     cudaq)](api/language          | -   [TraceInstructionType (class  |
| s/python_api.html#cudaq.to_bools) |     in                            |
| -   [to_dict (cudaq.Resources     |     cudaq.ptsbe                   |
|                                   | )](api/languages/python_api.html# |
| attribute)](api/languages/python_ | cudaq.ptsbe.TraceInstructionType) |
| api.html#cudaq.Resources.to_dict) | -   [trajectories                 |
| -   [to_json                      |                                   |
|     (                             |   (cudaq.ptsbe.PTSBEExecutionData |
| cudaq.operators.spin.SpinOperator |     property)](api/lang           |
|     attribute)](api/la            | uages/python_api.html#cudaq.ptsbe |
| nguages/python_api.html#cudaq.ope | .PTSBEExecutionData.trajectories) |
| rators.spin.SpinOperator.to_json) | -   [trajectory_id                |
|     -   [(cuda                    |     (cudaq.ptsbe.KrausTrajectory  |
| q.operators.spin.SpinOperatorTerm |     property)](api/la             |
|         attribute)](api/langua    | nguages/python_api.html#cudaq.pts |
| ges/python_api.html#cudaq.operato | be.KrausTrajectory.trajectory_id) |
| rs.spin.SpinOperatorTerm.to_json) | -   [translate() (in module       |
| -   [to_json()                    |     cudaq)](api/languages         |
|     (cudaq.PyKernelDecorator      | /python_api.html#cudaq.translate) |
|     metho                         | -   [trim                         |
| d)](api/languages/python_api.html |     (cu                           |
| #cudaq.PyKernelDecorator.to_json) | daq.operators.boson.BosonOperator |
| -   [to_matrix                    |     attribute)](api/l             |
|     (cu                           | anguages/python_api.html#cudaq.op |
| daq.operators.boson.BosonOperator | erators.boson.BosonOperator.trim) |
|     attribute)](api/langua        |     -   [(cudaq.                  |
| ges/python_api.html#cudaq.operato | operators.fermion.FermionOperator |
| rs.boson.BosonOperator.to_matrix) |         attribute)](api/langu     |
|     -   [(cudaq.ope               | ages/python_api.html#cudaq.operat |
| rators.boson.BosonOperatorElement | ors.fermion.FermionOperator.trim) |
|                                   |     -                             |
|     attribute)](api/languages/pyt |  [(cudaq.operators.MatrixOperator |
| hon_api.html#cudaq.operators.boso |         attribute)](              |
| n.BosonOperatorElement.to_matrix) | api/languages/python_api.html#cud |
|     -   [(cudaq.                  | aq.operators.MatrixOperator.trim) |
| operators.boson.BosonOperatorTerm |     -   [(                        |
|                                   | cudaq.operators.spin.SpinOperator |
|        attribute)](api/languages/ |         attribute)](api           |
| python_api.html#cudaq.operators.b | /languages/python_api.html#cudaq. |
| oson.BosonOperatorTerm.to_matrix) | operators.spin.SpinOperator.trim) |
|     -   [(cudaq.                  | -   [type                         |
| operators.fermion.FermionOperator |     (c                            |
|                                   | udaq.ptsbe.ShotAllocationStrategy |
|        attribute)](api/languages/ |     property)](api/               |
| python_api.html#cudaq.operators.f | languages/python_api.html#cudaq.p |
| ermion.FermionOperator.to_matrix) | tsbe.ShotAllocationStrategy.type) |
|     -   [(cudaq.operato           |     -                             |
| rs.fermion.FermionOperatorElement |    [(cudaq.ptsbe.TraceInstruction |
|                                   |         property)                 |
| attribute)](api/languages/python_ | ](api/languages/python_api.html#c |
| api.html#cudaq.operators.fermion. | udaq.ptsbe.TraceInstruction.type) |
| FermionOperatorElement.to_matrix) | -   [type_to_str()                |
|     -   [(cudaq.oper              |     (cudaq.PyKernelDecorator      |
| ators.fermion.FermionOperatorTerm |     static                        |
|                                   |     method)](                     |
|    attribute)](api/languages/pyth | api/languages/python_api.html#cud |
| on_api.html#cudaq.operators.fermi | aq.PyKernelDecorator.type_to_str) |
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
