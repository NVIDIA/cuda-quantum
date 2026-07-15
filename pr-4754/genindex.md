::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4754
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
    -   [Implement a Hardware
        Backend](using/extending/backend.html){.reference .internal}
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
    -   [Package & Distribute a Backend
        Plugin](using/extending/packaging.html){.reference .internal}
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
| -   [cachedCompiledModule()       | -   [cudaq                        |
|     (cudaq.PyKernelDecorator      | ::phase_flip_channel::num_targets |
|     method)](api/langu            |     (C++                          |
| ages/python_api.html#cudaq.PyKern |     member)](api/langu            |
| elDecorator.cachedCompiledModule) | ages/cpp_api.html#_CPPv4N5cudaq18 |
| -   [canonicalize                 | phase_flip_channel11num_targetsE) |
|     (cu                           | -   [cudaq::product_op (C++       |
| daq.operators.boson.BosonOperator |                                   |
|     attribute)](api/languages     |  class)](api/languages/cpp_api.ht |
| /python_api.html#cudaq.operators. | ml#_CPPv4I0EN5cudaq10product_opE) |
| boson.BosonOperator.canonicalize) | -   [cudaq::product_op::begin     |
|     -   [(cudaq.                  |     (C++                          |
| operators.boson.BosonOperatorTerm |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
|     attribute)](api/languages/pyt | PPv4NK5cudaq10product_op5beginEv) |
| hon_api.html#cudaq.operators.boso | -                                 |
| n.BosonOperatorTerm.canonicalize) |  [cudaq::product_op::canonicalize |
|     -   [(cudaq.                  |     (C++                          |
| operators.fermion.FermionOperator |     func                          |
|                                   | tion)](api/languages/cpp_api.html |
|     attribute)](api/languages/pyt | #_CPPv4N5cudaq10product_op12canon |
| hon_api.html#cudaq.operators.ferm | icalizeERKNSt3setINSt6size_tEEE), |
| ion.FermionOperator.canonicalize) |     [\[1\]](api                   |
|     -   [(cudaq.oper              | /languages/cpp_api.html#_CPPv4N5c |
| ators.fermion.FermionOperatorTerm | udaq10product_op12canonicalizeEv) |
|                                   | -   [                             |
| attribute)](api/languages/python_ | cudaq::product_op::const_iterator |
| api.html#cudaq.operators.fermion. |     (C++                          |
| FermionOperatorTerm.canonicalize) |     struct)](api/                 |
|     -                             | languages/cpp_api.html#_CPPv4N5cu |
|  [(cudaq.operators.MatrixOperator | daq10product_op14const_iteratorE) |
|         attribute)](api/lang      | -   [cudaq::product_o             |
| uages/python_api.html#cudaq.opera | p::const_iterator::const_iterator |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     fu                            |
| udaq.operators.MatrixOperatorTerm | nction)](api/languages/cpp_api.ht |
|         attribute)](api/language  | ml#_CPPv4N5cudaq10product_op14con |
| s/python_api.html#cudaq.operators | st_iterator14const_iteratorEPK10p |
| .MatrixOperatorTerm.canonicalize) | roduct_opI9HandlerTyENSt6size_tE) |
|     -   [(                        | -   [cudaq::produ                 |
| cudaq.operators.spin.SpinOperator | ct_op::const_iterator::operator!= |
|         attribute)](api/languag   |     (C++                          |
| es/python_api.html#cudaq.operator |     fun                           |
| s.spin.SpinOperator.canonicalize) | ction)](api/languages/cpp_api.htm |
|     -   [(cuda                    | l#_CPPv4NK5cudaq10product_op14con |
| q.operators.spin.SpinOperatorTerm | st_iteratorneERK14const_iterator) |
|                                   | -   [cudaq::produ                 |
|       attribute)](api/languages/p | ct_op::const_iterator::operator\* |
| ython_api.html#cudaq.operators.sp |     (C++                          |
| in.SpinOperatorTerm.canonicalize) |     function)](api/lang           |
| -   [captured_variables()         | uages/cpp_api.html#_CPPv4NK5cudaq |
|     (cudaq.PyKernelDecorator      | 10product_op14const_iteratormlEv) |
|     method)](api/lan              | -   [cudaq::produ                 |
| guages/python_api.html#cudaq.PyKe | ct_op::const_iterator::operator++ |
| rnelDecorator.captured_variables) |     (C++                          |
| -   [CentralDifference (class in  |     function)](api/lang           |
|     cudaq.gradients)              | uages/cpp_api.html#_CPPv4N5cudaq1 |
| ](api/languages/python_api.html#c | 0product_op14const_iteratorppEi), |
| udaq.gradients.CentralDifference) |     [\[1\]](api/lan               |
| -   [channel                      | guages/cpp_api.html#_CPPv4N5cudaq |
|     (cudaq.ptsbe.TraceInstruction | 10product_op14const_iteratorppEv) |
|     property)](a                  | -   [cudaq::produc                |
| pi/languages/python_api.html#cuda | t_op::const_iterator::operator\-- |
| q.ptsbe.TraceInstruction.channel) |     (C++                          |
| -   [circuit_location             |     function)](api/lang           |
|     (cudaq.ptsbe.KrausSelection   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     property)](api/lang           | 0product_op14const_iteratormmEi), |
| uages/python_api.html#cudaq.ptsbe |     [\[1\]](api/lan               |
| .KrausSelection.circuit_location) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [clear (cudaq.Resources       | 10product_op14const_iteratormmEv) |
|                                   | -   [cudaq::produc                |
|   attribute)](api/languages/pytho | t_op::const_iterator::operator-\> |
| n_api.html#cudaq.Resources.clear) |     (C++                          |
|     -   [(cudaq.SampleResult      |     function)](api/lan            |
|         a                         | guages/cpp_api.html#_CPPv4N5cudaq |
| ttribute)](api/languages/python_a | 10product_op14const_iteratorptEv) |
| pi.html#cudaq.SampleResult.clear) | -   [cudaq::produ                 |
| -   [COBYLA (class in             | ct_op::const_iterator::operator== |
|     cudaq.o                       |     (C++                          |
| ptimizers)](api/languages/python_ |     fun                           |
| api.html#cudaq.optimizers.COBYLA) | ction)](api/languages/cpp_api.htm |
| -   [coefficient                  | l#_CPPv4NK5cudaq10product_op14con |
|     (cudaq.                       | st_iteratoreqERK14const_iterator) |
| operators.boson.BosonOperatorTerm | -   [cudaq::product_op::degrees   |
|     property)](api/languages/py   |     (C++                          |
| thon_api.html#cudaq.operators.bos |     function)                     |
| on.BosonOperatorTerm.coefficient) | ](api/languages/cpp_api.html#_CPP |
|     -   [(cudaq.oper              | v4NK5cudaq10product_op7degreesEv) |
| ators.fermion.FermionOperatorTerm | -   [cudaq::product_op::dump (C++ |
|                                   |     functi                        |
|   property)](api/languages/python | on)](api/languages/cpp_api.html#_ |
| _api.html#cudaq.operators.fermion | CPPv4NK5cudaq10product_op4dumpEv) |
| .FermionOperatorTerm.coefficient) | -   [cudaq::product_op::end (C++  |
|     -   [(c                       |     funct                         |
| udaq.operators.MatrixOperatorTerm | ion)](api/languages/cpp_api.html# |
|         property)](api/languag    | _CPPv4NK5cudaq10product_op3endEv) |
| es/python_api.html#cudaq.operator | -   [c                            |
| s.MatrixOperatorTerm.coefficient) | udaq::product_op::get_coefficient |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     function)](api/lan            |
|         property)](api/languages/ | guages/cpp_api.html#_CPPv4NK5cuda |
| python_api.html#cudaq.operators.s | q10product_op15get_coefficientEv) |
| pin.SpinOperatorTerm.coefficient) | -                                 |
| -   [col_count                    |   [cudaq::product_op::get_term_id |
|     (cudaq.KrausOperator          |     (C++                          |
|     prope                         |     function)](api                |
| rty)](api/languages/python_api.ht | /languages/cpp_api.html#_CPPv4NK5 |
| ml#cudaq.KrausOperator.col_count) | cudaq10product_op11get_term_idEv) |
| -   [compile()                    | -                                 |
|     (cudaq.PyKernelDecorator      |   [cudaq::product_op::is_identity |
|     metho                         |     (C++                          |
| d)](api/languages/python_api.html |     function)](api                |
| #cudaq.PyKernelDecorator.compile) | /languages/cpp_api.html#_CPPv4NK5 |
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
| -   [cudaq::CusvState (C++        | ptsbe::ExhaustiveSamplingStrategy |
|                                   |     (C++                          |
|    class)](api/languages/cpp_api. |     class)](api/langua            |
| html#_CPPv4I0EN5cudaq9CusvStateE) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
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
| -   [cudaq::Executi               | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| onContext::invocationResultBuffer | ple_result18has_execution_dataEv) |
|     (C++                          | -   [cudaq::pt                    |
|     member)](api/languages/cpp_   | sbe::sample_result::sample_result |
| api.html#_CPPv4N5cudaq16Execution |     (C++                          |
| Context22invocationResultBufferE) |     function)](api/l              |
| -   [cu                           | anguages/cpp_api.html#_CPPv4N5cud |
| daq::ExecutionContext::kernelName | aq5ptsbe13sample_result13sample_r |
|     (C++                          | esultERRN5cudaq13sample_resultE), |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda |  [\[1\]](api/languages/cpp_api.ht |
| q16ExecutionContext10kernelNameE) | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| -   [cud                          | sult13sample_resultERRN5cudaq13sa |
| aq::ExecutionContext::kernelTrace | mple_resultE18PTSBEExecutionData) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     member)](api/lan              | sample_result::set_execution_data |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11kernelTraceE) |     function)](api/               |
| -   [cudaq:                       | languages/cpp_api.html#_CPPv4N5cu |
| :ExecutionContext::msm_dimensions | daq5ptsbe13sample_result18set_exe |
|     (C++                          | cution_dataE18PTSBEExecutionData) |
|     member)](api/langua           | -   [cud                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E | aq::ptsbe::ShotAllocationStrategy |
| xecutionContext14msm_dimensionsE) |     (C++                          |
| -   [cudaq::                      |     struct)](using                |
| ExecutionContext::msm_prob_err_id | /examples/ptsbe.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe22ShotAllocationStrategyE) |
|     member)](api/languag          | -   [cudaq::ptsbe::ShotAllocatio  |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | nStrategy::ShotAllocationStrategy |
| ecutionContext15msm_prob_err_idE) |     (C++                          |
| -   [cudaq::Ex                    |     function)                     |
| ecutionContext::msm_probabilities | ](using/examples/ptsbe.html#_CPPv |
|     (C++                          | 4N5cudaq5ptsbe22ShotAllocationStr |
|     member)](api/languages        | ategy22ShotAllocationStrategyE4Ty |
| /cpp_api.html#_CPPv4N5cudaq16Exec | pedNSt8optionalINSt8uint64_tEEE), |
| utionContext17msm_probabilitiesE) |     [\[1\                         |
| -                                 | ]](using/examples/ptsbe.html#_CPP |
|    [cudaq::ExecutionContext::name | v4N5cudaq5ptsbe22ShotAllocationSt |
|     (C++                          | rategy22ShotAllocationStrategyEv) |
|     member)]                      | -   [cudaq::pt                    |
| (api/languages/cpp_api.html#_CPPv | sbe::ShotAllocationStrategy::Type |
| 4N5cudaq16ExecutionContext4nameE) |     (C++                          |
| -   [cu                           |     enum)](using/exam             |
| daq::ExecutionContext::noiseModel | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|     (C++                          | be22ShotAllocationStrategy4TypeE) |
|     member)](api/la               | -   [cudaq::ptsbe::ShotAllocatio  |
| nguages/cpp_api.html#_CPPv4N5cuda | nStrategy::Type::HIGH_WEIGHT_BIAS |
| q16ExecutionContext10noiseModelE) |     (C++                          |
| -   [cudaq::Exe                   |     enumerat                      |
| cutionContext::numberTrajectories | or)](using/examples/ptsbe.html#_C |
|     (C++                          | PPv4N5cudaq5ptsbe22ShotAllocation |
|     member)](api/languages/       | Strategy4Type16HIGH_WEIGHT_BIASE) |
| cpp_api.html#_CPPv4N5cudaq16Execu | -   [cudaq::ptsbe::ShotAllocati   |
| tionContext18numberTrajectoriesE) | onStrategy::Type::LOW_WEIGHT_BIAS |
| -   [c                            |     (C++                          |
| udaq::ExecutionContext::optResult |     enumera                       |
|     (C++                          | tor)](using/examples/ptsbe.html#_ |
|     member)](api/                 | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| languages/cpp_api.html#_CPPv4N5cu | nStrategy4Type15LOW_WEIGHT_BIASE) |
| daq16ExecutionContext9optResultE) | -   [cudaq::ptsbe::ShotAlloc      |
| -                                 | ationStrategy::Type::PROPORTIONAL |
|   [cudaq::ExecutionContext::qpuId |     (C++                          |
|     (C++                          |     enum                          |
|     member)](                     | erator)](using/examples/ptsbe.htm |
| api/languages/cpp_api.html#_CPPv4 | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| N5cudaq16ExecutionContext5qpuIdE) | tionStrategy4Type12PROPORTIONALE) |
| -   [cudaq                        | -   [cudaq::ptsbe::Shot           |
| ::ExecutionContext::registerNames | AllocationStrategy::Type::UNIFORM |
|     (C++                          |     (C++                          |
|     member)](api/langu            |                                   |
| ages/cpp_api.html#_CPPv4N5cudaq16 |   enumerator)](using/examples/pts |
| ExecutionContext13registerNamesE) | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| -   [cu                           | AllocationStrategy4Type7UNIFORME) |
| daq::ExecutionContext::reorderIdx | -                                 |
|     (C++                          |   [cudaq::ptsbe::TraceInstruction |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     struct)](                     |
| q16ExecutionContext10reorderIdxE) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | N5cudaq5ptsbe16TraceInstructionE) |
|  [cudaq::ExecutionContext::result | -   [cudaq:                       |
|     (C++                          | :ptsbe::TraceInstruction::channel |
|     member)](a                    |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     member)](api/lang             |
| 5cudaq16ExecutionContext6resultE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -                                 | ptsbe16TraceInstruction7channelE) |
|   [cudaq::ExecutionContext::shots | -   [cudaq::                      |
|     (C++                          | ptsbe::TraceInstruction::controls |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     member)](api/langu            |
| N5cudaq16ExecutionContext5shotsE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [cudaq::                      | tsbe16TraceInstruction8controlsE) |
| ExecutionContext::simulationState | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::name |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     member)](api/l                |
| ecutionContext15simulationStateE) | anguages/cpp_api.html#_CPPv4N5cud |
| -                                 | aq5ptsbe16TraceInstruction4nameE) |
|    [cudaq::ExecutionContext::spin | -   [cudaq                        |
|     (C++                          | ::ptsbe::TraceInstruction::params |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     member)](api/lan              |
| 4N5cudaq16ExecutionContext4spinE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [cudaq::                      | 5ptsbe16TraceInstruction6paramsE) |
| ExecutionContext::totalIterations | -   [cudaq:                       |
|     (C++                          | :ptsbe::TraceInstruction::targets |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     member)](api/lang             |
| ecutionContext15totalIterationsE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -   [cudaq::ExecutionResult (C++  | ptsbe16TraceInstruction7targetsE) |
|     st                            | -   [cudaq::ptsbe::T              |
| ruct)](api/languages/cpp_api.html | raceInstruction::TraceInstruction |
| #_CPPv4N5cudaq15ExecutionResultE) |     (C++                          |
| -   [cud                          |                                   |
| aq::ExecutionResult::appendResult |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4N5cudaq5ptsbe16Trace |
|     functio                       | Instruction16TraceInstructionE20T |
| n)](api/languages/cpp_api.html#_C | raceInstructionTypeNSt6stringENSt |
| PPv4N5cudaq15ExecutionResult12app | 6vectorINSt6size_tEEENSt6vectorIN |
| endResultENSt6stringENSt6size_tE) | St6size_tEEENSt6vectorIdEENSt8opt |
| -   [cu                           | ionalIN5cudaq13kraus_channelEEE), |
| daq::ExecutionResult::deserialize |     [\[1\]](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
|     function)                     | eInstruction16TraceInstructionEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cud                          |
| v4N5cudaq15ExecutionResult11deser | aq::ptsbe::TraceInstruction::type |
| ializeERNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq:                       |     member)](api/l                |
| :ExecutionResult::ExecutionResult | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq5ptsbe16TraceInstruction4typeE) |
|     functio                       | -   [c                            |
| n)](api/languages/cpp_api.html#_C | udaq::ptsbe::TraceInstructionType |
| PPv4N5cudaq15ExecutionResult15Exe |     (C++                          |
| cutionResultE16CountsDictionary), |     enum)](api/                   |
|     [\[1\]](api/lan               | languages/cpp_api.html#_CPPv4N5cu |
| guages/cpp_api.html#_CPPv4N5cudaq | daq5ptsbe20TraceInstructionTypeE) |
| 15ExecutionResult15ExecutionResul | -   [cudaq::                      |
| tE16CountsDictionaryNSt6stringE), | ptsbe::TraceInstructionType::Gate |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     enumerator)](api/langu        |
| Pv4N5cudaq15ExecutionResult15Exec | ages/cpp_api.html#_CPPv4N5cudaq5p |
| utionResultE16CountsDictionaryd), | tsbe20TraceInstructionType4GateE) |
|                                   | -   [cudaq::ptsbe::               |
|    [\[3\]](api/languages/cpp_api. | TraceInstructionType::Measurement |
| html#_CPPv4N5cudaq15ExecutionResu |     (C++                          |
| lt15ExecutionResultENSt6stringE), |                                   |
|     [\[4\                         |    enumerator)](api/languages/cpp |
| ]](api/languages/cpp_api.html#_CP | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| Pv4N5cudaq15ExecutionResult15Exec | aceInstructionType11MeasurementE) |
| utionResultERK15ExecutionResult), | -   [cudaq::p                     |
|     [\[5\]](api/language          | tsbe::TraceInstructionType::Noise |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     (C++                          |
| cutionResult15ExecutionResultEd), |     enumerator)](api/langua       |
|     [\[6\]](api/languag           | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | sbe20TraceInstructionType5NoiseE) |
| ecutionResult15ExecutionResultEv) | -   [                             |
| -   [                             | cudaq::ptsbe::TrajectoryPredicate |
| cudaq::ExecutionResult::operator= |     (C++                          |
|     (C++                          |     type)](api                    |
|     function)](api/languages/     | /languages/cpp_api.html#_CPPv4N5c |
| cpp_api.html#_CPPv4N5cudaq15Execu | udaq5ptsbe19TrajectoryPredicateE) |
| tionResultaSERK15ExecutionResult) | -   [cudaq::QPU (C++              |
| -   [c                            |     class)](api/languages         |
| udaq::ExecutionResult::operator== | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     (C++                          | -   [cudaq::QPU::beginExecution   |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4NK5cudaq15Execu |     function                      |
| tionResulteqERK15ExecutionResult) | )](api/languages/cpp_api.html#_CP |
| -   [cud                          | Pv4N5cudaq3QPU14beginExecutionEv) |
| aq::ExecutionResult::registerName | -   [cuda                         |
|     (C++                          | q::QPU::configureExecutionContext |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     funct                         |
| 15ExecutionResult12registerNameE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq                        | _CPPv4NK5cudaq3QPU25configureExec |
| ::ExecutionResult::sequentialData | utionContextER16ExecutionContext) |
|     (C++                          | -   [cudaq::QPU::endExecution     |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     functi                        |
| ExecutionResult14sequentialDataE) | on)](api/languages/cpp_api.html#_ |
| -   [                             | CPPv4N5cudaq3QPU12endExecutionEv) |
| cudaq::ExecutionResult::serialize | -   [cudaq::QPU::enqueue (C++     |
|     (C++                          |     function)](ap                 |
|     function)](api/l              | i/languages/cpp_api.html#_CPPv4N5 |
| anguages/cpp_api.html#_CPPv4NK5cu | cudaq3QPU7enqueueER11QuantumTask) |
| daq15ExecutionResult9serializeEv) | -   [cud                          |
| -   [cudaq::fermion_handler (C++  | aq::QPU::finalizeExecutionContext |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     func                          |
| #_CPPv4N5cudaq15fermion_handlerE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::fermion_op (C++       | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     type)](api/languages/cpp_api  | utionContextER16ExecutionContext) |
| .html#_CPPv4N5cudaq10fermion_opE) | -   [cudaq::QPU::getCompileTarget |
| -   [cudaq::fermion_op_term (C++  |     (C++                          |
|                                   |     function)](api/languages/c    |
| type)](api/languages/cpp_api.html | pp_api.html#_CPPv4N5cudaq3QPU16ge |
| #_CPPv4N5cudaq15fermion_op_termE) | tCompileTargetERK13sample_policy) |
| -   [cudaq::FermioniqQPU (C++     | -   [cudaq::QPU::getConnectivity  |
|                                   |     (C++                          |
|   class)](api/languages/cpp_api.h |     function)                     |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::get_state (C++        | v4N5cudaq3QPU15getConnectivityEv) |
|                                   | -                                 |
|    function)](api/languages/cpp_a | [cudaq::QPU::getExecutionThreadId |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     (C++                          |
| ateEDaRR13QuantumKernelDpRR4Args) |     function)](api/               |
| -   [cudaq::gradient (C++         | languages/cpp_api.html#_CPPv4NK5c |
|     class)](api/languages/cpp_    | udaq3QPU20getExecutionThreadIdEv) |
| api.html#_CPPv4N5cudaq8gradientE) | -   [cudaq::QPU::getNumQubits     |
| -   [cudaq::gradient::clone (C++  |     (C++                          |
|     fun                           |     functi                        |
| ction)](api/languages/cpp_api.htm | on)](api/languages/cpp_api.html#_ |
| l#_CPPv4N5cudaq8gradient5cloneEv) | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| -   [cudaq::gradient::compute     | -   [                             |
|     (C++                          | cudaq::QPU::getRemoteCapabilities |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     function)](api/l              |
| ient7computeERKNSt6vectorIdEERKNS | anguages/cpp_api.html#_CPPv4NK5cu |
| t8functionIFdNSt6vectorIdEEEEEd), | daq3QPU21getRemoteCapabilitiesEv) |
|     [\[1\]](ap                    | -   [cudaq::QPU::isEmulated (C++  |
| i/languages/cpp_api.html#_CPPv4N5 |     func                          |
| cudaq8gradient7computeERKNSt6vect | tion)](api/languages/cpp_api.html |
| orIdEERNSt6vectorIdEERK7spin_opd) | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| -   [cudaq::gradient::gradient    | -   [cudaq::QPU::isSimulator (C++ |
|     (C++                          |     funct                         |
|     function)](api/lang           | ion)](api/languages/cpp_api.html# |
| uages/cpp_api.html#_CPPv4I00EN5cu | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| daq8gradient8gradientER7KernelT), | -   [cudaq::QPU::onRandomSeedSet  |
|                                   |     (C++                          |
|    [\[1\]](api/languages/cpp_api. |     function)](api/lang           |
| html#_CPPv4I00EN5cudaq8gradient8g | uages/cpp_api.html#_CPPv4N5cudaq3 |
| radientER7KernelTRR10ArgsMapper), | QPU15onRandomSeedSetENSt6size_tE) |
|     [\[2\                         | -   [cudaq::QPU::QPU (C++         |
| ]](api/languages/cpp_api.html#_CP |     functio                       |
| Pv4I00EN5cudaq8gradient8gradientE | n)](api/languages/cpp_api.html#_C |
| RR13QuantumKernelRR10ArgsMapper), | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
|     [\[3                          |                                   |
| \]](api/languages/cpp_api.html#_C |  [\[1\]](api/languages/cpp_api.ht |
| PPv4N5cudaq8gradient8gradientERRN | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| St8functionIFvNSt6vectorIdEEEEE), |     [\[2\]](api/languages/cpp_    |
|     [\[                           | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| 4\]](api/languages/cpp_api.html#_ | -   [cudaq::QPU::setId (C++       |
| CPPv4N5cudaq8gradient8gradientEv) |     function                      |
| -   [cudaq::gradient::setArgs     | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     fu                            | -   [cudaq::QPU::setShots (C++    |
| nction)](api/languages/cpp_api.ht |     f                             |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | unction)](api/languages/cpp_api.h |
| tArgsEvR13QuantumKernelDpRR4Args) | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| -   [cudaq::gradient::setKernel   | -   [cudaq::                      |
|     (C++                          | QPU::supportsExplicitMeasurements |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     function)](api/languag        |
| ient9setKernelEvR13QuantumKernel) | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| -   [cud                          | 28supportsExplicitMeasurementsEv) |
| aq::gradients::central_difference | -   [cudaq::QPU::\~QPU (C++       |
|     (C++                          |     function)](api/languages/cp   |
|     class)](api/la                | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QPUState (C++         |
| q9gradients18central_differenceE) |     class)](api/languages/cpp_    |
| -   [cudaq::gra                   | api.html#_CPPv4N5cudaq8QPUStateE) |
| dients::central_difference::clone | -   [cudaq::qreg (C++             |
|     (C++                          |     class)](api/lan               |
|     function)](api/languages      | guages/cpp_api.html#_CPPv4I_NSt6s |
| /cpp_api.html#_CPPv4N5cudaq9gradi | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| ents18central_difference5cloneEv) | -   [cudaq::qreg::back (C++       |
| -   [cudaq::gradi                 |     function)                     |
| ents::central_difference::compute | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4backENSt6size_tE), |
|     function)](                   |     [\[1\]](api/languages/cpp_ap  |
| api/languages/cpp_api.html#_CPPv4 | i.html#_CPPv4N5cudaq4qreg4backEv) |
| N5cudaq9gradients18central_differ | -   [cudaq::qreg::begin (C++      |
| ence7computeERKNSt6vectorIdEERKNS |                                   |
| t8functionIFdNSt6vectorIdEEEEEd), |  function)](api/languages/cpp_api |
|                                   | .html#_CPPv4N5cudaq4qreg5beginEv) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::qreg::clear (C++      |
| tml#_CPPv4N5cudaq9gradients18cent |                                   |
| ral_difference7computeERKNSt6vect |  function)](api/languages/cpp_api |
| orIdEERNSt6vectorIdEERK7spin_opd) | .html#_CPPv4N5cudaq4qreg5clearEv) |
| -   [cudaq::gradie                | -   [cudaq::qreg::front (C++      |
| nts::central_difference::gradient |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     functio                       | 4N5cudaq4qreg5frontENSt6size_tE), |
| n)](api/languages/cpp_api.html#_C |     [\[1\]](api/languages/cpp_api |
| PPv4I00EN5cudaq9gradients18centra | .html#_CPPv4N5cudaq4qreg5frontEv) |
| l_difference8gradientER7KernelT), | -   [cudaq::qreg::operator\[\]    |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     functi                        |
| q9gradients18central_difference8g | on)](api/languages/cpp_api.html#_ |
| radientER7KernelTRR10ArgsMapper), | CPPv4N5cudaq4qregixEKNSt6size_tE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::qreg::qreg (C++       |
| api.html#_CPPv4I00EN5cudaq9gradie |     function)                     |
| nts18central_difference8gradientE | ](api/languages/cpp_api.html#_CPP |
| RR13QuantumKernelRR10ArgsMapper), | v4N5cudaq4qreg4qregENSt6size_tE), |
|     [\[3\]](api/languages/cpp     |     [\[1\]](api/languages/cpp_ap  |
| _api.html#_CPPv4N5cudaq9gradients | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| 18central_difference8gradientERRN | -   [cudaq::qreg::size (C++       |
| St8functionIFvNSt6vectorIdEEEEE), |                                   |
|     [\[4\]](api/languages/cp      |  function)](api/languages/cpp_api |
| p_api.html#_CPPv4N5cudaq9gradient | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| s18central_difference8gradientEv) | -   [cudaq::qreg::slice (C++      |
| -   [cud                          |     function)](api/langu          |
| aq::gradients::forward_difference | ages/cpp_api.html#_CPPv4N5cudaq4q |
|     (C++                          | reg5sliceENSt6size_tENSt6size_tE) |
|     class)](api/la                | -   [cudaq::qreg::value_type (C++ |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q9gradients18forward_differenceE) | type)](api/languages/cpp_api.html |
| -   [cudaq::gra                   | #_CPPv4N5cudaq4qreg10value_typeE) |
| dients::forward_difference::clone | -   [cudaq::qspan (C++            |
|     (C++                          |     class)](api/lang              |
|     function)](api/languages      | uages/cpp_api.html#_CPPv4I_NSt6si |
| /cpp_api.html#_CPPv4N5cudaq9gradi | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| ents18forward_difference5cloneEv) | -   [cudaq::QuakeValue (C++       |
| -   [cudaq::gradi                 |     class)](api/languages/cpp_api |
| ents::forward_difference::compute | .html#_CPPv4N5cudaq10QuakeValueE) |
|     (C++                          | -   [cudaq::Q                     |
|     function)](                   | uakeValue::canValidateNumElements |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq9gradients18forward_differ |     function)](api/languages      |
| ence7computeERKNSt6vectorIdEERKNS | /cpp_api.html#_CPPv4N5cudaq10Quak |
| t8functionIFdNSt6vectorIdEEEEEd), | eValue22canValidateNumElementsEv) |
|                                   | -                                 |
|   [\[1\]](api/languages/cpp_api.h |  [cudaq::QuakeValue::constantSize |
| tml#_CPPv4N5cudaq9gradients18forw |     (C++                          |
| ard_difference7computeERKNSt6vect |     function)](api                |
| orIdEERNSt6vectorIdEERK7spin_opd) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gradie                | udaq10QuakeValue12constantSizeEv) |
| nts::forward_difference::gradient | -   [cudaq::QuakeValue::dump (C++ |
|     (C++                          |     function)](api/lan            |
|     functio                       | guages/cpp_api.html#_CPPv4N5cudaq |
| n)](api/languages/cpp_api.html#_C | 10QuakeValue4dumpERNSt7ostreamE), |
| PPv4I00EN5cudaq9gradients18forwar |     [\                            |
| d_difference8gradientER7KernelT), | [1\]](api/languages/cpp_api.html# |
|     [\[1\]](api/langua            | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq                        |
| q9gradients18forward_difference8g | ::QuakeValue::getRequiredElements |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\]](api/languages/cpp_    |     function)](api/langua         |
| api.html#_CPPv4I00EN5cudaq9gradie | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| nts18forward_difference8gradientE | uakeValue19getRequiredElementsEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::QuakeValue::getValue  |
|     [\[3\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4N5cudaq9gradients |     function)]                    |
| 18forward_difference8gradientERRN | (api/languages/cpp_api.html#_CPPv |
| St8functionIFvNSt6vectorIdEEEEE), | 4NK5cudaq10QuakeValue8getValueEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::QuakeValue::inverse   |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18forward_difference8gradientEv) |     function)                     |
| -   [                             | ](api/languages/cpp_api.html#_CPP |
| cudaq::gradients::parameter_shift | v4NK5cudaq10QuakeValue7inverseEv) |
|     (C++                          | -   [cudaq::QuakeValue::isStdVec  |
|     class)](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)                     |
| udaq9gradients15parameter_shiftE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::                      | v4N5cudaq10QuakeValue8isStdVecEv) |
| gradients::parameter_shift::clone | -                                 |
|     (C++                          |    [cudaq::QuakeValue::operator\* |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     function)](api                |
| adients15parameter_shift5cloneEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gr                    | udaq10QuakeValuemlE10QuakeValue), |
| adients::parameter_shift::compute |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     function                      | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::QuakeValue::operator+ |
| Pv4N5cudaq9gradients15parameter_s |     (C++                          |
| hift7computeERKNSt6vectorIdEERKNS |     function)](api                |
| t8functionIFdNSt6vectorIdEEEEEd), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]](api/languages/cpp_ap  | udaq10QuakeValueplE10QuakeValue), |
| i.html#_CPPv4N5cudaq9gradients15p |     [                             |
| arameter_shift7computeERKNSt6vect | \[1\]](api/languages/cpp_api.html |
| orIdEERNSt6vectorIdEERK7spin_opd) | #_CPPv4N5cudaq10QuakeValueplEKd), |
| -   [cudaq::gra                   |                                   |
| dients::parameter_shift::gradient | [\[2\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|     func                          | -   [cudaq::QuakeValue::operator- |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4I00EN5cudaq9gradients15par |     function)](api                |
| ameter_shift8gradientER7KernelT), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]](api/lan               | udaq10QuakeValuemiE10QuakeValue), |
| guages/cpp_api.html#_CPPv4I00EN5c |     [                             |
| udaq9gradients15parameter_shift8g | \[1\]](api/languages/cpp_api.html |
| radientER7KernelTRR10ArgsMapper), | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     [\[2\]](api/languages/c       |     [                             |
| pp_api.html#_CPPv4I00EN5cudaq9gra | \[2\]](api/languages/cpp_api.html |
| dients15parameter_shift8gradientE | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| RR13QuantumKernelRR10ArgsMapper), |                                   |
|     [\[3\]](api/languages/        | [\[3\]](api/languages/cpp_api.htm |
| cpp_api.html#_CPPv4N5cudaq9gradie | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| nts15parameter_shift8gradientERRN | -   [cudaq::QuakeValue::operator/ |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[4\]](api/languages         |     function)](api                |
| /cpp_api.html#_CPPv4N5cudaq9gradi | /languages/cpp_api.html#_CPPv4N5c |
| ents15parameter_shift8gradientEv) | udaq10QuakeValuedvE10QuakeValue), |
| -   [cudaq::kernel_builder (C++   |                                   |
|     clas                          | [\[1\]](api/languages/cpp_api.htm |
| s)](api/languages/cpp_api.html#_C | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| PPv4IDpEN5cudaq14kernel_builderE) | -                                 |
| -   [c                            |  [cudaq::QuakeValue::operator\[\] |
| udaq::kernel_builder::constantVal |     (C++                          |
|     (C++                          |     function)](api                |
|     function)](api/la             | /languages/cpp_api.html#_CPPv4N5c |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq10QuakeValueixEKNSt6size_tE), |
| q14kernel_builder11constantValEd) |     [\[1\]](api/                  |
| -                                 | languages/cpp_api.html#_CPPv4N5cu |
|  [cudaq::kernel_builder::detector | daq10QuakeValueixERK10QuakeValue) |
|     (C++                          | -                                 |
|                                   |    [cudaq::QuakeValue::QuakeValue |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4IDpEN5cudaq14kernel |     function)](api/languag        |
| _builder8detectorEvDpRR8MeasArgs) | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| -                                 | akeValue10QuakeValueERN4mlir20Imp |
| [cudaq::kernel_builder::detectors | licitLocOpBuilderEN4mlir5ValueE), |
|     (C++                          |     [\[1\]                        |
|     func                          | ](api/languages/cpp_api.html#_CPP |
| tion)](api/languages/cpp_api.html | v4N5cudaq10QuakeValue10QuakeValue |
| #_CPPv4N5cudaq14kernel_builder9de | ERN4mlir20ImplicitLocOpBuilderEd) |
| tectorsE10QuakeValue10QuakeValue) | -   [cudaq::QuakeValue::size (C++ |
| -   [cu                           |     funct                         |
| daq::kernel_builder::getArguments | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|     function)](api/lan            | -   [cudaq::QuakeValue::slice     |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 14kernel_builder12getArgumentsEv) |     function)](api/languages/cpp_ |
| -   [cu                           | api.html#_CPPv4N5cudaq10QuakeValu |
| daq::kernel_builder::getNumParams | e5sliceEKNSt6size_tEKNSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_platform (C++ |
|     function)](api/lan            |     cl                            |
| guages/cpp_api.html#_CPPv4N5cudaq | ass)](api/languages/cpp_api.html# |
| 14kernel_builder12getNumParamsEv) | _CPPv4N5cudaq16quantum_platformE) |
| -   [c                            | -   [cudaq:                       |
| udaq::kernel_builder::isArgStdVec | :quantum_platform::beginExecution |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/languag        |
| p_api.html#_CPPv4N5cudaq14kernel_ | es/cpp_api.html#_CPPv4N5cudaq16qu |
| builder11isArgStdVecENSt6size_tE) | antum_platform14beginExecutionEv) |
| -   [cuda                         | -   [cudaq::quantum_pl            |
| q::kernel_builder::kernel_builder | atform::configureExecutionContext |
|     (C++                          |     (C++                          |
|     function)](api/languages/cpp  |     function)](api/lang           |
| _api.html#_CPPv4N5cudaq14kernel_b | uages/cpp_api.html#_CPPv4NK5cudaq |
| uilder14kernel_builderERNSt6vecto | 16quantum_platform25configureExec |
| rIN6detail17KernelBuilderTypeEEE) | utionContextER16ExecutionContext) |
| -   [cudaq::k                     | -   [cuda                         |
| ernel_builder::logical_observable | q::quantum_platform::connectivity |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api/langu          |
| ](api/languages/cpp_api.html#_CPP | ages/cpp_api.html#_CPPv4N5cudaq16 |
| v4IDpEN5cudaq14kernel_builder18lo | quantum_platform12connectivityEv) |
| gical_observableEvDpRR8MeasArgs), | -   [cuda                         |
|     [\[1\]](ap                    | q::quantum_platform::endExecution |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq14kernel_builder18logical_ob |     function)](api/langu          |
| servableE10QuakeValueNSt6size_tE) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [cudaq::kernel_builder::name  | quantum_platform12endExecutionEv) |
|     (C++                          | -   [cudaq::q                     |
|     function)                     | uantum_platform::enqueueAsyncTask |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq14kernel_builder4nameEv) |     function)](api/languages/     |
| -                                 | cpp_api.html#_CPPv4N5cudaq16quant |
|    [cudaq::kernel_builder::qalloc | um_platform16enqueueAsyncTaskEKNS |
|     (C++                          | t6size_tER19KernelExecutionTask), |
|     function)](api/language       |     [\[1\]](api/languag           |
| s/cpp_api.html#_CPPv4N5cudaq14ker | es/cpp_api.html#_CPPv4N5cudaq16qu |
| nel_builder6qallocE10QuakeValue), | antum_platform16enqueueAsyncTaskE |
|     [\[1\]](api/language          | KNSt6size_tERNSt8functionIFvvEEE) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::quantum_p             |
| nel_builder6qallocEKNSt6size_tE), | latform::finalizeExecutionContext |
|     [\[2                          |     (C++                          |
| \]](api/languages/cpp_api.html#_C |     function)](api/languages/c    |
| PPv4N5cudaq14kernel_builder6qallo | pp_api.html#_CPPv4NK5cudaq16quant |
| cERNSt6vectorINSt7complexIdEEEE), | um_platform24finalizeExecutionCon |
|     [\[3\]](                      | textERN5cudaq16ExecutionContextE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qua                   |
| N5cudaq14kernel_builder6qallocEv) | ntum_platform::get_codegen_config |
| -   [cudaq::kernel_builder::swap  |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/language       | pp_api.html#_CPPv4N5cudaq16quantu |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | m_platform18get_codegen_configEv) |
| 4kernel_builder4swapEvRK10QuakeVa | -   [cuda                         |
| lueRK10QuakeValueRK10QuakeValue), | q::quantum_platform::get_exec_ctx |
|                                   |     (C++                          |
| [\[1\]](api/languages/cpp_api.htm |     function)](api/langua         |
| l#_CPPv4I00EN5cudaq14kernel_build | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| er4swapEvRKNSt6vectorI10QuakeValu | quantum_platform12get_exec_ctxEv) |
| eEERK10QuakeValueRK10QuakeValue), | -   [c                            |
|                                   | udaq::quantum_platform::get_noise |
| [\[2\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14kernel_builder4s |     function)](api/languages/c    |
| wapERK10QuakeValueRK10QuakeValue) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [cudaq::KernelExecutionTask   | m_platform9get_noiseENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     type                          | :quantum_platform::get_num_qubits |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq19KernelExecutionTaskE) |                                   |
| -   [cudaq::KernelThunkResultType | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4NK5cudaq16quantum_plat |
|     struct)]                      | form14get_num_qubitsENSt6size_tE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::quantum_              |
| 4N5cudaq21KernelThunkResultTypeE) | platform::get_remote_capabilities |
| -   [cudaq::KernelThunkType (C++  |     (C++                          |
|                                   |     function)                     |
| type)](api/languages/cpp_api.html | ](api/languages/cpp_api.html#_CPP |
| #_CPPv4N5cudaq15KernelThunkTypeE) | v4NK5cudaq16quantum_platform23get |
| -   [cudaq::kraus_channel (C++    | _remote_capabilitiesENSt6size_tE) |
|                                   | -   [cudaq::qua                   |
|  class)](api/languages/cpp_api.ht | ntum_platform::get_runtime_target |
| ml#_CPPv4N5cudaq13kraus_channelE) |     (C++                          |
| -   [cudaq::kraus_channel::empty  |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4NK5cudaq16quantu |
|     function)]                    | m_platform18get_runtime_targetEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cud                          |
| 4NK5cudaq13kraus_channel5emptyEv) | aq::quantum_platform::is_emulated |
| -   [cudaq::kraus_c               |     (C++                          |
| hannel::generateUnitaryParameters |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|                                   | pi.html#_CPPv4NK5cudaq16quantum_p |
|    function)](api/languages/cpp_a | latform11is_emulatedENSt6size_tE) |
| pi.html#_CPPv4N5cudaq13kraus_chan | -   [cudaq::                      |
| nel25generateUnitaryParametersEv) | quantum_platform::is_library_mode |
| -                                 |     (C++                          |
|    [cudaq::kraus_channel::get_ops |     function)](api/languages      |
|     (C++                          | /cpp_api.html#_CPPv4NK5cudaq16qua |
|     function)](a                  | ntum_platform15is_library_modeEv) |
| pi/languages/cpp_api.html#_CPPv4N | -   [c                            |
| K5cudaq13kraus_channel7get_opsEv) | udaq::quantum_platform::is_remote |
| -   [cud                          |     (C++                          |
| aq::kraus_channel::identity_flags |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4NK5cudaq16quantu |
|     member)](api/lan              | m_platform9is_remoteENSt6size_tE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cuda                         |
| 13kraus_channel14identity_flagsE) | q::quantum_platform::is_simulator |
| -   [cud                          |     (C++                          |
| aq::kraus_channel::is_identity_op |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq16quantum_pl |
|    function)](api/languages/cpp_a | atform12is_simulatorENSt6size_tE) |
| pi.html#_CPPv4NK5cudaq13kraus_cha | -   [c                            |
| nnel14is_identity_opENSt6size_tE) | udaq::quantum_platform::launchVQE |
| -   [cudaq::                      |     (C++                          |
| kraus_channel::is_unitary_mixture |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/languages      | N5cudaq16quantum_platform9launchV |
| /cpp_api.html#_CPPv4NK5cudaq13kra | QEEKNSt6stringEPKvPN5cudaq8gradie |
| us_channel18is_unitary_mixtureEv) | ntERKN5cudaq7spin_opERN5cudaq9opt |
| -   [cu                           | imizerEKiKNSt6size_tENSt6size_tE) |
| daq::kraus_channel::kraus_channel | -   [cudaq:                       |
|     (C++                          | :quantum_platform::list_platforms |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4IDpEN5cu |     function)](api/languag        |
| daq13kraus_channel13kraus_channel | es/cpp_api.html#_CPPv4N5cudaq16qu |
| EDpRRNSt16initializer_listI1TEE), | antum_platform14list_platformsEv) |
|                                   | -                                 |
|  [\[1\]](api/languages/cpp_api.ht |    [cudaq::quantum_platform::name |
| ml#_CPPv4N5cudaq13kraus_channel13 |     (C++                          |
| kraus_channelERK13kraus_channel), |     function)](a                  |
|     [\[2\]                        | pi/languages/cpp_api.html#_CPPv4N |
| ](api/languages/cpp_api.html#_CPP | K5cudaq16quantum_platform4nameEv) |
| v4N5cudaq13kraus_channel13kraus_c | -   [                             |
| hannelERKNSt6vectorI8kraus_opEE), | cudaq::quantum_platform::num_qpus |
|     [\[3\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/l              |
| v4N5cudaq13kraus_channel13kraus_c | anguages/cpp_api.html#_CPPv4NK5cu |
| hannelERRNSt6vectorI8kraus_opEE), | daq16quantum_platform8num_qpusEv) |
|     [\[4\]](api/lan               | -   [cudaq::                      |
| guages/cpp_api.html#_CPPv4N5cudaq | quantum_platform::onRandomSeedSet |
| 13kraus_channel13kraus_channelEv) |     (C++                          |
| -                                 |                                   |
| [cudaq::kraus_channel::noise_type | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq16quantum_platf |
|     member)](api                  | orm15onRandomSeedSetENSt6size_tE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq:                       |
| udaq13kraus_channel10noise_typeE) | :quantum_platform::reset_exec_ctx |
| -                                 |     (C++                          |
|   [cudaq::kraus_channel::op_names |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     member)](                     | antum_platform14reset_exec_ctxEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cud                          |
| N5cudaq13kraus_channel8op_namesE) | aq::quantum_platform::reset_noise |
| -                                 |     (C++                          |
|  [cudaq::kraus_channel::operator= |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq16quantum_p |
|     function)](api/langua         | latform11reset_noiseENSt6size_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq13k | -   [cuda                         |
| raus_channelaSERK13kraus_channel) | q::quantum_platform::set_exec_ctx |
| -   [c                            |     (C++                          |
| udaq::kraus_channel::operator\[\] |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/l              | _CPPv4N5cudaq16quantum_platform12 |
| anguages/cpp_api.html#_CPPv4N5cud | set_exec_ctxEP16ExecutionContext) |
| aq13kraus_channelixEKNSt6size_tE) | -   [c                            |
| -                                 | udaq::quantum_platform::set_noise |
| [cudaq::kraus_channel::parameters |     (C++                          |
|     (C++                          |     function                      |
|     member)](api                  | )](api/languages/cpp_api.html#_CP |
| /languages/cpp_api.html#_CPPv4N5c | Pv4N5cudaq16quantum_platform9set_ |
| udaq13kraus_channel10parametersE) | noiseEPK11noise_modelNSt6size_tE) |
| -   [cudaq::krau                  | -   [cudaq::quantum_platfor       |
| s_channel::populateDefaultOpNames | m::supports_explicit_measurements |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/l              |
| p_api.html#_CPPv4N5cudaq13kraus_c | anguages/cpp_api.html#_CPPv4NK5cu |
| hannel22populateDefaultOpNamesEv) | daq16quantum_platform30supports_e |
| -   [cu                           | xplicit_measurementsENSt6size_tE) |
| daq::kraus_channel::probabilities | -   [cudaq::quantum_pla           |
|     (C++                          | tform::supports_task_distribution |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     fu                            |
| q13kraus_channel13probabilitiesE) | nction)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4NK5cudaq16quantum_platfo |
|  [cudaq::kraus_channel::push_back | rm26supports_task_distributionEv) |
|     (C++                          | -   [cudaq::quantum               |
|     function)](api                | _platform::with_execution_context |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel9push_backE8kr |     function)                     |
| aus_opNSt8optionalINSt6stringEEE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus_channel::size   | v4I0DpEN5cudaq16quantum_platform2 |
|     (C++                          | 2with_execution_contextEDaR16Exec |
|     function)                     | utionContextRR8CallableDpRR4Args) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::QuantumTask (C++      |
| v4NK5cudaq13kraus_channel4sizeEv) |     type)](api/languages/cpp_api. |
| -   [                             | html#_CPPv4N5cudaq11QuantumTaskE) |
| cudaq::kraus_channel::unitary_ops | -   [cudaq::qubit (C++            |
|     (C++                          |     type)](api/languages/c        |
|     member)](api/                 | pp_api.html#_CPPv4N5cudaq5qubitE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::QubitConnectivity     |
| daq13kraus_channel11unitary_opsE) |     (C++                          |
| -   [cudaq::kraus_op (C++         |     ty                            |
|     struct)](api/languages/cpp_   | pe)](api/languages/cpp_api.html#_ |
| api.html#_CPPv4N5cudaq8kraus_opE) | CPPv4N5cudaq17QubitConnectivityE) |
| -   [cudaq::kraus_op::adjoint     | -   [cudaq::QubitEdge (C++        |
|     (C++                          |     type)](api/languages/cpp_a    |
|     functi                        | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| on)](api/languages/cpp_api.html#_ | -   [cudaq::qudit (C++            |
| CPPv4NK5cudaq8kraus_op7adjointEv) |     clas                          |
| -   [cudaq::kraus_op::data (C++   | s)](api/languages/cpp_api.html#_C |
|                                   | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|  member)](api/languages/cpp_api.h | -   [cudaq::qudit::qudit (C++     |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |                                   |
| -   [cudaq::kraus_op::kraus_op    | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qudit5quditEv) |
|     func                          | -   [cudaq::qvector (C++          |
| tion)](api/languages/cpp_api.html |     class)                        |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | ](api/languages/cpp_api.html#_CPP |
| opERRNSt16initializer_listI1TEE), | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|                                   | -   [cudaq::qvector::back (C++    |
|  [\[1\]](api/languages/cpp_api.ht |     function)](a                  |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | pi/languages/cpp_api.html#_CPPv4N |
| pENSt6vectorIN5cudaq7complexEEE), | 5cudaq7qvector4backENSt6size_tE), |
|     [\[2\]](api/l                 |                                   |
| anguages/cpp_api.html#_CPPv4N5cud |   [\[1\]](api/languages/cpp_api.h |
| aq8kraus_op8kraus_opERK8kraus_op) | tml#_CPPv4N5cudaq7qvector4backEv) |
| -   [cudaq::kraus_op::nCols (C++  | -   [cudaq::qvector::begin (C++   |
|                                   |     fu                            |
| member)](api/languages/cpp_api.ht | nction)](api/languages/cpp_api.ht |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | ml#_CPPv4N5cudaq7qvector5beginEv) |
| -   [cudaq::kraus_op::nRows (C++  | -   [cudaq::qvector::clear (C++   |
|                                   |     fu                            |
| member)](api/languages/cpp_api.ht | nction)](api/languages/cpp_api.ht |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | ml#_CPPv4N5cudaq7qvector5clearEv) |
| -   [cudaq::kraus_op::operator=   | -   [cudaq::qvector::end (C++     |
|     (C++                          |                                   |
|     function)                     | function)](api/languages/cpp_api. |
| ](api/languages/cpp_api.html#_CPP | html#_CPPv4N5cudaq7qvector3endEv) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cudaq::qvector::front (C++   |
| -   [cudaq::kraus_op::precision   |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     memb                          | cudaq7qvector5frontENSt6size_tE), |
| er)](api/languages/cpp_api.html#_ |                                   |
| CPPv4N5cudaq8kraus_op9precisionE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::KrausSelection (C++   | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     s                             | -   [cudaq::qvector::operator=    |
| truct)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14KrausSelectionE) |     functio                       |
| -   [cudaq:                       | n)](api/languages/cpp_api.html#_C |
| :KrausSelection::circuit_location | PPv4N5cudaq7qvectoraSERK7qvector) |
|     (C++                          | -   [cudaq::qvector::operator\[\] |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14K |     function)                     |
| rausSelection16circuit_locationE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq7qvectorixEKNSt6size_tE) |
|  [cudaq::KrausSelection::is_error | -   [cudaq::qvector::qvector (C++ |
|     (C++                          |     function)](api/               |
|     member)](a                    | languages/cpp_api.html#_CPPv4N5cu |
| pi/languages/cpp_api.html#_CPPv4N | daq7qvector7qvectorENSt6size_tE), |
| 5cudaq14KrausSelection8is_errorE) |     [\[1\]](a                     |
| -   [cudaq::Kra                   | pi/languages/cpp_api.html#_CPPv4N |
| usSelection::kraus_operator_index | 5cudaq7qvector7qvectorERK5state), |
|     (C++                          |     [\[2\]](api                   |
|     member)](api/languages/       | /languages/cpp_api.html#_CPPv4N5c |
| cpp_api.html#_CPPv4N5cudaq14Kraus | udaq7qvector7qvectorERK7qvector), |
| Selection20kraus_operator_indexE) |     [\[3\]](ap                    |
| -   [cuda                         | i/languages/cpp_api.html#_CPPv4N5 |
| q::KrausSelection::KrausSelection | cudaq7qvector7qvectorERR7qvector) |
|     (C++                          | -   [cudaq::qvector::size (C++    |
|     function)](a                  |     fu                            |
| pi/languages/cpp_api.html#_CPPv4N | nction)](api/languages/cpp_api.ht |
| 5cudaq14KrausSelection14KrausSele | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| ctionENSt6size_tENSt6vectorINSt6s | -   [cudaq::qvector::slice (C++   |
| ize_tEEENSt6stringENSt6size_tEb), |     function)](api/language       |
|     [\[1\]](api/langu             | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| ages/cpp_api.html#_CPPv4N5cudaq14 | tor5sliceENSt6size_tENSt6size_tE) |
| KrausSelection14KrausSelectionEv) | -   [cudaq::qvector::value_type   |
| -                                 |     (C++                          |
|   [cudaq::KrausSelection::op_name |     typ                           |
|     (C++                          | e)](api/languages/cpp_api.html#_C |
|     member)](                     | PPv4N5cudaq7qvector10value_typeE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qview (C++            |
| N5cudaq14KrausSelection7op_nameE) |     clas                          |
| -   [                             | s)](api/languages/cpp_api.html#_C |
| cudaq::KrausSelection::operator== | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     (C++                          | -   [cudaq::qview::back (C++      |
|     function)](api/languages      |     function)                     |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | ](api/languages/cpp_api.html#_CPP |
| usSelectioneqERK14KrausSelection) | v4N5cudaq5qview4backENSt6size_tE) |
| -                                 | -   [cudaq::qview::begin (C++     |
|    [cudaq::KrausSelection::qubits |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     member)]                      | html#_CPPv4N5cudaq5qview5beginEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::qview::end (C++       |
| 4N5cudaq14KrausSelection6qubitsE) |                                   |
| -   [cudaq::KrausTrajectory (C++  |   function)](api/languages/cpp_ap |
|     st                            | i.html#_CPPv4N5cudaq5qview3endEv) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::qview::front (C++     |
| #_CPPv4N5cudaq15KrausTrajectoryE) |     function)](                   |
| -                                 | api/languages/cpp_api.html#_CPPv4 |
|  [cudaq::KrausTrajectory::builder | N5cudaq5qview5frontENSt6size_tE), |
|     (C++                          |                                   |
|     function)](ap                 |    [\[1\]](api/languages/cpp_api. |
| i/languages/cpp_api.html#_CPPv4N5 | html#_CPPv4N5cudaq5qview5frontEv) |
| cudaq15KrausTrajectory7builderEv) | -   [cudaq::qview::operator\[\]   |
| -   [cu                           |     (C++                          |
| daq::KrausTrajectory::countErrors |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)](api/lang           | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::qview::qview (C++     |
| 15KrausTrajectory11countErrorsEv) |     functio                       |
| -   [                             | n)](api/languages/cpp_api.html#_C |
| cudaq::KrausTrajectory::isOrdered | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     (C++                          |     [\[1                          |
|     function)](api/l              | \]](api/languages/cpp_api.html#_C |
| anguages/cpp_api.html#_CPPv4NK5cu | PPv4N5cudaq5qview5qviewERK5qview) |
| daq15KrausTrajectory9isOrderedEv) | -   [cudaq::qview::size (C++      |
| -   [cudaq::                      |                                   |
| KrausTrajectory::kraus_selections | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4NK5cudaq5qview4sizeEv) |
|     member)](api/languag          | -   [cudaq::qview::slice (C++     |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |     function)](api/langua         |
| ausTrajectory16kraus_selectionsE) | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| -   [cudaq:                       | iew5sliceENSt6size_tENSt6size_tE) |
| :KrausTrajectory::KrausTrajectory | -   [cudaq::qview::value_type     |
|     (C++                          |     (C++                          |
|     function                      |     t                             |
| )](api/languages/cpp_api.html#_CP | ype)](api/languages/cpp_api.html# |
| Pv4N5cudaq15KrausTrajectory15Krau | _CPPv4N5cudaq5qview10value_typeE) |
| sTrajectoryENSt6size_tENSt6vector | -   [cudaq::range (C++            |
| I14KrausSelectionEEdNSt6size_tE), |     fun                           |
|     [\[1\]](api/languag           | ction)](api/languages/cpp_api.htm |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| ausTrajectory15KrausTrajectoryEv) | orI11ElementTypeEE11ElementType), |
| -   [cudaq::Kr                    |     [\[1\]](api/languages/cpp_    |
| ausTrajectory::measurement_counts | api.html#_CPPv4I0EN5cudaq5rangeEN |
|     (C++                          | St6vectorI11ElementTypeEE11Elemen |
|     member)](api/languages        | tType11ElementType11ElementType), |
| /cpp_api.html#_CPPv4N5cudaq15Krau |     [                             |
| sTrajectory18measurement_countsE) | \[2\]](api/languages/cpp_api.html |
| -   [cud                          | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| aq::KrausTrajectory::multiplicity | -   [cudaq::real (C++             |
|     (C++                          |     type)](api/languages/         |
|     member)](api/lan              | cpp_api.html#_CPPv4N5cudaq4realE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::registry (C++         |
| 15KrausTrajectory12multiplicityE) |     type)](api/languages/cpp_     |
| -   [                             | api.html#_CPPv4N5cudaq8registryE) |
| cudaq::KrausTrajectory::num_shots | -                                 |
|     (C++                          |  [cudaq::registry::RegisteredType |
|     member)](api                  |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     class)](api/                  |
| udaq15KrausTrajectory9num_shotsE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [c                            | 5cudaq8registry14RegisteredTypeE) |
| udaq::KrausTrajectory::operator== | -   [cudaq::RemoteCapabilities    |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     struc                         |
| pp_api.html#_CPPv4NK5cudaq15Kraus | t)](api/languages/cpp_api.html#_C |
| TrajectoryeqERK15KrausTrajectory) | PPv4N5cudaq18RemoteCapabilitiesE) |
| -   [cu                           | -   [cudaq::Remot                 |
| daq::KrausTrajectory::probability | eCapabilities::RemoteCapabilities |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api/languages/cpp  |
| nguages/cpp_api.html#_CPPv4N5cuda | _api.html#_CPPv4N5cudaq18RemoteCa |
| q15KrausTrajectory11probabilityE) | pabilities18RemoteCapabilitiesEb) |
| -   [cuda                         | -   [cudaq:                       |
| q::KrausTrajectory::trajectory_id | :RemoteCapabilities::stateOverlap |
|     (C++                          |     (C++                          |
|     member)](api/lang             |     member)](api/langua           |
| uages/cpp_api.html#_CPPv4N5cudaq1 | ges/cpp_api.html#_CPPv4N5cudaq18R |
| 5KrausTrajectory13trajectory_idE) | emoteCapabilities12stateOverlapE) |
| -                                 | -                                 |
|   [cudaq::KrausTrajectory::weight |   [cudaq::RemoteCapabilities::vqe |
|     (C++                          |     (C++                          |
|     member)](                     |     member)](                     |
| api/languages/cpp_api.html#_CPPv4 | api/languages/cpp_api.html#_CPPv4 |
| N5cudaq15KrausTrajectory6weightE) | N5cudaq18RemoteCapabilities3vqeE) |
| -                                 | -   [cudaq::Resources (C++        |
|    [cudaq::KrausTrajectoryBuilder |     class)](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     class)](                      | -   [cudaq::run (C++              |
| api/languages/cpp_api.html#_CPPv4 |     function)]                    |
| N5cudaq22KrausTrajectoryBuilderE) | (api/languages/cpp_api.html#_CPPv |
| -   [cud                          | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| aq::KrausTrajectoryBuilder::build | 5invoke_result_tINSt7decay_tI13Qu |
|     (C++                          | antumKernelEEDpNSt7decay_tI4ARGSE |
|     function)](api/lang           | EEEEENSt6size_tERN5cudaq11noise_m |
| uages/cpp_api.html#_CPPv4NK5cudaq | odelERR13QuantumKernelDpRR4ARGS), |
| 22KrausTrajectoryBuilder5buildEv) |     [\[1\]](api/langu             |
| -   [cud                          | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| aq::KrausTrajectoryBuilder::setId | daq3runENSt6vectorINSt15invoke_re |
|     (C++                          | sult_tINSt7decay_tI13QuantumKerne |
|     function)](api/languages/cpp  | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| _api.html#_CPPv4N5cudaq22KrausTra | ize_tERR13QuantumKernelDpRR4ARGS) |
| jectoryBuilder5setIdENSt6size_tE) | -   [cudaq::run_async (C++        |
| -   [cudaq::Kraus                 |     functio                       |
| TrajectoryBuilder::setProbability | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     function)](api/languages/cpp  | tureINSt6vectorINSt15invoke_resul |
| _api.html#_CPPv4N5cudaq22KrausTra | t_tINSt7decay_tI13QuantumKernelEE |
| jectoryBuilder14setProbabilityEd) | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| -   [cudaq::Krau                  | ze_tENSt6size_tERN5cudaq11noise_m |
| sTrajectoryBuilder::setSelections | odelERR13QuantumKernelDpRR4ARGS), |
|     (C++                          |     [\[1\]](api/la                |
|     function)](api/languag        | nguages/cpp_api.html#_CPPv4I0DpEN |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | 5cudaq9run_asyncENSt6futureINSt6v |
| ausTrajectoryBuilder13setSelectio | ectorINSt15invoke_result_tINSt7de |
| nsENSt6vectorI14KrausSelectionEE) | cay_tI13QuantumKernelEEDpNSt7deca |
| -   [cudaq::logical_observable    | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4ARGS) |
|     function)](api/languages/c    | -   [cudaq::RuntimeTarget (C++    |
| pp_api.html#_CPPv4IDpEN5cudaq18lo |                                   |
| gical_observableEvDpRR8MeasArgs), | struct)](api/languages/cpp_api.ht |
|     [\[1\]](api/l                 | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cudaq::sample (C++           |
| aq18logical_observableERKNSt6vect |     function)](api/languages/c    |
| orI14measure_resultEENSt6size_tE) | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| -   [cudaq::M2DSparseMatrix (C++  | mpleE13sample_resultRK14sample_op |
|     st                            | tionsRR13QuantumKernelDpRR4Args), |
| ruct)](api/languages/cpp_api.html |     [\[1\                         |
| #_CPPv4N5cudaq15M2DSparseMatrixE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::M2OSparseMatrix (C++  | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     st                            | esultRR13QuantumKernelDpRR4Args), |
| ruct)](api/languages/cpp_api.html |     [\                            |
| #_CPPv4N5cudaq15M2OSparseMatrixE) | [2\]](api/languages/cpp_api.html# |
| -   [cudaq::matrix_callback (C++  | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|     c                             | ize_tERR13QuantumKernelDpRR4Args) |
| lass)](api/languages/cpp_api.html | -   [cudaq::sample_options (C++   |
| #_CPPv4N5cudaq15matrix_callbackE) |     s                             |
| -   [cudaq::matrix_handler (C++   | truct)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq14sample_optionsE) |
| class)](api/languages/cpp_api.htm | -   [cudaq::sample_result (C++    |
| l#_CPPv4N5cudaq14matrix_handlerE) |                                   |
| -   [cudaq::mat                   |  class)](api/languages/cpp_api.ht |
| rix_handler::commutation_behavior | ml#_CPPv4N5cudaq13sample_resultE) |
|     (C++                          | -   [cudaq::sample_result::append |
|     struct)](api/languages/       |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq14matri |     function)](api/languages/cpp_ |
| x_handler20commutation_behaviorE) | api.html#_CPPv4N5cudaq13sample_re |
| -                                 | sult6appendERK15ExecutionResultb) |
|    [cudaq::matrix_handler::define | -   [cudaq::sample_result::begin  |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)]                    |
| pi/languages/cpp_api.html#_CPPv4N | (api/languages/cpp_api.html#_CPPv |
| 5cudaq14matrix_handler6defineENSt | 4N5cudaq13sample_result5beginEv), |
| 6stringENSt6vectorINSt7int64_tEEE |     [\[1\]]                       |
| RR15matrix_callbackRKNSt13unorder | (api/languages/cpp_api.html#_CPPv |
| ed_mapINSt6stringENSt6stringEEE), | 4NK5cudaq13sample_result5beginEv) |
|                                   | -   [cudaq::sample_result::cbegin |
| [\[1\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14matrix_handler6d |     function)](                   |
| efineENSt6stringENSt6vectorINSt7i | api/languages/cpp_api.html#_CPPv4 |
| nt64_tEEERR15matrix_callbackRR20d | NK5cudaq13sample_result6cbeginEv) |
| iag_matrix_callbackRKNSt13unorder | -   [cudaq::sample_result::cend   |
| ed_mapINSt6stringENSt6stringEEE), |     (C++                          |
|     [\[2\]](                      |     function)                     |
| api/languages/cpp_api.html#_CPPv4 | ](api/languages/cpp_api.html#_CPP |
| N5cudaq14matrix_handler6defineENS | v4NK5cudaq13sample_result4cendEv) |
| t6stringENSt6vectorINSt7int64_tEE | -   [cudaq::sample_result::clear  |
| ERR15matrix_callbackRRNSt13unorde |     (C++                          |
| red_mapINSt6stringENSt6stringEEE) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|   [cudaq::matrix_handler::degrees | v4N5cudaq13sample_result5clearEv) |
|     (C++                          | -   [cudaq::sample_result::count  |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4NK |     function)](                   |
| 5cudaq14matrix_handler7degreesEv) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | NK5cudaq13sample_result5countENSt |
|  [cudaq::matrix_handler::displace | 11string_viewEKNSt11string_viewE) |
|     (C++                          | -   [                             |
|     function)](api/language       | cudaq::sample_result::deserialize |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8displaceENSt6size_tE) |     functio                       |
| -   [cudaq::matrix                | n)](api/languages/cpp_api.html#_C |
| _handler::get_expected_dimensions | PPv4N5cudaq13sample_result11deser |
|     (C++                          | ializeERNSt6vectorINSt6size_tEEE) |
|                                   | -   [cudaq::sample_result::dump   |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     function)](api/languag        |
| ndler23get_expected_dimensionsEv) | es/cpp_api.html#_CPPv4NK5cudaq13s |
| -   [cudaq::matrix_ha             | ample_result4dumpERNSt7ostreamE), |
| ndler::get_parameter_descriptions |     [\[1\]                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|                                   | v4NK5cudaq13sample_result4dumpEv) |
| function)](api/languages/cpp_api. | -   [cudaq::sample_result::end    |
| html#_CPPv4NK5cudaq14matrix_handl |     (C++                          |
| er26get_parameter_descriptionsEv) |     function                      |
| -   [c                            | )](api/languages/cpp_api.html#_CP |
| udaq::matrix_handler::instantiate | Pv4N5cudaq13sample_result3endEv), |
|     (C++                          |     [\[1\                         |
|     function)](a                  | ]](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4NK5cudaq13sample_result3endEv) |
| 5cudaq14matrix_handler11instantia | -   [                             |
| teENSt6stringERKNSt6vectorINSt6si | cudaq::sample_result::expectation |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[1\]](                      |     f                             |
| api/languages/cpp_api.html#_CPPv4 | unction)](api/languages/cpp_api.h |
| N5cudaq14matrix_handler11instanti | tml#_CPPv4NK5cudaq13sample_result |
| ateENSt6stringERRNSt6vectorINSt6s | 11expectationEKNSt11string_viewE) |
| ize_tEEERK20commutation_behavior) | -   [c                            |
| -   [cuda                         | udaq::sample_result::get_marginal |
| q::matrix_handler::matrix_handler |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     function)](api/languag        | api.html#_CPPv4NK5cudaq13sample_r |
| es/cpp_api.html#_CPPv4I0_NSt11ena | esult12get_marginalERKNSt6vectorI |
| ble_if_tINSt12is_base_of_vI16oper | NSt6size_tEEEKNSt11string_viewE), |
| ator_handler1TEEbEEEN5cudaq14matr |     [\[1\]](api/languages/cpp_    |
| ix_handler14matrix_handlerERK1T), | api.html#_CPPv4NK5cudaq13sample_r |
|     [\[1\]](ap                    | esult12get_marginalERRKNSt6vector |
| i/languages/cpp_api.html#_CPPv4I0 | INSt6size_tEEEKNSt11string_viewE) |
| _NSt11enable_if_tINSt12is_base_of | -   [cuda                         |
| _vI16operator_handler1TEEbEEEN5cu | q::sample_result::get_total_shots |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erERK1TRK20commutation_behavior), |     function)](api/langua         |
|     [\[2\]](api/languages/cpp_ap  | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| i.html#_CPPv4N5cudaq14matrix_hand | sample_result15get_total_shotsEv) |
| ler14matrix_handlerENSt6size_tE), | -   [cuda                         |
|     [\[3\]](api/                  | q::sample_result::has_even_parity |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     fun                           |
| erENSt6stringERKNSt6vectorINSt6si | ction)](api/languages/cpp_api.htm |
| ze_tEEERK20commutation_behavior), | l#_CPPv4N5cudaq13sample_result15h |
|     [\[4\]](api/                  | as_even_parityENSt11string_viewE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cuda                         |
| daq14matrix_handler14matrix_handl | q::sample_result::has_expectation |
| erENSt6stringERRNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     funct                         |
|     [\                            | ion)](api/languages/cpp_api.html# |
| [5\]](api/languages/cpp_api.html# | _CPPv4NK5cudaq13sample_result15ha |
| _CPPv4N5cudaq14matrix_handler14ma | s_expectationEKNSt11string_viewE) |
| trix_handlerERK14matrix_handler), | -   [cu                           |
|     [                             | daq::sample_result::most_probable |
| \[6\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq14matrix_handler14m |     fun                           |
| atrix_handlerERR14matrix_handler) | ction)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4NK5cudaq13sample_result13 |
|  [cudaq::matrix_handler::momentum | most_probableEKNSt11string_viewE) |
|     (C++                          | -                                 |
|     function)](api/language       | [cudaq::sample_result::operator+= |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8momentumENSt6size_tE) |     function)](api/langua         |
| -                                 | ges/cpp_api.html#_CPPv4N5cudaq13s |
|    [cudaq::matrix_handler::number | ample_resultpLERK13sample_result) |
|     (C++                          | -                                 |
|     function)](api/langua         |  [cudaq::sample_result::operator= |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6numberENSt6size_tE) |     function)](api/langua         |
| -                                 | ges/cpp_api.html#_CPPv4N5cudaq13s |
| [cudaq::matrix_handler::operator= | ample_resultaSERR13sample_result) |
|     (C++                          | -                                 |
|     fun                           | [cudaq::sample_result::operator== |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     function)](api/languag        |
| NSt7is_sameI1T14matrix_handlerE5v | es/cpp_api.html#_CPPv4NK5cudaq13s |
| alueENSt12is_base_of_vI16operator | ample_resulteqERK13sample_result) |
| _handler1TEEEbEEEN5cudaq14matrix_ | -   [                             |
| handleraSER14matrix_handlerRK1T), | cudaq::sample_result::probability |
|     [\[1\]](api/languages         |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq14matr |     function)](api/lan            |
| ix_handleraSERK14matrix_handler), | guages/cpp_api.html#_CPPv4NK5cuda |
|     [\[2\]](api/language          | q13sample_result11probabilityENSt |
| s/cpp_api.html#_CPPv4N5cudaq14mat | 11string_viewEKNSt11string_viewE) |
| rix_handleraSERR14matrix_handler) | -   [cud                          |
| -   [                             | aq::sample_result::register_names |
| cudaq::matrix_handler::operator== |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](api/languages      | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| /cpp_api.html#_CPPv4NK5cudaq14mat | 3sample_result14register_namesEv) |
| rix_handlereqERK14matrix_handler) | -                                 |
| -                                 |    [cudaq::sample_result::reorder |
|    [cudaq::matrix_handler::parity |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/langua         | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ges/cpp_api.html#_CPPv4N5cudaq14m | ample_result7reorderERKNSt6vector |
| atrix_handler6parityENSt6size_tE) | INSt6size_tEEEKNSt11string_viewE) |
| -                                 | -   [cu                           |
|  [cudaq::matrix_handler::position | daq::sample_result::sample_result |
|     (C++                          |     (C++                          |
|     function)](api/language       |     func                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat | tion)](api/languages/cpp_api.html |
| rix_handler8positionENSt6size_tE) | #_CPPv4N5cudaq13sample_result13sa |
| -   [cudaq::                      | mple_resultERK15ExecutionResult), |
| matrix_handler::remove_definition |     [\[1\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     fu                            | q13sample_result13sample_resultER |
| nction)](api/languages/cpp_api.ht | KNSt6vectorI15ExecutionResultEE), |
| ml#_CPPv4N5cudaq14matrix_handler1 |                                   |
| 7remove_definitionERKNSt6stringE) |  [\[2\]](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq13sample_result13 |
|   [cudaq::matrix_handler::squeeze | sample_resultERR13sample_result), |
|     (C++                          |     [                             |
|     function)](api/languag        | \[3\]](api/languages/cpp_api.html |
| es/cpp_api.html#_CPPv4N5cudaq14ma | #_CPPv4N5cudaq13sample_result13sa |
| trix_handler7squeezeENSt6size_tE) | mple_resultERR15ExecutionResult), |
| -   [cudaq::m                     |     [\[4\]](api/lan               |
| atrix_handler::to_diagonal_matrix | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 13sample_result13sample_resultEdR |
|     function)](api/lang           | KNSt6vectorI15ExecutionResultEE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[5\]](api/lan               |
| 14matrix_handler18to_diagonal_mat | guages/cpp_api.html#_CPPv4N5cudaq |
| rixERNSt13unordered_mapINSt6size_ | 13sample_result13sample_resultEv) |
| tENSt7int64_tEEERKNSt13unordered_ | -                                 |
| mapINSt6stringENSt7complexIdEEEE) |  [cudaq::sample_result::serialize |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::to_matrix |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4NK5 |
|     function)                     | cudaq13sample_result9serializeEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::sample_result::size   |
| v4NK5cudaq14matrix_handler9to_mat |     (C++                          |
| rixERNSt13unordered_mapINSt6size_ |     function)](api/languages/c    |
| tENSt7int64_tEEERKNSt13unordered_ | pp_api.html#_CPPv4NK5cudaq13sampl |
| mapINSt6stringENSt7complexIdEEEE) | e_result4sizeEKNSt11string_viewE) |
| -                                 | -   [cudaq::sample_result::to_map |
| [cudaq::matrix_handler::to_string |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api/               | _api.html#_CPPv4NK5cudaq13sample_ |
| languages/cpp_api.html#_CPPv4NK5c | result6to_mapEKNSt11string_viewE) |
| udaq14matrix_handler9to_stringEb) | -   [cuda                         |
| -                                 | q::sample_result::\~sample_result |
| [cudaq::matrix_handler::unique_id |     (C++                          |
|     (C++                          |     funct                         |
|     function)](api/               | ion)](api/languages/cpp_api.html# |
| languages/cpp_api.html#_CPPv4NK5c | _CPPv4N5cudaq13sample_resultD0Ev) |
| udaq14matrix_handler9unique_idEv) | -   [cudaq::scalar_callback (C++  |
| -   [cudaq:                       |     c                             |
| :matrix_handler::\~matrix_handler | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_callbackE) |
|     functi                        | -   [c                            |
| on)](api/languages/cpp_api.html#_ | udaq::scalar_callback::operator() |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     (C++                          |
| -   [cudaq::matrix_op (C++        |     function)](api/language       |
|     type)](api/languages/cpp_a    | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| pi.html#_CPPv4N5cudaq9matrix_opE) | alar_callbackclERKNSt13unordered_ |
| -   [cudaq::matrix_op_term (C++   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [                             |
|  type)](api/languages/cpp_api.htm | cudaq::scalar_callback::operator= |
| l#_CPPv4N5cudaq14matrix_op_termE) |     (C++                          |
| -                                 |     function)](api/languages/c    |
|    [cudaq::mdiag_operator_handler | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _callbackaSERK15scalar_callback), |
|     class)](                      |     [\[1\]](api/languages/        |
| api/languages/cpp_api.html#_CPPv4 | cpp_api.html#_CPPv4N5cudaq15scala |
| N5cudaq22mdiag_operator_handlerE) | r_callbackaSERR15scalar_callback) |
| -   [cudaq::measure_handle (C++   | -   [cudaq:                       |
|                                   | :scalar_callback::scalar_callback |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14measure_handleE) |     function)](api/languag        |
| -   [cudaq::measure_result (C++   | es/cpp_api.html#_CPPv4I0_NSt11ena |
|                                   | ble_if_tINSt16is_invocable_r_vINS |
|  type)](api/languages/cpp_api.htm | t7complexIdEE8CallableRKNSt13unor |
| l#_CPPv4N5cudaq14measure_resultE) | dered_mapINSt6stringENSt7complexI |
| -   [cudaq::mpi (C++              | dEEEEEEbEEEN5cudaq15scalar_callba |
|     type)](api/languages          | ck15scalar_callbackERR8Callable), |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) |     [\[1\                         |
| -   [cudaq::mpi::all_gather (C++  | ]](api/languages/cpp_api.html#_CP |
|     fu                            | Pv4N5cudaq15scalar_callback15scal |
| nction)](api/languages/cpp_api.ht | ar_callbackERK15scalar_callback), |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     [\[2                          |
| RNSt6vectorIdEERKNSt6vectorIdEE), | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq15scalar_callback15sca |
|   [\[1\]](api/languages/cpp_api.h | lar_callbackERR15scalar_callback) |
| tml#_CPPv4N5cudaq3mpi10all_gather | -   [cudaq::scalar_operator (C++  |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     c                             |
| -   [cudaq::mpi::all_reduce (C++  | lass)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq15scalar_operatorE) |
|  function)](api/languages/cpp_api | -                                 |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | [cudaq::scalar_operator::evaluate |
| reduceE1TRK1TRK14BinaryFunction), |     (C++                          |
|     [\[1\]](api/langu             |                                   |
| ages/cpp_api.html#_CPPv4I00EN5cud |    function)](api/languages/cpp_a |
| aq3mpi10all_reduceE1TRK1TRK4Func) | pi.html#_CPPv4NK5cudaq15scalar_op |
| -   [cudaq::mpi::broadcast (C++   | erator8evaluateERKNSt13unordered_ |
|     function)](api/               | mapINSt6stringENSt7complexIdEEEE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::scalar_ope            |
| daq3mpi9broadcastERNSt6stringEi), | rator::get_parameter_descriptions |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     f                             |
| q3mpi9broadcastERNSt6vectorIdEEi) | unction)](api/languages/cpp_api.h |
| -   [cudaq::mpi::finalize (C++    | tml#_CPPv4NK5cudaq15scalar_operat |
|     f                             | or26get_parameter_descriptionsEv) |
| unction)](api/languages/cpp_api.h | -   [cu                           |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | daq::scalar_operator::is_constant |
| -   [cudaq::mpi::initialize (C++  |     (C++                          |
|     function                      |     function)](api/lang           |
| )](api/languages/cpp_api.html#_CP | uages/cpp_api.html#_CPPv4NK5cudaq |
| Pv4N5cudaq3mpi10initializeEiPPc), | 15scalar_operator11is_constantEv) |
|     [                             | -   [c                            |
| \[1\]](api/languages/cpp_api.html | udaq::scalar_operator::operator\* |
| #_CPPv4N5cudaq3mpi10initializeEv) |     (C++                          |
| -   [cudaq::mpi::is_initialized   |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function                      | Pv4N5cudaq15scalar_operatormlENSt |
| )](api/languages/cpp_api.html#_CP | 7complexIdEERK15scalar_operator), |
| Pv4N5cudaq3mpi14is_initializedEv) |     [\[1\                         |
| -   [cudaq::mpi::num_ranks (C++   | ]](api/languages/cpp_api.html#_CP |
|     fu                            | Pv4N5cudaq15scalar_operatormlENSt |
| nction)](api/languages/cpp_api.ht | 7complexIdEERR15scalar_operator), |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     [\[2\]](api/languages/cp      |
| -   [cudaq::mpi::rank (C++        | p_api.html#_CPPv4N5cudaq15scalar_ |
|                                   | operatormlEdRK15scalar_operator), |
|    function)](api/languages/cpp_a |     [\[3\]](api/languages/cp      |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::noise_model (C++      | operatormlEdRR15scalar_operator), |
|                                   |     [\[4\]](api/languages         |
|    class)](api/languages/cpp_api. | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| html#_CPPv4N5cudaq11noise_modelE) | alar_operatormlENSt7complexIdEE), |
| -   [cudaq::n                     |     [\[5\]](api/languages/cpp     |
| oise_model::add_all_qubit_channel | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatormlERK15scalar_operator), |
|     function)](api                |     [\[6\]]                       |
| /languages/cpp_api.html#_CPPv4IDp | (api/languages/cpp_api.html#_CPPv |
| EN5cudaq11noise_model21add_all_qu | 4NKR5cudaq15scalar_operatormlEd), |
| bit_channelEvRK13kraus_channeli), |     [\[7\]](api/language          |
|     [\[1\]](api/langua            | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| ges/cpp_api.html#_CPPv4N5cudaq11n | alar_operatormlENSt7complexIdEE), |
| oise_model21add_all_qubit_channel |     [\[8\]](api/languages/cp      |
| ERKNSt6stringERK13kraus_channeli) | p_api.html#_CPPv4NO5cudaq15scalar |
| -                                 | _operatormlERK15scalar_operator), |
|  [cudaq::noise_model::add_channel |     [\[9\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     funct                         | Pv4NO5cudaq15scalar_operatormlEd) |
| ion)](api/languages/cpp_api.html# | -   [cu                           |
| _CPPv4IDpEN5cudaq11noise_model11a | daq::scalar_operator::operator\*= |
| dd_channelEvRK15PredicateFuncTy), |     (C++                          |
|     [\[1\]](api/languages/cpp_    |     function)](api/languag        |
| api.html#_CPPv4IDpEN5cudaq11noise | es/cpp_api.html#_CPPv4N5cudaq15sc |
| _model11add_channelEvRKNSt6vector | alar_operatormLENSt7complexIdEE), |
| INSt6size_tEEERK13kraus_channel), |     [\[1\]](api/languages/c       |
|     [\[2\]](ap                    | pp_api.html#_CPPv4N5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatormLERK15scalar_operator), |
| cudaq11noise_model11add_channelER |     [\[2                          |
| KNSt6stringERK15PredicateFuncTy), | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq15scalar_operatormLEd) |
| [\[3\]](api/languages/cpp_api.htm | -   [                             |
| l#_CPPv4N5cudaq11noise_model11add | cudaq::scalar_operator::operator+ |
| _channelERKNSt6stringERKNSt6vecto |     (C++                          |
| rINSt6size_tEEERK13kraus_channel) |     function                      |
| -   [cudaq::noise_model::empty    | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatorplENSt |
|     function                      | 7complexIdEERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[1\                         |
| Pv4NK5cudaq11noise_model5emptyEv) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatorplENSt |
| [cudaq::noise_model::get_channels | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     function)](api/l              | p_api.html#_CPPv4N5cudaq15scalar_ |
| anguages/cpp_api.html#_CPPv4I0ENK | operatorplEdRK15scalar_operator), |
| 5cudaq11noise_model12get_channels |     [\[3\]](api/languages/cp      |
| ENSt6vectorI13kraus_channelEERKNS | p_api.html#_CPPv4N5cudaq15scalar_ |
| t6vectorINSt6size_tEEERKNSt6vecto | operatorplEdRR15scalar_operator), |
| rINSt6size_tEEERKNSt6vectorIdEE), |     [\[4\]](api/languages         |
|     [\[1\]](api/languages/cpp_a   | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| pi.html#_CPPv4NK5cudaq11noise_mod | alar_operatorplENSt7complexIdEE), |
| el12get_channelsERKNSt6stringERKN |     [\[5\]](api/languages/cpp     |
| St6vectorINSt6size_tEEERKNSt6vect | _api.html#_CPPv4NKR5cudaq15scalar |
| orINSt6size_tEEERKNSt6vectorIdEE) | _operatorplERK15scalar_operator), |
| -                                 |     [\[6\]]                       |
|  [cudaq::noise_model::noise_model | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatorplEd), |
|     function)](api                |     [\[7\]]                       |
| /languages/cpp_api.html#_CPPv4N5c | (api/languages/cpp_api.html#_CPPv |
| udaq11noise_model11noise_modelEv) | 4NKR5cudaq15scalar_operatorplEv), |
| -   [cu                           |     [\[8\]](api/language          |
| daq::noise_model::PredicateFuncTy | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatorplENSt7complexIdEE), |
|     type)](api/la                 |     [\[9\]](api/languages/cp      |
| nguages/cpp_api.html#_CPPv4N5cuda | p_api.html#_CPPv4NO5cudaq15scalar |
| q11noise_model15PredicateFuncTyE) | _operatorplERK15scalar_operator), |
| -   [cud                          |     [\[10\]                       |
| aq::noise_model::register_channel | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatorplEd), |
|     function)](api/languages      |     [\[11\                        |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | ]](api/languages/cpp_api.html#_CP |
| noise_model16register_channelEvv) | Pv4NO5cudaq15scalar_operatorplEv) |
| -   [cudaq::                      | -   [c                            |
| noise_model::requires_constructor | udaq::scalar_operator::operator+= |
|     (C++                          |     (C++                          |
|     type)](api/languages/cp       |     function)](api/languag        |
| p_api.html#_CPPv4I0DpEN5cudaq11no | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ise_model20requires_constructorE) | alar_operatorpLENSt7complexIdEE), |
| -   [cudaq::noise_model_type (C++ |     [\[1\]](api/languages/c       |
|     e                             | pp_api.html#_CPPv4N5cudaq15scalar |
| num)](api/languages/cpp_api.html# | _operatorpLERK15scalar_operator), |
| _CPPv4N5cudaq16noise_model_typeE) |     [\[2                          |
| -   [cudaq::no                    | \]](api/languages/cpp_api.html#_C |
| ise_model_type::amplitude_damping | PPv4N5cudaq15scalar_operatorpLEd) |
|     (C++                          | -   [                             |
|     enumerator)](api/languages    | cudaq::scalar_operator::operator- |
| /cpp_api.html#_CPPv4N5cudaq16nois |     (C++                          |
| e_model_type17amplitude_dampingE) |     function                      |
| -   [cudaq::noise_mode            | )](api/languages/cpp_api.html#_CP |
| l_type::amplitude_damping_channel | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     e                             |     [\[1\                         |
| numerator)](api/languages/cpp_api | ]](api/languages/cpp_api.html#_CP |
| .html#_CPPv4N5cudaq16noise_model_ | Pv4N5cudaq15scalar_operatormiENSt |
| type25amplitude_damping_channelE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::n                     |     [\[2\]](api/languages/cp      |
| oise_model_type::bit_flip_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRK15scalar_operator), |
|     enumerator)](api/language     |     [\[3\]](api/languages/cp      |
| s/cpp_api.html#_CPPv4N5cudaq16noi | p_api.html#_CPPv4N5cudaq15scalar_ |
| se_model_type16bit_flip_channelE) | operatormiEdRR15scalar_operator), |
| -   [cudaq::                      |     [\[4\]](api/languages         |
| noise_model_type::depolarization1 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     enumerator)](api/languag      |     [\[5\]](api/languages/cpp     |
| es/cpp_api.html#_CPPv4N5cudaq16no | _api.html#_CPPv4NKR5cudaq15scalar |
| ise_model_type15depolarization1E) | _operatormiERK15scalar_operator), |
| -   [cudaq::                      |     [\[6\]]                       |
| noise_model_type::depolarization2 | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEd), |
|     enumerator)](api/languag      |     [\[7\]]                       |
| es/cpp_api.html#_CPPv4N5cudaq16no | (api/languages/cpp_api.html#_CPPv |
| ise_model_type15depolarization2E) | 4NKR5cudaq15scalar_operatormiEv), |
| -   [cudaq::noise_m               |     [\[8\]](api/language          |
| odel_type::depolarization_channel | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|                                   |     [\[9\]](api/languages/cp      |
|   enumerator)](api/languages/cpp_ | p_api.html#_CPPv4NO5cudaq15scalar |
| api.html#_CPPv4N5cudaq16noise_mod | _operatormiERK15scalar_operator), |
| el_type22depolarization_channelE) |     [\[10\]                       |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|  [cudaq::noise_model_type::pauli1 | v4NO5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[11\                        |
|     enumerator)](a                | ]](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4NO5cudaq15scalar_operatormiEv) |
| 5cudaq16noise_model_type6pauli1E) | -   [c                            |
| -                                 | udaq::scalar_operator::operator-= |
|  [cudaq::noise_model_type::pauli2 |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](a                | es/cpp_api.html#_CPPv4N5cudaq15sc |
| pi/languages/cpp_api.html#_CPPv4N | alar_operatormIENSt7complexIdEE), |
| 5cudaq16noise_model_type6pauli2E) |     [\[1\]](api/languages/c       |
| -   [cudaq                        | pp_api.html#_CPPv4N5cudaq15scalar |
| ::noise_model_type::phase_damping | _operatormIERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     enumerator)](api/langu        | \]](api/languages/cpp_api.html#_C |
| ages/cpp_api.html#_CPPv4N5cudaq16 | PPv4N5cudaq15scalar_operatormIEd) |
| noise_model_type13phase_dampingE) | -   [                             |
| -   [cudaq::noi                   | cudaq::scalar_operator::operator/ |
| se_model_type::phase_flip_channel |     (C++                          |
|     (C++                          |     function                      |
|     enumerator)](api/languages/   | )](api/languages/cpp_api.html#_CP |
| cpp_api.html#_CPPv4N5cudaq16noise | Pv4N5cudaq15scalar_operatordvENSt |
| _model_type18phase_flip_channelE) | 7complexIdEERK15scalar_operator), |
| -                                 |     [\[1\                         |
| [cudaq::noise_model_type::unknown | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|     enumerator)](ap               | 7complexIdEERR15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[2\]](api/languages/cp      |
| cudaq16noise_model_type7unknownE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatordvEdRK15scalar_operator), |
| [cudaq::noise_model_type::x_error |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](ap               | operatordvEdRR15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[4\]](api/languages         |
| cudaq16noise_model_type7x_errorE) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -                                 | alar_operatordvENSt7complexIdEE), |
| [cudaq::noise_model_type::y_error |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|     enumerator)](ap               | _operatordvERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[6\]]                       |
| cudaq16noise_model_type7y_errorE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatordvEd), |
| [cudaq::noise_model_type::z_error |     [\[7\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     enumerator)](ap               | alar_operatordvENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[8\]](api/languages/cp      |
| cudaq16noise_model_type7z_errorE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::num_available_gpus    | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     function                      | ]](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4NO5cudaq15scalar_operatordvEd) |
| Pv4N5cudaq18num_available_gpusEv) | -   [c                            |
| -   [cudaq::observe (C++          | udaq::scalar_operator::operator/= |
|     function)]                    |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/languag        |
| 4I00DpEN5cudaq7observeENSt6vector | es/cpp_api.html#_CPPv4N5cudaq15sc |
| I14observe_resultEERR13QuantumKer | alar_operatordVENSt7complexIdEE), |
| nelRK15SpinOpContainerDpRR4Args), |     [\[1\]](api/languages/c       |
|     [\[1\]](api/languages/cpp_ap  | pp_api.html#_CPPv4N5cudaq15scalar |
| i.html#_CPPv4I0DpEN5cudaq7observe | _operatordVERK15scalar_operator), |
| E14observe_resultNSt6size_tERR13Q |     [\[2                          |
| uantumKernelRK7spin_opDpRR4Args), | \]](api/languages/cpp_api.html#_C |
|     [\[                           | PPv4N5cudaq15scalar_operatordVEd) |
| 2\]](api/languages/cpp_api.html#_ | -   [                             |
| CPPv4I0DpEN5cudaq7observeE14obser | cudaq::scalar_operator::operator= |
| ve_resultRK15observe_optionsRR13Q |     (C++                          |
| uantumKernelRK7spin_opDpRR4Args), |     function)](api/languages/c    |
|     [\[3\]](api/lang              | pp_api.html#_CPPv4N5cudaq15scalar |
| uages/cpp_api.html#_CPPv4I0DpEN5c | _operatoraSERK15scalar_operator), |
| udaq7observeE14observe_resultRR13 |     [\[1\]](api/languages/        |
| QuantumKernelRK7spin_opDpRR4Args) | cpp_api.html#_CPPv4N5cudaq15scala |
| -   [cudaq::observe_options (C++  | r_operatoraSERR15scalar_operator) |
|     st                            | -   [c                            |
| ruct)](api/languages/cpp_api.html | udaq::scalar_operator::operator== |
| #_CPPv4N5cudaq15observe_optionsE) |     (C++                          |
| -   [cudaq::observe_result (C++   |     function)](api/languages/c    |
|                                   | pp_api.html#_CPPv4NK5cudaq15scala |
| class)](api/languages/cpp_api.htm | r_operatoreqERK15scalar_operator) |
| l#_CPPv4N5cudaq14observe_resultE) | -   [cudaq:                       |
| -                                 | :scalar_operator::scalar_operator |
|    [cudaq::observe_result::counts |     (C++                          |
|     (C++                          |     func                          |
|     function)](api/languages/c    | tion)](api/languages/cpp_api.html |
| pp_api.html#_CPPv4N5cudaq14observ | #_CPPv4N5cudaq15scalar_operator15 |
| e_result6countsERK12spin_op_term) | scalar_operatorENSt7complexIdEE), |
| -   [cudaq::observe_result::dump  |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     function)                     | scalar_operator15scalar_operatorE |
| ](api/languages/cpp_api.html#_CPP | RK15scalar_callbackRRNSt13unorder |
| v4N5cudaq14observe_result4dumpEv) | ed_mapINSt6stringENSt6stringEEE), |
| -   [c                            |     [\[2\                         |
| udaq::observe_result::expectation | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operator15scal |
|                                   | ar_operatorERK15scalar_operator), |
| function)](api/languages/cpp_api. |     [\[3\]](api/langu             |
| html#_CPPv4N5cudaq14observe_resul | ages/cpp_api.html#_CPPv4N5cudaq15 |
| t11expectationERK12spin_op_term), | scalar_operator15scalar_operatorE |
|     [\[1\]](api/la                | RR15scalar_callbackRRNSt13unorder |
| nguages/cpp_api.html#_CPPv4N5cuda | ed_mapINSt6stringENSt6stringEEE), |
| q14observe_result11expectationEv) |     [\[4\                         |
| -   [cuda                         | ]](api/languages/cpp_api.html#_CP |
| q::observe_result::id_coefficient | Pv4N5cudaq15scalar_operator15scal |
|     (C++                          | ar_operatorERR15scalar_operator), |
|     function)](api/langu          |     [\[5\]](api/language          |
| ages/cpp_api.html#_CPPv4N5cudaq14 | s/cpp_api.html#_CPPv4N5cudaq15sca |
| observe_result14id_coefficientEv) | lar_operator15scalar_operatorEd), |
| -   [cuda                         |     [\[6\]](api/languag           |
| q::observe_result::observe_result | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operator15scalar_operatorEv) |
|                                   | -   [                             |
|   function)](api/languages/cpp_ap | cudaq::scalar_operator::to_matrix |
| i.html#_CPPv4N5cudaq14observe_res |     (C++                          |
| ult14observe_resultEdRK7spin_op), |                                   |
|     [\[1\]](a                     |   function)](api/languages/cpp_ap |
| pi/languages/cpp_api.html#_CPPv4N | i.html#_CPPv4NK5cudaq15scalar_ope |
| 5cudaq14observe_result14observe_r | rator9to_matrixERKNSt13unordered_ |
| esultEdRK7spin_op13sample_result) | mapINSt6stringENSt7complexIdEEEE) |
| -                                 | -   [                             |
|  [cudaq::observe_result::operator | cudaq::scalar_operator::to_string |
|     double (C++                   |     (C++                          |
|     functio                       |     function)](api/l              |
| n)](api/languages/cpp_api.html#_C | anguages/cpp_api.html#_CPPv4NK5cu |
| PPv4N5cudaq14observe_resultcvdEv) | daq15scalar_operator9to_stringEv) |
| -                                 | -   [cudaq::s                     |
|  [cudaq::observe_result::raw_data | calar_operator::\~scalar_operator |
|     (C++                          |     (C++                          |
|     function)](ap                 |     functio                       |
| i/languages/cpp_api.html#_CPPv4N5 | n)](api/languages/cpp_api.html#_C |
| cudaq14observe_result8raw_dataEv) | PPv4N5cudaq15scalar_operatorD0Ev) |
| -   [cudaq::operator_handler (C++ | -   [cudaq::set_noise (C++        |
|     cl                            |     function)](api/langu          |
| ass)](api/languages/cpp_api.html# | ages/cpp_api.html#_CPPv4N5cudaq9s |
| _CPPv4N5cudaq16operator_handlerE) | et_noiseERKN5cudaq11noise_modelE) |
| -   [cudaq::optimizable_function  | -   [cudaq::set_random_seed (C++  |
|     (C++                          |     function)](api/               |
|     class)                        | languages/cpp_api.html#_CPPv4N5cu |
| ](api/languages/cpp_api.html#_CPP | daq15set_random_seedENSt6size_tE) |
| v4N5cudaq20optimizable_functionE) | -   [cudaq::simulation_precision  |
| -   [cudaq::optimization_result   |     (C++                          |
|     (C++                          |     enum)                         |
|     type                          | ](api/languages/cpp_api.html#_CPP |
| )](api/languages/cpp_api.html#_CP | v4N5cudaq20simulation_precisionE) |
| Pv4N5cudaq19optimization_resultE) | -   [                             |
| -   [cudaq::optimizer (C++        | cudaq::simulation_precision::fp32 |
|     class)](api/languages/cpp_a   |     (C++                          |
| pi.html#_CPPv4N5cudaq9optimizerE) |     enumerator)](api              |
| -   [cudaq::optimizer::optimize   | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq20simulation_precision4fp32E) |
|                                   | -   [                             |
|  function)](api/languages/cpp_api | cudaq::simulation_precision::fp64 |
| .html#_CPPv4N5cudaq9optimizer8opt |     (C++                          |
| imizeEKiRR20optimizable_function) |     enumerator)](api              |
| -   [cu                           | /languages/cpp_api.html#_CPPv4N5c |
| daq::optimizer::requiresGradients | udaq20simulation_precision4fp64E) |
|     (C++                          | -   [cudaq::SimulationState (C++  |
|     function)](api/la             |     c                             |
| nguages/cpp_api.html#_CPPv4N5cuda | lass)](api/languages/cpp_api.html |
| q9optimizer17requiresGradientsEv) | #_CPPv4N5cudaq15SimulationStateE) |
| -   [cudaq::orca (C++             | -   [                             |
|     type)](api/languages/         | cudaq::SimulationState::precision |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     (C++                          |
| -   [cudaq::orca::sample (C++     |     enum)](api                    |
|     function)](api/languages/c    | /languages/cpp_api.html#_CPPv4N5c |
| pp_api.html#_CPPv4N5cudaq4orca6sa | udaq15SimulationState9precisionE) |
| mpleERNSt6vectorINSt6size_tEEERNS | -   [cudaq:                       |
| t6vectorINSt6size_tEEERNSt6vector | :SimulationState::precision::fp32 |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     (C++                          |
|     [\[1\]]                       |     enumerator)](api/lang         |
| (api/languages/cpp_api.html#_CPPv | uages/cpp_api.html#_CPPv4N5cudaq1 |
| 4N5cudaq4orca6sampleERNSt6vectorI | 5SimulationState9precision4fp32E) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [cudaq:                       |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | :SimulationState::precision::fp64 |
| -   [cudaq::orca::sample_async    |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
| function)](api/languages/cpp_api. | 5SimulationState9precision4fp64E) |
| html#_CPPv4N5cudaq4orca12sample_a | -                                 |
| syncERNSt6vectorINSt6size_tEEERNS |   [cudaq::SimulationState::Tensor |
| t6vectorINSt6size_tEEERNSt6vector |     (C++                          |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     struct)](                     |
|     [\[1\]](api/la                | api/languages/cpp_api.html#_CPPv4 |
| nguages/cpp_api.html#_CPPv4N5cuda | N5cudaq15SimulationState6TensorE) |
| q4orca12sample_asyncERNSt6vectorI | -   [cudaq::spin_handler (C++     |
| NSt6size_tEEERNSt6vectorINSt6size |                                   |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |   class)](api/languages/cpp_api.h |
| -   [cudaq::OrcaRemoteRESTQPU     | tml#_CPPv4N5cudaq12spin_handlerE) |
|     (C++                          | -   [cudaq:                       |
|     cla                           | :spin_handler::to_diagonal_matrix |
| ss)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |     function)](api/la             |
| -   [cudaq::pauli1 (C++           | nguages/cpp_api.html#_CPPv4NK5cud |
|     class)](api/languages/cp      | aq12spin_handler18to_diagonal_mat |
| p_api.html#_CPPv4N5cudaq6pauli1E) | rixERNSt13unordered_mapINSt6size_ |
| -                                 | tENSt7int64_tEEERKNSt13unordered_ |
|    [cudaq::pauli1::num_parameters | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -                                 |
|     member)]                      |   [cudaq::spin_handler::to_matrix |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq6pauli114num_parametersE) |     function                      |
| -   [cudaq::pauli1::num_targets   | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq12spin_handler9to_matri |
|     membe                         | xERKNSt6stringENSt7complexIdEEb), |
| r)](api/languages/cpp_api.html#_C |     [\[1                          |
| PPv4N5cudaq6pauli111num_targetsE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::pauli1::pauli1 (C++   | PPv4NK5cudaq12spin_handler9to_mat |
|     function)](api/languages/cpp_ | rixERNSt13unordered_mapINSt6size_ |
| api.html#_CPPv4N5cudaq6pauli16pau | tENSt7int64_tEEERKNSt13unordered_ |
| li1ERKNSt6vectorIN5cudaq4realEEE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::pauli2 (C++           | -   [cuda                         |
|     class)](api/languages/cp      | q::spin_handler::to_sparse_matrix |
| p_api.html#_CPPv4N5cudaq6pauli2E) |     (C++                          |
| -                                 |     function)](api/               |
|    [cudaq::pauli2::num_parameters | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq12spin_handler16to_sparse_matr |
|     member)]                      | ixERKNSt6stringENSt7complexIdEEb) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq6pauli214num_parametersE) |   [cudaq::spin_handler::to_string |
| -   [cudaq::pauli2::num_targets   |     (C++                          |
|     (C++                          |     function)](ap                 |
|     membe                         | i/languages/cpp_api.html#_CPPv4NK |
| r)](api/languages/cpp_api.html#_C | 5cudaq12spin_handler9to_stringEb) |
| PPv4N5cudaq6pauli211num_targetsE) | -                                 |
| -   [cudaq::pauli2::pauli2 (C++   |   [cudaq::spin_handler::unique_id |
|     function)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq6pauli26pau |     function)](ap                 |
| li2ERKNSt6vectorIN5cudaq4realEEE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::phase_damping (C++    | 5cudaq12spin_handler9unique_idEv) |
|                                   | -   [cudaq::spin_op (C++          |
|  class)](api/languages/cpp_api.ht |     type)](api/languages/cpp      |
| ml#_CPPv4N5cudaq13phase_dampingE) | _api.html#_CPPv4N5cudaq7spin_opE) |
| -   [cud                          | -   [cudaq::spin_op_term (C++     |
| aq::phase_damping::num_parameters |                                   |
|     (C++                          |    type)](api/languages/cpp_api.h |
|     member)](api/lan              | tml#_CPPv4N5cudaq12spin_op_termE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::state (C++            |
| 13phase_damping14num_parametersE) |     class)](api/languages/c       |
| -   [                             | pp_api.html#_CPPv4N5cudaq5stateE) |
| cudaq::phase_damping::num_targets | -   [cudaq::state::amplitude (C++ |
|     (C++                          |     function)](api/lang           |
|     member)](api/                 | uages/cpp_api.html#_CPPv4N5cudaq5 |
| languages/cpp_api.html#_CPPv4N5cu | state9amplitudeERKNSt6vectorIiEE) |
| daq13phase_damping11num_targetsE) | -   [cudaq::state::amplitudes     |
| -   [cudaq::phase_flip_channel    |     (C++                          |
|     (C++                          |     f                             |
|     clas                          | unction)](api/languages/cpp_api.h |
| s)](api/languages/cpp_api.html#_C | tml#_CPPv4N5cudaq5state10amplitud |
| PPv4N5cudaq18phase_flip_channelE) | esERKNSt6vectorINSt6vectorIiEEEE) |
| -   [cudaq::p                     | -   [cudaq::state::dump (C++      |
| hase_flip_channel::num_parameters |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     member)](api/language         | 5cudaq5state4dumpERNSt7ostreamE), |
| s/cpp_api.html#_CPPv4N5cudaq18pha |                                   |
| se_flip_channel14num_parametersE) |    [\[1\]](api/languages/cpp_api. |
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
