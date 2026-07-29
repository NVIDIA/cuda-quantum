::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-5007
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
| -   [cudaq::CusvState (C++        | ptsbe::ExhaustiveSamplingStrategy |
|                                   |     (C++                          |
|    class)](api/languages/cpp_api. |     class)](api/langua            |
| html#_CPPv4I0EN5cudaq9CusvStateE) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq::DefaultQPU (C++       | sbe26ExhaustiveSamplingStrategyE) |
|     class)](api/languages/cpp_api | -   [cudaq::ptsbe::               |
| .html#_CPPv4N5cudaq10DefaultQPUE) | ExhaustiveSamplingStrategy::clone |
| -   [cudaq::dem_from_kernel (C++  |     (C++                          |
|     function)](api                |     function)](api/languages/cpp_ |
| /languages/cpp_api.html#_CPPv4I0D | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| pEN5cudaq15dem_from_kernelENSt6st | haustiveSamplingStrategy5cloneEv) |
| ringERR13QuantumKernelDpRR4Args), | -   [cu                           |
|     [                             | daq::ptsbe::ExhaustiveSamplingStr |
| \[1\]](api/languages/cpp_api.html | ategy::ExhaustiveSamplingStrategy |
| #_CPPv4I0DpEN5cudaq15dem_from_ker |     (C++                          |
| nelENSt6stringERR13QuantumKernelP |     function)](api/la             |
| KN5cudaq11noise_modelEDpRR4Args), | nguages/cpp_api.html#_CPPv4N5cuda |
|     [\[2\]](api/languages/cp      | q5ptsbe26ExhaustiveSamplingStrate |
| p_api.html#_CPPv4I0DpEN5cudaq15de | gy26ExhaustiveSamplingStrategyEv) |
| m_from_kernelENSt6stringERR13Quan | -                                 |
| tumKernelPKN5cudaq11noise_modelER |    [cudaq::ptsbe::ExhaustiveSampl |
| KN5cudaq11dem_optionsEDpRR4Args), | ingStrategy::generateTrajectories |
|     [\[3\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4I0 |     function)](api/languag        |
| DpEN5cudaq15dem_from_kernelENSt6s | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| tringERR13QuantumKernelPKN5cudaq1 | sbe26ExhaustiveSamplingStrategy20 |
| 1noise_modelERKN5cudaq11dem_optio | generateTrajectoriesENSt4spanIKN6 |
| nsERN5cudaq15M2DSparseMatrixERN5c | detail10NoisePointEEENSt6size_tE) |
| udaq15M2OSparseMatrixEDpRR4Args), | -   [cudaq::ptsbe:                |
|     [\[4\]](api/language          | :ExhaustiveSamplingStrategy::name |
| s/cpp_api.html#_CPPv4I0DpEN5cudaq |     (C++                          |
| 15dem_from_kernelENSt6stringERR13 |     function)](api/languages/cpp  |
| QuantumKernelPKN5cudaq11noise_mod | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| elERN5cudaq15M2DSparseMatrixERN5c | xhaustiveSamplingStrategy4nameEv) |
| udaq15M2OSparseMatrixEDpRR4Args), | -   [cuda                         |
|     [\[5\]](api/languages/cpp_api | q::ptsbe::ExhaustiveSamplingStrat |
| .html#_CPPv4I0DpEN5cudaq15dem_fro | egy::\~ExhaustiveSamplingStrategy |
| m_kernelENSt6stringERR13QuantumKe |     (C++                          |
| rnelRN5cudaq15M2DSparseMatrixERN5 |     function)](api/languages      |
| cudaq15M2OSparseMatrixEDpRR4Args) | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| -   [cudaq::dem_options (C++      | 26ExhaustiveSamplingStrategyD0Ev) |
|                                   | -   [cuda                         |
|   struct)](api/languages/cpp_api. | q::ptsbe::OrderedSamplingStrategy |
| html#_CPPv4N5cudaq11dem_optionsE) |     (C++                          |
| -   [cudaq::d                     |     class)](api/lan               |
| em_options::allow_gauge_detectors | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 5ptsbe23OrderedSamplingStrategyE) |
|     member)](api/language         | -   [cudaq::ptsb                  |
| s/cpp_api.html#_CPPv4N5cudaq11dem | e::OrderedSamplingStrategy::clone |
| _options21allow_gauge_detectorsE) |     (C++                          |
| -   [cudaq::dem_options::appr     |     function)](api/languages/c    |
| oximate_disjoint_errors_threshold | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
|     (C++                          | 3OrderedSamplingStrategy5cloneEv) |
|     memb                          | -   [cudaq::ptsbe::OrderedSampl   |
| er)](api/languages/cpp_api.html#_ | ingStrategy::generateTrajectories |
| CPPv4N5cudaq11dem_options37approx |     (C++                          |
| imate_disjoint_errors_thresholdE) |     function)](api/lang           |
| -   [cuda                         | uages/cpp_api.html#_CPPv4NK5cudaq |
| q::dem_options::block_decompositi | 5ptsbe23OrderedSamplingStrategy20 |
| on_from_introducing_remnant_edges | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     member)](api/lang             | -   [cudaq::pts                   |
| uages/cpp_api.html#_CPPv4N5cudaq1 | be::OrderedSamplingStrategy::name |
| 1dem_options50block_decomposition |     (C++                          |
| _from_introducing_remnant_edgesE) |     function)](api/languages/     |
| -   [cud                          | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| aq::dem_options::decompose_errors | 23OrderedSamplingStrategy4nameEv) |
|     (C++                          | -                                 |
|     member)](api/lan              |    [cudaq::ptsbe::OrderedSampling |
| guages/cpp_api.html#_CPPv4N5cudaq | Strategy::OrderedSamplingStrategy |
| 11dem_options16decompose_errorsE) |     (C++                          |
| -                                 |     function)](                   |
|   [cudaq::dem_options::fold_loops | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5ptsbe23OrderedSamplingStr |
|     member)](a                    | ategy23OrderedSamplingStrategyEv) |
| pi/languages/cpp_api.html#_CPPv4N | -                                 |
| 5cudaq11dem_options10fold_loopsE) |  [cudaq::ptsbe::OrderedSamplingSt |
| -   [cudaq::dem_optio             | rategy::\~OrderedSamplingStrategy |
| ns::ignore_decomposition_failures |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     member)](api/languages/cpp_ap | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| i.html#_CPPv4N5cudaq11dem_options | sbe23OrderedSamplingStrategyD0Ev) |
| 29ignore_decomposition_failuresE) | -   [cudaq::pts                   |
| -   [cudaq::dem_opt               | be::ProbabilisticSamplingStrategy |
| ions::return_measurement_matrices |     (C++                          |
|     (C++                          |     class)](api/languages         |
|     member)](api/languages/cpp_   | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| api.html#_CPPv4N5cudaq11dem_optio | 29ProbabilisticSamplingStrategyE) |
| ns27return_measurement_matricesE) | -   [cudaq::ptsbe::Pro            |
| -   [cudaq::depolarization1 (C++  | babilisticSamplingStrategy::clone |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |                                   |
| #_CPPv4N5cudaq15depolarization1E) |  function)](api/languages/cpp_api |
| -   [cudaq::depolarization2 (C++  | .html#_CPPv4NK5cudaq5ptsbe29Proba |
|     c                             | bilisticSamplingStrategy5cloneEv) |
| lass)](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq15depolarization2E) | [cudaq::ptsbe::ProbabilisticSampl |
| -   [cudaq:                       | ingStrategy::generateTrajectories |
| :depolarization2::depolarization2 |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     function)](api/languages/cp   | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| p_api.html#_CPPv4N5cudaq15depolar | 29ProbabilisticSamplingStrategy20 |
| ization215depolarization2EK4real) | generateTrajectoriesENSt4spanIKN6 |
| -   [cudaq                        | detail10NoisePointEEENSt6size_tE) |
| ::depolarization2::num_parameters | -   [cudaq::ptsbe::Pr             |
|     (C++                          | obabilisticSamplingStrategy::name |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |                                   |
| depolarization214num_parametersE) |   function)](api/languages/cpp_ap |
| -   [cu                           | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| daq::depolarization2::num_targets | abilisticSamplingStrategy4nameEv) |
|     (C++                          | -   [cudaq::p                     |
|     member)](api/la               | tsbe::ProbabilisticSamplingStrate |
| nguages/cpp_api.html#_CPPv4N5cuda | gy::ProbabilisticSamplingStrategy |
| q15depolarization211num_targetsE) |     (C++                          |
| -                                 |     function)]                    |
|    [cudaq::depolarization_channel | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4N5cudaq5ptsbe29ProbabilisticSamp |
|     class)](                      | lingStrategy29ProbabilisticSampli |
| api/languages/cpp_api.html#_CPPv4 | ngStrategyENSt8optionalINSt8uint6 |
| N5cudaq22depolarization_channelE) | 4_tEEENSt8optionalINSt6size_tEEE) |
| -   [cudaq::depol                 | -   [cudaq::pts                   |
| arization_channel::num_parameters | be::ProbabilisticSamplingStrategy |
|     (C++                          | ::\~ProbabilisticSamplingStrategy |
|     member)](api/languages/cp     |     (C++                          |
| p_api.html#_CPPv4N5cudaq22depolar |     function)](api/languages/cp   |
| ization_channel14num_parametersE) | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| -   [cudaq::de                    | robabilisticSamplingStrategyD0Ev) |
| polarization_channel::num_targets | -                                 |
|     (C++                          | [cudaq::ptsbe::PTSBEExecutionData |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq22depo |     struct)](ap                   |
| larization_channel11num_targetsE) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [cudaq::detail (C++           | cudaq5ptsbe18PTSBEExecutionDataE) |
|     type)](api/languages/cp       | -   [cudaq::ptsbe::PTSBE          |
| p_api.html#_CPPv4N5cudaq6detailE) | ExecutionData::count_instructions |
| -   [cudaq::detail::future (C++   |     (C++                          |
|                                   |     function)](api/l              |
|   class)](api/languages/cpp_api.h | anguages/cpp_api.html#_CPPv4NK5cu |
| tml#_CPPv4N5cudaq6detail6futureE) | daq5ptsbe18PTSBEExecutionData18co |
| -                                 | unt_instructionsE20TraceInstructi |
|    [cudaq::detail::future::future | onTypeNSt8optionalINSt6stringEEE) |
|     (C++                          | -   [cudaq::ptsbe::P              |
|     functi                        | TSBEExecutionData::get_trajectory |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq6detail6future6future |     function                      |
| ERNSt6vectorI3JobEERNSt6stringERN | )](api/languages/cpp_api.html#_CP |
| St3mapINSt6stringENSt6stringEEE), | Pv4NK5cudaq5ptsbe18PTSBEExecution |
|     [\[1\]](api/lan               | Data14get_trajectoryENSt6size_tE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::ptsbe:                |
| 6detail6future6futureERR6future), | :PTSBEExecutionData::instructions |
|     [\[2\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     member)](api/languages/cp     |
| v4N5cudaq6detail6future6futureEv) | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| -   [c                            | TSBEExecutionData12instructionsE) |
| udaq::detail::kernel_builder_base | -   [cudaq::ptsbe:                |
|     (C++                          | :PTSBEExecutionData::trajectories |
|     class)](api/                  |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     member)](api/languages/cp     |
| daq6detail19kernel_builder_baseE) | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| -   [cudaq::detail::              | TSBEExecutionData12trajectoriesE) |
| kernel_builder_base::operator\<\< | -   [cudaq::ptsbe::PTSBEOptions   |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     struc                         |
| ages/cpp_api.html#_CPPv4N5cudaq6d | t)](api/languages/cpp_api.html#_C |
| etail19kernel_builder_baselsERNSt | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| 7ostreamERK19kernel_builder_base) | -   [cudaq::ptsbe::PTSB           |
| -                                 | EOptions::include_sequential_data |
| [cudaq::detail::KernelBuilderType |     (C++                          |
|     (C++                          |                                   |
|     class)](ap                    |    member)](api/languages/cpp_api |
| i/languages/cpp_api.html#_CPPv4N5 | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| cudaq6detail17KernelBuilderTypeE) | ptions23include_sequential_dataE) |
| -   [cudaq::                      | -   [cudaq::ptsb                  |
| detail::KernelBuilderType::create | e::PTSBEOptions::max_trajectories |
|     (C++                          |     (C++                          |
|     function                      |     member)](api/languages/       |
| )](api/languages/cpp_api.html#_CP | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| Pv4N5cudaq6detail17KernelBuilderT | 2PTSBEOptions16max_trajectoriesE) |
| ype6createEPN4mlir11MLIRContextE) | -   [cudaq::ptsbe::PT             |
| -   [cudaq::detail::Ker           | SBEOptions::return_execution_data |
| nelBuilderType::KernelBuilderType |     (C++                          |
|     (C++                          |     member)](api/languages/cpp_a  |
|     function)](api/lan            | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| guages/cpp_api.html#_CPPv4N5cudaq | EOptions21return_execution_dataE) |
| 6detail17KernelBuilderType17Kerne | -   [cudaq::pts                   |
| lBuilderTypeERRNSt8functionIFN4ml | be::PTSBEOptions::shot_allocation |
| ir4TypeEPN4mlir11MLIRContextEEEE) |     (C++                          |
| -   [cudaq::detector (C++         |     member)](api/languages        |
|     function)](api                | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| /languages/cpp_api.html#_CPPv4IDp | 12PTSBEOptions15shot_allocationE) |
| EN5cudaq8detectorEvDpRR8MeasArgs) | -   [cud                          |
| -   [cudaq::detectors (C++        | aq::ptsbe::PTSBEOptions::strategy |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq9detecto |     member)](api/l                |
| rsERKNSt6vectorI14measure_resultE | anguages/cpp_api.html#_CPPv4N5cud |
| ERKNSt6vectorI14measure_resultEE) | aq5ptsbe12PTSBEOptions8strategyE) |
| -   [cudaq::diag_matrix_callback  | -   [cudaq::ptsbe::PTSBETrace     |
|     (C++                          |     (C++                          |
|     class)                        |     t                             |
| ](api/languages/cpp_api.html#_CPP | ype)](api/languages/cpp_api.html# |
| v4N5cudaq20diag_matrix_callbackE) | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| -   [cudaq::dyn (C++              | -   [                             |
|     member)](api/languages        | cudaq::ptsbe::PTSSamplingStrategy |
| /cpp_api.html#_CPPv4N5cudaq3dynE) |     (C++                          |
| -   [cudaq::ExecutionContext (C++ |     class)](api                   |
|     cl                            | /languages/cpp_api.html#_CPPv4N5c |
| ass)](api/languages/cpp_api.html# | udaq5ptsbe19PTSSamplingStrategyE) |
| _CPPv4N5cudaq16ExecutionContextE) | -   [cudaq::                      |
| -   [c                            | ptsbe::PTSSamplingStrategy::clone |
| udaq::ExecutionContext::asyncExec |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     member)](api/                 | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| languages/cpp_api.html#_CPPv4N5cu | sbe19PTSSamplingStrategy5cloneEv) |
| daq16ExecutionContext9asyncExecE) | -   [cudaq::ptsbe::PTSSampl       |
| -   [cud                          | ingStrategy::generateTrajectories |
| aq::ExecutionContext::asyncResult |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](api/lan              | languages/cpp_api.html#_CPPv4NK5c |
| guages/cpp_api.html#_CPPv4N5cudaq | udaq5ptsbe19PTSSamplingStrategy20 |
| 16ExecutionContext11asyncResultE) | generateTrajectoriesENSt4spanIKN6 |
| -   [cudaq:                       | detail10NoisePointEEENSt6size_tE) |
| :ExecutionContext::batchIteration | -   [cudaq:                       |
|     (C++                          | :ptsbe::PTSSamplingStrategy::name |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     function)](api/langua         |
| xecutionContext14batchIterationE) | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| -   [cudaq::E                     | tsbe19PTSSamplingStrategy4nameEv) |
| xecutionContext::canHandleObserve | -   [cudaq::ptsbe::PTSSampli      |
|     (C++                          | ngStrategy::\~PTSSamplingStrategy |
|     member)](api/language         |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     function)](api/la             |
| cutionContext16canHandleObserveE) | nguages/cpp_api.html#_CPPv4N5cuda |
| -   [cudaq::Executio              | q5ptsbe19PTSSamplingStrategyD0Ev) |
| nContext::deferredKernelException | -   [cudaq::ptsbe::sample (C++    |
|     (C++                          |                                   |
|     member)](api/languages/cpp_a  |  function)](api/languages/cpp_api |
| pi.html#_CPPv4N5cudaq16ExecutionC | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| ontext23deferredKernelExceptionE) | mpleE13sample_resultRK14sample_op |
| -   [cudaq::E                     | tionsRR13QuantumKernelDpRR4Args), |
| xecutionContext::ExecutionContext |     [\[1\]](api                   |
|     (C++                          | /languages/cpp_api.html#_CPPv4I0D |
|     func                          | pEN5cudaq5ptsbe6sampleE13sample_r |
| tion)](api/languages/cpp_api.html | esultRKN5cudaq11noise_modelENSt6s |
| #_CPPv4N5cudaq16ExecutionContext1 | ize_tERR13QuantumKernelDpRR4Args) |
| 6ExecutionContextERKNSt6stringE), | -   [cudaq::ptsbe::sample_async   |
|     [\[1\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16Execu |     function)](a                  |
| tionContext16ExecutionContextERKN | pi/languages/cpp_api.html#_CPPv4I |
| St6stringENSt6size_tENSt6size_tE) | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| -   [cudaq::E                     | 9async_sample_resultRK14sample_op |
| xecutionContext::expectationValue | tionsRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\[1\]](api/languages/cp      |
|     member)](api/language         | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | be12sample_asyncE19async_sample_r |
| cutionContext16expectationValueE) | esultRKN5cudaq11noise_modelENSt6s |
| -   [cudaq::Execu                 | ize_tERR13QuantumKernelDpRR4Args) |
| tionContext::explicitMeasurements | -   [cudaq::ptsbe::sample_options |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     struct)                       |
| p_api.html#_CPPv4N5cudaq16Executi | ](api/languages/cpp_api.html#_CPP |
| onContext20explicitMeasurementsE) | v4N5cudaq5ptsbe14sample_optionsE) |
| -   [cuda                         | -   [cudaq::ptsbe::sample_result  |
| q::ExecutionContext::futureResult |     (C++                          |
|     (C++                          |     class                         |
|     member)](api/lang             | )](api/languages/cpp_api.html#_CP |
| uages/cpp_api.html#_CPPv4N5cudaq1 | Pv4N5cudaq5ptsbe13sample_resultE) |
| 6ExecutionContext12futureResultE) | -   [cudaq::pts                   |
| -   [cudaq::ExecutionContext      | be::sample_result::execution_data |
| ::hasConditionalsOnMeasureResults |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     mem                           | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| ber)](api/languages/cpp_api.html# | 3sample_result14execution_dataEv) |
| _CPPv4N5cudaq16ExecutionContext31 | -   [cudaq::ptsbe::               |
| hasConditionalsOnMeasureResultsE) | sample_result::has_execution_data |
| -   [cudaq:                       |     (C++                          |
| :ExecutionContext::inKernelLaunch |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     member)](api/langua           | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| ges/cpp_api.html#_CPPv4N5cudaq16E | ple_result18has_execution_dataEv) |
| xecutionContext14inKernelLaunchE) | -   [cudaq::pt                    |
| -   [cu                           | sbe::sample_result::sample_result |
| daq::ExecutionContext::kernelName |     (C++                          |
|     (C++                          |     function)](api/l              |
|     member)](api/la               | anguages/cpp_api.html#_CPPv4N5cud |
| nguages/cpp_api.html#_CPPv4N5cuda | aq5ptsbe13sample_result13sample_r |
| q16ExecutionContext10kernelNameE) | esultERRN5cudaq13sample_resultE), |
| -   [cud                          |                                   |
| aq::ExecutionContext::kernelTrace |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq5ptsbe13sample_re |
|     member)](api/lan              | sult13sample_resultERRN5cudaq13sa |
| guages/cpp_api.html#_CPPv4N5cudaq | mple_resultE18PTSBEExecutionData) |
| 16ExecutionContext11kernelTraceE) | -   [cudaq::ptsbe::               |
| -   [cudaq:                       | sample_result::set_execution_data |
| :ExecutionContext::msm_dimensions |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](api/langua           | languages/cpp_api.html#_CPPv4N5cu |
| ges/cpp_api.html#_CPPv4N5cudaq16E | daq5ptsbe13sample_result18set_exe |
| xecutionContext14msm_dimensionsE) | cution_dataE18PTSBEExecutionData) |
| -   [cudaq::                      | -   [cud                          |
| ExecutionContext::msm_prob_err_id | aq::ptsbe::ShotAllocationStrategy |
|     (C++                          |     (C++                          |
|     member)](api/languag          |     struct)](using                |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | /examples/ptsbe.html#_CPPv4N5cuda |
| ecutionContext15msm_prob_err_idE) | q5ptsbe22ShotAllocationStrategyE) |
| -   [cudaq::Ex                    | -   [cudaq::ptsbe::ShotAllocatio  |
| ecutionContext::msm_probabilities | nStrategy::ShotAllocationStrategy |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)                     |
| /cpp_api.html#_CPPv4N5cudaq16Exec | ](using/examples/ptsbe.html#_CPPv |
| utionContext17msm_probabilitiesE) | 4N5cudaq5ptsbe22ShotAllocationStr |
| -                                 | ategy22ShotAllocationStrategyE4Ty |
|    [cudaq::ExecutionContext::name | pedNSt8optionalINSt8uint64_tEEE), |
|     (C++                          |     [\[1\                         |
|     member)]                      | ]](using/examples/ptsbe.html#_CPP |
| (api/languages/cpp_api.html#_CPPv | v4N5cudaq5ptsbe22ShotAllocationSt |
| 4N5cudaq16ExecutionContext4nameE) | rategy22ShotAllocationStrategyEv) |
| -   [cu                           | -   [cudaq::pt                    |
| daq::ExecutionContext::noiseModel | sbe::ShotAllocationStrategy::Type |
|     (C++                          |     (C++                          |
|     member)](api/la               |     enum)](using/exam             |
| nguages/cpp_api.html#_CPPv4N5cuda | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| q16ExecutionContext10noiseModelE) | be22ShotAllocationStrategy4TypeE) |
| -   [cudaq::Exe                   | -   [cudaq::ptsbe::ShotAllocatio  |
| cutionContext::numberTrajectories | nStrategy::Type::HIGH_WEIGHT_BIAS |
|     (C++                          |     (C++                          |
|     member)](api/languages/       |     enumerat                      |
| cpp_api.html#_CPPv4N5cudaq16Execu | or)](using/examples/ptsbe.html#_C |
| tionContext18numberTrajectoriesE) | PPv4N5cudaq5ptsbe22ShotAllocation |
| -   [c                            | Strategy4Type16HIGH_WEIGHT_BIASE) |
| udaq::ExecutionContext::optResult | -   [cudaq::ptsbe::ShotAllocati   |
|     (C++                          | onStrategy::Type::LOW_WEIGHT_BIAS |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     enumera                       |
| daq16ExecutionContext9optResultE) | tor)](using/examples/ptsbe.html#_ |
| -                                 | CPPv4N5cudaq5ptsbe22ShotAllocatio |
|   [cudaq::ExecutionContext::qpuId | nStrategy4Type15LOW_WEIGHT_BIASE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAlloc      |
|     member)](                     | ationStrategy::Type::PROPORTIONAL |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5qpuIdE) |     enum                          |
| -   [cudaq                        | erator)](using/examples/ptsbe.htm |
| ::ExecutionContext::registerNames | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
|     (C++                          | tionStrategy4Type12PROPORTIONALE) |
|     member)](api/langu            | -   [cudaq::ptsbe::Shot           |
| ages/cpp_api.html#_CPPv4N5cudaq16 | AllocationStrategy::Type::UNIFORM |
| ExecutionContext13registerNamesE) |     (C++                          |
| -   [cu                           |                                   |
| daq::ExecutionContext::reorderIdx |   enumerator)](using/examples/pts |
|     (C++                          | be.html#_CPPv4N5cudaq5ptsbe22Shot |
|     member)](api/la               | AllocationStrategy4Type7UNIFORME) |
| nguages/cpp_api.html#_CPPv4N5cuda | -                                 |
| q16ExecutionContext10reorderIdxE) |   [cudaq::ptsbe::TraceInstruction |
| -                                 |     (C++                          |
|  [cudaq::ExecutionContext::result |     struct)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     member)](a                    | N5cudaq5ptsbe16TraceInstructionE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq:                       |
| 5cudaq16ExecutionContext6resultE) | :ptsbe::TraceInstruction::channel |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::shots |     member)](api/lang             |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     member)](                     | ptsbe16TraceInstruction7channelE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::                      |
| N5cudaq16ExecutionContext5shotsE) | ptsbe::TraceInstruction::controls |
| -   [cudaq::                      |     (C++                          |
| ExecutionContext::simulationState |     member)](api/langu            |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     member)](api/languag          | tsbe16TraceInstruction8controlsE) |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | -   [cud                          |
| ecutionContext15simulationStateE) | aq::ptsbe::TraceInstruction::name |
| -                                 |     (C++                          |
|    [cudaq::ExecutionContext::spin |     member)](api/l                |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     member)]                      | aq5ptsbe16TraceInstruction4nameE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq                        |
| 4N5cudaq16ExecutionContext4spinE) | ::ptsbe::TraceInstruction::params |
| -   [cudaq::                      |     (C++                          |
| ExecutionContext::totalIterations |     member)](api/lan              |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     member)](api/languag          | 5ptsbe16TraceInstruction6paramsE) |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | -   [cudaq:                       |
| ecutionContext15totalIterationsE) | :ptsbe::TraceInstruction::targets |
| -   [cudaq::ExecutionResult (C++  |     (C++                          |
|     st                            |     member)](api/lang             |
| ruct)](api/languages/cpp_api.html | uages/cpp_api.html#_CPPv4N5cudaq5 |
| #_CPPv4N5cudaq15ExecutionResultE) | ptsbe16TraceInstruction7targetsE) |
| -   [cud                          | -   [cudaq::ptsbe::T              |
| aq::ExecutionResult::appendResult | raceInstruction::TraceInstruction |
|     (C++                          |     (C++                          |
|     functio                       |                                   |
| n)](api/languages/cpp_api.html#_C |   function)](api/languages/cpp_ap |
| PPv4N5cudaq15ExecutionResult12app | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| endResultENSt6stringENSt6size_tE) | Instruction16TraceInstructionE20T |
| -   [cu                           | raceInstructionTypeNSt6stringENSt |
| daq::ExecutionResult::deserialize | 6vectorINSt6size_tEEENSt6vectorIN |
|     (C++                          | St6size_tEEENSt6vectorIdEENSt8opt |
|     function)                     | ionalIN5cudaq13kraus_channelEEE), |
| ](api/languages/cpp_api.html#_CPP |     [\[1\]](api/languages/cpp_a   |
| v4N5cudaq15ExecutionResult11deser | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| ializeERNSt6vectorINSt6size_tEEE) | eInstruction16TraceInstructionEv) |
| -   [cudaq:                       | -   [cud                          |
| :ExecutionResult::ExecutionResult | aq::ptsbe::TraceInstruction::type |
|     (C++                          |     (C++                          |
|     functio                       |     member)](api/l                |
| n)](api/languages/cpp_api.html#_C | anguages/cpp_api.html#_CPPv4N5cud |
| PPv4N5cudaq15ExecutionResult15Exe | aq5ptsbe16TraceInstruction4typeE) |
| cutionResultE16CountsDictionary), | -   [c                            |
|     [\[1\]](api/lan               | udaq::ptsbe::TraceInstructionType |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult15ExecutionResul |     enum)](api/                   |
| tE16CountsDictionaryNSt6stringE), | languages/cpp_api.html#_CPPv4N5cu |
|     [\[2\                         | daq5ptsbe20TraceInstructionTypeE) |
| ]](api/languages/cpp_api.html#_CP | -   [cudaq::                      |
| Pv4N5cudaq15ExecutionResult15Exec | ptsbe::TraceInstructionType::Gate |
| utionResultE16CountsDictionaryd), |     (C++                          |
|                                   |     enumerator)](api/langu        |
|    [\[3\]](api/languages/cpp_api. | ages/cpp_api.html#_CPPv4N5cudaq5p |
| html#_CPPv4N5cudaq15ExecutionResu | tsbe20TraceInstructionType4GateE) |
| lt15ExecutionResultENSt6stringE), | -   [cudaq::ptsbe::               |
|     [\[4\                         | TraceInstructionType::Measurement |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |                                   |
| utionResultERK15ExecutionResult), |    enumerator)](api/languages/cpp |
|     [\[5\]](api/language          | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | aceInstructionType11MeasurementE) |
| cutionResult15ExecutionResultEd), | -   [cudaq::p                     |
|     [\[6\]](api/languag           | tsbe::TraceInstructionType::Noise |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |     (C++                          |
| ecutionResult15ExecutionResultEv) |     enumerator)](api/langua       |
| -   [                             | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| cudaq::ExecutionResult::operator= | sbe20TraceInstructionType5NoiseE) |
|     (C++                          | -   [                             |
|     function)](api/languages/     | cudaq::ptsbe::TrajectoryPredicate |
| cpp_api.html#_CPPv4N5cudaq15Execu |     (C++                          |
| tionResultaSERK15ExecutionResult) |     type)](api                    |
| -   [c                            | /languages/cpp_api.html#_CPPv4N5c |
| udaq::ExecutionResult::operator== | udaq5ptsbe19TrajectoryPredicateE) |
|     (C++                          | -   [cudaq::QPU (C++              |
|     function)](api/languages/c    |     class)](api/languages         |
| pp_api.html#_CPPv4NK5cudaq15Execu | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| tionResulteqERK15ExecutionResult) | -   [cudaq::QPU::beginExecution   |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::registerName |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)](api/lan              | Pv4N5cudaq3QPU14beginExecutionEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cuda                         |
| 15ExecutionResult12registerNameE) | q::QPU::configureExecutionContext |
| -   [cudaq                        |     (C++                          |
| ::ExecutionResult::sequentialData |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     member)](api/langu            | _CPPv4NK5cudaq3QPU25configureExec |
| ages/cpp_api.html#_CPPv4N5cudaq15 | utionContextER16ExecutionContext) |
| ExecutionResult14sequentialDataE) | -   [cudaq::QPU::endExecution     |
| -   [                             |     (C++                          |
| cudaq::ExecutionResult::serialize |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     function)](api/l              | CPPv4N5cudaq3QPU12endExecutionEv) |
| anguages/cpp_api.html#_CPPv4NK5cu | -   [cudaq::QPU::enqueue (C++     |
| daq15ExecutionResult9serializeEv) |     function)](ap                 |
| -   [cudaq::fermion_handler (C++  | i/languages/cpp_api.html#_CPPv4N5 |
|     c                             | cudaq3QPU7enqueueER11QuantumTask) |
| lass)](api/languages/cpp_api.html | -   [cud                          |
| #_CPPv4N5cudaq15fermion_handlerE) | aq::QPU::finalizeExecutionContext |
| -   [cudaq::fermion_op (C++       |     (C++                          |
|     type)](api/languages/cpp_api  |     func                          |
| .html#_CPPv4N5cudaq10fermion_opE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::fermion_op_term (C++  | #_CPPv4NK5cudaq3QPU24finalizeExec |
|                                   | utionContextER16ExecutionContext) |
| type)](api/languages/cpp_api.html | -   [cudaq::QPU::getCompileTarget |
| #_CPPv4N5cudaq15fermion_op_termE) |     (C++                          |
| -   [cudaq::FermioniqQPU (C++     |     function)](api/languages/c    |
|                                   | pp_api.html#_CPPv4N5cudaq3QPU16ge |
|   class)](api/languages/cpp_api.h | tCompileTargetERK13sample_policy) |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | -   [cudaq::QPU::getConnectivity  |
| -   [cudaq::get_state (C++        |     (C++                          |
|                                   |     function)                     |
|    function)](api/languages/cpp_a | ](api/languages/cpp_api.html#_CPP |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | v4N5cudaq3QPU15getConnectivityEv) |
| ateEDaRR13QuantumKernelDpRR4Args) | -                                 |
| -   [cudaq::GPUEmulatedQPU (C++   | [cudaq::QPU::getExecutionThreadId |
|                                   |     (C++                          |
| class)](api/languages/cpp_api.htm |     function)](api/               |
| l#_CPPv4N5cudaq14GPUEmulatedQPUE) | languages/cpp_api.html#_CPPv4NK5c |
| -   [cudaq::gradient (C++         | udaq3QPU20getExecutionThreadIdEv) |
|     class)](api/languages/cpp_    | -   [cudaq::QPU::getNumQubits     |
| api.html#_CPPv4N5cudaq8gradientE) |     (C++                          |
| -   [cudaq::gradient::clone (C++  |     functi                        |
|     fun                           | on)](api/languages/cpp_api.html#_ |
| ction)](api/languages/cpp_api.htm | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| l#_CPPv4N5cudaq8gradient5cloneEv) | -   [                             |
| -   [cudaq::gradient::compute     | cudaq::QPU::getRemoteCapabilities |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api/l              |
| s/cpp_api.html#_CPPv4N5cudaq8grad | anguages/cpp_api.html#_CPPv4NK5cu |
| ient7computeERKNSt6vectorIdEERKNS | daq3QPU21getRemoteCapabilitiesEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -                                 |
|     [\[1\]](ap                    |  [cudaq::QPU::InKernelLaunchScope |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq8gradient7computeERKNSt6vect |     struct)](a                    |
| orIdEERNSt6vectorIdEERK7spin_opd) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::gradient::gradient    | 5cudaq3QPU19InKernelLaunchScopeE) |
|     (C++                          | -   [cudaq::QPU::isEmulated (C++  |
|     function)](api/lang           |     func                          |
| uages/cpp_api.html#_CPPv4I00EN5cu | tion)](api/languages/cpp_api.html |
| daq8gradient8gradientER7KernelT), | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|                                   | -   [cudaq::QPU::isSimulator (C++ |
|    [\[1\]](api/languages/cpp_api. |     funct                         |
| html#_CPPv4I00EN5cudaq8gradient8g | ion)](api/languages/cpp_api.html# |
| radientER7KernelTRR10ArgsMapper), | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     [\[2\                         | -   [cudaq::QPU::onRandomSeedSet  |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4I00EN5cudaq8gradient8gradientE |     function)](api/lang           |
| RR13QuantumKernelRR10ArgsMapper), | uages/cpp_api.html#_CPPv4N5cudaq3 |
|     [\[3                          | QPU15onRandomSeedSetENSt6size_tE) |
| \]](api/languages/cpp_api.html#_C | -   [cudaq::QPU::QPU (C++         |
| PPv4N5cudaq8gradient8gradientERRN |     functio                       |
| St8functionIFvNSt6vectorIdEEEEE), | n)](api/languages/cpp_api.html#_C |
|     [\[                           | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| 4\]](api/languages/cpp_api.html#_ |                                   |
| CPPv4N5cudaq8gradient8gradientEv) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::gradient::setArgs     | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     (C++                          |     [\[2\]](api/languages/cpp_    |
|     fu                            | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| nction)](api/languages/cpp_api.ht | -   [cudaq::QPU::setId (C++       |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     function                      |
| tArgsEvR13QuantumKernelDpRR4Args) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::gradient::setKernel   | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::setShots (C++    |
|     function)](api/languages/c    |     f                             |
| pp_api.html#_CPPv4I0EN5cudaq8grad | unction)](api/languages/cpp_api.h |
| ient9setKernelEvR13QuantumKernel) | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| -   [cud                          | -   [cudaq::                      |
| aq::gradients::central_difference | QPU::supportsExplicitMeasurements |
|     (C++                          |     (C++                          |
|     class)](api/la                |     function)](api/languag        |
| nguages/cpp_api.html#_CPPv4N5cuda | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| q9gradients18central_differenceE) | 28supportsExplicitMeasurementsEv) |
| -   [cudaq::gra                   | -   [cudaq::QPU::\~QPU (C++       |
| dients::central_difference::clone |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     function)](api/languages      | -   [cudaq::QPUState (C++         |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     class)](api/languages/cpp_    |
| ents18central_difference5cloneEv) | api.html#_CPPv4N5cudaq8QPUStateE) |
| -   [cudaq::gradi                 | -   [cudaq::qreg (C++             |
| ents::central_difference::compute |     class)](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4I_NSt6s |
|     function)](                   | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qreg::back (C++       |
| N5cudaq9gradients18central_differ |     function)                     |
| ence7computeERKNSt6vectorIdEERKNS | ](api/languages/cpp_api.html#_CPP |
| t8functionIFdNSt6vectorIdEEEEEd), | v4N5cudaq4qreg4backENSt6size_tE), |
|                                   |     [\[1\]](api/languages/cpp_ap  |
|   [\[1\]](api/languages/cpp_api.h | i.html#_CPPv4N5cudaq4qreg4backEv) |
| tml#_CPPv4N5cudaq9gradients18cent | -   [cudaq::qreg::begin (C++      |
| ral_difference7computeERKNSt6vect |                                   |
| orIdEERNSt6vectorIdEERK7spin_opd) |  function)](api/languages/cpp_api |
| -   [cudaq::gradie                | .html#_CPPv4N5cudaq4qreg5beginEv) |
| nts::central_difference::gradient | -   [cudaq::qreg::clear (C++      |
|     (C++                          |                                   |
|     functio                       |  function)](api/languages/cpp_api |
| n)](api/languages/cpp_api.html#_C | .html#_CPPv4N5cudaq4qreg5clearEv) |
| PPv4I00EN5cudaq9gradients18centra | -   [cudaq::qreg::front (C++      |
| l_difference8gradientER7KernelT), |     function)]                    |
|     [\[1\]](api/langua            | (api/languages/cpp_api.html#_CPPv |
| ges/cpp_api.html#_CPPv4I00EN5cuda | 4N5cudaq4qreg5frontENSt6size_tE), |
| q9gradients18central_difference8g |     [\[1\]](api/languages/cpp_api |
| radientER7KernelTRR10ArgsMapper), | .html#_CPPv4N5cudaq4qreg5frontEv) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::qreg::operator\[\]    |
| api.html#_CPPv4I00EN5cudaq9gradie |     (C++                          |
| nts18central_difference8gradientE |     functi                        |
| RR13QuantumKernelRR10ArgsMapper), | on)](api/languages/cpp_api.html#_ |
|     [\[3\]](api/languages/cpp     | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::qreg::qreg (C++       |
| 18central_difference8gradientERRN |     function)                     |
| St8functionIFvNSt6vectorIdEEEEE), | ](api/languages/cpp_api.html#_CPP |
|     [\[4\]](api/languages/cp      | v4N5cudaq4qreg4qregENSt6size_tE), |
| p_api.html#_CPPv4N5cudaq9gradient |     [\[1\]](api/languages/cpp_ap  |
| s18central_difference8gradientEv) | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| -   [cud                          | -   [cudaq::qreg::size (C++       |
| aq::gradients::forward_difference |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     class)](api/la                | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::qreg::slice (C++      |
| q9gradients18forward_differenceE) |     function)](api/langu          |
| -   [cudaq::gra                   | ages/cpp_api.html#_CPPv4N5cudaq4q |
| dients::forward_difference::clone | reg5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qreg::value_type (C++ |
|     function)](api/languages      |                                   |
| /cpp_api.html#_CPPv4N5cudaq9gradi | type)](api/languages/cpp_api.html |
| ents18forward_difference5cloneEv) | #_CPPv4N5cudaq4qreg10value_typeE) |
| -   [cudaq::gradi                 | -   [cudaq::qspan (C++            |
| ents::forward_difference::compute |     class)](api/lang              |
|     (C++                          | uages/cpp_api.html#_CPPv4I_NSt6si |
|     function)](                   | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::QuakeValue (C++       |
| N5cudaq9gradients18forward_differ |     class)](api/languages/cpp_api |
| ence7computeERKNSt6vectorIdEERKNS | .html#_CPPv4N5cudaq10QuakeValueE) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::Q                     |
|                                   | uakeValue::canValidateNumElements |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq9gradients18forw |     function)](api/languages      |
| ard_difference7computeERKNSt6vect | /cpp_api.html#_CPPv4N5cudaq10Quak |
| orIdEERNSt6vectorIdEERK7spin_opd) | eValue22canValidateNumElementsEv) |
| -   [cudaq::gradie                | -                                 |
| nts::forward_difference::gradient |  [cudaq::QuakeValue::constantSize |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api                |
| n)](api/languages/cpp_api.html#_C | /languages/cpp_api.html#_CPPv4N5c |
| PPv4I00EN5cudaq9gradients18forwar | udaq10QuakeValue12constantSizeEv) |
| d_difference8gradientER7KernelT), | -   [cudaq::QuakeValue::dump (C++ |
|     [\[1\]](api/langua            |     function)](api/lan            |
| ges/cpp_api.html#_CPPv4I00EN5cuda | guages/cpp_api.html#_CPPv4N5cudaq |
| q9gradients18forward_difference8g | 10QuakeValue4dumpERNSt7ostreamE), |
| radientER7KernelTRR10ArgsMapper), |     [\                            |
|     [\[2\]](api/languages/cpp_    | [1\]](api/languages/cpp_api.html# |
| api.html#_CPPv4I00EN5cudaq9gradie | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| nts18forward_difference8gradientE | -   [cudaq                        |
| RR13QuantumKernelRR10ArgsMapper), | ::QuakeValue::getRequiredElements |
|     [\[3\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4N5cudaq9gradients |     function)](api/langua         |
| 18forward_difference8gradientERRN | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| St8functionIFvNSt6vectorIdEEEEE), | uakeValue19getRequiredElementsEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::QuakeValue::getValue  |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18forward_difference8gradientEv) |     function)]                    |
| -   [                             | (api/languages/cpp_api.html#_CPPv |
| cudaq::gradients::parameter_shift | 4NK5cudaq10QuakeValue8getValueEv) |
|     (C++                          | -   [cudaq::QuakeValue::inverse   |
|     class)](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)                     |
| udaq9gradients15parameter_shiftE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::                      | v4NK5cudaq10QuakeValue7inverseEv) |
| gradients::parameter_shift::clone | -   [cudaq::QuakeValue::isStdVec  |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)                     |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | ](api/languages/cpp_api.html#_CPP |
| adients15parameter_shift5cloneEv) | v4N5cudaq10QuakeValue8isStdVecEv) |
| -   [cudaq::gr                    | -                                 |
| adients::parameter_shift::compute |    [cudaq::QuakeValue::operator\* |
|     (C++                          |     (C++                          |
|     function                      |     function)](api                |
| )](api/languages/cpp_api.html#_CP | /languages/cpp_api.html#_CPPv4N5c |
| Pv4N5cudaq9gradients15parameter_s | udaq10QuakeValuemlE10QuakeValue), |
| hift7computeERKNSt6vectorIdEERKNS |                                   |
| t8functionIFdNSt6vectorIdEEEEEd), | [\[1\]](api/languages/cpp_api.htm |
|     [\[1\]](api/languages/cpp_ap  | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| i.html#_CPPv4N5cudaq9gradients15p | -   [cudaq::QuakeValue::operator+ |
| arameter_shift7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |     function)](api                |
| -   [cudaq::gra                   | /languages/cpp_api.html#_CPPv4N5c |
| dients::parameter_shift::gradient | udaq10QuakeValueplE10QuakeValue), |
|     (C++                          |     [                             |
|     func                          | \[1\]](api/languages/cpp_api.html |
| tion)](api/languages/cpp_api.html | #_CPPv4N5cudaq10QuakeValueplEKd), |
| #_CPPv4I00EN5cudaq9gradients15par |                                   |
| ameter_shift8gradientER7KernelT), | [\[2\]](api/languages/cpp_api.htm |
|     [\[1\]](api/lan               | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| guages/cpp_api.html#_CPPv4I00EN5c | -   [cudaq::QuakeValue::operator- |
| udaq9gradients15parameter_shift8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api                |
|     [\[2\]](api/languages/c       | /languages/cpp_api.html#_CPPv4N5c |
| pp_api.html#_CPPv4I00EN5cudaq9gra | udaq10QuakeValuemiE10QuakeValue), |
| dients15parameter_shift8gradientE |     [                             |
| RR13QuantumKernelRR10ArgsMapper), | \[1\]](api/languages/cpp_api.html |
|     [\[3\]](api/languages/        | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| cpp_api.html#_CPPv4N5cudaq9gradie |     [                             |
| nts15parameter_shift8gradientERRN | \[2\]](api/languages/cpp_api.html |
| St8functionIFvNSt6vectorIdEEEEE), | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     [\[4\]](api/languages         |                                   |
| /cpp_api.html#_CPPv4N5cudaq9gradi | [\[3\]](api/languages/cpp_api.htm |
| ents15parameter_shift8gradientEv) | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| -   [cudaq::kernel_builder (C++   | -   [cudaq::QuakeValue::operator/ |
|     clas                          |     (C++                          |
| s)](api/languages/cpp_api.html#_C |     function)](api                |
| PPv4IDpEN5cudaq14kernel_builderE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [c                            | udaq10QuakeValuedvE10QuakeValue), |
| udaq::kernel_builder::constantVal |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     function)](api/la             | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| nguages/cpp_api.html#_CPPv4N5cuda | -                                 |
| q14kernel_builder11constantValEd) |  [cudaq::QuakeValue::operator\[\] |
| -                                 |     (C++                          |
|  [cudaq::kernel_builder::detector |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|                                   | udaq10QuakeValueixEKNSt6size_tE), |
|    function)](api/languages/cpp_a |     [\[1\]](api/                  |
| pi.html#_CPPv4IDpEN5cudaq14kernel | languages/cpp_api.html#_CPPv4N5cu |
| _builder8detectorEvDpRR8MeasArgs) | daq10QuakeValueixERK10QuakeValue) |
| -                                 | -                                 |
| [cudaq::kernel_builder::detectors |    [cudaq::QuakeValue::QuakeValue |
|     (C++                          |     (C++                          |
|     func                          |     function)](api/languag        |
| tion)](api/languages/cpp_api.html | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| #_CPPv4N5cudaq14kernel_builder9de | akeValue10QuakeValueERN4mlir20Imp |
| tectorsE10QuakeValue10QuakeValue) | licitLocOpBuilderEN4mlir5ValueE), |
| -   [cu                           |     [\[1\]                        |
| daq::kernel_builder::getArguments | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq10QuakeValue10QuakeValue |
|     function)](api/lan            | ERN4mlir20ImplicitLocOpBuilderEd) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::QuakeValue::size (C++ |
| 14kernel_builder12getArgumentsEv) |     funct                         |
| -   [cu                           | ion)](api/languages/cpp_api.html# |
| daq::kernel_builder::getNumParams | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|     (C++                          | -   [cudaq::QuakeValue::slice     |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languages/cpp_ |
| 14kernel_builder12getNumParamsEv) | api.html#_CPPv4N5cudaq10QuakeValu |
| -   [c                            | e5sliceEKNSt6size_tEKNSt6size_tE) |
| udaq::kernel_builder::isArgStdVec | -   [cudaq::quantum_platform (C++ |
|     (C++                          |     cl                            |
|     function)](api/languages/cp   | ass)](api/languages/cpp_api.html# |
| p_api.html#_CPPv4N5cudaq14kernel_ | _CPPv4N5cudaq16quantum_platformE) |
| builder11isArgStdVecENSt6size_tE) | -   [cudaq:                       |
| -   [cuda                         | :quantum_platform::beginExecution |
| q::kernel_builder::kernel_builder |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/languages/cpp  | es/cpp_api.html#_CPPv4N5cudaq16qu |
| _api.html#_CPPv4N5cudaq14kernel_b | antum_platform14beginExecutionEv) |
| uilder14kernel_builderERNSt6vecto | -   [cudaq::quantum_pl            |
| rIN6detail17KernelBuilderTypeEEE) | atform::configureExecutionContext |
| -   [cudaq::k                     |     (C++                          |
| ernel_builder::logical_observable |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4NK5cudaq |
|     function)                     | 16quantum_platform25configureExec |
| ](api/languages/cpp_api.html#_CPP | utionContextER16ExecutionContext) |
| v4IDpEN5cudaq14kernel_builder18lo | -   [cuda                         |
| gical_observableEvDpRR8MeasArgs), | q::quantum_platform::connectivity |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/langu          |
| cudaq14kernel_builder18logical_ob | ages/cpp_api.html#_CPPv4N5cudaq16 |
| servableE10QuakeValueNSt6size_tE) | quantum_platform12connectivityEv) |
| -   [cudaq::kernel_builder::name  | -   [cuda                         |
|     (C++                          | q::quantum_platform::endExecution |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/langu          |
| v4N5cudaq14kernel_builder4nameEv) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -                                 | quantum_platform12endExecutionEv) |
|    [cudaq::kernel_builder::qalloc | -   [cudaq::q                     |
|     (C++                          | uantum_platform::enqueueAsyncTask |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     function)](api/languages/     |
| nel_builder6qallocE10QuakeValue), | cpp_api.html#_CPPv4N5cudaq16quant |
|     [\[1\]](api/language          | um_platform16enqueueAsyncTaskEKNS |
| s/cpp_api.html#_CPPv4N5cudaq14ker | t6size_tER19KernelExecutionTask), |
| nel_builder6qallocEKNSt6size_tE), |     [\[1\]](api/languag           |
|     [\[2                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
| \]](api/languages/cpp_api.html#_C | antum_platform16enqueueAsyncTaskE |
| PPv4N5cudaq14kernel_builder6qallo | KNSt6size_tERNSt8functionIFvvEEE) |
| cERNSt6vectorINSt7complexIdEEEE), | -   [cudaq::quantum_p             |
|     [\[3\]](                      | latform::finalizeExecutionContext |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14kernel_builder6qallocEv) |     function)](api/languages/c    |
| -   [cudaq::kernel_builder::swap  | pp_api.html#_CPPv4NK5cudaq16quant |
|     (C++                          | um_platform24finalizeExecutionCon |
|     function)](api/language       | textERN5cudaq16ExecutionContextE) |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | -   [cudaq::qua                   |
| 4kernel_builder4swapEvRK10QuakeVa | ntum_platform::get_codegen_config |
| lueRK10QuakeValueRK10QuakeValue), |     (C++                          |
|                                   |     function)](api/languages/c    |
| [\[1\]](api/languages/cpp_api.htm | pp_api.html#_CPPv4N5cudaq16quantu |
| l#_CPPv4I00EN5cudaq14kernel_build | m_platform18get_codegen_configEv) |
| er4swapEvRKNSt6vectorI10QuakeValu | -   [cuda                         |
| eEERK10QuakeValueRK10QuakeValue), | q::quantum_platform::get_exec_ctx |
|                                   |     (C++                          |
| [\[2\]](api/languages/cpp_api.htm |     function)](api/langua         |
| l#_CPPv4N5cudaq14kernel_builder4s | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| wapERK10QuakeValueRK10QuakeValue) | quantum_platform12get_exec_ctxEv) |
| -   [cudaq::KernelExecutionTask   | -   [c                            |
|     (C++                          | udaq::quantum_platform::get_noise |
|     type                          |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/languages/c    |
| Pv4N5cudaq19KernelExecutionTaskE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [cudaq::KernelThunkResultType | m_platform9get_noiseENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     struct)]                      | :quantum_platform::get_num_qubits |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq21KernelThunkResultTypeE) |                                   |
| -   [cudaq::KernelThunkType (C++  | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq16quantum_plat |
| type)](api/languages/cpp_api.html | form14get_num_qubitsENSt6size_tE) |
| #_CPPv4N5cudaq15KernelThunkTypeE) | -   [cudaq::quantum_              |
| -   [cudaq::kraus_channel (C++    | platform::get_remote_capabilities |
|                                   |     (C++                          |
|  class)](api/languages/cpp_api.ht |     function)                     |
| ml#_CPPv4N5cudaq13kraus_channelE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus_channel::empty  | v4NK5cudaq16quantum_platform23get |
|     (C++                          | _remote_capabilitiesENSt6size_tE) |
|     function)]                    | -   [cudaq::qua                   |
| (api/languages/cpp_api.html#_CPPv | ntum_platform::get_runtime_target |
| 4NK5cudaq13kraus_channel5emptyEv) |     (C++                          |
| -   [cudaq::kraus_c               |     function)](api/languages/cp   |
| hannel::generateUnitaryParameters | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform18get_runtime_targetEv) |
|                                   | -   [cud                          |
|    function)](api/languages/cpp_a | aq::quantum_platform::is_emulated |
| pi.html#_CPPv4N5cudaq13kraus_chan |     (C++                          |
| nel25generateUnitaryParametersEv) |                                   |
| -                                 |    function)](api/languages/cpp_a |
|    [cudaq::kraus_channel::get_ops | pi.html#_CPPv4NK5cudaq16quantum_p |
|     (C++                          | latform11is_emulatedENSt6size_tE) |
|     function)](a                  | -   [cudaq::                      |
| pi/languages/cpp_api.html#_CPPv4N | quantum_platform::is_library_mode |
| K5cudaq13kraus_channel7get_opsEv) |     (C++                          |
| -   [cud                          |     function)](api/languages      |
| aq::kraus_channel::identity_flags | /cpp_api.html#_CPPv4NK5cudaq16qua |
|     (C++                          | ntum_platform15is_library_modeEv) |
|     member)](api/lan              | -   [c                            |
| guages/cpp_api.html#_CPPv4N5cudaq | udaq::quantum_platform::is_remote |
| 13kraus_channel14identity_flagsE) |     (C++                          |
| -   [cud                          |     function)](api/languages/cp   |
| aq::kraus_channel::is_identity_op | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform9is_remoteENSt6size_tE) |
|                                   | -   [cuda                         |
|    function)](api/languages/cpp_a | q::quantum_platform::is_simulator |
| pi.html#_CPPv4NK5cudaq13kraus_cha |     (C++                          |
| nnel14is_identity_opENSt6size_tE) |                                   |
| -   [cudaq::                      |   function)](api/languages/cpp_ap |
| kraus_channel::is_unitary_mixture | i.html#_CPPv4NK5cudaq16quantum_pl |
|     (C++                          | atform12is_simulatorENSt6size_tE) |
|     function)](api/languages      | -   [c                            |
| /cpp_api.html#_CPPv4NK5cudaq13kra | udaq::quantum_platform::launchVQE |
| us_channel18is_unitary_mixtureEv) |     (C++                          |
| -   [cu                           |     function)](                   |
| daq::kraus_channel::kraus_channel | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq16quantum_platform9launchV |
|     function)](api/lang           | QEEKNSt6stringEPKvPN5cudaq8gradie |
| uages/cpp_api.html#_CPPv4IDpEN5cu | ntERKN5cudaq7spin_opERN5cudaq9opt |
| daq13kraus_channel13kraus_channel | imizerEKiKNSt6size_tENSt6size_tE) |
| EDpRRNSt16initializer_listI1TEE), | -   [cudaq:                       |
|                                   | :quantum_platform::list_platforms |
|  [\[1\]](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13kraus_channel13 |     function)](api/languag        |
| kraus_channelERK13kraus_channel), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[2\]                        | antum_platform14list_platformsEv) |
| ](api/languages/cpp_api.html#_CPP | -                                 |
| v4N5cudaq13kraus_channel13kraus_c |    [cudaq::quantum_platform::name |
| hannelERKNSt6vectorI8kraus_opEE), |     (C++                          |
|     [\[3\]                        |     function)](a                  |
| ](api/languages/cpp_api.html#_CPP | pi/languages/cpp_api.html#_CPPv4N |
| v4N5cudaq13kraus_channel13kraus_c | K5cudaq16quantum_platform4nameEv) |
| hannelERRNSt6vectorI8kraus_opEE), | -   [                             |
|     [\[4\]](api/lan               | cudaq::quantum_platform::num_qpus |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 13kraus_channel13kraus_channelEv) |     function)](api/l              |
| -                                 | anguages/cpp_api.html#_CPPv4NK5cu |
| [cudaq::kraus_channel::noise_type | daq16quantum_platform8num_qpusEv) |
|     (C++                          | -   [cudaq::                      |
|     member)](api                  | quantum_platform::onRandomSeedSet |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel10noise_typeE) |                                   |
| -                                 | function)](api/languages/cpp_api. |
|   [cudaq::kraus_channel::op_names | html#_CPPv4N5cudaq16quantum_platf |
|     (C++                          | orm15onRandomSeedSetENSt6size_tE) |
|     member)](                     | -   [cudaq:                       |
| api/languages/cpp_api.html#_CPPv4 | :quantum_platform::reset_exec_ctx |
| N5cudaq13kraus_channel8op_namesE) |     (C++                          |
| -                                 |     function)](api/languag        |
|  [cudaq::kraus_channel::operator= | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14reset_exec_ctxEv) |
|     function)](api/langua         | -   [cud                          |
| ges/cpp_api.html#_CPPv4N5cudaq13k | aq::quantum_platform::reset_noise |
| raus_channelaSERK13kraus_channel) |     (C++                          |
| -   [c                            |     function)](api/languages/cpp_ |
| udaq::kraus_channel::operator\[\] | api.html#_CPPv4N5cudaq16quantum_p |
|     (C++                          | latform11reset_noiseENSt6size_tE) |
|     function)](api/l              | -   [cuda                         |
| anguages/cpp_api.html#_CPPv4N5cud | q::quantum_platform::set_exec_ctx |
| aq13kraus_channelixEKNSt6size_tE) |     (C++                          |
| -                                 |     funct                         |
| [cudaq::kraus_channel::parameters | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq16quantum_platform12 |
|     member)](api                  | set_exec_ctxEP16ExecutionContext) |
| /languages/cpp_api.html#_CPPv4N5c | -   [c                            |
| udaq13kraus_channel10parametersE) | udaq::quantum_platform::set_noise |
| -   [cudaq::krau                  |     (C++                          |
| s_channel::populateDefaultOpNames |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function)](api/languages/cp   | Pv4N5cudaq16quantum_platform9set_ |
| p_api.html#_CPPv4N5cudaq13kraus_c | noiseEPK11noise_modelNSt6size_tE) |
| hannel22populateDefaultOpNamesEv) | -   [cudaq::quantum_platfor       |
| -   [cu                           | m::supports_explicit_measurements |
| daq::kraus_channel::probabilities |     (C++                          |
|     (C++                          |     function)](api/l              |
|     member)](api/la               | anguages/cpp_api.html#_CPPv4NK5cu |
| nguages/cpp_api.html#_CPPv4N5cuda | daq16quantum_platform30supports_e |
| q13kraus_channel13probabilitiesE) | xplicit_measurementsENSt6size_tE) |
| -                                 | -   [cudaq::quantum_pla           |
|  [cudaq::kraus_channel::push_back | tform::supports_task_distribution |
|     (C++                          |     (C++                          |
|     function)](api                |     fu                            |
| /languages/cpp_api.html#_CPPv4N5c | nction)](api/languages/cpp_api.ht |
| udaq13kraus_channel9push_backE8kr | ml#_CPPv4NK5cudaq16quantum_platfo |
| aus_opNSt8optionalINSt6stringEEE) | rm26supports_task_distributionEv) |
| -   [cudaq::kraus_channel::size   | -   [cudaq::quantum               |
|     (C++                          | _platform::with_execution_context |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)                     |
| v4NK5cudaq13kraus_channel4sizeEv) | ](api/languages/cpp_api.html#_CPP |
| -   [                             | v4I0DpEN5cudaq16quantum_platform2 |
| cudaq::kraus_channel::unitary_ops | 2with_execution_contextEDaR16Exec |
|     (C++                          | utionContextRR8CallableDpRR4Args) |
|     member)](api/                 | -   [cudaq::QuantumTask (C++      |
| languages/cpp_api.html#_CPPv4N5cu |     type)](api/languages/cpp_api. |
| daq13kraus_channel11unitary_opsE) | html#_CPPv4N5cudaq11QuantumTaskE) |
| -   [cudaq::kraus_op (C++         | -   [cudaq::qubit (C++            |
|     struct)](api/languages/cpp_   |     type)](api/languages/c        |
| api.html#_CPPv4N5cudaq8kraus_opE) | pp_api.html#_CPPv4N5cudaq5qubitE) |
| -   [cudaq::kraus_op::adjoint     | -   [cudaq::QubitConnectivity     |
|     (C++                          |     (C++                          |
|     functi                        |     ty                            |
| on)](api/languages/cpp_api.html#_ | pe)](api/languages/cpp_api.html#_ |
| CPPv4NK5cudaq8kraus_op7adjointEv) | CPPv4N5cudaq17QubitConnectivityE) |
| -   [cudaq::kraus_op::data (C++   | -   [cudaq::QubitEdge (C++        |
|                                   |     type)](api/languages/cpp_a    |
|  member)](api/languages/cpp_api.h | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | -   [cudaq::qudit (C++            |
| -   [cudaq::kraus_op::kraus_op    |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     func                          | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| tion)](api/languages/cpp_api.html | -   [cudaq::qudit::qudit (C++     |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |                                   |
| opERRNSt16initializer_listI1TEE), | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4N5cudaq5qudit5quditEv) |
|  [\[1\]](api/languages/cpp_api.ht | -   [cudaq::QuEraRemoteRESTQPU    |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o |     (C++                          |
| pENSt6vectorIN5cudaq7complexEEE), |     clas                          |
|     [\[2\]](api/l                 | s)](api/languages/cpp_api.html#_C |
| anguages/cpp_api.html#_CPPv4N5cud | PPv4N5cudaq18QuEraRemoteRESTQPUE) |
| aq8kraus_op8kraus_opERK8kraus_op) | -   [cudaq::qvector (C++          |
| -   [cudaq::kraus_op::nCols (C++  |     class)                        |
|                                   | ](api/languages/cpp_api.html#_CPP |
| member)](api/languages/cpp_api.ht | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | -   [cudaq::qvector::back (C++    |
| -   [cudaq::kraus_op::nRows (C++  |     function)](a                  |
|                                   | pi/languages/cpp_api.html#_CPPv4N |
| member)](api/languages/cpp_api.ht | 5cudaq7qvector4backENSt6size_tE), |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) |                                   |
| -   [cudaq::kraus_op::operator=   |   [\[1\]](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq7qvector4backEv) |
|     function)                     | -   [cudaq::qvector::begin (C++   |
| ](api/languages/cpp_api.html#_CPP |     fu                            |
| v4N5cudaq8kraus_opaSERK8kraus_op) | nction)](api/languages/cpp_api.ht |
| -   [cudaq::kraus_op::precision   | ml#_CPPv4N5cudaq7qvector5beginEv) |
|     (C++                          | -   [cudaq::qvector::clear (C++   |
|     memb                          |     fu                            |
| er)](api/languages/cpp_api.html#_ | nction)](api/languages/cpp_api.ht |
| CPPv4N5cudaq8kraus_op9precisionE) | ml#_CPPv4N5cudaq7qvector5clearEv) |
| -   [cudaq::KrausSelection (C++   | -   [cudaq::qvector::end (C++     |
|     s                             |                                   |
| truct)](api/languages/cpp_api.htm | function)](api/languages/cpp_api. |
| l#_CPPv4N5cudaq14KrausSelectionE) | html#_CPPv4N5cudaq7qvector3endEv) |
| -   [cudaq:                       | -   [cudaq::qvector::front (C++   |
| :KrausSelection::circuit_location |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     member)](api/langua           | cudaq7qvector5frontENSt6size_tE), |
| ges/cpp_api.html#_CPPv4N5cudaq14K |                                   |
| rausSelection16circuit_locationE) |  [\[1\]](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq7qvector5frontEv) |
|  [cudaq::KrausSelection::is_error | -   [cudaq::qvector::operator=    |
|     (C++                          |     (C++                          |
|     member)](a                    |     functio                       |
| pi/languages/cpp_api.html#_CPPv4N | n)](api/languages/cpp_api.html#_C |
| 5cudaq14KrausSelection8is_errorE) | PPv4N5cudaq7qvectoraSERK7qvector) |
| -   [cudaq::Kra                   | -   [cudaq::qvector::operator\[\] |
| usSelection::kraus_operator_index |     (C++                          |
|     (C++                          |     function)                     |
|     member)](api/languages/       | ](api/languages/cpp_api.html#_CPP |
| cpp_api.html#_CPPv4N5cudaq14Kraus | v4N5cudaq7qvectorixEKNSt6size_tE) |
| Selection20kraus_operator_indexE) | -   [cudaq::qvector::qvector (C++ |
| -   [cuda                         |     function)](api/               |
| q::KrausSelection::KrausSelection | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq7qvector7qvectorENSt6size_tE), |
|     function)](a                  |     [\[1\]](a                     |
| pi/languages/cpp_api.html#_CPPv4N | pi/languages/cpp_api.html#_CPPv4N |
| 5cudaq14KrausSelection14KrausSele | 5cudaq7qvector7qvectorERK5state), |
| ctionENSt6size_tENSt6vectorINSt6s |     [\[2\]](api                   |
| ize_tEEENSt6stringENSt6size_tEb), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]](api/langu             | udaq7qvector7qvectorERK7qvector), |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     [\[3\]](ap                    |
| KrausSelection14KrausSelectionEv) | i/languages/cpp_api.html#_CPPv4N5 |
| -                                 | cudaq7qvector7qvectorERR7qvector) |
|   [cudaq::KrausSelection::op_name | -   [cudaq::qvector::size (C++    |
|     (C++                          |     fu                            |
|     member)](                     | nction)](api/languages/cpp_api.ht |
| api/languages/cpp_api.html#_CPPv4 | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| N5cudaq14KrausSelection7op_nameE) | -   [cudaq::qvector::slice (C++   |
| -   [                             |     function)](api/language       |
| cudaq::KrausSelection::operator== | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|     (C++                          | tor5sliceENSt6size_tENSt6size_tE) |
|     function)](api/languages      | -   [cudaq::qvector::value_type   |
| /cpp_api.html#_CPPv4NK5cudaq14Kra |     (C++                          |
| usSelectioneqERK14KrausSelection) |     typ                           |
| -                                 | e)](api/languages/cpp_api.html#_C |
|    [cudaq::KrausSelection::qubits | PPv4N5cudaq7qvector10value_typeE) |
|     (C++                          | -   [cudaq::qview (C++            |
|     member)]                      |     clas                          |
| (api/languages/cpp_api.html#_CPPv | s)](api/languages/cpp_api.html#_C |
| 4N5cudaq14KrausSelection6qubitsE) | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| -   [cudaq::KrausTrajectory (C++  | -   [cudaq::qview::back (C++      |
|     st                            |     function)                     |
| ruct)](api/languages/cpp_api.html | ](api/languages/cpp_api.html#_CPP |
| #_CPPv4N5cudaq15KrausTrajectoryE) | v4N5cudaq5qview4backENSt6size_tE) |
| -                                 | -   [cudaq::qview::begin (C++     |
|  [cudaq::KrausTrajectory::builder |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function)](ap                 | html#_CPPv4N5cudaq5qview5beginEv) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq::qview::end (C++       |
| cudaq15KrausTrajectory7builderEv) |                                   |
| -   [cu                           |   function)](api/languages/cpp_ap |
| daq::KrausTrajectory::countErrors | i.html#_CPPv4N5cudaq5qview3endEv) |
|     (C++                          | -   [cudaq::qview::front (C++     |
|     function)](api/lang           |     function)](                   |
| uages/cpp_api.html#_CPPv4NK5cudaq | api/languages/cpp_api.html#_CPPv4 |
| 15KrausTrajectory11countErrorsEv) | N5cudaq5qview5frontENSt6size_tE), |
| -   [                             |                                   |
| cudaq::KrausTrajectory::isOrdered |    [\[1\]](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qview5frontEv) |
|     function)](api/l              | -   [cudaq::qview::operator\[\]   |
| anguages/cpp_api.html#_CPPv4NK5cu |     (C++                          |
| daq15KrausTrajectory9isOrderedEv) |     functio                       |
| -   [cudaq::                      | n)](api/languages/cpp_api.html#_C |
| KrausTrajectory::kraus_selections | PPv4N5cudaq5qviewixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qview::qview (C++     |
|     member)](api/languag          |     functio                       |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | n)](api/languages/cpp_api.html#_C |
| ausTrajectory16kraus_selectionsE) | PPv4I0EN5cudaq5qview5qviewERR1R), |
| -   [cudaq:                       |     [\[1                          |
| :KrausTrajectory::KrausTrajectory | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq5qview5qviewERK5qview) |
|     function                      | -   [cudaq::qview::size (C++      |
| )](api/languages/cpp_api.html#_CP |                                   |
| Pv4N5cudaq15KrausTrajectory15Krau | function)](api/languages/cpp_api. |
| sTrajectoryENSt6size_tENSt6vector | html#_CPPv4NK5cudaq5qview4sizeEv) |
| I14KrausSelectionEEdNSt6size_tE), | -   [cudaq::qview::slice (C++     |
|     [\[1\]](api/languag           |     function)](api/langua         |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| ausTrajectory15KrausTrajectoryEv) | iew5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::Kr                    | -   [cudaq::qview::value_type     |
| ausTrajectory::measurement_counts |     (C++                          |
|     (C++                          |     t                             |
|     member)](api/languages        | ype)](api/languages/cpp_api.html# |
| /cpp_api.html#_CPPv4N5cudaq15Krau | _CPPv4N5cudaq5qview10value_typeE) |
| sTrajectory18measurement_countsE) | -   [cudaq::range (C++            |
| -   [cud                          |     fun                           |
| aq::KrausTrajectory::multiplicity | ction)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
|     member)](api/lan              | orI11ElementTypeEE11ElementType), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[1\]](api/languages/cpp_    |
| 15KrausTrajectory12multiplicityE) | api.html#_CPPv4I0EN5cudaq5rangeEN |
| -   [                             | St6vectorI11ElementTypeEE11Elemen |
| cudaq::KrausTrajectory::num_shots | tType11ElementType11ElementType), |
|     (C++                          |     [                             |
|     member)](api                  | \[2\]](api/languages/cpp_api.html |
| /languages/cpp_api.html#_CPPv4N5c | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| udaq15KrausTrajectory9num_shotsE) | -   [cudaq::real (C++             |
| -   [c                            |     type)](api/languages/         |
| udaq::KrausTrajectory::operator== | cpp_api.html#_CPPv4N5cudaq4realE) |
|     (C++                          | -   [cudaq::registry (C++         |
|     function)](api/languages/c    |     type)](api/languages/cpp_     |
| pp_api.html#_CPPv4NK5cudaq15Kraus | api.html#_CPPv4N5cudaq8registryE) |
| TrajectoryeqERK15KrausTrajectory) | -                                 |
| -   [cu                           |  [cudaq::registry::RegisteredType |
| daq::KrausTrajectory::probability |     (C++                          |
|     (C++                          |     class)](api/                  |
|     member)](api/la               | languages/cpp_api.html#_CPPv4I0EN |
| nguages/cpp_api.html#_CPPv4N5cuda | 5cudaq8registry14RegisteredTypeE) |
| q15KrausTrajectory11probabilityE) | -   [cudaq::RemoteCapabilities    |
| -   [cuda                         |     (C++                          |
| q::KrausTrajectory::trajectory_id |     struc                         |
|     (C++                          | t)](api/languages/cpp_api.html#_C |
|     member)](api/lang             | PPv4N5cudaq18RemoteCapabilitiesE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::Remot                 |
| 5KrausTrajectory13trajectory_idE) | eCapabilities::RemoteCapabilities |
| -                                 |     (C++                          |
|   [cudaq::KrausTrajectory::weight |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4N5cudaq18RemoteCa |
|     member)](                     | pabilities18RemoteCapabilitiesEb) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq:                       |
| N5cudaq15KrausTrajectory6weightE) | :RemoteCapabilities::stateOverlap |
| -                                 |     (C++                          |
|    [cudaq::KrausTrajectoryBuilder |     member)](api/langua           |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq18R |
|     class)](                      | emoteCapabilities12stateOverlapE) |
| api/languages/cpp_api.html#_CPPv4 | -                                 |
| N5cudaq22KrausTrajectoryBuilderE) |   [cudaq::RemoteCapabilities::vqe |
| -   [cud                          |     (C++                          |
| aq::KrausTrajectoryBuilder::build |     member)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/lang           | N5cudaq18RemoteCapabilities3vqeE) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::RemoteRESTQPU (C++    |
| 22KrausTrajectoryBuilder5buildEv) |                                   |
| -   [cud                          |  class)](api/languages/cpp_api.ht |
| aq::KrausTrajectoryBuilder::setId | ml#_CPPv4N5cudaq13RemoteRESTQPUE) |
|     (C++                          | -   [cudaq::Resources (C++        |
|     function)](api/languages/cpp  |     class)](api/languages/cpp_a   |
| _api.html#_CPPv4N5cudaq22KrausTra | pi.html#_CPPv4N5cudaq9ResourcesE) |
| jectoryBuilder5setIdENSt6size_tE) | -   [cudaq::run (C++              |
| -   [cudaq::Kraus                 |     function)]                    |
| TrajectoryBuilder::setProbability | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
|     function)](api/languages/cpp  | 5invoke_result_tINSt7decay_tI13Qu |
| _api.html#_CPPv4N5cudaq22KrausTra | antumKernelEEDpNSt7decay_tI4ARGSE |
| jectoryBuilder14setProbabilityEd) | EEEEENSt6size_tERN5cudaq11noise_m |
| -   [cudaq::Krau                  | odelERR13QuantumKernelDpRR4ARGS), |
| sTrajectoryBuilder::setSelections |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0DpEN5cu |
|     function)](api/languag        | daq3runENSt6vectorINSt15invoke_re |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | sult_tINSt7decay_tI13QuantumKerne |
| ausTrajectoryBuilder13setSelectio | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| nsENSt6vectorI14KrausSelectionEE) | ize_tERR13QuantumKernelDpRR4ARGS) |
| -   [cudaq::logical_observable    | -   [cudaq::run_async (C++        |
|     (C++                          |     functio                       |
|     function)](api/languages/c    | n)](api/languages/cpp_api.html#_C |
| pp_api.html#_CPPv4IDpEN5cudaq18lo | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| gical_observableEvDpRR8MeasArgs), | tureINSt6vectorINSt15invoke_resul |
|     [\[1\]](api/l                 | t_tINSt7decay_tI13QuantumKernelEE |
| anguages/cpp_api.html#_CPPv4N5cud | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| aq18logical_observableERKNSt6vect | ze_tENSt6size_tERN5cudaq11noise_m |
| orI14measure_resultEENSt6size_tE) | odelERR13QuantumKernelDpRR4ARGS), |
| -   [cudaq::M2DSparseMatrix (C++  |     [\[1\]](api/la                |
|     st                            | nguages/cpp_api.html#_CPPv4I0DpEN |
| ruct)](api/languages/cpp_api.html | 5cudaq9run_asyncENSt6futureINSt6v |
| #_CPPv4N5cudaq15M2DSparseMatrixE) | ectorINSt15invoke_result_tINSt7de |
| -   [cudaq::M2OSparseMatrix (C++  | cay_tI13QuantumKernelEEDpNSt7deca |
|     st                            | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| ruct)](api/languages/cpp_api.html | ize_tERR13QuantumKernelDpRR4ARGS) |
| #_CPPv4N5cudaq15M2OSparseMatrixE) | -   [cudaq::RuntimeTarget (C++    |
| -   [cudaq::matrix_callback (C++  |                                   |
|     c                             | struct)](api/languages/cpp_api.ht |
| lass)](api/languages/cpp_api.html | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| #_CPPv4N5cudaq15matrix_callbackE) | -   [cudaq::sample (C++           |
| -   [cudaq::matrix_handler (C++   |     function)](api/languages/c    |
|                                   | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| class)](api/languages/cpp_api.htm | mpleE13sample_resultRK14sample_op |
| l#_CPPv4N5cudaq14matrix_handlerE) | tionsRR13QuantumKernelDpRR4Args), |
| -   [cudaq::mat                   |     [\[1\                         |
| rix_handler::commutation_behavior | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     struct)](api/languages/       | esultRR13QuantumKernelDpRR4Args), |
| cpp_api.html#_CPPv4N5cudaq14matri |     [\                            |
| x_handler20commutation_behaviorE) | [2\]](api/languages/cpp_api.html# |
| -                                 | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|    [cudaq::matrix_handler::define | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::sample_options (C++   |
|     function)](a                  |     s                             |
| pi/languages/cpp_api.html#_CPPv4N | truct)](api/languages/cpp_api.htm |
| 5cudaq14matrix_handler6defineENSt | l#_CPPv4N5cudaq14sample_optionsE) |
| 6stringENSt6vectorINSt7int64_tEEE | -   [cudaq::sample_result (C++    |
| RR15matrix_callbackRKNSt13unorder |                                   |
| ed_mapINSt6stringENSt6stringEEE), |  class)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4N5cudaq13sample_resultE) |
| [\[1\]](api/languages/cpp_api.htm | -   [cudaq::sample_result::append |
| l#_CPPv4N5cudaq14matrix_handler6d |     (C++                          |
| efineENSt6stringENSt6vectorINSt7i |     function)](api/languages/cpp_ |
| nt64_tEEERR15matrix_callbackRR20d | api.html#_CPPv4N5cudaq13sample_re |
| iag_matrix_callbackRKNSt13unorder | sult6appendERK15ExecutionResultb) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::sample_result::begin  |
|     [\[2\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)]                    |
| N5cudaq14matrix_handler6defineENS | (api/languages/cpp_api.html#_CPPv |
| t6stringENSt6vectorINSt7int64_tEE | 4N5cudaq13sample_result5beginEv), |
| ERR15matrix_callbackRRNSt13unorde |     [\[1\]]                       |
| red_mapINSt6stringENSt6stringEEE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NK5cudaq13sample_result5beginEv) |
|   [cudaq::matrix_handler::degrees | -   [cudaq::sample_result::cbegin |
|     (C++                          |     (C++                          |
|     function)](ap                 |     function)](                   |
| i/languages/cpp_api.html#_CPPv4NK | api/languages/cpp_api.html#_CPPv4 |
| 5cudaq14matrix_handler7degreesEv) | NK5cudaq13sample_result6cbeginEv) |
| -                                 | -   [cudaq::sample_result::cend   |
|  [cudaq::matrix_handler::displace |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/language       | ](api/languages/cpp_api.html#_CPP |
| s/cpp_api.html#_CPPv4N5cudaq14mat | v4NK5cudaq13sample_result4cendEv) |
| rix_handler8displaceENSt6size_tE) | -   [cudaq::sample_result::clear  |
| -   [cudaq::matrix                |     (C++                          |
| _handler::get_expected_dimensions |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq13sample_result5clearEv) |
|    function)](api/languages/cpp_a | -   [cudaq::sample_result::count  |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     (C++                          |
| ndler23get_expected_dimensionsEv) |     function)](                   |
| -   [cudaq::matrix_ha             | api/languages/cpp_api.html#_CPPv4 |
| ndler::get_parameter_descriptions | NK5cudaq13sample_result5countENSt |
|     (C++                          | 11string_viewEKNSt11string_viewE) |
|                                   | -   [                             |
| function)](api/languages/cpp_api. | cudaq::sample_result::deserialize |
| html#_CPPv4NK5cudaq14matrix_handl |     (C++                          |
| er26get_parameter_descriptionsEv) |     functio                       |
| -   [c                            | n)](api/languages/cpp_api.html#_C |
| udaq::matrix_handler::instantiate | PPv4N5cudaq13sample_result11deser |
|     (C++                          | ializeERNSt6vectorINSt6size_tEEE) |
|     function)](a                  | -   [cudaq::sample_result::dump   |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14matrix_handler11instantia |     function)](api/languag        |
| teENSt6stringERKNSt6vectorINSt6si | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ze_tEEERK20commutation_behavior), | ample_result4dumpERNSt7ostreamE), |
|     [\[1\]](                      |     [\[1\]                        |
| api/languages/cpp_api.html#_CPPv4 | ](api/languages/cpp_api.html#_CPP |
| N5cudaq14matrix_handler11instanti | v4NK5cudaq13sample_result4dumpEv) |
| ateENSt6stringERRNSt6vectorINSt6s | -   [cudaq::sample_result::end    |
| ize_tEEERK20commutation_behavior) |     (C++                          |
| -   [cuda                         |     function                      |
| q::matrix_handler::matrix_handler | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq13sample_result3endEv), |
|     function)](api/languag        |     [\[1\                         |
| es/cpp_api.html#_CPPv4I0_NSt11ena | ]](api/languages/cpp_api.html#_CP |
| ble_if_tINSt12is_base_of_vI16oper | Pv4NK5cudaq13sample_result3endEv) |
| ator_handler1TEEbEEEN5cudaq14matr | -   [                             |
| ix_handler14matrix_handlerERK1T), | cudaq::sample_result::expectation |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4I0 |     f                             |
| _NSt11enable_if_tINSt12is_base_of | unction)](api/languages/cpp_api.h |
| _vI16operator_handler1TEEbEEEN5cu | tml#_CPPv4NK5cudaq13sample_result |
| daq14matrix_handler14matrix_handl | 11expectationEKNSt11string_viewE) |
| erERK1TRK20commutation_behavior), | -   [c                            |
|     [\[2\]](api/languages/cpp_ap  | udaq::sample_result::get_marginal |
| i.html#_CPPv4N5cudaq14matrix_hand |     (C++                          |
| ler14matrix_handlerENSt6size_tE), |     function)](api/languages/cpp_ |
|     [\[3\]](api/                  | api.html#_CPPv4NK5cudaq13sample_r |
| languages/cpp_api.html#_CPPv4N5cu | esult12get_marginalERKNSt6vectorI |
| daq14matrix_handler14matrix_handl | NSt6size_tEEEKNSt11string_viewE), |
| erENSt6stringERKNSt6vectorINSt6si |     [\[1\]](api/languages/cpp_    |
| ze_tEEERK20commutation_behavior), | api.html#_CPPv4NK5cudaq13sample_r |
|     [\[4\]](api/                  | esult12get_marginalERRKNSt6vector |
| languages/cpp_api.html#_CPPv4N5cu | INSt6size_tEEEKNSt11string_viewE) |
| daq14matrix_handler14matrix_handl | -   [cuda                         |
| erENSt6stringERRNSt6vectorINSt6si | q::sample_result::get_total_shots |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\                            |     function)](api/langua         |
| [5\]](api/languages/cpp_api.html# | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| _CPPv4N5cudaq14matrix_handler14ma | sample_result15get_total_shotsEv) |
| trix_handlerERK14matrix_handler), | -   [cuda                         |
|     [                             | q::sample_result::has_even_parity |
| \[6\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq14matrix_handler14m |     fun                           |
| atrix_handlerERR14matrix_handler) | ction)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq13sample_result15h |
|  [cudaq::matrix_handler::momentum | as_even_parityENSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|     function)](api/language       | q::sample_result::has_expectation |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8momentumENSt6size_tE) |     funct                         |
| -                                 | ion)](api/languages/cpp_api.html# |
|    [cudaq::matrix_handler::number | _CPPv4NK5cudaq13sample_result15ha |
|     (C++                          | s_expectationEKNSt11string_viewE) |
|     function)](api/langua         | -   [cu                           |
| ges/cpp_api.html#_CPPv4N5cudaq14m | daq::sample_result::most_probable |
| atrix_handler6numberENSt6size_tE) |     (C++                          |
| -                                 |     fun                           |
| [cudaq::matrix_handler::operator= | ction)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NK5cudaq13sample_result13 |
|     fun                           | most_probableEKNSt11string_viewE) |
| ction)](api/languages/cpp_api.htm | -                                 |
| l#_CPPv4I0_NSt11enable_if_tIXaant | [cudaq::sample_result::operator+= |
| NSt7is_sameI1T14matrix_handlerE5v |     (C++                          |
| alueENSt12is_base_of_vI16operator |     function)](api/langua         |
| _handler1TEEEbEEEN5cudaq14matrix_ | ges/cpp_api.html#_CPPv4N5cudaq13s |
| handleraSER14matrix_handlerRK1T), | ample_resultpLERK13sample_result) |
|     [\[1\]](api/languages         | -                                 |
| /cpp_api.html#_CPPv4N5cudaq14matr |  [cudaq::sample_result::operator= |
| ix_handleraSERK14matrix_handler), |     (C++                          |
|     [\[2\]](api/language          |     function)](api/langua         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ges/cpp_api.html#_CPPv4N5cudaq13s |
| rix_handleraSERR14matrix_handler) | ample_resultaSERR13sample_result) |
| -   [                             | -                                 |
| cudaq::matrix_handler::operator== | [cudaq::sample_result::operator== |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/languag        |
| /cpp_api.html#_CPPv4NK5cudaq14mat | es/cpp_api.html#_CPPv4NK5cudaq13s |
| rix_handlereqERK14matrix_handler) | ample_resulteqERK13sample_result) |
| -                                 | -   [                             |
|    [cudaq::matrix_handler::parity | cudaq::sample_result::probability |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/lan            |
| ges/cpp_api.html#_CPPv4N5cudaq14m | guages/cpp_api.html#_CPPv4NK5cuda |
| atrix_handler6parityENSt6size_tE) | q13sample_result11probabilityENSt |
| -                                 | 11string_viewEKNSt11string_viewE) |
|  [cudaq::matrix_handler::position | -   [cud                          |
|     (C++                          | aq::sample_result::register_names |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     function)](api/langu          |
| rix_handler8positionENSt6size_tE) | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| -   [cudaq::                      | 3sample_result14register_namesEv) |
| matrix_handler::remove_definition | -                                 |
|     (C++                          |    [cudaq::sample_result::reorder |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/langua         |
| ml#_CPPv4N5cudaq14matrix_handler1 | ges/cpp_api.html#_CPPv4N5cudaq13s |
| 7remove_definitionERKNSt6stringE) | ample_result7reorderERKNSt6vector |
| -                                 | INSt6size_tEEEKNSt11string_viewE) |
|   [cudaq::matrix_handler::squeeze | -   [cu                           |
|     (C++                          | daq::sample_result::sample_result |
|     function)](api/languag        |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     func                          |
| trix_handler7squeezeENSt6size_tE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::m                     | #_CPPv4N5cudaq13sample_result13sa |
| atrix_handler::to_diagonal_matrix | mple_resultERK15ExecutionResult), |
|     (C++                          |     [\[1\]](api/la                |
|     function)](api/lang           | nguages/cpp_api.html#_CPPv4N5cuda |
| uages/cpp_api.html#_CPPv4NK5cudaq | q13sample_result13sample_resultER |
| 14matrix_handler18to_diagonal_mat | KNSt6vectorI15ExecutionResultEE), |
| rixERNSt13unordered_mapINSt6size_ |                                   |
| tENSt7int64_tEEERKNSt13unordered_ |  [\[2\]](api/languages/cpp_api.ht |
| mapINSt6stringENSt7complexIdEEEE) | ml#_CPPv4N5cudaq13sample_result13 |
| -                                 | sample_resultERR13sample_result), |
| [cudaq::matrix_handler::to_matrix |     [                             |
|     (C++                          | \[3\]](api/languages/cpp_api.html |
|     function)                     | #_CPPv4N5cudaq13sample_result13sa |
| ](api/languages/cpp_api.html#_CPP | mple_resultERR15ExecutionResult), |
| v4NK5cudaq14matrix_handler9to_mat |     [\[4\]](api/lan               |
| rixERNSt13unordered_mapINSt6size_ | guages/cpp_api.html#_CPPv4N5cudaq |
| tENSt7int64_tEEERKNSt13unordered_ | 13sample_result13sample_resultEdR |
| mapINSt6stringENSt7complexIdEEEE) | KNSt6vectorI15ExecutionResultEE), |
| -                                 |     [\[5\]](api/lan               |
| [cudaq::matrix_handler::to_string | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 13sample_result13sample_resultEv) |
|     function)](api/               | -                                 |
| languages/cpp_api.html#_CPPv4NK5c |  [cudaq::sample_result::serialize |
| udaq14matrix_handler9to_stringEb) |     (C++                          |
| -                                 |     function)](api                |
| [cudaq::matrix_handler::unique_id | /languages/cpp_api.html#_CPPv4NK5 |
|     (C++                          | cudaq13sample_result9serializeEv) |
|     function)](api/               | -   [cudaq::sample_result::size   |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14matrix_handler9unique_idEv) |     function)](api/languages/c    |
| -   [cudaq:                       | pp_api.html#_CPPv4NK5cudaq13sampl |
| :matrix_handler::\~matrix_handler | e_result4sizeEKNSt11string_viewE) |
|     (C++                          | -   [cudaq::sample_result::to_map |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)](api/languages/cpp  |
| CPPv4N5cudaq14matrix_handlerD0Ev) | _api.html#_CPPv4NK5cudaq13sample_ |
| -   [cudaq::matrix_op (C++        | result6to_mapEKNSt11string_viewE) |
|     type)](api/languages/cpp_a    | -   [cuda                         |
| pi.html#_CPPv4N5cudaq9matrix_opE) | q::sample_result::\~sample_result |
| -   [cudaq::matrix_op_term (C++   |     (C++                          |
|                                   |     funct                         |
|  type)](api/languages/cpp_api.htm | ion)](api/languages/cpp_api.html# |
| l#_CPPv4N5cudaq14matrix_op_termE) | _CPPv4N5cudaq13sample_resultD0Ev) |
| -                                 | -   [cudaq::scalar_callback (C++  |
|    [cudaq::mdiag_operator_handler |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|     class)](                      | #_CPPv4N5cudaq15scalar_callbackE) |
| api/languages/cpp_api.html#_CPPv4 | -   [c                            |
| N5cudaq22mdiag_operator_handlerE) | udaq::scalar_callback::operator() |
| -   [cudaq::measure_handle (C++   |     (C++                          |
|                                   |     function)](api/language       |
| class)](api/languages/cpp_api.htm | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| l#_CPPv4N5cudaq14measure_handleE) | alar_callbackclERKNSt13unordered_ |
| -   [cudaq::measure_result (C++   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [                             |
|  type)](api/languages/cpp_api.htm | cudaq::scalar_callback::operator= |
| l#_CPPv4N5cudaq14measure_resultE) |     (C++                          |
| -   [cudaq::mpi (C++              |     function)](api/languages/c    |
|     type)](api/languages          | pp_api.html#_CPPv4N5cudaq15scalar |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | _callbackaSERK15scalar_callback), |
| -   [cudaq::mpi::all_gather (C++  |     [\[1\]](api/languages/        |
|     fu                            | cpp_api.html#_CPPv4N5cudaq15scala |
| nction)](api/languages/cpp_api.ht | r_callbackaSERR15scalar_callback) |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | -   [cudaq:                       |
| RNSt6vectorIdEERKNSt6vectorIdEE), | :scalar_callback::scalar_callback |
|                                   |     (C++                          |
|   [\[1\]](api/languages/cpp_api.h |     function)](api/languag        |
| tml#_CPPv4N5cudaq3mpi10all_gather | es/cpp_api.html#_CPPv4I0_NSt11ena |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | ble_if_tINSt16is_invocable_r_vINS |
| -   [cudaq::mpi::all_reduce (C++  | t7complexIdEE8CallableRKNSt13unor |
|                                   | dered_mapINSt6stringENSt7complexI |
|  function)](api/languages/cpp_api | dEEEEEEbEEEN5cudaq15scalar_callba |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | ck15scalar_callbackERR8Callable), |
| reduceE1TRK1TRK14BinaryFunction), |     [\[1\                         |
|     [\[1\]](api/langu             | ]](api/languages/cpp_api.html#_CP |
| ages/cpp_api.html#_CPPv4I00EN5cud | Pv4N5cudaq15scalar_callback15scal |
| aq3mpi10all_reduceE1TRK1TRK4Func) | ar_callbackERK15scalar_callback), |
| -   [cudaq::mpi::broadcast (C++   |     [\[2                          |
|     function)](api/               | \]](api/languages/cpp_api.html#_C |
| languages/cpp_api.html#_CPPv4N5cu | PPv4N5cudaq15scalar_callback15sca |
| daq3mpi9broadcastERNSt6stringEi), | lar_callbackERR15scalar_callback) |
|     [\[1\]](api/la                | -   [cudaq::scalar_operator (C++  |
| nguages/cpp_api.html#_CPPv4N5cuda |     c                             |
| q3mpi9broadcastERNSt6vectorIdEEi) | lass)](api/languages/cpp_api.html |
| -   [cudaq::mpi::finalize (C++    | #_CPPv4N5cudaq15scalar_operatorE) |
|     f                             | -                                 |
| unction)](api/languages/cpp_api.h | [cudaq::scalar_operator::evaluate |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     (C++                          |
| -   [cudaq::mpi::initialize (C++  |                                   |
|     function                      |    function)](api/languages/cpp_a |
| )](api/languages/cpp_api.html#_CP | pi.html#_CPPv4NK5cudaq15scalar_op |
| Pv4N5cudaq3mpi10initializeEiPPc), | erator8evaluateERKNSt13unordered_ |
|     [                             | mapINSt6stringENSt7complexIdEEEE) |
| \[1\]](api/languages/cpp_api.html | -   [cudaq::scalar_ope            |
| #_CPPv4N5cudaq3mpi10initializeEv) | rator::get_parameter_descriptions |
| -   [cudaq::mpi::is_initialized   |     (C++                          |
|     (C++                          |     f                             |
|     function                      | unction)](api/languages/cpp_api.h |
| )](api/languages/cpp_api.html#_CP | tml#_CPPv4NK5cudaq15scalar_operat |
| Pv4N5cudaq3mpi14is_initializedEv) | or26get_parameter_descriptionsEv) |
| -   [cudaq::mpi::num_ranks (C++   | -   [cu                           |
|     fu                            | daq::scalar_operator::is_constant |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     function)](api/lang           |
| -   [cudaq::mpi::rank (C++        | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 15scalar_operator11is_constantEv) |
|    function)](api/languages/cpp_a | -   [c                            |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | udaq::scalar_operator::operator\* |
| -   [cudaq::noise_model (C++      |     (C++                          |
|                                   |     function                      |
|    class)](api/languages/cpp_api. | )](api/languages/cpp_api.html#_CP |
| html#_CPPv4N5cudaq11noise_modelE) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::n                     | 7complexIdEERK15scalar_operator), |
| oise_model::add_all_qubit_channel |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api                | Pv4N5cudaq15scalar_operatormlENSt |
| /languages/cpp_api.html#_CPPv4IDp | 7complexIdEERR15scalar_operator), |
| EN5cudaq11noise_model21add_all_qu |     [\[2\]](api/languages/cp      |
| bit_channelEvRK13kraus_channeli), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](api/langua            | operatormlEdRK15scalar_operator), |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     [\[3\]](api/languages/cp      |
| oise_model21add_all_qubit_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
| ERKNSt6stringERK13kraus_channeli) | operatormlEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
|  [cudaq::noise_model::add_channel | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormlENSt7complexIdEE), |
|     funct                         |     [\[5\]](api/languages/cpp     |
| ion)](api/languages/cpp_api.html# | _api.html#_CPPv4NKR5cudaq15scalar |
| _CPPv4IDpEN5cudaq11noise_model11a | _operatormlERK15scalar_operator), |
| dd_channelEvRK15PredicateFuncTy), |     [\[6\]]                       |
|     [\[1\]](api/languages/cpp_    | (api/languages/cpp_api.html#_CPPv |
| api.html#_CPPv4IDpEN5cudaq11noise | 4NKR5cudaq15scalar_operatormlEd), |
| _model11add_channelEvRKNSt6vector |     [\[7\]](api/language          |
| INSt6size_tEEERK13kraus_channel), | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     [\[2\]](ap                    | alar_operatormlENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[8\]](api/languages/cp      |
| cudaq11noise_model11add_channelER | p_api.html#_CPPv4NO5cudaq15scalar |
| KNSt6stringERK15PredicateFuncTy), | _operatormlERK15scalar_operator), |
|                                   |     [\[9\                         |
| [\[3\]](api/languages/cpp_api.htm | ]](api/languages/cpp_api.html#_CP |
| l#_CPPv4N5cudaq11noise_model11add | Pv4NO5cudaq15scalar_operatormlEd) |
| _channelERKNSt6stringERKNSt6vecto | -   [cu                           |
| rINSt6size_tEEERK13kraus_channel) | daq::scalar_operator::operator\*= |
| -   [cudaq::noise_model::empty    |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function                      | es/cpp_api.html#_CPPv4N5cudaq15sc |
| )](api/languages/cpp_api.html#_CP | alar_operatormLENSt7complexIdEE), |
| Pv4NK5cudaq11noise_model5emptyEv) |     [\[1\]](api/languages/c       |
| -                                 | pp_api.html#_CPPv4N5cudaq15scalar |
| [cudaq::noise_model::get_channels | _operatormLERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     function)](api/l              | \]](api/languages/cpp_api.html#_C |
| anguages/cpp_api.html#_CPPv4I0ENK | PPv4N5cudaq15scalar_operatormLEd) |
| 5cudaq11noise_model12get_channels | -   [                             |
| ENSt6vectorI13kraus_channelEERKNS | cudaq::scalar_operator::operator+ |
| t6vectorINSt6size_tEEERKNSt6vecto |     (C++                          |
| rINSt6size_tEEERKNSt6vectorIdEE), |     function                      |
|     [\[1\]](api/languages/cpp_a   | )](api/languages/cpp_api.html#_CP |
| pi.html#_CPPv4NK5cudaq11noise_mod | Pv4N5cudaq15scalar_operatorplENSt |
| el12get_channelsERKNSt6stringERKN | 7complexIdEERK15scalar_operator), |
| St6vectorINSt6size_tEEERKNSt6vect |     [\[1\                         |
| orINSt6size_tEEERKNSt6vectorIdEE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatorplENSt |
|  [cudaq::noise_model::noise_model | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     function)](api                | p_api.html#_CPPv4N5cudaq15scalar_ |
| /languages/cpp_api.html#_CPPv4N5c | operatorplEdRK15scalar_operator), |
| udaq11noise_model11noise_modelEv) |     [\[3\]](api/languages/cp      |
| -   [cu                           | p_api.html#_CPPv4N5cudaq15scalar_ |
| daq::noise_model::PredicateFuncTy | operatorplEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     type)](api/la                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| nguages/cpp_api.html#_CPPv4N5cuda | alar_operatorplENSt7complexIdEE), |
| q11noise_model15PredicateFuncTyE) |     [\[5\]](api/languages/cpp     |
| -   [cud                          | _api.html#_CPPv4NKR5cudaq15scalar |
| aq::noise_model::register_channel | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function)](api/languages      | (api/languages/cpp_api.html#_CPPv |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | 4NKR5cudaq15scalar_operatorplEd), |
| noise_model16register_channelEvv) |     [\[7\]]                       |
| -   [cudaq::                      | (api/languages/cpp_api.html#_CPPv |
| noise_model::requires_constructor | 4NKR5cudaq15scalar_operatorplEv), |
|     (C++                          |     [\[8\]](api/language          |
|     type)](api/languages/cp       | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| p_api.html#_CPPv4I0DpEN5cudaq11no | alar_operatorplENSt7complexIdEE), |
| ise_model20requires_constructorE) |     [\[9\]](api/languages/cp      |
| -   [cudaq::noise_model_type (C++ | p_api.html#_CPPv4NO5cudaq15scalar |
|     e                             | _operatorplERK15scalar_operator), |
| num)](api/languages/cpp_api.html# |     [\[10\]                       |
| _CPPv4N5cudaq16noise_model_typeE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::no                    | v4NO5cudaq15scalar_operatorplEd), |
| ise_model_type::amplitude_damping |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](api/languages    | Pv4NO5cudaq15scalar_operatorplEv) |
| /cpp_api.html#_CPPv4N5cudaq16nois | -   [c                            |
| e_model_type17amplitude_dampingE) | udaq::scalar_operator::operator+= |
| -   [cudaq::noise_mode            |     (C++                          |
| l_type::amplitude_damping_channel |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     e                             | alar_operatorpLENSt7complexIdEE), |
| numerator)](api/languages/cpp_api |     [\[1\]](api/languages/c       |
| .html#_CPPv4N5cudaq16noise_model_ | pp_api.html#_CPPv4N5cudaq15scalar |
| type25amplitude_damping_channelE) | _operatorpLERK15scalar_operator), |
| -   [cudaq::n                     |     [\[2                          |
| oise_model_type::bit_flip_channel | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatorpLEd) |
|     enumerator)](api/language     | -   [                             |
| s/cpp_api.html#_CPPv4N5cudaq16noi | cudaq::scalar_operator::operator- |
| se_model_type16bit_flip_channelE) |     (C++                          |
| -   [cudaq::                      |     function                      |
| noise_model_type::depolarization1 | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormiENSt |
|     enumerator)](api/languag      | 7complexIdEERK15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[1\                         |
| ise_model_type15depolarization1E) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::                      | Pv4N5cudaq15scalar_operatormiENSt |
| noise_model_type::depolarization2 | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     enumerator)](api/languag      | p_api.html#_CPPv4N5cudaq15scalar_ |
| es/cpp_api.html#_CPPv4N5cudaq16no | operatormiEdRK15scalar_operator), |
| ise_model_type15depolarization2E) |     [\[3\]](api/languages/cp      |
| -   [cudaq::noise_m               | p_api.html#_CPPv4N5cudaq15scalar_ |
| odel_type::depolarization_channel | operatormiEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|                                   | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|   enumerator)](api/languages/cpp_ | alar_operatormiENSt7complexIdEE), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[5\]](api/languages/cpp     |
| el_type22depolarization_channelE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -                                 | _operatormiERK15scalar_operator), |
|  [cudaq::noise_model_type::pauli1 |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     enumerator)](a                | 4NKR5cudaq15scalar_operatormiEd), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[7\]]                       |
| 5cudaq16noise_model_type6pauli1E) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatormiEv), |
|  [cudaq::noise_model_type::pauli2 |     [\[8\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     enumerator)](a                | alar_operatormiENSt7complexIdEE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[9\]](api/languages/cp      |
| 5cudaq16noise_model_type6pauli2E) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq                        | _operatormiERK15scalar_operator), |
| ::noise_model_type::phase_damping |     [\[10\]                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     enumerator)](api/langu        | v4NO5cudaq15scalar_operatormiEd), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     [\[11\                        |
| noise_model_type13phase_dampingE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noi                   | Pv4NO5cudaq15scalar_operatormiEv) |
| se_model_type::phase_flip_channel | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator-= |
|     enumerator)](api/languages/   |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16noise |     function)](api/languag        |
| _model_type18phase_flip_channelE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -                                 | alar_operatormIENSt7complexIdEE), |
| [cudaq::noise_model_type::unknown |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](ap               | _operatormIERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[2                          |
| cudaq16noise_model_type7unknownE) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq15scalar_operatormIEd) |
| [cudaq::noise_model_type::x_error | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator/ |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function                      |
| cudaq16noise_model_type7x_errorE) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatordvENSt |
| [cudaq::noise_model_type::y_error | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](ap               | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4N5cudaq15scalar_operatordvENSt |
| cudaq16noise_model_type7y_errorE) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
| [cudaq::noise_model_type::z_error | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRK15scalar_operator), |
|     enumerator)](ap               |     [\[3\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4N5cudaq15scalar_ |
| cudaq16noise_model_type7z_errorE) | operatordvEdRR15scalar_operator), |
| -   [cudaq::num_available_gpus    |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     function                      | alar_operatordvENSt7complexIdEE), |
| )](api/languages/cpp_api.html#_CP |     [\[5\]](api/languages/cpp     |
| Pv4N5cudaq18num_available_gpusEv) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::observe (C++          | _operatordvERK15scalar_operator), |
|     function)]                    |     [\[6\]]                       |
| (api/languages/cpp_api.html#_CPPv | (api/languages/cpp_api.html#_CPPv |
| 4I00DpEN5cudaq7observeENSt6vector | 4NKR5cudaq15scalar_operatordvEd), |
| I14observe_resultEERR13QuantumKer |     [\[7\]](api/language          |
| nelRK15SpinOpContainerDpRR4Args), | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     [\[1\]](api/languages/cpp_ap  | alar_operatordvENSt7complexIdEE), |
| i.html#_CPPv4I0DpEN5cudaq7observe |     [\[8\]](api/languages/cp      |
| E14observe_resultNSt6size_tERR13Q | p_api.html#_CPPv4NO5cudaq15scalar |
| uantumKernelRK7spin_opDpRR4Args), | _operatordvERK15scalar_operator), |
|     [\[                           |     [\[9\                         |
| 2\]](api/languages/cpp_api.html#_ | ]](api/languages/cpp_api.html#_CP |
| CPPv4I0DpEN5cudaq7observeE14obser | Pv4NO5cudaq15scalar_operatordvEd) |
| ve_resultRK15observe_optionsRR13Q | -   [c                            |
| uantumKernelRK7spin_opDpRR4Args), | udaq::scalar_operator::operator/= |
|     [\[3\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     function)](api/languag        |
| udaq7observeE14observe_resultRR13 | es/cpp_api.html#_CPPv4N5cudaq15sc |
| QuantumKernelRK7spin_opDpRR4Args) | alar_operatordVENSt7complexIdEE), |
| -   [cudaq::observe_options (C++  |     [\[1\]](api/languages/c       |
|     st                            | pp_api.html#_CPPv4N5cudaq15scalar |
| ruct)](api/languages/cpp_api.html | _operatordVERK15scalar_operator), |
| #_CPPv4N5cudaq15observe_optionsE) |     [\[2                          |
| -   [cudaq::observe_result (C++   | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq15scalar_operatordVEd) |
| class)](api/languages/cpp_api.htm | -   [                             |
| l#_CPPv4N5cudaq14observe_resultE) | cudaq::scalar_operator::operator= |
| -                                 |     (C++                          |
|    [cudaq::observe_result::counts |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function)](api/languages/c    | _operatoraSERK15scalar_operator), |
| pp_api.html#_CPPv4N5cudaq14observ |     [\[1\]](api/languages/        |
| e_result6countsERK12spin_op_term) | cpp_api.html#_CPPv4N5cudaq15scala |
| -   [cudaq::observe_result::dump  | r_operatoraSERR15scalar_operator) |
|     (C++                          | -   [c                            |
|     function)                     | udaq::scalar_operator::operator== |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq14observe_result4dumpEv) |     function)](api/languages/c    |
| -   [c                            | pp_api.html#_CPPv4NK5cudaq15scala |
| udaq::observe_result::expectation | r_operatoreqERK15scalar_operator) |
|     (C++                          | -   [cudaq:                       |
|                                   | :scalar_operator::scalar_operator |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq14observe_resul |     func                          |
| t11expectationERK12spin_op_term), | tion)](api/languages/cpp_api.html |
|     [\[1\]](api/la                | #_CPPv4N5cudaq15scalar_operator15 |
| nguages/cpp_api.html#_CPPv4N5cuda | scalar_operatorENSt7complexIdEE), |
| q14observe_result11expectationEv) |     [\[1\]](api/langu             |
| -   [cuda                         | ages/cpp_api.html#_CPPv4N5cudaq15 |
| q::observe_result::id_coefficient | scalar_operator15scalar_operatorE |
|     (C++                          | RK15scalar_callbackRRNSt13unorder |
|     function)](api/langu          | ed_mapINSt6stringENSt6stringEEE), |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     [\[2\                         |
| observe_result14id_coefficientEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cuda                         | Pv4N5cudaq15scalar_operator15scal |
| q::observe_result::observe_result | ar_operatorERK15scalar_operator), |
|     (C++                          |     [\[3\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq15 |
|   function)](api/languages/cpp_ap | scalar_operator15scalar_operatorE |
| i.html#_CPPv4N5cudaq14observe_res | RR15scalar_callbackRRNSt13unorder |
| ult14observe_resultEdRK7spin_op), | ed_mapINSt6stringENSt6stringEEE), |
|     [\[1\]](a                     |     [\[4\                         |
| pi/languages/cpp_api.html#_CPPv4N | ]](api/languages/cpp_api.html#_CP |
| 5cudaq14observe_result14observe_r | Pv4N5cudaq15scalar_operator15scal |
| esultEdRK7spin_op13sample_result) | ar_operatorERR15scalar_operator), |
| -                                 |     [\[5\]](api/language          |
|  [cudaq::observe_result::operator | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     double (C++                   | lar_operator15scalar_operatorEd), |
|     functio                       |     [\[6\]](api/languag           |
| n)](api/languages/cpp_api.html#_C | es/cpp_api.html#_CPPv4N5cudaq15sc |
| PPv4N5cudaq14observe_resultcvdEv) | alar_operator15scalar_operatorEv) |
| -                                 | -   [                             |
|  [cudaq::observe_result::raw_data | cudaq::scalar_operator::to_matrix |
|     (C++                          |     (C++                          |
|     function)](ap                 |                                   |
| i/languages/cpp_api.html#_CPPv4N5 |   function)](api/languages/cpp_ap |
| cudaq14observe_result8raw_dataEv) | i.html#_CPPv4NK5cudaq15scalar_ope |
| -   [cudaq::operator_handler (C++ | rator9to_matrixERKNSt13unordered_ |
|     cl                            | mapINSt6stringENSt7complexIdEEEE) |
| ass)](api/languages/cpp_api.html# | -   [                             |
| _CPPv4N5cudaq16operator_handlerE) | cudaq::scalar_operator::to_string |
| -   [cudaq::optimizable_function  |     (C++                          |
|     (C++                          |     function)](api/l              |
|     class)                        | anguages/cpp_api.html#_CPPv4NK5cu |
| ](api/languages/cpp_api.html#_CPP | daq15scalar_operator9to_stringEv) |
| v4N5cudaq20optimizable_functionE) | -   [cudaq::s                     |
| -   [cudaq::optimization_result   | calar_operator::\~scalar_operator |
|     (C++                          |     (C++                          |
|     type                          |     functio                       |
| )](api/languages/cpp_api.html#_CP | n)](api/languages/cpp_api.html#_C |
| Pv4N5cudaq19optimization_resultE) | PPv4N5cudaq15scalar_operatorD0Ev) |
| -   [cudaq::optimizer (C++        | -   [cudaq::set_noise (C++        |
|     class)](api/languages/cpp_a   |     function)](api/langu          |
| pi.html#_CPPv4N5cudaq9optimizerE) | ages/cpp_api.html#_CPPv4N5cudaq9s |
| -   [cudaq::optimizer::optimize   | et_noiseERKN5cudaq11noise_modelE) |
|     (C++                          | -   [cudaq::set_random_seed (C++  |
|                                   |     function)](api/               |
|  function)](api/languages/cpp_api | languages/cpp_api.html#_CPPv4N5cu |
| .html#_CPPv4N5cudaq9optimizer8opt | daq15set_random_seedENSt6size_tE) |
| imizeEKiRR20optimizable_function) | -   [cudaq::simulation_precision  |
| -   [cu                           |     (C++                          |
| daq::optimizer::requiresGradients |     enum)                         |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/la             | v4N5cudaq20simulation_precisionE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [                             |
| q9optimizer17requiresGradientsEv) | cudaq::simulation_precision::fp32 |
| -   [cudaq::orca (C++             |     (C++                          |
|     type)](api/languages/         |     enumerator)](api              |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::orca::sample (C++     | udaq20simulation_precision4fp32E) |
|     function)](api/languages/c    | -   [                             |
| pp_api.html#_CPPv4N5cudaq4orca6sa | cudaq::simulation_precision::fp64 |
| mpleERNSt6vectorINSt6size_tEEERNS |     (C++                          |
| t6vectorINSt6size_tEEERNSt6vector |     enumerator)](api              |
| IdEERNSt6vectorIdEEiNSt6size_tE), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]]                       | udaq20simulation_precision4fp64E) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::SimulationState (C++  |
| 4N5cudaq4orca6sampleERNSt6vectorI |     c                             |
| NSt6size_tEEERNSt6vectorINSt6size | lass)](api/languages/cpp_api.html |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | #_CPPv4N5cudaq15SimulationStateE) |
| -   [cudaq::orca::sample_async    | -   [                             |
|     (C++                          | cudaq::SimulationState::precision |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     enum)](api                    |
| html#_CPPv4N5cudaq4orca12sample_a | /languages/cpp_api.html#_CPPv4N5c |
| syncERNSt6vectorINSt6size_tEEERNS | udaq15SimulationState9precisionE) |
| t6vectorINSt6size_tEEERNSt6vector | -   [cudaq:                       |
| IdEERNSt6vectorIdEEiNSt6size_tE), | :SimulationState::precision::fp32 |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     enumerator)](api/lang         |
| q4orca12sample_asyncERNSt6vectorI | uages/cpp_api.html#_CPPv4N5cudaq1 |
| NSt6size_tEEERNSt6vectorINSt6size | 5SimulationState9precision4fp32E) |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | -   [cudaq:                       |
| -   [cudaq::OrcaRemoteRESTQPU     | :SimulationState::precision::fp64 |
|     (C++                          |     (C++                          |
|     cla                           |     enumerator)](api/lang         |
| ss)](api/languages/cpp_api.html#_ | uages/cpp_api.html#_CPPv4N5cudaq1 |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | 5SimulationState9precision4fp64E) |
| -   [cudaq::other_policies (C++   | -                                 |
|     s                             |   [cudaq::SimulationState::Tensor |
| truct)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14other_policiesE) |     struct)](                     |
| -   [cudaq::PasqalRemoteRESTQPU   | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq15SimulationState6TensorE) |
|     class                         | -   [cudaq::spin_handler (C++     |
| )](api/languages/cpp_api.html#_CP |                                   |
| Pv4N5cudaq19PasqalRemoteRESTQPUE) |   class)](api/languages/cpp_api.h |
| -   [cudaq::pauli1 (C++           | tml#_CPPv4N5cudaq12spin_handlerE) |
|     class)](api/languages/cp      | -   [cudaq:                       |
| p_api.html#_CPPv4N5cudaq6pauli1E) | :spin_handler::to_diagonal_matrix |
| -                                 |     (C++                          |
|    [cudaq::pauli1::num_parameters |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4NK5cud |
|     member)]                      | aq12spin_handler18to_diagonal_mat |
| (api/languages/cpp_api.html#_CPPv | rixERNSt13unordered_mapINSt6size_ |
| 4N5cudaq6pauli114num_parametersE) | tENSt7int64_tEEERKNSt13unordered_ |
| -   [cudaq::pauli1::num_targets   | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -                                 |
|     membe                         |   [cudaq::spin_handler::to_matrix |
| r)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq6pauli111num_targetsE) |     function                      |
| -   [cudaq::pauli1::pauli1 (C++   | )](api/languages/cpp_api.html#_CP |
|     function)](api/languages/cpp_ | Pv4N5cudaq12spin_handler9to_matri |
| api.html#_CPPv4N5cudaq6pauli16pau | xERKNSt6stringENSt7complexIdEEb), |
| li1ERKNSt6vectorIN5cudaq4realEEE) |     [\[1                          |
| -   [cudaq::pauli2 (C++           | \]](api/languages/cpp_api.html#_C |
|     class)](api/languages/cp      | PPv4NK5cudaq12spin_handler9to_mat |
| p_api.html#_CPPv4N5cudaq6pauli2E) | rixERNSt13unordered_mapINSt6size_ |
| -                                 | tENSt7int64_tEEERKNSt13unordered_ |
|    [cudaq::pauli2::num_parameters | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [cuda                         |
|     member)]                      | q::spin_handler::to_sparse_matrix |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq6pauli214num_parametersE) |     function)](api/               |
| -   [cudaq::pauli2::num_targets   | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq12spin_handler16to_sparse_matr |
|     membe                         | ixERKNSt6stringENSt7complexIdEEb) |
| r)](api/languages/cpp_api.html#_C | -                                 |
| PPv4N5cudaq6pauli211num_targetsE) |   [cudaq::spin_handler::to_string |
| -   [cudaq::pauli2::pauli2 (C++   |     (C++                          |
|     function)](api/languages/cpp_ |     function)](ap                 |
| api.html#_CPPv4N5cudaq6pauli26pau | i/languages/cpp_api.html#_CPPv4NK |
| li2ERKNSt6vectorIN5cudaq4realEEE) | 5cudaq12spin_handler9to_stringEb) |
| -   [cudaq::phase_damping (C++    | -                                 |
|                                   |   [cudaq::spin_handler::unique_id |
|  class)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13phase_dampingE) |     function)](ap                 |
| -   [cud                          | i/languages/cpp_api.html#_CPPv4NK |
| aq::phase_damping::num_parameters | 5cudaq12spin_handler9unique_idEv) |
|     (C++                          | -   [cudaq::spin_op (C++          |
|     member)](api/lan              |     type)](api/languages/cpp      |
| guages/cpp_api.html#_CPPv4N5cudaq | _api.html#_CPPv4N5cudaq7spin_opE) |
| 13phase_damping14num_parametersE) | -   [cudaq::spin_op_term (C++     |
| -   [                             |                                   |
| cudaq::phase_damping::num_targets |    type)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq12spin_op_termE) |
|     member)](api/                 | -   [cudaq::state (C++            |
| languages/cpp_api.html#_CPPv4N5cu |     class)](api/languages/c       |
| daq13phase_damping11num_targetsE) | pp_api.html#_CPPv4N5cudaq5stateE) |
| -   [cudaq::phase_flip_channel    | -   [cudaq::state::amplitude (C++ |
|     (C++                          |     function)](api/lang           |
|     clas                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
| s)](api/languages/cpp_api.html#_C | state9amplitudeERKNSt6vectorIiEE) |
| PPv4N5cudaq18phase_flip_channelE) | -   [cudaq::state::amplitudes     |
| -   [cudaq::p                     |     (C++                          |
| hase_flip_channel::num_parameters |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     member)](api/language         | tml#_CPPv4N5cudaq5state10amplitud |
| s/cpp_api.html#_CPPv4N5cudaq18pha | esERKNSt6vectorINSt6vectorIiEEEE) |
| se_flip_channel14num_parametersE) | -   [cudaq::state::dump (C++      |
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
