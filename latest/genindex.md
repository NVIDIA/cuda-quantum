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
|     cudaq.con                     |                                   |
| trib)](api/languages/python_api.h |                                   |
| tml#cudaq.contrib.angular_encode) |                                   |
| -   [annotations (cudaq.DEMResult |                                   |
|     pro                           |                                   |
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
| -   [canonicalize                 | -   [cudaq::product_op::begin     |
|     (cu                           |     (C++                          |
| daq.operators.boson.BosonOperator |     functio                       |
|     attribute)](api/languages     | n)](api/languages/cpp_api.html#_C |
| /python_api.html#cudaq.operators. | PPv4NK5cudaq10product_op5beginEv) |
| boson.BosonOperator.canonicalize) | -                                 |
|     -   [(cudaq.                  |  [cudaq::product_op::canonicalize |
| operators.boson.BosonOperatorTerm |     (C++                          |
|                                   |     func                          |
|     attribute)](api/languages/pyt | tion)](api/languages/cpp_api.html |
| hon_api.html#cudaq.operators.boso | #_CPPv4N5cudaq10product_op12canon |
| n.BosonOperatorTerm.canonicalize) | icalizeERKNSt3setINSt6size_tEEE), |
|     -   [(cudaq.                  |     [\[1\]](api                   |
| operators.fermion.FermionOperator | /languages/cpp_api.html#_CPPv4N5c |
|                                   | udaq10product_op12canonicalizeEv) |
|     attribute)](api/languages/pyt | -   [                             |
| hon_api.html#cudaq.operators.ferm | cudaq::product_op::const_iterator |
| ion.FermionOperator.canonicalize) |     (C++                          |
|     -   [(cudaq.oper              |     struct)](api/                 |
| ators.fermion.FermionOperatorTerm | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq10product_op14const_iteratorE) |
| attribute)](api/languages/python_ | -   [cudaq::product_o             |
| api.html#cudaq.operators.fermion. | p::const_iterator::const_iterator |
| FermionOperatorTerm.canonicalize) |     (C++                          |
|     -                             |     fu                            |
|  [(cudaq.operators.MatrixOperator | nction)](api/languages/cpp_api.ht |
|         attribute)](api/lang      | ml#_CPPv4N5cudaq10product_op14con |
| uages/python_api.html#cudaq.opera | st_iterator14const_iteratorEPK10p |
| tors.MatrixOperator.canonicalize) | roduct_opI9HandlerTyENSt6size_tE) |
|     -   [(c                       | -   [cudaq::produ                 |
| udaq.operators.MatrixOperatorTerm | ct_op::const_iterator::operator!= |
|         attribute)](api/language  |     (C++                          |
| s/python_api.html#cudaq.operators |     fun                           |
| .MatrixOperatorTerm.canonicalize) | ction)](api/languages/cpp_api.htm |
|     -   [(                        | l#_CPPv4NK5cudaq10product_op14con |
| cudaq.operators.spin.SpinOperator | st_iteratorneERK14const_iterator) |
|         attribute)](api/languag   | -   [cudaq::produ                 |
| es/python_api.html#cudaq.operator | ct_op::const_iterator::operator\* |
| s.spin.SpinOperator.canonicalize) |     (C++                          |
|     -   [(cuda                    |     function)](api/lang           |
| q.operators.spin.SpinOperatorTerm | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 10product_op14const_iteratormlEv) |
|       attribute)](api/languages/p | -   [cudaq::produ                 |
| ython_api.html#cudaq.operators.sp | ct_op::const_iterator::operator++ |
| in.SpinOperatorTerm.canonicalize) |     (C++                          |
| -   [captured_variables()         |     function)](api/lang           |
|     (cudaq.PyKernelDecorator      | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     method)](api/lan              | 0product_op14const_iteratorppEi), |
| guages/python_api.html#cudaq.PyKe |     [\[1\]](api/lan               |
| rnelDecorator.captured_variables) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [CentralDifference (class in  | 10product_op14const_iteratorppEv) |
|     cudaq.gradients)              | -   [cudaq::produc                |
| ](api/languages/python_api.html#c | t_op::const_iterator::operator\-- |
| udaq.gradients.CentralDifference) |     (C++                          |
| -   [channel                      |     function)](api/lang           |
|     (cudaq.ptsbe.TraceInstruction | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     property)](a                  | 0product_op14const_iteratormmEi), |
| pi/languages/python_api.html#cuda |     [\[1\]](api/lan               |
| q.ptsbe.TraceInstruction.channel) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [circuit_location             | 10product_op14const_iteratormmEv) |
|     (cudaq.ptsbe.KrausSelection   | -   [cudaq::produc                |
|     property)](api/lang           | t_op::const_iterator::operator-\> |
| uages/python_api.html#cudaq.ptsbe |     (C++                          |
| .KrausSelection.circuit_location) |     function)](api/lan            |
| -   [clear (cudaq.Resources       | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 10product_op14const_iteratorptEv) |
|   attribute)](api/languages/pytho | -   [cudaq::produ                 |
| n_api.html#cudaq.Resources.clear) | ct_op::const_iterator::operator== |
|     -   [(cudaq.SampleResult      |     (C++                          |
|         a                         |     fun                           |
| ttribute)](api/languages/python_a | ction)](api/languages/cpp_api.htm |
| pi.html#cudaq.SampleResult.clear) | l#_CPPv4NK5cudaq10product_op14con |
| -   [COBYLA (class in             | st_iteratoreqERK14const_iterator) |
|     cudaq.o                       | -   [cudaq::product_op::degrees   |
| ptimizers)](api/languages/python_ |     (C++                          |
| api.html#cudaq.optimizers.COBYLA) |     function)                     |
| -   [coefficient                  | ](api/languages/cpp_api.html#_CPP |
|     (cudaq.                       | v4NK5cudaq10product_op7degreesEv) |
| operators.boson.BosonOperatorTerm | -   [cudaq::product_op::dump (C++ |
|     property)](api/languages/py   |     functi                        |
| thon_api.html#cudaq.operators.bos | on)](api/languages/cpp_api.html#_ |
| on.BosonOperatorTerm.coefficient) | CPPv4NK5cudaq10product_op4dumpEv) |
|     -   [(cudaq.oper              | -   [cudaq::product_op::end (C++  |
| ators.fermion.FermionOperatorTerm |     funct                         |
|                                   | ion)](api/languages/cpp_api.html# |
|   property)](api/languages/python | _CPPv4NK5cudaq10product_op3endEv) |
| _api.html#cudaq.operators.fermion | -   [c                            |
| .FermionOperatorTerm.coefficient) | udaq::product_op::get_coefficient |
|     -   [(c                       |     (C++                          |
| udaq.operators.MatrixOperatorTerm |     function)](api/lan            |
|         property)](api/languag    | guages/cpp_api.html#_CPPv4NK5cuda |
| es/python_api.html#cudaq.operator | q10product_op15get_coefficientEv) |
| s.MatrixOperatorTerm.coefficient) | -                                 |
|     -   [(cuda                    |   [cudaq::product_op::get_term_id |
| q.operators.spin.SpinOperatorTerm |     (C++                          |
|         property)](api/languages/ |     function)](api                |
| python_api.html#cudaq.operators.s | /languages/cpp_api.html#_CPPv4NK5 |
| pin.SpinOperatorTerm.coefficient) | cudaq10product_op11get_term_idEv) |
| -   [col_count                    | -                                 |
|     (cudaq.KrausOperator          |   [cudaq::product_op::is_identity |
|     prope                         |     (C++                          |
| rty)](api/languages/python_api.ht |     function)](api                |
| ml#cudaq.KrausOperator.col_count) | /languages/cpp_api.html#_CPPv4NK5 |
| -   [compile()                    | cudaq10product_op11is_identityEv) |
|     (cudaq.PyKernelDecorator      | -   [cudaq::product_op::num_ops   |
|     metho                         |     (C++                          |
| d)](api/languages/python_api.html |     function)                     |
| #cudaq.PyKernelDecorator.compile) | ](api/languages/cpp_api.html#_CPP |
| -   [compiledModuleCache()        | v4NK5cudaq10product_op7num_opsEv) |
|     (cudaq.PyKernelDecorator      | -                                 |
|     method)](api/lang             |    [cudaq::product_op::operator\* |
| uages/python_api.html#cudaq.PyKer |     (C++                          |
| nelDecorator.compiledModuleCache) |     function)](api/languages/     |
| -   [ComplexMatrix (class in      | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     cudaq)](api/languages/pyt     | oduct_opmlE10product_opI1TERK15sc |
| hon_api.html#cudaq.ComplexMatrix) | alar_operatorRK10product_opI1TE), |
| -   [compute                      |     [\[1\]](api/languages/        |
|     (                             | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| cudaq.gradients.CentralDifference | oduct_opmlE10product_opI1TERK15sc |
|     attribute)](api/la            | alar_operatorRR10product_opI1TE), |
| nguages/python_api.html#cudaq.gra |     [\[2\]](api/languages/        |
| dients.CentralDifference.compute) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(                        | oduct_opmlE10product_opI1TERR15sc |
| cudaq.gradients.ForwardDifference | alar_operatorRK10product_opI1TE), |
|         attribute)](api/la        |     [\[3\]](api/languages/        |
| nguages/python_api.html#cudaq.gra | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| dients.ForwardDifference.compute) | oduct_opmlE10product_opI1TERR15sc |
|     -                             | alar_operatorRR10product_opI1TE), |
|  [(cudaq.gradients.ParameterShift |     [\[4\]](api/                  |
|         attribute)](api           | languages/cpp_api.html#_CPPv4I0EN |
| /languages/python_api.html#cudaq. | 5cudaq10product_opmlE6sum_opI1TER |
| gradients.ParameterShift.compute) | K15scalar_operatorRK6sum_opI1TE), |
| -   [const()                      |     [\[5\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|   (cudaq.operators.ScalarOperator | 5cudaq10product_opmlE6sum_opI1TER |
|     class                         | K15scalar_operatorRR6sum_opI1TE), |
|     method)](a                    |     [\[6\]](api/                  |
| pi/languages/python_api.html#cuda | languages/cpp_api.html#_CPPv4I0EN |
| q.operators.ScalarOperator.const) | 5cudaq10product_opmlE6sum_opI1TER |
| -   [controls                     | R15scalar_operatorRK6sum_opI1TE), |
|     (cudaq.ptsbe.TraceInstruction |     [\[7\]](api/                  |
|     property)](ap                 | languages/cpp_api.html#_CPPv4I0EN |
| i/languages/python_api.html#cudaq | 5cudaq10product_opmlE6sum_opI1TER |
| .ptsbe.TraceInstruction.controls) | R15scalar_operatorRR6sum_opI1TE), |
| -   [copy                         |     [\[8\]](api/languages         |
|     (cu                           | /cpp_api.html#_CPPv4NK5cudaq10pro |
| daq.operators.boson.BosonOperator | duct_opmlERK6sum_opI9HandlerTyE), |
|     attribute)](api/l             |     [\[9\]](api/languages/cpp_a   |
| anguages/python_api.html#cudaq.op | pi.html#_CPPv4NKR5cudaq10product_ |
| erators.boson.BosonOperator.copy) | opmlERK10product_opI9HandlerTyE), |
|     -   [(cudaq.                  |     [\[10\]](api/language         |
| operators.boson.BosonOperatorTerm | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|         attribute)](api/langu     | roduct_opmlERK15scalar_operator), |
| ages/python_api.html#cudaq.operat |     [\[11\]](api/languages/cpp_a  |
| ors.boson.BosonOperatorTerm.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [(cudaq.                  | opmlERR10product_opI9HandlerTyE), |
| operators.fermion.FermionOperator |     [\[12\]](api/language         |
|         attribute)](api/langu     | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ages/python_api.html#cudaq.operat | roduct_opmlERR15scalar_operator), |
| ors.fermion.FermionOperator.copy) |     [\[13\]](api/languages/cpp_   |
|     -   [(cudaq.oper              | api.html#_CPPv4NO5cudaq10product_ |
| ators.fermion.FermionOperatorTerm | opmlERK10product_opI9HandlerTyE), |
|         attribute)](api/languages |     [\[14\]](api/languag          |
| /python_api.html#cudaq.operators. | es/cpp_api.html#_CPPv4NO5cudaq10p |
| fermion.FermionOperatorTerm.copy) | roduct_opmlERK15scalar_operator), |
|     -                             |     [\[15\]](api/languages/cpp_   |
|  [(cudaq.operators.MatrixOperator | api.html#_CPPv4NO5cudaq10product_ |
|         attribute)](              | opmlERR10product_opI9HandlerTyE), |
| api/languages/python_api.html#cud |     [\[16\]](api/langua           |
| aq.operators.MatrixOperator.copy) | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     -   [(c                       | product_opmlERR15scalar_operator) |
| udaq.operators.MatrixOperatorTerm | -                                 |
|         attribute)](api/          |   [cudaq::product_op::operator\*= |
| languages/python_api.html#cudaq.o |     (C++                          |
| perators.MatrixOperatorTerm.copy) |     function)](api/languages/cpp  |
|     -   [(                        | _api.html#_CPPv4N5cudaq10product_ |
| cudaq.operators.spin.SpinOperator | opmLERK10product_opI9HandlerTyE), |
|         attribute)](api           |     [\[1\]](api/langua            |
| /languages/python_api.html#cudaq. | ges/cpp_api.html#_CPPv4N5cudaq10p |
| operators.spin.SpinOperator.copy) | roduct_opmLERK15scalar_operator), |
|     -   [(cuda                    |     [\[2\]](api/languages/cp      |
| q.operators.spin.SpinOperatorTerm | p_api.html#_CPPv4N5cudaq10product |
|         attribute)](api/lan       | _opmLERR10product_opI9HandlerTyE) |
| guages/python_api.html#cudaq.oper | -   [cudaq::product_op::operator+ |
| ators.spin.SpinOperatorTerm.copy) |     (C++                          |
| -   [count (cudaq.Resources       |     function)](api/langu          |
|                                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|   attribute)](api/languages/pytho | q10product_opplE6sum_opI1TERK15sc |
| n_api.html#cudaq.Resources.count) | alar_operatorRK10product_opI1TE), |
|     -   [(cudaq.SampleResult      |     [\[1\]](api/                  |
|         a                         | languages/cpp_api.html#_CPPv4I0EN |
| ttribute)](api/languages/python_a | 5cudaq10product_opplE6sum_opI1TER |
| pi.html#cudaq.SampleResult.count) | K15scalar_operatorRK6sum_opI1TE), |
| -   [count_controls               |     [\[2\]](api/langu             |
|     (cudaq.Resources              | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     attribu                       | q10product_opplE6sum_opI1TERK15sc |
| te)](api/languages/python_api.htm | alar_operatorRR10product_opI1TE), |
| l#cudaq.Resources.count_controls) |     [\[3\]](api/                  |
| -   [count_instructions           | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opplE6sum_opI1TER |
|   (cudaq.ptsbe.PTSBEExecutionData | K15scalar_operatorRR6sum_opI1TE), |
|     attribute)](api/languages/    |     [\[4\]](api/langu             |
| python_api.html#cudaq.ptsbe.PTSBE | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ExecutionData.count_instructions) | q10product_opplE6sum_opI1TERR15sc |
| -   [counts (cudaq.ObserveResult  | alar_operatorRK10product_opI1TE), |
|     att                           |     [\[5\]](api/                  |
| ribute)](api/languages/python_api | languages/cpp_api.html#_CPPv4I0EN |
| .html#cudaq.ObserveResult.counts) | 5cudaq10product_opplE6sum_opI1TER |
|     -   [(cudaq.SampleResult      | R15scalar_operatorRK6sum_opI1TE), |
|         p                         |     [\[6\]](api/langu             |
| roperty)](api/languages/python_ap | ages/cpp_api.html#_CPPv4I0EN5cuda |
| i.html#cudaq.SampleResult.counts) | q10product_opplE6sum_opI1TERR15sc |
| -   [csr_spmatrix (C++            | alar_operatorRR10product_opI1TE), |
|     type)](api/languages/c        |     [\[7\]](api/                  |
| pp_api.html#_CPPv412csr_spmatrix) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq                         | 5cudaq10product_opplE6sum_opI1TER |
|     -   [module](api/langua       | R15scalar_operatorRR6sum_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[8\]](api/languages/cpp_a   |
| -   [cudaq (C++                   | pi.html#_CPPv4NKR5cudaq10product_ |
|     type)](api/lan                | opplERK10product_opI9HandlerTyE), |
| guages/cpp_api.html#_CPPv45cudaq) |     [\[9\]](api/language          |
| -   [cudaq.apply_noise() (in      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     module                        | roduct_opplERK15scalar_operator), |
|     cudaq)](api/languages/python_ |     [\[10\]](api/languages/       |
| api.html#cudaq.cudaq.apply_noise) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   cudaq.boson                   | duct_opplERK6sum_opI9HandlerTyE), |
|     -   [module](api/languages/py |     [\[11\]](api/languages/cpp_a  |
| thon_api.html#module-cudaq.boson) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.fermion                 | opplERR10product_opI9HandlerTyE), |
|                                   |     [\[12\]](api/language         |
|   -   [module](api/languages/pyth | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| on_api.html#module-cudaq.fermion) | roduct_opplERR15scalar_operator), |
| -   cudaq.operators.custom        |     [\[13\]](api/languages/       |
|     -   [mo                       | cpp_api.html#_CPPv4NKR5cudaq10pro |
| dule](api/languages/python_api.ht | duct_opplERR6sum_opI9HandlerTyE), |
| ml#module-cudaq.operators.custom) |     [\[                           |
| -   cudaq.spin                    | 14\]](api/languages/cpp_api.html# |
|     -   [module](api/languages/p  | _CPPv4NKR5cudaq10product_opplEv), |
| ython_api.html#module-cudaq.spin) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::amplitude_damping     | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opplERK10product_opI9HandlerTyE), |
|     cla                           |     [\[16\]](api/languag          |
| ss)](api/languages/cpp_api.html#_ | es/cpp_api.html#_CPPv4NO5cudaq10p |
| CPPv4N5cudaq17amplitude_dampingE) | roduct_opplERK15scalar_operator), |
| -                                 |     [\[17\]](api/languages        |
| [cudaq::amplitude_damping_channel | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opplERK6sum_opI9HandlerTyE), |
|     class)](api                   |     [\[18\]](api/languages/cpp_   |
| /languages/cpp_api.html#_CPPv4N5c | api.html#_CPPv4NO5cudaq10product_ |
| udaq25amplitude_damping_channelE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::amplitud              |     [\[19\]](api/languag          |
| e_damping_channel::num_parameters | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opplERR15scalar_operator), |
|     member)](api/languages/cpp_a  |     [\[20\]](api/languages        |
| pi.html#_CPPv4N5cudaq25amplitude_ | /cpp_api.html#_CPPv4NO5cudaq10pro |
| damping_channel14num_parametersE) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cudaq::ampli                 |     [                             |
| tude_damping_channel::num_targets | \[21\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NO5cudaq10product_opplEv) |
|     member)](api/languages/cp     | -   [cudaq::product_op::operator- |
| p_api.html#_CPPv4N5cudaq25amplitu |     (C++                          |
| de_damping_channel11num_targetsE) |     function)](api/langu          |
| -   [cudaq::AnalogRemoteRESTQPU   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERK15sc |
|     class                         | alar_operatorRK10product_opI1TE), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/                  |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::apply_noise (C++      | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/               | K15scalar_operatorRK6sum_opI1TE), |
| languages/cpp_api.html#_CPPv4I0Dp |     [\[2\]](api/langu             |
| EN5cudaq11apply_noiseEvDpRR4Args) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::async_result (C++     | q10product_opmiE6sum_opI1TERK15sc |
|     c                             | alar_operatorRR10product_opI1TE), |
| lass)](api/languages/cpp_api.html |     [\[3\]](api/                  |
| #_CPPv4I0EN5cudaq12async_resultE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::async_result::get     | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRR6sum_opI1TE), |
|     functi                        |     [\[4\]](api/langu             |
| on)](api/languages/cpp_api.html#_ | ages/cpp_api.html#_CPPv4I0EN5cuda |
| CPPv4N5cudaq12async_result3getEv) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq::async_sample_result   | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[5\]](api/                  |
|     type                          | languages/cpp_api.html#_CPPv4I0EN |
| )](api/languages/cpp_api.html#_CP | 5cudaq10product_opmiE6sum_opI1TER |
| Pv4N5cudaq19async_sample_resultE) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::BaseRemoteRESTQPU     |     [\[6\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cla                           | q10product_opmiE6sum_opI1TERR15sc |
| ss)](api/languages/cpp_api.html#_ | alar_operatorRR10product_opI1TE), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [\[7\]](api/                  |
| -   [cudaq::bit_flip_channel (C++ | languages/cpp_api.html#_CPPv4I0EN |
|     cl                            | 5cudaq10product_opmiE6sum_opI1TER |
| ass)](api/languages/cpp_api.html# | R15scalar_operatorRR6sum_opI1TE), |
| _CPPv4N5cudaq16bit_flip_channelE) |     [\[8\]](api/languages/cpp_a   |
| -   [cudaq:                       | pi.html#_CPPv4NKR5cudaq10product_ |
| :bit_flip_channel::num_parameters | opmiERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[9\]](api/language          |
|     member)](api/langua           | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ges/cpp_api.html#_CPPv4N5cudaq16b | roduct_opmiERK15scalar_operator), |
| it_flip_channel14num_parametersE) |     [\[10\]](api/languages/       |
| -   [cud                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
| aq::bit_flip_channel::num_targets | duct_opmiERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     member)](api/lan              | pi.html#_CPPv4NKR5cudaq10product_ |
| guages/cpp_api.html#_CPPv4N5cudaq | opmiERR10product_opI9HandlerTyE), |
| 16bit_flip_channel11num_targetsE) |     [\[12\]](api/language         |
| -   [cudaq::boson_handler (C++    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opmiERR15scalar_operator), |
|  class)](api/languages/cpp_api.ht |     [\[13\]](api/languages/       |
| ml#_CPPv4N5cudaq13boson_handlerE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::boson_op (C++         | duct_opmiERR6sum_opI9HandlerTyE), |
|     type)](api/languages/cpp_     |     [\[                           |
| api.html#_CPPv4N5cudaq8boson_opE) | 14\]](api/languages/cpp_api.html# |
| -   [cudaq::boson_op_term (C++    | _CPPv4NKR5cudaq10product_opmiEv), |
|                                   |     [\[15\]](api/languages/cpp_   |
|   type)](api/languages/cpp_api.ht | api.html#_CPPv4NO5cudaq10product_ |
| ml#_CPPv4N5cudaq13boson_op_termE) | opmiERK10product_opI9HandlerTyE), |
| -   [cudaq::CodeGenConfig (C++    |     [\[16\]](api/languag          |
|                                   | es/cpp_api.html#_CPPv4NO5cudaq10p |
| struct)](api/languages/cpp_api.ht | roduct_opmiERK15scalar_operator), |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     [\[17\]](api/languages        |
| -   [cudaq::commutation_relations | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERK6sum_opI9HandlerTyE), |
|     struct)]                      |     [\[18\]](api/languages/cpp_   |
| (api/languages/cpp_api.html#_CPPv | api.html#_CPPv4NO5cudaq10product_ |
| 4N5cudaq21commutation_relationsE) | opmiERR10product_opI9HandlerTyE), |
| -   [cudaq::complex (C++          |     [\[19\]](api/languag          |
|     type)](api/languages/cpp      | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _api.html#_CPPv4N5cudaq7complexE) | roduct_opmiERR15scalar_operator), |
| -   [cudaq::complex_matrix (C++   |     [\[20\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq10pro |
| class)](api/languages/cpp_api.htm | duct_opmiERR6sum_opI9HandlerTyE), |
| l#_CPPv4N5cudaq14complex_matrixE) |     [                             |
| -                                 | \[21\]](api/languages/cpp_api.htm |
|   [cudaq::complex_matrix::adjoint | l#_CPPv4NO5cudaq10product_opmiEv) |
|     (C++                          | -   [cudaq::product_op::operator/ |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/language       |
| 5cudaq14complex_matrix7adjointEv) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::                      | roduct_opdvERK15scalar_operator), |
| complex_matrix::diagonal_elements |     [\[1\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/languages      | roduct_opdvERR15scalar_operator), |
| /cpp_api.html#_CPPv4NK5cudaq14com |     [\[2\]](api/languag           |
| plex_matrix17diagonal_elementsEi) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::complex_matrix::dump  | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[3\]](api/langua            |
|     function)](api/language       | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| s/cpp_api.html#_CPPv4NK5cudaq14co | product_opdvERR15scalar_operator) |
| mplex_matrix4dumpERNSt7ostreamE), | -                                 |
|     [\[1\]]                       |    [cudaq::product_op::operator/= |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4NK5cudaq14complex_matrix4dumpEv) |     function)](api/langu          |
| -   [c                            | ages/cpp_api.html#_CPPv4N5cudaq10 |
| udaq::complex_matrix::eigenvalues | product_opdVERK15scalar_operator) |
|     (C++                          | -   [cudaq::product_op::operator= |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4NK5cuda |     function)](api/l              |
| q14complex_matrix11eigenvaluesEv) | anguages/cpp_api.html#_CPPv4I00EN |
| -   [cu                           | 5cudaq10product_opaSER10product_o |
| daq::complex_matrix::eigenvectors | pI9HandlerTyERK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/languages/cpp     |
|     function)](api/lang           | _api.html#_CPPv4N5cudaq10product_ |
| uages/cpp_api.html#_CPPv4NK5cudaq | opaSERK10product_opI9HandlerTyE), |
| 14complex_matrix12eigenvectorsEv) |     [\[2\]](api/languages/cp      |
| -   [c                            | p_api.html#_CPPv4N5cudaq10product |
| udaq::complex_matrix::exponential | _opaSERR10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     function)](api/la             |    [cudaq::product_op::operator== |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14complex_matrix11exponentialEv) |     function)](api/languages/cpp  |
| -                                 | _api.html#_CPPv4NK5cudaq10product |
|  [cudaq::complex_matrix::identity | _opeqERK10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     function)](api/languages      |  [cudaq::product_op::operator\[\] |
| /cpp_api.html#_CPPv4N5cudaq14comp |     (C++                          |
| lex_matrix8identityEKNSt6size_tE) |     function)](ap                 |
| -                                 | i/languages/cpp_api.html#_CPPv4NK |
| [cudaq::complex_matrix::kronecker | 5cudaq10product_opixENSt6size_tE) |
|     (C++                          | -                                 |
|     function)](api/lang           |    [cudaq::product_op::product_op |
| uages/cpp_api.html#_CPPv4I00EN5cu |     (C++                          |
| daq14complex_matrix9kroneckerE14c |     f                             |
| omplex_matrix8Iterable8Iterable), | unction)](api/languages/cpp_api.h |
|     [\[1\]](api/l                 | tml#_CPPv4I00EN5cudaq10product_op |
| anguages/cpp_api.html#_CPPv4N5cud | 10product_opERK10product_opI1TE), |
| aq14complex_matrix9kroneckerERK14 |     [\[1\]]                       |
| complex_matrixRK14complex_matrix) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::c                     | 4I00EN5cudaq10product_op10product |
| omplex_matrix::minimal_eigenvalue | _opERK10product_opI1TERKN14matrix |
|     (C++                          | _handler20commutation_behaviorE), |
|     function)](api/languages/     |                                   |
| cpp_api.html#_CPPv4NK5cudaq14comp |   [\[2\]](api/languages/cpp_api.h |
| lex_matrix18minimal_eigenvalueEv) | tml#_CPPv4N5cudaq10product_op10pr |
| -   [                             | oduct_opENSt6size_tENSt6size_tE), |
| cudaq::complex_matrix::operator() |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     function)](api/languages/cpp  | _op10product_opENSt7complexIdEE), |
| _api.html#_CPPv4N5cudaq14complex_ |     [\[4\]](api/l                 |
| matrixclENSt6size_tENSt6size_tE), | anguages/cpp_api.html#_CPPv4N5cud |
|     [\[1\]](api/languages/cpp     | aq10product_op10product_opERK10pr |
| _api.html#_CPPv4NK5cudaq14complex | oduct_opI9HandlerTyENSt6size_tE), |
| _matrixclENSt6size_tENSt6size_tE) |     [\[5\]](api/l                 |
| -   [                             | anguages/cpp_api.html#_CPPv4N5cud |
| cudaq::complex_matrix::operator\* | aq10product_op10product_opERR10pr |
|     (C++                          | oduct_opI9HandlerTyENSt6size_tE), |
|     function)](api/langua         |     [\[6\]](api/languages         |
| ges/cpp_api.html#_CPPv4N5cudaq14c | /cpp_api.html#_CPPv4N5cudaq10prod |
| omplex_matrixmlEN14complex_matrix | uct_op10product_opERR9HandlerTy), |
| 10value_typeERK14complex_matrix), |     [\[7\]](ap                    |
|     [\[1\]                        | i/languages/cpp_api.html#_CPPv4N5 |
| ](api/languages/cpp_api.html#_CPP | cudaq10product_op10product_opEd), |
| v4N5cudaq14complex_matrixmlERK14c |     [\[8\]](a                     |
| omplex_matrixRK14complex_matrix), | pi/languages/cpp_api.html#_CPPv4N |
|                                   | 5cudaq10product_op10product_opEv) |
|  [\[2\]](api/languages/cpp_api.ht | -   [cuda                         |
| ml#_CPPv4N5cudaq14complex_matrixm | q::product_op::to_diagonal_matrix |
| lERK14complex_matrixRKNSt6vectorI |     (C++                          |
| N14complex_matrix10value_typeEEE) |     function)](api/               |
| -                                 | languages/cpp_api.html#_CPPv4NK5c |
| [cudaq::complex_matrix::operator+ | udaq10product_op18to_diagonal_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function                      | ENSt7int64_tEEERKNSt13unordered_m |
| )](api/languages/cpp_api.html#_CP | apINSt6stringENSt7complexIdEEEEb) |
| Pv4N5cudaq14complex_matrixplERK14 | -   [cudaq::product_op::to_matrix |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -                                 |     funct                         |
| [cudaq::complex_matrix::operator- | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NK5cudaq10product_op9to_mat |
|     function                      | rixENSt13unordered_mapINSt6size_t |
| )](api/languages/cpp_api.html#_CP | ENSt7int64_tEEERKNSt13unordered_m |
| Pv4N5cudaq14complex_matrixmiERK14 | apINSt6stringENSt7complexIdEEEEb) |
| complex_matrixRK14complex_matrix) | -   [cu                           |
| -   [cu                           | daq::product_op::to_sparse_matrix |
| daq::complex_matrix::operator\[\] |     (C++                          |
|     (C++                          |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|  function)](api/languages/cpp_api | 5cudaq10product_op16to_sparse_mat |
| .html#_CPPv4N5cudaq14complex_matr | rixENSt13unordered_mapINSt6size_t |
| ixixERKNSt6vectorINSt6size_tEEE), | ENSt7int64_tEEERKNSt13unordered_m |
|     [\[1\]](api/languages/cpp_api | apINSt6stringENSt7complexIdEEEEb) |
| .html#_CPPv4NK5cudaq14complex_mat | -   [cudaq::product_op::to_string |
| rixixERKNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq::complex_matrix::power |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)]                    | NK5cudaq10product_op9to_stringEv) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq14complex_matrix5powerEi) |  [cudaq::product_op::\~product_op |
| -                                 |     (C++                          |
|  [cudaq::complex_matrix::set_zero |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function)](ap                 | ml#_CPPv4N5cudaq10product_opD0Ev) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq::ptsbe (C++            |
| cudaq14complex_matrix8set_zeroEv) |     type)](api/languages/c        |
| -                                 | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| [cudaq::complex_matrix::to_string | -   [cudaq::p                     |
|     (C++                          | tsbe::ConditionalSamplingStrategy |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     class)](api/languag           |
| udaq14complex_matrix9to_stringEv) | es/cpp_api.html#_CPPv4N5cudaq5pts |
| -   [                             | be27ConditionalSamplingStrategyE) |
| cudaq::complex_matrix::value_type | -   [cudaq::ptsbe::C              |
|     (C++                          | onditionalSamplingStrategy::clone |
|     type)](api/                   |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq14complex_matrix10value_typeE) |    function)](api/languages/cpp_a |
| -   [cudaq::contrib (C++          | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
|     type)](api/languages/cpp      | ditionalSamplingStrategy5cloneEv) |
| _api.html#_CPPv4N5cudaq7contribE) | -   [cuda                         |
| -                                 | q::ptsbe::ConditionalSamplingStra |
| [cudaq::contrib::amplitude_encode | tegy::ConditionalSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api/lang           |
| s/cpp_api.html#_CPPv4N5cudaq7cont | uages/cpp_api.html#_CPPv4N5cudaq5 |
| rib16amplitude_encodeENSt4spanIKN | ptsbe27ConditionalSamplingStrateg |
| St7complexIdEEEENSt7complexIdEE), | y27ConditionalSamplingStrategyE19 |
|     [\[1\]](api/language          | TrajectoryPredicateNSt8uint64_tE) |
| s/cpp_api.html#_CPPv4N5cudaq7cont | -                                 |
| rib16amplitude_encodeENSt4spanIKN |   [cudaq::ptsbe::ConditionalSampl |
| St7complexIfEEEENSt7complexIdEE), | ingStrategy::generateTrajectories |
|     [\[2\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/language       |
| v4N5cudaq7contrib16amplitude_enco | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| deENSt4spanIKdEENSt7complexIdEE), | be27ConditionalSamplingStrategy20 |
|     [\[3\]                        | generateTrajectoriesENSt4spanIKN6 |
| ](api/languages/cpp_api.html#_CPP | detail10NoisePointEEENSt6size_tE) |
| v4N5cudaq7contrib16amplitude_enco | -   [cudaq::ptsbe::               |
| deENSt4spanIKfEENSt7complexIdEE), | ConditionalSamplingStrategy::name |
|                                   |     (C++                          |
| [\[4\]](api/languages/cpp_api.htm |     function)](api/languages/cpp_ |
| l#_CPPv4N5cudaq7contrib16amplitud | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| e_encodeERK5stateNSt7complexIdEE) | nditionalSamplingStrategy4nameEv) |
| -                                 | -   [cudaq:                       |
|   [cudaq::contrib::angular_encode | :ptsbe::ConditionalSamplingStrate |
|     (C++                          | gy::\~ConditionalSamplingStrategy |
|                                   |     (C++                          |
|  function)](api/languages/cpp_api |     function)](api/languages/     |
| .html#_CPPv4I0EN5cudaq7contrib14a | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| ngular_encodeEvRR6KernelR10QuakeV | 7ConditionalSamplingStrategyD0Ev) |
| alueNSt4spanIKdEE12RotationAxis), | -                                 |
|     [\[1\]](api/languages/cpp_api | [cudaq::ptsbe::detail::NoisePoint |
| .html#_CPPv4I0EN5cudaq7contrib14a |     (C++                          |
| ngular_encodeEvRR6KernelR10QuakeV |     struct)](a                    |
| alueR10QuakeValue12RotationAxis), | pi/languages/cpp_api.html#_CPPv4N |
|                                   | 5cudaq5ptsbe6detail10NoisePointE) |
|   [\[2\]](api/languages/cpp_api.h | -   [cudaq::p                     |
| tml#_CPPv4I0EN5cudaq7contrib14ang | tsbe::detail::NoisePoint::channel |
| ular_encodeEvRR6KernelR10QuakeVal |     (C++                          |
| ueRKNSt6vectorIdEE12RotationAxis) |     member)](api/langu            |
| -   [cudaq::contrib::draw (C++    | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     function)                     | tsbe6detail10NoisePoint7channelE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::ptsbe::det            |
| v4I0DpEN5cudaq7contrib4drawENSt6s | ail::NoisePoint::circuit_location |
| tringERR13QuantumKernelDpRR4Args) |     (C++                          |
| -                                 |     member)](api/languages/cpp_a  |
| [cudaq::contrib::get_unitary_cmat | pi.html#_CPPv4N5cudaq5ptsbe6detai |
|     (C++                          | l10NoisePoint16circuit_locationE) |
|     function)](api/languages/cp   | -   [cudaq::p                     |
| p_api.html#_CPPv4I0DpEN5cudaq7con | tsbe::detail::NoisePoint::op_name |
| trib16get_unitary_cmatE14complex_ |     (C++                          |
| matrixRR13QuantumKernelDpRR4Args) |     member)](api/langu            |
| -   [cudaq::contrib::RotationAxis | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     (C++                          | tsbe6detail10NoisePoint7op_nameE) |
|     enum)                         | -   [cudaq::                      |
| ](api/languages/cpp_api.html#_CPP | ptsbe::detail::NoisePoint::qubits |
| v4N5cudaq7contrib12RotationAxisE) |     (C++                          |
| -                                 |     member)](api/lang             |
|  [cudaq::contrib::RotationAxis::X | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     (C++                          | ptsbe6detail10NoisePoint6qubitsE) |
|     enumerator)](                 | -   [cudaq::                      |
| api/languages/cpp_api.html#_CPPv4 | ptsbe::ExhaustiveSamplingStrategy |
| N5cudaq7contrib12RotationAxis1XE) |     (C++                          |
| -                                 |     class)](api/langua            |
|  [cudaq::contrib::RotationAxis::Y | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     (C++                          | sbe26ExhaustiveSamplingStrategyE) |
|     enumerator)](                 | -   [cudaq::ptsbe::               |
| api/languages/cpp_api.html#_CPPv4 | ExhaustiveSamplingStrategy::clone |
| N5cudaq7contrib12RotationAxis1YE) |     (C++                          |
| -                                 |     function)](api/languages/cpp_ |
|  [cudaq::contrib::RotationAxis::Z | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
|     (C++                          | haustiveSamplingStrategy5cloneEv) |
|     enumerator)](                 | -   [cu                           |
| api/languages/cpp_api.html#_CPPv4 | daq::ptsbe::ExhaustiveSamplingStr |
| N5cudaq7contrib12RotationAxis1ZE) | ategy::ExhaustiveSamplingStrategy |
| -   [cudaq::cudaq_json (C++       |     (C++                          |
|     class)](api/languages/cpp_api |     function)](api/la             |
| .html#_CPPv4N5cudaq10cudaq_jsonE) | nguages/cpp_api.html#_CPPv4N5cuda |
| -   [cudaq::DefaultQPU (C++       | q5ptsbe26ExhaustiveSamplingStrate |
|     class)](api/languages/cpp_api | gy26ExhaustiveSamplingStrategyEv) |
| .html#_CPPv4N5cudaq10DefaultQPUE) | -                                 |
| -   [cudaq::dem_from_kernel (C++  |    [cudaq::ptsbe::ExhaustiveSampl |
|     function)](api                | ingStrategy::generateTrajectories |
| /languages/cpp_api.html#_CPPv4I0D |     (C++                          |
| pEN5cudaq15dem_from_kernelENSt6st |     function)](api/languag        |
| ringERR13QuantumKernelDpRR4Args), | es/cpp_api.html#_CPPv4NK5cudaq5pt |
|     [                             | sbe26ExhaustiveSamplingStrategy20 |
| \[1\]](api/languages/cpp_api.html | generateTrajectoriesENSt4spanIKN6 |
| #_CPPv4I0DpEN5cudaq15dem_from_ker | detail10NoisePointEEENSt6size_tE) |
| nelENSt6stringERR13QuantumKernelP | -   [cudaq::ptsbe:                |
| KN5cudaq11noise_modelEDpRR4Args), | :ExhaustiveSamplingStrategy::name |
|     [\[2\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4I0DpEN5cudaq15de |     function)](api/languages/cpp  |
| m_from_kernelENSt6stringERR13Quan | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| tumKernelPKN5cudaq11noise_modelER | xhaustiveSamplingStrategy4nameEv) |
| KN5cudaq11dem_optionsEDpRR4Args), | -   [cuda                         |
|     [\[3\]](ap                    | q::ptsbe::ExhaustiveSamplingStrat |
| i/languages/cpp_api.html#_CPPv4I0 | egy::\~ExhaustiveSamplingStrategy |
| DpEN5cudaq15dem_from_kernelENSt6s |     (C++                          |
| tringERR13QuantumKernelPKN5cudaq1 |     function)](api/languages      |
| 1noise_modelERKN5cudaq11dem_optio | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| nsERN5cudaq15M2DSparseMatrixERN5c | 26ExhaustiveSamplingStrategyD0Ev) |
| udaq15M2OSparseMatrixEDpRR4Args), | -   [cuda                         |
|     [\[4\]](api/language          | q::ptsbe::OrderedSamplingStrategy |
| s/cpp_api.html#_CPPv4I0DpEN5cudaq |     (C++                          |
| 15dem_from_kernelENSt6stringERR13 |     class)](api/lan               |
| QuantumKernelPKN5cudaq11noise_mod | guages/cpp_api.html#_CPPv4N5cudaq |
| elERN5cudaq15M2DSparseMatrixERN5c | 5ptsbe23OrderedSamplingStrategyE) |
| udaq15M2OSparseMatrixEDpRR4Args), | -   [cudaq::ptsb                  |
|     [\[5\]](api/languages/cpp_api | e::OrderedSamplingStrategy::clone |
| .html#_CPPv4I0DpEN5cudaq15dem_fro |     (C++                          |
| m_kernelENSt6stringERR13QuantumKe |     function)](api/languages/c    |
| rnelRN5cudaq15M2DSparseMatrixERN5 | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| cudaq15M2OSparseMatrixEDpRR4Args) | 3OrderedSamplingStrategy5cloneEv) |
| -   [cudaq::dem_options (C++      | -   [cudaq::ptsbe::OrderedSampl   |
|                                   | ingStrategy::generateTrajectories |
|   struct)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq11dem_optionsE) |     function)](api/lang           |
| -   [cudaq::d                     | uages/cpp_api.html#_CPPv4NK5cudaq |
| em_options::allow_gauge_detectors | 5ptsbe23OrderedSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/language         | detail10NoisePointEEENSt6size_tE) |
| s/cpp_api.html#_CPPv4N5cudaq11dem | -   [cudaq::pts                   |
| _options21allow_gauge_detectorsE) | be::OrderedSamplingStrategy::name |
| -   [cudaq::dem_options::appr     |     (C++                          |
| oximate_disjoint_errors_threshold |     function)](api/languages/     |
|     (C++                          | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     memb                          | 23OrderedSamplingStrategy4nameEv) |
| er)](api/languages/cpp_api.html#_ | -                                 |
| CPPv4N5cudaq11dem_options37approx |    [cudaq::ptsbe::OrderedSampling |
| imate_disjoint_errors_thresholdE) | Strategy::OrderedSamplingStrategy |
| -   [cuda                         |     (C++                          |
| q::dem_options::block_decompositi |     function)](                   |
| on_from_introducing_remnant_edges | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5ptsbe23OrderedSamplingStr |
|     member)](api/lang             | ategy23OrderedSamplingStrategyEv) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -                                 |
| 1dem_options50block_decomposition |  [cudaq::ptsbe::OrderedSamplingSt |
| _from_introducing_remnant_edgesE) | rategy::\~OrderedSamplingStrategy |
| -   [cud                          |     (C++                          |
| aq::dem_options::decompose_errors |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     member)](api/lan              | sbe23OrderedSamplingStrategyD0Ev) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::pts                   |
| 11dem_options16decompose_errorsE) | be::ProbabilisticSamplingStrategy |
| -                                 |     (C++                          |
|   [cudaq::dem_options::fold_loops |     class)](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     member)](a                    | 29ProbabilisticSamplingStrategyE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::ptsbe::Pro            |
| 5cudaq11dem_options10fold_loopsE) | babilisticSamplingStrategy::clone |
| -   [cudaq::dem_optio             |     (C++                          |
| ns::ignore_decomposition_failures |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     member)](api/languages/cpp_ap | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| i.html#_CPPv4N5cudaq11dem_options | bilisticSamplingStrategy5cloneEv) |
| 29ignore_decomposition_failuresE) | -                                 |
| -   [cudaq::dem_opt               | [cudaq::ptsbe::ProbabilisticSampl |
| ions::return_measurement_matrices | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     member)](api/languages/cpp_   |     function)](api/languages/     |
| api.html#_CPPv4N5cudaq11dem_optio | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| ns27return_measurement_matricesE) | 29ProbabilisticSamplingStrategy20 |
| -   [cudaq::depolarization1 (C++  | generateTrajectoriesENSt4spanIKN6 |
|     c                             | detail10NoisePointEEENSt6size_tE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::ptsbe::Pr             |
| #_CPPv4N5cudaq15depolarization1E) | obabilisticSamplingStrategy::name |
| -   [cudaq::depolarization2 (C++  |     (C++                          |
|     c                             |                                   |
| lass)](api/languages/cpp_api.html |   function)](api/languages/cpp_ap |
| #_CPPv4N5cudaq15depolarization2E) | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| -   [cudaq:                       | abilisticSamplingStrategy4nameEv) |
| :depolarization2::depolarization2 | -   [cudaq::p                     |
|     (C++                          | tsbe::ProbabilisticSamplingStrate |
|     function)](api/languages/cp   | gy::ProbabilisticSamplingStrategy |
| p_api.html#_CPPv4N5cudaq15depolar |     (C++                          |
| ization215depolarization2EK4real) |     function)]                    |
| -   [cudaq                        | (api/languages/cpp_api.html#_CPPv |
| ::depolarization2::num_parameters | 4N5cudaq5ptsbe29ProbabilisticSamp |
|     (C++                          | lingStrategy29ProbabilisticSampli |
|     member)](api/langu            | ngStrategyENSt8optionalINSt8uint6 |
| ages/cpp_api.html#_CPPv4N5cudaq15 | 4_tEEENSt8optionalINSt6size_tEEE) |
| depolarization214num_parametersE) | -   [cudaq::pts                   |
| -   [cu                           | be::ProbabilisticSamplingStrategy |
| daq::depolarization2::num_targets | ::\~ProbabilisticSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api/languages/cp   |
| nguages/cpp_api.html#_CPPv4N5cuda | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| q15depolarization211num_targetsE) | robabilisticSamplingStrategyD0Ev) |
| -                                 | -                                 |
|    [cudaq::depolarization_channel | [cudaq::ptsbe::PTSBEExecutionData |
|     (C++                          |     (C++                          |
|     class)](                      |     struct)](ap                   |
| api/languages/cpp_api.html#_CPPv4 | i/languages/cpp_api.html#_CPPv4N5 |
| N5cudaq22depolarization_channelE) | cudaq5ptsbe18PTSBEExecutionDataE) |
| -   [cudaq::depol                 | -   [cudaq::ptsbe::PTSBE          |
| arization_channel::num_parameters | ExecutionData::count_instructions |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     function)](api/l              |
| p_api.html#_CPPv4N5cudaq22depolar | anguages/cpp_api.html#_CPPv4NK5cu |
| ization_channel14num_parametersE) | daq5ptsbe18PTSBEExecutionData18co |
| -   [cudaq::de                    | unt_instructionsE20TraceInstructi |
| polarization_channel::num_targets | onTypeNSt8optionalINSt6stringEEE) |
|     (C++                          | -   [cudaq::ptsbe::P              |
|     member)](api/languages        | TSBEExecutionData::get_trajectory |
| /cpp_api.html#_CPPv4N5cudaq22depo |     (C++                          |
| larization_channel11num_targetsE) |     function                      |
| -   [cudaq::detail (C++           | )](api/languages/cpp_api.html#_CP |
|     type)](api/languages/cp       | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| p_api.html#_CPPv4N5cudaq6detailE) | Data14get_trajectoryENSt6size_tE) |
| -   [cudaq::detail::future (C++   | -   [cudaq::ptsbe:                |
|                                   | :PTSBEExecutionData::instructions |
|   class)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq6detail6futureE) |     member)](api/languages/cp     |
| -                                 | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|    [cudaq::detail::future::future | TSBEExecutionData12instructionsE) |
|     (C++                          | -   [cudaq::ptsbe:                |
|     functi                        | :PTSBEExecutionData::trajectories |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq6detail6future6future |     member)](api/languages/cp     |
| ERNSt6vectorI3JobEERNSt6stringERN | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| St3mapINSt6stringENSt6stringEEE), | TSBEExecutionData12trajectoriesE) |
|     [\[1\]](api/lan               | -   [cudaq::ptsbe::PTSBEOptions   |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 6detail6future6futureERR6future), |     struc                         |
|     [\[2\]                        | t)](api/languages/cpp_api.html#_C |
| ](api/languages/cpp_api.html#_CPP | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| v4N5cudaq6detail6future6futureEv) | -   [cudaq::ptsbe::PTSB           |
| -   [c                            | EOptions::include_sequential_data |
| udaq::detail::kernel_builder_base |     (C++                          |
|     (C++                          |                                   |
|     class)](api/                  |    member)](api/languages/cpp_api |
| languages/cpp_api.html#_CPPv4N5cu | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| daq6detail19kernel_builder_baseE) | ptions23include_sequential_dataE) |
| -   [cudaq::detail::              | -   [cudaq::ptsb                  |
| kernel_builder_base::operator\<\< | e::PTSBEOptions::max_trajectories |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     member)](api/languages/       |
| ages/cpp_api.html#_CPPv4N5cudaq6d | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| etail19kernel_builder_baselsERNSt | 2PTSBEOptions16max_trajectoriesE) |
| 7ostreamERK19kernel_builder_base) | -   [cudaq::ptsbe::PT             |
| -                                 | SBEOptions::return_execution_data |
| [cudaq::detail::KernelBuilderType |     (C++                          |
|     (C++                          |     member)](api/languages/cpp_a  |
|     class)](ap                    | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| i/languages/cpp_api.html#_CPPv4N5 | EOptions21return_execution_dataE) |
| cudaq6detail17KernelBuilderTypeE) | -   [cudaq::pts                   |
| -   [cudaq::                      | be::PTSBEOptions::shot_allocation |
| detail::KernelBuilderType::create |     (C++                          |
|     (C++                          |     member)](api/languages        |
|     function                      | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| )](api/languages/cpp_api.html#_CP | 12PTSBEOptions15shot_allocationE) |
| Pv4N5cudaq6detail17KernelBuilderT | -   [cud                          |
| ype6createEPN4mlir11MLIRContextE) | aq::ptsbe::PTSBEOptions::strategy |
| -   [cudaq::detail::Ker           |     (C++                          |
| nelBuilderType::KernelBuilderType |     member)](api/l                |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     function)](api/lan            | aq5ptsbe12PTSBEOptions8strategyE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::ptsbe::PTSBETrace     |
| 6detail17KernelBuilderType17Kerne |     (C++                          |
| lBuilderTypeERRNSt8functionIFN4ml |     t                             |
| ir4TypeEPN4mlir11MLIRContextEEEE) | ype)](api/languages/cpp_api.html# |
| -   [cudaq::detector (C++         | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
|     function)](api                | -   [                             |
| /languages/cpp_api.html#_CPPv4IDp | cudaq::ptsbe::PTSSamplingStrategy |
| EN5cudaq8detectorEvDpRR8MeasArgs) |     (C++                          |
| -   [cudaq::detectors (C++        |     class)](api                   |
|     function)](api/languages/c    | /languages/cpp_api.html#_CPPv4N5c |
| pp_api.html#_CPPv4N5cudaq9detecto | udaq5ptsbe19PTSSamplingStrategyE) |
| rsERKNSt6vectorI14measure_resultE | -   [cudaq::                      |
| ERKNSt6vectorI14measure_resultEE) | ptsbe::PTSSamplingStrategy::clone |
| -   [cudaq::diag_matrix_callback  |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     class)                        | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| ](api/languages/cpp_api.html#_CPP | sbe19PTSSamplingStrategy5cloneEv) |
| v4N5cudaq20diag_matrix_callbackE) | -   [cudaq::ptsbe::PTSSampl       |
| -   [cudaq::dyn (C++              | ingStrategy::generateTrajectories |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq3dynE) |     function)](api/               |
| -   [cudaq::ExecutionContext (C++ | languages/cpp_api.html#_CPPv4NK5c |
|     cl                            | udaq5ptsbe19PTSSamplingStrategy20 |
| ass)](api/languages/cpp_api.html# | generateTrajectoriesENSt4spanIKN6 |
| _CPPv4N5cudaq16ExecutionContextE) | detail10NoisePointEEENSt6size_tE) |
| -   [c                            | -   [cudaq:                       |
| udaq::ExecutionContext::asyncExec | :ptsbe::PTSSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     member)](api/                 |     function)](api/langua         |
| languages/cpp_api.html#_CPPv4N5cu | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| daq16ExecutionContext9asyncExecE) | tsbe19PTSSamplingStrategy4nameEv) |
| -   [cud                          | -   [cudaq::ptsbe::PTSSampli      |
| aq::ExecutionContext::asyncResult | ngStrategy::\~PTSSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/la             |
| guages/cpp_api.html#_CPPv4N5cudaq | nguages/cpp_api.html#_CPPv4N5cuda |
| 16ExecutionContext11asyncResultE) | q5ptsbe19PTSSamplingStrategyD0Ev) |
| -   [cudaq:                       | -   [cudaq::ptsbe::sample (C++    |
| :ExecutionContext::batchIteration |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     member)](api/langua           | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| ges/cpp_api.html#_CPPv4N5cudaq16E | mpleE13sample_resultRK14sample_op |
| xecutionContext14batchIterationE) | tionsRR13QuantumKernelDpRR4Args), |
| -   [cudaq::E                     |     [\[1\]](api                   |
| xecutionContext::canHandleObserve | /languages/cpp_api.html#_CPPv4I0D |
|     (C++                          | pEN5cudaq5ptsbe6sampleE13sample_r |
|     member)](api/language         | esultRKN5cudaq11noise_modelENSt6s |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | ize_tERR13QuantumKernelDpRR4Args) |
| cutionContext16canHandleObserveE) | -   [cudaq::ptsbe::sample_async   |
| -   [cudaq::Executio              |     (C++                          |
| nContext::deferredKernelException |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4I |
|     member)](api/languages/cpp_a  | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| pi.html#_CPPv4N5cudaq16ExecutionC | 9async_sample_resultRK14sample_op |
| ontext23deferredKernelExceptionE) | tionsRR13QuantumKernelDpRR4Args), |
| -   [cudaq::E                     |     [\[1\]](api/languages/cp      |
| xecutionContext::ExecutionContext | p_api.html#_CPPv4I0DpEN5cudaq5pts |
|     (C++                          | be12sample_asyncE19async_sample_r |
|     func                          | esultRKN5cudaq11noise_modelENSt6s |
| tion)](api/languages/cpp_api.html | ize_tERR13QuantumKernelDpRR4Args) |
| #_CPPv4N5cudaq16ExecutionContext1 | -   [cudaq::ptsbe::sample_options |
| 6ExecutionContextERKNSt6stringE), |     (C++                          |
|     [\[1\]](api/languages/        |     struct)                       |
| cpp_api.html#_CPPv4N5cudaq16Execu | ](api/languages/cpp_api.html#_CPP |
| tionContext16ExecutionContextERKN | v4N5cudaq5ptsbe14sample_optionsE) |
| St6stringENSt6size_tENSt6size_tE) | -   [cudaq::ptsbe::sample_result  |
| -   [cudaq::Execu                 |     (C++                          |
| tionContext::explicitMeasurements |     class                         |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)](api/languages/cp     | Pv4N5cudaq5ptsbe13sample_resultE) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::pts                   |
| onContext20explicitMeasurementsE) | be::sample_result::execution_data |
| -   [cuda                         |     (C++                          |
| q::ExecutionContext::futureResult |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
|     member)](api/lang             | 3sample_result14execution_dataEv) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::ptsbe::               |
| 6ExecutionContext12futureResultE) | sample_result::has_execution_data |
| -   [cudaq::ExecutionContext      |     (C++                          |
| ::hasConditionalsOnMeasureResults |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     mem                           | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| ber)](api/languages/cpp_api.html# | ple_result18has_execution_dataEv) |
| _CPPv4N5cudaq16ExecutionContext31 | -   [cudaq::pt                    |
| hasConditionalsOnMeasureResultsE) | sbe::sample_result::sample_result |
| -   [cudaq:                       |     (C++                          |
| :ExecutionContext::inKernelLaunch |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     member)](api/langua           | aq5ptsbe13sample_result13sample_r |
| ges/cpp_api.html#_CPPv4N5cudaq16E | esultERRN5cudaq13sample_resultE), |
| xecutionContext14inKernelLaunchE) |                                   |
| -   [cu                           |  [\[1\]](api/languages/cpp_api.ht |
| daq::ExecutionContext::kernelName | ml#_CPPv4N5cudaq5ptsbe13sample_re |
|     (C++                          | sult13sample_resultERRN5cudaq13sa |
|     member)](api/la               | mple_resultE18PTSBEExecutionData) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::ptsbe::               |
| q16ExecutionContext10kernelNameE) | sample_result::set_execution_data |
| -   [cud                          |     (C++                          |
| aq::ExecutionContext::kernelTrace |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     member)](api/lan              | daq5ptsbe13sample_result18set_exe |
| guages/cpp_api.html#_CPPv4N5cudaq | cution_dataE18PTSBEExecutionData) |
| 16ExecutionContext11kernelTraceE) | -   [cud                          |
| -                                 | aq::ptsbe::ShotAllocationStrategy |
|    [cudaq::ExecutionContext::name |     (C++                          |
|     (C++                          |     struct)](using                |
|     member)]                      | /examples/ptsbe.html#_CPPv4N5cuda |
| (api/languages/cpp_api.html#_CPPv | q5ptsbe22ShotAllocationStrategyE) |
| 4N5cudaq16ExecutionContext4nameE) | -   [cudaq::ptsbe::ShotAllocatio  |
| -   [cu                           | nStrategy::ShotAllocationStrategy |
| daq::ExecutionContext::noiseModel |     (C++                          |
|     (C++                          |     function)                     |
|     member)](api/la               | ](using/examples/ptsbe.html#_CPPv |
| nguages/cpp_api.html#_CPPv4N5cuda | 4N5cudaq5ptsbe22ShotAllocationStr |
| q16ExecutionContext10noiseModelE) | ategy22ShotAllocationStrategyE4Ty |
| -   [cudaq::Exe                   | pedNSt8optionalINSt8uint64_tEEE), |
| cutionContext::numberTrajectories |     [\[1\                         |
|     (C++                          | ]](using/examples/ptsbe.html#_CPP |
|     member)](api/languages/       | v4N5cudaq5ptsbe22ShotAllocationSt |
| cpp_api.html#_CPPv4N5cudaq16Execu | rategy22ShotAllocationStrategyEv) |
| tionContext18numberTrajectoriesE) | -   [cudaq::pt                    |
| -   [c                            | sbe::ShotAllocationStrategy::Type |
| udaq::ExecutionContext::optResult |     (C++                          |
|     (C++                          |     enum)](using/exam             |
|     member)](api/                 | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| languages/cpp_api.html#_CPPv4N5cu | be22ShotAllocationStrategy4TypeE) |
| daq16ExecutionContext9optResultE) | -   [cudaq::ptsbe::ShotAllocatio  |
| -                                 | nStrategy::Type::HIGH_WEIGHT_BIAS |
|   [cudaq::ExecutionContext::qpuId |     (C++                          |
|     (C++                          |     enumerat                      |
|     member)](                     | or)](using/examples/ptsbe.html#_C |
| api/languages/cpp_api.html#_CPPv4 | PPv4N5cudaq5ptsbe22ShotAllocation |
| N5cudaq16ExecutionContext5qpuIdE) | Strategy4Type16HIGH_WEIGHT_BIASE) |
| -   [cudaq                        | -   [cudaq::ptsbe::ShotAllocati   |
| ::ExecutionContext::registerNames | onStrategy::Type::LOW_WEIGHT_BIAS |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     enumera                       |
| ages/cpp_api.html#_CPPv4N5cudaq16 | tor)](using/examples/ptsbe.html#_ |
| ExecutionContext13registerNamesE) | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| -   [cu                           | nStrategy4Type15LOW_WEIGHT_BIASE) |
| daq::ExecutionContext::reorderIdx | -   [cudaq::ptsbe::ShotAlloc      |
|     (C++                          | ationStrategy::Type::PROPORTIONAL |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     enum                          |
| q16ExecutionContext10reorderIdxE) | erator)](using/examples/ptsbe.htm |
| -                                 | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
|   [cudaq::ExecutionContext::shots | tionStrategy4Type12PROPORTIONALE) |
|     (C++                          | -   [cudaq::ptsbe::Shot           |
|     member)](                     | AllocationStrategy::Type::UNIFORM |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5shotsE) |                                   |
| -   [cudaq::                      |   enumerator)](using/examples/pts |
| ExecutionContext::simulationState | be.html#_CPPv4N5cudaq5ptsbe22Shot |
|     (C++                          | AllocationStrategy4Type7UNIFORME) |
|     member)](api/languag          | -                                 |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |   [cudaq::ptsbe::TraceInstruction |
| ecutionContext15simulationStateE) |     (C++                          |
| -                                 |     struct)](                     |
|    [cudaq::ExecutionContext::spin | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5ptsbe16TraceInstructionE) |
|     member)]                      | -   [cudaq:                       |
| (api/languages/cpp_api.html#_CPPv | :ptsbe::TraceInstruction::channel |
| 4N5cudaq16ExecutionContext4spinE) |     (C++                          |
| -   [cudaq::                      |     member)](api/lang             |
| ExecutionContext::totalIterations | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     (C++                          | ptsbe16TraceInstruction7channelE) |
|     member)](api/languag          | -   [cudaq::                      |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | ptsbe::TraceInstruction::controls |
| ecutionContext15totalIterationsE) |     (C++                          |
| -   [cudaq::ExecutionResult (C++  |     member)](api/langu            |
|     st                            | ages/cpp_api.html#_CPPv4N5cudaq5p |
| ruct)](api/languages/cpp_api.html | tsbe16TraceInstruction8controlsE) |
| #_CPPv4N5cudaq15ExecutionResultE) | -   [cud                          |
| -   [cud                          | aq::ptsbe::TraceInstruction::name |
| aq::ExecutionResult::appendResult |     (C++                          |
|     (C++                          |     member)](api/l                |
|     functio                       | anguages/cpp_api.html#_CPPv4N5cud |
| n)](api/languages/cpp_api.html#_C | aq5ptsbe16TraceInstruction4nameE) |
| PPv4N5cudaq15ExecutionResult12app | -   [cudaq                        |
| endResultENSt6stringENSt6size_tE) | ::ptsbe::TraceInstruction::params |
| -   [cu                           |     (C++                          |
| daq::ExecutionResult::deserialize |     member)](api/lan              |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     function)                     | 5ptsbe16TraceInstruction6paramsE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq:                       |
| v4N5cudaq15ExecutionResult11deser | :ptsbe::TraceInstruction::targets |
| ializeERNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq:                       |     member)](api/lang             |
| :ExecutionResult::ExecutionResult | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     (C++                          | ptsbe16TraceInstruction7targetsE) |
|     functio                       | -   [cudaq::ptsbe::T              |
| n)](api/languages/cpp_api.html#_C | raceInstruction::TraceInstruction |
| PPv4N5cudaq15ExecutionResult15Exe |     (C++                          |
| cutionResultE16CountsDictionary), |                                   |
|     [\[1\]](api/lan               |   function)](api/languages/cpp_ap |
| guages/cpp_api.html#_CPPv4N5cudaq | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| 15ExecutionResult15ExecutionResul | Instruction16TraceInstructionE20T |
| tE16CountsDictionaryNSt6stringE), | raceInstructionTypeNSt6stringENSt |
|     [\[2\                         | 6vectorINSt6size_tEEENSt6vectorIN |
| ]](api/languages/cpp_api.html#_CP | St6size_tEEENSt6vectorIdEENSt8opt |
| Pv4N5cudaq15ExecutionResult15Exec | ionalIN5cudaq13kraus_channelEEE), |
| utionResultE16CountsDictionaryd), |     [\[1\]](api/languages/cpp_a   |
|                                   | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
|    [\[3\]](api/languages/cpp_api. | eInstruction16TraceInstructionEv) |
| html#_CPPv4N5cudaq15ExecutionResu | -   [cud                          |
| lt15ExecutionResultENSt6stringE), | aq::ptsbe::TraceInstruction::type |
|     [\[4\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     member)](api/l                |
| Pv4N5cudaq15ExecutionResult15Exec | anguages/cpp_api.html#_CPPv4N5cud |
| utionResultERK15ExecutionResult), | aq5ptsbe16TraceInstruction4typeE) |
|     [\[5\]](api/language          | -   [c                            |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | udaq::ptsbe::TraceInstructionType |
| cutionResult15ExecutionResultEd), |     (C++                          |
|     [\[6\]](api/languag           |     enum)](api/                   |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | languages/cpp_api.html#_CPPv4N5cu |
| ecutionResult15ExecutionResultEv) | daq5ptsbe20TraceInstructionTypeE) |
| -   [                             | -   [cudaq::                      |
| cudaq::ExecutionResult::operator= | ptsbe::TraceInstructionType::Gate |
|     (C++                          |     (C++                          |
|     function)](api/languages/     |     enumerator)](api/langu        |
| cpp_api.html#_CPPv4N5cudaq15Execu | ages/cpp_api.html#_CPPv4N5cudaq5p |
| tionResultaSERK15ExecutionResult) | tsbe20TraceInstructionType4GateE) |
| -   [c                            | -   [cudaq::ptsbe::               |
| udaq::ExecutionResult::operator== | TraceInstructionType::Measurement |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |                                   |
| pp_api.html#_CPPv4NK5cudaq15Execu |    enumerator)](api/languages/cpp |
| tionResulteqERK15ExecutionResult) | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| -   [cud                          | aceInstructionType11MeasurementE) |
| aq::ExecutionResult::registerName | -   [cudaq::p                     |
|     (C++                          | tsbe::TraceInstructionType::Noise |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     enumerator)](api/langua       |
| 15ExecutionResult12registerNameE) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq                        | sbe20TraceInstructionType5NoiseE) |
| ::ExecutionResult::sequentialData | -   [                             |
|     (C++                          | cudaq::ptsbe::TrajectoryPredicate |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     type)](api                    |
| ExecutionResult14sequentialDataE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [                             | udaq5ptsbe19TrajectoryPredicateE) |
| cudaq::ExecutionResult::serialize | -   [cudaq::QPU (C++              |
|     (C++                          |     class)](api/languages         |
|     function)](api/l              | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| anguages/cpp_api.html#_CPPv4NK5cu | -   [cudaq::QPU::beginExecution   |
| daq15ExecutionResult9serializeEv) |     (C++                          |
| -   [cudaq::fermion_handler (C++  |     function                      |
|     c                             | )](api/languages/cpp_api.html#_CP |
| lass)](api/languages/cpp_api.html | Pv4N5cudaq3QPU14beginExecutionEv) |
| #_CPPv4N5cudaq15fermion_handlerE) | -   [cuda                         |
| -   [cudaq::fermion_op (C++       | q::QPU::configureExecutionContext |
|     type)](api/languages/cpp_api  |     (C++                          |
| .html#_CPPv4N5cudaq10fermion_opE) |     funct                         |
| -   [cudaq::fermion_op_term (C++  | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4NK5cudaq3QPU25configureExec |
| type)](api/languages/cpp_api.html | utionContextER16ExecutionContext) |
| #_CPPv4N5cudaq15fermion_op_termE) | -   [cudaq::QPU::endExecution     |
| -   [cudaq::FermioniqQPU (C++     |     (C++                          |
|                                   |     functi                        |
|   class)](api/languages/cpp_api.h | on)](api/languages/cpp_api.html#_ |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | CPPv4N5cudaq3QPU12endExecutionEv) |
| -   [cudaq::get_state (C++        | -   [cudaq::QPU::enqueue (C++     |
|                                   |     function)](ap                 |
|    function)](api/languages/cpp_a | i/languages/cpp_api.html#_CPPv4N5 |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | cudaq3QPU7enqueueER11QuantumTask) |
| ateEDaRR13QuantumKernelDpRR4Args) | -   [cud                          |
| -   [cudaq::GPUEmulatedQPU (C++   | aq::QPU::finalizeExecutionContext |
|                                   |     (C++                          |
| class)](api/languages/cpp_api.htm |     func                          |
| l#_CPPv4N5cudaq14GPUEmulatedQPUE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::gradient (C++         | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     class)](api/languages/cpp_    | utionContextER16ExecutionContext) |
| api.html#_CPPv4N5cudaq8gradientE) | -   [cudaq::QPU::getCompileTarget |
| -   [cudaq::gradient::clone (C++  |     (C++                          |
|     fun                           |     function)]                    |
| ction)](api/languages/cpp_api.htm | (api/languages/cpp_api.html#_CPPv |
| l#_CPPv4N5cudaq8gradient5cloneEv) | 4N5cudaq3QPU16getCompileTargetEb) |
| -   [cudaq::gradient::compute     | -   [cudaq::QPU::getConnectivity  |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)                     |
| s/cpp_api.html#_CPPv4N5cudaq8grad | ](api/languages/cpp_api.html#_CPP |
| ient7computeERKNSt6vectorIdEERKNS | v4N5cudaq3QPU15getConnectivityEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -                                 |
|     [\[1\]](ap                    | [cudaq::QPU::getExecutionThreadId |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq8gradient7computeERKNSt6vect |     function)](api/               |
| orIdEERNSt6vectorIdEERK7spin_opd) | languages/cpp_api.html#_CPPv4NK5c |
| -   [cudaq::gradient::gradient    | udaq3QPU20getExecutionThreadIdEv) |
|     (C++                          | -   [cudaq::QPU::getNumQubits     |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4I00EN5cu |     functi                        |
| daq8gradient8gradientER7KernelT), | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|    [\[1\]](api/languages/cpp_api. | -   [                             |
| html#_CPPv4I00EN5cudaq8gradient8g | cudaq::QPU::getRemoteCapabilities |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\                         |     function)](api/l              |
| ]](api/languages/cpp_api.html#_CP | anguages/cpp_api.html#_CPPv4NK5cu |
| Pv4I00EN5cudaq8gradient8gradientE | daq3QPU21getRemoteCapabilitiesEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::QPU::isEmulated (C++  |
|     [\[3                          |     func                          |
| \]](api/languages/cpp_api.html#_C | tion)](api/languages/cpp_api.html |
| PPv4N5cudaq8gradient8gradientERRN | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QPU::isSimulator (C++ |
|     [\[                           |     funct                         |
| 4\]](api/languages/cpp_api.html#_ | ion)](api/languages/cpp_api.html# |
| CPPv4N5cudaq8gradient8gradientEv) | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| -   [cudaq::gradient::setArgs     | -   [cudaq::QPU::onRandomSeedSet  |
|     (C++                          |     (C++                          |
|     fu                            |     function)](api/lang           |
| nction)](api/languages/cpp_api.ht | uages/cpp_api.html#_CPPv4N5cudaq3 |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | QPU15onRandomSeedSetENSt6size_tE) |
| tArgsEvR13QuantumKernelDpRR4Args) | -   [cudaq::QPU::QPU (C++         |
| -   [cudaq::gradient::setKernel   |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)](api/languages/c    | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| pp_api.html#_CPPv4I0EN5cudaq8grad |                                   |
| ient9setKernelEvR13QuantumKernel) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cud                          | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| aq::gradients::central_difference |     [\[2\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     class)](api/la                | -   [cudaq::QPU::setId (C++       |
| nguages/cpp_api.html#_CPPv4N5cuda |     function                      |
| q9gradients18central_differenceE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::gra                   | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| dients::central_difference::clone | -   [cudaq::QPU::setShots (C++    |
|     (C++                          |     f                             |
|     function)](api/languages      | unction)](api/languages/cpp_api.h |
| /cpp_api.html#_CPPv4N5cudaq9gradi | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| ents18central_difference5cloneEv) | -   [cudaq::QPU::\~QPU (C++       |
| -   [cudaq::gradi                 |     function)](api/languages/cp   |
| ents::central_difference::compute | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     (C++                          | -   [cudaq::QPUState (C++         |
|     function)](                   |     class)](api/languages/cpp_    |
| api/languages/cpp_api.html#_CPPv4 | api.html#_CPPv4N5cudaq8QPUStateE) |
| N5cudaq9gradients18central_differ | -   [cudaq::qreg (C++             |
| ence7computeERKNSt6vectorIdEERKNS |     class)](api/lan               |
| t8functionIFdNSt6vectorIdEEEEEd), | guages/cpp_api.html#_CPPv4I_NSt6s |
|                                   | ize_tE_NSt6size_tEEN5cudaq4qregE) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::qreg::back (C++       |
| tml#_CPPv4N5cudaq9gradients18cent |     function)                     |
| ral_difference7computeERKNSt6vect | ](api/languages/cpp_api.html#_CPP |
| orIdEERNSt6vectorIdEERK7spin_opd) | v4N5cudaq4qreg4backENSt6size_tE), |
| -   [cudaq::gradie                |     [\[1\]](api/languages/cpp_ap  |
| nts::central_difference::gradient | i.html#_CPPv4N5cudaq4qreg4backEv) |
|     (C++                          | -   [cudaq::qreg::begin (C++      |
|     functio                       |                                   |
| n)](api/languages/cpp_api.html#_C |  function)](api/languages/cpp_api |
| PPv4I00EN5cudaq9gradients18centra | .html#_CPPv4N5cudaq4qreg5beginEv) |
| l_difference8gradientER7KernelT), | -   [cudaq::qreg::clear (C++      |
|     [\[1\]](api/langua            |                                   |
| ges/cpp_api.html#_CPPv4I00EN5cuda |  function)](api/languages/cpp_api |
| q9gradients18central_difference8g | .html#_CPPv4N5cudaq4qreg5clearEv) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::qreg::front (C++      |
|     [\[2\]](api/languages/cpp_    |     function)]                    |
| api.html#_CPPv4I00EN5cudaq9gradie | (api/languages/cpp_api.html#_CPPv |
| nts18central_difference8gradientE | 4N5cudaq4qreg5frontENSt6size_tE), |
| RR13QuantumKernelRR10ArgsMapper), |     [\[1\]](api/languages/cpp_api |
|     [\[3\]](api/languages/cpp     | .html#_CPPv4N5cudaq4qreg5frontEv) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::qreg::operator\[\]    |
| 18central_difference8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     functi                        |
|     [\[4\]](api/languages/cp      | on)](api/languages/cpp_api.html#_ |
| p_api.html#_CPPv4N5cudaq9gradient | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| s18central_difference8gradientEv) | -   [cudaq::qreg::qreg (C++       |
| -   [cud                          |     function)                     |
| aq::gradients::forward_difference | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4qregENSt6size_tE), |
|     class)](api/la                |     [\[1\]](api/languages/cpp_ap  |
| nguages/cpp_api.html#_CPPv4N5cuda | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| q9gradients18forward_differenceE) | -   [cudaq::qreg::size (C++       |
| -   [cudaq::gra                   |                                   |
| dients::forward_difference::clone |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     function)](api/languages      | -   [cudaq::qreg::slice (C++      |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)](api/langu          |
| ents18forward_difference5cloneEv) | ages/cpp_api.html#_CPPv4N5cudaq4q |
| -   [cudaq::gradi                 | reg5sliceENSt6size_tENSt6size_tE) |
| ents::forward_difference::compute | -   [cudaq::qreg::value_type (C++ |
|     (C++                          |                                   |
|     function)](                   | type)](api/languages/cpp_api.html |
| api/languages/cpp_api.html#_CPPv4 | #_CPPv4N5cudaq4qreg10value_typeE) |
| N5cudaq9gradients18forward_differ | -   [cudaq::qspan (C++            |
| ence7computeERKNSt6vectorIdEERKNS |     class)](api/lang              |
| t8functionIFdNSt6vectorIdEEEEEd), | uages/cpp_api.html#_CPPv4I_NSt6si |
|                                   | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::QuakeValue (C++       |
| tml#_CPPv4N5cudaq9gradients18forw |     class)](api/languages/cpp_api |
| ard_difference7computeERKNSt6vect | .html#_CPPv4N5cudaq10QuakeValueE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::Q                     |
| -   [cudaq::gradie                | uakeValue::canValidateNumElements |
| nts::forward_difference::gradient |     (C++                          |
|     (C++                          |     function)](api/languages      |
|     functio                       | /cpp_api.html#_CPPv4N5cudaq10Quak |
| n)](api/languages/cpp_api.html#_C | eValue22canValidateNumElementsEv) |
| PPv4I00EN5cudaq9gradients18forwar | -                                 |
| d_difference8gradientER7KernelT), |  [cudaq::QuakeValue::constantSize |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     function)](api                |
| q9gradients18forward_difference8g | /languages/cpp_api.html#_CPPv4N5c |
| radientER7KernelTRR10ArgsMapper), | udaq10QuakeValue12constantSizeEv) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::QuakeValue::dump (C++ |
| api.html#_CPPv4I00EN5cudaq9gradie |     function)](api/lan            |
| nts18forward_difference8gradientE | guages/cpp_api.html#_CPPv4N5cudaq |
| RR13QuantumKernelRR10ArgsMapper), | 10QuakeValue4dumpERNSt7ostreamE), |
|     [\[3\]](api/languages/cpp     |     [\                            |
| _api.html#_CPPv4N5cudaq9gradients | [1\]](api/languages/cpp_api.html# |
| 18forward_difference8gradientERRN | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq                        |
|     [\[4\]](api/languages/cp      | ::QuakeValue::getRequiredElements |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18forward_difference8gradientEv) |     function)](api/langua         |
| -   [                             | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| cudaq::gradients::parameter_shift | uakeValue19getRequiredElementsEv) |
|     (C++                          | -   [cudaq::QuakeValue::getValue  |
|     class)](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)]                    |
| udaq9gradients15parameter_shiftE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::                      | 4NK5cudaq10QuakeValue8getValueEv) |
| gradients::parameter_shift::clone | -   [cudaq::QuakeValue::inverse   |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)                     |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | ](api/languages/cpp_api.html#_CPP |
| adients15parameter_shift5cloneEv) | v4NK5cudaq10QuakeValue7inverseEv) |
| -   [cudaq::gr                    | -                                 |
| adients::parameter_shift::compute |    [cudaq::QuakeValue::isSequence |
|     (C++                          |     (C++                          |
|     function                      |     function)](a                  |
| )](api/languages/cpp_api.html#_CP | pi/languages/cpp_api.html#_CPPv4N |
| Pv4N5cudaq9gradients15parameter_s | 5cudaq10QuakeValue10isSequenceEv) |
| hift7computeERKNSt6vectorIdEERKNS | -                                 |
| t8functionIFdNSt6vectorIdEEEEEd), |    [cudaq::QuakeValue::operator\* |
|     [\[1\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4N5cudaq9gradients15p |     function)](api                |
| arameter_shift7computeERKNSt6vect | /languages/cpp_api.html#_CPPv4N5c |
| orIdEERNSt6vectorIdEERK7spin_opd) | udaq10QuakeValuemlE10QuakeValue), |
| -   [cudaq::gra                   |                                   |
| dients::parameter_shift::gradient | [\[1\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|     func                          | -   [cudaq::QuakeValue::operator+ |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4I00EN5cudaq9gradients15par |     function)](api                |
| ameter_shift8gradientER7KernelT), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]](api/lan               | udaq10QuakeValueplE10QuakeValue), |
| guages/cpp_api.html#_CPPv4I00EN5c |     [                             |
| udaq9gradients15parameter_shift8g | \[1\]](api/languages/cpp_api.html |
| radientER7KernelTRR10ArgsMapper), | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     [\[2\]](api/languages/c       |                                   |
| pp_api.html#_CPPv4I00EN5cudaq9gra | [\[2\]](api/languages/cpp_api.htm |
| dients15parameter_shift8gradientE | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::QuakeValue::operator- |
|     [\[3\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq9gradie |     function)](api                |
| nts15parameter_shift8gradientERRN | /languages/cpp_api.html#_CPPv4N5c |
| St8functionIFvNSt6vectorIdEEEEE), | udaq10QuakeValuemiE10QuakeValue), |
|     [\[4\]](api/languages         |     [                             |
| /cpp_api.html#_CPPv4N5cudaq9gradi | \[1\]](api/languages/cpp_api.html |
| ents15parameter_shift8gradientEv) | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| -   [cudaq::kernel_builder (C++   |     [                             |
|     clas                          | \[2\]](api/languages/cpp_api.html |
| s)](api/languages/cpp_api.html#_C | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| PPv4IDpEN5cudaq14kernel_builderE) |                                   |
| -   [c                            | [\[3\]](api/languages/cpp_api.htm |
| udaq::kernel_builder::constantVal | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     (C++                          | -   [cudaq::QuakeValue::operator/ |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api                |
| q14kernel_builder11constantValEd) | /languages/cpp_api.html#_CPPv4N5c |
| -                                 | udaq10QuakeValuedvE10QuakeValue), |
|  [cudaq::kernel_builder::detector |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
|    function)](api/languages/cpp_a | -                                 |
| pi.html#_CPPv4IDpEN5cudaq14kernel |  [cudaq::QuakeValue::operator\[\] |
| _builder8detectorEvDpRR8MeasArgs) |     (C++                          |
| -                                 |     function)](api                |
| [cudaq::kernel_builder::detectors | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValueixEKNSt6size_tE), |
|     func                          |     [\[1\]](api/                  |
| tion)](api/languages/cpp_api.html | languages/cpp_api.html#_CPPv4N5cu |
| #_CPPv4N5cudaq14kernel_builder9de | daq10QuakeValueixERK10QuakeValue) |
| tectorsE10QuakeValue10QuakeValue) | -                                 |
| -   [cu                           |    [cudaq::QuakeValue::QuakeValue |
| daq::kernel_builder::getArguments |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/lan            | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| guages/cpp_api.html#_CPPv4N5cudaq | akeValue10QuakeValueERN4mlir20Imp |
| 14kernel_builder12getArgumentsEv) | licitLocOpBuilderEN4mlir5ValueE), |
| -   [cu                           |     [\[1\]                        |
| daq::kernel_builder::getNumParams | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq10QuakeValue10QuakeValue |
|     function)](api/lan            | ERN4mlir20ImplicitLocOpBuilderEd) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::QuakeValue::size (C++ |
| 14kernel_builder12getNumParamsEv) |     funct                         |
| -   [cud                          | ion)](api/languages/cpp_api.html# |
| aq::kernel_builder::isArgSequence | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|     (C++                          | -   [cudaq::QuakeValue::slice     |
|     function)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq14kernel_bu |     function)](api/languages/cpp_ |
| ilder13isArgSequenceENSt6size_tE) | api.html#_CPPv4N5cudaq10QuakeValu |
| -   [cuda                         | e5sliceEKNSt6size_tEKNSt6size_tE) |
| q::kernel_builder::kernel_builder | -   [cudaq::quantum_platform (C++ |
|     (C++                          |     cl                            |
|     function)](api/languages/cpp  | ass)](api/languages/cpp_api.html# |
| _api.html#_CPPv4N5cudaq14kernel_b | _CPPv4N5cudaq16quantum_platformE) |
| uilder14kernel_builderERNSt6vecto | -   [cudaq:                       |
| rIN6detail17KernelBuilderTypeEEE) | :quantum_platform::beginExecution |
| -   [cudaq::k                     |     (C++                          |
| ernel_builder::logical_observable |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     function)                     | antum_platform14beginExecutionEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::quantum_pl            |
| v4IDpEN5cudaq14kernel_builder18lo | atform::configureExecutionContext |
| gical_observableEvDpRR8MeasArgs), |     (C++                          |
|     [\[1\]](ap                    |     function)](api/lang           |
| i/languages/cpp_api.html#_CPPv4N5 | uages/cpp_api.html#_CPPv4NK5cudaq |
| cudaq14kernel_builder18logical_ob | 16quantum_platform25configureExec |
| servableE10QuakeValueNSt6size_tE) | utionContextER16ExecutionContext) |
| -   [cudaq::kernel_builder::name  | -   [cuda                         |
|     (C++                          | q::quantum_platform::connectivity |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/langu          |
| v4N5cudaq14kernel_builder4nameEv) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -                                 | quantum_platform12connectivityEv) |
|    [cudaq::kernel_builder::qalloc | -   [cuda                         |
|     (C++                          | q::quantum_platform::endExecution |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     function)](api/langu          |
| nel_builder6qallocE10QuakeValue), | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     [\[1\]](api/language          | quantum_platform12endExecutionEv) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::q                     |
| nel_builder6qallocEKNSt6size_tE), | uantum_platform::enqueueAsyncTask |
|     [\[2                          |     (C++                          |
| \]](api/languages/cpp_api.html#_C |     function)](api/languages/     |
| PPv4N5cudaq14kernel_builder6qallo | cpp_api.html#_CPPv4N5cudaq16quant |
| cERNSt6vectorINSt7complexIdEEEE), | um_platform16enqueueAsyncTaskEKNS |
|     [\[3\]](                      | t6size_tER19KernelExecutionTask), |
| api/languages/cpp_api.html#_CPPv4 |     [\[1\]](api/languag           |
| N5cudaq14kernel_builder6qallocEv) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cudaq::kernel_builder::swap  | antum_platform16enqueueAsyncTaskE |
|     (C++                          | KNSt6size_tERNSt8functionIFvvEEE) |
|     function)](api/language       | -   [cudaq::quantum_p             |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | latform::finalizeExecutionContext |
| 4kernel_builder4swapEvRK10QuakeVa |     (C++                          |
| lueRK10QuakeValueRK10QuakeValue), |     function)](api/languages/c    |
|                                   | pp_api.html#_CPPv4NK5cudaq16quant |
| [\[1\]](api/languages/cpp_api.htm | um_platform24finalizeExecutionCon |
| l#_CPPv4I00EN5cudaq14kernel_build | textERN5cudaq16ExecutionContextE) |
| er4swapEvRKNSt6vectorI10QuakeValu | -   [cudaq::qua                   |
| eEERK10QuakeValueRK10QuakeValue), | ntum_platform::get_codegen_config |
|                                   |     (C++                          |
| [\[2\]](api/languages/cpp_api.htm |     function)](api/languages/c    |
| l#_CPPv4N5cudaq14kernel_builder4s | pp_api.html#_CPPv4N5cudaq16quantu |
| wapERK10QuakeValueRK10QuakeValue) | m_platform18get_codegen_configEv) |
| -   [cudaq::KernelExecutionTask   | -   [cuda                         |
|     (C++                          | q::quantum_platform::get_exec_ctx |
|     type                          |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/langua         |
| Pv4N5cudaq19KernelExecutionTaskE) | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| -   [cudaq::KernelThunkResultType | quantum_platform12get_exec_ctxEv) |
|     (C++                          | -   [c                            |
|     struct)]                      | udaq::quantum_platform::get_noise |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq21KernelThunkResultTypeE) |     function)](api/languages/c    |
| -   [cudaq::KernelThunkType (C++  | pp_api.html#_CPPv4N5cudaq16quantu |
|                                   | m_platform9get_noiseENSt6size_tE) |
| type)](api/languages/cpp_api.html | -   [cudaq:                       |
| #_CPPv4N5cudaq15KernelThunkTypeE) | :quantum_platform::get_num_qubits |
| -   [cudaq::kraus_channel (C++    |     (C++                          |
|                                   |                                   |
|  class)](api/languages/cpp_api.ht | function)](api/languages/cpp_api. |
| ml#_CPPv4N5cudaq13kraus_channelE) | html#_CPPv4NK5cudaq16quantum_plat |
| -   [cudaq::kraus_channel::empty  | form14get_num_qubitsENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_              |
|     function)]                    | platform::get_remote_capabilities |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4NK5cudaq13kraus_channel5emptyEv) |     function)                     |
| -   [cudaq::kraus_c               | ](api/languages/cpp_api.html#_CPP |
| hannel::generateUnitaryParameters | v4NK5cudaq16quantum_platform23get |
|     (C++                          | _remote_capabilitiesENSt6size_tE) |
|                                   | -   [cudaq::qua                   |
|    function)](api/languages/cpp_a | ntum_platform::get_runtime_target |
| pi.html#_CPPv4N5cudaq13kraus_chan |     (C++                          |
| nel25generateUnitaryParametersEv) |     function)](api/languages/cp   |
| -                                 | p_api.html#_CPPv4NK5cudaq16quantu |
|    [cudaq::kraus_channel::get_ops | m_platform18get_runtime_targetEv) |
|     (C++                          | -   [cud                          |
|     function)](a                  | aq::quantum_platform::is_emulated |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| K5cudaq13kraus_channel7get_opsEv) |                                   |
| -   [cud                          |    function)](api/languages/cpp_a |
| aq::kraus_channel::identity_flags | pi.html#_CPPv4NK5cudaq16quantum_p |
|     (C++                          | latform11is_emulatedENSt6size_tE) |
|     member)](api/lan              | -   [cudaq::                      |
| guages/cpp_api.html#_CPPv4N5cudaq | quantum_platform::is_library_mode |
| 13kraus_channel14identity_flagsE) |     (C++                          |
| -   [cud                          |     function)](api/languages      |
| aq::kraus_channel::is_identity_op | /cpp_api.html#_CPPv4NK5cudaq16qua |
|     (C++                          | ntum_platform15is_library_modeEv) |
|                                   | -   [c                            |
|    function)](api/languages/cpp_a | udaq::quantum_platform::is_remote |
| pi.html#_CPPv4NK5cudaq13kraus_cha |     (C++                          |
| nnel14is_identity_opENSt6size_tE) |     function)](api/languages/cp   |
| -   [cudaq::                      | p_api.html#_CPPv4NK5cudaq16quantu |
| kraus_channel::is_unitary_mixture | m_platform9is_remoteENSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     function)](api/languages      | q::quantum_platform::is_simulator |
| /cpp_api.html#_CPPv4NK5cudaq13kra |     (C++                          |
| us_channel18is_unitary_mixtureEv) |                                   |
| -   [cu                           |   function)](api/languages/cpp_ap |
| daq::kraus_channel::kraus_channel | i.html#_CPPv4NK5cudaq16quantum_pl |
|     (C++                          | atform12is_simulatorENSt6size_tE) |
|     function)](api/lang           | -   [c                            |
| uages/cpp_api.html#_CPPv4IDpEN5cu | udaq::quantum_platform::launchVQE |
| daq13kraus_channel13kraus_channel |     (C++                          |
| EDpRRNSt16initializer_listI1TEE), |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|  [\[1\]](api/languages/cpp_api.ht | N5cudaq16quantum_platform9launchV |
| ml#_CPPv4N5cudaq13kraus_channel13 | QEEKNSt6stringEPKvPN5cudaq8gradie |
| kraus_channelERK13kraus_channel), | ntERKN5cudaq7spin_opERN5cudaq9opt |
|     [\[2\]                        | imizerEKiKNSt6size_tENSt6size_tE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq:                       |
| v4N5cudaq13kraus_channel13kraus_c | :quantum_platform::list_platforms |
| hannelERKNSt6vectorI8kraus_opEE), |     (C++                          |
|     [\[3\]                        |     function)](api/languag        |
| ](api/languages/cpp_api.html#_CPP | es/cpp_api.html#_CPPv4N5cudaq16qu |
| v4N5cudaq13kraus_channel13kraus_c | antum_platform14list_platformsEv) |
| hannelERRNSt6vectorI8kraus_opEE), | -                                 |
|     [\[4\]](api/lan               |    [cudaq::quantum_platform::name |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 13kraus_channel13kraus_channelEv) |     function)](a                  |
| -                                 | pi/languages/cpp_api.html#_CPPv4N |
| [cudaq::kraus_channel::noise_type | K5cudaq16quantum_platform4nameEv) |
|     (C++                          | -   [                             |
|     member)](api                  | cudaq::quantum_platform::num_qpus |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel10noise_typeE) |     function)](api/l              |
| -                                 | anguages/cpp_api.html#_CPPv4NK5cu |
|   [cudaq::kraus_channel::op_names | daq16quantum_platform8num_qpusEv) |
|     (C++                          | -   [cudaq::                      |
|     member)](                     | quantum_platform::onRandomSeedSet |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq13kraus_channel8op_namesE) |                                   |
| -                                 | function)](api/languages/cpp_api. |
|  [cudaq::kraus_channel::operator= | html#_CPPv4N5cudaq16quantum_platf |
|     (C++                          | orm15onRandomSeedSetENSt6size_tE) |
|     function)](api/langua         | -   [cudaq:                       |
| ges/cpp_api.html#_CPPv4N5cudaq13k | :quantum_platform::reset_exec_ctx |
| raus_channelaSERK13kraus_channel) |     (C++                          |
| -   [c                            |     function)](api/languag        |
| udaq::kraus_channel::operator\[\] | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14reset_exec_ctxEv) |
|     function)](api/l              | -   [cud                          |
| anguages/cpp_api.html#_CPPv4N5cud | aq::quantum_platform::reset_noise |
| aq13kraus_channelixEKNSt6size_tE) |     (C++                          |
| -                                 |     function)](api/languages/cpp_ |
| [cudaq::kraus_channel::parameters | api.html#_CPPv4N5cudaq16quantum_p |
|     (C++                          | latform11reset_noiseENSt6size_tE) |
|     member)](api                  | -   [cuda                         |
| /languages/cpp_api.html#_CPPv4N5c | q::quantum_platform::set_exec_ctx |
| udaq13kraus_channel10parametersE) |     (C++                          |
| -   [cudaq::krau                  |     funct                         |
| s_channel::populateDefaultOpNames | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq16quantum_platform12 |
|     function)](api/languages/cp   | set_exec_ctxEP16ExecutionContext) |
| p_api.html#_CPPv4N5cudaq13kraus_c | -   [c                            |
| hannel22populateDefaultOpNamesEv) | udaq::quantum_platform::set_noise |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::probabilities |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)](api/la               | Pv4N5cudaq16quantum_platform9set_ |
| nguages/cpp_api.html#_CPPv4N5cuda | noiseEPK11noise_modelNSt6size_tE) |
| q13kraus_channel13probabilitiesE) | -   [cudaq::quantum_platfor       |
| -                                 | m::supports_explicit_measurements |
|  [cudaq::kraus_channel::push_back |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)](api                | anguages/cpp_api.html#_CPPv4NK5cu |
| /languages/cpp_api.html#_CPPv4N5c | daq16quantum_platform30supports_e |
| udaq13kraus_channel9push_backE8kr | xplicit_measurementsENSt6size_tE) |
| aus_opNSt8optionalINSt6stringEEE) | -   [cuda                         |
| -   [cudaq::kraus_channel::size   | q::quantum_platform::supports_jit |
|     (C++                          |     (C++                          |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP |   function)](api/languages/cpp_ap |
| v4NK5cudaq13kraus_channel4sizeEv) | i.html#_CPPv4NK5cudaq16quantum_pl |
| -   [                             | atform12supports_jitENSt6size_tE) |
| cudaq::kraus_channel::unitary_ops | -   [cudaq::quantum_pla           |
|     (C++                          | tform::supports_task_distribution |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     fu                            |
| daq13kraus_channel11unitary_opsE) | nction)](api/languages/cpp_api.ht |
| -   [cudaq::kraus_op (C++         | ml#_CPPv4NK5cudaq16quantum_platfo |
|     struct)](api/languages/cpp_   | rm26supports_task_distributionEv) |
| api.html#_CPPv4N5cudaq8kraus_opE) | -   [cudaq::quantum               |
| -   [cudaq::kraus_op::adjoint     | _platform::with_execution_context |
|     (C++                          |     (C++                          |
|     functi                        |     function)                     |
| on)](api/languages/cpp_api.html#_ | ](api/languages/cpp_api.html#_CPP |
| CPPv4NK5cudaq8kraus_op7adjointEv) | v4I0DpEN5cudaq16quantum_platform2 |
| -   [cudaq::kraus_op::data (C++   | 2with_execution_contextEDaR16Exec |
|                                   | utionContextRR8CallableDpRR4Args) |
|  member)](api/languages/cpp_api.h | -   [cudaq::QuantumTask (C++      |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     type)](api/languages/cpp_api. |
| -   [cudaq::kraus_op::kraus_op    | html#_CPPv4N5cudaq11QuantumTaskE) |
|     (C++                          | -   [cudaq::qubit (C++            |
|     func                          |     type)](api/languages/c        |
| tion)](api/languages/cpp_api.html | pp_api.html#_CPPv4N5cudaq5qubitE) |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | -   [cudaq::QubitConnectivity     |
| opERRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     ty                            |
|  [\[1\]](api/languages/cpp_api.ht | pe)](api/languages/cpp_api.html#_ |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | CPPv4N5cudaq17QubitConnectivityE) |
| pENSt6vectorIN5cudaq7complexEEE), | -   [cudaq::QubitEdge (C++        |
|     [\[2\]](api/l                 |     type)](api/languages/cpp_a    |
| anguages/cpp_api.html#_CPPv4N5cud | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| aq8kraus_op8kraus_opERK8kraus_op) | -   [cudaq::qudit (C++            |
| -   [cudaq::kraus_op::nCols (C++  |     clas                          |
|                                   | s)](api/languages/cpp_api.html#_C |
| member)](api/languages/cpp_api.ht | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | -   [cudaq::qudit::qudit (C++     |
| -   [cudaq::kraus_op::nRows (C++  |                                   |
|                                   | function)](api/languages/cpp_api. |
| member)](api/languages/cpp_api.ht | html#_CPPv4N5cudaq5qudit5quditEv) |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | -   [cudaq::QuEraRemoteRESTQPU    |
| -   [cudaq::kraus_op::operator=   |     (C++                          |
|     (C++                          |     clas                          |
|     function)                     | s)](api/languages/cpp_api.html#_C |
| ](api/languages/cpp_api.html#_CPP | PPv4N5cudaq18QuEraRemoteRESTQPUE) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cudaq::qvector (C++          |
| -   [cudaq::kraus_op::precision   |     class)                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     memb                          | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| er)](api/languages/cpp_api.html#_ | -   [cudaq::qvector::back (C++    |
| CPPv4N5cudaq8kraus_op9precisionE) |     function)](a                  |
| -   [cudaq::KrausSelection (C++   | pi/languages/cpp_api.html#_CPPv4N |
|     s                             | 5cudaq7qvector4backENSt6size_tE), |
| truct)](api/languages/cpp_api.htm |                                   |
| l#_CPPv4N5cudaq14KrausSelectionE) |   [\[1\]](api/languages/cpp_api.h |
| -   [cudaq:                       | tml#_CPPv4N5cudaq7qvector4backEv) |
| :KrausSelection::circuit_location | -   [cudaq::qvector::begin (C++   |
|     (C++                          |     fu                            |
|     member)](api/langua           | nction)](api/languages/cpp_api.ht |
| ges/cpp_api.html#_CPPv4N5cudaq14K | ml#_CPPv4N5cudaq7qvector5beginEv) |
| rausSelection16circuit_locationE) | -   [cudaq::qvector::clear (C++   |
| -                                 |     fu                            |
|  [cudaq::KrausSelection::is_error | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     member)](a                    | -   [cudaq::qvector::end (C++     |
| pi/languages/cpp_api.html#_CPPv4N |                                   |
| 5cudaq14KrausSelection8is_errorE) | function)](api/languages/cpp_api. |
| -   [cudaq::Kra                   | html#_CPPv4N5cudaq7qvector3endEv) |
| usSelection::kraus_operator_index | -   [cudaq::qvector::front (C++   |
|     (C++                          |     function)](ap                 |
|     member)](api/languages/       | i/languages/cpp_api.html#_CPPv4N5 |
| cpp_api.html#_CPPv4N5cudaq14Kraus | cudaq7qvector5frontENSt6size_tE), |
| Selection20kraus_operator_indexE) |                                   |
| -   [cuda                         |  [\[1\]](api/languages/cpp_api.ht |
| q::KrausSelection::KrausSelection | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     (C++                          | -   [cudaq::qvector::operator=    |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     functio                       |
| 5cudaq14KrausSelection14KrausSele | n)](api/languages/cpp_api.html#_C |
| ctionENSt6size_tENSt6vectorINSt6s | PPv4N5cudaq7qvectoraSERK7qvector) |
| ize_tEEENSt6stringENSt6size_tEb), | -   [cudaq::qvector::operator\[\] |
|     [\[1\]](api/langu             |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     function)                     |
| KrausSelection14KrausSelectionEv) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq7qvectorixEKNSt6size_tE) |
|   [cudaq::KrausSelection::op_name | -   [cudaq::qvector::qvector (C++ |
|     (C++                          |     function)](api/               |
|     member)](                     | languages/cpp_api.html#_CPPv4N5cu |
| api/languages/cpp_api.html#_CPPv4 | daq7qvector7qvectorENSt6size_tE), |
| N5cudaq14KrausSelection7op_nameE) |     [\[1\]](a                     |
| -   [                             | pi/languages/cpp_api.html#_CPPv4N |
| cudaq::KrausSelection::operator== | 5cudaq7qvector7qvectorERK5state), |
|     (C++                          |     [\[2\]](api                   |
|     function)](api/languages      | /languages/cpp_api.html#_CPPv4N5c |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | udaq7qvector7qvectorERK7qvector), |
| usSelectioneqERK14KrausSelection) |     [\[3\]](ap                    |
| -                                 | i/languages/cpp_api.html#_CPPv4N5 |
|    [cudaq::KrausSelection::qubits | cudaq7qvector7qvectorERR7qvector) |
|     (C++                          | -   [cudaq::qvector::size (C++    |
|     member)]                      |     fu                            |
| (api/languages/cpp_api.html#_CPPv | nction)](api/languages/cpp_api.ht |
| 4N5cudaq14KrausSelection6qubitsE) | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| -   [cudaq::KrausTrajectory (C++  | -   [cudaq::qvector::slice (C++   |
|     st                            |     function)](api/language       |
| ruct)](api/languages/cpp_api.html | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| #_CPPv4N5cudaq15KrausTrajectoryE) | tor5sliceENSt6size_tENSt6size_tE) |
| -                                 | -   [cudaq::qvector::value_type   |
|  [cudaq::KrausTrajectory::builder |     (C++                          |
|     (C++                          |     typ                           |
|     function)](ap                 | e)](api/languages/cpp_api.html#_C |
| i/languages/cpp_api.html#_CPPv4N5 | PPv4N5cudaq7qvector10value_typeE) |
| cudaq15KrausTrajectory7builderEv) | -   [cudaq::qview (C++            |
| -   [cu                           |     clas                          |
| daq::KrausTrajectory::countErrors | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     function)](api/lang           | -   [cudaq::qview::back (C++      |
| uages/cpp_api.html#_CPPv4NK5cudaq |     function)                     |
| 15KrausTrajectory11countErrorsEv) | ](api/languages/cpp_api.html#_CPP |
| -   [                             | v4N5cudaq5qview4backENSt6size_tE) |
| cudaq::KrausTrajectory::isOrdered | -   [cudaq::qview::begin (C++     |
|     (C++                          |                                   |
|     function)](api/l              | function)](api/languages/cpp_api. |
| anguages/cpp_api.html#_CPPv4NK5cu | html#_CPPv4N5cudaq5qview5beginEv) |
| daq15KrausTrajectory9isOrderedEv) | -   [cudaq::qview::end (C++       |
| -   [cudaq::                      |                                   |
| KrausTrajectory::kraus_selections |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4N5cudaq5qview3endEv) |
|     member)](api/languag          | -   [cudaq::qview::front (C++     |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |     function)](                   |
| ausTrajectory16kraus_selectionsE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq:                       | N5cudaq5qview5frontENSt6size_tE), |
| :KrausTrajectory::KrausTrajectory |                                   |
|     (C++                          |    [\[1\]](api/languages/cpp_api. |
|     function                      | html#_CPPv4N5cudaq5qview5frontEv) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qview::operator\[\]   |
| Pv4N5cudaq15KrausTrajectory15Krau |     (C++                          |
| sTrajectoryENSt6size_tENSt6vector |     functio                       |
| I14KrausSelectionEEdNSt6size_tE), | n)](api/languages/cpp_api.html#_C |
|     [\[1\]](api/languag           | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | -   [cudaq::qview::qview (C++     |
| ausTrajectory15KrausTrajectoryEv) |     functio                       |
| -   [cudaq::Kr                    | n)](api/languages/cpp_api.html#_C |
| ausTrajectory::measurement_counts | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     (C++                          |     [\[1                          |
|     member)](api/languages        | \]](api/languages/cpp_api.html#_C |
| /cpp_api.html#_CPPv4N5cudaq15Krau | PPv4N5cudaq5qview5qviewERK5qview) |
| sTrajectory18measurement_countsE) | -   [cudaq::qview::size (C++      |
| -   [cud                          |                                   |
| aq::KrausTrajectory::multiplicity | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4NK5cudaq5qview4sizeEv) |
|     member)](api/lan              | -   [cudaq::qview::slice (C++     |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/langua         |
| 15KrausTrajectory12multiplicityE) | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| -   [                             | iew5sliceENSt6size_tENSt6size_tE) |
| cudaq::KrausTrajectory::num_shots | -   [cudaq::qview::value_type     |
|     (C++                          |     (C++                          |
|     member)](api                  |     t                             |
| /languages/cpp_api.html#_CPPv4N5c | ype)](api/languages/cpp_api.html# |
| udaq15KrausTrajectory9num_shotsE) | _CPPv4N5cudaq5qview10value_typeE) |
| -   [c                            | -   [cudaq::range (C++            |
| udaq::KrausTrajectory::operator== |     fun                           |
|     (C++                          | ction)](api/languages/cpp_api.htm |
|     function)](api/languages/c    | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| pp_api.html#_CPPv4NK5cudaq15Kraus | orI11ElementTypeEE11ElementType), |
| TrajectoryeqERK15KrausTrajectory) |     [\[1\]](api/languages/cpp_    |
| -   [cu                           | api.html#_CPPv4I0EN5cudaq5rangeEN |
| daq::KrausTrajectory::probability | St6vectorI11ElementTypeEE11Elemen |
|     (C++                          | tType11ElementType11ElementType), |
|     member)](api/la               |     [                             |
| nguages/cpp_api.html#_CPPv4N5cuda | \[2\]](api/languages/cpp_api.html |
| q15KrausTrajectory11probabilityE) | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| -   [cuda                         | -   [cudaq::real (C++             |
| q::KrausTrajectory::trajectory_id |     type)](api/languages/         |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq4realE) |
|     member)](api/lang             | -   [cudaq::registry (C++         |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     type)](api/languages/cpp_     |
| 5KrausTrajectory13trajectory_idE) | api.html#_CPPv4N5cudaq8registryE) |
| -                                 | -                                 |
|   [cudaq::KrausTrajectory::weight |  [cudaq::registry::RegisteredType |
|     (C++                          |     (C++                          |
|     member)](                     |     class)](api/                  |
| api/languages/cpp_api.html#_CPPv4 | languages/cpp_api.html#_CPPv4I0EN |
| N5cudaq15KrausTrajectory6weightE) | 5cudaq8registry14RegisteredTypeE) |
| -                                 | -   [cudaq::RemoteCapabilities    |
|    [cudaq::KrausTrajectoryBuilder |     (C++                          |
|     (C++                          |     struc                         |
|     class)](                      | t)](api/languages/cpp_api.html#_C |
| api/languages/cpp_api.html#_CPPv4 | PPv4N5cudaq18RemoteCapabilitiesE) |
| N5cudaq22KrausTrajectoryBuilderE) | -   [cudaq::Remot                 |
| -   [cud                          | eCapabilities::RemoteCapabilities |
| aq::KrausTrajectoryBuilder::build |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api/lang           | _api.html#_CPPv4N5cudaq18RemoteCa |
| uages/cpp_api.html#_CPPv4NK5cudaq | pabilities18RemoteCapabilitiesEb) |
| 22KrausTrajectoryBuilder5buildEv) | -   [cudaq:                       |
| -   [cud                          | :RemoteCapabilities::stateOverlap |
| aq::KrausTrajectoryBuilder::setId |     (C++                          |
|     (C++                          |     member)](api/langua           |
|     function)](api/languages/cpp  | ges/cpp_api.html#_CPPv4N5cudaq18R |
| _api.html#_CPPv4N5cudaq22KrausTra | emoteCapabilities12stateOverlapE) |
| jectoryBuilder5setIdENSt6size_tE) | -                                 |
| -   [cudaq::Kraus                 |   [cudaq::RemoteCapabilities::vqe |
| TrajectoryBuilder::setProbability |     (C++                          |
|     (C++                          |     member)](                     |
|     function)](api/languages/cpp  | api/languages/cpp_api.html#_CPPv4 |
| _api.html#_CPPv4N5cudaq22KrausTra | N5cudaq18RemoteCapabilities3vqeE) |
| jectoryBuilder14setProbabilityEd) | -   [cudaq::RemoteRESTQPU (C++    |
| -   [cudaq::Krau                  |                                   |
| sTrajectoryBuilder::setSelections |  class)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq13RemoteRESTQPUE) |
|     function)](api/languag        | -   [cudaq::Resources (C++        |
| es/cpp_api.html#_CPPv4N5cudaq22Kr |     class)](api/languages/cpp_a   |
| ausTrajectoryBuilder13setSelectio | pi.html#_CPPv4N5cudaq9ResourcesE) |
| nsENSt6vectorI14KrausSelectionEE) | -   [cudaq::run (C++              |
| -   [cudaq::logical_observable    |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/languages/c    | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| pp_api.html#_CPPv4IDpEN5cudaq18lo | 5invoke_result_tINSt7decay_tI13Qu |
| gical_observableEvDpRR8MeasArgs), | antumKernelEEDpNSt7decay_tI4ARGSE |
|     [\[1\]](api/l                 | EEEEENSt6size_tERN5cudaq11noise_m |
| anguages/cpp_api.html#_CPPv4N5cud | odelERR13QuantumKernelDpRR4ARGS), |
| aq18logical_observableERKNSt6vect |     [\[1\]](api/langu             |
| orI14measure_resultEENSt6size_tE) | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| -   [cudaq::M2DSparseMatrix (C++  | daq3runENSt6vectorINSt15invoke_re |
|     st                            | sult_tINSt7decay_tI13QuantumKerne |
| ruct)](api/languages/cpp_api.html | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| #_CPPv4N5cudaq15M2DSparseMatrixE) | ize_tERR13QuantumKernelDpRR4ARGS) |
| -   [cudaq::M2OSparseMatrix (C++  | -   [cudaq::run_async (C++        |
|     st                            |     functio                       |
| ruct)](api/languages/cpp_api.html | n)](api/languages/cpp_api.html#_C |
| #_CPPv4N5cudaq15M2OSparseMatrixE) | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| -   [cudaq::matrix_callback (C++  | tureINSt6vectorINSt15invoke_resul |
|     c                             | t_tINSt7decay_tI13QuantumKernelEE |
| lass)](api/languages/cpp_api.html | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| #_CPPv4N5cudaq15matrix_callbackE) | ze_tENSt6size_tERN5cudaq11noise_m |
| -   [cudaq::matrix_handler (C++   | odelERR13QuantumKernelDpRR4ARGS), |
|                                   |     [\[1\]](api/la                |
| class)](api/languages/cpp_api.htm | nguages/cpp_api.html#_CPPv4I0DpEN |
| l#_CPPv4N5cudaq14matrix_handlerE) | 5cudaq9run_asyncENSt6futureINSt6v |
| -   [cudaq::mat                   | ectorINSt15invoke_result_tINSt7de |
| rix_handler::commutation_behavior | cay_tI13QuantumKernelEEDpNSt7deca |
|     (C++                          | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     struct)](api/languages/       | ize_tERR13QuantumKernelDpRR4ARGS) |
| cpp_api.html#_CPPv4N5cudaq14matri | -   [cudaq::RuntimeTarget (C++    |
| x_handler20commutation_behaviorE) |                                   |
| -                                 | struct)](api/languages/cpp_api.ht |
|    [cudaq::matrix_handler::define | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     (C++                          | -   [cudaq::sample (C++           |
|     function)](a                  |     function)](api/languages/c    |
| pi/languages/cpp_api.html#_CPPv4N | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| 5cudaq14matrix_handler6defineENSt | mpleE13sample_resultRK14sample_op |
| 6stringENSt6vectorINSt7int64_tEEE | tionsRR13QuantumKernelDpRR4Args), |
| RR15matrix_callbackRKNSt13unorder |     [\[1\                         |
| ed_mapINSt6stringENSt6stringEEE), | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4I0DpEN5cudaq6sampleE13sample_r |
| [\[1\]](api/languages/cpp_api.htm | esultRR13QuantumKernelDpRR4Args), |
| l#_CPPv4N5cudaq14matrix_handler6d |     [\                            |
| efineENSt6stringENSt6vectorINSt7i | [2\]](api/languages/cpp_api.html# |
| nt64_tEEERR15matrix_callbackRR20d | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| iag_matrix_callbackRKNSt13unorder | ize_tERR13QuantumKernelDpRR4Args) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::sample_options (C++   |
|     [\[2\]](                      |     s                             |
| api/languages/cpp_api.html#_CPPv4 | truct)](api/languages/cpp_api.htm |
| N5cudaq14matrix_handler6defineENS | l#_CPPv4N5cudaq14sample_optionsE) |
| t6stringENSt6vectorINSt7int64_tEE | -   [cudaq::sample_result (C++    |
| ERR15matrix_callbackRRNSt13unorde |                                   |
| red_mapINSt6stringENSt6stringEEE) |  class)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq13sample_resultE) |
|   [cudaq::matrix_handler::degrees | -   [cudaq::sample_result::append |
|     (C++                          |     (C++                          |
|     function)](ap                 |     function)](api/languages/cpp_ |
| i/languages/cpp_api.html#_CPPv4NK | api.html#_CPPv4N5cudaq13sample_re |
| 5cudaq14matrix_handler7degreesEv) | sult6appendERK15ExecutionResultb) |
| -                                 | -   [cudaq::sample_result::begin  |
|  [cudaq::matrix_handler::displace |     (C++                          |
|     (C++                          |     function)]                    |
|     function)](api/language       | (api/languages/cpp_api.html#_CPPv |
| s/cpp_api.html#_CPPv4N5cudaq14mat | 4N5cudaq13sample_result5beginEv), |
| rix_handler8displaceENSt6size_tE) |     [\[1\]]                       |
| -   [cudaq::matrix                | (api/languages/cpp_api.html#_CPPv |
| _handler::get_expected_dimensions | 4NK5cudaq13sample_result5beginEv) |
|     (C++                          | -   [cudaq::sample_result::cbegin |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](                   |
| pi.html#_CPPv4NK5cudaq14matrix_ha | api/languages/cpp_api.html#_CPPv4 |
| ndler23get_expected_dimensionsEv) | NK5cudaq13sample_result6cbeginEv) |
| -   [cudaq::matrix_ha             | -   [cudaq::sample_result::cend   |
| ndler::get_parameter_descriptions |     (C++                          |
|     (C++                          |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
| function)](api/languages/cpp_api. | v4NK5cudaq13sample_result4cendEv) |
| html#_CPPv4NK5cudaq14matrix_handl | -   [cudaq::sample_result::clear  |
| er26get_parameter_descriptionsEv) |     (C++                          |
| -   [c                            |     function)                     |
| udaq::matrix_handler::instantiate | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq13sample_result5clearEv) |
|     function)](a                  | -   [cudaq::sample_result::count  |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14matrix_handler11instantia |     function)](                   |
| teENSt6stringERKNSt6vectorINSt6si | api/languages/cpp_api.html#_CPPv4 |
| ze_tEEERK20commutation_behavior), | NK5cudaq13sample_result5countENSt |
|     [\[1\]](                      | 11string_viewEKNSt11string_viewE) |
| api/languages/cpp_api.html#_CPPv4 | -   [                             |
| N5cudaq14matrix_handler11instanti | cudaq::sample_result::deserialize |
| ateENSt6stringERRNSt6vectorINSt6s |     (C++                          |
| ize_tEEERK20commutation_behavior) |     functio                       |
| -   [cuda                         | n)](api/languages/cpp_api.html#_C |
| q::matrix_handler::matrix_handler | PPv4N5cudaq13sample_result11deser |
|     (C++                          | ializeERNSt6vectorINSt6size_tEEE) |
|     function)](api/languag        | -   [cudaq::sample_result::dump   |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     (C++                          |
| ble_if_tINSt12is_base_of_vI16oper |     function)](api/languag        |
| ator_handler1TEEbEEEN5cudaq14matr | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ix_handler14matrix_handlerERK1T), | ample_result4dumpERNSt7ostreamE), |
|     [\[1\]](ap                    |     [\[1\]                        |
| i/languages/cpp_api.html#_CPPv4I0 | ](api/languages/cpp_api.html#_CPP |
| _NSt11enable_if_tINSt12is_base_of | v4NK5cudaq13sample_result4dumpEv) |
| _vI16operator_handler1TEEbEEEN5cu | -   [cudaq::sample_result::end    |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erERK1TRK20commutation_behavior), |     function                      |
|     [\[2\]](api/languages/cpp_ap  | )](api/languages/cpp_api.html#_CP |
| i.html#_CPPv4N5cudaq14matrix_hand | Pv4N5cudaq13sample_result3endEv), |
| ler14matrix_handlerENSt6size_tE), |     [\[1\                         |
|     [\[3\]](api/                  | ]](api/languages/cpp_api.html#_CP |
| languages/cpp_api.html#_CPPv4N5cu | Pv4NK5cudaq13sample_result3endEv) |
| daq14matrix_handler14matrix_handl | -   [                             |
| erENSt6stringERKNSt6vectorINSt6si | cudaq::sample_result::expectation |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[4\]](api/                  |     f                             |
| languages/cpp_api.html#_CPPv4N5cu | unction)](api/languages/cpp_api.h |
| daq14matrix_handler14matrix_handl | tml#_CPPv4NK5cudaq13sample_result |
| erENSt6stringERRNSt6vectorINSt6si | 11expectationEKNSt11string_viewE) |
| ze_tEEERK20commutation_behavior), | -   [cuda                         |
|     [\                            | q::sample_result::get_annotations |
| [5\]](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq14matrix_handler14ma |     function)](api/langua         |
| trix_handlerERK14matrix_handler), | ges/cpp_api.html#_CPPv4NK5cudaq13 |
|     [                             | sample_result15get_annotationsEv) |
| \[6\]](api/languages/cpp_api.html | -   [c                            |
| #_CPPv4N5cudaq14matrix_handler14m | udaq::sample_result::get_marginal |
| atrix_handlerERR14matrix_handler) |     (C++                          |
| -                                 |     function)](api/languages/cpp_ |
|  [cudaq::matrix_handler::momentum | api.html#_CPPv4NK5cudaq13sample_r |
|     (C++                          | esult12get_marginalERKNSt6vectorI |
|     function)](api/language       | NSt6size_tEEEKNSt11string_viewE), |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     [\[1\]](api/languages/cpp_    |
| rix_handler8momentumENSt6size_tE) | api.html#_CPPv4NK5cudaq13sample_r |
| -                                 | esult12get_marginalERRKNSt6vector |
|    [cudaq::matrix_handler::number | INSt6size_tEEEKNSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|     function)](api/langua         | q::sample_result::get_total_shots |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6numberENSt6size_tE) |     function)](api/langua         |
| -                                 | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| [cudaq::matrix_handler::operator= | sample_result15get_total_shotsEv) |
|     (C++                          | -   [cuda                         |
|     fun                           | q::sample_result::has_even_parity |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     fun                           |
| NSt7is_sameI1T14matrix_handlerE5v | ction)](api/languages/cpp_api.htm |
| alueENSt12is_base_of_vI16operator | l#_CPPv4N5cudaq13sample_result15h |
| _handler1TEEEbEEEN5cudaq14matrix_ | as_even_parityENSt11string_viewE) |
| handleraSER14matrix_handlerRK1T), | -   [cuda                         |
|     [\[1\]](api/languages         | q::sample_result::has_expectation |
| /cpp_api.html#_CPPv4N5cudaq14matr |     (C++                          |
| ix_handleraSERK14matrix_handler), |     funct                         |
|     [\[2\]](api/language          | ion)](api/languages/cpp_api.html# |
| s/cpp_api.html#_CPPv4N5cudaq14mat | _CPPv4NK5cudaq13sample_result15ha |
| rix_handleraSERR14matrix_handler) | s_expectationEKNSt11string_viewE) |
| -   [                             | -   [cu                           |
| cudaq::matrix_handler::operator== | daq::sample_result::most_probable |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     fun                           |
| /cpp_api.html#_CPPv4NK5cudaq14mat | ction)](api/languages/cpp_api.htm |
| rix_handlereqERK14matrix_handler) | l#_CPPv4NK5cudaq13sample_result13 |
| -                                 | most_probableEKNSt11string_viewE) |
|    [cudaq::matrix_handler::parity | -                                 |
|     (C++                          | [cudaq::sample_result::operator+= |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     function)](api/langua         |
| atrix_handler6parityENSt6size_tE) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -                                 | ample_resultpLERK13sample_result) |
|  [cudaq::matrix_handler::position | -                                 |
|     (C++                          |  [cudaq::sample_result::operator= |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     function)](api/langua         |
| rix_handler8positionENSt6size_tE) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -   [cudaq::                      | ample_resultaSERR13sample_result) |
| matrix_handler::remove_definition | -                                 |
|     (C++                          | [cudaq::sample_result::operator== |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/languag        |
| ml#_CPPv4N5cudaq14matrix_handler1 | es/cpp_api.html#_CPPv4NK5cudaq13s |
| 7remove_definitionERKNSt6stringE) | ample_resulteqERK13sample_result) |
| -                                 | -   [                             |
|   [cudaq::matrix_handler::squeeze | cudaq::sample_result::probability |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)](api/lan            |
| es/cpp_api.html#_CPPv4N5cudaq14ma | guages/cpp_api.html#_CPPv4NK5cuda |
| trix_handler7squeezeENSt6size_tE) | q13sample_result11probabilityENSt |
| -   [cudaq::m                     | 11string_viewEKNSt11string_viewE) |
| atrix_handler::to_diagonal_matrix | -   [cud                          |
|     (C++                          | aq::sample_result::register_names |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     function)](api/langu          |
| 14matrix_handler18to_diagonal_mat | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| rixERNSt13unordered_mapINSt6size_ | 3sample_result14register_namesEv) |
| tENSt7int64_tEEERKNSt13unordered_ | -                                 |
| mapINSt6stringENSt7complexIdEEEE) |    [cudaq::sample_result::reorder |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::to_matrix |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function)                     | ample_result7reorderERKNSt6vector |
| ](api/languages/cpp_api.html#_CPP | INSt6size_tEEEKNSt11string_viewE) |
| v4NK5cudaq14matrix_handler9to_mat | -   [cu                           |
| rixERNSt13unordered_mapINSt6size_ | daq::sample_result::sample_result |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     function)](api/               |
| -                                 | languages/cpp_api.html#_CPPv4N5cu |
| [cudaq::matrix_handler::to_string | daq13sample_result13sample_result |
|     (C++                          | E16CountsDictionary10cudaq_json), |
|     function)](api/               |     [                             |
| languages/cpp_api.html#_CPPv4NK5c | \[1\]](api/languages/cpp_api.html |
| udaq14matrix_handler9to_stringEb) | #_CPPv4N5cudaq13sample_result13sa |
| -                                 | mple_resultERK15ExecutionResult), |
| [cudaq::matrix_handler::unique_id |     [\[2\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     function)](api/               | q13sample_result13sample_resultER |
| languages/cpp_api.html#_CPPv4NK5c | KNSt6vectorI15ExecutionResultEE), |
| udaq14matrix_handler9unique_idEv) |                                   |
| -   [cudaq:                       |  [\[3\]](api/languages/cpp_api.ht |
| :matrix_handler::\~matrix_handler | ml#_CPPv4N5cudaq13sample_result13 |
|     (C++                          | sample_resultERR13sample_result), |
|     functi                        |     [                             |
| on)](api/languages/cpp_api.html#_ | \[4\]](api/languages/cpp_api.html |
| CPPv4N5cudaq14matrix_handlerD0Ev) | #_CPPv4N5cudaq13sample_result13sa |
| -   [cudaq::matrix_op (C++        | mple_resultERR15ExecutionResult), |
|     type)](api/languages/cpp_a    |     [\[5\]](api/lan               |
| pi.html#_CPPv4N5cudaq9matrix_opE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [cudaq::matrix_op_term (C++   | 13sample_result13sample_resultEdR |
|                                   | KNSt6vectorI15ExecutionResultEE), |
|  type)](api/languages/cpp_api.htm |     [\[6\]](api/lan               |
| l#_CPPv4N5cudaq14matrix_op_termE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -                                 | 13sample_result13sample_resultEv) |
|    [cudaq::mdiag_operator_handler | -                                 |
|     (C++                          |  [cudaq::sample_result::serialize |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api                |
| N5cudaq22mdiag_operator_handlerE) | /languages/cpp_api.html#_CPPv4NK5 |
| -   [cudaq::measure_handle (C++   | cudaq13sample_result9serializeEv) |
|                                   | -   [cudaq::sample_result::size   |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14measure_handleE) |     function)](api/languages/c    |
| -   [cudaq::measure_result (C++   | pp_api.html#_CPPv4NK5cudaq13sampl |
|                                   | e_result4sizeEKNSt11string_viewE) |
|  type)](api/languages/cpp_api.htm | -   [cudaq::sample_result::to_map |
| l#_CPPv4N5cudaq14measure_resultE) |     (C++                          |
| -   [cudaq::mpi (C++              |     function)](api/languages/cpp  |
|     type)](api/languages          | _api.html#_CPPv4NK5cudaq13sample_ |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | result6to_mapEKNSt11string_viewE) |
| -   [cudaq::mpi::all_gather (C++  | -   [cuda                         |
|     fu                            | q::sample_result::\~sample_result |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     funct                         |
| RNSt6vectorIdEERKNSt6vectorIdEE), | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq13sample_resultD0Ev) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::scalar_callback (C++  |
| tml#_CPPv4N5cudaq3mpi10all_gather |     c                             |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | lass)](api/languages/cpp_api.html |
| -   [cudaq::mpi::all_reduce (C++  | #_CPPv4N5cudaq15scalar_callbackE) |
|                                   | -   [c                            |
|  function)](api/languages/cpp_api | udaq::scalar_callback::operator() |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     (C++                          |
| reduceE1TRK1TRK14BinaryFunction), |     function)](api/language       |
|     [\[1\]](api/langu             | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| ages/cpp_api.html#_CPPv4I00EN5cud | alar_callbackclERKNSt13unordered_ |
| aq3mpi10all_reduceE1TRK1TRK4Func) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::mpi::broadcast (C++   | -   [                             |
|     function)](api/               | cudaq::scalar_callback::operator= |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq3mpi9broadcastERNSt6stringEi), |     function)](api/languages/c    |
|     [\[1\]](api/la                | pp_api.html#_CPPv4N5cudaq15scalar |
| nguages/cpp_api.html#_CPPv4N5cuda | _callbackaSERK15scalar_callback), |
| q3mpi9broadcastERNSt6vectorIdEEi) |     [\[1\]](api/languages/        |
| -   [cudaq::mpi::finalize (C++    | cpp_api.html#_CPPv4N5cudaq15scala |
|     f                             | r_callbackaSERR15scalar_callback) |
| unction)](api/languages/cpp_api.h | -   [cudaq:                       |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | :scalar_callback::scalar_callback |
| -   [cudaq::mpi::initialize (C++  |     (C++                          |
|     function                      |     function)](api/languag        |
| )](api/languages/cpp_api.html#_CP | es/cpp_api.html#_CPPv4I0_NSt11ena |
| Pv4N5cudaq3mpi10initializeEiPPc), | ble_if_tINSt16is_invocable_r_vINS |
|     [                             | t7complexIdEE8CallableRKNSt13unor |
| \[1\]](api/languages/cpp_api.html | dered_mapINSt6stringENSt7complexI |
| #_CPPv4N5cudaq3mpi10initializeEv) | dEEEEEEbEEEN5cudaq15scalar_callba |
| -   [cudaq::mpi::is_initialized   | ck15scalar_callbackERR8Callable), |
|     (C++                          |     [\[1\                         |
|     function                      | ]](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4N5cudaq15scalar_callback15scal |
| Pv4N5cudaq3mpi14is_initializedEv) | ar_callbackERK15scalar_callback), |
| -   [cudaq::mpi::num_ranks (C++   |     [\[2                          |
|     fu                            | \]](api/languages/cpp_api.html#_C |
| nction)](api/languages/cpp_api.ht | PPv4N5cudaq15scalar_callback15sca |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | lar_callbackERR15scalar_callback) |
| -   [cudaq::mpi::rank (C++        | -   [cudaq::scalar_operator (C++  |
|                                   |     c                             |
|    function)](api/languages/cpp_a | lass)](api/languages/cpp_api.html |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | #_CPPv4N5cudaq15scalar_operatorE) |
| -   [cudaq::noise_model (C++      | -                                 |
|                                   | [cudaq::scalar_operator::evaluate |
|    class)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq11noise_modelE) |                                   |
| -   [cudaq::n                     |    function)](api/languages/cpp_a |
| oise_model::add_all_qubit_channel | pi.html#_CPPv4NK5cudaq15scalar_op |
|     (C++                          | erator8evaluateERKNSt13unordered_ |
|     function)](api                | mapINSt6stringENSt7complexIdEEEE) |
| /languages/cpp_api.html#_CPPv4IDp | -   [cudaq::scalar_ope            |
| EN5cudaq11noise_model21add_all_qu | rator::get_parameter_descriptions |
| bit_channelEvRK13kraus_channeli), |     (C++                          |
|     [\[1\]](api/langua            |     f                             |
| ges/cpp_api.html#_CPPv4N5cudaq11n | unction)](api/languages/cpp_api.h |
| oise_model21add_all_qubit_channel | tml#_CPPv4NK5cudaq15scalar_operat |
| ERKNSt6stringERK13kraus_channeli) | or26get_parameter_descriptionsEv) |
| -                                 | -   [cu                           |
|  [cudaq::noise_model::add_channel | daq::scalar_operator::is_constant |
|     (C++                          |     (C++                          |
|     funct                         |     function)](api/lang           |
| ion)](api/languages/cpp_api.html# | uages/cpp_api.html#_CPPv4NK5cudaq |
| _CPPv4IDpEN5cudaq11noise_model11a | 15scalar_operator11is_constantEv) |
| dd_channelEvRK15PredicateFuncTy), | -   [c                            |
|     [\[1\]](api/languages/cpp_    | udaq::scalar_operator::operator\* |
| api.html#_CPPv4IDpEN5cudaq11noise |     (C++                          |
| _model11add_channelEvRKNSt6vector |     function                      |
| INSt6size_tEEERK13kraus_channel), | )](api/languages/cpp_api.html#_CP |
|     [\[2\]](ap                    | Pv4N5cudaq15scalar_operatormlENSt |
| i/languages/cpp_api.html#_CPPv4N5 | 7complexIdEERK15scalar_operator), |
| cudaq11noise_model11add_channelER |     [\[1\                         |
| KNSt6stringERK15PredicateFuncTy), | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_operatormlENSt |
| [\[3\]](api/languages/cpp_api.htm | 7complexIdEERR15scalar_operator), |
| l#_CPPv4N5cudaq11noise_model11add |     [\[2\]](api/languages/cp      |
| _channelERKNSt6stringERKNSt6vecto | p_api.html#_CPPv4N5cudaq15scalar_ |
| rINSt6size_tEEERK13kraus_channel) | operatormlEdRK15scalar_operator), |
| -   [cudaq::noise_model::empty    |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     function                      | operatormlEdRR15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[4\]](api/languages         |
| Pv4NK5cudaq11noise_model5emptyEv) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -                                 | alar_operatormlENSt7complexIdEE), |
| [cudaq::noise_model::get_channels |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|     function)](api/l              | _operatormlERK15scalar_operator), |
| anguages/cpp_api.html#_CPPv4I0ENK |     [\[6\]]                       |
| 5cudaq11noise_model12get_channels | (api/languages/cpp_api.html#_CPPv |
| ENSt6vectorI13kraus_channelEERKNS | 4NKR5cudaq15scalar_operatormlEd), |
| t6vectorINSt6size_tEEERKNSt6vecto |     [\[7\]](api/language          |
| rINSt6size_tEEERKNSt6vectorIdEE), | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     [\[1\]](api/languages/cpp_a   | alar_operatormlENSt7complexIdEE), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[8\]](api/languages/cp      |
| el12get_channelsERKNSt6stringERKN | p_api.html#_CPPv4NO5cudaq15scalar |
| St6vectorINSt6size_tEEERKNSt6vect | _operatormlERK15scalar_operator), |
| orINSt6size_tEEERKNSt6vectorIdEE) |     [\[9\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|  [cudaq::noise_model::noise_model | Pv4NO5cudaq15scalar_operatormlEd) |
|     (C++                          | -   [cu                           |
|     function)](api                | daq::scalar_operator::operator\*= |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq11noise_model11noise_modelEv) |     function)](api/languag        |
| -   [cu                           | es/cpp_api.html#_CPPv4N5cudaq15sc |
| daq::noise_model::PredicateFuncTy | alar_operatormLENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     type)](api/la                 | pp_api.html#_CPPv4N5cudaq15scalar |
| nguages/cpp_api.html#_CPPv4N5cuda | _operatormLERK15scalar_operator), |
| q11noise_model15PredicateFuncTyE) |     [\[2                          |
| -   [cud                          | \]](api/languages/cpp_api.html#_C |
| aq::noise_model::register_channel | PPv4N5cudaq15scalar_operatormLEd) |
|     (C++                          | -   [                             |
|     function)](api/languages      | cudaq::scalar_operator::operator+ |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     (C++                          |
| noise_model16register_channelEvv) |     function                      |
| -   [cudaq::                      | )](api/languages/cpp_api.html#_CP |
| noise_model::requires_constructor | Pv4N5cudaq15scalar_operatorplENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     type)](api/languages/cp       |     [\[1\                         |
| p_api.html#_CPPv4I0DpEN5cudaq11no | ]](api/languages/cpp_api.html#_CP |
| ise_model20requires_constructorE) | Pv4N5cudaq15scalar_operatorplENSt |
| -   [cudaq::noise_model_type (C++ | 7complexIdEERR15scalar_operator), |
|     e                             |     [\[2\]](api/languages/cp      |
| num)](api/languages/cpp_api.html# | p_api.html#_CPPv4N5cudaq15scalar_ |
| _CPPv4N5cudaq16noise_model_typeE) | operatorplEdRK15scalar_operator), |
| -   [cudaq::no                    |     [\[3\]](api/languages/cp      |
| ise_model_type::amplitude_damping | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatorplEdRR15scalar_operator), |
|     enumerator)](api/languages    |     [\[4\]](api/languages         |
| /cpp_api.html#_CPPv4N5cudaq16nois | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| e_model_type17amplitude_dampingE) | alar_operatorplENSt7complexIdEE), |
| -   [cudaq::noise_mode            |     [\[5\]](api/languages/cpp     |
| l_type::amplitude_damping_channel | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatorplERK15scalar_operator), |
|     e                             |     [\[6\]]                       |
| numerator)](api/languages/cpp_api | (api/languages/cpp_api.html#_CPPv |
| .html#_CPPv4N5cudaq16noise_model_ | 4NKR5cudaq15scalar_operatorplEd), |
| type25amplitude_damping_channelE) |     [\[7\]]                       |
| -   [cudaq::n                     | (api/languages/cpp_api.html#_CPPv |
| oise_model_type::bit_flip_channel | 4NKR5cudaq15scalar_operatorplEv), |
|     (C++                          |     [\[8\]](api/language          |
|     enumerator)](api/language     | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| s/cpp_api.html#_CPPv4N5cudaq16noi | alar_operatorplENSt7complexIdEE), |
| se_model_type16bit_flip_channelE) |     [\[9\]](api/languages/cp      |
| -   [cudaq::                      | p_api.html#_CPPv4NO5cudaq15scalar |
| noise_model_type::depolarization1 | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[10\]                       |
|     enumerator)](api/languag      | ](api/languages/cpp_api.html#_CPP |
| es/cpp_api.html#_CPPv4N5cudaq16no | v4NO5cudaq15scalar_operatorplEd), |
| ise_model_type15depolarization1E) |     [\[11\                        |
| -   [cudaq::                      | ]](api/languages/cpp_api.html#_CP |
| noise_model_type::depolarization2 | Pv4NO5cudaq15scalar_operatorplEv) |
|     (C++                          | -   [c                            |
|     enumerator)](api/languag      | udaq::scalar_operator::operator+= |
| es/cpp_api.html#_CPPv4N5cudaq16no |     (C++                          |
| ise_model_type15depolarization2E) |     function)](api/languag        |
| -   [cudaq::noise_m               | es/cpp_api.html#_CPPv4N5cudaq15sc |
| odel_type::depolarization_channel | alar_operatorpLENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|                                   | pp_api.html#_CPPv4N5cudaq15scalar |
|   enumerator)](api/languages/cpp_ | _operatorpLERK15scalar_operator), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[2                          |
| el_type22depolarization_channelE) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq15scalar_operatorpLEd) |
|  [cudaq::noise_model_type::pauli1 | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator- |
|     enumerator)](a                |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function                      |
| 5cudaq16noise_model_type6pauli1E) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatormiENSt |
|  [cudaq::noise_model_type::pauli2 | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](a                | ]](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4N5cudaq15scalar_operatormiENSt |
| 5cudaq16noise_model_type6pauli2E) | 7complexIdEERR15scalar_operator), |
| -   [cudaq                        |     [\[2\]](api/languages/cp      |
| ::noise_model_type::phase_damping | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRK15scalar_operator), |
|     enumerator)](api/langu        |     [\[3\]](api/languages/cp      |
| ages/cpp_api.html#_CPPv4N5cudaq16 | p_api.html#_CPPv4N5cudaq15scalar_ |
| noise_model_type13phase_dampingE) | operatormiEdRR15scalar_operator), |
| -   [cudaq::noi                   |     [\[4\]](api/languages         |
| se_model_type::phase_flip_channel | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     enumerator)](api/languages/   |     [\[5\]](api/languages/cpp     |
| cpp_api.html#_CPPv4N5cudaq16noise | _api.html#_CPPv4NKR5cudaq15scalar |
| _model_type18phase_flip_channelE) | _operatormiERK15scalar_operator), |
| -                                 |     [\[6\]]                       |
| [cudaq::noise_model_type::unknown | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEd), |
|     enumerator)](ap               |     [\[7\]]                       |
| i/languages/cpp_api.html#_CPPv4N5 | (api/languages/cpp_api.html#_CPPv |
| cudaq16noise_model_type7unknownE) | 4NKR5cudaq15scalar_operatormiEv), |
| -                                 |     [\[8\]](api/language          |
| [cudaq::noise_model_type::x_error | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     enumerator)](ap               |     [\[9\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4NO5cudaq15scalar |
| cudaq16noise_model_type7x_errorE) | _operatormiERK15scalar_operator), |
| -                                 |     [\[10\]                       |
| [cudaq::noise_model_type::y_error | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatormiEd), |
|     enumerator)](ap               |     [\[11\                        |
| i/languages/cpp_api.html#_CPPv4N5 | ]](api/languages/cpp_api.html#_CP |
| cudaq16noise_model_type7y_errorE) | Pv4NO5cudaq15scalar_operatormiEv) |
| -                                 | -   [c                            |
| [cudaq::noise_model_type::z_error | udaq::scalar_operator::operator-= |
|     (C++                          |     (C++                          |
|     enumerator)](ap               |     function)](api/languag        |
| i/languages/cpp_api.html#_CPPv4N5 | es/cpp_api.html#_CPPv4N5cudaq15sc |
| cudaq16noise_model_type7z_errorE) | alar_operatormIENSt7complexIdEE), |
| -   [cudaq::num_available_gpus    |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function                      | _operatormIERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[2                          |
| Pv4N5cudaq18num_available_gpusEv) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::observe (C++          | PPv4N5cudaq15scalar_operatormIEd) |
|     function)]                    | -   [                             |
| (api/languages/cpp_api.html#_CPPv | cudaq::scalar_operator::operator/ |
| 4I00DpEN5cudaq7observeENSt6vector |     (C++                          |
| I14observe_resultEERR13QuantumKer |     function                      |
| nelRK15SpinOpContainerDpRR4Args), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/languages/cpp_ap  | Pv4N5cudaq15scalar_operatordvENSt |
| i.html#_CPPv4I0DpEN5cudaq7observe | 7complexIdEERK15scalar_operator), |
| E14observe_resultNSt6size_tERR13Q |     [\[1\                         |
| uantumKernelRK7spin_opDpRR4Args), | ]](api/languages/cpp_api.html#_CP |
|     [\[                           | Pv4N5cudaq15scalar_operatordvENSt |
| 2\]](api/languages/cpp_api.html#_ | 7complexIdEERR15scalar_operator), |
| CPPv4I0DpEN5cudaq7observeE14obser |     [\[2\]](api/languages/cp      |
| ve_resultRK15observe_optionsRR13Q | p_api.html#_CPPv4N5cudaq15scalar_ |
| uantumKernelRK7spin_opDpRR4Args), | operatordvEdRK15scalar_operator), |
|     [\[3\]](api/lang              |     [\[3\]](api/languages/cp      |
| uages/cpp_api.html#_CPPv4I0DpEN5c | p_api.html#_CPPv4N5cudaq15scalar_ |
| udaq7observeE14observe_resultRR13 | operatordvEdRR15scalar_operator), |
| QuantumKernelRK7spin_opDpRR4Args) |     [\[4\]](api/languages         |
| -   [cudaq::observe_options (C++  | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     st                            | alar_operatordvENSt7complexIdEE), |
| ruct)](api/languages/cpp_api.html |     [\[5\]](api/languages/cpp     |
| #_CPPv4N5cudaq15observe_optionsE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::observe_result (C++   | _operatordvERK15scalar_operator), |
|                                   |     [\[6\]]                       |
| class)](api/languages/cpp_api.htm | (api/languages/cpp_api.html#_CPPv |
| l#_CPPv4N5cudaq14observe_resultE) | 4NKR5cudaq15scalar_operatordvEd), |
| -                                 |     [\[7\]](api/language          |
|    [cudaq::observe_result::counts | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatordvENSt7complexIdEE), |
|     function)](api/languages/c    |     [\[8\]](api/languages/cp      |
| pp_api.html#_CPPv4N5cudaq14observ | p_api.html#_CPPv4NO5cudaq15scalar |
| e_result6countsERK12spin_op_term) | _operatordvERK15scalar_operator), |
| -   [cudaq::observe_result::dump  |     [\[9\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)                     | Pv4NO5cudaq15scalar_operatordvEd) |
| ](api/languages/cpp_api.html#_CPP | -   [c                            |
| v4N5cudaq14observe_result4dumpEv) | udaq::scalar_operator::operator/= |
| -   [c                            |     (C++                          |
| udaq::observe_result::expectation |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|                                   | alar_operatordVENSt7complexIdEE), |
| function)](api/languages/cpp_api. |     [\[1\]](api/languages/c       |
| html#_CPPv4N5cudaq14observe_resul | pp_api.html#_CPPv4N5cudaq15scalar |
| t11expectationERK12spin_op_term), | _operatordVERK15scalar_operator), |
|     [\[1\]](api/la                |     [\[2                          |
| nguages/cpp_api.html#_CPPv4N5cuda | \]](api/languages/cpp_api.html#_C |
| q14observe_result11expectationEv) | PPv4N5cudaq15scalar_operatordVEd) |
| -   [cuda                         | -   [                             |
| q::observe_result::id_coefficient | cudaq::scalar_operator::operator= |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     function)](api/languages/c    |
| ages/cpp_api.html#_CPPv4N5cudaq14 | pp_api.html#_CPPv4N5cudaq15scalar |
| observe_result14id_coefficientEv) | _operatoraSERK15scalar_operator), |
| -   [cuda                         |     [\[1\]](api/languages/        |
| q::observe_result::observe_result | cpp_api.html#_CPPv4N5cudaq15scala |
|     (C++                          | r_operatoraSERR15scalar_operator) |
|                                   | -   [c                            |
|   function)](api/languages/cpp_ap | udaq::scalar_operator::operator== |
| i.html#_CPPv4N5cudaq14observe_res |     (C++                          |
| ult14observe_resultEdRK7spin_op), |     function)](api/languages/c    |
|     [\[1\]](a                     | pp_api.html#_CPPv4NK5cudaq15scala |
| pi/languages/cpp_api.html#_CPPv4N | r_operatoreqERK15scalar_operator) |
| 5cudaq14observe_result14observe_r | -   [cudaq:                       |
| esultEdRK7spin_op13sample_result) | :scalar_operator::scalar_operator |
| -                                 |     (C++                          |
|  [cudaq::observe_result::operator |     func                          |
|     double (C++                   | tion)](api/languages/cpp_api.html |
|     functio                       | #_CPPv4N5cudaq15scalar_operator15 |
| n)](api/languages/cpp_api.html#_C | scalar_operatorENSt7complexIdEE), |
| PPv4N5cudaq14observe_resultcvdEv) |     [\[1\]](api/langu             |
| -                                 | ages/cpp_api.html#_CPPv4N5cudaq15 |
|  [cudaq::observe_result::raw_data | scalar_operator15scalar_operatorE |
|     (C++                          | RK15scalar_callbackRRNSt13unorder |
|     function)](ap                 | ed_mapINSt6stringENSt6stringEEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[2\                         |
| cudaq14observe_result8raw_dataEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::operator_handler (C++ | Pv4N5cudaq15scalar_operator15scal |
|     cl                            | ar_operatorERK15scalar_operator), |
| ass)](api/languages/cpp_api.html# |     [\[3\]](api/langu             |
| _CPPv4N5cudaq16operator_handlerE) | ages/cpp_api.html#_CPPv4N5cudaq15 |
| -   [cudaq::optimizable_function  | scalar_operator15scalar_operatorE |
|     (C++                          | RR15scalar_callbackRRNSt13unorder |
|     class)                        | ed_mapINSt6stringENSt6stringEEE), |
| ](api/languages/cpp_api.html#_CPP |     [\[4\                         |
| v4N5cudaq20optimizable_functionE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::optimization_result   | Pv4N5cudaq15scalar_operator15scal |
|     (C++                          | ar_operatorERR15scalar_operator), |
|     type                          |     [\[5\]](api/language          |
| )](api/languages/cpp_api.html#_CP | s/cpp_api.html#_CPPv4N5cudaq15sca |
| Pv4N5cudaq19optimization_resultE) | lar_operator15scalar_operatorEd), |
| -   [cudaq::optimizer (C++        |     [\[6\]](api/languag           |
|     class)](api/languages/cpp_a   | es/cpp_api.html#_CPPv4N5cudaq15sc |
| pi.html#_CPPv4N5cudaq9optimizerE) | alar_operator15scalar_operatorEv) |
| -   [cudaq::optimizer::optimize   | -   [                             |
|     (C++                          | cudaq::scalar_operator::to_matrix |
|                                   |     (C++                          |
|  function)](api/languages/cpp_api |                                   |
| .html#_CPPv4N5cudaq9optimizer8opt |   function)](api/languages/cpp_ap |
| imizeEKiRR20optimizable_function) | i.html#_CPPv4NK5cudaq15scalar_ope |
| -   [cu                           | rator9to_matrixERKNSt13unordered_ |
| daq::optimizer::requiresGradients | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [                             |
|     function)](api/la             | cudaq::scalar_operator::to_string |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9optimizer17requiresGradientsEv) |     function)](api/l              |
| -   [cudaq::orca (C++             | anguages/cpp_api.html#_CPPv4NK5cu |
|     type)](api/languages/         | daq15scalar_operator9to_stringEv) |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | -   [cudaq::s                     |
| -   [cudaq::orca::sample (C++     | calar_operator::\~scalar_operator |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     functio                       |
| mpleERNSt6vectorINSt6size_tEEERNS | n)](api/languages/cpp_api.html#_C |
| t6vectorINSt6size_tEEERNSt6vector | PPv4N5cudaq15scalar_operatorD0Ev) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [cudaq::set_noise (C++        |
|     [\[1\]]                       |     function)](api/langu          |
| (api/languages/cpp_api.html#_CPPv | ages/cpp_api.html#_CPPv4N5cudaq9s |
| 4N5cudaq4orca6sampleERNSt6vectorI | et_noiseERKN5cudaq11noise_modelE) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [cudaq::set_random_seed (C++  |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     function)](api/               |
| -   [cudaq::orca::sample_async    | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq15set_random_seedENSt6size_tE) |
|                                   | -   [cudaq::simulation_precision  |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq4orca12sample_a |     enum)                         |
| syncERNSt6vectorINSt6size_tEEERNS | ](api/languages/cpp_api.html#_CPP |
| t6vectorINSt6size_tEEERNSt6vector | v4N5cudaq20simulation_precisionE) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [                             |
|     [\[1\]](api/la                | cudaq::simulation_precision::fp32 |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q4orca12sample_asyncERNSt6vectorI |     enumerator)](api              |
| NSt6size_tEEERNSt6vectorINSt6size | /languages/cpp_api.html#_CPPv4N5c |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | udaq20simulation_precision4fp32E) |
| -   [cudaq::OrcaRemoteRESTQPU     | -   [                             |
|     (C++                          | cudaq::simulation_precision::fp64 |
|     cla                           |     (C++                          |
| ss)](api/languages/cpp_api.html#_ |     enumerator)](api              |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::other_policies (C++   | udaq20simulation_precision4fp64E) |
|     s                             | -   [cudaq::SimulationState (C++  |
| truct)](api/languages/cpp_api.htm |     c                             |
| l#_CPPv4N5cudaq14other_policiesE) | lass)](api/languages/cpp_api.html |
| -   [cudaq::PasqalRemoteRESTQPU   | #_CPPv4N5cudaq15SimulationStateE) |
|     (C++                          | -   [                             |
|     class                         | cudaq::SimulationState::precision |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq19PasqalRemoteRESTQPUE) |     enum)](api                    |
| -   [cudaq::pauli1 (C++           | /languages/cpp_api.html#_CPPv4N5c |
|     class)](api/languages/cp      | udaq15SimulationState9precisionE) |
| p_api.html#_CPPv4N5cudaq6pauli1E) | -   [cudaq:                       |
| -                                 | :SimulationState::precision::fp32 |
|    [cudaq::pauli1::num_parameters |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     member)]                      | uages/cpp_api.html#_CPPv4N5cudaq1 |
| (api/languages/cpp_api.html#_CPPv | 5SimulationState9precision4fp32E) |
| 4N5cudaq6pauli114num_parametersE) | -   [cudaq:                       |
| -   [cudaq::pauli1::num_targets   | :SimulationState::precision::fp64 |
|     (C++                          |     (C++                          |
|     membe                         |     enumerator)](api/lang         |
| r)](api/languages/cpp_api.html#_C | uages/cpp_api.html#_CPPv4N5cudaq1 |
| PPv4N5cudaq6pauli111num_targetsE) | 5SimulationState9precision4fp64E) |
| -   [cudaq::pauli1::pauli1 (C++   | -                                 |
|     function)](api/languages/cpp_ |   [cudaq::SimulationState::Tensor |
| api.html#_CPPv4N5cudaq6pauli16pau |     (C++                          |
| li1ERKNSt6vectorIN5cudaq4realEEE) |     struct)](                     |
| -   [cudaq::pauli2 (C++           | api/languages/cpp_api.html#_CPPv4 |
|     class)](api/languages/cp      | N5cudaq15SimulationState6TensorE) |
| p_api.html#_CPPv4N5cudaq6pauli2E) | -   [cudaq::spin_handler (C++     |
| -                                 |                                   |
|    [cudaq::pauli2::num_parameters |   class)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq12spin_handlerE) |
|     member)]                      | -   [cudaq:                       |
| (api/languages/cpp_api.html#_CPPv | :spin_handler::to_diagonal_matrix |
| 4N5cudaq6pauli214num_parametersE) |     (C++                          |
| -   [cudaq::pauli2::num_targets   |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4NK5cud |
|     membe                         | aq12spin_handler18to_diagonal_mat |
| r)](api/languages/cpp_api.html#_C | rixERNSt13unordered_mapINSt6size_ |
| PPv4N5cudaq6pauli211num_targetsE) | tENSt7int64_tEEERKNSt13unordered_ |
| -   [cudaq::pauli2::pauli2 (C++   | mapINSt6stringENSt7complexIdEEEE) |
|     function)](api/languages/cpp_ | -                                 |
| api.html#_CPPv4N5cudaq6pauli26pau |   [cudaq::spin_handler::to_matrix |
| li2ERKNSt6vectorIN5cudaq4realEEE) |     (C++                          |
| -   [cudaq::phase_damping (C++    |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|  class)](api/languages/cpp_api.ht | Pv4N5cudaq12spin_handler9to_matri |
| ml#_CPPv4N5cudaq13phase_dampingE) | xERKNSt6stringENSt7complexIdEEb), |
| -   [cud                          |     [\[1                          |
| aq::phase_damping::num_parameters | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4NK5cudaq12spin_handler9to_mat |
|     member)](api/lan              | rixERNSt13unordered_mapINSt6size_ |
| guages/cpp_api.html#_CPPv4N5cudaq | tENSt7int64_tEEERKNSt13unordered_ |
| 13phase_damping14num_parametersE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [                             | -   [cuda                         |
| cudaq::phase_damping::num_targets | q::spin_handler::to_sparse_matrix |
|     (C++                          |     (C++                          |
|     member)](api/                 |     function)](api/               |
| languages/cpp_api.html#_CPPv4N5cu | languages/cpp_api.html#_CPPv4N5cu |
| daq13phase_damping11num_targetsE) | daq12spin_handler16to_sparse_matr |
| -   [cudaq::phase_flip_channel    | ixERKNSt6stringENSt7complexIdEEb) |
|     (C++                          | -                                 |
|     clas                          |   [cudaq::spin_handler::to_string |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq18phase_flip_channelE) |     function)](ap                 |
| -   [cudaq::p                     | i/languages/cpp_api.html#_CPPv4NK |
| hase_flip_channel::num_parameters | 5cudaq12spin_handler9to_stringEb) |
|     (C++                          | -                                 |
|     member)](api/language         |   [cudaq::spin_handler::unique_id |
| s/cpp_api.html#_CPPv4N5cudaq18pha |     (C++                          |
| se_flip_channel14num_parametersE) |     function)](ap                 |
| -   [cudaq                        | i/languages/cpp_api.html#_CPPv4NK |
| ::phase_flip_channel::num_targets | 5cudaq12spin_handler9unique_idEv) |
|     (C++                          | -   [cudaq::spin_op (C++          |
|     member)](api/langu            |     type)](api/languages/cpp      |
| ages/cpp_api.html#_CPPv4N5cudaq18 | _api.html#_CPPv4N5cudaq7spin_opE) |
| phase_flip_channel11num_targetsE) | -   [cudaq::spin_op_term (C++     |
| -   [cudaq::product_op (C++       |                                   |
|                                   |    type)](api/languages/cpp_api.h |
|  class)](api/languages/cpp_api.ht | tml#_CPPv4N5cudaq12spin_op_termE) |
| ml#_CPPv4I0EN5cudaq10product_opE) | -   [cudaq::state (C++            |
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
|     attribute)](api/l             | -   [resources                    |
| anguages/python_api.html#cudaq.op |     (cudaq.EstimateResult         |
| erators.spin.SpinOperator.random) |     proper                        |
| -   [rank() (in module            | ty)](api/languages/python_api.htm |
|     cudaq.mpi)](api/language      | l#cudaq.EstimateResult.resources) |
| s/python_api.html#cudaq.mpi.rank) | -   [right_multiply               |
| -   [register_names               |     (cudaq.SuperOperator          |
|     (cudaq.SampleResult           |     attribute)]                   |
|     property)                     | (api/languages/python_api.html#cu |
| ](api/languages/python_api.html#c | daq.SuperOperator.right_multiply) |
| udaq.SampleResult.register_names) | -   [row_count                    |
| -                                 |     (cudaq.KrausOperator          |
|   [register_set_target_callback() |     prope                         |
|     (in module                    | rty)](api/languages/python_api.ht |
|     cudaq)]                       | ml#cudaq.KrausOperator.row_count) |
| (api/languages/python_api.html#cu | -   [run() (in module             |
| daq.register_set_target_callback) |     cudaq)](api/lan               |
| -   [reset_target() (in module    | guages/python_api.html#cudaq.run) |
|     cudaq)](api/languages/py      | -   [run_async() (in module       |
| thon_api.html#cudaq.reset_target) |     cudaq)](api/languages         |
| -   [resolve_captured_arguments() | /python_api.html#cudaq.run_async) |
|     (cudaq.PyKernelDecorator      | -   [RydbergHamiltonian (class in |
|     method)](api/languages/p      |     cudaq.operators)]             |
| ython_api.html#cudaq.PyKernelDeco | (api/languages/python_api.html#cu |
| rator.resolve_captured_arguments) | daq.operators.RydbergHamiltonian) |
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
| -   [to_matrix                    | daq.operators.boson.BosonOperator |
|     (cu                           |     attribute)](api/l             |
| daq.operators.boson.BosonOperator | anguages/python_api.html#cudaq.op |
|     attribute)](api/langua        | erators.boson.BosonOperator.trim) |
| ges/python_api.html#cudaq.operato |     -   [(cudaq.                  |
| rs.boson.BosonOperator.to_matrix) | operators.fermion.FermionOperator |
|     -   [(cudaq.ope               |         attribute)](api/langu     |
| rators.boson.BosonOperatorElement | ages/python_api.html#cudaq.operat |
|                                   | ors.fermion.FermionOperator.trim) |
|     attribute)](api/languages/pyt |     -                             |
| hon_api.html#cudaq.operators.boso |  [(cudaq.operators.MatrixOperator |
| n.BosonOperatorElement.to_matrix) |         attribute)](              |
|     -   [(cudaq.                  | api/languages/python_api.html#cud |
| operators.boson.BosonOperatorTerm | aq.operators.MatrixOperator.trim) |
|                                   |     -   [(                        |
|        attribute)](api/languages/ | cudaq.operators.spin.SpinOperator |
| python_api.html#cudaq.operators.b |         attribute)](api           |
| oson.BosonOperatorTerm.to_matrix) | /languages/python_api.html#cudaq. |
|     -   [(cudaq.                  | operators.spin.SpinOperator.trim) |
| operators.fermion.FermionOperator | -   [type                         |
|                                   |     (c                            |
|        attribute)](api/languages/ | udaq.ptsbe.ShotAllocationStrategy |
| python_api.html#cudaq.operators.f |     property)](api/               |
| ermion.FermionOperator.to_matrix) | languages/python_api.html#cudaq.p |
|     -   [(cudaq.operato           | tsbe.ShotAllocationStrategy.type) |
| rs.fermion.FermionOperatorElement |     -                             |
|                                   |    [(cudaq.ptsbe.TraceInstruction |
| attribute)](api/languages/python_ |         property)                 |
| api.html#cudaq.operators.fermion. | ](api/languages/python_api.html#c |
| FermionOperatorElement.to_matrix) | udaq.ptsbe.TraceInstruction.type) |
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
