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

-   [CUDA-Q Logical](./preview/logical/index.html#http://){.reference
    .external}
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
| -   [canonicalize                 | -   [cudaq::phase_flip_channel    |
|     (cu                           |     (C++                          |
| daq.operators.boson.BosonOperator |     clas                          |
|     attribute)](api/languages     | s)](api/languages/cpp_api.html#_C |
| /python_api.html#cudaq.operators. | PPv4N5cudaq18phase_flip_channelE) |
| boson.BosonOperator.canonicalize) | -   [cudaq::p                     |
|     -   [(cudaq.                  | hase_flip_channel::num_parameters |
| operators.boson.BosonOperatorTerm |     (C++                          |
|                                   |     member)](api/language         |
|     attribute)](api/languages/pyt | s/cpp_api.html#_CPPv4N5cudaq18pha |
| hon_api.html#cudaq.operators.boso | se_flip_channel14num_parametersE) |
| n.BosonOperatorTerm.canonicalize) | -   [cudaq                        |
|     -   [(cudaq.                  | ::phase_flip_channel::num_targets |
| operators.fermion.FermionOperator |     (C++                          |
|                                   |     member)](api/langu            |
|     attribute)](api/languages/pyt | ages/cpp_api.html#_CPPv4N5cudaq18 |
| hon_api.html#cudaq.operators.ferm | phase_flip_channel11num_targetsE) |
| ion.FermionOperator.canonicalize) | -   [cudaq::product_op (C++       |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |  class)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4I0EN5cudaq10product_opE) |
| attribute)](api/languages/python_ | -   [cudaq::product_op::begin     |
| api.html#cudaq.operators.fermion. |     (C++                          |
| FermionOperatorTerm.canonicalize) |     functio                       |
|     -                             | n)](api/languages/cpp_api.html#_C |
|  [(cudaq.operators.MatrixOperator | PPv4NK5cudaq10product_op5beginEv) |
|         attribute)](api/lang      | -                                 |
| uages/python_api.html#cudaq.opera |  [cudaq::product_op::canonicalize |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     func                          |
| udaq.operators.MatrixOperatorTerm | tion)](api/languages/cpp_api.html |
|         attribute)](api/language  | #_CPPv4N5cudaq10product_op12canon |
| s/python_api.html#cudaq.operators | icalizeERKNSt3setINSt6size_tEEE), |
| .MatrixOperatorTerm.canonicalize) |     [\[1\]](api                   |
|     -   [(                        | /languages/cpp_api.html#_CPPv4N5c |
| cudaq.operators.spin.SpinOperator | udaq10product_op12canonicalizeEv) |
|         attribute)](api/languag   | -   [                             |
| es/python_api.html#cudaq.operator | cudaq::product_op::const_iterator |
| s.spin.SpinOperator.canonicalize) |     (C++                          |
|     -   [(cuda                    |     struct)](api/                 |
| q.operators.spin.SpinOperatorTerm | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq10product_op14const_iteratorE) |
|       attribute)](api/languages/p | -   [cudaq::product_o             |
| ython_api.html#cudaq.operators.sp | p::const_iterator::const_iterator |
| in.SpinOperatorTerm.canonicalize) |     (C++                          |
| -   [captured_variables()         |     fu                            |
|     (cudaq.PyKernelDecorator      | nction)](api/languages/cpp_api.ht |
|     method)](api/lan              | ml#_CPPv4N5cudaq10product_op14con |
| guages/python_api.html#cudaq.PyKe | st_iterator14const_iteratorEPK10p |
| rnelDecorator.captured_variables) | roduct_opI9HandlerTyENSt6size_tE) |
| -   [CentralDifference (class in  | -   [cudaq::produ                 |
|     cudaq.gradients)              | ct_op::const_iterator::operator!= |
| ](api/languages/python_api.html#c |     (C++                          |
| udaq.gradients.CentralDifference) |     fun                           |
| -   [channel                      | ction)](api/languages/cpp_api.htm |
|     (cudaq.ptsbe.TraceInstruction | l#_CPPv4NK5cudaq10product_op14con |
|     property)](a                  | st_iteratorneERK14const_iterator) |
| pi/languages/python_api.html#cuda | -   [cudaq::produ                 |
| q.ptsbe.TraceInstruction.channel) | ct_op::const_iterator::operator\* |
| -   [circuit_location             |     (C++                          |
|     (cudaq.ptsbe.KrausSelection   |     function)](api/lang           |
|     property)](api/lang           | uages/cpp_api.html#_CPPv4NK5cudaq |
| uages/python_api.html#cudaq.ptsbe | 10product_op14const_iteratormlEv) |
| .KrausSelection.circuit_location) | -   [cudaq::produ                 |
| -   [clear (cudaq.Resources       | ct_op::const_iterator::operator++ |
|                                   |     (C++                          |
|   attribute)](api/languages/pytho |     function)](api/lang           |
| n_api.html#cudaq.Resources.clear) | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     -   [(cudaq.SampleResult      | 0product_op14const_iteratorppEi), |
|         a                         |     [\[1\]](api/lan               |
| ttribute)](api/languages/python_a | guages/cpp_api.html#_CPPv4N5cudaq |
| pi.html#cudaq.SampleResult.clear) | 10product_op14const_iteratorppEv) |
| -   [CliffordTSequence (class in  | -   [cudaq::produc                |
|     cudaq.sy                      | t_op::const_iterator::operator\-- |
| nth)](api/languages/python_api.ht |     (C++                          |
| ml#cudaq.synth.CliffordTSequence) |     function)](api/lang           |
| -   [COBYLA (class in             | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     cudaq.o                       | 0product_op14const_iteratormmEi), |
| ptimizers)](api/languages/python_ |     [\[1\]](api/lan               |
| api.html#cudaq.optimizers.COBYLA) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [coefficient                  | 10product_op14const_iteratormmEv) |
|     (cudaq.                       | -   [cudaq::produc                |
| operators.boson.BosonOperatorTerm | t_op::const_iterator::operator-\> |
|     property)](api/languages/py   |     (C++                          |
| thon_api.html#cudaq.operators.bos |     function)](api/lan            |
| on.BosonOperatorTerm.coefficient) | guages/cpp_api.html#_CPPv4N5cudaq |
|     -   [(cudaq.oper              | 10product_op14const_iteratorptEv) |
| ators.fermion.FermionOperatorTerm | -   [cudaq::produ                 |
|                                   | ct_op::const_iterator::operator== |
|   property)](api/languages/python |     (C++                          |
| _api.html#cudaq.operators.fermion |     fun                           |
| .FermionOperatorTerm.coefficient) | ction)](api/languages/cpp_api.htm |
|     -   [(c                       | l#_CPPv4NK5cudaq10product_op14con |
| udaq.operators.MatrixOperatorTerm | st_iteratoreqERK14const_iterator) |
|         property)](api/languag    | -   [cudaq::product_op::degrees   |
| es/python_api.html#cudaq.operator |     (C++                          |
| s.MatrixOperatorTerm.coefficient) |     function)                     |
|     -   [(cuda                    | ](api/languages/cpp_api.html#_CPP |
| q.operators.spin.SpinOperatorTerm | v4NK5cudaq10product_op7degreesEv) |
|         property)](api/languages/ | -   [cudaq::product_op::dump (C++ |
| python_api.html#cudaq.operators.s |     functi                        |
| pin.SpinOperatorTerm.coefficient) | on)](api/languages/cpp_api.html#_ |
| -   [col_count                    | CPPv4NK5cudaq10product_op4dumpEv) |
|     (cudaq.KrausOperator          | -   [cudaq::product_op::end (C++  |
|     prope                         |     funct                         |
| rty)](api/languages/python_api.ht | ion)](api/languages/cpp_api.html# |
| ml#cudaq.KrausOperator.col_count) | _CPPv4NK5cudaq10product_op3endEv) |
| -   [compile()                    | -   [c                            |
|     (cudaq.PyKernelDecorator      | udaq::product_op::get_coefficient |
|     metho                         |     (C++                          |
| d)](api/languages/python_api.html |     function)](api/lan            |
| #cudaq.PyKernelDecorator.compile) | guages/cpp_api.html#_CPPv4NK5cuda |
| -   [compiledModuleCache()        | q10product_op15get_coefficientEv) |
|     (cudaq.PyKernelDecorator      | -                                 |
|     method)](api/lang             |   [cudaq::product_op::get_term_id |
| uages/python_api.html#cudaq.PyKer |     (C++                          |
| nelDecorator.compiledModuleCache) |     function)](api                |
| -   [ComplexMatrix (class in      | /languages/cpp_api.html#_CPPv4NK5 |
|     cudaq)](api/languages/pyt     | cudaq10product_op11get_term_idEv) |
| hon_api.html#cudaq.ComplexMatrix) | -                                 |
| -   [compute                      |   [cudaq::product_op::is_identity |
|     (                             |     (C++                          |
| cudaq.gradients.CentralDifference |     function)](api                |
|     attribute)](api/la            | /languages/cpp_api.html#_CPPv4NK5 |
| nguages/python_api.html#cudaq.gra | cudaq10product_op11is_identityEv) |
| dients.CentralDifference.compute) | -   [cudaq::product_op::num_ops   |
|     -   [(                        |     (C++                          |
| cudaq.gradients.ForwardDifference |     function)                     |
|         attribute)](api/la        | ](api/languages/cpp_api.html#_CPP |
| nguages/python_api.html#cudaq.gra | v4NK5cudaq10product_op7num_opsEv) |
| dients.ForwardDifference.compute) | -                                 |
|     -                             |    [cudaq::product_op::operator\* |
|  [(cudaq.gradients.ParameterShift |     (C++                          |
|         attribute)](api           |     function)](api/languages/     |
| /languages/python_api.html#cudaq. | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| gradients.ParameterShift.compute) | oduct_opmlE10product_opI1TERK15sc |
| -   [const()                      | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]](api/languages/        |
|   (cudaq.operators.ScalarOperator | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     class                         | oduct_opmlE10product_opI1TERK15sc |
|     method)](a                    | alar_operatorRR10product_opI1TE), |
| pi/languages/python_api.html#cuda |     [\[2\]](api/languages/        |
| q.operators.ScalarOperator.const) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [controls                     | oduct_opmlE10product_opI1TERR15sc |
|     (cudaq.ptsbe.TraceInstruction | alar_operatorRK10product_opI1TE), |
|     property)](ap                 |     [\[3\]](api/languages/        |
| i/languages/python_api.html#cudaq | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| .ptsbe.TraceInstruction.controls) | oduct_opmlE10product_opI1TERR15sc |
| -   [copy                         | alar_operatorRR10product_opI1TE), |
|     (cu                           |     [\[4\]](api/                  |
| daq.operators.boson.BosonOperator | languages/cpp_api.html#_CPPv4I0EN |
|     attribute)](api/l             | 5cudaq10product_opmlE6sum_opI1TER |
| anguages/python_api.html#cudaq.op | K15scalar_operatorRK6sum_opI1TE), |
| erators.boson.BosonOperator.copy) |     [\[5\]](api/                  |
|     -   [(cudaq.                  | languages/cpp_api.html#_CPPv4I0EN |
| operators.boson.BosonOperatorTerm | 5cudaq10product_opmlE6sum_opI1TER |
|         attribute)](api/langu     | K15scalar_operatorRR6sum_opI1TE), |
| ages/python_api.html#cudaq.operat |     [\[6\]](api/                  |
| ors.boson.BosonOperatorTerm.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cudaq.                  | 5cudaq10product_opmlE6sum_opI1TER |
| operators.fermion.FermionOperator | R15scalar_operatorRK6sum_opI1TE), |
|         attribute)](api/langu     |     [\[7\]](api/                  |
| ages/python_api.html#cudaq.operat | languages/cpp_api.html#_CPPv4I0EN |
| ors.fermion.FermionOperator.copy) | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [(cudaq.oper              | R15scalar_operatorRR6sum_opI1TE), |
| ators.fermion.FermionOperatorTerm |     [\[8\]](api/languages         |
|         attribute)](api/languages | /cpp_api.html#_CPPv4NK5cudaq10pro |
| /python_api.html#cudaq.operators. | duct_opmlERK6sum_opI9HandlerTyE), |
| fermion.FermionOperatorTerm.copy) |     [\[9\]](api/languages/cpp_a   |
|     -                             | pi.html#_CPPv4NKR5cudaq10product_ |
|  [(cudaq.operators.MatrixOperator | opmlERK10product_opI9HandlerTyE), |
|         attribute)](              |     [\[10\]](api/language         |
| api/languages/python_api.html#cud | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| aq.operators.MatrixOperator.copy) | roduct_opmlERK15scalar_operator), |
|     -   [(c                       |     [\[11\]](api/languages/cpp_a  |
| udaq.operators.MatrixOperatorTerm | pi.html#_CPPv4NKR5cudaq10product_ |
|         attribute)](api/          | opmlERR10product_opI9HandlerTyE), |
| languages/python_api.html#cudaq.o |     [\[12\]](api/language         |
| perators.MatrixOperatorTerm.copy) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [(                        | roduct_opmlERR15scalar_operator), |
| cudaq.operators.spin.SpinOperator |     [\[13\]](api/languages/cpp_   |
|         attribute)](api           | api.html#_CPPv4NO5cudaq10product_ |
| /languages/python_api.html#cudaq. | opmlERK10product_opI9HandlerTyE), |
| operators.spin.SpinOperator.copy) |     [\[14\]](api/languag          |
|     -   [(cuda                    | es/cpp_api.html#_CPPv4NO5cudaq10p |
| q.operators.spin.SpinOperatorTerm | roduct_opmlERK15scalar_operator), |
|         attribute)](api/lan       |     [\[15\]](api/languages/cpp_   |
| guages/python_api.html#cudaq.oper | api.html#_CPPv4NO5cudaq10product_ |
| ators.spin.SpinOperatorTerm.copy) | opmlERR10product_opI9HandlerTyE), |
| -   [count (cudaq.Resources       |     [\[16\]](api/langua           |
|                                   | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|   attribute)](api/languages/pytho | product_opmlERR15scalar_operator) |
| n_api.html#cudaq.Resources.count) | -                                 |
|     -   [(cudaq.SampleResult      |   [cudaq::product_op::operator\*= |
|         a                         |     (C++                          |
| ttribute)](api/languages/python_a |     function)](api/languages/cpp  |
| pi.html#cudaq.SampleResult.count) | _api.html#_CPPv4N5cudaq10product_ |
| -   [count_controls               | opmLERK10product_opI9HandlerTyE), |
|     (cudaq.Resources              |     [\[1\]](api/langua            |
|     attribu                       | ges/cpp_api.html#_CPPv4N5cudaq10p |
| te)](api/languages/python_api.htm | roduct_opmLERK15scalar_operator), |
| l#cudaq.Resources.count_controls) |     [\[2\]](api/languages/cp      |
| -   [count_instructions           | p_api.html#_CPPv4N5cudaq10product |
|                                   | _opmLERR10product_opI9HandlerTyE) |
|   (cudaq.ptsbe.PTSBEExecutionData | -   [cudaq::product_op::operator+ |
|     attribute)](api/languages/    |     (C++                          |
| python_api.html#cudaq.ptsbe.PTSBE |     function)](api/langu          |
| ExecutionData.count_instructions) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [counts (cudaq.ObserveResult  | q10product_opplE6sum_opI1TERK15sc |
|     att                           | alar_operatorRK10product_opI1TE), |
| ribute)](api/languages/python_api |     [\[1\]](api/                  |
| .html#cudaq.ObserveResult.counts) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cudaq.SampleResult      | 5cudaq10product_opplE6sum_opI1TER |
|         p                         | K15scalar_operatorRK6sum_opI1TE), |
| roperty)](api/languages/python_ap |     [\[2\]](api/langu             |
| i.html#cudaq.SampleResult.counts) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [csr_spmatrix (C++            | q10product_opplE6sum_opI1TERK15sc |
|     type)](api/languages/c        | alar_operatorRR10product_opI1TE), |
| pp_api.html#_CPPv412csr_spmatrix) |     [\[3\]](api/                  |
| -   cudaq                         | languages/cpp_api.html#_CPPv4I0EN |
|     -   [module](api/langua       | 5cudaq10product_opplE6sum_opI1TER |
| ges/python_api.html#module-cudaq) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq (C++                   |     [\[4\]](api/langu             |
|     type)](api/lan                | ages/cpp_api.html#_CPPv4I0EN5cuda |
| guages/cpp_api.html#_CPPv45cudaq) | q10product_opplE6sum_opI1TERR15sc |
| -   [cudaq.apply_noise() (in      | alar_operatorRK10product_opI1TE), |
|     module                        |     [\[5\]](api/                  |
|     cudaq)](api/languages/python_ | languages/cpp_api.html#_CPPv4I0EN |
| api.html#cudaq.cudaq.apply_noise) | 5cudaq10product_opplE6sum_opI1TER |
| -   cudaq.boson                   | R15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/languages/py |     [\[6\]](api/langu             |
| thon_api.html#module-cudaq.boson) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq.fermion                 | q10product_opplE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|   -   [module](api/languages/pyth |     [\[7\]](api/                  |
| on_api.html#module-cudaq.fermion) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.operators.custom        | 5cudaq10product_opplE6sum_opI1TER |
|     -   [mo                       | R15scalar_operatorRR6sum_opI1TE), |
| dule](api/languages/python_api.ht |     [\[8\]](api/languages/cpp_a   |
| ml#module-cudaq.operators.custom) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.spin                    | opplERK10product_opI9HandlerTyE), |
|     -   [module](api/languages/p  |     [\[9\]](api/language          |
| ython_api.html#module-cudaq.spin) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::amplitude_damping     | roduct_opplERK15scalar_operator), |
|     (C++                          |     [\[10\]](api/languages/       |
|     cla                           | cpp_api.html#_CPPv4NKR5cudaq10pro |
| ss)](api/languages/cpp_api.html#_ | duct_opplERK6sum_opI9HandlerTyE), |
| CPPv4N5cudaq17amplitude_dampingE) |     [\[11\]](api/languages/cpp_a  |
| -                                 | pi.html#_CPPv4NKR5cudaq10product_ |
| [cudaq::amplitude_damping_channel | opplERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[12\]](api/language         |
|     class)](api                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /languages/cpp_api.html#_CPPv4N5c | roduct_opplERR15scalar_operator), |
| udaq25amplitude_damping_channelE) |     [\[13\]](api/languages/       |
| -   [cudaq::amplitud              | cpp_api.html#_CPPv4NKR5cudaq10pro |
| e_damping_channel::num_parameters | duct_opplERR6sum_opI9HandlerTyE), |
|     (C++                          |     [\[                           |
|     member)](api/languages/cpp_a  | 14\]](api/languages/cpp_api.html# |
| pi.html#_CPPv4N5cudaq25amplitude_ | _CPPv4NKR5cudaq10product_opplEv), |
| damping_channel14num_parametersE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::ampli                 | api.html#_CPPv4NO5cudaq10product_ |
| tude_damping_channel::num_targets | opplERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[16\]](api/languag          |
|     member)](api/languages/cp     | es/cpp_api.html#_CPPv4NO5cudaq10p |
| p_api.html#_CPPv4N5cudaq25amplitu | roduct_opplERK15scalar_operator), |
| de_damping_channel11num_targetsE) |     [\[17\]](api/languages        |
| -   [cudaq::AnalogRemoteRESTQPU   | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opplERK6sum_opI9HandlerTyE), |
|     class                         |     [\[18\]](api/languages/cpp_   |
| )](api/languages/cpp_api.html#_CP | api.html#_CPPv4NO5cudaq10product_ |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::apply_noise (C++      |     [\[19\]](api/languag          |
|     function)](api/               | es/cpp_api.html#_CPPv4NO5cudaq10p |
| languages/cpp_api.html#_CPPv4I0Dp | roduct_opplERR15scalar_operator), |
| EN5cudaq11apply_noiseEvDpRR4Args) |     [\[20\]](api/languages        |
| -   [cudaq::async_result (C++     | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     c                             | duct_opplERR6sum_opI9HandlerTyE), |
| lass)](api/languages/cpp_api.html |     [                             |
| #_CPPv4I0EN5cudaq12async_resultE) | \[21\]](api/languages/cpp_api.htm |
| -   [cudaq::async_result::get     | l#_CPPv4NO5cudaq10product_opplEv) |
|     (C++                          | -   [cudaq::product_op::operator- |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)](api/langu          |
| CPPv4N5cudaq12async_result3getEv) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::async_sample_result   | q10product_opmiE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRK10product_opI1TE), |
|     type                          |     [\[1\]](api/                  |
| )](api/languages/cpp_api.html#_CP | languages/cpp_api.html#_CPPv4I0EN |
| Pv4N5cudaq19async_sample_resultE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::BaseRemoteRESTQPU     | K15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[2\]](api/langu             |
|     cla                           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ss)](api/languages/cpp_api.html#_ | q10product_opmiE6sum_opI1TERK15sc |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | alar_operatorRR10product_opI1TE), |
| -   [cudaq::bit_flip_channel (C++ |     [\[3\]](api/                  |
|     cl                            | languages/cpp_api.html#_CPPv4I0EN |
| ass)](api/languages/cpp_api.html# | 5cudaq10product_opmiE6sum_opI1TER |
| _CPPv4N5cudaq16bit_flip_channelE) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq:                       |     [\[4\]](api/langu             |
| :bit_flip_channel::num_parameters | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERR15sc |
|     member)](api/langua           | alar_operatorRK10product_opI1TE), |
| ges/cpp_api.html#_CPPv4N5cudaq16b |     [\[5\]](api/                  |
| it_flip_channel14num_parametersE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cud                          | 5cudaq10product_opmiE6sum_opI1TER |
| aq::bit_flip_channel::num_targets | R15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[6\]](api/langu             |
|     member)](api/lan              | ages/cpp_api.html#_CPPv4I0EN5cuda |
| guages/cpp_api.html#_CPPv4N5cudaq | q10product_opmiE6sum_opI1TERR15sc |
| 16bit_flip_channel11num_targetsE) | alar_operatorRR10product_opI1TE), |
| -   [cudaq::boson_handler (C++    |     [\[7\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|  class)](api/languages/cpp_api.ht | 5cudaq10product_opmiE6sum_opI1TER |
| ml#_CPPv4N5cudaq13boson_handlerE) | R15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq::boson_op (C++         |     [\[8\]](api/languages/cpp_a   |
|     type)](api/languages/cpp_     | pi.html#_CPPv4NKR5cudaq10product_ |
| api.html#_CPPv4N5cudaq8boson_opE) | opmiERK10product_opI9HandlerTyE), |
| -   [cudaq::boson_op_term (C++    |     [\[9\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|   type)](api/languages/cpp_api.ht | roduct_opmiERK15scalar_operator), |
| ml#_CPPv4N5cudaq13boson_op_termE) |     [\[10\]](api/languages/       |
| -   [cudaq::CodeGenConfig (C++    | cpp_api.html#_CPPv4NKR5cudaq10pro |
|                                   | duct_opmiERK6sum_opI9HandlerTyE), |
| struct)](api/languages/cpp_api.ht |     [\[11\]](api/languages/cpp_a  |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [cudaq::commutation_relations | opmiERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[12\]](api/language         |
|     struct)]                      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| (api/languages/cpp_api.html#_CPPv | roduct_opmiERR15scalar_operator), |
| 4N5cudaq21commutation_relationsE) |     [\[13\]](api/languages/       |
| -   [cudaq::complex (C++          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     type)](api/languages/cpp      | duct_opmiERR6sum_opI9HandlerTyE), |
| _api.html#_CPPv4N5cudaq7complexE) |     [\[                           |
| -   [cudaq::complex_matrix (C++   | 14\]](api/languages/cpp_api.html# |
|                                   | _CPPv4NKR5cudaq10product_opmiEv), |
| class)](api/languages/cpp_api.htm |     [\[15\]](api/languages/cpp_   |
| l#_CPPv4N5cudaq14complex_matrixE) | api.html#_CPPv4NO5cudaq10product_ |
| -                                 | opmiERK10product_opI9HandlerTyE), |
|   [cudaq::complex_matrix::adjoint |     [\[16\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     function)](a                  | roduct_opmiERK15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[17\]](api/languages        |
| 5cudaq14complex_matrix7adjointEv) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::                      | duct_opmiERK6sum_opI9HandlerTyE), |
| complex_matrix::diagonal_elements |     [\[18\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     function)](api/languages      | opmiERR10product_opI9HandlerTyE), |
| /cpp_api.html#_CPPv4NK5cudaq14com |     [\[19\]](api/languag          |
| plex_matrix17diagonal_elementsEi) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::complex_matrix::dump  | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[20\]](api/languages        |
|     function)](api/language       | /cpp_api.html#_CPPv4NO5cudaq10pro |
| s/cpp_api.html#_CPPv4NK5cudaq14co | duct_opmiERR6sum_opI9HandlerTyE), |
| mplex_matrix4dumpERNSt7ostreamE), |     [                             |
|     [\[1\]]                       | \[21\]](api/languages/cpp_api.htm |
| (api/languages/cpp_api.html#_CPPv | l#_CPPv4NO5cudaq10product_opmiEv) |
| 4NK5cudaq14complex_matrix4dumpEv) | -   [cudaq::product_op::operator/ |
| -   [c                            |     (C++                          |
| udaq::complex_matrix::eigenvalues |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/lan            | roduct_opdvERK15scalar_operator), |
| guages/cpp_api.html#_CPPv4NK5cuda |     [\[1\]](api/language          |
| q14complex_matrix11eigenvaluesEv) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cu                           | roduct_opdvERR15scalar_operator), |
| daq::complex_matrix::eigenvectors |     [\[2\]](api/languag           |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     function)](api/lang           | roduct_opdvERK15scalar_operator), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[3\]](api/langua            |
| 14complex_matrix12eigenvectorsEv) | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| -   [c                            | product_opdvERR15scalar_operator) |
| udaq::complex_matrix::exponential | -                                 |
|     (C++                          |    [cudaq::product_op::operator/= |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/langu          |
| q14complex_matrix11exponentialEv) | ages/cpp_api.html#_CPPv4N5cudaq10 |
| -                                 | product_opdVERK15scalar_operator) |
|  [cudaq::complex_matrix::identity | -   [cudaq::product_op::operator= |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/l              |
| /cpp_api.html#_CPPv4N5cudaq14comp | anguages/cpp_api.html#_CPPv4I00EN |
| lex_matrix8identityEKNSt6size_tE) | 5cudaq10product_opaSER10product_o |
| -                                 | pI9HandlerTyERK10product_opI1TE), |
| [cudaq::complex_matrix::kronecker |     [\[1\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4N5cudaq10product_ |
|     function)](api/lang           | opaSERK10product_opI9HandlerTyE), |
| uages/cpp_api.html#_CPPv4I00EN5cu |     [\[2\]](api/languages/cp      |
| daq14complex_matrix9kroneckerE14c | p_api.html#_CPPv4N5cudaq10product |
| omplex_matrix8Iterable8Iterable), | _opaSERR10product_opI9HandlerTyE) |
|     [\[1\]](api/l                 | -                                 |
| anguages/cpp_api.html#_CPPv4N5cud |    [cudaq::product_op::operator== |
| aq14complex_matrix9kroneckerERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     function)](api/languages/cpp  |
| -   [cudaq::c                     | _api.html#_CPPv4NK5cudaq10product |
| omplex_matrix::minimal_eigenvalue | _opeqERK10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     function)](api/languages/     |  [cudaq::product_op::operator\[\] |
| cpp_api.html#_CPPv4NK5cudaq14comp |     (C++                          |
| lex_matrix18minimal_eigenvalueEv) |     function)](ap                 |
| -   [                             | i/languages/cpp_api.html#_CPPv4NK |
| cudaq::complex_matrix::operator() | 5cudaq10product_opixENSt6size_tE) |
|     (C++                          | -                                 |
|     function)](api/languages/cpp  |    [cudaq::product_op::product_op |
| _api.html#_CPPv4N5cudaq14complex_ |     (C++                          |
| matrixclENSt6size_tENSt6size_tE), |     f                             |
|     [\[1\]](api/languages/cpp     | unction)](api/languages/cpp_api.h |
| _api.html#_CPPv4NK5cudaq14complex | tml#_CPPv4I00EN5cudaq10product_op |
| _matrixclENSt6size_tENSt6size_tE) | 10product_opERK10product_opI1TE), |
| -   [                             |     [\[1\]]                       |
| cudaq::complex_matrix::operator\* | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4I00EN5cudaq10product_op10product |
|     function)](api/langua         | _opERK10product_opI1TERKN14matrix |
| ges/cpp_api.html#_CPPv4N5cudaq14c | _handler20commutation_behaviorE), |
| omplex_matrixmlEN14complex_matrix |                                   |
| 10value_typeERK14complex_matrix), |   [\[2\]](api/languages/cpp_api.h |
|     [\[1\]                        | tml#_CPPv4N5cudaq10product_op10pr |
| ](api/languages/cpp_api.html#_CPP | oduct_opENSt6size_tENSt6size_tE), |
| v4N5cudaq14complex_matrixmlERK14c |     [\[3\]](api/languages/cp      |
| omplex_matrixRK14complex_matrix), | p_api.html#_CPPv4N5cudaq10product |
|                                   | _op10product_opENSt7complexIdEE), |
|  [\[2\]](api/languages/cpp_api.ht |     [\[4\]](api/l                 |
| ml#_CPPv4N5cudaq14complex_matrixm | anguages/cpp_api.html#_CPPv4N5cud |
| lERK14complex_matrixRKNSt6vectorI | aq10product_op10product_opERK10pr |
| N14complex_matrix10value_typeEEE) | oduct_opI9HandlerTyENSt6size_tE), |
| -                                 |     [\[5\]](api/l                 |
| [cudaq::complex_matrix::operator+ | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq10product_op10product_opERR10pr |
|     function                      | oduct_opI9HandlerTyENSt6size_tE), |
| )](api/languages/cpp_api.html#_CP |     [\[6\]](api/languages         |
| Pv4N5cudaq14complex_matrixplERK14 | /cpp_api.html#_CPPv4N5cudaq10prod |
| complex_matrixRK14complex_matrix) | uct_op10product_opERR9HandlerTy), |
| -                                 |     [\[7\]](ap                    |
| [cudaq::complex_matrix::operator- | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq10product_op10product_opEd), |
|     function                      |     [\[8\]](a                     |
| )](api/languages/cpp_api.html#_CP | pi/languages/cpp_api.html#_CPPv4N |
| Pv4N5cudaq14complex_matrixmiERK14 | 5cudaq10product_op10product_opEv) |
| complex_matrixRK14complex_matrix) | -   [cuda                         |
| -   [cu                           | q::product_op::to_diagonal_matrix |
| daq::complex_matrix::operator\[\] |     (C++                          |
|     (C++                          |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4NK5c |
|  function)](api/languages/cpp_api | udaq10product_op18to_diagonal_mat |
| .html#_CPPv4N5cudaq14complex_matr | rixENSt13unordered_mapINSt6size_t |
| ixixERKNSt6vectorINSt6size_tEEE), | ENSt7int64_tEEERKNSt13unordered_m |
|     [\[1\]](api/languages/cpp_api | apINSt6stringENSt7complexIdEEEEb) |
| .html#_CPPv4NK5cudaq14complex_mat | -   [cudaq::product_op::to_matrix |
| rixixERKNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq::complex_matrix::power |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)]                    | _CPPv4NK5cudaq10product_op9to_mat |
| (api/languages/cpp_api.html#_CPPv | rixENSt13unordered_mapINSt6size_t |
| 4N5cudaq14complex_matrix5powerEi) | ENSt7int64_tEEERKNSt13unordered_m |
| -                                 | apINSt6stringENSt7complexIdEEEEb) |
|  [cudaq::complex_matrix::set_zero | -   [cu                           |
|     (C++                          | daq::product_op::to_sparse_matrix |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](ap                 |
| cudaq14complex_matrix8set_zeroEv) | i/languages/cpp_api.html#_CPPv4NK |
| -                                 | 5cudaq10product_op16to_sparse_mat |
| [cudaq::complex_matrix::to_string | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/               | apINSt6stringENSt7complexIdEEEEb) |
| languages/cpp_api.html#_CPPv4NK5c | -   [cudaq::product_op::to_string |
| udaq14complex_matrix9to_stringEv) |     (C++                          |
| -   [                             |     function)](                   |
| cudaq::complex_matrix::value_type | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq10product_op9to_stringEv) |
|     type)](api/                   | -                                 |
| languages/cpp_api.html#_CPPv4N5cu |  [cudaq::product_op::\~product_op |
| daq14complex_matrix10value_typeE) |     (C++                          |
| -   [cudaq::contrib (C++          |     fu                            |
|     type)](api/languages/cpp      | nction)](api/languages/cpp_api.ht |
| _api.html#_CPPv4N5cudaq7contribE) | ml#_CPPv4N5cudaq10product_opD0Ev) |
| -                                 | -   [cudaq::ptsbe (C++            |
| [cudaq::contrib::amplitude_encode |     type)](api/languages/c        |
|     (C++                          | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
|     function)](api/language       | -   [cudaq::p                     |
| s/cpp_api.html#_CPPv4N5cudaq7cont | tsbe::ConditionalSamplingStrategy |
| rib16amplitude_encodeENSt4spanIKN |     (C++                          |
| St7complexIdEEEENSt7complexIdEE), |     class)](api/languag           |
|     [\[1\]](api/language          | es/cpp_api.html#_CPPv4N5cudaq5pts |
| s/cpp_api.html#_CPPv4N5cudaq7cont | be27ConditionalSamplingStrategyE) |
| rib16amplitude_encodeENSt4spanIKN | -   [cudaq::ptsbe::C              |
| St7complexIfEEEENSt7complexIdEE), | onditionalSamplingStrategy::clone |
|     [\[2\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4N5cudaq7contrib16amplitude_enco |    function)](api/languages/cpp_a |
| deENSt4spanIKdEENSt7complexIdEE), | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
|     [\[3\]                        | ditionalSamplingStrategy5cloneEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cuda                         |
| v4N5cudaq7contrib16amplitude_enco | q::ptsbe::ConditionalSamplingStra |
| deENSt4spanIKfEENSt7complexIdEE), | tegy::ConditionalSamplingStrategy |
|                                   |     (C++                          |
| [\[4\]](api/languages/cpp_api.htm |     function)](api/lang           |
| l#_CPPv4N5cudaq7contrib16amplitud | uages/cpp_api.html#_CPPv4N5cudaq5 |
| e_encodeERK5stateNSt7complexIdEE) | ptsbe27ConditionalSamplingStrateg |
| -                                 | y27ConditionalSamplingStrategyE19 |
|   [cudaq::contrib::angular_encode | TrajectoryPredicateNSt8uint64_tE) |
|     (C++                          | -                                 |
|                                   |   [cudaq::ptsbe::ConditionalSampl |
|  function)](api/languages/cpp_api | ingStrategy::generateTrajectories |
| .html#_CPPv4I0EN5cudaq7contrib14a |     (C++                          |
| ngular_encodeEvRR6KernelR10QuakeV |     function)](api/language       |
| alueNSt4spanIKdEE12RotationAxis), | s/cpp_api.html#_CPPv4NK5cudaq5pts |
|     [\[1\]](api/languages/cpp_api | be27ConditionalSamplingStrategy20 |
| .html#_CPPv4I0EN5cudaq7contrib14a | generateTrajectoriesENSt4spanIKN6 |
| ngular_encodeEvRR6KernelR10QuakeV | detail10NoisePointEEENSt6size_tE) |
| alueR10QuakeValue12RotationAxis), | -   [cudaq::ptsbe::               |
|                                   | ConditionalSamplingStrategy::name |
|   [\[2\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4I0EN5cudaq7contrib14ang |     function)](api/languages/cpp_ |
| ular_encodeEvRR6KernelR10QuakeVal | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| ueRKNSt6vectorIdEE12RotationAxis) | nditionalSamplingStrategy4nameEv) |
| -   [cudaq::contrib::draw (C++    | -   [cudaq:                       |
|     function)                     | :ptsbe::ConditionalSamplingStrate |
| ](api/languages/cpp_api.html#_CPP | gy::\~ConditionalSamplingStrategy |
| v4I0DpEN5cudaq7contrib4drawENSt6s |     (C++                          |
| tringERR13QuantumKernelDpRR4Args) |     function)](api/languages/     |
| -                                 | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| [cudaq::contrib::get_unitary_cmat | 7ConditionalSamplingStrategyD0Ev) |
|     (C++                          | -                                 |
|     function)](api/languages/cp   | [cudaq::ptsbe::detail::NoisePoint |
| p_api.html#_CPPv4I0DpEN5cudaq7con |     (C++                          |
| trib16get_unitary_cmatE14complex_ |     struct)](a                    |
| matrixRR13QuantumKernelDpRR4Args) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::contrib::RotationAxis | 5cudaq5ptsbe6detail10NoisePointE) |
|     (C++                          | -   [cudaq::p                     |
|     enum)                         | tsbe::detail::NoisePoint::channel |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq7contrib12RotationAxisE) |     member)](api/langu            |
| -                                 | ages/cpp_api.html#_CPPv4N5cudaq5p |
|  [cudaq::contrib::RotationAxis::X | tsbe6detail10NoisePoint7channelE) |
|     (C++                          | -   [cudaq::ptsbe::det            |
|     enumerator)](                 | ail::NoisePoint::circuit_location |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq7contrib12RotationAxis1XE) |     member)](api/languages/cpp_a  |
| -                                 | pi.html#_CPPv4N5cudaq5ptsbe6detai |
|  [cudaq::contrib::RotationAxis::Y | l10NoisePoint16circuit_locationE) |
|     (C++                          | -   [cudaq::p                     |
|     enumerator)](                 | tsbe::detail::NoisePoint::op_name |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq7contrib12RotationAxis1YE) |     member)](api/langu            |
| -                                 | ages/cpp_api.html#_CPPv4N5cudaq5p |
|  [cudaq::contrib::RotationAxis::Z | tsbe6detail10NoisePoint7op_nameE) |
|     (C++                          | -   [cudaq::                      |
|     enumerator)](                 | ptsbe::detail::NoisePoint::qubits |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq7contrib12RotationAxis1ZE) |     member)](api/lang             |
| -   [cudaq::cudaq_json (C++       | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     class)](api/languages/cpp_api | ptsbe6detail10NoisePoint6qubitsE) |
| .html#_CPPv4N5cudaq10cudaq_jsonE) | -   [cudaq::                      |
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
| -   [cudaq::Execu                 |     (C++                          |
| tionContext::explicitMeasurements |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4I |
|     member)](api/languages/cp     | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| p_api.html#_CPPv4N5cudaq16Executi | 9async_sample_resultRK14sample_op |
| onContext20explicitMeasurementsE) | tionsRR13QuantumKernelDpRR4Args), |
| -   [cuda                         |     [\[1\]](api/languages/cp      |
| q::ExecutionContext::futureResult | p_api.html#_CPPv4I0DpEN5cudaq5pts |
|     (C++                          | be12sample_asyncE19async_sample_r |
|     member)](api/lang             | esultRKN5cudaq11noise_modelENSt6s |
| uages/cpp_api.html#_CPPv4N5cudaq1 | ize_tERR13QuantumKernelDpRR4Args) |
| 6ExecutionContext12futureResultE) | -   [cudaq::ptsbe::sample_options |
| -   [cudaq::ExecutionContext      |     (C++                          |
| ::hasConditionalsOnMeasureResults |     struct)                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     mem                           | v4N5cudaq5ptsbe14sample_optionsE) |
| ber)](api/languages/cpp_api.html# | -   [cudaq::ptsbe::sample_result  |
| _CPPv4N5cudaq16ExecutionContext31 |     (C++                          |
| hasConditionalsOnMeasureResultsE) |     class                         |
| -   [cudaq:                       | )](api/languages/cpp_api.html#_CP |
| :ExecutionContext::inKernelLaunch | Pv4N5cudaq5ptsbe13sample_resultE) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/langua           | be::sample_result::execution_data |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14inKernelLaunchE) |     function)](api/languages/c    |
| -   [cu                           | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| daq::ExecutionContext::kernelName | 3sample_result14execution_dataEv) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     member)](api/la               | sample_result::has_execution_data |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10kernelNameE) |                                   |
| -   [cud                          |    function)](api/languages/cpp_a |
| aq::ExecutionContext::kernelTrace | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
|     (C++                          | ple_result18has_execution_dataEv) |
|     member)](api/lan              | -   [cudaq::pt                    |
| guages/cpp_api.html#_CPPv4N5cudaq | sbe::sample_result::sample_result |
| 16ExecutionContext11kernelTraceE) |     (C++                          |
| -                                 |     function)](api/l              |
|    [cudaq::ExecutionContext::name | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq5ptsbe13sample_result13sample_r |
|     member)]                      | esultERRN5cudaq13sample_resultE), |
| (api/languages/cpp_api.html#_CPPv |                                   |
| 4N5cudaq16ExecutionContext4nameE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cu                           | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| daq::ExecutionContext::noiseModel | sult13sample_resultERRN5cudaq13sa |
|     (C++                          | mple_resultE18PTSBEExecutionData) |
|     member)](api/la               | -   [cudaq::ptsbe::               |
| nguages/cpp_api.html#_CPPv4N5cuda | sample_result::set_execution_data |
| q16ExecutionContext10noiseModelE) |     (C++                          |
| -   [cudaq::Exe                   |     function)](api/               |
| cutionContext::numberTrajectories | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq5ptsbe13sample_result18set_exe |
|     member)](api/languages/       | cution_dataE18PTSBEExecutionData) |
| cpp_api.html#_CPPv4N5cudaq16Execu | -   [cud                          |
| tionContext18numberTrajectoriesE) | aq::ptsbe::ShotAllocationStrategy |
| -   [c                            |     (C++                          |
| udaq::ExecutionContext::optResult |     struct)](using                |
|     (C++                          | /examples/ptsbe.html#_CPPv4N5cuda |
|     member)](api/                 | q5ptsbe22ShotAllocationStrategyE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::ptsbe::ShotAllocatio  |
| daq16ExecutionContext9optResultE) | nStrategy::ShotAllocationStrategy |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::qpuId |     function)                     |
|     (C++                          | ](using/examples/ptsbe.html#_CPPv |
|     member)](                     | 4N5cudaq5ptsbe22ShotAllocationStr |
| api/languages/cpp_api.html#_CPPv4 | ategy22ShotAllocationStrategyE4Ty |
| N5cudaq16ExecutionContext5qpuIdE) | pedNSt8optionalINSt8uint64_tEEE), |
| -   [cudaq                        |     [\[1\                         |
| ::ExecutionContext::registerNames | ]](using/examples/ptsbe.html#_CPP |
|     (C++                          | v4N5cudaq5ptsbe22ShotAllocationSt |
|     member)](api/langu            | rategy22ShotAllocationStrategyEv) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [cudaq::pt                    |
| ExecutionContext13registerNamesE) | sbe::ShotAllocationStrategy::Type |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::reorderIdx |     enum)](using/exam             |
|     (C++                          | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|     member)](api/la               | be22ShotAllocationStrategy4TypeE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::ptsbe::ShotAllocatio  |
| q16ExecutionContext10reorderIdxE) | nStrategy::Type::HIGH_WEIGHT_BIAS |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::shots |     enumerat                      |
|     (C++                          | or)](using/examples/ptsbe.html#_C |
|     member)](                     | PPv4N5cudaq5ptsbe22ShotAllocation |
| api/languages/cpp_api.html#_CPPv4 | Strategy4Type16HIGH_WEIGHT_BIASE) |
| N5cudaq16ExecutionContext5shotsE) | -   [cudaq::ptsbe::ShotAllocati   |
| -   [cudaq::                      | onStrategy::Type::LOW_WEIGHT_BIAS |
| ExecutionContext::simulationState |     (C++                          |
|     (C++                          |     enumera                       |
|     member)](api/languag          | tor)](using/examples/ptsbe.html#_ |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| ecutionContext15simulationStateE) | nStrategy4Type15LOW_WEIGHT_BIASE) |
| -                                 | -   [cudaq::ptsbe::ShotAlloc      |
|    [cudaq::ExecutionContext::spin | ationStrategy::Type::PROPORTIONAL |
|     (C++                          |     (C++                          |
|     member)]                      |     enum                          |
| (api/languages/cpp_api.html#_CPPv | erator)](using/examples/ptsbe.htm |
| 4N5cudaq16ExecutionContext4spinE) | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| -   [cudaq::                      | tionStrategy4Type12PROPORTIONALE) |
| ExecutionContext::totalIterations | -   [cudaq::ptsbe::Shot           |
|     (C++                          | AllocationStrategy::Type::UNIFORM |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |                                   |
| ecutionContext15totalIterationsE) |   enumerator)](using/examples/pts |
| -   [cudaq::ExecutionResult (C++  | be.html#_CPPv4N5cudaq5ptsbe22Shot |
|     st                            | AllocationStrategy4Type7UNIFORME) |
| ruct)](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq15ExecutionResultE) |   [cudaq::ptsbe::TraceInstruction |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::appendResult |     struct)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     functio                       | N5cudaq5ptsbe16TraceInstructionE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq:                       |
| PPv4N5cudaq15ExecutionResult12app | :ptsbe::TraceInstruction::channel |
| endResultENSt6stringENSt6size_tE) |     (C++                          |
| -   [cu                           |     member)](api/lang             |
| daq::ExecutionResult::deserialize | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     (C++                          | ptsbe16TraceInstruction7channelE) |
|     function)                     | -   [cudaq::                      |
| ](api/languages/cpp_api.html#_CPP | ptsbe::TraceInstruction::controls |
| v4N5cudaq15ExecutionResult11deser |     (C++                          |
| ializeERNSt6vectorINSt6size_tEEE) |     member)](api/langu            |
| -   [cudaq:                       | ages/cpp_api.html#_CPPv4N5cudaq5p |
| :ExecutionResult::ExecutionResult | tsbe16TraceInstruction8controlsE) |
|     (C++                          | -   [cud                          |
|     functio                       | aq::ptsbe::TraceInstruction::name |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq15ExecutionResult15Exe |     member)](api/l                |
| cutionResultE16CountsDictionary), | anguages/cpp_api.html#_CPPv4N5cud |
|     [\[1\]](api/lan               | aq5ptsbe16TraceInstruction4nameE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq                        |
| 15ExecutionResult15ExecutionResul | ::ptsbe::TraceInstruction::params |
| tE16CountsDictionaryNSt6stringE), |     (C++                          |
|     [\[2\                         |     member)](api/lan              |
| ]](api/languages/cpp_api.html#_CP | guages/cpp_api.html#_CPPv4N5cudaq |
| Pv4N5cudaq15ExecutionResult15Exec | 5ptsbe16TraceInstruction6paramsE) |
| utionResultE16CountsDictionaryd), | -   [cudaq:                       |
|                                   | :ptsbe::TraceInstruction::targets |
|    [\[3\]](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq15ExecutionResu |     member)](api/lang             |
| lt15ExecutionResultENSt6stringE), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     [\[4\                         | ptsbe16TraceInstruction7targetsE) |
| ]](api/languages/cpp_api.html#_CP | -   [cudaq::ptsbe::T              |
| Pv4N5cudaq15ExecutionResult15Exec | raceInstruction::TraceInstruction |
| utionResultERK15ExecutionResult), |     (C++                          |
|     [\[5\]](api/language          |                                   |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |   function)](api/languages/cpp_ap |
| cutionResult15ExecutionResultEd), | i.html#_CPPv4N5cudaq5ptsbe16Trace |
|     [\[6\]](api/languag           | Instruction16TraceInstructionE20T |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | raceInstructionTypeNSt6stringENSt |
| ecutionResult15ExecutionResultEv) | 6vectorINSt6size_tEEENSt6vectorIN |
| -   [                             | St6size_tEEENSt6vectorIdEENSt8opt |
| cudaq::ExecutionResult::operator= | ionalIN5cudaq13kraus_channelEEE), |
|     (C++                          |     [\[1\]](api/languages/cpp_a   |
|     function)](api/languages/     | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| cpp_api.html#_CPPv4N5cudaq15Execu | eInstruction16TraceInstructionEv) |
| tionResultaSERK15ExecutionResult) | -   [cud                          |
| -   [c                            | aq::ptsbe::TraceInstruction::type |
| udaq::ExecutionResult::operator== |     (C++                          |
|     (C++                          |     member)](api/l                |
|     function)](api/languages/c    | anguages/cpp_api.html#_CPPv4N5cud |
| pp_api.html#_CPPv4NK5cudaq15Execu | aq5ptsbe16TraceInstruction4typeE) |
| tionResulteqERK15ExecutionResult) | -   [c                            |
| -   [cud                          | udaq::ptsbe::TraceInstructionType |
| aq::ExecutionResult::registerName |     (C++                          |
|     (C++                          |     enum)](api/                   |
|     member)](api/lan              | languages/cpp_api.html#_CPPv4N5cu |
| guages/cpp_api.html#_CPPv4N5cudaq | daq5ptsbe20TraceInstructionTypeE) |
| 15ExecutionResult12registerNameE) | -   [cudaq::                      |
| -   [cudaq                        | ptsbe::TraceInstructionType::Gate |
| ::ExecutionResult::sequentialData |     (C++                          |
|     (C++                          |     enumerator)](api/langu        |
|     member)](api/langu            | ages/cpp_api.html#_CPPv4N5cudaq5p |
| ages/cpp_api.html#_CPPv4N5cudaq15 | tsbe20TraceInstructionType4GateE) |
| ExecutionResult14sequentialDataE) | -   [cudaq::ptsbe::               |
| -   [                             | TraceInstructionType::Measurement |
| cudaq::ExecutionResult::serialize |     (C++                          |
|     (C++                          |                                   |
|     function)](api/l              |    enumerator)](api/languages/cpp |
| anguages/cpp_api.html#_CPPv4NK5cu | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| daq15ExecutionResult9serializeEv) | aceInstructionType11MeasurementE) |
| -   [cudaq::fermion_handler (C++  | -   [cudaq::p                     |
|     c                             | tsbe::TraceInstructionType::Noise |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15fermion_handlerE) |     enumerator)](api/langua       |
| -   [cudaq::fermion_op (C++       | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     type)](api/languages/cpp_api  | sbe20TraceInstructionType5NoiseE) |
| .html#_CPPv4N5cudaq10fermion_opE) | -   [                             |
| -   [cudaq::fermion_op_term (C++  | cudaq::ptsbe::TrajectoryPredicate |
|                                   |     (C++                          |
| type)](api/languages/cpp_api.html |     type)](api                    |
| #_CPPv4N5cudaq15fermion_op_termE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::FermioniqQPU (C++     | udaq5ptsbe19TrajectoryPredicateE) |
|                                   | -   [cudaq::QPU (C++              |
|   class)](api/languages/cpp_api.h |     class)](api/languages         |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| -   [cudaq::get_state (C++        | -   [cudaq::QPU::beginExecution   |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function                      |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | )](api/languages/cpp_api.html#_CP |
| ateEDaRR13QuantumKernelDpRR4Args) | Pv4N5cudaq3QPU14beginExecutionEv) |
| -   [cudaq::GPUEmulatedQPU (C++   | -   [cuda                         |
|                                   | q::QPU::configureExecutionContext |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14GPUEmulatedQPUE) |     funct                         |
| -   [cudaq::gradient (C++         | ion)](api/languages/cpp_api.html# |
|     class)](api/languages/cpp_    | _CPPv4NK5cudaq3QPU25configureExec |
| api.html#_CPPv4N5cudaq8gradientE) | utionContextER16ExecutionContext) |
| -   [cudaq::gradient::clone (C++  | -   [cudaq::QPU::endExecution     |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     functi                        |
| l#_CPPv4N5cudaq8gradient5cloneEv) | on)](api/languages/cpp_api.html#_ |
| -   [cudaq::gradient::compute     | CPPv4N5cudaq3QPU12endExecutionEv) |
|     (C++                          | -   [cudaq::QPU::enqueue (C++     |
|     function)](api/language       |     function)](ap                 |
| s/cpp_api.html#_CPPv4N5cudaq8grad | i/languages/cpp_api.html#_CPPv4N5 |
| ient7computeERKNSt6vectorIdEERKNS | cudaq3QPU7enqueueER11QuantumTask) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cud                          |
|     [\[1\]](ap                    | aq::QPU::finalizeExecutionContext |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq8gradient7computeERKNSt6vect |     func                          |
| orIdEERNSt6vectorIdEERK7spin_opd) | tion)](api/languages/cpp_api.html |
| -   [cudaq::gradient::gradient    | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     function)](api/lang           | -   [cudaq::QPU::getCompileTarget |
| uages/cpp_api.html#_CPPv4I00EN5cu |     (C++                          |
| daq8gradient8gradientER7KernelT), |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|    [\[1\]](api/languages/cpp_api. | 4N5cudaq3QPU16getCompileTargetEb) |
| html#_CPPv4I00EN5cudaq8gradient8g | -   [cudaq::QPU::getConnectivity  |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\                         |     function)                     |
| ]](api/languages/cpp_api.html#_CP | ](api/languages/cpp_api.html#_CPP |
| Pv4I00EN5cudaq8gradient8gradientE | v4N5cudaq3QPU15getConnectivityEv) |
| RR13QuantumKernelRR10ArgsMapper), | -                                 |
|     [\[3                          | [cudaq::QPU::getExecutionThreadId |
| \]](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq8gradient8gradientERRN |     function)](api/               |
| St8functionIFvNSt6vectorIdEEEEE), | languages/cpp_api.html#_CPPv4NK5c |
|     [\[                           | udaq3QPU20getExecutionThreadIdEv) |
| 4\]](api/languages/cpp_api.html#_ | -   [cudaq::QPU::getNumQubits     |
| CPPv4N5cudaq8gradient8gradientEv) |     (C++                          |
| -   [cudaq::gradient::setArgs     |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     fu                            | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| nction)](api/languages/cpp_api.ht | -   [cudaq::QPU::isEmulated (C++  |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     func                          |
| tArgsEvR13QuantumKernelDpRR4Args) | tion)](api/languages/cpp_api.html |
| -   [cudaq::gradient::setKernel   | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     (C++                          | -   [cudaq::QPU::isSimulator (C++ |
|     function)](api/languages/c    |     funct                         |
| pp_api.html#_CPPv4I0EN5cudaq8grad | ion)](api/languages/cpp_api.html# |
| ient9setKernelEvR13QuantumKernel) | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| -   [cud                          | -   [cudaq::QPU::onRandomSeedSet  |
| aq::gradients::central_difference |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     class)](api/la                | uages/cpp_api.html#_CPPv4N5cudaq3 |
| nguages/cpp_api.html#_CPPv4N5cuda | QPU15onRandomSeedSetENSt6size_tE) |
| q9gradients18central_differenceE) | -   [cudaq::QPU::QPU (C++         |
| -   [cudaq::gra                   |     functio                       |
| dients::central_difference::clone | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
|     function)](api/languages      |                                   |
| /cpp_api.html#_CPPv4N5cudaq9gradi |  [\[1\]](api/languages/cpp_api.ht |
| ents18central_difference5cloneEv) | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| -   [cudaq::gradi                 |     [\[2\]](api/languages/cpp_    |
| ents::central_difference::compute | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     (C++                          | -   [cudaq::QPU::setId (C++       |
|     function)](                   |     function                      |
| api/languages/cpp_api.html#_CPPv4 | )](api/languages/cpp_api.html#_CP |
| N5cudaq9gradients18central_differ | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| ence7computeERKNSt6vectorIdEERKNS | -   [cudaq::QPU::setShots (C++    |
| t8functionIFdNSt6vectorIdEEEEEd), |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|   [\[1\]](api/languages/cpp_api.h | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| tml#_CPPv4N5cudaq9gradients18cent | -   [cudaq::QPU::\~QPU (C++       |
| ral_difference7computeERKNSt6vect |     function)](api/languages/cp   |
| orIdEERNSt6vectorIdEERK7spin_opd) | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| -   [cudaq::gradie                | -   [cudaq::QPUState (C++         |
| nts::central_difference::gradient |     class)](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4N5cudaq8QPUStateE) |
|     functio                       | -   [cudaq::qreg (C++             |
| n)](api/languages/cpp_api.html#_C |     class)](api/lan               |
| PPv4I00EN5cudaq9gradients18centra | guages/cpp_api.html#_CPPv4I_NSt6s |
| l_difference8gradientER7KernelT), | ize_tE_NSt6size_tEEN5cudaq4qregE) |
|     [\[1\]](api/langua            | -   [cudaq::qreg::back (C++       |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     function)                     |
| q9gradients18central_difference8g | ](api/languages/cpp_api.html#_CPP |
| radientER7KernelTRR10ArgsMapper), | v4N5cudaq4qreg4backENSt6size_tE), |
|     [\[2\]](api/languages/cpp_    |     [\[1\]](api/languages/cpp_ap  |
| api.html#_CPPv4I00EN5cudaq9gradie | i.html#_CPPv4N5cudaq4qreg4backEv) |
| nts18central_difference8gradientE | -   [cudaq::qreg::begin (C++      |
| RR13QuantumKernelRR10ArgsMapper), |                                   |
|     [\[3\]](api/languages/cpp     |  function)](api/languages/cpp_api |
| _api.html#_CPPv4N5cudaq9gradients | .html#_CPPv4N5cudaq4qreg5beginEv) |
| 18central_difference8gradientERRN | -   [cudaq::qreg::clear (C++      |
| St8functionIFvNSt6vectorIdEEEEE), |                                   |
|     [\[4\]](api/languages/cp      |  function)](api/languages/cpp_api |
| p_api.html#_CPPv4N5cudaq9gradient | .html#_CPPv4N5cudaq4qreg5clearEv) |
| s18central_difference8gradientEv) | -   [cudaq::qreg::front (C++      |
| -   [cud                          |     function)]                    |
| aq::gradients::forward_difference | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4N5cudaq4qreg5frontENSt6size_tE), |
|     class)](api/la                |     [\[1\]](api/languages/cpp_api |
| nguages/cpp_api.html#_CPPv4N5cuda | .html#_CPPv4N5cudaq4qreg5frontEv) |
| q9gradients18forward_differenceE) | -   [cudaq::qreg::operator\[\]    |
| -   [cudaq::gra                   |     (C++                          |
| dients::forward_difference::clone |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     function)](api/languages      | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::qreg::qreg (C++       |
| ents18forward_difference5cloneEv) |     function)                     |
| -   [cudaq::gradi                 | ](api/languages/cpp_api.html#_CPP |
| ents::forward_difference::compute | v4N5cudaq4qreg4qregENSt6size_tE), |
|     (C++                          |     [\[1\]](api/languages/cpp_ap  |
|     function)](                   | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qreg::size (C++       |
| N5cudaq9gradients18forward_differ |                                   |
| ence7computeERKNSt6vectorIdEERKNS |  function)](api/languages/cpp_api |
| t8functionIFdNSt6vectorIdEEEEEd), | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|                                   | -   [cudaq::qreg::slice (C++      |
|   [\[1\]](api/languages/cpp_api.h |     function)](api/langu          |
| tml#_CPPv4N5cudaq9gradients18forw | ages/cpp_api.html#_CPPv4N5cudaq4q |
| ard_difference7computeERKNSt6vect | reg5sliceENSt6size_tENSt6size_tE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::qreg::value_type (C++ |
| -   [cudaq::gradie                |                                   |
| nts::forward_difference::gradient | type)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq4qreg10value_typeE) |
|     functio                       | -   [cudaq::qspan (C++            |
| n)](api/languages/cpp_api.html#_C |     class)](api/lang              |
| PPv4I00EN5cudaq9gradients18forwar | uages/cpp_api.html#_CPPv4I_NSt6si |
| d_difference8gradientER7KernelT), | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     [\[1\]](api/langua            | -   [cudaq::QuakeValue (C++       |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     class)](api/languages/cpp_api |
| q9gradients18forward_difference8g | .html#_CPPv4N5cudaq10QuakeValueE) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::Q                     |
|     [\[2\]](api/languages/cpp_    | uakeValue::canValidateNumElements |
| api.html#_CPPv4I00EN5cudaq9gradie |     (C++                          |
| nts18forward_difference8gradientE |     function)](api/languages      |
| RR13QuantumKernelRR10ArgsMapper), | /cpp_api.html#_CPPv4N5cudaq10Quak |
|     [\[3\]](api/languages/cpp     | eValue22canValidateNumElementsEv) |
| _api.html#_CPPv4N5cudaq9gradients | -                                 |
| 18forward_difference8gradientERRN |  [cudaq::QuakeValue::constantSize |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[4\]](api/languages/cp      |     function)](api                |
| p_api.html#_CPPv4N5cudaq9gradient | /languages/cpp_api.html#_CPPv4N5c |
| s18forward_difference8gradientEv) | udaq10QuakeValue12constantSizeEv) |
| -   [                             | -   [cudaq::QuakeValue::dump (C++ |
| cudaq::gradients::parameter_shift |     function)](api/lan            |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     class)](api                   | 10QuakeValue4dumpERNSt7ostreamE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\                            |
| udaq9gradients15parameter_shiftE) | [1\]](api/languages/cpp_api.html# |
| -   [cudaq::                      | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| gradients::parameter_shift::clone | -   [cudaq                        |
|     (C++                          | ::QuakeValue::getRequiredElements |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     function)](api/langua         |
| adients15parameter_shift5cloneEv) | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| -   [cudaq::gr                    | uakeValue19getRequiredElementsEv) |
| adients::parameter_shift::compute | -   [cudaq::QuakeValue::getValue  |
|     (C++                          |     (C++                          |
|     function                      |     function)]                    |
| )](api/languages/cpp_api.html#_CP | (api/languages/cpp_api.html#_CPPv |
| Pv4N5cudaq9gradients15parameter_s | 4NK5cudaq10QuakeValue8getValueEv) |
| hift7computeERKNSt6vectorIdEERKNS | -   [cudaq::QuakeValue::inverse   |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|     [\[1\]](api/languages/cpp_ap  |     function)                     |
| i.html#_CPPv4N5cudaq9gradients15p | ](api/languages/cpp_api.html#_CPP |
| arameter_shift7computeERKNSt6vect | v4NK5cudaq10QuakeValue7inverseEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -                                 |
| -   [cudaq::gra                   |    [cudaq::QuakeValue::isSequence |
| dients::parameter_shift::gradient |     (C++                          |
|     (C++                          |     function)](a                  |
|     func                          | pi/languages/cpp_api.html#_CPPv4N |
| tion)](api/languages/cpp_api.html | 5cudaq10QuakeValue10isSequenceEv) |
| #_CPPv4I00EN5cudaq9gradients15par | -                                 |
| ameter_shift8gradientER7KernelT), |    [cudaq::QuakeValue::operator\* |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4I00EN5c |     function)](api                |
| udaq9gradients15parameter_shift8g | /languages/cpp_api.html#_CPPv4N5c |
| radientER7KernelTRR10ArgsMapper), | udaq10QuakeValuemlE10QuakeValue), |
|     [\[2\]](api/languages/c       |                                   |
| pp_api.html#_CPPv4I00EN5cudaq9gra | [\[1\]](api/languages/cpp_api.htm |
| dients15parameter_shift8gradientE | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::QuakeValue::operator+ |
|     [\[3\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq9gradie |     function)](api                |
| nts15parameter_shift8gradientERRN | /languages/cpp_api.html#_CPPv4N5c |
| St8functionIFvNSt6vectorIdEEEEE), | udaq10QuakeValueplE10QuakeValue), |
|     [\[4\]](api/languages         |     [                             |
| /cpp_api.html#_CPPv4N5cudaq9gradi | \[1\]](api/languages/cpp_api.html |
| ents15parameter_shift8gradientEv) | #_CPPv4N5cudaq10QuakeValueplEKd), |
| -   [cudaq::kernel_builder (C++   |                                   |
|     clas                          | [\[2\]](api/languages/cpp_api.htm |
| s)](api/languages/cpp_api.html#_C | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| PPv4IDpEN5cudaq14kernel_builderE) | -   [cudaq::QuakeValue::operator- |
| -   [c                            |     (C++                          |
| udaq::kernel_builder::constantVal |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/la             | udaq10QuakeValuemiE10QuakeValue), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [                             |
| q14kernel_builder11constantValEd) | \[1\]](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|  [cudaq::kernel_builder::detector |     [                             |
|     (C++                          | \[2\]](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|    function)](api/languages/cpp_a |                                   |
| pi.html#_CPPv4IDpEN5cudaq14kernel | [\[3\]](api/languages/cpp_api.htm |
| _builder8detectorEvDpRR8MeasArgs) | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| -                                 | -   [cudaq::QuakeValue::operator/ |
| [cudaq::kernel_builder::detectors |     (C++                          |
|     (C++                          |     function)](api                |
|     func                          | /languages/cpp_api.html#_CPPv4N5c |
| tion)](api/languages/cpp_api.html | udaq10QuakeValuedvE10QuakeValue), |
| #_CPPv4N5cudaq14kernel_builder9de |                                   |
| tectorsE10QuakeValue10QuakeValue) | [\[1\]](api/languages/cpp_api.htm |
| -   [cu                           | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| daq::kernel_builder::getArguments | -                                 |
|     (C++                          |  [cudaq::QuakeValue::operator\[\] |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api                |
| 14kernel_builder12getArgumentsEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cu                           | udaq10QuakeValueixEKNSt6size_tE), |
| daq::kernel_builder::getNumParams |     [\[1\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     function)](api/lan            | daq10QuakeValueixERK10QuakeValue) |
| guages/cpp_api.html#_CPPv4N5cudaq | -                                 |
| 14kernel_builder12getNumParamsEv) |    [cudaq::QuakeValue::QuakeValue |
| -   [cud                          |     (C++                          |
| aq::kernel_builder::isArgSequence |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     function)](api/languages/cpp_ | akeValue10QuakeValueERN4mlir20Imp |
| api.html#_CPPv4N5cudaq14kernel_bu | licitLocOpBuilderEN4mlir5ValueE), |
| ilder13isArgSequenceENSt6size_tE) |     [\[1\]                        |
| -   [cuda                         | ](api/languages/cpp_api.html#_CPP |
| q::kernel_builder::kernel_builder | v4N5cudaq10QuakeValue10QuakeValue |
|     (C++                          | ERN4mlir20ImplicitLocOpBuilderEd) |
|     function)](api/languages/cpp  | -   [cudaq::QuakeValue::size (C++ |
| _api.html#_CPPv4N5cudaq14kernel_b |     funct                         |
| uilder14kernel_builderERNSt6vecto | ion)](api/languages/cpp_api.html# |
| rIN6detail17KernelBuilderTypeEEE) | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| -   [cudaq::k                     | -   [cudaq::QuakeValue::slice     |
| ernel_builder::logical_observable |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     function)                     | api.html#_CPPv4N5cudaq10QuakeValu |
| ](api/languages/cpp_api.html#_CPP | e5sliceEKNSt6size_tEKNSt6size_tE) |
| v4IDpEN5cudaq14kernel_builder18lo | -   [cudaq::quantum_platform (C++ |
| gical_observableEvDpRR8MeasArgs), |     cl                            |
|     [\[1\]](ap                    | ass)](api/languages/cpp_api.html# |
| i/languages/cpp_api.html#_CPPv4N5 | _CPPv4N5cudaq16quantum_platformE) |
| cudaq14kernel_builder18logical_ob | -   [cudaq:                       |
| servableE10QuakeValueNSt6size_tE) | :quantum_platform::beginExecution |
| -   [cudaq::kernel_builder::name  |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)                     | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ](api/languages/cpp_api.html#_CPP | antum_platform14beginExecutionEv) |
| v4N5cudaq14kernel_builder4nameEv) | -   [cudaq::quantum_pl            |
| -                                 | atform::configureExecutionContext |
|    [cudaq::kernel_builder::qalloc |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     function)](api/language       | uages/cpp_api.html#_CPPv4NK5cudaq |
| s/cpp_api.html#_CPPv4N5cudaq14ker | 16quantum_platform25configureExec |
| nel_builder6qallocE10QuakeValue), | utionContextER16ExecutionContext) |
|     [\[1\]](api/language          | -   [cuda                         |
| s/cpp_api.html#_CPPv4N5cudaq14ker | q::quantum_platform::connectivity |
| nel_builder6qallocEKNSt6size_tE), |     (C++                          |
|     [\[2                          |     function)](api/langu          |
| \]](api/languages/cpp_api.html#_C | ages/cpp_api.html#_CPPv4N5cudaq16 |
| PPv4N5cudaq14kernel_builder6qallo | quantum_platform12connectivityEv) |
| cERNSt6vectorINSt7complexIdEEEE), | -   [cuda                         |
|     [\[3\]](                      | q::quantum_platform::endExecution |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14kernel_builder6qallocEv) |     function)](api/langu          |
| -   [cudaq::kernel_builder::swap  | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     (C++                          | quantum_platform12endExecutionEv) |
|     function)](api/language       | -   [cudaq::q                     |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | uantum_platform::enqueueAsyncTask |
| 4kernel_builder4swapEvRK10QuakeVa |     (C++                          |
| lueRK10QuakeValueRK10QuakeValue), |     function)](api/languages/     |
|                                   | cpp_api.html#_CPPv4N5cudaq16quant |
| [\[1\]](api/languages/cpp_api.htm | um_platform16enqueueAsyncTaskEKNS |
| l#_CPPv4I00EN5cudaq14kernel_build | t6size_tER19KernelExecutionTask), |
| er4swapEvRKNSt6vectorI10QuakeValu |     [\[1\]](api/languag           |
| eEERK10QuakeValueRK10QuakeValue), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|                                   | antum_platform16enqueueAsyncTaskE |
| [\[2\]](api/languages/cpp_api.htm | KNSt6size_tERNSt8functionIFvvEEE) |
| l#_CPPv4N5cudaq14kernel_builder4s | -   [cudaq::quantum_p             |
| wapERK10QuakeValueRK10QuakeValue) | latform::finalizeExecutionContext |
| -   [cudaq::KernelExecutionTask   |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     type                          | pp_api.html#_CPPv4NK5cudaq16quant |
| )](api/languages/cpp_api.html#_CP | um_platform24finalizeExecutionCon |
| Pv4N5cudaq19KernelExecutionTaskE) | textERN5cudaq16ExecutionContextE) |
| -   [cudaq::KernelThunkResultType | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_codegen_config |
|     struct)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/languages/c    |
| 4N5cudaq21KernelThunkResultTypeE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [cudaq::KernelThunkType (C++  | m_platform18get_codegen_configEv) |
|                                   | -   [cuda                         |
| type)](api/languages/cpp_api.html | q::quantum_platform::get_exec_ctx |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     (C++                          |
| -   [cudaq::kraus_channel (C++    |     function)](api/langua         |
|                                   | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|  class)](api/languages/cpp_api.ht | quantum_platform12get_exec_ctxEv) |
| ml#_CPPv4N5cudaq13kraus_channelE) | -   [c                            |
| -   [cudaq::kraus_channel::empty  | udaq::quantum_platform::get_noise |
|     (C++                          |     (C++                          |
|     function)]                    |     function)](api/languages/c    |
| (api/languages/cpp_api.html#_CPPv | pp_api.html#_CPPv4N5cudaq16quantu |
| 4NK5cudaq13kraus_channel5emptyEv) | m_platform9get_noiseENSt6size_tE) |
| -   [cudaq::kraus_c               | -   [cudaq:                       |
| hannel::generateUnitaryParameters | :quantum_platform::get_num_qubits |
|     (C++                          |     (C++                          |
|                                   |                                   |
|    function)](api/languages/cpp_a | function)](api/languages/cpp_api. |
| pi.html#_CPPv4N5cudaq13kraus_chan | html#_CPPv4NK5cudaq16quantum_plat |
| nel25generateUnitaryParametersEv) | form14get_num_qubitsENSt6size_tE) |
| -                                 | -   [cudaq::qua                   |
|    [cudaq::kraus_channel::get_ops | ntum_platform::get_runtime_target |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](api/languages/cp   |
| pi/languages/cpp_api.html#_CPPv4N | p_api.html#_CPPv4NK5cudaq16quantu |
| K5cudaq13kraus_channel7get_opsEv) | m_platform18get_runtime_targetEv) |
| -   [cud                          | -   [cud                          |
| aq::kraus_channel::identity_flags | aq::quantum_platform::is_emulated |
|     (C++                          |     (C++                          |
|     member)](api/lan              |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq |    function)](api/languages/cpp_a |
| 13kraus_channel14identity_flagsE) | pi.html#_CPPv4NK5cudaq16quantum_p |
| -   [cud                          | latform11is_emulatedENSt6size_tE) |
| aq::kraus_channel::is_identity_op | -   [cudaq::                      |
|     (C++                          | quantum_platform::is_library_mode |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](api/languages      |
| pi.html#_CPPv4NK5cudaq13kraus_cha | /cpp_api.html#_CPPv4NK5cudaq16qua |
| nnel14is_identity_opENSt6size_tE) | ntum_platform15is_library_modeEv) |
| -   [cudaq::                      | -   [c                            |
| kraus_channel::is_unitary_mixture | udaq::quantum_platform::is_remote |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/languages/cp   |
| /cpp_api.html#_CPPv4NK5cudaq13kra | p_api.html#_CPPv4NK5cudaq16quantu |
| us_channel18is_unitary_mixtureEv) | m_platform9is_remoteENSt6size_tE) |
| -   [cu                           | -   [cuda                         |
| daq::kraus_channel::kraus_channel | q::quantum_platform::is_simulator |
|     (C++                          |     (C++                          |
|     function)](api/lang           |                                   |
| uages/cpp_api.html#_CPPv4IDpEN5cu |   function)](api/languages/cpp_ap |
| daq13kraus_channel13kraus_channel | i.html#_CPPv4NK5cudaq16quantum_pl |
| EDpRRNSt16initializer_listI1TEE), | atform12is_simulatorENSt6size_tE) |
|                                   | -   [cudaq:                       |
|  [\[1\]](api/languages/cpp_api.ht | :quantum_platform::list_platforms |
| ml#_CPPv4N5cudaq13kraus_channel13 |     (C++                          |
| kraus_channelERK13kraus_channel), |     function)](api/languag        |
|     [\[2\]                        | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ](api/languages/cpp_api.html#_CPP | antum_platform14list_platformsEv) |
| v4N5cudaq13kraus_channel13kraus_c | -                                 |
| hannelERKNSt6vectorI8kraus_opEE), |    [cudaq::quantum_platform::name |
|     [\[3\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](a                  |
| v4N5cudaq13kraus_channel13kraus_c | pi/languages/cpp_api.html#_CPPv4N |
| hannelERRNSt6vectorI8kraus_opEE), | K5cudaq16quantum_platform4nameEv) |
|     [\[4\]](api/lan               | -   [                             |
| guages/cpp_api.html#_CPPv4N5cudaq | cudaq::quantum_platform::num_qpus |
| 13kraus_channel13kraus_channelEv) |     (C++                          |
| -                                 |     function)](api/l              |
| [cudaq::kraus_channel::noise_type | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq16quantum_platform8num_qpusEv) |
|     member)](api                  | -   [cudaq::                      |
| /languages/cpp_api.html#_CPPv4N5c | quantum_platform::onRandomSeedSet |
| udaq13kraus_channel10noise_typeE) |     (C++                          |
| -                                 |                                   |
|   [cudaq::kraus_channel::op_names | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq16quantum_platf |
|     member)](                     | orm15onRandomSeedSetENSt6size_tE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq:                       |
| N5cudaq13kraus_channel8op_namesE) | :quantum_platform::reset_exec_ctx |
| -                                 |     (C++                          |
|  [cudaq::kraus_channel::operator= |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     function)](api/langua         | antum_platform14reset_exec_ctxEv) |
| ges/cpp_api.html#_CPPv4N5cudaq13k | -   [cud                          |
| raus_channelaSERK13kraus_channel) | aq::quantum_platform::reset_noise |
| -   [c                            |     (C++                          |
| udaq::kraus_channel::operator\[\] |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq16quantum_p |
|     function)](api/l              | latform11reset_noiseENSt6size_tE) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cuda                         |
| aq13kraus_channelixEKNSt6size_tE) | q::quantum_platform::set_exec_ctx |
| -                                 |     (C++                          |
| [cudaq::kraus_channel::parameters |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     member)](api                  | _CPPv4N5cudaq16quantum_platform12 |
| /languages/cpp_api.html#_CPPv4N5c | set_exec_ctxEP16ExecutionContext) |
| udaq13kraus_channel10parametersE) | -   [c                            |
| -   [cudaq::krau                  | udaq::quantum_platform::set_noise |
| s_channel::populateDefaultOpNames |     (C++                          |
|     (C++                          |     function                      |
|     function)](api/languages/cp   | )](api/languages/cpp_api.html#_CP |
| p_api.html#_CPPv4N5cudaq13kraus_c | Pv4N5cudaq16quantum_platform9set_ |
| hannel22populateDefaultOpNamesEv) | noiseEPK11noise_modelNSt6size_tE) |
| -   [cu                           | -   [cudaq::quantum_platfor       |
| daq::kraus_channel::probabilities | m::supports_explicit_measurements |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api/l              |
| nguages/cpp_api.html#_CPPv4N5cuda | anguages/cpp_api.html#_CPPv4NK5cu |
| q13kraus_channel13probabilitiesE) | daq16quantum_platform30supports_e |
| -                                 | xplicit_measurementsENSt6size_tE) |
|  [cudaq::kraus_channel::push_back | -   [cuda                         |
|     (C++                          | q::quantum_platform::supports_jit |
|     function)](api                |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |                                   |
| udaq13kraus_channel9push_backE8kr |   function)](api/languages/cpp_ap |
| aus_opNSt8optionalINSt6stringEEE) | i.html#_CPPv4NK5cudaq16quantum_pl |
| -   [cudaq::kraus_channel::size   | atform12supports_jitENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_pla           |
|     function)                     | tform::supports_task_distribution |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4NK5cudaq13kraus_channel4sizeEv) |     fu                            |
| -   [                             | nction)](api/languages/cpp_api.ht |
| cudaq::kraus_channel::unitary_ops | ml#_CPPv4NK5cudaq16quantum_platfo |
|     (C++                          | rm26supports_task_distributionEv) |
|     member)](api/                 | -   [cudaq::quantum               |
| languages/cpp_api.html#_CPPv4N5cu | _platform::with_execution_context |
| daq13kraus_channel11unitary_opsE) |     (C++                          |
| -   [cudaq::kraus_op (C++         |     function)                     |
|     struct)](api/languages/cpp_   | ](api/languages/cpp_api.html#_CPP |
| api.html#_CPPv4N5cudaq8kraus_opE) | v4I0DpEN5cudaq16quantum_platform2 |
| -   [cudaq::kraus_op::adjoint     | 2with_execution_contextEDaR16Exec |
|     (C++                          | utionContextRR8CallableDpRR4Args) |
|     functi                        | -   [cudaq::QuantumTask (C++      |
| on)](api/languages/cpp_api.html#_ |     type)](api/languages/cpp_api. |
| CPPv4NK5cudaq8kraus_op7adjointEv) | html#_CPPv4N5cudaq11QuantumTaskE) |
| -   [cudaq::kraus_op::data (C++   | -   [cudaq::qubit (C++            |
|                                   |     type)](api/languages/c        |
|  member)](api/languages/cpp_api.h | pp_api.html#_CPPv4N5cudaq5qubitE) |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | -   [cudaq::QubitConnectivity     |
| -   [cudaq::kraus_op::kraus_op    |     (C++                          |
|     (C++                          |     ty                            |
|     func                          | pe)](api/languages/cpp_api.html#_ |
| tion)](api/languages/cpp_api.html | CPPv4N5cudaq17QubitConnectivityE) |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | -   [cudaq::QubitEdge (C++        |
| opERRNSt16initializer_listI1TEE), |     type)](api/languages/cpp_a    |
|                                   | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|  [\[1\]](api/languages/cpp_api.ht | -   [cudaq::qudit (C++            |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o |     clas                          |
| pENSt6vectorIN5cudaq7complexEEE), | s)](api/languages/cpp_api.html#_C |
|     [\[2\]](api/l                 | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cudaq::qudit::qudit (C++     |
| aq8kraus_op8kraus_opERK8kraus_op) |                                   |
| -   [cudaq::kraus_op::nCols (C++  | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4N5cudaq5qudit5quditEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::QuEraRemoteRESTQPU    |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     (C++                          |
| -   [cudaq::kraus_op::nRows (C++  |     clas                          |
|                                   | s)](api/languages/cpp_api.html#_C |
| member)](api/languages/cpp_api.ht | PPv4N5cudaq18QuEraRemoteRESTQPUE) |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | -   [cudaq::qvector (C++          |
| -   [cudaq::kraus_op::operator=   |     class)                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)                     | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::qvector::back (C++    |
| v4N5cudaq8kraus_opaSERK8kraus_op) |     function)](a                  |
| -   [cudaq::kraus_op::precision   | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq7qvector4backENSt6size_tE), |
|     memb                          |                                   |
| er)](api/languages/cpp_api.html#_ |   [\[1\]](api/languages/cpp_api.h |
| CPPv4N5cudaq8kraus_op9precisionE) | tml#_CPPv4N5cudaq7qvector4backEv) |
| -   [cudaq::KrausSelection (C++   | -   [cudaq::qvector::begin (C++   |
|     s                             |     fu                            |
| truct)](api/languages/cpp_api.htm | nction)](api/languages/cpp_api.ht |
| l#_CPPv4N5cudaq14KrausSelectionE) | ml#_CPPv4N5cudaq7qvector5beginEv) |
| -   [cudaq:                       | -   [cudaq::qvector::clear (C++   |
| :KrausSelection::circuit_location |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     member)](api/langua           | ml#_CPPv4N5cudaq7qvector5clearEv) |
| ges/cpp_api.html#_CPPv4N5cudaq14K | -   [cudaq::qvector::end (C++     |
| rausSelection16circuit_locationE) |                                   |
| -                                 | function)](api/languages/cpp_api. |
|  [cudaq::KrausSelection::is_error | html#_CPPv4N5cudaq7qvector3endEv) |
|     (C++                          | -   [cudaq::qvector::front (C++   |
|     member)](a                    |     function)](ap                 |
| pi/languages/cpp_api.html#_CPPv4N | i/languages/cpp_api.html#_CPPv4N5 |
| 5cudaq14KrausSelection8is_errorE) | cudaq7qvector5frontENSt6size_tE), |
| -   [cudaq::Kra                   |                                   |
| usSelection::kraus_operator_index |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     member)](api/languages/       | -   [cudaq::qvector::operator=    |
| cpp_api.html#_CPPv4N5cudaq14Kraus |     (C++                          |
| Selection20kraus_operator_indexE) |     functio                       |
| -   [cuda                         | n)](api/languages/cpp_api.html#_C |
| q::KrausSelection::KrausSelection | PPv4N5cudaq7qvectoraSERK7qvector) |
|     (C++                          | -   [cudaq::qvector::operator\[\] |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)                     |
| 5cudaq14KrausSelection14KrausSele | ](api/languages/cpp_api.html#_CPP |
| ctionENSt6size_tENSt6vectorINSt6s | v4N5cudaq7qvectorixEKNSt6size_tE) |
| ize_tEEENSt6stringENSt6size_tEb), | -   [cudaq::qvector::qvector (C++ |
|     [\[1\]](api/langu             |     function)](api/               |
| ages/cpp_api.html#_CPPv4N5cudaq14 | languages/cpp_api.html#_CPPv4N5cu |
| KrausSelection14KrausSelectionEv) | daq7qvector7qvectorENSt6size_tE), |
| -                                 |     [\[1\]](a                     |
|   [cudaq::KrausSelection::op_name | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq7qvector7qvectorERK5state), |
|     member)](                     |     [\[2\]](api                   |
| api/languages/cpp_api.html#_CPPv4 | /languages/cpp_api.html#_CPPv4N5c |
| N5cudaq14KrausSelection7op_nameE) | udaq7qvector7qvectorERK7qvector), |
| -   [                             |     [\[3\]](ap                    |
| cudaq::KrausSelection::operator== | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq7qvector7qvectorERR7qvector) |
|     function)](api/languages      | -   [cudaq::qvector::size (C++    |
| /cpp_api.html#_CPPv4NK5cudaq14Kra |     fu                            |
| usSelectioneqERK14KrausSelection) | nction)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|    [cudaq::KrausSelection::qubits | -   [cudaq::qvector::slice (C++   |
|     (C++                          |     function)](api/language       |
|     member)]                      | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| (api/languages/cpp_api.html#_CPPv | tor5sliceENSt6size_tENSt6size_tE) |
| 4N5cudaq14KrausSelection6qubitsE) | -   [cudaq::qvector::value_type   |
| -   [cudaq::KrausTrajectory (C++  |     (C++                          |
|     st                            |     typ                           |
| ruct)](api/languages/cpp_api.html | e)](api/languages/cpp_api.html#_C |
| #_CPPv4N5cudaq15KrausTrajectoryE) | PPv4N5cudaq7qvector10value_typeE) |
| -                                 | -   [cudaq::qview (C++            |
|  [cudaq::KrausTrajectory::builder |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     function)](ap                 | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq::qview::back (C++      |
| cudaq15KrausTrajectory7builderEv) |     function)                     |
| -   [cu                           | ](api/languages/cpp_api.html#_CPP |
| daq::KrausTrajectory::countErrors | v4N5cudaq5qview4backENSt6size_tE) |
|     (C++                          | -   [cudaq::qview::begin (C++     |
|     function)](api/lang           |                                   |
| uages/cpp_api.html#_CPPv4NK5cudaq | function)](api/languages/cpp_api. |
| 15KrausTrajectory11countErrorsEv) | html#_CPPv4N5cudaq5qview5beginEv) |
| -   [                             | -   [cudaq::qview::end (C++       |
| cudaq::KrausTrajectory::isOrdered |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)](api/l              | i.html#_CPPv4N5cudaq5qview3endEv) |
| anguages/cpp_api.html#_CPPv4NK5cu | -   [cudaq::qview::front (C++     |
| daq15KrausTrajectory9isOrderedEv) |     function)](                   |
| -   [cudaq::                      | api/languages/cpp_api.html#_CPPv4 |
| KrausTrajectory::kraus_selections | N5cudaq5qview5frontENSt6size_tE), |
|     (C++                          |                                   |
|     member)](api/languag          |    [\[1\]](api/languages/cpp_api. |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | html#_CPPv4N5cudaq5qview5frontEv) |
| ausTrajectory16kraus_selectionsE) | -   [cudaq::qview::operator\[\]   |
| -   [cudaq:                       |     (C++                          |
| :KrausTrajectory::KrausTrajectory |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function                      | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qview::qview (C++     |
| Pv4N5cudaq15KrausTrajectory15Krau |     functio                       |
| sTrajectoryENSt6size_tENSt6vector | n)](api/languages/cpp_api.html#_C |
| I14KrausSelectionEEdNSt6size_tE), | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     [\[1\]](api/languag           |     [\[1                          |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | \]](api/languages/cpp_api.html#_C |
| ausTrajectory15KrausTrajectoryEv) | PPv4N5cudaq5qview5qviewERK5qview) |
| -   [cudaq::Kr                    | -   [cudaq::qview::size (C++      |
| ausTrajectory::measurement_counts |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     member)](api/languages        | html#_CPPv4NK5cudaq5qview4sizeEv) |
| /cpp_api.html#_CPPv4N5cudaq15Krau | -   [cudaq::qview::slice (C++     |
| sTrajectory18measurement_countsE) |     function)](api/langua         |
| -   [cud                          | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| aq::KrausTrajectory::multiplicity | iew5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qview::value_type     |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     t                             |
| 15KrausTrajectory12multiplicityE) | ype)](api/languages/cpp_api.html# |
| -   [                             | _CPPv4N5cudaq5qview10value_typeE) |
| cudaq::KrausTrajectory::num_shots | -   [cudaq::range (C++            |
|     (C++                          |     fun                           |
|     member)](api                  | ction)](api/languages/cpp_api.htm |
| /languages/cpp_api.html#_CPPv4N5c | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| udaq15KrausTrajectory9num_shotsE) | orI11ElementTypeEE11ElementType), |
| -   [c                            |     [\[1\]](api/languages/cpp_    |
| udaq::KrausTrajectory::operator== | api.html#_CPPv4I0EN5cudaq5rangeEN |
|     (C++                          | St6vectorI11ElementTypeEE11Elemen |
|     function)](api/languages/c    | tType11ElementType11ElementType), |
| pp_api.html#_CPPv4NK5cudaq15Kraus |     [                             |
| TrajectoryeqERK15KrausTrajectory) | \[2\]](api/languages/cpp_api.html |
| -   [cu                           | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| daq::KrausTrajectory::probability | -   [cudaq::real (C++             |
|     (C++                          |     type)](api/languages/         |
|     member)](api/la               | cpp_api.html#_CPPv4N5cudaq4realE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::registry (C++         |
| q15KrausTrajectory11probabilityE) |     type)](api/languages/cpp_     |
| -   [cuda                         | api.html#_CPPv4N5cudaq8registryE) |
| q::KrausTrajectory::trajectory_id | -                                 |
|     (C++                          |  [cudaq::registry::RegisteredType |
|     member)](api/lang             |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     class)](api/                  |
| 5KrausTrajectory13trajectory_idE) | languages/cpp_api.html#_CPPv4I0EN |
| -                                 | 5cudaq8registry14RegisteredTypeE) |
|   [cudaq::KrausTrajectory::weight | -   [cudaq::RemoteRESTQPU (C++    |
|     (C++                          |                                   |
|     member)](                     |  class)](api/languages/cpp_api.ht |
| api/languages/cpp_api.html#_CPPv4 | ml#_CPPv4N5cudaq13RemoteRESTQPUE) |
| N5cudaq15KrausTrajectory6weightE) | -   [cudaq::Resources (C++        |
| -                                 |     class)](api/languages/cpp_a   |
|    [cudaq::KrausTrajectoryBuilder | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     (C++                          | -   [cudaq::run (C++              |
|     class)](                      |     function)]                    |
| api/languages/cpp_api.html#_CPPv4 | (api/languages/cpp_api.html#_CPPv |
| N5cudaq22KrausTrajectoryBuilderE) | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| -   [cud                          | 5invoke_result_tINSt7decay_tI13Qu |
| aq::KrausTrajectoryBuilder::build | antumKernelEEDpNSt7decay_tI4ARGSE |
|     (C++                          | EEEEENSt6size_tERN5cudaq11noise_m |
|     function)](api/lang           | odelERR13QuantumKernelDpRR4ARGS), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[1\]](api/langu             |
| 22KrausTrajectoryBuilder5buildEv) | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| -   [cud                          | daq3runENSt6vectorINSt15invoke_re |
| aq::KrausTrajectoryBuilder::setId | sult_tINSt7decay_tI13QuantumKerne |
|     (C++                          | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
|     function)](api/languages/cpp  | ize_tERR13QuantumKernelDpRR4ARGS) |
| _api.html#_CPPv4N5cudaq22KrausTra | -   [cudaq::run_async (C++        |
| jectoryBuilder5setIdENSt6size_tE) |     functio                       |
| -   [cudaq::Kraus                 | n)](api/languages/cpp_api.html#_C |
| TrajectoryBuilder::setProbability | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     (C++                          | tureINSt6vectorINSt15invoke_resul |
|     function)](api/languages/cpp  | t_tINSt7decay_tI13QuantumKernelEE |
| _api.html#_CPPv4N5cudaq22KrausTra | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| jectoryBuilder14setProbabilityEd) | ze_tENSt6size_tERN5cudaq11noise_m |
| -   [cudaq::Krau                  | odelERR13QuantumKernelDpRR4ARGS), |
| sTrajectoryBuilder::setSelections |     [\[1\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4I0DpEN |
|     function)](api/languag        | 5cudaq9run_asyncENSt6futureINSt6v |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | ectorINSt15invoke_result_tINSt7de |
| ausTrajectoryBuilder13setSelectio | cay_tI13QuantumKernelEEDpNSt7deca |
| nsENSt6vectorI14KrausSelectionEE) | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| -   [cudaq::logical_observable    | ize_tERR13QuantumKernelDpRR4ARGS) |
|     (C++                          | -   [cudaq::RuntimeTarget (C++    |
|     function)](api/languages/c    |                                   |
| pp_api.html#_CPPv4IDpEN5cudaq18lo | struct)](api/languages/cpp_api.ht |
| gical_observableEvDpRR8MeasArgs), | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     [\[1\]](api/l                 | -   [cudaq::sample (C++           |
| anguages/cpp_api.html#_CPPv4N5cud |     function)](api/languages/c    |
| aq18logical_observableERKNSt6vect | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| orI14measure_resultEENSt6size_tE) | mpleE13sample_resultRK14sample_op |
| -   [cudaq::M2DSparseMatrix (C++  | tionsRR13QuantumKernelDpRR4Args), |
|     st                            |     [\[1\                         |
| ruct)](api/languages/cpp_api.html | ]](api/languages/cpp_api.html#_CP |
| #_CPPv4N5cudaq15M2DSparseMatrixE) | Pv4I0DpEN5cudaq6sampleE13sample_r |
| -   [cudaq::M2OSparseMatrix (C++  | esultRR13QuantumKernelDpRR4Args), |
|     st                            |     [\                            |
| ruct)](api/languages/cpp_api.html | [2\]](api/languages/cpp_api.html# |
| #_CPPv4N5cudaq15M2OSparseMatrixE) | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| -   [cudaq::matrix_callback (C++  | ize_tERR13QuantumKernelDpRR4Args) |
|     c                             | -   [cudaq::sample_options (C++   |
| lass)](api/languages/cpp_api.html |     s                             |
| #_CPPv4N5cudaq15matrix_callbackE) | truct)](api/languages/cpp_api.htm |
| -   [cudaq::matrix_handler (C++   | l#_CPPv4N5cudaq14sample_optionsE) |
|                                   | -   [cudaq::sample_result (C++    |
| class)](api/languages/cpp_api.htm |                                   |
| l#_CPPv4N5cudaq14matrix_handlerE) |  class)](api/languages/cpp_api.ht |
| -   [cudaq::mat                   | ml#_CPPv4N5cudaq13sample_resultE) |
| rix_handler::commutation_behavior | -   [cudaq::sample_result::append |
|     (C++                          |     (C++                          |
|     struct)](api/languages/       |     function)](api/languages/cpp_ |
| cpp_api.html#_CPPv4N5cudaq14matri | api.html#_CPPv4N5cudaq13sample_re |
| x_handler20commutation_behaviorE) | sult6appendERK15ExecutionResultb) |
| -                                 | -   [cudaq::sample_result::begin  |
|    [cudaq::matrix_handler::define |     (C++                          |
|     (C++                          |     function)]                    |
|     function)](a                  | (api/languages/cpp_api.html#_CPPv |
| pi/languages/cpp_api.html#_CPPv4N | 4N5cudaq13sample_result5beginEv), |
| 5cudaq14matrix_handler6defineENSt |     [\[1\]]                       |
| 6stringENSt6vectorINSt7int64_tEEE | (api/languages/cpp_api.html#_CPPv |
| RR15matrix_callbackRKNSt13unorder | 4NK5cudaq13sample_result5beginEv) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::sample_result::cbegin |
|                                   |     (C++                          |
| [\[1\]](api/languages/cpp_api.htm |     function)](                   |
| l#_CPPv4N5cudaq14matrix_handler6d | api/languages/cpp_api.html#_CPPv4 |
| efineENSt6stringENSt6vectorINSt7i | NK5cudaq13sample_result6cbeginEv) |
| nt64_tEEERR15matrix_callbackRR20d | -   [cudaq::sample_result::cend   |
| iag_matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     function)                     |
|     [\[2\]](                      | ](api/languages/cpp_api.html#_CPP |
| api/languages/cpp_api.html#_CPPv4 | v4NK5cudaq13sample_result4cendEv) |
| N5cudaq14matrix_handler6defineENS | -   [cudaq::sample_result::clear  |
| t6stringENSt6vectorINSt7int64_tEE |     (C++                          |
| ERR15matrix_callbackRRNSt13unorde |     function)                     |
| red_mapINSt6stringENSt6stringEEE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq13sample_result5clearEv) |
|   [cudaq::matrix_handler::degrees | -   [cudaq::sample_result::count  |
|     (C++                          |     (C++                          |
|     function)](ap                 |     function)](                   |
| i/languages/cpp_api.html#_CPPv4NK | api/languages/cpp_api.html#_CPPv4 |
| 5cudaq14matrix_handler7degreesEv) | NK5cudaq13sample_result5countENSt |
| -                                 | 11string_viewEKNSt11string_viewE) |
|  [cudaq::matrix_handler::displace | -   [                             |
|     (C++                          | cudaq::sample_result::deserialize |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     functio                       |
| rix_handler8displaceENSt6size_tE) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::matrix                | PPv4N5cudaq13sample_result11deser |
| _handler::get_expected_dimensions | ializeERNSt6vectorINSt6size_tEEE) |
|     (C++                          | -   [cudaq::sample_result::dump   |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](api/languag        |
| pi.html#_CPPv4NK5cudaq14matrix_ha | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ndler23get_expected_dimensionsEv) | ample_result4dumpERNSt7ostreamE), |
| -   [cudaq::matrix_ha             |     [\[1\]                        |
| ndler::get_parameter_descriptions | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq13sample_result4dumpEv) |
|                                   | -   [cudaq::sample_result::end    |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4NK5cudaq14matrix_handl |     function                      |
| er26get_parameter_descriptionsEv) | )](api/languages/cpp_api.html#_CP |
| -   [c                            | Pv4N5cudaq13sample_result3endEv), |
| udaq::matrix_handler::instantiate |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](a                  | Pv4NK5cudaq13sample_result3endEv) |
| pi/languages/cpp_api.html#_CPPv4N | -   [                             |
| 5cudaq14matrix_handler11instantia | cudaq::sample_result::expectation |
| teENSt6stringERKNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     f                             |
|     [\[1\]](                      | unction)](api/languages/cpp_api.h |
| api/languages/cpp_api.html#_CPPv4 | tml#_CPPv4NK5cudaq13sample_result |
| N5cudaq14matrix_handler11instanti | 11expectationEKNSt11string_viewE) |
| ateENSt6stringERRNSt6vectorINSt6s | -   [cuda                         |
| ize_tEEERK20commutation_behavior) | q::sample_result::get_annotations |
| -   [cuda                         |     (C++                          |
| q::matrix_handler::matrix_handler |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq13 |
|     function)](api/languag        | sample_result15get_annotationsEv) |
| es/cpp_api.html#_CPPv4I0_NSt11ena | -   [c                            |
| ble_if_tINSt12is_base_of_vI16oper | udaq::sample_result::get_marginal |
| ator_handler1TEEbEEEN5cudaq14matr |     (C++                          |
| ix_handler14matrix_handlerERK1T), |     function)](api/languages/cpp_ |
|     [\[1\]](ap                    | api.html#_CPPv4NK5cudaq13sample_r |
| i/languages/cpp_api.html#_CPPv4I0 | esult12get_marginalERKNSt6vectorI |
| _NSt11enable_if_tINSt12is_base_of | NSt6size_tEEEKNSt11string_viewE), |
| _vI16operator_handler1TEEbEEEN5cu |     [\[1\]](api/languages/cpp_    |
| daq14matrix_handler14matrix_handl | api.html#_CPPv4NK5cudaq13sample_r |
| erERK1TRK20commutation_behavior), | esult12get_marginalERRKNSt6vector |
|     [\[2\]](api/languages/cpp_ap  | INSt6size_tEEEKNSt11string_viewE) |
| i.html#_CPPv4N5cudaq14matrix_hand | -   [cuda                         |
| ler14matrix_handlerENSt6size_tE), | q::sample_result::get_total_shots |
|     [\[3\]](api/                  |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/langua         |
| daq14matrix_handler14matrix_handl | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| erENSt6stringERKNSt6vectorINSt6si | sample_result15get_total_shotsEv) |
| ze_tEEERK20commutation_behavior), | -   [cuda                         |
|     [\[4\]](api/                  | q::sample_result::has_even_parity |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     fun                           |
| erENSt6stringERRNSt6vectorINSt6si | ction)](api/languages/cpp_api.htm |
| ze_tEEERK20commutation_behavior), | l#_CPPv4N5cudaq13sample_result15h |
|     [\                            | as_even_parityENSt11string_viewE) |
| [5\]](api/languages/cpp_api.html# | -   [cuda                         |
| _CPPv4N5cudaq14matrix_handler14ma | q::sample_result::has_expectation |
| trix_handlerERK14matrix_handler), |     (C++                          |
|     [                             |     funct                         |
| \[6\]](api/languages/cpp_api.html | ion)](api/languages/cpp_api.html# |
| #_CPPv4N5cudaq14matrix_handler14m | _CPPv4NK5cudaq13sample_result15ha |
| atrix_handlerERR14matrix_handler) | s_expectationEKNSt11string_viewE) |
| -                                 | -   [cu                           |
|  [cudaq::matrix_handler::momentum | daq::sample_result::most_probable |
|     (C++                          |     (C++                          |
|     function)](api/language       |     fun                           |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ction)](api/languages/cpp_api.htm |
| rix_handler8momentumENSt6size_tE) | l#_CPPv4NK5cudaq13sample_result13 |
| -                                 | most_probableEKNSt11string_viewE) |
|    [cudaq::matrix_handler::number | -                                 |
|     (C++                          | [cudaq::sample_result::operator+= |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     function)](api/langua         |
| atrix_handler6numberENSt6size_tE) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -                                 | ample_resultpLERK13sample_result) |
| [cudaq::matrix_handler::operator= | -                                 |
|     (C++                          |  [cudaq::sample_result::operator= |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     function)](api/langua         |
| l#_CPPv4I0_NSt11enable_if_tIXaant | ges/cpp_api.html#_CPPv4N5cudaq13s |
| NSt7is_sameI1T14matrix_handlerE5v | ample_resultaSERR13sample_result) |
| alueENSt12is_base_of_vI16operator | -                                 |
| _handler1TEEEbEEEN5cudaq14matrix_ | [cudaq::sample_result::operator== |
| handleraSER14matrix_handlerRK1T), |     (C++                          |
|     [\[1\]](api/languages         |     function)](api/languag        |
| /cpp_api.html#_CPPv4N5cudaq14matr | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ix_handleraSERK14matrix_handler), | ample_resulteqERK13sample_result) |
|     [\[2\]](api/language          | -   [                             |
| s/cpp_api.html#_CPPv4N5cudaq14mat | cudaq::sample_result::probability |
| rix_handleraSERR14matrix_handler) |     (C++                          |
| -   [                             |     function)](api/lan            |
| cudaq::matrix_handler::operator== | guages/cpp_api.html#_CPPv4NK5cuda |
|     (C++                          | q13sample_result11probabilityENSt |
|     function)](api/languages      | 11string_viewEKNSt11string_viewE) |
| /cpp_api.html#_CPPv4NK5cudaq14mat | -   [cud                          |
| rix_handlereqERK14matrix_handler) | aq::sample_result::register_names |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::parity |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4NK5cudaq1 |
|     function)](api/langua         | 3sample_result14register_namesEv) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -                                 |
| atrix_handler6parityENSt6size_tE) |    [cudaq::sample_result::reorder |
| -                                 |     (C++                          |
|  [cudaq::matrix_handler::position |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function)](api/language       | ample_result7reorderERKNSt6vector |
| s/cpp_api.html#_CPPv4N5cudaq14mat | INSt6size_tEEEKNSt11string_viewE) |
| rix_handler8positionENSt6size_tE) | -   [cu                           |
| -   [cudaq::                      | daq::sample_result::sample_result |
| matrix_handler::remove_definition |     (C++                          |
|     (C++                          |     function)](api/               |
|     fu                            | languages/cpp_api.html#_CPPv4N5cu |
| nction)](api/languages/cpp_api.ht | daq13sample_result13sample_result |
| ml#_CPPv4N5cudaq14matrix_handler1 | E16CountsDictionary10cudaq_json), |
| 7remove_definitionERKNSt6stringE) |     [                             |
| -                                 | \[1\]](api/languages/cpp_api.html |
|   [cudaq::matrix_handler::squeeze | #_CPPv4N5cudaq13sample_result13sa |
|     (C++                          | mple_resultERK15ExecutionResult), |
|     function)](api/languag        |     [\[2\]](api/la                |
| es/cpp_api.html#_CPPv4N5cudaq14ma | nguages/cpp_api.html#_CPPv4N5cuda |
| trix_handler7squeezeENSt6size_tE) | q13sample_result13sample_resultER |
| -   [cudaq::m                     | KNSt6vectorI15ExecutionResultEE), |
| atrix_handler::to_diagonal_matrix |                                   |
|     (C++                          |  [\[3\]](api/languages/cpp_api.ht |
|     function)](api/lang           | ml#_CPPv4N5cudaq13sample_result13 |
| uages/cpp_api.html#_CPPv4NK5cudaq | sample_resultERR13sample_result), |
| 14matrix_handler18to_diagonal_mat |     [                             |
| rixERNSt13unordered_mapINSt6size_ | \[4\]](api/languages/cpp_api.html |
| tENSt7int64_tEEERKNSt13unordered_ | #_CPPv4N5cudaq13sample_result13sa |
| mapINSt6stringENSt7complexIdEEEE) | mple_resultERR15ExecutionResult), |
| -                                 |     [\[5\]](api/lan               |
| [cudaq::matrix_handler::to_matrix | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 13sample_result13sample_resultEdR |
|     function)                     | KNSt6vectorI15ExecutionResultEE), |
| ](api/languages/cpp_api.html#_CPP |     [\[6\]](api/lan               |
| v4NK5cudaq14matrix_handler9to_mat | guages/cpp_api.html#_CPPv4N5cudaq |
| rixERNSt13unordered_mapINSt6size_ | 13sample_result13sample_resultEv) |
| tENSt7int64_tEEERKNSt13unordered_ | -                                 |
| mapINSt6stringENSt7complexIdEEEE) |  [cudaq::sample_result::serialize |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::to_string |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4NK5 |
|     function)](api/               | cudaq13sample_result9serializeEv) |
| languages/cpp_api.html#_CPPv4NK5c | -   [cudaq::sample_result::size   |
| udaq14matrix_handler9to_stringEb) |     (C++                          |
| -                                 |     function)](api/languages/c    |
| [cudaq::matrix_handler::unique_id | pp_api.html#_CPPv4NK5cudaq13sampl |
|     (C++                          | e_result4sizeEKNSt11string_viewE) |
|     function)](api/               | -   [cudaq::sample_result::to_map |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14matrix_handler9unique_idEv) |     function)](api/languages/cpp  |
| -   [cudaq:                       | _api.html#_CPPv4NK5cudaq13sample_ |
| :matrix_handler::\~matrix_handler | result6to_mapEKNSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|     functi                        | q::sample_result::\~sample_result |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     funct                         |
| -   [cudaq::matrix_op (C++        | ion)](api/languages/cpp_api.html# |
|     type)](api/languages/cpp_a    | _CPPv4N5cudaq13sample_resultD0Ev) |
| pi.html#_CPPv4N5cudaq9matrix_opE) | -   [cudaq::scalar_callback (C++  |
| -   [cudaq::matrix_op_term (C++   |     c                             |
|                                   | lass)](api/languages/cpp_api.html |
|  type)](api/languages/cpp_api.htm | #_CPPv4N5cudaq15scalar_callbackE) |
| l#_CPPv4N5cudaq14matrix_op_termE) | -   [c                            |
| -                                 | udaq::scalar_callback::operator() |
|    [cudaq::mdiag_operator_handler |     (C++                          |
|     (C++                          |     function)](api/language       |
|     class)](                      | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| api/languages/cpp_api.html#_CPPv4 | alar_callbackclERKNSt13unordered_ |
| N5cudaq22mdiag_operator_handlerE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::measure_handle (C++   | -   [                             |
|                                   | cudaq::scalar_callback::operator= |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14measure_handleE) |     function)](api/languages/c    |
| -   [cudaq::measure_result (C++   | pp_api.html#_CPPv4N5cudaq15scalar |
|                                   | _callbackaSERK15scalar_callback), |
|  type)](api/languages/cpp_api.htm |     [\[1\]](api/languages/        |
| l#_CPPv4N5cudaq14measure_resultE) | cpp_api.html#_CPPv4N5cudaq15scala |
| -   [cudaq::mpi (C++              | r_callbackaSERR15scalar_callback) |
|     type)](api/languages          | -   [cudaq:                       |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | :scalar_callback::scalar_callback |
| -   [cudaq::mpi::all_gather (C++  |     (C++                          |
|     fu                            |     function)](api/languag        |
| nction)](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4I0_NSt11ena |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | ble_if_tINSt16is_invocable_r_vINS |
| RNSt6vectorIdEERKNSt6vectorIdEE), | t7complexIdEE8CallableRKNSt13unor |
|                                   | dered_mapINSt6stringENSt7complexI |
|   [\[1\]](api/languages/cpp_api.h | dEEEEEEbEEEN5cudaq15scalar_callba |
| tml#_CPPv4N5cudaq3mpi10all_gather | ck15scalar_callbackERR8Callable), |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     [\[1\                         |
| -   [cudaq::mpi::all_reduce (C++  | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_callback15scal |
|  function)](api/languages/cpp_api | ar_callbackERK15scalar_callback), |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     [\[2                          |
| reduceE1TRK1TRK14BinaryFunction), | \]](api/languages/cpp_api.html#_C |
|     [\[1\]](api/langu             | PPv4N5cudaq15scalar_callback15sca |
| ages/cpp_api.html#_CPPv4I00EN5cud | lar_callbackERR15scalar_callback) |
| aq3mpi10all_reduceE1TRK1TRK4Func) | -   [cudaq::scalar_operator (C++  |
| -   [cudaq::mpi::broadcast (C++   |     c                             |
|     function)](api/               | lass)](api/languages/cpp_api.html |
| languages/cpp_api.html#_CPPv4N5cu | #_CPPv4N5cudaq15scalar_operatorE) |
| daq3mpi9broadcastERNSt6stringEi), | -                                 |
|     [\[1\]](api/la                | [cudaq::scalar_operator::evaluate |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q3mpi9broadcastERNSt6vectorIdEEi) |                                   |
| -   [cudaq::mpi::finalize (C++    |    function)](api/languages/cpp_a |
|     f                             | pi.html#_CPPv4NK5cudaq15scalar_op |
| unction)](api/languages/cpp_api.h | erator8evaluateERKNSt13unordered_ |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::mpi::initialize (C++  | -   [cudaq::scalar_ope            |
|     function                      | rator::get_parameter_descriptions |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq3mpi10initializeEiPPc), |     f                             |
|     [                             | unction)](api/languages/cpp_api.h |
| \[1\]](api/languages/cpp_api.html | tml#_CPPv4NK5cudaq15scalar_operat |
| #_CPPv4N5cudaq3mpi10initializeEv) | or26get_parameter_descriptionsEv) |
| -   [cudaq::mpi::is_initialized   | -   [cu                           |
|     (C++                          | daq::scalar_operator::is_constant |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/lang           |
| Pv4N5cudaq3mpi14is_initializedEv) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cudaq::mpi::num_ranks (C++   | 15scalar_operator11is_constantEv) |
|     fu                            | -   [c                            |
| nction)](api/languages/cpp_api.ht | udaq::scalar_operator::operator\* |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     (C++                          |
| -   [cudaq::mpi::rank (C++        |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|    function)](api/languages/cpp_a | Pv4N5cudaq15scalar_operatormlENSt |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::noise_model (C++      |     [\[1\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
|    class)](api/languages/cpp_api. | Pv4N5cudaq15scalar_operatormlENSt |
| html#_CPPv4N5cudaq11noise_modelE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::n                     |     [\[2\]](api/languages/cp      |
| oise_model::add_all_qubit_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormlEdRK15scalar_operator), |
|     function)](api                |     [\[3\]](api/languages/cp      |
| /languages/cpp_api.html#_CPPv4IDp | p_api.html#_CPPv4N5cudaq15scalar_ |
| EN5cudaq11noise_model21add_all_qu | operatormlEdRR15scalar_operator), |
| bit_channelEvRK13kraus_channeli), |     [\[4\]](api/languages         |
|     [\[1\]](api/langua            | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ges/cpp_api.html#_CPPv4N5cudaq11n | alar_operatormlENSt7complexIdEE), |
| oise_model21add_all_qubit_channel |     [\[5\]](api/languages/cpp     |
| ERKNSt6stringERK13kraus_channeli) | _api.html#_CPPv4NKR5cudaq15scalar |
| -                                 | _operatormlERK15scalar_operator), |
|  [cudaq::noise_model::add_channel |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     funct                         | 4NKR5cudaq15scalar_operatormlEd), |
| ion)](api/languages/cpp_api.html# |     [\[7\]](api/language          |
| _CPPv4IDpEN5cudaq11noise_model11a | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| dd_channelEvRK15PredicateFuncTy), | alar_operatormlENSt7complexIdEE), |
|     [\[1\]](api/languages/cpp_    |     [\[8\]](api/languages/cp      |
| api.html#_CPPv4IDpEN5cudaq11noise | p_api.html#_CPPv4NO5cudaq15scalar |
| _model11add_channelEvRKNSt6vector | _operatormlERK15scalar_operator), |
| INSt6size_tEEERK13kraus_channel), |     [\[9\                         |
|     [\[2\]](ap                    | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4NO5cudaq15scalar_operatormlEd) |
| cudaq11noise_model11add_channelER | -   [cu                           |
| KNSt6stringERK15PredicateFuncTy), | daq::scalar_operator::operator\*= |
|                                   |     (C++                          |
| [\[3\]](api/languages/cpp_api.htm |     function)](api/languag        |
| l#_CPPv4N5cudaq11noise_model11add | es/cpp_api.html#_CPPv4N5cudaq15sc |
| _channelERKNSt6stringERKNSt6vecto | alar_operatormLENSt7complexIdEE), |
| rINSt6size_tEEERK13kraus_channel) |     [\[1\]](api/languages/c       |
| -   [cudaq::noise_model::empty    | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatormLERK15scalar_operator), |
|     function                      |     [\[2                          |
| )](api/languages/cpp_api.html#_CP | \]](api/languages/cpp_api.html#_C |
| Pv4NK5cudaq11noise_model5emptyEv) | PPv4N5cudaq15scalar_operatormLEd) |
| -                                 | -   [                             |
| [cudaq::noise_model::get_channels | cudaq::scalar_operator::operator+ |
|     (C++                          |     (C++                          |
|     function)](api/l              |     function                      |
| anguages/cpp_api.html#_CPPv4I0ENK | )](api/languages/cpp_api.html#_CP |
| 5cudaq11noise_model12get_channels | Pv4N5cudaq15scalar_operatorplENSt |
| ENSt6vectorI13kraus_channelEERKNS | 7complexIdEERK15scalar_operator), |
| t6vectorINSt6size_tEEERKNSt6vecto |     [\[1\                         |
| rINSt6size_tEEERKNSt6vectorIdEE), | ]](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/languages/cpp_a   | Pv4N5cudaq15scalar_operatorplENSt |
| pi.html#_CPPv4NK5cudaq11noise_mod | 7complexIdEERR15scalar_operator), |
| el12get_channelsERKNSt6stringERKN |     [\[2\]](api/languages/cp      |
| St6vectorINSt6size_tEEERKNSt6vect | p_api.html#_CPPv4N5cudaq15scalar_ |
| orINSt6size_tEEERKNSt6vectorIdEE) | operatorplEdRK15scalar_operator), |
| -                                 |     [\[3\]](api/languages/cp      |
|  [cudaq::noise_model::noise_model | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatorplEdRR15scalar_operator), |
|     function)](api                |     [\[4\]](api/languages         |
| /languages/cpp_api.html#_CPPv4N5c | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| udaq11noise_model11noise_modelEv) | alar_operatorplENSt7complexIdEE), |
| -   [cu                           |     [\[5\]](api/languages/cpp     |
| daq::noise_model::PredicateFuncTy | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatorplERK15scalar_operator), |
|     type)](api/la                 |     [\[6\]]                       |
| nguages/cpp_api.html#_CPPv4N5cuda | (api/languages/cpp_api.html#_CPPv |
| q11noise_model15PredicateFuncTyE) | 4NKR5cudaq15scalar_operatorplEd), |
| -   [cud                          |     [\[7\]]                       |
| aq::noise_model::register_channel | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatorplEv), |
|     function)](api/languages      |     [\[8\]](api/language          |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| noise_model16register_channelEvv) | alar_operatorplENSt7complexIdEE), |
| -   [cudaq::                      |     [\[9\]](api/languages/cp      |
| noise_model::requires_constructor | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatorplERK15scalar_operator), |
|     type)](api/languages/cp       |     [\[10\]                       |
| p_api.html#_CPPv4I0DpEN5cudaq11no | ](api/languages/cpp_api.html#_CPP |
| ise_model20requires_constructorE) | v4NO5cudaq15scalar_operatorplEd), |
| -   [cudaq::noise_model_type (C++ |     [\[11\                        |
|     e                             | ]](api/languages/cpp_api.html#_CP |
| num)](api/languages/cpp_api.html# | Pv4NO5cudaq15scalar_operatorplEv) |
| _CPPv4N5cudaq16noise_model_typeE) | -   [c                            |
| -   [cudaq::no                    | udaq::scalar_operator::operator+= |
| ise_model_type::amplitude_damping |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](api/languages    | es/cpp_api.html#_CPPv4N5cudaq15sc |
| /cpp_api.html#_CPPv4N5cudaq16nois | alar_operatorpLENSt7complexIdEE), |
| e_model_type17amplitude_dampingE) |     [\[1\]](api/languages/c       |
| -   [cudaq::noise_mode            | pp_api.html#_CPPv4N5cudaq15scalar |
| l_type::amplitude_damping_channel | _operatorpLERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     e                             | \]](api/languages/cpp_api.html#_C |
| numerator)](api/languages/cpp_api | PPv4N5cudaq15scalar_operatorpLEd) |
| .html#_CPPv4N5cudaq16noise_model_ | -   [                             |
| type25amplitude_damping_channelE) | cudaq::scalar_operator::operator- |
| -   [cudaq::n                     |     (C++                          |
| oise_model_type::bit_flip_channel |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     enumerator)](api/language     | Pv4N5cudaq15scalar_operatormiENSt |
| s/cpp_api.html#_CPPv4N5cudaq16noi | 7complexIdEERK15scalar_operator), |
| se_model_type16bit_flip_channelE) |     [\[1\                         |
| -   [cudaq::                      | ]](api/languages/cpp_api.html#_CP |
| noise_model_type::depolarization1 | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     enumerator)](api/languag      |     [\[2\]](api/languages/cp      |
| es/cpp_api.html#_CPPv4N5cudaq16no | p_api.html#_CPPv4N5cudaq15scalar_ |
| ise_model_type15depolarization1E) | operatormiEdRK15scalar_operator), |
| -   [cudaq::                      |     [\[3\]](api/languages/cp      |
| noise_model_type::depolarization2 | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRR15scalar_operator), |
|     enumerator)](api/languag      |     [\[4\]](api/languages         |
| es/cpp_api.html#_CPPv4N5cudaq16no | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ise_model_type15depolarization2E) | alar_operatormiENSt7complexIdEE), |
| -   [cudaq::noise_m               |     [\[5\]](api/languages/cpp     |
| odel_type::depolarization_channel | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatormiERK15scalar_operator), |
|                                   |     [\[6\]]                       |
|   enumerator)](api/languages/cpp_ | (api/languages/cpp_api.html#_CPPv |
| api.html#_CPPv4N5cudaq16noise_mod | 4NKR5cudaq15scalar_operatormiEd), |
| el_type22depolarization_channelE) |     [\[7\]]                       |
| -                                 | (api/languages/cpp_api.html#_CPPv |
|  [cudaq::noise_model_type::pauli1 | 4NKR5cudaq15scalar_operatormiEv), |
|     (C++                          |     [\[8\]](api/language          |
|     enumerator)](a                | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| pi/languages/cpp_api.html#_CPPv4N | alar_operatormiENSt7complexIdEE), |
| 5cudaq16noise_model_type6pauli1E) |     [\[9\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4NO5cudaq15scalar |
|  [cudaq::noise_model_type::pauli2 | _operatormiERK15scalar_operator), |
|     (C++                          |     [\[10\]                       |
|     enumerator)](a                | ](api/languages/cpp_api.html#_CPP |
| pi/languages/cpp_api.html#_CPPv4N | v4NO5cudaq15scalar_operatormiEd), |
| 5cudaq16noise_model_type6pauli2E) |     [\[11\                        |
| -   [cudaq                        | ]](api/languages/cpp_api.html#_CP |
| ::noise_model_type::phase_damping | Pv4NO5cudaq15scalar_operatormiEv) |
|     (C++                          | -   [c                            |
|     enumerator)](api/langu        | udaq::scalar_operator::operator-= |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| noise_model_type13phase_dampingE) |     function)](api/languag        |
| -   [cudaq::noi                   | es/cpp_api.html#_CPPv4N5cudaq15sc |
| se_model_type::phase_flip_channel | alar_operatormIENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     enumerator)](api/languages/   | pp_api.html#_CPPv4N5cudaq15scalar |
| cpp_api.html#_CPPv4N5cudaq16noise | _operatormIERK15scalar_operator), |
| _model_type18phase_flip_channelE) |     [\[2                          |
| -                                 | \]](api/languages/cpp_api.html#_C |
| [cudaq::noise_model_type::unknown | PPv4N5cudaq15scalar_operatormIEd) |
|     (C++                          | -   [                             |
|     enumerator)](ap               | cudaq::scalar_operator::operator/ |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7unknownE) |     function                      |
| -                                 | )](api/languages/cpp_api.html#_CP |
| [cudaq::noise_model_type::x_error | Pv4N5cudaq15scalar_operatordvENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     enumerator)](ap               |     [\[1\                         |
| i/languages/cpp_api.html#_CPPv4N5 | ]](api/languages/cpp_api.html#_CP |
| cudaq16noise_model_type7x_errorE) | Pv4N5cudaq15scalar_operatordvENSt |
| -                                 | 7complexIdEERR15scalar_operator), |
| [cudaq::noise_model_type::y_error |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](ap               | operatordvEdRK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[3\]](api/languages/cp      |
| cudaq16noise_model_type7y_errorE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatordvEdRR15scalar_operator), |
| [cudaq::noise_model_type::z_error |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     enumerator)](ap               | alar_operatordvENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[5\]](api/languages/cpp     |
| cudaq16noise_model_type7z_errorE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::num_available_gpus    | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function                      | (api/languages/cpp_api.html#_CPPv |
| )](api/languages/cpp_api.html#_CP | 4NKR5cudaq15scalar_operatordvEd), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[7\]](api/language          |
| -   [cudaq::observe (C++          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     function)]                    | alar_operatordvENSt7complexIdEE), |
| (api/languages/cpp_api.html#_CPPv |     [\[8\]](api/languages/cp      |
| 4I00DpEN5cudaq7observeENSt6vector | p_api.html#_CPPv4NO5cudaq15scalar |
| I14observe_resultEERR13QuantumKer | _operatordvERK15scalar_operator), |
| nelRK15SpinOpContainerDpRR4Args), |     [\[9\                         |
|     [\[1\]](api/languages/cpp_ap  | ]](api/languages/cpp_api.html#_CP |
| i.html#_CPPv4I0DpEN5cudaq7observe | Pv4NO5cudaq15scalar_operatordvEd) |
| E14observe_resultNSt6size_tERR13Q | -   [c                            |
| uantumKernelRK7spin_opDpRR4Args), | udaq::scalar_operator::operator/= |
|     [\[                           |     (C++                          |
| 2\]](api/languages/cpp_api.html#_ |     function)](api/languag        |
| CPPv4I0DpEN5cudaq7observeE14obser | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ve_resultRK15observe_optionsRR13Q | alar_operatordVENSt7complexIdEE), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[1\]](api/languages/c       |
|     [\[3\]](api/lang              | pp_api.html#_CPPv4N5cudaq15scalar |
| uages/cpp_api.html#_CPPv4I0DpEN5c | _operatordVERK15scalar_operator), |
| udaq7observeE14observe_resultRR13 |     [\[2                          |
| QuantumKernelRK7spin_opDpRR4Args) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::observe_options (C++  | PPv4N5cudaq15scalar_operatordVEd) |
|     st                            | -   [                             |
| ruct)](api/languages/cpp_api.html | cudaq::scalar_operator::operator= |
| #_CPPv4N5cudaq15observe_optionsE) |     (C++                          |
| -   [cudaq::observe_result (C++   |     function)](api/languages/c    |
|                                   | pp_api.html#_CPPv4N5cudaq15scalar |
| class)](api/languages/cpp_api.htm | _operatoraSERK15scalar_operator), |
| l#_CPPv4N5cudaq14observe_resultE) |     [\[1\]](api/languages/        |
| -                                 | cpp_api.html#_CPPv4N5cudaq15scala |
|    [cudaq::observe_result::counts | r_operatoraSERR15scalar_operator) |
|     (C++                          | -   [c                            |
|     function)](api/languages/c    | udaq::scalar_operator::operator== |
| pp_api.html#_CPPv4N5cudaq14observ |     (C++                          |
| e_result6countsERK12spin_op_term) |     function)](api/languages/c    |
| -   [cudaq::observe_result::dump  | pp_api.html#_CPPv4NK5cudaq15scala |
|     (C++                          | r_operatoreqERK15scalar_operator) |
|     function)                     | -   [cudaq:                       |
| ](api/languages/cpp_api.html#_CPP | :scalar_operator::scalar_operator |
| v4N5cudaq14observe_result4dumpEv) |     (C++                          |
| -   [c                            |     func                          |
| udaq::observe_result::expectation | tion)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_operator15 |
|                                   | scalar_operatorENSt7complexIdEE), |
| function)](api/languages/cpp_api. |     [\[1\]](api/langu             |
| html#_CPPv4N5cudaq14observe_resul | ages/cpp_api.html#_CPPv4N5cudaq15 |
| t11expectationERK12spin_op_term), | scalar_operator15scalar_operatorE |
|     [\[1\]](api/la                | RK15scalar_callbackRRNSt13unorder |
| nguages/cpp_api.html#_CPPv4N5cuda | ed_mapINSt6stringENSt6stringEEE), |
| q14observe_result11expectationEv) |     [\[2\                         |
| -   [cuda                         | ]](api/languages/cpp_api.html#_CP |
| q::observe_result::id_coefficient | Pv4N5cudaq15scalar_operator15scal |
|     (C++                          | ar_operatorERK15scalar_operator), |
|     function)](api/langu          |     [\[3\]](api/langu             |
| ages/cpp_api.html#_CPPv4N5cudaq14 | ages/cpp_api.html#_CPPv4N5cudaq15 |
| observe_result14id_coefficientEv) | scalar_operator15scalar_operatorE |
| -   [cuda                         | RR15scalar_callbackRRNSt13unorder |
| q::observe_result::observe_result | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[4\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
|   function)](api/languages/cpp_ap | Pv4N5cudaq15scalar_operator15scal |
| i.html#_CPPv4N5cudaq14observe_res | ar_operatorERR15scalar_operator), |
| ult14observe_resultEdRK7spin_op), |     [\[5\]](api/language          |
|     [\[1\]](a                     | s/cpp_api.html#_CPPv4N5cudaq15sca |
| pi/languages/cpp_api.html#_CPPv4N | lar_operator15scalar_operatorEd), |
| 5cudaq14observe_result14observe_r |     [\[6\]](api/languag           |
| esultEdRK7spin_op13sample_result) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -                                 | alar_operator15scalar_operatorEv) |
|  [cudaq::observe_result::operator | -   [                             |
|     double (C++                   | cudaq::scalar_operator::to_matrix |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |                                   |
| PPv4N5cudaq14observe_resultcvdEv) |   function)](api/languages/cpp_ap |
| -                                 | i.html#_CPPv4NK5cudaq15scalar_ope |
|  [cudaq::observe_result::raw_data | rator9to_matrixERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|     function)](ap                 | -   [                             |
| i/languages/cpp_api.html#_CPPv4N5 | cudaq::scalar_operator::to_string |
| cudaq14observe_result8raw_dataEv) |     (C++                          |
| -   [cudaq::operator_handler (C++ |     function)](api/l              |
|     cl                            | anguages/cpp_api.html#_CPPv4NK5cu |
| ass)](api/languages/cpp_api.html# | daq15scalar_operator9to_stringEv) |
| _CPPv4N5cudaq16operator_handlerE) | -   [cudaq::s                     |
| -   [cudaq::optimizable_function  | calar_operator::\~scalar_operator |
|     (C++                          |     (C++                          |
|     class)                        |     functio                       |
| ](api/languages/cpp_api.html#_CPP | n)](api/languages/cpp_api.html#_C |
| v4N5cudaq20optimizable_functionE) | PPv4N5cudaq15scalar_operatorD0Ev) |
| -   [cudaq::optimization_result   | -   [cudaq::set_noise (C++        |
|     (C++                          |     function)](api/langu          |
|     type                          | ages/cpp_api.html#_CPPv4N5cudaq9s |
| )](api/languages/cpp_api.html#_CP | et_noiseERKN5cudaq11noise_modelE) |
| Pv4N5cudaq19optimization_resultE) | -   [cudaq::set_random_seed (C++  |
| -   [cudaq::optimizer (C++        |     function)](api/               |
|     class)](api/languages/cpp_a   | languages/cpp_api.html#_CPPv4N5cu |
| pi.html#_CPPv4N5cudaq9optimizerE) | daq15set_random_seedENSt6size_tE) |
| -   [cudaq::optimizer::optimize   | -   [cudaq::simulation_precision  |
|     (C++                          |     (C++                          |
|                                   |     enum)                         |
|  function)](api/languages/cpp_api | ](api/languages/cpp_api.html#_CPP |
| .html#_CPPv4N5cudaq9optimizer8opt | v4N5cudaq20simulation_precisionE) |
| imizeEKiRR20optimizable_function) | -   [                             |
| -   [cu                           | cudaq::simulation_precision::fp32 |
| daq::optimizer::requiresGradients |     (C++                          |
|     (C++                          |     enumerator)](api              |
|     function)](api/la             | /languages/cpp_api.html#_CPPv4N5c |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq20simulation_precision4fp32E) |
| q9optimizer17requiresGradientsEv) | -   [                             |
| -   [cudaq::orca (C++             | cudaq::simulation_precision::fp64 |
|     type)](api/languages/         |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     enumerator)](api              |
| -   [cudaq::orca::sample (C++     | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/languages/c    | udaq20simulation_precision4fp64E) |
| pp_api.html#_CPPv4N5cudaq4orca6sa | -   [cudaq::SimulationState (C++  |
| mpleERNSt6vectorINSt6size_tEEERNS |     c                             |
| t6vectorINSt6size_tEEERNSt6vector | lass)](api/languages/cpp_api.html |
| IdEERNSt6vectorIdEEiNSt6size_tE), | #_CPPv4N5cudaq15SimulationStateE) |
|     [\[1\]]                       | -   [                             |
| (api/languages/cpp_api.html#_CPPv | cudaq::SimulationState::precision |
| 4N5cudaq4orca6sampleERNSt6vectorI |     (C++                          |
| NSt6size_tEEERNSt6vectorINSt6size |     enum)](api                    |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::orca::sample_async    | udaq15SimulationState9precisionE) |
|     (C++                          | -   [cudaq:                       |
|                                   | :SimulationState::precision::fp32 |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq4orca12sample_a |     enumerator)](api/lang         |
| syncERNSt6vectorINSt6size_tEEERNS | uages/cpp_api.html#_CPPv4N5cudaq1 |
| t6vectorINSt6size_tEEERNSt6vector | 5SimulationState9precision4fp32E) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [cudaq:                       |
|     [\[1\]](api/la                | :SimulationState::precision::fp64 |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q4orca12sample_asyncERNSt6vectorI |     enumerator)](api/lang         |
| NSt6size_tEEERNSt6vectorINSt6size | uages/cpp_api.html#_CPPv4N5cudaq1 |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | 5SimulationState9precision4fp64E) |
| -   [cudaq::OrcaRemoteRESTQPU     | -                                 |
|     (C++                          |   [cudaq::SimulationState::Tensor |
|     cla                           |     (C++                          |
| ss)](api/languages/cpp_api.html#_ |     struct)](                     |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::other_policies (C++   | N5cudaq15SimulationState6TensorE) |
|     s                             | -   [cudaq::spin_handler (C++     |
| truct)](api/languages/cpp_api.htm |                                   |
| l#_CPPv4N5cudaq14other_policiesE) |   class)](api/languages/cpp_api.h |
| -   [cudaq::PasqalRemoteRESTQPU   | tml#_CPPv4N5cudaq12spin_handlerE) |
|     (C++                          | -   [cudaq:                       |
|     class                         | :spin_handler::to_diagonal_matrix |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq19PasqalRemoteRESTQPUE) |     function)](api/la             |
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
