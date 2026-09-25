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
        -   [How to pre-compile a target config
            YAML](using/extending/packaging.html#how-to-pre-compile-a-target-config-yaml){.reference
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
            -   [Baking a target into the pre-compiled database instead
                of shipping a
                plugin](using/extending/packaging.html#baking-a-target-into-the-pre-compiled-database-instead-of-shipping-a-plugin){.reference
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
            -   [[`exp_pauli`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#exp-pauli){.reference
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

-   [CUDA-Q Logical](/preview/logical/#http://){.reference .external}
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
|     cudaq.con                     | -   [availability_diagnostic      |
| trib)](api/languages/python_api.h |     (cudaq.Target                 |
| tml#cudaq.contrib.angular_encode) |     property)](a                  |
| -   [annotations (cudaq.DEMResult | pi/languages/python_api.html#cuda |
|     pro                           | q.Target.availability_diagnostic) |
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
| -   [canonicalize                 | -   [cudaq::pauli2::pauli2 (C++   |
|     (cu                           |     function)](api/languages/cpp_ |
| daq.operators.boson.BosonOperator | api.html#_CPPv4N5cudaq6pauli26pau |
|     attribute)](api/languages     | li2ERKNSt6vectorIN5cudaq4realEEE) |
| /python_api.html#cudaq.operators. | -   [cudaq::phase_damping (C++    |
| boson.BosonOperator.canonicalize) |                                   |
|     -   [(cudaq.                  |  class)](api/languages/cpp_api.ht |
| operators.boson.BosonOperatorTerm | ml#_CPPv4N5cudaq13phase_dampingE) |
|                                   | -   [cud                          |
|     attribute)](api/languages/pyt | aq::phase_damping::num_parameters |
| hon_api.html#cudaq.operators.boso |     (C++                          |
| n.BosonOperatorTerm.canonicalize) |     member)](api/lan              |
|     -   [(cudaq.                  | guages/cpp_api.html#_CPPv4N5cudaq |
| operators.fermion.FermionOperator | 13phase_damping14num_parametersE) |
|                                   | -   [                             |
|     attribute)](api/languages/pyt | cudaq::phase_damping::num_targets |
| hon_api.html#cudaq.operators.ferm |     (C++                          |
| ion.FermionOperator.canonicalize) |     member)](api/                 |
|     -   [(cudaq.oper              | languages/cpp_api.html#_CPPv4N5cu |
| ators.fermion.FermionOperatorTerm | daq13phase_damping11num_targetsE) |
|                                   | -   [cudaq::phase_flip_channel    |
| attribute)](api/languages/python_ |     (C++                          |
| api.html#cudaq.operators.fermion. |     clas                          |
| FermionOperatorTerm.canonicalize) | s)](api/languages/cpp_api.html#_C |
|     -                             | PPv4N5cudaq18phase_flip_channelE) |
|  [(cudaq.operators.MatrixOperator | -   [cudaq::p                     |
|         attribute)](api/lang      | hase_flip_channel::num_parameters |
| uages/python_api.html#cudaq.opera |     (C++                          |
| tors.MatrixOperator.canonicalize) |     member)](api/language         |
|     -   [(c                       | s/cpp_api.html#_CPPv4N5cudaq18pha |
| udaq.operators.MatrixOperatorTerm | se_flip_channel14num_parametersE) |
|         attribute)](api/language  | -   [cudaq                        |
| s/python_api.html#cudaq.operators | ::phase_flip_channel::num_targets |
| .MatrixOperatorTerm.canonicalize) |     (C++                          |
|     -   [(                        |     member)](api/langu            |
| cudaq.operators.spin.SpinOperator | ages/cpp_api.html#_CPPv4N5cudaq18 |
|         attribute)](api/languag   | phase_flip_channel11num_targetsE) |
| es/python_api.html#cudaq.operator | -   [cudaq::product_op (C++       |
| s.spin.SpinOperator.canonicalize) |                                   |
|     -   [(cuda                    |  class)](api/languages/cpp_api.ht |
| q.operators.spin.SpinOperatorTerm | ml#_CPPv4I0EN5cudaq10product_opE) |
|                                   | -   [cudaq::product_op::begin     |
|       attribute)](api/languages/p |     (C++                          |
| ython_api.html#cudaq.operators.sp |     functio                       |
| in.SpinOperatorTerm.canonicalize) | n)](api/languages/cpp_api.html#_C |
| -   [captured_variables()         | PPv4NK5cudaq10product_op5beginEv) |
|     (cudaq.PyKernelDecorator      | -                                 |
|     method)](api/lan              |  [cudaq::product_op::canonicalize |
| guages/python_api.html#cudaq.PyKe |     (C++                          |
| rnelDecorator.captured_variables) |     func                          |
| -   [CentralDifference (class in  | tion)](api/languages/cpp_api.html |
|     cudaq.gradients)              | #_CPPv4N5cudaq10product_op12canon |
| ](api/languages/python_api.html#c | icalizeERKNSt3setINSt6size_tEEE), |
| udaq.gradients.CentralDifference) |     [\[1\]](api                   |
| -   [channel                      | /languages/cpp_api.html#_CPPv4N5c |
|     (cudaq.ptsbe.TraceInstruction | udaq10product_op12canonicalizeEv) |
|     property)](a                  | -   [                             |
| pi/languages/python_api.html#cuda | cudaq::product_op::const_iterator |
| q.ptsbe.TraceInstruction.channel) |     (C++                          |
| -   [circuit_location             |     struct)](api/                 |
|     (cudaq.ptsbe.KrausSelection   | languages/cpp_api.html#_CPPv4N5cu |
|     property)](api/lang           | daq10product_op14const_iteratorE) |
| uages/python_api.html#cudaq.ptsbe | -   [cudaq::product_o             |
| .KrausSelection.circuit_location) | p::const_iterator::const_iterator |
| -   [clear (cudaq.Resources       |     (C++                          |
|                                   |     fu                            |
|   attribute)](api/languages/pytho | nction)](api/languages/cpp_api.ht |
| n_api.html#cudaq.Resources.clear) | ml#_CPPv4N5cudaq10product_op14con |
|     -   [(cudaq.SampleResult      | st_iterator14const_iteratorEPK10p |
|         a                         | roduct_opI9HandlerTyENSt6size_tE) |
| ttribute)](api/languages/python_a | -   [cudaq::produ                 |
| pi.html#cudaq.SampleResult.clear) | ct_op::const_iterator::operator!= |
| -   [CliffordTSequence (class in  |     (C++                          |
|     cudaq.sy                      |     fun                           |
| nth)](api/languages/python_api.ht | ction)](api/languages/cpp_api.htm |
| ml#cudaq.synth.CliffordTSequence) | l#_CPPv4NK5cudaq10product_op14con |
| -   [COBYLA (class in             | st_iteratorneERK14const_iterator) |
|     cudaq.o                       | -   [cudaq::produ                 |
| ptimizers)](api/languages/python_ | ct_op::const_iterator::operator\* |
| api.html#cudaq.optimizers.COBYLA) |     (C++                          |
| -   [coefficient                  |     function)](api/lang           |
|     (cudaq.                       | uages/cpp_api.html#_CPPv4NK5cudaq |
| operators.boson.BosonOperatorTerm | 10product_op14const_iteratormlEv) |
|     property)](api/languages/py   | -   [cudaq::produ                 |
| thon_api.html#cudaq.operators.bos | ct_op::const_iterator::operator++ |
| on.BosonOperatorTerm.coefficient) |     (C++                          |
|     -   [(cudaq.oper              |     function)](api/lang           |
| ators.fermion.FermionOperatorTerm | uages/cpp_api.html#_CPPv4N5cudaq1 |
|                                   | 0product_op14const_iteratorppEi), |
|   property)](api/languages/python |     [\[1\]](api/lan               |
| _api.html#cudaq.operators.fermion | guages/cpp_api.html#_CPPv4N5cudaq |
| .FermionOperatorTerm.coefficient) | 10product_op14const_iteratorppEv) |
|     -   [(c                       | -   [cudaq::produc                |
| udaq.operators.MatrixOperatorTerm | t_op::const_iterator::operator\-- |
|         property)](api/languag    |     (C++                          |
| es/python_api.html#cudaq.operator |     function)](api/lang           |
| s.MatrixOperatorTerm.coefficient) | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     -   [(cuda                    | 0product_op14const_iteratormmEi), |
| q.operators.spin.SpinOperatorTerm |     [\[1\]](api/lan               |
|         property)](api/languages/ | guages/cpp_api.html#_CPPv4N5cudaq |
| python_api.html#cudaq.operators.s | 10product_op14const_iteratormmEv) |
| pin.SpinOperatorTerm.coefficient) | -   [cudaq::produc                |
| -   [col_count                    | t_op::const_iterator::operator-\> |
|     (cudaq.KrausOperator          |     (C++                          |
|     prope                         |     function)](api/lan            |
| rty)](api/languages/python_api.ht | guages/cpp_api.html#_CPPv4N5cudaq |
| ml#cudaq.KrausOperator.col_count) | 10product_op14const_iteratorptEv) |
| -   [compile()                    | -   [cudaq::produ                 |
|     (cudaq.PyKernelDecorator      | ct_op::const_iterator::operator== |
|     metho                         |     (C++                          |
| d)](api/languages/python_api.html |     fun                           |
| #cudaq.PyKernelDecorator.compile) | ction)](api/languages/cpp_api.htm |
| -   [compiledModuleCache()        | l#_CPPv4NK5cudaq10product_op14con |
|     (cudaq.PyKernelDecorator      | st_iteratoreqERK14const_iterator) |
|     method)](api/lang             | -   [cudaq::product_op::degrees   |
| uages/python_api.html#cudaq.PyKer |     (C++                          |
| nelDecorator.compiledModuleCache) |     function)                     |
| -   [ComplexMatrix (class in      | ](api/languages/cpp_api.html#_CPP |
|     cudaq)](api/languages/pyt     | v4NK5cudaq10product_op7degreesEv) |
| hon_api.html#cudaq.ComplexMatrix) | -   [cudaq::product_op::dump (C++ |
| -   [compute                      |     functi                        |
|     (                             | on)](api/languages/cpp_api.html#_ |
| cudaq.gradients.CentralDifference | CPPv4NK5cudaq10product_op4dumpEv) |
|     attribute)](api/la            | -   [cudaq::product_op::end (C++  |
| nguages/python_api.html#cudaq.gra |     funct                         |
| dients.CentralDifference.compute) | ion)](api/languages/cpp_api.html# |
|     -   [(                        | _CPPv4NK5cudaq10product_op3endEv) |
| cudaq.gradients.ForwardDifference | -   [c                            |
|         attribute)](api/la        | udaq::product_op::get_coefficient |
| nguages/python_api.html#cudaq.gra |     (C++                          |
| dients.ForwardDifference.compute) |     function)](api/lan            |
|     -                             | guages/cpp_api.html#_CPPv4NK5cuda |
|  [(cudaq.gradients.ParameterShift | q10product_op15get_coefficientEv) |
|         attribute)](api           | -                                 |
| /languages/python_api.html#cudaq. |   [cudaq::product_op::get_term_id |
| gradients.ParameterShift.compute) |     (C++                          |
| -   [const()                      |     function)](api                |
|                                   | /languages/cpp_api.html#_CPPv4NK5 |
|   (cudaq.operators.ScalarOperator | cudaq10product_op11get_term_idEv) |
|     class                         | -                                 |
|     method)](a                    |   [cudaq::product_op::is_identity |
| pi/languages/python_api.html#cuda |     (C++                          |
| q.operators.ScalarOperator.const) |     function)](api                |
| -   [controls                     | /languages/cpp_api.html#_CPPv4NK5 |
|     (cudaq.ptsbe.TraceInstruction | cudaq10product_op11is_identityEv) |
|     property)](ap                 | -   [cudaq::product_op::num_ops   |
| i/languages/python_api.html#cudaq |     (C++                          |
| .ptsbe.TraceInstruction.controls) |     function)                     |
| -   [copy                         | ](api/languages/cpp_api.html#_CPP |
|     (cu                           | v4NK5cudaq10product_op7num_opsEv) |
| daq.operators.boson.BosonOperator | -                                 |
|     attribute)](api/l             |    [cudaq::product_op::operator\* |
| anguages/python_api.html#cudaq.op |     (C++                          |
| erators.boson.BosonOperator.copy) |     function)](api/languages/     |
|     -   [(cudaq.                  | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| operators.boson.BosonOperatorTerm | oduct_opmlE10product_opI1TERK15sc |
|         attribute)](api/langu     | alar_operatorRK10product_opI1TE), |
| ages/python_api.html#cudaq.operat |     [\[1\]](api/languages/        |
| ors.boson.BosonOperatorTerm.copy) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(cudaq.                  | oduct_opmlE10product_opI1TERK15sc |
| operators.fermion.FermionOperator | alar_operatorRR10product_opI1TE), |
|         attribute)](api/langu     |     [\[2\]](api/languages/        |
| ages/python_api.html#cudaq.operat | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| ors.fermion.FermionOperator.copy) | oduct_opmlE10product_opI1TERR15sc |
|     -   [(cudaq.oper              | alar_operatorRK10product_opI1TE), |
| ators.fermion.FermionOperatorTerm |     [\[3\]](api/languages/        |
|         attribute)](api/languages | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| /python_api.html#cudaq.operators. | oduct_opmlE10product_opI1TERR15sc |
| fermion.FermionOperatorTerm.copy) | alar_operatorRR10product_opI1TE), |
|     -                             |     [\[4\]](api/                  |
|  [(cudaq.operators.MatrixOperator | languages/cpp_api.html#_CPPv4I0EN |
|         attribute)](              | 5cudaq10product_opmlE6sum_opI1TER |
| api/languages/python_api.html#cud | K15scalar_operatorRK6sum_opI1TE), |
| aq.operators.MatrixOperator.copy) |     [\[5\]](api/                  |
|     -   [(c                       | languages/cpp_api.html#_CPPv4I0EN |
| udaq.operators.MatrixOperatorTerm | 5cudaq10product_opmlE6sum_opI1TER |
|         attribute)](api/          | K15scalar_operatorRR6sum_opI1TE), |
| languages/python_api.html#cudaq.o |     [\[6\]](api/                  |
| perators.MatrixOperatorTerm.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(                        | 5cudaq10product_opmlE6sum_opI1TER |
| cudaq.operators.spin.SpinOperator | R15scalar_operatorRK6sum_opI1TE), |
|         attribute)](api           |     [\[7\]](api/                  |
| /languages/python_api.html#cudaq. | languages/cpp_api.html#_CPPv4I0EN |
| operators.spin.SpinOperator.copy) | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [(cuda                    | R15scalar_operatorRR6sum_opI1TE), |
| q.operators.spin.SpinOperatorTerm |     [\[8\]](api/languages         |
|         attribute)](api/lan       | /cpp_api.html#_CPPv4NK5cudaq10pro |
| guages/python_api.html#cudaq.oper | duct_opmlERK6sum_opI9HandlerTyE), |
| ators.spin.SpinOperatorTerm.copy) |     [\[9\]](api/languages/cpp_a   |
| -   [count (cudaq.Resources       | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opmlERK10product_opI9HandlerTyE), |
|   attribute)](api/languages/pytho |     [\[10\]](api/language         |
| n_api.html#cudaq.Resources.count) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [(cudaq.SampleResult      | roduct_opmlERK15scalar_operator), |
|         a                         |     [\[11\]](api/languages/cpp_a  |
| ttribute)](api/languages/python_a | pi.html#_CPPv4NKR5cudaq10product_ |
| pi.html#cudaq.SampleResult.count) | opmlERR10product_opI9HandlerTyE), |
| -   [count_controls               |     [\[12\]](api/language         |
|     (cudaq.Resources              | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     attribu                       | roduct_opmlERR15scalar_operator), |
| te)](api/languages/python_api.htm |     [\[13\]](api/languages/cpp_   |
| l#cudaq.Resources.count_controls) | api.html#_CPPv4NO5cudaq10product_ |
| -   [count_instructions           | opmlERK10product_opI9HandlerTyE), |
|                                   |     [\[14\]](api/languag          |
|   (cudaq.ptsbe.PTSBEExecutionData | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     attribute)](api/languages/    | roduct_opmlERK15scalar_operator), |
| python_api.html#cudaq.ptsbe.PTSBE |     [\[15\]](api/languages/cpp_   |
| ExecutionData.count_instructions) | api.html#_CPPv4NO5cudaq10product_ |
| -   [counts (cudaq.ObserveResult  | opmlERR10product_opI9HandlerTyE), |
|     att                           |     [\[16\]](api/langua           |
| ribute)](api/languages/python_api | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| .html#cudaq.ObserveResult.counts) | product_opmlERR15scalar_operator) |
|     -   [(cudaq.SampleResult      | -                                 |
|         p                         |   [cudaq::product_op::operator\*= |
| roperty)](api/languages/python_ap |     (C++                          |
| i.html#cudaq.SampleResult.counts) |     function)](api/languages/cpp  |
| -   [csr_spmatrix (C++            | _api.html#_CPPv4N5cudaq10product_ |
|     type)](api/languages/c        | opmLERK10product_opI9HandlerTyE), |
| pp_api.html#_CPPv412csr_spmatrix) |     [\[1\]](api/langua            |
| -   cudaq                         | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     -   [module](api/langua       | roduct_opmLERK15scalar_operator), |
| ges/python_api.html#module-cudaq) |     [\[2\]](api/languages/cp      |
| -   [cudaq (C++                   | p_api.html#_CPPv4N5cudaq10product |
|     type)](api/lan                | _opmLERR10product_opI9HandlerTyE) |
| guages/cpp_api.html#_CPPv45cudaq) | -   [cudaq::product_op::operator+ |
| -   [cudaq.apply_noise() (in      |     (C++                          |
|     module                        |     function)](api/langu          |
|     cudaq)](api/languages/python_ | ages/cpp_api.html#_CPPv4I0EN5cuda |
| api.html#cudaq.cudaq.apply_noise) | q10product_opplE6sum_opI1TERK15sc |
| -   cudaq.boson                   | alar_operatorRK10product_opI1TE), |
|     -   [module](api/languages/py |     [\[1\]](api/                  |
| thon_api.html#module-cudaq.boson) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.fermion                 | 5cudaq10product_opplE6sum_opI1TER |
|                                   | K15scalar_operatorRK6sum_opI1TE), |
|   -   [module](api/languages/pyth |     [\[2\]](api/langu             |
| on_api.html#module-cudaq.fermion) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq.operators.custom        | q10product_opplE6sum_opI1TERK15sc |
|     -   [mo                       | alar_operatorRR10product_opI1TE), |
| dule](api/languages/python_api.ht |     [\[3\]](api/                  |
| ml#module-cudaq.operators.custom) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.spin                    | 5cudaq10product_opplE6sum_opI1TER |
|     -   [module](api/languages/p  | K15scalar_operatorRR6sum_opI1TE), |
| ython_api.html#module-cudaq.spin) |     [\[4\]](api/langu             |
| -   [cudaq::amplitude_damping     | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opplE6sum_opI1TERR15sc |
|     cla                           | alar_operatorRK10product_opI1TE), |
| ss)](api/languages/cpp_api.html#_ |     [\[5\]](api/                  |
| CPPv4N5cudaq17amplitude_dampingE) | languages/cpp_api.html#_CPPv4I0EN |
| -                                 | 5cudaq10product_opplE6sum_opI1TER |
| [cudaq::amplitude_damping_channel | R15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[6\]](api/langu             |
|     class)](api                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
| /languages/cpp_api.html#_CPPv4N5c | q10product_opplE6sum_opI1TERR15sc |
| udaq25amplitude_damping_channelE) | alar_operatorRR10product_opI1TE), |
| -   [cudaq::amplitud              |     [\[7\]](api/                  |
| e_damping_channel::num_parameters | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     member)](api/languages/cpp_a  | R15scalar_operatorRR6sum_opI1TE), |
| pi.html#_CPPv4N5cudaq25amplitude_ |     [\[8\]](api/languages/cpp_a   |
| damping_channel14num_parametersE) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [cudaq::ampli                 | opplERK10product_opI9HandlerTyE), |
| tude_damping_channel::num_targets |     [\[9\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     member)](api/languages/cp     | roduct_opplERK15scalar_operator), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [\[10\]](api/languages/       |
| de_damping_channel11num_targetsE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::AnalogRemoteRESTQPU   | duct_opplERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     class                         | pi.html#_CPPv4NKR5cudaq10product_ |
| )](api/languages/cpp_api.html#_CP | opplERR10product_opI9HandlerTyE), |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) |     [\[12\]](api/language         |
| -   [cudaq::apply_noise (C++      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/               | roduct_opplERR15scalar_operator), |
| languages/cpp_api.html#_CPPv4I0Dp |     [\[13\]](api/languages/       |
| EN5cudaq11apply_noiseEvDpRR4Args) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::async_result (C++     | duct_opplERR6sum_opI9HandlerTyE), |
|     c                             |     [\[                           |
| lass)](api/languages/cpp_api.html | 14\]](api/languages/cpp_api.html# |
| #_CPPv4I0EN5cudaq12async_resultE) | _CPPv4NKR5cudaq10product_opplEv), |
| -   [cudaq::async_result::get     |     [\[15\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     functi                        | opplERK10product_opI9HandlerTyE), |
| on)](api/languages/cpp_api.html#_ |     [\[16\]](api/languag          |
| CPPv4N5cudaq12async_result3getEv) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::async_sample_result   | roduct_opplERK15scalar_operator), |
|     (C++                          |     [\[17\]](api/languages        |
|     type                          | /cpp_api.html#_CPPv4NO5cudaq10pro |
| )](api/languages/cpp_api.html#_CP | duct_opplERK6sum_opI9HandlerTyE), |
| Pv4N5cudaq19async_sample_resultE) |     [\[18\]](api/languages/cpp_   |
| -   [cudaq::BaseRemoteRESTQPU     | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opplERR10product_opI9HandlerTyE), |
|     cla                           |     [\[19\]](api/languag          |
| ss)](api/languages/cpp_api.html#_ | es/cpp_api.html#_CPPv4NO5cudaq10p |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | roduct_opplERR15scalar_operator), |
| -   [cudaq::bit_flip_channel (C++ |     [\[20\]](api/languages        |
|     cl                            | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ass)](api/languages/cpp_api.html# | duct_opplERR6sum_opI9HandlerTyE), |
| _CPPv4N5cudaq16bit_flip_channelE) |     [                             |
| -   [cudaq:                       | \[21\]](api/languages/cpp_api.htm |
| :bit_flip_channel::num_parameters | l#_CPPv4NO5cudaq10product_opplEv) |
|     (C++                          | -   [cudaq::product_op::operator- |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16b |     function)](api/langu          |
| it_flip_channel14num_parametersE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cud                          | q10product_opmiE6sum_opI1TERK15sc |
| aq::bit_flip_channel::num_targets | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     member)](api/lan              | languages/cpp_api.html#_CPPv4I0EN |
| guages/cpp_api.html#_CPPv4N5cudaq | 5cudaq10product_opmiE6sum_opI1TER |
| 16bit_flip_channel11num_targetsE) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::boson_handler (C++    |     [\[2\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|  class)](api/languages/cpp_api.ht | q10product_opmiE6sum_opI1TERK15sc |
| ml#_CPPv4N5cudaq13boson_handlerE) | alar_operatorRR10product_opI1TE), |
| -   [cudaq::boson_op (C++         |     [\[3\]](api/                  |
|     type)](api/languages/cpp_     | languages/cpp_api.html#_CPPv4I0EN |
| api.html#_CPPv4N5cudaq8boson_opE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::boson_op_term (C++    | K15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[4\]](api/langu             |
|   type)](api/languages/cpp_api.ht | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ml#_CPPv4N5cudaq13boson_op_termE) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq::CodeGenConfig (C++    | alar_operatorRK10product_opI1TE), |
|                                   |     [\[5\]](api/                  |
| struct)](api/languages/cpp_api.ht | languages/cpp_api.html#_CPPv4I0EN |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::commutation_relations | R15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[6\]](api/langu             |
|     struct)]                      | ages/cpp_api.html#_CPPv4I0EN5cuda |
| (api/languages/cpp_api.html#_CPPv | q10product_opmiE6sum_opI1TERR15sc |
| 4N5cudaq21commutation_relationsE) | alar_operatorRR10product_opI1TE), |
| -   [cudaq::complex (C++          |     [\[7\]](api/                  |
|     type)](api/languages/cpp      | languages/cpp_api.html#_CPPv4I0EN |
| _api.html#_CPPv4N5cudaq7complexE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::complex_matrix (C++   | R15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[8\]](api/languages/cpp_a   |
| class)](api/languages/cpp_api.htm | pi.html#_CPPv4NKR5cudaq10product_ |
| l#_CPPv4N5cudaq14complex_matrixE) | opmiERK10product_opI9HandlerTyE), |
| -                                 |     [\[9\]](api/language          |
|   [cudaq::complex_matrix::adjoint | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opmiERK15scalar_operator), |
|     function)](a                  |     [\[10\]](api/languages/       |
| pi/languages/cpp_api.html#_CPPv4N | cpp_api.html#_CPPv4NKR5cudaq10pro |
| 5cudaq14complex_matrix7adjointEv) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::                      |     [\[11\]](api/languages/cpp_a  |
| complex_matrix::diagonal_elements | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmiERR10product_opI9HandlerTyE), |
|     function)](api/languages      |     [\[12\]](api/language         |
| /cpp_api.html#_CPPv4NK5cudaq14com | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| plex_matrix17diagonal_elementsEi) | roduct_opmiERR15scalar_operator), |
| -   [cudaq::complex_matrix::dump  |     [\[13\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     function)](api/language       | duct_opmiERR6sum_opI9HandlerTyE), |
| s/cpp_api.html#_CPPv4NK5cudaq14co |     [\[                           |
| mplex_matrix4dumpERNSt7ostreamE), | 14\]](api/languages/cpp_api.html# |
|     [\[1\]]                       | _CPPv4NKR5cudaq10product_opmiEv), |
| (api/languages/cpp_api.html#_CPPv |     [\[15\]](api/languages/cpp_   |
| 4NK5cudaq14complex_matrix4dumpEv) | api.html#_CPPv4NO5cudaq10product_ |
| -   [c                            | opmiERK10product_opI9HandlerTyE), |
| udaq::complex_matrix::eigenvalues |     [\[16\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     function)](api/lan            | roduct_opmiERK15scalar_operator), |
| guages/cpp_api.html#_CPPv4NK5cuda |     [\[17\]](api/languages        |
| q14complex_matrix11eigenvaluesEv) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cu                           | duct_opmiERK6sum_opI9HandlerTyE), |
| daq::complex_matrix::eigenvectors |     [\[18\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     function)](api/lang           | opmiERR10product_opI9HandlerTyE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[19\]](api/languag          |
| 14complex_matrix12eigenvectorsEv) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [c                            | roduct_opmiERR15scalar_operator), |
| udaq::complex_matrix::exponential |     [\[20\]](api/languages        |
|     (C++                          | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     function)](api/la             | duct_opmiERR6sum_opI9HandlerTyE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [                             |
| q14complex_matrix11exponentialEv) | \[21\]](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4NO5cudaq10product_opmiEv) |
|  [cudaq::complex_matrix::identity | -   [cudaq::product_op::operator/ |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/language       |
| /cpp_api.html#_CPPv4N5cudaq14comp | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| lex_matrix8identityEKNSt6size_tE) | roduct_opdvERK15scalar_operator), |
| -                                 |     [\[1\]](api/language          |
| [cudaq::complex_matrix::kronecker | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opdvERR15scalar_operator), |
|     function)](api/lang           |     [\[2\]](api/languag           |
| uages/cpp_api.html#_CPPv4I00EN5cu | es/cpp_api.html#_CPPv4NO5cudaq10p |
| daq14complex_matrix9kroneckerE14c | roduct_opdvERK15scalar_operator), |
| omplex_matrix8Iterable8Iterable), |     [\[3\]](api/langua            |
|     [\[1\]](api/l                 | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| anguages/cpp_api.html#_CPPv4N5cud | product_opdvERR15scalar_operator) |
| aq14complex_matrix9kroneckerERK14 | -                                 |
| complex_matrixRK14complex_matrix) |    [cudaq::product_op::operator/= |
| -   [cudaq::c                     |     (C++                          |
| omplex_matrix::minimal_eigenvalue |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     function)](api/languages/     | product_opdVERK15scalar_operator) |
| cpp_api.html#_CPPv4NK5cudaq14comp | -   [cudaq::product_op::operator= |
| lex_matrix18minimal_eigenvalueEv) |     (C++                          |
| -   [                             |     function)](api/l              |
| cudaq::complex_matrix::operator() | anguages/cpp_api.html#_CPPv4I00EN |
|     (C++                          | 5cudaq10product_opaSER10product_o |
|     function)](api/languages/cpp  | pI9HandlerTyERK10product_opI1TE), |
| _api.html#_CPPv4N5cudaq14complex_ |     [\[1\]](api/languages/cpp     |
| matrixclENSt6size_tENSt6size_tE), | _api.html#_CPPv4N5cudaq10product_ |
|     [\[1\]](api/languages/cpp     | opaSERK10product_opI9HandlerTyE), |
| _api.html#_CPPv4NK5cudaq14complex |     [\[2\]](api/languages/cp      |
| _matrixclENSt6size_tENSt6size_tE) | p_api.html#_CPPv4N5cudaq10product |
| -   [                             | _opaSERR10product_opI9HandlerTyE) |
| cudaq::complex_matrix::operator\* | -                                 |
|     (C++                          |    [cudaq::product_op::operator== |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     function)](api/languages/cpp  |
| omplex_matrixmlEN14complex_matrix | _api.html#_CPPv4NK5cudaq10product |
| 10value_typeERK14complex_matrix), | _opeqERK10product_opI9HandlerTyE) |
|     [\[1\]                        | -                                 |
| ](api/languages/cpp_api.html#_CPP |  [cudaq::product_op::operator\[\] |
| v4N5cudaq14complex_matrixmlERK14c |     (C++                          |
| omplex_matrixRK14complex_matrix), |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|  [\[2\]](api/languages/cpp_api.ht | 5cudaq10product_opixENSt6size_tE) |
| ml#_CPPv4N5cudaq14complex_matrixm | -                                 |
| lERK14complex_matrixRKNSt6vectorI |    [cudaq::product_op::product_op |
| N14complex_matrix10value_typeEEE) |     (C++                          |
| -                                 |     f                             |
| [cudaq::complex_matrix::operator+ | unction)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4I00EN5cudaq10product_op |
|     function                      | 10product_opERK10product_opI1TE), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]]                       |
| Pv4N5cudaq14complex_matrixplERK14 | (api/languages/cpp_api.html#_CPPv |
| complex_matrixRK14complex_matrix) | 4I00EN5cudaq10product_op10product |
| -                                 | _opERK10product_opI1TERKN14matrix |
| [cudaq::complex_matrix::operator- | _handler20commutation_behaviorE), |
|     (C++                          |                                   |
|     function                      |   [\[2\]](api/languages/cpp_api.h |
| )](api/languages/cpp_api.html#_CP | tml#_CPPv4N5cudaq10product_op10pr |
| Pv4N5cudaq14complex_matrixmiERK14 | oduct_opENSt6size_tENSt6size_tE), |
| complex_matrixRK14complex_matrix) |     [\[3\]](api/languages/cp      |
| -   [cu                           | p_api.html#_CPPv4N5cudaq10product |
| daq::complex_matrix::operator\[\] | _op10product_opENSt7complexIdEE), |
|     (C++                          |     [\[4\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|  function)](api/languages/cpp_api | aq10product_op10product_opERK10pr |
| .html#_CPPv4N5cudaq14complex_matr | oduct_opI9HandlerTyENSt6size_tE), |
| ixixERKNSt6vectorINSt6size_tEEE), |     [\[5\]](api/l                 |
|     [\[1\]](api/languages/cpp_api | anguages/cpp_api.html#_CPPv4N5cud |
| .html#_CPPv4NK5cudaq14complex_mat | aq10product_op10product_opERR10pr |
| rixixERKNSt6vectorINSt6size_tEEE) | oduct_opI9HandlerTyENSt6size_tE), |
| -   [cudaq::complex_matrix::power |     [\[6\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq10prod |
|     function)]                    | uct_op10product_opERR9HandlerTy), |
| (api/languages/cpp_api.html#_CPPv |     [\[7\]](ap                    |
| 4N5cudaq14complex_matrix5powerEi) | i/languages/cpp_api.html#_CPPv4N5 |
| -                                 | cudaq10product_op10product_opEd), |
|  [cudaq::complex_matrix::set_zero |     [\[8\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function)](ap                 | 5cudaq10product_op10product_opEv) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cuda                         |
| cudaq14complex_matrix8set_zeroEv) | q::product_op::to_diagonal_matrix |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::to_string |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4NK5c |
|     function)](api/               | udaq10product_op18to_diagonal_mat |
| languages/cpp_api.html#_CPPv4NK5c | rixENSt13unordered_mapINSt6size_t |
| udaq14complex_matrix9to_stringEv) | ENSt7int64_tEEERKNSt13unordered_m |
| -   [                             | apINSt6stringENSt7complexIdEEEEb) |
| cudaq::complex_matrix::value_type | -   [cudaq::product_op::to_matrix |
|     (C++                          |     (C++                          |
|     type)](api/                   |     funct                         |
| languages/cpp_api.html#_CPPv4N5cu | ion)](api/languages/cpp_api.html# |
| daq14complex_matrix10value_typeE) | _CPPv4NK5cudaq10product_op9to_mat |
| -   [cudaq::contrib (C++          | rixENSt13unordered_mapINSt6size_t |
|     type)](api/languages/cpp      | ENSt7int64_tEEERKNSt13unordered_m |
| _api.html#_CPPv4N5cudaq7contribE) | apINSt6stringENSt7complexIdEEEEb) |
| -                                 | -   [cu                           |
| [cudaq::contrib::amplitude_encode | daq::product_op::to_sparse_matrix |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](ap                 |
| s/cpp_api.html#_CPPv4N5cudaq7cont | i/languages/cpp_api.html#_CPPv4NK |
| rib16amplitude_encodeENSt4spanIKN | 5cudaq10product_op16to_sparse_mat |
| St7complexIdEEEENSt7complexIdEE), | rixENSt13unordered_mapINSt6size_t |
|     [\[1\]](api/language          | ENSt7int64_tEEERKNSt13unordered_m |
| s/cpp_api.html#_CPPv4N5cudaq7cont | apINSt6stringENSt7complexIdEEEEb) |
| rib16amplitude_encodeENSt4spanIKN | -   [cudaq::product_op::to_string |
| St7complexIfEEEENSt7complexIdEE), |     (C++                          |
|     [\[2\]                        |     function)](                   |
| ](api/languages/cpp_api.html#_CPP | api/languages/cpp_api.html#_CPPv4 |
| v4N5cudaq7contrib16amplitude_enco | NK5cudaq10product_op9to_stringEv) |
| deENSt4spanIKdEENSt7complexIdEE), | -                                 |
|     [\[3\]                        |  [cudaq::product_op::\~product_op |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq7contrib16amplitude_enco |     fu                            |
| deENSt4spanIKfEENSt7complexIdEE), | nction)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4N5cudaq10product_opD0Ev) |
| [\[4\]](api/languages/cpp_api.htm | -   [cudaq::ptsbe (C++            |
| l#_CPPv4N5cudaq7contrib16amplitud |     type)](api/languages/c        |
| e_encodeERK5stateNSt7complexIdEE) | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| -                                 | -   [cudaq::p                     |
|   [cudaq::contrib::angular_encode | tsbe::ConditionalSamplingStrategy |
|     (C++                          |     (C++                          |
|                                   |     class)](api/languag           |
|  function)](api/languages/cpp_api | es/cpp_api.html#_CPPv4N5cudaq5pts |
| .html#_CPPv4I0EN5cudaq7contrib14a | be27ConditionalSamplingStrategyE) |
| ngular_encodeEvRR6KernelR10QuakeV | -   [cudaq::ptsbe::C              |
| alueNSt4spanIKdEE12RotationAxis), | onditionalSamplingStrategy::clone |
|     [\[1\]](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I0EN5cudaq7contrib14a |                                   |
| ngular_encodeEvRR6KernelR10QuakeV |    function)](api/languages/cpp_a |
| alueR10QuakeValue12RotationAxis), | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
|                                   | ditionalSamplingStrategy5cloneEv) |
|   [\[2\]](api/languages/cpp_api.h | -   [cuda                         |
| tml#_CPPv4I0EN5cudaq7contrib14ang | q::ptsbe::ConditionalSamplingStra |
| ular_encodeEvRR6KernelR10QuakeVal | tegy::ConditionalSamplingStrategy |
| ueRKNSt6vectorIdEE12RotationAxis) |     (C++                          |
| -   [cudaq::contrib::draw (C++    |     function)](api/lang           |
|     function)                     | uages/cpp_api.html#_CPPv4N5cudaq5 |
| ](api/languages/cpp_api.html#_CPP | ptsbe27ConditionalSamplingStrateg |
| v4I0DpEN5cudaq7contrib4drawENSt6s | y27ConditionalSamplingStrategyE19 |
| tringERR13QuantumKernelDpRR4Args) | TrajectoryPredicateNSt8uint64_tE) |
| -                                 | -                                 |
| [cudaq::contrib::get_unitary_cmat |   [cudaq::ptsbe::ConditionalSampl |
|     (C++                          | ingStrategy::generateTrajectories |
|     function)](api/languages/cp   |     (C++                          |
| p_api.html#_CPPv4I0DpEN5cudaq7con |     function)](api/language       |
| trib16get_unitary_cmatE14complex_ | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| matrixRR13QuantumKernelDpRR4Args) | be27ConditionalSamplingStrategy20 |
| -   [cudaq::contrib::RotationAxis | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     enum)                         | -   [cudaq::ptsbe::               |
| ](api/languages/cpp_api.html#_CPP | ConditionalSamplingStrategy::name |
| v4N5cudaq7contrib12RotationAxisE) |     (C++                          |
| -                                 |     function)](api/languages/cpp_ |
|  [cudaq::contrib::RotationAxis::X | api.html#_CPPv4NK5cudaq5ptsbe27Co |
|     (C++                          | nditionalSamplingStrategy4nameEv) |
|     enumerator)](                 | -   [cudaq:                       |
| api/languages/cpp_api.html#_CPPv4 | :ptsbe::ConditionalSamplingStrate |
| N5cudaq7contrib12RotationAxis1XE) | gy::\~ConditionalSamplingStrategy |
| -                                 |     (C++                          |
|  [cudaq::contrib::RotationAxis::Y |     function)](api/languages/     |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
|     enumerator)](                 | 7ConditionalSamplingStrategyD0Ev) |
| api/languages/cpp_api.html#_CPPv4 | -                                 |
| N5cudaq7contrib12RotationAxis1YE) | [cudaq::ptsbe::detail::NoisePoint |
| -                                 |     (C++                          |
|  [cudaq::contrib::RotationAxis::Z |     struct)](a                    |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     enumerator)](                 | 5cudaq5ptsbe6detail10NoisePointE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::p                     |
| N5cudaq7contrib12RotationAxis1ZE) | tsbe::detail::NoisePoint::channel |
| -   [cudaq::cudaq_json (C++       |     (C++                          |
|     class)](api/languages/cpp_api |     member)](api/langu            |
| .html#_CPPv4N5cudaq10cudaq_jsonE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [cudaq::DefaultQPU (C++       | tsbe6detail10NoisePoint7channelE) |
|     class)](api/languages/cpp_api | -   [cudaq::ptsbe::det            |
| .html#_CPPv4N5cudaq10DefaultQPUE) | ail::NoisePoint::circuit_location |
| -   [cudaq::dem_from_kernel (C++  |     (C++                          |
|     function)](api                |     member)](api/languages/cpp_a  |
| /languages/cpp_api.html#_CPPv4I0D | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| pEN5cudaq15dem_from_kernelENSt6st | l10NoisePoint16circuit_locationE) |
| ringERR13QuantumKernelDpRR4Args), | -   [cudaq::p                     |
|     [                             | tsbe::detail::NoisePoint::op_name |
| \[1\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4I0DpEN5cudaq15dem_from_ker |     member)](api/langu            |
| nelENSt6stringERR13QuantumKernelP | ages/cpp_api.html#_CPPv4N5cudaq5p |
| KN5cudaq11noise_modelEDpRR4Args), | tsbe6detail10NoisePoint7op_nameE) |
|     [\[2\]](api/languages/cp      | -   [cudaq::                      |
| p_api.html#_CPPv4I0DpEN5cudaq15de | ptsbe::detail::NoisePoint::qubits |
| m_from_kernelENSt6stringERR13Quan |     (C++                          |
| tumKernelPKN5cudaq11noise_modelER |     member)](api/lang             |
| KN5cudaq11dem_optionsEDpRR4Args), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     [\[3\]](ap                    | ptsbe6detail10NoisePoint6qubitsE) |
| i/languages/cpp_api.html#_CPPv4I0 | -   [cudaq::                      |
| DpEN5cudaq15dem_from_kernelENSt6s | ptsbe::ExhaustiveSamplingStrategy |
| tringERR13QuantumKernelPKN5cudaq1 |     (C++                          |
| 1noise_modelERKN5cudaq11dem_optio |     class)](api/langua            |
| nsERN5cudaq15M2DSparseMatrixERN5c | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| udaq15M2OSparseMatrixEDpRR4Args), | sbe26ExhaustiveSamplingStrategyE) |
|     [\[4\]](api/language          | -   [cudaq::ptsbe::               |
| s/cpp_api.html#_CPPv4I0DpEN5cudaq | ExhaustiveSamplingStrategy::clone |
| 15dem_from_kernelENSt6stringERR13 |     (C++                          |
| QuantumKernelPKN5cudaq11noise_mod |     function)](api/languages/cpp_ |
| elERN5cudaq15M2DSparseMatrixERN5c | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| udaq15M2OSparseMatrixEDpRR4Args), | haustiveSamplingStrategy5cloneEv) |
|     [\[5\]](api/languages/cpp_api | -   [cu                           |
| .html#_CPPv4I0DpEN5cudaq15dem_fro | daq::ptsbe::ExhaustiveSamplingStr |
| m_kernelENSt6stringERR13QuantumKe | ategy::ExhaustiveSamplingStrategy |
| rnelRN5cudaq15M2DSparseMatrixERN5 |     (C++                          |
| cudaq15M2OSparseMatrixEDpRR4Args) |     function)](api/la             |
| -   [cudaq::dem_options (C++      | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q5ptsbe26ExhaustiveSamplingStrate |
|   struct)](api/languages/cpp_api. | gy26ExhaustiveSamplingStrategyEv) |
| html#_CPPv4N5cudaq11dem_optionsE) | -                                 |
| -   [cudaq::d                     |    [cudaq::ptsbe::ExhaustiveSampl |
| em_options::allow_gauge_detectors | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     member)](api/language         |     function)](api/languag        |
| s/cpp_api.html#_CPPv4N5cudaq11dem | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| _options21allow_gauge_detectorsE) | sbe26ExhaustiveSamplingStrategy20 |
| -   [cudaq::dem_options::appr     | generateTrajectoriesENSt4spanIKN6 |
| oximate_disjoint_errors_threshold | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe:                |
|     memb                          | :ExhaustiveSamplingStrategy::name |
| er)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq11dem_options37approx |     function)](api/languages/cpp  |
| imate_disjoint_errors_thresholdE) | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| -   [cuda                         | xhaustiveSamplingStrategy4nameEv) |
| q::dem_options::block_decompositi | -   [cuda                         |
| on_from_introducing_remnant_edges | q::ptsbe::ExhaustiveSamplingStrat |
|     (C++                          | egy::\~ExhaustiveSamplingStrategy |
|     member)](api/lang             |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     function)](api/languages      |
| 1dem_options50block_decomposition | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| _from_introducing_remnant_edgesE) | 26ExhaustiveSamplingStrategyD0Ev) |
| -   [cud                          | -   [cuda                         |
| aq::dem_options::decompose_errors | q::ptsbe::OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     class)](api/lan               |
| guages/cpp_api.html#_CPPv4N5cudaq | guages/cpp_api.html#_CPPv4N5cudaq |
| 11dem_options16decompose_errorsE) | 5ptsbe23OrderedSamplingStrategyE) |
| -                                 | -   [cudaq::ptsb                  |
|   [cudaq::dem_options::fold_loops | e::OrderedSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     member)](a                    |     function)](api/languages/c    |
| pi/languages/cpp_api.html#_CPPv4N | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| 5cudaq11dem_options10fold_loopsE) | 3OrderedSamplingStrategy5cloneEv) |
| -   [cudaq::dem_optio             | -   [cudaq::ptsbe::OrderedSampl   |
| ns::ignore_decomposition_failures | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     member)](api/languages/cpp_ap |     function)](api/lang           |
| i.html#_CPPv4N5cudaq11dem_options | uages/cpp_api.html#_CPPv4NK5cudaq |
| 29ignore_decomposition_failuresE) | 5ptsbe23OrderedSamplingStrategy20 |
| -   [cudaq::dem_opt               | generateTrajectoriesENSt4spanIKN6 |
| ions::return_measurement_matrices | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/languages/cpp_   | be::OrderedSamplingStrategy::name |
| api.html#_CPPv4N5cudaq11dem_optio |     (C++                          |
| ns27return_measurement_matricesE) |     function)](api/languages/     |
| -   [cudaq::depolarization1 (C++  | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     c                             | 23OrderedSamplingStrategy4nameEv) |
| lass)](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq15depolarization1E) |    [cudaq::ptsbe::OrderedSampling |
| -   [cudaq::depolarization2 (C++  | Strategy::OrderedSamplingStrategy |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     function)](                   |
| #_CPPv4N5cudaq15depolarization2E) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq:                       | N5cudaq5ptsbe23OrderedSamplingStr |
| :depolarization2::depolarization2 | ategy23OrderedSamplingStrategyEv) |
|     (C++                          | -                                 |
|     function)](api/languages/cp   |  [cudaq::ptsbe::OrderedSamplingSt |
| p_api.html#_CPPv4N5cudaq15depolar | rategy::\~OrderedSamplingStrategy |
| ization215depolarization2EK4real) |     (C++                          |
| -   [cudaq                        |     function)](api/langua         |
| ::depolarization2::num_parameters | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     (C++                          | sbe23OrderedSamplingStrategyD0Ev) |
|     member)](api/langu            | -   [cudaq::pts                   |
| ages/cpp_api.html#_CPPv4N5cudaq15 | be::ProbabilisticSamplingStrategy |
| depolarization214num_parametersE) |     (C++                          |
| -   [cu                           |     class)](api/languages         |
| daq::depolarization2::num_targets | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     (C++                          | 29ProbabilisticSamplingStrategyE) |
|     member)](api/la               | -   [cudaq::ptsbe::Pro            |
| nguages/cpp_api.html#_CPPv4N5cuda | babilisticSamplingStrategy::clone |
| q15depolarization211num_targetsE) |     (C++                          |
| -                                 |                                   |
|    [cudaq::depolarization_channel |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4NK5cudaq5ptsbe29Proba |
|     class)](                      | bilisticSamplingStrategy5cloneEv) |
| api/languages/cpp_api.html#_CPPv4 | -                                 |
| N5cudaq22depolarization_channelE) | [cudaq::ptsbe::ProbabilisticSampl |
| -   [cudaq::depol                 | ingStrategy::generateTrajectories |
| arization_channel::num_parameters |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     member)](api/languages/cp     | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| p_api.html#_CPPv4N5cudaq22depolar | 29ProbabilisticSamplingStrategy20 |
| ization_channel14num_parametersE) | generateTrajectoriesENSt4spanIKN6 |
| -   [cudaq::de                    | detail10NoisePointEEENSt6size_tE) |
| polarization_channel::num_targets | -   [cudaq::ptsbe::Pr             |
|     (C++                          | obabilisticSamplingStrategy::name |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq22depo |                                   |
| larization_channel11num_targetsE) |   function)](api/languages/cpp_ap |
| -   [cudaq::detail (C++           | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
|     type)](api/languages/cp       | abilisticSamplingStrategy4nameEv) |
| p_api.html#_CPPv4N5cudaq6detailE) | -   [cudaq::p                     |
| -   [cudaq::detail::future (C++   | tsbe::ProbabilisticSamplingStrate |
|                                   | gy::ProbabilisticSamplingStrategy |
|   class)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq6detail6futureE) |     function)]                    |
| -                                 | (api/languages/cpp_api.html#_CPPv |
|    [cudaq::detail::future::future | 4N5cudaq5ptsbe29ProbabilisticSamp |
|     (C++                          | lingStrategy29ProbabilisticSampli |
|     functi                        | ngStrategyENSt8optionalINSt8uint6 |
| on)](api/languages/cpp_api.html#_ | 4_tEEENSt8optionalINSt6size_tEEE) |
| CPPv4N5cudaq6detail6future6future | -   [cudaq::pts                   |
| ERNSt6vectorI3JobEERNSt6stringERN | be::ProbabilisticSamplingStrategy |
| St3mapINSt6stringENSt6stringEEE), | ::\~ProbabilisticSamplingStrategy |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languages/cp   |
| 6detail6future6futureERR6future), | p_api.html#_CPPv4N5cudaq5ptsbe29P |
|     [\[2\]                        | robabilisticSamplingStrategyD0Ev) |
| ](api/languages/cpp_api.html#_CPP | -                                 |
| v4N5cudaq6detail6future6futureEv) | [cudaq::ptsbe::PTSBEExecutionData |
| -   [c                            |     (C++                          |
| udaq::detail::kernel_builder_base |     struct)](ap                   |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     class)](api/                  | cudaq5ptsbe18PTSBEExecutionDataE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::ptsbe::PTSBE          |
| daq6detail19kernel_builder_baseE) | ExecutionData::count_instructions |
| -   [cudaq::detail::              |     (C++                          |
| kernel_builder_base::operator\<\< |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     function)](api/langu          | daq5ptsbe18PTSBEExecutionData18co |
| ages/cpp_api.html#_CPPv4N5cudaq6d | unt_instructionsE20TraceInstructi |
| etail19kernel_builder_baselsERNSt | onTypeNSt8optionalINSt6stringEEE) |
| 7ostreamERK19kernel_builder_base) | -   [cudaq::ptsbe::P              |
| -                                 | TSBEExecutionData::get_trajectory |
| [cudaq::detail::KernelBuilderType |     (C++                          |
|     (C++                          |     function                      |
|     class)](ap                    | )](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| cudaq6detail17KernelBuilderTypeE) | Data14get_trajectoryENSt6size_tE) |
| -   [cudaq::                      | -   [cudaq::ptsbe:                |
| detail::KernelBuilderType::create | :PTSBEExecutionData::instructions |
|     (C++                          |     (C++                          |
|     function                      |     member)](api/languages/cp     |
| )](api/languages/cpp_api.html#_CP | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| Pv4N5cudaq6detail17KernelBuilderT | TSBEExecutionData12instructionsE) |
| ype6createEPN4mlir11MLIRContextE) | -   [cudaq::ptsbe:                |
| -   [cudaq::detail::Ker           | :PTSBEExecutionData::trajectories |
| nelBuilderType::KernelBuilderType |     (C++                          |
|     (C++                          |     member)](api/languages/cp     |
|     function)](api/lan            | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| guages/cpp_api.html#_CPPv4N5cudaq | TSBEExecutionData12trajectoriesE) |
| 6detail17KernelBuilderType17Kerne | -   [cudaq::ptsbe::PTSBEOptions   |
| lBuilderTypeERRNSt8functionIFN4ml |     (C++                          |
| ir4TypeEPN4mlir11MLIRContextEEEE) |     struc                         |
| -   [cudaq::detector (C++         | t)](api/languages/cpp_api.html#_C |
|     function)](api                | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| /languages/cpp_api.html#_CPPv4IDp | -   [cudaq::ptsbe::PTSB           |
| EN5cudaq8detectorEvDpRR8MeasArgs) | EOptions::include_sequential_data |
| -   [cudaq::detectors (C++        |     (C++                          |
|     function)](api/languages/c    |                                   |
| pp_api.html#_CPPv4N5cudaq9detecto |    member)](api/languages/cpp_api |
| rsERKNSt6vectorI14measure_resultE | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| ERKNSt6vectorI14measure_resultEE) | ptions23include_sequential_dataE) |
| -   [cudaq::diag_matrix_callback  | -   [cudaq::ptsb                  |
|     (C++                          | e::PTSBEOptions::max_trajectories |
|     class)                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     member)](api/languages/       |
| v4N5cudaq20diag_matrix_callbackE) | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| -   [cudaq::dyn (C++              | 2PTSBEOptions16max_trajectoriesE) |
|     member)](api/languages        | -   [cudaq::ptsbe::PT             |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | SBEOptions::return_execution_data |
| -   [cudaq::ExecutionContext (C++ |     (C++                          |
|     cl                            |     member)](api/languages/cpp_a  |
| ass)](api/languages/cpp_api.html# | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| _CPPv4N5cudaq16ExecutionContextE) | EOptions21return_execution_dataE) |
| -   [c                            | -   [cudaq::pts                   |
| udaq::ExecutionContext::asyncExec | be::PTSBEOptions::shot_allocation |
|     (C++                          |     (C++                          |
|     member)](api/                 |     member)](api/languages        |
| languages/cpp_api.html#_CPPv4N5cu | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| daq16ExecutionContext9asyncExecE) | 12PTSBEOptions15shot_allocationE) |
| -   [cud                          | -   [cud                          |
| aq::ExecutionContext::asyncResult | aq::ptsbe::PTSBEOptions::strategy |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     member)](api/l                |
| guages/cpp_api.html#_CPPv4N5cudaq | anguages/cpp_api.html#_CPPv4N5cud |
| 16ExecutionContext11asyncResultE) | aq5ptsbe12PTSBEOptions8strategyE) |
| -   [cudaq:                       | -   [cudaq::ptsbe::PTSBETrace     |
| :ExecutionContext::batchIteration |     (C++                          |
|     (C++                          |     t                             |
|     member)](api/langua           | ype)](api/languages/cpp_api.html# |
| ges/cpp_api.html#_CPPv4N5cudaq16E | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| xecutionContext14batchIterationE) | -   [                             |
| -   [cudaq::E                     | cudaq::ptsbe::PTSSamplingStrategy |
| xecutionContext::canHandleObserve |     (C++                          |
|     (C++                          |     class)](api                   |
|     member)](api/language         | /languages/cpp_api.html#_CPPv4N5c |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | udaq5ptsbe19PTSSamplingStrategyE) |
| cutionContext16canHandleObserveE) | -   [cudaq::                      |
| -   [cudaq::Executio              | ptsbe::PTSSamplingStrategy::clone |
| nContext::deferredKernelException |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     member)](api/languages/cpp_a  | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| pi.html#_CPPv4N5cudaq16ExecutionC | sbe19PTSSamplingStrategy5cloneEv) |
| ontext23deferredKernelExceptionE) | -   [cudaq::ptsbe::PTSSampl       |
| -   [cudaq::E                     | ingStrategy::generateTrajectories |
| xecutionContext::ExecutionContext |     (C++                          |
|     (C++                          |     function)](api/               |
|     func                          | languages/cpp_api.html#_CPPv4NK5c |
| tion)](api/languages/cpp_api.html | udaq5ptsbe19PTSSamplingStrategy20 |
| #_CPPv4N5cudaq16ExecutionContext1 | generateTrajectoriesENSt4spanIKN6 |
| 6ExecutionContextERKNSt6stringE), | detail10NoisePointEEENSt6size_tE) |
|     [\[1\]](api/languages/        | -   [cudaq:                       |
| cpp_api.html#_CPPv4N5cudaq16Execu | :ptsbe::PTSSamplingStrategy::name |
| tionContext16ExecutionContextERKN |     (C++                          |
| St6stringENSt6size_tENSt6size_tE) |     function)](api/langua         |
| -   [cudaq::Execu                 | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| tionContext::explicitMeasurements | tsbe19PTSSamplingStrategy4nameEv) |
|     (C++                          | -   [cudaq::ptsbe::PTSSampli      |
|     member)](api/languages/cp     | ngStrategy::\~PTSSamplingStrategy |
| p_api.html#_CPPv4N5cudaq16Executi |     (C++                          |
| onContext20explicitMeasurementsE) |     function)](api/la             |
| -   [cuda                         | nguages/cpp_api.html#_CPPv4N5cuda |
| q::ExecutionContext::futureResult | q5ptsbe19PTSSamplingStrategyD0Ev) |
|     (C++                          | -   [cudaq::ptsbe::sample (C++    |
|     member)](api/lang             |                                   |
| uages/cpp_api.html#_CPPv4N5cudaq1 |  function)](api/languages/cpp_api |
| 6ExecutionContext12futureResultE) | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| -   [cudaq::ExecutionContext      | mpleE13sample_resultRK14sample_op |
| ::hasConditionalsOnMeasureResults | tionsRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\[1\]](api                   |
|     mem                           | /languages/cpp_api.html#_CPPv4I0D |
| ber)](api/languages/cpp_api.html# | pEN5cudaq5ptsbe6sampleE13sample_r |
| _CPPv4N5cudaq16ExecutionContext31 | esultRKN5cudaq11noise_modelENSt6s |
| hasConditionalsOnMeasureResultsE) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [cudaq:                       | -   [cudaq::ptsbe::sample_async   |
| :ExecutionContext::inKernelLaunch |     (C++                          |
|     (C++                          |     function)](a                  |
|     member)](api/langua           | pi/languages/cpp_api.html#_CPPv4I |
| ges/cpp_api.html#_CPPv4N5cudaq16E | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| xecutionContext14inKernelLaunchE) | 9async_sample_resultRK14sample_op |
| -   [cu                           | tionsRR13QuantumKernelDpRR4Args), |
| daq::ExecutionContext::kernelName |     [\[1\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4I0DpEN5cudaq5pts |
|     member)](api/la               | be12sample_asyncE19async_sample_r |
| nguages/cpp_api.html#_CPPv4N5cuda | esultRKN5cudaq11noise_modelENSt6s |
| q16ExecutionContext10kernelNameE) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [cud                          | -   [cudaq::ptsbe::sample_options |
| aq::ExecutionContext::kernelTrace |     (C++                          |
|     (C++                          |     struct)                       |
|     member)](api/lan              | ](api/languages/cpp_api.html#_CPP |
| guages/cpp_api.html#_CPPv4N5cudaq | v4N5cudaq5ptsbe14sample_optionsE) |
| 16ExecutionContext11kernelTraceE) | -   [cudaq::ptsbe::sample_result  |
| -                                 |     (C++                          |
|    [cudaq::ExecutionContext::name |     class                         |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)]                      | Pv4N5cudaq5ptsbe13sample_resultE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::pts                   |
| 4N5cudaq16ExecutionContext4nameE) | be::sample_result::execution_data |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::noiseModel |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
|     member)](api/la               | 3sample_result14execution_dataEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::ptsbe::               |
| q16ExecutionContext10noiseModelE) | sample_result::has_execution_data |
| -   [cudaq::Exe                   |     (C++                          |
| cutionContext::numberTrajectories |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     member)](api/languages/       | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| cpp_api.html#_CPPv4N5cudaq16Execu | ple_result18has_execution_dataEv) |
| tionContext18numberTrajectoriesE) | -   [cudaq::pt                    |
| -   [c                            | sbe::sample_result::sample_result |
| udaq::ExecutionContext::optResult |     (C++                          |
|     (C++                          |     function)](api/l              |
|     member)](api/                 | anguages/cpp_api.html#_CPPv4N5cud |
| languages/cpp_api.html#_CPPv4N5cu | aq5ptsbe13sample_result13sample_r |
| daq16ExecutionContext9optResultE) | esultERRN5cudaq13sample_resultE), |
| -                                 |                                   |
|   [cudaq::ExecutionContext::qpuId |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq5ptsbe13sample_re |
|     member)](                     | sult13sample_resultERRN5cudaq13sa |
| api/languages/cpp_api.html#_CPPv4 | mple_resultE18PTSBEExecutionData) |
| N5cudaq16ExecutionContext5qpuIdE) | -   [cudaq::ptsbe::               |
| -   [cudaq                        | sample_result::set_execution_data |
| ::ExecutionContext::registerNames |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](api/langu            | languages/cpp_api.html#_CPPv4N5cu |
| ages/cpp_api.html#_CPPv4N5cudaq16 | daq5ptsbe13sample_result18set_exe |
| ExecutionContext13registerNamesE) | cution_dataE18PTSBEExecutionData) |
| -   [cu                           | -   [cud                          |
| daq::ExecutionContext::reorderIdx | aq::ptsbe::ShotAllocationStrategy |
|     (C++                          |     (C++                          |
|     member)](api/la               |     struct)](using                |
| nguages/cpp_api.html#_CPPv4N5cuda | /examples/ptsbe.html#_CPPv4N5cuda |
| q16ExecutionContext10reorderIdxE) | q5ptsbe22ShotAllocationStrategyE) |
| -                                 | -   [cudaq::ptsbe::ShotAllocatio  |
|   [cudaq::ExecutionContext::shots | nStrategy::ShotAllocationStrategy |
|     (C++                          |     (C++                          |
|     member)](                     |     function)                     |
| api/languages/cpp_api.html#_CPPv4 | ](using/examples/ptsbe.html#_CPPv |
| N5cudaq16ExecutionContext5shotsE) | 4N5cudaq5ptsbe22ShotAllocationStr |
| -   [cudaq::                      | ategy22ShotAllocationStrategyE4Ty |
| ExecutionContext::simulationState | pedNSt8optionalINSt8uint64_tEEE), |
|     (C++                          |     [\[1\                         |
|     member)](api/languag          | ]](using/examples/ptsbe.html#_CPP |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | v4N5cudaq5ptsbe22ShotAllocationSt |
| ecutionContext15simulationStateE) | rategy22ShotAllocationStrategyEv) |
| -                                 | -   [cudaq::pt                    |
|    [cudaq::ExecutionContext::spin | sbe::ShotAllocationStrategy::Type |
|     (C++                          |     (C++                          |
|     member)]                      |     enum)](using/exam             |
| (api/languages/cpp_api.html#_CPPv | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| 4N5cudaq16ExecutionContext4spinE) | be22ShotAllocationStrategy4TypeE) |
| -   [cudaq::                      | -   [cudaq::ptsbe::ShotAllocatio  |
| ExecutionContext::totalIterations | nStrategy::Type::HIGH_WEIGHT_BIAS |
|     (C++                          |     (C++                          |
|     member)](api/languag          |     enumerat                      |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | or)](using/examples/ptsbe.html#_C |
| ecutionContext15totalIterationsE) | PPv4N5cudaq5ptsbe22ShotAllocation |
| -   [cudaq::ExecutionResult (C++  | Strategy4Type16HIGH_WEIGHT_BIASE) |
|     st                            | -   [cudaq::ptsbe::ShotAllocati   |
| ruct)](api/languages/cpp_api.html | onStrategy::Type::LOW_WEIGHT_BIAS |
| #_CPPv4N5cudaq15ExecutionResultE) |     (C++                          |
| -   [cud                          |     enumera                       |
| aq::ExecutionResult::appendResult | tor)](using/examples/ptsbe.html#_ |
|     (C++                          | CPPv4N5cudaq5ptsbe22ShotAllocatio |
|     functio                       | nStrategy4Type15LOW_WEIGHT_BIASE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::ptsbe::ShotAlloc      |
| PPv4N5cudaq15ExecutionResult12app | ationStrategy::Type::PROPORTIONAL |
| endResultENSt6stringENSt6size_tE) |     (C++                          |
| -   [cu                           |     enum                          |
| daq::ExecutionResult::deserialize | erator)](using/examples/ptsbe.htm |
|     (C++                          | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
|     function)                     | tionStrategy4Type12PROPORTIONALE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::ptsbe::Shot           |
| v4N5cudaq15ExecutionResult11deser | AllocationStrategy::Type::UNIFORM |
| ializeERNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq:                       |                                   |
| :ExecutionResult::ExecutionResult |   enumerator)](using/examples/pts |
|     (C++                          | be.html#_CPPv4N5cudaq5ptsbe22Shot |
|     functio                       | AllocationStrategy4Type7UNIFORME) |
| n)](api/languages/cpp_api.html#_C | -                                 |
| PPv4N5cudaq15ExecutionResult15Exe |   [cudaq::ptsbe::TraceInstruction |
| cutionResultE16CountsDictionary), |     (C++                          |
|     [\[1\]](api/lan               |     struct)](                     |
| guages/cpp_api.html#_CPPv4N5cudaq | api/languages/cpp_api.html#_CPPv4 |
| 15ExecutionResult15ExecutionResul | N5cudaq5ptsbe16TraceInstructionE) |
| tE16CountsDictionaryNSt6stringE), | -   [cudaq:                       |
|     [\[2\                         | :ptsbe::TraceInstruction::channel |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     member)](api/lang             |
| utionResultE16CountsDictionaryd), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|                                   | ptsbe16TraceInstruction7channelE) |
|    [\[3\]](api/languages/cpp_api. | -   [cudaq::                      |
| html#_CPPv4N5cudaq15ExecutionResu | ptsbe::TraceInstruction::controls |
| lt15ExecutionResultENSt6stringE), |     (C++                          |
|     [\[4\                         |     member)](api/langu            |
| ]](api/languages/cpp_api.html#_CP | ages/cpp_api.html#_CPPv4N5cudaq5p |
| Pv4N5cudaq15ExecutionResult15Exec | tsbe16TraceInstruction8controlsE) |
| utionResultERK15ExecutionResult), | -   [cud                          |
|     [\[5\]](api/language          | aq::ptsbe::TraceInstruction::name |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     (C++                          |
| cutionResult15ExecutionResultEd), |     member)](api/l                |
|     [\[6\]](api/languag           | anguages/cpp_api.html#_CPPv4N5cud |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | aq5ptsbe16TraceInstruction4nameE) |
| ecutionResult15ExecutionResultEv) | -   [cudaq                        |
| -   [                             | ::ptsbe::TraceInstruction::params |
| cudaq::ExecutionResult::operator= |     (C++                          |
|     (C++                          |     member)](api/lan              |
|     function)](api/languages/     | guages/cpp_api.html#_CPPv4N5cudaq |
| cpp_api.html#_CPPv4N5cudaq15Execu | 5ptsbe16TraceInstruction6paramsE) |
| tionResultaSERK15ExecutionResult) | -   [cudaq:                       |
| -   [c                            | :ptsbe::TraceInstruction::targets |
| udaq::ExecutionResult::operator== |     (C++                          |
|     (C++                          |     member)](api/lang             |
|     function)](api/languages/c    | uages/cpp_api.html#_CPPv4N5cudaq5 |
| pp_api.html#_CPPv4NK5cudaq15Execu | ptsbe16TraceInstruction7targetsE) |
| tionResulteqERK15ExecutionResult) | -   [cudaq::ptsbe::T              |
| -   [cud                          | raceInstruction::TraceInstruction |
| aq::ExecutionResult::registerName |     (C++                          |
|     (C++                          |                                   |
|     member)](api/lan              |   function)](api/languages/cpp_ap |
| guages/cpp_api.html#_CPPv4N5cudaq | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| 15ExecutionResult12registerNameE) | Instruction16TraceInstructionE20T |
| -   [cudaq                        | raceInstructionTypeNSt6stringENSt |
| ::ExecutionResult::sequentialData | 6vectorINSt6size_tEEENSt6vectorIN |
|     (C++                          | St6size_tEEENSt6vectorIdEENSt8opt |
|     member)](api/langu            | ionalIN5cudaq13kraus_channelEEE), |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     [\[1\]](api/languages/cpp_a   |
| ExecutionResult14sequentialDataE) | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| -   [                             | eInstruction16TraceInstructionEv) |
| cudaq::ExecutionResult::serialize | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::type |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4NK5cu |     member)](api/l                |
| daq15ExecutionResult9serializeEv) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::fermion_handler (C++  | aq5ptsbe16TraceInstruction4typeE) |
|     c                             | -   [c                            |
| lass)](api/languages/cpp_api.html | udaq::ptsbe::TraceInstructionType |
| #_CPPv4N5cudaq15fermion_handlerE) |     (C++                          |
| -   [cudaq::fermion_op (C++       |     enum)](api/                   |
|     type)](api/languages/cpp_api  | languages/cpp_api.html#_CPPv4N5cu |
| .html#_CPPv4N5cudaq10fermion_opE) | daq5ptsbe20TraceInstructionTypeE) |
| -   [cudaq::fermion_op_term (C++  | -   [cudaq::                      |
|                                   | ptsbe::TraceInstructionType::Gate |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15fermion_op_termE) |     enumerator)](api/langu        |
| -   [cudaq::FermioniqQPU (C++     | ages/cpp_api.html#_CPPv4N5cudaq5p |
|                                   | tsbe20TraceInstructionType4GateE) |
|   class)](api/languages/cpp_api.h | -   [cudaq::ptsbe::               |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | TraceInstructionType::Measurement |
| -   [cudaq::get_state (C++        |     (C++                          |
|                                   |                                   |
|    function)](api/languages/cpp_a |    enumerator)](api/languages/cpp |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| ateEDaRR13QuantumKernelDpRR4Args) | aceInstructionType11MeasurementE) |
| -   [cudaq::GPUEmulatedQPU (C++   | -   [cudaq::p                     |
|                                   | tsbe::TraceInstructionType::Noise |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14GPUEmulatedQPUE) |     enumerator)](api/langua       |
| -   [cudaq::gradient (C++         | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     class)](api/languages/cpp_    | sbe20TraceInstructionType5NoiseE) |
| api.html#_CPPv4N5cudaq8gradientE) | -   [                             |
| -   [cudaq::gradient::clone (C++  | cudaq::ptsbe::TrajectoryPredicate |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     type)](api                    |
| l#_CPPv4N5cudaq8gradient5cloneEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gradient::compute     | udaq5ptsbe19TrajectoryPredicateE) |
|     (C++                          | -   [cudaq::QPU (C++              |
|     function)](api/language       |     class)](api/languages         |
| s/cpp_api.html#_CPPv4N5cudaq8grad | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| ient7computeERKNSt6vectorIdEERKNS | -   [cudaq::QPU::beginExecution   |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|     [\[1\]](ap                    |     function                      |
| i/languages/cpp_api.html#_CPPv4N5 | )](api/languages/cpp_api.html#_CP |
| cudaq8gradient7computeERKNSt6vect | Pv4N5cudaq3QPU14beginExecutionEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cuda                         |
| -   [cudaq::gradient::gradient    | q::QPU::configureExecutionContext |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     funct                         |
| uages/cpp_api.html#_CPPv4I00EN5cu | ion)](api/languages/cpp_api.html# |
| daq8gradient8gradientER7KernelT), | _CPPv4NK5cudaq3QPU25configureExec |
|                                   | utionContextER16ExecutionContext) |
|    [\[1\]](api/languages/cpp_api. | -   [cudaq::QPU::endExecution     |
| html#_CPPv4I00EN5cudaq8gradient8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     functi                        |
|     [\[2\                         | on)](api/languages/cpp_api.html#_ |
| ]](api/languages/cpp_api.html#_CP | CPPv4N5cudaq3QPU12endExecutionEv) |
| Pv4I00EN5cudaq8gradient8gradientE | -   [cudaq::QPU::enqueue (C++     |
| RR13QuantumKernelRR10ArgsMapper), |     function)](ap                 |
|     [\[3                          | i/languages/cpp_api.html#_CPPv4N5 |
| \]](api/languages/cpp_api.html#_C | cudaq3QPU7enqueueER11QuantumTask) |
| PPv4N5cudaq8gradient8gradientERRN | -   [cud                          |
| St8functionIFvNSt6vectorIdEEEEE), | aq::QPU::finalizeExecutionContext |
|     [\[                           |     (C++                          |
| 4\]](api/languages/cpp_api.html#_ |     func                          |
| CPPv4N5cudaq8gradient8gradientEv) | tion)](api/languages/cpp_api.html |
| -   [cudaq::gradient::setArgs     | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     fu                            | -                                 |
| nction)](api/languages/cpp_api.ht | [cudaq::QPU::getExecutionThreadId |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     (C++                          |
| tArgsEvR13QuantumKernelDpRR4Args) |     function)](api/               |
| -   [cudaq::gradient::setKernel   | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq3QPU20getExecutionThreadIdEv) |
|     function)](api/languages/c    | -   [cudaq::QPU::isEmulated (C++  |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     func                          |
| ient9setKernelEvR13QuantumKernel) | tion)](api/languages/cpp_api.html |
| -   [cud                          | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| aq::gradients::central_difference | -   [cudaq::QPU::isSimulator (C++ |
|     (C++                          |     funct                         |
|     class)](api/la                | ion)](api/languages/cpp_api.html# |
| nguages/cpp_api.html#_CPPv4N5cuda | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| q9gradients18central_differenceE) | -   [cudaq::QPU::onRandomSeedSet  |
| -   [cudaq::gra                   |     (C++                          |
| dients::central_difference::clone |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq3 |
|     function)](api/languages      | QPU15onRandomSeedSetENSt6size_tE) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QPU::QPU (C++         |
| ents18central_difference5cloneEv) |     functio                       |
| -   [cudaq::gradi                 | n)](api/languages/cpp_api.html#_C |
| ents::central_difference::compute | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
|     (C++                          |                                   |
|     function)](                   |  [\[1\]](api/languages/cpp_api.ht |
| api/languages/cpp_api.html#_CPPv4 | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| N5cudaq9gradients18central_differ |     [\[2\]](api/languages/cpp_    |
| ence7computeERKNSt6vectorIdEERKNS | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::QPU::setId (C++       |
|                                   |     function                      |
|   [\[1\]](api/languages/cpp_api.h | )](api/languages/cpp_api.html#_CP |
| tml#_CPPv4N5cudaq9gradients18cent | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| ral_difference7computeERKNSt6vect | -   [cudaq::QPU::setShots (C++    |
| orIdEERNSt6vectorIdEERK7spin_opd) |     f                             |
| -   [cudaq::gradie                | unction)](api/languages/cpp_api.h |
| nts::central_difference::gradient | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
|     (C++                          | -   [cudaq::QPU::\~QPU (C++       |
|     functio                       |     function)](api/languages/cp   |
| n)](api/languages/cpp_api.html#_C | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| PPv4I00EN5cudaq9gradients18centra | -   [cudaq::QPUState (C++         |
| l_difference8gradientER7KernelT), |     class)](api/languages/cpp_    |
|     [\[1\]](api/langua            | api.html#_CPPv4N5cudaq8QPUStateE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::qreg (C++             |
| q9gradients18central_difference8g |     class)](api/lan               |
| radientER7KernelTRR10ArgsMapper), | guages/cpp_api.html#_CPPv4I_NSt6s |
|     [\[2\]](api/languages/cpp_    | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| api.html#_CPPv4I00EN5cudaq9gradie | -   [cudaq::qreg::back (C++       |
| nts18central_difference8gradientE |     function)                     |
| RR13QuantumKernelRR10ArgsMapper), | ](api/languages/cpp_api.html#_CPP |
|     [\[3\]](api/languages/cpp     | v4N5cudaq4qreg4backENSt6size_tE), |
| _api.html#_CPPv4N5cudaq9gradients |     [\[1\]](api/languages/cpp_ap  |
| 18central_difference8gradientERRN | i.html#_CPPv4N5cudaq4qreg4backEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::qreg::begin (C++      |
|     [\[4\]](api/languages/cp      |                                   |
| p_api.html#_CPPv4N5cudaq9gradient |  function)](api/languages/cpp_api |
| s18central_difference8gradientEv) | .html#_CPPv4N5cudaq4qreg5beginEv) |
| -   [cud                          | -   [cudaq::qreg::clear (C++      |
| aq::gradients::forward_difference |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     class)](api/la                | .html#_CPPv4N5cudaq4qreg5clearEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::qreg::front (C++      |
| q9gradients18forward_differenceE) |     function)]                    |
| -   [cudaq::gra                   | (api/languages/cpp_api.html#_CPPv |
| dients::forward_difference::clone | 4N5cudaq4qreg5frontENSt6size_tE), |
|     (C++                          |     [\[1\]](api/languages/cpp_api |
|     function)](api/languages      | .html#_CPPv4N5cudaq4qreg5frontEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::qreg::operator\[\]    |
| ents18forward_difference5cloneEv) |     (C++                          |
| -   [cudaq::gradi                 |     functi                        |
| ents::forward_difference::compute | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq4qregixEKNSt6size_tE) |
|     function)](                   | -   [cudaq::qreg::qreg (C++       |
| api/languages/cpp_api.html#_CPPv4 |     function)                     |
| N5cudaq9gradients18forward_differ | ](api/languages/cpp_api.html#_CPP |
| ence7computeERKNSt6vectorIdEERKNS | v4N5cudaq4qreg4qregENSt6size_tE), |
| t8functionIFdNSt6vectorIdEEEEEd), |     [\[1\]](api/languages/cpp_ap  |
|                                   | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::qreg::size (C++       |
| tml#_CPPv4N5cudaq9gradients18forw |                                   |
| ard_difference7computeERKNSt6vect |  function)](api/languages/cpp_api |
| orIdEERNSt6vectorIdEERK7spin_opd) | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| -   [cudaq::gradie                | -   [cudaq::qreg::slice (C++      |
| nts::forward_difference::gradient |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq4q |
|     functio                       | reg5sliceENSt6size_tENSt6size_tE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::qreg::value_type (C++ |
| PPv4I00EN5cudaq9gradients18forwar |                                   |
| d_difference8gradientER7KernelT), | type)](api/languages/cpp_api.html |
|     [\[1\]](api/langua            | #_CPPv4N5cudaq4qreg10value_typeE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::qspan (C++            |
| q9gradients18forward_difference8g |     class)](api/lang              |
| radientER7KernelTRR10ArgsMapper), | uages/cpp_api.html#_CPPv4I_NSt6si |
|     [\[2\]](api/languages/cpp_    | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| api.html#_CPPv4I00EN5cudaq9gradie | -   [cudaq::QuakeValue (C++       |
| nts18forward_difference8gradientE |     class)](api/languages/cpp_api |
| RR13QuantumKernelRR10ArgsMapper), | .html#_CPPv4N5cudaq10QuakeValueE) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::Q                     |
| _api.html#_CPPv4N5cudaq9gradients | uakeValue::canValidateNumElements |
| 18forward_difference8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](api/languages      |
|     [\[4\]](api/languages/cp      | /cpp_api.html#_CPPv4N5cudaq10Quak |
| p_api.html#_CPPv4N5cudaq9gradient | eValue22canValidateNumElementsEv) |
| s18forward_difference8gradientEv) | -                                 |
| -   [                             |  [cudaq::QuakeValue::constantSize |
| cudaq::gradients::parameter_shift |     (C++                          |
|     (C++                          |     function)](api                |
|     class)](api                   | /languages/cpp_api.html#_CPPv4N5c |
| /languages/cpp_api.html#_CPPv4N5c | udaq10QuakeValue12constantSizeEv) |
| udaq9gradients15parameter_shiftE) | -   [cudaq::QuakeValue::dump (C++ |
| -   [cudaq::                      |     function)](api/lan            |
| gradients::parameter_shift::clone | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 10QuakeValue4dumpERNSt7ostreamE), |
|     function)](api/langua         |     [\                            |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | [1\]](api/languages/cpp_api.html# |
| adients15parameter_shift5cloneEv) | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| -   [cudaq::gr                    | -   [cudaq                        |
| adients::parameter_shift::compute | ::QuakeValue::getRequiredElements |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/langua         |
| )](api/languages/cpp_api.html#_CP | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| Pv4N5cudaq9gradients15parameter_s | uakeValue19getRequiredElementsEv) |
| hift7computeERKNSt6vectorIdEERKNS | -   [cudaq::QuakeValue::getValue  |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|     [\[1\]](api/languages/cpp_ap  |     function)]                    |
| i.html#_CPPv4N5cudaq9gradients15p | (api/languages/cpp_api.html#_CPPv |
| arameter_shift7computeERKNSt6vect | 4NK5cudaq10QuakeValue8getValueEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QuakeValue::inverse   |
| -   [cudaq::gra                   |     (C++                          |
| dients::parameter_shift::gradient |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     func                          | v4NK5cudaq10QuakeValue7inverseEv) |
| tion)](api/languages/cpp_api.html | -                                 |
| #_CPPv4I00EN5cudaq9gradients15par |    [cudaq::QuakeValue::isSequence |
| ameter_shift8gradientER7KernelT), |     (C++                          |
|     [\[1\]](api/lan               |     function)](a                  |
| guages/cpp_api.html#_CPPv4I00EN5c | pi/languages/cpp_api.html#_CPPv4N |
| udaq9gradients15parameter_shift8g | 5cudaq10QuakeValue10isSequenceEv) |
| radientER7KernelTRR10ArgsMapper), | -                                 |
|     [\[2\]](api/languages/c       |    [cudaq::QuakeValue::operator\* |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     (C++                          |
| dients15parameter_shift8gradientE |     function)](api                |
| RR13QuantumKernelRR10ArgsMapper), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[3\]](api/languages/        | udaq10QuakeValuemlE10QuakeValue), |
| cpp_api.html#_CPPv4N5cudaq9gradie |                                   |
| nts15parameter_shift8gradientERRN | [\[1\]](api/languages/cpp_api.htm |
| St8functionIFvNSt6vectorIdEEEEE), | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|     [\[4\]](api/languages         | -   [cudaq::QuakeValue::operator+ |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents15parameter_shift8gradientEv) |     function)](api                |
| -   [cudaq::kernel_builder (C++   | /languages/cpp_api.html#_CPPv4N5c |
|     clas                          | udaq10QuakeValueplE10QuakeValue), |
| s)](api/languages/cpp_api.html#_C |     [                             |
| PPv4IDpEN5cudaq14kernel_builderE) | \[1\]](api/languages/cpp_api.html |
| -   [c                            | #_CPPv4N5cudaq10QuakeValueplEKd), |
| udaq::kernel_builder::constantVal |                                   |
|     (C++                          | [\[2\]](api/languages/cpp_api.htm |
|     function)](api/la             | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QuakeValue::operator- |
| q14kernel_builder11constantValEd) |     (C++                          |
| -                                 |     function)](api                |
|  [cudaq::kernel_builder::detector | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuemiE10QuakeValue), |
|                                   |     [                             |
|    function)](api/languages/cpp_a | \[1\]](api/languages/cpp_api.html |
| pi.html#_CPPv4IDpEN5cudaq14kernel | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| _builder8detectorEvDpRR8MeasArgs) |     [                             |
| -                                 | \[2\]](api/languages/cpp_api.html |
| [cudaq::kernel_builder::detectors | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     (C++                          |                                   |
|     func                          | [\[3\]](api/languages/cpp_api.htm |
| tion)](api/languages/cpp_api.html | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| #_CPPv4N5cudaq14kernel_builder9de | -   [cudaq::QuakeValue::operator/ |
| tectorsE10QuakeValue10QuakeValue) |     (C++                          |
| -   [cu                           |     function)](api                |
| daq::kernel_builder::getArguments | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuedvE10QuakeValue), |
|     function)](api/lan            |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq | [\[1\]](api/languages/cpp_api.htm |
| 14kernel_builder12getArgumentsEv) | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| -   [cu                           | -                                 |
| daq::kernel_builder::getNumParams |  [cudaq::QuakeValue::operator\[\] |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)](api                |
| guages/cpp_api.html#_CPPv4N5cudaq | /languages/cpp_api.html#_CPPv4N5c |
| 14kernel_builder12getNumParamsEv) | udaq10QuakeValueixEKNSt6size_tE), |
| -   [cud                          |     [\[1\]](api/                  |
| aq::kernel_builder::isArgSequence | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq10QuakeValueixERK10QuakeValue) |
|     function)](api/languages/cpp_ | -                                 |
| api.html#_CPPv4N5cudaq14kernel_bu |    [cudaq::QuakeValue::QuakeValue |
| ilder13isArgSequenceENSt6size_tE) |     (C++                          |
| -   [cuda                         |     function)](api/languag        |
| q::kernel_builder::kernel_builder | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     (C++                          | akeValue10QuakeValueERN4mlir20Imp |
|     function)](api/languages/cpp  | licitLocOpBuilderEN4mlir5ValueE), |
| _api.html#_CPPv4N5cudaq14kernel_b |     [\[1\]                        |
| uilder14kernel_builderERNSt6vecto | ](api/languages/cpp_api.html#_CPP |
| rIN6detail17KernelBuilderTypeEEE) | v4N5cudaq10QuakeValue10QuakeValue |
| -   [cudaq::k                     | ERN4mlir20ImplicitLocOpBuilderEd) |
| ernel_builder::logical_observable | -   [cudaq::QuakeValue::size (C++ |
|     (C++                          |     funct                         |
|     function)                     | ion)](api/languages/cpp_api.html# |
| ](api/languages/cpp_api.html#_CPP | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| v4IDpEN5cudaq14kernel_builder18lo | -   [cudaq::QuakeValue::slice     |
| gical_observableEvDpRR8MeasArgs), |     (C++                          |
|     [\[1\]](ap                    |     function)](api/languages/cpp_ |
| i/languages/cpp_api.html#_CPPv4N5 | api.html#_CPPv4N5cudaq10QuakeValu |
| cudaq14kernel_builder18logical_ob | e5sliceEKNSt6size_tEKNSt6size_tE) |
| servableE10QuakeValueNSt6size_tE) | -   [cudaq::quantum_platform (C++ |
| -   [cudaq::kernel_builder::name  |     cl                            |
|     (C++                          | ass)](api/languages/cpp_api.html# |
|     function)                     | _CPPv4N5cudaq16quantum_platformE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq:                       |
| v4N5cudaq14kernel_builder4nameEv) | :quantum_platform::beginExecution |
| -                                 |     (C++                          |
|    [cudaq::kernel_builder::qalloc |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     function)](api/language       | antum_platform14beginExecutionEv) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::quantum_pl            |
| nel_builder6qallocE10QuakeValue), | atform::configureExecutionContext |
|     [\[1\]](api/language          |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     function)](api/lang           |
| nel_builder6qallocEKNSt6size_tE), | uages/cpp_api.html#_CPPv4NK5cudaq |
|     [\[2                          | 16quantum_platform25configureExec |
| \]](api/languages/cpp_api.html#_C | utionContextER16ExecutionContext) |
| PPv4N5cudaq14kernel_builder6qallo | -   [cuda                         |
| cERNSt6vectorINSt7complexIdEEEE), | q::quantum_platform::endExecution |
|     [\[3\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/langu          |
| N5cudaq14kernel_builder6qallocEv) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [cudaq::kernel_builder::swap  | quantum_platform12endExecutionEv) |
|     (C++                          | -   [cudaq::q                     |
|     function)](api/language       | uantum_platform::enqueueAsyncTask |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     (C++                          |
| 4kernel_builder4swapEvRK10QuakeVa |     function)](api/languages/     |
| lueRK10QuakeValueRK10QuakeValue), | cpp_api.html#_CPPv4N5cudaq16quant |
|                                   | um_platform16enqueueAsyncTaskEKNS |
| [\[1\]](api/languages/cpp_api.htm | t6size_tER19KernelExecutionTask), |
| l#_CPPv4I00EN5cudaq14kernel_build |     [\[1\]](api/languag           |
| er4swapEvRKNSt6vectorI10QuakeValu | es/cpp_api.html#_CPPv4N5cudaq16qu |
| eEERK10QuakeValueRK10QuakeValue), | antum_platform16enqueueAsyncTaskE |
|                                   | KNSt6size_tERNSt8functionIFvvEEE) |
| [\[2\]](api/languages/cpp_api.htm | -   [cudaq::quantum_p             |
| l#_CPPv4N5cudaq14kernel_builder4s | latform::finalizeExecutionContext |
| wapERK10QuakeValueRK10QuakeValue) |     (C++                          |
| -   [cudaq::KernelExecutionTask   |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq16quant |
|     type                          | um_platform24finalizeExecutionCon |
| )](api/languages/cpp_api.html#_CP | textERN5cudaq16ExecutionContextE) |
| Pv4N5cudaq19KernelExecutionTaskE) | -   [cudaq::qua                   |
| -   [cudaq::KernelThunkResultType | ntum_platform::get_codegen_config |
|     (C++                          |     (C++                          |
|     struct)]                      |     function)](api/languages/c    |
| (api/languages/cpp_api.html#_CPPv | pp_api.html#_CPPv4N5cudaq16quantu |
| 4N5cudaq21KernelThunkResultTypeE) | m_platform18get_codegen_configEv) |
| -   [cudaq::KernelThunkType (C++  | -   [cuda                         |
|                                   | q::quantum_platform::get_exec_ctx |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     function)](api/langua         |
| -   [cudaq::kraus_channel (C++    | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|                                   | quantum_platform12get_exec_ctxEv) |
|  class)](api/languages/cpp_api.ht | -   [c                            |
| ml#_CPPv4N5cudaq13kraus_channelE) | udaq::quantum_platform::get_noise |
| -   [cudaq::kraus_channel::empty  |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)]                    | pp_api.html#_CPPv4N5cudaq16quantu |
| (api/languages/cpp_api.html#_CPPv | m_platform9get_noiseENSt6size_tE) |
| 4NK5cudaq13kraus_channel5emptyEv) | -   [cudaq::qua                   |
| -   [cudaq::kraus_c               | ntum_platform::get_runtime_target |
| hannel::generateUnitaryParameters |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|                                   | p_api.html#_CPPv4NK5cudaq16quantu |
|    function)](api/languages/cpp_a | m_platform18get_runtime_targetEv) |
| pi.html#_CPPv4N5cudaq13kraus_chan | -   [cud                          |
| nel25generateUnitaryParametersEv) | aq::quantum_platform::is_emulated |
| -                                 |     (C++                          |
|    [cudaq::kraus_channel::get_ops |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     function)](a                  | pi.html#_CPPv4NK5cudaq16quantum_p |
| pi/languages/cpp_api.html#_CPPv4N | latform11is_emulatedENSt6size_tE) |
| K5cudaq13kraus_channel7get_opsEv) | -   [cudaq::                      |
| -   [cud                          | quantum_platform::is_library_mode |
| aq::kraus_channel::identity_flags |     (C++                          |
|     (C++                          |     function)](api/languages      |
|     member)](api/lan              | /cpp_api.html#_CPPv4NK5cudaq16qua |
| guages/cpp_api.html#_CPPv4N5cudaq | ntum_platform15is_library_modeEv) |
| 13kraus_channel14identity_flagsE) | -   [c                            |
| -   [cud                          | udaq::quantum_platform::is_remote |
| aq::kraus_channel::is_identity_op |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|                                   | p_api.html#_CPPv4NK5cudaq16quantu |
|    function)](api/languages/cpp_a | m_platform9is_remoteENSt6size_tE) |
| pi.html#_CPPv4NK5cudaq13kraus_cha | -   [cuda                         |
| nnel14is_identity_opENSt6size_tE) | q::quantum_platform::is_simulator |
| -   [cudaq::                      |     (C++                          |
| kraus_channel::is_unitary_mixture |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)](api/languages      | i.html#_CPPv4NK5cudaq16quantum_pl |
| /cpp_api.html#_CPPv4NK5cudaq13kra | atform12is_simulatorENSt6size_tE) |
| us_channel18is_unitary_mixtureEv) | -   [cudaq:                       |
| -   [cu                           | :quantum_platform::list_platforms |
| daq::kraus_channel::kraus_channel |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/lang           | es/cpp_api.html#_CPPv4N5cudaq16qu |
| uages/cpp_api.html#_CPPv4IDpEN5cu | antum_platform14list_platformsEv) |
| daq13kraus_channel13kraus_channel | -                                 |
| EDpRRNSt16initializer_listI1TEE), |    [cudaq::quantum_platform::name |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     function)](a                  |
| ml#_CPPv4N5cudaq13kraus_channel13 | pi/languages/cpp_api.html#_CPPv4N |
| kraus_channelERK13kraus_channel), | K5cudaq16quantum_platform4nameEv) |
|     [\[2\]                        | -   [                             |
| ](api/languages/cpp_api.html#_CPP | cudaq::quantum_platform::num_qpus |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERKNSt6vectorI8kraus_opEE), |     function)](api/l              |
|     [\[3\]                        | anguages/cpp_api.html#_CPPv4NK5cu |
| ](api/languages/cpp_api.html#_CPP | daq16quantum_platform8num_qpusEv) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::                      |
| hannelERRNSt6vectorI8kraus_opEE), | quantum_platform::onRandomSeedSet |
|     [\[4\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 13kraus_channel13kraus_channelEv) | function)](api/languages/cpp_api. |
| -                                 | html#_CPPv4N5cudaq16quantum_platf |
| [cudaq::kraus_channel::noise_type | orm15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     member)](api                  | :quantum_platform::reset_exec_ctx |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel10noise_typeE) |     function)](api/languag        |
| -                                 | es/cpp_api.html#_CPPv4N5cudaq16qu |
|   [cudaq::kraus_channel::op_names | antum_platform14reset_exec_ctxEv) |
|     (C++                          | -   [cud                          |
|     member)](                     | aq::quantum_platform::reset_noise |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq13kraus_channel8op_namesE) |     function)](api/languages/cpp_ |
| -                                 | api.html#_CPPv4N5cudaq16quantum_p |
|  [cudaq::kraus_channel::operator= | latform11reset_noiseENSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     function)](api/langua         | q::quantum_platform::set_exec_ctx |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     (C++                          |
| raus_channelaSERK13kraus_channel) |     funct                         |
| -   [c                            | ion)](api/languages/cpp_api.html# |
| udaq::kraus_channel::operator\[\] | _CPPv4N5cudaq16quantum_platform12 |
|     (C++                          | set_exec_ctxEP16ExecutionContext) |
|     function)](api/l              | -   [c                            |
| anguages/cpp_api.html#_CPPv4N5cud | udaq::quantum_platform::set_noise |
| aq13kraus_channelixEKNSt6size_tE) |     (C++                          |
| -                                 |     function                      |
| [cudaq::kraus_channel::parameters | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq16quantum_platform9set_ |
|     member)](api                  | noiseEPK11noise_modelNSt6size_tE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::quantum_platfor       |
| udaq13kraus_channel10parametersE) | m::supports_explicit_measurements |
| -   [cudaq::krau                  |     (C++                          |
| s_channel::populateDefaultOpNames |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     function)](api/languages/cp   | daq16quantum_platform30supports_e |
| p_api.html#_CPPv4N5cudaq13kraus_c | xplicit_measurementsENSt6size_tE) |
| hannel22populateDefaultOpNamesEv) | -   [cuda                         |
| -   [cu                           | q::quantum_platform::supports_jit |
| daq::kraus_channel::probabilities |     (C++                          |
|     (C++                          |                                   |
|     member)](api/la               |   function)](api/languages/cpp_ap |
| nguages/cpp_api.html#_CPPv4N5cuda | i.html#_CPPv4NK5cudaq16quantum_pl |
| q13kraus_channel13probabilitiesE) | atform12supports_jitENSt6size_tE) |
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
| -   [cudaq::kraus_op::adjoint     | -   [cudaq::qudit (C++            |
|     (C++                          |     clas                          |
|     functi                        | s)](api/languages/cpp_api.html#_C |
| on)](api/languages/cpp_api.html#_ | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| CPPv4NK5cudaq8kraus_op7adjointEv) | -   [cudaq::qudit::qudit (C++     |
| -   [cudaq::kraus_op::data (C++   |                                   |
|                                   | function)](api/languages/cpp_api. |
|  member)](api/languages/cpp_api.h | html#_CPPv4N5cudaq5qudit5quditEv) |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | -   [cudaq::QuEraRemoteRESTQPU    |
| -   [cudaq::kraus_op::kraus_op    |     (C++                          |
|     (C++                          |     clas                          |
|     func                          | s)](api/languages/cpp_api.html#_C |
| tion)](api/languages/cpp_api.html | PPv4N5cudaq18QuEraRemoteRESTQPUE) |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | -   [cudaq::qvector (C++          |
| opERRNSt16initializer_listI1TEE), |     class)                        |
|                                   | ](api/languages/cpp_api.html#_CPP |
|  [\[1\]](api/languages/cpp_api.ht | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | -   [cudaq::qvector::back (C++    |
| pENSt6vectorIN5cudaq7complexEEE), |     function)](a                  |
|     [\[2\]](api/l                 | pi/languages/cpp_api.html#_CPPv4N |
| anguages/cpp_api.html#_CPPv4N5cud | 5cudaq7qvector4backENSt6size_tE), |
| aq8kraus_op8kraus_opERK8kraus_op) |                                   |
| -   [cudaq::kraus_op::nCols (C++  |   [\[1\]](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq7qvector4backEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::qvector::begin (C++   |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     fu                            |
| -   [cudaq::kraus_op::nRows (C++  | nction)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4N5cudaq7qvector5beginEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::qvector::clear (C++   |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) |     fu                            |
| -   [cudaq::kraus_op::operator=   | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     function)                     | -   [cudaq::qvector::end (C++     |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4N5cudaq8kraus_opaSERK8kraus_op) | function)](api/languages/cpp_api. |
| -   [cudaq::kraus_op::precision   | html#_CPPv4N5cudaq7qvector3endEv) |
|     (C++                          | -   [cudaq::qvector::front (C++   |
|     memb                          |     function)](ap                 |
| er)](api/languages/cpp_api.html#_ | i/languages/cpp_api.html#_CPPv4N5 |
| CPPv4N5cudaq8kraus_op9precisionE) | cudaq7qvector5frontENSt6size_tE), |
| -   [cudaq::KrausSelection (C++   |                                   |
|     s                             |  [\[1\]](api/languages/cpp_api.ht |
| truct)](api/languages/cpp_api.htm | ml#_CPPv4N5cudaq7qvector5frontEv) |
| l#_CPPv4N5cudaq14KrausSelectionE) | -   [cudaq::qvector::operator=    |
| -   [cudaq:                       |     (C++                          |
| :KrausSelection::circuit_location |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](api/langua           | PPv4N5cudaq7qvectoraSERK7qvector) |
| ges/cpp_api.html#_CPPv4N5cudaq14K | -   [cudaq::qvector::operator\[\] |
| rausSelection16circuit_locationE) |     (C++                          |
| -                                 |     function)                     |
|  [cudaq::KrausSelection::is_error | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     member)](a                    | -   [cudaq::qvector::qvector (C++ |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/               |
| 5cudaq14KrausSelection8is_errorE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::Kra                   | daq7qvector7qvectorENSt6size_tE), |
| usSelection::kraus_operator_index |     [\[1\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     member)](api/languages/       | 5cudaq7qvector7qvectorERK5state), |
| cpp_api.html#_CPPv4N5cudaq14Kraus |     [\[2\]](api                   |
| Selection20kraus_operator_indexE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cuda                         | udaq7qvector7qvectorERK7qvector), |
| q::KrausSelection::KrausSelection |     [\[3\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](a                  | cudaq7qvector7qvectorERR7qvector) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::qvector::size (C++    |
| 5cudaq14KrausSelection14KrausSele |     fu                            |
| ctionENSt6size_tENSt6vectorINSt6s | nction)](api/languages/cpp_api.ht |
| ize_tEEENSt6stringENSt6size_tEb), | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     [\[1\]](api/langu             | -   [cudaq::qvector::slice (C++   |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     function)](api/language       |
| KrausSelection14KrausSelectionEv) | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| -                                 | tor5sliceENSt6size_tENSt6size_tE) |
|   [cudaq::KrausSelection::op_name | -   [cudaq::qvector::value_type   |
|     (C++                          |     (C++                          |
|     member)](                     |     typ                           |
| api/languages/cpp_api.html#_CPPv4 | e)](api/languages/cpp_api.html#_C |
| N5cudaq14KrausSelection7op_nameE) | PPv4N5cudaq7qvector10value_typeE) |
| -   [                             | -   [cudaq::qview (C++            |
| cudaq::KrausSelection::operator== |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     function)](api/languages      | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | -   [cudaq::qview::back (C++      |
| usSelectioneqERK14KrausSelection) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|    [cudaq::KrausSelection::qubits | v4N5cudaq5qview4backENSt6size_tE) |
|     (C++                          | -   [cudaq::qview::begin (C++     |
|     member)]                      |                                   |
| (api/languages/cpp_api.html#_CPPv | function)](api/languages/cpp_api. |
| 4N5cudaq14KrausSelection6qubitsE) | html#_CPPv4N5cudaq5qview5beginEv) |
| -   [cudaq::KrausTrajectory (C++  | -   [cudaq::qview::end (C++       |
|     st                            |                                   |
| ruct)](api/languages/cpp_api.html |   function)](api/languages/cpp_ap |
| #_CPPv4N5cudaq15KrausTrajectoryE) | i.html#_CPPv4N5cudaq5qview3endEv) |
| -                                 | -   [cudaq::qview::front (C++     |
|  [cudaq::KrausTrajectory::builder |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](ap                 | N5cudaq5qview5frontENSt6size_tE), |
| i/languages/cpp_api.html#_CPPv4N5 |                                   |
| cudaq15KrausTrajectory7builderEv) |    [\[1\]](api/languages/cpp_api. |
| -   [cu                           | html#_CPPv4N5cudaq5qview5frontEv) |
| daq::KrausTrajectory::countErrors | -   [cudaq::qview::operator\[\]   |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     functio                       |
| uages/cpp_api.html#_CPPv4NK5cudaq | n)](api/languages/cpp_api.html#_C |
| 15KrausTrajectory11countErrorsEv) | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| -   [                             | -   [cudaq::qview::qview (C++     |
| cudaq::KrausTrajectory::isOrdered |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)](api/l              | PPv4I0EN5cudaq5qview5qviewERR1R), |
| anguages/cpp_api.html#_CPPv4NK5cu |     [\[1                          |
| daq15KrausTrajectory9isOrderedEv) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::                      | PPv4N5cudaq5qview5qviewERK5qview) |
| KrausTrajectory::kraus_selections | -   [cudaq::qview::size (C++      |
|     (C++                          |                                   |
|     member)](api/languag          | function)](api/languages/cpp_api. |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | html#_CPPv4NK5cudaq5qview4sizeEv) |
| ausTrajectory16kraus_selectionsE) | -   [cudaq::qview::slice (C++     |
| -   [cudaq:                       |     function)](api/langua         |
| :KrausTrajectory::KrausTrajectory | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|     (C++                          | iew5sliceENSt6size_tENSt6size_tE) |
|     function                      | -   [cudaq::qview::value_type     |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15KrausTrajectory15Krau |     t                             |
| sTrajectoryENSt6size_tENSt6vector | ype)](api/languages/cpp_api.html# |
| I14KrausSelectionEEdNSt6size_tE), | _CPPv4N5cudaq5qview10value_typeE) |
|     [\[1\]](api/languag           | -   [cudaq::range (C++            |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |     fun                           |
| ausTrajectory15KrausTrajectoryEv) | ction)](api/languages/cpp_api.htm |
| -   [cudaq::Kr                    | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| ausTrajectory::measurement_counts | orI11ElementTypeEE11ElementType), |
|     (C++                          |     [\[1\]](api/languages/cpp_    |
|     member)](api/languages        | api.html#_CPPv4I0EN5cudaq5rangeEN |
| /cpp_api.html#_CPPv4N5cudaq15Krau | St6vectorI11ElementTypeEE11Elemen |
| sTrajectory18measurement_countsE) | tType11ElementType11ElementType), |
| -   [cud                          |     [                             |
| aq::KrausTrajectory::multiplicity | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq5rangeENSt6size_tE) |
|     member)](api/lan              | -   [cudaq::real (C++             |
| guages/cpp_api.html#_CPPv4N5cudaq |     type)](api/languages/         |
| 15KrausTrajectory12multiplicityE) | cpp_api.html#_CPPv4N5cudaq4realE) |
| -   [                             | -   [cudaq::registry (C++         |
| cudaq::KrausTrajectory::num_shots |     type)](api/languages/cpp_     |
|     (C++                          | api.html#_CPPv4N5cudaq8registryE) |
|     member)](api                  | -                                 |
| /languages/cpp_api.html#_CPPv4N5c |  [cudaq::registry::RegisteredType |
| udaq15KrausTrajectory9num_shotsE) |     (C++                          |
| -   [c                            |     class)](api/                  |
| udaq::KrausTrajectory::operator== | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq8registry14RegisteredTypeE) |
|     function)](api/languages/c    | -   [cudaq::RemoteRESTQPU (C++    |
| pp_api.html#_CPPv4NK5cudaq15Kraus |                                   |
| TrajectoryeqERK15KrausTrajectory) |  class)](api/languages/cpp_api.ht |
| -   [cu                           | ml#_CPPv4N5cudaq13RemoteRESTQPUE) |
| daq::KrausTrajectory::probability | -   [cudaq::Resources (C++        |
|     (C++                          |     class)](api/languages/cpp_a   |
|     member)](api/la               | pi.html#_CPPv4N5cudaq9ResourcesE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::run (C++              |
| q15KrausTrajectory11probabilityE) |     function)]                    |
| -   [cuda                         | (api/languages/cpp_api.html#_CPPv |
| q::KrausTrajectory::trajectory_id | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
|     (C++                          | 5invoke_result_tINSt7decay_tI13Qu |
|     member)](api/lang             | antumKernelEEDpNSt7decay_tI4ARGSE |
| uages/cpp_api.html#_CPPv4N5cudaq1 | EEEEENSt6size_tERN5cudaq11noise_m |
| 5KrausTrajectory13trajectory_idE) | odelERR13QuantumKernelDpRR4ARGS), |
| -                                 |     [\[1\]](api/langu             |
|   [cudaq::KrausTrajectory::weight | ages/cpp_api.html#_CPPv4I0DpEN5cu |
|     (C++                          | daq3runENSt6vectorINSt15invoke_re |
|     member)](                     | sult_tINSt7decay_tI13QuantumKerne |
| api/languages/cpp_api.html#_CPPv4 | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| N5cudaq15KrausTrajectory6weightE) | ize_tERR13QuantumKernelDpRR4ARGS) |
| -                                 | -   [cudaq::run_async (C++        |
|    [cudaq::KrausTrajectoryBuilder |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     class)](                      | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| api/languages/cpp_api.html#_CPPv4 | tureINSt6vectorINSt15invoke_resul |
| N5cudaq22KrausTrajectoryBuilderE) | t_tINSt7decay_tI13QuantumKernelEE |
| -   [cud                          | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| aq::KrausTrajectoryBuilder::build | ze_tENSt6size_tERN5cudaq11noise_m |
|     (C++                          | odelERR13QuantumKernelDpRR4ARGS), |
|     function)](api/lang           |     [\[1\]](api/la                |
| uages/cpp_api.html#_CPPv4NK5cudaq | nguages/cpp_api.html#_CPPv4I0DpEN |
| 22KrausTrajectoryBuilder5buildEv) | 5cudaq9run_asyncENSt6futureINSt6v |
| -   [cud                          | ectorINSt15invoke_result_tINSt7de |
| aq::KrausTrajectoryBuilder::setId | cay_tI13QuantumKernelEEDpNSt7deca |
|     (C++                          | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     function)](api/languages/cpp  | ize_tERR13QuantumKernelDpRR4ARGS) |
| _api.html#_CPPv4N5cudaq22KrausTra | -   [cudaq::RuntimeTarget (C++    |
| jectoryBuilder5setIdENSt6size_tE) |                                   |
| -   [cudaq::Kraus                 | struct)](api/languages/cpp_api.ht |
| TrajectoryBuilder::setProbability | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     (C++                          | -   [cudaq::sample (C++           |
|     function)](api/languages/cpp  |     function)](api/languages/c    |
| _api.html#_CPPv4N5cudaq22KrausTra | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| jectoryBuilder14setProbabilityEd) | mpleE13sample_resultRK14sample_op |
| -   [cudaq::Krau                  | tionsRR13QuantumKernelDpRR4Args), |
| sTrajectoryBuilder::setSelections |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/languag        | Pv4I0DpEN5cudaq6sampleE13sample_r |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | esultRR13QuantumKernelDpRR4Args), |
| ausTrajectoryBuilder13setSelectio |     [\                            |
| nsENSt6vectorI14KrausSelectionEE) | [2\]](api/languages/cpp_api.html# |
| -   [cudaq::logical_observable    | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4Args) |
|     function)](api/languages/c    | -   [cudaq::sample_options (C++   |
| pp_api.html#_CPPv4IDpEN5cudaq18lo |     s                             |
| gical_observableEvDpRR8MeasArgs), | truct)](api/languages/cpp_api.htm |
|     [\[1\]](api/l                 | l#_CPPv4N5cudaq14sample_optionsE) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cudaq::sample_result (C++    |
| aq18logical_observableERKNSt6vect |                                   |
| orI14measure_resultEENSt6size_tE) |  class)](api/languages/cpp_api.ht |
| -   [cudaq::M2DSparseMatrix (C++  | ml#_CPPv4N5cudaq13sample_resultE) |
|     st                            | -   [cudaq::sample_result::append |
| ruct)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15M2DSparseMatrixE) |     function)](api/languages/cpp_ |
| -   [cudaq::M2OSparseMatrix (C++  | api.html#_CPPv4N5cudaq13sample_re |
|     st                            | sult6appendERK15ExecutionResultb) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::sample_result::begin  |
| #_CPPv4N5cudaq15M2OSparseMatrixE) |     (C++                          |
| -   [cudaq::matrix_callback (C++  |     function)]                    |
|     c                             | (api/languages/cpp_api.html#_CPPv |
| lass)](api/languages/cpp_api.html | 4N5cudaq13sample_result5beginEv), |
| #_CPPv4N5cudaq15matrix_callbackE) |     [\[1\]]                       |
| -   [cudaq::matrix_handler (C++   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq13sample_result5beginEv) |
| class)](api/languages/cpp_api.htm | -   [cudaq::sample_result::cbegin |
| l#_CPPv4N5cudaq14matrix_handlerE) |     (C++                          |
| -   [cudaq::mat                   |     function)](                   |
| rix_handler::commutation_behavior | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq13sample_result6cbeginEv) |
|     struct)](api/languages/       | -   [cudaq::sample_result::cend   |
| cpp_api.html#_CPPv4N5cudaq14matri |     (C++                          |
| x_handler20commutation_behaviorE) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|    [cudaq::matrix_handler::define | v4NK5cudaq13sample_result4cendEv) |
|     (C++                          | -   [cudaq::sample_result::clear  |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)                     |
| 5cudaq14matrix_handler6defineENSt | ](api/languages/cpp_api.html#_CPP |
| 6stringENSt6vectorINSt7int64_tEEE | v4N5cudaq13sample_result5clearEv) |
| RR15matrix_callbackRKNSt13unorder | -   [cudaq::sample_result::count  |
| ed_mapINSt6stringENSt6stringEEE), |     (C++                          |
|                                   |     function)](                   |
| [\[1\]](api/languages/cpp_api.htm | api/languages/cpp_api.html#_CPPv4 |
| l#_CPPv4N5cudaq14matrix_handler6d | NK5cudaq13sample_result5countENSt |
| efineENSt6stringENSt6vectorINSt7i | 11string_viewEKNSt11string_viewE) |
| nt64_tEEERR15matrix_callbackRR20d | -   [                             |
| iag_matrix_callbackRKNSt13unorder | cudaq::sample_result::deserialize |
| ed_mapINSt6stringENSt6stringEEE), |     (C++                          |
|     [\[2\]](                      |     functio                       |
| api/languages/cpp_api.html#_CPPv4 | n)](api/languages/cpp_api.html#_C |
| N5cudaq14matrix_handler6defineENS | PPv4N5cudaq13sample_result11deser |
| t6stringENSt6vectorINSt7int64_tEE | ializeERNSt6vectorINSt6size_tEEE) |
| ERR15matrix_callbackRRNSt13unorde | -   [cudaq::sample_result::dump   |
| red_mapINSt6stringENSt6stringEEE) |     (C++                          |
| -                                 |     function)](api/languag        |
|   [cudaq::matrix_handler::degrees | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     (C++                          | ample_result4dumpERNSt7ostreamE), |
|     function)](ap                 |     [\[1\]                        |
| i/languages/cpp_api.html#_CPPv4NK | ](api/languages/cpp_api.html#_CPP |
| 5cudaq14matrix_handler7degreesEv) | v4NK5cudaq13sample_result4dumpEv) |
| -                                 | -   [cudaq::sample_result::end    |
|  [cudaq::matrix_handler::displace |     (C++                          |
|     (C++                          |     function                      |
|     function)](api/language       | )](api/languages/cpp_api.html#_CP |
| s/cpp_api.html#_CPPv4N5cudaq14mat | Pv4N5cudaq13sample_result3endEv), |
| rix_handler8displaceENSt6size_tE) |     [\[1\                         |
| -   [cudaq::matrix                | ]](api/languages/cpp_api.html#_CP |
| _handler::get_expected_dimensions | Pv4NK5cudaq13sample_result3endEv) |
|     (C++                          | -   [                             |
|                                   | cudaq::sample_result::expectation |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     f                             |
| ndler23get_expected_dimensionsEv) | unction)](api/languages/cpp_api.h |
| -   [cudaq::matrix_ha             | tml#_CPPv4NK5cudaq13sample_result |
| ndler::get_parameter_descriptions | 11expectationEKNSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|                                   | q::sample_result::get_annotations |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4NK5cudaq14matrix_handl |     function)](api/langua         |
| er26get_parameter_descriptionsEv) | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| -   [c                            | sample_result15get_annotationsEv) |
| udaq::matrix_handler::instantiate | -   [c                            |
|     (C++                          | udaq::sample_result::get_marginal |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/languages/cpp_ |
| 5cudaq14matrix_handler11instantia | api.html#_CPPv4NK5cudaq13sample_r |
| teENSt6stringERKNSt6vectorINSt6si | esult12get_marginalERKNSt6vectorI |
| ze_tEEERK20commutation_behavior), | NSt6size_tEEEKNSt11string_viewE), |
|     [\[1\]](                      |     [\[1\]](api/languages/cpp_    |
| api/languages/cpp_api.html#_CPPv4 | api.html#_CPPv4NK5cudaq13sample_r |
| N5cudaq14matrix_handler11instanti | esult12get_marginalERRKNSt6vector |
| ateENSt6stringERRNSt6vectorINSt6s | INSt6size_tEEEKNSt11string_viewE) |
| ize_tEEERK20commutation_behavior) | -   [cuda                         |
| -   [cuda                         | q::sample_result::get_total_shots |
| q::matrix_handler::matrix_handler |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/languag        | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| es/cpp_api.html#_CPPv4I0_NSt11ena | sample_result15get_total_shotsEv) |
| ble_if_tINSt12is_base_of_vI16oper | -   [cuda                         |
| ator_handler1TEEbEEEN5cudaq14matr | q::sample_result::has_even_parity |
| ix_handler14matrix_handlerERK1T), |     (C++                          |
|     [\[1\]](ap                    |     fun                           |
| i/languages/cpp_api.html#_CPPv4I0 | ction)](api/languages/cpp_api.htm |
| _NSt11enable_if_tINSt12is_base_of | l#_CPPv4N5cudaq13sample_result15h |
| _vI16operator_handler1TEEbEEEN5cu | as_even_parityENSt11string_viewE) |
| daq14matrix_handler14matrix_handl | -   [cuda                         |
| erERK1TRK20commutation_behavior), | q::sample_result::has_expectation |
|     [\[2\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4N5cudaq14matrix_hand |     funct                         |
| ler14matrix_handlerENSt6size_tE), | ion)](api/languages/cpp_api.html# |
|     [\[3\]](api/                  | _CPPv4NK5cudaq13sample_result15ha |
| languages/cpp_api.html#_CPPv4N5cu | s_expectationEKNSt11string_viewE) |
| daq14matrix_handler14matrix_handl | -   [cu                           |
| erENSt6stringERKNSt6vectorINSt6si | daq::sample_result::most_probable |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[4\]](api/                  |     fun                           |
| languages/cpp_api.html#_CPPv4N5cu | ction)](api/languages/cpp_api.htm |
| daq14matrix_handler14matrix_handl | l#_CPPv4NK5cudaq13sample_result13 |
| erENSt6stringERRNSt6vectorINSt6si | most_probableEKNSt11string_viewE) |
| ze_tEEERK20commutation_behavior), | -                                 |
|     [\                            | [cudaq::sample_result::operator+= |
| [5\]](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq14matrix_handler14ma |     function)](api/langua         |
| trix_handlerERK14matrix_handler), | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     [                             | ample_resultpLERK13sample_result) |
| \[6\]](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq14matrix_handler14m |  [cudaq::sample_result::operator= |
| atrix_handlerERR14matrix_handler) |     (C++                          |
| -                                 |     function)](api/langua         |
|  [cudaq::matrix_handler::momentum | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     (C++                          | ample_resultaSERR13sample_result) |
|     function)](api/language       | -                                 |
| s/cpp_api.html#_CPPv4N5cudaq14mat | [cudaq::sample_result::operator== |
| rix_handler8momentumENSt6size_tE) |     (C++                          |
| -                                 |     function)](api/languag        |
|    [cudaq::matrix_handler::number | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     (C++                          | ample_resulteqERK13sample_result) |
|     function)](api/langua         | -   [                             |
| ges/cpp_api.html#_CPPv4N5cudaq14m | cudaq::sample_result::probability |
| atrix_handler6numberENSt6size_tE) |     (C++                          |
| -                                 |     function)](api/lan            |
| [cudaq::matrix_handler::operator= | guages/cpp_api.html#_CPPv4NK5cuda |
|     (C++                          | q13sample_result11probabilityENSt |
|     fun                           | 11string_viewEKNSt11string_viewE) |
| ction)](api/languages/cpp_api.htm | -   [cud                          |
| l#_CPPv4I0_NSt11enable_if_tIXaant | aq::sample_result::register_names |
| NSt7is_sameI1T14matrix_handlerE5v |     (C++                          |
| alueENSt12is_base_of_vI16operator |     function)](api/langu          |
| _handler1TEEEbEEEN5cudaq14matrix_ | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| handleraSER14matrix_handlerRK1T), | 3sample_result14register_namesEv) |
|     [\[1\]](api/languages         | -                                 |
| /cpp_api.html#_CPPv4N5cudaq14matr |    [cudaq::sample_result::reorder |
| ix_handleraSERK14matrix_handler), |     (C++                          |
|     [\[2\]](api/language          |     function)](api/langua         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ges/cpp_api.html#_CPPv4N5cudaq13s |
| rix_handleraSERR14matrix_handler) | ample_result7reorderERKNSt6vector |
| -   [                             | INSt6size_tEEEKNSt11string_viewE) |
| cudaq::matrix_handler::operator== | -   [cu                           |
|     (C++                          | daq::sample_result::sample_result |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4NK5cudaq14mat |     function)](api/               |
| rix_handlereqERK14matrix_handler) | languages/cpp_api.html#_CPPv4N5cu |
| -                                 | daq13sample_result13sample_result |
|    [cudaq::matrix_handler::parity | E16CountsDictionary10cudaq_json), |
|     (C++                          |     [                             |
|     function)](api/langua         | \[1\]](api/languages/cpp_api.html |
| ges/cpp_api.html#_CPPv4N5cudaq14m | #_CPPv4N5cudaq13sample_result13sa |
| atrix_handler6parityENSt6size_tE) | mple_resultERK15ExecutionResult), |
| -                                 |     [\[2\]](api/la                |
|  [cudaq::matrix_handler::position | nguages/cpp_api.html#_CPPv4N5cuda |
|     (C++                          | q13sample_result13sample_resultER |
|     function)](api/language       | KNSt6vectorI15ExecutionResultEE), |
| s/cpp_api.html#_CPPv4N5cudaq14mat |                                   |
| rix_handler8positionENSt6size_tE) |  [\[3\]](api/languages/cpp_api.ht |
| -   [cudaq::                      | ml#_CPPv4N5cudaq13sample_result13 |
| matrix_handler::remove_definition | sample_resultERR13sample_result), |
|     (C++                          |     [                             |
|     fu                            | \[4\]](api/languages/cpp_api.html |
| nction)](api/languages/cpp_api.ht | #_CPPv4N5cudaq13sample_result13sa |
| ml#_CPPv4N5cudaq14matrix_handler1 | mple_resultERR15ExecutionResult), |
| 7remove_definitionERKNSt6stringE) |     [\[5\]](api/lan               |
| -                                 | guages/cpp_api.html#_CPPv4N5cudaq |
|   [cudaq::matrix_handler::squeeze | 13sample_result13sample_resultEdR |
|     (C++                          | KNSt6vectorI15ExecutionResultEE), |
|     function)](api/languag        |     [\[6\]](api/lan               |
| es/cpp_api.html#_CPPv4N5cudaq14ma | guages/cpp_api.html#_CPPv4N5cudaq |
| trix_handler7squeezeENSt6size_tE) | 13sample_result13sample_resultEv) |
| -   [cudaq::m                     | -                                 |
| atrix_handler::to_diagonal_matrix |  [cudaq::sample_result::serialize |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](api                |
| uages/cpp_api.html#_CPPv4NK5cudaq | /languages/cpp_api.html#_CPPv4NK5 |
| 14matrix_handler18to_diagonal_mat | cudaq13sample_result9serializeEv) |
| rixERNSt13unordered_mapINSt6size_ | -   [cudaq::sample_result::size   |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq13sampl |
| [cudaq::matrix_handler::to_matrix | e_result4sizeEKNSt11string_viewE) |
|     (C++                          | -   [cudaq::sample_result::to_map |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/cpp  |
| v4NK5cudaq14matrix_handler9to_mat | _api.html#_CPPv4NK5cudaq13sample_ |
| rixERNSt13unordered_mapINSt6size_ | result6to_mapEKNSt11string_viewE) |
| tENSt7int64_tEEERKNSt13unordered_ | -   [cuda                         |
| mapINSt6stringENSt7complexIdEEEE) | q::sample_result::\~sample_result |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::to_string |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/               | _CPPv4N5cudaq13sample_resultD0Ev) |
| languages/cpp_api.html#_CPPv4NK5c | -   [cudaq::scalar_callback (C++  |
| udaq14matrix_handler9to_stringEb) |     c                             |
| -                                 | lass)](api/languages/cpp_api.html |
| [cudaq::matrix_handler::unique_id | #_CPPv4N5cudaq15scalar_callbackE) |
|     (C++                          | -   [c                            |
|     function)](api/               | udaq::scalar_callback::operator() |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14matrix_handler9unique_idEv) |     function)](api/language       |
| -   [cudaq:                       | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| :matrix_handler::\~matrix_handler | alar_callbackclERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|     functi                        | -   [                             |
| on)](api/languages/cpp_api.html#_ | cudaq::scalar_callback::operator= |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     (C++                          |
| -   [cudaq::matrix_op (C++        |     function)](api/languages/c    |
|     type)](api/languages/cpp_a    | pp_api.html#_CPPv4N5cudaq15scalar |
| pi.html#_CPPv4N5cudaq9matrix_opE) | _callbackaSERK15scalar_callback), |
| -   [cudaq::matrix_op_term (C++   |     [\[1\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4N5cudaq15scala |
|  type)](api/languages/cpp_api.htm | r_callbackaSERR15scalar_callback) |
| l#_CPPv4N5cudaq14matrix_op_termE) | -   [cudaq:                       |
| -                                 | :scalar_callback::scalar_callback |
|    [cudaq::mdiag_operator_handler |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     class)](                      | es/cpp_api.html#_CPPv4I0_NSt11ena |
| api/languages/cpp_api.html#_CPPv4 | ble_if_tINSt16is_invocable_r_vINS |
| N5cudaq22mdiag_operator_handlerE) | t7complexIdEE8CallableRKNSt13unor |
| -   [cudaq::measure_handle (C++   | dered_mapINSt6stringENSt7complexI |
|                                   | dEEEEEEbEEEN5cudaq15scalar_callba |
| class)](api/languages/cpp_api.htm | ck15scalar_callbackERR8Callable), |
| l#_CPPv4N5cudaq14measure_handleE) |     [\[1\                         |
| -   [cudaq::measure_result (C++   | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_callback15scal |
|  type)](api/languages/cpp_api.htm | ar_callbackERK15scalar_callback), |
| l#_CPPv4N5cudaq14measure_resultE) |     [\[2                          |
| -   [cudaq::mpi (C++              | \]](api/languages/cpp_api.html#_C |
|     type)](api/languages          | PPv4N5cudaq15scalar_callback15sca |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | lar_callbackERR15scalar_callback) |
| -   [cudaq::mpi::all_gather (C++  | -   [cudaq::scalar_operator (C++  |
|     fu                            |     c                             |
| nction)](api/languages/cpp_api.ht | lass)](api/languages/cpp_api.html |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | #_CPPv4N5cudaq15scalar_operatorE) |
| RNSt6vectorIdEERKNSt6vectorIdEE), | -                                 |
|                                   | [cudaq::scalar_operator::evaluate |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi10all_gather |                                   |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |    function)](api/languages/cpp_a |
| -   [cudaq::mpi::all_reduce (C++  | pi.html#_CPPv4NK5cudaq15scalar_op |
|                                   | erator8evaluateERKNSt13unordered_ |
|  function)](api/languages/cpp_api | mapINSt6stringENSt7complexIdEEEE) |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | -   [cudaq::scalar_ope            |
| reduceE1TRK1TRK14BinaryFunction), | rator::get_parameter_descriptions |
|     [\[1\]](api/langu             |     (C++                          |
| ages/cpp_api.html#_CPPv4I00EN5cud |     f                             |
| aq3mpi10all_reduceE1TRK1TRK4Func) | unction)](api/languages/cpp_api.h |
| -   [cudaq::mpi::broadcast (C++   | tml#_CPPv4NK5cudaq15scalar_operat |
|     function)](api/               | or26get_parameter_descriptionsEv) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cu                           |
| daq3mpi9broadcastERNSt6stringEi), | daq::scalar_operator::is_constant |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/lang           |
| q3mpi9broadcastERNSt6vectorIdEEi) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cudaq::mpi::finalize (C++    | 15scalar_operator11is_constantEv) |
|     f                             | -   [c                            |
| unction)](api/languages/cpp_api.h | udaq::scalar_operator::operator\* |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     (C++                          |
| -   [cudaq::mpi::initialize (C++  |     function                      |
|     function                      | )](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4N5cudaq15scalar_operatormlENSt |
| Pv4N5cudaq3mpi10initializeEiPPc), | 7complexIdEERK15scalar_operator), |
|     [                             |     [\[1\                         |
| \[1\]](api/languages/cpp_api.html | ]](api/languages/cpp_api.html#_CP |
| #_CPPv4N5cudaq3mpi10initializeEv) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::mpi::is_initialized   | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     function                      | p_api.html#_CPPv4N5cudaq15scalar_ |
| )](api/languages/cpp_api.html#_CP | operatormlEdRK15scalar_operator), |
| Pv4N5cudaq3mpi14is_initializedEv) |     [\[3\]](api/languages/cp      |
| -   [cudaq::mpi::num_ranks (C++   | p_api.html#_CPPv4N5cudaq15scalar_ |
|     fu                            | operatormlEdRR15scalar_operator), |
| nction)](api/languages/cpp_api.ht |     [\[4\]](api/languages         |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -   [cudaq::mpi::rank (C++        | alar_operatormlENSt7complexIdEE), |
|                                   |     [\[5\]](api/languages/cpp     |
|    function)](api/languages/cpp_a | _api.html#_CPPv4NKR5cudaq15scalar |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | _operatormlERK15scalar_operator), |
| -   [cudaq::noise_model (C++      |     [\[6\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|    class)](api/languages/cpp_api. | 4NKR5cudaq15scalar_operatormlEd), |
| html#_CPPv4N5cudaq11noise_modelE) |     [\[7\]](api/language          |
| -   [cudaq::n                     | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| oise_model::add_all_qubit_channel | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[8\]](api/languages/cp      |
|     function)](api                | p_api.html#_CPPv4NO5cudaq15scalar |
| /languages/cpp_api.html#_CPPv4IDp | _operatormlERK15scalar_operator), |
| EN5cudaq11noise_model21add_all_qu |     [\[9\                         |
| bit_channelEvRK13kraus_channeli), | ]](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/langua            | Pv4NO5cudaq15scalar_operatormlEd) |
| ges/cpp_api.html#_CPPv4N5cudaq11n | -   [cu                           |
| oise_model21add_all_qubit_channel | daq::scalar_operator::operator\*= |
| ERKNSt6stringERK13kraus_channeli) |     (C++                          |
| -                                 |     function)](api/languag        |
|  [cudaq::noise_model::add_channel | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormLENSt7complexIdEE), |
|     funct                         |     [\[1\]](api/languages/c       |
| ion)](api/languages/cpp_api.html# | pp_api.html#_CPPv4N5cudaq15scalar |
| _CPPv4IDpEN5cudaq11noise_model11a | _operatormLERK15scalar_operator), |
| dd_channelEvRK15PredicateFuncTy), |     [\[2                          |
|     [\[1\]](api/languages/cpp_    | \]](api/languages/cpp_api.html#_C |
| api.html#_CPPv4IDpEN5cudaq11noise | PPv4N5cudaq15scalar_operatormLEd) |
| _model11add_channelEvRKNSt6vector | -   [                             |
| INSt6size_tEEERK13kraus_channel), | cudaq::scalar_operator::operator+ |
|     [\[2\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function                      |
| cudaq11noise_model11add_channelER | )](api/languages/cpp_api.html#_CP |
| KNSt6stringERK15PredicateFuncTy), | Pv4N5cudaq15scalar_operatorplENSt |
|                                   | 7complexIdEERK15scalar_operator), |
| [\[3\]](api/languages/cpp_api.htm |     [\[1\                         |
| l#_CPPv4N5cudaq11noise_model11add | ]](api/languages/cpp_api.html#_CP |
| _channelERKNSt6stringERKNSt6vecto | Pv4N5cudaq15scalar_operatorplENSt |
| rINSt6size_tEEERK13kraus_channel) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::noise_model::empty    |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     function                      | operatorplEdRK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[3\]](api/languages/cp      |
| Pv4NK5cudaq11noise_model5emptyEv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatorplEdRR15scalar_operator), |
| [cudaq::noise_model::get_channels |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     function)](api/l              | alar_operatorplENSt7complexIdEE), |
| anguages/cpp_api.html#_CPPv4I0ENK |     [\[5\]](api/languages/cpp     |
| 5cudaq11noise_model12get_channels | _api.html#_CPPv4NKR5cudaq15scalar |
| ENSt6vectorI13kraus_channelEERKNS | _operatorplERK15scalar_operator), |
| t6vectorINSt6size_tEEERKNSt6vecto |     [\[6\]]                       |
| rINSt6size_tEEERKNSt6vectorIdEE), | (api/languages/cpp_api.html#_CPPv |
|     [\[1\]](api/languages/cpp_a   | 4NKR5cudaq15scalar_operatorplEd), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[7\]]                       |
| el12get_channelsERKNSt6stringERKN | (api/languages/cpp_api.html#_CPPv |
| St6vectorINSt6size_tEEERKNSt6vect | 4NKR5cudaq15scalar_operatorplEv), |
| orINSt6size_tEEERKNSt6vectorIdEE) |     [\[8\]](api/language          |
| -                                 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|  [cudaq::noise_model::noise_model | alar_operatorplENSt7complexIdEE), |
|     (C++                          |     [\[9\]](api/languages/cp      |
|     function)](api                | p_api.html#_CPPv4NO5cudaq15scalar |
| /languages/cpp_api.html#_CPPv4N5c | _operatorplERK15scalar_operator), |
| udaq11noise_model11noise_modelEv) |     [\[10\]                       |
| -   [cu                           | ](api/languages/cpp_api.html#_CPP |
| daq::noise_model::PredicateFuncTy | v4NO5cudaq15scalar_operatorplEd), |
|     (C++                          |     [\[11\                        |
|     type)](api/la                 | ]](api/languages/cpp_api.html#_CP |
| nguages/cpp_api.html#_CPPv4N5cuda | Pv4NO5cudaq15scalar_operatorplEv) |
| q11noise_model15PredicateFuncTyE) | -   [c                            |
| -   [cud                          | udaq::scalar_operator::operator+= |
| aq::noise_model::register_channel |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/languages      | es/cpp_api.html#_CPPv4N5cudaq15sc |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | alar_operatorpLENSt7complexIdEE), |
| noise_model16register_channelEvv) |     [\[1\]](api/languages/c       |
| -   [cudaq::                      | pp_api.html#_CPPv4N5cudaq15scalar |
| noise_model::requires_constructor | _operatorpLERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     type)](api/languages/cp       | \]](api/languages/cpp_api.html#_C |
| p_api.html#_CPPv4I0DpEN5cudaq11no | PPv4N5cudaq15scalar_operatorpLEd) |
| ise_model20requires_constructorE) | -   [                             |
| -   [cudaq::noise_model_type (C++ | cudaq::scalar_operator::operator- |
|     e                             |     (C++                          |
| num)](api/languages/cpp_api.html# |     function                      |
| _CPPv4N5cudaq16noise_model_typeE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::no                    | Pv4N5cudaq15scalar_operatormiENSt |
| ise_model_type::amplitude_damping | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](api/languages    | ]](api/languages/cpp_api.html#_CP |
| /cpp_api.html#_CPPv4N5cudaq16nois | Pv4N5cudaq15scalar_operatormiENSt |
| e_model_type17amplitude_dampingE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::noise_mode            |     [\[2\]](api/languages/cp      |
| l_type::amplitude_damping_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRK15scalar_operator), |
|     e                             |     [\[3\]](api/languages/cp      |
| numerator)](api/languages/cpp_api | p_api.html#_CPPv4N5cudaq15scalar_ |
| .html#_CPPv4N5cudaq16noise_model_ | operatormiEdRR15scalar_operator), |
| type25amplitude_damping_channelE) |     [\[4\]](api/languages         |
| -   [cudaq::n                     | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| oise_model_type::bit_flip_channel | alar_operatormiENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     enumerator)](api/language     | _api.html#_CPPv4NKR5cudaq15scalar |
| s/cpp_api.html#_CPPv4N5cudaq16noi | _operatormiERK15scalar_operator), |
| se_model_type16bit_flip_channelE) |     [\[6\]]                       |
| -   [cudaq::                      | (api/languages/cpp_api.html#_CPPv |
| noise_model_type::depolarization1 | 4NKR5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[7\]]                       |
|     enumerator)](api/languag      | (api/languages/cpp_api.html#_CPPv |
| es/cpp_api.html#_CPPv4N5cudaq16no | 4NKR5cudaq15scalar_operatormiEv), |
| ise_model_type15depolarization1E) |     [\[8\]](api/language          |
| -   [cudaq::                      | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| noise_model_type::depolarization2 | alar_operatormiENSt7complexIdEE), |
|     (C++                          |     [\[9\]](api/languages/cp      |
|     enumerator)](api/languag      | p_api.html#_CPPv4NO5cudaq15scalar |
| es/cpp_api.html#_CPPv4N5cudaq16no | _operatormiERK15scalar_operator), |
| ise_model_type15depolarization2E) |     [\[10\]                       |
| -   [cudaq::noise_m               | ](api/languages/cpp_api.html#_CPP |
| odel_type::depolarization_channel | v4NO5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[11\                        |
|                                   | ]](api/languages/cpp_api.html#_CP |
|   enumerator)](api/languages/cpp_ | Pv4NO5cudaq15scalar_operatormiEv) |
| api.html#_CPPv4N5cudaq16noise_mod | -   [c                            |
| el_type22depolarization_channelE) | udaq::scalar_operator::operator-= |
| -                                 |     (C++                          |
|  [cudaq::noise_model_type::pauli1 |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     enumerator)](a                | alar_operatormIENSt7complexIdEE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[1\]](api/languages/c       |
| 5cudaq16noise_model_type6pauli1E) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _operatormIERK15scalar_operator), |
|  [cudaq::noise_model_type::pauli2 |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     enumerator)](a                | PPv4N5cudaq15scalar_operatormIEd) |
| pi/languages/cpp_api.html#_CPPv4N | -   [                             |
| 5cudaq16noise_model_type6pauli2E) | cudaq::scalar_operator::operator/ |
| -   [cudaq                        |     (C++                          |
| ::noise_model_type::phase_damping |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     enumerator)](api/langu        | Pv4N5cudaq15scalar_operatordvENSt |
| ages/cpp_api.html#_CPPv4N5cudaq16 | 7complexIdEERK15scalar_operator), |
| noise_model_type13phase_dampingE) |     [\[1\                         |
| -   [cudaq::noi                   | ]](api/languages/cpp_api.html#_CP |
| se_model_type::phase_flip_channel | Pv4N5cudaq15scalar_operatordvENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     enumerator)](api/languages/   |     [\[2\]](api/languages/cp      |
| cpp_api.html#_CPPv4N5cudaq16noise | p_api.html#_CPPv4N5cudaq15scalar_ |
| _model_type18phase_flip_channelE) | operatordvEdRK15scalar_operator), |
| -                                 |     [\[3\]](api/languages/cp      |
| [cudaq::noise_model_type::unknown | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRR15scalar_operator), |
|     enumerator)](ap               |     [\[4\]](api/languages         |
| i/languages/cpp_api.html#_CPPv4N5 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| cudaq16noise_model_type7unknownE) | alar_operatordvENSt7complexIdEE), |
| -                                 |     [\[5\]](api/languages/cpp     |
| [cudaq::noise_model_type::x_error | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatordvERK15scalar_operator), |
|     enumerator)](ap               |     [\[6\]]                       |
| i/languages/cpp_api.html#_CPPv4N5 | (api/languages/cpp_api.html#_CPPv |
| cudaq16noise_model_type7x_errorE) | 4NKR5cudaq15scalar_operatordvEd), |
| -                                 |     [\[7\]](api/language          |
| [cudaq::noise_model_type::y_error | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatordvENSt7complexIdEE), |
|     enumerator)](ap               |     [\[8\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4NO5cudaq15scalar |
| cudaq16noise_model_type7y_errorE) | _operatordvERK15scalar_operator), |
| -                                 |     [\[9\                         |
| [cudaq::noise_model_type::z_error | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatordvEd) |
|     enumerator)](ap               | -   [c                            |
| i/languages/cpp_api.html#_CPPv4N5 | udaq::scalar_operator::operator/= |
| cudaq16noise_model_type7z_errorE) |     (C++                          |
| -   [cudaq::num_available_gpus    |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     function                      | alar_operatordVENSt7complexIdEE), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/languages/c       |
| Pv4N5cudaq18num_available_gpusEv) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::observe (C++          | _operatordVERK15scalar_operator), |
|     function)]                    |     [\[2                          |
| (api/languages/cpp_api.html#_CPPv | \]](api/languages/cpp_api.html#_C |
| 4I00DpEN5cudaq7observeENSt6vector | PPv4N5cudaq15scalar_operatordVEd) |
| I14observe_resultEERR13QuantumKer | -   [                             |
| nelRK15SpinOpContainerDpRR4Args), | cudaq::scalar_operator::operator= |
|     [\[1\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4I0DpEN5cudaq7observe |     function)](api/languages/c    |
| E14observe_resultNSt6size_tERR13Q | pp_api.html#_CPPv4N5cudaq15scalar |
| uantumKernelRK7spin_opDpRR4Args), | _operatoraSERK15scalar_operator), |
|     [\[                           |     [\[1\]](api/languages/        |
| 2\]](api/languages/cpp_api.html#_ | cpp_api.html#_CPPv4N5cudaq15scala |
| CPPv4I0DpEN5cudaq7observeE14obser | r_operatoraSERR15scalar_operator) |
| ve_resultRK15observe_optionsRR13Q | -   [c                            |
| uantumKernelRK7spin_opDpRR4Args), | udaq::scalar_operator::operator== |
|     [\[3\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     function)](api/languages/c    |
| udaq7observeE14observe_resultRR13 | pp_api.html#_CPPv4NK5cudaq15scala |
| QuantumKernelRK7spin_opDpRR4Args) | r_operatoreqERK15scalar_operator) |
| -   [cudaq::observe_options (C++  | -   [cudaq:                       |
|     st                            | :scalar_operator::scalar_operator |
| ruct)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15observe_optionsE) |     func                          |
| -   [cudaq::observe_result (C++   | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq15scalar_operator15 |
| class)](api/languages/cpp_api.htm | scalar_operatorENSt7complexIdEE), |
| l#_CPPv4N5cudaq14observe_resultE) |     [\[1\]](api/langu             |
| -                                 | ages/cpp_api.html#_CPPv4N5cudaq15 |
|    [cudaq::observe_result::counts | scalar_operator15scalar_operatorE |
|     (C++                          | RK15scalar_callbackRRNSt13unorder |
|     function)](api/languages/c    | ed_mapINSt6stringENSt6stringEEE), |
| pp_api.html#_CPPv4N5cudaq14observ |     [\[2\                         |
| e_result6countsERK12spin_op_term) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::observe_result::dump  | Pv4N5cudaq15scalar_operator15scal |
|     (C++                          | ar_operatorERK15scalar_operator), |
|     function)                     |     [\[3\]](api/langu             |
| ](api/languages/cpp_api.html#_CPP | ages/cpp_api.html#_CPPv4N5cudaq15 |
| v4N5cudaq14observe_result4dumpEv) | scalar_operator15scalar_operatorE |
| -   [c                            | RR15scalar_callbackRRNSt13unorder |
| udaq::observe_result::expectation | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[4\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
| function)](api/languages/cpp_api. | Pv4N5cudaq15scalar_operator15scal |
| html#_CPPv4N5cudaq14observe_resul | ar_operatorERR15scalar_operator), |
| t11expectationERK12spin_op_term), |     [\[5\]](api/language          |
|     [\[1\]](api/la                | s/cpp_api.html#_CPPv4N5cudaq15sca |
| nguages/cpp_api.html#_CPPv4N5cuda | lar_operator15scalar_operatorEd), |
| q14observe_result11expectationEv) |     [\[6\]](api/languag           |
| -   [cuda                         | es/cpp_api.html#_CPPv4N5cudaq15sc |
| q::observe_result::id_coefficient | alar_operator15scalar_operatorEv) |
|     (C++                          | -   [                             |
|     function)](api/langu          | cudaq::scalar_operator::to_matrix |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     (C++                          |
| observe_result14id_coefficientEv) |                                   |
| -   [cuda                         |   function)](api/languages/cpp_ap |
| q::observe_result::observe_result | i.html#_CPPv4NK5cudaq15scalar_ope |
|     (C++                          | rator9to_matrixERKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|   function)](api/languages/cpp_ap | -   [                             |
| i.html#_CPPv4N5cudaq14observe_res | cudaq::scalar_operator::to_string |
| ult14observe_resultEdRK7spin_op), |     (C++                          |
|     [\[1\]](a                     |     function)](api/l              |
| pi/languages/cpp_api.html#_CPPv4N | anguages/cpp_api.html#_CPPv4NK5cu |
| 5cudaq14observe_result14observe_r | daq15scalar_operator9to_stringEv) |
| esultEdRK7spin_op13sample_result) | -   [cudaq::s                     |
| -                                 | calar_operator::\~scalar_operator |
|  [cudaq::observe_result::operator |     (C++                          |
|     double (C++                   |     functio                       |
|     functio                       | n)](api/languages/cpp_api.html#_C |
| n)](api/languages/cpp_api.html#_C | PPv4N5cudaq15scalar_operatorD0Ev) |
| PPv4N5cudaq14observe_resultcvdEv) | -   [cudaq::set_noise (C++        |
| -                                 |     function)](api/langu          |
|  [cudaq::observe_result::raw_data | ages/cpp_api.html#_CPPv4N5cudaq9s |
|     (C++                          | et_noiseERKN5cudaq11noise_modelE) |
|     function)](ap                 | -   [cudaq::set_random_seed (C++  |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/               |
| cudaq14observe_result8raw_dataEv) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::operator_handler (C++ | daq15set_random_seedENSt6size_tE) |
|     cl                            | -   [cudaq::simulation_precision  |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16operator_handlerE) |     enum)                         |
| -   [cudaq::optimizable_function  | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq20simulation_precisionE) |
|     class)                        | -   [                             |
| ](api/languages/cpp_api.html#_CPP | cudaq::simulation_precision::fp32 |
| v4N5cudaq20optimizable_functionE) |     (C++                          |
| -   [cudaq::optimization_result   |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     type                          | udaq20simulation_precision4fp32E) |
| )](api/languages/cpp_api.html#_CP | -   [                             |
| Pv4N5cudaq19optimization_resultE) | cudaq::simulation_precision::fp64 |
| -   [cudaq::optimizer (C++        |     (C++                          |
|     class)](api/languages/cpp_a   |     enumerator)](api              |
| pi.html#_CPPv4N5cudaq9optimizerE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::optimizer::optimize   | udaq20simulation_precision4fp64E) |
|     (C++                          | -   [cudaq::SimulationState (C++  |
|                                   |     c                             |
|  function)](api/languages/cpp_api | lass)](api/languages/cpp_api.html |
| .html#_CPPv4N5cudaq9optimizer8opt | #_CPPv4N5cudaq15SimulationStateE) |
| imizeEKiRR20optimizable_function) | -   [                             |
| -   [cu                           | cudaq::SimulationState::precision |
| daq::optimizer::requiresGradients |     (C++                          |
|     (C++                          |     enum)](api                    |
|     function)](api/la             | /languages/cpp_api.html#_CPPv4N5c |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq15SimulationState9precisionE) |
| q9optimizer17requiresGradientsEv) | -   [cudaq:                       |
| -   [cudaq::orca (C++             | :SimulationState::precision::fp32 |
|     type)](api/languages/         |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     enumerator)](api/lang         |
| -   [cudaq::orca::sample (C++     | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     function)](api/languages/c    | 5SimulationState9precision4fp32E) |
| pp_api.html#_CPPv4N5cudaq4orca6sa | -   [cudaq:                       |
| mpleERNSt6vectorINSt6size_tEEERNS | :SimulationState::precision::fp64 |
| t6vectorINSt6size_tEEERNSt6vector |     (C++                          |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     enumerator)](api/lang         |
|     [\[1\]]                       | uages/cpp_api.html#_CPPv4N5cudaq1 |
| (api/languages/cpp_api.html#_CPPv | 5SimulationState9precision4fp64E) |
| 4N5cudaq4orca6sampleERNSt6vectorI | -                                 |
| NSt6size_tEEERNSt6vectorINSt6size |   [cudaq::SimulationState::Tensor |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     (C++                          |
| -   [cudaq::orca::sample_async    |     struct)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq15SimulationState6TensorE) |
| function)](api/languages/cpp_api. | -   [cudaq::spin_handler (C++     |
| html#_CPPv4N5cudaq4orca12sample_a |                                   |
| syncERNSt6vectorINSt6size_tEEERNS |   class)](api/languages/cpp_api.h |
| t6vectorINSt6size_tEEERNSt6vector | tml#_CPPv4N5cudaq12spin_handlerE) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [cudaq:                       |
|     [\[1\]](api/la                | :spin_handler::to_diagonal_matrix |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q4orca12sample_asyncERNSt6vectorI |     function)](api/la             |
| NSt6size_tEEERNSt6vectorINSt6size | nguages/cpp_api.html#_CPPv4NK5cud |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | aq12spin_handler18to_diagonal_mat |
| -   [cudaq::OrcaRemoteRESTQPU     | rixERNSt13unordered_mapINSt6size_ |
|     (C++                          | tENSt7int64_tEEERKNSt13unordered_ |
|     cla                           | mapINSt6stringENSt7complexIdEEEE) |
| ss)](api/languages/cpp_api.html#_ | -                                 |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |   [cudaq::spin_handler::to_matrix |
| -   [cudaq::other_policies (C++   |     (C++                          |
|     s                             |     function                      |
| truct)](api/languages/cpp_api.htm | )](api/languages/cpp_api.html#_CP |
| l#_CPPv4N5cudaq14other_policiesE) | Pv4N5cudaq12spin_handler9to_matri |
| -   [cudaq::PasqalRemoteRESTQPU   | xERKNSt6stringENSt7complexIdEEb), |
|     (C++                          |     [\[1                          |
|     class                         | \]](api/languages/cpp_api.html#_C |
| )](api/languages/cpp_api.html#_CP | PPv4NK5cudaq12spin_handler9to_mat |
| Pv4N5cudaq19PasqalRemoteRESTQPUE) | rixERNSt13unordered_mapINSt6size_ |
| -   [cudaq::pauli1 (C++           | tENSt7int64_tEEERKNSt13unordered_ |
|     class)](api/languages/cp      | mapINSt6stringENSt7complexIdEEEE) |
| p_api.html#_CPPv4N5cudaq6pauli1E) | -   [cuda                         |
| -                                 | q::spin_handler::to_sparse_matrix |
|    [cudaq::pauli1::num_parameters |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)]                      | languages/cpp_api.html#_CPPv4N5cu |
| (api/languages/cpp_api.html#_CPPv | daq12spin_handler16to_sparse_matr |
| 4N5cudaq6pauli114num_parametersE) | ixERKNSt6stringENSt7complexIdEEb) |
| -   [cudaq::pauli1::num_targets   | -                                 |
|     (C++                          |   [cudaq::spin_handler::to_string |
|     membe                         |     (C++                          |
| r)](api/languages/cpp_api.html#_C |     function)](ap                 |
| PPv4N5cudaq6pauli111num_targetsE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::pauli1::pauli1 (C++   | 5cudaq12spin_handler9to_stringEb) |
|     function)](api/languages/cpp_ | -                                 |
| api.html#_CPPv4N5cudaq6pauli16pau |   [cudaq::spin_handler::unique_id |
| li1ERKNSt6vectorIN5cudaq4realEEE) |     (C++                          |
| -   [cudaq::pauli2 (C++           |     function)](ap                 |
|     class)](api/languages/cp      | i/languages/cpp_api.html#_CPPv4NK |
| p_api.html#_CPPv4N5cudaq6pauli2E) | 5cudaq12spin_handler9unique_idEv) |
| -                                 | -   [cudaq::spin_op (C++          |
|    [cudaq::pauli2::num_parameters |     type)](api/languages/cpp      |
|     (C++                          | _api.html#_CPPv4N5cudaq7spin_opE) |
|     member)]                      | -   [cudaq::spin_op_term (C++     |
| (api/languages/cpp_api.html#_CPPv |                                   |
| 4N5cudaq6pauli214num_parametersE) |    type)](api/languages/cpp_api.h |
| -   [cudaq::pauli2::num_targets   | tml#_CPPv4N5cudaq12spin_op_termE) |
|     (C++                          | -   [cudaq::state (C++            |
|     membe                         |     class)](api/languages/c       |
| r)](api/languages/cpp_api.html#_C | pp_api.html#_CPPv4N5cudaq5stateE) |
| PPv4N5cudaq6pauli211num_targetsE) | -   [cudaq::state::amplitude (C++ |
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
