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
    -   [Defining targets and customizing
        compilation](using/basics/compile_kernel.html){.reference
        .internal}
        -   [Compiling for built-in
            targets](using/basics/compile_kernel.html#compiling-for-built-in-targets){.reference
            .internal}
        -   [Controlling the optimization level of the
            compiler](using/basics/compile_kernel.html#controlling-the-optimization-level-of-the-compiler){.reference
            .internal}
        -   [Using custom compile targets for finer-grained compilation
            control](using/basics/compile_kernel.html#using-custom-compile-targets-for-finer-grained-compilation-control){.reference
            .internal}
        -   [Customizing all the
            way](using/basics/compile_kernel.html#customizing-all-the-way){.reference
            .internal}
    -   [Troubleshooting](using/basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](using/basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
        -   [Python
            Stack-Traces](using/basics/troubleshooting.html#python-stack-traces){.reference
            .internal}
        -   [LLVM-based Python
            packages](using/basics/troubleshooting.html#llvm-based-python-packages){.reference
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
            -   [[`cuStabilizer`{.docutils .literal
                .notranslate}]{.pre}](using/backends/sims/noisy.html#custabilizer){.reference
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
        -   [Testing](using/realtime/cpu_transport.html#testing){.reference
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
-   [CUDA-Q QEC](using/cudaq-qec/cudaq-qec.html){.reference .internal}
-   [CUDA-Q
    Algorithms](using/cudaq-algorithms/cudaq-algorithms.html){.reference
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
        -   [Value-semantics IR change
            log](using/extending/compiler/ir_changelog.html){.reference
            .internal}
            -   [Producing this
                IR](using/extending/compiler/ir_changelog.html#producing-this-ir){.reference
                .internal}
            -   [Upcoming release ([`main`{.docutils .literal
                .notranslate}]{.pre}
                branch)](using/extending/compiler/ir_changelog.html#upcoming-release-main-branch){.reference
                .internal}
            -   [0.16.0](using/extending/compiler/ir_changelog.html#id5){.reference
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
        -   [Using compile targets to customize
            compilation](using/extending/compiler/compile_targets.html){.reference
            .internal}
            -   [Constructing single-stage compile
                targets](using/extending/compiler/compile_targets.html#constructing-single-stage-compile-targets){.reference
                .internal}
            -   [Setting compile targets with [`set_target`{.docutils
                .literal
                .notranslate}]{.pre}](using/extending/compiler/compile_targets.html#setting-compile-targets-with-set-target){.reference
                .internal}
            -   [Example](using/extending/compiler/compile_targets.html#example){.reference
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
            -   [[`compile()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.compile){.reference
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
            -   [Qiskit and OpenQASM
                Interoperability](api/languages/python_api.html#qiskit-and-openqasm-interoperability){.reference
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

-   [cudaq-logical](preview/logical/index.html){.reference .external}
-   [cudaq-edem](using/edem.html){.reference .internal}
    -   [Standalone build](using/edem.html#standalone-build){.reference
        .internal}
    -   [Python bindings and
        wheel](using/edem.html#python-bindings-and-wheel){.reference
        .internal}
    -   [CUDA-Q build](using/edem.html#cuda-q-build){.reference
        .internal}
    -   [Installed
        package](using/edem.html#installed-package){.reference
        .internal}
    -   [Packaging checks](using/edem.html#packaging-checks){.reference
        .internal}
    -   [Native interface and
        memory](using/edem.html#native-interface-and-memory){.reference
        .internal}
    -   [C++ ownership
        helper](using/edem.html#c-ownership-helper){.reference
        .internal}
    -   [Component
        boundary](using/edem.html#component-boundary){.reference
        .internal}
    -   [Fault
        distributions](using/edem.html#fault-distributions){.reference
        .internal}
    -   [Python
        construction](using/edem.html#python-construction){.reference
        .internal}
    -   [Packed format](using/edem.html#packed-format){.reference
        .internal}
    -   [Serial composition and DEM
        projection](using/edem.html#serial-composition-and-dem-projection){.reference
        .internal}
    -   [Exact noise conversion and column
        coalescing](using/edem.html#exact-noise-conversion-and-column-coalescing){.reference
        .internal}
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
| -   [canonicalize                 | -   [cud                          |
|     (cu                           | aq::phase_damping::num_parameters |
| daq.operators.boson.BosonOperator |     (C++                          |
|     attribute)](api/languages     |     member)](api/lan              |
| /python_api.html#cudaq.operators. | guages/cpp_api.html#_CPPv4N5cudaq |
| boson.BosonOperator.canonicalize) | 13phase_damping14num_parametersE) |
|     -   [(cudaq.                  | -   [                             |
| operators.boson.BosonOperatorTerm | cudaq::phase_damping::num_targets |
|                                   |     (C++                          |
|     attribute)](api/languages/pyt |     member)](api/                 |
| hon_api.html#cudaq.operators.boso | languages/cpp_api.html#_CPPv4N5cu |
| n.BosonOperatorTerm.canonicalize) | daq13phase_damping11num_targetsE) |
|     -   [(cudaq.                  | -   [cudaq::phase_flip_channel    |
| operators.fermion.FermionOperator |     (C++                          |
|                                   |     clas                          |
|     attribute)](api/languages/pyt | s)](api/languages/cpp_api.html#_C |
| hon_api.html#cudaq.operators.ferm | PPv4N5cudaq18phase_flip_channelE) |
| ion.FermionOperator.canonicalize) | -   [cudaq::p                     |
|     -   [(cudaq.oper              | hase_flip_channel::num_parameters |
| ators.fermion.FermionOperatorTerm |     (C++                          |
|                                   |     member)](api/language         |
| attribute)](api/languages/python_ | s/cpp_api.html#_CPPv4N5cudaq18pha |
| api.html#cudaq.operators.fermion. | se_flip_channel14num_parametersE) |
| FermionOperatorTerm.canonicalize) | -   [cudaq                        |
|     -                             | ::phase_flip_channel::num_targets |
|  [(cudaq.operators.MatrixOperator |     (C++                          |
|         attribute)](api/lang      |     member)](api/langu            |
| uages/python_api.html#cudaq.opera | ages/cpp_api.html#_CPPv4N5cudaq18 |
| tors.MatrixOperator.canonicalize) | phase_flip_channel11num_targetsE) |
|     -   [(c                       | -   [cudaq::product_op (C++       |
| udaq.operators.MatrixOperatorTerm |                                   |
|         attribute)](api/language  |  class)](api/languages/cpp_api.ht |
| s/python_api.html#cudaq.operators | ml#_CPPv4I0EN5cudaq10product_opE) |
| .MatrixOperatorTerm.canonicalize) | -   [cudaq::product_op::begin     |
|     -   [(                        |     (C++                          |
| cudaq.operators.spin.SpinOperator |     functio                       |
|         attribute)](api/languag   | n)](api/languages/cpp_api.html#_C |
| es/python_api.html#cudaq.operator | PPv4NK5cudaq10product_op5beginEv) |
| s.spin.SpinOperator.canonicalize) | -                                 |
|     -   [(cuda                    |  [cudaq::product_op::canonicalize |
| q.operators.spin.SpinOperatorTerm |     (C++                          |
|                                   |     func                          |
|       attribute)](api/languages/p | tion)](api/languages/cpp_api.html |
| ython_api.html#cudaq.operators.sp | #_CPPv4N5cudaq10product_op12canon |
| in.SpinOperatorTerm.canonicalize) | icalizeERKNSt3setINSt6size_tEEE), |
| -   [captured_variables()         |     [\[1\]](api                   |
|     (cudaq.PyKernelDecorator      | /languages/cpp_api.html#_CPPv4N5c |
|     method)](api/lan              | udaq10product_op12canonicalizeEv) |
| guages/python_api.html#cudaq.PyKe | -   [                             |
| rnelDecorator.captured_variables) | cudaq::product_op::const_iterator |
| -   [CentralDifference (class in  |     (C++                          |
|     cudaq.gradients)              |     struct)](api/                 |
| ](api/languages/python_api.html#c | languages/cpp_api.html#_CPPv4N5cu |
| udaq.gradients.CentralDifference) | daq10product_op14const_iteratorE) |
| -   [channel                      | -   [cudaq::product_o             |
|     (cudaq.ptsbe.TraceInstruction | p::const_iterator::const_iterator |
|     property)](a                  |     (C++                          |
| pi/languages/python_api.html#cuda |     fu                            |
| q.ptsbe.TraceInstruction.channel) | nction)](api/languages/cpp_api.ht |
| -   [circuit_location             | ml#_CPPv4N5cudaq10product_op14con |
|     (cudaq.ptsbe.KrausSelection   | st_iterator14const_iteratorEPK10p |
|     property)](api/lang           | roduct_opI9HandlerTyENSt6size_tE) |
| uages/python_api.html#cudaq.ptsbe | -   [cudaq::produ                 |
| .KrausSelection.circuit_location) | ct_op::const_iterator::operator!= |
| -   [clear (cudaq.Resources       |     (C++                          |
|                                   |     fun                           |
|   attribute)](api/languages/pytho | ction)](api/languages/cpp_api.htm |
| n_api.html#cudaq.Resources.clear) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(cudaq.SampleResult      | st_iteratorneERK14const_iterator) |
|         a                         | -   [cudaq::produ                 |
| ttribute)](api/languages/python_a | ct_op::const_iterator::operator\* |
| pi.html#cudaq.SampleResult.clear) |     (C++                          |
| -   [CliffordTSequence (class in  |     function)](api/lang           |
|     cudaq.sy                      | uages/cpp_api.html#_CPPv4NK5cudaq |
| nth)](api/languages/python_api.ht | 10product_op14const_iteratormlEv) |
| ml#cudaq.synth.CliffordTSequence) | -   [cudaq::produ                 |
| -   [COBYLA (class in             | ct_op::const_iterator::operator++ |
|     cudaq.o                       |     (C++                          |
| ptimizers)](api/languages/python_ |     function)](api/lang           |
| api.html#cudaq.optimizers.COBYLA) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [coefficient                  | 0product_op14const_iteratorppEi), |
|     (cudaq.                       |     [\[1\]](api/lan               |
| operators.boson.BosonOperatorTerm | guages/cpp_api.html#_CPPv4N5cudaq |
|     property)](api/languages/py   | 10product_op14const_iteratorppEv) |
| thon_api.html#cudaq.operators.bos | -   [cudaq::produc                |
| on.BosonOperatorTerm.coefficient) | t_op::const_iterator::operator\-- |
|     -   [(cudaq.oper              |     (C++                          |
| ators.fermion.FermionOperatorTerm |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|   property)](api/languages/python | 0product_op14const_iteratormmEi), |
| _api.html#cudaq.operators.fermion |     [\[1\]](api/lan               |
| .FermionOperatorTerm.coefficient) | guages/cpp_api.html#_CPPv4N5cudaq |
|     -   [(c                       | 10product_op14const_iteratormmEv) |
| udaq.operators.MatrixOperatorTerm | -   [cudaq::produc                |
|         property)](api/languag    | t_op::const_iterator::operator-\> |
| es/python_api.html#cudaq.operator |     (C++                          |
| s.MatrixOperatorTerm.coefficient) |     function)](api/lan            |
|     -   [(cuda                    | guages/cpp_api.html#_CPPv4N5cudaq |
| q.operators.spin.SpinOperatorTerm | 10product_op14const_iteratorptEv) |
|         property)](api/languages/ | -   [cudaq::produ                 |
| python_api.html#cudaq.operators.s | ct_op::const_iterator::operator== |
| pin.SpinOperatorTerm.coefficient) |     (C++                          |
| -   [col_count                    |     fun                           |
|     (cudaq.KrausOperator          | ction)](api/languages/cpp_api.htm |
|     prope                         | l#_CPPv4NK5cudaq10product_op14con |
| rty)](api/languages/python_api.ht | st_iteratoreqERK14const_iterator) |
| ml#cudaq.KrausOperator.col_count) | -   [cudaq::product_op::degrees   |
| -   [compile()                    |     (C++                          |
|     (cudaq.PyKernelDecorator      |     function)                     |
|     metho                         | ](api/languages/cpp_api.html#_CPP |
| d)](api/languages/python_api.html | v4NK5cudaq10product_op7degreesEv) |
| #cudaq.PyKernelDecorator.compile) | -   [cudaq::product_op::dump (C++ |
|     -   [(in module               |     functi                        |
|         cudaq)](api/languag       | on)](api/languages/cpp_api.html#_ |
| es/python_api.html#cudaq.compile) | CPPv4NK5cudaq10product_op4dumpEv) |
| -   [compiledModuleCache()        | -   [cudaq::product_op::end (C++  |
|     (cudaq.PyKernelDecorator      |     funct                         |
|     method)](api/lang             | ion)](api/languages/cpp_api.html# |
| uages/python_api.html#cudaq.PyKer | _CPPv4NK5cudaq10product_op3endEv) |
| nelDecorator.compiledModuleCache) | -   [c                            |
| -   [ComplexMatrix (class in      | udaq::product_op::get_coefficient |
|     cudaq)](api/languages/pyt     |     (C++                          |
| hon_api.html#cudaq.ComplexMatrix) |     function)](api/lan            |
| -   [compute                      | guages/cpp_api.html#_CPPv4NK5cuda |
|     (                             | q10product_op15get_coefficientEv) |
| cudaq.gradients.CentralDifference | -                                 |
|     attribute)](api/la            |   [cudaq::product_op::get_term_id |
| nguages/python_api.html#cudaq.gra |     (C++                          |
| dients.CentralDifference.compute) |     function)](api                |
|     -   [(                        | /languages/cpp_api.html#_CPPv4NK5 |
| cudaq.gradients.ForwardDifference | cudaq10product_op11get_term_idEv) |
|         attribute)](api/la        | -                                 |
| nguages/python_api.html#cudaq.gra |   [cudaq::product_op::is_identity |
| dients.ForwardDifference.compute) |     (C++                          |
|     -                             |     function)](api                |
|  [(cudaq.gradients.ParameterShift | /languages/cpp_api.html#_CPPv4NK5 |
|         attribute)](api           | cudaq10product_op11is_identityEv) |
| /languages/python_api.html#cudaq. | -   [cudaq::product_op::num_ops   |
| gradients.ParameterShift.compute) |     (C++                          |
| -   [const()                      |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
|   (cudaq.operators.ScalarOperator | v4NK5cudaq10product_op7num_opsEv) |
|     class                         | -                                 |
|     method)](a                    |    [cudaq::product_op::operator\* |
| pi/languages/python_api.html#cuda |     (C++                          |
| q.operators.ScalarOperator.const) |     function)](api/languages/     |
| -   [controls                     | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     (cudaq.ptsbe.TraceInstruction | oduct_opmlE10product_opI1TERK15sc |
|     property)](ap                 | alar_operatorRK10product_opI1TE), |
| i/languages/python_api.html#cudaq |     [\[1\]](api/languages/        |
| .ptsbe.TraceInstruction.controls) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [copy                         | oduct_opmlE10product_opI1TERK15sc |
|     (cu                           | alar_operatorRR10product_opI1TE), |
| daq.operators.boson.BosonOperator |     [\[2\]](api/languages/        |
|     attribute)](api/l             | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| anguages/python_api.html#cudaq.op | oduct_opmlE10product_opI1TERR15sc |
| erators.boson.BosonOperator.copy) | alar_operatorRK10product_opI1TE), |
|     -   [(cudaq.                  |     [\[3\]](api/languages/        |
| operators.boson.BosonOperatorTerm | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|         attribute)](api/langu     | oduct_opmlE10product_opI1TERR15sc |
| ages/python_api.html#cudaq.operat | alar_operatorRR10product_opI1TE), |
| ors.boson.BosonOperatorTerm.copy) |     [\[4\]](api/                  |
|     -   [(cudaq.                  | languages/cpp_api.html#_CPPv4I0EN |
| operators.fermion.FermionOperator | 5cudaq10product_opmlE6sum_opI1TER |
|         attribute)](api/langu     | K15scalar_operatorRK6sum_opI1TE), |
| ages/python_api.html#cudaq.operat |     [\[5\]](api/                  |
| ors.fermion.FermionOperator.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cudaq.oper              | 5cudaq10product_opmlE6sum_opI1TER |
| ators.fermion.FermionOperatorTerm | K15scalar_operatorRR6sum_opI1TE), |
|         attribute)](api/languages |     [\[6\]](api/                  |
| /python_api.html#cudaq.operators. | languages/cpp_api.html#_CPPv4I0EN |
| fermion.FermionOperatorTerm.copy) | 5cudaq10product_opmlE6sum_opI1TER |
|     -                             | R15scalar_operatorRK6sum_opI1TE), |
|  [(cudaq.operators.MatrixOperator |     [\[7\]](api/                  |
|         attribute)](              | languages/cpp_api.html#_CPPv4I0EN |
| api/languages/python_api.html#cud | 5cudaq10product_opmlE6sum_opI1TER |
| aq.operators.MatrixOperator.copy) | R15scalar_operatorRR6sum_opI1TE), |
|     -   [(c                       |     [\[8\]](api/languages         |
| udaq.operators.MatrixOperatorTerm | /cpp_api.html#_CPPv4NK5cudaq10pro |
|         attribute)](api/          | duct_opmlERK6sum_opI9HandlerTyE), |
| languages/python_api.html#cudaq.o |     [\[9\]](api/languages/cpp_a   |
| perators.MatrixOperatorTerm.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [(                        | opmlERK10product_opI9HandlerTyE), |
| cudaq.operators.spin.SpinOperator |     [\[10\]](api/language         |
|         attribute)](api           | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /languages/python_api.html#cudaq. | roduct_opmlERK15scalar_operator), |
| operators.spin.SpinOperator.copy) |     [\[11\]](api/languages/cpp_a  |
|     -   [(cuda                    | pi.html#_CPPv4NKR5cudaq10product_ |
| q.operators.spin.SpinOperatorTerm | opmlERR10product_opI9HandlerTyE), |
|         attribute)](api/lan       |     [\[12\]](api/language         |
| guages/python_api.html#cudaq.oper | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ators.spin.SpinOperatorTerm.copy) | roduct_opmlERR15scalar_operator), |
| -   [count (cudaq.Resources       |     [\[13\]](api/languages/cpp_   |
|                                   | api.html#_CPPv4NO5cudaq10product_ |
|   attribute)](api/languages/pytho | opmlERK10product_opI9HandlerTyE), |
| n_api.html#cudaq.Resources.count) |     [\[14\]](api/languag          |
|     -   [(cudaq.SampleResult      | es/cpp_api.html#_CPPv4NO5cudaq10p |
|         a                         | roduct_opmlERK15scalar_operator), |
| ttribute)](api/languages/python_a |     [\[15\]](api/languages/cpp_   |
| pi.html#cudaq.SampleResult.count) | api.html#_CPPv4NO5cudaq10product_ |
| -   [count_controls               | opmlERR10product_opI9HandlerTyE), |
|     (cudaq.Resources              |     [\[16\]](api/langua           |
|     attribu                       | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| te)](api/languages/python_api.htm | product_opmlERR15scalar_operator) |
| l#cudaq.Resources.count_controls) | -                                 |
| -   [count_instructions           |   [cudaq::product_op::operator\*= |
|                                   |     (C++                          |
|   (cudaq.ptsbe.PTSBEExecutionData |     function)](api/languages/cpp  |
|     attribute)](api/languages/    | _api.html#_CPPv4N5cudaq10product_ |
| python_api.html#cudaq.ptsbe.PTSBE | opmLERK10product_opI9HandlerTyE), |
| ExecutionData.count_instructions) |     [\[1\]](api/langua            |
| -   [counts (cudaq.ObserveResult  | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     att                           | roduct_opmLERK15scalar_operator), |
| ribute)](api/languages/python_api |     [\[2\]](api/languages/cp      |
| .html#cudaq.ObserveResult.counts) | p_api.html#_CPPv4N5cudaq10product |
|     -   [(cudaq.SampleResult      | _opmLERR10product_opI9HandlerTyE) |
|         p                         | -   [cudaq::product_op::operator+ |
| roperty)](api/languages/python_ap |     (C++                          |
| i.html#cudaq.SampleResult.counts) |     function)](api/langu          |
| -   [csr_spmatrix (C++            | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     type)](api/languages/c        | q10product_opplE6sum_opI1TERK15sc |
| pp_api.html#_CPPv412csr_spmatrix) | alar_operatorRK10product_opI1TE), |
| -   cudaq                         |     [\[1\]](api/                  |
|     -   [module](api/langua       | languages/cpp_api.html#_CPPv4I0EN |
| ges/python_api.html#module-cudaq) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq (C++                   | K15scalar_operatorRK6sum_opI1TE), |
|     type)](api/lan                |     [\[2\]](api/langu             |
| guages/cpp_api.html#_CPPv45cudaq) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq.apply_noise() (in      | q10product_opplE6sum_opI1TERK15sc |
|     module                        | alar_operatorRR10product_opI1TE), |
|     cudaq)](api/languages/python_ |     [\[3\]](api/                  |
| api.html#cudaq.cudaq.apply_noise) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.boson                   | 5cudaq10product_opplE6sum_opI1TER |
|     -   [module](api/languages/py | K15scalar_operatorRR6sum_opI1TE), |
| thon_api.html#module-cudaq.boson) |     [\[4\]](api/langu             |
| -   cudaq.fermion                 | ages/cpp_api.html#_CPPv4I0EN5cuda |
|                                   | q10product_opplE6sum_opI1TERR15sc |
|   -   [module](api/languages/pyth | alar_operatorRK10product_opI1TE), |
| on_api.html#module-cudaq.fermion) |     [\[5\]](api/                  |
| -   cudaq.operators.custom        | languages/cpp_api.html#_CPPv4I0EN |
|     -   [mo                       | 5cudaq10product_opplE6sum_opI1TER |
| dule](api/languages/python_api.ht | R15scalar_operatorRK6sum_opI1TE), |
| ml#module-cudaq.operators.custom) |     [\[6\]](api/langu             |
| -   cudaq.spin                    | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     -   [module](api/languages/p  | q10product_opplE6sum_opI1TERR15sc |
| ython_api.html#module-cudaq.spin) | alar_operatorRR10product_opI1TE), |
| -   [cudaq::amplitude_damping     |     [\[7\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     cla                           | 5cudaq10product_opplE6sum_opI1TER |
| ss)](api/languages/cpp_api.html#_ | R15scalar_operatorRR6sum_opI1TE), |
| CPPv4N5cudaq17amplitude_dampingE) |     [\[8\]](api/languages/cpp_a   |
| -                                 | pi.html#_CPPv4NKR5cudaq10product_ |
| [cudaq::amplitude_damping_channel | opplERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[9\]](api/language          |
|     class)](api                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /languages/cpp_api.html#_CPPv4N5c | roduct_opplERK15scalar_operator), |
| udaq25amplitude_damping_channelE) |     [\[10\]](api/languages/       |
| -   [cudaq::amplitud              | cpp_api.html#_CPPv4NKR5cudaq10pro |
| e_damping_channel::num_parameters | duct_opplERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     member)](api/languages/cpp_a  | pi.html#_CPPv4NKR5cudaq10product_ |
| pi.html#_CPPv4N5cudaq25amplitude_ | opplERR10product_opI9HandlerTyE), |
| damping_channel14num_parametersE) |     [\[12\]](api/language         |
| -   [cudaq::ampli                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| tude_damping_channel::num_targets | roduct_opplERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     member)](api/languages/cp     | cpp_api.html#_CPPv4NKR5cudaq10pro |
| p_api.html#_CPPv4N5cudaq25amplitu | duct_opplERR6sum_opI9HandlerTyE), |
| de_damping_channel11num_targetsE) |     [\[                           |
| -   [cudaq::AnalogRemoteRESTQPU   | 14\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NKR5cudaq10product_opplEv), |
|     class                         |     [\[15\]](api/languages/cpp_   |
| )](api/languages/cpp_api.html#_CP | api.html#_CPPv4NO5cudaq10product_ |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq::apply_noise (C++      |     [\[16\]](api/languag          |
|     function)](api/               | es/cpp_api.html#_CPPv4NO5cudaq10p |
| languages/cpp_api.html#_CPPv4I0Dp | roduct_opplERK15scalar_operator), |
| EN5cudaq11apply_noiseEvDpRR4Args) |     [\[17\]](api/languages        |
| -   [cudaq::async_result (C++     | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     c                             | duct_opplERK6sum_opI9HandlerTyE), |
| lass)](api/languages/cpp_api.html |     [\[18\]](api/languages/cpp_   |
| #_CPPv4I0EN5cudaq12async_resultE) | api.html#_CPPv4NO5cudaq10product_ |
| -   [cudaq::async_result::get     | opplERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[19\]](api/languag          |
|     functi                        | es/cpp_api.html#_CPPv4NO5cudaq10p |
| on)](api/languages/cpp_api.html#_ | roduct_opplERR15scalar_operator), |
| CPPv4N5cudaq12async_result3getEv) |     [\[20\]](api/languages        |
| -   [cudaq::async_sample_result   | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opplERR6sum_opI9HandlerTyE), |
|     type                          |     [                             |
| )](api/languages/cpp_api.html#_CP | \[21\]](api/languages/cpp_api.htm |
| Pv4N5cudaq19async_sample_resultE) | l#_CPPv4NO5cudaq10product_opplEv) |
| -   [cudaq::BaseRemoteRESTQPU     | -   [cudaq::product_op::operator- |
|     (C++                          |     (C++                          |
|     cla                           |     function)](api/langu          |
| ss)](api/languages/cpp_api.html#_ | ages/cpp_api.html#_CPPv4I0EN5cuda |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | q10product_opmiE6sum_opI1TERK15sc |
| -   [cudaq::bit_flip_channel (C++ | alar_operatorRK10product_opI1TE), |
|     cl                            |     [\[1\]](api/                  |
| ass)](api/languages/cpp_api.html# | languages/cpp_api.html#_CPPv4I0EN |
| _CPPv4N5cudaq16bit_flip_channelE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq:                       | K15scalar_operatorRK6sum_opI1TE), |
| :bit_flip_channel::num_parameters |     [\[2\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     member)](api/langua           | q10product_opmiE6sum_opI1TERK15sc |
| ges/cpp_api.html#_CPPv4N5cudaq16b | alar_operatorRR10product_opI1TE), |
| it_flip_channel14num_parametersE) |     [\[3\]](api/                  |
| -   [cud                          | languages/cpp_api.html#_CPPv4I0EN |
| aq::bit_flip_channel::num_targets | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRR6sum_opI1TE), |
|     member)](api/lan              |     [\[4\]](api/langu             |
| guages/cpp_api.html#_CPPv4N5cudaq | ages/cpp_api.html#_CPPv4I0EN5cuda |
| 16bit_flip_channel11num_targetsE) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq::boson_handler (C++    | alar_operatorRK10product_opI1TE), |
|                                   |     [\[5\]](api/                  |
|  class)](api/languages/cpp_api.ht | languages/cpp_api.html#_CPPv4I0EN |
| ml#_CPPv4N5cudaq13boson_handlerE) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::boson_op (C++         | R15scalar_operatorRK6sum_opI1TE), |
|     type)](api/languages/cpp_     |     [\[6\]](api/langu             |
| api.html#_CPPv4N5cudaq8boson_opE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::boson_op_term (C++    | q10product_opmiE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|   type)](api/languages/cpp_api.ht |     [\[7\]](api/                  |
| ml#_CPPv4N5cudaq13boson_op_termE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::CodeGenConfig (C++    | 5cudaq10product_opmiE6sum_opI1TER |
|                                   | R15scalar_operatorRR6sum_opI1TE), |
| struct)](api/languages/cpp_api.ht |     [\[8\]](api/languages/cpp_a   |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [cudaq::commutation_relations | opmiERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[9\]](api/language          |
|     struct)]                      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| (api/languages/cpp_api.html#_CPPv | roduct_opmiERK15scalar_operator), |
| 4N5cudaq21commutation_relationsE) |     [\[10\]](api/languages/       |
| -   [cudaq::complex (C++          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     type)](api/languages/cpp      | duct_opmiERK6sum_opI9HandlerTyE), |
| _api.html#_CPPv4N5cudaq7complexE) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq::complex_matrix (C++   | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opmiERR10product_opI9HandlerTyE), |
| class)](api/languages/cpp_api.htm |     [\[12\]](api/language         |
| l#_CPPv4N5cudaq14complex_matrixE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -                                 | roduct_opmiERR15scalar_operator), |
|   [cudaq::complex_matrix::adjoint |     [\[13\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     function)](a                  | duct_opmiERR6sum_opI9HandlerTyE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[                           |
| 5cudaq14complex_matrix7adjointEv) | 14\]](api/languages/cpp_api.html# |
| -   [cudaq::                      | _CPPv4NKR5cudaq10product_opmiEv), |
| complex_matrix::diagonal_elements |     [\[15\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     function)](api/languages      | opmiERK10product_opI9HandlerTyE), |
| /cpp_api.html#_CPPv4NK5cudaq14com |     [\[16\]](api/languag          |
| plex_matrix17diagonal_elementsEi) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::complex_matrix::dump  | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[17\]](api/languages        |
|     function)](api/language       | /cpp_api.html#_CPPv4NO5cudaq10pro |
| s/cpp_api.html#_CPPv4NK5cudaq14co | duct_opmiERK6sum_opI9HandlerTyE), |
| mplex_matrix4dumpERNSt7ostreamE), |     [\[18\]](api/languages/cpp_   |
|     [\[1\]]                       | api.html#_CPPv4NO5cudaq10product_ |
| (api/languages/cpp_api.html#_CPPv | opmiERR10product_opI9HandlerTyE), |
| 4NK5cudaq14complex_matrix4dumpEv) |     [\[19\]](api/languag          |
| -   [c                            | es/cpp_api.html#_CPPv4NO5cudaq10p |
| udaq::complex_matrix::eigenvalues | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[20\]](api/languages        |
|     function)](api/lan            | /cpp_api.html#_CPPv4NO5cudaq10pro |
| guages/cpp_api.html#_CPPv4NK5cuda | duct_opmiERR6sum_opI9HandlerTyE), |
| q14complex_matrix11eigenvaluesEv) |     [                             |
| -   [cu                           | \[21\]](api/languages/cpp_api.htm |
| daq::complex_matrix::eigenvectors | l#_CPPv4NO5cudaq10product_opmiEv) |
|     (C++                          | -   [cudaq::product_op::operator/ |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     function)](api/language       |
| 14complex_matrix12eigenvectorsEv) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [c                            | roduct_opdvERK15scalar_operator), |
| udaq::complex_matrix::exponential |     [\[1\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/la             | roduct_opdvERR15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[2\]](api/languag           |
| q14complex_matrix11exponentialEv) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -                                 | roduct_opdvERK15scalar_operator), |
|  [cudaq::complex_matrix::identity |     [\[3\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     function)](api/languages      | product_opdvERR15scalar_operator) |
| /cpp_api.html#_CPPv4N5cudaq14comp | -                                 |
| lex_matrix8identityEKNSt6size_tE) |    [cudaq::product_op::operator/= |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::kronecker |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     function)](api/lang           | product_opdVERK15scalar_operator) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -   [cudaq::product_op::operator= |
| daq14complex_matrix9kroneckerE14c |     (C++                          |
| omplex_matrix8Iterable8Iterable), |     function)](api/l              |
|     [\[1\]](api/l                 | anguages/cpp_api.html#_CPPv4I00EN |
| anguages/cpp_api.html#_CPPv4N5cud | 5cudaq10product_opaSER10product_o |
| aq14complex_matrix9kroneckerERK14 | pI9HandlerTyERK10product_opI1TE), |
| complex_matrixRK14complex_matrix) |     [\[1\]](api/languages/cpp     |
| -   [cudaq::c                     | _api.html#_CPPv4N5cudaq10product_ |
| omplex_matrix::minimal_eigenvalue | opaSERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     function)](api/languages/     | p_api.html#_CPPv4N5cudaq10product |
| cpp_api.html#_CPPv4NK5cudaq14comp | _opaSERR10product_opI9HandlerTyE) |
| lex_matrix18minimal_eigenvalueEv) | -                                 |
| -   [                             |    [cudaq::product_op::operator== |
| cudaq::complex_matrix::operator() |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api/languages/cpp  | _api.html#_CPPv4NK5cudaq10product |
| _api.html#_CPPv4N5cudaq14complex_ | _opeqERK10product_opI9HandlerTyE) |
| matrixclENSt6size_tENSt6size_tE), | -                                 |
|     [\[1\]](api/languages/cpp     |  [cudaq::product_op::operator\[\] |
| _api.html#_CPPv4NK5cudaq14complex |     (C++                          |
| _matrixclENSt6size_tENSt6size_tE) |     function)](ap                 |
| -   [                             | i/languages/cpp_api.html#_CPPv4NK |
| cudaq::complex_matrix::operator\* | 5cudaq10product_opixENSt6size_tE) |
|     (C++                          | -                                 |
|     function)](api/langua         |    [cudaq::product_op::product_op |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     (C++                          |
| omplex_matrixmlEN14complex_matrix |     f                             |
| 10value_typeERK14complex_matrix), | unction)](api/languages/cpp_api.h |
|     [\[1\]                        | tml#_CPPv4I00EN5cudaq10product_op |
| ](api/languages/cpp_api.html#_CPP | 10product_opERK10product_opI1TE), |
| v4N5cudaq14complex_matrixmlERK14c |     [\[1\]]                       |
| omplex_matrixRK14complex_matrix), | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I00EN5cudaq10product_op10product |
|  [\[2\]](api/languages/cpp_api.ht | _opERK10product_opI1TERKN14matrix |
| ml#_CPPv4N5cudaq14complex_matrixm | _handler20commutation_behaviorE), |
| lERK14complex_matrixRKNSt6vectorI |                                   |
| N14complex_matrix10value_typeEEE) |   [\[2\]](api/languages/cpp_api.h |
| -                                 | tml#_CPPv4N5cudaq10product_op10pr |
| [cudaq::complex_matrix::operator+ | oduct_opENSt6size_tENSt6size_tE), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     function                      | p_api.html#_CPPv4N5cudaq10product |
| )](api/languages/cpp_api.html#_CP | _op10product_opENSt7complexIdEE), |
| Pv4N5cudaq14complex_matrixplERK14 |     [\[4\]](api/l                 |
| complex_matrixRK14complex_matrix) | anguages/cpp_api.html#_CPPv4N5cud |
| -                                 | aq10product_op10product_opERK10pr |
| [cudaq::complex_matrix::operator- | oduct_opI9HandlerTyENSt6size_tE), |
|     (C++                          |     [\[5\]](api/l                 |
|     function                      | anguages/cpp_api.html#_CPPv4N5cud |
| )](api/languages/cpp_api.html#_CP | aq10product_op10product_opERR10pr |
| Pv4N5cudaq14complex_matrixmiERK14 | oduct_opI9HandlerTyENSt6size_tE), |
| complex_matrixRK14complex_matrix) |     [\[6\]](api/languages         |
| -   [cu                           | /cpp_api.html#_CPPv4N5cudaq10prod |
| daq::complex_matrix::operator\[\] | uct_op10product_opERR9HandlerTy), |
|     (C++                          |     [\[7\]](ap                    |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|  function)](api/languages/cpp_api | cudaq10product_op10product_opEd), |
| .html#_CPPv4N5cudaq14complex_matr |     [\[8\]](a                     |
| ixixERKNSt6vectorINSt6size_tEEE), | pi/languages/cpp_api.html#_CPPv4N |
|     [\[1\]](api/languages/cpp_api | 5cudaq10product_op10product_opEv) |
| .html#_CPPv4NK5cudaq14complex_mat | -   [cuda                         |
| rixixERKNSt6vectorINSt6size_tEEE) | q::product_op::to_diagonal_matrix |
| -   [cudaq::complex_matrix::power |     (C++                          |
|     (C++                          |     function)](api/               |
|     function)]                    | languages/cpp_api.html#_CPPv4NK5c |
| (api/languages/cpp_api.html#_CPPv | udaq10product_op18to_diagonal_mat |
| 4N5cudaq14complex_matrix5powerEi) | rixENSt13unordered_mapINSt6size_t |
| -                                 | ENSt7int64_tEEERKNSt13unordered_m |
|  [cudaq::complex_matrix::set_zero | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cudaq::product_op::to_matrix |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     funct                         |
| cudaq14complex_matrix8set_zeroEv) | ion)](api/languages/cpp_api.html# |
| -                                 | _CPPv4NK5cudaq10product_op9to_mat |
| [cudaq::complex_matrix::to_string | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/               | apINSt6stringENSt7complexIdEEEEb) |
| languages/cpp_api.html#_CPPv4NK5c | -   [cu                           |
| udaq14complex_matrix9to_stringEv) | daq::product_op::to_sparse_matrix |
| -   [                             |     (C++                          |
| cudaq::complex_matrix::value_type |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     type)](api/                   | 5cudaq10product_op16to_sparse_mat |
| languages/cpp_api.html#_CPPv4N5cu | rixENSt13unordered_mapINSt6size_t |
| daq14complex_matrix10value_typeE) | ENSt7int64_tEEERKNSt13unordered_m |
| -   [cudaq::contrib (C++          | apINSt6stringENSt7complexIdEEEEb) |
|     type)](api/languages/cpp      | -   [cudaq::product_op::to_string |
| _api.html#_CPPv4N5cudaq7contribE) |     (C++                          |
| -                                 |     function)](                   |
| [cudaq::contrib::amplitude_encode | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq10product_op9to_stringEv) |
|     function)](api/language       | -                                 |
| s/cpp_api.html#_CPPv4N5cudaq7cont |  [cudaq::product_op::\~product_op |
| rib16amplitude_encodeENSt4spanIKN |     (C++                          |
| St7complexIdEEEENSt7complexIdEE), |     fu                            |
|     [\[1\]](api/language          | nction)](api/languages/cpp_api.ht |
| s/cpp_api.html#_CPPv4N5cudaq7cont | ml#_CPPv4N5cudaq10product_opD0Ev) |
| rib16amplitude_encodeENSt4spanIKN | -   [cudaq::ptsbe (C++            |
| St7complexIfEEEENSt7complexIdEE), |     type)](api/languages/c        |
|     [\[2\]                        | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::p                     |
| v4N5cudaq7contrib16amplitude_enco | tsbe::ConditionalSamplingStrategy |
| deENSt4spanIKdEENSt7complexIdEE), |     (C++                          |
|     [\[3\]                        |     class)](api/languag           |
| ](api/languages/cpp_api.html#_CPP | es/cpp_api.html#_CPPv4N5cudaq5pts |
| v4N5cudaq7contrib16amplitude_enco | be27ConditionalSamplingStrategyE) |
| deENSt4spanIKfEENSt7complexIdEE), | -   [cudaq::ptsbe::C              |
|                                   | onditionalSamplingStrategy::clone |
| [\[4\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq7contrib16amplitud |                                   |
| e_encodeERK5stateNSt7complexIdEE) |    function)](api/languages/cpp_a |
| -                                 | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
|   [cudaq::contrib::angular_encode | ditionalSamplingStrategy5cloneEv) |
|     (C++                          | -   [cuda                         |
|                                   | q::ptsbe::ConditionalSamplingStra |
|  function)](api/languages/cpp_api | tegy::ConditionalSamplingStrategy |
| .html#_CPPv4I0EN5cudaq7contrib14a |     (C++                          |
| ngular_encodeEvRR6KernelR10QuakeV |     function)](api/lang           |
| alueNSt4spanIKdEE12RotationAxis), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     [\[1\]](api/languages/cpp_api | ptsbe27ConditionalSamplingStrateg |
| .html#_CPPv4I0EN5cudaq7contrib14a | y27ConditionalSamplingStrategyE19 |
| ngular_encodeEvRR6KernelR10QuakeV | TrajectoryPredicateNSt8uint64_tE) |
| alueR10QuakeValue12RotationAxis), | -                                 |
|                                   |   [cudaq::ptsbe::ConditionalSampl |
|   [\[2\]](api/languages/cpp_api.h | ingStrategy::generateTrajectories |
| tml#_CPPv4I0EN5cudaq7contrib14ang |     (C++                          |
| ular_encodeEvRR6KernelR10QuakeVal |     function)](api/language       |
| ueRKNSt6vectorIdEE12RotationAxis) | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| -   [cudaq::contrib::draw (C++    | be27ConditionalSamplingStrategy20 |
|     function)                     | generateTrajectoriesENSt4spanIKN6 |
| ](api/languages/cpp_api.html#_CPP | detail10NoisePointEEENSt6size_tE) |
| v4I0DpEN5cudaq7contrib4drawENSt6s | -   [cudaq::ptsbe::               |
| tringERR13QuantumKernelDpRR4Args) | ConditionalSamplingStrategy::name |
| -                                 |     (C++                          |
| [cudaq::contrib::get_unitary_cmat |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4NK5cudaq5ptsbe27Co |
|     function)](api/languages/cp   | nditionalSamplingStrategy4nameEv) |
| p_api.html#_CPPv4I0DpEN5cudaq7con | -   [cudaq:                       |
| trib16get_unitary_cmatE14complex_ | :ptsbe::ConditionalSamplingStrate |
| matrixRR13QuantumKernelDpRR4Args) | gy::\~ConditionalSamplingStrategy |
| -   [cudaq::contrib::RotationAxis |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     enum)                         | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| ](api/languages/cpp_api.html#_CPP | 7ConditionalSamplingStrategyD0Ev) |
| v4N5cudaq7contrib12RotationAxisE) | -                                 |
| -                                 | [cudaq::ptsbe::detail::NoisePoint |
|  [cudaq::contrib::RotationAxis::X |     (C++                          |
|     (C++                          |     struct)](a                    |
|     enumerator)](                 | pi/languages/cpp_api.html#_CPPv4N |
| api/languages/cpp_api.html#_CPPv4 | 5cudaq5ptsbe6detail10NoisePointE) |
| N5cudaq7contrib12RotationAxis1XE) | -   [cudaq::p                     |
| -                                 | tsbe::detail::NoisePoint::channel |
|  [cudaq::contrib::RotationAxis::Y |     (C++                          |
|     (C++                          |     member)](api/langu            |
|     enumerator)](                 | ages/cpp_api.html#_CPPv4N5cudaq5p |
| api/languages/cpp_api.html#_CPPv4 | tsbe6detail10NoisePoint7channelE) |
| N5cudaq7contrib12RotationAxis1YE) | -   [cudaq::ptsbe::det            |
| -                                 | ail::NoisePoint::circuit_location |
|  [cudaq::contrib::RotationAxis::Z |     (C++                          |
|     (C++                          |     member)](api/languages/cpp_a  |
|     enumerator)](                 | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| api/languages/cpp_api.html#_CPPv4 | l10NoisePoint16circuit_locationE) |
| N5cudaq7contrib12RotationAxis1ZE) | -   [cudaq::p                     |
| -   [cudaq::cudaq_json (C++       | tsbe::detail::NoisePoint::op_name |
|     class)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4N5cudaq10cudaq_jsonE) |     member)](api/langu            |
| -   [cudaq::DefaultQPU (C++       | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     class)](api/languages/cpp_api | tsbe6detail10NoisePoint7op_nameE) |
| .html#_CPPv4N5cudaq10DefaultQPUE) | -   [cudaq::                      |
| -   [cudaq::dem_from_kernel (C++  | ptsbe::detail::NoisePoint::qubits |
|     function)](api                |     (C++                          |
| /languages/cpp_api.html#_CPPv4I0D |     member)](api/lang             |
| pEN5cudaq15dem_from_kernelENSt6st | uages/cpp_api.html#_CPPv4N5cudaq5 |
| ringERR13QuantumKernelDpRR4Args), | ptsbe6detail10NoisePoint6qubitsE) |
|     [                             | -   [cudaq::                      |
| \[1\]](api/languages/cpp_api.html | ptsbe::ExhaustiveSamplingStrategy |
| #_CPPv4I0DpEN5cudaq15dem_from_ker |     (C++                          |
| nelENSt6stringERR13QuantumKernelP |     class)](api/langua            |
| KN5cudaq11noise_modelEDpRR4Args), | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     [\[2\]](api/languages/cp      | sbe26ExhaustiveSamplingStrategyE) |
| p_api.html#_CPPv4I0DpEN5cudaq15de | -   [cudaq::ptsbe::               |
| m_from_kernelENSt6stringERR13Quan | ExhaustiveSamplingStrategy::clone |
| tumKernelPKN5cudaq11noise_modelER |     (C++                          |
| KN5cudaq11dem_optionsEDpRR4Args), |     function)](api/languages/cpp_ |
|     [\[3\]](ap                    | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| i/languages/cpp_api.html#_CPPv4I0 | haustiveSamplingStrategy5cloneEv) |
| DpEN5cudaq15dem_from_kernelENSt6s | -   [cu                           |
| tringERR13QuantumKernelPKN5cudaq1 | daq::ptsbe::ExhaustiveSamplingStr |
| 1noise_modelERKN5cudaq11dem_optio | ategy::ExhaustiveSamplingStrategy |
| nsERN5cudaq15M2DSparseMatrixERN5c |     (C++                          |
| udaq15M2OSparseMatrixEDpRR4Args), |     function)](api/la             |
|     [\[4\]](api/language          | nguages/cpp_api.html#_CPPv4N5cuda |
| s/cpp_api.html#_CPPv4I0DpEN5cudaq | q5ptsbe26ExhaustiveSamplingStrate |
| 15dem_from_kernelENSt6stringERR13 | gy26ExhaustiveSamplingStrategyEv) |
| QuantumKernelPKN5cudaq11noise_mod | -                                 |
| elERN5cudaq15M2DSparseMatrixERN5c |    [cudaq::ptsbe::ExhaustiveSampl |
| udaq15M2OSparseMatrixEDpRR4Args), | ingStrategy::generateTrajectories |
|     [\[5\]](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I0DpEN5cudaq15dem_fro |     function)](api/languag        |
| m_kernelENSt6stringERR13QuantumKe | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| rnelRN5cudaq15M2DSparseMatrixERN5 | sbe26ExhaustiveSamplingStrategy20 |
| cudaq15M2OSparseMatrixEDpRR4Args) | generateTrajectoriesENSt4spanIKN6 |
| -   [cudaq::dem_options (C++      | detail10NoisePointEEENSt6size_tE) |
|                                   | -   [cudaq::ptsbe:                |
|   struct)](api/languages/cpp_api. | :ExhaustiveSamplingStrategy::name |
| html#_CPPv4N5cudaq11dem_optionsE) |     (C++                          |
| -   [cudaq::d                     |     function)](api/languages/cpp  |
| em_options::allow_gauge_detectors | _api.html#_CPPv4NK5cudaq5ptsbe26E |
|     (C++                          | xhaustiveSamplingStrategy4nameEv) |
|     member)](api/language         | -   [cuda                         |
| s/cpp_api.html#_CPPv4N5cudaq11dem | q::ptsbe::ExhaustiveSamplingStrat |
| _options21allow_gauge_detectorsE) | egy::\~ExhaustiveSamplingStrategy |
| -   [cudaq::dem_options::appr     |     (C++                          |
| oximate_disjoint_errors_threshold |     function)](api/languages      |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     memb                          | 26ExhaustiveSamplingStrategyD0Ev) |
| er)](api/languages/cpp_api.html#_ | -   [cuda                         |
| CPPv4N5cudaq11dem_options37approx | q::ptsbe::OrderedSamplingStrategy |
| imate_disjoint_errors_thresholdE) |     (C++                          |
| -   [cuda                         |     class)](api/lan               |
| q::dem_options::block_decompositi | guages/cpp_api.html#_CPPv4N5cudaq |
| on_from_introducing_remnant_edges | 5ptsbe23OrderedSamplingStrategyE) |
|     (C++                          | -   [cudaq::ptsb                  |
|     member)](api/lang             | e::OrderedSamplingStrategy::clone |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     (C++                          |
| 1dem_options50block_decomposition |     function)](api/languages/c    |
| _from_introducing_remnant_edgesE) | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| -   [cud                          | 3OrderedSamplingStrategy5cloneEv) |
| aq::dem_options::decompose_errors | -   [cudaq::ptsbe::OrderedSampl   |
|     (C++                          | ingStrategy::generateTrajectories |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/lang           |
| 11dem_options16decompose_errorsE) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -                                 | 5ptsbe23OrderedSamplingStrategy20 |
|   [cudaq::dem_options::fold_loops | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     member)](a                    | -   [cudaq::pts                   |
| pi/languages/cpp_api.html#_CPPv4N | be::OrderedSamplingStrategy::name |
| 5cudaq11dem_options10fold_loopsE) |     (C++                          |
| -   [cudaq::dem_optio             |     function)](api/languages/     |
| ns::ignore_decomposition_failures | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     (C++                          | 23OrderedSamplingStrategy4nameEv) |
|     member)](api/languages/cpp_ap | -                                 |
| i.html#_CPPv4N5cudaq11dem_options |    [cudaq::ptsbe::OrderedSampling |
| 29ignore_decomposition_failuresE) | Strategy::OrderedSamplingStrategy |
| -   [cudaq::dem_opt               |     (C++                          |
| ions::return_measurement_matrices |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     member)](api/languages/cpp_   | N5cudaq5ptsbe23OrderedSamplingStr |
| api.html#_CPPv4N5cudaq11dem_optio | ategy23OrderedSamplingStrategyEv) |
| ns27return_measurement_matricesE) | -                                 |
| -   [cudaq::depolarization1 (C++  |  [cudaq::ptsbe::OrderedSamplingSt |
|     c                             | rategy::\~OrderedSamplingStrategy |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15depolarization1E) |     function)](api/langua         |
| -   [cudaq::depolarization2 (C++  | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     c                             | sbe23OrderedSamplingStrategyD0Ev) |
| lass)](api/languages/cpp_api.html | -   [cudaq::pts                   |
| #_CPPv4N5cudaq15depolarization2E) | be::ProbabilisticSamplingStrategy |
| -   [cudaq:                       |     (C++                          |
| :depolarization2::depolarization2 |     class)](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     function)](api/languages/cp   | 29ProbabilisticSamplingStrategyE) |
| p_api.html#_CPPv4N5cudaq15depolar | -   [cudaq::ptsbe::Pro            |
| ization215depolarization2EK4real) | babilisticSamplingStrategy::clone |
| -   [cudaq                        |     (C++                          |
| ::depolarization2::num_parameters |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     member)](api/langu            | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| ages/cpp_api.html#_CPPv4N5cudaq15 | bilisticSamplingStrategy5cloneEv) |
| depolarization214num_parametersE) | -                                 |
| -   [cu                           | [cudaq::ptsbe::ProbabilisticSampl |
| daq::depolarization2::num_targets | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api/languages/     |
| nguages/cpp_api.html#_CPPv4N5cuda | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| q15depolarization211num_targetsE) | 29ProbabilisticSamplingStrategy20 |
| -                                 | generateTrajectoriesENSt4spanIKN6 |
|    [cudaq::depolarization_channel | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe::Pr             |
|     class)](                      | obabilisticSamplingStrategy::name |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22depolarization_channelE) |                                   |
| -   [cudaq::depol                 |   function)](api/languages/cpp_ap |
| arization_channel::num_parameters | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
|     (C++                          | abilisticSamplingStrategy4nameEv) |
|     member)](api/languages/cp     | -   [cudaq::p                     |
| p_api.html#_CPPv4N5cudaq22depolar | tsbe::ProbabilisticSamplingStrate |
| ization_channel14num_parametersE) | gy::ProbabilisticSamplingStrategy |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     member)](api/languages        | 4N5cudaq5ptsbe29ProbabilisticSamp |
| /cpp_api.html#_CPPv4N5cudaq22depo | lingStrategy29ProbabilisticSampli |
| larization_channel11num_targetsE) | ngStrategyENSt8optionalINSt8uint6 |
| -   [cudaq::detail (C++           | 4_tEEENSt8optionalINSt6size_tEEE) |
|     type)](api/languages/cp       | -   [cudaq::pts                   |
| p_api.html#_CPPv4N5cudaq6detailE) | be::ProbabilisticSamplingStrategy |
| -   [cudaq::detail::future (C++   | ::\~ProbabilisticSamplingStrategy |
|                                   |     (C++                          |
|   class)](api/languages/cpp_api.h |     function)](api/languages/cp   |
| tml#_CPPv4N5cudaq6detail6futureE) | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| -                                 | robabilisticSamplingStrategyD0Ev) |
|    [cudaq::detail::future::future | -                                 |
|     (C++                          | [cudaq::ptsbe::PTSBEExecutionData |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     struct)](ap                   |
| CPPv4N5cudaq6detail6future6future | i/languages/cpp_api.html#_CPPv4N5 |
| ERNSt6vectorI3JobEERNSt6stringERN | cudaq5ptsbe18PTSBEExecutionDataE) |
| St3mapINSt6stringENSt6stringEEE), | -   [cudaq::ptsbe::PTSBE          |
|     [\[1\]](api/lan               | ExecutionData::count_instructions |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 6detail6future6futureERR6future), |     function)](api/l              |
|     [\[2\]                        | anguages/cpp_api.html#_CPPv4NK5cu |
| ](api/languages/cpp_api.html#_CPP | daq5ptsbe18PTSBEExecutionData18co |
| v4N5cudaq6detail6future6futureEv) | unt_instructionsE20TraceInstructi |
| -   [c                            | onTypeNSt8optionalINSt6stringEEE) |
| udaq::detail::kernel_builder_base | -   [cudaq::ptsbe::P              |
|     (C++                          | TSBEExecutionData::get_trajectory |
|     class)](api/                  |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function                      |
| daq6detail19kernel_builder_baseE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::detail::              | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| kernel_builder_base::operator\<\< | Data14get_trajectoryENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe:                |
|     function)](api/langu          | :PTSBEExecutionData::instructions |
| ages/cpp_api.html#_CPPv4N5cudaq6d |     (C++                          |
| etail19kernel_builder_baselsERNSt |     member)](api/languages/cp     |
| 7ostreamERK19kernel_builder_base) | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| -                                 | TSBEExecutionData12instructionsE) |
| [cudaq::detail::KernelBuilderType | -   [cudaq::ptsbe:                |
|     (C++                          | :PTSBEExecutionData::trajectories |
|     class)](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     member)](api/languages/cp     |
| cudaq6detail17KernelBuilderTypeE) | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| -   [cudaq::                      | TSBEExecutionData12trajectoriesE) |
| detail::KernelBuilderType::create | -   [cudaq::ptsbe::PTSBEOptions   |
|     (C++                          |     (C++                          |
|     function                      |     struc                         |
| )](api/languages/cpp_api.html#_CP | t)](api/languages/cpp_api.html#_C |
| Pv4N5cudaq6detail17KernelBuilderT | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| ype6createEPN4mlir11MLIRContextE) | -   [cudaq::ptsbe::PTSB           |
| -   [cudaq::detail::Ker           | EOptions::include_sequential_data |
| nelBuilderType::KernelBuilderType |     (C++                          |
|     (C++                          |                                   |
|     function)](api/lan            |    member)](api/languages/cpp_api |
| guages/cpp_api.html#_CPPv4N5cudaq | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| 6detail17KernelBuilderType17Kerne | ptions23include_sequential_dataE) |
| lBuilderTypeERRNSt8functionIFN4ml | -   [cudaq::ptsb                  |
| ir4TypeEPN4mlir11MLIRContextEEEE) | e::PTSBEOptions::max_trajectories |
| -   [cudaq::detector (C++         |     (C++                          |
|     function)](api                |     member)](api/languages/       |
| /languages/cpp_api.html#_CPPv4IDp | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| EN5cudaq8detectorEvDpRR8MeasArgs) | 2PTSBEOptions16max_trajectoriesE) |
| -   [cudaq::detectors (C++        | -   [cudaq::ptsbe::PT             |
|     function)](api/languages/c    | SBEOptions::return_execution_data |
| pp_api.html#_CPPv4N5cudaq9detecto |     (C++                          |
| rsERKNSt6vectorI14measure_resultE |     member)](api/languages/cpp_a  |
| ERKNSt6vectorI14measure_resultEE) | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| -   [cudaq::diag_matrix_callback  | EOptions21return_execution_dataE) |
|     (C++                          | -   [cudaq::pts                   |
|     class)                        | be::PTSBEOptions::shot_allocation |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq20diag_matrix_callbackE) |     member)](api/languages        |
| -   [cudaq::dyn (C++              | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     member)](api/languages        | 12PTSBEOptions15shot_allocationE) |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | -   [cud                          |
| -   [cudaq::ExecutionContext (C++ | aq::ptsbe::PTSBEOptions::strategy |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     member)](api/l                |
| _CPPv4N5cudaq16ExecutionContextE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [c                            | aq5ptsbe12PTSBEOptions8strategyE) |
| udaq::ExecutionContext::asyncExec | -   [cudaq::ptsbe::PTSBETrace     |
|     (C++                          |     (C++                          |
|     member)](api/                 |     t                             |
| languages/cpp_api.html#_CPPv4N5cu | ype)](api/languages/cpp_api.html# |
| daq16ExecutionContext9asyncExecE) | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| -   [cud                          | -   [                             |
| aq::ExecutionContext::asyncResult | cudaq::ptsbe::PTSSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     class)](api                   |
| guages/cpp_api.html#_CPPv4N5cudaq | /languages/cpp_api.html#_CPPv4N5c |
| 16ExecutionContext11asyncResultE) | udaq5ptsbe19PTSSamplingStrategyE) |
| -   [cudaq:                       | -   [cudaq::                      |
| :ExecutionContext::batchIteration | ptsbe::PTSSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     member)](api/langua           |     function)](api/languag        |
| ges/cpp_api.html#_CPPv4N5cudaq16E | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| xecutionContext14batchIterationE) | sbe19PTSSamplingStrategy5cloneEv) |
| -   [cudaq::E                     | -   [cudaq::ptsbe::PTSSampl       |
| xecutionContext::canHandleObserve | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     member)](api/language         |     function)](api/               |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | languages/cpp_api.html#_CPPv4NK5c |
| cutionContext16canHandleObserveE) | udaq5ptsbe19PTSSamplingStrategy20 |
| -   [cudaq::Executio              | generateTrajectoriesENSt4spanIKN6 |
| nContext::deferredKernelException | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     member)](api/languages/cpp_a  | :ptsbe::PTSSamplingStrategy::name |
| pi.html#_CPPv4N5cudaq16ExecutionC |     (C++                          |
| ontext23deferredKernelExceptionE) |     function)](api/langua         |
| -   [cudaq::E                     | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| xecutionContext::ExecutionContext | tsbe19PTSSamplingStrategy4nameEv) |
|     (C++                          | -   [cudaq::ptsbe::PTSSampli      |
|     func                          | ngStrategy::\~PTSSamplingStrategy |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq16ExecutionContext1 |     function)](api/la             |
| 6ExecutionContextERKNSt6stringE), | nguages/cpp_api.html#_CPPv4N5cuda |
|     [\[1\]](api/languages/        | q5ptsbe19PTSSamplingStrategyD0Ev) |
| cpp_api.html#_CPPv4N5cudaq16Execu | -   [cudaq::ptsbe::sample (C++    |
| tionContext16ExecutionContextERKN |                                   |
| St6stringENSt6size_tENSt6size_tE) |  function)](api/languages/cpp_api |
| -   [cudaq::Execu                 | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| tionContext::explicitMeasurements | mpleE13sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](api/languages/cp     |     [\[1\]](api                   |
| p_api.html#_CPPv4N5cudaq16Executi | /languages/cpp_api.html#_CPPv4I0D |
| onContext20explicitMeasurementsE) | pEN5cudaq5ptsbe6sampleE13sample_r |
| -   [cuda                         | esultRKN5cudaq11noise_modelENSt6s |
| q::ExecutionContext::futureResult | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_async   |
|     member)](api/lang             |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     function)](a                  |
| 6ExecutionContext12futureResultE) | pi/languages/cpp_api.html#_CPPv4I |
| -   [cudaq::ExecutionContext      | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| ::hasConditionalsOnMeasureResults | 9async_sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     mem                           |     [\[1\]](api/languages/cp      |
| ber)](api/languages/cpp_api.html# | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| _CPPv4N5cudaq16ExecutionContext31 | be12sample_asyncE19async_sample_r |
| hasConditionalsOnMeasureResultsE) | esultRKN5cudaq11noise_modelENSt6s |
| -   [cudaq:                       | ize_tERR13QuantumKernelDpRR4Args) |
| :ExecutionContext::inKernelLaunch | -   [cudaq::ptsbe::sample_options |
|     (C++                          |     (C++                          |
|     member)](api/langua           |     struct)                       |
| ges/cpp_api.html#_CPPv4N5cudaq16E | ](api/languages/cpp_api.html#_CPP |
| xecutionContext14inKernelLaunchE) | v4N5cudaq5ptsbe14sample_optionsE) |
| -   [cu                           | -   [cudaq::ptsbe::sample_result  |
| daq::ExecutionContext::kernelName |     (C++                          |
|     (C++                          |     class                         |
|     member)](api/la               | )](api/languages/cpp_api.html#_CP |
| nguages/cpp_api.html#_CPPv4N5cuda | Pv4N5cudaq5ptsbe13sample_resultE) |
| q16ExecutionContext10kernelNameE) | -   [cudaq::pts                   |
| -   [cud                          | be::sample_result::execution_data |
| aq::ExecutionContext::kernelTrace |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     member)](api/lan              | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| guages/cpp_api.html#_CPPv4N5cudaq | 3sample_result14execution_dataEv) |
| 16ExecutionContext11kernelTraceE) | -   [cudaq::ptsbe::               |
| -                                 | sample_result::has_execution_data |
|    [cudaq::ExecutionContext::name |     (C++                          |
|     (C++                          |                                   |
|     member)]                      |    function)](api/languages/cpp_a |
| (api/languages/cpp_api.html#_CPPv | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| 4N5cudaq16ExecutionContext4nameE) | ple_result18has_execution_dataEv) |
| -   [cu                           | -   [cudaq::pt                    |
| daq::ExecutionContext::noiseModel | sbe::sample_result::sample_result |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api/l              |
| nguages/cpp_api.html#_CPPv4N5cuda | anguages/cpp_api.html#_CPPv4N5cud |
| q16ExecutionContext10noiseModelE) | aq5ptsbe13sample_result13sample_r |
| -   [cudaq::Exe                   | esultERRN5cudaq13sample_resultE), |
| cutionContext::numberTrajectories |                                   |
|     (C++                          |  [\[1\]](api/languages/cpp_api.ht |
|     member)](api/languages/       | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| cpp_api.html#_CPPv4N5cudaq16Execu | sult13sample_resultERRN5cudaq13sa |
| tionContext18numberTrajectoriesE) | mple_resultE18PTSBEExecutionData) |
| -   [c                            | -   [cudaq::ptsbe::               |
| udaq::ExecutionContext::optResult | sample_result::set_execution_data |
|     (C++                          |     (C++                          |
|     member)](api/                 |     function)](api/               |
| languages/cpp_api.html#_CPPv4N5cu | languages/cpp_api.html#_CPPv4N5cu |
| daq16ExecutionContext9optResultE) | daq5ptsbe13sample_result18set_exe |
| -                                 | cution_dataE18PTSBEExecutionData) |
|   [cudaq::ExecutionContext::qpuId | -   [cud                          |
|     (C++                          | aq::ptsbe::ShotAllocationStrategy |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     struct)](using                |
| N5cudaq16ExecutionContext5qpuIdE) | /examples/ptsbe.html#_CPPv4N5cuda |
| -   [cudaq                        | q5ptsbe22ShotAllocationStrategyE) |
| ::ExecutionContext::registerNames | -   [cudaq::ptsbe::ShotAllocatio  |
|     (C++                          | nStrategy::ShotAllocationStrategy |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     function)                     |
| ExecutionContext13registerNamesE) | ](using/examples/ptsbe.html#_CPPv |
| -   [cu                           | 4N5cudaq5ptsbe22ShotAllocationStr |
| daq::ExecutionContext::reorderIdx | ategy22ShotAllocationStrategyE4Ty |
|     (C++                          | pedNSt8optionalINSt8uint64_tEEE), |
|     member)](api/la               |     [\[1\                         |
| nguages/cpp_api.html#_CPPv4N5cuda | ]](using/examples/ptsbe.html#_CPP |
| q16ExecutionContext10reorderIdxE) | v4N5cudaq5ptsbe22ShotAllocationSt |
| -                                 | rategy22ShotAllocationStrategyEv) |
|   [cudaq::ExecutionContext::shots | -   [cudaq::pt                    |
|     (C++                          | sbe::ShotAllocationStrategy::Type |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     enum)](using/exam             |
| N5cudaq16ExecutionContext5shotsE) | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| -   [cudaq::                      | be22ShotAllocationStrategy4TypeE) |
| ExecutionContext::simulationState | -   [cudaq::ptsbe::ShotAllocatio  |
|     (C++                          | nStrategy::Type::HIGH_WEIGHT_BIAS |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     enumerat                      |
| ecutionContext15simulationStateE) | or)](using/examples/ptsbe.html#_C |
| -                                 | PPv4N5cudaq5ptsbe22ShotAllocation |
|    [cudaq::ExecutionContext::spin | Strategy4Type16HIGH_WEIGHT_BIASE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAllocati   |
|     member)]                      | onStrategy::Type::LOW_WEIGHT_BIAS |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq16ExecutionContext4spinE) |     enumera                       |
| -   [cudaq::                      | tor)](using/examples/ptsbe.html#_ |
| ExecutionContext::totalIterations | CPPv4N5cudaq5ptsbe22ShotAllocatio |
|     (C++                          | nStrategy4Type15LOW_WEIGHT_BIASE) |
|     member)](api/languag          | -   [cudaq::ptsbe::ShotAlloc      |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | ationStrategy::Type::PROPORTIONAL |
| ecutionContext15totalIterationsE) |     (C++                          |
| -   [cudaq::ExecutionResult (C++  |     enum                          |
|     st                            | erator)](using/examples/ptsbe.htm |
| ruct)](api/languages/cpp_api.html | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| #_CPPv4N5cudaq15ExecutionResultE) | tionStrategy4Type12PROPORTIONALE) |
| -   [cud                          | -   [cudaq::ptsbe::Shot           |
| aq::ExecutionResult::appendResult | AllocationStrategy::Type::UNIFORM |
|     (C++                          |     (C++                          |
|     functio                       |                                   |
| n)](api/languages/cpp_api.html#_C |   enumerator)](using/examples/pts |
| PPv4N5cudaq15ExecutionResult12app | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| endResultENSt6stringENSt6size_tE) | AllocationStrategy4Type7UNIFORME) |
| -   [cu                           | -                                 |
| daq::ExecutionResult::deserialize |   [cudaq::ptsbe::TraceInstruction |
|     (C++                          |     (C++                          |
|     function)                     |     struct)](                     |
| ](api/languages/cpp_api.html#_CPP | api/languages/cpp_api.html#_CPPv4 |
| v4N5cudaq15ExecutionResult11deser | N5cudaq5ptsbe16TraceInstructionE) |
| ializeERNSt6vectorINSt6size_tEEE) | -   [cudaq:                       |
| -   [cudaq:                       | :ptsbe::TraceInstruction::channel |
| :ExecutionResult::ExecutionResult |     (C++                          |
|     (C++                          |     member)](api/lang             |
|     functio                       | uages/cpp_api.html#_CPPv4N5cudaq5 |
| n)](api/languages/cpp_api.html#_C | ptsbe16TraceInstruction7channelE) |
| PPv4N5cudaq15ExecutionResult15Exe | -   [cudaq::                      |
| cutionResultE16CountsDictionary), | ptsbe::TraceInstruction::controls |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     member)](api/langu            |
| 15ExecutionResult15ExecutionResul | ages/cpp_api.html#_CPPv4N5cudaq5p |
| tE16CountsDictionaryNSt6stringE), | tsbe16TraceInstruction8controlsE) |
|     [\[2\                         | -   [cud                          |
| ]](api/languages/cpp_api.html#_CP | aq::ptsbe::TraceInstruction::name |
| Pv4N5cudaq15ExecutionResult15Exec |     (C++                          |
| utionResultE16CountsDictionaryd), |     member)](api/l                |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|    [\[3\]](api/languages/cpp_api. | aq5ptsbe16TraceInstruction4nameE) |
| html#_CPPv4N5cudaq15ExecutionResu | -   [cudaq                        |
| lt15ExecutionResultENSt6stringE), | ::ptsbe::TraceInstruction::params |
|     [\[4\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     member)](api/lan              |
| Pv4N5cudaq15ExecutionResult15Exec | guages/cpp_api.html#_CPPv4N5cudaq |
| utionResultERK15ExecutionResult), | 5ptsbe16TraceInstruction6paramsE) |
|     [\[5\                         | -   [cudaq:                       |
| ]](api/languages/cpp_api.html#_CP | :ptsbe::TraceInstruction::targets |
| Pv4N5cudaq15ExecutionResult15Exec |     (C++                          |
| utionResultERR15ExecutionResult), |     member)](api/lang             |
|     [\[6\]](api/language          | uages/cpp_api.html#_CPPv4N5cudaq5 |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | ptsbe16TraceInstruction7targetsE) |
| cutionResult15ExecutionResultEd), | -   [cudaq::ptsbe::T              |
|     [\[7\]](api/languag           | raceInstruction::TraceInstruction |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |     (C++                          |
| ecutionResult15ExecutionResultEv) |                                   |
| -   [                             |   function)](api/languages/cpp_ap |
| cudaq::ExecutionResult::operator= | i.html#_CPPv4N5cudaq5ptsbe16Trace |
|     (C++                          | Instruction16TraceInstructionE20T |
|     function)](api/languages/c    | raceInstructionTypeNSt6stringENSt |
| pp_api.html#_CPPv4N5cudaq15Execut | 6vectorINSt6size_tEEENSt6vectorIN |
| ionResultaSERK15ExecutionResult), | St6size_tEEENSt6vectorIdEENSt8opt |
|     [\[1\]](api/languages/        | ionalIN5cudaq13kraus_channelEEE), |
| cpp_api.html#_CPPv4N5cudaq15Execu |     [\[1\]](api/languages/cpp_a   |
| tionResultaSERR15ExecutionResult) | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| -   [c                            | eInstruction16TraceInstructionEv) |
| udaq::ExecutionResult::operator== | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::type |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4NK5cudaq15Execu |     member)](api/l                |
| tionResulteqERK15ExecutionResult) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cud                          | aq5ptsbe16TraceInstruction4typeE) |
| aq::ExecutionResult::registerName | -   [c                            |
|     (C++                          | udaq::ptsbe::TraceInstructionType |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     enum)](api/                   |
| 15ExecutionResult12registerNameE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq                        | daq5ptsbe20TraceInstructionTypeE) |
| ::ExecutionResult::sequentialData | -   [cudaq::                      |
|     (C++                          | ptsbe::TraceInstructionType::Gate |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     enumerator)](api/langu        |
| ExecutionResult14sequentialDataE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [                             | tsbe20TraceInstructionType4GateE) |
| cudaq::ExecutionResult::serialize | -   [cudaq::ptsbe::               |
|     (C++                          | TraceInstructionType::Measurement |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4NK5cu |                                   |
| daq15ExecutionResult9serializeEv) |    enumerator)](api/languages/cpp |
| -   [cudaq::fermion_handler (C++  | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
|     c                             | aceInstructionType11MeasurementE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::p                     |
| #_CPPv4N5cudaq15fermion_handlerE) | tsbe::TraceInstructionType::Noise |
| -   [cudaq::fermion_op (C++       |     (C++                          |
|     type)](api/languages/cpp_api  |     enumerator)](api/langua       |
| .html#_CPPv4N5cudaq10fermion_opE) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq::fermion_op_term (C++  | sbe20TraceInstructionType5NoiseE) |
|                                   | -   [                             |
| type)](api/languages/cpp_api.html | cudaq::ptsbe::TrajectoryPredicate |
| #_CPPv4N5cudaq15fermion_op_termE) |     (C++                          |
| -   [cudaq::FermioniqQPU (C++     |     type)](api                    |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
|   class)](api/languages/cpp_api.h | udaq5ptsbe19TrajectoryPredicateE) |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | -   [cudaq::QPU (C++              |
| -   [cudaq::get_state (C++        |     class)](api/languages         |
|                                   | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|    function)](api/languages/cpp_a | -   [cudaq::QPU::beginExecution   |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     (C++                          |
| ateEDaRR13QuantumKernelDpRR4Args) |     function                      |
| -   [cudaq::GPUEmulatedQPU (C++   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq3QPU14beginExecutionEv) |
| class)](api/languages/cpp_api.htm | -   [cuda                         |
| l#_CPPv4N5cudaq14GPUEmulatedQPUE) | q::QPU::configureExecutionContext |
| -   [cudaq::gradient (C++         |     (C++                          |
|     class)](api/languages/cpp_    |     funct                         |
| api.html#_CPPv4N5cudaq8gradientE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::gradient::clone (C++  | _CPPv4NK5cudaq3QPU25configureExec |
|     fun                           | utionContextER16ExecutionContext) |
| ction)](api/languages/cpp_api.htm | -   [cudaq::QPU::endExecution     |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     (C++                          |
| -   [cudaq::gradient::compute     |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     function)](api/language       | CPPv4N5cudaq3QPU12endExecutionEv) |
| s/cpp_api.html#_CPPv4N5cudaq8grad | -   [cudaq::QPU::enqueue (C++     |
| ient7computeERKNSt6vectorIdEERKNS |     function)](ap                 |
| t8functionIFdNSt6vectorIdEEEEEd), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[1\]](ap                    | cudaq3QPU7enqueueER11QuantumTask) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cud                          |
| cudaq8gradient7computeERKNSt6vect | aq::QPU::finalizeExecutionContext |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradient::gradient    |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     function)](api/lang           | #_CPPv4NK5cudaq3QPU24finalizeExec |
| uages/cpp_api.html#_CPPv4I00EN5cu | utionContextER16ExecutionContext) |
| daq8gradient8gradientER7KernelT), | -   [cudaq::QPU::getCompileTarget |
|                                   |     (C++                          |
|    [\[1\]](api/languages/cpp_api. |     function)](api/languages/c    |
| html#_CPPv4I00EN5cudaq8gradient8g | pp_api.html#_CPPv4N5cudaq3QPU16ge |
| radientER7KernelTRR10ArgsMapper), | tCompileTargetEPK13RuntimeTarget) |
|     [\[2\                         | -                                 |
| ]](api/languages/cpp_api.html#_CP | [cudaq::QPU::getExecutionThreadId |
| Pv4I00EN5cudaq8gradient8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/               |
|     [\[3                          | languages/cpp_api.html#_CPPv4NK5c |
| \]](api/languages/cpp_api.html#_C | udaq3QPU20getExecutionThreadIdEv) |
| PPv4N5cudaq8gradient8gradientERRN | -   [cudaq::QPU::isEmulated (C++  |
| St8functionIFvNSt6vectorIdEEEEE), |     func                          |
|     [\[                           | tion)](api/languages/cpp_api.html |
| 4\]](api/languages/cpp_api.html#_ | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| CPPv4N5cudaq8gradient8gradientEv) | -   [cudaq::QPU::isSimulator (C++ |
| -   [cudaq::gradient::setArgs     |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     fu                            | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| nction)](api/languages/cpp_api.ht | -   [cudaq::QPU::onRandomSeedSet  |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     (C++                          |
| tArgsEvR13QuantumKernelDpRR4Args) |     function)](api/lang           |
| -   [cudaq::gradient::setKernel   | uages/cpp_api.html#_CPPv4N5cudaq3 |
|     (C++                          | QPU15onRandomSeedSetENSt6size_tE) |
|     function)](api/languages/c    | -   [cudaq::QPU::QPU (C++         |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     functio                       |
| ient9setKernelEvR13QuantumKernel) | n)](api/languages/cpp_api.html#_C |
| -   [cud                          | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| aq::gradients::central_difference |                                   |
|     (C++                          |  [\[1\]](api/languages/cpp_api.ht |
|     class)](api/la                | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[2\]](api/languages/cpp_    |
| q9gradients18central_differenceE) | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| -   [cudaq::gra                   | -   [cudaq::QPU::setId (C++       |
| dients::central_difference::clone |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function)](api/languages      | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QPU::setShots (C++    |
| ents18central_difference5cloneEv) |     f                             |
| -   [cudaq::gradi                 | unction)](api/languages/cpp_api.h |
| ents::central_difference::compute | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
|     (C++                          | -   [c                            |
|     function)](                   | udaq::QPU::shutdownExecutionQueue |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq9gradients18central_differ |     function)](api/l              |
| ence7computeERKNSt6vectorIdEERKNS | anguages/cpp_api.html#_CPPv4N5cud |
| t8functionIFdNSt6vectorIdEEEEEd), | aq3QPU22shutdownExecutionQueueEv) |
|                                   | -   [cudaq::QPU::\~QPU (C++       |
|   [\[1\]](api/languages/cpp_api.h |     function)](api/languages/cp   |
| tml#_CPPv4N5cudaq9gradients18cent | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| ral_difference7computeERKNSt6vect | -   [cudaq::QPUState (C++         |
| orIdEERNSt6vectorIdEERK7spin_opd) |     class)](api/languages/cpp_    |
| -   [cudaq::gradie                | api.html#_CPPv4N5cudaq8QPUStateE) |
| nts::central_difference::gradient | -   [cudaq::qreg (C++             |
|     (C++                          |     class)](api/lan               |
|     functio                       | guages/cpp_api.html#_CPPv4I_NSt6s |
| n)](api/languages/cpp_api.html#_C | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| PPv4I00EN5cudaq9gradients18centra | -   [cudaq::qreg::back (C++       |
| l_difference8gradientER7KernelT), |     function)                     |
|     [\[1\]](api/langua            | ](api/languages/cpp_api.html#_CPP |
| ges/cpp_api.html#_CPPv4I00EN5cuda | v4N5cudaq4qreg4backENSt6size_tE), |
| q9gradients18central_difference8g |     [\[1\]](api/languages/cpp_ap  |
| radientER7KernelTRR10ArgsMapper), | i.html#_CPPv4N5cudaq4qreg4backEv) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::qreg::begin (C++      |
| api.html#_CPPv4I00EN5cudaq9gradie |                                   |
| nts18central_difference8gradientE |  function)](api/languages/cpp_api |
| RR13QuantumKernelRR10ArgsMapper), | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::qreg::clear (C++      |
| _api.html#_CPPv4N5cudaq9gradients |                                   |
| 18central_difference8gradientERRN |  function)](api/languages/cpp_api |
| St8functionIFvNSt6vectorIdEEEEE), | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qreg::front (C++      |
| p_api.html#_CPPv4N5cudaq9gradient |     function)]                    |
| s18central_difference8gradientEv) | (api/languages/cpp_api.html#_CPPv |
| -   [cud                          | 4N5cudaq4qreg5frontENSt6size_tE), |
| aq::gradients::forward_difference |     [\[1\]](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5frontEv) |
|     class)](api/la                | -   [cudaq::qreg::operator\[\]    |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9gradients18forward_differenceE) |     functi                        |
| -   [cudaq::gra                   | on)](api/languages/cpp_api.html#_ |
| dients::forward_difference::clone | CPPv4N5cudaq4qregixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qreg::qreg (C++       |
|     function)](api/languages      |     function)                     |
| /cpp_api.html#_CPPv4N5cudaq9gradi | ](api/languages/cpp_api.html#_CPP |
| ents18forward_difference5cloneEv) | v4N5cudaq4qreg4qregENSt6size_tE), |
| -   [cudaq::gradi                 |     [\[1\]](api/languages/cpp_ap  |
| ents::forward_difference::compute | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     (C++                          | -   [cudaq::qreg::size (C++       |
|     function)](                   |                                   |
| api/languages/cpp_api.html#_CPPv4 |  function)](api/languages/cpp_api |
| N5cudaq9gradients18forward_differ | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| ence7computeERKNSt6vectorIdEERKNS | -   [cudaq::qreg::slice (C++      |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api/langu          |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq4q |
|   [\[1\]](api/languages/cpp_api.h | reg5sliceENSt6size_tENSt6size_tE) |
| tml#_CPPv4N5cudaq9gradients18forw | -   [cudaq::qreg::value_type (C++ |
| ard_difference7computeERKNSt6vect |                                   |
| orIdEERNSt6vectorIdEERK7spin_opd) | type)](api/languages/cpp_api.html |
| -   [cudaq::gradie                | #_CPPv4N5cudaq4qreg10value_typeE) |
| nts::forward_difference::gradient | -   [cudaq::qspan (C++            |
|     (C++                          |     class)](api/lang              |
|     functio                       | uages/cpp_api.html#_CPPv4I_NSt6si |
| n)](api/languages/cpp_api.html#_C | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| PPv4I00EN5cudaq9gradients18forwar | -   [cudaq::QuakeValue (C++       |
| d_difference8gradientER7KernelT), |     class)](api/languages/cpp_api |
|     [\[1\]](api/langua            | .html#_CPPv4N5cudaq10QuakeValueE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::Q                     |
| q9gradients18forward_difference8g | uakeValue::canValidateNumElements |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\]](api/languages/cpp_    |     function)](api/languages      |
| api.html#_CPPv4I00EN5cudaq9gradie | /cpp_api.html#_CPPv4N5cudaq10Quak |
| nts18forward_difference8gradientE | eValue22canValidateNumElementsEv) |
| RR13QuantumKernelRR10ArgsMapper), | -                                 |
|     [\[3\]](api/languages/cpp     |  [cudaq::QuakeValue::constantSize |
| _api.html#_CPPv4N5cudaq9gradients |     (C++                          |
| 18forward_difference8gradientERRN |     function)](api                |
| St8functionIFvNSt6vectorIdEEEEE), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[4\]](api/languages/cp      | udaq10QuakeValue12constantSizeEv) |
| p_api.html#_CPPv4N5cudaq9gradient | -   [cudaq::QuakeValue::dump (C++ |
| s18forward_difference8gradientEv) |     function)](api/lan            |
| -   [                             | guages/cpp_api.html#_CPPv4N5cudaq |
| cudaq::gradients::parameter_shift | 10QuakeValue4dumpERNSt7ostreamE), |
|     (C++                          |     [\                            |
|     class)](api                   | [1\]](api/languages/cpp_api.html# |
| /languages/cpp_api.html#_CPPv4N5c | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| udaq9gradients15parameter_shiftE) | -   [cudaq                        |
| -   [cudaq::                      | ::QuakeValue::getRequiredElements |
| gradients::parameter_shift::clone |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/langua         | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | uakeValue19getRequiredElementsEv) |
| adients15parameter_shift5cloneEv) | -   [cudaq::QuakeValue::getValue  |
| -   [cudaq::gr                    |     (C++                          |
| adients::parameter_shift::compute |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function                      | 4NK5cudaq10QuakeValue8getValueEv) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::QuakeValue::inverse   |
| Pv4N5cudaq9gradients15parameter_s |     (C++                          |
| hift7computeERKNSt6vectorIdEERKNS |     function)                     |
| t8functionIFdNSt6vectorIdEEEEEd), | ](api/languages/cpp_api.html#_CPP |
|     [\[1\]](api/languages/cpp_ap  | v4NK5cudaq10QuakeValue7inverseEv) |
| i.html#_CPPv4N5cudaq9gradients15p | -                                 |
| arameter_shift7computeERKNSt6vect |    [cudaq::QuakeValue::isSequence |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gra                   |     function)](a                  |
| dients::parameter_shift::gradient | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq10QuakeValue10isSequenceEv) |
|     func                          | -                                 |
| tion)](api/languages/cpp_api.html |    [cudaq::QuakeValue::operator\* |
| #_CPPv4I00EN5cudaq9gradients15par |     (C++                          |
| ameter_shift8gradientER7KernelT), |     function)](api                |
|     [\[1\]](api/lan               | /languages/cpp_api.html#_CPPv4N5c |
| guages/cpp_api.html#_CPPv4I00EN5c | udaq10QuakeValuemlE10QuakeValue), |
| udaq9gradients15parameter_shift8g |                                   |
| radientER7KernelTRR10ArgsMapper), | [\[1\]](api/languages/cpp_api.htm |
|     [\[2\]](api/languages/c       | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| pp_api.html#_CPPv4I00EN5cudaq9gra | -   [cudaq::QuakeValue::operator+ |
| dients15parameter_shift8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api                |
|     [\[3\]](api/languages/        | /languages/cpp_api.html#_CPPv4N5c |
| cpp_api.html#_CPPv4N5cudaq9gradie | udaq10QuakeValueplE10QuakeValue), |
| nts15parameter_shift8gradientERRN |     [                             |
| St8functionIFvNSt6vectorIdEEEEE), | \[1\]](api/languages/cpp_api.html |
|     [\[4\]](api/languages         | #_CPPv4N5cudaq10QuakeValueplEKd), |
| /cpp_api.html#_CPPv4N5cudaq9gradi |                                   |
| ents15parameter_shift8gradientEv) | [\[2\]](api/languages/cpp_api.htm |
| -   [cudaq::kernel_builder (C++   | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|     clas                          | -   [cudaq::QuakeValue::operator- |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4IDpEN5cudaq14kernel_builderE) |     function)](api                |
| -   [c                            | /languages/cpp_api.html#_CPPv4N5c |
| udaq::kernel_builder::constantVal | udaq10QuakeValuemiE10QuakeValue), |
|     (C++                          |     [                             |
|     function)](api/la             | \[1\]](api/languages/cpp_api.html |
| nguages/cpp_api.html#_CPPv4N5cuda | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| q14kernel_builder11constantValEd) |     [                             |
| -                                 | \[2\]](api/languages/cpp_api.html |
|  [cudaq::kernel_builder::detector | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     (C++                          |                                   |
|                                   | [\[3\]](api/languages/cpp_api.htm |
|    function)](api/languages/cpp_a | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| pi.html#_CPPv4IDpEN5cudaq14kernel | -   [cudaq::QuakeValue::operator/ |
| _builder8detectorEvDpRR8MeasArgs) |     (C++                          |
| -                                 |     function)](api                |
| [cudaq::kernel_builder::detectors | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuedvE10QuakeValue), |
|     func                          |                                   |
| tion)](api/languages/cpp_api.html | [\[1\]](api/languages/cpp_api.htm |
| #_CPPv4N5cudaq14kernel_builder9de | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| tectorsE10QuakeValue10QuakeValue) | -                                 |
| -   [cu                           |  [cudaq::QuakeValue::operator\[\] |
| daq::kernel_builder::getArguments |     (C++                          |
|     (C++                          |     function)](api                |
|     function)](api/lan            | /languages/cpp_api.html#_CPPv4N5c |
| guages/cpp_api.html#_CPPv4N5cudaq | udaq10QuakeValueixEKNSt6size_tE), |
| 14kernel_builder12getArgumentsEv) |     [\[1\]](api/                  |
| -   [cu                           | languages/cpp_api.html#_CPPv4N5cu |
| daq::kernel_builder::getNumParams | daq10QuakeValueixERK10QuakeValue) |
|     (C++                          | -                                 |
|     function)](api/lan            |    [cudaq::QuakeValue::QuakeValue |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 14kernel_builder12getNumParamsEv) |     function)](api/languag        |
| -   [cud                          | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| aq::kernel_builder::isArgSequence | akeValue10QuakeValueERN4mlir20Imp |
|     (C++                          | licitLocOpBuilderEN4mlir5ValueE), |
|     function)](api/languages/cpp_ |     [\[1\]                        |
| api.html#_CPPv4N5cudaq14kernel_bu | ](api/languages/cpp_api.html#_CPP |
| ilder13isArgSequenceENSt6size_tE) | v4N5cudaq10QuakeValue10QuakeValue |
| -   [cuda                         | ERN4mlir20ImplicitLocOpBuilderEd) |
| q::kernel_builder::kernel_builder | -   [cudaq::QuakeValue::size (C++ |
|     (C++                          |     funct                         |
|     function)](api/languages/cpp  | ion)](api/languages/cpp_api.html# |
| _api.html#_CPPv4N5cudaq14kernel_b | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| uilder14kernel_builderERNSt6vecto | -   [cudaq::QuakeValue::slice     |
| rIN6detail17KernelBuilderTypeEEE) |     (C++                          |
| -   [cudaq::k                     |     function)](api/languages/cpp_ |
| ernel_builder::logical_observable | api.html#_CPPv4N5cudaq10QuakeValu |
|     (C++                          | e5sliceEKNSt6size_tEKNSt6size_tE) |
|     function)                     | -   [cudaq::quantum_platform (C++ |
| ](api/languages/cpp_api.html#_CPP |     cl                            |
| v4IDpEN5cudaq14kernel_builder18lo | ass)](api/languages/cpp_api.html# |
| gical_observableEvDpRR8MeasArgs), | _CPPv4N5cudaq16quantum_platformE) |
|     [\[1\]](ap                    | -   [cudaq:                       |
| i/languages/cpp_api.html#_CPPv4N5 | :quantum_platform::beginExecution |
| cudaq14kernel_builder18logical_ob |     (C++                          |
| servableE10QuakeValueNSt6size_tE) |     function)](api/languag        |
| -   [cudaq::kernel_builder::name  | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14beginExecutionEv) |
|     function)                     | -   [cudaq::quantum_pl            |
| ](api/languages/cpp_api.html#_CPP | atform::configureExecutionContext |
| v4N5cudaq14kernel_builder4nameEv) |     (C++                          |
| -                                 |     function)](api/lang           |
|    [cudaq::kernel_builder::qalloc | uages/cpp_api.html#_CPPv4NK5cudaq |
|     (C++                          | 16quantum_platform25configureExec |
|     function)](api/language       | utionContextER16ExecutionContext) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cuda                         |
| nel_builder6qallocE10QuakeValue), | q::quantum_platform::endExecution |
|     [\[1\]](api/language          |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     function)](api/langu          |
| nel_builder6qallocEKNSt6size_tE), | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     [\[2                          | quantum_platform12endExecutionEv) |
| \]](api/languages/cpp_api.html#_C | -   [cudaq::q                     |
| PPv4N5cudaq14kernel_builder6qallo | uantum_platform::enqueueAsyncTask |
| cERNSt6vectorINSt7complexIdEEEE), |     (C++                          |
|     [\[3\]](                      |     function)](api/languages/     |
| api/languages/cpp_api.html#_CPPv4 | cpp_api.html#_CPPv4N5cudaq16quant |
| N5cudaq14kernel_builder6qallocEv) | um_platform16enqueueAsyncTaskEKNS |
| -   [cudaq::kernel_builder::swap  | t6size_tER19KernelExecutionTask), |
|     (C++                          |     [\[1\]](api/languag           |
|     function)](api/language       | es/cpp_api.html#_CPPv4N5cudaq16qu |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | antum_platform16enqueueAsyncTaskE |
| 4kernel_builder4swapEvRK10QuakeVa | KNSt6size_tERNSt8functionIFvvEEE) |
| lueRK10QuakeValueRK10QuakeValue), | -   [cudaq::quantum_p             |
|                                   | latform::finalizeExecutionContext |
| [\[1\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4I00EN5cudaq14kernel_build |     function)](api/languages/c    |
| er4swapEvRKNSt6vectorI10QuakeValu | pp_api.html#_CPPv4NK5cudaq16quant |
| eEERK10QuakeValueRK10QuakeValue), | um_platform24finalizeExecutionCon |
|                                   | textERN5cudaq16ExecutionContextE) |
| [\[2\]](api/languages/cpp_api.htm | -   [cudaq::qua                   |
| l#_CPPv4N5cudaq14kernel_builder4s | ntum_platform::get_codegen_config |
| wapERK10QuakeValueRK10QuakeValue) |     (C++                          |
| -   [cudaq::KernelExecutionTask   |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq16quantu |
|     type                          | m_platform18get_codegen_configEv) |
| )](api/languages/cpp_api.html#_CP | -   [cuda                         |
| Pv4N5cudaq19KernelExecutionTaskE) | q::quantum_platform::get_exec_ctx |
| -   [cudaq::KernelThunkResultType |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     struct)]                      | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| (api/languages/cpp_api.html#_CPPv | quantum_platform12get_exec_ctxEv) |
| 4N5cudaq21KernelThunkResultTypeE) | -   [c                            |
| -   [cudaq::KernelThunkType (C++  | udaq::quantum_platform::get_noise |
|                                   |     (C++                          |
| type)](api/languages/cpp_api.html |     function)](api/languages/c    |
| #_CPPv4N5cudaq15KernelThunkTypeE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [cudaq::kraus_channel (C++    | m_platform9get_noiseENSt6size_tE) |
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
| pi.html#_CPPv4NK5cudaq13kraus_cha | -   [cudaq:                       |
| nnel14is_identity_opENSt6size_tE) | :quantum_platform::list_platforms |
| -   [cudaq::                      |     (C++                          |
| kraus_channel::is_unitary_mixture |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     function)](api/languages      | antum_platform14list_platformsEv) |
| /cpp_api.html#_CPPv4NK5cudaq13kra | -                                 |
| us_channel18is_unitary_mixtureEv) |    [cudaq::quantum_platform::name |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::kraus_channel |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function)](api/lang           | K5cudaq16quantum_platform4nameEv) |
| uages/cpp_api.html#_CPPv4IDpEN5cu | -   [                             |
| daq13kraus_channel13kraus_channel | cudaq::quantum_platform::num_qpus |
| EDpRRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)](api/l              |
|  [\[1\]](api/languages/cpp_api.ht | anguages/cpp_api.html#_CPPv4NK5cu |
| ml#_CPPv4N5cudaq13kraus_channel13 | daq16quantum_platform8num_qpusEv) |
| kraus_channelERK13kraus_channel), | -   [cudaq::                      |
|     [\[2\]                        | quantum_platform::onRandomSeedSet |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq13kraus_channel13kraus_c |                                   |
| hannelERKNSt6vectorI8kraus_opEE), | function)](api/languages/cpp_api. |
|     [\[3\]                        | html#_CPPv4N5cudaq16quantum_platf |
| ](api/languages/cpp_api.html#_CPP | orm15onRandomSeedSetENSt6size_tE) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::q                     |
| hannelERRNSt6vectorI8kraus_opEE), | uantum_platform::prepareExecution |
|     [\[4\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     fu                            |
| 13kraus_channel13kraus_channelEv) | nction)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4NK5cudaq16quantum_platfo |
| [cudaq::kraus_channel::noise_type | rm16prepareExecutionENSt6size_tE) |
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
|     functio                       | s)](api/languages/cpp_api.html#_C |
| n)](api/languages/cpp_api.html#_C | PPv4N5cudaq18QuEraRemoteRESTQPUE) |
| PPv4I0EN5cudaq8kraus_op8kraus_opE | -   [cudaq::qvector (C++          |
| RKNSt6vectorINSt7complexI1TEEEE), |     class)                        |
|     [                             | ](api/languages/cpp_api.html#_CPP |
| \[1\]](api/languages/cpp_api.html | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | -   [cudaq::qvector::back (C++    |
| opERRNSt16initializer_listI1TEE), |     function)](a                  |
|     [\[2\]](api/languages/cpp_a   | pi/languages/cpp_api.html#_CPPv4N |
| pi.html#_CPPv4N5cudaq8kraus_op8kr | 5cudaq7qvector4backENSt6size_tE), |
| aus_opENSt6vectorI10value_typeEE) |                                   |
| -   [cudaq::kraus_op::nCols (C++  |   [\[1\]](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq7qvector4backEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::qvector::begin (C++   |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     fu                            |
| -   [cudaq::kraus_op::nRows (C++  | nction)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4N5cudaq7qvector5beginEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::qvector::clear (C++   |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) |     fu                            |
| -   [cudaq::kraus_op::value_type  | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     type                          | -   [cudaq::qvector::end (C++     |
| )](api/languages/cpp_api.html#_CP |                                   |
| Pv4N5cudaq8kraus_op10value_typeE) | function)](api/languages/cpp_api. |
| -   [cudaq::KrausSelection (C++   | html#_CPPv4N5cudaq7qvector3endEv) |
|     s                             | -   [cudaq::qvector::front (C++   |
| truct)](api/languages/cpp_api.htm |     function)](ap                 |
| l#_CPPv4N5cudaq14KrausSelectionE) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [cudaq:                       | cudaq7qvector5frontENSt6size_tE), |
| :KrausSelection::circuit_location |                                   |
|     (C++                          |  [\[1\]](api/languages/cpp_api.ht |
|     member)](api/langua           | ml#_CPPv4N5cudaq7qvector5frontEv) |
| ges/cpp_api.html#_CPPv4N5cudaq14K | -   [cudaq::qvector::operator=    |
| rausSelection16circuit_locationE) |     (C++                          |
| -                                 |     functio                       |
|  [cudaq::KrausSelection::is_error | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq7qvectoraSERK7qvector) |
|     member)](a                    | -   [cudaq::qvector::operator\[\] |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14KrausSelection8is_errorE) |     function)                     |
| -   [cudaq::Kra                   | ](api/languages/cpp_api.html#_CPP |
| usSelection::kraus_operator_index | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qvector::qvector (C++ |
|     member)](api/languages/       |     function)](api/               |
| cpp_api.html#_CPPv4N5cudaq14Kraus | languages/cpp_api.html#_CPPv4N5cu |
| Selection20kraus_operator_indexE) | daq7qvector7qvectorENSt6size_tE), |
| -   [cuda                         |     [\[1\]](a                     |
| q::KrausSelection::KrausSelection | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq7qvector7qvectorERK5state), |
|     function)](a                  |     [\[2\]](api                   |
| pi/languages/cpp_api.html#_CPPv4N | /languages/cpp_api.html#_CPPv4N5c |
| 5cudaq14KrausSelection14KrausSele | udaq7qvector7qvectorERK7qvector), |
| ctionENSt6size_tENSt6vectorINSt6s |     [\[3\]](ap                    |
| ize_tEEENSt6stringENSt6size_tEb), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[1\]](api/langu             | cudaq7qvector7qvectorERR7qvector) |
| ages/cpp_api.html#_CPPv4N5cudaq14 | -   [cudaq::qvector::size (C++    |
| KrausSelection14KrausSelectionEv) |     fu                            |
| -                                 | nction)](api/languages/cpp_api.ht |
|   [cudaq::KrausSelection::op_name | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     (C++                          | -   [cudaq::qvector::slice (C++   |
|     member)](                     |     function)](api/language       |
| api/languages/cpp_api.html#_CPPv4 | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| N5cudaq14KrausSelection7op_nameE) | tor5sliceENSt6size_tENSt6size_tE) |
| -   [                             | -   [cudaq::qvector::value_type   |
| cudaq::KrausSelection::operator== |     (C++                          |
|     (C++                          |     typ                           |
|     function)](api/languages      | e)](api/languages/cpp_api.html#_C |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | PPv4N5cudaq7qvector10value_typeE) |
| usSelectioneqERK14KrausSelection) | -   [cudaq::qview (C++            |
| -                                 |     clas                          |
|    [cudaq::KrausSelection::qubits | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     member)]                      | -   [cudaq::qview::back (C++      |
| (api/languages/cpp_api.html#_CPPv |     function)                     |
| 4N5cudaq14KrausSelection6qubitsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::KrausTrajectory (C++  | v4N5cudaq5qview4backENSt6size_tE) |
|     st                            | -   [cudaq::qview::begin (C++     |
| ruct)](api/languages/cpp_api.html |                                   |
| #_CPPv4N5cudaq15KrausTrajectoryE) | function)](api/languages/cpp_api. |
| -                                 | html#_CPPv4N5cudaq5qview5beginEv) |
|  [cudaq::KrausTrajectory::builder | -   [cudaq::qview::end (C++       |
|     (C++                          |                                   |
|     function)](ap                 |   function)](api/languages/cpp_ap |
| i/languages/cpp_api.html#_CPPv4N5 | i.html#_CPPv4N5cudaq5qview3endEv) |
| cudaq15KrausTrajectory7builderEv) | -   [cudaq::qview::front (C++     |
| -   [cu                           |     function)](                   |
| daq::KrausTrajectory::countErrors | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5qview5frontENSt6size_tE), |
|     function)](api/lang           |                                   |
| uages/cpp_api.html#_CPPv4NK5cudaq |    [\[1\]](api/languages/cpp_api. |
| 15KrausTrajectory11countErrorsEv) | html#_CPPv4N5cudaq5qview5frontEv) |
| -   [                             | -   [cudaq::qview::operator\[\]   |
| cudaq::KrausTrajectory::isOrdered |     (C++                          |
|     (C++                          |     functio                       |
|     function)](api/l              | n)](api/languages/cpp_api.html#_C |
| anguages/cpp_api.html#_CPPv4NK5cu | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| daq15KrausTrajectory9isOrderedEv) | -   [cudaq::qview::qview (C++     |
| -   [cudaq::                      |     functio                       |
| KrausTrajectory::kraus_selections | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     member)](api/languag          |     [\[1                          |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | \]](api/languages/cpp_api.html#_C |
| ausTrajectory16kraus_selectionsE) | PPv4N5cudaq5qview5qviewERK5qview) |
| -   [cudaq:                       | -   [cudaq::qview::size (C++      |
| :KrausTrajectory::KrausTrajectory |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function                      | html#_CPPv4NK5cudaq5qview4sizeEv) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qview::slice (C++     |
| Pv4N5cudaq15KrausTrajectory15Krau |     function)](api/langua         |
| sTrajectoryENSt6size_tENSt6vector | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| I14KrausSelectionEEdNSt6size_tE), | iew5sliceENSt6size_tENSt6size_tE) |
|     [\[1\]](api/languag           | -   [cudaq::qview::value_type     |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |     (C++                          |
| ausTrajectory15KrausTrajectoryEv) |     t                             |
| -   [cudaq::Kr                    | ype)](api/languages/cpp_api.html# |
| ausTrajectory::measurement_counts | _CPPv4N5cudaq5qview10value_typeE) |
|     (C++                          | -   [cudaq::range (C++            |
|     member)](api/languages        |     fun                           |
| /cpp_api.html#_CPPv4N5cudaq15Krau | ction)](api/languages/cpp_api.htm |
| sTrajectory18measurement_countsE) | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| -   [cud                          | orI11ElementTypeEE11ElementType), |
| aq::KrausTrajectory::multiplicity |     [\[1\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4I0EN5cudaq5rangeEN |
|     member)](api/lan              | St6vectorI11ElementTypeEE11Elemen |
| guages/cpp_api.html#_CPPv4N5cudaq | tType11ElementType11ElementType), |
| 15KrausTrajectory12multiplicityE) |     [                             |
| -   [                             | \[2\]](api/languages/cpp_api.html |
| cudaq::KrausTrajectory::num_shots | #_CPPv4N5cudaq5rangeENSt6size_tE) |
|     (C++                          | -   [cudaq::real (C++             |
|     member)](api                  |     type)](api/languages/         |
| /languages/cpp_api.html#_CPPv4N5c | cpp_api.html#_CPPv4N5cudaq4realE) |
| udaq15KrausTrajectory9num_shotsE) | -   [cudaq::registry (C++         |
| -   [c                            |     type)](api/languages/cpp_     |
| udaq::KrausTrajectory::operator== | api.html#_CPPv4N5cudaq8registryE) |
|     (C++                          | -                                 |
|     function)](api/languages/c    |  [cudaq::registry::RegisteredType |
| pp_api.html#_CPPv4NK5cudaq15Kraus |     (C++                          |
| TrajectoryeqERK15KrausTrajectory) |     class)](api/                  |
| -   [cu                           | languages/cpp_api.html#_CPPv4I0EN |
| daq::KrausTrajectory::probability | 5cudaq8registry14RegisteredTypeE) |
|     (C++                          | -   [cudaq::RemoteRESTQPU (C++    |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda |  class)](api/languages/cpp_api.ht |
| q15KrausTrajectory11probabilityE) | ml#_CPPv4N5cudaq13RemoteRESTQPUE) |
| -   [cuda                         | -   [cudaq::Resources (C++        |
| q::KrausTrajectory::trajectory_id |     class)](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     member)](api/lang             | -   [cudaq::run (C++              |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     function)]                    |
| 5KrausTrajectory13trajectory_idE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
|   [cudaq::KrausTrajectory::weight | 5invoke_result_tINSt7decay_tI13Qu |
|     (C++                          | antumKernelEEDpNSt7decay_tI4ARGSE |
|     member)](                     | EEEEENSt6size_tERN5cudaq11noise_m |
| api/languages/cpp_api.html#_CPPv4 | odelERR13QuantumKernelDpRR4ARGS), |
| N5cudaq15KrausTrajectory6weightE) |     [\[1\]](api/langu             |
| -                                 | ages/cpp_api.html#_CPPv4I0DpEN5cu |
|    [cudaq::KrausTrajectoryBuilder | daq3runENSt6vectorINSt15invoke_re |
|     (C++                          | sult_tINSt7decay_tI13QuantumKerne |
|     class)](                      | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| api/languages/cpp_api.html#_CPPv4 | ize_tERR13QuantumKernelDpRR4ARGS) |
| N5cudaq22KrausTrajectoryBuilderE) | -   [cudaq::run_async (C++        |
| -   [cud                          |     functio                       |
| aq::KrausTrajectoryBuilder::build | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     function)](api/lang           | tureINSt6vectorINSt15invoke_resul |
| uages/cpp_api.html#_CPPv4NK5cudaq | t_tINSt7decay_tI13QuantumKernelEE |
| 22KrausTrajectoryBuilder5buildEv) | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| -   [cud                          | ze_tENSt6size_tERN5cudaq11noise_m |
| aq::KrausTrajectoryBuilder::setId | odelERR13QuantumKernelDpRR4ARGS), |
|     (C++                          |     [\[1\]](api/la                |
|     function)](api/languages/cpp  | nguages/cpp_api.html#_CPPv4I0DpEN |
| _api.html#_CPPv4N5cudaq22KrausTra | 5cudaq9run_asyncENSt6futureINSt6v |
| jectoryBuilder5setIdENSt6size_tE) | ectorINSt15invoke_result_tINSt7de |
| -   [cudaq::Kraus                 | cay_tI13QuantumKernelEEDpNSt7deca |
| TrajectoryBuilder::setProbability | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4ARGS) |
|     function)](api/languages/cpp  | -   [cudaq::RuntimeTarget (C++    |
| _api.html#_CPPv4N5cudaq22KrausTra |                                   |
| jectoryBuilder14setProbabilityEd) | struct)](api/languages/cpp_api.ht |
| -   [cudaq::Krau                  | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| sTrajectoryBuilder::setSelections | -   [cudaq::sample (C++           |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/languag        | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | mpleE13sample_resultRK14sample_op |
| ausTrajectoryBuilder13setSelectio | tionsRR13QuantumKernelDpRR4Args), |
| nsENSt6vectorI14KrausSelectionEE) |     [\[1\                         |
| -   [cudaq::logical_observable    | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     function)](api/languages/c    | esultRR13QuantumKernelDpRR4Args), |
| pp_api.html#_CPPv4IDpEN5cudaq18lo |     [\                            |
| gical_observableEvDpRR8MeasArgs), | [2\]](api/languages/cpp_api.html# |
|     [\[1\]](api/l                 | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| anguages/cpp_api.html#_CPPv4N5cud | ize_tERR13QuantumKernelDpRR4Args) |
| aq18logical_observableERKNSt6vect | -   [cudaq::sample_options (C++   |
| orI14measure_resultEENSt6size_tE) |     s                             |
| -   [cudaq::M2DSparseMatrix (C++  | truct)](api/languages/cpp_api.htm |
|     st                            | l#_CPPv4N5cudaq14sample_optionsE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::sample_result (C++    |
| #_CPPv4N5cudaq15M2DSparseMatrixE) |                                   |
| -   [cudaq::M2OSparseMatrix (C++  |  class)](api/languages/cpp_api.ht |
|     st                            | ml#_CPPv4N5cudaq13sample_resultE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::sample_result::append |
| #_CPPv4N5cudaq15M2OSparseMatrixE) |     (C++                          |
| -   [cudaq::matrix_callback (C++  |                                   |
|     c                             |    function)](api/languages/cpp_a |
| lass)](api/languages/cpp_api.html | pi.html#_CPPv4N5cudaq13sample_res |
| #_CPPv4N5cudaq15matrix_callbackE) | ult6appendERK15ExecutionResultb), |
| -   [cudaq::matrix_handler (C++   |     [\[1\]](api/languages/cpp_    |
|                                   | api.html#_CPPv4N5cudaq13sample_re |
| class)](api/languages/cpp_api.htm | sult6appendERR15ExecutionResultb) |
| l#_CPPv4N5cudaq14matrix_handlerE) | -   [cudaq::sample_result::begin  |
| -   [cudaq::mat                   |     (C++                          |
| rix_handler::commutation_behavior |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     struct)](api/languages/       | 4N5cudaq13sample_result5beginEv), |
| cpp_api.html#_CPPv4N5cudaq14matri |     [\[1\]]                       |
| x_handler20commutation_behaviorE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NK5cudaq13sample_result5beginEv) |
|    [cudaq::matrix_handler::define | -   [cudaq::sample_result::cbegin |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](                   |
| pi/languages/cpp_api.html#_CPPv4N | api/languages/cpp_api.html#_CPPv4 |
| 5cudaq14matrix_handler6defineENSt | NK5cudaq13sample_result6cbeginEv) |
| 6stringENSt6vectorINSt7int64_tEEE | -   [cudaq::sample_result::cend   |
| RR15matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
| [\[1\]](api/languages/cpp_api.htm | v4NK5cudaq13sample_result4cendEv) |
| l#_CPPv4N5cudaq14matrix_handler6d | -   [cudaq::sample_result::clear  |
| efineENSt6stringENSt6vectorINSt7i |     (C++                          |
| nt64_tEEERR15matrix_callbackRR20d |     function)                     |
| iag_matrix_callbackRKNSt13unorder | ](api/languages/cpp_api.html#_CPP |
| ed_mapINSt6stringENSt6stringEEE), | v4N5cudaq13sample_result5clearEv) |
|     [\[2\]](                      | -   [cudaq::sample_result::count  |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14matrix_handler6defineENS |     function)](                   |
| t6stringENSt6vectorINSt7int64_tEE | api/languages/cpp_api.html#_CPPv4 |
| ERR15matrix_callbackRRNSt13unorde | NK5cudaq13sample_result5countENSt |
| red_mapINSt6stringENSt6stringEEE) | 11string_viewEKNSt11string_viewE) |
| -                                 | -   [                             |
|   [cudaq::matrix_handler::degrees | cudaq::sample_result::deserialize |
|     (C++                          |     (C++                          |
|     function)](ap                 |     functio                       |
| i/languages/cpp_api.html#_CPPv4NK | n)](api/languages/cpp_api.html#_C |
| 5cudaq14matrix_handler7degreesEv) | PPv4N5cudaq13sample_result11deser |
| -                                 | ializeERNSt6vectorINSt6size_tEEE) |
|  [cudaq::matrix_handler::displace | -   [cudaq::sample_result::dump   |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api/languag        |
| s/cpp_api.html#_CPPv4N5cudaq14mat | es/cpp_api.html#_CPPv4NK5cudaq13s |
| rix_handler8displaceENSt6size_tE) | ample_result4dumpERNSt7ostreamE), |
| -   [cudaq::matrix                |     [\[1\]                        |
| _handler::get_expected_dimensions | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq13sample_result4dumpEv) |
|                                   | -   [cudaq::sample_result::end    |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     function                      |
| ndler23get_expected_dimensionsEv) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::matrix_ha             | Pv4N5cudaq13sample_result3endEv), |
| ndler::get_parameter_descriptions |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq13sample_result3endEv) |
| function)](api/languages/cpp_api. | -   [                             |
| html#_CPPv4NK5cudaq14matrix_handl | cudaq::sample_result::expectation |
| er26get_parameter_descriptionsEv) |     (C++                          |
| -   [c                            |     f                             |
| udaq::matrix_handler::instantiate | unction)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4NK5cudaq13sample_result |
|     function)](a                  | 11expectationEKNSt11string_viewE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cuda                         |
| 5cudaq14matrix_handler11instantia | q::sample_result::get_annotations |
| teENSt6stringERKNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     function)](api/langua         |
|     [\[1\]](                      | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| api/languages/cpp_api.html#_CPPv4 | sample_result15get_annotationsEv) |
| N5cudaq14matrix_handler11instanti | -   [c                            |
| ateENSt6stringERRNSt6vectorINSt6s | udaq::sample_result::get_marginal |
| ize_tEEERK20commutation_behavior) |     (C++                          |
| -   [cuda                         |     function)](api/languages/cpp_ |
| q::matrix_handler::matrix_handler | api.html#_CPPv4NK5cudaq13sample_r |
|     (C++                          | esult12get_marginalERKNSt6vectorI |
|     function)](api/languag        | NSt6size_tEEEKNSt11string_viewE), |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     [\[1\]](api/languages/cpp_    |
| ble_if_tINSt12is_base_of_vI16oper | api.html#_CPPv4NK5cudaq13sample_r |
| ator_handler1TEEbEEEN5cudaq14matr | esult12get_marginalERRKNSt6vector |
| ix_handler14matrix_handlerERK1T), | INSt6size_tEEEKNSt11string_viewE) |
|     [\[1\]](ap                    | -   [cuda                         |
| i/languages/cpp_api.html#_CPPv4I0 | q::sample_result::get_total_shots |
| _NSt11enable_if_tINSt12is_base_of |     (C++                          |
| _vI16operator_handler1TEEbEEEN5cu |     function)](api/langua         |
| daq14matrix_handler14matrix_handl | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| erERK1TRK20commutation_behavior), | sample_result15get_total_shotsEv) |
|     [\[2\]](api/languages/cpp_ap  | -   [cuda                         |
| i.html#_CPPv4N5cudaq14matrix_hand | q::sample_result::has_even_parity |
| ler14matrix_handlerENSt6size_tE), |     (C++                          |
|     [\[3\]](api/                  |     fun                           |
| languages/cpp_api.html#_CPPv4N5cu | ction)](api/languages/cpp_api.htm |
| daq14matrix_handler14matrix_handl | l#_CPPv4N5cudaq13sample_result15h |
| erENSt6stringERKNSt6vectorINSt6si | as_even_parityENSt11string_viewE) |
| ze_tEEERK20commutation_behavior), | -   [cuda                         |
|     [\[4\]](api/                  | q::sample_result::has_expectation |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     funct                         |
| erENSt6stringERRNSt6vectorINSt6si | ion)](api/languages/cpp_api.html# |
| ze_tEEERK20commutation_behavior), | _CPPv4NK5cudaq13sample_result15ha |
|     [\                            | s_expectationEKNSt11string_viewE) |
| [5\]](api/languages/cpp_api.html# | -   [cu                           |
| _CPPv4N5cudaq14matrix_handler14ma | daq::sample_result::most_probable |
| trix_handlerERK14matrix_handler), |     (C++                          |
|     [                             |     fun                           |
| \[6\]](api/languages/cpp_api.html | ction)](api/languages/cpp_api.htm |
| #_CPPv4N5cudaq14matrix_handler14m | l#_CPPv4NK5cudaq13sample_result13 |
| atrix_handlerERR14matrix_handler) | most_probableEKNSt11string_viewE) |
| -                                 | -                                 |
|  [cudaq::matrix_handler::momentum | [cudaq::sample_result::operator+= |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api/langua         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ges/cpp_api.html#_CPPv4N5cudaq13s |
| rix_handler8momentumENSt6size_tE) | ample_resultpLERK13sample_result) |
| -                                 | -                                 |
|    [cudaq::matrix_handler::number |  [cudaq::sample_result::operator= |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/langua         |
| ges/cpp_api.html#_CPPv4N5cudaq14m | ges/cpp_api.html#_CPPv4N5cudaq13s |
| atrix_handler6numberENSt6size_tE) | ample_resultaSERR13sample_result) |
| -                                 | -                                 |
| [cudaq::matrix_handler::operator= | [cudaq::sample_result::operator== |
|     (C++                          |     (C++                          |
|     fun                           |     function)](api/languag        |
| ction)](api/languages/cpp_api.htm | es/cpp_api.html#_CPPv4NK5cudaq13s |
| l#_CPPv4I0_NSt11enable_if_tIXaant | ample_resulteqERK13sample_result) |
| NSt7is_sameI1T14matrix_handlerE5v | -   [                             |
| alueENSt12is_base_of_vI16operator | cudaq::sample_result::probability |
| _handler1TEEEbEEEN5cudaq14matrix_ |     (C++                          |
| handleraSER14matrix_handlerRK1T), |     function)](api/lan            |
|     [\[1\]](api/languages         | guages/cpp_api.html#_CPPv4NK5cuda |
| /cpp_api.html#_CPPv4N5cudaq14matr | q13sample_result11probabilityENSt |
| ix_handleraSERK14matrix_handler), | 11string_viewEKNSt11string_viewE) |
|     [\[2\]](api/language          | -   [cud                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat | aq::sample_result::register_names |
| rix_handleraSERR14matrix_handler) |     (C++                          |
| -   [                             |     function)](api/langu          |
| cudaq::matrix_handler::operator== | ages/cpp_api.html#_CPPv4NK5cudaq1 |
|     (C++                          | 3sample_result14register_namesEv) |
|     function)](api/languages      | -                                 |
| /cpp_api.html#_CPPv4NK5cudaq14mat |    [cudaq::sample_result::reorder |
| rix_handlereqERK14matrix_handler) |     (C++                          |
| -                                 |     function)](api/langua         |
|    [cudaq::matrix_handler::parity | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     (C++                          | ample_result7reorderERKNSt6vector |
|     function)](api/langua         | INSt6size_tEEEKNSt11string_viewE) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -   [cu                           |
| atrix_handler6parityENSt6size_tE) | daq::sample_result::sample_result |
| -                                 |     (C++                          |
|  [cudaq::matrix_handler::position |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     function)](api/language       | daq13sample_result13sample_result |
| s/cpp_api.html#_CPPv4N5cudaq14mat | E16CountsDictionary10cudaq_json), |
| rix_handler8positionENSt6size_tE) |     [                             |
| -   [cudaq::                      | \[1\]](api/languages/cpp_api.html |
| matrix_handler::remove_definition | #_CPPv4N5cudaq13sample_result13sa |
|     (C++                          | mple_resultERK15ExecutionResult), |
|     fu                            |     [\[2\]](api/la                |
| nction)](api/languages/cpp_api.ht | nguages/cpp_api.html#_CPPv4N5cuda |
| ml#_CPPv4N5cudaq14matrix_handler1 | q13sample_result13sample_resultER |
| 7remove_definitionERKNSt6stringE) | KNSt6vectorI15ExecutionResultEE), |
| -                                 |                                   |
|   [cudaq::matrix_handler::squeeze |  [\[3\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq13sample_result13 |
|     function)](api/languag        | sample_resultERR13sample_result), |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     [                             |
| trix_handler7squeezeENSt6size_tE) | \[4\]](api/languages/cpp_api.html |
| -   [cudaq::m                     | #_CPPv4N5cudaq13sample_result13sa |
| atrix_handler::to_diagonal_matrix | mple_resultERR15ExecutionResult), |
|     (C++                          |     [\[5\]](api/lan               |
|     function)](api/lang           | guages/cpp_api.html#_CPPv4N5cudaq |
| uages/cpp_api.html#_CPPv4NK5cudaq | 13sample_result13sample_resultEdR |
| 14matrix_handler18to_diagonal_mat | KNSt6vectorI15ExecutionResultEE), |
| rixERNSt13unordered_mapINSt6size_ |     [\[6\]](api/lan               |
| tENSt7int64_tEEERKNSt13unordered_ | guages/cpp_api.html#_CPPv4N5cudaq |
| mapINSt6stringENSt7complexIdEEEE) | 13sample_result13sample_resultEv) |
| -                                 | -                                 |
| [cudaq::matrix_handler::to_matrix |  [cudaq::sample_result::serialize |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api                |
| ](api/languages/cpp_api.html#_CPP | /languages/cpp_api.html#_CPPv4NK5 |
| v4NK5cudaq14matrix_handler9to_mat | cudaq13sample_result9serializeEv) |
| rixERNSt13unordered_mapINSt6size_ | -   [cudaq::sample_result::size   |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq13sampl |
| [cudaq::matrix_handler::to_string | e_result4sizeEKNSt11string_viewE) |
|     (C++                          | -   [cudaq::sample_result::to_map |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     function)](api/languages/cpp  |
| udaq14matrix_handler9to_stringEb) | _api.html#_CPPv4NK5cudaq13sample_ |
| -                                 | result6to_mapEKNSt11string_viewE) |
| [cudaq::matrix_handler::unique_id | -   [cuda                         |
|     (C++                          | q::sample_result::\~sample_result |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     funct                         |
| udaq14matrix_handler9unique_idEv) | ion)](api/languages/cpp_api.html# |
| -   [cudaq:                       | _CPPv4N5cudaq13sample_resultD0Ev) |
| :matrix_handler::\~matrix_handler | -   [cudaq::scalar_callback (C++  |
|     (C++                          |     c                             |
|     functi                        | lass)](api/languages/cpp_api.html |
| on)](api/languages/cpp_api.html#_ | #_CPPv4N5cudaq15scalar_callbackE) |
| CPPv4N5cudaq14matrix_handlerD0Ev) | -   [c                            |
| -   [cudaq::matrix_op (C++        | udaq::scalar_callback::operator() |
|     type)](api/languages/cpp_a    |     (C++                          |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     function)](api/language       |
| -   [cudaq::matrix_op_term (C++   | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|                                   | alar_callbackclERKNSt13unordered_ |
|  type)](api/languages/cpp_api.htm | mapINSt6stringENSt7complexIdEEEE) |
| l#_CPPv4N5cudaq14matrix_op_termE) | -   [                             |
| -                                 | cudaq::scalar_callback::operator= |
|    [cudaq::mdiag_operator_handler |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     class)](                      | pp_api.html#_CPPv4N5cudaq15scalar |
| api/languages/cpp_api.html#_CPPv4 | _callbackaSERK15scalar_callback), |
| N5cudaq22mdiag_operator_handlerE) |     [\[1\]](api/languages/        |
| -   [cudaq::measure_handle (C++   | cpp_api.html#_CPPv4N5cudaq15scala |
|                                   | r_callbackaSERR15scalar_callback) |
| class)](api/languages/cpp_api.htm | -   [cudaq:                       |
| l#_CPPv4N5cudaq14measure_handleE) | :scalar_callback::scalar_callback |
| -   [cudaq::measure_result (C++   |     (C++                          |
|                                   |     function)](api/languag        |
|  type)](api/languages/cpp_api.htm | es/cpp_api.html#_CPPv4I0_NSt11ena |
| l#_CPPv4N5cudaq14measure_resultE) | ble_if_tINSt16is_invocable_r_vINS |
| -   [cudaq::mpi (C++              | t7complexIdEE8CallableRKNSt13unor |
|     type)](api/languages          | dered_mapINSt6stringENSt7complexI |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | dEEEEEEbEEEN5cudaq15scalar_callba |
| -   [cudaq::mpi::all_gather (C++  | ck15scalar_callbackERR8Callable), |
|     fu                            |     [\[1\                         |
| nction)](api/languages/cpp_api.ht | ]](api/languages/cpp_api.html#_CP |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | Pv4N5cudaq15scalar_callback15scal |
| RNSt6vectorIdEERKNSt6vectorIdEE), | ar_callbackERK15scalar_callback), |
|                                   |     [\[2                          |
|   [\[1\]](api/languages/cpp_api.h | \]](api/languages/cpp_api.html#_C |
| tml#_CPPv4N5cudaq3mpi10all_gather | PPv4N5cudaq15scalar_callback15sca |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | lar_callbackERR15scalar_callback) |
| -   [cudaq::mpi::all_reduce (C++  | -   [cudaq::scalar_operator (C++  |
|                                   |     c                             |
|  function)](api/languages/cpp_api | lass)](api/languages/cpp_api.html |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | #_CPPv4N5cudaq15scalar_operatorE) |
| reduceE1TRK1TRK14BinaryFunction), | -                                 |
|     [\[1\]](api/langu             | [cudaq::scalar_operator::evaluate |
| ages/cpp_api.html#_CPPv4I00EN5cud |     (C++                          |
| aq3mpi10all_reduceE1TRK1TRK4Func) |                                   |
| -   [cudaq::mpi::broadcast (C++   |    function)](api/languages/cpp_a |
|     function)](api/               | pi.html#_CPPv4NK5cudaq15scalar_op |
| languages/cpp_api.html#_CPPv4N5cu | erator8evaluateERKNSt13unordered_ |
| daq3mpi9broadcastERNSt6stringEi), | mapINSt6stringENSt7complexIdEEEE) |
|     [\[1\]](api/la                | -   [cudaq::scalar_ope            |
| nguages/cpp_api.html#_CPPv4N5cuda | rator::get_parameter_descriptions |
| q3mpi9broadcastERNSt6vectorIdEEi) |     (C++                          |
| -   [cudaq::mpi::finalize (C++    |     f                             |
|     f                             | unction)](api/languages/cpp_api.h |
| unction)](api/languages/cpp_api.h | tml#_CPPv4NK5cudaq15scalar_operat |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | or26get_parameter_descriptionsEv) |
| -   [cudaq::mpi::initialize (C++  | -   [cu                           |
|     function                      | daq::scalar_operator::is_constant |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq3mpi10initializeEiPPc), |     function)](api/lang           |
|     [                             | uages/cpp_api.html#_CPPv4NK5cudaq |
| \[1\]](api/languages/cpp_api.html | 15scalar_operator11is_constantEv) |
| #_CPPv4N5cudaq3mpi10initializeEv) | -   [c                            |
| -   [cudaq::mpi::is_initialized   | udaq::scalar_operator::operator\* |
|     (C++                          |     (C++                          |
|     function                      |     function                      |
| )](api/languages/cpp_api.html#_CP | )](api/languages/cpp_api.html#_CP |
| Pv4N5cudaq3mpi14is_initializedEv) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::mpi::num_ranks (C++   | 7complexIdEERK15scalar_operator), |
|     fu                            |     [\[1\                         |
| nction)](api/languages/cpp_api.ht | ]](api/languages/cpp_api.html#_CP |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::mpi::rank (C++        | 7complexIdEERR15scalar_operator), |
|                                   |     [\[2\]](api/languages/cp      |
|    function)](api/languages/cpp_a | p_api.html#_CPPv4N5cudaq15scalar_ |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | operatormlEdRK15scalar_operator), |
| -   [cudaq::noise_model (C++      |     [\[3\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|    class)](api/languages/cpp_api. | operatormlEdRR15scalar_operator), |
| html#_CPPv4N5cudaq11noise_modelE) |     [\[4\]](api/languages         |
| -   [cudaq::n                     | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| oise_model::add_all_qubit_channel | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     function)](api                | _api.html#_CPPv4NKR5cudaq15scalar |
| /languages/cpp_api.html#_CPPv4IDp | _operatormlERK15scalar_operator), |
| EN5cudaq11noise_model21add_all_qu |     [\[6\]]                       |
| bit_channelEvRK13kraus_channeli), | (api/languages/cpp_api.html#_CPPv |
|     [\[1\]](api/langua            | 4NKR5cudaq15scalar_operatormlEd), |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     [\[7\]](api/language          |
| oise_model21add_all_qubit_channel | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| ERKNSt6stringERK13kraus_channeli) | alar_operatormlENSt7complexIdEE), |
| -                                 |     [\[8\]](api/languages/cp      |
|  [cudaq::noise_model::add_channel | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatormlERK15scalar_operator), |
|     funct                         |     [\[9\                         |
| ion)](api/languages/cpp_api.html# | ]](api/languages/cpp_api.html#_CP |
| _CPPv4IDpEN5cudaq11noise_model11a | Pv4NO5cudaq15scalar_operatormlEd) |
| dd_channelEvRK15PredicateFuncTy), | -   [cu                           |
|     [\[1\]](api/languages/cpp_    | daq::scalar_operator::operator\*= |
| api.html#_CPPv4IDpEN5cudaq11noise |     (C++                          |
| _model11add_channelEvRKNSt6vector |     function)](api/languag        |
| INSt6size_tEEERK13kraus_channel), | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     [\[2\]](ap                    | alar_operatormLENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[1\]](api/languages/c       |
| cudaq11noise_model11add_channelER | pp_api.html#_CPPv4N5cudaq15scalar |
| KNSt6stringERK15PredicateFuncTy), | _operatormLERK15scalar_operator), |
|                                   |     [\[2                          |
| [\[3\]](api/languages/cpp_api.htm | \]](api/languages/cpp_api.html#_C |
| l#_CPPv4N5cudaq11noise_model11add | PPv4N5cudaq15scalar_operatormLEd) |
| _channelERKNSt6stringERKNSt6vecto | -   [                             |
| rINSt6size_tEEERK13kraus_channel) | cudaq::scalar_operator::operator+ |
| -   [cudaq::noise_model::empty    |     (C++                          |
|     (C++                          |     function                      |
|     function                      | )](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4N5cudaq15scalar_operatorplENSt |
| Pv4NK5cudaq11noise_model5emptyEv) | 7complexIdEERK15scalar_operator), |
| -                                 |     [\[1\                         |
| [cudaq::noise_model::get_channels | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatorplENSt |
|     function)](api/l              | 7complexIdEERR15scalar_operator), |
| anguages/cpp_api.html#_CPPv4I0ENK |     [\[2\]](api/languages/cp      |
| 5cudaq11noise_model12get_channels | p_api.html#_CPPv4N5cudaq15scalar_ |
| ENSt6vectorI13kraus_channelEERKNS | operatorplEdRK15scalar_operator), |
| t6vectorINSt6size_tEEERKNSt6vecto |     [\[3\]](api/languages/cp      |
| rINSt6size_tEEERKNSt6vectorIdEE), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](api/languages/cpp_a   | operatorplEdRR15scalar_operator), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[4\]](api/languages         |
| el12get_channelsERKNSt6stringERKN | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| St6vectorINSt6size_tEEERKNSt6vect | alar_operatorplENSt7complexIdEE), |
| orINSt6size_tEEERKNSt6vectorIdEE) |     [\[5\]](api/languages/cpp     |
| -                                 | _api.html#_CPPv4NKR5cudaq15scalar |
|  [cudaq::noise_model::noise_model | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function)](api                | (api/languages/cpp_api.html#_CPPv |
| /languages/cpp_api.html#_CPPv4N5c | 4NKR5cudaq15scalar_operatorplEd), |
| udaq11noise_model11noise_modelEv) |     [\[7\]]                       |
| -   [cu                           | (api/languages/cpp_api.html#_CPPv |
| daq::noise_model::PredicateFuncTy | 4NKR5cudaq15scalar_operatorplEv), |
|     (C++                          |     [\[8\]](api/language          |
|     type)](api/la                 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| nguages/cpp_api.html#_CPPv4N5cuda | alar_operatorplENSt7complexIdEE), |
| q11noise_model15PredicateFuncTyE) |     [\[9\]](api/languages/cp      |
| -   [cud                          | p_api.html#_CPPv4NO5cudaq15scalar |
| aq::noise_model::register_channel | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[10\]                       |
|     function)](api/languages      | ](api/languages/cpp_api.html#_CPP |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | v4NO5cudaq15scalar_operatorplEd), |
| noise_model16register_channelEvv) |     [\[11\                        |
| -   [cudaq::                      | ]](api/languages/cpp_api.html#_CP |
| noise_model::requires_constructor | Pv4NO5cudaq15scalar_operatorplEv) |
|     (C++                          | -   [c                            |
|     type)](api/languages/cp       | udaq::scalar_operator::operator+= |
| p_api.html#_CPPv4I0DpEN5cudaq11no |     (C++                          |
| ise_model20requires_constructorE) |     function)](api/languag        |
| -   [cudaq::noise_model_type (C++ | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     e                             | alar_operatorpLENSt7complexIdEE), |
| num)](api/languages/cpp_api.html# |     [\[1\]](api/languages/c       |
| _CPPv4N5cudaq16noise_model_typeE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::no                    | _operatorpLERK15scalar_operator), |
| ise_model_type::amplitude_damping |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     enumerator)](api/languages    | PPv4N5cudaq15scalar_operatorpLEd) |
| /cpp_api.html#_CPPv4N5cudaq16nois | -   [                             |
| e_model_type17amplitude_dampingE) | cudaq::scalar_operator::operator- |
| -   [cudaq::noise_mode            |     (C++                          |
| l_type::amplitude_damping_channel |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     e                             | Pv4N5cudaq15scalar_operatormiENSt |
| numerator)](api/languages/cpp_api | 7complexIdEERK15scalar_operator), |
| .html#_CPPv4N5cudaq16noise_model_ |     [\[1\                         |
| type25amplitude_damping_channelE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::n                     | Pv4N5cudaq15scalar_operatormiENSt |
| oise_model_type::bit_flip_channel | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     enumerator)](api/language     | p_api.html#_CPPv4N5cudaq15scalar_ |
| s/cpp_api.html#_CPPv4N5cudaq16noi | operatormiEdRK15scalar_operator), |
| se_model_type16bit_flip_channelE) |     [\[3\]](api/languages/cp      |
| -   [cudaq::                      | p_api.html#_CPPv4N5cudaq15scalar_ |
| noise_model_type::depolarization1 | operatormiEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     enumerator)](api/languag      | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| es/cpp_api.html#_CPPv4N5cudaq16no | alar_operatormiENSt7complexIdEE), |
| ise_model_type15depolarization1E) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::                      | _api.html#_CPPv4NKR5cudaq15scalar |
| noise_model_type::depolarization2 | _operatormiERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     enumerator)](api/languag      | (api/languages/cpp_api.html#_CPPv |
| es/cpp_api.html#_CPPv4N5cudaq16no | 4NKR5cudaq15scalar_operatormiEd), |
| ise_model_type15depolarization2E) |     [\[7\]]                       |
| -   [cudaq::noise_m               | (api/languages/cpp_api.html#_CPPv |
| odel_type::depolarization_channel | 4NKR5cudaq15scalar_operatormiEv), |
|     (C++                          |     [\[8\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|   enumerator)](api/languages/cpp_ | alar_operatormiENSt7complexIdEE), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[9\]](api/languages/cp      |
| el_type22depolarization_channelE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -                                 | _operatormiERK15scalar_operator), |
|  [cudaq::noise_model_type::pauli1 |     [\[10\]                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     enumerator)](a                | v4NO5cudaq15scalar_operatormiEd), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[11\                        |
| 5cudaq16noise_model_type6pauli1E) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4NO5cudaq15scalar_operatormiEv) |
|  [cudaq::noise_model_type::pauli2 | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator-= |
|     enumerator)](a                |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/languag        |
| 5cudaq16noise_model_type6pauli2E) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq                        | alar_operatormIENSt7complexIdEE), |
| ::noise_model_type::phase_damping |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](api/langu        | _operatormIERK15scalar_operator), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     [\[2                          |
| noise_model_type13phase_dampingE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::noi                   | PPv4N5cudaq15scalar_operatormIEd) |
| se_model_type::phase_flip_channel | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator/ |
|     enumerator)](api/languages/   |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16noise |     function                      |
| _model_type18phase_flip_channelE) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatordvENSt |
| [cudaq::noise_model_type::unknown | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](ap               | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4N5cudaq15scalar_operatordvENSt |
| cudaq16noise_model_type7unknownE) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
| [cudaq::noise_model_type::x_error | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRK15scalar_operator), |
|     enumerator)](ap               |     [\[3\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4N5cudaq15scalar_ |
| cudaq16noise_model_type7x_errorE) | operatordvEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
| [cudaq::noise_model_type::y_error | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatordvENSt7complexIdEE), |
|     enumerator)](ap               |     [\[5\]](api/languages/cpp     |
| i/languages/cpp_api.html#_CPPv4N5 | _api.html#_CPPv4NKR5cudaq15scalar |
| cudaq16noise_model_type7y_errorE) | _operatordvERK15scalar_operator), |
| -                                 |     [\[6\]]                       |
| [cudaq::noise_model_type::z_error | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatordvEd), |
|     enumerator)](ap               |     [\[7\]](api/language          |
| i/languages/cpp_api.html#_CPPv4N5 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| cudaq16noise_model_type7z_errorE) | alar_operatordvENSt7complexIdEE), |
| -   [cudaq::num_available_gpus    |     [\[8\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     function                      | _operatordvERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[9\                         |
| Pv4N5cudaq18num_available_gpusEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::observe (C++          | Pv4NO5cudaq15scalar_operatordvEd) |
|     function)]                    | -   [c                            |
| (api/languages/cpp_api.html#_CPPv | udaq::scalar_operator::operator/= |
| 4I00DpEN5cudaq7observeENSt6vector |     (C++                          |
| I14observe_resultEERR13QuantumKer |     function)](api/languag        |
| nelRK15SpinOpContainerDpRR4Args), | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     [\[1\]](api/languages/cpp_ap  | alar_operatordVENSt7complexIdEE), |
| i.html#_CPPv4I0DpEN5cudaq7observe |     [\[1\]](api/languages/c       |
| E14observe_resultNSt6size_tERR13Q | pp_api.html#_CPPv4N5cudaq15scalar |
| uantumKernelRK7spin_opDpRR4Args), | _operatordVERK15scalar_operator), |
|     [\[                           |     [\[2                          |
| 2\]](api/languages/cpp_api.html#_ | \]](api/languages/cpp_api.html#_C |
| CPPv4I0DpEN5cudaq7observeE14obser | PPv4N5cudaq15scalar_operatordVEd) |
| ve_resultRK15observe_optionsRR13Q | -   [                             |
| uantumKernelRK7spin_opDpRR4Args), | cudaq::scalar_operator::operator= |
|     [\[3\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     function)](api/languages/c    |
| udaq7observeE14observe_resultRR13 | pp_api.html#_CPPv4N5cudaq15scalar |
| QuantumKernelRK7spin_opDpRR4Args) | _operatoraSERK15scalar_operator), |
| -   [cudaq::observe_options (C++  |     [\[1\]](api/languages/        |
|     st                            | cpp_api.html#_CPPv4N5cudaq15scala |
| ruct)](api/languages/cpp_api.html | r_operatoraSERR15scalar_operator) |
| #_CPPv4N5cudaq15observe_optionsE) | -   [c                            |
| -   [cudaq::observe_result (C++   | udaq::scalar_operator::operator== |
|                                   |     (C++                          |
| class)](api/languages/cpp_api.htm |     function)](api/languages/c    |
| l#_CPPv4N5cudaq14observe_resultE) | pp_api.html#_CPPv4NK5cudaq15scala |
| -                                 | r_operatoreqERK15scalar_operator) |
|    [cudaq::observe_result::counts | -   [cudaq:                       |
|     (C++                          | :scalar_operator::scalar_operator |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq14observ |     func                          |
| e_result6countsERK12spin_op_term) | tion)](api/languages/cpp_api.html |
| -   [cudaq::observe_result::dump  | #_CPPv4N5cudaq15scalar_operator15 |
|     (C++                          | scalar_operatorENSt7complexIdEE), |
|     function)                     |     [\[1\]](api/langu             |
| ](api/languages/cpp_api.html#_CPP | ages/cpp_api.html#_CPPv4N5cudaq15 |
| v4N5cudaq14observe_result4dumpEv) | scalar_operator15scalar_operatorE |
| -   [c                            | RK15scalar_callbackRRNSt13unorder |
| udaq::observe_result::expectation | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[2\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
| function)](api/languages/cpp_api. | Pv4N5cudaq15scalar_operator15scal |
| html#_CPPv4N5cudaq14observe_resul | ar_operatorERK15scalar_operator), |
| t11expectationERK12spin_op_term), |     [\[3\]](api/langu             |
|     [\[1\]](api/la                | ages/cpp_api.html#_CPPv4N5cudaq15 |
| nguages/cpp_api.html#_CPPv4N5cuda | scalar_operator15scalar_operatorE |
| q14observe_result11expectationEv) | RR15scalar_callbackRRNSt13unorder |
| -   [cuda                         | ed_mapINSt6stringENSt6stringEEE), |
| q::observe_result::id_coefficient |     [\[4\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/langu          | Pv4N5cudaq15scalar_operator15scal |
| ages/cpp_api.html#_CPPv4N5cudaq14 | ar_operatorERR15scalar_operator), |
| observe_result14id_coefficientEv) |     [\[5\]](api/language          |
| -   [cuda                         | s/cpp_api.html#_CPPv4N5cudaq15sca |
| q::observe_result::observe_result | lar_operator15scalar_operatorEd), |
|     (C++                          |     [\[6\]](api/languag           |
|                                   | es/cpp_api.html#_CPPv4N5cudaq15sc |
|   function)](api/languages/cpp_ap | alar_operator15scalar_operatorEv) |
| i.html#_CPPv4N5cudaq14observe_res | -   [                             |
| ult14observe_resultEdRK7spin_op), | cudaq::scalar_operator::to_matrix |
|     [\[1\]](a                     |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |                                   |
| 5cudaq14observe_result14observe_r |   function)](api/languages/cpp_ap |
| esultEdRK7spin_op13sample_result) | i.html#_CPPv4NK5cudaq15scalar_ope |
| -                                 | rator9to_matrixERKNSt13unordered_ |
|  [cudaq::observe_result::operator | mapINSt6stringENSt7complexIdEEEE) |
|     double (C++                   | -   [                             |
|     functio                       | cudaq::scalar_operator::to_string |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq14observe_resultcvdEv) |     function)](api/l              |
| -                                 | anguages/cpp_api.html#_CPPv4NK5cu |
|  [cudaq::observe_result::raw_data | daq15scalar_operator9to_stringEv) |
|     (C++                          | -   [cudaq::s                     |
|     function)](ap                 | calar_operator::\~scalar_operator |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq14observe_result8raw_dataEv) |     functio                       |
| -   [cudaq::operator_handler (C++ | n)](api/languages/cpp_api.html#_C |
|     cl                            | PPv4N5cudaq15scalar_operatorD0Ev) |
| ass)](api/languages/cpp_api.html# | -   [cudaq::set_noise (C++        |
| _CPPv4N5cudaq16operator_handlerE) |     function)](api/langu          |
| -   [cudaq::optimizable_function  | ages/cpp_api.html#_CPPv4N5cudaq9s |
|     (C++                          | et_noiseERKN5cudaq11noise_modelE) |
|     class)                        | -   [cudaq::set_random_seed (C++  |
| ](api/languages/cpp_api.html#_CPP |     function)](api/               |
| v4N5cudaq20optimizable_functionE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::optimization_result   | daq15set_random_seedENSt6size_tE) |
|     (C++                          | -   [cudaq::simulation_precision  |
|     type                          |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     enum)                         |
| Pv4N5cudaq19optimization_resultE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::optimizer (C++        | v4N5cudaq20simulation_precisionE) |
|     class)](api/languages/cpp_a   | -   [                             |
| pi.html#_CPPv4N5cudaq9optimizerE) | cudaq::simulation_precision::fp32 |
| -   [cudaq::optimizer::optimize   |     (C++                          |
|     (C++                          |     enumerator)](api              |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
|  function)](api/languages/cpp_api | udaq20simulation_precision4fp32E) |
| .html#_CPPv4N5cudaq9optimizer8opt | -   [                             |
| imizeEKiRR20optimizable_function) | cudaq::simulation_precision::fp64 |
| -   [cu                           |     (C++                          |
| daq::optimizer::requiresGradients |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/la             | udaq20simulation_precision4fp64E) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::SimulationState (C++  |
| q9optimizer17requiresGradientsEv) |     c                             |
| -   [cudaq::orca (C++             | lass)](api/languages/cpp_api.html |
|     type)](api/languages/         | #_CPPv4N5cudaq15SimulationStateE) |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | -   [                             |
| -   [cudaq::orca::sample (C++     | cudaq::SimulationState::precision |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     enum)](api                    |
| mpleERNSt6vectorINSt6size_tEEERNS | /languages/cpp_api.html#_CPPv4N5c |
| t6vectorINSt6size_tEEERNSt6vector | udaq15SimulationState9precisionE) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [cudaq:                       |
|     [\[1\]]                       | :SimulationState::precision::fp32 |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq4orca6sampleERNSt6vectorI |     enumerator)](api/lang         |
| NSt6size_tEEERNSt6vectorINSt6size | uages/cpp_api.html#_CPPv4N5cudaq1 |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | 5SimulationState9precision4fp32E) |
| -   [cudaq::orca::sample_async    | -   [cudaq:                       |
|     (C++                          | :SimulationState::precision::fp64 |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     enumerator)](api/lang         |
| html#_CPPv4N5cudaq4orca12sample_a | uages/cpp_api.html#_CPPv4N5cudaq1 |
| syncERNSt6vectorINSt6size_tEEERNS | 5SimulationState9precision4fp64E) |
| t6vectorINSt6size_tEEERNSt6vector | -                                 |
| IdEERNSt6vectorIdEEiNSt6size_tE), |   [cudaq::SimulationState::Tensor |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     struct)](                     |
| q4orca12sample_asyncERNSt6vectorI | api/languages/cpp_api.html#_CPPv4 |
| NSt6size_tEEERNSt6vectorINSt6size | N5cudaq15SimulationState6TensorE) |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | -   [cudaq::spin_handler (C++     |
| -   [cudaq::OrcaRemoteRESTQPU     |                                   |
|     (C++                          |   class)](api/languages/cpp_api.h |
|     cla                           | tml#_CPPv4N5cudaq12spin_handlerE) |
| ss)](api/languages/cpp_api.html#_ | -   [cudaq:                       |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | :spin_handler::to_diagonal_matrix |
| -   [cudaq::other_policies (C++   |     (C++                          |
|     s                             |     function)](api/la             |
| truct)](api/languages/cpp_api.htm | nguages/cpp_api.html#_CPPv4NK5cud |
| l#_CPPv4N5cudaq14other_policiesE) | aq12spin_handler18to_diagonal_mat |
| -   [cudaq::PasqalRemoteRESTQPU   | rixERNSt13unordered_mapINSt6size_ |
|     (C++                          | tENSt7int64_tEEERKNSt13unordered_ |
|     class                         | mapINSt6stringENSt7complexIdEEEE) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq19PasqalRemoteRESTQPUE) |   [cudaq::spin_handler::to_matrix |
| -   [cudaq::pauli1 (C++           |     (C++                          |
|     class)](api/languages/cp      |     function                      |
| p_api.html#_CPPv4N5cudaq6pauli1E) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq12spin_handler9to_matri |
|    [cudaq::pauli1::num_parameters | xERKNSt6stringENSt7complexIdEEb), |
|     (C++                          |     [\[1                          |
|     member)]                      | \]](api/languages/cpp_api.html#_C |
| (api/languages/cpp_api.html#_CPPv | PPv4NK5cudaq12spin_handler9to_mat |
| 4N5cudaq6pauli114num_parametersE) | rixERNSt13unordered_mapINSt6size_ |
| -   [cudaq::pauli1::num_targets   | tENSt7int64_tEEERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|     membe                         | -   [cuda                         |
| r)](api/languages/cpp_api.html#_C | q::spin_handler::to_sparse_matrix |
| PPv4N5cudaq6pauli111num_targetsE) |     (C++                          |
| -   [cudaq::pauli1::pauli1 (C++   |     function)](api/               |
|     function)](api/languages/cpp_ | languages/cpp_api.html#_CPPv4N5cu |
| api.html#_CPPv4N5cudaq6pauli16pau | daq12spin_handler16to_sparse_matr |
| li1ERKNSt6vectorIN5cudaq4realEEE) | ixERKNSt6stringENSt7complexIdEEb) |
| -   [cudaq::pauli2 (C++           | -                                 |
|     class)](api/languages/cp      |   [cudaq::spin_handler::to_string |
| p_api.html#_CPPv4N5cudaq6pauli2E) |     (C++                          |
| -                                 |     function)](ap                 |
|    [cudaq::pauli2::num_parameters | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq12spin_handler9to_stringEb) |
|     member)]                      | -                                 |
| (api/languages/cpp_api.html#_CPPv |   [cudaq::spin_handler::unique_id |
| 4N5cudaq6pauli214num_parametersE) |     (C++                          |
| -   [cudaq::pauli2::num_targets   |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     membe                         | 5cudaq12spin_handler9unique_idEv) |
| r)](api/languages/cpp_api.html#_C | -   [cudaq::spin_op (C++          |
| PPv4N5cudaq6pauli211num_targetsE) |     type)](api/languages/cpp      |
| -   [cudaq::pauli2::pauli2 (C++   | _api.html#_CPPv4N5cudaq7spin_opE) |
|     function)](api/languages/cpp_ | -   [cudaq::spin_op_term (C++     |
| api.html#_CPPv4N5cudaq6pauli26pau |                                   |
| li2ERKNSt6vectorIN5cudaq4realEEE) |    type)](api/languages/cpp_api.h |
| -   [cudaq::phase_damping (C++    | tml#_CPPv4N5cudaq12spin_op_termE) |
|                                   | -   [cudaq::state (C++            |
|  class)](api/languages/cpp_api.ht |     class)](api/languages/c       |
| ml#_CPPv4N5cudaq13phase_dampingE) | pp_api.html#_CPPv4N5cudaq5stateE) |
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
| -   [scalarCaster()               |   property)](api/languages/python |
|     (cudaq.PyKernelDecorator      | _api.html#cudaq.Target.simulator) |
|     method)](a                    | -   [slice() (cudaq.QuakeValue    |
| pi/languages/python_api.html#cuda |     method)](api/languages/python |
| q.PyKernelDecorator.scalarCaster) | _api.html#cudaq.QuakeValue.slice) |
| -   [ScalarOperator (class in     | -   [SpinOperator (class in       |
|     cudaq.operato                 |     cudaq.operators.spin)         |
| rs)](api/languages/python_api.htm | ](api/languages/python_api.html#c |
| l#cudaq.operators.ScalarOperator) | udaq.operators.spin.SpinOperator) |
| -   [Schedule (class in           | -   [SpinOperatorElement (class   |
|     cudaq)](api/language          |     in                            |
| s/python_api.html#cudaq.Schedule) |     cudaq.operators.spin)](api/l  |
| -   [serialize                    | anguages/python_api.html#cudaq.op |
|     (                             | erators.spin.SpinOperatorElement) |
| cudaq.operators.spin.SpinOperator | -   [SpinOperatorTerm (class in   |
|     attribute)](api/lang          |     cudaq.operators.spin)](ap     |
| uages/python_api.html#cudaq.opera | i/languages/python_api.html#cudaq |
| tors.spin.SpinOperator.serialize) | .operators.spin.SpinOperatorTerm) |
|     -   [(cuda                    | -   [split_communicator() (in     |
| q.operators.spin.SpinOperatorTerm |     module                        |
|         attribute)](api/language  |     cudaq                         |
| s/python_api.html#cudaq.operators | .mpi)](api/languages/python_api.h |
| .spin.SpinOperatorTerm.serialize) | tml#cudaq.mpi.split_communicator) |
|     -   [(cudaq.SampleResult      | -   [SPSA (class in               |
|         attri                     |     cudaq                         |
| bute)](api/languages/python_api.h | .optimizers)](api/languages/pytho |
| tml#cudaq.SampleResult.serialize) | n_api.html#cudaq.optimizers.SPSA) |
| -   [set_communicator() (in       | -   [State (class in              |
|     module                        |     cudaq)](api/langu             |
|     cud                           | ages/python_api.html#cudaq.State) |
| aq.mpi)](api/languages/python_api | -   [step_size                    |
| .html#cudaq.mpi.set_communicator) |     (cudaq.optimizers.Adam        |
| -   [set_noise() (in module       |     propert                       |
|     cudaq)](api/languages         | y)](api/languages/python_api.html |
| /python_api.html#cudaq.set_noise) | #cudaq.optimizers.Adam.step_size) |
| -   [set_random_seed() (in module |     -   [(cudaq.optimizers.SGD    |
|     cudaq)](api/languages/pytho   |         proper                    |
| n_api.html#cudaq.set_random_seed) | ty)](api/languages/python_api.htm |
| -   [set_target() (in module      | l#cudaq.optimizers.SGD.step_size) |
|     cudaq)](api/languages/        |     -   [(cudaq.optimizers.SPSA   |
| python_api.html#cudaq.set_target) |         propert                   |
| -   [SGD (class in                | y)](api/languages/python_api.html |
|     cuda                          | #cudaq.optimizers.SPSA.step_size) |
| q.optimizers)](api/languages/pyth | -   [SuperOperator (class in      |
| on_api.html#cudaq.optimizers.SGD) |     cudaq)](api/languages/pyt     |
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
