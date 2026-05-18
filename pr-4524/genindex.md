::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4524
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
    -   [Max-Cut with QAOA](applications/python/qaoa.html){.reference
        .internal}
    -   [Molecular docking via
        DC-QAOA](applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
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
    -   [Bernstein-Vazirani
        Algorithm](applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](applications/python/hybrid_quantum_neural_networks.html){.reference
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
    -   [Quantum Fourier
        Transform](applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](applications/python/quantum_teleportation.html#Teleportation-explained){.reference
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
    -   [Compiling Unitaries Using Diffusion
        Models](applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
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
    -   [ADAPT-QAOA
        algorithm](applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A_j\\)]{.math .notranslate
            .nohighlight}:](applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H_C,A_j\]\\)]{.math .notranslate
            .nohighlight}:](applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A_i\\)]{.math .notranslate
            .nohighlight}\]](applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](applications/python/edge_detection.html#Post-processing){.reference
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
    -   [Grover's
        Algorithm](applications/python/grovers.html){.reference
        .internal}
        -   [Overview](applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
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
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
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
| -   [AmplitudeDampingChannel      | -   [AsyncObserveResult (class in |
|     (class in                     |                                   |
|     cu                            |    cudaq)](api/languages/python_a |
| daq)](api/languages/python_api.ht | pi.html#cudaq.AsyncObserveResult) |
| ml#cudaq.AmplitudeDampingChannel) | -   [AsyncSampleResult (class in  |
| -   [amplitudes (cudaq.State      |     cudaq)](api/languages/python_ |
|                                   | api.html#cudaq.AsyncSampleResult) |
|  attribute)](api/languages/python | -   [AsyncStateResult (class in   |
| _api.html#cudaq.State.amplitudes) |     cudaq)](api/languages/python  |
|                                   | _api.html#cudaq.AsyncStateResult) |
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
| -   [canonicalize                 | -   [cudaq::produ                 |
|     (cu                           | ct_op::const_iterator::operator== |
| daq.operators.boson.BosonOperator |     (C++                          |
|     attribute)](api/languages     |     fun                           |
| /python_api.html#cudaq.operators. | ction)](api/languages/cpp_api.htm |
| boson.BosonOperator.canonicalize) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(cudaq.                  | st_iteratoreqERK14const_iterator) |
| operators.boson.BosonOperatorTerm | -   [cudaq::product_op::degrees   |
|                                   |     (C++                          |
|     attribute)](api/languages/pyt |     function)                     |
| hon_api.html#cudaq.operators.boso | ](api/languages/cpp_api.html#_CPP |
| n.BosonOperatorTerm.canonicalize) | v4NK5cudaq10product_op7degreesEv) |
|     -   [(cudaq.                  | -   [cudaq::product_op::dump (C++ |
| operators.fermion.FermionOperator |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|     attribute)](api/languages/pyt | CPPv4NK5cudaq10product_op4dumpEv) |
| hon_api.html#cudaq.operators.ferm | -   [cudaq::product_op::end (C++  |
| ion.FermionOperator.canonicalize) |     funct                         |
|     -   [(cudaq.oper              | ion)](api/languages/cpp_api.html# |
| ators.fermion.FermionOperatorTerm | _CPPv4NK5cudaq10product_op3endEv) |
|                                   | -   [c                            |
| attribute)](api/languages/python_ | udaq::product_op::get_coefficient |
| api.html#cudaq.operators.fermion. |     (C++                          |
| FermionOperatorTerm.canonicalize) |     function)](api/lan            |
|     -                             | guages/cpp_api.html#_CPPv4NK5cuda |
|  [(cudaq.operators.MatrixOperator | q10product_op15get_coefficientEv) |
|         attribute)](api/lang      | -                                 |
| uages/python_api.html#cudaq.opera |   [cudaq::product_op::get_term_id |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     function)](api                |
| udaq.operators.MatrixOperatorTerm | /languages/cpp_api.html#_CPPv4NK5 |
|         attribute)](api/language  | cudaq10product_op11get_term_idEv) |
| s/python_api.html#cudaq.operators | -                                 |
| .MatrixOperatorTerm.canonicalize) |   [cudaq::product_op::is_identity |
|     -   [(                        |     (C++                          |
| cudaq.operators.spin.SpinOperator |     function)](api                |
|         attribute)](api/languag   | /languages/cpp_api.html#_CPPv4NK5 |
| es/python_api.html#cudaq.operator | cudaq10product_op11is_identityEv) |
| s.spin.SpinOperator.canonicalize) | -   [cudaq::product_op::num_ops   |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
|       attribute)](api/languages/p | v4NK5cudaq10product_op7num_opsEv) |
| ython_api.html#cudaq.operators.sp | -                                 |
| in.SpinOperatorTerm.canonicalize) |    [cudaq::product_op::operator\* |
| -   [captured_variables()         |     (C++                          |
|     (cudaq.PyKernelDecorator      |     function)](api/languages/     |
|     method)](api/lan              | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| guages/python_api.html#cudaq.PyKe | oduct_opmlE10product_opI1TERK15sc |
| rnelDecorator.captured_variables) | alar_operatorRK10product_opI1TE), |
| -   [CentralDifference (class in  |     [\[1\]](api/languages/        |
|     cudaq.gradients)              | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| ](api/languages/python_api.html#c | oduct_opmlE10product_opI1TERK15sc |
| udaq.gradients.CentralDifference) | alar_operatorRR10product_opI1TE), |
| -   [channel                      |     [\[2\]](api/languages/        |
|     (cudaq.ptsbe.TraceInstruction | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     property)](a                  | oduct_opmlE10product_opI1TERR15sc |
| pi/languages/python_api.html#cuda | alar_operatorRK10product_opI1TE), |
| q.ptsbe.TraceInstruction.channel) |     [\[3\]](api/languages/        |
| -   [circuit_location             | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     (cudaq.ptsbe.KrausSelection   | oduct_opmlE10product_opI1TERR15sc |
|     property)](api/lang           | alar_operatorRR10product_opI1TE), |
| uages/python_api.html#cudaq.ptsbe |     [\[4\]](api/                  |
| .KrausSelection.circuit_location) | languages/cpp_api.html#_CPPv4I0EN |
| -   [clear (cudaq.Resources       | 5cudaq10product_opmlE6sum_opI1TER |
|                                   | K15scalar_operatorRK6sum_opI1TE), |
|   attribute)](api/languages/pytho |     [\[5\]](api/                  |
| n_api.html#cudaq.Resources.clear) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cudaq.SampleResult      | 5cudaq10product_opmlE6sum_opI1TER |
|         a                         | K15scalar_operatorRR6sum_opI1TE), |
| ttribute)](api/languages/python_a |     [\[6\]](api/                  |
| pi.html#cudaq.SampleResult.clear) | languages/cpp_api.html#_CPPv4I0EN |
| -   [COBYLA (class in             | 5cudaq10product_opmlE6sum_opI1TER |
|     cudaq.o                       | R15scalar_operatorRK6sum_opI1TE), |
| ptimizers)](api/languages/python_ |     [\[7\]](api/                  |
| api.html#cudaq.optimizers.COBYLA) | languages/cpp_api.html#_CPPv4I0EN |
| -   [coefficient                  | 5cudaq10product_opmlE6sum_opI1TER |
|     (cudaq.                       | R15scalar_operatorRR6sum_opI1TE), |
| operators.boson.BosonOperatorTerm |     [\[8\]](api/languages         |
|     property)](api/languages/py   | /cpp_api.html#_CPPv4NK5cudaq10pro |
| thon_api.html#cudaq.operators.bos | duct_opmlERK6sum_opI9HandlerTyE), |
| on.BosonOperatorTerm.coefficient) |     [\[9\]](api/languages/cpp_a   |
|     -   [(cudaq.oper              | pi.html#_CPPv4NKR5cudaq10product_ |
| ators.fermion.FermionOperatorTerm | opmlERK10product_opI9HandlerTyE), |
|                                   |     [\[10\]](api/language         |
|   property)](api/languages/python | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| _api.html#cudaq.operators.fermion | roduct_opmlERK15scalar_operator), |
| .FermionOperatorTerm.coefficient) |     [\[11\]](api/languages/cpp_a  |
|     -   [(c                       | pi.html#_CPPv4NKR5cudaq10product_ |
| udaq.operators.MatrixOperatorTerm | opmlERR10product_opI9HandlerTyE), |
|         property)](api/languag    |     [\[12\]](api/language         |
| es/python_api.html#cudaq.operator | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| s.MatrixOperatorTerm.coefficient) | roduct_opmlERR15scalar_operator), |
|     -   [(cuda                    |     [\[13\]](api/languages/cpp_   |
| q.operators.spin.SpinOperatorTerm | api.html#_CPPv4NO5cudaq10product_ |
|         property)](api/languages/ | opmlERK10product_opI9HandlerTyE), |
| python_api.html#cudaq.operators.s |     [\[14\]](api/languag          |
| pin.SpinOperatorTerm.coefficient) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [col_count                    | roduct_opmlERK15scalar_operator), |
|     (cudaq.KrausOperator          |     [\[15\]](api/languages/cpp_   |
|     prope                         | api.html#_CPPv4NO5cudaq10product_ |
| rty)](api/languages/python_api.ht | opmlERR10product_opI9HandlerTyE), |
| ml#cudaq.KrausOperator.col_count) |     [\[16\]](api/langua           |
| -   [compile()                    | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     (cudaq.PyKernelDecorator      | product_opmlERR15scalar_operator) |
|     metho                         | -                                 |
| d)](api/languages/python_api.html |   [cudaq::product_op::operator\*= |
| #cudaq.PyKernelDecorator.compile) |     (C++                          |
| -   [ComplexMatrix (class in      |     function)](api/languages/cpp  |
|     cudaq)](api/languages/pyt     | _api.html#_CPPv4N5cudaq10product_ |
| hon_api.html#cudaq.ComplexMatrix) | opmLERK10product_opI9HandlerTyE), |
| -   [compute                      |     [\[1\]](api/langua            |
|     (                             | ges/cpp_api.html#_CPPv4N5cudaq10p |
| cudaq.gradients.CentralDifference | roduct_opmLERK15scalar_operator), |
|     attribute)](api/la            |     [\[2\]](api/languages/cp      |
| nguages/python_api.html#cudaq.gra | p_api.html#_CPPv4N5cudaq10product |
| dients.CentralDifference.compute) | _opmLERR10product_opI9HandlerTyE) |
|     -   [(                        | -   [cudaq::product_op::operator+ |
| cudaq.gradients.ForwardDifference |     (C++                          |
|         attribute)](api/la        |     function)](api/langu          |
| nguages/python_api.html#cudaq.gra | ages/cpp_api.html#_CPPv4I0EN5cuda |
| dients.ForwardDifference.compute) | q10product_opplE6sum_opI1TERK15sc |
|     -                             | alar_operatorRK10product_opI1TE), |
|  [(cudaq.gradients.ParameterShift |     [\[1\]](api/                  |
|         attribute)](api           | languages/cpp_api.html#_CPPv4I0EN |
| /languages/python_api.html#cudaq. | 5cudaq10product_opplE6sum_opI1TER |
| gradients.ParameterShift.compute) | K15scalar_operatorRK6sum_opI1TE), |
| -   [const()                      |     [\[2\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|   (cudaq.operators.ScalarOperator | q10product_opplE6sum_opI1TERK15sc |
|     class                         | alar_operatorRR10product_opI1TE), |
|     method)](a                    |     [\[3\]](api/                  |
| pi/languages/python_api.html#cuda | languages/cpp_api.html#_CPPv4I0EN |
| q.operators.ScalarOperator.const) | 5cudaq10product_opplE6sum_opI1TER |
| -   [controls                     | K15scalar_operatorRR6sum_opI1TE), |
|     (cudaq.ptsbe.TraceInstruction |     [\[4\]](api/langu             |
|     property)](ap                 | ages/cpp_api.html#_CPPv4I0EN5cuda |
| i/languages/python_api.html#cudaq | q10product_opplE6sum_opI1TERR15sc |
| .ptsbe.TraceInstruction.controls) | alar_operatorRK10product_opI1TE), |
| -   [copy                         |     [\[5\]](api/                  |
|     (cu                           | languages/cpp_api.html#_CPPv4I0EN |
| daq.operators.boson.BosonOperator | 5cudaq10product_opplE6sum_opI1TER |
|     attribute)](api/l             | R15scalar_operatorRK6sum_opI1TE), |
| anguages/python_api.html#cudaq.op |     [\[6\]](api/langu             |
| erators.boson.BosonOperator.copy) | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     -   [(cudaq.                  | q10product_opplE6sum_opI1TERR15sc |
| operators.boson.BosonOperatorTerm | alar_operatorRR10product_opI1TE), |
|         attribute)](api/langu     |     [\[7\]](api/                  |
| ages/python_api.html#cudaq.operat | languages/cpp_api.html#_CPPv4I0EN |
| ors.boson.BosonOperatorTerm.copy) | 5cudaq10product_opplE6sum_opI1TER |
|     -   [(cudaq.                  | R15scalar_operatorRR6sum_opI1TE), |
| operators.fermion.FermionOperator |     [\[8\]](api/languages/cpp_a   |
|         attribute)](api/langu     | pi.html#_CPPv4NKR5cudaq10product_ |
| ages/python_api.html#cudaq.operat | opplERK10product_opI9HandlerTyE), |
| ors.fermion.FermionOperator.copy) |     [\[9\]](api/language          |
|     -   [(cudaq.oper              | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ators.fermion.FermionOperatorTerm | roduct_opplERK15scalar_operator), |
|         attribute)](api/languages |     [\[10\]](api/languages/       |
| /python_api.html#cudaq.operators. | cpp_api.html#_CPPv4NKR5cudaq10pro |
| fermion.FermionOperatorTerm.copy) | duct_opplERK6sum_opI9HandlerTyE), |
|     -                             |     [\[11\]](api/languages/cpp_a  |
|  [(cudaq.operators.MatrixOperator | pi.html#_CPPv4NKR5cudaq10product_ |
|         attribute)](              | opplERR10product_opI9HandlerTyE), |
| api/languages/python_api.html#cud |     [\[12\]](api/language         |
| aq.operators.MatrixOperator.copy) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [(c                       | roduct_opplERR15scalar_operator), |
| udaq.operators.MatrixOperatorTerm |     [\[13\]](api/languages/       |
|         attribute)](api/          | cpp_api.html#_CPPv4NKR5cudaq10pro |
| languages/python_api.html#cudaq.o | duct_opplERR6sum_opI9HandlerTyE), |
| perators.MatrixOperatorTerm.copy) |     [\[                           |
|     -   [(                        | 14\]](api/languages/cpp_api.html# |
| cudaq.operators.spin.SpinOperator | _CPPv4NKR5cudaq10product_opplEv), |
|         attribute)](api           |     [\[15\]](api/languages/cpp_   |
| /languages/python_api.html#cudaq. | api.html#_CPPv4NO5cudaq10product_ |
| operators.spin.SpinOperator.copy) | opplERK10product_opI9HandlerTyE), |
|     -   [(cuda                    |     [\[16\]](api/languag          |
| q.operators.spin.SpinOperatorTerm | es/cpp_api.html#_CPPv4NO5cudaq10p |
|         attribute)](api/lan       | roduct_opplERK15scalar_operator), |
| guages/python_api.html#cudaq.oper |     [\[17\]](api/languages        |
| ators.spin.SpinOperatorTerm.copy) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [count (cudaq.Resources       | duct_opplERK6sum_opI9HandlerTyE), |
|                                   |     [\[18\]](api/languages/cpp_   |
|   attribute)](api/languages/pytho | api.html#_CPPv4NO5cudaq10product_ |
| n_api.html#cudaq.Resources.count) | opplERR10product_opI9HandlerTyE), |
|     -   [(cudaq.SampleResult      |     [\[19\]](api/languag          |
|         a                         | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ttribute)](api/languages/python_a | roduct_opplERR15scalar_operator), |
| pi.html#cudaq.SampleResult.count) |     [\[20\]](api/languages        |
| -   [count_controls               | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (cudaq.Resources              | duct_opplERR6sum_opI9HandlerTyE), |
|     attribu                       |     [                             |
| te)](api/languages/python_api.htm | \[21\]](api/languages/cpp_api.htm |
| l#cudaq.Resources.count_controls) | l#_CPPv4NO5cudaq10product_opplEv) |
| -   [count_instructions           | -   [cudaq::product_op::operator- |
|                                   |     (C++                          |
|   (cudaq.ptsbe.PTSBEExecutionData |     function)](api/langu          |
|     attribute)](api/languages/    | ages/cpp_api.html#_CPPv4I0EN5cuda |
| python_api.html#cudaq.ptsbe.PTSBE | q10product_opmiE6sum_opI1TERK15sc |
| ExecutionData.count_instructions) | alar_operatorRK10product_opI1TE), |
| -   [counts (cudaq.ObserveResult  |     [\[1\]](api/                  |
|     att                           | languages/cpp_api.html#_CPPv4I0EN |
| ribute)](api/languages/python_api | 5cudaq10product_opmiE6sum_opI1TER |
| .html#cudaq.ObserveResult.counts) | K15scalar_operatorRK6sum_opI1TE), |
| -   [csr_spmatrix (C++            |     [\[2\]](api/langu             |
|     type)](api/languages/c        | ages/cpp_api.html#_CPPv4I0EN5cuda |
| pp_api.html#_CPPv412csr_spmatrix) | q10product_opmiE6sum_opI1TERK15sc |
| -   cudaq                         | alar_operatorRR10product_opI1TE), |
|     -   [module](api/langua       |     [\[3\]](api/                  |
| ges/python_api.html#module-cudaq) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq (C++                   | 5cudaq10product_opmiE6sum_opI1TER |
|     type)](api/lan                | K15scalar_operatorRR6sum_opI1TE), |
| guages/cpp_api.html#_CPPv45cudaq) |     [\[4\]](api/langu             |
| -   [cudaq.apply_noise() (in      | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     module                        | q10product_opmiE6sum_opI1TERR15sc |
|     cudaq)](api/languages/python_ | alar_operatorRK10product_opI1TE), |
| api.html#cudaq.cudaq.apply_noise) |     [\[5\]](api/                  |
| -   cudaq.boson                   | languages/cpp_api.html#_CPPv4I0EN |
|     -   [module](api/languages/py | 5cudaq10product_opmiE6sum_opI1TER |
| thon_api.html#module-cudaq.boson) | R15scalar_operatorRK6sum_opI1TE), |
| -   cudaq.fermion                 |     [\[6\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|   -   [module](api/languages/pyth | q10product_opmiE6sum_opI1TERR15sc |
| on_api.html#module-cudaq.fermion) | alar_operatorRR10product_opI1TE), |
| -   cudaq.operators.custom        |     [\[7\]](api/                  |
|     -   [mo                       | languages/cpp_api.html#_CPPv4I0EN |
| dule](api/languages/python_api.ht | 5cudaq10product_opmiE6sum_opI1TER |
| ml#module-cudaq.operators.custom) | R15scalar_operatorRR6sum_opI1TE), |
| -   cudaq.spin                    |     [\[8\]](api/languages/cpp_a   |
|     -   [module](api/languages/p  | pi.html#_CPPv4NKR5cudaq10product_ |
| ython_api.html#module-cudaq.spin) | opmiERK10product_opI9HandlerTyE), |
| -   [cudaq::amplitude_damping     |     [\[9\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     cla                           | roduct_opmiERK15scalar_operator), |
| ss)](api/languages/cpp_api.html#_ |     [\[10\]](api/languages/       |
| CPPv4N5cudaq17amplitude_dampingE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -                                 | duct_opmiERK6sum_opI9HandlerTyE), |
| [cudaq::amplitude_damping_channel |     [\[11\]](api/languages/cpp_a  |
|     (C++                          | pi.html#_CPPv4NKR5cudaq10product_ |
|     class)](api                   | opmiERR10product_opI9HandlerTyE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[12\]](api/language         |
| udaq25amplitude_damping_channelE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::amplitud              | roduct_opmiERR15scalar_operator), |
| e_damping_channel::num_parameters |     [\[13\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     member)](api/languages/cpp_a  | duct_opmiERR6sum_opI9HandlerTyE), |
| pi.html#_CPPv4N5cudaq25amplitude_ |     [\[                           |
| damping_channel14num_parametersE) | 14\]](api/languages/cpp_api.html# |
| -   [cudaq::ampli                 | _CPPv4NKR5cudaq10product_opmiEv), |
| tude_damping_channel::num_targets |     [\[15\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     member)](api/languages/cp     | opmiERK10product_opI9HandlerTyE), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [\[16\]](api/languag          |
| de_damping_channel11num_targetsE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::AnalogRemoteRESTQPU   | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[17\]](api/languages        |
|     class                         | /cpp_api.html#_CPPv4NO5cudaq10pro |
| )](api/languages/cpp_api.html#_CP | duct_opmiERK6sum_opI9HandlerTyE), |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) |     [\[18\]](api/languages/cpp_   |
| -   [cudaq::apply_noise (C++      | api.html#_CPPv4NO5cudaq10product_ |
|     function)](api/               | opmiERR10product_opI9HandlerTyE), |
| languages/cpp_api.html#_CPPv4I0Dp |     [\[19\]](api/languag          |
| EN5cudaq11apply_noiseEvDpRR4Args) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::async_result (C++     | roduct_opmiERR15scalar_operator), |
|     c                             |     [\[20\]](api/languages        |
| lass)](api/languages/cpp_api.html | /cpp_api.html#_CPPv4NO5cudaq10pro |
| #_CPPv4I0EN5cudaq12async_resultE) | duct_opmiERR6sum_opI9HandlerTyE), |
| -   [cudaq::async_result::get     |     [                             |
|     (C++                          | \[21\]](api/languages/cpp_api.htm |
|     functi                        | l#_CPPv4NO5cudaq10product_opmiEv) |
| on)](api/languages/cpp_api.html#_ | -   [cudaq::product_op::operator/ |
| CPPv4N5cudaq12async_result3getEv) |     (C++                          |
| -   [cudaq::async_sample_result   |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     type                          | roduct_opdvERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/language          |
| Pv4N5cudaq19async_sample_resultE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::BaseRemoteRESTQPU     | roduct_opdvERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languag           |
|     cla                           | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ss)](api/languages/cpp_api.html#_ | roduct_opdvERK15scalar_operator), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [\[3\]](api/langua            |
| -   [cudaq::bit_flip_channel (C++ | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     cl                            | product_opdvERR15scalar_operator) |
| ass)](api/languages/cpp_api.html# | -                                 |
| _CPPv4N5cudaq16bit_flip_channelE) |    [cudaq::product_op::operator/= |
| -   [cudaq:                       |     (C++                          |
| :bit_flip_channel::num_parameters |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     member)](api/langua           | product_opdVERK15scalar_operator) |
| ges/cpp_api.html#_CPPv4N5cudaq16b | -   [cudaq::product_op::operator= |
| it_flip_channel14num_parametersE) |     (C++                          |
| -   [cud                          |     function)](api/l              |
| aq::bit_flip_channel::num_targets | anguages/cpp_api.html#_CPPv4I00EN |
|     (C++                          | 5cudaq10product_opaSER10product_o |
|     member)](api/lan              | pI9HandlerTyERK10product_opI1TE), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[1\]](api/languages/cpp     |
| 16bit_flip_channel11num_targetsE) | _api.html#_CPPv4N5cudaq10product_ |
| -   [cudaq::boson_handler (C++    | opaSERK10product_opI9HandlerTyE), |
|                                   |     [\[2\]](api/languages/cp      |
|  class)](api/languages/cpp_api.ht | p_api.html#_CPPv4N5cudaq10product |
| ml#_CPPv4N5cudaq13boson_handlerE) | _opaSERR10product_opI9HandlerTyE) |
| -   [cudaq::boson_op (C++         | -                                 |
|     type)](api/languages/cpp_     |    [cudaq::product_op::operator== |
| api.html#_CPPv4N5cudaq8boson_opE) |     (C++                          |
| -   [cudaq::boson_op_term (C++    |     function)](api/languages/cpp  |
|                                   | _api.html#_CPPv4NK5cudaq10product |
|   type)](api/languages/cpp_api.ht | _opeqERK10product_opI9HandlerTyE) |
| ml#_CPPv4N5cudaq13boson_op_termE) | -                                 |
| -   [cudaq::CodeGenConfig (C++    |  [cudaq::product_op::operator\[\] |
|                                   |     (C++                          |
| struct)](api/languages/cpp_api.ht |     function)](ap                 |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::commutation_relations | 5cudaq10product_opixENSt6size_tE) |
|     (C++                          | -                                 |
|     struct)]                      |    [cudaq::product_op::product_op |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq21commutation_relationsE) |     f                             |
| -   [cudaq::complex (C++          | unction)](api/languages/cpp_api.h |
|     type)](api/languages/cpp      | tml#_CPPv4I00EN5cudaq10product_op |
| _api.html#_CPPv4N5cudaq7complexE) | 10product_opERK10product_opI1TE), |
| -   [cudaq::complex_matrix (C++   |     [\[1\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
| class)](api/languages/cpp_api.htm | 4I00EN5cudaq10product_op10product |
| l#_CPPv4N5cudaq14complex_matrixE) | _opERK10product_opI1TERKN14matrix |
| -                                 | _handler20commutation_behaviorE), |
|   [cudaq::complex_matrix::adjoint |                                   |
|     (C++                          |   [\[2\]](api/languages/cpp_api.h |
|     function)](a                  | tml#_CPPv4N5cudaq10product_op10pr |
| pi/languages/cpp_api.html#_CPPv4N | oduct_opENSt6size_tENSt6size_tE), |
| 5cudaq14complex_matrix7adjointEv) |     [\[3\]](api/languages/cp      |
| -   [cudaq::                      | p_api.html#_CPPv4N5cudaq10product |
| complex_matrix::diagonal_elements | _op10product_opENSt7complexIdEE), |
|     (C++                          |     [\[4\]](api/l                 |
|     function)](api/languages      | anguages/cpp_api.html#_CPPv4N5cud |
| /cpp_api.html#_CPPv4NK5cudaq14com | aq10product_op10product_opERK10pr |
| plex_matrix17diagonal_elementsEi) | oduct_opI9HandlerTyENSt6size_tE), |
| -   [cudaq::complex_matrix::dump  |     [\[5\]](api/l                 |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     function)](api/language       | aq10product_op10product_opERR10pr |
| s/cpp_api.html#_CPPv4NK5cudaq14co | oduct_opI9HandlerTyENSt6size_tE), |
| mplex_matrix4dumpERNSt7ostreamE), |     [\[6\]](api/languages         |
|     [\[1\]]                       | /cpp_api.html#_CPPv4N5cudaq10prod |
| (api/languages/cpp_api.html#_CPPv | uct_op10product_opERR9HandlerTy), |
| 4NK5cudaq14complex_matrix4dumpEv) |     [\[7\]](ap                    |
| -   [c                            | i/languages/cpp_api.html#_CPPv4N5 |
| udaq::complex_matrix::eigenvalues | cudaq10product_op10product_opEd), |
|     (C++                          |     [\[8\]](a                     |
|     function)](api/lan            | pi/languages/cpp_api.html#_CPPv4N |
| guages/cpp_api.html#_CPPv4NK5cuda | 5cudaq10product_op10product_opEv) |
| q14complex_matrix11eigenvaluesEv) | -   [cuda                         |
| -   [cu                           | q::product_op::to_diagonal_matrix |
| daq::complex_matrix::eigenvectors |     (C++                          |
|     (C++                          |     function)](api/               |
|     function)](api/lang           | languages/cpp_api.html#_CPPv4NK5c |
| uages/cpp_api.html#_CPPv4NK5cudaq | udaq10product_op18to_diagonal_mat |
| 14complex_matrix12eigenvectorsEv) | rixENSt13unordered_mapINSt6size_t |
| -   [c                            | ENSt7int64_tEEERKNSt13unordered_m |
| udaq::complex_matrix::exponential | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cudaq::product_op::to_matrix |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     funct                         |
| q14complex_matrix11exponentialEv) | ion)](api/languages/cpp_api.html# |
| -                                 | _CPPv4NK5cudaq10product_op9to_mat |
|  [cudaq::complex_matrix::identity | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/languages      | apINSt6stringENSt7complexIdEEEEb) |
| /cpp_api.html#_CPPv4N5cudaq14comp | -   [cu                           |
| lex_matrix8identityEKNSt6size_tE) | daq::product_op::to_sparse_matrix |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::kronecker |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     function)](api/lang           | 5cudaq10product_op16to_sparse_mat |
| uages/cpp_api.html#_CPPv4I00EN5cu | rixENSt13unordered_mapINSt6size_t |
| daq14complex_matrix9kroneckerE14c | ENSt7int64_tEEERKNSt13unordered_m |
| omplex_matrix8Iterable8Iterable), | apINSt6stringENSt7complexIdEEEEb) |
|     [\[1\]](api/l                 | -   [cudaq::product_op::to_string |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq14complex_matrix9kroneckerERK14 |     function)](                   |
| complex_matrixRK14complex_matrix) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::c                     | NK5cudaq10product_op9to_stringEv) |
| omplex_matrix::minimal_eigenvalue | -                                 |
|     (C++                          |  [cudaq::product_op::\~product_op |
|     function)](api/languages/     |     (C++                          |
| cpp_api.html#_CPPv4NK5cudaq14comp |     fu                            |
| lex_matrix18minimal_eigenvalueEv) | nction)](api/languages/cpp_api.ht |
| -   [                             | ml#_CPPv4N5cudaq10product_opD0Ev) |
| cudaq::complex_matrix::operator() | -   [cudaq::ptsbe (C++            |
|     (C++                          |     type)](api/languages/c        |
|     function)](api/languages/cpp  | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| _api.html#_CPPv4N5cudaq14complex_ | -   [cudaq::p                     |
| matrixclENSt6size_tENSt6size_tE), | tsbe::ConditionalSamplingStrategy |
|     [\[1\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4NK5cudaq14complex |     class)](api/languag           |
| _matrixclENSt6size_tENSt6size_tE) | es/cpp_api.html#_CPPv4N5cudaq5pts |
| -   [                             | be27ConditionalSamplingStrategyE) |
| cudaq::complex_matrix::operator\* | -   [cudaq::ptsbe::C              |
|     (C++                          | onditionalSamplingStrategy::clone |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14c |                                   |
| omplex_matrixmlEN14complex_matrix |    function)](api/languages/cpp_a |
| 10value_typeERK14complex_matrix), | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
|     [\[1\]                        | ditionalSamplingStrategy5cloneEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cuda                         |
| v4N5cudaq14complex_matrixmlERK14c | q::ptsbe::ConditionalSamplingStra |
| omplex_matrixRK14complex_matrix), | tegy::ConditionalSamplingStrategy |
|                                   |     (C++                          |
|  [\[2\]](api/languages/cpp_api.ht |     function)](api/lang           |
| ml#_CPPv4N5cudaq14complex_matrixm | uages/cpp_api.html#_CPPv4N5cudaq5 |
| lERK14complex_matrixRKNSt6vectorI | ptsbe27ConditionalSamplingStrateg |
| N14complex_matrix10value_typeEEE) | y27ConditionalSamplingStrategyE19 |
| -                                 | TrajectoryPredicateNSt8uint64_tE) |
| [cudaq::complex_matrix::operator+ | -                                 |
|     (C++                          |   [cudaq::ptsbe::ConditionalSampl |
|     function                      | ingStrategy::generateTrajectories |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq14complex_matrixplERK14 |     function)](api/language       |
| complex_matrixRK14complex_matrix) | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| -                                 | be27ConditionalSamplingStrategy20 |
| [cudaq::complex_matrix::operator- | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     function                      | -   [cudaq::ptsbe::               |
| )](api/languages/cpp_api.html#_CP | ConditionalSamplingStrategy::name |
| Pv4N5cudaq14complex_matrixmiERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     function)](api/languages/cpp_ |
| -   [cu                           | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| daq::complex_matrix::operator\[\] | nditionalSamplingStrategy4nameEv) |
|     (C++                          | -   [cudaq:                       |
|                                   | :ptsbe::ConditionalSamplingStrate |
|  function)](api/languages/cpp_api | gy::\~ConditionalSamplingStrategy |
| .html#_CPPv4N5cudaq14complex_matr |     (C++                          |
| ixixERKNSt6vectorINSt6size_tEEE), |     function)](api/languages/     |
|     [\[1\]](api/languages/cpp_api | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| .html#_CPPv4NK5cudaq14complex_mat | 7ConditionalSamplingStrategyD0Ev) |
| rixixERKNSt6vectorINSt6size_tEEE) | -                                 |
| -   [cudaq::complex_matrix::power | [cudaq::ptsbe::detail::NoisePoint |
|     (C++                          |     (C++                          |
|     function)]                    |     struct)](a                    |
| (api/languages/cpp_api.html#_CPPv | pi/languages/cpp_api.html#_CPPv4N |
| 4N5cudaq14complex_matrix5powerEi) | 5cudaq5ptsbe6detail10NoisePointE) |
| -                                 | -   [cudaq::p                     |
|  [cudaq::complex_matrix::set_zero | tsbe::detail::NoisePoint::channel |
|     (C++                          |     (C++                          |
|     function)](ap                 |     member)](api/langu            |
| i/languages/cpp_api.html#_CPPv4N5 | ages/cpp_api.html#_CPPv4N5cudaq5p |
| cudaq14complex_matrix8set_zeroEv) | tsbe6detail10NoisePoint7channelE) |
| -                                 | -   [cudaq::ptsbe::det            |
| [cudaq::complex_matrix::to_string | ail::NoisePoint::circuit_location |
|     (C++                          |     (C++                          |
|     function)](api/               |     member)](api/languages/cpp_a  |
| languages/cpp_api.html#_CPPv4NK5c | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| udaq14complex_matrix9to_stringEv) | l10NoisePoint16circuit_locationE) |
| -   [                             | -   [cudaq::p                     |
| cudaq::complex_matrix::value_type | tsbe::detail::NoisePoint::op_name |
|     (C++                          |     (C++                          |
|     type)](api/                   |     member)](api/langu            |
| languages/cpp_api.html#_CPPv4N5cu | ages/cpp_api.html#_CPPv4N5cudaq5p |
| daq14complex_matrix10value_typeE) | tsbe6detail10NoisePoint7op_nameE) |
| -   [cudaq::contrib (C++          | -   [cudaq::                      |
|     type)](api/languages/cpp      | ptsbe::detail::NoisePoint::qubits |
| _api.html#_CPPv4N5cudaq7contribE) |     (C++                          |
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
| -   [cudaq::CusvState (C++        |     function)](api/languages/cpp_ |
|                                   | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
|    class)](api/languages/cpp_api. | haustiveSamplingStrategy5cloneEv) |
| html#_CPPv4I0EN5cudaq9CusvStateE) | -   [cu                           |
| -   [cudaq::depolarization1 (C++  | daq::ptsbe::ExhaustiveSamplingStr |
|     c                             | ategy::ExhaustiveSamplingStrategy |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15depolarization1E) |     function)](api/la             |
| -   [cudaq::depolarization2 (C++  | nguages/cpp_api.html#_CPPv4N5cuda |
|     c                             | q5ptsbe26ExhaustiveSamplingStrate |
| lass)](api/languages/cpp_api.html | gy26ExhaustiveSamplingStrategyEv) |
| #_CPPv4N5cudaq15depolarization2E) | -                                 |
| -   [cudaq:                       |    [cudaq::ptsbe::ExhaustiveSampl |
| :depolarization2::depolarization2 | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/languag        |
| p_api.html#_CPPv4N5cudaq15depolar | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| ization215depolarization2EK4real) | sbe26ExhaustiveSamplingStrategy20 |
| -   [cudaq                        | generateTrajectoriesENSt4spanIKN6 |
| ::depolarization2::num_parameters | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe:                |
|     member)](api/langu            | :ExhaustiveSamplingStrategy::name |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| depolarization214num_parametersE) |     function)](api/languages/cpp  |
| -   [cu                           | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| daq::depolarization2::num_targets | xhaustiveSamplingStrategy4nameEv) |
|     (C++                          | -   [cuda                         |
|     member)](api/la               | q::ptsbe::ExhaustiveSamplingStrat |
| nguages/cpp_api.html#_CPPv4N5cuda | egy::\~ExhaustiveSamplingStrategy |
| q15depolarization211num_targetsE) |     (C++                          |
| -                                 |     function)](api/languages      |
|    [cudaq::depolarization_channel | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     (C++                          | 26ExhaustiveSamplingStrategyD0Ev) |
|     class)](                      | -   [cuda                         |
| api/languages/cpp_api.html#_CPPv4 | q::ptsbe::OrderedSamplingStrategy |
| N5cudaq22depolarization_channelE) |     (C++                          |
| -   [cudaq::depol                 |     class)](api/lan               |
| arization_channel::num_parameters | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 5ptsbe23OrderedSamplingStrategyE) |
|     member)](api/languages/cp     | -   [cudaq::ptsb                  |
| p_api.html#_CPPv4N5cudaq22depolar | e::OrderedSamplingStrategy::clone |
| ization_channel14num_parametersE) |     (C++                          |
| -   [cudaq::de                    |     function)](api/languages/c    |
| polarization_channel::num_targets | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
|     (C++                          | 3OrderedSamplingStrategy5cloneEv) |
|     member)](api/languages        | -   [cudaq::ptsbe::OrderedSampl   |
| /cpp_api.html#_CPPv4N5cudaq22depo | ingStrategy::generateTrajectories |
| larization_channel11num_targetsE) |     (C++                          |
| -   [cudaq::details (C++          |     function)](api/lang           |
|     type)](api/languages/cpp      | uages/cpp_api.html#_CPPv4NK5cudaq |
| _api.html#_CPPv4N5cudaq7detailsE) | 5ptsbe23OrderedSamplingStrategy20 |
| -   [cudaq::details::future (C++  | generateTrajectoriesENSt4spanIKN6 |
|                                   | detail10NoisePointEEENSt6size_tE) |
|  class)](api/languages/cpp_api.ht | -   [cudaq::pts                   |
| ml#_CPPv4N5cudaq7details6futureE) | be::OrderedSamplingStrategy::name |
| -                                 |     (C++                          |
|   [cudaq::details::future::future |     function)](api/languages/     |
|     (C++                          | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     functio                       | 23OrderedSamplingStrategy4nameEv) |
| n)](api/languages/cpp_api.html#_C | -                                 |
| PPv4N5cudaq7details6future6future |    [cudaq::ptsbe::OrderedSampling |
| ERNSt6vectorI3JobEERNSt6stringERN | Strategy::OrderedSamplingStrategy |
| St3mapINSt6stringENSt6stringEEE), |     (C++                          |
|     [\[1\]](api/lang              |     function)](                   |
| uages/cpp_api.html#_CPPv4N5cudaq7 | api/languages/cpp_api.html#_CPPv4 |
| details6future6futureERR6future), | N5cudaq5ptsbe23OrderedSamplingStr |
|     [\[2\]]                       | ategy23OrderedSamplingStrategyEv) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq7details6future6futureEv) |  [cudaq::ptsbe::OrderedSamplingSt |
| -   [cu                           | rategy::\~OrderedSamplingStrategy |
| daq::details::kernel_builder_base |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     class)](api/l                 | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| anguages/cpp_api.html#_CPPv4N5cud | sbe23OrderedSamplingStrategyD0Ev) |
| aq7details19kernel_builder_baseE) | -   [cudaq::pts                   |
| -   [cudaq::details::             | be::ProbabilisticSamplingStrategy |
| kernel_builder_base::operator\<\< |     (C++                          |
|     (C++                          |     class)](api/languages         |
|     function)](api/langua         | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| ges/cpp_api.html#_CPPv4N5cudaq7de | 29ProbabilisticSamplingStrategyE) |
| tails19kernel_builder_baselsERNSt | -   [cudaq::ptsbe::Pro            |
| 7ostreamERK19kernel_builder_base) | babilisticSamplingStrategy::clone |
| -   [                             |     (C++                          |
| cudaq::details::KernelBuilderType |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     class)](api                   | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| /languages/cpp_api.html#_CPPv4N5c | bilisticSamplingStrategy5cloneEv) |
| udaq7details17KernelBuilderTypeE) | -                                 |
| -   [cudaq::d                     | [cudaq::ptsbe::ProbabilisticSampl |
| etails::KernelBuilderType::create | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api/languages/     |
| ](api/languages/cpp_api.html#_CPP | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| v4N5cudaq7details17KernelBuilderT | 29ProbabilisticSamplingStrategy20 |
| ype6createEPN4mlir11MLIRContextE) | generateTrajectoriesENSt4spanIKN6 |
| -   [cudaq::details::Ker          | detail10NoisePointEEENSt6size_tE) |
| nelBuilderType::KernelBuilderType | -   [cudaq::ptsbe::Pr             |
|     (C++                          | obabilisticSamplingStrategy::name |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |                                   |
| details17KernelBuilderType17Kerne |   function)](api/languages/cpp_ap |
| lBuilderTypeERRNSt8functionIFN4ml | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| ir4TypeEPN4mlir11MLIRContextEEEE) | abilisticSamplingStrategy4nameEv) |
| -   [cudaq::diag_matrix_callback  | -   [cudaq::p                     |
|     (C++                          | tsbe::ProbabilisticSamplingStrate |
|     class)                        | gy::ProbabilisticSamplingStrategy |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq20diag_matrix_callbackE) |     function)]                    |
| -   [cudaq::dyn (C++              | (api/languages/cpp_api.html#_CPPv |
|     member)](api/languages        | 4N5cudaq5ptsbe29ProbabilisticSamp |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | lingStrategy29ProbabilisticSampli |
| -   [cudaq::ExecutionContext (C++ | ngStrategyENSt8optionalINSt8uint6 |
|     cl                            | 4_tEEENSt8optionalINSt6size_tEEE) |
| ass)](api/languages/cpp_api.html# | -   [cudaq::pts                   |
| _CPPv4N5cudaq16ExecutionContextE) | be::ProbabilisticSamplingStrategy |
| -   [c                            | ::\~ProbabilisticSamplingStrategy |
| udaq::ExecutionContext::asyncExec |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api/                 | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| languages/cpp_api.html#_CPPv4N5cu | robabilisticSamplingStrategyD0Ev) |
| daq16ExecutionContext9asyncExecE) | -                                 |
| -   [cud                          | [cudaq::ptsbe::PTSBEExecutionData |
| aq::ExecutionContext::asyncResult |     (C++                          |
|     (C++                          |     struct)](ap                   |
|     member)](api/lan              | i/languages/cpp_api.html#_CPPv4N5 |
| guages/cpp_api.html#_CPPv4N5cudaq | cudaq5ptsbe18PTSBEExecutionDataE) |
| 16ExecutionContext11asyncResultE) | -   [cudaq::ptsbe::PTSBE          |
| -   [cudaq:                       | ExecutionData::count_instructions |
| :ExecutionContext::batchIteration |     (C++                          |
|     (C++                          |     function)](api/l              |
|     member)](api/langua           | anguages/cpp_api.html#_CPPv4NK5cu |
| ges/cpp_api.html#_CPPv4N5cudaq16E | daq5ptsbe18PTSBEExecutionData18co |
| xecutionContext14batchIterationE) | unt_instructionsE20TraceInstructi |
| -   [cudaq::E                     | onTypeNSt8optionalINSt6stringEEE) |
| xecutionContext::canHandleObserve | -   [cudaq::ptsbe::P              |
|     (C++                          | TSBEExecutionData::get_trajectory |
|     member)](api/language         |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     function                      |
| cutionContext16canHandleObserveE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::E                     | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| xecutionContext::ExecutionContext | Data14get_trajectoryENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe:                |
|     func                          | :PTSBEExecutionData::instructions |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq16ExecutionContext1 |     member)](api/languages/cp     |
| 6ExecutionContextERKNSt6stringE), | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     [\[1\]](api/languages/        | TSBEExecutionData12instructionsE) |
| cpp_api.html#_CPPv4N5cudaq16Execu | -   [cudaq::ptsbe:                |
| tionContext16ExecutionContextERKN | :PTSBEExecutionData::trajectories |
| St6stringENSt6size_tENSt6size_tE) |     (C++                          |
| -   [cudaq::E                     |     member)](api/languages/cp     |
| xecutionContext::expectationValue | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     (C++                          | TSBEExecutionData12trajectoriesE) |
|     member)](api/language         | -   [cudaq::ptsbe::PTSBEOptions   |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     (C++                          |
| cutionContext16expectationValueE) |     struc                         |
| -   [cudaq::Execu                 | t)](api/languages/cpp_api.html#_C |
| tionContext::explicitMeasurements | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
|     (C++                          | -   [cudaq::ptsbe::PTSB           |
|     member)](api/languages/cp     | EOptions::include_sequential_data |
| p_api.html#_CPPv4N5cudaq16Executi |     (C++                          |
| onContext20explicitMeasurementsE) |                                   |
| -   [cuda                         |    member)](api/languages/cpp_api |
| q::ExecutionContext::futureResult | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
|     (C++                          | ptions23include_sequential_dataE) |
|     member)](api/lang             | -   [cudaq::ptsb                  |
| uages/cpp_api.html#_CPPv4N5cudaq1 | e::PTSBEOptions::max_trajectories |
| 6ExecutionContext12futureResultE) |     (C++                          |
| -   [cudaq::ExecutionContext      |     member)](api/languages/       |
| ::hasConditionalsOnMeasureResults | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
|     (C++                          | 2PTSBEOptions16max_trajectoriesE) |
|     mem                           | -   [cudaq::ptsbe::PT             |
| ber)](api/languages/cpp_api.html# | SBEOptions::return_execution_data |
| _CPPv4N5cudaq16ExecutionContext31 |     (C++                          |
| hasConditionalsOnMeasureResultsE) |     member)](api/languages/cpp_a  |
| -   [cudaq::Executi               | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| onContext::invocationResultBuffer | EOptions21return_execution_dataE) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/languages/cpp_   | be::PTSBEOptions::shot_allocation |
| api.html#_CPPv4N5cudaq16Execution |     (C++                          |
| Context22invocationResultBufferE) |     member)](api/languages        |
| -   [cu                           | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| daq::ExecutionContext::kernelName | 12PTSBEOptions15shot_allocationE) |
|     (C++                          | -   [cud                          |
|     member)](api/la               | aq::ptsbe::PTSBEOptions::strategy |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10kernelNameE) |     member)](api/l                |
| -   [cud                          | anguages/cpp_api.html#_CPPv4N5cud |
| aq::ExecutionContext::kernelTrace | aq5ptsbe12PTSBEOptions8strategyE) |
|     (C++                          | -   [cudaq::ptsbe::PTSBETrace     |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     t                             |
| 16ExecutionContext11kernelTraceE) | ype)](api/languages/cpp_api.html# |
| -   [cudaq:                       | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| :ExecutionContext::msm_dimensions | -   [                             |
|     (C++                          | cudaq::ptsbe::PTSSamplingStrategy |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     class)](api                   |
| xecutionContext14msm_dimensionsE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::                      | udaq5ptsbe19PTSSamplingStrategyE) |
| ExecutionContext::msm_prob_err_id | -   [cudaq::                      |
|     (C++                          | ptsbe::PTSSamplingStrategy::clone |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api/languag        |
| ecutionContext15msm_prob_err_idE) | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| -   [cudaq::Ex                    | sbe19PTSSamplingStrategy5cloneEv) |
| ecutionContext::msm_probabilities | -   [cudaq::ptsbe::PTSSampl       |
|     (C++                          | ingStrategy::generateTrajectories |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq16Exec |     function)](api/               |
| utionContext17msm_probabilitiesE) | languages/cpp_api.html#_CPPv4NK5c |
| -                                 | udaq5ptsbe19PTSSamplingStrategy20 |
|    [cudaq::ExecutionContext::name | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     member)]                      | -   [cudaq:                       |
| (api/languages/cpp_api.html#_CPPv | :ptsbe::PTSSamplingStrategy::name |
| 4N5cudaq16ExecutionContext4nameE) |     (C++                          |
| -   [cu                           |     function)](api/langua         |
| daq::ExecutionContext::noiseModel | ges/cpp_api.html#_CPPv4NK5cudaq5p |
|     (C++                          | tsbe19PTSSamplingStrategy4nameEv) |
|     member)](api/la               | -   [cudaq::ptsbe::PTSSampli      |
| nguages/cpp_api.html#_CPPv4N5cuda | ngStrategy::\~PTSSamplingStrategy |
| q16ExecutionContext10noiseModelE) |     (C++                          |
| -   [cudaq::Exe                   |     function)](api/la             |
| cutionContext::numberTrajectories | nguages/cpp_api.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe19PTSSamplingStrategyD0Ev) |
|     member)](api/languages/       | -   [cudaq::ptsbe::sample (C++    |
| cpp_api.html#_CPPv4N5cudaq16Execu |                                   |
| tionContext18numberTrajectoriesE) |  function)](api/languages/cpp_api |
| -   [c                            | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| udaq::ExecutionContext::optResult | mpleE13sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](api/                 |     [\[1\]](api                   |
| languages/cpp_api.html#_CPPv4N5cu | /languages/cpp_api.html#_CPPv4I0D |
| daq16ExecutionContext9optResultE) | pEN5cudaq5ptsbe6sampleE13sample_r |
| -                                 | esultRKN5cudaq11noise_modelENSt6s |
|   [cudaq::ExecutionContext::qpuId | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_async   |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](a                  |
| N5cudaq16ExecutionContext5qpuIdE) | pi/languages/cpp_api.html#_CPPv4I |
| -   [cudaq                        | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| ::ExecutionContext::registerNames | 9async_sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](api/langu            |     [\[1\]](api/languages/cp      |
| ages/cpp_api.html#_CPPv4N5cudaq16 | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| ExecutionContext13registerNamesE) | be12sample_asyncE19async_sample_r |
| -   [cu                           | esultRKN5cudaq11noise_modelENSt6s |
| daq::ExecutionContext::reorderIdx | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_options |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     struct)                       |
| q16ExecutionContext10reorderIdxE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq5ptsbe14sample_optionsE) |
|  [cudaq::ExecutionContext::result | -   [cudaq::ptsbe::sample_result  |
|     (C++                          |     (C++                          |
|     member)](a                    |     class                         |
| pi/languages/cpp_api.html#_CPPv4N | )](api/languages/cpp_api.html#_CP |
| 5cudaq16ExecutionContext6resultE) | Pv4N5cudaq5ptsbe13sample_resultE) |
| -                                 | -   [cudaq::pts                   |
|   [cudaq::ExecutionContext::shots | be::sample_result::execution_data |
|     (C++                          |     (C++                          |
|     member)](                     |     function)](api/languages/c    |
| api/languages/cpp_api.html#_CPPv4 | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| N5cudaq16ExecutionContext5shotsE) | 3sample_result14execution_dataEv) |
| -   [cudaq::                      | -   [cudaq::ptsbe::               |
| ExecutionContext::simulationState | sample_result::has_execution_data |
|     (C++                          |     (C++                          |
|     member)](api/languag          |                                   |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |    function)](api/languages/cpp_a |
| ecutionContext15simulationStateE) | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| -                                 | ple_result18has_execution_dataEv) |
|    [cudaq::ExecutionContext::spin | -   [cudaq::pt                    |
|     (C++                          | sbe::sample_result::sample_result |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/l              |
| 4N5cudaq16ExecutionContext4spinE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::                      | aq5ptsbe13sample_result13sample_r |
| ExecutionContext::totalIterations | esultERRN5cudaq13sample_resultE), |
|     (C++                          |                                   |
|     member)](api/languag          |  [\[1\]](api/languages/cpp_api.ht |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| ecutionContext15totalIterationsE) | sult13sample_resultERRN5cudaq13sa |
| -   [cudaq::Executio              | mple_resultE18PTSBEExecutionData) |
| nContext::warnedNamedMeasurements | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::set_execution_data |
|     member)](api/languages/cpp_a  |     (C++                          |
| pi.html#_CPPv4N5cudaq16ExecutionC |     function)](api/               |
| ontext23warnedNamedMeasurementsE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::ExecutionResult (C++  | daq5ptsbe13sample_result18set_exe |
|     st                            | cution_dataE18PTSBEExecutionData) |
| ruct)](api/languages/cpp_api.html | -   [cud                          |
| #_CPPv4N5cudaq15ExecutionResultE) | aq::ptsbe::ShotAllocationStrategy |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::appendResult |     struct)](using                |
|     (C++                          | /examples/ptsbe.html#_CPPv4N5cuda |
|     functio                       | q5ptsbe22ShotAllocationStrategyE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::ptsbe::ShotAllocatio  |
| PPv4N5cudaq15ExecutionResult12app | nStrategy::ShotAllocationStrategy |
| endResultENSt6stringENSt6size_tE) |     (C++                          |
| -   [cu                           |     function)                     |
| daq::ExecutionResult::deserialize | ](using/examples/ptsbe.html#_CPPv |
|     (C++                          | 4N5cudaq5ptsbe22ShotAllocationStr |
|     function)                     | ategy22ShotAllocationStrategyE4Ty |
| ](api/languages/cpp_api.html#_CPP | pedNSt8optionalINSt8uint64_tEEE), |
| v4N5cudaq15ExecutionResult11deser |     [\[1\                         |
| ializeERNSt6vectorINSt6size_tEEE) | ]](using/examples/ptsbe.html#_CPP |
| -   [cudaq:                       | v4N5cudaq5ptsbe22ShotAllocationSt |
| :ExecutionResult::ExecutionResult | rategy22ShotAllocationStrategyEv) |
|     (C++                          | -   [cudaq::pt                    |
|     functio                       | sbe::ShotAllocationStrategy::Type |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq15ExecutionResult15Exe |     enum)](using/exam             |
| cutionResultE16CountsDictionary), | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|     [\[1\]](api/lan               | be22ShotAllocationStrategy4TypeE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::ptsbe::ShotAllocatio  |
| 15ExecutionResult15ExecutionResul | nStrategy::Type::HIGH_WEIGHT_BIAS |
| tE16CountsDictionaryNSt6stringE), |     (C++                          |
|     [\[2\                         |     enumerat                      |
| ]](api/languages/cpp_api.html#_CP | or)](using/examples/ptsbe.html#_C |
| Pv4N5cudaq15ExecutionResult15Exec | PPv4N5cudaq5ptsbe22ShotAllocation |
| utionResultE16CountsDictionaryd), | Strategy4Type16HIGH_WEIGHT_BIASE) |
|                                   | -   [cudaq::ptsbe::ShotAllocati   |
|    [\[3\]](api/languages/cpp_api. | onStrategy::Type::LOW_WEIGHT_BIAS |
| html#_CPPv4N5cudaq15ExecutionResu |     (C++                          |
| lt15ExecutionResultENSt6stringE), |     enumera                       |
|     [\[4\                         | tor)](using/examples/ptsbe.html#_ |
| ]](api/languages/cpp_api.html#_CP | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| Pv4N5cudaq15ExecutionResult15Exec | nStrategy4Type15LOW_WEIGHT_BIASE) |
| utionResultERK15ExecutionResult), | -   [cudaq::ptsbe::ShotAlloc      |
|     [\[5\]](api/language          | ationStrategy::Type::PROPORTIONAL |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     (C++                          |
| cutionResult15ExecutionResultEd), |     enum                          |
|     [\[6\]](api/languag           | erator)](using/examples/ptsbe.htm |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| ecutionResult15ExecutionResultEv) | tionStrategy4Type12PROPORTIONALE) |
| -   [                             | -   [cudaq::ptsbe::Shot           |
| cudaq::ExecutionResult::operator= | AllocationStrategy::Type::UNIFORM |
|     (C++                          |     (C++                          |
|     function)](api/languages/     |                                   |
| cpp_api.html#_CPPv4N5cudaq15Execu |   enumerator)](using/examples/pts |
| tionResultaSERK15ExecutionResult) | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| -   [c                            | AllocationStrategy4Type7UNIFORME) |
| udaq::ExecutionResult::operator== | -                                 |
|     (C++                          |   [cudaq::ptsbe::TraceInstruction |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4NK5cudaq15Execu |     struct)](                     |
| tionResulteqERK15ExecutionResult) | api/languages/cpp_api.html#_CPPv4 |
| -   [cud                          | N5cudaq5ptsbe16TraceInstructionE) |
| aq::ExecutionResult::registerName | -   [cudaq:                       |
|     (C++                          | :ptsbe::TraceInstruction::channel |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     member)](api/lang             |
| 15ExecutionResult12registerNameE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -   [cudaq                        | ptsbe16TraceInstruction7channelE) |
| ::ExecutionResult::sequentialData | -   [cudaq::                      |
|     (C++                          | ptsbe::TraceInstruction::controls |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     member)](api/langu            |
| ExecutionResult14sequentialDataE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [                             | tsbe16TraceInstruction8controlsE) |
| cudaq::ExecutionResult::serialize | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::name |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4NK5cu |     member)](api/l                |
| daq15ExecutionResult9serializeEv) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::fermion_handler (C++  | aq5ptsbe16TraceInstruction4nameE) |
|     c                             | -   [cudaq                        |
| lass)](api/languages/cpp_api.html | ::ptsbe::TraceInstruction::params |
| #_CPPv4N5cudaq15fermion_handlerE) |     (C++                          |
| -   [cudaq::fermion_op (C++       |     member)](api/lan              |
|     type)](api/languages/cpp_api  | guages/cpp_api.html#_CPPv4N5cudaq |
| .html#_CPPv4N5cudaq10fermion_opE) | 5ptsbe16TraceInstruction6paramsE) |
| -   [cudaq::fermion_op_term (C++  | -   [cudaq:                       |
|                                   | :ptsbe::TraceInstruction::targets |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15fermion_op_termE) |     member)](api/lang             |
| -   [cudaq::FermioniqQPU (C++     | uages/cpp_api.html#_CPPv4N5cudaq5 |
|                                   | ptsbe16TraceInstruction7targetsE) |
|   class)](api/languages/cpp_api.h | -   [cudaq::ptsbe::T              |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | raceInstruction::TraceInstruction |
| -   [cudaq::get_state (C++        |     (C++                          |
|                                   |                                   |
|    function)](api/languages/cpp_a |   function)](api/languages/cpp_ap |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| ateEDaRR13QuantumKernelDpRR4Args) | Instruction16TraceInstructionE20T |
| -   [cudaq::gradient (C++         | raceInstructionTypeNSt6stringENSt |
|     class)](api/languages/cpp_    | 6vectorINSt6size_tEEENSt6vectorIN |
| api.html#_CPPv4N5cudaq8gradientE) | St6size_tEEENSt6vectorIdEENSt8opt |
| -   [cudaq::gradient::clone (C++  | ionalIN5cudaq13kraus_channelEEE), |
|     fun                           |     [\[1\]](api/languages/cpp_a   |
| ction)](api/languages/cpp_api.htm | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| l#_CPPv4N5cudaq8gradient5cloneEv) | eInstruction16TraceInstructionEv) |
| -   [cudaq::gradient::compute     | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::type |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     member)](api/l                |
| ient7computeERKNSt6vectorIdEERKNS | anguages/cpp_api.html#_CPPv4N5cud |
| t8functionIFdNSt6vectorIdEEEEEd), | aq5ptsbe16TraceInstruction4typeE) |
|     [\[1\]](ap                    | -   [c                            |
| i/languages/cpp_api.html#_CPPv4N5 | udaq::ptsbe::TraceInstructionType |
| cudaq8gradient7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |     enum)](api/                   |
| -   [cudaq::gradient::gradient    | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq5ptsbe20TraceInstructionTypeE) |
|     function)](api/lang           | -   [cudaq::                      |
| uages/cpp_api.html#_CPPv4I00EN5cu | ptsbe::TraceInstructionType::Gate |
| daq8gradient8gradientER7KernelT), |     (C++                          |
|                                   |     enumerator)](api/langu        |
|    [\[1\]](api/languages/cpp_api. | ages/cpp_api.html#_CPPv4N5cudaq5p |
| html#_CPPv4I00EN5cudaq8gradient8g | tsbe20TraceInstructionType4GateE) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::ptsbe::               |
|     [\[2\                         | TraceInstructionType::Measurement |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4I00EN5cudaq8gradient8gradientE |                                   |
| RR13QuantumKernelRR10ArgsMapper), |    enumerator)](api/languages/cpp |
|     [\[3                          | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| \]](api/languages/cpp_api.html#_C | aceInstructionType11MeasurementE) |
| PPv4N5cudaq8gradient8gradientERRN | -   [cudaq::p                     |
| St8functionIFvNSt6vectorIdEEEEE), | tsbe::TraceInstructionType::Noise |
|     [\[                           |     (C++                          |
| 4\]](api/languages/cpp_api.html#_ |     enumerator)](api/langua       |
| CPPv4N5cudaq8gradient8gradientEv) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq::gradient::setArgs     | sbe20TraceInstructionType5NoiseE) |
|     (C++                          | -   [                             |
|     fu                            | cudaq::ptsbe::TrajectoryPredicate |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     type)](api                    |
| tArgsEvR13QuantumKernelDpRR4Args) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gradient::setKernel   | udaq5ptsbe19TrajectoryPredicateE) |
|     (C++                          | -   [cudaq::QPU (C++              |
|     function)](api/languages/c    |     class)](api/languages         |
| pp_api.html#_CPPv4I0EN5cudaq8grad | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| ient9setKernelEvR13QuantumKernel) | -   [cudaq::QPU::beginExecution   |
| -   [cud                          |     (C++                          |
| aq::gradients::central_difference |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     class)](api/la                | Pv4N5cudaq3QPU14beginExecutionEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cuda                         |
| q9gradients18central_differenceE) | q::QPU::configureExecutionContext |
| -   [cudaq::gra                   |     (C++                          |
| dients::central_difference::clone |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/languages      | _CPPv4NK5cudaq3QPU25configureExec |
| /cpp_api.html#_CPPv4N5cudaq9gradi | utionContextER16ExecutionContext) |
| ents18central_difference5cloneEv) | -   [cudaq::QPU::endExecution     |
| -   [cudaq::gradi                 |     (C++                          |
| ents::central_difference::compute |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     function)](                   | CPPv4N5cudaq3QPU12endExecutionEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::QPU::enqueue (C++     |
| N5cudaq9gradients18central_differ |     function)](ap                 |
| ence7computeERKNSt6vectorIdEERKNS | i/languages/cpp_api.html#_CPPv4N5 |
| t8functionIFdNSt6vectorIdEEEEEd), | cudaq3QPU7enqueueER11QuantumTask) |
|                                   | -   [cud                          |
|   [\[1\]](api/languages/cpp_api.h | aq::QPU::finalizeExecutionContext |
| tml#_CPPv4N5cudaq9gradients18cent |     (C++                          |
| ral_difference7computeERKNSt6vect |     func                          |
| orIdEERNSt6vectorIdEERK7spin_opd) | tion)](api/languages/cpp_api.html |
| -   [cudaq::gradie                | #_CPPv4NK5cudaq3QPU24finalizeExec |
| nts::central_difference::gradient | utionContextER16ExecutionContext) |
|     (C++                          | -   [cudaq::QPU::getConnectivity  |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |     function)                     |
| PPv4I00EN5cudaq9gradients18centra | ](api/languages/cpp_api.html#_CPP |
| l_difference8gradientER7KernelT), | v4N5cudaq3QPU15getConnectivityEv) |
|     [\[1\]](api/langua            | -                                 |
| ges/cpp_api.html#_CPPv4I00EN5cuda | [cudaq::QPU::getExecutionThreadId |
| q9gradients18central_difference8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api/               |
|     [\[2\]](api/languages/cpp_    | languages/cpp_api.html#_CPPv4NK5c |
| api.html#_CPPv4I00EN5cudaq9gradie | udaq3QPU20getExecutionThreadIdEv) |
| nts18central_difference8gradientE | -   [cudaq::QPU::getNumQubits     |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3\]](api/languages/cpp     |     functi                        |
| _api.html#_CPPv4N5cudaq9gradients | on)](api/languages/cpp_api.html#_ |
| 18central_difference8gradientERRN | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [                             |
|     [\[4\]](api/languages/cp      | cudaq::QPU::getRemoteCapabilities |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18central_difference8gradientEv) |     function)](api/l              |
| -   [cud                          | anguages/cpp_api.html#_CPPv4NK5cu |
| aq::gradients::forward_difference | daq3QPU21getRemoteCapabilitiesEv) |
|     (C++                          | -   [cudaq::QPU::isEmulated (C++  |
|     class)](api/la                |     func                          |
| nguages/cpp_api.html#_CPPv4N5cuda | tion)](api/languages/cpp_api.html |
| q9gradients18forward_differenceE) | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| -   [cudaq::gra                   | -   [cudaq::QPU::isSimulator (C++ |
| dients::forward_difference::clone |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/languages      | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QPU::onRandomSeedSet  |
| ents18forward_difference5cloneEv) |     (C++                          |
| -   [cudaq::gradi                 |     function)](api/lang           |
| ents::forward_difference::compute | uages/cpp_api.html#_CPPv4N5cudaq3 |
|     (C++                          | QPU15onRandomSeedSetENSt6size_tE) |
|     function)](                   | -   [cudaq::QPU::QPU (C++         |
| api/languages/cpp_api.html#_CPPv4 |     functio                       |
| N5cudaq9gradients18forward_differ | n)](api/languages/cpp_api.html#_C |
| ence7computeERKNSt6vectorIdEERKNS | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| t8functionIFdNSt6vectorIdEEEEEd), |                                   |
|                                   |  [\[1\]](api/languages/cpp_api.ht |
|   [\[1\]](api/languages/cpp_api.h | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| tml#_CPPv4N5cudaq9gradients18forw |     [\[2\]](api/languages/cpp_    |
| ard_difference7computeERKNSt6vect | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QPU::setId (C++       |
| -   [cudaq::gradie                |     function                      |
| nts::forward_difference::gradient | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     functio                       | -   [cudaq::QPU::setShots (C++    |
| n)](api/languages/cpp_api.html#_C |     f                             |
| PPv4I00EN5cudaq9gradients18forwar | unction)](api/languages/cpp_api.h |
| d_difference8gradientER7KernelT), | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
|     [\[1\]](api/langua            | -   [cudaq::                      |
| ges/cpp_api.html#_CPPv4I00EN5cuda | QPU::supportsExplicitMeasurements |
| q9gradients18forward_difference8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api/languag        |
|     [\[2\]](api/languages/cpp_    | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| api.html#_CPPv4I00EN5cudaq9gradie | 28supportsExplicitMeasurementsEv) |
| nts18forward_difference8gradientE | -   [cudaq::QPU::\~QPU (C++       |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/languages/cp   |
|     [\[3\]](api/languages/cpp     | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::QPUState (C++         |
| 18forward_difference8gradientERRN |     class)](api/languages/cpp_    |
| St8functionIFvNSt6vectorIdEEEEE), | api.html#_CPPv4N5cudaq8QPUStateE) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qreg (C++             |
| p_api.html#_CPPv4N5cudaq9gradient |     class)](api/lan               |
| s18forward_difference8gradientEv) | guages/cpp_api.html#_CPPv4I_NSt6s |
| -   [                             | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| cudaq::gradients::parameter_shift | -   [cudaq::qreg::back (C++       |
|     (C++                          |     function)                     |
|     class)](api                   | ](api/languages/cpp_api.html#_CPP |
| /languages/cpp_api.html#_CPPv4N5c | v4N5cudaq4qreg4backENSt6size_tE), |
| udaq9gradients15parameter_shiftE) |     [\[1\]](api/languages/cpp_ap  |
| -   [cudaq::                      | i.html#_CPPv4N5cudaq4qreg4backEv) |
| gradients::parameter_shift::clone | -   [cudaq::qreg::begin (C++      |
|     (C++                          |                                   |
|     function)](api/langua         |  function)](api/languages/cpp_api |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | .html#_CPPv4N5cudaq4qreg5beginEv) |
| adients15parameter_shift5cloneEv) | -   [cudaq::qreg::clear (C++      |
| -   [cudaq::gr                    |                                   |
| adients::parameter_shift::compute |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     function                      | -   [cudaq::qreg::front (C++      |
| )](api/languages/cpp_api.html#_CP |     function)]                    |
| Pv4N5cudaq9gradients15parameter_s | (api/languages/cpp_api.html#_CPPv |
| hift7computeERKNSt6vectorIdEERKNS | 4N5cudaq4qreg5frontENSt6size_tE), |
| t8functionIFdNSt6vectorIdEEEEEd), |     [\[1\]](api/languages/cpp_api |
|     [\[1\]](api/languages/cpp_ap  | .html#_CPPv4N5cudaq4qreg5frontEv) |
| i.html#_CPPv4N5cudaq9gradients15p | -   [cudaq::qreg::operator\[\]    |
| arameter_shift7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |     functi                        |
| -   [cudaq::gra                   | on)](api/languages/cpp_api.html#_ |
| dients::parameter_shift::gradient | CPPv4N5cudaq4qregixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qreg::qreg (C++       |
|     func                          |     function)                     |
| tion)](api/languages/cpp_api.html | ](api/languages/cpp_api.html#_CPP |
| #_CPPv4I00EN5cudaq9gradients15par | v4N5cudaq4qreg4qregENSt6size_tE), |
| ameter_shift8gradientER7KernelT), |     [\[1\]](api/languages/cpp_ap  |
|     [\[1\]](api/lan               | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| guages/cpp_api.html#_CPPv4I00EN5c | -   [cudaq::qreg::size (C++       |
| udaq9gradients15parameter_shift8g |                                   |
| radientER7KernelTRR10ArgsMapper), |  function)](api/languages/cpp_api |
|     [\[2\]](api/languages/c       | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| pp_api.html#_CPPv4I00EN5cudaq9gra | -   [cudaq::qreg::slice (C++      |
| dients15parameter_shift8gradientE |     function)](api/langu          |
| RR13QuantumKernelRR10ArgsMapper), | ages/cpp_api.html#_CPPv4N5cudaq4q |
|     [\[3\]](api/languages/        | reg5sliceENSt6size_tENSt6size_tE) |
| cpp_api.html#_CPPv4N5cudaq9gradie | -   [cudaq::qreg::value_type (C++ |
| nts15parameter_shift8gradientERRN |                                   |
| St8functionIFvNSt6vectorIdEEEEE), | type)](api/languages/cpp_api.html |
|     [\[4\]](api/languages         | #_CPPv4N5cudaq4qreg10value_typeE) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::qspan (C++            |
| ents15parameter_shift8gradientEv) |     class)](api/lang              |
| -   [cudaq::kernel_builder (C++   | uages/cpp_api.html#_CPPv4I_NSt6si |
|     clas                          | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| s)](api/languages/cpp_api.html#_C | -   [cudaq::QuakeValue (C++       |
| PPv4IDpEN5cudaq14kernel_builderE) |     class)](api/languages/cpp_api |
| -   [c                            | .html#_CPPv4N5cudaq10QuakeValueE) |
| udaq::kernel_builder::constantVal | -   [cudaq::Q                     |
|     (C++                          | uakeValue::canValidateNumElements |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languages      |
| q14kernel_builder11constantValEd) | /cpp_api.html#_CPPv4N5cudaq10Quak |
| -   [cu                           | eValue22canValidateNumElementsEv) |
| daq::kernel_builder::getArguments | -                                 |
|     (C++                          |  [cudaq::QuakeValue::constantSize |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api                |
| 14kernel_builder12getArgumentsEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cu                           | udaq10QuakeValue12constantSizeEv) |
| daq::kernel_builder::getNumParams | -   [cudaq::QuakeValue::dump (C++ |
|     (C++                          |     function)](api/lan            |
|     function)](api/lan            | guages/cpp_api.html#_CPPv4N5cudaq |
| guages/cpp_api.html#_CPPv4N5cudaq | 10QuakeValue4dumpERNSt7ostreamE), |
| 14kernel_builder12getNumParamsEv) |     [\                            |
| -   [c                            | [1\]](api/languages/cpp_api.html# |
| udaq::kernel_builder::isArgStdVec | _CPPv4N5cudaq10QuakeValue4dumpEv) |
|     (C++                          | -   [cudaq                        |
|     function)](api/languages/cp   | ::QuakeValue::getRequiredElements |
| p_api.html#_CPPv4N5cudaq14kernel_ |     (C++                          |
| builder11isArgStdVecENSt6size_tE) |     function)](api/langua         |
| -   [cuda                         | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| q::kernel_builder::kernel_builder | uakeValue19getRequiredElementsEv) |
|     (C++                          | -   [cudaq::QuakeValue::getValue  |
|     function)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq14kernel_bu |     function)]                    |
| ilder14kernel_builderERNSt6vector | (api/languages/cpp_api.html#_CPPv |
| IN7details17KernelBuilderTypeEEE) | 4NK5cudaq10QuakeValue8getValueEv) |
| -   [cudaq::kernel_builder::name  | -   [cudaq::QuakeValue::inverse   |
|     (C++                          |     (C++                          |
|     function)                     |     function)                     |
| ](api/languages/cpp_api.html#_CPP | ](api/languages/cpp_api.html#_CPP |
| v4N5cudaq14kernel_builder4nameEv) | v4NK5cudaq10QuakeValue7inverseEv) |
| -                                 | -   [cudaq::QuakeValue::isStdVec  |
|    [cudaq::kernel_builder::qalloc |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/language       | ](api/languages/cpp_api.html#_CPP |
| s/cpp_api.html#_CPPv4N5cudaq14ker | v4N5cudaq10QuakeValue8isStdVecEv) |
| nel_builder6qallocE10QuakeValue), | -                                 |
|     [\[1\]](api/language          |    [cudaq::QuakeValue::operator\* |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     (C++                          |
| nel_builder6qallocEKNSt6size_tE), |     function)](api                |
|     [\[2                          | /languages/cpp_api.html#_CPPv4N5c |
| \]](api/languages/cpp_api.html#_C | udaq10QuakeValuemlE10QuakeValue), |
| PPv4N5cudaq14kernel_builder6qallo |                                   |
| cERNSt6vectorINSt7complexIdEEEE), | [\[1\]](api/languages/cpp_api.htm |
|     [\[3\]](                      | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::QuakeValue::operator+ |
| N5cudaq14kernel_builder6qallocEv) |     (C++                          |
| -   [cudaq::kernel_builder::swap  |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/language       | udaq10QuakeValueplE10QuakeValue), |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     [                             |
| 4kernel_builder4swapEvRK10QuakeVa | \[1\]](api/languages/cpp_api.html |
| lueRK10QuakeValueRK10QuakeValue), | #_CPPv4N5cudaq10QuakeValueplEKd), |
|                                   |                                   |
| [\[1\]](api/languages/cpp_api.htm | [\[2\]](api/languages/cpp_api.htm |
| l#_CPPv4I00EN5cudaq14kernel_build | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| er4swapEvRKNSt6vectorI10QuakeValu | -   [cudaq::QuakeValue::operator- |
| eEERK10QuakeValueRK10QuakeValue), |     (C++                          |
|                                   |     function)](api                |
| [\[2\]](api/languages/cpp_api.htm | /languages/cpp_api.html#_CPPv4N5c |
| l#_CPPv4N5cudaq14kernel_builder4s | udaq10QuakeValuemiE10QuakeValue), |
| wapERK10QuakeValueRK10QuakeValue) |     [                             |
| -   [cudaq::KernelExecutionTask   | \[1\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     type                          |     [                             |
| )](api/languages/cpp_api.html#_CP | \[2\]](api/languages/cpp_api.html |
| Pv4N5cudaq19KernelExecutionTaskE) | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| -   [cudaq::KernelThunkResultType |                                   |
|     (C++                          | [\[3\]](api/languages/cpp_api.htm |
|     struct)]                      | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::QuakeValue::operator/ |
| 4N5cudaq21KernelThunkResultTypeE) |     (C++                          |
| -   [cudaq::KernelThunkType (C++  |     function)](api                |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
| type)](api/languages/cpp_api.html | udaq10QuakeValuedvE10QuakeValue), |
| #_CPPv4N5cudaq15KernelThunkTypeE) |                                   |
| -   [cudaq::kraus_channel (C++    | [\[1\]](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
|  class)](api/languages/cpp_api.ht | -                                 |
| ml#_CPPv4N5cudaq13kraus_channelE) |  [cudaq::QuakeValue::operator\[\] |
| -   [cudaq::kraus_channel::empty  |     (C++                          |
|     (C++                          |     function)](api                |
|     function)]                    | /languages/cpp_api.html#_CPPv4N5c |
| (api/languages/cpp_api.html#_CPPv | udaq10QuakeValueixEKNSt6size_tE), |
| 4NK5cudaq13kraus_channel5emptyEv) |     [\[1\]](api/                  |
| -   [cudaq::kraus_c               | languages/cpp_api.html#_CPPv4N5cu |
| hannel::generateUnitaryParameters | daq10QuakeValueixERK10QuakeValue) |
|     (C++                          | -                                 |
|                                   |    [cudaq::QuakeValue::QuakeValue |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4N5cudaq13kraus_chan |     function)](api/languag        |
| nel25generateUnitaryParametersEv) | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| -                                 | akeValue10QuakeValueERN4mlir20Imp |
|    [cudaq::kraus_channel::get_ops | licitLocOpBuilderEN4mlir5ValueE), |
|     (C++                          |     [\[1\]                        |
|     function)](a                  | ](api/languages/cpp_api.html#_CPP |
| pi/languages/cpp_api.html#_CPPv4N | v4N5cudaq10QuakeValue10QuakeValue |
| K5cudaq13kraus_channel7get_opsEv) | ERN4mlir20ImplicitLocOpBuilderEd) |
| -   [cud                          | -   [cudaq::QuakeValue::size (C++ |
| aq::kraus_channel::identity_flags |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     member)](api/lan              | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::QuakeValue::slice     |
| 13kraus_channel14identity_flagsE) |     (C++                          |
| -   [cud                          |     function)](api/languages/cpp_ |
| aq::kraus_channel::is_identity_op | api.html#_CPPv4N5cudaq10QuakeValu |
|     (C++                          | e5sliceEKNSt6size_tEKNSt6size_tE) |
|                                   | -   [cudaq::quantum_platform (C++ |
|    function)](api/languages/cpp_a |     cl                            |
| pi.html#_CPPv4NK5cudaq13kraus_cha | ass)](api/languages/cpp_api.html# |
| nnel14is_identity_opENSt6size_tE) | _CPPv4N5cudaq16quantum_platformE) |
| -   [cudaq::                      | -   [cudaq:                       |
| kraus_channel::is_unitary_mixture | :quantum_platform::beginExecution |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/languag        |
| /cpp_api.html#_CPPv4NK5cudaq13kra | es/cpp_api.html#_CPPv4N5cudaq16qu |
| us_channel18is_unitary_mixtureEv) | antum_platform14beginExecutionEv) |
| -   [cu                           | -   [cudaq::quantum_pl            |
| daq::kraus_channel::kraus_channel | atform::configureExecutionContext |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](api/lang           |
| uages/cpp_api.html#_CPPv4IDpEN5cu | uages/cpp_api.html#_CPPv4NK5cudaq |
| daq13kraus_channel13kraus_channel | 16quantum_platform25configureExec |
| EDpRRNSt16initializer_listI1TEE), | utionContextER16ExecutionContext) |
|                                   | -   [cuda                         |
|  [\[1\]](api/languages/cpp_api.ht | q::quantum_platform::connectivity |
| ml#_CPPv4N5cudaq13kraus_channel13 |     (C++                          |
| kraus_channelERK13kraus_channel), |     function)](api/langu          |
|     [\[2\]                        | ages/cpp_api.html#_CPPv4N5cudaq16 |
| ](api/languages/cpp_api.html#_CPP | quantum_platform12connectivityEv) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cuda                         |
| hannelERKNSt6vectorI8kraus_opEE), | q::quantum_platform::endExecution |
|     [\[3\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/langu          |
| v4N5cudaq13kraus_channel13kraus_c | ages/cpp_api.html#_CPPv4N5cudaq16 |
| hannelERRNSt6vectorI8kraus_opEE), | quantum_platform12endExecutionEv) |
|     [\[4\]](api/lan               | -   [cudaq::q                     |
| guages/cpp_api.html#_CPPv4N5cudaq | uantum_platform::enqueueAsyncTask |
| 13kraus_channel13kraus_channelEv) |     (C++                          |
| -                                 |     function)](api/languages/     |
| [cudaq::kraus_channel::noise_type | cpp_api.html#_CPPv4N5cudaq16quant |
|     (C++                          | um_platform16enqueueAsyncTaskEKNS |
|     member)](api                  | t6size_tER19KernelExecutionTask), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[1\]](api/languag           |
| udaq13kraus_channel10noise_typeE) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -                                 | antum_platform16enqueueAsyncTaskE |
|   [cudaq::kraus_channel::op_names | KNSt6size_tERNSt8functionIFvvEEE) |
|     (C++                          | -   [cudaq::quantum_p             |
|     member)](                     | latform::finalizeExecutionContext |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq13kraus_channel8op_namesE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq16quant |
|  [cudaq::kraus_channel::operator= | um_platform24finalizeExecutionCon |
|     (C++                          | textERN5cudaq16ExecutionContextE) |
|     function)](api/langua         | -   [cudaq::qua                   |
| ges/cpp_api.html#_CPPv4N5cudaq13k | ntum_platform::get_codegen_config |
| raus_channelaSERK13kraus_channel) |     (C++                          |
| -   [c                            |     function)](api/languages/c    |
| udaq::kraus_channel::operator\[\] | pp_api.html#_CPPv4N5cudaq16quantu |
|     (C++                          | m_platform18get_codegen_configEv) |
|     function)](api/l              | -   [cuda                         |
| anguages/cpp_api.html#_CPPv4N5cud | q::quantum_platform::get_exec_ctx |
| aq13kraus_channelixEKNSt6size_tE) |     (C++                          |
| -                                 |     function)](api/langua         |
| [cudaq::kraus_channel::parameters | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|     (C++                          | quantum_platform12get_exec_ctxEv) |
|     member)](api                  | -   [c                            |
| /languages/cpp_api.html#_CPPv4N5c | udaq::quantum_platform::get_noise |
| udaq13kraus_channel10parametersE) |     (C++                          |
| -   [cudaq::krau                  |     function)](api/languages/c    |
| s_channel::populateDefaultOpNames | pp_api.html#_CPPv4N5cudaq16quantu |
|     (C++                          | m_platform9get_noiseENSt6size_tE) |
|     function)](api/languages/cp   | -   [cudaq:                       |
| p_api.html#_CPPv4N5cudaq13kraus_c | :quantum_platform::get_num_qubits |
| hannel22populateDefaultOpNamesEv) |     (C++                          |
| -   [cu                           |                                   |
| daq::kraus_channel::probabilities | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4NK5cudaq16quantum_plat |
|     member)](api/la               | form14get_num_qubitsENSt6size_tE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::quantum_              |
| q13kraus_channel13probabilitiesE) | platform::get_remote_capabilities |
| -                                 |     (C++                          |
|  [cudaq::kraus_channel::push_back |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api                | v4NK5cudaq16quantum_platform23get |
| /languages/cpp_api.html#_CPPv4N5c | _remote_capabilitiesENSt6size_tE) |
| udaq13kraus_channel9push_backE8kr | -   [cudaq::qua                   |
| aus_opNSt8optionalINSt6stringEEE) | ntum_platform::get_runtime_target |
| -   [cudaq::kraus_channel::size   |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     function)                     | p_api.html#_CPPv4NK5cudaq16quantu |
| ](api/languages/cpp_api.html#_CPP | m_platform18get_runtime_targetEv) |
| v4NK5cudaq13kraus_channel4sizeEv) | -   [cud                          |
| -   [                             | aq::quantum_platform::is_emulated |
| cudaq::kraus_channel::unitary_ops |     (C++                          |
|     (C++                          |                                   |
|     member)](api/                 |    function)](api/languages/cpp_a |
| languages/cpp_api.html#_CPPv4N5cu | pi.html#_CPPv4NK5cudaq16quantum_p |
| daq13kraus_channel11unitary_opsE) | latform11is_emulatedENSt6size_tE) |
| -   [cudaq::kraus_op (C++         | -   [c                            |
|     struct)](api/languages/cpp_   | udaq::quantum_platform::is_remote |
| api.html#_CPPv4N5cudaq8kraus_opE) |     (C++                          |
| -   [cudaq::kraus_op::adjoint     |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4NK5cudaq16quantu |
|     functi                        | m_platform9is_remoteENSt6size_tE) |
| on)](api/languages/cpp_api.html#_ | -   [cuda                         |
| CPPv4NK5cudaq8kraus_op7adjointEv) | q::quantum_platform::is_simulator |
| -   [cudaq::kraus_op::data (C++   |     (C++                          |
|                                   |                                   |
|  member)](api/languages/cpp_api.h |   function)](api/languages/cpp_ap |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | i.html#_CPPv4NK5cudaq16quantum_pl |
| -   [cudaq::kraus_op::kraus_op    | atform12is_simulatorENSt6size_tE) |
|     (C++                          | -   [c                            |
|     func                          | udaq::quantum_platform::launchVQE |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |     function)](                   |
| opERRNSt16initializer_listI1TEE), | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq16quantum_platform9launchV |
|  [\[1\]](api/languages/cpp_api.ht | QEEKNSt6stringEPKvPN5cudaq8gradie |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | ntERKN5cudaq7spin_opERN5cudaq9opt |
| pENSt6vectorIN5cudaq7complexEEE), | imizerEKiKNSt6size_tENSt6size_tE) |
|     [\[2\]](api/l                 | -   [cudaq:                       |
| anguages/cpp_api.html#_CPPv4N5cud | :quantum_platform::list_platforms |
| aq8kraus_op8kraus_opERK8kraus_op) |     (C++                          |
| -   [cudaq::kraus_op::nCols (C++  |     function)](api/languag        |
|                                   | es/cpp_api.html#_CPPv4N5cudaq16qu |
| member)](api/languages/cpp_api.ht | antum_platform14list_platformsEv) |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | -                                 |
| -   [cudaq::kraus_op::nRows (C++  |    [cudaq::quantum_platform::name |
|                                   |     (C++                          |
| member)](api/languages/cpp_api.ht |     function)](a                  |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::kraus_op::operator=   | K5cudaq16quantum_platform4nameEv) |
|     (C++                          | -   [                             |
|     function)                     | cudaq::quantum_platform::num_qpus |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq8kraus_opaSERK8kraus_op) |     function)](api/l              |
| -   [cudaq::kraus_op::precision   | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq16quantum_platform8num_qpusEv) |
|     memb                          | -   [cudaq::                      |
| er)](api/languages/cpp_api.html#_ | quantum_platform::onRandomSeedSet |
| CPPv4N5cudaq8kraus_op9precisionE) |     (C++                          |
| -   [cudaq::KrausSelection (C++   |                                   |
|     s                             | function)](api/languages/cpp_api. |
| truct)](api/languages/cpp_api.htm | html#_CPPv4N5cudaq16quantum_platf |
| l#_CPPv4N5cudaq14KrausSelectionE) | orm15onRandomSeedSetENSt6size_tE) |
| -   [cudaq:                       | -   [cudaq:                       |
| :KrausSelection::circuit_location | :quantum_platform::reset_exec_ctx |
|     (C++                          |     (C++                          |
|     member)](api/langua           |     function)](api/languag        |
| ges/cpp_api.html#_CPPv4N5cudaq14K | es/cpp_api.html#_CPPv4N5cudaq16qu |
| rausSelection16circuit_locationE) | antum_platform14reset_exec_ctxEv) |
| -                                 | -   [cud                          |
|  [cudaq::KrausSelection::is_error | aq::quantum_platform::reset_noise |
|     (C++                          |     (C++                          |
|     member)](a                    |     function)](api/languages/cpp_ |
| pi/languages/cpp_api.html#_CPPv4N | api.html#_CPPv4N5cudaq16quantum_p |
| 5cudaq14KrausSelection8is_errorE) | latform11reset_noiseENSt6size_tE) |
| -   [cudaq::Kra                   | -   [cuda                         |
| usSelection::kraus_operator_index | q::quantum_platform::set_exec_ctx |
|     (C++                          |     (C++                          |
|     member)](api/languages/       |     funct                         |
| cpp_api.html#_CPPv4N5cudaq14Kraus | ion)](api/languages/cpp_api.html# |
| Selection20kraus_operator_indexE) | _CPPv4N5cudaq16quantum_platform12 |
| -   [cuda                         | set_exec_ctxEP16ExecutionContext) |
| q::KrausSelection::KrausSelection | -   [c                            |
|     (C++                          | udaq::quantum_platform::set_noise |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function                      |
| 5cudaq14KrausSelection14KrausSele | )](api/languages/cpp_api.html#_CP |
| ctionENSt6size_tENSt6vectorINSt6s | Pv4N5cudaq16quantum_platform9set_ |
| ize_tEEENSt6stringENSt6size_tEb), | noiseEPK11noise_modelNSt6size_tE) |
|     [\[1\]](api/langu             | -   [cudaq::quantum_platfor       |
| ages/cpp_api.html#_CPPv4N5cudaq14 | m::supports_explicit_measurements |
| KrausSelection14KrausSelectionEv) |     (C++                          |
| -                                 |     function)](api/l              |
|   [cudaq::KrausSelection::op_name | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq16quantum_platform30supports_e |
|     member)](                     | xplicit_measurementsENSt6size_tE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::quantum_pla           |
| N5cudaq14KrausSelection7op_nameE) | tform::supports_task_distribution |
| -   [                             |     (C++                          |
| cudaq::KrausSelection::operator== |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function)](api/languages      | ml#_CPPv4NK5cudaq16quantum_platfo |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | rm26supports_task_distributionEv) |
| usSelectioneqERK14KrausSelection) | -   [cudaq::quantum               |
| -                                 | _platform::with_execution_context |
|    [cudaq::KrausSelection::qubits |     (C++                          |
|     (C++                          |     function)                     |
|     member)]                      | ](api/languages/cpp_api.html#_CPP |
| (api/languages/cpp_api.html#_CPPv | v4I0DpEN5cudaq16quantum_platform2 |
| 4N5cudaq14KrausSelection6qubitsE) | 2with_execution_contextEDaR16Exec |
| -   [cudaq::KrausTrajectory (C++  | utionContextRR8CallableDpRR4Args) |
|     st                            | -   [cudaq::QuantumTask (C++      |
| ruct)](api/languages/cpp_api.html |     type)](api/languages/cpp_api. |
| #_CPPv4N5cudaq15KrausTrajectoryE) | html#_CPPv4N5cudaq11QuantumTaskE) |
| -                                 | -   [cudaq::qubit (C++            |
|  [cudaq::KrausTrajectory::builder |     type)](api/languages/c        |
|     (C++                          | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     function)](ap                 | -   [cudaq::QubitConnectivity     |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq15KrausTrajectory7builderEv) |     ty                            |
| -   [cu                           | pe)](api/languages/cpp_api.html#_ |
| daq::KrausTrajectory::countErrors | CPPv4N5cudaq17QubitConnectivityE) |
|     (C++                          | -   [cudaq::QubitEdge (C++        |
|     function)](api/lang           |     type)](api/languages/cpp_a    |
| uages/cpp_api.html#_CPPv4NK5cudaq | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| 15KrausTrajectory11countErrorsEv) | -   [cudaq::qudit (C++            |
| -   [                             |     clas                          |
| cudaq::KrausTrajectory::isOrdered | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     function)](api/l              | -   [cudaq::qudit::qudit (C++     |
| anguages/cpp_api.html#_CPPv4NK5cu |                                   |
| daq15KrausTrajectory9isOrderedEv) | function)](api/languages/cpp_api. |
| -   [cudaq::                      | html#_CPPv4N5cudaq5qudit5quditEv) |
| KrausTrajectory::kraus_selections | -   [cudaq::qvector (C++          |
|     (C++                          |     class)                        |
|     member)](api/languag          | ](api/languages/cpp_api.html#_CPP |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| ausTrajectory16kraus_selectionsE) | -   [cudaq::qvector::back (C++    |
| -   [cudaq:                       |     function)](a                  |
| :KrausTrajectory::KrausTrajectory | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq7qvector4backENSt6size_tE), |
|     function                      |                                   |
| )](api/languages/cpp_api.html#_CP |   [\[1\]](api/languages/cpp_api.h |
| Pv4N5cudaq15KrausTrajectory15Krau | tml#_CPPv4N5cudaq7qvector4backEv) |
| sTrajectoryENSt6size_tENSt6vector | -   [cudaq::qvector::begin (C++   |
| I14KrausSelectionEEdNSt6size_tE), |     fu                            |
|     [\[1\]](api/languag           | nction)](api/languages/cpp_api.ht |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | ml#_CPPv4N5cudaq7qvector5beginEv) |
| ausTrajectory15KrausTrajectoryEv) | -   [cudaq::qvector::clear (C++   |
| -   [cudaq::Kr                    |     fu                            |
| ausTrajectory::measurement_counts | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     member)](api/languages        | -   [cudaq::qvector::end (C++     |
| /cpp_api.html#_CPPv4N5cudaq15Krau |                                   |
| sTrajectory18measurement_countsE) | function)](api/languages/cpp_api. |
| -   [cud                          | html#_CPPv4N5cudaq7qvector3endEv) |
| aq::KrausTrajectory::multiplicity | -   [cudaq::qvector::front (C++   |
|     (C++                          |     function)](ap                 |
|     member)](api/lan              | i/languages/cpp_api.html#_CPPv4N5 |
| guages/cpp_api.html#_CPPv4N5cudaq | cudaq7qvector5frontENSt6size_tE), |
| 15KrausTrajectory12multiplicityE) |                                   |
| -   [                             |  [\[1\]](api/languages/cpp_api.ht |
| cudaq::KrausTrajectory::num_shots | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     (C++                          | -   [cudaq::qvector::operator=    |
|     member)](api                  |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     functio                       |
| udaq15KrausTrajectory9num_shotsE) | n)](api/languages/cpp_api.html#_C |
| -   [c                            | PPv4N5cudaq7qvectoraSERK7qvector) |
| udaq::KrausTrajectory::operator== | -   [cudaq::qvector::operator\[\] |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     function)                     |
| pp_api.html#_CPPv4NK5cudaq15Kraus | ](api/languages/cpp_api.html#_CPP |
| TrajectoryeqERK15KrausTrajectory) | v4N5cudaq7qvectorixEKNSt6size_tE) |
| -   [cu                           | -   [cudaq::qvector::qvector (C++ |
| daq::KrausTrajectory::probability |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     member)](api/la               | daq7qvector7qvectorENSt6size_tE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[1\]](a                     |
| q15KrausTrajectory11probabilityE) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cuda                         | 5cudaq7qvector7qvectorERK5state), |
| q::KrausTrajectory::trajectory_id |     [\[2\]](api                   |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/lang             | udaq7qvector7qvectorERK7qvector), |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     [\[3\]](ap                    |
| 5KrausTrajectory13trajectory_idE) | i/languages/cpp_api.html#_CPPv4N5 |
| -                                 | cudaq7qvector7qvectorERR7qvector) |
|   [cudaq::KrausTrajectory::weight | -   [cudaq::qvector::size (C++    |
|     (C++                          |     fu                            |
|     member)](                     | nction)](api/languages/cpp_api.ht |
| api/languages/cpp_api.html#_CPPv4 | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| N5cudaq15KrausTrajectory6weightE) | -   [cudaq::qvector::slice (C++   |
| -                                 |     function)](api/language       |
|    [cudaq::KrausTrajectoryBuilder | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|     (C++                          | tor5sliceENSt6size_tENSt6size_tE) |
|     class)](                      | -   [cudaq::qvector::value_type   |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22KrausTrajectoryBuilderE) |     typ                           |
| -   [cud                          | e)](api/languages/cpp_api.html#_C |
| aq::KrausTrajectoryBuilder::build | PPv4N5cudaq7qvector10value_typeE) |
|     (C++                          | -   [cudaq::qview (C++            |
|     function)](api/lang           |     clas                          |
| uages/cpp_api.html#_CPPv4NK5cudaq | s)](api/languages/cpp_api.html#_C |
| 22KrausTrajectoryBuilder5buildEv) | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| -   [cud                          | -   [cudaq::qview::back (C++      |
| aq::KrausTrajectoryBuilder::setId |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/languages/cpp  | v4N5cudaq5qview4backENSt6size_tE) |
| _api.html#_CPPv4N5cudaq22KrausTra | -   [cudaq::qview::begin (C++     |
| jectoryBuilder5setIdENSt6size_tE) |                                   |
| -   [cudaq::Kraus                 | function)](api/languages/cpp_api. |
| TrajectoryBuilder::setProbability | html#_CPPv4N5cudaq5qview5beginEv) |
|     (C++                          | -   [cudaq::qview::end (C++       |
|     function)](api/languages/cpp  |                                   |
| _api.html#_CPPv4N5cudaq22KrausTra |   function)](api/languages/cpp_ap |
| jectoryBuilder14setProbabilityEd) | i.html#_CPPv4N5cudaq5qview3endEv) |
| -   [cudaq::Krau                  | -   [cudaq::qview::front (C++     |
| sTrajectoryBuilder::setSelections |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/languag        | N5cudaq5qview5frontENSt6size_tE), |
| es/cpp_api.html#_CPPv4N5cudaq22Kr |                                   |
| ausTrajectoryBuilder13setSelectio |    [\[1\]](api/languages/cpp_api. |
| nsENSt6vectorI14KrausSelectionEE) | html#_CPPv4N5cudaq5qview5frontEv) |
| -   [cudaq::matrix_callback (C++  | -   [cudaq::qview::operator\[\]   |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     functio                       |
| #_CPPv4N5cudaq15matrix_callbackE) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::matrix_handler (C++   | PPv4N5cudaq5qviewixEKNSt6size_tE) |
|                                   | -   [cudaq::qview::qview (C++     |
| class)](api/languages/cpp_api.htm |     functio                       |
| l#_CPPv4N5cudaq14matrix_handlerE) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::mat                   | PPv4I0EN5cudaq5qview5qviewERR1R), |
| rix_handler::commutation_behavior |     [\[1                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     struct)](api/languages/       | PPv4N5cudaq5qview5qviewERK5qview) |
| cpp_api.html#_CPPv4N5cudaq14matri | -   [cudaq::qview::size (C++      |
| x_handler20commutation_behaviorE) |                                   |
| -                                 | function)](api/languages/cpp_api. |
|    [cudaq::matrix_handler::define | html#_CPPv4NK5cudaq5qview4sizeEv) |
|     (C++                          | -   [cudaq::qview::slice (C++     |
|     function)](a                  |     function)](api/langua         |
| pi/languages/cpp_api.html#_CPPv4N | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| 5cudaq14matrix_handler6defineENSt | iew5sliceENSt6size_tENSt6size_tE) |
| 6stringENSt6vectorINSt7int64_tEEE | -   [cudaq::qview::value_type     |
| RR15matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     t                             |
|                                   | ype)](api/languages/cpp_api.html# |
| [\[1\]](api/languages/cpp_api.htm | _CPPv4N5cudaq5qview10value_typeE) |
| l#_CPPv4N5cudaq14matrix_handler6d | -   [cudaq::range (C++            |
| efineENSt6stringENSt6vectorINSt7i |     fun                           |
| nt64_tEEERR15matrix_callbackRR20d | ction)](api/languages/cpp_api.htm |
| iag_matrix_callbackRKNSt13unorder | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| ed_mapINSt6stringENSt6stringEEE), | orI11ElementTypeEE11ElementType), |
|     [\[2\]](                      |     [\[1\]](api/languages/cpp_    |
| api/languages/cpp_api.html#_CPPv4 | api.html#_CPPv4I0EN5cudaq5rangeEN |
| N5cudaq14matrix_handler6defineENS | St6vectorI11ElementTypeEE11Elemen |
| t6stringENSt6vectorINSt7int64_tEE | tType11ElementType11ElementType), |
| ERR15matrix_callbackRRNSt13unorde |     [                             |
| red_mapINSt6stringENSt6stringEEE) | \[2\]](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq5rangeENSt6size_tE) |
|   [cudaq::matrix_handler::degrees | -   [cudaq::real (C++             |
|     (C++                          |     type)](api/languages/         |
|     function)](ap                 | cpp_api.html#_CPPv4N5cudaq4realE) |
| i/languages/cpp_api.html#_CPPv4NK | -   [cudaq::registry (C++         |
| 5cudaq14matrix_handler7degreesEv) |     type)](api/languages/cpp_     |
| -                                 | api.html#_CPPv4N5cudaq8registryE) |
|  [cudaq::matrix_handler::displace | -                                 |
|     (C++                          |  [cudaq::registry::RegisteredType |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     class)](api/                  |
| rix_handler8displaceENSt6size_tE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::matrix                | 5cudaq8registry14RegisteredTypeE) |
| _handler::get_expected_dimensions | -   [cudaq::RemoteCapabilities    |
|     (C++                          |     (C++                          |
|                                   |     struc                         |
|    function)](api/languages/cpp_a | t)](api/languages/cpp_api.html#_C |
| pi.html#_CPPv4NK5cudaq14matrix_ha | PPv4N5cudaq18RemoteCapabilitiesE) |
| ndler23get_expected_dimensionsEv) | -   [cudaq::Remot                 |
| -   [cudaq::matrix_ha             | eCapabilities::RemoteCapabilities |
| ndler::get_parameter_descriptions |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|                                   | _api.html#_CPPv4N5cudaq18RemoteCa |
| function)](api/languages/cpp_api. | pabilities18RemoteCapabilitiesEb) |
| html#_CPPv4NK5cudaq14matrix_handl | -   [cudaq:                       |
| er26get_parameter_descriptionsEv) | :RemoteCapabilities::stateOverlap |
| -   [c                            |     (C++                          |
| udaq::matrix_handler::instantiate |     member)](api/langua           |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq18R |
|     function)](a                  | emoteCapabilities12stateOverlapE) |
| pi/languages/cpp_api.html#_CPPv4N | -                                 |
| 5cudaq14matrix_handler11instantia |   [cudaq::RemoteCapabilities::vqe |
| teENSt6stringERKNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     member)](                     |
|     [\[1\]](                      | api/languages/cpp_api.html#_CPPv4 |
| api/languages/cpp_api.html#_CPPv4 | N5cudaq18RemoteCapabilities3vqeE) |
| N5cudaq14matrix_handler11instanti | -   [cudaq::Resources (C++        |
| ateENSt6stringERRNSt6vectorINSt6s |     class)](api/languages/cpp_a   |
| ize_tEEERK20commutation_behavior) | pi.html#_CPPv4N5cudaq9ResourcesE) |
| -   [cuda                         | -   [cudaq::run (C++              |
| q::matrix_handler::matrix_handler |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/languag        | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| es/cpp_api.html#_CPPv4I0_NSt11ena | 5invoke_result_tINSt7decay_tI13Qu |
| ble_if_tINSt12is_base_of_vI16oper | antumKernelEEDpNSt7decay_tI4ARGSE |
| ator_handler1TEEbEEEN5cudaq14matr | EEEEENSt6size_tERN5cudaq11noise_m |
| ix_handler14matrix_handlerERK1T), | odelERR13QuantumKernelDpRR4ARGS), |
|     [\[1\]](ap                    |     [\[1\]](api/langu             |
| i/languages/cpp_api.html#_CPPv4I0 | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| _NSt11enable_if_tINSt12is_base_of | daq3runENSt6vectorINSt15invoke_re |
| _vI16operator_handler1TEEbEEEN5cu | sult_tINSt7decay_tI13QuantumKerne |
| daq14matrix_handler14matrix_handl | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| erERK1TRK20commutation_behavior), | ize_tERR13QuantumKernelDpRR4ARGS) |
|     [\[2\]](api/languages/cpp_ap  | -   [cudaq::run_async (C++        |
| i.html#_CPPv4N5cudaq14matrix_hand |     functio                       |
| ler14matrix_handlerENSt6size_tE), | n)](api/languages/cpp_api.html#_C |
|     [\[3\]](api/                  | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| languages/cpp_api.html#_CPPv4N5cu | tureINSt6vectorINSt15invoke_resul |
| daq14matrix_handler14matrix_handl | t_tINSt7decay_tI13QuantumKernelEE |
| erENSt6stringERKNSt6vectorINSt6si | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| ze_tEEERK20commutation_behavior), | ze_tENSt6size_tERN5cudaq11noise_m |
|     [\[4\]](api/                  | odelERR13QuantumKernelDpRR4ARGS), |
| languages/cpp_api.html#_CPPv4N5cu |     [\[1\]](api/la                |
| daq14matrix_handler14matrix_handl | nguages/cpp_api.html#_CPPv4I0DpEN |
| erENSt6stringERRNSt6vectorINSt6si | 5cudaq9run_asyncENSt6futureINSt6v |
| ze_tEEERK20commutation_behavior), | ectorINSt15invoke_result_tINSt7de |
|     [\                            | cay_tI13QuantumKernelEEDpNSt7deca |
| [5\]](api/languages/cpp_api.html# | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| _CPPv4N5cudaq14matrix_handler14ma | ize_tERR13QuantumKernelDpRR4ARGS) |
| trix_handlerERK14matrix_handler), | -   [cudaq::RuntimeTarget (C++    |
|     [                             |                                   |
| \[6\]](api/languages/cpp_api.html | struct)](api/languages/cpp_api.ht |
| #_CPPv4N5cudaq14matrix_handler14m | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| atrix_handlerERR14matrix_handler) | -   [cudaq::sample (C++           |
| -                                 |     function)](api/languages/c    |
|  [cudaq::matrix_handler::momentum | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     (C++                          | mpleE13sample_resultRK14sample_op |
|     function)](api/language       | tionsRR13QuantumKernelDpRR4Args), |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     [\[1\                         |
| rix_handler8momentumENSt6size_tE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4I0DpEN5cudaq6sampleE13sample_r |
|    [cudaq::matrix_handler::number | esultRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\                            |
|     function)](api/langua         | [2\]](api/languages/cpp_api.html# |
| ges/cpp_api.html#_CPPv4N5cudaq14m | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| atrix_handler6numberENSt6size_tE) | ize_tERR13QuantumKernelDpRR4Args) |
| -                                 | -   [cudaq::sample_options (C++   |
| [cudaq::matrix_handler::operator= |     s                             |
|     (C++                          | truct)](api/languages/cpp_api.htm |
|     fun                           | l#_CPPv4N5cudaq14sample_optionsE) |
| ction)](api/languages/cpp_api.htm | -   [cudaq::sample_result (C++    |
| l#_CPPv4I0_NSt11enable_if_tIXaant |                                   |
| NSt7is_sameI1T14matrix_handlerE5v |  class)](api/languages/cpp_api.ht |
| alueENSt12is_base_of_vI16operator | ml#_CPPv4N5cudaq13sample_resultE) |
| _handler1TEEEbEEEN5cudaq14matrix_ | -   [cudaq::sample_result::append |
| handleraSER14matrix_handlerRK1T), |     (C++                          |
|     [\[1\]](api/languages         |     function)](api/languages/cpp_ |
| /cpp_api.html#_CPPv4N5cudaq14matr | api.html#_CPPv4N5cudaq13sample_re |
| ix_handleraSERK14matrix_handler), | sult6appendERK15ExecutionResultb) |
|     [\[2\]](api/language          | -   [cudaq::sample_result::begin  |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handleraSERR14matrix_handler) |     function)]                    |
| -   [                             | (api/languages/cpp_api.html#_CPPv |
| cudaq::matrix_handler::operator== | 4N5cudaq13sample_result5beginEv), |
|     (C++                          |     [\[1\]]                       |
|     function)](api/languages      | (api/languages/cpp_api.html#_CPPv |
| /cpp_api.html#_CPPv4NK5cudaq14mat | 4NK5cudaq13sample_result5beginEv) |
| rix_handlereqERK14matrix_handler) | -   [cudaq::sample_result::cbegin |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::parity |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/langua         | NK5cudaq13sample_result6cbeginEv) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -   [cudaq::sample_result::cend   |
| atrix_handler6parityENSt6size_tE) |     (C++                          |
| -                                 |     function)                     |
|  [cudaq::matrix_handler::position | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq13sample_result4cendEv) |
|     function)](api/language       | -   [cudaq::sample_result::clear  |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8positionENSt6size_tE) |     function)                     |
| -   [cudaq::                      | ](api/languages/cpp_api.html#_CPP |
| matrix_handler::remove_definition | v4N5cudaq13sample_result5clearEv) |
|     (C++                          | -   [cudaq::sample_result::count  |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](                   |
| ml#_CPPv4N5cudaq14matrix_handler1 | api/languages/cpp_api.html#_CPPv4 |
| 7remove_definitionERKNSt6stringE) | NK5cudaq13sample_result5countENSt |
| -                                 | 11string_viewEKNSt11string_viewE) |
|   [cudaq::matrix_handler::squeeze | -   [                             |
|     (C++                          | cudaq::sample_result::deserialize |
|     function)](api/languag        |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     functio                       |
| trix_handler7squeezeENSt6size_tE) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::m                     | PPv4N5cudaq13sample_result11deser |
| atrix_handler::to_diagonal_matrix | ializeERNSt6vectorINSt6size_tEEE) |
|     (C++                          | -   [cudaq::sample_result::dump   |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     function)](api/languag        |
| 14matrix_handler18to_diagonal_mat | es/cpp_api.html#_CPPv4NK5cudaq13s |
| rixERNSt13unordered_mapINSt6size_ | ample_result4dumpERNSt7ostreamE), |
| tENSt7int64_tEEERKNSt13unordered_ |     [\[1\]                        |
| mapINSt6stringENSt7complexIdEEEE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4NK5cudaq13sample_result4dumpEv) |
| [cudaq::matrix_handler::to_matrix | -   [cudaq::sample_result::end    |
|     (C++                          |     (C++                          |
|     function)                     |     function                      |
| ](api/languages/cpp_api.html#_CPP | )](api/languages/cpp_api.html#_CP |
| v4NK5cudaq14matrix_handler9to_mat | Pv4N5cudaq13sample_result3endEv), |
| rixERNSt13unordered_mapINSt6size_ |     [\[1\                         |
| tENSt7int64_tEEERKNSt13unordered_ | ]](api/languages/cpp_api.html#_CP |
| mapINSt6stringENSt7complexIdEEEE) | Pv4NK5cudaq13sample_result3endEv) |
| -                                 | -   [                             |
| [cudaq::matrix_handler::to_string | cudaq::sample_result::expectation |
|     (C++                          |     (C++                          |
|     function)](api/               |     f                             |
| languages/cpp_api.html#_CPPv4NK5c | unction)](api/languages/cpp_api.h |
| udaq14matrix_handler9to_stringEb) | tml#_CPPv4NK5cudaq13sample_result |
| -                                 | 11expectationEKNSt11string_viewE) |
| [cudaq::matrix_handler::unique_id | -   [c                            |
|     (C++                          | udaq::sample_result::get_marginal |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     function)](api/languages/cpp_ |
| udaq14matrix_handler9unique_idEv) | api.html#_CPPv4NK5cudaq13sample_r |
| -   [cudaq:                       | esult12get_marginalERKNSt6vectorI |
| :matrix_handler::\~matrix_handler | NSt6size_tEEEKNSt11string_viewE), |
|     (C++                          |     [\[1\]](api/languages/cpp_    |
|     functi                        | api.html#_CPPv4NK5cudaq13sample_r |
| on)](api/languages/cpp_api.html#_ | esult12get_marginalERRKNSt6vector |
| CPPv4N5cudaq14matrix_handlerD0Ev) | INSt6size_tEEEKNSt11string_viewE) |
| -   [cudaq::matrix_op (C++        | -   [cuda                         |
|     type)](api/languages/cpp_a    | q::sample_result::get_total_shots |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     (C++                          |
| -   [cudaq::matrix_op_term (C++   |     function)](api/langua         |
|                                   | ges/cpp_api.html#_CPPv4NK5cudaq13 |
|  type)](api/languages/cpp_api.htm | sample_result15get_total_shotsEv) |
| l#_CPPv4N5cudaq14matrix_op_termE) | -   [cuda                         |
| -                                 | q::sample_result::has_even_parity |
|    [cudaq::mdiag_operator_handler |     (C++                          |
|     (C++                          |     fun                           |
|     class)](                      | ction)](api/languages/cpp_api.htm |
| api/languages/cpp_api.html#_CPPv4 | l#_CPPv4N5cudaq13sample_result15h |
| N5cudaq22mdiag_operator_handlerE) | as_even_parityENSt11string_viewE) |
| -   [cudaq::mpi (C++              | -   [cuda                         |
|     type)](api/languages          | q::sample_result::has_expectation |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) |     (C++                          |
| -   [cudaq::mpi::all_gather (C++  |     funct                         |
|     fu                            | ion)](api/languages/cpp_api.html# |
| nction)](api/languages/cpp_api.ht | _CPPv4NK5cudaq13sample_result15ha |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | s_expectationEKNSt11string_viewE) |
| RNSt6vectorIdEERKNSt6vectorIdEE), | -   [cu                           |
|                                   | daq::sample_result::most_probable |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi10all_gather |     fun                           |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | ction)](api/languages/cpp_api.htm |
| -   [cudaq::mpi::all_reduce (C++  | l#_CPPv4NK5cudaq13sample_result13 |
|                                   | most_probableEKNSt11string_viewE) |
|  function)](api/languages/cpp_api | -                                 |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | [cudaq::sample_result::operator+= |
| reduceE1TRK1TRK14BinaryFunction), |     (C++                          |
|     [\[1\]](api/langu             |     function)](api/langua         |
| ages/cpp_api.html#_CPPv4I00EN5cud | ges/cpp_api.html#_CPPv4N5cudaq13s |
| aq3mpi10all_reduceE1TRK1TRK4Func) | ample_resultpLERK13sample_result) |
| -   [cudaq::mpi::broadcast (C++   | -                                 |
|     function)](api/               |  [cudaq::sample_result::operator= |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq3mpi9broadcastERNSt6stringEi), |     function)](api/langua         |
|     [\[1\]](api/la                | ges/cpp_api.html#_CPPv4N5cudaq13s |
| nguages/cpp_api.html#_CPPv4N5cuda | ample_resultaSERR13sample_result) |
| q3mpi9broadcastERNSt6vectorIdEEi) | -                                 |
| -   [cudaq::mpi::finalize (C++    | [cudaq::sample_result::operator== |
|     f                             |     (C++                          |
| unction)](api/languages/cpp_api.h |     function)](api/languag        |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | es/cpp_api.html#_CPPv4NK5cudaq13s |
| -   [cudaq::mpi::initialize (C++  | ample_resulteqERK13sample_result) |
|     function                      | -   [                             |
| )](api/languages/cpp_api.html#_CP | cudaq::sample_result::probability |
| Pv4N5cudaq3mpi10initializeEiPPc), |     (C++                          |
|     [                             |     function)](api/lan            |
| \[1\]](api/languages/cpp_api.html | guages/cpp_api.html#_CPPv4NK5cuda |
| #_CPPv4N5cudaq3mpi10initializeEv) | q13sample_result11probabilityENSt |
| -   [cudaq::mpi::is_initialized   | 11string_viewEKNSt11string_viewE) |
|     (C++                          | -   [cud                          |
|     function                      | aq::sample_result::register_names |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq3mpi14is_initializedEv) |     function)](api/langu          |
| -   [cudaq::mpi::num_ranks (C++   | ages/cpp_api.html#_CPPv4NK5cudaq1 |
|     fu                            | 3sample_result14register_namesEv) |
| nction)](api/languages/cpp_api.ht | -                                 |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |    [cudaq::sample_result::reorder |
| -   [cudaq::mpi::rank (C++        |     (C++                          |
|                                   |     function)](api/langua         |
|    function)](api/languages/cpp_a | ges/cpp_api.html#_CPPv4N5cudaq13s |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | ample_result7reorderERKNSt6vector |
| -   [cudaq::noise_model (C++      | INSt6size_tEEEKNSt11string_viewE) |
|                                   | -   [cu                           |
|    class)](api/languages/cpp_api. | daq::sample_result::sample_result |
| html#_CPPv4N5cudaq11noise_modelE) |     (C++                          |
| -   [cudaq::n                     |     func                          |
| oise_model::add_all_qubit_channel | tion)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq13sample_result13sa |
|     function)](api                | mple_resultERK15ExecutionResult), |
| /languages/cpp_api.html#_CPPv4IDp |     [\[1\]](api/la                |
| EN5cudaq11noise_model21add_all_qu | nguages/cpp_api.html#_CPPv4N5cuda |
| bit_channelEvRK13kraus_channeli), | q13sample_result13sample_resultER |
|     [\[1\]](api/langua            | KNSt6vectorI15ExecutionResultEE), |
| ges/cpp_api.html#_CPPv4N5cudaq11n |                                   |
| oise_model21add_all_qubit_channel |  [\[2\]](api/languages/cpp_api.ht |
| ERKNSt6stringERK13kraus_channeli) | ml#_CPPv4N5cudaq13sample_result13 |
| -                                 | sample_resultERR13sample_result), |
|  [cudaq::noise_model::add_channel |     [                             |
|     (C++                          | \[3\]](api/languages/cpp_api.html |
|     funct                         | #_CPPv4N5cudaq13sample_result13sa |
| ion)](api/languages/cpp_api.html# | mple_resultERR15ExecutionResult), |
| _CPPv4IDpEN5cudaq11noise_model11a |     [\[4\]](api/lan               |
| dd_channelEvRK15PredicateFuncTy), | guages/cpp_api.html#_CPPv4N5cudaq |
|     [\[1\]](api/languages/cpp_    | 13sample_result13sample_resultEdR |
| api.html#_CPPv4IDpEN5cudaq11noise | KNSt6vectorI15ExecutionResultEE), |
| _model11add_channelEvRKNSt6vector |     [\[5\]](api/lan               |
| INSt6size_tEEERK13kraus_channel), | guages/cpp_api.html#_CPPv4N5cudaq |
|     [\[2\]](ap                    | 13sample_result13sample_resultEv) |
| i/languages/cpp_api.html#_CPPv4N5 | -                                 |
| cudaq11noise_model11add_channelER |  [cudaq::sample_result::serialize |
| KNSt6stringERK15PredicateFuncTy), |     (C++                          |
|                                   |     function)](api                |
| [\[3\]](api/languages/cpp_api.htm | /languages/cpp_api.html#_CPPv4NK5 |
| l#_CPPv4N5cudaq11noise_model11add | cudaq13sample_result9serializeEv) |
| _channelERKNSt6stringERKNSt6vecto | -   [cudaq::sample_result::size   |
| rINSt6size_tEEERK13kraus_channel) |     (C++                          |
| -   [cudaq::noise_model::empty    |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq13sampl |
|     function                      | e_result4sizeEKNSt11string_viewE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::sample_result::to_map |
| Pv4NK5cudaq11noise_model5emptyEv) |     (C++                          |
| -                                 |     function)](api/languages/cpp  |
| [cudaq::noise_model::get_channels | _api.html#_CPPv4NK5cudaq13sample_ |
|     (C++                          | result6to_mapEKNSt11string_viewE) |
|     function)](api/l              | -   [cuda                         |
| anguages/cpp_api.html#_CPPv4I0ENK | q::sample_result::\~sample_result |
| 5cudaq11noise_model12get_channels |     (C++                          |
| ENSt6vectorI13kraus_channelEERKNS |     funct                         |
| t6vectorINSt6size_tEEERKNSt6vecto | ion)](api/languages/cpp_api.html# |
| rINSt6size_tEEERKNSt6vectorIdEE), | _CPPv4N5cudaq13sample_resultD0Ev) |
|     [\[1\]](api/languages/cpp_a   | -   [cudaq::scalar_callback (C++  |
| pi.html#_CPPv4NK5cudaq11noise_mod |     c                             |
| el12get_channelsERKNSt6stringERKN | lass)](api/languages/cpp_api.html |
| St6vectorINSt6size_tEEERKNSt6vect | #_CPPv4N5cudaq15scalar_callbackE) |
| orINSt6size_tEEERKNSt6vectorIdEE) | -   [c                            |
| -                                 | udaq::scalar_callback::operator() |
|  [cudaq::noise_model::noise_model |     (C++                          |
|     (C++                          |     function)](api/language       |
|     function)](api                | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| /languages/cpp_api.html#_CPPv4N5c | alar_callbackclERKNSt13unordered_ |
| udaq11noise_model11noise_modelEv) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cu                           | -   [                             |
| daq::noise_model::PredicateFuncTy | cudaq::scalar_callback::operator= |
|     (C++                          |     (C++                          |
|     type)](api/la                 |     function)](api/languages/c    |
| nguages/cpp_api.html#_CPPv4N5cuda | pp_api.html#_CPPv4N5cudaq15scalar |
| q11noise_model15PredicateFuncTyE) | _callbackaSERK15scalar_callback), |
| -   [cud                          |     [\[1\]](api/languages/        |
| aq::noise_model::register_channel | cpp_api.html#_CPPv4N5cudaq15scala |
|     (C++                          | r_callbackaSERR15scalar_callback) |
|     function)](api/languages      | -   [cudaq:                       |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | :scalar_callback::scalar_callback |
| noise_model16register_channelEvv) |     (C++                          |
| -   [cudaq::                      |     function)](api/languag        |
| noise_model::requires_constructor | es/cpp_api.html#_CPPv4I0_NSt11ena |
|     (C++                          | ble_if_tINSt16is_invocable_r_vINS |
|     type)](api/languages/cp       | t7complexIdEE8CallableRKNSt13unor |
| p_api.html#_CPPv4I0DpEN5cudaq11no | dered_mapINSt6stringENSt7complexI |
| ise_model20requires_constructorE) | dEEEEEEbEEEN5cudaq15scalar_callba |
| -   [cudaq::noise_model_type (C++ | ck15scalar_callbackERR8Callable), |
|     e                             |     [\[1\                         |
| num)](api/languages/cpp_api.html# | ]](api/languages/cpp_api.html#_CP |
| _CPPv4N5cudaq16noise_model_typeE) | Pv4N5cudaq15scalar_callback15scal |
| -   [cudaq::no                    | ar_callbackERK15scalar_callback), |
| ise_model_type::amplitude_damping |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     enumerator)](api/languages    | PPv4N5cudaq15scalar_callback15sca |
| /cpp_api.html#_CPPv4N5cudaq16nois | lar_callbackERR15scalar_callback) |
| e_model_type17amplitude_dampingE) | -   [cudaq::scalar_operator (C++  |
| -   [cudaq::noise_mode            |     c                             |
| l_type::amplitude_damping_channel | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_operatorE) |
|     e                             | -                                 |
| numerator)](api/languages/cpp_api | [cudaq::scalar_operator::evaluate |
| .html#_CPPv4N5cudaq16noise_model_ |     (C++                          |
| type25amplitude_damping_channelE) |                                   |
| -   [cudaq::n                     |    function)](api/languages/cpp_a |
| oise_model_type::bit_flip_channel | pi.html#_CPPv4NK5cudaq15scalar_op |
|     (C++                          | erator8evaluateERKNSt13unordered_ |
|     enumerator)](api/language     | mapINSt6stringENSt7complexIdEEEE) |
| s/cpp_api.html#_CPPv4N5cudaq16noi | -   [cudaq::scalar_ope            |
| se_model_type16bit_flip_channelE) | rator::get_parameter_descriptions |
| -   [cudaq::                      |     (C++                          |
| noise_model_type::depolarization1 |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     enumerator)](api/languag      | tml#_CPPv4NK5cudaq15scalar_operat |
| es/cpp_api.html#_CPPv4N5cudaq16no | or26get_parameter_descriptionsEv) |
| ise_model_type15depolarization1E) | -   [cu                           |
| -   [cudaq::                      | daq::scalar_operator::is_constant |
| noise_model_type::depolarization2 |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     enumerator)](api/languag      | uages/cpp_api.html#_CPPv4NK5cudaq |
| es/cpp_api.html#_CPPv4N5cudaq16no | 15scalar_operator11is_constantEv) |
| ise_model_type15depolarization2E) | -   [c                            |
| -   [cudaq::noise_m               | udaq::scalar_operator::operator\* |
| odel_type::depolarization_channel |     (C++                          |
|     (C++                          |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|   enumerator)](api/languages/cpp_ | Pv4N5cudaq15scalar_operatormlENSt |
| api.html#_CPPv4N5cudaq16noise_mod | 7complexIdEERK15scalar_operator), |
| el_type22depolarization_channelE) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|  [cudaq::noise_model_type::pauli1 | Pv4N5cudaq15scalar_operatormlENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     enumerator)](a                |     [\[2\]](api/languages/cp      |
| pi/languages/cpp_api.html#_CPPv4N | p_api.html#_CPPv4N5cudaq15scalar_ |
| 5cudaq16noise_model_type6pauli1E) | operatormlEdRK15scalar_operator), |
| -                                 |     [\[3\]](api/languages/cp      |
|  [cudaq::noise_model_type::pauli2 | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormlEdRR15scalar_operator), |
|     enumerator)](a                |     [\[4\]](api/languages         |
| pi/languages/cpp_api.html#_CPPv4N | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| 5cudaq16noise_model_type6pauli2E) | alar_operatormlENSt7complexIdEE), |
| -   [cudaq                        |     [\[5\]](api/languages/cpp     |
| ::noise_model_type::phase_damping | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatormlERK15scalar_operator), |
|     enumerator)](api/langu        |     [\[6\]]                       |
| ages/cpp_api.html#_CPPv4N5cudaq16 | (api/languages/cpp_api.html#_CPPv |
| noise_model_type13phase_dampingE) | 4NKR5cudaq15scalar_operatormlEd), |
| -   [cudaq::noi                   |     [\[7\]](api/language          |
| se_model_type::phase_flip_channel | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormlENSt7complexIdEE), |
|     enumerator)](api/languages/   |     [\[8\]](api/languages/cp      |
| cpp_api.html#_CPPv4N5cudaq16noise | p_api.html#_CPPv4NO5cudaq15scalar |
| _model_type18phase_flip_channelE) | _operatormlERK15scalar_operator), |
| -                                 |     [\[9\                         |
| [cudaq::noise_model_type::unknown | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatormlEd) |
|     enumerator)](ap               | -   [cu                           |
| i/languages/cpp_api.html#_CPPv4N5 | daq::scalar_operator::operator\*= |
| cudaq16noise_model_type7unknownE) |     (C++                          |
| -                                 |     function)](api/languag        |
| [cudaq::noise_model_type::x_error | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormLENSt7complexIdEE), |
|     enumerator)](ap               |     [\[1\]](api/languages/c       |
| i/languages/cpp_api.html#_CPPv4N5 | pp_api.html#_CPPv4N5cudaq15scalar |
| cudaq16noise_model_type7x_errorE) | _operatormLERK15scalar_operator), |
| -                                 |     [\[2                          |
| [cudaq::noise_model_type::y_error | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatormLEd) |
|     enumerator)](ap               | -   [                             |
| i/languages/cpp_api.html#_CPPv4N5 | cudaq::scalar_operator::operator+ |
| cudaq16noise_model_type7y_errorE) |     (C++                          |
| -                                 |     function                      |
| [cudaq::noise_model_type::z_error | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatorplENSt |
|     enumerator)](ap               | 7complexIdEERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[1\                         |
| cudaq16noise_model_type7z_errorE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::num_available_gpus    | Pv4N5cudaq15scalar_operatorplENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     function                      |     [\[2\]](api/languages/cp      |
| )](api/languages/cpp_api.html#_CP | p_api.html#_CPPv4N5cudaq15scalar_ |
| Pv4N5cudaq18num_available_gpusEv) | operatorplEdRK15scalar_operator), |
| -   [cudaq::observe (C++          |     [\[3\]](api/languages/cp      |
|     function)]                    | p_api.html#_CPPv4N5cudaq15scalar_ |
| (api/languages/cpp_api.html#_CPPv | operatorplEdRR15scalar_operator), |
| 4I00DpEN5cudaq7observeENSt6vector |     [\[4\]](api/languages         |
| I14observe_resultEERR13QuantumKer | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| nelRK15SpinOpContainerDpRR4Args), | alar_operatorplENSt7complexIdEE), |
|     [\[1\]](api/languages/cpp_ap  |     [\[5\]](api/languages/cpp     |
| i.html#_CPPv4I0DpEN5cudaq7observe | _api.html#_CPPv4NKR5cudaq15scalar |
| E14observe_resultNSt6size_tERR13Q | _operatorplERK15scalar_operator), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[6\]]                       |
|     [\[                           | (api/languages/cpp_api.html#_CPPv |
| 2\]](api/languages/cpp_api.html#_ | 4NKR5cudaq15scalar_operatorplEd), |
| CPPv4I0DpEN5cudaq7observeE14obser |     [\[7\]]                       |
| ve_resultRK15observe_optionsRR13Q | (api/languages/cpp_api.html#_CPPv |
| uantumKernelRK7spin_opDpRR4Args), | 4NKR5cudaq15scalar_operatorplEv), |
|     [\[3\]](api/lang              |     [\[8\]](api/language          |
| uages/cpp_api.html#_CPPv4I0DpEN5c | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| udaq7observeE14observe_resultRR13 | alar_operatorplENSt7complexIdEE), |
| QuantumKernelRK7spin_opDpRR4Args) |     [\[9\]](api/languages/cp      |
| -   [cudaq::observe_options (C++  | p_api.html#_CPPv4NO5cudaq15scalar |
|     st                            | _operatorplERK15scalar_operator), |
| ruct)](api/languages/cpp_api.html |     [\[10\]                       |
| #_CPPv4N5cudaq15observe_optionsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::observe_result (C++   | v4NO5cudaq15scalar_operatorplEd), |
|                                   |     [\[11\                        |
| class)](api/languages/cpp_api.htm | ]](api/languages/cpp_api.html#_CP |
| l#_CPPv4N5cudaq14observe_resultE) | Pv4NO5cudaq15scalar_operatorplEv) |
| -                                 | -   [c                            |
|    [cudaq::observe_result::counts | udaq::scalar_operator::operator+= |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     function)](api/languag        |
| pp_api.html#_CPPv4N5cudaq14observ | es/cpp_api.html#_CPPv4N5cudaq15sc |
| e_result6countsERK12spin_op_term) | alar_operatorpLENSt7complexIdEE), |
| -   [cudaq::observe_result::dump  |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function)                     | _operatorpLERK15scalar_operator), |
| ](api/languages/cpp_api.html#_CPP |     [\[2                          |
| v4N5cudaq14observe_result4dumpEv) | \]](api/languages/cpp_api.html#_C |
| -   [c                            | PPv4N5cudaq15scalar_operatorpLEd) |
| udaq::observe_result::expectation | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator- |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     function                      |
| html#_CPPv4N5cudaq14observe_resul | )](api/languages/cpp_api.html#_CP |
| t11expectationERK12spin_op_term), | Pv4N5cudaq15scalar_operatormiENSt |
|     [\[1\]](api/la                | 7complexIdEERK15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[1\                         |
| q14observe_result11expectationEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cuda                         | Pv4N5cudaq15scalar_operatormiENSt |
| q::observe_result::id_coefficient | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     function)](api/langu          | p_api.html#_CPPv4N5cudaq15scalar_ |
| ages/cpp_api.html#_CPPv4N5cudaq14 | operatormiEdRK15scalar_operator), |
| observe_result14id_coefficientEv) |     [\[3\]](api/languages/cp      |
| -   [cuda                         | p_api.html#_CPPv4N5cudaq15scalar_ |
| q::observe_result::observe_result | operatormiEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|                                   | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|   function)](api/languages/cpp_ap | alar_operatormiENSt7complexIdEE), |
| i.html#_CPPv4N5cudaq14observe_res |     [\[5\]](api/languages/cpp     |
| ult14observe_resultEdRK7spin_op), | _api.html#_CPPv4NKR5cudaq15scalar |
|     [\[1\]](a                     | _operatormiERK15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[6\]]                       |
| 5cudaq14observe_result14observe_r | (api/languages/cpp_api.html#_CPPv |
| esultEdRK7spin_op13sample_result) | 4NKR5cudaq15scalar_operatormiEd), |
| -                                 |     [\[7\]]                       |
|  [cudaq::observe_result::operator | (api/languages/cpp_api.html#_CPPv |
|     double (C++                   | 4NKR5cudaq15scalar_operatormiEv), |
|     functio                       |     [\[8\]](api/language          |
| n)](api/languages/cpp_api.html#_C | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| PPv4N5cudaq14observe_resultcvdEv) | alar_operatormiENSt7complexIdEE), |
| -                                 |     [\[9\]](api/languages/cp      |
|  [cudaq::observe_result::raw_data | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatormiERK15scalar_operator), |
|     function)](ap                 |     [\[10\]                       |
| i/languages/cpp_api.html#_CPPv4N5 | ](api/languages/cpp_api.html#_CPP |
| cudaq14observe_result8raw_dataEv) | v4NO5cudaq15scalar_operatormiEd), |
| -   [cudaq::operator_handler (C++ |     [\[11\                        |
|     cl                            | ]](api/languages/cpp_api.html#_CP |
| ass)](api/languages/cpp_api.html# | Pv4NO5cudaq15scalar_operatormiEv) |
| _CPPv4N5cudaq16operator_handlerE) | -   [c                            |
| -   [cudaq::optimizable_function  | udaq::scalar_operator::operator-= |
|     (C++                          |     (C++                          |
|     class)                        |     function)](api/languag        |
| ](api/languages/cpp_api.html#_CPP | es/cpp_api.html#_CPPv4N5cudaq15sc |
| v4N5cudaq20optimizable_functionE) | alar_operatormIENSt7complexIdEE), |
| -   [cudaq::optimization_result   |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     type                          | _operatormIERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[2                          |
| Pv4N5cudaq19optimization_resultE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::optimizer (C++        | PPv4N5cudaq15scalar_operatormIEd) |
|     class)](api/languages/cpp_a   | -   [                             |
| pi.html#_CPPv4N5cudaq9optimizerE) | cudaq::scalar_operator::operator/ |
| -   [cudaq::optimizer::optimize   |     (C++                          |
|     (C++                          |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|  function)](api/languages/cpp_api | Pv4N5cudaq15scalar_operatordvENSt |
| .html#_CPPv4N5cudaq9optimizer8opt | 7complexIdEERK15scalar_operator), |
| imizeEKiRR20optimizable_function) |     [\[1\                         |
| -   [cu                           | ]](api/languages/cpp_api.html#_CP |
| daq::optimizer::requiresGradients | Pv4N5cudaq15scalar_operatordvENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     function)](api/la             |     [\[2\]](api/languages/cp      |
| nguages/cpp_api.html#_CPPv4N5cuda | p_api.html#_CPPv4N5cudaq15scalar_ |
| q9optimizer17requiresGradientsEv) | operatordvEdRK15scalar_operator), |
| -   [cudaq::orca (C++             |     [\[3\]](api/languages/cp      |
|     type)](api/languages/         | p_api.html#_CPPv4N5cudaq15scalar_ |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | operatordvEdRR15scalar_operator), |
| -   [cudaq::orca::sample (C++     |     [\[4\]](api/languages         |
|     function)](api/languages/c    | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| pp_api.html#_CPPv4N5cudaq4orca6sa | alar_operatordvENSt7complexIdEE), |
| mpleERNSt6vectorINSt6size_tEEERNS |     [\[5\]](api/languages/cpp     |
| t6vectorINSt6size_tEEERNSt6vector | _api.html#_CPPv4NKR5cudaq15scalar |
| IdEERNSt6vectorIdEEiNSt6size_tE), | _operatordvERK15scalar_operator), |
|     [\[1\]]                       |     [\[6\]]                       |
| (api/languages/cpp_api.html#_CPPv | (api/languages/cpp_api.html#_CPPv |
| 4N5cudaq4orca6sampleERNSt6vectorI | 4NKR5cudaq15scalar_operatordvEd), |
| NSt6size_tEEERNSt6vectorINSt6size |     [\[7\]](api/language          |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::orca::sample_async    | alar_operatordvENSt7complexIdEE), |
|     (C++                          |     [\[8\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4NO5cudaq15scalar |
| function)](api/languages/cpp_api. | _operatordvERK15scalar_operator), |
| html#_CPPv4N5cudaq4orca12sample_a |     [\[9\                         |
| syncERNSt6vectorINSt6size_tEEERNS | ]](api/languages/cpp_api.html#_CP |
| t6vectorINSt6size_tEEERNSt6vector | Pv4NO5cudaq15scalar_operatordvEd) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [c                            |
|     [\[1\]](api/la                | udaq::scalar_operator::operator/= |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q4orca12sample_asyncERNSt6vectorI |     function)](api/languag        |
| NSt6size_tEEERNSt6vectorINSt6size | es/cpp_api.html#_CPPv4N5cudaq15sc |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | alar_operatordVENSt7complexIdEE), |
| -   [cudaq::OrcaRemoteRESTQPU     |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     cla                           | _operatordVERK15scalar_operator), |
| ss)](api/languages/cpp_api.html#_ |     [\[2                          |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::pauli1 (C++           | PPv4N5cudaq15scalar_operatordVEd) |
|     class)](api/languages/cp      | -   [                             |
| p_api.html#_CPPv4N5cudaq6pauli1E) | cudaq::scalar_operator::operator= |
| -                                 |     (C++                          |
|    [cudaq::pauli1::num_parameters |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     member)]                      | _operatoraSERK15scalar_operator), |
| (api/languages/cpp_api.html#_CPPv |     [\[1\]](api/languages/        |
| 4N5cudaq6pauli114num_parametersE) | cpp_api.html#_CPPv4N5cudaq15scala |
| -   [cudaq::pauli1::num_targets   | r_operatoraSERR15scalar_operator) |
|     (C++                          | -   [c                            |
|     membe                         | udaq::scalar_operator::operator== |
| r)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq6pauli111num_targetsE) |     function)](api/languages/c    |
| -   [cudaq::pauli1::pauli1 (C++   | pp_api.html#_CPPv4NK5cudaq15scala |
|     function)](api/languages/cpp_ | r_operatoreqERK15scalar_operator) |
| api.html#_CPPv4N5cudaq6pauli16pau | -   [cudaq:                       |
| li1ERKNSt6vectorIN5cudaq4realEEE) | :scalar_operator::scalar_operator |
| -   [cudaq::pauli2 (C++           |     (C++                          |
|     class)](api/languages/cp      |     func                          |
| p_api.html#_CPPv4N5cudaq6pauli2E) | tion)](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq15scalar_operator15 |
|    [cudaq::pauli2::num_parameters | scalar_operatorENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/langu             |
|     member)]                      | ages/cpp_api.html#_CPPv4N5cudaq15 |
| (api/languages/cpp_api.html#_CPPv | scalar_operator15scalar_operatorE |
| 4N5cudaq6pauli214num_parametersE) | RK15scalar_callbackRRNSt13unorder |
| -   [cudaq::pauli2::num_targets   | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[2\                         |
|     membe                         | ]](api/languages/cpp_api.html#_CP |
| r)](api/languages/cpp_api.html#_C | Pv4N5cudaq15scalar_operator15scal |
| PPv4N5cudaq6pauli211num_targetsE) | ar_operatorERK15scalar_operator), |
| -   [cudaq::pauli2::pauli2 (C++   |     [\[3\]](api/langu             |
|     function)](api/languages/cpp_ | ages/cpp_api.html#_CPPv4N5cudaq15 |
| api.html#_CPPv4N5cudaq6pauli26pau | scalar_operator15scalar_operatorE |
| li2ERKNSt6vectorIN5cudaq4realEEE) | RR15scalar_callbackRRNSt13unorder |
| -   [cudaq::phase_damping (C++    | ed_mapINSt6stringENSt6stringEEE), |
|                                   |     [\[4\                         |
|  class)](api/languages/cpp_api.ht | ]](api/languages/cpp_api.html#_CP |
| ml#_CPPv4N5cudaq13phase_dampingE) | Pv4N5cudaq15scalar_operator15scal |
| -   [cud                          | ar_operatorERR15scalar_operator), |
| aq::phase_damping::num_parameters |     [\[5\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     member)](api/lan              | lar_operator15scalar_operatorEd), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[6\]](api/languag           |
| 13phase_damping14num_parametersE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [                             | alar_operator15scalar_operatorEv) |
| cudaq::phase_damping::num_targets | -   [                             |
|     (C++                          | cudaq::scalar_operator::to_matrix |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq13phase_damping11num_targetsE) |   function)](api/languages/cpp_ap |
| -   [cudaq::phase_flip_channel    | i.html#_CPPv4NK5cudaq15scalar_ope |
|     (C++                          | rator9to_matrixERKNSt13unordered_ |
|     clas                          | mapINSt6stringENSt7complexIdEEEE) |
| s)](api/languages/cpp_api.html#_C | -   [                             |
| PPv4N5cudaq18phase_flip_channelE) | cudaq::scalar_operator::to_string |
| -   [cudaq::p                     |     (C++                          |
| hase_flip_channel::num_parameters |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     member)](api/language         | daq15scalar_operator9to_stringEv) |
| s/cpp_api.html#_CPPv4N5cudaq18pha | -   [cudaq::s                     |
| se_flip_channel14num_parametersE) | calar_operator::\~scalar_operator |
| -   [cudaq                        |     (C++                          |
| ::phase_flip_channel::num_targets |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](api/langu            | PPv4N5cudaq15scalar_operatorD0Ev) |
| ages/cpp_api.html#_CPPv4N5cudaq18 | -   [cudaq::set_noise (C++        |
| phase_flip_channel11num_targetsE) |     function)](api/langu          |
| -   [cudaq::product_op (C++       | ages/cpp_api.html#_CPPv4N5cudaq9s |
|                                   | et_noiseERKN5cudaq11noise_modelE) |
|  class)](api/languages/cpp_api.ht | -   [cudaq::set_random_seed (C++  |
| ml#_CPPv4I0EN5cudaq10product_opE) |     function)](api/               |
| -   [cudaq::product_op::begin     | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq15set_random_seedENSt6size_tE) |
|     functio                       | -   [cudaq::simulation_precision  |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4NK5cudaq10product_op5beginEv) |     enum)                         |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|  [cudaq::product_op::canonicalize | v4N5cudaq20simulation_precisionE) |
|     (C++                          | -   [                             |
|     func                          | cudaq::simulation_precision::fp32 |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq10product_op12canon |     enumerator)](api              |
| icalizeERKNSt3setINSt6size_tEEE), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]](api                   | udaq20simulation_precision4fp32E) |
| /languages/cpp_api.html#_CPPv4N5c | -   [                             |
| udaq10product_op12canonicalizeEv) | cudaq::simulation_precision::fp64 |
| -   [                             |     (C++                          |
| cudaq::product_op::const_iterator |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     struct)](api/                 | udaq20simulation_precision4fp64E) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::SimulationState (C++  |
| daq10product_op14const_iteratorE) |     c                             |
| -   [cudaq::product_o             | lass)](api/languages/cpp_api.html |
| p::const_iterator::const_iterator | #_CPPv4N5cudaq15SimulationStateE) |
|     (C++                          | -   [                             |
|     fu                            | cudaq::SimulationState::precision |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq10product_op14con |     enum)](api                    |
| st_iterator14const_iteratorEPK10p | /languages/cpp_api.html#_CPPv4N5c |
| roduct_opI9HandlerTyENSt6size_tE) | udaq15SimulationState9precisionE) |
| -   [cudaq::produ                 | -   [cudaq:                       |
| ct_op::const_iterator::operator!= | :SimulationState::precision::fp32 |
|     (C++                          |     (C++                          |
|     fun                           |     enumerator)](api/lang         |
| ction)](api/languages/cpp_api.htm | uages/cpp_api.html#_CPPv4N5cudaq1 |
| l#_CPPv4NK5cudaq10product_op14con | 5SimulationState9precision4fp32E) |
| st_iteratorneERK14const_iterator) | -   [cudaq:                       |
| -   [cudaq::produ                 | :SimulationState::precision::fp64 |
| ct_op::const_iterator::operator\* |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     function)](api/lang           | uages/cpp_api.html#_CPPv4N5cudaq1 |
| uages/cpp_api.html#_CPPv4NK5cudaq | 5SimulationState9precision4fp64E) |
| 10product_op14const_iteratormlEv) | -                                 |
| -   [cudaq::produ                 |   [cudaq::SimulationState::Tensor |
| ct_op::const_iterator::operator++ |     (C++                          |
|     (C++                          |     struct)](                     |
|     function)](api/lang           | api/languages/cpp_api.html#_CPPv4 |
| uages/cpp_api.html#_CPPv4N5cudaq1 | N5cudaq15SimulationState6TensorE) |
| 0product_op14const_iteratorppEi), | -   [cudaq::spin_handler (C++     |
|     [\[1\]](api/lan               |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq |   class)](api/languages/cpp_api.h |
| 10product_op14const_iteratorppEv) | tml#_CPPv4N5cudaq12spin_handlerE) |
| -   [cudaq::produc                | -   [cudaq:                       |
| t_op::const_iterator::operator\-- | :spin_handler::to_diagonal_matrix |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](api/la             |
| uages/cpp_api.html#_CPPv4N5cudaq1 | nguages/cpp_api.html#_CPPv4NK5cud |
| 0product_op14const_iteratormmEi), | aq12spin_handler18to_diagonal_mat |
|     [\[1\]](api/lan               | rixERNSt13unordered_mapINSt6size_ |
| guages/cpp_api.html#_CPPv4N5cudaq | tENSt7int64_tEEERKNSt13unordered_ |
| 10product_op14const_iteratormmEv) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::produc                | -                                 |
| t_op::const_iterator::operator-\> |   [cudaq::spin_handler::to_matrix |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function                      |
| guages/cpp_api.html#_CPPv4N5cudaq | )](api/languages/cpp_api.html#_CP |
| 10product_op14const_iteratorptEv) | Pv4N5cudaq12spin_handler9to_matri |
|                                   | xERKNSt6stringENSt7complexIdEEb), |
|                                   |     [\[1                          |
|                                   | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4NK5cudaq12spin_handler9to_mat |
|                                   | rixERNSt13unordered_mapINSt6size_ |
|                                   | tENSt7int64_tEEERKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [cuda                         |
|                                   | q::spin_handler::to_sparse_matrix |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq12spin_handler16to_sparse_matr |
|                                   | ixERKNSt6stringENSt7complexIdEEb) |
|                                   | -                                 |
|                                   |   [cudaq::spin_handler::to_string |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq12spin_handler9to_stringEb) |
|                                   | -                                 |
|                                   |   [cudaq::spin_handler::unique_id |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq12spin_handler9unique_idEv) |
|                                   | -   [cudaq::spin_op (C++          |
|                                   |     type)](api/languages/cpp      |
|                                   | _api.html#_CPPv4N5cudaq7spin_opE) |
|                                   | -   [cudaq::spin_op_term (C++     |
|                                   |                                   |
|                                   |    type)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq12spin_op_termE) |
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
|     -   [(in module               | -   [distribute_terms             |
|         cudaq.operators.cus       |     (cu                           |
| tom)](api/languages/python_api.ht | daq.operators.boson.BosonOperator |
| ml#cudaq.operators.custom.define) |     attribute)](api/languages/pyt |
| -   [degrees                      | hon_api.html#cudaq.operators.boso |
|     (cu                           | n.BosonOperator.distribute_terms) |
| daq.operators.boson.BosonOperator |     -   [(cudaq.                  |
|     property)](api/lang           | operators.fermion.FermionOperator |
| uages/python_api.html#cudaq.opera |                                   |
| tors.boson.BosonOperator.degrees) | attribute)](api/languages/python_ |
|     -   [(cudaq.ope               | api.html#cudaq.operators.fermion. |
| rators.boson.BosonOperatorElement | FermionOperator.distribute_terms) |
|                                   |     -                             |
|        property)](api/languages/p |  [(cudaq.operators.MatrixOperator |
| ython_api.html#cudaq.operators.bo |         attribute)](api/language  |
| son.BosonOperatorElement.degrees) | s/python_api.html#cudaq.operators |
|     -   [(cudaq.                  | .MatrixOperator.distribute_terms) |
| operators.boson.BosonOperatorTerm |     -   [(                        |
|         property)](api/language   | cudaq.operators.spin.SpinOperator |
| s/python_api.html#cudaq.operators |                                   |
| .boson.BosonOperatorTerm.degrees) |       attribute)](api/languages/p |
|     -   [(cudaq.                  | ython_api.html#cudaq.operators.sp |
| operators.fermion.FermionOperator | in.SpinOperator.distribute_terms) |
|         property)](api/language   |     -   [(cuda                    |
| s/python_api.html#cudaq.operators | q.operators.spin.SpinOperatorTerm |
| .fermion.FermionOperator.degrees) |                                   |
|     -   [(cudaq.operato           |   attribute)](api/languages/pytho |
| rs.fermion.FermionOperatorElement | n_api.html#cudaq.operators.spin.S |
|                                   | pinOperatorTerm.distribute_terms) |
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
| -   [Depolarization1 (class in    | languages/python_api.html#cudaq.o |
|     cudaq)](api/languages/pytho   | perators.MatrixOperatorTerm.dump) |
| n_api.html#cudaq.Depolarization1) |     -   [(                        |
| -   [Depolarization2 (class in    | cudaq.operators.spin.SpinOperator |
|     cudaq)](api/languages/pytho   |         attribute)](api           |
| n_api.html#cudaq.Depolarization2) | /languages/python_api.html#cudaq. |
| -   [DepolarizationChannel (class | operators.spin.SpinOperator.dump) |
|     in                            |     -   [(cuda                    |
|                                   | q.operators.spin.SpinOperatorTerm |
| cudaq)](api/languages/python_api. |         attribute)](api/lan       |
| html#cudaq.DepolarizationChannel) | guages/python_api.html#cudaq.oper |
| -   [depth (cudaq.Resources       | ators.spin.SpinOperatorTerm.dump) |
|                                   |     -   [(cudaq.Resources         |
|    property)](api/languages/pytho |                                   |
| n_api.html#cudaq.Resources.depth) |    attribute)](api/languages/pyth |
| -   [depth_for_arity              | on_api.html#cudaq.Resources.dump) |
|     (cudaq.Resources              |     -   [(cudaq.SampleResult      |
|     attribut                      |                                   |
| e)](api/languages/python_api.html | attribute)](api/languages/python_ |
| #cudaq.Resources.depth_for_arity) | api.html#cudaq.SampleResult.dump) |
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

+-----------------------------------------------------------------------+
| -   [launch_args_required() (cudaq.PyKernelDecorator                  |
|     method)](api/la                                                   |
| nguages/python_api.html#cudaq.PyKernelDecorator.launch_args_required) |
| -   [LBFGS (class in                                                  |
|     cud                                                               |
| aq.optimizers)](api/languages/python_api.html#cudaq.optimizers.LBFGS) |
| -   [left_multiply (cudaq.SuperOperator                               |
|     attribu                                                           |
| te)](api/languages/python_api.html#cudaq.SuperOperator.left_multiply) |
| -   [left_right_multiply (cudaq.SuperOperator                         |
|     attribute)](a                                                     |
| pi/languages/python_api.html#cudaq.SuperOperator.left_right_multiply) |
| -   [LOW_WEIGHT_BIAS (cudaq.ptsbe.ShotAllocationType                  |
|     attribute)](api/lang                                              |
| uages/python_api.html#cudaq.ptsbe.ShotAllocationType.LOW_WEIGHT_BIAS) |
| -   [lower_bounds (cudaq.optimizers.Adam                              |
|     propert                                                           |
| y)](api/languages/python_api.html#cudaq.optimizers.Adam.lower_bounds) |
|     -   [(cudaq.optimizers.COBYLA                                     |
|         property)                                                     |
| ](api/languages/python_api.html#cudaq.optimizers.COBYLA.lower_bounds) |
|     -   [(cudaq.optimizers.GradientDescent                            |
|         property)](api/lan                                            |
| guages/python_api.html#cudaq.optimizers.GradientDescent.lower_bounds) |
|     -   [(cudaq.optimizers.LBFGS                                      |
|         property                                                      |
| )](api/languages/python_api.html#cudaq.optimizers.LBFGS.lower_bounds) |
|     -   [(cudaq.optimizers.NelderMead                                 |
|         property)](ap                                                 |
| i/languages/python_api.html#cudaq.optimizers.NelderMead.lower_bounds) |
|     -   [(cudaq.optimizers.SGD                                        |
|         proper                                                        |
| ty)](api/languages/python_api.html#cudaq.optimizers.SGD.lower_bounds) |
|     -   [(cudaq.optimizers.SPSA                                       |
|         propert                                                       |
| y)](api/languages/python_api.html#cudaq.optimizers.SPSA.lower_bounds) |
+-----------------------------------------------------------------------+

## M {#M}

+-----------------------------------+-----------------------------------+
| -   [make_kernel() (in module     | -   [merge_kernel()               |
|     cudaq)](api/languages/p       |     (cudaq.PyKernelDecorator      |
| ython_api.html#cudaq.make_kernel) |     method)](a                    |
| -   [MatrixOperator (class in     | pi/languages/python_api.html#cuda |
|     cudaq.operato                 | q.PyKernelDecorator.merge_kernel) |
| rs)](api/languages/python_api.htm | -   [merge_quake_source()         |
| l#cudaq.operators.MatrixOperator) |     (cudaq.PyKernelDecorator      |
| -   [MatrixOperatorElement (class |     method)](api/lan              |
|     in                            | guages/python_api.html#cudaq.PyKe |
|     cudaq.operators)](ap          | rnelDecorator.merge_quake_source) |
| i/languages/python_api.html#cudaq | -   [min_degree                   |
| .operators.MatrixOperatorElement) |     (cu                           |
| -   [MatrixOperatorTerm (class in | daq.operators.boson.BosonOperator |
|     cudaq.operators)]             |     property)](api/languag        |
| (api/languages/python_api.html#cu | es/python_api.html#cudaq.operator |
| daq.operators.MatrixOperatorTerm) | s.boson.BosonOperator.min_degree) |
| -   [max_degree                   |     -   [(cudaq.                  |
|     (cu                           | operators.boson.BosonOperatorTerm |
| daq.operators.boson.BosonOperator |                                   |
|     property)](api/languag        |        property)](api/languages/p |
| es/python_api.html#cudaq.operator | ython_api.html#cudaq.operators.bo |
| s.boson.BosonOperator.max_degree) | son.BosonOperatorTerm.min_degree) |
|     -   [(cudaq.                  |     -   [(cudaq.                  |
| operators.boson.BosonOperatorTerm | operators.fermion.FermionOperator |
|                                   |                                   |
|        property)](api/languages/p |        property)](api/languages/p |
| ython_api.html#cudaq.operators.bo | ython_api.html#cudaq.operators.fe |
| son.BosonOperatorTerm.max_degree) | rmion.FermionOperator.min_degree) |
|     -   [(cudaq.                  |     -   [(cudaq.oper              |
| operators.fermion.FermionOperator | ators.fermion.FermionOperatorTerm |
|                                   |                                   |
|        property)](api/languages/p |    property)](api/languages/pytho |
| ython_api.html#cudaq.operators.fe | n_api.html#cudaq.operators.fermio |
| rmion.FermionOperator.max_degree) | n.FermionOperatorTerm.min_degree) |
|     -   [(cudaq.oper              |     -                             |
| ators.fermion.FermionOperatorTerm |  [(cudaq.operators.MatrixOperator |
|                                   |         property)](api/la         |
|    property)](api/languages/pytho | nguages/python_api.html#cudaq.ope |
| n_api.html#cudaq.operators.fermio | rators.MatrixOperator.min_degree) |
| n.FermionOperatorTerm.max_degree) |     -   [(c                       |
|     -                             | udaq.operators.MatrixOperatorTerm |
|  [(cudaq.operators.MatrixOperator |         property)](api/langua     |
|         property)](api/la         | ges/python_api.html#cudaq.operato |
| nguages/python_api.html#cudaq.ope | rs.MatrixOperatorTerm.min_degree) |
| rators.MatrixOperator.max_degree) |     -   [(                        |
|     -   [(c                       | cudaq.operators.spin.SpinOperator |
| udaq.operators.MatrixOperatorTerm |         property)](api/langu      |
|         property)](api/langua     | ages/python_api.html#cudaq.operat |
| ges/python_api.html#cudaq.operato | ors.spin.SpinOperator.min_degree) |
| rs.MatrixOperatorTerm.max_degree) |     -   [(cuda                    |
|     -   [(                        | q.operators.spin.SpinOperatorTerm |
| cudaq.operators.spin.SpinOperator |         property)](api/languages  |
|         property)](api/langu      | /python_api.html#cudaq.operators. |
| ages/python_api.html#cudaq.operat | spin.SpinOperatorTerm.min_degree) |
| ors.spin.SpinOperator.max_degree) | -   [minimal_eigenvalue           |
|     -   [(cuda                    |     (cudaq.ComplexMatrix          |
| q.operators.spin.SpinOperatorTerm |     attribute)](api               |
|         property)](api/languages  | /languages/python_api.html#cudaq. |
| /python_api.html#cudaq.operators. | ComplexMatrix.minimal_eigenvalue) |
| spin.SpinOperatorTerm.max_degree) | -   module                        |
| -   [max_iterations               |     -   [cudaq](api/langua        |
|     (cudaq.optimizers.Adam        | ges/python_api.html#module-cudaq) |
|     property)](a                  |     -                             |
| pi/languages/python_api.html#cuda |    [cudaq.boson](api/languages/py |
| q.optimizers.Adam.max_iterations) | thon_api.html#module-cudaq.boson) |
|     -   [(cudaq.optimizers.COBYLA |     -   [                         |
|         property)](api            | cudaq.fermion](api/languages/pyth |
| /languages/python_api.html#cudaq. | on_api.html#module-cudaq.fermion) |
| optimizers.COBYLA.max_iterations) |     -   [cudaq.operators.cu       |
|     -   [                         | stom](api/languages/python_api.ht |
| (cudaq.optimizers.GradientDescent | ml#module-cudaq.operators.custom) |
|         property)](api/language   |                                   |
| s/python_api.html#cudaq.optimizer |  -   [cudaq.spin](api/languages/p |
| s.GradientDescent.max_iterations) | ython_api.html#module-cudaq.spin) |
|     -   [(cudaq.optimizers.LBFGS  | -   [most_probable                |
|         property)](ap             |     (cudaq.SampleResult           |
| i/languages/python_api.html#cudaq |     attribute                     |
| .optimizers.LBFGS.max_iterations) | )](api/languages/python_api.html# |
|                                   | cudaq.SampleResult.most_probable) |
| -   [(cudaq.optimizers.NelderMead | -   [multi_qubit_depth            |
|         property)](api/lan        |     (cudaq.Resources              |
| guages/python_api.html#cudaq.opti |     property)                     |
| mizers.NelderMead.max_iterations) | ](api/languages/python_api.html#c |
|     -   [(cudaq.optimizers.SGD    | udaq.Resources.multi_qubit_depth) |
|         property)](               | -   [multi_qubit_gate_count       |
| api/languages/python_api.html#cud |     (cudaq.Resources              |
| aq.optimizers.SGD.max_iterations) |     property)](api                |
|     -   [(cudaq.optimizers.SPSA   | /languages/python_api.html#cudaq. |
|         property)](a              | Resources.multi_qubit_gate_count) |
| pi/languages/python_api.html#cuda | -   [multiplicity                 |
| q.optimizers.SPSA.max_iterations) |     (cudaq.ptsbe.KrausTrajectory  |
| -   [mdiag_sparse_matrix (C++     |     property)](api/l              |
|     type)](api/languages/cpp_api. | anguages/python_api.html#cudaq.pt |
| html#_CPPv419mdiag_sparse_matrix) | sbe.KrausTrajectory.multiplicity) |
| -   [measurement_counts           |                                   |
|     (cudaq.ptsbe.KrausTrajectory  |                                   |
|     property)](api/languag        |                                   |
| es/python_api.html#cudaq.ptsbe.Kr |                                   |
| ausTrajectory.measurement_counts) |                                   |
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
| -   [to_dict (cudaq.Resources     | ython_api.html#cudaq.operators.sp |
|                                   | in.SpinOperatorElement.to_string) |
| attribute)](api/languages/python_ |     -   [(cuda                    |
| api.html#cudaq.Resources.to_dict) | q.operators.spin.SpinOperatorTerm |
| -   [to_json                      |         attribute)](api/language  |
|     (                             | s/python_api.html#cudaq.operators |
| cudaq.operators.spin.SpinOperator | .spin.SpinOperatorTerm.to_string) |
|     attribute)](api/la            | -   [TraceInstruction (class in   |
| nguages/python_api.html#cudaq.ope |     cudaq.p                       |
| rators.spin.SpinOperator.to_json) | tsbe)](api/languages/python_api.h |
|     -   [(cuda                    | tml#cudaq.ptsbe.TraceInstruction) |
| q.operators.spin.SpinOperatorTerm | -   [TraceInstructionType (class  |
|         attribute)](api/langua    |     in                            |
| ges/python_api.html#cudaq.operato |     cudaq.ptsbe                   |
| rs.spin.SpinOperatorTerm.to_json) | )](api/languages/python_api.html# |
| -   [to_json()                    | cudaq.ptsbe.TraceInstructionType) |
|     (cudaq.PyKernelDecorator      | -   [trajectories                 |
|     metho                         |                                   |
| d)](api/languages/python_api.html |   (cudaq.ptsbe.PTSBEExecutionData |
| #cudaq.PyKernelDecorator.to_json) |     property)](api/lang           |
| -   [to_matrix                    | uages/python_api.html#cudaq.ptsbe |
|     (cu                           | .PTSBEExecutionData.trajectories) |
| daq.operators.boson.BosonOperator | -   [trajectory_id                |
|     attribute)](api/langua        |     (cudaq.ptsbe.KrausTrajectory  |
| ges/python_api.html#cudaq.operato |     property)](api/la             |
| rs.boson.BosonOperator.to_matrix) | nguages/python_api.html#cudaq.pts |
|     -   [(cudaq.ope               | be.KrausTrajectory.trajectory_id) |
| rators.boson.BosonOperatorElement | -   [translate() (in module       |
|                                   |     cudaq)](api/languages         |
|     attribute)](api/languages/pyt | /python_api.html#cudaq.translate) |
| hon_api.html#cudaq.operators.boso | -   [trim                         |
| n.BosonOperatorElement.to_matrix) |     (cu                           |
|     -   [(cudaq.                  | daq.operators.boson.BosonOperator |
| operators.boson.BosonOperatorTerm |     attribute)](api/l             |
|                                   | anguages/python_api.html#cudaq.op |
|        attribute)](api/languages/ | erators.boson.BosonOperator.trim) |
| python_api.html#cudaq.operators.b |     -   [(cudaq.                  |
| oson.BosonOperatorTerm.to_matrix) | operators.fermion.FermionOperator |
|     -   [(cudaq.                  |         attribute)](api/langu     |
| operators.fermion.FermionOperator | ages/python_api.html#cudaq.operat |
|                                   | ors.fermion.FermionOperator.trim) |
|        attribute)](api/languages/ |     -                             |
| python_api.html#cudaq.operators.f |  [(cudaq.operators.MatrixOperator |
| ermion.FermionOperator.to_matrix) |         attribute)](              |
|     -   [(cudaq.operato           | api/languages/python_api.html#cud |
| rs.fermion.FermionOperatorElement | aq.operators.MatrixOperator.trim) |
|                                   |     -   [(                        |
| attribute)](api/languages/python_ | cudaq.operators.spin.SpinOperator |
| api.html#cudaq.operators.fermion. |         attribute)](api           |
| FermionOperatorElement.to_matrix) | /languages/python_api.html#cudaq. |
|     -   [(cudaq.oper              | operators.spin.SpinOperator.trim) |
| ators.fermion.FermionOperatorTerm | -   [type                         |
|                                   |     (c                            |
|    attribute)](api/languages/pyth | udaq.ptsbe.ShotAllocationStrategy |
| on_api.html#cudaq.operators.fermi |     property)](api/               |
| on.FermionOperatorTerm.to_matrix) | languages/python_api.html#cudaq.p |
|     -                             | tsbe.ShotAllocationStrategy.type) |
|  [(cudaq.operators.MatrixOperator |     -                             |
|         attribute)](api/l         |    [(cudaq.ptsbe.TraceInstruction |
| anguages/python_api.html#cudaq.op |         property)                 |
| erators.MatrixOperator.to_matrix) | ](api/languages/python_api.html#c |
|     -   [(cuda                    | udaq.ptsbe.TraceInstruction.type) |
| q.operators.MatrixOperatorElement | -   [type_to_str()                |
|         attribute)](api/language  |     (cudaq.PyKernelDecorator      |
| s/python_api.html#cudaq.operators |     static                        |
| .MatrixOperatorElement.to_matrix) |     method)](                     |
|     -   [(c                       | api/languages/python_api.html#cud |
| udaq.operators.MatrixOperatorTerm | aq.PyKernelDecorator.type_to_str) |
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
