::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4418
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
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](using/examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
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
            -   [Multi-QPU + Other
                Backends](using/backends/sims/mqpusims.html#multi-qpu-other-backends){.reference
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
| -   [canonicalize                 | -   [cudaq::product_op::dump (C++ |
|     (cu                           |     functi                        |
| daq.operators.boson.BosonOperator | on)](api/languages/cpp_api.html#_ |
|     attribute)](api/languages     | CPPv4NK5cudaq10product_op4dumpEv) |
| /python_api.html#cudaq.operators. | -   [cudaq::product_op::end (C++  |
| boson.BosonOperator.canonicalize) |     funct                         |
|     -   [(cudaq.                  | ion)](api/languages/cpp_api.html# |
| operators.boson.BosonOperatorTerm | _CPPv4NK5cudaq10product_op3endEv) |
|                                   | -   [c                            |
|     attribute)](api/languages/pyt | udaq::product_op::get_coefficient |
| hon_api.html#cudaq.operators.boso |     (C++                          |
| n.BosonOperatorTerm.canonicalize) |     function)](api/lan            |
|     -   [(cudaq.                  | guages/cpp_api.html#_CPPv4NK5cuda |
| operators.fermion.FermionOperator | q10product_op15get_coefficientEv) |
|                                   | -                                 |
|     attribute)](api/languages/pyt |   [cudaq::product_op::get_term_id |
| hon_api.html#cudaq.operators.ferm |     (C++                          |
| ion.FermionOperator.canonicalize) |     function)](api                |
|     -   [(cudaq.oper              | /languages/cpp_api.html#_CPPv4NK5 |
| ators.fermion.FermionOperatorTerm | cudaq10product_op11get_term_idEv) |
|                                   | -                                 |
| attribute)](api/languages/python_ |   [cudaq::product_op::is_identity |
| api.html#cudaq.operators.fermion. |     (C++                          |
| FermionOperatorTerm.canonicalize) |     function)](api                |
|     -                             | /languages/cpp_api.html#_CPPv4NK5 |
|  [(cudaq.operators.MatrixOperator | cudaq10product_op11is_identityEv) |
|         attribute)](api/lang      | -   [cudaq::product_op::num_ops   |
| uages/python_api.html#cudaq.opera |     (C++                          |
| tors.MatrixOperator.canonicalize) |     function)                     |
|     -   [(c                       | ](api/languages/cpp_api.html#_CPP |
| udaq.operators.MatrixOperatorTerm | v4NK5cudaq10product_op7num_opsEv) |
|         attribute)](api/language  | -                                 |
| s/python_api.html#cudaq.operators |    [cudaq::product_op::operator\* |
| .MatrixOperatorTerm.canonicalize) |     (C++                          |
|     -   [(                        |     function)](api/languages/     |
| cudaq.operators.spin.SpinOperator | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|         attribute)](api/languag   | oduct_opmlE10product_opI1TERK15sc |
| es/python_api.html#cudaq.operator | alar_operatorRK10product_opI1TE), |
| s.spin.SpinOperator.canonicalize) |     [\[1\]](api/languages/        |
|     -   [(cuda                    | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| q.operators.spin.SpinOperatorTerm | oduct_opmlE10product_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|       attribute)](api/languages/p |     [\[2\]](api/languages/        |
| ython_api.html#cudaq.operators.sp | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| in.SpinOperatorTerm.canonicalize) | oduct_opmlE10product_opI1TERR15sc |
| -   [captured_variables()         | alar_operatorRK10product_opI1TE), |
|     (cudaq.PyKernelDecorator      |     [\[3\]](api/languages/        |
|     method)](api/lan              | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| guages/python_api.html#cudaq.PyKe | oduct_opmlE10product_opI1TERR15sc |
| rnelDecorator.captured_variables) | alar_operatorRR10product_opI1TE), |
| -   [CentralDifference (class in  |     [\[4\]](api/                  |
|     cudaq.gradients)              | languages/cpp_api.html#_CPPv4I0EN |
| ](api/languages/python_api.html#c | 5cudaq10product_opmlE6sum_opI1TER |
| udaq.gradients.CentralDifference) | K15scalar_operatorRK6sum_opI1TE), |
| -   [channel                      |     [\[5\]](api/                  |
|     (cudaq.ptsbe.TraceInstruction | languages/cpp_api.html#_CPPv4I0EN |
|     property)](a                  | 5cudaq10product_opmlE6sum_opI1TER |
| pi/languages/python_api.html#cuda | K15scalar_operatorRR6sum_opI1TE), |
| q.ptsbe.TraceInstruction.channel) |     [\[6\]](api/                  |
| -   [circuit_location             | languages/cpp_api.html#_CPPv4I0EN |
|     (cudaq.ptsbe.KrausSelection   | 5cudaq10product_opmlE6sum_opI1TER |
|     property)](api/lang           | R15scalar_operatorRK6sum_opI1TE), |
| uages/python_api.html#cudaq.ptsbe |     [\[7\]](api/                  |
| .KrausSelection.circuit_location) | languages/cpp_api.html#_CPPv4I0EN |
| -   [clear (cudaq.Resources       | 5cudaq10product_opmlE6sum_opI1TER |
|                                   | R15scalar_operatorRR6sum_opI1TE), |
|   attribute)](api/languages/pytho |     [\[8\]](api/languages         |
| n_api.html#cudaq.Resources.clear) | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     -   [(cudaq.SampleResult      | duct_opmlERK6sum_opI9HandlerTyE), |
|         a                         |     [\[9\]](api/languages/cpp_a   |
| ttribute)](api/languages/python_a | pi.html#_CPPv4NKR5cudaq10product_ |
| pi.html#cudaq.SampleResult.clear) | opmlERK10product_opI9HandlerTyE), |
| -   [COBYLA (class in             |     [\[10\]](api/language         |
|     cudaq.o                       | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ptimizers)](api/languages/python_ | roduct_opmlERK15scalar_operator), |
| api.html#cudaq.optimizers.COBYLA) |     [\[11\]](api/languages/cpp_a  |
| -   [coefficient                  | pi.html#_CPPv4NKR5cudaq10product_ |
|     (cudaq.                       | opmlERR10product_opI9HandlerTyE), |
| operators.boson.BosonOperatorTerm |     [\[12\]](api/language         |
|     property)](api/languages/py   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| thon_api.html#cudaq.operators.bos | roduct_opmlERR15scalar_operator), |
| on.BosonOperatorTerm.coefficient) |     [\[13\]](api/languages/cpp_   |
|     -   [(cudaq.oper              | api.html#_CPPv4NO5cudaq10product_ |
| ators.fermion.FermionOperatorTerm | opmlERK10product_opI9HandlerTyE), |
|                                   |     [\[14\]](api/languag          |
|   property)](api/languages/python | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _api.html#cudaq.operators.fermion | roduct_opmlERK15scalar_operator), |
| .FermionOperatorTerm.coefficient) |     [\[15\]](api/languages/cpp_   |
|     -   [(c                       | api.html#_CPPv4NO5cudaq10product_ |
| udaq.operators.MatrixOperatorTerm | opmlERR10product_opI9HandlerTyE), |
|         property)](api/languag    |     [\[16\]](api/langua           |
| es/python_api.html#cudaq.operator | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| s.MatrixOperatorTerm.coefficient) | product_opmlERR15scalar_operator) |
|     -   [(cuda                    | -                                 |
| q.operators.spin.SpinOperatorTerm |   [cudaq::product_op::operator\*= |
|         property)](api/languages/ |     (C++                          |
| python_api.html#cudaq.operators.s |     function)](api/languages/cpp  |
| pin.SpinOperatorTerm.coefficient) | _api.html#_CPPv4N5cudaq10product_ |
| -   [col_count                    | opmLERK10product_opI9HandlerTyE), |
|     (cudaq.KrausOperator          |     [\[1\]](api/langua            |
|     prope                         | ges/cpp_api.html#_CPPv4N5cudaq10p |
| rty)](api/languages/python_api.ht | roduct_opmLERK15scalar_operator), |
| ml#cudaq.KrausOperator.col_count) |     [\[2\]](api/languages/cp      |
| -   [compile()                    | p_api.html#_CPPv4N5cudaq10product |
|     (cudaq.PyKernelDecorator      | _opmLERR10product_opI9HandlerTyE) |
|     metho                         | -   [cudaq::product_op::operator+ |
| d)](api/languages/python_api.html |     (C++                          |
| #cudaq.PyKernelDecorator.compile) |     function)](api/langu          |
| -   [ComplexMatrix (class in      | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cudaq)](api/languages/pyt     | q10product_opplE6sum_opI1TERK15sc |
| hon_api.html#cudaq.ComplexMatrix) | alar_operatorRK10product_opI1TE), |
| -   [compute                      |     [\[1\]](api/                  |
|     (                             | languages/cpp_api.html#_CPPv4I0EN |
| cudaq.gradients.CentralDifference | 5cudaq10product_opplE6sum_opI1TER |
|     attribute)](api/la            | K15scalar_operatorRK6sum_opI1TE), |
| nguages/python_api.html#cudaq.gra |     [\[2\]](api/langu             |
| dients.CentralDifference.compute) | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     -   [(                        | q10product_opplE6sum_opI1TERK15sc |
| cudaq.gradients.ForwardDifference | alar_operatorRR10product_opI1TE), |
|         attribute)](api/la        |     [\[3\]](api/                  |
| nguages/python_api.html#cudaq.gra | languages/cpp_api.html#_CPPv4I0EN |
| dients.ForwardDifference.compute) | 5cudaq10product_opplE6sum_opI1TER |
|     -                             | K15scalar_operatorRR6sum_opI1TE), |
|  [(cudaq.gradients.ParameterShift |     [\[4\]](api/langu             |
|         attribute)](api           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| /languages/python_api.html#cudaq. | q10product_opplE6sum_opI1TERR15sc |
| gradients.ParameterShift.compute) | alar_operatorRK10product_opI1TE), |
| -   [const()                      |     [\[5\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|   (cudaq.operators.ScalarOperator | 5cudaq10product_opplE6sum_opI1TER |
|     class                         | R15scalar_operatorRK6sum_opI1TE), |
|     method)](a                    |     [\[6\]](api/langu             |
| pi/languages/python_api.html#cuda | ages/cpp_api.html#_CPPv4I0EN5cuda |
| q.operators.ScalarOperator.const) | q10product_opplE6sum_opI1TERR15sc |
| -   [controls                     | alar_operatorRR10product_opI1TE), |
|     (cudaq.ptsbe.TraceInstruction |     [\[7\]](api/                  |
|     property)](ap                 | languages/cpp_api.html#_CPPv4I0EN |
| i/languages/python_api.html#cudaq | 5cudaq10product_opplE6sum_opI1TER |
| .ptsbe.TraceInstruction.controls) | R15scalar_operatorRR6sum_opI1TE), |
| -   [copy                         |     [\[8\]](api/languages/cpp_a   |
|     (cu                           | pi.html#_CPPv4NKR5cudaq10product_ |
| daq.operators.boson.BosonOperator | opplERK10product_opI9HandlerTyE), |
|     attribute)](api/l             |     [\[9\]](api/language          |
| anguages/python_api.html#cudaq.op | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| erators.boson.BosonOperator.copy) | roduct_opplERK15scalar_operator), |
|     -   [(cudaq.                  |     [\[10\]](api/languages/       |
| operators.boson.BosonOperatorTerm | cpp_api.html#_CPPv4NKR5cudaq10pro |
|         attribute)](api/langu     | duct_opplERK6sum_opI9HandlerTyE), |
| ages/python_api.html#cudaq.operat |     [\[11\]](api/languages/cpp_a  |
| ors.boson.BosonOperatorTerm.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [(cudaq.                  | opplERR10product_opI9HandlerTyE), |
| operators.fermion.FermionOperator |     [\[12\]](api/language         |
|         attribute)](api/langu     | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ages/python_api.html#cudaq.operat | roduct_opplERR15scalar_operator), |
| ors.fermion.FermionOperator.copy) |     [\[13\]](api/languages/       |
|     -   [(cudaq.oper              | cpp_api.html#_CPPv4NKR5cudaq10pro |
| ators.fermion.FermionOperatorTerm | duct_opplERR6sum_opI9HandlerTyE), |
|         attribute)](api/languages |     [\[                           |
| /python_api.html#cudaq.operators. | 14\]](api/languages/cpp_api.html# |
| fermion.FermionOperatorTerm.copy) | _CPPv4NKR5cudaq10product_opplEv), |
|     -                             |     [\[15\]](api/languages/cpp_   |
|  [(cudaq.operators.MatrixOperator | api.html#_CPPv4NO5cudaq10product_ |
|         attribute)](              | opplERK10product_opI9HandlerTyE), |
| api/languages/python_api.html#cud |     [\[16\]](api/languag          |
| aq.operators.MatrixOperator.copy) | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     -   [(c                       | roduct_opplERK15scalar_operator), |
| udaq.operators.MatrixOperatorTerm |     [\[17\]](api/languages        |
|         attribute)](api/          | /cpp_api.html#_CPPv4NO5cudaq10pro |
| languages/python_api.html#cudaq.o | duct_opplERK6sum_opI9HandlerTyE), |
| perators.MatrixOperatorTerm.copy) |     [\[18\]](api/languages/cpp_   |
|     -   [(                        | api.html#_CPPv4NO5cudaq10product_ |
| cudaq.operators.spin.SpinOperator | opplERR10product_opI9HandlerTyE), |
|         attribute)](api           |     [\[19\]](api/languag          |
| /languages/python_api.html#cudaq. | es/cpp_api.html#_CPPv4NO5cudaq10p |
| operators.spin.SpinOperator.copy) | roduct_opplERR15scalar_operator), |
|     -   [(cuda                    |     [\[20\]](api/languages        |
| q.operators.spin.SpinOperatorTerm | /cpp_api.html#_CPPv4NO5cudaq10pro |
|         attribute)](api/lan       | duct_opplERR6sum_opI9HandlerTyE), |
| guages/python_api.html#cudaq.oper |     [                             |
| ators.spin.SpinOperatorTerm.copy) | \[21\]](api/languages/cpp_api.htm |
| -   [count (cudaq.Resources       | l#_CPPv4NO5cudaq10product_opplEv) |
|                                   | -   [cudaq::product_op::operator- |
|   attribute)](api/languages/pytho |     (C++                          |
| n_api.html#cudaq.Resources.count) |     function)](api/langu          |
|     -   [(cudaq.SampleResult      | ages/cpp_api.html#_CPPv4I0EN5cuda |
|         a                         | q10product_opmiE6sum_opI1TERK15sc |
| ttribute)](api/languages/python_a | alar_operatorRK10product_opI1TE), |
| pi.html#cudaq.SampleResult.count) |     [\[1\]](api/                  |
| -   [count_controls               | languages/cpp_api.html#_CPPv4I0EN |
|     (cudaq.Resources              | 5cudaq10product_opmiE6sum_opI1TER |
|     attribu                       | K15scalar_operatorRK6sum_opI1TE), |
| te)](api/languages/python_api.htm |     [\[2\]](api/langu             |
| l#cudaq.Resources.count_controls) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [count_instructions           | q10product_opmiE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|   (cudaq.ptsbe.PTSBEExecutionData |     [\[3\]](api/                  |
|     attribute)](api/languages/    | languages/cpp_api.html#_CPPv4I0EN |
| python_api.html#cudaq.ptsbe.PTSBE | 5cudaq10product_opmiE6sum_opI1TER |
| ExecutionData.count_instructions) | K15scalar_operatorRR6sum_opI1TE), |
| -   [counts (cudaq.ObserveResult  |     [\[4\]](api/langu             |
|     att                           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ribute)](api/languages/python_api | q10product_opmiE6sum_opI1TERR15sc |
| .html#cudaq.ObserveResult.counts) | alar_operatorRK10product_opI1TE), |
| -   [csr_spmatrix (C++            |     [\[5\]](api/                  |
|     type)](api/languages/c        | languages/cpp_api.html#_CPPv4I0EN |
| pp_api.html#_CPPv412csr_spmatrix) | 5cudaq10product_opmiE6sum_opI1TER |
| -   cudaq                         | R15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/langua       |     [\[6\]](api/langu             |
| ges/python_api.html#module-cudaq) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq (C++                   | q10product_opmiE6sum_opI1TERR15sc |
|     type)](api/lan                | alar_operatorRR10product_opI1TE), |
| guages/cpp_api.html#_CPPv45cudaq) |     [\[7\]](api/                  |
| -   [cudaq.apply_noise() (in      | languages/cpp_api.html#_CPPv4I0EN |
|     module                        | 5cudaq10product_opmiE6sum_opI1TER |
|     cudaq)](api/languages/python_ | R15scalar_operatorRR6sum_opI1TE), |
| api.html#cudaq.cudaq.apply_noise) |     [\[8\]](api/languages/cpp_a   |
| -   cudaq.boson                   | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [module](api/languages/py | opmiERK10product_opI9HandlerTyE), |
| thon_api.html#module-cudaq.boson) |     [\[9\]](api/language          |
| -   cudaq.fermion                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opmiERK15scalar_operator), |
|   -   [module](api/languages/pyth |     [\[10\]](api/languages/       |
| on_api.html#module-cudaq.fermion) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   cudaq.operators.custom        | duct_opmiERK6sum_opI9HandlerTyE), |
|     -   [mo                       |     [\[11\]](api/languages/cpp_a  |
| dule](api/languages/python_api.ht | pi.html#_CPPv4NKR5cudaq10product_ |
| ml#module-cudaq.operators.custom) | opmiERR10product_opI9HandlerTyE), |
| -   cudaq.spin                    |     [\[12\]](api/language         |
|     -   [module](api/languages/p  | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ython_api.html#module-cudaq.spin) | roduct_opmiERR15scalar_operator), |
| -   [cudaq::amplitude_damping     |     [\[13\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     cla                           | duct_opmiERR6sum_opI9HandlerTyE), |
| ss)](api/languages/cpp_api.html#_ |     [\[                           |
| CPPv4N5cudaq17amplitude_dampingE) | 14\]](api/languages/cpp_api.html# |
| -                                 | _CPPv4NKR5cudaq10product_opmiEv), |
| [cudaq::amplitude_damping_channel |     [\[15\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     class)](api                   | opmiERK10product_opI9HandlerTyE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[16\]](api/languag          |
| udaq25amplitude_damping_channelE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::amplitud              | roduct_opmiERK15scalar_operator), |
| e_damping_channel::num_parameters |     [\[17\]](api/languages        |
|     (C++                          | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     member)](api/languages/cpp_a  | duct_opmiERK6sum_opI9HandlerTyE), |
| pi.html#_CPPv4N5cudaq25amplitude_ |     [\[18\]](api/languages/cpp_   |
| damping_channel14num_parametersE) | api.html#_CPPv4NO5cudaq10product_ |
| -   [cudaq::ampli                 | opmiERR10product_opI9HandlerTyE), |
| tude_damping_channel::num_targets |     [\[19\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     member)](api/languages/cp     | roduct_opmiERR15scalar_operator), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [\[20\]](api/languages        |
| de_damping_channel11num_targetsE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::AnalogRemoteRESTQPU   | duct_opmiERR6sum_opI9HandlerTyE), |
|     (C++                          |     [                             |
|     class                         | \[21\]](api/languages/cpp_api.htm |
| )](api/languages/cpp_api.html#_CP | l#_CPPv4NO5cudaq10product_opmiEv) |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | -   [cudaq::product_op::operator/ |
| -   [cudaq::apply_noise (C++      |     (C++                          |
|     function)](api/               |     function)](api/language       |
| languages/cpp_api.html#_CPPv4I0Dp | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| EN5cudaq11apply_noiseEvDpRR4Args) | roduct_opdvERK15scalar_operator), |
| -   [cudaq::async_result (C++     |     [\[1\]](api/language          |
|     c                             | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| lass)](api/languages/cpp_api.html | roduct_opdvERR15scalar_operator), |
| #_CPPv4I0EN5cudaq12async_resultE) |     [\[2\]](api/languag           |
| -   [cudaq::async_result::get     | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opdvERK15scalar_operator), |
|     functi                        |     [\[3\]](api/langua            |
| on)](api/languages/cpp_api.html#_ | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| CPPv4N5cudaq12async_result3getEv) | product_opdvERR15scalar_operator) |
| -   [cudaq::async_sample_result   | -                                 |
|     (C++                          |    [cudaq::product_op::operator/= |
|     type                          |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/langu          |
| Pv4N5cudaq19async_sample_resultE) | ages/cpp_api.html#_CPPv4N5cudaq10 |
| -   [cudaq::BaseRemoteRESTQPU     | product_opdVERK15scalar_operator) |
|     (C++                          | -   [cudaq::product_op::operator= |
|     cla                           |     (C++                          |
| ss)](api/languages/cpp_api.html#_ |     function)](api/la             |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | nguages/cpp_api.html#_CPPv4I0_NSt |
| -                                 | 11enable_if_tIXaantNSt7is_sameI1T |
|    [cudaq::BaseRemoteSimulatorQPU | 9HandlerTyE5valueENSt16is_constru |
|     (C++                          | ctibleI9HandlerTy1TE5valueEEbEEEN |
|     class)](                      | 5cudaq10product_opaSER10product_o |
| api/languages/cpp_api.html#_CPPv4 | pI9HandlerTyERK10product_opI1TE), |
| N5cudaq22BaseRemoteSimulatorQPUE) |     [\[1\]](api/languages/cpp     |
| -   [cudaq::bit_flip_channel (C++ | _api.html#_CPPv4N5cudaq10product_ |
|     cl                            | opaSERK10product_opI9HandlerTyE), |
| ass)](api/languages/cpp_api.html# |     [\[2\]](api/languages/cp      |
| _CPPv4N5cudaq16bit_flip_channelE) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq:                       | _opaSERR10product_opI9HandlerTyE) |
| :bit_flip_channel::num_parameters | -                                 |
|     (C++                          |    [cudaq::product_op::operator== |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16b |     function)](api/languages/cpp  |
| it_flip_channel14num_parametersE) | _api.html#_CPPv4NK5cudaq10product |
| -   [cud                          | _opeqERK10product_opI9HandlerTyE) |
| aq::bit_flip_channel::num_targets | -                                 |
|     (C++                          |  [cudaq::product_op::operator\[\] |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](ap                 |
| 16bit_flip_channel11num_targetsE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::boson_handler (C++    | 5cudaq10product_opixENSt6size_tE) |
|                                   | -                                 |
|  class)](api/languages/cpp_api.ht |    [cudaq::product_op::product_op |
| ml#_CPPv4N5cudaq13boson_handlerE) |     (C++                          |
| -   [cudaq::boson_op (C++         |     function)](api/languages/c    |
|     type)](api/languages/cpp_     | pp_api.html#_CPPv4I0_NSt11enable_ |
| api.html#_CPPv4N5cudaq8boson_opE) | if_tIXaaNSt7is_sameI9HandlerTy14m |
| -   [cudaq::boson_op_term (C++    | atrix_handlerE5valueEaantNSt7is_s |
|                                   | ameI1T9HandlerTyE5valueENSt16is_c |
|   type)](api/languages/cpp_api.ht | onstructibleI9HandlerTy1TE5valueE |
| ml#_CPPv4N5cudaq13boson_op_termE) | EbEEEN5cudaq10product_op10product |
| -   [cudaq::CodeGenConfig (C++    | _opERK10product_opI1TERKN14matrix |
|                                   | _handler20commutation_behaviorE), |
| struct)](api/languages/cpp_api.ht |                                   |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::commutation_relations | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|     (C++                          | tNSt7is_sameI1T9HandlerTyE5valueE |
|     struct)]                      | NSt16is_constructibleI9HandlerTy1 |
| (api/languages/cpp_api.html#_CPPv | TE5valueEEbEEEN5cudaq10product_op |
| 4N5cudaq21commutation_relationsE) | 10product_opERK10product_opI1TE), |
| -   [cudaq::complex (C++          |                                   |
|     type)](api/languages/cpp      |   [\[2\]](api/languages/cpp_api.h |
| _api.html#_CPPv4N5cudaq7complexE) | tml#_CPPv4N5cudaq10product_op10pr |
| -   [cudaq::complex_matrix (C++   | oduct_opENSt6size_tENSt6size_tE), |
|                                   |     [\[3\]](api/languages/cp      |
| class)](api/languages/cpp_api.htm | p_api.html#_CPPv4N5cudaq10product |
| l#_CPPv4N5cudaq14complex_matrixE) | _op10product_opENSt7complexIdEE), |
| -                                 |     [\[4\]](api/l                 |
|   [cudaq::complex_matrix::adjoint | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq10product_op10product_opERK10pr |
|     function)](a                  | oduct_opI9HandlerTyENSt6size_tE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[5\]](api/l                 |
| 5cudaq14complex_matrix7adjointEv) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::                      | aq10product_op10product_opERR10pr |
| complex_matrix::diagonal_elements | oduct_opI9HandlerTyENSt6size_tE), |
|     (C++                          |     [\[6\]](api/languages         |
|     function)](api/languages      | /cpp_api.html#_CPPv4N5cudaq10prod |
| /cpp_api.html#_CPPv4NK5cudaq14com | uct_op10product_opERR9HandlerTy), |
| plex_matrix17diagonal_elementsEi) |     [\[7\]](ap                    |
| -   [cudaq::complex_matrix::dump  | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq10product_op10product_opEd), |
|     function)](api/language       |     [\[8\]](a                     |
| s/cpp_api.html#_CPPv4NK5cudaq14co | pi/languages/cpp_api.html#_CPPv4N |
| mplex_matrix4dumpERNSt7ostreamE), | 5cudaq10product_op10product_opEv) |
|     [\[1\]]                       | -   [cuda                         |
| (api/languages/cpp_api.html#_CPPv | q::product_op::to_diagonal_matrix |
| 4NK5cudaq14complex_matrix4dumpEv) |     (C++                          |
| -   [c                            |     function)](api/               |
| udaq::complex_matrix::eigenvalues | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq10product_op18to_diagonal_mat |
|     function)](api/lan            | rixENSt13unordered_mapINSt6size_t |
| guages/cpp_api.html#_CPPv4NK5cuda | ENSt7int64_tEEERKNSt13unordered_m |
| q14complex_matrix11eigenvaluesEv) | apINSt6stringENSt7complexIdEEEEb) |
| -   [cu                           | -   [cudaq::product_op::to_matrix |
| daq::complex_matrix::eigenvectors |     (C++                          |
|     (C++                          |     funct                         |
|     function)](api/lang           | ion)](api/languages/cpp_api.html# |
| uages/cpp_api.html#_CPPv4NK5cudaq | _CPPv4NK5cudaq10product_op9to_mat |
| 14complex_matrix12eigenvectorsEv) | rixENSt13unordered_mapINSt6size_t |
| -   [c                            | ENSt7int64_tEEERKNSt13unordered_m |
| udaq::complex_matrix::exponential | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cu                           |
|     function)](api/la             | daq::product_op::to_sparse_matrix |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14complex_matrix11exponentialEv) |     function)](ap                 |
| -                                 | i/languages/cpp_api.html#_CPPv4NK |
|  [cudaq::complex_matrix::identity | 5cudaq10product_op16to_sparse_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function)](api/languages      | ENSt7int64_tEEERKNSt13unordered_m |
| /cpp_api.html#_CPPv4N5cudaq14comp | apINSt6stringENSt7complexIdEEEEb) |
| lex_matrix8identityEKNSt6size_tE) | -   [cudaq::product_op::to_string |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::kronecker |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/lang           | NK5cudaq10product_op9to_stringEv) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -                                 |
| daq14complex_matrix9kroneckerE14c |  [cudaq::product_op::\~product_op |
| omplex_matrix8Iterable8Iterable), |     (C++                          |
|     [\[1\]](api/l                 |     fu                            |
| anguages/cpp_api.html#_CPPv4N5cud | nction)](api/languages/cpp_api.ht |
| aq14complex_matrix9kroneckerERK14 | ml#_CPPv4N5cudaq10product_opD0Ev) |
| complex_matrixRK14complex_matrix) | -   [cudaq::ptsbe (C++            |
| -   [cudaq::c                     |     type)](api/languages/c        |
| omplex_matrix::minimal_eigenvalue | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
|     (C++                          | -   [cudaq::p                     |
|     function)](api/languages/     | tsbe::ConditionalSamplingStrategy |
| cpp_api.html#_CPPv4NK5cudaq14comp |     (C++                          |
| lex_matrix18minimal_eigenvalueEv) |     class)](api/languag           |
| -   [                             | es/cpp_api.html#_CPPv4N5cudaq5pts |
| cudaq::complex_matrix::operator() | be27ConditionalSamplingStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::C              |
|     function)](api/languages/cpp  | onditionalSamplingStrategy::clone |
| _api.html#_CPPv4N5cudaq14complex_ |     (C++                          |
| matrixclENSt6size_tENSt6size_tE), |                                   |
|     [\[1\]](api/languages/cpp     |    function)](api/languages/cpp_a |
| _api.html#_CPPv4NK5cudaq14complex | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| _matrixclENSt6size_tENSt6size_tE) | ditionalSamplingStrategy5cloneEv) |
| -   [                             | -   [cuda                         |
| cudaq::complex_matrix::operator\* | q::ptsbe::ConditionalSamplingStra |
|     (C++                          | tegy::ConditionalSamplingStrategy |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     function)](api/lang           |
| omplex_matrixmlEN14complex_matrix | uages/cpp_api.html#_CPPv4N5cudaq5 |
| 10value_typeERK14complex_matrix), | ptsbe27ConditionalSamplingStrateg |
|     [\[1\]                        | y27ConditionalSamplingStrategyE19 |
| ](api/languages/cpp_api.html#_CPP | TrajectoryPredicateNSt8uint64_tE) |
| v4N5cudaq14complex_matrixmlERK14c | -                                 |
| omplex_matrixRK14complex_matrix), |   [cudaq::ptsbe::ConditionalSampl |
|                                   | ingStrategy::generateTrajectories |
|  [\[2\]](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14complex_matrixm |     function)](api/language       |
| lERK14complex_matrixRKNSt6vectorI | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| N14complex_matrix10value_typeEEE) | be27ConditionalSamplingStrategy20 |
| -                                 | generateTrajectoriesENSt4spanIKN6 |
| [cudaq::complex_matrix::operator+ | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     function                      | ConditionalSamplingStrategy::name |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq14complex_matrixplERK14 |     function)](api/languages/cpp_ |
| complex_matrixRK14complex_matrix) | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| -                                 | nditionalSamplingStrategy4nameEv) |
| [cudaq::complex_matrix::operator- | -   [cudaq:                       |
|     (C++                          | :ptsbe::ConditionalSamplingStrate |
|     function                      | gy::\~ConditionalSamplingStrategy |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq14complex_matrixmiERK14 |     function)](api/languages/     |
| complex_matrixRK14complex_matrix) | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| -   [cu                           | 7ConditionalSamplingStrategyD0Ev) |
| daq::complex_matrix::operator\[\] | -                                 |
|     (C++                          | [cudaq::ptsbe::detail::NoisePoint |
|                                   |     (C++                          |
|  function)](api/languages/cpp_api |     struct)](a                    |
| .html#_CPPv4N5cudaq14complex_matr | pi/languages/cpp_api.html#_CPPv4N |
| ixixERKNSt6vectorINSt6size_tEEE), | 5cudaq5ptsbe6detail10NoisePointE) |
|     [\[1\]](api/languages/cpp_api | -   [cudaq::p                     |
| .html#_CPPv4NK5cudaq14complex_mat | tsbe::detail::NoisePoint::channel |
| rixixERKNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq::complex_matrix::power |     member)](api/langu            |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     function)]                    | tsbe6detail10NoisePoint7channelE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::ptsbe::det            |
| 4N5cudaq14complex_matrix5powerEi) | ail::NoisePoint::circuit_location |
| -                                 |     (C++                          |
|  [cudaq::complex_matrix::set_zero |     member)](api/languages/cpp_a  |
|     (C++                          | pi.html#_CPPv4N5cudaq5ptsbe6detai |
|     function)](ap                 | l10NoisePoint16circuit_locationE) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq::p                     |
| cudaq14complex_matrix8set_zeroEv) | tsbe::detail::NoisePoint::op_name |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::to_string |     member)](api/langu            |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     function)](api/               | tsbe6detail10NoisePoint7op_nameE) |
| languages/cpp_api.html#_CPPv4NK5c | -   [cudaq::                      |
| udaq14complex_matrix9to_stringEv) | ptsbe::detail::NoisePoint::qubits |
| -   [                             |     (C++                          |
| cudaq::complex_matrix::value_type |     member)](api/lang             |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     type)](api/                   | ptsbe6detail10NoisePoint6qubitsE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::                      |
| daq14complex_matrix10value_typeE) | ptsbe::ExhaustiveSamplingStrategy |
| -   [cudaq::contrib (C++          |     (C++                          |
|     type)](api/languages/cpp      |     class)](api/langua            |
| _api.html#_CPPv4N5cudaq7contribE) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq::contrib::draw (C++    | sbe26ExhaustiveSamplingStrategyE) |
|     function)                     | -   [cudaq::ptsbe::               |
| ](api/languages/cpp_api.html#_CPP | ExhaustiveSamplingStrategy::clone |
| v4I0DpEN5cudaq7contrib4drawENSt6s |     (C++                          |
| tringERR13QuantumKernelDpRR4Args) |     function)](api/languages/cpp_ |
| -                                 | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| [cudaq::contrib::get_unitary_cmat | haustiveSamplingStrategy5cloneEv) |
|     (C++                          | -   [cu                           |
|     function)](api/languages/cp   | daq::ptsbe::ExhaustiveSamplingStr |
| p_api.html#_CPPv4I0DpEN5cudaq7con | ategy::ExhaustiveSamplingStrategy |
| trib16get_unitary_cmatE14complex_ |     (C++                          |
| matrixRR13QuantumKernelDpRR4Args) |     function)](api/la             |
| -   [cudaq::CusvState (C++        | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q5ptsbe26ExhaustiveSamplingStrate |
|    class)](api/languages/cpp_api. | gy26ExhaustiveSamplingStrategyEv) |
| html#_CPPv4I0EN5cudaq9CusvStateE) | -                                 |
| -   [cudaq::depolarization1 (C++  |    [cudaq::ptsbe::ExhaustiveSampl |
|     c                             | ingStrategy::generateTrajectories |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15depolarization1E) |     function)](api/languag        |
| -   [cudaq::depolarization2 (C++  | es/cpp_api.html#_CPPv4NK5cudaq5pt |
|     c                             | sbe26ExhaustiveSamplingStrategy20 |
| lass)](api/languages/cpp_api.html | generateTrajectoriesENSt4spanIKN6 |
| #_CPPv4N5cudaq15depolarization2E) | detail10NoisePointEEENSt6size_tE) |
| -   [cudaq:                       | -   [cudaq::ptsbe:                |
| :depolarization2::depolarization2 | :ExhaustiveSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/languages/cpp  |
| p_api.html#_CPPv4N5cudaq15depolar | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| ization215depolarization2EK4real) | xhaustiveSamplingStrategy4nameEv) |
| -   [cudaq                        | -   [cuda                         |
| ::depolarization2::num_parameters | q::ptsbe::ExhaustiveSamplingStrat |
|     (C++                          | egy::\~ExhaustiveSamplingStrategy |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     function)](api/languages      |
| depolarization214num_parametersE) | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| -   [cu                           | 26ExhaustiveSamplingStrategyD0Ev) |
| daq::depolarization2::num_targets | -   [cuda                         |
|     (C++                          | q::ptsbe::OrderedSamplingStrategy |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     class)](api/lan               |
| q15depolarization211num_targetsE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -                                 | 5ptsbe23OrderedSamplingStrategyE) |
|    [cudaq::depolarization_channel | -   [cudaq::ptsb                  |
|     (C++                          | e::OrderedSamplingStrategy::clone |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languages/c    |
| N5cudaq22depolarization_channelE) | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| -   [cudaq::depol                 | 3OrderedSamplingStrategy5cloneEv) |
| arization_channel::num_parameters | -   [cudaq::ptsbe::OrderedSampl   |
|     (C++                          | ingStrategy::generateTrajectories |
|     member)](api/languages/cp     |     (C++                          |
| p_api.html#_CPPv4N5cudaq22depolar |     function)](api/lang           |
| ization_channel14num_parametersE) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cudaq::de                    | 5ptsbe23OrderedSamplingStrategy20 |
| polarization_channel::num_targets | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     member)](api/languages        | -   [cudaq::pts                   |
| /cpp_api.html#_CPPv4N5cudaq22depo | be::OrderedSamplingStrategy::name |
| larization_channel11num_targetsE) |     (C++                          |
| -   [cudaq::details (C++          |     function)](api/languages/     |
|     type)](api/languages/cpp      | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| _api.html#_CPPv4N5cudaq7detailsE) | 23OrderedSamplingStrategy4nameEv) |
| -   [cudaq::details::future (C++  | -                                 |
|                                   |    [cudaq::ptsbe::OrderedSampling |
|  class)](api/languages/cpp_api.ht | Strategy::OrderedSamplingStrategy |
| ml#_CPPv4N5cudaq7details6futureE) |     (C++                          |
| -                                 |     function)](                   |
|   [cudaq::details::future::future | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5ptsbe23OrderedSamplingStr |
|     functio                       | ategy23OrderedSamplingStrategyEv) |
| n)](api/languages/cpp_api.html#_C | -                                 |
| PPv4N5cudaq7details6future6future |  [cudaq::ptsbe::OrderedSamplingSt |
| ERNSt6vectorI3JobEERNSt6stringERN | rategy::\~OrderedSamplingStrategy |
| St3mapINSt6stringENSt6stringEEE), |     (C++                          |
|     [\[1\]](api/lang              |     function)](api/langua         |
| uages/cpp_api.html#_CPPv4N5cudaq7 | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| details6future6futureERR6future), | sbe23OrderedSamplingStrategyD0Ev) |
|     [\[2\]]                       | -   [cudaq::pts                   |
| (api/languages/cpp_api.html#_CPPv | be::ProbabilisticSamplingStrategy |
| 4N5cudaq7details6future6futureEv) |     (C++                          |
| -   [cu                           |     class)](api/languages         |
| daq::details::kernel_builder_base | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     (C++                          | 29ProbabilisticSamplingStrategyE) |
|     class)](api/l                 | -   [cudaq::ptsbe::Pro            |
| anguages/cpp_api.html#_CPPv4N5cud | babilisticSamplingStrategy::clone |
| aq7details19kernel_builder_baseE) |     (C++                          |
| -   [cudaq::details::             |                                   |
| kernel_builder_base::operator\<\< |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4NK5cudaq5ptsbe29Proba |
|     function)](api/langua         | bilisticSamplingStrategy5cloneEv) |
| ges/cpp_api.html#_CPPv4N5cudaq7de | -                                 |
| tails19kernel_builder_baselsERNSt | [cudaq::ptsbe::ProbabilisticSampl |
| 7ostreamERK19kernel_builder_base) | ingStrategy::generateTrajectories |
| -   [                             |     (C++                          |
| cudaq::details::KernelBuilderType |     function)](api/languages/     |
|     (C++                          | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     class)](api                   | 29ProbabilisticSamplingStrategy20 |
| /languages/cpp_api.html#_CPPv4N5c | generateTrajectoriesENSt4spanIKN6 |
| udaq7details17KernelBuilderTypeE) | detail10NoisePointEEENSt6size_tE) |
| -   [cudaq::d                     | -   [cudaq::ptsbe::Pr             |
| etails::KernelBuilderType::create | obabilisticSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP |   function)](api/languages/cpp_ap |
| v4N5cudaq7details17KernelBuilderT | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| ype6createEPN4mlir11MLIRContextE) | abilisticSamplingStrategy4nameEv) |
| -   [cudaq::details::Ker          | -   [cudaq::p                     |
| nelBuilderType::KernelBuilderType | tsbe::ProbabilisticSamplingStrate |
|     (C++                          | gy::ProbabilisticSamplingStrategy |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     function)]                    |
| details17KernelBuilderType17Kerne | (api/languages/cpp_api.html#_CPPv |
| lBuilderTypeERRNSt8functionIFN4ml | 4N5cudaq5ptsbe29ProbabilisticSamp |
| ir4TypeEPN4mlir11MLIRContextEEEE) | lingStrategy29ProbabilisticSampli |
| -   [cudaq::diag_matrix_callback  | ngStrategyENSt8optionalINSt8uint6 |
|     (C++                          | 4_tEEENSt8optionalINSt6size_tEEE) |
|     class)                        | -   [cudaq::pts                   |
| ](api/languages/cpp_api.html#_CPP | be::ProbabilisticSamplingStrategy |
| v4N5cudaq20diag_matrix_callbackE) | ::\~ProbabilisticSamplingStrategy |
| -   [cudaq::dyn (C++              |     (C++                          |
|     member)](api/languages        |     function)](api/languages/cp   |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| -   [cudaq::ExecutionContext (C++ | robabilisticSamplingStrategyD0Ev) |
|     cl                            | -                                 |
| ass)](api/languages/cpp_api.html# | [cudaq::ptsbe::PTSBEExecutionData |
| _CPPv4N5cudaq16ExecutionContextE) |     (C++                          |
| -   [cudaq                        |     struct)](ap                   |
| ::ExecutionContext::amplitudeMaps | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq5ptsbe18PTSBEExecutionDataE) |
|     member)](api/langu            | -   [cudaq::ptsbe::PTSBE          |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ExecutionData::count_instructions |
| ExecutionContext13amplitudeMapsE) |     (C++                          |
| -   [c                            |     function)](api/l              |
| udaq::ExecutionContext::asyncExec | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq5ptsbe18PTSBEExecutionData18co |
|     member)](api/                 | unt_instructionsE20TraceInstructi |
| languages/cpp_api.html#_CPPv4N5cu | onTypeNSt8optionalINSt6stringEEE) |
| daq16ExecutionContext9asyncExecE) | -   [cudaq::ptsbe::P              |
| -   [cud                          | TSBEExecutionData::get_trajectory |
| aq::ExecutionContext::asyncResult |     (C++                          |
|     (C++                          |     function                      |
|     member)](api/lan              | )](api/languages/cpp_api.html#_CP |
| guages/cpp_api.html#_CPPv4N5cudaq | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| 16ExecutionContext11asyncResultE) | Data14get_trajectoryENSt6size_tE) |
| -   [cudaq:                       | -   [cudaq::ptsbe:                |
| :ExecutionContext::batchIteration | :PTSBEExecutionData::instructions |
|     (C++                          |     (C++                          |
|     member)](api/langua           |     member)](api/languages/cp     |
| ges/cpp_api.html#_CPPv4N5cudaq16E | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| xecutionContext14batchIterationE) | TSBEExecutionData12instructionsE) |
| -   [cudaq::E                     | -   [cudaq::ptsbe:                |
| xecutionContext::canHandleObserve | :PTSBEExecutionData::trajectories |
|     (C++                          |     (C++                          |
|     member)](api/language         |     member)](api/languages/cp     |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| cutionContext16canHandleObserveE) | TSBEExecutionData12trajectoriesE) |
| -   [cudaq::E                     | -   [cudaq::ptsbe::PTSBEOptions   |
| xecutionContext::ExecutionContext |     (C++                          |
|     (C++                          |     struc                         |
|     func                          | t)](api/languages/cpp_api.html#_C |
| tion)](api/languages/cpp_api.html | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| #_CPPv4N5cudaq16ExecutionContext1 | -   [cudaq::ptsbe::PTSB           |
| 6ExecutionContextERKNSt6stringE), | EOptions::include_sequential_data |
|     [\[1\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16Execu |                                   |
| tionContext16ExecutionContextERKN |    member)](api/languages/cpp_api |
| St6stringENSt6size_tENSt6size_tE) | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| -   [cudaq::E                     | ptions23include_sequential_dataE) |
| xecutionContext::expectationValue | -   [cudaq::ptsb                  |
|     (C++                          | e::PTSBEOptions::max_trajectories |
|     member)](api/language         |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     member)](api/languages/       |
| cutionContext16expectationValueE) | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| -   [cudaq::Execu                 | 2PTSBEOptions16max_trajectoriesE) |
| tionContext::explicitMeasurements | -   [cudaq::ptsbe::PT             |
|     (C++                          | SBEOptions::return_execution_data |
|     member)](api/languages/cp     |     (C++                          |
| p_api.html#_CPPv4N5cudaq16Executi |     member)](api/languages/cpp_a  |
| onContext20explicitMeasurementsE) | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| -   [cuda                         | EOptions21return_execution_dataE) |
| q::ExecutionContext::futureResult | -   [cudaq::pts                   |
|     (C++                          | be::PTSBEOptions::shot_allocation |
|     member)](api/lang             |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     member)](api/languages        |
| 6ExecutionContext12futureResultE) | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| -   [cudaq::ExecutionContext      | 12PTSBEOptions15shot_allocationE) |
| ::hasConditionalsOnMeasureResults | -   [cud                          |
|     (C++                          | aq::ptsbe::PTSBEOptions::strategy |
|     mem                           |     (C++                          |
| ber)](api/languages/cpp_api.html# |     member)](api/l                |
| _CPPv4N5cudaq16ExecutionContext31 | anguages/cpp_api.html#_CPPv4N5cud |
| hasConditionalsOnMeasureResultsE) | aq5ptsbe12PTSBEOptions8strategyE) |
| -   [cudaq::Executi               | -   [cudaq::ptsbe::PTSBETrace     |
| onContext::invocationResultBuffer |     (C++                          |
|     (C++                          |     t                             |
|     member)](api/languages/cpp_   | ype)](api/languages/cpp_api.html# |
| api.html#_CPPv4N5cudaq16Execution | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| Context22invocationResultBufferE) | -   [                             |
| -   [cu                           | cudaq::ptsbe::PTSSamplingStrategy |
| daq::ExecutionContext::kernelName |     (C++                          |
|     (C++                          |     class)](api                   |
|     member)](api/la               | /languages/cpp_api.html#_CPPv4N5c |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq5ptsbe19PTSSamplingStrategyE) |
| q16ExecutionContext10kernelNameE) | -   [cudaq::                      |
| -   [cud                          | ptsbe::PTSSamplingStrategy::clone |
| aq::ExecutionContext::kernelTrace |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     member)](api/lan              | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| guages/cpp_api.html#_CPPv4N5cudaq | sbe19PTSSamplingStrategy5cloneEv) |
| 16ExecutionContext11kernelTraceE) | -   [cudaq::ptsbe::PTSSampl       |
| -   [cudaq:                       | ingStrategy::generateTrajectories |
| :ExecutionContext::msm_dimensions |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](api/langua           | languages/cpp_api.html#_CPPv4NK5c |
| ges/cpp_api.html#_CPPv4N5cudaq16E | udaq5ptsbe19PTSSamplingStrategy20 |
| xecutionContext14msm_dimensionsE) | generateTrajectoriesENSt4spanIKN6 |
| -   [cudaq::                      | detail10NoisePointEEENSt6size_tE) |
| ExecutionContext::msm_prob_err_id | -   [cudaq:                       |
|     (C++                          | :ptsbe::PTSSamplingStrategy::name |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api/langua         |
| ecutionContext15msm_prob_err_idE) | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| -   [cudaq::Ex                    | tsbe19PTSSamplingStrategy4nameEv) |
| ecutionContext::msm_probabilities | -   [cudaq::ptsbe::PTSSampli      |
|     (C++                          | ngStrategy::\~PTSSamplingStrategy |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq16Exec |     function)](api/la             |
| utionContext17msm_probabilitiesE) | nguages/cpp_api.html#_CPPv4N5cuda |
| -                                 | q5ptsbe19PTSSamplingStrategyD0Ev) |
|    [cudaq::ExecutionContext::name | -   [cudaq::ptsbe::sample (C++    |
|     (C++                          |                                   |
|     member)]                      |  function)](api/languages/cpp_api |
| (api/languages/cpp_api.html#_CPPv | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| 4N5cudaq16ExecutionContext4nameE) | mpleE13sample_resultRK14sample_op |
| -   [cu                           | tionsRR13QuantumKernelDpRR4Args), |
| daq::ExecutionContext::noiseModel |     [\[1\]](api                   |
|     (C++                          | /languages/cpp_api.html#_CPPv4I0D |
|     member)](api/la               | pEN5cudaq5ptsbe6sampleE13sample_r |
| nguages/cpp_api.html#_CPPv4N5cuda | esultRKN5cudaq11noise_modelENSt6s |
| q16ExecutionContext10noiseModelE) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [cudaq::Exe                   | -   [cudaq::ptsbe::sample_async   |
| cutionContext::numberTrajectories |     (C++                          |
|     (C++                          |     function)](a                  |
|     member)](api/languages/       | pi/languages/cpp_api.html#_CPPv4I |
| cpp_api.html#_CPPv4N5cudaq16Execu | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| tionContext18numberTrajectoriesE) | 9async_sample_resultRK14sample_op |
| -   [c                            | tionsRR13QuantumKernelDpRR4Args), |
| udaq::ExecutionContext::optResult |     [\[1\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4I0DpEN5cudaq5pts |
|     member)](api/                 | be12sample_asyncE19async_sample_r |
| languages/cpp_api.html#_CPPv4N5cu | esultRKN5cudaq11noise_modelENSt6s |
| daq16ExecutionContext9optResultE) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [cudaq::Execu                 | -   [cudaq::ptsbe::sample_options |
| tionContext::overlapComputeStates |     (C++                          |
|     (C++                          |     struct)                       |
|     member)](api/languages/cp     | ](api/languages/cpp_api.html#_CPP |
| p_api.html#_CPPv4N5cudaq16Executi | v4N5cudaq5ptsbe14sample_optionsE) |
| onContext20overlapComputeStatesE) | -   [cudaq::ptsbe::sample_result  |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::overlapResult |     class                         |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)](api/langu            | Pv4N5cudaq5ptsbe13sample_resultE) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [cudaq::pts                   |
| ExecutionContext13overlapResultE) | be::sample_result::execution_data |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::qpuId |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
|     member)](                     | 3sample_result14execution_dataEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::ptsbe::               |
| N5cudaq16ExecutionContext5qpuIdE) | sample_result::has_execution_data |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::registerNames |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     member)](api/langu            | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ple_result18has_execution_dataEv) |
| ExecutionContext13registerNamesE) | -   [cudaq::pt                    |
| -   [cu                           | sbe::sample_result::sample_result |
| daq::ExecutionContext::reorderIdx |     (C++                          |
|     (C++                          |     function)](api/l              |
|     member)](api/la               | anguages/cpp_api.html#_CPPv4N5cud |
| nguages/cpp_api.html#_CPPv4N5cuda | aq5ptsbe13sample_result13sample_r |
| q16ExecutionContext10reorderIdxE) | esultERRN5cudaq13sample_resultE), |
| -                                 |                                   |
|  [cudaq::ExecutionContext::result |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq5ptsbe13sample_re |
|     member)](a                    | sult13sample_resultERRN5cudaq13sa |
| pi/languages/cpp_api.html#_CPPv4N | mple_resultE18PTSBEExecutionData) |
| 5cudaq16ExecutionContext6resultE) | -   [cudaq::ptsbe::               |
| -                                 | sample_result::set_execution_data |
|   [cudaq::ExecutionContext::shots |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](                     | languages/cpp_api.html#_CPPv4N5cu |
| api/languages/cpp_api.html#_CPPv4 | daq5ptsbe13sample_result18set_exe |
| N5cudaq16ExecutionContext5shotsE) | cution_dataE18PTSBEExecutionData) |
| -   [cudaq::                      | -   [cud                          |
| ExecutionContext::simulationState | aq::ptsbe::ShotAllocationStrategy |
|     (C++                          |     (C++                          |
|     member)](api/languag          |     struct)](using                |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | /examples/ptsbe.html#_CPPv4N5cuda |
| ecutionContext15simulationStateE) | q5ptsbe22ShotAllocationStrategyE) |
| -                                 | -   [cudaq::ptsbe::ShotAllocatio  |
|    [cudaq::ExecutionContext::spin | nStrategy::ShotAllocationStrategy |
|     (C++                          |     (C++                          |
|     member)]                      |     function)                     |
| (api/languages/cpp_api.html#_CPPv | ](using/examples/ptsbe.html#_CPPv |
| 4N5cudaq16ExecutionContext4spinE) | 4N5cudaq5ptsbe22ShotAllocationStr |
| -   [cudaq::                      | ategy22ShotAllocationStrategyE4Ty |
| ExecutionContext::totalIterations | pedNSt8optionalINSt8uint64_tEEE), |
|     (C++                          |     [\[1\                         |
|     member)](api/languag          | ]](using/examples/ptsbe.html#_CPP |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | v4N5cudaq5ptsbe22ShotAllocationSt |
| ecutionContext15totalIterationsE) | rategy22ShotAllocationStrategyEv) |
| -   [cudaq::Executio              | -   [cudaq::pt                    |
| nContext::warnedNamedMeasurements | sbe::ShotAllocationStrategy::Type |
|     (C++                          |     (C++                          |
|     member)](api/languages/cpp_a  |     enum)](using/exam             |
| pi.html#_CPPv4N5cudaq16ExecutionC | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| ontext23warnedNamedMeasurementsE) | be22ShotAllocationStrategy4TypeE) |
| -   [cudaq::ExecutionResult (C++  | -   [cudaq::ptsbe::ShotAllocatio  |
|     st                            | nStrategy::Type::HIGH_WEIGHT_BIAS |
| ruct)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15ExecutionResultE) |     enumerat                      |
| -   [cud                          | or)](using/examples/ptsbe.html#_C |
| aq::ExecutionResult::appendResult | PPv4N5cudaq5ptsbe22ShotAllocation |
|     (C++                          | Strategy4Type16HIGH_WEIGHT_BIASE) |
|     functio                       | -   [cudaq::ptsbe::ShotAllocati   |
| n)](api/languages/cpp_api.html#_C | onStrategy::Type::LOW_WEIGHT_BIAS |
| PPv4N5cudaq15ExecutionResult12app |     (C++                          |
| endResultENSt6stringENSt6size_tE) |     enumera                       |
| -   [cu                           | tor)](using/examples/ptsbe.html#_ |
| daq::ExecutionResult::deserialize | CPPv4N5cudaq5ptsbe22ShotAllocatio |
|     (C++                          | nStrategy4Type15LOW_WEIGHT_BIASE) |
|     function)                     | -   [cudaq::ptsbe::ShotAlloc      |
| ](api/languages/cpp_api.html#_CPP | ationStrategy::Type::PROPORTIONAL |
| v4N5cudaq15ExecutionResult11deser |     (C++                          |
| ializeERNSt6vectorINSt6size_tEEE) |     enum                          |
| -   [cudaq:                       | erator)](using/examples/ptsbe.htm |
| :ExecutionResult::ExecutionResult | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
|     (C++                          | tionStrategy4Type12PROPORTIONALE) |
|     functio                       | -   [cudaq::ptsbe::Shot           |
| n)](api/languages/cpp_api.html#_C | AllocationStrategy::Type::UNIFORM |
| PPv4N5cudaq15ExecutionResult15Exe |     (C++                          |
| cutionResultE16CountsDictionary), |                                   |
|     [\[1\]](api/lan               |   enumerator)](using/examples/pts |
| guages/cpp_api.html#_CPPv4N5cudaq | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| 15ExecutionResult15ExecutionResul | AllocationStrategy4Type7UNIFORME) |
| tE16CountsDictionaryNSt6stringE), | -                                 |
|     [\[2\                         |   [cudaq::ptsbe::TraceInstruction |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     struct)](                     |
| utionResultE16CountsDictionaryd), | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq5ptsbe16TraceInstructionE) |
|    [\[3\]](api/languages/cpp_api. | -   [cudaq:                       |
| html#_CPPv4N5cudaq15ExecutionResu | :ptsbe::TraceInstruction::channel |
| lt15ExecutionResultENSt6stringE), |     (C++                          |
|     [\[4\                         |     member)](api/lang             |
| ]](api/languages/cpp_api.html#_CP | uages/cpp_api.html#_CPPv4N5cudaq5 |
| Pv4N5cudaq15ExecutionResult15Exec | ptsbe16TraceInstruction7channelE) |
| utionResultERK15ExecutionResult), | -   [cudaq::                      |
|     [\[5\]](api/language          | ptsbe::TraceInstruction::controls |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     (C++                          |
| cutionResult15ExecutionResultEd), |     member)](api/langu            |
|     [\[6\]](api/languag           | ages/cpp_api.html#_CPPv4N5cudaq5p |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | tsbe16TraceInstruction8controlsE) |
| ecutionResult15ExecutionResultEv) | -   [cud                          |
| -   [                             | aq::ptsbe::TraceInstruction::name |
| cudaq::ExecutionResult::operator= |     (C++                          |
|     (C++                          |     member)](api/l                |
|     function)](api/languages/     | anguages/cpp_api.html#_CPPv4N5cud |
| cpp_api.html#_CPPv4N5cudaq15Execu | aq5ptsbe16TraceInstruction4nameE) |
| tionResultaSERK15ExecutionResult) | -   [cudaq                        |
| -   [c                            | ::ptsbe::TraceInstruction::params |
| udaq::ExecutionResult::operator== |     (C++                          |
|     (C++                          |     member)](api/lan              |
|     function)](api/languages/c    | guages/cpp_api.html#_CPPv4N5cudaq |
| pp_api.html#_CPPv4NK5cudaq15Execu | 5ptsbe16TraceInstruction6paramsE) |
| tionResulteqERK15ExecutionResult) | -   [cudaq:                       |
| -   [cud                          | :ptsbe::TraceInstruction::targets |
| aq::ExecutionResult::registerName |     (C++                          |
|     (C++                          |     member)](api/lang             |
|     member)](api/lan              | uages/cpp_api.html#_CPPv4N5cudaq5 |
| guages/cpp_api.html#_CPPv4N5cudaq | ptsbe16TraceInstruction7targetsE) |
| 15ExecutionResult12registerNameE) | -   [cudaq::ptsbe::T              |
| -   [cudaq                        | raceInstruction::TraceInstruction |
| ::ExecutionResult::sequentialData |     (C++                          |
|     (C++                          |                                   |
|     member)](api/langu            |   function)](api/languages/cpp_ap |
| ages/cpp_api.html#_CPPv4N5cudaq15 | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| ExecutionResult14sequentialDataE) | Instruction16TraceInstructionE20T |
| -   [                             | raceInstructionTypeNSt6stringENSt |
| cudaq::ExecutionResult::serialize | 6vectorINSt6size_tEEENSt6vectorIN |
|     (C++                          | St6size_tEEENSt6vectorIdEENSt8opt |
|     function)](api/l              | ionalIN5cudaq13kraus_channelEEE), |
| anguages/cpp_api.html#_CPPv4NK5cu |     [\[1\]](api/languages/cpp_a   |
| daq15ExecutionResult9serializeEv) | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| -   [cudaq::fermion_handler (C++  | eInstruction16TraceInstructionEv) |
|     c                             | -   [cud                          |
| lass)](api/languages/cpp_api.html | aq::ptsbe::TraceInstruction::type |
| #_CPPv4N5cudaq15fermion_handlerE) |     (C++                          |
| -   [cudaq::fermion_op (C++       |     member)](api/l                |
|     type)](api/languages/cpp_api  | anguages/cpp_api.html#_CPPv4N5cud |
| .html#_CPPv4N5cudaq10fermion_opE) | aq5ptsbe16TraceInstruction4typeE) |
| -   [cudaq::fermion_op_term (C++  | -   [c                            |
|                                   | udaq::ptsbe::TraceInstructionType |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15fermion_op_termE) |     enum)](api/                   |
| -   [cudaq::FermioniqBaseQPU (C++ | languages/cpp_api.html#_CPPv4N5cu |
|     cl                            | daq5ptsbe20TraceInstructionTypeE) |
| ass)](api/languages/cpp_api.html# | -   [cudaq::                      |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | ptsbe::TraceInstructionType::Gate |
| -   [cudaq::get_state (C++        |     (C++                          |
|                                   |     enumerator)](api/langu        |
|    function)](api/languages/cpp_a | ages/cpp_api.html#_CPPv4N5cudaq5p |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | tsbe20TraceInstructionType4GateE) |
| ateEDaRR13QuantumKernelDpRR4Args) | -   [cudaq::ptsbe::               |
| -   [cudaq::gradient (C++         | TraceInstructionType::Measurement |
|     class)](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4N5cudaq8gradientE) |                                   |
| -   [cudaq::gradient::clone (C++  |    enumerator)](api/languages/cpp |
|     fun                           | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| ction)](api/languages/cpp_api.htm | aceInstructionType11MeasurementE) |
| l#_CPPv4N5cudaq8gradient5cloneEv) | -   [cudaq::p                     |
| -   [cudaq::gradient::compute     | tsbe::TraceInstructionType::Noise |
|     (C++                          |     (C++                          |
|     function)](api/language       |     enumerator)](api/langua       |
| s/cpp_api.html#_CPPv4N5cudaq8grad | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| ient7computeERKNSt6vectorIdEERKNS | sbe20TraceInstructionType5NoiseE) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [                             |
|     [\[1\]](ap                    | cudaq::ptsbe::TrajectoryPredicate |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq8gradient7computeERKNSt6vect |     type)](api                    |
| orIdEERNSt6vectorIdEERK7spin_opd) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gradient::gradient    | udaq5ptsbe19TrajectoryPredicateE) |
|     (C++                          | -   [cudaq::QPU (C++              |
|     function)](api/lang           |     class)](api/languages         |
| uages/cpp_api.html#_CPPv4I00EN5cu | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| daq8gradient8gradientER7KernelT), | -   [cudaq::QPU::beginExecution   |
|                                   |     (C++                          |
|    [\[1\]](api/languages/cpp_api. |     function                      |
| html#_CPPv4I00EN5cudaq8gradient8g | )](api/languages/cpp_api.html#_CP |
| radientER7KernelTRR10ArgsMapper), | Pv4N5cudaq3QPU14beginExecutionEv) |
|     [\[2\                         | -   [cuda                         |
| ]](api/languages/cpp_api.html#_CP | q::QPU::configureExecutionContext |
| Pv4I00EN5cudaq8gradient8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     funct                         |
|     [\[3                          | ion)](api/languages/cpp_api.html# |
| \]](api/languages/cpp_api.html#_C | _CPPv4NK5cudaq3QPU25configureExec |
| PPv4N5cudaq8gradient8gradientERRN | utionContextER16ExecutionContext) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QPU::endExecution     |
|     [\[                           |     (C++                          |
| 4\]](api/languages/cpp_api.html#_ |     functi                        |
| CPPv4N5cudaq8gradient8gradientEv) | on)](api/languages/cpp_api.html#_ |
| -   [cudaq::gradient::setArgs     | CPPv4N5cudaq3QPU12endExecutionEv) |
|     (C++                          | -   [cudaq::QPU::enqueue (C++     |
|     fu                            |     function)](ap                 |
| nction)](api/languages/cpp_api.ht | i/languages/cpp_api.html#_CPPv4N5 |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | cudaq3QPU7enqueueER11QuantumTask) |
| tArgsEvR13QuantumKernelDpRR4Args) | -   [cud                          |
| -   [cudaq::gradient::setKernel   | aq::QPU::finalizeExecutionContext |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     func                          |
| pp_api.html#_CPPv4I0EN5cudaq8grad | tion)](api/languages/cpp_api.html |
| ient9setKernelEvR13QuantumKernel) | #_CPPv4NK5cudaq3QPU24finalizeExec |
| -   [cud                          | utionContextER16ExecutionContext) |
| aq::gradients::central_difference | -   [cudaq::QPU::getConnectivity  |
|     (C++                          |     (C++                          |
|     class)](api/la                |     function)                     |
| nguages/cpp_api.html#_CPPv4N5cuda | ](api/languages/cpp_api.html#_CPP |
| q9gradients18central_differenceE) | v4N5cudaq3QPU15getConnectivityEv) |
| -   [cudaq::gra                   | -                                 |
| dients::central_difference::clone | [cudaq::QPU::getExecutionThreadId |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/               |
| /cpp_api.html#_CPPv4N5cudaq9gradi | languages/cpp_api.html#_CPPv4NK5c |
| ents18central_difference5cloneEv) | udaq3QPU20getExecutionThreadIdEv) |
| -   [cudaq::gradi                 | -   [cudaq::QPU::getNumQubits     |
| ents::central_difference::compute |     (C++                          |
|     (C++                          |     functi                        |
|     function)](                   | on)](api/languages/cpp_api.html#_ |
| api/languages/cpp_api.html#_CPPv4 | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| N5cudaq9gradients18central_differ | -   [                             |
| ence7computeERKNSt6vectorIdEERKNS | cudaq::QPU::getRemoteCapabilities |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|                                   |     function)](api/l              |
|   [\[1\]](api/languages/cpp_api.h | anguages/cpp_api.html#_CPPv4NK5cu |
| tml#_CPPv4N5cudaq9gradients18cent | daq3QPU21getRemoteCapabilitiesEv) |
| ral_difference7computeERKNSt6vect | -   [cudaq::QPU::isEmulated (C++  |
| orIdEERNSt6vectorIdEERK7spin_opd) |     func                          |
| -   [cudaq::gradie                | tion)](api/languages/cpp_api.html |
| nts::central_difference::gradient | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     (C++                          | -   [cudaq::QPU::isSimulator (C++ |
|     functio                       |     funct                         |
| n)](api/languages/cpp_api.html#_C | ion)](api/languages/cpp_api.html# |
| PPv4I00EN5cudaq9gradients18centra | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| l_difference8gradientER7KernelT), | -   [cudaq::QPU::launchKernel     |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     function)](api                |
| q9gradients18central_difference8g | /languages/cpp_api.html#_CPPv4N5c |
| radientER7KernelTRR10ArgsMapper), | udaq3QPU12launchKernelERKNSt6stri |
|     [\[2\]](api/languages/cpp_    | ngE15KernelThunkTypePvNSt8uint64_ |
| api.html#_CPPv4I00EN5cudaq9gradie | tENSt8uint64_tERKNSt6vectorIPvEE) |
| nts18central_difference8gradientE | -   [cudaq::QPU::onRandomSeedSet  |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3\]](api/languages/cpp     |     function)](api/lang           |
| _api.html#_CPPv4N5cudaq9gradients | uages/cpp_api.html#_CPPv4N5cudaq3 |
| 18central_difference8gradientERRN | QPU15onRandomSeedSetENSt6size_tE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QPU::QPU (C++         |
|     [\[4\]](api/languages/cp      |     functio                       |
| p_api.html#_CPPv4N5cudaq9gradient | n)](api/languages/cpp_api.html#_C |
| s18central_difference8gradientEv) | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| -   [cud                          |                                   |
| aq::gradients::forward_difference |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     class)](api/la                |     [\[2\]](api/languages/cpp_    |
| nguages/cpp_api.html#_CPPv4N5cuda | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| q9gradients18forward_differenceE) | -   [cudaq::QPU::setId (C++       |
| -   [cudaq::gra                   |     function                      |
| dients::forward_difference::clone | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     function)](api/languages      | -   [cudaq::QPU::setShots (C++    |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     f                             |
| ents18forward_difference5cloneEv) | unction)](api/languages/cpp_api.h |
| -   [cudaq::gradi                 | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| ents::forward_difference::compute | -   [cudaq::                      |
|     (C++                          | QPU::supportsExplicitMeasurements |
|     function)](                   |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languag        |
| N5cudaq9gradients18forward_differ | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| ence7computeERKNSt6vectorIdEERKNS | 28supportsExplicitMeasurementsEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::QPU::\~QPU (C++       |
|                                   |     function)](api/languages/cp   |
|   [\[1\]](api/languages/cpp_api.h | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| tml#_CPPv4N5cudaq9gradients18forw | -   [cudaq::QPUState (C++         |
| ard_difference7computeERKNSt6vect |     class)](api/languages/cpp_    |
| orIdEERNSt6vectorIdEERK7spin_opd) | api.html#_CPPv4N5cudaq8QPUStateE) |
| -   [cudaq::gradie                | -   [cudaq::qreg (C++             |
| nts::forward_difference::gradient |     class)](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4I_NSt6s |
|     functio                       | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::qreg::back (C++       |
| PPv4I00EN5cudaq9gradients18forwar |     function)                     |
| d_difference8gradientER7KernelT), | ](api/languages/cpp_api.html#_CPP |
|     [\[1\]](api/langua            | v4N5cudaq4qreg4backENSt6size_tE), |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     [\[1\]](api/languages/cpp_ap  |
| q9gradients18forward_difference8g | i.html#_CPPv4N5cudaq4qreg4backEv) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::qreg::begin (C++      |
|     [\[2\]](api/languages/cpp_    |                                   |
| api.html#_CPPv4I00EN5cudaq9gradie |  function)](api/languages/cpp_api |
| nts18forward_difference8gradientE | .html#_CPPv4N5cudaq4qreg5beginEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::qreg::clear (C++      |
|     [\[3\]](api/languages/cpp     |                                   |
| _api.html#_CPPv4N5cudaq9gradients |  function)](api/languages/cpp_api |
| 18forward_difference8gradientERRN | .html#_CPPv4N5cudaq4qreg5clearEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::qreg::front (C++      |
|     [\[4\]](api/languages/cp      |     function)]                    |
| p_api.html#_CPPv4N5cudaq9gradient | (api/languages/cpp_api.html#_CPPv |
| s18forward_difference8gradientEv) | 4N5cudaq4qreg5frontENSt6size_tE), |
| -   [                             |     [\[1\]](api/languages/cpp_api |
| cudaq::gradients::parameter_shift | .html#_CPPv4N5cudaq4qreg5frontEv) |
|     (C++                          | -   [cudaq::qreg::operator\[\]    |
|     class)](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     functi                        |
| udaq9gradients15parameter_shiftE) | on)](api/languages/cpp_api.html#_ |
| -   [cudaq::                      | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| gradients::parameter_shift::clone | -   [cudaq::qreg::qreg (C++       |
|     (C++                          |     function)                     |
|     function)](api/langua         | ](api/languages/cpp_api.html#_CPP |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | v4N5cudaq4qreg4qregENSt6size_tE), |
| adients15parameter_shift5cloneEv) |     [\[1\]](api/languages/cpp_ap  |
| -   [cudaq::gr                    | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| adients::parameter_shift::compute | -   [cudaq::qreg::size (C++       |
|     (C++                          |                                   |
|     function                      |  function)](api/languages/cpp_api |
| )](api/languages/cpp_api.html#_CP | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| Pv4N5cudaq9gradients15parameter_s | -   [cudaq::qreg::slice (C++      |
| hift7computeERKNSt6vectorIdEERKNS |     function)](api/langu          |
| t8functionIFdNSt6vectorIdEEEEEd), | ages/cpp_api.html#_CPPv4N5cudaq4q |
|     [\[1\]](api/languages/cpp_ap  | reg5sliceENSt6size_tENSt6size_tE) |
| i.html#_CPPv4N5cudaq9gradients15p | -   [cudaq::qreg::value_type (C++ |
| arameter_shift7computeERKNSt6vect |                                   |
| orIdEERNSt6vectorIdEERK7spin_opd) | type)](api/languages/cpp_api.html |
| -   [cudaq::gra                   | #_CPPv4N5cudaq4qreg10value_typeE) |
| dients::parameter_shift::gradient | -   [cudaq::qspan (C++            |
|     (C++                          |     class)](api/lang              |
|     func                          | uages/cpp_api.html#_CPPv4I_NSt6si |
| tion)](api/languages/cpp_api.html | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| #_CPPv4I00EN5cudaq9gradients15par | -   [cudaq::QuakeValue (C++       |
| ameter_shift8gradientER7KernelT), |     class)](api/languages/cpp_api |
|     [\[1\]](api/lan               | .html#_CPPv4N5cudaq10QuakeValueE) |
| guages/cpp_api.html#_CPPv4I00EN5c | -   [cudaq::Q                     |
| udaq9gradients15parameter_shift8g | uakeValue::canValidateNumElements |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\]](api/languages/c       |     function)](api/languages      |
| pp_api.html#_CPPv4I00EN5cudaq9gra | /cpp_api.html#_CPPv4N5cudaq10Quak |
| dients15parameter_shift8gradientE | eValue22canValidateNumElementsEv) |
| RR13QuantumKernelRR10ArgsMapper), | -                                 |
|     [\[3\]](api/languages/        |  [cudaq::QuakeValue::constantSize |
| cpp_api.html#_CPPv4N5cudaq9gradie |     (C++                          |
| nts15parameter_shift8gradientERRN |     function)](api                |
| St8functionIFvNSt6vectorIdEEEEE), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[4\]](api/languages         | udaq10QuakeValue12constantSizeEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QuakeValue::dump (C++ |
| ents15parameter_shift8gradientEv) |     function)](api/lan            |
| -   [cudaq::kernel_builder (C++   | guages/cpp_api.html#_CPPv4N5cudaq |
|     clas                          | 10QuakeValue4dumpERNSt7ostreamE), |
| s)](api/languages/cpp_api.html#_C |     [\                            |
| PPv4IDpEN5cudaq14kernel_builderE) | [1\]](api/languages/cpp_api.html# |
| -   [c                            | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| udaq::kernel_builder::constantVal | -   [cudaq                        |
|     (C++                          | ::QuakeValue::getRequiredElements |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/langua         |
| q14kernel_builder11constantValEd) | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| -   [cu                           | uakeValue19getRequiredElementsEv) |
| daq::kernel_builder::getArguments | -   [cudaq::QuakeValue::getValue  |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)]                    |
| guages/cpp_api.html#_CPPv4N5cudaq | (api/languages/cpp_api.html#_CPPv |
| 14kernel_builder12getArgumentsEv) | 4NK5cudaq10QuakeValue8getValueEv) |
| -   [cu                           | -   [cudaq::QuakeValue::inverse   |
| daq::kernel_builder::getNumParams |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/lan            | ](api/languages/cpp_api.html#_CPP |
| guages/cpp_api.html#_CPPv4N5cudaq | v4NK5cudaq10QuakeValue7inverseEv) |
| 14kernel_builder12getNumParamsEv) | -   [cudaq::QuakeValue::isStdVec  |
| -   [c                            |     (C++                          |
| udaq::kernel_builder::isArgStdVec |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/languages/cp   | v4N5cudaq10QuakeValue8isStdVecEv) |
| p_api.html#_CPPv4N5cudaq14kernel_ | -                                 |
| builder11isArgStdVecENSt6size_tE) |    [cudaq::QuakeValue::operator\* |
| -   [cuda                         |     (C++                          |
| q::kernel_builder::kernel_builder |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/languages/cpp_ | udaq10QuakeValuemlE10QuakeValue), |
| api.html#_CPPv4N5cudaq14kernel_bu |                                   |
| ilder14kernel_builderERNSt6vector | [\[1\]](api/languages/cpp_api.htm |
| IN7details17KernelBuilderTypeEEE) | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| -   [cudaq::kernel_builder::name  | -   [cudaq::QuakeValue::operator+ |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api                |
| ](api/languages/cpp_api.html#_CPP | /languages/cpp_api.html#_CPPv4N5c |
| v4N5cudaq14kernel_builder4nameEv) | udaq10QuakeValueplE10QuakeValue), |
| -                                 |     [                             |
|    [cudaq::kernel_builder::qalloc | \[1\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     function)](api/language       |                                   |
| s/cpp_api.html#_CPPv4N5cudaq14ker | [\[2\]](api/languages/cpp_api.htm |
| nel_builder6qallocE10QuakeValue), | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|     [\[1\]](api/language          | -   [cudaq::QuakeValue::operator- |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     (C++                          |
| nel_builder6qallocEKNSt6size_tE), |     function)](api                |
|     [\[2                          | /languages/cpp_api.html#_CPPv4N5c |
| \]](api/languages/cpp_api.html#_C | udaq10QuakeValuemiE10QuakeValue), |
| PPv4N5cudaq14kernel_builder6qallo |     [                             |
| cERNSt6vectorINSt7complexIdEEEE), | \[1\]](api/languages/cpp_api.html |
|     [\[3\]](                      | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| api/languages/cpp_api.html#_CPPv4 |     [                             |
| N5cudaq14kernel_builder6qallocEv) | \[2\]](api/languages/cpp_api.html |
| -   [cudaq::kernel_builder::swap  | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     (C++                          |                                   |
|     function)](api/language       | [\[3\]](api/languages/cpp_api.htm |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| 4kernel_builder4swapEvRK10QuakeVa | -   [cudaq::QuakeValue::operator/ |
| lueRK10QuakeValueRK10QuakeValue), |     (C++                          |
|                                   |     function)](api                |
| [\[1\]](api/languages/cpp_api.htm | /languages/cpp_api.html#_CPPv4N5c |
| l#_CPPv4I00EN5cudaq14kernel_build | udaq10QuakeValuedvE10QuakeValue), |
| er4swapEvRKNSt6vectorI10QuakeValu |                                   |
| eEERK10QuakeValueRK10QuakeValue), | [\[1\]](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| [\[2\]](api/languages/cpp_api.htm | -                                 |
| l#_CPPv4N5cudaq14kernel_builder4s |  [cudaq::QuakeValue::operator\[\] |
| wapERK10QuakeValueRK10QuakeValue) |     (C++                          |
| -   [cudaq::KernelExecutionTask   |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     type                          | udaq10QuakeValueixEKNSt6size_tE), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/                  |
| Pv4N5cudaq19KernelExecutionTaskE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::KernelThunkResultType | daq10QuakeValueixERK10QuakeValue) |
|     (C++                          | -                                 |
|     struct)]                      |    [cudaq::QuakeValue::QuakeValue |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq21KernelThunkResultTypeE) |     function)](api/languag        |
| -   [cudaq::KernelThunkType (C++  | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|                                   | akeValue10QuakeValueERN4mlir20Imp |
| type)](api/languages/cpp_api.html | licitLocOpBuilderEN4mlir5ValueE), |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     [\[1\]                        |
| -   [cudaq::kraus_channel (C++    | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq10QuakeValue10QuakeValue |
|  class)](api/languages/cpp_api.ht | ERN4mlir20ImplicitLocOpBuilderEd) |
| ml#_CPPv4N5cudaq13kraus_channelE) | -   [cudaq::QuakeValue::size (C++ |
| -   [cudaq::kraus_channel::empty  |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)]                    | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::QuakeValue::slice     |
| 4NK5cudaq13kraus_channel5emptyEv) |     (C++                          |
| -   [cudaq::kraus_c               |     function)](api/languages/cpp_ |
| hannel::generateUnitaryParameters | api.html#_CPPv4N5cudaq10QuakeValu |
|     (C++                          | e5sliceEKNSt6size_tEKNSt6size_tE) |
|                                   | -   [cudaq::quantum_platform (C++ |
|    function)](api/languages/cpp_a |     cl                            |
| pi.html#_CPPv4N5cudaq13kraus_chan | ass)](api/languages/cpp_api.html# |
| nel25generateUnitaryParametersEv) | _CPPv4N5cudaq16quantum_platformE) |
| -                                 | -   [cudaq:                       |
|    [cudaq::kraus_channel::get_ops | :quantum_platform::beginExecution |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](api/languag        |
| pi/languages/cpp_api.html#_CPPv4N | es/cpp_api.html#_CPPv4N5cudaq16qu |
| K5cudaq13kraus_channel7get_opsEv) | antum_platform14beginExecutionEv) |
| -   [cud                          | -   [cudaq::quantum_pl            |
| aq::kraus_channel::identity_flags | atform::configureExecutionContext |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/lang           |
| guages/cpp_api.html#_CPPv4N5cudaq | uages/cpp_api.html#_CPPv4NK5cudaq |
| 13kraus_channel14identity_flagsE) | 16quantum_platform25configureExec |
| -   [cud                          | utionContextER16ExecutionContext) |
| aq::kraus_channel::is_identity_op | -   [cuda                         |
|     (C++                          | q::quantum_platform::connectivity |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](api/langu          |
| pi.html#_CPPv4NK5cudaq13kraus_cha | ages/cpp_api.html#_CPPv4N5cudaq16 |
| nnel14is_identity_opENSt6size_tE) | quantum_platform12connectivityEv) |
| -   [cudaq::                      | -   [cuda                         |
| kraus_channel::is_unitary_mixture | q::quantum_platform::endExecution |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/langu          |
| /cpp_api.html#_CPPv4NK5cudaq13kra | ages/cpp_api.html#_CPPv4N5cudaq16 |
| us_channel18is_unitary_mixtureEv) | quantum_platform12endExecutionEv) |
| -   [cu                           | -   [cudaq::q                     |
| daq::kraus_channel::kraus_channel | uantum_platform::enqueueAsyncTask |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](api/languages/     |
| uages/cpp_api.html#_CPPv4IDpEN5cu | cpp_api.html#_CPPv4N5cudaq16quant |
| daq13kraus_channel13kraus_channel | um_platform16enqueueAsyncTaskEKNS |
| EDpRRNSt16initializer_listI1TEE), | t6size_tER19KernelExecutionTask), |
|                                   |     [\[1\]](api/languag           |
|  [\[1\]](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ml#_CPPv4N5cudaq13kraus_channel13 | antum_platform16enqueueAsyncTaskE |
| kraus_channelERK13kraus_channel), | KNSt6size_tERNSt8functionIFvvEEE) |
|     [\[2\]                        | -   [cudaq::quantum_p             |
| ](api/languages/cpp_api.html#_CPP | latform::finalizeExecutionContext |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERKNSt6vectorI8kraus_opEE), |     function)](api/languages/c    |
|     [\[3\]                        | pp_api.html#_CPPv4NK5cudaq16quant |
| ](api/languages/cpp_api.html#_CPP | um_platform24finalizeExecutionCon |
| v4N5cudaq13kraus_channel13kraus_c | textERN5cudaq16ExecutionContextE) |
| hannelERRNSt6vectorI8kraus_opEE), | -   [cudaq::qua                   |
|     [\[4\]](api/lan               | ntum_platform::get_codegen_config |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 13kraus_channel13kraus_channelEv) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4N5cudaq16quantu |
| [cudaq::kraus_channel::noise_type | m_platform18get_codegen_configEv) |
|     (C++                          | -   [cuda                         |
|     member)](api                  | q::quantum_platform::get_exec_ctx |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel10noise_typeE) |     function)](api/langua         |
| -                                 | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|   [cudaq::kraus_channel::op_names | quantum_platform12get_exec_ctxEv) |
|     (C++                          | -   [c                            |
|     member)](                     | udaq::quantum_platform::get_noise |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq13kraus_channel8op_namesE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4N5cudaq16quantu |
|  [cudaq::kraus_channel::operator= | m_platform9get_noiseENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/langua         | :quantum_platform::get_num_qubits |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     (C++                          |
| raus_channelaSERK13kraus_channel) |                                   |
| -   [c                            | function)](api/languages/cpp_api. |
| udaq::kraus_channel::operator\[\] | html#_CPPv4NK5cudaq16quantum_plat |
|     (C++                          | form14get_num_qubitsENSt6size_tE) |
|     function)](api/l              | -   [cudaq::quantum_              |
| anguages/cpp_api.html#_CPPv4N5cud | platform::get_remote_capabilities |
| aq13kraus_channelixEKNSt6size_tE) |     (C++                          |
| -                                 |     function)                     |
| [cudaq::kraus_channel::parameters | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq16quantum_platform23get |
|     member)](api                  | _remote_capabilitiesENSt6size_tE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::qua                   |
| udaq13kraus_channel10parametersE) | ntum_platform::get_runtime_target |
| -   [cudaq::krau                  |     (C++                          |
| s_channel::populateDefaultOpNames |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4NK5cudaq16quantu |
|     function)](api/languages/cp   | m_platform18get_runtime_targetEv) |
| p_api.html#_CPPv4N5cudaq13kraus_c | -   [cuda                         |
| hannel22populateDefaultOpNamesEv) | q::quantum_platform::getLogStream |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::probabilities |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     member)](api/la               | quantum_platform12getLogStreamEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cud                          |
| q13kraus_channel13probabilitiesE) | aq::quantum_platform::is_emulated |
| -                                 |     (C++                          |
|  [cudaq::kraus_channel::push_back |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     function)](api                | pi.html#_CPPv4NK5cudaq16quantum_p |
| /languages/cpp_api.html#_CPPv4N5c | latform11is_emulatedENSt6size_tE) |
| udaq13kraus_channel9push_backE8kr | -   [c                            |
| aus_opNSt8optionalINSt6stringEEE) | udaq::quantum_platform::is_remote |
| -   [cudaq::kraus_channel::size   |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     function)                     | p_api.html#_CPPv4NK5cudaq16quantu |
| ](api/languages/cpp_api.html#_CPP | m_platform9is_remoteENSt6size_tE) |
| v4NK5cudaq13kraus_channel4sizeEv) | -   [cuda                         |
| -   [                             | q::quantum_platform::is_simulator |
| cudaq::kraus_channel::unitary_ops |     (C++                          |
|     (C++                          |                                   |
|     member)](api/                 |   function)](api/languages/cpp_ap |
| languages/cpp_api.html#_CPPv4N5cu | i.html#_CPPv4NK5cudaq16quantum_pl |
| daq13kraus_channel11unitary_opsE) | atform12is_simulatorENSt6size_tE) |
| -   [cudaq::kraus_op (C++         | -   [c                            |
|     struct)](api/languages/cpp_   | udaq::quantum_platform::launchVQE |
| api.html#_CPPv4N5cudaq8kraus_opE) |     (C++                          |
| -   [cudaq::kraus_op::adjoint     |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     functi                        | N5cudaq16quantum_platform9launchV |
| on)](api/languages/cpp_api.html#_ | QEEKNSt6stringEPKvPN5cudaq8gradie |
| CPPv4NK5cudaq8kraus_op7adjointEv) | ntERKN5cudaq7spin_opERN5cudaq9opt |
| -   [cudaq::kraus_op::data (C++   | imizerEKiKNSt6size_tENSt6size_tE) |
|                                   | -   [cudaq:                       |
|  member)](api/languages/cpp_api.h | :quantum_platform::list_platforms |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     (C++                          |
| -   [cudaq::kraus_op::kraus_op    |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     func                          | antum_platform14list_platformsEv) |
| tion)](api/languages/cpp_api.html | -                                 |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |    [cudaq::quantum_platform::name |
| opERRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)](a                  |
|  [\[1\]](api/languages/cpp_api.ht | pi/languages/cpp_api.html#_CPPv4N |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | K5cudaq16quantum_platform4nameEv) |
| pENSt6vectorIN5cudaq7complexEEE), | -   [                             |
|     [\[2\]](api/l                 | cudaq::quantum_platform::num_qpus |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq8kraus_op8kraus_opERK8kraus_op) |     function)](api/l              |
| -   [cudaq::kraus_op::nCols (C++  | anguages/cpp_api.html#_CPPv4NK5cu |
|                                   | daq16quantum_platform8num_qpusEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::                      |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | quantum_platform::onRandomSeedSet |
| -   [cudaq::kraus_op::nRows (C++  |     (C++                          |
|                                   |                                   |
| member)](api/languages/cpp_api.ht | function)](api/languages/cpp_api. |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | html#_CPPv4N5cudaq16quantum_platf |
| -   [cudaq::kraus_op::operator=   | orm15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     function)                     | :quantum_platform::reset_exec_ctx |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq8kraus_opaSERK8kraus_op) |     function)](api/languag        |
| -   [cudaq::kraus_op::precision   | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14reset_exec_ctxEv) |
|     memb                          | -   [cud                          |
| er)](api/languages/cpp_api.html#_ | aq::quantum_platform::reset_noise |
| CPPv4N5cudaq8kraus_op9precisionE) |     (C++                          |
| -   [cudaq::KrausSelection (C++   |     function)](api/languages/cpp_ |
|     s                             | api.html#_CPPv4N5cudaq16quantum_p |
| truct)](api/languages/cpp_api.htm | latform11reset_noiseENSt6size_tE) |
| l#_CPPv4N5cudaq14KrausSelectionE) | -   [cudaq:                       |
| -   [cudaq:                       | :quantum_platform::resetLogStream |
| :KrausSelection::circuit_location |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     member)](api/langua           | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ges/cpp_api.html#_CPPv4N5cudaq14K | antum_platform14resetLogStreamEv) |
| rausSelection16circuit_locationE) | -   [cuda                         |
| -                                 | q::quantum_platform::set_exec_ctx |
|  [cudaq::KrausSelection::is_error |     (C++                          |
|     (C++                          |     funct                         |
|     member)](a                    | ion)](api/languages/cpp_api.html# |
| pi/languages/cpp_api.html#_CPPv4N | _CPPv4N5cudaq16quantum_platform12 |
| 5cudaq14KrausSelection8is_errorE) | set_exec_ctxEP16ExecutionContext) |
| -   [cudaq::Kra                   | -   [c                            |
| usSelection::kraus_operator_index | udaq::quantum_platform::set_noise |
|     (C++                          |     (C++                          |
|     member)](api/languages/       |     function                      |
| cpp_api.html#_CPPv4N5cudaq14Kraus | )](api/languages/cpp_api.html#_CP |
| Selection20kraus_operator_indexE) | Pv4N5cudaq16quantum_platform9set_ |
| -   [cuda                         | noiseEPK11noise_modelNSt6size_tE) |
| q::KrausSelection::KrausSelection | -   [cuda                         |
|     (C++                          | q::quantum_platform::setLogStream |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |                                   |
| 5cudaq14KrausSelection14KrausSele |  function)](api/languages/cpp_api |
| ctionENSt6size_tENSt6vectorINSt6s | .html#_CPPv4N5cudaq16quantum_plat |
| ize_tEEENSt6stringENSt6size_tEb), | form12setLogStreamERNSt7ostreamE) |
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
| ndler23get_expected_dimensionsEv) | -   [cudaq::Remo                  |
| -   [cudaq::matrix_ha             | teCapabilities::isRemoteSimulator |
| ndler::get_parameter_descriptions |     (C++                          |
|     (C++                          |     member)](api/languages/c      |
|                                   | pp_api.html#_CPPv4N5cudaq18Remote |
| function)](api/languages/cpp_api. | Capabilities17isRemoteSimulatorE) |
| html#_CPPv4NK5cudaq14matrix_handl | -   [cudaq::Remot                 |
| er26get_parameter_descriptionsEv) | eCapabilities::RemoteCapabilities |
| -   [c                            |     (C++                          |
| udaq::matrix_handler::instantiate |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4N5cudaq18RemoteCa |
|     function)](a                  | pabilities18RemoteCapabilitiesEb) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq:                       |
| 5cudaq14matrix_handler11instantia | :RemoteCapabilities::stateOverlap |
| teENSt6stringERKNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     member)](api/langua           |
|     [\[1\]](                      | ges/cpp_api.html#_CPPv4N5cudaq18R |
| api/languages/cpp_api.html#_CPPv4 | emoteCapabilities12stateOverlapE) |
| N5cudaq14matrix_handler11instanti | -                                 |
| ateENSt6stringERRNSt6vectorINSt6s |   [cudaq::RemoteCapabilities::vqe |
| ize_tEEERK20commutation_behavior) |     (C++                          |
| -   [cuda                         |     member)](                     |
| q::matrix_handler::matrix_handler | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq18RemoteCapabilities3vqeE) |
|     function)](api/languag        | -   [cudaq::RemoteSimulationState |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     (C++                          |
| ble_if_tINSt12is_base_of_vI16oper |     class)]                       |
| ator_handler1TEEbEEEN5cudaq14matr | (api/languages/cpp_api.html#_CPPv |
| ix_handler14matrix_handlerERK1T), | 4N5cudaq21RemoteSimulationStateE) |
|     [\[1\]](ap                    | -   [cudaq::Resources (C++        |
| i/languages/cpp_api.html#_CPPv4I0 |     class)](api/languages/cpp_a   |
| _NSt11enable_if_tINSt12is_base_of | pi.html#_CPPv4N5cudaq9ResourcesE) |
| _vI16operator_handler1TEEbEEEN5cu | -   [cudaq::run (C++              |
| daq14matrix_handler14matrix_handl |     function)]                    |
| erERK1TRK20commutation_behavior), | (api/languages/cpp_api.html#_CPPv |
|     [\[2\]](api/languages/cpp_ap  | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| i.html#_CPPv4N5cudaq14matrix_hand | 5invoke_result_tINSt7decay_tI13Qu |
| ler14matrix_handlerENSt6size_tE), | antumKernelEEDpNSt7decay_tI4ARGSE |
|     [\[3\]](api/                  | EEEEENSt6size_tERN5cudaq11noise_m |
| languages/cpp_api.html#_CPPv4N5cu | odelERR13QuantumKernelDpRR4ARGS), |
| daq14matrix_handler14matrix_handl |     [\[1\]](api/langu             |
| erENSt6stringERKNSt6vectorINSt6si | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| ze_tEEERK20commutation_behavior), | daq3runENSt6vectorINSt15invoke_re |
|     [\[4\]](api/                  | sult_tINSt7decay_tI13QuantumKerne |
| languages/cpp_api.html#_CPPv4N5cu | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| daq14matrix_handler14matrix_handl | ize_tERR13QuantumKernelDpRR4ARGS) |
| erENSt6stringERRNSt6vectorINSt6si | -   [cudaq::run_async (C++        |
| ze_tEEERK20commutation_behavior), |     functio                       |
|     [\                            | n)](api/languages/cpp_api.html#_C |
| [5\]](api/languages/cpp_api.html# | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| _CPPv4N5cudaq14matrix_handler14ma | tureINSt6vectorINSt15invoke_resul |
| trix_handlerERK14matrix_handler), | t_tINSt7decay_tI13QuantumKernelEE |
|     [                             | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| \[6\]](api/languages/cpp_api.html | ze_tENSt6size_tERN5cudaq11noise_m |
| #_CPPv4N5cudaq14matrix_handler14m | odelERR13QuantumKernelDpRR4ARGS), |
| atrix_handlerERR14matrix_handler) |     [\[1\]](api/la                |
| -                                 | nguages/cpp_api.html#_CPPv4I0DpEN |
|  [cudaq::matrix_handler::momentum | 5cudaq9run_asyncENSt6futureINSt6v |
|     (C++                          | ectorINSt15invoke_result_tINSt7de |
|     function)](api/language       | cay_tI13QuantumKernelEEDpNSt7deca |
| s/cpp_api.html#_CPPv4N5cudaq14mat | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| rix_handler8momentumENSt6size_tE) | ize_tERR13QuantumKernelDpRR4ARGS) |
| -                                 | -   [cudaq::RuntimeTarget (C++    |
|    [cudaq::matrix_handler::number |                                   |
|     (C++                          | struct)](api/languages/cpp_api.ht |
|     function)](api/langua         | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -   [cudaq::sample (C++           |
| atrix_handler6numberENSt6size_tE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| [cudaq::matrix_handler::operator= | mpleE13sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     fun                           |     [\[1\                         |
| ction)](api/languages/cpp_api.htm | ]](api/languages/cpp_api.html#_CP |
| l#_CPPv4I0_NSt11enable_if_tIXaant | Pv4I0DpEN5cudaq6sampleE13sample_r |
| NSt7is_sameI1T14matrix_handlerE5v | esultRR13QuantumKernelDpRR4Args), |
| alueENSt12is_base_of_vI16operator |     [\                            |
| _handler1TEEEbEEEN5cudaq14matrix_ | [2\]](api/languages/cpp_api.html# |
| handleraSER14matrix_handlerRK1T), | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|     [\[1\]](api/languages         | ize_tERR13QuantumKernelDpRR4Args) |
| /cpp_api.html#_CPPv4N5cudaq14matr | -   [cudaq::sample_options (C++   |
| ix_handleraSERK14matrix_handler), |     s                             |
|     [\[2\]](api/language          | truct)](api/languages/cpp_api.htm |
| s/cpp_api.html#_CPPv4N5cudaq14mat | l#_CPPv4N5cudaq14sample_optionsE) |
| rix_handleraSERR14matrix_handler) | -   [cudaq::samp                  |
| -   [                             | le_options::explicit_measurements |
| cudaq::matrix_handler::operator== |     (C++                          |
|     (C++                          |     member)](api/languages/c      |
|     function)](api/languages      | pp_api.html#_CPPv4N5cudaq14sample |
| /cpp_api.html#_CPPv4NK5cudaq14mat | _options21explicit_measurementsE) |
| rix_handlereqERK14matrix_handler) | -   [cudaq::sample_result (C++    |
| -                                 |                                   |
|    [cudaq::matrix_handler::parity |  class)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq13sample_resultE) |
|     function)](api/langua         | -   [cudaq::sample_result::append |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6parityENSt6size_tE) |     function)](api/languages/cpp_ |
| -                                 | api.html#_CPPv4N5cudaq13sample_re |
|  [cudaq::matrix_handler::position | sult6appendERK15ExecutionResultb) |
|     (C++                          | -   [cudaq::sample_result::begin  |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     function)]                    |
| rix_handler8positionENSt6size_tE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::                      | 4N5cudaq13sample_result5beginEv), |
| matrix_handler::remove_definition |     [\[1\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     fu                            | 4NK5cudaq13sample_result5beginEv) |
| nction)](api/languages/cpp_api.ht | -   [cudaq::sample_result::cbegin |
| ml#_CPPv4N5cudaq14matrix_handler1 |     (C++                          |
| 7remove_definitionERKNSt6stringE) |     function)](                   |
| -                                 | api/languages/cpp_api.html#_CPPv4 |
|   [cudaq::matrix_handler::squeeze | NK5cudaq13sample_result6cbeginEv) |
|     (C++                          | -   [cudaq::sample_result::cend   |
|     function)](api/languag        |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     function)                     |
| trix_handler7squeezeENSt6size_tE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::m                     | v4NK5cudaq13sample_result4cendEv) |
| atrix_handler::to_diagonal_matrix | -   [cudaq::sample_result::clear  |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)                     |
| uages/cpp_api.html#_CPPv4NK5cudaq | ](api/languages/cpp_api.html#_CPP |
| 14matrix_handler18to_diagonal_mat | v4N5cudaq13sample_result5clearEv) |
| rixERNSt13unordered_mapINSt6size_ | -   [cudaq::sample_result::count  |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     function)](                   |
| -                                 | api/languages/cpp_api.html#_CPPv4 |
| [cudaq::matrix_handler::to_matrix | NK5cudaq13sample_result5countENSt |
|     (C++                          | 11string_viewEKNSt11string_viewE) |
|     function)                     | -   [                             |
| ](api/languages/cpp_api.html#_CPP | cudaq::sample_result::deserialize |
| v4NK5cudaq14matrix_handler9to_mat |     (C++                          |
| rixERNSt13unordered_mapINSt6size_ |     functio                       |
| tENSt7int64_tEEERKNSt13unordered_ | n)](api/languages/cpp_api.html#_C |
| mapINSt6stringENSt7complexIdEEEE) | PPv4N5cudaq13sample_result11deser |
| -                                 | ializeERNSt6vectorINSt6size_tEEE) |
| [cudaq::matrix_handler::to_string | -   [cudaq::sample_result::dump   |
|     (C++                          |     (C++                          |
|     function)](api/               |     function)](api/languag        |
| languages/cpp_api.html#_CPPv4NK5c | es/cpp_api.html#_CPPv4NK5cudaq13s |
| udaq14matrix_handler9to_stringEb) | ample_result4dumpERNSt7ostreamE), |
| -                                 |     [\[1\]                        |
| [cudaq::matrix_handler::unique_id | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq13sample_result4dumpEv) |
|     function)](api/               | -   [cudaq::sample_result::end    |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14matrix_handler9unique_idEv) |     function                      |
| -   [cudaq:                       | )](api/languages/cpp_api.html#_CP |
| :matrix_handler::\~matrix_handler | Pv4N5cudaq13sample_result3endEv), |
|     (C++                          |     [\[1\                         |
|     functi                        | ]](api/languages/cpp_api.html#_CP |
| on)](api/languages/cpp_api.html#_ | Pv4NK5cudaq13sample_result3endEv) |
| CPPv4N5cudaq14matrix_handlerD0Ev) | -   [                             |
| -   [cudaq::matrix_op (C++        | cudaq::sample_result::expectation |
|     type)](api/languages/cpp_a    |     (C++                          |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     f                             |
| -   [cudaq::matrix_op_term (C++   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4NK5cudaq13sample_result |
|  type)](api/languages/cpp_api.htm | 11expectationEKNSt11string_viewE) |
| l#_CPPv4N5cudaq14matrix_op_termE) | -   [c                            |
| -                                 | udaq::sample_result::get_marginal |
|    [cudaq::mdiag_operator_handler |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     class)](                      | api.html#_CPPv4NK5cudaq13sample_r |
| api/languages/cpp_api.html#_CPPv4 | esult12get_marginalERKNSt6vectorI |
| N5cudaq22mdiag_operator_handlerE) | NSt6size_tEEEKNSt11string_viewE), |
| -   [cudaq::mpi (C++              |     [\[1\]](api/languages/cpp_    |
|     type)](api/languages          | api.html#_CPPv4NK5cudaq13sample_r |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | esult12get_marginalERRKNSt6vector |
| -   [cudaq::mpi::all_gather (C++  | INSt6size_tEEEKNSt11string_viewE) |
|     fu                            | -   [cuda                         |
| nction)](api/languages/cpp_api.ht | q::sample_result::get_total_shots |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     (C++                          |
| RNSt6vectorIdEERKNSt6vectorIdEE), |     function)](api/langua         |
|                                   | ges/cpp_api.html#_CPPv4NK5cudaq13 |
|   [\[1\]](api/languages/cpp_api.h | sample_result15get_total_shotsEv) |
| tml#_CPPv4N5cudaq3mpi10all_gather | -   [cuda                         |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | q::sample_result::has_even_parity |
| -   [cudaq::mpi::all_reduce (C++  |     (C++                          |
|                                   |     fun                           |
|  function)](api/languages/cpp_api | ction)](api/languages/cpp_api.htm |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | l#_CPPv4N5cudaq13sample_result15h |
| reduceE1TRK1TRK14BinaryFunction), | as_even_parityENSt11string_viewE) |
|     [\[1\]](api/langu             | -   [cuda                         |
| ages/cpp_api.html#_CPPv4I00EN5cud | q::sample_result::has_expectation |
| aq3mpi10all_reduceE1TRK1TRK4Func) |     (C++                          |
| -   [cudaq::mpi::broadcast (C++   |     funct                         |
|     function)](api/               | ion)](api/languages/cpp_api.html# |
| languages/cpp_api.html#_CPPv4N5cu | _CPPv4NK5cudaq13sample_result15ha |
| daq3mpi9broadcastERNSt6stringEi), | s_expectationEKNSt11string_viewE) |
|     [\[1\]](api/la                | -   [cu                           |
| nguages/cpp_api.html#_CPPv4N5cuda | daq::sample_result::most_probable |
| q3mpi9broadcastERNSt6vectorIdEEi) |     (C++                          |
| -   [cudaq::mpi::finalize (C++    |     fun                           |
|     f                             | ction)](api/languages/cpp_api.htm |
| unction)](api/languages/cpp_api.h | l#_CPPv4NK5cudaq13sample_result13 |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | most_probableEKNSt11string_viewE) |
| -   [cudaq::mpi::initialize (C++  | -                                 |
|     function                      | [cudaq::sample_result::operator+= |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq3mpi10initializeEiPPc), |     function)](api/langua         |
|     [                             | ges/cpp_api.html#_CPPv4N5cudaq13s |
| \[1\]](api/languages/cpp_api.html | ample_resultpLERK13sample_result) |
| #_CPPv4N5cudaq3mpi10initializeEv) | -                                 |
| -   [cudaq::mpi::is_initialized   |  [cudaq::sample_result::operator= |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/langua         |
| )](api/languages/cpp_api.html#_CP | ges/cpp_api.html#_CPPv4N5cudaq13s |
| Pv4N5cudaq3mpi14is_initializedEv) | ample_resultaSERR13sample_result) |
| -   [cudaq::mpi::num_ranks (C++   | -                                 |
|     fu                            | [cudaq::sample_result::operator== |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     function)](api/languag        |
| -   [cudaq::mpi::rank (C++        | es/cpp_api.html#_CPPv4NK5cudaq13s |
|                                   | ample_resulteqERK13sample_result) |
|    function)](api/languages/cpp_a | -   [                             |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | cudaq::sample_result::probability |
| -   [cudaq::noise_model (C++      |     (C++                          |
|                                   |     function)](api/lan            |
|    class)](api/languages/cpp_api. | guages/cpp_api.html#_CPPv4NK5cuda |
| html#_CPPv4N5cudaq11noise_modelE) | q13sample_result11probabilityENSt |
| -   [cudaq::n                     | 11string_viewEKNSt11string_viewE) |
| oise_model::add_all_qubit_channel | -   [cud                          |
|     (C++                          | aq::sample_result::register_names |
|     function)](api                |     (C++                          |
| /languages/cpp_api.html#_CPPv4IDp |     function)](api/langu          |
| EN5cudaq11noise_model21add_all_qu | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| bit_channelEvRK13kraus_channeli), | 3sample_result14register_namesEv) |
|     [\[1\]](api/langua            | -                                 |
| ges/cpp_api.html#_CPPv4N5cudaq11n |    [cudaq::sample_result::reorder |
| oise_model21add_all_qubit_channel |     (C++                          |
| ERKNSt6stringERK13kraus_channeli) |     function)](api/langua         |
| -                                 | ges/cpp_api.html#_CPPv4N5cudaq13s |
|  [cudaq::noise_model::add_channel | ample_result7reorderERKNSt6vector |
|     (C++                          | INSt6size_tEEEKNSt11string_viewE) |
|     funct                         | -   [cu                           |
| ion)](api/languages/cpp_api.html# | daq::sample_result::sample_result |
| _CPPv4IDpEN5cudaq11noise_model11a |     (C++                          |
| dd_channelEvRK15PredicateFuncTy), |     func                          |
|     [\[1\]](api/languages/cpp_    | tion)](api/languages/cpp_api.html |
| api.html#_CPPv4IDpEN5cudaq11noise | #_CPPv4N5cudaq13sample_result13sa |
| _model11add_channelEvRKNSt6vector | mple_resultERK15ExecutionResult), |
| INSt6size_tEEERK13kraus_channel), |     [\[1\]](api/la                |
|     [\[2\]](ap                    | nguages/cpp_api.html#_CPPv4N5cuda |
| i/languages/cpp_api.html#_CPPv4N5 | q13sample_result13sample_resultER |
| cudaq11noise_model11add_channelER | KNSt6vectorI15ExecutionResultEE), |
| KNSt6stringERK15PredicateFuncTy), |                                   |
|                                   |  [\[2\]](api/languages/cpp_api.ht |
| [\[3\]](api/languages/cpp_api.htm | ml#_CPPv4N5cudaq13sample_result13 |
| l#_CPPv4N5cudaq11noise_model11add | sample_resultERR13sample_result), |
| _channelERKNSt6stringERKNSt6vecto |     [                             |
| rINSt6size_tEEERK13kraus_channel) | \[3\]](api/languages/cpp_api.html |
| -   [cudaq::noise_model::empty    | #_CPPv4N5cudaq13sample_result13sa |
|     (C++                          | mple_resultERR15ExecutionResult), |
|     function                      |     [\[4\]](api/lan               |
| )](api/languages/cpp_api.html#_CP | guages/cpp_api.html#_CPPv4N5cudaq |
| Pv4NK5cudaq11noise_model5emptyEv) | 13sample_result13sample_resultEdR |
| -                                 | KNSt6vectorI15ExecutionResultEE), |
| [cudaq::noise_model::get_channels |     [\[5\]](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     function)](api/l              | 13sample_result13sample_resultEv) |
| anguages/cpp_api.html#_CPPv4I0ENK | -                                 |
| 5cudaq11noise_model12get_channels |  [cudaq::sample_result::serialize |
| ENSt6vectorI13kraus_channelEERKNS |     (C++                          |
| t6vectorINSt6size_tEEERKNSt6vecto |     function)](api                |
| rINSt6size_tEEERKNSt6vectorIdEE), | /languages/cpp_api.html#_CPPv4NK5 |
|     [\[1\]](api/languages/cpp_a   | cudaq13sample_result9serializeEv) |
| pi.html#_CPPv4NK5cudaq11noise_mod | -   [cudaq::sample_result::size   |
| el12get_channelsERKNSt6stringERKN |     (C++                          |
| St6vectorINSt6size_tEEERKNSt6vect |     function)](api/languages/c    |
| orINSt6size_tEEERKNSt6vectorIdEE) | pp_api.html#_CPPv4NK5cudaq13sampl |
| -                                 | e_result4sizeEKNSt11string_viewE) |
|  [cudaq::noise_model::noise_model | -   [cudaq::sample_result::to_map |
|     (C++                          |     (C++                          |
|     function)](api                |     function)](api/languages/cpp  |
| /languages/cpp_api.html#_CPPv4N5c | _api.html#_CPPv4NK5cudaq13sample_ |
| udaq11noise_model11noise_modelEv) | result6to_mapEKNSt11string_viewE) |
| -   [cu                           | -   [cuda                         |
| daq::noise_model::PredicateFuncTy | q::sample_result::\~sample_result |
|     (C++                          |     (C++                          |
|     type)](api/la                 |     funct                         |
| nguages/cpp_api.html#_CPPv4N5cuda | ion)](api/languages/cpp_api.html# |
| q11noise_model15PredicateFuncTyE) | _CPPv4N5cudaq13sample_resultD0Ev) |
| -   [cud                          | -   [cudaq::scalar_callback (C++  |
| aq::noise_model::register_channel |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|     function)](api/languages      | #_CPPv4N5cudaq15scalar_callbackE) |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | -   [c                            |
| noise_model16register_channelEvv) | udaq::scalar_callback::operator() |
| -   [cudaq::                      |     (C++                          |
| noise_model::requires_constructor |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|     type)](api/languages/cp       | alar_callbackclERKNSt13unordered_ |
| p_api.html#_CPPv4I0DpEN5cudaq11no | mapINSt6stringENSt7complexIdEEEE) |
| ise_model20requires_constructorE) | -   [                             |
| -   [cudaq::noise_model_type (C++ | cudaq::scalar_callback::operator= |
|     e                             |     (C++                          |
| num)](api/languages/cpp_api.html# |     function)](api/languages/c    |
| _CPPv4N5cudaq16noise_model_typeE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::no                    | _callbackaSERK15scalar_callback), |
| ise_model_type::amplitude_damping |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|     enumerator)](api/languages    | r_callbackaSERR15scalar_callback) |
| /cpp_api.html#_CPPv4N5cudaq16nois | -   [cudaq:                       |
| e_model_type17amplitude_dampingE) | :scalar_callback::scalar_callback |
| -   [cudaq::noise_mode            |     (C++                          |
| l_type::amplitude_damping_channel |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4I0_NSt11ena |
|     e                             | ble_if_tINSt16is_invocable_r_vINS |
| numerator)](api/languages/cpp_api | t7complexIdEE8CallableRKNSt13unor |
| .html#_CPPv4N5cudaq16noise_model_ | dered_mapINSt6stringENSt7complexI |
| type25amplitude_damping_channelE) | dEEEEEEbEEEN5cudaq15scalar_callba |
| -   [cudaq::n                     | ck15scalar_callbackERR8Callable), |
| oise_model_type::bit_flip_channel |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](api/language     | Pv4N5cudaq15scalar_callback15scal |
| s/cpp_api.html#_CPPv4N5cudaq16noi | ar_callbackERK15scalar_callback), |
| se_model_type16bit_flip_channelE) |     [\[2                          |
| -   [cudaq::                      | \]](api/languages/cpp_api.html#_C |
| noise_model_type::depolarization1 | PPv4N5cudaq15scalar_callback15sca |
|     (C++                          | lar_callbackERR15scalar_callback) |
|     enumerator)](api/languag      | -   [cudaq::scalar_operator (C++  |
| es/cpp_api.html#_CPPv4N5cudaq16no |     c                             |
| ise_model_type15depolarization1E) | lass)](api/languages/cpp_api.html |
| -   [cudaq::                      | #_CPPv4N5cudaq15scalar_operatorE) |
| noise_model_type::depolarization2 | -                                 |
|     (C++                          | [cudaq::scalar_operator::evaluate |
|     enumerator)](api/languag      |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16no |                                   |
| ise_model_type15depolarization2E) |    function)](api/languages/cpp_a |
| -   [cudaq::noise_m               | pi.html#_CPPv4NK5cudaq15scalar_op |
| odel_type::depolarization_channel | erator8evaluateERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [cudaq::scalar_ope            |
|   enumerator)](api/languages/cpp_ | rator::get_parameter_descriptions |
| api.html#_CPPv4N5cudaq16noise_mod |     (C++                          |
| el_type22depolarization_channelE) |     f                             |
| -                                 | unction)](api/languages/cpp_api.h |
|  [cudaq::noise_model_type::pauli1 | tml#_CPPv4NK5cudaq15scalar_operat |
|     (C++                          | or26get_parameter_descriptionsEv) |
|     enumerator)](a                | -   [cu                           |
| pi/languages/cpp_api.html#_CPPv4N | daq::scalar_operator::is_constant |
| 5cudaq16noise_model_type6pauli1E) |     (C++                          |
| -                                 |     function)](api/lang           |
|  [cudaq::noise_model_type::pauli2 | uages/cpp_api.html#_CPPv4NK5cudaq |
|     (C++                          | 15scalar_operator11is_constantEv) |
|     enumerator)](a                | -   [c                            |
| pi/languages/cpp_api.html#_CPPv4N | udaq::scalar_operator::operator\* |
| 5cudaq16noise_model_type6pauli2E) |     (C++                          |
| -   [cudaq                        |     function                      |
| ::noise_model_type::phase_damping | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormlENSt |
|     enumerator)](api/langu        | 7complexIdEERK15scalar_operator), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     [\[1\                         |
| noise_model_type13phase_dampingE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noi                   | Pv4N5cudaq15scalar_operatormlENSt |
| se_model_type::phase_flip_channel | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     enumerator)](api/languages/   | p_api.html#_CPPv4N5cudaq15scalar_ |
| cpp_api.html#_CPPv4N5cudaq16noise | operatormlEdRK15scalar_operator), |
| _model_type18phase_flip_channelE) |     [\[3\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4N5cudaq15scalar_ |
| [cudaq::noise_model_type::unknown | operatormlEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     enumerator)](ap               | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| i/languages/cpp_api.html#_CPPv4N5 | alar_operatormlENSt7complexIdEE), |
| cudaq16noise_model_type7unknownE) |     [\[5\]](api/languages/cpp     |
| -                                 | _api.html#_CPPv4NKR5cudaq15scalar |
| [cudaq::noise_model_type::x_error | _operatormlERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     enumerator)](ap               | (api/languages/cpp_api.html#_CPPv |
| i/languages/cpp_api.html#_CPPv4N5 | 4NKR5cudaq15scalar_operatormlEd), |
| cudaq16noise_model_type7x_errorE) |     [\[7\]](api/language          |
| -                                 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| [cudaq::noise_model_type::y_error | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[8\]](api/languages/cp      |
|     enumerator)](ap               | p_api.html#_CPPv4NO5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatormlERK15scalar_operator), |
| cudaq16noise_model_type7y_errorE) |     [\[9\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
| [cudaq::noise_model_type::z_error | Pv4NO5cudaq15scalar_operatormlEd) |
|     (C++                          | -   [cu                           |
|     enumerator)](ap               | daq::scalar_operator::operator\*= |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7z_errorE) |     function)](api/languag        |
| -   [cudaq::num_available_gpus    | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormLENSt7complexIdEE), |
|     function                      |     [\[1\]](api/languages/c       |
| )](api/languages/cpp_api.html#_CP | pp_api.html#_CPPv4N5cudaq15scalar |
| Pv4N5cudaq18num_available_gpusEv) | _operatormLERK15scalar_operator), |
| -   [cudaq::observe (C++          |     [\[2                          |
|     function)]                    | \]](api/languages/cpp_api.html#_C |
| (api/languages/cpp_api.html#_CPPv | PPv4N5cudaq15scalar_operatormLEd) |
| 4I00DpEN5cudaq7observeENSt6vector | -   [                             |
| I14observe_resultEERR13QuantumKer | cudaq::scalar_operator::operator+ |
| nelRK15SpinOpContainerDpRR4Args), |     (C++                          |
|     [\[1\]](api/languages/cpp_ap  |     function                      |
| i.html#_CPPv4I0DpEN5cudaq7observe | )](api/languages/cpp_api.html#_CP |
| E14observe_resultNSt6size_tERR13Q | Pv4N5cudaq15scalar_operatorplENSt |
| uantumKernelRK7spin_opDpRR4Args), | 7complexIdEERK15scalar_operator), |
|     [\[                           |     [\[1\                         |
| 2\]](api/languages/cpp_api.html#_ | ]](api/languages/cpp_api.html#_CP |
| CPPv4I0DpEN5cudaq7observeE14obser | Pv4N5cudaq15scalar_operatorplENSt |
| ve_resultRK15observe_optionsRR13Q | 7complexIdEERR15scalar_operator), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[2\]](api/languages/cp      |
|     [\[3\]](api/lang              | p_api.html#_CPPv4N5cudaq15scalar_ |
| uages/cpp_api.html#_CPPv4I0DpEN5c | operatorplEdRK15scalar_operator), |
| udaq7observeE14observe_resultRR13 |     [\[3\]](api/languages/cp      |
| QuantumKernelRK7spin_opDpRR4Args) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::observe_options (C++  | operatorplEdRR15scalar_operator), |
|     st                            |     [\[4\]](api/languages         |
| ruct)](api/languages/cpp_api.html | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| #_CPPv4N5cudaq15observe_optionsE) | alar_operatorplENSt7complexIdEE), |
| -   [cudaq::observe_result (C++   |     [\[5\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4NKR5cudaq15scalar |
| class)](api/languages/cpp_api.htm | _operatorplERK15scalar_operator), |
| l#_CPPv4N5cudaq14observe_resultE) |     [\[6\]]                       |
| -                                 | (api/languages/cpp_api.html#_CPPv |
|    [cudaq::observe_result::counts | 4NKR5cudaq15scalar_operatorplEd), |
|     (C++                          |     [\[7\]]                       |
|     function)](api/languages/c    | (api/languages/cpp_api.html#_CPPv |
| pp_api.html#_CPPv4N5cudaq14observ | 4NKR5cudaq15scalar_operatorplEv), |
| e_result6countsERK12spin_op_term) |     [\[8\]](api/language          |
| -   [cudaq::observe_result::dump  | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatorplENSt7complexIdEE), |
|     function)                     |     [\[9\]](api/languages/cp      |
| ](api/languages/cpp_api.html#_CPP | p_api.html#_CPPv4NO5cudaq15scalar |
| v4N5cudaq14observe_result4dumpEv) | _operatorplERK15scalar_operator), |
| -   [c                            |     [\[10\]                       |
| udaq::observe_result::expectation | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatorplEd), |
|                                   |     [\[11\                        |
| function)](api/languages/cpp_api. | ]](api/languages/cpp_api.html#_CP |
| html#_CPPv4N5cudaq14observe_resul | Pv4NO5cudaq15scalar_operatorplEv) |
| t11expectationERK12spin_op_term), | -   [c                            |
|     [\[1\]](api/la                | udaq::scalar_operator::operator+= |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14observe_result11expectationEv) |     function)](api/languag        |
| -   [cuda                         | es/cpp_api.html#_CPPv4N5cudaq15sc |
| q::observe_result::id_coefficient | alar_operatorpLENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     function)](api/langu          | pp_api.html#_CPPv4N5cudaq15scalar |
| ages/cpp_api.html#_CPPv4N5cudaq14 | _operatorpLERK15scalar_operator), |
| observe_result14id_coefficientEv) |     [\[2                          |
| -   [cuda                         | \]](api/languages/cpp_api.html#_C |
| q::observe_result::observe_result | PPv4N5cudaq15scalar_operatorpLEd) |
|     (C++                          | -   [                             |
|                                   | cudaq::scalar_operator::operator- |
|   function)](api/languages/cpp_ap |     (C++                          |
| i.html#_CPPv4N5cudaq14observe_res |     function                      |
| ult14observe_resultEdRK7spin_op), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]](a                     | Pv4N5cudaq15scalar_operatormiENSt |
| pi/languages/cpp_api.html#_CPPv4N | 7complexIdEERK15scalar_operator), |
| 5cudaq14observe_result14observe_r |     [\[1\                         |
| esultEdRK7spin_op13sample_result) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatormiENSt |
|  [cudaq::observe_result::operator | 7complexIdEERR15scalar_operator), |
|     double (C++                   |     [\[2\]](api/languages/cp      |
|     functio                       | p_api.html#_CPPv4N5cudaq15scalar_ |
| n)](api/languages/cpp_api.html#_C | operatormiEdRK15scalar_operator), |
| PPv4N5cudaq14observe_resultcvdEv) |     [\[3\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4N5cudaq15scalar_ |
|  [cudaq::observe_result::raw_data | operatormiEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     function)](ap                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| i/languages/cpp_api.html#_CPPv4N5 | alar_operatormiENSt7complexIdEE), |
| cudaq14observe_result8raw_dataEv) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::operator_handler (C++ | _api.html#_CPPv4NKR5cudaq15scalar |
|     cl                            | _operatormiERK15scalar_operator), |
| ass)](api/languages/cpp_api.html# |     [\[6\]]                       |
| _CPPv4N5cudaq16operator_handlerE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::optimizable_function  | 4NKR5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[7\]]                       |
|     class)                        | (api/languages/cpp_api.html#_CPPv |
| ](api/languages/cpp_api.html#_CPP | 4NKR5cudaq15scalar_operatormiEv), |
| v4N5cudaq20optimizable_functionE) |     [\[8\]](api/language          |
| -   [cudaq::optimization_result   | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     type                          |     [\[9\]](api/languages/cp      |
| )](api/languages/cpp_api.html#_CP | p_api.html#_CPPv4NO5cudaq15scalar |
| Pv4N5cudaq19optimization_resultE) | _operatormiERK15scalar_operator), |
| -   [cudaq::optimizer (C++        |     [\[10\]                       |
|     class)](api/languages/cpp_a   | ](api/languages/cpp_api.html#_CPP |
| pi.html#_CPPv4N5cudaq9optimizerE) | v4NO5cudaq15scalar_operatormiEd), |
| -   [cudaq::optimizer::optimize   |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4NO5cudaq15scalar_operatormiEv) |
|  function)](api/languages/cpp_api | -   [c                            |
| .html#_CPPv4N5cudaq9optimizer8opt | udaq::scalar_operator::operator-= |
| imizeEKiRR20optimizable_function) |     (C++                          |
| -   [cu                           |     function)](api/languag        |
| daq::optimizer::requiresGradients | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormIENSt7complexIdEE), |
|     function)](api/la             |     [\[1\]](api/languages/c       |
| nguages/cpp_api.html#_CPPv4N5cuda | pp_api.html#_CPPv4N5cudaq15scalar |
| q9optimizer17requiresGradientsEv) | _operatormIERK15scalar_operator), |
| -   [cudaq::orca (C++             |     [\[2                          |
|     type)](api/languages/         | \]](api/languages/cpp_api.html#_C |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | PPv4N5cudaq15scalar_operatormIEd) |
| -   [cudaq::orca::sample (C++     | -   [                             |
|     function)](api/languages/c    | cudaq::scalar_operator::operator/ |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     (C++                          |
| mpleERNSt6vectorINSt6size_tEEERNS |     function                      |
| t6vectorINSt6size_tEEERNSt6vector | )](api/languages/cpp_api.html#_CP |
| IdEERNSt6vectorIdEEiNSt6size_tE), | Pv4N5cudaq15scalar_operatordvENSt |
|     [\[1\]]                       | 7complexIdEERK15scalar_operator), |
| (api/languages/cpp_api.html#_CPPv |     [\[1\                         |
| 4N5cudaq4orca6sampleERNSt6vectorI | ]](api/languages/cpp_api.html#_CP |
| NSt6size_tEEERNSt6vectorINSt6size | Pv4N5cudaq15scalar_operatordvENSt |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::orca::sample_async    |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|                                   | operatordvEdRK15scalar_operator), |
| function)](api/languages/cpp_api. |     [\[3\]](api/languages/cp      |
| html#_CPPv4N5cudaq4orca12sample_a | p_api.html#_CPPv4N5cudaq15scalar_ |
| syncERNSt6vectorINSt6size_tEEERNS | operatordvEdRR15scalar_operator), |
| t6vectorINSt6size_tEEERNSt6vector |     [\[4\]](api/languages         |
| IdEERNSt6vectorIdEEiNSt6size_tE), | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     [\[1\]](api/la                | alar_operatordvENSt7complexIdEE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[5\]](api/languages/cpp     |
| q4orca12sample_asyncERNSt6vectorI | _api.html#_CPPv4NKR5cudaq15scalar |
| NSt6size_tEEERNSt6vectorINSt6size | _operatordvERK15scalar_operator), |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     [\[6\]]                       |
| -   [cudaq::OrcaRemoteRESTQPU     | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatordvEd), |
|     cla                           |     [\[7\]](api/language          |
| ss)](api/languages/cpp_api.html#_ | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | alar_operatordvENSt7complexIdEE), |
| -   [cudaq::pauli1 (C++           |     [\[8\]](api/languages/cp      |
|     class)](api/languages/cp      | p_api.html#_CPPv4NO5cudaq15scalar |
| p_api.html#_CPPv4N5cudaq6pauli1E) | _operatordvERK15scalar_operator), |
| -                                 |     [\[9\                         |
|    [cudaq::pauli1::num_parameters | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatordvEd) |
|     member)]                      | -   [c                            |
| (api/languages/cpp_api.html#_CPPv | udaq::scalar_operator::operator/= |
| 4N5cudaq6pauli114num_parametersE) |     (C++                          |
| -   [cudaq::pauli1::num_targets   |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     membe                         | alar_operatordVENSt7complexIdEE), |
| r)](api/languages/cpp_api.html#_C |     [\[1\]](api/languages/c       |
| PPv4N5cudaq6pauli111num_targetsE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::pauli1::pauli1 (C++   | _operatordVERK15scalar_operator), |
|     function)](api/languages/cpp_ |     [\[2                          |
| api.html#_CPPv4N5cudaq6pauli16pau | \]](api/languages/cpp_api.html#_C |
| li1ERKNSt6vectorIN5cudaq4realEEE) | PPv4N5cudaq15scalar_operatordVEd) |
| -   [cudaq::pauli2 (C++           | -   [                             |
|     class)](api/languages/cp      | cudaq::scalar_operator::operator= |
| p_api.html#_CPPv4N5cudaq6pauli2E) |     (C++                          |
| -                                 |     function)](api/languages/c    |
|    [cudaq::pauli2::num_parameters | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatoraSERK15scalar_operator), |
|     member)]                      |     [\[1\]](api/languages/        |
| (api/languages/cpp_api.html#_CPPv | cpp_api.html#_CPPv4N5cudaq15scala |
| 4N5cudaq6pauli214num_parametersE) | r_operatoraSERR15scalar_operator) |
| -   [cudaq::pauli2::num_targets   | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator== |
|     membe                         |     (C++                          |
| r)](api/languages/cpp_api.html#_C |     function)](api/languages/c    |
| PPv4N5cudaq6pauli211num_targetsE) | pp_api.html#_CPPv4NK5cudaq15scala |
| -   [cudaq::pauli2::pauli2 (C++   | r_operatoreqERK15scalar_operator) |
|     function)](api/languages/cpp_ | -   [cudaq:                       |
| api.html#_CPPv4N5cudaq6pauli26pau | :scalar_operator::scalar_operator |
| li2ERKNSt6vectorIN5cudaq4realEEE) |     (C++                          |
| -   [cudaq::phase_damping (C++    |     func                          |
|                                   | tion)](api/languages/cpp_api.html |
|  class)](api/languages/cpp_api.ht | #_CPPv4N5cudaq15scalar_operator15 |
| ml#_CPPv4N5cudaq13phase_dampingE) | scalar_operatorENSt7complexIdEE), |
| -   [cud                          |     [\[1\]](api/langu             |
| aq::phase_damping::num_parameters | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     (C++                          | scalar_operator15scalar_operatorE |
|     member)](api/lan              | RK15scalar_callbackRRNSt13unorder |
| guages/cpp_api.html#_CPPv4N5cudaq | ed_mapINSt6stringENSt6stringEEE), |
| 13phase_damping14num_parametersE) |     [\[2\                         |
| -   [                             | ]](api/languages/cpp_api.html#_CP |
| cudaq::phase_damping::num_targets | Pv4N5cudaq15scalar_operator15scal |
|     (C++                          | ar_operatorERK15scalar_operator), |
|     member)](api/                 |     [\[3\]](api/langu             |
| languages/cpp_api.html#_CPPv4N5cu | ages/cpp_api.html#_CPPv4N5cudaq15 |
| daq13phase_damping11num_targetsE) | scalar_operator15scalar_operatorE |
| -   [cudaq::phase_flip_channel    | RR15scalar_callbackRRNSt13unorder |
|     (C++                          | ed_mapINSt6stringENSt6stringEEE), |
|     clas                          |     [\[4\                         |
| s)](api/languages/cpp_api.html#_C | ]](api/languages/cpp_api.html#_CP |
| PPv4N5cudaq18phase_flip_channelE) | Pv4N5cudaq15scalar_operator15scal |
| -   [cudaq::p                     | ar_operatorERR15scalar_operator), |
| hase_flip_channel::num_parameters |     [\[5\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     member)](api/language         | lar_operator15scalar_operatorEd), |
| s/cpp_api.html#_CPPv4N5cudaq18pha |     [\[6\]](api/languag           |
| se_flip_channel14num_parametersE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq                        | alar_operator15scalar_operatorEv) |
| ::phase_flip_channel::num_targets | -   [                             |
|     (C++                          | cudaq::scalar_operator::to_matrix |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq18 |                                   |
| phase_flip_channel11num_targetsE) |   function)](api/languages/cpp_ap |
| -   [cudaq::product_op (C++       | i.html#_CPPv4NK5cudaq15scalar_ope |
|                                   | rator9to_matrixERKNSt13unordered_ |
|  class)](api/languages/cpp_api.ht | mapINSt6stringENSt7complexIdEEEE) |
| ml#_CPPv4I0EN5cudaq10product_opE) | -   [                             |
| -   [cudaq::product_op::begin     | cudaq::scalar_operator::to_string |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api/l              |
| n)](api/languages/cpp_api.html#_C | anguages/cpp_api.html#_CPPv4NK5cu |
| PPv4NK5cudaq10product_op5beginEv) | daq15scalar_operator9to_stringEv) |
| -                                 | -   [cudaq::s                     |
|  [cudaq::product_op::canonicalize | calar_operator::\~scalar_operator |
|     (C++                          |     (C++                          |
|     func                          |     functio                       |
| tion)](api/languages/cpp_api.html | n)](api/languages/cpp_api.html#_C |
| #_CPPv4N5cudaq10product_op12canon | PPv4N5cudaq15scalar_operatorD0Ev) |
| icalizeERKNSt3setINSt6size_tEEE), | -   [cudaq::set_noise (C++        |
|     [\[1\]](api                   |     function)](api/langu          |
| /languages/cpp_api.html#_CPPv4N5c | ages/cpp_api.html#_CPPv4N5cudaq9s |
| udaq10product_op12canonicalizeEv) | et_noiseERKN5cudaq11noise_modelE) |
| -   [                             | -   [cudaq::set_random_seed (C++  |
| cudaq::product_op::const_iterator |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     struct)](api/                 | daq15set_random_seedENSt6size_tE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::simulation_precision  |
| daq10product_op14const_iteratorE) |     (C++                          |
| -   [cudaq::product_o             |     enum)                         |
| p::const_iterator::const_iterator | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq20simulation_precisionE) |
|     fu                            | -   [                             |
| nction)](api/languages/cpp_api.ht | cudaq::simulation_precision::fp32 |
| ml#_CPPv4N5cudaq10product_op14con |     (C++                          |
| st_iterator14const_iteratorEPK10p |     enumerator)](api              |
| roduct_opI9HandlerTyENSt6size_tE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::produ                 | udaq20simulation_precision4fp32E) |
| ct_op::const_iterator::operator!= | -   [                             |
|     (C++                          | cudaq::simulation_precision::fp64 |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     enumerator)](api              |
| l#_CPPv4NK5cudaq10product_op14con | /languages/cpp_api.html#_CPPv4N5c |
| st_iteratorneERK14const_iterator) | udaq20simulation_precision4fp64E) |
| -   [cudaq::produ                 | -   [cudaq::SimulationState (C++  |
| ct_op::const_iterator::operator\* |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|     function)](api/lang           | #_CPPv4N5cudaq15SimulationStateE) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [                             |
| 10product_op14const_iteratormlEv) | cudaq::SimulationState::precision |
| -   [cudaq::produ                 |     (C++                          |
| ct_op::const_iterator::operator++ |     enum)](api                    |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/lang           | udaq15SimulationState9precisionE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq:                       |
| 0product_op14const_iteratorppEi), | :SimulationState::precision::fp32 |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     enumerator)](api/lang         |
| 10product_op14const_iteratorppEv) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [cudaq::produc                | 5SimulationState9precision4fp32E) |
| t_op::const_iterator::operator\-- | -   [cudaq:                       |
|     (C++                          | :SimulationState::precision::fp64 |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     enumerator)](api/lang         |
| 0product_op14const_iteratormmEi), | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     [\[1\]](api/lan               | 5SimulationState9precision4fp64E) |
| guages/cpp_api.html#_CPPv4N5cudaq | -                                 |
| 10product_op14const_iteratormmEv) |   [cudaq::SimulationState::Tensor |
| -   [cudaq::produc                |     (C++                          |
| t_op::const_iterator::operator-\> |     struct)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/lan            | N5cudaq15SimulationState6TensorE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::spin_handler (C++     |
| 10product_op14const_iteratorptEv) |                                   |
| -   [cudaq::produ                 |   class)](api/languages/cpp_api.h |
| ct_op::const_iterator::operator== | tml#_CPPv4N5cudaq12spin_handlerE) |
|     (C++                          | -   [cudaq:                       |
|     fun                           | :spin_handler::to_diagonal_matrix |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4NK5cudaq10product_op14con |     function)](api/la             |
| st_iteratoreqERK14const_iterator) | nguages/cpp_api.html#_CPPv4NK5cud |
| -   [cudaq::product_op::degrees   | aq12spin_handler18to_diagonal_mat |
|     (C++                          | rixERNSt13unordered_mapINSt6size_ |
|     function)                     | tENSt7int64_tEEERKNSt13unordered_ |
| ](api/languages/cpp_api.html#_CPP | mapINSt6stringENSt7complexIdEEEE) |
| v4NK5cudaq10product_op7degreesEv) | -                                 |
|                                   |   [cudaq::spin_handler::to_matrix |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq12spin_handler9to_matri |
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
|                                   |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4I0_NSt11enable_if_tIXaantNSt7 |
|                                   | is_sameI1T9HandlerTyE5valueENSt16 |
|                                   | is_constructibleI9HandlerTy1TE5va |
|                                   | lueEEbEEEN5cudaq6sum_opaSER6sum_o |
|                                   | pI9HandlerTyERK10product_opI1TE), |
|                                   |                                   |
|                                   |  [\[1\]](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|                                   | tNSt7is_sameI1T9HandlerTyE5valueE |
|                                   | NSt16is_constructibleI9HandlerTy1 |
|                                   | TE5valueEEbEEEN5cudaq6sum_opaSER6 |
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
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | I0_NSt11enable_if_tIXaaNSt7is_sam |
|                                   | eI9HandlerTy14matrix_handlerE5val |
|                                   | ueEaantNSt7is_sameI1T9HandlerTyE5 |
|                                   | valueENSt16is_constructibleI9Hand |
|                                   | lerTy1TE5valueEEbEEEN5cudaq6sum_o |
|                                   | p6sum_opERK6sum_opI1TERKN14matrix |
|                                   | _handler20commutation_behaviorE), |
|                                   |     [\[1\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4I0_NSt11e |
|                                   | nable_if_tIXaantNSt7is_sameI1T9Ha |
|                                   | ndlerTyE5valueENSt16is_constructi |
|                                   | bleI9HandlerTy1TE5valueEEbEEEN5cu |
|                                   | daq6sum_op6sum_opERK6sum_opI1TE), |
|                                   |     [\[2\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4IDp_NSt |
|                                   | 11enable_if_tIXaaNSt11conjunction |
|                                   | IDpNSt7is_sameI10product_opI9Hand |
|                                   | lerTyE4ArgsEEE5valueEsZ4ArgsEbEEE |
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
| -   [f_tol (cudaq.optimizers.Adam | -   [from_json                    |
|     pro                           |     (                             |
| perty)](api/languages/python_api. | cudaq.gradients.CentralDifference |
| html#cudaq.optimizers.Adam.f_tol) |     attribute)](api/lang          |
|     -   [(cudaq.optimizers.SGD    | uages/python_api.html#cudaq.gradi |
|         pr                        | ents.CentralDifference.from_json) |
| operty)](api/languages/python_api |     -   [(                        |
| .html#cudaq.optimizers.SGD.f_tol) | cudaq.gradients.ForwardDifference |
| -   [FermionOperator (class in    |         attribute)](api/lang      |
|                                   | uages/python_api.html#cudaq.gradi |
|    cudaq.operators.fermion)](api/ | ents.ForwardDifference.from_json) |
| languages/python_api.html#cudaq.o |     -                             |
| perators.fermion.FermionOperator) |  [(cudaq.gradients.ParameterShift |
| -   [FermionOperatorElement       |         attribute)](api/l         |
|     (class in                     | anguages/python_api.html#cudaq.gr |
|     cuda                          | adients.ParameterShift.from_json) |
| q.operators.fermion)](api/languag |     -   [(                        |
| es/python_api.html#cudaq.operator | cudaq.operators.spin.SpinOperator |
| s.fermion.FermionOperatorElement) |         attribute)](api/lang      |
| -   [FermionOperatorTerm (class   | uages/python_api.html#cudaq.opera |
|     in                            | tors.spin.SpinOperator.from_json) |
|     c                             |     -   [(cuda                    |
| udaq.operators.fermion)](api/lang | q.operators.spin.SpinOperatorTerm |
| uages/python_api.html#cudaq.opera |         attribute)](api/language  |
| tors.fermion.FermionOperatorTerm) | s/python_api.html#cudaq.operators |
| -   [final_expectation_values     | .spin.SpinOperatorTerm.from_json) |
|     (cudaq.EvolveResult           |     -   [(cudaq.optimizers.Adam   |
|     attribute)](api/lang          |         attribut                  |
| uages/python_api.html#cudaq.Evolv | e)](api/languages/python_api.html |
| eResult.final_expectation_values) | #cudaq.optimizers.Adam.from_json) |
| -   [final_state                  |     -   [(cudaq.optimizers.COBYLA |
|     (cudaq.EvolveResult           |         attribute)                |
|     attribu                       | ](api/languages/python_api.html#c |
| te)](api/languages/python_api.htm | udaq.optimizers.COBYLA.from_json) |
| l#cudaq.EvolveResult.final_state) |     -   [                         |
| -   [finalize() (in module        | (cudaq.optimizers.GradientDescent |
|     cudaq.mpi)](api/languages/py  |         attribute)](api/lan       |
| thon_api.html#cudaq.mpi.finalize) | guages/python_api.html#cudaq.opti |
| -   [for_each_pauli               | mizers.GradientDescent.from_json) |
|     (                             |     -   [(cudaq.optimizers.LBFGS  |
| cudaq.operators.spin.SpinOperator |         attribute                 |
|     attribute)](api/languages     | )](api/languages/python_api.html# |
| /python_api.html#cudaq.operators. | cudaq.optimizers.LBFGS.from_json) |
| spin.SpinOperator.for_each_pauli) |                                   |
|     -   [(cuda                    | -   [(cudaq.optimizers.NelderMead |
| q.operators.spin.SpinOperatorTerm |         attribute)](ap            |
|                                   | i/languages/python_api.html#cudaq |
|     attribute)](api/languages/pyt | .optimizers.NelderMead.from_json) |
| hon_api.html#cudaq.operators.spin |     -   [(cudaq.optimizers.SGD    |
| .SpinOperatorTerm.for_each_pauli) |         attribu                   |
| -   [for_each_term                | te)](api/languages/python_api.htm |
|     (                             | l#cudaq.optimizers.SGD.from_json) |
| cudaq.operators.spin.SpinOperator |     -   [(cudaq.optimizers.SPSA   |
|     attribute)](api/language      |         attribut                  |
| s/python_api.html#cudaq.operators | e)](api/languages/python_api.html |
| .spin.SpinOperator.for_each_term) | #cudaq.optimizers.SPSA.from_json) |
| -   [ForwardDifference (class in  | -   [from_json()                  |
|     cudaq.gradients)              |     (cudaq.PyKernelDecorator      |
| ](api/languages/python_api.html#c |     static                        |
| udaq.gradients.ForwardDifference) |     method)                       |
| -   [from_data (cudaq.State       | ](api/languages/python_api.html#c |
|                                   | udaq.PyKernelDecorator.from_json) |
|   attribute)](api/languages/pytho | -   [from_word                    |
| n_api.html#cudaq.State.from_data) |     (                             |
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
| -   [I (cudaq.spin.Pauli          | -   [instructions                 |
|     attribute)](api/languages/py  |                                   |
| thon_api.html#cudaq.spin.Pauli.I) |   (cudaq.ptsbe.PTSBEExecutionData |
| -   [id                           |     property)](api/lang           |
|     (cuda                         | uages/python_api.html#cudaq.ptsbe |
| q.operators.MatrixOperatorElement | .PTSBEExecutionData.instructions) |
|     property)](api/l              | -   [intermediate_states          |
| anguages/python_api.html#cudaq.op |     (cudaq.EvolveResult           |
| erators.MatrixOperatorElement.id) |     attribute)](api               |
| -   [identity                     | /languages/python_api.html#cudaq. |
|     (cu                           | EvolveResult.intermediate_states) |
| daq.operators.boson.BosonOperator | -   [IntermediateResultSave       |
|     attribute)](api/langu         |     (class in                     |
| ages/python_api.html#cudaq.operat |     c                             |
| ors.boson.BosonOperator.identity) | udaq)](api/languages/python_api.h |
|     -   [(cudaq.                  | tml#cudaq.IntermediateResultSave) |
| operators.fermion.FermionOperator | -   [is_compiled()                |
|         attribute)](api/languages |     (cudaq.PyKernelDecorator      |
| /python_api.html#cudaq.operators. |     method)](                     |
| fermion.FermionOperator.identity) | api/languages/python_api.html#cud |
|     -                             | aq.PyKernelDecorator.is_compiled) |
|  [(cudaq.operators.MatrixOperator | -   [is_constant                  |
|         attribute)](api/          |                                   |
| languages/python_api.html#cudaq.o |   (cudaq.operators.ScalarOperator |
| perators.MatrixOperator.identity) |     attribute)](api/lan           |
|     -   [(                        | guages/python_api.html#cudaq.oper |
| cudaq.operators.spin.SpinOperator | ators.ScalarOperator.is_constant) |
|         attribute)](api/lan       | -   [is_emulated (cudaq.Target    |
| guages/python_api.html#cudaq.oper |     a                             |
| ators.spin.SpinOperator.identity) | ttribute)](api/languages/python_a |
| -   [initial_parameters           | pi.html#cudaq.Target.is_emulated) |
|     (cudaq.optimizers.Adam        | -   [is_error                     |
|     property)](api/l              |     (cudaq.ptsbe.KrausSelection   |
| anguages/python_api.html#cudaq.op |     property)](                   |
| timizers.Adam.initial_parameters) | api/languages/python_api.html#cud |
|     -   [(cudaq.optimizers.COBYLA | aq.ptsbe.KrausSelection.is_error) |
|         property)](api/lan        | -   [is_identity                  |
| guages/python_api.html#cudaq.opti |     (cudaq.                       |
| mizers.COBYLA.initial_parameters) | operators.boson.BosonOperatorTerm |
|     -   [                         |     attribute)](api/languages/py  |
| (cudaq.optimizers.GradientDescent | thon_api.html#cudaq.operators.bos |
|                                   | on.BosonOperatorTerm.is_identity) |
|       property)](api/languages/py |     -   [(cudaq.oper              |
| thon_api.html#cudaq.optimizers.Gr | ators.fermion.FermionOperatorTerm |
| adientDescent.initial_parameters) |                                   |
|     -   [(cudaq.optimizers.LBFGS  |  attribute)](api/languages/python |
|         property)](api/la         | _api.html#cudaq.operators.fermion |
| nguages/python_api.html#cudaq.opt | .FermionOperatorTerm.is_identity) |
| imizers.LBFGS.initial_parameters) |     -   [(c                       |
|                                   | udaq.operators.MatrixOperatorTerm |
| -   [(cudaq.optimizers.NelderMead |         attribute)](api/languag   |
|         property)](api/languag    | es/python_api.html#cudaq.operator |
| es/python_api.html#cudaq.optimize | s.MatrixOperatorTerm.is_identity) |
| rs.NelderMead.initial_parameters) |     -   [(                        |
|     -   [(cudaq.optimizers.SGD    | cudaq.operators.spin.SpinOperator |
|         property)](api/           |         attribute)](api/langua    |
| languages/python_api.html#cudaq.o | ges/python_api.html#cudaq.operato |
| ptimizers.SGD.initial_parameters) | rs.spin.SpinOperator.is_identity) |
|     -   [(cudaq.optimizers.SPSA   |     -   [(cuda                    |
|         property)](api/l          | q.operators.spin.SpinOperatorTerm |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.SPSA.initial_parameters) |        attribute)](api/languages/ |
| -   [initialize() (in module      | python_api.html#cudaq.operators.s |
|                                   | pin.SpinOperatorTerm.is_identity) |
|    cudaq.mpi)](api/languages/pyth | -   [is_initialized() (in module  |
| on_api.html#cudaq.mpi.initialize) |     c                             |
| -   [initialize_cudaq() (in       | udaq.mpi)](api/languages/python_a |
|     module                        | pi.html#cudaq.mpi.is_initialized) |
|     cudaq)](api/languages/python  | -   [is_on_gpu (cudaq.State       |
| _api.html#cudaq.initialize_cudaq) |                                   |
| -   [InitialState (in module      |   attribute)](api/languages/pytho |
|     cudaq.dynamics.helpers)](     | n_api.html#cudaq.State.is_on_gpu) |
| api/languages/python_api.html#cud | -   [is_remote (cudaq.Target      |
| aq.dynamics.helpers.InitialState) |                                   |
| -   [InitialStateType (class in   |  attribute)](api/languages/python |
|     cudaq)](api/languages/python  | _api.html#cudaq.Target.is_remote) |
| _api.html#cudaq.InitialStateType) | -   [is_remote_simulator          |
| -   [instantiate()                |     (cudaq.Target                 |
|     (cudaq.operators              |     attribute                     |
|     m                             | )](api/languages/python_api.html# |
| ethod)](api/languages/python_api. | cudaq.Target.is_remote_simulator) |
| html#cudaq.operators.instantiate) | -   [items (cudaq.SampleResult    |
|     -   [(in module               |     a                             |
|         cudaq.operators.custom)]  | ttribute)](api/languages/python_a |
| (api/languages/python_api.html#cu | pi.html#cudaq.SampleResult.items) |
| daq.operators.custom.instantiate) |                                   |
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
| -   [observe() (in module         | -   [OptimizationResult (in       |
|     cudaq)](api/languag           |     module                        |
| es/python_api.html#cudaq.observe) |                                   |
| -   [observe_async() (in module   |    cudaq)](api/languages/python_a |
|     cudaq)](api/languages/pyt     | pi.html#cudaq.OptimizationResult) |
| hon_api.html#cudaq.observe_async) | -   [OrderedSamplingStrategy      |
| -   [ObserveResult (class in      |     (class in                     |
|     cudaq)](api/languages/pyt     |     cudaq.ptsbe)](                |
| hon_api.html#cudaq.ObserveResult) | api/languages/python_api.html#cud |
| -   [op_name                      | aq.ptsbe.OrderedSamplingStrategy) |
|     (cudaq.ptsbe.KrausSelection   | -   [overlap (cudaq.State         |
|     property)]                    |     attribute)](api/languages/pyt |
| (api/languages/python_api.html#cu | hon_api.html#cudaq.State.overlap) |
| daq.ptsbe.KrausSelection.op_name) |                                   |
| -   [OperatorSum (in module       |                                   |
|     cudaq.oper                    |                                   |
| ators)](api/languages/python_api. |                                   |
| html#cudaq.operators.OperatorSum) |                                   |
| -   [ops_count                    |                                   |
|     (cudaq.                       |                                   |
| operators.boson.BosonOperatorTerm |                                   |
|     property)](api/languages/     |                                   |
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
|     -   [(cudaq.SampleResult      | -   [SPSA (class in               |
|         attri                     |     cudaq                         |
| bute)](api/languages/python_api.h | .optimizers)](api/languages/pytho |
| tml#cudaq.SampleResult.serialize) | n_api.html#cudaq.optimizers.SPSA) |
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
|       property)](api/languages/py |     -   [(                        |
| thon_api.html#cudaq.operators.fer | cudaq.operators.spin.SpinOperator |
| mion.FermionOperatorTerm.term_id) |         attribute)](api/lang      |
|     -   [(c                       | uages/python_api.html#cudaq.opera |
| udaq.operators.MatrixOperatorTerm | tors.spin.SpinOperator.to_string) |
|         property)](api/lan        |     -   [(cudaq.o                 |
| guages/python_api.html#cudaq.oper | perators.spin.SpinOperatorElement |
| ators.MatrixOperatorTerm.term_id) |                                   |
|     -   [(cuda                    |       attribute)](api/languages/p |
| q.operators.spin.SpinOperatorTerm | ython_api.html#cudaq.operators.sp |
|         property)](api/langua     | in.SpinOperatorElement.to_string) |
| ges/python_api.html#cudaq.operato |     -   [(cuda                    |
| rs.spin.SpinOperatorTerm.term_id) | q.operators.spin.SpinOperatorTerm |
| -   [to_dict (cudaq.Resources     |         attribute)](api/language  |
|                                   | s/python_api.html#cudaq.operators |
| attribute)](api/languages/python_ | .spin.SpinOperatorTerm.to_string) |
| api.html#cudaq.Resources.to_dict) | -   [TraceInstruction (class in   |
| -   [to_json                      |     cudaq.p                       |
|     (                             | tsbe)](api/languages/python_api.h |
| cudaq.gradients.CentralDifference | tml#cudaq.ptsbe.TraceInstruction) |
|     attribute)](api/la            | -   [TraceInstructionType (class  |
| nguages/python_api.html#cudaq.gra |     in                            |
| dients.CentralDifference.to_json) |     cudaq.ptsbe                   |
|     -   [(                        | )](api/languages/python_api.html# |
| cudaq.gradients.ForwardDifference | cudaq.ptsbe.TraceInstructionType) |
|         attribute)](api/la        | -   [trajectories                 |
| nguages/python_api.html#cudaq.gra |                                   |
| dients.ForwardDifference.to_json) |   (cudaq.ptsbe.PTSBEExecutionData |
|     -                             |     property)](api/lang           |
|  [(cudaq.gradients.ParameterShift | uages/python_api.html#cudaq.ptsbe |
|         attribute)](api           | .PTSBEExecutionData.trajectories) |
| /languages/python_api.html#cudaq. | -   [trajectory_id                |
| gradients.ParameterShift.to_json) |     (cudaq.ptsbe.KrausTrajectory  |
|     -   [(                        |     property)](api/la             |
| cudaq.operators.spin.SpinOperator | nguages/python_api.html#cudaq.pts |
|         attribute)](api/la        | be.KrausTrajectory.trajectory_id) |
| nguages/python_api.html#cudaq.ope | -   [translate() (in module       |
| rators.spin.SpinOperator.to_json) |     cudaq)](api/languages         |
|     -   [(cuda                    | /python_api.html#cudaq.translate) |
| q.operators.spin.SpinOperatorTerm | -   [trim                         |
|         attribute)](api/langua    |     (cu                           |
| ges/python_api.html#cudaq.operato | daq.operators.boson.BosonOperator |
| rs.spin.SpinOperatorTerm.to_json) |     attribute)](api/l             |
|     -   [(cudaq.optimizers.Adam   | anguages/python_api.html#cudaq.op |
|         attrib                    | erators.boson.BosonOperator.trim) |
| ute)](api/languages/python_api.ht |     -   [(cudaq.                  |
| ml#cudaq.optimizers.Adam.to_json) | operators.fermion.FermionOperator |
|     -   [(cudaq.optimizers.COBYLA |         attribute)](api/langu     |
|         attribut                  | ages/python_api.html#cudaq.operat |
| e)](api/languages/python_api.html | ors.fermion.FermionOperator.trim) |
| #cudaq.optimizers.COBYLA.to_json) |     -                             |
|     -   [                         |  [(cudaq.operators.MatrixOperator |
| (cudaq.optimizers.GradientDescent |         attribute)](              |
|         attribute)](api/l         | api/languages/python_api.html#cud |
| anguages/python_api.html#cudaq.op | aq.operators.MatrixOperator.trim) |
| timizers.GradientDescent.to_json) |     -   [(                        |
|     -   [(cudaq.optimizers.LBFGS  | cudaq.operators.spin.SpinOperator |
|         attribu                   |         attribute)](api           |
| te)](api/languages/python_api.htm | /languages/python_api.html#cudaq. |
| l#cudaq.optimizers.LBFGS.to_json) | operators.spin.SpinOperator.trim) |
|                                   | -   [type                         |
| -   [(cudaq.optimizers.NelderMead |     (c                            |
|         attribute)](              | udaq.ptsbe.ShotAllocationStrategy |
| api/languages/python_api.html#cud |     property)](api/               |
| aq.optimizers.NelderMead.to_json) | languages/python_api.html#cudaq.p |
|     -   [(cudaq.optimizers.SGD    | tsbe.ShotAllocationStrategy.type) |
|         attri                     |     -                             |
| bute)](api/languages/python_api.h |    [(cudaq.ptsbe.TraceInstruction |
| tml#cudaq.optimizers.SGD.to_json) |         property)                 |
|     -   [(cudaq.optimizers.SPSA   | ](api/languages/python_api.html#c |
|         attrib                    | udaq.ptsbe.TraceInstruction.type) |
| ute)](api/languages/python_api.ht | -   [type_to_str()                |
| ml#cudaq.optimizers.SPSA.to_json) |     (cudaq.PyKernelDecorator      |
| -   [to_json()                    |     static                        |
|     (cudaq.PyKernelDecorator      |     method)](                     |
|     metho                         | api/languages/python_api.html#cud |
| d)](api/languages/python_api.html | aq.PyKernelDecorator.type_to_str) |
| #cudaq.PyKernelDecorator.to_json) |                                   |
| -   [to_matrix                    |                                   |
|     (cu                           |                                   |
| daq.operators.boson.BosonOperator |                                   |
|     attribute)](api/langua        |                                   |
| ges/python_api.html#cudaq.operato |                                   |
| rs.boson.BosonOperator.to_matrix) |                                   |
|     -   [(cudaq.ope               |                                   |
| rators.boson.BosonOperatorElement |                                   |
|                                   |                                   |
|     attribute)](api/languages/pyt |                                   |
| hon_api.html#cudaq.operators.boso |                                   |
| n.BosonOperatorElement.to_matrix) |                                   |
|     -   [(cudaq.                  |                                   |
| operators.boson.BosonOperatorTerm |                                   |
|                                   |                                   |
|        attribute)](api/languages/ |                                   |
| python_api.html#cudaq.operators.b |                                   |
| oson.BosonOperatorTerm.to_matrix) |                                   |
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
