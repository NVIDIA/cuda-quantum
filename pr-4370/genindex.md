::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4370
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
| -   [canonicalize                 | -   [cudaq::product_op::degrees   |
|     (cu                           |     (C++                          |
| daq.operators.boson.BosonOperator |     function)                     |
|     attribute)](api/languages     | ](api/languages/cpp_api.html#_CPP |
| /python_api.html#cudaq.operators. | v4NK5cudaq10product_op7degreesEv) |
| boson.BosonOperator.canonicalize) | -   [cudaq::product_op::dump (C++ |
|     -   [(cudaq.                  |     functi                        |
| operators.boson.BosonOperatorTerm | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq10product_op4dumpEv) |
|     attribute)](api/languages/pyt | -   [cudaq::product_op::end (C++  |
| hon_api.html#cudaq.operators.boso |     funct                         |
| n.BosonOperatorTerm.canonicalize) | ion)](api/languages/cpp_api.html# |
|     -   [(cudaq.                  | _CPPv4NK5cudaq10product_op3endEv) |
| operators.fermion.FermionOperator | -   [c                            |
|                                   | udaq::product_op::get_coefficient |
|     attribute)](api/languages/pyt |     (C++                          |
| hon_api.html#cudaq.operators.ferm |     function)](api/lan            |
| ion.FermionOperator.canonicalize) | guages/cpp_api.html#_CPPv4NK5cuda |
|     -   [(cudaq.oper              | q10product_op15get_coefficientEv) |
| ators.fermion.FermionOperatorTerm | -                                 |
|                                   |   [cudaq::product_op::get_term_id |
| attribute)](api/languages/python_ |     (C++                          |
| api.html#cudaq.operators.fermion. |     function)](api                |
| FermionOperatorTerm.canonicalize) | /languages/cpp_api.html#_CPPv4NK5 |
|     -                             | cudaq10product_op11get_term_idEv) |
|  [(cudaq.operators.MatrixOperator | -                                 |
|         attribute)](api/lang      |   [cudaq::product_op::is_identity |
| uages/python_api.html#cudaq.opera |     (C++                          |
| tors.MatrixOperator.canonicalize) |     function)](api                |
|     -   [(c                       | /languages/cpp_api.html#_CPPv4NK5 |
| udaq.operators.MatrixOperatorTerm | cudaq10product_op11is_identityEv) |
|         attribute)](api/language  | -   [cudaq::product_op::num_ops   |
| s/python_api.html#cudaq.operators |     (C++                          |
| .MatrixOperatorTerm.canonicalize) |     function)                     |
|     -   [(                        | ](api/languages/cpp_api.html#_CPP |
| cudaq.operators.spin.SpinOperator | v4NK5cudaq10product_op7num_opsEv) |
|         attribute)](api/languag   | -                                 |
| es/python_api.html#cudaq.operator |    [cudaq::product_op::operator\* |
| s.spin.SpinOperator.canonicalize) |     (C++                          |
|     -   [(cuda                    |     function)](api/languages/     |
| q.operators.spin.SpinOperatorTerm | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|                                   | oduct_opmlE10product_opI1TERK15sc |
|       attribute)](api/languages/p | alar_operatorRK10product_opI1TE), |
| ython_api.html#cudaq.operators.sp |     [\[1\]](api/languages/        |
| in.SpinOperatorTerm.canonicalize) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [captured_variables()         | oduct_opmlE10product_opI1TERK15sc |
|     (cudaq.PyKernelDecorator      | alar_operatorRR10product_opI1TE), |
|     method)](api/lan              |     [\[2\]](api/languages/        |
| guages/python_api.html#cudaq.PyKe | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| rnelDecorator.captured_variables) | oduct_opmlE10product_opI1TERR15sc |
| -   [CentralDifference (class in  | alar_operatorRK10product_opI1TE), |
|     cudaq.gradients)              |     [\[3\]](api/languages/        |
| ](api/languages/python_api.html#c | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| udaq.gradients.CentralDifference) | oduct_opmlE10product_opI1TERR15sc |
| -   [channel                      | alar_operatorRR10product_opI1TE), |
|     (cudaq.ptsbe.TraceInstruction |     [\[4\]](api/                  |
|     property)](a                  | languages/cpp_api.html#_CPPv4I0EN |
| pi/languages/python_api.html#cuda | 5cudaq10product_opmlE6sum_opI1TER |
| q.ptsbe.TraceInstruction.channel) | K15scalar_operatorRK6sum_opI1TE), |
| -   [circuit_location             |     [\[5\]](api/                  |
|     (cudaq.ptsbe.KrausSelection   | languages/cpp_api.html#_CPPv4I0EN |
|     property)](api/lang           | 5cudaq10product_opmlE6sum_opI1TER |
| uages/python_api.html#cudaq.ptsbe | K15scalar_operatorRR6sum_opI1TE), |
| .KrausSelection.circuit_location) |     [\[6\]](api/                  |
| -   [clear (cudaq.Resources       | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opmlE6sum_opI1TER |
|   attribute)](api/languages/pytho | R15scalar_operatorRK6sum_opI1TE), |
| n_api.html#cudaq.Resources.clear) |     [\[7\]](api/                  |
|     -   [(cudaq.SampleResult      | languages/cpp_api.html#_CPPv4I0EN |
|         a                         | 5cudaq10product_opmlE6sum_opI1TER |
| ttribute)](api/languages/python_a | R15scalar_operatorRR6sum_opI1TE), |
| pi.html#cudaq.SampleResult.clear) |     [\[8\]](api/languages         |
| -   [COBYLA (class in             | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     cudaq.o                       | duct_opmlERK6sum_opI9HandlerTyE), |
| ptimizers)](api/languages/python_ |     [\[9\]](api/languages/cpp_a   |
| api.html#cudaq.optimizers.COBYLA) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [coefficient                  | opmlERK10product_opI9HandlerTyE), |
|     (cudaq.                       |     [\[10\]](api/language         |
| operators.boson.BosonOperatorTerm | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     property)](api/languages/py   | roduct_opmlERK15scalar_operator), |
| thon_api.html#cudaq.operators.bos |     [\[11\]](api/languages/cpp_a  |
| on.BosonOperatorTerm.coefficient) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [(cudaq.oper              | opmlERR10product_opI9HandlerTyE), |
| ators.fermion.FermionOperatorTerm |     [\[12\]](api/language         |
|                                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|   property)](api/languages/python | roduct_opmlERR15scalar_operator), |
| _api.html#cudaq.operators.fermion |     [\[13\]](api/languages/cpp_   |
| .FermionOperatorTerm.coefficient) | api.html#_CPPv4NO5cudaq10product_ |
|     -   [(c                       | opmlERK10product_opI9HandlerTyE), |
| udaq.operators.MatrixOperatorTerm |     [\[14\]](api/languag          |
|         property)](api/languag    | es/cpp_api.html#_CPPv4NO5cudaq10p |
| es/python_api.html#cudaq.operator | roduct_opmlERK15scalar_operator), |
| s.MatrixOperatorTerm.coefficient) |     [\[15\]](api/languages/cpp_   |
|     -   [(cuda                    | api.html#_CPPv4NO5cudaq10product_ |
| q.operators.spin.SpinOperatorTerm | opmlERR10product_opI9HandlerTyE), |
|         property)](api/languages/ |     [\[16\]](api/langua           |
| python_api.html#cudaq.operators.s | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| pin.SpinOperatorTerm.coefficient) | product_opmlERR15scalar_operator) |
| -   [col_count                    | -                                 |
|     (cudaq.KrausOperator          |   [cudaq::product_op::operator\*= |
|     prope                         |     (C++                          |
| rty)](api/languages/python_api.ht |     function)](api/languages/cpp  |
| ml#cudaq.KrausOperator.col_count) | _api.html#_CPPv4N5cudaq10product_ |
| -   [compile()                    | opmLERK10product_opI9HandlerTyE), |
|     (cudaq.PyKernelDecorator      |     [\[1\]](api/langua            |
|     metho                         | ges/cpp_api.html#_CPPv4N5cudaq10p |
| d)](api/languages/python_api.html | roduct_opmLERK15scalar_operator), |
| #cudaq.PyKernelDecorator.compile) |     [\[2\]](api/languages/cp      |
| -   [ComplexMatrix (class in      | p_api.html#_CPPv4N5cudaq10product |
|     cudaq)](api/languages/pyt     | _opmLERR10product_opI9HandlerTyE) |
| hon_api.html#cudaq.ComplexMatrix) | -   [cudaq::product_op::operator+ |
| -   [compute                      |     (C++                          |
|     (                             |     function)](api/langu          |
| cudaq.gradients.CentralDifference | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     attribute)](api/la            | q10product_opplE6sum_opI1TERK15sc |
| nguages/python_api.html#cudaq.gra | alar_operatorRK10product_opI1TE), |
| dients.CentralDifference.compute) |     [\[1\]](api/                  |
|     -   [(                        | languages/cpp_api.html#_CPPv4I0EN |
| cudaq.gradients.ForwardDifference | 5cudaq10product_opplE6sum_opI1TER |
|         attribute)](api/la        | K15scalar_operatorRK6sum_opI1TE), |
| nguages/python_api.html#cudaq.gra |     [\[2\]](api/langu             |
| dients.ForwardDifference.compute) | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     -                             | q10product_opplE6sum_opI1TERK15sc |
|  [(cudaq.gradients.ParameterShift | alar_operatorRR10product_opI1TE), |
|         attribute)](api           |     [\[3\]](api/                  |
| /languages/python_api.html#cudaq. | languages/cpp_api.html#_CPPv4I0EN |
| gradients.ParameterShift.compute) | 5cudaq10product_opplE6sum_opI1TER |
| -   [const()                      | K15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[4\]](api/langu             |
|   (cudaq.operators.ScalarOperator | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     class                         | q10product_opplE6sum_opI1TERR15sc |
|     method)](a                    | alar_operatorRK10product_opI1TE), |
| pi/languages/python_api.html#cuda |     [\[5\]](api/                  |
| q.operators.ScalarOperator.const) | languages/cpp_api.html#_CPPv4I0EN |
| -   [controls                     | 5cudaq10product_opplE6sum_opI1TER |
|     (cudaq.ptsbe.TraceInstruction | R15scalar_operatorRK6sum_opI1TE), |
|     property)](ap                 |     [\[6\]](api/langu             |
| i/languages/python_api.html#cudaq | ages/cpp_api.html#_CPPv4I0EN5cuda |
| .ptsbe.TraceInstruction.controls) | q10product_opplE6sum_opI1TERR15sc |
| -   [copy                         | alar_operatorRR10product_opI1TE), |
|     (cu                           |     [\[7\]](api/                  |
| daq.operators.boson.BosonOperator | languages/cpp_api.html#_CPPv4I0EN |
|     attribute)](api/l             | 5cudaq10product_opplE6sum_opI1TER |
| anguages/python_api.html#cudaq.op | R15scalar_operatorRR6sum_opI1TE), |
| erators.boson.BosonOperator.copy) |     [\[8\]](api/languages/cpp_a   |
|     -   [(cudaq.                  | pi.html#_CPPv4NKR5cudaq10product_ |
| operators.boson.BosonOperatorTerm | opplERK10product_opI9HandlerTyE), |
|         attribute)](api/langu     |     [\[9\]](api/language          |
| ages/python_api.html#cudaq.operat | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ors.boson.BosonOperatorTerm.copy) | roduct_opplERK15scalar_operator), |
|     -   [(cudaq.                  |     [\[10\]](api/languages/       |
| operators.fermion.FermionOperator | cpp_api.html#_CPPv4NKR5cudaq10pro |
|         attribute)](api/langu     | duct_opplERK6sum_opI9HandlerTyE), |
| ages/python_api.html#cudaq.operat |     [\[11\]](api/languages/cpp_a  |
| ors.fermion.FermionOperator.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [(cudaq.oper              | opplERR10product_opI9HandlerTyE), |
| ators.fermion.FermionOperatorTerm |     [\[12\]](api/language         |
|         attribute)](api/languages | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /python_api.html#cudaq.operators. | roduct_opplERR15scalar_operator), |
| fermion.FermionOperatorTerm.copy) |     [\[13\]](api/languages/       |
|     -                             | cpp_api.html#_CPPv4NKR5cudaq10pro |
|  [(cudaq.operators.MatrixOperator | duct_opplERR6sum_opI9HandlerTyE), |
|         attribute)](              |     [\[                           |
| api/languages/python_api.html#cud | 14\]](api/languages/cpp_api.html# |
| aq.operators.MatrixOperator.copy) | _CPPv4NKR5cudaq10product_opplEv), |
|     -   [(c                       |     [\[15\]](api/languages/cpp_   |
| udaq.operators.MatrixOperatorTerm | api.html#_CPPv4NO5cudaq10product_ |
|         attribute)](api/          | opplERK10product_opI9HandlerTyE), |
| languages/python_api.html#cudaq.o |     [\[16\]](api/languag          |
| perators.MatrixOperatorTerm.copy) | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     -   [(                        | roduct_opplERK15scalar_operator), |
| cudaq.operators.spin.SpinOperator |     [\[17\]](api/languages        |
|         attribute)](api           | /cpp_api.html#_CPPv4NO5cudaq10pro |
| /languages/python_api.html#cudaq. | duct_opplERK6sum_opI9HandlerTyE), |
| operators.spin.SpinOperator.copy) |     [\[18\]](api/languages/cpp_   |
|     -   [(cuda                    | api.html#_CPPv4NO5cudaq10product_ |
| q.operators.spin.SpinOperatorTerm | opplERR10product_opI9HandlerTyE), |
|         attribute)](api/lan       |     [\[19\]](api/languag          |
| guages/python_api.html#cudaq.oper | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ators.spin.SpinOperatorTerm.copy) | roduct_opplERR15scalar_operator), |
| -   [count (cudaq.Resources       |     [\[20\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq10pro |
|   attribute)](api/languages/pytho | duct_opplERR6sum_opI9HandlerTyE), |
| n_api.html#cudaq.Resources.count) |     [                             |
|     -   [(cudaq.SampleResult      | \[21\]](api/languages/cpp_api.htm |
|         a                         | l#_CPPv4NO5cudaq10product_opplEv) |
| ttribute)](api/languages/python_a | -   [cudaq::product_op::operator- |
| pi.html#cudaq.SampleResult.count) |     (C++                          |
| -   [count_controls               |     function)](api/langu          |
|     (cudaq.Resources              | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     attribu                       | q10product_opmiE6sum_opI1TERK15sc |
| te)](api/languages/python_api.htm | alar_operatorRK10product_opI1TE), |
| l#cudaq.Resources.count_controls) |     [\[1\]](api/                  |
| -   [count_instructions           | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opmiE6sum_opI1TER |
|   (cudaq.ptsbe.PTSBEExecutionData | K15scalar_operatorRK6sum_opI1TE), |
|     attribute)](api/languages/    |     [\[2\]](api/langu             |
| python_api.html#cudaq.ptsbe.PTSBE | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ExecutionData.count_instructions) | q10product_opmiE6sum_opI1TERK15sc |
| -   [counts (cudaq.ObserveResult  | alar_operatorRR10product_opI1TE), |
|     att                           |     [\[3\]](api/                  |
| ribute)](api/languages/python_api | languages/cpp_api.html#_CPPv4I0EN |
| .html#cudaq.ObserveResult.counts) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [csr_spmatrix (C++            | K15scalar_operatorRR6sum_opI1TE), |
|     type)](api/languages/c        |     [\[4\]](api/langu             |
| pp_api.html#_CPPv412csr_spmatrix) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq                         | q10product_opmiE6sum_opI1TERR15sc |
|     -   [module](api/langua       | alar_operatorRK10product_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[5\]](api/                  |
| -   [cudaq (C++                   | languages/cpp_api.html#_CPPv4I0EN |
|     type)](api/lan                | 5cudaq10product_opmiE6sum_opI1TER |
| guages/cpp_api.html#_CPPv45cudaq) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq.apply_noise() (in      |     [\[6\]](api/langu             |
|     module                        | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cudaq)](api/languages/python_ | q10product_opmiE6sum_opI1TERR15sc |
| api.html#cudaq.cudaq.apply_noise) | alar_operatorRR10product_opI1TE), |
| -   cudaq.boson                   |     [\[7\]](api/                  |
|     -   [module](api/languages/py | languages/cpp_api.html#_CPPv4I0EN |
| thon_api.html#module-cudaq.boson) | 5cudaq10product_opmiE6sum_opI1TER |
| -   cudaq.fermion                 | R15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[8\]](api/languages/cpp_a   |
|   -   [module](api/languages/pyth | pi.html#_CPPv4NKR5cudaq10product_ |
| on_api.html#module-cudaq.fermion) | opmiERK10product_opI9HandlerTyE), |
| -   cudaq.operators.custom        |     [\[9\]](api/language          |
|     -   [mo                       | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| dule](api/languages/python_api.ht | roduct_opmiERK15scalar_operator), |
| ml#module-cudaq.operators.custom) |     [\[10\]](api/languages/       |
| -   cudaq.spin                    | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     -   [module](api/languages/p  | duct_opmiERK6sum_opI9HandlerTyE), |
| ython_api.html#module-cudaq.spin) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq::amplitude_damping     | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmiERR10product_opI9HandlerTyE), |
|     cla                           |     [\[12\]](api/language         |
| ss)](api/languages/cpp_api.html#_ | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| CPPv4N5cudaq17amplitude_dampingE) | roduct_opmiERR15scalar_operator), |
| -                                 |     [\[13\]](api/languages/       |
| [cudaq::amplitude_damping_channel | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     (C++                          | duct_opmiERR6sum_opI9HandlerTyE), |
|     class)](api                   |     [\[                           |
| /languages/cpp_api.html#_CPPv4N5c | 14\]](api/languages/cpp_api.html# |
| udaq25amplitude_damping_channelE) | _CPPv4NKR5cudaq10product_opmiEv), |
| -   [cudaq::amplitud              |     [\[15\]](api/languages/cpp_   |
| e_damping_channel::num_parameters | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmiERK10product_opI9HandlerTyE), |
|     member)](api/languages/cpp_a  |     [\[16\]](api/languag          |
| pi.html#_CPPv4N5cudaq25amplitude_ | es/cpp_api.html#_CPPv4NO5cudaq10p |
| damping_channel14num_parametersE) | roduct_opmiERK15scalar_operator), |
| -   [cudaq::ampli                 |     [\[17\]](api/languages        |
| tude_damping_channel::num_targets | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERK6sum_opI9HandlerTyE), |
|     member)](api/languages/cp     |     [\[18\]](api/languages/cpp_   |
| p_api.html#_CPPv4N5cudaq25amplitu | api.html#_CPPv4NO5cudaq10product_ |
| de_damping_channel11num_targetsE) | opmiERR10product_opI9HandlerTyE), |
| -   [cudaq::AnalogRemoteRESTQPU   |     [\[19\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     class                         | roduct_opmiERR15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[20\]](api/languages        |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::apply_noise (C++      | duct_opmiERR6sum_opI9HandlerTyE), |
|     function)](api/               |     [                             |
| languages/cpp_api.html#_CPPv4I0Dp | \[21\]](api/languages/cpp_api.htm |
| EN5cudaq11apply_noiseEvDpRR4Args) | l#_CPPv4NO5cudaq10product_opmiEv) |
| -   [cudaq::async_result (C++     | -   [cudaq::product_op::operator/ |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     function)](api/language       |
| #_CPPv4I0EN5cudaq12async_resultE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::async_result::get     | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/language          |
|     functi                        | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| on)](api/languages/cpp_api.html#_ | roduct_opdvERR15scalar_operator), |
| CPPv4N5cudaq12async_result3getEv) |     [\[2\]](api/languag           |
| -   [cudaq::async_sample_result   | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opdvERK15scalar_operator), |
|     type                          |     [\[3\]](api/langua            |
| )](api/languages/cpp_api.html#_CP | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| Pv4N5cudaq19async_sample_resultE) | product_opdvERR15scalar_operator) |
| -   [cudaq::BaseRemoteRESTQPU     | -                                 |
|     (C++                          |    [cudaq::product_op::operator/= |
|     cla                           |     (C++                          |
| ss)](api/languages/cpp_api.html#_ |     function)](api/langu          |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | ages/cpp_api.html#_CPPv4N5cudaq10 |
| -                                 | product_opdVERK15scalar_operator) |
|    [cudaq::BaseRemoteSimulatorQPU | -   [cudaq::product_op::operator= |
|     (C++                          |     (C++                          |
|     class)](                      |     function)](api/l              |
| api/languages/cpp_api.html#_CPPv4 | anguages/cpp_api.html#_CPPv4I00EN |
| N5cudaq22BaseRemoteSimulatorQPUE) | 5cudaq10product_opaSER10product_o |
| -   [cudaq::bit_flip_channel (C++ | pI9HandlerTyERK10product_opI1TE), |
|     cl                            |     [\[1\]](api/languages/cpp     |
| ass)](api/languages/cpp_api.html# | _api.html#_CPPv4N5cudaq10product_ |
| _CPPv4N5cudaq16bit_flip_channelE) | opaSERK10product_opI9HandlerTyE), |
| -   [cudaq:                       |     [\[2\]](api/languages/cp      |
| :bit_flip_channel::num_parameters | p_api.html#_CPPv4N5cudaq10product |
|     (C++                          | _opaSERR10product_opI9HandlerTyE) |
|     member)](api/langua           | -                                 |
| ges/cpp_api.html#_CPPv4N5cudaq16b |    [cudaq::product_op::operator== |
| it_flip_channel14num_parametersE) |     (C++                          |
| -   [cud                          |     function)](api/languages/cpp  |
| aq::bit_flip_channel::num_targets | _api.html#_CPPv4NK5cudaq10product |
|     (C++                          | _opeqERK10product_opI9HandlerTyE) |
|     member)](api/lan              | -                                 |
| guages/cpp_api.html#_CPPv4N5cudaq |  [cudaq::product_op::operator\[\] |
| 16bit_flip_channel11num_targetsE) |     (C++                          |
| -   [cudaq::boson_handler (C++    |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|  class)](api/languages/cpp_api.ht | 5cudaq10product_opixENSt6size_tE) |
| ml#_CPPv4N5cudaq13boson_handlerE) | -                                 |
| -   [cudaq::boson_op (C++         |    [cudaq::product_op::product_op |
|     type)](api/languages/cpp_     |     (C++                          |
| api.html#_CPPv4N5cudaq8boson_opE) |     f                             |
| -   [cudaq::boson_op_term (C++    | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4I00EN5cudaq10product_op |
|   type)](api/languages/cpp_api.ht | 10product_opERK10product_opI1TE), |
| ml#_CPPv4N5cudaq13boson_op_termE) |     [\[1\]]                       |
| -   [cudaq::CodeGenConfig (C++    | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I00EN5cudaq10product_op10product |
| struct)](api/languages/cpp_api.ht | _opERK10product_opI1TERKN14matrix |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | _handler20commutation_behaviorE), |
| -   [cudaq::commutation_relations |                                   |
|     (C++                          |   [\[2\]](api/languages/cpp_api.h |
|     struct)]                      | tml#_CPPv4N5cudaq10product_op10pr |
| (api/languages/cpp_api.html#_CPPv | oduct_opENSt6size_tENSt6size_tE), |
| 4N5cudaq21commutation_relationsE) |     [\[3\]](api/languages/cp      |
| -   [cudaq::complex (C++          | p_api.html#_CPPv4N5cudaq10product |
|     type)](api/languages/cpp      | _op10product_opENSt7complexIdEE), |
| _api.html#_CPPv4N5cudaq7complexE) |     [\[4\]](api/l                 |
| -   [cudaq::complex_matrix (C++   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq10product_op10product_opERK10pr |
| class)](api/languages/cpp_api.htm | oduct_opI9HandlerTyENSt6size_tE), |
| l#_CPPv4N5cudaq14complex_matrixE) |     [\[5\]](api/l                 |
| -                                 | anguages/cpp_api.html#_CPPv4N5cud |
|   [cudaq::complex_matrix::adjoint | aq10product_op10product_opERR10pr |
|     (C++                          | oduct_opI9HandlerTyENSt6size_tE), |
|     function)](a                  |     [\[6\]](api/languages         |
| pi/languages/cpp_api.html#_CPPv4N | /cpp_api.html#_CPPv4N5cudaq10prod |
| 5cudaq14complex_matrix7adjointEv) | uct_op10product_opERR9HandlerTy), |
| -   [cudaq::                      |     [\[7\]](ap                    |
| complex_matrix::diagonal_elements | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq10product_op10product_opEd), |
|     function)](api/languages      |     [\[8\]](a                     |
| /cpp_api.html#_CPPv4NK5cudaq14com | pi/languages/cpp_api.html#_CPPv4N |
| plex_matrix17diagonal_elementsEi) | 5cudaq10product_op10product_opEv) |
| -   [cudaq::complex_matrix::dump  | -   [cuda                         |
|     (C++                          | q::product_op::to_diagonal_matrix |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4NK5cudaq14co |     function)](api/               |
| mplex_matrix4dumpERNSt7ostreamE), | languages/cpp_api.html#_CPPv4NK5c |
|     [\[1\]]                       | udaq10product_op18to_diagonal_mat |
| (api/languages/cpp_api.html#_CPPv | rixENSt13unordered_mapINSt6size_t |
| 4NK5cudaq14complex_matrix4dumpEv) | ENSt7int64_tEEERKNSt13unordered_m |
| -   [c                            | apINSt6stringENSt7complexIdEEEEb) |
| udaq::complex_matrix::eigenvalues | -   [cudaq::product_op::to_matrix |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     funct                         |
| guages/cpp_api.html#_CPPv4NK5cuda | ion)](api/languages/cpp_api.html# |
| q14complex_matrix11eigenvaluesEv) | _CPPv4NK5cudaq10product_op9to_mat |
| -   [cu                           | rixENSt13unordered_mapINSt6size_t |
| daq::complex_matrix::eigenvectors | ENSt7int64_tEEERKNSt13unordered_m |
|     (C++                          | apINSt6stringENSt7complexIdEEEEb) |
|     function)](api/lang           | -   [cu                           |
| uages/cpp_api.html#_CPPv4NK5cudaq | daq::product_op::to_sparse_matrix |
| 14complex_matrix12eigenvectorsEv) |     (C++                          |
| -   [c                            |     function)](ap                 |
| udaq::complex_matrix::exponential | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq10product_op16to_sparse_mat |
|     function)](api/la             | rixENSt13unordered_mapINSt6size_t |
| nguages/cpp_api.html#_CPPv4N5cuda | ENSt7int64_tEEERKNSt13unordered_m |
| q14complex_matrix11exponentialEv) | apINSt6stringENSt7complexIdEEEEb) |
| -                                 | -   [cudaq::product_op::to_string |
|  [cudaq::complex_matrix::identity |     (C++                          |
|     (C++                          |     function)](                   |
|     function)](api/languages      | api/languages/cpp_api.html#_CPPv4 |
| /cpp_api.html#_CPPv4N5cudaq14comp | NK5cudaq10product_op9to_stringEv) |
| lex_matrix8identityEKNSt6size_tE) | -                                 |
| -                                 |  [cudaq::product_op::\~product_op |
| [cudaq::complex_matrix::kronecker |     (C++                          |
|     (C++                          |     fu                            |
|     function)](api/lang           | nction)](api/languages/cpp_api.ht |
| uages/cpp_api.html#_CPPv4I00EN5cu | ml#_CPPv4N5cudaq10product_opD0Ev) |
| daq14complex_matrix9kroneckerE14c | -   [cudaq::ptsbe (C++            |
| omplex_matrix8Iterable8Iterable), |     type)](api/languages/c        |
|     [\[1\]](api/l                 | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cudaq::p                     |
| aq14complex_matrix9kroneckerERK14 | tsbe::ConditionalSamplingStrategy |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -   [cudaq::c                     |     class)](api/languag           |
| omplex_matrix::minimal_eigenvalue | es/cpp_api.html#_CPPv4N5cudaq5pts |
|     (C++                          | be27ConditionalSamplingStrategyE) |
|     function)](api/languages/     | -   [cudaq::ptsbe::C              |
| cpp_api.html#_CPPv4NK5cudaq14comp | onditionalSamplingStrategy::clone |
| lex_matrix18minimal_eigenvalueEv) |     (C++                          |
| -   [                             |                                   |
| cudaq::complex_matrix::operator() |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
|     function)](api/languages/cpp  | ditionalSamplingStrategy5cloneEv) |
| _api.html#_CPPv4N5cudaq14complex_ | -   [cuda                         |
| matrixclENSt6size_tENSt6size_tE), | q::ptsbe::ConditionalSamplingStra |
|     [\[1\]](api/languages/cpp     | tegy::ConditionalSamplingStrategy |
| _api.html#_CPPv4NK5cudaq14complex |     (C++                          |
| _matrixclENSt6size_tENSt6size_tE) |     function)](api/lang           |
| -   [                             | uages/cpp_api.html#_CPPv4N5cudaq5 |
| cudaq::complex_matrix::operator\* | ptsbe27ConditionalSamplingStrateg |
|     (C++                          | y27ConditionalSamplingStrategyE19 |
|     function)](api/langua         | TrajectoryPredicateNSt8uint64_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq14c | -                                 |
| omplex_matrixmlEN14complex_matrix |   [cudaq::ptsbe::ConditionalSampl |
| 10value_typeERK14complex_matrix), | ingStrategy::generateTrajectories |
|     [\[1\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/language       |
| v4N5cudaq14complex_matrixmlERK14c | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| omplex_matrixRK14complex_matrix), | be27ConditionalSamplingStrategy20 |
|                                   | generateTrajectoriesENSt4spanIKN6 |
|  [\[2\]](api/languages/cpp_api.ht | detail10NoisePointEEENSt6size_tE) |
| ml#_CPPv4N5cudaq14complex_matrixm | -   [cudaq::ptsbe::               |
| lERK14complex_matrixRKNSt6vectorI | ConditionalSamplingStrategy::name |
| N14complex_matrix10value_typeEEE) |     (C++                          |
| -                                 |     function)](api/languages/cpp_ |
| [cudaq::complex_matrix::operator+ | api.html#_CPPv4NK5cudaq5ptsbe27Co |
|     (C++                          | nditionalSamplingStrategy4nameEv) |
|     function                      | -   [cudaq:                       |
| )](api/languages/cpp_api.html#_CP | :ptsbe::ConditionalSamplingStrate |
| Pv4N5cudaq14complex_matrixplERK14 | gy::\~ConditionalSamplingStrategy |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -                                 |     function)](api/languages/     |
| [cudaq::complex_matrix::operator- | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
|     (C++                          | 7ConditionalSamplingStrategyD0Ev) |
|     function                      | -                                 |
| )](api/languages/cpp_api.html#_CP | [cudaq::ptsbe::detail::NoisePoint |
| Pv4N5cudaq14complex_matrixmiERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     struct)](a                    |
| -   [cu                           | pi/languages/cpp_api.html#_CPPv4N |
| daq::complex_matrix::operator\[\] | 5cudaq5ptsbe6detail10NoisePointE) |
|     (C++                          | -   [cudaq::p                     |
|                                   | tsbe::detail::NoisePoint::channel |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4N5cudaq14complex_matr |     member)](api/langu            |
| ixixERKNSt6vectorINSt6size_tEEE), | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     [\[1\]](api/languages/cpp_api | tsbe6detail10NoisePoint7channelE) |
| .html#_CPPv4NK5cudaq14complex_mat | -   [cudaq::ptsbe::det            |
| rixixERKNSt6vectorINSt6size_tEEE) | ail::NoisePoint::circuit_location |
| -   [cudaq::complex_matrix::power |     (C++                          |
|     (C++                          |     member)](api/languages/cpp_a  |
|     function)]                    | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| (api/languages/cpp_api.html#_CPPv | l10NoisePoint16circuit_locationE) |
| 4N5cudaq14complex_matrix5powerEi) | -   [cudaq::p                     |
| -                                 | tsbe::detail::NoisePoint::op_name |
|  [cudaq::complex_matrix::set_zero |     (C++                          |
|     (C++                          |     member)](api/langu            |
|     function)](ap                 | ages/cpp_api.html#_CPPv4N5cudaq5p |
| i/languages/cpp_api.html#_CPPv4N5 | tsbe6detail10NoisePoint7op_nameE) |
| cudaq14complex_matrix8set_zeroEv) | -   [cudaq::                      |
| -                                 | ptsbe::detail::NoisePoint::qubits |
| [cudaq::complex_matrix::to_string |     (C++                          |
|     (C++                          |     member)](api/lang             |
|     function)](api/               | uages/cpp_api.html#_CPPv4N5cudaq5 |
| languages/cpp_api.html#_CPPv4NK5c | ptsbe6detail10NoisePoint6qubitsE) |
| udaq14complex_matrix9to_stringEv) | -   [cudaq::                      |
| -   [                             | ptsbe::ExhaustiveSamplingStrategy |
| cudaq::complex_matrix::value_type |     (C++                          |
|     (C++                          |     class)](api/langua            |
|     type)](api/                   | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| languages/cpp_api.html#_CPPv4N5cu | sbe26ExhaustiveSamplingStrategyE) |
| daq14complex_matrix10value_typeE) | -   [cudaq::ptsbe::               |
| -   [cudaq::contrib (C++          | ExhaustiveSamplingStrategy::clone |
|     type)](api/languages/cpp      |     (C++                          |
| _api.html#_CPPv4N5cudaq7contribE) |     function)](api/languages/cpp_ |
| -   [cudaq::contrib::draw (C++    | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
|     function)                     | haustiveSamplingStrategy5cloneEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cu                           |
| v4I0DpEN5cudaq7contrib4drawENSt6s | daq::ptsbe::ExhaustiveSamplingStr |
| tringERR13QuantumKernelDpRR4Args) | ategy::ExhaustiveSamplingStrategy |
| -                                 |     (C++                          |
| [cudaq::contrib::get_unitary_cmat |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     function)](api/languages/cp   | q5ptsbe26ExhaustiveSamplingStrate |
| p_api.html#_CPPv4I0DpEN5cudaq7con | gy26ExhaustiveSamplingStrategyEv) |
| trib16get_unitary_cmatE14complex_ | -                                 |
| matrixRR13QuantumKernelDpRR4Args) |    [cudaq::ptsbe::ExhaustiveSampl |
| -   [cudaq::CusvState (C++        | ingStrategy::generateTrajectories |
|                                   |     (C++                          |
|    class)](api/languages/cpp_api. |     function)](api/languag        |
| html#_CPPv4I0EN5cudaq9CusvStateE) | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| -   [cudaq::depolarization1 (C++  | sbe26ExhaustiveSamplingStrategy20 |
|     c                             | generateTrajectoriesENSt4spanIKN6 |
| lass)](api/languages/cpp_api.html | detail10NoisePointEEENSt6size_tE) |
| #_CPPv4N5cudaq15depolarization1E) | -   [cudaq::ptsbe:                |
| -   [cudaq::depolarization2 (C++  | :ExhaustiveSamplingStrategy::name |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     function)](api/languages/cpp  |
| #_CPPv4N5cudaq15depolarization2E) | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| -   [cudaq:                       | xhaustiveSamplingStrategy4nameEv) |
| :depolarization2::depolarization2 | -   [cuda                         |
|     (C++                          | q::ptsbe::ExhaustiveSamplingStrat |
|     function)](api/languages/cp   | egy::\~ExhaustiveSamplingStrategy |
| p_api.html#_CPPv4N5cudaq15depolar |     (C++                          |
| ization215depolarization2EK4real) |     function)](api/languages      |
| -   [cudaq                        | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| ::depolarization2::num_parameters | 26ExhaustiveSamplingStrategyD0Ev) |
|     (C++                          | -   [cuda                         |
|     member)](api/langu            | q::ptsbe::OrderedSamplingStrategy |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| depolarization214num_parametersE) |     class)](api/lan               |
| -   [cu                           | guages/cpp_api.html#_CPPv4N5cudaq |
| daq::depolarization2::num_targets | 5ptsbe23OrderedSamplingStrategyE) |
|     (C++                          | -   [cudaq::ptsb                  |
|     member)](api/la               | e::OrderedSamplingStrategy::clone |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q15depolarization211num_targetsE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
|    [cudaq::depolarization_channel | 3OrderedSamplingStrategy5cloneEv) |
|     (C++                          | -   [cudaq::ptsbe::OrderedSampl   |
|     class)](                      | ingStrategy::generateTrajectories |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22depolarization_channelE) |     function)](api/lang           |
| -   [cudaq::depol                 | uages/cpp_api.html#_CPPv4NK5cudaq |
| arization_channel::num_parameters | 5ptsbe23OrderedSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/languages/cp     | detail10NoisePointEEENSt6size_tE) |
| p_api.html#_CPPv4N5cudaq22depolar | -   [cudaq::pts                   |
| ization_channel14num_parametersE) | be::OrderedSamplingStrategy::name |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     function)](api/languages/     |
|     (C++                          | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     member)](api/languages        | 23OrderedSamplingStrategy4nameEv) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -                                 |
| larization_channel11num_targetsE) |    [cudaq::ptsbe::OrderedSampling |
| -   [cudaq::details (C++          | Strategy::OrderedSamplingStrategy |
|     type)](api/languages/cpp      |     (C++                          |
| _api.html#_CPPv4N5cudaq7detailsE) |     function)](                   |
| -   [cudaq::details::future (C++  | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq5ptsbe23OrderedSamplingStr |
|  class)](api/languages/cpp_api.ht | ategy23OrderedSamplingStrategyEv) |
| ml#_CPPv4N5cudaq7details6futureE) | -                                 |
| -                                 |  [cudaq::ptsbe::OrderedSamplingSt |
|   [cudaq::details::future::future | rategy::\~OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api/langua         |
| n)](api/languages/cpp_api.html#_C | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| PPv4N5cudaq7details6future6future | sbe23OrderedSamplingStrategyD0Ev) |
| ERNSt6vectorI3JobEERNSt6stringERN | -   [cudaq::pts                   |
| St3mapINSt6stringENSt6stringEEE), | be::ProbabilisticSamplingStrategy |
|     [\[1\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     class)](api/languages         |
| details6future6futureERR6future), | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     [\[2\]]                       | 29ProbabilisticSamplingStrategyE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::ptsbe::Pro            |
| 4N5cudaq7details6future6futureEv) | babilisticSamplingStrategy::clone |
| -   [cu                           |     (C++                          |
| daq::details::kernel_builder_base |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     class)](api/l                 | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| anguages/cpp_api.html#_CPPv4N5cud | bilisticSamplingStrategy5cloneEv) |
| aq7details19kernel_builder_baseE) | -                                 |
| -   [cudaq::details::             | [cudaq::ptsbe::ProbabilisticSampl |
| kernel_builder_base::operator\<\< | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/languages/     |
| ges/cpp_api.html#_CPPv4N5cudaq7de | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| tails19kernel_builder_baselsERNSt | 29ProbabilisticSamplingStrategy20 |
| 7ostreamERK19kernel_builder_base) | generateTrajectoriesENSt4spanIKN6 |
| -   [                             | detail10NoisePointEEENSt6size_tE) |
| cudaq::details::KernelBuilderType | -   [cudaq::ptsbe::Pr             |
|     (C++                          | obabilisticSamplingStrategy::name |
|     class)](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |                                   |
| udaq7details17KernelBuilderTypeE) |   function)](api/languages/cpp_ap |
| -   [cudaq::d                     | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| etails::KernelBuilderType::create | abilisticSamplingStrategy4nameEv) |
|     (C++                          | -   [cudaq::p                     |
|     function)                     | tsbe::ProbabilisticSamplingStrate |
| ](api/languages/cpp_api.html#_CPP | gy::ProbabilisticSamplingStrategy |
| v4N5cudaq7details17KernelBuilderT |     (C++                          |
| ype6createEPN4mlir11MLIRContextE) |     function)]                    |
| -   [cudaq::details::Ker          | (api/languages/cpp_api.html#_CPPv |
| nelBuilderType::KernelBuilderType | 4N5cudaq5ptsbe29ProbabilisticSamp |
|     (C++                          | lingStrategy29ProbabilisticSampli |
|     function)](api/lang           | ngStrategyENSt8optionalINSt8uint6 |
| uages/cpp_api.html#_CPPv4N5cudaq7 | 4_tEEENSt8optionalINSt6size_tEEE) |
| details17KernelBuilderType17Kerne | -   [cudaq::pts                   |
| lBuilderTypeERRNSt8functionIFN4ml | be::ProbabilisticSamplingStrategy |
| ir4TypeEPN4mlir11MLIRContextEEEE) | ::\~ProbabilisticSamplingStrategy |
| -   [cudaq::diag_matrix_callback  |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     class)                        | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| ](api/languages/cpp_api.html#_CPP | robabilisticSamplingStrategyD0Ev) |
| v4N5cudaq20diag_matrix_callbackE) | -                                 |
| -   [cudaq::dyn (C++              | [cudaq::ptsbe::PTSBEExecutionData |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq3dynE) |     struct)](ap                   |
| -   [cudaq::ExecutionContext (C++ | i/languages/cpp_api.html#_CPPv4N5 |
|     cl                            | cudaq5ptsbe18PTSBEExecutionDataE) |
| ass)](api/languages/cpp_api.html# | -   [cudaq::ptsbe::PTSBE          |
| _CPPv4N5cudaq16ExecutionContextE) | ExecutionData::count_instructions |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::amplitudeMaps |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     member)](api/langu            | daq5ptsbe18PTSBEExecutionData18co |
| ages/cpp_api.html#_CPPv4N5cudaq16 | unt_instructionsE20TraceInstructi |
| ExecutionContext13amplitudeMapsE) | onTypeNSt8optionalINSt6stringEEE) |
| -   [c                            | -   [cudaq::ptsbe::P              |
| udaq::ExecutionContext::asyncExec | TSBEExecutionData::get_trajectory |
|     (C++                          |     (C++                          |
|     member)](api/                 |     function                      |
| languages/cpp_api.html#_CPPv4N5cu | )](api/languages/cpp_api.html#_CP |
| daq16ExecutionContext9asyncExecE) | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| -   [cud                          | Data14get_trajectoryENSt6size_tE) |
| aq::ExecutionContext::asyncResult | -   [cudaq::ptsbe:                |
|     (C++                          | :PTSBEExecutionData::instructions |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     member)](api/languages/cp     |
| 16ExecutionContext11asyncResultE) | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| -   [cudaq:                       | TSBEExecutionData12instructionsE) |
| :ExecutionContext::batchIteration | -   [cudaq::ptsbe:                |
|     (C++                          | :PTSBEExecutionData::trajectories |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     member)](api/languages/cp     |
| xecutionContext14batchIterationE) | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| -   [cudaq::E                     | TSBEExecutionData12trajectoriesE) |
| xecutionContext::canHandleObserve | -   [cudaq::ptsbe::PTSBEOptions   |
|     (C++                          |     (C++                          |
|     member)](api/language         |     struc                         |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | t)](api/languages/cpp_api.html#_C |
| cutionContext16canHandleObserveE) | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| -   [cudaq::E                     | -   [cudaq::ptsbe::PTSB           |
| xecutionContext::ExecutionContext | EOptions::include_sequential_data |
|     (C++                          |     (C++                          |
|     func                          |                                   |
| tion)](api/languages/cpp_api.html |    member)](api/languages/cpp_api |
| #_CPPv4N5cudaq16ExecutionContext1 | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| 6ExecutionContextERKNSt6stringE), | ptions23include_sequential_dataE) |
|     [\[1\]](api/languages/        | -   [cudaq::ptsb                  |
| cpp_api.html#_CPPv4N5cudaq16Execu | e::PTSBEOptions::max_trajectories |
| tionContext16ExecutionContextERKN |     (C++                          |
| St6stringENSt6size_tENSt6size_tE) |     member)](api/languages/       |
| -   [cudaq::E                     | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| xecutionContext::expectationValue | 2PTSBEOptions16max_trajectoriesE) |
|     (C++                          | -   [cudaq::ptsbe::PT             |
|     member)](api/language         | SBEOptions::return_execution_data |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     (C++                          |
| cutionContext16expectationValueE) |     member)](api/languages/cpp_a  |
| -   [cudaq::Execu                 | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| tionContext::explicitMeasurements | EOptions21return_execution_dataE) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/languages/cp     | be::PTSBEOptions::shot_allocation |
| p_api.html#_CPPv4N5cudaq16Executi |     (C++                          |
| onContext20explicitMeasurementsE) |     member)](api/languages        |
| -   [cuda                         | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| q::ExecutionContext::futureResult | 12PTSBEOptions15shot_allocationE) |
|     (C++                          | -   [cud                          |
|     member)](api/lang             | aq::ptsbe::PTSBEOptions::strategy |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     (C++                          |
| 6ExecutionContext12futureResultE) |     member)](api/l                |
| -   [cudaq::ExecutionContext      | anguages/cpp_api.html#_CPPv4N5cud |
| ::hasConditionalsOnMeasureResults | aq5ptsbe12PTSBEOptions8strategyE) |
|     (C++                          | -   [cudaq::ptsbe::PTSBETrace     |
|     mem                           |     (C++                          |
| ber)](api/languages/cpp_api.html# |     t                             |
| _CPPv4N5cudaq16ExecutionContext31 | ype)](api/languages/cpp_api.html# |
| hasConditionalsOnMeasureResultsE) | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| -   [cudaq::Executi               | -   [                             |
| onContext::invocationResultBuffer | cudaq::ptsbe::PTSSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/languages/cpp_   |     class)](api                   |
| api.html#_CPPv4N5cudaq16Execution | /languages/cpp_api.html#_CPPv4N5c |
| Context22invocationResultBufferE) | udaq5ptsbe19PTSSamplingStrategyE) |
| -   [cu                           | -   [cudaq::                      |
| daq::ExecutionContext::kernelName | ptsbe::PTSSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api/languag        |
| nguages/cpp_api.html#_CPPv4N5cuda | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| q16ExecutionContext10kernelNameE) | sbe19PTSSamplingStrategy5cloneEv) |
| -   [cud                          | -   [cudaq::ptsbe::PTSSampl       |
| aq::ExecutionContext::kernelTrace | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/               |
| guages/cpp_api.html#_CPPv4N5cudaq | languages/cpp_api.html#_CPPv4NK5c |
| 16ExecutionContext11kernelTraceE) | udaq5ptsbe19PTSSamplingStrategy20 |
| -   [cudaq:                       | generateTrajectoriesENSt4spanIKN6 |
| :ExecutionContext::msm_dimensions | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     member)](api/langua           | :ptsbe::PTSSamplingStrategy::name |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14msm_dimensionsE) |     function)](api/langua         |
| -   [cudaq::                      | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| ExecutionContext::msm_prob_err_id | tsbe19PTSSamplingStrategy4nameEv) |
|     (C++                          | -   [cudaq::ptsbe::PTSSampli      |
|     member)](api/languag          | ngStrategy::\~PTSSamplingStrategy |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15msm_prob_err_idE) |     function)](api/la             |
| -   [cudaq::Ex                    | nguages/cpp_api.html#_CPPv4N5cuda |
| ecutionContext::msm_probabilities | q5ptsbe19PTSSamplingStrategyD0Ev) |
|     (C++                          | -   [cudaq::ptsbe::sample (C++    |
|     member)](api/languages        |                                   |
| /cpp_api.html#_CPPv4N5cudaq16Exec |  function)](api/languages/cpp_api |
| utionContext17msm_probabilitiesE) | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| -                                 | mpleE13sample_resultRK14sample_op |
|    [cudaq::ExecutionContext::name | tionsRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\[1\]](api                   |
|     member)]                      | /languages/cpp_api.html#_CPPv4I0D |
| (api/languages/cpp_api.html#_CPPv | pEN5cudaq5ptsbe6sampleE13sample_r |
| 4N5cudaq16ExecutionContext4nameE) | esultRKN5cudaq11noise_modelENSt6s |
| -   [cu                           | ize_tERR13QuantumKernelDpRR4Args) |
| daq::ExecutionContext::noiseModel | -   [cudaq::ptsbe::sample_async   |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](a                  |
| nguages/cpp_api.html#_CPPv4N5cuda | pi/languages/cpp_api.html#_CPPv4I |
| q16ExecutionContext10noiseModelE) | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| -   [cudaq::Exe                   | 9async_sample_resultRK14sample_op |
| cutionContext::numberTrajectories | tionsRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\[1\]](api/languages/cp      |
|     member)](api/languages/       | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| cpp_api.html#_CPPv4N5cudaq16Execu | be12sample_asyncE19async_sample_r |
| tionContext18numberTrajectoriesE) | esultRKN5cudaq11noise_modelENSt6s |
| -   [c                            | ize_tERR13QuantumKernelDpRR4Args) |
| udaq::ExecutionContext::optResult | -   [cudaq::ptsbe::sample_options |
|     (C++                          |     (C++                          |
|     member)](api/                 |     struct)                       |
| languages/cpp_api.html#_CPPv4N5cu | ](api/languages/cpp_api.html#_CPP |
| daq16ExecutionContext9optResultE) | v4N5cudaq5ptsbe14sample_optionsE) |
| -   [cudaq::Execu                 | -   [cudaq::ptsbe::sample_result  |
| tionContext::overlapComputeStates |     (C++                          |
|     (C++                          |     class                         |
|     member)](api/languages/cp     | )](api/languages/cpp_api.html#_CP |
| p_api.html#_CPPv4N5cudaq16Executi | Pv4N5cudaq5ptsbe13sample_resultE) |
| onContext20overlapComputeStatesE) | -   [cudaq::pts                   |
| -   [cudaq                        | be::sample_result::execution_data |
| ::ExecutionContext::overlapResult |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     member)](api/langu            | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| ages/cpp_api.html#_CPPv4N5cudaq16 | 3sample_result14execution_dataEv) |
| ExecutionContext13overlapResultE) | -   [cudaq::ptsbe::               |
| -                                 | sample_result::has_execution_data |
|   [cudaq::ExecutionContext::qpuId |     (C++                          |
|     (C++                          |                                   |
|     member)](                     |    function)](api/languages/cpp_a |
| api/languages/cpp_api.html#_CPPv4 | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| N5cudaq16ExecutionContext5qpuIdE) | ple_result18has_execution_dataEv) |
| -   [cudaq                        | -   [cudaq::pt                    |
| ::ExecutionContext::registerNames | sbe::sample_result::sample_result |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api/l              |
| ages/cpp_api.html#_CPPv4N5cudaq16 | anguages/cpp_api.html#_CPPv4N5cud |
| ExecutionContext13registerNamesE) | aq5ptsbe13sample_result13sample_r |
| -   [cu                           | esultERRN5cudaq13sample_resultE), |
| daq::ExecutionContext::reorderIdx |                                   |
|     (C++                          |  [\[1\]](api/languages/cpp_api.ht |
|     member)](api/la               | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| nguages/cpp_api.html#_CPPv4N5cuda | sult13sample_resultERRN5cudaq13sa |
| q16ExecutionContext10reorderIdxE) | mple_resultE18PTSBEExecutionData) |
| -                                 | -   [cudaq::ptsbe::               |
|  [cudaq::ExecutionContext::result | sample_result::set_execution_data |
|     (C++                          |     (C++                          |
|     member)](a                    |     function)](api/               |
| pi/languages/cpp_api.html#_CPPv4N | languages/cpp_api.html#_CPPv4N5cu |
| 5cudaq16ExecutionContext6resultE) | daq5ptsbe13sample_result18set_exe |
| -                                 | cution_dataE18PTSBEExecutionData) |
|   [cudaq::ExecutionContext::shots | -   [cud                          |
|     (C++                          | aq::ptsbe::ShotAllocationStrategy |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     struct)](using                |
| N5cudaq16ExecutionContext5shotsE) | /examples/ptsbe.html#_CPPv4N5cuda |
| -   [cudaq::                      | q5ptsbe22ShotAllocationStrategyE) |
| ExecutionContext::simulationState | -   [cudaq::ptsbe::ShotAllocatio  |
|     (C++                          | nStrategy::ShotAllocationStrategy |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)                     |
| ecutionContext15simulationStateE) | ](using/examples/ptsbe.html#_CPPv |
| -                                 | 4N5cudaq5ptsbe22ShotAllocationStr |
|    [cudaq::ExecutionContext::spin | ategy22ShotAllocationStrategyE4Ty |
|     (C++                          | pedNSt8optionalINSt8uint64_tEEE), |
|     member)]                      |     [\[1\                         |
| (api/languages/cpp_api.html#_CPPv | ]](using/examples/ptsbe.html#_CPP |
| 4N5cudaq16ExecutionContext4spinE) | v4N5cudaq5ptsbe22ShotAllocationSt |
| -   [cudaq::                      | rategy22ShotAllocationStrategyEv) |
| ExecutionContext::totalIterations | -   [cudaq::pt                    |
|     (C++                          | sbe::ShotAllocationStrategy::Type |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     enum)](using/exam             |
| ecutionContext15totalIterationsE) | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| -   [cudaq::Executio              | be22ShotAllocationStrategy4TypeE) |
| nContext::warnedNamedMeasurements | -   [cudaq::ptsbe::ShotAllocatio  |
|     (C++                          | nStrategy::Type::HIGH_WEIGHT_BIAS |
|     member)](api/languages/cpp_a  |     (C++                          |
| pi.html#_CPPv4N5cudaq16ExecutionC |     enumerat                      |
| ontext23warnedNamedMeasurementsE) | or)](using/examples/ptsbe.html#_C |
| -   [cudaq::ExecutionResult (C++  | PPv4N5cudaq5ptsbe22ShotAllocation |
|     st                            | Strategy4Type16HIGH_WEIGHT_BIASE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::ptsbe::ShotAllocati   |
| #_CPPv4N5cudaq15ExecutionResultE) | onStrategy::Type::LOW_WEIGHT_BIAS |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::appendResult |     enumera                       |
|     (C++                          | tor)](using/examples/ptsbe.html#_ |
|     functio                       | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| n)](api/languages/cpp_api.html#_C | nStrategy4Type15LOW_WEIGHT_BIASE) |
| PPv4N5cudaq15ExecutionResult12app | -   [cudaq::ptsbe::ShotAlloc      |
| endResultENSt6stringENSt6size_tE) | ationStrategy::Type::PROPORTIONAL |
| -   [cu                           |     (C++                          |
| daq::ExecutionResult::deserialize |     enum                          |
|     (C++                          | erator)](using/examples/ptsbe.htm |
|     function)                     | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| ](api/languages/cpp_api.html#_CPP | tionStrategy4Type12PROPORTIONALE) |
| v4N5cudaq15ExecutionResult11deser | -   [cudaq::ptsbe::Shot           |
| ializeERNSt6vectorINSt6size_tEEE) | AllocationStrategy::Type::UNIFORM |
| -   [cudaq:                       |     (C++                          |
| :ExecutionResult::ExecutionResult |                                   |
|     (C++                          |   enumerator)](using/examples/pts |
|     functio                       | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| n)](api/languages/cpp_api.html#_C | AllocationStrategy4Type7UNIFORME) |
| PPv4N5cudaq15ExecutionResult15Exe | -                                 |
| cutionResultE16CountsDictionary), |   [cudaq::ptsbe::TraceInstruction |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     struct)](                     |
| 15ExecutionResult15ExecutionResul | api/languages/cpp_api.html#_CPPv4 |
| tE16CountsDictionaryNSt6stringE), | N5cudaq5ptsbe16TraceInstructionE) |
|     [\[2\                         | -   [cudaq:                       |
| ]](api/languages/cpp_api.html#_CP | :ptsbe::TraceInstruction::channel |
| Pv4N5cudaq15ExecutionResult15Exec |     (C++                          |
| utionResultE16CountsDictionaryd), |     member)](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq5 |
|    [\[3\]](api/languages/cpp_api. | ptsbe16TraceInstruction7channelE) |
| html#_CPPv4N5cudaq15ExecutionResu | -   [cudaq::                      |
| lt15ExecutionResultENSt6stringE), | ptsbe::TraceInstruction::controls |
|     [\[4\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     member)](api/langu            |
| Pv4N5cudaq15ExecutionResult15Exec | ages/cpp_api.html#_CPPv4N5cudaq5p |
| utionResultERK15ExecutionResult), | tsbe16TraceInstruction8controlsE) |
|     [\[5\]](api/language          | -   [cud                          |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | aq::ptsbe::TraceInstruction::name |
| cutionResult15ExecutionResultEd), |     (C++                          |
|     [\[6\]](api/languag           |     member)](api/l                |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | anguages/cpp_api.html#_CPPv4N5cud |
| ecutionResult15ExecutionResultEv) | aq5ptsbe16TraceInstruction4nameE) |
| -   [                             | -   [cudaq                        |
| cudaq::ExecutionResult::operator= | ::ptsbe::TraceInstruction::params |
|     (C++                          |     (C++                          |
|     function)](api/languages/     |     member)](api/lan              |
| cpp_api.html#_CPPv4N5cudaq15Execu | guages/cpp_api.html#_CPPv4N5cudaq |
| tionResultaSERK15ExecutionResult) | 5ptsbe16TraceInstruction6paramsE) |
| -   [c                            | -   [cudaq:                       |
| udaq::ExecutionResult::operator== | :ptsbe::TraceInstruction::targets |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     member)](api/lang             |
| pp_api.html#_CPPv4NK5cudaq15Execu | uages/cpp_api.html#_CPPv4N5cudaq5 |
| tionResulteqERK15ExecutionResult) | ptsbe16TraceInstruction7targetsE) |
| -   [cud                          | -   [cudaq::ptsbe::T              |
| aq::ExecutionResult::registerName | raceInstruction::TraceInstruction |
|     (C++                          |     (C++                          |
|     member)](api/lan              |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq |   function)](api/languages/cpp_ap |
| 15ExecutionResult12registerNameE) | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| -   [cudaq                        | Instruction16TraceInstructionE20T |
| ::ExecutionResult::sequentialData | raceInstructionTypeNSt6stringENSt |
|     (C++                          | 6vectorINSt6size_tEEENSt6vectorIN |
|     member)](api/langu            | St6size_tEEENSt6vectorIdEENSt8opt |
| ages/cpp_api.html#_CPPv4N5cudaq15 | ionalIN5cudaq13kraus_channelEEE), |
| ExecutionResult14sequentialDataE) |     [\[1\]](api/languages/cpp_a   |
| -   [                             | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| cudaq::ExecutionResult::serialize | eInstruction16TraceInstructionEv) |
|     (C++                          | -   [cud                          |
|     function)](api/l              | aq::ptsbe::TraceInstruction::type |
| anguages/cpp_api.html#_CPPv4NK5cu |     (C++                          |
| daq15ExecutionResult9serializeEv) |     member)](api/l                |
| -   [cudaq::fermion_handler (C++  | anguages/cpp_api.html#_CPPv4N5cud |
|     c                             | aq5ptsbe16TraceInstruction4typeE) |
| lass)](api/languages/cpp_api.html | -   [c                            |
| #_CPPv4N5cudaq15fermion_handlerE) | udaq::ptsbe::TraceInstructionType |
| -   [cudaq::fermion_op (C++       |     (C++                          |
|     type)](api/languages/cpp_api  |     enum)](api/                   |
| .html#_CPPv4N5cudaq10fermion_opE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::fermion_op_term (C++  | daq5ptsbe20TraceInstructionTypeE) |
|                                   | -   [cudaq::                      |
| type)](api/languages/cpp_api.html | ptsbe::TraceInstructionType::Gate |
| #_CPPv4N5cudaq15fermion_op_termE) |     (C++                          |
| -   [cudaq::FermioniqQPU (C++     |     enumerator)](api/langu        |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq5p |
|   class)](api/languages/cpp_api.h | tsbe20TraceInstructionType4GateE) |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | -   [cudaq::ptsbe::               |
| -   [cudaq::get_state (C++        | TraceInstructionType::Measurement |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |                                   |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |    enumerator)](api/languages/cpp |
| ateEDaRR13QuantumKernelDpRR4Args) | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| -   [cudaq::gradient (C++         | aceInstructionType11MeasurementE) |
|     class)](api/languages/cpp_    | -   [cudaq::p                     |
| api.html#_CPPv4N5cudaq8gradientE) | tsbe::TraceInstructionType::Noise |
| -   [cudaq::gradient::clone (C++  |     (C++                          |
|     fun                           |     enumerator)](api/langua       |
| ction)](api/languages/cpp_api.htm | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| l#_CPPv4N5cudaq8gradient5cloneEv) | sbe20TraceInstructionType5NoiseE) |
| -   [cudaq::gradient::compute     | -   [                             |
|     (C++                          | cudaq::ptsbe::TrajectoryPredicate |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     type)](api                    |
| ient7computeERKNSt6vectorIdEERKNS | /languages/cpp_api.html#_CPPv4N5c |
| t8functionIFdNSt6vectorIdEEEEEd), | udaq5ptsbe19TrajectoryPredicateE) |
|     [\[1\]](ap                    | -   [cudaq::QPU (C++              |
| i/languages/cpp_api.html#_CPPv4N5 |     class)](api/languages         |
| cudaq8gradient7computeERKNSt6vect | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QPU::beginExecution   |
| -   [cudaq::gradient::gradient    |     (C++                          |
|     (C++                          |     function                      |
|     function)](api/lang           | )](api/languages/cpp_api.html#_CP |
| uages/cpp_api.html#_CPPv4I00EN5cu | Pv4N5cudaq3QPU14beginExecutionEv) |
| daq8gradient8gradientER7KernelT), | -   [cuda                         |
|                                   | q::QPU::configureExecutionContext |
|    [\[1\]](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4I00EN5cudaq8gradient8g |     funct                         |
| radientER7KernelTRR10ArgsMapper), | ion)](api/languages/cpp_api.html# |
|     [\[2\                         | _CPPv4NK5cudaq3QPU25configureExec |
| ]](api/languages/cpp_api.html#_CP | utionContextER16ExecutionContext) |
| Pv4I00EN5cudaq8gradient8gradientE | -   [cudaq::QPU::endExecution     |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3                          |     functi                        |
| \]](api/languages/cpp_api.html#_C | on)](api/languages/cpp_api.html#_ |
| PPv4N5cudaq8gradient8gradientERRN | CPPv4N5cudaq3QPU12endExecutionEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QPU::enqueue (C++     |
|     [\[                           |     function)](ap                 |
| 4\]](api/languages/cpp_api.html#_ | i/languages/cpp_api.html#_CPPv4N5 |
| CPPv4N5cudaq8gradient8gradientEv) | cudaq3QPU7enqueueER11QuantumTask) |
| -   [cudaq::gradient::setArgs     | -   [cud                          |
|     (C++                          | aq::QPU::finalizeExecutionContext |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     func                          |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | tion)](api/languages/cpp_api.html |
| tArgsEvR13QuantumKernelDpRR4Args) | #_CPPv4NK5cudaq3QPU24finalizeExec |
| -   [cudaq::gradient::setKernel   | utionContextER16ExecutionContext) |
|     (C++                          | -   [cudaq::QPU::getConnectivity  |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     function)                     |
| ient9setKernelEvR13QuantumKernel) | ](api/languages/cpp_api.html#_CPP |
| -   [cud                          | v4N5cudaq3QPU15getConnectivityEv) |
| aq::gradients::central_difference | -                                 |
|     (C++                          | [cudaq::QPU::getExecutionThreadId |
|     class)](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/               |
| q9gradients18central_differenceE) | languages/cpp_api.html#_CPPv4NK5c |
| -   [cudaq::gra                   | udaq3QPU20getExecutionThreadIdEv) |
| dients::central_difference::clone | -   [cudaq::QPU::getNumQubits     |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     functi                        |
| /cpp_api.html#_CPPv4N5cudaq9gradi | on)](api/languages/cpp_api.html#_ |
| ents18central_difference5cloneEv) | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| -   [cudaq::gradi                 | -   [                             |
| ents::central_difference::compute | cudaq::QPU::getRemoteCapabilities |
|     (C++                          |     (C++                          |
|     function)](                   |     function)](api/l              |
| api/languages/cpp_api.html#_CPPv4 | anguages/cpp_api.html#_CPPv4NK5cu |
| N5cudaq9gradients18central_differ | daq3QPU21getRemoteCapabilitiesEv) |
| ence7computeERKNSt6vectorIdEERKNS | -   [cudaq::QPU::isEmulated (C++  |
| t8functionIFdNSt6vectorIdEEEEEd), |     func                          |
|                                   | tion)](api/languages/cpp_api.html |
|   [\[1\]](api/languages/cpp_api.h | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| tml#_CPPv4N5cudaq9gradients18cent | -   [cudaq::QPU::isSimulator (C++ |
| ral_difference7computeERKNSt6vect |     funct                         |
| orIdEERNSt6vectorIdEERK7spin_opd) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::gradie                | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| nts::central_difference::gradient | -   [cudaq::QPU::launchKernel     |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api                |
| n)](api/languages/cpp_api.html#_C | /languages/cpp_api.html#_CPPv4N5c |
| PPv4I00EN5cudaq9gradients18centra | udaq3QPU12launchKernelERKNSt6stri |
| l_difference8gradientER7KernelT), | ngE15KernelThunkTypePvNSt8uint64_ |
|     [\[1\]](api/langua            | tENSt8uint64_tERKNSt6vectorIPvEE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::QPU::onRandomSeedSet  |
| q9gradients18central_difference8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api/lang           |
|     [\[2\]](api/languages/cpp_    | uages/cpp_api.html#_CPPv4N5cudaq3 |
| api.html#_CPPv4I00EN5cudaq9gradie | QPU15onRandomSeedSetENSt6size_tE) |
| nts18central_difference8gradientE | -   [cudaq::QPU::QPU (C++         |
| RR13QuantumKernelRR10ArgsMapper), |     functio                       |
|     [\[3\]](api/languages/cpp     | n)](api/languages/cpp_api.html#_C |
| _api.html#_CPPv4N5cudaq9gradients | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| 18central_difference8gradientERRN |                                   |
| St8functionIFvNSt6vectorIdEEEEE), |  [\[1\]](api/languages/cpp_api.ht |
|     [\[4\]](api/languages/cp      | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| p_api.html#_CPPv4N5cudaq9gradient |     [\[2\]](api/languages/cpp_    |
| s18central_difference8gradientEv) | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| -   [cud                          | -   [cudaq::QPU::setId (C++       |
| aq::gradients::forward_difference |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     class)](api/la                | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QPU::setShots (C++    |
| q9gradients18forward_differenceE) |     f                             |
| -   [cudaq::gra                   | unction)](api/languages/cpp_api.h |
| dients::forward_difference::clone | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
|     (C++                          | -   [cudaq::                      |
|     function)](api/languages      | QPU::supportsExplicitMeasurements |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18forward_difference5cloneEv) |     function)](api/languag        |
| -   [cudaq::gradi                 | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| ents::forward_difference::compute | 28supportsExplicitMeasurementsEv) |
|     (C++                          | -   [cudaq::QPU::\~QPU (C++       |
|     function)](                   |     function)](api/languages/cp   |
| api/languages/cpp_api.html#_CPPv4 | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| N5cudaq9gradients18forward_differ | -   [cudaq::QPUState (C++         |
| ence7computeERKNSt6vectorIdEERKNS |     class)](api/languages/cpp_    |
| t8functionIFdNSt6vectorIdEEEEEd), | api.html#_CPPv4N5cudaq8QPUStateE) |
|                                   | -   [cudaq::qreg (C++             |
|   [\[1\]](api/languages/cpp_api.h |     class)](api/lan               |
| tml#_CPPv4N5cudaq9gradients18forw | guages/cpp_api.html#_CPPv4I_NSt6s |
| ard_difference7computeERKNSt6vect | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::qreg::back (C++       |
| -   [cudaq::gradie                |     function)                     |
| nts::forward_difference::gradient | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4backENSt6size_tE), |
|     functio                       |     [\[1\]](api/languages/cpp_ap  |
| n)](api/languages/cpp_api.html#_C | i.html#_CPPv4N5cudaq4qreg4backEv) |
| PPv4I00EN5cudaq9gradients18forwar | -   [cudaq::qreg::begin (C++      |
| d_difference8gradientER7KernelT), |                                   |
|     [\[1\]](api/langua            |  function)](api/languages/cpp_api |
| ges/cpp_api.html#_CPPv4I00EN5cuda | .html#_CPPv4N5cudaq4qreg5beginEv) |
| q9gradients18forward_difference8g | -   [cudaq::qreg::clear (C++      |
| radientER7KernelTRR10ArgsMapper), |                                   |
|     [\[2\]](api/languages/cpp_    |  function)](api/languages/cpp_api |
| api.html#_CPPv4I00EN5cudaq9gradie | .html#_CPPv4N5cudaq4qreg5clearEv) |
| nts18forward_difference8gradientE | -   [cudaq::qreg::front (C++      |
| RR13QuantumKernelRR10ArgsMapper), |     function)]                    |
|     [\[3\]](api/languages/cpp     | (api/languages/cpp_api.html#_CPPv |
| _api.html#_CPPv4N5cudaq9gradients | 4N5cudaq4qreg5frontENSt6size_tE), |
| 18forward_difference8gradientERRN |     [\[1\]](api/languages/cpp_api |
| St8functionIFvNSt6vectorIdEEEEE), | .html#_CPPv4N5cudaq4qreg5frontEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qreg::operator\[\]    |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18forward_difference8gradientEv) |     functi                        |
| -   [                             | on)](api/languages/cpp_api.html#_ |
| cudaq::gradients::parameter_shift | CPPv4N5cudaq4qregixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qreg::qreg (C++       |
|     class)](api                   |     function)                     |
| /languages/cpp_api.html#_CPPv4N5c | ](api/languages/cpp_api.html#_CPP |
| udaq9gradients15parameter_shiftE) | v4N5cudaq4qreg4qregENSt6size_tE), |
| -   [cudaq::                      |     [\[1\]](api/languages/cpp_ap  |
| gradients::parameter_shift::clone | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     (C++                          | -   [cudaq::qreg::size (C++       |
|     function)](api/langua         |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |  function)](api/languages/cpp_api |
| adients15parameter_shift5cloneEv) | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| -   [cudaq::gr                    | -   [cudaq::qreg::slice (C++      |
| adients::parameter_shift::compute |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq4q |
|     function                      | reg5sliceENSt6size_tENSt6size_tE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qreg::value_type (C++ |
| Pv4N5cudaq9gradients15parameter_s |                                   |
| hift7computeERKNSt6vectorIdEERKNS | type)](api/languages/cpp_api.html |
| t8functionIFdNSt6vectorIdEEEEEd), | #_CPPv4N5cudaq4qreg10value_typeE) |
|     [\[1\]](api/languages/cpp_ap  | -   [cudaq::qspan (C++            |
| i.html#_CPPv4N5cudaq9gradients15p |     class)](api/lang              |
| arameter_shift7computeERKNSt6vect | uages/cpp_api.html#_CPPv4I_NSt6si |
| orIdEERNSt6vectorIdEERK7spin_opd) | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| -   [cudaq::gra                   | -   [cudaq::QuakeValue (C++       |
| dients::parameter_shift::gradient |     class)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq10QuakeValueE) |
|     func                          | -   [cudaq::Q                     |
| tion)](api/languages/cpp_api.html | uakeValue::canValidateNumElements |
| #_CPPv4I00EN5cudaq9gradients15par |     (C++                          |
| ameter_shift8gradientER7KernelT), |     function)](api/languages      |
|     [\[1\]](api/lan               | /cpp_api.html#_CPPv4N5cudaq10Quak |
| guages/cpp_api.html#_CPPv4I00EN5c | eValue22canValidateNumElementsEv) |
| udaq9gradients15parameter_shift8g | -                                 |
| radientER7KernelTRR10ArgsMapper), |  [cudaq::QuakeValue::constantSize |
|     [\[2\]](api/languages/c       |     (C++                          |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     function)](api                |
| dients15parameter_shift8gradientE | /languages/cpp_api.html#_CPPv4N5c |
| RR13QuantumKernelRR10ArgsMapper), | udaq10QuakeValue12constantSizeEv) |
|     [\[3\]](api/languages/        | -   [cudaq::QuakeValue::dump (C++ |
| cpp_api.html#_CPPv4N5cudaq9gradie |     function)](api/lan            |
| nts15parameter_shift8gradientERRN | guages/cpp_api.html#_CPPv4N5cudaq |
| St8functionIFvNSt6vectorIdEEEEE), | 10QuakeValue4dumpERNSt7ostreamE), |
|     [\[4\]](api/languages         |     [\                            |
| /cpp_api.html#_CPPv4N5cudaq9gradi | [1\]](api/languages/cpp_api.html# |
| ents15parameter_shift8gradientEv) | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| -   [cudaq::kernel_builder (C++   | -   [cudaq                        |
|     clas                          | ::QuakeValue::getRequiredElements |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4IDpEN5cudaq14kernel_builderE) |     function)](api/langua         |
| -   [c                            | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| udaq::kernel_builder::constantVal | uakeValue19getRequiredElementsEv) |
|     (C++                          | -   [cudaq::QuakeValue::getValue  |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)]                    |
| q14kernel_builder11constantValEd) | (api/languages/cpp_api.html#_CPPv |
| -   [cu                           | 4NK5cudaq10QuakeValue8getValueEv) |
| daq::kernel_builder::getArguments | -   [cudaq::QuakeValue::inverse   |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)                     |
| guages/cpp_api.html#_CPPv4N5cudaq | ](api/languages/cpp_api.html#_CPP |
| 14kernel_builder12getArgumentsEv) | v4NK5cudaq10QuakeValue7inverseEv) |
| -   [cu                           | -   [cudaq::QuakeValue::isStdVec  |
| daq::kernel_builder::getNumParams |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/lan            | ](api/languages/cpp_api.html#_CPP |
| guages/cpp_api.html#_CPPv4N5cudaq | v4N5cudaq10QuakeValue8isStdVecEv) |
| 14kernel_builder12getNumParamsEv) | -                                 |
| -   [c                            |    [cudaq::QuakeValue::operator\* |
| udaq::kernel_builder::isArgStdVec |     (C++                          |
|     (C++                          |     function)](api                |
|     function)](api/languages/cp   | /languages/cpp_api.html#_CPPv4N5c |
| p_api.html#_CPPv4N5cudaq14kernel_ | udaq10QuakeValuemlE10QuakeValue), |
| builder11isArgStdVecENSt6size_tE) |                                   |
| -   [cuda                         | [\[1\]](api/languages/cpp_api.htm |
| q::kernel_builder::kernel_builder | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|     (C++                          | -   [cudaq::QuakeValue::operator+ |
|     function)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq14kernel_bu |     function)](api                |
| ilder14kernel_builderERNSt6vector | /languages/cpp_api.html#_CPPv4N5c |
| IN7details17KernelBuilderTypeEEE) | udaq10QuakeValueplE10QuakeValue), |
| -   [cudaq::kernel_builder::name  |     [                             |
|     (C++                          | \[1\]](api/languages/cpp_api.html |
|     function)                     | #_CPPv4N5cudaq10QuakeValueplEKd), |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4N5cudaq14kernel_builder4nameEv) | [\[2\]](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|    [cudaq::kernel_builder::qalloc | -   [cudaq::QuakeValue::operator- |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api                |
| s/cpp_api.html#_CPPv4N5cudaq14ker | /languages/cpp_api.html#_CPPv4N5c |
| nel_builder6qallocE10QuakeValue), | udaq10QuakeValuemiE10QuakeValue), |
|     [\[1\]](api/language          |     [                             |
| s/cpp_api.html#_CPPv4N5cudaq14ker | \[1\]](api/languages/cpp_api.html |
| nel_builder6qallocEKNSt6size_tE), | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     [\[2                          |     [                             |
| \]](api/languages/cpp_api.html#_C | \[2\]](api/languages/cpp_api.html |
| PPv4N5cudaq14kernel_builder6qallo | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| cERNSt6vectorINSt7complexIdEEEE), |                                   |
|     [\[3\]](                      | [\[3\]](api/languages/cpp_api.htm |
| api/languages/cpp_api.html#_CPPv4 | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| N5cudaq14kernel_builder6qallocEv) | -   [cudaq::QuakeValue::operator/ |
| -   [cudaq::kernel_builder::swap  |     (C++                          |
|     (C++                          |     function)](api                |
|     function)](api/language       | /languages/cpp_api.html#_CPPv4N5c |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | udaq10QuakeValuedvE10QuakeValue), |
| 4kernel_builder4swapEvRK10QuakeVa |                                   |
| lueRK10QuakeValueRK10QuakeValue), | [\[1\]](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| [\[1\]](api/languages/cpp_api.htm | -                                 |
| l#_CPPv4I00EN5cudaq14kernel_build |  [cudaq::QuakeValue::operator\[\] |
| er4swapEvRKNSt6vectorI10QuakeValu |     (C++                          |
| eEERK10QuakeValueRK10QuakeValue), |     function)](api                |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
| [\[2\]](api/languages/cpp_api.htm | udaq10QuakeValueixEKNSt6size_tE), |
| l#_CPPv4N5cudaq14kernel_builder4s |     [\[1\]](api/                  |
| wapERK10QuakeValueRK10QuakeValue) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::KernelExecutionTask   | daq10QuakeValueixERK10QuakeValue) |
|     (C++                          | -                                 |
|     type                          |    [cudaq::QuakeValue::QuakeValue |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq19KernelExecutionTaskE) |     function)](api/languag        |
| -   [cudaq::KernelThunkResultType | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     (C++                          | akeValue10QuakeValueERN4mlir20Imp |
|     struct)]                      | licitLocOpBuilderEN4mlir5ValueE), |
| (api/languages/cpp_api.html#_CPPv |     [\[1\]                        |
| 4N5cudaq21KernelThunkResultTypeE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::KernelThunkType (C++  | v4N5cudaq10QuakeValue10QuakeValue |
|                                   | ERN4mlir20ImplicitLocOpBuilderEd) |
| type)](api/languages/cpp_api.html | -   [cudaq::QuakeValue::size (C++ |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     funct                         |
| -   [cudaq::kraus_channel (C++    | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|  class)](api/languages/cpp_api.ht | -   [cudaq::QuakeValue::slice     |
| ml#_CPPv4N5cudaq13kraus_channelE) |     (C++                          |
| -   [cudaq::kraus_channel::empty  |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq10QuakeValu |
|     function)]                    | e5sliceEKNSt6size_tEKNSt6size_tE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::quantum_platform (C++ |
| 4NK5cudaq13kraus_channel5emptyEv) |     cl                            |
| -   [cudaq::kraus_c               | ass)](api/languages/cpp_api.html# |
| hannel::generateUnitaryParameters | _CPPv4N5cudaq16quantum_platformE) |
|     (C++                          | -   [cudaq:                       |
|                                   | :quantum_platform::beginExecution |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4N5cudaq13kraus_chan |     function)](api/languag        |
| nel25generateUnitaryParametersEv) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -                                 | antum_platform14beginExecutionEv) |
|    [cudaq::kraus_channel::get_ops | -   [cudaq::quantum_pl            |
|     (C++                          | atform::configureExecutionContext |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/lang           |
| K5cudaq13kraus_channel7get_opsEv) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cud                          | 16quantum_platform25configureExec |
| aq::kraus_channel::identity_flags | utionContextER16ExecutionContext) |
|     (C++                          | -   [cuda                         |
|     member)](api/lan              | q::quantum_platform::connectivity |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 13kraus_channel14identity_flagsE) |     function)](api/langu          |
| -   [cud                          | ages/cpp_api.html#_CPPv4N5cudaq16 |
| aq::kraus_channel::is_identity_op | quantum_platform12connectivityEv) |
|     (C++                          | -   [cuda                         |
|                                   | q::quantum_platform::endExecution |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4NK5cudaq13kraus_cha |     function)](api/langu          |
| nnel14is_identity_opENSt6size_tE) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [cudaq::                      | quantum_platform12endExecutionEv) |
| kraus_channel::is_unitary_mixture | -   [cudaq::q                     |
|     (C++                          | uantum_platform::enqueueAsyncTask |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4NK5cudaq13kra |     function)](api/languages/     |
| us_channel18is_unitary_mixtureEv) | cpp_api.html#_CPPv4N5cudaq16quant |
| -   [cu                           | um_platform16enqueueAsyncTaskEKNS |
| daq::kraus_channel::kraus_channel | t6size_tER19KernelExecutionTask), |
|     (C++                          |     [\[1\]](api/languag           |
|     function)](api/lang           | es/cpp_api.html#_CPPv4N5cudaq16qu |
| uages/cpp_api.html#_CPPv4IDpEN5cu | antum_platform16enqueueAsyncTaskE |
| daq13kraus_channel13kraus_channel | KNSt6size_tERNSt8functionIFvvEEE) |
| EDpRRNSt16initializer_listI1TEE), | -   [cudaq::quantum_p             |
|                                   | latform::finalizeExecutionContext |
|  [\[1\]](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13kraus_channel13 |     function)](api/languages/c    |
| kraus_channelERK13kraus_channel), | pp_api.html#_CPPv4NK5cudaq16quant |
|     [\[2\]                        | um_platform24finalizeExecutionCon |
| ](api/languages/cpp_api.html#_CPP | textERN5cudaq16ExecutionContextE) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::qua                   |
| hannelERKNSt6vectorI8kraus_opEE), | ntum_platform::get_codegen_config |
|     [\[3\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/c    |
| v4N5cudaq13kraus_channel13kraus_c | pp_api.html#_CPPv4N5cudaq16quantu |
| hannelERRNSt6vectorI8kraus_opEE), | m_platform18get_codegen_configEv) |
|     [\[4\]](api/lan               | -   [cuda                         |
| guages/cpp_api.html#_CPPv4N5cudaq | q::quantum_platform::get_exec_ctx |
| 13kraus_channel13kraus_channelEv) |     (C++                          |
| -                                 |     function)](api/langua         |
| [cudaq::kraus_channel::noise_type | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|     (C++                          | quantum_platform12get_exec_ctxEv) |
|     member)](api                  | -   [c                            |
| /languages/cpp_api.html#_CPPv4N5c | udaq::quantum_platform::get_noise |
| udaq13kraus_channel10noise_typeE) |     (C++                          |
| -                                 |     function)](api/languages/c    |
|   [cudaq::kraus_channel::op_names | pp_api.html#_CPPv4N5cudaq16quantu |
|     (C++                          | m_platform9get_noiseENSt6size_tE) |
|     member)](                     | -   [cudaq:                       |
| api/languages/cpp_api.html#_CPPv4 | :quantum_platform::get_num_qubits |
| N5cudaq13kraus_channel8op_namesE) |     (C++                          |
| -                                 |                                   |
|  [cudaq::kraus_channel::operator= | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4NK5cudaq16quantum_plat |
|     function)](api/langua         | form14get_num_qubitsENSt6size_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq13k | -   [cudaq::quantum_              |
| raus_channelaSERK13kraus_channel) | platform::get_remote_capabilities |
| -   [c                            |     (C++                          |
| udaq::kraus_channel::operator\[\] |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/l              | v4NK5cudaq16quantum_platform23get |
| anguages/cpp_api.html#_CPPv4N5cud | _remote_capabilitiesENSt6size_tE) |
| aq13kraus_channelixEKNSt6size_tE) | -   [cudaq::qua                   |
| -                                 | ntum_platform::get_runtime_target |
| [cudaq::kraus_channel::parameters |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api                  | p_api.html#_CPPv4NK5cudaq16quantu |
| /languages/cpp_api.html#_CPPv4N5c | m_platform18get_runtime_targetEv) |
| udaq13kraus_channel10parametersE) | -   [cuda                         |
| -   [cudaq::krau                  | q::quantum_platform::getLogStream |
| s_channel::populateDefaultOpNames |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](api/languages/cp   | ages/cpp_api.html#_CPPv4N5cudaq16 |
| p_api.html#_CPPv4N5cudaq13kraus_c | quantum_platform12getLogStreamEv) |
| hannel22populateDefaultOpNamesEv) | -   [cud                          |
| -   [cu                           | aq::quantum_platform::is_emulated |
| daq::kraus_channel::probabilities |     (C++                          |
|     (C++                          |                                   |
|     member)](api/la               |    function)](api/languages/cpp_a |
| nguages/cpp_api.html#_CPPv4N5cuda | pi.html#_CPPv4NK5cudaq16quantum_p |
| q13kraus_channel13probabilitiesE) | latform11is_emulatedENSt6size_tE) |
| -                                 | -   [c                            |
|  [cudaq::kraus_channel::push_back | udaq::quantum_platform::is_remote |
|     (C++                          |     (C++                          |
|     function)](api                |     function)](api/languages/cp   |
| /languages/cpp_api.html#_CPPv4N5c | p_api.html#_CPPv4NK5cudaq16quantu |
| udaq13kraus_channel9push_backE8kr | m_platform9is_remoteENSt6size_tE) |
| aus_opNSt8optionalINSt6stringEEE) | -   [cuda                         |
| -   [cudaq::kraus_channel::size   | q::quantum_platform::is_simulator |
|     (C++                          |     (C++                          |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP |   function)](api/languages/cpp_ap |
| v4NK5cudaq13kraus_channel4sizeEv) | i.html#_CPPv4NK5cudaq16quantum_pl |
| -   [                             | atform12is_simulatorENSt6size_tE) |
| cudaq::kraus_channel::unitary_ops | -   [c                            |
|     (C++                          | udaq::quantum_platform::launchVQE |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](                   |
| daq13kraus_channel11unitary_opsE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::kraus_op (C++         | N5cudaq16quantum_platform9launchV |
|     struct)](api/languages/cpp_   | QEEKNSt6stringEPKvPN5cudaq8gradie |
| api.html#_CPPv4N5cudaq8kraus_opE) | ntERKN5cudaq7spin_opERN5cudaq9opt |
| -   [cudaq::kraus_op::adjoint     | imizerEKiKNSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     functi                        | :quantum_platform::list_platforms |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4NK5cudaq8kraus_op7adjointEv) |     function)](api/languag        |
| -   [cudaq::kraus_op::data (C++   | es/cpp_api.html#_CPPv4N5cudaq16qu |
|                                   | antum_platform14list_platformsEv) |
|  member)](api/languages/cpp_api.h | -                                 |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |    [cudaq::quantum_platform::name |
| -   [cudaq::kraus_op::kraus_op    |     (C++                          |
|     (C++                          |     function)](a                  |
|     func                          | pi/languages/cpp_api.html#_CPPv4N |
| tion)](api/languages/cpp_api.html | K5cudaq16quantum_platform4nameEv) |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | -   [                             |
| opERRNSt16initializer_listI1TEE), | cudaq::quantum_platform::num_qpus |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     function)](api/l              |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | anguages/cpp_api.html#_CPPv4NK5cu |
| pENSt6vectorIN5cudaq7complexEEE), | daq16quantum_platform8num_qpusEv) |
|     [\[2\]](api/l                 | -   [cudaq::                      |
| anguages/cpp_api.html#_CPPv4N5cud | quantum_platform::onRandomSeedSet |
| aq8kraus_op8kraus_opERK8kraus_op) |     (C++                          |
| -   [cudaq::kraus_op::nCols (C++  |                                   |
|                                   | function)](api/languages/cpp_api. |
| member)](api/languages/cpp_api.ht | html#_CPPv4N5cudaq16quantum_platf |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | orm15onRandomSeedSetENSt6size_tE) |
| -   [cudaq::kraus_op::nRows (C++  | -   [cudaq:                       |
|                                   | :quantum_platform::reset_exec_ctx |
| member)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) |     function)](api/languag        |
| -   [cudaq::kraus_op::operator=   | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14reset_exec_ctxEv) |
|     function)                     | -   [cud                          |
| ](api/languages/cpp_api.html#_CPP | aq::quantum_platform::reset_noise |
| v4N5cudaq8kraus_opaSERK8kraus_op) |     (C++                          |
| -   [cudaq::kraus_op::precision   |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq16quantum_p |
|     memb                          | latform11reset_noiseENSt6size_tE) |
| er)](api/languages/cpp_api.html#_ | -   [cudaq:                       |
| CPPv4N5cudaq8kraus_op9precisionE) | :quantum_platform::resetLogStream |
| -   [cudaq::KrausSelection (C++   |     (C++                          |
|     s                             |     function)](api/languag        |
| truct)](api/languages/cpp_api.htm | es/cpp_api.html#_CPPv4N5cudaq16qu |
| l#_CPPv4N5cudaq14KrausSelectionE) | antum_platform14resetLogStreamEv) |
| -   [cudaq:                       | -   [cuda                         |
| :KrausSelection::circuit_location | q::quantum_platform::set_exec_ctx |
|     (C++                          |     (C++                          |
|     member)](api/langua           |     funct                         |
| ges/cpp_api.html#_CPPv4N5cudaq14K | ion)](api/languages/cpp_api.html# |
| rausSelection16circuit_locationE) | _CPPv4N5cudaq16quantum_platform12 |
| -                                 | set_exec_ctxEP16ExecutionContext) |
|  [cudaq::KrausSelection::is_error | -   [c                            |
|     (C++                          | udaq::quantum_platform::set_noise |
|     member)](a                    |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function                      |
| 5cudaq14KrausSelection8is_errorE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::Kra                   | Pv4N5cudaq16quantum_platform9set_ |
| usSelection::kraus_operator_index | noiseEPK11noise_modelNSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     member)](api/languages/       | q::quantum_platform::setLogStream |
| cpp_api.html#_CPPv4N5cudaq14Kraus |     (C++                          |
| Selection20kraus_operator_indexE) |                                   |
| -   [cuda                         |  function)](api/languages/cpp_api |
| q::KrausSelection::KrausSelection | .html#_CPPv4N5cudaq16quantum_plat |
|     (C++                          | form12setLogStreamERNSt7ostreamE) |
|     function)](a                  | -   [cudaq::quantum_platfor       |
| pi/languages/cpp_api.html#_CPPv4N | m::supports_explicit_measurements |
| 5cudaq14KrausSelection14KrausSele |     (C++                          |
| ctionENSt6size_tENSt6vectorINSt6s |     function)](api/l              |
| ize_tEEENSt6stringENSt6size_tEb), | anguages/cpp_api.html#_CPPv4NK5cu |
|     [\[1\]](api/langu             | daq16quantum_platform30supports_e |
| ages/cpp_api.html#_CPPv4N5cudaq14 | xplicit_measurementsENSt6size_tE) |
| KrausSelection14KrausSelectionEv) | -   [cudaq::quantum_pla           |
| -                                 | tform::supports_task_distribution |
|   [cudaq::KrausSelection::op_name |     (C++                          |
|     (C++                          |     fu                            |
|     member)](                     | nction)](api/languages/cpp_api.ht |
| api/languages/cpp_api.html#_CPPv4 | ml#_CPPv4NK5cudaq16quantum_platfo |
| N5cudaq14KrausSelection7op_nameE) | rm26supports_task_distributionEv) |
| -   [                             | -   [cudaq::quantum               |
| cudaq::KrausSelection::operator== | _platform::with_execution_context |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)                     |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | ](api/languages/cpp_api.html#_CPP |
| usSelectioneqERK14KrausSelection) | v4I0DpEN5cudaq16quantum_platform2 |
| -                                 | 2with_execution_contextEDaR16Exec |
|    [cudaq::KrausSelection::qubits | utionContextRR8CallableDpRR4Args) |
|     (C++                          | -   [cudaq::QuantumTask (C++      |
|     member)]                      |     type)](api/languages/cpp_api. |
| (api/languages/cpp_api.html#_CPPv | html#_CPPv4N5cudaq11QuantumTaskE) |
| 4N5cudaq14KrausSelection6qubitsE) | -   [cudaq::qubit (C++            |
| -   [cudaq::KrausTrajectory (C++  |     type)](api/languages/c        |
|     st                            | pp_api.html#_CPPv4N5cudaq5qubitE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::QubitConnectivity     |
| #_CPPv4N5cudaq15KrausTrajectoryE) |     (C++                          |
| -                                 |     ty                            |
|  [cudaq::KrausTrajectory::builder | pe)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq17QubitConnectivityE) |
|     function)](ap                 | -   [cudaq::QubitEdge (C++        |
| i/languages/cpp_api.html#_CPPv4N5 |     type)](api/languages/cpp_a    |
| cudaq15KrausTrajectory7builderEv) | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| -   [cu                           | -   [cudaq::qudit (C++            |
| daq::KrausTrajectory::countErrors |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     function)](api/lang           | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::qudit::qudit (C++     |
| 15KrausTrajectory11countErrorsEv) |                                   |
| -   [                             | function)](api/languages/cpp_api. |
| cudaq::KrausTrajectory::isOrdered | html#_CPPv4N5cudaq5qudit5quditEv) |
|     (C++                          | -   [cudaq::qvector (C++          |
|     function)](api/l              |     class)                        |
| anguages/cpp_api.html#_CPPv4NK5cu | ](api/languages/cpp_api.html#_CPP |
| daq15KrausTrajectory9isOrderedEv) | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| -   [cudaq::                      | -   [cudaq::qvector::back (C++    |
| KrausTrajectory::kraus_selections |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     member)](api/languag          | 5cudaq7qvector4backENSt6size_tE), |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |                                   |
| ausTrajectory16kraus_selectionsE) |   [\[1\]](api/languages/cpp_api.h |
| -   [cudaq:                       | tml#_CPPv4N5cudaq7qvector4backEv) |
| :KrausTrajectory::KrausTrajectory | -   [cudaq::qvector::begin (C++   |
|     (C++                          |     fu                            |
|     function                      | nction)](api/languages/cpp_api.ht |
| )](api/languages/cpp_api.html#_CP | ml#_CPPv4N5cudaq7qvector5beginEv) |
| Pv4N5cudaq15KrausTrajectory15Krau | -   [cudaq::qvector::clear (C++   |
| sTrajectoryENSt6size_tENSt6vector |     fu                            |
| I14KrausSelectionEEdNSt6size_tE), | nction)](api/languages/cpp_api.ht |
|     [\[1\]](api/languag           | ml#_CPPv4N5cudaq7qvector5clearEv) |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | -   [cudaq::qvector::end (C++     |
| ausTrajectory15KrausTrajectoryEv) |                                   |
| -   [cudaq::Kr                    | function)](api/languages/cpp_api. |
| ausTrajectory::measurement_counts | html#_CPPv4N5cudaq7qvector3endEv) |
|     (C++                          | -   [cudaq::qvector::front (C++   |
|     member)](api/languages        |     function)](ap                 |
| /cpp_api.html#_CPPv4N5cudaq15Krau | i/languages/cpp_api.html#_CPPv4N5 |
| sTrajectory18measurement_countsE) | cudaq7qvector5frontENSt6size_tE), |
| -   [cud                          |                                   |
| aq::KrausTrajectory::multiplicity |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     member)](api/lan              | -   [cudaq::qvector::operator=    |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15KrausTrajectory12multiplicityE) |     functio                       |
| -   [                             | n)](api/languages/cpp_api.html#_C |
| cudaq::KrausTrajectory::num_shots | PPv4N5cudaq7qvectoraSERK7qvector) |
|     (C++                          | -   [cudaq::qvector::operator\[\] |
|     member)](api                  |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)                     |
| udaq15KrausTrajectory9num_shotsE) | ](api/languages/cpp_api.html#_CPP |
| -   [c                            | v4N5cudaq7qvectorixEKNSt6size_tE) |
| udaq::KrausTrajectory::operator== | -   [cudaq::qvector::qvector (C++ |
|     (C++                          |     function)](api/               |
|     function)](api/languages/c    | languages/cpp_api.html#_CPPv4N5cu |
| pp_api.html#_CPPv4NK5cudaq15Kraus | daq7qvector7qvectorENSt6size_tE), |
| TrajectoryeqERK15KrausTrajectory) |     [\[1\]](a                     |
| -   [cu                           | pi/languages/cpp_api.html#_CPPv4N |
| daq::KrausTrajectory::probability | 5cudaq7qvector7qvectorERK5state), |
|     (C++                          |     [\[2\]](api                   |
|     member)](api/la               | /languages/cpp_api.html#_CPPv4N5c |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq7qvector7qvectorERK7qvector), |
| q15KrausTrajectory11probabilityE) |     [\[3\]](ap                    |
| -   [cuda                         | i/languages/cpp_api.html#_CPPv4N5 |
| q::KrausTrajectory::trajectory_id | cudaq7qvector7qvectorERR7qvector) |
|     (C++                          | -   [cudaq::qvector::size (C++    |
|     member)](api/lang             |     fu                            |
| uages/cpp_api.html#_CPPv4N5cudaq1 | nction)](api/languages/cpp_api.ht |
| 5KrausTrajectory13trajectory_idE) | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| -                                 | -   [cudaq::qvector::slice (C++   |
|   [cudaq::KrausTrajectory::weight |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|     member)](                     | tor5sliceENSt6size_tENSt6size_tE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qvector::value_type   |
| N5cudaq15KrausTrajectory6weightE) |     (C++                          |
| -                                 |     typ                           |
|    [cudaq::KrausTrajectoryBuilder | e)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq7qvector10value_typeE) |
|     class)](                      | -   [cudaq::qview (C++            |
| api/languages/cpp_api.html#_CPPv4 |     clas                          |
| N5cudaq22KrausTrajectoryBuilderE) | s)](api/languages/cpp_api.html#_C |
| -   [cud                          | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| aq::KrausTrajectoryBuilder::build | -   [cudaq::qview::back (C++      |
|     (C++                          |     function)                     |
|     function)](api/lang           | ](api/languages/cpp_api.html#_CPP |
| uages/cpp_api.html#_CPPv4NK5cudaq | v4N5cudaq5qview4backENSt6size_tE) |
| 22KrausTrajectoryBuilder5buildEv) | -   [cudaq::qview::begin (C++     |
| -   [cud                          |                                   |
| aq::KrausTrajectoryBuilder::setId | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qview5beginEv) |
|     function)](api/languages/cpp  | -   [cudaq::qview::end (C++       |
| _api.html#_CPPv4N5cudaq22KrausTra |                                   |
| jectoryBuilder5setIdENSt6size_tE) |   function)](api/languages/cpp_ap |
| -   [cudaq::Kraus                 | i.html#_CPPv4N5cudaq5qview3endEv) |
| TrajectoryBuilder::setProbability | -   [cudaq::qview::front (C++     |
|     (C++                          |     function)](                   |
|     function)](api/languages/cpp  | api/languages/cpp_api.html#_CPPv4 |
| _api.html#_CPPv4N5cudaq22KrausTra | N5cudaq5qview5frontENSt6size_tE), |
| jectoryBuilder14setProbabilityEd) |                                   |
| -   [cudaq::Krau                  |    [\[1\]](api/languages/cpp_api. |
| sTrajectoryBuilder::setSelections | html#_CPPv4N5cudaq5qview5frontEv) |
|     (C++                          | -   [cudaq::qview::operator\[\]   |
|     function)](api/languag        |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq22Kr |     functio                       |
| ausTrajectoryBuilder13setSelectio | n)](api/languages/cpp_api.html#_C |
| nsENSt6vectorI14KrausSelectionEE) | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| -   [cudaq::matrix_callback (C++  | -   [cudaq::qview::qview (C++     |
|     c                             |     functio                       |
| lass)](api/languages/cpp_api.html | n)](api/languages/cpp_api.html#_C |
| #_CPPv4N5cudaq15matrix_callbackE) | PPv4I0EN5cudaq5qview5qviewERR1R), |
| -   [cudaq::matrix_handler (C++   |     [\[1                          |
|                                   | \]](api/languages/cpp_api.html#_C |
| class)](api/languages/cpp_api.htm | PPv4N5cudaq5qview5qviewERK5qview) |
| l#_CPPv4N5cudaq14matrix_handlerE) | -   [cudaq::qview::size (C++      |
| -   [cudaq::mat                   |                                   |
| rix_handler::commutation_behavior | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4NK5cudaq5qview4sizeEv) |
|     struct)](api/languages/       | -   [cudaq::qview::slice (C++     |
| cpp_api.html#_CPPv4N5cudaq14matri |     function)](api/langua         |
| x_handler20commutation_behaviorE) | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| -                                 | iew5sliceENSt6size_tENSt6size_tE) |
|    [cudaq::matrix_handler::define | -   [cudaq::qview::value_type     |
|     (C++                          |     (C++                          |
|     function)](a                  |     t                             |
| pi/languages/cpp_api.html#_CPPv4N | ype)](api/languages/cpp_api.html# |
| 5cudaq14matrix_handler6defineENSt | _CPPv4N5cudaq5qview10value_typeE) |
| 6stringENSt6vectorINSt7int64_tEEE | -   [cudaq::range (C++            |
| RR15matrix_callbackRKNSt13unorder |     fun                           |
| ed_mapINSt6stringENSt6stringEEE), | ction)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| [\[1\]](api/languages/cpp_api.htm | orI11ElementTypeEE11ElementType), |
| l#_CPPv4N5cudaq14matrix_handler6d |     [\[1\]](api/languages/cpp_    |
| efineENSt6stringENSt6vectorINSt7i | api.html#_CPPv4I0EN5cudaq5rangeEN |
| nt64_tEEERR15matrix_callbackRR20d | St6vectorI11ElementTypeEE11Elemen |
| iag_matrix_callbackRKNSt13unorder | tType11ElementType11ElementType), |
| ed_mapINSt6stringENSt6stringEEE), |     [                             |
|     [\[2\]](                      | \[2\]](api/languages/cpp_api.html |
| api/languages/cpp_api.html#_CPPv4 | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| N5cudaq14matrix_handler6defineENS | -   [cudaq::real (C++             |
| t6stringENSt6vectorINSt7int64_tEE |     type)](api/languages/         |
| ERR15matrix_callbackRRNSt13unorde | cpp_api.html#_CPPv4N5cudaq4realE) |
| red_mapINSt6stringENSt6stringEEE) | -   [cudaq::registry (C++         |
| -                                 |     type)](api/languages/cpp_     |
|   [cudaq::matrix_handler::degrees | api.html#_CPPv4N5cudaq8registryE) |
|     (C++                          | -                                 |
|     function)](ap                 |  [cudaq::registry::RegisteredType |
| i/languages/cpp_api.html#_CPPv4NK |     (C++                          |
| 5cudaq14matrix_handler7degreesEv) |     class)](api/                  |
| -                                 | languages/cpp_api.html#_CPPv4I0EN |
|  [cudaq::matrix_handler::displace | 5cudaq8registry14RegisteredTypeE) |
|     (C++                          | -   [cudaq::RemoteCapabilities    |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     struc                         |
| rix_handler8displaceENSt6size_tE) | t)](api/languages/cpp_api.html#_C |
| -   [cudaq::matrix                | PPv4N5cudaq18RemoteCapabilitiesE) |
| _handler::get_expected_dimensions | -   [cudaq::Remo                  |
|     (C++                          | teCapabilities::isRemoteSimulator |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     member)](api/languages/c      |
| pi.html#_CPPv4NK5cudaq14matrix_ha | pp_api.html#_CPPv4N5cudaq18Remote |
| ndler23get_expected_dimensionsEv) | Capabilities17isRemoteSimulatorE) |
| -   [cudaq::matrix_ha             | -   [cudaq::Remot                 |
| ndler::get_parameter_descriptions | eCapabilities::RemoteCapabilities |
|     (C++                          |     (C++                          |
|                                   |     function)](api/languages/cpp  |
| function)](api/languages/cpp_api. | _api.html#_CPPv4N5cudaq18RemoteCa |
| html#_CPPv4NK5cudaq14matrix_handl | pabilities18RemoteCapabilitiesEb) |
| er26get_parameter_descriptionsEv) | -   [cudaq:                       |
| -   [c                            | :RemoteCapabilities::stateOverlap |
| udaq::matrix_handler::instantiate |     (C++                          |
|     (C++                          |     member)](api/langua           |
|     function)](a                  | ges/cpp_api.html#_CPPv4N5cudaq18R |
| pi/languages/cpp_api.html#_CPPv4N | emoteCapabilities12stateOverlapE) |
| 5cudaq14matrix_handler11instantia | -                                 |
| teENSt6stringERKNSt6vectorINSt6si |   [cudaq::RemoteCapabilities::vqe |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[1\]](                      |     member)](                     |
| api/languages/cpp_api.html#_CPPv4 | api/languages/cpp_api.html#_CPPv4 |
| N5cudaq14matrix_handler11instanti | N5cudaq18RemoteCapabilities3vqeE) |
| ateENSt6stringERRNSt6vectorINSt6s | -   [cudaq::RemoteSimulationState |
| ize_tEEERK20commutation_behavior) |     (C++                          |
| -   [cuda                         |     class)]                       |
| q::matrix_handler::matrix_handler | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4N5cudaq21RemoteSimulationStateE) |
|     function)](api/languag        | -   [cudaq::Resources (C++        |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     class)](api/languages/cpp_a   |
| ble_if_tINSt12is_base_of_vI16oper | pi.html#_CPPv4N5cudaq9ResourcesE) |
| ator_handler1TEEbEEEN5cudaq14matr | -   [cudaq::run (C++              |
| ix_handler14matrix_handlerERK1T), |     function)]                    |
|     [\[1\]](ap                    | (api/languages/cpp_api.html#_CPPv |
| i/languages/cpp_api.html#_CPPv4I0 | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| _NSt11enable_if_tINSt12is_base_of | 5invoke_result_tINSt7decay_tI13Qu |
| _vI16operator_handler1TEEbEEEN5cu | antumKernelEEDpNSt7decay_tI4ARGSE |
| daq14matrix_handler14matrix_handl | EEEEENSt6size_tERN5cudaq11noise_m |
| erERK1TRK20commutation_behavior), | odelERR13QuantumKernelDpRR4ARGS), |
|     [\[2\]](api/languages/cpp_ap  |     [\[1\]](api/langu             |
| i.html#_CPPv4N5cudaq14matrix_hand | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| ler14matrix_handlerENSt6size_tE), | daq3runENSt6vectorINSt15invoke_re |
|     [\[3\]](api/                  | sult_tINSt7decay_tI13QuantumKerne |
| languages/cpp_api.html#_CPPv4N5cu | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| daq14matrix_handler14matrix_handl | ize_tERR13QuantumKernelDpRR4ARGS) |
| erENSt6stringERKNSt6vectorINSt6si | -   [cudaq::run_async (C++        |
| ze_tEEERK20commutation_behavior), |     functio                       |
|     [\[4\]](api/                  | n)](api/languages/cpp_api.html#_C |
| languages/cpp_api.html#_CPPv4N5cu | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| daq14matrix_handler14matrix_handl | tureINSt6vectorINSt15invoke_resul |
| erENSt6stringERRNSt6vectorINSt6si | t_tINSt7decay_tI13QuantumKernelEE |
| ze_tEEERK20commutation_behavior), | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|     [\                            | ze_tENSt6size_tERN5cudaq11noise_m |
| [5\]](api/languages/cpp_api.html# | odelERR13QuantumKernelDpRR4ARGS), |
| _CPPv4N5cudaq14matrix_handler14ma |     [\[1\]](api/la                |
| trix_handlerERK14matrix_handler), | nguages/cpp_api.html#_CPPv4I0DpEN |
|     [                             | 5cudaq9run_asyncENSt6futureINSt6v |
| \[6\]](api/languages/cpp_api.html | ectorINSt15invoke_result_tINSt7de |
| #_CPPv4N5cudaq14matrix_handler14m | cay_tI13QuantumKernelEEDpNSt7deca |
| atrix_handlerERR14matrix_handler) | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| -                                 | ize_tERR13QuantumKernelDpRR4ARGS) |
|  [cudaq::matrix_handler::momentum | -   [cudaq::RuntimeTarget (C++    |
|     (C++                          |                                   |
|     function)](api/language       | struct)](api/languages/cpp_api.ht |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| rix_handler8momentumENSt6size_tE) | -   [cudaq::sample (C++           |
| -                                 |     function)](api/languages/c    |
|    [cudaq::matrix_handler::number | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     (C++                          | mpleE13sample_resultRK14sample_op |
|     function)](api/langua         | tionsRR13QuantumKernelDpRR4Args), |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     [\[1\                         |
| atrix_handler6numberENSt6size_tE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4I0DpEN5cudaq6sampleE13sample_r |
| [cudaq::matrix_handler::operator= | esultRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\                            |
|     fun                           | [2\]](api/languages/cpp_api.html# |
| ction)](api/languages/cpp_api.htm | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| l#_CPPv4I0_NSt11enable_if_tIXaant | ize_tERR13QuantumKernelDpRR4Args) |
| NSt7is_sameI1T14matrix_handlerE5v | -   [cudaq::sample_options (C++   |
| alueENSt12is_base_of_vI16operator |     s                             |
| _handler1TEEEbEEEN5cudaq14matrix_ | truct)](api/languages/cpp_api.htm |
| handleraSER14matrix_handlerRK1T), | l#_CPPv4N5cudaq14sample_optionsE) |
|     [\[1\]](api/languages         | -   [cudaq::sample_result (C++    |
| /cpp_api.html#_CPPv4N5cudaq14matr |                                   |
| ix_handleraSERK14matrix_handler), |  class)](api/languages/cpp_api.ht |
|     [\[2\]](api/language          | ml#_CPPv4N5cudaq13sample_resultE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::sample_result::append |
| rix_handleraSERR14matrix_handler) |     (C++                          |
| -   [                             |     function)](api/languages/cpp_ |
| cudaq::matrix_handler::operator== | api.html#_CPPv4N5cudaq13sample_re |
|     (C++                          | sult6appendERK15ExecutionResultb) |
|     function)](api/languages      | -   [cudaq::sample_result::begin  |
| /cpp_api.html#_CPPv4NK5cudaq14mat |     (C++                          |
| rix_handlereqERK14matrix_handler) |     function)]                    |
| -                                 | (api/languages/cpp_api.html#_CPPv |
|    [cudaq::matrix_handler::parity | 4N5cudaq13sample_result5beginEv), |
|     (C++                          |     [\[1\]]                       |
|     function)](api/langua         | (api/languages/cpp_api.html#_CPPv |
| ges/cpp_api.html#_CPPv4N5cudaq14m | 4NK5cudaq13sample_result5beginEv) |
| atrix_handler6parityENSt6size_tE) | -   [cudaq::sample_result::cbegin |
| -                                 |     (C++                          |
|  [cudaq::matrix_handler::position |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/language       | NK5cudaq13sample_result6cbeginEv) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::sample_result::cend   |
| rix_handler8positionENSt6size_tE) |     (C++                          |
| -   [cudaq::                      |     function)                     |
| matrix_handler::remove_definition | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq13sample_result4cendEv) |
|     fu                            | -   [cudaq::sample_result::clear  |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14matrix_handler1 |     function)                     |
| 7remove_definitionERKNSt6stringE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq13sample_result5clearEv) |
|   [cudaq::matrix_handler::squeeze | -   [cudaq::sample_result::count  |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)](                   |
| es/cpp_api.html#_CPPv4N5cudaq14ma | api/languages/cpp_api.html#_CPPv4 |
| trix_handler7squeezeENSt6size_tE) | NK5cudaq13sample_result5countENSt |
| -   [cudaq::m                     | 11string_viewEKNSt11string_viewE) |
| atrix_handler::to_diagonal_matrix | -   [                             |
|     (C++                          | cudaq::sample_result::deserialize |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     functio                       |
| 14matrix_handler18to_diagonal_mat | n)](api/languages/cpp_api.html#_C |
| rixERNSt13unordered_mapINSt6size_ | PPv4N5cudaq13sample_result11deser |
| tENSt7int64_tEEERKNSt13unordered_ | ializeERNSt6vectorINSt6size_tEEE) |
| mapINSt6stringENSt7complexIdEEEE) | -   [cudaq::sample_result::dump   |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::to_matrix |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     function)                     | ample_result4dumpERNSt7ostreamE), |
| ](api/languages/cpp_api.html#_CPP |     [\[1\]                        |
| v4NK5cudaq14matrix_handler9to_mat | ](api/languages/cpp_api.html#_CPP |
| rixERNSt13unordered_mapINSt6size_ | v4NK5cudaq13sample_result4dumpEv) |
| tENSt7int64_tEEERKNSt13unordered_ | -   [cudaq::sample_result::end    |
| mapINSt6stringENSt7complexIdEEEE) |     (C++                          |
| -                                 |     function                      |
| [cudaq::matrix_handler::to_string | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq13sample_result3endEv), |
|     function)](api/               |     [\[1\                         |
| languages/cpp_api.html#_CPPv4NK5c | ]](api/languages/cpp_api.html#_CP |
| udaq14matrix_handler9to_stringEb) | Pv4NK5cudaq13sample_result3endEv) |
| -                                 | -   [                             |
| [cudaq::matrix_handler::unique_id | cudaq::sample_result::expectation |
|     (C++                          |     (C++                          |
|     function)](api/               |     f                             |
| languages/cpp_api.html#_CPPv4NK5c | unction)](api/languages/cpp_api.h |
| udaq14matrix_handler9unique_idEv) | tml#_CPPv4NK5cudaq13sample_result |
| -   [cudaq:                       | 11expectationEKNSt11string_viewE) |
| :matrix_handler::\~matrix_handler | -   [c                            |
|     (C++                          | udaq::sample_result::get_marginal |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)](api/languages/cpp_ |
| CPPv4N5cudaq14matrix_handlerD0Ev) | api.html#_CPPv4NK5cudaq13sample_r |
| -   [cudaq::matrix_op (C++        | esult12get_marginalERKNSt6vectorI |
|     type)](api/languages/cpp_a    | NSt6size_tEEEKNSt11string_viewE), |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     [\[1\]](api/languages/cpp_    |
| -   [cudaq::matrix_op_term (C++   | api.html#_CPPv4NK5cudaq13sample_r |
|                                   | esult12get_marginalERRKNSt6vector |
|  type)](api/languages/cpp_api.htm | INSt6size_tEEEKNSt11string_viewE) |
| l#_CPPv4N5cudaq14matrix_op_termE) | -   [cuda                         |
| -                                 | q::sample_result::get_total_shots |
|    [cudaq::mdiag_operator_handler |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     class)](                      | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| api/languages/cpp_api.html#_CPPv4 | sample_result15get_total_shotsEv) |
| N5cudaq22mdiag_operator_handlerE) | -   [cuda                         |
| -   [cudaq::mpi (C++              | q::sample_result::has_even_parity |
|     type)](api/languages          |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) |     fun                           |
| -   [cudaq::mpi::all_gather (C++  | ction)](api/languages/cpp_api.htm |
|     fu                            | l#_CPPv4N5cudaq13sample_result15h |
| nction)](api/languages/cpp_api.ht | as_even_parityENSt11string_viewE) |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | -   [cuda                         |
| RNSt6vectorIdEERKNSt6vectorIdEE), | q::sample_result::has_expectation |
|                                   |     (C++                          |
|   [\[1\]](api/languages/cpp_api.h |     funct                         |
| tml#_CPPv4N5cudaq3mpi10all_gather | ion)](api/languages/cpp_api.html# |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | _CPPv4NK5cudaq13sample_result15ha |
| -   [cudaq::mpi::all_reduce (C++  | s_expectationEKNSt11string_viewE) |
|                                   | -   [cu                           |
|  function)](api/languages/cpp_api | daq::sample_result::most_probable |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     (C++                          |
| reduceE1TRK1TRK14BinaryFunction), |     fun                           |
|     [\[1\]](api/langu             | ction)](api/languages/cpp_api.htm |
| ages/cpp_api.html#_CPPv4I00EN5cud | l#_CPPv4NK5cudaq13sample_result13 |
| aq3mpi10all_reduceE1TRK1TRK4Func) | most_probableEKNSt11string_viewE) |
| -   [cudaq::mpi::broadcast (C++   | -                                 |
|     function)](api/               | [cudaq::sample_result::operator+= |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq3mpi9broadcastERNSt6stringEi), |     function)](api/langua         |
|     [\[1\]](api/la                | ges/cpp_api.html#_CPPv4N5cudaq13s |
| nguages/cpp_api.html#_CPPv4N5cuda | ample_resultpLERK13sample_result) |
| q3mpi9broadcastERNSt6vectorIdEEi) | -                                 |
| -   [cudaq::mpi::finalize (C++    |  [cudaq::sample_result::operator= |
|     f                             |     (C++                          |
| unction)](api/languages/cpp_api.h |     function)](api/langua         |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -   [cudaq::mpi::initialize (C++  | ample_resultaSERR13sample_result) |
|     function                      | -                                 |
| )](api/languages/cpp_api.html#_CP | [cudaq::sample_result::operator== |
| Pv4N5cudaq3mpi10initializeEiPPc), |     (C++                          |
|     [                             |     function)](api/languag        |
| \[1\]](api/languages/cpp_api.html | es/cpp_api.html#_CPPv4NK5cudaq13s |
| #_CPPv4N5cudaq3mpi10initializeEv) | ample_resulteqERK13sample_result) |
| -   [cudaq::mpi::is_initialized   | -   [                             |
|     (C++                          | cudaq::sample_result::probability |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/lan            |
| Pv4N5cudaq3mpi14is_initializedEv) | guages/cpp_api.html#_CPPv4NK5cuda |
| -   [cudaq::mpi::num_ranks (C++   | q13sample_result11probabilityENSt |
|     fu                            | 11string_viewEKNSt11string_viewE) |
| nction)](api/languages/cpp_api.ht | -   [cud                          |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | aq::sample_result::register_names |
| -   [cudaq::mpi::rank (C++        |     (C++                          |
|                                   |     function)](api/langu          |
|    function)](api/languages/cpp_a | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | 3sample_result14register_namesEv) |
| -   [cudaq::noise_model (C++      | -                                 |
|                                   |    [cudaq::sample_result::reorder |
|    class)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq11noise_modelE) |     function)](api/langua         |
| -   [cudaq::n                     | ges/cpp_api.html#_CPPv4N5cudaq13s |
| oise_model::add_all_qubit_channel | ample_result7reorderERKNSt6vector |
|     (C++                          | INSt6size_tEEEKNSt11string_viewE) |
|     function)](api                | -   [cu                           |
| /languages/cpp_api.html#_CPPv4IDp | daq::sample_result::sample_result |
| EN5cudaq11noise_model21add_all_qu |     (C++                          |
| bit_channelEvRK13kraus_channeli), |     func                          |
|     [\[1\]](api/langua            | tion)](api/languages/cpp_api.html |
| ges/cpp_api.html#_CPPv4N5cudaq11n | #_CPPv4N5cudaq13sample_result13sa |
| oise_model21add_all_qubit_channel | mple_resultERK15ExecutionResult), |
| ERKNSt6stringERK13kraus_channeli) |     [\[1\]](api/la                |
| -                                 | nguages/cpp_api.html#_CPPv4N5cuda |
|  [cudaq::noise_model::add_channel | q13sample_result13sample_resultER |
|     (C++                          | KNSt6vectorI15ExecutionResultEE), |
|     funct                         |                                   |
| ion)](api/languages/cpp_api.html# |  [\[2\]](api/languages/cpp_api.ht |
| _CPPv4IDpEN5cudaq11noise_model11a | ml#_CPPv4N5cudaq13sample_result13 |
| dd_channelEvRK15PredicateFuncTy), | sample_resultERR13sample_result), |
|     [\[1\]](api/languages/cpp_    |     [                             |
| api.html#_CPPv4IDpEN5cudaq11noise | \[3\]](api/languages/cpp_api.html |
| _model11add_channelEvRKNSt6vector | #_CPPv4N5cudaq13sample_result13sa |
| INSt6size_tEEERK13kraus_channel), | mple_resultERR15ExecutionResult), |
|     [\[2\]](ap                    |     [\[4\]](api/lan               |
| i/languages/cpp_api.html#_CPPv4N5 | guages/cpp_api.html#_CPPv4N5cudaq |
| cudaq11noise_model11add_channelER | 13sample_result13sample_resultEdR |
| KNSt6stringERK15PredicateFuncTy), | KNSt6vectorI15ExecutionResultEE), |
|                                   |     [\[5\]](api/lan               |
| [\[3\]](api/languages/cpp_api.htm | guages/cpp_api.html#_CPPv4N5cudaq |
| l#_CPPv4N5cudaq11noise_model11add | 13sample_result13sample_resultEv) |
| _channelERKNSt6stringERKNSt6vecto | -                                 |
| rINSt6size_tEEERK13kraus_channel) |  [cudaq::sample_result::serialize |
| -   [cudaq::noise_model::empty    |     (C++                          |
|     (C++                          |     function)](api                |
|     function                      | /languages/cpp_api.html#_CPPv4NK5 |
| )](api/languages/cpp_api.html#_CP | cudaq13sample_result9serializeEv) |
| Pv4NK5cudaq11noise_model5emptyEv) | -   [cudaq::sample_result::size   |
| -                                 |     (C++                          |
| [cudaq::noise_model::get_channels |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq13sampl |
|     function)](api/l              | e_result4sizeEKNSt11string_viewE) |
| anguages/cpp_api.html#_CPPv4I0ENK | -   [cudaq::sample_result::to_map |
| 5cudaq11noise_model12get_channels |     (C++                          |
| ENSt6vectorI13kraus_channelEERKNS |     function)](api/languages/cpp  |
| t6vectorINSt6size_tEEERKNSt6vecto | _api.html#_CPPv4NK5cudaq13sample_ |
| rINSt6size_tEEERKNSt6vectorIdEE), | result6to_mapEKNSt11string_viewE) |
|     [\[1\]](api/languages/cpp_a   | -   [cuda                         |
| pi.html#_CPPv4NK5cudaq11noise_mod | q::sample_result::\~sample_result |
| el12get_channelsERKNSt6stringERKN |     (C++                          |
| St6vectorINSt6size_tEEERKNSt6vect |     funct                         |
| orINSt6size_tEEERKNSt6vectorIdEE) | ion)](api/languages/cpp_api.html# |
| -                                 | _CPPv4N5cudaq13sample_resultD0Ev) |
|  [cudaq::noise_model::noise_model | -   [cudaq::scalar_callback (C++  |
|     (C++                          |     c                             |
|     function)](api                | lass)](api/languages/cpp_api.html |
| /languages/cpp_api.html#_CPPv4N5c | #_CPPv4N5cudaq15scalar_callbackE) |
| udaq11noise_model11noise_modelEv) | -   [c                            |
| -   [cu                           | udaq::scalar_callback::operator() |
| daq::noise_model::PredicateFuncTy |     (C++                          |
|     (C++                          |     function)](api/language       |
|     type)](api/la                 | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| nguages/cpp_api.html#_CPPv4N5cuda | alar_callbackclERKNSt13unordered_ |
| q11noise_model15PredicateFuncTyE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cud                          | -   [                             |
| aq::noise_model::register_channel | cudaq::scalar_callback::operator= |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/languages/c    |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | pp_api.html#_CPPv4N5cudaq15scalar |
| noise_model16register_channelEvv) | _callbackaSERK15scalar_callback), |
| -   [cudaq::                      |     [\[1\]](api/languages/        |
| noise_model::requires_constructor | cpp_api.html#_CPPv4N5cudaq15scala |
|     (C++                          | r_callbackaSERR15scalar_callback) |
|     type)](api/languages/cp       | -   [cudaq:                       |
| p_api.html#_CPPv4I0DpEN5cudaq11no | :scalar_callback::scalar_callback |
| ise_model20requires_constructorE) |     (C++                          |
| -   [cudaq::noise_model_type (C++ |     function)](api/languag        |
|     e                             | es/cpp_api.html#_CPPv4I0_NSt11ena |
| num)](api/languages/cpp_api.html# | ble_if_tINSt16is_invocable_r_vINS |
| _CPPv4N5cudaq16noise_model_typeE) | t7complexIdEE8CallableRKNSt13unor |
| -   [cudaq::no                    | dered_mapINSt6stringENSt7complexI |
| ise_model_type::amplitude_damping | dEEEEEEbEEEN5cudaq15scalar_callba |
|     (C++                          | ck15scalar_callbackERR8Callable), |
|     enumerator)](api/languages    |     [\[1\                         |
| /cpp_api.html#_CPPv4N5cudaq16nois | ]](api/languages/cpp_api.html#_CP |
| e_model_type17amplitude_dampingE) | Pv4N5cudaq15scalar_callback15scal |
| -   [cudaq::noise_mode            | ar_callbackERK15scalar_callback), |
| l_type::amplitude_damping_channel |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     e                             | PPv4N5cudaq15scalar_callback15sca |
| numerator)](api/languages/cpp_api | lar_callbackERR15scalar_callback) |
| .html#_CPPv4N5cudaq16noise_model_ | -   [cudaq::scalar_operator (C++  |
| type25amplitude_damping_channelE) |     c                             |
| -   [cudaq::n                     | lass)](api/languages/cpp_api.html |
| oise_model_type::bit_flip_channel | #_CPPv4N5cudaq15scalar_operatorE) |
|     (C++                          | -                                 |
|     enumerator)](api/language     | [cudaq::scalar_operator::evaluate |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     (C++                          |
| se_model_type16bit_flip_channelE) |                                   |
| -   [cudaq::                      |    function)](api/languages/cpp_a |
| noise_model_type::depolarization1 | pi.html#_CPPv4NK5cudaq15scalar_op |
|     (C++                          | erator8evaluateERKNSt13unordered_ |
|     enumerator)](api/languag      | mapINSt6stringENSt7complexIdEEEE) |
| es/cpp_api.html#_CPPv4N5cudaq16no | -   [cudaq::scalar_ope            |
| ise_model_type15depolarization1E) | rator::get_parameter_descriptions |
| -   [cudaq::                      |     (C++                          |
| noise_model_type::depolarization2 |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     enumerator)](api/languag      | tml#_CPPv4NK5cudaq15scalar_operat |
| es/cpp_api.html#_CPPv4N5cudaq16no | or26get_parameter_descriptionsEv) |
| ise_model_type15depolarization2E) | -   [cu                           |
| -   [cudaq::noise_m               | daq::scalar_operator::is_constant |
| odel_type::depolarization_channel |     (C++                          |
|     (C++                          |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|   enumerator)](api/languages/cpp_ | 15scalar_operator11is_constantEv) |
| api.html#_CPPv4N5cudaq16noise_mod | -   [c                            |
| el_type22depolarization_channelE) | udaq::scalar_operator::operator\* |
| -                                 |     (C++                          |
|  [cudaq::noise_model_type::pauli1 |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     enumerator)](a                | Pv4N5cudaq15scalar_operatormlENSt |
| pi/languages/cpp_api.html#_CPPv4N | 7complexIdEERK15scalar_operator), |
| 5cudaq16noise_model_type6pauli1E) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|  [cudaq::noise_model_type::pauli2 | Pv4N5cudaq15scalar_operatormlENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     enumerator)](a                |     [\[2\]](api/languages/cp      |
| pi/languages/cpp_api.html#_CPPv4N | p_api.html#_CPPv4N5cudaq15scalar_ |
| 5cudaq16noise_model_type6pauli2E) | operatormlEdRK15scalar_operator), |
| -   [cudaq                        |     [\[3\]](api/languages/cp      |
| ::noise_model_type::phase_damping | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormlEdRR15scalar_operator), |
|     enumerator)](api/langu        |     [\[4\]](api/languages         |
| ages/cpp_api.html#_CPPv4N5cudaq16 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| noise_model_type13phase_dampingE) | alar_operatormlENSt7complexIdEE), |
| -   [cudaq::noi                   |     [\[5\]](api/languages/cpp     |
| se_model_type::phase_flip_channel | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatormlERK15scalar_operator), |
|     enumerator)](api/languages/   |     [\[6\]]                       |
| cpp_api.html#_CPPv4N5cudaq16noise | (api/languages/cpp_api.html#_CPPv |
| _model_type18phase_flip_channelE) | 4NKR5cudaq15scalar_operatormlEd), |
| -                                 |     [\[7\]](api/language          |
| [cudaq::noise_model_type::unknown | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormlENSt7complexIdEE), |
|     enumerator)](ap               |     [\[8\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4NO5cudaq15scalar |
| cudaq16noise_model_type7unknownE) | _operatormlERK15scalar_operator), |
| -                                 |     [\[9\                         |
| [cudaq::noise_model_type::x_error | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatormlEd) |
|     enumerator)](ap               | -   [cu                           |
| i/languages/cpp_api.html#_CPPv4N5 | daq::scalar_operator::operator\*= |
| cudaq16noise_model_type7x_errorE) |     (C++                          |
| -                                 |     function)](api/languag        |
| [cudaq::noise_model_type::y_error | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormLENSt7complexIdEE), |
|     enumerator)](ap               |     [\[1\]](api/languages/c       |
| i/languages/cpp_api.html#_CPPv4N5 | pp_api.html#_CPPv4N5cudaq15scalar |
| cudaq16noise_model_type7y_errorE) | _operatormLERK15scalar_operator), |
| -                                 |     [\[2                          |
| [cudaq::noise_model_type::z_error | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatormLEd) |
|     enumerator)](ap               | -   [                             |
| i/languages/cpp_api.html#_CPPv4N5 | cudaq::scalar_operator::operator+ |
| cudaq16noise_model_type7z_errorE) |     (C++                          |
| -   [cudaq::num_available_gpus    |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function                      | Pv4N5cudaq15scalar_operatorplENSt |
| )](api/languages/cpp_api.html#_CP | 7complexIdEERK15scalar_operator), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[1\                         |
| -   [cudaq::observe (C++          | ]](api/languages/cpp_api.html#_CP |
|     function)]                    | Pv4N5cudaq15scalar_operatorplENSt |
| (api/languages/cpp_api.html#_CPPv | 7complexIdEERR15scalar_operator), |
| 4I00DpEN5cudaq7observeENSt6vector |     [\[2\]](api/languages/cp      |
| I14observe_resultEERR13QuantumKer | p_api.html#_CPPv4N5cudaq15scalar_ |
| nelRK15SpinOpContainerDpRR4Args), | operatorplEdRK15scalar_operator), |
|     [\[1\]](api/languages/cpp_ap  |     [\[3\]](api/languages/cp      |
| i.html#_CPPv4I0DpEN5cudaq7observe | p_api.html#_CPPv4N5cudaq15scalar_ |
| E14observe_resultNSt6size_tERR13Q | operatorplEdRR15scalar_operator), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[4\]](api/languages         |
|     [\[                           | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| 2\]](api/languages/cpp_api.html#_ | alar_operatorplENSt7complexIdEE), |
| CPPv4I0DpEN5cudaq7observeE14obser |     [\[5\]](api/languages/cpp     |
| ve_resultRK15observe_optionsRR13Q | _api.html#_CPPv4NKR5cudaq15scalar |
| uantumKernelRK7spin_opDpRR4Args), | _operatorplERK15scalar_operator), |
|     [\[3\]](api/lang              |     [\[6\]]                       |
| uages/cpp_api.html#_CPPv4I0DpEN5c | (api/languages/cpp_api.html#_CPPv |
| udaq7observeE14observe_resultRR13 | 4NKR5cudaq15scalar_operatorplEd), |
| QuantumKernelRK7spin_opDpRR4Args) |     [\[7\]]                       |
| -   [cudaq::observe_options (C++  | (api/languages/cpp_api.html#_CPPv |
|     st                            | 4NKR5cudaq15scalar_operatorplEv), |
| ruct)](api/languages/cpp_api.html |     [\[8\]](api/language          |
| #_CPPv4N5cudaq15observe_optionsE) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::observe_result (C++   | alar_operatorplENSt7complexIdEE), |
|                                   |     [\[9\]](api/languages/cp      |
| class)](api/languages/cpp_api.htm | p_api.html#_CPPv4NO5cudaq15scalar |
| l#_CPPv4N5cudaq14observe_resultE) | _operatorplERK15scalar_operator), |
| -                                 |     [\[10\]                       |
|    [cudaq::observe_result::counts | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatorplEd), |
|     function)](api/languages/c    |     [\[11\                        |
| pp_api.html#_CPPv4N5cudaq14observ | ]](api/languages/cpp_api.html#_CP |
| e_result6countsERK12spin_op_term) | Pv4NO5cudaq15scalar_operatorplEv) |
| -   [cudaq::observe_result::dump  | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator+= |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languag        |
| v4N5cudaq14observe_result4dumpEv) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [c                            | alar_operatorpLENSt7complexIdEE), |
| udaq::observe_result::expectation |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|                                   | _operatorpLERK15scalar_operator), |
| function)](api/languages/cpp_api. |     [\[2                          |
| html#_CPPv4N5cudaq14observe_resul | \]](api/languages/cpp_api.html#_C |
| t11expectationERK12spin_op_term), | PPv4N5cudaq15scalar_operatorpLEd) |
|     [\[1\]](api/la                | -   [                             |
| nguages/cpp_api.html#_CPPv4N5cuda | cudaq::scalar_operator::operator- |
| q14observe_result11expectationEv) |     (C++                          |
| -   [cuda                         |     function                      |
| q::observe_result::id_coefficient | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormiENSt |
|     function)](api/langu          | 7complexIdEERK15scalar_operator), |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     [\[1\                         |
| observe_result14id_coefficientEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cuda                         | Pv4N5cudaq15scalar_operatormiENSt |
| q::observe_result::observe_result | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|   function)](api/languages/cpp_ap | operatormiEdRK15scalar_operator), |
| i.html#_CPPv4N5cudaq14observe_res |     [\[3\]](api/languages/cp      |
| ult14observe_resultEdRK7spin_op), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](a                     | operatormiEdRR15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[4\]](api/languages         |
| 5cudaq14observe_result14observe_r | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| esultEdRK7spin_op13sample_result) | alar_operatormiENSt7complexIdEE), |
| -                                 |     [\[5\]](api/languages/cpp     |
|  [cudaq::observe_result::operator | _api.html#_CPPv4NKR5cudaq15scalar |
|     double (C++                   | _operatormiERK15scalar_operator), |
|     functio                       |     [\[6\]]                       |
| n)](api/languages/cpp_api.html#_C | (api/languages/cpp_api.html#_CPPv |
| PPv4N5cudaq14observe_resultcvdEv) | 4NKR5cudaq15scalar_operatormiEd), |
| -                                 |     [\[7\]]                       |
|  [cudaq::observe_result::raw_data | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEv), |
|     function)](ap                 |     [\[8\]](api/language          |
| i/languages/cpp_api.html#_CPPv4N5 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| cudaq14observe_result8raw_dataEv) | alar_operatormiENSt7complexIdEE), |
| -   [cudaq::operator_handler (C++ |     [\[9\]](api/languages/cp      |
|     cl                            | p_api.html#_CPPv4NO5cudaq15scalar |
| ass)](api/languages/cpp_api.html# | _operatormiERK15scalar_operator), |
| _CPPv4N5cudaq16operator_handlerE) |     [\[10\]                       |
| -   [cudaq::optimizable_function  | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatormiEd), |
|     class)                        |     [\[11\                        |
| ](api/languages/cpp_api.html#_CPP | ]](api/languages/cpp_api.html#_CP |
| v4N5cudaq20optimizable_functionE) | Pv4NO5cudaq15scalar_operatormiEv) |
| -   [cudaq::optimization_result   | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator-= |
|     type                          |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/languag        |
| Pv4N5cudaq19optimization_resultE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::optimizer (C++        | alar_operatormIENSt7complexIdEE), |
|     class)](api/languages/cpp_a   |     [\[1\]](api/languages/c       |
| pi.html#_CPPv4N5cudaq9optimizerE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::optimizer::optimize   | _operatormIERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|                                   | \]](api/languages/cpp_api.html#_C |
|  function)](api/languages/cpp_api | PPv4N5cudaq15scalar_operatormIEd) |
| .html#_CPPv4N5cudaq9optimizer8opt | -   [                             |
| imizeEKiRR20optimizable_function) | cudaq::scalar_operator::operator/ |
| -   [cu                           |     (C++                          |
| daq::optimizer::requiresGradients |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function)](api/la             | Pv4N5cudaq15scalar_operatordvENSt |
| nguages/cpp_api.html#_CPPv4N5cuda | 7complexIdEERK15scalar_operator), |
| q9optimizer17requiresGradientsEv) |     [\[1\                         |
| -   [cudaq::orca (C++             | ]](api/languages/cpp_api.html#_CP |
|     type)](api/languages/         | Pv4N5cudaq15scalar_operatordvENSt |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::orca::sample (C++     |     [\[2\]](api/languages/cp      |
|     function)](api/languages/c    | p_api.html#_CPPv4N5cudaq15scalar_ |
| pp_api.html#_CPPv4N5cudaq4orca6sa | operatordvEdRK15scalar_operator), |
| mpleERNSt6vectorINSt6size_tEEERNS |     [\[3\]](api/languages/cp      |
| t6vectorINSt6size_tEEERNSt6vector | p_api.html#_CPPv4N5cudaq15scalar_ |
| IdEERNSt6vectorIdEEiNSt6size_tE), | operatordvEdRR15scalar_operator), |
|     [\[1\]]                       |     [\[4\]](api/languages         |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| 4N5cudaq4orca6sampleERNSt6vectorI | alar_operatordvENSt7complexIdEE), |
| NSt6size_tEEERNSt6vectorINSt6size |     [\[5\]](api/languages/cpp     |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::orca::sample_async    | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
| function)](api/languages/cpp_api. | 4NKR5cudaq15scalar_operatordvEd), |
| html#_CPPv4N5cudaq4orca12sample_a |     [\[7\]](api/language          |
| syncERNSt6vectorINSt6size_tEEERNS | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| t6vectorINSt6size_tEEERNSt6vector | alar_operatordvENSt7complexIdEE), |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     [\[8\]](api/languages/cp      |
|     [\[1\]](api/la                | p_api.html#_CPPv4NO5cudaq15scalar |
| nguages/cpp_api.html#_CPPv4N5cuda | _operatordvERK15scalar_operator), |
| q4orca12sample_asyncERNSt6vectorI |     [\[9\                         |
| NSt6size_tEEERNSt6vectorINSt6size | ]](api/languages/cpp_api.html#_CP |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | Pv4NO5cudaq15scalar_operatordvEd) |
| -   [cudaq::OrcaRemoteRESTQPU     | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator/= |
|     cla                           |     (C++                          |
| ss)](api/languages/cpp_api.html#_ |     function)](api/languag        |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::pauli1 (C++           | alar_operatordVENSt7complexIdEE), |
|     class)](api/languages/cp      |     [\[1\]](api/languages/c       |
| p_api.html#_CPPv4N5cudaq6pauli1E) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _operatordVERK15scalar_operator), |
|    [cudaq::pauli1::num_parameters |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     member)]                      | PPv4N5cudaq15scalar_operatordVEd) |
| (api/languages/cpp_api.html#_CPPv | -   [                             |
| 4N5cudaq6pauli114num_parametersE) | cudaq::scalar_operator::operator= |
| -   [cudaq::pauli1::num_targets   |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     membe                         | pp_api.html#_CPPv4N5cudaq15scalar |
| r)](api/languages/cpp_api.html#_C | _operatoraSERK15scalar_operator), |
| PPv4N5cudaq6pauli111num_targetsE) |     [\[1\]](api/languages/        |
| -   [cudaq::pauli1::pauli1 (C++   | cpp_api.html#_CPPv4N5cudaq15scala |
|     function)](api/languages/cpp_ | r_operatoraSERR15scalar_operator) |
| api.html#_CPPv4N5cudaq6pauli16pau | -   [c                            |
| li1ERKNSt6vectorIN5cudaq4realEEE) | udaq::scalar_operator::operator== |
| -   [cudaq::pauli2 (C++           |     (C++                          |
|     class)](api/languages/cp      |     function)](api/languages/c    |
| p_api.html#_CPPv4N5cudaq6pauli2E) | pp_api.html#_CPPv4NK5cudaq15scala |
| -                                 | r_operatoreqERK15scalar_operator) |
|    [cudaq::pauli2::num_parameters | -   [cudaq:                       |
|     (C++                          | :scalar_operator::scalar_operator |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     func                          |
| 4N5cudaq6pauli214num_parametersE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::pauli2::num_targets   | #_CPPv4N5cudaq15scalar_operator15 |
|     (C++                          | scalar_operatorENSt7complexIdEE), |
|     membe                         |     [\[1\]](api/langu             |
| r)](api/languages/cpp_api.html#_C | ages/cpp_api.html#_CPPv4N5cudaq15 |
| PPv4N5cudaq6pauli211num_targetsE) | scalar_operator15scalar_operatorE |
| -   [cudaq::pauli2::pauli2 (C++   | RK15scalar_callbackRRNSt13unorder |
|     function)](api/languages/cpp_ | ed_mapINSt6stringENSt6stringEEE), |
| api.html#_CPPv4N5cudaq6pauli26pau |     [\[2\                         |
| li2ERKNSt6vectorIN5cudaq4realEEE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::phase_damping (C++    | Pv4N5cudaq15scalar_operator15scal |
|                                   | ar_operatorERK15scalar_operator), |
|  class)](api/languages/cpp_api.ht |     [\[3\]](api/langu             |
| ml#_CPPv4N5cudaq13phase_dampingE) | ages/cpp_api.html#_CPPv4N5cudaq15 |
| -   [cud                          | scalar_operator15scalar_operatorE |
| aq::phase_damping::num_parameters | RR15scalar_callbackRRNSt13unorder |
|     (C++                          | ed_mapINSt6stringENSt6stringEEE), |
|     member)](api/lan              |     [\[4\                         |
| guages/cpp_api.html#_CPPv4N5cudaq | ]](api/languages/cpp_api.html#_CP |
| 13phase_damping14num_parametersE) | Pv4N5cudaq15scalar_operator15scal |
| -   [                             | ar_operatorERR15scalar_operator), |
| cudaq::phase_damping::num_targets |     [\[5\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     member)](api/                 | lar_operator15scalar_operatorEd), |
| languages/cpp_api.html#_CPPv4N5cu |     [\[6\]](api/languag           |
| daq13phase_damping11num_targetsE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::phase_flip_channel    | alar_operator15scalar_operatorEv) |
|     (C++                          | -   [                             |
|     clas                          | cudaq::scalar_operator::to_matrix |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq18phase_flip_channelE) |                                   |
| -   [cudaq::p                     |   function)](api/languages/cpp_ap |
| hase_flip_channel::num_parameters | i.html#_CPPv4NK5cudaq15scalar_ope |
|     (C++                          | rator9to_matrixERKNSt13unordered_ |
|     member)](api/language         | mapINSt6stringENSt7complexIdEEEE) |
| s/cpp_api.html#_CPPv4N5cudaq18pha | -   [                             |
| se_flip_channel14num_parametersE) | cudaq::scalar_operator::to_string |
| -   [cudaq                        |     (C++                          |
| ::phase_flip_channel::num_targets |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     member)](api/langu            | daq15scalar_operator9to_stringEv) |
| ages/cpp_api.html#_CPPv4N5cudaq18 | -   [cudaq::s                     |
| phase_flip_channel11num_targetsE) | calar_operator::\~scalar_operator |
| -   [cudaq::product_op (C++       |     (C++                          |
|                                   |     functio                       |
|  class)](api/languages/cpp_api.ht | n)](api/languages/cpp_api.html#_C |
| ml#_CPPv4I0EN5cudaq10product_opE) | PPv4N5cudaq15scalar_operatorD0Ev) |
| -   [cudaq::product_op::begin     | -   [cudaq::set_noise (C++        |
|     (C++                          |     function)](api/langu          |
|     functio                       | ages/cpp_api.html#_CPPv4N5cudaq9s |
| n)](api/languages/cpp_api.html#_C | et_noiseERKN5cudaq11noise_modelE) |
| PPv4NK5cudaq10product_op5beginEv) | -   [cudaq::set_random_seed (C++  |
| -                                 |     function)](api/               |
|  [cudaq::product_op::canonicalize | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq15set_random_seedENSt6size_tE) |
|     func                          | -   [cudaq::simulation_precision  |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq10product_op12canon |     enum)                         |
| icalizeERKNSt3setINSt6size_tEEE), | ](api/languages/cpp_api.html#_CPP |
|     [\[1\]](api                   | v4N5cudaq20simulation_precisionE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [                             |
| udaq10product_op12canonicalizeEv) | cudaq::simulation_precision::fp32 |
| -   [                             |     (C++                          |
| cudaq::product_op::const_iterator |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     struct)](api/                 | udaq20simulation_precision4fp32E) |
| languages/cpp_api.html#_CPPv4N5cu | -   [                             |
| daq10product_op14const_iteratorE) | cudaq::simulation_precision::fp64 |
| -   [cudaq::product_o             |     (C++                          |
| p::const_iterator::const_iterator |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     fu                            | udaq20simulation_precision4fp64E) |
| nction)](api/languages/cpp_api.ht | -   [cudaq::SimulationState (C++  |
| ml#_CPPv4N5cudaq10product_op14con |     c                             |
| st_iterator14const_iteratorEPK10p | lass)](api/languages/cpp_api.html |
| roduct_opI9HandlerTyENSt6size_tE) | #_CPPv4N5cudaq15SimulationStateE) |
| -   [cudaq::produ                 | -   [                             |
| ct_op::const_iterator::operator!= | cudaq::SimulationState::precision |
|     (C++                          |     (C++                          |
|     fun                           |     enum)](api                    |
| ction)](api/languages/cpp_api.htm | /languages/cpp_api.html#_CPPv4N5c |
| l#_CPPv4NK5cudaq10product_op14con | udaq15SimulationState9precisionE) |
| st_iteratorneERK14const_iterator) | -   [cudaq:                       |
| -   [cudaq::produ                 | :SimulationState::precision::fp32 |
| ct_op::const_iterator::operator\* |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     function)](api/lang           | uages/cpp_api.html#_CPPv4N5cudaq1 |
| uages/cpp_api.html#_CPPv4NK5cudaq | 5SimulationState9precision4fp32E) |
| 10product_op14const_iteratormlEv) | -   [cudaq:                       |
| -   [cudaq::produ                 | :SimulationState::precision::fp64 |
| ct_op::const_iterator::operator++ |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     function)](api/lang           | uages/cpp_api.html#_CPPv4N5cudaq1 |
| uages/cpp_api.html#_CPPv4N5cudaq1 | 5SimulationState9precision4fp64E) |
| 0product_op14const_iteratorppEi), | -                                 |
|     [\[1\]](api/lan               |   [cudaq::SimulationState::Tensor |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 10product_op14const_iteratorppEv) |     struct)](                     |
| -   [cudaq::produc                | api/languages/cpp_api.html#_CPPv4 |
| t_op::const_iterator::operator\-- | N5cudaq15SimulationState6TensorE) |
|     (C++                          | -   [cudaq::spin_handler (C++     |
|     function)](api/lang           |                                   |
| uages/cpp_api.html#_CPPv4N5cudaq1 |   class)](api/languages/cpp_api.h |
| 0product_op14const_iteratormmEi), | tml#_CPPv4N5cudaq12spin_handlerE) |
|     [\[1\]](api/lan               | -   [cudaq:                       |
| guages/cpp_api.html#_CPPv4N5cudaq | :spin_handler::to_diagonal_matrix |
| 10product_op14const_iteratormmEv) |     (C++                          |
| -   [cudaq::produc                |     function)](api/la             |
| t_op::const_iterator::operator-\> | nguages/cpp_api.html#_CPPv4NK5cud |
|     (C++                          | aq12spin_handler18to_diagonal_mat |
|     function)](api/lan            | rixERNSt13unordered_mapINSt6size_ |
| guages/cpp_api.html#_CPPv4N5cudaq | tENSt7int64_tEEERKNSt13unordered_ |
| 10product_op14const_iteratorptEv) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::produ                 | -                                 |
| ct_op::const_iterator::operator== |   [cudaq::spin_handler::to_matrix |
|     (C++                          |     (C++                          |
|     fun                           |     function                      |
| ction)](api/languages/cpp_api.htm | )](api/languages/cpp_api.html#_CP |
| l#_CPPv4NK5cudaq10product_op14con | Pv4N5cudaq12spin_handler9to_matri |
| st_iteratoreqERK14const_iterator) | xERKNSt6stringENSt7complexIdEEb), |
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
| cudaq.gradients.CentralDifference | .spin.SpinOperatorTerm.to_string) |
|     attribute)](api/la            | -   [TraceInstruction (class in   |
| nguages/python_api.html#cudaq.gra |     cudaq.p                       |
| dients.CentralDifference.to_json) | tsbe)](api/languages/python_api.h |
|     -   [(                        | tml#cudaq.ptsbe.TraceInstruction) |
| cudaq.gradients.ForwardDifference | -   [TraceInstructionType (class  |
|         attribute)](api/la        |     in                            |
| nguages/python_api.html#cudaq.gra |     cudaq.ptsbe                   |
| dients.ForwardDifference.to_json) | )](api/languages/python_api.html# |
|     -                             | cudaq.ptsbe.TraceInstructionType) |
|  [(cudaq.gradients.ParameterShift | -   [trajectories                 |
|         attribute)](api           |                                   |
| /languages/python_api.html#cudaq. |   (cudaq.ptsbe.PTSBEExecutionData |
| gradients.ParameterShift.to_json) |     property)](api/lang           |
|     -   [(                        | uages/python_api.html#cudaq.ptsbe |
| cudaq.operators.spin.SpinOperator | .PTSBEExecutionData.trajectories) |
|         attribute)](api/la        | -   [trajectory_id                |
| nguages/python_api.html#cudaq.ope |     (cudaq.ptsbe.KrausTrajectory  |
| rators.spin.SpinOperator.to_json) |     property)](api/la             |
|     -   [(cuda                    | nguages/python_api.html#cudaq.pts |
| q.operators.spin.SpinOperatorTerm | be.KrausTrajectory.trajectory_id) |
|         attribute)](api/langua    | -   [translate() (in module       |
| ges/python_api.html#cudaq.operato |     cudaq)](api/languages         |
| rs.spin.SpinOperatorTerm.to_json) | /python_api.html#cudaq.translate) |
|     -   [(cudaq.optimizers.Adam   | -   [trim                         |
|         attrib                    |     (cu                           |
| ute)](api/languages/python_api.ht | daq.operators.boson.BosonOperator |
| ml#cudaq.optimizers.Adam.to_json) |     attribute)](api/l             |
|     -   [(cudaq.optimizers.COBYLA | anguages/python_api.html#cudaq.op |
|         attribut                  | erators.boson.BosonOperator.trim) |
| e)](api/languages/python_api.html |     -   [(cudaq.                  |
| #cudaq.optimizers.COBYLA.to_json) | operators.fermion.FermionOperator |
|     -   [                         |         attribute)](api/langu     |
| (cudaq.optimizers.GradientDescent | ages/python_api.html#cudaq.operat |
|         attribute)](api/l         | ors.fermion.FermionOperator.trim) |
| anguages/python_api.html#cudaq.op |     -                             |
| timizers.GradientDescent.to_json) |  [(cudaq.operators.MatrixOperator |
|     -   [(cudaq.optimizers.LBFGS  |         attribute)](              |
|         attribu                   | api/languages/python_api.html#cud |
| te)](api/languages/python_api.htm | aq.operators.MatrixOperator.trim) |
| l#cudaq.optimizers.LBFGS.to_json) |     -   [(                        |
|                                   | cudaq.operators.spin.SpinOperator |
| -   [(cudaq.optimizers.NelderMead |         attribute)](api           |
|         attribute)](              | /languages/python_api.html#cudaq. |
| api/languages/python_api.html#cud | operators.spin.SpinOperator.trim) |
| aq.optimizers.NelderMead.to_json) | -   [type                         |
|     -   [(cudaq.optimizers.SGD    |     (c                            |
|         attri                     | udaq.ptsbe.ShotAllocationStrategy |
| bute)](api/languages/python_api.h |     property)](api/               |
| tml#cudaq.optimizers.SGD.to_json) | languages/python_api.html#cudaq.p |
|     -   [(cudaq.optimizers.SPSA   | tsbe.ShotAllocationStrategy.type) |
|         attrib                    |     -                             |
| ute)](api/languages/python_api.ht |    [(cudaq.ptsbe.TraceInstruction |
| ml#cudaq.optimizers.SPSA.to_json) |         property)                 |
| -   [to_json()                    | ](api/languages/python_api.html#c |
|     (cudaq.PyKernelDecorator      | udaq.ptsbe.TraceInstruction.type) |
|     metho                         | -   [type_to_str()                |
| d)](api/languages/python_api.html |     (cudaq.PyKernelDecorator      |
| #cudaq.PyKernelDecorator.to_json) |     static                        |
| -   [to_matrix                    |     method)](                     |
|     (cu                           | api/languages/python_api.html#cud |
| daq.operators.boson.BosonOperator | aq.PyKernelDecorator.type_to_str) |
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
