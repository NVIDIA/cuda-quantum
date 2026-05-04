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
|     class)](                      |     function)](api/la             |
| api/languages/cpp_api.html#_CPPv4 | nguages/cpp_api.html#_CPPv4I0_NSt |
| N5cudaq22BaseRemoteSimulatorQPUE) | 11enable_if_tIXaantNSt7is_sameI1T |
| -   [cudaq::bit_flip_channel (C++ | 9HandlerTyE5valueENSt16is_constru |
|     cl                            | ctibleI9HandlerTy1TE5valueEEbEEEN |
| ass)](api/languages/cpp_api.html# | 5cudaq10product_opaSER10product_o |
| _CPPv4N5cudaq16bit_flip_channelE) | pI9HandlerTyERK10product_opI1TE), |
| -   [cudaq:                       |     [\[1\]](api/languages/cpp     |
| :bit_flip_channel::num_parameters | _api.html#_CPPv4N5cudaq10product_ |
|     (C++                          | opaSERK10product_opI9HandlerTyE), |
|     member)](api/langua           |     [\[2\]](api/languages/cp      |
| ges/cpp_api.html#_CPPv4N5cudaq16b | p_api.html#_CPPv4N5cudaq10product |
| it_flip_channel14num_parametersE) | _opaSERR10product_opI9HandlerTyE) |
| -   [cud                          | -                                 |
| aq::bit_flip_channel::num_targets |    [cudaq::product_op::operator== |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/languages/cpp  |
| guages/cpp_api.html#_CPPv4N5cudaq | _api.html#_CPPv4NK5cudaq10product |
| 16bit_flip_channel11num_targetsE) | _opeqERK10product_opI9HandlerTyE) |
| -   [cudaq::boson_handler (C++    | -                                 |
|                                   |  [cudaq::product_op::operator\[\] |
|  class)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13boson_handlerE) |     function)](ap                 |
| -   [cudaq::boson_op (C++         | i/languages/cpp_api.html#_CPPv4NK |
|     type)](api/languages/cpp_     | 5cudaq10product_opixENSt6size_tE) |
| api.html#_CPPv4N5cudaq8boson_opE) | -                                 |
| -   [cudaq::boson_op_term (C++    |    [cudaq::product_op::product_op |
|                                   |     (C++                          |
|   type)](api/languages/cpp_api.ht |     function)](api/languages/c    |
| ml#_CPPv4N5cudaq13boson_op_termE) | pp_api.html#_CPPv4I0_NSt11enable_ |
| -   [cudaq::CodeGenConfig (C++    | if_tIXaaNSt7is_sameI9HandlerTy14m |
|                                   | atrix_handlerE5valueEaantNSt7is_s |
| struct)](api/languages/cpp_api.ht | ameI1T9HandlerTyE5valueENSt16is_c |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | onstructibleI9HandlerTy1TE5valueE |
| -   [cudaq::commutation_relations | EbEEEN5cudaq10product_op10product |
|     (C++                          | _opERK10product_opI1TERKN14matrix |
|     struct)]                      | _handler20commutation_behaviorE), |
| (api/languages/cpp_api.html#_CPPv |                                   |
| 4N5cudaq21commutation_relationsE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::complex (C++          | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|     type)](api/languages/cpp      | tNSt7is_sameI1T9HandlerTyE5valueE |
| _api.html#_CPPv4N5cudaq7complexE) | NSt16is_constructibleI9HandlerTy1 |
| -   [cudaq::complex_matrix (C++   | TE5valueEEbEEEN5cudaq10product_op |
|                                   | 10product_opERK10product_opI1TE), |
| class)](api/languages/cpp_api.htm |                                   |
| l#_CPPv4N5cudaq14complex_matrixE) |   [\[2\]](api/languages/cpp_api.h |
| -                                 | tml#_CPPv4N5cudaq10product_op10pr |
|   [cudaq::complex_matrix::adjoint | oduct_opENSt6size_tENSt6size_tE), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     function)](a                  | p_api.html#_CPPv4N5cudaq10product |
| pi/languages/cpp_api.html#_CPPv4N | _op10product_opENSt7complexIdEE), |
| 5cudaq14complex_matrix7adjointEv) |     [\[4\]](api/l                 |
| -   [cudaq::                      | anguages/cpp_api.html#_CPPv4N5cud |
| complex_matrix::diagonal_elements | aq10product_op10product_opERK10pr |
|     (C++                          | oduct_opI9HandlerTyENSt6size_tE), |
|     function)](api/languages      |     [\[5\]](api/l                 |
| /cpp_api.html#_CPPv4NK5cudaq14com | anguages/cpp_api.html#_CPPv4N5cud |
| plex_matrix17diagonal_elementsEi) | aq10product_op10product_opERR10pr |
| -   [cudaq::complex_matrix::dump  | oduct_opI9HandlerTyENSt6size_tE), |
|     (C++                          |     [\[6\]](api/languages         |
|     function)](api/language       | /cpp_api.html#_CPPv4N5cudaq10prod |
| s/cpp_api.html#_CPPv4NK5cudaq14co | uct_op10product_opERR9HandlerTy), |
| mplex_matrix4dumpERNSt7ostreamE), |     [\[7\]](ap                    |
|     [\[1\]]                       | i/languages/cpp_api.html#_CPPv4N5 |
| (api/languages/cpp_api.html#_CPPv | cudaq10product_op10product_opEd), |
| 4NK5cudaq14complex_matrix4dumpEv) |     [\[8\]](a                     |
| -   [c                            | pi/languages/cpp_api.html#_CPPv4N |
| udaq::complex_matrix::eigenvalues | 5cudaq10product_op10product_opEv) |
|     (C++                          | -   [cuda                         |
|     function)](api/lan            | q::product_op::to_diagonal_matrix |
| guages/cpp_api.html#_CPPv4NK5cuda |     (C++                          |
| q14complex_matrix11eigenvaluesEv) |     function)](api/               |
| -   [cu                           | languages/cpp_api.html#_CPPv4NK5c |
| daq::complex_matrix::eigenvectors | udaq10product_op18to_diagonal_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function)](api/lang           | ENSt7int64_tEEERKNSt13unordered_m |
| uages/cpp_api.html#_CPPv4NK5cudaq | apINSt6stringENSt7complexIdEEEEb) |
| 14complex_matrix12eigenvectorsEv) | -   [cudaq::product_op::to_matrix |
| -   [c                            |     (C++                          |
| udaq::complex_matrix::exponential |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/la             | _CPPv4NK5cudaq10product_op9to_mat |
| nguages/cpp_api.html#_CPPv4N5cuda | rixENSt13unordered_mapINSt6size_t |
| q14complex_matrix11exponentialEv) | ENSt7int64_tEEERKNSt13unordered_m |
| -                                 | apINSt6stringENSt7complexIdEEEEb) |
|  [cudaq::complex_matrix::identity | -   [cu                           |
|     (C++                          | daq::product_op::to_sparse_matrix |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq14comp |     function)](ap                 |
| lex_matrix8identityEKNSt6size_tE) | i/languages/cpp_api.html#_CPPv4NK |
| -                                 | 5cudaq10product_op16to_sparse_mat |
| [cudaq::complex_matrix::kronecker | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/lang           | apINSt6stringENSt7complexIdEEEEb) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -   [cudaq::product_op::to_string |
| daq14complex_matrix9kroneckerE14c |     (C++                          |
| omplex_matrix8Iterable8Iterable), |     function)](                   |
|     [\[1\]](api/l                 | api/languages/cpp_api.html#_CPPv4 |
| anguages/cpp_api.html#_CPPv4N5cud | NK5cudaq10product_op9to_stringEv) |
| aq14complex_matrix9kroneckerERK14 | -                                 |
| complex_matrixRK14complex_matrix) |  [cudaq::product_op::\~product_op |
| -   [cudaq::c                     |     (C++                          |
| omplex_matrix::minimal_eigenvalue |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function)](api/languages/     | ml#_CPPv4N5cudaq10product_opD0Ev) |
| cpp_api.html#_CPPv4NK5cudaq14comp | -   [cudaq::ptsbe (C++            |
| lex_matrix18minimal_eigenvalueEv) |     type)](api/languages/c        |
| -   [                             | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| cudaq::complex_matrix::operator() | -   [cudaq::p                     |
|     (C++                          | tsbe::ConditionalSamplingStrategy |
|     function)](api/languages/cpp  |     (C++                          |
| _api.html#_CPPv4N5cudaq14complex_ |     class)](api/languag           |
| matrixclENSt6size_tENSt6size_tE), | es/cpp_api.html#_CPPv4N5cudaq5pts |
|     [\[1\]](api/languages/cpp     | be27ConditionalSamplingStrategyE) |
| _api.html#_CPPv4NK5cudaq14complex | -   [cudaq::ptsbe::C              |
| _matrixclENSt6size_tENSt6size_tE) | onditionalSamplingStrategy::clone |
| -   [                             |     (C++                          |
| cudaq::complex_matrix::operator\* |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     function)](api/langua         | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| ges/cpp_api.html#_CPPv4N5cudaq14c | ditionalSamplingStrategy5cloneEv) |
| omplex_matrixmlEN14complex_matrix | -   [cuda                         |
| 10value_typeERK14complex_matrix), | q::ptsbe::ConditionalSamplingStra |
|     [\[1\]                        | tegy::ConditionalSamplingStrategy |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq14complex_matrixmlERK14c |     function)](api/lang           |
| omplex_matrixRK14complex_matrix), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|                                   | ptsbe27ConditionalSamplingStrateg |
|  [\[2\]](api/languages/cpp_api.ht | y27ConditionalSamplingStrategyE19 |
| ml#_CPPv4N5cudaq14complex_matrixm | TrajectoryPredicateNSt8uint64_tE) |
| lERK14complex_matrixRKNSt6vectorI | -                                 |
| N14complex_matrix10value_typeEEE) |   [cudaq::ptsbe::ConditionalSampl |
| -                                 | ingStrategy::generateTrajectories |
| [cudaq::complex_matrix::operator+ |     (C++                          |
|     (C++                          |     function)](api/language       |
|     function                      | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| )](api/languages/cpp_api.html#_CP | be27ConditionalSamplingStrategy20 |
| Pv4N5cudaq14complex_matrixplERK14 | generateTrajectoriesENSt4spanIKN6 |
| complex_matrixRK14complex_matrix) | detail10NoisePointEEENSt6size_tE) |
| -                                 | -   [cudaq::ptsbe::               |
| [cudaq::complex_matrix::operator- | ConditionalSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/languages/cpp_ |
| )](api/languages/cpp_api.html#_CP | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| Pv4N5cudaq14complex_matrixmiERK14 | nditionalSamplingStrategy4nameEv) |
| complex_matrixRK14complex_matrix) | -   [cudaq:                       |
| -   [cu                           | :ptsbe::ConditionalSamplingStrate |
| daq::complex_matrix::operator\[\] | gy::\~ConditionalSamplingStrategy |
|     (C++                          |     (C++                          |
|                                   |     function)](api/languages/     |
|  function)](api/languages/cpp_api | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| .html#_CPPv4N5cudaq14complex_matr | 7ConditionalSamplingStrategyD0Ev) |
| ixixERKNSt6vectorINSt6size_tEEE), | -                                 |
|     [\[1\]](api/languages/cpp_api | [cudaq::ptsbe::detail::NoisePoint |
| .html#_CPPv4NK5cudaq14complex_mat |     (C++                          |
| rixixERKNSt6vectorINSt6size_tEEE) |     struct)](a                    |
| -   [cudaq::complex_matrix::power | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq5ptsbe6detail10NoisePointE) |
|     function)]                    | -   [cudaq::p                     |
| (api/languages/cpp_api.html#_CPPv | tsbe::detail::NoisePoint::channel |
| 4N5cudaq14complex_matrix5powerEi) |     (C++                          |
| -                                 |     member)](api/langu            |
|  [cudaq::complex_matrix::set_zero | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     (C++                          | tsbe6detail10NoisePoint7channelE) |
|     function)](ap                 | -   [cudaq::ptsbe::det            |
| i/languages/cpp_api.html#_CPPv4N5 | ail::NoisePoint::circuit_location |
| cudaq14complex_matrix8set_zeroEv) |     (C++                          |
| -                                 |     member)](api/languages/cpp_a  |
| [cudaq::complex_matrix::to_string | pi.html#_CPPv4N5cudaq5ptsbe6detai |
|     (C++                          | l10NoisePoint16circuit_locationE) |
|     function)](api/               | -   [cudaq::p                     |
| languages/cpp_api.html#_CPPv4NK5c | tsbe::detail::NoisePoint::op_name |
| udaq14complex_matrix9to_stringEv) |     (C++                          |
| -   [                             |     member)](api/langu            |
| cudaq::complex_matrix::value_type | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     (C++                          | tsbe6detail10NoisePoint7op_nameE) |
|     type)](api/                   | -   [cudaq::                      |
| languages/cpp_api.html#_CPPv4N5cu | ptsbe::detail::NoisePoint::qubits |
| daq14complex_matrix10value_typeE) |     (C++                          |
| -   [cudaq::contrib (C++          |     member)](api/lang             |
|     type)](api/languages/cpp      | uages/cpp_api.html#_CPPv4N5cudaq5 |
| _api.html#_CPPv4N5cudaq7contribE) | ptsbe6detail10NoisePoint6qubitsE) |
| -   [cudaq::contrib::draw (C++    | -   [cudaq::                      |
|     function)                     | ptsbe::ExhaustiveSamplingStrategy |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4I0DpEN5cudaq7contrib4drawENSt6s |     class)](api/langua            |
| tringERR13QuantumKernelDpRR4Args) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -                                 | sbe26ExhaustiveSamplingStrategyE) |
| [cudaq::contrib::get_unitary_cmat | -   [cudaq::ptsbe::               |
|     (C++                          | ExhaustiveSamplingStrategy::clone |
|     function)](api/languages/cp   |     (C++                          |
| p_api.html#_CPPv4I0DpEN5cudaq7con |     function)](api/languages/cpp_ |
| trib16get_unitary_cmatE14complex_ | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| matrixRR13QuantumKernelDpRR4Args) | haustiveSamplingStrategy5cloneEv) |
| -   [cudaq::CusvState (C++        | -   [cu                           |
|                                   | daq::ptsbe::ExhaustiveSamplingStr |
|    class)](api/languages/cpp_api. | ategy::ExhaustiveSamplingStrategy |
| html#_CPPv4I0EN5cudaq9CusvStateE) |     (C++                          |
| -   [cudaq::depolarization1 (C++  |     function)](api/la             |
|     c                             | nguages/cpp_api.html#_CPPv4N5cuda |
| lass)](api/languages/cpp_api.html | q5ptsbe26ExhaustiveSamplingStrate |
| #_CPPv4N5cudaq15depolarization1E) | gy26ExhaustiveSamplingStrategyEv) |
| -   [cudaq::depolarization2 (C++  | -                                 |
|     c                             |    [cudaq::ptsbe::ExhaustiveSampl |
| lass)](api/languages/cpp_api.html | ingStrategy::generateTrajectories |
| #_CPPv4N5cudaq15depolarization2E) |     (C++                          |
| -   [cudaq:                       |     function)](api/languag        |
| :depolarization2::depolarization2 | es/cpp_api.html#_CPPv4NK5cudaq5pt |
|     (C++                          | sbe26ExhaustiveSamplingStrategy20 |
|     function)](api/languages/cp   | generateTrajectoriesENSt4spanIKN6 |
| p_api.html#_CPPv4N5cudaq15depolar | detail10NoisePointEEENSt6size_tE) |
| ization215depolarization2EK4real) | -   [cudaq::ptsbe:                |
| -   [cudaq                        | :ExhaustiveSamplingStrategy::name |
| ::depolarization2::num_parameters |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     member)](api/langu            | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| ages/cpp_api.html#_CPPv4N5cudaq15 | xhaustiveSamplingStrategy4nameEv) |
| depolarization214num_parametersE) | -   [cuda                         |
| -   [cu                           | q::ptsbe::ExhaustiveSamplingStrat |
| daq::depolarization2::num_targets | egy::\~ExhaustiveSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api/languages      |
| nguages/cpp_api.html#_CPPv4N5cuda | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| q15depolarization211num_targetsE) | 26ExhaustiveSamplingStrategyD0Ev) |
| -                                 | -   [cuda                         |
|    [cudaq::depolarization_channel | q::ptsbe::OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     class)](                      |     class)](api/lan               |
| api/languages/cpp_api.html#_CPPv4 | guages/cpp_api.html#_CPPv4N5cudaq |
| N5cudaq22depolarization_channelE) | 5ptsbe23OrderedSamplingStrategyE) |
| -   [cudaq::depol                 | -   [cudaq::ptsb                  |
| arization_channel::num_parameters | e::OrderedSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     function)](api/languages/c    |
| p_api.html#_CPPv4N5cudaq22depolar | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
| ization_channel14num_parametersE) | 3OrderedSamplingStrategy5cloneEv) |
| -   [cudaq::de                    | -   [cudaq::ptsbe::OrderedSampl   |
| polarization_channel::num_targets | ingStrategy::generateTrajectories |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)](api/lang           |
| /cpp_api.html#_CPPv4N5cudaq22depo | uages/cpp_api.html#_CPPv4NK5cudaq |
| larization_channel11num_targetsE) | 5ptsbe23OrderedSamplingStrategy20 |
| -   [cudaq::details (C++          | generateTrajectoriesENSt4spanIKN6 |
|     type)](api/languages/cpp      | detail10NoisePointEEENSt6size_tE) |
| _api.html#_CPPv4N5cudaq7detailsE) | -   [cudaq::pts                   |
| -   [cudaq::details::future (C++  | be::OrderedSamplingStrategy::name |
|                                   |     (C++                          |
|  class)](api/languages/cpp_api.ht |     function)](api/languages/     |
| ml#_CPPv4N5cudaq7details6futureE) | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| -                                 | 23OrderedSamplingStrategy4nameEv) |
|   [cudaq::details::future::future | -                                 |
|     (C++                          |    [cudaq::ptsbe::OrderedSampling |
|     functio                       | Strategy::OrderedSamplingStrategy |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq7details6future6future |     function)](                   |
| ERNSt6vectorI3JobEERNSt6stringERN | api/languages/cpp_api.html#_CPPv4 |
| St3mapINSt6stringENSt6stringEEE), | N5cudaq5ptsbe23OrderedSamplingStr |
|     [\[1\]](api/lang              | ategy23OrderedSamplingStrategyEv) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -                                 |
| details6future6futureERR6future), |  [cudaq::ptsbe::OrderedSamplingSt |
|     [\[2\]]                       | rategy::\~OrderedSamplingStrategy |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq7details6future6futureEv) |     function)](api/langua         |
| -   [cu                           | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| daq::details::kernel_builder_base | sbe23OrderedSamplingStrategyD0Ev) |
|     (C++                          | -   [cudaq::pts                   |
|     class)](api/l                 | be::ProbabilisticSamplingStrategy |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq7details19kernel_builder_baseE) |     class)](api/languages         |
| -   [cudaq::details::             | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| kernel_builder_base::operator\<\< | 29ProbabilisticSamplingStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::Pro            |
|     function)](api/langua         | babilisticSamplingStrategy::clone |
| ges/cpp_api.html#_CPPv4N5cudaq7de |     (C++                          |
| tails19kernel_builder_baselsERNSt |                                   |
| 7ostreamERK19kernel_builder_base) |  function)](api/languages/cpp_api |
| -   [                             | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| cudaq::details::KernelBuilderType | bilisticSamplingStrategy5cloneEv) |
|     (C++                          | -                                 |
|     class)](api                   | [cudaq::ptsbe::ProbabilisticSampl |
| /languages/cpp_api.html#_CPPv4N5c | ingStrategy::generateTrajectories |
| udaq7details17KernelBuilderTypeE) |     (C++                          |
| -   [cudaq::d                     |     function)](api/languages/     |
| etails::KernelBuilderType::create | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     (C++                          | 29ProbabilisticSamplingStrategy20 |
|     function)                     | generateTrajectoriesENSt4spanIKN6 |
| ](api/languages/cpp_api.html#_CPP | detail10NoisePointEEENSt6size_tE) |
| v4N5cudaq7details17KernelBuilderT | -   [cudaq::ptsbe::Pr             |
| ype6createEPN4mlir11MLIRContextE) | obabilisticSamplingStrategy::name |
| -   [cudaq::details::Ker          |     (C++                          |
| nelBuilderType::KernelBuilderType |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)](api/lang           | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| uages/cpp_api.html#_CPPv4N5cudaq7 | abilisticSamplingStrategy4nameEv) |
| details17KernelBuilderType17Kerne | -   [cudaq::p                     |
| lBuilderTypeERRNSt8functionIFN4ml | tsbe::ProbabilisticSamplingStrate |
| ir4TypeEPN4mlir11MLIRContextEEEE) | gy::ProbabilisticSamplingStrategy |
| -   [cudaq::diag_matrix_callback  |     (C++                          |
|     (C++                          |     function)]                    |
|     class)                        | (api/languages/cpp_api.html#_CPPv |
| ](api/languages/cpp_api.html#_CPP | 4N5cudaq5ptsbe29ProbabilisticSamp |
| v4N5cudaq20diag_matrix_callbackE) | lingStrategy29ProbabilisticSampli |
| -   [cudaq::dyn (C++              | ngStrategyENSt8optionalINSt8uint6 |
|     member)](api/languages        | 4_tEEENSt8optionalINSt6size_tEEE) |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | -   [cudaq::pts                   |
| -   [cudaq::ExecutionContext (C++ | be::ProbabilisticSamplingStrategy |
|     cl                            | ::\~ProbabilisticSamplingStrategy |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16ExecutionContextE) |     function)](api/languages/cp   |
| -   [cudaq                        | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| ::ExecutionContext::amplitudeMaps | robabilisticSamplingStrategyD0Ev) |
|     (C++                          | -                                 |
|     member)](api/langu            | [cudaq::ptsbe::PTSBEExecutionData |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13amplitudeMapsE) |     struct)](ap                   |
| -   [c                            | i/languages/cpp_api.html#_CPPv4N5 |
| udaq::ExecutionContext::asyncExec | cudaq5ptsbe18PTSBEExecutionDataE) |
|     (C++                          | -   [cudaq::ptsbe::PTSBE          |
|     member)](api/                 | ExecutionData::count_instructions |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9asyncExecE) |     function)](api/l              |
| -   [cud                          | anguages/cpp_api.html#_CPPv4NK5cu |
| aq::ExecutionContext::asyncResult | daq5ptsbe18PTSBEExecutionData18co |
|     (C++                          | unt_instructionsE20TraceInstructi |
|     member)](api/lan              | onTypeNSt8optionalINSt6stringEEE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::ptsbe::P              |
| 16ExecutionContext11asyncResultE) | TSBEExecutionData::get_trajectory |
| -   [cudaq:                       |     (C++                          |
| :ExecutionContext::batchIteration |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)](api/langua           | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| ges/cpp_api.html#_CPPv4N5cudaq16E | Data14get_trajectoryENSt6size_tE) |
| xecutionContext14batchIterationE) | -   [cudaq::ptsbe:                |
| -   [cudaq::E                     | :PTSBEExecutionData::instructions |
| xecutionContext::canHandleObserve |     (C++                          |
|     (C++                          |     member)](api/languages/cp     |
|     member)](api/language         | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | TSBEExecutionData12instructionsE) |
| cutionContext16canHandleObserveE) | -   [cudaq::ptsbe:                |
| -   [cudaq::E                     | :PTSBEExecutionData::trajectories |
| xecutionContext::ExecutionContext |     (C++                          |
|     (C++                          |     member)](api/languages/cp     |
|     func                          | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| tion)](api/languages/cpp_api.html | TSBEExecutionData12trajectoriesE) |
| #_CPPv4N5cudaq16ExecutionContext1 | -   [cudaq::ptsbe::PTSBEOptions   |
| 6ExecutionContextERKNSt6stringE), |     (C++                          |
|     [\[1\]](api/languages/        |     struc                         |
| cpp_api.html#_CPPv4N5cudaq16Execu | t)](api/languages/cpp_api.html#_C |
| tionContext16ExecutionContextERKN | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| St6stringENSt6size_tENSt6size_tE) | -   [cudaq::ptsbe::PTSB           |
| -   [cudaq::E                     | EOptions::include_sequential_data |
| xecutionContext::expectationValue |     (C++                          |
|     (C++                          |                                   |
|     member)](api/language         |    member)](api/languages/cpp_api |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| cutionContext16expectationValueE) | ptions23include_sequential_dataE) |
| -   [cudaq::Execu                 | -   [cudaq::ptsb                  |
| tionContext::explicitMeasurements | e::PTSBEOptions::max_trajectories |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     member)](api/languages/       |
| p_api.html#_CPPv4N5cudaq16Executi | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| onContext20explicitMeasurementsE) | 2PTSBEOptions16max_trajectoriesE) |
| -   [cuda                         | -   [cudaq::ptsbe::PT             |
| q::ExecutionContext::futureResult | SBEOptions::return_execution_data |
|     (C++                          |     (C++                          |
|     member)](api/lang             |     member)](api/languages/cpp_a  |
| uages/cpp_api.html#_CPPv4N5cudaq1 | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| 6ExecutionContext12futureResultE) | EOptions21return_execution_dataE) |
| -   [cudaq::ExecutionContext      | -   [cudaq::pts                   |
| ::hasConditionalsOnMeasureResults | be::PTSBEOptions::shot_allocation |
|     (C++                          |     (C++                          |
|     mem                           |     member)](api/languages        |
| ber)](api/languages/cpp_api.html# | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| _CPPv4N5cudaq16ExecutionContext31 | 12PTSBEOptions15shot_allocationE) |
| hasConditionalsOnMeasureResultsE) | -   [cud                          |
| -   [cudaq::Executi               | aq::ptsbe::PTSBEOptions::strategy |
| onContext::invocationResultBuffer |     (C++                          |
|     (C++                          |     member)](api/l                |
|     member)](api/languages/cpp_   | anguages/cpp_api.html#_CPPv4N5cud |
| api.html#_CPPv4N5cudaq16Execution | aq5ptsbe12PTSBEOptions8strategyE) |
| Context22invocationResultBufferE) | -   [cudaq::ptsbe::PTSBETrace     |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::kernelName |     t                             |
|     (C++                          | ype)](api/languages/cpp_api.html# |
|     member)](api/la               | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [                             |
| q16ExecutionContext10kernelNameE) | cudaq::ptsbe::PTSSamplingStrategy |
| -   [cud                          |     (C++                          |
| aq::ExecutionContext::kernelTrace |     class)](api                   |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/lan              | udaq5ptsbe19PTSSamplingStrategyE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::                      |
| 16ExecutionContext11kernelTraceE) | ptsbe::PTSSamplingStrategy::clone |
| -   [cudaq:                       |     (C++                          |
| :ExecutionContext::msm_dimensions |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4NK5cudaq5pt |
|     member)](api/langua           | sbe19PTSSamplingStrategy5cloneEv) |
| ges/cpp_api.html#_CPPv4N5cudaq16E | -   [cudaq::ptsbe::PTSSampl       |
| xecutionContext14msm_dimensionsE) | ingStrategy::generateTrajectories |
| -   [cudaq::                      |     (C++                          |
| ExecutionContext::msm_prob_err_id |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4NK5c |
|     member)](api/languag          | udaq5ptsbe19PTSSamplingStrategy20 |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | generateTrajectoriesENSt4spanIKN6 |
| ecutionContext15msm_prob_err_idE) | detail10NoisePointEEENSt6size_tE) |
| -   [cudaq::Ex                    | -   [cudaq:                       |
| ecutionContext::msm_probabilities | :ptsbe::PTSSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)](api/langua         |
| /cpp_api.html#_CPPv4N5cudaq16Exec | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| utionContext17msm_probabilitiesE) | tsbe19PTSSamplingStrategy4nameEv) |
| -                                 | -   [cudaq::ptsbe::PTSSampli      |
|    [cudaq::ExecutionContext::name | ngStrategy::\~PTSSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)]                      |     function)](api/la             |
| (api/languages/cpp_api.html#_CPPv | nguages/cpp_api.html#_CPPv4N5cuda |
| 4N5cudaq16ExecutionContext4nameE) | q5ptsbe19PTSSamplingStrategyD0Ev) |
| -   [cu                           | -   [cudaq::ptsbe::sample (C++    |
| daq::ExecutionContext::noiseModel |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     member)](api/la               | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| nguages/cpp_api.html#_CPPv4N5cuda | mpleE13sample_resultRK14sample_op |
| q16ExecutionContext10noiseModelE) | tionsRR13QuantumKernelDpRR4Args), |
| -   [cudaq::Exe                   |     [\[1\]](api                   |
| cutionContext::numberTrajectories | /languages/cpp_api.html#_CPPv4I0D |
|     (C++                          | pEN5cudaq5ptsbe6sampleE13sample_r |
|     member)](api/languages/       | esultRKN5cudaq11noise_modelENSt6s |
| cpp_api.html#_CPPv4N5cudaq16Execu | ize_tERR13QuantumKernelDpRR4Args) |
| tionContext18numberTrajectoriesE) | -   [cudaq::ptsbe::sample_async   |
| -   [c                            |     (C++                          |
| udaq::ExecutionContext::optResult |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4I |
|     member)](api/                 | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| languages/cpp_api.html#_CPPv4N5cu | 9async_sample_resultRK14sample_op |
| daq16ExecutionContext9optResultE) | tionsRR13QuantumKernelDpRR4Args), |
| -   [cudaq::Execu                 |     [\[1\]](api/languages/cp      |
| tionContext::overlapComputeStates | p_api.html#_CPPv4I0DpEN5cudaq5pts |
|     (C++                          | be12sample_asyncE19async_sample_r |
|     member)](api/languages/cp     | esultRKN5cudaq11noise_modelENSt6s |
| p_api.html#_CPPv4N5cudaq16Executi | ize_tERR13QuantumKernelDpRR4Args) |
| onContext20overlapComputeStatesE) | -   [cudaq::ptsbe::sample_options |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::overlapResult |     struct)                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/langu            | v4N5cudaq5ptsbe14sample_optionsE) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [cudaq::ptsbe::sample_result  |
| ExecutionContext13overlapResultE) |     (C++                          |
| -                                 |     class                         |
|   [cudaq::ExecutionContext::qpuId | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq5ptsbe13sample_resultE) |
|     member)](                     | -   [cudaq::pts                   |
| api/languages/cpp_api.html#_CPPv4 | be::sample_result::execution_data |
| N5cudaq16ExecutionContext5qpuIdE) |     (C++                          |
| -   [cudaq                        |     function)](api/languages/c    |
| ::ExecutionContext::registerNames | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
|     (C++                          | 3sample_result14execution_dataEv) |
|     member)](api/langu            | -   [cudaq::ptsbe::               |
| ages/cpp_api.html#_CPPv4N5cudaq16 | sample_result::has_execution_data |
| ExecutionContext13registerNamesE) |     (C++                          |
| -   [cu                           |                                   |
| daq::ExecutionContext::reorderIdx |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
|     member)](api/la               | ple_result18has_execution_dataEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::pt                    |
| q16ExecutionContext10reorderIdxE) | sbe::sample_result::sample_result |
| -                                 |     (C++                          |
|  [cudaq::ExecutionContext::result |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     member)](a                    | aq5ptsbe13sample_result13sample_r |
| pi/languages/cpp_api.html#_CPPv4N | esultERRN5cudaq13sample_resultE), |
| 5cudaq16ExecutionContext6resultE) |                                   |
| -                                 |  [\[1\]](api/languages/cpp_api.ht |
|   [cudaq::ExecutionContext::shots | ml#_CPPv4N5cudaq5ptsbe13sample_re |
|     (C++                          | sult13sample_resultERRN5cudaq13sa |
|     member)](                     | mple_resultE18PTSBEExecutionData) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::ptsbe::               |
| N5cudaq16ExecutionContext5shotsE) | sample_result::set_execution_data |
| -   [cudaq::                      |     (C++                          |
| ExecutionContext::simulationState |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     member)](api/languag          | daq5ptsbe13sample_result18set_exe |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | cution_dataE18PTSBEExecutionData) |
| ecutionContext15simulationStateE) | -   [cud                          |
| -                                 | aq::ptsbe::ShotAllocationStrategy |
|    [cudaq::ExecutionContext::spin |     (C++                          |
|     (C++                          |     struct)](using                |
|     member)]                      | /examples/ptsbe.html#_CPPv4N5cuda |
| (api/languages/cpp_api.html#_CPPv | q5ptsbe22ShotAllocationStrategyE) |
| 4N5cudaq16ExecutionContext4spinE) | -   [cudaq::ptsbe::ShotAllocatio  |
| -   [cudaq::                      | nStrategy::ShotAllocationStrategy |
| ExecutionContext::totalIterations |     (C++                          |
|     (C++                          |     function)                     |
|     member)](api/languag          | ](using/examples/ptsbe.html#_CPPv |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | 4N5cudaq5ptsbe22ShotAllocationStr |
| ecutionContext15totalIterationsE) | ategy22ShotAllocationStrategyE4Ty |
| -   [cudaq::Executio              | pedNSt8optionalINSt8uint64_tEEE), |
| nContext::warnedNamedMeasurements |     [\[1\                         |
|     (C++                          | ]](using/examples/ptsbe.html#_CPP |
|     member)](api/languages/cpp_a  | v4N5cudaq5ptsbe22ShotAllocationSt |
| pi.html#_CPPv4N5cudaq16ExecutionC | rategy22ShotAllocationStrategyEv) |
| ontext23warnedNamedMeasurementsE) | -   [cudaq::pt                    |
| -   [cudaq::ExecutionResult (C++  | sbe::ShotAllocationStrategy::Type |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |     enum)](using/exam             |
| #_CPPv4N5cudaq15ExecutionResultE) | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| -   [cud                          | be22ShotAllocationStrategy4TypeE) |
| aq::ExecutionResult::appendResult | -   [cudaq::ptsbe::ShotAllocatio  |
|     (C++                          | nStrategy::Type::HIGH_WEIGHT_BIAS |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |     enumerat                      |
| PPv4N5cudaq15ExecutionResult12app | or)](using/examples/ptsbe.html#_C |
| endResultENSt6stringENSt6size_tE) | PPv4N5cudaq5ptsbe22ShotAllocation |
| -   [cu                           | Strategy4Type16HIGH_WEIGHT_BIASE) |
| daq::ExecutionResult::deserialize | -   [cudaq::ptsbe::ShotAllocati   |
|     (C++                          | onStrategy::Type::LOW_WEIGHT_BIAS |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     enumera                       |
| v4N5cudaq15ExecutionResult11deser | tor)](using/examples/ptsbe.html#_ |
| ializeERNSt6vectorINSt6size_tEEE) | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| -   [cudaq:                       | nStrategy4Type15LOW_WEIGHT_BIASE) |
| :ExecutionResult::ExecutionResult | -   [cudaq::ptsbe::ShotAlloc      |
|     (C++                          | ationStrategy::Type::PROPORTIONAL |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |     enum                          |
| PPv4N5cudaq15ExecutionResult15Exe | erator)](using/examples/ptsbe.htm |
| cutionResultE16CountsDictionary), | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
|     [\[1\]](api/lan               | tionStrategy4Type12PROPORTIONALE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::ptsbe::Shot           |
| 15ExecutionResult15ExecutionResul | AllocationStrategy::Type::UNIFORM |
| tE16CountsDictionaryNSt6stringE), |     (C++                          |
|     [\[2\                         |                                   |
| ]](api/languages/cpp_api.html#_CP |   enumerator)](using/examples/pts |
| Pv4N5cudaq15ExecutionResult15Exec | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| utionResultE16CountsDictionaryd), | AllocationStrategy4Type7UNIFORME) |
|                                   | -                                 |
|    [\[3\]](api/languages/cpp_api. |   [cudaq::ptsbe::TraceInstruction |
| html#_CPPv4N5cudaq15ExecutionResu |     (C++                          |
| lt15ExecutionResultENSt6stringE), |     struct)](                     |
|     [\[4\                         | api/languages/cpp_api.html#_CPPv4 |
| ]](api/languages/cpp_api.html#_CP | N5cudaq5ptsbe16TraceInstructionE) |
| Pv4N5cudaq15ExecutionResult15Exec | -   [cudaq:                       |
| utionResultERK15ExecutionResult), | :ptsbe::TraceInstruction::channel |
|     [\[5\]](api/language          |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     member)](api/lang             |
| cutionResult15ExecutionResultEd), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     [\[6\]](api/languag           | ptsbe16TraceInstruction7channelE) |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | -   [cudaq::                      |
| ecutionResult15ExecutionResultEv) | ptsbe::TraceInstruction::controls |
| -   [                             |     (C++                          |
| cudaq::ExecutionResult::operator= |     member)](api/langu            |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq5p |
|     function)](api/languages/     | tsbe16TraceInstruction8controlsE) |
| cpp_api.html#_CPPv4N5cudaq15Execu | -   [cud                          |
| tionResultaSERK15ExecutionResult) | aq::ptsbe::TraceInstruction::name |
| -   [c                            |     (C++                          |
| udaq::ExecutionResult::operator== |     member)](api/l                |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     function)](api/languages/c    | aq5ptsbe16TraceInstruction4nameE) |
| pp_api.html#_CPPv4NK5cudaq15Execu | -   [cudaq                        |
| tionResulteqERK15ExecutionResult) | ::ptsbe::TraceInstruction::params |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::registerName |     member)](api/lan              |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     member)](api/lan              | 5ptsbe16TraceInstruction6paramsE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq:                       |
| 15ExecutionResult12registerNameE) | :ptsbe::TraceInstruction::targets |
| -   [cudaq                        |     (C++                          |
| ::ExecutionResult::sequentialData |     member)](api/lang             |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     member)](api/langu            | ptsbe16TraceInstruction7targetsE) |
| ages/cpp_api.html#_CPPv4N5cudaq15 | -   [cudaq::ptsbe::T              |
| ExecutionResult14sequentialDataE) | raceInstruction::TraceInstruction |
| -   [                             |     (C++                          |
| cudaq::ExecutionResult::serialize |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)](api/l              | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| anguages/cpp_api.html#_CPPv4NK5cu | Instruction16TraceInstructionE20T |
| daq15ExecutionResult9serializeEv) | raceInstructionTypeNSt6stringENSt |
| -   [cudaq::fermion_handler (C++  | 6vectorINSt6size_tEEENSt6vectorIN |
|     c                             | St6size_tEEENSt6vectorIdEENSt8opt |
| lass)](api/languages/cpp_api.html | ionalIN5cudaq13kraus_channelEEE), |
| #_CPPv4N5cudaq15fermion_handlerE) |     [\[1\]](api/languages/cpp_a   |
| -   [cudaq::fermion_op (C++       | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
|     type)](api/languages/cpp_api  | eInstruction16TraceInstructionEv) |
| .html#_CPPv4N5cudaq10fermion_opE) | -   [cud                          |
| -   [cudaq::fermion_op_term (C++  | aq::ptsbe::TraceInstruction::type |
|                                   |     (C++                          |
| type)](api/languages/cpp_api.html |     member)](api/l                |
| #_CPPv4N5cudaq15fermion_op_termE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::FermioniqQPU (C++     | aq5ptsbe16TraceInstruction4typeE) |
|                                   | -   [c                            |
|   class)](api/languages/cpp_api.h | udaq::ptsbe::TraceInstructionType |
| tml#_CPPv4N5cudaq12FermioniqQPUE) |     (C++                          |
| -   [cudaq::get_state (C++        |     enum)](api/                   |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|    function)](api/languages/cpp_a | daq5ptsbe20TraceInstructionTypeE) |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | -   [cudaq::                      |
| ateEDaRR13QuantumKernelDpRR4Args) | ptsbe::TraceInstructionType::Gate |
| -   [cudaq::gradient (C++         |     (C++                          |
|     class)](api/languages/cpp_    |     enumerator)](api/langu        |
| api.html#_CPPv4N5cudaq8gradientE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [cudaq::gradient::clone (C++  | tsbe20TraceInstructionType4GateE) |
|     fun                           | -   [cudaq::ptsbe::               |
| ction)](api/languages/cpp_api.htm | TraceInstructionType::Measurement |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     (C++                          |
| -   [cudaq::gradient::compute     |                                   |
|     (C++                          |    enumerator)](api/languages/cpp |
|     function)](api/language       | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| s/cpp_api.html#_CPPv4N5cudaq8grad | aceInstructionType11MeasurementE) |
| ient7computeERKNSt6vectorIdEERKNS | -   [cudaq::p                     |
| t8functionIFdNSt6vectorIdEEEEEd), | tsbe::TraceInstructionType::Noise |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     enumerator)](api/langua       |
| cudaq8gradient7computeERKNSt6vect | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| orIdEERNSt6vectorIdEERK7spin_opd) | sbe20TraceInstructionType5NoiseE) |
| -   [cudaq::gradient::gradient    | -   [                             |
|     (C++                          | cudaq::ptsbe::TrajectoryPredicate |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4I00EN5cu |     type)](api                    |
| daq8gradient8gradientER7KernelT), | /languages/cpp_api.html#_CPPv4N5c |
|                                   | udaq5ptsbe19TrajectoryPredicateE) |
|    [\[1\]](api/languages/cpp_api. | -   [cudaq::QPU (C++              |
| html#_CPPv4I00EN5cudaq8gradient8g |     class)](api/languages         |
| radientER7KernelTRR10ArgsMapper), | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     [\[2\                         | -   [cudaq::QPU::beginExecution   |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4I00EN5cudaq8gradient8gradientE |     function                      |
| RR13QuantumKernelRR10ArgsMapper), | )](api/languages/cpp_api.html#_CP |
|     [\[3                          | Pv4N5cudaq3QPU14beginExecutionEv) |
| \]](api/languages/cpp_api.html#_C | -   [cuda                         |
| PPv4N5cudaq8gradient8gradientERRN | q::QPU::configureExecutionContext |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[                           |     funct                         |
| 4\]](api/languages/cpp_api.html#_ | ion)](api/languages/cpp_api.html# |
| CPPv4N5cudaq8gradient8gradientEv) | _CPPv4NK5cudaq3QPU25configureExec |
| -   [cudaq::gradient::setArgs     | utionContextER16ExecutionContext) |
|     (C++                          | -   [cudaq::QPU::endExecution     |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     functi                        |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | on)](api/languages/cpp_api.html#_ |
| tArgsEvR13QuantumKernelDpRR4Args) | CPPv4N5cudaq3QPU12endExecutionEv) |
| -   [cudaq::gradient::setKernel   | -   [cudaq::QPU::enqueue (C++     |
|     (C++                          |     function)](ap                 |
|     function)](api/languages/c    | i/languages/cpp_api.html#_CPPv4N5 |
| pp_api.html#_CPPv4I0EN5cudaq8grad | cudaq3QPU7enqueueER11QuantumTask) |
| ient9setKernelEvR13QuantumKernel) | -   [cud                          |
| -   [cud                          | aq::QPU::finalizeExecutionContext |
| aq::gradients::central_difference |     (C++                          |
|     (C++                          |     func                          |
|     class)](api/la                | tion)](api/languages/cpp_api.html |
| nguages/cpp_api.html#_CPPv4N5cuda | #_CPPv4NK5cudaq3QPU24finalizeExec |
| q9gradients18central_differenceE) | utionContextER16ExecutionContext) |
| -   [cudaq::gra                   | -   [cudaq::QPU::getConnectivity  |
| dients::central_difference::clone |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/languages      | ](api/languages/cpp_api.html#_CPP |
| /cpp_api.html#_CPPv4N5cudaq9gradi | v4N5cudaq3QPU15getConnectivityEv) |
| ents18central_difference5cloneEv) | -                                 |
| -   [cudaq::gradi                 | [cudaq::QPU::getExecutionThreadId |
| ents::central_difference::compute |     (C++                          |
|     (C++                          |     function)](api/               |
|     function)](                   | languages/cpp_api.html#_CPPv4NK5c |
| api/languages/cpp_api.html#_CPPv4 | udaq3QPU20getExecutionThreadIdEv) |
| N5cudaq9gradients18central_differ | -   [cudaq::QPU::getNumQubits     |
| ence7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|   [\[1\]](api/languages/cpp_api.h | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| tml#_CPPv4N5cudaq9gradients18cent | -   [                             |
| ral_difference7computeERKNSt6vect | cudaq::QPU::getRemoteCapabilities |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradie                |     function)](api/l              |
| nts::central_difference::gradient | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq3QPU21getRemoteCapabilitiesEv) |
|     functio                       | -   [cudaq::QPU::isEmulated (C++  |
| n)](api/languages/cpp_api.html#_C |     func                          |
| PPv4I00EN5cudaq9gradients18centra | tion)](api/languages/cpp_api.html |
| l_difference8gradientER7KernelT), | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     [\[1\]](api/langua            | -   [cudaq::QPU::isSimulator (C++ |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     funct                         |
| q9gradients18central_difference8g | ion)](api/languages/cpp_api.html# |
| radientER7KernelTRR10ArgsMapper), | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::QPU::launchKernel     |
| api.html#_CPPv4I00EN5cudaq9gradie |     (C++                          |
| nts18central_difference8gradientE |     function)](api                |
| RR13QuantumKernelRR10ArgsMapper), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[3\]](api/languages/cpp     | udaq3QPU12launchKernelERKNSt6stri |
| _api.html#_CPPv4N5cudaq9gradients | ngE15KernelThunkTypePvNSt8uint64_ |
| 18central_difference8gradientERRN | tENSt8uint64_tERKNSt6vectorIPvEE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QPU::onRandomSeedSet  |
|     [\[4\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq9gradient |     function)](api/lang           |
| s18central_difference8gradientEv) | uages/cpp_api.html#_CPPv4N5cudaq3 |
| -   [cud                          | QPU15onRandomSeedSetENSt6size_tE) |
| aq::gradients::forward_difference | -   [cudaq::QPU::QPU (C++         |
|     (C++                          |     functio                       |
|     class)](api/la                | n)](api/languages/cpp_api.html#_C |
| nguages/cpp_api.html#_CPPv4N5cuda | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| q9gradients18forward_differenceE) |                                   |
| -   [cudaq::gra                   |  [\[1\]](api/languages/cpp_api.ht |
| dients::forward_difference::clone | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     (C++                          |     [\[2\]](api/languages/cpp_    |
|     function)](api/languages      | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QPU::setId (C++       |
| ents18forward_difference5cloneEv) |     function                      |
| -   [cudaq::gradi                 | )](api/languages/cpp_api.html#_CP |
| ents::forward_difference::compute | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::setShots (C++    |
|     function)](                   |     f                             |
| api/languages/cpp_api.html#_CPPv4 | unction)](api/languages/cpp_api.h |
| N5cudaq9gradients18forward_differ | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| ence7computeERKNSt6vectorIdEERKNS | -   [cudaq::                      |
| t8functionIFdNSt6vectorIdEEEEEd), | QPU::supportsExplicitMeasurements |
|                                   |     (C++                          |
|   [\[1\]](api/languages/cpp_api.h |     function)](api/languag        |
| tml#_CPPv4N5cudaq9gradients18forw | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| ard_difference7computeERKNSt6vect | 28supportsExplicitMeasurementsEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QPU::\~QPU (C++       |
| -   [cudaq::gradie                |     function)](api/languages/cp   |
| nts::forward_difference::gradient | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     (C++                          | -   [cudaq::QPUState (C++         |
|     functio                       |     class)](api/languages/cpp_    |
| n)](api/languages/cpp_api.html#_C | api.html#_CPPv4N5cudaq8QPUStateE) |
| PPv4I00EN5cudaq9gradients18forwar | -   [cudaq::qreg (C++             |
| d_difference8gradientER7KernelT), |     class)](api/lan               |
|     [\[1\]](api/langua            | guages/cpp_api.html#_CPPv4I_NSt6s |
| ges/cpp_api.html#_CPPv4I00EN5cuda | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| q9gradients18forward_difference8g | -   [cudaq::qreg::back (C++       |
| radientER7KernelTRR10ArgsMapper), |     function)                     |
|     [\[2\]](api/languages/cpp_    | ](api/languages/cpp_api.html#_CPP |
| api.html#_CPPv4I00EN5cudaq9gradie | v4N5cudaq4qreg4backENSt6size_tE), |
| nts18forward_difference8gradientE |     [\[1\]](api/languages/cpp_ap  |
| RR13QuantumKernelRR10ArgsMapper), | i.html#_CPPv4N5cudaq4qreg4backEv) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::qreg::begin (C++      |
| _api.html#_CPPv4N5cudaq9gradients |                                   |
| 18forward_difference8gradientERRN |  function)](api/languages/cpp_api |
| St8functionIFvNSt6vectorIdEEEEE), | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qreg::clear (C++      |
| p_api.html#_CPPv4N5cudaq9gradient |                                   |
| s18forward_difference8gradientEv) |  function)](api/languages/cpp_api |
| -   [                             | .html#_CPPv4N5cudaq4qreg5clearEv) |
| cudaq::gradients::parameter_shift | -   [cudaq::qreg::front (C++      |
|     (C++                          |     function)]                    |
|     class)](api                   | (api/languages/cpp_api.html#_CPPv |
| /languages/cpp_api.html#_CPPv4N5c | 4N5cudaq4qreg5frontENSt6size_tE), |
| udaq9gradients15parameter_shiftE) |     [\[1\]](api/languages/cpp_api |
| -   [cudaq::                      | .html#_CPPv4N5cudaq4qreg5frontEv) |
| gradients::parameter_shift::clone | -   [cudaq::qreg::operator\[\]    |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     functi                        |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | on)](api/languages/cpp_api.html#_ |
| adients15parameter_shift5cloneEv) | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| -   [cudaq::gr                    | -   [cudaq::qreg::qreg (C++       |
| adients::parameter_shift::compute |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function                      | v4N5cudaq4qreg4qregENSt6size_tE), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/languages/cpp_ap  |
| Pv4N5cudaq9gradients15parameter_s | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| hift7computeERKNSt6vectorIdEERKNS | -   [cudaq::qreg::size (C++       |
| t8functionIFdNSt6vectorIdEEEEEd), |                                   |
|     [\[1\]](api/languages/cpp_ap  |  function)](api/languages/cpp_api |
| i.html#_CPPv4N5cudaq9gradients15p | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| arameter_shift7computeERKNSt6vect | -   [cudaq::qreg::slice (C++      |
| orIdEERNSt6vectorIdEERK7spin_opd) |     function)](api/langu          |
| -   [cudaq::gra                   | ages/cpp_api.html#_CPPv4N5cudaq4q |
| dients::parameter_shift::gradient | reg5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qreg::value_type (C++ |
|     func                          |                                   |
| tion)](api/languages/cpp_api.html | type)](api/languages/cpp_api.html |
| #_CPPv4I00EN5cudaq9gradients15par | #_CPPv4N5cudaq4qreg10value_typeE) |
| ameter_shift8gradientER7KernelT), | -   [cudaq::qspan (C++            |
|     [\[1\]](api/lan               |     class)](api/lang              |
| guages/cpp_api.html#_CPPv4I00EN5c | uages/cpp_api.html#_CPPv4I_NSt6si |
| udaq9gradients15parameter_shift8g | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::QuakeValue (C++       |
|     [\[2\]](api/languages/c       |     class)](api/languages/cpp_api |
| pp_api.html#_CPPv4I00EN5cudaq9gra | .html#_CPPv4N5cudaq10QuakeValueE) |
| dients15parameter_shift8gradientE | -   [cudaq::Q                     |
| RR13QuantumKernelRR10ArgsMapper), | uakeValue::canValidateNumElements |
|     [\[3\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq9gradie |     function)](api/languages      |
| nts15parameter_shift8gradientERRN | /cpp_api.html#_CPPv4N5cudaq10Quak |
| St8functionIFvNSt6vectorIdEEEEE), | eValue22canValidateNumElementsEv) |
|     [\[4\]](api/languages         | -                                 |
| /cpp_api.html#_CPPv4N5cudaq9gradi |  [cudaq::QuakeValue::constantSize |
| ents15parameter_shift8gradientEv) |     (C++                          |
| -   [cudaq::kernel_builder (C++   |     function)](api                |
|     clas                          | /languages/cpp_api.html#_CPPv4N5c |
| s)](api/languages/cpp_api.html#_C | udaq10QuakeValue12constantSizeEv) |
| PPv4IDpEN5cudaq14kernel_builderE) | -   [cudaq::QuakeValue::dump (C++ |
| -   [c                            |     function)](api/lan            |
| udaq::kernel_builder::constantVal | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 10QuakeValue4dumpERNSt7ostreamE), |
|     function)](api/la             |     [\                            |
| nguages/cpp_api.html#_CPPv4N5cuda | [1\]](api/languages/cpp_api.html# |
| q14kernel_builder11constantValEd) | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| -   [cu                           | -   [cudaq                        |
| daq::kernel_builder::getArguments | ::QuakeValue::getRequiredElements |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)](api/langua         |
| guages/cpp_api.html#_CPPv4N5cudaq | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| 14kernel_builder12getArgumentsEv) | uakeValue19getRequiredElementsEv) |
| -   [cu                           | -   [cudaq::QuakeValue::getValue  |
| daq::kernel_builder::getNumParams |     (C++                          |
|     (C++                          |     function)]                    |
|     function)](api/lan            | (api/languages/cpp_api.html#_CPPv |
| guages/cpp_api.html#_CPPv4N5cudaq | 4NK5cudaq10QuakeValue8getValueEv) |
| 14kernel_builder12getNumParamsEv) | -   [cudaq::QuakeValue::inverse   |
| -   [c                            |     (C++                          |
| udaq::kernel_builder::isArgStdVec |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/languages/cp   | v4NK5cudaq10QuakeValue7inverseEv) |
| p_api.html#_CPPv4N5cudaq14kernel_ | -   [cudaq::QuakeValue::isStdVec  |
| builder11isArgStdVecENSt6size_tE) |     (C++                          |
| -   [cuda                         |     function)                     |
| q::kernel_builder::kernel_builder | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq10QuakeValue8isStdVecEv) |
|     function)](api/languages/cpp_ | -                                 |
| api.html#_CPPv4N5cudaq14kernel_bu |    [cudaq::QuakeValue::operator\* |
| ilder14kernel_builderERNSt6vector |     (C++                          |
| IN7details17KernelBuilderTypeEEE) |     function)](api                |
| -   [cudaq::kernel_builder::name  | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuemlE10QuakeValue), |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP | [\[1\]](api/languages/cpp_api.htm |
| v4N5cudaq14kernel_builder4nameEv) | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| -                                 | -   [cudaq::QuakeValue::operator+ |
|    [cudaq::kernel_builder::qalloc |     (C++                          |
|     (C++                          |     function)](api                |
|     function)](api/language       | /languages/cpp_api.html#_CPPv4N5c |
| s/cpp_api.html#_CPPv4N5cudaq14ker | udaq10QuakeValueplE10QuakeValue), |
| nel_builder6qallocE10QuakeValue), |     [                             |
|     [\[1\]](api/language          | \[1\]](api/languages/cpp_api.html |
| s/cpp_api.html#_CPPv4N5cudaq14ker | #_CPPv4N5cudaq10QuakeValueplEKd), |
| nel_builder6qallocEKNSt6size_tE), |                                   |
|     [\[2                          | [\[2\]](api/languages/cpp_api.htm |
| \]](api/languages/cpp_api.html#_C | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| PPv4N5cudaq14kernel_builder6qallo | -   [cudaq::QuakeValue::operator- |
| cERNSt6vectorINSt7complexIdEEEE), |     (C++                          |
|     [\[3\]](                      |     function)](api                |
| api/languages/cpp_api.html#_CPPv4 | /languages/cpp_api.html#_CPPv4N5c |
| N5cudaq14kernel_builder6qallocEv) | udaq10QuakeValuemiE10QuakeValue), |
| -   [cudaq::kernel_builder::swap  |     [                             |
|     (C++                          | \[1\]](api/languages/cpp_api.html |
|     function)](api/language       | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     [                             |
| 4kernel_builder4swapEvRK10QuakeVa | \[2\]](api/languages/cpp_api.html |
| lueRK10QuakeValueRK10QuakeValue), | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|                                   |                                   |
| [\[1\]](api/languages/cpp_api.htm | [\[3\]](api/languages/cpp_api.htm |
| l#_CPPv4I00EN5cudaq14kernel_build | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| er4swapEvRKNSt6vectorI10QuakeValu | -   [cudaq::QuakeValue::operator/ |
| eEERK10QuakeValueRK10QuakeValue), |     (C++                          |
|                                   |     function)](api                |
| [\[2\]](api/languages/cpp_api.htm | /languages/cpp_api.html#_CPPv4N5c |
| l#_CPPv4N5cudaq14kernel_builder4s | udaq10QuakeValuedvE10QuakeValue), |
| wapERK10QuakeValueRK10QuakeValue) |                                   |
| -   [cudaq::KernelExecutionTask   | [\[1\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
|     type                          | -                                 |
| )](api/languages/cpp_api.html#_CP |  [cudaq::QuakeValue::operator\[\] |
| Pv4N5cudaq19KernelExecutionTaskE) |     (C++                          |
| -   [cudaq::KernelThunkResultType |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     struct)]                      | udaq10QuakeValueixEKNSt6size_tE), |
| (api/languages/cpp_api.html#_CPPv |     [\[1\]](api/                  |
| 4N5cudaq21KernelThunkResultTypeE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::KernelThunkType (C++  | daq10QuakeValueixERK10QuakeValue) |
|                                   | -                                 |
| type)](api/languages/cpp_api.html |    [cudaq::QuakeValue::QuakeValue |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     (C++                          |
| -   [cudaq::kraus_channel (C++    |     function)](api/languag        |
|                                   | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|  class)](api/languages/cpp_api.ht | akeValue10QuakeValueERN4mlir20Imp |
| ml#_CPPv4N5cudaq13kraus_channelE) | licitLocOpBuilderEN4mlir5ValueE), |
| -   [cudaq::kraus_channel::empty  |     [\[1\]                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)]                    | v4N5cudaq10QuakeValue10QuakeValue |
| (api/languages/cpp_api.html#_CPPv | ERN4mlir20ImplicitLocOpBuilderEd) |
| 4NK5cudaq13kraus_channel5emptyEv) | -   [cudaq::QuakeValue::size (C++ |
| -   [cudaq::kraus_c               |     funct                         |
| hannel::generateUnitaryParameters | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|                                   | -   [cudaq::QuakeValue::slice     |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4N5cudaq13kraus_chan |     function)](api/languages/cpp_ |
| nel25generateUnitaryParametersEv) | api.html#_CPPv4N5cudaq10QuakeValu |
| -                                 | e5sliceEKNSt6size_tEKNSt6size_tE) |
|    [cudaq::kraus_channel::get_ops | -   [cudaq::quantum_platform (C++ |
|     (C++                          |     cl                            |
|     function)](a                  | ass)](api/languages/cpp_api.html# |
| pi/languages/cpp_api.html#_CPPv4N | _CPPv4N5cudaq16quantum_platformE) |
| K5cudaq13kraus_channel7get_opsEv) | -   [cudaq:                       |
| -   [cud                          | :quantum_platform::beginExecution |
| aq::kraus_channel::identity_flags |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     member)](api/lan              | es/cpp_api.html#_CPPv4N5cudaq16qu |
| guages/cpp_api.html#_CPPv4N5cudaq | antum_platform14beginExecutionEv) |
| 13kraus_channel14identity_flagsE) | -   [cudaq::quantum_pl            |
| -   [cud                          | atform::configureExecutionContext |
| aq::kraus_channel::is_identity_op |     (C++                          |
|     (C++                          |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|    function)](api/languages/cpp_a | 16quantum_platform25configureExec |
| pi.html#_CPPv4NK5cudaq13kraus_cha | utionContextER16ExecutionContext) |
| nnel14is_identity_opENSt6size_tE) | -   [cuda                         |
| -   [cudaq::                      | q::quantum_platform::connectivity |
| kraus_channel::is_unitary_mixture |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](api/languages      | ages/cpp_api.html#_CPPv4N5cudaq16 |
| /cpp_api.html#_CPPv4NK5cudaq13kra | quantum_platform12connectivityEv) |
| us_channel18is_unitary_mixtureEv) | -   [cuda                         |
| -   [cu                           | q::quantum_platform::endExecution |
| daq::kraus_channel::kraus_channel |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](api/lang           | ages/cpp_api.html#_CPPv4N5cudaq16 |
| uages/cpp_api.html#_CPPv4IDpEN5cu | quantum_platform12endExecutionEv) |
| daq13kraus_channel13kraus_channel | -   [cudaq::q                     |
| EDpRRNSt16initializer_listI1TEE), | uantum_platform::enqueueAsyncTask |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     function)](api/languages/     |
| ml#_CPPv4N5cudaq13kraus_channel13 | cpp_api.html#_CPPv4N5cudaq16quant |
| kraus_channelERK13kraus_channel), | um_platform16enqueueAsyncTaskEKNS |
|     [\[2\]                        | t6size_tER19KernelExecutionTask), |
| ](api/languages/cpp_api.html#_CPP |     [\[1\]](api/languag           |
| v4N5cudaq13kraus_channel13kraus_c | es/cpp_api.html#_CPPv4N5cudaq16qu |
| hannelERKNSt6vectorI8kraus_opEE), | antum_platform16enqueueAsyncTaskE |
|     [\[3\]                        | KNSt6size_tERNSt8functionIFvvEEE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::quantum_p             |
| v4N5cudaq13kraus_channel13kraus_c | latform::finalizeExecutionContext |
| hannelERRNSt6vectorI8kraus_opEE), |     (C++                          |
|     [\[4\]](api/lan               |     function)](api/languages/c    |
| guages/cpp_api.html#_CPPv4N5cudaq | pp_api.html#_CPPv4NK5cudaq16quant |
| 13kraus_channel13kraus_channelEv) | um_platform24finalizeExecutionCon |
| -                                 | textERN5cudaq16ExecutionContextE) |
| [cudaq::kraus_channel::noise_type | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_codegen_config |
|     member)](api                  |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)](api/languages/c    |
| udaq13kraus_channel10noise_typeE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -                                 | m_platform18get_codegen_configEv) |
|   [cudaq::kraus_channel::op_names | -   [cuda                         |
|     (C++                          | q::quantum_platform::get_exec_ctx |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/langua         |
| N5cudaq13kraus_channel8op_namesE) | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| -                                 | quantum_platform12get_exec_ctxEv) |
|  [cudaq::kraus_channel::operator= | -   [c                            |
|     (C++                          | udaq::quantum_platform::get_noise |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     function)](api/languages/c    |
| raus_channelaSERK13kraus_channel) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [c                            | m_platform9get_noiseENSt6size_tE) |
| udaq::kraus_channel::operator\[\] | -   [cudaq:                       |
|     (C++                          | :quantum_platform::get_num_qubits |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4N5cud |                                   |
| aq13kraus_channelixEKNSt6size_tE) | function)](api/languages/cpp_api. |
| -                                 | html#_CPPv4NK5cudaq16quantum_plat |
| [cudaq::kraus_channel::parameters | form14get_num_qubitsENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_              |
|     member)](api                  | platform::get_remote_capabilities |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel10parametersE) |     function)                     |
| -   [cudaq::krau                  | ](api/languages/cpp_api.html#_CPP |
| s_channel::populateDefaultOpNames | v4NK5cudaq16quantum_platform23get |
|     (C++                          | _remote_capabilitiesENSt6size_tE) |
|     function)](api/languages/cp   | -   [cudaq::qua                   |
| p_api.html#_CPPv4N5cudaq13kraus_c | ntum_platform::get_runtime_target |
| hannel22populateDefaultOpNamesEv) |     (C++                          |
| -   [cu                           |     function)](api/languages/cp   |
| daq::kraus_channel::probabilities | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform18get_runtime_targetEv) |
|     member)](api/la               | -   [cuda                         |
| nguages/cpp_api.html#_CPPv4N5cuda | q::quantum_platform::getLogStream |
| q13kraus_channel13probabilitiesE) |     (C++                          |
| -                                 |     function)](api/langu          |
|  [cudaq::kraus_channel::push_back | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     (C++                          | quantum_platform12getLogStreamEv) |
|     function)](api                | -   [cud                          |
| /languages/cpp_api.html#_CPPv4N5c | aq::quantum_platform::is_emulated |
| udaq13kraus_channel9push_backE8kr |     (C++                          |
| aus_opNSt8optionalINSt6stringEEE) |                                   |
| -   [cudaq::kraus_channel::size   |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq16quantum_p |
|     function)                     | latform11is_emulatedENSt6size_tE) |
| ](api/languages/cpp_api.html#_CPP | -   [c                            |
| v4NK5cudaq13kraus_channel4sizeEv) | udaq::quantum_platform::is_remote |
| -   [                             |     (C++                          |
| cudaq::kraus_channel::unitary_ops |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4NK5cudaq16quantu |
|     member)](api/                 | m_platform9is_remoteENSt6size_tE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cuda                         |
| daq13kraus_channel11unitary_opsE) | q::quantum_platform::is_simulator |
| -   [cudaq::kraus_op (C++         |     (C++                          |
|     struct)](api/languages/cpp_   |                                   |
| api.html#_CPPv4N5cudaq8kraus_opE) |   function)](api/languages/cpp_ap |
| -   [cudaq::kraus_op::adjoint     | i.html#_CPPv4NK5cudaq16quantum_pl |
|     (C++                          | atform12is_simulatorENSt6size_tE) |
|     functi                        | -   [c                            |
| on)](api/languages/cpp_api.html#_ | udaq::quantum_platform::launchVQE |
| CPPv4NK5cudaq8kraus_op7adjointEv) |     (C++                          |
| -   [cudaq::kraus_op::data (C++   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|  member)](api/languages/cpp_api.h | N5cudaq16quantum_platform9launchV |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | QEEKNSt6stringEPKvPN5cudaq8gradie |
| -   [cudaq::kraus_op::kraus_op    | ntERKN5cudaq7spin_opERN5cudaq9opt |
|     (C++                          | imizerEKiKNSt6size_tENSt6size_tE) |
|     func                          | -   [cudaq:                       |
| tion)](api/languages/cpp_api.html | :quantum_platform::list_platforms |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |     (C++                          |
| opERRNSt16initializer_listI1TEE), |     function)](api/languag        |
|                                   | es/cpp_api.html#_CPPv4N5cudaq16qu |
|  [\[1\]](api/languages/cpp_api.ht | antum_platform14list_platformsEv) |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | -                                 |
| pENSt6vectorIN5cudaq7complexEEE), |    [cudaq::quantum_platform::name |
|     [\[2\]](api/l                 |     (C++                          |
| anguages/cpp_api.html#_CPPv4N5cud |     function)](a                  |
| aq8kraus_op8kraus_opERK8kraus_op) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::kraus_op::nCols (C++  | K5cudaq16quantum_platform4nameEv) |
|                                   | -   [                             |
| member)](api/languages/cpp_api.ht | cudaq::quantum_platform::num_qpus |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     (C++                          |
| -   [cudaq::kraus_op::nRows (C++  |     function)](api/l              |
|                                   | anguages/cpp_api.html#_CPPv4NK5cu |
| member)](api/languages/cpp_api.ht | daq16quantum_platform8num_qpusEv) |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | -   [cudaq::                      |
| -   [cudaq::kraus_op::operator=   | quantum_platform::onRandomSeedSet |
|     (C++                          |     (C++                          |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP | function)](api/languages/cpp_api. |
| v4N5cudaq8kraus_opaSERK8kraus_op) | html#_CPPv4N5cudaq16quantum_platf |
| -   [cudaq::kraus_op::precision   | orm15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     memb                          | :quantum_platform::reset_exec_ctx |
| er)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8kraus_op9precisionE) |     function)](api/languag        |
| -   [cudaq::KrausSelection (C++   | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     s                             | antum_platform14reset_exec_ctxEv) |
| truct)](api/languages/cpp_api.htm | -   [cud                          |
| l#_CPPv4N5cudaq14KrausSelectionE) | aq::quantum_platform::reset_noise |
| -   [cudaq:                       |     (C++                          |
| :KrausSelection::circuit_location |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq16quantum_p |
|     member)](api/langua           | latform11reset_noiseENSt6size_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq14K | -   [cudaq:                       |
| rausSelection16circuit_locationE) | :quantum_platform::resetLogStream |
| -                                 |     (C++                          |
|  [cudaq::KrausSelection::is_error |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     member)](a                    | antum_platform14resetLogStreamEv) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cuda                         |
| 5cudaq14KrausSelection8is_errorE) | q::quantum_platform::set_exec_ctx |
| -   [cudaq::Kra                   |     (C++                          |
| usSelection::kraus_operator_index |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     member)](api/languages/       | _CPPv4N5cudaq16quantum_platform12 |
| cpp_api.html#_CPPv4N5cudaq14Kraus | set_exec_ctxEP16ExecutionContext) |
| Selection20kraus_operator_indexE) | -   [c                            |
| -   [cuda                         | udaq::quantum_platform::set_noise |
| q::KrausSelection::KrausSelection |     (C++                          |
|     (C++                          |     function                      |
|     function)](a                  | )](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4N5cudaq16quantum_platform9set_ |
| 5cudaq14KrausSelection14KrausSele | noiseEPK11noise_modelNSt6size_tE) |
| ctionENSt6size_tENSt6vectorINSt6s | -   [cuda                         |
| ize_tEEENSt6stringENSt6size_tEb), | q::quantum_platform::setLogStream |
|     [\[1\]](api/langu             |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq14 |                                   |
| KrausSelection14KrausSelectionEv) |  function)](api/languages/cpp_api |
| -                                 | .html#_CPPv4N5cudaq16quantum_plat |
|   [cudaq::KrausSelection::op_name | form12setLogStreamERNSt7ostreamE) |
|     (C++                          | -   [cudaq::quantum_platfor       |
|     member)](                     | m::supports_explicit_measurements |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14KrausSelection7op_nameE) |     function)](api/l              |
| -   [                             | anguages/cpp_api.html#_CPPv4NK5cu |
| cudaq::KrausSelection::operator== | daq16quantum_platform30supports_e |
|     (C++                          | xplicit_measurementsENSt6size_tE) |
|     function)](api/languages      | -   [cudaq::quantum_pla           |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | tform::supports_task_distribution |
| usSelectioneqERK14KrausSelection) |     (C++                          |
| -                                 |     fu                            |
|    [cudaq::KrausSelection::qubits | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq16quantum_platfo |
|     member)]                      | rm26supports_task_distributionEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::quantum               |
| 4N5cudaq14KrausSelection6qubitsE) | _platform::with_execution_context |
| -   [cudaq::KrausTrajectory (C++  |     (C++                          |
|     st                            |     function)                     |
| ruct)](api/languages/cpp_api.html | ](api/languages/cpp_api.html#_CPP |
| #_CPPv4N5cudaq15KrausTrajectoryE) | v4I0DpEN5cudaq16quantum_platform2 |
| -                                 | 2with_execution_contextEDaR16Exec |
|  [cudaq::KrausTrajectory::builder | utionContextRR8CallableDpRR4Args) |
|     (C++                          | -   [cudaq::QuantumTask (C++      |
|     function)](ap                 |     type)](api/languages/cpp_api. |
| i/languages/cpp_api.html#_CPPv4N5 | html#_CPPv4N5cudaq11QuantumTaskE) |
| cudaq15KrausTrajectory7builderEv) | -   [cudaq::qubit (C++            |
| -   [cu                           |     type)](api/languages/c        |
| daq::KrausTrajectory::countErrors | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     (C++                          | -   [cudaq::QubitConnectivity     |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     ty                            |
| 15KrausTrajectory11countErrorsEv) | pe)](api/languages/cpp_api.html#_ |
| -   [                             | CPPv4N5cudaq17QubitConnectivityE) |
| cudaq::KrausTrajectory::isOrdered | -   [cudaq::QubitEdge (C++        |
|     (C++                          |     type)](api/languages/cpp_a    |
|     function)](api/l              | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| anguages/cpp_api.html#_CPPv4NK5cu | -   [cudaq::qudit (C++            |
| daq15KrausTrajectory9isOrderedEv) |     clas                          |
| -   [cudaq::                      | s)](api/languages/cpp_api.html#_C |
| KrausTrajectory::kraus_selections | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     (C++                          | -   [cudaq::qudit::qudit (C++     |
|     member)](api/languag          |                                   |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | function)](api/languages/cpp_api. |
| ausTrajectory16kraus_selectionsE) | html#_CPPv4N5cudaq5qudit5quditEv) |
| -   [cudaq:                       | -   [cudaq::qvector (C++          |
| :KrausTrajectory::KrausTrajectory |     class)                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function                      | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qvector::back (C++    |
| Pv4N5cudaq15KrausTrajectory15Krau |     function)](a                  |
| sTrajectoryENSt6size_tENSt6vector | pi/languages/cpp_api.html#_CPPv4N |
| I14KrausSelectionEEdNSt6size_tE), | 5cudaq7qvector4backENSt6size_tE), |
|     [\[1\]](api/languag           |                                   |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |   [\[1\]](api/languages/cpp_api.h |
| ausTrajectory15KrausTrajectoryEv) | tml#_CPPv4N5cudaq7qvector4backEv) |
| -   [cudaq::Kr                    | -   [cudaq::qvector::begin (C++   |
| ausTrajectory::measurement_counts |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     member)](api/languages        | ml#_CPPv4N5cudaq7qvector5beginEv) |
| /cpp_api.html#_CPPv4N5cudaq15Krau | -   [cudaq::qvector::clear (C++   |
| sTrajectory18measurement_countsE) |     fu                            |
| -   [cud                          | nction)](api/languages/cpp_api.ht |
| aq::KrausTrajectory::multiplicity | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     (C++                          | -   [cudaq::qvector::end (C++     |
|     member)](api/lan              |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq | function)](api/languages/cpp_api. |
| 15KrausTrajectory12multiplicityE) | html#_CPPv4N5cudaq7qvector3endEv) |
| -   [                             | -   [cudaq::qvector::front (C++   |
| cudaq::KrausTrajectory::num_shots |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     member)](api                  | cudaq7qvector5frontENSt6size_tE), |
| /languages/cpp_api.html#_CPPv4N5c |                                   |
| udaq15KrausTrajectory9num_shotsE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [c                            | ml#_CPPv4N5cudaq7qvector5frontEv) |
| udaq::KrausTrajectory::operator== | -   [cudaq::qvector::operator=    |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     functio                       |
| pp_api.html#_CPPv4NK5cudaq15Kraus | n)](api/languages/cpp_api.html#_C |
| TrajectoryeqERK15KrausTrajectory) | PPv4N5cudaq7qvectoraSERK7qvector) |
| -   [cu                           | -   [cudaq::qvector::operator\[\] |
| daq::KrausTrajectory::probability |     (C++                          |
|     (C++                          |     function)                     |
|     member)](api/la               | ](api/languages/cpp_api.html#_CPP |
| nguages/cpp_api.html#_CPPv4N5cuda | v4N5cudaq7qvectorixEKNSt6size_tE) |
| q15KrausTrajectory11probabilityE) | -   [cudaq::qvector::qvector (C++ |
| -   [cuda                         |     function)](api/               |
| q::KrausTrajectory::trajectory_id | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq7qvector7qvectorENSt6size_tE), |
|     member)](api/lang             |     [\[1\]](a                     |
| uages/cpp_api.html#_CPPv4N5cudaq1 | pi/languages/cpp_api.html#_CPPv4N |
| 5KrausTrajectory13trajectory_idE) | 5cudaq7qvector7qvectorERK5state), |
| -                                 |     [\[2\]](api                   |
|   [cudaq::KrausTrajectory::weight | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq7qvector7qvectorERK7qvector), |
|     member)](                     |     [\[3\]](ap                    |
| api/languages/cpp_api.html#_CPPv4 | i/languages/cpp_api.html#_CPPv4N5 |
| N5cudaq15KrausTrajectory6weightE) | cudaq7qvector7qvectorERR7qvector) |
| -                                 | -   [cudaq::qvector::size (C++    |
|    [cudaq::KrausTrajectoryBuilder |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     class)](                      | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qvector::slice (C++   |
| N5cudaq22KrausTrajectoryBuilderE) |     function)](api/language       |
| -   [cud                          | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| aq::KrausTrajectoryBuilder::build | tor5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qvector::value_type   |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     typ                           |
| 22KrausTrajectoryBuilder5buildEv) | e)](api/languages/cpp_api.html#_C |
| -   [cud                          | PPv4N5cudaq7qvector10value_typeE) |
| aq::KrausTrajectoryBuilder::setId | -   [cudaq::qview (C++            |
|     (C++                          |     clas                          |
|     function)](api/languages/cpp  | s)](api/languages/cpp_api.html#_C |
| _api.html#_CPPv4N5cudaq22KrausTra | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| jectoryBuilder5setIdENSt6size_tE) | -   [cudaq::qview::back (C++      |
| -   [cudaq::Kraus                 |     function)                     |
| TrajectoryBuilder::setProbability | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq5qview4backENSt6size_tE) |
|     function)](api/languages/cpp  | -   [cudaq::qview::begin (C++     |
| _api.html#_CPPv4N5cudaq22KrausTra |                                   |
| jectoryBuilder14setProbabilityEd) | function)](api/languages/cpp_api. |
| -   [cudaq::Krau                  | html#_CPPv4N5cudaq5qview5beginEv) |
| sTrajectoryBuilder::setSelections | -   [cudaq::qview::end (C++       |
|     (C++                          |                                   |
|     function)](api/languag        |   function)](api/languages/cpp_ap |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | i.html#_CPPv4N5cudaq5qview3endEv) |
| ausTrajectoryBuilder13setSelectio | -   [cudaq::qview::front (C++     |
| nsENSt6vectorI14KrausSelectionEE) |     function)](                   |
| -   [cudaq::matrix_callback (C++  | api/languages/cpp_api.html#_CPPv4 |
|     c                             | N5cudaq5qview5frontENSt6size_tE), |
| lass)](api/languages/cpp_api.html |                                   |
| #_CPPv4N5cudaq15matrix_callbackE) |    [\[1\]](api/languages/cpp_api. |
| -   [cudaq::matrix_handler (C++   | html#_CPPv4N5cudaq5qview5frontEv) |
|                                   | -   [cudaq::qview::operator\[\]   |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14matrix_handlerE) |     functio                       |
| -   [cudaq::mat                   | n)](api/languages/cpp_api.html#_C |
| rix_handler::commutation_behavior | PPv4N5cudaq5qviewixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qview::qview (C++     |
|     struct)](api/languages/       |     functio                       |
| cpp_api.html#_CPPv4N5cudaq14matri | n)](api/languages/cpp_api.html#_C |
| x_handler20commutation_behaviorE) | PPv4I0EN5cudaq5qview5qviewERR1R), |
| -                                 |     [\[1                          |
|    [cudaq::matrix_handler::define | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq5qview5qviewERK5qview) |
|     function)](a                  | -   [cudaq::qview::size (C++      |
| pi/languages/cpp_api.html#_CPPv4N |                                   |
| 5cudaq14matrix_handler6defineENSt | function)](api/languages/cpp_api. |
| 6stringENSt6vectorINSt7int64_tEEE | html#_CPPv4NK5cudaq5qview4sizeEv) |
| RR15matrix_callbackRKNSt13unorder | -   [cudaq::qview::slice (C++     |
| ed_mapINSt6stringENSt6stringEEE), |     function)](api/langua         |
|                                   | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| [\[1\]](api/languages/cpp_api.htm | iew5sliceENSt6size_tENSt6size_tE) |
| l#_CPPv4N5cudaq14matrix_handler6d | -   [cudaq::qview::value_type     |
| efineENSt6stringENSt6vectorINSt7i |     (C++                          |
| nt64_tEEERR15matrix_callbackRR20d |     t                             |
| iag_matrix_callbackRKNSt13unorder | ype)](api/languages/cpp_api.html# |
| ed_mapINSt6stringENSt6stringEEE), | _CPPv4N5cudaq5qview10value_typeE) |
|     [\[2\]](                      | -   [cudaq::range (C++            |
| api/languages/cpp_api.html#_CPPv4 |     fun                           |
| N5cudaq14matrix_handler6defineENS | ction)](api/languages/cpp_api.htm |
| t6stringENSt6vectorINSt7int64_tEE | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| ERR15matrix_callbackRRNSt13unorde | orI11ElementTypeEE11ElementType), |
| red_mapINSt6stringENSt6stringEEE) |     [\[1\]](api/languages/cpp_    |
| -                                 | api.html#_CPPv4I0EN5cudaq5rangeEN |
|   [cudaq::matrix_handler::degrees | St6vectorI11ElementTypeEE11Elemen |
|     (C++                          | tType11ElementType11ElementType), |
|     function)](ap                 |     [                             |
| i/languages/cpp_api.html#_CPPv4NK | \[2\]](api/languages/cpp_api.html |
| 5cudaq14matrix_handler7degreesEv) | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| -                                 | -   [cudaq::real (C++             |
|  [cudaq::matrix_handler::displace |     type)](api/languages/         |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq4realE) |
|     function)](api/language       | -   [cudaq::registry (C++         |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     type)](api/languages/cpp_     |
| rix_handler8displaceENSt6size_tE) | api.html#_CPPv4N5cudaq8registryE) |
| -   [cudaq::matrix                | -                                 |
| _handler::get_expected_dimensions |  [cudaq::registry::RegisteredType |
|     (C++                          |     (C++                          |
|                                   |     class)](api/                  |
|    function)](api/languages/cpp_a | languages/cpp_api.html#_CPPv4I0EN |
| pi.html#_CPPv4NK5cudaq14matrix_ha | 5cudaq8registry14RegisteredTypeE) |
| ndler23get_expected_dimensionsEv) | -   [cudaq::RemoteCapabilities    |
| -   [cudaq::matrix_ha             |     (C++                          |
| ndler::get_parameter_descriptions |     struc                         |
|     (C++                          | t)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq18RemoteCapabilitiesE) |
| function)](api/languages/cpp_api. | -   [cudaq::Remo                  |
| html#_CPPv4NK5cudaq14matrix_handl | teCapabilities::isRemoteSimulator |
| er26get_parameter_descriptionsEv) |     (C++                          |
| -   [c                            |     member)](api/languages/c      |
| udaq::matrix_handler::instantiate | pp_api.html#_CPPv4N5cudaq18Remote |
|     (C++                          | Capabilities17isRemoteSimulatorE) |
|     function)](a                  | -   [cudaq::Remot                 |
| pi/languages/cpp_api.html#_CPPv4N | eCapabilities::RemoteCapabilities |
| 5cudaq14matrix_handler11instantia |     (C++                          |
| teENSt6stringERKNSt6vectorINSt6si |     function)](api/languages/cpp  |
| ze_tEEERK20commutation_behavior), | _api.html#_CPPv4N5cudaq18RemoteCa |
|     [\[1\]](                      | pabilities18RemoteCapabilitiesEb) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq:                       |
| N5cudaq14matrix_handler11instanti | :RemoteCapabilities::stateOverlap |
| ateENSt6stringERRNSt6vectorINSt6s |     (C++                          |
| ize_tEEERK20commutation_behavior) |     member)](api/langua           |
| -   [cuda                         | ges/cpp_api.html#_CPPv4N5cudaq18R |
| q::matrix_handler::matrix_handler | emoteCapabilities12stateOverlapE) |
|     (C++                          | -                                 |
|     function)](api/languag        |   [cudaq::RemoteCapabilities::vqe |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     (C++                          |
| ble_if_tINSt12is_base_of_vI16oper |     member)](                     |
| ator_handler1TEEbEEEN5cudaq14matr | api/languages/cpp_api.html#_CPPv4 |
| ix_handler14matrix_handlerERK1T), | N5cudaq18RemoteCapabilities3vqeE) |
|     [\[1\]](ap                    | -   [cudaq::RemoteSimulationState |
| i/languages/cpp_api.html#_CPPv4I0 |     (C++                          |
| _NSt11enable_if_tINSt12is_base_of |     class)]                       |
| _vI16operator_handler1TEEbEEEN5cu | (api/languages/cpp_api.html#_CPPv |
| daq14matrix_handler14matrix_handl | 4N5cudaq21RemoteSimulationStateE) |
| erERK1TRK20commutation_behavior), | -   [cudaq::Resources (C++        |
|     [\[2\]](api/languages/cpp_ap  |     class)](api/languages/cpp_a   |
| i.html#_CPPv4N5cudaq14matrix_hand | pi.html#_CPPv4N5cudaq9ResourcesE) |
| ler14matrix_handlerENSt6size_tE), | -   [cudaq::run (C++              |
|     [\[3\]](api/                  |     function)]                    |
| languages/cpp_api.html#_CPPv4N5cu | (api/languages/cpp_api.html#_CPPv |
| daq14matrix_handler14matrix_handl | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| erENSt6stringERKNSt6vectorINSt6si | 5invoke_result_tINSt7decay_tI13Qu |
| ze_tEEERK20commutation_behavior), | antumKernelEEDpNSt7decay_tI4ARGSE |
|     [\[4\]](api/                  | EEEEENSt6size_tERN5cudaq11noise_m |
| languages/cpp_api.html#_CPPv4N5cu | odelERR13QuantumKernelDpRR4ARGS), |
| daq14matrix_handler14matrix_handl |     [\[1\]](api/langu             |
| erENSt6stringERRNSt6vectorINSt6si | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| ze_tEEERK20commutation_behavior), | daq3runENSt6vectorINSt15invoke_re |
|     [\                            | sult_tINSt7decay_tI13QuantumKerne |
| [5\]](api/languages/cpp_api.html# | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| _CPPv4N5cudaq14matrix_handler14ma | ize_tERR13QuantumKernelDpRR4ARGS) |
| trix_handlerERK14matrix_handler), | -   [cudaq::run_async (C++        |
|     [                             |     functio                       |
| \[6\]](api/languages/cpp_api.html | n)](api/languages/cpp_api.html#_C |
| #_CPPv4N5cudaq14matrix_handler14m | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| atrix_handlerERR14matrix_handler) | tureINSt6vectorINSt15invoke_resul |
| -                                 | t_tINSt7decay_tI13QuantumKernelEE |
|  [cudaq::matrix_handler::momentum | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|     (C++                          | ze_tENSt6size_tERN5cudaq11noise_m |
|     function)](api/language       | odelERR13QuantumKernelDpRR4ARGS), |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     [\[1\]](api/la                |
| rix_handler8momentumENSt6size_tE) | nguages/cpp_api.html#_CPPv4I0DpEN |
| -                                 | 5cudaq9run_asyncENSt6futureINSt6v |
|    [cudaq::matrix_handler::number | ectorINSt15invoke_result_tINSt7de |
|     (C++                          | cay_tI13QuantumKernelEEDpNSt7deca |
|     function)](api/langua         | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| ges/cpp_api.html#_CPPv4N5cudaq14m | ize_tERR13QuantumKernelDpRR4ARGS) |
| atrix_handler6numberENSt6size_tE) | -   [cudaq::RuntimeTarget (C++    |
| -                                 |                                   |
| [cudaq::matrix_handler::operator= | struct)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     fun                           | -   [cudaq::sample (C++           |
| ction)](api/languages/cpp_api.htm |     function)](api/languages/c    |
| l#_CPPv4I0_NSt11enable_if_tIXaant | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| NSt7is_sameI1T14matrix_handlerE5v | mpleE13sample_resultRK14sample_op |
| alueENSt12is_base_of_vI16operator | tionsRR13QuantumKernelDpRR4Args), |
| _handler1TEEEbEEEN5cudaq14matrix_ |     [\[1\                         |
| handleraSER14matrix_handlerRK1T), | ]](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/languages         | Pv4I0DpEN5cudaq6sampleE13sample_r |
| /cpp_api.html#_CPPv4N5cudaq14matr | esultRR13QuantumKernelDpRR4Args), |
| ix_handleraSERK14matrix_handler), |     [\                            |
|     [\[2\]](api/language          | [2\]](api/languages/cpp_api.html# |
| s/cpp_api.html#_CPPv4N5cudaq14mat | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| rix_handleraSERR14matrix_handler) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [                             | -   [cudaq::sample_options (C++   |
| cudaq::matrix_handler::operator== |     s                             |
|     (C++                          | truct)](api/languages/cpp_api.htm |
|     function)](api/languages      | l#_CPPv4N5cudaq14sample_optionsE) |
| /cpp_api.html#_CPPv4NK5cudaq14mat | -   [cudaq::sample_result (C++    |
| rix_handlereqERK14matrix_handler) |                                   |
| -                                 |  class)](api/languages/cpp_api.ht |
|    [cudaq::matrix_handler::parity | ml#_CPPv4N5cudaq13sample_resultE) |
|     (C++                          | -   [cudaq::sample_result::append |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     function)](api/languages/cpp_ |
| atrix_handler6parityENSt6size_tE) | api.html#_CPPv4N5cudaq13sample_re |
| -                                 | sult6appendERK15ExecutionResultb) |
|  [cudaq::matrix_handler::position | -   [cudaq::sample_result::begin  |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)]                    |
| s/cpp_api.html#_CPPv4N5cudaq14mat | (api/languages/cpp_api.html#_CPPv |
| rix_handler8positionENSt6size_tE) | 4N5cudaq13sample_result5beginEv), |
| -   [cudaq::                      |     [\[1\]]                       |
| matrix_handler::remove_definition | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NK5cudaq13sample_result5beginEv) |
|     fu                            | -   [cudaq::sample_result::cbegin |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14matrix_handler1 |     function)](                   |
| 7remove_definitionERKNSt6stringE) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | NK5cudaq13sample_result6cbeginEv) |
|   [cudaq::matrix_handler::squeeze | -   [cudaq::sample_result::cend   |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)                     |
| es/cpp_api.html#_CPPv4N5cudaq14ma | ](api/languages/cpp_api.html#_CPP |
| trix_handler7squeezeENSt6size_tE) | v4NK5cudaq13sample_result4cendEv) |
| -   [cudaq::m                     | -   [cudaq::sample_result::clear  |
| atrix_handler::to_diagonal_matrix |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/lang           | ](api/languages/cpp_api.html#_CPP |
| uages/cpp_api.html#_CPPv4NK5cudaq | v4N5cudaq13sample_result5clearEv) |
| 14matrix_handler18to_diagonal_mat | -   [cudaq::sample_result::count  |
| rixERNSt13unordered_mapINSt6size_ |     (C++                          |
| tENSt7int64_tEEERKNSt13unordered_ |     function)](                   |
| mapINSt6stringENSt7complexIdEEEE) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | NK5cudaq13sample_result5countENSt |
| [cudaq::matrix_handler::to_matrix | 11string_viewEKNSt11string_viewE) |
|     (C++                          | -   [                             |
|     function)                     | cudaq::sample_result::deserialize |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4NK5cudaq14matrix_handler9to_mat |     functio                       |
| rixERNSt13unordered_mapINSt6size_ | n)](api/languages/cpp_api.html#_C |
| tENSt7int64_tEEERKNSt13unordered_ | PPv4N5cudaq13sample_result11deser |
| mapINSt6stringENSt7complexIdEEEE) | ializeERNSt6vectorINSt6size_tEEE) |
| -                                 | -   [cudaq::sample_result::dump   |
| [cudaq::matrix_handler::to_string |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/               | es/cpp_api.html#_CPPv4NK5cudaq13s |
| languages/cpp_api.html#_CPPv4NK5c | ample_result4dumpERNSt7ostreamE), |
| udaq14matrix_handler9to_stringEb) |     [\[1\]                        |
| -                                 | ](api/languages/cpp_api.html#_CPP |
| [cudaq::matrix_handler::unique_id | v4NK5cudaq13sample_result4dumpEv) |
|     (C++                          | -   [cudaq::sample_result::end    |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     function                      |
| udaq14matrix_handler9unique_idEv) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq:                       | Pv4N5cudaq13sample_result3endEv), |
| :matrix_handler::\~matrix_handler |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     functi                        | Pv4NK5cudaq13sample_result3endEv) |
| on)](api/languages/cpp_api.html#_ | -   [                             |
| CPPv4N5cudaq14matrix_handlerD0Ev) | cudaq::sample_result::expectation |
| -   [cudaq::matrix_op (C++        |     (C++                          |
|     type)](api/languages/cpp_a    |     f                             |
| pi.html#_CPPv4N5cudaq9matrix_opE) | unction)](api/languages/cpp_api.h |
| -   [cudaq::matrix_op_term (C++   | tml#_CPPv4NK5cudaq13sample_result |
|                                   | 11expectationEKNSt11string_viewE) |
|  type)](api/languages/cpp_api.htm | -   [c                            |
| l#_CPPv4N5cudaq14matrix_op_termE) | udaq::sample_result::get_marginal |
| -                                 |     (C++                          |
|    [cudaq::mdiag_operator_handler |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4NK5cudaq13sample_r |
|     class)](                      | esult12get_marginalERKNSt6vectorI |
| api/languages/cpp_api.html#_CPPv4 | NSt6size_tEEEKNSt11string_viewE), |
| N5cudaq22mdiag_operator_handlerE) |     [\[1\]](api/languages/cpp_    |
| -   [cudaq::mpi (C++              | api.html#_CPPv4NK5cudaq13sample_r |
|     type)](api/languages          | esult12get_marginalERRKNSt6vector |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | INSt6size_tEEEKNSt11string_viewE) |
| -   [cudaq::mpi::all_gather (C++  | -   [cuda                         |
|     fu                            | q::sample_result::get_total_shots |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     function)](api/langua         |
| RNSt6vectorIdEERKNSt6vectorIdEE), | ges/cpp_api.html#_CPPv4NK5cudaq13 |
|                                   | sample_result15get_total_shotsEv) |
|   [\[1\]](api/languages/cpp_api.h | -   [cuda                         |
| tml#_CPPv4N5cudaq3mpi10all_gather | q::sample_result::has_even_parity |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     (C++                          |
| -   [cudaq::mpi::all_reduce (C++  |     fun                           |
|                                   | ction)](api/languages/cpp_api.htm |
|  function)](api/languages/cpp_api | l#_CPPv4N5cudaq13sample_result15h |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | as_even_parityENSt11string_viewE) |
| reduceE1TRK1TRK14BinaryFunction), | -   [cuda                         |
|     [\[1\]](api/langu             | q::sample_result::has_expectation |
| ages/cpp_api.html#_CPPv4I00EN5cud |     (C++                          |
| aq3mpi10all_reduceE1TRK1TRK4Func) |     funct                         |
| -   [cudaq::mpi::broadcast (C++   | ion)](api/languages/cpp_api.html# |
|     function)](api/               | _CPPv4NK5cudaq13sample_result15ha |
| languages/cpp_api.html#_CPPv4N5cu | s_expectationEKNSt11string_viewE) |
| daq3mpi9broadcastERNSt6stringEi), | -   [cu                           |
|     [\[1\]](api/la                | daq::sample_result::most_probable |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q3mpi9broadcastERNSt6vectorIdEEi) |     fun                           |
| -   [cudaq::mpi::finalize (C++    | ction)](api/languages/cpp_api.htm |
|     f                             | l#_CPPv4NK5cudaq13sample_result13 |
| unction)](api/languages/cpp_api.h | most_probableEKNSt11string_viewE) |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | -                                 |
| -   [cudaq::mpi::initialize (C++  | [cudaq::sample_result::operator+= |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/langua         |
| Pv4N5cudaq3mpi10initializeEiPPc), | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     [                             | ample_resultpLERK13sample_result) |
| \[1\]](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq3mpi10initializeEv) |  [cudaq::sample_result::operator= |
| -   [cudaq::mpi::is_initialized   |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function                      | ges/cpp_api.html#_CPPv4N5cudaq13s |
| )](api/languages/cpp_api.html#_CP | ample_resultaSERR13sample_result) |
| Pv4N5cudaq3mpi14is_initializedEv) | -                                 |
| -   [cudaq::mpi::num_ranks (C++   | [cudaq::sample_result::operator== |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/languag        |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | es/cpp_api.html#_CPPv4NK5cudaq13s |
| -   [cudaq::mpi::rank (C++        | ample_resulteqERK13sample_result) |
|                                   | -   [                             |
|    function)](api/languages/cpp_a | cudaq::sample_result::probability |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     (C++                          |
| -   [cudaq::noise_model (C++      |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4NK5cuda |
|    class)](api/languages/cpp_api. | q13sample_result11probabilityENSt |
| html#_CPPv4N5cudaq11noise_modelE) | 11string_viewEKNSt11string_viewE) |
| -   [cudaq::n                     | -   [cud                          |
| oise_model::add_all_qubit_channel | aq::sample_result::register_names |
|     (C++                          |     (C++                          |
|     function)](api                |     function)](api/langu          |
| /languages/cpp_api.html#_CPPv4IDp | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| EN5cudaq11noise_model21add_all_qu | 3sample_result14register_namesEv) |
| bit_channelEvRK13kraus_channeli), | -                                 |
|     [\[1\]](api/langua            |    [cudaq::sample_result::reorder |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     (C++                          |
| oise_model21add_all_qubit_channel |     function)](api/langua         |
| ERKNSt6stringERK13kraus_channeli) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -                                 | ample_result7reorderERKNSt6vector |
|  [cudaq::noise_model::add_channel | INSt6size_tEEEKNSt11string_viewE) |
|     (C++                          | -   [cu                           |
|     funct                         | daq::sample_result::sample_result |
| ion)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4IDpEN5cudaq11noise_model11a |     func                          |
| dd_channelEvRK15PredicateFuncTy), | tion)](api/languages/cpp_api.html |
|     [\[1\]](api/languages/cpp_    | #_CPPv4N5cudaq13sample_result13sa |
| api.html#_CPPv4IDpEN5cudaq11noise | mple_resultERK15ExecutionResult), |
| _model11add_channelEvRKNSt6vector |     [\[1\]](api/la                |
| INSt6size_tEEERK13kraus_channel), | nguages/cpp_api.html#_CPPv4N5cuda |
|     [\[2\]](ap                    | q13sample_result13sample_resultER |
| i/languages/cpp_api.html#_CPPv4N5 | KNSt6vectorI15ExecutionResultEE), |
| cudaq11noise_model11add_channelER |                                   |
| KNSt6stringERK15PredicateFuncTy), |  [\[2\]](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4N5cudaq13sample_result13 |
| [\[3\]](api/languages/cpp_api.htm | sample_resultERR13sample_result), |
| l#_CPPv4N5cudaq11noise_model11add |     [                             |
| _channelERKNSt6stringERKNSt6vecto | \[3\]](api/languages/cpp_api.html |
| rINSt6size_tEEERK13kraus_channel) | #_CPPv4N5cudaq13sample_result13sa |
| -   [cudaq::noise_model::empty    | mple_resultERR15ExecutionResult), |
|     (C++                          |     [\[4\]](api/lan               |
|     function                      | guages/cpp_api.html#_CPPv4N5cudaq |
| )](api/languages/cpp_api.html#_CP | 13sample_result13sample_resultEdR |
| Pv4NK5cudaq11noise_model5emptyEv) | KNSt6vectorI15ExecutionResultEE), |
| -                                 |     [\[5\]](api/lan               |
| [cudaq::noise_model::get_channels | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 13sample_result13sample_resultEv) |
|     function)](api/l              | -                                 |
| anguages/cpp_api.html#_CPPv4I0ENK |  [cudaq::sample_result::serialize |
| 5cudaq11noise_model12get_channels |     (C++                          |
| ENSt6vectorI13kraus_channelEERKNS |     function)](api                |
| t6vectorINSt6size_tEEERKNSt6vecto | /languages/cpp_api.html#_CPPv4NK5 |
| rINSt6size_tEEERKNSt6vectorIdEE), | cudaq13sample_result9serializeEv) |
|     [\[1\]](api/languages/cpp_a   | -   [cudaq::sample_result::size   |
| pi.html#_CPPv4NK5cudaq11noise_mod |     (C++                          |
| el12get_channelsERKNSt6stringERKN |     function)](api/languages/c    |
| St6vectorINSt6size_tEEERKNSt6vect | pp_api.html#_CPPv4NK5cudaq13sampl |
| orINSt6size_tEEERKNSt6vectorIdEE) | e_result4sizeEKNSt11string_viewE) |
| -                                 | -   [cudaq::sample_result::to_map |
|  [cudaq::noise_model::noise_model |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api                | _api.html#_CPPv4NK5cudaq13sample_ |
| /languages/cpp_api.html#_CPPv4N5c | result6to_mapEKNSt11string_viewE) |
| udaq11noise_model11noise_modelEv) | -   [cuda                         |
| -   [cu                           | q::sample_result::\~sample_result |
| daq::noise_model::PredicateFuncTy |     (C++                          |
|     (C++                          |     funct                         |
|     type)](api/la                 | ion)](api/languages/cpp_api.html# |
| nguages/cpp_api.html#_CPPv4N5cuda | _CPPv4N5cudaq13sample_resultD0Ev) |
| q11noise_model15PredicateFuncTyE) | -   [cudaq::scalar_callback (C++  |
| -   [cud                          |     c                             |
| aq::noise_model::register_channel | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_callbackE) |
|     function)](api/languages      | -   [c                            |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | udaq::scalar_callback::operator() |
| noise_model16register_channelEvv) |     (C++                          |
| -   [cudaq::                      |     function)](api/language       |
| noise_model::requires_constructor | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|     (C++                          | alar_callbackclERKNSt13unordered_ |
|     type)](api/languages/cp       | mapINSt6stringENSt7complexIdEEEE) |
| p_api.html#_CPPv4I0DpEN5cudaq11no | -   [                             |
| ise_model20requires_constructorE) | cudaq::scalar_callback::operator= |
| -   [cudaq::noise_model_type (C++ |     (C++                          |
|     e                             |     function)](api/languages/c    |
| num)](api/languages/cpp_api.html# | pp_api.html#_CPPv4N5cudaq15scalar |
| _CPPv4N5cudaq16noise_model_typeE) | _callbackaSERK15scalar_callback), |
| -   [cudaq::no                    |     [\[1\]](api/languages/        |
| ise_model_type::amplitude_damping | cpp_api.html#_CPPv4N5cudaq15scala |
|     (C++                          | r_callbackaSERR15scalar_callback) |
|     enumerator)](api/languages    | -   [cudaq:                       |
| /cpp_api.html#_CPPv4N5cudaq16nois | :scalar_callback::scalar_callback |
| e_model_type17amplitude_dampingE) |     (C++                          |
| -   [cudaq::noise_mode            |     function)](api/languag        |
| l_type::amplitude_damping_channel | es/cpp_api.html#_CPPv4I0_NSt11ena |
|     (C++                          | ble_if_tINSt16is_invocable_r_vINS |
|     e                             | t7complexIdEE8CallableRKNSt13unor |
| numerator)](api/languages/cpp_api | dered_mapINSt6stringENSt7complexI |
| .html#_CPPv4N5cudaq16noise_model_ | dEEEEEEbEEEN5cudaq15scalar_callba |
| type25amplitude_damping_channelE) | ck15scalar_callbackERR8Callable), |
| -   [cudaq::n                     |     [\[1\                         |
| oise_model_type::bit_flip_channel | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_callback15scal |
|     enumerator)](api/language     | ar_callbackERK15scalar_callback), |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     [\[2                          |
| se_model_type16bit_flip_channelE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::                      | PPv4N5cudaq15scalar_callback15sca |
| noise_model_type::depolarization1 | lar_callbackERR15scalar_callback) |
|     (C++                          | -   [cudaq::scalar_operator (C++  |
|     enumerator)](api/languag      |     c                             |
| es/cpp_api.html#_CPPv4N5cudaq16no | lass)](api/languages/cpp_api.html |
| ise_model_type15depolarization1E) | #_CPPv4N5cudaq15scalar_operatorE) |
| -   [cudaq::                      | -                                 |
| noise_model_type::depolarization2 | [cudaq::scalar_operator::evaluate |
|     (C++                          |     (C++                          |
|     enumerator)](api/languag      |                                   |
| es/cpp_api.html#_CPPv4N5cudaq16no |    function)](api/languages/cpp_a |
| ise_model_type15depolarization2E) | pi.html#_CPPv4NK5cudaq15scalar_op |
| -   [cudaq::noise_m               | erator8evaluateERKNSt13unordered_ |
| odel_type::depolarization_channel | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [cudaq::scalar_ope            |
|                                   | rator::get_parameter_descriptions |
|   enumerator)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq16noise_mod |     f                             |
| el_type22depolarization_channelE) | unction)](api/languages/cpp_api.h |
| -                                 | tml#_CPPv4NK5cudaq15scalar_operat |
|  [cudaq::noise_model_type::pauli1 | or26get_parameter_descriptionsEv) |
|     (C++                          | -   [cu                           |
|     enumerator)](a                | daq::scalar_operator::is_constant |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq16noise_model_type6pauli1E) |     function)](api/lang           |
| -                                 | uages/cpp_api.html#_CPPv4NK5cudaq |
|  [cudaq::noise_model_type::pauli2 | 15scalar_operator11is_constantEv) |
|     (C++                          | -   [c                            |
|     enumerator)](a                | udaq::scalar_operator::operator\* |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq16noise_model_type6pauli2E) |     function                      |
| -   [cudaq                        | )](api/languages/cpp_api.html#_CP |
| ::noise_model_type::phase_damping | Pv4N5cudaq15scalar_operatormlENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     enumerator)](api/langu        |     [\[1\                         |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ]](api/languages/cpp_api.html#_CP |
| noise_model_type13phase_dampingE) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::noi                   | 7complexIdEERR15scalar_operator), |
| se_model_type::phase_flip_channel |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](api/languages/   | operatormlEdRK15scalar_operator), |
| cpp_api.html#_CPPv4N5cudaq16noise |     [\[3\]](api/languages/cp      |
| _model_type18phase_flip_channelE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatormlEdRR15scalar_operator), |
| [cudaq::noise_model_type::unknown |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     enumerator)](ap               | alar_operatormlENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[5\]](api/languages/cpp     |
| cudaq16noise_model_type7unknownE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -                                 | _operatormlERK15scalar_operator), |
| [cudaq::noise_model_type::x_error |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     enumerator)](ap               | 4NKR5cudaq15scalar_operatormlEd), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[7\]](api/language          |
| cudaq16noise_model_type7x_errorE) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -                                 | alar_operatormlENSt7complexIdEE), |
| [cudaq::noise_model_type::y_error |     [\[8\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     enumerator)](ap               | _operatormlERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[9\                         |
| cudaq16noise_model_type7y_errorE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4NO5cudaq15scalar_operatormlEd) |
| [cudaq::noise_model_type::z_error | -   [cu                           |
|     (C++                          | daq::scalar_operator::operator\*= |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languag        |
| cudaq16noise_model_type7z_errorE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::num_available_gpus    | alar_operatormLENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     function                      | pp_api.html#_CPPv4N5cudaq15scalar |
| )](api/languages/cpp_api.html#_CP | _operatormLERK15scalar_operator), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[2                          |
| -   [cudaq::observe (C++          | \]](api/languages/cpp_api.html#_C |
|     function)]                    | PPv4N5cudaq15scalar_operatormLEd) |
| (api/languages/cpp_api.html#_CPPv | -   [                             |
| 4I00DpEN5cudaq7observeENSt6vector | cudaq::scalar_operator::operator+ |
| I14observe_resultEERR13QuantumKer |     (C++                          |
| nelRK15SpinOpContainerDpRR4Args), |     function                      |
|     [\[1\]](api/languages/cpp_ap  | )](api/languages/cpp_api.html#_CP |
| i.html#_CPPv4I0DpEN5cudaq7observe | Pv4N5cudaq15scalar_operatorplENSt |
| E14observe_resultNSt6size_tERR13Q | 7complexIdEERK15scalar_operator), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[1\                         |
|     [\[                           | ]](api/languages/cpp_api.html#_CP |
| 2\]](api/languages/cpp_api.html#_ | Pv4N5cudaq15scalar_operatorplENSt |
| CPPv4I0DpEN5cudaq7observeE14obser | 7complexIdEERR15scalar_operator), |
| ve_resultRK15observe_optionsRR13Q |     [\[2\]](api/languages/cp      |
| uantumKernelRK7spin_opDpRR4Args), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[3\]](api/lang              | operatorplEdRK15scalar_operator), |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     [\[3\]](api/languages/cp      |
| udaq7observeE14observe_resultRR13 | p_api.html#_CPPv4N5cudaq15scalar_ |
| QuantumKernelRK7spin_opDpRR4Args) | operatorplEdRR15scalar_operator), |
| -   [cudaq::observe_options (C++  |     [\[4\]](api/languages         |
|     st                            | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ruct)](api/languages/cpp_api.html | alar_operatorplENSt7complexIdEE), |
| #_CPPv4N5cudaq15observe_optionsE) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::observe_result (C++   | _api.html#_CPPv4NKR5cudaq15scalar |
|                                   | _operatorplERK15scalar_operator), |
| class)](api/languages/cpp_api.htm |     [\[6\]]                       |
| l#_CPPv4N5cudaq14observe_resultE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatorplEd), |
|    [cudaq::observe_result::counts |     [\[7\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/languages/c    | 4NKR5cudaq15scalar_operatorplEv), |
| pp_api.html#_CPPv4N5cudaq14observ |     [\[8\]](api/language          |
| e_result6countsERK12spin_op_term) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::observe_result::dump  | alar_operatorplENSt7complexIdEE), |
|     (C++                          |     [\[9\]](api/languages/cp      |
|     function)                     | p_api.html#_CPPv4NO5cudaq15scalar |
| ](api/languages/cpp_api.html#_CPP | _operatorplERK15scalar_operator), |
| v4N5cudaq14observe_result4dumpEv) |     [\[10\]                       |
| -   [c                            | ](api/languages/cpp_api.html#_CPP |
| udaq::observe_result::expectation | v4NO5cudaq15scalar_operatorplEd), |
|     (C++                          |     [\[11\                        |
|                                   | ]](api/languages/cpp_api.html#_CP |
| function)](api/languages/cpp_api. | Pv4NO5cudaq15scalar_operatorplEv) |
| html#_CPPv4N5cudaq14observe_resul | -   [c                            |
| t11expectationERK12spin_op_term), | udaq::scalar_operator::operator+= |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languag        |
| q14observe_result11expectationEv) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cuda                         | alar_operatorpLENSt7complexIdEE), |
| q::observe_result::id_coefficient |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function)](api/langu          | _operatorpLERK15scalar_operator), |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     [\[2                          |
| observe_result14id_coefficientEv) | \]](api/languages/cpp_api.html#_C |
| -   [cuda                         | PPv4N5cudaq15scalar_operatorpLEd) |
| q::observe_result::observe_result | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator- |
|                                   |     (C++                          |
|   function)](api/languages/cpp_ap |     function                      |
| i.html#_CPPv4N5cudaq14observe_res | )](api/languages/cpp_api.html#_CP |
| ult14observe_resultEdRK7spin_op), | Pv4N5cudaq15scalar_operatormiENSt |
|     [\[1\]](a                     | 7complexIdEERK15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[1\                         |
| 5cudaq14observe_result14observe_r | ]](api/languages/cpp_api.html#_CP |
| esultEdRK7spin_op13sample_result) | Pv4N5cudaq15scalar_operatormiENSt |
| -                                 | 7complexIdEERR15scalar_operator), |
|  [cudaq::observe_result::operator |     [\[2\]](api/languages/cp      |
|     double (C++                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|     functio                       | operatormiEdRK15scalar_operator), |
| n)](api/languages/cpp_api.html#_C |     [\[3\]](api/languages/cp      |
| PPv4N5cudaq14observe_resultcvdEv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatormiEdRR15scalar_operator), |
|  [cudaq::observe_result::raw_data |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     function)](ap                 | alar_operatormiENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[5\]](api/languages/cpp     |
| cudaq14observe_result8raw_dataEv) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::operator_handler (C++ | _operatormiERK15scalar_operator), |
|     cl                            |     [\[6\]]                       |
| ass)](api/languages/cpp_api.html# | (api/languages/cpp_api.html#_CPPv |
| _CPPv4N5cudaq16operator_handlerE) | 4NKR5cudaq15scalar_operatormiEd), |
| -   [cudaq::optimizable_function  |     [\[7\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     class)                        | 4NKR5cudaq15scalar_operatormiEv), |
| ](api/languages/cpp_api.html#_CPP |     [\[8\]](api/language          |
| v4N5cudaq20optimizable_functionE) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::optimization_result   | alar_operatormiENSt7complexIdEE), |
|     (C++                          |     [\[9\]](api/languages/cp      |
|     type                          | p_api.html#_CPPv4NO5cudaq15scalar |
| )](api/languages/cpp_api.html#_CP | _operatormiERK15scalar_operator), |
| Pv4N5cudaq19optimization_resultE) |     [\[10\]                       |
| -   [cudaq::optimizer (C++        | ](api/languages/cpp_api.html#_CPP |
|     class)](api/languages/cpp_a   | v4NO5cudaq15scalar_operatormiEd), |
| pi.html#_CPPv4N5cudaq9optimizerE) |     [\[11\                        |
| -   [cudaq::optimizer::optimize   | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatormiEv) |
|                                   | -   [c                            |
|  function)](api/languages/cpp_api | udaq::scalar_operator::operator-= |
| .html#_CPPv4N5cudaq9optimizer8opt |     (C++                          |
| imizeEKiRR20optimizable_function) |     function)](api/languag        |
| -   [cu                           | es/cpp_api.html#_CPPv4N5cudaq15sc |
| daq::optimizer::requiresGradients | alar_operatormIENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     function)](api/la             | pp_api.html#_CPPv4N5cudaq15scalar |
| nguages/cpp_api.html#_CPPv4N5cuda | _operatormIERK15scalar_operator), |
| q9optimizer17requiresGradientsEv) |     [\[2                          |
| -   [cudaq::orca (C++             | \]](api/languages/cpp_api.html#_C |
|     type)](api/languages/         | PPv4N5cudaq15scalar_operatormIEd) |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | -   [                             |
| -   [cudaq::orca::sample (C++     | cudaq::scalar_operator::operator/ |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     function                      |
| mpleERNSt6vectorINSt6size_tEEERNS | )](api/languages/cpp_api.html#_CP |
| t6vectorINSt6size_tEEERNSt6vector | Pv4N5cudaq15scalar_operatordvENSt |
| IdEERNSt6vectorIdEEiNSt6size_tE), | 7complexIdEERK15scalar_operator), |
|     [\[1\]]                       |     [\[1\                         |
| (api/languages/cpp_api.html#_CPPv | ]](api/languages/cpp_api.html#_CP |
| 4N5cudaq4orca6sampleERNSt6vectorI | Pv4N5cudaq15scalar_operatordvENSt |
| NSt6size_tEEERNSt6vectorINSt6size | 7complexIdEERR15scalar_operator), |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     [\[2\]](api/languages/cp      |
| -   [cudaq::orca::sample_async    | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRK15scalar_operator), |
|                                   |     [\[3\]](api/languages/cp      |
| function)](api/languages/cpp_api. | p_api.html#_CPPv4N5cudaq15scalar_ |
| html#_CPPv4N5cudaq4orca12sample_a | operatordvEdRR15scalar_operator), |
| syncERNSt6vectorINSt6size_tEEERNS |     [\[4\]](api/languages         |
| t6vectorINSt6size_tEEERNSt6vector | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| IdEERNSt6vectorIdEEiNSt6size_tE), | alar_operatordvENSt7complexIdEE), |
|     [\[1\]](api/la                |     [\[5\]](api/languages/cpp     |
| nguages/cpp_api.html#_CPPv4N5cuda | _api.html#_CPPv4NKR5cudaq15scalar |
| q4orca12sample_asyncERNSt6vectorI | _operatordvERK15scalar_operator), |
| NSt6size_tEEERNSt6vectorINSt6size |     [\[6\]]                       |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::OrcaRemoteRESTQPU     | 4NKR5cudaq15scalar_operatordvEd), |
|     (C++                          |     [\[7\]](api/language          |
|     cla                           | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| ss)](api/languages/cpp_api.html#_ | alar_operatordvENSt7complexIdEE), |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |     [\[8\]](api/languages/cp      |
| -   [cudaq::pauli1 (C++           | p_api.html#_CPPv4NO5cudaq15scalar |
|     class)](api/languages/cp      | _operatordvERK15scalar_operator), |
| p_api.html#_CPPv4N5cudaq6pauli1E) |     [\[9\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|    [cudaq::pauli1::num_parameters | Pv4NO5cudaq15scalar_operatordvEd) |
|     (C++                          | -   [c                            |
|     member)]                      | udaq::scalar_operator::operator/= |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq6pauli114num_parametersE) |     function)](api/languag        |
| -   [cudaq::pauli1::num_targets   | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatordVENSt7complexIdEE), |
|     membe                         |     [\[1\]](api/languages/c       |
| r)](api/languages/cpp_api.html#_C | pp_api.html#_CPPv4N5cudaq15scalar |
| PPv4N5cudaq6pauli111num_targetsE) | _operatordVERK15scalar_operator), |
| -   [cudaq::pauli1::pauli1 (C++   |     [\[2                          |
|     function)](api/languages/cpp_ | \]](api/languages/cpp_api.html#_C |
| api.html#_CPPv4N5cudaq6pauli16pau | PPv4N5cudaq15scalar_operatordVEd) |
| li1ERKNSt6vectorIN5cudaq4realEEE) | -   [                             |
| -   [cudaq::pauli2 (C++           | cudaq::scalar_operator::operator= |
|     class)](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq6pauli2E) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4N5cudaq15scalar |
|    [cudaq::pauli2::num_parameters | _operatoraSERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/languages/        |
|     member)]                      | cpp_api.html#_CPPv4N5cudaq15scala |
| (api/languages/cpp_api.html#_CPPv | r_operatoraSERR15scalar_operator) |
| 4N5cudaq6pauli214num_parametersE) | -   [c                            |
| -   [cudaq::pauli2::num_targets   | udaq::scalar_operator::operator== |
|     (C++                          |     (C++                          |
|     membe                         |     function)](api/languages/c    |
| r)](api/languages/cpp_api.html#_C | pp_api.html#_CPPv4NK5cudaq15scala |
| PPv4N5cudaq6pauli211num_targetsE) | r_operatoreqERK15scalar_operator) |
| -   [cudaq::pauli2::pauli2 (C++   | -   [cudaq:                       |
|     function)](api/languages/cpp_ | :scalar_operator::scalar_operator |
| api.html#_CPPv4N5cudaq6pauli26pau |     (C++                          |
| li2ERKNSt6vectorIN5cudaq4realEEE) |     func                          |
| -   [cudaq::phase_damping (C++    | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq15scalar_operator15 |
|  class)](api/languages/cpp_api.ht | scalar_operatorENSt7complexIdEE), |
| ml#_CPPv4N5cudaq13phase_dampingE) |     [\[1\]](api/langu             |
| -   [cud                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
| aq::phase_damping::num_parameters | scalar_operator15scalar_operatorE |
|     (C++                          | RK15scalar_callbackRRNSt13unorder |
|     member)](api/lan              | ed_mapINSt6stringENSt6stringEEE), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[2\                         |
| 13phase_damping14num_parametersE) | ]](api/languages/cpp_api.html#_CP |
| -   [                             | Pv4N5cudaq15scalar_operator15scal |
| cudaq::phase_damping::num_targets | ar_operatorERK15scalar_operator), |
|     (C++                          |     [\[3\]](api/langu             |
|     member)](api/                 | ages/cpp_api.html#_CPPv4N5cudaq15 |
| languages/cpp_api.html#_CPPv4N5cu | scalar_operator15scalar_operatorE |
| daq13phase_damping11num_targetsE) | RR15scalar_callbackRRNSt13unorder |
| -   [cudaq::phase_flip_channel    | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[4\                         |
|     clas                          | ]](api/languages/cpp_api.html#_CP |
| s)](api/languages/cpp_api.html#_C | Pv4N5cudaq15scalar_operator15scal |
| PPv4N5cudaq18phase_flip_channelE) | ar_operatorERR15scalar_operator), |
| -   [cudaq::p                     |     [\[5\]](api/language          |
| hase_flip_channel::num_parameters | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     (C++                          | lar_operator15scalar_operatorEd), |
|     member)](api/language         |     [\[6\]](api/languag           |
| s/cpp_api.html#_CPPv4N5cudaq18pha | es/cpp_api.html#_CPPv4N5cudaq15sc |
| se_flip_channel14num_parametersE) | alar_operator15scalar_operatorEv) |
| -   [cudaq                        | -   [                             |
| ::phase_flip_channel::num_targets | cudaq::scalar_operator::to_matrix |
|     (C++                          |     (C++                          |
|     member)](api/langu            |                                   |
| ages/cpp_api.html#_CPPv4N5cudaq18 |   function)](api/languages/cpp_ap |
| phase_flip_channel11num_targetsE) | i.html#_CPPv4NK5cudaq15scalar_ope |
| -   [cudaq::product_op (C++       | rator9to_matrixERKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|  class)](api/languages/cpp_api.ht | -   [                             |
| ml#_CPPv4I0EN5cudaq10product_opE) | cudaq::scalar_operator::to_string |
| -   [cudaq::product_op::begin     |     (C++                          |
|     (C++                          |     function)](api/l              |
|     functio                       | anguages/cpp_api.html#_CPPv4NK5cu |
| n)](api/languages/cpp_api.html#_C | daq15scalar_operator9to_stringEv) |
| PPv4NK5cudaq10product_op5beginEv) | -   [cudaq::s                     |
| -                                 | calar_operator::\~scalar_operator |
|  [cudaq::product_op::canonicalize |     (C++                          |
|     (C++                          |     functio                       |
|     func                          | n)](api/languages/cpp_api.html#_C |
| tion)](api/languages/cpp_api.html | PPv4N5cudaq15scalar_operatorD0Ev) |
| #_CPPv4N5cudaq10product_op12canon | -   [cudaq::set_noise (C++        |
| icalizeERKNSt3setINSt6size_tEEE), |     function)](api/langu          |
|     [\[1\]](api                   | ages/cpp_api.html#_CPPv4N5cudaq9s |
| /languages/cpp_api.html#_CPPv4N5c | et_noiseERKN5cudaq11noise_modelE) |
| udaq10product_op12canonicalizeEv) | -   [cudaq::set_random_seed (C++  |
| -   [                             |     function)](api/               |
| cudaq::product_op::const_iterator | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq15set_random_seedENSt6size_tE) |
|     struct)](api/                 | -   [cudaq::simulation_precision  |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq10product_op14const_iteratorE) |     enum)                         |
| -   [cudaq::product_o             | ](api/languages/cpp_api.html#_CPP |
| p::const_iterator::const_iterator | v4N5cudaq20simulation_precisionE) |
|     (C++                          | -   [                             |
|     fu                            | cudaq::simulation_precision::fp32 |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq10product_op14con |     enumerator)](api              |
| st_iterator14const_iteratorEPK10p | /languages/cpp_api.html#_CPPv4N5c |
| roduct_opI9HandlerTyENSt6size_tE) | udaq20simulation_precision4fp32E) |
| -   [cudaq::produ                 | -   [                             |
| ct_op::const_iterator::operator!= | cudaq::simulation_precision::fp64 |
|     (C++                          |     (C++                          |
|     fun                           |     enumerator)](api              |
| ction)](api/languages/cpp_api.htm | /languages/cpp_api.html#_CPPv4N5c |
| l#_CPPv4NK5cudaq10product_op14con | udaq20simulation_precision4fp64E) |
| st_iteratorneERK14const_iterator) | -   [cudaq::SimulationState (C++  |
| -   [cudaq::produ                 |     c                             |
| ct_op::const_iterator::operator\* | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15SimulationStateE) |
|     function)](api/lang           | -   [                             |
| uages/cpp_api.html#_CPPv4NK5cudaq | cudaq::SimulationState::precision |
| 10product_op14const_iteratormlEv) |     (C++                          |
| -   [cudaq::produ                 |     enum)](api                    |
| ct_op::const_iterator::operator++ | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq15SimulationState9precisionE) |
|     function)](api/lang           | -   [cudaq:                       |
| uages/cpp_api.html#_CPPv4N5cudaq1 | :SimulationState::precision::fp32 |
| 0product_op14const_iteratorppEi), |     (C++                          |
|     [\[1\]](api/lan               |     enumerator)](api/lang         |
| guages/cpp_api.html#_CPPv4N5cudaq | uages/cpp_api.html#_CPPv4N5cudaq1 |
| 10product_op14const_iteratorppEv) | 5SimulationState9precision4fp32E) |
| -   [cudaq::produc                | -   [cudaq:                       |
| t_op::const_iterator::operator\-- | :SimulationState::precision::fp64 |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     enumerator)](api/lang         |
| uages/cpp_api.html#_CPPv4N5cudaq1 | uages/cpp_api.html#_CPPv4N5cudaq1 |
| 0product_op14const_iteratormmEi), | 5SimulationState9precision4fp64E) |
|     [\[1\]](api/lan               | -                                 |
| guages/cpp_api.html#_CPPv4N5cudaq |   [cudaq::SimulationState::Tensor |
| 10product_op14const_iteratormmEv) |     (C++                          |
| -   [cudaq::produc                |     struct)](                     |
| t_op::const_iterator::operator-\> | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq15SimulationState6TensorE) |
|     function)](api/lan            | -   [cudaq::spin_handler (C++     |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 10product_op14const_iteratorptEv) |   class)](api/languages/cpp_api.h |
| -   [cudaq::produ                 | tml#_CPPv4N5cudaq12spin_handlerE) |
| ct_op::const_iterator::operator== | -   [cudaq:                       |
|     (C++                          | :spin_handler::to_diagonal_matrix |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     function)](api/la             |
| l#_CPPv4NK5cudaq10product_op14con | nguages/cpp_api.html#_CPPv4NK5cud |
| st_iteratoreqERK14const_iterator) | aq12spin_handler18to_diagonal_mat |
|                                   | rixERNSt13unordered_mapINSt6size_ |
|                                   | tENSt7int64_tEEERKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -                                 |
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
