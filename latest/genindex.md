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
| -   [cachedCompiledModule()       | -   [cudaq::produc                |
|     (cudaq.PyKernelDecorator      | t_op::const_iterator::operator-\> |
|     method)](api/langu            |     (C++                          |
| ages/python_api.html#cudaq.PyKern |     function)](api/lan            |
| elDecorator.cachedCompiledModule) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [canonicalize                 | 10product_op14const_iteratorptEv) |
|     (cu                           | -   [cudaq::produ                 |
| daq.operators.boson.BosonOperator | ct_op::const_iterator::operator== |
|     attribute)](api/languages     |     (C++                          |
| /python_api.html#cudaq.operators. |     fun                           |
| boson.BosonOperator.canonicalize) | ction)](api/languages/cpp_api.htm |
|     -   [(cudaq.                  | l#_CPPv4NK5cudaq10product_op14con |
| operators.boson.BosonOperatorTerm | st_iteratoreqERK14const_iterator) |
|                                   | -   [cudaq::product_op::degrees   |
|     attribute)](api/languages/pyt |     (C++                          |
| hon_api.html#cudaq.operators.boso |     function)                     |
| n.BosonOperatorTerm.canonicalize) | ](api/languages/cpp_api.html#_CPP |
|     -   [(cudaq.                  | v4NK5cudaq10product_op7degreesEv) |
| operators.fermion.FermionOperator | -   [cudaq::product_op::dump (C++ |
|                                   |     functi                        |
|     attribute)](api/languages/pyt | on)](api/languages/cpp_api.html#_ |
| hon_api.html#cudaq.operators.ferm | CPPv4NK5cudaq10product_op4dumpEv) |
| ion.FermionOperator.canonicalize) | -   [cudaq::product_op::end (C++  |
|     -   [(cudaq.oper              |     funct                         |
| ators.fermion.FermionOperatorTerm | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4NK5cudaq10product_op3endEv) |
| attribute)](api/languages/python_ | -   [c                            |
| api.html#cudaq.operators.fermion. | udaq::product_op::get_coefficient |
| FermionOperatorTerm.canonicalize) |     (C++                          |
|     -                             |     function)](api/lan            |
|  [(cudaq.operators.MatrixOperator | guages/cpp_api.html#_CPPv4NK5cuda |
|         attribute)](api/lang      | q10product_op15get_coefficientEv) |
| uages/python_api.html#cudaq.opera | -                                 |
| tors.MatrixOperator.canonicalize) |   [cudaq::product_op::get_term_id |
|     -   [(c                       |     (C++                          |
| udaq.operators.MatrixOperatorTerm |     function)](api                |
|         attribute)](api/language  | /languages/cpp_api.html#_CPPv4NK5 |
| s/python_api.html#cudaq.operators | cudaq10product_op11get_term_idEv) |
| .MatrixOperatorTerm.canonicalize) | -                                 |
|     -   [(                        |   [cudaq::product_op::is_identity |
| cudaq.operators.spin.SpinOperator |     (C++                          |
|         attribute)](api/languag   |     function)](api                |
| es/python_api.html#cudaq.operator | /languages/cpp_api.html#_CPPv4NK5 |
| s.spin.SpinOperator.canonicalize) | cudaq10product_op11is_identityEv) |
|     -   [(cuda                    | -   [cudaq::product_op::num_ops   |
| q.operators.spin.SpinOperatorTerm |     (C++                          |
|                                   |     function)                     |
|       attribute)](api/languages/p | ](api/languages/cpp_api.html#_CPP |
| ython_api.html#cudaq.operators.sp | v4NK5cudaq10product_op7num_opsEv) |
| in.SpinOperatorTerm.canonicalize) | -                                 |
| -   [captured_variables()         |    [cudaq::product_op::operator\* |
|     (cudaq.PyKernelDecorator      |     (C++                          |
|     method)](api/lan              |     function)](api/languages/     |
| guages/python_api.html#cudaq.PyKe | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| rnelDecorator.captured_variables) | oduct_opmlE10product_opI1TERK15sc |
| -   [CentralDifference (class in  | alar_operatorRK10product_opI1TE), |
|     cudaq.gradients)              |     [\[1\]](api/languages/        |
| ](api/languages/python_api.html#c | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| udaq.gradients.CentralDifference) | oduct_opmlE10product_opI1TERK15sc |
| -   [channel                      | alar_operatorRR10product_opI1TE), |
|     (cudaq.ptsbe.TraceInstruction |     [\[2\]](api/languages/        |
|     property)](a                  | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| pi/languages/python_api.html#cuda | oduct_opmlE10product_opI1TERR15sc |
| q.ptsbe.TraceInstruction.channel) | alar_operatorRK10product_opI1TE), |
| -   [circuit_location             |     [\[3\]](api/languages/        |
|     (cudaq.ptsbe.KrausSelection   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     property)](api/lang           | oduct_opmlE10product_opI1TERR15sc |
| uages/python_api.html#cudaq.ptsbe | alar_operatorRR10product_opI1TE), |
| .KrausSelection.circuit_location) |     [\[4\]](api/                  |
| -   [clear (cudaq.Resources       | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opmlE6sum_opI1TER |
|   attribute)](api/languages/pytho | K15scalar_operatorRK6sum_opI1TE), |
| n_api.html#cudaq.Resources.clear) |     [\[5\]](api/                  |
|     -   [(cudaq.SampleResult      | languages/cpp_api.html#_CPPv4I0EN |
|         a                         | 5cudaq10product_opmlE6sum_opI1TER |
| ttribute)](api/languages/python_a | K15scalar_operatorRR6sum_opI1TE), |
| pi.html#cudaq.SampleResult.clear) |     [\[6\]](api/                  |
| -   [COBYLA (class in             | languages/cpp_api.html#_CPPv4I0EN |
|     cudaq.o                       | 5cudaq10product_opmlE6sum_opI1TER |
| ptimizers)](api/languages/python_ | R15scalar_operatorRK6sum_opI1TE), |
| api.html#cudaq.optimizers.COBYLA) |     [\[7\]](api/                  |
| -   [coefficient                  | languages/cpp_api.html#_CPPv4I0EN |
|     (cudaq.                       | 5cudaq10product_opmlE6sum_opI1TER |
| operators.boson.BosonOperatorTerm | R15scalar_operatorRR6sum_opI1TE), |
|     property)](api/languages/py   |     [\[8\]](api/languages         |
| thon_api.html#cudaq.operators.bos | /cpp_api.html#_CPPv4NK5cudaq10pro |
| on.BosonOperatorTerm.coefficient) | duct_opmlERK6sum_opI9HandlerTyE), |
|     -   [(cudaq.oper              |     [\[9\]](api/languages/cpp_a   |
| ators.fermion.FermionOperatorTerm | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opmlERK10product_opI9HandlerTyE), |
|   property)](api/languages/python |     [\[10\]](api/language         |
| _api.html#cudaq.operators.fermion | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| .FermionOperatorTerm.coefficient) | roduct_opmlERK15scalar_operator), |
|     -   [(c                       |     [\[11\]](api/languages/cpp_a  |
| udaq.operators.MatrixOperatorTerm | pi.html#_CPPv4NKR5cudaq10product_ |
|         property)](api/languag    | opmlERR10product_opI9HandlerTyE), |
| es/python_api.html#cudaq.operator |     [\[12\]](api/language         |
| s.MatrixOperatorTerm.coefficient) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [(cuda                    | roduct_opmlERR15scalar_operator), |
| q.operators.spin.SpinOperatorTerm |     [\[13\]](api/languages/cpp_   |
|         property)](api/languages/ | api.html#_CPPv4NO5cudaq10product_ |
| python_api.html#cudaq.operators.s | opmlERK10product_opI9HandlerTyE), |
| pin.SpinOperatorTerm.coefficient) |     [\[14\]](api/languag          |
| -   [col_count                    | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (cudaq.KrausOperator          | roduct_opmlERK15scalar_operator), |
|     prope                         |     [\[15\]](api/languages/cpp_   |
| rty)](api/languages/python_api.ht | api.html#_CPPv4NO5cudaq10product_ |
| ml#cudaq.KrausOperator.col_count) | opmlERR10product_opI9HandlerTyE), |
| -   [compile()                    |     [\[16\]](api/langua           |
|     (cudaq.PyKernelDecorator      | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     metho                         | product_opmlERR15scalar_operator) |
| d)](api/languages/python_api.html | -                                 |
| #cudaq.PyKernelDecorator.compile) |   [cudaq::product_op::operator\*= |
| -   [ComplexMatrix (class in      |     (C++                          |
|     cudaq)](api/languages/pyt     |     function)](api/languages/cpp  |
| hon_api.html#cudaq.ComplexMatrix) | _api.html#_CPPv4N5cudaq10product_ |
| -   [compute                      | opmLERK10product_opI9HandlerTyE), |
|     (                             |     [\[1\]](api/langua            |
| cudaq.gradients.CentralDifference | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     attribute)](api/la            | roduct_opmLERK15scalar_operator), |
| nguages/python_api.html#cudaq.gra |     [\[2\]](api/languages/cp      |
| dients.CentralDifference.compute) | p_api.html#_CPPv4N5cudaq10product |
|     -   [(                        | _opmLERR10product_opI9HandlerTyE) |
| cudaq.gradients.ForwardDifference | -   [cudaq::product_op::operator+ |
|         attribute)](api/la        |     (C++                          |
| nguages/python_api.html#cudaq.gra |     function)](api/langu          |
| dients.ForwardDifference.compute) | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     -                             | q10product_opplE6sum_opI1TERK15sc |
|  [(cudaq.gradients.ParameterShift | alar_operatorRK10product_opI1TE), |
|         attribute)](api           |     [\[1\]](api/                  |
| /languages/python_api.html#cudaq. | languages/cpp_api.html#_CPPv4I0EN |
| gradients.ParameterShift.compute) | 5cudaq10product_opplE6sum_opI1TER |
| -   [const()                      | K15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[2\]](api/langu             |
|   (cudaq.operators.ScalarOperator | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     class                         | q10product_opplE6sum_opI1TERK15sc |
|     method)](a                    | alar_operatorRR10product_opI1TE), |
| pi/languages/python_api.html#cuda |     [\[3\]](api/                  |
| q.operators.ScalarOperator.const) | languages/cpp_api.html#_CPPv4I0EN |
| -   [controls                     | 5cudaq10product_opplE6sum_opI1TER |
|     (cudaq.ptsbe.TraceInstruction | K15scalar_operatorRR6sum_opI1TE), |
|     property)](ap                 |     [\[4\]](api/langu             |
| i/languages/python_api.html#cudaq | ages/cpp_api.html#_CPPv4I0EN5cuda |
| .ptsbe.TraceInstruction.controls) | q10product_opplE6sum_opI1TERR15sc |
| -   [copy                         | alar_operatorRK10product_opI1TE), |
|     (cu                           |     [\[5\]](api/                  |
| daq.operators.boson.BosonOperator | languages/cpp_api.html#_CPPv4I0EN |
|     attribute)](api/l             | 5cudaq10product_opplE6sum_opI1TER |
| anguages/python_api.html#cudaq.op | R15scalar_operatorRK6sum_opI1TE), |
| erators.boson.BosonOperator.copy) |     [\[6\]](api/langu             |
|     -   [(cudaq.                  | ages/cpp_api.html#_CPPv4I0EN5cuda |
| operators.boson.BosonOperatorTerm | q10product_opplE6sum_opI1TERR15sc |
|         attribute)](api/langu     | alar_operatorRR10product_opI1TE), |
| ages/python_api.html#cudaq.operat |     [\[7\]](api/                  |
| ors.boson.BosonOperatorTerm.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cudaq.                  | 5cudaq10product_opplE6sum_opI1TER |
| operators.fermion.FermionOperator | R15scalar_operatorRR6sum_opI1TE), |
|         attribute)](api/langu     |     [\[8\]](api/languages/cpp_a   |
| ages/python_api.html#cudaq.operat | pi.html#_CPPv4NKR5cudaq10product_ |
| ors.fermion.FermionOperator.copy) | opplERK10product_opI9HandlerTyE), |
|     -   [(cudaq.oper              |     [\[9\]](api/language          |
| ators.fermion.FermionOperatorTerm | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|         attribute)](api/languages | roduct_opplERK15scalar_operator), |
| /python_api.html#cudaq.operators. |     [\[10\]](api/languages/       |
| fermion.FermionOperatorTerm.copy) | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     -                             | duct_opplERK6sum_opI9HandlerTyE), |
|  [(cudaq.operators.MatrixOperator |     [\[11\]](api/languages/cpp_a  |
|         attribute)](              | pi.html#_CPPv4NKR5cudaq10product_ |
| api/languages/python_api.html#cud | opplERR10product_opI9HandlerTyE), |
| aq.operators.MatrixOperator.copy) |     [\[12\]](api/language         |
|     -   [(c                       | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| udaq.operators.MatrixOperatorTerm | roduct_opplERR15scalar_operator), |
|         attribute)](api/          |     [\[13\]](api/languages/       |
| languages/python_api.html#cudaq.o | cpp_api.html#_CPPv4NKR5cudaq10pro |
| perators.MatrixOperatorTerm.copy) | duct_opplERR6sum_opI9HandlerTyE), |
|     -   [(                        |     [\[                           |
| cudaq.operators.spin.SpinOperator | 14\]](api/languages/cpp_api.html# |
|         attribute)](api           | _CPPv4NKR5cudaq10product_opplEv), |
| /languages/python_api.html#cudaq. |     [\[15\]](api/languages/cpp_   |
| operators.spin.SpinOperator.copy) | api.html#_CPPv4NO5cudaq10product_ |
|     -   [(cuda                    | opplERK10product_opI9HandlerTyE), |
| q.operators.spin.SpinOperatorTerm |     [\[16\]](api/languag          |
|         attribute)](api/lan       | es/cpp_api.html#_CPPv4NO5cudaq10p |
| guages/python_api.html#cudaq.oper | roduct_opplERK15scalar_operator), |
| ators.spin.SpinOperatorTerm.copy) |     [\[17\]](api/languages        |
| -   [count (cudaq.Resources       | /cpp_api.html#_CPPv4NO5cudaq10pro |
|                                   | duct_opplERK6sum_opI9HandlerTyE), |
|   attribute)](api/languages/pytho |     [\[18\]](api/languages/cpp_   |
| n_api.html#cudaq.Resources.count) | api.html#_CPPv4NO5cudaq10product_ |
|     -   [(cudaq.SampleResult      | opplERR10product_opI9HandlerTyE), |
|         a                         |     [\[19\]](api/languag          |
| ttribute)](api/languages/python_a | es/cpp_api.html#_CPPv4NO5cudaq10p |
| pi.html#cudaq.SampleResult.count) | roduct_opplERR15scalar_operator), |
| -   [count_controls               |     [\[20\]](api/languages        |
|     (cudaq.Resources              | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     attribu                       | duct_opplERR6sum_opI9HandlerTyE), |
| te)](api/languages/python_api.htm |     [                             |
| l#cudaq.Resources.count_controls) | \[21\]](api/languages/cpp_api.htm |
| -   [count_instructions           | l#_CPPv4NO5cudaq10product_opplEv) |
|                                   | -   [cudaq::product_op::operator- |
|   (cudaq.ptsbe.PTSBEExecutionData |     (C++                          |
|     attribute)](api/languages/    |     function)](api/langu          |
| python_api.html#cudaq.ptsbe.PTSBE | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ExecutionData.count_instructions) | q10product_opmiE6sum_opI1TERK15sc |
| -   [counts (cudaq.ObserveResult  | alar_operatorRK10product_opI1TE), |
|     att                           |     [\[1\]](api/                  |
| ribute)](api/languages/python_api | languages/cpp_api.html#_CPPv4I0EN |
| .html#cudaq.ObserveResult.counts) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [csr_spmatrix (C++            | K15scalar_operatorRK6sum_opI1TE), |
|     type)](api/languages/c        |     [\[2\]](api/langu             |
| pp_api.html#_CPPv412csr_spmatrix) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq                         | q10product_opmiE6sum_opI1TERK15sc |
|     -   [module](api/langua       | alar_operatorRR10product_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[3\]](api/                  |
| -   [cudaq (C++                   | languages/cpp_api.html#_CPPv4I0EN |
|     type)](api/lan                | 5cudaq10product_opmiE6sum_opI1TER |
| guages/cpp_api.html#_CPPv45cudaq) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq.apply_noise() (in      |     [\[4\]](api/langu             |
|     module                        | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cudaq)](api/languages/python_ | q10product_opmiE6sum_opI1TERR15sc |
| api.html#cudaq.cudaq.apply_noise) | alar_operatorRK10product_opI1TE), |
| -   cudaq.boson                   |     [\[5\]](api/                  |
|     -   [module](api/languages/py | languages/cpp_api.html#_CPPv4I0EN |
| thon_api.html#module-cudaq.boson) | 5cudaq10product_opmiE6sum_opI1TER |
| -   cudaq.fermion                 | R15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[6\]](api/langu             |
|   -   [module](api/languages/pyth | ages/cpp_api.html#_CPPv4I0EN5cuda |
| on_api.html#module-cudaq.fermion) | q10product_opmiE6sum_opI1TERR15sc |
| -   cudaq.operators.custom        | alar_operatorRR10product_opI1TE), |
|     -   [mo                       |     [\[7\]](api/                  |
| dule](api/languages/python_api.ht | languages/cpp_api.html#_CPPv4I0EN |
| ml#module-cudaq.operators.custom) | 5cudaq10product_opmiE6sum_opI1TER |
| -   cudaq.spin                    | R15scalar_operatorRR6sum_opI1TE), |
|     -   [module](api/languages/p  |     [\[8\]](api/languages/cpp_a   |
| ython_api.html#module-cudaq.spin) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [cudaq::amplitude_damping     | opmiERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[9\]](api/language          |
|     cla                           | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ss)](api/languages/cpp_api.html#_ | roduct_opmiERK15scalar_operator), |
| CPPv4N5cudaq17amplitude_dampingE) |     [\[10\]](api/languages/       |
| -                                 | cpp_api.html#_CPPv4NKR5cudaq10pro |
| [cudaq::amplitude_damping_channel | duct_opmiERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     class)](api                   | pi.html#_CPPv4NKR5cudaq10product_ |
| /languages/cpp_api.html#_CPPv4N5c | opmiERR10product_opI9HandlerTyE), |
| udaq25amplitude_damping_channelE) |     [\[12\]](api/language         |
| -   [cudaq::amplitud              | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| e_damping_channel::num_parameters | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     member)](api/languages/cpp_a  | cpp_api.html#_CPPv4NKR5cudaq10pro |
| pi.html#_CPPv4N5cudaq25amplitude_ | duct_opmiERR6sum_opI9HandlerTyE), |
| damping_channel14num_parametersE) |     [\[                           |
| -   [cudaq::ampli                 | 14\]](api/languages/cpp_api.html# |
| tude_damping_channel::num_targets | _CPPv4NKR5cudaq10product_opmiEv), |
|     (C++                          |     [\[15\]](api/languages/cpp_   |
|     member)](api/languages/cp     | api.html#_CPPv4NO5cudaq10product_ |
| p_api.html#_CPPv4N5cudaq25amplitu | opmiERK10product_opI9HandlerTyE), |
| de_damping_channel11num_targetsE) |     [\[16\]](api/languag          |
| -   [cudaq::AnalogRemoteRESTQPU   | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmiERK15scalar_operator), |
|     class                         |     [\[17\]](api/languages        |
| )](api/languages/cpp_api.html#_CP | /cpp_api.html#_CPPv4NO5cudaq10pro |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::apply_noise (C++      |     [\[18\]](api/languages/cpp_   |
|     function)](api/               | api.html#_CPPv4NO5cudaq10product_ |
| languages/cpp_api.html#_CPPv4I0Dp | opmiERR10product_opI9HandlerTyE), |
| EN5cudaq11apply_noiseEvDpRR4Args) |     [\[19\]](api/languag          |
| -   [cudaq::async_result (C++     | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     c                             | roduct_opmiERR15scalar_operator), |
| lass)](api/languages/cpp_api.html |     [\[20\]](api/languages        |
| #_CPPv4I0EN5cudaq12async_resultE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::async_result::get     | duct_opmiERR6sum_opI9HandlerTyE), |
|     (C++                          |     [                             |
|     functi                        | \[21\]](api/languages/cpp_api.htm |
| on)](api/languages/cpp_api.html#_ | l#_CPPv4NO5cudaq10product_opmiEv) |
| CPPv4N5cudaq12async_result3getEv) | -   [cudaq::product_op::operator/ |
| -   [cudaq::async_sample_result   |     (C++                          |
|     (C++                          |     function)](api/language       |
|     type                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| )](api/languages/cpp_api.html#_CP | roduct_opdvERK15scalar_operator), |
| Pv4N5cudaq19async_sample_resultE) |     [\[1\]](api/language          |
| -   [cudaq::BaseRemoteRESTQPU     | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opdvERR15scalar_operator), |
|     cla                           |     [\[2\]](api/languag           |
| ss)](api/languages/cpp_api.html#_ | es/cpp_api.html#_CPPv4NO5cudaq10p |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | roduct_opdvERK15scalar_operator), |
| -   [cudaq::bit_flip_channel (C++ |     [\[3\]](api/langua            |
|     cl                            | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| ass)](api/languages/cpp_api.html# | product_opdvERR15scalar_operator) |
| _CPPv4N5cudaq16bit_flip_channelE) | -                                 |
| -   [cudaq:                       |    [cudaq::product_op::operator/= |
| :bit_flip_channel::num_parameters |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     member)](api/langua           | ages/cpp_api.html#_CPPv4N5cudaq10 |
| ges/cpp_api.html#_CPPv4N5cudaq16b | product_opdVERK15scalar_operator) |
| it_flip_channel14num_parametersE) | -   [cudaq::product_op::operator= |
| -   [cud                          |     (C++                          |
| aq::bit_flip_channel::num_targets |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4I00EN |
|     member)](api/lan              | 5cudaq10product_opaSER10product_o |
| guages/cpp_api.html#_CPPv4N5cudaq | pI9HandlerTyERK10product_opI1TE), |
| 16bit_flip_channel11num_targetsE) |     [\[1\]](api/languages/cpp     |
| -   [cudaq::boson_handler (C++    | _api.html#_CPPv4N5cudaq10product_ |
|                                   | opaSERK10product_opI9HandlerTyE), |
|  class)](api/languages/cpp_api.ht |     [\[2\]](api/languages/cp      |
| ml#_CPPv4N5cudaq13boson_handlerE) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::boson_op (C++         | _opaSERR10product_opI9HandlerTyE) |
|     type)](api/languages/cpp_     | -                                 |
| api.html#_CPPv4N5cudaq8boson_opE) |    [cudaq::product_op::operator== |
| -   [cudaq::boson_op_term (C++    |     (C++                          |
|                                   |     function)](api/languages/cpp  |
|   type)](api/languages/cpp_api.ht | _api.html#_CPPv4NK5cudaq10product |
| ml#_CPPv4N5cudaq13boson_op_termE) | _opeqERK10product_opI9HandlerTyE) |
| -   [cudaq::CodeGenConfig (C++    | -                                 |
|                                   |  [cudaq::product_op::operator\[\] |
| struct)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     function)](ap                 |
| -   [cudaq::commutation_relations | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq10product_opixENSt6size_tE) |
|     struct)]                      | -                                 |
| (api/languages/cpp_api.html#_CPPv |    [cudaq::product_op::product_op |
| 4N5cudaq21commutation_relationsE) |     (C++                          |
| -   [cudaq::complex (C++          |     f                             |
|     type)](api/languages/cpp      | unction)](api/languages/cpp_api.h |
| _api.html#_CPPv4N5cudaq7complexE) | tml#_CPPv4I00EN5cudaq10product_op |
| -   [cudaq::complex_matrix (C++   | 10product_opERK10product_opI1TE), |
|                                   |     [\[1\]]                       |
| class)](api/languages/cpp_api.htm | (api/languages/cpp_api.html#_CPPv |
| l#_CPPv4N5cudaq14complex_matrixE) | 4I00EN5cudaq10product_op10product |
| -                                 | _opERK10product_opI1TERKN14matrix |
|   [cudaq::complex_matrix::adjoint | _handler20commutation_behaviorE), |
|     (C++                          |                                   |
|     function)](a                  |   [\[2\]](api/languages/cpp_api.h |
| pi/languages/cpp_api.html#_CPPv4N | tml#_CPPv4N5cudaq10product_op10pr |
| 5cudaq14complex_matrix7adjointEv) | oduct_opENSt6size_tENSt6size_tE), |
| -   [cudaq::                      |     [\[3\]](api/languages/cp      |
| complex_matrix::diagonal_elements | p_api.html#_CPPv4N5cudaq10product |
|     (C++                          | _op10product_opENSt7complexIdEE), |
|     function)](api/languages      |     [\[4\]](api/l                 |
| /cpp_api.html#_CPPv4NK5cudaq14com | anguages/cpp_api.html#_CPPv4N5cud |
| plex_matrix17diagonal_elementsEi) | aq10product_op10product_opERK10pr |
| -   [cudaq::complex_matrix::dump  | oduct_opI9HandlerTyENSt6size_tE), |
|     (C++                          |     [\[5\]](api/l                 |
|     function)](api/language       | anguages/cpp_api.html#_CPPv4N5cud |
| s/cpp_api.html#_CPPv4NK5cudaq14co | aq10product_op10product_opERR10pr |
| mplex_matrix4dumpERNSt7ostreamE), | oduct_opI9HandlerTyENSt6size_tE), |
|     [\[1\]]                       |     [\[6\]](api/languages         |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4N5cudaq10prod |
| 4NK5cudaq14complex_matrix4dumpEv) | uct_op10product_opERR9HandlerTy), |
| -   [c                            |     [\[7\]](ap                    |
| udaq::complex_matrix::eigenvalues | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq10product_op10product_opEd), |
|     function)](api/lan            |     [\[8\]](a                     |
| guages/cpp_api.html#_CPPv4NK5cuda | pi/languages/cpp_api.html#_CPPv4N |
| q14complex_matrix11eigenvaluesEv) | 5cudaq10product_op10product_opEv) |
| -   [cu                           | -   [cuda                         |
| daq::complex_matrix::eigenvectors | q::product_op::to_diagonal_matrix |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](api/               |
| uages/cpp_api.html#_CPPv4NK5cudaq | languages/cpp_api.html#_CPPv4NK5c |
| 14complex_matrix12eigenvectorsEv) | udaq10product_op18to_diagonal_mat |
| -   [c                            | rixENSt13unordered_mapINSt6size_t |
| udaq::complex_matrix::exponential | ENSt7int64_tEEERKNSt13unordered_m |
|     (C++                          | apINSt6stringENSt7complexIdEEEEb) |
|     function)](api/la             | -   [cudaq::product_op::to_matrix |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14complex_matrix11exponentialEv) |     funct                         |
| -                                 | ion)](api/languages/cpp_api.html# |
|  [cudaq::complex_matrix::identity | _CPPv4NK5cudaq10product_op9to_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function)](api/languages      | ENSt7int64_tEEERKNSt13unordered_m |
| /cpp_api.html#_CPPv4N5cudaq14comp | apINSt6stringENSt7complexIdEEEEb) |
| lex_matrix8identityEKNSt6size_tE) | -   [cu                           |
| -                                 | daq::product_op::to_sparse_matrix |
| [cudaq::complex_matrix::kronecker |     (C++                          |
|     (C++                          |     function)](ap                 |
|     function)](api/lang           | i/languages/cpp_api.html#_CPPv4NK |
| uages/cpp_api.html#_CPPv4I00EN5cu | 5cudaq10product_op16to_sparse_mat |
| daq14complex_matrix9kroneckerE14c | rixENSt13unordered_mapINSt6size_t |
| omplex_matrix8Iterable8Iterable), | ENSt7int64_tEEERKNSt13unordered_m |
|     [\[1\]](api/l                 | apINSt6stringENSt7complexIdEEEEb) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cudaq::product_op::to_string |
| aq14complex_matrix9kroneckerERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     function)](                   |
| -   [cudaq::c                     | api/languages/cpp_api.html#_CPPv4 |
| omplex_matrix::minimal_eigenvalue | NK5cudaq10product_op9to_stringEv) |
|     (C++                          | -                                 |
|     function)](api/languages/     |  [cudaq::product_op::\~product_op |
| cpp_api.html#_CPPv4NK5cudaq14comp |     (C++                          |
| lex_matrix18minimal_eigenvalueEv) |     fu                            |
| -   [                             | nction)](api/languages/cpp_api.ht |
| cudaq::complex_matrix::operator() | ml#_CPPv4N5cudaq10product_opD0Ev) |
|     (C++                          | -   [cudaq::ptsbe (C++            |
|     function)](api/languages/cpp  |     type)](api/languages/c        |
| _api.html#_CPPv4N5cudaq14complex_ | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| matrixclENSt6size_tENSt6size_tE), | -   [cudaq::p                     |
|     [\[1\]](api/languages/cpp     | tsbe::ConditionalSamplingStrategy |
| _api.html#_CPPv4NK5cudaq14complex |     (C++                          |
| _matrixclENSt6size_tENSt6size_tE) |     class)](api/languag           |
| -   [                             | es/cpp_api.html#_CPPv4N5cudaq5pts |
| cudaq::complex_matrix::operator\* | be27ConditionalSamplingStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::C              |
|     function)](api/langua         | onditionalSamplingStrategy::clone |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     (C++                          |
| omplex_matrixmlEN14complex_matrix |                                   |
| 10value_typeERK14complex_matrix), |    function)](api/languages/cpp_a |
|     [\[1\]                        | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| ](api/languages/cpp_api.html#_CPP | ditionalSamplingStrategy5cloneEv) |
| v4N5cudaq14complex_matrixmlERK14c | -   [cuda                         |
| omplex_matrixRK14complex_matrix), | q::ptsbe::ConditionalSamplingStra |
|                                   | tegy::ConditionalSamplingStrategy |
|  [\[2\]](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14complex_matrixm |     function)](api/lang           |
| lERK14complex_matrixRKNSt6vectorI | uages/cpp_api.html#_CPPv4N5cudaq5 |
| N14complex_matrix10value_typeEEE) | ptsbe27ConditionalSamplingStrateg |
| -                                 | y27ConditionalSamplingStrategyE19 |
| [cudaq::complex_matrix::operator+ | TrajectoryPredicateNSt8uint64_tE) |
|     (C++                          | -                                 |
|     function                      |   [cudaq::ptsbe::ConditionalSampl |
| )](api/languages/cpp_api.html#_CP | ingStrategy::generateTrajectories |
| Pv4N5cudaq14complex_matrixplERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     function)](api/language       |
| -                                 | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| [cudaq::complex_matrix::operator- | be27ConditionalSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     function                      | detail10NoisePointEEENSt6size_tE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::ptsbe::               |
| Pv4N5cudaq14complex_matrixmiERK14 | ConditionalSamplingStrategy::name |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -   [cu                           |     function)](api/languages/cpp_ |
| daq::complex_matrix::operator\[\] | api.html#_CPPv4NK5cudaq5ptsbe27Co |
|     (C++                          | nditionalSamplingStrategy4nameEv) |
|                                   | -   [cudaq:                       |
|  function)](api/languages/cpp_api | :ptsbe::ConditionalSamplingStrate |
| .html#_CPPv4N5cudaq14complex_matr | gy::\~ConditionalSamplingStrategy |
| ixixERKNSt6vectorINSt6size_tEEE), |     (C++                          |
|     [\[1\]](api/languages/cpp_api |     function)](api/languages/     |
| .html#_CPPv4NK5cudaq14complex_mat | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| rixixERKNSt6vectorINSt6size_tEEE) | 7ConditionalSamplingStrategyD0Ev) |
| -   [cudaq::complex_matrix::power | -                                 |
|     (C++                          | [cudaq::ptsbe::detail::NoisePoint |
|     function)]                    |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     struct)](a                    |
| 4N5cudaq14complex_matrix5powerEi) | pi/languages/cpp_api.html#_CPPv4N |
| -                                 | 5cudaq5ptsbe6detail10NoisePointE) |
|  [cudaq::complex_matrix::set_zero | -   [cudaq::p                     |
|     (C++                          | tsbe::detail::NoisePoint::channel |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     member)](api/langu            |
| cudaq14complex_matrix8set_zeroEv) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -                                 | tsbe6detail10NoisePoint7channelE) |
| [cudaq::complex_matrix::to_string | -   [cudaq::ptsbe::det            |
|     (C++                          | ail::NoisePoint::circuit_location |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     member)](api/languages/cpp_a  |
| udaq14complex_matrix9to_stringEv) | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| -   [                             | l10NoisePoint16circuit_locationE) |
| cudaq::complex_matrix::value_type | -   [cudaq::p                     |
|     (C++                          | tsbe::detail::NoisePoint::op_name |
|     type)](api/                   |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     member)](api/langu            |
| daq14complex_matrix10value_typeE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [cudaq::contrib (C++          | tsbe6detail10NoisePoint7op_nameE) |
|     type)](api/languages/cpp      | -   [cudaq::                      |
| _api.html#_CPPv4N5cudaq7contribE) | ptsbe::detail::NoisePoint::qubits |
| -   [cudaq::contrib::draw (C++    |     (C++                          |
|     function)                     |     member)](api/lang             |
| ](api/languages/cpp_api.html#_CPP | uages/cpp_api.html#_CPPv4N5cudaq5 |
| v4I0DpEN5cudaq7contrib4drawENSt6s | ptsbe6detail10NoisePoint6qubitsE) |
| tringERR13QuantumKernelDpRR4Args) | -   [cudaq::                      |
| -                                 | ptsbe::ExhaustiveSamplingStrategy |
| [cudaq::contrib::get_unitary_cmat |     (C++                          |
|     (C++                          |     class)](api/langua            |
|     function)](api/languages/cp   | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| p_api.html#_CPPv4I0DpEN5cudaq7con | sbe26ExhaustiveSamplingStrategyE) |
| trib16get_unitary_cmatE14complex_ | -   [cudaq::ptsbe::               |
| matrixRR13QuantumKernelDpRR4Args) | ExhaustiveSamplingStrategy::clone |
| -   [cudaq::CusvState (C++        |     (C++                          |
|                                   |     function)](api/languages/cpp_ |
|    class)](api/languages/cpp_api. | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| html#_CPPv4I0EN5cudaq9CusvStateE) | haustiveSamplingStrategy5cloneEv) |
| -   [cudaq::depolarization1 (C++  | -   [cu                           |
|     c                             | daq::ptsbe::ExhaustiveSamplingStr |
| lass)](api/languages/cpp_api.html | ategy::ExhaustiveSamplingStrategy |
| #_CPPv4N5cudaq15depolarization1E) |     (C++                          |
| -   [cudaq::depolarization2 (C++  |     function)](api/la             |
|     c                             | nguages/cpp_api.html#_CPPv4N5cuda |
| lass)](api/languages/cpp_api.html | q5ptsbe26ExhaustiveSamplingStrate |
| #_CPPv4N5cudaq15depolarization2E) | gy26ExhaustiveSamplingStrategyEv) |
| -   [cudaq:                       | -                                 |
| :depolarization2::depolarization2 |    [cudaq::ptsbe::ExhaustiveSampl |
|     (C++                          | ingStrategy::generateTrajectories |
|     function)](api/languages/cp   |     (C++                          |
| p_api.html#_CPPv4N5cudaq15depolar |     function)](api/languag        |
| ization215depolarization2EK4real) | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| -   [cudaq                        | sbe26ExhaustiveSamplingStrategy20 |
| ::depolarization2::num_parameters | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     member)](api/langu            | -   [cudaq::ptsbe:                |
| ages/cpp_api.html#_CPPv4N5cudaq15 | :ExhaustiveSamplingStrategy::name |
| depolarization214num_parametersE) |     (C++                          |
| -   [cu                           |     function)](api/languages/cpp  |
| daq::depolarization2::num_targets | _api.html#_CPPv4NK5cudaq5ptsbe26E |
|     (C++                          | xhaustiveSamplingStrategy4nameEv) |
|     member)](api/la               | -   [cuda                         |
| nguages/cpp_api.html#_CPPv4N5cuda | q::ptsbe::ExhaustiveSamplingStrat |
| q15depolarization211num_targetsE) | egy::\~ExhaustiveSamplingStrategy |
| -                                 |     (C++                          |
|    [cudaq::depolarization_channel |     function)](api/languages      |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     class)](                      | 26ExhaustiveSamplingStrategyD0Ev) |
| api/languages/cpp_api.html#_CPPv4 | -   [cuda                         |
| N5cudaq22depolarization_channelE) | q::ptsbe::OrderedSamplingStrategy |
| -   [cudaq::depol                 |     (C++                          |
| arization_channel::num_parameters |     class)](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     member)](api/languages/cp     | 5ptsbe23OrderedSamplingStrategyE) |
| p_api.html#_CPPv4N5cudaq22depolar | -   [cudaq::ptsb                  |
| ization_channel14num_parametersE) | e::OrderedSamplingStrategy::clone |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
|     member)](api/languages        | 3OrderedSamplingStrategy5cloneEv) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -   [cudaq::ptsbe::OrderedSampl   |
| larization_channel11num_targetsE) | ingStrategy::generateTrajectories |
| -   [cudaq::details (C++          |     (C++                          |
|     type)](api/languages/cpp      |     function)](api/lang           |
| _api.html#_CPPv4N5cudaq7detailsE) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cudaq::details::future (C++  | 5ptsbe23OrderedSamplingStrategy20 |
|                                   | generateTrajectoriesENSt4spanIKN6 |
|  class)](api/languages/cpp_api.ht | detail10NoisePointEEENSt6size_tE) |
| ml#_CPPv4N5cudaq7details6futureE) | -   [cudaq::pts                   |
| -                                 | be::OrderedSamplingStrategy::name |
|   [cudaq::details::future::future |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     functio                       | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| n)](api/languages/cpp_api.html#_C | 23OrderedSamplingStrategy4nameEv) |
| PPv4N5cudaq7details6future6future | -                                 |
| ERNSt6vectorI3JobEERNSt6stringERN |    [cudaq::ptsbe::OrderedSampling |
| St3mapINSt6stringENSt6stringEEE), | Strategy::OrderedSamplingStrategy |
|     [\[1\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     function)](                   |
| details6future6futureERR6future), | api/languages/cpp_api.html#_CPPv4 |
|     [\[2\]]                       | N5cudaq5ptsbe23OrderedSamplingStr |
| (api/languages/cpp_api.html#_CPPv | ategy23OrderedSamplingStrategyEv) |
| 4N5cudaq7details6future6futureEv) | -                                 |
| -   [cu                           |  [cudaq::ptsbe::OrderedSamplingSt |
| daq::details::kernel_builder_base | rategy::\~OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     class)](api/l                 |     function)](api/langua         |
| anguages/cpp_api.html#_CPPv4N5cud | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| aq7details19kernel_builder_baseE) | sbe23OrderedSamplingStrategyD0Ev) |
| -   [cudaq::details::             | -   [cudaq::pts                   |
| kernel_builder_base::operator\<\< | be::ProbabilisticSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     class)](api/languages         |
| ges/cpp_api.html#_CPPv4N5cudaq7de | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| tails19kernel_builder_baselsERNSt | 29ProbabilisticSamplingStrategyE) |
| 7ostreamERK19kernel_builder_base) | -   [cudaq::ptsbe::Pro            |
| -   [                             | babilisticSamplingStrategy::clone |
| cudaq::details::KernelBuilderType |     (C++                          |
|     (C++                          |                                   |
|     class)](api                   |  function)](api/languages/cpp_api |
| /languages/cpp_api.html#_CPPv4N5c | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| udaq7details17KernelBuilderTypeE) | bilisticSamplingStrategy5cloneEv) |
| -   [cudaq::d                     | -                                 |
| etails::KernelBuilderType::create | [cudaq::ptsbe::ProbabilisticSampl |
|     (C++                          | ingStrategy::generateTrajectories |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/     |
| v4N5cudaq7details17KernelBuilderT | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| ype6createEPN4mlir11MLIRContextE) | 29ProbabilisticSamplingStrategy20 |
| -   [cudaq::details::Ker          | generateTrajectoriesENSt4spanIKN6 |
| nelBuilderType::KernelBuilderType | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe::Pr             |
|     function)](api/lang           | obabilisticSamplingStrategy::name |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     (C++                          |
| details17KernelBuilderType17Kerne |                                   |
| lBuilderTypeERRNSt8functionIFN4ml |   function)](api/languages/cpp_ap |
| ir4TypeEPN4mlir11MLIRContextEEEE) | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| -   [cudaq::diag_matrix_callback  | abilisticSamplingStrategy4nameEv) |
|     (C++                          | -   [cudaq::p                     |
|     class)                        | tsbe::ProbabilisticSamplingStrate |
| ](api/languages/cpp_api.html#_CPP | gy::ProbabilisticSamplingStrategy |
| v4N5cudaq20diag_matrix_callbackE) |     (C++                          |
| -   [cudaq::dyn (C++              |     function)]                    |
|     member)](api/languages        | (api/languages/cpp_api.html#_CPPv |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | 4N5cudaq5ptsbe29ProbabilisticSamp |
| -   [cudaq::ExecutionContext (C++ | lingStrategy29ProbabilisticSampli |
|     cl                            | ngStrategyENSt8optionalINSt8uint6 |
| ass)](api/languages/cpp_api.html# | 4_tEEENSt8optionalINSt6size_tEEE) |
| _CPPv4N5cudaq16ExecutionContextE) | -   [cudaq::pts                   |
| -   [c                            | be::ProbabilisticSamplingStrategy |
| udaq::ExecutionContext::asyncExec | ::\~ProbabilisticSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/                 |     function)](api/languages/cp   |
| languages/cpp_api.html#_CPPv4N5cu | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| daq16ExecutionContext9asyncExecE) | robabilisticSamplingStrategyD0Ev) |
| -   [cud                          | -                                 |
| aq::ExecutionContext::asyncResult | [cudaq::ptsbe::PTSBEExecutionData |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     struct)](ap                   |
| guages/cpp_api.html#_CPPv4N5cudaq | i/languages/cpp_api.html#_CPPv4N5 |
| 16ExecutionContext11asyncResultE) | cudaq5ptsbe18PTSBEExecutionDataE) |
| -   [cudaq:                       | -   [cudaq::ptsbe::PTSBE          |
| :ExecutionContext::batchIteration | ExecutionData::count_instructions |
|     (C++                          |     (C++                          |
|     member)](api/langua           |     function)](api/l              |
| ges/cpp_api.html#_CPPv4N5cudaq16E | anguages/cpp_api.html#_CPPv4NK5cu |
| xecutionContext14batchIterationE) | daq5ptsbe18PTSBEExecutionData18co |
| -   [cudaq::E                     | unt_instructionsE20TraceInstructi |
| xecutionContext::canHandleObserve | onTypeNSt8optionalINSt6stringEEE) |
|     (C++                          | -   [cudaq::ptsbe::P              |
|     member)](api/language         | TSBEExecutionData::get_trajectory |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     (C++                          |
| cutionContext16canHandleObserveE) |     function                      |
| -   [cudaq::E                     | )](api/languages/cpp_api.html#_CP |
| xecutionContext::ExecutionContext | Pv4NK5cudaq5ptsbe18PTSBEExecution |
|     (C++                          | Data14get_trajectoryENSt6size_tE) |
|     func                          | -   [cudaq::ptsbe:                |
| tion)](api/languages/cpp_api.html | :PTSBEExecutionData::instructions |
| #_CPPv4N5cudaq16ExecutionContext1 |     (C++                          |
| 6ExecutionContextERKNSt6stringE), |     member)](api/languages/cp     |
|     [\[1\]](api/languages/        | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| cpp_api.html#_CPPv4N5cudaq16Execu | TSBEExecutionData12instructionsE) |
| tionContext16ExecutionContextERKN | -   [cudaq::ptsbe:                |
| St6stringENSt6size_tENSt6size_tE) | :PTSBEExecutionData::trajectories |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::expectationValue |     member)](api/languages/cp     |
|     (C++                          | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     member)](api/language         | TSBEExecutionData12trajectoriesE) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::ptsbe::PTSBEOptions   |
| cutionContext16expectationValueE) |     (C++                          |
| -   [cudaq::Execu                 |     struc                         |
| tionContext::explicitMeasurements | t)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
|     member)](api/languages/cp     | -   [cudaq::ptsbe::PTSB           |
| p_api.html#_CPPv4N5cudaq16Executi | EOptions::include_sequential_data |
| onContext20explicitMeasurementsE) |     (C++                          |
| -   [cuda                         |                                   |
| q::ExecutionContext::futureResult |    member)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
|     member)](api/lang             | ptions23include_sequential_dataE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::ptsb                  |
| 6ExecutionContext12futureResultE) | e::PTSBEOptions::max_trajectories |
| -   [cudaq::ExecutionContext      |     (C++                          |
| ::hasConditionalsOnMeasureResults |     member)](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
|     mem                           | 2PTSBEOptions16max_trajectoriesE) |
| ber)](api/languages/cpp_api.html# | -   [cudaq::ptsbe::PT             |
| _CPPv4N5cudaq16ExecutionContext31 | SBEOptions::return_execution_data |
| hasConditionalsOnMeasureResultsE) |     (C++                          |
| -   [cudaq::Executi               |     member)](api/languages/cpp_a  |
| onContext::invocationResultBuffer | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
|     (C++                          | EOptions21return_execution_dataE) |
|     member)](api/languages/cpp_   | -   [cudaq::pts                   |
| api.html#_CPPv4N5cudaq16Execution | be::PTSBEOptions::shot_allocation |
| Context22invocationResultBufferE) |     (C++                          |
| -   [cu                           |     member)](api/languages        |
| daq::ExecutionContext::kernelName | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     (C++                          | 12PTSBEOptions15shot_allocationE) |
|     member)](api/la               | -   [cud                          |
| nguages/cpp_api.html#_CPPv4N5cuda | aq::ptsbe::PTSBEOptions::strategy |
| q16ExecutionContext10kernelNameE) |     (C++                          |
| -   [cud                          |     member)](api/l                |
| aq::ExecutionContext::kernelTrace | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq5ptsbe12PTSBEOptions8strategyE) |
|     member)](api/lan              | -   [cudaq::ptsbe::PTSBETrace     |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11kernelTraceE) |     t                             |
| -   [cudaq:                       | ype)](api/languages/cpp_api.html# |
| :ExecutionContext::msm_dimensions | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
|     (C++                          | -   [                             |
|     member)](api/langua           | cudaq::ptsbe::PTSSamplingStrategy |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14msm_dimensionsE) |     class)](api                   |
| -   [cudaq::                      | /languages/cpp_api.html#_CPPv4N5c |
| ExecutionContext::msm_prob_err_id | udaq5ptsbe19PTSSamplingStrategyE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/languag          | ptsbe::PTSSamplingStrategy::clone |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15msm_prob_err_idE) |     function)](api/languag        |
| -   [cudaq::Ex                    | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| ecutionContext::msm_probabilities | sbe19PTSSamplingStrategy5cloneEv) |
|     (C++                          | -   [cudaq::ptsbe::PTSSampl       |
|     member)](api/languages        | ingStrategy::generateTrajectories |
| /cpp_api.html#_CPPv4N5cudaq16Exec |     (C++                          |
| utionContext17msm_probabilitiesE) |     function)](api/               |
| -                                 | languages/cpp_api.html#_CPPv4NK5c |
|    [cudaq::ExecutionContext::name | udaq5ptsbe19PTSSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)]                      | detail10NoisePointEEENSt6size_tE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq:                       |
| 4N5cudaq16ExecutionContext4nameE) | :ptsbe::PTSSamplingStrategy::name |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::noiseModel |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq5p |
|     member)](api/la               | tsbe19PTSSamplingStrategy4nameEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::ptsbe::PTSSampli      |
| q16ExecutionContext10noiseModelE) | ngStrategy::\~PTSSamplingStrategy |
| -   [cudaq::Exe                   |     (C++                          |
| cutionContext::numberTrajectories |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     member)](api/languages/       | q5ptsbe19PTSSamplingStrategyD0Ev) |
| cpp_api.html#_CPPv4N5cudaq16Execu | -   [cudaq::ptsbe::sample (C++    |
| tionContext18numberTrajectoriesE) |                                   |
| -   [c                            |  function)](api/languages/cpp_api |
| udaq::ExecutionContext::optResult | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
|     (C++                          | mpleE13sample_resultRK14sample_op |
|     member)](api/                 | tionsRR13QuantumKernelDpRR4Args), |
| languages/cpp_api.html#_CPPv4N5cu |     [\[1\]](api                   |
| daq16ExecutionContext9optResultE) | /languages/cpp_api.html#_CPPv4I0D |
| -                                 | pEN5cudaq5ptsbe6sampleE13sample_r |
|   [cudaq::ExecutionContext::qpuId | esultRKN5cudaq11noise_modelENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4Args) |
|     member)](                     | -   [cudaq::ptsbe::sample_async   |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5qpuIdE) |     function)](a                  |
| -   [cudaq                        | pi/languages/cpp_api.html#_CPPv4I |
| ::ExecutionContext::registerNames | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
|     (C++                          | 9async_sample_resultRK14sample_op |
|     member)](api/langu            | tionsRR13QuantumKernelDpRR4Args), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     [\[1\]](api/languages/cp      |
| ExecutionContext13registerNamesE) | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| -   [cu                           | be12sample_asyncE19async_sample_r |
| daq::ExecutionContext::reorderIdx | esultRKN5cudaq11noise_modelENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4Args) |
|     member)](api/la               | -   [cudaq::ptsbe::sample_options |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10reorderIdxE) |     struct)                       |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|  [cudaq::ExecutionContext::result | v4N5cudaq5ptsbe14sample_optionsE) |
|     (C++                          | -   [cudaq::ptsbe::sample_result  |
|     member)](a                    |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     class                         |
| 5cudaq16ExecutionContext6resultE) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq5ptsbe13sample_resultE) |
|   [cudaq::ExecutionContext::shots | -   [cudaq::pts                   |
|     (C++                          | be::sample_result::execution_data |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languages/c    |
| N5cudaq16ExecutionContext5shotsE) | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| -   [cudaq::                      | 3sample_result14execution_dataEv) |
| ExecutionContext::simulationState | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::has_execution_data |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |                                   |
| ecutionContext15simulationStateE) |    function)](api/languages/cpp_a |
| -                                 | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
|    [cudaq::ExecutionContext::spin | ple_result18has_execution_dataEv) |
|     (C++                          | -   [cudaq::pt                    |
|     member)]                      | sbe::sample_result::sample_result |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq16ExecutionContext4spinE) |     function)](api/l              |
| -   [cudaq::                      | anguages/cpp_api.html#_CPPv4N5cud |
| ExecutionContext::totalIterations | aq5ptsbe13sample_result13sample_r |
|     (C++                          | esultERRN5cudaq13sample_resultE), |
|     member)](api/languag          |                                   |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |  [\[1\]](api/languages/cpp_api.ht |
| ecutionContext15totalIterationsE) | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| -   [cudaq::Executio              | sult13sample_resultERRN5cudaq13sa |
| nContext::warnedNamedMeasurements | mple_resultE18PTSBEExecutionData) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     member)](api/languages/cpp_a  | sample_result::set_execution_data |
| pi.html#_CPPv4N5cudaq16ExecutionC |     (C++                          |
| ontext23warnedNamedMeasurementsE) |     function)](api/               |
| -   [cudaq::ExecutionResult (C++  | languages/cpp_api.html#_CPPv4N5cu |
|     st                            | daq5ptsbe13sample_result18set_exe |
| ruct)](api/languages/cpp_api.html | cution_dataE18PTSBEExecutionData) |
| #_CPPv4N5cudaq15ExecutionResultE) | -   [cud                          |
| -   [cud                          | aq::ptsbe::ShotAllocationStrategy |
| aq::ExecutionResult::appendResult |     (C++                          |
|     (C++                          |     struct)](using                |
|     functio                       | /examples/ptsbe.html#_CPPv4N5cuda |
| n)](api/languages/cpp_api.html#_C | q5ptsbe22ShotAllocationStrategyE) |
| PPv4N5cudaq15ExecutionResult12app | -   [cudaq::ptsbe::ShotAllocatio  |
| endResultENSt6stringENSt6size_tE) | nStrategy::ShotAllocationStrategy |
| -   [cu                           |     (C++                          |
| daq::ExecutionResult::deserialize |     function)                     |
|     (C++                          | ](using/examples/ptsbe.html#_CPPv |
|     function)                     | 4N5cudaq5ptsbe22ShotAllocationStr |
| ](api/languages/cpp_api.html#_CPP | ategy22ShotAllocationStrategyE4Ty |
| v4N5cudaq15ExecutionResult11deser | pedNSt8optionalINSt8uint64_tEEE), |
| ializeERNSt6vectorINSt6size_tEEE) |     [\[1\                         |
| -   [cudaq:                       | ]](using/examples/ptsbe.html#_CPP |
| :ExecutionResult::ExecutionResult | v4N5cudaq5ptsbe22ShotAllocationSt |
|     (C++                          | rategy22ShotAllocationStrategyEv) |
|     functio                       | -   [cudaq::pt                    |
| n)](api/languages/cpp_api.html#_C | sbe::ShotAllocationStrategy::Type |
| PPv4N5cudaq15ExecutionResult15Exe |     (C++                          |
| cutionResultE16CountsDictionary), |     enum)](using/exam             |
|     [\[1\]](api/lan               | ples/ptsbe.html#_CPPv4N5cudaq5pts |
| guages/cpp_api.html#_CPPv4N5cudaq | be22ShotAllocationStrategy4TypeE) |
| 15ExecutionResult15ExecutionResul | -   [cudaq::ptsbe::ShotAllocatio  |
| tE16CountsDictionaryNSt6stringE), | nStrategy::Type::HIGH_WEIGHT_BIAS |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     enumerat                      |
| Pv4N5cudaq15ExecutionResult15Exec | or)](using/examples/ptsbe.html#_C |
| utionResultE16CountsDictionaryd), | PPv4N5cudaq5ptsbe22ShotAllocation |
|                                   | Strategy4Type16HIGH_WEIGHT_BIASE) |
|    [\[3\]](api/languages/cpp_api. | -   [cudaq::ptsbe::ShotAllocati   |
| html#_CPPv4N5cudaq15ExecutionResu | onStrategy::Type::LOW_WEIGHT_BIAS |
| lt15ExecutionResultENSt6stringE), |     (C++                          |
|     [\[4\                         |     enumera                       |
| ]](api/languages/cpp_api.html#_CP | tor)](using/examples/ptsbe.html#_ |
| Pv4N5cudaq15ExecutionResult15Exec | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| utionResultERK15ExecutionResult), | nStrategy4Type15LOW_WEIGHT_BIASE) |
|     [\[5\]](api/language          | -   [cudaq::ptsbe::ShotAlloc      |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | ationStrategy::Type::PROPORTIONAL |
| cutionResult15ExecutionResultEd), |     (C++                          |
|     [\[6\]](api/languag           |     enum                          |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | erator)](using/examples/ptsbe.htm |
| ecutionResult15ExecutionResultEv) | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| -   [                             | tionStrategy4Type12PROPORTIONALE) |
| cudaq::ExecutionResult::operator= | -   [cudaq::ptsbe::Shot           |
|     (C++                          | AllocationStrategy::Type::UNIFORM |
|     function)](api/languages/     |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq15Execu |                                   |
| tionResultaSERK15ExecutionResult) |   enumerator)](using/examples/pts |
| -   [c                            | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| udaq::ExecutionResult::operator== | AllocationStrategy4Type7UNIFORME) |
|     (C++                          | -                                 |
|     function)](api/languages/c    |   [cudaq::ptsbe::TraceInstruction |
| pp_api.html#_CPPv4NK5cudaq15Execu |     (C++                          |
| tionResulteqERK15ExecutionResult) |     struct)](                     |
| -   [cud                          | api/languages/cpp_api.html#_CPPv4 |
| aq::ExecutionResult::registerName | N5cudaq5ptsbe16TraceInstructionE) |
|     (C++                          | -   [cudaq:                       |
|     member)](api/lan              | :ptsbe::TraceInstruction::channel |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult12registerNameE) |     member)](api/lang             |
| -   [cudaq                        | uages/cpp_api.html#_CPPv4N5cudaq5 |
| ::ExecutionResult::sequentialData | ptsbe16TraceInstruction7channelE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/langu            | ptsbe::TraceInstruction::controls |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| ExecutionResult14sequentialDataE) |     member)](api/langu            |
| -   [                             | ages/cpp_api.html#_CPPv4N5cudaq5p |
| cudaq::ExecutionResult::serialize | tsbe16TraceInstruction8controlsE) |
|     (C++                          | -   [cud                          |
|     function)](api/l              | aq::ptsbe::TraceInstruction::name |
| anguages/cpp_api.html#_CPPv4NK5cu |     (C++                          |
| daq15ExecutionResult9serializeEv) |     member)](api/l                |
| -   [cudaq::fermion_handler (C++  | anguages/cpp_api.html#_CPPv4N5cud |
|     c                             | aq5ptsbe16TraceInstruction4nameE) |
| lass)](api/languages/cpp_api.html | -   [cudaq                        |
| #_CPPv4N5cudaq15fermion_handlerE) | ::ptsbe::TraceInstruction::params |
| -   [cudaq::fermion_op (C++       |     (C++                          |
|     type)](api/languages/cpp_api  |     member)](api/lan              |
| .html#_CPPv4N5cudaq10fermion_opE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [cudaq::fermion_op_term (C++  | 5ptsbe16TraceInstruction6paramsE) |
|                                   | -   [cudaq:                       |
| type)](api/languages/cpp_api.html | :ptsbe::TraceInstruction::targets |
| #_CPPv4N5cudaq15fermion_op_termE) |     (C++                          |
| -   [cudaq::FermioniqQPU (C++     |     member)](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq5 |
|   class)](api/languages/cpp_api.h | ptsbe16TraceInstruction7targetsE) |
| tml#_CPPv4N5cudaq12FermioniqQPUE) | -   [cudaq::ptsbe::T              |
| -   [cudaq::get_state (C++        | raceInstruction::TraceInstruction |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |                                   |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |   function)](api/languages/cpp_ap |
| ateEDaRR13QuantumKernelDpRR4Args) | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| -   [cudaq::gradient (C++         | Instruction16TraceInstructionE20T |
|     class)](api/languages/cpp_    | raceInstructionTypeNSt6stringENSt |
| api.html#_CPPv4N5cudaq8gradientE) | 6vectorINSt6size_tEEENSt6vectorIN |
| -   [cudaq::gradient::clone (C++  | St6size_tEEENSt6vectorIdEENSt8opt |
|     fun                           | ionalIN5cudaq13kraus_channelEEE), |
| ction)](api/languages/cpp_api.htm |     [\[1\]](api/languages/cpp_a   |
| l#_CPPv4N5cudaq8gradient5cloneEv) | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| -   [cudaq::gradient::compute     | eInstruction16TraceInstructionEv) |
|     (C++                          | -   [cud                          |
|     function)](api/language       | aq::ptsbe::TraceInstruction::type |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     (C++                          |
| ient7computeERKNSt6vectorIdEERKNS |     member)](api/l                |
| t8functionIFdNSt6vectorIdEEEEEd), | anguages/cpp_api.html#_CPPv4N5cud |
|     [\[1\]](ap                    | aq5ptsbe16TraceInstruction4typeE) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [c                            |
| cudaq8gradient7computeERKNSt6vect | udaq::ptsbe::TraceInstructionType |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradient::gradient    |     enum)](api/                   |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     function)](api/lang           | daq5ptsbe20TraceInstructionTypeE) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -   [cudaq::                      |
| daq8gradient8gradientER7KernelT), | ptsbe::TraceInstructionType::Gate |
|                                   |     (C++                          |
|    [\[1\]](api/languages/cpp_api. |     enumerator)](api/langu        |
| html#_CPPv4I00EN5cudaq8gradient8g | ages/cpp_api.html#_CPPv4N5cudaq5p |
| radientER7KernelTRR10ArgsMapper), | tsbe20TraceInstructionType4GateE) |
|     [\[2\                         | -   [cudaq::ptsbe::               |
| ]](api/languages/cpp_api.html#_CP | TraceInstructionType::Measurement |
| Pv4I00EN5cudaq8gradient8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |                                   |
|     [\[3                          |    enumerator)](api/languages/cpp |
| \]](api/languages/cpp_api.html#_C | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| PPv4N5cudaq8gradient8gradientERRN | aceInstructionType11MeasurementE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::p                     |
|     [\[                           | tsbe::TraceInstructionType::Noise |
| 4\]](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8gradient8gradientEv) |     enumerator)](api/langua       |
| -   [cudaq::gradient::setArgs     | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     (C++                          | sbe20TraceInstructionType5NoiseE) |
|     fu                            | -   [                             |
| nction)](api/languages/cpp_api.ht | cudaq::ptsbe::TrajectoryPredicate |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     (C++                          |
| tArgsEvR13QuantumKernelDpRR4Args) |     type)](api                    |
| -   [cudaq::gradient::setKernel   | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq5ptsbe19TrajectoryPredicateE) |
|     function)](api/languages/c    | -   [cudaq::QPU (C++              |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     class)](api/languages         |
| ient9setKernelEvR13QuantumKernel) | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| -   [cud                          | -   [cudaq::QPU::beginExecution   |
| aq::gradients::central_difference |     (C++                          |
|     (C++                          |     function                      |
|     class)](api/la                | )](api/languages/cpp_api.html#_CP |
| nguages/cpp_api.html#_CPPv4N5cuda | Pv4N5cudaq3QPU14beginExecutionEv) |
| q9gradients18central_differenceE) | -   [cuda                         |
| -   [cudaq::gra                   | q::QPU::configureExecutionContext |
| dients::central_difference::clone |     (C++                          |
|     (C++                          |     funct                         |
|     function)](api/languages      | ion)](api/languages/cpp_api.html# |
| /cpp_api.html#_CPPv4N5cudaq9gradi | _CPPv4NK5cudaq3QPU25configureExec |
| ents18central_difference5cloneEv) | utionContextER16ExecutionContext) |
| -   [cudaq::gradi                 | -   [cudaq::QPU::endExecution     |
| ents::central_difference::compute |     (C++                          |
|     (C++                          |     functi                        |
|     function)](                   | on)](api/languages/cpp_api.html#_ |
| api/languages/cpp_api.html#_CPPv4 | CPPv4N5cudaq3QPU12endExecutionEv) |
| N5cudaq9gradients18central_differ | -   [cudaq::QPU::enqueue (C++     |
| ence7computeERKNSt6vectorIdEERKNS |     function)](ap                 |
| t8functionIFdNSt6vectorIdEEEEEd), | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq3QPU7enqueueER11QuantumTask) |
|   [\[1\]](api/languages/cpp_api.h | -   [cud                          |
| tml#_CPPv4N5cudaq9gradients18cent | aq::QPU::finalizeExecutionContext |
| ral_difference7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |     func                          |
| -   [cudaq::gradie                | tion)](api/languages/cpp_api.html |
| nts::central_difference::gradient | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     functio                       | -   [cudaq::QPU::getCompileTarget |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4I00EN5cudaq9gradients18centra |     function)](api/languages/     |
| l_difference8gradientER7KernelT), | cpp_api.html#_CPPv4N5cudaq3QPU16g |
|     [\[1\]](api/langua            | etCompileTargetER13sample_policy) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::QPU::getConnectivity  |
| q9gradients18central_difference8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)                     |
|     [\[2\]](api/languages/cpp_    | ](api/languages/cpp_api.html#_CPP |
| api.html#_CPPv4I00EN5cudaq9gradie | v4N5cudaq3QPU15getConnectivityEv) |
| nts18central_difference8gradientE | -                                 |
| RR13QuantumKernelRR10ArgsMapper), | [cudaq::QPU::getExecutionThreadId |
|     [\[3\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4N5cudaq9gradients |     function)](api/               |
| 18central_difference8gradientERRN | languages/cpp_api.html#_CPPv4NK5c |
| St8functionIFvNSt6vectorIdEEEEE), | udaq3QPU20getExecutionThreadIdEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::QPU::getNumQubits     |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18central_difference8gradientEv) |     functi                        |
| -   [cud                          | on)](api/languages/cpp_api.html#_ |
| aq::gradients::forward_difference | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     (C++                          | -   [                             |
|     class)](api/la                | cudaq::QPU::getRemoteCapabilities |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9gradients18forward_differenceE) |     function)](api/l              |
| -   [cudaq::gra                   | anguages/cpp_api.html#_CPPv4NK5cu |
| dients::forward_difference::clone | daq3QPU21getRemoteCapabilitiesEv) |
|     (C++                          | -   [cudaq::QPU::isEmulated (C++  |
|     function)](api/languages      |     func                          |
| /cpp_api.html#_CPPv4N5cudaq9gradi | tion)](api/languages/cpp_api.html |
| ents18forward_difference5cloneEv) | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| -   [cudaq::gradi                 | -   [cudaq::QPU::isSimulator (C++ |
| ents::forward_difference::compute |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](                   | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::QPU::onRandomSeedSet  |
| N5cudaq9gradients18forward_differ |     (C++                          |
| ence7computeERKNSt6vectorIdEERKNS |     function)](api/lang           |
| t8functionIFdNSt6vectorIdEEEEEd), | uages/cpp_api.html#_CPPv4N5cudaq3 |
|                                   | QPU15onRandomSeedSetENSt6size_tE) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::QPU::QPU (C++         |
| tml#_CPPv4N5cudaq9gradients18forw |     functio                       |
| ard_difference7computeERKNSt6vect | n)](api/languages/cpp_api.html#_C |
| orIdEERNSt6vectorIdEERK7spin_opd) | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| -   [cudaq::gradie                |                                   |
| nts::forward_difference::gradient |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     functio                       |     [\[2\]](api/languages/cpp_    |
| n)](api/languages/cpp_api.html#_C | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| PPv4I00EN5cudaq9gradients18forwar | -   [cudaq::QPU::setId (C++       |
| d_difference8gradientER7KernelT), |     function                      |
|     [\[1\]](api/langua            | )](api/languages/cpp_api.html#_CP |
| ges/cpp_api.html#_CPPv4I00EN5cuda | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| q9gradients18forward_difference8g | -   [cudaq::QPU::setShots (C++    |
| radientER7KernelTRR10ArgsMapper), |     f                             |
|     [\[2\]](api/languages/cpp_    | unction)](api/languages/cpp_api.h |
| api.html#_CPPv4I00EN5cudaq9gradie | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| nts18forward_difference8gradientE | -   [cudaq::                      |
| RR13QuantumKernelRR10ArgsMapper), | QPU::supportsExplicitMeasurements |
|     [\[3\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4N5cudaq9gradients |     function)](api/languag        |
| 18forward_difference8gradientERRN | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| St8functionIFvNSt6vectorIdEEEEE), | 28supportsExplicitMeasurementsEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::QPU::\~QPU (C++       |
| p_api.html#_CPPv4N5cudaq9gradient |     function)](api/languages/cp   |
| s18forward_difference8gradientEv) | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| -   [                             | -   [cudaq::QPUState (C++         |
| cudaq::gradients::parameter_shift |     class)](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4N5cudaq8QPUStateE) |
|     class)](api                   | -   [cudaq::qreg (C++             |
| /languages/cpp_api.html#_CPPv4N5c |     class)](api/lan               |
| udaq9gradients15parameter_shiftE) | guages/cpp_api.html#_CPPv4I_NSt6s |
| -   [cudaq::                      | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| gradients::parameter_shift::clone | -   [cudaq::qreg::back (C++       |
|     (C++                          |     function)                     |
|     function)](api/langua         | ](api/languages/cpp_api.html#_CPP |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | v4N5cudaq4qreg4backENSt6size_tE), |
| adients15parameter_shift5cloneEv) |     [\[1\]](api/languages/cpp_ap  |
| -   [cudaq::gr                    | i.html#_CPPv4N5cudaq4qreg4backEv) |
| adients::parameter_shift::compute | -   [cudaq::qreg::begin (C++      |
|     (C++                          |                                   |
|     function                      |  function)](api/languages/cpp_api |
| )](api/languages/cpp_api.html#_CP | .html#_CPPv4N5cudaq4qreg5beginEv) |
| Pv4N5cudaq9gradients15parameter_s | -   [cudaq::qreg::clear (C++      |
| hift7computeERKNSt6vectorIdEERKNS |                                   |
| t8functionIFdNSt6vectorIdEEEEEd), |  function)](api/languages/cpp_api |
|     [\[1\]](api/languages/cpp_ap  | .html#_CPPv4N5cudaq4qreg5clearEv) |
| i.html#_CPPv4N5cudaq9gradients15p | -   [cudaq::qreg::front (C++      |
| arameter_shift7computeERKNSt6vect |     function)]                    |
| orIdEERNSt6vectorIdEERK7spin_opd) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::gra                   | 4N5cudaq4qreg5frontENSt6size_tE), |
| dients::parameter_shift::gradient |     [\[1\]](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5frontEv) |
|     func                          | -   [cudaq::qreg::operator\[\]    |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4I00EN5cudaq9gradients15par |     functi                        |
| ameter_shift8gradientER7KernelT), | on)](api/languages/cpp_api.html#_ |
|     [\[1\]](api/lan               | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| guages/cpp_api.html#_CPPv4I00EN5c | -   [cudaq::qreg::qreg (C++       |
| udaq9gradients15parameter_shift8g |     function)                     |
| radientER7KernelTRR10ArgsMapper), | ](api/languages/cpp_api.html#_CPP |
|     [\[2\]](api/languages/c       | v4N5cudaq4qreg4qregENSt6size_tE), |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     [\[1\]](api/languages/cpp_ap  |
| dients15parameter_shift8gradientE | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::qreg::size (C++       |
|     [\[3\]](api/languages/        |                                   |
| cpp_api.html#_CPPv4N5cudaq9gradie |  function)](api/languages/cpp_api |
| nts15parameter_shift8gradientERRN | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::qreg::slice (C++      |
|     [\[4\]](api/languages         |     function)](api/langu          |
| /cpp_api.html#_CPPv4N5cudaq9gradi | ages/cpp_api.html#_CPPv4N5cudaq4q |
| ents15parameter_shift8gradientEv) | reg5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::kernel_builder (C++   | -   [cudaq::qreg::value_type (C++ |
|     clas                          |                                   |
| s)](api/languages/cpp_api.html#_C | type)](api/languages/cpp_api.html |
| PPv4IDpEN5cudaq14kernel_builderE) | #_CPPv4N5cudaq4qreg10value_typeE) |
| -   [c                            | -   [cudaq::qspan (C++            |
| udaq::kernel_builder::constantVal |     class)](api/lang              |
|     (C++                          | uages/cpp_api.html#_CPPv4I_NSt6si |
|     function)](api/la             | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QuakeValue (C++       |
| q14kernel_builder11constantValEd) |     class)](api/languages/cpp_api |
| -                                 | .html#_CPPv4N5cudaq10QuakeValueE) |
|  [cudaq::kernel_builder::detector | -   [cudaq::Q                     |
|     (C++                          | uakeValue::canValidateNumElements |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](api/languages      |
| pi.html#_CPPv4IDpEN5cudaq14kernel | /cpp_api.html#_CPPv4N5cudaq10Quak |
| _builder8detectorEvDpRR8MeasArgs) | eValue22canValidateNumElementsEv) |
| -                                 | -                                 |
| [cudaq::kernel_builder::detectors |  [cudaq::QuakeValue::constantSize |
|     (C++                          |     (C++                          |
|     func                          |     function)](api                |
| tion)](api/languages/cpp_api.html | /languages/cpp_api.html#_CPPv4N5c |
| #_CPPv4N5cudaq14kernel_builder9de | udaq10QuakeValue12constantSizeEv) |
| tectorsE10QuakeValue10QuakeValue) | -   [cudaq::QuakeValue::dump (C++ |
| -   [cu                           |     function)](api/lan            |
| daq::kernel_builder::getArguments | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 10QuakeValue4dumpERNSt7ostreamE), |
|     function)](api/lan            |     [\                            |
| guages/cpp_api.html#_CPPv4N5cudaq | [1\]](api/languages/cpp_api.html# |
| 14kernel_builder12getArgumentsEv) | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| -   [cu                           | -   [cudaq                        |
| daq::kernel_builder::getNumParams | ::QuakeValue::getRequiredElements |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)](api/langua         |
| guages/cpp_api.html#_CPPv4N5cudaq | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| 14kernel_builder12getNumParamsEv) | uakeValue19getRequiredElementsEv) |
| -   [c                            | -   [cudaq::QuakeValue::getValue  |
| udaq::kernel_builder::isArgStdVec |     (C++                          |
|     (C++                          |     function)]                    |
|     function)](api/languages/cp   | (api/languages/cpp_api.html#_CPPv |
| p_api.html#_CPPv4N5cudaq14kernel_ | 4NK5cudaq10QuakeValue8getValueEv) |
| builder11isArgStdVecENSt6size_tE) | -   [cudaq::QuakeValue::inverse   |
| -   [cuda                         |     (C++                          |
| q::kernel_builder::kernel_builder |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/languages/cpp_ | v4NK5cudaq10QuakeValue7inverseEv) |
| api.html#_CPPv4N5cudaq14kernel_bu | -   [cudaq::QuakeValue::isStdVec  |
| ilder14kernel_builderERNSt6vector |     (C++                          |
| IN7details17KernelBuilderTypeEEE) |     function)                     |
| -   [cudaq::k                     | ](api/languages/cpp_api.html#_CPP |
| ernel_builder::logical_observable | v4N5cudaq10QuakeValue8isStdVecEv) |
|     (C++                          | -                                 |
|     function)                     |    [cudaq::QuakeValue::operator\* |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4IDpEN5cudaq14kernel_builder18lo |     function)](api                |
| gical_observableEvDpRR8MeasArgs), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]](ap                    | udaq10QuakeValuemlE10QuakeValue), |
| i/languages/cpp_api.html#_CPPv4N5 |                                   |
| cudaq14kernel_builder18logical_ob | [\[1\]](api/languages/cpp_api.htm |
| servableE10QuakeValueNSt6size_tE) | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
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
| p_api.html#_CPPv4N5cudaq13kraus_c | -   [cud                          |
| hannel22populateDefaultOpNamesEv) | aq::quantum_platform::is_emulated |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::probabilities |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     member)](api/la               | pi.html#_CPPv4NK5cudaq16quantum_p |
| nguages/cpp_api.html#_CPPv4N5cuda | latform11is_emulatedENSt6size_tE) |
| q13kraus_channel13probabilitiesE) | -   [cudaq::                      |
| -                                 | quantum_platform::is_library_mode |
|  [cudaq::kraus_channel::push_back |     (C++                          |
|     (C++                          |     function)](api/languages      |
|     function)](api                | /cpp_api.html#_CPPv4NK5cudaq16qua |
| /languages/cpp_api.html#_CPPv4N5c | ntum_platform15is_library_modeEv) |
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
| l#_CPPv4N5cudaq14KrausSelectionE) | -   [cuda                         |
| -   [cudaq:                       | q::quantum_platform::set_exec_ctx |
| :KrausSelection::circuit_location |     (C++                          |
|     (C++                          |     funct                         |
|     member)](api/langua           | ion)](api/languages/cpp_api.html# |
| ges/cpp_api.html#_CPPv4N5cudaq14K | _CPPv4N5cudaq16quantum_platform12 |
| rausSelection16circuit_locationE) | set_exec_ctxEP16ExecutionContext) |
| -                                 | -   [c                            |
|  [cudaq::KrausSelection::is_error | udaq::quantum_platform::set_noise |
|     (C++                          |     (C++                          |
|     member)](a                    |     function                      |
| pi/languages/cpp_api.html#_CPPv4N | )](api/languages/cpp_api.html#_CP |
| 5cudaq14KrausSelection8is_errorE) | Pv4N5cudaq16quantum_platform9set_ |
| -   [cudaq::Kra                   | noiseEPK11noise_modelNSt6size_tE) |
| usSelection::kraus_operator_index | -   [cudaq::quantum_platfor       |
|     (C++                          | m::supports_explicit_measurements |
|     member)](api/languages/       |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq14Kraus |     function)](api/l              |
| Selection20kraus_operator_indexE) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cuda                         | daq16quantum_platform30supports_e |
| q::KrausSelection::KrausSelection | xplicit_measurementsENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_pla           |
|     function)](a                  | tform::supports_task_distribution |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14KrausSelection14KrausSele |     fu                            |
| ctionENSt6size_tENSt6vectorINSt6s | nction)](api/languages/cpp_api.ht |
| ize_tEEENSt6stringENSt6size_tEb), | ml#_CPPv4NK5cudaq16quantum_platfo |
|     [\[1\]](api/langu             | rm26supports_task_distributionEv) |
| ages/cpp_api.html#_CPPv4N5cudaq14 | -   [cudaq::quantum               |
| KrausSelection14KrausSelectionEv) | _platform::with_execution_context |
| -                                 |     (C++                          |
|   [cudaq::KrausSelection::op_name |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](                     | v4I0DpEN5cudaq16quantum_platform2 |
| api/languages/cpp_api.html#_CPPv4 | 2with_execution_contextEDaR16Exec |
| N5cudaq14KrausSelection7op_nameE) | utionContextRR8CallableDpRR4Args) |
| -   [                             | -   [cudaq::QuantumTask (C++      |
| cudaq::KrausSelection::operator== |     type)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq11QuantumTaskE) |
|     function)](api/languages      | -   [cudaq::qubit (C++            |
| /cpp_api.html#_CPPv4NK5cudaq14Kra |     type)](api/languages/c        |
| usSelectioneqERK14KrausSelection) | pp_api.html#_CPPv4N5cudaq5qubitE) |
| -                                 | -   [cudaq::QubitConnectivity     |
|    [cudaq::KrausSelection::qubits |     (C++                          |
|     (C++                          |     ty                            |
|     member)]                      | pe)](api/languages/cpp_api.html#_ |
| (api/languages/cpp_api.html#_CPPv | CPPv4N5cudaq17QubitConnectivityE) |
| 4N5cudaq14KrausSelection6qubitsE) | -   [cudaq::QubitEdge (C++        |
| -   [cudaq::KrausTrajectory (C++  |     type)](api/languages/cpp_a    |
|     st                            | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::qudit (C++            |
| #_CPPv4N5cudaq15KrausTrajectoryE) |     clas                          |
| -                                 | s)](api/languages/cpp_api.html#_C |
|  [cudaq::KrausTrajectory::builder | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     (C++                          | -   [cudaq::qudit::qudit (C++     |
|     function)](ap                 |                                   |
| i/languages/cpp_api.html#_CPPv4N5 | function)](api/languages/cpp_api. |
| cudaq15KrausTrajectory7builderEv) | html#_CPPv4N5cudaq5qudit5quditEv) |
| -   [cu                           | -   [cudaq::qvector (C++          |
| daq::KrausTrajectory::countErrors |     class)                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/lang           | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::qvector::back (C++    |
| 15KrausTrajectory11countErrorsEv) |     function)](a                  |
| -   [                             | pi/languages/cpp_api.html#_CPPv4N |
| cudaq::KrausTrajectory::isOrdered | 5cudaq7qvector4backENSt6size_tE), |
|     (C++                          |                                   |
|     function)](api/l              |   [\[1\]](api/languages/cpp_api.h |
| anguages/cpp_api.html#_CPPv4NK5cu | tml#_CPPv4N5cudaq7qvector4backEv) |
| daq15KrausTrajectory9isOrderedEv) | -   [cudaq::qvector::begin (C++   |
| -   [cudaq::                      |     fu                            |
| KrausTrajectory::kraus_selections | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5beginEv) |
|     member)](api/languag          | -   [cudaq::qvector::clear (C++   |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |     fu                            |
| ausTrajectory16kraus_selectionsE) | nction)](api/languages/cpp_api.ht |
| -   [cudaq:                       | ml#_CPPv4N5cudaq7qvector5clearEv) |
| :KrausTrajectory::KrausTrajectory | -   [cudaq::qvector::end (C++     |
|     (C++                          |                                   |
|     function                      | function)](api/languages/cpp_api. |
| )](api/languages/cpp_api.html#_CP | html#_CPPv4N5cudaq7qvector3endEv) |
| Pv4N5cudaq15KrausTrajectory15Krau | -   [cudaq::qvector::front (C++   |
| sTrajectoryENSt6size_tENSt6vector |     function)](ap                 |
| I14KrausSelectionEEdNSt6size_tE), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[1\]](api/languag           | cudaq7qvector5frontENSt6size_tE), |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |                                   |
| ausTrajectory15KrausTrajectoryEv) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::Kr                    | ml#_CPPv4N5cudaq7qvector5frontEv) |
| ausTrajectory::measurement_counts | -   [cudaq::qvector::operator=    |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     functio                       |
| /cpp_api.html#_CPPv4N5cudaq15Krau | n)](api/languages/cpp_api.html#_C |
| sTrajectory18measurement_countsE) | PPv4N5cudaq7qvectoraSERK7qvector) |
| -   [cud                          | -   [cudaq::qvector::operator\[\] |
| aq::KrausTrajectory::multiplicity |     (C++                          |
|     (C++                          |     function)                     |
|     member)](api/lan              | ](api/languages/cpp_api.html#_CPP |
| guages/cpp_api.html#_CPPv4N5cudaq | v4N5cudaq7qvectorixEKNSt6size_tE) |
| 15KrausTrajectory12multiplicityE) | -   [cudaq::qvector::qvector (C++ |
| -   [                             |     function)](api/               |
| cudaq::KrausTrajectory::num_shots | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq7qvector7qvectorENSt6size_tE), |
|     member)](api                  |     [\[1\]](a                     |
| /languages/cpp_api.html#_CPPv4N5c | pi/languages/cpp_api.html#_CPPv4N |
| udaq15KrausTrajectory9num_shotsE) | 5cudaq7qvector7qvectorERK5state), |
| -   [c                            |     [\[2\]](api                   |
| udaq::KrausTrajectory::operator== | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq7qvector7qvectorERK7qvector), |
|     function)](api/languages/c    |     [\[3\]](ap                    |
| pp_api.html#_CPPv4NK5cudaq15Kraus | i/languages/cpp_api.html#_CPPv4N5 |
| TrajectoryeqERK15KrausTrajectory) | cudaq7qvector7qvectorERR7qvector) |
| -   [cu                           | -   [cudaq::qvector::size (C++    |
| daq::KrausTrajectory::probability |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     member)](api/la               | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::qvector::slice (C++   |
| q15KrausTrajectory11probabilityE) |     function)](api/language       |
| -   [cuda                         | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| q::KrausTrajectory::trajectory_id | tor5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qvector::value_type   |
|     member)](api/lang             |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     typ                           |
| 5KrausTrajectory13trajectory_idE) | e)](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq7qvector10value_typeE) |
|   [cudaq::KrausTrajectory::weight | -   [cudaq::qview (C++            |
|     (C++                          |     clas                          |
|     member)](                     | s)](api/languages/cpp_api.html#_C |
| api/languages/cpp_api.html#_CPPv4 | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| N5cudaq15KrausTrajectory6weightE) | -   [cudaq::qview::back (C++      |
| -                                 |     function)                     |
|    [cudaq::KrausTrajectoryBuilder | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq5qview4backENSt6size_tE) |
|     class)](                      | -   [cudaq::qview::begin (C++     |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq22KrausTrajectoryBuilderE) | function)](api/languages/cpp_api. |
| -   [cud                          | html#_CPPv4N5cudaq5qview5beginEv) |
| aq::KrausTrajectoryBuilder::build | -   [cudaq::qview::end (C++       |
|     (C++                          |                                   |
|     function)](api/lang           |   function)](api/languages/cpp_ap |
| uages/cpp_api.html#_CPPv4NK5cudaq | i.html#_CPPv4N5cudaq5qview3endEv) |
| 22KrausTrajectoryBuilder5buildEv) | -   [cudaq::qview::front (C++     |
| -   [cud                          |     function)](                   |
| aq::KrausTrajectoryBuilder::setId | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5qview5frontENSt6size_tE), |
|     function)](api/languages/cpp  |                                   |
| _api.html#_CPPv4N5cudaq22KrausTra |    [\[1\]](api/languages/cpp_api. |
| jectoryBuilder5setIdENSt6size_tE) | html#_CPPv4N5cudaq5qview5frontEv) |
| -   [cudaq::Kraus                 | -   [cudaq::qview::operator\[\]   |
| TrajectoryBuilder::setProbability |     (C++                          |
|     (C++                          |     functio                       |
|     function)](api/languages/cpp  | n)](api/languages/cpp_api.html#_C |
| _api.html#_CPPv4N5cudaq22KrausTra | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| jectoryBuilder14setProbabilityEd) | -   [cudaq::qview::qview (C++     |
| -   [cudaq::Krau                  |     functio                       |
| sTrajectoryBuilder::setSelections | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     function)](api/languag        |     [\[1                          |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | \]](api/languages/cpp_api.html#_C |
| ausTrajectoryBuilder13setSelectio | PPv4N5cudaq5qview5qviewERK5qview) |
| nsENSt6vectorI14KrausSelectionEE) | -   [cudaq::qview::size (C++      |
| -   [cudaq::matrix_callback (C++  |                                   |
|     c                             | function)](api/languages/cpp_api. |
| lass)](api/languages/cpp_api.html | html#_CPPv4NK5cudaq5qview4sizeEv) |
| #_CPPv4N5cudaq15matrix_callbackE) | -   [cudaq::qview::slice (C++     |
| -   [cudaq::matrix_handler (C++   |     function)](api/langua         |
|                                   | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| class)](api/languages/cpp_api.htm | iew5sliceENSt6size_tENSt6size_tE) |
| l#_CPPv4N5cudaq14matrix_handlerE) | -   [cudaq::qview::value_type     |
| -   [cudaq::mat                   |     (C++                          |
| rix_handler::commutation_behavior |     t                             |
|     (C++                          | ype)](api/languages/cpp_api.html# |
|     struct)](api/languages/       | _CPPv4N5cudaq5qview10value_typeE) |
| cpp_api.html#_CPPv4N5cudaq14matri | -   [cudaq::range (C++            |
| x_handler20commutation_behaviorE) |     fun                           |
| -                                 | ction)](api/languages/cpp_api.htm |
|    [cudaq::matrix_handler::define | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
|     (C++                          | orI11ElementTypeEE11ElementType), |
|     function)](a                  |     [\[1\]](api/languages/cpp_    |
| pi/languages/cpp_api.html#_CPPv4N | api.html#_CPPv4I0EN5cudaq5rangeEN |
| 5cudaq14matrix_handler6defineENSt | St6vectorI11ElementTypeEE11Elemen |
| 6stringENSt6vectorINSt7int64_tEEE | tType11ElementType11ElementType), |
| RR15matrix_callbackRKNSt13unorder |     [                             |
| ed_mapINSt6stringENSt6stringEEE), | \[2\]](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| [\[1\]](api/languages/cpp_api.htm | -   [cudaq::real (C++             |
| l#_CPPv4N5cudaq14matrix_handler6d |     type)](api/languages/         |
| efineENSt6stringENSt6vectorINSt7i | cpp_api.html#_CPPv4N5cudaq4realE) |
| nt64_tEEERR15matrix_callbackRR20d | -   [cudaq::registry (C++         |
| iag_matrix_callbackRKNSt13unorder |     type)](api/languages/cpp_     |
| ed_mapINSt6stringENSt6stringEEE), | api.html#_CPPv4N5cudaq8registryE) |
|     [\[2\]](                      | -                                 |
| api/languages/cpp_api.html#_CPPv4 |  [cudaq::registry::RegisteredType |
| N5cudaq14matrix_handler6defineENS |     (C++                          |
| t6stringENSt6vectorINSt7int64_tEE |     class)](api/                  |
| ERR15matrix_callbackRRNSt13unorde | languages/cpp_api.html#_CPPv4I0EN |
| red_mapINSt6stringENSt6stringEEE) | 5cudaq8registry14RegisteredTypeE) |
| -                                 | -   [cudaq::RemoteCapabilities    |
|   [cudaq::matrix_handler::degrees |     (C++                          |
|     (C++                          |     struc                         |
|     function)](ap                 | t)](api/languages/cpp_api.html#_C |
| i/languages/cpp_api.html#_CPPv4NK | PPv4N5cudaq18RemoteCapabilitiesE) |
| 5cudaq14matrix_handler7degreesEv) | -   [cudaq::Remot                 |
| -                                 | eCapabilities::RemoteCapabilities |
|  [cudaq::matrix_handler::displace |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api/language       | _api.html#_CPPv4N5cudaq18RemoteCa |
| s/cpp_api.html#_CPPv4N5cudaq14mat | pabilities18RemoteCapabilitiesEb) |
| rix_handler8displaceENSt6size_tE) | -   [cudaq:                       |
| -   [cudaq::matrix                | :RemoteCapabilities::stateOverlap |
| _handler::get_expected_dimensions |     (C++                          |
|     (C++                          |     member)](api/langua           |
|                                   | ges/cpp_api.html#_CPPv4N5cudaq18R |
|    function)](api/languages/cpp_a | emoteCapabilities12stateOverlapE) |
| pi.html#_CPPv4NK5cudaq14matrix_ha | -                                 |
| ndler23get_expected_dimensionsEv) |   [cudaq::RemoteCapabilities::vqe |
| -   [cudaq::matrix_ha             |     (C++                          |
| ndler::get_parameter_descriptions |     member)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq18RemoteCapabilities3vqeE) |
| function)](api/languages/cpp_api. | -   [cudaq::Resources (C++        |
| html#_CPPv4NK5cudaq14matrix_handl |     class)](api/languages/cpp_a   |
| er26get_parameter_descriptionsEv) | pi.html#_CPPv4N5cudaq9ResourcesE) |
| -   [c                            | -   [cudaq::run (C++              |
| udaq::matrix_handler::instantiate |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](a                  | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| pi/languages/cpp_api.html#_CPPv4N | 5invoke_result_tINSt7decay_tI13Qu |
| 5cudaq14matrix_handler11instantia | antumKernelEEDpNSt7decay_tI4ARGSE |
| teENSt6stringERKNSt6vectorINSt6si | EEEEENSt6size_tERN5cudaq11noise_m |
| ze_tEEERK20commutation_behavior), | odelERR13QuantumKernelDpRR4ARGS), |
|     [\[1\]](                      |     [\[1\]](api/langu             |
| api/languages/cpp_api.html#_CPPv4 | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| N5cudaq14matrix_handler11instanti | daq3runENSt6vectorINSt15invoke_re |
| ateENSt6stringERRNSt6vectorINSt6s | sult_tINSt7decay_tI13QuantumKerne |
| ize_tEEERK20commutation_behavior) | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| -   [cuda                         | ize_tERR13QuantumKernelDpRR4ARGS) |
| q::matrix_handler::matrix_handler | -   [cudaq::run_async (C++        |
|     (C++                          |     functio                       |
|     function)](api/languag        | n)](api/languages/cpp_api.html#_C |
| es/cpp_api.html#_CPPv4I0_NSt11ena | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| ble_if_tINSt12is_base_of_vI16oper | tureINSt6vectorINSt15invoke_resul |
| ator_handler1TEEbEEEN5cudaq14matr | t_tINSt7decay_tI13QuantumKernelEE |
| ix_handler14matrix_handlerERK1T), | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|     [\[1\]](ap                    | ze_tENSt6size_tERN5cudaq11noise_m |
| i/languages/cpp_api.html#_CPPv4I0 | odelERR13QuantumKernelDpRR4ARGS), |
| _NSt11enable_if_tINSt12is_base_of |     [\[1\]](api/la                |
| _vI16operator_handler1TEEbEEEN5cu | nguages/cpp_api.html#_CPPv4I0DpEN |
| daq14matrix_handler14matrix_handl | 5cudaq9run_asyncENSt6futureINSt6v |
| erERK1TRK20commutation_behavior), | ectorINSt15invoke_result_tINSt7de |
|     [\[2\]](api/languages/cpp_ap  | cay_tI13QuantumKernelEEDpNSt7deca |
| i.html#_CPPv4N5cudaq14matrix_hand | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| ler14matrix_handlerENSt6size_tE), | ize_tERR13QuantumKernelDpRR4ARGS) |
|     [\[3\]](api/                  | -   [cudaq::RuntimeTarget (C++    |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq14matrix_handler14matrix_handl | struct)](api/languages/cpp_api.ht |
| erENSt6stringERKNSt6vectorINSt6si | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| ze_tEEERK20commutation_behavior), | -   [cudaq::sample (C++           |
|     [\[4\]](api/                  |     function)](api/languages/c    |
| languages/cpp_api.html#_CPPv4N5cu | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| daq14matrix_handler14matrix_handl | mpleE13sample_resultRK14sample_op |
| erENSt6stringERRNSt6vectorINSt6si | tionsRR13QuantumKernelDpRR4Args), |
| ze_tEEERK20commutation_behavior), |     [\[1\                         |
|     [\                            | ]](api/languages/cpp_api.html#_CP |
| [5\]](api/languages/cpp_api.html# | Pv4I0DpEN5cudaq6sampleE13sample_r |
| _CPPv4N5cudaq14matrix_handler14ma | esultRR13QuantumKernelDpRR4Args), |
| trix_handlerERK14matrix_handler), |     [\                            |
|     [                             | [2\]](api/languages/cpp_api.html# |
| \[6\]](api/languages/cpp_api.html | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| #_CPPv4N5cudaq14matrix_handler14m | ize_tERR13QuantumKernelDpRR4Args) |
| atrix_handlerERR14matrix_handler) | -   [cudaq::sample_options (C++   |
| -                                 |     s                             |
|  [cudaq::matrix_handler::momentum | truct)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq14sample_optionsE) |
|     function)](api/language       | -   [cudaq::sample_result (C++    |
| s/cpp_api.html#_CPPv4N5cudaq14mat |                                   |
| rix_handler8momentumENSt6size_tE) |  class)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq13sample_resultE) |
|    [cudaq::matrix_handler::number | -   [cudaq::sample_result::append |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/languages/cpp_ |
| ges/cpp_api.html#_CPPv4N5cudaq14m | api.html#_CPPv4N5cudaq13sample_re |
| atrix_handler6numberENSt6size_tE) | sult6appendERK15ExecutionResultb) |
| -                                 | -   [cudaq::sample_result::begin  |
| [cudaq::matrix_handler::operator= |     (C++                          |
|     (C++                          |     function)]                    |
|     fun                           | (api/languages/cpp_api.html#_CPPv |
| ction)](api/languages/cpp_api.htm | 4N5cudaq13sample_result5beginEv), |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     [\[1\]]                       |
| NSt7is_sameI1T14matrix_handlerE5v | (api/languages/cpp_api.html#_CPPv |
| alueENSt12is_base_of_vI16operator | 4NK5cudaq13sample_result5beginEv) |
| _handler1TEEEbEEEN5cudaq14matrix_ | -   [cudaq::sample_result::cbegin |
| handleraSER14matrix_handlerRK1T), |     (C++                          |
|     [\[1\]](api/languages         |     function)](                   |
| /cpp_api.html#_CPPv4N5cudaq14matr | api/languages/cpp_api.html#_CPPv4 |
| ix_handleraSERK14matrix_handler), | NK5cudaq13sample_result6cbeginEv) |
|     [\[2\]](api/language          | -   [cudaq::sample_result::cend   |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handleraSERR14matrix_handler) |     function)                     |
| -   [                             | ](api/languages/cpp_api.html#_CPP |
| cudaq::matrix_handler::operator== | v4NK5cudaq13sample_result4cendEv) |
|     (C++                          | -   [cudaq::sample_result::clear  |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4NK5cudaq14mat |     function)                     |
| rix_handlereqERK14matrix_handler) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq13sample_result5clearEv) |
|    [cudaq::matrix_handler::parity | -   [cudaq::sample_result::count  |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](                   |
| ges/cpp_api.html#_CPPv4N5cudaq14m | api/languages/cpp_api.html#_CPPv4 |
| atrix_handler6parityENSt6size_tE) | NK5cudaq13sample_result5countENSt |
| -                                 | 11string_viewEKNSt11string_viewE) |
|  [cudaq::matrix_handler::position | -   [                             |
|     (C++                          | cudaq::sample_result::deserialize |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     functio                       |
| rix_handler8positionENSt6size_tE) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::                      | PPv4N5cudaq13sample_result11deser |
| matrix_handler::remove_definition | ializeERNSt6vectorINSt6size_tEEE) |
|     (C++                          | -   [cudaq::sample_result::dump   |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/languag        |
| ml#_CPPv4N5cudaq14matrix_handler1 | es/cpp_api.html#_CPPv4NK5cudaq13s |
| 7remove_definitionERKNSt6stringE) | ample_result4dumpERNSt7ostreamE), |
| -                                 |     [\[1\]                        |
|   [cudaq::matrix_handler::squeeze | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq13sample_result4dumpEv) |
|     function)](api/languag        | -   [cudaq::sample_result::end    |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     (C++                          |
| trix_handler7squeezeENSt6size_tE) |     function                      |
| -   [cudaq::m                     | )](api/languages/cpp_api.html#_CP |
| atrix_handler::to_diagonal_matrix | Pv4N5cudaq13sample_result3endEv), |
|     (C++                          |     [\[1\                         |
|     function)](api/lang           | ]](api/languages/cpp_api.html#_CP |
| uages/cpp_api.html#_CPPv4NK5cudaq | Pv4NK5cudaq13sample_result3endEv) |
| 14matrix_handler18to_diagonal_mat | -   [                             |
| rixERNSt13unordered_mapINSt6size_ | cudaq::sample_result::expectation |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     f                             |
| -                                 | unction)](api/languages/cpp_api.h |
| [cudaq::matrix_handler::to_matrix | tml#_CPPv4NK5cudaq13sample_result |
|     (C++                          | 11expectationEKNSt11string_viewE) |
|     function)                     | -   [c                            |
| ](api/languages/cpp_api.html#_CPP | udaq::sample_result::get_marginal |
| v4NK5cudaq14matrix_handler9to_mat |     (C++                          |
| rixERNSt13unordered_mapINSt6size_ |     function)](api/languages/cpp_ |
| tENSt7int64_tEEERKNSt13unordered_ | api.html#_CPPv4NK5cudaq13sample_r |
| mapINSt6stringENSt7complexIdEEEE) | esult12get_marginalERKNSt6vectorI |
| -                                 | NSt6size_tEEEKNSt11string_viewE), |
| [cudaq::matrix_handler::to_string |     [\[1\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4NK5cudaq13sample_r |
|     function)](api/               | esult12get_marginalERRKNSt6vector |
| languages/cpp_api.html#_CPPv4NK5c | INSt6size_tEEEKNSt11string_viewE) |
| udaq14matrix_handler9to_stringEb) | -   [cuda                         |
| -                                 | q::sample_result::get_total_shots |
| [cudaq::matrix_handler::unique_id |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/               | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| languages/cpp_api.html#_CPPv4NK5c | sample_result15get_total_shotsEv) |
| udaq14matrix_handler9unique_idEv) | -   [cuda                         |
| -   [cudaq:                       | q::sample_result::has_even_parity |
| :matrix_handler::\~matrix_handler |     (C++                          |
|     (C++                          |     fun                           |
|     functi                        | ction)](api/languages/cpp_api.htm |
| on)](api/languages/cpp_api.html#_ | l#_CPPv4N5cudaq13sample_result15h |
| CPPv4N5cudaq14matrix_handlerD0Ev) | as_even_parityENSt11string_viewE) |
| -   [cudaq::matrix_op (C++        | -   [cuda                         |
|     type)](api/languages/cpp_a    | q::sample_result::has_expectation |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     (C++                          |
| -   [cudaq::matrix_op_term (C++   |     funct                         |
|                                   | ion)](api/languages/cpp_api.html# |
|  type)](api/languages/cpp_api.htm | _CPPv4NK5cudaq13sample_result15ha |
| l#_CPPv4N5cudaq14matrix_op_termE) | s_expectationEKNSt11string_viewE) |
| -                                 | -   [cu                           |
|    [cudaq::mdiag_operator_handler | daq::sample_result::most_probable |
|     (C++                          |     (C++                          |
|     class)](                      |     fun                           |
| api/languages/cpp_api.html#_CPPv4 | ction)](api/languages/cpp_api.htm |
| N5cudaq22mdiag_operator_handlerE) | l#_CPPv4NK5cudaq13sample_result13 |
| -   [cudaq::mpi (C++              | most_probableEKNSt11string_viewE) |
|     type)](api/languages          | -                                 |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | [cudaq::sample_result::operator+= |
| -   [cudaq::mpi::all_gather (C++  |     (C++                          |
|     fu                            |     function)](api/langua         |
| nction)](api/languages/cpp_api.ht | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | ample_resultpLERK13sample_result) |
| RNSt6vectorIdEERKNSt6vectorIdEE), | -                                 |
|                                   |  [cudaq::sample_result::operator= |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi10all_gather |     function)](api/langua         |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -   [cudaq::mpi::all_reduce (C++  | ample_resultaSERR13sample_result) |
|                                   | -                                 |
|  function)](api/languages/cpp_api | [cudaq::sample_result::operator== |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     (C++                          |
| reduceE1TRK1TRK14BinaryFunction), |     function)](api/languag        |
|     [\[1\]](api/langu             | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ages/cpp_api.html#_CPPv4I00EN5cud | ample_resulteqERK13sample_result) |
| aq3mpi10all_reduceE1TRK1TRK4Func) | -   [                             |
| -   [cudaq::mpi::broadcast (C++   | cudaq::sample_result::probability |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/lan            |
| daq3mpi9broadcastERNSt6stringEi), | guages/cpp_api.html#_CPPv4NK5cuda |
|     [\[1\]](api/la                | q13sample_result11probabilityENSt |
| nguages/cpp_api.html#_CPPv4N5cuda | 11string_viewEKNSt11string_viewE) |
| q3mpi9broadcastERNSt6vectorIdEEi) | -   [cud                          |
| -   [cudaq::mpi::finalize (C++    | aq::sample_result::register_names |
|     f                             |     (C++                          |
| unction)](api/languages/cpp_api.h |     function)](api/langu          |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| -   [cudaq::mpi::initialize (C++  | 3sample_result14register_namesEv) |
|     function                      | -                                 |
| )](api/languages/cpp_api.html#_CP |    [cudaq::sample_result::reorder |
| Pv4N5cudaq3mpi10initializeEiPPc), |     (C++                          |
|     [                             |     function)](api/langua         |
| \[1\]](api/languages/cpp_api.html | ges/cpp_api.html#_CPPv4N5cudaq13s |
| #_CPPv4N5cudaq3mpi10initializeEv) | ample_result7reorderERKNSt6vector |
| -   [cudaq::mpi::is_initialized   | INSt6size_tEEEKNSt11string_viewE) |
|     (C++                          | -   [cu                           |
|     function                      | daq::sample_result::sample_result |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq3mpi14is_initializedEv) |     func                          |
| -   [cudaq::mpi::num_ranks (C++   | tion)](api/languages/cpp_api.html |
|     fu                            | #_CPPv4N5cudaq13sample_result13sa |
| nction)](api/languages/cpp_api.ht | mple_resultERK15ExecutionResult), |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     [\[1\]](api/la                |
| -   [cudaq::mpi::rank (C++        | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q13sample_result13sample_resultER |
|    function)](api/languages/cpp_a | KNSt6vectorI15ExecutionResultEE), |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |                                   |
| -   [cudaq::noise_model (C++      |  [\[2\]](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4N5cudaq13sample_result13 |
|    class)](api/languages/cpp_api. | sample_resultERR13sample_result), |
| html#_CPPv4N5cudaq11noise_modelE) |     [                             |
| -   [cudaq::n                     | \[3\]](api/languages/cpp_api.html |
| oise_model::add_all_qubit_channel | #_CPPv4N5cudaq13sample_result13sa |
|     (C++                          | mple_resultERR15ExecutionResult), |
|     function)](api                |     [\[4\]](api/lan               |
| /languages/cpp_api.html#_CPPv4IDp | guages/cpp_api.html#_CPPv4N5cudaq |
| EN5cudaq11noise_model21add_all_qu | 13sample_result13sample_resultEdR |
| bit_channelEvRK13kraus_channeli), | KNSt6vectorI15ExecutionResultEE), |
|     [\[1\]](api/langua            |     [\[5\]](api/lan               |
| ges/cpp_api.html#_CPPv4N5cudaq11n | guages/cpp_api.html#_CPPv4N5cudaq |
| oise_model21add_all_qubit_channel | 13sample_result13sample_resultEv) |
| ERKNSt6stringERK13kraus_channeli) | -                                 |
| -                                 |  [cudaq::sample_result::serialize |
|  [cudaq::noise_model::add_channel |     (C++                          |
|     (C++                          |     function)](api                |
|     funct                         | /languages/cpp_api.html#_CPPv4NK5 |
| ion)](api/languages/cpp_api.html# | cudaq13sample_result9serializeEv) |
| _CPPv4IDpEN5cudaq11noise_model11a | -   [cudaq::sample_result::size   |
| dd_channelEvRK15PredicateFuncTy), |     (C++                          |
|     [\[1\]](api/languages/cpp_    |     function)](api/languages/c    |
| api.html#_CPPv4IDpEN5cudaq11noise | pp_api.html#_CPPv4NK5cudaq13sampl |
| _model11add_channelEvRKNSt6vector | e_result4sizeEKNSt11string_viewE) |
| INSt6size_tEEERK13kraus_channel), | -   [cudaq::sample_result::to_map |
|     [\[2\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languages/cpp  |
| cudaq11noise_model11add_channelER | _api.html#_CPPv4NK5cudaq13sample_ |
| KNSt6stringERK15PredicateFuncTy), | result6to_mapEKNSt11string_viewE) |
|                                   | -   [cuda                         |
| [\[3\]](api/languages/cpp_api.htm | q::sample_result::\~sample_result |
| l#_CPPv4N5cudaq11noise_model11add |     (C++                          |
| _channelERKNSt6stringERKNSt6vecto |     funct                         |
| rINSt6size_tEEERK13kraus_channel) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::noise_model::empty    | _CPPv4N5cudaq13sample_resultD0Ev) |
|     (C++                          | -   [cudaq::scalar_callback (C++  |
|     function                      |     c                             |
| )](api/languages/cpp_api.html#_CP | lass)](api/languages/cpp_api.html |
| Pv4NK5cudaq11noise_model5emptyEv) | #_CPPv4N5cudaq15scalar_callbackE) |
| -                                 | -   [c                            |
| [cudaq::noise_model::get_channels | udaq::scalar_callback::operator() |
|     (C++                          |     (C++                          |
|     function)](api/l              |     function)](api/language       |
| anguages/cpp_api.html#_CPPv4I0ENK | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| 5cudaq11noise_model12get_channels | alar_callbackclERKNSt13unordered_ |
| ENSt6vectorI13kraus_channelEERKNS | mapINSt6stringENSt7complexIdEEEE) |
| t6vectorINSt6size_tEEERKNSt6vecto | -   [                             |
| rINSt6size_tEEERKNSt6vectorIdEE), | cudaq::scalar_callback::operator= |
|     [\[1\]](api/languages/cpp_a   |     (C++                          |
| pi.html#_CPPv4NK5cudaq11noise_mod |     function)](api/languages/c    |
| el12get_channelsERKNSt6stringERKN | pp_api.html#_CPPv4N5cudaq15scalar |
| St6vectorINSt6size_tEEERKNSt6vect | _callbackaSERK15scalar_callback), |
| orINSt6size_tEEERKNSt6vectorIdEE) |     [\[1\]](api/languages/        |
| -                                 | cpp_api.html#_CPPv4N5cudaq15scala |
|  [cudaq::noise_model::noise_model | r_callbackaSERR15scalar_callback) |
|     (C++                          | -   [cudaq:                       |
|     function)](api                | :scalar_callback::scalar_callback |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq11noise_model11noise_modelEv) |     function)](api/languag        |
| -   [cu                           | es/cpp_api.html#_CPPv4I0_NSt11ena |
| daq::noise_model::PredicateFuncTy | ble_if_tINSt16is_invocable_r_vINS |
|     (C++                          | t7complexIdEE8CallableRKNSt13unor |
|     type)](api/la                 | dered_mapINSt6stringENSt7complexI |
| nguages/cpp_api.html#_CPPv4N5cuda | dEEEEEEbEEEN5cudaq15scalar_callba |
| q11noise_model15PredicateFuncTyE) | ck15scalar_callbackERR8Callable), |
| -   [cud                          |     [\[1\                         |
| aq::noise_model::register_channel | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_callback15scal |
|     function)](api/languages      | ar_callbackERK15scalar_callback), |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     [\[2                          |
| noise_model16register_channelEvv) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::                      | PPv4N5cudaq15scalar_callback15sca |
| noise_model::requires_constructor | lar_callbackERR15scalar_callback) |
|     (C++                          | -   [cudaq::scalar_operator (C++  |
|     type)](api/languages/cp       |     c                             |
| p_api.html#_CPPv4I0DpEN5cudaq11no | lass)](api/languages/cpp_api.html |
| ise_model20requires_constructorE) | #_CPPv4N5cudaq15scalar_operatorE) |
| -   [cudaq::noise_model_type (C++ | -                                 |
|     e                             | [cudaq::scalar_operator::evaluate |
| num)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16noise_model_typeE) |                                   |
| -   [cudaq::no                    |    function)](api/languages/cpp_a |
| ise_model_type::amplitude_damping | pi.html#_CPPv4NK5cudaq15scalar_op |
|     (C++                          | erator8evaluateERKNSt13unordered_ |
|     enumerator)](api/languages    | mapINSt6stringENSt7complexIdEEEE) |
| /cpp_api.html#_CPPv4N5cudaq16nois | -   [cudaq::scalar_ope            |
| e_model_type17amplitude_dampingE) | rator::get_parameter_descriptions |
| -   [cudaq::noise_mode            |     (C++                          |
| l_type::amplitude_damping_channel |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     e                             | tml#_CPPv4NK5cudaq15scalar_operat |
| numerator)](api/languages/cpp_api | or26get_parameter_descriptionsEv) |
| .html#_CPPv4N5cudaq16noise_model_ | -   [cu                           |
| type25amplitude_damping_channelE) | daq::scalar_operator::is_constant |
| -   [cudaq::n                     |     (C++                          |
| oise_model_type::bit_flip_channel |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4NK5cudaq |
|     enumerator)](api/language     | 15scalar_operator11is_constantEv) |
| s/cpp_api.html#_CPPv4N5cudaq16noi | -   [c                            |
| se_model_type16bit_flip_channelE) | udaq::scalar_operator::operator\* |
| -   [cudaq::                      |     (C++                          |
| noise_model_type::depolarization1 |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     enumerator)](api/languag      | Pv4N5cudaq15scalar_operatormlENSt |
| es/cpp_api.html#_CPPv4N5cudaq16no | 7complexIdEERK15scalar_operator), |
| ise_model_type15depolarization1E) |     [\[1\                         |
| -   [cudaq::                      | ]](api/languages/cpp_api.html#_CP |
| noise_model_type::depolarization2 | Pv4N5cudaq15scalar_operatormlENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     enumerator)](api/languag      |     [\[2\]](api/languages/cp      |
| es/cpp_api.html#_CPPv4N5cudaq16no | p_api.html#_CPPv4N5cudaq15scalar_ |
| ise_model_type15depolarization2E) | operatormlEdRK15scalar_operator), |
| -   [cudaq::noise_m               |     [\[3\]](api/languages/cp      |
| odel_type::depolarization_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormlEdRR15scalar_operator), |
|                                   |     [\[4\]](api/languages         |
|   enumerator)](api/languages/cpp_ | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| api.html#_CPPv4N5cudaq16noise_mod | alar_operatormlENSt7complexIdEE), |
| el_type22depolarization_channelE) |     [\[5\]](api/languages/cpp     |
| -                                 | _api.html#_CPPv4NKR5cudaq15scalar |
|  [cudaq::noise_model_type::pauli1 | _operatormlERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     enumerator)](a                | (api/languages/cpp_api.html#_CPPv |
| pi/languages/cpp_api.html#_CPPv4N | 4NKR5cudaq15scalar_operatormlEd), |
| 5cudaq16noise_model_type6pauli1E) |     [\[7\]](api/language          |
| -                                 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|  [cudaq::noise_model_type::pauli2 | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[8\]](api/languages/cp      |
|     enumerator)](a                | p_api.html#_CPPv4NO5cudaq15scalar |
| pi/languages/cpp_api.html#_CPPv4N | _operatormlERK15scalar_operator), |
| 5cudaq16noise_model_type6pauli2E) |     [\[9\                         |
| -   [cudaq                        | ]](api/languages/cpp_api.html#_CP |
| ::noise_model_type::phase_damping | Pv4NO5cudaq15scalar_operatormlEd) |
|     (C++                          | -   [cu                           |
|     enumerator)](api/langu        | daq::scalar_operator::operator\*= |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| noise_model_type13phase_dampingE) |     function)](api/languag        |
| -   [cudaq::noi                   | es/cpp_api.html#_CPPv4N5cudaq15sc |
| se_model_type::phase_flip_channel | alar_operatormLENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     enumerator)](api/languages/   | pp_api.html#_CPPv4N5cudaq15scalar |
| cpp_api.html#_CPPv4N5cudaq16noise | _operatormLERK15scalar_operator), |
| _model_type18phase_flip_channelE) |     [\[2                          |
| -                                 | \]](api/languages/cpp_api.html#_C |
| [cudaq::noise_model_type::unknown | PPv4N5cudaq15scalar_operatormLEd) |
|     (C++                          | -   [                             |
|     enumerator)](ap               | cudaq::scalar_operator::operator+ |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7unknownE) |     function                      |
| -                                 | )](api/languages/cpp_api.html#_CP |
| [cudaq::noise_model_type::x_error | Pv4N5cudaq15scalar_operatorplENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     enumerator)](ap               |     [\[1\                         |
| i/languages/cpp_api.html#_CPPv4N5 | ]](api/languages/cpp_api.html#_CP |
| cudaq16noise_model_type7x_errorE) | Pv4N5cudaq15scalar_operatorplENSt |
| -                                 | 7complexIdEERR15scalar_operator), |
| [cudaq::noise_model_type::y_error |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](ap               | operatorplEdRK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[3\]](api/languages/cp      |
| cudaq16noise_model_type7y_errorE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatorplEdRR15scalar_operator), |
| [cudaq::noise_model_type::z_error |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     enumerator)](ap               | alar_operatorplENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[5\]](api/languages/cpp     |
| cudaq16noise_model_type7z_errorE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::num_available_gpus    | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function                      | (api/languages/cpp_api.html#_CPPv |
| )](api/languages/cpp_api.html#_CP | 4NKR5cudaq15scalar_operatorplEd), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[7\]]                       |
| -   [cudaq::observe (C++          | (api/languages/cpp_api.html#_CPPv |
|     function)]                    | 4NKR5cudaq15scalar_operatorplEv), |
| (api/languages/cpp_api.html#_CPPv |     [\[8\]](api/language          |
| 4I00DpEN5cudaq7observeENSt6vector | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| I14observe_resultEERR13QuantumKer | alar_operatorplENSt7complexIdEE), |
| nelRK15SpinOpContainerDpRR4Args), |     [\[9\]](api/languages/cp      |
|     [\[1\]](api/languages/cpp_ap  | p_api.html#_CPPv4NO5cudaq15scalar |
| i.html#_CPPv4I0DpEN5cudaq7observe | _operatorplERK15scalar_operator), |
| E14observe_resultNSt6size_tERR13Q |     [\[10\]                       |
| uantumKernelRK7spin_opDpRR4Args), | ](api/languages/cpp_api.html#_CPP |
|     [\[                           | v4NO5cudaq15scalar_operatorplEd), |
| 2\]](api/languages/cpp_api.html#_ |     [\[11\                        |
| CPPv4I0DpEN5cudaq7observeE14obser | ]](api/languages/cpp_api.html#_CP |
| ve_resultRK15observe_optionsRR13Q | Pv4NO5cudaq15scalar_operatorplEv) |
| uantumKernelRK7spin_opDpRR4Args), | -   [c                            |
|     [\[3\]](api/lang              | udaq::scalar_operator::operator+= |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     (C++                          |
| udaq7observeE14observe_resultRR13 |     function)](api/languag        |
| QuantumKernelRK7spin_opDpRR4Args) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::observe_options (C++  | alar_operatorpLENSt7complexIdEE), |
|     st                            |     [\[1\]](api/languages/c       |
| ruct)](api/languages/cpp_api.html | pp_api.html#_CPPv4N5cudaq15scalar |
| #_CPPv4N5cudaq15observe_optionsE) | _operatorpLERK15scalar_operator), |
| -   [cudaq::observe_result (C++   |     [\[2                          |
|                                   | \]](api/languages/cpp_api.html#_C |
| class)](api/languages/cpp_api.htm | PPv4N5cudaq15scalar_operatorpLEd) |
| l#_CPPv4N5cudaq14observe_resultE) | -   [                             |
| -                                 | cudaq::scalar_operator::operator- |
|    [cudaq::observe_result::counts |     (C++                          |
|     (C++                          |     function                      |
|     function)](api/languages/c    | )](api/languages/cpp_api.html#_CP |
| pp_api.html#_CPPv4N5cudaq14observ | Pv4N5cudaq15scalar_operatormiENSt |
| e_result6countsERK12spin_op_term) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::observe_result::dump  |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)                     | Pv4N5cudaq15scalar_operatormiENSt |
| ](api/languages/cpp_api.html#_CPP | 7complexIdEERR15scalar_operator), |
| v4N5cudaq14observe_result4dumpEv) |     [\[2\]](api/languages/cp      |
| -   [c                            | p_api.html#_CPPv4N5cudaq15scalar_ |
| udaq::observe_result::expectation | operatormiEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
| function)](api/languages/cpp_api. | operatormiEdRR15scalar_operator), |
| html#_CPPv4N5cudaq14observe_resul |     [\[4\]](api/languages         |
| t11expectationERK12spin_op_term), | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     [\[1\]](api/la                | alar_operatormiENSt7complexIdEE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[5\]](api/languages/cpp     |
| q14observe_result11expectationEv) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cuda                         | _operatormiERK15scalar_operator), |
| q::observe_result::id_coefficient |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/langu          | 4NKR5cudaq15scalar_operatormiEd), |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     [\[7\]]                       |
| observe_result14id_coefficientEv) | (api/languages/cpp_api.html#_CPPv |
| -   [cuda                         | 4NKR5cudaq15scalar_operatormiEv), |
| q::observe_result::observe_result |     [\[8\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|                                   | alar_operatormiENSt7complexIdEE), |
|   function)](api/languages/cpp_ap |     [\[9\]](api/languages/cp      |
| i.html#_CPPv4N5cudaq14observe_res | p_api.html#_CPPv4NO5cudaq15scalar |
| ult14observe_resultEdRK7spin_op), | _operatormiERK15scalar_operator), |
|     [\[1\]](a                     |     [\[10\]                       |
| pi/languages/cpp_api.html#_CPPv4N | ](api/languages/cpp_api.html#_CPP |
| 5cudaq14observe_result14observe_r | v4NO5cudaq15scalar_operatormiEd), |
| esultEdRK7spin_op13sample_result) |     [\[11\                        |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|  [cudaq::observe_result::operator | Pv4NO5cudaq15scalar_operatormiEv) |
|     double (C++                   | -   [c                            |
|     functio                       | udaq::scalar_operator::operator-= |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq14observe_resultcvdEv) |     function)](api/languag        |
| -                                 | es/cpp_api.html#_CPPv4N5cudaq15sc |
|  [cudaq::observe_result::raw_data | alar_operatormIENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     function)](ap                 | pp_api.html#_CPPv4N5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatormIERK15scalar_operator), |
| cudaq14observe_result8raw_dataEv) |     [\[2                          |
| -   [cudaq::operator_handler (C++ | \]](api/languages/cpp_api.html#_C |
|     cl                            | PPv4N5cudaq15scalar_operatormIEd) |
| ass)](api/languages/cpp_api.html# | -   [                             |
| _CPPv4N5cudaq16operator_handlerE) | cudaq::scalar_operator::operator/ |
| -   [cudaq::optimizable_function  |     (C++                          |
|     (C++                          |     function                      |
|     class)                        | )](api/languages/cpp_api.html#_CP |
| ](api/languages/cpp_api.html#_CPP | Pv4N5cudaq15scalar_operatordvENSt |
| v4N5cudaq20optimizable_functionE) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::optimization_result   |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     type                          | Pv4N5cudaq15scalar_operatordvENSt |
| )](api/languages/cpp_api.html#_CP | 7complexIdEERR15scalar_operator), |
| Pv4N5cudaq19optimization_resultE) |     [\[2\]](api/languages/cp      |
| -   [cudaq::optimizer (C++        | p_api.html#_CPPv4N5cudaq15scalar_ |
|     class)](api/languages/cpp_a   | operatordvEdRK15scalar_operator), |
| pi.html#_CPPv4N5cudaq9optimizerE) |     [\[3\]](api/languages/cp      |
| -   [cudaq::optimizer::optimize   | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRR15scalar_operator), |
|                                   |     [\[4\]](api/languages         |
|  function)](api/languages/cpp_api | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| .html#_CPPv4N5cudaq9optimizer8opt | alar_operatordvENSt7complexIdEE), |
| imizeEKiRR20optimizable_function) |     [\[5\]](api/languages/cpp     |
| -   [cu                           | _api.html#_CPPv4NKR5cudaq15scalar |
| daq::optimizer::requiresGradients | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function)](api/la             | (api/languages/cpp_api.html#_CPPv |
| nguages/cpp_api.html#_CPPv4N5cuda | 4NKR5cudaq15scalar_operatordvEd), |
| q9optimizer17requiresGradientsEv) |     [\[7\]](api/language          |
| -   [cudaq::orca (C++             | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     type)](api/languages/         | alar_operatordvENSt7complexIdEE), |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     [\[8\]](api/languages/cp      |
| -   [cudaq::orca::sample (C++     | p_api.html#_CPPv4NO5cudaq15scalar |
|     function)](api/languages/c    | _operatordvERK15scalar_operator), |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     [\[9\                         |
| mpleERNSt6vectorINSt6size_tEEERNS | ]](api/languages/cpp_api.html#_CP |
| t6vectorINSt6size_tEEERNSt6vector | Pv4NO5cudaq15scalar_operatordvEd) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [c                            |
|     [\[1\]]                       | udaq::scalar_operator::operator/= |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq4orca6sampleERNSt6vectorI |     function)](api/languag        |
| NSt6size_tEEERNSt6vectorINSt6size | es/cpp_api.html#_CPPv4N5cudaq15sc |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | alar_operatordVENSt7complexIdEE), |
| -   [cudaq::orca::sample_async    |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|                                   | _operatordVERK15scalar_operator), |
| function)](api/languages/cpp_api. |     [\[2                          |
| html#_CPPv4N5cudaq4orca12sample_a | \]](api/languages/cpp_api.html#_C |
| syncERNSt6vectorINSt6size_tEEERNS | PPv4N5cudaq15scalar_operatordVEd) |
| t6vectorINSt6size_tEEERNSt6vector | -   [                             |
| IdEERNSt6vectorIdEEiNSt6size_tE), | cudaq::scalar_operator::operator= |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languages/c    |
| q4orca12sample_asyncERNSt6vectorI | pp_api.html#_CPPv4N5cudaq15scalar |
| NSt6size_tEEERNSt6vectorINSt6size | _operatoraSERK15scalar_operator), |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     [\[1\]](api/languages/        |
| -   [cudaq::OrcaRemoteRESTQPU     | cpp_api.html#_CPPv4N5cudaq15scala |
|     (C++                          | r_operatoraSERR15scalar_operator) |
|     cla                           | -   [c                            |
| ss)](api/languages/cpp_api.html#_ | udaq::scalar_operator::operator== |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |     (C++                          |
| -   [cudaq::pauli1 (C++           |     function)](api/languages/c    |
|     class)](api/languages/cp      | pp_api.html#_CPPv4NK5cudaq15scala |
| p_api.html#_CPPv4N5cudaq6pauli1E) | r_operatoreqERK15scalar_operator) |
| -                                 | -   [cudaq:                       |
|    [cudaq::pauli1::num_parameters | :scalar_operator::scalar_operator |
|     (C++                          |     (C++                          |
|     member)]                      |     func                          |
| (api/languages/cpp_api.html#_CPPv | tion)](api/languages/cpp_api.html |
| 4N5cudaq6pauli114num_parametersE) | #_CPPv4N5cudaq15scalar_operator15 |
| -   [cudaq::pauli1::num_targets   | scalar_operatorENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/langu             |
|     membe                         | ages/cpp_api.html#_CPPv4N5cudaq15 |
| r)](api/languages/cpp_api.html#_C | scalar_operator15scalar_operatorE |
| PPv4N5cudaq6pauli111num_targetsE) | RK15scalar_callbackRRNSt13unorder |
| -   [cudaq::pauli1::pauli1 (C++   | ed_mapINSt6stringENSt6stringEEE), |
|     function)](api/languages/cpp_ |     [\[2\                         |
| api.html#_CPPv4N5cudaq6pauli16pau | ]](api/languages/cpp_api.html#_CP |
| li1ERKNSt6vectorIN5cudaq4realEEE) | Pv4N5cudaq15scalar_operator15scal |
| -   [cudaq::pauli2 (C++           | ar_operatorERK15scalar_operator), |
|     class)](api/languages/cp      |     [\[3\]](api/langu             |
| p_api.html#_CPPv4N5cudaq6pauli2E) | ages/cpp_api.html#_CPPv4N5cudaq15 |
| -                                 | scalar_operator15scalar_operatorE |
|    [cudaq::pauli2::num_parameters | RR15scalar_callbackRRNSt13unorder |
|     (C++                          | ed_mapINSt6stringENSt6stringEEE), |
|     member)]                      |     [\[4\                         |
| (api/languages/cpp_api.html#_CPPv | ]](api/languages/cpp_api.html#_CP |
| 4N5cudaq6pauli214num_parametersE) | Pv4N5cudaq15scalar_operator15scal |
| -   [cudaq::pauli2::num_targets   | ar_operatorERR15scalar_operator), |
|     (C++                          |     [\[5\]](api/language          |
|     membe                         | s/cpp_api.html#_CPPv4N5cudaq15sca |
| r)](api/languages/cpp_api.html#_C | lar_operator15scalar_operatorEd), |
| PPv4N5cudaq6pauli211num_targetsE) |     [\[6\]](api/languag           |
| -   [cudaq::pauli2::pauli2 (C++   | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     function)](api/languages/cpp_ | alar_operator15scalar_operatorEv) |
| api.html#_CPPv4N5cudaq6pauli26pau | -   [                             |
| li2ERKNSt6vectorIN5cudaq4realEEE) | cudaq::scalar_operator::to_matrix |
| -   [cudaq::phase_damping (C++    |     (C++                          |
|                                   |                                   |
|  class)](api/languages/cpp_api.ht |   function)](api/languages/cpp_ap |
| ml#_CPPv4N5cudaq13phase_dampingE) | i.html#_CPPv4NK5cudaq15scalar_ope |
| -   [cud                          | rator9to_matrixERKNSt13unordered_ |
| aq::phase_damping::num_parameters | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [                             |
|     member)](api/lan              | cudaq::scalar_operator::to_string |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 13phase_damping14num_parametersE) |     function)](api/l              |
| -   [                             | anguages/cpp_api.html#_CPPv4NK5cu |
| cudaq::phase_damping::num_targets | daq15scalar_operator9to_stringEv) |
|     (C++                          | -   [cudaq::s                     |
|     member)](api/                 | calar_operator::\~scalar_operator |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq13phase_damping11num_targetsE) |     functio                       |
| -   [cudaq::phase_flip_channel    | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatorD0Ev) |
|     clas                          | -   [cudaq::set_noise (C++        |
| s)](api/languages/cpp_api.html#_C |     function)](api/langu          |
| PPv4N5cudaq18phase_flip_channelE) | ages/cpp_api.html#_CPPv4N5cudaq9s |
| -   [cudaq::p                     | et_noiseERKN5cudaq11noise_modelE) |
| hase_flip_channel::num_parameters | -   [cudaq::set_random_seed (C++  |
|     (C++                          |     function)](api/               |
|     member)](api/language         | languages/cpp_api.html#_CPPv4N5cu |
| s/cpp_api.html#_CPPv4N5cudaq18pha | daq15set_random_seedENSt6size_tE) |
| se_flip_channel14num_parametersE) | -   [cudaq::simulation_precision  |
| -   [cudaq                        |     (C++                          |
| ::phase_flip_channel::num_targets |     enum)                         |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/langu            | v4N5cudaq20simulation_precisionE) |
| ages/cpp_api.html#_CPPv4N5cudaq18 | -   [                             |
| phase_flip_channel11num_targetsE) | cudaq::simulation_precision::fp32 |
| -   [cudaq::product_op (C++       |     (C++                          |
|                                   |     enumerator)](api              |
|  class)](api/languages/cpp_api.ht | /languages/cpp_api.html#_CPPv4N5c |
| ml#_CPPv4I0EN5cudaq10product_opE) | udaq20simulation_precision4fp32E) |
| -   [cudaq::product_op::begin     | -   [                             |
|     (C++                          | cudaq::simulation_precision::fp64 |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |     enumerator)](api              |
| PPv4NK5cudaq10product_op5beginEv) | /languages/cpp_api.html#_CPPv4N5c |
| -                                 | udaq20simulation_precision4fp64E) |
|  [cudaq::product_op::canonicalize | -   [cudaq::SimulationState (C++  |
|     (C++                          |     c                             |
|     func                          | lass)](api/languages/cpp_api.html |
| tion)](api/languages/cpp_api.html | #_CPPv4N5cudaq15SimulationStateE) |
| #_CPPv4N5cudaq10product_op12canon | -   [                             |
| icalizeERKNSt3setINSt6size_tEEE), | cudaq::SimulationState::precision |
|     [\[1\]](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     enum)](api                    |
| udaq10product_op12canonicalizeEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [                             | udaq15SimulationState9precisionE) |
| cudaq::product_op::const_iterator | -   [cudaq:                       |
|     (C++                          | :SimulationState::precision::fp32 |
|     struct)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     enumerator)](api/lang         |
| daq10product_op14const_iteratorE) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [cudaq::product_o             | 5SimulationState9precision4fp32E) |
| p::const_iterator::const_iterator | -   [cudaq:                       |
|     (C++                          | :SimulationState::precision::fp64 |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     enumerator)](api/lang         |
| ml#_CPPv4N5cudaq10product_op14con | uages/cpp_api.html#_CPPv4N5cudaq1 |
| st_iterator14const_iteratorEPK10p | 5SimulationState9precision4fp64E) |
| roduct_opI9HandlerTyENSt6size_tE) | -                                 |
| -   [cudaq::produ                 |   [cudaq::SimulationState::Tensor |
| ct_op::const_iterator::operator!= |     (C++                          |
|     (C++                          |     struct)](                     |
|     fun                           | api/languages/cpp_api.html#_CPPv4 |
| ction)](api/languages/cpp_api.htm | N5cudaq15SimulationState6TensorE) |
| l#_CPPv4NK5cudaq10product_op14con | -   [cudaq::spin_handler (C++     |
| st_iteratorneERK14const_iterator) |                                   |
| -   [cudaq::produ                 |   class)](api/languages/cpp_api.h |
| ct_op::const_iterator::operator\* | tml#_CPPv4N5cudaq12spin_handlerE) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/lang           | :spin_handler::to_diagonal_matrix |
| uages/cpp_api.html#_CPPv4NK5cudaq |     (C++                          |
| 10product_op14const_iteratormlEv) |     function)](api/la             |
| -   [cudaq::produ                 | nguages/cpp_api.html#_CPPv4NK5cud |
| ct_op::const_iterator::operator++ | aq12spin_handler18to_diagonal_mat |
|     (C++                          | rixERNSt13unordered_mapINSt6size_ |
|     function)](api/lang           | tENSt7int64_tEEERKNSt13unordered_ |
| uages/cpp_api.html#_CPPv4N5cudaq1 | mapINSt6stringENSt7complexIdEEEE) |
| 0product_op14const_iteratorppEi), | -                                 |
|     [\[1\]](api/lan               |   [cudaq::spin_handler::to_matrix |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 10product_op14const_iteratorppEv) |     function                      |
| -   [cudaq::produc                | )](api/languages/cpp_api.html#_CP |
| t_op::const_iterator::operator\-- | Pv4N5cudaq12spin_handler9to_matri |
|     (C++                          | xERKNSt6stringENSt7complexIdEEb), |
|     function)](api/lang           |     [\[1                          |
| uages/cpp_api.html#_CPPv4N5cudaq1 | \]](api/languages/cpp_api.html#_C |
| 0product_op14const_iteratormmEi), | PPv4NK5cudaq12spin_handler9to_mat |
|     [\[1\]](api/lan               | rixERNSt13unordered_mapINSt6size_ |
| guages/cpp_api.html#_CPPv4N5cudaq | tENSt7int64_tEEERKNSt13unordered_ |
| 10product_op14const_iteratormmEv) | mapINSt6stringENSt7complexIdEEEE) |
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
