::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4276
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
| -   [circuit_location             | alar_operatorRR10product_opI1TE), |
|     (cudaq.ptsbe.KrausSelection   |     [\[4\]](api/                  |
|     property)](api/lang           | languages/cpp_api.html#_CPPv4I0EN |
| uages/python_api.html#cudaq.ptsbe | 5cudaq10product_opmlE6sum_opI1TER |
| .KrausSelection.circuit_location) | K15scalar_operatorRK6sum_opI1TE), |
| -   [clear (cudaq.Resources       |     [\[5\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|   attribute)](api/languages/pytho | 5cudaq10product_opmlE6sum_opI1TER |
| n_api.html#cudaq.Resources.clear) | K15scalar_operatorRR6sum_opI1TE), |
|     -   [(cudaq.SampleResult      |     [\[6\]](api/                  |
|         a                         | languages/cpp_api.html#_CPPv4I0EN |
| ttribute)](api/languages/python_a | 5cudaq10product_opmlE6sum_opI1TER |
| pi.html#cudaq.SampleResult.clear) | R15scalar_operatorRK6sum_opI1TE), |
| -   [COBYLA (class in             |     [\[7\]](api/                  |
|     cudaq.o                       | languages/cpp_api.html#_CPPv4I0EN |
| ptimizers)](api/languages/python_ | 5cudaq10product_opmlE6sum_opI1TER |
| api.html#cudaq.optimizers.COBYLA) | R15scalar_operatorRR6sum_opI1TE), |
| -   [coefficient                  |     [\[8\]](api/languages         |
|     (cudaq.                       | /cpp_api.html#_CPPv4NK5cudaq10pro |
| operators.boson.BosonOperatorTerm | duct_opmlERK6sum_opI9HandlerTyE), |
|     property)](api/languages/py   |     [\[9\]](api/languages/cpp_a   |
| thon_api.html#cudaq.operators.bos | pi.html#_CPPv4NKR5cudaq10product_ |
| on.BosonOperatorTerm.coefficient) | opmlERK10product_opI9HandlerTyE), |
|     -   [(cudaq.oper              |     [\[10\]](api/language         |
| ators.fermion.FermionOperatorTerm | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opmlERK15scalar_operator), |
|   property)](api/languages/python |     [\[11\]](api/languages/cpp_a  |
| _api.html#cudaq.operators.fermion | pi.html#_CPPv4NKR5cudaq10product_ |
| .FermionOperatorTerm.coefficient) | opmlERR10product_opI9HandlerTyE), |
|     -   [(c                       |     [\[12\]](api/language         |
| udaq.operators.MatrixOperatorTerm | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|         property)](api/languag    | roduct_opmlERR15scalar_operator), |
| es/python_api.html#cudaq.operator |     [\[13\]](api/languages/cpp_   |
| s.MatrixOperatorTerm.coefficient) | api.html#_CPPv4NO5cudaq10product_ |
|     -   [(cuda                    | opmlERK10product_opI9HandlerTyE), |
| q.operators.spin.SpinOperatorTerm |     [\[14\]](api/languag          |
|         property)](api/languages/ | es/cpp_api.html#_CPPv4NO5cudaq10p |
| python_api.html#cudaq.operators.s | roduct_opmlERK15scalar_operator), |
| pin.SpinOperatorTerm.coefficient) |     [\[15\]](api/languages/cpp_   |
| -   [col_count                    | api.html#_CPPv4NO5cudaq10product_ |
|     (cudaq.KrausOperator          | opmlERR10product_opI9HandlerTyE), |
|     prope                         |     [\[16\]](api/langua           |
| rty)](api/languages/python_api.ht | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| ml#cudaq.KrausOperator.col_count) | product_opmlERR15scalar_operator) |
| -   [compile()                    | -                                 |
|     (cudaq.PyKernelDecorator      |   [cudaq::product_op::operator\*= |
|     metho                         |     (C++                          |
| d)](api/languages/python_api.html |     function)](api/languages/cpp  |
| #cudaq.PyKernelDecorator.compile) | _api.html#_CPPv4N5cudaq10product_ |
| -   [ComplexMatrix (class in      | opmLERK10product_opI9HandlerTyE), |
|     cudaq)](api/languages/pyt     |     [\[1\]](api/langua            |
| hon_api.html#cudaq.ComplexMatrix) | ges/cpp_api.html#_CPPv4N5cudaq10p |
| -   [compute                      | roduct_opmLERK15scalar_operator), |
|     (                             |     [\[2\]](api/languages/cp      |
| cudaq.gradients.CentralDifference | p_api.html#_CPPv4N5cudaq10product |
|     attribute)](api/la            | _opmLERR10product_opI9HandlerTyE) |
| nguages/python_api.html#cudaq.gra | -   [cudaq::product_op::operator+ |
| dients.CentralDifference.compute) |     (C++                          |
|     -   [(                        |     function)](api/langu          |
| cudaq.gradients.ForwardDifference | ages/cpp_api.html#_CPPv4I0EN5cuda |
|         attribute)](api/la        | q10product_opplE6sum_opI1TERK15sc |
| nguages/python_api.html#cudaq.gra | alar_operatorRK10product_opI1TE), |
| dients.ForwardDifference.compute) |     [\[1\]](api/                  |
|     -                             | languages/cpp_api.html#_CPPv4I0EN |
|  [(cudaq.gradients.ParameterShift | 5cudaq10product_opplE6sum_opI1TER |
|         attribute)](api           | K15scalar_operatorRK6sum_opI1TE), |
| /languages/python_api.html#cudaq. |     [\[2\]](api/langu             |
| gradients.ParameterShift.compute) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [const()                      | q10product_opplE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|   (cudaq.operators.ScalarOperator |     [\[3\]](api/                  |
|     class                         | languages/cpp_api.html#_CPPv4I0EN |
|     method)](a                    | 5cudaq10product_opplE6sum_opI1TER |
| pi/languages/python_api.html#cuda | K15scalar_operatorRR6sum_opI1TE), |
| q.operators.ScalarOperator.const) |     [\[4\]](api/langu             |
| -   [controls                     | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (cudaq.ptsbe.TraceInstruction | q10product_opplE6sum_opI1TERR15sc |
|     property)](ap                 | alar_operatorRK10product_opI1TE), |
| i/languages/python_api.html#cudaq |     [\[5\]](api/                  |
| .ptsbe.TraceInstruction.controls) | languages/cpp_api.html#_CPPv4I0EN |
| -   [copy                         | 5cudaq10product_opplE6sum_opI1TER |
|     (cu                           | R15scalar_operatorRK6sum_opI1TE), |
| daq.operators.boson.BosonOperator |     [\[6\]](api/langu             |
|     attribute)](api/l             | ages/cpp_api.html#_CPPv4I0EN5cuda |
| anguages/python_api.html#cudaq.op | q10product_opplE6sum_opI1TERR15sc |
| erators.boson.BosonOperator.copy) | alar_operatorRR10product_opI1TE), |
|     -   [(cudaq.                  |     [\[7\]](api/                  |
| operators.boson.BosonOperatorTerm | languages/cpp_api.html#_CPPv4I0EN |
|         attribute)](api/langu     | 5cudaq10product_opplE6sum_opI1TER |
| ages/python_api.html#cudaq.operat | R15scalar_operatorRR6sum_opI1TE), |
| ors.boson.BosonOperatorTerm.copy) |     [\[8\]](api/languages/cpp_a   |
|     -   [(cudaq.                  | pi.html#_CPPv4NKR5cudaq10product_ |
| operators.fermion.FermionOperator | opplERK10product_opI9HandlerTyE), |
|         attribute)](api/langu     |     [\[9\]](api/language          |
| ages/python_api.html#cudaq.operat | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ors.fermion.FermionOperator.copy) | roduct_opplERK15scalar_operator), |
|     -   [(cudaq.oper              |     [\[10\]](api/languages/       |
| ators.fermion.FermionOperatorTerm | cpp_api.html#_CPPv4NKR5cudaq10pro |
|         attribute)](api/languages | duct_opplERK6sum_opI9HandlerTyE), |
| /python_api.html#cudaq.operators. |     [\[11\]](api/languages/cpp_a  |
| fermion.FermionOperatorTerm.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -                             | opplERR10product_opI9HandlerTyE), |
|  [(cudaq.operators.MatrixOperator |     [\[12\]](api/language         |
|         attribute)](              | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| api/languages/python_api.html#cud | roduct_opplERR15scalar_operator), |
| aq.operators.MatrixOperator.copy) |     [\[13\]](api/languages/       |
|     -   [(c                       | cpp_api.html#_CPPv4NKR5cudaq10pro |
| udaq.operators.MatrixOperatorTerm | duct_opplERR6sum_opI9HandlerTyE), |
|         attribute)](api/          |     [\[                           |
| languages/python_api.html#cudaq.o | 14\]](api/languages/cpp_api.html# |
| perators.MatrixOperatorTerm.copy) | _CPPv4NKR5cudaq10product_opplEv), |
|     -   [(                        |     [\[15\]](api/languages/cpp_   |
| cudaq.operators.spin.SpinOperator | api.html#_CPPv4NO5cudaq10product_ |
|         attribute)](api           | opplERK10product_opI9HandlerTyE), |
| /languages/python_api.html#cudaq. |     [\[16\]](api/languag          |
| operators.spin.SpinOperator.copy) | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     -   [(cuda                    | roduct_opplERK15scalar_operator), |
| q.operators.spin.SpinOperatorTerm |     [\[17\]](api/languages        |
|         attribute)](api/lan       | /cpp_api.html#_CPPv4NO5cudaq10pro |
| guages/python_api.html#cudaq.oper | duct_opplERK6sum_opI9HandlerTyE), |
| ators.spin.SpinOperatorTerm.copy) |     [\[18\]](api/languages/cpp_   |
| -   [count (cudaq.Resources       | api.html#_CPPv4NO5cudaq10product_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|   attribute)](api/languages/pytho |     [\[19\]](api/languag          |
| n_api.html#cudaq.Resources.count) | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     -   [(cudaq.SampleResult      | roduct_opplERR15scalar_operator), |
|         a                         |     [\[20\]](api/languages        |
| ttribute)](api/languages/python_a | /cpp_api.html#_CPPv4NO5cudaq10pro |
| pi.html#cudaq.SampleResult.count) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [count_controls               |     [                             |
|     (cudaq.Resources              | \[21\]](api/languages/cpp_api.htm |
|     attribu                       | l#_CPPv4NO5cudaq10product_opplEv) |
| te)](api/languages/python_api.htm | -   [cudaq::product_op::operator- |
| l#cudaq.Resources.count_controls) |     (C++                          |
| -   [count_instructions           |     function)](api/langu          |
|                                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|   (cudaq.ptsbe.PTSBEExecutionData | q10product_opmiE6sum_opI1TERK15sc |
|     attribute)](api/languages/    | alar_operatorRK10product_opI1TE), |
| python_api.html#cudaq.ptsbe.PTSBE |     [\[1\]](api/                  |
| ExecutionData.count_instructions) | languages/cpp_api.html#_CPPv4I0EN |
| -   [counts (cudaq.ObserveResult  | 5cudaq10product_opmiE6sum_opI1TER |
|     att                           | K15scalar_operatorRK6sum_opI1TE), |
| ribute)](api/languages/python_api |     [\[2\]](api/langu             |
| .html#cudaq.ObserveResult.counts) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [csr_spmatrix (C++            | q10product_opmiE6sum_opI1TERK15sc |
|     type)](api/languages/c        | alar_operatorRR10product_opI1TE), |
| pp_api.html#_CPPv412csr_spmatrix) |     [\[3\]](api/                  |
| -   cudaq                         | languages/cpp_api.html#_CPPv4I0EN |
|     -   [module](api/langua       | 5cudaq10product_opmiE6sum_opI1TER |
| ges/python_api.html#module-cudaq) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq (C++                   |     [\[4\]](api/langu             |
|     type)](api/lan                | ages/cpp_api.html#_CPPv4I0EN5cuda |
| guages/cpp_api.html#_CPPv45cudaq) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq.apply_noise() (in      | alar_operatorRK10product_opI1TE), |
|     module                        |     [\[5\]](api/                  |
|     cudaq)](api/languages/python_ | languages/cpp_api.html#_CPPv4I0EN |
| api.html#cudaq.cudaq.apply_noise) | 5cudaq10product_opmiE6sum_opI1TER |
| -   cudaq.boson                   | R15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/languages/py |     [\[6\]](api/langu             |
| thon_api.html#module-cudaq.boson) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq.fermion                 | q10product_opmiE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|   -   [module](api/languages/pyth |     [\[7\]](api/                  |
| on_api.html#module-cudaq.fermion) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.operators.custom        | 5cudaq10product_opmiE6sum_opI1TER |
|     -   [mo                       | R15scalar_operatorRR6sum_opI1TE), |
| dule](api/languages/python_api.ht |     [\[8\]](api/languages/cpp_a   |
| ml#module-cudaq.operators.custom) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.spin                    | opmiERK10product_opI9HandlerTyE), |
|     -   [module](api/languages/p  |     [\[9\]](api/language          |
| ython_api.html#module-cudaq.spin) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::amplitude_damping     | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[10\]](api/languages/       |
|     cla                           | cpp_api.html#_CPPv4NKR5cudaq10pro |
| ss)](api/languages/cpp_api.html#_ | duct_opmiERK6sum_opI9HandlerTyE), |
| CPPv4N5cudaq17amplitude_dampingE) |     [\[11\]](api/languages/cpp_a  |
| -                                 | pi.html#_CPPv4NKR5cudaq10product_ |
| [cudaq::amplitude_damping_channel | opmiERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[12\]](api/language         |
|     class)](api                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /languages/cpp_api.html#_CPPv4N5c | roduct_opmiERR15scalar_operator), |
| udaq25amplitude_damping_channelE) |     [\[13\]](api/languages/       |
| -   [cudaq::amplitud              | cpp_api.html#_CPPv4NKR5cudaq10pro |
| e_damping_channel::num_parameters | duct_opmiERR6sum_opI9HandlerTyE), |
|     (C++                          |     [\[                           |
|     member)](api/languages/cpp_a  | 14\]](api/languages/cpp_api.html# |
| pi.html#_CPPv4N5cudaq25amplitude_ | _CPPv4NKR5cudaq10product_opmiEv), |
| damping_channel14num_parametersE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::ampli                 | api.html#_CPPv4NO5cudaq10product_ |
| tude_damping_channel::num_targets | opmiERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[16\]](api/languag          |
|     member)](api/languages/cp     | es/cpp_api.html#_CPPv4NO5cudaq10p |
| p_api.html#_CPPv4N5cudaq25amplitu | roduct_opmiERK15scalar_operator), |
| de_damping_channel11num_targetsE) |     [\[17\]](api/languages        |
| -   [cudaq::AnalogRemoteRESTQPU   | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERK6sum_opI9HandlerTyE), |
|     class                         |     [\[18\]](api/languages/cpp_   |
| )](api/languages/cpp_api.html#_CP | api.html#_CPPv4NO5cudaq10product_ |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | opmiERR10product_opI9HandlerTyE), |
| -   [cudaq::apply_noise (C++      |     [\[19\]](api/languag          |
|     function)](api/               | es/cpp_api.html#_CPPv4NO5cudaq10p |
| languages/cpp_api.html#_CPPv4I0Dp | roduct_opmiERR15scalar_operator), |
| EN5cudaq11apply_noiseEvDpRR4Args) |     [\[20\]](api/languages        |
| -   [cudaq::async_result (C++     | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     c                             | duct_opmiERR6sum_opI9HandlerTyE), |
| lass)](api/languages/cpp_api.html |     [                             |
| #_CPPv4I0EN5cudaq12async_resultE) | \[21\]](api/languages/cpp_api.htm |
| -   [cudaq::async_result::get     | l#_CPPv4NO5cudaq10product_opmiEv) |
|     (C++                          | -   [cudaq::product_op::operator/ |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)](api/language       |
| CPPv4N5cudaq12async_result3getEv) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::async_sample_result   | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/language          |
|     type                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| )](api/languages/cpp_api.html#_CP | roduct_opdvERR15scalar_operator), |
| Pv4N5cudaq19async_sample_resultE) |     [\[2\]](api/languag           |
| -   [cudaq::BaseRemoteRESTQPU     | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opdvERK15scalar_operator), |
|     cla                           |     [\[3\]](api/langua            |
| ss)](api/languages/cpp_api.html#_ | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | product_opdvERR15scalar_operator) |
| -   [cudaq::bit_flip_channel (C++ | -                                 |
|     cl                            |    [cudaq::product_op::operator/= |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16bit_flip_channelE) |     function)](api/langu          |
| -   [cudaq:                       | ages/cpp_api.html#_CPPv4N5cudaq10 |
| :bit_flip_channel::num_parameters | product_opdVERK15scalar_operator) |
|     (C++                          | -   [cudaq::product_op::operator= |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16b |     function)](api/la             |
| it_flip_channel14num_parametersE) | nguages/cpp_api.html#_CPPv4I0_NSt |
| -   [cud                          | 11enable_if_tIXaantNSt7is_sameI1T |
| aq::bit_flip_channel::num_targets | 9HandlerTyE5valueENSt16is_constru |
|     (C++                          | ctibleI9HandlerTy1TE5valueEEbEEEN |
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
| -   [cudaq::complex (C++          |     function)](api/languages/c    |
|     type)](api/languages/cpp      | pp_api.html#_CPPv4I0_NSt11enable_ |
| _api.html#_CPPv4N5cudaq7complexE) | if_tIXaaNSt7is_sameI9HandlerTy14m |
| -   [cudaq::complex_matrix (C++   | atrix_handlerE5valueEaantNSt7is_s |
|                                   | ameI1T9HandlerTyE5valueENSt16is_c |
| class)](api/languages/cpp_api.htm | onstructibleI9HandlerTy1TE5valueE |
| l#_CPPv4N5cudaq14complex_matrixE) | EbEEEN5cudaq10product_op10product |
| -                                 | _opERK10product_opI1TERKN14matrix |
|   [cudaq::complex_matrix::adjoint | _handler20commutation_behaviorE), |
|     (C++                          |                                   |
|     function)](a                  |  [\[1\]](api/languages/cpp_api.ht |
| pi/languages/cpp_api.html#_CPPv4N | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| 5cudaq14complex_matrix7adjointEv) | tNSt7is_sameI1T9HandlerTyE5valueE |
| -   [cudaq::                      | NSt16is_constructibleI9HandlerTy1 |
| complex_matrix::diagonal_elements | TE5valueEEbEEEN5cudaq10product_op |
|     (C++                          | 10product_opERK10product_opI1TE), |
|     function)](api/languages      |                                   |
| /cpp_api.html#_CPPv4NK5cudaq14com |   [\[2\]](api/languages/cpp_api.h |
| plex_matrix17diagonal_elementsEi) | tml#_CPPv4N5cudaq10product_op10pr |
| -   [cudaq::complex_matrix::dump  | oduct_opENSt6size_tENSt6size_tE), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     function)](api/language       | p_api.html#_CPPv4N5cudaq10product |
| s/cpp_api.html#_CPPv4NK5cudaq14co | _op10product_opENSt7complexIdEE), |
| mplex_matrix4dumpERNSt7ostreamE), |     [\[4\]](api/l                 |
|     [\[1\]]                       | anguages/cpp_api.html#_CPPv4N5cud |
| (api/languages/cpp_api.html#_CPPv | aq10product_op10product_opERK10pr |
| 4NK5cudaq14complex_matrix4dumpEv) | oduct_opI9HandlerTyENSt6size_tE), |
| -   [c                            |     [\[5\]](api/l                 |
| udaq::complex_matrix::eigenvalues | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq10product_op10product_opERR10pr |
|     function)](api/lan            | oduct_opI9HandlerTyENSt6size_tE), |
| guages/cpp_api.html#_CPPv4NK5cuda |     [\[6\]](api/languages         |
| q14complex_matrix11eigenvaluesEv) | /cpp_api.html#_CPPv4N5cudaq10prod |
| -   [cu                           | uct_op10product_opERR9HandlerTy), |
| daq::complex_matrix::eigenvectors |     [\[7\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/lang           | cudaq10product_op10product_opEd), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[8\]](a                     |
| 14complex_matrix12eigenvectorsEv) | pi/languages/cpp_api.html#_CPPv4N |
| -   [c                            | 5cudaq10product_op10product_opEv) |
| udaq::complex_matrix::exponential | -   [cuda                         |
|     (C++                          | q::product_op::to_diagonal_matrix |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/               |
| q14complex_matrix11exponentialEv) | languages/cpp_api.html#_CPPv4NK5c |
| -                                 | udaq10product_op18to_diagonal_mat |
|  [cudaq::complex_matrix::identity | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/languages      | apINSt6stringENSt7complexIdEEEEb) |
| /cpp_api.html#_CPPv4N5cudaq14comp | -   [cudaq::product_op::to_matrix |
| lex_matrix8identityEKNSt6size_tE) |     (C++                          |
| -                                 |     funct                         |
| [cudaq::complex_matrix::kronecker | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NK5cudaq10product_op9to_mat |
|     function)](api/lang           | rixENSt13unordered_mapINSt6size_t |
| uages/cpp_api.html#_CPPv4I00EN5cu | ENSt7int64_tEEERKNSt13unordered_m |
| daq14complex_matrix9kroneckerE14c | apINSt6stringENSt7complexIdEEEEb) |
| omplex_matrix8Iterable8Iterable), | -   [cu                           |
|     [\[1\]](api/l                 | daq::product_op::to_sparse_matrix |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq14complex_matrix9kroneckerERK14 |     function)](ap                 |
| complex_matrixRK14complex_matrix) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::c                     | 5cudaq10product_op16to_sparse_mat |
| omplex_matrix::minimal_eigenvalue | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/languages/     | apINSt6stringENSt7complexIdEEEEb) |
| cpp_api.html#_CPPv4NK5cudaq14comp | -   [cudaq::product_op::to_string |
| lex_matrix18minimal_eigenvalueEv) |     (C++                          |
| -   [                             |     function)](                   |
| cudaq::complex_matrix::operator() | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq10product_op9to_stringEv) |
|     function)](api/languages/cpp  | -                                 |
| _api.html#_CPPv4N5cudaq14complex_ |  [cudaq::product_op::\~product_op |
| matrixclENSt6size_tENSt6size_tE), |     (C++                          |
|     [\[1\]](api/languages/cpp     |     fu                            |
| _api.html#_CPPv4NK5cudaq14complex | nction)](api/languages/cpp_api.ht |
| _matrixclENSt6size_tENSt6size_tE) | ml#_CPPv4N5cudaq10product_opD0Ev) |
| -   [                             | -   [cudaq::ptsbe (C++            |
| cudaq::complex_matrix::operator\* |     type)](api/languages/c        |
|     (C++                          | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
|     function)](api/langua         | -   [cudaq::p                     |
| ges/cpp_api.html#_CPPv4N5cudaq14c | tsbe::ConditionalSamplingStrategy |
| omplex_matrixmlEN14complex_matrix |     (C++                          |
| 10value_typeERK14complex_matrix), |     class)](api/languag           |
|     [\[1\]                        | es/cpp_api.html#_CPPv4N5cudaq5pts |
| ](api/languages/cpp_api.html#_CPP | be27ConditionalSamplingStrategyE) |
| v4N5cudaq14complex_matrixmlERK14c | -   [cudaq::ptsbe::C              |
| omplex_matrixRK14complex_matrix), | onditionalSamplingStrategy::clone |
|                                   |     (C++                          |
|  [\[2\]](api/languages/cpp_api.ht |                                   |
| ml#_CPPv4N5cudaq14complex_matrixm |    function)](api/languages/cpp_a |
| lERK14complex_matrixRKNSt6vectorI | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| N14complex_matrix10value_typeEEE) | ditionalSamplingStrategy5cloneEv) |
| -                                 | -   [cuda                         |
| [cudaq::complex_matrix::operator+ | q::ptsbe::ConditionalSamplingStra |
|     (C++                          | tegy::ConditionalSamplingStrategy |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/lang           |
| Pv4N5cudaq14complex_matrixplERK14 | uages/cpp_api.html#_CPPv4N5cudaq5 |
| complex_matrixRK14complex_matrix) | ptsbe27ConditionalSamplingStrateg |
| -                                 | y27ConditionalSamplingStrategyE19 |
| [cudaq::complex_matrix::operator- | TrajectoryPredicateNSt8uint64_tE) |
|     (C++                          | -                                 |
|     function                      |   [cudaq::ptsbe::ConditionalSampl |
| )](api/languages/cpp_api.html#_CP | ingStrategy::generateTrajectories |
| Pv4N5cudaq14complex_matrixmiERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     function)](api/language       |
| -   [cu                           | s/cpp_api.html#_CPPv4NK5cudaq5pts |
| daq::complex_matrix::operator\[\] | be27ConditionalSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|                                   | detail10NoisePointEEENSt6size_tE) |
|  function)](api/languages/cpp_api | -   [cudaq::ptsbe::               |
| .html#_CPPv4N5cudaq14complex_matr | ConditionalSamplingStrategy::name |
| ixixERKNSt6vectorINSt6size_tEEE), |     (C++                          |
|     [\[1\]](api/languages/cpp_api |     function)](api/languages/cpp_ |
| .html#_CPPv4NK5cudaq14complex_mat | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| rixixERKNSt6vectorINSt6size_tEEE) | nditionalSamplingStrategy4nameEv) |
| -   [cudaq::complex_matrix::power | -   [cudaq:                       |
|     (C++                          | :ptsbe::ConditionalSamplingStrate |
|     function)]                    | gy::\~ConditionalSamplingStrategy |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq14complex_matrix5powerEi) |     function)](api/languages/     |
| -                                 | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
|  [cudaq::complex_matrix::set_zero | 7ConditionalSamplingStrategyD0Ev) |
|     (C++                          | -                                 |
|     function)](ap                 | [cudaq::ptsbe::detail::NoisePoint |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq14complex_matrix8set_zeroEv) |     struct)](a                    |
| -                                 | pi/languages/cpp_api.html#_CPPv4N |
| [cudaq::complex_matrix::to_string | 5cudaq5ptsbe6detail10NoisePointE) |
|     (C++                          | -   [cudaq::p                     |
|     function)](api/               | tsbe::detail::NoisePoint::channel |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14complex_matrix9to_stringEv) |     member)](api/langu            |
| -   [                             | ages/cpp_api.html#_CPPv4N5cudaq5p |
| cudaq::complex_matrix::value_type | tsbe6detail10NoisePoint7channelE) |
|     (C++                          | -   [cudaq::ptsbe::det            |
|     type)](api/                   | ail::NoisePoint::circuit_location |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq14complex_matrix10value_typeE) |     member)](api/languages/cpp_a  |
| -   [cudaq::contrib (C++          | pi.html#_CPPv4N5cudaq5ptsbe6detai |
|     type)](api/languages/cpp      | l10NoisePoint16circuit_locationE) |
| _api.html#_CPPv4N5cudaq7contribE) | -   [cudaq::p                     |
| -   [cudaq::contrib::draw (C++    | tsbe::detail::NoisePoint::op_name |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     member)](api/langu            |
| v4I0DpEN5cudaq7contrib4drawENSt6s | ages/cpp_api.html#_CPPv4N5cudaq5p |
| tringERR13QuantumKernelDpRR4Args) | tsbe6detail10NoisePoint7op_nameE) |
| -                                 | -   [cudaq::                      |
| [cudaq::contrib::get_unitary_cmat | ptsbe::detail::NoisePoint::qubits |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     member)](api/lang             |
| p_api.html#_CPPv4I0DpEN5cudaq7con | uages/cpp_api.html#_CPPv4N5cudaq5 |
| trib16get_unitary_cmatE14complex_ | ptsbe6detail10NoisePoint6qubitsE) |
| matrixRR13QuantumKernelDpRR4Args) | -   [cudaq::                      |
| -   [cudaq::CusvState (C++        | ptsbe::ExhaustiveSamplingStrategy |
|                                   |     (C++                          |
|    class)](api/languages/cpp_api. |     class)](api/langua            |
| html#_CPPv4I0EN5cudaq9CusvStateE) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq::depolarization1 (C++  | sbe26ExhaustiveSamplingStrategyE) |
|     c                             | -   [cudaq::ptsbe::               |
| lass)](api/languages/cpp_api.html | ExhaustiveSamplingStrategy::clone |
| #_CPPv4N5cudaq15depolarization1E) |     (C++                          |
| -   [cudaq::depolarization2 (C++  |     function)](api/languages/cpp_ |
|     c                             | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| lass)](api/languages/cpp_api.html | haustiveSamplingStrategy5cloneEv) |
| #_CPPv4N5cudaq15depolarization2E) | -   [cu                           |
| -   [cudaq:                       | daq::ptsbe::ExhaustiveSamplingStr |
| :depolarization2::depolarization2 | ategy::ExhaustiveSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/la             |
| p_api.html#_CPPv4N5cudaq15depolar | nguages/cpp_api.html#_CPPv4N5cuda |
| ization215depolarization2EK4real) | q5ptsbe26ExhaustiveSamplingStrate |
| -   [cudaq                        | gy26ExhaustiveSamplingStrategyEv) |
| ::depolarization2::num_parameters | -                                 |
|     (C++                          |    [cudaq::ptsbe::ExhaustiveSampl |
|     member)](api/langu            | ingStrategy::generateTrajectories |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| depolarization214num_parametersE) |     function)](api/languag        |
| -   [cu                           | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| daq::depolarization2::num_targets | sbe26ExhaustiveSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/la               | detail10NoisePointEEENSt6size_tE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::ptsbe:                |
| q15depolarization211num_targetsE) | :ExhaustiveSamplingStrategy::name |
| -                                 |     (C++                          |
|    [cudaq::depolarization_channel |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq5ptsbe26E |
|     class)](                      | xhaustiveSamplingStrategy4nameEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cuda                         |
| N5cudaq22depolarization_channelE) | q::ptsbe::ExhaustiveSamplingStrat |
| -   [cudaq::depol                 | egy::\~ExhaustiveSamplingStrategy |
| arization_channel::num_parameters |     (C++                          |
|     (C++                          |     function)](api/languages      |
|     member)](api/languages/cp     | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| p_api.html#_CPPv4N5cudaq22depolar | 26ExhaustiveSamplingStrategyD0Ev) |
| ization_channel14num_parametersE) | -   [cuda                         |
| -   [cudaq::de                    | q::ptsbe::OrderedSamplingStrategy |
| polarization_channel::num_targets |     (C++                          |
|     (C++                          |     class)](api/lan               |
|     member)](api/languages        | guages/cpp_api.html#_CPPv4N5cudaq |
| /cpp_api.html#_CPPv4N5cudaq22depo | 5ptsbe23OrderedSamplingStrategyE) |
| larization_channel11num_targetsE) | -   [cudaq::ptsb                  |
| -   [cudaq::details (C++          | e::OrderedSamplingStrategy::clone |
|     type)](api/languages/cpp      |     (C++                          |
| _api.html#_CPPv4N5cudaq7detailsE) |     function)](api/languages/c    |
| -   [cudaq::details::future (C++  | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
|                                   | 3OrderedSamplingStrategy5cloneEv) |
|  class)](api/languages/cpp_api.ht | -   [cudaq::ptsbe::OrderedSampl   |
| ml#_CPPv4N5cudaq7details6futureE) | ingStrategy::generateTrajectories |
| -                                 |     (C++                          |
|   [cudaq::details::future::future |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4NK5cudaq |
|     functio                       | 5ptsbe23OrderedSamplingStrategy20 |
| n)](api/languages/cpp_api.html#_C | generateTrajectoriesENSt4spanIKN6 |
| PPv4N5cudaq7details6future6future | detail10NoisePointEEENSt6size_tE) |
| ERNSt6vectorI3JobEERNSt6stringERN | -   [cudaq::pts                   |
| St3mapINSt6stringENSt6stringEEE), | be::OrderedSamplingStrategy::name |
|     [\[1\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     function)](api/languages/     |
| details6future6futureERR6future), | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
|     [\[2\]]                       | 23OrderedSamplingStrategy4nameEv) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq7details6future6futureEv) |    [cudaq::ptsbe::OrderedSampling |
| -   [cu                           | Strategy::OrderedSamplingStrategy |
| daq::details::kernel_builder_base |     (C++                          |
|     (C++                          |     function)](                   |
|     class)](api/l                 | api/languages/cpp_api.html#_CPPv4 |
| anguages/cpp_api.html#_CPPv4N5cud | N5cudaq5ptsbe23OrderedSamplingStr |
| aq7details19kernel_builder_baseE) | ategy23OrderedSamplingStrategyEv) |
| -   [cudaq::details::             | -                                 |
| kernel_builder_base::operator\<\< |  [cudaq::ptsbe::OrderedSamplingSt |
|     (C++                          | rategy::\~OrderedSamplingStrategy |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq7de |     function)](api/langua         |
| tails19kernel_builder_baselsERNSt | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| 7ostreamERK19kernel_builder_base) | sbe23OrderedSamplingStrategyD0Ev) |
| -   [                             | -   [cudaq::pts                   |
| cudaq::details::KernelBuilderType | be::ProbabilisticSamplingStrategy |
|     (C++                          |     (C++                          |
|     class)](api                   |     class)](api/languages         |
| /languages/cpp_api.html#_CPPv4N5c | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| udaq7details17KernelBuilderTypeE) | 29ProbabilisticSamplingStrategyE) |
| -   [cudaq::d                     | -   [cudaq::ptsbe::Pro            |
| etails::KernelBuilderType::create | babilisticSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP |  function)](api/languages/cpp_api |
| v4N5cudaq7details17KernelBuilderT | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| ype6createEPN4mlir11MLIRContextE) | bilisticSamplingStrategy5cloneEv) |
| -   [cudaq::details::Ker          | -                                 |
| nelBuilderType::KernelBuilderType | [cudaq::ptsbe::ProbabilisticSampl |
|     (C++                          | ingStrategy::generateTrajectories |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     function)](api/languages/     |
| details17KernelBuilderType17Kerne | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| lBuilderTypeERRNSt8functionIFN4ml | 29ProbabilisticSamplingStrategy20 |
| ir4TypeEPN4mlir11MLIRContextEEEE) | generateTrajectoriesENSt4spanIKN6 |
| -   [cudaq::diag_matrix_callback  | detail10NoisePointEEENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe::Pr             |
|     class)                        | obabilisticSamplingStrategy::name |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq20diag_matrix_callbackE) |                                   |
| -   [cudaq::dyn (C++              |   function)](api/languages/cpp_ap |
|     member)](api/languages        | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | abilisticSamplingStrategy4nameEv) |
| -   [cudaq::ExecutionContext (C++ | -   [cudaq::p                     |
|     cl                            | tsbe::ProbabilisticSamplingStrate |
| ass)](api/languages/cpp_api.html# | gy::ProbabilisticSamplingStrategy |
| _CPPv4N5cudaq16ExecutionContextE) |     (C++                          |
| -   [c                            |     function)]                    |
| udaq::ExecutionContext::asyncExec | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4N5cudaq5ptsbe29ProbabilisticSamp |
|     member)](api/                 | lingStrategy29ProbabilisticSampli |
| languages/cpp_api.html#_CPPv4N5cu | ngStrategyENSt8optionalINSt8uint6 |
| daq16ExecutionContext9asyncExecE) | 4_tEEENSt8optionalINSt6size_tEEE) |
| -   [cud                          | -   [cudaq::pts                   |
| aq::ExecutionContext::asyncResult | be::ProbabilisticSamplingStrategy |
|     (C++                          | ::\~ProbabilisticSamplingStrategy |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languages/cp   |
| 16ExecutionContext11asyncResultE) | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| -   [cudaq:                       | robabilisticSamplingStrategyD0Ev) |
| :ExecutionContext::batchIteration | -                                 |
|     (C++                          | [cudaq::ptsbe::PTSBEExecutionData |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     struct)](ap                   |
| xecutionContext14batchIterationE) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [cudaq::E                     | cudaq5ptsbe18PTSBEExecutionDataE) |
| xecutionContext::canHandleObserve | -   [cudaq::ptsbe::PTSBE          |
|     (C++                          | ExecutionData::count_instructions |
|     member)](api/language         |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     function)](api/l              |
| cutionContext16canHandleObserveE) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cudaq::E                     | daq5ptsbe18PTSBEExecutionData18co |
| xecutionContext::ExecutionContext | unt_instructionsE20TraceInstructi |
|     (C++                          | onTypeNSt8optionalINSt6stringEEE) |
|     func                          | -   [cudaq::ptsbe::P              |
| tion)](api/languages/cpp_api.html | TSBEExecutionData::get_trajectory |
| #_CPPv4N5cudaq16ExecutionContext1 |     (C++                          |
| 6ExecutionContextERKNSt6stringE), |     function                      |
|     [\[1\]](api/languages/        | )](api/languages/cpp_api.html#_CP |
| cpp_api.html#_CPPv4N5cudaq16Execu | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| tionContext16ExecutionContextERKN | Data14get_trajectoryENSt6size_tE) |
| St6stringENSt6size_tENSt6size_tE) | -   [cudaq::ptsbe:                |
| -   [cudaq::E                     | :PTSBEExecutionData::instructions |
| xecutionContext::expectationValue |     (C++                          |
|     (C++                          |     member)](api/languages/cp     |
|     member)](api/language         | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | TSBEExecutionData12instructionsE) |
| cutionContext16expectationValueE) | -   [cudaq::ptsbe:                |
| -   [cudaq::Execu                 | :PTSBEExecutionData::trajectories |
| tionContext::explicitMeasurements |     (C++                          |
|     (C++                          |     member)](api/languages/cp     |
|     member)](api/languages/cp     | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| p_api.html#_CPPv4N5cudaq16Executi | TSBEExecutionData12trajectoriesE) |
| onContext20explicitMeasurementsE) | -   [cudaq::ptsbe::PTSBEOptions   |
| -   [cuda                         |     (C++                          |
| q::ExecutionContext::futureResult |     struc                         |
|     (C++                          | t)](api/languages/cpp_api.html#_C |
|     member)](api/lang             | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::ptsbe::PTSB           |
| 6ExecutionContext12futureResultE) | EOptions::include_sequential_data |
| -   [cudaq::ExecutionContext      |     (C++                          |
| ::hasConditionalsOnMeasureResults |                                   |
|     (C++                          |    member)](api/languages/cpp_api |
|     mem                           | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| ber)](api/languages/cpp_api.html# | ptions23include_sequential_dataE) |
| _CPPv4N5cudaq16ExecutionContext31 | -   [cudaq::ptsb                  |
| hasConditionalsOnMeasureResultsE) | e::PTSBEOptions::max_trajectories |
| -   [cudaq::Executi               |     (C++                          |
| onContext::invocationResultBuffer |     member)](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
|     member)](api/languages/cpp_   | 2PTSBEOptions16max_trajectoriesE) |
| api.html#_CPPv4N5cudaq16Execution | -   [cudaq::ptsbe::PT             |
| Context22invocationResultBufferE) | SBEOptions::return_execution_data |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::kernelName |     member)](api/languages/cpp_a  |
|     (C++                          | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
|     member)](api/la               | EOptions21return_execution_dataE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::pts                   |
| q16ExecutionContext10kernelNameE) | be::PTSBEOptions::shot_allocation |
| -   [cud                          |     (C++                          |
| aq::ExecutionContext::kernelTrace |     member)](api/languages        |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     member)](api/lan              | 12PTSBEOptions15shot_allocationE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cud                          |
| 16ExecutionContext11kernelTraceE) | aq::ptsbe::PTSBEOptions::strategy |
| -   [cudaq:                       |     (C++                          |
| :ExecutionContext::msm_dimensions |     member)](api/l                |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     member)](api/langua           | aq5ptsbe12PTSBEOptions8strategyE) |
| ges/cpp_api.html#_CPPv4N5cudaq16E | -   [cudaq::ptsbe::PTSBETrace     |
| xecutionContext14msm_dimensionsE) |     (C++                          |
| -   [cudaq::                      |     t                             |
| ExecutionContext::msm_prob_err_id | ype)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
|     member)](api/languag          | -   [                             |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | cudaq::ptsbe::PTSSamplingStrategy |
| ecutionContext15msm_prob_err_idE) |     (C++                          |
| -   [cudaq::Ex                    |     class)](api                   |
| ecutionContext::msm_probabilities | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq5ptsbe19PTSSamplingStrategyE) |
|     member)](api/languages        | -   [cudaq::                      |
| /cpp_api.html#_CPPv4N5cudaq16Exec | ptsbe::PTSSamplingStrategy::clone |
| utionContext17msm_probabilitiesE) |     (C++                          |
| -                                 |     function)](api/languag        |
|    [cudaq::ExecutionContext::name | es/cpp_api.html#_CPPv4NK5cudaq5pt |
|     (C++                          | sbe19PTSSamplingStrategy5cloneEv) |
|     member)]                      | -   [cudaq::ptsbe::PTSSampl       |
| (api/languages/cpp_api.html#_CPPv | ingStrategy::generateTrajectories |
| 4N5cudaq16ExecutionContext4nameE) |     (C++                          |
| -   [cu                           |     function)](api/               |
| daq::ExecutionContext::noiseModel | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq5ptsbe19PTSSamplingStrategy20 |
|     member)](api/la               | generateTrajectoriesENSt4spanIKN6 |
| nguages/cpp_api.html#_CPPv4N5cuda | detail10NoisePointEEENSt6size_tE) |
| q16ExecutionContext10noiseModelE) | -   [cudaq:                       |
| -   [cudaq::Exe                   | :ptsbe::PTSSamplingStrategy::name |
| cutionContext::numberTrajectories |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     member)](api/languages/       | ges/cpp_api.html#_CPPv4NK5cudaq5p |
| cpp_api.html#_CPPv4N5cudaq16Execu | tsbe19PTSSamplingStrategy4nameEv) |
| tionContext18numberTrajectoriesE) | -   [cudaq::ptsbe::PTSSampli      |
| -   [c                            | ngStrategy::\~PTSSamplingStrategy |
| udaq::ExecutionContext::optResult |     (C++                          |
|     (C++                          |     function)](api/la             |
|     member)](api/                 | nguages/cpp_api.html#_CPPv4N5cuda |
| languages/cpp_api.html#_CPPv4N5cu | q5ptsbe19PTSSamplingStrategyD0Ev) |
| daq16ExecutionContext9optResultE) | -   [cudaq::ptsbe::sample (C++    |
| -                                 |                                   |
|   [cudaq::ExecutionContext::qpuId |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
|     member)](                     | mpleE13sample_resultRK14sample_op |
| api/languages/cpp_api.html#_CPPv4 | tionsRR13QuantumKernelDpRR4Args), |
| N5cudaq16ExecutionContext5qpuIdE) |     [\[1\]](api                   |
| -   [cudaq                        | /languages/cpp_api.html#_CPPv4I0D |
| ::ExecutionContext::registerNames | pEN5cudaq5ptsbe6sampleE13sample_r |
|     (C++                          | esultRKN5cudaq11noise_modelENSt6s |
|     member)](api/langu            | ize_tERR13QuantumKernelDpRR4Args) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [cudaq::ptsbe::sample_async   |
| ExecutionContext13registerNamesE) |     (C++                          |
| -   [cu                           |     function)](a                  |
| daq::ExecutionContext::reorderIdx | pi/languages/cpp_api.html#_CPPv4I |
|     (C++                          | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
|     member)](api/la               | 9async_sample_resultRK14sample_op |
| nguages/cpp_api.html#_CPPv4N5cuda | tionsRR13QuantumKernelDpRR4Args), |
| q16ExecutionContext10reorderIdxE) |     [\[1\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4I0DpEN5cudaq5pts |
|  [cudaq::ExecutionContext::result | be12sample_asyncE19async_sample_r |
|     (C++                          | esultRKN5cudaq11noise_modelENSt6s |
|     member)](a                    | ize_tERR13QuantumKernelDpRR4Args) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::ptsbe::sample_options |
| 5cudaq16ExecutionContext6resultE) |     (C++                          |
| -                                 |     struct)                       |
|   [cudaq::ExecutionContext::shots | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq5ptsbe14sample_optionsE) |
|     member)](                     | -   [cudaq::ptsbe::sample_result  |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5shotsE) |     class                         |
| -   [cudaq::                      | )](api/languages/cpp_api.html#_CP |
| ExecutionContext::simulationState | Pv4N5cudaq5ptsbe13sample_resultE) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/languag          | be::sample_result::execution_data |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15simulationStateE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
|    [cudaq::ExecutionContext::spin | 3sample_result14execution_dataEv) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     member)]                      | sample_result::has_execution_data |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq16ExecutionContext4spinE) |                                   |
| -   [cudaq::                      |    function)](api/languages/cpp_a |
| ExecutionContext::totalIterations | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
|     (C++                          | ple_result18has_execution_dataEv) |
|     member)](api/languag          | -   [cudaq::pt                    |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | sbe::sample_result::sample_result |
| ecutionContext15totalIterationsE) |     (C++                          |
| -   [cudaq::Executio              |     function)](api/l              |
| nContext::warnedNamedMeasurements | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq5ptsbe13sample_result13sample_r |
|     member)](api/languages/cpp_a  | esultERRN5cudaq13sample_resultE), |
| pi.html#_CPPv4N5cudaq16ExecutionC |                                   |
| ontext23warnedNamedMeasurementsE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::ExecutionResult (C++  | ml#_CPPv4N5cudaq5ptsbe13sample_re |
|     st                            | sult13sample_resultERRN5cudaq13sa |
| ruct)](api/languages/cpp_api.html | mple_resultE18PTSBEExecutionData) |
| #_CPPv4N5cudaq15ExecutionResultE) | -   [cudaq::ptsbe::               |
| -   [cud                          | sample_result::set_execution_data |
| aq::ExecutionResult::appendResult |     (C++                          |
|     (C++                          |     function)](api/               |
|     functio                       | languages/cpp_api.html#_CPPv4N5cu |
| n)](api/languages/cpp_api.html#_C | daq5ptsbe13sample_result18set_exe |
| PPv4N5cudaq15ExecutionResult12app | cution_dataE18PTSBEExecutionData) |
| endResultENSt6stringENSt6size_tE) | -   [cud                          |
| -   [cu                           | aq::ptsbe::ShotAllocationStrategy |
| daq::ExecutionResult::deserialize |     (C++                          |
|     (C++                          |     struct)](using                |
|     function)                     | /examples/ptsbe.html#_CPPv4N5cuda |
| ](api/languages/cpp_api.html#_CPP | q5ptsbe22ShotAllocationStrategyE) |
| v4N5cudaq15ExecutionResult11deser | -   [cudaq::ptsbe::ShotAllocatio  |
| ializeERNSt6vectorINSt6size_tEEE) | nStrategy::ShotAllocationStrategy |
| -   [cudaq:                       |     (C++                          |
| :ExecutionResult::ExecutionResult |     function)                     |
|     (C++                          | ](using/examples/ptsbe.html#_CPPv |
|     functio                       | 4N5cudaq5ptsbe22ShotAllocationStr |
| n)](api/languages/cpp_api.html#_C | ategy22ShotAllocationStrategyE4Ty |
| PPv4N5cudaq15ExecutionResult15Exe | pedNSt8optionalINSt8uint64_tEEE), |
| cutionResultE16CountsDictionary), |     [\[1\                         |
|     [\[1\]](api/lan               | ]](using/examples/ptsbe.html#_CPP |
| guages/cpp_api.html#_CPPv4N5cudaq | v4N5cudaq5ptsbe22ShotAllocationSt |
| 15ExecutionResult15ExecutionResul | rategy22ShotAllocationStrategyEv) |
| tE16CountsDictionaryNSt6stringE), | -   [cudaq::pt                    |
|     [\[2\                         | sbe::ShotAllocationStrategy::Type |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     enum)](using/exam             |
| utionResultE16CountsDictionaryd), | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|                                   | be22ShotAllocationStrategy4TypeE) |
|    [\[3\]](api/languages/cpp_api. | -   [cudaq::ptsbe::ShotAllocatio  |
| html#_CPPv4N5cudaq15ExecutionResu | nStrategy::Type::HIGH_WEIGHT_BIAS |
| lt15ExecutionResultENSt6stringE), |     (C++                          |
|     [\[4\                         |     enumerat                      |
| ]](api/languages/cpp_api.html#_CP | or)](using/examples/ptsbe.html#_C |
| Pv4N5cudaq15ExecutionResult15Exec | PPv4N5cudaq5ptsbe22ShotAllocation |
| utionResultERK15ExecutionResult), | Strategy4Type16HIGH_WEIGHT_BIASE) |
|     [\[5\]](api/language          | -   [cudaq::ptsbe::ShotAllocati   |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | onStrategy::Type::LOW_WEIGHT_BIAS |
| cutionResult15ExecutionResultEd), |     (C++                          |
|     [\[6\]](api/languag           |     enumera                       |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | tor)](using/examples/ptsbe.html#_ |
| ecutionResult15ExecutionResultEv) | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| -   [                             | nStrategy4Type15LOW_WEIGHT_BIASE) |
| cudaq::ExecutionResult::operator= | -   [cudaq::ptsbe::ShotAlloc      |
|     (C++                          | ationStrategy::Type::PROPORTIONAL |
|     function)](api/languages/     |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq15Execu |     enum                          |
| tionResultaSERK15ExecutionResult) | erator)](using/examples/ptsbe.htm |
| -   [c                            | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| udaq::ExecutionResult::operator== | tionStrategy4Type12PROPORTIONALE) |
|     (C++                          | -   [cudaq::ptsbe::Shot           |
|     function)](api/languages/c    | AllocationStrategy::Type::UNIFORM |
| pp_api.html#_CPPv4NK5cudaq15Execu |     (C++                          |
| tionResulteqERK15ExecutionResult) |                                   |
| -   [cud                          |   enumerator)](using/examples/pts |
| aq::ExecutionResult::registerName | be.html#_CPPv4N5cudaq5ptsbe22Shot |
|     (C++                          | AllocationStrategy4Type7UNIFORME) |
|     member)](api/lan              | -                                 |
| guages/cpp_api.html#_CPPv4N5cudaq |   [cudaq::ptsbe::TraceInstruction |
| 15ExecutionResult12registerNameE) |     (C++                          |
| -   [cudaq                        |     struct)](                     |
| ::ExecutionResult::sequentialData | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5ptsbe16TraceInstructionE) |
|     member)](api/langu            | -   [cudaq:                       |
| ages/cpp_api.html#_CPPv4N5cudaq15 | :ptsbe::TraceInstruction::channel |
| ExecutionResult14sequentialDataE) |     (C++                          |
| -   [                             |     member)](api/lang             |
| cudaq::ExecutionResult::serialize | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     (C++                          | ptsbe16TraceInstruction7channelE) |
|     function)](api/l              | -   [cudaq::                      |
| anguages/cpp_api.html#_CPPv4NK5cu | ptsbe::TraceInstruction::controls |
| daq15ExecutionResult9serializeEv) |     (C++                          |
| -   [cudaq::fermion_handler (C++  |     member)](api/langu            |
|     c                             | ages/cpp_api.html#_CPPv4N5cudaq5p |
| lass)](api/languages/cpp_api.html | tsbe16TraceInstruction8controlsE) |
| #_CPPv4N5cudaq15fermion_handlerE) | -   [cud                          |
| -   [cudaq::fermion_op (C++       | aq::ptsbe::TraceInstruction::name |
|     type)](api/languages/cpp_api  |     (C++                          |
| .html#_CPPv4N5cudaq10fermion_opE) |     member)](api/l                |
| -   [cudaq::fermion_op_term (C++  | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq5ptsbe16TraceInstruction4nameE) |
| type)](api/languages/cpp_api.html | -   [cudaq                        |
| #_CPPv4N5cudaq15fermion_op_termE) | ::ptsbe::TraceInstruction::params |
| -   [cudaq::FermioniqBaseQPU (C++ |     (C++                          |
|     cl                            |     member)](api/lan              |
| ass)](api/languages/cpp_api.html# | guages/cpp_api.html#_CPPv4N5cudaq |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | 5ptsbe16TraceInstruction6paramsE) |
| -   [cudaq::get_state (C++        | -   [cudaq:                       |
|                                   | :ptsbe::TraceInstruction::targets |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     member)](api/lang             |
| ateEDaRR13QuantumKernelDpRR4Args) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -   [cudaq::gradient (C++         | ptsbe16TraceInstruction7targetsE) |
|     class)](api/languages/cpp_    | -   [cudaq::ptsbe::T              |
| api.html#_CPPv4N5cudaq8gradientE) | raceInstruction::TraceInstruction |
| -   [cudaq::gradient::clone (C++  |     (C++                          |
|     fun                           |                                   |
| ction)](api/languages/cpp_api.htm |   function)](api/languages/cpp_ap |
| l#_CPPv4N5cudaq8gradient5cloneEv) | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| -   [cudaq::gradient::compute     | Instruction16TraceInstructionE20T |
|     (C++                          | raceInstructionTypeNSt6stringENSt |
|     function)](api/language       | 6vectorINSt6size_tEEENSt6vectorIN |
| s/cpp_api.html#_CPPv4N5cudaq8grad | St6size_tEEENSt6vectorIdEENSt8opt |
| ient7computeERKNSt6vectorIdEERKNS | ionalIN5cudaq13kraus_channelEEE), |
| t8functionIFdNSt6vectorIdEEEEEd), |     [\[1\]](api/languages/cpp_a   |
|     [\[1\]](ap                    | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| i/languages/cpp_api.html#_CPPv4N5 | eInstruction16TraceInstructionEv) |
| cudaq8gradient7computeERKNSt6vect | -   [cud                          |
| orIdEERNSt6vectorIdEERK7spin_opd) | aq::ptsbe::TraceInstruction::type |
| -   [cudaq::gradient::gradient    |     (C++                          |
|     (C++                          |     member)](api/l                |
|     function)](api/lang           | anguages/cpp_api.html#_CPPv4N5cud |
| uages/cpp_api.html#_CPPv4I00EN5cu | aq5ptsbe16TraceInstruction4typeE) |
| daq8gradient8gradientER7KernelT), | -   [c                            |
|                                   | udaq::ptsbe::TraceInstructionType |
|    [\[1\]](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4I00EN5cudaq8gradient8g |     enum)](api/                   |
| radientER7KernelTRR10ArgsMapper), | languages/cpp_api.html#_CPPv4N5cu |
|     [\[2\                         | daq5ptsbe20TraceInstructionTypeE) |
| ]](api/languages/cpp_api.html#_CP | -   [cudaq::                      |
| Pv4I00EN5cudaq8gradient8gradientE | ptsbe::TraceInstructionType::Gate |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3                          |     enumerator)](api/langu        |
| \]](api/languages/cpp_api.html#_C | ages/cpp_api.html#_CPPv4N5cudaq5p |
| PPv4N5cudaq8gradient8gradientERRN | tsbe20TraceInstructionType4GateE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::ptsbe::               |
|     [\[                           | TraceInstructionType::Measurement |
| 4\]](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8gradient8gradientEv) |                                   |
| -   [cudaq::gradient::setArgs     |    enumerator)](api/languages/cpp |
|     (C++                          | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
|     fu                            | aceInstructionType11MeasurementE) |
| nction)](api/languages/cpp_api.ht | -   [cudaq::p                     |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | tsbe::TraceInstructionType::Noise |
| tArgsEvR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::gradient::setKernel   |     enumerator)](api/langua       |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     function)](api/languages/c    | sbe20TraceInstructionType5NoiseE) |
| pp_api.html#_CPPv4I0EN5cudaq8grad | -   [                             |
| ient9setKernelEvR13QuantumKernel) | cudaq::ptsbe::TrajectoryPredicate |
| -   [cud                          |     (C++                          |
| aq::gradients::central_difference |     type)](api                    |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     class)](api/la                | udaq5ptsbe19TrajectoryPredicateE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QPU (C++              |
| q9gradients18central_differenceE) |     class)](api/languages         |
| -   [cudaq::gra                   | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| dients::central_difference::clone | -   [cudaq::QPU::beginExecution   |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function                      |
| /cpp_api.html#_CPPv4N5cudaq9gradi | )](api/languages/cpp_api.html#_CP |
| ents18central_difference5cloneEv) | Pv4N5cudaq3QPU14beginExecutionEv) |
| -   [cudaq::gradi                 | -   [cuda                         |
| ents::central_difference::compute | q::QPU::configureExecutionContext |
|     (C++                          |     (C++                          |
|     function)](                   |     funct                         |
| api/languages/cpp_api.html#_CPPv4 | ion)](api/languages/cpp_api.html# |
| N5cudaq9gradients18central_differ | _CPPv4NK5cudaq3QPU25configureExec |
| ence7computeERKNSt6vectorIdEERKNS | utionContextER16ExecutionContext) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::QPU::endExecution     |
|                                   |     (C++                          |
|   [\[1\]](api/languages/cpp_api.h |     functi                        |
| tml#_CPPv4N5cudaq9gradients18cent | on)](api/languages/cpp_api.html#_ |
| ral_difference7computeERKNSt6vect | CPPv4N5cudaq3QPU12endExecutionEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QPU::enqueue (C++     |
| -   [cudaq::gradie                |     function)](ap                 |
| nts::central_difference::gradient | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq3QPU7enqueueER11QuantumTask) |
|     functio                       | -   [cud                          |
| n)](api/languages/cpp_api.html#_C | aq::QPU::finalizeExecutionContext |
| PPv4I00EN5cudaq9gradients18centra |     (C++                          |
| l_difference8gradientER7KernelT), |     func                          |
|     [\[1\]](api/langua            | tion)](api/languages/cpp_api.html |
| ges/cpp_api.html#_CPPv4I00EN5cuda | #_CPPv4NK5cudaq3QPU24finalizeExec |
| q9gradients18central_difference8g | utionContextER16ExecutionContext) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::QPU::getConnectivity  |
|     [\[2\]](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4I00EN5cudaq9gradie |     function)                     |
| nts18central_difference8gradientE | ](api/languages/cpp_api.html#_CPP |
| RR13QuantumKernelRR10ArgsMapper), | v4N5cudaq3QPU15getConnectivityEv) |
|     [\[3\]](api/languages/cpp     | -                                 |
| _api.html#_CPPv4N5cudaq9gradients | [cudaq::QPU::getExecutionThreadId |
| 18central_difference8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](api/               |
|     [\[4\]](api/languages/cp      | languages/cpp_api.html#_CPPv4NK5c |
| p_api.html#_CPPv4N5cudaq9gradient | udaq3QPU20getExecutionThreadIdEv) |
| s18central_difference8gradientEv) | -   [cudaq::QPU::getNumQubits     |
| -   [cud                          |     (C++                          |
| aq::gradients::forward_difference |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     class)](api/la                | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [                             |
| q9gradients18forward_differenceE) | cudaq::QPU::getRemoteCapabilities |
| -   [cudaq::gra                   |     (C++                          |
| dients::forward_difference::clone |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     function)](api/languages      | daq3QPU21getRemoteCapabilitiesEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QPU::isEmulated (C++  |
| ents18forward_difference5cloneEv) |     func                          |
| -   [cudaq::gradi                 | tion)](api/languages/cpp_api.html |
| ents::forward_difference::compute | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     (C++                          | -   [cudaq::QPU::isSimulator (C++ |
|     function)](                   |     funct                         |
| api/languages/cpp_api.html#_CPPv4 | ion)](api/languages/cpp_api.html# |
| N5cudaq9gradients18forward_differ | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| ence7computeERKNSt6vectorIdEERKNS | -   [cudaq::QPU::launchKernel     |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|                                   |     function)](api                |
|   [\[1\]](api/languages/cpp_api.h | /languages/cpp_api.html#_CPPv4N5c |
| tml#_CPPv4N5cudaq9gradients18forw | udaq3QPU12launchKernelERKNSt6stri |
| ard_difference7computeERKNSt6vect | ngE15KernelThunkTypePvNSt8uint64_ |
| orIdEERNSt6vectorIdEERK7spin_opd) | tENSt8uint64_tERKNSt6vectorIPvEE) |
| -   [cudaq::gradie                | -   [cudaq::QPU::onRandomSeedSet  |
| nts::forward_difference::gradient |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     functio                       | uages/cpp_api.html#_CPPv4N5cudaq3 |
| n)](api/languages/cpp_api.html#_C | QPU15onRandomSeedSetENSt6size_tE) |
| PPv4I00EN5cudaq9gradients18forwar | -   [cudaq::QPU::QPU (C++         |
| d_difference8gradientER7KernelT), |     functio                       |
|     [\[1\]](api/langua            | n)](api/languages/cpp_api.html#_C |
| ges/cpp_api.html#_CPPv4I00EN5cuda | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| q9gradients18forward_difference8g |                                   |
| radientER7KernelTRR10ArgsMapper), |  [\[1\]](api/languages/cpp_api.ht |
|     [\[2\]](api/languages/cpp_    | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| api.html#_CPPv4I00EN5cudaq9gradie |     [\[2\]](api/languages/cpp_    |
| nts18forward_difference8gradientE | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::QPU::setId (C++       |
|     [\[3\]](api/languages/cpp     |     function                      |
| _api.html#_CPPv4N5cudaq9gradients | )](api/languages/cpp_api.html#_CP |
| 18forward_difference8gradientERRN | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QPU::setShots (C++    |
|     [\[4\]](api/languages/cp      |     f                             |
| p_api.html#_CPPv4N5cudaq9gradient | unction)](api/languages/cpp_api.h |
| s18forward_difference8gradientEv) | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| -   [                             | -   [cudaq::                      |
| cudaq::gradients::parameter_shift | QPU::supportsExplicitMeasurements |
|     (C++                          |     (C++                          |
|     class)](api                   |     function)](api/languag        |
| /languages/cpp_api.html#_CPPv4N5c | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| udaq9gradients15parameter_shiftE) | 28supportsExplicitMeasurementsEv) |
| -   [cudaq::                      | -   [cudaq::QPU::\~QPU (C++       |
| gradients::parameter_shift::clone |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     function)](api/langua         | -   [cudaq::QPUState (C++         |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     class)](api/languages/cpp_    |
| adients15parameter_shift5cloneEv) | api.html#_CPPv4N5cudaq8QPUStateE) |
| -   [cudaq::gr                    | -   [cudaq::qreg (C++             |
| adients::parameter_shift::compute |     class)](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4I_NSt6s |
|     function                      | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qreg::back (C++       |
| Pv4N5cudaq9gradients15parameter_s |     function)                     |
| hift7computeERKNSt6vectorIdEERKNS | ](api/languages/cpp_api.html#_CPP |
| t8functionIFdNSt6vectorIdEEEEEd), | v4N5cudaq4qreg4backENSt6size_tE), |
|     [\[1\]](api/languages/cpp_ap  |     [\[1\]](api/languages/cpp_ap  |
| i.html#_CPPv4N5cudaq9gradients15p | i.html#_CPPv4N5cudaq4qreg4backEv) |
| arameter_shift7computeERKNSt6vect | -   [cudaq::qreg::begin (C++      |
| orIdEERNSt6vectorIdEERK7spin_opd) |                                   |
| -   [cudaq::gra                   |  function)](api/languages/cpp_api |
| dients::parameter_shift::gradient | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     (C++                          | -   [cudaq::qreg::clear (C++      |
|     func                          |                                   |
| tion)](api/languages/cpp_api.html |  function)](api/languages/cpp_api |
| #_CPPv4I00EN5cudaq9gradients15par | .html#_CPPv4N5cudaq4qreg5clearEv) |
| ameter_shift8gradientER7KernelT), | -   [cudaq::qreg::front (C++      |
|     [\[1\]](api/lan               |     function)]                    |
| guages/cpp_api.html#_CPPv4I00EN5c | (api/languages/cpp_api.html#_CPPv |
| udaq9gradients15parameter_shift8g | 4N5cudaq4qreg5frontENSt6size_tE), |
| radientER7KernelTRR10ArgsMapper), |     [\[1\]](api/languages/cpp_api |
|     [\[2\]](api/languages/c       | .html#_CPPv4N5cudaq4qreg5frontEv) |
| pp_api.html#_CPPv4I00EN5cudaq9gra | -   [cudaq::qreg::operator\[\]    |
| dients15parameter_shift8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     functi                        |
|     [\[3\]](api/languages/        | on)](api/languages/cpp_api.html#_ |
| cpp_api.html#_CPPv4N5cudaq9gradie | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| nts15parameter_shift8gradientERRN | -   [cudaq::qreg::qreg (C++       |
| St8functionIFvNSt6vectorIdEEEEE), |     function)                     |
|     [\[4\]](api/languages         | ](api/languages/cpp_api.html#_CPP |
| /cpp_api.html#_CPPv4N5cudaq9gradi | v4N5cudaq4qreg4qregENSt6size_tE), |
| ents15parameter_shift8gradientEv) |     [\[1\]](api/languages/cpp_ap  |
| -   [cudaq::kernel_builder (C++   | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     clas                          | -   [cudaq::qreg::size (C++       |
| s)](api/languages/cpp_api.html#_C |                                   |
| PPv4IDpEN5cudaq14kernel_builderE) |  function)](api/languages/cpp_api |
| -   [c                            | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| udaq::kernel_builder::constantVal | -   [cudaq::qreg::slice (C++      |
|     (C++                          |     function)](api/langu          |
|     function)](api/la             | ages/cpp_api.html#_CPPv4N5cudaq4q |
| nguages/cpp_api.html#_CPPv4N5cuda | reg5sliceENSt6size_tENSt6size_tE) |
| q14kernel_builder11constantValEd) | -   [cudaq::qreg::value_type (C++ |
| -   [cu                           |                                   |
| daq::kernel_builder::getArguments | type)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq4qreg10value_typeE) |
|     function)](api/lan            | -   [cudaq::qspan (C++            |
| guages/cpp_api.html#_CPPv4N5cudaq |     class)](api/lang              |
| 14kernel_builder12getArgumentsEv) | uages/cpp_api.html#_CPPv4I_NSt6si |
| -   [cu                           | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| daq::kernel_builder::getNumParams | -   [cudaq::QuakeValue (C++       |
|     (C++                          |     class)](api/languages/cpp_api |
|     function)](api/lan            | .html#_CPPv4N5cudaq10QuakeValueE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::Q                     |
| 14kernel_builder12getNumParamsEv) | uakeValue::canValidateNumElements |
| -   [c                            |     (C++                          |
| udaq::kernel_builder::isArgStdVec |     function)](api/languages      |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq10Quak |
|     function)](api/languages/cp   | eValue22canValidateNumElementsEv) |
| p_api.html#_CPPv4N5cudaq14kernel_ | -                                 |
| builder11isArgStdVecENSt6size_tE) |  [cudaq::QuakeValue::constantSize |
| -   [cuda                         |     (C++                          |
| q::kernel_builder::kernel_builder |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/languages/cpp_ | udaq10QuakeValue12constantSizeEv) |
| api.html#_CPPv4N5cudaq14kernel_bu | -   [cudaq::QuakeValue::dump (C++ |
| ilder14kernel_builderERNSt6vector |     function)](api/lan            |
| IN7details17KernelBuilderTypeEEE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [cudaq::kernel_builder::name  | 10QuakeValue4dumpERNSt7ostreamE), |
|     (C++                          |     [\                            |
|     function)                     | [1\]](api/languages/cpp_api.html# |
| ](api/languages/cpp_api.html#_CPP | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| v4N5cudaq14kernel_builder4nameEv) | -   [cudaq                        |
| -                                 | ::QuakeValue::getRequiredElements |
|    [cudaq::kernel_builder::qalloc |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/language       | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| s/cpp_api.html#_CPPv4N5cudaq14ker | uakeValue19getRequiredElementsEv) |
| nel_builder6qallocE10QuakeValue), | -   [cudaq::QuakeValue::getValue  |
|     [\[1\]](api/language          |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     function)]                    |
| nel_builder6qallocEKNSt6size_tE), | (api/languages/cpp_api.html#_CPPv |
|     [\[2                          | 4NK5cudaq10QuakeValue8getValueEv) |
| \]](api/languages/cpp_api.html#_C | -   [cudaq::QuakeValue::inverse   |
| PPv4N5cudaq14kernel_builder6qallo |     (C++                          |
| cERNSt6vectorINSt7complexIdEEEE), |     function)                     |
|     [\[3\]](                      | ](api/languages/cpp_api.html#_CPP |
| api/languages/cpp_api.html#_CPPv4 | v4NK5cudaq10QuakeValue7inverseEv) |
| N5cudaq14kernel_builder6qallocEv) | -   [cudaq::QuakeValue::isStdVec  |
| -   [cudaq::kernel_builder::swap  |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/language       | ](api/languages/cpp_api.html#_CPP |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | v4N5cudaq10QuakeValue8isStdVecEv) |
| 4kernel_builder4swapEvRK10QuakeVa | -                                 |
| lueRK10QuakeValueRK10QuakeValue), |    [cudaq::QuakeValue::operator\* |
|                                   |     (C++                          |
| [\[1\]](api/languages/cpp_api.htm |     function)](api                |
| l#_CPPv4I00EN5cudaq14kernel_build | /languages/cpp_api.html#_CPPv4N5c |
| er4swapEvRKNSt6vectorI10QuakeValu | udaq10QuakeValuemlE10QuakeValue), |
| eEERK10QuakeValueRK10QuakeValue), |                                   |
|                                   | [\[1\]](api/languages/cpp_api.htm |
| [\[2\]](api/languages/cpp_api.htm | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| l#_CPPv4N5cudaq14kernel_builder4s | -   [cudaq::QuakeValue::operator+ |
| wapERK10QuakeValueRK10QuakeValue) |     (C++                          |
| -   [cudaq::KernelExecutionTask   |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     type                          | udaq10QuakeValueplE10QuakeValue), |
| )](api/languages/cpp_api.html#_CP |     [                             |
| Pv4N5cudaq19KernelExecutionTaskE) | \[1\]](api/languages/cpp_api.html |
| -   [cudaq::KernelThunkResultType | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     (C++                          |                                   |
|     struct)]                      | [\[2\]](api/languages/cpp_api.htm |
| (api/languages/cpp_api.html#_CPPv | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| 4N5cudaq21KernelThunkResultTypeE) | -   [cudaq::QuakeValue::operator- |
| -   [cudaq::KernelThunkType (C++  |     (C++                          |
|                                   |     function)](api                |
| type)](api/languages/cpp_api.html | /languages/cpp_api.html#_CPPv4N5c |
| #_CPPv4N5cudaq15KernelThunkTypeE) | udaq10QuakeValuemiE10QuakeValue), |
| -   [cudaq::kraus_channel (C++    |     [                             |
|                                   | \[1\]](api/languages/cpp_api.html |
|  class)](api/languages/cpp_api.ht | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| ml#_CPPv4N5cudaq13kraus_channelE) |     [                             |
| -   [cudaq::kraus_channel::empty  | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     function)]                    |                                   |
| (api/languages/cpp_api.html#_CPPv | [\[3\]](api/languages/cpp_api.htm |
| 4NK5cudaq13kraus_channel5emptyEv) | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| -   [cudaq::kraus_c               | -   [cudaq::QuakeValue::operator/ |
| hannel::generateUnitaryParameters |     (C++                          |
|     (C++                          |     function)](api                |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
|    function)](api/languages/cpp_a | udaq10QuakeValuedvE10QuakeValue), |
| pi.html#_CPPv4N5cudaq13kraus_chan |                                   |
| nel25generateUnitaryParametersEv) | [\[1\]](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
|    [cudaq::kraus_channel::get_ops | -                                 |
|     (C++                          |  [cudaq::QuakeValue::operator\[\] |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api                |
| K5cudaq13kraus_channel7get_opsEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cud                          | udaq10QuakeValueixEKNSt6size_tE), |
| aq::kraus_channel::identity_flags |     [\[1\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     member)](api/lan              | daq10QuakeValueixERK10QuakeValue) |
| guages/cpp_api.html#_CPPv4N5cudaq | -                                 |
| 13kraus_channel14identity_flagsE) |    [cudaq::QuakeValue::QuakeValue |
| -   [cud                          |     (C++                          |
| aq::kraus_channel::is_identity_op |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|                                   | akeValue10QuakeValueERN4mlir20Imp |
|    function)](api/languages/cpp_a | licitLocOpBuilderEN4mlir5ValueE), |
| pi.html#_CPPv4NK5cudaq13kraus_cha |     [\[1\]                        |
| nnel14is_identity_opENSt6size_tE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::                      | v4N5cudaq10QuakeValue10QuakeValue |
| kraus_channel::is_unitary_mixture | ERN4mlir20ImplicitLocOpBuilderEd) |
|     (C++                          | -   [cudaq::QuakeValue::size (C++ |
|     function)](api/languages      |     funct                         |
| /cpp_api.html#_CPPv4NK5cudaq13kra | ion)](api/languages/cpp_api.html# |
| us_channel18is_unitary_mixtureEv) | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| -   [cu                           | -   [cudaq::QuakeValue::slice     |
| daq::kraus_channel::kraus_channel |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     function)](api/lang           | api.html#_CPPv4N5cudaq10QuakeValu |
| uages/cpp_api.html#_CPPv4IDpEN5cu | e5sliceEKNSt6size_tEKNSt6size_tE) |
| daq13kraus_channel13kraus_channel | -   [cudaq::quantum_platform (C++ |
| EDpRRNSt16initializer_listI1TEE), |     cl                            |
|                                   | ass)](api/languages/cpp_api.html# |
|  [\[1\]](api/languages/cpp_api.ht | _CPPv4N5cudaq16quantum_platformE) |
| ml#_CPPv4N5cudaq13kraus_channel13 | -   [cudaq:                       |
| kraus_channelERK13kraus_channel), | :quantum_platform::beginExecution |
|     [\[2\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languag        |
| v4N5cudaq13kraus_channel13kraus_c | es/cpp_api.html#_CPPv4N5cudaq16qu |
| hannelERKNSt6vectorI8kraus_opEE), | antum_platform14beginExecutionEv) |
|     [\[3\]                        | -   [cudaq::quantum_pl            |
| ](api/languages/cpp_api.html#_CPP | atform::configureExecutionContext |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERRNSt6vectorI8kraus_opEE), |     function)](api/lang           |
|     [\[4\]](api/lan               | uages/cpp_api.html#_CPPv4NK5cudaq |
| guages/cpp_api.html#_CPPv4N5cudaq | 16quantum_platform25configureExec |
| 13kraus_channel13kraus_channelEv) | utionContextER16ExecutionContext) |
| -                                 | -   [cuda                         |
| [cudaq::kraus_channel::noise_type | q::quantum_platform::connectivity |
|     (C++                          |     (C++                          |
|     member)](api                  |     function)](api/langu          |
| /languages/cpp_api.html#_CPPv4N5c | ages/cpp_api.html#_CPPv4N5cudaq16 |
| udaq13kraus_channel10noise_typeE) | quantum_platform12connectivityEv) |
| -                                 | -   [cuda                         |
|   [cudaq::kraus_channel::op_names | q::quantum_platform::endExecution |
|     (C++                          |     (C++                          |
|     member)](                     |     function)](api/langu          |
| api/languages/cpp_api.html#_CPPv4 | ages/cpp_api.html#_CPPv4N5cudaq16 |
| N5cudaq13kraus_channel8op_namesE) | quantum_platform12endExecutionEv) |
| -                                 | -   [cudaq::q                     |
|  [cudaq::kraus_channel::operator= | uantum_platform::enqueueAsyncTask |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/languages/     |
| ges/cpp_api.html#_CPPv4N5cudaq13k | cpp_api.html#_CPPv4N5cudaq16quant |
| raus_channelaSERK13kraus_channel) | um_platform16enqueueAsyncTaskEKNS |
| -   [c                            | t6size_tER19KernelExecutionTask), |
| udaq::kraus_channel::operator\[\] |     [\[1\]](api/languag           |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     function)](api/l              | antum_platform16enqueueAsyncTaskE |
| anguages/cpp_api.html#_CPPv4N5cud | KNSt6size_tERNSt8functionIFvvEEE) |
| aq13kraus_channelixEKNSt6size_tE) | -   [cudaq::quantum_p             |
| -                                 | latform::finalizeExecutionContext |
| [cudaq::kraus_channel::parameters |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     member)](api                  | pp_api.html#_CPPv4NK5cudaq16quant |
| /languages/cpp_api.html#_CPPv4N5c | um_platform24finalizeExecutionCon |
| udaq13kraus_channel10parametersE) | textERN5cudaq16ExecutionContextE) |
| -   [cudaq::krau                  | -   [cudaq::qua                   |
| s_channel::populateDefaultOpNames | ntum_platform::get_codegen_config |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/languages/c    |
| p_api.html#_CPPv4N5cudaq13kraus_c | pp_api.html#_CPPv4N5cudaq16quantu |
| hannel22populateDefaultOpNamesEv) | m_platform18get_codegen_configEv) |
| -   [cu                           | -   [cuda                         |
| daq::kraus_channel::probabilities | q::quantum_platform::get_exec_ctx |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api/langua         |
| nguages/cpp_api.html#_CPPv4N5cuda | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| q13kraus_channel13probabilitiesE) | quantum_platform12get_exec_ctxEv) |
| -                                 | -   [c                            |
|  [cudaq::kraus_channel::push_back | udaq::quantum_platform::get_noise |
|     (C++                          |     (C++                          |
|     function)](api                |     function)](api/languages/c    |
| /languages/cpp_api.html#_CPPv4N5c | pp_api.html#_CPPv4N5cudaq16quantu |
| udaq13kraus_channel9push_backE8kr | m_platform9get_noiseENSt6size_tE) |
| aus_opNSt8optionalINSt6stringEEE) | -   [cudaq:                       |
| -   [cudaq::kraus_channel::size   | :quantum_platform::get_num_qubits |
|     (C++                          |     (C++                          |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP | function)](api/languages/cpp_api. |
| v4NK5cudaq13kraus_channel4sizeEv) | html#_CPPv4NK5cudaq16quantum_plat |
| -   [                             | form14get_num_qubitsENSt6size_tE) |
| cudaq::kraus_channel::unitary_ops | -   [cudaq::quantum_              |
|     (C++                          | platform::get_remote_capabilities |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)                     |
| daq13kraus_channel11unitary_opsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus_op (C++         | v4NK5cudaq16quantum_platform23get |
|     struct)](api/languages/cpp_   | _remote_capabilitiesENSt6size_tE) |
| api.html#_CPPv4N5cudaq8kraus_opE) | -   [cudaq::qua                   |
| -   [cudaq::kraus_op::adjoint     | ntum_platform::get_runtime_target |
|     (C++                          |     (C++                          |
|     functi                        |     function)](api/languages/cp   |
| on)](api/languages/cpp_api.html#_ | p_api.html#_CPPv4NK5cudaq16quantu |
| CPPv4NK5cudaq8kraus_op7adjointEv) | m_platform18get_runtime_targetEv) |
| -   [cudaq::kraus_op::data (C++   | -   [cud                          |
|                                   | aq::quantum_platform::is_emulated |
|  member)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |                                   |
| -   [cudaq::kraus_op::kraus_op    |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq16quantum_p |
|     func                          | latform11is_emulatedENSt6size_tE) |
| tion)](api/languages/cpp_api.html | -   [c                            |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | udaq::quantum_platform::is_remote |
| opERRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)](api/languages/cp   |
|  [\[1\]](api/languages/cpp_api.ht | p_api.html#_CPPv4NK5cudaq16quantu |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | m_platform9is_remoteENSt6size_tE) |
| pENSt6vectorIN5cudaq7complexEEE), | -   [cuda                         |
|     [\[2\]](api/l                 | q::quantum_platform::is_simulator |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq8kraus_op8kraus_opERK8kraus_op) |                                   |
| -   [cudaq::kraus_op::nCols (C++  |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq16quantum_pl |
| member)](api/languages/cpp_api.ht | atform12is_simulatorENSt6size_tE) |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | -   [c                            |
| -   [cudaq::kraus_op::nRows (C++  | udaq::quantum_platform::launchVQE |
|                                   |     (C++                          |
| member)](api/languages/cpp_api.ht |     function)](                   |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::kraus_op::operator=   | N5cudaq16quantum_platform9launchV |
|     (C++                          | QEEKNSt6stringEPKvPN5cudaq8gradie |
|     function)                     | ntERKN5cudaq7spin_opERN5cudaq9opt |
| ](api/languages/cpp_api.html#_CPP | imizerEKiKNSt6size_tENSt6size_tE) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cudaq:                       |
| -   [cudaq::kraus_op::precision   | :quantum_platform::list_platforms |
|     (C++                          |     (C++                          |
|     memb                          |     function)](api/languag        |
| er)](api/languages/cpp_api.html#_ | es/cpp_api.html#_CPPv4N5cudaq16qu |
| CPPv4N5cudaq8kraus_op9precisionE) | antum_platform14list_platformsEv) |
| -   [cudaq::KrausSelection (C++   | -                                 |
|     s                             |    [cudaq::quantum_platform::name |
| truct)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14KrausSelectionE) |     function)](a                  |
| -   [cudaq:                       | pi/languages/cpp_api.html#_CPPv4N |
| :KrausSelection::circuit_location | K5cudaq16quantum_platform4nameEv) |
|     (C++                          | -   [                             |
|     member)](api/langua           | cudaq::quantum_platform::num_qpus |
| ges/cpp_api.html#_CPPv4N5cudaq14K |     (C++                          |
| rausSelection16circuit_locationE) |     function)](api/l              |
| -                                 | anguages/cpp_api.html#_CPPv4NK5cu |
|  [cudaq::KrausSelection::is_error | daq16quantum_platform8num_qpusEv) |
|     (C++                          | -   [cudaq::                      |
|     member)](a                    | quantum_platform::onRandomSeedSet |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14KrausSelection8is_errorE) |                                   |
| -   [cudaq::Kra                   | function)](api/languages/cpp_api. |
| usSelection::kraus_operator_index | html#_CPPv4N5cudaq16quantum_platf |
|     (C++                          | orm15onRandomSeedSetENSt6size_tE) |
|     member)](api/languages/       | -   [cudaq:                       |
| cpp_api.html#_CPPv4N5cudaq14Kraus | :quantum_platform::reset_exec_ctx |
| Selection20kraus_operator_indexE) |     (C++                          |
| -   [cuda                         |     function)](api/languag        |
| q::KrausSelection::KrausSelection | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14reset_exec_ctxEv) |
|     function)](a                  | -   [cud                          |
| pi/languages/cpp_api.html#_CPPv4N | aq::quantum_platform::reset_noise |
| 5cudaq14KrausSelection14KrausSele |     (C++                          |
| ctionENSt6size_tENSt6vectorINSt6s |     function)](api/languages/cpp_ |
| ize_tEEENSt6stringENSt6size_tEb), | api.html#_CPPv4N5cudaq16quantum_p |
|     [\[1\]](api/langu             | latform11reset_noiseENSt6size_tE) |
| ages/cpp_api.html#_CPPv4N5cudaq14 | -   [cuda                         |
| KrausSelection14KrausSelectionEv) | q::quantum_platform::set_exec_ctx |
| -                                 |     (C++                          |
|   [cudaq::KrausSelection::op_name |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     member)](                     | _CPPv4N5cudaq16quantum_platform12 |
| api/languages/cpp_api.html#_CPPv4 | set_exec_ctxEP16ExecutionContext) |
| N5cudaq14KrausSelection7op_nameE) | -   [c                            |
| -   [                             | udaq::quantum_platform::set_noise |
| cudaq::KrausSelection::operator== |     (C++                          |
|     (C++                          |     function                      |
|     function)](api/languages      | )](api/languages/cpp_api.html#_CP |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | Pv4N5cudaq16quantum_platform9set_ |
| usSelectioneqERK14KrausSelection) | noiseEPK11noise_modelNSt6size_tE) |
| -                                 | -   [cudaq::quantum_platfor       |
|    [cudaq::KrausSelection::qubits | m::supports_explicit_measurements |
|     (C++                          |     (C++                          |
|     member)]                      |     function)](api/l              |
| (api/languages/cpp_api.html#_CPPv | anguages/cpp_api.html#_CPPv4NK5cu |
| 4N5cudaq14KrausSelection6qubitsE) | daq16quantum_platform30supports_e |
| -   [cudaq::KrausTrajectory (C++  | xplicit_measurementsENSt6size_tE) |
|     st                            | -   [cudaq::quantum_pla           |
| ruct)](api/languages/cpp_api.html | tform::supports_task_distribution |
| #_CPPv4N5cudaq15KrausTrajectoryE) |     (C++                          |
| -                                 |     fu                            |
|  [cudaq::KrausTrajectory::builder | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq16quantum_platfo |
|     function)](ap                 | rm26supports_task_distributionEv) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq::quantum               |
| cudaq15KrausTrajectory7builderEv) | _platform::with_execution_context |
| -   [cu                           |     (C++                          |
| daq::KrausTrajectory::countErrors |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/lang           | v4I0DpEN5cudaq16quantum_platform2 |
| uages/cpp_api.html#_CPPv4NK5cudaq | 2with_execution_contextEDaR16Exec |
| 15KrausTrajectory11countErrorsEv) | utionContextRR8CallableDpRR4Args) |
| -   [                             | -   [cudaq::QuantumTask (C++      |
| cudaq::KrausTrajectory::isOrdered |     type)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq11QuantumTaskE) |
|     function)](api/l              | -   [cudaq::qubit (C++            |
| anguages/cpp_api.html#_CPPv4NK5cu |     type)](api/languages/c        |
| daq15KrausTrajectory9isOrderedEv) | pp_api.html#_CPPv4N5cudaq5qubitE) |
| -   [cudaq::                      | -   [cudaq::QubitConnectivity     |
| KrausTrajectory::kraus_selections |     (C++                          |
|     (C++                          |     ty                            |
|     member)](api/languag          | pe)](api/languages/cpp_api.html#_ |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | CPPv4N5cudaq17QubitConnectivityE) |
| ausTrajectory16kraus_selectionsE) | -   [cudaq::QubitEdge (C++        |
| -   [cudaq:                       |     type)](api/languages/cpp_a    |
| :KrausTrajectory::KrausTrajectory | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|     (C++                          | -   [cudaq::qudit (C++            |
|     function                      |     clas                          |
| )](api/languages/cpp_api.html#_CP | s)](api/languages/cpp_api.html#_C |
| Pv4N5cudaq15KrausTrajectory15Krau | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| sTrajectoryENSt6size_tENSt6vector | -   [cudaq::qudit::qudit (C++     |
| I14KrausSelectionEEdNSt6size_tE), |                                   |
|     [\[1\]](api/languag           | function)](api/languages/cpp_api. |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | html#_CPPv4N5cudaq5qudit5quditEv) |
| ausTrajectory15KrausTrajectoryEv) | -   [cudaq::qvector (C++          |
| -   [cudaq::Kr                    |     class)                        |
| ausTrajectory::measurement_counts | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     member)](api/languages        | -   [cudaq::qvector::back (C++    |
| /cpp_api.html#_CPPv4N5cudaq15Krau |     function)](a                  |
| sTrajectory18measurement_countsE) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cud                          | 5cudaq7qvector4backENSt6size_tE), |
| aq::KrausTrajectory::multiplicity |                                   |
|     (C++                          |   [\[1\]](api/languages/cpp_api.h |
|     member)](api/lan              | tml#_CPPv4N5cudaq7qvector4backEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qvector::begin (C++   |
| 15KrausTrajectory12multiplicityE) |     fu                            |
| -   [                             | nction)](api/languages/cpp_api.ht |
| cudaq::KrausTrajectory::num_shots | ml#_CPPv4N5cudaq7qvector5beginEv) |
|     (C++                          | -   [cudaq::qvector::clear (C++   |
|     member)](api                  |     fu                            |
| /languages/cpp_api.html#_CPPv4N5c | nction)](api/languages/cpp_api.ht |
| udaq15KrausTrajectory9num_shotsE) | ml#_CPPv4N5cudaq7qvector5clearEv) |
| -   [c                            | -   [cudaq::qvector::end (C++     |
| udaq::KrausTrajectory::operator== |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function)](api/languages/c    | html#_CPPv4N5cudaq7qvector3endEv) |
| pp_api.html#_CPPv4NK5cudaq15Kraus | -   [cudaq::qvector::front (C++   |
| TrajectoryeqERK15KrausTrajectory) |     function)](ap                 |
| -   [cu                           | i/languages/cpp_api.html#_CPPv4N5 |
| daq::KrausTrajectory::probability | cudaq7qvector5frontENSt6size_tE), |
|     (C++                          |                                   |
|     member)](api/la               |  [\[1\]](api/languages/cpp_api.ht |
| nguages/cpp_api.html#_CPPv4N5cuda | ml#_CPPv4N5cudaq7qvector5frontEv) |
| q15KrausTrajectory11probabilityE) | -   [cudaq::qvector::operator=    |
| -   [cuda                         |     (C++                          |
| q::KrausTrajectory::trajectory_id |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](api/lang             | PPv4N5cudaq7qvectoraSERK7qvector) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::qvector::operator\[\] |
| 5KrausTrajectory13trajectory_idE) |     (C++                          |
| -                                 |     function)                     |
|   [cudaq::KrausTrajectory::weight | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     member)](                     | -   [cudaq::qvector::qvector (C++ |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/               |
| N5cudaq15KrausTrajectory6weightE) | languages/cpp_api.html#_CPPv4N5cu |
| -                                 | daq7qvector7qvectorENSt6size_tE), |
|    [cudaq::KrausTrajectoryBuilder |     [\[1\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     class)](                      | 5cudaq7qvector7qvectorERK5state), |
| api/languages/cpp_api.html#_CPPv4 |     [\[2\]](api                   |
| N5cudaq22KrausTrajectoryBuilderE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cud                          | udaq7qvector7qvectorERK7qvector), |
| aq::KrausTrajectoryBuilder::build |     [\[3\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/lang           | cudaq7qvector7qvectorERR7qvector) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::qvector::size (C++    |
| 22KrausTrajectoryBuilder5buildEv) |     fu                            |
| -   [cud                          | nction)](api/languages/cpp_api.ht |
| aq::KrausTrajectoryBuilder::setId | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     (C++                          | -   [cudaq::qvector::slice (C++   |
|     function)](api/languages/cpp  |     function)](api/language       |
| _api.html#_CPPv4N5cudaq22KrausTra | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| jectoryBuilder5setIdENSt6size_tE) | tor5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::Kraus                 | -   [cudaq::qvector::value_type   |
| TrajectoryBuilder::setProbability |     (C++                          |
|     (C++                          |     typ                           |
|     function)](api/languages/cpp  | e)](api/languages/cpp_api.html#_C |
| _api.html#_CPPv4N5cudaq22KrausTra | PPv4N5cudaq7qvector10value_typeE) |
| jectoryBuilder14setProbabilityEd) | -   [cudaq::qview (C++            |
| -   [cudaq::Krau                  |     clas                          |
| sTrajectoryBuilder::setSelections | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     function)](api/languag        | -   [cudaq::qview::back (C++      |
| es/cpp_api.html#_CPPv4N5cudaq22Kr |     function)                     |
| ausTrajectoryBuilder13setSelectio | ](api/languages/cpp_api.html#_CPP |
| nsENSt6vectorI14KrausSelectionEE) | v4N5cudaq5qview4backENSt6size_tE) |
| -   [cudaq::matrix_callback (C++  | -   [cudaq::qview::begin (C++     |
|     c                             |                                   |
| lass)](api/languages/cpp_api.html | function)](api/languages/cpp_api. |
| #_CPPv4N5cudaq15matrix_callbackE) | html#_CPPv4N5cudaq5qview5beginEv) |
| -   [cudaq::matrix_handler (C++   | -   [cudaq::qview::end (C++       |
|                                   |                                   |
| class)](api/languages/cpp_api.htm |   function)](api/languages/cpp_ap |
| l#_CPPv4N5cudaq14matrix_handlerE) | i.html#_CPPv4N5cudaq5qview3endEv) |
| -   [cudaq::mat                   | -   [cudaq::qview::front (C++     |
| rix_handler::commutation_behavior |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     struct)](api/languages/       | N5cudaq5qview5frontENSt6size_tE), |
| cpp_api.html#_CPPv4N5cudaq14matri |                                   |
| x_handler20commutation_behaviorE) |    [\[1\]](api/languages/cpp_api. |
| -                                 | html#_CPPv4N5cudaq5qview5frontEv) |
|    [cudaq::matrix_handler::define | -   [cudaq::qview::operator\[\]   |
|     (C++                          |     (C++                          |
|     function)](a                  |     functio                       |
| pi/languages/cpp_api.html#_CPPv4N | n)](api/languages/cpp_api.html#_C |
| 5cudaq14matrix_handler6defineENSt | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| 6stringENSt6vectorINSt7int64_tEEE | -   [cudaq::qview::qview (C++     |
| RR15matrix_callbackRKNSt13unorder |     functio                       |
| ed_mapINSt6stringENSt6stringEEE), | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4I0EN5cudaq5qview5qviewERR1R), |
| [\[1\]](api/languages/cpp_api.htm |     [\[1                          |
| l#_CPPv4N5cudaq14matrix_handler6d | \]](api/languages/cpp_api.html#_C |
| efineENSt6stringENSt6vectorINSt7i | PPv4N5cudaq5qview5qviewERK5qview) |
| nt64_tEEERR15matrix_callbackRR20d | -   [cudaq::qview::size (C++      |
| iag_matrix_callbackRKNSt13unorder |                                   |
| ed_mapINSt6stringENSt6stringEEE), | function)](api/languages/cpp_api. |
|     [\[2\]](                      | html#_CPPv4NK5cudaq5qview4sizeEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qview::slice (C++     |
| N5cudaq14matrix_handler6defineENS |     function)](api/langua         |
| t6stringENSt6vectorINSt7int64_tEE | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| ERR15matrix_callbackRRNSt13unorde | iew5sliceENSt6size_tENSt6size_tE) |
| red_mapINSt6stringENSt6stringEEE) | -   [cudaq::qview::value_type     |
| -                                 |     (C++                          |
|   [cudaq::matrix_handler::degrees |     t                             |
|     (C++                          | ype)](api/languages/cpp_api.html# |
|     function)](ap                 | _CPPv4N5cudaq5qview10value_typeE) |
| i/languages/cpp_api.html#_CPPv4NK | -   [cudaq::range (C++            |
| 5cudaq14matrix_handler7degreesEv) |     fun                           |
| -                                 | ction)](api/languages/cpp_api.htm |
|  [cudaq::matrix_handler::displace | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
|     (C++                          | orI11ElementTypeEE11ElementType), |
|     function)](api/language       |     [\[1\]](api/languages/cpp_    |
| s/cpp_api.html#_CPPv4N5cudaq14mat | api.html#_CPPv4I0EN5cudaq5rangeEN |
| rix_handler8displaceENSt6size_tE) | St6vectorI11ElementTypeEE11Elemen |
| -   [cudaq::matrix                | tType11ElementType11ElementType), |
| _handler::get_expected_dimensions |     [                             |
|     (C++                          | \[2\]](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq5rangeENSt6size_tE) |
|    function)](api/languages/cpp_a | -   [cudaq::real (C++             |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     type)](api/languages/         |
| ndler23get_expected_dimensionsEv) | cpp_api.html#_CPPv4N5cudaq4realE) |
| -   [cudaq::matrix_ha             | -   [cudaq::registry (C++         |
| ndler::get_parameter_descriptions |     type)](api/languages/cpp_     |
|     (C++                          | api.html#_CPPv4N5cudaq8registryE) |
|                                   | -                                 |
| function)](api/languages/cpp_api. |  [cudaq::registry::RegisteredType |
| html#_CPPv4NK5cudaq14matrix_handl |     (C++                          |
| er26get_parameter_descriptionsEv) |     class)](api/                  |
| -   [c                            | languages/cpp_api.html#_CPPv4I0EN |
| udaq::matrix_handler::instantiate | 5cudaq8registry14RegisteredTypeE) |
|     (C++                          | -   [cudaq::RemoteCapabilities    |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     struc                         |
| 5cudaq14matrix_handler11instantia | t)](api/languages/cpp_api.html#_C |
| teENSt6stringERKNSt6vectorINSt6si | PPv4N5cudaq18RemoteCapabilitiesE) |
| ze_tEEERK20commutation_behavior), | -   [cudaq::Remot                 |
|     [\[1\]](                      | eCapabilities::RemoteCapabilities |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14matrix_handler11instanti |     function)](api/languages/cpp  |
| ateENSt6stringERRNSt6vectorINSt6s | _api.html#_CPPv4N5cudaq18RemoteCa |
| ize_tEEERK20commutation_behavior) | pabilities18RemoteCapabilitiesEb) |
| -   [cuda                         | -   [cudaq:                       |
| q::matrix_handler::matrix_handler | :RemoteCapabilities::stateOverlap |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     member)](api/langua           |
| es/cpp_api.html#_CPPv4I0_NSt11ena | ges/cpp_api.html#_CPPv4N5cudaq18R |
| ble_if_tINSt12is_base_of_vI16oper | emoteCapabilities12stateOverlapE) |
| ator_handler1TEEbEEEN5cudaq14matr | -                                 |
| ix_handler14matrix_handlerERK1T), |   [cudaq::RemoteCapabilities::vqe |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4I0 |     member)](                     |
| _NSt11enable_if_tINSt12is_base_of | api/languages/cpp_api.html#_CPPv4 |
| _vI16operator_handler1TEEbEEEN5cu | N5cudaq18RemoteCapabilities3vqeE) |
| daq14matrix_handler14matrix_handl | -   [cudaq::Resources (C++        |
| erERK1TRK20commutation_behavior), |     class)](api/languages/cpp_a   |
|     [\[2\]](api/languages/cpp_ap  | pi.html#_CPPv4N5cudaq9ResourcesE) |
| i.html#_CPPv4N5cudaq14matrix_hand | -   [cudaq::run (C++              |
| ler14matrix_handlerENSt6size_tE), |     function)]                    |
|     [\[3\]](api/                  | (api/languages/cpp_api.html#_CPPv |
| languages/cpp_api.html#_CPPv4N5cu | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| daq14matrix_handler14matrix_handl | 5invoke_result_tINSt7decay_tI13Qu |
| erENSt6stringERKNSt6vectorINSt6si | antumKernelEEDpNSt7decay_tI4ARGSE |
| ze_tEEERK20commutation_behavior), | EEEEENSt6size_tERN5cudaq11noise_m |
|     [\[4\]](api/                  | odelERR13QuantumKernelDpRR4ARGS), |
| languages/cpp_api.html#_CPPv4N5cu |     [\[1\]](api/langu             |
| daq14matrix_handler14matrix_handl | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| erENSt6stringERRNSt6vectorINSt6si | daq3runENSt6vectorINSt15invoke_re |
| ze_tEEERK20commutation_behavior), | sult_tINSt7decay_tI13QuantumKerne |
|     [\                            | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| [5\]](api/languages/cpp_api.html# | ize_tERR13QuantumKernelDpRR4ARGS) |
| _CPPv4N5cudaq14matrix_handler14ma | -   [cudaq::run_async (C++        |
| trix_handlerERK14matrix_handler), |     functio                       |
|     [                             | n)](api/languages/cpp_api.html#_C |
| \[6\]](api/languages/cpp_api.html | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| #_CPPv4N5cudaq14matrix_handler14m | tureINSt6vectorINSt15invoke_resul |
| atrix_handlerERR14matrix_handler) | t_tINSt7decay_tI13QuantumKernelEE |
| -                                 | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|  [cudaq::matrix_handler::momentum | ze_tENSt6size_tERN5cudaq11noise_m |
|     (C++                          | odelERR13QuantumKernelDpRR4ARGS), |
|     function)](api/language       |     [\[1\]](api/la                |
| s/cpp_api.html#_CPPv4N5cudaq14mat | nguages/cpp_api.html#_CPPv4I0DpEN |
| rix_handler8momentumENSt6size_tE) | 5cudaq9run_asyncENSt6futureINSt6v |
| -                                 | ectorINSt15invoke_result_tINSt7de |
|    [cudaq::matrix_handler::number | cay_tI13QuantumKernelEEDpNSt7deca |
|     (C++                          | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     function)](api/langua         | ize_tERR13QuantumKernelDpRR4ARGS) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -   [cudaq::RuntimeTarget (C++    |
| atrix_handler6numberENSt6size_tE) |                                   |
| -                                 | struct)](api/languages/cpp_api.ht |
| [cudaq::matrix_handler::operator= | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     (C++                          | -   [cudaq::sample (C++           |
|     fun                           |     function)](api/languages/c    |
| ction)](api/languages/cpp_api.htm | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| l#_CPPv4I0_NSt11enable_if_tIXaant | mpleE13sample_resultRK14sample_op |
| NSt7is_sameI1T14matrix_handlerE5v | tionsRR13QuantumKernelDpRR4Args), |
| alueENSt12is_base_of_vI16operator |     [\[1\                         |
| _handler1TEEEbEEEN5cudaq14matrix_ | ]](api/languages/cpp_api.html#_CP |
| handleraSER14matrix_handlerRK1T), | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     [\[1\]](api/languages         | esultRR13QuantumKernelDpRR4Args), |
| /cpp_api.html#_CPPv4N5cudaq14matr |     [\                            |
| ix_handleraSERK14matrix_handler), | [2\]](api/languages/cpp_api.html# |
|     [\[2\]](api/language          | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ize_tERR13QuantumKernelDpRR4Args) |
| rix_handleraSERR14matrix_handler) | -   [cudaq::sample_options (C++   |
| -   [                             |     s                             |
| cudaq::matrix_handler::operator== | truct)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq14sample_optionsE) |
|     function)](api/languages      | -   [cudaq::sample_result (C++    |
| /cpp_api.html#_CPPv4NK5cudaq14mat |                                   |
| rix_handlereqERK14matrix_handler) |  class)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq13sample_resultE) |
|    [cudaq::matrix_handler::parity | -   [cudaq::sample_result::append |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/languages/cpp_ |
| ges/cpp_api.html#_CPPv4N5cudaq14m | api.html#_CPPv4N5cudaq13sample_re |
| atrix_handler6parityENSt6size_tE) | sult6appendERK15ExecutionResultb) |
| -                                 | -   [cudaq::sample_result::begin  |
|  [cudaq::matrix_handler::position |     (C++                          |
|     (C++                          |     function)]                    |
|     function)](api/language       | (api/languages/cpp_api.html#_CPPv |
| s/cpp_api.html#_CPPv4N5cudaq14mat | 4N5cudaq13sample_result5beginEv), |
| rix_handler8positionENSt6size_tE) |     [\[1\]]                       |
| -   [cudaq::                      | (api/languages/cpp_api.html#_CPPv |
| matrix_handler::remove_definition | 4NK5cudaq13sample_result5beginEv) |
|     (C++                          | -   [cudaq::sample_result::cbegin |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](                   |
| ml#_CPPv4N5cudaq14matrix_handler1 | api/languages/cpp_api.html#_CPPv4 |
| 7remove_definitionERKNSt6stringE) | NK5cudaq13sample_result6cbeginEv) |
| -                                 | -   [cudaq::sample_result::cend   |
|   [cudaq::matrix_handler::squeeze |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/languag        | ](api/languages/cpp_api.html#_CPP |
| es/cpp_api.html#_CPPv4N5cudaq14ma | v4NK5cudaq13sample_result4cendEv) |
| trix_handler7squeezeENSt6size_tE) | -   [cudaq::sample_result::clear  |
| -   [cudaq::m                     |     (C++                          |
| atrix_handler::to_diagonal_matrix |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/lang           | v4N5cudaq13sample_result5clearEv) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::sample_result::count  |
| 14matrix_handler18to_diagonal_mat |     (C++                          |
| rixERNSt13unordered_mapINSt6size_ |     function)](                   |
| tENSt7int64_tEEERKNSt13unordered_ | api/languages/cpp_api.html#_CPPv4 |
| mapINSt6stringENSt7complexIdEEEE) | NK5cudaq13sample_result5countENSt |
| -                                 | 11string_viewEKNSt11string_viewE) |
| [cudaq::matrix_handler::to_matrix | -   [                             |
|     (C++                          | cudaq::sample_result::deserialize |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     functio                       |
| v4NK5cudaq14matrix_handler9to_mat | n)](api/languages/cpp_api.html#_C |
| rixERNSt13unordered_mapINSt6size_ | PPv4N5cudaq13sample_result11deser |
| tENSt7int64_tEEERKNSt13unordered_ | ializeERNSt6vectorINSt6size_tEEE) |
| mapINSt6stringENSt7complexIdEEEE) | -   [cudaq::sample_result::dump   |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::to_string |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     function)](api/               | ample_result4dumpERNSt7ostreamE), |
| languages/cpp_api.html#_CPPv4NK5c |     [\[1\]                        |
| udaq14matrix_handler9to_stringEb) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4NK5cudaq13sample_result4dumpEv) |
| [cudaq::matrix_handler::unique_id | -   [cudaq::sample_result::end    |
|     (C++                          |     (C++                          |
|     function)](api/               |     function                      |
| languages/cpp_api.html#_CPPv4NK5c | )](api/languages/cpp_api.html#_CP |
| udaq14matrix_handler9unique_idEv) | Pv4N5cudaq13sample_result3endEv), |
| -   [cudaq:                       |     [\[1\                         |
| :matrix_handler::\~matrix_handler | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NK5cudaq13sample_result3endEv) |
|     functi                        | -   [                             |
| on)](api/languages/cpp_api.html#_ | cudaq::sample_result::expectation |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     (C++                          |
| -   [cudaq::matrix_op (C++        |     f                             |
|     type)](api/languages/cpp_a    | unction)](api/languages/cpp_api.h |
| pi.html#_CPPv4N5cudaq9matrix_opE) | tml#_CPPv4NK5cudaq13sample_result |
| -   [cudaq::matrix_op_term (C++   | 11expectationEKNSt11string_viewE) |
|                                   | -   [c                            |
|  type)](api/languages/cpp_api.htm | udaq::sample_result::get_marginal |
| l#_CPPv4N5cudaq14matrix_op_termE) |     (C++                          |
| -                                 |     function)](api/languages/cpp_ |
|    [cudaq::mdiag_operator_handler | api.html#_CPPv4NK5cudaq13sample_r |
|     (C++                          | esult12get_marginalERKNSt6vectorI |
|     class)](                      | NSt6size_tEEEKNSt11string_viewE), |
| api/languages/cpp_api.html#_CPPv4 |     [\[1\]](api/languages/cpp_    |
| N5cudaq22mdiag_operator_handlerE) | api.html#_CPPv4NK5cudaq13sample_r |
| -   [cudaq::mpi (C++              | esult12get_marginalERRKNSt6vector |
|     type)](api/languages          | INSt6size_tEEEKNSt11string_viewE) |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | -   [cuda                         |
| -   [cudaq::mpi::all_gather (C++  | q::sample_result::get_total_shots |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/langua         |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| RNSt6vectorIdEERKNSt6vectorIdEE), | sample_result15get_total_shotsEv) |
|                                   | -   [cuda                         |
|   [\[1\]](api/languages/cpp_api.h | q::sample_result::has_even_parity |
| tml#_CPPv4N5cudaq3mpi10all_gather |     (C++                          |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     fun                           |
| -   [cudaq::mpi::all_reduce (C++  | ction)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq13sample_result15h |
|  function)](api/languages/cpp_api | as_even_parityENSt11string_viewE) |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | -   [cuda                         |
| reduceE1TRK1TRK14BinaryFunction), | q::sample_result::has_expectation |
|     [\[1\]](api/langu             |     (C++                          |
| ages/cpp_api.html#_CPPv4I00EN5cud |     funct                         |
| aq3mpi10all_reduceE1TRK1TRK4Func) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::mpi::broadcast (C++   | _CPPv4NK5cudaq13sample_result15ha |
|     function)](api/               | s_expectationEKNSt11string_viewE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cu                           |
| daq3mpi9broadcastERNSt6stringEi), | daq::sample_result::most_probable |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     fun                           |
| q3mpi9broadcastERNSt6vectorIdEEi) | ction)](api/languages/cpp_api.htm |
| -   [cudaq::mpi::finalize (C++    | l#_CPPv4NK5cudaq13sample_result13 |
|     f                             | most_probableEKNSt11string_viewE) |
| unction)](api/languages/cpp_api.h | -                                 |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | [cudaq::sample_result::operator+= |
| -   [cudaq::mpi::initialize (C++  |     (C++                          |
|     function                      |     function)](api/langua         |
| )](api/languages/cpp_api.html#_CP | ges/cpp_api.html#_CPPv4N5cudaq13s |
| Pv4N5cudaq3mpi10initializeEiPPc), | ample_resultpLERK13sample_result) |
|     [                             | -                                 |
| \[1\]](api/languages/cpp_api.html |  [cudaq::sample_result::operator= |
| #_CPPv4N5cudaq3mpi10initializeEv) |     (C++                          |
| -   [cudaq::mpi::is_initialized   |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function                      | ample_resultaSERR13sample_result) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq3mpi14is_initializedEv) | [cudaq::sample_result::operator== |
| -   [cudaq::mpi::num_ranks (C++   |     (C++                          |
|     fu                            |     function)](api/languag        |
| nction)](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | ample_resulteqERK13sample_result) |
| -   [cudaq::mpi::rank (C++        | -   [                             |
|                                   | cudaq::sample_result::probability |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     function)](api/lan            |
| -   [cudaq::noise_model (C++      | guages/cpp_api.html#_CPPv4NK5cuda |
|                                   | q13sample_result11probabilityENSt |
|    class)](api/languages/cpp_api. | 11string_viewEKNSt11string_viewE) |
| html#_CPPv4N5cudaq11noise_modelE) | -   [cud                          |
| -   [cudaq::n                     | aq::sample_result::register_names |
| oise_model::add_all_qubit_channel |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](api                | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| /languages/cpp_api.html#_CPPv4IDp | 3sample_result14register_namesEv) |
| EN5cudaq11noise_model21add_all_qu | -                                 |
| bit_channelEvRK13kraus_channeli), |    [cudaq::sample_result::reorder |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     function)](api/langua         |
| oise_model21add_all_qubit_channel | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ERKNSt6stringERK13kraus_channeli) | ample_result7reorderERKNSt6vector |
| -                                 | INSt6size_tEEEKNSt11string_viewE) |
|  [cudaq::noise_model::add_channel | -   [cu                           |
|     (C++                          | daq::sample_result::sample_result |
|     funct                         |     (C++                          |
| ion)](api/languages/cpp_api.html# |     func                          |
| _CPPv4IDpEN5cudaq11noise_model11a | tion)](api/languages/cpp_api.html |
| dd_channelEvRK15PredicateFuncTy), | #_CPPv4N5cudaq13sample_result13sa |
|     [\[1\]](api/languages/cpp_    | mple_resultERK15ExecutionResult), |
| api.html#_CPPv4IDpEN5cudaq11noise |     [\[1\]](api/la                |
| _model11add_channelEvRKNSt6vector | nguages/cpp_api.html#_CPPv4N5cuda |
| INSt6size_tEEERK13kraus_channel), | q13sample_result13sample_resultER |
|     [\[2\]](ap                    | KNSt6vectorI15ExecutionResultEE), |
| i/languages/cpp_api.html#_CPPv4N5 |                                   |
| cudaq11noise_model11add_channelER |  [\[2\]](api/languages/cpp_api.ht |
| KNSt6stringERK15PredicateFuncTy), | ml#_CPPv4N5cudaq13sample_result13 |
|                                   | sample_resultERR13sample_result), |
| [\[3\]](api/languages/cpp_api.htm |     [                             |
| l#_CPPv4N5cudaq11noise_model11add | \[3\]](api/languages/cpp_api.html |
| _channelERKNSt6stringERKNSt6vecto | #_CPPv4N5cudaq13sample_result13sa |
| rINSt6size_tEEERK13kraus_channel) | mple_resultERR15ExecutionResult), |
| -   [cudaq::noise_model::empty    |     [\[4\]](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     function                      | 13sample_result13sample_resultEdR |
| )](api/languages/cpp_api.html#_CP | KNSt6vectorI15ExecutionResultEE), |
| Pv4NK5cudaq11noise_model5emptyEv) |     [\[5\]](api/lan               |
| -                                 | guages/cpp_api.html#_CPPv4N5cudaq |
| [cudaq::noise_model::get_channels | 13sample_result13sample_resultEv) |
|     (C++                          | -                                 |
|     function)](api/l              |  [cudaq::sample_result::serialize |
| anguages/cpp_api.html#_CPPv4I0ENK |     (C++                          |
| 5cudaq11noise_model12get_channels |     function)](api                |
| ENSt6vectorI13kraus_channelEERKNS | /languages/cpp_api.html#_CPPv4NK5 |
| t6vectorINSt6size_tEEERKNSt6vecto | cudaq13sample_result9serializeEv) |
| rINSt6size_tEEERKNSt6vectorIdEE), | -   [cudaq::sample_result::size   |
|     [\[1\]](api/languages/cpp_a   |     (C++                          |
| pi.html#_CPPv4NK5cudaq11noise_mod |     function)](api/languages/c    |
| el12get_channelsERKNSt6stringERKN | pp_api.html#_CPPv4NK5cudaq13sampl |
| St6vectorINSt6size_tEEERKNSt6vect | e_result4sizeEKNSt11string_viewE) |
| orINSt6size_tEEERKNSt6vectorIdEE) | -   [cudaq::sample_result::to_map |
| -                                 |     (C++                          |
|  [cudaq::noise_model::noise_model |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq13sample_ |
|     function)](api                | result6to_mapEKNSt11string_viewE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cuda                         |
| udaq11noise_model11noise_modelEv) | q::sample_result::\~sample_result |
| -   [cu                           |     (C++                          |
| daq::noise_model::PredicateFuncTy |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     type)](api/la                 | _CPPv4N5cudaq13sample_resultD0Ev) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::scalar_callback (C++  |
| q11noise_model15PredicateFuncTyE) |     c                             |
| -   [cud                          | lass)](api/languages/cpp_api.html |
| aq::noise_model::register_channel | #_CPPv4N5cudaq15scalar_callbackE) |
|     (C++                          | -   [c                            |
|     function)](api/languages      | udaq::scalar_callback::operator() |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     (C++                          |
| noise_model16register_channelEvv) |     function)](api/language       |
| -   [cudaq::                      | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| noise_model::requires_constructor | alar_callbackclERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|     type)](api/languages/cp       | -   [                             |
| p_api.html#_CPPv4I0DpEN5cudaq11no | cudaq::scalar_callback::operator= |
| ise_model20requires_constructorE) |     (C++                          |
| -   [cudaq::noise_model_type (C++ |     function)](api/languages/c    |
|     e                             | pp_api.html#_CPPv4N5cudaq15scalar |
| num)](api/languages/cpp_api.html# | _callbackaSERK15scalar_callback), |
| _CPPv4N5cudaq16noise_model_typeE) |     [\[1\]](api/languages/        |
| -   [cudaq::no                    | cpp_api.html#_CPPv4N5cudaq15scala |
| ise_model_type::amplitude_damping | r_callbackaSERR15scalar_callback) |
|     (C++                          | -   [cudaq:                       |
|     enumerator)](api/languages    | :scalar_callback::scalar_callback |
| /cpp_api.html#_CPPv4N5cudaq16nois |     (C++                          |
| e_model_type17amplitude_dampingE) |     function)](api/languag        |
| -   [cudaq::noise_mode            | es/cpp_api.html#_CPPv4I0_NSt11ena |
| l_type::amplitude_damping_channel | ble_if_tINSt16is_invocable_r_vINS |
|     (C++                          | t7complexIdEE8CallableRKNSt13unor |
|     e                             | dered_mapINSt6stringENSt7complexI |
| numerator)](api/languages/cpp_api | dEEEEEEbEEEN5cudaq15scalar_callba |
| .html#_CPPv4N5cudaq16noise_model_ | ck15scalar_callbackERR8Callable), |
| type25amplitude_damping_channelE) |     [\[1\                         |
| -   [cudaq::n                     | ]](api/languages/cpp_api.html#_CP |
| oise_model_type::bit_flip_channel | Pv4N5cudaq15scalar_callback15scal |
|     (C++                          | ar_callbackERK15scalar_callback), |
|     enumerator)](api/language     |     [\[2                          |
| s/cpp_api.html#_CPPv4N5cudaq16noi | \]](api/languages/cpp_api.html#_C |
| se_model_type16bit_flip_channelE) | PPv4N5cudaq15scalar_callback15sca |
| -   [cudaq::                      | lar_callbackERR15scalar_callback) |
| noise_model_type::depolarization1 | -   [cudaq::scalar_operator (C++  |
|     (C++                          |     c                             |
|     enumerator)](api/languag      | lass)](api/languages/cpp_api.html |
| es/cpp_api.html#_CPPv4N5cudaq16no | #_CPPv4N5cudaq15scalar_operatorE) |
| ise_model_type15depolarization1E) | -                                 |
| -   [cudaq::                      | [cudaq::scalar_operator::evaluate |
| noise_model_type::depolarization2 |     (C++                          |
|     (C++                          |                                   |
|     enumerator)](api/languag      |    function)](api/languages/cpp_a |
| es/cpp_api.html#_CPPv4N5cudaq16no | pi.html#_CPPv4NK5cudaq15scalar_op |
| ise_model_type15depolarization2E) | erator8evaluateERKNSt13unordered_ |
| -   [cudaq::noise_m               | mapINSt6stringENSt7complexIdEEEE) |
| odel_type::depolarization_channel | -   [cudaq::scalar_ope            |
|     (C++                          | rator::get_parameter_descriptions |
|                                   |     (C++                          |
|   enumerator)](api/languages/cpp_ |     f                             |
| api.html#_CPPv4N5cudaq16noise_mod | unction)](api/languages/cpp_api.h |
| el_type22depolarization_channelE) | tml#_CPPv4NK5cudaq15scalar_operat |
| -                                 | or26get_parameter_descriptionsEv) |
|  [cudaq::noise_model_type::pauli1 | -   [cu                           |
|     (C++                          | daq::scalar_operator::is_constant |
|     enumerator)](a                |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/lang           |
| 5cudaq16noise_model_type6pauli1E) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -                                 | 15scalar_operator11is_constantEv) |
|  [cudaq::noise_model_type::pauli2 | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator\* |
|     enumerator)](a                |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function                      |
| 5cudaq16noise_model_type6pauli2E) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq                        | Pv4N5cudaq15scalar_operatormlENSt |
| ::noise_model_type::phase_damping | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](api/langu        | ]](api/languages/cpp_api.html#_CP |
| ages/cpp_api.html#_CPPv4N5cudaq16 | Pv4N5cudaq15scalar_operatormlENSt |
| noise_model_type13phase_dampingE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::noi                   |     [\[2\]](api/languages/cp      |
| se_model_type::phase_flip_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormlEdRK15scalar_operator), |
|     enumerator)](api/languages/   |     [\[3\]](api/languages/cp      |
| cpp_api.html#_CPPv4N5cudaq16noise | p_api.html#_CPPv4N5cudaq15scalar_ |
| _model_type18phase_flip_channelE) | operatormlEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
| [cudaq::noise_model_type::unknown | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormlENSt7complexIdEE), |
|     enumerator)](ap               |     [\[5\]](api/languages/cpp     |
| i/languages/cpp_api.html#_CPPv4N5 | _api.html#_CPPv4NKR5cudaq15scalar |
| cudaq16noise_model_type7unknownE) | _operatormlERK15scalar_operator), |
| -                                 |     [\[6\]]                       |
| [cudaq::noise_model_type::x_error | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormlEd), |
|     enumerator)](ap               |     [\[7\]](api/language          |
| i/languages/cpp_api.html#_CPPv4N5 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| cudaq16noise_model_type7x_errorE) | alar_operatormlENSt7complexIdEE), |
| -                                 |     [\[8\]](api/languages/cp      |
| [cudaq::noise_model_type::y_error | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatormlERK15scalar_operator), |
|     enumerator)](ap               |     [\[9\                         |
| i/languages/cpp_api.html#_CPPv4N5 | ]](api/languages/cpp_api.html#_CP |
| cudaq16noise_model_type7y_errorE) | Pv4NO5cudaq15scalar_operatormlEd) |
| -                                 | -   [cu                           |
| [cudaq::noise_model_type::z_error | daq::scalar_operator::operator\*= |
|     (C++                          |     (C++                          |
|     enumerator)](ap               |     function)](api/languag        |
| i/languages/cpp_api.html#_CPPv4N5 | es/cpp_api.html#_CPPv4N5cudaq15sc |
| cudaq16noise_model_type7z_errorE) | alar_operatormLENSt7complexIdEE), |
| -   [cudaq::num_available_gpus    |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function                      | _operatormLERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[2                          |
| Pv4N5cudaq18num_available_gpusEv) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::observe (C++          | PPv4N5cudaq15scalar_operatormLEd) |
|     function)]                    | -   [                             |
| (api/languages/cpp_api.html#_CPPv | cudaq::scalar_operator::operator+ |
| 4I00DpEN5cudaq7observeENSt6vector |     (C++                          |
| I14observe_resultEERR13QuantumKer |     function                      |
| nelRK15SpinOpContainerDpRR4Args), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/languages/cpp_ap  | Pv4N5cudaq15scalar_operatorplENSt |
| i.html#_CPPv4I0DpEN5cudaq7observe | 7complexIdEERK15scalar_operator), |
| E14observe_resultNSt6size_tERR13Q |     [\[1\                         |
| uantumKernelRK7spin_opDpRR4Args), | ]](api/languages/cpp_api.html#_CP |
|     [\[                           | Pv4N5cudaq15scalar_operatorplENSt |
| 2\]](api/languages/cpp_api.html#_ | 7complexIdEERR15scalar_operator), |
| CPPv4I0DpEN5cudaq7observeE14obser |     [\[2\]](api/languages/cp      |
| ve_resultRK15observe_optionsRR13Q | p_api.html#_CPPv4N5cudaq15scalar_ |
| uantumKernelRK7spin_opDpRR4Args), | operatorplEdRK15scalar_operator), |
|     [\[3\]](api/lang              |     [\[3\]](api/languages/cp      |
| uages/cpp_api.html#_CPPv4I0DpEN5c | p_api.html#_CPPv4N5cudaq15scalar_ |
| udaq7observeE14observe_resultRR13 | operatorplEdRR15scalar_operator), |
| QuantumKernelRK7spin_opDpRR4Args) |     [\[4\]](api/languages         |
| -   [cudaq::observe_options (C++  | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     st                            | alar_operatorplENSt7complexIdEE), |
| ruct)](api/languages/cpp_api.html |     [\[5\]](api/languages/cpp     |
| #_CPPv4N5cudaq15observe_optionsE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::observe_result (C++   | _operatorplERK15scalar_operator), |
|                                   |     [\[6\]]                       |
| class)](api/languages/cpp_api.htm | (api/languages/cpp_api.html#_CPPv |
| l#_CPPv4N5cudaq14observe_resultE) | 4NKR5cudaq15scalar_operatorplEd), |
| -                                 |     [\[7\]]                       |
|    [cudaq::observe_result::counts | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatorplEv), |
|     function)](api/languages/c    |     [\[8\]](api/language          |
| pp_api.html#_CPPv4N5cudaq14observ | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| e_result6countsERK12spin_op_term) | alar_operatorplENSt7complexIdEE), |
| -   [cudaq::observe_result::dump  |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     function)                     | _operatorplERK15scalar_operator), |
| ](api/languages/cpp_api.html#_CPP |     [\[10\]                       |
| v4N5cudaq14observe_result4dumpEv) | ](api/languages/cpp_api.html#_CPP |
| -   [c                            | v4NO5cudaq15scalar_operatorplEd), |
| udaq::observe_result::expectation |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4NO5cudaq15scalar_operatorplEv) |
| function)](api/languages/cpp_api. | -   [c                            |
| html#_CPPv4N5cudaq14observe_resul | udaq::scalar_operator::operator+= |
| t11expectationERK12spin_op_term), |     (C++                          |
|     [\[1\]](api/la                |     function)](api/languag        |
| nguages/cpp_api.html#_CPPv4N5cuda | es/cpp_api.html#_CPPv4N5cudaq15sc |
| q14observe_result11expectationEv) | alar_operatorpLENSt7complexIdEE), |
| -   [cuda                         |     [\[1\]](api/languages/c       |
| q::observe_result::id_coefficient | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatorpLERK15scalar_operator), |
|     function)](api/langu          |     [\[2                          |
| ages/cpp_api.html#_CPPv4N5cudaq14 | \]](api/languages/cpp_api.html#_C |
| observe_result14id_coefficientEv) | PPv4N5cudaq15scalar_operatorpLEd) |
| -   [cuda                         | -   [                             |
| q::observe_result::observe_result | cudaq::scalar_operator::operator- |
|     (C++                          |     (C++                          |
|                                   |     function                      |
|   function)](api/languages/cpp_ap | )](api/languages/cpp_api.html#_CP |
| i.html#_CPPv4N5cudaq14observe_res | Pv4N5cudaq15scalar_operatormiENSt |
| ult14observe_resultEdRK7spin_op), | 7complexIdEERK15scalar_operator), |
|     [\[1\]](a                     |     [\[1\                         |
| pi/languages/cpp_api.html#_CPPv4N | ]](api/languages/cpp_api.html#_CP |
| 5cudaq14observe_result14observe_r | Pv4N5cudaq15scalar_operatormiENSt |
| esultEdRK7spin_op13sample_result) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
|  [cudaq::observe_result::operator | p_api.html#_CPPv4N5cudaq15scalar_ |
|     double (C++                   | operatormiEdRK15scalar_operator), |
|     functio                       |     [\[3\]](api/languages/cp      |
| n)](api/languages/cpp_api.html#_C | p_api.html#_CPPv4N5cudaq15scalar_ |
| PPv4N5cudaq14observe_resultcvdEv) | operatormiEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
|  [cudaq::observe_result::raw_data | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     function)](ap                 |     [\[5\]](api/languages/cpp     |
| i/languages/cpp_api.html#_CPPv4N5 | _api.html#_CPPv4NKR5cudaq15scalar |
| cudaq14observe_result8raw_dataEv) | _operatormiERK15scalar_operator), |
| -   [cudaq::operator_handler (C++ |     [\[6\]]                       |
|     cl                            | (api/languages/cpp_api.html#_CPPv |
| ass)](api/languages/cpp_api.html# | 4NKR5cudaq15scalar_operatormiEd), |
| _CPPv4N5cudaq16operator_handlerE) |     [\[7\]]                       |
| -   [cudaq::optimizable_function  | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEv), |
|     class)                        |     [\[8\]](api/language          |
| ](api/languages/cpp_api.html#_CPP | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| v4N5cudaq20optimizable_functionE) | alar_operatormiENSt7complexIdEE), |
| -   [cudaq::optimization_result   |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     type                          | _operatormiERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[10\]                       |
| Pv4N5cudaq19optimization_resultE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::optimizer (C++        | v4NO5cudaq15scalar_operatormiEd), |
|     class)](api/languages/cpp_a   |     [\[11\                        |
| pi.html#_CPPv4N5cudaq9optimizerE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::optimizer::optimize   | Pv4NO5cudaq15scalar_operatormiEv) |
|     (C++                          | -   [c                            |
|                                   | udaq::scalar_operator::operator-= |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4N5cudaq9optimizer8opt |     function)](api/languag        |
| imizeEKiRR20optimizable_function) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cu                           | alar_operatormIENSt7complexIdEE), |
| daq::optimizer::requiresGradients |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function)](api/la             | _operatormIERK15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[2                          |
| q9optimizer17requiresGradientsEv) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::orca (C++             | PPv4N5cudaq15scalar_operatormIEd) |
|     type)](api/languages/         | -   [                             |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | cudaq::scalar_operator::operator/ |
| -   [cudaq::orca::sample (C++     |     (C++                          |
|     function)](api/languages/c    |     function                      |
| pp_api.html#_CPPv4N5cudaq4orca6sa | )](api/languages/cpp_api.html#_CP |
| mpleERNSt6vectorINSt6size_tEEERNS | Pv4N5cudaq15scalar_operatordvENSt |
| t6vectorINSt6size_tEEERNSt6vector | 7complexIdEERK15scalar_operator), |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     [\[1\                         |
|     [\[1\]]                       | ]](api/languages/cpp_api.html#_CP |
| (api/languages/cpp_api.html#_CPPv | Pv4N5cudaq15scalar_operatordvENSt |
| 4N5cudaq4orca6sampleERNSt6vectorI | 7complexIdEERR15scalar_operator), |
| NSt6size_tEEERNSt6vectorINSt6size |     [\[2\]](api/languages/cp      |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::orca::sample_async    | operatordvEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
| function)](api/languages/cpp_api. | operatordvEdRR15scalar_operator), |
| html#_CPPv4N5cudaq4orca12sample_a |     [\[4\]](api/languages         |
| syncERNSt6vectorINSt6size_tEEERNS | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| t6vectorINSt6size_tEEERNSt6vector | alar_operatordvENSt7complexIdEE), |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     [\[5\]](api/languages/cpp     |
|     [\[1\]](api/la                | _api.html#_CPPv4NKR5cudaq15scalar |
| nguages/cpp_api.html#_CPPv4N5cuda | _operatordvERK15scalar_operator), |
| q4orca12sample_asyncERNSt6vectorI |     [\[6\]]                       |
| NSt6size_tEEERNSt6vectorINSt6size | (api/languages/cpp_api.html#_CPPv |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | 4NKR5cudaq15scalar_operatordvEd), |
| -   [cudaq::OrcaRemoteRESTQPU     |     [\[7\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     cla                           | alar_operatordvENSt7complexIdEE), |
| ss)](api/languages/cpp_api.html#_ |     [\[8\]](api/languages/cp      |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::pauli1 (C++           | _operatordvERK15scalar_operator), |
|     class)](api/languages/cp      |     [\[9\                         |
| p_api.html#_CPPv4N5cudaq6pauli1E) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4NO5cudaq15scalar_operatordvEd) |
|    [cudaq::pauli1::num_parameters | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator/= |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/languag        |
| 4N5cudaq6pauli114num_parametersE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::pauli1::num_targets   | alar_operatordVENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     membe                         | pp_api.html#_CPPv4N5cudaq15scalar |
| r)](api/languages/cpp_api.html#_C | _operatordVERK15scalar_operator), |
| PPv4N5cudaq6pauli111num_targetsE) |     [\[2                          |
| -   [cudaq::pauli1::pauli1 (C++   | \]](api/languages/cpp_api.html#_C |
|     function)](api/languages/cpp_ | PPv4N5cudaq15scalar_operatordVEd) |
| api.html#_CPPv4N5cudaq6pauli16pau | -   [                             |
| li1ERKNSt6vectorIN5cudaq4realEEE) | cudaq::scalar_operator::operator= |
| -   [cudaq::pauli2 (C++           |     (C++                          |
|     class)](api/languages/cp      |     function)](api/languages/c    |
| p_api.html#_CPPv4N5cudaq6pauli2E) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _operatoraSERK15scalar_operator), |
|    [cudaq::pauli2::num_parameters |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|     member)]                      | r_operatoraSERR15scalar_operator) |
| (api/languages/cpp_api.html#_CPPv | -   [c                            |
| 4N5cudaq6pauli214num_parametersE) | udaq::scalar_operator::operator== |
| -   [cudaq::pauli2::num_targets   |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     membe                         | pp_api.html#_CPPv4NK5cudaq15scala |
| r)](api/languages/cpp_api.html#_C | r_operatoreqERK15scalar_operator) |
| PPv4N5cudaq6pauli211num_targetsE) | -   [cudaq:                       |
| -   [cudaq::pauli2::pauli2 (C++   | :scalar_operator::scalar_operator |
|     function)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq6pauli26pau |     func                          |
| li2ERKNSt6vectorIN5cudaq4realEEE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::phase_damping (C++    | #_CPPv4N5cudaq15scalar_operator15 |
|                                   | scalar_operatorENSt7complexIdEE), |
|  class)](api/languages/cpp_api.ht |     [\[1\]](api/langu             |
| ml#_CPPv4N5cudaq13phase_dampingE) | ages/cpp_api.html#_CPPv4N5cudaq15 |
| -   [cud                          | scalar_operator15scalar_operatorE |
| aq::phase_damping::num_parameters | RK15scalar_callbackRRNSt13unorder |
|     (C++                          | ed_mapINSt6stringENSt6stringEEE), |
|     member)](api/lan              |     [\[2\                         |
| guages/cpp_api.html#_CPPv4N5cudaq | ]](api/languages/cpp_api.html#_CP |
| 13phase_damping14num_parametersE) | Pv4N5cudaq15scalar_operator15scal |
| -   [                             | ar_operatorERK15scalar_operator), |
| cudaq::phase_damping::num_targets |     [\[3\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     member)](api/                 | scalar_operator15scalar_operatorE |
| languages/cpp_api.html#_CPPv4N5cu | RR15scalar_callbackRRNSt13unorder |
| daq13phase_damping11num_targetsE) | ed_mapINSt6stringENSt6stringEEE), |
| -   [cudaq::phase_flip_channel    |     [\[4\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     clas                          | Pv4N5cudaq15scalar_operator15scal |
| s)](api/languages/cpp_api.html#_C | ar_operatorERR15scalar_operator), |
| PPv4N5cudaq18phase_flip_channelE) |     [\[5\]](api/language          |
| -   [cudaq::p                     | s/cpp_api.html#_CPPv4N5cudaq15sca |
| hase_flip_channel::num_parameters | lar_operator15scalar_operatorEd), |
|     (C++                          |     [\[6\]](api/languag           |
|     member)](api/language         | es/cpp_api.html#_CPPv4N5cudaq15sc |
| s/cpp_api.html#_CPPv4N5cudaq18pha | alar_operator15scalar_operatorEv) |
| se_flip_channel14num_parametersE) | -   [                             |
| -   [cudaq                        | cudaq::scalar_operator::to_matrix |
| ::phase_flip_channel::num_targets |     (C++                          |
|     (C++                          |                                   |
|     member)](api/langu            |   function)](api/languages/cpp_ap |
| ages/cpp_api.html#_CPPv4N5cudaq18 | i.html#_CPPv4NK5cudaq15scalar_ope |
| phase_flip_channel11num_targetsE) | rator9to_matrixERKNSt13unordered_ |
| -   [cudaq::product_op (C++       | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [                             |
|  class)](api/languages/cpp_api.ht | cudaq::scalar_operator::to_string |
| ml#_CPPv4I0EN5cudaq10product_opE) |     (C++                          |
| -   [cudaq::product_op::begin     |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     functio                       | daq15scalar_operator9to_stringEv) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::s                     |
| PPv4NK5cudaq10product_op5beginEv) | calar_operator::\~scalar_operator |
| -                                 |     (C++                          |
|  [cudaq::product_op::canonicalize |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     func                          | PPv4N5cudaq15scalar_operatorD0Ev) |
| tion)](api/languages/cpp_api.html | -   [cudaq::set_noise (C++        |
| #_CPPv4N5cudaq10product_op12canon |     function)](api/langu          |
| icalizeERKNSt3setINSt6size_tEEE), | ages/cpp_api.html#_CPPv4N5cudaq9s |
|     [\[1\]](api                   | et_noiseERKN5cudaq11noise_modelE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::set_random_seed (C++  |
| udaq10product_op12canonicalizeEv) |     function)](api/               |
| -   [                             | languages/cpp_api.html#_CPPv4N5cu |
| cudaq::product_op::const_iterator | daq15set_random_seedENSt6size_tE) |
|     (C++                          | -   [cudaq::simulation_precision  |
|     struct)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     enum)                         |
| daq10product_op14const_iteratorE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::product_o             | v4N5cudaq20simulation_precisionE) |
| p::const_iterator::const_iterator | -   [                             |
|     (C++                          | cudaq::simulation_precision::fp32 |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     enumerator)](api              |
| ml#_CPPv4N5cudaq10product_op14con | /languages/cpp_api.html#_CPPv4N5c |
| st_iterator14const_iteratorEPK10p | udaq20simulation_precision4fp32E) |
| roduct_opI9HandlerTyENSt6size_tE) | -   [                             |
| -   [cudaq::produ                 | cudaq::simulation_precision::fp64 |
| ct_op::const_iterator::operator!= |     (C++                          |
|     (C++                          |     enumerator)](api              |
|     fun                           | /languages/cpp_api.html#_CPPv4N5c |
| ction)](api/languages/cpp_api.htm | udaq20simulation_precision4fp64E) |
| l#_CPPv4NK5cudaq10product_op14con | -   [cudaq::SimulationState (C++  |
| st_iteratorneERK14const_iterator) |     c                             |
| -   [cudaq::produ                 | lass)](api/languages/cpp_api.html |
| ct_op::const_iterator::operator\* | #_CPPv4N5cudaq15SimulationStateE) |
|     (C++                          | -   [                             |
|     function)](api/lang           | cudaq::SimulationState::precision |
| uages/cpp_api.html#_CPPv4NK5cudaq |     (C++                          |
| 10product_op14const_iteratormlEv) |     enum)](api                    |
| -   [cudaq::produ                 | /languages/cpp_api.html#_CPPv4N5c |
| ct_op::const_iterator::operator++ | udaq15SimulationState9precisionE) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/lang           | :SimulationState::precision::fp32 |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     (C++                          |
| 0product_op14const_iteratorppEi), |     enumerator)](api/lang         |
|     [\[1\]](api/lan               | uages/cpp_api.html#_CPPv4N5cudaq1 |
| guages/cpp_api.html#_CPPv4N5cudaq | 5SimulationState9precision4fp32E) |
| 10product_op14const_iteratorppEv) | -   [cudaq:                       |
| -   [cudaq::produc                | :SimulationState::precision::fp64 |
| t_op::const_iterator::operator\-- |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     function)](api/lang           | uages/cpp_api.html#_CPPv4N5cudaq1 |
| uages/cpp_api.html#_CPPv4N5cudaq1 | 5SimulationState9precision4fp64E) |
| 0product_op14const_iteratormmEi), | -                                 |
|     [\[1\]](api/lan               |   [cudaq::SimulationState::Tensor |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 10product_op14const_iteratormmEv) |     struct)](                     |
| -   [cudaq::produc                | api/languages/cpp_api.html#_CPPv4 |
| t_op::const_iterator::operator-\> | N5cudaq15SimulationState6TensorE) |
|     (C++                          | -   [cudaq::spin_handler (C++     |
|     function)](api/lan            |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq |   class)](api/languages/cpp_api.h |
| 10product_op14const_iteratorptEv) | tml#_CPPv4N5cudaq12spin_handlerE) |
| -   [cudaq::produ                 | -   [cudaq:                       |
| ct_op::const_iterator::operator== | :spin_handler::to_diagonal_matrix |
|     (C++                          |     (C++                          |
|     fun                           |     function)](api/la             |
| ction)](api/languages/cpp_api.htm | nguages/cpp_api.html#_CPPv4NK5cud |
| l#_CPPv4NK5cudaq10product_op14con | aq12spin_handler18to_diagonal_mat |
| st_iteratoreqERK14const_iterator) | rixERNSt13unordered_mapINSt6size_ |
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
| -   [parameters                   | -   [PhaseDamping (class in       |
|     (cudaq.KrausChannel           |     cudaq)](api/languages/py      |
|     prope                         | thon_api.html#cudaq.PhaseDamping) |
| rty)](api/languages/python_api.ht | -   [PhaseFlipChannel (class in   |
| ml#cudaq.KrausChannel.parameters) |     cudaq)](api/languages/python  |
|     -   [(cu                      | _api.html#cudaq.PhaseFlipChannel) |
| daq.operators.boson.BosonOperator | -   [platform (cudaq.Target       |
|         property)](api/languag    |                                   |
| es/python_api.html#cudaq.operator |    property)](api/languages/pytho |
| s.boson.BosonOperator.parameters) | n_api.html#cudaq.Target.platform) |
|     -   [(cudaq.                  | -   [prepare_call()               |
| operators.boson.BosonOperatorTerm |     (cudaq.PyKernelDecorator      |
|                                   |     method)](a                    |
|        property)](api/languages/p | pi/languages/python_api.html#cuda |
| ython_api.html#cudaq.operators.bo | q.PyKernelDecorator.prepare_call) |
| son.BosonOperatorTerm.parameters) | -                                 |
|     -   [(cudaq.                  |    [ProbabilisticSamplingStrategy |
| operators.fermion.FermionOperator |     (class in                     |
|                                   |     cudaq.ptsbe)](api/la          |
|        property)](api/languages/p | nguages/python_api.html#cudaq.pts |
| ython_api.html#cudaq.operators.fe | be.ProbabilisticSamplingStrategy) |
| rmion.FermionOperator.parameters) | -   [probability                  |
|     -   [(cudaq.oper              |     (cudaq.ptsbe.KrausTrajectory  |
| ators.fermion.FermionOperatorTerm |     property)](api/               |
|                                   | languages/python_api.html#cudaq.p |
|    property)](api/languages/pytho | tsbe.KrausTrajectory.probability) |
| n_api.html#cudaq.operators.fermio |     -   [(cudaq.SampleResult      |
| n.FermionOperatorTerm.parameters) |         attribu                   |
|     -                             | te)](api/languages/python_api.htm |
|  [(cudaq.operators.MatrixOperator | l#cudaq.SampleResult.probability) |
|         property)](api/la         | -   [process_call_arguments()     |
| nguages/python_api.html#cudaq.ope |     (cudaq.PyKernelDecorator      |
| rators.MatrixOperator.parameters) |     method)](api/languag          |
|     -   [(cuda                    | es/python_api.html#cudaq.PyKernel |
| q.operators.MatrixOperatorElement | Decorator.process_call_arguments) |
|         property)](api/languages  | -   [ProductOperator (in module   |
| /python_api.html#cudaq.operators. |     cudaq.operator                |
| MatrixOperatorElement.parameters) | s)](api/languages/python_api.html |
|     -   [(c                       | #cudaq.operators.ProductOperator) |
| udaq.operators.MatrixOperatorTerm | -   [PROPORTIONAL                 |
|         property)](api/langua     |                                   |
| ges/python_api.html#cudaq.operato |   (cudaq.ptsbe.ShotAllocationType |
| rs.MatrixOperatorTerm.parameters) |     attribute)](api/lang          |
|     -                             | uages/python_api.html#cudaq.ptsbe |
|  [(cudaq.operators.ScalarOperator | .ShotAllocationType.PROPORTIONAL) |
|         property)](api/la         | -   [ptsbe_execution_data         |
| nguages/python_api.html#cudaq.ope |                                   |
| rators.ScalarOperator.parameters) |    (cudaq.ptsbe.PTSBESampleResult |
|     -   [(                        |     property)](api/languages/p    |
| cudaq.operators.spin.SpinOperator | ython_api.html#cudaq.ptsbe.PTSBES |
|         property)](api/langu      | ampleResult.ptsbe_execution_data) |
| ages/python_api.html#cudaq.operat | -   [PTSBEExecutionData (class in |
| ors.spin.SpinOperator.parameters) |     cudaq.pts                     |
|     -   [(cuda                    | be)](api/languages/python_api.htm |
| q.operators.spin.SpinOperatorTerm | l#cudaq.ptsbe.PTSBEExecutionData) |
|         property)](api/languages  | -   [PTSBESampleResult (class in  |
| /python_api.html#cudaq.operators. |     cudaq.pt                      |
| spin.SpinOperatorTerm.parameters) | sbe)](api/languages/python_api.ht |
| -   [ParameterShift (class in     | ml#cudaq.ptsbe.PTSBESampleResult) |
|     cudaq.gradien                 | -   [PTSSamplingStrategy (class   |
| ts)](api/languages/python_api.htm |     in                            |
| l#cudaq.gradients.ParameterShift) |     cudaq.ptsb                    |
| -   [params                       | e)](api/languages/python_api.html |
|     (cudaq.ptsbe.TraceInstruction | #cudaq.ptsbe.PTSSamplingStrategy) |
|     property)](                   | -   [PyKernel (class in           |
| api/languages/python_api.html#cud |     cudaq)](api/language          |
| aq.ptsbe.TraceInstruction.params) | s/python_api.html#cudaq.PyKernel) |
| -   [Pauli1 (class in             | -   [PyKernelDecorator (class in  |
|     cudaq)](api/langua            |     cudaq)](api/languages/python_ |
| ges/python_api.html#cudaq.Pauli1) | api.html#cudaq.PyKernelDecorator) |
| -   [Pauli2 (class in             |                                   |
|     cudaq)](api/langua            |                                   |
| ges/python_api.html#cudaq.Pauli2) |                                   |
| -   [per_qubit_depth              |                                   |
|     (cudaq.Resources              |                                   |
|     propert                       |                                   |
| y)](api/languages/python_api.html |                                   |
| #cudaq.Resources.per_qubit_depth) |                                   |
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
| cudaq.operators.spin.SpinOperator | tml#cudaq.ptsbe.TraceInstruction) |
|     attribute)](api/la            | -   [TraceInstructionType (class  |
| nguages/python_api.html#cudaq.ope |     in                            |
| rators.spin.SpinOperator.to_json) |     cudaq.ptsbe                   |
|     -   [(cuda                    | )](api/languages/python_api.html# |
| q.operators.spin.SpinOperatorTerm | cudaq.ptsbe.TraceInstructionType) |
|         attribute)](api/langua    | -   [trajectories                 |
| ges/python_api.html#cudaq.operato |                                   |
| rs.spin.SpinOperatorTerm.to_json) |   (cudaq.ptsbe.PTSBEExecutionData |
| -   [to_json()                    |     property)](api/lang           |
|     (cudaq.PyKernelDecorator      | uages/python_api.html#cudaq.ptsbe |
|     metho                         | .PTSBEExecutionData.trajectories) |
| d)](api/languages/python_api.html | -   [trajectory_id                |
| #cudaq.PyKernelDecorator.to_json) |     (cudaq.ptsbe.KrausTrajectory  |
| -   [to_matrix                    |     property)](api/la             |
|     (cu                           | nguages/python_api.html#cudaq.pts |
| daq.operators.boson.BosonOperator | be.KrausTrajectory.trajectory_id) |
|     attribute)](api/langua        | -   [translate() (in module       |
| ges/python_api.html#cudaq.operato |     cudaq)](api/languages         |
| rs.boson.BosonOperator.to_matrix) | /python_api.html#cudaq.translate) |
|     -   [(cudaq.ope               | -   [trim                         |
| rators.boson.BosonOperatorElement |     (cu                           |
|                                   | daq.operators.boson.BosonOperator |
|     attribute)](api/languages/pyt |     attribute)](api/l             |
| hon_api.html#cudaq.operators.boso | anguages/python_api.html#cudaq.op |
| n.BosonOperatorElement.to_matrix) | erators.boson.BosonOperator.trim) |
|     -   [(cudaq.                  |     -   [(cudaq.                  |
| operators.boson.BosonOperatorTerm | operators.fermion.FermionOperator |
|                                   |         attribute)](api/langu     |
|        attribute)](api/languages/ | ages/python_api.html#cudaq.operat |
| python_api.html#cudaq.operators.b | ors.fermion.FermionOperator.trim) |
| oson.BosonOperatorTerm.to_matrix) |     -                             |
|     -   [(cudaq.                  |  [(cudaq.operators.MatrixOperator |
| operators.fermion.FermionOperator |         attribute)](              |
|                                   | api/languages/python_api.html#cud |
|        attribute)](api/languages/ | aq.operators.MatrixOperator.trim) |
| python_api.html#cudaq.operators.f |     -   [(                        |
| ermion.FermionOperator.to_matrix) | cudaq.operators.spin.SpinOperator |
|     -   [(cudaq.operato           |         attribute)](api           |
| rs.fermion.FermionOperatorElement | /languages/python_api.html#cudaq. |
|                                   | operators.spin.SpinOperator.trim) |
| attribute)](api/languages/python_ | -   [type                         |
| api.html#cudaq.operators.fermion. |     (c                            |
| FermionOperatorElement.to_matrix) | udaq.ptsbe.ShotAllocationStrategy |
|     -   [(cudaq.oper              |     property)](api/               |
| ators.fermion.FermionOperatorTerm | languages/python_api.html#cudaq.p |
|                                   | tsbe.ShotAllocationStrategy.type) |
|    attribute)](api/languages/pyth |     -                             |
| on_api.html#cudaq.operators.fermi |    [(cudaq.ptsbe.TraceInstruction |
| on.FermionOperatorTerm.to_matrix) |         property)                 |
|     -                             | ](api/languages/python_api.html#c |
|  [(cudaq.operators.MatrixOperator | udaq.ptsbe.TraceInstruction.type) |
|         attribute)](api/l         | -   [type_to_str()                |
| anguages/python_api.html#cudaq.op |     (cudaq.PyKernelDecorator      |
| erators.MatrixOperator.to_matrix) |     static                        |
|     -   [(cuda                    |     method)](                     |
| q.operators.MatrixOperatorElement | api/languages/python_api.html#cud |
|         attribute)](api/language  | aq.PyKernelDecorator.type_to_str) |
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
