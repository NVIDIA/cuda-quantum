::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-3781
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
    -   [Dynamics
        Examples](using/examples/dynamics_examples.html){.reference
        .internal}
        -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
            Model)](examples/python/dynamics/dynamics_intro_1.html){.reference
            .internal}
            -   [Why dynamics simulations vs. circuit
                simulations?](examples/python/dynamics/dynamics_intro_1.html#Why-dynamics-simulations-vs.-circuit-simulations?){.reference
                .internal}
            -   [Functionality](examples/python/dynamics/dynamics_intro_1.html#Functionality){.reference
                .internal}
            -   [Performance](examples/python/dynamics/dynamics_intro_1.html#Performance){.reference
                .internal}
            -   [Section 1 - Simulating the Jaynes-Cummings
                Hamiltonian](examples/python/dynamics/dynamics_intro_1.html#Section-1---Simulating-the-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Exercise 1 - Simulating a many-photon Jaynes-Cummings
                Hamiltonian](examples/python/dynamics/dynamics_intro_1.html#Exercise-1---Simulating-a-many-photon-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Section 2 - Simulating open quantum systems with the
                [`collapse_operators`{.docutils .literal
                .notranslate}]{.pre}](examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L_i\\)]{.math .notranslate
                .nohighlight}](examples/python/dynamics/dynamics_intro_1.html#Exercise-2---Adding-additional-jump-operators-L_i){.reference
                .internal}
            -   [Section 3 - Many qubits coupled to the
                resonator](examples/python/dynamics/dynamics_intro_1.html#Section-3---Many-qubits-coupled-to-the-resonator){.reference
                .internal}
        -   [Introduction to CUDA-Q Dynamics (Time Dependent
            Hamiltonians)](examples/python/dynamics/dynamics_intro_2.html){.reference
            .internal}
            -   [The Landau-Zener
                model](examples/python/dynamics/dynamics_intro_2.html#The-Landau-Zener-model){.reference
                .internal}
            -   [Section 1 - Implementing time dependent
                terms](examples/python/dynamics/dynamics_intro_2.html#Section-1---Implementing-time-dependent-terms){.reference
                .internal}
            -   [Section 2 - Implementing custom
                operators](examples/python/dynamics/dynamics_intro_2.html#Section-2---Implementing-custom-operators){.reference
                .internal}
            -   [Section 3 - Heisenberg Model with a time-varying
                magnetic
                field](examples/python/dynamics/dynamics_intro_2.html#Section-3---Heisenberg-Model-with-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 1 - Define a time-varying magnetic
                field](examples/python/dynamics/dynamics_intro_2.html#Exercise-1---Define-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 2
                (Optional)](examples/python/dynamics/dynamics_intro_2.html#Exercise-2-(Optional)){.reference
                .internal}
        -   [Superconducting
            Qubits](examples/python/dynamics/superconducting.html){.reference
            .internal}
            -   [Cavity
                QED](examples/python/dynamics/superconducting.html#Cavity-QED){.reference
                .internal}
            -   [Cross
                Resonance](examples/python/dynamics/superconducting.html#Cross-Resonance){.reference
                .internal}
            -   [Transmon
                Resonator](examples/python/dynamics/superconducting.html#Transmon-Resonator){.reference
                .internal}
        -   [Spin
            Qubits](examples/python/dynamics/spinqubits.html){.reference
            .internal}
            -   [Silicon Spin
                Qubit](examples/python/dynamics/spinqubits.html#Silicon-Spin-Qubit){.reference
                .internal}
            -   [Heisenberg
                Model](examples/python/dynamics/spinqubits.html#Heisenberg-Model){.reference
                .internal}
        -   [Trapped Ion
            Qubits](examples/python/dynamics/iontrap.html){.reference
            .internal}
            -   [GHZ
                state](examples/python/dynamics/iontrap.html#GHZ-state){.reference
                .internal}
        -   [Control](examples/python/dynamics/control.html){.reference
            .internal}
            -   [Gate
                Calibration](examples/python/dynamics/control.html#Gate-Calibration){.reference
                .internal}
            -   [Pulse](examples/python/dynamics/control.html#Pulse){.reference
                .internal}
            -   [Qubit
                Control](examples/python/dynamics/control.html#Qubit-Control){.reference
                .internal}
            -   [Qubit
                Dynamics](examples/python/dynamics/control.html#Qubit-Dynamics){.reference
                .internal}
            -   [Landau-Zenner](examples/python/dynamics/control.html#Landau-Zenner){.reference
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
        -   [5. Compuare
            results](applications/python/qsci.html#5.-Compuare-results){.reference
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
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the [`[[4,2,2]]`{.docutils
            .literal .notranslate}]{.pre}
            encoding](applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
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
    -   [Quantum
        Transformer](applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](applications/python/quantum_transformer.html#Attention-Maps){.reference
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
            -   [The SKQD Algorithm: Matrix Construction
                Details](applications/python/skqd.html#The-SKQD-Algorithm:-Matrix-Construction-Details){.reference
                .internal}
        -   [Results Analysis and
            Convergence](applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [GPU Acceleration for
            Postprocessing](applications/python/skqd.html#GPU-Acceleration-for-Postprocessing){.reference
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
[**S**](#S) \| [**T**](#T) \| [**U**](#U) \| [**V**](#V) \| [**X**](#X)
\| [**Y**](#Y) \| [**Z**](#Z)
:::

## \_ {#_}

+-----------------------------------+-----------------------------------+
| -   [\_\_add\_\_()                | -   [\_\_iter\_\_()               |
|     (cudaq.QuakeValue             |     (cudaq.SampleResult           |
|                                   |     m                             |
|   method)](api/languages/python_a | ethod)](api/languages/python_api. |
| pi.html#cudaq.QuakeValue.__add__) | html#cudaq.SampleResult.__iter__) |
| -   [\_\_call\_\_()               | -   [\_\_len\_\_()                |
|     (cudaq.PyKernelDecorator      |     (cudaq.SampleResult           |
|     method                        |                                   |
| )](api/languages/python_api.html# | method)](api/languages/python_api |
| cudaq.PyKernelDecorator.__call__) | .html#cudaq.SampleResult.__len__) |
| -   [\_\_getitem\_\_()            | -   [\_\_mul\_\_()                |
|     (cudaq.ComplexMatrix          |     (cudaq.QuakeValue             |
|     metho                         |                                   |
| d)](api/languages/python_api.html |   method)](api/languages/python_a |
| #cudaq.ComplexMatrix.__getitem__) | pi.html#cudaq.QuakeValue.__mul__) |
|     -   [(cudaq.KrausChannel      | -   [\_\_neg\_\_()                |
|         meth                      |     (cudaq.QuakeValue             |
| od)](api/languages/python_api.htm |                                   |
| l#cudaq.KrausChannel.__getitem__) |   method)](api/languages/python_a |
|     -   [(cudaq.QuakeValue        | pi.html#cudaq.QuakeValue.__neg__) |
|         me                        | -   [\_\_radd\_\_()               |
| thod)](api/languages/python_api.h |     (cudaq.QuakeValue             |
| tml#cudaq.QuakeValue.__getitem__) |                                   |
|     -   [(cudaq.SampleResult      |  method)](api/languages/python_ap |
|         meth                      | i.html#cudaq.QuakeValue.__radd__) |
| od)](api/languages/python_api.htm | -   [\_\_rmul\_\_()               |
| l#cudaq.SampleResult.__getitem__) |     (cudaq.QuakeValue             |
| -   [\_\_init\_\_()               |                                   |
|                                   |  method)](api/languages/python_ap |
|    (cudaq.AmplitudeDampingChannel | i.html#cudaq.QuakeValue.__rmul__) |
|     method)](api                  | -   [\_\_rsub\_\_()               |
| /languages/python_api.html#cudaq. |     (cudaq.QuakeValue             |
| AmplitudeDampingChannel.__init__) |                                   |
|     -   [(cudaq.BitFlipChannel    |  method)](api/languages/python_ap |
|         met                       | i.html#cudaq.QuakeValue.__rsub__) |
| hod)](api/languages/python_api.ht | -   [\_\_str\_\_()                |
| ml#cudaq.BitFlipChannel.__init__) |     (cudaq.ComplexMatrix          |
|                                   |     m                             |
| -   [(cudaq.DepolarizationChannel | ethod)](api/languages/python_api. |
|         method)](a                | html#cudaq.ComplexMatrix.__str__) |
| pi/languages/python_api.html#cuda |     -   [(cudaq.PyKernelDecorator |
| q.DepolarizationChannel.__init__) |         metho                     |
|     -   [(cudaq.NoiseModel        | d)](api/languages/python_api.html |
|                                   | #cudaq.PyKernelDecorator.__str__) |
|  method)](api/languages/python_ap | -   [\_\_sub\_\_()                |
| i.html#cudaq.NoiseModel.__init__) |     (cudaq.QuakeValue             |
|     -   [(c                       |                                   |
| udaq.operators.RydbergHamiltonian |   method)](api/languages/python_a |
|         method)](api/lang         | pi.html#cudaq.QuakeValue.__sub__) |
| uages/python_api.html#cudaq.opera |                                   |
| tors.RydbergHamiltonian.__init__) |                                   |
|     -   [(cudaq.PhaseFlipChannel  |                                   |
|         metho                     |                                   |
| d)](api/languages/python_api.html |                                   |
| #cudaq.PhaseFlipChannel.__init__) |                                   |
+-----------------------------------+-----------------------------------+

## A {#A}

+-----------------------------------+-----------------------------------+
| -   [Adam (class in               | -   [append() (cudaq.KrausChannel |
|     cudaq                         |                                   |
| .optimizers)](api/languages/pytho |  method)](api/languages/python_ap |
| n_api.html#cudaq.optimizers.Adam) | i.html#cudaq.KrausChannel.append) |
| -   [add_all_qubit_channel()      | -   [argument_count               |
|     (cudaq.NoiseModel             |     (cudaq.PyKernel               |
|     method)](api                  |     attrib                        |
| /languages/python_api.html#cudaq. | ute)](api/languages/python_api.ht |
| NoiseModel.add_all_qubit_channel) | ml#cudaq.PyKernel.argument_count) |
| -   [add_channel()                | -   [arguments (cudaq.PyKernel    |
|     (cudaq.NoiseModel             |     a                             |
|     me                            | ttribute)](api/languages/python_a |
| thod)](api/languages/python_api.h | pi.html#cudaq.PyKernel.arguments) |
| tml#cudaq.NoiseModel.add_channel) | -   [as_pauli()                   |
| -   [all_gather() (in module      |     (cudaq.o                      |
|                                   | perators.spin.SpinOperatorElement |
|    cudaq.mpi)](api/languages/pyth |     method)](api/languages/       |
| on_api.html#cudaq.mpi.all_gather) | python_api.html#cudaq.operators.s |
| -   [amplitude() (cudaq.State     | pin.SpinOperatorElement.as_pauli) |
|     method)](api/languages/pytho  | -   [AsyncEvolveResult (class in  |
| n_api.html#cudaq.State.amplitude) |     cudaq)](api/languages/python_ |
| -   [AmplitudeDampingChannel      | api.html#cudaq.AsyncEvolveResult) |
|     (class in                     | -   [AsyncObserveResult (class in |
|     cu                            |                                   |
| daq)](api/languages/python_api.ht |    cudaq)](api/languages/python_a |
| ml#cudaq.AmplitudeDampingChannel) | pi.html#cudaq.AsyncObserveResult) |
| -   [amplitudes() (cudaq.State    | -   [AsyncSampleResult (class in  |
|     method)](api/languages/python |     cudaq)](api/languages/python_ |
| _api.html#cudaq.State.amplitudes) | api.html#cudaq.AsyncSampleResult) |
| -   [annihilate() (in module      | -   [AsyncStateResult (class in   |
|     c                             |     cudaq)](api/languages/python  |
| udaq.boson)](api/languages/python | _api.html#cudaq.AsyncStateResult) |
| _api.html#cudaq.boson.annihilate) |                                   |
|     -   [(in module               |                                   |
|         cudaq                     |                                   |
| .fermion)](api/languages/python_a |                                   |
| pi.html#cudaq.fermion.annihilate) |                                   |
+-----------------------------------+-----------------------------------+

## B {#B}

+-----------------------------------+-----------------------------------+
| -   [BaseIntegrator (class in     | -   [beta_reduction()             |
|                                   |     (cudaq.PyKernelDecorator      |
| cudaq.dynamics.integrator)](api/l |     method)](api                  |
| anguages/python_api.html#cudaq.dy | /languages/python_api.html#cudaq. |
| namics.integrator.BaseIntegrator) | PyKernelDecorator.beta_reduction) |
| -   [batch_size                   | -   [BitFlipChannel (class in     |
|     (cudaq.optimizers.Adam        |     cudaq)](api/languages/pyth    |
|     property                      | on_api.html#cudaq.BitFlipChannel) |
| )](api/languages/python_api.html# | -   [BosonOperator (class in      |
| cudaq.optimizers.Adam.batch_size) |     cudaq.operators.boson)](      |
|     -   [(cudaq.optimizers.SGD    | api/languages/python_api.html#cud |
|         propert                   | aq.operators.boson.BosonOperator) |
| y)](api/languages/python_api.html | -   [BosonOperatorElement (class  |
| #cudaq.optimizers.SGD.batch_size) |     in                            |
| -   [beta1 (cudaq.optimizers.Adam |                                   |
|     pro                           |   cudaq.operators.boson)](api/lan |
| perty)](api/languages/python_api. | guages/python_api.html#cudaq.oper |
| html#cudaq.optimizers.Adam.beta1) | ators.boson.BosonOperatorElement) |
| -   [beta2 (cudaq.optimizers.Adam | -   [BosonOperatorTerm (class in  |
|     pro                           |     cudaq.operators.boson)](api/  |
| perty)](api/languages/python_api. | languages/python_api.html#cudaq.o |
| html#cudaq.optimizers.Adam.beta2) | perators.boson.BosonOperatorTerm) |
|                                   | -   [broadcast() (in module       |
|                                   |     cudaq.mpi)](api/languages/pyt |
|                                   | hon_api.html#cudaq.mpi.broadcast) |
+-----------------------------------+-----------------------------------+

## C {#C}

+-----------------------------------+-----------------------------------+
| -   [canonicalize()               | -                                 |
|     (cu                           |    [cudaq::pauli1::num_parameters |
| daq.operators.boson.BosonOperator |     (C++                          |
|     method)](api/languages        |     member)]                      |
| /python_api.html#cudaq.operators. | (api/languages/cpp_api.html#_CPPv |
| boson.BosonOperator.canonicalize) | 4N5cudaq6pauli114num_parametersE) |
|     -   [(cudaq.                  | -   [cudaq::pauli1::num_targets   |
| operators.boson.BosonOperatorTerm |     (C++                          |
|                                   |     membe                         |
|        method)](api/languages/pyt | r)](api/languages/cpp_api.html#_C |
| hon_api.html#cudaq.operators.boso | PPv4N5cudaq6pauli111num_targetsE) |
| n.BosonOperatorTerm.canonicalize) | -   [cudaq::pauli1::pauli1 (C++   |
|     -   [(cudaq.                  |     function)](api/languages/cpp_ |
| operators.fermion.FermionOperator | api.html#_CPPv4N5cudaq6pauli16pau |
|                                   | li1ERKNSt6vectorIN5cudaq4realEEE) |
|        method)](api/languages/pyt | -   [cudaq::pauli2 (C++           |
| hon_api.html#cudaq.operators.ferm |     class)](api/languages/cp      |
| ion.FermionOperator.canonicalize) | p_api.html#_CPPv4N5cudaq6pauli2E) |
|     -   [(cudaq.oper              | -                                 |
| ators.fermion.FermionOperatorTerm |    [cudaq::pauli2::num_parameters |
|                                   |     (C++                          |
|    method)](api/languages/python_ |     member)]                      |
| api.html#cudaq.operators.fermion. | (api/languages/cpp_api.html#_CPPv |
| FermionOperatorTerm.canonicalize) | 4N5cudaq6pauli214num_parametersE) |
|     -                             | -   [cudaq::pauli2::num_targets   |
|  [(cudaq.operators.MatrixOperator |     (C++                          |
|         method)](api/lang         |     membe                         |
| uages/python_api.html#cudaq.opera | r)](api/languages/cpp_api.html#_C |
| tors.MatrixOperator.canonicalize) | PPv4N5cudaq6pauli211num_targetsE) |
|     -   [(c                       | -   [cudaq::pauli2::pauli2 (C++   |
| udaq.operators.MatrixOperatorTerm |     function)](api/languages/cpp_ |
|         method)](api/language     | api.html#_CPPv4N5cudaq6pauli26pau |
| s/python_api.html#cudaq.operators | li2ERKNSt6vectorIN5cudaq4realEEE) |
| .MatrixOperatorTerm.canonicalize) | -   [cudaq::phase_damping (C++    |
|     -   [(                        |                                   |
| cudaq.operators.spin.SpinOperator |  class)](api/languages/cpp_api.ht |
|         method)](api/languag      | ml#_CPPv4N5cudaq13phase_dampingE) |
| es/python_api.html#cudaq.operator | -   [cud                          |
| s.spin.SpinOperator.canonicalize) | aq::phase_damping::num_parameters |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     member)](api/lan              |
|         method)](api/languages/p  | guages/cpp_api.html#_CPPv4N5cudaq |
| ython_api.html#cudaq.operators.sp | 13phase_damping14num_parametersE) |
| in.SpinOperatorTerm.canonicalize) | -   [                             |
| -   [canonicalized() (in module   | cudaq::phase_damping::num_targets |
|     cuda                          |     (C++                          |
| q.boson)](api/languages/python_ap |     member)](api/                 |
| i.html#cudaq.boson.canonicalized) | languages/cpp_api.html#_CPPv4N5cu |
|     -   [(in module               | daq13phase_damping11num_targetsE) |
|         cudaq.fe                  | -   [cudaq::phase_flip_channel    |
| rmion)](api/languages/python_api. |     (C++                          |
| html#cudaq.fermion.canonicalized) |     clas                          |
|     -   [(in module               | s)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq18phase_flip_channelE) |
|        cudaq.operators.custom)](a | -   [cudaq::p                     |
| pi/languages/python_api.html#cuda | hase_flip_channel::num_parameters |
| q.operators.custom.canonicalized) |     (C++                          |
|     -   [(in module               |     member)](api/language         |
|         cu                        | s/cpp_api.html#_CPPv4N5cudaq18pha |
| daq.spin)](api/languages/python_a | se_flip_channel14num_parametersE) |
| pi.html#cudaq.spin.canonicalized) | -   [cudaq                        |
| -   [captured_variables()         | ::phase_flip_channel::num_targets |
|     (cudaq.PyKernelDecorator      |     (C++                          |
|     method)](api/lan              |     member)](api/langu            |
| guages/python_api.html#cudaq.PyKe | ages/cpp_api.html#_CPPv4N5cudaq18 |
| rnelDecorator.captured_variables) | phase_flip_channel11num_targetsE) |
| -   [CentralDifference (class in  | -   [cudaq::product_op (C++       |
|     cudaq.gradients)              |                                   |
| ](api/languages/python_api.html#c |  class)](api/languages/cpp_api.ht |
| udaq.gradients.CentralDifference) | ml#_CPPv4I0EN5cudaq10product_opE) |
| -   [clear() (cudaq.Resources     | -   [cudaq::product_op::begin     |
|     method)](api/languages/pytho  |     (C++                          |
| n_api.html#cudaq.Resources.clear) |     functio                       |
|     -   [(cudaq.SampleResult      | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4NK5cudaq10product_op5beginEv) |
|   method)](api/languages/python_a | -                                 |
| pi.html#cudaq.SampleResult.clear) |  [cudaq::product_op::canonicalize |
| -   [COBYLA (class in             |     (C++                          |
|     cudaq.o                       |     func                          |
| ptimizers)](api/languages/python_ | tion)](api/languages/cpp_api.html |
| api.html#cudaq.optimizers.COBYLA) | #_CPPv4N5cudaq10product_op12canon |
| -   [coefficient                  | icalizeERKNSt3setINSt6size_tEEE), |
|     (cudaq.                       |     [\[1\]](api                   |
| operators.boson.BosonOperatorTerm | /languages/cpp_api.html#_CPPv4N5c |
|     property)](api/languages/py   | udaq10product_op12canonicalizeEv) |
| thon_api.html#cudaq.operators.bos | -   [                             |
| on.BosonOperatorTerm.coefficient) | cudaq::product_op::const_iterator |
|     -   [(cudaq.oper              |     (C++                          |
| ators.fermion.FermionOperatorTerm |     struct)](api/                 |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|   property)](api/languages/python | daq10product_op14const_iteratorE) |
| _api.html#cudaq.operators.fermion | -   [cudaq::product_o             |
| .FermionOperatorTerm.coefficient) | p::const_iterator::const_iterator |
|     -   [(c                       |     (C++                          |
| udaq.operators.MatrixOperatorTerm |     fu                            |
|         property)](api/languag    | nction)](api/languages/cpp_api.ht |
| es/python_api.html#cudaq.operator | ml#_CPPv4N5cudaq10product_op14con |
| s.MatrixOperatorTerm.coefficient) | st_iterator14const_iteratorEPK10p |
|     -   [(cuda                    | roduct_opI9HandlerTyENSt6size_tE) |
| q.operators.spin.SpinOperatorTerm | -   [cudaq::produ                 |
|         property)](api/languages/ | ct_op::const_iterator::operator!= |
| python_api.html#cudaq.operators.s |     (C++                          |
| pin.SpinOperatorTerm.coefficient) |     fun                           |
| -   [col_count                    | ction)](api/languages/cpp_api.htm |
|     (cudaq.KrausOperator          | l#_CPPv4NK5cudaq10product_op14con |
|     prope                         | st_iteratorneERK14const_iterator) |
| rty)](api/languages/python_api.ht | -   [cudaq::produ                 |
| ml#cudaq.KrausOperator.col_count) | ct_op::const_iterator::operator\* |
| -   [compile()                    |     (C++                          |
|     (cudaq.PyKernelDecorator      |     function)](api/lang           |
|     metho                         | uages/cpp_api.html#_CPPv4NK5cudaq |
| d)](api/languages/python_api.html | 10product_op14const_iteratormlEv) |
| #cudaq.PyKernelDecorator.compile) | -   [cudaq::produ                 |
| -   [ComplexMatrix (class in      | ct_op::const_iterator::operator++ |
|     cudaq)](api/languages/pyt     |     (C++                          |
| hon_api.html#cudaq.ComplexMatrix) |     function)](api/lang           |
| -   [compute()                    | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     (                             | 0product_op14const_iteratorppEi), |
| cudaq.gradients.CentralDifference |     [\[1\]](api/lan               |
|     method)](api/la               | guages/cpp_api.html#_CPPv4N5cudaq |
| nguages/python_api.html#cudaq.gra | 10product_op14const_iteratorppEv) |
| dients.CentralDifference.compute) | -   [cudaq::produc                |
|     -   [(                        | t_op::const_iterator::operator\-- |
| cudaq.gradients.ForwardDifference |     (C++                          |
|         method)](api/la           |     function)](api/lang           |
| nguages/python_api.html#cudaq.gra | uages/cpp_api.html#_CPPv4N5cudaq1 |
| dients.ForwardDifference.compute) | 0product_op14const_iteratormmEi), |
|     -                             |     [\[1\]](api/lan               |
|  [(cudaq.gradients.ParameterShift | guages/cpp_api.html#_CPPv4N5cudaq |
|         method)](api              | 10product_op14const_iteratormmEv) |
| /languages/python_api.html#cudaq. | -   [cudaq::produc                |
| gradients.ParameterShift.compute) | t_op::const_iterator::operator-\> |
| -   [const()                      |     (C++                          |
|                                   |     function)](api/lan            |
|   (cudaq.operators.ScalarOperator | guages/cpp_api.html#_CPPv4N5cudaq |
|     class                         | 10product_op14const_iteratorptEv) |
|     method)](a                    | -   [cudaq::produ                 |
| pi/languages/python_api.html#cuda | ct_op::const_iterator::operator== |
| q.operators.ScalarOperator.const) |     (C++                          |
| -   [copy()                       |     fun                           |
|     (cu                           | ction)](api/languages/cpp_api.htm |
| daq.operators.boson.BosonOperator | l#_CPPv4NK5cudaq10product_op14con |
|     method)](api/l                | st_iteratoreqERK14const_iterator) |
| anguages/python_api.html#cudaq.op | -   [cudaq::product_op::degrees   |
| erators.boson.BosonOperator.copy) |     (C++                          |
|     -   [(cudaq.                  |     function)                     |
| operators.boson.BosonOperatorTerm | ](api/languages/cpp_api.html#_CPP |
|         method)](api/langu        | v4NK5cudaq10product_op7degreesEv) |
| ages/python_api.html#cudaq.operat | -   [cudaq::product_op::dump (C++ |
| ors.boson.BosonOperatorTerm.copy) |     functi                        |
|     -   [(cudaq.                  | on)](api/languages/cpp_api.html#_ |
| operators.fermion.FermionOperator | CPPv4NK5cudaq10product_op4dumpEv) |
|         method)](api/langu        | -   [cudaq::product_op::end (C++  |
| ages/python_api.html#cudaq.operat |     funct                         |
| ors.fermion.FermionOperator.copy) | ion)](api/languages/cpp_api.html# |
|     -   [(cudaq.oper              | _CPPv4NK5cudaq10product_op3endEv) |
| ators.fermion.FermionOperatorTerm | -   [c                            |
|         method)](api/languages    | udaq::product_op::get_coefficient |
| /python_api.html#cudaq.operators. |     (C++                          |
| fermion.FermionOperatorTerm.copy) |     function)](api/lan            |
|     -                             | guages/cpp_api.html#_CPPv4NK5cuda |
|  [(cudaq.operators.MatrixOperator | q10product_op15get_coefficientEv) |
|         method)](                 | -                                 |
| api/languages/python_api.html#cud |   [cudaq::product_op::get_term_id |
| aq.operators.MatrixOperator.copy) |     (C++                          |
|     -   [(c                       |     function)](api                |
| udaq.operators.MatrixOperatorTerm | /languages/cpp_api.html#_CPPv4NK5 |
|         method)](api/             | cudaq10product_op11get_term_idEv) |
| languages/python_api.html#cudaq.o | -                                 |
| perators.MatrixOperatorTerm.copy) |   [cudaq::product_op::is_identity |
|     -   [(                        |     (C++                          |
| cudaq.operators.spin.SpinOperator |     function)](api                |
|         method)](api              | /languages/cpp_api.html#_CPPv4NK5 |
| /languages/python_api.html#cudaq. | cudaq10product_op11is_identityEv) |
| operators.spin.SpinOperator.copy) | -   [cudaq::product_op::num_ops   |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     function)                     |
|         method)](api/lan          | ](api/languages/cpp_api.html#_CPP |
| guages/python_api.html#cudaq.oper | v4NK5cudaq10product_op7num_opsEv) |
| ators.spin.SpinOperatorTerm.copy) | -                                 |
| -   [count() (cudaq.Resources     |    [cudaq::product_op::operator\* |
|     method)](api/languages/pytho  |     (C++                          |
| n_api.html#cudaq.Resources.count) |     function)](api/languages/     |
|     -   [(cudaq.SampleResult      | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|                                   | oduct_opmlE10product_opI1TERK15sc |
|   method)](api/languages/python_a | alar_operatorRK10product_opI1TE), |
| pi.html#cudaq.SampleResult.count) |     [\[1\]](api/languages/        |
| -   [count_controls()             | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     (cudaq.Resources              | oduct_opmlE10product_opI1TERK15sc |
|     meth                          | alar_operatorRR10product_opI1TE), |
| od)](api/languages/python_api.htm |     [\[2\]](api/languages/        |
| l#cudaq.Resources.count_controls) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [counts()                     | oduct_opmlE10product_opI1TERR15sc |
|     (cudaq.ObserveResult          | alar_operatorRK10product_opI1TE), |
|                                   |     [\[3\]](api/languages/        |
| method)](api/languages/python_api | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| .html#cudaq.ObserveResult.counts) | oduct_opmlE10product_opI1TERR15sc |
| -   [create() (in module          | alar_operatorRR10product_opI1TE), |
|                                   |     [\[4\]](api/                  |
|    cudaq.boson)](api/languages/py | languages/cpp_api.html#_CPPv4I0EN |
| thon_api.html#cudaq.boson.create) | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [(in module               | K15scalar_operatorRK6sum_opI1TE), |
|         c                         |     [\[5\]](api/                  |
| udaq.fermion)](api/languages/pyth | languages/cpp_api.html#_CPPv4I0EN |
| on_api.html#cudaq.fermion.create) | 5cudaq10product_opmlE6sum_opI1TER |
| -   [csr_spmatrix (C++            | K15scalar_operatorRR6sum_opI1TE), |
|     type)](api/languages/c        |     [\[6\]](api/                  |
| pp_api.html#_CPPv412csr_spmatrix) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq                         | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [module](api/langua       | R15scalar_operatorRK6sum_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[7\]](api/                  |
| -   [cudaq (C++                   | languages/cpp_api.html#_CPPv4I0EN |
|     type)](api/lan                | 5cudaq10product_opmlE6sum_opI1TER |
| guages/cpp_api.html#_CPPv45cudaq) | R15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq.apply_noise() (in      |     [\[8\]](api/languages         |
|     module                        | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     cudaq)](api/languages/python_ | duct_opmlERK6sum_opI9HandlerTyE), |
| api.html#cudaq.cudaq.apply_noise) |     [\[9\]](api/languages/cpp_a   |
| -   cudaq.boson                   | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [module](api/languages/py | opmlERK10product_opI9HandlerTyE), |
| thon_api.html#module-cudaq.boson) |     [\[10\]](api/language         |
| -   cudaq.fermion                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opmlERK15scalar_operator), |
|   -   [module](api/languages/pyth |     [\[11\]](api/languages/cpp_a  |
| on_api.html#module-cudaq.fermion) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.operators.custom        | opmlERR10product_opI9HandlerTyE), |
|     -   [mo                       |     [\[12\]](api/language         |
| dule](api/languages/python_api.ht | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ml#module-cudaq.operators.custom) | roduct_opmlERR15scalar_operator), |
| -   cudaq.spin                    |     [\[13\]](api/languages/cpp_   |
|     -   [module](api/languages/p  | api.html#_CPPv4NO5cudaq10product_ |
| ython_api.html#module-cudaq.spin) | opmlERK10product_opI9HandlerTyE), |
| -   [cudaq::amplitude_damping     |     [\[14\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     cla                           | roduct_opmlERK15scalar_operator), |
| ss)](api/languages/cpp_api.html#_ |     [\[15\]](api/languages/cpp_   |
| CPPv4N5cudaq17amplitude_dampingE) | api.html#_CPPv4NO5cudaq10product_ |
| -                                 | opmlERR10product_opI9HandlerTyE), |
| [cudaq::amplitude_damping_channel |     [\[16\]](api/langua           |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     class)](api                   | product_opmlERR15scalar_operator) |
| /languages/cpp_api.html#_CPPv4N5c | -                                 |
| udaq25amplitude_damping_channelE) |   [cudaq::product_op::operator\*= |
| -   [cudaq::amplitud              |     (C++                          |
| e_damping_channel::num_parameters |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4N5cudaq10product_ |
|     member)](api/languages/cpp_a  | opmLERK10product_opI9HandlerTyE), |
| pi.html#_CPPv4N5cudaq25amplitude_ |     [\[1\]](api/langua            |
| damping_channel14num_parametersE) | ges/cpp_api.html#_CPPv4N5cudaq10p |
| -   [cudaq::ampli                 | roduct_opmLERK15scalar_operator), |
| tude_damping_channel::num_targets |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     member)](api/languages/cp     | _opmLERR10product_opI9HandlerTyE) |
| p_api.html#_CPPv4N5cudaq25amplitu | -   [cudaq::product_op::operator+ |
| de_damping_channel11num_targetsE) |     (C++                          |
| -   [cudaq::AnalogRemoteRESTQPU   |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     class                         | q10product_opplE6sum_opI1TERK15sc |
| )](api/languages/cpp_api.html#_CP | alar_operatorRK10product_opI1TE), |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) |     [\[1\]](api/                  |
| -   [cudaq::apply_noise (C++      | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/               | 5cudaq10product_opplE6sum_opI1TER |
| languages/cpp_api.html#_CPPv4I0Dp | K15scalar_operatorRK6sum_opI1TE), |
| EN5cudaq11apply_noiseEvDpRR4Args) |     [\[2\]](api/langu             |
| -   [cudaq::async_result (C++     | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     c                             | q10product_opplE6sum_opI1TERK15sc |
| lass)](api/languages/cpp_api.html | alar_operatorRR10product_opI1TE), |
| #_CPPv4I0EN5cudaq12async_resultE) |     [\[3\]](api/                  |
| -   [cudaq::async_result::get     | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     functi                        | K15scalar_operatorRR6sum_opI1TE), |
| on)](api/languages/cpp_api.html#_ |     [\[4\]](api/langu             |
| CPPv4N5cudaq12async_result3getEv) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::async_sample_result   | q10product_opplE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRK10product_opI1TE), |
|     type                          |     [\[5\]](api/                  |
| )](api/languages/cpp_api.html#_CP | languages/cpp_api.html#_CPPv4I0EN |
| Pv4N5cudaq19async_sample_resultE) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::BaseRemoteRESTQPU     | R15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[6\]](api/langu             |
|     cla                           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ss)](api/languages/cpp_api.html#_ | q10product_opplE6sum_opI1TERR15sc |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | alar_operatorRR10product_opI1TE), |
| -                                 |     [\[7\]](api/                  |
|    [cudaq::BaseRemoteSimulatorQPU | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     class)](                      | R15scalar_operatorRR6sum_opI1TE), |
| api/languages/cpp_api.html#_CPPv4 |     [\[8\]](api/languages/cpp_a   |
| N5cudaq22BaseRemoteSimulatorQPUE) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [cudaq::bit_flip_channel (C++ | opplERK10product_opI9HandlerTyE), |
|     cl                            |     [\[9\]](api/language          |
| ass)](api/languages/cpp_api.html# | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| _CPPv4N5cudaq16bit_flip_channelE) | roduct_opplERK15scalar_operator), |
| -   [cudaq:                       |     [\[10\]](api/languages/       |
| :bit_flip_channel::num_parameters | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     (C++                          | duct_opplERK6sum_opI9HandlerTyE), |
|     member)](api/langua           |     [\[11\]](api/languages/cpp_a  |
| ges/cpp_api.html#_CPPv4N5cudaq16b | pi.html#_CPPv4NKR5cudaq10product_ |
| it_flip_channel14num_parametersE) | opplERR10product_opI9HandlerTyE), |
| -   [cud                          |     [\[12\]](api/language         |
| aq::bit_flip_channel::num_targets | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opplERR15scalar_operator), |
|     member)](api/lan              |     [\[13\]](api/languages/       |
| guages/cpp_api.html#_CPPv4N5cudaq | cpp_api.html#_CPPv4NKR5cudaq10pro |
| 16bit_flip_channel11num_targetsE) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cudaq::boson_handler (C++    |     [\[                           |
|                                   | 14\]](api/languages/cpp_api.html# |
|  class)](api/languages/cpp_api.ht | _CPPv4NKR5cudaq10product_opplEv), |
| ml#_CPPv4N5cudaq13boson_handlerE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::boson_op (C++         | api.html#_CPPv4NO5cudaq10product_ |
|     type)](api/languages/cpp_     | opplERK10product_opI9HandlerTyE), |
| api.html#_CPPv4N5cudaq8boson_opE) |     [\[16\]](api/languag          |
| -   [cudaq::boson_op_term (C++    | es/cpp_api.html#_CPPv4NO5cudaq10p |
|                                   | roduct_opplERK15scalar_operator), |
|   type)](api/languages/cpp_api.ht |     [\[17\]](api/languages        |
| ml#_CPPv4N5cudaq13boson_op_termE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::CodeGenConfig (C++    | duct_opplERK6sum_opI9HandlerTyE), |
|                                   |     [\[18\]](api/languages/cpp_   |
| struct)](api/languages/cpp_api.ht | api.html#_CPPv4NO5cudaq10product_ |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::commutation_relations |     [\[19\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     struct)]                      | roduct_opplERR15scalar_operator), |
| (api/languages/cpp_api.html#_CPPv |     [\[20\]](api/languages        |
| 4N5cudaq21commutation_relationsE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::complex (C++          | duct_opplERR6sum_opI9HandlerTyE), |
|     type)](api/languages/cpp      |     [                             |
| _api.html#_CPPv4N5cudaq7complexE) | \[21\]](api/languages/cpp_api.htm |
| -   [cudaq::complex_matrix (C++   | l#_CPPv4NO5cudaq10product_opplEv) |
|                                   | -   [cudaq::product_op::operator- |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14complex_matrixE) |     function)](api/langu          |
| -                                 | ages/cpp_api.html#_CPPv4I0EN5cuda |
|   [cudaq::complex_matrix::adjoint | q10product_opmiE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRK10product_opI1TE), |
|     function)](a                  |     [\[1\]](api/                  |
| pi/languages/cpp_api.html#_CPPv4N | languages/cpp_api.html#_CPPv4I0EN |
| 5cudaq14complex_matrix7adjointEv) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [cudaq::                      | K15scalar_operatorRK6sum_opI1TE), |
| complex_matrix::diagonal_elements |     [\[2\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     function)](api/languages      | q10product_opmiE6sum_opI1TERK15sc |
| /cpp_api.html#_CPPv4NK5cudaq14com | alar_operatorRR10product_opI1TE), |
| plex_matrix17diagonal_elementsEi) |     [\[3\]](api/                  |
| -   [cudaq::complex_matrix::dump  | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/language       | K15scalar_operatorRR6sum_opI1TE), |
| s/cpp_api.html#_CPPv4NK5cudaq14co |     [\[4\]](api/langu             |
| mplex_matrix4dumpERNSt7ostreamE), | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     [\[1\]]                       | q10product_opmiE6sum_opI1TERR15sc |
| (api/languages/cpp_api.html#_CPPv | alar_operatorRK10product_opI1TE), |
| 4NK5cudaq14complex_matrix4dumpEv) |     [\[5\]](api/                  |
| -   [c                            | languages/cpp_api.html#_CPPv4I0EN |
| udaq::complex_matrix::eigenvalues | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | R15scalar_operatorRK6sum_opI1TE), |
|     function)](api/lan            |     [\[6\]](api/langu             |
| guages/cpp_api.html#_CPPv4NK5cuda | ages/cpp_api.html#_CPPv4I0EN5cuda |
| q14complex_matrix11eigenvaluesEv) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cu                           | alar_operatorRR10product_opI1TE), |
| daq::complex_matrix::eigenvectors |     [\[7\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/lang           | 5cudaq10product_opmiE6sum_opI1TER |
| uages/cpp_api.html#_CPPv4NK5cudaq | R15scalar_operatorRR6sum_opI1TE), |
| 14complex_matrix12eigenvectorsEv) |     [\[8\]](api/languages/cpp_a   |
| -   [c                            | pi.html#_CPPv4NKR5cudaq10product_ |
| udaq::complex_matrix::exponential | opmiERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[9\]](api/language          |
|     function)](api/la             | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| nguages/cpp_api.html#_CPPv4N5cuda | roduct_opmiERK15scalar_operator), |
| q14complex_matrix11exponentialEv) |     [\[10\]](api/languages/       |
| -                                 | cpp_api.html#_CPPv4NKR5cudaq10pro |
|  [cudaq::complex_matrix::identity | duct_opmiERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     function)](api/languages      | pi.html#_CPPv4NKR5cudaq10product_ |
| /cpp_api.html#_CPPv4N5cudaq14comp | opmiERR10product_opI9HandlerTyE), |
| lex_matrix8identityEKNSt6size_tE) |     [\[12\]](api/language         |
| -                                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| [cudaq::complex_matrix::kronecker | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     function)](api/lang           | cpp_api.html#_CPPv4NKR5cudaq10pro |
| uages/cpp_api.html#_CPPv4I00EN5cu | duct_opmiERR6sum_opI9HandlerTyE), |
| daq14complex_matrix9kroneckerE14c |     [\[                           |
| omplex_matrix8Iterable8Iterable), | 14\]](api/languages/cpp_api.html# |
|     [\[1\]](api/l                 | _CPPv4NKR5cudaq10product_opmiEv), |
| anguages/cpp_api.html#_CPPv4N5cud |     [\[15\]](api/languages/cpp_   |
| aq14complex_matrix9kroneckerERK14 | api.html#_CPPv4NO5cudaq10product_ |
| complex_matrixRK14complex_matrix) | opmiERK10product_opI9HandlerTyE), |
| -   [cudaq::c                     |     [\[16\]](api/languag          |
| omplex_matrix::minimal_eigenvalue | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmiERK15scalar_operator), |
|     function)](api/languages/     |     [\[17\]](api/languages        |
| cpp_api.html#_CPPv4NK5cudaq14comp | /cpp_api.html#_CPPv4NO5cudaq10pro |
| lex_matrix18minimal_eigenvalueEv) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [                             |     [\[18\]](api/languages/cpp_   |
| cudaq::complex_matrix::operator() | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmiERR10product_opI9HandlerTyE), |
|     function)](api/languages/cpp  |     [\[19\]](api/languag          |
| _api.html#_CPPv4N5cudaq14complex_ | es/cpp_api.html#_CPPv4NO5cudaq10p |
| matrixclENSt6size_tENSt6size_tE), | roduct_opmiERR15scalar_operator), |
|     [\[1\]](api/languages/cpp     |     [\[20\]](api/languages        |
| _api.html#_CPPv4NK5cudaq14complex | /cpp_api.html#_CPPv4NO5cudaq10pro |
| _matrixclENSt6size_tENSt6size_tE) | duct_opmiERR6sum_opI9HandlerTyE), |
| -   [                             |     [                             |
| cudaq::complex_matrix::operator\* | \[21\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NO5cudaq10product_opmiEv) |
|     function)](api/langua         | -   [cudaq::product_op::operator/ |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     (C++                          |
| omplex_matrixmlEN14complex_matrix |     function)](api/language       |
| 10value_typeERK14complex_matrix), | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     [\[1\]                        | roduct_opdvERK15scalar_operator), |
| ](api/languages/cpp_api.html#_CPP |     [\[1\]](api/language          |
| v4N5cudaq14complex_matrixmlERK14c | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| omplex_matrixRK14complex_matrix), | roduct_opdvERR15scalar_operator), |
|                                   |     [\[2\]](api/languag           |
|  [\[2\]](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ml#_CPPv4N5cudaq14complex_matrixm | roduct_opdvERK15scalar_operator), |
| lERK14complex_matrixRKNSt6vectorI |     [\[3\]](api/langua            |
| N14complex_matrix10value_typeEEE) | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| -                                 | product_opdvERR15scalar_operator) |
| [cudaq::complex_matrix::operator+ | -                                 |
|     (C++                          |    [cudaq::product_op::operator/= |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/langu          |
| Pv4N5cudaq14complex_matrixplERK14 | ages/cpp_api.html#_CPPv4N5cudaq10 |
| complex_matrixRK14complex_matrix) | product_opdVERK15scalar_operator) |
| -                                 | -   [cudaq::product_op::operator= |
| [cudaq::complex_matrix::operator- |     (C++                          |
|     (C++                          |     function)](api/la             |
|     function                      | nguages/cpp_api.html#_CPPv4I0_NSt |
| )](api/languages/cpp_api.html#_CP | 11enable_if_tIXaantNSt7is_sameI1T |
| Pv4N5cudaq14complex_matrixmiERK14 | 9HandlerTyE5valueENSt16is_constru |
| complex_matrixRK14complex_matrix) | ctibleI9HandlerTy1TE5valueEEbEEEN |
| -   [cu                           | 5cudaq10product_opaSER10product_o |
| daq::complex_matrix::operator\[\] | pI9HandlerTyERK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq10product_ |
|  function)](api/languages/cpp_api | opaSERK10product_opI9HandlerTyE), |
| .html#_CPPv4N5cudaq14complex_matr |     [\[2\]](api/languages/cp      |
| ixixERKNSt6vectorINSt6size_tEEE), | p_api.html#_CPPv4N5cudaq10product |
|     [\[1\]](api/languages/cpp_api | _opaSERR10product_opI9HandlerTyE) |
| .html#_CPPv4NK5cudaq14complex_mat | -                                 |
| rixixERKNSt6vectorINSt6size_tEEE) |    [cudaq::product_op::operator== |
| -   [cudaq::complex_matrix::power |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)]                    | _api.html#_CPPv4NK5cudaq10product |
| (api/languages/cpp_api.html#_CPPv | _opeqERK10product_opI9HandlerTyE) |
| 4N5cudaq14complex_matrix5powerEi) | -                                 |
| -                                 |  [cudaq::product_op::operator\[\] |
|  [cudaq::complex_matrix::set_zero |     (C++                          |
|     (C++                          |     function)](ap                 |
|     function)](ap                 | i/languages/cpp_api.html#_CPPv4NK |
| i/languages/cpp_api.html#_CPPv4N5 | 5cudaq10product_opixENSt6size_tE) |
| cudaq14complex_matrix8set_zeroEv) | -                                 |
| -                                 |    [cudaq::product_op::product_op |
| [cudaq::complex_matrix::to_string |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/               | pp_api.html#_CPPv4I0_NSt11enable_ |
| languages/cpp_api.html#_CPPv4NK5c | if_tIXaaNSt7is_sameI9HandlerTy14m |
| udaq14complex_matrix9to_stringEv) | atrix_handlerE5valueEaantNSt7is_s |
| -   [                             | ameI1T9HandlerTyE5valueENSt16is_c |
| cudaq::complex_matrix::value_type | onstructibleI9HandlerTy1TE5valueE |
|     (C++                          | EbEEEN5cudaq10product_op10product |
|     type)](api/                   | _opERK10product_opI1TERKN14matrix |
| languages/cpp_api.html#_CPPv4N5cu | _handler20commutation_behaviorE), |
| daq14complex_matrix10value_typeE) |                                   |
| -   [cudaq::contrib (C++          |  [\[1\]](api/languages/cpp_api.ht |
|     type)](api/languages/cpp      | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| _api.html#_CPPv4N5cudaq7contribE) | tNSt7is_sameI1T9HandlerTyE5valueE |
| -   [cudaq::contrib::draw (C++    | NSt16is_constructibleI9HandlerTy1 |
|     function)                     | TE5valueEEbEEEN5cudaq10product_op |
| ](api/languages/cpp_api.html#_CPP | 10product_opERK10product_opI1TE), |
| v4I0DpEN5cudaq7contrib4drawENSt6s |                                   |
| tringERR13QuantumKernelDpRR4Args) |   [\[2\]](api/languages/cpp_api.h |
| -                                 | tml#_CPPv4N5cudaq10product_op10pr |
| [cudaq::contrib::get_unitary_cmat | oduct_opENSt6size_tENSt6size_tE), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     function)](api/languages/cp   | p_api.html#_CPPv4N5cudaq10product |
| p_api.html#_CPPv4I0DpEN5cudaq7con | _op10product_opENSt7complexIdEE), |
| trib16get_unitary_cmatE14complex_ |     [\[4\]](api/l                 |
| matrixRR13QuantumKernelDpRR4Args) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::CusvState (C++        | aq10product_op10product_opERK10pr |
|                                   | oduct_opI9HandlerTyENSt6size_tE), |
|    class)](api/languages/cpp_api. |     [\[5\]](api/l                 |
| html#_CPPv4I0EN5cudaq9CusvStateE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::depolarization1 (C++  | aq10product_op10product_opERR10pr |
|     c                             | oduct_opI9HandlerTyENSt6size_tE), |
| lass)](api/languages/cpp_api.html |     [\[6\]](api/languages         |
| #_CPPv4N5cudaq15depolarization1E) | /cpp_api.html#_CPPv4N5cudaq10prod |
| -   [cudaq::depolarization2 (C++  | uct_op10product_opERR9HandlerTy), |
|     c                             |     [\[7\]](ap                    |
| lass)](api/languages/cpp_api.html | i/languages/cpp_api.html#_CPPv4N5 |
| #_CPPv4N5cudaq15depolarization2E) | cudaq10product_op10product_opEd), |
| -   [cudaq:                       |     [\[8\]](a                     |
| :depolarization2::depolarization2 | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq10product_op10product_opEv) |
|     function)](api/languages/cp   | -   [cuda                         |
| p_api.html#_CPPv4N5cudaq15depolar | q::product_op::to_diagonal_matrix |
| ization215depolarization2EK4real) |     (C++                          |
| -   [cudaq                        |     function)](api/               |
| ::depolarization2::num_parameters | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq10product_op18to_diagonal_mat |
|     member)](api/langu            | rixENSt13unordered_mapINSt6size_t |
| ages/cpp_api.html#_CPPv4N5cudaq15 | ENSt7int64_tEEERKNSt13unordered_m |
| depolarization214num_parametersE) | apINSt6stringENSt7complexIdEEEEb) |
| -   [cu                           | -   [cudaq::product_op::to_matrix |
| daq::depolarization2::num_targets |     (C++                          |
|     (C++                          |     funct                         |
|     member)](api/la               | ion)](api/languages/cpp_api.html# |
| nguages/cpp_api.html#_CPPv4N5cuda | _CPPv4NK5cudaq10product_op9to_mat |
| q15depolarization211num_targetsE) | rixENSt13unordered_mapINSt6size_t |
| -                                 | ENSt7int64_tEEERKNSt13unordered_m |
|    [cudaq::depolarization_channel | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cu                           |
|     class)](                      | daq::product_op::to_sparse_matrix |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22depolarization_channelE) |     function)](ap                 |
| -   [cudaq::depol                 | i/languages/cpp_api.html#_CPPv4NK |
| arization_channel::num_parameters | 5cudaq10product_op16to_sparse_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     member)](api/languages/cp     | ENSt7int64_tEEERKNSt13unordered_m |
| p_api.html#_CPPv4N5cudaq22depolar | apINSt6stringENSt7complexIdEEEEb) |
| ization_channel14num_parametersE) | -   [cudaq::product_op::to_string |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     member)](api/languages        | NK5cudaq10product_op9to_stringEv) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -                                 |
| larization_channel11num_targetsE) |  [cudaq::product_op::\~product_op |
| -   [cudaq::details (C++          |     (C++                          |
|     type)](api/languages/cpp      |     fu                            |
| _api.html#_CPPv4N5cudaq7detailsE) | nction)](api/languages/cpp_api.ht |
| -   [cudaq::details::future (C++  | ml#_CPPv4N5cudaq10product_opD0Ev) |
|                                   | -   [cudaq::QPU (C++              |
|  class)](api/languages/cpp_api.ht |     class)](api/languages         |
| ml#_CPPv4N5cudaq7details6futureE) | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| -                                 | -   [cudaq::QPU::beginExecution   |
|   [cudaq::details::future::future |     (C++                          |
|     (C++                          |     function                      |
|     functio                       | )](api/languages/cpp_api.html#_CP |
| n)](api/languages/cpp_api.html#_C | Pv4N5cudaq3QPU14beginExecutionEv) |
| PPv4N5cudaq7details6future6future | -   [cuda                         |
| ERNSt6vectorI3JobEERNSt6stringERN | q::QPU::configureExecutionContext |
| St3mapINSt6stringENSt6stringEEE), |     (C++                          |
|     [\[1\]](api/lang              |     funct                         |
| uages/cpp_api.html#_CPPv4N5cudaq7 | ion)](api/languages/cpp_api.html# |
| details6future6futureERR6future), | _CPPv4NK5cudaq3QPU25configureExec |
|     [\[2\]]                       | utionContextER16ExecutionContext) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::QPU::endExecution     |
| 4N5cudaq7details6future6futureEv) |     (C++                          |
| -   [cu                           |     functi                        |
| daq::details::kernel_builder_base | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq3QPU12endExecutionEv) |
|     class)](api/l                 | -   [cudaq::QPU::enqueue (C++     |
| anguages/cpp_api.html#_CPPv4N5cud |     function)](ap                 |
| aq7details19kernel_builder_baseE) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [cudaq::details::             | cudaq3QPU7enqueueER11QuantumTask) |
| kernel_builder_base::operator\<\< | -   [cud                          |
|     (C++                          | aq::QPU::finalizeExecutionContext |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq7de |     func                          |
| tails19kernel_builder_baselsERNSt | tion)](api/languages/cpp_api.html |
| 7ostreamERK19kernel_builder_base) | #_CPPv4NK5cudaq3QPU24finalizeExec |
| -   [                             | utionContextER16ExecutionContext) |
| cudaq::details::KernelBuilderType | -   [cudaq::QPU::getConnectivity  |
|     (C++                          |     (C++                          |
|     class)](api                   |     function)                     |
| /languages/cpp_api.html#_CPPv4N5c | ](api/languages/cpp_api.html#_CPP |
| udaq7details17KernelBuilderTypeE) | v4N5cudaq3QPU15getConnectivityEv) |
| -   [cudaq::d                     | -                                 |
| etails::KernelBuilderType::create | [cudaq::QPU::getExecutionThreadId |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api/               |
| ](api/languages/cpp_api.html#_CPP | languages/cpp_api.html#_CPPv4NK5c |
| v4N5cudaq7details17KernelBuilderT | udaq3QPU20getExecutionThreadIdEv) |
| ype6createEPN4mlir11MLIRContextE) | -   [cudaq::QPU::getNumQubits     |
| -   [cudaq::details::Ker          |     (C++                          |
| nelBuilderType::KernelBuilderType |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     function)](api/lang           | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -   [                             |
| details17KernelBuilderType17Kerne | cudaq::QPU::getRemoteCapabilities |
| lBuilderTypeERRNSt8functionIFN4ml |     (C++                          |
| ir4TypeEPN4mlir11MLIRContextEEEE) |     function)](api/l              |
| -   [cudaq::diag_matrix_callback  | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq3QPU21getRemoteCapabilitiesEv) |
|     class)                        | -   [cudaq::QPU::isEmulated (C++  |
| ](api/languages/cpp_api.html#_CPP |     func                          |
| v4N5cudaq20diag_matrix_callbackE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::dyn (C++              | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     member)](api/languages        | -   [cudaq::QPU::isSimulator (C++ |
| /cpp_api.html#_CPPv4N5cudaq3dynE) |     funct                         |
| -   [cudaq::ExecutionContext (C++ | ion)](api/languages/cpp_api.html# |
|     cl                            | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| ass)](api/languages/cpp_api.html# | -   [cudaq::QPU::launchKernel     |
| _CPPv4N5cudaq16ExecutionContextE) |     (C++                          |
| -   [cudaq                        |     function)](api/               |
| ::ExecutionContext::amplitudeMaps | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq3QPU12launchKernelERKNSt6strin |
|     member)](api/langu            | gE15KernelThunkTypePvNSt8uint64_t |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ENSt8uint64_tERKNSt6vectorIPvEE), |
| ExecutionContext13amplitudeMapsE) |                                   |
| -   [c                            |  [\[1\]](api/languages/cpp_api.ht |
| udaq::ExecutionContext::asyncExec | ml#_CPPv4N5cudaq3QPU12launchKerne |
|     (C++                          | lERKNSt6stringERKNSt6vectorIPvEE) |
|     member)](api/                 | -   [cudaq::QPU::onRandomSeedSet  |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9asyncExecE) |     function)](api/lang           |
| -   [cud                          | uages/cpp_api.html#_CPPv4N5cudaq3 |
| aq::ExecutionContext::asyncResult | QPU15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::QPU (C++         |
|     member)](api/lan              |     functio                       |
| guages/cpp_api.html#_CPPv4N5cudaq | n)](api/languages/cpp_api.html#_C |
| 16ExecutionContext11asyncResultE) | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| -   [cudaq:                       |                                   |
| :ExecutionContext::batchIteration |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     member)](api/langua           |     [\[2\]](api/languages/cpp_    |
| ges/cpp_api.html#_CPPv4N5cudaq16E | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| xecutionContext14batchIterationE) | -   [cudaq::QPU::setId (C++       |
| -   [cudaq::E                     |     function                      |
| xecutionContext::canHandleObserve | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     member)](api/language         | -   [cudaq::QPU::setShots (C++    |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     f                             |
| cutionContext16canHandleObserveE) | unction)](api/languages/cpp_api.h |
| -   [cudaq::E                     | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| xecutionContext::ExecutionContext | -   [cudaq::                      |
|     (C++                          | QPU::supportsExplicitMeasurements |
|     func                          |     (C++                          |
| tion)](api/languages/cpp_api.html |     function)](api/languag        |
| #_CPPv4N5cudaq16ExecutionContext1 | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| 6ExecutionContextERKNSt6stringE), | 28supportsExplicitMeasurementsEv) |
|     [\[1\]](api/languages/        | -   [cudaq::QPU::\~QPU (C++       |
| cpp_api.html#_CPPv4N5cudaq16Execu |     function)](api/languages/cp   |
| tionContext16ExecutionContextERKN | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| St6stringENSt6size_tENSt6size_tE) | -   [cudaq::QPUState (C++         |
| -   [cudaq::E                     |     class)](api/languages/cpp_    |
| xecutionContext::expectationValue | api.html#_CPPv4N5cudaq8QPUStateE) |
|     (C++                          | -   [cudaq::qreg (C++             |
|     member)](api/language         |     class)](api/lan               |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | guages/cpp_api.html#_CPPv4I_NSt6s |
| cutionContext16expectationValueE) | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| -   [cudaq::Execu                 | -   [cudaq::qreg::back (C++       |
| tionContext::explicitMeasurements |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/languages/cp     | v4N5cudaq4qreg4backENSt6size_tE), |
| p_api.html#_CPPv4N5cudaq16Executi |     [\[1\]](api/languages/cpp_ap  |
| onContext20explicitMeasurementsE) | i.html#_CPPv4N5cudaq4qreg4backEv) |
| -   [cuda                         | -   [cudaq::qreg::begin (C++      |
| q::ExecutionContext::futureResult |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     member)](api/lang             | .html#_CPPv4N5cudaq4qreg5beginEv) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::qreg::clear (C++      |
| 6ExecutionContext12futureResultE) |                                   |
| -   [cudaq::ExecutionContext      |  function)](api/languages/cpp_api |
| ::hasConditionalsOnMeasureResults | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     (C++                          | -   [cudaq::qreg::front (C++      |
|     mem                           |     function)]                    |
| ber)](api/languages/cpp_api.html# | (api/languages/cpp_api.html#_CPPv |
| _CPPv4N5cudaq16ExecutionContext31 | 4N5cudaq4qreg5frontENSt6size_tE), |
| hasConditionalsOnMeasureResultsE) |     [\[1\]](api/languages/cpp_api |
| -   [cudaq::Executi               | .html#_CPPv4N5cudaq4qreg5frontEv) |
| onContext::invocationResultBuffer | -   [cudaq::qreg::operator\[\]    |
|     (C++                          |     (C++                          |
|     member)](api/languages/cpp_   |     functi                        |
| api.html#_CPPv4N5cudaq16Execution | on)](api/languages/cpp_api.html#_ |
| Context22invocationResultBufferE) | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| -   [cu                           | -   [cudaq::qreg::qreg (C++       |
| daq::ExecutionContext::kernelName |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/la               | v4N5cudaq4qreg4qregENSt6size_tE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[1\]](api/languages/cpp_ap  |
| q16ExecutionContext10kernelNameE) | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| -   [cud                          | -   [cudaq::qreg::size (C++       |
| aq::ExecutionContext::kernelTrace |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     member)](api/lan              | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qreg::slice (C++      |
| 16ExecutionContext11kernelTraceE) |     function)](api/langu          |
| -   [cudaq:                       | ages/cpp_api.html#_CPPv4N5cudaq4q |
| :ExecutionContext::msm_dimensions | reg5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qreg::value_type (C++ |
|     member)](api/langua           |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq16E | type)](api/languages/cpp_api.html |
| xecutionContext14msm_dimensionsE) | #_CPPv4N5cudaq4qreg10value_typeE) |
| -   [cudaq::                      | -   [cudaq::qspan (C++            |
| ExecutionContext::msm_prob_err_id |     class)](api/lang              |
|     (C++                          | uages/cpp_api.html#_CPPv4I_NSt6si |
|     member)](api/languag          | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | -   [cudaq::QuakeValue (C++       |
| ecutionContext15msm_prob_err_idE) |     class)](api/languages/cpp_api |
| -   [cudaq::Ex                    | .html#_CPPv4N5cudaq10QuakeValueE) |
| ecutionContext::msm_probabilities | -   [cudaq::Q                     |
|     (C++                          | uakeValue::canValidateNumElements |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq16Exec |     function)](api/languages      |
| utionContext17msm_probabilitiesE) | /cpp_api.html#_CPPv4N5cudaq10Quak |
| -                                 | eValue22canValidateNumElementsEv) |
|    [cudaq::ExecutionContext::name | -                                 |
|     (C++                          |  [cudaq::QuakeValue::constantSize |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api                |
| 4N5cudaq16ExecutionContext4nameE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cu                           | udaq10QuakeValue12constantSizeEv) |
| daq::ExecutionContext::noiseModel | -   [cudaq::QuakeValue::dump (C++ |
|     (C++                          |     function)](api/lan            |
|     member)](api/la               | guages/cpp_api.html#_CPPv4N5cudaq |
| nguages/cpp_api.html#_CPPv4N5cuda | 10QuakeValue4dumpERNSt7ostreamE), |
| q16ExecutionContext10noiseModelE) |     [\                            |
| -   [cudaq::Exe                   | [1\]](api/languages/cpp_api.html# |
| cutionContext::numberTrajectories | _CPPv4N5cudaq10QuakeValue4dumpEv) |
|     (C++                          | -   [cudaq                        |
|     member)](api/languages/       | ::QuakeValue::getRequiredElements |
| cpp_api.html#_CPPv4N5cudaq16Execu |     (C++                          |
| tionContext18numberTrajectoriesE) |     function)](api/langua         |
| -   [c                            | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| udaq::ExecutionContext::optResult | uakeValue19getRequiredElementsEv) |
|     (C++                          | -   [cudaq::QuakeValue::getValue  |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)]                    |
| daq16ExecutionContext9optResultE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::Execu                 | 4NK5cudaq10QuakeValue8getValueEv) |
| tionContext::overlapComputeStates | -   [cudaq::QuakeValue::inverse   |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     function)                     |
| p_api.html#_CPPv4N5cudaq16Executi | ](api/languages/cpp_api.html#_CPP |
| onContext20overlapComputeStatesE) | v4NK5cudaq10QuakeValue7inverseEv) |
| -   [cudaq                        | -   [cudaq::QuakeValue::isStdVec  |
| ::ExecutionContext::overlapResult |     (C++                          |
|     (C++                          |     function)                     |
|     member)](api/langu            | ](api/languages/cpp_api.html#_CPP |
| ages/cpp_api.html#_CPPv4N5cudaq16 | v4N5cudaq10QuakeValue8isStdVecEv) |
| ExecutionContext13overlapResultE) | -                                 |
| -                                 |    [cudaq::QuakeValue::operator\* |
|   [cudaq::ExecutionContext::qpuId |     (C++                          |
|     (C++                          |     function)](api                |
|     member)](                     | /languages/cpp_api.html#_CPPv4N5c |
| api/languages/cpp_api.html#_CPPv4 | udaq10QuakeValuemlE10QuakeValue), |
| N5cudaq16ExecutionContext5qpuIdE) |                                   |
| -   [cudaq                        | [\[1\]](api/languages/cpp_api.htm |
| ::ExecutionContext::registerNames | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|     (C++                          | -   [cudaq::QuakeValue::operator+ |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     function)](api                |
| ExecutionContext13registerNamesE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cu                           | udaq10QuakeValueplE10QuakeValue), |
| daq::ExecutionContext::reorderIdx |     [                             |
|     (C++                          | \[1\]](api/languages/cpp_api.html |
|     member)](api/la               | #_CPPv4N5cudaq10QuakeValueplEKd), |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q16ExecutionContext10reorderIdxE) | [\[2\]](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|  [cudaq::ExecutionContext::result | -   [cudaq::QuakeValue::operator- |
|     (C++                          |     (C++                          |
|     member)](a                    |     function)](api                |
| pi/languages/cpp_api.html#_CPPv4N | /languages/cpp_api.html#_CPPv4N5c |
| 5cudaq16ExecutionContext6resultE) | udaq10QuakeValuemiE10QuakeValue), |
| -                                 |     [                             |
|   [cudaq::ExecutionContext::shots | \[1\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     member)](                     |     [                             |
| api/languages/cpp_api.html#_CPPv4 | \[2\]](api/languages/cpp_api.html |
| N5cudaq16ExecutionContext5shotsE) | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| -   [cudaq::                      |                                   |
| ExecutionContext::simulationState | [\[3\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     member)](api/languag          | -   [cudaq::QuakeValue::operator/ |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15simulationStateE) |     function)](api                |
| -                                 | /languages/cpp_api.html#_CPPv4N5c |
|    [cudaq::ExecutionContext::spin | udaq10QuakeValuedvE10QuakeValue), |
|     (C++                          |                                   |
|     member)]                      | [\[1\]](api/languages/cpp_api.htm |
| (api/languages/cpp_api.html#_CPPv | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| 4N5cudaq16ExecutionContext4spinE) | -                                 |
| -   [cudaq::                      |  [cudaq::QuakeValue::operator\[\] |
| ExecutionContext::totalIterations |     (C++                          |
|     (C++                          |     function)](api                |
|     member)](api/languag          | /languages/cpp_api.html#_CPPv4N5c |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | udaq10QuakeValueixEKNSt6size_tE), |
| ecutionContext15totalIterationsE) |     [\[1\]](api/                  |
| -   [cudaq::Executio              | languages/cpp_api.html#_CPPv4N5cu |
| nContext::warnedNamedMeasurements | daq10QuakeValueixERK10QuakeValue) |
|     (C++                          | -                                 |
|     member)](api/languages/cpp_a  |    [cudaq::QuakeValue::QuakeValue |
| pi.html#_CPPv4N5cudaq16ExecutionC |     (C++                          |
| ontext23warnedNamedMeasurementsE) |     function)](api/languag        |
| -   [cudaq::ExecutionResult (C++  | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     st                            | akeValue10QuakeValueERN4mlir20Imp |
| ruct)](api/languages/cpp_api.html | licitLocOpBuilderEN4mlir5ValueE), |
| #_CPPv4N5cudaq15ExecutionResultE) |     [\[1\]                        |
| -   [cud                          | ](api/languages/cpp_api.html#_CPP |
| aq::ExecutionResult::appendResult | v4N5cudaq10QuakeValue10QuakeValue |
|     (C++                          | ERN4mlir20ImplicitLocOpBuilderEd) |
|     functio                       | -   [cudaq::QuakeValue::size (C++ |
| n)](api/languages/cpp_api.html#_C |     funct                         |
| PPv4N5cudaq15ExecutionResult12app | ion)](api/languages/cpp_api.html# |
| endResultENSt6stringENSt6size_tE) | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| -   [cu                           | -   [cudaq::QuakeValue::slice     |
| daq::ExecutionResult::deserialize |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     function)                     | api.html#_CPPv4N5cudaq10QuakeValu |
| ](api/languages/cpp_api.html#_CPP | e5sliceEKNSt6size_tEKNSt6size_tE) |
| v4N5cudaq15ExecutionResult11deser | -   [cudaq::quantum_platform (C++ |
| ializeERNSt6vectorINSt6size_tEEE) |     cl                            |
| -   [cudaq:                       | ass)](api/languages/cpp_api.html# |
| :ExecutionResult::ExecutionResult | _CPPv4N5cudaq16quantum_platformE) |
|     (C++                          | -   [cudaq:                       |
|     functio                       | :quantum_platform::beginExecution |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq15ExecutionResult15Exe |     function)](api/languag        |
| cutionResultE16CountsDictionary), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[1\]](api/lan               | antum_platform14beginExecutionEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::quantum_pl            |
| 15ExecutionResult15ExecutionResul | atform::configureExecutionContext |
| tE16CountsDictionaryNSt6stringE), |     (C++                          |
|     [\[2\                         |     function)](api/lang           |
| ]](api/languages/cpp_api.html#_CP | uages/cpp_api.html#_CPPv4NK5cudaq |
| Pv4N5cudaq15ExecutionResult15Exec | 16quantum_platform25configureExec |
| utionResultE16CountsDictionaryd), | utionContextER16ExecutionContext) |
|                                   | -   [cuda                         |
|    [\[3\]](api/languages/cpp_api. | q::quantum_platform::connectivity |
| html#_CPPv4N5cudaq15ExecutionResu |     (C++                          |
| lt15ExecutionResultENSt6stringE), |     function)](api/langu          |
|     [\[4\                         | ages/cpp_api.html#_CPPv4N5cudaq16 |
| ]](api/languages/cpp_api.html#_CP | quantum_platform12connectivityEv) |
| Pv4N5cudaq15ExecutionResult15Exec | -   [cuda                         |
| utionResultERK15ExecutionResult), | q::quantum_platform::endExecution |
|     [\[5\]](api/language          |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     function)](api/langu          |
| cutionResult15ExecutionResultEd), | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     [\[6\]](api/languag           | quantum_platform12endExecutionEv) |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | -   [cudaq::q                     |
| ecutionResult15ExecutionResultEv) | uantum_platform::enqueueAsyncTask |
| -   [                             |     (C++                          |
| cudaq::ExecutionResult::operator= |     function)](api/languages/     |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq16quant |
|     function)](api/languages/     | um_platform16enqueueAsyncTaskEKNS |
| cpp_api.html#_CPPv4N5cudaq15Execu | t6size_tER19KernelExecutionTask), |
| tionResultaSERK15ExecutionResult) |     [\[1\]](api/languag           |
| -   [c                            | es/cpp_api.html#_CPPv4N5cudaq16qu |
| udaq::ExecutionResult::operator== | antum_platform16enqueueAsyncTaskE |
|     (C++                          | KNSt6size_tERNSt8functionIFvvEEE) |
|     function)](api/languages/c    | -   [cudaq::quantum_p             |
| pp_api.html#_CPPv4NK5cudaq15Execu | latform::finalizeExecutionContext |
| tionResulteqERK15ExecutionResult) |     (C++                          |
| -   [cud                          |     function)](api/languages/c    |
| aq::ExecutionResult::registerName | pp_api.html#_CPPv4NK5cudaq16quant |
|     (C++                          | um_platform24finalizeExecutionCon |
|     member)](api/lan              | textERN5cudaq16ExecutionContextE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qua                   |
| 15ExecutionResult12registerNameE) | ntum_platform::get_codegen_config |
| -   [cudaq                        |     (C++                          |
| ::ExecutionResult::sequentialData |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq16quantu |
|     member)](api/langu            | m_platform18get_codegen_configEv) |
| ages/cpp_api.html#_CPPv4N5cudaq15 | -   [cuda                         |
| ExecutionResult14sequentialDataE) | q::quantum_platform::get_exec_ctx |
| -   [                             |     (C++                          |
| cudaq::ExecutionResult::serialize |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|     function)](api/l              | quantum_platform12get_exec_ctxEv) |
| anguages/cpp_api.html#_CPPv4NK5cu | -   [c                            |
| daq15ExecutionResult9serializeEv) | udaq::quantum_platform::get_noise |
| -   [cudaq::fermion_handler (C++  |     (C++                          |
|     c                             |     function)](api/languages/c    |
| lass)](api/languages/cpp_api.html | pp_api.html#_CPPv4N5cudaq16quantu |
| #_CPPv4N5cudaq15fermion_handlerE) | m_platform9get_noiseENSt6size_tE) |
| -   [cudaq::fermion_op (C++       | -   [cudaq:                       |
|     type)](api/languages/cpp_api  | :quantum_platform::get_num_qubits |
| .html#_CPPv4N5cudaq10fermion_opE) |     (C++                          |
| -   [cudaq::fermion_op_term (C++  |                                   |
|                                   | function)](api/languages/cpp_api. |
| type)](api/languages/cpp_api.html | html#_CPPv4NK5cudaq16quantum_plat |
| #_CPPv4N5cudaq15fermion_op_termE) | form14get_num_qubitsENSt6size_tE) |
| -   [cudaq::FermioniqBaseQPU (C++ | -   [cudaq::quantum_              |
|     cl                            | platform::get_remote_capabilities |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16FermioniqBaseQPUE) |     function)                     |
| -   [cudaq::get_state (C++        | ](api/languages/cpp_api.html#_CPP |
|                                   | v4NK5cudaq16quantum_platform23get |
|    function)](api/languages/cpp_a | _remote_capabilitiesENSt6size_tE) |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | -   [cudaq::qua                   |
| ateEDaRR13QuantumKernelDpRR4Args) | ntum_platform::get_runtime_target |
| -   [cudaq::gradient (C++         |     (C++                          |
|     class)](api/languages/cpp_    |     function)](api/languages/cp   |
| api.html#_CPPv4N5cudaq8gradientE) | p_api.html#_CPPv4NK5cudaq16quantu |
| -   [cudaq::gradient::clone (C++  | m_platform18get_runtime_targetEv) |
|     fun                           | -   [cuda                         |
| ction)](api/languages/cpp_api.htm | q::quantum_platform::getLogStream |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     (C++                          |
| -   [cudaq::gradient::compute     |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     function)](api/language       | quantum_platform12getLogStreamEv) |
| s/cpp_api.html#_CPPv4N5cudaq8grad | -   [cud                          |
| ient7computeERKNSt6vectorIdEERKNS | aq::quantum_platform::is_emulated |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|     [\[1\]](ap                    |                                   |
| i/languages/cpp_api.html#_CPPv4N5 |    function)](api/languages/cpp_a |
| cudaq8gradient7computeERKNSt6vect | pi.html#_CPPv4NK5cudaq16quantum_p |
| orIdEERNSt6vectorIdEERK7spin_opd) | latform11is_emulatedENSt6size_tE) |
| -   [cudaq::gradient::gradient    | -   [c                            |
|     (C++                          | udaq::quantum_platform::is_remote |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4I00EN5cu |     function)](api/languages/cp   |
| daq8gradient8gradientER7KernelT), | p_api.html#_CPPv4NK5cudaq16quantu |
|                                   | m_platform9is_remoteENSt6size_tE) |
|    [\[1\]](api/languages/cpp_api. | -   [cuda                         |
| html#_CPPv4I00EN5cudaq8gradient8g | q::quantum_platform::is_simulator |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\                         |                                   |
| ]](api/languages/cpp_api.html#_CP |   function)](api/languages/cpp_ap |
| Pv4I00EN5cudaq8gradient8gradientE | i.html#_CPPv4NK5cudaq16quantum_pl |
| RR13QuantumKernelRR10ArgsMapper), | atform12is_simulatorENSt6size_tE) |
|     [\[3                          | -   [c                            |
| \]](api/languages/cpp_api.html#_C | udaq::quantum_platform::launchVQE |
| PPv4N5cudaq8gradient8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](                   |
|     [\[                           | api/languages/cpp_api.html#_CPPv4 |
| 4\]](api/languages/cpp_api.html#_ | N5cudaq16quantum_platform9launchV |
| CPPv4N5cudaq8gradient8gradientEv) | QEEKNSt6stringEPKvPN5cudaq8gradie |
| -   [cudaq::gradient::setArgs     | ntERKN5cudaq7spin_opERN5cudaq9opt |
|     (C++                          | imizerEKiKNSt6size_tENSt6size_tE) |
|     fu                            | -   [cudaq:                       |
| nction)](api/languages/cpp_api.ht | :quantum_platform::list_platforms |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     (C++                          |
| tArgsEvR13QuantumKernelDpRR4Args) |     function)](api/languag        |
| -   [cudaq::gradient::setKernel   | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14list_platformsEv) |
|     function)](api/languages/c    | -                                 |
| pp_api.html#_CPPv4I0EN5cudaq8grad |    [cudaq::quantum_platform::name |
| ient9setKernelEvR13QuantumKernel) |     (C++                          |
| -   [cud                          |     function)](a                  |
| aq::gradients::central_difference | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | K5cudaq16quantum_platform4nameEv) |
|     class)](api/la                | -   [                             |
| nguages/cpp_api.html#_CPPv4N5cuda | cudaq::quantum_platform::num_qpus |
| q9gradients18central_differenceE) |     (C++                          |
| -   [cudaq::gra                   |     function)](api/l              |
| dients::central_difference::clone | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq16quantum_platform8num_qpusEv) |
|     function)](api/languages      | -   [cudaq::                      |
| /cpp_api.html#_CPPv4N5cudaq9gradi | quantum_platform::onRandomSeedSet |
| ents18central_difference5cloneEv) |     (C++                          |
| -   [cudaq::gradi                 |                                   |
| ents::central_difference::compute | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq16quantum_platf |
|     function)](                   | orm15onRandomSeedSetENSt6size_tE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq:                       |
| N5cudaq9gradients18central_differ | :quantum_platform::reset_exec_ctx |
| ence7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api/languag        |
|                                   | es/cpp_api.html#_CPPv4N5cudaq16qu |
|   [\[1\]](api/languages/cpp_api.h | antum_platform14reset_exec_ctxEv) |
| tml#_CPPv4N5cudaq9gradients18cent | -   [cud                          |
| ral_difference7computeERKNSt6vect | aq::quantum_platform::reset_noise |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradie                |     function)](api/languages/cpp_ |
| nts::central_difference::gradient | api.html#_CPPv4N5cudaq16quantum_p |
|     (C++                          | latform11reset_noiseENSt6size_tE) |
|     functio                       | -   [cudaq:                       |
| n)](api/languages/cpp_api.html#_C | :quantum_platform::resetLogStream |
| PPv4I00EN5cudaq9gradients18centra |     (C++                          |
| l_difference8gradientER7KernelT), |     function)](api/languag        |
|     [\[1\]](api/langua            | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ges/cpp_api.html#_CPPv4I00EN5cuda | antum_platform14resetLogStreamEv) |
| q9gradients18central_difference8g | -   [cuda                         |
| radientER7KernelTRR10ArgsMapper), | q::quantum_platform::set_exec_ctx |
|     [\[2\]](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4I00EN5cudaq9gradie |     funct                         |
| nts18central_difference8gradientE | ion)](api/languages/cpp_api.html# |
| RR13QuantumKernelRR10ArgsMapper), | _CPPv4N5cudaq16quantum_platform12 |
|     [\[3\]](api/languages/cpp     | set_exec_ctxEP16ExecutionContext) |
| _api.html#_CPPv4N5cudaq9gradients | -   [c                            |
| 18central_difference8gradientERRN | udaq::quantum_platform::set_noise |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[4\]](api/languages/cp      |     function                      |
| p_api.html#_CPPv4N5cudaq9gradient | )](api/languages/cpp_api.html#_CP |
| s18central_difference8gradientEv) | Pv4N5cudaq16quantum_platform9set_ |
| -   [cud                          | noiseEPK11noise_modelNSt6size_tE) |
| aq::gradients::forward_difference | -   [cuda                         |
|     (C++                          | q::quantum_platform::setLogStream |
|     class)](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q9gradients18forward_differenceE) |  function)](api/languages/cpp_api |
| -   [cudaq::gra                   | .html#_CPPv4N5cudaq16quantum_plat |
| dients::forward_difference::clone | form12setLogStreamERNSt7ostreamE) |
|     (C++                          | -   [cudaq::quantum_platfor       |
|     function)](api/languages      | m::supports_explicit_measurements |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18forward_difference5cloneEv) |     function)](api/l              |
| -   [cudaq::gradi                 | anguages/cpp_api.html#_CPPv4NK5cu |
| ents::forward_difference::compute | daq16quantum_platform30supports_e |
|     (C++                          | xplicit_measurementsENSt6size_tE) |
|     function)](                   | -   [cudaq::quantum_pla           |
| api/languages/cpp_api.html#_CPPv4 | tform::supports_task_distribution |
| N5cudaq9gradients18forward_differ |     (C++                          |
| ence7computeERKNSt6vectorIdEERKNS |     fu                            |
| t8functionIFdNSt6vectorIdEEEEEd), | nction)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4NK5cudaq16quantum_platfo |
|   [\[1\]](api/languages/cpp_api.h | rm26supports_task_distributionEv) |
| tml#_CPPv4N5cudaq9gradients18forw | -   [cudaq::quantum               |
| ard_difference7computeERKNSt6vect | _platform::with_execution_context |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradie                |     function)                     |
| nts::forward_difference::gradient | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4I0DpEN5cudaq16quantum_platform2 |
|     functio                       | 2with_execution_contextEDaR16Exec |
| n)](api/languages/cpp_api.html#_C | utionContextRR8CallableDpRR4Args) |
| PPv4I00EN5cudaq9gradients18forwar | -   [cudaq::QuantumTask (C++      |
| d_difference8gradientER7KernelT), |     type)](api/languages/cpp_api. |
|     [\[1\]](api/langua            | html#_CPPv4N5cudaq11QuantumTaskE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::qubit (C++            |
| q9gradients18forward_difference8g |     type)](api/languages/c        |
| radientER7KernelTRR10ArgsMapper), | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::QubitConnectivity     |
| api.html#_CPPv4I00EN5cudaq9gradie |     (C++                          |
| nts18forward_difference8gradientE |     ty                            |
| RR13QuantumKernelRR10ArgsMapper), | pe)](api/languages/cpp_api.html#_ |
|     [\[3\]](api/languages/cpp     | CPPv4N5cudaq17QubitConnectivityE) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::QubitEdge (C++        |
| 18forward_difference8gradientERRN |     type)](api/languages/cpp_a    |
| St8functionIFvNSt6vectorIdEEEEE), | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qudit (C++            |
| p_api.html#_CPPv4N5cudaq9gradient |     clas                          |
| s18forward_difference8gradientEv) | s)](api/languages/cpp_api.html#_C |
| -   [                             | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| cudaq::gradients::parameter_shift | -   [cudaq::qudit::qudit (C++     |
|     (C++                          |                                   |
|     class)](api                   | function)](api/languages/cpp_api. |
| /languages/cpp_api.html#_CPPv4N5c | html#_CPPv4N5cudaq5qudit5quditEv) |
| udaq9gradients15parameter_shiftE) | -   [cudaq::qvector (C++          |
| -   [cudaq::                      |     class)                        |
| gradients::parameter_shift::clone | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     function)](api/langua         | -   [cudaq::qvector::back (C++    |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     function)](a                  |
| adients15parameter_shift5cloneEv) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::gr                    | 5cudaq7qvector4backENSt6size_tE), |
| adients::parameter_shift::compute |                                   |
|     (C++                          |   [\[1\]](api/languages/cpp_api.h |
|     function                      | tml#_CPPv4N5cudaq7qvector4backEv) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qvector::begin (C++   |
| Pv4N5cudaq9gradients15parameter_s |     fu                            |
| hift7computeERKNSt6vectorIdEERKNS | nction)](api/languages/cpp_api.ht |
| t8functionIFdNSt6vectorIdEEEEEd), | ml#_CPPv4N5cudaq7qvector5beginEv) |
|     [\[1\]](api/languages/cpp_ap  | -   [cudaq::qvector::clear (C++   |
| i.html#_CPPv4N5cudaq9gradients15p |     fu                            |
| arameter_shift7computeERKNSt6vect | nction)](api/languages/cpp_api.ht |
| orIdEERNSt6vectorIdEERK7spin_opd) | ml#_CPPv4N5cudaq7qvector5clearEv) |
| -   [cudaq::gra                   | -   [cudaq::qvector::end (C++     |
| dients::parameter_shift::gradient |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     func                          | html#_CPPv4N5cudaq7qvector3endEv) |
| tion)](api/languages/cpp_api.html | -   [cudaq::qvector::front (C++   |
| #_CPPv4I00EN5cudaq9gradients15par |     function)](ap                 |
| ameter_shift8gradientER7KernelT), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[1\]](api/lan               | cudaq7qvector5frontENSt6size_tE), |
| guages/cpp_api.html#_CPPv4I00EN5c |                                   |
| udaq9gradients15parameter_shift8g |  [\[1\]](api/languages/cpp_api.ht |
| radientER7KernelTRR10ArgsMapper), | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     [\[2\]](api/languages/c       | -   [cudaq::qvector::operator=    |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     (C++                          |
| dients15parameter_shift8gradientE |     functio                       |
| RR13QuantumKernelRR10ArgsMapper), | n)](api/languages/cpp_api.html#_C |
|     [\[3\]](api/languages/        | PPv4N5cudaq7qvectoraSERK7qvector) |
| cpp_api.html#_CPPv4N5cudaq9gradie | -   [cudaq::qvector::operator\[\] |
| nts15parameter_shift8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)                     |
|     [\[4\]](api/languages         | ](api/languages/cpp_api.html#_CPP |
| /cpp_api.html#_CPPv4N5cudaq9gradi | v4N5cudaq7qvectorixEKNSt6size_tE) |
| ents15parameter_shift8gradientEv) | -   [cudaq::qvector::qvector (C++ |
| -   [cudaq::kernel_builder (C++   |     function)](api/               |
|     clas                          | languages/cpp_api.html#_CPPv4N5cu |
| s)](api/languages/cpp_api.html#_C | daq7qvector7qvectorENSt6size_tE), |
| PPv4IDpEN5cudaq14kernel_builderE) |     [\[1\]](a                     |
| -   [c                            | pi/languages/cpp_api.html#_CPPv4N |
| udaq::kernel_builder::constantVal | 5cudaq7qvector7qvectorERK5state), |
|     (C++                          |     [\[2\]](api                   |
|     function)](api/la             | /languages/cpp_api.html#_CPPv4N5c |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq7qvector7qvectorERK7qvector), |
| q14kernel_builder11constantValEd) |     [\[3\]](api/languages/cpp     |
| -   [cu                           | _api.html#_CPPv4N5cudaq7qvector7q |
| daq::kernel_builder::getArguments | vectorERKNSt6vectorI7complexEEb), |
|     (C++                          |     [\[4\]](ap                    |
|     function)](api/lan            | i/languages/cpp_api.html#_CPPv4N5 |
| guages/cpp_api.html#_CPPv4N5cudaq | cudaq7qvector7qvectorERR7qvector) |
| 14kernel_builder12getArgumentsEv) | -   [cudaq::qvector::size (C++    |
| -   [cu                           |     fu                            |
| daq::kernel_builder::getNumParams | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     function)](api/lan            | -   [cudaq::qvector::slice (C++   |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/language       |
| 14kernel_builder12getNumParamsEv) | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| -   [c                            | tor5sliceENSt6size_tENSt6size_tE) |
| udaq::kernel_builder::isArgStdVec | -   [cudaq::qvector::value_type   |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     typ                           |
| p_api.html#_CPPv4N5cudaq14kernel_ | e)](api/languages/cpp_api.html#_C |
| builder11isArgStdVecENSt6size_tE) | PPv4N5cudaq7qvector10value_typeE) |
| -   [cuda                         | -   [cudaq::qview (C++            |
| q::kernel_builder::kernel_builder |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     function)](api/languages/cpp_ | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| api.html#_CPPv4N5cudaq14kernel_bu | -   [cudaq::qview::back (C++      |
| ilder14kernel_builderERNSt6vector |     function)                     |
| IN7details17KernelBuilderTypeEEE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kernel_builder::name  | v4N5cudaq5qview4backENSt6size_tE) |
|     (C++                          | -   [cudaq::qview::begin (C++     |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP | function)](api/languages/cpp_api. |
| v4N5cudaq14kernel_builder4nameEv) | html#_CPPv4N5cudaq5qview5beginEv) |
| -                                 | -   [cudaq::qview::end (C++       |
|    [cudaq::kernel_builder::qalloc |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)](api/language       | i.html#_CPPv4N5cudaq5qview3endEv) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::qview::front (C++     |
| nel_builder6qallocE10QuakeValue), |     function)](                   |
|     [\[1\]](api/language          | api/languages/cpp_api.html#_CPPv4 |
| s/cpp_api.html#_CPPv4N5cudaq14ker | N5cudaq5qview5frontENSt6size_tE), |
| nel_builder6qallocEKNSt6size_tE), |                                   |
|     [\[2                          |    [\[1\]](api/languages/cpp_api. |
| \]](api/languages/cpp_api.html#_C | html#_CPPv4N5cudaq5qview5frontEv) |
| PPv4N5cudaq14kernel_builder6qallo | -   [cudaq::qview::operator\[\]   |
| cERNSt6vectorINSt7complexIdEEEE), |     (C++                          |
|     [\[3\]](                      |     functio                       |
| api/languages/cpp_api.html#_CPPv4 | n)](api/languages/cpp_api.html#_C |
| N5cudaq14kernel_builder6qallocEv) | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| -   [cudaq::kernel_builder::swap  | -   [cudaq::qview::qview (C++     |
|     (C++                          |     functio                       |
|     function)](api/language       | n)](api/languages/cpp_api.html#_C |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | PPv4I0EN5cudaq5qview5qviewERR1R), |
| 4kernel_builder4swapEvRK10QuakeVa |     [\[1                          |
| lueRK10QuakeValueRK10QuakeValue), | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq5qview5qviewERK5qview) |
| [\[1\]](api/languages/cpp_api.htm | -   [cudaq::qview::size (C++      |
| l#_CPPv4I00EN5cudaq14kernel_build |                                   |
| er4swapEvRKNSt6vectorI10QuakeValu | function)](api/languages/cpp_api. |
| eEERK10QuakeValueRK10QuakeValue), | html#_CPPv4NK5cudaq5qview4sizeEv) |
|                                   | -   [cudaq::qview::slice (C++     |
| [\[2\]](api/languages/cpp_api.htm |     function)](api/langua         |
| l#_CPPv4N5cudaq14kernel_builder4s | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| wapERK10QuakeValueRK10QuakeValue) | iew5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::KernelExecutionTask   | -   [cudaq::qview::value_type     |
|     (C++                          |     (C++                          |
|     type                          |     t                             |
| )](api/languages/cpp_api.html#_CP | ype)](api/languages/cpp_api.html# |
| Pv4N5cudaq19KernelExecutionTaskE) | _CPPv4N5cudaq5qview10value_typeE) |
| -   [cudaq::KernelThunkResultType | -   [cudaq::range (C++            |
|     (C++                          |     fun                           |
|     struct)]                      | ction)](api/languages/cpp_api.htm |
| (api/languages/cpp_api.html#_CPPv | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| 4N5cudaq21KernelThunkResultTypeE) | orI11ElementTypeEE11ElementType), |
| -   [cudaq::KernelThunkType (C++  |     [\[1\]](api/languages/cpp_    |
|                                   | api.html#_CPPv4I0EN5cudaq5rangeEN |
| type)](api/languages/cpp_api.html | St6vectorI11ElementTypeEE11Elemen |
| #_CPPv4N5cudaq15KernelThunkTypeE) | tType11ElementType11ElementType), |
| -   [cudaq::kraus_channel (C++    |     [                             |
|                                   | \[2\]](api/languages/cpp_api.html |
|  class)](api/languages/cpp_api.ht | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| ml#_CPPv4N5cudaq13kraus_channelE) | -   [cudaq::real (C++             |
| -   [cudaq::kraus_channel::empty  |     type)](api/languages/         |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq4realE) |
|     function)]                    | -   [cudaq::registry (C++         |
| (api/languages/cpp_api.html#_CPPv |     type)](api/languages/cpp_     |
| 4NK5cudaq13kraus_channel5emptyEv) | api.html#_CPPv4N5cudaq8registryE) |
| -   [cudaq::kraus_c               | -                                 |
| hannel::generateUnitaryParameters |  [cudaq::registry::RegisteredType |
|     (C++                          |     (C++                          |
|                                   |     class)](api/                  |
|    function)](api/languages/cpp_a | languages/cpp_api.html#_CPPv4I0EN |
| pi.html#_CPPv4N5cudaq13kraus_chan | 5cudaq8registry14RegisteredTypeE) |
| nel25generateUnitaryParametersEv) | -   [cudaq::RemoteCapabilities    |
| -                                 |     (C++                          |
|    [cudaq::kraus_channel::get_ops |     struc                         |
|     (C++                          | t)](api/languages/cpp_api.html#_C |
|     function)](a                  | PPv4N5cudaq18RemoteCapabilitiesE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::Remo                  |
| K5cudaq13kraus_channel7get_opsEv) | teCapabilities::isRemoteSimulator |
| -   [cudaq::                      |     (C++                          |
| kraus_channel::is_unitary_mixture |     member)](api/languages/c      |
|     (C++                          | pp_api.html#_CPPv4N5cudaq18Remote |
|     function)](api/languages      | Capabilities17isRemoteSimulatorE) |
| /cpp_api.html#_CPPv4NK5cudaq13kra | -   [cudaq::Remot                 |
| us_channel18is_unitary_mixtureEv) | eCapabilities::RemoteCapabilities |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::kraus_channel |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4N5cudaq18RemoteCa |
|     function)](api/lang           | pabilities18RemoteCapabilitiesEb) |
| uages/cpp_api.html#_CPPv4IDpEN5cu | -   [cudaq:                       |
| daq13kraus_channel13kraus_channel | :RemoteCapabilities::stateOverlap |
| EDpRRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     member)](api/langua           |
|  [\[1\]](api/languages/cpp_api.ht | ges/cpp_api.html#_CPPv4N5cudaq18R |
| ml#_CPPv4N5cudaq13kraus_channel13 | emoteCapabilities12stateOverlapE) |
| kraus_channelERK13kraus_channel), | -                                 |
|     [\[2\]                        |   [cudaq::RemoteCapabilities::vqe |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq13kraus_channel13kraus_c |     member)](                     |
| hannelERKNSt6vectorI8kraus_opEE), | api/languages/cpp_api.html#_CPPv4 |
|     [\[3\]                        | N5cudaq18RemoteCapabilities3vqeE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::RemoteSimulationState |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERRNSt6vectorI8kraus_opEE), |     class)]                       |
|     [\[4\]](api/lan               | (api/languages/cpp_api.html#_CPPv |
| guages/cpp_api.html#_CPPv4N5cudaq | 4N5cudaq21RemoteSimulationStateE) |
| 13kraus_channel13kraus_channelEv) | -   [cudaq::Resources (C++        |
| -                                 |     class)](api/languages/cpp_a   |
| [cudaq::kraus_channel::noise_type | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     (C++                          | -   [cudaq::run (C++              |
|     member)](api                  |     function)]                    |
| /languages/cpp_api.html#_CPPv4N5c | (api/languages/cpp_api.html#_CPPv |
| udaq13kraus_channel10noise_typeE) | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| -                                 | 5invoke_result_tINSt7decay_tI13Qu |
|   [cudaq::kraus_channel::op_names | antumKernelEEDpNSt7decay_tI4ARGSE |
|     (C++                          | EEEEENSt6size_tERN5cudaq11noise_m |
|     member)](                     | odelERR13QuantumKernelDpRR4ARGS), |
| api/languages/cpp_api.html#_CPPv4 |     [\[1\]](api/langu             |
| N5cudaq13kraus_channel8op_namesE) | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| -                                 | daq3runENSt6vectorINSt15invoke_re |
|  [cudaq::kraus_channel::operator= | sult_tINSt7decay_tI13QuantumKerne |
|     (C++                          | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
|     function)](api/langua         | ize_tERR13QuantumKernelDpRR4ARGS) |
| ges/cpp_api.html#_CPPv4N5cudaq13k | -   [cudaq::run_async (C++        |
| raus_channelaSERK13kraus_channel) |     functio                       |
| -   [c                            | n)](api/languages/cpp_api.html#_C |
| udaq::kraus_channel::operator\[\] | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     (C++                          | tureINSt6vectorINSt15invoke_resul |
|     function)](api/l              | t_tINSt7decay_tI13QuantumKernelEE |
| anguages/cpp_api.html#_CPPv4N5cud | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| aq13kraus_channelixEKNSt6size_tE) | ze_tENSt6size_tERN5cudaq11noise_m |
| -                                 | odelERR13QuantumKernelDpRR4ARGS), |
| [cudaq::kraus_channel::parameters |     [\[1\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4I0DpEN |
|     member)](api                  | 5cudaq9run_asyncENSt6futureINSt6v |
| /languages/cpp_api.html#_CPPv4N5c | ectorINSt15invoke_result_tINSt7de |
| udaq13kraus_channel10parametersE) | cay_tI13QuantumKernelEEDpNSt7deca |
| -   [cudaq::krau                  | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| s_channel::populateDefaultOpNames | ize_tERR13QuantumKernelDpRR4ARGS) |
|     (C++                          | -   [cudaq::RuntimeTarget (C++    |
|     function)](api/languages/cp   |                                   |
| p_api.html#_CPPv4N5cudaq13kraus_c | struct)](api/languages/cpp_api.ht |
| hannel22populateDefaultOpNamesEv) | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| -   [cu                           | -   [cudaq::sample (C++           |
| daq::kraus_channel::probabilities |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     member)](api/la               | mpleE13sample_resultRK14sample_op |
| nguages/cpp_api.html#_CPPv4N5cuda | tionsRR13QuantumKernelDpRR4Args), |
| q13kraus_channel13probabilitiesE) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|  [cudaq::kraus_channel::push_back | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     (C++                          | esultRR13QuantumKernelDpRR4Args), |
|     function)](api                |     [\                            |
| /languages/cpp_api.html#_CPPv4N5c | [2\]](api/languages/cpp_api.html# |
| udaq13kraus_channel9push_backE8kr | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| aus_opNSt8optionalINSt6stringEEE) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [cudaq::kraus_channel::size   | -   [cudaq::sample_options (C++   |
|     (C++                          |     s                             |
|     function)                     | truct)](api/languages/cpp_api.htm |
| ](api/languages/cpp_api.html#_CPP | l#_CPPv4N5cudaq14sample_optionsE) |
| v4NK5cudaq13kraus_channel4sizeEv) | -   [cudaq::sample_result (C++    |
| -   [                             |                                   |
| cudaq::kraus_channel::unitary_ops |  class)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq13sample_resultE) |
|     member)](api/                 | -   [cudaq::sample_result::append |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq13kraus_channel11unitary_opsE) |     function)](api/languages/cpp_ |
| -   [cudaq::kraus_op (C++         | api.html#_CPPv4N5cudaq13sample_re |
|     struct)](api/languages/cpp_   | sult6appendERK15ExecutionResultb) |
| api.html#_CPPv4N5cudaq8kraus_opE) | -   [cudaq::sample_result::begin  |
| -   [cudaq::kraus_op::adjoint     |     (C++                          |
|     (C++                          |     function)]                    |
|     functi                        | (api/languages/cpp_api.html#_CPPv |
| on)](api/languages/cpp_api.html#_ | 4N5cudaq13sample_result5beginEv), |
| CPPv4NK5cudaq8kraus_op7adjointEv) |     [\[1\]]                       |
| -   [cudaq::kraus_op::data (C++   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq13sample_result5beginEv) |
|  member)](api/languages/cpp_api.h | -   [cudaq::sample_result::cbegin |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     (C++                          |
| -   [cudaq::kraus_op::kraus_op    |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     func                          | NK5cudaq13sample_result6cbeginEv) |
| tion)](api/languages/cpp_api.html | -   [cudaq::sample_result::cend   |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |     (C++                          |
| opERRNSt16initializer_listI1TEE), |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
|  [\[1\]](api/languages/cpp_api.ht | v4NK5cudaq13sample_result4cendEv) |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | -   [cudaq::sample_result::clear  |
| pENSt6vectorIN5cudaq7complexEEE), |     (C++                          |
|     [\[2\]](api/l                 |     function)                     |
| anguages/cpp_api.html#_CPPv4N5cud | ](api/languages/cpp_api.html#_CPP |
| aq8kraus_op8kraus_opERK8kraus_op) | v4N5cudaq13sample_result5clearEv) |
| -   [cudaq::kraus_op::nCols (C++  | -   [cudaq::sample_result::count  |
|                                   |     (C++                          |
| member)](api/languages/cpp_api.ht |     function)](                   |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::kraus_op::nRows (C++  | NK5cudaq13sample_result5countENSt |
|                                   | 11string_viewEKNSt11string_viewE) |
| member)](api/languages/cpp_api.ht | -   [                             |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | cudaq::sample_result::deserialize |
| -   [cudaq::kraus_op::operator=   |     (C++                          |
|     (C++                          |     functio                       |
|     function)                     | n)](api/languages/cpp_api.html#_C |
| ](api/languages/cpp_api.html#_CPP | PPv4N5cudaq13sample_result11deser |
| v4N5cudaq8kraus_opaSERK8kraus_op) | ializeERNSt6vectorINSt6size_tEEE) |
| -   [cudaq::kraus_op::precision   | -   [cudaq::sample_result::dump   |
|     (C++                          |     (C++                          |
|     memb                          |     function)](api/languag        |
| er)](api/languages/cpp_api.html#_ | es/cpp_api.html#_CPPv4NK5cudaq13s |
| CPPv4N5cudaq8kraus_op9precisionE) | ample_result4dumpERNSt7ostreamE), |
| -   [cudaq::matrix_callback (C++  |     [\[1\]                        |
|     c                             | ](api/languages/cpp_api.html#_CPP |
| lass)](api/languages/cpp_api.html | v4NK5cudaq13sample_result4dumpEv) |
| #_CPPv4N5cudaq15matrix_callbackE) | -   [cudaq::sample_result::end    |
| -   [cudaq::matrix_handler (C++   |     (C++                          |
|                                   |     function                      |
| class)](api/languages/cpp_api.htm | )](api/languages/cpp_api.html#_CP |
| l#_CPPv4N5cudaq14matrix_handlerE) | Pv4N5cudaq13sample_result3endEv), |
| -   [cudaq::mat                   |     [\[1\                         |
| rix_handler::commutation_behavior | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NK5cudaq13sample_result3endEv) |
|     struct)](api/languages/       | -   [                             |
| cpp_api.html#_CPPv4N5cudaq14matri | cudaq::sample_result::expectation |
| x_handler20commutation_behaviorE) |     (C++                          |
| -                                 |     f                             |
|    [cudaq::matrix_handler::define | unction)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4NK5cudaq13sample_result |
|     function)](a                  | 11expectationEKNSt11string_viewE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [c                            |
| 5cudaq14matrix_handler6defineENSt | udaq::sample_result::get_marginal |
| 6stringENSt6vectorINSt7int64_tEEE |     (C++                          |
| RR15matrix_callbackRKNSt13unorder |     function)](api/languages/cpp_ |
| ed_mapINSt6stringENSt6stringEEE), | api.html#_CPPv4NK5cudaq13sample_r |
|                                   | esult12get_marginalERKNSt6vectorI |
| [\[1\]](api/languages/cpp_api.htm | NSt6size_tEEEKNSt11string_viewE), |
| l#_CPPv4N5cudaq14matrix_handler6d |     [\[1\]](api/languages/cpp_    |
| efineENSt6stringENSt6vectorINSt7i | api.html#_CPPv4NK5cudaq13sample_r |
| nt64_tEEERR15matrix_callbackRR20d | esult12get_marginalERRKNSt6vector |
| iag_matrix_callbackRKNSt13unorder | INSt6size_tEEEKNSt11string_viewE) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cuda                         |
|     [\[2\]](                      | q::sample_result::get_total_shots |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14matrix_handler6defineENS |     function)](api/langua         |
| t6stringENSt6vectorINSt7int64_tEE | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| ERR15matrix_callbackRRNSt13unorde | sample_result15get_total_shotsEv) |
| red_mapINSt6stringENSt6stringEEE) | -   [cuda                         |
| -                                 | q::sample_result::has_even_parity |
|   [cudaq::matrix_handler::degrees |     (C++                          |
|     (C++                          |     fun                           |
|     function)](ap                 | ction)](api/languages/cpp_api.htm |
| i/languages/cpp_api.html#_CPPv4NK | l#_CPPv4N5cudaq13sample_result15h |
| 5cudaq14matrix_handler7degreesEv) | as_even_parityENSt11string_viewE) |
| -                                 | -   [cuda                         |
|  [cudaq::matrix_handler::displace | q::sample_result::has_expectation |
|     (C++                          |     (C++                          |
|     function)](api/language       |     funct                         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ion)](api/languages/cpp_api.html# |
| rix_handler8displaceENSt6size_tE) | _CPPv4NK5cudaq13sample_result15ha |
| -   [cudaq::matrix                | s_expectationEKNSt11string_viewE) |
| _handler::get_expected_dimensions | -   [cu                           |
|     (C++                          | daq::sample_result::most_probable |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     fun                           |
| pi.html#_CPPv4NK5cudaq14matrix_ha | ction)](api/languages/cpp_api.htm |
| ndler23get_expected_dimensionsEv) | l#_CPPv4NK5cudaq13sample_result13 |
| -   [cudaq::matrix_ha             | most_probableEKNSt11string_viewE) |
| ndler::get_parameter_descriptions | -                                 |
|     (C++                          | [cudaq::sample_result::operator+= |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     function)](api/langua         |
| html#_CPPv4NK5cudaq14matrix_handl | ges/cpp_api.html#_CPPv4N5cudaq13s |
| er26get_parameter_descriptionsEv) | ample_resultpLERK13sample_result) |
| -   [c                            | -                                 |
| udaq::matrix_handler::instantiate |  [cudaq::sample_result::operator= |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](api/langua         |
| pi/languages/cpp_api.html#_CPPv4N | ges/cpp_api.html#_CPPv4N5cudaq13s |
| 5cudaq14matrix_handler11instantia | ample_resultaSERR13sample_result) |
| teENSt6stringERKNSt6vectorINSt6si | -                                 |
| ze_tEEERK20commutation_behavior), | [cudaq::sample_result::operator== |
|     [\[1\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languag        |
| N5cudaq14matrix_handler11instanti | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ateENSt6stringERRNSt6vectorINSt6s | ample_resulteqERK13sample_result) |
| ize_tEEERK20commutation_behavior) | -   [                             |
| -   [cuda                         | cudaq::sample_result::probability |
| q::matrix_handler::matrix_handler |     (C++                          |
|     (C++                          |     function)](api/lan            |
|     function)](api/languag        | guages/cpp_api.html#_CPPv4NK5cuda |
| es/cpp_api.html#_CPPv4I0_NSt11ena | q13sample_result11probabilityENSt |
| ble_if_tINSt12is_base_of_vI16oper | 11string_viewEKNSt11string_viewE) |
| ator_handler1TEEbEEEN5cudaq14matr | -   [cud                          |
| ix_handler14matrix_handlerERK1T), | aq::sample_result::register_names |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4I0 |     function)](api/langu          |
| _NSt11enable_if_tINSt12is_base_of | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| _vI16operator_handler1TEEbEEEN5cu | 3sample_result14register_namesEv) |
| daq14matrix_handler14matrix_handl | -                                 |
| erERK1TRK20commutation_behavior), |    [cudaq::sample_result::reorder |
|     [\[2\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4N5cudaq14matrix_hand |     function)](api/langua         |
| ler14matrix_handlerENSt6size_tE), | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     [\[3\]](api/                  | ample_result7reorderERKNSt6vector |
| languages/cpp_api.html#_CPPv4N5cu | INSt6size_tEEEKNSt11string_viewE) |
| daq14matrix_handler14matrix_handl | -   [cu                           |
| erENSt6stringERKNSt6vectorINSt6si | daq::sample_result::sample_result |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[4\]](api/                  |     func                          |
| languages/cpp_api.html#_CPPv4N5cu | tion)](api/languages/cpp_api.html |
| daq14matrix_handler14matrix_handl | #_CPPv4N5cudaq13sample_result13sa |
| erENSt6stringERRNSt6vectorINSt6si | mple_resultERK15ExecutionResult), |
| ze_tEEERK20commutation_behavior), |     [\[1\]](api/la                |
|     [\                            | nguages/cpp_api.html#_CPPv4N5cuda |
| [5\]](api/languages/cpp_api.html# | q13sample_result13sample_resultER |
| _CPPv4N5cudaq14matrix_handler14ma | KNSt6vectorI15ExecutionResultEE), |
| trix_handlerERK14matrix_handler), |                                   |
|     [                             |  [\[2\]](api/languages/cpp_api.ht |
| \[6\]](api/languages/cpp_api.html | ml#_CPPv4N5cudaq13sample_result13 |
| #_CPPv4N5cudaq14matrix_handler14m | sample_resultERR13sample_result), |
| atrix_handlerERR14matrix_handler) |     [                             |
| -                                 | \[3\]](api/languages/cpp_api.html |
|  [cudaq::matrix_handler::momentum | #_CPPv4N5cudaq13sample_result13sa |
|     (C++                          | mple_resultERR15ExecutionResult), |
|     function)](api/language       |     [\[4\]](api/lan               |
| s/cpp_api.html#_CPPv4N5cudaq14mat | guages/cpp_api.html#_CPPv4N5cudaq |
| rix_handler8momentumENSt6size_tE) | 13sample_result13sample_resultEdR |
| -                                 | KNSt6vectorI15ExecutionResultEE), |
|    [cudaq::matrix_handler::number |     [\[5\]](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     function)](api/langua         | 13sample_result13sample_resultEv) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -                                 |
| atrix_handler6numberENSt6size_tE) |  [cudaq::sample_result::serialize |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::operator= |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4NK5 |
|     fun                           | cudaq13sample_result9serializeEv) |
| ction)](api/languages/cpp_api.htm | -   [cudaq::sample_result::size   |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     (C++                          |
| NSt7is_sameI1T14matrix_handlerE5v |     function)](api/languages/c    |
| alueENSt12is_base_of_vI16operator | pp_api.html#_CPPv4NK5cudaq13sampl |
| _handler1TEEEbEEEN5cudaq14matrix_ | e_result4sizeEKNSt11string_viewE) |
| handleraSER14matrix_handlerRK1T), | -   [cudaq::sample_result::to_map |
|     [\[1\]](api/languages         |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq14matr |     function)](api/languages/cpp  |
| ix_handleraSERK14matrix_handler), | _api.html#_CPPv4NK5cudaq13sample_ |
|     [\[2\]](api/language          | result6to_mapEKNSt11string_viewE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cuda                         |
| rix_handleraSERR14matrix_handler) | q::sample_result::\~sample_result |
| -   [                             |     (C++                          |
| cudaq::matrix_handler::operator== |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/languages      | _CPPv4N5cudaq13sample_resultD0Ev) |
| /cpp_api.html#_CPPv4NK5cudaq14mat | -   [cudaq::scalar_callback (C++  |
| rix_handlereqERK14matrix_handler) |     c                             |
| -                                 | lass)](api/languages/cpp_api.html |
|    [cudaq::matrix_handler::parity | #_CPPv4N5cudaq15scalar_callbackE) |
|     (C++                          | -   [c                            |
|     function)](api/langua         | udaq::scalar_callback::operator() |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6parityENSt6size_tE) |     function)](api/language       |
| -                                 | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|  [cudaq::matrix_handler::position | alar_callbackclERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|     function)](api/language       | -   [                             |
| s/cpp_api.html#_CPPv4N5cudaq14mat | cudaq::scalar_callback::operator= |
| rix_handler8positionENSt6size_tE) |     (C++                          |
| -   [cudaq::                      |     function)](api/languages/c    |
| matrix_handler::remove_definition | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _callbackaSERK15scalar_callback), |
|     fu                            |     [\[1\]](api/languages/        |
| nction)](api/languages/cpp_api.ht | cpp_api.html#_CPPv4N5cudaq15scala |
| ml#_CPPv4N5cudaq14matrix_handler1 | r_callbackaSERR15scalar_callback) |
| 7remove_definitionERKNSt6stringE) | -   [cudaq:                       |
| -                                 | :scalar_callback::scalar_callback |
|   [cudaq::matrix_handler::squeeze |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/languag        | es/cpp_api.html#_CPPv4I0_NSt11ena |
| es/cpp_api.html#_CPPv4N5cudaq14ma | ble_if_tINSt16is_invocable_r_vINS |
| trix_handler7squeezeENSt6size_tE) | t7complexIdEE8CallableRKNSt13unor |
| -   [cudaq::m                     | dered_mapINSt6stringENSt7complexI |
| atrix_handler::to_diagonal_matrix | dEEEEEEbEEEN5cudaq15scalar_callba |
|     (C++                          | ck15scalar_callbackERR8Callable), |
|     function)](api/lang           |     [\[1\                         |
| uages/cpp_api.html#_CPPv4NK5cudaq | ]](api/languages/cpp_api.html#_CP |
| 14matrix_handler18to_diagonal_mat | Pv4N5cudaq15scalar_callback15scal |
| rixERNSt13unordered_mapINSt6size_ | ar_callbackERK15scalar_callback), |
| tENSt7int64_tEEERKNSt13unordered_ |     [\[2                          |
| mapINSt6stringENSt7complexIdEEEE) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq15scalar_callback15sca |
| [cudaq::matrix_handler::to_matrix | lar_callbackERR15scalar_callback) |
|     (C++                          | -   [cudaq::scalar_operator (C++  |
|     function)                     |     c                             |
| ](api/languages/cpp_api.html#_CPP | lass)](api/languages/cpp_api.html |
| v4NK5cudaq14matrix_handler9to_mat | #_CPPv4N5cudaq15scalar_operatorE) |
| rixERNSt13unordered_mapINSt6size_ | -                                 |
| tENSt7int64_tEEERKNSt13unordered_ | [cudaq::scalar_operator::evaluate |
| mapINSt6stringENSt7complexIdEEEE) |     (C++                          |
| -                                 |                                   |
| [cudaq::matrix_handler::to_string |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq15scalar_op |
|     function)](api/               | erator8evaluateERKNSt13unordered_ |
| languages/cpp_api.html#_CPPv4NK5c | mapINSt6stringENSt7complexIdEEEE) |
| udaq14matrix_handler9to_stringEb) | -   [cudaq::scalar_ope            |
| -                                 | rator::get_parameter_descriptions |
| [cudaq::matrix_handler::unique_id |     (C++                          |
|     (C++                          |     f                             |
|     function)](api/               | unction)](api/languages/cpp_api.h |
| languages/cpp_api.html#_CPPv4NK5c | tml#_CPPv4NK5cudaq15scalar_operat |
| udaq14matrix_handler9unique_idEv) | or26get_parameter_descriptionsEv) |
| -   [cudaq:                       | -   [cu                           |
| :matrix_handler::\~matrix_handler | daq::scalar_operator::is_constant |
|     (C++                          |     (C++                          |
|     functi                        |     function)](api/lang           |
| on)](api/languages/cpp_api.html#_ | uages/cpp_api.html#_CPPv4NK5cudaq |
| CPPv4N5cudaq14matrix_handlerD0Ev) | 15scalar_operator11is_constantEv) |
| -   [cudaq::matrix_op (C++        | -   [c                            |
|     type)](api/languages/cpp_a    | udaq::scalar_operator::operator\* |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     (C++                          |
| -   [cudaq::matrix_op_term (C++   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|  type)](api/languages/cpp_api.htm | Pv4N5cudaq15scalar_operatormlENSt |
| l#_CPPv4N5cudaq14matrix_op_termE) | 7complexIdEERK15scalar_operator), |
| -                                 |     [\[1\                         |
|    [cudaq::mdiag_operator_handler | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormlENSt |
|     class)](                      | 7complexIdEERR15scalar_operator), |
| api/languages/cpp_api.html#_CPPv4 |     [\[2\]](api/languages/cp      |
| N5cudaq22mdiag_operator_handlerE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::mpi (C++              | operatormlEdRK15scalar_operator), |
|     type)](api/languages          |     [\[3\]](api/languages/cp      |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::mpi::all_gather (C++  | operatormlEdRR15scalar_operator), |
|     fu                            |     [\[4\]](api/languages         |
| nction)](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | alar_operatormlENSt7complexIdEE), |
| RNSt6vectorIdEERKNSt6vectorIdEE), |     [\[5\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4NKR5cudaq15scalar |
|   [\[1\]](api/languages/cpp_api.h | _operatormlERK15scalar_operator), |
| tml#_CPPv4N5cudaq3mpi10all_gather |     [\[6\]]                       |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::mpi::all_reduce (C++  | 4NKR5cudaq15scalar_operatormlEd), |
|                                   |     [\[7\]](api/language          |
|  function)](api/languages/cpp_api | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | alar_operatormlENSt7complexIdEE), |
| reduceE1TRK1TRK14BinaryFunction), |     [\[8\]](api/languages/cp      |
|     [\[1\]](api/langu             | p_api.html#_CPPv4NO5cudaq15scalar |
| ages/cpp_api.html#_CPPv4I00EN5cud | _operatormlERK15scalar_operator), |
| aq3mpi10all_reduceE1TRK1TRK4Func) |     [\[9\                         |
| -   [cudaq::mpi::broadcast (C++   | ]](api/languages/cpp_api.html#_CP |
|     function)](api/               | Pv4NO5cudaq15scalar_operatormlEd) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cu                           |
| daq3mpi9broadcastERNSt6stringEi), | daq::scalar_operator::operator\*= |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languag        |
| q3mpi9broadcastERNSt6vectorIdEEi) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::mpi::finalize (C++    | alar_operatormLENSt7complexIdEE), |
|     f                             |     [\[1\]](api/languages/c       |
| unction)](api/languages/cpp_api.h | pp_api.html#_CPPv4N5cudaq15scalar |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | _operatormLERK15scalar_operator), |
| -   [cudaq::mpi::initialize (C++  |     [\[2                          |
|     function                      | \]](api/languages/cpp_api.html#_C |
| )](api/languages/cpp_api.html#_CP | PPv4N5cudaq15scalar_operatormLEd) |
| Pv4N5cudaq3mpi10initializeEiPPc), | -   [                             |
|     [                             | cudaq::scalar_operator::operator+ |
| \[1\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq3mpi10initializeEv) |     function                      |
| -   [cudaq::mpi::is_initialized   | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatorplENSt |
|     function                      | 7complexIdEERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[1\                         |
| Pv4N5cudaq3mpi14is_initializedEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::mpi::num_ranks (C++   | Pv4N5cudaq15scalar_operatorplENSt |
|     fu                            | 7complexIdEERR15scalar_operator), |
| nction)](api/languages/cpp_api.ht |     [\[2\]](api/languages/cp      |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::mpi::rank (C++        | operatorplEdRK15scalar_operator), |
|                                   |     [\[3\]](api/languages/cp      |
|    function)](api/languages/cpp_a | p_api.html#_CPPv4N5cudaq15scalar_ |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | operatorplEdRR15scalar_operator), |
| -   [cudaq::noise_model (C++      |     [\[4\]](api/languages         |
|                                   | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|    class)](api/languages/cpp_api. | alar_operatorplENSt7complexIdEE), |
| html#_CPPv4N5cudaq11noise_modelE) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::n                     | _api.html#_CPPv4NKR5cudaq15scalar |
| oise_model::add_all_qubit_channel | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function)](api                | (api/languages/cpp_api.html#_CPPv |
| /languages/cpp_api.html#_CPPv4IDp | 4NKR5cudaq15scalar_operatorplEd), |
| EN5cudaq11noise_model21add_all_qu |     [\[7\]]                       |
| bit_channelEvRK13kraus_channeli), | (api/languages/cpp_api.html#_CPPv |
|     [\[1\]](api/langua            | 4NKR5cudaq15scalar_operatorplEv), |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     [\[8\]](api/language          |
| oise_model21add_all_qubit_channel | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| ERKNSt6stringERK13kraus_channeli) | alar_operatorplENSt7complexIdEE), |
| -                                 |     [\[9\]](api/languages/cp      |
|  [cudaq::noise_model::add_channel | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatorplERK15scalar_operator), |
|     funct                         |     [\[10\]                       |
| ion)](api/languages/cpp_api.html# | ](api/languages/cpp_api.html#_CPP |
| _CPPv4IDpEN5cudaq11noise_model11a | v4NO5cudaq15scalar_operatorplEd), |
| dd_channelEvRK15PredicateFuncTy), |     [\[11\                        |
|     [\[1\]](api/languages/cpp_    | ]](api/languages/cpp_api.html#_CP |
| api.html#_CPPv4IDpEN5cudaq11noise | Pv4NO5cudaq15scalar_operatorplEv) |
| _model11add_channelEvRKNSt6vector | -   [c                            |
| INSt6size_tEEERK13kraus_channel), | udaq::scalar_operator::operator+= |
|     [\[2\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languag        |
| cudaq11noise_model11add_channelER | es/cpp_api.html#_CPPv4N5cudaq15sc |
| KNSt6stringERK15PredicateFuncTy), | alar_operatorpLENSt7complexIdEE), |
|                                   |     [\[1\]](api/languages/c       |
| [\[3\]](api/languages/cpp_api.htm | pp_api.html#_CPPv4N5cudaq15scalar |
| l#_CPPv4N5cudaq11noise_model11add | _operatorpLERK15scalar_operator), |
| _channelERKNSt6stringERKNSt6vecto |     [\[2                          |
| rINSt6size_tEEERK13kraus_channel) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::noise_model::empty    | PPv4N5cudaq15scalar_operatorpLEd) |
|     (C++                          | -   [                             |
|     function                      | cudaq::scalar_operator::operator- |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4NK5cudaq11noise_model5emptyEv) |     function                      |
| -                                 | )](api/languages/cpp_api.html#_CP |
| [cudaq::noise_model::get_channels | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     function)](api/l              |     [\[1\                         |
| anguages/cpp_api.html#_CPPv4I0ENK | ]](api/languages/cpp_api.html#_CP |
| 5cudaq11noise_model12get_channels | Pv4N5cudaq15scalar_operatormiENSt |
| ENSt6vectorI13kraus_channelEERKNS | 7complexIdEERR15scalar_operator), |
| t6vectorINSt6size_tEEERKNSt6vecto |     [\[2\]](api/languages/cp      |
| rINSt6size_tEEERKNSt6vectorIdEE), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](api/languages/cpp_a   | operatormiEdRK15scalar_operator), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[3\]](api/languages/cp      |
| el12get_channelsERKNSt6stringERKN | p_api.html#_CPPv4N5cudaq15scalar_ |
| St6vectorINSt6size_tEEERKNSt6vect | operatormiEdRR15scalar_operator), |
| orINSt6size_tEEERKNSt6vectorIdEE) |     [\[4\]](api/languages         |
| -                                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|  [cudaq::noise_model::noise_model | alar_operatormiENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     function)](api                | _api.html#_CPPv4NKR5cudaq15scalar |
| /languages/cpp_api.html#_CPPv4N5c | _operatormiERK15scalar_operator), |
| udaq11noise_model11noise_modelEv) |     [\[6\]]                       |
| -   [cu                           | (api/languages/cpp_api.html#_CPPv |
| daq::noise_model::PredicateFuncTy | 4NKR5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[7\]]                       |
|     type)](api/la                 | (api/languages/cpp_api.html#_CPPv |
| nguages/cpp_api.html#_CPPv4N5cuda | 4NKR5cudaq15scalar_operatormiEv), |
| q11noise_model15PredicateFuncTyE) |     [\[8\]](api/language          |
| -   [cud                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| aq::noise_model::register_channel | alar_operatormiENSt7complexIdEE), |
|     (C++                          |     [\[9\]](api/languages/cp      |
|     function)](api/languages      | p_api.html#_CPPv4NO5cudaq15scalar |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | _operatormiERK15scalar_operator), |
| noise_model16register_channelEvv) |     [\[10\]                       |
| -   [cudaq::                      | ](api/languages/cpp_api.html#_CPP |
| noise_model::requires_constructor | v4NO5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[11\                        |
|     type)](api/languages/cp       | ]](api/languages/cpp_api.html#_CP |
| p_api.html#_CPPv4I0DpEN5cudaq11no | Pv4NO5cudaq15scalar_operatormiEv) |
| ise_model20requires_constructorE) | -   [c                            |
| -   [cudaq::noise_model_type (C++ | udaq::scalar_operator::operator-= |
|     e                             |     (C++                          |
| num)](api/languages/cpp_api.html# |     function)](api/languag        |
| _CPPv4N5cudaq16noise_model_typeE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::no                    | alar_operatormIENSt7complexIdEE), |
| ise_model_type::amplitude_damping |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](api/languages    | _operatormIERK15scalar_operator), |
| /cpp_api.html#_CPPv4N5cudaq16nois |     [\[2                          |
| e_model_type17amplitude_dampingE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::noise_mode            | PPv4N5cudaq15scalar_operatormIEd) |
| l_type::amplitude_damping_channel | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator/ |
|     e                             |     (C++                          |
| numerator)](api/languages/cpp_api |     function                      |
| .html#_CPPv4N5cudaq16noise_model_ | )](api/languages/cpp_api.html#_CP |
| type25amplitude_damping_channelE) | Pv4N5cudaq15scalar_operatordvENSt |
| -   [cudaq::n                     | 7complexIdEERK15scalar_operator), |
| oise_model_type::bit_flip_channel |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](api/language     | Pv4N5cudaq15scalar_operatordvENSt |
| s/cpp_api.html#_CPPv4N5cudaq16noi | 7complexIdEERR15scalar_operator), |
| se_model_type16bit_flip_channelE) |     [\[2\]](api/languages/cp      |
| -   [cudaq::                      | p_api.html#_CPPv4N5cudaq15scalar_ |
| noise_model_type::depolarization1 | operatordvEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     enumerator)](api/languag      | p_api.html#_CPPv4N5cudaq15scalar_ |
| es/cpp_api.html#_CPPv4N5cudaq16no | operatordvEdRR15scalar_operator), |
| ise_model_type15depolarization1E) |     [\[4\]](api/languages         |
| -   [cudaq::                      | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| noise_model_type::depolarization2 | alar_operatordvENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     enumerator)](api/languag      | _api.html#_CPPv4NKR5cudaq15scalar |
| es/cpp_api.html#_CPPv4N5cudaq16no | _operatordvERK15scalar_operator), |
| ise_model_type15depolarization2E) |     [\[6\]]                       |
| -   [cudaq::noise_m               | (api/languages/cpp_api.html#_CPPv |
| odel_type::depolarization_channel | 4NKR5cudaq15scalar_operatordvEd), |
|     (C++                          |     [\[7\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|   enumerator)](api/languages/cpp_ | alar_operatordvENSt7complexIdEE), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[8\]](api/languages/cp      |
| el_type22depolarization_channelE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -                                 | _operatordvERK15scalar_operator), |
|  [cudaq::noise_model_type::pauli1 |     [\[9\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](a                | Pv4NO5cudaq15scalar_operatordvEd) |
| pi/languages/cpp_api.html#_CPPv4N | -   [c                            |
| 5cudaq16noise_model_type6pauli1E) | udaq::scalar_operator::operator/= |
| -                                 |     (C++                          |
|  [cudaq::noise_model_type::pauli2 |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     enumerator)](a                | alar_operatordVENSt7complexIdEE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[1\]](api/languages/c       |
| 5cudaq16noise_model_type6pauli2E) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq                        | _operatordVERK15scalar_operator), |
| ::noise_model_type::phase_damping |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     enumerator)](api/langu        | PPv4N5cudaq15scalar_operatordVEd) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [                             |
| noise_model_type13phase_dampingE) | cudaq::scalar_operator::operator= |
| -   [cudaq::noi                   |     (C++                          |
| se_model_type::phase_flip_channel |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](api/languages/   | _operatoraSERK15scalar_operator), |
| cpp_api.html#_CPPv4N5cudaq16noise |     [\[1\]](api/languages/        |
| _model_type18phase_flip_channelE) | cpp_api.html#_CPPv4N5cudaq15scala |
| -                                 | r_operatoraSERR15scalar_operator) |
| [cudaq::noise_model_type::unknown | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator== |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languages/c    |
| cudaq16noise_model_type7unknownE) | pp_api.html#_CPPv4NK5cudaq15scala |
| -                                 | r_operatoreqERK15scalar_operator) |
| [cudaq::noise_model_type::x_error | -   [cudaq:                       |
|     (C++                          | :scalar_operator::scalar_operator |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     func                          |
| cudaq16noise_model_type7x_errorE) | tion)](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq15scalar_operator15 |
| [cudaq::noise_model_type::y_error | scalar_operatorENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/langu             |
|     enumerator)](ap               | ages/cpp_api.html#_CPPv4N5cudaq15 |
| i/languages/cpp_api.html#_CPPv4N5 | scalar_operator15scalar_operatorE |
| cudaq16noise_model_type7y_errorE) | RK15scalar_callbackRRNSt13unorder |
| -                                 | ed_mapINSt6stringENSt6stringEEE), |
| [cudaq::noise_model_type::z_error |     [\[2\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](ap               | Pv4N5cudaq15scalar_operator15scal |
| i/languages/cpp_api.html#_CPPv4N5 | ar_operatorERK15scalar_operator), |
| cudaq16noise_model_type7z_errorE) |     [\[3\]](api/langu             |
| -   [cudaq::num_available_gpus    | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     (C++                          | scalar_operator15scalar_operatorE |
|     function                      | RR15scalar_callbackRRNSt13unorder |
| )](api/languages/cpp_api.html#_CP | ed_mapINSt6stringENSt6stringEEE), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[4\                         |
| -   [cudaq::observe (C++          | ]](api/languages/cpp_api.html#_CP |
|     function)]                    | Pv4N5cudaq15scalar_operator15scal |
| (api/languages/cpp_api.html#_CPPv | ar_operatorERR15scalar_operator), |
| 4I00DpEN5cudaq7observeENSt6vector |     [\[5\]](api/language          |
| I14observe_resultEERR13QuantumKer | s/cpp_api.html#_CPPv4N5cudaq15sca |
| nelRK15SpinOpContainerDpRR4Args), | lar_operator15scalar_operatorEd), |
|     [\[1\]](api/languages/cpp_ap  |     [\[6\]](api/languag           |
| i.html#_CPPv4I0DpEN5cudaq7observe | es/cpp_api.html#_CPPv4N5cudaq15sc |
| E14observe_resultNSt6size_tERR13Q | alar_operator15scalar_operatorEv) |
| uantumKernelRK7spin_opDpRR4Args), | -   [                             |
|     [\[                           | cudaq::scalar_operator::to_matrix |
| 2\]](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4I0DpEN5cudaq7observeE14obser |                                   |
| ve_resultRK15observe_optionsRR13Q |   function)](api/languages/cpp_ap |
| uantumKernelRK7spin_opDpRR4Args), | i.html#_CPPv4NK5cudaq15scalar_ope |
|     [\[3\]](api/lang              | rator9to_matrixERKNSt13unordered_ |
| uages/cpp_api.html#_CPPv4I0DpEN5c | mapINSt6stringENSt7complexIdEEEE) |
| udaq7observeE14observe_resultRR13 | -   [                             |
| QuantumKernelRK7spin_opDpRR4Args) | cudaq::scalar_operator::to_string |
| -   [cudaq::observe_options (C++  |     (C++                          |
|     st                            |     function)](api/l              |
| ruct)](api/languages/cpp_api.html | anguages/cpp_api.html#_CPPv4NK5cu |
| #_CPPv4N5cudaq15observe_optionsE) | daq15scalar_operator9to_stringEv) |
| -   [cudaq::observe_result (C++   | -   [cudaq::s                     |
|                                   | calar_operator::\~scalar_operator |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14observe_resultE) |     functio                       |
| -                                 | n)](api/languages/cpp_api.html#_C |
|    [cudaq::observe_result::counts | PPv4N5cudaq15scalar_operatorD0Ev) |
|     (C++                          | -   [cudaq::set_noise (C++        |
|     function)](api/languages/c    |     function)](api/langu          |
| pp_api.html#_CPPv4N5cudaq14observ | ages/cpp_api.html#_CPPv4N5cudaq9s |
| e_result6countsERK12spin_op_term) | et_noiseERKN5cudaq11noise_modelE) |
| -   [cudaq::observe_result::dump  | -   [cudaq::set_random_seed (C++  |
|     (C++                          |     function)](api/               |
|     function)                     | languages/cpp_api.html#_CPPv4N5cu |
| ](api/languages/cpp_api.html#_CPP | daq15set_random_seedENSt6size_tE) |
| v4N5cudaq14observe_result4dumpEv) | -   [cudaq::simulation_precision  |
| -   [c                            |     (C++                          |
| udaq::observe_result::expectation |     enum)                         |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq20simulation_precisionE) |
| function)](api/languages/cpp_api. | -   [                             |
| html#_CPPv4N5cudaq14observe_resul | cudaq::simulation_precision::fp32 |
| t11expectationERK12spin_op_term), |     (C++                          |
|     [\[1\]](api/la                |     enumerator)](api              |
| nguages/cpp_api.html#_CPPv4N5cuda | /languages/cpp_api.html#_CPPv4N5c |
| q14observe_result11expectationEv) | udaq20simulation_precision4fp32E) |
| -   [cuda                         | -   [                             |
| q::observe_result::id_coefficient | cudaq::simulation_precision::fp64 |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     enumerator)](api              |
| ages/cpp_api.html#_CPPv4N5cudaq14 | /languages/cpp_api.html#_CPPv4N5c |
| observe_result14id_coefficientEv) | udaq20simulation_precision4fp64E) |
| -   [cuda                         | -   [cudaq::SimulationState (C++  |
| q::observe_result::observe_result |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq15SimulationStateE) |
|   function)](api/languages/cpp_ap | -   [                             |
| i.html#_CPPv4N5cudaq14observe_res | cudaq::SimulationState::precision |
| ult14observe_resultEdRK7spin_op), |     (C++                          |
|     [\[1\]](a                     |     enum)](api                    |
| pi/languages/cpp_api.html#_CPPv4N | /languages/cpp_api.html#_CPPv4N5c |
| 5cudaq14observe_result14observe_r | udaq15SimulationState9precisionE) |
| esultEdRK7spin_op13sample_result) | -   [cudaq:                       |
| -                                 | :SimulationState::precision::fp32 |
|  [cudaq::observe_result::operator |     (C++                          |
|     double (C++                   |     enumerator)](api/lang         |
|     functio                       | uages/cpp_api.html#_CPPv4N5cudaq1 |
| n)](api/languages/cpp_api.html#_C | 5SimulationState9precision4fp32E) |
| PPv4N5cudaq14observe_resultcvdEv) | -   [cudaq:                       |
| -                                 | :SimulationState::precision::fp64 |
|  [cudaq::observe_result::raw_data |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     function)](ap                 | uages/cpp_api.html#_CPPv4N5cudaq1 |
| i/languages/cpp_api.html#_CPPv4N5 | 5SimulationState9precision4fp64E) |
| cudaq14observe_result8raw_dataEv) | -                                 |
| -   [cudaq::operator_handler (C++ |   [cudaq::SimulationState::Tensor |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     struct)](                     |
| _CPPv4N5cudaq16operator_handlerE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::optimizable_function  | N5cudaq15SimulationState6TensorE) |
|     (C++                          | -   [cudaq::spin_handler (C++     |
|     class)                        |                                   |
| ](api/languages/cpp_api.html#_CPP |   class)](api/languages/cpp_api.h |
| v4N5cudaq20optimizable_functionE) | tml#_CPPv4N5cudaq12spin_handlerE) |
| -   [cudaq::optimization_result   | -   [cudaq:                       |
|     (C++                          | :spin_handler::to_diagonal_matrix |
|     type                          |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/la             |
| Pv4N5cudaq19optimization_resultE) | nguages/cpp_api.html#_CPPv4NK5cud |
| -   [cudaq::optimizer (C++        | aq12spin_handler18to_diagonal_mat |
|     class)](api/languages/cpp_a   | rixERNSt13unordered_mapINSt6size_ |
| pi.html#_CPPv4N5cudaq9optimizerE) | tENSt7int64_tEEERKNSt13unordered_ |
| -   [cudaq::optimizer::optimize   | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -                                 |
|                                   |   [cudaq::spin_handler::to_matrix |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4N5cudaq9optimizer8opt |     function                      |
| imizeEKiRR20optimizable_function) | )](api/languages/cpp_api.html#_CP |
| -   [cu                           | Pv4N5cudaq12spin_handler9to_matri |
| daq::optimizer::requiresGradients | xERKNSt6stringENSt7complexIdEEb), |
|     (C++                          |     [\[1                          |
|     function)](api/la             | \]](api/languages/cpp_api.html#_C |
| nguages/cpp_api.html#_CPPv4N5cuda | PPv4NK5cudaq12spin_handler9to_mat |
| q9optimizer17requiresGradientsEv) | rixERNSt13unordered_mapINSt6size_ |
| -   [cudaq::orca (C++             | tENSt7int64_tEEERKNSt13unordered_ |
|     type)](api/languages/         | mapINSt6stringENSt7complexIdEEEE) |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | -   [cuda                         |
| -   [cudaq::orca::sample (C++     | q::spin_handler::to_sparse_matrix |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     function)](api/               |
| mpleERNSt6vectorINSt6size_tEEERNS | languages/cpp_api.html#_CPPv4N5cu |
| t6vectorINSt6size_tEEERNSt6vector | daq12spin_handler16to_sparse_matr |
| IdEERNSt6vectorIdEEiNSt6size_tE), | ixERKNSt6stringENSt7complexIdEEb) |
|     [\[1\]]                       | -                                 |
| (api/languages/cpp_api.html#_CPPv |   [cudaq::spin_handler::to_string |
| 4N5cudaq4orca6sampleERNSt6vectorI |     (C++                          |
| NSt6size_tEEERNSt6vectorINSt6size |     function)](ap                 |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::orca::sample_async    | 5cudaq12spin_handler9to_stringEb) |
|     (C++                          | -                                 |
|                                   |   [cudaq::spin_handler::unique_id |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq4orca12sample_a |     function)](ap                 |
| syncERNSt6vectorINSt6size_tEEERNS | i/languages/cpp_api.html#_CPPv4NK |
| t6vectorINSt6size_tEEERNSt6vector | 5cudaq12spin_handler9unique_idEv) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [cudaq::spin_op (C++          |
|     [\[1\]](api/la                |     type)](api/languages/cpp      |
| nguages/cpp_api.html#_CPPv4N5cuda | _api.html#_CPPv4N5cudaq7spin_opE) |
| q4orca12sample_asyncERNSt6vectorI | -   [cudaq::spin_op_term (C++     |
| NSt6size_tEEERNSt6vectorINSt6size |                                   |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |    type)](api/languages/cpp_api.h |
| -   [cudaq::OrcaRemoteRESTQPU     | tml#_CPPv4N5cudaq12spin_op_termE) |
|     (C++                          | -   [cudaq::state (C++            |
|     cla                           |     class)](api/languages/c       |
| ss)](api/languages/cpp_api.html#_ | pp_api.html#_CPPv4N5cudaq5stateE) |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | -   [cudaq::state::amplitude (C++ |
| -   [cudaq::pauli1 (C++           |     function)](api/lang           |
|     class)](api/languages/cp      | uages/cpp_api.html#_CPPv4N5cudaq5 |
| p_api.html#_CPPv4N5cudaq6pauli1E) | state9amplitudeERKNSt6vectorIiEE) |
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
| -   [define() (cudaq.operators    | -   [deserialize()                |
|     method)](api/languages/python |     (cudaq.SampleResult           |
| _api.html#cudaq.operators.define) |     meth                          |
|     -   [(cuda                    | od)](api/languages/python_api.htm |
| q.operators.MatrixOperatorElement | l#cudaq.SampleResult.deserialize) |
|         class                     | -   [displace() (in module        |
|         method)](api/langu        |     cudaq.operators.custo         |
| ages/python_api.html#cudaq.operat | m)](api/languages/python_api.html |
| ors.MatrixOperatorElement.define) | #cudaq.operators.custom.displace) |
|     -   [(in module               | -   [distribute_terms()           |
|         cudaq.operators.cus       |     (cu                           |
| tom)](api/languages/python_api.ht | daq.operators.boson.BosonOperator |
| ml#cudaq.operators.custom.define) |     method)](api/languages/pyt    |
| -   [degrees                      | hon_api.html#cudaq.operators.boso |
|     (cu                           | n.BosonOperator.distribute_terms) |
| daq.operators.boson.BosonOperator |     -   [(cudaq.                  |
|     property)](api/lang           | operators.fermion.FermionOperator |
| uages/python_api.html#cudaq.opera |                                   |
| tors.boson.BosonOperator.degrees) |    method)](api/languages/python_ |
|     -   [(cudaq.ope               | api.html#cudaq.operators.fermion. |
| rators.boson.BosonOperatorElement | FermionOperator.distribute_terms) |
|                                   |     -                             |
|        property)](api/languages/p |  [(cudaq.operators.MatrixOperator |
| ython_api.html#cudaq.operators.bo |         method)](api/language     |
| son.BosonOperatorElement.degrees) | s/python_api.html#cudaq.operators |
|     -   [(cudaq.                  | .MatrixOperator.distribute_terms) |
| operators.boson.BosonOperatorTerm |     -   [(                        |
|         property)](api/language   | cudaq.operators.spin.SpinOperator |
| s/python_api.html#cudaq.operators |         method)](api/languages/p  |
| .boson.BosonOperatorTerm.degrees) | ython_api.html#cudaq.operators.sp |
|     -   [(cudaq.                  | in.SpinOperator.distribute_terms) |
| operators.fermion.FermionOperator |     -   [(cuda                    |
|         property)](api/language   | q.operators.spin.SpinOperatorTerm |
| s/python_api.html#cudaq.operators |                                   |
| .fermion.FermionOperator.degrees) |      method)](api/languages/pytho |
|     -   [(cudaq.operato           | n_api.html#cudaq.operators.spin.S |
| rs.fermion.FermionOperatorElement | pinOperatorTerm.distribute_terms) |
|                                   | -   [draw() (in module            |
|    property)](api/languages/pytho |     cudaq)](api/lang              |
| n_api.html#cudaq.operators.fermio | uages/python_api.html#cudaq.draw) |
| n.FermionOperatorElement.degrees) | -   [dump() (cudaq.ComplexMatrix  |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |   method)](api/languages/python_a |
|                                   | pi.html#cudaq.ComplexMatrix.dump) |
|       property)](api/languages/py |     -   [(cudaq.ObserveResult     |
| thon_api.html#cudaq.operators.fer |                                   |
| mion.FermionOperatorTerm.degrees) |   method)](api/languages/python_a |
|     -                             | pi.html#cudaq.ObserveResult.dump) |
|  [(cudaq.operators.MatrixOperator |     -   [(cu                      |
|         property)](api            | daq.operators.boson.BosonOperator |
| /languages/python_api.html#cudaq. |         method)](api/l            |
| operators.MatrixOperator.degrees) | anguages/python_api.html#cudaq.op |
|     -   [(cuda                    | erators.boson.BosonOperator.dump) |
| q.operators.MatrixOperatorElement |     -   [(cudaq.                  |
|         property)](api/langua     | operators.boson.BosonOperatorTerm |
| ges/python_api.html#cudaq.operato |         method)](api/langu        |
| rs.MatrixOperatorElement.degrees) | ages/python_api.html#cudaq.operat |
|     -   [(c                       | ors.boson.BosonOperatorTerm.dump) |
| udaq.operators.MatrixOperatorTerm |     -   [(cudaq.                  |
|         property)](api/lan        | operators.fermion.FermionOperator |
| guages/python_api.html#cudaq.oper |         method)](api/langu        |
| ators.MatrixOperatorTerm.degrees) | ages/python_api.html#cudaq.operat |
|     -   [(                        | ors.fermion.FermionOperator.dump) |
| cudaq.operators.spin.SpinOperator |     -   [(cudaq.oper              |
|         property)](api/la         | ators.fermion.FermionOperatorTerm |
| nguages/python_api.html#cudaq.ope |         method)](api/languages    |
| rators.spin.SpinOperator.degrees) | /python_api.html#cudaq.operators. |
|     -   [(cudaq.o                 | fermion.FermionOperatorTerm.dump) |
| perators.spin.SpinOperatorElement |     -                             |
|         property)](api/languages  |  [(cudaq.operators.MatrixOperator |
| /python_api.html#cudaq.operators. |         method)](                 |
| spin.SpinOperatorElement.degrees) | api/languages/python_api.html#cud |
|     -   [(cuda                    | aq.operators.MatrixOperator.dump) |
| q.operators.spin.SpinOperatorTerm |     -   [(c                       |
|         property)](api/langua     | udaq.operators.MatrixOperatorTerm |
| ges/python_api.html#cudaq.operato |         method)](api/             |
| rs.spin.SpinOperatorTerm.degrees) | languages/python_api.html#cudaq.o |
| -                                 | perators.MatrixOperatorTerm.dump) |
|  [delete_cache_execution_engine() |     -   [(                        |
|     (cudaq.PyKernelDecorator      | cudaq.operators.spin.SpinOperator |
|     method)](api/languages/pyth   |         method)](api              |
| on_api.html#cudaq.PyKernelDecorat | /languages/python_api.html#cudaq. |
| or.delete_cache_execution_engine) | operators.spin.SpinOperator.dump) |
| -   [Depolarization1 (class in    |     -   [(cuda                    |
|     cudaq)](api/languages/pytho   | q.operators.spin.SpinOperatorTerm |
| n_api.html#cudaq.Depolarization1) |         method)](api/lan          |
| -   [Depolarization2 (class in    | guages/python_api.html#cudaq.oper |
|     cudaq)](api/languages/pytho   | ators.spin.SpinOperatorTerm.dump) |
| n_api.html#cudaq.Depolarization2) |     -   [(cudaq.Resources         |
| -   [DepolarizationChannel (class |                                   |
|     in                            |       method)](api/languages/pyth |
|                                   | on_api.html#cudaq.Resources.dump) |
| cudaq)](api/languages/python_api. |     -   [(cudaq.SampleResult      |
| html#cudaq.DepolarizationChannel) |                                   |
| -   [description (cudaq.Target    |    method)](api/languages/python_ |
|                                   | api.html#cudaq.SampleResult.dump) |
| property)](api/languages/python_a |     -   [(cudaq.State             |
| pi.html#cudaq.Target.description) |         method)](api/languages/   |
|                                   | python_api.html#cudaq.State.dump) |
+-----------------------------------+-----------------------------------+

## E {#E}

+-----------------------------------+-----------------------------------+
| -   [ElementaryOperator (in       | -   [evaluate()                   |
|     module                        |                                   |
|     cudaq.operators)]             |   (cudaq.operators.ScalarOperator |
| (api/languages/python_api.html#cu |     method)](api/                 |
| daq.operators.ElementaryOperator) | languages/python_api.html#cudaq.o |
| -   [empty()                      | perators.ScalarOperator.evaluate) |
|     (cu                           | -   [evaluate_coefficient()       |
| daq.operators.boson.BosonOperator |     (cudaq.                       |
|     static                        | operators.boson.BosonOperatorTerm |
|     method)](api/la               |     m                             |
| nguages/python_api.html#cudaq.ope | ethod)](api/languages/python_api. |
| rators.boson.BosonOperator.empty) | html#cudaq.operators.boson.BosonO |
|     -   [(cudaq.                  | peratorTerm.evaluate_coefficient) |
| operators.fermion.FermionOperator |     -   [(cudaq.oper              |
|         static                    | ators.fermion.FermionOperatorTerm |
|         method)](api/langua       |         metho                     |
| ges/python_api.html#cudaq.operato | d)](api/languages/python_api.html |
| rs.fermion.FermionOperator.empty) | #cudaq.operators.fermion.FermionO |
|     -                             | peratorTerm.evaluate_coefficient) |
|  [(cudaq.operators.MatrixOperator |     -   [(c                       |
|         static                    | udaq.operators.MatrixOperatorTerm |
|         method)](a                |                                   |
| pi/languages/python_api.html#cuda |     method)](api/languages/python |
| q.operators.MatrixOperator.empty) | _api.html#cudaq.operators.MatrixO |
|     -   [(                        | peratorTerm.evaluate_coefficient) |
| cudaq.operators.spin.SpinOperator |     -   [(cuda                    |
|         static                    | q.operators.spin.SpinOperatorTerm |
|         method)](api/             |                                   |
| languages/python_api.html#cudaq.o |  method)](api/languages/python_ap |
| perators.spin.SpinOperator.empty) | i.html#cudaq.operators.spin.SpinO |
|     -   [(in module               | peratorTerm.evaluate_coefficient) |
|                                   | -   [evolve() (in module          |
|     cudaq.boson)](api/languages/p |     cudaq)](api/langua            |
| ython_api.html#cudaq.boson.empty) | ges/python_api.html#cudaq.evolve) |
|     -   [(in module               | -   [evolve_async() (in module    |
|                                   |     cudaq)](api/languages/py      |
| cudaq.fermion)](api/languages/pyt | thon_api.html#cudaq.evolve_async) |
| hon_api.html#cudaq.fermion.empty) | -   [EvolveResult (class in       |
|     -   [(in module               |     cudaq)](api/languages/py      |
|         cudaq.operators.cu        | thon_api.html#cudaq.EvolveResult) |
| stom)](api/languages/python_api.h | -   [expectation()                |
| tml#cudaq.operators.custom.empty) |     (cudaq.ObserveResult          |
|     -   [(in module               |     metho                         |
|                                   | d)](api/languages/python_api.html |
|       cudaq.spin)](api/languages/ | #cudaq.ObserveResult.expectation) |
| python_api.html#cudaq.spin.empty) |     -   [(cudaq.SampleResult      |
| -   [empty_op()                   |         meth                      |
|     (                             | od)](api/languages/python_api.htm |
| cudaq.operators.spin.SpinOperator | l#cudaq.SampleResult.expectation) |
|     static                        | -   [expectation_values()         |
|     method)](api/lan              |     (cudaq.EvolveResult           |
| guages/python_api.html#cudaq.oper |     method)](ap                   |
| ators.spin.SpinOperator.empty_op) | i/languages/python_api.html#cudaq |
| -   [enable_return_to_log()       | .EvolveResult.expectation_values) |
|     (cudaq.PyKernelDecorator      | -   [expectation_z()              |
|     method)](api/langu            |     (cudaq.SampleResult           |
| ages/python_api.html#cudaq.PyKern |     method                        |
| elDecorator.enable_return_to_log) | )](api/languages/python_api.html# |
| -   [epsilon                      | cudaq.SampleResult.expectation_z) |
|     (cudaq.optimizers.Adam        | -   [expected_dimensions          |
|     prope                         |     (cuda                         |
| rty)](api/languages/python_api.ht | q.operators.MatrixOperatorElement |
| ml#cudaq.optimizers.Adam.epsilon) |                                   |
| -   [estimate_resources() (in     | property)](api/languages/python_a |
|     module                        | pi.html#cudaq.operators.MatrixOpe |
|                                   | ratorElement.expected_dimensions) |
|    cudaq)](api/languages/python_a |                                   |
| pi.html#cudaq.estimate_resources) |                                   |
+-----------------------------------+-----------------------------------+

## F {#F}

+-----------------------------------+-----------------------------------+
| -   [f_tol (cudaq.optimizers.Adam | -   [from_json()                  |
|     pro                           |     (                             |
| perty)](api/languages/python_api. | cudaq.gradients.CentralDifference |
| html#cudaq.optimizers.Adam.f_tol) |     static                        |
|     -   [(cudaq.optimizers.SGD    |     method)](api/lang             |
|         pr                        | uages/python_api.html#cudaq.gradi |
| operty)](api/languages/python_api | ents.CentralDifference.from_json) |
| .html#cudaq.optimizers.SGD.f_tol) |     -   [(                        |
| -   [FermionOperator (class in    | cudaq.gradients.ForwardDifference |
|                                   |         static                    |
|    cudaq.operators.fermion)](api/ |         method)](api/lang         |
| languages/python_api.html#cudaq.o | uages/python_api.html#cudaq.gradi |
| perators.fermion.FermionOperator) | ents.ForwardDifference.from_json) |
| -   [FermionOperatorElement       |     -                             |
|     (class in                     |  [(cudaq.gradients.ParameterShift |
|     cuda                          |         static                    |
| q.operators.fermion)](api/languag |         method)](api/l            |
| es/python_api.html#cudaq.operator | anguages/python_api.html#cudaq.gr |
| s.fermion.FermionOperatorElement) | adients.ParameterShift.from_json) |
| -   [FermionOperatorTerm (class   |     -   [(                        |
|     in                            | cudaq.operators.spin.SpinOperator |
|     c                             |         static                    |
| udaq.operators.fermion)](api/lang |         method)](api/lang         |
| uages/python_api.html#cudaq.opera | uages/python_api.html#cudaq.opera |
| tors.fermion.FermionOperatorTerm) | tors.spin.SpinOperator.from_json) |
| -   [final_expectation_values()   |     -   [(cuda                    |
|     (cudaq.EvolveResult           | q.operators.spin.SpinOperatorTerm |
|     method)](api/lang             |         static                    |
| uages/python_api.html#cudaq.Evolv |         method)](api/language     |
| eResult.final_expectation_values) | s/python_api.html#cudaq.operators |
| -   [final_state()                | .spin.SpinOperatorTerm.from_json) |
|     (cudaq.EvolveResult           |     -   [(cudaq.optimizers.Adam   |
|     meth                          |         static                    |
| od)](api/languages/python_api.htm |         metho                     |
| l#cudaq.EvolveResult.final_state) | d)](api/languages/python_api.html |
| -   [finalize() (in module        | #cudaq.optimizers.Adam.from_json) |
|     cudaq.mpi)](api/languages/py  |     -   [(cudaq.optimizers.COBYLA |
| thon_api.html#cudaq.mpi.finalize) |         static                    |
| -   [for_each_pauli()             |         method)                   |
|     (                             | ](api/languages/python_api.html#c |
| cudaq.operators.spin.SpinOperator | udaq.optimizers.COBYLA.from_json) |
|     method)](api/languages        |     -   [                         |
| /python_api.html#cudaq.operators. | (cudaq.optimizers.GradientDescent |
| spin.SpinOperator.for_each_pauli) |         static                    |
|     -   [(cuda                    |         method)](api/lan          |
| q.operators.spin.SpinOperatorTerm | guages/python_api.html#cudaq.opti |
|                                   | mizers.GradientDescent.from_json) |
|        method)](api/languages/pyt |     -   [(cudaq.optimizers.LBFGS  |
| hon_api.html#cudaq.operators.spin |         static                    |
| .SpinOperatorTerm.for_each_pauli) |         method                    |
| -   [for_each_term()              | )](api/languages/python_api.html# |
|     (                             | cudaq.optimizers.LBFGS.from_json) |
| cudaq.operators.spin.SpinOperator |                                   |
|     method)](api/language         | -   [(cudaq.optimizers.NelderMead |
| s/python_api.html#cudaq.operators |         static                    |
| .spin.SpinOperator.for_each_term) |         method)](ap               |
| -   [ForwardDifference (class in  | i/languages/python_api.html#cudaq |
|     cudaq.gradients)              | .optimizers.NelderMead.from_json) |
| ](api/languages/python_api.html#c |     -   [(cudaq.optimizers.SGD    |
| udaq.gradients.ForwardDifference) |         static                    |
| -   [from_data() (cudaq.State     |         meth                      |
|     static                        | od)](api/languages/python_api.htm |
|     method)](api/languages/pytho  | l#cudaq.optimizers.SGD.from_json) |
| n_api.html#cudaq.State.from_data) |     -   [(cudaq.optimizers.SPSA   |
|                                   |         static                    |
|                                   |         metho                     |
|                                   | d)](api/languages/python_api.html |
|                                   | #cudaq.optimizers.SPSA.from_json) |
|                                   |     -   [(cudaq.PyKernelDecorator |
|                                   |         static                    |
|                                   |         method)                   |
|                                   | ](api/languages/python_api.html#c |
|                                   | udaq.PyKernelDecorator.from_json) |
|                                   | -   [from_word()                  |
|                                   |     (                             |
|                                   | cudaq.operators.spin.SpinOperator |
|                                   |     static                        |
|                                   |     method)](api/lang             |
|                                   | uages/python_api.html#cudaq.opera |
|                                   | tors.spin.SpinOperator.from_word) |
+-----------------------------------+-----------------------------------+

## G {#G}

+-----------------------------------+-----------------------------------+
| -   [gamma (cudaq.optimizers.SPSA | -   [get_register_counts()        |
|     pro                           |     (cudaq.SampleResult           |
| perty)](api/languages/python_api. |     method)](api                  |
| html#cudaq.optimizers.SPSA.gamma) | /languages/python_api.html#cudaq. |
| -   [get()                        | SampleResult.get_register_counts) |
|     (cudaq.AsyncEvolveResult      | -   [get_sequential_data()        |
|     m                             |     (cudaq.SampleResult           |
| ethod)](api/languages/python_api. |     method)](api                  |
| html#cudaq.AsyncEvolveResult.get) | /languages/python_api.html#cudaq. |
|                                   | SampleResult.get_sequential_data) |
|    -   [(cudaq.AsyncObserveResult | -   [get_spin()                   |
|         me                        |     (cudaq.ObserveResult          |
| thod)](api/languages/python_api.h |     me                            |
| tml#cudaq.AsyncObserveResult.get) | thod)](api/languages/python_api.h |
|     -   [(cudaq.AsyncStateResult  | tml#cudaq.ObserveResult.get_spin) |
|                                   | -   [get_state() (in module       |
| method)](api/languages/python_api |     cudaq)](api/languages         |
| .html#cudaq.AsyncStateResult.get) | /python_api.html#cudaq.get_state) |
| -   [get_binary_symplectic_form() | -   [get_state_async() (in module |
|     (cuda                         |     cudaq)](api/languages/pytho   |
| q.operators.spin.SpinOperatorTerm | n_api.html#cudaq.get_state_async) |
|     metho                         | -   [get_state_refval()           |
| d)](api/languages/python_api.html |     (cudaq.State                  |
| #cudaq.operators.spin.SpinOperato |     me                            |
| rTerm.get_binary_symplectic_form) | thod)](api/languages/python_api.h |
| -   [get_channels()               | tml#cudaq.State.get_state_refval) |
|     (cudaq.NoiseModel             | -   [get_target() (in module      |
|     met                           |     cudaq)](api/languages/        |
| hod)](api/languages/python_api.ht | python_api.html#cudaq.get_target) |
| ml#cudaq.NoiseModel.get_channels) | -   [get_targets() (in module     |
| -   [get_coefficient()            |     cudaq)](api/languages/p       |
|     (                             | ython_api.html#cudaq.get_targets) |
| cudaq.operators.spin.SpinOperator | -   [get_term_count()             |
|     method)](api/languages/       |     (                             |
| python_api.html#cudaq.operators.s | cudaq.operators.spin.SpinOperator |
| pin.SpinOperator.get_coefficient) |     method)](api/languages        |
|     -   [(cuda                    | /python_api.html#cudaq.operators. |
| q.operators.spin.SpinOperatorTerm | spin.SpinOperator.get_term_count) |
|                                   | -   [get_total_shots()            |
|       method)](api/languages/pyth |     (cudaq.SampleResult           |
| on_api.html#cudaq.operators.spin. |     method)]                      |
| SpinOperatorTerm.get_coefficient) | (api/languages/python_api.html#cu |
| -   [get_marginal_counts()        | daq.SampleResult.get_total_shots) |
|     (cudaq.SampleResult           | -   [getTensor() (cudaq.State     |
|     method)](api                  |     method)](api/languages/pytho  |
| /languages/python_api.html#cudaq. | n_api.html#cudaq.State.getTensor) |
| SampleResult.get_marginal_counts) | -   [getTensors() (cudaq.State    |
| -   [get_ops()                    |     method)](api/languages/python |
|     (cudaq.KrausChannel           | _api.html#cudaq.State.getTensors) |
|                                   | -   [gradient (class in           |
| method)](api/languages/python_api |     cudaq.g                       |
| .html#cudaq.KrausChannel.get_ops) | radients)](api/languages/python_a |
| -   [get_pauli_word()             | pi.html#cudaq.gradients.gradient) |
|     (cuda                         | -   [GradientDescent (class in    |
| q.operators.spin.SpinOperatorTerm |     cudaq.optimizers              |
|     method)](api/languages/pyt    | )](api/languages/python_api.html# |
| hon_api.html#cudaq.operators.spin | cudaq.optimizers.GradientDescent) |
| .SpinOperatorTerm.get_pauli_word) |                                   |
| -   [get_precision()              |                                   |
|     (cudaq.Target                 |                                   |
|                                   |                                   |
| method)](api/languages/python_api |                                   |
| .html#cudaq.Target.get_precision) |                                   |
| -   [get_qubit_count()            |                                   |
|     (                             |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|     method)](api/languages/       |                                   |
| python_api.html#cudaq.operators.s |                                   |
| pin.SpinOperator.get_qubit_count) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|                                   |                                   |
|       method)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.spin. |                                   |
| SpinOperatorTerm.get_qubit_count) |                                   |
| -   [get_raw_data()               |                                   |
|     (                             |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|     method)](api/languag          |                                   |
| es/python_api.html#cudaq.operator |                                   |
| s.spin.SpinOperator.get_raw_data) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         method)](api/languages/p  |                                   |
| ython_api.html#cudaq.operators.sp |                                   |
| in.SpinOperatorTerm.get_raw_data) |                                   |
+-----------------------------------+-----------------------------------+

## H {#H}

+-----------------------------------+-----------------------------------+
| -   [handle_call_arguments()      | -   [has_target() (in module      |
|     (cudaq.PyKernelDecorator      |     cudaq)](api/languages/        |
|     method)](api/langua           | python_api.html#cudaq.has_target) |
| ges/python_api.html#cudaq.PyKerne |                                   |
| lDecorator.handle_call_arguments) |                                   |
+-----------------------------------+-----------------------------------+

## I {#I}

+-----------------------------------+-----------------------------------+
| -   [i() (in module               | -   [initialize() (in module      |
|     cudaq.spin)](api/langua       |                                   |
| ges/python_api.html#cudaq.spin.i) |    cudaq.mpi)](api/languages/pyth |
| -   [id                           | on_api.html#cudaq.mpi.initialize) |
|     (cuda                         | -   [initialize_cudaq() (in       |
| q.operators.MatrixOperatorElement |     module                        |
|     property)](api/l              |     cudaq)](api/languages/python  |
| anguages/python_api.html#cudaq.op | _api.html#cudaq.initialize_cudaq) |
| erators.MatrixOperatorElement.id) | -   [InitialState (in module      |
| -   [identities() (in module      |     cudaq.dynamics.helpers)](     |
|     c                             | api/languages/python_api.html#cud |
| udaq.boson)](api/languages/python | aq.dynamics.helpers.InitialState) |
| _api.html#cudaq.boson.identities) | -   [InitialStateType (class in   |
|     -   [(in module               |     cudaq)](api/languages/python  |
|         cudaq                     | _api.html#cudaq.InitialStateType) |
| .fermion)](api/languages/python_a | -   [instantiate()                |
| pi.html#cudaq.fermion.identities) |     (cudaq.operators              |
|     -   [(in module               |     m                             |
|         cudaq.operators.custom)   | ethod)](api/languages/python_api. |
| ](api/languages/python_api.html#c | html#cudaq.operators.instantiate) |
| udaq.operators.custom.identities) |     -   [(in module               |
|     -   [(in module               |         cudaq.operators.custom)]  |
|                                   | (api/languages/python_api.html#cu |
|  cudaq.spin)](api/languages/pytho | daq.operators.custom.instantiate) |
| n_api.html#cudaq.spin.identities) | -   [intermediate_states()        |
| -   [identity()                   |     (cudaq.EvolveResult           |
|     (cu                           |     method)](api                  |
| daq.operators.boson.BosonOperator | /languages/python_api.html#cudaq. |
|     static                        | EvolveResult.intermediate_states) |
|     method)](api/langu            | -   [IntermediateResultSave       |
| ages/python_api.html#cudaq.operat |     (class in                     |
| ors.boson.BosonOperator.identity) |     c                             |
|     -   [(cudaq.                  | udaq)](api/languages/python_api.h |
| operators.fermion.FermionOperator | tml#cudaq.IntermediateResultSave) |
|         static                    | -   [is_compiled()                |
|         method)](api/languages    |     (cudaq.PyKernelDecorator      |
| /python_api.html#cudaq.operators. |     method)](                     |
| fermion.FermionOperator.identity) | api/languages/python_api.html#cud |
|     -                             | aq.PyKernelDecorator.is_compiled) |
|  [(cudaq.operators.MatrixOperator | -   [is_constant()                |
|         static                    |                                   |
|         method)](api/             |   (cudaq.operators.ScalarOperator |
| languages/python_api.html#cudaq.o |     method)](api/lan              |
| perators.MatrixOperator.identity) | guages/python_api.html#cudaq.oper |
|     -   [(                        | ators.ScalarOperator.is_constant) |
| cudaq.operators.spin.SpinOperator | -   [is_emulated() (cudaq.Target  |
|         static                    |                                   |
|         method)](api/lan          |   method)](api/languages/python_a |
| guages/python_api.html#cudaq.oper | pi.html#cudaq.Target.is_emulated) |
| ators.spin.SpinOperator.identity) | -   [is_identity()                |
|     -   [(in module               |     (cudaq.                       |
|                                   | operators.boson.BosonOperatorTerm |
|  cudaq.boson)](api/languages/pyth |     method)](api/languages/py     |
| on_api.html#cudaq.boson.identity) | thon_api.html#cudaq.operators.bos |
|     -   [(in module               | on.BosonOperatorTerm.is_identity) |
|         cud                       |     -   [(cudaq.oper              |
| aq.fermion)](api/languages/python | ators.fermion.FermionOperatorTerm |
| _api.html#cudaq.fermion.identity) |                                   |
|     -   [(in module               |     method)](api/languages/python |
|                                   | _api.html#cudaq.operators.fermion |
|    cudaq.spin)](api/languages/pyt | .FermionOperatorTerm.is_identity) |
| hon_api.html#cudaq.spin.identity) |     -   [(c                       |
| -   [initial_parameters           | udaq.operators.MatrixOperatorTerm |
|     (cudaq.optimizers.Adam        |         method)](api/languag      |
|     property)](api/l              | es/python_api.html#cudaq.operator |
| anguages/python_api.html#cudaq.op | s.MatrixOperatorTerm.is_identity) |
| timizers.Adam.initial_parameters) |     -   [(                        |
|     -   [(cudaq.optimizers.COBYLA | cudaq.operators.spin.SpinOperator |
|         property)](api/lan        |         method)](api/langua       |
| guages/python_api.html#cudaq.opti | ges/python_api.html#cudaq.operato |
| mizers.COBYLA.initial_parameters) | rs.spin.SpinOperator.is_identity) |
|     -   [                         |     -   [(cuda                    |
| (cudaq.optimizers.GradientDescent | q.operators.spin.SpinOperatorTerm |
|                                   |         method)](api/languages/   |
|       property)](api/languages/py | python_api.html#cudaq.operators.s |
| thon_api.html#cudaq.optimizers.Gr | pin.SpinOperatorTerm.is_identity) |
| adientDescent.initial_parameters) | -   [is_initialized() (in module  |
|     -   [(cudaq.optimizers.LBFGS  |     c                             |
|         property)](api/la         | udaq.mpi)](api/languages/python_a |
| nguages/python_api.html#cudaq.opt | pi.html#cudaq.mpi.is_initialized) |
| imizers.LBFGS.initial_parameters) | -   [is_on_gpu() (cudaq.State     |
|                                   |     method)](api/languages/pytho  |
| -   [(cudaq.optimizers.NelderMead | n_api.html#cudaq.State.is_on_gpu) |
|         property)](api/languag    | -   [is_remote() (cudaq.Target    |
| es/python_api.html#cudaq.optimize |     method)](api/languages/python |
| rs.NelderMead.initial_parameters) | _api.html#cudaq.Target.is_remote) |
|     -   [(cudaq.optimizers.SGD    | -   [is_remote_simulator()        |
|         property)](api/           |     (cudaq.Target                 |
| languages/python_api.html#cudaq.o |     method                        |
| ptimizers.SGD.initial_parameters) | )](api/languages/python_api.html# |
|     -   [(cudaq.optimizers.SPSA   | cudaq.Target.is_remote_simulator) |
|         property)](api/l          | -   [items() (cudaq.SampleResult  |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.SPSA.initial_parameters) |   method)](api/languages/python_a |
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
+-----------------------------------+-----------------------------------+

## L {#L}

+-----------------------------------+-----------------------------------+
| -   [launch_args_required()       | -   [lower_quake_to_codegen()     |
|     (cudaq.PyKernelDecorator      |     (cudaq.PyKernelDecorator      |
|     method)](api/langu            |     method)](api/languag          |
| ages/python_api.html#cudaq.PyKern | es/python_api.html#cudaq.PyKernel |
| elDecorator.launch_args_required) | Decorator.lower_quake_to_codegen) |
| -   [LBFGS (class in              |                                   |
|     cudaq.                        |                                   |
| optimizers)](api/languages/python |                                   |
| _api.html#cudaq.optimizers.LBFGS) |                                   |
| -   [left_multiply()              |                                   |
|     (cudaq.SuperOperator static   |                                   |
|     method)                       |                                   |
| ](api/languages/python_api.html#c |                                   |
| udaq.SuperOperator.left_multiply) |                                   |
| -   [left_right_multiply()        |                                   |
|     (cudaq.SuperOperator static   |                                   |
|     method)](api/                 |                                   |
| languages/python_api.html#cudaq.S |                                   |
| uperOperator.left_right_multiply) |                                   |
| -   [lower_bounds                 |                                   |
|     (cudaq.optimizers.Adam        |                                   |
|     property)]                    |                                   |
| (api/languages/python_api.html#cu |                                   |
| daq.optimizers.Adam.lower_bounds) |                                   |
|     -   [(cudaq.optimizers.COBYLA |                                   |
|         property)](a              |                                   |
| pi/languages/python_api.html#cuda |                                   |
| q.optimizers.COBYLA.lower_bounds) |                                   |
|     -   [                         |                                   |
| (cudaq.optimizers.GradientDescent |                                   |
|         property)](api/langua     |                                   |
| ges/python_api.html#cudaq.optimiz |                                   |
| ers.GradientDescent.lower_bounds) |                                   |
|     -   [(cudaq.optimizers.LBFGS  |                                   |
|         property)](               |                                   |
| api/languages/python_api.html#cud |                                   |
| aq.optimizers.LBFGS.lower_bounds) |                                   |
|                                   |                                   |
| -   [(cudaq.optimizers.NelderMead |                                   |
|         property)](api/l          |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.NelderMead.lower_bounds) |                                   |
|     -   [(cudaq.optimizers.SGD    |                                   |
|         property)                 |                                   |
| ](api/languages/python_api.html#c |                                   |
| udaq.optimizers.SGD.lower_bounds) |                                   |
|     -   [(cudaq.optimizers.SPSA   |                                   |
|         property)]                |                                   |
| (api/languages/python_api.html#cu |                                   |
| daq.optimizers.SPSA.lower_bounds) |                                   |
+-----------------------------------+-----------------------------------+

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
| ors.spin.SpinOperator.max_degree) | -   [minimal_eigenvalue()         |
|     -   [(cuda                    |     (cudaq.ComplexMatrix          |
| q.operators.spin.SpinOperatorTerm |     method)](api                  |
|         property)](api/languages  | /languages/python_api.html#cudaq. |
| /python_api.html#cudaq.operators. | ComplexMatrix.minimal_eigenvalue) |
| spin.SpinOperatorTerm.max_degree) | -   [minus() (in module           |
| -   [max_iterations               |     cudaq.spin)](api/languages/   |
|     (cudaq.optimizers.Adam        | python_api.html#cudaq.spin.minus) |
|     property)](a                  | -   module                        |
| pi/languages/python_api.html#cuda |     -   [cudaq](api/langua        |
| q.optimizers.Adam.max_iterations) | ges/python_api.html#module-cudaq) |
|     -   [(cudaq.optimizers.COBYLA |     -                             |
|         property)](api            |    [cudaq.boson](api/languages/py |
| /languages/python_api.html#cudaq. | thon_api.html#module-cudaq.boson) |
| optimizers.COBYLA.max_iterations) |     -   [                         |
|     -   [                         | cudaq.fermion](api/languages/pyth |
| (cudaq.optimizers.GradientDescent | on_api.html#module-cudaq.fermion) |
|         property)](api/language   |     -   [cudaq.operators.cu       |
| s/python_api.html#cudaq.optimizer | stom](api/languages/python_api.ht |
| s.GradientDescent.max_iterations) | ml#module-cudaq.operators.custom) |
|     -   [(cudaq.optimizers.LBFGS  |                                   |
|         property)](ap             |  -   [cudaq.spin](api/languages/p |
| i/languages/python_api.html#cudaq | ython_api.html#module-cudaq.spin) |
| .optimizers.LBFGS.max_iterations) | -   [momentum() (in module        |
|                                   |                                   |
| -   [(cudaq.optimizers.NelderMead |  cudaq.boson)](api/languages/pyth |
|         property)](api/lan        | on_api.html#cudaq.boson.momentum) |
| guages/python_api.html#cudaq.opti |     -   [(in module               |
| mizers.NelderMead.max_iterations) |         cudaq.operators.custo     |
|     -   [(cudaq.optimizers.SGD    | m)](api/languages/python_api.html |
|         property)](               | #cudaq.operators.custom.momentum) |
| api/languages/python_api.html#cud | -   [most_probable()              |
| aq.optimizers.SGD.max_iterations) |     (cudaq.SampleResult           |
|     -   [(cudaq.optimizers.SPSA   |     method                        |
|         property)](a              | )](api/languages/python_api.html# |
| pi/languages/python_api.html#cuda | cudaq.SampleResult.most_probable) |
| q.optimizers.SPSA.max_iterations) |                                   |
| -   [mdiag_sparse_matrix (C++     |                                   |
|     type)](api/languages/cpp_api. |                                   |
| html#_CPPv419mdiag_sparse_matrix) |                                   |
+-----------------------------------+-----------------------------------+

## N {#N}

+-----------------------------------+-----------------------------------+
| -   [name (cudaq.PyKernel         | -   [num_qpus() (cudaq.Target     |
|     attribute)](api/languages/pyt |     method)](api/languages/pytho  |
| hon_api.html#cudaq.PyKernel.name) | n_api.html#cudaq.Target.num_qpus) |
|                                   | -   [num_qubits() (cudaq.State    |
|   -   [(cudaq.SimulationPrecision |     method)](api/languages/python |
|         proper                    | _api.html#cudaq.State.num_qubits) |
| ty)](api/languages/python_api.htm | -   [num_ranks() (in module       |
| l#cudaq.SimulationPrecision.name) |     cudaq.mpi)](api/languages/pyt |
|     -   [(cudaq.spin.Pauli        | hon_api.html#cudaq.mpi.num_ranks) |
|                                   | -   [num_rows()                   |
|    property)](api/languages/pytho |     (cudaq.ComplexMatrix          |
| n_api.html#cudaq.spin.Pauli.name) |     me                            |
|     -   [(cudaq.Target            | thod)](api/languages/python_api.h |
|                                   | tml#cudaq.ComplexMatrix.num_rows) |
|        property)](api/languages/p | -   [number() (in module          |
| ython_api.html#cudaq.Target.name) |                                   |
| -   [NelderMead (class in         |    cudaq.boson)](api/languages/py |
|     cudaq.optim                   | thon_api.html#cudaq.boson.number) |
| izers)](api/languages/python_api. |     -   [(in module               |
| html#cudaq.optimizers.NelderMead) |         c                         |
| -   [NoiseModel (class in         | udaq.fermion)](api/languages/pyth |
|     cudaq)](api/languages/        | on_api.html#cudaq.fermion.number) |
| python_api.html#cudaq.NoiseModel) |     -   [(in module               |
| -   [num_available_gpus() (in     |         cudaq.operators.cus       |
|     module                        | tom)](api/languages/python_api.ht |
|                                   | ml#cudaq.operators.custom.number) |
|    cudaq)](api/languages/python_a | -   [nvqir::MPSSimulationState    |
| pi.html#cudaq.num_available_gpus) |     (C++                          |
| -   [num_columns()                |     class)]                       |
|     (cudaq.ComplexMatrix          | (api/languages/cpp_api.html#_CPPv |
|     metho                         | 4I0EN5nvqir18MPSSimulationStateE) |
| d)](api/languages/python_api.html | -                                 |
| #cudaq.ComplexMatrix.num_columns) |  [nvqir::TensorNetSimulationState |
|                                   |     (C++                          |
|                                   |     class)](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4I0EN5 |
|                                   | nvqir24TensorNetSimulationStateE) |
+-----------------------------------+-----------------------------------+

## O {#O}

+-----------------------------------+-----------------------------------+
| -   [observe() (in module         | -   [OptimizationResult (class in |
|     cudaq)](api/languag           |                                   |
| es/python_api.html#cudaq.observe) |    cudaq)](api/languages/python_a |
| -   [observe_async() (in module   | pi.html#cudaq.OptimizationResult) |
|     cudaq)](api/languages/pyt     | -   [overlap() (cudaq.State       |
| hon_api.html#cudaq.observe_async) |     method)](api/languages/pyt    |
| -   [ObserveResult (class in      | hon_api.html#cudaq.State.overlap) |
|     cudaq)](api/languages/pyt     |                                   |
| hon_api.html#cudaq.ObserveResult) |                                   |
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
| -   [parameters                   | -   [Pauli (class in              |
|     (cu                           |     cudaq.spin)](api/languages/   |
| daq.operators.boson.BosonOperator | python_api.html#cudaq.spin.Pauli) |
|     property)](api/languag        | -   [Pauli1 (class in             |
| es/python_api.html#cudaq.operator |     cudaq)](api/langua            |
| s.boson.BosonOperator.parameters) | ges/python_api.html#cudaq.Pauli1) |
|     -   [(cudaq.                  | -   [Pauli2 (class in             |
| operators.boson.BosonOperatorTerm |     cudaq)](api/langua            |
|                                   | ges/python_api.html#cudaq.Pauli2) |
|        property)](api/languages/p | -   [PhaseDamping (class in       |
| ython_api.html#cudaq.operators.bo |     cudaq)](api/languages/py      |
| son.BosonOperatorTerm.parameters) | thon_api.html#cudaq.PhaseDamping) |
|     -   [(cudaq.                  | -   [PhaseFlipChannel (class in   |
| operators.fermion.FermionOperator |     cudaq)](api/languages/python  |
|                                   | _api.html#cudaq.PhaseFlipChannel) |
|        property)](api/languages/p | -   [platform (cudaq.Target       |
| ython_api.html#cudaq.operators.fe |                                   |
| rmion.FermionOperator.parameters) |    property)](api/languages/pytho |
|     -   [(cudaq.oper              | n_api.html#cudaq.Target.platform) |
| ators.fermion.FermionOperatorTerm | -   [plus() (in module            |
|                                   |     cudaq.spin)](api/languages    |
|    property)](api/languages/pytho | /python_api.html#cudaq.spin.plus) |
| n_api.html#cudaq.operators.fermio | -   [position() (in module        |
| n.FermionOperatorTerm.parameters) |                                   |
|     -                             |  cudaq.boson)](api/languages/pyth |
|  [(cudaq.operators.MatrixOperator | on_api.html#cudaq.boson.position) |
|         property)](api/la         |     -   [(in module               |
| nguages/python_api.html#cudaq.ope |         cudaq.operators.custo     |
| rators.MatrixOperator.parameters) | m)](api/languages/python_api.html |
|     -   [(cuda                    | #cudaq.operators.custom.position) |
| q.operators.MatrixOperatorElement | -   [probability()                |
|         property)](api/languages  |     (cudaq.SampleResult           |
| /python_api.html#cudaq.operators. |     meth                          |
| MatrixOperatorElement.parameters) | od)](api/languages/python_api.htm |
|     -   [(c                       | l#cudaq.SampleResult.probability) |
| udaq.operators.MatrixOperatorTerm | -   [ProductOperator (in module   |
|         property)](api/langua     |     cudaq.operator                |
| ges/python_api.html#cudaq.operato | s)](api/languages/python_api.html |
| rs.MatrixOperatorTerm.parameters) | #cudaq.operators.ProductOperator) |
|     -                             | -   [PyKernel (class in           |
|  [(cudaq.operators.ScalarOperator |     cudaq)](api/language          |
|         property)](api/la         | s/python_api.html#cudaq.PyKernel) |
| nguages/python_api.html#cudaq.ope | -   [PyKernelDecorator (class in  |
| rators.ScalarOperator.parameters) |     cudaq)](api/languages/python_ |
|     -   [(                        | api.html#cudaq.PyKernelDecorator) |
| cudaq.operators.spin.SpinOperator |                                   |
|         property)](api/langu      |                                   |
| ages/python_api.html#cudaq.operat |                                   |
| ors.spin.SpinOperator.parameters) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         property)](api/languages  |                                   |
| /python_api.html#cudaq.operators. |                                   |
| spin.SpinOperatorTerm.parameters) |                                   |
| -   [ParameterShift (class in     |                                   |
|     cudaq.gradien                 |                                   |
| ts)](api/languages/python_api.htm |                                   |
| l#cudaq.gradients.ParameterShift) |                                   |
| -   [parity() (in module          |                                   |
|     cudaq.operators.cus           |                                   |
| tom)](api/languages/python_api.ht |                                   |
| ml#cudaq.operators.custom.parity) |                                   |
+-----------------------------------+-----------------------------------+

## Q {#Q}

+-----------------------------------+-----------------------------------+
| -   [qkeModule                    | -   [qubit (class in              |
|     (cudaq.PyKernelDecorator      |     cudaq)](api/langu             |
|     property)                     | ages/python_api.html#cudaq.qubit) |
| ](api/languages/python_api.html#c | -   [qubit_count                  |
| udaq.PyKernelDecorator.qkeModule) |     (                             |
| -   [qreg (in module              | cudaq.operators.spin.SpinOperator |
|     cudaq)](api/lang              |     property)](api/langua         |
| uages/python_api.html#cudaq.qreg) | ges/python_api.html#cudaq.operato |
| -   [QuakeValue (class in         | rs.spin.SpinOperator.qubit_count) |
|     cudaq)](api/languages/        |     -   [(cuda                    |
| python_api.html#cudaq.QuakeValue) | q.operators.spin.SpinOperatorTerm |
|                                   |         property)](api/languages/ |
|                                   | python_api.html#cudaq.operators.s |
|                                   | pin.SpinOperatorTerm.qubit_count) |
|                                   | -   [qvector (class in            |
|                                   |     cudaq)](api/languag           |
|                                   | es/python_api.html#cudaq.qvector) |
+-----------------------------------+-----------------------------------+

## R {#R}

+-----------------------------------+-----------------------------------+
| -   [random()                     | -   [Resources (class in          |
|     (                             |     cudaq)](api/languages         |
| cudaq.operators.spin.SpinOperator | /python_api.html#cudaq.Resources) |
|     static                        | -   [right_multiply()             |
|     method)](api/l                |     (cudaq.SuperOperator static   |
| anguages/python_api.html#cudaq.op |     method)]                      |
| erators.spin.SpinOperator.random) | (api/languages/python_api.html#cu |
| -   [rank() (in module            | daq.SuperOperator.right_multiply) |
|     cudaq.mpi)](api/language      | -   [row_count                    |
| s/python_api.html#cudaq.mpi.rank) |     (cudaq.KrausOperator          |
| -   [register_names               |     prope                         |
|     (cudaq.SampleResult           | rty)](api/languages/python_api.ht |
|     attribute)                    | ml#cudaq.KrausOperator.row_count) |
| ](api/languages/python_api.html#c | -   [run() (in module             |
| udaq.SampleResult.register_names) |     cudaq)](api/lan               |
| -                                 | guages/python_api.html#cudaq.run) |
|   [register_set_target_callback() | -   [run_async() (in module       |
|     (in module                    |     cudaq)](api/languages         |
|     cudaq)]                       | /python_api.html#cudaq.run_async) |
| (api/languages/python_api.html#cu | -   [RydbergHamiltonian (class in |
| daq.register_set_target_callback) |     cudaq.operators)]             |
| -   [reset_target() (in module    | (api/languages/python_api.html#cu |
|     cudaq)](api/languages/py      | daq.operators.RydbergHamiltonian) |
| thon_api.html#cudaq.reset_target) |                                   |
+-----------------------------------+-----------------------------------+

## S {#S}

+-----------------------------------+-----------------------------------+
| -   [sample() (in module          | -   [SimulationPrecision (class   |
|     cudaq)](api/langua            |     in                            |
| ges/python_api.html#cudaq.sample) |                                   |
|     -   [(in module               |   cudaq)](api/languages/python_ap |
|                                   | i.html#cudaq.SimulationPrecision) |
|      cudaq.orca)](api/languages/p | -   [simulator (cudaq.Target      |
| ython_api.html#cudaq.orca.sample) |                                   |
| -   [sample_async() (in module    |   property)](api/languages/python |
|     cudaq)](api/languages/py      | _api.html#cudaq.Target.simulator) |
| thon_api.html#cudaq.sample_async) | -   [slice() (cudaq.QuakeValue    |
| -   [SampleResult (class in       |     method)](api/languages/python |
|     cudaq)](api/languages/py      | _api.html#cudaq.QuakeValue.slice) |
| thon_api.html#cudaq.SampleResult) | -   [SpinOperator (class in       |
| -   [ScalarOperator (class in     |     cudaq.operators.spin)         |
|     cudaq.operato                 | ](api/languages/python_api.html#c |
| rs)](api/languages/python_api.htm | udaq.operators.spin.SpinOperator) |
| l#cudaq.operators.ScalarOperator) | -   [SpinOperatorElement (class   |
| -   [Schedule (class in           |     in                            |
|     cudaq)](api/language          |     cudaq.operators.spin)](api/l  |
| s/python_api.html#cudaq.Schedule) | anguages/python_api.html#cudaq.op |
| -   [serialize()                  | erators.spin.SpinOperatorElement) |
|     (                             | -   [SpinOperatorTerm (class in   |
| cudaq.operators.spin.SpinOperator |     cudaq.operators.spin)](ap     |
|     method)](api/lang             | i/languages/python_api.html#cudaq |
| uages/python_api.html#cudaq.opera | .operators.spin.SpinOperatorTerm) |
| tors.spin.SpinOperator.serialize) | -   [SPSA (class in               |
|     -   [(cuda                    |     cudaq                         |
| q.operators.spin.SpinOperatorTerm | .optimizers)](api/languages/pytho |
|         method)](api/language     | n_api.html#cudaq.optimizers.SPSA) |
| s/python_api.html#cudaq.operators | -   [squeeze() (in module         |
| .spin.SpinOperatorTerm.serialize) |     cudaq.operators.cust          |
|     -   [(cudaq.SampleResult      | om)](api/languages/python_api.htm |
|         me                        | l#cudaq.operators.custom.squeeze) |
| thod)](api/languages/python_api.h | -   [State (class in              |
| tml#cudaq.SampleResult.serialize) |     cudaq)](api/langu             |
| -   [set_noise() (in module       | ages/python_api.html#cudaq.State) |
|     cudaq)](api/languages         | -   [step_size                    |
| /python_api.html#cudaq.set_noise) |     (cudaq.optimizers.Adam        |
| -   [set_random_seed() (in module |     propert                       |
|     cudaq)](api/languages/pytho   | y)](api/languages/python_api.html |
| n_api.html#cudaq.set_random_seed) | #cudaq.optimizers.Adam.step_size) |
| -   [set_target() (in module      |     -   [(cudaq.optimizers.SGD    |
|     cudaq)](api/languages/        |         proper                    |
| python_api.html#cudaq.set_target) | ty)](api/languages/python_api.htm |
| -   [SGD (class in                | l#cudaq.optimizers.SGD.step_size) |
|     cuda                          |     -   [(cudaq.optimizers.SPSA   |
| q.optimizers)](api/languages/pyth |         propert                   |
| on_api.html#cudaq.optimizers.SGD) | y)](api/languages/python_api.html |
| -   [signatureWithCallables()     | #cudaq.optimizers.SPSA.step_size) |
|     (cudaq.PyKernelDecorator      | -   [SuperOperator (class in      |
|     method)](api/languag          |     cudaq)](api/languages/pyt     |
| es/python_api.html#cudaq.PyKernel | hon_api.html#cudaq.SuperOperator) |
| Decorator.signatureWithCallables) | -   [supports_compilation()       |
|                                   |     (cudaq.PyKernelDecorator      |
|                                   |     method)](api/langu            |
|                                   | ages/python_api.html#cudaq.PyKern |
|                                   | elDecorator.supports_compilation) |
+-----------------------------------+-----------------------------------+

## T {#T}

+-----------------------------------+-----------------------------------+
| -   [Target (class in             | -   [to_numpy()                   |
|     cudaq)](api/langua            |     (cudaq.ComplexMatrix          |
| ges/python_api.html#cudaq.Target) |     me                            |
| -   [target                       | thod)](api/languages/python_api.h |
|     (cudaq.ope                    | tml#cudaq.ComplexMatrix.to_numpy) |
| rators.boson.BosonOperatorElement | -   [to_sparse_matrix()           |
|     property)](api/languages/     |     (cu                           |
| python_api.html#cudaq.operators.b | daq.operators.boson.BosonOperator |
| oson.BosonOperatorElement.target) |     method)](api/languages/pyt    |
|     -   [(cudaq.operato           | hon_api.html#cudaq.operators.boso |
| rs.fermion.FermionOperatorElement | n.BosonOperator.to_sparse_matrix) |
|                                   |     -   [(cudaq.                  |
|     property)](api/languages/pyth | operators.boson.BosonOperatorTerm |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorElement.target) |    method)](api/languages/python_ |
|     -   [(cudaq.o                 | api.html#cudaq.operators.boson.Bo |
| perators.spin.SpinOperatorElement | sonOperatorTerm.to_sparse_matrix) |
|         property)](api/language   |     -   [(cudaq.                  |
| s/python_api.html#cudaq.operators | operators.fermion.FermionOperator |
| .spin.SpinOperatorElement.target) |                                   |
| -   [Tensor (class in             |    method)](api/languages/python_ |
|     cudaq)](api/langua            | api.html#cudaq.operators.fermion. |
| ges/python_api.html#cudaq.Tensor) | FermionOperator.to_sparse_matrix) |
| -   [term_count                   |     -   [(cudaq.oper              |
|     (cu                           | ators.fermion.FermionOperatorTerm |
| daq.operators.boson.BosonOperator |         m                         |
|     property)](api/languag        | ethod)](api/languages/python_api. |
| es/python_api.html#cudaq.operator | html#cudaq.operators.fermion.Ferm |
| s.boson.BosonOperator.term_count) | ionOperatorTerm.to_sparse_matrix) |
|     -   [(cudaq.                  |     -   [(                        |
| operators.fermion.FermionOperator | cudaq.operators.spin.SpinOperator |
|                                   |         method)](api/languages/p  |
|        property)](api/languages/p | ython_api.html#cudaq.operators.sp |
| ython_api.html#cudaq.operators.fe | in.SpinOperator.to_sparse_matrix) |
| rmion.FermionOperator.term_count) |     -   [(cuda                    |
|     -                             | q.operators.spin.SpinOperatorTerm |
|  [(cudaq.operators.MatrixOperator |                                   |
|         property)](api/la         |      method)](api/languages/pytho |
| nguages/python_api.html#cudaq.ope | n_api.html#cudaq.operators.spin.S |
| rators.MatrixOperator.term_count) | pinOperatorTerm.to_sparse_matrix) |
|     -   [(                        | -   [to_string()                  |
| cudaq.operators.spin.SpinOperator |     (cudaq.ope                    |
|         property)](api/langu      | rators.boson.BosonOperatorElement |
| ages/python_api.html#cudaq.operat |     method)](api/languages/pyt    |
| ors.spin.SpinOperator.term_count) | hon_api.html#cudaq.operators.boso |
|     -   [(cuda                    | n.BosonOperatorElement.to_string) |
| q.operators.spin.SpinOperatorTerm |     -   [(cudaq.operato           |
|         property)](api/languages  | rs.fermion.FermionOperatorElement |
| /python_api.html#cudaq.operators. |                                   |
| spin.SpinOperatorTerm.term_count) |    method)](api/languages/python_ |
| -   [term_id                      | api.html#cudaq.operators.fermion. |
|     (cudaq.                       | FermionOperatorElement.to_string) |
| operators.boson.BosonOperatorTerm |     -   [(cuda                    |
|     property)](api/language       | q.operators.MatrixOperatorElement |
| s/python_api.html#cudaq.operators |         method)](api/language     |
| .boson.BosonOperatorTerm.term_id) | s/python_api.html#cudaq.operators |
|     -   [(cudaq.oper              | .MatrixOperatorElement.to_string) |
| ators.fermion.FermionOperatorTerm |     -   [(                        |
|                                   | cudaq.operators.spin.SpinOperator |
|       property)](api/languages/py |         method)](api/lang         |
| thon_api.html#cudaq.operators.fer | uages/python_api.html#cudaq.opera |
| mion.FermionOperatorTerm.term_id) | tors.spin.SpinOperator.to_string) |
|     -   [(c                       |     -   [(cudaq.o                 |
| udaq.operators.MatrixOperatorTerm | perators.spin.SpinOperatorElement |
|         property)](api/lan        |         method)](api/languages/p  |
| guages/python_api.html#cudaq.oper | ython_api.html#cudaq.operators.sp |
| ators.MatrixOperatorTerm.term_id) | in.SpinOperatorElement.to_string) |
|     -   [(cuda                    |     -   [(cuda                    |
| q.operators.spin.SpinOperatorTerm | q.operators.spin.SpinOperatorTerm |
|         property)](api/langua     |         method)](api/language     |
| ges/python_api.html#cudaq.operato | s/python_api.html#cudaq.operators |
| rs.spin.SpinOperatorTerm.term_id) | .spin.SpinOperatorTerm.to_string) |
| -   [to_dict() (cudaq.Resources   | -   [translate() (in module       |
|                                   |     cudaq)](api/languages         |
|    method)](api/languages/python_ | /python_api.html#cudaq.translate) |
| api.html#cudaq.Resources.to_dict) | -   [trim()                       |
| -   [to_json()                    |     (cu                           |
|     (                             | daq.operators.boson.BosonOperator |
| cudaq.gradients.CentralDifference |     method)](api/l                |
|     method)](api/la               | anguages/python_api.html#cudaq.op |
| nguages/python_api.html#cudaq.gra | erators.boson.BosonOperator.trim) |
| dients.CentralDifference.to_json) |     -   [(cudaq.                  |
|     -   [(                        | operators.fermion.FermionOperator |
| cudaq.gradients.ForwardDifference |         method)](api/langu        |
|         method)](api/la           | ages/python_api.html#cudaq.operat |
| nguages/python_api.html#cudaq.gra | ors.fermion.FermionOperator.trim) |
| dients.ForwardDifference.to_json) |     -                             |
|     -                             |  [(cudaq.operators.MatrixOperator |
|  [(cudaq.gradients.ParameterShift |         method)](                 |
|         method)](api              | api/languages/python_api.html#cud |
| /languages/python_api.html#cudaq. | aq.operators.MatrixOperator.trim) |
| gradients.ParameterShift.to_json) |     -   [(                        |
|     -   [(                        | cudaq.operators.spin.SpinOperator |
| cudaq.operators.spin.SpinOperator |         method)](api              |
|         method)](api/la           | /languages/python_api.html#cudaq. |
| nguages/python_api.html#cudaq.ope | operators.spin.SpinOperator.trim) |
| rators.spin.SpinOperator.to_json) | -   [type_to_str()                |
|     -   [(cuda                    |     (cudaq.PyKernelDecorator      |
| q.operators.spin.SpinOperatorTerm |     static                        |
|         method)](api/langua       |     method)](                     |
| ges/python_api.html#cudaq.operato | api/languages/python_api.html#cud |
| rs.spin.SpinOperatorTerm.to_json) | aq.PyKernelDecorator.type_to_str) |
|     -   [(cudaq.optimizers.Adam   |                                   |
|         met                       |                                   |
| hod)](api/languages/python_api.ht |                                   |
| ml#cudaq.optimizers.Adam.to_json) |                                   |
|     -   [(cudaq.optimizers.COBYLA |                                   |
|         metho                     |                                   |
| d)](api/languages/python_api.html |                                   |
| #cudaq.optimizers.COBYLA.to_json) |                                   |
|     -   [                         |                                   |
| (cudaq.optimizers.GradientDescent |                                   |
|         method)](api/l            |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.GradientDescent.to_json) |                                   |
|     -   [(cudaq.optimizers.LBFGS  |                                   |
|         meth                      |                                   |
| od)](api/languages/python_api.htm |                                   |
| l#cudaq.optimizers.LBFGS.to_json) |                                   |
|                                   |                                   |
| -   [(cudaq.optimizers.NelderMead |                                   |
|         method)](                 |                                   |
| api/languages/python_api.html#cud |                                   |
| aq.optimizers.NelderMead.to_json) |                                   |
|     -   [(cudaq.optimizers.SGD    |                                   |
|         me                        |                                   |
| thod)](api/languages/python_api.h |                                   |
| tml#cudaq.optimizers.SGD.to_json) |                                   |
|     -   [(cudaq.optimizers.SPSA   |                                   |
|         met                       |                                   |
| hod)](api/languages/python_api.ht |                                   |
| ml#cudaq.optimizers.SPSA.to_json) |                                   |
|     -   [(cudaq.PyKernelDecorator |                                   |
|         metho                     |                                   |
| d)](api/languages/python_api.html |                                   |
| #cudaq.PyKernelDecorator.to_json) |                                   |
| -   [to_matrix()                  |                                   |
|     (cu                           |                                   |
| daq.operators.boson.BosonOperator |                                   |
|     method)](api/langua           |                                   |
| ges/python_api.html#cudaq.operato |                                   |
| rs.boson.BosonOperator.to_matrix) |                                   |
|     -   [(cudaq.ope               |                                   |
| rators.boson.BosonOperatorElement |                                   |
|                                   |                                   |
|        method)](api/languages/pyt |                                   |
| hon_api.html#cudaq.operators.boso |                                   |
| n.BosonOperatorElement.to_matrix) |                                   |
|     -   [(cudaq.                  |                                   |
| operators.boson.BosonOperatorTerm |                                   |
|         method)](api/languages/   |                                   |
| python_api.html#cudaq.operators.b |                                   |
| oson.BosonOperatorTerm.to_matrix) |                                   |
|     -   [(cudaq.                  |                                   |
| operators.fermion.FermionOperator |                                   |
|         method)](api/languages/   |                                   |
| python_api.html#cudaq.operators.f |                                   |
| ermion.FermionOperator.to_matrix) |                                   |
|     -   [(cudaq.operato           |                                   |
| rs.fermion.FermionOperatorElement |                                   |
|                                   |                                   |
|    method)](api/languages/python_ |                                   |
| api.html#cudaq.operators.fermion. |                                   |
| FermionOperatorElement.to_matrix) |                                   |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |                                   |
|                                   |                                   |
|       method)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorTerm.to_matrix) |                                   |
|     -                             |                                   |
|  [(cudaq.operators.MatrixOperator |                                   |
|         method)](api/l            |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| erators.MatrixOperator.to_matrix) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.MatrixOperatorElement |                                   |
|         method)](api/language     |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .MatrixOperatorElement.to_matrix) |                                   |
|     -   [(c                       |                                   |
| udaq.operators.MatrixOperatorTerm |                                   |
|         method)](api/langu        |                                   |
| ages/python_api.html#cudaq.operat |                                   |
| ors.MatrixOperatorTerm.to_matrix) |                                   |
|     -                             |                                   |
|  [(cudaq.operators.ScalarOperator |                                   |
|         method)](api/l            |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| erators.ScalarOperator.to_matrix) |                                   |
|     -   [(                        |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|         method)](api/lang         |                                   |
| uages/python_api.html#cudaq.opera |                                   |
| tors.spin.SpinOperator.to_matrix) |                                   |
|     -   [(cudaq.o                 |                                   |
| perators.spin.SpinOperatorElement |                                   |
|         method)](api/languages/p  |                                   |
| ython_api.html#cudaq.operators.sp |                                   |
| in.SpinOperatorElement.to_matrix) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         method)](api/language     |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .spin.SpinOperatorTerm.to_matrix) |                                   |
+-----------------------------------+-----------------------------------+

## U {#U}

+-----------------------------------------------------------------------+
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
| -   [values() (cudaq.SampleResult | -   [vqe() (in module             |
|                                   |     cudaq)](api/lan               |
|  method)](api/languages/python_ap | guages/python_api.html#cudaq.vqe) |
| i.html#cudaq.SampleResult.values) |                                   |
+-----------------------------------+-----------------------------------+

## X {#X}

+-----------------------------------+-----------------------------------+
| -   [x() (in module               | -   [XError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.x) | ges/python_api.html#cudaq.XError) |
+-----------------------------------+-----------------------------------+

## Y {#Y}

+-----------------------------------+-----------------------------------+
| -   [y() (in module               | -   [YError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.y) | ges/python_api.html#cudaq.YError) |
+-----------------------------------+-----------------------------------+

## Z {#Z}

+-----------------------------------+-----------------------------------+
| -   [z() (in module               | -   [ZError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.z) | ges/python_api.html#cudaq.ZError) |
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
