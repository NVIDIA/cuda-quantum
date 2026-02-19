::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4002
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
            -   [Submission from
                C++](using/backends/cloud/braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](using/backends/cloud/braket.html#submission-from-python){.reference
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
| -   [canonicalize()               | -   [cudaq::pauli1::pauli1 (C++   |
|     (cu                           |     function)](api/languages/cpp_ |
| daq.operators.boson.BosonOperator | api.html#_CPPv4N5cudaq6pauli16pau |
|     method)](api/languages        | li1ERKNSt6vectorIN5cudaq4realEEE) |
| /python_api.html#cudaq.operators. | -   [cudaq::pauli2 (C++           |
| boson.BosonOperator.canonicalize) |     class)](api/languages/cp      |
|     -   [(cudaq.                  | p_api.html#_CPPv4N5cudaq6pauli2E) |
| operators.boson.BosonOperatorTerm | -                                 |
|                                   |    [cudaq::pauli2::num_parameters |
|        method)](api/languages/pyt |     (C++                          |
| hon_api.html#cudaq.operators.boso |     member)]                      |
| n.BosonOperatorTerm.canonicalize) | (api/languages/cpp_api.html#_CPPv |
|     -   [(cudaq.                  | 4N5cudaq6pauli214num_parametersE) |
| operators.fermion.FermionOperator | -   [cudaq::pauli2::num_targets   |
|                                   |     (C++                          |
|        method)](api/languages/pyt |     membe                         |
| hon_api.html#cudaq.operators.ferm | r)](api/languages/cpp_api.html#_C |
| ion.FermionOperator.canonicalize) | PPv4N5cudaq6pauli211num_targetsE) |
|     -   [(cudaq.oper              | -   [cudaq::pauli2::pauli2 (C++   |
| ators.fermion.FermionOperatorTerm |     function)](api/languages/cpp_ |
|                                   | api.html#_CPPv4N5cudaq6pauli26pau |
|    method)](api/languages/python_ | li2ERKNSt6vectorIN5cudaq4realEEE) |
| api.html#cudaq.operators.fermion. | -   [cudaq::phase_damping (C++    |
| FermionOperatorTerm.canonicalize) |                                   |
|     -                             |  class)](api/languages/cpp_api.ht |
|  [(cudaq.operators.MatrixOperator | ml#_CPPv4N5cudaq13phase_dampingE) |
|         method)](api/lang         | -   [cud                          |
| uages/python_api.html#cudaq.opera | aq::phase_damping::num_parameters |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     member)](api/lan              |
| udaq.operators.MatrixOperatorTerm | guages/cpp_api.html#_CPPv4N5cudaq |
|         method)](api/language     | 13phase_damping14num_parametersE) |
| s/python_api.html#cudaq.operators | -   [                             |
| .MatrixOperatorTerm.canonicalize) | cudaq::phase_damping::num_targets |
|     -   [(                        |     (C++                          |
| cudaq.operators.spin.SpinOperator |     member)](api/                 |
|         method)](api/languag      | languages/cpp_api.html#_CPPv4N5cu |
| es/python_api.html#cudaq.operator | daq13phase_damping11num_targetsE) |
| s.spin.SpinOperator.canonicalize) | -   [cudaq::phase_flip_channel    |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     clas                          |
|         method)](api/languages/p  | s)](api/languages/cpp_api.html#_C |
| ython_api.html#cudaq.operators.sp | PPv4N5cudaq18phase_flip_channelE) |
| in.SpinOperatorTerm.canonicalize) | -   [cudaq::p                     |
| -   [canonicalized() (in module   | hase_flip_channel::num_parameters |
|     cuda                          |     (C++                          |
| q.boson)](api/languages/python_ap |     member)](api/language         |
| i.html#cudaq.boson.canonicalized) | s/cpp_api.html#_CPPv4N5cudaq18pha |
|     -   [(in module               | se_flip_channel14num_parametersE) |
|         cudaq.fe                  | -   [cudaq                        |
| rmion)](api/languages/python_api. | ::phase_flip_channel::num_targets |
| html#cudaq.fermion.canonicalized) |     (C++                          |
|     -   [(in module               |     member)](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq18 |
|        cudaq.operators.custom)](a | phase_flip_channel11num_targetsE) |
| pi/languages/python_api.html#cuda | -   [cudaq::product_op (C++       |
| q.operators.custom.canonicalized) |                                   |
|     -   [(in module               |  class)](api/languages/cpp_api.ht |
|         cu                        | ml#_CPPv4I0EN5cudaq10product_opE) |
| daq.spin)](api/languages/python_a | -   [cudaq::product_op::begin     |
| pi.html#cudaq.spin.canonicalized) |     (C++                          |
| -   [CentralDifference (class in  |     functio                       |
|     cudaq.gradients)              | n)](api/languages/cpp_api.html#_C |
| ](api/languages/python_api.html#c | PPv4NK5cudaq10product_op5beginEv) |
| udaq.gradients.CentralDifference) | -                                 |
| -   [clear() (cudaq.Resources     |  [cudaq::product_op::canonicalize |
|     method)](api/languages/pytho  |     (C++                          |
| n_api.html#cudaq.Resources.clear) |     func                          |
|     -   [(cudaq.SampleResult      | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq10product_op12canon |
|   method)](api/languages/python_a | icalizeERKNSt3setINSt6size_tEEE), |
| pi.html#cudaq.SampleResult.clear) |     [\[1\]](api                   |
| -   [COBYLA (class in             | /languages/cpp_api.html#_CPPv4N5c |
|     cudaq.o                       | udaq10product_op12canonicalizeEv) |
| ptimizers)](api/languages/python_ | -   [                             |
| api.html#cudaq.optimizers.COBYLA) | cudaq::product_op::const_iterator |
| -   [coefficient                  |     (C++                          |
|     (cudaq.                       |     struct)](api/                 |
| operators.boson.BosonOperatorTerm | languages/cpp_api.html#_CPPv4N5cu |
|     property)](api/languages/py   | daq10product_op14const_iteratorE) |
| thon_api.html#cudaq.operators.bos | -   [cudaq::product_o             |
| on.BosonOperatorTerm.coefficient) | p::const_iterator::const_iterator |
|     -   [(cudaq.oper              |     (C++                          |
| ators.fermion.FermionOperatorTerm |     fu                            |
|                                   | nction)](api/languages/cpp_api.ht |
|   property)](api/languages/python | ml#_CPPv4N5cudaq10product_op14con |
| _api.html#cudaq.operators.fermion | st_iterator14const_iteratorEPK10p |
| .FermionOperatorTerm.coefficient) | roduct_opI9HandlerTyENSt6size_tE) |
|     -   [(c                       | -   [cudaq::produ                 |
| udaq.operators.MatrixOperatorTerm | ct_op::const_iterator::operator!= |
|         property)](api/languag    |     (C++                          |
| es/python_api.html#cudaq.operator |     fun                           |
| s.MatrixOperatorTerm.coefficient) | ction)](api/languages/cpp_api.htm |
|     -   [(cuda                    | l#_CPPv4NK5cudaq10product_op14con |
| q.operators.spin.SpinOperatorTerm | st_iteratorneERK14const_iterator) |
|         property)](api/languages/ | -   [cudaq::produ                 |
| python_api.html#cudaq.operators.s | ct_op::const_iterator::operator\* |
| pin.SpinOperatorTerm.coefficient) |     (C++                          |
| -   [col_count                    |     function)](api/lang           |
|     (cudaq.KrausOperator          | uages/cpp_api.html#_CPPv4NK5cudaq |
|     prope                         | 10product_op14const_iteratormlEv) |
| rty)](api/languages/python_api.ht | -   [cudaq::produ                 |
| ml#cudaq.KrausOperator.col_count) | ct_op::const_iterator::operator++ |
| -   [ComplexMatrix (class in      |     (C++                          |
|     cudaq)](api/languages/pyt     |     function)](api/lang           |
| hon_api.html#cudaq.ComplexMatrix) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [compute()                    | 0product_op14const_iteratorppEi), |
|     (                             |     [\[1\]](api/lan               |
| cudaq.gradients.CentralDifference | guages/cpp_api.html#_CPPv4N5cudaq |
|     method)](api/la               | 10product_op14const_iteratorppEv) |
| nguages/python_api.html#cudaq.gra | -   [cudaq::produc                |
| dients.CentralDifference.compute) | t_op::const_iterator::operator\-- |
|     -   [(                        |     (C++                          |
| cudaq.gradients.ForwardDifference |     function)](api/lang           |
|         method)](api/la           | uages/cpp_api.html#_CPPv4N5cudaq1 |
| nguages/python_api.html#cudaq.gra | 0product_op14const_iteratormmEi), |
| dients.ForwardDifference.compute) |     [\[1\]](api/lan               |
|     -                             | guages/cpp_api.html#_CPPv4N5cudaq |
|  [(cudaq.gradients.ParameterShift | 10product_op14const_iteratormmEv) |
|         method)](api              | -   [cudaq::produc                |
| /languages/python_api.html#cudaq. | t_op::const_iterator::operator-\> |
| gradients.ParameterShift.compute) |     (C++                          |
| -   [const()                      |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|   (cudaq.operators.ScalarOperator | 10product_op14const_iteratorptEv) |
|     class                         | -   [cudaq::produ                 |
|     method)](a                    | ct_op::const_iterator::operator== |
| pi/languages/python_api.html#cuda |     (C++                          |
| q.operators.ScalarOperator.const) |     fun                           |
| -   [copy()                       | ction)](api/languages/cpp_api.htm |
|     (cu                           | l#_CPPv4NK5cudaq10product_op14con |
| daq.operators.boson.BosonOperator | st_iteratoreqERK14const_iterator) |
|     method)](api/l                | -   [cudaq::product_op::degrees   |
| anguages/python_api.html#cudaq.op |     (C++                          |
| erators.boson.BosonOperator.copy) |     function)                     |
|     -   [(cudaq.                  | ](api/languages/cpp_api.html#_CPP |
| operators.boson.BosonOperatorTerm | v4NK5cudaq10product_op7degreesEv) |
|         method)](api/langu        | -   [cudaq::product_op::dump (C++ |
| ages/python_api.html#cudaq.operat |     functi                        |
| ors.boson.BosonOperatorTerm.copy) | on)](api/languages/cpp_api.html#_ |
|     -   [(cudaq.                  | CPPv4NK5cudaq10product_op4dumpEv) |
| operators.fermion.FermionOperator | -   [cudaq::product_op::end (C++  |
|         method)](api/langu        |     funct                         |
| ages/python_api.html#cudaq.operat | ion)](api/languages/cpp_api.html# |
| ors.fermion.FermionOperator.copy) | _CPPv4NK5cudaq10product_op3endEv) |
|     -   [(cudaq.oper              | -   [c                            |
| ators.fermion.FermionOperatorTerm | udaq::product_op::get_coefficient |
|         method)](api/languages    |     (C++                          |
| /python_api.html#cudaq.operators. |     function)](api/lan            |
| fermion.FermionOperatorTerm.copy) | guages/cpp_api.html#_CPPv4NK5cuda |
|     -                             | q10product_op15get_coefficientEv) |
|  [(cudaq.operators.MatrixOperator | -                                 |
|         method)](                 |   [cudaq::product_op::get_term_id |
| api/languages/python_api.html#cud |     (C++                          |
| aq.operators.MatrixOperator.copy) |     function)](api                |
|     -   [(c                       | /languages/cpp_api.html#_CPPv4NK5 |
| udaq.operators.MatrixOperatorTerm | cudaq10product_op11get_term_idEv) |
|         method)](api/             | -                                 |
| languages/python_api.html#cudaq.o |   [cudaq::product_op::is_identity |
| perators.MatrixOperatorTerm.copy) |     (C++                          |
|     -   [(                        |     function)](api                |
| cudaq.operators.spin.SpinOperator | /languages/cpp_api.html#_CPPv4NK5 |
|         method)](api              | cudaq10product_op11is_identityEv) |
| /languages/python_api.html#cudaq. | -   [cudaq::product_op::num_ops   |
| operators.spin.SpinOperator.copy) |     (C++                          |
|     -   [(cuda                    |     function)                     |
| q.operators.spin.SpinOperatorTerm | ](api/languages/cpp_api.html#_CPP |
|         method)](api/lan          | v4NK5cudaq10product_op7num_opsEv) |
| guages/python_api.html#cudaq.oper | -                                 |
| ators.spin.SpinOperatorTerm.copy) |    [cudaq::product_op::operator\* |
| -   [count() (cudaq.Resources     |     (C++                          |
|     method)](api/languages/pytho  |     function)](api/languages/     |
| n_api.html#cudaq.Resources.count) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(cudaq.SampleResult      | oduct_opmlE10product_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|   method)](api/languages/python_a |     [\[1\]](api/languages/        |
| pi.html#cudaq.SampleResult.count) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [count_controls()             | oduct_opmlE10product_opI1TERK15sc |
|     (cudaq.Resources              | alar_operatorRR10product_opI1TE), |
|     meth                          |     [\[2\]](api/languages/        |
| od)](api/languages/python_api.htm | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| l#cudaq.Resources.count_controls) | oduct_opmlE10product_opI1TERR15sc |
| -   [counts()                     | alar_operatorRK10product_opI1TE), |
|     (cudaq.ObserveResult          |     [\[3\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| method)](api/languages/python_api | oduct_opmlE10product_opI1TERR15sc |
| .html#cudaq.ObserveResult.counts) | alar_operatorRR10product_opI1TE), |
| -   [create() (in module          |     [\[4\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|    cudaq.boson)](api/languages/py | 5cudaq10product_opmlE6sum_opI1TER |
| thon_api.html#cudaq.boson.create) | K15scalar_operatorRK6sum_opI1TE), |
|     -   [(in module               |     [\[5\]](api/                  |
|         c                         | languages/cpp_api.html#_CPPv4I0EN |
| udaq.fermion)](api/languages/pyth | 5cudaq10product_opmlE6sum_opI1TER |
| on_api.html#cudaq.fermion.create) | K15scalar_operatorRR6sum_opI1TE), |
| -   [csr_spmatrix (C++            |     [\[6\]](api/                  |
|     type)](api/languages/c        | languages/cpp_api.html#_CPPv4I0EN |
| pp_api.html#_CPPv412csr_spmatrix) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq                         | R15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/langua       |     [\[7\]](api/                  |
| ges/python_api.html#module-cudaq) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq (C++                   | 5cudaq10product_opmlE6sum_opI1TER |
|     type)](api/lan                | R15scalar_operatorRR6sum_opI1TE), |
| guages/cpp_api.html#_CPPv45cudaq) |     [\[8\]](api/languages         |
| -   [cudaq.apply_noise() (in      | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     module                        | duct_opmlERK6sum_opI9HandlerTyE), |
|     cudaq)](api/languages/python_ |     [\[9\]](api/languages/cpp_a   |
| api.html#cudaq.cudaq.apply_noise) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.boson                   | opmlERK10product_opI9HandlerTyE), |
|     -   [module](api/languages/py |     [\[10\]](api/language         |
| thon_api.html#module-cudaq.boson) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   cudaq.fermion                 | roduct_opmlERK15scalar_operator), |
|                                   |     [\[11\]](api/languages/cpp_a  |
|   -   [module](api/languages/pyth | pi.html#_CPPv4NKR5cudaq10product_ |
| on_api.html#module-cudaq.fermion) | opmlERR10product_opI9HandlerTyE), |
| -   cudaq.operators.custom        |     [\[12\]](api/language         |
|     -   [mo                       | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| dule](api/languages/python_api.ht | roduct_opmlERR15scalar_operator), |
| ml#module-cudaq.operators.custom) |     [\[13\]](api/languages/cpp_   |
| -   cudaq.spin                    | api.html#_CPPv4NO5cudaq10product_ |
|     -   [module](api/languages/p  | opmlERK10product_opI9HandlerTyE), |
| ython_api.html#module-cudaq.spin) |     [\[14\]](api/languag          |
| -   [cudaq::amplitude_damping     | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmlERK15scalar_operator), |
|     cla                           |     [\[15\]](api/languages/cpp_   |
| ss)](api/languages/cpp_api.html#_ | api.html#_CPPv4NO5cudaq10product_ |
| CPPv4N5cudaq17amplitude_dampingE) | opmlERR10product_opI9HandlerTyE), |
| -                                 |     [\[16\]](api/langua           |
| [cudaq::amplitude_damping_channel | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     (C++                          | product_opmlERR15scalar_operator) |
|     class)](api                   | -                                 |
| /languages/cpp_api.html#_CPPv4N5c |   [cudaq::product_op::operator\*= |
| udaq25amplitude_damping_channelE) |     (C++                          |
| -   [cudaq::amplitud              |     function)](api/languages/cpp  |
| e_damping_channel::num_parameters | _api.html#_CPPv4N5cudaq10product_ |
|     (C++                          | opmLERK10product_opI9HandlerTyE), |
|     member)](api/languages/cpp_a  |     [\[1\]](api/langua            |
| pi.html#_CPPv4N5cudaq25amplitude_ | ges/cpp_api.html#_CPPv4N5cudaq10p |
| damping_channel14num_parametersE) | roduct_opmLERK15scalar_operator), |
| -   [cudaq::ampli                 |     [\[2\]](api/languages/cp      |
| tude_damping_channel::num_targets | p_api.html#_CPPv4N5cudaq10product |
|     (C++                          | _opmLERR10product_opI9HandlerTyE) |
|     member)](api/languages/cp     | -   [cudaq::product_op::operator+ |
| p_api.html#_CPPv4N5cudaq25amplitu |     (C++                          |
| de_damping_channel11num_targetsE) |     function)](api/langu          |
| -   [cudaq::AnalogRemoteRESTQPU   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opplE6sum_opI1TERK15sc |
|     class                         | alar_operatorRK10product_opI1TE), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/                  |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::apply_noise (C++      | 5cudaq10product_opplE6sum_opI1TER |
|     function)](api/               | K15scalar_operatorRK6sum_opI1TE), |
| languages/cpp_api.html#_CPPv4I0Dp |     [\[2\]](api/langu             |
| EN5cudaq11apply_noiseEvDpRR4Args) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::async_result (C++     | q10product_opplE6sum_opI1TERK15sc |
|     c                             | alar_operatorRR10product_opI1TE), |
| lass)](api/languages/cpp_api.html |     [\[3\]](api/                  |
| #_CPPv4I0EN5cudaq12async_resultE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::async_result::get     | 5cudaq10product_opplE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRR6sum_opI1TE), |
|     functi                        |     [\[4\]](api/langu             |
| on)](api/languages/cpp_api.html#_ | ages/cpp_api.html#_CPPv4I0EN5cuda |
| CPPv4N5cudaq12async_result3getEv) | q10product_opplE6sum_opI1TERR15sc |
| -   [cudaq::async_sample_result   | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[5\]](api/                  |
|     type                          | languages/cpp_api.html#_CPPv4I0EN |
| )](api/languages/cpp_api.html#_CP | 5cudaq10product_opplE6sum_opI1TER |
| Pv4N5cudaq19async_sample_resultE) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::BaseRemoteRESTQPU     |     [\[6\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cla                           | q10product_opplE6sum_opI1TERR15sc |
| ss)](api/languages/cpp_api.html#_ | alar_operatorRR10product_opI1TE), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [\[7\]](api/                  |
| -                                 | languages/cpp_api.html#_CPPv4I0EN |
|    [cudaq::BaseRemoteSimulatorQPU | 5cudaq10product_opplE6sum_opI1TER |
|     (C++                          | R15scalar_operatorRR6sum_opI1TE), |
|     class)](                      |     [\[8\]](api/languages/cpp_a   |
| api/languages/cpp_api.html#_CPPv4 | pi.html#_CPPv4NKR5cudaq10product_ |
| N5cudaq22BaseRemoteSimulatorQPUE) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq::bit_flip_channel (C++ |     [\[9\]](api/language          |
|     cl                            | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ass)](api/languages/cpp_api.html# | roduct_opplERK15scalar_operator), |
| _CPPv4N5cudaq16bit_flip_channelE) |     [\[10\]](api/languages/       |
| -   [cudaq:                       | cpp_api.html#_CPPv4NKR5cudaq10pro |
| :bit_flip_channel::num_parameters | duct_opplERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     member)](api/langua           | pi.html#_CPPv4NKR5cudaq10product_ |
| ges/cpp_api.html#_CPPv4N5cudaq16b | opplERR10product_opI9HandlerTyE), |
| it_flip_channel14num_parametersE) |     [\[12\]](api/language         |
| -   [cud                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| aq::bit_flip_channel::num_targets | roduct_opplERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     member)](api/lan              | cpp_api.html#_CPPv4NKR5cudaq10pro |
| guages/cpp_api.html#_CPPv4N5cudaq | duct_opplERR6sum_opI9HandlerTyE), |
| 16bit_flip_channel11num_targetsE) |     [\[                           |
| -   [cudaq::boson_handler (C++    | 14\]](api/languages/cpp_api.html# |
|                                   | _CPPv4NKR5cudaq10product_opplEv), |
|  class)](api/languages/cpp_api.ht |     [\[15\]](api/languages/cpp_   |
| ml#_CPPv4N5cudaq13boson_handlerE) | api.html#_CPPv4NO5cudaq10product_ |
| -   [cudaq::boson_op (C++         | opplERK10product_opI9HandlerTyE), |
|     type)](api/languages/cpp_     |     [\[16\]](api/languag          |
| api.html#_CPPv4N5cudaq8boson_opE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::boson_op_term (C++    | roduct_opplERK15scalar_operator), |
|                                   |     [\[17\]](api/languages        |
|   type)](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ml#_CPPv4N5cudaq13boson_op_termE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq::CodeGenConfig (C++    |     [\[18\]](api/languages/cpp_   |
|                                   | api.html#_CPPv4NO5cudaq10product_ |
| struct)](api/languages/cpp_api.ht | opplERR10product_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     [\[19\]](api/languag          |
| -   [cudaq::commutation_relations | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opplERR15scalar_operator), |
|     struct)]                      |     [\[20\]](api/languages        |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4NO5cudaq10pro |
| 4N5cudaq21commutation_relationsE) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cudaq::complex (C++          |     [                             |
|     type)](api/languages/cpp      | \[21\]](api/languages/cpp_api.htm |
| _api.html#_CPPv4N5cudaq7complexE) | l#_CPPv4NO5cudaq10product_opplEv) |
| -   [cudaq::complex_matrix (C++   | -   [cudaq::product_op::operator- |
|                                   |     (C++                          |
| class)](api/languages/cpp_api.htm |     function)](api/langu          |
| l#_CPPv4N5cudaq14complex_matrixE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -                                 | q10product_opmiE6sum_opI1TERK15sc |
|   [cudaq::complex_matrix::adjoint | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     function)](a                  | languages/cpp_api.html#_CPPv4I0EN |
| pi/languages/cpp_api.html#_CPPv4N | 5cudaq10product_opmiE6sum_opI1TER |
| 5cudaq14complex_matrix7adjointEv) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::                      |     [\[2\]](api/langu             |
| complex_matrix::diagonal_elements | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERK15sc |
|     function)](api/languages      | alar_operatorRR10product_opI1TE), |
| /cpp_api.html#_CPPv4NK5cudaq14com |     [\[3\]](api/                  |
| plex_matrix17diagonal_elementsEi) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::complex_matrix::dump  | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRR6sum_opI1TE), |
|     function)](api/language       |     [\[4\]](api/langu             |
| s/cpp_api.html#_CPPv4NK5cudaq14co | ages/cpp_api.html#_CPPv4I0EN5cuda |
| mplex_matrix4dumpERNSt7ostreamE), | q10product_opmiE6sum_opI1TERR15sc |
|     [\[1\]]                       | alar_operatorRK10product_opI1TE), |
| (api/languages/cpp_api.html#_CPPv |     [\[5\]](api/                  |
| 4NK5cudaq14complex_matrix4dumpEv) | languages/cpp_api.html#_CPPv4I0EN |
| -   [c                            | 5cudaq10product_opmiE6sum_opI1TER |
| udaq::complex_matrix::eigenvalues | R15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[6\]](api/langu             |
|     function)](api/lan            | ages/cpp_api.html#_CPPv4I0EN5cuda |
| guages/cpp_api.html#_CPPv4NK5cuda | q10product_opmiE6sum_opI1TERR15sc |
| q14complex_matrix11eigenvaluesEv) | alar_operatorRR10product_opI1TE), |
| -   [cu                           |     [\[7\]](api/                  |
| daq::complex_matrix::eigenvectors | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/lang           | R15scalar_operatorRR6sum_opI1TE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[8\]](api/languages/cpp_a   |
| 14complex_matrix12eigenvectorsEv) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [c                            | opmiERK10product_opI9HandlerTyE), |
| udaq::complex_matrix::exponential |     [\[9\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/la             | roduct_opmiERK15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[10\]](api/languages/       |
| q14complex_matrix11exponentialEv) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -                                 | duct_opmiERK6sum_opI9HandlerTyE), |
|  [cudaq::complex_matrix::identity |     [\[11\]](api/languages/cpp_a  |
|     (C++                          | pi.html#_CPPv4NKR5cudaq10product_ |
|     function)](api/languages      | opmiERR10product_opI9HandlerTyE), |
| /cpp_api.html#_CPPv4N5cudaq14comp |     [\[12\]](api/language         |
| lex_matrix8identityEKNSt6size_tE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -                                 | roduct_opmiERR15scalar_operator), |
| [cudaq::complex_matrix::kronecker |     [\[13\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     function)](api/lang           | duct_opmiERR6sum_opI9HandlerTyE), |
| uages/cpp_api.html#_CPPv4I00EN5cu |     [\[                           |
| daq14complex_matrix9kroneckerE14c | 14\]](api/languages/cpp_api.html# |
| omplex_matrix8Iterable8Iterable), | _CPPv4NKR5cudaq10product_opmiEv), |
|     [\[1\]](api/l                 |     [\[15\]](api/languages/cpp_   |
| anguages/cpp_api.html#_CPPv4N5cud | api.html#_CPPv4NO5cudaq10product_ |
| aq14complex_matrix9kroneckerERK14 | opmiERK10product_opI9HandlerTyE), |
| complex_matrixRK14complex_matrix) |     [\[16\]](api/languag          |
| -   [cudaq::c                     | es/cpp_api.html#_CPPv4NO5cudaq10p |
| omplex_matrix::minimal_eigenvalue | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[17\]](api/languages        |
|     function)](api/languages/     | /cpp_api.html#_CPPv4NO5cudaq10pro |
| cpp_api.html#_CPPv4NK5cudaq14comp | duct_opmiERK6sum_opI9HandlerTyE), |
| lex_matrix18minimal_eigenvalueEv) |     [\[18\]](api/languages/cpp_   |
| -   [                             | api.html#_CPPv4NO5cudaq10product_ |
| cudaq::complex_matrix::operator() | opmiERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[19\]](api/languag          |
|     function)](api/languages/cpp  | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _api.html#_CPPv4N5cudaq14complex_ | roduct_opmiERR15scalar_operator), |
| matrixclENSt6size_tENSt6size_tE), |     [\[20\]](api/languages        |
|     [\[1\]](api/languages/cpp     | /cpp_api.html#_CPPv4NO5cudaq10pro |
| _api.html#_CPPv4NK5cudaq14complex | duct_opmiERR6sum_opI9HandlerTyE), |
| _matrixclENSt6size_tENSt6size_tE) |     [                             |
| -   [                             | \[21\]](api/languages/cpp_api.htm |
| cudaq::complex_matrix::operator\* | l#_CPPv4NO5cudaq10product_opmiEv) |
|     (C++                          | -   [cudaq::product_op::operator/ |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     function)](api/language       |
| omplex_matrixmlEN14complex_matrix | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| 10value_typeERK14complex_matrix), | roduct_opdvERK15scalar_operator), |
|     [\[1\]                        |     [\[1\]](api/language          |
| ](api/languages/cpp_api.html#_CPP | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| v4N5cudaq14complex_matrixmlERK14c | roduct_opdvERR15scalar_operator), |
| omplex_matrixRK14complex_matrix), |     [\[2\]](api/languag           |
|                                   | es/cpp_api.html#_CPPv4NO5cudaq10p |
|  [\[2\]](api/languages/cpp_api.ht | roduct_opdvERK15scalar_operator), |
| ml#_CPPv4N5cudaq14complex_matrixm |     [\[3\]](api/langua            |
| lERK14complex_matrixRKNSt6vectorI | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| N14complex_matrix10value_typeEEE) | product_opdvERR15scalar_operator) |
| -                                 | -                                 |
| [cudaq::complex_matrix::operator+ |    [cudaq::product_op::operator/= |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/langu          |
| )](api/languages/cpp_api.html#_CP | ages/cpp_api.html#_CPPv4N5cudaq10 |
| Pv4N5cudaq14complex_matrixplERK14 | product_opdVERK15scalar_operator) |
| complex_matrixRK14complex_matrix) | -   [cudaq::product_op::operator= |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::operator- |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4I0_NSt |
|     function                      | 11enable_if_tIXaantNSt7is_sameI1T |
| )](api/languages/cpp_api.html#_CP | 9HandlerTyE5valueENSt16is_constru |
| Pv4N5cudaq14complex_matrixmiERK14 | ctibleI9HandlerTy1TE5valueEEbEEEN |
| complex_matrixRK14complex_matrix) | 5cudaq10product_opaSER10product_o |
| -   [cu                           | pI9HandlerTyERK10product_opI1TE), |
| daq::complex_matrix::operator\[\] |     [\[1\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4N5cudaq10product_ |
|                                   | opaSERK10product_opI9HandlerTyE), |
|  function)](api/languages/cpp_api |     [\[2\]](api/languages/cp      |
| .html#_CPPv4N5cudaq14complex_matr | p_api.html#_CPPv4N5cudaq10product |
| ixixERKNSt6vectorINSt6size_tEEE), | _opaSERR10product_opI9HandlerTyE) |
|     [\[1\]](api/languages/cpp_api | -                                 |
| .html#_CPPv4NK5cudaq14complex_mat |    [cudaq::product_op::operator== |
| rixixERKNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq::complex_matrix::power |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq10product |
|     function)]                    | _opeqERK10product_opI9HandlerTyE) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq14complex_matrix5powerEi) |  [cudaq::product_op::operator\[\] |
| -                                 |     (C++                          |
|  [cudaq::complex_matrix::set_zero |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     function)](ap                 | 5cudaq10product_opixENSt6size_tE) |
| i/languages/cpp_api.html#_CPPv4N5 | -                                 |
| cudaq14complex_matrix8set_zeroEv) |    [cudaq::product_op::product_op |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::to_string |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4I0_NSt11enable_ |
|     function)](api/               | if_tIXaaNSt7is_sameI9HandlerTy14m |
| languages/cpp_api.html#_CPPv4NK5c | atrix_handlerE5valueEaantNSt7is_s |
| udaq14complex_matrix9to_stringEv) | ameI1T9HandlerTyE5valueENSt16is_c |
| -   [                             | onstructibleI9HandlerTy1TE5valueE |
| cudaq::complex_matrix::value_type | EbEEEN5cudaq10product_op10product |
|     (C++                          | _opERK10product_opI1TERKN14matrix |
|     type)](api/                   | _handler20commutation_behaviorE), |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq14complex_matrix10value_typeE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::contrib (C++          | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|     type)](api/languages/cpp      | tNSt7is_sameI1T9HandlerTyE5valueE |
| _api.html#_CPPv4N5cudaq7contribE) | NSt16is_constructibleI9HandlerTy1 |
| -   [cudaq::contrib::draw (C++    | TE5valueEEbEEEN5cudaq10product_op |
|     function)                     | 10product_opERK10product_opI1TE), |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4I0DpEN5cudaq7contrib4drawENSt6s |   [\[2\]](api/languages/cpp_api.h |
| tringERR13QuantumKernelDpRR4Args) | tml#_CPPv4N5cudaq10product_op10pr |
| -                                 | oduct_opENSt6size_tENSt6size_tE), |
| [cudaq::contrib::get_unitary_cmat |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     function)](api/languages/cp   | _op10product_opENSt7complexIdEE), |
| p_api.html#_CPPv4I0DpEN5cudaq7con |     [\[4\]](api/l                 |
| trib16get_unitary_cmatE14complex_ | anguages/cpp_api.html#_CPPv4N5cud |
| matrixRR13QuantumKernelDpRR4Args) | aq10product_op10product_opERK10pr |
| -   [cudaq::CusvState (C++        | oduct_opI9HandlerTyENSt6size_tE), |
|                                   |     [\[5\]](api/l                 |
|    class)](api/languages/cpp_api. | anguages/cpp_api.html#_CPPv4N5cud |
| html#_CPPv4I0EN5cudaq9CusvStateE) | aq10product_op10product_opERR10pr |
| -   [cudaq::depolarization1 (C++  | oduct_opI9HandlerTyENSt6size_tE), |
|     c                             |     [\[6\]](api/languages         |
| lass)](api/languages/cpp_api.html | /cpp_api.html#_CPPv4N5cudaq10prod |
| #_CPPv4N5cudaq15depolarization1E) | uct_op10product_opERR9HandlerTy), |
| -   [cudaq::depolarization2 (C++  |     [\[7\]](ap                    |
|     c                             | i/languages/cpp_api.html#_CPPv4N5 |
| lass)](api/languages/cpp_api.html | cudaq10product_op10product_opEd), |
| #_CPPv4N5cudaq15depolarization2E) |     [\[8\]](a                     |
| -   [cudaq:                       | pi/languages/cpp_api.html#_CPPv4N |
| :depolarization2::depolarization2 | 5cudaq10product_op10product_opEv) |
|     (C++                          | -   [cuda                         |
|     function)](api/languages/cp   | q::product_op::to_diagonal_matrix |
| p_api.html#_CPPv4N5cudaq15depolar |     (C++                          |
| ization215depolarization2EK4real) |     function)](api/               |
| -   [cudaq                        | languages/cpp_api.html#_CPPv4NK5c |
| ::depolarization2::num_parameters | udaq10product_op18to_diagonal_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     member)](api/langu            | ENSt7int64_tEEERKNSt13unordered_m |
| ages/cpp_api.html#_CPPv4N5cudaq15 | apINSt6stringENSt7complexIdEEEEb) |
| depolarization214num_parametersE) | -   [cudaq::product_op::to_matrix |
| -   [cu                           |     (C++                          |
| daq::depolarization2::num_targets |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     member)](api/la               | _CPPv4NK5cudaq10product_op9to_mat |
| nguages/cpp_api.html#_CPPv4N5cuda | rixENSt13unordered_mapINSt6size_t |
| q15depolarization211num_targetsE) | ENSt7int64_tEEERKNSt13unordered_m |
| -                                 | apINSt6stringENSt7complexIdEEEEb) |
|    [cudaq::depolarization_channel | -   [cu                           |
|     (C++                          | daq::product_op::to_sparse_matrix |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](ap                 |
| N5cudaq22depolarization_channelE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::depol                 | 5cudaq10product_op16to_sparse_mat |
| arization_channel::num_parameters | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     member)](api/languages/cp     | apINSt6stringENSt7complexIdEEEEb) |
| p_api.html#_CPPv4N5cudaq22depolar | -   [cudaq::product_op::to_string |
| ization_channel14num_parametersE) |     (C++                          |
| -   [cudaq::de                    |     function)](                   |
| polarization_channel::num_targets | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq10product_op9to_stringEv) |
|     member)](api/languages        | -                                 |
| /cpp_api.html#_CPPv4N5cudaq22depo |  [cudaq::product_op::\~product_op |
| larization_channel11num_targetsE) |     (C++                          |
| -   [cudaq::details (C++          |     fu                            |
|     type)](api/languages/cpp      | nction)](api/languages/cpp_api.ht |
| _api.html#_CPPv4N5cudaq7detailsE) | ml#_CPPv4N5cudaq10product_opD0Ev) |
| -   [cudaq::details::future (C++  | -   [cudaq::QPU (C++              |
|                                   |     class)](api/languages         |
|  class)](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| ml#_CPPv4N5cudaq7details6futureE) | -   [cudaq::QPU::beginExecution   |
| -                                 |     (C++                          |
|   [cudaq::details::future::future |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     functio                       | Pv4N5cudaq3QPU14beginExecutionEv) |
| n)](api/languages/cpp_api.html#_C | -   [cuda                         |
| PPv4N5cudaq7details6future6future | q::QPU::configureExecutionContext |
| ERNSt6vectorI3JobEERNSt6stringERN |     (C++                          |
| St3mapINSt6stringENSt6stringEEE), |     funct                         |
|     [\[1\]](api/lang              | ion)](api/languages/cpp_api.html# |
| uages/cpp_api.html#_CPPv4N5cudaq7 | _CPPv4NK5cudaq3QPU25configureExec |
| details6future6futureERR6future), | utionContextER16ExecutionContext) |
|     [\[2\]]                       | -   [cudaq::QPU::endExecution     |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq7details6future6futureEv) |     functi                        |
| -   [cu                           | on)](api/languages/cpp_api.html#_ |
| daq::details::kernel_builder_base | CPPv4N5cudaq3QPU12endExecutionEv) |
|     (C++                          | -   [cudaq::QPU::enqueue (C++     |
|     class)](api/l                 |     function)](ap                 |
| anguages/cpp_api.html#_CPPv4N5cud | i/languages/cpp_api.html#_CPPv4N5 |
| aq7details19kernel_builder_baseE) | cudaq3QPU7enqueueER11QuantumTask) |
| -   [cudaq::details::             | -   [cud                          |
| kernel_builder_base::operator\<\< | aq::QPU::finalizeExecutionContext |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     func                          |
| ges/cpp_api.html#_CPPv4N5cudaq7de | tion)](api/languages/cpp_api.html |
| tails19kernel_builder_baselsERNSt | #_CPPv4NK5cudaq3QPU24finalizeExec |
| 7ostreamERK19kernel_builder_base) | utionContextER16ExecutionContext) |
| -   [                             | -   [cudaq::QPU::getConnectivity  |
| cudaq::details::KernelBuilderType |     (C++                          |
|     (C++                          |     function)                     |
|     class)](api                   | ](api/languages/cpp_api.html#_CPP |
| /languages/cpp_api.html#_CPPv4N5c | v4N5cudaq3QPU15getConnectivityEv) |
| udaq7details17KernelBuilderTypeE) | -                                 |
| -   [cudaq::d                     | [cudaq::QPU::getExecutionThreadId |
| etails::KernelBuilderType::create |     (C++                          |
|     (C++                          |     function)](api/               |
|     function)                     | languages/cpp_api.html#_CPPv4NK5c |
| ](api/languages/cpp_api.html#_CPP | udaq3QPU20getExecutionThreadIdEv) |
| v4N5cudaq7details17KernelBuilderT | -   [cudaq::QPU::getNumQubits     |
| ype6createEPN4mlir11MLIRContextE) |     (C++                          |
| -   [cudaq::details::Ker          |     functi                        |
| nelBuilderType::KernelBuilderType | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     function)](api/lang           | -   [                             |
| uages/cpp_api.html#_CPPv4N5cudaq7 | cudaq::QPU::getRemoteCapabilities |
| details17KernelBuilderType17Kerne |     (C++                          |
| lBuilderTypeERRNSt8functionIFN4ml |     function)](api/l              |
| ir4TypeEPN4mlir11MLIRContextEEEE) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cudaq::diag_matrix_callback  | daq3QPU21getRemoteCapabilitiesEv) |
|     (C++                          | -   [cudaq::QPU::isEmulated (C++  |
|     class)                        |     func                          |
| ](api/languages/cpp_api.html#_CPP | tion)](api/languages/cpp_api.html |
| v4N5cudaq20diag_matrix_callbackE) | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| -   [cudaq::dyn (C++              | -   [cudaq::QPU::isSimulator (C++ |
|     member)](api/languages        |     funct                         |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::ExecutionContext (C++ | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     cl                            | -   [cudaq::QPU::launchKernel     |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16ExecutionContextE) |     function)](api/               |
| -   [cudaq                        | languages/cpp_api.html#_CPPv4N5cu |
| ::ExecutionContext::amplitudeMaps | daq3QPU12launchKernelERKNSt6strin |
|     (C++                          | gE15KernelThunkTypePvNSt8uint64_t |
|     member)](api/langu            | ENSt8uint64_tERKNSt6vectorIPvEE), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |                                   |
| ExecutionContext13amplitudeMapsE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [c                            | ml#_CPPv4N5cudaq3QPU12launchKerne |
| udaq::ExecutionContext::asyncExec | lERKNSt6stringERKNSt6vectorIPvEE) |
|     (C++                          | -   [cudaq::QPU::onRandomSeedSet  |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/lang           |
| daq16ExecutionContext9asyncExecE) | uages/cpp_api.html#_CPPv4N5cudaq3 |
| -   [cud                          | QPU15onRandomSeedSetENSt6size_tE) |
| aq::ExecutionContext::asyncResult | -   [cudaq::QPU::QPU (C++         |
|     (C++                          |     functio                       |
|     member)](api/lan              | n)](api/languages/cpp_api.html#_C |
| guages/cpp_api.html#_CPPv4N5cudaq | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| 16ExecutionContext11asyncResultE) |                                   |
| -   [cudaq:                       |  [\[1\]](api/languages/cpp_api.ht |
| :ExecutionContext::batchIteration | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     (C++                          |     [\[2\]](api/languages/cpp_    |
|     member)](api/langua           | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| ges/cpp_api.html#_CPPv4N5cudaq16E | -   [cudaq::QPU::setId (C++       |
| xecutionContext14batchIterationE) |     function                      |
| -   [cudaq::E                     | )](api/languages/cpp_api.html#_CP |
| xecutionContext::canHandleObserve | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::setShots (C++    |
|     member)](api/language         |     f                             |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | unction)](api/languages/cpp_api.h |
| cutionContext16canHandleObserveE) | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| -   [cudaq::E                     | -   [cudaq:                       |
| xecutionContext::ExecutionContext | :QPU::supportsConditionalFeedback |
|     (C++                          |     (C++                          |
|     func                          |     function)](api/langua         |
| tion)](api/languages/cpp_api.html | ges/cpp_api.html#_CPPv4N5cudaq3QP |
| #_CPPv4N5cudaq16ExecutionContext1 | U27supportsConditionalFeedbackEv) |
| 6ExecutionContextERKNSt6stringE), | -   [cudaq::                      |
|     [\[1\]](api/languages/        | QPU::supportsExplicitMeasurements |
| cpp_api.html#_CPPv4N5cudaq16Execu |     (C++                          |
| tionContext16ExecutionContextERKN |     function)](api/languag        |
| St6stringENSt6size_tENSt6size_tE) | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| -   [cudaq::E                     | 28supportsExplicitMeasurementsEv) |
| xecutionContext::expectationValue | -   [cudaq::QPU::\~QPU (C++       |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api/language         | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::QPUState (C++         |
| cutionContext16expectationValueE) |     class)](api/languages/cpp_    |
| -   [cudaq::Execu                 | api.html#_CPPv4N5cudaq8QPUStateE) |
| tionContext::explicitMeasurements | -   [cudaq::qreg (C++             |
|     (C++                          |     class)](api/lan               |
|     member)](api/languages/cp     | guages/cpp_api.html#_CPPv4I_NSt6s |
| p_api.html#_CPPv4N5cudaq16Executi | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| onContext20explicitMeasurementsE) | -   [cudaq::qreg::back (C++       |
| -   [cuda                         |     function)                     |
| q::ExecutionContext::futureResult | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4backENSt6size_tE), |
|     member)](api/lang             |     [\[1\]](api/languages/cpp_ap  |
| uages/cpp_api.html#_CPPv4N5cudaq1 | i.html#_CPPv4N5cudaq4qreg4backEv) |
| 6ExecutionContext12futureResultE) | -   [cudaq::qreg::begin (C++      |
| -   [cudaq::ExecutionContext      |                                   |
| ::hasConditionalsOnMeasureResults |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     mem                           | -   [cudaq::qreg::clear (C++      |
| ber)](api/languages/cpp_api.html# |                                   |
| _CPPv4N5cudaq16ExecutionContext31 |  function)](api/languages/cpp_api |
| hasConditionalsOnMeasureResultsE) | .html#_CPPv4N5cudaq4qreg5clearEv) |
| -   [cudaq::Executi               | -   [cudaq::qreg::front (C++      |
| onContext::invocationResultBuffer |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     member)](api/languages/cpp_   | 4N5cudaq4qreg5frontENSt6size_tE), |
| api.html#_CPPv4N5cudaq16Execution |     [\[1\]](api/languages/cpp_api |
| Context22invocationResultBufferE) | .html#_CPPv4N5cudaq4qreg5frontEv) |
| -   [cu                           | -   [cudaq::qreg::operator\[\]    |
| daq::ExecutionContext::kernelName |     (C++                          |
|     (C++                          |     functi                        |
|     member)](api/la               | on)](api/languages/cpp_api.html#_ |
| nguages/cpp_api.html#_CPPv4N5cuda | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| q16ExecutionContext10kernelNameE) | -   [cudaq::qreg::qreg (C++       |
| -   [cud                          |     function)                     |
| aq::ExecutionContext::kernelTrace | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4qregENSt6size_tE), |
|     member)](api/lan              |     [\[1\]](api/languages/cpp_ap  |
| guages/cpp_api.html#_CPPv4N5cudaq | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| 16ExecutionContext11kernelTraceE) | -   [cudaq::qreg::size (C++       |
| -   [cudaq:                       |                                   |
| :ExecutionContext::msm_dimensions |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     member)](api/langua           | -   [cudaq::qreg::slice (C++      |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     function)](api/langu          |
| xecutionContext14msm_dimensionsE) | ages/cpp_api.html#_CPPv4N5cudaq4q |
| -   [cudaq::                      | reg5sliceENSt6size_tENSt6size_tE) |
| ExecutionContext::msm_prob_err_id | -   [cudaq::qreg::value_type (C++ |
|     (C++                          |                                   |
|     member)](api/languag          | type)](api/languages/cpp_api.html |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | #_CPPv4N5cudaq4qreg10value_typeE) |
| ecutionContext15msm_prob_err_idE) | -   [cudaq::qspan (C++            |
| -   [cudaq::Ex                    |     class)](api/lang              |
| ecutionContext::msm_probabilities | uages/cpp_api.html#_CPPv4I_NSt6si |
|     (C++                          | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     member)](api/languages        | -   [cudaq::QuakeValue (C++       |
| /cpp_api.html#_CPPv4N5cudaq16Exec |     class)](api/languages/cpp_api |
| utionContext17msm_probabilitiesE) | .html#_CPPv4N5cudaq10QuakeValueE) |
| -                                 | -   [cudaq::Q                     |
|    [cudaq::ExecutionContext::name | uakeValue::canValidateNumElements |
|     (C++                          |     (C++                          |
|     member)]                      |     function)](api/languages      |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4N5cudaq10Quak |
| 4N5cudaq16ExecutionContext4nameE) | eValue22canValidateNumElementsEv) |
| -   [cu                           | -                                 |
| daq::ExecutionContext::noiseModel |  [cudaq::QuakeValue::constantSize |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api                |
| nguages/cpp_api.html#_CPPv4N5cuda | /languages/cpp_api.html#_CPPv4N5c |
| q16ExecutionContext10noiseModelE) | udaq10QuakeValue12constantSizeEv) |
| -   [cudaq::Exe                   | -   [cudaq::QuakeValue::dump (C++ |
| cutionContext::numberTrajectories |     function)](api/lan            |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     member)](api/languages/       | 10QuakeValue4dumpERNSt7ostreamE), |
| cpp_api.html#_CPPv4N5cudaq16Execu |     [\                            |
| tionContext18numberTrajectoriesE) | [1\]](api/languages/cpp_api.html# |
| -   [c                            | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| udaq::ExecutionContext::optResult | -   [cudaq                        |
|     (C++                          | ::QuakeValue::getRequiredElements |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/langua         |
| daq16ExecutionContext9optResultE) | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| -   [cudaq::Execu                 | uakeValue19getRequiredElementsEv) |
| tionContext::overlapComputeStates | -   [cudaq::QuakeValue::getValue  |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     function)]                    |
| p_api.html#_CPPv4N5cudaq16Executi | (api/languages/cpp_api.html#_CPPv |
| onContext20overlapComputeStatesE) | 4NK5cudaq10QuakeValue8getValueEv) |
| -   [cudaq                        | -   [cudaq::QuakeValue::inverse   |
| ::ExecutionContext::overlapResult |     (C++                          |
|     (C++                          |     function)                     |
|     member)](api/langu            | ](api/languages/cpp_api.html#_CPP |
| ages/cpp_api.html#_CPPv4N5cudaq16 | v4NK5cudaq10QuakeValue7inverseEv) |
| ExecutionContext13overlapResultE) | -   [cudaq::QuakeValue::isStdVec  |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::qpuId |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](                     | v4N5cudaq10QuakeValue8isStdVecEv) |
| api/languages/cpp_api.html#_CPPv4 | -                                 |
| N5cudaq16ExecutionContext5qpuIdE) |    [cudaq::QuakeValue::operator\* |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::registerNames |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/langu            | udaq10QuakeValuemlE10QuakeValue), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |                                   |
| ExecutionContext13registerNamesE) | [\[1\]](api/languages/cpp_api.htm |
| -   [cu                           | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| daq::ExecutionContext::reorderIdx | -   [cudaq::QuakeValue::operator+ |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)](api                |
| nguages/cpp_api.html#_CPPv4N5cuda | /languages/cpp_api.html#_CPPv4N5c |
| q16ExecutionContext10reorderIdxE) | udaq10QuakeValueplE10QuakeValue), |
| -                                 |     [                             |
|  [cudaq::ExecutionContext::result | \[1\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     member)](a                    |                                   |
| pi/languages/cpp_api.html#_CPPv4N | [\[2\]](api/languages/cpp_api.htm |
| 5cudaq16ExecutionContext6resultE) | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| -                                 | -   [cudaq::QuakeValue::operator- |
|   [cudaq::ExecutionContext::shots |     (C++                          |
|     (C++                          |     function)](api                |
|     member)](                     | /languages/cpp_api.html#_CPPv4N5c |
| api/languages/cpp_api.html#_CPPv4 | udaq10QuakeValuemiE10QuakeValue), |
| N5cudaq16ExecutionContext5shotsE) |     [                             |
| -   [cudaq::                      | \[1\]](api/languages/cpp_api.html |
| ExecutionContext::simulationState | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     (C++                          |     [                             |
|     member)](api/languag          | \[2\]](api/languages/cpp_api.html |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| ecutionContext15simulationStateE) |                                   |
| -                                 | [\[3\]](api/languages/cpp_api.htm |
|    [cudaq::ExecutionContext::spin | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     (C++                          | -   [cudaq::QuakeValue::operator/ |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api                |
| 4N5cudaq16ExecutionContext4spinE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::                      | udaq10QuakeValuedvE10QuakeValue), |
| ExecutionContext::totalIterations |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     member)](api/languag          | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | -                                 |
| ecutionContext15totalIterationsE) |  [cudaq::QuakeValue::operator\[\] |
| -   [cudaq::Executio              |     (C++                          |
| nContext::warnedNamedMeasurements |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/languages/cpp_a  | udaq10QuakeValueixEKNSt6size_tE), |
| pi.html#_CPPv4N5cudaq16ExecutionC |     [\[1\]](api/                  |
| ontext23warnedNamedMeasurementsE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::ExecutionResult (C++  | daq10QuakeValueixERK10QuakeValue) |
|     st                            | -                                 |
| ruct)](api/languages/cpp_api.html |    [cudaq::QuakeValue::QuakeValue |
| #_CPPv4N5cudaq15ExecutionResultE) |     (C++                          |
| -   [cud                          |     function)](api/languag        |
| aq::ExecutionResult::appendResult | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     (C++                          | akeValue10QuakeValueERN4mlir20Imp |
|     functio                       | licitLocOpBuilderEN4mlir5ValueE), |
| n)](api/languages/cpp_api.html#_C |     [\[1\]                        |
| PPv4N5cudaq15ExecutionResult12app | ](api/languages/cpp_api.html#_CPP |
| endResultENSt6stringENSt6size_tE) | v4N5cudaq10QuakeValue10QuakeValue |
| -   [cu                           | ERN4mlir20ImplicitLocOpBuilderEd) |
| daq::ExecutionResult::deserialize | -   [cudaq::QuakeValue::size (C++ |
|     (C++                          |     funct                         |
|     function)                     | ion)](api/languages/cpp_api.html# |
| ](api/languages/cpp_api.html#_CPP | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| v4N5cudaq15ExecutionResult11deser | -   [cudaq::QuakeValue::slice     |
| ializeERNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq:                       |     function)](api/languages/cpp_ |
| :ExecutionResult::ExecutionResult | api.html#_CPPv4N5cudaq10QuakeValu |
|     (C++                          | e5sliceEKNSt6size_tEKNSt6size_tE) |
|     functio                       | -   [cudaq::quantum_platform (C++ |
| n)](api/languages/cpp_api.html#_C |     cl                            |
| PPv4N5cudaq15ExecutionResult15Exe | ass)](api/languages/cpp_api.html# |
| cutionResultE16CountsDictionary), | _CPPv4N5cudaq16quantum_platformE) |
|     [\[1\]](api/lan               | -   [cudaq:                       |
| guages/cpp_api.html#_CPPv4N5cudaq | :quantum_platform::beginExecution |
| 15ExecutionResult15ExecutionResul |     (C++                          |
| tE16CountsDictionaryNSt6stringE), |     function)](api/languag        |
|     [\[2\                         | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ]](api/languages/cpp_api.html#_CP | antum_platform14beginExecutionEv) |
| Pv4N5cudaq15ExecutionResult15Exec | -   [cudaq::quantum_pl            |
| utionResultE16CountsDictionaryd), | atform::configureExecutionContext |
|                                   |     (C++                          |
|    [\[3\]](api/languages/cpp_api. |     function)](api/lang           |
| html#_CPPv4N5cudaq15ExecutionResu | uages/cpp_api.html#_CPPv4NK5cudaq |
| lt15ExecutionResultENSt6stringE), | 16quantum_platform25configureExec |
|     [\[4\                         | utionContextER16ExecutionContext) |
| ]](api/languages/cpp_api.html#_CP | -   [cuda                         |
| Pv4N5cudaq15ExecutionResult15Exec | q::quantum_platform::connectivity |
| utionResultERK15ExecutionResult), |     (C++                          |
|     [\[5\]](api/language          |     function)](api/langu          |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | ages/cpp_api.html#_CPPv4N5cudaq16 |
| cutionResult15ExecutionResultEd), | quantum_platform12connectivityEv) |
|     [\[6\]](api/languag           | -   [cuda                         |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | q::quantum_platform::endExecution |
| ecutionResult15ExecutionResultEv) |     (C++                          |
| -   [                             |     function)](api/langu          |
| cudaq::ExecutionResult::operator= | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     (C++                          | quantum_platform12endExecutionEv) |
|     function)](api/languages/     | -   [cudaq::q                     |
| cpp_api.html#_CPPv4N5cudaq15Execu | uantum_platform::enqueueAsyncTask |
| tionResultaSERK15ExecutionResult) |     (C++                          |
| -   [c                            |     function)](api/languages/     |
| udaq::ExecutionResult::operator== | cpp_api.html#_CPPv4N5cudaq16quant |
|     (C++                          | um_platform16enqueueAsyncTaskEKNS |
|     function)](api/languages/c    | t6size_tER19KernelExecutionTask), |
| pp_api.html#_CPPv4NK5cudaq15Execu |     [\[1\]](api/languag           |
| tionResulteqERK15ExecutionResult) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cud                          | antum_platform16enqueueAsyncTaskE |
| aq::ExecutionResult::registerName | KNSt6size_tERNSt8functionIFvvEEE) |
|     (C++                          | -   [cudaq::quantum_p             |
|     member)](api/lan              | latform::finalizeExecutionContext |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult12registerNameE) |     function)](api/languages/c    |
| -   [cudaq                        | pp_api.html#_CPPv4NK5cudaq16quant |
| ::ExecutionResult::sequentialData | um_platform24finalizeExecutionCon |
|     (C++                          | textERN5cudaq16ExecutionContextE) |
|     member)](api/langu            | -   [cudaq::qua                   |
| ages/cpp_api.html#_CPPv4N5cudaq15 | ntum_platform::get_codegen_config |
| ExecutionResult14sequentialDataE) |     (C++                          |
| -   [                             |     function)](api/languages/c    |
| cudaq::ExecutionResult::serialize | pp_api.html#_CPPv4N5cudaq16quantu |
|     (C++                          | m_platform18get_codegen_configEv) |
|     function)](api/l              | -   [cuda                         |
| anguages/cpp_api.html#_CPPv4NK5cu | q::quantum_platform::get_exec_ctx |
| daq15ExecutionResult9serializeEv) |     (C++                          |
| -   [cudaq::fermion_handler (C++  |     function)](api/langua         |
|     c                             | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| lass)](api/languages/cpp_api.html | quantum_platform12get_exec_ctxEv) |
| #_CPPv4N5cudaq15fermion_handlerE) | -   [c                            |
| -   [cudaq::fermion_op (C++       | udaq::quantum_platform::get_noise |
|     type)](api/languages/cpp_api  |     (C++                          |
| .html#_CPPv4N5cudaq10fermion_opE) |     function)](api/languages/c    |
| -   [cudaq::fermion_op_term (C++  | pp_api.html#_CPPv4N5cudaq16quantu |
|                                   | m_platform9get_noiseENSt6size_tE) |
| type)](api/languages/cpp_api.html | -   [cudaq:                       |
| #_CPPv4N5cudaq15fermion_op_termE) | :quantum_platform::get_num_qubits |
| -   [cudaq::FermioniqBaseQPU (C++ |     (C++                          |
|     cl                            |                                   |
| ass)](api/languages/cpp_api.html# | function)](api/languages/cpp_api. |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | html#_CPPv4NK5cudaq16quantum_plat |
| -   [cudaq::get_state (C++        | form14get_num_qubitsENSt6size_tE) |
|                                   | -   [cudaq::quantum_              |
|    function)](api/languages/cpp_a | platform::get_remote_capabilities |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     (C++                          |
| ateEDaRR13QuantumKernelDpRR4Args) |     function)                     |
| -   [cudaq::gradient (C++         | ](api/languages/cpp_api.html#_CPP |
|     class)](api/languages/cpp_    | v4NK5cudaq16quantum_platform23get |
| api.html#_CPPv4N5cudaq8gradientE) | _remote_capabilitiesENSt6size_tE) |
| -   [cudaq::gradient::clone (C++  | -   [cudaq::qua                   |
|     fun                           | ntum_platform::get_runtime_target |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     function)](api/languages/cp   |
| -   [cudaq::gradient::compute     | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform18get_runtime_targetEv) |
|     function)](api/language       | -   [cuda                         |
| s/cpp_api.html#_CPPv4N5cudaq8grad | q::quantum_platform::getLogStream |
| ient7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api/langu          |
|     [\[1\]](ap                    | ages/cpp_api.html#_CPPv4N5cudaq16 |
| i/languages/cpp_api.html#_CPPv4N5 | quantum_platform12getLogStreamEv) |
| cudaq8gradient7computeERKNSt6vect | -   [cud                          |
| orIdEERNSt6vectorIdEERK7spin_opd) | aq::quantum_platform::is_emulated |
| -   [cudaq::gradient::gradient    |     (C++                          |
|     (C++                          |                                   |
|     function)](api/lang           |    function)](api/languages/cpp_a |
| uages/cpp_api.html#_CPPv4I00EN5cu | pi.html#_CPPv4NK5cudaq16quantum_p |
| daq8gradient8gradientER7KernelT), | latform11is_emulatedENSt6size_tE) |
|                                   | -   [c                            |
|    [\[1\]](api/languages/cpp_api. | udaq::quantum_platform::is_remote |
| html#_CPPv4I00EN5cudaq8gradient8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api/languages/cp   |
|     [\[2\                         | p_api.html#_CPPv4NK5cudaq16quantu |
| ]](api/languages/cpp_api.html#_CP | m_platform9is_remoteENSt6size_tE) |
| Pv4I00EN5cudaq8gradient8gradientE | -   [cuda                         |
| RR13QuantumKernelRR10ArgsMapper), | q::quantum_platform::is_simulator |
|     [\[3                          |     (C++                          |
| \]](api/languages/cpp_api.html#_C |                                   |
| PPv4N5cudaq8gradient8gradientERRN |   function)](api/languages/cpp_ap |
| St8functionIFvNSt6vectorIdEEEEE), | i.html#_CPPv4NK5cudaq16quantum_pl |
|     [\[                           | atform12is_simulatorENSt6size_tE) |
| 4\]](api/languages/cpp_api.html#_ | -   [c                            |
| CPPv4N5cudaq8gradient8gradientEv) | udaq::quantum_platform::launchVQE |
| -   [cudaq::gradient::setArgs     |     (C++                          |
|     (C++                          |     function)](                   |
|     fu                            | api/languages/cpp_api.html#_CPPv4 |
| nction)](api/languages/cpp_api.ht | N5cudaq16quantum_platform9launchV |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | QEEKNSt6stringEPKvPN5cudaq8gradie |
| tArgsEvR13QuantumKernelDpRR4Args) | ntERKN5cudaq7spin_opERN5cudaq9opt |
| -   [cudaq::gradient::setKernel   | imizerEKiKNSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/languages/c    | :quantum_platform::list_platforms |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     (C++                          |
| ient9setKernelEvR13QuantumKernel) |     function)](api/languag        |
| -   [cud                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
| aq::gradients::central_difference | antum_platform14list_platformsEv) |
|     (C++                          | -                                 |
|     class)](api/la                |    [cudaq::quantum_platform::name |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9gradients18central_differenceE) |     function)](a                  |
| -   [cudaq::gra                   | pi/languages/cpp_api.html#_CPPv4N |
| dients::central_difference::clone | K5cudaq16quantum_platform4nameEv) |
|     (C++                          | -   [                             |
|     function)](api/languages      | cudaq::quantum_platform::num_qpus |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18central_difference5cloneEv) |     function)](api/l              |
| -   [cudaq::gradi                 | anguages/cpp_api.html#_CPPv4NK5cu |
| ents::central_difference::compute | daq16quantum_platform8num_qpusEv) |
|     (C++                          | -   [cudaq::                      |
|     function)](                   | quantum_platform::onRandomSeedSet |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq9gradients18central_differ |                                   |
| ence7computeERKNSt6vectorIdEERKNS | function)](api/languages/cpp_api. |
| t8functionIFdNSt6vectorIdEEEEEd), | html#_CPPv4N5cudaq16quantum_platf |
|                                   | orm15onRandomSeedSetENSt6size_tE) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq:                       |
| tml#_CPPv4N5cudaq9gradients18cent | :quantum_platform::reset_exec_ctx |
| ral_difference7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |     function)](api/languag        |
| -   [cudaq::gradie                | es/cpp_api.html#_CPPv4N5cudaq16qu |
| nts::central_difference::gradient | antum_platform14reset_exec_ctxEv) |
|     (C++                          | -   [cud                          |
|     functio                       | aq::quantum_platform::reset_noise |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4I00EN5cudaq9gradients18centra |     function)](api/languages/cpp_ |
| l_difference8gradientER7KernelT), | api.html#_CPPv4N5cudaq16quantum_p |
|     [\[1\]](api/langua            | latform11reset_noiseENSt6size_tE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq:                       |
| q9gradients18central_difference8g | :quantum_platform::resetLogStream |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\]](api/languages/cpp_    |     function)](api/languag        |
| api.html#_CPPv4I00EN5cudaq9gradie | es/cpp_api.html#_CPPv4N5cudaq16qu |
| nts18central_difference8gradientE | antum_platform14resetLogStreamEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cuda                         |
|     [\[3\]](api/languages/cpp     | q::quantum_platform::set_exec_ctx |
| _api.html#_CPPv4N5cudaq9gradients |     (C++                          |
| 18central_difference8gradientERRN |     funct                         |
| St8functionIFvNSt6vectorIdEEEEE), | ion)](api/languages/cpp_api.html# |
|     [\[4\]](api/languages/cp      | _CPPv4N5cudaq16quantum_platform12 |
| p_api.html#_CPPv4N5cudaq9gradient | set_exec_ctxEP16ExecutionContext) |
| s18central_difference8gradientEv) | -   [c                            |
| -   [cud                          | udaq::quantum_platform::set_noise |
| aq::gradients::forward_difference |     (C++                          |
|     (C++                          |     function                      |
|     class)](api/la                | )](api/languages/cpp_api.html#_CP |
| nguages/cpp_api.html#_CPPv4N5cuda | Pv4N5cudaq16quantum_platform9set_ |
| q9gradients18forward_differenceE) | noiseEPK11noise_modelNSt6size_tE) |
| -   [cudaq::gra                   | -   [cuda                         |
| dients::forward_difference::clone | q::quantum_platform::setLogStream |
|     (C++                          |     (C++                          |
|     function)](api/languages      |                                   |
| /cpp_api.html#_CPPv4N5cudaq9gradi |  function)](api/languages/cpp_api |
| ents18forward_difference5cloneEv) | .html#_CPPv4N5cudaq16quantum_plat |
| -   [cudaq::gradi                 | form12setLogStreamERNSt7ostreamE) |
| ents::forward_difference::compute | -   [cudaq::quantum_platfo        |
|     (C++                          | rm::supports_conditional_feedback |
|     function)](                   |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/               |
| N5cudaq9gradients18forward_differ | languages/cpp_api.html#_CPPv4NK5c |
| ence7computeERKNSt6vectorIdEERKNS | udaq16quantum_platform29supports_ |
| t8functionIFdNSt6vectorIdEEEEEd), | conditional_feedbackENSt6size_tE) |
|                                   | -   [cudaq::quantum_platfor       |
|   [\[1\]](api/languages/cpp_api.h | m::supports_explicit_measurements |
| tml#_CPPv4N5cudaq9gradients18forw |     (C++                          |
| ard_difference7computeERKNSt6vect |     function)](api/l              |
| orIdEERNSt6vectorIdEERK7spin_opd) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cudaq::gradie                | daq16quantum_platform30supports_e |
| nts::forward_difference::gradient | xplicit_measurementsENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_pla           |
|     functio                       | tform::supports_task_distribution |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4I00EN5cudaq9gradients18forwar |     fu                            |
| d_difference8gradientER7KernelT), | nction)](api/languages/cpp_api.ht |
|     [\[1\]](api/langua            | ml#_CPPv4NK5cudaq16quantum_platfo |
| ges/cpp_api.html#_CPPv4I00EN5cuda | rm26supports_task_distributionEv) |
| q9gradients18forward_difference8g | -   [cudaq::quantum               |
| radientER7KernelTRR10ArgsMapper), | _platform::with_execution_context |
|     [\[2\]](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4I00EN5cudaq9gradie |     function)                     |
| nts18forward_difference8gradientE | ](api/languages/cpp_api.html#_CPP |
| RR13QuantumKernelRR10ArgsMapper), | v4I0DpEN5cudaq16quantum_platform2 |
|     [\[3\]](api/languages/cpp     | 2with_execution_contextEDaR16Exec |
| _api.html#_CPPv4N5cudaq9gradients | utionContextRR8CallableDpRR4Args) |
| 18forward_difference8gradientERRN | -   [cudaq::QuantumTask (C++      |
| St8functionIFvNSt6vectorIdEEEEE), |     type)](api/languages/cpp_api. |
|     [\[4\]](api/languages/cp      | html#_CPPv4N5cudaq11QuantumTaskE) |
| p_api.html#_CPPv4N5cudaq9gradient | -   [cudaq::qubit (C++            |
| s18forward_difference8gradientEv) |     type)](api/languages/c        |
| -   [                             | pp_api.html#_CPPv4N5cudaq5qubitE) |
| cudaq::gradients::parameter_shift | -   [cudaq::QubitConnectivity     |
|     (C++                          |     (C++                          |
|     class)](api                   |     ty                            |
| /languages/cpp_api.html#_CPPv4N5c | pe)](api/languages/cpp_api.html#_ |
| udaq9gradients15parameter_shiftE) | CPPv4N5cudaq17QubitConnectivityE) |
| -   [cudaq::                      | -   [cudaq::QubitEdge (C++        |
| gradients::parameter_shift::clone |     type)](api/languages/cpp_a    |
|     (C++                          | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|     function)](api/langua         | -   [cudaq::qudit (C++            |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     clas                          |
| adients15parameter_shift5cloneEv) | s)](api/languages/cpp_api.html#_C |
| -   [cudaq::gr                    | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| adients::parameter_shift::compute | -   [cudaq::qudit::qudit (C++     |
|     (C++                          |                                   |
|     function                      | function)](api/languages/cpp_api. |
| )](api/languages/cpp_api.html#_CP | html#_CPPv4N5cudaq5qudit5quditEv) |
| Pv4N5cudaq9gradients15parameter_s | -   [cudaq::qvector (C++          |
| hift7computeERKNSt6vectorIdEERKNS |     class)                        |
| t8functionIFdNSt6vectorIdEEEEEd), | ](api/languages/cpp_api.html#_CPP |
|     [\[1\]](api/languages/cpp_ap  | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| i.html#_CPPv4N5cudaq9gradients15p | -   [cudaq::qvector::back (C++    |
| arameter_shift7computeERKNSt6vect |     function)](a                  |
| orIdEERNSt6vectorIdEERK7spin_opd) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::gra                   | 5cudaq7qvector4backENSt6size_tE), |
| dients::parameter_shift::gradient |                                   |
|     (C++                          |   [\[1\]](api/languages/cpp_api.h |
|     func                          | tml#_CPPv4N5cudaq7qvector4backEv) |
| tion)](api/languages/cpp_api.html | -   [cudaq::qvector::begin (C++   |
| #_CPPv4I00EN5cudaq9gradients15par |     fu                            |
| ameter_shift8gradientER7KernelT), | nction)](api/languages/cpp_api.ht |
|     [\[1\]](api/lan               | ml#_CPPv4N5cudaq7qvector5beginEv) |
| guages/cpp_api.html#_CPPv4I00EN5c | -   [cudaq::qvector::clear (C++   |
| udaq9gradients15parameter_shift8g |     fu                            |
| radientER7KernelTRR10ArgsMapper), | nction)](api/languages/cpp_api.ht |
|     [\[2\]](api/languages/c       | ml#_CPPv4N5cudaq7qvector5clearEv) |
| pp_api.html#_CPPv4I00EN5cudaq9gra | -   [cudaq::qvector::end (C++     |
| dients15parameter_shift8gradientE |                                   |
| RR13QuantumKernelRR10ArgsMapper), | function)](api/languages/cpp_api. |
|     [\[3\]](api/languages/        | html#_CPPv4N5cudaq7qvector3endEv) |
| cpp_api.html#_CPPv4N5cudaq9gradie | -   [cudaq::qvector::front (C++   |
| nts15parameter_shift8gradientERRN |     function)](ap                 |
| St8functionIFvNSt6vectorIdEEEEE), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[4\]](api/languages         | cudaq7qvector5frontENSt6size_tE), |
| /cpp_api.html#_CPPv4N5cudaq9gradi |                                   |
| ents15parameter_shift8gradientEv) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::kernel_builder (C++   | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     clas                          | -   [cudaq::qvector::operator=    |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4IDpEN5cudaq14kernel_builderE) |     functio                       |
| -   [c                            | n)](api/languages/cpp_api.html#_C |
| udaq::kernel_builder::constantVal | PPv4N5cudaq7qvectoraSERK7qvector) |
|     (C++                          | -   [cudaq::qvector::operator\[\] |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)                     |
| q14kernel_builder11constantValEd) | ](api/languages/cpp_api.html#_CPP |
| -   [cu                           | v4N5cudaq7qvectorixEKNSt6size_tE) |
| daq::kernel_builder::getArguments | -   [cudaq::qvector::qvector (C++ |
|     (C++                          |     function)](api/               |
|     function)](api/lan            | languages/cpp_api.html#_CPPv4N5cu |
| guages/cpp_api.html#_CPPv4N5cudaq | daq7qvector7qvectorENSt6size_tE), |
| 14kernel_builder12getArgumentsEv) |     [\[1\]](a                     |
| -   [cu                           | pi/languages/cpp_api.html#_CPPv4N |
| daq::kernel_builder::getNumParams | 5cudaq7qvector7qvectorERK5state), |
|     (C++                          |     [\[2\]](api                   |
|     function)](api/lan            | /languages/cpp_api.html#_CPPv4N5c |
| guages/cpp_api.html#_CPPv4N5cudaq | udaq7qvector7qvectorERK7qvector), |
| 14kernel_builder12getNumParamsEv) |     [\[3\]](api/languages/cpp     |
| -   [c                            | _api.html#_CPPv4N5cudaq7qvector7q |
| udaq::kernel_builder::isArgStdVec | vectorERKNSt6vectorI7complexEEb), |
|     (C++                          |     [\[4\]](ap                    |
|     function)](api/languages/cp   | i/languages/cpp_api.html#_CPPv4N5 |
| p_api.html#_CPPv4N5cudaq14kernel_ | cudaq7qvector7qvectorERR7qvector) |
| builder11isArgStdVecENSt6size_tE) | -   [cudaq::qvector::size (C++    |
| -   [cuda                         |     fu                            |
| q::kernel_builder::kernel_builder | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     function)](api/languages/cpp_ | -   [cudaq::qvector::slice (C++   |
| api.html#_CPPv4N5cudaq14kernel_bu |     function)](api/language       |
| ilder14kernel_builderERNSt6vector | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| IN7details17KernelBuilderTypeEEE) | tor5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::kernel_builder::name  | -   [cudaq::qvector::value_type   |
|     (C++                          |     (C++                          |
|     function)                     |     typ                           |
| ](api/languages/cpp_api.html#_CPP | e)](api/languages/cpp_api.html#_C |
| v4N5cudaq14kernel_builder4nameEv) | PPv4N5cudaq7qvector10value_typeE) |
| -                                 | -   [cudaq::qview (C++            |
|    [cudaq::kernel_builder::qalloc |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     function)](api/language       | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::qview::back (C++      |
| nel_builder6qallocE10QuakeValue), |     function)                     |
|     [\[1\]](api/language          | ](api/languages/cpp_api.html#_CPP |
| s/cpp_api.html#_CPPv4N5cudaq14ker | v4N5cudaq5qview4backENSt6size_tE) |
| nel_builder6qallocEKNSt6size_tE), | -   [cudaq::qview::begin (C++     |
|     [\[2                          |                                   |
| \]](api/languages/cpp_api.html#_C | function)](api/languages/cpp_api. |
| PPv4N5cudaq14kernel_builder6qallo | html#_CPPv4N5cudaq5qview5beginEv) |
| cERNSt6vectorINSt7complexIdEEEE), | -   [cudaq::qview::end (C++       |
|     [\[3\]](                      |                                   |
| api/languages/cpp_api.html#_CPPv4 |   function)](api/languages/cpp_ap |
| N5cudaq14kernel_builder6qallocEv) | i.html#_CPPv4N5cudaq5qview3endEv) |
| -   [cudaq::kernel_builder::swap  | -   [cudaq::qview::front (C++     |
|     (C++                          |     function)](                   |
|     function)](api/language       | api/languages/cpp_api.html#_CPPv4 |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | N5cudaq5qview5frontENSt6size_tE), |
| 4kernel_builder4swapEvRK10QuakeVa |                                   |
| lueRK10QuakeValueRK10QuakeValue), |    [\[1\]](api/languages/cpp_api. |
|                                   | html#_CPPv4N5cudaq5qview5frontEv) |
| [\[1\]](api/languages/cpp_api.htm | -   [cudaq::qview::operator\[\]   |
| l#_CPPv4I00EN5cudaq14kernel_build |     (C++                          |
| er4swapEvRKNSt6vectorI10QuakeValu |     functio                       |
| eEERK10QuakeValueRK10QuakeValue), | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| [\[2\]](api/languages/cpp_api.htm | -   [cudaq::qview::qview (C++     |
| l#_CPPv4N5cudaq14kernel_builder4s |     functio                       |
| wapERK10QuakeValueRK10QuakeValue) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::KernelExecutionTask   | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     (C++                          |     [\[1                          |
|     type                          | \]](api/languages/cpp_api.html#_C |
| )](api/languages/cpp_api.html#_CP | PPv4N5cudaq5qview5qviewERK5qview) |
| Pv4N5cudaq19KernelExecutionTaskE) | -   [cudaq::qview::size (C++      |
| -   [cudaq::KernelThunkResultType |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     struct)]                      | html#_CPPv4NK5cudaq5qview4sizeEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::qview::slice (C++     |
| 4N5cudaq21KernelThunkResultTypeE) |     function)](api/langua         |
| -   [cudaq::KernelThunkType (C++  | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|                                   | iew5sliceENSt6size_tENSt6size_tE) |
| type)](api/languages/cpp_api.html | -   [cudaq::qview::value_type     |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     (C++                          |
| -   [cudaq::kraus_channel (C++    |     t                             |
|                                   | ype)](api/languages/cpp_api.html# |
|  class)](api/languages/cpp_api.ht | _CPPv4N5cudaq5qview10value_typeE) |
| ml#_CPPv4N5cudaq13kraus_channelE) | -   [cudaq::range (C++            |
| -   [cudaq::kraus_channel::empty  |     fun                           |
|     (C++                          | ction)](api/languages/cpp_api.htm |
|     function)]                    | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| (api/languages/cpp_api.html#_CPPv | orI11ElementTypeEE11ElementType), |
| 4NK5cudaq13kraus_channel5emptyEv) |     [\[1\]](api/languages/cpp_    |
| -   [cudaq::kraus_c               | api.html#_CPPv4I0EN5cudaq5rangeEN |
| hannel::generateUnitaryParameters | St6vectorI11ElementTypeEE11Elemen |
|     (C++                          | tType11ElementType11ElementType), |
|                                   |     [                             |
|    function)](api/languages/cpp_a | \[2\]](api/languages/cpp_api.html |
| pi.html#_CPPv4N5cudaq13kraus_chan | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| nel25generateUnitaryParametersEv) | -   [cudaq::real (C++             |
| -                                 |     type)](api/languages/         |
|    [cudaq::kraus_channel::get_ops | cpp_api.html#_CPPv4N5cudaq4realE) |
|     (C++                          | -   [cudaq::registry (C++         |
|     function)](a                  |     type)](api/languages/cpp_     |
| pi/languages/cpp_api.html#_CPPv4N | api.html#_CPPv4N5cudaq8registryE) |
| K5cudaq13kraus_channel7get_opsEv) | -                                 |
| -   [cudaq::                      |  [cudaq::registry::RegisteredType |
| kraus_channel::is_unitary_mixture |     (C++                          |
|     (C++                          |     class)](api/                  |
|     function)](api/languages      | languages/cpp_api.html#_CPPv4I0EN |
| /cpp_api.html#_CPPv4NK5cudaq13kra | 5cudaq8registry14RegisteredTypeE) |
| us_channel18is_unitary_mixtureEv) | -   [cudaq::RemoteCapabilities    |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::kraus_channel |     struc                         |
|     (C++                          | t)](api/languages/cpp_api.html#_C |
|     function)](api/lang           | PPv4N5cudaq18RemoteCapabilitiesE) |
| uages/cpp_api.html#_CPPv4IDpEN5cu | -   [cudaq::Remo                  |
| daq13kraus_channel13kraus_channel | teCapabilities::isRemoteSimulator |
| EDpRRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     member)](api/languages/c      |
|  [\[1\]](api/languages/cpp_api.ht | pp_api.html#_CPPv4N5cudaq18Remote |
| ml#_CPPv4N5cudaq13kraus_channel13 | Capabilities17isRemoteSimulatorE) |
| kraus_channelERK13kraus_channel), | -   [cudaq::Remot                 |
|     [\[2\]                        | eCapabilities::RemoteCapabilities |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq13kraus_channel13kraus_c |     function)](api/languages/cpp  |
| hannelERKNSt6vectorI8kraus_opEE), | _api.html#_CPPv4N5cudaq18RemoteCa |
|     [\[3\]                        | pabilities18RemoteCapabilitiesEb) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq:                       |
| v4N5cudaq13kraus_channel13kraus_c | :RemoteCapabilities::stateOverlap |
| hannelERRNSt6vectorI8kraus_opEE), |     (C++                          |
|     [\[4\]](api/lan               |     member)](api/langua           |
| guages/cpp_api.html#_CPPv4N5cudaq | ges/cpp_api.html#_CPPv4N5cudaq18R |
| 13kraus_channel13kraus_channelEv) | emoteCapabilities12stateOverlapE) |
| -                                 | -                                 |
| [cudaq::kraus_channel::noise_type |   [cudaq::RemoteCapabilities::vqe |
|     (C++                          |     (C++                          |
|     member)](api                  |     member)](                     |
| /languages/cpp_api.html#_CPPv4N5c | api/languages/cpp_api.html#_CPPv4 |
| udaq13kraus_channel10noise_typeE) | N5cudaq18RemoteCapabilities3vqeE) |
| -                                 | -   [cudaq::RemoteSimulationState |
|   [cudaq::kraus_channel::op_names |     (C++                          |
|     (C++                          |     class)]                       |
|     member)](                     | (api/languages/cpp_api.html#_CPPv |
| api/languages/cpp_api.html#_CPPv4 | 4N5cudaq21RemoteSimulationStateE) |
| N5cudaq13kraus_channel8op_namesE) | -   [cudaq::Resources (C++        |
| -                                 |     class)](api/languages/cpp_a   |
|  [cudaq::kraus_channel::operator= | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     (C++                          | -   [cudaq::run (C++              |
|     function)](api/langua         |     function)]                    |
| ges/cpp_api.html#_CPPv4N5cudaq13k | (api/languages/cpp_api.html#_CPPv |
| raus_channelaSERK13kraus_channel) | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| -   [c                            | 5invoke_result_tINSt7decay_tI13Qu |
| udaq::kraus_channel::operator\[\] | antumKernelEEDpNSt7decay_tI4ARGSE |
|     (C++                          | EEEEENSt6size_tERN5cudaq11noise_m |
|     function)](api/l              | odelERR13QuantumKernelDpRR4ARGS), |
| anguages/cpp_api.html#_CPPv4N5cud |     [\[1\]](api/langu             |
| aq13kraus_channelixEKNSt6size_tE) | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| -                                 | daq3runENSt6vectorINSt15invoke_re |
| [cudaq::kraus_channel::parameters | sult_tINSt7decay_tI13QuantumKerne |
|     (C++                          | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
|     member)](api                  | ize_tERR13QuantumKernelDpRR4ARGS) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::run_async (C++        |
| udaq13kraus_channel10parametersE) |     functio                       |
| -   [cudaq::krau                  | n)](api/languages/cpp_api.html#_C |
| s_channel::populateDefaultOpNames | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     (C++                          | tureINSt6vectorINSt15invoke_resul |
|     function)](api/languages/cp   | t_tINSt7decay_tI13QuantumKernelEE |
| p_api.html#_CPPv4N5cudaq13kraus_c | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| hannel22populateDefaultOpNamesEv) | ze_tENSt6size_tERN5cudaq11noise_m |
| -   [cu                           | odelERR13QuantumKernelDpRR4ARGS), |
| daq::kraus_channel::probabilities |     [\[1\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4I0DpEN |
|     member)](api/la               | 5cudaq9run_asyncENSt6futureINSt6v |
| nguages/cpp_api.html#_CPPv4N5cuda | ectorINSt15invoke_result_tINSt7de |
| q13kraus_channel13probabilitiesE) | cay_tI13QuantumKernelEEDpNSt7deca |
| -                                 | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|  [cudaq::kraus_channel::push_back | ize_tERR13QuantumKernelDpRR4ARGS) |
|     (C++                          | -   [cudaq::RuntimeTarget (C++    |
|     function)](api                |                                   |
| /languages/cpp_api.html#_CPPv4N5c | struct)](api/languages/cpp_api.ht |
| udaq13kraus_channel9push_backE8kr | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| aus_opNSt8optionalINSt6stringEEE) | -   [cudaq::sample (C++           |
| -   [cudaq::kraus_channel::size   |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     function)                     | mpleE13sample_resultRK14sample_op |
| ](api/languages/cpp_api.html#_CPP | tionsRR13QuantumKernelDpRR4Args), |
| v4NK5cudaq13kraus_channel4sizeEv) |     [\[1\                         |
| -   [                             | ]](api/languages/cpp_api.html#_CP |
| cudaq::kraus_channel::unitary_ops | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     (C++                          | esultRR13QuantumKernelDpRR4Args), |
|     member)](api/                 |     [\                            |
| languages/cpp_api.html#_CPPv4N5cu | [2\]](api/languages/cpp_api.html# |
| daq13kraus_channel11unitary_opsE) | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| -   [cudaq::kraus_op (C++         | ize_tERR13QuantumKernelDpRR4Args) |
|     struct)](api/languages/cpp_   | -   [cudaq::sample_options (C++   |
| api.html#_CPPv4N5cudaq8kraus_opE) |     s                             |
| -   [cudaq::kraus_op::adjoint     | truct)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq14sample_optionsE) |
|     functi                        | -   [cudaq::sample_result (C++    |
| on)](api/languages/cpp_api.html#_ |                                   |
| CPPv4NK5cudaq8kraus_op7adjointEv) |  class)](api/languages/cpp_api.ht |
| -   [cudaq::kraus_op::data (C++   | ml#_CPPv4N5cudaq13sample_resultE) |
|                                   | -   [cudaq::sample_result::append |
|  member)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     function)](api/languages/cpp_ |
| -   [cudaq::kraus_op::kraus_op    | api.html#_CPPv4N5cudaq13sample_re |
|     (C++                          | sult6appendERK15ExecutionResultb) |
|     func                          | -   [cudaq::sample_result::begin  |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |     function)]                    |
| opERRNSt16initializer_listI1TEE), | (api/languages/cpp_api.html#_CPPv |
|                                   | 4N5cudaq13sample_result5beginEv), |
|  [\[1\]](api/languages/cpp_api.ht |     [\[1\]]                       |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | (api/languages/cpp_api.html#_CPPv |
| pENSt6vectorIN5cudaq7complexEEE), | 4NK5cudaq13sample_result5beginEv) |
|     [\[2\]](api/l                 | -   [cudaq::sample_result::cbegin |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq8kraus_op8kraus_opERK8kraus_op) |     function)](                   |
| -   [cudaq::kraus_op::nCols (C++  | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq13sample_result6cbeginEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::sample_result::cend   |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     (C++                          |
| -   [cudaq::kraus_op::nRows (C++  |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
| member)](api/languages/cpp_api.ht | v4NK5cudaq13sample_result4cendEv) |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | -   [cudaq::sample_result::clear  |
| -   [cudaq::kraus_op::operator=   |     (C++                          |
|     (C++                          |     function)                     |
|     function)                     | ](api/languages/cpp_api.html#_CPP |
| ](api/languages/cpp_api.html#_CPP | v4N5cudaq13sample_result5clearEv) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cudaq::sample_result::count  |
| -   [cudaq::kraus_op::precision   |     (C++                          |
|     (C++                          |     function)](                   |
|     memb                          | api/languages/cpp_api.html#_CPPv4 |
| er)](api/languages/cpp_api.html#_ | NK5cudaq13sample_result5countENSt |
| CPPv4N5cudaq8kraus_op9precisionE) | 11string_viewEKNSt11string_viewE) |
| -   [cudaq::matrix_callback (C++  | -   [                             |
|     c                             | cudaq::sample_result::deserialize |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15matrix_callbackE) |     functio                       |
| -   [cudaq::matrix_handler (C++   | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq13sample_result11deser |
| class)](api/languages/cpp_api.htm | ializeERNSt6vectorINSt6size_tEEE) |
| l#_CPPv4N5cudaq14matrix_handlerE) | -   [cudaq::sample_result::dump   |
| -   [cudaq::mat                   |     (C++                          |
| rix_handler::commutation_behavior |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     struct)](api/languages/       | ample_result4dumpERNSt7ostreamE), |
| cpp_api.html#_CPPv4N5cudaq14matri |     [\[1\]                        |
| x_handler20commutation_behaviorE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4NK5cudaq13sample_result4dumpEv) |
|    [cudaq::matrix_handler::define | -   [cudaq::sample_result::end    |
|     (C++                          |     (C++                          |
|     function)](a                  |     function                      |
| pi/languages/cpp_api.html#_CPPv4N | )](api/languages/cpp_api.html#_CP |
| 5cudaq14matrix_handler6defineENSt | Pv4N5cudaq13sample_result3endEv), |
| 6stringENSt6vectorINSt7int64_tEEE |     [\[1\                         |
| RR15matrix_callbackRKNSt13unorder | ]](api/languages/cpp_api.html#_CP |
| ed_mapINSt6stringENSt6stringEEE), | Pv4NK5cudaq13sample_result3endEv) |
|                                   | -   [                             |
| [\[1\]](api/languages/cpp_api.htm | cudaq::sample_result::expectation |
| l#_CPPv4N5cudaq14matrix_handler6d |     (C++                          |
| efineENSt6stringENSt6vectorINSt7i |     f                             |
| nt64_tEEERR15matrix_callbackRR20d | unction)](api/languages/cpp_api.h |
| iag_matrix_callbackRKNSt13unorder | tml#_CPPv4NK5cudaq13sample_result |
| ed_mapINSt6stringENSt6stringEEE), | 11expectationEKNSt11string_viewE) |
|     [\[2\]](                      | -   [c                            |
| api/languages/cpp_api.html#_CPPv4 | udaq::sample_result::get_marginal |
| N5cudaq14matrix_handler6defineENS |     (C++                          |
| t6stringENSt6vectorINSt7int64_tEE |     function)](api/languages/cpp_ |
| ERR15matrix_callbackRRNSt13unorde | api.html#_CPPv4NK5cudaq13sample_r |
| red_mapINSt6stringENSt6stringEEE) | esult12get_marginalERKNSt6vectorI |
| -                                 | NSt6size_tEEEKNSt11string_viewE), |
|   [cudaq::matrix_handler::degrees |     [\[1\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4NK5cudaq13sample_r |
|     function)](ap                 | esult12get_marginalERRKNSt6vector |
| i/languages/cpp_api.html#_CPPv4NK | INSt6size_tEEEKNSt11string_viewE) |
| 5cudaq14matrix_handler7degreesEv) | -   [cuda                         |
| -                                 | q::sample_result::get_total_shots |
|  [cudaq::matrix_handler::displace |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/language       | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| s/cpp_api.html#_CPPv4N5cudaq14mat | sample_result15get_total_shotsEv) |
| rix_handler8displaceENSt6size_tE) | -   [cuda                         |
| -   [cudaq::matrix                | q::sample_result::has_even_parity |
| _handler::get_expected_dimensions |     (C++                          |
|     (C++                          |     fun                           |
|                                   | ction)](api/languages/cpp_api.htm |
|    function)](api/languages/cpp_a | l#_CPPv4N5cudaq13sample_result15h |
| pi.html#_CPPv4NK5cudaq14matrix_ha | as_even_parityENSt11string_viewE) |
| ndler23get_expected_dimensionsEv) | -   [cuda                         |
| -   [cudaq::matrix_ha             | q::sample_result::has_expectation |
| ndler::get_parameter_descriptions |     (C++                          |
|     (C++                          |     funct                         |
|                                   | ion)](api/languages/cpp_api.html# |
| function)](api/languages/cpp_api. | _CPPv4NK5cudaq13sample_result15ha |
| html#_CPPv4NK5cudaq14matrix_handl | s_expectationEKNSt11string_viewE) |
| er26get_parameter_descriptionsEv) | -   [cu                           |
| -   [c                            | daq::sample_result::most_probable |
| udaq::matrix_handler::instantiate |     (C++                          |
|     (C++                          |     fun                           |
|     function)](a                  | ction)](api/languages/cpp_api.htm |
| pi/languages/cpp_api.html#_CPPv4N | l#_CPPv4NK5cudaq13sample_result13 |
| 5cudaq14matrix_handler11instantia | most_probableEKNSt11string_viewE) |
| teENSt6stringERKNSt6vectorINSt6si | -                                 |
| ze_tEEERK20commutation_behavior), | [cudaq::sample_result::operator+= |
|     [\[1\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/langua         |
| N5cudaq14matrix_handler11instanti | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ateENSt6stringERRNSt6vectorINSt6s | ample_resultpLERK13sample_result) |
| ize_tEEERK20commutation_behavior) | -                                 |
| -   [cuda                         |  [cudaq::sample_result::operator= |
| q::matrix_handler::matrix_handler |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/languag        | ges/cpp_api.html#_CPPv4N5cudaq13s |
| es/cpp_api.html#_CPPv4I0_NSt11ena | ample_resultaSERR13sample_result) |
| ble_if_tINSt12is_base_of_vI16oper | -                                 |
| ator_handler1TEEbEEEN5cudaq14matr | [cudaq::sample_result::operator== |
| ix_handler14matrix_handlerERK1T), |     (C++                          |
|     [\[1\]](ap                    |     function)](api/languag        |
| i/languages/cpp_api.html#_CPPv4I0 | es/cpp_api.html#_CPPv4NK5cudaq13s |
| _NSt11enable_if_tINSt12is_base_of | ample_resulteqERK13sample_result) |
| _vI16operator_handler1TEEbEEEN5cu | -   [                             |
| daq14matrix_handler14matrix_handl | cudaq::sample_result::probability |
| erERK1TRK20commutation_behavior), |     (C++                          |
|     [\[2\]](api/languages/cpp_ap  |     function)](api/lan            |
| i.html#_CPPv4N5cudaq14matrix_hand | guages/cpp_api.html#_CPPv4NK5cuda |
| ler14matrix_handlerENSt6size_tE), | q13sample_result11probabilityENSt |
|     [\[3\]](api/                  | 11string_viewEKNSt11string_viewE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cud                          |
| daq14matrix_handler14matrix_handl | aq::sample_result::register_names |
| erENSt6stringERKNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     function)](api/langu          |
|     [\[4\]](api/                  | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| languages/cpp_api.html#_CPPv4N5cu | 3sample_result14register_namesEv) |
| daq14matrix_handler14matrix_handl | -                                 |
| erENSt6stringERRNSt6vectorINSt6si |    [cudaq::sample_result::reorder |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\                            |     function)](api/langua         |
| [5\]](api/languages/cpp_api.html# | ges/cpp_api.html#_CPPv4N5cudaq13s |
| _CPPv4N5cudaq14matrix_handler14ma | ample_result7reorderERKNSt6vector |
| trix_handlerERK14matrix_handler), | INSt6size_tEEEKNSt11string_viewE) |
|     [                             | -   [cu                           |
| \[6\]](api/languages/cpp_api.html | daq::sample_result::sample_result |
| #_CPPv4N5cudaq14matrix_handler14m |     (C++                          |
| atrix_handlerERR14matrix_handler) |     func                          |
| -                                 | tion)](api/languages/cpp_api.html |
|  [cudaq::matrix_handler::momentum | #_CPPv4N5cudaq13sample_result13sa |
|     (C++                          | mple_resultERK15ExecutionResult), |
|     function)](api/language       |     [\[1\]](api/la                |
| s/cpp_api.html#_CPPv4N5cudaq14mat | nguages/cpp_api.html#_CPPv4N5cuda |
| rix_handler8momentumENSt6size_tE) | q13sample_result13sample_resultER |
| -                                 | KNSt6vectorI15ExecutionResultEE), |
|    [cudaq::matrix_handler::number |                                   |
|     (C++                          |  [\[2\]](api/languages/cpp_api.ht |
|     function)](api/langua         | ml#_CPPv4N5cudaq13sample_result13 |
| ges/cpp_api.html#_CPPv4N5cudaq14m | sample_resultERR13sample_result), |
| atrix_handler6numberENSt6size_tE) |     [                             |
| -                                 | \[3\]](api/languages/cpp_api.html |
| [cudaq::matrix_handler::operator= | #_CPPv4N5cudaq13sample_result13sa |
|     (C++                          | mple_resultERR15ExecutionResult), |
|     fun                           |     [\[4\]](api/lan               |
| ction)](api/languages/cpp_api.htm | guages/cpp_api.html#_CPPv4N5cudaq |
| l#_CPPv4I0_NSt11enable_if_tIXaant | 13sample_result13sample_resultEdR |
| NSt7is_sameI1T14matrix_handlerE5v | KNSt6vectorI15ExecutionResultEE), |
| alueENSt12is_base_of_vI16operator |     [\[5\]](api/lan               |
| _handler1TEEEbEEEN5cudaq14matrix_ | guages/cpp_api.html#_CPPv4N5cudaq |
| handleraSER14matrix_handlerRK1T), | 13sample_result13sample_resultEv) |
|     [\[1\]](api/languages         | -                                 |
| /cpp_api.html#_CPPv4N5cudaq14matr |  [cudaq::sample_result::serialize |
| ix_handleraSERK14matrix_handler), |     (C++                          |
|     [\[2\]](api/language          |     function)](api                |
| s/cpp_api.html#_CPPv4N5cudaq14mat | /languages/cpp_api.html#_CPPv4NK5 |
| rix_handleraSERR14matrix_handler) | cudaq13sample_result9serializeEv) |
| -   [                             | -   [cudaq::sample_result::size   |
| cudaq::matrix_handler::operator== |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/languages      | pp_api.html#_CPPv4NK5cudaq13sampl |
| /cpp_api.html#_CPPv4NK5cudaq14mat | e_result4sizeEKNSt11string_viewE) |
| rix_handlereqERK14matrix_handler) | -   [cudaq::sample_result::to_map |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::parity |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq13sample_ |
|     function)](api/langua         | result6to_mapEKNSt11string_viewE) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -   [cuda                         |
| atrix_handler6parityENSt6size_tE) | q::sample_result::\~sample_result |
| -                                 |     (C++                          |
|  [cudaq::matrix_handler::position |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/language       | _CPPv4N5cudaq13sample_resultD0Ev) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::scalar_callback (C++  |
| rix_handler8positionENSt6size_tE) |     c                             |
| -   [cudaq::                      | lass)](api/languages/cpp_api.html |
| matrix_handler::remove_definition | #_CPPv4N5cudaq15scalar_callbackE) |
|     (C++                          | -   [c                            |
|     fu                            | udaq::scalar_callback::operator() |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14matrix_handler1 |     function)](api/language       |
| 7remove_definitionERKNSt6stringE) | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| -                                 | alar_callbackclERKNSt13unordered_ |
|   [cudaq::matrix_handler::squeeze | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [                             |
|     function)](api/languag        | cudaq::scalar_callback::operator= |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     (C++                          |
| trix_handler7squeezeENSt6size_tE) |     function)](api/languages/c    |
| -   [cudaq::m                     | pp_api.html#_CPPv4N5cudaq15scalar |
| atrix_handler::to_diagonal_matrix | _callbackaSERK15scalar_callback), |
|     (C++                          |     [\[1\]](api/languages/        |
|     function)](api/lang           | cpp_api.html#_CPPv4N5cudaq15scala |
| uages/cpp_api.html#_CPPv4NK5cudaq | r_callbackaSERR15scalar_callback) |
| 14matrix_handler18to_diagonal_mat | -   [cudaq:                       |
| rixERNSt13unordered_mapINSt6size_ | :scalar_callback::scalar_callback |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     function)](api/languag        |
| -                                 | es/cpp_api.html#_CPPv4I0_NSt11ena |
| [cudaq::matrix_handler::to_matrix | ble_if_tINSt16is_invocable_r_vINS |
|     (C++                          | t7complexIdEE8CallableRKNSt13unor |
|     function)                     | dered_mapINSt6stringENSt7complexI |
| ](api/languages/cpp_api.html#_CPP | dEEEEEEbEEEN5cudaq15scalar_callba |
| v4NK5cudaq14matrix_handler9to_mat | ck15scalar_callbackERR8Callable), |
| rixERNSt13unordered_mapINSt6size_ |     [\[1\                         |
| tENSt7int64_tEEERKNSt13unordered_ | ]](api/languages/cpp_api.html#_CP |
| mapINSt6stringENSt7complexIdEEEE) | Pv4N5cudaq15scalar_callback15scal |
| -                                 | ar_callbackERK15scalar_callback), |
| [cudaq::matrix_handler::to_string |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     function)](api/               | PPv4N5cudaq15scalar_callback15sca |
| languages/cpp_api.html#_CPPv4NK5c | lar_callbackERR15scalar_callback) |
| udaq14matrix_handler9to_stringEb) | -   [cudaq::scalar_operator (C++  |
| -                                 |     c                             |
| [cudaq::matrix_handler::unique_id | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_operatorE) |
|     function)](api/               | -                                 |
| languages/cpp_api.html#_CPPv4NK5c | [cudaq::scalar_operator::evaluate |
| udaq14matrix_handler9unique_idEv) |     (C++                          |
| -   [cudaq:                       |                                   |
| :matrix_handler::\~matrix_handler |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq15scalar_op |
|     functi                        | erator8evaluateERKNSt13unordered_ |
| on)](api/languages/cpp_api.html#_ | mapINSt6stringENSt7complexIdEEEE) |
| CPPv4N5cudaq14matrix_handlerD0Ev) | -   [cudaq::scalar_ope            |
| -   [cudaq::matrix_op (C++        | rator::get_parameter_descriptions |
|     type)](api/languages/cpp_a    |     (C++                          |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     f                             |
| -   [cudaq::matrix_op_term (C++   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4NK5cudaq15scalar_operat |
|  type)](api/languages/cpp_api.htm | or26get_parameter_descriptionsEv) |
| l#_CPPv4N5cudaq14matrix_op_termE) | -   [cu                           |
| -                                 | daq::scalar_operator::is_constant |
|    [cudaq::mdiag_operator_handler |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     class)](                      | uages/cpp_api.html#_CPPv4NK5cudaq |
| api/languages/cpp_api.html#_CPPv4 | 15scalar_operator11is_constantEv) |
| N5cudaq22mdiag_operator_handlerE) | -   [c                            |
| -   [cudaq::mpi (C++              | udaq::scalar_operator::operator\* |
|     type)](api/languages          |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) |     function                      |
| -   [cudaq::mpi::all_gather (C++  | )](api/languages/cpp_api.html#_CP |
|     fu                            | Pv4N5cudaq15scalar_operatormlENSt |
| nction)](api/languages/cpp_api.ht | 7complexIdEERK15scalar_operator), |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     [\[1\                         |
| RNSt6vectorIdEERKNSt6vectorIdEE), | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_operatormlENSt |
|   [\[1\]](api/languages/cpp_api.h | 7complexIdEERR15scalar_operator), |
| tml#_CPPv4N5cudaq3mpi10all_gather |     [\[2\]](api/languages/cp      |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::mpi::all_reduce (C++  | operatormlEdRK15scalar_operator), |
|                                   |     [\[3\]](api/languages/cp      |
|  function)](api/languages/cpp_api | p_api.html#_CPPv4N5cudaq15scalar_ |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | operatormlEdRR15scalar_operator), |
| reduceE1TRK1TRK14BinaryFunction), |     [\[4\]](api/languages         |
|     [\[1\]](api/langu             | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ages/cpp_api.html#_CPPv4I00EN5cud | alar_operatormlENSt7complexIdEE), |
| aq3mpi10all_reduceE1TRK1TRK4Func) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::mpi::broadcast (C++   | _api.html#_CPPv4NKR5cudaq15scalar |
|     function)](api/               | _operatormlERK15scalar_operator), |
| languages/cpp_api.html#_CPPv4N5cu |     [\[6\]]                       |
| daq3mpi9broadcastERNSt6stringEi), | (api/languages/cpp_api.html#_CPPv |
|     [\[1\]](api/la                | 4NKR5cudaq15scalar_operatormlEd), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[7\]](api/language          |
| q3mpi9broadcastERNSt6vectorIdEEi) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::mpi::finalize (C++    | alar_operatormlENSt7complexIdEE), |
|     f                             |     [\[8\]](api/languages/cp      |
| unction)](api/languages/cpp_api.h | p_api.html#_CPPv4NO5cudaq15scalar |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | _operatormlERK15scalar_operator), |
| -   [cudaq::mpi::initialize (C++  |     [\[9\                         |
|     function                      | ]](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4NO5cudaq15scalar_operatormlEd) |
| Pv4N5cudaq3mpi10initializeEiPPc), | -   [cu                           |
|     [                             | daq::scalar_operator::operator\*= |
| \[1\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq3mpi10initializeEv) |     function)](api/languag        |
| -   [cudaq::mpi::is_initialized   | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormLENSt7complexIdEE), |
|     function                      |     [\[1\]](api/languages/c       |
| )](api/languages/cpp_api.html#_CP | pp_api.html#_CPPv4N5cudaq15scalar |
| Pv4N5cudaq3mpi14is_initializedEv) | _operatormLERK15scalar_operator), |
| -   [cudaq::mpi::num_ranks (C++   |     [\[2                          |
|     fu                            | \]](api/languages/cpp_api.html#_C |
| nction)](api/languages/cpp_api.ht | PPv4N5cudaq15scalar_operatormLEd) |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | -   [                             |
| -   [cudaq::mpi::rank (C++        | cudaq::scalar_operator::operator+ |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function                      |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise_model (C++      | Pv4N5cudaq15scalar_operatorplENSt |
|                                   | 7complexIdEERK15scalar_operator), |
|    class)](api/languages/cpp_api. |     [\[1\                         |
| html#_CPPv4N5cudaq11noise_modelE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::n                     | Pv4N5cudaq15scalar_operatorplENSt |
| oise_model::add_all_qubit_channel | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     function)](api                | p_api.html#_CPPv4N5cudaq15scalar_ |
| /languages/cpp_api.html#_CPPv4IDp | operatorplEdRK15scalar_operator), |
| EN5cudaq11noise_model21add_all_qu |     [\[3\]](api/languages/cp      |
| bit_channelEvRK13kraus_channeli), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](api/langua            | operatorplEdRR15scalar_operator), |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     [\[4\]](api/languages         |
| oise_model21add_all_qubit_channel | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ERKNSt6stringERK13kraus_channeli) | alar_operatorplENSt7complexIdEE), |
| -                                 |     [\[5\]](api/languages/cpp     |
|  [cudaq::noise_model::add_channel | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatorplERK15scalar_operator), |
|     funct                         |     [\[6\]]                       |
| ion)](api/languages/cpp_api.html# | (api/languages/cpp_api.html#_CPPv |
| _CPPv4IDpEN5cudaq11noise_model11a | 4NKR5cudaq15scalar_operatorplEd), |
| dd_channelEvRK15PredicateFuncTy), |     [\[7\]]                       |
|     [\[1\]](api/languages/cpp_    | (api/languages/cpp_api.html#_CPPv |
| api.html#_CPPv4IDpEN5cudaq11noise | 4NKR5cudaq15scalar_operatorplEv), |
| _model11add_channelEvRKNSt6vector |     [\[8\]](api/language          |
| INSt6size_tEEERK13kraus_channel), | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     [\[2\]](ap                    | alar_operatorplENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[9\]](api/languages/cp      |
| cudaq11noise_model11add_channelER | p_api.html#_CPPv4NO5cudaq15scalar |
| KNSt6stringERK15PredicateFuncTy), | _operatorplERK15scalar_operator), |
|                                   |     [\[10\]                       |
| [\[3\]](api/languages/cpp_api.htm | ](api/languages/cpp_api.html#_CPP |
| l#_CPPv4N5cudaq11noise_model11add | v4NO5cudaq15scalar_operatorplEd), |
| _channelERKNSt6stringERKNSt6vecto |     [\[11\                        |
| rINSt6size_tEEERK13kraus_channel) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise_model::empty    | Pv4NO5cudaq15scalar_operatorplEv) |
|     (C++                          | -   [c                            |
|     function                      | udaq::scalar_operator::operator+= |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4NK5cudaq11noise_model5emptyEv) |     function)](api/languag        |
| -                                 | es/cpp_api.html#_CPPv4N5cudaq15sc |
| [cudaq::noise_model::get_channels | alar_operatorpLENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     function)](api/l              | pp_api.html#_CPPv4N5cudaq15scalar |
| anguages/cpp_api.html#_CPPv4I0ENK | _operatorpLERK15scalar_operator), |
| 5cudaq11noise_model12get_channels |     [\[2                          |
| ENSt6vectorI13kraus_channelEERKNS | \]](api/languages/cpp_api.html#_C |
| t6vectorINSt6size_tEEERKNSt6vecto | PPv4N5cudaq15scalar_operatorpLEd) |
| rINSt6size_tEEERKNSt6vectorIdEE), | -   [                             |
|     [\[1\]](api/languages/cpp_a   | cudaq::scalar_operator::operator- |
| pi.html#_CPPv4NK5cudaq11noise_mod |     (C++                          |
| el12get_channelsERKNSt6stringERKN |     function                      |
| St6vectorINSt6size_tEEERKNSt6vect | )](api/languages/cpp_api.html#_CP |
| orINSt6size_tEEERKNSt6vectorIdEE) | Pv4N5cudaq15scalar_operatormiENSt |
| -                                 | 7complexIdEERK15scalar_operator), |
|  [cudaq::noise_model::noise_model |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api                | Pv4N5cudaq15scalar_operatormiENSt |
| /languages/cpp_api.html#_CPPv4N5c | 7complexIdEERR15scalar_operator), |
| udaq11noise_model11noise_modelEv) |     [\[2\]](api/languages/cp      |
| -   [cu                           | p_api.html#_CPPv4N5cudaq15scalar_ |
| daq::noise_model::PredicateFuncTy | operatormiEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     type)](api/la                 | p_api.html#_CPPv4N5cudaq15scalar_ |
| nguages/cpp_api.html#_CPPv4N5cuda | operatormiEdRR15scalar_operator), |
| q11noise_model15PredicateFuncTyE) |     [\[4\]](api/languages         |
| -   [cud                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| aq::noise_model::register_channel | alar_operatormiENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     function)](api/languages      | _api.html#_CPPv4NKR5cudaq15scalar |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | _operatormiERK15scalar_operator), |
| noise_model16register_channelEvv) |     [\[6\]]                       |
| -   [cudaq::                      | (api/languages/cpp_api.html#_CPPv |
| noise_model::requires_constructor | 4NKR5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[7\]]                       |
|     type)](api/languages/cp       | (api/languages/cpp_api.html#_CPPv |
| p_api.html#_CPPv4I0DpEN5cudaq11no | 4NKR5cudaq15scalar_operatormiEv), |
| ise_model20requires_constructorE) |     [\[8\]](api/language          |
| -   [cudaq::noise_model_type (C++ | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     e                             | alar_operatormiENSt7complexIdEE), |
| num)](api/languages/cpp_api.html# |     [\[9\]](api/languages/cp      |
| _CPPv4N5cudaq16noise_model_typeE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::no                    | _operatormiERK15scalar_operator), |
| ise_model_type::amplitude_damping |     [\[10\]                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     enumerator)](api/languages    | v4NO5cudaq15scalar_operatormiEd), |
| /cpp_api.html#_CPPv4N5cudaq16nois |     [\[11\                        |
| e_model_type17amplitude_dampingE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise_mode            | Pv4NO5cudaq15scalar_operatormiEv) |
| l_type::amplitude_damping_channel | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator-= |
|     e                             |     (C++                          |
| numerator)](api/languages/cpp_api |     function)](api/languag        |
| .html#_CPPv4N5cudaq16noise_model_ | es/cpp_api.html#_CPPv4N5cudaq15sc |
| type25amplitude_damping_channelE) | alar_operatormIENSt7complexIdEE), |
| -   [cudaq::n                     |     [\[1\]](api/languages/c       |
| oise_model_type::bit_flip_channel | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatormIERK15scalar_operator), |
|     enumerator)](api/language     |     [\[2                          |
| s/cpp_api.html#_CPPv4N5cudaq16noi | \]](api/languages/cpp_api.html#_C |
| se_model_type16bit_flip_channelE) | PPv4N5cudaq15scalar_operatormIEd) |
| -   [cudaq::                      | -   [                             |
| noise_model_type::depolarization1 | cudaq::scalar_operator::operator/ |
|     (C++                          |     (C++                          |
|     enumerator)](api/languag      |     function                      |
| es/cpp_api.html#_CPPv4N5cudaq16no | )](api/languages/cpp_api.html#_CP |
| ise_model_type15depolarization1E) | Pv4N5cudaq15scalar_operatordvENSt |
| -   [cudaq::                      | 7complexIdEERK15scalar_operator), |
| noise_model_type::depolarization2 |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](api/languag      | Pv4N5cudaq15scalar_operatordvENSt |
| es/cpp_api.html#_CPPv4N5cudaq16no | 7complexIdEERR15scalar_operator), |
| ise_model_type15depolarization2E) |     [\[2\]](api/languages/cp      |
| -   [cudaq::noise_m               | p_api.html#_CPPv4N5cudaq15scalar_ |
| odel_type::depolarization_channel | operatordvEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|   enumerator)](api/languages/cpp_ | operatordvEdRR15scalar_operator), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[4\]](api/languages         |
| el_type22depolarization_channelE) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -                                 | alar_operatordvENSt7complexIdEE), |
|  [cudaq::noise_model_type::pauli1 |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|     enumerator)](a                | _operatordvERK15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[6\]]                       |
| 5cudaq16noise_model_type6pauli1E) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatordvEd), |
|  [cudaq::noise_model_type::pauli2 |     [\[7\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     enumerator)](a                | alar_operatordvENSt7complexIdEE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[8\]](api/languages/cp      |
| 5cudaq16noise_model_type6pauli2E) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq                        | _operatordvERK15scalar_operator), |
| ::noise_model_type::phase_damping |     [\[9\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](api/langu        | Pv4NO5cudaq15scalar_operatordvEd) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [c                            |
| noise_model_type13phase_dampingE) | udaq::scalar_operator::operator/= |
| -   [cudaq::noi                   |     (C++                          |
| se_model_type::phase_flip_channel |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     enumerator)](api/languages/   | alar_operatordVENSt7complexIdEE), |
| cpp_api.html#_CPPv4N5cudaq16noise |     [\[1\]](api/languages/c       |
| _model_type18phase_flip_channelE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _operatordVERK15scalar_operator), |
| [cudaq::noise_model_type::unknown |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     enumerator)](ap               | PPv4N5cudaq15scalar_operatordVEd) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [                             |
| cudaq16noise_model_type7unknownE) | cudaq::scalar_operator::operator= |
| -                                 |     (C++                          |
| [cudaq::noise_model_type::x_error |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](ap               | _operatoraSERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[1\]](api/languages/        |
| cudaq16noise_model_type7x_errorE) | cpp_api.html#_CPPv4N5cudaq15scala |
| -                                 | r_operatoraSERR15scalar_operator) |
| [cudaq::noise_model_type::y_error | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator== |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languages/c    |
| cudaq16noise_model_type7y_errorE) | pp_api.html#_CPPv4NK5cudaq15scala |
| -                                 | r_operatoreqERK15scalar_operator) |
| [cudaq::noise_model_type::z_error | -   [cudaq:                       |
|     (C++                          | :scalar_operator::scalar_operator |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     func                          |
| cudaq16noise_model_type7z_errorE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::num_available_gpus    | #_CPPv4N5cudaq15scalar_operator15 |
|     (C++                          | scalar_operatorENSt7complexIdEE), |
|     function                      |     [\[1\]](api/langu             |
| )](api/languages/cpp_api.html#_CP | ages/cpp_api.html#_CPPv4N5cudaq15 |
| Pv4N5cudaq18num_available_gpusEv) | scalar_operator15scalar_operatorE |
| -   [cudaq::observe (C++          | RK15scalar_callbackRRNSt13unorder |
|     function)]                    | ed_mapINSt6stringENSt6stringEEE), |
| (api/languages/cpp_api.html#_CPPv |     [\[2\                         |
| 4I00DpEN5cudaq7observeENSt6vector | ]](api/languages/cpp_api.html#_CP |
| I14observe_resultEERR13QuantumKer | Pv4N5cudaq15scalar_operator15scal |
| nelRK15SpinOpContainerDpRR4Args), | ar_operatorERK15scalar_operator), |
|     [\[1\]](api/languages/cpp_ap  |     [\[3\]](api/langu             |
| i.html#_CPPv4I0DpEN5cudaq7observe | ages/cpp_api.html#_CPPv4N5cudaq15 |
| E14observe_resultNSt6size_tERR13Q | scalar_operator15scalar_operatorE |
| uantumKernelRK7spin_opDpRR4Args), | RR15scalar_callbackRRNSt13unorder |
|     [\[                           | ed_mapINSt6stringENSt6stringEEE), |
| 2\]](api/languages/cpp_api.html#_ |     [\[4\                         |
| CPPv4I0DpEN5cudaq7observeE14obser | ]](api/languages/cpp_api.html#_CP |
| ve_resultRK15observe_optionsRR13Q | Pv4N5cudaq15scalar_operator15scal |
| uantumKernelRK7spin_opDpRR4Args), | ar_operatorERR15scalar_operator), |
|     [\[3\]](api/lang              |     [\[5\]](api/language          |
| uages/cpp_api.html#_CPPv4I0DpEN5c | s/cpp_api.html#_CPPv4N5cudaq15sca |
| udaq7observeE14observe_resultRR13 | lar_operator15scalar_operatorEd), |
| QuantumKernelRK7spin_opDpRR4Args) |     [\[6\]](api/languag           |
| -   [cudaq::observe_options (C++  | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     st                            | alar_operator15scalar_operatorEv) |
| ruct)](api/languages/cpp_api.html | -   [                             |
| #_CPPv4N5cudaq15observe_optionsE) | cudaq::scalar_operator::to_matrix |
| -   [cudaq::observe_result (C++   |     (C++                          |
|                                   |                                   |
| class)](api/languages/cpp_api.htm |   function)](api/languages/cpp_ap |
| l#_CPPv4N5cudaq14observe_resultE) | i.html#_CPPv4NK5cudaq15scalar_ope |
| -                                 | rator9to_matrixERKNSt13unordered_ |
|    [cudaq::observe_result::counts | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [                             |
|     function)](api/languages/c    | cudaq::scalar_operator::to_string |
| pp_api.html#_CPPv4N5cudaq14observ |     (C++                          |
| e_result6countsERK12spin_op_term) |     function)](api/l              |
| -   [cudaq::observe_result::dump  | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq15scalar_operator9to_stringEv) |
|     function)                     | -   [cudaq::s                     |
| ](api/languages/cpp_api.html#_CPP | calar_operator::\~scalar_operator |
| v4N5cudaq14observe_result4dumpEv) |     (C++                          |
| -   [c                            |     functio                       |
| udaq::observe_result::expectation | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatorD0Ev) |
|                                   | -   [cudaq::set_noise (C++        |
| function)](api/languages/cpp_api. |     function)](api/langu          |
| html#_CPPv4N5cudaq14observe_resul | ages/cpp_api.html#_CPPv4N5cudaq9s |
| t11expectationERK12spin_op_term), | et_noiseERKN5cudaq11noise_modelE) |
|     [\[1\]](api/la                | -   [cudaq::set_random_seed (C++  |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/               |
| q14observe_result11expectationEv) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cuda                         | daq15set_random_seedENSt6size_tE) |
| q::observe_result::id_coefficient | -   [cudaq::simulation_precision  |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     enum)                         |
| ages/cpp_api.html#_CPPv4N5cudaq14 | ](api/languages/cpp_api.html#_CPP |
| observe_result14id_coefficientEv) | v4N5cudaq20simulation_precisionE) |
| -   [cuda                         | -   [                             |
| q::observe_result::observe_result | cudaq::simulation_precision::fp32 |
|     (C++                          |     (C++                          |
|                                   |     enumerator)](api              |
|   function)](api/languages/cpp_ap | /languages/cpp_api.html#_CPPv4N5c |
| i.html#_CPPv4N5cudaq14observe_res | udaq20simulation_precision4fp32E) |
| ult14observe_resultEdRK7spin_op), | -   [                             |
|     [\[1\]](a                     | cudaq::simulation_precision::fp64 |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14observe_result14observe_r |     enumerator)](api              |
| esultEdRK7spin_op13sample_result) | /languages/cpp_api.html#_CPPv4N5c |
| -                                 | udaq20simulation_precision4fp64E) |
|  [cudaq::observe_result::operator | -   [cudaq::SimulationState (C++  |
|     double (C++                   |     c                             |
|     functio                       | lass)](api/languages/cpp_api.html |
| n)](api/languages/cpp_api.html#_C | #_CPPv4N5cudaq15SimulationStateE) |
| PPv4N5cudaq14observe_resultcvdEv) | -   [                             |
| -                                 | cudaq::SimulationState::precision |
|  [cudaq::observe_result::raw_data |     (C++                          |
|     (C++                          |     enum)](api                    |
|     function)](ap                 | /languages/cpp_api.html#_CPPv4N5c |
| i/languages/cpp_api.html#_CPPv4N5 | udaq15SimulationState9precisionE) |
| cudaq14observe_result8raw_dataEv) | -   [cudaq:                       |
| -   [cudaq::operator_handler (C++ | :SimulationState::precision::fp32 |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     enumerator)](api/lang         |
| _CPPv4N5cudaq16operator_handlerE) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [cudaq::optimizable_function  | 5SimulationState9precision4fp32E) |
|     (C++                          | -   [cudaq:                       |
|     class)                        | :SimulationState::precision::fp64 |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq20optimizable_functionE) |     enumerator)](api/lang         |
| -   [cudaq::optimization_result   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     (C++                          | 5SimulationState9precision4fp64E) |
|     type                          | -                                 |
| )](api/languages/cpp_api.html#_CP |   [cudaq::SimulationState::Tensor |
| Pv4N5cudaq19optimization_resultE) |     (C++                          |
| -   [cudaq::optimizer (C++        |     struct)](                     |
|     class)](api/languages/cpp_a   | api/languages/cpp_api.html#_CPPv4 |
| pi.html#_CPPv4N5cudaq9optimizerE) | N5cudaq15SimulationState6TensorE) |
| -   [cudaq::optimizer::optimize   | -   [cudaq::spin_handler (C++     |
|     (C++                          |                                   |
|                                   |   class)](api/languages/cpp_api.h |
|  function)](api/languages/cpp_api | tml#_CPPv4N5cudaq12spin_handlerE) |
| .html#_CPPv4N5cudaq9optimizer8opt | -   [cudaq:                       |
| imizeEKiRR20optimizable_function) | :spin_handler::to_diagonal_matrix |
| -   [cu                           |     (C++                          |
| daq::optimizer::requiresGradients |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4NK5cud |
|     function)](api/la             | aq12spin_handler18to_diagonal_mat |
| nguages/cpp_api.html#_CPPv4N5cuda | rixERNSt13unordered_mapINSt6size_ |
| q9optimizer17requiresGradientsEv) | tENSt7int64_tEEERKNSt13unordered_ |
| -   [cudaq::orca (C++             | mapINSt6stringENSt7complexIdEEEE) |
|     type)](api/languages/         | -                                 |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |   [cudaq::spin_handler::to_matrix |
| -   [cudaq::orca::sample (C++     |     (C++                          |
|     function)](api/languages/c    |     function                      |
| pp_api.html#_CPPv4N5cudaq4orca6sa | )](api/languages/cpp_api.html#_CP |
| mpleERNSt6vectorINSt6size_tEEERNS | Pv4N5cudaq12spin_handler9to_matri |
| t6vectorINSt6size_tEEERNSt6vector | xERKNSt6stringENSt7complexIdEEb), |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     [\[1                          |
|     [\[1\]]                       | \]](api/languages/cpp_api.html#_C |
| (api/languages/cpp_api.html#_CPPv | PPv4NK5cudaq12spin_handler9to_mat |
| 4N5cudaq4orca6sampleERNSt6vectorI | rixERNSt13unordered_mapINSt6size_ |
| NSt6size_tEEERNSt6vectorINSt6size | tENSt7int64_tEEERKNSt13unordered_ |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::orca::sample_async    | -   [cuda                         |
|     (C++                          | q::spin_handler::to_sparse_matrix |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     function)](api/               |
| html#_CPPv4N5cudaq4orca12sample_a | languages/cpp_api.html#_CPPv4N5cu |
| syncERNSt6vectorINSt6size_tEEERNS | daq12spin_handler16to_sparse_matr |
| t6vectorINSt6size_tEEERNSt6vector | ixERKNSt6stringENSt7complexIdEEb) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -                                 |
|     [\[1\]](api/la                |   [cudaq::spin_handler::to_string |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q4orca12sample_asyncERNSt6vectorI |     function)](ap                 |
| NSt6size_tEEERNSt6vectorINSt6size | i/languages/cpp_api.html#_CPPv4NK |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | 5cudaq12spin_handler9to_stringEb) |
| -   [cudaq::OrcaRemoteRESTQPU     | -                                 |
|     (C++                          |   [cudaq::spin_handler::unique_id |
|     cla                           |     (C++                          |
| ss)](api/languages/cpp_api.html#_ |     function)](ap                 |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::pauli1 (C++           | 5cudaq12spin_handler9unique_idEv) |
|     class)](api/languages/cp      | -   [cudaq::spin_op (C++          |
| p_api.html#_CPPv4N5cudaq6pauli1E) |     type)](api/languages/cpp      |
| -                                 | _api.html#_CPPv4N5cudaq7spin_opE) |
|    [cudaq::pauli1::num_parameters | -   [cudaq::spin_op_term (C++     |
|     (C++                          |                                   |
|     member)]                      |    type)](api/languages/cpp_api.h |
| (api/languages/cpp_api.html#_CPPv | tml#_CPPv4N5cudaq12spin_op_termE) |
| 4N5cudaq6pauli114num_parametersE) | -   [cudaq::state (C++            |
| -   [cudaq::pauli1::num_targets   |     class)](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq5stateE) |
|     membe                         | -   [cudaq::state::amplitude (C++ |
| r)](api/languages/cpp_api.html#_C |     function)](api/lang           |
| PPv4N5cudaq6pauli111num_targetsE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
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
|         static                    | -   [is_constant()                |
|         method)](api/languages    |                                   |
| /python_api.html#cudaq.operators. |   (cudaq.operators.ScalarOperator |
| fermion.FermionOperator.identity) |     method)](api/lan              |
|     -                             | guages/python_api.html#cudaq.oper |
|  [(cudaq.operators.MatrixOperator | ators.ScalarOperator.is_constant) |
|         static                    | -   [is_emulated() (cudaq.Target  |
|         method)](api/             |                                   |
| languages/python_api.html#cudaq.o |   method)](api/languages/python_a |
| perators.MatrixOperator.identity) | pi.html#cudaq.Target.is_emulated) |
|     -   [(                        | -   [is_identity()                |
| cudaq.operators.spin.SpinOperator |     (cudaq.                       |
|         static                    | operators.boson.BosonOperatorTerm |
|         method)](api/lan          |     method)](api/languages/py     |
| guages/python_api.html#cudaq.oper | thon_api.html#cudaq.operators.bos |
| ators.spin.SpinOperator.identity) | on.BosonOperatorTerm.is_identity) |
|     -   [(in module               |     -   [(cudaq.oper              |
|                                   | ators.fermion.FermionOperatorTerm |
|  cudaq.boson)](api/languages/pyth |                                   |
| on_api.html#cudaq.boson.identity) |     method)](api/languages/python |
|     -   [(in module               | _api.html#cudaq.operators.fermion |
|         cud                       | .FermionOperatorTerm.is_identity) |
| aq.fermion)](api/languages/python |     -   [(c                       |
| _api.html#cudaq.fermion.identity) | udaq.operators.MatrixOperatorTerm |
|     -   [(in module               |         method)](api/languag      |
|                                   | es/python_api.html#cudaq.operator |
|    cudaq.spin)](api/languages/pyt | s.MatrixOperatorTerm.is_identity) |
| hon_api.html#cudaq.spin.identity) |     -   [(                        |
| -   [initial_parameters           | cudaq.operators.spin.SpinOperator |
|     (cudaq.optimizers.Adam        |         method)](api/langua       |
|     property)](api/l              | ges/python_api.html#cudaq.operato |
| anguages/python_api.html#cudaq.op | rs.spin.SpinOperator.is_identity) |
| timizers.Adam.initial_parameters) |     -   [(cuda                    |
|     -   [(cudaq.optimizers.COBYLA | q.operators.spin.SpinOperatorTerm |
|         property)](api/lan        |         method)](api/languages/   |
| guages/python_api.html#cudaq.opti | python_api.html#cudaq.operators.s |
| mizers.COBYLA.initial_parameters) | pin.SpinOperatorTerm.is_identity) |
|     -   [                         | -   [is_initialized() (in module  |
| (cudaq.optimizers.GradientDescent |     c                             |
|                                   | udaq.mpi)](api/languages/python_a |
|       property)](api/languages/py | pi.html#cudaq.mpi.is_initialized) |
| thon_api.html#cudaq.optimizers.Gr | -   [is_on_gpu() (cudaq.State     |
| adientDescent.initial_parameters) |     method)](api/languages/pytho  |
|     -   [(cudaq.optimizers.LBFGS  | n_api.html#cudaq.State.is_on_gpu) |
|         property)](api/la         | -   [is_remote() (cudaq.Target    |
| nguages/python_api.html#cudaq.opt |     method)](api/languages/python |
| imizers.LBFGS.initial_parameters) | _api.html#cudaq.Target.is_remote) |
|                                   | -   [is_remote_simulator()        |
| -   [(cudaq.optimizers.NelderMead |     (cudaq.Target                 |
|         property)](api/languag    |     method                        |
| es/python_api.html#cudaq.optimize | )](api/languages/python_api.html# |
| rs.NelderMead.initial_parameters) | cudaq.Target.is_remote_simulator) |
|     -   [(cudaq.optimizers.SGD    | -   [items() (cudaq.SampleResult  |
|         property)](api/           |                                   |
| languages/python_api.html#cudaq.o |   method)](api/languages/python_a |
| ptimizers.SGD.initial_parameters) | pi.html#cudaq.SampleResult.items) |
|     -   [(cudaq.optimizers.SPSA   |                                   |
|         property)](api/l          |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.SPSA.initial_parameters) |                                   |
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
| -   [parameters                   | -   [Pauli1 (class in             |
|     (cu                           |     cudaq)](api/langua            |
| daq.operators.boson.BosonOperator | ges/python_api.html#cudaq.Pauli1) |
|     property)](api/languag        | -   [Pauli2 (class in             |
| es/python_api.html#cudaq.operator |     cudaq)](api/langua            |
| s.boson.BosonOperator.parameters) | ges/python_api.html#cudaq.Pauli2) |
|     -   [(cudaq.                  | -   [PhaseDamping (class in       |
| operators.boson.BosonOperatorTerm |     cudaq)](api/languages/py      |
|                                   | thon_api.html#cudaq.PhaseDamping) |
|        property)](api/languages/p | -   [PhaseFlipChannel (class in   |
| ython_api.html#cudaq.operators.bo |     cudaq)](api/languages/python  |
| son.BosonOperatorTerm.parameters) | _api.html#cudaq.PhaseFlipChannel) |
|     -   [(cudaq.                  | -   [platform (cudaq.Target       |
| operators.fermion.FermionOperator |                                   |
|                                   |    property)](api/languages/pytho |
|        property)](api/languages/p | n_api.html#cudaq.Target.platform) |
| ython_api.html#cudaq.operators.fe | -   [plus() (in module            |
| rmion.FermionOperator.parameters) |     cudaq.spin)](api/languages    |
|     -   [(cudaq.oper              | /python_api.html#cudaq.spin.plus) |
| ators.fermion.FermionOperatorTerm | -   [position() (in module        |
|                                   |                                   |
|    property)](api/languages/pytho |  cudaq.boson)](api/languages/pyth |
| n_api.html#cudaq.operators.fermio | on_api.html#cudaq.boson.position) |
| n.FermionOperatorTerm.parameters) |     -   [(in module               |
|     -                             |         cudaq.operators.custo     |
|  [(cudaq.operators.MatrixOperator | m)](api/languages/python_api.html |
|         property)](api/la         | #cudaq.operators.custom.position) |
| nguages/python_api.html#cudaq.ope | -   [pre_compile()                |
| rators.MatrixOperator.parameters) |     (cudaq.PyKernelDecorator      |
|     -   [(cuda                    |     method)](                     |
| q.operators.MatrixOperatorElement | api/languages/python_api.html#cud |
|         property)](api/languages  | aq.PyKernelDecorator.pre_compile) |
| /python_api.html#cudaq.operators. | -   [probability()                |
| MatrixOperatorElement.parameters) |     (cudaq.SampleResult           |
|     -   [(c                       |     meth                          |
| udaq.operators.MatrixOperatorTerm | od)](api/languages/python_api.htm |
|         property)](api/langua     | l#cudaq.SampleResult.probability) |
| ges/python_api.html#cudaq.operato | -   [ProductOperator (in module   |
| rs.MatrixOperatorTerm.parameters) |     cudaq.operator                |
|     -                             | s)](api/languages/python_api.html |
|  [(cudaq.operators.ScalarOperator | #cudaq.operators.ProductOperator) |
|         property)](api/la         | -   [PyKernel (class in           |
| nguages/python_api.html#cudaq.ope |     cudaq)](api/language          |
| rators.ScalarOperator.parameters) | s/python_api.html#cudaq.PyKernel) |
|     -   [(                        | -   [PyKernelDecorator (class in  |
| cudaq.operators.spin.SpinOperator |     cudaq)](api/languages/python_ |
|         property)](api/langu      | api.html#cudaq.PyKernelDecorator) |
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
| -   [Pauli (class in              |                                   |
|     cudaq.spin)](api/languages/   |                                   |
| python_api.html#cudaq.spin.Pauli) |                                   |
+-----------------------------------+-----------------------------------+

## Q {#Q}

+-----------------------------------+-----------------------------------+
| -   [qreg (in module              | -   [qubit_count                  |
|     cudaq)](api/lang              |     (                             |
| uages/python_api.html#cudaq.qreg) | cudaq.operators.spin.SpinOperator |
| -   [QuakeValue (class in         |     property)](api/langua         |
|     cudaq)](api/languages/        | ges/python_api.html#cudaq.operato |
| python_api.html#cudaq.QuakeValue) | rs.spin.SpinOperator.qubit_count) |
| -   [qubit (class in              |     -   [(cuda                    |
|     cudaq)](api/langu             | q.operators.spin.SpinOperatorTerm |
| ages/python_api.html#cudaq.qubit) |         property)](api/languages/ |
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
| -   [sample() (in module          | -   [signatureWithCallables()     |
|     cudaq)](api/langua            |     (cudaq.PyKernelDecorator      |
| ges/python_api.html#cudaq.sample) |     method)](api/languag          |
|     -   [(in module               | es/python_api.html#cudaq.PyKernel |
|                                   | Decorator.signatureWithCallables) |
|      cudaq.orca)](api/languages/p | -   [SimulationPrecision (class   |
| ython_api.html#cudaq.orca.sample) |     in                            |
| -   [sample_async() (in module    |                                   |
|     cudaq)](api/languages/py      |   cudaq)](api/languages/python_ap |
| thon_api.html#cudaq.sample_async) | i.html#cudaq.SimulationPrecision) |
| -   [SampleResult (class in       | -   [simulator (cudaq.Target      |
|     cudaq)](api/languages/py      |                                   |
| thon_api.html#cudaq.SampleResult) |   property)](api/languages/python |
| -   [ScalarOperator (class in     | _api.html#cudaq.Target.simulator) |
|     cudaq.operato                 | -   [slice() (cudaq.QuakeValue    |
| rs)](api/languages/python_api.htm |     method)](api/languages/python |
| l#cudaq.operators.ScalarOperator) | _api.html#cudaq.QuakeValue.slice) |
| -   [Schedule (class in           | -   [SpinOperator (class in       |
|     cudaq)](api/language          |     cudaq.operators.spin)         |
| s/python_api.html#cudaq.Schedule) | ](api/languages/python_api.html#c |
| -   [serialize()                  | udaq.operators.spin.SpinOperator) |
|     (                             | -   [SpinOperatorElement (class   |
| cudaq.operators.spin.SpinOperator |     in                            |
|     method)](api/lang             |     cudaq.operators.spin)](api/l  |
| uages/python_api.html#cudaq.opera | anguages/python_api.html#cudaq.op |
| tors.spin.SpinOperator.serialize) | erators.spin.SpinOperatorElement) |
|     -   [(cuda                    | -   [SpinOperatorTerm (class in   |
| q.operators.spin.SpinOperatorTerm |     cudaq.operators.spin)](ap     |
|         method)](api/language     | i/languages/python_api.html#cudaq |
| s/python_api.html#cudaq.operators | .operators.spin.SpinOperatorTerm) |
| .spin.SpinOperatorTerm.serialize) | -   [SPSA (class in               |
|     -   [(cudaq.SampleResult      |     cudaq                         |
|         me                        | .optimizers)](api/languages/pytho |
| thod)](api/languages/python_api.h | n_api.html#cudaq.optimizers.SPSA) |
| tml#cudaq.SampleResult.serialize) | -   [squeeze() (in module         |
| -   [set_noise() (in module       |     cudaq.operators.cust          |
|     cudaq)](api/languages         | om)](api/languages/python_api.htm |
| /python_api.html#cudaq.set_noise) | l#cudaq.operators.custom.squeeze) |
| -   [set_random_seed() (in module | -   [State (class in              |
|     cudaq)](api/languages/pytho   |     cudaq)](api/langu             |
| n_api.html#cudaq.set_random_seed) | ages/python_api.html#cudaq.State) |
| -   [set_target() (in module      | -   [step_size                    |
|     cudaq)](api/languages/        |     (cudaq.optimizers.Adam        |
| python_api.html#cudaq.set_target) |     propert                       |
| -   [SGD (class in                | y)](api/languages/python_api.html |
|     cuda                          | #cudaq.optimizers.Adam.step_size) |
| q.optimizers)](api/languages/pyth |     -   [(cudaq.optimizers.SGD    |
| on_api.html#cudaq.optimizers.SGD) |         proper                    |
|                                   | ty)](api/languages/python_api.htm |
|                                   | l#cudaq.optimizers.SGD.step_size) |
|                                   |     -   [(cudaq.optimizers.SPSA   |
|                                   |         propert                   |
|                                   | y)](api/languages/python_api.html |
|                                   | #cudaq.optimizers.SPSA.step_size) |
|                                   | -   [SuperOperator (class in      |
|                                   |     cudaq)](api/languages/pyt     |
|                                   | hon_api.html#cudaq.SuperOperator) |
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
