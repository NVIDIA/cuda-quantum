::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4032
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
| -   [ComplexMatrix (class in      |     (C++                          |
|     cudaq)](api/languages/pyt     |     function)](api/lang           |
| hon_api.html#cudaq.ComplexMatrix) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [compute()                    | 10product_op14const_iteratormlEv) |
|     (                             | -   [cudaq::produ                 |
| cudaq.gradients.CentralDifference | ct_op::const_iterator::operator++ |
|     method)](api/la               |     (C++                          |
| nguages/python_api.html#cudaq.gra |     function)](api/lang           |
| dients.CentralDifference.compute) | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     -   [(                        | 0product_op14const_iteratorppEi), |
| cudaq.gradients.ForwardDifference |     [\[1\]](api/lan               |
|         method)](api/la           | guages/cpp_api.html#_CPPv4N5cudaq |
| nguages/python_api.html#cudaq.gra | 10product_op14const_iteratorppEv) |
| dients.ForwardDifference.compute) | -   [cudaq::produc                |
|     -                             | t_op::const_iterator::operator\-- |
|  [(cudaq.gradients.ParameterShift |     (C++                          |
|         method)](api              |     function)](api/lang           |
| /languages/python_api.html#cudaq. | uages/cpp_api.html#_CPPv4N5cudaq1 |
| gradients.ParameterShift.compute) | 0product_op14const_iteratormmEi), |
| -   [const()                      |     [\[1\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|   (cudaq.operators.ScalarOperator | 10product_op14const_iteratormmEv) |
|     class                         | -   [cudaq::produc                |
|     method)](a                    | t_op::const_iterator::operator-\> |
| pi/languages/python_api.html#cuda |     (C++                          |
| q.operators.ScalarOperator.const) |     function)](api/lan            |
| -   [copy()                       | guages/cpp_api.html#_CPPv4N5cudaq |
|     (cu                           | 10product_op14const_iteratorptEv) |
| daq.operators.boson.BosonOperator | -   [cudaq::produ                 |
|     method)](api/l                | ct_op::const_iterator::operator== |
| anguages/python_api.html#cudaq.op |     (C++                          |
| erators.boson.BosonOperator.copy) |     fun                           |
|     -   [(cudaq.                  | ction)](api/languages/cpp_api.htm |
| operators.boson.BosonOperatorTerm | l#_CPPv4NK5cudaq10product_op14con |
|         method)](api/langu        | st_iteratoreqERK14const_iterator) |
| ages/python_api.html#cudaq.operat | -   [cudaq::product_op::degrees   |
| ors.boson.BosonOperatorTerm.copy) |     (C++                          |
|     -   [(cudaq.                  |     function)                     |
| operators.fermion.FermionOperator | ](api/languages/cpp_api.html#_CPP |
|         method)](api/langu        | v4NK5cudaq10product_op7degreesEv) |
| ages/python_api.html#cudaq.operat | -   [cudaq::product_op::dump (C++ |
| ors.fermion.FermionOperator.copy) |     functi                        |
|     -   [(cudaq.oper              | on)](api/languages/cpp_api.html#_ |
| ators.fermion.FermionOperatorTerm | CPPv4NK5cudaq10product_op4dumpEv) |
|         method)](api/languages    | -   [cudaq::product_op::end (C++  |
| /python_api.html#cudaq.operators. |     funct                         |
| fermion.FermionOperatorTerm.copy) | ion)](api/languages/cpp_api.html# |
|     -                             | _CPPv4NK5cudaq10product_op3endEv) |
|  [(cudaq.operators.MatrixOperator | -   [c                            |
|         method)](                 | udaq::product_op::get_coefficient |
| api/languages/python_api.html#cud |     (C++                          |
| aq.operators.MatrixOperator.copy) |     function)](api/lan            |
|     -   [(c                       | guages/cpp_api.html#_CPPv4NK5cuda |
| udaq.operators.MatrixOperatorTerm | q10product_op15get_coefficientEv) |
|         method)](api/             | -                                 |
| languages/python_api.html#cudaq.o |   [cudaq::product_op::get_term_id |
| perators.MatrixOperatorTerm.copy) |     (C++                          |
|     -   [(                        |     function)](api                |
| cudaq.operators.spin.SpinOperator | /languages/cpp_api.html#_CPPv4NK5 |
|         method)](api              | cudaq10product_op11get_term_idEv) |
| /languages/python_api.html#cudaq. | -                                 |
| operators.spin.SpinOperator.copy) |   [cudaq::product_op::is_identity |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     function)](api                |
|         method)](api/lan          | /languages/cpp_api.html#_CPPv4NK5 |
| guages/python_api.html#cudaq.oper | cudaq10product_op11is_identityEv) |
| ators.spin.SpinOperatorTerm.copy) | -   [cudaq::product_op::num_ops   |
| -   [count() (cudaq.Resources     |     (C++                          |
|     method)](api/languages/pytho  |     function)                     |
| n_api.html#cudaq.Resources.count) | ](api/languages/cpp_api.html#_CPP |
|     -   [(cudaq.SampleResult      | v4NK5cudaq10product_op7num_opsEv) |
|                                   | -                                 |
|   method)](api/languages/python_a |    [cudaq::product_op::operator\* |
| pi.html#cudaq.SampleResult.count) |     (C++                          |
| -   [count_controls()             |     function)](api/languages/     |
|     (cudaq.Resources              | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     meth                          | oduct_opmlE10product_opI1TERK15sc |
| od)](api/languages/python_api.htm | alar_operatorRK10product_opI1TE), |
| l#cudaq.Resources.count_controls) |     [\[1\]](api/languages/        |
| -   [counts()                     | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     (cudaq.ObserveResult          | oduct_opmlE10product_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
| method)](api/languages/python_api |     [\[2\]](api/languages/        |
| .html#cudaq.ObserveResult.counts) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [create() (in module          | oduct_opmlE10product_opI1TERR15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|    cudaq.boson)](api/languages/py |     [\[3\]](api/languages/        |
| thon_api.html#cudaq.boson.create) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(in module               | oduct_opmlE10product_opI1TERR15sc |
|         c                         | alar_operatorRR10product_opI1TE), |
| udaq.fermion)](api/languages/pyth |     [\[4\]](api/                  |
| on_api.html#cudaq.fermion.create) | languages/cpp_api.html#_CPPv4I0EN |
| -   [csr_spmatrix (C++            | 5cudaq10product_opmlE6sum_opI1TER |
|     type)](api/languages/c        | K15scalar_operatorRK6sum_opI1TE), |
| pp_api.html#_CPPv412csr_spmatrix) |     [\[5\]](api/                  |
| -   cudaq                         | languages/cpp_api.html#_CPPv4I0EN |
|     -   [module](api/langua       | 5cudaq10product_opmlE6sum_opI1TER |
| ges/python_api.html#module-cudaq) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq (C++                   |     [\[6\]](api/                  |
|     type)](api/lan                | languages/cpp_api.html#_CPPv4I0EN |
| guages/cpp_api.html#_CPPv45cudaq) | 5cudaq10product_opmlE6sum_opI1TER |
| -   [cudaq.apply_noise() (in      | R15scalar_operatorRK6sum_opI1TE), |
|     module                        |     [\[7\]](api/                  |
|     cudaq)](api/languages/python_ | languages/cpp_api.html#_CPPv4I0EN |
| api.html#cudaq.cudaq.apply_noise) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq.boson                   | R15scalar_operatorRR6sum_opI1TE), |
|     -   [module](api/languages/py |     [\[8\]](api/languages         |
| thon_api.html#module-cudaq.boson) | /cpp_api.html#_CPPv4NK5cudaq10pro |
| -   cudaq.fermion                 | duct_opmlERK6sum_opI9HandlerTyE), |
|                                   |     [\[9\]](api/languages/cpp_a   |
|   -   [module](api/languages/pyth | pi.html#_CPPv4NKR5cudaq10product_ |
| on_api.html#module-cudaq.fermion) | opmlERK10product_opI9HandlerTyE), |
| -   cudaq.operators.custom        |     [\[10\]](api/language         |
|     -   [mo                       | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| dule](api/languages/python_api.ht | roduct_opmlERK15scalar_operator), |
| ml#module-cudaq.operators.custom) |     [\[11\]](api/languages/cpp_a  |
| -   cudaq.spin                    | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [module](api/languages/p  | opmlERR10product_opI9HandlerTyE), |
| ython_api.html#module-cudaq.spin) |     [\[12\]](api/language         |
| -   [cudaq::amplitude_damping     | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opmlERR15scalar_operator), |
|     cla                           |     [\[13\]](api/languages/cpp_   |
| ss)](api/languages/cpp_api.html#_ | api.html#_CPPv4NO5cudaq10product_ |
| CPPv4N5cudaq17amplitude_dampingE) | opmlERK10product_opI9HandlerTyE), |
| -                                 |     [\[14\]](api/languag          |
| [cudaq::amplitude_damping_channel | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmlERK15scalar_operator), |
|     class)](api                   |     [\[15\]](api/languages/cpp_   |
| /languages/cpp_api.html#_CPPv4N5c | api.html#_CPPv4NO5cudaq10product_ |
| udaq25amplitude_damping_channelE) | opmlERR10product_opI9HandlerTyE), |
| -   [cudaq::amplitud              |     [\[16\]](api/langua           |
| e_damping_channel::num_parameters | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     (C++                          | product_opmlERR15scalar_operator) |
|     member)](api/languages/cpp_a  | -                                 |
| pi.html#_CPPv4N5cudaq25amplitude_ |   [cudaq::product_op::operator\*= |
| damping_channel14num_parametersE) |     (C++                          |
| -   [cudaq::ampli                 |     function)](api/languages/cpp  |
| tude_damping_channel::num_targets | _api.html#_CPPv4N5cudaq10product_ |
|     (C++                          | opmLERK10product_opI9HandlerTyE), |
|     member)](api/languages/cp     |     [\[1\]](api/langua            |
| p_api.html#_CPPv4N5cudaq25amplitu | ges/cpp_api.html#_CPPv4N5cudaq10p |
| de_damping_channel11num_targetsE) | roduct_opmLERK15scalar_operator), |
| -   [cudaq::AnalogRemoteRESTQPU   |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     class                         | _opmLERR10product_opI9HandlerTyE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::product_op::operator+ |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) |     (C++                          |
| -   [cudaq::apply_noise (C++      |     function)](api/langu          |
|     function)](api/               | ages/cpp_api.html#_CPPv4I0EN5cuda |
| languages/cpp_api.html#_CPPv4I0Dp | q10product_opplE6sum_opI1TERK15sc |
| EN5cudaq11apply_noiseEvDpRR4Args) | alar_operatorRK10product_opI1TE), |
| -   [cudaq::async_result (C++     |     [\[1\]](api/                  |
|     c                             | languages/cpp_api.html#_CPPv4I0EN |
| lass)](api/languages/cpp_api.html | 5cudaq10product_opplE6sum_opI1TER |
| #_CPPv4I0EN5cudaq12async_resultE) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::async_result::get     |     [\[2\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     functi                        | q10product_opplE6sum_opI1TERK15sc |
| on)](api/languages/cpp_api.html#_ | alar_operatorRR10product_opI1TE), |
| CPPv4N5cudaq12async_result3getEv) |     [\[3\]](api/                  |
| -   [cudaq::async_sample_result   | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     type                          | K15scalar_operatorRR6sum_opI1TE), |
| )](api/languages/cpp_api.html#_CP |     [\[4\]](api/langu             |
| Pv4N5cudaq19async_sample_resultE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::BaseRemoteRESTQPU     | q10product_opplE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRK10product_opI1TE), |
|     cla                           |     [\[5\]](api/                  |
| ss)](api/languages/cpp_api.html#_ | languages/cpp_api.html#_CPPv4I0EN |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | 5cudaq10product_opplE6sum_opI1TER |
| -                                 | R15scalar_operatorRK6sum_opI1TE), |
|    [cudaq::BaseRemoteSimulatorQPU |     [\[6\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     class)](                      | q10product_opplE6sum_opI1TERR15sc |
| api/languages/cpp_api.html#_CPPv4 | alar_operatorRR10product_opI1TE), |
| N5cudaq22BaseRemoteSimulatorQPUE) |     [\[7\]](api/                  |
| -   [cudaq::bit_flip_channel (C++ | languages/cpp_api.html#_CPPv4I0EN |
|     cl                            | 5cudaq10product_opplE6sum_opI1TER |
| ass)](api/languages/cpp_api.html# | R15scalar_operatorRR6sum_opI1TE), |
| _CPPv4N5cudaq16bit_flip_channelE) |     [\[8\]](api/languages/cpp_a   |
| -   [cudaq:                       | pi.html#_CPPv4NKR5cudaq10product_ |
| :bit_flip_channel::num_parameters | opplERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[9\]](api/language          |
|     member)](api/langua           | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ges/cpp_api.html#_CPPv4N5cudaq16b | roduct_opplERK15scalar_operator), |
| it_flip_channel14num_parametersE) |     [\[10\]](api/languages/       |
| -   [cud                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
| aq::bit_flip_channel::num_targets | duct_opplERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     member)](api/lan              | pi.html#_CPPv4NKR5cudaq10product_ |
| guages/cpp_api.html#_CPPv4N5cudaq | opplERR10product_opI9HandlerTyE), |
| 16bit_flip_channel11num_targetsE) |     [\[12\]](api/language         |
| -   [cudaq::boson_handler (C++    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opplERR15scalar_operator), |
|  class)](api/languages/cpp_api.ht |     [\[13\]](api/languages/       |
| ml#_CPPv4N5cudaq13boson_handlerE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::boson_op (C++         | duct_opplERR6sum_opI9HandlerTyE), |
|     type)](api/languages/cpp_     |     [\[                           |
| api.html#_CPPv4N5cudaq8boson_opE) | 14\]](api/languages/cpp_api.html# |
| -   [cudaq::boson_op_term (C++    | _CPPv4NKR5cudaq10product_opplEv), |
|                                   |     [\[15\]](api/languages/cpp_   |
|   type)](api/languages/cpp_api.ht | api.html#_CPPv4NO5cudaq10product_ |
| ml#_CPPv4N5cudaq13boson_op_termE) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq::CodeGenConfig (C++    |     [\[16\]](api/languag          |
|                                   | es/cpp_api.html#_CPPv4NO5cudaq10p |
| struct)](api/languages/cpp_api.ht | roduct_opplERK15scalar_operator), |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     [\[17\]](api/languages        |
| -   [cudaq::commutation_relations | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opplERK6sum_opI9HandlerTyE), |
|     struct)]                      |     [\[18\]](api/languages/cpp_   |
| (api/languages/cpp_api.html#_CPPv | api.html#_CPPv4NO5cudaq10product_ |
| 4N5cudaq21commutation_relationsE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::complex (C++          |     [\[19\]](api/languag          |
|     type)](api/languages/cpp      | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _api.html#_CPPv4N5cudaq7complexE) | roduct_opplERR15scalar_operator), |
| -   [cudaq::complex_matrix (C++   |     [\[20\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq10pro |
| class)](api/languages/cpp_api.htm | duct_opplERR6sum_opI9HandlerTyE), |
| l#_CPPv4N5cudaq14complex_matrixE) |     [                             |
| -                                 | \[21\]](api/languages/cpp_api.htm |
|   [cudaq::complex_matrix::adjoint | l#_CPPv4NO5cudaq10product_opplEv) |
|     (C++                          | -   [cudaq::product_op::operator- |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/langu          |
| 5cudaq14complex_matrix7adjointEv) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::                      | q10product_opmiE6sum_opI1TERK15sc |
| complex_matrix::diagonal_elements | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     function)](api/languages      | languages/cpp_api.html#_CPPv4I0EN |
| /cpp_api.html#_CPPv4NK5cudaq14com | 5cudaq10product_opmiE6sum_opI1TER |
| plex_matrix17diagonal_elementsEi) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::complex_matrix::dump  |     [\[2\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     function)](api/language       | q10product_opmiE6sum_opI1TERK15sc |
| s/cpp_api.html#_CPPv4NK5cudaq14co | alar_operatorRR10product_opI1TE), |
| mplex_matrix4dumpERNSt7ostreamE), |     [\[3\]](api/                  |
|     [\[1\]]                       | languages/cpp_api.html#_CPPv4I0EN |
| (api/languages/cpp_api.html#_CPPv | 5cudaq10product_opmiE6sum_opI1TER |
| 4NK5cudaq14complex_matrix4dumpEv) | K15scalar_operatorRR6sum_opI1TE), |
| -   [c                            |     [\[4\]](api/langu             |
| udaq::complex_matrix::eigenvalues | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERR15sc |
|     function)](api/lan            | alar_operatorRK10product_opI1TE), |
| guages/cpp_api.html#_CPPv4NK5cuda |     [\[5\]](api/                  |
| q14complex_matrix11eigenvaluesEv) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cu                           | 5cudaq10product_opmiE6sum_opI1TER |
| daq::complex_matrix::eigenvectors | R15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[6\]](api/langu             |
|     function)](api/lang           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| uages/cpp_api.html#_CPPv4NK5cudaq | q10product_opmiE6sum_opI1TERR15sc |
| 14complex_matrix12eigenvectorsEv) | alar_operatorRR10product_opI1TE), |
| -   [c                            |     [\[7\]](api/                  |
| udaq::complex_matrix::exponential | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/la             | R15scalar_operatorRR6sum_opI1TE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[8\]](api/languages/cpp_a   |
| q14complex_matrix11exponentialEv) | pi.html#_CPPv4NKR5cudaq10product_ |
| -                                 | opmiERK10product_opI9HandlerTyE), |
|  [cudaq::complex_matrix::identity |     [\[9\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/languages      | roduct_opmiERK15scalar_operator), |
| /cpp_api.html#_CPPv4N5cudaq14comp |     [\[10\]](api/languages/       |
| lex_matrix8identityEKNSt6size_tE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -                                 | duct_opmiERK6sum_opI9HandlerTyE), |
| [cudaq::complex_matrix::kronecker |     [\[11\]](api/languages/cpp_a  |
|     (C++                          | pi.html#_CPPv4NKR5cudaq10product_ |
|     function)](api/lang           | opmiERR10product_opI9HandlerTyE), |
| uages/cpp_api.html#_CPPv4I00EN5cu |     [\[12\]](api/language         |
| daq14complex_matrix9kroneckerE14c | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| omplex_matrix8Iterable8Iterable), | roduct_opmiERR15scalar_operator), |
|     [\[1\]](api/l                 |     [\[13\]](api/languages/       |
| anguages/cpp_api.html#_CPPv4N5cud | cpp_api.html#_CPPv4NKR5cudaq10pro |
| aq14complex_matrix9kroneckerERK14 | duct_opmiERR6sum_opI9HandlerTyE), |
| complex_matrixRK14complex_matrix) |     [\[                           |
| -   [cudaq::c                     | 14\]](api/languages/cpp_api.html# |
| omplex_matrix::minimal_eigenvalue | _CPPv4NKR5cudaq10product_opmiEv), |
|     (C++                          |     [\[15\]](api/languages/cpp_   |
|     function)](api/languages/     | api.html#_CPPv4NO5cudaq10product_ |
| cpp_api.html#_CPPv4NK5cudaq14comp | opmiERK10product_opI9HandlerTyE), |
| lex_matrix18minimal_eigenvalueEv) |     [\[16\]](api/languag          |
| -   [                             | es/cpp_api.html#_CPPv4NO5cudaq10p |
| cudaq::complex_matrix::operator() | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[17\]](api/languages        |
|     function)](api/languages/cpp  | /cpp_api.html#_CPPv4NO5cudaq10pro |
| _api.html#_CPPv4N5cudaq14complex_ | duct_opmiERK6sum_opI9HandlerTyE), |
| matrixclENSt6size_tENSt6size_tE), |     [\[18\]](api/languages/cpp_   |
|     [\[1\]](api/languages/cpp     | api.html#_CPPv4NO5cudaq10product_ |
| _api.html#_CPPv4NK5cudaq14complex | opmiERR10product_opI9HandlerTyE), |
| _matrixclENSt6size_tENSt6size_tE) |     [\[19\]](api/languag          |
| -   [                             | es/cpp_api.html#_CPPv4NO5cudaq10p |
| cudaq::complex_matrix::operator\* | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[20\]](api/languages        |
|     function)](api/langua         | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ges/cpp_api.html#_CPPv4N5cudaq14c | duct_opmiERR6sum_opI9HandlerTyE), |
| omplex_matrixmlEN14complex_matrix |     [                             |
| 10value_typeERK14complex_matrix), | \[21\]](api/languages/cpp_api.htm |
|     [\[1\]                        | l#_CPPv4NO5cudaq10product_opmiEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::product_op::operator/ |
| v4N5cudaq14complex_matrixmlERK14c |     (C++                          |
| omplex_matrixRK14complex_matrix), |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|  [\[2\]](api/languages/cpp_api.ht | roduct_opdvERK15scalar_operator), |
| ml#_CPPv4N5cudaq14complex_matrixm |     [\[1\]](api/language          |
| lERK14complex_matrixRKNSt6vectorI | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| N14complex_matrix10value_typeEEE) | roduct_opdvERR15scalar_operator), |
| -                                 |     [\[2\]](api/languag           |
| [cudaq::complex_matrix::operator+ | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opdvERK15scalar_operator), |
|     function                      |     [\[3\]](api/langua            |
| )](api/languages/cpp_api.html#_CP | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| Pv4N5cudaq14complex_matrixplERK14 | product_opdvERR15scalar_operator) |
| complex_matrixRK14complex_matrix) | -                                 |
| -                                 |    [cudaq::product_op::operator/= |
| [cudaq::complex_matrix::operator- |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function                      | ages/cpp_api.html#_CPPv4N5cudaq10 |
| )](api/languages/cpp_api.html#_CP | product_opdVERK15scalar_operator) |
| Pv4N5cudaq14complex_matrixmiERK14 | -   [cudaq::product_op::operator= |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -   [cu                           |     function)](api/la             |
| daq::complex_matrix::operator\[\] | nguages/cpp_api.html#_CPPv4I0_NSt |
|     (C++                          | 11enable_if_tIXaantNSt7is_sameI1T |
|                                   | 9HandlerTyE5valueENSt16is_constru |
|  function)](api/languages/cpp_api | ctibleI9HandlerTy1TE5valueEEbEEEN |
| .html#_CPPv4N5cudaq14complex_matr | 5cudaq10product_opaSER10product_o |
| ixixERKNSt6vectorINSt6size_tEEE), | pI9HandlerTyERK10product_opI1TE), |
|     [\[1\]](api/languages/cpp_api |     [\[1\]](api/languages/cpp     |
| .html#_CPPv4NK5cudaq14complex_mat | _api.html#_CPPv4N5cudaq10product_ |
| rixixERKNSt6vectorINSt6size_tEEE) | opaSERK10product_opI9HandlerTyE), |
| -   [cudaq::complex_matrix::power |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     function)]                    | _opaSERR10product_opI9HandlerTyE) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq14complex_matrix5powerEi) |    [cudaq::product_op::operator== |
| -                                 |     (C++                          |
|  [cudaq::complex_matrix::set_zero |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq10product |
|     function)](ap                 | _opeqERK10product_opI9HandlerTyE) |
| i/languages/cpp_api.html#_CPPv4N5 | -                                 |
| cudaq14complex_matrix8set_zeroEv) |  [cudaq::product_op::operator\[\] |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::to_string |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     function)](api/               | 5cudaq10product_opixENSt6size_tE) |
| languages/cpp_api.html#_CPPv4NK5c | -                                 |
| udaq14complex_matrix9to_stringEv) |    [cudaq::product_op::product_op |
| -   [                             |     (C++                          |
| cudaq::complex_matrix::value_type |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4I0_NSt11enable_ |
|     type)](api/                   | if_tIXaaNSt7is_sameI9HandlerTy14m |
| languages/cpp_api.html#_CPPv4N5cu | atrix_handlerE5valueEaantNSt7is_s |
| daq14complex_matrix10value_typeE) | ameI1T9HandlerTyE5valueENSt16is_c |
| -   [cudaq::contrib (C++          | onstructibleI9HandlerTy1TE5valueE |
|     type)](api/languages/cpp      | EbEEEN5cudaq10product_op10product |
| _api.html#_CPPv4N5cudaq7contribE) | _opERK10product_opI1TERKN14matrix |
| -   [cudaq::contrib::draw (C++    | _handler20commutation_behaviorE), |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP |  [\[1\]](api/languages/cpp_api.ht |
| v4I0DpEN5cudaq7contrib4drawENSt6s | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| tringERR13QuantumKernelDpRR4Args) | tNSt7is_sameI1T9HandlerTyE5valueE |
| -                                 | NSt16is_constructibleI9HandlerTy1 |
| [cudaq::contrib::get_unitary_cmat | TE5valueEEbEEEN5cudaq10product_op |
|     (C++                          | 10product_opERK10product_opI1TE), |
|     function)](api/languages/cp   |                                   |
| p_api.html#_CPPv4I0DpEN5cudaq7con |   [\[2\]](api/languages/cpp_api.h |
| trib16get_unitary_cmatE14complex_ | tml#_CPPv4N5cudaq10product_op10pr |
| matrixRR13QuantumKernelDpRR4Args) | oduct_opENSt6size_tENSt6size_tE), |
| -   [cudaq::CusvState (C++        |     [\[3\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq10product |
|    class)](api/languages/cpp_api. | _op10product_opENSt7complexIdEE), |
| html#_CPPv4I0EN5cudaq9CusvStateE) |     [\[4\]](api/l                 |
| -   [cudaq::depolarization1 (C++  | anguages/cpp_api.html#_CPPv4N5cud |
|     c                             | aq10product_op10product_opERK10pr |
| lass)](api/languages/cpp_api.html | oduct_opI9HandlerTyENSt6size_tE), |
| #_CPPv4N5cudaq15depolarization1E) |     [\[5\]](api/l                 |
| -   [cudaq::depolarization2 (C++  | anguages/cpp_api.html#_CPPv4N5cud |
|     c                             | aq10product_op10product_opERR10pr |
| lass)](api/languages/cpp_api.html | oduct_opI9HandlerTyENSt6size_tE), |
| #_CPPv4N5cudaq15depolarization2E) |     [\[6\]](api/languages         |
| -   [cudaq:                       | /cpp_api.html#_CPPv4N5cudaq10prod |
| :depolarization2::depolarization2 | uct_op10product_opERR9HandlerTy), |
|     (C++                          |     [\[7\]](ap                    |
|     function)](api/languages/cp   | i/languages/cpp_api.html#_CPPv4N5 |
| p_api.html#_CPPv4N5cudaq15depolar | cudaq10product_op10product_opEd), |
| ization215depolarization2EK4real) |     [\[8\]](a                     |
| -   [cudaq                        | pi/languages/cpp_api.html#_CPPv4N |
| ::depolarization2::num_parameters | 5cudaq10product_op10product_opEv) |
|     (C++                          | -   [cuda                         |
|     member)](api/langu            | q::product_op::to_diagonal_matrix |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| depolarization214num_parametersE) |     function)](api/               |
| -   [cu                           | languages/cpp_api.html#_CPPv4NK5c |
| daq::depolarization2::num_targets | udaq10product_op18to_diagonal_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     member)](api/la               | ENSt7int64_tEEERKNSt13unordered_m |
| nguages/cpp_api.html#_CPPv4N5cuda | apINSt6stringENSt7complexIdEEEEb) |
| q15depolarization211num_targetsE) | -   [cudaq::product_op::to_matrix |
| -                                 |     (C++                          |
|    [cudaq::depolarization_channel |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     class)](                      | _CPPv4NK5cudaq10product_op9to_mat |
| api/languages/cpp_api.html#_CPPv4 | rixENSt13unordered_mapINSt6size_t |
| N5cudaq22depolarization_channelE) | ENSt7int64_tEEERKNSt13unordered_m |
| -   [cudaq::depol                 | apINSt6stringENSt7complexIdEEEEb) |
| arization_channel::num_parameters | -   [cu                           |
|     (C++                          | daq::product_op::to_sparse_matrix |
|     member)](api/languages/cp     |     (C++                          |
| p_api.html#_CPPv4N5cudaq22depolar |     function)](ap                 |
| ization_channel14num_parametersE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::de                    | 5cudaq10product_op16to_sparse_mat |
| polarization_channel::num_targets | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     member)](api/languages        | apINSt6stringENSt7complexIdEEEEb) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -   [cudaq::product_op::to_string |
| larization_channel11num_targetsE) |     (C++                          |
| -   [cudaq::details (C++          |     function)](                   |
|     type)](api/languages/cpp      | api/languages/cpp_api.html#_CPPv4 |
| _api.html#_CPPv4N5cudaq7detailsE) | NK5cudaq10product_op9to_stringEv) |
| -   [cudaq::details::future (C++  | -                                 |
|                                   |  [cudaq::product_op::\~product_op |
|  class)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq7details6futureE) |     fu                            |
| -                                 | nction)](api/languages/cpp_api.ht |
|   [cudaq::details::future::future | ml#_CPPv4N5cudaq10product_opD0Ev) |
|     (C++                          | -   [cudaq::QPU (C++              |
|     functio                       |     class)](api/languages         |
| n)](api/languages/cpp_api.html#_C | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| PPv4N5cudaq7details6future6future | -   [cudaq::QPU::beginExecution   |
| ERNSt6vectorI3JobEERNSt6stringERN |     (C++                          |
| St3mapINSt6stringENSt6stringEEE), |     function                      |
|     [\[1\]](api/lang              | )](api/languages/cpp_api.html#_CP |
| uages/cpp_api.html#_CPPv4N5cudaq7 | Pv4N5cudaq3QPU14beginExecutionEv) |
| details6future6futureERR6future), | -   [cuda                         |
|     [\[2\]]                       | q::QPU::configureExecutionContext |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq7details6future6futureEv) |     funct                         |
| -   [cu                           | ion)](api/languages/cpp_api.html# |
| daq::details::kernel_builder_base | _CPPv4NK5cudaq3QPU25configureExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     class)](api/l                 | -   [cudaq::QPU::endExecution     |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq7details19kernel_builder_baseE) |     functi                        |
| -   [cudaq::details::             | on)](api/languages/cpp_api.html#_ |
| kernel_builder_base::operator\<\< | CPPv4N5cudaq3QPU12endExecutionEv) |
|     (C++                          | -   [cudaq::QPU::enqueue (C++     |
|     function)](api/langua         |     function)](ap                 |
| ges/cpp_api.html#_CPPv4N5cudaq7de | i/languages/cpp_api.html#_CPPv4N5 |
| tails19kernel_builder_baselsERNSt | cudaq3QPU7enqueueER11QuantumTask) |
| 7ostreamERK19kernel_builder_base) | -   [cud                          |
| -   [                             | aq::QPU::finalizeExecutionContext |
| cudaq::details::KernelBuilderType |     (C++                          |
|     (C++                          |     func                          |
|     class)](api                   | tion)](api/languages/cpp_api.html |
| /languages/cpp_api.html#_CPPv4N5c | #_CPPv4NK5cudaq3QPU24finalizeExec |
| udaq7details17KernelBuilderTypeE) | utionContextER16ExecutionContext) |
| -   [cudaq::d                     | -   [cudaq::QPU::getConnectivity  |
| etails::KernelBuilderType::create |     (C++                          |
|     (C++                          |     function)                     |
|     function)                     | ](api/languages/cpp_api.html#_CPP |
| ](api/languages/cpp_api.html#_CPP | v4N5cudaq3QPU15getConnectivityEv) |
| v4N5cudaq7details17KernelBuilderT | -                                 |
| ype6createEPN4mlir11MLIRContextE) | [cudaq::QPU::getExecutionThreadId |
| -   [cudaq::details::Ker          |     (C++                          |
| nelBuilderType::KernelBuilderType |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4NK5c |
|     function)](api/lang           | udaq3QPU20getExecutionThreadIdEv) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -   [cudaq::QPU::getNumQubits     |
| details17KernelBuilderType17Kerne |     (C++                          |
| lBuilderTypeERRNSt8functionIFN4ml |     functi                        |
| ir4TypeEPN4mlir11MLIRContextEEEE) | on)](api/languages/cpp_api.html#_ |
| -   [cudaq::diag_matrix_callback  | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     (C++                          | -   [                             |
|     class)                        | cudaq::QPU::getRemoteCapabilities |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq20diag_matrix_callbackE) |     function)](api/l              |
| -   [cudaq::dyn (C++              | anguages/cpp_api.html#_CPPv4NK5cu |
|     member)](api/languages        | daq3QPU21getRemoteCapabilitiesEv) |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | -   [cudaq::QPU::isEmulated (C++  |
| -   [cudaq::ExecutionContext (C++ |     func                          |
|     cl                            | tion)](api/languages/cpp_api.html |
| ass)](api/languages/cpp_api.html# | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| _CPPv4N5cudaq16ExecutionContextE) | -   [cudaq::QPU::isSimulator (C++ |
| -   [cudaq                        |     funct                         |
| ::ExecutionContext::amplitudeMaps | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     member)](api/langu            | -   [cudaq::QPU::launchKernel     |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13amplitudeMapsE) |     function)](api/               |
| -   [c                            | languages/cpp_api.html#_CPPv4N5cu |
| udaq::ExecutionContext::asyncExec | daq3QPU12launchKernelERKNSt6strin |
|     (C++                          | gE15KernelThunkTypePvNSt8uint64_t |
|     member)](api/                 | ENSt8uint64_tERKNSt6vectorIPvEE), |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq16ExecutionContext9asyncExecE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cud                          | ml#_CPPv4N5cudaq3QPU12launchKerne |
| aq::ExecutionContext::asyncResult | lERKNSt6stringERKNSt6vectorIPvEE) |
|     (C++                          | -   [cudaq::QPU::onRandomSeedSet  |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/lang           |
| 16ExecutionContext11asyncResultE) | uages/cpp_api.html#_CPPv4N5cudaq3 |
| -   [cudaq:                       | QPU15onRandomSeedSetENSt6size_tE) |
| :ExecutionContext::batchIteration | -   [cudaq::QPU::QPU (C++         |
|     (C++                          |     functio                       |
|     member)](api/langua           | n)](api/languages/cpp_api.html#_C |
| ges/cpp_api.html#_CPPv4N5cudaq16E | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| xecutionContext14batchIterationE) |                                   |
| -   [cudaq::E                     |  [\[1\]](api/languages/cpp_api.ht |
| xecutionContext::canHandleObserve | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     (C++                          |     [\[2\]](api/languages/cpp_    |
|     member)](api/language         | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::QPU::setId (C++       |
| cutionContext16canHandleObserveE) |     function                      |
| -   [cudaq::E                     | )](api/languages/cpp_api.html#_CP |
| xecutionContext::ExecutionContext | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::setShots (C++    |
|     func                          |     f                             |
| tion)](api/languages/cpp_api.html | unction)](api/languages/cpp_api.h |
| #_CPPv4N5cudaq16ExecutionContext1 | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
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
| ents::forward_difference::compute | -   [cudaq::quantum_platfor       |
|     (C++                          | m::supports_explicit_measurements |
|     function)](                   |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/l              |
| N5cudaq9gradients18forward_differ | anguages/cpp_api.html#_CPPv4NK5cu |
| ence7computeERKNSt6vectorIdEERKNS | daq16quantum_platform30supports_e |
| t8functionIFdNSt6vectorIdEEEEEd), | xplicit_measurementsENSt6size_tE) |
|                                   | -   [cudaq::quantum_pla           |
|   [\[1\]](api/languages/cpp_api.h | tform::supports_task_distribution |
| tml#_CPPv4N5cudaq9gradients18forw |     (C++                          |
| ard_difference7computeERKNSt6vect |     fu                            |
| orIdEERNSt6vectorIdEERK7spin_opd) | nction)](api/languages/cpp_api.ht |
| -   [cudaq::gradie                | ml#_CPPv4NK5cudaq16quantum_platfo |
| nts::forward_difference::gradient | rm26supports_task_distributionEv) |
|     (C++                          | -   [cudaq::quantum               |
|     functio                       | _platform::with_execution_context |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4I00EN5cudaq9gradients18forwar |     function)                     |
| d_difference8gradientER7KernelT), | ](api/languages/cpp_api.html#_CPP |
|     [\[1\]](api/langua            | v4I0DpEN5cudaq16quantum_platform2 |
| ges/cpp_api.html#_CPPv4I00EN5cuda | 2with_execution_contextEDaR16Exec |
| q9gradients18forward_difference8g | utionContextRR8CallableDpRR4Args) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::QuantumTask (C++      |
|     [\[2\]](api/languages/cpp_    |     type)](api/languages/cpp_api. |
| api.html#_CPPv4I00EN5cudaq9gradie | html#_CPPv4N5cudaq11QuantumTaskE) |
| nts18forward_difference8gradientE | -   [cudaq::qubit (C++            |
| RR13QuantumKernelRR10ArgsMapper), |     type)](api/languages/c        |
|     [\[3\]](api/languages/cpp     | pp_api.html#_CPPv4N5cudaq5qubitE) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::QubitConnectivity     |
| 18forward_difference8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     ty                            |
|     [\[4\]](api/languages/cp      | pe)](api/languages/cpp_api.html#_ |
| p_api.html#_CPPv4N5cudaq9gradient | CPPv4N5cudaq17QubitConnectivityE) |
| s18forward_difference8gradientEv) | -   [cudaq::QubitEdge (C++        |
| -   [                             |     type)](api/languages/cpp_a    |
| cudaq::gradients::parameter_shift | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|     (C++                          | -   [cudaq::qudit (C++            |
|     class)](api                   |     clas                          |
| /languages/cpp_api.html#_CPPv4N5c | s)](api/languages/cpp_api.html#_C |
| udaq9gradients15parameter_shiftE) | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| -   [cudaq::                      | -   [cudaq::qudit::qudit (C++     |
| gradients::parameter_shift::clone |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function)](api/langua         | html#_CPPv4N5cudaq5qudit5quditEv) |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | -   [cudaq::qvector (C++          |
| adients15parameter_shift5cloneEv) |     class)                        |
| -   [cudaq::gr                    | ](api/languages/cpp_api.html#_CPP |
| adients::parameter_shift::compute | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     (C++                          | -   [cudaq::qvector::back (C++    |
|     function                      |     function)](a                  |
| )](api/languages/cpp_api.html#_CP | pi/languages/cpp_api.html#_CPPv4N |
| Pv4N5cudaq9gradients15parameter_s | 5cudaq7qvector4backENSt6size_tE), |
| hift7computeERKNSt6vectorIdEERKNS |                                   |
| t8functionIFdNSt6vectorIdEEEEEd), |   [\[1\]](api/languages/cpp_api.h |
|     [\[1\]](api/languages/cpp_ap  | tml#_CPPv4N5cudaq7qvector4backEv) |
| i.html#_CPPv4N5cudaq9gradients15p | -   [cudaq::qvector::begin (C++   |
| arameter_shift7computeERKNSt6vect |     fu                            |
| orIdEERNSt6vectorIdEERK7spin_opd) | nction)](api/languages/cpp_api.ht |
| -   [cudaq::gra                   | ml#_CPPv4N5cudaq7qvector5beginEv) |
| dients::parameter_shift::gradient | -   [cudaq::qvector::clear (C++   |
|     (C++                          |     fu                            |
|     func                          | nction)](api/languages/cpp_api.ht |
| tion)](api/languages/cpp_api.html | ml#_CPPv4N5cudaq7qvector5clearEv) |
| #_CPPv4I00EN5cudaq9gradients15par | -   [cudaq::qvector::end (C++     |
| ameter_shift8gradientER7KernelT), |                                   |
|     [\[1\]](api/lan               | function)](api/languages/cpp_api. |
| guages/cpp_api.html#_CPPv4I00EN5c | html#_CPPv4N5cudaq7qvector3endEv) |
| udaq9gradients15parameter_shift8g | -   [cudaq::qvector::front (C++   |
| radientER7KernelTRR10ArgsMapper), |     function)](ap                 |
|     [\[2\]](api/languages/c       | i/languages/cpp_api.html#_CPPv4N5 |
| pp_api.html#_CPPv4I00EN5cudaq9gra | cudaq7qvector5frontENSt6size_tE), |
| dients15parameter_shift8gradientE |                                   |
| RR13QuantumKernelRR10ArgsMapper), |  [\[1\]](api/languages/cpp_api.ht |
|     [\[3\]](api/languages/        | ml#_CPPv4N5cudaq7qvector5frontEv) |
| cpp_api.html#_CPPv4N5cudaq9gradie | -   [cudaq::qvector::operator=    |
| nts15parameter_shift8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     functio                       |
|     [\[4\]](api/languages         | n)](api/languages/cpp_api.html#_C |
| /cpp_api.html#_CPPv4N5cudaq9gradi | PPv4N5cudaq7qvectoraSERK7qvector) |
| ents15parameter_shift8gradientEv) | -   [cudaq::qvector::operator\[\] |
| -   [cudaq::kernel_builder (C++   |     (C++                          |
|     clas                          |     function)                     |
| s)](api/languages/cpp_api.html#_C | ](api/languages/cpp_api.html#_CPP |
| PPv4IDpEN5cudaq14kernel_builderE) | v4N5cudaq7qvectorixEKNSt6size_tE) |
| -   [c                            | -   [cudaq::qvector::qvector (C++ |
| udaq::kernel_builder::constantVal |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     function)](api/la             | daq7qvector7qvectorENSt6size_tE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[1\]](a                     |
| q14kernel_builder11constantValEd) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cu                           | 5cudaq7qvector7qvectorERK5state), |
| daq::kernel_builder::getArguments |     [\[2\]](api                   |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/lan            | udaq7qvector7qvectorERK7qvector), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[3\]](api/languages/cpp     |
| 14kernel_builder12getArgumentsEv) | _api.html#_CPPv4N5cudaq7qvector7q |
| -   [cu                           | vectorERKNSt6vectorI7complexEEb), |
| daq::kernel_builder::getNumParams |     [\[4\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/lan            | cudaq7qvector7qvectorERR7qvector) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qvector::size (C++    |
| 14kernel_builder12getNumParamsEv) |     fu                            |
| -   [c                            | nction)](api/languages/cpp_api.ht |
| udaq::kernel_builder::isArgStdVec | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     (C++                          | -   [cudaq::qvector::slice (C++   |
|     function)](api/languages/cp   |     function)](api/language       |
| p_api.html#_CPPv4N5cudaq14kernel_ | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| builder11isArgStdVecENSt6size_tE) | tor5sliceENSt6size_tENSt6size_tE) |
| -   [cuda                         | -   [cudaq::qvector::value_type   |
| q::kernel_builder::kernel_builder |     (C++                          |
|     (C++                          |     typ                           |
|     function)](api/languages/cpp_ | e)](api/languages/cpp_api.html#_C |
| api.html#_CPPv4N5cudaq14kernel_bu | PPv4N5cudaq7qvector10value_typeE) |
| ilder14kernel_builderERNSt6vector | -   [cudaq::qview (C++            |
| IN7details17KernelBuilderTypeEEE) |     clas                          |
| -   [cudaq::kernel_builder::name  | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     function)                     | -   [cudaq::qview::back (C++      |
| ](api/languages/cpp_api.html#_CPP |     function)                     |
| v4N5cudaq14kernel_builder4nameEv) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq5qview4backENSt6size_tE) |
|    [cudaq::kernel_builder::qalloc | -   [cudaq::qview::begin (C++     |
|     (C++                          |                                   |
|     function)](api/language       | function)](api/languages/cpp_api. |
| s/cpp_api.html#_CPPv4N5cudaq14ker | html#_CPPv4N5cudaq5qview5beginEv) |
| nel_builder6qallocE10QuakeValue), | -   [cudaq::qview::end (C++       |
|     [\[1\]](api/language          |                                   |
| s/cpp_api.html#_CPPv4N5cudaq14ker |   function)](api/languages/cpp_ap |
| nel_builder6qallocEKNSt6size_tE), | i.html#_CPPv4N5cudaq5qview3endEv) |
|     [\[2                          | -   [cudaq::qview::front (C++     |
| \]](api/languages/cpp_api.html#_C |     function)](                   |
| PPv4N5cudaq14kernel_builder6qallo | api/languages/cpp_api.html#_CPPv4 |
| cERNSt6vectorINSt7complexIdEEEE), | N5cudaq5qview5frontENSt6size_tE), |
|     [\[3\]](                      |                                   |
| api/languages/cpp_api.html#_CPPv4 |    [\[1\]](api/languages/cpp_api. |
| N5cudaq14kernel_builder6qallocEv) | html#_CPPv4N5cudaq5qview5frontEv) |
| -   [cudaq::kernel_builder::swap  | -   [cudaq::qview::operator\[\]   |
|     (C++                          |     (C++                          |
|     function)](api/language       |     functio                       |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | n)](api/languages/cpp_api.html#_C |
| 4kernel_builder4swapEvRK10QuakeVa | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| lueRK10QuakeValueRK10QuakeValue), | -   [cudaq::qview::qview (C++     |
|                                   |     functio                       |
| [\[1\]](api/languages/cpp_api.htm | n)](api/languages/cpp_api.html#_C |
| l#_CPPv4I00EN5cudaq14kernel_build | PPv4I0EN5cudaq5qview5qviewERR1R), |
| er4swapEvRKNSt6vectorI10QuakeValu |     [\[1                          |
| eEERK10QuakeValueRK10QuakeValue), | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq5qview5qviewERK5qview) |
| [\[2\]](api/languages/cpp_api.htm | -   [cudaq::qview::size (C++      |
| l#_CPPv4N5cudaq14kernel_builder4s |                                   |
| wapERK10QuakeValueRK10QuakeValue) | function)](api/languages/cpp_api. |
| -   [cudaq::KernelExecutionTask   | html#_CPPv4NK5cudaq5qview4sizeEv) |
|     (C++                          | -   [cudaq::qview::slice (C++     |
|     type                          |     function)](api/langua         |
| )](api/languages/cpp_api.html#_CP | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| Pv4N5cudaq19KernelExecutionTaskE) | iew5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::KernelThunkResultType | -   [cudaq::qview::value_type     |
|     (C++                          |     (C++                          |
|     struct)]                      |     t                             |
| (api/languages/cpp_api.html#_CPPv | ype)](api/languages/cpp_api.html# |
| 4N5cudaq21KernelThunkResultTypeE) | _CPPv4N5cudaq5qview10value_typeE) |
| -   [cudaq::KernelThunkType (C++  | -   [cudaq::range (C++            |
|                                   |     fun                           |
| type)](api/languages/cpp_api.html | ction)](api/languages/cpp_api.htm |
| #_CPPv4N5cudaq15KernelThunkTypeE) | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| -   [cudaq::kraus_channel (C++    | orI11ElementTypeEE11ElementType), |
|                                   |     [\[1\]](api/languages/cpp_    |
|  class)](api/languages/cpp_api.ht | api.html#_CPPv4I0EN5cudaq5rangeEN |
| ml#_CPPv4N5cudaq13kraus_channelE) | St6vectorI11ElementTypeEE11Elemen |
| -   [cudaq::kraus_channel::empty  | tType11ElementType11ElementType), |
|     (C++                          |     [                             |
|     function)]                    | \[2\]](api/languages/cpp_api.html |
| (api/languages/cpp_api.html#_CPPv | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| 4NK5cudaq13kraus_channel5emptyEv) | -   [cudaq::real (C++             |
| -   [cudaq::kraus_c               |     type)](api/languages/         |
| hannel::generateUnitaryParameters | cpp_api.html#_CPPv4N5cudaq4realE) |
|     (C++                          | -   [cudaq::registry (C++         |
|                                   |     type)](api/languages/cpp_     |
|    function)](api/languages/cpp_a | api.html#_CPPv4N5cudaq8registryE) |
| pi.html#_CPPv4N5cudaq13kraus_chan | -                                 |
| nel25generateUnitaryParametersEv) |  [cudaq::registry::RegisteredType |
| -                                 |     (C++                          |
|    [cudaq::kraus_channel::get_ops |     class)](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](a                  | 5cudaq8registry14RegisteredTypeE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::RemoteCapabilities    |
| K5cudaq13kraus_channel7get_opsEv) |     (C++                          |
| -   [cudaq::                      |     struc                         |
| kraus_channel::is_unitary_mixture | t)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq18RemoteCapabilitiesE) |
|     function)](api/languages      | -   [cudaq::Remo                  |
| /cpp_api.html#_CPPv4NK5cudaq13kra | teCapabilities::isRemoteSimulator |
| us_channel18is_unitary_mixtureEv) |     (C++                          |
| -   [cu                           |     member)](api/languages/c      |
| daq::kraus_channel::kraus_channel | pp_api.html#_CPPv4N5cudaq18Remote |
|     (C++                          | Capabilities17isRemoteSimulatorE) |
|     function)](api/lang           | -   [cudaq::Remot                 |
| uages/cpp_api.html#_CPPv4IDpEN5cu | eCapabilities::RemoteCapabilities |
| daq13kraus_channel13kraus_channel |     (C++                          |
| EDpRRNSt16initializer_listI1TEE), |     function)](api/languages/cpp  |
|                                   | _api.html#_CPPv4N5cudaq18RemoteCa |
|  [\[1\]](api/languages/cpp_api.ht | pabilities18RemoteCapabilitiesEb) |
| ml#_CPPv4N5cudaq13kraus_channel13 | -   [cudaq:                       |
| kraus_channelERK13kraus_channel), | :RemoteCapabilities::stateOverlap |
|     [\[2\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     member)](api/langua           |
| v4N5cudaq13kraus_channel13kraus_c | ges/cpp_api.html#_CPPv4N5cudaq18R |
| hannelERKNSt6vectorI8kraus_opEE), | emoteCapabilities12stateOverlapE) |
|     [\[3\]                        | -                                 |
| ](api/languages/cpp_api.html#_CPP |   [cudaq::RemoteCapabilities::vqe |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERRNSt6vectorI8kraus_opEE), |     member)](                     |
|     [\[4\]](api/lan               | api/languages/cpp_api.html#_CPPv4 |
| guages/cpp_api.html#_CPPv4N5cudaq | N5cudaq18RemoteCapabilities3vqeE) |
| 13kraus_channel13kraus_channelEv) | -   [cudaq::RemoteSimulationState |
| -                                 |     (C++                          |
| [cudaq::kraus_channel::noise_type |     class)]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     member)](api                  | 4N5cudaq21RemoteSimulationStateE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::Resources (C++        |
| udaq13kraus_channel10noise_typeE) |     class)](api/languages/cpp_a   |
| -                                 | pi.html#_CPPv4N5cudaq9ResourcesE) |
|   [cudaq::kraus_channel::op_names | -   [cudaq::run (C++              |
|     (C++                          |     function)]                    |
|     member)](                     | (api/languages/cpp_api.html#_CPPv |
| api/languages/cpp_api.html#_CPPv4 | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| N5cudaq13kraus_channel8op_namesE) | 5invoke_result_tINSt7decay_tI13Qu |
| -                                 | antumKernelEEDpNSt7decay_tI4ARGSE |
|  [cudaq::kraus_channel::operator= | EEEEENSt6size_tERN5cudaq11noise_m |
|     (C++                          | odelERR13QuantumKernelDpRR4ARGS), |
|     function)](api/langua         |     [\[1\]](api/langu             |
| ges/cpp_api.html#_CPPv4N5cudaq13k | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| raus_channelaSERK13kraus_channel) | daq3runENSt6vectorINSt15invoke_re |
| -   [c                            | sult_tINSt7decay_tI13QuantumKerne |
| udaq::kraus_channel::operator\[\] | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4ARGS) |
|     function)](api/l              | -   [cudaq::run_async (C++        |
| anguages/cpp_api.html#_CPPv4N5cud |     functio                       |
| aq13kraus_channelixEKNSt6size_tE) | n)](api/languages/cpp_api.html#_C |
| -                                 | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| [cudaq::kraus_channel::parameters | tureINSt6vectorINSt15invoke_resul |
|     (C++                          | t_tINSt7decay_tI13QuantumKernelEE |
|     member)](api                  | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| /languages/cpp_api.html#_CPPv4N5c | ze_tENSt6size_tERN5cudaq11noise_m |
| udaq13kraus_channel10parametersE) | odelERR13QuantumKernelDpRR4ARGS), |
| -   [cudaq::krau                  |     [\[1\]](api/la                |
| s_channel::populateDefaultOpNames | nguages/cpp_api.html#_CPPv4I0DpEN |
|     (C++                          | 5cudaq9run_asyncENSt6futureINSt6v |
|     function)](api/languages/cp   | ectorINSt15invoke_result_tINSt7de |
| p_api.html#_CPPv4N5cudaq13kraus_c | cay_tI13QuantumKernelEEDpNSt7deca |
| hannel22populateDefaultOpNamesEv) | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| -   [cu                           | ize_tERR13QuantumKernelDpRR4ARGS) |
| daq::kraus_channel::probabilities | -   [cudaq::RuntimeTarget (C++    |
|     (C++                          |                                   |
|     member)](api/la               | struct)](api/languages/cpp_api.ht |
| nguages/cpp_api.html#_CPPv4N5cuda | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| q13kraus_channel13probabilitiesE) | -   [cudaq::sample (C++           |
| -                                 |     function)](api/languages/c    |
|  [cudaq::kraus_channel::push_back | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     (C++                          | mpleE13sample_resultRK14sample_op |
|     function)](api                | tionsRR13QuantumKernelDpRR4Args), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[1\                         |
| udaq13kraus_channel9push_backE8kr | ]](api/languages/cpp_api.html#_CP |
| aus_opNSt8optionalINSt6stringEEE) | Pv4I0DpEN5cudaq6sampleE13sample_r |
| -   [cudaq::kraus_channel::size   | esultRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\                            |
|     function)                     | [2\]](api/languages/cpp_api.html# |
| ](api/languages/cpp_api.html#_CPP | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| v4NK5cudaq13kraus_channel4sizeEv) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [                             | -   [cudaq::sample_options (C++   |
| cudaq::kraus_channel::unitary_ops |     s                             |
|     (C++                          | truct)](api/languages/cpp_api.htm |
|     member)](api/                 | l#_CPPv4N5cudaq14sample_optionsE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::sample_result (C++    |
| daq13kraus_channel11unitary_opsE) |                                   |
| -   [cudaq::kraus_op (C++         |  class)](api/languages/cpp_api.ht |
|     struct)](api/languages/cpp_   | ml#_CPPv4N5cudaq13sample_resultE) |
| api.html#_CPPv4N5cudaq8kraus_opE) | -   [cudaq::sample_result::append |
| -   [cudaq::kraus_op::adjoint     |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     functi                        | api.html#_CPPv4N5cudaq13sample_re |
| on)](api/languages/cpp_api.html#_ | sult6appendERK15ExecutionResultb) |
| CPPv4NK5cudaq8kraus_op7adjointEv) | -   [cudaq::sample_result::begin  |
| -   [cudaq::kraus_op::data (C++   |     (C++                          |
|                                   |     function)]                    |
|  member)](api/languages/cpp_api.h | (api/languages/cpp_api.html#_CPPv |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | 4N5cudaq13sample_result5beginEv), |
| -   [cudaq::kraus_op::kraus_op    |     [\[1\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     func                          | 4NK5cudaq13sample_result5beginEv) |
| tion)](api/languages/cpp_api.html | -   [cudaq::sample_result::cbegin |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |     (C++                          |
| opERRNSt16initializer_listI1TEE), |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|  [\[1\]](api/languages/cpp_api.ht | NK5cudaq13sample_result6cbeginEv) |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | -   [cudaq::sample_result::cend   |
| pENSt6vectorIN5cudaq7complexEEE), |     (C++                          |
|     [\[2\]](api/l                 |     function)                     |
| anguages/cpp_api.html#_CPPv4N5cud | ](api/languages/cpp_api.html#_CPP |
| aq8kraus_op8kraus_opERK8kraus_op) | v4NK5cudaq13sample_result4cendEv) |
| -   [cudaq::kraus_op::nCols (C++  | -   [cudaq::sample_result::clear  |
|                                   |     (C++                          |
| member)](api/languages/cpp_api.ht |     function)                     |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus_op::nRows (C++  | v4N5cudaq13sample_result5clearEv) |
|                                   | -   [cudaq::sample_result::count  |
| member)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) |     function)](                   |
| -   [cudaq::kraus_op::operator=   | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq13sample_result5countENSt |
|     function)                     | 11string_viewEKNSt11string_viewE) |
| ](api/languages/cpp_api.html#_CPP | -   [                             |
| v4N5cudaq8kraus_opaSERK8kraus_op) | cudaq::sample_result::deserialize |
| -   [cudaq::kraus_op::precision   |     (C++                          |
|     (C++                          |     functio                       |
|     memb                          | n)](api/languages/cpp_api.html#_C |
| er)](api/languages/cpp_api.html#_ | PPv4N5cudaq13sample_result11deser |
| CPPv4N5cudaq8kraus_op9precisionE) | ializeERNSt6vectorINSt6size_tEEE) |
| -   [cudaq::matrix_callback (C++  | -   [cudaq::sample_result::dump   |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     function)](api/languag        |
| #_CPPv4N5cudaq15matrix_callbackE) | es/cpp_api.html#_CPPv4NK5cudaq13s |
| -   [cudaq::matrix_handler (C++   | ample_result4dumpERNSt7ostreamE), |
|                                   |     [\[1\]                        |
| class)](api/languages/cpp_api.htm | ](api/languages/cpp_api.html#_CPP |
| l#_CPPv4N5cudaq14matrix_handlerE) | v4NK5cudaq13sample_result4dumpEv) |
| -   [cudaq::mat                   | -   [cudaq::sample_result::end    |
| rix_handler::commutation_behavior |     (C++                          |
|     (C++                          |     function                      |
|     struct)](api/languages/       | )](api/languages/cpp_api.html#_CP |
| cpp_api.html#_CPPv4N5cudaq14matri | Pv4N5cudaq13sample_result3endEv), |
| x_handler20commutation_behaviorE) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|    [cudaq::matrix_handler::define | Pv4NK5cudaq13sample_result3endEv) |
|     (C++                          | -   [                             |
|     function)](a                  | cudaq::sample_result::expectation |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14matrix_handler6defineENSt |     f                             |
| 6stringENSt6vectorINSt7int64_tEEE | unction)](api/languages/cpp_api.h |
| RR15matrix_callbackRKNSt13unorder | tml#_CPPv4NK5cudaq13sample_result |
| ed_mapINSt6stringENSt6stringEEE), | 11expectationEKNSt11string_viewE) |
|                                   | -   [c                            |
| [\[1\]](api/languages/cpp_api.htm | udaq::sample_result::get_marginal |
| l#_CPPv4N5cudaq14matrix_handler6d |     (C++                          |
| efineENSt6stringENSt6vectorINSt7i |     function)](api/languages/cpp_ |
| nt64_tEEERR15matrix_callbackRR20d | api.html#_CPPv4NK5cudaq13sample_r |
| iag_matrix_callbackRKNSt13unorder | esult12get_marginalERKNSt6vectorI |
| ed_mapINSt6stringENSt6stringEEE), | NSt6size_tEEEKNSt11string_viewE), |
|     [\[2\]](                      |     [\[1\]](api/languages/cpp_    |
| api/languages/cpp_api.html#_CPPv4 | api.html#_CPPv4NK5cudaq13sample_r |
| N5cudaq14matrix_handler6defineENS | esult12get_marginalERRKNSt6vector |
| t6stringENSt6vectorINSt7int64_tEE | INSt6size_tEEEKNSt11string_viewE) |
| ERR15matrix_callbackRRNSt13unorde | -   [cuda                         |
| red_mapINSt6stringENSt6stringEEE) | q::sample_result::get_total_shots |
| -                                 |     (C++                          |
|   [cudaq::matrix_handler::degrees |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq13 |
|     function)](ap                 | sample_result15get_total_shotsEv) |
| i/languages/cpp_api.html#_CPPv4NK | -   [cuda                         |
| 5cudaq14matrix_handler7degreesEv) | q::sample_result::has_even_parity |
| -                                 |     (C++                          |
|  [cudaq::matrix_handler::displace |     fun                           |
|     (C++                          | ction)](api/languages/cpp_api.htm |
|     function)](api/language       | l#_CPPv4N5cudaq13sample_result15h |
| s/cpp_api.html#_CPPv4N5cudaq14mat | as_even_parityENSt11string_viewE) |
| rix_handler8displaceENSt6size_tE) | -   [cuda                         |
| -   [cudaq::matrix                | q::sample_result::has_expectation |
| _handler::get_expected_dimensions |     (C++                          |
|     (C++                          |     funct                         |
|                                   | ion)](api/languages/cpp_api.html# |
|    function)](api/languages/cpp_a | _CPPv4NK5cudaq13sample_result15ha |
| pi.html#_CPPv4NK5cudaq14matrix_ha | s_expectationEKNSt11string_viewE) |
| ndler23get_expected_dimensionsEv) | -   [cu                           |
| -   [cudaq::matrix_ha             | daq::sample_result::most_probable |
| ndler::get_parameter_descriptions |     (C++                          |
|     (C++                          |     fun                           |
|                                   | ction)](api/languages/cpp_api.htm |
| function)](api/languages/cpp_api. | l#_CPPv4NK5cudaq13sample_result13 |
| html#_CPPv4NK5cudaq14matrix_handl | most_probableEKNSt11string_viewE) |
| er26get_parameter_descriptionsEv) | -                                 |
| -   [c                            | [cudaq::sample_result::operator+= |
| udaq::matrix_handler::instantiate |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](a                  | ges/cpp_api.html#_CPPv4N5cudaq13s |
| pi/languages/cpp_api.html#_CPPv4N | ample_resultpLERK13sample_result) |
| 5cudaq14matrix_handler11instantia | -                                 |
| teENSt6stringERKNSt6vectorINSt6si |  [cudaq::sample_result::operator= |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[1\]](                      |     function)](api/langua         |
| api/languages/cpp_api.html#_CPPv4 | ges/cpp_api.html#_CPPv4N5cudaq13s |
| N5cudaq14matrix_handler11instanti | ample_resultaSERR13sample_result) |
| ateENSt6stringERRNSt6vectorINSt6s | -                                 |
| ize_tEEERK20commutation_behavior) | [cudaq::sample_result::operator== |
| -   [cuda                         |     (C++                          |
| q::matrix_handler::matrix_handler |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     function)](api/languag        | ample_resulteqERK13sample_result) |
| es/cpp_api.html#_CPPv4I0_NSt11ena | -   [                             |
| ble_if_tINSt12is_base_of_vI16oper | cudaq::sample_result::probability |
| ator_handler1TEEbEEEN5cudaq14matr |     (C++                          |
| ix_handler14matrix_handlerERK1T), |     function)](api/lan            |
|     [\[1\]](ap                    | guages/cpp_api.html#_CPPv4NK5cuda |
| i/languages/cpp_api.html#_CPPv4I0 | q13sample_result11probabilityENSt |
| _NSt11enable_if_tINSt12is_base_of | 11string_viewEKNSt11string_viewE) |
| _vI16operator_handler1TEEbEEEN5cu | -   [cud                          |
| daq14matrix_handler14matrix_handl | aq::sample_result::register_names |
| erERK1TRK20commutation_behavior), |     (C++                          |
|     [\[2\]](api/languages/cpp_ap  |     function)](api/langu          |
| i.html#_CPPv4N5cudaq14matrix_hand | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| ler14matrix_handlerENSt6size_tE), | 3sample_result14register_namesEv) |
|     [\[3\]](api/                  | -                                 |
| languages/cpp_api.html#_CPPv4N5cu |    [cudaq::sample_result::reorder |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erENSt6stringERKNSt6vectorINSt6si |     function)](api/langua         |
| ze_tEEERK20commutation_behavior), | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     [\[4\]](api/                  | ample_result7reorderERKNSt6vector |
| languages/cpp_api.html#_CPPv4N5cu | INSt6size_tEEEKNSt11string_viewE) |
| daq14matrix_handler14matrix_handl | -   [cu                           |
| erENSt6stringERRNSt6vectorINSt6si | daq::sample_result::sample_result |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\                            |     func                          |
| [5\]](api/languages/cpp_api.html# | tion)](api/languages/cpp_api.html |
| _CPPv4N5cudaq14matrix_handler14ma | #_CPPv4N5cudaq13sample_result13sa |
| trix_handlerERK14matrix_handler), | mple_resultERK15ExecutionResult), |
|     [                             |     [\[1\]](api/la                |
| \[6\]](api/languages/cpp_api.html | nguages/cpp_api.html#_CPPv4N5cuda |
| #_CPPv4N5cudaq14matrix_handler14m | q13sample_result13sample_resultER |
| atrix_handlerERR14matrix_handler) | KNSt6vectorI15ExecutionResultEE), |
| -                                 |                                   |
|  [cudaq::matrix_handler::momentum |  [\[2\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq13sample_result13 |
|     function)](api/language       | sample_resultERR13sample_result), |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     [                             |
| rix_handler8momentumENSt6size_tE) | \[3\]](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq13sample_result13sa |
|    [cudaq::matrix_handler::number | mple_resultERR15ExecutionResult), |
|     (C++                          |     [\[4\]](api/lan               |
|     function)](api/langua         | guages/cpp_api.html#_CPPv4N5cudaq |
| ges/cpp_api.html#_CPPv4N5cudaq14m | 13sample_result13sample_resultEdR |
| atrix_handler6numberENSt6size_tE) | KNSt6vectorI15ExecutionResultEE), |
| -                                 |     [\[5\]](api/lan               |
| [cudaq::matrix_handler::operator= | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 13sample_result13sample_resultEv) |
|     fun                           | -                                 |
| ction)](api/languages/cpp_api.htm |  [cudaq::sample_result::serialize |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     (C++                          |
| NSt7is_sameI1T14matrix_handlerE5v |     function)](api                |
| alueENSt12is_base_of_vI16operator | /languages/cpp_api.html#_CPPv4NK5 |
| _handler1TEEEbEEEN5cudaq14matrix_ | cudaq13sample_result9serializeEv) |
| handleraSER14matrix_handlerRK1T), | -   [cudaq::sample_result::size   |
|     [\[1\]](api/languages         |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq14matr |     function)](api/languages/c    |
| ix_handleraSERK14matrix_handler), | pp_api.html#_CPPv4NK5cudaq13sampl |
|     [\[2\]](api/language          | e_result4sizeEKNSt11string_viewE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::sample_result::to_map |
| rix_handleraSERR14matrix_handler) |     (C++                          |
| -   [                             |     function)](api/languages/cpp  |
| cudaq::matrix_handler::operator== | _api.html#_CPPv4NK5cudaq13sample_ |
|     (C++                          | result6to_mapEKNSt11string_viewE) |
|     function)](api/languages      | -   [cuda                         |
| /cpp_api.html#_CPPv4NK5cudaq14mat | q::sample_result::\~sample_result |
| rix_handlereqERK14matrix_handler) |     (C++                          |
| -                                 |     funct                         |
|    [cudaq::matrix_handler::parity | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq13sample_resultD0Ev) |
|     function)](api/langua         | -   [cudaq::scalar_callback (C++  |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     c                             |
| atrix_handler6parityENSt6size_tE) | lass)](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq15scalar_callbackE) |
|  [cudaq::matrix_handler::position | -   [c                            |
|     (C++                          | udaq::scalar_callback::operator() |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     function)](api/language       |
| rix_handler8positionENSt6size_tE) | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| -   [cudaq::                      | alar_callbackclERKNSt13unordered_ |
| matrix_handler::remove_definition | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [                             |
|     fu                            | cudaq::scalar_callback::operator= |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14matrix_handler1 |     function)](api/languages/c    |
| 7remove_definitionERKNSt6stringE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _callbackaSERK15scalar_callback), |
|   [cudaq::matrix_handler::squeeze |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|     function)](api/languag        | r_callbackaSERR15scalar_callback) |
| es/cpp_api.html#_CPPv4N5cudaq14ma | -   [cudaq:                       |
| trix_handler7squeezeENSt6size_tE) | :scalar_callback::scalar_callback |
| -   [cudaq::m                     |     (C++                          |
| atrix_handler::to_diagonal_matrix |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4I0_NSt11ena |
|     function)](api/lang           | ble_if_tINSt16is_invocable_r_vINS |
| uages/cpp_api.html#_CPPv4NK5cudaq | t7complexIdEE8CallableRKNSt13unor |
| 14matrix_handler18to_diagonal_mat | dered_mapINSt6stringENSt7complexI |
| rixERNSt13unordered_mapINSt6size_ | dEEEEEEbEEEN5cudaq15scalar_callba |
| tENSt7int64_tEEERKNSt13unordered_ | ck15scalar_callbackERR8Callable), |
| mapINSt6stringENSt7complexIdEEEE) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
| [cudaq::matrix_handler::to_matrix | Pv4N5cudaq15scalar_callback15scal |
|     (C++                          | ar_callbackERK15scalar_callback), |
|     function)                     |     [\[2                          |
| ](api/languages/cpp_api.html#_CPP | \]](api/languages/cpp_api.html#_C |
| v4NK5cudaq14matrix_handler9to_mat | PPv4N5cudaq15scalar_callback15sca |
| rixERNSt13unordered_mapINSt6size_ | lar_callbackERR15scalar_callback) |
| tENSt7int64_tEEERKNSt13unordered_ | -   [cudaq::scalar_operator (C++  |
| mapINSt6stringENSt7complexIdEEEE) |     c                             |
| -                                 | lass)](api/languages/cpp_api.html |
| [cudaq::matrix_handler::to_string | #_CPPv4N5cudaq15scalar_operatorE) |
|     (C++                          | -                                 |
|     function)](api/               | [cudaq::scalar_operator::evaluate |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14matrix_handler9to_stringEb) |                                   |
| -                                 |    function)](api/languages/cpp_a |
| [cudaq::matrix_handler::unique_id | pi.html#_CPPv4NK5cudaq15scalar_op |
|     (C++                          | erator8evaluateERKNSt13unordered_ |
|     function)](api/               | mapINSt6stringENSt7complexIdEEEE) |
| languages/cpp_api.html#_CPPv4NK5c | -   [cudaq::scalar_ope            |
| udaq14matrix_handler9unique_idEv) | rator::get_parameter_descriptions |
| -   [cudaq:                       |     (C++                          |
| :matrix_handler::\~matrix_handler |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     functi                        | tml#_CPPv4NK5cudaq15scalar_operat |
| on)](api/languages/cpp_api.html#_ | or26get_parameter_descriptionsEv) |
| CPPv4N5cudaq14matrix_handlerD0Ev) | -   [cu                           |
| -   [cudaq::matrix_op (C++        | daq::scalar_operator::is_constant |
|     type)](api/languages/cpp_a    |     (C++                          |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     function)](api/lang           |
| -   [cudaq::matrix_op_term (C++   | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 15scalar_operator11is_constantEv) |
|  type)](api/languages/cpp_api.htm | -   [c                            |
| l#_CPPv4N5cudaq14matrix_op_termE) | udaq::scalar_operator::operator\* |
| -                                 |     (C++                          |
|    [cudaq::mdiag_operator_handler |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     class)](                      | Pv4N5cudaq15scalar_operatormlENSt |
| api/languages/cpp_api.html#_CPPv4 | 7complexIdEERK15scalar_operator), |
| N5cudaq22mdiag_operator_handlerE) |     [\[1\                         |
| -   [cudaq::mpi (C++              | ]](api/languages/cpp_api.html#_CP |
|     type)](api/languages          | Pv4N5cudaq15scalar_operatormlENSt |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::mpi::all_gather (C++  |     [\[2\]](api/languages/cp      |
|     fu                            | p_api.html#_CPPv4N5cudaq15scalar_ |
| nction)](api/languages/cpp_api.ht | operatormlEdRK15scalar_operator), |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     [\[3\]](api/languages/cp      |
| RNSt6vectorIdEERKNSt6vectorIdEE), | p_api.html#_CPPv4N5cudaq15scalar_ |
|                                   | operatormlEdRR15scalar_operator), |
|   [\[1\]](api/languages/cpp_api.h |     [\[4\]](api/languages         |
| tml#_CPPv4N5cudaq3mpi10all_gather | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | alar_operatormlENSt7complexIdEE), |
| -   [cudaq::mpi::all_reduce (C++  |     [\[5\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4NKR5cudaq15scalar |
|  function)](api/languages/cpp_api | _operatormlERK15scalar_operator), |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     [\[6\]]                       |
| reduceE1TRK1TRK14BinaryFunction), | (api/languages/cpp_api.html#_CPPv |
|     [\[1\]](api/langu             | 4NKR5cudaq15scalar_operatormlEd), |
| ages/cpp_api.html#_CPPv4I00EN5cud |     [\[7\]](api/language          |
| aq3mpi10all_reduceE1TRK1TRK4Func) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::mpi::broadcast (C++   | alar_operatormlENSt7complexIdEE), |
|     function)](api/               |     [\[8\]](api/languages/cp      |
| languages/cpp_api.html#_CPPv4N5cu | p_api.html#_CPPv4NO5cudaq15scalar |
| daq3mpi9broadcastERNSt6stringEi), | _operatormlERK15scalar_operator), |
|     [\[1\]](api/la                |     [\[9\                         |
| nguages/cpp_api.html#_CPPv4N5cuda | ]](api/languages/cpp_api.html#_CP |
| q3mpi9broadcastERNSt6vectorIdEEi) | Pv4NO5cudaq15scalar_operatormlEd) |
| -   [cudaq::mpi::finalize (C++    | -   [cu                           |
|     f                             | daq::scalar_operator::operator\*= |
| unction)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     function)](api/languag        |
| -   [cudaq::mpi::initialize (C++  | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     function                      | alar_operatormLENSt7complexIdEE), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/languages/c       |
| Pv4N5cudaq3mpi10initializeEiPPc), | pp_api.html#_CPPv4N5cudaq15scalar |
|     [                             | _operatormLERK15scalar_operator), |
| \[1\]](api/languages/cpp_api.html |     [\[2                          |
| #_CPPv4N5cudaq3mpi10initializeEv) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::mpi::is_initialized   | PPv4N5cudaq15scalar_operatormLEd) |
|     (C++                          | -   [                             |
|     function                      | cudaq::scalar_operator::operator+ |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq3mpi14is_initializedEv) |     function                      |
| -   [cudaq::mpi::num_ranks (C++   | )](api/languages/cpp_api.html#_CP |
|     fu                            | Pv4N5cudaq15scalar_operatorplENSt |
| nction)](api/languages/cpp_api.ht | 7complexIdEERK15scalar_operator), |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     [\[1\                         |
| -   [cudaq::mpi::rank (C++        | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_operatorplENSt |
|    function)](api/languages/cpp_a | 7complexIdEERR15scalar_operator), |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     [\[2\]](api/languages/cp      |
| -   [cudaq::noise_model (C++      | p_api.html#_CPPv4N5cudaq15scalar_ |
|                                   | operatorplEdRK15scalar_operator), |
|    class)](api/languages/cpp_api. |     [\[3\]](api/languages/cp      |
| html#_CPPv4N5cudaq11noise_modelE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::n                     | operatorplEdRR15scalar_operator), |
| oise_model::add_all_qubit_channel |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     function)](api                | alar_operatorplENSt7complexIdEE), |
| /languages/cpp_api.html#_CPPv4IDp |     [\[5\]](api/languages/cpp     |
| EN5cudaq11noise_model21add_all_qu | _api.html#_CPPv4NKR5cudaq15scalar |
| bit_channelEvRK13kraus_channeli), | _operatorplERK15scalar_operator), |
|     [\[1\]](api/langua            |     [\[6\]]                       |
| ges/cpp_api.html#_CPPv4N5cudaq11n | (api/languages/cpp_api.html#_CPPv |
| oise_model21add_all_qubit_channel | 4NKR5cudaq15scalar_operatorplEd), |
| ERKNSt6stringERK13kraus_channeli) |     [\[7\]]                       |
| -                                 | (api/languages/cpp_api.html#_CPPv |
|  [cudaq::noise_model::add_channel | 4NKR5cudaq15scalar_operatorplEv), |
|     (C++                          |     [\[8\]](api/language          |
|     funct                         | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| ion)](api/languages/cpp_api.html# | alar_operatorplENSt7complexIdEE), |
| _CPPv4IDpEN5cudaq11noise_model11a |     [\[9\]](api/languages/cp      |
| dd_channelEvRK15PredicateFuncTy), | p_api.html#_CPPv4NO5cudaq15scalar |
|     [\[1\]](api/languages/cpp_    | _operatorplERK15scalar_operator), |
| api.html#_CPPv4IDpEN5cudaq11noise |     [\[10\]                       |
| _model11add_channelEvRKNSt6vector | ](api/languages/cpp_api.html#_CPP |
| INSt6size_tEEERK13kraus_channel), | v4NO5cudaq15scalar_operatorplEd), |
|     [\[2\]](ap                    |     [\[11\                        |
| i/languages/cpp_api.html#_CPPv4N5 | ]](api/languages/cpp_api.html#_CP |
| cudaq11noise_model11add_channelER | Pv4NO5cudaq15scalar_operatorplEv) |
| KNSt6stringERK15PredicateFuncTy), | -   [c                            |
|                                   | udaq::scalar_operator::operator+= |
| [\[3\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq11noise_model11add |     function)](api/languag        |
| _channelERKNSt6stringERKNSt6vecto | es/cpp_api.html#_CPPv4N5cudaq15sc |
| rINSt6size_tEEERK13kraus_channel) | alar_operatorpLENSt7complexIdEE), |
| -   [cudaq::noise_model::empty    |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function                      | _operatorpLERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[2                          |
| Pv4NK5cudaq11noise_model5emptyEv) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq15scalar_operatorpLEd) |
| [cudaq::noise_model::get_channels | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator- |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4I0ENK |     function                      |
| 5cudaq11noise_model12get_channels | )](api/languages/cpp_api.html#_CP |
| ENSt6vectorI13kraus_channelEERKNS | Pv4N5cudaq15scalar_operatormiENSt |
| t6vectorINSt6size_tEEERKNSt6vecto | 7complexIdEERK15scalar_operator), |
| rINSt6size_tEEERKNSt6vectorIdEE), |     [\[1\                         |
|     [\[1\]](api/languages/cpp_a   | ]](api/languages/cpp_api.html#_CP |
| pi.html#_CPPv4NK5cudaq11noise_mod | Pv4N5cudaq15scalar_operatormiENSt |
| el12get_channelsERKNSt6stringERKN | 7complexIdEERR15scalar_operator), |
| St6vectorINSt6size_tEEERKNSt6vect |     [\[2\]](api/languages/cp      |
| orINSt6size_tEEERKNSt6vectorIdEE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -                                 | operatormiEdRK15scalar_operator), |
|  [cudaq::noise_model::noise_model |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     function)](api                | operatormiEdRR15scalar_operator), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[4\]](api/languages         |
| udaq11noise_model11noise_modelEv) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -   [cu                           | alar_operatormiENSt7complexIdEE), |
| daq::noise_model::PredicateFuncTy |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|     type)](api/la                 | _operatormiERK15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[6\]]                       |
| q11noise_model15PredicateFuncTyE) | (api/languages/cpp_api.html#_CPPv |
| -   [cud                          | 4NKR5cudaq15scalar_operatormiEd), |
| aq::noise_model::register_channel |     [\[7\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/languages      | 4NKR5cudaq15scalar_operatormiEv), |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     [\[8\]](api/language          |
| noise_model16register_channelEvv) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::                      | alar_operatormiENSt7complexIdEE), |
| noise_model::requires_constructor |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     type)](api/languages/cp       | _operatormiERK15scalar_operator), |
| p_api.html#_CPPv4I0DpEN5cudaq11no |     [\[10\]                       |
| ise_model20requires_constructorE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::noise_model_type (C++ | v4NO5cudaq15scalar_operatormiEd), |
|     e                             |     [\[11\                        |
| num)](api/languages/cpp_api.html# | ]](api/languages/cpp_api.html#_CP |
| _CPPv4N5cudaq16noise_model_typeE) | Pv4NO5cudaq15scalar_operatormiEv) |
| -   [cudaq::no                    | -   [c                            |
| ise_model_type::amplitude_damping | udaq::scalar_operator::operator-= |
|     (C++                          |     (C++                          |
|     enumerator)](api/languages    |     function)](api/languag        |
| /cpp_api.html#_CPPv4N5cudaq16nois | es/cpp_api.html#_CPPv4N5cudaq15sc |
| e_model_type17amplitude_dampingE) | alar_operatormIENSt7complexIdEE), |
| -   [cudaq::noise_mode            |     [\[1\]](api/languages/c       |
| l_type::amplitude_damping_channel | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatormIERK15scalar_operator), |
|     e                             |     [\[2                          |
| numerator)](api/languages/cpp_api | \]](api/languages/cpp_api.html#_C |
| .html#_CPPv4N5cudaq16noise_model_ | PPv4N5cudaq15scalar_operatormIEd) |
| type25amplitude_damping_channelE) | -   [                             |
| -   [cudaq::n                     | cudaq::scalar_operator::operator/ |
| oise_model_type::bit_flip_channel |     (C++                          |
|     (C++                          |     function                      |
|     enumerator)](api/language     | )](api/languages/cpp_api.html#_CP |
| s/cpp_api.html#_CPPv4N5cudaq16noi | Pv4N5cudaq15scalar_operatordvENSt |
| se_model_type16bit_flip_channelE) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::                      |     [\[1\                         |
| noise_model_type::depolarization1 | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|     enumerator)](api/languag      | 7complexIdEERR15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[2\]](api/languages/cp      |
| ise_model_type15depolarization1E) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::                      | operatordvEdRK15scalar_operator), |
| noise_model_type::depolarization2 |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](api/languag      | operatordvEdRR15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[4\]](api/languages         |
| ise_model_type15depolarization2E) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -   [cudaq::noise_m               | alar_operatordvENSt7complexIdEE), |
| odel_type::depolarization_channel |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|                                   | _operatordvERK15scalar_operator), |
|   enumerator)](api/languages/cpp_ |     [\[6\]]                       |
| api.html#_CPPv4N5cudaq16noise_mod | (api/languages/cpp_api.html#_CPPv |
| el_type22depolarization_channelE) | 4NKR5cudaq15scalar_operatordvEd), |
| -                                 |     [\[7\]](api/language          |
|  [cudaq::noise_model_type::pauli1 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatordvENSt7complexIdEE), |
|     enumerator)](a                |     [\[8\]](api/languages/cp      |
| pi/languages/cpp_api.html#_CPPv4N | p_api.html#_CPPv4NO5cudaq15scalar |
| 5cudaq16noise_model_type6pauli1E) | _operatordvERK15scalar_operator), |
| -                                 |     [\[9\                         |
|  [cudaq::noise_model_type::pauli2 | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatordvEd) |
|     enumerator)](a                | -   [c                            |
| pi/languages/cpp_api.html#_CPPv4N | udaq::scalar_operator::operator/= |
| 5cudaq16noise_model_type6pauli2E) |     (C++                          |
| -   [cudaq                        |     function)](api/languag        |
| ::noise_model_type::phase_damping | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatordVENSt7complexIdEE), |
|     enumerator)](api/langu        |     [\[1\]](api/languages/c       |
| ages/cpp_api.html#_CPPv4N5cudaq16 | pp_api.html#_CPPv4N5cudaq15scalar |
| noise_model_type13phase_dampingE) | _operatordVERK15scalar_operator), |
| -   [cudaq::noi                   |     [\[2                          |
| se_model_type::phase_flip_channel | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatordVEd) |
|     enumerator)](api/languages/   | -   [                             |
| cpp_api.html#_CPPv4N5cudaq16noise | cudaq::scalar_operator::operator= |
| _model_type18phase_flip_channelE) |     (C++                          |
| -                                 |     function)](api/languages/c    |
| [cudaq::noise_model_type::unknown | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatoraSERK15scalar_operator), |
|     enumerator)](ap               |     [\[1\]](api/languages/        |
| i/languages/cpp_api.html#_CPPv4N5 | cpp_api.html#_CPPv4N5cudaq15scala |
| cudaq16noise_model_type7unknownE) | r_operatoraSERR15scalar_operator) |
| -                                 | -   [c                            |
| [cudaq::noise_model_type::x_error | udaq::scalar_operator::operator== |
|     (C++                          |     (C++                          |
|     enumerator)](ap               |     function)](api/languages/c    |
| i/languages/cpp_api.html#_CPPv4N5 | pp_api.html#_CPPv4NK5cudaq15scala |
| cudaq16noise_model_type7x_errorE) | r_operatoreqERK15scalar_operator) |
| -                                 | -   [cudaq:                       |
| [cudaq::noise_model_type::y_error | :scalar_operator::scalar_operator |
|     (C++                          |     (C++                          |
|     enumerator)](ap               |     func                          |
| i/languages/cpp_api.html#_CPPv4N5 | tion)](api/languages/cpp_api.html |
| cudaq16noise_model_type7y_errorE) | #_CPPv4N5cudaq15scalar_operator15 |
| -                                 | scalar_operatorENSt7complexIdEE), |
| [cudaq::noise_model_type::z_error |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     enumerator)](ap               | scalar_operator15scalar_operatorE |
| i/languages/cpp_api.html#_CPPv4N5 | RK15scalar_callbackRRNSt13unorder |
| cudaq16noise_model_type7z_errorE) | ed_mapINSt6stringENSt6stringEEE), |
| -   [cudaq::num_available_gpus    |     [\[2\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function                      | Pv4N5cudaq15scalar_operator15scal |
| )](api/languages/cpp_api.html#_CP | ar_operatorERK15scalar_operator), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[3\]](api/langu             |
| -   [cudaq::observe (C++          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     function)]                    | scalar_operator15scalar_operatorE |
| (api/languages/cpp_api.html#_CPPv | RR15scalar_callbackRRNSt13unorder |
| 4I00DpEN5cudaq7observeENSt6vector | ed_mapINSt6stringENSt6stringEEE), |
| I14observe_resultEERR13QuantumKer |     [\[4\                         |
| nelRK15SpinOpContainerDpRR4Args), | ]](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/languages/cpp_ap  | Pv4N5cudaq15scalar_operator15scal |
| i.html#_CPPv4I0DpEN5cudaq7observe | ar_operatorERR15scalar_operator), |
| E14observe_resultNSt6size_tERR13Q |     [\[5\]](api/language          |
| uantumKernelRK7spin_opDpRR4Args), | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     [\[                           | lar_operator15scalar_operatorEd), |
| 2\]](api/languages/cpp_api.html#_ |     [\[6\]](api/languag           |
| CPPv4I0DpEN5cudaq7observeE14obser | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ve_resultRK15observe_optionsRR13Q | alar_operator15scalar_operatorEv) |
| uantumKernelRK7spin_opDpRR4Args), | -   [                             |
|     [\[3\]](api/lang              | cudaq::scalar_operator::to_matrix |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     (C++                          |
| udaq7observeE14observe_resultRR13 |                                   |
| QuantumKernelRK7spin_opDpRR4Args) |   function)](api/languages/cpp_ap |
| -   [cudaq::observe_options (C++  | i.html#_CPPv4NK5cudaq15scalar_ope |
|     st                            | rator9to_matrixERKNSt13unordered_ |
| ruct)](api/languages/cpp_api.html | mapINSt6stringENSt7complexIdEEEE) |
| #_CPPv4N5cudaq15observe_optionsE) | -   [                             |
| -   [cudaq::observe_result (C++   | cudaq::scalar_operator::to_string |
|                                   |     (C++                          |
| class)](api/languages/cpp_api.htm |     function)](api/l              |
| l#_CPPv4N5cudaq14observe_resultE) | anguages/cpp_api.html#_CPPv4NK5cu |
| -                                 | daq15scalar_operator9to_stringEv) |
|    [cudaq::observe_result::counts | -   [cudaq::s                     |
|     (C++                          | calar_operator::\~scalar_operator |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq14observ |     functio                       |
| e_result6countsERK12spin_op_term) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::observe_result::dump  | PPv4N5cudaq15scalar_operatorD0Ev) |
|     (C++                          | -   [cudaq::set_noise (C++        |
|     function)                     |     function)](api/langu          |
| ](api/languages/cpp_api.html#_CPP | ages/cpp_api.html#_CPPv4N5cudaq9s |
| v4N5cudaq14observe_result4dumpEv) | et_noiseERKN5cudaq11noise_modelE) |
| -   [c                            | -   [cudaq::set_random_seed (C++  |
| udaq::observe_result::expectation |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq15set_random_seedENSt6size_tE) |
| function)](api/languages/cpp_api. | -   [cudaq::simulation_precision  |
| html#_CPPv4N5cudaq14observe_resul |     (C++                          |
| t11expectationERK12spin_op_term), |     enum)                         |
|     [\[1\]](api/la                | ](api/languages/cpp_api.html#_CPP |
| nguages/cpp_api.html#_CPPv4N5cuda | v4N5cudaq20simulation_precisionE) |
| q14observe_result11expectationEv) | -   [                             |
| -   [cuda                         | cudaq::simulation_precision::fp32 |
| q::observe_result::id_coefficient |     (C++                          |
|     (C++                          |     enumerator)](api              |
|     function)](api/langu          | /languages/cpp_api.html#_CPPv4N5c |
| ages/cpp_api.html#_CPPv4N5cudaq14 | udaq20simulation_precision4fp32E) |
| observe_result14id_coefficientEv) | -   [                             |
| -   [cuda                         | cudaq::simulation_precision::fp64 |
| q::observe_result::observe_result |     (C++                          |
|     (C++                          |     enumerator)](api              |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
|   function)](api/languages/cpp_ap | udaq20simulation_precision4fp64E) |
| i.html#_CPPv4N5cudaq14observe_res | -   [cudaq::SimulationState (C++  |
| ult14observe_resultEdRK7spin_op), |     c                             |
|     [\[1\]](a                     | lass)](api/languages/cpp_api.html |
| pi/languages/cpp_api.html#_CPPv4N | #_CPPv4N5cudaq15SimulationStateE) |
| 5cudaq14observe_result14observe_r | -   [                             |
| esultEdRK7spin_op13sample_result) | cudaq::SimulationState::precision |
| -                                 |     (C++                          |
|  [cudaq::observe_result::operator |     enum)](api                    |
|     double (C++                   | /languages/cpp_api.html#_CPPv4N5c |
|     functio                       | udaq15SimulationState9precisionE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq:                       |
| PPv4N5cudaq14observe_resultcvdEv) | :SimulationState::precision::fp32 |
| -                                 |     (C++                          |
|  [cudaq::observe_result::raw_data |     enumerator)](api/lang         |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     function)](ap                 | 5SimulationState9precision4fp32E) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq:                       |
| cudaq14observe_result8raw_dataEv) | :SimulationState::precision::fp64 |
| -   [cudaq::operator_handler (C++ |     (C++                          |
|     cl                            |     enumerator)](api/lang         |
| ass)](api/languages/cpp_api.html# | uages/cpp_api.html#_CPPv4N5cudaq1 |
| _CPPv4N5cudaq16operator_handlerE) | 5SimulationState9precision4fp64E) |
| -   [cudaq::optimizable_function  | -                                 |
|     (C++                          |   [cudaq::SimulationState::Tensor |
|     class)                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     struct)](                     |
| v4N5cudaq20optimizable_functionE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::optimization_result   | N5cudaq15SimulationState6TensorE) |
|     (C++                          | -   [cudaq::spin_handler (C++     |
|     type                          |                                   |
| )](api/languages/cpp_api.html#_CP |   class)](api/languages/cpp_api.h |
| Pv4N5cudaq19optimization_resultE) | tml#_CPPv4N5cudaq12spin_handlerE) |
| -   [cudaq::optimizer (C++        | -   [cudaq:                       |
|     class)](api/languages/cpp_a   | :spin_handler::to_diagonal_matrix |
| pi.html#_CPPv4N5cudaq9optimizerE) |     (C++                          |
| -   [cudaq::optimizer::optimize   |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4NK5cud |
|                                   | aq12spin_handler18to_diagonal_mat |
|  function)](api/languages/cpp_api | rixERNSt13unordered_mapINSt6size_ |
| .html#_CPPv4N5cudaq9optimizer8opt | tENSt7int64_tEEERKNSt13unordered_ |
| imizeEKiRR20optimizable_function) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cu                           | -                                 |
| daq::optimizer::requiresGradients |   [cudaq::spin_handler::to_matrix |
|     (C++                          |     (C++                          |
|     function)](api/la             |     function                      |
| nguages/cpp_api.html#_CPPv4N5cuda | )](api/languages/cpp_api.html#_CP |
| q9optimizer17requiresGradientsEv) | Pv4N5cudaq12spin_handler9to_matri |
| -   [cudaq::orca (C++             | xERKNSt6stringENSt7complexIdEEb), |
|     type)](api/languages/         |     [\[1                          |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::orca::sample (C++     | PPv4NK5cudaq12spin_handler9to_mat |
|     function)](api/languages/c    | rixERNSt13unordered_mapINSt6size_ |
| pp_api.html#_CPPv4N5cudaq4orca6sa | tENSt7int64_tEEERKNSt13unordered_ |
| mpleERNSt6vectorINSt6size_tEEERNS | mapINSt6stringENSt7complexIdEEEE) |
| t6vectorINSt6size_tEEERNSt6vector | -   [cuda                         |
| IdEERNSt6vectorIdEEiNSt6size_tE), | q::spin_handler::to_sparse_matrix |
|     [\[1\]]                       |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/               |
| 4N5cudaq4orca6sampleERNSt6vectorI | languages/cpp_api.html#_CPPv4N5cu |
| NSt6size_tEEERNSt6vectorINSt6size | daq12spin_handler16to_sparse_matr |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | ixERKNSt6stringENSt7complexIdEEb) |
| -   [cudaq::orca::sample_async    | -                                 |
|     (C++                          |   [cudaq::spin_handler::to_string |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     function)](ap                 |
| html#_CPPv4N5cudaq4orca12sample_a | i/languages/cpp_api.html#_CPPv4NK |
| syncERNSt6vectorINSt6size_tEEERNS | 5cudaq12spin_handler9to_stringEb) |
| t6vectorINSt6size_tEEERNSt6vector | -                                 |
| IdEERNSt6vectorIdEEiNSt6size_tE), |   [cudaq::spin_handler::unique_id |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](ap                 |
| q4orca12sample_asyncERNSt6vectorI | i/languages/cpp_api.html#_CPPv4NK |
| NSt6size_tEEERNSt6vectorINSt6size | 5cudaq12spin_handler9unique_idEv) |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | -   [cudaq::spin_op (C++          |
| -   [cudaq::OrcaRemoteRESTQPU     |     type)](api/languages/cpp      |
|     (C++                          | _api.html#_CPPv4N5cudaq7spin_opE) |
|     cla                           | -   [cudaq::spin_op_term (C++     |
| ss)](api/languages/cpp_api.html#_ |                                   |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |    type)](api/languages/cpp_api.h |
| -   [cudaq::pauli1 (C++           | tml#_CPPv4N5cudaq12spin_op_termE) |
|     class)](api/languages/cp      | -   [cudaq::state (C++            |
| p_api.html#_CPPv4N5cudaq6pauli1E) |     class)](api/languages/c       |
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
