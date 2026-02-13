::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-3948
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
| -   [canonicalize()               | -   [cudaq::pauli1::num_targets   |
|     (cu                           |     (C++                          |
| daq.operators.boson.BosonOperator |     membe                         |
|     method)](api/languages        | r)](api/languages/cpp_api.html#_C |
| /python_api.html#cudaq.operators. | PPv4N5cudaq6pauli111num_targetsE) |
| boson.BosonOperator.canonicalize) | -   [cudaq::pauli1::pauli1 (C++   |
|     -   [(cudaq.                  |     function)](api/languages/cpp_ |
| operators.boson.BosonOperatorTerm | api.html#_CPPv4N5cudaq6pauli16pau |
|                                   | li1ERKNSt6vectorIN5cudaq4realEEE) |
|        method)](api/languages/pyt | -   [cudaq::pauli2 (C++           |
| hon_api.html#cudaq.operators.boso |     class)](api/languages/cp      |
| n.BosonOperatorTerm.canonicalize) | p_api.html#_CPPv4N5cudaq6pauli2E) |
|     -   [(cudaq.                  | -                                 |
| operators.fermion.FermionOperator |    [cudaq::pauli2::num_parameters |
|                                   |     (C++                          |
|        method)](api/languages/pyt |     member)]                      |
| hon_api.html#cudaq.operators.ferm | (api/languages/cpp_api.html#_CPPv |
| ion.FermionOperator.canonicalize) | 4N5cudaq6pauli214num_parametersE) |
|     -   [(cudaq.oper              | -   [cudaq::pauli2::num_targets   |
| ators.fermion.FermionOperatorTerm |     (C++                          |
|                                   |     membe                         |
|    method)](api/languages/python_ | r)](api/languages/cpp_api.html#_C |
| api.html#cudaq.operators.fermion. | PPv4N5cudaq6pauli211num_targetsE) |
| FermionOperatorTerm.canonicalize) | -   [cudaq::pauli2::pauli2 (C++   |
|     -                             |     function)](api/languages/cpp_ |
|  [(cudaq.operators.MatrixOperator | api.html#_CPPv4N5cudaq6pauli26pau |
|         method)](api/lang         | li2ERKNSt6vectorIN5cudaq4realEEE) |
| uages/python_api.html#cudaq.opera | -   [cudaq::phase_damping (C++    |
| tors.MatrixOperator.canonicalize) |                                   |
|     -   [(c                       |  class)](api/languages/cpp_api.ht |
| udaq.operators.MatrixOperatorTerm | ml#_CPPv4N5cudaq13phase_dampingE) |
|         method)](api/language     | -   [cud                          |
| s/python_api.html#cudaq.operators | aq::phase_damping::num_parameters |
| .MatrixOperatorTerm.canonicalize) |     (C++                          |
|     -   [(                        |     member)](api/lan              |
| cudaq.operators.spin.SpinOperator | guages/cpp_api.html#_CPPv4N5cudaq |
|         method)](api/languag      | 13phase_damping14num_parametersE) |
| es/python_api.html#cudaq.operator | -   [                             |
| s.spin.SpinOperator.canonicalize) | cudaq::phase_damping::num_targets |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     member)](api/                 |
|         method)](api/languages/p  | languages/cpp_api.html#_CPPv4N5cu |
| ython_api.html#cudaq.operators.sp | daq13phase_damping11num_targetsE) |
| in.SpinOperatorTerm.canonicalize) | -   [cudaq::phase_flip_channel    |
| -   [canonicalized() (in module   |     (C++                          |
|     cuda                          |     clas                          |
| q.boson)](api/languages/python_ap | s)](api/languages/cpp_api.html#_C |
| i.html#cudaq.boson.canonicalized) | PPv4N5cudaq18phase_flip_channelE) |
|     -   [(in module               | -   [cudaq::p                     |
|         cudaq.fe                  | hase_flip_channel::num_parameters |
| rmion)](api/languages/python_api. |     (C++                          |
| html#cudaq.fermion.canonicalized) |     member)](api/language         |
|     -   [(in module               | s/cpp_api.html#_CPPv4N5cudaq18pha |
|                                   | se_flip_channel14num_parametersE) |
|        cudaq.operators.custom)](a | -   [cudaq                        |
| pi/languages/python_api.html#cuda | ::phase_flip_channel::num_targets |
| q.operators.custom.canonicalized) |     (C++                          |
|     -   [(in module               |     member)](api/langu            |
|         cu                        | ages/cpp_api.html#_CPPv4N5cudaq18 |
| daq.spin)](api/languages/python_a | phase_flip_channel11num_targetsE) |
| pi.html#cudaq.spin.canonicalized) | -   [cudaq::product_op (C++       |
| -   [captured_variables()         |                                   |
|     (cudaq.PyKernelDecorator      |  class)](api/languages/cpp_api.ht |
|     method)](api/lan              | ml#_CPPv4I0EN5cudaq10product_opE) |
| guages/python_api.html#cudaq.PyKe | -   [cudaq::product_op::begin     |
| rnelDecorator.captured_variables) |     (C++                          |
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
| -   [compile()                    |     (C++                          |
|     (cudaq.PyKernelDecorator      |     function)](api/lang           |
|     metho                         | uages/cpp_api.html#_CPPv4N5cudaq1 |
| d)](api/languages/python_api.html | 0product_op14const_iteratorppEi), |
| #cudaq.PyKernelDecorator.compile) |     [\[1\]](api/lan               |
| -   [ComplexMatrix (class in      | guages/cpp_api.html#_CPPv4N5cudaq |
|     cudaq)](api/languages/pyt     | 10product_op14const_iteratorppEv) |
| hon_api.html#cudaq.ComplexMatrix) | -   [cudaq::produc                |
| -   [compute()                    | t_op::const_iterator::operator\-- |
|     (                             |     (C++                          |
| cudaq.gradients.CentralDifference |     function)](api/lang           |
|     method)](api/la               | uages/cpp_api.html#_CPPv4N5cudaq1 |
| nguages/python_api.html#cudaq.gra | 0product_op14const_iteratormmEi), |
| dients.CentralDifference.compute) |     [\[1\]](api/lan               |
|     -   [(                        | guages/cpp_api.html#_CPPv4N5cudaq |
| cudaq.gradients.ForwardDifference | 10product_op14const_iteratormmEv) |
|         method)](api/la           | -   [cudaq::produc                |
| nguages/python_api.html#cudaq.gra | t_op::const_iterator::operator-\> |
| dients.ForwardDifference.compute) |     (C++                          |
|     -                             |     function)](api/lan            |
|  [(cudaq.gradients.ParameterShift | guages/cpp_api.html#_CPPv4N5cudaq |
|         method)](api              | 10product_op14const_iteratorptEv) |
| /languages/python_api.html#cudaq. | -   [cudaq::produ                 |
| gradients.ParameterShift.compute) | ct_op::const_iterator::operator== |
| -   [const()                      |     (C++                          |
|                                   |     fun                           |
|   (cudaq.operators.ScalarOperator | ction)](api/languages/cpp_api.htm |
|     class                         | l#_CPPv4NK5cudaq10product_op14con |
|     method)](a                    | st_iteratoreqERK14const_iterator) |
| pi/languages/python_api.html#cuda | -   [cudaq::product_op::degrees   |
| q.operators.ScalarOperator.const) |     (C++                          |
| -   [copy()                       |     function)                     |
|     (cu                           | ](api/languages/cpp_api.html#_CPP |
| daq.operators.boson.BosonOperator | v4NK5cudaq10product_op7degreesEv) |
|     method)](api/l                | -   [cudaq::product_op::dump (C++ |
| anguages/python_api.html#cudaq.op |     functi                        |
| erators.boson.BosonOperator.copy) | on)](api/languages/cpp_api.html#_ |
|     -   [(cudaq.                  | CPPv4NK5cudaq10product_op4dumpEv) |
| operators.boson.BosonOperatorTerm | -   [cudaq::product_op::end (C++  |
|         method)](api/langu        |     funct                         |
| ages/python_api.html#cudaq.operat | ion)](api/languages/cpp_api.html# |
| ors.boson.BosonOperatorTerm.copy) | _CPPv4NK5cudaq10product_op3endEv) |
|     -   [(cudaq.                  | -   [c                            |
| operators.fermion.FermionOperator | udaq::product_op::get_coefficient |
|         method)](api/langu        |     (C++                          |
| ages/python_api.html#cudaq.operat |     function)](api/lan            |
| ors.fermion.FermionOperator.copy) | guages/cpp_api.html#_CPPv4NK5cuda |
|     -   [(cudaq.oper              | q10product_op15get_coefficientEv) |
| ators.fermion.FermionOperatorTerm | -                                 |
|         method)](api/languages    |   [cudaq::product_op::get_term_id |
| /python_api.html#cudaq.operators. |     (C++                          |
| fermion.FermionOperatorTerm.copy) |     function)](api                |
|     -                             | /languages/cpp_api.html#_CPPv4NK5 |
|  [(cudaq.operators.MatrixOperator | cudaq10product_op11get_term_idEv) |
|         method)](                 | -                                 |
| api/languages/python_api.html#cud |   [cudaq::product_op::is_identity |
| aq.operators.MatrixOperator.copy) |     (C++                          |
|     -   [(c                       |     function)](api                |
| udaq.operators.MatrixOperatorTerm | /languages/cpp_api.html#_CPPv4NK5 |
|         method)](api/             | cudaq10product_op11is_identityEv) |
| languages/python_api.html#cudaq.o | -   [cudaq::product_op::num_ops   |
| perators.MatrixOperatorTerm.copy) |     (C++                          |
|     -   [(                        |     function)                     |
| cudaq.operators.spin.SpinOperator | ](api/languages/cpp_api.html#_CPP |
|         method)](api              | v4NK5cudaq10product_op7num_opsEv) |
| /languages/python_api.html#cudaq. | -                                 |
| operators.spin.SpinOperator.copy) |    [cudaq::product_op::operator\* |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     function)](api/languages/     |
|         method)](api/lan          | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| guages/python_api.html#cudaq.oper | oduct_opmlE10product_opI1TERK15sc |
| ators.spin.SpinOperatorTerm.copy) | alar_operatorRK10product_opI1TE), |
| -   [count() (cudaq.Resources     |     [\[1\]](api/languages/        |
|     method)](api/languages/pytho  | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| n_api.html#cudaq.Resources.count) | oduct_opmlE10product_opI1TERK15sc |
|     -   [(cudaq.SampleResult      | alar_operatorRR10product_opI1TE), |
|                                   |     [\[2\]](api/languages/        |
|   method)](api/languages/python_a | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| pi.html#cudaq.SampleResult.count) | oduct_opmlE10product_opI1TERR15sc |
| -   [count_controls()             | alar_operatorRK10product_opI1TE), |
|     (cudaq.Resources              |     [\[3\]](api/languages/        |
|     meth                          | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| od)](api/languages/python_api.htm | oduct_opmlE10product_opI1TERR15sc |
| l#cudaq.Resources.count_controls) | alar_operatorRR10product_opI1TE), |
| -   [counts()                     |     [\[4\]](api/                  |
|     (cudaq.ObserveResult          | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opmlE6sum_opI1TER |
| method)](api/languages/python_api | K15scalar_operatorRK6sum_opI1TE), |
| .html#cudaq.ObserveResult.counts) |     [\[5\]](api/                  |
| -   [create() (in module          | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opmlE6sum_opI1TER |
|    cudaq.boson)](api/languages/py | K15scalar_operatorRR6sum_opI1TE), |
| thon_api.html#cudaq.boson.create) |     [\[6\]](api/                  |
|     -   [(in module               | languages/cpp_api.html#_CPPv4I0EN |
|         c                         | 5cudaq10product_opmlE6sum_opI1TER |
| udaq.fermion)](api/languages/pyth | R15scalar_operatorRK6sum_opI1TE), |
| on_api.html#cudaq.fermion.create) |     [\[7\]](api/                  |
| -   [csr_spmatrix (C++            | languages/cpp_api.html#_CPPv4I0EN |
|     type)](api/languages/c        | 5cudaq10product_opmlE6sum_opI1TER |
| pp_api.html#_CPPv412csr_spmatrix) | R15scalar_operatorRR6sum_opI1TE), |
| -   cudaq                         |     [\[8\]](api/languages         |
|     -   [module](api/langua       | /cpp_api.html#_CPPv4NK5cudaq10pro |
| ges/python_api.html#module-cudaq) | duct_opmlERK6sum_opI9HandlerTyE), |
| -   [cudaq (C++                   |     [\[9\]](api/languages/cpp_a   |
|     type)](api/lan                | pi.html#_CPPv4NKR5cudaq10product_ |
| guages/cpp_api.html#_CPPv45cudaq) | opmlERK10product_opI9HandlerTyE), |
| -   [cudaq.apply_noise() (in      |     [\[10\]](api/language         |
|     module                        | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     cudaq)](api/languages/python_ | roduct_opmlERK15scalar_operator), |
| api.html#cudaq.cudaq.apply_noise) |     [\[11\]](api/languages/cpp_a  |
| -   cudaq.boson                   | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [module](api/languages/py | opmlERR10product_opI9HandlerTyE), |
| thon_api.html#module-cudaq.boson) |     [\[12\]](api/language         |
| -   cudaq.fermion                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opmlERR15scalar_operator), |
|   -   [module](api/languages/pyth |     [\[13\]](api/languages/cpp_   |
| on_api.html#module-cudaq.fermion) | api.html#_CPPv4NO5cudaq10product_ |
| -   cudaq.operators.custom        | opmlERK10product_opI9HandlerTyE), |
|     -   [mo                       |     [\[14\]](api/languag          |
| dule](api/languages/python_api.ht | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ml#module-cudaq.operators.custom) | roduct_opmlERK15scalar_operator), |
| -   cudaq.spin                    |     [\[15\]](api/languages/cpp_   |
|     -   [module](api/languages/p  | api.html#_CPPv4NO5cudaq10product_ |
| ython_api.html#module-cudaq.spin) | opmlERR10product_opI9HandlerTyE), |
| -   [cudaq::amplitude_damping     |     [\[16\]](api/langua           |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     cla                           | product_opmlERR15scalar_operator) |
| ss)](api/languages/cpp_api.html#_ | -                                 |
| CPPv4N5cudaq17amplitude_dampingE) |   [cudaq::product_op::operator\*= |
| -                                 |     (C++                          |
| [cudaq::amplitude_damping_channel |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4N5cudaq10product_ |
|     class)](api                   | opmLERK10product_opI9HandlerTyE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[1\]](api/langua            |
| udaq25amplitude_damping_channelE) | ges/cpp_api.html#_CPPv4N5cudaq10p |
| -   [cudaq::amplitud              | roduct_opmLERK15scalar_operator), |
| e_damping_channel::num_parameters |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     member)](api/languages/cpp_a  | _opmLERR10product_opI9HandlerTyE) |
| pi.html#_CPPv4N5cudaq25amplitude_ | -   [cudaq::product_op::operator+ |
| damping_channel14num_parametersE) |     (C++                          |
| -   [cudaq::ampli                 |     function)](api/langu          |
| tude_damping_channel::num_targets | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opplE6sum_opI1TERK15sc |
|     member)](api/languages/cp     | alar_operatorRK10product_opI1TE), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [\[1\]](api/                  |
| de_damping_channel11num_targetsE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::AnalogRemoteRESTQPU   | 5cudaq10product_opplE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRK6sum_opI1TE), |
|     class                         |     [\[2\]](api/langu             |
| )](api/languages/cpp_api.html#_CP | ages/cpp_api.html#_CPPv4I0EN5cuda |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | q10product_opplE6sum_opI1TERK15sc |
| -   [cudaq::apply_noise (C++      | alar_operatorRR10product_opI1TE), |
|     function)](api/               |     [\[3\]](api/                  |
| languages/cpp_api.html#_CPPv4I0Dp | languages/cpp_api.html#_CPPv4I0EN |
| EN5cudaq11apply_noiseEvDpRR4Args) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::async_result (C++     | K15scalar_operatorRR6sum_opI1TE), |
|     c                             |     [\[4\]](api/langu             |
| lass)](api/languages/cpp_api.html | ages/cpp_api.html#_CPPv4I0EN5cuda |
| #_CPPv4I0EN5cudaq12async_resultE) | q10product_opplE6sum_opI1TERR15sc |
| -   [cudaq::async_result::get     | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[5\]](api/                  |
|     functi                        | languages/cpp_api.html#_CPPv4I0EN |
| on)](api/languages/cpp_api.html#_ | 5cudaq10product_opplE6sum_opI1TER |
| CPPv4N5cudaq12async_result3getEv) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::async_sample_result   |     [\[6\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     type                          | q10product_opplE6sum_opI1TERR15sc |
| )](api/languages/cpp_api.html#_CP | alar_operatorRR10product_opI1TE), |
| Pv4N5cudaq19async_sample_resultE) |     [\[7\]](api/                  |
| -   [cudaq::BaseRemoteRESTQPU     | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     cla                           | R15scalar_operatorRR6sum_opI1TE), |
| ss)](api/languages/cpp_api.html#_ |     [\[8\]](api/languages/cpp_a   |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | pi.html#_CPPv4NKR5cudaq10product_ |
| -                                 | opplERK10product_opI9HandlerTyE), |
|    [cudaq::BaseRemoteSimulatorQPU |     [\[9\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     class)](                      | roduct_opplERK15scalar_operator), |
| api/languages/cpp_api.html#_CPPv4 |     [\[10\]](api/languages/       |
| N5cudaq22BaseRemoteSimulatorQPUE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::bit_flip_channel (C++ | duct_opplERK6sum_opI9HandlerTyE), |
|     cl                            |     [\[11\]](api/languages/cpp_a  |
| ass)](api/languages/cpp_api.html# | pi.html#_CPPv4NKR5cudaq10product_ |
| _CPPv4N5cudaq16bit_flip_channelE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq:                       |     [\[12\]](api/language         |
| :bit_flip_channel::num_parameters | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opplERR15scalar_operator), |
|     member)](api/langua           |     [\[13\]](api/languages/       |
| ges/cpp_api.html#_CPPv4N5cudaq16b | cpp_api.html#_CPPv4NKR5cudaq10pro |
| it_flip_channel14num_parametersE) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cud                          |     [\[                           |
| aq::bit_flip_channel::num_targets | 14\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NKR5cudaq10product_opplEv), |
|     member)](api/lan              |     [\[15\]](api/languages/cpp_   |
| guages/cpp_api.html#_CPPv4N5cudaq | api.html#_CPPv4NO5cudaq10product_ |
| 16bit_flip_channel11num_targetsE) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq::boson_handler (C++    |     [\[16\]](api/languag          |
|                                   | es/cpp_api.html#_CPPv4NO5cudaq10p |
|  class)](api/languages/cpp_api.ht | roduct_opplERK15scalar_operator), |
| ml#_CPPv4N5cudaq13boson_handlerE) |     [\[17\]](api/languages        |
| -   [cudaq::boson_op (C++         | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     type)](api/languages/cpp_     | duct_opplERK6sum_opI9HandlerTyE), |
| api.html#_CPPv4N5cudaq8boson_opE) |     [\[18\]](api/languages/cpp_   |
| -   [cudaq::boson_op_term (C++    | api.html#_CPPv4NO5cudaq10product_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|   type)](api/languages/cpp_api.ht |     [\[19\]](api/languag          |
| ml#_CPPv4N5cudaq13boson_op_termE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::CodeGenConfig (C++    | roduct_opplERR15scalar_operator), |
|                                   |     [\[20\]](api/languages        |
| struct)](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cudaq::commutation_relations |     [                             |
|     (C++                          | \[21\]](api/languages/cpp_api.htm |
|     struct)]                      | l#_CPPv4NO5cudaq10product_opplEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::product_op::operator- |
| 4N5cudaq21commutation_relationsE) |     (C++                          |
| -   [cudaq::complex (C++          |     function)](api/langu          |
|     type)](api/languages/cpp      | ages/cpp_api.html#_CPPv4I0EN5cuda |
| _api.html#_CPPv4N5cudaq7complexE) | q10product_opmiE6sum_opI1TERK15sc |
| -   [cudaq::complex_matrix (C++   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]](api/                  |
| class)](api/languages/cpp_api.htm | languages/cpp_api.html#_CPPv4I0EN |
| l#_CPPv4N5cudaq14complex_matrixE) | 5cudaq10product_opmiE6sum_opI1TER |
| -                                 | K15scalar_operatorRK6sum_opI1TE), |
|   [cudaq::complex_matrix::adjoint |     [\[2\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     function)](a                  | q10product_opmiE6sum_opI1TERK15sc |
| pi/languages/cpp_api.html#_CPPv4N | alar_operatorRR10product_opI1TE), |
| 5cudaq14complex_matrix7adjointEv) |     [\[3\]](api/                  |
| -   [cudaq::                      | languages/cpp_api.html#_CPPv4I0EN |
| complex_matrix::diagonal_elements | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRR6sum_opI1TE), |
|     function)](api/languages      |     [\[4\]](api/langu             |
| /cpp_api.html#_CPPv4NK5cudaq14com | ages/cpp_api.html#_CPPv4I0EN5cuda |
| plex_matrix17diagonal_elementsEi) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq::complex_matrix::dump  | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[5\]](api/                  |
|     function)](api/language       | languages/cpp_api.html#_CPPv4I0EN |
| s/cpp_api.html#_CPPv4NK5cudaq14co | 5cudaq10product_opmiE6sum_opI1TER |
| mplex_matrix4dumpERNSt7ostreamE), | R15scalar_operatorRK6sum_opI1TE), |
|     [\[1\]]                       |     [\[6\]](api/langu             |
| (api/languages/cpp_api.html#_CPPv | ages/cpp_api.html#_CPPv4I0EN5cuda |
| 4NK5cudaq14complex_matrix4dumpEv) | q10product_opmiE6sum_opI1TERR15sc |
| -   [c                            | alar_operatorRR10product_opI1TE), |
| udaq::complex_matrix::eigenvalues |     [\[7\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/lan            | 5cudaq10product_opmiE6sum_opI1TER |
| guages/cpp_api.html#_CPPv4NK5cuda | R15scalar_operatorRR6sum_opI1TE), |
| q14complex_matrix11eigenvaluesEv) |     [\[8\]](api/languages/cpp_a   |
| -   [cu                           | pi.html#_CPPv4NKR5cudaq10product_ |
| daq::complex_matrix::eigenvectors | opmiERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[9\]](api/language          |
|     function)](api/lang           | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| uages/cpp_api.html#_CPPv4NK5cudaq | roduct_opmiERK15scalar_operator), |
| 14complex_matrix12eigenvectorsEv) |     [\[10\]](api/languages/       |
| -   [c                            | cpp_api.html#_CPPv4NKR5cudaq10pro |
| udaq::complex_matrix::exponential | duct_opmiERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     function)](api/la             | pi.html#_CPPv4NKR5cudaq10product_ |
| nguages/cpp_api.html#_CPPv4N5cuda | opmiERR10product_opI9HandlerTyE), |
| q14complex_matrix11exponentialEv) |     [\[12\]](api/language         |
| -                                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|  [cudaq::complex_matrix::identity | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     function)](api/languages      | cpp_api.html#_CPPv4NKR5cudaq10pro |
| /cpp_api.html#_CPPv4N5cudaq14comp | duct_opmiERR6sum_opI9HandlerTyE), |
| lex_matrix8identityEKNSt6size_tE) |     [\[                           |
| -                                 | 14\]](api/languages/cpp_api.html# |
| [cudaq::complex_matrix::kronecker | _CPPv4NKR5cudaq10product_opmiEv), |
|     (C++                          |     [\[15\]](api/languages/cpp_   |
|     function)](api/lang           | api.html#_CPPv4NO5cudaq10product_ |
| uages/cpp_api.html#_CPPv4I00EN5cu | opmiERK10product_opI9HandlerTyE), |
| daq14complex_matrix9kroneckerE14c |     [\[16\]](api/languag          |
| omplex_matrix8Iterable8Iterable), | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     [\[1\]](api/l                 | roduct_opmiERK15scalar_operator), |
| anguages/cpp_api.html#_CPPv4N5cud |     [\[17\]](api/languages        |
| aq14complex_matrix9kroneckerERK14 | /cpp_api.html#_CPPv4NO5cudaq10pro |
| complex_matrixRK14complex_matrix) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::c                     |     [\[18\]](api/languages/cpp_   |
| omplex_matrix::minimal_eigenvalue | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmiERR10product_opI9HandlerTyE), |
|     function)](api/languages/     |     [\[19\]](api/languag          |
| cpp_api.html#_CPPv4NK5cudaq14comp | es/cpp_api.html#_CPPv4NO5cudaq10p |
| lex_matrix18minimal_eigenvalueEv) | roduct_opmiERR15scalar_operator), |
| -   [                             |     [\[20\]](api/languages        |
| cudaq::complex_matrix::operator() | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERR6sum_opI9HandlerTyE), |
|     function)](api/languages/cpp  |     [                             |
| _api.html#_CPPv4N5cudaq14complex_ | \[21\]](api/languages/cpp_api.htm |
| matrixclENSt6size_tENSt6size_tE), | l#_CPPv4NO5cudaq10product_opmiEv) |
|     [\[1\]](api/languages/cpp     | -   [cudaq::product_op::operator/ |
| _api.html#_CPPv4NK5cudaq14complex |     (C++                          |
| _matrixclENSt6size_tENSt6size_tE) |     function)](api/language       |
| -   [                             | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| cudaq::complex_matrix::operator\* | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/language          |
|     function)](api/langua         | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ges/cpp_api.html#_CPPv4N5cudaq14c | roduct_opdvERR15scalar_operator), |
| omplex_matrixmlEN14complex_matrix |     [\[2\]](api/languag           |
| 10value_typeERK14complex_matrix), | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     [\[1\]                        | roduct_opdvERK15scalar_operator), |
| ](api/languages/cpp_api.html#_CPP |     [\[3\]](api/langua            |
| v4N5cudaq14complex_matrixmlERK14c | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| omplex_matrixRK14complex_matrix), | product_opdvERR15scalar_operator) |
|                                   | -                                 |
|  [\[2\]](api/languages/cpp_api.ht |    [cudaq::product_op::operator/= |
| ml#_CPPv4N5cudaq14complex_matrixm |     (C++                          |
| lERK14complex_matrixRKNSt6vectorI |     function)](api/langu          |
| N14complex_matrix10value_typeEEE) | ages/cpp_api.html#_CPPv4N5cudaq10 |
| -                                 | product_opdVERK15scalar_operator) |
| [cudaq::complex_matrix::operator+ | -   [cudaq::product_op::operator= |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/la             |
| )](api/languages/cpp_api.html#_CP | nguages/cpp_api.html#_CPPv4I0_NSt |
| Pv4N5cudaq14complex_matrixplERK14 | 11enable_if_tIXaantNSt7is_sameI1T |
| complex_matrixRK14complex_matrix) | 9HandlerTyE5valueENSt16is_constru |
| -                                 | ctibleI9HandlerTy1TE5valueEEbEEEN |
| [cudaq::complex_matrix::operator- | 5cudaq10product_opaSER10product_o |
|     (C++                          | pI9HandlerTyERK10product_opI1TE), |
|     function                      |     [\[1\]](api/languages/cpp     |
| )](api/languages/cpp_api.html#_CP | _api.html#_CPPv4N5cudaq10product_ |
| Pv4N5cudaq14complex_matrixmiERK14 | opaSERK10product_opI9HandlerTyE), |
| complex_matrixRK14complex_matrix) |     [\[2\]](api/languages/cp      |
| -   [cu                           | p_api.html#_CPPv4N5cudaq10product |
| daq::complex_matrix::operator\[\] | _opaSERR10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|                                   |    [cudaq::product_op::operator== |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4N5cudaq14complex_matr |     function)](api/languages/cpp  |
| ixixERKNSt6vectorINSt6size_tEEE), | _api.html#_CPPv4NK5cudaq10product |
|     [\[1\]](api/languages/cpp_api | _opeqERK10product_opI9HandlerTyE) |
| .html#_CPPv4NK5cudaq14complex_mat | -                                 |
| rixixERKNSt6vectorINSt6size_tEEE) |  [cudaq::product_op::operator\[\] |
| -   [cudaq::complex_matrix::power |     (C++                          |
|     (C++                          |     function)](ap                 |
|     function)]                    | i/languages/cpp_api.html#_CPPv4NK |
| (api/languages/cpp_api.html#_CPPv | 5cudaq10product_opixENSt6size_tE) |
| 4N5cudaq14complex_matrix5powerEi) | -                                 |
| -                                 |    [cudaq::product_op::product_op |
|  [cudaq::complex_matrix::set_zero |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)](ap                 | pp_api.html#_CPPv4I0_NSt11enable_ |
| i/languages/cpp_api.html#_CPPv4N5 | if_tIXaaNSt7is_sameI9HandlerTy14m |
| cudaq14complex_matrix8set_zeroEv) | atrix_handlerE5valueEaantNSt7is_s |
| -                                 | ameI1T9HandlerTyE5valueENSt16is_c |
| [cudaq::complex_matrix::to_string | onstructibleI9HandlerTy1TE5valueE |
|     (C++                          | EbEEEN5cudaq10product_op10product |
|     function)](api/               | _opERK10product_opI1TERKN14matrix |
| languages/cpp_api.html#_CPPv4NK5c | _handler20commutation_behaviorE), |
| udaq14complex_matrix9to_stringEv) |                                   |
| -   [                             |  [\[1\]](api/languages/cpp_api.ht |
| cudaq::complex_matrix::value_type | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|     (C++                          | tNSt7is_sameI1T9HandlerTyE5valueE |
|     type)](api/                   | NSt16is_constructibleI9HandlerTy1 |
| languages/cpp_api.html#_CPPv4N5cu | TE5valueEEbEEEN5cudaq10product_op |
| daq14complex_matrix10value_typeE) | 10product_opERK10product_opI1TE), |
| -   [cudaq::contrib (C++          |                                   |
|     type)](api/languages/cpp      |   [\[2\]](api/languages/cpp_api.h |
| _api.html#_CPPv4N5cudaq7contribE) | tml#_CPPv4N5cudaq10product_op10pr |
| -   [cudaq::contrib::draw (C++    | oduct_opENSt6size_tENSt6size_tE), |
|     function)                     |     [\[3\]](api/languages/cp      |
| ](api/languages/cpp_api.html#_CPP | p_api.html#_CPPv4N5cudaq10product |
| v4I0DpEN5cudaq7contrib4drawENSt6s | _op10product_opENSt7complexIdEE), |
| tringERR13QuantumKernelDpRR4Args) |     [\[4\]](api/l                 |
| -                                 | anguages/cpp_api.html#_CPPv4N5cud |
| [cudaq::contrib::get_unitary_cmat | aq10product_op10product_opERK10pr |
|     (C++                          | oduct_opI9HandlerTyENSt6size_tE), |
|     function)](api/languages/cp   |     [\[5\]](api/l                 |
| p_api.html#_CPPv4I0DpEN5cudaq7con | anguages/cpp_api.html#_CPPv4N5cud |
| trib16get_unitary_cmatE14complex_ | aq10product_op10product_opERR10pr |
| matrixRR13QuantumKernelDpRR4Args) | oduct_opI9HandlerTyENSt6size_tE), |
| -   [cudaq::CusvState (C++        |     [\[6\]](api/languages         |
|                                   | /cpp_api.html#_CPPv4N5cudaq10prod |
|    class)](api/languages/cpp_api. | uct_op10product_opERR9HandlerTy), |
| html#_CPPv4I0EN5cudaq9CusvStateE) |     [\[7\]](ap                    |
| -   [cudaq::depolarization1 (C++  | i/languages/cpp_api.html#_CPPv4N5 |
|     c                             | cudaq10product_op10product_opEd), |
| lass)](api/languages/cpp_api.html |     [\[8\]](a                     |
| #_CPPv4N5cudaq15depolarization1E) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::depolarization2 (C++  | 5cudaq10product_op10product_opEv) |
|     c                             | -   [cuda                         |
| lass)](api/languages/cpp_api.html | q::product_op::to_diagonal_matrix |
| #_CPPv4N5cudaq15depolarization2E) |     (C++                          |
| -   [cudaq:                       |     function)](api/               |
| :depolarization2::depolarization2 | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq10product_op18to_diagonal_mat |
|     function)](api/languages/cp   | rixENSt13unordered_mapINSt6size_t |
| p_api.html#_CPPv4N5cudaq15depolar | ENSt7int64_tEEERKNSt13unordered_m |
| ization215depolarization2EK4real) | apINSt6stringENSt7complexIdEEEEb) |
| -   [cudaq                        | -   [cudaq::product_op::to_matrix |
| ::depolarization2::num_parameters |     (C++                          |
|     (C++                          |     funct                         |
|     member)](api/langu            | ion)](api/languages/cpp_api.html# |
| ages/cpp_api.html#_CPPv4N5cudaq15 | _CPPv4NK5cudaq10product_op9to_mat |
| depolarization214num_parametersE) | rixENSt13unordered_mapINSt6size_t |
| -   [cu                           | ENSt7int64_tEEERKNSt13unordered_m |
| daq::depolarization2::num_targets | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cu                           |
|     member)](api/la               | daq::product_op::to_sparse_matrix |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q15depolarization211num_targetsE) |     function)](ap                 |
| -                                 | i/languages/cpp_api.html#_CPPv4NK |
|    [cudaq::depolarization_channel | 5cudaq10product_op16to_sparse_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     class)](                      | ENSt7int64_tEEERKNSt13unordered_m |
| api/languages/cpp_api.html#_CPPv4 | apINSt6stringENSt7complexIdEEEEb) |
| N5cudaq22depolarization_channelE) | -   [cudaq::product_op::to_string |
| -   [cudaq::depol                 |     (C++                          |
| arization_channel::num_parameters |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     member)](api/languages/cp     | NK5cudaq10product_op9to_stringEv) |
| p_api.html#_CPPv4N5cudaq22depolar | -                                 |
| ization_channel14num_parametersE) |  [cudaq::product_op::\~product_op |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     member)](api/languages        | ml#_CPPv4N5cudaq10product_opD0Ev) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -   [cudaq::QPU (C++              |
| larization_channel11num_targetsE) |     class)](api/languages         |
| -   [cudaq::details (C++          | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     type)](api/languages/cpp      | -   [cudaq::QPU::beginExecution   |
| _api.html#_CPPv4N5cudaq7detailsE) |     (C++                          |
| -   [cudaq::details::future (C++  |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|  class)](api/languages/cpp_api.ht | Pv4N5cudaq3QPU14beginExecutionEv) |
| ml#_CPPv4N5cudaq7details6futureE) | -   [cuda                         |
| -                                 | q::QPU::configureExecutionContext |
|   [cudaq::details::future::future |     (C++                          |
|     (C++                          |     funct                         |
|     functio                       | ion)](api/languages/cpp_api.html# |
| n)](api/languages/cpp_api.html#_C | _CPPv4NK5cudaq3QPU25configureExec |
| PPv4N5cudaq7details6future6future | utionContextER16ExecutionContext) |
| ERNSt6vectorI3JobEERNSt6stringERN | -   [cudaq::QPU::endExecution     |
| St3mapINSt6stringENSt6stringEEE), |     (C++                          |
|     [\[1\]](api/lang              |     functi                        |
| uages/cpp_api.html#_CPPv4N5cudaq7 | on)](api/languages/cpp_api.html#_ |
| details6future6futureERR6future), | CPPv4N5cudaq3QPU12endExecutionEv) |
|     [\[2\]]                       | -   [cudaq::QPU::enqueue (C++     |
| (api/languages/cpp_api.html#_CPPv |     function)](ap                 |
| 4N5cudaq7details6future6futureEv) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [cu                           | cudaq3QPU7enqueueER11QuantumTask) |
| daq::details::kernel_builder_base | -   [cud                          |
|     (C++                          | aq::QPU::finalizeExecutionContext |
|     class)](api/l                 |     (C++                          |
| anguages/cpp_api.html#_CPPv4N5cud |     func                          |
| aq7details19kernel_builder_baseE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::details::             | #_CPPv4NK5cudaq3QPU24finalizeExec |
| kernel_builder_base::operator\<\< | utionContextER16ExecutionContext) |
|     (C++                          | -   [cudaq::QPU::getConnectivity  |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq7de |     function)                     |
| tails19kernel_builder_baselsERNSt | ](api/languages/cpp_api.html#_CPP |
| 7ostreamERK19kernel_builder_base) | v4N5cudaq3QPU15getConnectivityEv) |
| -   [                             | -                                 |
| cudaq::details::KernelBuilderType | [cudaq::QPU::getExecutionThreadId |
|     (C++                          |     (C++                          |
|     class)](api                   |     function)](api/               |
| /languages/cpp_api.html#_CPPv4N5c | languages/cpp_api.html#_CPPv4NK5c |
| udaq7details17KernelBuilderTypeE) | udaq3QPU20getExecutionThreadIdEv) |
| -   [cudaq::d                     | -   [cudaq::QPU::getNumQubits     |
| etails::KernelBuilderType::create |     (C++                          |
|     (C++                          |     functi                        |
|     function)                     | on)](api/languages/cpp_api.html#_ |
| ](api/languages/cpp_api.html#_CPP | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| v4N5cudaq7details17KernelBuilderT | -   [                             |
| ype6createEPN4mlir11MLIRContextE) | cudaq::QPU::getRemoteCapabilities |
| -   [cudaq::details::Ker          |     (C++                          |
| nelBuilderType::KernelBuilderType |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     function)](api/lang           | daq3QPU21getRemoteCapabilitiesEv) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -   [cudaq::QPU::isEmulated (C++  |
| details17KernelBuilderType17Kerne |     func                          |
| lBuilderTypeERRNSt8functionIFN4ml | tion)](api/languages/cpp_api.html |
| ir4TypeEPN4mlir11MLIRContextEEEE) | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| -   [cudaq::diag_matrix_callback  | -   [cudaq::QPU::isSimulator (C++ |
|     (C++                          |     funct                         |
|     class)                        | ion)](api/languages/cpp_api.html# |
| ](api/languages/cpp_api.html#_CPP | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| v4N5cudaq20diag_matrix_callbackE) | -   [cudaq::QPU::launchKernel     |
| -   [cudaq::dyn (C++              |     (C++                          |
|     member)](api/languages        |     function)](api/               |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::ExecutionContext (C++ | daq3QPU12launchKernelERKNSt6strin |
|     cl                            | gE15KernelThunkTypePvNSt8uint64_t |
| ass)](api/languages/cpp_api.html# | ENSt8uint64_tERKNSt6vectorIPvEE), |
| _CPPv4N5cudaq16ExecutionContextE) |                                   |
| -   [cudaq                        |  [\[1\]](api/languages/cpp_api.ht |
| ::ExecutionContext::amplitudeMaps | ml#_CPPv4N5cudaq3QPU12launchKerne |
|     (C++                          | lERKNSt6stringERKNSt6vectorIPvEE) |
|     member)](api/langu            | -   [cudaq::QPU::onRandomSeedSet  |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13amplitudeMapsE) |     function)](api/lang           |
| -   [c                            | uages/cpp_api.html#_CPPv4N5cudaq3 |
| udaq::ExecutionContext::asyncExec | QPU15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::QPU (C++         |
|     member)](api/                 |     functio                       |
| languages/cpp_api.html#_CPPv4N5cu | n)](api/languages/cpp_api.html#_C |
| daq16ExecutionContext9asyncExecE) | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| -   [cud                          |                                   |
| aq::ExecutionContext::asyncResult |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     member)](api/lan              |     [\[2\]](api/languages/cpp_    |
| guages/cpp_api.html#_CPPv4N5cudaq | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| 16ExecutionContext11asyncResultE) | -   [cudaq::QPU::setId (C++       |
| -   [cudaq:                       |     function                      |
| :ExecutionContext::batchIteration | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     member)](api/langua           | -   [cudaq::QPU::setShots (C++    |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     f                             |
| xecutionContext14batchIterationE) | unction)](api/languages/cpp_api.h |
| -   [cudaq::E                     | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| xecutionContext::canHandleObserve | -   [cudaq:                       |
|     (C++                          | :QPU::supportsConditionalFeedback |
|     member)](api/language         |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     function)](api/langua         |
| cutionContext16canHandleObserveE) | ges/cpp_api.html#_CPPv4N5cudaq3QP |
| -   [cudaq::E                     | U27supportsConditionalFeedbackEv) |
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
|     (C++                          | -   [cudaq::quantum_platfo        |
|     function)](api/languages      | rm::supports_conditional_feedback |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18forward_difference5cloneEv) |     function)](api/               |
| -   [cudaq::gradi                 | languages/cpp_api.html#_CPPv4NK5c |
| ents::forward_difference::compute | udaq16quantum_platform29supports_ |
|     (C++                          | conditional_feedbackENSt6size_tE) |
|     function)](                   | -   [cudaq::quantum_platfor       |
| api/languages/cpp_api.html#_CPPv4 | m::supports_explicit_measurements |
| N5cudaq9gradients18forward_differ |     (C++                          |
| ence7computeERKNSt6vectorIdEERKNS |     function)](api/l              |
| t8functionIFdNSt6vectorIdEEEEEd), | anguages/cpp_api.html#_CPPv4NK5cu |
|                                   | daq16quantum_platform30supports_e |
|   [\[1\]](api/languages/cpp_api.h | xplicit_measurementsENSt6size_tE) |
| tml#_CPPv4N5cudaq9gradients18forw | -   [cudaq::quantum_pla           |
| ard_difference7computeERKNSt6vect | tform::supports_task_distribution |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradie                |     fu                            |
| nts::forward_difference::gradient | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq16quantum_platfo |
|     functio                       | rm26supports_task_distributionEv) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::quantum               |
| PPv4I00EN5cudaq9gradients18forwar | _platform::with_execution_context |
| d_difference8gradientER7KernelT), |     (C++                          |
|     [\[1\]](api/langua            |     function)                     |
| ges/cpp_api.html#_CPPv4I00EN5cuda | ](api/languages/cpp_api.html#_CPP |
| q9gradients18forward_difference8g | v4I0DpEN5cudaq16quantum_platform2 |
| radientER7KernelTRR10ArgsMapper), | 2with_execution_contextEDaR16Exec |
|     [\[2\]](api/languages/cpp_    | utionContextRR8CallableDpRR4Args) |
| api.html#_CPPv4I00EN5cudaq9gradie | -   [cudaq::QuantumTask (C++      |
| nts18forward_difference8gradientE |     type)](api/languages/cpp_api. |
| RR13QuantumKernelRR10ArgsMapper), | html#_CPPv4N5cudaq11QuantumTaskE) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::qubit (C++            |
| _api.html#_CPPv4N5cudaq9gradients |     type)](api/languages/c        |
| 18forward_difference8gradientERRN | pp_api.html#_CPPv4N5cudaq5qubitE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QubitConnectivity     |
|     [\[4\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq9gradient |     ty                            |
| s18forward_difference8gradientEv) | pe)](api/languages/cpp_api.html#_ |
| -   [                             | CPPv4N5cudaq17QubitConnectivityE) |
| cudaq::gradients::parameter_shift | -   [cudaq::QubitEdge (C++        |
|     (C++                          |     type)](api/languages/cpp_a    |
|     class)](api                   | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::qudit (C++            |
| udaq9gradients15parameter_shiftE) |     clas                          |
| -   [cudaq::                      | s)](api/languages/cpp_api.html#_C |
| gradients::parameter_shift::clone | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     (C++                          | -   [cudaq::qudit::qudit (C++     |
|     function)](api/langua         |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | function)](api/languages/cpp_api. |
| adients15parameter_shift5cloneEv) | html#_CPPv4N5cudaq5qudit5quditEv) |
| -   [cudaq::gr                    | -   [cudaq::qvector (C++          |
| adients::parameter_shift::compute |     class)                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function                      | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qvector::back (C++    |
| Pv4N5cudaq9gradients15parameter_s |     function)](a                  |
| hift7computeERKNSt6vectorIdEERKNS | pi/languages/cpp_api.html#_CPPv4N |
| t8functionIFdNSt6vectorIdEEEEEd), | 5cudaq7qvector4backENSt6size_tE), |
|     [\[1\]](api/languages/cpp_ap  |                                   |
| i.html#_CPPv4N5cudaq9gradients15p |   [\[1\]](api/languages/cpp_api.h |
| arameter_shift7computeERKNSt6vect | tml#_CPPv4N5cudaq7qvector4backEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::qvector::begin (C++   |
| -   [cudaq::gra                   |     fu                            |
| dients::parameter_shift::gradient | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5beginEv) |
|     func                          | -   [cudaq::qvector::clear (C++   |
| tion)](api/languages/cpp_api.html |     fu                            |
| #_CPPv4I00EN5cudaq9gradients15par | nction)](api/languages/cpp_api.ht |
| ameter_shift8gradientER7KernelT), | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     [\[1\]](api/lan               | -   [cudaq::qvector::end (C++     |
| guages/cpp_api.html#_CPPv4I00EN5c |                                   |
| udaq9gradients15parameter_shift8g | function)](api/languages/cpp_api. |
| radientER7KernelTRR10ArgsMapper), | html#_CPPv4N5cudaq7qvector3endEv) |
|     [\[2\]](api/languages/c       | -   [cudaq::qvector::front (C++   |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     function)](ap                 |
| dients15parameter_shift8gradientE | i/languages/cpp_api.html#_CPPv4N5 |
| RR13QuantumKernelRR10ArgsMapper), | cudaq7qvector5frontENSt6size_tE), |
|     [\[3\]](api/languages/        |                                   |
| cpp_api.html#_CPPv4N5cudaq9gradie |  [\[1\]](api/languages/cpp_api.ht |
| nts15parameter_shift8gradientERRN | ml#_CPPv4N5cudaq7qvector5frontEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::qvector::operator=    |
|     [\[4\]](api/languages         |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     functio                       |
| ents15parameter_shift8gradientEv) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::kernel_builder (C++   | PPv4N5cudaq7qvectoraSERK7qvector) |
|     clas                          | -   [cudaq::qvector::operator\[\] |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4IDpEN5cudaq14kernel_builderE) |     function)                     |
| -   [c                            | ](api/languages/cpp_api.html#_CPP |
| udaq::kernel_builder::constantVal | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qvector::qvector (C++ |
|     function)](api/la             |     function)](api/               |
| nguages/cpp_api.html#_CPPv4N5cuda | languages/cpp_api.html#_CPPv4N5cu |
| q14kernel_builder11constantValEd) | daq7qvector7qvectorENSt6size_tE), |
| -   [cu                           |     [\[1\]](a                     |
| daq::kernel_builder::getArguments | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq7qvector7qvectorERK5state), |
|     function)](api/lan            |     [\[2\]](api                   |
| guages/cpp_api.html#_CPPv4N5cudaq | /languages/cpp_api.html#_CPPv4N5c |
| 14kernel_builder12getArgumentsEv) | udaq7qvector7qvectorERK7qvector), |
| -   [cu                           |     [\[3\]](api/languages/cpp     |
| daq::kernel_builder::getNumParams | _api.html#_CPPv4N5cudaq7qvector7q |
|     (C++                          | vectorERKNSt6vectorI7complexEEb), |
|     function)](api/lan            |     [\[4\]](ap                    |
| guages/cpp_api.html#_CPPv4N5cudaq | i/languages/cpp_api.html#_CPPv4N5 |
| 14kernel_builder12getNumParamsEv) | cudaq7qvector7qvectorERR7qvector) |
| -   [c                            | -   [cudaq::qvector::size (C++    |
| udaq::kernel_builder::isArgStdVec |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function)](api/languages/cp   | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| p_api.html#_CPPv4N5cudaq14kernel_ | -   [cudaq::qvector::slice (C++   |
| builder11isArgStdVecENSt6size_tE) |     function)](api/language       |
| -   [cuda                         | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| q::kernel_builder::kernel_builder | tor5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qvector::value_type   |
|     function)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq14kernel_bu |     typ                           |
| ilder14kernel_builderERNSt6vector | e)](api/languages/cpp_api.html#_C |
| IN7details17KernelBuilderTypeEEE) | PPv4N5cudaq7qvector10value_typeE) |
| -   [cudaq::kernel_builder::name  | -   [cudaq::qview (C++            |
|     (C++                          |     clas                          |
|     function)                     | s)](api/languages/cpp_api.html#_C |
| ](api/languages/cpp_api.html#_CPP | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| v4N5cudaq14kernel_builder4nameEv) | -   [cudaq::qview::back (C++      |
| -                                 |     function)                     |
|    [cudaq::kernel_builder::qalloc | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq5qview4backENSt6size_tE) |
|     function)](api/language       | -   [cudaq::qview::begin (C++     |
| s/cpp_api.html#_CPPv4N5cudaq14ker |                                   |
| nel_builder6qallocE10QuakeValue), | function)](api/languages/cpp_api. |
|     [\[1\]](api/language          | html#_CPPv4N5cudaq5qview5beginEv) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::qview::end (C++       |
| nel_builder6qallocEKNSt6size_tE), |                                   |
|     [\[2                          |   function)](api/languages/cpp_ap |
| \]](api/languages/cpp_api.html#_C | i.html#_CPPv4N5cudaq5qview3endEv) |
| PPv4N5cudaq14kernel_builder6qallo | -   [cudaq::qview::front (C++     |
| cERNSt6vectorINSt7complexIdEEEE), |     function)](                   |
|     [\[3\]](                      | api/languages/cpp_api.html#_CPPv4 |
| api/languages/cpp_api.html#_CPPv4 | N5cudaq5qview5frontENSt6size_tE), |
| N5cudaq14kernel_builder6qallocEv) |                                   |
| -   [cudaq::kernel_builder::swap  |    [\[1\]](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qview5frontEv) |
|     function)](api/language       | -   [cudaq::qview::operator\[\]   |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     (C++                          |
| 4kernel_builder4swapEvRK10QuakeVa |     functio                       |
| lueRK10QuakeValueRK10QuakeValue), | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| [\[1\]](api/languages/cpp_api.htm | -   [cudaq::qview::qview (C++     |
| l#_CPPv4I00EN5cudaq14kernel_build |     functio                       |
| er4swapEvRKNSt6vectorI10QuakeValu | n)](api/languages/cpp_api.html#_C |
| eEERK10QuakeValueRK10QuakeValue), | PPv4I0EN5cudaq5qview5qviewERR1R), |
|                                   |     [\[1                          |
| [\[2\]](api/languages/cpp_api.htm | \]](api/languages/cpp_api.html#_C |
| l#_CPPv4N5cudaq14kernel_builder4s | PPv4N5cudaq5qview5qviewERK5qview) |
| wapERK10QuakeValueRK10QuakeValue) | -   [cudaq::qview::size (C++      |
| -   [cudaq::KernelExecutionTask   |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     type                          | html#_CPPv4NK5cudaq5qview4sizeEv) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qview::slice (C++     |
| Pv4N5cudaq19KernelExecutionTaskE) |     function)](api/langua         |
| -   [cudaq::KernelThunkResultType | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|     (C++                          | iew5sliceENSt6size_tENSt6size_tE) |
|     struct)]                      | -   [cudaq::qview::value_type     |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq21KernelThunkResultTypeE) |     t                             |
| -   [cudaq::KernelThunkType (C++  | ype)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq5qview10value_typeE) |
| type)](api/languages/cpp_api.html | -   [cudaq::range (C++            |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     fun                           |
| -   [cudaq::kraus_channel (C++    | ction)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
|  class)](api/languages/cpp_api.ht | orI11ElementTypeEE11ElementType), |
| ml#_CPPv4N5cudaq13kraus_channelE) |     [\[1\]](api/languages/cpp_    |
| -   [cudaq::kraus_channel::empty  | api.html#_CPPv4I0EN5cudaq5rangeEN |
|     (C++                          | St6vectorI11ElementTypeEE11Elemen |
|     function)]                    | tType11ElementType11ElementType), |
| (api/languages/cpp_api.html#_CPPv |     [                             |
| 4NK5cudaq13kraus_channel5emptyEv) | \[2\]](api/languages/cpp_api.html |
| -   [cudaq::kraus_c               | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| hannel::generateUnitaryParameters | -   [cudaq::real (C++             |
|     (C++                          |     type)](api/languages/         |
|                                   | cpp_api.html#_CPPv4N5cudaq4realE) |
|    function)](api/languages/cpp_a | -   [cudaq::registry (C++         |
| pi.html#_CPPv4N5cudaq13kraus_chan |     type)](api/languages/cpp_     |
| nel25generateUnitaryParametersEv) | api.html#_CPPv4N5cudaq8registryE) |
| -                                 | -                                 |
|    [cudaq::kraus_channel::get_ops |  [cudaq::registry::RegisteredType |
|     (C++                          |     (C++                          |
|     function)](a                  |     class)](api/                  |
| pi/languages/cpp_api.html#_CPPv4N | languages/cpp_api.html#_CPPv4I0EN |
| K5cudaq13kraus_channel7get_opsEv) | 5cudaq8registry14RegisteredTypeE) |
| -   [cudaq::                      | -   [cudaq::RemoteCapabilities    |
| kraus_channel::is_unitary_mixture |     (C++                          |
|     (C++                          |     struc                         |
|     function)](api/languages      | t)](api/languages/cpp_api.html#_C |
| /cpp_api.html#_CPPv4NK5cudaq13kra | PPv4N5cudaq18RemoteCapabilitiesE) |
| us_channel18is_unitary_mixtureEv) | -   [cudaq::Remo                  |
| -   [cu                           | teCapabilities::isRemoteSimulator |
| daq::kraus_channel::kraus_channel |     (C++                          |
|     (C++                          |     member)](api/languages/c      |
|     function)](api/lang           | pp_api.html#_CPPv4N5cudaq18Remote |
| uages/cpp_api.html#_CPPv4IDpEN5cu | Capabilities17isRemoteSimulatorE) |
| daq13kraus_channel13kraus_channel | -   [cudaq::Remot                 |
| EDpRRNSt16initializer_listI1TEE), | eCapabilities::RemoteCapabilities |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     function)](api/languages/cpp  |
| ml#_CPPv4N5cudaq13kraus_channel13 | _api.html#_CPPv4N5cudaq18RemoteCa |
| kraus_channelERK13kraus_channel), | pabilities18RemoteCapabilitiesEb) |
|     [\[2\]                        | -   [cudaq:                       |
| ](api/languages/cpp_api.html#_CPP | :RemoteCapabilities::stateOverlap |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERKNSt6vectorI8kraus_opEE), |     member)](api/langua           |
|     [\[3\]                        | ges/cpp_api.html#_CPPv4N5cudaq18R |
| ](api/languages/cpp_api.html#_CPP | emoteCapabilities12stateOverlapE) |
| v4N5cudaq13kraus_channel13kraus_c | -                                 |
| hannelERRNSt6vectorI8kraus_opEE), |   [cudaq::RemoteCapabilities::vqe |
|     [\[4\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     member)](                     |
| 13kraus_channel13kraus_channelEv) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | N5cudaq18RemoteCapabilities3vqeE) |
| [cudaq::kraus_channel::noise_type | -   [cudaq::RemoteSimulationState |
|     (C++                          |     (C++                          |
|     member)](api                  |     class)]                       |
| /languages/cpp_api.html#_CPPv4N5c | (api/languages/cpp_api.html#_CPPv |
| udaq13kraus_channel10noise_typeE) | 4N5cudaq21RemoteSimulationStateE) |
| -                                 | -   [cudaq::Resources (C++        |
|   [cudaq::kraus_channel::op_names |     class)](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     member)](                     | -   [cudaq::run (C++              |
| api/languages/cpp_api.html#_CPPv4 |     function)]                    |
| N5cudaq13kraus_channel8op_namesE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
|  [cudaq::kraus_channel::operator= | 5invoke_result_tINSt7decay_tI13Qu |
|     (C++                          | antumKernelEEDpNSt7decay_tI4ARGSE |
|     function)](api/langua         | EEEEENSt6size_tERN5cudaq11noise_m |
| ges/cpp_api.html#_CPPv4N5cudaq13k | odelERR13QuantumKernelDpRR4ARGS), |
| raus_channelaSERK13kraus_channel) |     [\[1\]](api/langu             |
| -   [c                            | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| udaq::kraus_channel::operator\[\] | daq3runENSt6vectorINSt15invoke_re |
|     (C++                          | sult_tINSt7decay_tI13QuantumKerne |
|     function)](api/l              | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| anguages/cpp_api.html#_CPPv4N5cud | ize_tERR13QuantumKernelDpRR4ARGS) |
| aq13kraus_channelixEKNSt6size_tE) | -   [cudaq::run_async (C++        |
| -                                 |     functio                       |
| [cudaq::kraus_channel::parameters | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     member)](api                  | tureINSt6vectorINSt15invoke_resul |
| /languages/cpp_api.html#_CPPv4N5c | t_tINSt7decay_tI13QuantumKernelEE |
| udaq13kraus_channel10parametersE) | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| -   [cudaq::krau                  | ze_tENSt6size_tERN5cudaq11noise_m |
| s_channel::populateDefaultOpNames | odelERR13QuantumKernelDpRR4ARGS), |
|     (C++                          |     [\[1\]](api/la                |
|     function)](api/languages/cp   | nguages/cpp_api.html#_CPPv4I0DpEN |
| p_api.html#_CPPv4N5cudaq13kraus_c | 5cudaq9run_asyncENSt6futureINSt6v |
| hannel22populateDefaultOpNamesEv) | ectorINSt15invoke_result_tINSt7de |
| -   [cu                           | cay_tI13QuantumKernelEEDpNSt7deca |
| daq::kraus_channel::probabilities | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4ARGS) |
|     member)](api/la               | -   [cudaq::RuntimeTarget (C++    |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q13kraus_channel13probabilitiesE) | struct)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|  [cudaq::kraus_channel::push_back | -   [cudaq::sample (C++           |
|     (C++                          |     function)](api/languages/c    |
|     function)](api                | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| /languages/cpp_api.html#_CPPv4N5c | mpleE13sample_resultRK14sample_op |
| udaq13kraus_channel9push_backE8kr | tionsRR13QuantumKernelDpRR4Args), |
| aus_opNSt8optionalINSt6stringEEE) |     [\[1\                         |
| -   [cudaq::kraus_channel::size   | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     function)                     | esultRR13QuantumKernelDpRR4Args), |
| ](api/languages/cpp_api.html#_CPP |     [\                            |
| v4NK5cudaq13kraus_channel4sizeEv) | [2\]](api/languages/cpp_api.html# |
| -   [                             | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| cudaq::kraus_channel::unitary_ops | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::sample_options (C++   |
|     member)](api/                 |     s                             |
| languages/cpp_api.html#_CPPv4N5cu | truct)](api/languages/cpp_api.htm |
| daq13kraus_channel11unitary_opsE) | l#_CPPv4N5cudaq14sample_optionsE) |
| -   [cudaq::kraus_op (C++         | -   [cudaq::sample_result (C++    |
|     struct)](api/languages/cpp_   |                                   |
| api.html#_CPPv4N5cudaq8kraus_opE) |  class)](api/languages/cpp_api.ht |
| -   [cudaq::kraus_op::adjoint     | ml#_CPPv4N5cudaq13sample_resultE) |
|     (C++                          | -   [cudaq::sample_result::append |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)](api/languages/cpp_ |
| CPPv4NK5cudaq8kraus_op7adjointEv) | api.html#_CPPv4N5cudaq13sample_re |
| -   [cudaq::kraus_op::data (C++   | sult6appendERK15ExecutionResultb) |
|                                   | -   [cudaq::sample_result::begin  |
|  member)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     function)]                    |
| -   [cudaq::kraus_op::kraus_op    | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4N5cudaq13sample_result5beginEv), |
|     func                          |     [\[1\]]                       |
| tion)](api/languages/cpp_api.html | (api/languages/cpp_api.html#_CPPv |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | 4NK5cudaq13sample_result5beginEv) |
| opERRNSt16initializer_listI1TEE), | -   [cudaq::sample_result::cbegin |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     function)](                   |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | api/languages/cpp_api.html#_CPPv4 |
| pENSt6vectorIN5cudaq7complexEEE), | NK5cudaq13sample_result6cbeginEv) |
|     [\[2\]](api/l                 | -   [cudaq::sample_result::cend   |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq8kraus_op8kraus_opERK8kraus_op) |     function)                     |
| -   [cudaq::kraus_op::nCols (C++  | ](api/languages/cpp_api.html#_CPP |
|                                   | v4NK5cudaq13sample_result4cendEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::sample_result::clear  |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     (C++                          |
| -   [cudaq::kraus_op::nRows (C++  |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
| member)](api/languages/cpp_api.ht | v4N5cudaq13sample_result5clearEv) |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | -   [cudaq::sample_result::count  |
| -   [cudaq::kraus_op::operator=   |     (C++                          |
|     (C++                          |     function)](                   |
|     function)                     | api/languages/cpp_api.html#_CPPv4 |
| ](api/languages/cpp_api.html#_CPP | NK5cudaq13sample_result5countENSt |
| v4N5cudaq8kraus_opaSERK8kraus_op) | 11string_viewEKNSt11string_viewE) |
| -   [cudaq::kraus_op::precision   | -   [                             |
|     (C++                          | cudaq::sample_result::deserialize |
|     memb                          |     (C++                          |
| er)](api/languages/cpp_api.html#_ |     functio                       |
| CPPv4N5cudaq8kraus_op9precisionE) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::matrix_callback (C++  | PPv4N5cudaq13sample_result11deser |
|     c                             | ializeERNSt6vectorINSt6size_tEEE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::sample_result::dump   |
| #_CPPv4N5cudaq15matrix_callbackE) |     (C++                          |
| -   [cudaq::matrix_handler (C++   |     function)](api/languag        |
|                                   | es/cpp_api.html#_CPPv4NK5cudaq13s |
| class)](api/languages/cpp_api.htm | ample_result4dumpERNSt7ostreamE), |
| l#_CPPv4N5cudaq14matrix_handlerE) |     [\[1\]                        |
| -   [cudaq::mat                   | ](api/languages/cpp_api.html#_CPP |
| rix_handler::commutation_behavior | v4NK5cudaq13sample_result4dumpEv) |
|     (C++                          | -   [cudaq::sample_result::end    |
|     struct)](api/languages/       |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq14matri |     function                      |
| x_handler20commutation_behaviorE) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq13sample_result3endEv), |
|    [cudaq::matrix_handler::define |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](a                  | Pv4NK5cudaq13sample_result3endEv) |
| pi/languages/cpp_api.html#_CPPv4N | -   [                             |
| 5cudaq14matrix_handler6defineENSt | cudaq::sample_result::expectation |
| 6stringENSt6vectorINSt7int64_tEEE |     (C++                          |
| RR15matrix_callbackRKNSt13unorder |     f                             |
| ed_mapINSt6stringENSt6stringEEE), | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4NK5cudaq13sample_result |
| [\[1\]](api/languages/cpp_api.htm | 11expectationEKNSt11string_viewE) |
| l#_CPPv4N5cudaq14matrix_handler6d | -   [c                            |
| efineENSt6stringENSt6vectorINSt7i | udaq::sample_result::get_marginal |
| nt64_tEEERR15matrix_callbackRR20d |     (C++                          |
| iag_matrix_callbackRKNSt13unorder |     function)](api/languages/cpp_ |
| ed_mapINSt6stringENSt6stringEEE), | api.html#_CPPv4NK5cudaq13sample_r |
|     [\[2\]](                      | esult12get_marginalERKNSt6vectorI |
| api/languages/cpp_api.html#_CPPv4 | NSt6size_tEEEKNSt11string_viewE), |
| N5cudaq14matrix_handler6defineENS |     [\[1\]](api/languages/cpp_    |
| t6stringENSt6vectorINSt7int64_tEE | api.html#_CPPv4NK5cudaq13sample_r |
| ERR15matrix_callbackRRNSt13unorde | esult12get_marginalERRKNSt6vector |
| red_mapINSt6stringENSt6stringEEE) | INSt6size_tEEEKNSt11string_viewE) |
| -                                 | -   [cuda                         |
|   [cudaq::matrix_handler::degrees | q::sample_result::get_total_shots |
|     (C++                          |     (C++                          |
|     function)](ap                 |     function)](api/langua         |
| i/languages/cpp_api.html#_CPPv4NK | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| 5cudaq14matrix_handler7degreesEv) | sample_result15get_total_shotsEv) |
| -                                 | -   [cuda                         |
|  [cudaq::matrix_handler::displace | q::sample_result::has_even_parity |
|     (C++                          |     (C++                          |
|     function)](api/language       |     fun                           |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ction)](api/languages/cpp_api.htm |
| rix_handler8displaceENSt6size_tE) | l#_CPPv4N5cudaq13sample_result15h |
| -   [cudaq::matrix                | as_even_parityENSt11string_viewE) |
| _handler::get_expected_dimensions | -   [cuda                         |
|     (C++                          | q::sample_result::has_expectation |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     funct                         |
| pi.html#_CPPv4NK5cudaq14matrix_ha | ion)](api/languages/cpp_api.html# |
| ndler23get_expected_dimensionsEv) | _CPPv4NK5cudaq13sample_result15ha |
| -   [cudaq::matrix_ha             | s_expectationEKNSt11string_viewE) |
| ndler::get_parameter_descriptions | -   [cu                           |
|     (C++                          | daq::sample_result::most_probable |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     fun                           |
| html#_CPPv4NK5cudaq14matrix_handl | ction)](api/languages/cpp_api.htm |
| er26get_parameter_descriptionsEv) | l#_CPPv4NK5cudaq13sample_result13 |
| -   [c                            | most_probableEKNSt11string_viewE) |
| udaq::matrix_handler::instantiate | -                                 |
|     (C++                          | [cudaq::sample_result::operator+= |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/langua         |
| 5cudaq14matrix_handler11instantia | ges/cpp_api.html#_CPPv4N5cudaq13s |
| teENSt6stringERKNSt6vectorINSt6si | ample_resultpLERK13sample_result) |
| ze_tEEERK20commutation_behavior), | -                                 |
|     [\[1\]](                      |  [cudaq::sample_result::operator= |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14matrix_handler11instanti |     function)](api/langua         |
| ateENSt6stringERRNSt6vectorINSt6s | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ize_tEEERK20commutation_behavior) | ample_resultaSERR13sample_result) |
| -   [cuda                         | -                                 |
| q::matrix_handler::matrix_handler | [cudaq::sample_result::operator== |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)](api/languag        |
| es/cpp_api.html#_CPPv4I0_NSt11ena | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ble_if_tINSt12is_base_of_vI16oper | ample_resulteqERK13sample_result) |
| ator_handler1TEEbEEEN5cudaq14matr | -   [                             |
| ix_handler14matrix_handlerERK1T), | cudaq::sample_result::probability |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4I0 |     function)](api/lan            |
| _NSt11enable_if_tINSt12is_base_of | guages/cpp_api.html#_CPPv4NK5cuda |
| _vI16operator_handler1TEEbEEEN5cu | q13sample_result11probabilityENSt |
| daq14matrix_handler14matrix_handl | 11string_viewEKNSt11string_viewE) |
| erERK1TRK20commutation_behavior), | -   [cud                          |
|     [\[2\]](api/languages/cpp_ap  | aq::sample_result::register_names |
| i.html#_CPPv4N5cudaq14matrix_hand |     (C++                          |
| ler14matrix_handlerENSt6size_tE), |     function)](api/langu          |
|     [\[3\]](api/                  | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| languages/cpp_api.html#_CPPv4N5cu | 3sample_result14register_namesEv) |
| daq14matrix_handler14matrix_handl | -                                 |
| erENSt6stringERKNSt6vectorINSt6si |    [cudaq::sample_result::reorder |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[4\]](api/                  |     function)](api/langua         |
| languages/cpp_api.html#_CPPv4N5cu | ges/cpp_api.html#_CPPv4N5cudaq13s |
| daq14matrix_handler14matrix_handl | ample_result7reorderERKNSt6vector |
| erENSt6stringERRNSt6vectorINSt6si | INSt6size_tEEEKNSt11string_viewE) |
| ze_tEEERK20commutation_behavior), | -   [cu                           |
|     [\                            | daq::sample_result::sample_result |
| [5\]](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq14matrix_handler14ma |     func                          |
| trix_handlerERK14matrix_handler), | tion)](api/languages/cpp_api.html |
|     [                             | #_CPPv4N5cudaq13sample_result13sa |
| \[6\]](api/languages/cpp_api.html | mple_resultERK15ExecutionResult), |
| #_CPPv4N5cudaq14matrix_handler14m |     [\[1\]](api/la                |
| atrix_handlerERR14matrix_handler) | nguages/cpp_api.html#_CPPv4N5cuda |
| -                                 | q13sample_result13sample_resultER |
|  [cudaq::matrix_handler::momentum | KNSt6vectorI15ExecutionResultEE), |
|     (C++                          |                                   |
|     function)](api/language       |  [\[2\]](api/languages/cpp_api.ht |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ml#_CPPv4N5cudaq13sample_result13 |
| rix_handler8momentumENSt6size_tE) | sample_resultERR13sample_result), |
| -                                 |     [                             |
|    [cudaq::matrix_handler::number | \[3\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq13sample_result13sa |
|     function)](api/langua         | mple_resultERR15ExecutionResult), |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     [\[4\]](api/lan               |
| atrix_handler6numberENSt6size_tE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -                                 | 13sample_result13sample_resultEdR |
| [cudaq::matrix_handler::operator= | KNSt6vectorI15ExecutionResultEE), |
|     (C++                          |     [\[5\]](api/lan               |
|     fun                           | guages/cpp_api.html#_CPPv4N5cudaq |
| ction)](api/languages/cpp_api.htm | 13sample_result13sample_resultEv) |
| l#_CPPv4I0_NSt11enable_if_tIXaant | -                                 |
| NSt7is_sameI1T14matrix_handlerE5v |  [cudaq::sample_result::serialize |
| alueENSt12is_base_of_vI16operator |     (C++                          |
| _handler1TEEEbEEEN5cudaq14matrix_ |     function)](api                |
| handleraSER14matrix_handlerRK1T), | /languages/cpp_api.html#_CPPv4NK5 |
|     [\[1\]](api/languages         | cudaq13sample_result9serializeEv) |
| /cpp_api.html#_CPPv4N5cudaq14matr | -   [cudaq::sample_result::size   |
| ix_handleraSERK14matrix_handler), |     (C++                          |
|     [\[2\]](api/language          |     function)](api/languages/c    |
| s/cpp_api.html#_CPPv4N5cudaq14mat | pp_api.html#_CPPv4NK5cudaq13sampl |
| rix_handleraSERR14matrix_handler) | e_result4sizeEKNSt11string_viewE) |
| -   [                             | -   [cudaq::sample_result::to_map |
| cudaq::matrix_handler::operator== |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api/languages      | _api.html#_CPPv4NK5cudaq13sample_ |
| /cpp_api.html#_CPPv4NK5cudaq14mat | result6to_mapEKNSt11string_viewE) |
| rix_handlereqERK14matrix_handler) | -   [cuda                         |
| -                                 | q::sample_result::\~sample_result |
|    [cudaq::matrix_handler::parity |     (C++                          |
|     (C++                          |     funct                         |
|     function)](api/langua         | ion)](api/languages/cpp_api.html# |
| ges/cpp_api.html#_CPPv4N5cudaq14m | _CPPv4N5cudaq13sample_resultD0Ev) |
| atrix_handler6parityENSt6size_tE) | -   [cudaq::scalar_callback (C++  |
| -                                 |     c                             |
|  [cudaq::matrix_handler::position | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_callbackE) |
|     function)](api/language       | -   [c                            |
| s/cpp_api.html#_CPPv4N5cudaq14mat | udaq::scalar_callback::operator() |
| rix_handler8positionENSt6size_tE) |     (C++                          |
| -   [cudaq::                      |     function)](api/language       |
| matrix_handler::remove_definition | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|     (C++                          | alar_callbackclERKNSt13unordered_ |
|     fu                            | mapINSt6stringENSt7complexIdEEEE) |
| nction)](api/languages/cpp_api.ht | -   [                             |
| ml#_CPPv4N5cudaq14matrix_handler1 | cudaq::scalar_callback::operator= |
| 7remove_definitionERKNSt6stringE) |     (C++                          |
| -                                 |     function)](api/languages/c    |
|   [cudaq::matrix_handler::squeeze | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _callbackaSERK15scalar_callback), |
|     function)](api/languag        |     [\[1\]](api/languages/        |
| es/cpp_api.html#_CPPv4N5cudaq14ma | cpp_api.html#_CPPv4N5cudaq15scala |
| trix_handler7squeezeENSt6size_tE) | r_callbackaSERR15scalar_callback) |
| -   [cudaq::m                     | -   [cudaq:                       |
| atrix_handler::to_diagonal_matrix | :scalar_callback::scalar_callback |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](api/languag        |
| uages/cpp_api.html#_CPPv4NK5cudaq | es/cpp_api.html#_CPPv4I0_NSt11ena |
| 14matrix_handler18to_diagonal_mat | ble_if_tINSt16is_invocable_r_vINS |
| rixERNSt13unordered_mapINSt6size_ | t7complexIdEE8CallableRKNSt13unor |
| tENSt7int64_tEEERKNSt13unordered_ | dered_mapINSt6stringENSt7complexI |
| mapINSt6stringENSt7complexIdEEEE) | dEEEEEEbEEEN5cudaq15scalar_callba |
| -                                 | ck15scalar_callbackERR8Callable), |
| [cudaq::matrix_handler::to_matrix |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)                     | Pv4N5cudaq15scalar_callback15scal |
| ](api/languages/cpp_api.html#_CPP | ar_callbackERK15scalar_callback), |
| v4NK5cudaq14matrix_handler9to_mat |     [\[2                          |
| rixERNSt13unordered_mapINSt6size_ | \]](api/languages/cpp_api.html#_C |
| tENSt7int64_tEEERKNSt13unordered_ | PPv4N5cudaq15scalar_callback15sca |
| mapINSt6stringENSt7complexIdEEEE) | lar_callbackERR15scalar_callback) |
| -                                 | -   [cudaq::scalar_operator (C++  |
| [cudaq::matrix_handler::to_string |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|     function)](api/               | #_CPPv4N5cudaq15scalar_operatorE) |
| languages/cpp_api.html#_CPPv4NK5c | -                                 |
| udaq14matrix_handler9to_stringEb) | [cudaq::scalar_operator::evaluate |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::unique_id |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     function)](api/               | pi.html#_CPPv4NK5cudaq15scalar_op |
| languages/cpp_api.html#_CPPv4NK5c | erator8evaluateERKNSt13unordered_ |
| udaq14matrix_handler9unique_idEv) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq:                       | -   [cudaq::scalar_ope            |
| :matrix_handler::\~matrix_handler | rator::get_parameter_descriptions |
|     (C++                          |     (C++                          |
|     functi                        |     f                             |
| on)](api/languages/cpp_api.html#_ | unction)](api/languages/cpp_api.h |
| CPPv4N5cudaq14matrix_handlerD0Ev) | tml#_CPPv4NK5cudaq15scalar_operat |
| -   [cudaq::matrix_op (C++        | or26get_parameter_descriptionsEv) |
|     type)](api/languages/cpp_a    | -   [cu                           |
| pi.html#_CPPv4N5cudaq9matrix_opE) | daq::scalar_operator::is_constant |
| -   [cudaq::matrix_op_term (C++   |     (C++                          |
|                                   |     function)](api/lang           |
|  type)](api/languages/cpp_api.htm | uages/cpp_api.html#_CPPv4NK5cudaq |
| l#_CPPv4N5cudaq14matrix_op_termE) | 15scalar_operator11is_constantEv) |
| -                                 | -   [c                            |
|    [cudaq::mdiag_operator_handler | udaq::scalar_operator::operator\* |
|     (C++                          |     (C++                          |
|     class)](                      |     function                      |
| api/languages/cpp_api.html#_CPPv4 | )](api/languages/cpp_api.html#_CP |
| N5cudaq22mdiag_operator_handlerE) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::mpi (C++              | 7complexIdEERK15scalar_operator), |
|     type)](api/languages          |     [\[1\                         |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::mpi::all_gather (C++  | Pv4N5cudaq15scalar_operatormlENSt |
|     fu                            | 7complexIdEERR15scalar_operator), |
| nction)](api/languages/cpp_api.ht |     [\[2\]](api/languages/cp      |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | p_api.html#_CPPv4N5cudaq15scalar_ |
| RNSt6vectorIdEERKNSt6vectorIdEE), | operatormlEdRK15scalar_operator), |
|                                   |     [\[3\]](api/languages/cp      |
|   [\[1\]](api/languages/cpp_api.h | p_api.html#_CPPv4N5cudaq15scalar_ |
| tml#_CPPv4N5cudaq3mpi10all_gather | operatormlEdRR15scalar_operator), |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     [\[4\]](api/languages         |
| -   [cudaq::mpi::all_reduce (C++  | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|                                   | alar_operatormlENSt7complexIdEE), |
|  function)](api/languages/cpp_api |     [\[5\]](api/languages/cpp     |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | _api.html#_CPPv4NKR5cudaq15scalar |
| reduceE1TRK1TRK14BinaryFunction), | _operatormlERK15scalar_operator), |
|     [\[1\]](api/langu             |     [\[6\]]                       |
| ages/cpp_api.html#_CPPv4I00EN5cud | (api/languages/cpp_api.html#_CPPv |
| aq3mpi10all_reduceE1TRK1TRK4Func) | 4NKR5cudaq15scalar_operatormlEd), |
| -   [cudaq::mpi::broadcast (C++   |     [\[7\]](api/language          |
|     function)](api/               | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| languages/cpp_api.html#_CPPv4N5cu | alar_operatormlENSt7complexIdEE), |
| daq3mpi9broadcastERNSt6stringEi), |     [\[8\]](api/languages/cp      |
|     [\[1\]](api/la                | p_api.html#_CPPv4NO5cudaq15scalar |
| nguages/cpp_api.html#_CPPv4N5cuda | _operatormlERK15scalar_operator), |
| q3mpi9broadcastERNSt6vectorIdEEi) |     [\[9\                         |
| -   [cudaq::mpi::finalize (C++    | ]](api/languages/cpp_api.html#_CP |
|     f                             | Pv4NO5cudaq15scalar_operatormlEd) |
| unction)](api/languages/cpp_api.h | -   [cu                           |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | daq::scalar_operator::operator\*= |
| -   [cudaq::mpi::initialize (C++  |     (C++                          |
|     function                      |     function)](api/languag        |
| )](api/languages/cpp_api.html#_CP | es/cpp_api.html#_CPPv4N5cudaq15sc |
| Pv4N5cudaq3mpi10initializeEiPPc), | alar_operatormLENSt7complexIdEE), |
|     [                             |     [\[1\]](api/languages/c       |
| \[1\]](api/languages/cpp_api.html | pp_api.html#_CPPv4N5cudaq15scalar |
| #_CPPv4N5cudaq3mpi10initializeEv) | _operatormLERK15scalar_operator), |
| -   [cudaq::mpi::is_initialized   |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     function                      | PPv4N5cudaq15scalar_operatormLEd) |
| )](api/languages/cpp_api.html#_CP | -   [                             |
| Pv4N5cudaq3mpi14is_initializedEv) | cudaq::scalar_operator::operator+ |
| -   [cudaq::mpi::num_ranks (C++   |     (C++                          |
|     fu                            |     function                      |
| nction)](api/languages/cpp_api.ht | )](api/languages/cpp_api.html#_CP |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | Pv4N5cudaq15scalar_operatorplENSt |
| -   [cudaq::mpi::rank (C++        | 7complexIdEERK15scalar_operator), |
|                                   |     [\[1\                         |
|    function)](api/languages/cpp_a | ]](api/languages/cpp_api.html#_CP |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | Pv4N5cudaq15scalar_operatorplENSt |
| -   [cudaq::noise_model (C++      | 7complexIdEERR15scalar_operator), |
|                                   |     [\[2\]](api/languages/cp      |
|    class)](api/languages/cpp_api. | p_api.html#_CPPv4N5cudaq15scalar_ |
| html#_CPPv4N5cudaq11noise_modelE) | operatorplEdRK15scalar_operator), |
| -   [cudaq::n                     |     [\[3\]](api/languages/cp      |
| oise_model::add_all_qubit_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatorplEdRR15scalar_operator), |
|     function)](api                |     [\[4\]](api/languages         |
| /languages/cpp_api.html#_CPPv4IDp | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| EN5cudaq11noise_model21add_all_qu | alar_operatorplENSt7complexIdEE), |
| bit_channelEvRK13kraus_channeli), |     [\[5\]](api/languages/cpp     |
|     [\[1\]](api/langua            | _api.html#_CPPv4NKR5cudaq15scalar |
| ges/cpp_api.html#_CPPv4N5cudaq11n | _operatorplERK15scalar_operator), |
| oise_model21add_all_qubit_channel |     [\[6\]]                       |
| ERKNSt6stringERK13kraus_channeli) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatorplEd), |
|  [cudaq::noise_model::add_channel |     [\[7\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     funct                         | 4NKR5cudaq15scalar_operatorplEv), |
| ion)](api/languages/cpp_api.html# |     [\[8\]](api/language          |
| _CPPv4IDpEN5cudaq11noise_model11a | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| dd_channelEvRK15PredicateFuncTy), | alar_operatorplENSt7complexIdEE), |
|     [\[1\]](api/languages/cpp_    |     [\[9\]](api/languages/cp      |
| api.html#_CPPv4IDpEN5cudaq11noise | p_api.html#_CPPv4NO5cudaq15scalar |
| _model11add_channelEvRKNSt6vector | _operatorplERK15scalar_operator), |
| INSt6size_tEEERK13kraus_channel), |     [\[10\]                       |
|     [\[2\]](ap                    | ](api/languages/cpp_api.html#_CPP |
| i/languages/cpp_api.html#_CPPv4N5 | v4NO5cudaq15scalar_operatorplEd), |
| cudaq11noise_model11add_channelER |     [\[11\                        |
| KNSt6stringERK15PredicateFuncTy), | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4NO5cudaq15scalar_operatorplEv) |
| [\[3\]](api/languages/cpp_api.htm | -   [c                            |
| l#_CPPv4N5cudaq11noise_model11add | udaq::scalar_operator::operator+= |
| _channelERKNSt6stringERKNSt6vecto |     (C++                          |
| rINSt6size_tEEERK13kraus_channel) |     function)](api/languag        |
| -   [cudaq::noise_model::empty    | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatorpLENSt7complexIdEE), |
|     function                      |     [\[1\]](api/languages/c       |
| )](api/languages/cpp_api.html#_CP | pp_api.html#_CPPv4N5cudaq15scalar |
| Pv4NK5cudaq11noise_model5emptyEv) | _operatorpLERK15scalar_operator), |
| -                                 |     [\[2                          |
| [cudaq::noise_model::get_channels | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatorpLEd) |
|     function)](api/l              | -   [                             |
| anguages/cpp_api.html#_CPPv4I0ENK | cudaq::scalar_operator::operator- |
| 5cudaq11noise_model12get_channels |     (C++                          |
| ENSt6vectorI13kraus_channelEERKNS |     function                      |
| t6vectorINSt6size_tEEERKNSt6vecto | )](api/languages/cpp_api.html#_CP |
| rINSt6size_tEEERKNSt6vectorIdEE), | Pv4N5cudaq15scalar_operatormiENSt |
|     [\[1\]](api/languages/cpp_a   | 7complexIdEERK15scalar_operator), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[1\                         |
| el12get_channelsERKNSt6stringERKN | ]](api/languages/cpp_api.html#_CP |
| St6vectorINSt6size_tEEERKNSt6vect | Pv4N5cudaq15scalar_operatormiENSt |
| orINSt6size_tEEERKNSt6vectorIdEE) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
|  [cudaq::noise_model::noise_model | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRK15scalar_operator), |
|     function)](api                |     [\[3\]](api/languages/cp      |
| /languages/cpp_api.html#_CPPv4N5c | p_api.html#_CPPv4N5cudaq15scalar_ |
| udaq11noise_model11noise_modelEv) | operatormiEdRR15scalar_operator), |
| -   [cu                           |     [\[4\]](api/languages         |
| daq::noise_model::PredicateFuncTy | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     type)](api/la                 |     [\[5\]](api/languages/cpp     |
| nguages/cpp_api.html#_CPPv4N5cuda | _api.html#_CPPv4NKR5cudaq15scalar |
| q11noise_model15PredicateFuncTyE) | _operatormiERK15scalar_operator), |
| -   [cud                          |     [\[6\]]                       |
| aq::noise_model::register_channel | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEd), |
|     function)](api/languages      |     [\[7\]]                       |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | (api/languages/cpp_api.html#_CPPv |
| noise_model16register_channelEvv) | 4NKR5cudaq15scalar_operatormiEv), |
| -   [cudaq::                      |     [\[8\]](api/language          |
| noise_model::requires_constructor | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     type)](api/languages/cp       |     [\[9\]](api/languages/cp      |
| p_api.html#_CPPv4I0DpEN5cudaq11no | p_api.html#_CPPv4NO5cudaq15scalar |
| ise_model20requires_constructorE) | _operatormiERK15scalar_operator), |
| -   [cudaq::noise_model_type (C++ |     [\[10\]                       |
|     e                             | ](api/languages/cpp_api.html#_CPP |
| num)](api/languages/cpp_api.html# | v4NO5cudaq15scalar_operatormiEd), |
| _CPPv4N5cudaq16noise_model_typeE) |     [\[11\                        |
| -   [cudaq::no                    | ]](api/languages/cpp_api.html#_CP |
| ise_model_type::amplitude_damping | Pv4NO5cudaq15scalar_operatormiEv) |
|     (C++                          | -   [c                            |
|     enumerator)](api/languages    | udaq::scalar_operator::operator-= |
| /cpp_api.html#_CPPv4N5cudaq16nois |     (C++                          |
| e_model_type17amplitude_dampingE) |     function)](api/languag        |
| -   [cudaq::noise_mode            | es/cpp_api.html#_CPPv4N5cudaq15sc |
| l_type::amplitude_damping_channel | alar_operatormIENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     e                             | pp_api.html#_CPPv4N5cudaq15scalar |
| numerator)](api/languages/cpp_api | _operatormIERK15scalar_operator), |
| .html#_CPPv4N5cudaq16noise_model_ |     [\[2                          |
| type25amplitude_damping_channelE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::n                     | PPv4N5cudaq15scalar_operatormIEd) |
| oise_model_type::bit_flip_channel | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator/ |
|     enumerator)](api/language     |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     function                      |
| se_model_type16bit_flip_channelE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::                      | Pv4N5cudaq15scalar_operatordvENSt |
| noise_model_type::depolarization1 | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](api/languag      | ]](api/languages/cpp_api.html#_CP |
| es/cpp_api.html#_CPPv4N5cudaq16no | Pv4N5cudaq15scalar_operatordvENSt |
| ise_model_type15depolarization1E) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::                      |     [\[2\]](api/languages/cp      |
| noise_model_type::depolarization2 | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRK15scalar_operator), |
|     enumerator)](api/languag      |     [\[3\]](api/languages/cp      |
| es/cpp_api.html#_CPPv4N5cudaq16no | p_api.html#_CPPv4N5cudaq15scalar_ |
| ise_model_type15depolarization2E) | operatordvEdRR15scalar_operator), |
| -   [cudaq::noise_m               |     [\[4\]](api/languages         |
| odel_type::depolarization_channel | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatordvENSt7complexIdEE), |
|                                   |     [\[5\]](api/languages/cpp     |
|   enumerator)](api/languages/cpp_ | _api.html#_CPPv4NKR5cudaq15scalar |
| api.html#_CPPv4N5cudaq16noise_mod | _operatordvERK15scalar_operator), |
| el_type22depolarization_channelE) |     [\[6\]]                       |
| -                                 | (api/languages/cpp_api.html#_CPPv |
|  [cudaq::noise_model_type::pauli1 | 4NKR5cudaq15scalar_operatordvEd), |
|     (C++                          |     [\[7\]](api/language          |
|     enumerator)](a                | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| pi/languages/cpp_api.html#_CPPv4N | alar_operatordvENSt7complexIdEE), |
| 5cudaq16noise_model_type6pauli1E) |     [\[8\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4NO5cudaq15scalar |
|  [cudaq::noise_model_type::pauli2 | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     enumerator)](a                | ]](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4NO5cudaq15scalar_operatordvEd) |
| 5cudaq16noise_model_type6pauli2E) | -   [c                            |
| -   [cudaq                        | udaq::scalar_operator::operator/= |
| ::noise_model_type::phase_damping |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](api/langu        | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ages/cpp_api.html#_CPPv4N5cudaq16 | alar_operatordVENSt7complexIdEE), |
| noise_model_type13phase_dampingE) |     [\[1\]](api/languages/c       |
| -   [cudaq::noi                   | pp_api.html#_CPPv4N5cudaq15scalar |
| se_model_type::phase_flip_channel | _operatordVERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     enumerator)](api/languages/   | \]](api/languages/cpp_api.html#_C |
| cpp_api.html#_CPPv4N5cudaq16noise | PPv4N5cudaq15scalar_operatordVEd) |
| _model_type18phase_flip_channelE) | -   [                             |
| -                                 | cudaq::scalar_operator::operator= |
| [cudaq::noise_model_type::unknown |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     enumerator)](ap               | pp_api.html#_CPPv4N5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatoraSERK15scalar_operator), |
| cudaq16noise_model_type7unknownE) |     [\[1\]](api/languages/        |
| -                                 | cpp_api.html#_CPPv4N5cudaq15scala |
| [cudaq::noise_model_type::x_error | r_operatoraSERR15scalar_operator) |
|     (C++                          | -   [c                            |
|     enumerator)](ap               | udaq::scalar_operator::operator== |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7x_errorE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq15scala |
| [cudaq::noise_model_type::y_error | r_operatoreqERK15scalar_operator) |
|     (C++                          | -   [cudaq:                       |
|     enumerator)](ap               | :scalar_operator::scalar_operator |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7y_errorE) |     func                          |
| -                                 | tion)](api/languages/cpp_api.html |
| [cudaq::noise_model_type::z_error | #_CPPv4N5cudaq15scalar_operator15 |
|     (C++                          | scalar_operatorENSt7complexIdEE), |
|     enumerator)](ap               |     [\[1\]](api/langu             |
| i/languages/cpp_api.html#_CPPv4N5 | ages/cpp_api.html#_CPPv4N5cudaq15 |
| cudaq16noise_model_type7z_errorE) | scalar_operator15scalar_operatorE |
| -   [cudaq::num_available_gpus    | RK15scalar_callbackRRNSt13unorder |
|     (C++                          | ed_mapINSt6stringENSt6stringEEE), |
|     function                      |     [\[2\                         |
| )](api/languages/cpp_api.html#_CP | ]](api/languages/cpp_api.html#_CP |
| Pv4N5cudaq18num_available_gpusEv) | Pv4N5cudaq15scalar_operator15scal |
| -   [cudaq::observe (C++          | ar_operatorERK15scalar_operator), |
|     function)]                    |     [\[3\]](api/langu             |
| (api/languages/cpp_api.html#_CPPv | ages/cpp_api.html#_CPPv4N5cudaq15 |
| 4I00DpEN5cudaq7observeENSt6vector | scalar_operator15scalar_operatorE |
| I14observe_resultEERR13QuantumKer | RR15scalar_callbackRRNSt13unorder |
| nelRK15SpinOpContainerDpRR4Args), | ed_mapINSt6stringENSt6stringEEE), |
|     [\[1\]](api/languages/cpp_ap  |     [\[4\                         |
| i.html#_CPPv4I0DpEN5cudaq7observe | ]](api/languages/cpp_api.html#_CP |
| E14observe_resultNSt6size_tERR13Q | Pv4N5cudaq15scalar_operator15scal |
| uantumKernelRK7spin_opDpRR4Args), | ar_operatorERR15scalar_operator), |
|     [\[                           |     [\[5\]](api/language          |
| 2\]](api/languages/cpp_api.html#_ | s/cpp_api.html#_CPPv4N5cudaq15sca |
| CPPv4I0DpEN5cudaq7observeE14obser | lar_operator15scalar_operatorEd), |
| ve_resultRK15observe_optionsRR13Q |     [\[6\]](api/languag           |
| uantumKernelRK7spin_opDpRR4Args), | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     [\[3\]](api/lang              | alar_operator15scalar_operatorEv) |
| uages/cpp_api.html#_CPPv4I0DpEN5c | -   [                             |
| udaq7observeE14observe_resultRR13 | cudaq::scalar_operator::to_matrix |
| QuantumKernelRK7spin_opDpRR4Args) |     (C++                          |
| -   [cudaq::observe_options (C++  |                                   |
|     st                            |   function)](api/languages/cpp_ap |
| ruct)](api/languages/cpp_api.html | i.html#_CPPv4NK5cudaq15scalar_ope |
| #_CPPv4N5cudaq15observe_optionsE) | rator9to_matrixERKNSt13unordered_ |
| -   [cudaq::observe_result (C++   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [                             |
| class)](api/languages/cpp_api.htm | cudaq::scalar_operator::to_string |
| l#_CPPv4N5cudaq14observe_resultE) |     (C++                          |
| -                                 |     function)](api/l              |
|    [cudaq::observe_result::counts | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq15scalar_operator9to_stringEv) |
|     function)](api/languages/c    | -   [cudaq::s                     |
| pp_api.html#_CPPv4N5cudaq14observ | calar_operator::\~scalar_operator |
| e_result6countsERK12spin_op_term) |     (C++                          |
| -   [cudaq::observe_result::dump  |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)                     | PPv4N5cudaq15scalar_operatorD0Ev) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::set_noise (C++        |
| v4N5cudaq14observe_result4dumpEv) |     function)](api/langu          |
| -   [c                            | ages/cpp_api.html#_CPPv4N5cudaq9s |
| udaq::observe_result::expectation | et_noiseERKN5cudaq11noise_modelE) |
|     (C++                          | -   [cudaq::set_random_seed (C++  |
|                                   |     function)](api/               |
| function)](api/languages/cpp_api. | languages/cpp_api.html#_CPPv4N5cu |
| html#_CPPv4N5cudaq14observe_resul | daq15set_random_seedENSt6size_tE) |
| t11expectationERK12spin_op_term), | -   [cudaq::simulation_precision  |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     enum)                         |
| q14observe_result11expectationEv) | ](api/languages/cpp_api.html#_CPP |
| -   [cuda                         | v4N5cudaq20simulation_precisionE) |
| q::observe_result::id_coefficient | -   [                             |
|     (C++                          | cudaq::simulation_precision::fp32 |
|     function)](api/langu          |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     enumerator)](api              |
| observe_result14id_coefficientEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cuda                         | udaq20simulation_precision4fp32E) |
| q::observe_result::observe_result | -   [                             |
|     (C++                          | cudaq::simulation_precision::fp64 |
|                                   |     (C++                          |
|   function)](api/languages/cpp_ap |     enumerator)](api              |
| i.html#_CPPv4N5cudaq14observe_res | /languages/cpp_api.html#_CPPv4N5c |
| ult14observe_resultEdRK7spin_op), | udaq20simulation_precision4fp64E) |
|     [\[1\]](a                     | -   [cudaq::SimulationState (C++  |
| pi/languages/cpp_api.html#_CPPv4N |     c                             |
| 5cudaq14observe_result14observe_r | lass)](api/languages/cpp_api.html |
| esultEdRK7spin_op13sample_result) | #_CPPv4N5cudaq15SimulationStateE) |
| -                                 | -   [                             |
|  [cudaq::observe_result::operator | cudaq::SimulationState::precision |
|     double (C++                   |     (C++                          |
|     functio                       |     enum)](api                    |
| n)](api/languages/cpp_api.html#_C | /languages/cpp_api.html#_CPPv4N5c |
| PPv4N5cudaq14observe_resultcvdEv) | udaq15SimulationState9precisionE) |
| -                                 | -   [cudaq:                       |
|  [cudaq::observe_result::raw_data | :SimulationState::precision::fp32 |
|     (C++                          |     (C++                          |
|     function)](ap                 |     enumerator)](api/lang         |
| i/languages/cpp_api.html#_CPPv4N5 | uages/cpp_api.html#_CPPv4N5cudaq1 |
| cudaq14observe_result8raw_dataEv) | 5SimulationState9precision4fp32E) |
| -   [cudaq::operator_handler (C++ | -   [cudaq:                       |
|     cl                            | :SimulationState::precision::fp64 |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16operator_handlerE) |     enumerator)](api/lang         |
| -   [cudaq::optimizable_function  | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     (C++                          | 5SimulationState9precision4fp64E) |
|     class)                        | -                                 |
| ](api/languages/cpp_api.html#_CPP |   [cudaq::SimulationState::Tensor |
| v4N5cudaq20optimizable_functionE) |     (C++                          |
| -   [cudaq::optimization_result   |     struct)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     type                          | N5cudaq15SimulationState6TensorE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::spin_handler (C++     |
| Pv4N5cudaq19optimization_resultE) |                                   |
| -   [cudaq::optimizer (C++        |   class)](api/languages/cpp_api.h |
|     class)](api/languages/cpp_a   | tml#_CPPv4N5cudaq12spin_handlerE) |
| pi.html#_CPPv4N5cudaq9optimizerE) | -   [cudaq:                       |
| -   [cudaq::optimizer::optimize   | :spin_handler::to_diagonal_matrix |
|     (C++                          |     (C++                          |
|                                   |     function)](api/la             |
|  function)](api/languages/cpp_api | nguages/cpp_api.html#_CPPv4NK5cud |
| .html#_CPPv4N5cudaq9optimizer8opt | aq12spin_handler18to_diagonal_mat |
| imizeEKiRR20optimizable_function) | rixERNSt13unordered_mapINSt6size_ |
| -   [cu                           | tENSt7int64_tEEERKNSt13unordered_ |
| daq::optimizer::requiresGradients | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -                                 |
|     function)](api/la             |   [cudaq::spin_handler::to_matrix |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9optimizer17requiresGradientsEv) |     function                      |
| -   [cudaq::orca (C++             | )](api/languages/cpp_api.html#_CP |
|     type)](api/languages/         | Pv4N5cudaq12spin_handler9to_matri |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | xERKNSt6stringENSt7complexIdEEb), |
| -   [cudaq::orca::sample (C++     |     [\[1                          |
|     function)](api/languages/c    | \]](api/languages/cpp_api.html#_C |
| pp_api.html#_CPPv4N5cudaq4orca6sa | PPv4NK5cudaq12spin_handler9to_mat |
| mpleERNSt6vectorINSt6size_tEEERNS | rixERNSt13unordered_mapINSt6size_ |
| t6vectorINSt6size_tEEERNSt6vector | tENSt7int64_tEEERKNSt13unordered_ |
| IdEERNSt6vectorIdEEiNSt6size_tE), | mapINSt6stringENSt7complexIdEEEE) |
|     [\[1\]]                       | -   [cuda                         |
| (api/languages/cpp_api.html#_CPPv | q::spin_handler::to_sparse_matrix |
| 4N5cudaq4orca6sampleERNSt6vectorI |     (C++                          |
| NSt6size_tEEERNSt6vectorINSt6size |     function)](api/               |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::orca::sample_async    | daq12spin_handler16to_sparse_matr |
|     (C++                          | ixERKNSt6stringENSt7complexIdEEb) |
|                                   | -                                 |
| function)](api/languages/cpp_api. |   [cudaq::spin_handler::to_string |
| html#_CPPv4N5cudaq4orca12sample_a |     (C++                          |
| syncERNSt6vectorINSt6size_tEEERNS |     function)](ap                 |
| t6vectorINSt6size_tEEERNSt6vector | i/languages/cpp_api.html#_CPPv4NK |
| IdEERNSt6vectorIdEEiNSt6size_tE), | 5cudaq12spin_handler9to_stringEb) |
|     [\[1\]](api/la                | -                                 |
| nguages/cpp_api.html#_CPPv4N5cuda |   [cudaq::spin_handler::unique_id |
| q4orca12sample_asyncERNSt6vectorI |     (C++                          |
| NSt6size_tEEERNSt6vectorINSt6size |     function)](ap                 |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::OrcaRemoteRESTQPU     | 5cudaq12spin_handler9unique_idEv) |
|     (C++                          | -   [cudaq::spin_op (C++          |
|     cla                           |     type)](api/languages/cpp      |
| ss)](api/languages/cpp_api.html#_ | _api.html#_CPPv4N5cudaq7spin_opE) |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | -   [cudaq::spin_op_term (C++     |
| -   [cudaq::pauli1 (C++           |                                   |
|     class)](api/languages/cp      |    type)](api/languages/cpp_api.h |
| p_api.html#_CPPv4N5cudaq6pauli1E) | tml#_CPPv4N5cudaq12spin_op_termE) |
| -                                 | -   [cudaq::state (C++            |
|    [cudaq::pauli1::num_parameters |     class)](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq5stateE) |
|     member)]                      | -   [cudaq::state::amplitude (C++ |
| (api/languages/cpp_api.html#_CPPv |     function)](api/lang           |
| 4N5cudaq6pauli114num_parametersE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
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
