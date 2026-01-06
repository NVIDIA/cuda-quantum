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
        -   [Built in CUDA-Q Optimizers and
            Gradients](examples/python/optimizers_gradients.html#Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
        -   [Third-Party
            Optimizers](examples/python/optimizers_gradients.html#Third-Party-Optimizers){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](examples/python/optimizers_gradients.html#Parallel-Parameter-Shift-Gradients){.reference
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
| -   [add_all_qubit_channel()      | -   [append() (cudaq.KrausChannel |
|     (cudaq.NoiseModel             |                                   |
|     method)](api                  |  method)](api/languages/python_ap |
| /languages/python_api.html#cudaq. | i.html#cudaq.KrausChannel.append) |
| NoiseModel.add_all_qubit_channel) | -   [argument_count               |
| -   [add_channel()                |     (cudaq.PyKernel               |
|     (cudaq.NoiseModel             |     attrib                        |
|     me                            | ute)](api/languages/python_api.ht |
| thod)](api/languages/python_api.h | ml#cudaq.PyKernel.argument_count) |
| tml#cudaq.NoiseModel.add_channel) | -   [arguments (cudaq.PyKernel    |
| -   [all_gather() (in module      |     a                             |
|                                   | ttribute)](api/languages/python_a |
|    cudaq.mpi)](api/languages/pyth | pi.html#cudaq.PyKernel.arguments) |
| on_api.html#cudaq.mpi.all_gather) | -   [as_pauli()                   |
| -   [amplitude() (cudaq.State     |     (cudaq.o                      |
|     method)](api/languages/pytho  | perators.spin.SpinOperatorElement |
| n_api.html#cudaq.State.amplitude) |     method)](api/languages/       |
| -   [AmplitudeDampingChannel      | python_api.html#cudaq.operators.s |
|     (class in                     | pin.SpinOperatorElement.as_pauli) |
|     cu                            | -   [AsyncEvolveResult (class in  |
| daq)](api/languages/python_api.ht |     cudaq)](api/languages/python_ |
| ml#cudaq.AmplitudeDampingChannel) | api.html#cudaq.AsyncEvolveResult) |
| -   [amplitudes() (cudaq.State    | -   [AsyncObserveResult (class in |
|     method)](api/languages/python |                                   |
| _api.html#cudaq.State.amplitudes) |    cudaq)](api/languages/python_a |
| -   [annihilate() (in module      | pi.html#cudaq.AsyncObserveResult) |
|     c                             | -   [AsyncSampleResult (class in  |
| udaq.boson)](api/languages/python |     cudaq)](api/languages/python_ |
| _api.html#cudaq.boson.annihilate) | api.html#cudaq.AsyncSampleResult) |
|     -   [(in module               | -   [AsyncStateResult (class in   |
|         cudaq                     |     cudaq)](api/languages/python  |
| .fermion)](api/languages/python_a | _api.html#cudaq.AsyncStateResult) |
| pi.html#cudaq.fermion.annihilate) |                                   |
+-----------------------------------+-----------------------------------+

## B {#B}

+-----------------------------------+-----------------------------------+
| -   [BaseIntegrator (class in     | -   [BosonOperatorElement (class  |
|                                   |     in                            |
| cudaq.dynamics.integrator)](api/l |                                   |
| anguages/python_api.html#cudaq.dy |   cudaq.operators.boson)](api/lan |
| namics.integrator.BaseIntegrator) | guages/python_api.html#cudaq.oper |
| -   [BitFlipChannel (class in     | ators.boson.BosonOperatorElement) |
|     cudaq)](api/languages/pyth    | -   [BosonOperatorTerm (class in  |
| on_api.html#cudaq.BitFlipChannel) |     cudaq.operators.boson)](api/  |
| -   [BosonOperator (class in      | languages/python_api.html#cudaq.o |
|     cudaq.operators.boson)](      | perators.boson.BosonOperatorTerm) |
| api/languages/python_api.html#cud | -   [broadcast() (in module       |
| aq.operators.boson.BosonOperator) |     cudaq.mpi)](api/languages/pyt |
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
| -   [CentralDifference (class in  | ::phase_flip_channel::num_targets |
|     cudaq.gradients)              |     (C++                          |
| ](api/languages/python_api.html#c |     member)](api/langu            |
| udaq.gradients.CentralDifference) | ages/cpp_api.html#_CPPv4N5cudaq18 |
| -   [clear() (cudaq.Resources     | phase_flip_channel11num_targetsE) |
|     method)](api/languages/pytho  | -   [cudaq::product_op (C++       |
| n_api.html#cudaq.Resources.clear) |                                   |
|     -   [(cudaq.SampleResult      |  class)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4I0EN5cudaq10product_opE) |
|   method)](api/languages/python_a | -   [cudaq::product_op::begin     |
| pi.html#cudaq.SampleResult.clear) |     (C++                          |
| -   [COBYLA (class in             |     functio                       |
|     cudaq.o                       | n)](api/languages/cpp_api.html#_C |
| ptimizers)](api/languages/python_ | PPv4NK5cudaq10product_op5beginEv) |
| api.html#cudaq.optimizers.COBYLA) | -                                 |
| -   [coefficient                  |  [cudaq::product_op::canonicalize |
|     (cudaq.                       |     (C++                          |
| operators.boson.BosonOperatorTerm |     func                          |
|     property)](api/languages/py   | tion)](api/languages/cpp_api.html |
| thon_api.html#cudaq.operators.bos | #_CPPv4N5cudaq10product_op12canon |
| on.BosonOperatorTerm.coefficient) | icalizeERKNSt3setINSt6size_tEEE), |
|     -   [(cudaq.oper              |     [\[1\]](api                   |
| ators.fermion.FermionOperatorTerm | /languages/cpp_api.html#_CPPv4N5c |
|                                   | udaq10product_op12canonicalizeEv) |
|   property)](api/languages/python | -   [                             |
| _api.html#cudaq.operators.fermion | cudaq::product_op::const_iterator |
| .FermionOperatorTerm.coefficient) |     (C++                          |
|     -   [(c                       |     struct)](api/                 |
| udaq.operators.MatrixOperatorTerm | languages/cpp_api.html#_CPPv4N5cu |
|         property)](api/languag    | daq10product_op14const_iteratorE) |
| es/python_api.html#cudaq.operator | -   [cudaq::product_o             |
| s.MatrixOperatorTerm.coefficient) | p::const_iterator::const_iterator |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     fu                            |
|         property)](api/languages/ | nction)](api/languages/cpp_api.ht |
| python_api.html#cudaq.operators.s | ml#_CPPv4N5cudaq10product_op14con |
| pin.SpinOperatorTerm.coefficient) | st_iterator14const_iteratorEPK10p |
| -   [col_count                    | roduct_opI9HandlerTyENSt6size_tE) |
|     (cudaq.KrausOperator          | -   [cudaq::produ                 |
|     prope                         | ct_op::const_iterator::operator!= |
| rty)](api/languages/python_api.ht |     (C++                          |
| ml#cudaq.KrausOperator.col_count) |     fun                           |
| -   [compile()                    | ction)](api/languages/cpp_api.htm |
|     (cudaq.PyKernelDecorator      | l#_CPPv4NK5cudaq10product_op14con |
|     metho                         | st_iteratorneERK14const_iterator) |
| d)](api/languages/python_api.html | -   [cudaq::produ                 |
| #cudaq.PyKernelDecorator.compile) | ct_op::const_iterator::operator\* |
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
| PPv4N5cudaq7details6future6future | -   [cudaq::QPU::enqueue (C++     |
| ERNSt6vectorI3JobEERNSt6stringERN |     function)](ap                 |
| St3mapINSt6stringENSt6stringEEE), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[1\]](api/lang              | cudaq3QPU7enqueueER11QuantumTask) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -   [cudaq::QPU::getConnectivity  |
| details6future6futureERR6future), |     (C++                          |
|     [\[2\]]                       |     function)                     |
| (api/languages/cpp_api.html#_CPPv | ](api/languages/cpp_api.html#_CPP |
| 4N5cudaq7details6future6futureEv) | v4N5cudaq3QPU15getConnectivityEv) |
| -   [cu                           | -                                 |
| daq::details::kernel_builder_base | [cudaq::QPU::getExecutionThreadId |
|     (C++                          |     (C++                          |
|     class)](api/l                 |     function)](api/               |
| anguages/cpp_api.html#_CPPv4N5cud | languages/cpp_api.html#_CPPv4NK5c |
| aq7details19kernel_builder_baseE) | udaq3QPU20getExecutionThreadIdEv) |
| -   [cudaq::details::             | -   [cudaq::QPU::getNumQubits     |
| kernel_builder_base::operator\<\< |     (C++                          |
|     (C++                          |     functi                        |
|     function)](api/langua         | on)](api/languages/cpp_api.html#_ |
| ges/cpp_api.html#_CPPv4N5cudaq7de | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| tails19kernel_builder_baselsERNSt | -   [                             |
| 7ostreamERK19kernel_builder_base) | cudaq::QPU::getRemoteCapabilities |
| -   [                             |     (C++                          |
| cudaq::details::KernelBuilderType |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     class)](api                   | daq3QPU21getRemoteCapabilitiesEv) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::QPU::isEmulated (C++  |
| udaq7details17KernelBuilderTypeE) |     func                          |
| -   [cudaq::d                     | tion)](api/languages/cpp_api.html |
| etails::KernelBuilderType::create | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     (C++                          | -   [cudaq::QPU::isSimulator (C++ |
|     function)                     |     funct                         |
| ](api/languages/cpp_api.html#_CPP | ion)](api/languages/cpp_api.html# |
| v4N5cudaq7details17KernelBuilderT | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| ype6createEPN4mlir11MLIRContextE) | -   [cudaq::QPU::launchKernel     |
| -   [cudaq::details::Ker          |     (C++                          |
| nelBuilderType::KernelBuilderType |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     function)](api/lang           | daq3QPU12launchKernelERKNSt6strin |
| uages/cpp_api.html#_CPPv4N5cudaq7 | gE15KernelThunkTypePvNSt8uint64_t |
| details17KernelBuilderType17Kerne | ENSt8uint64_tERKNSt6vectorIPvEE), |
| lBuilderTypeERRNSt8functionIFN4ml |                                   |
| ir4TypeEPN4mlir11MLIRContextEEEE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::diag_matrix_callback  | ml#_CPPv4N5cudaq3QPU12launchKerne |
|     (C++                          | lERKNSt6stringERKNSt6vectorIPvEE) |
|     class)                        | -   [cudaq::QPU::onRandomSeedSet  |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq20diag_matrix_callbackE) |     function)](api/lang           |
| -   [cudaq::dyn (C++              | uages/cpp_api.html#_CPPv4N5cudaq3 |
|     member)](api/languages        | QPU15onRandomSeedSetENSt6size_tE) |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | -   [cudaq::QPU::QPU (C++         |
| -   [cudaq::ExecutionContext (C++ |     functio                       |
|     cl                            | n)](api/languages/cpp_api.html#_C |
| ass)](api/languages/cpp_api.html# | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| _CPPv4N5cudaq16ExecutionContextE) |                                   |
| -   [cudaq                        |  [\[1\]](api/languages/cpp_api.ht |
| ::ExecutionContext::amplitudeMaps | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     (C++                          |     [\[2\]](api/languages/cpp_    |
|     member)](api/langu            | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [                             |
| ExecutionContext13amplitudeMapsE) | cudaq::QPU::resetExecutionContext |
| -   [c                            |     (C++                          |
| udaq::ExecutionContext::asyncExec |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     member)](api/                 | daq3QPU21resetExecutionContextEv) |
| languages/cpp_api.html#_CPPv4N5cu | -                                 |
| daq16ExecutionContext9asyncExecE) |  [cudaq::QPU::setExecutionContext |
| -   [cud                          |     (C++                          |
| aq::ExecutionContext::asyncResult |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     member)](api/lan              | i.html#_CPPv4N5cudaq3QPU19setExec |
| guages/cpp_api.html#_CPPv4N5cudaq | utionContextEP16ExecutionContext) |
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
| -   [cudaq::ExecutionResult (C++  | languages/cpp_api.html#_CPPv4N5cu |
|     st                            | daq10QuakeValueixERK10QuakeValue) |
| ruct)](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq15ExecutionResultE) |    [cudaq::QuakeValue::QuakeValue |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::appendResult |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     functio                       | akeValue10QuakeValueERN4mlir20Imp |
| n)](api/languages/cpp_api.html#_C | licitLocOpBuilderEN4mlir5ValueE), |
| PPv4N5cudaq15ExecutionResult12app |     [\[1\]                        |
| endResultENSt6stringENSt6size_tE) | ](api/languages/cpp_api.html#_CPP |
| -   [cu                           | v4N5cudaq10QuakeValue10QuakeValue |
| daq::ExecutionResult::deserialize | ERN4mlir20ImplicitLocOpBuilderEd) |
|     (C++                          | -   [cudaq::QuakeValue::size (C++ |
|     function)                     |     funct                         |
| ](api/languages/cpp_api.html#_CPP | ion)](api/languages/cpp_api.html# |
| v4N5cudaq15ExecutionResult11deser | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| ializeERNSt6vectorINSt6size_tEEE) | -   [cudaq::QuakeValue::slice     |
| -   [cudaq:                       |     (C++                          |
| :ExecutionResult::ExecutionResult |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq10QuakeValu |
|     functio                       | e5sliceEKNSt6size_tEKNSt6size_tE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::quantum_platform (C++ |
| PPv4N5cudaq15ExecutionResult15Exe |     cl                            |
| cutionResultE16CountsDictionary), | ass)](api/languages/cpp_api.html# |
|     [\[1\]](api/lan               | _CPPv4N5cudaq16quantum_platformE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cuda                         |
| 15ExecutionResult15ExecutionResul | q::quantum_platform::connectivity |
| tE16CountsDictionaryNSt6stringE), |     (C++                          |
|     [\[2\                         |     function)](api/langu          |
| ]](api/languages/cpp_api.html#_CP | ages/cpp_api.html#_CPPv4N5cudaq16 |
| Pv4N5cudaq15ExecutionResult15Exec | quantum_platform12connectivityEv) |
| utionResultE16CountsDictionaryd), | -   [cudaq::q                     |
|                                   | uantum_platform::enqueueAsyncTask |
|    [\[3\]](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq15ExecutionResu |     function)](api/languages/     |
| lt15ExecutionResultENSt6stringE), | cpp_api.html#_CPPv4N5cudaq16quant |
|     [\[4\                         | um_platform16enqueueAsyncTaskEKNS |
| ]](api/languages/cpp_api.html#_CP | t6size_tER19KernelExecutionTask), |
| Pv4N5cudaq15ExecutionResult15Exec |     [\[1\]](api/languag           |
| utionResultERK15ExecutionResult), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[5\]](api/language          | antum_platform16enqueueAsyncTaskE |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | KNSt6size_tERNSt8functionIFvvEEE) |
| cutionResult15ExecutionResultEd), | -   [cudaq::qua                   |
|     [\[6\]](api/languag           | ntum_platform::get_codegen_config |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |     (C++                          |
| ecutionResult15ExecutionResultEv) |     function)](api/languages/c    |
| -   [                             | pp_api.html#_CPPv4N5cudaq16quantu |
| cudaq::ExecutionResult::operator= | m_platform18get_codegen_configEv) |
|     (C++                          | -   [cudaq::                      |
|     function)](api/languages/     | quantum_platform::get_current_qpu |
| cpp_api.html#_CPPv4N5cudaq15Execu |     (C++                          |
| tionResultaSERK15ExecutionResult) |     function)](api/languages      |
| -   [c                            | /cpp_api.html#_CPPv4NK5cudaq16qua |
| udaq::ExecutionResult::operator== | ntum_platform15get_current_qpuEv) |
|     (C++                          | -   [cuda                         |
|     function)](api/languages/c    | q::quantum_platform::get_exec_ctx |
| pp_api.html#_CPPv4NK5cudaq15Execu |     (C++                          |
| tionResulteqERK15ExecutionResult) |     function)](api/langua         |
| -   [cud                          | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| aq::ExecutionResult::registerName | quantum_platform12get_exec_ctxEv) |
|     (C++                          | -   [c                            |
|     member)](api/lan              | udaq::quantum_platform::get_noise |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult12registerNameE) |     function)](api/languages/c    |
| -   [cudaq                        | pp_api.html#_CPPv4N5cudaq16quantu |
| ::ExecutionResult::sequentialData | m_platform9get_noiseENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     member)](api/langu            | :quantum_platform::get_num_qubits |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| ExecutionResult14sequentialDataE) |                                   |
| -   [                             | function)](api/languages/cpp_api. |
| cudaq::ExecutionResult::serialize | html#_CPPv4NK5cudaq16quantum_plat |
|     (C++                          | form14get_num_qubitsENSt6size_tE) |
|     function)](api/l              | -   [cudaq::quantum_              |
| anguages/cpp_api.html#_CPPv4NK5cu | platform::get_remote_capabilities |
| daq15ExecutionResult9serializeEv) |     (C++                          |
| -   [cudaq::fermion_handler (C++  |     function)                     |
|     c                             | ](api/languages/cpp_api.html#_CPP |
| lass)](api/languages/cpp_api.html | v4NK5cudaq16quantum_platform23get |
| #_CPPv4N5cudaq15fermion_handlerE) | _remote_capabilitiesENSt6size_tE) |
| -   [cudaq::fermion_op (C++       | -   [cudaq::qua                   |
|     type)](api/languages/cpp_api  | ntum_platform::get_runtime_target |
| .html#_CPPv4N5cudaq10fermion_opE) |     (C++                          |
| -   [cudaq::fermion_op_term (C++  |     function)](api/languages/cp   |
|                                   | p_api.html#_CPPv4NK5cudaq16quantu |
| type)](api/languages/cpp_api.html | m_platform18get_runtime_targetEv) |
| #_CPPv4N5cudaq15fermion_op_termE) | -   [cuda                         |
| -   [cudaq::FermioniqBaseQPU (C++ | q::quantum_platform::getLogStream |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     function)](api/langu          |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [cudaq::get_state (C++        | quantum_platform12getLogStreamEv) |
|                                   | -   [cud                          |
|    function)](api/languages/cpp_a | aq::quantum_platform::is_emulated |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     (C++                          |
| ateEDaRR13QuantumKernelDpRR4Args) |                                   |
| -   [cudaq::gradient (C++         |    function)](api/languages/cpp_a |
|     class)](api/languages/cpp_    | pi.html#_CPPv4NK5cudaq16quantum_p |
| api.html#_CPPv4N5cudaq8gradientE) | latform11is_emulatedENSt6size_tE) |
| -   [cudaq::gradient::clone (C++  | -   [c                            |
|     fun                           | udaq::quantum_platform::is_remote |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     function)](api/languages/cp   |
| -   [cudaq::gradient::compute     | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform9is_remoteENSt6size_tE) |
|     function)](api/language       | -   [cuda                         |
| s/cpp_api.html#_CPPv4N5cudaq8grad | q::quantum_platform::is_simulator |
| ient7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |                                   |
|     [\[1\]](ap                    |   function)](api/languages/cpp_ap |
| i/languages/cpp_api.html#_CPPv4N5 | i.html#_CPPv4NK5cudaq16quantum_pl |
| cudaq8gradient7computeERKNSt6vect | atform12is_simulatorENSt6size_tE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [c                            |
| -   [cudaq::gradient::gradient    | udaq::quantum_platform::launchVQE |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](                   |
| uages/cpp_api.html#_CPPv4I00EN5cu | api/languages/cpp_api.html#_CPPv4 |
| daq8gradient8gradientER7KernelT), | N5cudaq16quantum_platform9launchV |
|                                   | QEEKNSt6stringEPKvPN5cudaq8gradie |
|    [\[1\]](api/languages/cpp_api. | ntERKN5cudaq7spin_opERN5cudaq9opt |
| html#_CPPv4I00EN5cudaq8gradient8g | imizerEKiKNSt6size_tENSt6size_tE) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq:                       |
|     [\[2\                         | :quantum_platform::list_platforms |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4I00EN5cudaq8gradient8gradientE |     function)](api/languag        |
| RR13QuantumKernelRR10ArgsMapper), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[3                          | antum_platform14list_platformsEv) |
| \]](api/languages/cpp_api.html#_C | -                                 |
| PPv4N5cudaq8gradient8gradientERRN |    [cudaq::quantum_platform::name |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[                           |     function)](a                  |
| 4\]](api/languages/cpp_api.html#_ | pi/languages/cpp_api.html#_CPPv4N |
| CPPv4N5cudaq8gradient8gradientEv) | K5cudaq16quantum_platform4nameEv) |
| -   [cudaq::gradient::setArgs     | -   [                             |
|     (C++                          | cudaq::quantum_platform::num_qpus |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/l              |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | anguages/cpp_api.html#_CPPv4NK5cu |
| tArgsEvR13QuantumKernelDpRR4Args) | daq16quantum_platform8num_qpusEv) |
| -   [cudaq::gradient::setKernel   | -   [cudaq::                      |
|     (C++                          | quantum_platform::onRandomSeedSet |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4I0EN5cudaq8grad |                                   |
| ient9setKernelEvR13QuantumKernel) | function)](api/languages/cpp_api. |
| -   [cud                          | html#_CPPv4N5cudaq16quantum_platf |
| aq::gradients::central_difference | orm15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     class)](api/la                | :quantum_platform::reset_exec_ctx |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9gradients18central_differenceE) |     function)](api/languag        |
| -   [cudaq::gra                   | es/cpp_api.html#_CPPv4N5cudaq16qu |
| dients::central_difference::clone | antum_platform14reset_exec_ctxEv) |
|     (C++                          | -   [cud                          |
|     function)](api/languages      | aq::quantum_platform::reset_noise |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18central_difference5cloneEv) |     function)](api/languages/cpp_ |
| -   [cudaq::gradi                 | api.html#_CPPv4N5cudaq16quantum_p |
| ents::central_difference::compute | latform11reset_noiseENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     function)](                   | :quantum_platform::resetLogStream |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq9gradients18central_differ |     function)](api/languag        |
| ence7computeERKNSt6vectorIdEERKNS | es/cpp_api.html#_CPPv4N5cudaq16qu |
| t8functionIFdNSt6vectorIdEEEEEd), | antum_platform14resetLogStreamEv) |
|                                   | -   [cuda                         |
|   [\[1\]](api/languages/cpp_api.h | q::quantum_platform::set_exec_ctx |
| tml#_CPPv4N5cudaq9gradients18cent |     (C++                          |
| ral_difference7computeERKNSt6vect |     funct                         |
| orIdEERNSt6vectorIdEERK7spin_opd) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::gradie                | _CPPv4N5cudaq16quantum_platform12 |
| nts::central_difference::gradient | set_exec_ctxEP16ExecutionContext) |
|     (C++                          | -   [c                            |
|     functio                       | udaq::quantum_platform::set_noise |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4I00EN5cudaq9gradients18centra |     function                      |
| l_difference8gradientER7KernelT), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/langua            | Pv4N5cudaq16quantum_platform9set_ |
| ges/cpp_api.html#_CPPv4I00EN5cuda | noiseEPK11noise_modelNSt6size_tE) |
| q9gradients18central_difference8g | -   [cuda                         |
| radientER7KernelTRR10ArgsMapper), | q::quantum_platform::setLogStream |
|     [\[2\]](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4I00EN5cudaq9gradie |                                   |
| nts18central_difference8gradientE |  function)](api/languages/cpp_api |
| RR13QuantumKernelRR10ArgsMapper), | .html#_CPPv4N5cudaq16quantum_plat |
|     [\[3\]](api/languages/cpp     | form12setLogStreamERNSt7ostreamE) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::q                     |
| 18central_difference8gradientERRN | uantum_platform::setTargetBackend |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[4\]](api/languages/cp      |     fun                           |
| p_api.html#_CPPv4N5cudaq9gradient | ction)](api/languages/cpp_api.htm |
| s18central_difference8gradientEv) | l#_CPPv4N5cudaq16quantum_platform |
| -   [cud                          | 16setTargetBackendERKNSt6stringE) |
| aq::gradients::forward_difference | -   [cudaq::quantum_platfo        |
|     (C++                          | rm::supports_conditional_feedback |
|     class)](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/               |
| q9gradients18forward_differenceE) | languages/cpp_api.html#_CPPv4NK5c |
| -   [cudaq::gra                   | udaq16quantum_platform29supports_ |
| dients::forward_difference::clone | conditional_feedbackENSt6size_tE) |
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
| tml#_CPPv4N5cudaq9gradients18forw | -   [cudaq::QuantumTask (C++      |
| ard_difference7computeERKNSt6vect |     type)](api/languages/cpp_api. |
| orIdEERNSt6vectorIdEERK7spin_opd) | html#_CPPv4N5cudaq11QuantumTaskE) |
| -   [cudaq::gradie                | -   [cudaq::qubit (C++            |
| nts::forward_difference::gradient |     type)](api/languages/c        |
|     (C++                          | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     functio                       | -   [cudaq::QubitConnectivity     |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4I00EN5cudaq9gradients18forwar |     ty                            |
| d_difference8gradientER7KernelT), | pe)](api/languages/cpp_api.html#_ |
|     [\[1\]](api/langua            | CPPv4N5cudaq17QubitConnectivityE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::QubitEdge (C++        |
| q9gradients18forward_difference8g |     type)](api/languages/cpp_a    |
| radientER7KernelTRR10ArgsMapper), | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::qudit (C++            |
| api.html#_CPPv4I00EN5cudaq9gradie |     clas                          |
| nts18forward_difference8gradientE | s)](api/languages/cpp_api.html#_C |
| RR13QuantumKernelRR10ArgsMapper), | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::qudit::qudit (C++     |
| _api.html#_CPPv4N5cudaq9gradients |                                   |
| 18forward_difference8gradientERRN | function)](api/languages/cpp_api. |
| St8functionIFvNSt6vectorIdEEEEE), | html#_CPPv4N5cudaq5qudit5quditEv) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qvector (C++          |
| p_api.html#_CPPv4N5cudaq9gradient |     class)                        |
| s18forward_difference8gradientEv) | ](api/languages/cpp_api.html#_CPP |
| -   [                             | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| cudaq::gradients::parameter_shift | -   [cudaq::qvector::back (C++    |
|     (C++                          |     function)](a                  |
|     class)](api                   | pi/languages/cpp_api.html#_CPPv4N |
| /languages/cpp_api.html#_CPPv4N5c | 5cudaq7qvector4backENSt6size_tE), |
| udaq9gradients15parameter_shiftE) |                                   |
| -   [cudaq::                      |   [\[1\]](api/languages/cpp_api.h |
| gradients::parameter_shift::clone | tml#_CPPv4N5cudaq7qvector4backEv) |
|     (C++                          | -   [cudaq::qvector::begin (C++   |
|     function)](api/langua         |     fu                            |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | nction)](api/languages/cpp_api.ht |
| adients15parameter_shift5cloneEv) | ml#_CPPv4N5cudaq7qvector5beginEv) |
| -   [cudaq::gr                    | -   [cudaq::qvector::clear (C++   |
| adients::parameter_shift::compute |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function                      | ml#_CPPv4N5cudaq7qvector5clearEv) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qvector::end (C++     |
| Pv4N5cudaq9gradients15parameter_s |                                   |
| hift7computeERKNSt6vectorIdEERKNS | function)](api/languages/cpp_api. |
| t8functionIFdNSt6vectorIdEEEEEd), | html#_CPPv4N5cudaq7qvector3endEv) |
|     [\[1\]](api/languages/cpp_ap  | -   [cudaq::qvector::front (C++   |
| i.html#_CPPv4N5cudaq9gradients15p |     function)](ap                 |
| arameter_shift7computeERKNSt6vect | i/languages/cpp_api.html#_CPPv4N5 |
| orIdEERNSt6vectorIdEERK7spin_opd) | cudaq7qvector5frontENSt6size_tE), |
| -   [cudaq::gra                   |                                   |
| dients::parameter_shift::gradient |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     func                          | -   [cudaq::qvector::operator=    |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4I00EN5cudaq9gradients15par |     functio                       |
| ameter_shift8gradientER7KernelT), | n)](api/languages/cpp_api.html#_C |
|     [\[1\]](api/lan               | PPv4N5cudaq7qvectoraSERK7qvector) |
| guages/cpp_api.html#_CPPv4I00EN5c | -   [cudaq::qvector::operator\[\] |
| udaq9gradients15parameter_shift8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)                     |
|     [\[2\]](api/languages/c       | ](api/languages/cpp_api.html#_CPP |
| pp_api.html#_CPPv4I00EN5cudaq9gra | v4N5cudaq7qvectorixEKNSt6size_tE) |
| dients15parameter_shift8gradientE | -   [cudaq::qvector::qvector (C++ |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/               |
|     [\[3\]](api/languages/        | languages/cpp_api.html#_CPPv4N5cu |
| cpp_api.html#_CPPv4N5cudaq9gradie | daq7qvector7qvectorENSt6size_tE), |
| nts15parameter_shift8gradientERRN |     [\[1\]](a                     |
| St8functionIFvNSt6vectorIdEEEEE), | pi/languages/cpp_api.html#_CPPv4N |
|     [\[4\]](api/languages         | 5cudaq7qvector7qvectorERK5state), |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     [\[2\]](api                   |
| ents15parameter_shift8gradientEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::kernel_builder (C++   | udaq7qvector7qvectorERK7qvector), |
|     clas                          |     [\[3\]](api/languages/cpp     |
| s)](api/languages/cpp_api.html#_C | _api.html#_CPPv4N5cudaq7qvector7q |
| PPv4IDpEN5cudaq14kernel_builderE) | vectorERKNSt6vectorI7complexEEb), |
| -   [c                            |     [\[4\]](ap                    |
| udaq::kernel_builder::constantVal | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq7qvector7qvectorERR7qvector) |
|     function)](api/la             | -   [cudaq::qvector::size (C++    |
| nguages/cpp_api.html#_CPPv4N5cuda |     fu                            |
| q14kernel_builder11constantValEd) | nction)](api/languages/cpp_api.ht |
| -   [cu                           | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| daq::kernel_builder::getArguments | -   [cudaq::qvector::slice (C++   |
|     (C++                          |     function)](api/language       |
|     function)](api/lan            | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| guages/cpp_api.html#_CPPv4N5cudaq | tor5sliceENSt6size_tENSt6size_tE) |
| 14kernel_builder12getArgumentsEv) | -   [cudaq::qvector::value_type   |
| -   [cu                           |     (C++                          |
| daq::kernel_builder::getNumParams |     typ                           |
|     (C++                          | e)](api/languages/cpp_api.html#_C |
|     function)](api/lan            | PPv4N5cudaq7qvector10value_typeE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qview (C++            |
| 14kernel_builder12getNumParamsEv) |     clas                          |
| -   [c                            | s)](api/languages/cpp_api.html#_C |
| udaq::kernel_builder::isArgStdVec | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     (C++                          | -   [cudaq::qview::back (C++      |
|     function)](api/languages/cp   |     function)                     |
| p_api.html#_CPPv4N5cudaq14kernel_ | ](api/languages/cpp_api.html#_CPP |
| builder11isArgStdVecENSt6size_tE) | v4N5cudaq5qview4backENSt6size_tE) |
| -   [cuda                         | -   [cudaq::qview::begin (C++     |
| q::kernel_builder::kernel_builder |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function)](api/languages/cpp_ | html#_CPPv4N5cudaq5qview5beginEv) |
| api.html#_CPPv4N5cudaq14kernel_bu | -   [cudaq::qview::end (C++       |
| ilder14kernel_builderERNSt6vector |                                   |
| IN7details17KernelBuilderTypeEEE) |   function)](api/languages/cpp_ap |
| -   [cudaq::kernel_builder::name  | i.html#_CPPv4N5cudaq5qview3endEv) |
|     (C++                          | -   [cudaq::qview::front (C++     |
|     function)                     |     function)](                   |
| ](api/languages/cpp_api.html#_CPP | api/languages/cpp_api.html#_CPPv4 |
| v4N5cudaq14kernel_builder4nameEv) | N5cudaq5qview5frontENSt6size_tE), |
| -                                 |                                   |
|    [cudaq::kernel_builder::qalloc |    [\[1\]](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qview5frontEv) |
|     function)](api/language       | -   [cudaq::qview::operator\[\]   |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     (C++                          |
| nel_builder6qallocE10QuakeValue), |     functio                       |
|     [\[1\]](api/language          | n)](api/languages/cpp_api.html#_C |
| s/cpp_api.html#_CPPv4N5cudaq14ker | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| nel_builder6qallocEKNSt6size_tE), | -   [cudaq::qview::qview (C++     |
|     [\[2                          |     functio                       |
| \]](api/languages/cpp_api.html#_C | n)](api/languages/cpp_api.html#_C |
| PPv4N5cudaq14kernel_builder6qallo | PPv4I0EN5cudaq5qview5qviewERR1R), |
| cERNSt6vectorINSt7complexIdEEEE), |     [\[1                          |
|     [\[3\]](                      | \]](api/languages/cpp_api.html#_C |
| api/languages/cpp_api.html#_CPPv4 | PPv4N5cudaq5qview5qviewERK5qview) |
| N5cudaq14kernel_builder6qallocEv) | -   [cudaq::qview::size (C++      |
| -   [cudaq::kernel_builder::swap  |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function)](api/language       | html#_CPPv4NK5cudaq5qview4sizeEv) |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | -   [cudaq::qview::slice (C++     |
| 4kernel_builder4swapEvRK10QuakeVa |     function)](api/langua         |
| lueRK10QuakeValueRK10QuakeValue), | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|                                   | iew5sliceENSt6size_tENSt6size_tE) |
| [\[1\]](api/languages/cpp_api.htm | -   [cudaq::qview::value_type     |
| l#_CPPv4I00EN5cudaq14kernel_build |     (C++                          |
| er4swapEvRKNSt6vectorI10QuakeValu |     t                             |
| eEERK10QuakeValueRK10QuakeValue), | ype)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq5qview10value_typeE) |
| [\[2\]](api/languages/cpp_api.htm | -   [cudaq::range (C++            |
| l#_CPPv4N5cudaq14kernel_builder4s |     fun                           |
| wapERK10QuakeValueRK10QuakeValue) | ction)](api/languages/cpp_api.htm |
| -   [cudaq::KernelExecutionTask   | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
|     (C++                          | orI11ElementTypeEE11ElementType), |
|     type                          |     [\[1\]](api/languages/cpp_    |
| )](api/languages/cpp_api.html#_CP | api.html#_CPPv4I0EN5cudaq5rangeEN |
| Pv4N5cudaq19KernelExecutionTaskE) | St6vectorI11ElementTypeEE11Elemen |
| -   [cudaq::KernelThunkResultType | tType11ElementType11ElementType), |
|     (C++                          |     [                             |
|     struct)]                      | \[2\]](api/languages/cpp_api.html |
| (api/languages/cpp_api.html#_CPPv | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| 4N5cudaq21KernelThunkResultTypeE) | -   [cudaq::real (C++             |
| -   [cudaq::KernelThunkType (C++  |     type)](api/languages/         |
|                                   | cpp_api.html#_CPPv4N5cudaq4realE) |
| type)](api/languages/cpp_api.html | -   [cudaq::registry (C++         |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     type)](api/languages/cpp_     |
| -   [cudaq::kraus_channel (C++    | api.html#_CPPv4N5cudaq8registryE) |
|                                   | -                                 |
|  class)](api/languages/cpp_api.ht |  [cudaq::registry::RegisteredType |
| ml#_CPPv4N5cudaq13kraus_channelE) |     (C++                          |
| -   [cudaq::kraus_channel::empty  |     class)](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)]                    | 5cudaq8registry14RegisteredTypeE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::RemoteCapabilities    |
| 4NK5cudaq13kraus_channel5emptyEv) |     (C++                          |
| -   [cudaq::kraus_c               |     struc                         |
| hannel::generateUnitaryParameters | t)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq18RemoteCapabilitiesE) |
|                                   | -   [cudaq::Remo                  |
|    function)](api/languages/cpp_a | teCapabilities::isRemoteSimulator |
| pi.html#_CPPv4N5cudaq13kraus_chan |     (C++                          |
| nel25generateUnitaryParametersEv) |     member)](api/languages/c      |
| -                                 | pp_api.html#_CPPv4N5cudaq18Remote |
|    [cudaq::kraus_channel::get_ops | Capabilities17isRemoteSimulatorE) |
|     (C++                          | -   [cudaq::Remot                 |
|     function)](a                  | eCapabilities::RemoteCapabilities |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| K5cudaq13kraus_channel7get_opsEv) |     function)](api/languages/cpp  |
| -   [cudaq::                      | _api.html#_CPPv4N5cudaq18RemoteCa |
| kraus_channel::is_unitary_mixture | pabilities18RemoteCapabilitiesEb) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/languages      | :RemoteCapabilities::stateOverlap |
| /cpp_api.html#_CPPv4NK5cudaq13kra |     (C++                          |
| us_channel18is_unitary_mixtureEv) |     member)](api/langua           |
| -   [cu                           | ges/cpp_api.html#_CPPv4N5cudaq18R |
| daq::kraus_channel::kraus_channel | emoteCapabilities12stateOverlapE) |
|     (C++                          | -                                 |
|     function)](api/lang           |   [cudaq::RemoteCapabilities::vqe |
| uages/cpp_api.html#_CPPv4IDpEN5cu |     (C++                          |
| daq13kraus_channel13kraus_channel |     member)](                     |
| EDpRRNSt16initializer_listI1TEE), | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq18RemoteCapabilities3vqeE) |
|  [\[1\]](api/languages/cpp_api.ht | -   [cudaq::RemoteSimulationState |
| ml#_CPPv4N5cudaq13kraus_channel13 |     (C++                          |
| kraus_channelERK13kraus_channel), |     class)]                       |
|     [\[2\]                        | (api/languages/cpp_api.html#_CPPv |
| ](api/languages/cpp_api.html#_CPP | 4N5cudaq21RemoteSimulationStateE) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::Resources (C++        |
| hannelERKNSt6vectorI8kraus_opEE), |     class)](api/languages/cpp_a   |
|     [\[3\]                        | pi.html#_CPPv4N5cudaq9ResourcesE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::run (C++              |
| v4N5cudaq13kraus_channel13kraus_c |     function)]                    |
| hannelERRNSt6vectorI8kraus_opEE), | (api/languages/cpp_api.html#_CPPv |
|     [\[4\]](api/lan               | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| guages/cpp_api.html#_CPPv4N5cudaq | 5invoke_result_tINSt7decay_tI13Qu |
| 13kraus_channel13kraus_channelEv) | antumKernelEEDpNSt7decay_tI4ARGSE |
| -                                 | EEEEENSt6size_tERN5cudaq11noise_m |
| [cudaq::kraus_channel::noise_type | odelERR13QuantumKernelDpRR4ARGS), |
|     (C++                          |     [\[1\]](api/langu             |
|     member)](api                  | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| /languages/cpp_api.html#_CPPv4N5c | daq3runENSt6vectorINSt15invoke_re |
| udaq13kraus_channel10noise_typeE) | sult_tINSt7decay_tI13QuantumKerne |
| -                                 | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
|  [cudaq::kraus_channel::operator= | ize_tERR13QuantumKernelDpRR4ARGS) |
|     (C++                          | -   [cudaq::run_async (C++        |
|     function)](api/langua         |     functio                       |
| ges/cpp_api.html#_CPPv4N5cudaq13k | n)](api/languages/cpp_api.html#_C |
| raus_channelaSERK13kraus_channel) | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| -   [c                            | tureINSt6vectorINSt15invoke_resul |
| udaq::kraus_channel::operator\[\] | t_tINSt7decay_tI13QuantumKernelEE |
|     (C++                          | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|     function)](api/l              | ze_tENSt6size_tERN5cudaq11noise_m |
| anguages/cpp_api.html#_CPPv4N5cud | odelERR13QuantumKernelDpRR4ARGS), |
| aq13kraus_channelixEKNSt6size_tE) |     [\[1\]](api/la                |
| -                                 | nguages/cpp_api.html#_CPPv4I0DpEN |
| [cudaq::kraus_channel::parameters | 5cudaq9run_asyncENSt6futureINSt6v |
|     (C++                          | ectorINSt15invoke_result_tINSt7de |
|     member)](api                  | cay_tI13QuantumKernelEEDpNSt7deca |
| /languages/cpp_api.html#_CPPv4N5c | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| udaq13kraus_channel10parametersE) | ize_tERR13QuantumKernelDpRR4ARGS) |
| -   [cu                           | -   [cudaq::RuntimeTarget (C++    |
| daq::kraus_channel::probabilities |                                   |
|     (C++                          | struct)](api/languages/cpp_api.ht |
|     member)](api/la               | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::sample (C++           |
| q13kraus_channel13probabilitiesE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|  [cudaq::kraus_channel::push_back | mpleE13sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     function)](api/langua         |     [\[1\                         |
| ges/cpp_api.html#_CPPv4N5cudaq13k | ]](api/languages/cpp_api.html#_CP |
| raus_channel9push_backE8kraus_op) | Pv4I0DpEN5cudaq6sampleE13sample_r |
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
|     (C++                          |     function)](api/languages/cpp  |
|     functi                        | _api.html#_CPPv4N5cudaq13sample_r |
| on)](api/languages/cpp_api.html#_ | esult6appendER15ExecutionResultb) |
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
| N5cudaq14matrix_handler11instanti | ample_resultaSER13sample_result), |
| ateENSt6stringERRNSt6vectorINSt6s |     [\[1\]](api/langua            |
| ize_tEEERK20commutation_behavior) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -   [cuda                         | ample_resultaSERR13sample_result) |
| q::matrix_handler::matrix_handler | -                                 |
|     (C++                          | [cudaq::sample_result::operator== |
|     function)](api/languag        |     (C++                          |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     function)](api/languag        |
| ble_if_tINSt12is_base_of_vI16oper | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ator_handler1TEEbEEEN5cudaq14matr | ample_resulteqERK13sample_result) |
| ix_handler14matrix_handlerERK1T), | -   [                             |
|     [\[1\]](ap                    | cudaq::sample_result::probability |
| i/languages/cpp_api.html#_CPPv4I0 |     (C++                          |
| _NSt11enable_if_tINSt12is_base_of |     function)](api/lan            |
| _vI16operator_handler1TEEbEEEN5cu | guages/cpp_api.html#_CPPv4NK5cuda |
| daq14matrix_handler14matrix_handl | q13sample_result11probabilityENSt |
| erERK1TRK20commutation_behavior), | 11string_viewEKNSt11string_viewE) |
|     [\[2\]](api/languages/cpp_ap  | -   [cud                          |
| i.html#_CPPv4N5cudaq14matrix_hand | aq::sample_result::register_names |
| ler14matrix_handlerENSt6size_tE), |     (C++                          |
|     [\[3\]](api/                  |     function)](api/langu          |
| languages/cpp_api.html#_CPPv4N5cu | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| daq14matrix_handler14matrix_handl | 3sample_result14register_namesEv) |
| erENSt6stringERKNSt6vectorINSt6si | -                                 |
| ze_tEEERK20commutation_behavior), |    [cudaq::sample_result::reorder |
|     [\[4\]](api/                  |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/langua         |
| daq14matrix_handler14matrix_handl | ges/cpp_api.html#_CPPv4N5cudaq13s |
| erENSt6stringERRNSt6vectorINSt6si | ample_result7reorderERKNSt6vector |
| ze_tEEERK20commutation_behavior), | INSt6size_tEEEKNSt11string_viewE) |
|     [\                            | -   [cu                           |
| [5\]](api/languages/cpp_api.html# | daq::sample_result::sample_result |
| _CPPv4N5cudaq14matrix_handler14ma |     (C++                          |
| trix_handlerERK14matrix_handler), |     fun                           |
|     [                             | ction)](api/languages/cpp_api.htm |
| \[6\]](api/languages/cpp_api.html | l#_CPPv4N5cudaq13sample_result13s |
| #_CPPv4N5cudaq14matrix_handler14m | ample_resultER15ExecutionResult), |
| atrix_handlerERR14matrix_handler) |                                   |
| -                                 |  [\[1\]](api/languages/cpp_api.ht |
|  [cudaq::matrix_handler::momentum | ml#_CPPv4N5cudaq13sample_result13 |
|     (C++                          | sample_resultERK13sample_result), |
|     function)](api/language       |     [\[2\]](api/l                 |
| s/cpp_api.html#_CPPv4N5cudaq14mat | anguages/cpp_api.html#_CPPv4N5cud |
| rix_handler8momentumENSt6size_tE) | aq13sample_result13sample_resultE |
| -                                 | RNSt6vectorI15ExecutionResultEE), |
|    [cudaq::matrix_handler::number |                                   |
|     (C++                          |  [\[3\]](api/languages/cpp_api.ht |
|     function)](api/langua         | ml#_CPPv4N5cudaq13sample_result13 |
| ges/cpp_api.html#_CPPv4N5cudaq14m | sample_resultERR13sample_result), |
| atrix_handler6numberENSt6size_tE) |     [                             |
| -                                 | \[4\]](api/languages/cpp_api.html |
| [cudaq::matrix_handler::operator= | #_CPPv4N5cudaq13sample_result13sa |
|     (C++                          | mple_resultERR15ExecutionResult), |
|     fun                           |     [\[5\]](api/la                |
| ction)](api/languages/cpp_api.htm | nguages/cpp_api.html#_CPPv4N5cuda |
| l#_CPPv4I0_NSt11enable_if_tIXaant | q13sample_result13sample_resultEd |
| NSt7is_sameI1T14matrix_handlerE5v | RNSt6vectorI15ExecutionResultEE), |
| alueENSt12is_base_of_vI16operator |     [\[6\]](api/lan               |
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
|                                   |     [\[1                          |
|                                   | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq5state5stateERK5state) |
|                                   | -   [cudaq::state::to_host (C++   |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | I0ENK5cudaq5state7to_hostEvPNSt7c |
|                                   | omplexI10ScalarTypeEENSt6size_tE) |
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
| -   [define() (cudaq.operators    | -   [displace() (in module        |
|     method)](api/languages/python |     cudaq.operators.custo         |
| _api.html#cudaq.operators.define) | m)](api/languages/python_api.html |
|     -   [(cuda                    | #cudaq.operators.custom.displace) |
| q.operators.MatrixOperatorElement | -   [distribute_terms()           |
|         class                     |     (cu                           |
|         method)](api/langu        | daq.operators.boson.BosonOperator |
| ages/python_api.html#cudaq.operat |     method)](api/languages/pyt    |
| ors.MatrixOperatorElement.define) | hon_api.html#cudaq.operators.boso |
|     -   [(in module               | n.BosonOperator.distribute_terms) |
|         cudaq.operators.cus       |     -   [(cudaq.                  |
| tom)](api/languages/python_api.ht | operators.fermion.FermionOperator |
| ml#cudaq.operators.custom.define) |                                   |
| -   [degrees                      |    method)](api/languages/python_ |
|     (cu                           | api.html#cudaq.operators.fermion. |
| daq.operators.boson.BosonOperator | FermionOperator.distribute_terms) |
|     property)](api/lang           |     -                             |
| uages/python_api.html#cudaq.opera |  [(cudaq.operators.MatrixOperator |
| tors.boson.BosonOperator.degrees) |         method)](api/language     |
|     -   [(cudaq.ope               | s/python_api.html#cudaq.operators |
| rators.boson.BosonOperatorElement | .MatrixOperator.distribute_terms) |
|                                   |     -   [(                        |
|        property)](api/languages/p | cudaq.operators.spin.SpinOperator |
| ython_api.html#cudaq.operators.bo |         method)](api/languages/p  |
| son.BosonOperatorElement.degrees) | ython_api.html#cudaq.operators.sp |
|     -   [(cudaq.                  | in.SpinOperator.distribute_terms) |
| operators.boson.BosonOperatorTerm |     -   [(cuda                    |
|         property)](api/language   | q.operators.spin.SpinOperatorTerm |
| s/python_api.html#cudaq.operators |                                   |
| .boson.BosonOperatorTerm.degrees) |      method)](api/languages/pytho |
|     -   [(cudaq.                  | n_api.html#cudaq.operators.spin.S |
| operators.fermion.FermionOperator | pinOperatorTerm.distribute_terms) |
|         property)](api/language   | -   [draw() (in module            |
| s/python_api.html#cudaq.operators |     cudaq)](api/lang              |
| .fermion.FermionOperator.degrees) | uages/python_api.html#cudaq.draw) |
|     -   [(cudaq.operato           | -   [dump() (cudaq.ComplexMatrix  |
| rs.fermion.FermionOperatorElement |                                   |
|                                   |   method)](api/languages/python_a |
|    property)](api/languages/pytho | pi.html#cudaq.ComplexMatrix.dump) |
| n_api.html#cudaq.operators.fermio |     -   [(cudaq.ObserveResult     |
| n.FermionOperatorElement.degrees) |                                   |
|     -   [(cudaq.oper              |   method)](api/languages/python_a |
| ators.fermion.FermionOperatorTerm | pi.html#cudaq.ObserveResult.dump) |
|                                   |     -   [(cu                      |
|       property)](api/languages/py | daq.operators.boson.BosonOperator |
| thon_api.html#cudaq.operators.fer |         method)](api/l            |
| mion.FermionOperatorTerm.degrees) | anguages/python_api.html#cudaq.op |
|     -                             | erators.boson.BosonOperator.dump) |
|  [(cudaq.operators.MatrixOperator |     -   [(cudaq.                  |
|         property)](api            | operators.boson.BosonOperatorTerm |
| /languages/python_api.html#cudaq. |         method)](api/langu        |
| operators.MatrixOperator.degrees) | ages/python_api.html#cudaq.operat |
|     -   [(cuda                    | ors.boson.BosonOperatorTerm.dump) |
| q.operators.MatrixOperatorElement |     -   [(cudaq.                  |
|         property)](api/langua     | operators.fermion.FermionOperator |
| ges/python_api.html#cudaq.operato |         method)](api/langu        |
| rs.MatrixOperatorElement.degrees) | ages/python_api.html#cudaq.operat |
|     -   [(c                       | ors.fermion.FermionOperator.dump) |
| udaq.operators.MatrixOperatorTerm |     -   [(cudaq.oper              |
|         property)](api/lan        | ators.fermion.FermionOperatorTerm |
| guages/python_api.html#cudaq.oper |         method)](api/languages    |
| ators.MatrixOperatorTerm.degrees) | /python_api.html#cudaq.operators. |
|     -   [(                        | fermion.FermionOperatorTerm.dump) |
| cudaq.operators.spin.SpinOperator |     -                             |
|         property)](api/la         |  [(cudaq.operators.MatrixOperator |
| nguages/python_api.html#cudaq.ope |         method)](                 |
| rators.spin.SpinOperator.degrees) | api/languages/python_api.html#cud |
|     -   [(cudaq.o                 | aq.operators.MatrixOperator.dump) |
| perators.spin.SpinOperatorElement |     -   [(c                       |
|         property)](api/languages  | udaq.operators.MatrixOperatorTerm |
| /python_api.html#cudaq.operators. |         method)](api/             |
| spin.SpinOperatorElement.degrees) | languages/python_api.html#cudaq.o |
|     -   [(cuda                    | perators.MatrixOperatorTerm.dump) |
| q.operators.spin.SpinOperatorTerm |     -   [(                        |
|         property)](api/langua     | cudaq.operators.spin.SpinOperator |
| ges/python_api.html#cudaq.operato |         method)](api              |
| rs.spin.SpinOperatorTerm.degrees) | /languages/python_api.html#cudaq. |
| -   [Depolarization1 (class in    | operators.spin.SpinOperator.dump) |
|     cudaq)](api/languages/pytho   |     -   [(cuda                    |
| n_api.html#cudaq.Depolarization1) | q.operators.spin.SpinOperatorTerm |
| -   [Depolarization2 (class in    |         method)](api/lan          |
|     cudaq)](api/languages/pytho   | guages/python_api.html#cudaq.oper |
| n_api.html#cudaq.Depolarization2) | ators.spin.SpinOperatorTerm.dump) |
| -   [DepolarizationChannel (class |     -   [(cudaq.Resources         |
|     in                            |                                   |
|                                   |       method)](api/languages/pyth |
| cudaq)](api/languages/python_api. | on_api.html#cudaq.Resources.dump) |
| html#cudaq.DepolarizationChannel) |     -   [(cudaq.SampleResult      |
| -   [description (cudaq.Target    |                                   |
|                                   |    method)](api/languages/python_ |
| property)](api/languages/python_a | api.html#cudaq.SampleResult.dump) |
| pi.html#cudaq.Target.description) |     -   [(cudaq.State             |
| -   [deserialize()                |         method)](api/languages/   |
|     (cudaq.SampleResult           | python_api.html#cudaq.State.dump) |
|     meth                          |                                   |
| od)](api/languages/python_api.htm |                                   |
| l#cudaq.SampleResult.deserialize) |                                   |
+-----------------------------------+-----------------------------------+

## E {#E}

+-----------------------------------+-----------------------------------+
| -   [ElementaryOperator (in       | -   [evaluate_coefficient()       |
|     module                        |     (cudaq.                       |
|     cudaq.operators)]             | operators.boson.BosonOperatorTerm |
| (api/languages/python_api.html#cu |     m                             |
| daq.operators.ElementaryOperator) | ethod)](api/languages/python_api. |
| -   [empty()                      | html#cudaq.operators.boson.BosonO |
|     (cu                           | peratorTerm.evaluate_coefficient) |
| daq.operators.boson.BosonOperator |     -   [(cudaq.oper              |
|     static                        | ators.fermion.FermionOperatorTerm |
|     method)](api/la               |         metho                     |
| nguages/python_api.html#cudaq.ope | d)](api/languages/python_api.html |
| rators.boson.BosonOperator.empty) | #cudaq.operators.fermion.FermionO |
|     -   [(cudaq.                  | peratorTerm.evaluate_coefficient) |
| operators.fermion.FermionOperator |     -   [(c                       |
|         static                    | udaq.operators.MatrixOperatorTerm |
|         method)](api/langua       |                                   |
| ges/python_api.html#cudaq.operato |     method)](api/languages/python |
| rs.fermion.FermionOperator.empty) | _api.html#cudaq.operators.MatrixO |
|     -                             | peratorTerm.evaluate_coefficient) |
|  [(cudaq.operators.MatrixOperator |     -   [(cuda                    |
|         static                    | q.operators.spin.SpinOperatorTerm |
|         method)](a                |                                   |
| pi/languages/python_api.html#cuda |  method)](api/languages/python_ap |
| q.operators.MatrixOperator.empty) | i.html#cudaq.operators.spin.SpinO |
|     -   [(                        | peratorTerm.evaluate_coefficient) |
| cudaq.operators.spin.SpinOperator | -   [evolve() (in module          |
|         static                    |     cudaq)](api/langua            |
|         method)](api/             | ges/python_api.html#cudaq.evolve) |
| languages/python_api.html#cudaq.o | -   [evolve_async() (in module    |
| perators.spin.SpinOperator.empty) |     cudaq)](api/languages/py      |
|     -   [(in module               | thon_api.html#cudaq.evolve_async) |
|                                   | -   [EvolveResult (class in       |
|     cudaq.boson)](api/languages/p |     cudaq)](api/languages/py      |
| ython_api.html#cudaq.boson.empty) | thon_api.html#cudaq.EvolveResult) |
|     -   [(in module               | -   [expectation()                |
|                                   |     (cudaq.ObserveResult          |
| cudaq.fermion)](api/languages/pyt |     metho                         |
| hon_api.html#cudaq.fermion.empty) | d)](api/languages/python_api.html |
|     -   [(in module               | #cudaq.ObserveResult.expectation) |
|         cudaq.operators.cu        |     -   [(cudaq.SampleResult      |
| stom)](api/languages/python_api.h |         meth                      |
| tml#cudaq.operators.custom.empty) | od)](api/languages/python_api.htm |
|     -   [(in module               | l#cudaq.SampleResult.expectation) |
|                                   | -   [expectation_values()         |
|       cudaq.spin)](api/languages/ |     (cudaq.EvolveResult           |
| python_api.html#cudaq.spin.empty) |     method)](ap                   |
| -   [empty_op()                   | i/languages/python_api.html#cudaq |
|     (                             | .EvolveResult.expectation_values) |
| cudaq.operators.spin.SpinOperator | -   [expectation_z()              |
|     static                        |     (cudaq.SampleResult           |
|     method)](api/lan              |     method                        |
| guages/python_api.html#cudaq.oper | )](api/languages/python_api.html# |
| ators.spin.SpinOperator.empty_op) | cudaq.SampleResult.expectation_z) |
| -   [enable_return_to_log()       | -   [expected_dimensions          |
|     (cudaq.PyKernelDecorator      |     (cuda                         |
|     method)](api/langu            | q.operators.MatrixOperatorElement |
| ages/python_api.html#cudaq.PyKern |                                   |
| elDecorator.enable_return_to_log) | property)](api/languages/python_a |
| -   [estimate_resources() (in     | pi.html#cudaq.operators.MatrixOpe |
|     module                        | ratorElement.expected_dimensions) |
|                                   | -   [extract_c_function_pointer() |
|    cudaq)](api/languages/python_a |     (cudaq.PyKernelDecorator      |
| pi.html#cudaq.estimate_resources) |     method)](api/languages/p      |
| -   [evaluate()                   | ython_api.html#cudaq.PyKernelDeco |
|                                   | rator.extract_c_function_pointer) |
|   (cudaq.operators.ScalarOperator |                                   |
|     method)](api/                 |                                   |
| languages/python_api.html#cudaq.o |                                   |
| perators.ScalarOperator.evaluate) |                                   |
+-----------------------------------+-----------------------------------+

## F {#F}

+-----------------------------------+-----------------------------------+
| -   [FermionOperator (class in    | -   [from_json()                  |
|                                   |     (                             |
|    cudaq.operators.fermion)](api/ | cudaq.gradients.CentralDifference |
| languages/python_api.html#cudaq.o |     static                        |
| perators.fermion.FermionOperator) |     method)](api/lang             |
| -   [FermionOperatorElement       | uages/python_api.html#cudaq.gradi |
|     (class in                     | ents.CentralDifference.from_json) |
|     cuda                          |     -   [(                        |
| q.operators.fermion)](api/languag | cudaq.gradients.ForwardDifference |
| es/python_api.html#cudaq.operator |         static                    |
| s.fermion.FermionOperatorElement) |         method)](api/lang         |
| -   [FermionOperatorTerm (class   | uages/python_api.html#cudaq.gradi |
|     in                            | ents.ForwardDifference.from_json) |
|     c                             |     -                             |
| udaq.operators.fermion)](api/lang |  [(cudaq.gradients.ParameterShift |
| uages/python_api.html#cudaq.opera |         static                    |
| tors.fermion.FermionOperatorTerm) |         method)](api/l            |
| -   [final_expectation_values()   | anguages/python_api.html#cudaq.gr |
|     (cudaq.EvolveResult           | adients.ParameterShift.from_json) |
|     method)](api/lang             |     -   [(                        |
| uages/python_api.html#cudaq.Evolv | cudaq.operators.spin.SpinOperator |
| eResult.final_expectation_values) |         static                    |
| -   [final_state()                |         method)](api/lang         |
|     (cudaq.EvolveResult           | uages/python_api.html#cudaq.opera |
|     meth                          | tors.spin.SpinOperator.from_json) |
| od)](api/languages/python_api.htm |     -   [(cuda                    |
| l#cudaq.EvolveResult.final_state) | q.operators.spin.SpinOperatorTerm |
| -   [finalize() (in module        |         static                    |
|     cudaq.mpi)](api/languages/py  |         method)](api/language     |
| thon_api.html#cudaq.mpi.finalize) | s/python_api.html#cudaq.operators |
| -   [for_each_pauli()             | .spin.SpinOperatorTerm.from_json) |
|     (                             |     -   [(cudaq.optimizers.COBYLA |
| cudaq.operators.spin.SpinOperator |         static                    |
|     method)](api/languages        |         method)                   |
| /python_api.html#cudaq.operators. | ](api/languages/python_api.html#c |
| spin.SpinOperator.for_each_pauli) | udaq.optimizers.COBYLA.from_json) |
|     -   [(cuda                    |     -   [                         |
| q.operators.spin.SpinOperatorTerm | (cudaq.optimizers.GradientDescent |
|                                   |         static                    |
|        method)](api/languages/pyt |         method)](api/lan          |
| hon_api.html#cudaq.operators.spin | guages/python_api.html#cudaq.opti |
| .SpinOperatorTerm.for_each_pauli) | mizers.GradientDescent.from_json) |
| -   [for_each_term()              |     -   [(cudaq.optimizers.LBFGS  |
|     (                             |         static                    |
| cudaq.operators.spin.SpinOperator |         method                    |
|     method)](api/language         | )](api/languages/python_api.html# |
| s/python_api.html#cudaq.operators | cudaq.optimizers.LBFGS.from_json) |
| .spin.SpinOperator.for_each_term) |                                   |
| -   [ForwardDifference (class in  | -   [(cudaq.optimizers.NelderMead |
|     cudaq.gradients)              |         static                    |
| ](api/languages/python_api.html#c |         method)](ap               |
| udaq.gradients.ForwardDifference) | i/languages/python_api.html#cudaq |
| -   [from_data() (cudaq.State     | .optimizers.NelderMead.from_json) |
|     static                        |     -   [(cudaq.PyKernelDecorator |
|     method)](api/languages/pytho  |         static                    |
| n_api.html#cudaq.State.from_data) |         method)                   |
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
| -   [get()                        | -   [get_raw_data()               |
|     (cudaq.AsyncEvolveResult      |     (                             |
|     m                             | cudaq.operators.spin.SpinOperator |
| ethod)](api/languages/python_api. |     method)](api/languag          |
| html#cudaq.AsyncEvolveResult.get) | es/python_api.html#cudaq.operator |
|                                   | s.spin.SpinOperator.get_raw_data) |
|    -   [(cudaq.AsyncObserveResult |     -   [(cuda                    |
|         me                        | q.operators.spin.SpinOperatorTerm |
| thod)](api/languages/python_api.h |         method)](api/languages/p  |
| tml#cudaq.AsyncObserveResult.get) | ython_api.html#cudaq.operators.sp |
|     -   [(cudaq.AsyncSampleResult | in.SpinOperatorTerm.get_raw_data) |
|         m                         | -   [get_register_counts()        |
| ethod)](api/languages/python_api. |     (cudaq.SampleResult           |
| html#cudaq.AsyncSampleResult.get) |     method)](api                  |
|     -   [(cudaq.AsyncStateResult  | /languages/python_api.html#cudaq. |
|                                   | SampleResult.get_register_counts) |
| method)](api/languages/python_api | -   [get_sequential_data()        |
| .html#cudaq.AsyncStateResult.get) |     (cudaq.SampleResult           |
| -   [get_binary_symplectic_form() |     method)](api                  |
|     (cuda                         | /languages/python_api.html#cudaq. |
| q.operators.spin.SpinOperatorTerm | SampleResult.get_sequential_data) |
|     metho                         | -   [get_spin()                   |
| d)](api/languages/python_api.html |     (cudaq.ObserveResult          |
| #cudaq.operators.spin.SpinOperato |     me                            |
| rTerm.get_binary_symplectic_form) | thod)](api/languages/python_api.h |
| -   [get_channels()               | tml#cudaq.ObserveResult.get_spin) |
|     (cudaq.NoiseModel             | -   [get_state() (in module       |
|     met                           |     cudaq)](api/languages         |
| hod)](api/languages/python_api.ht | /python_api.html#cudaq.get_state) |
| ml#cudaq.NoiseModel.get_channels) | -   [get_state_async() (in module |
| -   [get_coefficient()            |     cudaq)](api/languages/pytho   |
|     (                             | n_api.html#cudaq.get_state_async) |
| cudaq.operators.spin.SpinOperator | -   [get_target() (in module      |
|     method)](api/languages/       |     cudaq)](api/languages/        |
| python_api.html#cudaq.operators.s | python_api.html#cudaq.get_target) |
| pin.SpinOperator.get_coefficient) | -   [get_targets() (in module     |
|     -   [(cuda                    |     cudaq)](api/languages/p       |
| q.operators.spin.SpinOperatorTerm | ython_api.html#cudaq.get_targets) |
|                                   | -   [get_term_count()             |
|       method)](api/languages/pyth |     (                             |
| on_api.html#cudaq.operators.spin. | cudaq.operators.spin.SpinOperator |
| SpinOperatorTerm.get_coefficient) |     method)](api/languages        |
| -   [get_marginal_counts()        | /python_api.html#cudaq.operators. |
|     (cudaq.SampleResult           | spin.SpinOperator.get_term_count) |
|     method)](api                  | -   [get_total_shots()            |
| /languages/python_api.html#cudaq. |     (cudaq.SampleResult           |
| SampleResult.get_marginal_counts) |     method)]                      |
| -   [get_ops()                    | (api/languages/python_api.html#cu |
|     (cudaq.KrausChannel           | daq.SampleResult.get_total_shots) |
|                                   | -   [getTensor() (cudaq.State     |
| method)](api/languages/python_api |     method)](api/languages/pytho  |
| .html#cudaq.KrausChannel.get_ops) | n_api.html#cudaq.State.getTensor) |
| -   [get_pauli_word()             | -   [getTensors() (cudaq.State    |
|     (cuda                         |     method)](api/languages/python |
| q.operators.spin.SpinOperatorTerm | _api.html#cudaq.State.getTensors) |
|     method)](api/languages/pyt    | -   [gradient (class in           |
| hon_api.html#cudaq.operators.spin |     cudaq.g                       |
| .SpinOperatorTerm.get_pauli_word) | radients)](api/languages/python_a |
| -   [get_precision()              | pi.html#cudaq.gradients.gradient) |
|     (cudaq.Target                 | -   [GradientDescent (class in    |
|                                   |     cudaq.optimizers              |
| method)](api/languages/python_api | )](api/languages/python_api.html# |
| .html#cudaq.Target.get_precision) | cudaq.optimizers.GradientDescent) |
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
+-----------------------------------+-----------------------------------+

## H {#H}

+-----------------------------------------------------------------------+
| -   [has_target() (in module                                          |
|     cudaq)](api/languages/python_api.html#cudaq.has_target)           |
+-----------------------------------------------------------------------+

## I {#I}

+-----------------------------------+-----------------------------------+
| -   [i() (in module               | -   [initialize_cudaq() (in       |
|     cudaq.spin)](api/langua       |     module                        |
| ges/python_api.html#cudaq.spin.i) |     cudaq)](api/languages/python  |
| -   [id                           | _api.html#cudaq.initialize_cudaq) |
|     (cuda                         | -   [InitialState (in module      |
| q.operators.MatrixOperatorElement |     cudaq.dynamics.helpers)](     |
|     property)](api/l              | api/languages/python_api.html#cud |
| anguages/python_api.html#cudaq.op | aq.dynamics.helpers.InitialState) |
| erators.MatrixOperatorElement.id) | -   [InitialStateType (class in   |
| -   [identities() (in module      |     cudaq)](api/languages/python  |
|     c                             | _api.html#cudaq.InitialStateType) |
| udaq.boson)](api/languages/python | -   [instantiate()                |
| _api.html#cudaq.boson.identities) |     (cudaq.operators              |
|     -   [(in module               |     m                             |
|         cudaq                     | ethod)](api/languages/python_api. |
| .fermion)](api/languages/python_a | html#cudaq.operators.instantiate) |
| pi.html#cudaq.fermion.identities) |     -   [(in module               |
|     -   [(in module               |         cudaq.operators.custom)]  |
|         cudaq.operators.custom)   | (api/languages/python_api.html#cu |
| ](api/languages/python_api.html#c | daq.operators.custom.instantiate) |
| udaq.operators.custom.identities) | -   [intermediate_states()        |
|     -   [(in module               |     (cudaq.EvolveResult           |
|                                   |     method)](api                  |
|  cudaq.spin)](api/languages/pytho | /languages/python_api.html#cudaq. |
| n_api.html#cudaq.spin.identities) | EvolveResult.intermediate_states) |
| -   [identity()                   | -   [IntermediateResultSave       |
|     (cu                           |     (class in                     |
| daq.operators.boson.BosonOperator |     c                             |
|     static                        | udaq)](api/languages/python_api.h |
|     method)](api/langu            | tml#cudaq.IntermediateResultSave) |
| ages/python_api.html#cudaq.operat | -   [is_constant()                |
| ors.boson.BosonOperator.identity) |                                   |
|     -   [(cudaq.                  |   (cudaq.operators.ScalarOperator |
| operators.fermion.FermionOperator |     method)](api/lan              |
|         static                    | guages/python_api.html#cudaq.oper |
|         method)](api/languages    | ators.ScalarOperator.is_constant) |
| /python_api.html#cudaq.operators. | -   [is_emulated() (cudaq.Target  |
| fermion.FermionOperator.identity) |                                   |
|     -                             |   method)](api/languages/python_a |
|  [(cudaq.operators.MatrixOperator | pi.html#cudaq.Target.is_emulated) |
|         static                    | -   [is_identity()                |
|         method)](api/             |     (cudaq.                       |
| languages/python_api.html#cudaq.o | operators.boson.BosonOperatorTerm |
| perators.MatrixOperator.identity) |     method)](api/languages/py     |
|     -   [(                        | thon_api.html#cudaq.operators.bos |
| cudaq.operators.spin.SpinOperator | on.BosonOperatorTerm.is_identity) |
|         static                    |     -   [(cudaq.oper              |
|         method)](api/lan          | ators.fermion.FermionOperatorTerm |
| guages/python_api.html#cudaq.oper |                                   |
| ators.spin.SpinOperator.identity) |     method)](api/languages/python |
|     -   [(in module               | _api.html#cudaq.operators.fermion |
|                                   | .FermionOperatorTerm.is_identity) |
|  cudaq.boson)](api/languages/pyth |     -   [(c                       |
| on_api.html#cudaq.boson.identity) | udaq.operators.MatrixOperatorTerm |
|     -   [(in module               |         method)](api/languag      |
|         cud                       | es/python_api.html#cudaq.operator |
| aq.fermion)](api/languages/python | s.MatrixOperatorTerm.is_identity) |
| _api.html#cudaq.fermion.identity) |     -   [(                        |
|     -   [(in module               | cudaq.operators.spin.SpinOperator |
|                                   |         method)](api/langua       |
|    cudaq.spin)](api/languages/pyt | ges/python_api.html#cudaq.operato |
| hon_api.html#cudaq.spin.identity) | rs.spin.SpinOperator.is_identity) |
| -   [initial_parameters           |     -   [(cuda                    |
|     (cudaq.optimizers.COBYLA      | q.operators.spin.SpinOperatorTerm |
|     property)](api/lan            |         method)](api/languages/   |
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
| -   [initialize() (in module      | -   [items() (cudaq.SampleResult  |
|                                   |                                   |
|    cudaq.mpi)](api/languages/pyth |   method)](api/languages/python_a |
| on_api.html#cudaq.mpi.initialize) | pi.html#cudaq.SampleResult.items) |
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
| -   [LBFGS (class in              | -   [lower_bounds                 |
|     cudaq.                        |     (cudaq.optimizers.COBYLA      |
| optimizers)](api/languages/python |     property)](a                  |
| _api.html#cudaq.optimizers.LBFGS) | pi/languages/python_api.html#cuda |
| -   [left_multiply()              | q.optimizers.COBYLA.lower_bounds) |
|     (cudaq.SuperOperator static   |     -   [                         |
|     method)                       | (cudaq.optimizers.GradientDescent |
| ](api/languages/python_api.html#c |         property)](api/langua     |
| udaq.SuperOperator.left_multiply) | ges/python_api.html#cudaq.optimiz |
| -   [left_right_multiply()        | ers.GradientDescent.lower_bounds) |
|     (cudaq.SuperOperator static   |     -   [(cudaq.optimizers.LBFGS  |
|     method)](api/                 |         property)](               |
| languages/python_api.html#cudaq.S | api/languages/python_api.html#cud |
| uperOperator.left_right_multiply) | aq.optimizers.LBFGS.lower_bounds) |
|                                   |                                   |
|                                   | -   [(cudaq.optimizers.NelderMead |
|                                   |         property)](api/l          |
|                                   | anguages/python_api.html#cudaq.op |
|                                   | timizers.NelderMead.lower_bounds) |
+-----------------------------------+-----------------------------------+

## M {#M}

+-----------------------------------+-----------------------------------+
| -   [make_kernel() (in module     | -   [min_degree                   |
|     cudaq)](api/languages/p       |     (cu                           |
| ython_api.html#cudaq.make_kernel) | daq.operators.boson.BosonOperator |
| -   [MatrixOperator (class in     |     property)](api/languag        |
|     cudaq.operato                 | es/python_api.html#cudaq.operator |
| rs)](api/languages/python_api.htm | s.boson.BosonOperator.min_degree) |
| l#cudaq.operators.MatrixOperator) |     -   [(cudaq.                  |
| -   [MatrixOperatorElement (class | operators.boson.BosonOperatorTerm |
|     in                            |                                   |
|     cudaq.operators)](ap          |        property)](api/languages/p |
| i/languages/python_api.html#cudaq | ython_api.html#cudaq.operators.bo |
| .operators.MatrixOperatorElement) | son.BosonOperatorTerm.min_degree) |
| -   [MatrixOperatorTerm (class in |     -   [(cudaq.                  |
|     cudaq.operators)]             | operators.fermion.FermionOperator |
| (api/languages/python_api.html#cu |                                   |
| daq.operators.MatrixOperatorTerm) |        property)](api/languages/p |
| -   [max_degree                   | ython_api.html#cudaq.operators.fe |
|     (cu                           | rmion.FermionOperator.min_degree) |
| daq.operators.boson.BosonOperator |     -   [(cudaq.oper              |
|     property)](api/languag        | ators.fermion.FermionOperatorTerm |
| es/python_api.html#cudaq.operator |                                   |
| s.boson.BosonOperator.max_degree) |    property)](api/languages/pytho |
|     -   [(cudaq.                  | n_api.html#cudaq.operators.fermio |
| operators.boson.BosonOperatorTerm | n.FermionOperatorTerm.min_degree) |
|                                   |     -                             |
|        property)](api/languages/p |  [(cudaq.operators.MatrixOperator |
| ython_api.html#cudaq.operators.bo |         property)](api/la         |
| son.BosonOperatorTerm.max_degree) | nguages/python_api.html#cudaq.ope |
|     -   [(cudaq.                  | rators.MatrixOperator.min_degree) |
| operators.fermion.FermionOperator |     -   [(c                       |
|                                   | udaq.operators.MatrixOperatorTerm |
|        property)](api/languages/p |         property)](api/langua     |
| ython_api.html#cudaq.operators.fe | ges/python_api.html#cudaq.operato |
| rmion.FermionOperator.max_degree) | rs.MatrixOperatorTerm.min_degree) |
|     -   [(cudaq.oper              |     -   [(                        |
| ators.fermion.FermionOperatorTerm | cudaq.operators.spin.SpinOperator |
|                                   |         property)](api/langu      |
|    property)](api/languages/pytho | ages/python_api.html#cudaq.operat |
| n_api.html#cudaq.operators.fermio | ors.spin.SpinOperator.min_degree) |
| n.FermionOperatorTerm.max_degree) |     -   [(cuda                    |
|     -                             | q.operators.spin.SpinOperatorTerm |
|  [(cudaq.operators.MatrixOperator |         property)](api/languages  |
|         property)](api/la         | /python_api.html#cudaq.operators. |
| nguages/python_api.html#cudaq.ope | spin.SpinOperatorTerm.min_degree) |
| rators.MatrixOperator.max_degree) | -   [minimal_eigenvalue()         |
|     -   [(c                       |     (cudaq.ComplexMatrix          |
| udaq.operators.MatrixOperatorTerm |     method)](api                  |
|         property)](api/langua     | /languages/python_api.html#cudaq. |
| ges/python_api.html#cudaq.operato | ComplexMatrix.minimal_eigenvalue) |
| rs.MatrixOperatorTerm.max_degree) | -   [minus() (in module           |
|     -   [(                        |     cudaq.spin)](api/languages/   |
| cudaq.operators.spin.SpinOperator | python_api.html#cudaq.spin.minus) |
|         property)](api/langu      | -   module                        |
| ages/python_api.html#cudaq.operat |     -   [cudaq](api/langua        |
| ors.spin.SpinOperator.max_degree) | ges/python_api.html#module-cudaq) |
|     -   [(cuda                    |     -                             |
| q.operators.spin.SpinOperatorTerm |    [cudaq.boson](api/languages/py |
|         property)](api/languages  | thon_api.html#module-cudaq.boson) |
| /python_api.html#cudaq.operators. |     -   [                         |
| spin.SpinOperatorTerm.max_degree) | cudaq.fermion](api/languages/pyth |
| -   [max_iterations               | on_api.html#module-cudaq.fermion) |
|     (cudaq.optimizers.COBYLA      |     -   [cudaq.operators.cu       |
|     property)](api                | stom](api/languages/python_api.ht |
| /languages/python_api.html#cudaq. | ml#module-cudaq.operators.custom) |
| optimizers.COBYLA.max_iterations) |                                   |
|     -   [                         |  -   [cudaq.spin](api/languages/p |
| (cudaq.optimizers.GradientDescent | ython_api.html#module-cudaq.spin) |
|         property)](api/language   | -   [momentum() (in module        |
| s/python_api.html#cudaq.optimizer |                                   |
| s.GradientDescent.max_iterations) |  cudaq.boson)](api/languages/pyth |
|     -   [(cudaq.optimizers.LBFGS  | on_api.html#cudaq.boson.momentum) |
|         property)](ap             |     -   [(in module               |
| i/languages/python_api.html#cudaq |         cudaq.operators.custo     |
| .optimizers.LBFGS.max_iterations) | m)](api/languages/python_api.html |
|                                   | #cudaq.operators.custom.momentum) |
| -   [(cudaq.optimizers.NelderMead | -   [most_probable()              |
|         property)](api/lan        |     (cudaq.SampleResult           |
| guages/python_api.html#cudaq.opti |     method                        |
| mizers.NelderMead.max_iterations) | )](api/languages/python_api.html# |
| -   [mdiag_sparse_matrix (C++     | cudaq.SampleResult.most_probable) |
|     type)](api/languages/cpp_api. |                                   |
| html#_CPPv419mdiag_sparse_matrix) |                                   |
| -   [merge_kernel()               |                                   |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](a                    |                                   |
| pi/languages/python_api.html#cuda |                                   |
| q.PyKernelDecorator.merge_kernel) |                                   |
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
|     cudaq)](api/languages/pyt     | -   [optimize()                   |
| hon_api.html#cudaq.observe_async) |     (cudaq.optimizers.COBYLA      |
| -   [ObserveResult (class in      |     method                        |
|     cudaq)](api/languages/pyt     | )](api/languages/python_api.html# |
| hon_api.html#cudaq.ObserveResult) | cudaq.optimizers.COBYLA.optimize) |
| -   [OperatorSum (in module       |     -   [                         |
|     cudaq.oper                    | (cudaq.optimizers.GradientDescent |
| ators)](api/languages/python_api. |         method)](api/la           |
| html#cudaq.operators.OperatorSum) | nguages/python_api.html#cudaq.opt |
| -   [ops_count                    | imizers.GradientDescent.optimize) |
|     (cudaq.                       |     -   [(cudaq.optimizers.LBFGS  |
| operators.boson.BosonOperatorTerm |         metho                     |
|     property)](api/languages/     | d)](api/languages/python_api.html |
| python_api.html#cudaq.operators.b | #cudaq.optimizers.LBFGS.optimize) |
| oson.BosonOperatorTerm.ops_count) |                                   |
|     -   [(cudaq.oper              | -   [(cudaq.optimizers.NelderMead |
| ators.fermion.FermionOperatorTerm |         method)](a                |
|                                   | pi/languages/python_api.html#cuda |
|     property)](api/languages/pyth | q.optimizers.NelderMead.optimize) |
| on_api.html#cudaq.operators.fermi | -   [optimizer (class in          |
| on.FermionOperatorTerm.ops_count) |     cudaq.opti                    |
|     -   [(c                       | mizers)](api/languages/python_api |
| udaq.operators.MatrixOperatorTerm | .html#cudaq.optimizers.optimizer) |
|         property)](api/langu      | -   [overlap() (cudaq.State       |
| ages/python_api.html#cudaq.operat |     method)](api/languages/pyt    |
| ors.MatrixOperatorTerm.ops_count) | hon_api.html#cudaq.State.overlap) |
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
| nguages/python_api.html#cudaq.ope | -   [probability()                |
| rators.MatrixOperator.parameters) |     (cudaq.SampleResult           |
|     -   [(cuda                    |     meth                          |
| q.operators.MatrixOperatorElement | od)](api/languages/python_api.htm |
|         property)](api/languages  | l#cudaq.SampleResult.probability) |
| /python_api.html#cudaq.operators. | -   [processCallableArg()         |
| MatrixOperatorElement.parameters) |     (cudaq.PyKernelDecorator      |
|     -   [(c                       |     method)](api/lan              |
| udaq.operators.MatrixOperatorTerm | guages/python_api.html#cudaq.PyKe |
|         property)](api/langua     | rnelDecorator.processCallableArg) |
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
| -   [random()                     | -   [reset_target() (in module    |
|     (                             |     cudaq)](api/languages/py      |
| cudaq.operators.spin.SpinOperator | thon_api.html#cudaq.reset_target) |
|     static                        | -   [Resources (class in          |
|     method)](api/l                |     cudaq)](api/languages         |
| anguages/python_api.html#cudaq.op | /python_api.html#cudaq.Resources) |
| erators.spin.SpinOperator.random) | -   [right_multiply()             |
| -   [rank() (in module            |     (cudaq.SuperOperator static   |
|     cudaq.mpi)](api/language      |     method)]                      |
| s/python_api.html#cudaq.mpi.rank) | (api/languages/python_api.html#cu |
| -   [register_names               | daq.SuperOperator.right_multiply) |
|     (cudaq.SampleResult           | -   [row_count                    |
|     attribute)                    |     (cudaq.KrausOperator          |
| ](api/languages/python_api.html#c |     prope                         |
| udaq.SampleResult.register_names) | rty)](api/languages/python_api.ht |
| -                                 | ml#cudaq.KrausOperator.row_count) |
|   [register_set_target_callback() | -   [run() (in module             |
|     (in module                    |     cudaq)](api/lan               |
|     cudaq)]                       | guages/python_api.html#cudaq.run) |
| (api/languages/python_api.html#cu | -   [run_async() (in module       |
| daq.register_set_target_callback) |     cudaq)](api/languages         |
| -   [requires_gradients()         | /python_api.html#cudaq.run_async) |
|     (cudaq.optimizers.COBYLA      | -   [RydbergHamiltonian (class in |
|     method)](api/lan              |     cudaq.operators)]             |
| guages/python_api.html#cudaq.opti | (api/languages/python_api.html#cu |
| mizers.COBYLA.requires_gradients) | daq.operators.RydbergHamiltonian) |
|     -   [                         |                                   |
| (cudaq.optimizers.GradientDescent |                                   |
|         method)](api/languages/py |                                   |
| thon_api.html#cudaq.optimizers.Gr |                                   |
| adientDescent.requires_gradients) |                                   |
|     -   [(cudaq.optimizers.LBFGS  |                                   |
|         method)](api/la           |                                   |
| nguages/python_api.html#cudaq.opt |                                   |
| imizers.LBFGS.requires_gradients) |                                   |
|                                   |                                   |
| -   [(cudaq.optimizers.NelderMead |                                   |
|         method)](api/languag      |                                   |
| es/python_api.html#cudaq.optimize |                                   |
| rs.NelderMead.requires_gradients) |                                   |
+-----------------------------------+-----------------------------------+

## S {#S}

+-----------------------------------+-----------------------------------+
| -   [sample() (in module          | -   [set_target() (in module      |
|     cudaq)](api/langua            |     cudaq)](api/languages/        |
| ges/python_api.html#cudaq.sample) | python_api.html#cudaq.set_target) |
|     -   [(in module               | -   [SimulationPrecision (class   |
|                                   |     in                            |
|      cudaq.orca)](api/languages/p |                                   |
| ython_api.html#cudaq.orca.sample) |   cudaq)](api/languages/python_ap |
| -   [sample_async() (in module    | i.html#cudaq.SimulationPrecision) |
|     cudaq)](api/languages/py      | -   [simulator (cudaq.Target      |
| thon_api.html#cudaq.sample_async) |                                   |
| -   [SampleResult (class in       |   property)](api/languages/python |
|     cudaq)](api/languages/py      | _api.html#cudaq.Target.simulator) |
| thon_api.html#cudaq.SampleResult) | -   [slice() (cudaq.QuakeValue    |
| -   [ScalarOperator (class in     |     method)](api/languages/python |
|     cudaq.operato                 | _api.html#cudaq.QuakeValue.slice) |
| rs)](api/languages/python_api.htm | -   [SpinOperator (class in       |
| l#cudaq.operators.ScalarOperator) |     cudaq.operators.spin)         |
| -   [Schedule (class in           | ](api/languages/python_api.html#c |
|     cudaq)](api/language          | udaq.operators.spin.SpinOperator) |
| s/python_api.html#cudaq.Schedule) | -   [SpinOperatorElement (class   |
| -   [serialize()                  |     in                            |
|     (                             |     cudaq.operators.spin)](api/l  |
| cudaq.operators.spin.SpinOperator | anguages/python_api.html#cudaq.op |
|     method)](api/lang             | erators.spin.SpinOperatorElement) |
| uages/python_api.html#cudaq.opera | -   [SpinOperatorTerm (class in   |
| tors.spin.SpinOperator.serialize) |     cudaq.operators.spin)](ap     |
|     -   [(cuda                    | i/languages/python_api.html#cudaq |
| q.operators.spin.SpinOperatorTerm | .operators.spin.SpinOperatorTerm) |
|         method)](api/language     | -   [squeeze() (in module         |
| s/python_api.html#cudaq.operators |     cudaq.operators.cust          |
| .spin.SpinOperatorTerm.serialize) | om)](api/languages/python_api.htm |
|     -   [(cudaq.SampleResult      | l#cudaq.operators.custom.squeeze) |
|         me                        | -   [State (class in              |
| thod)](api/languages/python_api.h |     cudaq)](api/langu             |
| tml#cudaq.SampleResult.serialize) | ages/python_api.html#cudaq.State) |
| -   [set_noise() (in module       | -   [SuperOperator (class in      |
|     cudaq)](api/languages         |     cudaq)](api/languages/pyt     |
| /python_api.html#cudaq.set_noise) | hon_api.html#cudaq.SuperOperator) |
| -   [set_random_seed() (in module | -                                 |
|     cudaq)](api/languages/pytho   |  [synthesize_callable_arguments() |
| n_api.html#cudaq.set_random_seed) |     (cudaq.PyKernelDecorator      |
|                                   |     method)](api/languages/pyth   |
|                                   | on_api.html#cudaq.PyKernelDecorat |
|                                   | or.synthesize_callable_arguments) |
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
| -   [term_id                      | n.BosonOperatorElement.to_string) |
|     (cudaq.                       |     -   [(cudaq.operato           |
| operators.boson.BosonOperatorTerm | rs.fermion.FermionOperatorElement |
|     property)](api/language       |                                   |
| s/python_api.html#cudaq.operators |    method)](api/languages/python_ |
| .boson.BosonOperatorTerm.term_id) | api.html#cudaq.operators.fermion. |
|     -   [(cudaq.oper              | FermionOperatorElement.to_string) |
| ators.fermion.FermionOperatorTerm |     -   [(cuda                    |
|                                   | q.operators.MatrixOperatorElement |
|       property)](api/languages/py |         method)](api/language     |
| thon_api.html#cudaq.operators.fer | s/python_api.html#cudaq.operators |
| mion.FermionOperatorTerm.term_id) | .MatrixOperatorElement.to_string) |
|     -   [(c                       |     -   [(                        |
| udaq.operators.MatrixOperatorTerm | cudaq.operators.spin.SpinOperator |
|         property)](api/lan        |         method)](api/lang         |
| guages/python_api.html#cudaq.oper | uages/python_api.html#cudaq.opera |
| ators.MatrixOperatorTerm.term_id) | tors.spin.SpinOperator.to_string) |
|     -   [(cuda                    |     -   [(cudaq.o                 |
| q.operators.spin.SpinOperatorTerm | perators.spin.SpinOperatorElement |
|         property)](api/langua     |         method)](api/languages/p  |
| ges/python_api.html#cudaq.operato | ython_api.html#cudaq.operators.sp |
| rs.spin.SpinOperatorTerm.term_id) | in.SpinOperatorElement.to_string) |
| -   [to_dict() (cudaq.Resources   |     -   [(cuda                    |
|                                   | q.operators.spin.SpinOperatorTerm |
|    method)](api/languages/python_ |         method)](api/language     |
| api.html#cudaq.Resources.to_dict) | s/python_api.html#cudaq.operators |
| -   [to_json()                    | .spin.SpinOperatorTerm.to_string) |
|     (                             | -   [translate() (in module       |
| cudaq.gradients.CentralDifference |     cudaq)](api/languages         |
|     method)](api/la               | /python_api.html#cudaq.translate) |
| nguages/python_api.html#cudaq.gra | -   [trim()                       |
| dients.CentralDifference.to_json) |     (cu                           |
|     -   [(                        | daq.operators.boson.BosonOperator |
| cudaq.gradients.ForwardDifference |     method)](api/l                |
|         method)](api/la           | anguages/python_api.html#cudaq.op |
| nguages/python_api.html#cudaq.gra | erators.boson.BosonOperator.trim) |
| dients.ForwardDifference.to_json) |     -   [(cudaq.                  |
|     -                             | operators.fermion.FermionOperator |
|  [(cudaq.gradients.ParameterShift |         method)](api/langu        |
|         method)](api              | ages/python_api.html#cudaq.operat |
| /languages/python_api.html#cudaq. | ors.fermion.FermionOperator.trim) |
| gradients.ParameterShift.to_json) |     -                             |
|     -   [(                        |  [(cudaq.operators.MatrixOperator |
| cudaq.operators.spin.SpinOperator |         method)](                 |
|         method)](api/la           | api/languages/python_api.html#cud |
| nguages/python_api.html#cudaq.ope | aq.operators.MatrixOperator.trim) |
| rators.spin.SpinOperator.to_json) |     -   [(                        |
|     -   [(cuda                    | cudaq.operators.spin.SpinOperator |
| q.operators.spin.SpinOperatorTerm |         method)](api              |
|         method)](api/langua       | /languages/python_api.html#cudaq. |
| ges/python_api.html#cudaq.operato | operators.spin.SpinOperator.trim) |
| rs.spin.SpinOperatorTerm.to_json) | -   [type_to_str()                |
|     -   [(cudaq.optimizers.COBYLA |     (cudaq.PyKernelDecorator      |
|         metho                     |     static                        |
| d)](api/languages/python_api.html |     method)](                     |
| #cudaq.optimizers.COBYLA.to_json) | api/languages/python_api.html#cud |
|     -   [                         | aq.PyKernelDecorator.type_to_str) |
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
| -   [upper_bounds (cudaq.optimizers.COBYLA                            |
|     property)                                                         |
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
