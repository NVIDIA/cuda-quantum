::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-3797
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
| -   [BaseIntegrator (class in     | -   [BosonOperator (class in      |
|                                   |     cudaq.operators.boson)](      |
| cudaq.dynamics.integrator)](api/l | api/languages/python_api.html#cud |
| anguages/python_api.html#cudaq.dy | aq.operators.boson.BosonOperator) |
| namics.integrator.BaseIntegrator) | -   [BosonOperatorElement (class  |
| -   [beta_reduction()             |     in                            |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](api                  |   cudaq.operators.boson)](api/lan |
| /languages/python_api.html#cudaq. | guages/python_api.html#cudaq.oper |
| PyKernelDecorator.beta_reduction) | ators.boson.BosonOperatorElement) |
| -   [BitFlipChannel (class in     | -   [BosonOperatorTerm (class in  |
|     cudaq)](api/languages/pyth    |     cudaq.operators.boson)](api/  |
| on_api.html#cudaq.BitFlipChannel) | languages/python_api.html#cudaq.o |
|                                   | perators.boson.BosonOperatorTerm) |
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
| -   [ComplexMatrix (class in      | ction)](api/languages/cpp_api.htm |
|     cudaq)](api/languages/pyt     | l#_CPPv4NK5cudaq10product_op14con |
| hon_api.html#cudaq.ComplexMatrix) | st_iteratorneERK14const_iterator) |
| -   [compute()                    | -   [cudaq::produ                 |
|     (                             | ct_op::const_iterator::operator\* |
| cudaq.gradients.CentralDifference |     (C++                          |
|     method)](api/la               |     function)](api/lang           |
| nguages/python_api.html#cudaq.gra | uages/cpp_api.html#_CPPv4NK5cudaq |
| dients.CentralDifference.compute) | 10product_op14const_iteratormlEv) |
|     -   [(                        | -   [cudaq::produ                 |
| cudaq.gradients.ForwardDifference | ct_op::const_iterator::operator++ |
|         method)](api/la           |     (C++                          |
| nguages/python_api.html#cudaq.gra |     function)](api/lang           |
| dients.ForwardDifference.compute) | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     -                             | 0product_op14const_iteratorppEi), |
|  [(cudaq.gradients.ParameterShift |     [\[1\]](api/lan               |
|         method)](api              | guages/cpp_api.html#_CPPv4N5cudaq |
| /languages/python_api.html#cudaq. | 10product_op14const_iteratorppEv) |
| gradients.ParameterShift.compute) | -   [cudaq::produc                |
| -   [const()                      | t_op::const_iterator::operator\-- |
|                                   |     (C++                          |
|   (cudaq.operators.ScalarOperator |     function)](api/lang           |
|     class                         | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     method)](a                    | 0product_op14const_iteratormmEi), |
| pi/languages/python_api.html#cuda |     [\[1\]](api/lan               |
| q.operators.ScalarOperator.const) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [copy()                       | 10product_op14const_iteratormmEv) |
|     (cu                           | -   [cudaq::produc                |
| daq.operators.boson.BosonOperator | t_op::const_iterator::operator-\> |
|     method)](api/l                |     (C++                          |
| anguages/python_api.html#cudaq.op |     function)](api/lan            |
| erators.boson.BosonOperator.copy) | guages/cpp_api.html#_CPPv4N5cudaq |
|     -   [(cudaq.                  | 10product_op14const_iteratorptEv) |
| operators.boson.BosonOperatorTerm | -   [cudaq::produ                 |
|         method)](api/langu        | ct_op::const_iterator::operator== |
| ages/python_api.html#cudaq.operat |     (C++                          |
| ors.boson.BosonOperatorTerm.copy) |     fun                           |
|     -   [(cudaq.                  | ction)](api/languages/cpp_api.htm |
| operators.fermion.FermionOperator | l#_CPPv4NK5cudaq10product_op14con |
|         method)](api/langu        | st_iteratoreqERK14const_iterator) |
| ages/python_api.html#cudaq.operat | -   [cudaq::product_op::degrees   |
| ors.fermion.FermionOperator.copy) |     (C++                          |
|     -   [(cudaq.oper              |     function)                     |
| ators.fermion.FermionOperatorTerm | ](api/languages/cpp_api.html#_CPP |
|         method)](api/languages    | v4NK5cudaq10product_op7degreesEv) |
| /python_api.html#cudaq.operators. | -   [cudaq::product_op::dump (C++ |
| fermion.FermionOperatorTerm.copy) |     functi                        |
|     -                             | on)](api/languages/cpp_api.html#_ |
|  [(cudaq.operators.MatrixOperator | CPPv4NK5cudaq10product_op4dumpEv) |
|         method)](                 | -   [cudaq::product_op::end (C++  |
| api/languages/python_api.html#cud |     funct                         |
| aq.operators.MatrixOperator.copy) | ion)](api/languages/cpp_api.html# |
|     -   [(c                       | _CPPv4NK5cudaq10product_op3endEv) |
| udaq.operators.MatrixOperatorTerm | -   [c                            |
|         method)](api/             | udaq::product_op::get_coefficient |
| languages/python_api.html#cudaq.o |     (C++                          |
| perators.MatrixOperatorTerm.copy) |     function)](api/lan            |
|     -   [(                        | guages/cpp_api.html#_CPPv4NK5cuda |
| cudaq.operators.spin.SpinOperator | q10product_op15get_coefficientEv) |
|         method)](api              | -                                 |
| /languages/python_api.html#cudaq. |   [cudaq::product_op::get_term_id |
| operators.spin.SpinOperator.copy) |     (C++                          |
|     -   [(cuda                    |     function)](api                |
| q.operators.spin.SpinOperatorTerm | /languages/cpp_api.html#_CPPv4NK5 |
|         method)](api/lan          | cudaq10product_op11get_term_idEv) |
| guages/python_api.html#cudaq.oper | -                                 |
| ators.spin.SpinOperatorTerm.copy) |   [cudaq::product_op::is_identity |
| -   [count() (cudaq.Resources     |     (C++                          |
|     method)](api/languages/pytho  |     function)](api                |
| n_api.html#cudaq.Resources.count) | /languages/cpp_api.html#_CPPv4NK5 |
|     -   [(cudaq.SampleResult      | cudaq10product_op11is_identityEv) |
|                                   | -   [cudaq::product_op::num_ops   |
|   method)](api/languages/python_a |     (C++                          |
| pi.html#cudaq.SampleResult.count) |     function)                     |
| -   [count_controls()             | ](api/languages/cpp_api.html#_CPP |
|     (cudaq.Resources              | v4NK5cudaq10product_op7num_opsEv) |
|     meth                          | -                                 |
| od)](api/languages/python_api.htm |    [cudaq::product_op::operator\* |
| l#cudaq.Resources.count_controls) |     (C++                          |
| -   [counts()                     |     function)](api/languages/     |
|     (cudaq.ObserveResult          | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|                                   | oduct_opmlE10product_opI1TERK15sc |
| method)](api/languages/python_api | alar_operatorRK10product_opI1TE), |
| .html#cudaq.ObserveResult.counts) |     [\[1\]](api/languages/        |
| -   [create() (in module          | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|                                   | oduct_opmlE10product_opI1TERK15sc |
|    cudaq.boson)](api/languages/py | alar_operatorRR10product_opI1TE), |
| thon_api.html#cudaq.boson.create) |     [\[2\]](api/languages/        |
|     -   [(in module               | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|         c                         | oduct_opmlE10product_opI1TERR15sc |
| udaq.fermion)](api/languages/pyth | alar_operatorRK10product_opI1TE), |
| on_api.html#cudaq.fermion.create) |     [\[3\]](api/languages/        |
| -   [csr_spmatrix (C++            | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     type)](api/languages/c        | oduct_opmlE10product_opI1TERR15sc |
| pp_api.html#_CPPv412csr_spmatrix) | alar_operatorRR10product_opI1TE), |
| -   cudaq                         |     [\[4\]](api/                  |
|     -   [module](api/langua       | languages/cpp_api.html#_CPPv4I0EN |
| ges/python_api.html#module-cudaq) | 5cudaq10product_opmlE6sum_opI1TER |
| -   [cudaq (C++                   | K15scalar_operatorRK6sum_opI1TE), |
|     type)](api/lan                |     [\[5\]](api/                  |
| guages/cpp_api.html#_CPPv45cudaq) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq.apply_noise() (in      | 5cudaq10product_opmlE6sum_opI1TER |
|     module                        | K15scalar_operatorRR6sum_opI1TE), |
|     cudaq)](api/languages/python_ |     [\[6\]](api/                  |
| api.html#cudaq.cudaq.apply_noise) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.boson                   | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [module](api/languages/py | R15scalar_operatorRK6sum_opI1TE), |
| thon_api.html#module-cudaq.boson) |     [\[7\]](api/                  |
| -   cudaq.fermion                 | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opmlE6sum_opI1TER |
|   -   [module](api/languages/pyth | R15scalar_operatorRR6sum_opI1TE), |
| on_api.html#module-cudaq.fermion) |     [\[8\]](api/languages         |
| -   cudaq.operators.custom        | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     -   [mo                       | duct_opmlERK6sum_opI9HandlerTyE), |
| dule](api/languages/python_api.ht |     [\[9\]](api/languages/cpp_a   |
| ml#module-cudaq.operators.custom) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.spin                    | opmlERK10product_opI9HandlerTyE), |
|     -   [module](api/languages/p  |     [\[10\]](api/language         |
| ython_api.html#module-cudaq.spin) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::amplitude_damping     | roduct_opmlERK15scalar_operator), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     cla                           | pi.html#_CPPv4NKR5cudaq10product_ |
| ss)](api/languages/cpp_api.html#_ | opmlERR10product_opI9HandlerTyE), |
| CPPv4N5cudaq17amplitude_dampingE) |     [\[12\]](api/language         |
| -                                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| [cudaq::amplitude_damping_channel | roduct_opmlERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/cpp_   |
|     class)](api                   | api.html#_CPPv4NO5cudaq10product_ |
| /languages/cpp_api.html#_CPPv4N5c | opmlERK10product_opI9HandlerTyE), |
| udaq25amplitude_damping_channelE) |     [\[14\]](api/languag          |
| -   [cudaq::amplitud              | es/cpp_api.html#_CPPv4NO5cudaq10p |
| e_damping_channel::num_parameters | roduct_opmlERK15scalar_operator), |
|     (C++                          |     [\[15\]](api/languages/cpp_   |
|     member)](api/languages/cpp_a  | api.html#_CPPv4NO5cudaq10product_ |
| pi.html#_CPPv4N5cudaq25amplitude_ | opmlERR10product_opI9HandlerTyE), |
| damping_channel14num_parametersE) |     [\[16\]](api/langua           |
| -   [cudaq::ampli                 | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| tude_damping_channel::num_targets | product_opmlERR15scalar_operator) |
|     (C++                          | -                                 |
|     member)](api/languages/cp     |   [cudaq::product_op::operator\*= |
| p_api.html#_CPPv4N5cudaq25amplitu |     (C++                          |
| de_damping_channel11num_targetsE) |     function)](api/languages/cpp  |
| -   [cudaq::AnalogRemoteRESTQPU   | _api.html#_CPPv4N5cudaq10product_ |
|     (C++                          | opmLERK10product_opI9HandlerTyE), |
|     class                         |     [\[1\]](api/langua            |
| )](api/languages/cpp_api.html#_CP | ges/cpp_api.html#_CPPv4N5cudaq10p |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | roduct_opmLERK15scalar_operator), |
| -   [cudaq::apply_noise (C++      |     [\[2\]](api/languages/cp      |
|     function)](api/               | p_api.html#_CPPv4N5cudaq10product |
| languages/cpp_api.html#_CPPv4I0Dp | _opmLERR10product_opI9HandlerTyE) |
| EN5cudaq11apply_noiseEvDpRR4Args) | -   [cudaq::product_op::operator+ |
| -   [cudaq::async_result (C++     |     (C++                          |
|     c                             |     function)](api/langu          |
| lass)](api/languages/cpp_api.html | ages/cpp_api.html#_CPPv4I0EN5cuda |
| #_CPPv4I0EN5cudaq12async_resultE) | q10product_opplE6sum_opI1TERK15sc |
| -   [cudaq::async_result::get     | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     functi                        | languages/cpp_api.html#_CPPv4I0EN |
| on)](api/languages/cpp_api.html#_ | 5cudaq10product_opplE6sum_opI1TER |
| CPPv4N5cudaq12async_result3getEv) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::async_sample_result   |     [\[2\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     type                          | q10product_opplE6sum_opI1TERK15sc |
| )](api/languages/cpp_api.html#_CP | alar_operatorRR10product_opI1TE), |
| Pv4N5cudaq19async_sample_resultE) |     [\[3\]](api/                  |
| -   [cudaq::BaseRemoteRESTQPU     | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     cla                           | K15scalar_operatorRR6sum_opI1TE), |
| ss)](api/languages/cpp_api.html#_ |     [\[4\]](api/langu             |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -                                 | q10product_opplE6sum_opI1TERR15sc |
|    [cudaq::BaseRemoteSimulatorQPU | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[5\]](api/                  |
|     class)](                      | languages/cpp_api.html#_CPPv4I0EN |
| api/languages/cpp_api.html#_CPPv4 | 5cudaq10product_opplE6sum_opI1TER |
| N5cudaq22BaseRemoteSimulatorQPUE) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::bit_flip_channel (C++ |     [\[6\]](api/langu             |
|     cl                            | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ass)](api/languages/cpp_api.html# | q10product_opplE6sum_opI1TERR15sc |
| _CPPv4N5cudaq16bit_flip_channelE) | alar_operatorRR10product_opI1TE), |
| -   [cudaq:                       |     [\[7\]](api/                  |
| :bit_flip_channel::num_parameters | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     member)](api/langua           | R15scalar_operatorRR6sum_opI1TE), |
| ges/cpp_api.html#_CPPv4N5cudaq16b |     [\[8\]](api/languages/cpp_a   |
| it_flip_channel14num_parametersE) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [cud                          | opplERK10product_opI9HandlerTyE), |
| aq::bit_flip_channel::num_targets |     [\[9\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     member)](api/lan              | roduct_opplERK15scalar_operator), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[10\]](api/languages/       |
| 16bit_flip_channel11num_targetsE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::boson_handler (C++    | duct_opplERK6sum_opI9HandlerTyE), |
|                                   |     [\[11\]](api/languages/cpp_a  |
|  class)](api/languages/cpp_api.ht | pi.html#_CPPv4NKR5cudaq10product_ |
| ml#_CPPv4N5cudaq13boson_handlerE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::boson_op (C++         |     [\[12\]](api/language         |
|     type)](api/languages/cpp_     | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| api.html#_CPPv4N5cudaq8boson_opE) | roduct_opplERR15scalar_operator), |
| -   [cudaq::boson_op_term (C++    |     [\[13\]](api/languages/       |
|                                   | cpp_api.html#_CPPv4NKR5cudaq10pro |
|   type)](api/languages/cpp_api.ht | duct_opplERR6sum_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13boson_op_termE) |     [\[                           |
| -   [cudaq::CodeGenConfig (C++    | 14\]](api/languages/cpp_api.html# |
|                                   | _CPPv4NKR5cudaq10product_opplEv), |
| struct)](api/languages/cpp_api.ht |     [\[15\]](api/languages/cpp_   |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | api.html#_CPPv4NO5cudaq10product_ |
| -   [cudaq::commutation_relations | opplERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[16\]](api/languag          |
|     struct)]                      | es/cpp_api.html#_CPPv4NO5cudaq10p |
| (api/languages/cpp_api.html#_CPPv | roduct_opplERK15scalar_operator), |
| 4N5cudaq21commutation_relationsE) |     [\[17\]](api/languages        |
| -   [cudaq::complex (C++          | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     type)](api/languages/cpp      | duct_opplERK6sum_opI9HandlerTyE), |
| _api.html#_CPPv4N5cudaq7complexE) |     [\[18\]](api/languages/cpp_   |
| -   [cudaq::complex_matrix (C++   | api.html#_CPPv4NO5cudaq10product_ |
|                                   | opplERR10product_opI9HandlerTyE), |
| class)](api/languages/cpp_api.htm |     [\[19\]](api/languag          |
| l#_CPPv4N5cudaq14complex_matrixE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -                                 | roduct_opplERR15scalar_operator), |
|   [cudaq::complex_matrix::adjoint |     [\[20\]](api/languages        |
|     (C++                          | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     function)](a                  | duct_opplERR6sum_opI9HandlerTyE), |
| pi/languages/cpp_api.html#_CPPv4N |     [                             |
| 5cudaq14complex_matrix7adjointEv) | \[21\]](api/languages/cpp_api.htm |
| -   [cudaq::                      | l#_CPPv4NO5cudaq10product_opplEv) |
| complex_matrix::diagonal_elements | -   [cudaq::product_op::operator- |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/langu          |
| /cpp_api.html#_CPPv4NK5cudaq14com | ages/cpp_api.html#_CPPv4I0EN5cuda |
| plex_matrix17diagonal_elementsEi) | q10product_opmiE6sum_opI1TERK15sc |
| -   [cudaq::complex_matrix::dump  | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     function)](api/language       | languages/cpp_api.html#_CPPv4I0EN |
| s/cpp_api.html#_CPPv4NK5cudaq14co | 5cudaq10product_opmiE6sum_opI1TER |
| mplex_matrix4dumpERNSt7ostreamE), | K15scalar_operatorRK6sum_opI1TE), |
|     [\[1\]]                       |     [\[2\]](api/langu             |
| (api/languages/cpp_api.html#_CPPv | ages/cpp_api.html#_CPPv4I0EN5cuda |
| 4NK5cudaq14complex_matrix4dumpEv) | q10product_opmiE6sum_opI1TERK15sc |
| -   [c                            | alar_operatorRR10product_opI1TE), |
| udaq::complex_matrix::eigenvalues |     [\[3\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/lan            | 5cudaq10product_opmiE6sum_opI1TER |
| guages/cpp_api.html#_CPPv4NK5cuda | K15scalar_operatorRR6sum_opI1TE), |
| q14complex_matrix11eigenvaluesEv) |     [\[4\]](api/langu             |
| -   [cu                           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| daq::complex_matrix::eigenvectors | q10product_opmiE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRK10product_opI1TE), |
|     function)](api/lang           |     [\[5\]](api/                  |
| uages/cpp_api.html#_CPPv4NK5cudaq | languages/cpp_api.html#_CPPv4I0EN |
| 14complex_matrix12eigenvectorsEv) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [c                            | R15scalar_operatorRK6sum_opI1TE), |
| udaq::complex_matrix::exponential |     [\[6\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     function)](api/la             | q10product_opmiE6sum_opI1TERR15sc |
| nguages/cpp_api.html#_CPPv4N5cuda | alar_operatorRR10product_opI1TE), |
| q14complex_matrix11exponentialEv) |     [\[7\]](api/                  |
| -                                 | languages/cpp_api.html#_CPPv4I0EN |
|  [cudaq::complex_matrix::identity | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | R15scalar_operatorRR6sum_opI1TE), |
|     function)](api/languages      |     [\[8\]](api/languages/cpp_a   |
| /cpp_api.html#_CPPv4N5cudaq14comp | pi.html#_CPPv4NKR5cudaq10product_ |
| lex_matrix8identityEKNSt6size_tE) | opmiERK10product_opI9HandlerTyE), |
| -                                 |     [\[9\]](api/language          |
| [cudaq::complex_matrix::kronecker | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opmiERK15scalar_operator), |
|     function)](api/lang           |     [\[10\]](api/languages/       |
| uages/cpp_api.html#_CPPv4I00EN5cu | cpp_api.html#_CPPv4NKR5cudaq10pro |
| daq14complex_matrix9kroneckerE14c | duct_opmiERK6sum_opI9HandlerTyE), |
| omplex_matrix8Iterable8Iterable), |     [\[11\]](api/languages/cpp_a  |
|     [\[1\]](api/l                 | pi.html#_CPPv4NKR5cudaq10product_ |
| anguages/cpp_api.html#_CPPv4N5cud | opmiERR10product_opI9HandlerTyE), |
| aq14complex_matrix9kroneckerERK14 |     [\[12\]](api/language         |
| complex_matrixRK14complex_matrix) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::c                     | roduct_opmiERR15scalar_operator), |
| omplex_matrix::minimal_eigenvalue |     [\[13\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     function)](api/languages/     | duct_opmiERR6sum_opI9HandlerTyE), |
| cpp_api.html#_CPPv4NK5cudaq14comp |     [\[                           |
| lex_matrix18minimal_eigenvalueEv) | 14\]](api/languages/cpp_api.html# |
| -   [                             | _CPPv4NKR5cudaq10product_opmiEv), |
| cudaq::complex_matrix::operator() |     [\[15\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     function)](api/languages/cpp  | opmiERK10product_opI9HandlerTyE), |
| _api.html#_CPPv4N5cudaq14complex_ |     [\[16\]](api/languag          |
| matrixclENSt6size_tENSt6size_tE), | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     [\[1\]](api/languages/cpp     | roduct_opmiERK15scalar_operator), |
| _api.html#_CPPv4NK5cudaq14complex |     [\[17\]](api/languages        |
| _matrixclENSt6size_tENSt6size_tE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [                             | duct_opmiERK6sum_opI9HandlerTyE), |
| cudaq::complex_matrix::operator\* |     [\[18\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     function)](api/langua         | opmiERR10product_opI9HandlerTyE), |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     [\[19\]](api/languag          |
| omplex_matrixmlEN14complex_matrix | es/cpp_api.html#_CPPv4NO5cudaq10p |
| 10value_typeERK14complex_matrix), | roduct_opmiERR15scalar_operator), |
|     [\[1\]                        |     [\[20\]](api/languages        |
| ](api/languages/cpp_api.html#_CPP | /cpp_api.html#_CPPv4NO5cudaq10pro |
| v4N5cudaq14complex_matrixmlERK14c | duct_opmiERR6sum_opI9HandlerTyE), |
| omplex_matrixRK14complex_matrix), |     [                             |
|                                   | \[21\]](api/languages/cpp_api.htm |
|  [\[2\]](api/languages/cpp_api.ht | l#_CPPv4NO5cudaq10product_opmiEv) |
| ml#_CPPv4N5cudaq14complex_matrixm | -   [cudaq::product_op::operator/ |
| lERK14complex_matrixRKNSt6vectorI |     (C++                          |
| N14complex_matrix10value_typeEEE) |     function)](api/language       |
| -                                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| [cudaq::complex_matrix::operator+ | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/language          |
|     function                      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| )](api/languages/cpp_api.html#_CP | roduct_opdvERR15scalar_operator), |
| Pv4N5cudaq14complex_matrixplERK14 |     [\[2\]](api/languag           |
| complex_matrixRK14complex_matrix) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -                                 | roduct_opdvERK15scalar_operator), |
| [cudaq::complex_matrix::operator- |     [\[3\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     function                      | product_opdvERR15scalar_operator) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq14complex_matrixmiERK14 |    [cudaq::product_op::operator/= |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -   [cu                           |     function)](api/langu          |
| daq::complex_matrix::operator\[\] | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     (C++                          | product_opdVERK15scalar_operator) |
|                                   | -   [cudaq::product_op::operator= |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4N5cudaq14complex_matr |     function)](api/la             |
| ixixERKNSt6vectorINSt6size_tEEE), | nguages/cpp_api.html#_CPPv4I0_NSt |
|     [\[1\]](api/languages/cpp_api | 11enable_if_tIXaantNSt7is_sameI1T |
| .html#_CPPv4NK5cudaq14complex_mat | 9HandlerTyE5valueENSt16is_constru |
| rixixERKNSt6vectorINSt6size_tEEE) | ctibleI9HandlerTy1TE5valueEEbEEEN |
| -   [cudaq::complex_matrix::power | 5cudaq10product_opaSER10product_o |
|     (C++                          | pI9HandlerTyERK10product_opI1TE), |
|     function)]                    |     [\[1\]](api/languages/cpp     |
| (api/languages/cpp_api.html#_CPPv | _api.html#_CPPv4N5cudaq10product_ |
| 4N5cudaq14complex_matrix5powerEi) | opaSERK10product_opI9HandlerTyE), |
| -                                 |     [\[2\]](api/languages/cp      |
|  [cudaq::complex_matrix::set_zero | p_api.html#_CPPv4N5cudaq10product |
|     (C++                          | _opaSERR10product_opI9HandlerTyE) |
|     function)](ap                 | -                                 |
| i/languages/cpp_api.html#_CPPv4N5 |    [cudaq::product_op::operator== |
| cudaq14complex_matrix8set_zeroEv) |     (C++                          |
| -                                 |     function)](api/languages/cpp  |
| [cudaq::complex_matrix::to_string | _api.html#_CPPv4NK5cudaq10product |
|     (C++                          | _opeqERK10product_opI9HandlerTyE) |
|     function)](api/               | -                                 |
| languages/cpp_api.html#_CPPv4NK5c |  [cudaq::product_op::operator\[\] |
| udaq14complex_matrix9to_stringEv) |     (C++                          |
| -   [                             |     function)](ap                 |
| cudaq::complex_matrix::value_type | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq10product_opixENSt6size_tE) |
|     type)](api/                   | -                                 |
| languages/cpp_api.html#_CPPv4N5cu |    [cudaq::product_op::product_op |
| daq14complex_matrix10value_typeE) |     (C++                          |
| -   [cudaq::contrib (C++          |     function)](api/languages/c    |
|     type)](api/languages/cpp      | pp_api.html#_CPPv4I0_NSt11enable_ |
| _api.html#_CPPv4N5cudaq7contribE) | if_tIXaaNSt7is_sameI9HandlerTy14m |
| -   [cudaq::contrib::draw (C++    | atrix_handlerE5valueEaantNSt7is_s |
|     function)                     | ameI1T9HandlerTyE5valueENSt16is_c |
| ](api/languages/cpp_api.html#_CPP | onstructibleI9HandlerTy1TE5valueE |
| v4I0DpEN5cudaq7contrib4drawENSt6s | EbEEEN5cudaq10product_op10product |
| tringERR13QuantumKernelDpRR4Args) | _opERK10product_opI1TERKN14matrix |
| -                                 | _handler20commutation_behaviorE), |
| [cudaq::contrib::get_unitary_cmat |                                   |
|     (C++                          |  [\[1\]](api/languages/cpp_api.ht |
|     function)](api/languages/cp   | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| p_api.html#_CPPv4I0DpEN5cudaq7con | tNSt7is_sameI1T9HandlerTyE5valueE |
| trib16get_unitary_cmatE14complex_ | NSt16is_constructibleI9HandlerTy1 |
| matrixRR13QuantumKernelDpRR4Args) | TE5valueEEbEEEN5cudaq10product_op |
| -   [cudaq::CusvState (C++        | 10product_opERK10product_opI1TE), |
|                                   |                                   |
|    class)](api/languages/cpp_api. |   [\[2\]](api/languages/cpp_api.h |
| html#_CPPv4I0EN5cudaq9CusvStateE) | tml#_CPPv4N5cudaq10product_op10pr |
| -   [cudaq::depolarization1 (C++  | oduct_opENSt6size_tENSt6size_tE), |
|     c                             |     [\[3\]](api/languages/cp      |
| lass)](api/languages/cpp_api.html | p_api.html#_CPPv4N5cudaq10product |
| #_CPPv4N5cudaq15depolarization1E) | _op10product_opENSt7complexIdEE), |
| -   [cudaq::depolarization2 (C++  |     [\[4\]](api/l                 |
|     c                             | anguages/cpp_api.html#_CPPv4N5cud |
| lass)](api/languages/cpp_api.html | aq10product_op10product_opERK10pr |
| #_CPPv4N5cudaq15depolarization2E) | oduct_opI9HandlerTyENSt6size_tE), |
| -   [cudaq:                       |     [\[5\]](api/l                 |
| :depolarization2::depolarization2 | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq10product_op10product_opERR10pr |
|     function)](api/languages/cp   | oduct_opI9HandlerTyENSt6size_tE), |
| p_api.html#_CPPv4N5cudaq15depolar |     [\[6\]](api/languages         |
| ization215depolarization2EK4real) | /cpp_api.html#_CPPv4N5cudaq10prod |
| -   [cudaq                        | uct_op10product_opERR9HandlerTy), |
| ::depolarization2::num_parameters |     [\[7\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     member)](api/langu            | cudaq10product_op10product_opEd), |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     [\[8\]](a                     |
| depolarization214num_parametersE) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cu                           | 5cudaq10product_op10product_opEv) |
| daq::depolarization2::num_targets | -   [cuda                         |
|     (C++                          | q::product_op::to_diagonal_matrix |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/               |
| q15depolarization211num_targetsE) | languages/cpp_api.html#_CPPv4NK5c |
| -                                 | udaq10product_op18to_diagonal_mat |
|    [cudaq::depolarization_channel | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     class)](                      | apINSt6stringENSt7complexIdEEEEb) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::product_op::to_matrix |
| N5cudaq22depolarization_channelE) |     (C++                          |
| -   [cudaq::depol                 |     funct                         |
| arization_channel::num_parameters | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NK5cudaq10product_op9to_mat |
|     member)](api/languages/cp     | rixENSt13unordered_mapINSt6size_t |
| p_api.html#_CPPv4N5cudaq22depolar | ENSt7int64_tEEERKNSt13unordered_m |
| ization_channel14num_parametersE) | apINSt6stringENSt7complexIdEEEEb) |
| -   [cudaq::de                    | -   [cu                           |
| polarization_channel::num_targets | daq::product_op::to_sparse_matrix |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)](ap                 |
| /cpp_api.html#_CPPv4N5cudaq22depo | i/languages/cpp_api.html#_CPPv4NK |
| larization_channel11num_targetsE) | 5cudaq10product_op16to_sparse_mat |
| -   [cudaq::details (C++          | rixENSt13unordered_mapINSt6size_t |
|     type)](api/languages/cpp      | ENSt7int64_tEEERKNSt13unordered_m |
| _api.html#_CPPv4N5cudaq7detailsE) | apINSt6stringENSt7complexIdEEEEb) |
| -   [cudaq::details::future (C++  | -   [cudaq::product_op::to_string |
|                                   |     (C++                          |
|  class)](api/languages/cpp_api.ht |     function)](                   |
| ml#_CPPv4N5cudaq7details6futureE) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | NK5cudaq10product_op9to_stringEv) |
|   [cudaq::details::future::future | -                                 |
|     (C++                          |  [cudaq::product_op::\~product_op |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |     fu                            |
| PPv4N5cudaq7details6future6future | nction)](api/languages/cpp_api.ht |
| ERNSt6vectorI3JobEERNSt6stringERN | ml#_CPPv4N5cudaq10product_opD0Ev) |
| St3mapINSt6stringENSt6stringEEE), | -   [cudaq::QPU (C++              |
|     [\[1\]](api/lang              |     class)](api/languages         |
| uages/cpp_api.html#_CPPv4N5cudaq7 | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| details6future6futureERR6future), | -   [cudaq::QPU::enqueue (C++     |
|     [\[2\]]                       |     function)](ap                 |
| (api/languages/cpp_api.html#_CPPv | i/languages/cpp_api.html#_CPPv4N5 |
| 4N5cudaq7details6future6futureEv) | cudaq3QPU7enqueueER11QuantumTask) |
| -   [cu                           | -   [cudaq::QPU::getConnectivity  |
| daq::details::kernel_builder_base |     (C++                          |
|     (C++                          |     function)                     |
|     class)](api/l                 | ](api/languages/cpp_api.html#_CPP |
| anguages/cpp_api.html#_CPPv4N5cud | v4N5cudaq3QPU15getConnectivityEv) |
| aq7details19kernel_builder_baseE) | -                                 |
| -   [cudaq::details::             | [cudaq::QPU::getExecutionThreadId |
| kernel_builder_base::operator\<\< |     (C++                          |
|     (C++                          |     function)](api/               |
|     function)](api/langua         | languages/cpp_api.html#_CPPv4NK5c |
| ges/cpp_api.html#_CPPv4N5cudaq7de | udaq3QPU20getExecutionThreadIdEv) |
| tails19kernel_builder_baselsERNSt | -   [cudaq::QPU::getNumQubits     |
| 7ostreamERK19kernel_builder_base) |     (C++                          |
| -   [                             |     functi                        |
| cudaq::details::KernelBuilderType | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     class)](api                   | -   [                             |
| /languages/cpp_api.html#_CPPv4N5c | cudaq::QPU::getRemoteCapabilities |
| udaq7details17KernelBuilderTypeE) |     (C++                          |
| -   [cudaq::d                     |     function)](api/l              |
| etails::KernelBuilderType::create | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq3QPU21getRemoteCapabilitiesEv) |
|     function)                     | -   [cudaq::QPU::isEmulated (C++  |
| ](api/languages/cpp_api.html#_CPP |     func                          |
| v4N5cudaq7details17KernelBuilderT | tion)](api/languages/cpp_api.html |
| ype6createEPN4mlir11MLIRContextE) | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| -   [cudaq::details::Ker          | -   [cudaq::QPU::isSimulator (C++ |
| nelBuilderType::KernelBuilderType |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/lang           | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -   [cudaq::QPU::launchKernel     |
| details17KernelBuilderType17Kerne |     (C++                          |
| lBuilderTypeERRNSt8functionIFN4ml |     function)](api/               |
| ir4TypeEPN4mlir11MLIRContextEEEE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::diag_matrix_callback  | daq3QPU12launchKernelERKNSt6strin |
|     (C++                          | gE15KernelThunkTypePvNSt8uint64_t |
|     class)                        | ENSt8uint64_tERKNSt6vectorIPvEE), |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4N5cudaq20diag_matrix_callbackE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::dyn (C++              | ml#_CPPv4N5cudaq3QPU12launchKerne |
|     member)](api/languages        | lERKNSt6stringERKNSt6vectorIPvEE) |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | -   [cudaq::QPU::onRandomSeedSet  |
| -   [cudaq::ExecutionContext (C++ |     (C++                          |
|     cl                            |     function)](api/lang           |
| ass)](api/languages/cpp_api.html# | uages/cpp_api.html#_CPPv4N5cudaq3 |
| _CPPv4N5cudaq16ExecutionContextE) | QPU15onRandomSeedSetENSt6size_tE) |
| -   [cudaq                        | -   [cudaq::QPU::QPU (C++         |
| ::ExecutionContext::amplitudeMaps |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](api/langu            | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |                                   |
| ExecutionContext13amplitudeMapsE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [c                            | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| udaq::ExecutionContext::asyncExec |     [\[2\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     member)](api/                 | -   [                             |
| languages/cpp_api.html#_CPPv4N5cu | cudaq::QPU::resetExecutionContext |
| daq16ExecutionContext9asyncExecE) |     (C++                          |
| -   [cud                          |     function)](api/               |
| aq::ExecutionContext::asyncResult | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq3QPU21resetExecutionContextEv) |
|     member)](api/lan              | -                                 |
| guages/cpp_api.html#_CPPv4N5cudaq |  [cudaq::QPU::setExecutionContext |
| 16ExecutionContext11asyncResultE) |     (C++                          |
| -   [cudaq:                       |                                   |
| :ExecutionContext::batchIteration |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4N5cudaq3QPU19setExec |
|     member)](api/langua           | utionContextEP16ExecutionContext) |
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
| -                                 | -   [cudaq::qreg::operator\[\]    |
|  [cudaq::ExecutionContext::jitEng |     (C++                          |
|     (C++                          |     functi                        |
|     member)](a                    | on)](api/languages/cpp_api.html#_ |
| pi/languages/cpp_api.html#_CPPv4N | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| 5cudaq16ExecutionContext6jitEngE) | -   [cudaq::qreg::qreg (C++       |
| -   [cu                           |     function)                     |
| daq::ExecutionContext::kernelName | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4qregENSt6size_tE), |
|     member)](api/la               |     [\[1\]](api/languages/cpp_ap  |
| nguages/cpp_api.html#_CPPv4N5cuda | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| q16ExecutionContext10kernelNameE) | -   [cudaq::qreg::size (C++       |
| -   [cud                          |                                   |
| aq::ExecutionContext::kernelTrace |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     member)](api/lan              | -   [cudaq::qreg::slice (C++      |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/langu          |
| 16ExecutionContext11kernelTraceE) | ages/cpp_api.html#_CPPv4N5cudaq4q |
| -   [cudaq:                       | reg5sliceENSt6size_tENSt6size_tE) |
| :ExecutionContext::msm_dimensions | -   [cudaq::qreg::value_type (C++ |
|     (C++                          |                                   |
|     member)](api/langua           | type)](api/languages/cpp_api.html |
| ges/cpp_api.html#_CPPv4N5cudaq16E | #_CPPv4N5cudaq4qreg10value_typeE) |
| xecutionContext14msm_dimensionsE) | -   [cudaq::qspan (C++            |
| -   [cudaq::                      |     class)](api/lang              |
| ExecutionContext::msm_prob_err_id | uages/cpp_api.html#_CPPv4I_NSt6si |
|     (C++                          | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     member)](api/languag          | -   [cudaq::QuakeValue (C++       |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     class)](api/languages/cpp_api |
| ecutionContext15msm_prob_err_idE) | .html#_CPPv4N5cudaq10QuakeValueE) |
| -   [cudaq::Ex                    | -   [cudaq::Q                     |
| ecutionContext::msm_probabilities | uakeValue::canValidateNumElements |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)](api/languages      |
| /cpp_api.html#_CPPv4N5cudaq16Exec | /cpp_api.html#_CPPv4N5cudaq10Quak |
| utionContext17msm_probabilitiesE) | eValue22canValidateNumElementsEv) |
| -                                 | -                                 |
|    [cudaq::ExecutionContext::name |  [cudaq::QuakeValue::constantSize |
|     (C++                          |     (C++                          |
|     member)]                      |     function)](api                |
| (api/languages/cpp_api.html#_CPPv | /languages/cpp_api.html#_CPPv4N5c |
| 4N5cudaq16ExecutionContext4nameE) | udaq10QuakeValue12constantSizeEv) |
| -   [cu                           | -   [cudaq::QuakeValue::dump (C++ |
| daq::ExecutionContext::noiseModel |     function)](api/lan            |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     member)](api/la               | 10QuakeValue4dumpERNSt7ostreamE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\                            |
| q16ExecutionContext10noiseModelE) | [1\]](api/languages/cpp_api.html# |
| -   [cudaq::Exe                   | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| cutionContext::numberTrajectories | -   [cudaq                        |
|     (C++                          | ::QuakeValue::getRequiredElements |
|     member)](api/languages/       |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16Execu |     function)](api/langua         |
| tionContext18numberTrajectoriesE) | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| -   [c                            | uakeValue19getRequiredElementsEv) |
| udaq::ExecutionContext::optResult | -   [cudaq::QuakeValue::getValue  |
|     (C++                          |     (C++                          |
|     member)](api/                 |     function)]                    |
| languages/cpp_api.html#_CPPv4N5cu | (api/languages/cpp_api.html#_CPPv |
| daq16ExecutionContext9optResultE) | 4NK5cudaq10QuakeValue8getValueEv) |
| -   [cudaq::Execu                 | -   [cudaq::QuakeValue::inverse   |
| tionContext::overlapComputeStates |     (C++                          |
|     (C++                          |     function)                     |
|     member)](api/languages/cp     | ](api/languages/cpp_api.html#_CPP |
| p_api.html#_CPPv4N5cudaq16Executi | v4NK5cudaq10QuakeValue7inverseEv) |
| onContext20overlapComputeStatesE) | -   [cudaq::QuakeValue::isStdVec  |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::overlapResult |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/langu            | v4N5cudaq10QuakeValue8isStdVecEv) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -                                 |
| ExecutionContext13overlapResultE) |    [cudaq::QuakeValue::operator\* |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::qpuId |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](                     | udaq10QuakeValuemlE10QuakeValue), |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq16ExecutionContext5qpuIdE) | [\[1\]](api/languages/cpp_api.htm |
| -   [cudaq                        | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| ::ExecutionContext::registerNames | -   [cudaq::QuakeValue::operator+ |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api                |
| ages/cpp_api.html#_CPPv4N5cudaq16 | /languages/cpp_api.html#_CPPv4N5c |
| ExecutionContext13registerNamesE) | udaq10QuakeValueplE10QuakeValue), |
| -   [cu                           |     [                             |
| daq::ExecutionContext::reorderIdx | \[1\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda | [\[2\]](api/languages/cpp_api.htm |
| q16ExecutionContext10reorderIdxE) | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| -                                 | -   [cudaq::QuakeValue::operator- |
|  [cudaq::ExecutionContext::result |     (C++                          |
|     (C++                          |     function)](api                |
|     member)](a                    | /languages/cpp_api.html#_CPPv4N5c |
| pi/languages/cpp_api.html#_CPPv4N | udaq10QuakeValuemiE10QuakeValue), |
| 5cudaq16ExecutionContext6resultE) |     [                             |
| -                                 | \[1\]](api/languages/cpp_api.html |
|   [cudaq::ExecutionContext::shots | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     (C++                          |     [                             |
|     member)](                     | \[2\]](api/languages/cpp_api.html |
| api/languages/cpp_api.html#_CPPv4 | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| N5cudaq16ExecutionContext5shotsE) |                                   |
| -   [cudaq::                      | [\[3\]](api/languages/cpp_api.htm |
| ExecutionContext::simulationState | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     (C++                          | -   [cudaq::QuakeValue::operator/ |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api                |
| ecutionContext15simulationStateE) | /languages/cpp_api.html#_CPPv4N5c |
| -                                 | udaq10QuakeValuedvE10QuakeValue), |
|    [cudaq::ExecutionContext::spin |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     member)]                      | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq16ExecutionContext4spinE) |  [cudaq::QuakeValue::operator\[\] |
| -   [cudaq::                      |     (C++                          |
| ExecutionContext::totalIterations |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/languag          | udaq10QuakeValueixEKNSt6size_tE), |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     [\[1\]](api/                  |
| ecutionContext15totalIterationsE) | languages/cpp_api.html#_CPPv4N5cu |
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
|     [\[1\]](api/lan               | -   [cuda                         |
| guages/cpp_api.html#_CPPv4N5cudaq | q::quantum_platform::connectivity |
| 15ExecutionResult15ExecutionResul |     (C++                          |
| tE16CountsDictionaryNSt6stringE), |     function)](api/langu          |
|     [\[2\                         | ages/cpp_api.html#_CPPv4N5cudaq16 |
| ]](api/languages/cpp_api.html#_CP | quantum_platform12connectivityEv) |
| Pv4N5cudaq15ExecutionResult15Exec | -   [cudaq::q                     |
| utionResultE16CountsDictionaryd), | uantum_platform::enqueueAsyncTask |
|                                   |     (C++                          |
|    [\[3\]](api/languages/cpp_api. |     function)](api/languages/     |
| html#_CPPv4N5cudaq15ExecutionResu | cpp_api.html#_CPPv4N5cudaq16quant |
| lt15ExecutionResultENSt6stringE), | um_platform16enqueueAsyncTaskEKNS |
|     [\[4\                         | t6size_tER19KernelExecutionTask), |
| ]](api/languages/cpp_api.html#_CP |     [\[1\]](api/languag           |
| Pv4N5cudaq15ExecutionResult15Exec | es/cpp_api.html#_CPPv4N5cudaq16qu |
| utionResultERK15ExecutionResult), | antum_platform16enqueueAsyncTaskE |
|     [\[5\]](api/language          | KNSt6size_tERNSt8functionIFvvEEE) |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | -   [cudaq::qua                   |
| cutionResult15ExecutionResultEd), | ntum_platform::get_codegen_config |
|     [\[6\]](api/languag           |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |     function)](api/languages/c    |
| ecutionResult15ExecutionResultEv) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [                             | m_platform18get_codegen_configEv) |
| cudaq::ExecutionResult::operator= | -   [cudaq::                      |
|     (C++                          | quantum_platform::get_current_qpu |
|     function)](api/languages/     |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq15Execu |     function)](api/languages      |
| tionResultaSERK15ExecutionResult) | /cpp_api.html#_CPPv4NK5cudaq16qua |
| -   [c                            | ntum_platform15get_current_qpuEv) |
| udaq::ExecutionResult::operator== | -   [cuda                         |
|     (C++                          | q::quantum_platform::get_exec_ctx |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4NK5cudaq15Execu |     function)](api/langua         |
| tionResulteqERK15ExecutionResult) | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| -   [cud                          | quantum_platform12get_exec_ctxEv) |
| aq::ExecutionResult::registerName | -   [c                            |
|     (C++                          | udaq::quantum_platform::get_noise |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languages/c    |
| 15ExecutionResult12registerNameE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [cudaq                        | m_platform9get_noiseENSt6size_tE) |
| ::ExecutionResult::sequentialData | -   [cudaq:                       |
|     (C++                          | :quantum_platform::get_num_qubits |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |                                   |
| ExecutionResult14sequentialDataE) | function)](api/languages/cpp_api. |
| -   [                             | html#_CPPv4NK5cudaq16quantum_plat |
| cudaq::ExecutionResult::serialize | form14get_num_qubitsENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_              |
|     function)](api/l              | platform::get_remote_capabilities |
| anguages/cpp_api.html#_CPPv4NK5cu |     (C++                          |
| daq15ExecutionResult9serializeEv) |     function)                     |
| -   [cudaq::fermion_handler (C++  | ](api/languages/cpp_api.html#_CPP |
|     c                             | v4NK5cudaq16quantum_platform23get |
| lass)](api/languages/cpp_api.html | _remote_capabilitiesENSt6size_tE) |
| #_CPPv4N5cudaq15fermion_handlerE) | -   [cudaq::qua                   |
| -   [cudaq::fermion_op (C++       | ntum_platform::get_runtime_target |
|     type)](api/languages/cpp_api  |     (C++                          |
| .html#_CPPv4N5cudaq10fermion_opE) |     function)](api/languages/cp   |
| -   [cudaq::fermion_op_term (C++  | p_api.html#_CPPv4NK5cudaq16quantu |
|                                   | m_platform18get_runtime_targetEv) |
| type)](api/languages/cpp_api.html | -   [cuda                         |
| #_CPPv4N5cudaq15fermion_op_termE) | q::quantum_platform::getLogStream |
| -   [cudaq::FermioniqBaseQPU (C++ |     (C++                          |
|     cl                            |     function)](api/langu          |
| ass)](api/languages/cpp_api.html# | ages/cpp_api.html#_CPPv4N5cudaq16 |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | quantum_platform12getLogStreamEv) |
| -   [cudaq::get_state (C++        | -   [cud                          |
|                                   | aq::quantum_platform::is_emulated |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |                                   |
| ateEDaRR13QuantumKernelDpRR4Args) |    function)](api/languages/cpp_a |
| -   [cudaq::gradient (C++         | pi.html#_CPPv4NK5cudaq16quantum_p |
|     class)](api/languages/cpp_    | latform11is_emulatedENSt6size_tE) |
| api.html#_CPPv4N5cudaq8gradientE) | -   [c                            |
| -   [cudaq::gradient::clone (C++  | udaq::quantum_platform::is_remote |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     function)](api/languages/cp   |
| l#_CPPv4N5cudaq8gradient5cloneEv) | p_api.html#_CPPv4NK5cudaq16quantu |
| -   [cudaq::gradient::compute     | m_platform9is_remoteENSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     function)](api/language       | q::quantum_platform::is_simulator |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     (C++                          |
| ient7computeERKNSt6vectorIdEERKNS |                                   |
| t8functionIFdNSt6vectorIdEEEEEd), |   function)](api/languages/cpp_ap |
|     [\[1\]](ap                    | i.html#_CPPv4NK5cudaq16quantum_pl |
| i/languages/cpp_api.html#_CPPv4N5 | atform12is_simulatorENSt6size_tE) |
| cudaq8gradient7computeERKNSt6vect | -   [c                            |
| orIdEERNSt6vectorIdEERK7spin_opd) | udaq::quantum_platform::launchVQE |
| -   [cudaq::gradient::gradient    |     (C++                          |
|     (C++                          |     function)](                   |
|     function)](api/lang           | api/languages/cpp_api.html#_CPPv4 |
| uages/cpp_api.html#_CPPv4I00EN5cu | N5cudaq16quantum_platform9launchV |
| daq8gradient8gradientER7KernelT), | QEEKNSt6stringEPKvPN5cudaq8gradie |
|                                   | ntERKN5cudaq7spin_opERN5cudaq9opt |
|    [\[1\]](api/languages/cpp_api. | imizerEKiKNSt6size_tENSt6size_tE) |
| html#_CPPv4I00EN5cudaq8gradient8g | -   [cudaq:                       |
| radientER7KernelTRR10ArgsMapper), | :quantum_platform::list_platforms |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     function)](api/languag        |
| Pv4I00EN5cudaq8gradient8gradientE | es/cpp_api.html#_CPPv4N5cudaq16qu |
| RR13QuantumKernelRR10ArgsMapper), | antum_platform14list_platformsEv) |
|     [\[3                          | -                                 |
| \]](api/languages/cpp_api.html#_C |    [cudaq::quantum_platform::name |
| PPv4N5cudaq8gradient8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](a                  |
|     [\[                           | pi/languages/cpp_api.html#_CPPv4N |
| 4\]](api/languages/cpp_api.html#_ | K5cudaq16quantum_platform4nameEv) |
| CPPv4N5cudaq8gradient8gradientEv) | -   [                             |
| -   [cudaq::gradient::setArgs     | cudaq::quantum_platform::num_qpus |
|     (C++                          |     (C++                          |
|     fu                            |     function)](api/l              |
| nction)](api/languages/cpp_api.ht | anguages/cpp_api.html#_CPPv4NK5cu |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | daq16quantum_platform8num_qpusEv) |
| tArgsEvR13QuantumKernelDpRR4Args) | -   [cudaq::                      |
| -   [cudaq::gradient::setKernel   | quantum_platform::onRandomSeedSet |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |                                   |
| pp_api.html#_CPPv4I0EN5cudaq8grad | function)](api/languages/cpp_api. |
| ient9setKernelEvR13QuantumKernel) | html#_CPPv4N5cudaq16quantum_platf |
| -   [cud                          | orm15onRandomSeedSetENSt6size_tE) |
| aq::gradients::central_difference | -   [cudaq:                       |
|     (C++                          | :quantum_platform::reset_exec_ctx |
|     class)](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languag        |
| q9gradients18central_differenceE) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cudaq::gra                   | antum_platform14reset_exec_ctxEv) |
| dients::central_difference::clone | -   [cud                          |
|     (C++                          | aq::quantum_platform::reset_noise |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)](api/languages/cpp_ |
| ents18central_difference5cloneEv) | api.html#_CPPv4N5cudaq16quantum_p |
| -   [cudaq::gradi                 | latform11reset_noiseENSt6size_tE) |
| ents::central_difference::compute | -   [cudaq:                       |
|     (C++                          | :quantum_platform::resetLogStream |
|     function)](                   |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languag        |
| N5cudaq9gradients18central_differ | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ence7computeERKNSt6vectorIdEERKNS | antum_platform14resetLogStreamEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cuda                         |
|                                   | q::quantum_platform::set_exec_ctx |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq9gradients18cent |     funct                         |
| ral_difference7computeERKNSt6vect | ion)](api/languages/cpp_api.html# |
| orIdEERNSt6vectorIdEERK7spin_opd) | _CPPv4N5cudaq16quantum_platform12 |
| -   [cudaq::gradie                | set_exec_ctxEP16ExecutionContext) |
| nts::central_difference::gradient | -   [c                            |
|     (C++                          | udaq::quantum_platform::set_noise |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |     function                      |
| PPv4I00EN5cudaq9gradients18centra | )](api/languages/cpp_api.html#_CP |
| l_difference8gradientER7KernelT), | Pv4N5cudaq16quantum_platform9set_ |
|     [\[1\]](api/langua            | noiseEPK11noise_modelNSt6size_tE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cuda                         |
| q9gradients18central_difference8g | q::quantum_platform::setLogStream |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\]](api/languages/cpp_    |                                   |
| api.html#_CPPv4I00EN5cudaq9gradie |  function)](api/languages/cpp_api |
| nts18central_difference8gradientE | .html#_CPPv4N5cudaq16quantum_plat |
| RR13QuantumKernelRR10ArgsMapper), | form12setLogStreamERNSt7ostreamE) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::q                     |
| _api.html#_CPPv4N5cudaq9gradients | uantum_platform::setTargetBackend |
| 18central_difference8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     fun                           |
|     [\[4\]](api/languages/cp      | ction)](api/languages/cpp_api.htm |
| p_api.html#_CPPv4N5cudaq9gradient | l#_CPPv4N5cudaq16quantum_platform |
| s18central_difference8gradientEv) | 16setTargetBackendERKNSt6stringE) |
| -   [cud                          | -   [cudaq::quantum_platfo        |
| aq::gradients::forward_difference | rm::supports_conditional_feedback |
|     (C++                          |     (C++                          |
|     class)](api/la                |     function)](api/               |
| nguages/cpp_api.html#_CPPv4N5cuda | languages/cpp_api.html#_CPPv4NK5c |
| q9gradients18forward_differenceE) | udaq16quantum_platform29supports_ |
| -   [cudaq::gra                   | conditional_feedbackENSt6size_tE) |
| dients::forward_difference::clone | -   [cudaq::quantum_platfor       |
|     (C++                          | m::supports_explicit_measurements |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)](api/l              |
| ents18forward_difference5cloneEv) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cudaq::gradi                 | daq16quantum_platform30supports_e |
| ents::forward_difference::compute | xplicit_measurementsENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_pla           |
|     function)](                   | tform::supports_task_distribution |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq9gradients18forward_differ |     fu                            |
| ence7computeERKNSt6vectorIdEERKNS | nction)](api/languages/cpp_api.ht |
| t8functionIFdNSt6vectorIdEEEEEd), | ml#_CPPv4NK5cudaq16quantum_platfo |
|                                   | rm26supports_task_distributionEv) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::QuantumTask (C++      |
| tml#_CPPv4N5cudaq9gradients18forw |     type)](api/languages/cpp_api. |
| ard_difference7computeERKNSt6vect | html#_CPPv4N5cudaq11QuantumTaskE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::qubit (C++            |
| -   [cudaq::gradie                |     type)](api/languages/c        |
| nts::forward_difference::gradient | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     (C++                          | -   [cudaq::QubitConnectivity     |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |     ty                            |
| PPv4I00EN5cudaq9gradients18forwar | pe)](api/languages/cpp_api.html#_ |
| d_difference8gradientER7KernelT), | CPPv4N5cudaq17QubitConnectivityE) |
|     [\[1\]](api/langua            | -   [cudaq::QubitEdge (C++        |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     type)](api/languages/cpp_a    |
| q9gradients18forward_difference8g | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::qudit (C++            |
|     [\[2\]](api/languages/cpp_    |     clas                          |
| api.html#_CPPv4I00EN5cudaq9gradie | s)](api/languages/cpp_api.html#_C |
| nts18forward_difference8gradientE | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::qudit::qudit (C++     |
|     [\[3\]](api/languages/cpp     |                                   |
| _api.html#_CPPv4N5cudaq9gradients | function)](api/languages/cpp_api. |
| 18forward_difference8gradientERRN | html#_CPPv4N5cudaq5qudit5quditEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::qvector (C++          |
|     [\[4\]](api/languages/cp      |     class)                        |
| p_api.html#_CPPv4N5cudaq9gradient | ](api/languages/cpp_api.html#_CPP |
| s18forward_difference8gradientEv) | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| -   [                             | -   [cudaq::qvector::back (C++    |
| cudaq::gradients::parameter_shift |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     class)](api                   | 5cudaq7qvector4backENSt6size_tE), |
| /languages/cpp_api.html#_CPPv4N5c |                                   |
| udaq9gradients15parameter_shiftE) |   [\[1\]](api/languages/cpp_api.h |
| -   [cudaq::                      | tml#_CPPv4N5cudaq7qvector4backEv) |
| gradients::parameter_shift::clone | -   [cudaq::qvector::begin (C++   |
|     (C++                          |     fu                            |
|     function)](api/langua         | nction)](api/languages/cpp_api.ht |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | ml#_CPPv4N5cudaq7qvector5beginEv) |
| adients15parameter_shift5cloneEv) | -   [cudaq::qvector::clear (C++   |
| -   [cudaq::gr                    |     fu                            |
| adients::parameter_shift::compute | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     function                      | -   [cudaq::qvector::end (C++     |
| )](api/languages/cpp_api.html#_CP |                                   |
| Pv4N5cudaq9gradients15parameter_s | function)](api/languages/cpp_api. |
| hift7computeERKNSt6vectorIdEERKNS | html#_CPPv4N5cudaq7qvector3endEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::qvector::front (C++   |
|     [\[1\]](api/languages/cpp_ap  |     function)](ap                 |
| i.html#_CPPv4N5cudaq9gradients15p | i/languages/cpp_api.html#_CPPv4N5 |
| arameter_shift7computeERKNSt6vect | cudaq7qvector5frontENSt6size_tE), |
| orIdEERNSt6vectorIdEERK7spin_opd) |                                   |
| -   [cudaq::gra                   |  [\[1\]](api/languages/cpp_api.ht |
| dients::parameter_shift::gradient | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     (C++                          | -   [cudaq::qvector::operator=    |
|     func                          |     (C++                          |
| tion)](api/languages/cpp_api.html |     functio                       |
| #_CPPv4I00EN5cudaq9gradients15par | n)](api/languages/cpp_api.html#_C |
| ameter_shift8gradientER7KernelT), | PPv4N5cudaq7qvectoraSERK7qvector) |
|     [\[1\]](api/lan               | -   [cudaq::qvector::operator\[\] |
| guages/cpp_api.html#_CPPv4I00EN5c |     (C++                          |
| udaq9gradients15parameter_shift8g |     function)                     |
| radientER7KernelTRR10ArgsMapper), | ](api/languages/cpp_api.html#_CPP |
|     [\[2\]](api/languages/c       | v4N5cudaq7qvectorixEKNSt6size_tE) |
| pp_api.html#_CPPv4I00EN5cudaq9gra | -   [cudaq::qvector::qvector (C++ |
| dients15parameter_shift8gradientE |     function)](api/               |
| RR13QuantumKernelRR10ArgsMapper), | languages/cpp_api.html#_CPPv4N5cu |
|     [\[3\]](api/languages/        | daq7qvector7qvectorENSt6size_tE), |
| cpp_api.html#_CPPv4N5cudaq9gradie |     [\[1\]](a                     |
| nts15parameter_shift8gradientERRN | pi/languages/cpp_api.html#_CPPv4N |
| St8functionIFvNSt6vectorIdEEEEE), | 5cudaq7qvector7qvectorERK5state), |
|     [\[4\]](api/languages         |     [\[2\]](api                   |
| /cpp_api.html#_CPPv4N5cudaq9gradi | /languages/cpp_api.html#_CPPv4N5c |
| ents15parameter_shift8gradientEv) | udaq7qvector7qvectorERK7qvector), |
| -   [cudaq::kernel_builder (C++   |     [\[3\]](ap                    |
|     clas                          | i/languages/cpp_api.html#_CPPv4N5 |
| s)](api/languages/cpp_api.html#_C | cudaq7qvector7qvectorERR7qvector) |
| PPv4IDpEN5cudaq14kernel_builderE) | -   [cudaq::qvector::size (C++    |
| -   [c                            |     fu                            |
| udaq::kernel_builder::constantVal | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     function)](api/la             | -   [cudaq::qvector::slice (C++   |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/language       |
| q14kernel_builder11constantValEd) | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| -   [cu                           | tor5sliceENSt6size_tENSt6size_tE) |
| daq::kernel_builder::getArguments | -   [cudaq::qvector::value_type   |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     typ                           |
| guages/cpp_api.html#_CPPv4N5cudaq | e)](api/languages/cpp_api.html#_C |
| 14kernel_builder12getArgumentsEv) | PPv4N5cudaq7qvector10value_typeE) |
| -   [cu                           | -   [cudaq::qview (C++            |
| daq::kernel_builder::getNumParams |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     function)](api/lan            | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qview::back (C++      |
| 14kernel_builder12getNumParamsEv) |     function)                     |
| -   [c                            | ](api/languages/cpp_api.html#_CPP |
| udaq::kernel_builder::isArgStdVec | v4N5cudaq5qview4backENSt6size_tE) |
|     (C++                          | -   [cudaq::qview::begin (C++     |
|     function)](api/languages/cp   |                                   |
| p_api.html#_CPPv4N5cudaq14kernel_ | function)](api/languages/cpp_api. |
| builder11isArgStdVecENSt6size_tE) | html#_CPPv4N5cudaq5qview5beginEv) |
| -   [cuda                         | -   [cudaq::qview::end (C++       |
| q::kernel_builder::kernel_builder |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)](api/languages/cpp_ | i.html#_CPPv4N5cudaq5qview3endEv) |
| api.html#_CPPv4N5cudaq14kernel_bu | -   [cudaq::qview::front (C++     |
| ilder14kernel_builderERNSt6vector |     function)](                   |
| IN7details17KernelBuilderTypeEEE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::kernel_builder::name  | N5cudaq5qview5frontENSt6size_tE), |
|     (C++                          |                                   |
|     function)                     |    [\[1\]](api/languages/cpp_api. |
| ](api/languages/cpp_api.html#_CPP | html#_CPPv4N5cudaq5qview5frontEv) |
| v4N5cudaq14kernel_builder4nameEv) | -   [cudaq::qview::operator\[\]   |
| -                                 |     (C++                          |
|    [cudaq::kernel_builder::qalloc |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)](api/language       | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::qview::qview (C++     |
| nel_builder6qallocE10QuakeValue), |     functio                       |
|     [\[1\]](api/language          | n)](api/languages/cpp_api.html#_C |
| s/cpp_api.html#_CPPv4N5cudaq14ker | PPv4I0EN5cudaq5qview5qviewERR1R), |
| nel_builder6qallocEKNSt6size_tE), |     [\[1                          |
|     [\[2                          | \]](api/languages/cpp_api.html#_C |
| \]](api/languages/cpp_api.html#_C | PPv4N5cudaq5qview5qviewERK5qview) |
| PPv4N5cudaq14kernel_builder6qallo | -   [cudaq::qview::size (C++      |
| cERNSt6vectorINSt7complexIdEEEE), |                                   |
|     [\[3\]](                      | function)](api/languages/cpp_api. |
| api/languages/cpp_api.html#_CPPv4 | html#_CPPv4NK5cudaq5qview4sizeEv) |
| N5cudaq14kernel_builder6qallocEv) | -   [cudaq::qview::slice (C++     |
| -   [cudaq::kernel_builder::swap  |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|     function)](api/language       | iew5sliceENSt6size_tENSt6size_tE) |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | -   [cudaq::qview::value_type     |
| 4kernel_builder4swapEvRK10QuakeVa |     (C++                          |
| lueRK10QuakeValueRK10QuakeValue), |     t                             |
|                                   | ype)](api/languages/cpp_api.html# |
| [\[1\]](api/languages/cpp_api.htm | _CPPv4N5cudaq5qview10value_typeE) |
| l#_CPPv4I00EN5cudaq14kernel_build | -   [cudaq::range (C++            |
| er4swapEvRKNSt6vectorI10QuakeValu |     fun                           |
| eEERK10QuakeValueRK10QuakeValue), | ction)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| [\[2\]](api/languages/cpp_api.htm | orI11ElementTypeEE11ElementType), |
| l#_CPPv4N5cudaq14kernel_builder4s |     [\[1\]](api/languages/cpp_    |
| wapERK10QuakeValueRK10QuakeValue) | api.html#_CPPv4I0EN5cudaq5rangeEN |
| -   [cudaq::KernelExecutionTask   | St6vectorI11ElementTypeEE11Elemen |
|     (C++                          | tType11ElementType11ElementType), |
|     type                          |     [                             |
| )](api/languages/cpp_api.html#_CP | \[2\]](api/languages/cpp_api.html |
| Pv4N5cudaq19KernelExecutionTaskE) | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| -   [cudaq::KernelThunkResultType | -   [cudaq::real (C++             |
|     (C++                          |     type)](api/languages/         |
|     struct)]                      | cpp_api.html#_CPPv4N5cudaq4realE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::registry (C++         |
| 4N5cudaq21KernelThunkResultTypeE) |     type)](api/languages/cpp_     |
| -   [cudaq::KernelThunkType (C++  | api.html#_CPPv4N5cudaq8registryE) |
|                                   | -                                 |
| type)](api/languages/cpp_api.html |  [cudaq::registry::RegisteredType |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     (C++                          |
| -   [cudaq::kraus_channel (C++    |     class)](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|  class)](api/languages/cpp_api.ht | 5cudaq8registry14RegisteredTypeE) |
| ml#_CPPv4N5cudaq13kraus_channelE) | -   [cudaq::RemoteCapabilities    |
| -   [cudaq::kraus_channel::empty  |     (C++                          |
|     (C++                          |     struc                         |
|     function)]                    | t)](api/languages/cpp_api.html#_C |
| (api/languages/cpp_api.html#_CPPv | PPv4N5cudaq18RemoteCapabilitiesE) |
| 4NK5cudaq13kraus_channel5emptyEv) | -   [cudaq::Remo                  |
| -   [cudaq::kraus_c               | teCapabilities::isRemoteSimulator |
| hannel::generateUnitaryParameters |     (C++                          |
|     (C++                          |     member)](api/languages/c      |
|                                   | pp_api.html#_CPPv4N5cudaq18Remote |
|    function)](api/languages/cpp_a | Capabilities17isRemoteSimulatorE) |
| pi.html#_CPPv4N5cudaq13kraus_chan | -   [cudaq::Remot                 |
| nel25generateUnitaryParametersEv) | eCapabilities::RemoteCapabilities |
| -                                 |     (C++                          |
|    [cudaq::kraus_channel::get_ops |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4N5cudaq18RemoteCa |
|     function)](a                  | pabilities18RemoteCapabilitiesEb) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq:                       |
| K5cudaq13kraus_channel7get_opsEv) | :RemoteCapabilities::stateOverlap |
| -   [cudaq::                      |     (C++                          |
| kraus_channel::is_unitary_mixture |     member)](api/langua           |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq18R |
|     function)](api/languages      | emoteCapabilities12stateOverlapE) |
| /cpp_api.html#_CPPv4NK5cudaq13kra | -                                 |
| us_channel18is_unitary_mixtureEv) |   [cudaq::RemoteCapabilities::vqe |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::kraus_channel |     member)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/lang           | N5cudaq18RemoteCapabilities3vqeE) |
| uages/cpp_api.html#_CPPv4IDpEN5cu | -   [cudaq::RemoteSimulationState |
| daq13kraus_channel13kraus_channel |     (C++                          |
| EDpRRNSt16initializer_listI1TEE), |     class)]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|  [\[1\]](api/languages/cpp_api.ht | 4N5cudaq21RemoteSimulationStateE) |
| ml#_CPPv4N5cudaq13kraus_channel13 | -   [cudaq::Resources (C++        |
| kraus_channelERK13kraus_channel), |     class)](api/languages/cpp_a   |
|     [\[2\]                        | pi.html#_CPPv4N5cudaq9ResourcesE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::run (C++              |
| v4N5cudaq13kraus_channel13kraus_c |     function)]                    |
| hannelERKNSt6vectorI8kraus_opEE), | (api/languages/cpp_api.html#_CPPv |
|     [\[3\]                        | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| ](api/languages/cpp_api.html#_CPP | 5invoke_result_tINSt7decay_tI13Qu |
| v4N5cudaq13kraus_channel13kraus_c | antumKernelEEDpNSt7decay_tI4ARGSE |
| hannelERRNSt6vectorI8kraus_opEE), | EEEEENSt6size_tERN5cudaq11noise_m |
|     [\[4\]](api/lan               | odelERR13QuantumKernelDpRR4ARGS), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[1\]](api/langu             |
| 13kraus_channel13kraus_channelEv) | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| -                                 | daq3runENSt6vectorINSt15invoke_re |
| [cudaq::kraus_channel::noise_type | sult_tINSt7decay_tI13QuantumKerne |
|     (C++                          | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
|     member)](api                  | ize_tERR13QuantumKernelDpRR4ARGS) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::run_async (C++        |
| udaq13kraus_channel10noise_typeE) |     functio                       |
| -                                 | n)](api/languages/cpp_api.html#_C |
|  [cudaq::kraus_channel::operator= | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     (C++                          | tureINSt6vectorINSt15invoke_resul |
|     function)](api/langua         | t_tINSt7decay_tI13QuantumKernelEE |
| ges/cpp_api.html#_CPPv4N5cudaq13k | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| raus_channelaSERK13kraus_channel) | ze_tENSt6size_tERN5cudaq11noise_m |
| -   [c                            | odelERR13QuantumKernelDpRR4ARGS), |
| udaq::kraus_channel::operator\[\] |     [\[1\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4I0DpEN |
|     function)](api/l              | 5cudaq9run_asyncENSt6futureINSt6v |
| anguages/cpp_api.html#_CPPv4N5cud | ectorINSt15invoke_result_tINSt7de |
| aq13kraus_channelixEKNSt6size_tE) | cay_tI13QuantumKernelEEDpNSt7deca |
| -                                 | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| [cudaq::kraus_channel::parameters | ize_tERR13QuantumKernelDpRR4ARGS) |
|     (C++                          | -   [cudaq::RuntimeTarget (C++    |
|     member)](api                  |                                   |
| /languages/cpp_api.html#_CPPv4N5c | struct)](api/languages/cpp_api.ht |
| udaq13kraus_channel10parametersE) | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| -   [cu                           | -   [cudaq::sample (C++           |
| daq::kraus_channel::probabilities |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     member)](api/la               | mpleE13sample_resultRK14sample_op |
| nguages/cpp_api.html#_CPPv4N5cuda | tionsRR13QuantumKernelDpRR4Args), |
| q13kraus_channel13probabilitiesE) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|  [cudaq::kraus_channel::push_back | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     (C++                          | esultRR13QuantumKernelDpRR4Args), |
|     function)](api/langua         |     [\                            |
| ges/cpp_api.html#_CPPv4N5cudaq13k | [2\]](api/languages/cpp_api.html# |
| raus_channel9push_backE8kraus_op) | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| -   [cudaq::kraus_channel::size   | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::sample_options (C++   |
|     function)                     |     s                             |
| ](api/languages/cpp_api.html#_CPP | truct)](api/languages/cpp_api.htm |
| v4NK5cudaq13kraus_channel4sizeEv) | l#_CPPv4N5cudaq14sample_optionsE) |
| -   [                             | -   [cudaq::sample_result (C++    |
| cudaq::kraus_channel::unitary_ops |                                   |
|     (C++                          |  class)](api/languages/cpp_api.ht |
|     member)](api/                 | ml#_CPPv4N5cudaq13sample_resultE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::sample_result::append |
| daq13kraus_channel11unitary_opsE) |     (C++                          |
| -   [cudaq::kraus_op (C++         |     function)](api/languages/cpp_ |
|     struct)](api/languages/cpp_   | api.html#_CPPv4N5cudaq13sample_re |
| api.html#_CPPv4N5cudaq8kraus_opE) | sult6appendERK15ExecutionResultb) |
| -   [cudaq::kraus_op::adjoint     | -   [cudaq::sample_result::begin  |
|     (C++                          |     (C++                          |
|     functi                        |     function)]                    |
| on)](api/languages/cpp_api.html#_ | (api/languages/cpp_api.html#_CPPv |
| CPPv4NK5cudaq8kraus_op7adjointEv) | 4N5cudaq13sample_result5beginEv), |
| -   [cudaq::kraus_op::data (C++   |     [\[1\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|  member)](api/languages/cpp_api.h | 4NK5cudaq13sample_result5beginEv) |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | -   [cudaq::sample_result::cbegin |
| -   [cudaq::kraus_op::kraus_op    |     (C++                          |
|     (C++                          |     function)](                   |
|     func                          | api/languages/cpp_api.html#_CPPv4 |
| tion)](api/languages/cpp_api.html | NK5cudaq13sample_result6cbeginEv) |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | -   [cudaq::sample_result::cend   |
| opERRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)                     |
|  [\[1\]](api/languages/cpp_api.ht | ](api/languages/cpp_api.html#_CPP |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | v4NK5cudaq13sample_result4cendEv) |
| pENSt6vectorIN5cudaq7complexEEE), | -   [cudaq::sample_result::clear  |
|     [\[2\]](api/l                 |     (C++                          |
| anguages/cpp_api.html#_CPPv4N5cud |     function)                     |
| aq8kraus_op8kraus_opERK8kraus_op) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus_op::nCols (C++  | v4N5cudaq13sample_result5clearEv) |
|                                   | -   [cudaq::sample_result::count  |
| member)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     function)](                   |
| -   [cudaq::kraus_op::nRows (C++  | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq13sample_result5countENSt |
| member)](api/languages/cpp_api.ht | 11string_viewEKNSt11string_viewE) |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | -   [                             |
| -   [cudaq::kraus_op::operator=   | cudaq::sample_result::deserialize |
|     (C++                          |     (C++                          |
|     function)                     |     functio                       |
| ](api/languages/cpp_api.html#_CPP | n)](api/languages/cpp_api.html#_C |
| v4N5cudaq8kraus_opaSERK8kraus_op) | PPv4N5cudaq13sample_result11deser |
| -   [cudaq::kraus_op::precision   | ializeERNSt6vectorINSt6size_tEEE) |
|     (C++                          | -   [cudaq::sample_result::dump   |
|     memb                          |     (C++                          |
| er)](api/languages/cpp_api.html#_ |     function)](api/languag        |
| CPPv4N5cudaq8kraus_op9precisionE) | es/cpp_api.html#_CPPv4NK5cudaq13s |
| -   [cudaq::matrix_callback (C++  | ample_result4dumpERNSt7ostreamE), |
|     c                             |     [\[1\]                        |
| lass)](api/languages/cpp_api.html | ](api/languages/cpp_api.html#_CPP |
| #_CPPv4N5cudaq15matrix_callbackE) | v4NK5cudaq13sample_result4dumpEv) |
| -   [cudaq::matrix_handler (C++   | -   [cudaq::sample_result::end    |
|                                   |     (C++                          |
| class)](api/languages/cpp_api.htm |     function                      |
| l#_CPPv4N5cudaq14matrix_handlerE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::mat                   | Pv4N5cudaq13sample_result3endEv), |
| rix_handler::commutation_behavior |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     struct)](api/languages/       | Pv4NK5cudaq13sample_result3endEv) |
| cpp_api.html#_CPPv4N5cudaq14matri | -   [                             |
| x_handler20commutation_behaviorE) | cudaq::sample_result::expectation |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::define |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     function)](a                  | tml#_CPPv4NK5cudaq13sample_result |
| pi/languages/cpp_api.html#_CPPv4N | 11expectationEKNSt11string_viewE) |
| 5cudaq14matrix_handler6defineENSt | -   [c                            |
| 6stringENSt6vectorINSt7int64_tEEE | udaq::sample_result::get_marginal |
| RR15matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     function)](api/languages/cpp_ |
|                                   | api.html#_CPPv4NK5cudaq13sample_r |
| [\[1\]](api/languages/cpp_api.htm | esult12get_marginalERKNSt6vectorI |
| l#_CPPv4N5cudaq14matrix_handler6d | NSt6size_tEEEKNSt11string_viewE), |
| efineENSt6stringENSt6vectorINSt7i |     [\[1\]](api/languages/cpp_    |
| nt64_tEEERR15matrix_callbackRR20d | api.html#_CPPv4NK5cudaq13sample_r |
| iag_matrix_callbackRKNSt13unorder | esult12get_marginalERRKNSt6vector |
| ed_mapINSt6stringENSt6stringEEE), | INSt6size_tEEEKNSt11string_viewE) |
|     [\[2\]](                      | -   [cuda                         |
| api/languages/cpp_api.html#_CPPv4 | q::sample_result::get_total_shots |
| N5cudaq14matrix_handler6defineENS |     (C++                          |
| t6stringENSt6vectorINSt7int64_tEE |     function)](api/langua         |
| ERR15matrix_callbackRRNSt13unorde | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| red_mapINSt6stringENSt6stringEEE) | sample_result15get_total_shotsEv) |
| -                                 | -   [cuda                         |
|   [cudaq::matrix_handler::degrees | q::sample_result::has_even_parity |
|     (C++                          |     (C++                          |
|     function)](ap                 |     fun                           |
| i/languages/cpp_api.html#_CPPv4NK | ction)](api/languages/cpp_api.htm |
| 5cudaq14matrix_handler7degreesEv) | l#_CPPv4N5cudaq13sample_result15h |
| -                                 | as_even_parityENSt11string_viewE) |
|  [cudaq::matrix_handler::displace | -   [cuda                         |
|     (C++                          | q::sample_result::has_expectation |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     funct                         |
| rix_handler8displaceENSt6size_tE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::matrix                | _CPPv4NK5cudaq13sample_result15ha |
| _handler::get_expected_dimensions | s_expectationEKNSt11string_viewE) |
|     (C++                          | -   [cu                           |
|                                   | daq::sample_result::most_probable |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     fun                           |
| ndler23get_expected_dimensionsEv) | ction)](api/languages/cpp_api.htm |
| -   [cudaq::matrix_ha             | l#_CPPv4NK5cudaq13sample_result13 |
| ndler::get_parameter_descriptions | most_probableEKNSt11string_viewE) |
|     (C++                          | -                                 |
|                                   | [cudaq::sample_result::operator+= |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4NK5cudaq14matrix_handl |     function)](api/langua         |
| er26get_parameter_descriptionsEv) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -   [c                            | ample_resultpLERK13sample_result) |
| udaq::matrix_handler::instantiate | -                                 |
|     (C++                          |  [cudaq::sample_result::operator= |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/langua         |
| 5cudaq14matrix_handler11instantia | ges/cpp_api.html#_CPPv4N5cudaq13s |
| teENSt6stringERKNSt6vectorINSt6si | ample_resultaSERR13sample_result) |
| ze_tEEERK20commutation_behavior), | -                                 |
|     [\[1\]](                      | [cudaq::sample_result::operator== |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14matrix_handler11instanti |     function)](api/languag        |
| ateENSt6stringERRNSt6vectorINSt6s | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ize_tEEERK20commutation_behavior) | ample_resulteqERK13sample_result) |
| -   [cuda                         | -   [                             |
| q::matrix_handler::matrix_handler | cudaq::sample_result::probability |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)](api/lan            |
| es/cpp_api.html#_CPPv4I0_NSt11ena | guages/cpp_api.html#_CPPv4NK5cuda |
| ble_if_tINSt12is_base_of_vI16oper | q13sample_result11probabilityENSt |
| ator_handler1TEEbEEEN5cudaq14matr | 11string_viewEKNSt11string_viewE) |
| ix_handler14matrix_handlerERK1T), | -   [cud                          |
|     [\[1\]](ap                    | aq::sample_result::register_names |
| i/languages/cpp_api.html#_CPPv4I0 |     (C++                          |
| _NSt11enable_if_tINSt12is_base_of |     function)](api/langu          |
| _vI16operator_handler1TEEbEEEN5cu | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| daq14matrix_handler14matrix_handl | 3sample_result14register_namesEv) |
| erERK1TRK20commutation_behavior), | -                                 |
|     [\[2\]](api/languages/cpp_ap  |    [cudaq::sample_result::reorder |
| i.html#_CPPv4N5cudaq14matrix_hand |     (C++                          |
| ler14matrix_handlerENSt6size_tE), |     function)](api/langua         |
|     [\[3\]](api/                  | ges/cpp_api.html#_CPPv4N5cudaq13s |
| languages/cpp_api.html#_CPPv4N5cu | ample_result7reorderERKNSt6vector |
| daq14matrix_handler14matrix_handl | INSt6size_tEEEKNSt11string_viewE) |
| erENSt6stringERKNSt6vectorINSt6si | -   [cu                           |
| ze_tEEERK20commutation_behavior), | daq::sample_result::sample_result |
|     [\[4\]](api/                  |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     func                          |
| daq14matrix_handler14matrix_handl | tion)](api/languages/cpp_api.html |
| erENSt6stringERRNSt6vectorINSt6si | #_CPPv4N5cudaq13sample_result13sa |
| ze_tEEERK20commutation_behavior), | mple_resultERK15ExecutionResult), |
|     [\                            |     [\[1\]](api/la                |
| [5\]](api/languages/cpp_api.html# | nguages/cpp_api.html#_CPPv4N5cuda |
| _CPPv4N5cudaq14matrix_handler14ma | q13sample_result13sample_resultER |
| trix_handlerERK14matrix_handler), | KNSt6vectorI15ExecutionResultEE), |
|     [                             |                                   |
| \[6\]](api/languages/cpp_api.html |  [\[2\]](api/languages/cpp_api.ht |
| #_CPPv4N5cudaq14matrix_handler14m | ml#_CPPv4N5cudaq13sample_result13 |
| atrix_handlerERR14matrix_handler) | sample_resultERR13sample_result), |
| -                                 |     [                             |
|  [cudaq::matrix_handler::momentum | \[3\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq13sample_result13sa |
|     function)](api/language       | mple_resultERR15ExecutionResult), |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     [\[4\]](api/lan               |
| rix_handler8momentumENSt6size_tE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -                                 | 13sample_result13sample_resultEdR |
|    [cudaq::matrix_handler::number | KNSt6vectorI15ExecutionResultEE), |
|     (C++                          |     [\[5\]](api/lan               |
|     function)](api/langua         | guages/cpp_api.html#_CPPv4N5cudaq |
| ges/cpp_api.html#_CPPv4N5cudaq14m | 13sample_result13sample_resultEv) |
| atrix_handler6numberENSt6size_tE) | -                                 |
| -                                 |  [cudaq::sample_result::serialize |
| [cudaq::matrix_handler::operator= |     (C++                          |
|     (C++                          |     function)](api                |
|     fun                           | /languages/cpp_api.html#_CPPv4NK5 |
| ction)](api/languages/cpp_api.htm | cudaq13sample_result9serializeEv) |
| l#_CPPv4I0_NSt11enable_if_tIXaant | -   [cudaq::sample_result::size   |
| NSt7is_sameI1T14matrix_handlerE5v |     (C++                          |
| alueENSt12is_base_of_vI16operator |     function)](api/languages/c    |
| _handler1TEEEbEEEN5cudaq14matrix_ | pp_api.html#_CPPv4NK5cudaq13sampl |
| handleraSER14matrix_handlerRK1T), | e_result4sizeEKNSt11string_viewE) |
|     [\[1\]](api/languages         | -   [cudaq::sample_result::to_map |
| /cpp_api.html#_CPPv4N5cudaq14matr |     (C++                          |
| ix_handleraSERK14matrix_handler), |     function)](api/languages/cpp  |
|     [\[2\]](api/language          | _api.html#_CPPv4NK5cudaq13sample_ |
| s/cpp_api.html#_CPPv4N5cudaq14mat | result6to_mapEKNSt11string_viewE) |
| rix_handleraSERR14matrix_handler) | -   [cuda                         |
| -   [                             | q::sample_result::\~sample_result |
| cudaq::matrix_handler::operator== |     (C++                          |
|     (C++                          |     funct                         |
|     function)](api/languages      | ion)](api/languages/cpp_api.html# |
| /cpp_api.html#_CPPv4NK5cudaq14mat | _CPPv4N5cudaq13sample_resultD0Ev) |
| rix_handlereqERK14matrix_handler) | -   [cudaq::scalar_callback (C++  |
| -                                 |     c                             |
|    [cudaq::matrix_handler::parity | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_callbackE) |
|     function)](api/langua         | -   [c                            |
| ges/cpp_api.html#_CPPv4N5cudaq14m | udaq::scalar_callback::operator() |
| atrix_handler6parityENSt6size_tE) |     (C++                          |
| -                                 |     function)](api/language       |
|  [cudaq::matrix_handler::position | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|     (C++                          | alar_callbackclERKNSt13unordered_ |
|     function)](api/language       | mapINSt6stringENSt7complexIdEEEE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [                             |
| rix_handler8positionENSt6size_tE) | cudaq::scalar_callback::operator= |
| -   [cudaq::                      |     (C++                          |
| matrix_handler::remove_definition |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     fu                            | _callbackaSERK15scalar_callback), |
| nction)](api/languages/cpp_api.ht |     [\[1\]](api/languages/        |
| ml#_CPPv4N5cudaq14matrix_handler1 | cpp_api.html#_CPPv4N5cudaq15scala |
| 7remove_definitionERKNSt6stringE) | r_callbackaSERR15scalar_callback) |
| -                                 | -   [cudaq:                       |
|   [cudaq::matrix_handler::squeeze | :scalar_callback::scalar_callback |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)](api/languag        |
| es/cpp_api.html#_CPPv4N5cudaq14ma | es/cpp_api.html#_CPPv4I0_NSt11ena |
| trix_handler7squeezeENSt6size_tE) | ble_if_tINSt16is_invocable_r_vINS |
| -   [cudaq::m                     | t7complexIdEE8CallableRKNSt13unor |
| atrix_handler::to_diagonal_matrix | dered_mapINSt6stringENSt7complexI |
|     (C++                          | dEEEEEEbEEEN5cudaq15scalar_callba |
|     function)](api/lang           | ck15scalar_callbackERR8Callable), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[1\                         |
| 14matrix_handler18to_diagonal_mat | ]](api/languages/cpp_api.html#_CP |
| rixERNSt13unordered_mapINSt6size_ | Pv4N5cudaq15scalar_callback15scal |
| tENSt7int64_tEEERKNSt13unordered_ | ar_callbackERK15scalar_callback), |
| mapINSt6stringENSt7complexIdEEEE) |     [\[2                          |
| -                                 | \]](api/languages/cpp_api.html#_C |
| [cudaq::matrix_handler::to_matrix | PPv4N5cudaq15scalar_callback15sca |
|     (C++                          | lar_callbackERR15scalar_callback) |
|     function)                     | -   [cudaq::scalar_operator (C++  |
| ](api/languages/cpp_api.html#_CPP |     c                             |
| v4NK5cudaq14matrix_handler9to_mat | lass)](api/languages/cpp_api.html |
| rixERNSt13unordered_mapINSt6size_ | #_CPPv4N5cudaq15scalar_operatorE) |
| tENSt7int64_tEEERKNSt13unordered_ | -                                 |
| mapINSt6stringENSt7complexIdEEEE) | [cudaq::scalar_operator::evaluate |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::to_string |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     function)](api/               | pi.html#_CPPv4NK5cudaq15scalar_op |
| languages/cpp_api.html#_CPPv4NK5c | erator8evaluateERKNSt13unordered_ |
| udaq14matrix_handler9to_stringEb) | mapINSt6stringENSt7complexIdEEEE) |
| -                                 | -   [cudaq::scalar_ope            |
| [cudaq::matrix_handler::unique_id | rator::get_parameter_descriptions |
|     (C++                          |     (C++                          |
|     function)](api/               |     f                             |
| languages/cpp_api.html#_CPPv4NK5c | unction)](api/languages/cpp_api.h |
| udaq14matrix_handler9unique_idEv) | tml#_CPPv4NK5cudaq15scalar_operat |
| -   [cudaq:                       | or26get_parameter_descriptionsEv) |
| :matrix_handler::\~matrix_handler | -   [cu                           |
|     (C++                          | daq::scalar_operator::is_constant |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)](api/lang           |
| CPPv4N5cudaq14matrix_handlerD0Ev) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cudaq::matrix_op (C++        | 15scalar_operator11is_constantEv) |
|     type)](api/languages/cpp_a    | -   [c                            |
| pi.html#_CPPv4N5cudaq9matrix_opE) | udaq::scalar_operator::operator\* |
| -   [cudaq::matrix_op_term (C++   |     (C++                          |
|                                   |     function                      |
|  type)](api/languages/cpp_api.htm | )](api/languages/cpp_api.html#_CP |
| l#_CPPv4N5cudaq14matrix_op_termE) | Pv4N5cudaq15scalar_operatormlENSt |
| -                                 | 7complexIdEERK15scalar_operator), |
|    [cudaq::mdiag_operator_handler |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     class)](                      | Pv4N5cudaq15scalar_operatormlENSt |
| api/languages/cpp_api.html#_CPPv4 | 7complexIdEERR15scalar_operator), |
| N5cudaq22mdiag_operator_handlerE) |     [\[2\]](api/languages/cp      |
| -   [cudaq::mpi (C++              | p_api.html#_CPPv4N5cudaq15scalar_ |
|     type)](api/languages          | operatormlEdRK15scalar_operator), |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) |     [\[3\]](api/languages/cp      |
| -   [cudaq::mpi::all_gather (C++  | p_api.html#_CPPv4N5cudaq15scalar_ |
|     fu                            | operatormlEdRR15scalar_operator), |
| nction)](api/languages/cpp_api.ht |     [\[4\]](api/languages         |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| RNSt6vectorIdEERKNSt6vectorIdEE), | alar_operatormlENSt7complexIdEE), |
|                                   |     [\[5\]](api/languages/cpp     |
|   [\[1\]](api/languages/cpp_api.h | _api.html#_CPPv4NKR5cudaq15scalar |
| tml#_CPPv4N5cudaq3mpi10all_gather | _operatormlERK15scalar_operator), |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     [\[6\]]                       |
| -   [cudaq::mpi::all_reduce (C++  | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NKR5cudaq15scalar_operatormlEd), |
|  function)](api/languages/cpp_api |     [\[7\]](api/language          |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| reduceE1TRK1TRK14BinaryFunction), | alar_operatormlENSt7complexIdEE), |
|     [\[1\]](api/langu             |     [\[8\]](api/languages/cp      |
| ages/cpp_api.html#_CPPv4I00EN5cud | p_api.html#_CPPv4NO5cudaq15scalar |
| aq3mpi10all_reduceE1TRK1TRK4Func) | _operatormlERK15scalar_operator), |
| -   [cudaq::mpi::broadcast (C++   |     [\[9\                         |
|     function)](api/               | ]](api/languages/cpp_api.html#_CP |
| languages/cpp_api.html#_CPPv4N5cu | Pv4NO5cudaq15scalar_operatormlEd) |
| daq3mpi9broadcastERNSt6stringEi), | -   [cu                           |
|     [\[1\]](api/la                | daq::scalar_operator::operator\*= |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q3mpi9broadcastERNSt6vectorIdEEi) |     function)](api/languag        |
| -   [cudaq::mpi::finalize (C++    | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     f                             | alar_operatormLENSt7complexIdEE), |
| unction)](api/languages/cpp_api.h |     [\[1\]](api/languages/c       |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::mpi::initialize (C++  | _operatormLERK15scalar_operator), |
|     function                      |     [\[2                          |
| )](api/languages/cpp_api.html#_CP | \]](api/languages/cpp_api.html#_C |
| Pv4N5cudaq3mpi10initializeEiPPc), | PPv4N5cudaq15scalar_operatormLEd) |
|     [                             | -   [                             |
| \[1\]](api/languages/cpp_api.html | cudaq::scalar_operator::operator+ |
| #_CPPv4N5cudaq3mpi10initializeEv) |     (C++                          |
| -   [cudaq::mpi::is_initialized   |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function                      | Pv4N5cudaq15scalar_operatorplENSt |
| )](api/languages/cpp_api.html#_CP | 7complexIdEERK15scalar_operator), |
| Pv4N5cudaq3mpi14is_initializedEv) |     [\[1\                         |
| -   [cudaq::mpi::num_ranks (C++   | ]](api/languages/cpp_api.html#_CP |
|     fu                            | Pv4N5cudaq15scalar_operatorplENSt |
| nction)](api/languages/cpp_api.ht | 7complexIdEERR15scalar_operator), |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     [\[2\]](api/languages/cp      |
| -   [cudaq::mpi::rank (C++        | p_api.html#_CPPv4N5cudaq15scalar_ |
|                                   | operatorplEdRK15scalar_operator), |
|    function)](api/languages/cpp_a |     [\[3\]](api/languages/cp      |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::noise_model (C++      | operatorplEdRR15scalar_operator), |
|                                   |     [\[4\]](api/languages         |
|    class)](api/languages/cpp_api. | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| html#_CPPv4N5cudaq11noise_modelE) | alar_operatorplENSt7complexIdEE), |
| -   [cudaq::n                     |     [\[5\]](api/languages/cpp     |
| oise_model::add_all_qubit_channel | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatorplERK15scalar_operator), |
|     function)](api                |     [\[6\]]                       |
| /languages/cpp_api.html#_CPPv4IDp | (api/languages/cpp_api.html#_CPPv |
| EN5cudaq11noise_model21add_all_qu | 4NKR5cudaq15scalar_operatorplEd), |
| bit_channelEvRK13kraus_channeli), |     [\[7\]]                       |
|     [\[1\]](api/langua            | (api/languages/cpp_api.html#_CPPv |
| ges/cpp_api.html#_CPPv4N5cudaq11n | 4NKR5cudaq15scalar_operatorplEv), |
| oise_model21add_all_qubit_channel |     [\[8\]](api/language          |
| ERKNSt6stringERK13kraus_channeli) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -                                 | alar_operatorplENSt7complexIdEE), |
|  [cudaq::noise_model::add_channel |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     funct                         | _operatorplERK15scalar_operator), |
| ion)](api/languages/cpp_api.html# |     [\[10\]                       |
| _CPPv4IDpEN5cudaq11noise_model11a | ](api/languages/cpp_api.html#_CPP |
| dd_channelEvRK15PredicateFuncTy), | v4NO5cudaq15scalar_operatorplEd), |
|     [\[1\]](api/languages/cpp_    |     [\[11\                        |
| api.html#_CPPv4IDpEN5cudaq11noise | ]](api/languages/cpp_api.html#_CP |
| _model11add_channelEvRKNSt6vector | Pv4NO5cudaq15scalar_operatorplEv) |
| INSt6size_tEEERK13kraus_channel), | -   [c                            |
|     [\[2\]](ap                    | udaq::scalar_operator::operator+= |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq11noise_model11add_channelER |     function)](api/languag        |
| KNSt6stringERK15PredicateFuncTy), | es/cpp_api.html#_CPPv4N5cudaq15sc |
|                                   | alar_operatorpLENSt7complexIdEE), |
| [\[3\]](api/languages/cpp_api.htm |     [\[1\]](api/languages/c       |
| l#_CPPv4N5cudaq11noise_model11add | pp_api.html#_CPPv4N5cudaq15scalar |
| _channelERKNSt6stringERKNSt6vecto | _operatorpLERK15scalar_operator), |
| rINSt6size_tEEERK13kraus_channel) |     [\[2                          |
| -   [cudaq::noise_model::empty    | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatorpLEd) |
|     function                      | -   [                             |
| )](api/languages/cpp_api.html#_CP | cudaq::scalar_operator::operator- |
| Pv4NK5cudaq11noise_model5emptyEv) |     (C++                          |
| -                                 |     function                      |
| [cudaq::noise_model::get_channels | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormiENSt |
|     function)](api/l              | 7complexIdEERK15scalar_operator), |
| anguages/cpp_api.html#_CPPv4I0ENK |     [\[1\                         |
| 5cudaq11noise_model12get_channels | ]](api/languages/cpp_api.html#_CP |
| ENSt6vectorI13kraus_channelEERKNS | Pv4N5cudaq15scalar_operatormiENSt |
| t6vectorINSt6size_tEEERKNSt6vecto | 7complexIdEERR15scalar_operator), |
| rINSt6size_tEEERKNSt6vectorIdEE), |     [\[2\]](api/languages/cp      |
|     [\[1\]](api/languages/cpp_a   | p_api.html#_CPPv4N5cudaq15scalar_ |
| pi.html#_CPPv4NK5cudaq11noise_mod | operatormiEdRK15scalar_operator), |
| el12get_channelsERKNSt6stringERKN |     [\[3\]](api/languages/cp      |
| St6vectorINSt6size_tEEERKNSt6vect | p_api.html#_CPPv4N5cudaq15scalar_ |
| orINSt6size_tEEERKNSt6vectorIdEE) | operatormiEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
|  [cudaq::noise_model::noise_model | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     function)](api                |     [\[5\]](api/languages/cpp     |
| /languages/cpp_api.html#_CPPv4N5c | _api.html#_CPPv4NKR5cudaq15scalar |
| udaq11noise_model11noise_modelEv) | _operatormiERK15scalar_operator), |
| -   [cu                           |     [\[6\]]                       |
| daq::noise_model::PredicateFuncTy | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEd), |
|     type)](api/la                 |     [\[7\]]                       |
| nguages/cpp_api.html#_CPPv4N5cuda | (api/languages/cpp_api.html#_CPPv |
| q11noise_model15PredicateFuncTyE) | 4NKR5cudaq15scalar_operatormiEv), |
| -   [cud                          |     [\[8\]](api/language          |
| aq::noise_model::register_channel | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     function)](api/languages      |     [\[9\]](api/languages/cp      |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | p_api.html#_CPPv4NO5cudaq15scalar |
| noise_model16register_channelEvv) | _operatormiERK15scalar_operator), |
| -   [cudaq::                      |     [\[10\]                       |
| noise_model::requires_constructor | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatormiEd), |
|     type)](api/languages/cp       |     [\[11\                        |
| p_api.html#_CPPv4I0DpEN5cudaq11no | ]](api/languages/cpp_api.html#_CP |
| ise_model20requires_constructorE) | Pv4NO5cudaq15scalar_operatormiEv) |
| -   [cudaq::noise_model_type (C++ | -   [c                            |
|     e                             | udaq::scalar_operator::operator-= |
| num)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16noise_model_typeE) |     function)](api/languag        |
| -   [cudaq::no                    | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ise_model_type::amplitude_damping | alar_operatormIENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     enumerator)](api/languages    | pp_api.html#_CPPv4N5cudaq15scalar |
| /cpp_api.html#_CPPv4N5cudaq16nois | _operatormIERK15scalar_operator), |
| e_model_type17amplitude_dampingE) |     [\[2                          |
| -   [cudaq::noise_mode            | \]](api/languages/cpp_api.html#_C |
| l_type::amplitude_damping_channel | PPv4N5cudaq15scalar_operatormIEd) |
|     (C++                          | -   [                             |
|     e                             | cudaq::scalar_operator::operator/ |
| numerator)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4N5cudaq16noise_model_ |     function                      |
| type25amplitude_damping_channelE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::n                     | Pv4N5cudaq15scalar_operatordvENSt |
| oise_model_type::bit_flip_channel | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](api/language     | ]](api/languages/cpp_api.html#_CP |
| s/cpp_api.html#_CPPv4N5cudaq16noi | Pv4N5cudaq15scalar_operatordvENSt |
| se_model_type16bit_flip_channelE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::                      |     [\[2\]](api/languages/cp      |
| noise_model_type::depolarization1 | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRK15scalar_operator), |
|     enumerator)](api/languag      |     [\[3\]](api/languages/cp      |
| es/cpp_api.html#_CPPv4N5cudaq16no | p_api.html#_CPPv4N5cudaq15scalar_ |
| ise_model_type15depolarization1E) | operatordvEdRR15scalar_operator), |
| -   [cudaq::                      |     [\[4\]](api/languages         |
| noise_model_type::depolarization2 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatordvENSt7complexIdEE), |
|     enumerator)](api/languag      |     [\[5\]](api/languages/cpp     |
| es/cpp_api.html#_CPPv4N5cudaq16no | _api.html#_CPPv4NKR5cudaq15scalar |
| ise_model_type15depolarization2E) | _operatordvERK15scalar_operator), |
| -   [cudaq::noise_m               |     [\[6\]]                       |
| odel_type::depolarization_channel | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatordvEd), |
|                                   |     [\[7\]](api/language          |
|   enumerator)](api/languages/cpp_ | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| api.html#_CPPv4N5cudaq16noise_mod | alar_operatordvENSt7complexIdEE), |
| el_type22depolarization_channelE) |     [\[8\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4NO5cudaq15scalar |
|  [cudaq::noise_model_type::pauli1 | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     enumerator)](a                | ]](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4NO5cudaq15scalar_operatordvEd) |
| 5cudaq16noise_model_type6pauli1E) | -   [c                            |
| -                                 | udaq::scalar_operator::operator/= |
|  [cudaq::noise_model_type::pauli2 |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](a                | es/cpp_api.html#_CPPv4N5cudaq15sc |
| pi/languages/cpp_api.html#_CPPv4N | alar_operatordVENSt7complexIdEE), |
| 5cudaq16noise_model_type6pauli2E) |     [\[1\]](api/languages/c       |
| -   [cudaq                        | pp_api.html#_CPPv4N5cudaq15scalar |
| ::noise_model_type::phase_damping | _operatordVERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     enumerator)](api/langu        | \]](api/languages/cpp_api.html#_C |
| ages/cpp_api.html#_CPPv4N5cudaq16 | PPv4N5cudaq15scalar_operatordVEd) |
| noise_model_type13phase_dampingE) | -   [                             |
| -   [cudaq::noi                   | cudaq::scalar_operator::operator= |
| se_model_type::phase_flip_channel |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     enumerator)](api/languages/   | pp_api.html#_CPPv4N5cudaq15scalar |
| cpp_api.html#_CPPv4N5cudaq16noise | _operatoraSERK15scalar_operator), |
| _model_type18phase_flip_channelE) |     [\[1\]](api/languages/        |
| -                                 | cpp_api.html#_CPPv4N5cudaq15scala |
| [cudaq::noise_model_type::unknown | r_operatoraSERR15scalar_operator) |
|     (C++                          | -   [c                            |
|     enumerator)](ap               | udaq::scalar_operator::operator== |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7unknownE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq15scala |
| [cudaq::noise_model_type::x_error | r_operatoreqERK15scalar_operator) |
|     (C++                          | -   [cudaq:                       |
|     enumerator)](ap               | :scalar_operator::scalar_operator |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7x_errorE) |     func                          |
| -                                 | tion)](api/languages/cpp_api.html |
| [cudaq::noise_model_type::y_error | #_CPPv4N5cudaq15scalar_operator15 |
|     (C++                          | scalar_operatorENSt7complexIdEE), |
|     enumerator)](ap               |     [\[1\]](api/langu             |
| i/languages/cpp_api.html#_CPPv4N5 | ages/cpp_api.html#_CPPv4N5cudaq15 |
| cudaq16noise_model_type7y_errorE) | scalar_operator15scalar_operatorE |
| -                                 | RK15scalar_callbackRRNSt13unorder |
| [cudaq::noise_model_type::z_error | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[2\                         |
|     enumerator)](ap               | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4N5cudaq15scalar_operator15scal |
| cudaq16noise_model_type7z_errorE) | ar_operatorERK15scalar_operator), |
| -   [cudaq::num_available_gpus    |     [\[3\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     function                      | scalar_operator15scalar_operatorE |
| )](api/languages/cpp_api.html#_CP | RR15scalar_callbackRRNSt13unorder |
| Pv4N5cudaq18num_available_gpusEv) | ed_mapINSt6stringENSt6stringEEE), |
| -   [cudaq::observe (C++          |     [\[4\                         |
|     function)]                    | ]](api/languages/cpp_api.html#_CP |
| (api/languages/cpp_api.html#_CPPv | Pv4N5cudaq15scalar_operator15scal |
| 4I00DpEN5cudaq7observeENSt6vector | ar_operatorERR15scalar_operator), |
| I14observe_resultEERR13QuantumKer |     [\[5\]](api/language          |
| nelRK15SpinOpContainerDpRR4Args), | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     [\[1\]](api/languages/cpp_ap  | lar_operator15scalar_operatorEd), |
| i.html#_CPPv4I0DpEN5cudaq7observe |     [\[6\]](api/languag           |
| E14observe_resultNSt6size_tERR13Q | es/cpp_api.html#_CPPv4N5cudaq15sc |
| uantumKernelRK7spin_opDpRR4Args), | alar_operator15scalar_operatorEv) |
|     [\[                           | -   [                             |
| 2\]](api/languages/cpp_api.html#_ | cudaq::scalar_operator::to_matrix |
| CPPv4I0DpEN5cudaq7observeE14obser |     (C++                          |
| ve_resultRK15observe_optionsRR13Q |                                   |
| uantumKernelRK7spin_opDpRR4Args), |   function)](api/languages/cpp_ap |
|     [\[3\]](api/lang              | i.html#_CPPv4NK5cudaq15scalar_ope |
| uages/cpp_api.html#_CPPv4I0DpEN5c | rator9to_matrixERKNSt13unordered_ |
| udaq7observeE14observe_resultRR13 | mapINSt6stringENSt7complexIdEEEE) |
| QuantumKernelRK7spin_opDpRR4Args) | -   [                             |
| -   [cudaq::observe_options (C++  | cudaq::scalar_operator::to_string |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |     function)](api/l              |
| #_CPPv4N5cudaq15observe_optionsE) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cudaq::observe_result (C++   | daq15scalar_operator9to_stringEv) |
|                                   | -   [cudaq::s                     |
| class)](api/languages/cpp_api.htm | calar_operator::\~scalar_operator |
| l#_CPPv4N5cudaq14observe_resultE) |     (C++                          |
| -                                 |     functio                       |
|    [cudaq::observe_result::counts | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatorD0Ev) |
|     function)](api/languages/c    | -   [cudaq::set_noise (C++        |
| pp_api.html#_CPPv4N5cudaq14observ |     function)](api/langu          |
| e_result6countsERK12spin_op_term) | ages/cpp_api.html#_CPPv4N5cudaq9s |
| -   [cudaq::observe_result::dump  | et_noiseERKN5cudaq11noise_modelE) |
|     (C++                          | -   [cudaq::set_random_seed (C++  |
|     function)                     |     function)](api/               |
| ](api/languages/cpp_api.html#_CPP | languages/cpp_api.html#_CPPv4N5cu |
| v4N5cudaq14observe_result4dumpEv) | daq15set_random_seedENSt6size_tE) |
| -   [c                            | -   [cudaq::simulation_precision  |
| udaq::observe_result::expectation |     (C++                          |
|     (C++                          |     enum)                         |
|                                   | ](api/languages/cpp_api.html#_CPP |
| function)](api/languages/cpp_api. | v4N5cudaq20simulation_precisionE) |
| html#_CPPv4N5cudaq14observe_resul | -   [                             |
| t11expectationERK12spin_op_term), | cudaq::simulation_precision::fp32 |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     enumerator)](api              |
| q14observe_result11expectationEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cuda                         | udaq20simulation_precision4fp32E) |
| q::observe_result::id_coefficient | -   [                             |
|     (C++                          | cudaq::simulation_precision::fp64 |
|     function)](api/langu          |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     enumerator)](api              |
| observe_result14id_coefficientEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cuda                         | udaq20simulation_precision4fp64E) |
| q::observe_result::observe_result | -   [cudaq::SimulationState (C++  |
|     (C++                          |     c                             |
|                                   | lass)](api/languages/cpp_api.html |
|   function)](api/languages/cpp_ap | #_CPPv4N5cudaq15SimulationStateE) |
| i.html#_CPPv4N5cudaq14observe_res | -   [                             |
| ult14observe_resultEdRK7spin_op), | cudaq::SimulationState::precision |
|     [\[1\]](a                     |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     enum)](api                    |
| 5cudaq14observe_result14observe_r | /languages/cpp_api.html#_CPPv4N5c |
| esultEdRK7spin_op13sample_result) | udaq15SimulationState9precisionE) |
| -                                 | -   [cudaq:                       |
|  [cudaq::observe_result::operator | :SimulationState::precision::fp32 |
|     double (C++                   |     (C++                          |
|     functio                       |     enumerator)](api/lang         |
| n)](api/languages/cpp_api.html#_C | uages/cpp_api.html#_CPPv4N5cudaq1 |
| PPv4N5cudaq14observe_resultcvdEv) | 5SimulationState9precision4fp32E) |
| -                                 | -   [cudaq:                       |
|  [cudaq::observe_result::raw_data | :SimulationState::precision::fp64 |
|     (C++                          |     (C++                          |
|     function)](ap                 |     enumerator)](api/lang         |
| i/languages/cpp_api.html#_CPPv4N5 | uages/cpp_api.html#_CPPv4N5cudaq1 |
| cudaq14observe_result8raw_dataEv) | 5SimulationState9precision4fp64E) |
| -   [cudaq::operator_handler (C++ | -                                 |
|     cl                            |   [cudaq::SimulationState::Tensor |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16operator_handlerE) |     struct)](                     |
| -   [cudaq::optimizable_function  | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq15SimulationState6TensorE) |
|     class)                        | -   [cudaq::spin_handler (C++     |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4N5cudaq20optimizable_functionE) |   class)](api/languages/cpp_api.h |
| -   [cudaq::optimization_result   | tml#_CPPv4N5cudaq12spin_handlerE) |
|     (C++                          | -   [cudaq:                       |
|     type                          | :spin_handler::to_diagonal_matrix |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq19optimization_resultE) |     function)](api/la             |
| -   [cudaq::optimizer (C++        | nguages/cpp_api.html#_CPPv4NK5cud |
|     class)](api/languages/cpp_a   | aq12spin_handler18to_diagonal_mat |
| pi.html#_CPPv4N5cudaq9optimizerE) | rixERNSt13unordered_mapINSt6size_ |
| -   [cudaq::optimizer::optimize   | tENSt7int64_tEEERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -                                 |
|  function)](api/languages/cpp_api |   [cudaq::spin_handler::to_matrix |
| .html#_CPPv4N5cudaq9optimizer8opt |     (C++                          |
| imizeEKiRR20optimizable_function) |     function                      |
| -   [cu                           | )](api/languages/cpp_api.html#_CP |
| daq::optimizer::requiresGradients | Pv4N5cudaq12spin_handler9to_matri |
|     (C++                          | xERKNSt6stringENSt7complexIdEEb), |
|     function)](api/la             |     [\[1                          |
| nguages/cpp_api.html#_CPPv4N5cuda | \]](api/languages/cpp_api.html#_C |
| q9optimizer17requiresGradientsEv) | PPv4NK5cudaq12spin_handler9to_mat |
| -   [cudaq::orca (C++             | rixERNSt13unordered_mapINSt6size_ |
|     type)](api/languages/         | tENSt7int64_tEEERKNSt13unordered_ |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::orca::sample (C++     | -   [cuda                         |
|     function)](api/languages/c    | q::spin_handler::to_sparse_matrix |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     (C++                          |
| mpleERNSt6vectorINSt6size_tEEERNS |     function)](api/               |
| t6vectorINSt6size_tEEERNSt6vector | languages/cpp_api.html#_CPPv4N5cu |
| IdEERNSt6vectorIdEEiNSt6size_tE), | daq12spin_handler16to_sparse_matr |
|     [\[1\]]                       | ixERKNSt6stringENSt7complexIdEEb) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq4orca6sampleERNSt6vectorI |   [cudaq::spin_handler::to_string |
| NSt6size_tEEERNSt6vectorINSt6size |     (C++                          |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     function)](ap                 |
| -   [cudaq::orca::sample_async    | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq12spin_handler9to_stringEb) |
|                                   | -                                 |
| function)](api/languages/cpp_api. |   [cudaq::spin_handler::unique_id |
| html#_CPPv4N5cudaq4orca12sample_a |     (C++                          |
| syncERNSt6vectorINSt6size_tEEERNS |     function)](ap                 |
| t6vectorINSt6size_tEEERNSt6vector | i/languages/cpp_api.html#_CPPv4NK |
| IdEERNSt6vectorIdEEiNSt6size_tE), | 5cudaq12spin_handler9unique_idEv) |
|     [\[1\]](api/la                | -   [cudaq::spin_op (C++          |
| nguages/cpp_api.html#_CPPv4N5cuda |     type)](api/languages/cpp      |
| q4orca12sample_asyncERNSt6vectorI | _api.html#_CPPv4N5cudaq7spin_opE) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [cudaq::spin_op_term (C++     |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |                                   |
| -   [cudaq::OrcaRemoteRESTQPU     |    type)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq12spin_op_termE) |
|     cla                           | -   [cudaq::state (C++            |
| ss)](api/languages/cpp_api.html#_ |     class)](api/languages/c       |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | pp_api.html#_CPPv4N5cudaq5stateE) |
| -   [cudaq::pauli1 (C++           | -   [cudaq::state::amplitude (C++ |
|     class)](api/languages/cp      |     function)](api/lang           |
| p_api.html#_CPPv4N5cudaq6pauli1E) | uages/cpp_api.html#_CPPv4N5cudaq5 |
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
| -   [get()                        | -   [get_register_counts()        |
|     (cudaq.AsyncEvolveResult      |     (cudaq.SampleResult           |
|     m                             |     method)](api                  |
| ethod)](api/languages/python_api. | /languages/python_api.html#cudaq. |
| html#cudaq.AsyncEvolveResult.get) | SampleResult.get_register_counts) |
|                                   | -   [get_sequential_data()        |
|    -   [(cudaq.AsyncObserveResult |     (cudaq.SampleResult           |
|         me                        |     method)](api                  |
| thod)](api/languages/python_api.h | /languages/python_api.html#cudaq. |
| tml#cudaq.AsyncObserveResult.get) | SampleResult.get_sequential_data) |
|     -   [(cudaq.AsyncStateResult  | -   [get_spin()                   |
|                                   |     (cudaq.ObserveResult          |
| method)](api/languages/python_api |     me                            |
| .html#cudaq.AsyncStateResult.get) | thod)](api/languages/python_api.h |
| -   [get_binary_symplectic_form() | tml#cudaq.ObserveResult.get_spin) |
|     (cuda                         | -   [get_state() (in module       |
| q.operators.spin.SpinOperatorTerm |     cudaq)](api/languages         |
|     metho                         | /python_api.html#cudaq.get_state) |
| d)](api/languages/python_api.html | -   [get_state_async() (in module |
| #cudaq.operators.spin.SpinOperato |     cudaq)](api/languages/pytho   |
| rTerm.get_binary_symplectic_form) | n_api.html#cudaq.get_state_async) |
| -   [get_channels()               | -   [get_state_refval()           |
|     (cudaq.NoiseModel             |     (cudaq.State                  |
|     met                           |     me                            |
| hod)](api/languages/python_api.ht | thod)](api/languages/python_api.h |
| ml#cudaq.NoiseModel.get_channels) | tml#cudaq.State.get_state_refval) |
| -   [get_coefficient()            | -   [get_target() (in module      |
|     (                             |     cudaq)](api/languages/        |
| cudaq.operators.spin.SpinOperator | python_api.html#cudaq.get_target) |
|     method)](api/languages/       | -   [get_targets() (in module     |
| python_api.html#cudaq.operators.s |     cudaq)](api/languages/p       |
| pin.SpinOperator.get_coefficient) | ython_api.html#cudaq.get_targets) |
|     -   [(cuda                    | -   [get_term_count()             |
| q.operators.spin.SpinOperatorTerm |     (                             |
|                                   | cudaq.operators.spin.SpinOperator |
|       method)](api/languages/pyth |     method)](api/languages        |
| on_api.html#cudaq.operators.spin. | /python_api.html#cudaq.operators. |
| SpinOperatorTerm.get_coefficient) | spin.SpinOperator.get_term_count) |
| -   [get_marginal_counts()        | -   [get_total_shots()            |
|     (cudaq.SampleResult           |     (cudaq.SampleResult           |
|     method)](api                  |     method)]                      |
| /languages/python_api.html#cudaq. | (api/languages/python_api.html#cu |
| SampleResult.get_marginal_counts) | daq.SampleResult.get_total_shots) |
| -   [get_ops()                    | -   [getTensor() (cudaq.State     |
|     (cudaq.KrausChannel           |     method)](api/languages/pytho  |
|                                   | n_api.html#cudaq.State.getTensor) |
| method)](api/languages/python_api | -   [getTensors() (cudaq.State    |
| .html#cudaq.KrausChannel.get_ops) |     method)](api/languages/python |
| -   [get_pauli_word()             | _api.html#cudaq.State.getTensors) |
|     (cuda                         | -   [gradient (class in           |
| q.operators.spin.SpinOperatorTerm |     cudaq.g                       |
|     method)](api/languages/pyt    | radients)](api/languages/python_a |
| hon_api.html#cudaq.operators.spin | pi.html#cudaq.gradients.gradient) |
| .SpinOperatorTerm.get_pauli_word) | -   [GradientDescent (class in    |
| -   [get_precision()              |     cudaq.optimizers              |
|     (cudaq.Target                 | )](api/languages/python_api.html# |
|                                   | cudaq.optimizers.GradientDescent) |
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
| -   [launch_args_required()       | -   [lower_bounds                 |
|     (cudaq.PyKernelDecorator      |     (cudaq.optimizers.COBYLA      |
|     method)](api/langu            |     property)](a                  |
| ages/python_api.html#cudaq.PyKern | pi/languages/python_api.html#cuda |
| elDecorator.launch_args_required) | q.optimizers.COBYLA.lower_bounds) |
| -   [LBFGS (class in              |     -   [                         |
|     cudaq.                        | (cudaq.optimizers.GradientDescent |
| optimizers)](api/languages/python |         property)](api/langua     |
| _api.html#cudaq.optimizers.LBFGS) | ges/python_api.html#cudaq.optimiz |
| -   [left_multiply()              | ers.GradientDescent.lower_bounds) |
|     (cudaq.SuperOperator static   |     -   [(cudaq.optimizers.LBFGS  |
|     method)                       |         property)](               |
| ](api/languages/python_api.html#c | api/languages/python_api.html#cud |
| udaq.SuperOperator.left_multiply) | aq.optimizers.LBFGS.lower_bounds) |
| -   [left_right_multiply()        |                                   |
|     (cudaq.SuperOperator static   | -   [(cudaq.optimizers.NelderMead |
|     method)](api/                 |         property)](api/l          |
| languages/python_api.html#cudaq.S | anguages/python_api.html#cudaq.op |
| uperOperator.left_right_multiply) | timizers.NelderMead.lower_bounds) |
|                                   | -   [lower_quake_to_codegen()     |
|                                   |     (cudaq.PyKernelDecorator      |
|                                   |     method)](api/languag          |
|                                   | es/python_api.html#cudaq.PyKernel |
|                                   | Decorator.lower_quake_to_codegen) |
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
| -   [merge_quake_source()         |                                   |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](api/lan              |                                   |
| guages/python_api.html#cudaq.PyKe |                                   |
| rnelDecorator.merge_quake_source) |                                   |
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
|     -   [(in module               | -   [signatureWithCallables()     |
|                                   |     (cudaq.PyKernelDecorator      |
|      cudaq.orca)](api/languages/p |     method)](api/languag          |
| ython_api.html#cudaq.orca.sample) | es/python_api.html#cudaq.PyKernel |
| -   [sample_async() (in module    | Decorator.signatureWithCallables) |
|     cudaq)](api/languages/py      | -   [SimulationPrecision (class   |
| thon_api.html#cudaq.sample_async) |     in                            |
| -   [SampleResult (class in       |                                   |
|     cudaq)](api/languages/py      |   cudaq)](api/languages/python_ap |
| thon_api.html#cudaq.SampleResult) | i.html#cudaq.SimulationPrecision) |
| -   [ScalarOperator (class in     | -   [simulator (cudaq.Target      |
|     cudaq.operato                 |                                   |
| rs)](api/languages/python_api.htm |   property)](api/languages/python |
| l#cudaq.operators.ScalarOperator) | _api.html#cudaq.Target.simulator) |
| -   [Schedule (class in           | -   [slice() (cudaq.QuakeValue    |
|     cudaq)](api/language          |     method)](api/languages/python |
| s/python_api.html#cudaq.Schedule) | _api.html#cudaq.QuakeValue.slice) |
| -   [serialize()                  | -   [SpinOperator (class in       |
|     (                             |     cudaq.operators.spin)         |
| cudaq.operators.spin.SpinOperator | ](api/languages/python_api.html#c |
|     method)](api/lang             | udaq.operators.spin.SpinOperator) |
| uages/python_api.html#cudaq.opera | -   [SpinOperatorElement (class   |
| tors.spin.SpinOperator.serialize) |     in                            |
|     -   [(cuda                    |     cudaq.operators.spin)](api/l  |
| q.operators.spin.SpinOperatorTerm | anguages/python_api.html#cudaq.op |
|         method)](api/language     | erators.spin.SpinOperatorElement) |
| s/python_api.html#cudaq.operators | -   [SpinOperatorTerm (class in   |
| .spin.SpinOperatorTerm.serialize) |     cudaq.operators.spin)](ap     |
|     -   [(cudaq.SampleResult      | i/languages/python_api.html#cudaq |
|         me                        | .operators.spin.SpinOperatorTerm) |
| thod)](api/languages/python_api.h | -   [squeeze() (in module         |
| tml#cudaq.SampleResult.serialize) |     cudaq.operators.cust          |
| -   [set_noise() (in module       | om)](api/languages/python_api.htm |
|     cudaq)](api/languages         | l#cudaq.operators.custom.squeeze) |
| /python_api.html#cudaq.set_noise) | -   [State (class in              |
| -   [set_random_seed() (in module |     cudaq)](api/langu             |
|     cudaq)](api/languages/pytho   | ages/python_api.html#cudaq.State) |
| n_api.html#cudaq.set_random_seed) | -   [SuperOperator (class in      |
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
