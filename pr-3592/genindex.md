::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-3592
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
        -   [Setup and
            Imports](applications/python/skqd.html#Setup-and-Imports){.reference
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
        -   [NVIDIA Quantum Cloud
            (nvqc)](using/backends/cloud/nvqc.html){.reference
            .internal}
            -   [Quick
                Start](using/backends/cloud/nvqc.html#quick-start){.reference
                .internal}
            -   [Simulator Backend
                Selection](using/backends/cloud/nvqc.html#simulator-backend-selection){.reference
                .internal}
            -   [Multiple
                GPUs](using/backends/cloud/nvqc.html#multiple-gpus){.reference
                .internal}
            -   [Multiple QPUs Asynchronous
                Execution](using/backends/cloud/nvqc.html#multiple-qpus-asynchronous-execution){.reference
                .internal}
            -   [FAQ](using/backends/cloud/nvqc.html#faq){.reference
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
| -   [canonicalize()               | -   [cudaq::pauli1 (C++           |
|     (cu                           |     class)](api/languages/cp      |
| daq.operators.boson.BosonOperator | p_api.html#_CPPv4N5cudaq6pauli1E) |
|     method)](api/languages        | -                                 |
| /python_api.html#cudaq.operators. |    [cudaq::pauli1::num_parameters |
| boson.BosonOperator.canonicalize) |     (C++                          |
|     -   [(cudaq.                  |     member)]                      |
| operators.boson.BosonOperatorTerm | (api/languages/cpp_api.html#_CPPv |
|                                   | 4N5cudaq6pauli114num_parametersE) |
|        method)](api/languages/pyt | -   [cudaq::pauli1::num_targets   |
| hon_api.html#cudaq.operators.boso |     (C++                          |
| n.BosonOperatorTerm.canonicalize) |     membe                         |
|     -   [(cudaq.                  | r)](api/languages/cpp_api.html#_C |
| operators.fermion.FermionOperator | PPv4N5cudaq6pauli111num_targetsE) |
|                                   | -   [cudaq::pauli1::pauli1 (C++   |
|        method)](api/languages/pyt |     function)](api/languages/cpp_ |
| hon_api.html#cudaq.operators.ferm | api.html#_CPPv4N5cudaq6pauli16pau |
| ion.FermionOperator.canonicalize) | li1ERKNSt6vectorIN5cudaq4realEEE) |
|     -   [(cudaq.oper              | -   [cudaq::pauli2 (C++           |
| ators.fermion.FermionOperatorTerm |     class)](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq6pauli2E) |
|    method)](api/languages/python_ | -                                 |
| api.html#cudaq.operators.fermion. |    [cudaq::pauli2::num_parameters |
| FermionOperatorTerm.canonicalize) |     (C++                          |
|     -                             |     member)]                      |
|  [(cudaq.operators.MatrixOperator | (api/languages/cpp_api.html#_CPPv |
|         method)](api/lang         | 4N5cudaq6pauli214num_parametersE) |
| uages/python_api.html#cudaq.opera | -   [cudaq::pauli2::num_targets   |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     membe                         |
| udaq.operators.MatrixOperatorTerm | r)](api/languages/cpp_api.html#_C |
|         method)](api/language     | PPv4N5cudaq6pauli211num_targetsE) |
| s/python_api.html#cudaq.operators | -   [cudaq::pauli2::pauli2 (C++   |
| .MatrixOperatorTerm.canonicalize) |     function)](api/languages/cpp_ |
|     -   [(                        | api.html#_CPPv4N5cudaq6pauli26pau |
| cudaq.operators.spin.SpinOperator | li2ERKNSt6vectorIN5cudaq4realEEE) |
|         method)](api/languag      | -   [cudaq::phase_damping (C++    |
| es/python_api.html#cudaq.operator |                                   |
| s.spin.SpinOperator.canonicalize) |  class)](api/languages/cpp_api.ht |
|     -   [(cuda                    | ml#_CPPv4N5cudaq13phase_dampingE) |
| q.operators.spin.SpinOperatorTerm | -   [cud                          |
|         method)](api/languages/p  | aq::phase_damping::num_parameters |
| ython_api.html#cudaq.operators.sp |     (C++                          |
| in.SpinOperatorTerm.canonicalize) |     member)](api/lan              |
| -   [canonicalized() (in module   | guages/cpp_api.html#_CPPv4N5cudaq |
|     cuda                          | 13phase_damping14num_parametersE) |
| q.boson)](api/languages/python_ap | -   [                             |
| i.html#cudaq.boson.canonicalized) | cudaq::phase_damping::num_targets |
|     -   [(in module               |     (C++                          |
|         cudaq.fe                  |     member)](api/                 |
| rmion)](api/languages/python_api. | languages/cpp_api.html#_CPPv4N5cu |
| html#cudaq.fermion.canonicalized) | daq13phase_damping11num_targetsE) |
|     -   [(in module               | -   [cudaq::phase_flip_channel    |
|                                   |     (C++                          |
|        cudaq.operators.custom)](a |     clas                          |
| pi/languages/python_api.html#cuda | s)](api/languages/cpp_api.html#_C |
| q.operators.custom.canonicalized) | PPv4N5cudaq18phase_flip_channelE) |
|     -   [(in module               | -   [cudaq::p                     |
|         cu                        | hase_flip_channel::num_parameters |
| daq.spin)](api/languages/python_a |     (C++                          |
| pi.html#cudaq.spin.canonicalized) |     member)](api/language         |
| -   [CentralDifference (class in  | s/cpp_api.html#_CPPv4N5cudaq18pha |
|     cudaq.gradients)              | se_flip_channel14num_parametersE) |
| ](api/languages/python_api.html#c | -   [cudaq                        |
| udaq.gradients.CentralDifference) | ::phase_flip_channel::num_targets |
| -   [clear() (cudaq.Resources     |     (C++                          |
|     method)](api/languages/pytho  |     member)](api/langu            |
| n_api.html#cudaq.Resources.clear) | ages/cpp_api.html#_CPPv4N5cudaq18 |
|     -   [(cudaq.SampleResult      | phase_flip_channel11num_targetsE) |
|                                   | -   [cudaq::product_op (C++       |
|   method)](api/languages/python_a |                                   |
| pi.html#cudaq.SampleResult.clear) |  class)](api/languages/cpp_api.ht |
| -   [COBYLA (class in             | ml#_CPPv4I0EN5cudaq10product_opE) |
|     cudaq.o                       | -   [cudaq::product_op::begin     |
| ptimizers)](api/languages/python_ |     (C++                          |
| api.html#cudaq.optimizers.COBYLA) |     functio                       |
| -   [coefficient                  | n)](api/languages/cpp_api.html#_C |
|     (cudaq.                       | PPv4NK5cudaq10product_op5beginEv) |
| operators.boson.BosonOperatorTerm | -                                 |
|     property)](api/languages/py   |  [cudaq::product_op::canonicalize |
| thon_api.html#cudaq.operators.bos |     (C++                          |
| on.BosonOperatorTerm.coefficient) |     func                          |
|     -   [(cudaq.oper              | tion)](api/languages/cpp_api.html |
| ators.fermion.FermionOperatorTerm | #_CPPv4N5cudaq10product_op12canon |
|                                   | icalizeERKNSt3setINSt6size_tEEE), |
|   property)](api/languages/python |     [\[1\]](api                   |
| _api.html#cudaq.operators.fermion | /languages/cpp_api.html#_CPPv4N5c |
| .FermionOperatorTerm.coefficient) | udaq10product_op12canonicalizeEv) |
|     -   [(c                       | -   [                             |
| udaq.operators.MatrixOperatorTerm | cudaq::product_op::const_iterator |
|         property)](api/languag    |     (C++                          |
| es/python_api.html#cudaq.operator |     struct)](api/                 |
| s.MatrixOperatorTerm.coefficient) | languages/cpp_api.html#_CPPv4N5cu |
|     -   [(cuda                    | daq10product_op14const_iteratorE) |
| q.operators.spin.SpinOperatorTerm | -   [cudaq::product_o             |
|         property)](api/languages/ | p::const_iterator::const_iterator |
| python_api.html#cudaq.operators.s |     (C++                          |
| pin.SpinOperatorTerm.coefficient) |     fu                            |
| -   [col_count                    | nction)](api/languages/cpp_api.ht |
|     (cudaq.KrausOperator          | ml#_CPPv4N5cudaq10product_op14con |
|     prope                         | st_iterator14const_iteratorEPK10p |
| rty)](api/languages/python_api.ht | roduct_opI9HandlerTyENSt6size_tE) |
| ml#cudaq.KrausOperator.col_count) | -   [cudaq::produ                 |
| -   [compile()                    | ct_op::const_iterator::operator!= |
|     (cudaq.PyKernelDecorator      |     (C++                          |
|     metho                         |     fun                           |
| d)](api/languages/python_api.html | ction)](api/languages/cpp_api.htm |
| #cudaq.PyKernelDecorator.compile) | l#_CPPv4NK5cudaq10product_op14con |
| -   [ComplexMatrix (class in      | st_iteratorneERK14const_iterator) |
|     cudaq)](api/languages/pyt     | -   [cudaq::produ                 |
| hon_api.html#cudaq.ComplexMatrix) | ct_op::const_iterator::operator\* |
| -   [compute()                    |     (C++                          |
|     (                             |     function)](api/lang           |
| cudaq.gradients.CentralDifference | uages/cpp_api.html#_CPPv4NK5cudaq |
|     method)](api/la               | 10product_op14const_iteratormlEv) |
| nguages/python_api.html#cudaq.gra | -   [cudaq::produ                 |
| dients.CentralDifference.compute) | ct_op::const_iterator::operator++ |
|     -   [(                        |     (C++                          |
| cudaq.gradients.ForwardDifference |     function)](api/lang           |
|         method)](api/la           | uages/cpp_api.html#_CPPv4N5cudaq1 |
| nguages/python_api.html#cudaq.gra | 0product_op14const_iteratorppEi), |
| dients.ForwardDifference.compute) |     [\[1\]](api/lan               |
|     -                             | guages/cpp_api.html#_CPPv4N5cudaq |
|  [(cudaq.gradients.ParameterShift | 10product_op14const_iteratorppEv) |
|         method)](api              | -   [cudaq::produc                |
| /languages/python_api.html#cudaq. | t_op::const_iterator::operator\-- |
| gradients.ParameterShift.compute) |     (C++                          |
| -   [const()                      |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|   (cudaq.operators.ScalarOperator | 0product_op14const_iteratormmEi), |
|     class                         |     [\[1\]](api/lan               |
|     method)](a                    | guages/cpp_api.html#_CPPv4N5cudaq |
| pi/languages/python_api.html#cuda | 10product_op14const_iteratormmEv) |
| q.operators.ScalarOperator.const) | -   [cudaq::produc                |
| -   [copy()                       | t_op::const_iterator::operator-\> |
|     (cu                           |     (C++                          |
| daq.operators.boson.BosonOperator |     function)](api/lan            |
|     method)](api/l                | guages/cpp_api.html#_CPPv4N5cudaq |
| anguages/python_api.html#cudaq.op | 10product_op14const_iteratorptEv) |
| erators.boson.BosonOperator.copy) | -   [cudaq::produ                 |
|     -   [(cudaq.                  | ct_op::const_iterator::operator== |
| operators.boson.BosonOperatorTerm |     (C++                          |
|         method)](api/langu        |     fun                           |
| ages/python_api.html#cudaq.operat | ction)](api/languages/cpp_api.htm |
| ors.boson.BosonOperatorTerm.copy) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(cudaq.                  | st_iteratoreqERK14const_iterator) |
| operators.fermion.FermionOperator | -   [cudaq::product_op::degrees   |
|         method)](api/langu        |     (C++                          |
| ages/python_api.html#cudaq.operat |     function)                     |
| ors.fermion.FermionOperator.copy) | ](api/languages/cpp_api.html#_CPP |
|     -   [(cudaq.oper              | v4NK5cudaq10product_op7degreesEv) |
| ators.fermion.FermionOperatorTerm | -   [cudaq::product_op::dump (C++ |
|         method)](api/languages    |     functi                        |
| /python_api.html#cudaq.operators. | on)](api/languages/cpp_api.html#_ |
| fermion.FermionOperatorTerm.copy) | CPPv4NK5cudaq10product_op4dumpEv) |
|     -                             | -   [cudaq::product_op::end (C++  |
|  [(cudaq.operators.MatrixOperator |     funct                         |
|         method)](                 | ion)](api/languages/cpp_api.html# |
| api/languages/python_api.html#cud | _CPPv4NK5cudaq10product_op3endEv) |
| aq.operators.MatrixOperator.copy) | -   [c                            |
|     -   [(c                       | udaq::product_op::get_coefficient |
| udaq.operators.MatrixOperatorTerm |     (C++                          |
|         method)](api/             |     function)](api/lan            |
| languages/python_api.html#cudaq.o | guages/cpp_api.html#_CPPv4NK5cuda |
| perators.MatrixOperatorTerm.copy) | q10product_op15get_coefficientEv) |
|     -   [(                        | -                                 |
| cudaq.operators.spin.SpinOperator |   [cudaq::product_op::get_term_id |
|         method)](api              |     (C++                          |
| /languages/python_api.html#cudaq. |     function)](api                |
| operators.spin.SpinOperator.copy) | /languages/cpp_api.html#_CPPv4NK5 |
|     -   [(cuda                    | cudaq10product_op11get_term_idEv) |
| q.operators.spin.SpinOperatorTerm | -                                 |
|         method)](api/lan          |   [cudaq::product_op::is_identity |
| guages/python_api.html#cudaq.oper |     (C++                          |
| ators.spin.SpinOperatorTerm.copy) |     function)](api                |
| -   [count() (cudaq.Resources     | /languages/cpp_api.html#_CPPv4NK5 |
|     method)](api/languages/pytho  | cudaq10product_op11is_identityEv) |
| n_api.html#cudaq.Resources.count) | -   [cudaq::product_op::num_ops   |
|     -   [(cudaq.SampleResult      |     (C++                          |
|                                   |     function)                     |
|   method)](api/languages/python_a | ](api/languages/cpp_api.html#_CPP |
| pi.html#cudaq.SampleResult.count) | v4NK5cudaq10product_op7num_opsEv) |
| -   [count_controls()             | -                                 |
|     (cudaq.Resources              |    [cudaq::product_op::operator\* |
|     meth                          |     (C++                          |
| od)](api/languages/python_api.htm |     function)](api/languages/     |
| l#cudaq.Resources.count_controls) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [counts()                     | oduct_opmlE10product_opI1TERK15sc |
|     (cudaq.ObserveResult          | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]](api/languages/        |
| method)](api/languages/python_api | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| .html#cudaq.ObserveResult.counts) | oduct_opmlE10product_opI1TERK15sc |
| -   [create() (in module          | alar_operatorRR10product_opI1TE), |
|                                   |     [\[2\]](api/languages/        |
|    cudaq.boson)](api/languages/py | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| thon_api.html#cudaq.boson.create) | oduct_opmlE10product_opI1TERR15sc |
|     -   [(in module               | alar_operatorRK10product_opI1TE), |
|         c                         |     [\[3\]](api/languages/        |
| udaq.fermion)](api/languages/pyth | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| on_api.html#cudaq.fermion.create) | oduct_opmlE10product_opI1TERR15sc |
| -   [csr_spmatrix (C++            | alar_operatorRR10product_opI1TE), |
|     type)](api/languages/c        |     [\[4\]](api/                  |
| pp_api.html#_CPPv412csr_spmatrix) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq                         | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [module](api/langua       | K15scalar_operatorRK6sum_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[5\]](api/                  |
| -   [cudaq (C++                   | languages/cpp_api.html#_CPPv4I0EN |
|     type)](api/lan                | 5cudaq10product_opmlE6sum_opI1TER |
| guages/cpp_api.html#_CPPv45cudaq) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq.apply_noise() (in      |     [\[6\]](api/                  |
|     module                        | languages/cpp_api.html#_CPPv4I0EN |
|     cudaq)](api/languages/python_ | 5cudaq10product_opmlE6sum_opI1TER |
| api.html#cudaq.cudaq.apply_noise) | R15scalar_operatorRK6sum_opI1TE), |
| -   cudaq.boson                   |     [\[7\]](api/                  |
|     -   [module](api/languages/py | languages/cpp_api.html#_CPPv4I0EN |
| thon_api.html#module-cudaq.boson) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq.fermion                 | R15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[8\]](api/languages         |
|   -   [module](api/languages/pyth | /cpp_api.html#_CPPv4NK5cudaq10pro |
| on_api.html#module-cudaq.fermion) | duct_opmlERK6sum_opI9HandlerTyE), |
| -   cudaq.operators.custom        |     [\[9\]](api/languages/cpp_a   |
|     -   [mo                       | pi.html#_CPPv4NKR5cudaq10product_ |
| dule](api/languages/python_api.ht | opmlERK10product_opI9HandlerTyE), |
| ml#module-cudaq.operators.custom) |     [\[10\]](api/language         |
| -   cudaq.spin                    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [module](api/languages/p  | roduct_opmlERK15scalar_operator), |
| ython_api.html#module-cudaq.spin) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq::amplitude_damping     | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmlERR10product_opI9HandlerTyE), |
|     cla                           |     [\[12\]](api/language         |
| ss)](api/languages/cpp_api.html#_ | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| CPPv4N5cudaq17amplitude_dampingE) | roduct_opmlERR15scalar_operator), |
| -                                 |     [\[13\]](api/languages/cpp_   |
| [cudaq::amplitude_damping_channel | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmlERK10product_opI9HandlerTyE), |
|     class)](api                   |     [\[14\]](api/languag          |
| /languages/cpp_api.html#_CPPv4N5c | es/cpp_api.html#_CPPv4NO5cudaq10p |
| udaq25amplitude_damping_channelE) | roduct_opmlERK15scalar_operator), |
| -   [cudaq::amplitud              |     [\[15\]](api/languages/cpp_   |
| e_damping_channel::num_parameters | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmlERR10product_opI9HandlerTyE), |
|     member)](api/languages/cpp_a  |     [\[16\]](api/langua           |
| pi.html#_CPPv4N5cudaq25amplitude_ | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| damping_channel14num_parametersE) | product_opmlERR15scalar_operator) |
| -   [cudaq::ampli                 | -                                 |
| tude_damping_channel::num_targets |   [cudaq::product_op::operator\*= |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     function)](api/languages/cpp  |
| p_api.html#_CPPv4N5cudaq25amplitu | _api.html#_CPPv4N5cudaq10product_ |
| de_damping_channel11num_targetsE) | opmLERK10product_opI9HandlerTyE), |
| -   [cudaq::AnalogRemoteRESTQPU   |     [\[1\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     class                         | roduct_opmLERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[2\]](api/languages/cp      |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::apply_noise (C++      | _opmLERR10product_opI9HandlerTyE) |
|     function)](api/               | -   [cudaq::product_op::operator+ |
| languages/cpp_api.html#_CPPv4I0Dp |     (C++                          |
| EN5cudaq11apply_noiseEvDpRR4Args) |     function)](api/langu          |
| -   [cudaq::async_result (C++     | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     c                             | q10product_opplE6sum_opI1TERK15sc |
| lass)](api/languages/cpp_api.html | alar_operatorRK10product_opI1TE), |
| #_CPPv4I0EN5cudaq12async_resultE) |     [\[1\]](api/                  |
| -   [cudaq::async_result::get     | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     functi                        | K15scalar_operatorRK6sum_opI1TE), |
| on)](api/languages/cpp_api.html#_ |     [\[2\]](api/langu             |
| CPPv4N5cudaq12async_result3getEv) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::async_sample_result   | q10product_opplE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     type                          |     [\[3\]](api/                  |
| )](api/languages/cpp_api.html#_CP | languages/cpp_api.html#_CPPv4I0EN |
| Pv4N5cudaq19async_sample_resultE) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::BaseNvcfSimulatorQPU  | K15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[4\]](api/langu             |
|     class)                        | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ](api/languages/cpp_api.html#_CPP | q10product_opplE6sum_opI1TERR15sc |
| v4N5cudaq20BaseNvcfSimulatorQPUE) | alar_operatorRK10product_opI1TE), |
| -   [cudaq::BaseRemoteRESTQPU     |     [\[5\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     cla                           | 5cudaq10product_opplE6sum_opI1TER |
| ss)](api/languages/cpp_api.html#_ | R15scalar_operatorRK6sum_opI1TE), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [\[6\]](api/langu             |
| -                                 | ages/cpp_api.html#_CPPv4I0EN5cuda |
|    [cudaq::BaseRemoteSimulatorQPU | q10product_opplE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     class)](                      |     [\[7\]](api/                  |
| api/languages/cpp_api.html#_CPPv4 | languages/cpp_api.html#_CPPv4I0EN |
| N5cudaq22BaseRemoteSimulatorQPUE) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::bit_flip_channel (C++ | R15scalar_operatorRR6sum_opI1TE), |
|     cl                            |     [\[8\]](api/languages/cpp_a   |
| ass)](api/languages/cpp_api.html# | pi.html#_CPPv4NKR5cudaq10product_ |
| _CPPv4N5cudaq16bit_flip_channelE) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq:                       |     [\[9\]](api/language          |
| :bit_flip_channel::num_parameters | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opplERK15scalar_operator), |
|     member)](api/langua           |     [\[10\]](api/languages/       |
| ges/cpp_api.html#_CPPv4N5cudaq16b | cpp_api.html#_CPPv4NKR5cudaq10pro |
| it_flip_channel14num_parametersE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cud                          |     [\[11\]](api/languages/cpp_a  |
| aq::bit_flip_channel::num_targets | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opplERR10product_opI9HandlerTyE), |
|     member)](api/lan              |     [\[12\]](api/language         |
| guages/cpp_api.html#_CPPv4N5cudaq | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| 16bit_flip_channel11num_targetsE) | roduct_opplERR15scalar_operator), |
| -   [cudaq::boson_handler (C++    |     [\[13\]](api/languages/       |
|                                   | cpp_api.html#_CPPv4NKR5cudaq10pro |
|  class)](api/languages/cpp_api.ht | duct_opplERR6sum_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13boson_handlerE) |     [\[                           |
| -   [cudaq::boson_op (C++         | 14\]](api/languages/cpp_api.html# |
|     type)](api/languages/cpp_     | _CPPv4NKR5cudaq10product_opplEv), |
| api.html#_CPPv4N5cudaq8boson_opE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::boson_op_term (C++    | api.html#_CPPv4NO5cudaq10product_ |
|                                   | opplERK10product_opI9HandlerTyE), |
|   type)](api/languages/cpp_api.ht |     [\[16\]](api/languag          |
| ml#_CPPv4N5cudaq13boson_op_termE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::CodeGenConfig (C++    | roduct_opplERK15scalar_operator), |
|                                   |     [\[17\]](api/languages        |
| struct)](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq::commutation_relations |     [\[18\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     struct)]                      | opplERR10product_opI9HandlerTyE), |
| (api/languages/cpp_api.html#_CPPv |     [\[19\]](api/languag          |
| 4N5cudaq21commutation_relationsE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::complex (C++          | roduct_opplERR15scalar_operator), |
|     type)](api/languages/cpp      |     [\[20\]](api/languages        |
| _api.html#_CPPv4N5cudaq7complexE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::complex_matrix (C++   | duct_opplERR6sum_opI9HandlerTyE), |
|                                   |     [                             |
| class)](api/languages/cpp_api.htm | \[21\]](api/languages/cpp_api.htm |
| l#_CPPv4N5cudaq14complex_matrixE) | l#_CPPv4NO5cudaq10product_opplEv) |
| -                                 | -   [cudaq::product_op::operator- |
|   [cudaq::complex_matrix::adjoint |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](a                  | ages/cpp_api.html#_CPPv4I0EN5cuda |
| pi/languages/cpp_api.html#_CPPv4N | q10product_opmiE6sum_opI1TERK15sc |
| 5cudaq14complex_matrix7adjointEv) | alar_operatorRK10product_opI1TE), |
| -   [cudaq::                      |     [\[1\]](api/                  |
| complex_matrix::diagonal_elements | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/languages      | K15scalar_operatorRK6sum_opI1TE), |
| /cpp_api.html#_CPPv4NK5cudaq14com |     [\[2\]](api/langu             |
| plex_matrix17diagonal_elementsEi) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::complex_matrix::dump  | q10product_opmiE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     function)](api/language       |     [\[3\]](api/                  |
| s/cpp_api.html#_CPPv4NK5cudaq14co | languages/cpp_api.html#_CPPv4I0EN |
| mplex_matrix4dumpERNSt7ostreamE), | 5cudaq10product_opmiE6sum_opI1TER |
|     [\[1\]]                       | K15scalar_operatorRR6sum_opI1TE), |
| (api/languages/cpp_api.html#_CPPv |     [\[4\]](api/langu             |
| 4NK5cudaq14complex_matrix4dumpEv) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [c                            | q10product_opmiE6sum_opI1TERR15sc |
| udaq::complex_matrix::eigenvalues | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[5\]](api/                  |
|     function)](api/lan            | languages/cpp_api.html#_CPPv4I0EN |
| guages/cpp_api.html#_CPPv4NK5cuda | 5cudaq10product_opmiE6sum_opI1TER |
| q14complex_matrix11eigenvaluesEv) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cu                           |     [\[6\]](api/langu             |
| daq::complex_matrix::eigenvectors | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERR15sc |
|     function)](api/lang           | alar_operatorRR10product_opI1TE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[7\]](api/                  |
| 14complex_matrix12eigenvectorsEv) | languages/cpp_api.html#_CPPv4I0EN |
| -   [c                            | 5cudaq10product_opmiE6sum_opI1TER |
| udaq::complex_matrix::exponential | R15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[8\]](api/languages/cpp_a   |
|     function)](api/la             | pi.html#_CPPv4NKR5cudaq10product_ |
| nguages/cpp_api.html#_CPPv4N5cuda | opmiERK10product_opI9HandlerTyE), |
| q14complex_matrix11exponentialEv) |     [\[9\]](api/language          |
| -                                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|  [cudaq::complex_matrix::identity | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[10\]](api/languages/       |
|     function)](api/languages      | cpp_api.html#_CPPv4NKR5cudaq10pro |
| /cpp_api.html#_CPPv4N5cudaq14comp | duct_opmiERK6sum_opI9HandlerTyE), |
| lex_matrix8identityEKNSt6size_tE) |     [\[11\]](api/languages/cpp_a  |
| -                                 | pi.html#_CPPv4NKR5cudaq10product_ |
| [cudaq::complex_matrix::kronecker | opmiERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[12\]](api/language         |
|     function)](api/lang           | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| uages/cpp_api.html#_CPPv4I00EN5cu | roduct_opmiERR15scalar_operator), |
| daq14complex_matrix9kroneckerE14c |     [\[13\]](api/languages/       |
| omplex_matrix8Iterable8Iterable), | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     [\[1\]](api/l                 | duct_opmiERR6sum_opI9HandlerTyE), |
| anguages/cpp_api.html#_CPPv4N5cud |     [\[                           |
| aq14complex_matrix9kroneckerERK14 | 14\]](api/languages/cpp_api.html# |
| complex_matrixRK14complex_matrix) | _CPPv4NKR5cudaq10product_opmiEv), |
| -   [cudaq::c                     |     [\[15\]](api/languages/cpp_   |
| omplex_matrix::minimal_eigenvalue | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmiERK10product_opI9HandlerTyE), |
|     function)](api/languages/     |     [\[16\]](api/languag          |
| cpp_api.html#_CPPv4NK5cudaq14comp | es/cpp_api.html#_CPPv4NO5cudaq10p |
| lex_matrix18minimal_eigenvalueEv) | roduct_opmiERK15scalar_operator), |
| -   [                             |     [\[17\]](api/languages        |
| cudaq::complex_matrix::operator() | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERK6sum_opI9HandlerTyE), |
|     function)](api/languages/cpp  |     [\[18\]](api/languages/cpp_   |
| _api.html#_CPPv4N5cudaq14complex_ | api.html#_CPPv4NO5cudaq10product_ |
| matrixclENSt6size_tENSt6size_tE), | opmiERR10product_opI9HandlerTyE), |
|     [\[1\]](api/languages/cpp     |     [\[19\]](api/languag          |
| _api.html#_CPPv4NK5cudaq14complex | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _matrixclENSt6size_tENSt6size_tE) | roduct_opmiERR15scalar_operator), |
| -   [                             |     [\[20\]](api/languages        |
| cudaq::complex_matrix::operator\* | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERR6sum_opI9HandlerTyE), |
|     function)](api/langua         |     [                             |
| ges/cpp_api.html#_CPPv4N5cudaq14c | \[21\]](api/languages/cpp_api.htm |
| omplex_matrixmlEN14complex_matrix | l#_CPPv4NO5cudaq10product_opmiEv) |
| 10value_typeERK14complex_matrix), | -   [cudaq::product_op::operator/ |
|     [\[1\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/language       |
| v4N5cudaq14complex_matrixmlERK14c | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| omplex_matrixRK14complex_matrix), | roduct_opdvERK15scalar_operator), |
|                                   |     [\[1\]](api/language          |
|  [\[2\]](api/languages/cpp_api.ht | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ml#_CPPv4N5cudaq14complex_matrixm | roduct_opdvERR15scalar_operator), |
| lERK14complex_matrixRKNSt6vectorI |     [\[2\]](api/languag           |
| N14complex_matrix10value_typeEEE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -                                 | roduct_opdvERK15scalar_operator), |
| [cudaq::complex_matrix::operator+ |     [\[3\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     function                      | product_opdvERR15scalar_operator) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq14complex_matrixplERK14 |    [cudaq::product_op::operator/= |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -                                 |     function)](api/langu          |
| [cudaq::complex_matrix::operator- | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     (C++                          | product_opdVERK15scalar_operator) |
|     function                      | -   [cudaq::product_op::operator= |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq14complex_matrixmiERK14 |     function)](api/la             |
| complex_matrixRK14complex_matrix) | nguages/cpp_api.html#_CPPv4I0_NSt |
| -   [cu                           | 11enable_if_tIXaantNSt7is_sameI1T |
| daq::complex_matrix::operator\[\] | 9HandlerTyE5valueENSt16is_constru |
|     (C++                          | ctibleI9HandlerTy1TE5valueEEbEEEN |
|                                   | 5cudaq10product_opaSER10product_o |
|  function)](api/languages/cpp_api | pI9HandlerTyERK10product_opI1TE), |
| .html#_CPPv4N5cudaq14complex_matr |     [\[1\]](api/languages/cpp     |
| ixixERKNSt6vectorINSt6size_tEEE), | _api.html#_CPPv4N5cudaq10product_ |
|     [\[1\]](api/languages/cpp_api | opaSERK10product_opI9HandlerTyE), |
| .html#_CPPv4NK5cudaq14complex_mat |     [\[2\]](api/languages/cp      |
| rixixERKNSt6vectorINSt6size_tEEE) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::complex_matrix::power | _opaSERR10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     function)]                    |    [cudaq::product_op::operator== |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq14complex_matrix5powerEi) |     function)](api/languages/cpp  |
| -                                 | _api.html#_CPPv4NK5cudaq10product |
|  [cudaq::complex_matrix::set_zero | _opeqERK10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     function)](ap                 |  [cudaq::product_op::operator\[\] |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq14complex_matrix8set_zeroEv) |     function)](ap                 |
| -                                 | i/languages/cpp_api.html#_CPPv4NK |
| [cudaq::complex_matrix::to_string | 5cudaq10product_opixENSt6size_tE) |
|     (C++                          | -                                 |
|     function)](api/               |    [cudaq::product_op::product_op |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14complex_matrix9to_stringEv) |     function)](api/languages/c    |
| -   [                             | pp_api.html#_CPPv4I0_NSt11enable_ |
| cudaq::complex_matrix::value_type | if_tIXaaNSt7is_sameI9HandlerTy14m |
|     (C++                          | atrix_handlerE5valueEaantNSt7is_s |
|     type)](api/                   | ameI1T9HandlerTyE5valueENSt16is_c |
| languages/cpp_api.html#_CPPv4N5cu | onstructibleI9HandlerTy1TE5valueE |
| daq14complex_matrix10value_typeE) | EbEEEN5cudaq10product_op10product |
| -   [cudaq::contrib (C++          | _opERK10product_opI1TERKN14matrix |
|     type)](api/languages/cpp      | _handler20commutation_behaviorE), |
| _api.html#_CPPv4N5cudaq7contribE) |                                   |
| -   [cudaq::contrib::draw (C++    |  [\[1\]](api/languages/cpp_api.ht |
|     function)]                    | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| (api/languages/cpp_api.html#_CPPv | tNSt7is_sameI1T9HandlerTyE5valueE |
| 4I0Dp0EN5cudaq7contrib4drawENSt6s | NSt16is_constructibleI9HandlerTy1 |
| tringERR13QuantumKernelDpRR4Args) | TE5valueEEbEEEN5cudaq10product_op |
| -                                 | 10product_opERK10product_opI1TE), |
| [cudaq::contrib::get_unitary_cmat |                                   |
|     (C++                          |   [\[2\]](api/languages/cpp_api.h |
|     function)](api/languages/cp   | tml#_CPPv4N5cudaq10product_op10pr |
| p_api.html#_CPPv4I0DpEN5cudaq7con | oduct_opENSt6size_tENSt6size_tE), |
| trib16get_unitary_cmatE14complex_ |     [\[3\]](api/languages/cp      |
| matrixRR13QuantumKernelDpRR4Args) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::CusvState (C++        | _op10product_opENSt7complexIdEE), |
|                                   |     [\[4\]](api/l                 |
|    class)](api/languages/cpp_api. | anguages/cpp_api.html#_CPPv4N5cud |
| html#_CPPv4I0EN5cudaq9CusvStateE) | aq10product_op10product_opERK10pr |
| -   [cudaq::depolarization1 (C++  | oduct_opI9HandlerTyENSt6size_tE), |
|     c                             |     [\[5\]](api/l                 |
| lass)](api/languages/cpp_api.html | anguages/cpp_api.html#_CPPv4N5cud |
| #_CPPv4N5cudaq15depolarization1E) | aq10product_op10product_opERR10pr |
| -   [cudaq::depolarization2 (C++  | oduct_opI9HandlerTyENSt6size_tE), |
|     c                             |     [\[6\]](api/languages         |
| lass)](api/languages/cpp_api.html | /cpp_api.html#_CPPv4N5cudaq10prod |
| #_CPPv4N5cudaq15depolarization2E) | uct_op10product_opERR9HandlerTy), |
| -   [cudaq:                       |     [\[7\]](ap                    |
| :depolarization2::depolarization2 | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq10product_op10product_opEd), |
|     function)](api/languages/cp   |     [\[8\]](a                     |
| p_api.html#_CPPv4N5cudaq15depolar | pi/languages/cpp_api.html#_CPPv4N |
| ization215depolarization2EK4real) | 5cudaq10product_op10product_opEv) |
| -   [cudaq                        | -   [cuda                         |
| ::depolarization2::num_parameters | q::product_op::to_diagonal_matrix |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api/               |
| ages/cpp_api.html#_CPPv4N5cudaq15 | languages/cpp_api.html#_CPPv4NK5c |
| depolarization214num_parametersE) | udaq10product_op18to_diagonal_mat |
| -   [cu                           | rixENSt13unordered_mapINSt6size_t |
| daq::depolarization2::num_targets | ENSt7int64_tEEERKNSt13unordered_m |
|     (C++                          | apINSt6stringENSt7complexIdEEEEb) |
|     member)](api/la               | -   [cudaq::product_op::to_matrix |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q15depolarization211num_targetsE) |     funct                         |
| -                                 | ion)](api/languages/cpp_api.html# |
|    [cudaq::depolarization_channel | _CPPv4NK5cudaq10product_op9to_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     class)](                      | ENSt7int64_tEEERKNSt13unordered_m |
| api/languages/cpp_api.html#_CPPv4 | apINSt6stringENSt7complexIdEEEEb) |
| N5cudaq22depolarization_channelE) | -   [cu                           |
| -   [cudaq::depol                 | daq::product_op::to_sparse_matrix |
| arization_channel::num_parameters |     (C++                          |
|     (C++                          |     function)](ap                 |
|     member)](api/languages/cp     | i/languages/cpp_api.html#_CPPv4NK |
| p_api.html#_CPPv4N5cudaq22depolar | 5cudaq10product_op16to_sparse_mat |
| ization_channel14num_parametersE) | rixENSt13unordered_mapINSt6size_t |
| -   [cudaq::de                    | ENSt7int64_tEEERKNSt13unordered_m |
| polarization_channel::num_targets | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cudaq::product_op::to_string |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq22depo |     function)](                   |
| larization_channel11num_targetsE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::details (C++          | NK5cudaq10product_op9to_stringEv) |
|     type)](api/languages/cpp      | -                                 |
| _api.html#_CPPv4N5cudaq7detailsE) |  [cudaq::product_op::\~product_op |
| -   [cudaq::details::future (C++  |     (C++                          |
|                                   |     fu                            |
|  class)](api/languages/cpp_api.ht | nction)](api/languages/cpp_api.ht |
| ml#_CPPv4N5cudaq7details6futureE) | ml#_CPPv4N5cudaq10product_opD0Ev) |
| -                                 | -   [cudaq::QPU (C++              |
|   [cudaq::details::future::future |     class)](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     functio                       | -   [cudaq::QPU::enqueue (C++     |
| n)](api/languages/cpp_api.html#_C |     function)](ap                 |
| PPv4N5cudaq7details6future6future | i/languages/cpp_api.html#_CPPv4N5 |
| ERNSt6vectorI3JobEERNSt6stringERN | cudaq3QPU7enqueueER11QuantumTask) |
| St3mapINSt6stringENSt6stringEEE), | -   [cudaq::QPU::getConnectivity  |
|     [\[1\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     function)                     |
| details6future6futureERR6future), | ](api/languages/cpp_api.html#_CPP |
|     [\[2\]]                       | v4N5cudaq3QPU15getConnectivityEv) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq7details6future6futureEv) | [cudaq::QPU::getExecutionThreadId |
| -   [cu                           |     (C++                          |
| daq::details::kernel_builder_base |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4NK5c |
|     class)](api/l                 | udaq3QPU20getExecutionThreadIdEv) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cudaq::QPU::getNumQubits     |
| aq7details19kernel_builder_baseE) |     (C++                          |
| -   [cudaq::details::             |     functi                        |
| kernel_builder_base::operator\<\< | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     function)](api/langua         | -   [                             |
| ges/cpp_api.html#_CPPv4N5cudaq7de | cudaq::QPU::getRemoteCapabilities |
| tails19kernel_builder_baselsERNSt |     (C++                          |
| 7ostreamERK19kernel_builder_base) |     function)](api/l              |
| -   [                             | anguages/cpp_api.html#_CPPv4NK5cu |
| cudaq::details::KernelBuilderType | daq3QPU21getRemoteCapabilitiesEv) |
|     (C++                          | -   [cudaq::QPU::isEmulated (C++  |
|     class)](api                   |     func                          |
| /languages/cpp_api.html#_CPPv4N5c | tion)](api/languages/cpp_api.html |
| udaq7details17KernelBuilderTypeE) | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| -   [cudaq::d                     | -   [cudaq::QPU::isSimulator (C++ |
| etails::KernelBuilderType::create |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)                     | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::QPU::launchKernel     |
| v4N5cudaq7details17KernelBuilderT |     (C++                          |
| ype6createEPN4mlir11MLIRContextE) |     function)](api/               |
| -   [cudaq::details::Ker          | languages/cpp_api.html#_CPPv4N5cu |
| nelBuilderType::KernelBuilderType | daq3QPU12launchKernelERKNSt6strin |
|     (C++                          | gE15KernelThunkTypePvNSt8uint64_t |
|     function)](api/lang           | ENSt8uint64_tERKNSt6vectorIPvEE), |
| uages/cpp_api.html#_CPPv4N5cudaq7 |                                   |
| details17KernelBuilderType17Kerne |  [\[1\]](api/languages/cpp_api.ht |
| lBuilderTypeERRNSt8functionIFN4ml | ml#_CPPv4N5cudaq3QPU12launchKerne |
| ir4TypeEPN4mlir11MLIRContextEEEE) | lERKNSt6stringERKNSt6vectorIPvEE) |
| -   [cudaq::diag_matrix_callback  | -   [cudaq::Q                     |
|     (C++                          | PU::launchSerializedCodeExecution |
|     class)                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)]                    |
| v4N5cudaq20diag_matrix_callbackE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::dyn (C++              | 4N5cudaq3QPU29launchSerializedCod |
|     member)](api/languages        | eExecutionERKNSt6stringERN5cudaq3 |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | 0SerializedCodeExecutionContextE) |
| -   [cudaq::ExecutionContext (C++ | -   [cudaq::QPU::onRandomSeedSet  |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     function)](api/lang           |
| _CPPv4N5cudaq16ExecutionContextE) | uages/cpp_api.html#_CPPv4N5cudaq3 |
| -   [cudaq                        | QPU15onRandomSeedSetENSt6size_tE) |
| ::ExecutionContext::amplitudeMaps | -   [cudaq::QPU::QPU (C++         |
|     (C++                          |     functio                       |
|     member)](api/langu            | n)](api/languages/cpp_api.html#_C |
| ages/cpp_api.html#_CPPv4N5cudaq16 | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| ExecutionContext13amplitudeMapsE) |                                   |
| -   [c                            |  [\[1\]](api/languages/cpp_api.ht |
| udaq::ExecutionContext::asyncExec | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     (C++                          |     [\[2\]](api/languages/cpp_    |
|     member)](api/                 | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| languages/cpp_api.html#_CPPv4N5cu | -   [                             |
| daq16ExecutionContext9asyncExecE) | cudaq::QPU::resetExecutionContext |
| -   [cud                          |     (C++                          |
| aq::ExecutionContext::asyncResult |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     member)](api/lan              | daq3QPU21resetExecutionContextEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -                                 |
| 16ExecutionContext11asyncResultE) |  [cudaq::QPU::setExecutionContext |
| -   [cudaq:                       |     (C++                          |
| :ExecutionContext::batchIteration |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     member)](api/langua           | i.html#_CPPv4N5cudaq3QPU19setExec |
| ges/cpp_api.html#_CPPv4N5cudaq16E | utionContextEP16ExecutionContext) |
| xecutionContext14batchIterationE) | -   [cudaq::QPU::setId (C++       |
| -   [cudaq::E                     |     function                      |
| xecutionContext::canHandleObserve | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     member)](api/language         | -   [cudaq::QPU::setShots (C++    |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     f                             |
| cutionContext16canHandleObserveE) | unction)](api/languages/cpp_api.h |
| -   [cudaq::E                     | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| xecutionContext::ExecutionContext | -   [cudaq:                       |
|     (C++                          | :QPU::supportsConditionalFeedback |
|     func                          |     (C++                          |
| tion)](api/languages/cpp_api.html |     function)](api/langua         |
| #_CPPv4N5cudaq16ExecutionContext1 | ges/cpp_api.html#_CPPv4N5cudaq3QP |
| 6ExecutionContextERKNSt6stringE), | U27supportsConditionalFeedbackEv) |
|     [\[1\]](api                   | -   [cudaq::                      |
| /languages/cpp_api.html#_CPPv4N5c | QPU::supportsExplicitMeasurements |
| udaq16ExecutionContext16Execution |     (C++                          |
| ContextERKNSt6stringENSt6size_tE) |     function)](api/languag        |
| -   [cudaq::E                     | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| xecutionContext::expectationValue | 28supportsExplicitMeasurementsEv) |
|     (C++                          | -   [cudaq::QPU::\~QPU (C++       |
|     member)](api/language         |     function)](api/languages/cp   |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| cutionContext16expectationValueE) | -   [cudaq::QPUState (C++         |
| -   [cudaq::Execu                 |     class)](api/languages/cpp_    |
| tionContext::explicitMeasurements | api.html#_CPPv4N5cudaq8QPUStateE) |
|     (C++                          | -   [cudaq::qreg (C++             |
|     member)](api/languages/cp     |     class)](api/lang              |
| p_api.html#_CPPv4N5cudaq16Executi | uages/cpp_api.html#_CPPv4I_NSt6si |
| onContext20explicitMeasurementsE) | ze_tE_NSt6size_tE0EN5cudaq4qregE) |
| -   [cuda                         | -   [cudaq::qreg::back (C++       |
| q::ExecutionContext::futureResult |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/lang             | v4N5cudaq4qreg4backENSt6size_tE), |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     [\[1\]](api/languages/cpp_ap  |
| 6ExecutionContext12futureResultE) | i.html#_CPPv4N5cudaq4qreg4backEv) |
| -   [cudaq::ExecutionContext      | -   [cudaq::qreg::begin (C++      |
| ::hasConditionalsOnMeasureResults |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     mem                           | .html#_CPPv4N5cudaq4qreg5beginEv) |
| ber)](api/languages/cpp_api.html# | -   [cudaq::qreg::clear (C++      |
| _CPPv4N5cudaq16ExecutionContext31 |                                   |
| hasConditionalsOnMeasureResultsE) |  function)](api/languages/cpp_api |
| -   [cudaq::Executi               | .html#_CPPv4N5cudaq4qreg5clearEv) |
| onContext::invocationResultBuffer | -   [cudaq::qreg::front (C++      |
|     (C++                          |     function)]                    |
|     member)](api/languages/cpp_   | (api/languages/cpp_api.html#_CPPv |
| api.html#_CPPv4N5cudaq16Execution | 4N5cudaq4qreg5frontENSt6size_tE), |
| Context22invocationResultBufferE) |     [\[1\]](api/languages/cpp_api |
| -   [cu                           | .html#_CPPv4N5cudaq4qreg5frontEv) |
| daq::ExecutionContext::kernelName | -   [cudaq::qreg::operator\[\]    |
|     (C++                          |     (C++                          |
|     member)](api/la               |     functi                        |
| nguages/cpp_api.html#_CPPv4N5cuda | on)](api/languages/cpp_api.html#_ |
| q16ExecutionContext10kernelNameE) | CPPv4N5cudaq4qregixEKNSt6size_tE) |
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
| -   [cudaq                        |    [cudaq::QuakeValue::operator\* |
| ::ExecutionContext::registerNames |     (C++                          |
|     (C++                          |     function)](api                |
|     member)](api/langu            | /languages/cpp_api.html#_CPPv4N5c |
| ages/cpp_api.html#_CPPv4N5cudaq16 | udaq10QuakeValuemlE10QuakeValue), |
| ExecutionContext13registerNamesE) |                                   |
| -   [cu                           | [\[1\]](api/languages/cpp_api.htm |
| daq::ExecutionContext::reorderIdx | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|     (C++                          | -   [cudaq::QuakeValue::operator+ |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api                |
| q16ExecutionContext10reorderIdxE) | /languages/cpp_api.html#_CPPv4N5c |
| -                                 | udaq10QuakeValueplE10QuakeValue), |
|  [cudaq::ExecutionContext::result |     [                             |
|     (C++                          | \[1\]](api/languages/cpp_api.html |
|     member)](a                    | #_CPPv4N5cudaq10QuakeValueplEKd), |
| pi/languages/cpp_api.html#_CPPv4N |                                   |
| 5cudaq16ExecutionContext6resultE) | [\[2\]](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|   [cudaq::ExecutionContext::shots | -   [cudaq::QuakeValue::operator- |
|     (C++                          |     (C++                          |
|     member)](                     |     function)](api                |
| api/languages/cpp_api.html#_CPPv4 | /languages/cpp_api.html#_CPPv4N5c |
| N5cudaq16ExecutionContext5shotsE) | udaq10QuakeValuemiE10QuakeValue), |
| -   [cudaq::                      |     [                             |
| ExecutionContext::simulationState | \[1\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     member)](api/languag          |     [                             |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | \[2\]](api/languages/cpp_api.html |
| ecutionContext15simulationStateE) | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| -                                 |                                   |
|    [cudaq::ExecutionContext::spin | [\[3\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     member)]                      | -   [cudaq::QuakeValue::operator/ |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq16ExecutionContext4spinE) |     function)](api                |
| -   [cudaq::                      | /languages/cpp_api.html#_CPPv4N5c |
| ExecutionContext::totalIterations | udaq10QuakeValuedvE10QuakeValue), |
|     (C++                          |                                   |
|     member)](api/languag          | [\[1\]](api/languages/cpp_api.htm |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| ecutionContext15totalIterationsE) | -                                 |
| -   [cudaq::ExecutionResult (C++  |  [cudaq::QuakeValue::operator\[\] |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |     function)](api                |
| #_CPPv4N5cudaq15ExecutionResultE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cud                          | udaq10QuakeValueixEKNSt6size_tE), |
| aq::ExecutionResult::appendResult |     [\[1\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     functio                       | daq10QuakeValueixERK10QuakeValue) |
| n)](api/languages/cpp_api.html#_C | -                                 |
| PPv4N5cudaq15ExecutionResult12app |    [cudaq::QuakeValue::QuakeValue |
| endResultENSt6stringENSt6size_tE) |     (C++                          |
| -   [cu                           |     function)](api/languag        |
| daq::ExecutionResult::deserialize | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     (C++                          | akeValue10QuakeValueERN4mlir20Imp |
|     function)                     | licitLocOpBuilderEN4mlir5ValueE), |
| ](api/languages/cpp_api.html#_CPP |     [\[1\]                        |
| v4N5cudaq15ExecutionResult11deser | ](api/languages/cpp_api.html#_CPP |
| ializeERNSt6vectorINSt6size_tEEE) | v4N5cudaq10QuakeValue10QuakeValue |
| -   [cudaq:                       | ERN4mlir20ImplicitLocOpBuilderEd) |
| :ExecutionResult::ExecutionResult | -   [cudaq::QuakeValue::size (C++ |
|     (C++                          |     funct                         |
|     functio                       | ion)](api/languages/cpp_api.html# |
| n)](api/languages/cpp_api.html#_C | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| PPv4N5cudaq15ExecutionResult15Exe | -   [cudaq::QuakeValue::slice     |
| cutionResultE16CountsDictionary), |     (C++                          |
|     [\[1\]](api/lan               |     function)](api/languages/cpp_ |
| guages/cpp_api.html#_CPPv4N5cudaq | api.html#_CPPv4N5cudaq10QuakeValu |
| 15ExecutionResult15ExecutionResul | e5sliceEKNSt6size_tEKNSt6size_tE) |
| tE16CountsDictionaryNSt6stringE), | -   [cudaq::quantum_platform (C++ |
|     [\[2\                         |     cl                            |
| ]](api/languages/cpp_api.html#_CP | ass)](api/languages/cpp_api.html# |
| Pv4N5cudaq15ExecutionResult15Exec | _CPPv4N5cudaq16quantum_platformE) |
| utionResultE16CountsDictionaryd), | -   [cud                          |
|                                   | aq::quantum_platform::clear_shots |
|    [\[3\]](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4N5cudaq15ExecutionResu |     function)](api/lang           |
| lt15ExecutionResultENSt6stringE), | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     [\[4\                         | 6quantum_platform11clear_shotsEv) |
| ]](api/languages/cpp_api.html#_CP | -   [cuda                         |
| Pv4N5cudaq15ExecutionResult15Exec | q::quantum_platform::connectivity |
| utionResultERK15ExecutionResult), |     (C++                          |
|     [\[5\]](api/language          |     function)](api/langu          |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | ages/cpp_api.html#_CPPv4N5cudaq16 |
| cutionResult15ExecutionResultEd), | quantum_platform12connectivityEv) |
|     [\[6\]](api/languag           | -   [cudaq::q                     |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | uantum_platform::enqueueAsyncTask |
| ecutionResult15ExecutionResultEv) |     (C++                          |
| -   [                             |     function)](api/languages/     |
| cudaq::ExecutionResult::operator= | cpp_api.html#_CPPv4N5cudaq16quant |
|     (C++                          | um_platform16enqueueAsyncTaskEKNS |
|     function)](api/languages/     | t6size_tER19KernelExecutionTask), |
| cpp_api.html#_CPPv4N5cudaq15Execu |     [\[1\]](api/languag           |
| tionResultaSERK15ExecutionResult) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [c                            | antum_platform16enqueueAsyncTaskE |
| udaq::ExecutionResult::operator== | KNSt6size_tERNSt8functionIFvvEEE) |
|     (C++                          | -   [cudaq::qua                   |
|     function)](api/languages/c    | ntum_platform::get_codegen_config |
| pp_api.html#_CPPv4NK5cudaq15Execu |     (C++                          |
| tionResulteqERK15ExecutionResult) |     function)](api/languages/c    |
| -   [cud                          | pp_api.html#_CPPv4N5cudaq16quantu |
| aq::ExecutionResult::registerName | m_platform18get_codegen_configEv) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/lan              | quantum_platform::get_current_qpu |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult12registerNameE) |     function)](api/language       |
| -   [cudaq                        | s/cpp_api.html#_CPPv4N5cudaq16qua |
| ::ExecutionResult::sequentialData | ntum_platform15get_current_qpuEv) |
|     (C++                          | -   [cuda                         |
|     member)](api/langu            | q::quantum_platform::get_exec_ctx |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| ExecutionResult14sequentialDataE) |     function)](api/langua         |
| -   [                             | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| cudaq::ExecutionResult::serialize | quantum_platform12get_exec_ctxEv) |
|     (C++                          | -   [c                            |
|     function)](api/l              | udaq::quantum_platform::get_noise |
| anguages/cpp_api.html#_CPPv4NK5cu |     (C++                          |
| daq15ExecutionResult9serializeEv) |     function)](api/l              |
| -   [cudaq::fermion_handler (C++  | anguages/cpp_api.html#_CPPv4N5cud |
|     c                             | aq16quantum_platform9get_noiseEv) |
| lass)](api/languages/cpp_api.html | -   [cudaq:                       |
| #_CPPv4N5cudaq15fermion_handlerE) | :quantum_platform::get_num_qubits |
| -   [cudaq::fermion_op (C++       |     (C++                          |
|     type)](api/languages/cpp_api  |                                   |
| .html#_CPPv4N5cudaq10fermion_opE) | function)](api/languages/cpp_api. |
| -   [cudaq::fermion_op_term (C++  | html#_CPPv4N5cudaq16quantum_platf |
|                                   | orm14get_num_qubitsENSt6size_tE), |
| type)](api/languages/cpp_api.html |     [\[1\]](api/languag           |
| #_CPPv4N5cudaq15fermion_op_termE) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cudaq::FermioniqBaseQPU (C++ | antum_platform14get_num_qubitsEv) |
|     cl                            | -   [cudaq::quantum_              |
| ass)](api/languages/cpp_api.html# | platform::get_remote_capabilities |
| _CPPv4N5cudaq16FermioniqBaseQPUE) |     (C++                          |
| -   [cudaq::get_state (C++        |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|    function)](api/languages/cpp_a | 4NK5cudaq16quantum_platform23get_ |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | remote_capabilitiesEKNSt6size_tE) |
| ateEDaRR13QuantumKernelDpRR4Args) | -   [cudaq::qua                   |
| -   [cudaq::gradient (C++         | ntum_platform::get_runtime_target |
|     class)](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4N5cudaq8gradientE) |     function)](api/languages/cp   |
| -   [cudaq::gradient::clone (C++  | p_api.html#_CPPv4NK5cudaq16quantu |
|     fun                           | m_platform18get_runtime_targetEv) |
| ction)](api/languages/cpp_api.htm | -   [c                            |
| l#_CPPv4N5cudaq8gradient5cloneEv) | udaq::quantum_platform::get_shots |
| -   [cudaq::gradient::compute     |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)](api/language       | anguages/cpp_api.html#_CPPv4N5cud |
| s/cpp_api.html#_CPPv4N5cudaq8grad | aq16quantum_platform9get_shotsEv) |
| ient7computeERKNSt6vectorIdEERKNS | -   [cuda                         |
| t8functionIFdNSt6vectorIdEEEEEd), | q::quantum_platform::getLogStream |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/langu          |
| cudaq8gradient7computeERKNSt6vect | ages/cpp_api.html#_CPPv4N5cudaq16 |
| orIdEERNSt6vectorIdEERK7spin_opd) | quantum_platform12getLogStreamEv) |
| -   [cudaq::gradient::gradient    | -   [cud                          |
|     (C++                          | aq::quantum_platform::is_emulated |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4I00EN5cu |                                   |
| daq8gradient8gradientER7KernelT), |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq16quantum_pl |
|    [\[1\]](api/languages/cpp_api. | atform11is_emulatedEKNSt6size_tE) |
| html#_CPPv4I00EN5cudaq8gradient8g | -   [c                            |
| radientER7KernelTRR10ArgsMapper), | udaq::quantum_platform::is_remote |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     function)](api/languages/cp   |
| Pv4I00EN5cudaq8gradient8gradientE | p_api.html#_CPPv4N5cudaq16quantum |
| RR13QuantumKernelRR10ArgsMapper), | _platform9is_remoteEKNSt6size_tE) |
|     [\[3                          | -   [cuda                         |
| \]](api/languages/cpp_api.html#_C | q::quantum_platform::is_simulator |
| PPv4N5cudaq8gradient8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |                                   |
|     [\[                           |  function)](api/languages/cpp_api |
| 4\]](api/languages/cpp_api.html#_ | .html#_CPPv4NK5cudaq16quantum_pla |
| CPPv4N5cudaq8gradient8gradientEv) | tform12is_simulatorEKNSt6size_tE) |
| -   [cudaq::gradient::setArgs     | -   [c                            |
|     (C++                          | udaq::quantum_platform::launchVQE |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |                                   |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | function)](api/languages/cpp_api. |
| tArgsEvR13QuantumKernelDpRR4Args) | html#_CPPv4N5cudaq16quantum_platf |
| -   [cudaq::gradient::setKernel   | orm9launchVQEEKNSt6stringEPKvPN5c |
|     (C++                          | udaq8gradientERKN5cudaq7spin_opER |
|     function)](api/languages/c    | N5cudaq9optimizerEKiKNSt6size_tE) |
| pp_api.html#_CPPv4I0EN5cudaq8grad | -   [cudaq:                       |
| ient9setKernelEvR13QuantumKernel) | :quantum_platform::list_platforms |
| -   [cud                          |     (C++                          |
| aq::gradients::central_difference |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     class)](api/la                | antum_platform14list_platformsEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -                                 |
| q9gradients18central_differenceE) |    [cudaq::quantum_platform::name |
| -   [cudaq::gra                   |     (C++                          |
| dients::central_difference::clone |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function)](api/languages      | K5cudaq16quantum_platform4nameEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [                             |
| ents18central_difference5cloneEv) | cudaq::quantum_platform::num_qpus |
| -   [cudaq::gradi                 |     (C++                          |
| ents::central_difference::compute |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     function)](                   | daq16quantum_platform8num_qpusEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::                      |
| N5cudaq9gradients18central_differ | quantum_platform::onRandomSeedSet |
| ence7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |                                   |
|                                   | function)](api/languages/cpp_api. |
|   [\[1\]](api/languages/cpp_api.h | html#_CPPv4N5cudaq16quantum_platf |
| tml#_CPPv4N5cudaq9gradients18cent | orm15onRandomSeedSetENSt6size_tE) |
| ral_difference7computeERKNSt6vect | -   [cudaq:                       |
| orIdEERNSt6vectorIdEERK7spin_opd) | :quantum_platform::reset_exec_ctx |
| -   [cudaq::gradie                |     (C++                          |
| nts::central_difference::gradient |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     functio                       | .html#_CPPv4N5cudaq16quantum_plat |
| n)](api/languages/cpp_api.html#_C | form14reset_exec_ctxENSt6size_tE) |
| PPv4I00EN5cudaq9gradients18centra | -   [cud                          |
| l_difference8gradientER7KernelT), | aq::quantum_platform::reset_noise |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     function)](api/lang           |
| q9gradients18central_difference8g | uages/cpp_api.html#_CPPv4N5cudaq1 |
| radientER7KernelTRR10ArgsMapper), | 6quantum_platform11reset_noiseEv) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq:                       |
| api.html#_CPPv4I00EN5cudaq9gradie | :quantum_platform::resetLogStream |
| nts18central_difference8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/languag        |
|     [\[3\]](api/languages/cpp     | es/cpp_api.html#_CPPv4N5cudaq16qu |
| _api.html#_CPPv4N5cudaq9gradients | antum_platform14resetLogStreamEv) |
| 18central_difference8gradientERRN | -   [cudaq::                      |
| St8functionIFvNSt6vectorIdEEEEE), | quantum_platform::set_current_qpu |
|     [\[4\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq9gradient |     f                             |
| s18central_difference8gradientEv) | unction)](api/languages/cpp_api.h |
| -   [cud                          | tml#_CPPv4N5cudaq16quantum_platfo |
| aq::gradients::forward_difference | rm15set_current_qpuEKNSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     class)](api/la                | q::quantum_platform::set_exec_ctx |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9gradients18forward_differenceE) |     function)](api/languages      |
| -   [cudaq::gra                   | /cpp_api.html#_CPPv4N5cudaq16quan |
| dients::forward_difference::clone | tum_platform12set_exec_ctxEPN5cud |
|     (C++                          | aq16ExecutionContextENSt6size_tE) |
|     function)](api/languages      | -   [c                            |
| /cpp_api.html#_CPPv4N5cudaq9gradi | udaq::quantum_platform::set_noise |
| ents18forward_difference5cloneEv) |     (C++                          |
| -   [cudaq::gradi                 |                                   |
| ents::forward_difference::compute |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4N5cudaq16quantum_pl |
|     function)](                   | atform9set_noiseEPK11noise_model) |
| api/languages/cpp_api.html#_CPPv4 | -   [c                            |
| N5cudaq9gradients18forward_differ | udaq::quantum_platform::set_shots |
| ence7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api/l              |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|   [\[1\]](api/languages/cpp_api.h | aq16quantum_platform9set_shotsEi) |
| tml#_CPPv4N5cudaq9gradients18forw | -   [cuda                         |
| ard_difference7computeERKNSt6vect | q::quantum_platform::setLogStream |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradie                |                                   |
| nts::forward_difference::gradient |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq16quantum_plat |
|     functio                       | form12setLogStreamERNSt7ostreamE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::q                     |
| PPv4I00EN5cudaq9gradients18forwar | uantum_platform::setTargetBackend |
| d_difference8gradientER7KernelT), |     (C++                          |
|     [\[1\]](api/langua            |     fun                           |
| ges/cpp_api.html#_CPPv4I00EN5cuda | ction)](api/languages/cpp_api.htm |
| q9gradients18forward_difference8g | l#_CPPv4N5cudaq16quantum_platform |
| radientER7KernelTRR10ArgsMapper), | 16setTargetBackendERKNSt6stringE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::quantum_platfo        |
| api.html#_CPPv4I00EN5cudaq9gradie | rm::supports_conditional_feedback |
| nts18forward_difference8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/l              |
|     [\[3\]](api/languages/cpp     | anguages/cpp_api.html#_CPPv4NK5cu |
| _api.html#_CPPv4N5cudaq9gradients | daq16quantum_platform29supports_c |
| 18forward_difference8gradientERRN | onditional_feedbackEKNSt6size_tE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::quantum_platfor       |
|     [\[4\]](api/languages/cp      | m::supports_explicit_measurements |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18forward_difference8gradientEv) |     function)](api/la             |
| -   [                             | nguages/cpp_api.html#_CPPv4NK5cud |
| cudaq::gradients::parameter_shift | aq16quantum_platform30supports_ex |
|     (C++                          | plicit_measurementsEKNSt6size_tE) |
|     class)](api                   | -   [cudaq::quantum_pla           |
| /languages/cpp_api.html#_CPPv4N5c | tform::supports_task_distribution |
| udaq9gradients15parameter_shiftE) |     (C++                          |
| -   [cudaq::                      |     fu                            |
| gradients::parameter_shift::clone | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq16quantum_platfo |
|     function)](api/langua         | rm26supports_task_distributionEv) |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | -   [cudaq::QuantumTask (C++      |
| adients15parameter_shift5cloneEv) |     type)](api/languages/cpp_api. |
| -   [cudaq::gr                    | html#_CPPv4N5cudaq11QuantumTaskE) |
| adients::parameter_shift::compute | -   [cudaq::qubit (C++            |
|     (C++                          |     type)](api/languages/c        |
|     function                      | pp_api.html#_CPPv4N5cudaq5qubitE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::QubitConnectivity     |
| Pv4N5cudaq9gradients15parameter_s |     (C++                          |
| hift7computeERKNSt6vectorIdEERKNS |     ty                            |
| t8functionIFdNSt6vectorIdEEEEEd), | pe)](api/languages/cpp_api.html#_ |
|     [\[1\]](api/languages/cpp_ap  | CPPv4N5cudaq17QubitConnectivityE) |
| i.html#_CPPv4N5cudaq9gradients15p | -   [cudaq::QubitEdge (C++        |
| arameter_shift7computeERKNSt6vect |     type)](api/languages/cpp_a    |
| orIdEERNSt6vectorIdEERK7spin_opd) | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| -   [cudaq::gra                   | -   [cudaq::qudit (C++            |
| dients::parameter_shift::gradient |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     func                          | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| tion)](api/languages/cpp_api.html | -   [cudaq::qudit::qudit (C++     |
| #_CPPv4I00EN5cudaq9gradients15par |                                   |
| ameter_shift8gradientER7KernelT), | function)](api/languages/cpp_api. |
|     [\[1\]](api/lan               | html#_CPPv4N5cudaq5qudit5quditEv) |
| guages/cpp_api.html#_CPPv4I00EN5c | -   [cudaq::qvector (C++          |
| udaq9gradients15parameter_shift8g |     class)                        |
| radientER7KernelTRR10ArgsMapper), | ](api/languages/cpp_api.html#_CPP |
|     [\[2\]](api/languages/c       | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| pp_api.html#_CPPv4I00EN5cudaq9gra | -   [cudaq::qvector::back (C++    |
| dients15parameter_shift8gradientE |     function)](a                  |
| RR13QuantumKernelRR10ArgsMapper), | pi/languages/cpp_api.html#_CPPv4N |
|     [\[3\]](api/languages/        | 5cudaq7qvector4backENSt6size_tE), |
| cpp_api.html#_CPPv4N5cudaq9gradie |                                   |
| nts15parameter_shift8gradientERRN |   [\[1\]](api/languages/cpp_api.h |
| St8functionIFvNSt6vectorIdEEEEE), | tml#_CPPv4N5cudaq7qvector4backEv) |
|     [\[4\]](api/languages         | -   [cudaq::qvector::begin (C++   |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     fu                            |
| ents15parameter_shift8gradientEv) | nction)](api/languages/cpp_api.ht |
| -   [cudaq::kernel_builder (C++   | ml#_CPPv4N5cudaq7qvector5beginEv) |
|     clas                          | -   [cudaq::qvector::clear (C++   |
| s)](api/languages/cpp_api.html#_C |     fu                            |
| PPv4IDpEN5cudaq14kernel_builderE) | nction)](api/languages/cpp_api.ht |
| -   [c                            | ml#_CPPv4N5cudaq7qvector5clearEv) |
| udaq::kernel_builder::constantVal | -   [cudaq::qvector::end (C++     |
|     (C++                          |                                   |
|     function)](api/la             | function)](api/languages/cpp_api. |
| nguages/cpp_api.html#_CPPv4N5cuda | html#_CPPv4N5cudaq7qvector3endEv) |
| q14kernel_builder11constantValEd) | -   [cudaq::qvector::front (C++   |
| -   [cu                           |     function)](ap                 |
| daq::kernel_builder::getArguments | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq7qvector5frontENSt6size_tE), |
|     function)](api/lan            |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq |  [\[1\]](api/languages/cpp_api.ht |
| 14kernel_builder12getArgumentsEv) | ml#_CPPv4N5cudaq7qvector5frontEv) |
| -   [cu                           | -   [cudaq::qvector::operator=    |
| daq::kernel_builder::getNumParams |     (C++                          |
|     (C++                          |     functio                       |
|     function)](api/lan            | n)](api/languages/cpp_api.html#_C |
| guages/cpp_api.html#_CPPv4N5cudaq | PPv4N5cudaq7qvectoraSERK7qvector) |
| 14kernel_builder12getNumParamsEv) | -   [cudaq::qvector::operator\[\] |
| -   [c                            |     (C++                          |
| udaq::kernel_builder::isArgStdVec |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/languages/cp   | v4N5cudaq7qvectorixEKNSt6size_tE) |
| p_api.html#_CPPv4N5cudaq14kernel_ | -   [cudaq::qvector::qvector (C++ |
| builder11isArgStdVecENSt6size_tE) |     function)](api/               |
| -   [cuda                         | languages/cpp_api.html#_CPPv4N5cu |
| q::kernel_builder::kernel_builder | daq7qvector7qvectorENSt6size_tE), |
|     (C++                          |     [\[1\]](a                     |
|     function)](api/languages/cpp_ | pi/languages/cpp_api.html#_CPPv4N |
| api.html#_CPPv4N5cudaq14kernel_bu | 5cudaq7qvector7qvectorERK5state), |
| ilder14kernel_builderERNSt6vector |     [\[2\]](api                   |
| IN7details17KernelBuilderTypeEEE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::kernel_builder::name  | udaq7qvector7qvectorERK7qvector), |
|     (C++                          |     [\[3\]](api/languages/cpp     |
|     function)                     | _api.html#_CPPv4N5cudaq7qvector7q |
| ](api/languages/cpp_api.html#_CPP | vectorERKNSt6vectorI7complexEEb), |
| v4N5cudaq14kernel_builder4nameEv) |     [\[4\]](ap                    |
| -                                 | i/languages/cpp_api.html#_CPPv4N5 |
|    [cudaq::kernel_builder::qalloc | cudaq7qvector7qvectorERR7qvector) |
|     (C++                          | -   [cudaq::qvector::size (C++    |
|     function)](api/language       |     fu                            |
| s/cpp_api.html#_CPPv4N5cudaq14ker | nction)](api/languages/cpp_api.ht |
| nel_builder6qallocE10QuakeValue), | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     [\[1\]](api/language          | -   [cudaq::qvector::slice (C++   |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     function)](api/language       |
| nel_builder6qallocEKNSt6size_tE), | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|     [\[2                          | tor5sliceENSt6size_tENSt6size_tE) |
| \]](api/languages/cpp_api.html#_C | -   [cudaq::qvector::value_type   |
| PPv4N5cudaq14kernel_builder6qallo |     (C++                          |
| cERNSt6vectorINSt7complexIdEEEE), |     typ                           |
|     [\[3\]](                      | e)](api/languages/cpp_api.html#_C |
| api/languages/cpp_api.html#_CPPv4 | PPv4N5cudaq7qvector10value_typeE) |
| N5cudaq14kernel_builder6qallocEv) | -   [cudaq::qview (C++            |
| -   [cudaq::kernel_builder::swap  |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     function)](api/language       | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | -   [cudaq::qview::value_type     |
| 4kernel_builder4swapEvRK10QuakeVa |     (C++                          |
| lueRK10QuakeValueRK10QuakeValue), |     t                             |
|                                   | ype)](api/languages/cpp_api.html# |
| [\[1\]](api/languages/cpp_api.htm | _CPPv4N5cudaq5qview10value_typeE) |
| l#_CPPv4I00EN5cudaq14kernel_build | -   [cudaq::range (C++            |
| er4swapEvRKNSt6vectorI10QuakeValu |     func                          |
| eEERK10QuakeValueRK10QuakeValue), | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4I00EN5cudaq5rangeENSt6vect |
| [\[2\]](api/languages/cpp_api.htm | orI11ElementTypeEE11ElementType), |
| l#_CPPv4N5cudaq14kernel_builder4s |     [\[1\]](api/languages/cpp_a   |
| wapERK10QuakeValueRK10QuakeValue) | pi.html#_CPPv4I00EN5cudaq5rangeEN |
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
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::Remot                 |
| K5cudaq13kraus_channel7get_opsEv) | eCapabilities::serializedCodeExec |
| -   [cudaq::                      |     (C++                          |
| kraus_channel::is_unitary_mixture |     member)](api/languages/cp     |
|     (C++                          | p_api.html#_CPPv4N5cudaq18RemoteC |
|     function)](api/languages      | apabilities18serializedCodeExecE) |
| /cpp_api.html#_CPPv4NK5cudaq13kra | -   [cudaq:                       |
| us_channel18is_unitary_mixtureEv) | :RemoteCapabilities::stateOverlap |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::kraus_channel |     member)](api/langua           |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq18R |
|     function)](api/lang           | emoteCapabilities12stateOverlapE) |
| uages/cpp_api.html#_CPPv4IDpEN5cu | -                                 |
| daq13kraus_channel13kraus_channel |   [cudaq::RemoteCapabilities::vqe |
| EDpRRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     member)](                     |
|  [\[1\]](api/languages/cpp_api.ht | api/languages/cpp_api.html#_CPPv4 |
| ml#_CPPv4N5cudaq13kraus_channel13 | N5cudaq18RemoteCapabilities3vqeE) |
| kraus_channelERK13kraus_channel), | -   [cudaq::RemoteSimulationState |
|     [\[2\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     class)]                       |
| v4N5cudaq13kraus_channel13kraus_c | (api/languages/cpp_api.html#_CPPv |
| hannelERKNSt6vectorI8kraus_opEE), | 4N5cudaq21RemoteSimulationStateE) |
|     [\[3\]                        | -   [cudaq::Resources (C++        |
| ](api/languages/cpp_api.html#_CPP |     class)](api/languages/cpp_a   |
| v4N5cudaq13kraus_channel13kraus_c | pi.html#_CPPv4N5cudaq9ResourcesE) |
| hannelERRNSt6vectorI8kraus_opEE), | -   [cudaq::run (C++              |
|     [\[4\]](api/lan               |     function)]                    |
| guages/cpp_api.html#_CPPv4N5cudaq | (api/languages/cpp_api.html#_CPPv |
| 13kraus_channel13kraus_channelEv) | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| -                                 | 5invoke_result_tINSt7decay_tI13Qu |
| [cudaq::kraus_channel::noise_type | antumKernelEEDpNSt7decay_tI4ARGSE |
|     (C++                          | EEEEENSt6size_tERN5cudaq11noise_m |
|     member)](api                  | odelERR13QuantumKernelDpRR4ARGS), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[1\]](api/langu             |
| udaq13kraus_channel10noise_typeE) | ages/cpp_api.html#_CPPv4I0DpEN5cu |
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
| -   [cu                           | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| daq::kraus_channel::probabilities | ize_tERR13QuantumKernelDpRR4ARGS) |
|     (C++                          | -   [cudaq::RuntimeTarget (C++    |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda | struct)](api/languages/cpp_api.ht |
| q13kraus_channel13probabilitiesE) | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| -                                 | -   [cudaq::sample (C++           |
|  [cudaq::kraus_channel::push_back |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4I0Dp0EN5cudaq6sa |
|     function)](api/langua         | mpleE13sample_resultRK14sample_op |
| ges/cpp_api.html#_CPPv4N5cudaq13k | tionsRR13QuantumKernelDpRR4Args), |
| raus_channel9push_backE8kraus_op) |     [\[1\]                        |
| -   [cudaq::kraus_channel::size   | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4I0Dp0EN5cudaq6sampleE13sample_r |
|     function)                     | esultRR13QuantumKernelDpRR4Args), |
| ](api/languages/cpp_api.html#_CPP |     [\[                           |
| v4NK5cudaq13kraus_channel4sizeEv) | 2\]](api/languages/cpp_api.html#_ |
| -   [                             | CPPv4I0Dp0EN5cudaq6sampleEDaNSt6s |
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
| on)](api/languages/cpp_api.html#_ |     function)](api/languages/cpp  |
| CPPv4NK5cudaq8kraus_op7adjointEv) | _api.html#_CPPv4N5cudaq13sample_r |
| -   [cudaq::kraus_op::data (C++   | esult6appendER15ExecutionResultb) |
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
| ize_tEEERK20commutation_behavior) | ample_resultaSER13sample_result), |
| -   [cuda                         |     [\[1\]](api/langua            |
| q::matrix_handler::matrix_handler | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     (C++                          | ample_resultaSERR13sample_result) |
|     function)](api/languag        | -                                 |
| es/cpp_api.html#_CPPv4I0_NSt11ena | [cudaq::sample_result::operator== |
| ble_if_tINSt12is_base_of_vI16oper |     (C++                          |
| ator_handler1TEEbEEEN5cudaq14matr |     function)](api/languag        |
| ix_handler14matrix_handlerERK1T), | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     [\[1\]](ap                    | ample_resulteqERK13sample_result) |
| i/languages/cpp_api.html#_CPPv4I0 | -   [                             |
| _NSt11enable_if_tINSt12is_base_of | cudaq::sample_result::probability |
| _vI16operator_handler1TEEbEEEN5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     function)](api/lan            |
| erERK1TRK20commutation_behavior), | guages/cpp_api.html#_CPPv4NK5cuda |
|     [\[2\]](api/languages/cpp_ap  | q13sample_result11probabilityENSt |
| i.html#_CPPv4N5cudaq14matrix_hand | 11string_viewEKNSt11string_viewE) |
| ler14matrix_handlerENSt6size_tE), | -   [cud                          |
|     [\[3\]](api/                  | aq::sample_result::register_names |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     function)](api/langu          |
| erENSt6stringERKNSt6vectorINSt6si | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| ze_tEEERK20commutation_behavior), | 3sample_result14register_namesEv) |
|     [\[4\]](api/                  | -                                 |
| languages/cpp_api.html#_CPPv4N5cu |    [cudaq::sample_result::reorder |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erENSt6stringERRNSt6vectorINSt6si |     function)](api/langua         |
| ze_tEEERK20commutation_behavior), | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     [\                            | ample_result7reorderERKNSt6vector |
| [5\]](api/languages/cpp_api.html# | INSt6size_tEEEKNSt11string_viewE) |
| _CPPv4N5cudaq14matrix_handler14ma | -   [cu                           |
| trix_handlerERK14matrix_handler), | daq::sample_result::sample_result |
|     [                             |     (C++                          |
| \[6\]](api/languages/cpp_api.html |     fun                           |
| #_CPPv4N5cudaq14matrix_handler14m | ction)](api/languages/cpp_api.htm |
| atrix_handlerERR14matrix_handler) | l#_CPPv4N5cudaq13sample_result13s |
| -                                 | ample_resultER15ExecutionResult), |
|  [cudaq::matrix_handler::momentum |                                   |
|     (C++                          |  [\[1\]](api/languages/cpp_api.ht |
|     function)](api/language       | ml#_CPPv4N5cudaq13sample_result13 |
| s/cpp_api.html#_CPPv4N5cudaq14mat | sample_resultERK13sample_result), |
| rix_handler8momentumENSt6size_tE) |     [\[2\]](api/l                 |
| -                                 | anguages/cpp_api.html#_CPPv4N5cud |
|    [cudaq::matrix_handler::number | aq13sample_result13sample_resultE |
|     (C++                          | RNSt6vectorI15ExecutionResultEE), |
|     function)](api/langua         |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq14m |  [\[3\]](api/languages/cpp_api.ht |
| atrix_handler6numberENSt6size_tE) | ml#_CPPv4N5cudaq13sample_result13 |
| -                                 | sample_resultERR13sample_result), |
| [cudaq::matrix_handler::operator= |     [                             |
|     (C++                          | \[4\]](api/languages/cpp_api.html |
|     fun                           | #_CPPv4N5cudaq13sample_result13sa |
| ction)](api/languages/cpp_api.htm | mple_resultERR15ExecutionResult), |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     [\[5\]](api/la                |
| NSt7is_sameI1T14matrix_handlerE5v | nguages/cpp_api.html#_CPPv4N5cuda |
| alueENSt12is_base_of_vI16operator | q13sample_result13sample_resultEd |
| _handler1TEEEbEEEN5cudaq14matrix_ | RNSt6vectorI15ExecutionResultEE), |
| handleraSER14matrix_handlerRK1T), |     [\[6\]](api/lan               |
|     [\[1\]](api/languages         | guages/cpp_api.html#_CPPv4N5cudaq |
| /cpp_api.html#_CPPv4N5cudaq14matr | 13sample_result13sample_resultEv) |
| ix_handleraSERK14matrix_handler), | -                                 |
|     [\[2\]](api/language          |  [cudaq::sample_result::serialize |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handleraSERR14matrix_handler) |     function)](api                |
| -   [                             | /languages/cpp_api.html#_CPPv4NK5 |
| cudaq::matrix_handler::operator== | cudaq13sample_result9serializeEv) |
|     (C++                          | -   [cudaq::sample_result::size   |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4NK5cudaq14mat |     function)](api/languages/c    |
| rix_handlereqERK14matrix_handler) | pp_api.html#_CPPv4NK5cudaq13sampl |
| -                                 | e_result4sizeEKNSt11string_viewE) |
|    [cudaq::matrix_handler::parity | -   [cudaq::sample_result::to_map |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/languages/cpp  |
| ges/cpp_api.html#_CPPv4N5cudaq14m | _api.html#_CPPv4NK5cudaq13sample_ |
| atrix_handler6parityENSt6size_tE) | result6to_mapEKNSt11string_viewE) |
| -                                 | -   [cuda                         |
|  [cudaq::matrix_handler::position | q::sample_result::\~sample_result |
|     (C++                          |     (C++                          |
|     function)](api/language       |     funct                         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ion)](api/languages/cpp_api.html# |
| rix_handler8positionENSt6size_tE) | _CPPv4N5cudaq13sample_resultD0Ev) |
| -   [cudaq::                      | -   [cudaq::scalar_callback (C++  |
| matrix_handler::remove_definition |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|     fu                            | #_CPPv4N5cudaq15scalar_callbackE) |
| nction)](api/languages/cpp_api.ht | -   [c                            |
| ml#_CPPv4N5cudaq14matrix_handler1 | udaq::scalar_callback::operator() |
| 7remove_definitionERKNSt6stringE) |     (C++                          |
| -                                 |     function)](api/language       |
|   [cudaq::matrix_handler::squeeze | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|     (C++                          | alar_callbackclERKNSt13unordered_ |
|     function)](api/languag        | mapINSt6stringENSt7complexIdEEEE) |
| es/cpp_api.html#_CPPv4N5cudaq14ma | -   [                             |
| trix_handler7squeezeENSt6size_tE) | cudaq::scalar_callback::operator= |
| -   [cudaq::m                     |     (C++                          |
| atrix_handler::to_diagonal_matrix |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function)](api/lang           | _callbackaSERK15scalar_callback), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[1\]](api/languages/        |
| 14matrix_handler18to_diagonal_mat | cpp_api.html#_CPPv4N5cudaq15scala |
| rixERNSt13unordered_mapINSt6size_ | r_callbackaSERR15scalar_callback) |
| tENSt7int64_tEEERKNSt13unordered_ | -   [cudaq:                       |
| mapINSt6stringENSt7complexIdEEEE) | :scalar_callback::scalar_callback |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::to_matrix |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4I0_NSt11ena |
|     function)                     | ble_if_tINSt16is_invocable_r_vINS |
| ](api/languages/cpp_api.html#_CPP | t7complexIdEE8CallableRKNSt13unor |
| v4NK5cudaq14matrix_handler9to_mat | dered_mapINSt6stringENSt7complexI |
| rixERNSt13unordered_mapINSt6size_ | dEEEEEEbEEEN5cudaq15scalar_callba |
| tENSt7int64_tEEERKNSt13unordered_ | ck15scalar_callbackERR8Callable), |
| mapINSt6stringENSt7complexIdEEEE) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
| [cudaq::matrix_handler::to_string | Pv4N5cudaq15scalar_callback15scal |
|     (C++                          | ar_callbackERK15scalar_callback), |
|     function)](api/               |     [\[2                          |
| languages/cpp_api.html#_CPPv4NK5c | \]](api/languages/cpp_api.html#_C |
| udaq14matrix_handler9to_stringEb) | PPv4N5cudaq15scalar_callback15sca |
| -                                 | lar_callbackERR15scalar_callback) |
| [cudaq::matrix_handler::unique_id | -   [cudaq::scalar_operator (C++  |
|     (C++                          |     c                             |
|     function)](api/               | lass)](api/languages/cpp_api.html |
| languages/cpp_api.html#_CPPv4NK5c | #_CPPv4N5cudaq15scalar_operatorE) |
| udaq14matrix_handler9unique_idEv) | -                                 |
| -   [cudaq:                       | [cudaq::scalar_operator::evaluate |
| :matrix_handler::\~matrix_handler |     (C++                          |
|     (C++                          |                                   |
|     functi                        |    function)](api/languages/cpp_a |
| on)](api/languages/cpp_api.html#_ | pi.html#_CPPv4NK5cudaq15scalar_op |
| CPPv4N5cudaq14matrix_handlerD0Ev) | erator8evaluateERKNSt13unordered_ |
| -   [cudaq::matrix_op (C++        | mapINSt6stringENSt7complexIdEEEE) |
|     type)](api/languages/cpp_a    | -   [cudaq::scalar_ope            |
| pi.html#_CPPv4N5cudaq9matrix_opE) | rator::get_parameter_descriptions |
| -   [cudaq::matrix_op_term (C++   |     (C++                          |
|                                   |     f                             |
|  type)](api/languages/cpp_api.htm | unction)](api/languages/cpp_api.h |
| l#_CPPv4N5cudaq14matrix_op_termE) | tml#_CPPv4NK5cudaq15scalar_operat |
| -                                 | or26get_parameter_descriptionsEv) |
|    [cudaq::mdiag_operator_handler | -   [cu                           |
|     (C++                          | daq::scalar_operator::is_constant |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/lang           |
| N5cudaq22mdiag_operator_handlerE) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cudaq::mpi (C++              | 15scalar_operator11is_constantEv) |
|     type)](api/languages          | -   [c                            |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | udaq::scalar_operator::operator\* |
| -   [cudaq::mpi::all_gather (C++  |     (C++                          |
|     fu                            |     function                      |
| nction)](api/languages/cpp_api.ht | )](api/languages/cpp_api.html#_CP |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | Pv4N5cudaq15scalar_operatormlENSt |
| RNSt6vectorIdEERKNSt6vectorIdEE), | 7complexIdEERK15scalar_operator), |
|                                   |     [\[1\                         |
|   [\[1\]](api/languages/cpp_api.h | ]](api/languages/cpp_api.html#_CP |
| tml#_CPPv4N5cudaq3mpi10all_gather | Pv4N5cudaq15scalar_operatormlENSt |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::mpi::all_reduce (C++  |     [\[2\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|  function)](api/languages/cpp_api | operatormlEdRK15scalar_operator), |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     [\[3\]](api/languages/cp      |
| reduceE1TRK1TRK14BinaryFunction), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](api/langu             | operatormlEdRR15scalar_operator), |
| ages/cpp_api.html#_CPPv4I00EN5cud |     [\[4\]](api/languages         |
| aq3mpi10all_reduceE1TRK1TRK4Func) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -   [cudaq::mpi::broadcast (C++   | alar_operatormlENSt7complexIdEE), |
|     function)](api/               |     [\[5\]](api/languages/cpp     |
| languages/cpp_api.html#_CPPv4N5cu | _api.html#_CPPv4NKR5cudaq15scalar |
| daq3mpi9broadcastERNSt6stringEi), | _operatormlERK15scalar_operator), |
|     [\[1\]](api/la                |     [\[6\]]                       |
| nguages/cpp_api.html#_CPPv4N5cuda | (api/languages/cpp_api.html#_CPPv |
| q3mpi9broadcastERNSt6vectorIdEEi) | 4NKR5cudaq15scalar_operatormlEd), |
| -   [cudaq::mpi::finalize (C++    |     [\[7\]](api/language          |
|     f                             | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| unction)](api/languages/cpp_api.h | alar_operatormlENSt7complexIdEE), |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     [\[8\]](api/languages/cp      |
| -   [cudaq::mpi::initialize (C++  | p_api.html#_CPPv4NO5cudaq15scalar |
|     function                      | _operatormlERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[9\                         |
| Pv4N5cudaq3mpi10initializeEiPPc), | ]](api/languages/cpp_api.html#_CP |
|     [                             | Pv4NO5cudaq15scalar_operatormlEd) |
| \[1\]](api/languages/cpp_api.html | -   [cu                           |
| #_CPPv4N5cudaq3mpi10initializeEv) | daq::scalar_operator::operator\*= |
| -   [cudaq::mpi::is_initialized   |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function                      | es/cpp_api.html#_CPPv4N5cudaq15sc |
| )](api/languages/cpp_api.html#_CP | alar_operatormLENSt7complexIdEE), |
| Pv4N5cudaq3mpi14is_initializedEv) |     [\[1\]](api/languages/c       |
| -   [cudaq::mpi::num_ranks (C++   | pp_api.html#_CPPv4N5cudaq15scalar |
|     fu                            | _operatormLERK15scalar_operator), |
| nction)](api/languages/cpp_api.ht |     [\[2                          |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::mpi::rank (C++        | PPv4N5cudaq15scalar_operatormLEd) |
|                                   | -   [                             |
|    function)](api/languages/cpp_a | cudaq::scalar_operator::operator+ |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     (C++                          |
| -   [cudaq::noise_model (C++      |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|    class)](api/languages/cpp_api. | Pv4N5cudaq15scalar_operatorplENSt |
| html#_CPPv4N5cudaq11noise_modelE) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::n                     |     [\[1\                         |
| oise_model::add_all_qubit_channel | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatorplENSt |
|     function)](api                | 7complexIdEERR15scalar_operator), |
| /languages/cpp_api.html#_CPPv4IDp |     [\[2\]](api/languages/cp      |
| EN5cudaq11noise_model21add_all_qu | p_api.html#_CPPv4N5cudaq15scalar_ |
| bit_channelEvRK13kraus_channeli), | operatorplEdRK15scalar_operator), |
|     [\[1\]](api/langua            |     [\[3\]](api/languages/cp      |
| ges/cpp_api.html#_CPPv4N5cudaq11n | p_api.html#_CPPv4N5cudaq15scalar_ |
| oise_model21add_all_qubit_channel | operatorplEdRR15scalar_operator), |
| ERKNSt6stringERK13kraus_channeli) |     [\[4\]](api/languages         |
| -                                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|  [cudaq::noise_model::add_channel | alar_operatorplENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     funct                         | _api.html#_CPPv4NKR5cudaq15scalar |
| ion)](api/languages/cpp_api.html# | _operatorplERK15scalar_operator), |
| _CPPv4IDpEN5cudaq11noise_model11a |     [\[6\]]                       |
| dd_channelEvRK15PredicateFuncTy), | (api/languages/cpp_api.html#_CPPv |
|     [\[1\]](api/languages/cpp_    | 4NKR5cudaq15scalar_operatorplEd), |
| api.html#_CPPv4IDpEN5cudaq11noise |     [\[7\]]                       |
| _model11add_channelEvRKNSt6vector | (api/languages/cpp_api.html#_CPPv |
| INSt6size_tEEERK13kraus_channel), | 4NKR5cudaq15scalar_operatorplEv), |
|     [\[2\]](ap                    |     [\[8\]](api/language          |
| i/languages/cpp_api.html#_CPPv4N5 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| cudaq11noise_model11add_channelER | alar_operatorplENSt7complexIdEE), |
| KNSt6stringERK15PredicateFuncTy), |     [\[9\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4NO5cudaq15scalar |
| [\[3\]](api/languages/cpp_api.htm | _operatorplERK15scalar_operator), |
| l#_CPPv4N5cudaq11noise_model11add |     [\[10\]                       |
| _channelERKNSt6stringERKNSt6vecto | ](api/languages/cpp_api.html#_CPP |
| rINSt6size_tEEERK13kraus_channel) | v4NO5cudaq15scalar_operatorplEd), |
| -   [cudaq::noise_model::empty    |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function                      | Pv4NO5cudaq15scalar_operatorplEv) |
| )](api/languages/cpp_api.html#_CP | -   [c                            |
| Pv4NK5cudaq11noise_model5emptyEv) | udaq::scalar_operator::operator+= |
| -                                 |     (C++                          |
| [cudaq::noise_model::get_channels |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     function)](api/l              | alar_operatorpLENSt7complexIdEE), |
| anguages/cpp_api.html#_CPPv4I0ENK |     [\[1\]](api/languages/c       |
| 5cudaq11noise_model12get_channels | pp_api.html#_CPPv4N5cudaq15scalar |
| ENSt6vectorI13kraus_channelEERKNS | _operatorpLERK15scalar_operator), |
| t6vectorINSt6size_tEEERKNSt6vecto |     [\[2                          |
| rINSt6size_tEEERKNSt6vectorIdEE), | \]](api/languages/cpp_api.html#_C |
|     [\[1\]](api/languages/cpp_a   | PPv4N5cudaq15scalar_operatorpLEd) |
| pi.html#_CPPv4NK5cudaq11noise_mod | -   [                             |
| el12get_channelsERKNSt6stringERKN | cudaq::scalar_operator::operator- |
| St6vectorINSt6size_tEEERKNSt6vect |     (C++                          |
| orINSt6size_tEEERKNSt6vectorIdEE) |     function                      |
| -                                 | )](api/languages/cpp_api.html#_CP |
|  [cudaq::noise_model::noise_model | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     function)](api                |     [\[1\                         |
| /languages/cpp_api.html#_CPPv4N5c | ]](api/languages/cpp_api.html#_CP |
| udaq11noise_model11noise_modelEv) | Pv4N5cudaq15scalar_operatormiENSt |
| -   [cu                           | 7complexIdEERR15scalar_operator), |
| daq::noise_model::PredicateFuncTy |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     type)](api/la                 | operatormiEdRK15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[3\]](api/languages/cp      |
| q11noise_model15PredicateFuncTyE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cud                          | operatormiEdRR15scalar_operator), |
| aq::noise_model::register_channel |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     function)](api/languages      | alar_operatormiENSt7complexIdEE), |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     [\[5\]](api/languages/cpp     |
| noise_model16register_channelEvv) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::                      | _operatormiERK15scalar_operator), |
| noise_model::requires_constructor |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     type)](api/languages/cp       | 4NKR5cudaq15scalar_operatormiEd), |
| p_api.html#_CPPv4I0DpEN5cudaq11no |     [\[7\]]                       |
| ise_model20requires_constructorE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::noise_model_type (C++ | 4NKR5cudaq15scalar_operatormiEv), |
|     e                             |     [\[8\]](api/language          |
| num)](api/languages/cpp_api.html# | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| _CPPv4N5cudaq16noise_model_typeE) | alar_operatormiENSt7complexIdEE), |
| -   [cudaq::no                    |     [\[9\]](api/languages/cp      |
| ise_model_type::amplitude_damping | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatormiERK15scalar_operator), |
|     enumerator)](api/languages    |     [\[10\]                       |
| /cpp_api.html#_CPPv4N5cudaq16nois | ](api/languages/cpp_api.html#_CPP |
| e_model_type17amplitude_dampingE) | v4NO5cudaq15scalar_operatormiEd), |
| -   [cudaq::noise_mode            |     [\[11\                        |
| l_type::amplitude_damping_channel | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatormiEv) |
|     e                             | -   [c                            |
| numerator)](api/languages/cpp_api | udaq::scalar_operator::operator-= |
| .html#_CPPv4N5cudaq16noise_model_ |     (C++                          |
| type25amplitude_damping_channelE) |     function)](api/languag        |
| -   [cudaq::n                     | es/cpp_api.html#_CPPv4N5cudaq15sc |
| oise_model_type::bit_flip_channel | alar_operatormIENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     enumerator)](api/language     | pp_api.html#_CPPv4N5cudaq15scalar |
| s/cpp_api.html#_CPPv4N5cudaq16noi | _operatormIERK15scalar_operator), |
| se_model_type16bit_flip_channelE) |     [\[2                          |
| -   [cudaq::                      | \]](api/languages/cpp_api.html#_C |
| noise_model_type::depolarization1 | PPv4N5cudaq15scalar_operatormIEd) |
|     (C++                          | -   [                             |
|     enumerator)](api/languag      | cudaq::scalar_operator::operator/ |
| es/cpp_api.html#_CPPv4N5cudaq16no |     (C++                          |
| ise_model_type15depolarization1E) |     function                      |
| -   [cudaq::                      | )](api/languages/cpp_api.html#_CP |
| noise_model_type::depolarization2 | Pv4N5cudaq15scalar_operatordvENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     enumerator)](api/languag      |     [\[1\                         |
| es/cpp_api.html#_CPPv4N5cudaq16no | ]](api/languages/cpp_api.html#_CP |
| ise_model_type15depolarization2E) | Pv4N5cudaq15scalar_operatordvENSt |
| -   [cudaq::noise_m               | 7complexIdEERR15scalar_operator), |
| odel_type::depolarization_channel |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|                                   | operatordvEdRK15scalar_operator), |
|   enumerator)](api/languages/cpp_ |     [\[3\]](api/languages/cp      |
| api.html#_CPPv4N5cudaq16noise_mod | p_api.html#_CPPv4N5cudaq15scalar_ |
| el_type22depolarization_channelE) | operatordvEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
|  [cudaq::noise_model_type::pauli1 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatordvENSt7complexIdEE), |
|     enumerator)](a                |     [\[5\]](api/languages/cpp     |
| pi/languages/cpp_api.html#_CPPv4N | _api.html#_CPPv4NKR5cudaq15scalar |
| 5cudaq16noise_model_type6pauli1E) | _operatordvERK15scalar_operator), |
| -                                 |     [\[6\]]                       |
|  [cudaq::noise_model_type::pauli2 | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatordvEd), |
|     enumerator)](a                |     [\[7\]](api/language          |
| pi/languages/cpp_api.html#_CPPv4N | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| 5cudaq16noise_model_type6pauli2E) | alar_operatordvENSt7complexIdEE), |
| -   [cudaq                        |     [\[8\]](api/languages/cp      |
| ::noise_model_type::phase_damping | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatordvERK15scalar_operator), |
|     enumerator)](api/langu        |     [\[9\                         |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ]](api/languages/cpp_api.html#_CP |
| noise_model_type13phase_dampingE) | Pv4NO5cudaq15scalar_operatordvEd) |
| -   [cudaq::noi                   | -   [c                            |
| se_model_type::phase_flip_channel | udaq::scalar_operator::operator/= |
|     (C++                          |     (C++                          |
|     enumerator)](api/languages/   |     function)](api/languag        |
| cpp_api.html#_CPPv4N5cudaq16noise | es/cpp_api.html#_CPPv4N5cudaq15sc |
| _model_type18phase_flip_channelE) | alar_operatordVENSt7complexIdEE), |
| -                                 |     [\[1\]](api/languages/c       |
| [cudaq::noise_model_type::unknown | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatordVERK15scalar_operator), |
|     enumerator)](ap               |     [\[2                          |
| i/languages/cpp_api.html#_CPPv4N5 | \]](api/languages/cpp_api.html#_C |
| cudaq16noise_model_type7unknownE) | PPv4N5cudaq15scalar_operatordVEd) |
| -                                 | -   [                             |
| [cudaq::noise_model_type::x_error | cudaq::scalar_operator::operator= |
|     (C++                          |     (C++                          |
|     enumerator)](ap               |     function)](api/languages/c    |
| i/languages/cpp_api.html#_CPPv4N5 | pp_api.html#_CPPv4N5cudaq15scalar |
| cudaq16noise_model_type7x_errorE) | _operatoraSERK15scalar_operator), |
| -                                 |     [\[1\]](api/languages/        |
| [cudaq::noise_model_type::y_error | cpp_api.html#_CPPv4N5cudaq15scala |
|     (C++                          | r_operatoraSERR15scalar_operator) |
|     enumerator)](ap               | -   [c                            |
| i/languages/cpp_api.html#_CPPv4N5 | udaq::scalar_operator::operator== |
| cudaq16noise_model_type7y_errorE) |     (C++                          |
| -                                 |     function)](api/languages/c    |
| [cudaq::noise_model_type::z_error | pp_api.html#_CPPv4NK5cudaq15scala |
|     (C++                          | r_operatoreqERK15scalar_operator) |
|     enumerator)](ap               | -   [cudaq:                       |
| i/languages/cpp_api.html#_CPPv4N5 | :scalar_operator::scalar_operator |
| cudaq16noise_model_type7z_errorE) |     (C++                          |
| -   [cudaq::num_available_gpus    |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     function                      | #_CPPv4N5cudaq15scalar_operator15 |
| )](api/languages/cpp_api.html#_CP | scalar_operatorENSt7complexIdEE), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[1\]](api/langu             |
| -   [cudaq::observe (C++          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     function)](                   | scalar_operator15scalar_operatorE |
| api/languages/cpp_api.html#_CPPv4 | RK15scalar_callbackRRNSt13unorder |
| I00Dp0EN5cudaq7observeENSt6vector | ed_mapINSt6stringENSt6stringEEE), |
| I14observe_resultEERR13QuantumKer |     [\[2\                         |
| nelRK15SpinOpContainerDpRR4Args), | ]](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/languages/cpp_api | Pv4N5cudaq15scalar_operator15scal |
| .html#_CPPv4I0Dp0EN5cudaq7observe | ar_operatorERK15scalar_operator), |
| E14observe_resultNSt6size_tERR13Q |     [\[3\]](api/langu             |
| uantumKernelRK7spin_opDpRR4Args), | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     [\[2                          | scalar_operator15scalar_operatorE |
| \]](api/languages/cpp_api.html#_C | RR15scalar_callbackRRNSt13unorder |
| PPv4I0Dp0EN5cudaq7observeE14obser | ed_mapINSt6stringENSt6stringEEE), |
| ve_resultRK15observe_optionsRR13Q |     [\[4\                         |
| uantumKernelRK7spin_opDpRR4Args), | ]](api/languages/cpp_api.html#_CP |
|     [\[3\]](api/langu             | Pv4N5cudaq15scalar_operator15scal |
| ages/cpp_api.html#_CPPv4I0Dp0EN5c | ar_operatorERR15scalar_operator), |
| udaq7observeE14observe_resultRR13 |     [\[5\]](api/language          |
| QuantumKernelRK7spin_opDpRR4Args) | s/cpp_api.html#_CPPv4N5cudaq15sca |
| -   [cudaq::observe_options (C++  | lar_operator15scalar_operatorEd), |
|     st                            |     [\[6\]](api/languag           |
| ruct)](api/languages/cpp_api.html | es/cpp_api.html#_CPPv4N5cudaq15sc |
| #_CPPv4N5cudaq15observe_optionsE) | alar_operator15scalar_operatorEv) |
| -   [cudaq::observe_result (C++   | -   [                             |
|                                   | cudaq::scalar_operator::to_matrix |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14observe_resultE) |                                   |
| -                                 |   function)](api/languages/cpp_ap |
|    [cudaq::observe_result::counts | i.html#_CPPv4NK5cudaq15scalar_ope |
|     (C++                          | rator9to_matrixERKNSt13unordered_ |
|     function)](api/languages/c    | mapINSt6stringENSt7complexIdEEEE) |
| pp_api.html#_CPPv4N5cudaq14observ | -   [                             |
| e_result6countsERK12spin_op_term) | cudaq::scalar_operator::to_string |
| -   [cudaq::observe_result::dump  |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)                     | anguages/cpp_api.html#_CPPv4NK5cu |
| ](api/languages/cpp_api.html#_CPP | daq15scalar_operator9to_stringEv) |
| v4N5cudaq14observe_result4dumpEv) | -   [cudaq::s                     |
| -   [c                            | calar_operator::\~scalar_operator |
| udaq::observe_result::expectation |     (C++                          |
|     (C++                          |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
| function)](api/languages/cpp_api. | PPv4N5cudaq15scalar_operatorD0Ev) |
| html#_CPPv4N5cudaq14observe_resul | -   [cuda                         |
| t11expectationERK12spin_op_term), | q::SerializedCodeExecutionContext |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     class)](api/lang              |
| q14observe_result11expectationEv) | uages/cpp_api.html#_CPPv4N5cudaq3 |
| -   [cuda                         | 0SerializedCodeExecutionContextE) |
| q::observe_result::id_coefficient | -   [cudaq::set_noise (C++        |
|     (C++                          |     function)](api/langu          |
|     function)](api/langu          | ages/cpp_api.html#_CPPv4N5cudaq9s |
| ages/cpp_api.html#_CPPv4N5cudaq14 | et_noiseERKN5cudaq11noise_modelE) |
| observe_result14id_coefficientEv) | -   [cudaq::set_random_seed (C++  |
| -   [cuda                         |     function)](api/               |
| q::observe_result::observe_result | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq15set_random_seedENSt6size_tE) |
|                                   | -   [cudaq::simulation_precision  |
|   function)](api/languages/cpp_ap |     (C++                          |
| i.html#_CPPv4N5cudaq14observe_res |     enum)                         |
| ult14observe_resultEdRK7spin_op), | ](api/languages/cpp_api.html#_CPP |
|     [\[1\]](a                     | v4N5cudaq20simulation_precisionE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [                             |
| 5cudaq14observe_result14observe_r | cudaq::simulation_precision::fp32 |
| esultEdRK7spin_op13sample_result) |     (C++                          |
| -                                 |     enumerator)](api              |
|  [cudaq::observe_result::operator | /languages/cpp_api.html#_CPPv4N5c |
|     double (C++                   | udaq20simulation_precision4fp32E) |
|     functio                       | -   [                             |
| n)](api/languages/cpp_api.html#_C | cudaq::simulation_precision::fp64 |
| PPv4N5cudaq14observe_resultcvdEv) |     (C++                          |
| -                                 |     enumerator)](api              |
|  [cudaq::observe_result::raw_data | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq20simulation_precision4fp64E) |
|     function)](ap                 | -   [cudaq::SimulationState (C++  |
| i/languages/cpp_api.html#_CPPv4N5 |     c                             |
| cudaq14observe_result8raw_dataEv) | lass)](api/languages/cpp_api.html |
| -   [cudaq::operator_handler (C++ | #_CPPv4N5cudaq15SimulationStateE) |
|     cl                            | -   [                             |
| ass)](api/languages/cpp_api.html# | cudaq::SimulationState::precision |
| _CPPv4N5cudaq16operator_handlerE) |     (C++                          |
| -   [cudaq::optimizable_function  |     enum)](api                    |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     class)                        | udaq15SimulationState9precisionE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq:                       |
| v4N5cudaq20optimizable_functionE) | :SimulationState::precision::fp32 |
| -   [cudaq::optimization_result   |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     type                          | uages/cpp_api.html#_CPPv4N5cudaq1 |
| )](api/languages/cpp_api.html#_CP | 5SimulationState9precision4fp32E) |
| Pv4N5cudaq19optimization_resultE) | -   [cudaq:                       |
| -   [cudaq::optimizer (C++        | :SimulationState::precision::fp64 |
|     class)](api/languages/cpp_a   |     (C++                          |
| pi.html#_CPPv4N5cudaq9optimizerE) |     enumerator)](api/lang         |
| -   [cudaq::optimizer::optimize   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     (C++                          | 5SimulationState9precision4fp64E) |
|                                   | -                                 |
|  function)](api/languages/cpp_api |   [cudaq::SimulationState::Tensor |
| .html#_CPPv4N5cudaq9optimizer8opt |     (C++                          |
| imizeEKiRR20optimizable_function) |     struct)](                     |
| -   [cu                           | api/languages/cpp_api.html#_CPPv4 |
| daq::optimizer::requiresGradients | N5cudaq15SimulationState6TensorE) |
|     (C++                          | -   [cudaq::spin_handler (C++     |
|     function)](api/la             |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda |   class)](api/languages/cpp_api.h |
| q9optimizer17requiresGradientsEv) | tml#_CPPv4N5cudaq12spin_handlerE) |
| -   [cudaq::orca (C++             | -   [cudaq:                       |
|     type)](api/languages/         | :spin_handler::to_diagonal_matrix |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     (C++                          |
| -   [cudaq::orca::sample (C++     |     function)](api/la             |
|     function)](api/languages/c    | nguages/cpp_api.html#_CPPv4NK5cud |
| pp_api.html#_CPPv4N5cudaq4orca6sa | aq12spin_handler18to_diagonal_mat |
| mpleERNSt6vectorINSt6size_tEEERNS | rixERNSt13unordered_mapINSt6size_ |
| t6vectorINSt6size_tEEERNSt6vector | tENSt7int64_tEEERKNSt13unordered_ |
| IdEERNSt6vectorIdEEiNSt6size_tE), | mapINSt6stringENSt7complexIdEEEE) |
|     [\[1\]]                       | -                                 |
| (api/languages/cpp_api.html#_CPPv |   [cudaq::spin_handler::to_matrix |
| 4N5cudaq4orca6sampleERNSt6vectorI |     (C++                          |
| NSt6size_tEEERNSt6vectorINSt6size |     function                      |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::orca::sample_async    | Pv4N5cudaq12spin_handler9to_matri |
|     (C++                          | xERKNSt6stringENSt7complexIdEEb), |
|                                   |     [\[1                          |
| function)](api/languages/cpp_api. | \]](api/languages/cpp_api.html#_C |
| html#_CPPv4N5cudaq4orca12sample_a | PPv4NK5cudaq12spin_handler9to_mat |
| syncERNSt6vectorINSt6size_tEEERNS | rixERNSt13unordered_mapINSt6size_ |
| t6vectorINSt6size_tEEERNSt6vector | tENSt7int64_tEEERKNSt13unordered_ |
| IdEERNSt6vectorIdEEiNSt6size_tE), | mapINSt6stringENSt7complexIdEEEE) |
|     [\[1\]](api/la                | -   [cuda                         |
| nguages/cpp_api.html#_CPPv4N5cuda | q::spin_handler::to_sparse_matrix |
| q4orca12sample_asyncERNSt6vectorI |     (C++                          |
| NSt6size_tEEERNSt6vectorINSt6size |     function)](api/               |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::OrcaRemoteRESTQPU     | daq12spin_handler16to_sparse_matr |
|     (C++                          | ixERKNSt6stringENSt7complexIdEEb) |
|     cla                           | -                                 |
| ss)](api/languages/cpp_api.html#_ |   [cudaq::spin_handler::to_string |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |     (C++                          |
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
| -   [to_json()                    |     -   [(cuda                    |
|     (                             | q.operators.spin.SpinOperatorTerm |
| cudaq.gradients.CentralDifference |         method)](api/language     |
|     method)](api/la               | s/python_api.html#cudaq.operators |
| nguages/python_api.html#cudaq.gra | .spin.SpinOperatorTerm.to_string) |
| dients.CentralDifference.to_json) | -   [translate() (in module       |
|     -   [(                        |     cudaq)](api/languages         |
| cudaq.gradients.ForwardDifference | /python_api.html#cudaq.translate) |
|         method)](api/la           | -   [trim()                       |
| nguages/python_api.html#cudaq.gra |     (cu                           |
| dients.ForwardDifference.to_json) | daq.operators.boson.BosonOperator |
|     -                             |     method)](api/l                |
|  [(cudaq.gradients.ParameterShift | anguages/python_api.html#cudaq.op |
|         method)](api              | erators.boson.BosonOperator.trim) |
| /languages/python_api.html#cudaq. |     -   [(cudaq.                  |
| gradients.ParameterShift.to_json) | operators.fermion.FermionOperator |
|     -   [(                        |         method)](api/langu        |
| cudaq.operators.spin.SpinOperator | ages/python_api.html#cudaq.operat |
|         method)](api/la           | ors.fermion.FermionOperator.trim) |
| nguages/python_api.html#cudaq.ope |     -                             |
| rators.spin.SpinOperator.to_json) |  [(cudaq.operators.MatrixOperator |
|     -   [(cuda                    |         method)](                 |
| q.operators.spin.SpinOperatorTerm | api/languages/python_api.html#cud |
|         method)](api/langua       | aq.operators.MatrixOperator.trim) |
| ges/python_api.html#cudaq.operato |     -   [(                        |
| rs.spin.SpinOperatorTerm.to_json) | cudaq.operators.spin.SpinOperator |
|     -   [(cudaq.optimizers.COBYLA |         method)](api              |
|         metho                     | /languages/python_api.html#cudaq. |
| d)](api/languages/python_api.html | operators.spin.SpinOperator.trim) |
| #cudaq.optimizers.COBYLA.to_json) | -   [type_to_str()                |
|     -   [                         |     (cudaq.PyKernelDecorator      |
| (cudaq.optimizers.GradientDescent |     static                        |
|         method)](api/l            |     method)](                     |
| anguages/python_api.html#cudaq.op | api/languages/python_api.html#cud |
| timizers.GradientDescent.to_json) | aq.PyKernelDecorator.type_to_str) |
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
© Copyright 2025, NVIDIA Corporation & Affiliates.
:::

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme) provided by
[Read the Docs](https://readthedocs.org).
:::
:::
:::
:::
