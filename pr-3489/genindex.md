::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-3489
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
| -   [canonicalize()               | -   [cudaq::OrcaRemoteRESTQPU     |
|     (cu                           |     (C++                          |
| daq.operators.boson.BosonOperator |     cla                           |
|     method)](api/languages        | ss)](api/languages/cpp_api.html#_ |
| /python_api.html#cudaq.operators. | CPPv4N5cudaq17OrcaRemoteRESTQPUE) |
| boson.BosonOperator.canonicalize) | -   [cudaq::pauli1 (C++           |
|     -   [(cudaq.                  |     class)](api/languages/cp      |
| operators.boson.BosonOperatorTerm | p_api.html#_CPPv4N5cudaq6pauli1E) |
|                                   | -                                 |
|        method)](api/languages/pyt |    [cudaq::pauli1::num_parameters |
| hon_api.html#cudaq.operators.boso |     (C++                          |
| n.BosonOperatorTerm.canonicalize) |     member)]                      |
|     -   [(cudaq.                  | (api/languages/cpp_api.html#_CPPv |
| operators.fermion.FermionOperator | 4N5cudaq6pauli114num_parametersE) |
|                                   | -   [cudaq::pauli1::num_targets   |
|        method)](api/languages/pyt |     (C++                          |
| hon_api.html#cudaq.operators.ferm |     membe                         |
| ion.FermionOperator.canonicalize) | r)](api/languages/cpp_api.html#_C |
|     -   [(cudaq.oper              | PPv4N5cudaq6pauli111num_targetsE) |
| ators.fermion.FermionOperatorTerm | -   [cudaq::pauli1::pauli1 (C++   |
|                                   |     function)](api/languages/cpp_ |
|    method)](api/languages/python_ | api.html#_CPPv4N5cudaq6pauli16pau |
| api.html#cudaq.operators.fermion. | li1ERKNSt6vectorIN5cudaq4realEEE) |
| FermionOperatorTerm.canonicalize) | -   [cudaq::pauli2 (C++           |
|     -                             |     class)](api/languages/cp      |
|  [(cudaq.operators.MatrixOperator | p_api.html#_CPPv4N5cudaq6pauli2E) |
|         method)](api/lang         | -                                 |
| uages/python_api.html#cudaq.opera |    [cudaq::pauli2::num_parameters |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     member)]                      |
| udaq.operators.MatrixOperatorTerm | (api/languages/cpp_api.html#_CPPv |
|         method)](api/language     | 4N5cudaq6pauli214num_parametersE) |
| s/python_api.html#cudaq.operators | -   [cudaq::pauli2::num_targets   |
| .MatrixOperatorTerm.canonicalize) |     (C++                          |
|     -   [(                        |     membe                         |
| cudaq.operators.spin.SpinOperator | r)](api/languages/cpp_api.html#_C |
|         method)](api/languag      | PPv4N5cudaq6pauli211num_targetsE) |
| es/python_api.html#cudaq.operator | -   [cudaq::pauli2::pauli2 (C++   |
| s.spin.SpinOperator.canonicalize) |     function)](api/languages/cpp_ |
|     -   [(cuda                    | api.html#_CPPv4N5cudaq6pauli26pau |
| q.operators.spin.SpinOperatorTerm | li2ERKNSt6vectorIN5cudaq4realEEE) |
|         method)](api/languages/p  | -   [cudaq::phase_damping (C++    |
| ython_api.html#cudaq.operators.sp |                                   |
| in.SpinOperatorTerm.canonicalize) |  class)](api/languages/cpp_api.ht |
| -   [canonicalized() (in module   | ml#_CPPv4N5cudaq13phase_dampingE) |
|     cuda                          | -   [cud                          |
| q.boson)](api/languages/python_ap | aq::phase_damping::num_parameters |
| i.html#cudaq.boson.canonicalized) |     (C++                          |
|     -   [(in module               |     member)](api/lan              |
|         cudaq.fe                  | guages/cpp_api.html#_CPPv4N5cudaq |
| rmion)](api/languages/python_api. | 13phase_damping14num_parametersE) |
| html#cudaq.fermion.canonicalized) | -   [                             |
|     -   [(in module               | cudaq::phase_damping::num_targets |
|                                   |     (C++                          |
|        cudaq.operators.custom)](a |     member)](api/                 |
| pi/languages/python_api.html#cuda | languages/cpp_api.html#_CPPv4N5cu |
| q.operators.custom.canonicalized) | daq13phase_damping11num_targetsE) |
|     -   [(in module               | -   [cudaq::phase_flip_channel    |
|         cu                        |     (C++                          |
| daq.spin)](api/languages/python_a |     clas                          |
| pi.html#cudaq.spin.canonicalized) | s)](api/languages/cpp_api.html#_C |
| -   [CentralDifference (class in  | PPv4N5cudaq18phase_flip_channelE) |
|     cudaq.gradients)              | -   [cudaq::p                     |
| ](api/languages/python_api.html#c | hase_flip_channel::num_parameters |
| udaq.gradients.CentralDifference) |     (C++                          |
| -   [clear() (cudaq.Resources     |     member)](api/language         |
|     method)](api/languages/pytho  | s/cpp_api.html#_CPPv4N5cudaq18pha |
| n_api.html#cudaq.Resources.clear) | se_flip_channel14num_parametersE) |
|     -   [(cudaq.SampleResult      | -   [cudaq                        |
|                                   | ::phase_flip_channel::num_targets |
|   method)](api/languages/python_a |     (C++                          |
| pi.html#cudaq.SampleResult.clear) |     member)](api/langu            |
| -   [COBYLA (class in             | ages/cpp_api.html#_CPPv4N5cudaq18 |
|     cudaq.o                       | phase_flip_channel11num_targetsE) |
| ptimizers)](api/languages/python_ | -   [cudaq::product_op (C++       |
| api.html#cudaq.optimizers.COBYLA) |                                   |
| -   [coefficient                  |  class)](api/languages/cpp_api.ht |
|     (cudaq.                       | ml#_CPPv4I0EN5cudaq10product_opE) |
| operators.boson.BosonOperatorTerm | -   [cudaq::product_op::begin     |
|     property)](api/languages/py   |     (C++                          |
| thon_api.html#cudaq.operators.bos |     functio                       |
| on.BosonOperatorTerm.coefficient) | n)](api/languages/cpp_api.html#_C |
|     -   [(cudaq.oper              | PPv4NK5cudaq10product_op5beginEv) |
| ators.fermion.FermionOperatorTerm | -                                 |
|                                   |  [cudaq::product_op::canonicalize |
|   property)](api/languages/python |     (C++                          |
| _api.html#cudaq.operators.fermion |     func                          |
| .FermionOperatorTerm.coefficient) | tion)](api/languages/cpp_api.html |
|     -   [(c                       | #_CPPv4N5cudaq10product_op12canon |
| udaq.operators.MatrixOperatorTerm | icalizeERKNSt3setINSt6size_tEEE), |
|         property)](api/languag    |     [\[1\]](api                   |
| es/python_api.html#cudaq.operator | /languages/cpp_api.html#_CPPv4N5c |
| s.MatrixOperatorTerm.coefficient) | udaq10product_op12canonicalizeEv) |
|     -   [(cuda                    | -   [                             |
| q.operators.spin.SpinOperatorTerm | cudaq::product_op::const_iterator |
|         property)](api/languages/ |     (C++                          |
| python_api.html#cudaq.operators.s |     struct)](api/                 |
| pin.SpinOperatorTerm.coefficient) | languages/cpp_api.html#_CPPv4N5cu |
| -   [col_count                    | daq10product_op14const_iteratorE) |
|     (cudaq.KrausOperator          | -   [cudaq::product_o             |
|     prope                         | p::const_iterator::const_iterator |
| rty)](api/languages/python_api.ht |     (C++                          |
| ml#cudaq.KrausOperator.col_count) |     fu                            |
| -   [compile()                    | nction)](api/languages/cpp_api.ht |
|     (cudaq.PyKernelDecorator      | ml#_CPPv4N5cudaq10product_op14con |
|     metho                         | st_iterator14const_iteratorEPK10p |
| d)](api/languages/python_api.html | roduct_opI9HandlerTyENSt6size_tE) |
| #cudaq.PyKernelDecorator.compile) | -   [cudaq::produ                 |
| -   [ComplexMatrix (class in      | ct_op::const_iterator::operator!= |
|     cudaq)](api/languages/pyt     |     (C++                          |
| hon_api.html#cudaq.ComplexMatrix) |     fun                           |
| -   [compute()                    | ction)](api/languages/cpp_api.htm |
|     (                             | l#_CPPv4NK5cudaq10product_op14con |
| cudaq.gradients.CentralDifference | st_iteratorneERK14const_iterator) |
|     method)](api/la               | -   [cudaq::produ                 |
| nguages/python_api.html#cudaq.gra | ct_op::const_iterator::operator\* |
| dients.CentralDifference.compute) |     (C++                          |
|     -   [(                        |     function)](api/lang           |
| cudaq.gradients.ForwardDifference | uages/cpp_api.html#_CPPv4NK5cudaq |
|         method)](api/la           | 10product_op14const_iteratormlEv) |
| nguages/python_api.html#cudaq.gra | -   [cudaq::produ                 |
| dients.ForwardDifference.compute) | ct_op::const_iterator::operator++ |
|     -                             |     (C++                          |
|  [(cudaq.gradients.ParameterShift |     function)](api/lang           |
|         method)](api              | uages/cpp_api.html#_CPPv4N5cudaq1 |
| /languages/python_api.html#cudaq. | 0product_op14const_iteratorppEi), |
| gradients.ParameterShift.compute) |     [\[1\]](api/lan               |
| -   [const()                      | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 10product_op14const_iteratorppEv) |
|   (cudaq.operators.ScalarOperator | -   [cudaq::produc                |
|     class                         | t_op::const_iterator::operator\-- |
|     method)](a                    |     (C++                          |
| pi/languages/python_api.html#cuda |     function)](api/lang           |
| q.operators.ScalarOperator.const) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [copy()                       | 0product_op14const_iteratormmEi), |
|     (cu                           |     [\[1\]](api/lan               |
| daq.operators.boson.BosonOperator | guages/cpp_api.html#_CPPv4N5cudaq |
|     method)](api/l                | 10product_op14const_iteratormmEv) |
| anguages/python_api.html#cudaq.op | -   [cudaq::produc                |
| erators.boson.BosonOperator.copy) | t_op::const_iterator::operator-\> |
|     -   [(cudaq.                  |     (C++                          |
| operators.boson.BosonOperatorTerm |     function)](api/lan            |
|         method)](api/langu        | guages/cpp_api.html#_CPPv4N5cudaq |
| ages/python_api.html#cudaq.operat | 10product_op14const_iteratorptEv) |
| ors.boson.BosonOperatorTerm.copy) | -   [cudaq::produ                 |
|     -   [(cudaq.                  | ct_op::const_iterator::operator== |
| operators.fermion.FermionOperator |     (C++                          |
|         method)](api/langu        |     fun                           |
| ages/python_api.html#cudaq.operat | ction)](api/languages/cpp_api.htm |
| ors.fermion.FermionOperator.copy) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(cudaq.oper              | st_iteratoreqERK14const_iterator) |
| ators.fermion.FermionOperatorTerm | -   [cudaq::product_op::degrees   |
|         method)](api/languages    |     (C++                          |
| /python_api.html#cudaq.operators. |     function)                     |
| fermion.FermionOperatorTerm.copy) | ](api/languages/cpp_api.html#_CPP |
|     -                             | v4NK5cudaq10product_op7degreesEv) |
|  [(cudaq.operators.MatrixOperator | -   [cudaq::product_op::dump (C++ |
|         method)](                 |     functi                        |
| api/languages/python_api.html#cud | on)](api/languages/cpp_api.html#_ |
| aq.operators.MatrixOperator.copy) | CPPv4NK5cudaq10product_op4dumpEv) |
|     -   [(c                       | -   [cudaq::product_op::end (C++  |
| udaq.operators.MatrixOperatorTerm |     funct                         |
|         method)](api/             | ion)](api/languages/cpp_api.html# |
| languages/python_api.html#cudaq.o | _CPPv4NK5cudaq10product_op3endEv) |
| perators.MatrixOperatorTerm.copy) | -   [c                            |
|     -   [(                        | udaq::product_op::get_coefficient |
| cudaq.operators.spin.SpinOperator |     (C++                          |
|         method)](api              |     function)](api/lan            |
| /languages/python_api.html#cudaq. | guages/cpp_api.html#_CPPv4NK5cuda |
| operators.spin.SpinOperator.copy) | q10product_op15get_coefficientEv) |
|     -   [(cuda                    | -                                 |
| q.operators.spin.SpinOperatorTerm |   [cudaq::product_op::get_term_id |
|         method)](api/lan          |     (C++                          |
| guages/python_api.html#cudaq.oper |     function)](api                |
| ators.spin.SpinOperatorTerm.copy) | /languages/cpp_api.html#_CPPv4NK5 |
| -   [count() (cudaq.Resources     | cudaq10product_op11get_term_idEv) |
|     method)](api/languages/pytho  | -                                 |
| n_api.html#cudaq.Resources.count) |   [cudaq::product_op::is_identity |
|     -   [(cudaq.SampleResult      |     (C++                          |
|                                   |     function)](api                |
|   method)](api/languages/python_a | /languages/cpp_api.html#_CPPv4NK5 |
| pi.html#cudaq.SampleResult.count) | cudaq10product_op11is_identityEv) |
| -   [count_controls()             | -   [cudaq::product_op::num_ops   |
|     (cudaq.Resources              |     (C++                          |
|     meth                          |     function)                     |
| od)](api/languages/python_api.htm | ](api/languages/cpp_api.html#_CPP |
| l#cudaq.Resources.count_controls) | v4NK5cudaq10product_op7num_opsEv) |
| -   [counts()                     | -                                 |
|     (cudaq.ObserveResult          |    [cudaq::product_op::operator\* |
|                                   |     (C++                          |
| method)](api/languages/python_api |     function)](api/languages/     |
| .html#cudaq.ObserveResult.counts) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [create() (in module          | oduct_opmlE10product_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|    cudaq.boson)](api/languages/py |     [\[1\]](api/languages/        |
| thon_api.html#cudaq.boson.create) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(in module               | oduct_opmlE10product_opI1TERK15sc |
|         c                         | alar_operatorRR10product_opI1TE), |
| udaq.fermion)](api/languages/pyth |     [\[2\]](api/languages/        |
| on_api.html#cudaq.fermion.create) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [csr_spmatrix (C++            | oduct_opmlE10product_opI1TERR15sc |
|     type)](api/languages/c        | alar_operatorRK10product_opI1TE), |
| pp_api.html#_CPPv412csr_spmatrix) |     [\[3\]](api/languages/        |
| -   cudaq                         | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [module](api/langua       | oduct_opmlE10product_opI1TERR15sc |
| ges/python_api.html#module-cudaq) | alar_operatorRR10product_opI1TE), |
| -   [cudaq (C++                   |     [\[4\]](api/                  |
|     type)](api/lan                | languages/cpp_api.html#_CPPv4I0EN |
| guages/cpp_api.html#_CPPv45cudaq) | 5cudaq10product_opmlE6sum_opI1TER |
| -   [cudaq.apply_noise() (in      | K15scalar_operatorRK6sum_opI1TE), |
|     module                        |     [\[5\]](api/                  |
|     cudaq)](api/languages/python_ | languages/cpp_api.html#_CPPv4I0EN |
| api.html#cudaq.cudaq.apply_noise) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq.boson                   | K15scalar_operatorRR6sum_opI1TE), |
|     -   [module](api/languages/py |     [\[6\]](api/                  |
| thon_api.html#module-cudaq.boson) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.fermion                 | 5cudaq10product_opmlE6sum_opI1TER |
|                                   | R15scalar_operatorRK6sum_opI1TE), |
|   -   [module](api/languages/pyth |     [\[7\]](api/                  |
| on_api.html#module-cudaq.fermion) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.operators.custom        | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [mo                       | R15scalar_operatorRR6sum_opI1TE), |
| dule](api/languages/python_api.ht |     [\[8\]](api/languages         |
| ml#module-cudaq.operators.custom) | /cpp_api.html#_CPPv4NK5cudaq10pro |
| -   cudaq.spin                    | duct_opmlERK6sum_opI9HandlerTyE), |
|     -   [module](api/languages/p  |     [\[9\]](api/languages/cpp_a   |
| ython_api.html#module-cudaq.spin) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [cudaq::amplitude_damping     | opmlERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[10\]](api/language         |
|     cla                           | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ss)](api/languages/cpp_api.html#_ | roduct_opmlERK15scalar_operator), |
| CPPv4N5cudaq17amplitude_dampingE) |     [\[11\]](api/languages/cpp_a  |
| -                                 | pi.html#_CPPv4NKR5cudaq10product_ |
| [cudaq::amplitude_damping_channel | opmlERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[12\]](api/language         |
|     class)](api                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /languages/cpp_api.html#_CPPv4N5c | roduct_opmlERR15scalar_operator), |
| udaq25amplitude_damping_channelE) |     [\[13\]](api/languages/cpp_   |
| -   [cudaq::amplitud              | api.html#_CPPv4NO5cudaq10product_ |
| e_damping_channel::num_parameters | opmlERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[14\]](api/languag          |
|     member)](api/languages/cpp_a  | es/cpp_api.html#_CPPv4NO5cudaq10p |
| pi.html#_CPPv4N5cudaq25amplitude_ | roduct_opmlERK15scalar_operator), |
| damping_channel14num_parametersE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::ampli                 | api.html#_CPPv4NO5cudaq10product_ |
| tude_damping_channel::num_targets | opmlERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[16\]](api/langua           |
|     member)](api/languages/cp     | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| p_api.html#_CPPv4N5cudaq25amplitu | product_opmlERR15scalar_operator) |
| de_damping_channel11num_targetsE) | -                                 |
| -   [cudaq::AnalogRemoteRESTQPU   |   [cudaq::product_op::operator\*= |
|     (C++                          |     (C++                          |
|     class                         |     function)](api/languages/cpp  |
| )](api/languages/cpp_api.html#_CP | _api.html#_CPPv4N5cudaq10product_ |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | opmLERK10product_opI9HandlerTyE), |
| -   [cudaq::apply_noise (C++      |     [\[1\]](api/langua            |
|     function)](api/               | ges/cpp_api.html#_CPPv4N5cudaq10p |
| languages/cpp_api.html#_CPPv4I0Dp | roduct_opmLERK15scalar_operator), |
| EN5cudaq11apply_noiseEvDpRR4Args) |     [\[2\]](api/languages/cp      |
| -   [cudaq::async_result (C++     | p_api.html#_CPPv4N5cudaq10product |
|     c                             | _opmLERR10product_opI9HandlerTyE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::product_op::operator+ |
| #_CPPv4I0EN5cudaq12async_resultE) |     (C++                          |
| -   [cudaq::async_result::get     |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     functi                        | q10product_opplE6sum_opI1TERK15sc |
| on)](api/languages/cpp_api.html#_ | alar_operatorRK10product_opI1TE), |
| CPPv4N5cudaq12async_result3getEv) |     [\[1\]](api/                  |
| -   [cudaq::async_sample_result   | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     type                          | K15scalar_operatorRK6sum_opI1TE), |
| )](api/languages/cpp_api.html#_CP |     [\[2\]](api/langu             |
| Pv4N5cudaq19async_sample_resultE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::BaseNvcfSimulatorQPU  | q10product_opplE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     class)                        |     [\[3\]](api/                  |
| ](api/languages/cpp_api.html#_CPP | languages/cpp_api.html#_CPPv4I0EN |
| v4N5cudaq20BaseNvcfSimulatorQPUE) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::BaseRemoteRESTQPU     | K15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[4\]](api/langu             |
|     cla                           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ss)](api/languages/cpp_api.html#_ | q10product_opplE6sum_opI1TERR15sc |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | alar_operatorRK10product_opI1TE), |
| -                                 |     [\[5\]](api/                  |
|    [cudaq::BaseRemoteSimulatorQPU | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     class)](                      | R15scalar_operatorRK6sum_opI1TE), |
| api/languages/cpp_api.html#_CPPv4 |     [\[6\]](api/langu             |
| N5cudaq22BaseRemoteSimulatorQPUE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::bit_flip_channel (C++ | q10product_opplE6sum_opI1TERR15sc |
|     cl                            | alar_operatorRR10product_opI1TE), |
| ass)](api/languages/cpp_api.html# |     [\[7\]](api/                  |
| _CPPv4N5cudaq16bit_flip_channelE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq:                       | 5cudaq10product_opplE6sum_opI1TER |
| :bit_flip_channel::num_parameters | R15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[8\]](api/languages/cpp_a   |
|     member)](api/langua           | pi.html#_CPPv4NKR5cudaq10product_ |
| ges/cpp_api.html#_CPPv4N5cudaq16b | opplERK10product_opI9HandlerTyE), |
| it_flip_channel14num_parametersE) |     [\[9\]](api/language          |
| -   [cud                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| aq::bit_flip_channel::num_targets | roduct_opplERK15scalar_operator), |
|     (C++                          |     [\[10\]](api/languages/       |
|     member)](api/lan              | cpp_api.html#_CPPv4NKR5cudaq10pro |
| guages/cpp_api.html#_CPPv4N5cudaq | duct_opplERK6sum_opI9HandlerTyE), |
| 16bit_flip_channel11num_targetsE) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq::boson_handler (C++    | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|  class)](api/languages/cpp_api.ht |     [\[12\]](api/language         |
| ml#_CPPv4N5cudaq13boson_handlerE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::boson_op (C++         | roduct_opplERR15scalar_operator), |
|     type)](api/languages/cpp_     |     [\[13\]](api/languages/       |
| api.html#_CPPv4N5cudaq8boson_opE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::boson_op_term (C++    | duct_opplERR6sum_opI9HandlerTyE), |
|                                   |     [\[                           |
|   type)](api/languages/cpp_api.ht | 14\]](api/languages/cpp_api.html# |
| ml#_CPPv4N5cudaq13boson_op_termE) | _CPPv4NKR5cudaq10product_opplEv), |
| -   [cudaq::CodeGenConfig (C++    |     [\[15\]](api/languages/cpp_   |
|                                   | api.html#_CPPv4NO5cudaq10product_ |
| struct)](api/languages/cpp_api.ht | opplERK10product_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     [\[16\]](api/languag          |
| -   [cudaq::commutation_relations | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opplERK15scalar_operator), |
|     struct)]                      |     [\[17\]](api/languages        |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4NO5cudaq10pro |
| 4N5cudaq21commutation_relationsE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq::complex (C++          |     [\[18\]](api/languages/cpp_   |
|     type)](api/languages/cpp      | api.html#_CPPv4NO5cudaq10product_ |
| _api.html#_CPPv4N5cudaq7complexE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::complex_matrix (C++   |     [\[19\]](api/languag          |
|                                   | es/cpp_api.html#_CPPv4NO5cudaq10p |
| class)](api/languages/cpp_api.htm | roduct_opplERR15scalar_operator), |
| l#_CPPv4N5cudaq14complex_matrixE) |     [\[20\]](api/languages        |
| -                                 | /cpp_api.html#_CPPv4NO5cudaq10pro |
|   [cudaq::complex_matrix::adjoint | duct_opplERR6sum_opI9HandlerTyE), |
|     (C++                          |     [                             |
|     function)](a                  | \[21\]](api/languages/cpp_api.htm |
| pi/languages/cpp_api.html#_CPPv4N | l#_CPPv4NO5cudaq10product_opplEv) |
| 5cudaq14complex_matrix7adjointEv) | -   [cudaq::product_op::operator- |
| -   [cudaq::                      |     (C++                          |
| complex_matrix::diagonal_elements |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     function)](api/languages      | q10product_opmiE6sum_opI1TERK15sc |
| /cpp_api.html#_CPPv4NK5cudaq14com | alar_operatorRK10product_opI1TE), |
| plex_matrix17diagonal_elementsEi) |     [\[1\]](api/                  |
| -   [cudaq::complex_matrix::dump  | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/language       | K15scalar_operatorRK6sum_opI1TE), |
| s/cpp_api.html#_CPPv4NK5cudaq14co |     [\[2\]](api/langu             |
| mplex_matrix4dumpERNSt7ostreamE), | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     [\[1\]]                       | q10product_opmiE6sum_opI1TERK15sc |
| (api/languages/cpp_api.html#_CPPv | alar_operatorRR10product_opI1TE), |
| 4NK5cudaq14complex_matrix4dumpEv) |     [\[3\]](api/                  |
| -   [c                            | languages/cpp_api.html#_CPPv4I0EN |
| udaq::complex_matrix::eigenvalues | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRR6sum_opI1TE), |
|     function)](api/lan            |     [\[4\]](api/langu             |
| guages/cpp_api.html#_CPPv4NK5cuda | ages/cpp_api.html#_CPPv4I0EN5cuda |
| q14complex_matrix11eigenvaluesEv) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cu                           | alar_operatorRK10product_opI1TE), |
| daq::complex_matrix::eigenvectors |     [\[5\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/lang           | 5cudaq10product_opmiE6sum_opI1TER |
| uages/cpp_api.html#_CPPv4NK5cudaq | R15scalar_operatorRK6sum_opI1TE), |
| 14complex_matrix12eigenvectorsEv) |     [\[6\]](api/langu             |
| -   [c                            | ages/cpp_api.html#_CPPv4I0EN5cuda |
| udaq::complex_matrix::exponential | q10product_opmiE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     function)](api/la             |     [\[7\]](api/                  |
| nguages/cpp_api.html#_CPPv4N5cuda | languages/cpp_api.html#_CPPv4I0EN |
| q14complex_matrix11exponentialEv) | 5cudaq10product_opmiE6sum_opI1TER |
| -                                 | R15scalar_operatorRR6sum_opI1TE), |
|  [cudaq::complex_matrix::identity |     [\[8\]](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4NKR5cudaq10product_ |
|     function)](api/languages      | opmiERK10product_opI9HandlerTyE), |
| /cpp_api.html#_CPPv4N5cudaq14comp |     [\[9\]](api/language          |
| lex_matrix8identityEKNSt6size_tE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -                                 | roduct_opmiERK15scalar_operator), |
| [cudaq::complex_matrix::kronecker |     [\[10\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     function)](api/lang           | duct_opmiERK6sum_opI9HandlerTyE), |
| uages/cpp_api.html#_CPPv4I00EN5cu |     [\[11\]](api/languages/cpp_a  |
| daq14complex_matrix9kroneckerE14c | pi.html#_CPPv4NKR5cudaq10product_ |
| omplex_matrix8Iterable8Iterable), | opmiERR10product_opI9HandlerTyE), |
|     [\[1\]](api/l                 |     [\[12\]](api/language         |
| anguages/cpp_api.html#_CPPv4N5cud | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| aq14complex_matrix9kroneckerERK14 | roduct_opmiERR15scalar_operator), |
| complex_matrixRK14complex_matrix) |     [\[13\]](api/languages/       |
| -   [cudaq::c                     | cpp_api.html#_CPPv4NKR5cudaq10pro |
| omplex_matrix::minimal_eigenvalue | duct_opmiERR6sum_opI9HandlerTyE), |
|     (C++                          |     [\[                           |
|     function)](api/languages/     | 14\]](api/languages/cpp_api.html# |
| cpp_api.html#_CPPv4NK5cudaq14comp | _CPPv4NKR5cudaq10product_opmiEv), |
| lex_matrix18minimal_eigenvalueEv) |     [\[15\]](api/languages/cpp_   |
| -   [                             | api.html#_CPPv4NO5cudaq10product_ |
| cudaq::complex_matrix::operator() | opmiERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[16\]](api/languag          |
|     function)](api/languages/cpp  | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _api.html#_CPPv4N5cudaq14complex_ | roduct_opmiERK15scalar_operator), |
| matrixclENSt6size_tENSt6size_tE), |     [\[17\]](api/languages        |
|     [\[1\]](api/languages/cpp     | /cpp_api.html#_CPPv4NO5cudaq10pro |
| _api.html#_CPPv4NK5cudaq14complex | duct_opmiERK6sum_opI9HandlerTyE), |
| _matrixclENSt6size_tENSt6size_tE) |     [\[18\]](api/languages/cpp_   |
| -   [                             | api.html#_CPPv4NO5cudaq10product_ |
| cudaq::complex_matrix::operator\* | opmiERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[19\]](api/languag          |
|     function)](api/langua         | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ges/cpp_api.html#_CPPv4N5cudaq14c | roduct_opmiERR15scalar_operator), |
| omplex_matrixmlEN14complex_matrix |     [\[20\]](api/languages        |
| 10value_typeERK14complex_matrix), | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     [\[1\]                        | duct_opmiERR6sum_opI9HandlerTyE), |
| ](api/languages/cpp_api.html#_CPP |     [                             |
| v4N5cudaq14complex_matrixmlERK14c | \[21\]](api/languages/cpp_api.htm |
| omplex_matrixRK14complex_matrix), | l#_CPPv4NO5cudaq10product_opmiEv) |
|                                   | -   [cudaq::product_op::operator/ |
|  [\[2\]](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14complex_matrixm |     function)](api/language       |
| lERK14complex_matrixRKNSt6vectorI | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| N14complex_matrix10value_typeEEE) | roduct_opdvERK15scalar_operator), |
| -                                 |     [\[1\]](api/language          |
| [cudaq::complex_matrix::operator+ | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opdvERR15scalar_operator), |
|     function                      |     [\[2\]](api/languag           |
| )](api/languages/cpp_api.html#_CP | es/cpp_api.html#_CPPv4NO5cudaq10p |
| Pv4N5cudaq14complex_matrixplERK14 | roduct_opdvERK15scalar_operator), |
| complex_matrixRK14complex_matrix) |     [\[3\]](api/langua            |
| -                                 | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| [cudaq::complex_matrix::operator- | product_opdvERR15scalar_operator) |
|     (C++                          | -                                 |
|     function                      |    [cudaq::product_op::operator/= |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq14complex_matrixmiERK14 |     function)](api/langu          |
| complex_matrixRK14complex_matrix) | ages/cpp_api.html#_CPPv4N5cudaq10 |
| -   [cu                           | product_opdVERK15scalar_operator) |
| daq::complex_matrix::operator\[\] | -   [cudaq::product_op::operator= |
|     (C++                          |     (C++                          |
|                                   |     function)](api/la             |
|  function)](api/languages/cpp_api | nguages/cpp_api.html#_CPPv4I0_NSt |
| .html#_CPPv4N5cudaq14complex_matr | 11enable_if_tIXaantNSt7is_sameI1T |
| ixixERKNSt6vectorINSt6size_tEEE), | 9HandlerTyE5valueENSt16is_constru |
|     [\[1\]](api/languages/cpp_api | ctibleI9HandlerTy1TE5valueEEbEEEN |
| .html#_CPPv4NK5cudaq14complex_mat | 5cudaq10product_opaSER10product_o |
| rixixERKNSt6vectorINSt6size_tEEE) | pI9HandlerTyERK10product_opI1TE), |
| -   [cudaq::complex_matrix::power |     [\[1\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4N5cudaq10product_ |
|     function)]                    | opaSERK10product_opI9HandlerTyE), |
| (api/languages/cpp_api.html#_CPPv |     [\[2\]](api/languages/cp      |
| 4N5cudaq14complex_matrix5powerEi) | p_api.html#_CPPv4N5cudaq10product |
| -                                 | _opaSERR10product_opI9HandlerTyE) |
|  [cudaq::complex_matrix::set_zero | -                                 |
|     (C++                          |    [cudaq::product_op::operator== |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languages/cpp  |
| cudaq14complex_matrix8set_zeroEv) | _api.html#_CPPv4NK5cudaq10product |
| -                                 | _opeqERK10product_opI9HandlerTyE) |
| [cudaq::complex_matrix::to_string | -                                 |
|     (C++                          |  [cudaq::product_op::operator\[\] |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     function)](ap                 |
| udaq14complex_matrix9to_stringEv) | i/languages/cpp_api.html#_CPPv4NK |
| -   [                             | 5cudaq10product_opixENSt6size_tE) |
| cudaq::complex_matrix::value_type | -                                 |
|     (C++                          |    [cudaq::product_op::product_op |
|     type)](api/                   |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/languages/c    |
| daq14complex_matrix10value_typeE) | pp_api.html#_CPPv4I0_NSt11enable_ |
| -   [cudaq::contrib (C++          | if_tIXaaNSt7is_sameI9HandlerTy14m |
|     type)](api/languages/cpp      | atrix_handlerE5valueEaantNSt7is_s |
| _api.html#_CPPv4N5cudaq7contribE) | ameI1T9HandlerTyE5valueENSt16is_c |
| -   [cudaq::contrib::draw (C++    | onstructibleI9HandlerTy1TE5valueE |
|     function)]                    | EbEEEN5cudaq10product_op10product |
| (api/languages/cpp_api.html#_CPPv | _opERK10product_opI1TERKN14matrix |
| 4I0Dp0EN5cudaq7contrib4drawENSt6s | _handler20commutation_behaviorE), |
| tringERR13QuantumKernelDpRR4Args) |                                   |
| -                                 |  [\[1\]](api/languages/cpp_api.ht |
| [cudaq::contrib::get_unitary_cmat | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|     (C++                          | tNSt7is_sameI1T9HandlerTyE5valueE |
|     function)](api/languages/cp   | NSt16is_constructibleI9HandlerTy1 |
| p_api.html#_CPPv4I0DpEN5cudaq7con | TE5valueEEbEEEN5cudaq10product_op |
| trib16get_unitary_cmatE14complex_ | 10product_opERK10product_opI1TE), |
| matrixRR13QuantumKernelDpRR4Args) |                                   |
| -   [cudaq::CusvState (C++        |   [\[2\]](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq10product_op10pr |
|    class)](api/languages/cpp_api. | oduct_opENSt6size_tENSt6size_tE), |
| html#_CPPv4I0EN5cudaq9CusvStateE) |     [\[3\]](api/languages/cp      |
| -   [cudaq::depolarization1 (C++  | p_api.html#_CPPv4N5cudaq10product |
|     c                             | _op10product_opENSt7complexIdEE), |
| lass)](api/languages/cpp_api.html |     [\[4\]](api/l                 |
| #_CPPv4N5cudaq15depolarization1E) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::depolarization2 (C++  | aq10product_op10product_opERK10pr |
|     c                             | oduct_opI9HandlerTyENSt6size_tE), |
| lass)](api/languages/cpp_api.html |     [\[5\]](api/l                 |
| #_CPPv4N5cudaq15depolarization2E) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq:                       | aq10product_op10product_opERR10pr |
| :depolarization2::depolarization2 | oduct_opI9HandlerTyENSt6size_tE), |
|     (C++                          |     [\[6\]](api/languages         |
|     function)](api/languages/cp   | /cpp_api.html#_CPPv4N5cudaq10prod |
| p_api.html#_CPPv4N5cudaq15depolar | uct_op10product_opERR9HandlerTy), |
| ization215depolarization2EK4real) |     [\[7\]](ap                    |
| -   [cudaq                        | i/languages/cpp_api.html#_CPPv4N5 |
| ::depolarization2::num_parameters | cudaq10product_op10product_opEd), |
|     (C++                          |     [\[8\]](a                     |
|     member)](api/langu            | pi/languages/cpp_api.html#_CPPv4N |
| ages/cpp_api.html#_CPPv4N5cudaq15 | 5cudaq10product_op10product_opEv) |
| depolarization214num_parametersE) | -   [cuda                         |
| -   [cu                           | q::product_op::to_diagonal_matrix |
| daq::depolarization2::num_targets |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](api/la               | languages/cpp_api.html#_CPPv4NK5c |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq10product_op18to_diagonal_mat |
| q15depolarization211num_targetsE) | rixENSt13unordered_mapINSt6size_t |
| -                                 | ENSt7int64_tEEERKNSt13unordered_m |
|    [cudaq::depolarization_channel | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cudaq::product_op::to_matrix |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     funct                         |
| N5cudaq22depolarization_channelE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::depol                 | _CPPv4NK5cudaq10product_op9to_mat |
| arization_channel::num_parameters | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     member)](api/languages/cp     | apINSt6stringENSt7complexIdEEEEb) |
| p_api.html#_CPPv4N5cudaq22depolar | -   [cu                           |
| ization_channel14num_parametersE) | daq::product_op::to_sparse_matrix |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     member)](api/languages        | 5cudaq10product_op16to_sparse_mat |
| /cpp_api.html#_CPPv4N5cudaq22depo | rixENSt13unordered_mapINSt6size_t |
| larization_channel11num_targetsE) | ENSt7int64_tEEERKNSt13unordered_m |
| -   [cudaq::details (C++          | apINSt6stringENSt7complexIdEEEEb) |
|     type)](api/languages/cpp      | -   [cudaq::product_op::to_string |
| _api.html#_CPPv4N5cudaq7detailsE) |     (C++                          |
| -   [cudaq::details::future (C++  |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|  class)](api/languages/cpp_api.ht | NK5cudaq10product_op9to_stringEv) |
| ml#_CPPv4N5cudaq7details6futureE) | -                                 |
| -                                 |  [cudaq::product_op::\~product_op |
|   [cudaq::details::future::future |     (C++                          |
|     (C++                          |     fu                            |
|     functio                       | nction)](api/languages/cpp_api.ht |
| n)](api/languages/cpp_api.html#_C | ml#_CPPv4N5cudaq10product_opD0Ev) |
| PPv4N5cudaq7details6future6future | -   [cudaq::QPU (C++              |
| ERNSt6vectorI3JobEERNSt6stringERN |     class)](api/languages         |
| St3mapINSt6stringENSt6stringEEE), | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     [\[1\]](api/lang              | -   [cudaq::QPU::enqueue (C++     |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     function)](ap                 |
| details6future6futureERR6future), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[2\]]                       | cudaq3QPU7enqueueER11QuantumTask) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::QPU::getConnectivity  |
| 4N5cudaq7details6future6futureEv) |     (C++                          |
| -   [cu                           |     function)                     |
| daq::details::kernel_builder_base | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq3QPU15getConnectivityEv) |
|     class)](api/l                 | -                                 |
| anguages/cpp_api.html#_CPPv4N5cud | [cudaq::QPU::getExecutionThreadId |
| aq7details19kernel_builder_baseE) |     (C++                          |
| -   [cudaq::details::             |     function)](api/               |
| kernel_builder_base::operator\<\< | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq3QPU20getExecutionThreadIdEv) |
|     function)](api/langua         | -   [cudaq::QPU::getNumQubits     |
| ges/cpp_api.html#_CPPv4N5cudaq7de |     (C++                          |
| tails19kernel_builder_baselsERNSt |     functi                        |
| 7ostreamERK19kernel_builder_base) | on)](api/languages/cpp_api.html#_ |
| -   [                             | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| cudaq::details::KernelBuilderType | -   [                             |
|     (C++                          | cudaq::QPU::getRemoteCapabilities |
|     class)](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)](api/l              |
| udaq7details17KernelBuilderTypeE) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cudaq::d                     | daq3QPU21getRemoteCapabilitiesEv) |
| etails::KernelBuilderType::create | -   [cudaq::QPU::isEmulated (C++  |
|     (C++                          |     func                          |
|     function)                     | tion)](api/languages/cpp_api.html |
| ](api/languages/cpp_api.html#_CPP | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| v4N5cudaq7details17KernelBuilderT | -   [cudaq::QPU::isSimulator (C++ |
| ype6createEPN4mlir11MLIRContextE) |     funct                         |
| -   [cudaq::details::Ker          | ion)](api/languages/cpp_api.html# |
| nelBuilderType::KernelBuilderType | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     (C++                          | -   [cudaq::QPU::launchKernel     |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     function)](api/               |
| details17KernelBuilderType17Kerne | languages/cpp_api.html#_CPPv4N5cu |
| lBuilderTypeERRNSt8functionIFN4ml | daq3QPU12launchKernelERKNSt6strin |
| ir4TypeEPN4mlir11MLIRContextEEEE) | gE15KernelThunkTypePvNSt8uint64_t |
| -   [cudaq::diag_matrix_callback  | ENSt8uint64_tERKNSt6vectorIPvEE), |
|     (C++                          |                                   |
|     class)                        |  [\[1\]](api/languages/cpp_api.ht |
| ](api/languages/cpp_api.html#_CPP | ml#_CPPv4N5cudaq3QPU12launchKerne |
| v4N5cudaq20diag_matrix_callbackE) | lERKNSt6stringERKNSt6vectorIPvEE) |
| -   [cudaq::dyn (C++              | -   [cudaq::Q                     |
|     member)](api/languages        | PU::launchSerializedCodeExecution |
| /cpp_api.html#_CPPv4N5cudaq3dynE) |     (C++                          |
| -   [cudaq::ExecutionContext (C++ |     function)]                    |
|     cl                            | (api/languages/cpp_api.html#_CPPv |
| ass)](api/languages/cpp_api.html# | 4N5cudaq3QPU29launchSerializedCod |
| _CPPv4N5cudaq16ExecutionContextE) | eExecutionERKNSt6stringERN5cudaq3 |
| -   [cudaq                        | 0SerializedCodeExecutionContextE) |
| ::ExecutionContext::amplitudeMaps | -   [cudaq::QPU::onRandomSeedSet  |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api/lang           |
| ages/cpp_api.html#_CPPv4N5cudaq16 | uages/cpp_api.html#_CPPv4N5cudaq3 |
| ExecutionContext13amplitudeMapsE) | QPU15onRandomSeedSetENSt6size_tE) |
| -   [c                            | -   [cudaq::QPU::QPU (C++         |
| udaq::ExecutionContext::asyncExec |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](api/                 | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq16ExecutionContext9asyncExecE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cud                          | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| aq::ExecutionContext::asyncResult |     [\[2\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     member)](api/lan              | -   [                             |
| guages/cpp_api.html#_CPPv4N5cudaq | cudaq::QPU::resetExecutionContext |
| 16ExecutionContext11asyncResultE) |     (C++                          |
| -   [cudaq:                       |     function)](api/               |
| :ExecutionContext::batchIteration | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq3QPU21resetExecutionContextEv) |
|     member)](api/langua           | -                                 |
| ges/cpp_api.html#_CPPv4N5cudaq16E |  [cudaq::QPU::setExecutionContext |
| xecutionContext14batchIterationE) |     (C++                          |
| -   [cudaq::E                     |                                   |
| xecutionContext::canHandleObserve |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4N5cudaq3QPU19setExec |
|     member)](api/language         | utionContextEP16ExecutionContext) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::QPU::setId (C++       |
| cutionContext16canHandleObserveE) |     function                      |
| -   [cudaq::E                     | )](api/languages/cpp_api.html#_CP |
| xecutionContext::ExecutionContext | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::setShots (C++    |
|     func                          |     f                             |
| tion)](api/languages/cpp_api.html | unction)](api/languages/cpp_api.h |
| #_CPPv4N5cudaq16ExecutionContext1 | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| 6ExecutionContextERKNSt6stringE), | -   [cudaq:                       |
|     [\[1\]](api                   | :QPU::supportsConditionalFeedback |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq16ExecutionContext16Execution |     function)](api/langua         |
| ContextERKNSt6stringENSt6size_tE) | ges/cpp_api.html#_CPPv4N5cudaq3QP |
| -   [cudaq::E                     | U27supportsConditionalFeedbackEv) |
| xecutionContext::expectationValue | -   [cudaq::                      |
|     (C++                          | QPU::supportsExplicitMeasurements |
|     member)](api/language         |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     function)](api/languag        |
| cutionContext16expectationValueE) | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| -   [cudaq::Execu                 | 28supportsExplicitMeasurementsEv) |
| tionContext::explicitMeasurements | -   [cudaq::QPU::\~QPU (C++       |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api/languages/cp     | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::QPUState (C++         |
| onContext20explicitMeasurementsE) |     class)](api/languages/cpp_    |
| -   [cuda                         | api.html#_CPPv4N5cudaq8QPUStateE) |
| q::ExecutionContext::futureResult | -   [cudaq::qreg (C++             |
|     (C++                          |     class)](api/lang              |
|     member)](api/lang             | uages/cpp_api.html#_CPPv4I_NSt6si |
| uages/cpp_api.html#_CPPv4N5cudaq1 | ze_tE_NSt6size_tE0EN5cudaq4qregE) |
| 6ExecutionContext12futureResultE) | -   [cudaq::qreg::back (C++       |
| -   [cudaq::ExecutionContext      |     function)                     |
| ::hasConditionalsOnMeasureResults | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4backENSt6size_tE), |
|     mem                           |     [\[1\]](api/languages/cpp_ap  |
| ber)](api/languages/cpp_api.html# | i.html#_CPPv4N5cudaq4qreg4backEv) |
| _CPPv4N5cudaq16ExecutionContext31 | -   [cudaq::qreg::begin (C++      |
| hasConditionalsOnMeasureResultsE) |                                   |
| -   [cudaq::Executi               |  function)](api/languages/cpp_api |
| onContext::invocationResultBuffer | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     (C++                          | -   [cudaq::qreg::clear (C++      |
|     member)](api/languages/cpp_   |                                   |
| api.html#_CPPv4N5cudaq16Execution |  function)](api/languages/cpp_api |
| Context22invocationResultBufferE) | .html#_CPPv4N5cudaq4qreg5clearEv) |
| -   [cu                           | -   [cudaq::qreg::front (C++      |
| daq::ExecutionContext::kernelName |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     member)](api/la               | 4N5cudaq4qreg5frontENSt6size_tE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[1\]](api/languages/cpp_api |
| q16ExecutionContext10kernelNameE) | .html#_CPPv4N5cudaq4qreg5frontEv) |
| -   [cud                          | -   [cudaq::qreg::operator\[\]    |
| aq::ExecutionContext::kernelTrace |     (C++                          |
|     (C++                          |     functi                        |
|     member)](api/lan              | on)](api/languages/cpp_api.html#_ |
| guages/cpp_api.html#_CPPv4N5cudaq | CPPv4N5cudaq4qregixEKNSt6size_tE) |
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
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::registerNames |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/langu            | v4N5cudaq10QuakeValue8isStdVecEv) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -                                 |
| ExecutionContext13registerNamesE) |    [cudaq::QuakeValue::operator\* |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::reorderIdx |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/la               | udaq10QuakeValuemlE10QuakeValue), |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q16ExecutionContext10reorderIdxE) | [\[1\]](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|  [cudaq::ExecutionContext::result | -   [cudaq::QuakeValue::operator+ |
|     (C++                          |     (C++                          |
|     member)](a                    |     function)](api                |
| pi/languages/cpp_api.html#_CPPv4N | /languages/cpp_api.html#_CPPv4N5c |
| 5cudaq16ExecutionContext6resultE) | udaq10QuakeValueplE10QuakeValue), |
| -                                 |     [                             |
|   [cudaq::ExecutionContext::shots | \[1\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     member)](                     |                                   |
| api/languages/cpp_api.html#_CPPv4 | [\[2\]](api/languages/cpp_api.htm |
| N5cudaq16ExecutionContext5shotsE) | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| -   [cudaq::                      | -   [cudaq::QuakeValue::operator- |
| ExecutionContext::simulationState |     (C++                          |
|     (C++                          |     function)](api                |
|     member)](api/languag          | /languages/cpp_api.html#_CPPv4N5c |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | udaq10QuakeValuemiE10QuakeValue), |
| ecutionContext15simulationStateE) |     [                             |
| -                                 | \[1\]](api/languages/cpp_api.html |
|    [cudaq::ExecutionContext::spin | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     (C++                          |     [                             |
|     member)]                      | \[2\]](api/languages/cpp_api.html |
| (api/languages/cpp_api.html#_CPPv | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| 4N5cudaq16ExecutionContext4spinE) |                                   |
| -   [cudaq::                      | [\[3\]](api/languages/cpp_api.htm |
| ExecutionContext::totalIterations | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     (C++                          | -   [cudaq::QuakeValue::operator/ |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api                |
| ecutionContext15totalIterationsE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::ExecutionResult (C++  | udaq10QuakeValuedvE10QuakeValue), |
|     st                            |                                   |
| ruct)](api/languages/cpp_api.html | [\[1\]](api/languages/cpp_api.htm |
| #_CPPv4N5cudaq15ExecutionResultE) | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| -   [cud                          | -                                 |
| aq::ExecutionResult::appendResult |  [cudaq::QuakeValue::operator\[\] |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api                |
| n)](api/languages/cpp_api.html#_C | /languages/cpp_api.html#_CPPv4N5c |
| PPv4N5cudaq15ExecutionResult12app | udaq10QuakeValueixEKNSt6size_tE), |
| endResultENSt6stringENSt6size_tE) |     [\[1\]](api/                  |
| -   [cu                           | languages/cpp_api.html#_CPPv4N5cu |
| daq::ExecutionResult::deserialize | daq10QuakeValueixERK10QuakeValue) |
|     (C++                          | -                                 |
|     function)                     |    [cudaq::QuakeValue::QuakeValue |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq15ExecutionResult11deser |     function)](api/languag        |
| ializeERNSt6vectorINSt6size_tEEE) | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| -   [cudaq:                       | akeValue10QuakeValueERN4mlir20Imp |
| :ExecutionResult::ExecutionResult | licitLocOpBuilderEN4mlir5ValueE), |
|     (C++                          |     [\[1\]                        |
|     functio                       | ](api/languages/cpp_api.html#_CPP |
| n)](api/languages/cpp_api.html#_C | v4N5cudaq10QuakeValue10QuakeValue |
| PPv4N5cudaq15ExecutionResult15Exe | ERN4mlir20ImplicitLocOpBuilderEd) |
| cutionResultE16CountsDictionary), | -   [cudaq::QuakeValue::size (C++ |
|     [\[1\]](api/lan               |     funct                         |
| guages/cpp_api.html#_CPPv4N5cudaq | ion)](api/languages/cpp_api.html# |
| 15ExecutionResult15ExecutionResul | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| tE16CountsDictionaryNSt6stringE), | -   [cudaq::QuakeValue::slice     |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     function)](api/languages/cpp_ |
| Pv4N5cudaq15ExecutionResult15Exec | api.html#_CPPv4N5cudaq10QuakeValu |
| utionResultE16CountsDictionaryd), | e5sliceEKNSt6size_tEKNSt6size_tE) |
|                                   | -   [cudaq::quantum_platform (C++ |
|    [\[3\]](api/languages/cpp_api. |     cl                            |
| html#_CPPv4N5cudaq15ExecutionResu | ass)](api/languages/cpp_api.html# |
| lt15ExecutionResultENSt6stringE), | _CPPv4N5cudaq16quantum_platformE) |
|     [\[4\                         | -   [cud                          |
| ]](api/languages/cpp_api.html#_CP | aq::quantum_platform::clear_shots |
| Pv4N5cudaq15ExecutionResult15Exec |     (C++                          |
| utionResultERK15ExecutionResult), |     function)](api/lang           |
|     [\[5\]](api/language          | uages/cpp_api.html#_CPPv4N5cudaq1 |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | 6quantum_platform11clear_shotsEv) |
| cutionResult15ExecutionResultEd), | -   [cuda                         |
|     [\[6\]](api/languag           | q::quantum_platform::connectivity |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |     (C++                          |
| ecutionResult15ExecutionResultEv) |     function)](api/langu          |
| -   [                             | ages/cpp_api.html#_CPPv4N5cudaq16 |
| cudaq::ExecutionResult::operator= | quantum_platform12connectivityEv) |
|     (C++                          | -   [cudaq::q                     |
|     function)](api/languages/     | uantum_platform::enqueueAsyncTask |
| cpp_api.html#_CPPv4N5cudaq15Execu |     (C++                          |
| tionResultaSERK15ExecutionResult) |     function)](api/languages/     |
| -   [c                            | cpp_api.html#_CPPv4N5cudaq16quant |
| udaq::ExecutionResult::operator== | um_platform16enqueueAsyncTaskEKNS |
|     (C++                          | t6size_tER19KernelExecutionTask), |
|     function)](api/languages/c    |     [\[1\]](api/languag           |
| pp_api.html#_CPPv4NK5cudaq15Execu | es/cpp_api.html#_CPPv4N5cudaq16qu |
| tionResulteqERK15ExecutionResult) | antum_platform16enqueueAsyncTaskE |
| -   [cud                          | KNSt6size_tERNSt8functionIFvvEEE) |
| aq::ExecutionResult::registerName | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_codegen_config |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languages/c    |
| 15ExecutionResult12registerNameE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [cudaq                        | m_platform18get_codegen_configEv) |
| ::ExecutionResult::sequentialData | -   [cudaq::                      |
|     (C++                          | quantum_platform::get_current_qpu |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     function)](api/language       |
| ExecutionResult14sequentialDataE) | s/cpp_api.html#_CPPv4N5cudaq16qua |
| -   [                             | ntum_platform15get_current_qpuEv) |
| cudaq::ExecutionResult::serialize | -   [cuda                         |
|     (C++                          | q::quantum_platform::get_exec_ctx |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4NK5cu |     function)](api/langua         |
| daq15ExecutionResult9serializeEv) | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| -   [cudaq::fermion_handler (C++  | quantum_platform12get_exec_ctxEv) |
|     c                             | -   [c                            |
| lass)](api/languages/cpp_api.html | udaq::quantum_platform::get_noise |
| #_CPPv4N5cudaq15fermion_handlerE) |     (C++                          |
| -   [cudaq::fermion_op (C++       |     function)](api/l              |
|     type)](api/languages/cpp_api  | anguages/cpp_api.html#_CPPv4N5cud |
| .html#_CPPv4N5cudaq10fermion_opE) | aq16quantum_platform9get_noiseEv) |
| -   [cudaq::fermion_op_term (C++  | -   [cudaq:                       |
|                                   | :quantum_platform::get_num_qubits |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15fermion_op_termE) |                                   |
| -   [cudaq::FermioniqBaseQPU (C++ | function)](api/languages/cpp_api. |
|     cl                            | html#_CPPv4N5cudaq16quantum_platf |
| ass)](api/languages/cpp_api.html# | orm14get_num_qubitsENSt6size_tE), |
| _CPPv4N5cudaq16FermioniqBaseQPUE) |     [\[1\]](api/languag           |
| -   [cudaq::get_state (C++        | es/cpp_api.html#_CPPv4N5cudaq16qu |
|                                   | antum_platform14get_num_qubitsEv) |
|    function)](api/languages/cpp_a | -   [cudaq::quantum_              |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | platform::get_remote_capabilities |
| ateEDaRR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::gradient (C++         |     function)]                    |
|     class)](api/languages/cpp_    | (api/languages/cpp_api.html#_CPPv |
| api.html#_CPPv4N5cudaq8gradientE) | 4NK5cudaq16quantum_platform23get_ |
| -   [cudaq::gradient::clone (C++  | remote_capabilitiesEKNSt6size_tE) |
|     fun                           | -   [c                            |
| ction)](api/languages/cpp_api.htm | udaq::quantum_platform::get_shots |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     (C++                          |
| -   [cudaq::gradient::compute     |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     function)](api/language       | aq16quantum_platform9get_shotsEv) |
| s/cpp_api.html#_CPPv4N5cudaq8grad | -   [cuda                         |
| ient7computeERKNSt6vectorIdEERKNS | q::quantum_platform::getLogStream |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|     [\[1\]](ap                    |     function)](api/langu          |
| i/languages/cpp_api.html#_CPPv4N5 | ages/cpp_api.html#_CPPv4N5cudaq16 |
| cudaq8gradient7computeERKNSt6vect | quantum_platform12getLogStreamEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cud                          |
| -   [cudaq::gradient::gradient    | aq::quantum_platform::is_emulated |
|     (C++                          |     (C++                          |
|     function)](api/lang           |                                   |
| uages/cpp_api.html#_CPPv4I00EN5cu |   function)](api/languages/cpp_ap |
| daq8gradient8gradientER7KernelT), | i.html#_CPPv4NK5cudaq16quantum_pl |
|                                   | atform11is_emulatedEKNSt6size_tE) |
|    [\[1\]](api/languages/cpp_api. | -   [c                            |
| html#_CPPv4I00EN5cudaq8gradient8g | udaq::quantum_platform::is_remote |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\                         |     function)](api/languages/cp   |
| ]](api/languages/cpp_api.html#_CP | p_api.html#_CPPv4N5cudaq16quantum |
| Pv4I00EN5cudaq8gradient8gradientE | _platform9is_remoteEKNSt6size_tE) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cuda                         |
|     [\[3                          | q::quantum_platform::is_simulator |
| \]](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq8gradient8gradientERRN |                                   |
| St8functionIFvNSt6vectorIdEEEEE), |  function)](api/languages/cpp_api |
|     [\[                           | .html#_CPPv4NK5cudaq16quantum_pla |
| 4\]](api/languages/cpp_api.html#_ | tform12is_simulatorEKNSt6size_tE) |
| CPPv4N5cudaq8gradient8gradientEv) | -   [c                            |
| -   [cudaq::gradient::setArgs     | udaq::quantum_platform::launchVQE |
|     (C++                          |     (C++                          |
|     fu                            |                                   |
| nction)](api/languages/cpp_api.ht | function)](api/languages/cpp_api. |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | html#_CPPv4N5cudaq16quantum_platf |
| tArgsEvR13QuantumKernelDpRR4Args) | orm9launchVQEEKNSt6stringEPKvPN5c |
| -   [cudaq::gradient::setKernel   | udaq8gradientERKN5cudaq7spin_opER |
|     (C++                          | N5cudaq9optimizerEKiKNSt6size_tE) |
|     function)](api/languages/c    | -   [cudaq:                       |
| pp_api.html#_CPPv4I0EN5cudaq8grad | :quantum_platform::list_platforms |
| ient9setKernelEvR13QuantumKernel) |     (C++                          |
| -   [cud                          |     function)](api/languag        |
| aq::gradients::central_difference | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14list_platformsEv) |
|     class)](api/la                | -                                 |
| nguages/cpp_api.html#_CPPv4N5cuda |    [cudaq::quantum_platform::name |
| q9gradients18central_differenceE) |     (C++                          |
| -   [cudaq::gra                   |     function)](a                  |
| dients::central_difference::clone | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | K5cudaq16quantum_platform4nameEv) |
|     function)](api/languages      | -   [                             |
| /cpp_api.html#_CPPv4N5cudaq9gradi | cudaq::quantum_platform::num_qpus |
| ents18central_difference5cloneEv) |     (C++                          |
| -   [cudaq::gradi                 |     function)](api/l              |
| ents::central_difference::compute | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq16quantum_platform8num_qpusEv) |
|     function)](                   | -   [cudaq::                      |
| api/languages/cpp_api.html#_CPPv4 | quantum_platform::onRandomSeedSet |
| N5cudaq9gradients18central_differ |     (C++                          |
| ence7computeERKNSt6vectorIdEERKNS |                                   |
| t8functionIFdNSt6vectorIdEEEEEd), | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4N5cudaq16quantum_platf |
|   [\[1\]](api/languages/cpp_api.h | orm15onRandomSeedSetENSt6size_tE) |
| tml#_CPPv4N5cudaq9gradients18cent | -   [cudaq:                       |
| ral_difference7computeERKNSt6vect | :quantum_platform::reset_exec_ctx |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradie                |                                   |
| nts::central_difference::gradient |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq16quantum_plat |
|     functio                       | form14reset_exec_ctxENSt6size_tE) |
| n)](api/languages/cpp_api.html#_C | -   [cud                          |
| PPv4I00EN5cudaq9gradients18centra | aq::quantum_platform::reset_noise |
| l_difference8gradientER7KernelT), |     (C++                          |
|     [\[1\]](api/langua            |     function)](api/lang           |
| ges/cpp_api.html#_CPPv4I00EN5cuda | uages/cpp_api.html#_CPPv4N5cudaq1 |
| q9gradients18central_difference8g | 6quantum_platform11reset_noiseEv) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq:                       |
|     [\[2\]](api/languages/cpp_    | :quantum_platform::resetLogStream |
| api.html#_CPPv4I00EN5cudaq9gradie |     (C++                          |
| nts18central_difference8gradientE |     function)](api/languag        |
| RR13QuantumKernelRR10ArgsMapper), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[3\]](api/languages/cpp     | antum_platform14resetLogStreamEv) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::                      |
| 18central_difference8gradientERRN | quantum_platform::set_current_qpu |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[4\]](api/languages/cp      |     f                             |
| p_api.html#_CPPv4N5cudaq9gradient | unction)](api/languages/cpp_api.h |
| s18central_difference8gradientEv) | tml#_CPPv4N5cudaq16quantum_platfo |
| -   [cud                          | rm15set_current_qpuEKNSt6size_tE) |
| aq::gradients::forward_difference | -   [cuda                         |
|     (C++                          | q::quantum_platform::set_exec_ctx |
|     class)](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languages      |
| q9gradients18forward_differenceE) | /cpp_api.html#_CPPv4N5cudaq16quan |
| -   [cudaq::gra                   | tum_platform12set_exec_ctxEPN5cud |
| dients::forward_difference::clone | aq16ExecutionContextENSt6size_tE) |
|     (C++                          | -   [c                            |
|     function)](api/languages      | udaq::quantum_platform::set_noise |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18forward_difference5cloneEv) |                                   |
| -   [cudaq::gradi                 |    function)](api/languages/cpp_a |
| ents::forward_difference::compute | pi.html#_CPPv4N5cudaq16quantum_pl |
|     (C++                          | atform9set_noiseEPK11noise_model) |
|     function)](                   | -   [c                            |
| api/languages/cpp_api.html#_CPPv4 | udaq::quantum_platform::set_shots |
| N5cudaq9gradients18forward_differ |     (C++                          |
| ence7computeERKNSt6vectorIdEERKNS |     function)](api/l              |
| t8functionIFdNSt6vectorIdEEEEEd), | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq16quantum_platform9set_shotsEi) |
|   [\[1\]](api/languages/cpp_api.h | -   [cuda                         |
| tml#_CPPv4N5cudaq9gradients18forw | q::quantum_platform::setLogStream |
| ard_difference7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |                                   |
| -   [cudaq::gradie                |  function)](api/languages/cpp_api |
| nts::forward_difference::gradient | .html#_CPPv4N5cudaq16quantum_plat |
|     (C++                          | form12setLogStreamERNSt7ostreamE) |
|     functio                       | -   [cudaq::q                     |
| n)](api/languages/cpp_api.html#_C | uantum_platform::setTargetBackend |
| PPv4I00EN5cudaq9gradients18forwar |     (C++                          |
| d_difference8gradientER7KernelT), |     fun                           |
|     [\[1\]](api/langua            | ction)](api/languages/cpp_api.htm |
| ges/cpp_api.html#_CPPv4I00EN5cuda | l#_CPPv4N5cudaq16quantum_platform |
| q9gradients18forward_difference8g | 16setTargetBackendERKNSt6stringE) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::quantum_platfo        |
|     [\[2\]](api/languages/cpp_    | rm::supports_conditional_feedback |
| api.html#_CPPv4I00EN5cudaq9gradie |     (C++                          |
| nts18forward_difference8gradientE |     function)](api/l              |
| RR13QuantumKernelRR10ArgsMapper), | anguages/cpp_api.html#_CPPv4NK5cu |
|     [\[3\]](api/languages/cpp     | daq16quantum_platform29supports_c |
| _api.html#_CPPv4N5cudaq9gradients | onditional_feedbackEKNSt6size_tE) |
| 18forward_difference8gradientERRN | -   [cudaq::quantum_platfor       |
| St8functionIFvNSt6vectorIdEEEEE), | m::supports_explicit_measurements |
|     [\[4\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq9gradient |     function)](api/la             |
| s18forward_difference8gradientEv) | nguages/cpp_api.html#_CPPv4NK5cud |
| -   [                             | aq16quantum_platform30supports_ex |
| cudaq::gradients::parameter_shift | plicit_measurementsEKNSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_pla           |
|     class)](api                   | tform::supports_task_distribution |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq9gradients15parameter_shiftE) |     fu                            |
| -   [cudaq::                      | nction)](api/languages/cpp_api.ht |
| gradients::parameter_shift::clone | ml#_CPPv4NK5cudaq16quantum_platfo |
|     (C++                          | rm26supports_task_distributionEv) |
|     function)](api/langua         | -   [cudaq::QuantumTask (C++      |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     type)](api/languages/cpp_api. |
| adients15parameter_shift5cloneEv) | html#_CPPv4N5cudaq11QuantumTaskE) |
| -   [cudaq::gr                    | -   [cudaq::qubit (C++            |
| adients::parameter_shift::compute |     type)](api/languages/c        |
|     (C++                          | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     function                      | -   [cudaq::QubitConnectivity     |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq9gradients15parameter_s |     ty                            |
| hift7computeERKNSt6vectorIdEERKNS | pe)](api/languages/cpp_api.html#_ |
| t8functionIFdNSt6vectorIdEEEEEd), | CPPv4N5cudaq17QubitConnectivityE) |
|     [\[1\]](api/languages/cpp_ap  | -   [cudaq::QubitEdge (C++        |
| i.html#_CPPv4N5cudaq9gradients15p |     type)](api/languages/cpp_a    |
| arameter_shift7computeERKNSt6vect | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::qudit (C++            |
| -   [cudaq::gra                   |     clas                          |
| dients::parameter_shift::gradient | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     func                          | -   [cudaq::qudit::qudit (C++     |
| tion)](api/languages/cpp_api.html |                                   |
| #_CPPv4I00EN5cudaq9gradients15par | function)](api/languages/cpp_api. |
| ameter_shift8gradientER7KernelT), | html#_CPPv4N5cudaq5qudit5quditEv) |
|     [\[1\]](api/lan               | -   [cudaq::qvector (C++          |
| guages/cpp_api.html#_CPPv4I00EN5c |     class)                        |
| udaq9gradients15parameter_shift8g | ](api/languages/cpp_api.html#_CPP |
| radientER7KernelTRR10ArgsMapper), | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     [\[2\]](api/languages/c       | -   [cudaq::qvector::back (C++    |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     function)](a                  |
| dients15parameter_shift8gradientE | pi/languages/cpp_api.html#_CPPv4N |
| RR13QuantumKernelRR10ArgsMapper), | 5cudaq7qvector4backENSt6size_tE), |
|     [\[3\]](api/languages/        |                                   |
| cpp_api.html#_CPPv4N5cudaq9gradie |   [\[1\]](api/languages/cpp_api.h |
| nts15parameter_shift8gradientERRN | tml#_CPPv4N5cudaq7qvector4backEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::qvector::begin (C++   |
|     [\[4\]](api/languages         |     fu                            |
| /cpp_api.html#_CPPv4N5cudaq9gradi | nction)](api/languages/cpp_api.ht |
| ents15parameter_shift8gradientEv) | ml#_CPPv4N5cudaq7qvector5beginEv) |
| -   [cudaq::kernel_builder (C++   | -   [cudaq::qvector::clear (C++   |
|     clas                          |     fu                            |
| s)](api/languages/cpp_api.html#_C | nction)](api/languages/cpp_api.ht |
| PPv4IDpEN5cudaq14kernel_builderE) | ml#_CPPv4N5cudaq7qvector5clearEv) |
| -   [c                            | -   [cudaq::qvector::end (C++     |
| udaq::kernel_builder::constantVal |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function)](api/la             | html#_CPPv4N5cudaq7qvector3endEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::qvector::front (C++   |
| q14kernel_builder11constantValEd) |     function)](ap                 |
| -   [cu                           | i/languages/cpp_api.html#_CPPv4N5 |
| daq::kernel_builder::getArguments | cudaq7qvector5frontENSt6size_tE), |
|     (C++                          |                                   |
|     function)](api/lan            |  [\[1\]](api/languages/cpp_api.ht |
| guages/cpp_api.html#_CPPv4N5cudaq | ml#_CPPv4N5cudaq7qvector5frontEv) |
| 14kernel_builder12getArgumentsEv) | -   [cudaq::qvector::operator=    |
| -   [cu                           |     (C++                          |
| daq::kernel_builder::getNumParams |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)](api/lan            | PPv4N5cudaq7qvectoraSERK7qvector) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qvector::operator\[\] |
| 14kernel_builder12getNumParamsEv) |     (C++                          |
| -   [c                            |     function)                     |
| udaq::kernel_builder::isArgStdVec | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     function)](api/languages/cp   | -   [cudaq::qvector::qvector (C++ |
| p_api.html#_CPPv4N5cudaq14kernel_ |     function)](api/               |
| builder11isArgStdVecENSt6size_tE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cuda                         | daq7qvector7qvectorENSt6size_tE), |
| q::kernel_builder::kernel_builder |     [\[1\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function)](api/languages/cpp_ | 5cudaq7qvector7qvectorERK5state), |
| api.html#_CPPv4N5cudaq14kernel_bu |     [\[2\]](api                   |
| ilder14kernel_builderERNSt6vector | /languages/cpp_api.html#_CPPv4N5c |
| IN7details17KernelBuilderTypeEEE) | udaq7qvector7qvectorERK7qvector), |
| -   [cudaq::kernel_builder::name  |     [\[3\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4N5cudaq7qvector7q |
|     function)                     | vectorERKNSt6vectorI7complexEEb), |
| ](api/languages/cpp_api.html#_CPP |     [\[4\]](ap                    |
| v4N5cudaq14kernel_builder4nameEv) | i/languages/cpp_api.html#_CPPv4N5 |
| -                                 | cudaq7qvector7qvectorERR7qvector) |
|    [cudaq::kernel_builder::qalloc | -   [cudaq::qvector::size (C++    |
|     (C++                          |     fu                            |
|     function)](api/language       | nction)](api/languages/cpp_api.ht |
| s/cpp_api.html#_CPPv4N5cudaq14ker | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| nel_builder6qallocE10QuakeValue), | -   [cudaq::qvector::slice (C++   |
|     [\[1\]](api/language          |     function)](api/language       |
| s/cpp_api.html#_CPPv4N5cudaq14ker | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| nel_builder6qallocEKNSt6size_tE), | tor5sliceENSt6size_tENSt6size_tE) |
|     [\[2                          | -   [cudaq::qvector::value_type   |
| \]](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq14kernel_builder6qallo |     typ                           |
| cERNSt6vectorINSt7complexIdEEEE), | e)](api/languages/cpp_api.html#_C |
|     [\[3\]](                      | PPv4N5cudaq7qvector10value_typeE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qview (C++            |
| N5cudaq14kernel_builder6qallocEv) |     clas                          |
| -   [cudaq::kernel_builder::swap  | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     function)](api/language       | -   [cudaq::qview::value_type     |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     (C++                          |
| 4kernel_builder4swapEvRK10QuakeVa |     t                             |
| lueRK10QuakeValueRK10QuakeValue), | ype)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq5qview10value_typeE) |
| [\[1\]](api/languages/cpp_api.htm | -   [cudaq::range (C++            |
| l#_CPPv4I00EN5cudaq14kernel_build |     func                          |
| er4swapEvRKNSt6vectorI10QuakeValu | tion)](api/languages/cpp_api.html |
| eEERK10QuakeValueRK10QuakeValue), | #_CPPv4I00EN5cudaq5rangeENSt6vect |
|                                   | orI11ElementTypeEE11ElementType), |
| [\[2\]](api/languages/cpp_api.htm |     [\[1\]](api/languages/cpp_a   |
| l#_CPPv4N5cudaq14kernel_builder4s | pi.html#_CPPv4I00EN5cudaq5rangeEN |
| wapERK10QuakeValueRK10QuakeValue) | St6vectorI11ElementTypeEE11Elemen |
| -   [cudaq::KernelExecutionTask   | tType11ElementType11ElementType), |
|     (C++                          |     [                             |
|     type                          | \[2\]](api/languages/cpp_api.html |
| )](api/languages/cpp_api.html#_CP | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| Pv4N5cudaq19KernelExecutionTaskE) | -   [cudaq::real (C++             |
| -   [cudaq::KernelThunkResultType |     type)](api/languages/         |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq4realE) |
|     struct)]                      | -   [cudaq::registry (C++         |
| (api/languages/cpp_api.html#_CPPv |     type)](api/languages/cpp_     |
| 4N5cudaq21KernelThunkResultTypeE) | api.html#_CPPv4N5cudaq8registryE) |
| -   [cudaq::KernelThunkType (C++  | -                                 |
|                                   |  [cudaq::registry::RegisteredType |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     class)](api/                  |
| -   [cudaq::kraus_channel (C++    | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq8registry14RegisteredTypeE) |
|  class)](api/languages/cpp_api.ht | -   [cudaq::RemoteCapabilities    |
| ml#_CPPv4N5cudaq13kraus_channelE) |     (C++                          |
| -   [cudaq::kraus_channel::empty  |     struc                         |
|     (C++                          | t)](api/languages/cpp_api.html#_C |
|     function)]                    | PPv4N5cudaq18RemoteCapabilitiesE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::Remo                  |
| 4NK5cudaq13kraus_channel5emptyEv) | teCapabilities::isRemoteSimulator |
| -   [cudaq::kraus_c               |     (C++                          |
| hannel::generateUnitaryParameters |     member)](api/languages/c      |
|     (C++                          | pp_api.html#_CPPv4N5cudaq18Remote |
|                                   | Capabilities17isRemoteSimulatorE) |
|    function)](api/languages/cpp_a | -   [cudaq::Remot                 |
| pi.html#_CPPv4N5cudaq13kraus_chan | eCapabilities::RemoteCapabilities |
| nel25generateUnitaryParametersEv) |     (C++                          |
| -                                 |     function)](api/languages/cpp  |
|    [cudaq::kraus_channel::get_ops | _api.html#_CPPv4N5cudaq18RemoteCa |
|     (C++                          | pabilities18RemoteCapabilitiesEb) |
|     function)](a                  | -   [cudaq::Remot                 |
| pi/languages/cpp_api.html#_CPPv4N | eCapabilities::serializedCodeExec |
| K5cudaq13kraus_channel7get_opsEv) |     (C++                          |
| -   [cudaq::                      |     member)](api/languages/cp     |
| kraus_channel::is_unitary_mixture | p_api.html#_CPPv4N5cudaq18RemoteC |
|     (C++                          | apabilities18serializedCodeExecE) |
|     function)](api/languages      | -   [cudaq:                       |
| /cpp_api.html#_CPPv4NK5cudaq13kra | :RemoteCapabilities::stateOverlap |
| us_channel18is_unitary_mixtureEv) |     (C++                          |
| -   [cu                           |     member)](api/langua           |
| daq::kraus_channel::kraus_channel | ges/cpp_api.html#_CPPv4N5cudaq18R |
|     (C++                          | emoteCapabilities12stateOverlapE) |
|     function)](api/lang           | -                                 |
| uages/cpp_api.html#_CPPv4IDpEN5cu |   [cudaq::RemoteCapabilities::vqe |
| daq13kraus_channel13kraus_channel |     (C++                          |
| EDpRRNSt16initializer_listI1TEE), |     member)](                     |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|  [\[1\]](api/languages/cpp_api.ht | N5cudaq18RemoteCapabilities3vqeE) |
| ml#_CPPv4N5cudaq13kraus_channel13 | -   [cudaq::RemoteSimulationState |
| kraus_channelERK13kraus_channel), |     (C++                          |
|     [\[2\]                        |     class)]                       |
| ](api/languages/cpp_api.html#_CPP | (api/languages/cpp_api.html#_CPPv |
| v4N5cudaq13kraus_channel13kraus_c | 4N5cudaq21RemoteSimulationStateE) |
| hannelERKNSt6vectorI8kraus_opEE), | -   [cudaq::Resources (C++        |
|     [\[3\]                        |     class)](api/languages/cpp_a   |
| ](api/languages/cpp_api.html#_CPP | pi.html#_CPPv4N5cudaq9ResourcesE) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::run (C++              |
| hannelERRNSt6vectorI8kraus_opEE), |     function)]                    |
|     [\[4\]](api/lan               | (api/languages/cpp_api.html#_CPPv |
| guages/cpp_api.html#_CPPv4N5cudaq | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| 13kraus_channel13kraus_channelEv) | 5invoke_result_tINSt7decay_tI13Qu |
| -                                 | antumKernelEEDpNSt7decay_tI4ARGSE |
| [cudaq::kraus_channel::noise_type | EEEEENSt6size_tERN5cudaq11noise_m |
|     (C++                          | odelERR13QuantumKernelDpRR4ARGS), |
|     member)](api                  |     [\[1\]](api/langu             |
| /languages/cpp_api.html#_CPPv4N5c | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| udaq13kraus_channel10noise_typeE) | daq3runENSt6vectorINSt15invoke_re |
| -                                 | sult_tINSt7decay_tI13QuantumKerne |
|  [cudaq::kraus_channel::operator= | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4ARGS) |
|     function)](api/langua         | -   [cudaq::run_async (C++        |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     functio                       |
| raus_channelaSERK13kraus_channel) | n)](api/languages/cpp_api.html#_C |
| -   [c                            | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| udaq::kraus_channel::operator\[\] | tureINSt6vectorINSt15invoke_resul |
|     (C++                          | t_tINSt7decay_tI13QuantumKernelEE |
|     function)](api/l              | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| anguages/cpp_api.html#_CPPv4N5cud | ze_tENSt6size_tERN5cudaq11noise_m |
| aq13kraus_channelixEKNSt6size_tE) | odelERR13QuantumKernelDpRR4ARGS), |
| -                                 |     [\[1\]](api/la                |
| [cudaq::kraus_channel::parameters | nguages/cpp_api.html#_CPPv4I0DpEN |
|     (C++                          | 5cudaq9run_asyncENSt6futureINSt6v |
|     member)](api                  | ectorINSt15invoke_result_tINSt7de |
| /languages/cpp_api.html#_CPPv4N5c | cay_tI13QuantumKernelEEDpNSt7deca |
| udaq13kraus_channel10parametersE) | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| -   [cu                           | ize_tERR13QuantumKernelDpRR4ARGS) |
| daq::kraus_channel::probabilities | -   [cudaq::sample (C++           |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api/la               | p_api.html#_CPPv4I0Dp0EN5cudaq6sa |
| nguages/cpp_api.html#_CPPv4N5cuda | mpleE13sample_resultRK14sample_op |
| q13kraus_channel13probabilitiesE) | tionsRR13QuantumKernelDpRR4Args), |
| -                                 |     [\[1\]                        |
|  [cudaq::kraus_channel::push_back | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4I0Dp0EN5cudaq6sampleE13sample_r |
|     function)](api/langua         | esultRR13QuantumKernelDpRR4Args), |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     [\[                           |
| raus_channel9push_backE8kraus_op) | 2\]](api/languages/cpp_api.html#_ |
| -   [cudaq::kraus_channel::size   | CPPv4I0Dp0EN5cudaq6sampleEDaNSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4Args) |
|     function)                     | -   [cudaq::sample_options (C++   |
| ](api/languages/cpp_api.html#_CPP |     s                             |
| v4NK5cudaq13kraus_channel4sizeEv) | truct)](api/languages/cpp_api.htm |
| -   [                             | l#_CPPv4N5cudaq14sample_optionsE) |
| cudaq::kraus_channel::unitary_ops | -   [cudaq::sample_result (C++    |
|     (C++                          |                                   |
|     member)](api/                 |  class)](api/languages/cpp_api.ht |
| languages/cpp_api.html#_CPPv4N5cu | ml#_CPPv4N5cudaq13sample_resultE) |
| daq13kraus_channel11unitary_opsE) | -   [cudaq::sample_result::append |
| -   [cudaq::kraus_op (C++         |     (C++                          |
|     struct)](api/languages/cpp_   |     function)](api/languages/cpp  |
| api.html#_CPPv4N5cudaq8kraus_opE) | _api.html#_CPPv4N5cudaq13sample_r |
| -   [cudaq::kraus_op::adjoint     | esult6appendER15ExecutionResultb) |
|     (C++                          | -   [cudaq::sample_result::begin  |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)]                    |
| CPPv4NK5cudaq8kraus_op7adjointEv) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::kraus_op::data (C++   | 4N5cudaq13sample_result5beginEv), |
|                                   |     [\[1\]]                       |
|  member)](api/languages/cpp_api.h | (api/languages/cpp_api.html#_CPPv |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | 4NK5cudaq13sample_result5beginEv) |
| -   [cudaq::kraus_op::kraus_op    | -   [cudaq::sample_result::cbegin |
|     (C++                          |     (C++                          |
|     func                          |     function)](                   |
| tion)](api/languages/cpp_api.html | api/languages/cpp_api.html#_CPPv4 |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | NK5cudaq13sample_result6cbeginEv) |
| opERRNSt16initializer_listI1TEE), | -   [cudaq::sample_result::cend   |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     function)                     |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | ](api/languages/cpp_api.html#_CPP |
| pENSt6vectorIN5cudaq7complexEEE), | v4NK5cudaq13sample_result4cendEv) |
|     [\[2\]](api/l                 | -   [cudaq::sample_result::clear  |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq8kraus_op8kraus_opERK8kraus_op) |     function)                     |
| -   [cudaq::kraus_op::nCols (C++  | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq13sample_result5clearEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::sample_result::count  |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     (C++                          |
| -   [cudaq::kraus_op::nRows (C++  |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
| member)](api/languages/cpp_api.ht | NK5cudaq13sample_result5countENSt |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | 11string_viewEKNSt11string_viewE) |
| -   [cudaq::kraus_op::operator=   | -   [                             |
|     (C++                          | cudaq::sample_result::deserialize |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     functio                       |
| v4N5cudaq8kraus_opaSERK8kraus_op) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::kraus_op::precision   | PPv4N5cudaq13sample_result11deser |
|     (C++                          | ializeERNSt6vectorINSt6size_tEEE) |
|     memb                          | -   [cudaq::sample_result::dump   |
| er)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8kraus_op9precisionE) |     function)](api/languag        |
| -   [cudaq::matrix_callback (C++  | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     c                             | ample_result4dumpERNSt7ostreamE), |
| lass)](api/languages/cpp_api.html |     [\[1\]                        |
| #_CPPv4N5cudaq15matrix_callbackE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::matrix_handler (C++   | v4NK5cudaq13sample_result4dumpEv) |
|                                   | -   [cudaq::sample_result::end    |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14matrix_handlerE) |     function                      |
| -   [cudaq::mat                   | )](api/languages/cpp_api.html#_CP |
| rix_handler::commutation_behavior | Pv4N5cudaq13sample_result3endEv), |
|     (C++                          |     [\[1\                         |
|     struct)](api/languages/       | ]](api/languages/cpp_api.html#_CP |
| cpp_api.html#_CPPv4N5cudaq14matri | Pv4NK5cudaq13sample_result3endEv) |
| x_handler20commutation_behaviorE) | -   [                             |
| -                                 | cudaq::sample_result::expectation |
|    [cudaq::matrix_handler::define |     (C++                          |
|     (C++                          |     f                             |
|     function)](a                  | unction)](api/languages/cpp_api.h |
| pi/languages/cpp_api.html#_CPPv4N | tml#_CPPv4NK5cudaq13sample_result |
| 5cudaq14matrix_handler6defineENSt | 11expectationEKNSt11string_viewE) |
| 6stringENSt6vectorINSt7int64_tEEE | -   [c                            |
| RR15matrix_callbackRKNSt13unorder | udaq::sample_result::get_marginal |
| ed_mapINSt6stringENSt6stringEEE), |     (C++                          |
|                                   |     function)](api/languages/cpp_ |
| [\[1\]](api/languages/cpp_api.htm | api.html#_CPPv4NK5cudaq13sample_r |
| l#_CPPv4N5cudaq14matrix_handler6d | esult12get_marginalERKNSt6vectorI |
| efineENSt6stringENSt6vectorINSt7i | NSt6size_tEEEKNSt11string_viewE), |
| nt64_tEEERR15matrix_callbackRR20d |     [\[1\]](api/languages/cpp_    |
| iag_matrix_callbackRKNSt13unorder | api.html#_CPPv4NK5cudaq13sample_r |
| ed_mapINSt6stringENSt6stringEEE), | esult12get_marginalERRKNSt6vector |
|     [\[2\]](                      | INSt6size_tEEEKNSt11string_viewE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cuda                         |
| N5cudaq14matrix_handler6defineENS | q::sample_result::get_total_shots |
| t6stringENSt6vectorINSt7int64_tEE |     (C++                          |
| ERR15matrix_callbackRRNSt13unorde |     function)](api/langua         |
| red_mapINSt6stringENSt6stringEEE) | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| -                                 | sample_result15get_total_shotsEv) |
|   [cudaq::matrix_handler::degrees | -   [cuda                         |
|     (C++                          | q::sample_result::has_even_parity |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4NK |     fun                           |
| 5cudaq14matrix_handler7degreesEv) | ction)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq13sample_result15h |
|  [cudaq::matrix_handler::displace | as_even_parityENSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|     function)](api/language       | q::sample_result::has_expectation |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8displaceENSt6size_tE) |     funct                         |
| -   [cudaq::matrix                | ion)](api/languages/cpp_api.html# |
| _handler::get_expected_dimensions | _CPPv4NK5cudaq13sample_result15ha |
|     (C++                          | s_expectationEKNSt11string_viewE) |
|                                   | -   [cu                           |
|    function)](api/languages/cpp_a | daq::sample_result::most_probable |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     (C++                          |
| ndler23get_expected_dimensionsEv) |     fun                           |
| -   [cudaq::matrix_ha             | ction)](api/languages/cpp_api.htm |
| ndler::get_parameter_descriptions | l#_CPPv4NK5cudaq13sample_result13 |
|     (C++                          | most_probableEKNSt11string_viewE) |
|                                   | -                                 |
| function)](api/languages/cpp_api. | [cudaq::sample_result::operator+= |
| html#_CPPv4NK5cudaq14matrix_handl |     (C++                          |
| er26get_parameter_descriptionsEv) |     function)](api/langua         |
| -   [c                            | ges/cpp_api.html#_CPPv4N5cudaq13s |
| udaq::matrix_handler::instantiate | ample_resultpLERK13sample_result) |
|     (C++                          | -                                 |
|     function)](a                  |  [cudaq::sample_result::operator= |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14matrix_handler11instantia |     function)](api/langua         |
| teENSt6stringERKNSt6vectorINSt6si | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ze_tEEERK20commutation_behavior), | ample_resultaSER13sample_result), |
|     [\[1\]](                      |     [\[1\]](api/langua            |
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
|     [\                            |     fun                           |
| [5\]](api/languages/cpp_api.html# | ction)](api/languages/cpp_api.htm |
| _CPPv4N5cudaq14matrix_handler14ma | l#_CPPv4N5cudaq13sample_result13s |
| trix_handlerERK14matrix_handler), | ample_resultER15ExecutionResult), |
|     [                             |                                   |
| \[6\]](api/languages/cpp_api.html |  [\[1\]](api/languages/cpp_api.ht |
| #_CPPv4N5cudaq14matrix_handler14m | ml#_CPPv4N5cudaq13sample_result13 |
| atrix_handlerERR14matrix_handler) | sample_resultERK13sample_result), |
| -                                 |     [\[2\]](api/l                 |
|  [cudaq::matrix_handler::momentum | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq13sample_result13sample_resultE |
|     function)](api/language       | RNSt6vectorI15ExecutionResultEE), |
| s/cpp_api.html#_CPPv4N5cudaq14mat |                                   |
| rix_handler8momentumENSt6size_tE) |  [\[3\]](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq13sample_result13 |
|    [cudaq::matrix_handler::number | sample_resultERR13sample_result), |
|     (C++                          |     [                             |
|     function)](api/langua         | \[4\]](api/languages/cpp_api.html |
| ges/cpp_api.html#_CPPv4N5cudaq14m | #_CPPv4N5cudaq13sample_result13sa |
| atrix_handler6numberENSt6size_tE) | mple_resultERR15ExecutionResult), |
| -                                 |     [\[5\]](api/la                |
| [cudaq::matrix_handler::operator= | nguages/cpp_api.html#_CPPv4N5cuda |
|     (C++                          | q13sample_result13sample_resultEd |
|     fun                           | RNSt6vectorI15ExecutionResultEE), |
| ction)](api/languages/cpp_api.htm |     [\[6\]](api/lan               |
| l#_CPPv4I0_NSt11enable_if_tIXaant | guages/cpp_api.html#_CPPv4N5cudaq |
| NSt7is_sameI1T14matrix_handlerE5v | 13sample_result13sample_resultEv) |
| alueENSt12is_base_of_vI16operator | -                                 |
| _handler1TEEEbEEEN5cudaq14matrix_ |  [cudaq::sample_result::serialize |
| handleraSER14matrix_handlerRK1T), |     (C++                          |
|     [\[1\]](api/languages         |     function)](api                |
| /cpp_api.html#_CPPv4N5cudaq14matr | /languages/cpp_api.html#_CPPv4NK5 |
| ix_handleraSERK14matrix_handler), | cudaq13sample_result9serializeEv) |
|     [\[2\]](api/language          | -   [cudaq::sample_result::size   |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handleraSERR14matrix_handler) |     function)](api/languages/c    |
| -   [                             | pp_api.html#_CPPv4NK5cudaq13sampl |
| cudaq::matrix_handler::operator== | e_result4sizeEKNSt11string_viewE) |
|     (C++                          | -   [cudaq::sample_result::to_map |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4NK5cudaq14mat |     function)](api/languages/cpp  |
| rix_handlereqERK14matrix_handler) | _api.html#_CPPv4NK5cudaq13sample_ |
| -                                 | result6to_mapEKNSt11string_viewE) |
|    [cudaq::matrix_handler::parity | -   [cuda                         |
|     (C++                          | q::sample_result::\~sample_result |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     funct                         |
| atrix_handler6parityENSt6size_tE) | ion)](api/languages/cpp_api.html# |
| -                                 | _CPPv4N5cudaq13sample_resultD0Ev) |
|  [cudaq::matrix_handler::position | -   [cudaq::scalar_callback (C++  |
|     (C++                          |     c                             |
|     function)](api/language       | lass)](api/languages/cpp_api.html |
| s/cpp_api.html#_CPPv4N5cudaq14mat | #_CPPv4N5cudaq15scalar_callbackE) |
| rix_handler8positionENSt6size_tE) | -   [c                            |
| -   [cudaq::                      | udaq::scalar_callback::operator() |
| matrix_handler::remove_definition |     (C++                          |
|     (C++                          |     function)](api/language       |
|     fu                            | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| nction)](api/languages/cpp_api.ht | alar_callbackclERKNSt13unordered_ |
| ml#_CPPv4N5cudaq14matrix_handler1 | mapINSt6stringENSt7complexIdEEEE) |
| 7remove_definitionERKNSt6stringE) | -   [                             |
| -                                 | cudaq::scalar_callback::operator= |
|   [cudaq::matrix_handler::squeeze |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/languag        | pp_api.html#_CPPv4N5cudaq15scalar |
| es/cpp_api.html#_CPPv4N5cudaq14ma | _callbackaSERK15scalar_callback), |
| trix_handler7squeezeENSt6size_tE) |     [\[1\]](api/languages/        |
| -   [cudaq::m                     | cpp_api.html#_CPPv4N5cudaq15scala |
| atrix_handler::to_diagonal_matrix | r_callbackaSERR15scalar_callback) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/lang           | :scalar_callback::scalar_callback |
| uages/cpp_api.html#_CPPv4NK5cudaq |     (C++                          |
| 14matrix_handler18to_diagonal_mat |     function)](api/languag        |
| rixERNSt13unordered_mapINSt6size_ | es/cpp_api.html#_CPPv4I0_NSt11ena |
| tENSt7int64_tEEERKNSt13unordered_ | ble_if_tINSt16is_invocable_r_vINS |
| mapINSt6stringENSt7complexIdEEEE) | t7complexIdEE8CallableRKNSt13unor |
| -                                 | dered_mapINSt6stringENSt7complexI |
| [cudaq::matrix_handler::to_matrix | dEEEEEEbEEEN5cudaq15scalar_callba |
|     (C++                          | ck15scalar_callbackERR8Callable), |
|     function)                     |     [\[1\                         |
| ](api/languages/cpp_api.html#_CPP | ]](api/languages/cpp_api.html#_CP |
| v4NK5cudaq14matrix_handler9to_mat | Pv4N5cudaq15scalar_callback15scal |
| rixERNSt13unordered_mapINSt6size_ | ar_callbackERK15scalar_callback), |
| tENSt7int64_tEEERKNSt13unordered_ |     [\[2                          |
| mapINSt6stringENSt7complexIdEEEE) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq15scalar_callback15sca |
| [cudaq::matrix_handler::to_string | lar_callbackERR15scalar_callback) |
|     (C++                          | -   [cudaq::scalar_operator (C++  |
|     function)](api/               |     c                             |
| languages/cpp_api.html#_CPPv4NK5c | lass)](api/languages/cpp_api.html |
| udaq14matrix_handler9to_stringEb) | #_CPPv4N5cudaq15scalar_operatorE) |
| -                                 | -                                 |
| [cudaq::matrix_handler::unique_id | [cudaq::scalar_operator::evaluate |
|     (C++                          |     (C++                          |
|     function)](api/               |                                   |
| languages/cpp_api.html#_CPPv4NK5c |    function)](api/languages/cpp_a |
| udaq14matrix_handler9unique_idEv) | pi.html#_CPPv4NK5cudaq15scalar_op |
| -   [cudaq:                       | erator8evaluateERKNSt13unordered_ |
| :matrix_handler::\~matrix_handler | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [cudaq::scalar_ope            |
|     functi                        | rator::get_parameter_descriptions |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     f                             |
| -   [cudaq::matrix_op (C++        | unction)](api/languages/cpp_api.h |
|     type)](api/languages/cpp_a    | tml#_CPPv4NK5cudaq15scalar_operat |
| pi.html#_CPPv4N5cudaq9matrix_opE) | or26get_parameter_descriptionsEv) |
| -   [cudaq::matrix_op_term (C++   | -   [cu                           |
|                                   | daq::scalar_operator::is_constant |
|  type)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14matrix_op_termE) |     function)](api/lang           |
| -                                 | uages/cpp_api.html#_CPPv4NK5cudaq |
|    [cudaq::mdiag_operator_handler | 15scalar_operator11is_constantEv) |
|     (C++                          | -   [c                            |
|     class)](                      | udaq::scalar_operator::operator\* |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22mdiag_operator_handlerE) |     function                      |
| -   [cudaq::mpi (C++              | )](api/languages/cpp_api.html#_CP |
|     type)](api/languages          | Pv4N5cudaq15scalar_operatormlENSt |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::mpi::all_gather (C++  |     [\[1\                         |
|     fu                            | ]](api/languages/cpp_api.html#_CP |
| nction)](api/languages/cpp_api.ht | Pv4N5cudaq15scalar_operatormlENSt |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | 7complexIdEERR15scalar_operator), |
| RNSt6vectorIdEERKNSt6vectorIdEE), |     [\[2\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|   [\[1\]](api/languages/cpp_api.h | operatormlEdRK15scalar_operator), |
| tml#_CPPv4N5cudaq3mpi10all_gather |     [\[3\]](api/languages/cp      |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::mpi::all_reduce (C++  | operatormlEdRR15scalar_operator), |
|                                   |     [\[4\]](api/languages         |
|  function)](api/languages/cpp_api | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | alar_operatormlENSt7complexIdEE), |
| reduceE1TRK1TRK14BinaryFunction), |     [\[5\]](api/languages/cpp     |
|     [\[1\]](api/langu             | _api.html#_CPPv4NKR5cudaq15scalar |
| ages/cpp_api.html#_CPPv4I00EN5cud | _operatormlERK15scalar_operator), |
| aq3mpi10all_reduceE1TRK1TRK4Func) |     [\[6\]]                       |
| -   [cudaq::mpi::broadcast (C++   | (api/languages/cpp_api.html#_CPPv |
|     function)](api/               | 4NKR5cudaq15scalar_operatormlEd), |
| languages/cpp_api.html#_CPPv4N5cu |     [\[7\]](api/language          |
| daq3mpi9broadcastERNSt6stringEi), | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     [\[1\]](api/la                | alar_operatormlENSt7complexIdEE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[8\]](api/languages/cp      |
| q3mpi9broadcastERNSt6vectorIdEEi) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::mpi::finalize (C++    | _operatormlERK15scalar_operator), |
|     f                             |     [\[9\                         |
| unction)](api/languages/cpp_api.h | ]](api/languages/cpp_api.html#_CP |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | Pv4NO5cudaq15scalar_operatormlEd) |
| -   [cudaq::mpi::initialize (C++  | -   [cu                           |
|     function                      | daq::scalar_operator::operator\*= |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq3mpi10initializeEiPPc), |     function)](api/languag        |
|     [                             | es/cpp_api.html#_CPPv4N5cudaq15sc |
| \[1\]](api/languages/cpp_api.html | alar_operatormLENSt7complexIdEE), |
| #_CPPv4N5cudaq3mpi10initializeEv) |     [\[1\]](api/languages/c       |
| -   [cudaq::mpi::is_initialized   | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatormLERK15scalar_operator), |
|     function                      |     [\[2                          |
| )](api/languages/cpp_api.html#_CP | \]](api/languages/cpp_api.html#_C |
| Pv4N5cudaq3mpi14is_initializedEv) | PPv4N5cudaq15scalar_operatormLEd) |
| -   [cudaq::mpi::num_ranks (C++   | -   [                             |
|     fu                            | cudaq::scalar_operator::operator+ |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     function                      |
| -   [cudaq::mpi::rank (C++        | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_operatorplENSt |
|    function)](api/languages/cpp_a | 7complexIdEERK15scalar_operator), |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     [\[1\                         |
| -   [cudaq::noise_model (C++      | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_operatorplENSt |
|    class)](api/languages/cpp_api. | 7complexIdEERR15scalar_operator), |
| html#_CPPv4N5cudaq11noise_modelE) |     [\[2\]](api/languages/cp      |
| -   [cudaq::n                     | p_api.html#_CPPv4N5cudaq15scalar_ |
| oise_model::add_all_qubit_channel | operatorplEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     function)](api                | p_api.html#_CPPv4N5cudaq15scalar_ |
| /languages/cpp_api.html#_CPPv4IDp | operatorplEdRR15scalar_operator), |
| EN5cudaq11noise_model21add_all_qu |     [\[4\]](api/languages         |
| bit_channelEvRK13kraus_channeli), | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     [\[1\]](api/langua            | alar_operatorplENSt7complexIdEE), |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     [\[5\]](api/languages/cpp     |
| oise_model21add_all_qubit_channel | _api.html#_CPPv4NKR5cudaq15scalar |
| ERKNSt6stringERK13kraus_channeli) | _operatorplERK15scalar_operator), |
| -                                 |     [\[6\]]                       |
|  [cudaq::noise_model::add_channel | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatorplEd), |
|     funct                         |     [\[7\]]                       |
| ion)](api/languages/cpp_api.html# | (api/languages/cpp_api.html#_CPPv |
| _CPPv4IDpEN5cudaq11noise_model11a | 4NKR5cudaq15scalar_operatorplEv), |
| dd_channelEvRK15PredicateFuncTy), |     [\[8\]](api/language          |
|     [\[1\]](api/languages/cpp_    | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| api.html#_CPPv4IDpEN5cudaq11noise | alar_operatorplENSt7complexIdEE), |
| _model11add_channelEvRKNSt6vector |     [\[9\]](api/languages/cp      |
| INSt6size_tEEERK13kraus_channel), | p_api.html#_CPPv4NO5cudaq15scalar |
|     [\[2\]](ap                    | _operatorplERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[10\]                       |
| cudaq11noise_model11add_channelER | ](api/languages/cpp_api.html#_CPP |
| KNSt6stringERK15PredicateFuncTy), | v4NO5cudaq15scalar_operatorplEd), |
|                                   |     [\[11\                        |
| [\[3\]](api/languages/cpp_api.htm | ]](api/languages/cpp_api.html#_CP |
| l#_CPPv4N5cudaq11noise_model11add | Pv4NO5cudaq15scalar_operatorplEv) |
| _channelERKNSt6stringERKNSt6vecto | -   [c                            |
| rINSt6size_tEEERK13kraus_channel) | udaq::scalar_operator::operator+= |
| -   [cudaq::noise_model::empty    |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function                      | es/cpp_api.html#_CPPv4N5cudaq15sc |
| )](api/languages/cpp_api.html#_CP | alar_operatorpLENSt7complexIdEE), |
| Pv4NK5cudaq11noise_model5emptyEv) |     [\[1\]](api/languages/c       |
| -                                 | pp_api.html#_CPPv4N5cudaq15scalar |
| [cudaq::noise_model::get_channels | _operatorpLERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     function)](api/l              | \]](api/languages/cpp_api.html#_C |
| anguages/cpp_api.html#_CPPv4I0ENK | PPv4N5cudaq15scalar_operatorpLEd) |
| 5cudaq11noise_model12get_channels | -   [                             |
| ENSt6vectorI13kraus_channelEERKNS | cudaq::scalar_operator::operator- |
| t6vectorINSt6size_tEEERKNSt6vecto |     (C++                          |
| rINSt6size_tEEERKNSt6vectorIdEE), |     function                      |
|     [\[1\]](api/languages/cpp_a   | )](api/languages/cpp_api.html#_CP |
| pi.html#_CPPv4NK5cudaq11noise_mod | Pv4N5cudaq15scalar_operatormiENSt |
| el12get_channelsERKNSt6stringERKN | 7complexIdEERK15scalar_operator), |
| St6vectorINSt6size_tEEERKNSt6vect |     [\[1\                         |
| orINSt6size_tEEERKNSt6vectorIdEE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatormiENSt |
|  [cudaq::noise_model::noise_model | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     function)](api                | p_api.html#_CPPv4N5cudaq15scalar_ |
| /languages/cpp_api.html#_CPPv4N5c | operatormiEdRK15scalar_operator), |
| udaq11noise_model11noise_modelEv) |     [\[3\]](api/languages/cp      |
| -   [cu                           | p_api.html#_CPPv4N5cudaq15scalar_ |
| daq::noise_model::PredicateFuncTy | operatormiEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     type)](api/la                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| nguages/cpp_api.html#_CPPv4N5cuda | alar_operatormiENSt7complexIdEE), |
| q11noise_model15PredicateFuncTyE) |     [\[5\]](api/languages/cpp     |
| -   [cud                          | _api.html#_CPPv4NKR5cudaq15scalar |
| aq::noise_model::register_channel | _operatormiERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function)](api/languages      | (api/languages/cpp_api.html#_CPPv |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | 4NKR5cudaq15scalar_operatormiEd), |
| noise_model16register_channelEvv) |     [\[7\]]                       |
| -   [cudaq::                      | (api/languages/cpp_api.html#_CPPv |
| noise_model::requires_constructor | 4NKR5cudaq15scalar_operatormiEv), |
|     (C++                          |     [\[8\]](api/language          |
|     type)](api/languages/cp       | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| p_api.html#_CPPv4I0DpEN5cudaq11no | alar_operatormiENSt7complexIdEE), |
| ise_model20requires_constructorE) |     [\[9\]](api/languages/cp      |
| -   [cudaq::noise_model_type (C++ | p_api.html#_CPPv4NO5cudaq15scalar |
|     e                             | _operatormiERK15scalar_operator), |
| num)](api/languages/cpp_api.html# |     [\[10\]                       |
| _CPPv4N5cudaq16noise_model_typeE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::no                    | v4NO5cudaq15scalar_operatormiEd), |
| ise_model_type::amplitude_damping |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](api/languages    | Pv4NO5cudaq15scalar_operatormiEv) |
| /cpp_api.html#_CPPv4N5cudaq16nois | -   [c                            |
| e_model_type17amplitude_dampingE) | udaq::scalar_operator::operator-= |
| -   [cudaq::noise_mode            |     (C++                          |
| l_type::amplitude_damping_channel |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     e                             | alar_operatormIENSt7complexIdEE), |
| numerator)](api/languages/cpp_api |     [\[1\]](api/languages/c       |
| .html#_CPPv4N5cudaq16noise_model_ | pp_api.html#_CPPv4N5cudaq15scalar |
| type25amplitude_damping_channelE) | _operatormIERK15scalar_operator), |
| -   [cudaq::n                     |     [\[2                          |
| oise_model_type::bit_flip_channel | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatormIEd) |
|     enumerator)](api/language     | -   [                             |
| s/cpp_api.html#_CPPv4N5cudaq16noi | cudaq::scalar_operator::operator/ |
| se_model_type16bit_flip_channelE) |     (C++                          |
| -   [cudaq::                      |     function                      |
| noise_model_type::depolarization1 | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|     enumerator)](api/languag      | 7complexIdEERK15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[1\                         |
| ise_model_type15depolarization1E) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::                      | Pv4N5cudaq15scalar_operatordvENSt |
| noise_model_type::depolarization2 | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     enumerator)](api/languag      | p_api.html#_CPPv4N5cudaq15scalar_ |
| es/cpp_api.html#_CPPv4N5cudaq16no | operatordvEdRK15scalar_operator), |
| ise_model_type15depolarization2E) |     [\[3\]](api/languages/cp      |
| -   [cudaq::noise_m               | p_api.html#_CPPv4N5cudaq15scalar_ |
| odel_type::depolarization_channel | operatordvEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|                                   | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|   enumerator)](api/languages/cpp_ | alar_operatordvENSt7complexIdEE), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[5\]](api/languages/cpp     |
| el_type22depolarization_channelE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -                                 | _operatordvERK15scalar_operator), |
|  [cudaq::noise_model_type::pauli1 |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     enumerator)](a                | 4NKR5cudaq15scalar_operatordvEd), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[7\]](api/language          |
| 5cudaq16noise_model_type6pauli1E) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -                                 | alar_operatordvENSt7complexIdEE), |
|  [cudaq::noise_model_type::pauli2 |     [\[8\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     enumerator)](a                | _operatordvERK15scalar_operator), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[9\                         |
| 5cudaq16noise_model_type6pauli2E) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq                        | Pv4NO5cudaq15scalar_operatordvEd) |
| ::noise_model_type::phase_damping | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator/= |
|     enumerator)](api/langu        |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     function)](api/languag        |
| noise_model_type13phase_dampingE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::noi                   | alar_operatordVENSt7complexIdEE), |
| se_model_type::phase_flip_channel |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](api/languages/   | _operatordVERK15scalar_operator), |
| cpp_api.html#_CPPv4N5cudaq16noise |     [\[2                          |
| _model_type18phase_flip_channelE) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq15scalar_operatordVEd) |
| [cudaq::noise_model_type::unknown | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator= |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languages/c    |
| cudaq16noise_model_type7unknownE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _operatoraSERK15scalar_operator), |
| [cudaq::noise_model_type::x_error |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|     enumerator)](ap               | r_operatoraSERR15scalar_operator) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [c                            |
| cudaq16noise_model_type7x_errorE) | udaq::scalar_operator::operator== |
| -                                 |     (C++                          |
| [cudaq::noise_model_type::y_error |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq15scala |
|     enumerator)](ap               | r_operatoreqERK15scalar_operator) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq:                       |
| cudaq16noise_model_type7y_errorE) | :scalar_operator::scalar_operator |
| -                                 |     (C++                          |
| [cudaq::noise_model_type::z_error |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     enumerator)](ap               | #_CPPv4N5cudaq15scalar_operator15 |
| i/languages/cpp_api.html#_CPPv4N5 | scalar_operatorENSt7complexIdEE), |
| cudaq16noise_model_type7z_errorE) |     [\[1\]](api/langu             |
| -   [cudaq::num_available_gpus    | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     (C++                          | scalar_operator15scalar_operatorE |
|     function                      | RK15scalar_callbackRRNSt13unorder |
| )](api/languages/cpp_api.html#_CP | ed_mapINSt6stringENSt6stringEEE), |
| Pv4N5cudaq18num_available_gpusEv) |     [\[2\                         |
| -   [cudaq::observe (C++          | ]](api/languages/cpp_api.html#_CP |
|     function)](                   | Pv4N5cudaq15scalar_operator15scal |
| api/languages/cpp_api.html#_CPPv4 | ar_operatorERK15scalar_operator), |
| I00Dp0EN5cudaq7observeENSt6vector |     [\[3\]](api/langu             |
| I14observe_resultEERR13QuantumKer | ages/cpp_api.html#_CPPv4N5cudaq15 |
| nelRK15SpinOpContainerDpRR4Args), | scalar_operator15scalar_operatorE |
|     [\[1\]](api/languages/cpp_api | RR15scalar_callbackRRNSt13unorder |
| .html#_CPPv4I0Dp0EN5cudaq7observe | ed_mapINSt6stringENSt6stringEEE), |
| E14observe_resultNSt6size_tERR13Q |     [\[4\                         |
| uantumKernelRK7spin_opDpRR4Args), | ]](api/languages/cpp_api.html#_CP |
|     [\[2                          | Pv4N5cudaq15scalar_operator15scal |
| \]](api/languages/cpp_api.html#_C | ar_operatorERR15scalar_operator), |
| PPv4I0Dp0EN5cudaq7observeE14obser |     [\[5\]](api/language          |
| ve_resultRK15observe_optionsRR13Q | s/cpp_api.html#_CPPv4N5cudaq15sca |
| uantumKernelRK7spin_opDpRR4Args), | lar_operator15scalar_operatorEd), |
|     [\[3\]](api/langu             |     [\[6\]](api/languag           |
| ages/cpp_api.html#_CPPv4I0Dp0EN5c | es/cpp_api.html#_CPPv4N5cudaq15sc |
| udaq7observeE14observe_resultRR13 | alar_operator15scalar_operatorEv) |
| QuantumKernelRK7spin_opDpRR4Args) | -   [                             |
| -   [cudaq::observe_options (C++  | cudaq::scalar_operator::to_matrix |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |                                   |
| #_CPPv4N5cudaq15observe_optionsE) |   function)](api/languages/cpp_ap |
| -   [cudaq::observe_result (C++   | i.html#_CPPv4NK5cudaq15scalar_ope |
|                                   | rator9to_matrixERKNSt13unordered_ |
| class)](api/languages/cpp_api.htm | mapINSt6stringENSt7complexIdEEEE) |
| l#_CPPv4N5cudaq14observe_resultE) | -   [                             |
| -                                 | cudaq::scalar_operator::to_string |
|    [cudaq::observe_result::counts |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)](api/languages/c    | anguages/cpp_api.html#_CPPv4NK5cu |
| pp_api.html#_CPPv4N5cudaq14observ | daq15scalar_operator9to_stringEv) |
| e_result6countsERK12spin_op_term) | -   [cudaq::s                     |
| -   [cudaq::observe_result::dump  | calar_operator::\~scalar_operator |
|     (C++                          |     (C++                          |
|     function)                     |     functio                       |
| ](api/languages/cpp_api.html#_CPP | n)](api/languages/cpp_api.html#_C |
| v4N5cudaq14observe_result4dumpEv) | PPv4N5cudaq15scalar_operatorD0Ev) |
| -   [c                            | -   [cuda                         |
| udaq::observe_result::expectation | q::SerializedCodeExecutionContext |
|     (C++                          |     (C++                          |
|                                   |     class)](api/lang              |
| function)](api/languages/cpp_api. | uages/cpp_api.html#_CPPv4N5cudaq3 |
| html#_CPPv4N5cudaq14observe_resul | 0SerializedCodeExecutionContextE) |
| t11expectationERK12spin_op_term), | -   [cudaq::set_noise (C++        |
|     [\[1\]](api/la                |     function)](api/langu          |
| nguages/cpp_api.html#_CPPv4N5cuda | ages/cpp_api.html#_CPPv4N5cudaq9s |
| q14observe_result11expectationEv) | et_noiseERKN5cudaq11noise_modelE) |
| -   [cuda                         | -   [cudaq::set_random_seed (C++  |
| q::observe_result::id_coefficient |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     function)](api/langu          | daq15set_random_seedENSt6size_tE) |
| ages/cpp_api.html#_CPPv4N5cudaq14 | -   [cudaq::simulation_precision  |
| observe_result14id_coefficientEv) |     (C++                          |
| -   [cuda                         |     enum)                         |
| q::observe_result::observe_result | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq20simulation_precisionE) |
|                                   | -   [                             |
|   function)](api/languages/cpp_ap | cudaq::simulation_precision::fp32 |
| i.html#_CPPv4N5cudaq14observe_res |     (C++                          |
| ult14observe_resultEdRK7spin_op), |     enumerator)](api              |
|     [\[1\]](a                     | /languages/cpp_api.html#_CPPv4N5c |
| pi/languages/cpp_api.html#_CPPv4N | udaq20simulation_precision4fp32E) |
| 5cudaq14observe_result14observe_r | -   [                             |
| esultEdRK7spin_op13sample_result) | cudaq::simulation_precision::fp64 |
| -                                 |     (C++                          |
|  [cudaq::observe_result::operator |     enumerator)](api              |
|     double (C++                   | /languages/cpp_api.html#_CPPv4N5c |
|     functio                       | udaq20simulation_precision4fp64E) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::SimulationState (C++  |
| PPv4N5cudaq14observe_resultcvdEv) |     c                             |
| -                                 | lass)](api/languages/cpp_api.html |
|  [cudaq::observe_result::raw_data | #_CPPv4N5cudaq15SimulationStateE) |
|     (C++                          | -   [                             |
|     function)](ap                 | cudaq::SimulationState::precision |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq14observe_result8raw_dataEv) |     enum)](api                    |
| -   [cudaq::operator_handler (C++ | /languages/cpp_api.html#_CPPv4N5c |
|     cl                            | udaq15SimulationState9precisionE) |
| ass)](api/languages/cpp_api.html# | -   [cudaq:                       |
| _CPPv4N5cudaq16operator_handlerE) | :SimulationState::precision::fp32 |
| -   [cudaq::optimizable_function  |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     class)                        | uages/cpp_api.html#_CPPv4N5cudaq1 |
| ](api/languages/cpp_api.html#_CPP | 5SimulationState9precision4fp32E) |
| v4N5cudaq20optimizable_functionE) | -   [cudaq:                       |
| -   [cudaq::optimization_result   | :SimulationState::precision::fp64 |
|     (C++                          |     (C++                          |
|     type                          |     enumerator)](api/lang         |
| )](api/languages/cpp_api.html#_CP | uages/cpp_api.html#_CPPv4N5cudaq1 |
| Pv4N5cudaq19optimization_resultE) | 5SimulationState9precision4fp64E) |
| -   [cudaq::optimizer (C++        | -                                 |
|     class)](api/languages/cpp_a   |   [cudaq::SimulationState::Tensor |
| pi.html#_CPPv4N5cudaq9optimizerE) |     (C++                          |
| -   [cudaq::optimizer::optimize   |     struct)](                     |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq15SimulationState6TensorE) |
|  function)](api/languages/cpp_api | -   [cudaq::spin_handler (C++     |
| .html#_CPPv4N5cudaq9optimizer8opt |                                   |
| imizeEKiRR20optimizable_function) |   class)](api/languages/cpp_api.h |
| -   [cu                           | tml#_CPPv4N5cudaq12spin_handlerE) |
| daq::optimizer::requiresGradients | -   [cudaq:                       |
|     (C++                          | :spin_handler::to_diagonal_matrix |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/la             |
| q9optimizer17requiresGradientsEv) | nguages/cpp_api.html#_CPPv4NK5cud |
| -   [cudaq::orca (C++             | aq12spin_handler18to_diagonal_mat |
|     type)](api/languages/         | rixERNSt13unordered_mapINSt6size_ |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | tENSt7int64_tEEERKNSt13unordered_ |
| -   [cudaq::orca::sample (C++     | mapINSt6stringENSt7complexIdEEEE) |
|     function)](api/languages/c    | -                                 |
| pp_api.html#_CPPv4N5cudaq4orca6sa |   [cudaq::spin_handler::to_matrix |
| mpleERNSt6vectorINSt6size_tEEERNS |     (C++                          |
| t6vectorINSt6size_tEEERNSt6vector |     function                      |
| IdEERNSt6vectorIdEEiNSt6size_tE), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]]                       | Pv4N5cudaq12spin_handler9to_matri |
| (api/languages/cpp_api.html#_CPPv | xERKNSt6stringENSt7complexIdEEb), |
| 4N5cudaq4orca6sampleERNSt6vectorI |     [\[1                          |
| NSt6size_tEEERNSt6vectorINSt6size | \]](api/languages/cpp_api.html#_C |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | PPv4NK5cudaq12spin_handler9to_mat |
| -   [cudaq::orca::sample_async    | rixERNSt13unordered_mapINSt6size_ |
|     (C++                          | tENSt7int64_tEEERKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
| function)](api/languages/cpp_api. | -   [cuda                         |
| html#_CPPv4N5cudaq4orca12sample_a | q::spin_handler::to_sparse_matrix |
| syncERNSt6vectorINSt6size_tEEERNS |     (C++                          |
| t6vectorINSt6size_tEEERNSt6vector |     function)](api/               |
| IdEERNSt6vectorIdEEiNSt6size_tE), | languages/cpp_api.html#_CPPv4N5cu |
|     [\[1\]](api/la                | daq12spin_handler16to_sparse_matr |
| nguages/cpp_api.html#_CPPv4N5cuda | ixERKNSt6stringENSt7complexIdEEb) |
| q4orca12sample_asyncERNSt6vectorI | -                                 |
| NSt6size_tEEERNSt6vectorINSt6size |   [cudaq::spin_handler::to_string |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     (C++                          |
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
