::: {.wy-grid-for-nav}
::: {.wy-side-scroll}
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: {.version}
pr-3467
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
        -   [Pooling the memory of multiple GPUs (`mgpu`{.code .docutils
            .literal
            .notranslate})](using/examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs (`mqpu`{.code
            .docutils .literal
            .notranslate})](using/examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](using/examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](using/examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends (`remote-mqpu`{.code .docutils
            .literal
            .notranslate})](using/examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
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
                `collapse_operators`{.docutils .literal
                .notranslate}](examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L\_i\\)]{.math .notranslate
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
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H\_2\\)]{.math
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
        -   [Using `Sample`{.docutils .literal .notranslate} to perform
            the Hadamard
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
        -   [Constructing circuits in the `[[4,2,2]]`{.docutils .literal
            .notranslate}
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
            -   [3. `Compute overlap`{.docutils .literal
                .notranslate}](applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
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
        -   [The problem Hamiltonian [\\(H\_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A\_j\\)]{.math .notranslate
            .nohighlight}:](applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H\_C,A\_j\]\\)]{.math .notranslate
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
            [\\(A\_i\\)]{.math .notranslate
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
            -   [`CMakeLists.txt`{.docutils .literal
                .notranslate}](using/extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](using/extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent `CMakeLists.txt`{.docutils .literal
                .notranslate}](using/extending/backend.html#update-parent-cmakelists-txt){.reference
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
        -   [`CircuitSimulator`{.code .docutils .literal
            .notranslate}](using/extending/nvqir_simulator.html#circuitsimulator){.reference
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
            -   [3.1. `cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}](specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. `cudaq::qubit`{.code .docutils .literal
                .notranslate}](specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. `cudaq::spin_op`{.code .docutils .literal
                .notranslate}](specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on `cudaq::qubit`{.code .docutils
                .literal
                .notranslate}](specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
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
            -   [12.1. `cudaq::sample`{.code .docutils .literal
                .notranslate}](specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. `cudaq::run`{.code .docutils .literal
                .notranslate}](specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. `cudaq::observe`{.code .docutils .literal
                .notranslate}](specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. `cudaq::optimizer`{.code .docutils .literal
                .notranslate} (deprecated, functionality moved to CUDA-Q
                libraries)](specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. `cudaq::gradient`{.code .docutils .literal
                .notranslate} (deprecated, functionality moved to CUDA-Q
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
            -   [`make_kernel()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [`PyKernel`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [`Kernel`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [`PyKernelDecorator`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [`kernel()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [`sample()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [`sample_async()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [`run()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [`run_async()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [`observe()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [`observe_async()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [`get_state()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [`get_state_async()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [`vqe()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [`draw()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [`translate()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [`estimate_resources()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
        -   [Backend
            Configuration](api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [`has_target()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [`get_target()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [`get_targets()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [`set_target()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [`reset_target()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [`set_noise()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [`unset_noise()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [`register_set_target_callback()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [`unregister_set_target_callback()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [`cudaq.apply_noise()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [`initialize_cudaq()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [`num_available_gpus()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [`set_random_seed()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [`evolve()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [`evolve_async()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [`Schedule`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [`BaseIntegrator`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [`InitialState`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [`InitialStateType`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [`IntermediateResultSave`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](api/languages/python_api.html#operators){.reference
            .internal}
            -   [`OperatorSum`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [`ProductOperator`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [`ElementaryOperator`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [`ScalarOperator`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [`RydbergHamiltonian`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [`SuperOperator`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [`operators.define()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [`operators.instantiate()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.operators.instantiate){.reference
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
            -   [`SimulationPrecision`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [`Target`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [`State`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [`Tensor`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [`QuakeValue`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [`qubit`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [`qreg`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [`qvector`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [`ComplexMatrix`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [`SampleResult`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [`AsyncSampleResult`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [`ObserveResult`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [`AsyncObserveResult`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [`AsyncStateResult`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [`OptimizationResult`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [`EvolveResult`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [`AsyncEvolveResult`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [`Resources`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.Resources){.reference
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
            -   [`initialize()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [`rank()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [`num_ranks()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [`all_gather()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [`broadcast()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [`is_initialized()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [`finalize()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [`sample()`{.docutils .literal
                .notranslate}](api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
    -   [Quantum Operations](api/default_ops.html){.reference .internal}
        -   [Unitary Operations on
            Qubits](api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [`x`{.code .docutils .literal
                .notranslate}](api/default_ops.html#x){.reference
                .internal}
            -   [`y`{.code .docutils .literal
                .notranslate}](api/default_ops.html#y){.reference
                .internal}
            -   [`z`{.code .docutils .literal
                .notranslate}](api/default_ops.html#z){.reference
                .internal}
            -   [`h`{.code .docutils .literal
                .notranslate}](api/default_ops.html#h){.reference
                .internal}
            -   [`r1`{.code .docutils .literal
                .notranslate}](api/default_ops.html#r1){.reference
                .internal}
            -   [`rx`{.code .docutils .literal
                .notranslate}](api/default_ops.html#rx){.reference
                .internal}
            -   [`ry`{.code .docutils .literal
                .notranslate}](api/default_ops.html#ry){.reference
                .internal}
            -   [`rz`{.code .docutils .literal
                .notranslate}](api/default_ops.html#rz){.reference
                .internal}
            -   [`s`{.code .docutils .literal
                .notranslate}](api/default_ops.html#s){.reference
                .internal}
            -   [`t`{.code .docutils .literal
                .notranslate}](api/default_ops.html#t){.reference
                .internal}
            -   [`swap`{.code .docutils .literal
                .notranslate}](api/default_ops.html#swap){.reference
                .internal}
            -   [`u3`{.code .docutils .literal
                .notranslate}](api/default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [`mz`{.code .docutils .literal
                .notranslate}](api/default_ops.html#mz){.reference
                .internal}
            -   [`mx`{.code .docutils .literal
                .notranslate}](api/default_ops.html#mx){.reference
                .internal}
            -   [`my`{.code .docutils .literal
                .notranslate}](api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [`create`{.code .docutils .literal
                .notranslate}](api/default_ops.html#create){.reference
                .internal}
            -   [`annihilate`{.code .docutils .literal
                .notranslate}](api/default_ops.html#annihilate){.reference
                .internal}
            -   [`phase_shift`{.code .docutils .literal
                .notranslate}](api/default_ops.html#phase-shift){.reference
                .internal}
            -   [`beam_splitter`{.code .docutils .literal
                .notranslate}](api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [`mz`{.code .docutils .literal
                .notranslate}](api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](index.html)

::: {.wy-nav-content}
::: {.rst-content}
::: {role="navigation" aria-label="Page navigation"}
-   [](index.html){.icon .icon-home}
-   Index
-   

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
Index
=====

::: {.genindex-jumpbox}
[**\_**](#_) \| [**A**](#A) \| [**B**](#B) \| [**C**](#C) \| [**D**](#D)
\| [**E**](#E) \| [**F**](#F) \| [**G**](#G) \| [**H**](#H) \|
[**I**](#I) \| [**K**](#K) \| [**L**](#L) \| [**M**](#M) \| [**N**](#N)
\| [**O**](#O) \| [**P**](#P) \| [**Q**](#Q) \| [**R**](#R) \|
[**S**](#S) \| [**T**](#T) \| [**U**](#U) \| [**V**](#V) \| [**X**](#X)
\| [**Y**](#Y) \| [**Z**](#Z)
:::

\_ {#_}
--

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

A {#A}
-

+-----------------------------------+-----------------------------------+
| -   [add\_all\_qubit\_channel()   | -   [append() (cudaq.KrausChannel |
|     (cudaq.NoiseModel             |                                   |
|     method)](api                  |  method)](api/languages/python_ap |
| /languages/python_api.html#cudaq. | i.html#cudaq.KrausChannel.append) |
| NoiseModel.add_all_qubit_channel) | -   [argument\_count              |
| -   [add\_channel()               |     (cudaq.PyKernel               |
|     (cudaq.NoiseModel             |     attrib                        |
|     me                            | ute)](api/languages/python_api.ht |
| thod)](api/languages/python_api.h | ml#cudaq.PyKernel.argument_count) |
| tml#cudaq.NoiseModel.add_channel) | -   [arguments (cudaq.PyKernel    |
| -   [all\_gather() (in module     |     a                             |
|                                   | ttribute)](api/languages/python_a |
|    cudaq.mpi)](api/languages/pyth | pi.html#cudaq.PyKernel.arguments) |
| on_api.html#cudaq.mpi.all_gather) | -   [as\_pauli()                  |
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

B {#B}
-

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

C {#C}
-

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
|        method)](api/languages/pyt |   [cudaq::pauli1::num\_parameters |
| hon_api.html#cudaq.operators.boso |     (C++                          |
| n.BosonOperatorTerm.canonicalize) |     member)]                      |
|     -   [(cudaq.                  | (api/languages/cpp_api.html#_CPPv |
| operators.fermion.FermionOperator | 4N5cudaq6pauli114num_parametersE) |
|                                   | -   [cudaq::pauli1::num\_targets  |
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
| uages/python_api.html#cudaq.opera |   [cudaq::pauli2::num\_parameters |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     member)]                      |
| udaq.operators.MatrixOperatorTerm | (api/languages/cpp_api.html#_CPPv |
|         method)](api/language     | 4N5cudaq6pauli214num_parametersE) |
| s/python_api.html#cudaq.operators | -   [cudaq::pauli2::num\_targets  |
| .MatrixOperatorTerm.canonicalize) |     (C++                          |
|     -   [(                        |     membe                         |
| cudaq.operators.spin.SpinOperator | r)](api/languages/cpp_api.html#_C |
|         method)](api/languag      | PPv4N5cudaq6pauli211num_targetsE) |
| es/python_api.html#cudaq.operator | -   [cudaq::pauli2::pauli2 (C++   |
| s.spin.SpinOperator.canonicalize) |     function)](api/languages/cpp_ |
|     -   [(cuda                    | api.html#_CPPv4N5cudaq6pauli26pau |
| q.operators.spin.SpinOperatorTerm | li2ERKNSt6vectorIN5cudaq4realEEE) |
|         method)](api/languages/p  | -   [cudaq::phase\_damping (C++   |
| ython_api.html#cudaq.operators.sp |                                   |
| in.SpinOperatorTerm.canonicalize) |  class)](api/languages/cpp_api.ht |
| -   [canonicalized() (in module   | ml#_CPPv4N5cudaq13phase_dampingE) |
|     cuda                          | -   [cudaq                        |
| q.boson)](api/languages/python_ap | ::phase\_damping::num\_parameters |
| i.html#cudaq.boson.canonicalized) |     (C++                          |
|     -   [(in module               |     member)](api/lan              |
|         cudaq.fe                  | guages/cpp_api.html#_CPPv4N5cudaq |
| rmion)](api/languages/python_api. | 13phase_damping14num_parametersE) |
| html#cudaq.fermion.canonicalized) | -   [cu                           |
|     -   [(in module               | daq::phase\_damping::num\_targets |
|                                   |     (C++                          |
|        cudaq.operators.custom)](a |     member)](api/                 |
| pi/languages/python_api.html#cuda | languages/cpp_api.html#_CPPv4N5cu |
| q.operators.custom.canonicalized) | daq13phase_damping11num_targetsE) |
|     -   [(in module               | -   [cudaq::phase\_flip\_channel  |
|         cu                        |     (C++                          |
| daq.spin)](api/languages/python_a |     clas                          |
| pi.html#cudaq.spin.canonicalized) | s)](api/languages/cpp_api.html#_C |
| -   [CentralDifference (class in  | PPv4N5cudaq18phase_flip_channelE) |
|     cudaq.gradients)              | -   [cudaq::phas                  |
| ](api/languages/python_api.html#c | e\_flip\_channel::num\_parameters |
| udaq.gradients.CentralDifference) |     (C++                          |
| -   [clear() (cudaq.Resources     |     member)](api/language         |
|     method)](api/languages/pytho  | s/cpp_api.html#_CPPv4N5cudaq18pha |
| n_api.html#cudaq.Resources.clear) | se_flip_channel14num_parametersE) |
|     -   [(cudaq.SampleResult      | -   [cudaq::p                     |
|                                   | hase\_flip\_channel::num\_targets |
|   method)](api/languages/python_a |     (C++                          |
| pi.html#cudaq.SampleResult.clear) |     member)](api/langu            |
| -   [COBYLA (class in             | ages/cpp_api.html#_CPPv4N5cudaq18 |
|     cudaq.o                       | phase_flip_channel11num_targetsE) |
| ptimizers)](api/languages/python_ | -   [cudaq::product\_op (C++      |
| api.html#cudaq.optimizers.COBYLA) |                                   |
| -   [coefficient                  |  class)](api/languages/cpp_api.ht |
|     (cudaq.                       | ml#_CPPv4I0EN5cudaq10product_opE) |
| operators.boson.BosonOperatorTerm | -   [cudaq::product\_op::begin    |
|     property)](api/languages/py   |     (C++                          |
| thon_api.html#cudaq.operators.bos |     functio                       |
| on.BosonOperatorTerm.coefficient) | n)](api/languages/cpp_api.html#_C |
|     -   [(cudaq.oper              | PPv4NK5cudaq10product_op5beginEv) |
| ators.fermion.FermionOperatorTerm | -                                 |
|                                   | [cudaq::product\_op::canonicalize |
|   property)](api/languages/python |     (C++                          |
| _api.html#cudaq.operators.fermion |     func                          |
| .FermionOperatorTerm.coefficient) | tion)](api/languages/cpp_api.html |
|     -   [(c                       | #_CPPv4N5cudaq10product_op12canon |
| udaq.operators.MatrixOperatorTerm | icalizeERKNSt3setINSt6size_tEEE), |
|         property)](api/languag    |     [\[1\]](api                   |
| es/python_api.html#cudaq.operator | /languages/cpp_api.html#_CPPv4N5c |
| s.MatrixOperatorTerm.coefficient) | udaq10product_op12canonicalizeEv) |
|     -   [(cuda                    | -   [cu                           |
| q.operators.spin.SpinOperatorTerm | daq::product\_op::const\_iterator |
|         property)](api/languages/ |     (C++                          |
| python_api.html#cudaq.operators.s |     struct)](api/                 |
| pin.SpinOperatorTerm.coefficient) | languages/cpp_api.html#_CPPv4N5cu |
| -   [col\_count                   | daq10product_op14const_iteratorE) |
|     (cudaq.KrausOperator          | -   [cudaq::product\_op:          |
|     prope                         | :const\_iterator::const\_iterator |
| rty)](api/languages/python_api.ht |     (C++                          |
| ml#cudaq.KrausOperator.col_count) |     fu                            |
| -   [compile()                    | nction)](api/languages/cpp_api.ht |
|     (cudaq.PyKernelDecorator      | ml#_CPPv4N5cudaq10product_op14con |
|     metho                         | st_iterator14const_iteratorEPK10p |
| d)](api/languages/python_api.html | roduct_opI9HandlerTyENSt6size_tE) |
| #cudaq.PyKernelDecorator.compile) | -   [cudaq::product               |
| -   [ComplexMatrix (class in      | \_op::const\_iterator::operator!= |
|     cudaq)](api/languages/pyt     |     (C++                          |
| hon_api.html#cudaq.ComplexMatrix) |     fun                           |
| -   [compute()                    | ction)](api/languages/cpp_api.htm |
|     (                             | l#_CPPv4NK5cudaq10product_op14con |
| cudaq.gradients.CentralDifference | st_iteratorneERK14const_iterator) |
|     method)](api/la               | -   [cudaq::product               |
| nguages/python_api.html#cudaq.gra | \_op::const\_iterator::operator\* |
| dients.CentralDifference.compute) |     (C++                          |
|     -   [(                        |     function)](api/lang           |
| cudaq.gradients.ForwardDifference | uages/cpp_api.html#_CPPv4NK5cudaq |
|         method)](api/la           | 10product_op14const_iteratormlEv) |
| nguages/python_api.html#cudaq.gra | -   [cudaq::product               |
| dients.ForwardDifference.compute) | \_op::const\_iterator::operator++ |
|     -                             |     (C++                          |
|  [(cudaq.gradients.ParameterShift |     function)](api/lang           |
|         method)](api              | uages/cpp_api.html#_CPPv4N5cudaq1 |
| /languages/python_api.html#cudaq. | 0product_op14const_iteratorppEi), |
| gradients.ParameterShift.compute) |     [\[1\]](api/lan               |
| -   [const()                      | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 10product_op14const_iteratorppEv) |
|   (cudaq.operators.ScalarOperator | -   [cudaq::product\              |
|     class                         | _op::const\_iterator::operator\-- |
|     method)](a                    |     (C++                          |
| pi/languages/python_api.html#cuda |     function)](api/lang           |
| q.operators.ScalarOperator.const) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [copy()                       | 0product_op14const_iteratormmEi), |
|     (cu                           |     [\[1\]](api/lan               |
| daq.operators.boson.BosonOperator | guages/cpp_api.html#_CPPv4N5cudaq |
|     method)](api/l                | 10product_op14const_iteratormmEv) |
| anguages/python_api.html#cudaq.op | -   [cudaq::product\              |
| erators.boson.BosonOperator.copy) | _op::const\_iterator::operator-\> |
|     -   [(cudaq.                  |     (C++                          |
| operators.boson.BosonOperatorTerm |     function)](api/lan            |
|         method)](api/langu        | guages/cpp_api.html#_CPPv4N5cudaq |
| ages/python_api.html#cudaq.operat | 10product_op14const_iteratorptEv) |
| ors.boson.BosonOperatorTerm.copy) | -   [cudaq::product               |
|     -   [(cudaq.                  | \_op::const\_iterator::operator== |
| operators.fermion.FermionOperator |     (C++                          |
|         method)](api/langu        |     fun                           |
| ages/python_api.html#cudaq.operat | ction)](api/languages/cpp_api.htm |
| ors.fermion.FermionOperator.copy) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(cudaq.oper              | st_iteratoreqERK14const_iterator) |
| ators.fermion.FermionOperatorTerm | -   [cudaq::product\_op::degrees  |
|         method)](api/languages    |     (C++                          |
| /python_api.html#cudaq.operators. |     function)                     |
| fermion.FermionOperatorTerm.copy) | ](api/languages/cpp_api.html#_CPP |
|     -                             | v4NK5cudaq10product_op7degreesEv) |
|  [(cudaq.operators.MatrixOperator | -   [cudaq::product\_op::dump     |
|         method)](                 |     (C++                          |
| api/languages/python_api.html#cud |     functi                        |
| aq.operators.MatrixOperator.copy) | on)](api/languages/cpp_api.html#_ |
|     -   [(c                       | CPPv4NK5cudaq10product_op4dumpEv) |
| udaq.operators.MatrixOperatorTerm | -   [cudaq::product\_op::end (C++ |
|         method)](api/             |     funct                         |
| languages/python_api.html#cudaq.o | ion)](api/languages/cpp_api.html# |
| perators.MatrixOperatorTerm.copy) | _CPPv4NK5cudaq10product_op3endEv) |
|     -   [(                        | -   [cud                          |
| cudaq.operators.spin.SpinOperator | aq::product\_op::get\_coefficient |
|         method)](api              |     (C++                          |
| /languages/python_api.html#cudaq. |     function)](api/lan            |
| operators.spin.SpinOperator.copy) | guages/cpp_api.html#_CPPv4NK5cuda |
|     -   [(cuda                    | q10product_op15get_coefficientEv) |
| q.operators.spin.SpinOperatorTerm | -   [                             |
|         method)](api/lan          | cudaq::product\_op::get\_term\_id |
| guages/python_api.html#cudaq.oper |     (C++                          |
| ators.spin.SpinOperatorTerm.copy) |     function)](api                |
| -   [count() (cudaq.Resources     | /languages/cpp_api.html#_CPPv4NK5 |
|     method)](api/languages/pytho  | cudaq10product_op11get_term_idEv) |
| n_api.html#cudaq.Resources.count) | -                                 |
|     -   [(cudaq.SampleResult      | [cudaq::product\_op::is\_identity |
|                                   |     (C++                          |
|   method)](api/languages/python_a |     function)](api                |
| pi.html#cudaq.SampleResult.count) | /languages/cpp_api.html#_CPPv4NK5 |
| -   [count\_controls()            | cudaq10product_op11is_identityEv) |
|     (cudaq.Resources              | -   [cudaq::product\_op::num\_ops |
|     meth                          |     (C++                          |
| od)](api/languages/python_api.htm |     function)                     |
| l#cudaq.Resources.count_controls) | ](api/languages/cpp_api.html#_CPP |
| -   [counts()                     | v4NK5cudaq10product_op7num_opsEv) |
|     (cudaq.ObserveResult          | -                                 |
|                                   |   [cudaq::product\_op::operator\* |
| method)](api/languages/python_api |     (C++                          |
| .html#cudaq.ObserveResult.counts) |     function)](api/languages/     |
| -   [create() (in module          | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|                                   | oduct_opmlE10product_opI1TERK15sc |
|    cudaq.boson)](api/languages/py | alar_operatorRK10product_opI1TE), |
| thon_api.html#cudaq.boson.create) |     [\[1\]](api/languages/        |
|     -   [(in module               | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|         c                         | oduct_opmlE10product_opI1TERK15sc |
| udaq.fermion)](api/languages/pyth | alar_operatorRR10product_opI1TE), |
| on_api.html#cudaq.fermion.create) |     [\[2\]](api/languages/        |
| -   [csr\_spmatrix (C++           | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     type)](api/languages/c        | oduct_opmlE10product_opI1TERR15sc |
| pp_api.html#_CPPv412csr_spmatrix) | alar_operatorRK10product_opI1TE), |
| -   cudaq                         |     [\[3\]](api/languages/        |
|     -   [module](api/langua       | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| ges/python_api.html#module-cudaq) | oduct_opmlE10product_opI1TERR15sc |
| -   [cudaq (C++                   | alar_operatorRR10product_opI1TE), |
|     type)](api/lan                |     [\[4\]](api/                  |
| guages/cpp_api.html#_CPPv45cudaq) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq.apply\_noise() (in     | 5cudaq10product_opmlE6sum_opI1TER |
|     module                        | K15scalar_operatorRK6sum_opI1TE), |
|     cudaq)](api/languages/python_ |     [\[5\]](api/                  |
| api.html#cudaq.cudaq.apply_noise) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.boson                   | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [module](api/languages/py | K15scalar_operatorRR6sum_opI1TE), |
| thon_api.html#module-cudaq.boson) |     [\[6\]](api/                  |
| -   cudaq.fermion                 | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opmlE6sum_opI1TER |
|   -   [module](api/languages/pyth | R15scalar_operatorRK6sum_opI1TE), |
| on_api.html#module-cudaq.fermion) |     [\[7\]](api/                  |
| -   cudaq.operators.custom        | languages/cpp_api.html#_CPPv4I0EN |
|     -   [mo                       | 5cudaq10product_opmlE6sum_opI1TER |
| dule](api/languages/python_api.ht | R15scalar_operatorRR6sum_opI1TE), |
| ml#module-cudaq.operators.custom) |     [\[8\]](api/languages         |
| -   cudaq.spin                    | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     -   [module](api/languages/p  | duct_opmlERK6sum_opI9HandlerTyE), |
| ython_api.html#module-cudaq.spin) |     [\[9\]](api/languages/cpp_a   |
| -   [cudaq::amplitude\_damping    | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmlERK10product_opI9HandlerTyE), |
|     cla                           |     [\[10\]](api/language         |
| ss)](api/languages/cpp_api.html#_ | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| CPPv4N5cudaq17amplitude_dampingE) | roduct_opmlERK15scalar_operator), |
| -   [c                            |     [\[11\]](api/languages/cpp_a  |
| udaq::amplitude\_damping\_channel | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmlERR10product_opI9HandlerTyE), |
|     class)](api                   |     [\[12\]](api/language         |
| /languages/cpp_api.html#_CPPv4N5c | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| udaq25amplitude_damping_channelE) | roduct_opmlERR15scalar_operator), |
| -   [cudaq::amplitude\_           |     [\[13\]](api/languages/cpp_   |
| damping\_channel::num\_parameters | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmlERK10product_opI9HandlerTyE), |
|     member)](api/languages/cpp_a  |     [\[14\]](api/languag          |
| pi.html#_CPPv4N5cudaq25amplitude_ | es/cpp_api.html#_CPPv4NO5cudaq10p |
| damping_channel14num_parametersE) | roduct_opmlERK15scalar_operator), |
| -   [cudaq::amplitud              |     [\[15\]](api/languages/cpp_   |
| e\_damping\_channel::num\_targets | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmlERR10product_opI9HandlerTyE), |
|     member)](api/languages/cp     |     [\[16\]](api/langua           |
| p_api.html#_CPPv4N5cudaq25amplitu | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| de_damping_channel11num_targetsE) | product_opmlERR15scalar_operator) |
| -   [cudaq::AnalogRemoteRESTQPU   | -                                 |
|     (C++                          |  [cudaq::product\_op::operator\*= |
|     class                         |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/languages/cpp  |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | _api.html#_CPPv4N5cudaq10product_ |
| -   [cudaq::apply\_noise (C++     | opmLERK10product_opI9HandlerTyE), |
|     function)](api/               |     [\[1\]](api/langua            |
| languages/cpp_api.html#_CPPv4I0Dp | ges/cpp_api.html#_CPPv4N5cudaq10p |
| EN5cudaq11apply_noiseEvDpRR4Args) | roduct_opmLERK15scalar_operator), |
| -   [cudaq::async\_result (C++    |     [\[2\]](api/languages/cp      |
|     c                             | p_api.html#_CPPv4N5cudaq10product |
| lass)](api/languages/cpp_api.html | _opmLERR10product_opI9HandlerTyE) |
| #_CPPv4I0EN5cudaq12async_resultE) | -                                 |
| -   [cudaq::async\_result::get    |    [cudaq::product\_op::operator+ |
|     (C++                          |     (C++                          |
|     functi                        |     function)](api/langu          |
| on)](api/languages/cpp_api.html#_ | ages/cpp_api.html#_CPPv4I0EN5cuda |
| CPPv4N5cudaq12async_result3getEv) | q10product_opplE6sum_opI1TERK15sc |
| -   [cudaq::async\_sample\_result | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     type                          | languages/cpp_api.html#_CPPv4I0EN |
| )](api/languages/cpp_api.html#_CP | 5cudaq10product_opplE6sum_opI1TER |
| Pv4N5cudaq19async_sample_resultE) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::BaseNvcfSimulatorQPU  |     [\[2\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     class)                        | q10product_opplE6sum_opI1TERK15sc |
| ](api/languages/cpp_api.html#_CPP | alar_operatorRR10product_opI1TE), |
| v4N5cudaq20BaseNvcfSimulatorQPUE) |     [\[3\]](api/                  |
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
| -   [cudaq::bit\_flip\_channel    |     [\[6\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cl                            | q10product_opplE6sum_opI1TERR15sc |
| ass)](api/languages/cpp_api.html# | alar_operatorRR10product_opI1TE), |
| _CPPv4N5cudaq16bit_flip_channelE) |     [\[7\]](api/                  |
| -   [cudaq::bi                    | languages/cpp_api.html#_CPPv4I0EN |
| t\_flip\_channel::num\_parameters | 5cudaq10product_opplE6sum_opI1TER |
|     (C++                          | R15scalar_operatorRR6sum_opI1TE), |
|     member)](api/langua           |     [\[8\]](api/languages/cpp_a   |
| ges/cpp_api.html#_CPPv4N5cudaq16b | pi.html#_CPPv4NKR5cudaq10product_ |
| it_flip_channel14num_parametersE) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq:                       |     [\[9\]](api/language          |
| :bit\_flip\_channel::num\_targets | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opplERK15scalar_operator), |
|     member)](api/lan              |     [\[10\]](api/languages/       |
| guages/cpp_api.html#_CPPv4N5cudaq | cpp_api.html#_CPPv4NKR5cudaq10pro |
| 16bit_flip_channel11num_targetsE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq::boson\_handler (C++   |     [\[11\]](api/languages/cpp_a  |
|                                   | pi.html#_CPPv4NKR5cudaq10product_ |
|  class)](api/languages/cpp_api.ht | opplERR10product_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13boson_handlerE) |     [\[12\]](api/language         |
| -   [cudaq::boson\_op (C++        | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     type)](api/languages/cpp_     | roduct_opplERR15scalar_operator), |
| api.html#_CPPv4N5cudaq8boson_opE) |     [\[13\]](api/languages/       |
| -   [cudaq::boson\_op\_term (C++  | cpp_api.html#_CPPv4NKR5cudaq10pro |
|                                   | duct_opplERR6sum_opI9HandlerTyE), |
|   type)](api/languages/cpp_api.ht |     [\[                           |
| ml#_CPPv4N5cudaq13boson_op_termE) | 14\]](api/languages/cpp_api.html# |
| -   [cudaq::CodeGenConfig (C++    | _CPPv4NKR5cudaq10product_opplEv), |
|                                   |     [\[15\]](api/languages/cpp_   |
| struct)](api/languages/cpp_api.ht | api.html#_CPPv4NO5cudaq10product_ |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | opplERK10product_opI9HandlerTyE), |
| -                                 |     [\[16\]](api/languag          |
|    [cudaq::commutation\_relations | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opplERK15scalar_operator), |
|     struct)]                      |     [\[17\]](api/languages        |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4NO5cudaq10pro |
| 4N5cudaq21commutation_relationsE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq::complex (C++          |     [\[18\]](api/languages/cpp_   |
|     type)](api/languages/cpp      | api.html#_CPPv4NO5cudaq10product_ |
| _api.html#_CPPv4N5cudaq7complexE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::complex\_matrix (C++  |     [\[19\]](api/languag          |
|                                   | es/cpp_api.html#_CPPv4NO5cudaq10p |
| class)](api/languages/cpp_api.htm | roduct_opplERR15scalar_operator), |
| l#_CPPv4N5cudaq14complex_matrixE) |     [\[20\]](api/languages        |
| -                                 | /cpp_api.html#_CPPv4NO5cudaq10pro |
|  [cudaq::complex\_matrix::adjoint | duct_opplERR6sum_opI9HandlerTyE), |
|     (C++                          |     [                             |
|     function)](a                  | \[21\]](api/languages/cpp_api.htm |
| pi/languages/cpp_api.html#_CPPv4N | l#_CPPv4NO5cudaq10product_opplEv) |
| 5cudaq14complex_matrix7adjointEv) | -                                 |
| -   [cudaq::co                    |    [cudaq::product\_op::operator- |
| mplex\_matrix::diagonal\_elements |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](api/languages      | ages/cpp_api.html#_CPPv4I0EN5cuda |
| /cpp_api.html#_CPPv4NK5cudaq14com | q10product_opmiE6sum_opI1TERK15sc |
| plex_matrix17diagonal_elementsEi) | alar_operatorRK10product_opI1TE), |
| -   [cudaq::complex\_matrix::dump |     [\[1\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/language       | 5cudaq10product_opmiE6sum_opI1TER |
| s/cpp_api.html#_CPPv4NK5cudaq14co | K15scalar_operatorRK6sum_opI1TE), |
| mplex_matrix4dumpERNSt7ostreamE), |     [\[2\]](api/langu             |
|     [\[1\]]                       | ages/cpp_api.html#_CPPv4I0EN5cuda |
| (api/languages/cpp_api.html#_CPPv | q10product_opmiE6sum_opI1TERK15sc |
| 4NK5cudaq14complex_matrix4dumpEv) | alar_operatorRR10product_opI1TE), |
| -   [cu                           |     [\[3\]](api/                  |
| daq::complex\_matrix::eigenvalues | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/lan            | K15scalar_operatorRR6sum_opI1TE), |
| guages/cpp_api.html#_CPPv4NK5cuda |     [\[4\]](api/langu             |
| q14complex_matrix11eigenvaluesEv) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cud                          | q10product_opmiE6sum_opI1TERR15sc |
| aq::complex\_matrix::eigenvectors | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[5\]](api/                  |
|     function)](api/lang           | languages/cpp_api.html#_CPPv4I0EN |
| uages/cpp_api.html#_CPPv4NK5cudaq | 5cudaq10product_opmiE6sum_opI1TER |
| 14complex_matrix12eigenvectorsEv) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cu                           |     [\[6\]](api/langu             |
| daq::complex\_matrix::exponential | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERR15sc |
|     function)](api/la             | alar_operatorRR10product_opI1TE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[7\]](api/                  |
| q14complex_matrix11exponentialEv) | languages/cpp_api.html#_CPPv4I0EN |
| -                                 | 5cudaq10product_opmiE6sum_opI1TER |
| [cudaq::complex\_matrix::identity | R15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[8\]](api/languages/cpp_a   |
|     function)](api/languages      | pi.html#_CPPv4NKR5cudaq10product_ |
| /cpp_api.html#_CPPv4N5cudaq14comp | opmiERK10product_opI9HandlerTyE), |
| lex_matrix8identityEKNSt6size_tE) |     [\[9\]](api/language          |
| -   [                             | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| cudaq::complex\_matrix::kronecker | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[10\]](api/languages/       |
|     function)](api/lang           | cpp_api.html#_CPPv4NKR5cudaq10pro |
| uages/cpp_api.html#_CPPv4I00EN5cu | duct_opmiERK6sum_opI9HandlerTyE), |
| daq14complex_matrix9kroneckerE14c |     [\[11\]](api/languages/cpp_a  |
| omplex_matrix8Iterable8Iterable), | pi.html#_CPPv4NKR5cudaq10product_ |
|     [\[1\]](api/l                 | opmiERR10product_opI9HandlerTyE), |
| anguages/cpp_api.html#_CPPv4N5cud |     [\[12\]](api/language         |
| aq14complex_matrix9kroneckerERK14 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| complex_matrixRK14complex_matrix) | roduct_opmiERR15scalar_operator), |
| -   [cudaq::com                   |     [\[13\]](api/languages/       |
| plex\_matrix::minimal\_eigenvalue | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     (C++                          | duct_opmiERR6sum_opI9HandlerTyE), |
|     function)](api/languages/     |     [\[                           |
| cpp_api.html#_CPPv4NK5cudaq14comp | 14\]](api/languages/cpp_api.html# |
| lex_matrix18minimal_eigenvalueEv) | _CPPv4NKR5cudaq10product_opmiEv), |
| -   [c                            |     [\[15\]](api/languages/cpp_   |
| udaq::complex\_matrix::operator() | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmiERK10product_opI9HandlerTyE), |
|     function)](api/languages/cpp  |     [\[16\]](api/languag          |
| _api.html#_CPPv4N5cudaq14complex_ | es/cpp_api.html#_CPPv4NO5cudaq10p |
| matrixclENSt6size_tENSt6size_tE), | roduct_opmiERK15scalar_operator), |
|     [\[1\]](api/languages/cpp     |     [\[17\]](api/languages        |
| _api.html#_CPPv4NK5cudaq14complex | /cpp_api.html#_CPPv4NO5cudaq10pro |
| _matrixclENSt6size_tENSt6size_tE) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [c                            |     [\[18\]](api/languages/cpp_   |
| udaq::complex\_matrix::operator\* | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmiERR10product_opI9HandlerTyE), |
|     function)](api/langua         |     [\[19\]](api/languag          |
| ges/cpp_api.html#_CPPv4N5cudaq14c | es/cpp_api.html#_CPPv4NO5cudaq10p |
| omplex_matrixmlEN14complex_matrix | roduct_opmiERR15scalar_operator), |
| 10value_typeERK14complex_matrix), |     [\[20\]](api/languages        |
|     [\[1\]                        | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ](api/languages/cpp_api.html#_CPP | duct_opmiERR6sum_opI9HandlerTyE), |
| v4N5cudaq14complex_matrixmlERK14c |     [                             |
| omplex_matrixRK14complex_matrix), | \[21\]](api/languages/cpp_api.htm |
|                                   | l#_CPPv4NO5cudaq10product_opmiEv) |
|  [\[2\]](api/languages/cpp_api.ht | -                                 |
| ml#_CPPv4N5cudaq14complex_matrixm |    [cudaq::product\_op::operator/ |
| lERK14complex_matrixRKNSt6vectorI |     (C++                          |
| N14complex_matrix10value_typeEEE) |     function)](api/language       |
| -   [                             | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| cudaq::complex\_matrix::operator+ | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/language          |
|     function                      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| )](api/languages/cpp_api.html#_CP | roduct_opdvERR15scalar_operator), |
| Pv4N5cudaq14complex_matrixplERK14 |     [\[2\]](api/languag           |
| complex_matrixRK14complex_matrix) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [                             | roduct_opdvERK15scalar_operator), |
| cudaq::complex\_matrix::operator- |     [\[3\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     function                      | product_opdvERR15scalar_operator) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq14complex_matrixmiERK14 |   [cudaq::product\_op::operator/= |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -   [cud                          |     function)](api/langu          |
| aq::complex\_matrix::operator\[\] | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     (C++                          | product_opdVERK15scalar_operator) |
|                                   | -                                 |
|  function)](api/languages/cpp_api |    [cudaq::product\_op::operator= |
| .html#_CPPv4N5cudaq14complex_matr |     (C++                          |
| ixixERKNSt6vectorINSt6size_tEEE), |     function)](api/la             |
|     [\[1\]](api/languages/cpp_api | nguages/cpp_api.html#_CPPv4I0_NSt |
| .html#_CPPv4NK5cudaq14complex_mat | 11enable_if_tIXaantNSt7is_sameI1T |
| rixixERKNSt6vectorINSt6size_tEEE) | 9HandlerTyE5valueENSt16is_constru |
| -                                 | ctibleI9HandlerTy1TE5valueEEbEEEN |
|    [cudaq::complex\_matrix::power | 5cudaq10product_opaSER10product_o |
|     (C++                          | pI9HandlerTyERK10product_opI1TE), |
|     function)]                    |     [\[1\]](api/languages/cpp     |
| (api/languages/cpp_api.html#_CPPv | _api.html#_CPPv4N5cudaq10product_ |
| 4N5cudaq14complex_matrix5powerEi) | opaSERK10product_opI9HandlerTyE), |
| -   [                             |     [\[2\]](api/languages/cp      |
| cudaq::complex\_matrix::set\_zero | p_api.html#_CPPv4N5cudaq10product |
|     (C++                          | _opaSERR10product_opI9HandlerTyE) |
|     function)](ap                 | -                                 |
| i/languages/cpp_api.html#_CPPv4N5 |   [cudaq::product\_op::operator== |
| cudaq14complex_matrix8set_zeroEv) |     (C++                          |
| -   [c                            |     function)](api/languages/cpp  |
| udaq::complex\_matrix::to\_string | _api.html#_CPPv4NK5cudaq10product |
|     (C++                          | _opeqERK10product_opI9HandlerTyE) |
|     function)](api/               | -                                 |
| languages/cpp_api.html#_CPPv4NK5c | [cudaq::product\_op::operator\[\] |
| udaq14complex_matrix9to_stringEv) |     (C++                          |
| -   [cu                           |     function)](ap                 |
| daq::complex\_matrix::value\_type | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq10product_opixENSt6size_tE) |
|     type)](api/                   | -                                 |
| languages/cpp_api.html#_CPPv4N5cu |  [cudaq::product\_op::product\_op |
| daq14complex_matrix10value_typeE) |     (C++                          |
| -   [cudaq::contrib (C++          |     function)](api/languages/c    |
|     type)](api/languages/cpp      | pp_api.html#_CPPv4I0_NSt11enable_ |
| _api.html#_CPPv4N5cudaq7contribE) | if_tIXaaNSt7is_sameI9HandlerTy14m |
| -   [cudaq::contrib::draw (C++    | atrix_handlerE5valueEaantNSt7is_s |
|     function)]                    | ameI1T9HandlerTyE5valueENSt16is_c |
| (api/languages/cpp_api.html#_CPPv | onstructibleI9HandlerTy1TE5valueE |
| 4I0Dp0EN5cudaq7contrib4drawENSt6s | EbEEEN5cudaq10product_op10product |
| tringERR13QuantumKernelDpRR4Args) | _opERK10product_opI1TERKN14matrix |
| -   [c                            | _handler20commutation_behaviorE), |
| udaq::contrib::get\_unitary\_cmat |                                   |
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
| -   [cudaq:                       | uct_op10product_opERR9HandlerTy), |
| :depolarization2::num\_parameters |     [\[7\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     member)](api/langu            | cudaq10product_op10product_opEd), |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     [\[8\]](a                     |
| depolarization214num_parametersE) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cud                          | 5cudaq10product_op10product_opEv) |
| aq::depolarization2::num\_targets | -   [cudaq::                      |
|     (C++                          | product\_op::to\_diagonal\_matrix |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/               |
| q15depolarization211num_targetsE) | languages/cpp_api.html#_CPPv4NK5c |
| -                                 | udaq10product_op18to_diagonal_mat |
|   [cudaq::depolarization\_channel | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     class)](                      | apINSt6stringENSt7complexIdEEEEb) |
| api/languages/cpp_api.html#_CPPv4 | -                                 |
| N5cudaq22depolarization_channelE) |   [cudaq::product\_op::to\_matrix |
| -   [cudaq::depolar               |     (C++                          |
| ization\_channel::num\_parameters |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     member)](api/languages/cp     | _CPPv4NK5cudaq10product_op9to_mat |
| p_api.html#_CPPv4N5cudaq22depolar | rixENSt13unordered_mapINSt6size_t |
| ization_channel14num_parametersE) | ENSt7int64_tEEERKNSt13unordered_m |
| -   [cudaq::depo                  | apINSt6stringENSt7complexIdEEEEb) |
| larization\_channel::num\_targets | -   [cudaq                        |
|     (C++                          | ::product\_op::to\_sparse\_matrix |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq22depo |     function)](ap                 |
| larization_channel11num_targetsE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::details (C++          | 5cudaq10product_op16to_sparse_mat |
|     type)](api/languages/cpp      | rixENSt13unordered_mapINSt6size_t |
| _api.html#_CPPv4N5cudaq7detailsE) | ENSt7int64_tEEERKNSt13unordered_m |
| -   [cudaq::details::future (C++  | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -                                 |
|  class)](api/languages/cpp_api.ht |   [cudaq::product\_op::to\_string |
| ml#_CPPv4N5cudaq7details6futureE) |     (C++                          |
| -                                 |     function)](                   |
|   [cudaq::details::future::future | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq10product_op9to_stringEv) |
|     functio                       | -   [                             |
| n)](api/languages/cpp_api.html#_C | cudaq::product\_op::\~product\_op |
| PPv4N5cudaq7details6future6future |     (C++                          |
| ERNSt6vectorI3JobEERNSt6stringERN |     fu                            |
| St3mapINSt6stringENSt6stringEEE), | nction)](api/languages/cpp_api.ht |
|     [\[1\]](api/lang              | ml#_CPPv4N5cudaq10product_opD0Ev) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -   [cudaq::QPU (C++              |
| details6future6futureERR6future), |     class)](api/languages         |
|     [\[2\]]                       | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::QPU::enqueue (C++     |
| 4N5cudaq7details6future6futureEv) |     function)](ap                 |
| -   [cuda                         | i/languages/cpp_api.html#_CPPv4N5 |
| q::details::kernel\_builder\_base | cudaq3QPU7enqueueER11QuantumTask) |
|     (C++                          | -   [cudaq::QPU::getConnectivity  |
|     class)](api/l                 |     (C++                          |
| anguages/cpp_api.html#_CPPv4N5cud |     function)                     |
| aq7details19kernel_builder_baseE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::details::ke           | v4N5cudaq3QPU15getConnectivityEv) |
| rnel\_builder\_base::operator\<\< | -                                 |
|     (C++                          | [cudaq::QPU::getExecutionThreadId |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq7de |     function)](api/               |
| tails19kernel_builder_baselsERNSt | languages/cpp_api.html#_CPPv4NK5c |
| 7ostreamERK19kernel_builder_base) | udaq3QPU20getExecutionThreadIdEv) |
| -   [                             | -   [cudaq::QPU::getNumQubits     |
| cudaq::details::KernelBuilderType |     (C++                          |
|     (C++                          |     functi                        |
|     class)](api                   | on)](api/languages/cpp_api.html#_ |
| /languages/cpp_api.html#_CPPv4N5c | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| udaq7details17KernelBuilderTypeE) | -   [                             |
| -   [cudaq::d                     | cudaq::QPU::getRemoteCapabilities |
| etails::KernelBuilderType::create |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)                     | anguages/cpp_api.html#_CPPv4NK5cu |
| ](api/languages/cpp_api.html#_CPP | daq3QPU21getRemoteCapabilitiesEv) |
| v4N5cudaq7details17KernelBuilderT | -   [cudaq::QPU::isEmulated (C++  |
| ype6createEPN4mlir11MLIRContextE) |     func                          |
| -   [cudaq::details::Ker          | tion)](api/languages/cpp_api.html |
| nelBuilderType::KernelBuilderType | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     (C++                          | -   [cudaq::QPU::isSimulator (C++ |
|     function)](api/lang           |     funct                         |
| uages/cpp_api.html#_CPPv4N5cudaq7 | ion)](api/languages/cpp_api.html# |
| details17KernelBuilderType17Kerne | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| lBuilderTypeERRNSt8functionIFN4ml | -   [cudaq::QPU::launchKernel     |
| ir4TypeEPN4mlir11MLIRContextEEEE) |     (C++                          |
| -                                 |     function)](api/               |
|    [cudaq::diag\_matrix\_callback | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq3QPU12launchKernelERKNSt6strin |
|     class)                        | gE15KernelThunkTypePvNSt8uint64_t |
| ](api/languages/cpp_api.html#_CPP | ENSt8uint64_tERKNSt6vectorIPvEE), |
| v4N5cudaq20diag_matrix_callbackE) |                                   |
| -   [cudaq::dyn (C++              |  [\[1\]](api/languages/cpp_api.ht |
|     member)](api/languages        | ml#_CPPv4N5cudaq3QPU12launchKerne |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | lERKNSt6stringERKNSt6vectorIPvEE) |
| -   [cudaq::ExecutionContext (C++ | -   [cudaq::Q                     |
|     cl                            | PU::launchSerializedCodeExecution |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16ExecutionContextE) |     function)]                    |
| -   [cudaq                        | (api/languages/cpp_api.html#_CPPv |
| ::ExecutionContext::amplitudeMaps | 4N5cudaq3QPU29launchSerializedCod |
|     (C++                          | eExecutionERKNSt6stringERN5cudaq3 |
|     member)](api/langu            | 0SerializedCodeExecutionContextE) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [cudaq::QPU::onRandomSeedSet  |
| ExecutionContext13amplitudeMapsE) |     (C++                          |
| -   [c                            |     function)](api/lang           |
| udaq::ExecutionContext::asyncExec | uages/cpp_api.html#_CPPv4N5cudaq3 |
|     (C++                          | QPU15onRandomSeedSetENSt6size_tE) |
|     member)](api/                 | -   [cudaq::QPU::QPU (C++         |
| languages/cpp_api.html#_CPPv4N5cu |     functio                       |
| daq16ExecutionContext9asyncExecE) | n)](api/languages/cpp_api.html#_C |
| -   [cud                          | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| aq::ExecutionContext::asyncResult |                                   |
|     (C++                          |  [\[1\]](api/languages/cpp_api.ht |
|     member)](api/lan              | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[2\]](api/languages/cpp_    |
| 16ExecutionContext11asyncResultE) | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| -   [cudaq:                       | -   [                             |
| :ExecutionContext::batchIteration | cudaq::QPU::resetExecutionContext |
|     (C++                          |     (C++                          |
|     member)](api/langua           |     function)](api/               |
| ges/cpp_api.html#_CPPv4N5cudaq16E | languages/cpp_api.html#_CPPv4N5cu |
| xecutionContext14batchIterationE) | daq3QPU21resetExecutionContextEv) |
| -   [cudaq::E                     | -                                 |
| xecutionContext::canHandleObserve |  [cudaq::QPU::setExecutionContext |
|     (C++                          |     (C++                          |
|     member)](api/language         |                                   |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |   function)](api/languages/cpp_ap |
| cutionContext16canHandleObserveE) | i.html#_CPPv4N5cudaq3QPU19setExec |
| -   [cudaq::E                     | utionContextEP16ExecutionContext) |
| xecutionContext::ExecutionContext | -   [cudaq::QPU::setId (C++       |
|     (C++                          |     function                      |
|     func                          | )](api/languages/cpp_api.html#_CP |
| tion)](api/languages/cpp_api.html | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| #_CPPv4N5cudaq16ExecutionContext1 | -   [cudaq::QPU::setShots (C++    |
| 6ExecutionContextERKNSt6stringE), |     f                             |
|     [\[1\]](api                   | unction)](api/languages/cpp_api.h |
| /languages/cpp_api.html#_CPPv4N5c | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| udaq16ExecutionContext16Execution | -   [cudaq:                       |
| ContextERKNSt6stringENSt6size_tE) | :QPU::supportsConditionalFeedback |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::expectationValue |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq3QP |
|     member)](api/language         | U27supportsConditionalFeedbackEv) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::                      |
| cutionContext16expectationValueE) | QPU::supportsExplicitMeasurements |
| -   [cudaq::Execu                 |     (C++                          |
| tionContext::explicitMeasurements |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq3QPU |
|     member)](api/languages/cp     | 28supportsExplicitMeasurementsEv) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::QPU::\~QPU (C++       |
| onContext20explicitMeasurementsE) |     function)](api/languages/cp   |
| -   [cuda                         | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| q::ExecutionContext::futureResult | -   [cudaq::QPUState (C++         |
|     (C++                          |     class)](api/languages/cpp_    |
|     member)](api/lang             | api.html#_CPPv4N5cudaq8QPUStateE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::qreg (C++             |
| 6ExecutionContext12futureResultE) |     class)](api/lang              |
| -   [cudaq::ExecutionContext      | uages/cpp_api.html#_CPPv4I_NSt6si |
| ::hasConditionalsOnMeasureResults | ze_tE_NSt6size_tE0EN5cudaq4qregE) |
|     (C++                          | -   [cudaq::qreg::back (C++       |
|     mem                           |     function)                     |
| ber)](api/languages/cpp_api.html# | ](api/languages/cpp_api.html#_CPP |
| _CPPv4N5cudaq16ExecutionContext31 | v4N5cudaq4qreg4backENSt6size_tE), |
| hasConditionalsOnMeasureResultsE) |     [\[1\]](api/languages/cpp_ap  |
| -   [cudaq::Executi               | i.html#_CPPv4N5cudaq4qreg4backEv) |
| onContext::invocationResultBuffer | -   [cudaq::qreg::begin (C++      |
|     (C++                          |                                   |
|     member)](api/languages/cpp_   |  function)](api/languages/cpp_api |
| api.html#_CPPv4N5cudaq16Execution | .html#_CPPv4N5cudaq4qreg5beginEv) |
| Context22invocationResultBufferE) | -   [cudaq::qreg::clear (C++      |
| -   [cu                           |                                   |
| daq::ExecutionContext::kernelName |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     member)](api/la               | -   [cudaq::qreg::front (C++      |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)]                    |
| q16ExecutionContext10kernelNameE) | (api/languages/cpp_api.html#_CPPv |
| -   [cud                          | 4N5cudaq4qreg5frontENSt6size_tE), |
| aq::ExecutionContext::kernelTrace |     [\[1\]](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5frontEv) |
|     member)](api/lan              | -   [cudaq::qreg::operator\[\]    |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11kernelTraceE) |     functi                        |
| -   [cudaq::                      | on)](api/languages/cpp_api.html#_ |
| ExecutionContext::msm\_dimensions | CPPv4N5cudaq4qregixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qreg::size (C++       |
|     member)](api/langua           |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq16E |  function)](api/languages/cpp_api |
| xecutionContext14msm_dimensionsE) | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| -   [cudaq::Exe                   | -   [cudaq::qreg::slice (C++      |
| cutionContext::msm\_prob\_err\_id |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq4q |
|     member)](api/languag          | reg5sliceENSt6size_tENSt6size_tE) |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | -   [cudaq::qreg::value\_type     |
| ecutionContext15msm_prob_err_idE) |     (C++                          |
| -   [cudaq::Exe                   |                                   |
| cutionContext::msm\_probabilities | type)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq4qreg10value_typeE) |
|     member)](api/languages        | -   [cudaq::qspan (C++            |
| /cpp_api.html#_CPPv4N5cudaq16Exec |     class)](api/lang              |
| utionContext17msm_probabilitiesE) | uages/cpp_api.html#_CPPv4I_NSt6si |
| -                                 | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|    [cudaq::ExecutionContext::name | -   [cudaq::QuakeValue (C++       |
|     (C++                          |     class)](api/languages/cpp_api |
|     member)]                      | .html#_CPPv4N5cudaq10QuakeValueE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::Q                     |
| 4N5cudaq16ExecutionContext4nameE) | uakeValue::canValidateNumElements |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::noiseModel |     function)](api/languages      |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq10Quak |
|     member)](api/la               | eValue22canValidateNumElementsEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -                                 |
| q16ExecutionContext10noiseModelE) |  [cudaq::QuakeValue::constantSize |
| -   [cudaq::Exe                   |     (C++                          |
| cutionContext::numberTrajectories |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/languages/       | udaq10QuakeValue12constantSizeEv) |
| cpp_api.html#_CPPv4N5cudaq16Execu | -   [cudaq::QuakeValue::dump (C++ |
| tionContext18numberTrajectoriesE) |     function)](api/lan            |
| -   [c                            | guages/cpp_api.html#_CPPv4N5cudaq |
| udaq::ExecutionContext::optResult | 10QuakeValue4dumpERNSt7ostreamE), |
|     (C++                          |     [\                            |
|     member)](api/                 | [1\]](api/languages/cpp_api.html# |
| languages/cpp_api.html#_CPPv4N5cu | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| daq16ExecutionContext9optResultE) | -   [cudaq                        |
| -   [cudaq::Execu                 | ::QuakeValue::getRequiredElements |
| tionContext::overlapComputeStates |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     member)](api/languages/cp     | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| p_api.html#_CPPv4N5cudaq16Executi | uakeValue19getRequiredElementsEv) |
| onContext20overlapComputeStatesE) | -   [cudaq::QuakeValue::getValue  |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::overlapResult |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     member)](api/langu            | 4NK5cudaq10QuakeValue8getValueEv) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [cudaq::QuakeValue::inverse   |
| ExecutionContext13overlapResultE) |     (C++                          |
| -   [cudaq                        |     function)                     |
| ::ExecutionContext::registerNames | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq10QuakeValue7inverseEv) |
|     member)](api/langu            | -   [cudaq::QuakeValue::isStdVec  |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13registerNamesE) |     function)                     |
| -   [cu                           | ](api/languages/cpp_api.html#_CPP |
| daq::ExecutionContext::reorderIdx | v4N5cudaq10QuakeValue8isStdVecEv) |
|     (C++                          | -                                 |
|     member)](api/la               |    [cudaq::QuakeValue::operator\* |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10reorderIdxE) |     function)](api                |
| -                                 | /languages/cpp_api.html#_CPPv4N5c |
|  [cudaq::ExecutionContext::result | udaq10QuakeValuemlE10QuakeValue), |
|     (C++                          |                                   |
|     member)](a                    | [\[1\]](api/languages/cpp_api.htm |
| pi/languages/cpp_api.html#_CPPv4N | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| 5cudaq16ExecutionContext6resultE) | -   [cudaq::QuakeValue::operator+ |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::shots |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](                     | udaq10QuakeValueplE10QuakeValue), |
| api/languages/cpp_api.html#_CPPv4 |     [                             |
| N5cudaq16ExecutionContext5shotsE) | \[1\]](api/languages/cpp_api.html |
| -   [cudaq::                      | #_CPPv4N5cudaq10QuakeValueplEKd), |
| ExecutionContext::simulationState |                                   |
|     (C++                          | [\[2\]](api/languages/cpp_api.htm |
|     member)](api/languag          | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | -   [cudaq::QuakeValue::operator- |
| ecutionContext15simulationStateE) |     (C++                          |
| -                                 |     function)](api                |
|    [cudaq::ExecutionContext::spin | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuemiE10QuakeValue), |
|     member)]                      |     [                             |
| (api/languages/cpp_api.html#_CPPv | \[1\]](api/languages/cpp_api.html |
| 4N5cudaq16ExecutionContext4spinE) | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| -   [cudaq::                      |     [                             |
| ExecutionContext::totalIterations | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     member)](api/languag          |                                   |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | [\[3\]](api/languages/cpp_api.htm |
| ecutionContext15totalIterationsE) | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| -   [cudaq::ExecutionResult (C++  | -   [cudaq::QuakeValue::operator/ |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |     function)](api                |
| #_CPPv4N5cudaq15ExecutionResultE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cud                          | udaq10QuakeValuedvE10QuakeValue), |
| aq::ExecutionResult::appendResult |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     functio                       | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| n)](api/languages/cpp_api.html#_C | -                                 |
| PPv4N5cudaq15ExecutionResult12app |  [cudaq::QuakeValue::operator\[\] |
| endResultENSt6stringENSt6size_tE) |     (C++                          |
| -   [cu                           |     function)](api                |
| daq::ExecutionResult::deserialize | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValueixEKNSt6size_tE), |
|     function)                     |     [\[1\]](api/                  |
| ](api/languages/cpp_api.html#_CPP | languages/cpp_api.html#_CPPv4N5cu |
| v4N5cudaq15ExecutionResult11deser | daq10QuakeValueixERK10QuakeValue) |
| ializeERNSt6vectorINSt6size_tEEE) | -                                 |
| -   [cudaq:                       |    [cudaq::QuakeValue::QuakeValue |
| :ExecutionResult::ExecutionResult |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     functio                       | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| n)](api/languages/cpp_api.html#_C | akeValue10QuakeValueERN4mlir20Imp |
| PPv4N5cudaq15ExecutionResult15Exe | licitLocOpBuilderEN4mlir5ValueE), |
| cutionResultE16CountsDictionary), |     [\[1\]                        |
|     [\[1\]](api/lan               | ](api/languages/cpp_api.html#_CPP |
| guages/cpp_api.html#_CPPv4N5cudaq | v4N5cudaq10QuakeValue10QuakeValue |
| 15ExecutionResult15ExecutionResul | ERN4mlir20ImplicitLocOpBuilderEd) |
| tE16CountsDictionaryNSt6stringE), | -   [cudaq::QuakeValue::size (C++ |
|     [\[2\                         |     funct                         |
| ]](api/languages/cpp_api.html#_CP | ion)](api/languages/cpp_api.html# |
| Pv4N5cudaq15ExecutionResult15Exec | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| utionResultE16CountsDictionaryd), | -   [cudaq::QuakeValue::slice     |
|                                   |     (C++                          |
|    [\[3\]](api/languages/cpp_api. |     function)](api/languages/cpp_ |
| html#_CPPv4N5cudaq15ExecutionResu | api.html#_CPPv4N5cudaq10QuakeValu |
| lt15ExecutionResultENSt6stringE), | e5sliceEKNSt6size_tEKNSt6size_tE) |
|     [\[4\                         | -   [cudaq::quantum\_platform     |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     cl                            |
| utionResultERK15ExecutionResult), | ass)](api/languages/cpp_api.html# |
|     [\[5\]](api/language          | _CPPv4N5cudaq16quantum_platformE) |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | -   [cudaq                        |
| cutionResult15ExecutionResultEd), | ::quantum\_platform::clear\_shots |
|     [\[6\]](api/languag           |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |     function)](api/lang           |
| ecutionResult15ExecutionResultEv) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [                             | 6quantum_platform11clear_shotsEv) |
| cudaq::ExecutionResult::operator= | -   [cudaq                        |
|     (C++                          | ::quantum\_platform::connectivity |
|     function)](api/languages/     |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq15Execu |     function)](api/langu          |
| tionResultaSERK15ExecutionResult) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [c                            | quantum_platform12connectivityEv) |
| udaq::ExecutionResult::operator== | -   [cudaq::qu                    |
|     (C++                          | antum\_platform::enqueueAsyncTask |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4NK5cudaq15Execu |     function)](api/languages/     |
| tionResulteqERK15ExecutionResult) | cpp_api.html#_CPPv4N5cudaq16quant |
| -   [cud                          | um_platform16enqueueAsyncTaskEKNS |
| aq::ExecutionResult::registerName | t6size_tER19KernelExecutionTask), |
|     (C++                          |     [\[1\]](api/languag           |
|     member)](api/lan              | es/cpp_api.html#_CPPv4N5cudaq16qu |
| guages/cpp_api.html#_CPPv4N5cudaq | antum_platform16enqueueAsyncTaskE |
| 15ExecutionResult12registerNameE) | KNSt6size_tERNSt8functionIFvvEEE) |
| -   [cudaq                        | -   [cudaq::quantu                |
| ::ExecutionResult::sequentialData | m\_platform::get\_codegen\_config |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api/languages/c    |
| ages/cpp_api.html#_CPPv4N5cudaq15 | pp_api.html#_CPPv4N5cudaq16quantu |
| ExecutionResult14sequentialDataE) | m_platform18get_codegen_configEv) |
| -   [                             | -   [cudaq::qua                   |
| cudaq::ExecutionResult::serialize | ntum\_platform::get\_current\_qpu |
|     (C++                          |     (C++                          |
|     function)](api/l              |     function)](api/language       |
| anguages/cpp_api.html#_CPPv4NK5cu | s/cpp_api.html#_CPPv4N5cudaq16qua |
| daq15ExecutionResult9serializeEv) | ntum_platform15get_current_qpuEv) |
| -   [cudaq::fermion\_handler (C++ | -   [cudaq::                      |
|     c                             | quantum\_platform::get\_exec\_ctx |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15fermion_handlerE) |     function)](api/langua         |
| -   [cudaq::fermion\_op (C++      | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|     type)](api/languages/cpp_api  | quantum_platform12get_exec_ctxEv) |
| .html#_CPPv4N5cudaq10fermion_opE) | -   [cud                          |
| -   [cudaq::fermion\_op\_term     | aq::quantum\_platform::get\_noise |
|     (C++                          |     (C++                          |
|                                   |     function)](api/l              |
| type)](api/languages/cpp_api.html | anguages/cpp_api.html#_CPPv4N5cud |
| #_CPPv4N5cudaq15fermion_op_termE) | aq16quantum_platform9get_noiseEv) |
| -   [cudaq::FermioniqBaseQPU (C++ | -   [cudaq::qu                    |
|     cl                            | antum\_platform::get\_num\_qubits |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16FermioniqBaseQPUE) |                                   |
| -   [cudaq::get\_state (C++       | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4N5cudaq16quantum_platf |
|    function)](api/languages/cpp_a | orm14get_num_qubitsENSt6size_tE), |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     [\[1\]](api/languag           |
| ateEDaRR13QuantumKernelDpRR4Args) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cudaq::gradient (C++         | antum_platform14get_num_qubitsEv) |
|     class)](api/languages/cpp_    | -   [cudaq::quantum\_pl           |
| api.html#_CPPv4N5cudaq8gradientE) | atform::get\_remote\_capabilities |
| -   [cudaq::gradient::clone (C++  |     (C++                          |
|     fun                           |     function)]                    |
| ction)](api/languages/cpp_api.htm | (api/languages/cpp_api.html#_CPPv |
| l#_CPPv4N5cudaq8gradient5cloneEv) | 4NK5cudaq16quantum_platform23get_ |
| -   [cudaq::gradient::compute     | remote_capabilitiesEKNSt6size_tE) |
|     (C++                          | -   [cud                          |
|     function)](api/language       | aq::quantum\_platform::get\_shots |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     (C++                          |
| ient7computeERKNSt6vectorIdEERKNS |     function)](api/l              |
| t8functionIFdNSt6vectorIdEEEEEd), | anguages/cpp_api.html#_CPPv4N5cud |
|     [\[1\]](ap                    | aq16quantum_platform9get_shotsEv) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq                        |
| cudaq8gradient7computeERKNSt6vect | ::quantum\_platform::getLogStream |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradient::gradient    |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     function)](api/lang           | quantum_platform12getLogStreamEv) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -   [cudaq                        |
| daq8gradient8gradientER7KernelT), | ::quantum\_platform::is\_emulated |
|                                   |     (C++                          |
|    [\[1\]](api/languages/cpp_api. |                                   |
| html#_CPPv4I00EN5cudaq8gradient8g |   function)](api/languages/cpp_ap |
| radientER7KernelTRR10ArgsMapper), | i.html#_CPPv4NK5cudaq16quantum_pl |
|     [\[2\                         | atform11is_emulatedEKNSt6size_tE) |
| ]](api/languages/cpp_api.html#_CP | -   [cud                          |
| Pv4I00EN5cudaq8gradient8gradientE | aq::quantum\_platform::is\_remote |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3                          |     function)](api/languages/cp   |
| \]](api/languages/cpp_api.html#_C | p_api.html#_CPPv4N5cudaq16quantum |
| PPv4N5cudaq8gradient8gradientERRN | _platform9is_remoteEKNSt6size_tE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq:                       |
|     [\[                           | :quantum\_platform::is\_simulator |
| 4\]](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8gradient8gradientEv) |                                   |
| -   [cudaq::gradient::setArgs     |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4NK5cudaq16quantum_pla |
|     fu                            | tform12is_simulatorEKNSt6size_tE) |
| nction)](api/languages/cpp_api.ht | -   [cu                           |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | daq::quantum\_platform::launchVQE |
| tArgsEvR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::gradient::setKernel   |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function)](api/languages/c    | html#_CPPv4N5cudaq16quantum_platf |
| pp_api.html#_CPPv4I0EN5cudaq8grad | orm9launchVQEEKNSt6stringEPKvPN5c |
| ient9setKernelEvR13QuantumKernel) | udaq8gradientERKN5cudaq7spin_opER |
| -   [cuda                         | N5cudaq9optimizerEKiKNSt6size_tE) |
| q::gradients::central\_difference | -   [cudaq::q                     |
|     (C++                          | uantum\_platform::list\_platforms |
|     class)](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languag        |
| q9gradients18central_differenceE) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cudaq::grad                  | antum_platform14list_platformsEv) |
| ients::central\_difference::clone | -                                 |
|     (C++                          |   [cudaq::quantum\_platform::name |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)](a                  |
| ents18central_difference5cloneEv) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::gradie                | K5cudaq16quantum_platform4nameEv) |
| nts::central\_difference::compute | -   [cu                           |
|     (C++                          | daq::quantum\_platform::num\_qpus |
|     function)](                   |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/l              |
| N5cudaq9gradients18central_differ | anguages/cpp_api.html#_CPPv4NK5cu |
| ence7computeERKNSt6vectorIdEERKNS | daq16quantum_platform8num_qpusEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::q                     |
|                                   | uantum\_platform::onRandomSeedSet |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq9gradients18cent |                                   |
| ral_difference7computeERKNSt6vect | function)](api/languages/cpp_api. |
| orIdEERNSt6vectorIdEERK7spin_opd) | html#_CPPv4N5cudaq16quantum_platf |
| -   [cudaq::gradien               | orm15onRandomSeedSetENSt6size_tE) |
| ts::central\_difference::gradient | -   [cudaq::qu                    |
|     (C++                          | antum\_platform::reset\_exec\_ctx |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |                                   |
| PPv4I00EN5cudaq9gradients18centra |  function)](api/languages/cpp_api |
| l_difference8gradientER7KernelT), | .html#_CPPv4N5cudaq16quantum_plat |
|     [\[1\]](api/langua            | form14reset_exec_ctxENSt6size_tE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq                        |
| q9gradients18central_difference8g | ::quantum\_platform::reset\_noise |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\]](api/languages/cpp_    |     function)](api/lang           |
| api.html#_CPPv4I00EN5cudaq9gradie | uages/cpp_api.html#_CPPv4N5cudaq1 |
| nts18central_difference8gradientE | 6quantum_platform11reset_noiseEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::                      |
|     [\[3\]](api/languages/cpp     | quantum\_platform::resetLogStream |
| _api.html#_CPPv4N5cudaq9gradients |     (C++                          |
| 18central_difference8gradientERRN |     function)](api/languag        |
| St8functionIFvNSt6vectorIdEEEEE), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[4\]](api/languages/cp      | antum_platform14resetLogStreamEv) |
| p_api.html#_CPPv4N5cudaq9gradient | -   [cudaq::qua                   |
| s18central_difference8gradientEv) | ntum\_platform::set\_current\_qpu |
| -   [cuda                         |     (C++                          |
| q::gradients::forward\_difference |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     class)](api/la                | tml#_CPPv4N5cudaq16quantum_platfo |
| nguages/cpp_api.html#_CPPv4N5cuda | rm15set_current_qpuEKNSt6size_tE) |
| q9gradients18forward_differenceE) | -   [cudaq::                      |
| -   [cudaq::grad                  | quantum\_platform::set\_exec\_ctx |
| ients::forward\_difference::clone |     (C++                          |
|     (C++                          |     function)](api/languages      |
|     function)](api/languages      | /cpp_api.html#_CPPv4N5cudaq16quan |
| /cpp_api.html#_CPPv4N5cudaq9gradi | tum_platform12set_exec_ctxEPN5cud |
| ents18forward_difference5cloneEv) | aq16ExecutionContextENSt6size_tE) |
| -   [cudaq::gradie                | -   [cud                          |
| nts::forward\_difference::compute | aq::quantum\_platform::set\_noise |
|     (C++                          |     (C++                          |
|     function)](                   |                                   |
| api/languages/cpp_api.html#_CPPv4 |    function)](api/languages/cpp_a |
| N5cudaq9gradients18forward_differ | pi.html#_CPPv4N5cudaq16quantum_pl |
| ence7computeERKNSt6vectorIdEERKNS | atform9set_noiseEPK11noise_model) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cud                          |
|                                   | aq::quantum\_platform::set\_shots |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq9gradients18forw |     function)](api/l              |
| ard_difference7computeERKNSt6vect | anguages/cpp_api.html#_CPPv4N5cud |
| orIdEERNSt6vectorIdEERK7spin_opd) | aq16quantum_platform9set_shotsEi) |
| -   [cudaq::gradien               | -   [cudaq                        |
| ts::forward\_difference::gradient | ::quantum\_platform::setLogStream |
|     (C++                          |     (C++                          |
|     functio                       |                                   |
| n)](api/languages/cpp_api.html#_C |  function)](api/languages/cpp_api |
| PPv4I00EN5cudaq9gradients18forwar | .html#_CPPv4N5cudaq16quantum_plat |
| d_difference8gradientER7KernelT), | form12setLogStreamERNSt7ostreamE) |
|     [\[1\]](api/langua            | -   [cudaq::qu                    |
| ges/cpp_api.html#_CPPv4I00EN5cuda | antum\_platform::setTargetBackend |
| q9gradients18forward_difference8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     fun                           |
|     [\[2\]](api/languages/cpp_    | ction)](api/languages/cpp_api.htm |
| api.html#_CPPv4I00EN5cudaq9gradie | l#_CPPv4N5cudaq16quantum_platform |
| nts18forward_difference8gradientE | 16setTargetBackendERKNSt6stringE) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::quantum\_platform     |
|     [\[3\]](api/languages/cpp     | ::supports\_conditional\_feedback |
| _api.html#_CPPv4N5cudaq9gradients |     (C++                          |
| 18forward_difference8gradientERRN |     function)](api/l              |
| St8functionIFvNSt6vectorIdEEEEE), | anguages/cpp_api.html#_CPPv4NK5cu |
|     [\[4\]](api/languages/cp      | daq16quantum_platform29supports_c |
| p_api.html#_CPPv4N5cudaq9gradient | onditional_feedbackEKNSt6size_tE) |
| s18forward_difference8gradientEv) | -   [cudaq::quantum\_platform:    |
| -   [c                            | :supports\_explicit\_measurements |
| udaq::gradients::parameter\_shift |     (C++                          |
|     (C++                          |     function)](api/la             |
|     class)](api                   | nguages/cpp_api.html#_CPPv4NK5cud |
| /languages/cpp_api.html#_CPPv4N5c | aq16quantum_platform30supports_ex |
| udaq9gradients15parameter_shiftE) | plicit_measurementsEKNSt6size_tE) |
| -   [cudaq::g                     | -   [cudaq::quantum\_platf        |
| radients::parameter\_shift::clone | orm::supports\_task\_distribution |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     fu                            |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | nction)](api/languages/cpp_api.ht |
| adients15parameter_shift5cloneEv) | ml#_CPPv4NK5cudaq16quantum_platfo |
| -   [cudaq::gra                   | rm26supports_task_distributionEv) |
| dients::parameter\_shift::compute | -   [cudaq::QuantumTask (C++      |
|     (C++                          |     type)](api/languages/cpp_api. |
|     function                      | html#_CPPv4N5cudaq11QuantumTaskE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qubit (C++            |
| Pv4N5cudaq9gradients15parameter_s |     type)](api/languages/c        |
| hift7computeERKNSt6vectorIdEERKNS | pp_api.html#_CPPv4N5cudaq5qubitE) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::QubitConnectivity     |
|     [\[1\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4N5cudaq9gradients15p |     ty                            |
| arameter_shift7computeERKNSt6vect | pe)](api/languages/cpp_api.html#_ |
| orIdEERNSt6vectorIdEERK7spin_opd) | CPPv4N5cudaq17QubitConnectivityE) |
| -   [cudaq::grad                  | -   [cudaq::QubitEdge (C++        |
| ients::parameter\_shift::gradient |     type)](api/languages/cpp_a    |
|     (C++                          | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|     func                          | -   [cudaq::qudit (C++            |
| tion)](api/languages/cpp_api.html |     clas                          |
| #_CPPv4I00EN5cudaq9gradients15par | s)](api/languages/cpp_api.html#_C |
| ameter_shift8gradientER7KernelT), | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     [\[1\]](api/lan               | -   [cudaq::qudit::qudit (C++     |
| guages/cpp_api.html#_CPPv4I00EN5c |                                   |
| udaq9gradients15parameter_shift8g | function)](api/languages/cpp_api. |
| radientER7KernelTRR10ArgsMapper), | html#_CPPv4N5cudaq5qudit5quditEv) |
|     [\[2\]](api/languages/c       | -   [cudaq::qvector (C++          |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     class)                        |
| dients15parameter_shift8gradientE | ](api/languages/cpp_api.html#_CPP |
| RR13QuantumKernelRR10ArgsMapper), | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     [\[3\]](api/languages/        | -   [cudaq::qvector::back (C++    |
| cpp_api.html#_CPPv4N5cudaq9gradie |     function)](a                  |
| nts15parameter_shift8gradientERRN | pi/languages/cpp_api.html#_CPPv4N |
| St8functionIFvNSt6vectorIdEEEEE), | 5cudaq7qvector4backENSt6size_tE), |
|     [\[4\]](api/languages         |                                   |
| /cpp_api.html#_CPPv4N5cudaq9gradi |   [\[1\]](api/languages/cpp_api.h |
| ents15parameter_shift8gradientEv) | tml#_CPPv4N5cudaq7qvector4backEv) |
| -   [cudaq::kernel\_builder (C++  | -   [cudaq::qvector::begin (C++   |
|     clas                          |     fu                            |
| s)](api/languages/cpp_api.html#_C | nction)](api/languages/cpp_api.ht |
| PPv4IDpEN5cudaq14kernel_builderE) | ml#_CPPv4N5cudaq7qvector5beginEv) |
| -   [cu                           | -   [cudaq::qvector::clear (C++   |
| daq::kernel\_builder::constantVal |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function)](api/la             | ml#_CPPv4N5cudaq7qvector5clearEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::qvector::end (C++     |
| q14kernel_builder11constantValEd) |                                   |
| -   [cud                          | function)](api/languages/cpp_api. |
| aq::kernel\_builder::getArguments | html#_CPPv4N5cudaq7qvector3endEv) |
|     (C++                          | -   [cudaq::qvector::front (C++   |
|     function)](api/lan            |     function)](ap                 |
| guages/cpp_api.html#_CPPv4N5cudaq | i/languages/cpp_api.html#_CPPv4N5 |
| 14kernel_builder12getArgumentsEv) | cudaq7qvector5frontENSt6size_tE), |
| -   [cud                          |                                   |
| aq::kernel\_builder::getNumParams |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     function)](api/lan            | -   [cudaq::qvector::operator=    |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 14kernel_builder12getNumParamsEv) |     functio                       |
| -   [cu                           | n)](api/languages/cpp_api.html#_C |
| daq::kernel\_builder::isArgStdVec | PPv4N5cudaq7qvectoraSERK7qvector) |
|     (C++                          | -   [cudaq::qvector::operator\[\] |
|     function)](api/languages/cp   |     (C++                          |
| p_api.html#_CPPv4N5cudaq14kernel_ |     function)                     |
| builder11isArgStdVecENSt6size_tE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq:                       | v4N5cudaq7qvectorixEKNSt6size_tE) |
| :kernel\_builder::kernel\_builder | -   [cudaq::qvector::qvector (C++ |
|     (C++                          |     function)](api/               |
|     function)](api/languages/cpp_ | languages/cpp_api.html#_CPPv4N5cu |
| api.html#_CPPv4N5cudaq14kernel_bu | daq7qvector7qvectorENSt6size_tE), |
| ilder14kernel_builderERNSt6vector |     [\[1\]](a                     |
| IN7details17KernelBuilderTypeEEE) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::kernel\_builder::name | 5cudaq7qvector7qvectorERK5state), |
|     (C++                          |     [\[2\]](api                   |
|     function)                     | /languages/cpp_api.html#_CPPv4N5c |
| ](api/languages/cpp_api.html#_CPP | udaq7qvector7qvectorERK7qvector), |
| v4N5cudaq14kernel_builder4nameEv) |     [\[3\]](api/languages/cpp     |
| -                                 | _api.html#_CPPv4N5cudaq7qvector7q |
|   [cudaq::kernel\_builder::qalloc | vectorERKNSt6vectorI7complexEEb), |
|     (C++                          |     [\[4\]](ap                    |
|     function)](api/language       | i/languages/cpp_api.html#_CPPv4N5 |
| s/cpp_api.html#_CPPv4N5cudaq14ker | cudaq7qvector7qvectorERR7qvector) |
| nel_builder6qallocE10QuakeValue), | -   [cudaq::qvector::size (C++    |
|     [\[1\]](api/language          |     fu                            |
| s/cpp_api.html#_CPPv4N5cudaq14ker | nction)](api/languages/cpp_api.ht |
| nel_builder6qallocEKNSt6size_tE), | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     [\[2                          | -   [cudaq::qvector::slice (C++   |
| \]](api/languages/cpp_api.html#_C |     function)](api/language       |
| PPv4N5cudaq14kernel_builder6qallo | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| cERNSt6vectorINSt7complexIdEEEE), | tor5sliceENSt6size_tENSt6size_tE) |
|     [\[3\]](                      | -   [cudaq::qvector::value\_type  |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14kernel_builder6qallocEv) |     typ                           |
| -   [cudaq::kernel\_builder::swap | e)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq7qvector10value_typeE) |
|     function)](api/language       | -   [cudaq::qview (C++            |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     clas                          |
| 4kernel_builder4swapEvRK10QuakeVa | s)](api/languages/cpp_api.html#_C |
| lueRK10QuakeValueRK10QuakeValue), | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|                                   | -   [cudaq::qview::value\_type    |
| [\[1\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4I00EN5cudaq14kernel_build |     t                             |
| er4swapEvRKNSt6vectorI10QuakeValu | ype)](api/languages/cpp_api.html# |
| eEERK10QuakeValueRK10QuakeValue), | _CPPv4N5cudaq5qview10value_typeE) |
|                                   | -   [cudaq::range (C++            |
| [\[2\]](api/languages/cpp_api.htm |     func                          |
| l#_CPPv4N5cudaq14kernel_builder4s | tion)](api/languages/cpp_api.html |
| wapERK10QuakeValueRK10QuakeValue) | #_CPPv4I00EN5cudaq5rangeENSt6vect |
| -   [cudaq::KernelExecutionTask   | orI11ElementTypeEE11ElementType), |
|     (C++                          |     [\[1\]](api/languages/cpp_a   |
|     type                          | pi.html#_CPPv4I00EN5cudaq5rangeEN |
| )](api/languages/cpp_api.html#_CP | St6vectorI11ElementTypeEE11Elemen |
| Pv4N5cudaq19KernelExecutionTaskE) | tType11ElementType11ElementType), |
| -   [cudaq::KernelThunkResultType |     [                             |
|     (C++                          | \[2\]](api/languages/cpp_api.html |
|     struct)]                      | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::real (C++             |
| 4N5cudaq21KernelThunkResultTypeE) |     type)](api/languages/         |
| -   [cudaq::KernelThunkType (C++  | cpp_api.html#_CPPv4N5cudaq4realE) |
|                                   | -   [cudaq::registry (C++         |
| type)](api/languages/cpp_api.html |     type)](api/languages/cpp_     |
| #_CPPv4N5cudaq15KernelThunkTypeE) | api.html#_CPPv4N5cudaq8registryE) |
| -   [cudaq::kraus\_channel (C++   | -                                 |
|                                   |  [cudaq::registry::RegisteredType |
|  class)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13kraus_channelE) |     class)](api/                  |
| -   [cudaq::kraus\_channel::empty | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq8registry14RegisteredTypeE) |
|     function)]                    | -   [cudaq::RemoteCapabilities    |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4NK5cudaq13kraus_channel5emptyEv) |     struc                         |
| -   [cudaq::kraus\_c              | t)](api/languages/cpp_api.html#_C |
| hannel::generateUnitaryParameters | PPv4N5cudaq18RemoteCapabilitiesE) |
|     (C++                          | -   [cudaq::Remo                  |
|                                   | teCapabilities::isRemoteSimulator |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4N5cudaq13kraus_chan |     member)](api/languages/c      |
| nel25generateUnitaryParametersEv) | pp_api.html#_CPPv4N5cudaq18Remote |
| -                                 | Capabilities17isRemoteSimulatorE) |
|  [cudaq::kraus\_channel::get\_ops | -   [cudaq::Remot                 |
|     (C++                          | eCapabilities::RemoteCapabilities |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/languages/cpp  |
| K5cudaq13kraus_channel7get_opsEv) | _api.html#_CPPv4N5cudaq18RemoteCa |
| -   [cudaq::kra                   | pabilities18RemoteCapabilitiesEb) |
| us\_channel::is\_unitary\_mixture | -   [cudaq::Remot                 |
|     (C++                          | eCapabilities::serializedCodeExec |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4NK5cudaq13kra |     member)](api/languages/cp     |
| us_channel18is_unitary_mixtureEv) | p_api.html#_CPPv4N5cudaq18RemoteC |
| -   [cuda                         | apabilities18serializedCodeExecE) |
| q::kraus\_channel::kraus\_channel | -   [cudaq:                       |
|     (C++                          | :RemoteCapabilities::stateOverlap |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4IDpEN5cu |     member)](api/langua           |
| daq13kraus_channel13kraus_channel | ges/cpp_api.html#_CPPv4N5cudaq18R |
| EDpRRNSt16initializer_listI1TEE), | emoteCapabilities12stateOverlapE) |
|                                   | -                                 |
|  [\[1\]](api/languages/cpp_api.ht |   [cudaq::RemoteCapabilities::vqe |
| ml#_CPPv4N5cudaq13kraus_channel13 |     (C++                          |
| kraus_channelERK13kraus_channel), |     member)](                     |
|     [\[2\]                        | api/languages/cpp_api.html#_CPPv4 |
| ](api/languages/cpp_api.html#_CPP | N5cudaq18RemoteCapabilities3vqeE) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::RemoteSimulationState |
| hannelERKNSt6vectorI8kraus_opEE), |     (C++                          |
|     [\[3\]                        |     class)]                       |
| ](api/languages/cpp_api.html#_CPP | (api/languages/cpp_api.html#_CPPv |
| v4N5cudaq13kraus_channel13kraus_c | 4N5cudaq21RemoteSimulationStateE) |
| hannelERRNSt6vectorI8kraus_opEE), | -   [cudaq::Resources (C++        |
|     [\[4\]](api/lan               |     class)](api/languages/cpp_a   |
| guages/cpp_api.html#_CPPv4N5cudaq | pi.html#_CPPv4N5cudaq9ResourcesE) |
| 13kraus_channel13kraus_channelEv) | -   [cudaq::run (C++              |
| -   [c                            |     function)]                    |
| udaq::kraus\_channel::noise\_type | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
|     member)](api                  | 5invoke_result_tINSt7decay_tI13Qu |
| /languages/cpp_api.html#_CPPv4N5c | antumKernelEEDpNSt7decay_tI4ARGSE |
| udaq13kraus_channel10noise_typeE) | EEEEENSt6size_tERN5cudaq11noise_m |
| -                                 | odelERR13QuantumKernelDpRR4ARGS), |
| [cudaq::kraus\_channel::operator= |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0DpEN5cu |
|     function)](api/langua         | daq3runENSt6vectorINSt15invoke_re |
| ges/cpp_api.html#_CPPv4N5cudaq13k | sult_tINSt7decay_tI13QuantumKerne |
| raus_channelaSERK13kraus_channel) | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| -   [cu                           | ize_tERR13QuantumKernelDpRR4ARGS) |
| daq::kraus\_channel::operator\[\] | -   [cudaq::run\_async (C++       |
|     (C++                          |     functio                       |
|     function)](api/l              | n)](api/languages/cpp_api.html#_C |
| anguages/cpp_api.html#_CPPv4N5cud | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| aq13kraus_channelixEKNSt6size_tE) | tureINSt6vectorINSt15invoke_resul |
| -   [                             | t_tINSt7decay_tI13QuantumKernelEE |
| cudaq::kraus\_channel::parameters | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|     (C++                          | ze_tENSt6size_tERN5cudaq11noise_m |
|     member)](api                  | odelERR13QuantumKernelDpRR4ARGS), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[1\]](api/la                |
| udaq13kraus_channel10parametersE) | nguages/cpp_api.html#_CPPv4I0DpEN |
| -   [cud                          | 5cudaq9run_asyncENSt6futureINSt6v |
| aq::kraus\_channel::probabilities | ectorINSt15invoke_result_tINSt7de |
|     (C++                          | cay_tI13QuantumKernelEEDpNSt7deca |
|     member)](api/la               | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| nguages/cpp_api.html#_CPPv4N5cuda | ize_tERR13QuantumKernelDpRR4ARGS) |
| q13kraus_channel13probabilitiesE) | -   [cudaq::sample (C++           |
| -   [                             |     function)](api/languages/cp   |
| cudaq::kraus\_channel::push\_back | p_api.html#_CPPv4I0Dp0EN5cudaq6sa |
|     (C++                          | mpleE13sample_resultRK14sample_op |
|     function)](api/langua         | tionsRR13QuantumKernelDpRR4Args), |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     [\[1\]                        |
| raus_channel9push_backE8kraus_op) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus\_channel::size  | v4I0Dp0EN5cudaq6sampleE13sample_r |
|     (C++                          | esultRR13QuantumKernelDpRR4Args), |
|     function)                     |     [\[                           |
| ](api/languages/cpp_api.html#_CPP | 2\]](api/languages/cpp_api.html#_ |
| v4NK5cudaq13kraus_channel4sizeEv) | CPPv4I0Dp0EN5cudaq6sampleEDaNSt6s |
| -   [cu                           | ize_tERR13QuantumKernelDpRR4Args) |
| daq::kraus\_channel::unitary\_ops | -   [cudaq::sample\_options (C++  |
|     (C++                          |     s                             |
|     member)](api/                 | truct)](api/languages/cpp_api.htm |
| languages/cpp_api.html#_CPPv4N5cu | l#_CPPv4N5cudaq14sample_optionsE) |
| daq13kraus_channel11unitary_opsE) | -   [cudaq::sample\_result (C++   |
| -   [cudaq::kraus\_op (C++        |                                   |
|     struct)](api/languages/cpp_   |  class)](api/languages/cpp_api.ht |
| api.html#_CPPv4N5cudaq8kraus_opE) | ml#_CPPv4N5cudaq13sample_resultE) |
| -   [cudaq::kraus\_op::adjoint    | -                                 |
|     (C++                          |    [cudaq::sample\_result::append |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)](api/languages/cpp  |
| CPPv4NK5cudaq8kraus_op7adjointEv) | _api.html#_CPPv4N5cudaq13sample_r |
| -   [cudaq::kraus\_op::data (C++  | esult6appendER15ExecutionResultb) |
|                                   | -   [cudaq::sample\_result::begin |
|  member)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     function)]                    |
| -   [cudaq::kraus\_op::kraus\_op  | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4N5cudaq13sample_result5beginEv), |
|     func                          |     [\[1\]]                       |
| tion)](api/languages/cpp_api.html | (api/languages/cpp_api.html#_CPPv |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | 4NK5cudaq13sample_result5beginEv) |
| opERRNSt16initializer_listI1TEE), | -                                 |
|                                   |    [cudaq::sample\_result::cbegin |
|  [\[1\]](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o |     function)](                   |
| pENSt6vectorIN5cudaq7complexEEE), | api/languages/cpp_api.html#_CPPv4 |
|     [\[2\]](api/l                 | NK5cudaq13sample_result6cbeginEv) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cudaq::sample\_result::cend  |
| aq8kraus_op8kraus_opERK8kraus_op) |     (C++                          |
| -   [cudaq::kraus\_op::nCols (C++ |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
| member)](api/languages/cpp_api.ht | v4NK5cudaq13sample_result4cendEv) |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | -   [cudaq::sample\_result::clear |
| -   [cudaq::kraus\_op::nRows (C++ |     (C++                          |
|                                   |     function)                     |
| member)](api/languages/cpp_api.ht | ](api/languages/cpp_api.html#_CPP |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | v4N5cudaq13sample_result5clearEv) |
| -   [cudaq::kraus\_op::operator=  | -   [cudaq::sample\_result::count |
|     (C++                          |     (C++                          |
|     function)                     |     function)](                   |
| ](api/languages/cpp_api.html#_CPP | api/languages/cpp_api.html#_CPPv4 |
| v4N5cudaq8kraus_opaSERK8kraus_op) | NK5cudaq13sample_result5countENSt |
| -   [cudaq::kraus\_op::precision  | 11string_viewEKNSt11string_viewE) |
|     (C++                          | -   [c                            |
|     memb                          | udaq::sample\_result::deserialize |
| er)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8kraus_op9precisionE) |     functio                       |
| -   [cudaq::matrix\_callback (C++ | n)](api/languages/cpp_api.html#_C |
|     c                             | PPv4N5cudaq13sample_result11deser |
| lass)](api/languages/cpp_api.html | ializeERNSt6vectorINSt6size_tEEE) |
| #_CPPv4N5cudaq15matrix_callbackE) | -   [cudaq::sample\_result::dump  |
| -   [cudaq::matrix\_handler (C++  |     (C++                          |
|                                   |     function)](api/languag        |
| class)](api/languages/cpp_api.htm | es/cpp_api.html#_CPPv4NK5cudaq13s |
| l#_CPPv4N5cudaq14matrix_handlerE) | ample_result4dumpERNSt7ostreamE), |
| -   [cudaq::matri                 |     [\[1\]                        |
| x\_handler::commutation\_behavior | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq13sample_result4dumpEv) |
|     struct)](api/languages/       | -   [cudaq::sample\_result::end   |
| cpp_api.html#_CPPv4N5cudaq14matri |     (C++                          |
| x_handler20commutation_behaviorE) |     function                      |
| -                                 | )](api/languages/cpp_api.html#_CP |
|   [cudaq::matrix\_handler::define | Pv4N5cudaq13sample_result3endEv), |
|     (C++                          |     [\[1\                         |
|     function)](a                  | ]](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4NK5cudaq13sample_result3endEv) |
| 5cudaq14matrix_handler6defineENSt | -   [c                            |
| 6stringENSt6vectorINSt7int64_tEEE | udaq::sample\_result::expectation |
| RR15matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
| [\[1\]](api/languages/cpp_api.htm | tml#_CPPv4NK5cudaq13sample_result |
| l#_CPPv4N5cudaq14matrix_handler6d | 11expectationEKNSt11string_viewE) |
| efineENSt6stringENSt6vectorINSt7i | -   [cud                          |
| nt64_tEEERR15matrix_callbackRR20d | aq::sample\_result::get\_marginal |
| iag_matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     function)](api/languages/cpp_ |
|     [\[2\]](                      | api.html#_CPPv4NK5cudaq13sample_r |
| api/languages/cpp_api.html#_CPPv4 | esult12get_marginalERKNSt6vectorI |
| N5cudaq14matrix_handler6defineENS | NSt6size_tEEEKNSt11string_viewE), |
| t6stringENSt6vectorINSt7int64_tEE |     [\[1\]](api/languages/cpp_    |
| ERR15matrix_callbackRRNSt13unorde | api.html#_CPPv4NK5cudaq13sample_r |
| red_mapINSt6stringENSt6stringEEE) | esult12get_marginalERRKNSt6vector |
| -                                 | INSt6size_tEEEKNSt11string_viewE) |
|  [cudaq::matrix\_handler::degrees | -   [cudaq::                      |
|     (C++                          | sample\_result::get\_total\_shots |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4NK |     function)](api/langua         |
| 5cudaq14matrix_handler7degreesEv) | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| -                                 | sample_result15get_total_shotsEv) |
| [cudaq::matrix\_handler::displace | -   [cudaq::                      |
|     (C++                          | sample\_result::has\_even\_parity |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     fun                           |
| rix_handler8displaceENSt6size_tE) | ction)](api/languages/cpp_api.htm |
| -   [cudaq::matrix\_h             | l#_CPPv4N5cudaq13sample_result15h |
| andler::get\_expected\_dimensions | as_even_parityENSt11string_viewE) |
|     (C++                          | -   [cudaq:                       |
|                                   | :sample\_result::has\_expectation |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     funct                         |
| ndler23get_expected_dimensionsEv) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::matrix\_hand          | _CPPv4NK5cudaq13sample_result15ha |
| ler::get\_parameter\_descriptions | s_expectationEKNSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|                                   | q::sample\_result::most\_probable |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4NK5cudaq14matrix_handl |     fun                           |
| er26get_parameter_descriptionsEv) | ction)](api/languages/cpp_api.htm |
| -   [cu                           | l#_CPPv4NK5cudaq13sample_result13 |
| daq::matrix\_handler::instantiate | most_probableEKNSt11string_viewE) |
|     (C++                          | -   [                             |
|     function)](a                  | cudaq::sample\_result::operator+= |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14matrix_handler11instantia |     function)](api/langua         |
| teENSt6stringERKNSt6vectorINSt6si | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ze_tEEERK20commutation_behavior), | ample_resultpLERK13sample_result) |
|     [\[1\]](                      | -                                 |
| api/languages/cpp_api.html#_CPPv4 | [cudaq::sample\_result::operator= |
| N5cudaq14matrix_handler11instanti |     (C++                          |
| ateENSt6stringERRNSt6vectorINSt6s |     function)](api/langua         |
| ize_tEEERK20commutation_behavior) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -   [cudaq:                       | ample_resultaSER13sample_result), |
| :matrix\_handler::matrix\_handler |     [\[1\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function)](api/languag        | ample_resultaSERR13sample_result) |
| es/cpp_api.html#_CPPv4I0_NSt11ena | -   [                             |
| ble_if_tINSt12is_base_of_vI16oper | cudaq::sample\_result::operator== |
| ator_handler1TEEbEEEN5cudaq14matr |     (C++                          |
| ix_handler14matrix_handlerERK1T), |     function)](api/languag        |
|     [\[1\]](ap                    | es/cpp_api.html#_CPPv4NK5cudaq13s |
| i/languages/cpp_api.html#_CPPv4I0 | ample_resulteqERK13sample_result) |
| _NSt11enable_if_tINSt12is_base_of | -   [c                            |
| _vI16operator_handler1TEEbEEEN5cu | udaq::sample\_result::probability |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erERK1TRK20commutation_behavior), |     function)](api/lan            |
|     [\[2\]](api/languages/cpp_ap  | guages/cpp_api.html#_CPPv4NK5cuda |
| i.html#_CPPv4N5cudaq14matrix_hand | q13sample_result11probabilityENSt |
| ler14matrix_handlerENSt6size_tE), | 11string_viewEKNSt11string_viewE) |
|     [\[3\]](api/                  | -   [cudaq                        |
| languages/cpp_api.html#_CPPv4N5cu | ::sample\_result::register\_names |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erENSt6stringERKNSt6vectorINSt6si |     function)](api/langu          |
| ze_tEEERK20commutation_behavior), | ages/cpp_api.html#_CPPv4NK5cudaq1 |
|     [\[4\]](api/                  | 3sample_result14register_namesEv) |
| languages/cpp_api.html#_CPPv4N5cu | -                                 |
| daq14matrix_handler14matrix_handl |   [cudaq::sample\_result::reorder |
| erENSt6stringERRNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     function)](api/langua         |
|     [\                            | ges/cpp_api.html#_CPPv4N5cudaq13s |
| [5\]](api/languages/cpp_api.html# | ample_result7reorderERKNSt6vector |
| _CPPv4N5cudaq14matrix_handler14ma | INSt6size_tEEEKNSt11string_viewE) |
| trix_handlerERK14matrix_handler), | -   [cuda                         |
|     [                             | q::sample\_result::sample\_result |
| \[6\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq14matrix_handler14m |     fun                           |
| atrix_handlerERR14matrix_handler) | ction)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq13sample_result13s |
| [cudaq::matrix\_handler::momentum | ample_resultER15ExecutionResult), |
|     (C++                          |                                   |
|     function)](api/language       |  [\[1\]](api/languages/cpp_api.ht |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ml#_CPPv4N5cudaq13sample_result13 |
| rix_handler8momentumENSt6size_tE) | sample_resultERK13sample_result), |
| -                                 |     [\[2\]](api/l                 |
|   [cudaq::matrix\_handler::number | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq13sample_result13sample_resultE |
|     function)](api/langua         | RNSt6vectorI15ExecutionResultEE), |
| ges/cpp_api.html#_CPPv4N5cudaq14m |                                   |
| atrix_handler6numberENSt6size_tE) |  [\[3\]](api/languages/cpp_api.ht |
| -   [                             | ml#_CPPv4N5cudaq13sample_result13 |
| cudaq::matrix\_handler::operator= | sample_resultERR13sample_result), |
|     (C++                          |     [                             |
|     fun                           | \[4\]](api/languages/cpp_api.html |
| ction)](api/languages/cpp_api.htm | #_CPPv4N5cudaq13sample_result13sa |
| l#_CPPv4I0_NSt11enable_if_tIXaant | mple_resultERR15ExecutionResult), |
| NSt7is_sameI1T14matrix_handlerE5v |     [\[5\]](api/la                |
| alueENSt12is_base_of_vI16operator | nguages/cpp_api.html#_CPPv4N5cuda |
| _handler1TEEEbEEEN5cudaq14matrix_ | q13sample_result13sample_resultEd |
| handleraSER14matrix_handlerRK1T), | RNSt6vectorI15ExecutionResultEE), |
|     [\[1\]](api/languages         |     [\[6\]](api/lan               |
| /cpp_api.html#_CPPv4N5cudaq14matr | guages/cpp_api.html#_CPPv4N5cudaq |
| ix_handleraSERK14matrix_handler), | 13sample_result13sample_resultEv) |
|     [\[2\]](api/language          | -                                 |
| s/cpp_api.html#_CPPv4N5cudaq14mat | [cudaq::sample\_result::serialize |
| rix_handleraSERR14matrix_handler) |     (C++                          |
| -   [c                            |     function)](api                |
| udaq::matrix\_handler::operator== | /languages/cpp_api.html#_CPPv4NK5 |
|     (C++                          | cudaq13sample_result9serializeEv) |
|     function)](api/languages      | -   [cudaq::sample\_result::size  |
| /cpp_api.html#_CPPv4NK5cudaq14mat |     (C++                          |
| rix_handlereqERK14matrix_handler) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq13sampl |
|   [cudaq::matrix\_handler::parity | e_result4sizeEKNSt11string_viewE) |
|     (C++                          | -                                 |
|     function)](api/langua         |   [cudaq::sample\_result::to\_map |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6parityENSt6size_tE) |     function)](api/languages/cpp  |
| -                                 | _api.html#_CPPv4NK5cudaq13sample_ |
| [cudaq::matrix\_handler::position | result6to_mapEKNSt11string_viewE) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/language       | :sample\_result::\~sample\_result |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8positionENSt6size_tE) |     funct                         |
| -   [cudaq::ma                    | ion)](api/languages/cpp_api.html# |
| trix\_handler::remove\_definition | _CPPv4N5cudaq13sample_resultD0Ev) |
|     (C++                          | -   [cudaq::scalar\_callback (C++ |
|     fu                            |     c                             |
| nction)](api/languages/cpp_api.ht | lass)](api/languages/cpp_api.html |
| ml#_CPPv4N5cudaq14matrix_handler1 | #_CPPv4N5cudaq15scalar_callbackE) |
| 7remove_definitionERKNSt6stringE) | -   [cu                           |
| -                                 | daq::scalar\_callback::operator() |
|  [cudaq::matrix\_handler::squeeze |     (C++                          |
|     (C++                          |     function)](api/language       |
|     function)](api/languag        | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| es/cpp_api.html#_CPPv4N5cudaq14ma | alar_callbackclERKNSt13unordered_ |
| trix_handler7squeezeENSt6size_tE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::matr                  | -   [c                            |
| ix\_handler::to\_diagonal\_matrix | udaq::scalar\_callback::operator= |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](api/languages/c    |
| uages/cpp_api.html#_CPPv4NK5cudaq | pp_api.html#_CPPv4N5cudaq15scalar |
| 14matrix_handler18to_diagonal_mat | _callbackaSERK15scalar_callback), |
| rixERNSt13unordered_mapINSt6size_ |     [\[1\]](api/languages/        |
| tENSt7int64_tEEERKNSt13unordered_ | cpp_api.html#_CPPv4N5cudaq15scala |
| mapINSt6stringENSt7complexIdEEEE) | r_callbackaSERR15scalar_callback) |
| -   [c                            | -   [cudaq::s                     |
| udaq::matrix\_handler::to\_matrix | calar\_callback::scalar\_callback |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api/languag        |
| ](api/languages/cpp_api.html#_CPP | es/cpp_api.html#_CPPv4I0_NSt11ena |
| v4NK5cudaq14matrix_handler9to_mat | ble_if_tINSt16is_invocable_r_vINS |
| rixERNSt13unordered_mapINSt6size_ | t7complexIdEE8CallableRKNSt13unor |
| tENSt7int64_tEEERKNSt13unordered_ | dered_mapINSt6stringENSt7complexI |
| mapINSt6stringENSt7complexIdEEEE) | dEEEEEEbEEEN5cudaq15scalar_callba |
| -   [c                            | ck15scalar_callbackERR8Callable), |
| udaq::matrix\_handler::to\_string |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/               | Pv4N5cudaq15scalar_callback15scal |
| languages/cpp_api.html#_CPPv4NK5c | ar_callbackERK15scalar_callback), |
| udaq14matrix_handler9to_stringEb) |     [\[2                          |
| -   [c                            | \]](api/languages/cpp_api.html#_C |
| udaq::matrix\_handler::unique\_id | PPv4N5cudaq15scalar_callback15sca |
|     (C++                          | lar_callbackERR15scalar_callback) |
|     function)](api/               | -   [cudaq::scalar\_operator (C++ |
| languages/cpp_api.html#_CPPv4NK5c |     c                             |
| udaq14matrix_handler9unique_idEv) | lass)](api/languages/cpp_api.html |
| -   [cudaq::m                     | #_CPPv4N5cudaq15scalar_operatorE) |
| atrix\_handler::\~matrix\_handler | -   [                             |
|     (C++                          | cudaq::scalar\_operator::evaluate |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |                                   |
| CPPv4N5cudaq14matrix_handlerD0Ev) |    function)](api/languages/cpp_a |
| -   [cudaq::matrix\_op (C++       | pi.html#_CPPv4NK5cudaq15scalar_op |
|     type)](api/languages/cpp_a    | erator8evaluateERKNSt13unordered_ |
| pi.html#_CPPv4N5cudaq9matrix_opE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::matrix\_op\_term (C++ | -   [cudaq::scalar\_opera         |
|                                   | tor::get\_parameter\_descriptions |
|  type)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14matrix_op_termE) |     f                             |
| -                                 | unction)](api/languages/cpp_api.h |
|  [cudaq::mdiag\_operator\_handler | tml#_CPPv4NK5cudaq15scalar_operat |
|     (C++                          | or26get_parameter_descriptionsEv) |
|     class)](                      | -   [cuda                         |
| api/languages/cpp_api.html#_CPPv4 | q::scalar\_operator::is\_constant |
| N5cudaq22mdiag_operator_handlerE) |     (C++                          |
| -   [cudaq::mpi (C++              |     function)](api/lang           |
|     type)](api/languages          | uages/cpp_api.html#_CPPv4NK5cudaq |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | 15scalar_operator11is_constantEv) |
| -   [cudaq::mpi::all\_gather (C++ | -   [cu                           |
|     fu                            | daq::scalar\_operator::operator\* |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     function                      |
| RNSt6vectorIdEERKNSt6vectorIdEE), | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_operatormlENSt |
|   [\[1\]](api/languages/cpp_api.h | 7complexIdEERK15scalar_operator), |
| tml#_CPPv4N5cudaq3mpi10all_gather |     [\[1\                         |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::mpi::all\_reduce (C++ | Pv4N5cudaq15scalar_operatormlENSt |
|                                   | 7complexIdEERR15scalar_operator), |
|  function)](api/languages/cpp_api |     [\[2\]](api/languages/cp      |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | p_api.html#_CPPv4N5cudaq15scalar_ |
| reduceE1TRK1TRK14BinaryFunction), | operatormlEdRK15scalar_operator), |
|     [\[1\]](api/langu             |     [\[3\]](api/languages/cp      |
| ages/cpp_api.html#_CPPv4I00EN5cud | p_api.html#_CPPv4N5cudaq15scalar_ |
| aq3mpi10all_reduceE1TRK1TRK4Func) | operatormlEdRR15scalar_operator), |
| -   [cudaq::mpi::broadcast (C++   |     [\[4\]](api/languages         |
|     function)](api/               | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| languages/cpp_api.html#_CPPv4N5cu | alar_operatormlENSt7complexIdEE), |
| daq3mpi9broadcastERNSt6stringEi), |     [\[5\]](api/languages/cpp     |
|     [\[1\]](api/la                | _api.html#_CPPv4NKR5cudaq15scalar |
| nguages/cpp_api.html#_CPPv4N5cuda | _operatormlERK15scalar_operator), |
| q3mpi9broadcastERNSt6vectorIdEEi) |     [\[6\]]                       |
| -   [cudaq::mpi::finalize (C++    | (api/languages/cpp_api.html#_CPPv |
|     f                             | 4NKR5cudaq15scalar_operatormlEd), |
| unction)](api/languages/cpp_api.h |     [\[7\]](api/language          |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::mpi::initialize (C++  | alar_operatormlENSt7complexIdEE), |
|     function                      |     [\[8\]](api/languages/cp      |
| )](api/languages/cpp_api.html#_CP | p_api.html#_CPPv4NO5cudaq15scalar |
| Pv4N5cudaq3mpi10initializeEiPPc), | _operatormlERK15scalar_operator), |
|     [                             |     [\[9\                         |
| \[1\]](api/languages/cpp_api.html | ]](api/languages/cpp_api.html#_CP |
| #_CPPv4N5cudaq3mpi10initializeEv) | Pv4NO5cudaq15scalar_operatormlEd) |
| -   [cudaq::mpi::is\_initialized  | -   [cud                          |
|     (C++                          | aq::scalar\_operator::operator\*= |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/languag        |
| Pv4N5cudaq3mpi14is_initializedEv) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::mpi::num\_ranks (C++  | alar_operatormLENSt7complexIdEE), |
|     fu                            |     [\[1\]](api/languages/c       |
| nction)](api/languages/cpp_api.ht | pp_api.html#_CPPv4N5cudaq15scalar |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | _operatormLERK15scalar_operator), |
| -   [cudaq::mpi::rank (C++        |     [\[2                          |
|                                   | \]](api/languages/cpp_api.html#_C |
|    function)](api/languages/cpp_a | PPv4N5cudaq15scalar_operatormLEd) |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | -   [c                            |
| -   [cudaq::noise\_model (C++     | udaq::scalar\_operator::operator+ |
|                                   |     (C++                          |
|    class)](api/languages/cpp_api. |     function                      |
| html#_CPPv4N5cudaq11noise_modelE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise                 | Pv4N5cudaq15scalar_operatorplENSt |
| \_model::add\_all\_qubit\_channel | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     function)](api                | ]](api/languages/cpp_api.html#_CP |
| /languages/cpp_api.html#_CPPv4IDp | Pv4N5cudaq15scalar_operatorplENSt |
| EN5cudaq11noise_model21add_all_qu | 7complexIdEERR15scalar_operator), |
| bit_channelEvRK13kraus_channeli), |     [\[2\]](api/languages/cp      |
|     [\[1\]](api/langua            | p_api.html#_CPPv4N5cudaq15scalar_ |
| ges/cpp_api.html#_CPPv4N5cudaq11n | operatorplEdRK15scalar_operator), |
| oise_model21add_all_qubit_channel |     [\[3\]](api/languages/cp      |
| ERKNSt6stringERK13kraus_channeli) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [                             | operatorplEdRR15scalar_operator), |
| cudaq::noise\_model::add\_channel |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     funct                         | alar_operatorplENSt7complexIdEE), |
| ion)](api/languages/cpp_api.html# |     [\[5\]](api/languages/cpp     |
| _CPPv4IDpEN5cudaq11noise_model11a | _api.html#_CPPv4NKR5cudaq15scalar |
| dd_channelEvRK15PredicateFuncTy), | _operatorplERK15scalar_operator), |
|     [\[1\]](api/languages/cpp_    |     [\[6\]]                       |
| api.html#_CPPv4IDpEN5cudaq11noise | (api/languages/cpp_api.html#_CPPv |
| _model11add_channelEvRKNSt6vector | 4NKR5cudaq15scalar_operatorplEd), |
| INSt6size_tEEERK13kraus_channel), |     [\[7\]]                       |
|     [\[2\]](ap                    | (api/languages/cpp_api.html#_CPPv |
| i/languages/cpp_api.html#_CPPv4N5 | 4NKR5cudaq15scalar_operatorplEv), |
| cudaq11noise_model11add_channelER |     [\[8\]](api/language          |
| KNSt6stringERK15PredicateFuncTy), | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|                                   | alar_operatorplENSt7complexIdEE), |
| [\[3\]](api/languages/cpp_api.htm |     [\[9\]](api/languages/cp      |
| l#_CPPv4N5cudaq11noise_model11add | p_api.html#_CPPv4NO5cudaq15scalar |
| _channelERKNSt6stringERKNSt6vecto | _operatorplERK15scalar_operator), |
| rINSt6size_tEEERK13kraus_channel) |     [\[10\]                       |
| -   [cudaq::noise\_model::empty   | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatorplEd), |
|     function                      |     [\[11\                        |
| )](api/languages/cpp_api.html#_CP | ]](api/languages/cpp_api.html#_CP |
| Pv4NK5cudaq11noise_model5emptyEv) | Pv4NO5cudaq15scalar_operatorplEv) |
| -   [c                            | -   [cu                           |
| udaq::noise\_model::get\_channels | daq::scalar\_operator::operator+= |
|     (C++                          |     (C++                          |
|     function)](api/l              |     function)](api/languag        |
| anguages/cpp_api.html#_CPPv4I0ENK | es/cpp_api.html#_CPPv4N5cudaq15sc |
| 5cudaq11noise_model12get_channels | alar_operatorpLENSt7complexIdEE), |
| ENSt6vectorI13kraus_channelEERKNS |     [\[1\]](api/languages/c       |
| t6vectorINSt6size_tEEERKNSt6vecto | pp_api.html#_CPPv4N5cudaq15scalar |
| rINSt6size_tEEERKNSt6vectorIdEE), | _operatorpLERK15scalar_operator), |
|     [\[1\]](api/languages/cpp_a   |     [\[2                          |
| pi.html#_CPPv4NK5cudaq11noise_mod | \]](api/languages/cpp_api.html#_C |
| el12get_channelsERKNSt6stringERKN | PPv4N5cudaq15scalar_operatorpLEd) |
| St6vectorINSt6size_tEEERKNSt6vect | -   [c                            |
| orINSt6size_tEEERKNSt6vectorIdEE) | udaq::scalar\_operator::operator- |
| -   [                             |     (C++                          |
| cudaq::noise\_model::noise\_model |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function)](api                | Pv4N5cudaq15scalar_operatormiENSt |
| /languages/cpp_api.html#_CPPv4N5c | 7complexIdEERK15scalar_operator), |
| udaq11noise_model11noise_modelEv) |     [\[1\                         |
| -   [cud                          | ]](api/languages/cpp_api.html#_CP |
| aq::noise\_model::PredicateFuncTy | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     type)](api/la                 |     [\[2\]](api/languages/cp      |
| nguages/cpp_api.html#_CPPv4N5cuda | p_api.html#_CPPv4N5cudaq15scalar_ |
| q11noise_model15PredicateFuncTyE) | operatormiEdRK15scalar_operator), |
| -   [cudaq                        |     [\[3\]](api/languages/cp      |
| ::noise\_model::register\_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRR15scalar_operator), |
|     function)](api/languages      |     [\[4\]](api/languages         |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| noise_model16register_channelEvv) | alar_operatormiENSt7complexIdEE), |
| -   [cudaq::no                    |     [\[5\]](api/languages/cpp     |
| ise\_model::requires\_constructor | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatormiERK15scalar_operator), |
|     type)](api/languages/cp       |     [\[6\]]                       |
| p_api.html#_CPPv4I0DpEN5cudaq11no | (api/languages/cpp_api.html#_CPPv |
| ise_model20requires_constructorE) | 4NKR5cudaq15scalar_operatormiEd), |
| -   [cudaq::noise\_model\_type    |     [\[7\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     e                             | 4NKR5cudaq15scalar_operatormiEv), |
| num)](api/languages/cpp_api.html# |     [\[8\]](api/language          |
| _CPPv4N5cudaq16noise_model_typeE) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::noise                 | alar_operatormiENSt7complexIdEE), |
| \_model\_type::amplitude\_damping |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     enumerator)](api/languages    | _operatormiERK15scalar_operator), |
| /cpp_api.html#_CPPv4N5cudaq16nois |     [\[10\]                       |
| e_model_type17amplitude_dampingE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::noise\_model\_        | v4NO5cudaq15scalar_operatormiEd), |
| type::amplitude\_damping\_channel |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     e                             | Pv4NO5cudaq15scalar_operatormiEv) |
| numerator)](api/languages/cpp_api | -   [cu                           |
| .html#_CPPv4N5cudaq16noise_model_ | daq::scalar\_operator::operator-= |
| type25amplitude_damping_channelE) |     (C++                          |
| -   [cudaq::noise                 |     function)](api/languag        |
| \_model\_type::bit\_flip\_channel | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatormIENSt7complexIdEE), |
|     enumerator)](api/language     |     [\[1\]](api/languages/c       |
| s/cpp_api.html#_CPPv4N5cudaq16noi | pp_api.html#_CPPv4N5cudaq15scalar |
| se_model_type16bit_flip_channelE) | _operatormIERK15scalar_operator), |
| -   [cudaq::no                    |     [\[2                          |
| ise\_model\_type::depolarization1 | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatormIEd) |
|     enumerator)](api/languag      | -   [c                            |
| es/cpp_api.html#_CPPv4N5cudaq16no | udaq::scalar\_operator::operator/ |
| ise_model_type15depolarization1E) |     (C++                          |
| -   [cudaq::no                    |     function                      |
| ise\_model\_type::depolarization2 | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|     enumerator)](api/languag      | 7complexIdEERK15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[1\                         |
| ise_model_type15depolarization2E) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise\_mod            | Pv4N5cudaq15scalar_operatordvENSt |
| el\_type::depolarization\_channel | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq15scalar_ |
|   enumerator)](api/languages/cpp_ | operatordvEdRK15scalar_operator), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[3\]](api/languages/cp      |
| el_type22depolarization_channelE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [                             | operatordvEdRR15scalar_operator), |
| cudaq::noise\_model\_type::pauli1 |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     enumerator)](a                | alar_operatordvENSt7complexIdEE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[5\]](api/languages/cpp     |
| 5cudaq16noise_model_type6pauli1E) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [                             | _operatordvERK15scalar_operator), |
| cudaq::noise\_model\_type::pauli2 |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     enumerator)](a                | 4NKR5cudaq15scalar_operatordvEd), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[7\]](api/language          |
| 5cudaq16noise_model_type6pauli2E) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::n                     | alar_operatordvENSt7complexIdEE), |
| oise\_model\_type::phase\_damping |     [\[8\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     enumerator)](api/langu        | _operatordvERK15scalar_operator), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     [\[9\                         |
| noise_model_type13phase_dampingE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise\_               | Pv4NO5cudaq15scalar_operatordvEd) |
| model\_type::phase\_flip\_channel | -   [cu                           |
|     (C++                          | daq::scalar\_operator::operator/= |
|     enumerator)](api/languages/   |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16noise |     function)](api/languag        |
| _model_type18phase_flip_channelE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [c                            | alar_operatordVENSt7complexIdEE), |
| udaq::noise\_model\_type::unknown |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](ap               | _operatordVERK15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[2                          |
| cudaq16noise_model_type7unknownE) | \]](api/languages/cpp_api.html#_C |
| -   [cu                           | PPv4N5cudaq15scalar_operatordVEd) |
| daq::noise\_model\_type::x\_error | -   [c                            |
|     (C++                          | udaq::scalar\_operator::operator= |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languages/c    |
| cudaq16noise_model_type7x_errorE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cu                           | _operatoraSERK15scalar_operator), |
| daq::noise\_model\_type::y\_error |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|     enumerator)](ap               | r_operatoraSERR15scalar_operator) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cu                           |
| cudaq16noise_model_type7y_errorE) | daq::scalar\_operator::operator== |
| -   [cu                           |     (C++                          |
| daq::noise\_model\_type::z\_error |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq15scala |
|     enumerator)](ap               | r_operatoreqERK15scalar_operator) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [cudaq::s                     |
| cudaq16noise_model_type7z_errorE) | calar\_operator::scalar\_operator |
| -   [cudaq::num\_available\_gpus  |     (C++                          |
|     (C++                          |     func                          |
|     function                      | tion)](api/languages/cpp_api.html |
| )](api/languages/cpp_api.html#_CP | #_CPPv4N5cudaq15scalar_operator15 |
| Pv4N5cudaq18num_available_gpusEv) | scalar_operatorENSt7complexIdEE), |
| -   [cudaq::observe (C++          |     [\[1\]](api/langu             |
|     function)](                   | ages/cpp_api.html#_CPPv4N5cudaq15 |
| api/languages/cpp_api.html#_CPPv4 | scalar_operator15scalar_operatorE |
| I00Dp0EN5cudaq7observeENSt6vector | RK15scalar_callbackRRNSt13unorder |
| I14observe_resultEERR13QuantumKer | ed_mapINSt6stringENSt6stringEEE), |
| nelRK15SpinOpContainerDpRR4Args), |     [\[2\                         |
|     [\[1\]](api/languages/cpp_api | ]](api/languages/cpp_api.html#_CP |
| .html#_CPPv4I0Dp0EN5cudaq7observe | Pv4N5cudaq15scalar_operator15scal |
| E14observe_resultNSt6size_tERR13Q | ar_operatorERK15scalar_operator), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[3\]](api/langu             |
|     [\[2                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
| \]](api/languages/cpp_api.html#_C | scalar_operator15scalar_operatorE |
| PPv4I0Dp0EN5cudaq7observeE14obser | RR15scalar_callbackRRNSt13unorder |
| ve_resultRK15observe_optionsRR13Q | ed_mapINSt6stringENSt6stringEEE), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[4\                         |
|     [\[3\]](api/langu             | ]](api/languages/cpp_api.html#_CP |
| ages/cpp_api.html#_CPPv4I0Dp0EN5c | Pv4N5cudaq15scalar_operator15scal |
| udaq7observeE14observe_resultRR13 | ar_operatorERR15scalar_operator), |
| QuantumKernelRK7spin_opDpRR4Args) |     [\[5\]](api/language          |
| -   [cudaq::observe\_options (C++ | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     st                            | lar_operator15scalar_operatorEd), |
| ruct)](api/languages/cpp_api.html |     [\[6\]](api/languag           |
| #_CPPv4N5cudaq15observe_optionsE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::observe\_result (C++  | alar_operator15scalar_operatorEv) |
|                                   | -   [cu                           |
| class)](api/languages/cpp_api.htm | daq::scalar\_operator::to\_matrix |
| l#_CPPv4N5cudaq14observe_resultE) |     (C++                          |
| -                                 |                                   |
|   [cudaq::observe\_result::counts |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4NK5cudaq15scalar_ope |
|     function)](api/languages/c    | rator9to_matrixERKNSt13unordered_ |
| pp_api.html#_CPPv4N5cudaq14observ | mapINSt6stringENSt7complexIdEEEE) |
| e_result6countsERK12spin_op_term) | -   [cu                           |
| -   [cudaq::observe\_result::dump | daq::scalar\_operator::to\_string |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api/l              |
| ](api/languages/cpp_api.html#_CPP | anguages/cpp_api.html#_CPPv4NK5cu |
| v4N5cudaq14observe_result4dumpEv) | daq15scalar_operator9to_stringEv) |
| -   [cu                           | -   [cudaq::sca                   |
| daq::observe\_result::expectation | lar\_operator::\~scalar\_operator |
|     (C++                          |     (C++                          |
|                                   |     functio                       |
| function)](api/languages/cpp_api. | n)](api/languages/cpp_api.html#_C |
| html#_CPPv4N5cudaq14observe_resul | PPv4N5cudaq15scalar_operatorD0Ev) |
| t11expectationERK12spin_op_term), | -   [cuda                         |
|     [\[1\]](api/la                | q::SerializedCodeExecutionContext |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14observe_result11expectationEv) |     class)](api/lang              |
| -   [cudaq:                       | uages/cpp_api.html#_CPPv4N5cudaq3 |
| :observe\_result::id\_coefficient | 0SerializedCodeExecutionContextE) |
|     (C++                          | -   [cudaq::set\_noise (C++       |
|     function)](api/langu          |     function)](api/langu          |
| ages/cpp_api.html#_CPPv4N5cudaq14 | ages/cpp_api.html#_CPPv4N5cudaq9s |
| observe_result14id_coefficientEv) | et_noiseERKN5cudaq11noise_modelE) |
| -   [cudaq:                       | -   [cudaq::set\_random\_seed     |
| :observe\_result::observe\_result |     (C++                          |
|     (C++                          |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|   function)](api/languages/cpp_ap | daq15set_random_seedENSt6size_tE) |
| i.html#_CPPv4N5cudaq14observe_res | -   [cudaq::simulation\_precision |
| ult14observe_resultEdRK7spin_op), |     (C++                          |
|     [\[1\]](a                     |     enum)                         |
| pi/languages/cpp_api.html#_CPPv4N | ](api/languages/cpp_api.html#_CPP |
| 5cudaq14observe_result14observe_r | v4N5cudaq20simulation_precisionE) |
| esultEdRK7spin_op13sample_result) | -   [c                            |
| -                                 | udaq::simulation\_precision::fp32 |
| [cudaq::observe\_result::operator |     (C++                          |
|     double (C++                   |     enumerator)](api              |
|     functio                       | /languages/cpp_api.html#_CPPv4N5c |
| n)](api/languages/cpp_api.html#_C | udaq20simulation_precision4fp32E) |
| PPv4N5cudaq14observe_resultcvdEv) | -   [c                            |
| -   [                             | udaq::simulation\_precision::fp64 |
| cudaq::observe\_result::raw\_data |     (C++                          |
|     (C++                          |     enumerator)](api              |
|     function)](ap                 | /languages/cpp_api.html#_CPPv4N5c |
| i/languages/cpp_api.html#_CPPv4N5 | udaq20simulation_precision4fp64E) |
| cudaq14observe_result8raw_dataEv) | -   [cudaq::SimulationState (C++  |
| -   [cudaq::operator\_handler     |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|     cl                            | #_CPPv4N5cudaq15SimulationStateE) |
| ass)](api/languages/cpp_api.html# | -   [                             |
| _CPPv4N5cudaq16operator_handlerE) | cudaq::SimulationState::precision |
| -   [cudaq::optimizable\_function |     (C++                          |
|     (C++                          |     enum)](api                    |
|     class)                        | /languages/cpp_api.html#_CPPv4N5c |
| ](api/languages/cpp_api.html#_CPP | udaq15SimulationState9precisionE) |
| v4N5cudaq20optimizable_functionE) | -   [cudaq:                       |
| -   [cudaq::optimization\_result  | :SimulationState::precision::fp32 |
|     (C++                          |     (C++                          |
|     type                          |     enumerator)](api/lang         |
| )](api/languages/cpp_api.html#_CP | uages/cpp_api.html#_CPPv4N5cudaq1 |
| Pv4N5cudaq19optimization_resultE) | 5SimulationState9precision4fp32E) |
| -   [cudaq::optimizer (C++        | -   [cudaq:                       |
|     class)](api/languages/cpp_a   | :SimulationState::precision::fp64 |
| pi.html#_CPPv4N5cudaq9optimizerE) |     (C++                          |
| -   [cudaq::optimizer::optimize   |     enumerator)](api/lang         |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq1 |
|                                   | 5SimulationState9precision4fp64E) |
|  function)](api/languages/cpp_api | -                                 |
| .html#_CPPv4N5cudaq9optimizer8opt |   [cudaq::SimulationState::Tensor |
| imizeEKiRR20optimizable_function) |     (C++                          |
| -   [cu                           |     struct)](                     |
| daq::optimizer::requiresGradients | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq15SimulationState6TensorE) |
|     function)](api/la             | -   [cudaq::spin\_handler (C++    |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q9optimizer17requiresGradientsEv) |   class)](api/languages/cpp_api.h |
| -   [cudaq::orca (C++             | tml#_CPPv4N5cudaq12spin_handlerE) |
|     type)](api/languages/         | -   [cudaq::sp                    |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | in\_handler::to\_diagonal\_matrix |
| -   [cudaq::orca::sample (C++     |     (C++                          |
|     function)](api/languages/c    |     function)](api/la             |
| pp_api.html#_CPPv4N5cudaq4orca6sa | nguages/cpp_api.html#_CPPv4NK5cud |
| mpleERNSt6vectorINSt6size_tEEERNS | aq12spin_handler18to_diagonal_mat |
| t6vectorINSt6size_tEEERNSt6vector | rixERNSt13unordered_mapINSt6size_ |
| IdEERNSt6vectorIdEEiNSt6size_tE), | tENSt7int64_tEEERKNSt13unordered_ |
|     [\[1\]]                       | mapINSt6stringENSt7complexIdEEEE) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq4orca6sampleERNSt6vectorI | [cudaq::spin\_handler::to\_matrix |
| NSt6size_tEEERNSt6vectorINSt6size |     (C++                          |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     function                      |
| -   [cudaq::orca::sample\_async   | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq12spin_handler9to_matri |
|                                   | xERKNSt6stringENSt7complexIdEEb), |
| function)](api/languages/cpp_api. |     [\[1                          |
| html#_CPPv4N5cudaq4orca12sample_a | \]](api/languages/cpp_api.html#_C |
| syncERNSt6vectorINSt6size_tEEERNS | PPv4NK5cudaq12spin_handler9to_mat |
| t6vectorINSt6size_tEEERNSt6vector | rixERNSt13unordered_mapINSt6size_ |
| IdEERNSt6vectorIdEEiNSt6size_tE), | tENSt7int64_tEEERKNSt13unordered_ |
|     [\[1\]](api/la                | mapINSt6stringENSt7complexIdEEEE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::                      |
| q4orca12sample_asyncERNSt6vectorI | spin\_handler::to\_sparse\_matrix |
| NSt6size_tEEERNSt6vectorINSt6size |     (C++                          |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq12spin_handler16to_sparse_matr |
|                                   | ixERKNSt6stringENSt7complexIdEEb) |
|                                   | -                                 |
|                                   | [cudaq::spin\_handler::to\_string |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq12spin_handler9to_stringEb) |
|                                   | -                                 |
|                                   | [cudaq::spin\_handler::unique\_id |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq12spin_handler9unique_idEv) |
|                                   | -   [cudaq::spin\_op (C++         |
|                                   |     type)](api/languages/cpp      |
|                                   | _api.html#_CPPv4N5cudaq7spin_opE) |
|                                   | -   [cudaq::spin\_op\_term (C++   |
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
|                                   | -   [cudaq::state::from\_data     |
|                                   |     (C++                          |
|                                   |     function)](api/la             |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q5state9from_dataERK10state_data) |
|                                   | -                                 |
|                                   |   [cudaq::state::get\_num\_qubits |
|                                   |     (C++                          |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq5state14get_num_qubitsEv) |
|                                   | -                                 |
|                                   |  [cudaq::state::get\_num\_tensors |
|                                   |     (C++                          |
|                                   |     function)](a                  |
|                                   | pi/languages/cpp_api.html#_CPPv4N |
|                                   | K5cudaq5state15get_num_tensorsEv) |
|                                   | -   [cudaq::state::get\_precision |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq5state13get_precisionEv) |
|                                   | -   [cudaq::state::get\_tensor    |
|                                   |     (C++                          |
|                                   |     function)](api/la             |
|                                   | nguages/cpp_api.html#_CPPv4NK5cud |
|                                   | aq5state10get_tensorENSt6size_tE) |
|                                   | -   [cudaq::state::get\_tensors   |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq5state11get_tensorsEv) |
|                                   | -   [cudaq::state::is\_on\_gpu    |
|                                   |     (C++                          |
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
|                                   | -   [cudaq::state::to\_host (C++  |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | I0ENK5cudaq5state7to_hostEvPNSt7c |
|                                   | omplexI10ScalarTypeEENSt6size_tE) |
|                                   | -   [cudaq::state\_data (C++      |
|                                   |     type)](api/languages/cpp_api  |
|                                   | .html#_CPPv4N5cudaq10state_dataE) |
|                                   | -   [cudaq::sum\_op (C++          |
|                                   |     class)](api/languages/cpp_a   |
|                                   | pi.html#_CPPv4I0EN5cudaq6sum_opE) |
|                                   | -   [cudaq::sum\_op::begin (C++   |
|                                   |     fu                            |
|                                   | nction)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4NK5cudaq6sum_op5beginEv) |
|                                   | -   [cudaq::sum\_op::canonicalize |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |  function)](api/languages/cpp_api |
|                                   | .html#_CPPv4N5cudaq6sum_op12canon |
|                                   | icalizeERKNSt3setINSt6size_tEEE), |
|                                   |     [\[1\]                        |
|                                   | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq6sum_op12canonicalizeEv) |
|                                   | -                                 |
|                                   |  [cudaq::sum\_op::const\_iterator |
|                                   |     (C++                          |
|                                   |     struct)]                      |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4N5cudaq6sum_op14const_iteratorE) |
|                                   | -   [cudaq::sum                   |
|                                   | \_op::const\_iterator::operator!= |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq6sum_op14con |
|                                   | st_iteratorneERK14const_iterator) |
|                                   | -   [cudaq::sum                   |
|                                   | \_op::const\_iterator::operator\* |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratormlEv) |
|                                   | -   [cudaq::sum                   |
|                                   | \_op::const\_iterator::operator++ |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratorppEv) |
|                                   | -   [cudaq::sum\                  |
|                                   | _op::const\_iterator::operator-\> |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratorptEv) |
|                                   | -   [cudaq::sum                   |
|                                   | \_op::const\_iterator::operator== |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq6sum_op14con |
|                                   | st_iteratoreqERK14const_iterator) |
|                                   | -   [cudaq::sum\_op::degrees (C++ |
|                                   |     func                          |
|                                   | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4NK5cudaq6sum_op7degreesEv) |
|                                   | -   [                             |
|                                   | cudaq::sum\_op::distribute\_terms |
|                                   |     (C++                          |
|                                   |     function)](api/languages      |
|                                   | /cpp_api.html#_CPPv4NK5cudaq6sum_ |
|                                   | op16distribute_termsENSt6size_tE) |
|                                   | -   [cudaq::sum\_op::dump (C++    |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4NK5cudaq6sum_op4dumpEv) |
|                                   | -   [cudaq::sum\_op::empty (C++   |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq6sum_op5emptyEv) |
|                                   | -   [cudaq::sum\_op::end (C++     |
|                                   |                                   |
|                                   | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq6sum_op3endEv) |
|                                   | -   [cudaq::sum\_op::identity     |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq6sum_op8identityENSt6size_tE), |
|                                   |     [                             |
|                                   | \[1\]](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq6sum_op8identityEv) |
|                                   | -   [cudaq::sum\_op::num\_terms   |
|                                   |     (C++                          |
|                                   |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq6sum_op9num_termsEv) |
|                                   | -   [cudaq::sum\_op::operator\*   |
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
|                                   | -   [cudaq::sum\_op::operator\*=  |
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
|                                   | -   [cudaq::sum\_op::operator+    |
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
|                                   | -   [cudaq::sum\_op::operator+=   |
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
|                                   | -   [cudaq::sum\_op::operator-    |
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
|                                   | -   [cudaq::sum\_op::operator-=   |
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
|                                   | -   [cudaq::sum\_op::operator/    |
|                                   |     (C++                          |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opdvERK15scalar_operator), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4NO5cu |
|                                   | daq6sum_opdvERK15scalar_operator) |
|                                   | -   [cudaq::sum\_op::operator/=   |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq6sum_opdVERK15scalar_operator) |
|                                   | -   [cudaq::sum\_op::operator=    |
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
|                                   | -   [cudaq::sum\_op::operator==   |
|                                   |     (C++                          |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4NK5cuda |
|                                   | q6sum_opeqERK6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum\_op::operator\[\] |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq6sum_opixENSt6size_tE) |
|                                   | -   [cudaq::sum\_op::sum\_op (C++ |
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
|                                   | -   [cud                          |
|                                   | aq::sum\_op::to\_diagonal\_matrix |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq6sum_op18to_diagonal_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -   [cudaq::sum\_op::to\_matrix   |
|                                   |     (C++                          |
|                                   |                                   |
|                                   | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq6sum_op9to_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -   [c                            |
|                                   | udaq::sum\_op::to\_sparse\_matrix |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq6sum_op16to_sparse_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -   [cudaq::sum\_op::to\_string   |
|                                   |     (C++                          |
|                                   |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq6sum_op9to_stringEv) |
|                                   | -   [cudaq::sum\_op::trim (C++    |
|                                   |     function)](api/l              |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_op4trimEdRKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [cudaq::sum\_op::\~sum\_op    |
|                                   |     (C++                          |
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
|                                   | -   [cudaq::unset\_noise (C++     |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq11unset_noiseEv) |
|                                   | -   [cudaq::x\_error (C++         |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7x_errorE) |
|                                   | -   [cudaq::y\_error (C++         |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7y_errorE) |
|                                   | -                                 |
|                                   | [cudaq::y\_error::num\_parameters |
|                                   |     (C++                          |
|                                   |     member)](                     |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq7y_error14num_parametersE) |
|                                   | -                                 |
|                                   |    [cudaq::y\_error::num\_targets |
|                                   |     (C++                          |
|                                   |     member                        |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq7y_error11num_targetsE) |
|                                   | -   [cudaq::z\_error (C++         |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7z_errorE) |
+-----------------------------------+-----------------------------------+

D {#D}
-

+-----------------------------------+-----------------------------------+
| -   [define() (cudaq.operators    | -   [displace() (in module        |
|     method)](api/languages/python |     cudaq.operators.custo         |
| _api.html#cudaq.operators.define) | m)](api/languages/python_api.html |
|     -   [(cuda                    | #cudaq.operators.custom.displace) |
| q.operators.MatrixOperatorElement | -   [distribute\_terms()          |
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

E {#E}
-

+-----------------------------------+-----------------------------------+
| -   [ElementaryOperator (in       | -   [evaluate\_coefficient()      |
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
| languages/python_api.html#cudaq.o | -   [evolve\_async() (in module   |
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
|                                   | -   [expectation\_values()        |
|       cudaq.spin)](api/languages/ |     (cudaq.EvolveResult           |
| python_api.html#cudaq.spin.empty) |     method)](ap                   |
| -   [empty\_op()                  | i/languages/python_api.html#cudaq |
|     (                             | .EvolveResult.expectation_values) |
| cudaq.operators.spin.SpinOperator | -   [expectation\_z()             |
|     static                        |     (cudaq.SampleResult           |
|     method)](api/lan              |     method                        |
| guages/python_api.html#cudaq.oper | )](api/languages/python_api.html# |
| ators.spin.SpinOperator.empty_op) | cudaq.SampleResult.expectation_z) |
| -   [enable\_return\_to\_log()    | -   [expected\_dimensions         |
|     (cudaq.PyKernelDecorator      |     (cuda                         |
|     method)](api/langu            | q.operators.MatrixOperatorElement |
| ages/python_api.html#cudaq.PyKern |                                   |
| elDecorator.enable_return_to_log) | property)](api/languages/python_a |
| -   [estimate\_resources() (in    | pi.html#cudaq.operators.MatrixOpe |
|     module                        | ratorElement.expected_dimensions) |
|                                   | -                                 |
|    cudaq)](api/languages/python_a |  [extract\_c\_function\_pointer() |
| pi.html#cudaq.estimate_resources) |     (cudaq.PyKernelDecorator      |
| -   [evaluate()                   |     method)](api/languages/p      |
|                                   | ython_api.html#cudaq.PyKernelDeco |
|   (cudaq.operators.ScalarOperator | rator.extract_c_function_pointer) |
|     method)](api/                 |                                   |
| languages/python_api.html#cudaq.o |                                   |
| perators.ScalarOperator.evaluate) |                                   |
+-----------------------------------+-----------------------------------+

F {#F}
-

+-----------------------------------+-----------------------------------+
| -   [FermionOperator (class in    | -   [from\_json()                 |
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
| -   [final\_expectation\_values() | anguages/python_api.html#cudaq.gr |
|     (cudaq.EvolveResult           | adients.ParameterShift.from_json) |
|     method)](api/lang             |     -   [(                        |
| uages/python_api.html#cudaq.Evolv | cudaq.operators.spin.SpinOperator |
| eResult.final_expectation_values) |         static                    |
| -   [final\_state()               |         method)](api/lang         |
|     (cudaq.EvolveResult           | uages/python_api.html#cudaq.opera |
|     meth                          | tors.spin.SpinOperator.from_json) |
| od)](api/languages/python_api.htm |     -   [(cuda                    |
| l#cudaq.EvolveResult.final_state) | q.operators.spin.SpinOperatorTerm |
| -   [finalize() (in module        |         static                    |
|     cudaq.mpi)](api/languages/py  |         method)](api/language     |
| thon_api.html#cudaq.mpi.finalize) | s/python_api.html#cudaq.operators |
| -   [for\_each\_pauli()           | .spin.SpinOperatorTerm.from_json) |
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
| -   [for\_each\_term()            |     -   [(cudaq.optimizers.LBFGS  |
|     (                             |         static                    |
| cudaq.operators.spin.SpinOperator |         method                    |
|     method)](api/language         | )](api/languages/python_api.html# |
| s/python_api.html#cudaq.operators | cudaq.optimizers.LBFGS.from_json) |
| .spin.SpinOperator.for_each_term) |                                   |
| -   [ForwardDifference (class in  | -   [(cudaq.optimizers.NelderMead |
|     cudaq.gradients)              |         static                    |
| ](api/languages/python_api.html#c |         method)](ap               |
| udaq.gradients.ForwardDifference) | i/languages/python_api.html#cudaq |
| -   [from\_data() (cudaq.State    | .optimizers.NelderMead.from_json) |
|     static                        |     -   [(cudaq.PyKernelDecorator |
|     method)](api/languages/pytho  |         static                    |
| n_api.html#cudaq.State.from_data) |         method)                   |
|                                   | ](api/languages/python_api.html#c |
|                                   | udaq.PyKernelDecorator.from_json) |
|                                   | -   [from\_word()                 |
|                                   |     (                             |
|                                   | cudaq.operators.spin.SpinOperator |
|                                   |     static                        |
|                                   |     method)](api/lang             |
|                                   | uages/python_api.html#cudaq.opera |
|                                   | tors.spin.SpinOperator.from_word) |
+-----------------------------------+-----------------------------------+

G {#G}
-

+-----------------------------------+-----------------------------------+
| -   [get()                        | -   [get\_raw\_data()             |
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
|         m                         | -   [get\_register\_counts()      |
| ethod)](api/languages/python_api. |     (cudaq.SampleResult           |
| html#cudaq.AsyncSampleResult.get) |     method)](api                  |
|     -   [(cudaq.AsyncStateResult  | /languages/python_api.html#cudaq. |
|                                   | SampleResult.get_register_counts) |
| method)](api/languages/python_api | -   [get\_sequential\_data()      |
| .html#cudaq.AsyncStateResult.get) |     (cudaq.SampleResult           |
| -                                 |     method)](api                  |
|  [get\_binary\_symplectic\_form() | /languages/python_api.html#cudaq. |
|     (cuda                         | SampleResult.get_sequential_data) |
| q.operators.spin.SpinOperatorTerm | -   [get\_spin()                  |
|     metho                         |     (cudaq.ObserveResult          |
| d)](api/languages/python_api.html |     me                            |
| #cudaq.operators.spin.SpinOperato | thod)](api/languages/python_api.h |
| rTerm.get_binary_symplectic_form) | tml#cudaq.ObserveResult.get_spin) |
| -   [get\_channels()              | -   [get\_state() (in module      |
|     (cudaq.NoiseModel             |     cudaq)](api/languages         |
|     met                           | /python_api.html#cudaq.get_state) |
| hod)](api/languages/python_api.ht | -   [get\_state\_async() (in      |
| ml#cudaq.NoiseModel.get_channels) |     module                        |
| -   [get\_coefficient()           |     cudaq)](api/languages/pytho   |
|     (                             | n_api.html#cudaq.get_state_async) |
| cudaq.operators.spin.SpinOperator | -   [get\_target() (in module     |
|     method)](api/languages/       |     cudaq)](api/languages/        |
| python_api.html#cudaq.operators.s | python_api.html#cudaq.get_target) |
| pin.SpinOperator.get_coefficient) | -   [get\_targets() (in module    |
|     -   [(cuda                    |     cudaq)](api/languages/p       |
| q.operators.spin.SpinOperatorTerm | ython_api.html#cudaq.get_targets) |
|                                   | -   [get\_term\_count()           |
|       method)](api/languages/pyth |     (                             |
| on_api.html#cudaq.operators.spin. | cudaq.operators.spin.SpinOperator |
| SpinOperatorTerm.get_coefficient) |     method)](api/languages        |
| -   [get\_marginal\_counts()      | /python_api.html#cudaq.operators. |
|     (cudaq.SampleResult           | spin.SpinOperator.get_term_count) |
|     method)](api                  | -   [get\_total\_shots()          |
| /languages/python_api.html#cudaq. |     (cudaq.SampleResult           |
| SampleResult.get_marginal_counts) |     method)]                      |
| -   [get\_ops()                   | (api/languages/python_api.html#cu |
|     (cudaq.KrausChannel           | daq.SampleResult.get_total_shots) |
|                                   | -   [getTensor() (cudaq.State     |
| method)](api/languages/python_api |     method)](api/languages/pytho  |
| .html#cudaq.KrausChannel.get_ops) | n_api.html#cudaq.State.getTensor) |
| -   [get\_pauli\_word()           | -   [getTensors() (cudaq.State    |
|     (cuda                         |     method)](api/languages/python |
| q.operators.spin.SpinOperatorTerm | _api.html#cudaq.State.getTensors) |
|     method)](api/languages/pyt    | -   [gradient (class in           |
| hon_api.html#cudaq.operators.spin |     cudaq.g                       |
| .SpinOperatorTerm.get_pauli_word) | radients)](api/languages/python_a |
| -   [get\_precision()             | pi.html#cudaq.gradients.gradient) |
|     (cudaq.Target                 | -   [GradientDescent (class in    |
|                                   |     cudaq.optimizers              |
| method)](api/languages/python_api | )](api/languages/python_api.html# |
| .html#cudaq.Target.get_precision) | cudaq.optimizers.GradientDescent) |
| -   [get\_qubit\_count()          |                                   |
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

H {#H}
-

+-----------------------------------------------------------------------+
| -   [has\_target() (in module                                         |
|     cudaq)](api/languages/python_api.html#cudaq.has_target)           |
+-----------------------------------------------------------------------+

I {#I}
-

+-----------------------------------+-----------------------------------+
| -   [i() (in module               | -   [initialize\_cudaq() (in      |
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
| udaq.operators.custom.identities) | -   [intermediate\_states()       |
|     -   [(in module               |     (cudaq.EvolveResult           |
|                                   |     method)](api                  |
|  cudaq.spin)](api/languages/pytho | /languages/python_api.html#cudaq. |
| n_api.html#cudaq.spin.identities) | EvolveResult.intermediate_states) |
| -   [identity()                   | -   [IntermediateResultSave       |
|     (cu                           |     (class in                     |
| daq.operators.boson.BosonOperator |     c                             |
|     static                        | udaq)](api/languages/python_api.h |
|     method)](api/langu            | tml#cudaq.IntermediateResultSave) |
| ages/python_api.html#cudaq.operat | -   [is\_constant()               |
| ors.boson.BosonOperator.identity) |                                   |
|     -   [(cudaq.                  |   (cudaq.operators.ScalarOperator |
| operators.fermion.FermionOperator |     method)](api/lan              |
|         static                    | guages/python_api.html#cudaq.oper |
|         method)](api/languages    | ators.ScalarOperator.is_constant) |
| /python_api.html#cudaq.operators. | -   [is\_emulated() (cudaq.Target |
| fermion.FermionOperator.identity) |                                   |
|     -                             |   method)](api/languages/python_a |
|  [(cudaq.operators.MatrixOperator | pi.html#cudaq.Target.is_emulated) |
|         static                    | -   [is\_identity()               |
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
| -   [initial\_parameters          |     -   [(cuda                    |
|     (cudaq.optimizers.COBYLA      | q.operators.spin.SpinOperatorTerm |
|     property)](api/lan            |         method)](api/languages/   |
| guages/python_api.html#cudaq.opti | python_api.html#cudaq.operators.s |
| mizers.COBYLA.initial_parameters) | pin.SpinOperatorTerm.is_identity) |
|     -   [                         | -   [is\_initialized() (in module |
| (cudaq.optimizers.GradientDescent |     c                             |
|                                   | udaq.mpi)](api/languages/python_a |
|       property)](api/languages/py | pi.html#cudaq.mpi.is_initialized) |
| thon_api.html#cudaq.optimizers.Gr | -   [is\_on\_gpu() (cudaq.State   |
| adientDescent.initial_parameters) |     method)](api/languages/pytho  |
|     -   [(cudaq.optimizers.LBFGS  | n_api.html#cudaq.State.is_on_gpu) |
|         property)](api/la         | -   [is\_remote() (cudaq.Target   |
| nguages/python_api.html#cudaq.opt |     method)](api/languages/python |
| imizers.LBFGS.initial_parameters) | _api.html#cudaq.Target.is_remote) |
|                                   | -   [is\_remote\_simulator()      |
| -   [(cudaq.optimizers.NelderMead |     (cudaq.Target                 |
|         property)](api/languag    |     method                        |
| es/python_api.html#cudaq.optimize | )](api/languages/python_api.html# |
| rs.NelderMead.initial_parameters) | cudaq.Target.is_remote_simulator) |
| -   [initialize() (in module      | -   [items() (cudaq.SampleResult  |
|                                   |                                   |
|    cudaq.mpi)](api/languages/pyth |   method)](api/languages/python_a |
| on_api.html#cudaq.mpi.initialize) | pi.html#cudaq.SampleResult.items) |
+-----------------------------------+-----------------------------------+

K {#K}
-

+-----------------------------------+-----------------------------------+
| -   [Kernel (in module            | -   [KrausChannel (class in       |
|     cudaq)](api/langua            |     cudaq)](api/languages/py      |
| ges/python_api.html#cudaq.Kernel) | thon_api.html#cudaq.KrausChannel) |
| -   [kernel() (in module          | -   [KrausOperator (class in      |
|     cudaq)](api/langua            |     cudaq)](api/languages/pyt     |
| ges/python_api.html#cudaq.kernel) | hon_api.html#cudaq.KrausOperator) |
+-----------------------------------+-----------------------------------+

L {#L}
-

+-----------------------------------+-----------------------------------+
| -   [LBFGS (class in              | -   [lower\_bounds                |
|     cudaq.                        |     (cudaq.optimizers.COBYLA      |
| optimizers)](api/languages/python |     property)](a                  |
| _api.html#cudaq.optimizers.LBFGS) | pi/languages/python_api.html#cuda |
| -   [left\_multiply()             | q.optimizers.COBYLA.lower_bounds) |
|     (cudaq.SuperOperator static   |     -   [                         |
|     method)                       | (cudaq.optimizers.GradientDescent |
| ](api/languages/python_api.html#c |         property)](api/langua     |
| udaq.SuperOperator.left_multiply) | ges/python_api.html#cudaq.optimiz |
| -   [left\_right\_multiply()      | ers.GradientDescent.lower_bounds) |
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

M {#M}
-

+-----------------------------------+-----------------------------------+
| -   [make\_kernel() (in module    | -   [min\_degree                  |
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
| -   [max\_degree                  | ython_api.html#cudaq.operators.fe |
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
| rators.MatrixOperator.max_degree) | -   [minimal\_eigenvalue()        |
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
| -   [max\_iterations              | on_api.html#module-cudaq.fermion) |
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
| -   [(cudaq.optimizers.NelderMead | -   [most\_probable()             |
|         property)](api/lan        |     (cudaq.SampleResult           |
| guages/python_api.html#cudaq.opti |     method                        |
| mizers.NelderMead.max_iterations) | )](api/languages/python_api.html# |
| -   [mdiag\_sparse\_matrix (C++   | cudaq.SampleResult.most_probable) |
|     type)](api/languages/cpp_api. |                                   |
| html#_CPPv419mdiag_sparse_matrix) |                                   |
| -   [merge\_kernel()              |                                   |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](a                    |                                   |
| pi/languages/python_api.html#cuda |                                   |
| q.PyKernelDecorator.merge_kernel) |                                   |
+-----------------------------------+-----------------------------------+

N {#N}
-

+-----------------------------------+-----------------------------------+
| -   [name (cudaq.PyKernel         | -   [num\_qpus() (cudaq.Target    |
|     attribute)](api/languages/pyt |     method)](api/languages/pytho  |
| hon_api.html#cudaq.PyKernel.name) | n_api.html#cudaq.Target.num_qpus) |
|                                   | -   [num\_qubits() (cudaq.State   |
|   -   [(cudaq.SimulationPrecision |     method)](api/languages/python |
|         proper                    | _api.html#cudaq.State.num_qubits) |
| ty)](api/languages/python_api.htm | -   [num\_ranks() (in module      |
| l#cudaq.SimulationPrecision.name) |     cudaq.mpi)](api/languages/pyt |
|     -   [(cudaq.spin.Pauli        | hon_api.html#cudaq.mpi.num_ranks) |
|                                   | -   [num\_rows()                  |
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
| -   [num\_available\_gpus() (in   |         cudaq.operators.cus       |
|     module                        | tom)](api/languages/python_api.ht |
|                                   | ml#cudaq.operators.custom.number) |
|    cudaq)](api/languages/python_a | -   [nvqir::MPSSimulationState    |
| pi.html#cudaq.num_available_gpus) |     (C++                          |
| -   [num\_columns()               |     class)]                       |
|     (cudaq.ComplexMatrix          | (api/languages/cpp_api.html#_CPPv |
|     metho                         | 4I0EN5nvqir18MPSSimulationStateE) |
| d)](api/languages/python_api.html | -                                 |
| #cudaq.ComplexMatrix.num_columns) |  [nvqir::TensorNetSimulationState |
|                                   |     (C++                          |
|                                   |     class)](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4I0EN5 |
|                                   | nvqir24TensorNetSimulationStateE) |
+-----------------------------------+-----------------------------------+

O {#O}
-

+-----------------------------------+-----------------------------------+
| -   [observe() (in module         | -   [OptimizationResult (class in |
|     cudaq)](api/languag           |                                   |
| es/python_api.html#cudaq.observe) |    cudaq)](api/languages/python_a |
| -   [observe\_async() (in module  | pi.html#cudaq.OptimizationResult) |
|     cudaq)](api/languages/pyt     | -   [optimize()                   |
| hon_api.html#cudaq.observe_async) |     (cudaq.optimizers.COBYLA      |
| -   [ObserveResult (class in      |     method                        |
|     cudaq)](api/languages/pyt     | )](api/languages/python_api.html# |
| hon_api.html#cudaq.ObserveResult) | cudaq.optimizers.COBYLA.optimize) |
| -   [OperatorSum (in module       |     -   [                         |
|     cudaq.oper                    | (cudaq.optimizers.GradientDescent |
| ators)](api/languages/python_api. |         method)](api/la           |
| html#cudaq.operators.OperatorSum) | nguages/python_api.html#cudaq.opt |
| -   [ops\_count                   | imizers.GradientDescent.optimize) |
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

P {#P}
-

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

Q {#Q}
-

+-----------------------------------+-----------------------------------+
| -   [qreg (in module              | -   [qubit\_count                 |
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

R {#R}
-

+-----------------------------------+-----------------------------------+
| -   [random()                     | -   [reset\_target() (in module   |
|     (                             |     cudaq)](api/languages/py      |
| cudaq.operators.spin.SpinOperator | thon_api.html#cudaq.reset_target) |
|     static                        | -   [Resources (class in          |
|     method)](api/l                |     cudaq)](api/languages         |
| anguages/python_api.html#cudaq.op | /python_api.html#cudaq.Resources) |
| erators.spin.SpinOperator.random) | -   [right\_multiply()            |
| -   [rank() (in module            |     (cudaq.SuperOperator static   |
|     cudaq.mpi)](api/language      |     method)]                      |
| s/python_api.html#cudaq.mpi.rank) | (api/languages/python_api.html#cu |
| -   [register\_names              | daq.SuperOperator.right_multiply) |
|     (cudaq.SampleResult           | -   [row\_count                   |
|     attribute)                    |     (cudaq.KrausOperator          |
| ](api/languages/python_api.html#c |     prope                         |
| udaq.SampleResult.register_names) | rty)](api/languages/python_api.ht |
| -   [                             | ml#cudaq.KrausOperator.row_count) |
| register\_set\_target\_callback() | -   [run() (in module             |
|     (in module                    |     cudaq)](api/lan               |
|     cudaq)]                       | guages/python_api.html#cudaq.run) |
| (api/languages/python_api.html#cu | -   [run\_async() (in module      |
| daq.register_set_target_callback) |     cudaq)](api/languages         |
| -   [requires\_gradients()        | /python_api.html#cudaq.run_async) |
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

S {#S}
-

+-----------------------------------+-----------------------------------+
| -   [sample() (in module          | -   [set\_target() (in module     |
|     cudaq)](api/langua            |     cudaq)](api/languages/        |
| ges/python_api.html#cudaq.sample) | python_api.html#cudaq.set_target) |
|     -   [(in module               | -   [SimulationPrecision (class   |
|                                   |     in                            |
|      cudaq.orca)](api/languages/p |                                   |
| ython_api.html#cudaq.orca.sample) |   cudaq)](api/languages/python_ap |
| -   [sample\_async() (in module   | i.html#cudaq.SimulationPrecision) |
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
| -   [set\_noise() (in module      | -   [SuperOperator (class in      |
|     cudaq)](api/languages         |     cudaq)](api/languages/pyt     |
| /python_api.html#cudaq.set_noise) | hon_api.html#cudaq.SuperOperator) |
| -   [set\_random\_seed() (in      | -   [                             |
|     module                        | synthesize\_callable\_arguments() |
|     cudaq)](api/languages/pytho   |     (cudaq.PyKernelDecorator      |
| n_api.html#cudaq.set_random_seed) |     method)](api/languages/pyth   |
|                                   | on_api.html#cudaq.PyKernelDecorat |
|                                   | or.synthesize_callable_arguments) |
+-----------------------------------+-----------------------------------+

T {#T}
-

+-----------------------------------+-----------------------------------+
| -   [Target (class in             | -   [to\_numpy()                  |
|     cudaq)](api/langua            |     (cudaq.ComplexMatrix          |
| ges/python_api.html#cudaq.Target) |     me                            |
| -   [target                       | thod)](api/languages/python_api.h |
|     (cudaq.ope                    | tml#cudaq.ComplexMatrix.to_numpy) |
| rators.boson.BosonOperatorElement | -   [to\_sparse\_matrix()         |
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
| -   [term\_count                  |     -   [(cudaq.oper              |
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
|     -   [(                        | -   [to\_string()                 |
| cudaq.operators.spin.SpinOperator |     (cudaq.ope                    |
|         property)](api/langu      | rators.boson.BosonOperatorElement |
| ages/python_api.html#cudaq.operat |     method)](api/languages/pyt    |
| ors.spin.SpinOperator.term_count) | hon_api.html#cudaq.operators.boso |
| -   [term\_id                     | n.BosonOperatorElement.to_string) |
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
| -   [to\_json()                   |     -   [(cuda                    |
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
| #cudaq.optimizers.COBYLA.to_json) | -   [type\_to\_str()              |
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
| -   [to\_matrix()                 |                                   |
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

U {#U}
-

+-----------------------------------------------------------------------+
| -   [unregister\_set\_target\_callback() (in module                   |
|     cudaq)                                                            |
| ](api/languages/python_api.html#cudaq.unregister_set_target_callback) |
| -   [unset\_noise() (in module                                        |
|     cudaq)](api/languages/python_api.html#cudaq.unset_noise)          |
| -   [upper\_bounds (cudaq.optimizers.COBYLA                           |
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

V {#V}
-

+-----------------------------------+-----------------------------------+
| -   [values() (cudaq.SampleResult | -   [vqe() (in module             |
|                                   |     cudaq)](api/lan               |
|  method)](api/languages/python_ap | guages/python_api.html#cudaq.vqe) |
| i.html#cudaq.SampleResult.values) |                                   |
+-----------------------------------+-----------------------------------+

X {#X}
-

+-----------------------------------+-----------------------------------+
| -   [x() (in module               | -   [XError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.x) | ges/python_api.html#cudaq.XError) |
+-----------------------------------+-----------------------------------+

Y {#Y}
-

+-----------------------------------+-----------------------------------+
| -   [y() (in module               | -   [YError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.y) | ges/python_api.html#cudaq.YError) |
+-----------------------------------+-----------------------------------+

Z {#Z}
-

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
