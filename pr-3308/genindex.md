::: {.wy-grid-for-nav}
::: {.wy-side-scroll}
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: {.version}
pr-3308
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
        -   [(a) Generate the molecular Hamiltonian using Hartree Fock
            molecular
            orbitals](applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Hartree-Fock-molecular-orbitals){.reference
            .internal}
            -   [Active space
                Hamiltonian:](applications/python/generate_fermionic_ham.html#Active-space-Hamiltonian:){.reference
                .internal}
        -   [(b) Generate the active space hamiltonian using HF
            molecular
            orbitals.](applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-hamiltonian-using-HF-molecular-orbitals.){.reference
            .internal}
        -   [(c) Generate the active space Hamiltonian using the natural
            orbitals computed from MP2
            simulation](applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
            .internal}
        -   [(d) Generate the active space Hamiltonian computed from the
            CASSCF molecular
            orbitals](applications/python/generate_fermionic_ham.html#(d)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
            .internal}
            -   [Generate the electronic Hamiltonian using
                ROHF](applications/python/generate_fermionic_ham.html#Generate-the-electronic-Hamiltonian-using-ROHF){.reference
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
| -   [canonicalize()               | -   [cudaq::pauli1 (C++           |
|     (cu                           |     class)](api/languages/cp      |
| daq.operators.boson.BosonOperator | p_api.html#_CPPv4N5cudaq6pauli1E) |
|     method)](api/languages        | -                                 |
| /python_api.html#cudaq.operators. |   [cudaq::pauli1::num\_parameters |
| boson.BosonOperator.canonicalize) |     (C++                          |
|     -   [(cudaq.                  |     member)]                      |
| operators.boson.BosonOperatorTerm | (api/languages/cpp_api.html#_CPPv |
|                                   | 4N5cudaq6pauli114num_parametersE) |
|        method)](api/languages/pyt | -   [cudaq::pauli1::num\_targets  |
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
| api.html#cudaq.operators.fermion. |   [cudaq::pauli2::num\_parameters |
| FermionOperatorTerm.canonicalize) |     (C++                          |
|     -                             |     member)]                      |
|  [(cudaq.operators.MatrixOperator | (api/languages/cpp_api.html#_CPPv |
|         method)](api/lang         | 4N5cudaq6pauli214num_parametersE) |
| uages/python_api.html#cudaq.opera | -   [cudaq::pauli2::num\_targets  |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     membe                         |
| udaq.operators.MatrixOperatorTerm | r)](api/languages/cpp_api.html#_C |
|         method)](api/language     | PPv4N5cudaq6pauli211num_targetsE) |
| s/python_api.html#cudaq.operators | -   [cudaq::pauli2::pauli2 (C++   |
| .MatrixOperatorTerm.canonicalize) |     function)](api/languages/cpp_ |
|     -   [(                        | api.html#_CPPv4N5cudaq6pauli26pau |
| cudaq.operators.spin.SpinOperator | li2ERKNSt6vectorIN5cudaq4realEEE) |
|         method)](api/languag      | -   [cudaq::phase\_damping (C++   |
| es/python_api.html#cudaq.operator |                                   |
| s.spin.SpinOperator.canonicalize) |  class)](api/languages/cpp_api.ht |
|     -   [(cuda                    | ml#_CPPv4N5cudaq13phase_dampingE) |
| q.operators.spin.SpinOperatorTerm | -   [cudaq                        |
|         method)](api/languages/p  | ::phase\_damping::num\_parameters |
| ython_api.html#cudaq.operators.sp |     (C++                          |
| in.SpinOperatorTerm.canonicalize) |     member)](api/lan              |
| -   [canonicalized() (in module   | guages/cpp_api.html#_CPPv4N5cudaq |
|     cuda                          | 13phase_damping14num_parametersE) |
| q.boson)](api/languages/python_ap | -   [cu                           |
| i.html#cudaq.boson.canonicalized) | daq::phase\_damping::num\_targets |
|     -   [(in module               |     (C++                          |
|         cudaq.fe                  |     member)](api/                 |
| rmion)](api/languages/python_api. | languages/cpp_api.html#_CPPv4N5cu |
| html#cudaq.fermion.canonicalized) | daq13phase_damping11num_targetsE) |
|     -   [(in module               | -   [cudaq::phase\_flip\_channel  |
|                                   |     (C++                          |
|        cudaq.operators.custom)](a |     clas                          |
| pi/languages/python_api.html#cuda | s)](api/languages/cpp_api.html#_C |
| q.operators.custom.canonicalized) | PPv4N5cudaq18phase_flip_channelE) |
|     -   [(in module               | -   [cudaq::phas                  |
|         cu                        | e\_flip\_channel::num\_parameters |
| daq.spin)](api/languages/python_a |     (C++                          |
| pi.html#cudaq.spin.canonicalized) |     member)](api/language         |
| -   [CentralDifference (class in  | s/cpp_api.html#_CPPv4N5cudaq18pha |
|     cudaq.gradients)              | se_flip_channel14num_parametersE) |
| ](api/languages/python_api.html#c | -   [cudaq::p                     |
| udaq.gradients.CentralDifference) | hase\_flip\_channel::num\_targets |
| -   [clear() (cudaq.SampleResult  |     (C++                          |
|                                   |     member)](api/langu            |
|   method)](api/languages/python_a | ages/cpp_api.html#_CPPv4N5cudaq18 |
| pi.html#cudaq.SampleResult.clear) | phase_flip_channel11num_targetsE) |
| -   [COBYLA (class in             | -   [cudaq::product\_op (C++      |
|     cudaq.o                       |                                   |
| ptimizers)](api/languages/python_ |  class)](api/languages/cpp_api.ht |
| api.html#cudaq.optimizers.COBYLA) | ml#_CPPv4I0EN5cudaq10product_opE) |
| -   [coefficient                  | -   [cudaq::product\_op::begin    |
|     (cudaq.                       |     (C++                          |
| operators.boson.BosonOperatorTerm |     functio                       |
|     property)](api/languages/py   | n)](api/languages/cpp_api.html#_C |
| thon_api.html#cudaq.operators.bos | PPv4NK5cudaq10product_op5beginEv) |
| on.BosonOperatorTerm.coefficient) | -                                 |
|     -   [(cudaq.oper              | [cudaq::product\_op::canonicalize |
| ators.fermion.FermionOperatorTerm |     (C++                          |
|                                   |     func                          |
|   property)](api/languages/python | tion)](api/languages/cpp_api.html |
| _api.html#cudaq.operators.fermion | #_CPPv4N5cudaq10product_op12canon |
| .FermionOperatorTerm.coefficient) | icalizeERKNSt3setINSt6size_tEEE), |
|     -   [(c                       |     [\[1\]](api                   |
| udaq.operators.MatrixOperatorTerm | /languages/cpp_api.html#_CPPv4N5c |
|         property)](api/languag    | udaq10product_op12canonicalizeEv) |
| es/python_api.html#cudaq.operator | -   [cu                           |
| s.MatrixOperatorTerm.coefficient) | daq::product\_op::const\_iterator |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     struct)](api/                 |
|         property)](api/languages/ | languages/cpp_api.html#_CPPv4N5cu |
| python_api.html#cudaq.operators.s | daq10product_op14const_iteratorE) |
| pin.SpinOperatorTerm.coefficient) | -   [cudaq::product\_op:          |
| -   [col\_count                   | :const\_iterator::const\_iterator |
|     (cudaq.KrausOperator          |     (C++                          |
|     prope                         |     fu                            |
| rty)](api/languages/python_api.ht | nction)](api/languages/cpp_api.ht |
| ml#cudaq.KrausOperator.col_count) | ml#_CPPv4N5cudaq10product_op14con |
| -   [compile()                    | st_iterator14const_iteratorEPK10p |
|     (cudaq.PyKernelDecorator      | roduct_opI9HandlerTyENSt6size_tE) |
|     metho                         | -   [cudaq::product               |
| d)](api/languages/python_api.html | \_op::const\_iterator::operator!= |
| #cudaq.PyKernelDecorator.compile) |     (C++                          |
| -   [ComplexMatrix (class in      |     fun                           |
|     cudaq)](api/languages/pyt     | ction)](api/languages/cpp_api.htm |
| hon_api.html#cudaq.ComplexMatrix) | l#_CPPv4NK5cudaq10product_op14con |
| -   [compute()                    | st_iteratorneERK14const_iterator) |
|     (                             | -   [cudaq::product               |
| cudaq.gradients.CentralDifference | \_op::const\_iterator::operator\* |
|     method)](api/la               |     (C++                          |
| nguages/python_api.html#cudaq.gra |     function)](api/lang           |
| dients.CentralDifference.compute) | uages/cpp_api.html#_CPPv4NK5cudaq |
|     -   [(                        | 10product_op14const_iteratormlEv) |
| cudaq.gradients.ForwardDifference | -   [cudaq::product               |
|         method)](api/la           | \_op::const\_iterator::operator++ |
| nguages/python_api.html#cudaq.gra |     (C++                          |
| dients.ForwardDifference.compute) |     function)](api/lang           |
|     -                             | uages/cpp_api.html#_CPPv4N5cudaq1 |
|  [(cudaq.gradients.ParameterShift | 0product_op14const_iteratorppEi), |
|         method)](api              |     [\[1\]](api/lan               |
| /languages/python_api.html#cudaq. | guages/cpp_api.html#_CPPv4N5cudaq |
| gradients.ParameterShift.compute) | 10product_op14const_iteratorppEv) |
| -   [const()                      | -   [cudaq::product\              |
|                                   | _op::const\_iterator::operator\-- |
|   (cudaq.operators.ScalarOperator |     (C++                          |
|     class                         |     function)](api/lang           |
|     method)](a                    | uages/cpp_api.html#_CPPv4N5cudaq1 |
| pi/languages/python_api.html#cuda | 0product_op14const_iteratormmEi), |
| q.operators.ScalarOperator.const) |     [\[1\]](api/lan               |
| -   [copy()                       | guages/cpp_api.html#_CPPv4N5cudaq |
|     (cu                           | 10product_op14const_iteratormmEv) |
| daq.operators.boson.BosonOperator | -   [cudaq::product\              |
|     method)](api/l                | _op::const\_iterator::operator-\> |
| anguages/python_api.html#cudaq.op |     (C++                          |
| erators.boson.BosonOperator.copy) |     function)](api/lan            |
|     -   [(cudaq.                  | guages/cpp_api.html#_CPPv4N5cudaq |
| operators.boson.BosonOperatorTerm | 10product_op14const_iteratorptEv) |
|         method)](api/langu        | -   [cudaq::product               |
| ages/python_api.html#cudaq.operat | \_op::const\_iterator::operator== |
| ors.boson.BosonOperatorTerm.copy) |     (C++                          |
|     -   [(cudaq.                  |     fun                           |
| operators.fermion.FermionOperator | ction)](api/languages/cpp_api.htm |
|         method)](api/langu        | l#_CPPv4NK5cudaq10product_op14con |
| ages/python_api.html#cudaq.operat | st_iteratoreqERK14const_iterator) |
| ors.fermion.FermionOperator.copy) | -   [cudaq::product\_op::degrees  |
|     -   [(cudaq.oper              |     (C++                          |
| ators.fermion.FermionOperatorTerm |     function)                     |
|         method)](api/languages    | ](api/languages/cpp_api.html#_CPP |
| /python_api.html#cudaq.operators. | v4NK5cudaq10product_op7degreesEv) |
| fermion.FermionOperatorTerm.copy) | -   [cudaq::product\_op::dump     |
|     -                             |     (C++                          |
|  [(cudaq.operators.MatrixOperator |     functi                        |
|         method)](                 | on)](api/languages/cpp_api.html#_ |
| api/languages/python_api.html#cud | CPPv4NK5cudaq10product_op4dumpEv) |
| aq.operators.MatrixOperator.copy) | -   [cudaq::product\_op::end (C++ |
|     -   [(c                       |     funct                         |
| udaq.operators.MatrixOperatorTerm | ion)](api/languages/cpp_api.html# |
|         method)](api/             | _CPPv4NK5cudaq10product_op3endEv) |
| languages/python_api.html#cudaq.o | -   [cud                          |
| perators.MatrixOperatorTerm.copy) | aq::product\_op::get\_coefficient |
|     -   [(                        |     (C++                          |
| cudaq.operators.spin.SpinOperator |     function)](api/lan            |
|         method)](api              | guages/cpp_api.html#_CPPv4NK5cuda |
| /languages/python_api.html#cudaq. | q10product_op15get_coefficientEv) |
| operators.spin.SpinOperator.copy) | -   [                             |
|     -   [(cuda                    | cudaq::product\_op::get\_term\_id |
| q.operators.spin.SpinOperatorTerm |     (C++                          |
|         method)](api/lan          |     function)](api                |
| guages/python_api.html#cudaq.oper | /languages/cpp_api.html#_CPPv4NK5 |
| ators.spin.SpinOperatorTerm.copy) | cudaq10product_op11get_term_idEv) |
| -   [count() (cudaq.SampleResult  | -                                 |
|                                   | [cudaq::product\_op::is\_identity |
|   method)](api/languages/python_a |     (C++                          |
| pi.html#cudaq.SampleResult.count) |     function)](api                |
| -   [counts()                     | /languages/cpp_api.html#_CPPv4NK5 |
|     (cudaq.ObserveResult          | cudaq10product_op11is_identityEv) |
|                                   | -   [cudaq::product\_op::num\_ops |
| method)](api/languages/python_api |     (C++                          |
| .html#cudaq.ObserveResult.counts) |     function)                     |
| -   [create() (in module          | ](api/languages/cpp_api.html#_CPP |
|                                   | v4NK5cudaq10product_op7num_opsEv) |
|    cudaq.boson)](api/languages/py | -                                 |
| thon_api.html#cudaq.boson.create) |   [cudaq::product\_op::operator\* |
|     -   [(in module               |     (C++                          |
|         c                         |     function)](api/languages/     |
| udaq.fermion)](api/languages/pyth | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| on_api.html#cudaq.fermion.create) | oduct_opmlE10product_opI1TERK15sc |
| -   [csr\_spmatrix (C++           | alar_operatorRK10product_opI1TE), |
|     type)](api/languages/c        |     [\[1\]](api/languages/        |
| pp_api.html#_CPPv412csr_spmatrix) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   cudaq                         | oduct_opmlE10product_opI1TERK15sc |
|     -   [module](api/langua       | alar_operatorRR10product_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[2\]](api/languages/        |
| -   [cudaq (C++                   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     type)](api/lan                | oduct_opmlE10product_opI1TERR15sc |
| guages/cpp_api.html#_CPPv45cudaq) | alar_operatorRK10product_opI1TE), |
| -   [cudaq.apply\_noise() (in     |     [\[3\]](api/languages/        |
|     module                        | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     cudaq)](api/languages/python_ | oduct_opmlE10product_opI1TERR15sc |
| api.html#cudaq.cudaq.apply_noise) | alar_operatorRR10product_opI1TE), |
| -   cudaq.boson                   |     [\[4\]](api/                  |
|     -   [module](api/languages/py | languages/cpp_api.html#_CPPv4I0EN |
| thon_api.html#module-cudaq.boson) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq.fermion                 | K15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[5\]](api/                  |
|   -   [module](api/languages/pyth | languages/cpp_api.html#_CPPv4I0EN |
| on_api.html#module-cudaq.fermion) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq.operators.custom        | K15scalar_operatorRR6sum_opI1TE), |
|     -   [mo                       |     [\[6\]](api/                  |
| dule](api/languages/python_api.ht | languages/cpp_api.html#_CPPv4I0EN |
| ml#module-cudaq.operators.custom) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq.spin                    | R15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/languages/p  |     [\[7\]](api/                  |
| ython_api.html#module-cudaq.spin) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::amplitude\_damping    | 5cudaq10product_opmlE6sum_opI1TER |
|     (C++                          | R15scalar_operatorRR6sum_opI1TE), |
|     cla                           |     [\[8\]](api/languages         |
| ss)](api/languages/cpp_api.html#_ | /cpp_api.html#_CPPv4NK5cudaq10pro |
| CPPv4N5cudaq17amplitude_dampingE) | duct_opmlERK6sum_opI9HandlerTyE), |
| -   [c                            |     [\[9\]](api/languages/cpp_a   |
| udaq::amplitude\_damping\_channel | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmlERK10product_opI9HandlerTyE), |
|     class)](api                   |     [\[10\]](api/language         |
| /languages/cpp_api.html#_CPPv4N5c | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| udaq25amplitude_damping_channelE) | roduct_opmlERK15scalar_operator), |
| -   [cudaq::amplitude\_           |     [\[11\]](api/languages/cpp_a  |
| damping\_channel::num\_parameters | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmlERR10product_opI9HandlerTyE), |
|     member)](api/languages/cpp_a  |     [\[12\]](api/language         |
| pi.html#_CPPv4N5cudaq25amplitude_ | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| damping_channel14num_parametersE) | roduct_opmlERR15scalar_operator), |
| -   [cudaq::amplitud              |     [\[13\]](api/languages/cpp_   |
| e\_damping\_channel::num\_targets | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmlERK10product_opI9HandlerTyE), |
|     member)](api/languages/cp     |     [\[14\]](api/languag          |
| p_api.html#_CPPv4N5cudaq25amplitu | es/cpp_api.html#_CPPv4NO5cudaq10p |
| de_damping_channel11num_targetsE) | roduct_opmlERK15scalar_operator), |
| -   [cudaq::AnalogRemoteRESTQPU   |     [\[15\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     class                         | opmlERR10product_opI9HandlerTyE), |
| )](api/languages/cpp_api.html#_CP |     [\[16\]](api/langua           |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| -   [cudaq::apply\_noise (C++     | product_opmlERR15scalar_operator) |
|     function)](api/               | -                                 |
| languages/cpp_api.html#_CPPv4I0Dp |  [cudaq::product\_op::operator\*= |
| EN5cudaq11apply_noiseEvDpRR4Args) |     (C++                          |
| -   [cudaq::async\_result (C++    |     function)](api/languages/cpp  |
|     c                             | _api.html#_CPPv4N5cudaq10product_ |
| lass)](api/languages/cpp_api.html | opmLERK10product_opI9HandlerTyE), |
| #_CPPv4I0EN5cudaq12async_resultE) |     [\[1\]](api/langua            |
| -   [cudaq::async\_result::get    | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     (C++                          | roduct_opmLERK15scalar_operator), |
|     functi                        |     [\[2\]](api/languages/cp      |
| on)](api/languages/cpp_api.html#_ | p_api.html#_CPPv4N5cudaq10product |
| CPPv4N5cudaq12async_result3getEv) | _opmLERR10product_opI9HandlerTyE) |
| -   [cudaq::async\_sample\_result | -                                 |
|     (C++                          |    [cudaq::product\_op::operator+ |
|     type                          |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/langu          |
| Pv4N5cudaq19async_sample_resultE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::BaseNvcfSimulatorQPU  | q10product_opplE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRK10product_opI1TE), |
|     class)                        |     [\[1\]](api/                  |
| ](api/languages/cpp_api.html#_CPP | languages/cpp_api.html#_CPPv4I0EN |
| v4N5cudaq20BaseNvcfSimulatorQPUE) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::BaseRemoteRESTQPU     | K15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[2\]](api/langu             |
|     cla                           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ss)](api/languages/cpp_api.html#_ | q10product_opplE6sum_opI1TERK15sc |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | alar_operatorRR10product_opI1TE), |
| -                                 |     [\[3\]](api/                  |
|    [cudaq::BaseRemoteSimulatorQPU | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     class)](                      | K15scalar_operatorRR6sum_opI1TE), |
| api/languages/cpp_api.html#_CPPv4 |     [\[4\]](api/langu             |
| N5cudaq22BaseRemoteSimulatorQPUE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::bit\_flip\_channel    | q10product_opplE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRK10product_opI1TE), |
|     cl                            |     [\[5\]](api/                  |
| ass)](api/languages/cpp_api.html# | languages/cpp_api.html#_CPPv4I0EN |
| _CPPv4N5cudaq16bit_flip_channelE) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::bi                    | R15scalar_operatorRK6sum_opI1TE), |
| t\_flip\_channel::num\_parameters |     [\[6\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     member)](api/langua           | q10product_opplE6sum_opI1TERR15sc |
| ges/cpp_api.html#_CPPv4N5cudaq16b | alar_operatorRR10product_opI1TE), |
| it_flip_channel14num_parametersE) |     [\[7\]](api/                  |
| -   [cudaq:                       | languages/cpp_api.html#_CPPv4I0EN |
| :bit\_flip\_channel::num\_targets | 5cudaq10product_opplE6sum_opI1TER |
|     (C++                          | R15scalar_operatorRR6sum_opI1TE), |
|     member)](api/lan              |     [\[8\]](api/languages/cpp_a   |
| guages/cpp_api.html#_CPPv4N5cudaq | pi.html#_CPPv4NKR5cudaq10product_ |
| 16bit_flip_channel11num_targetsE) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq::boson\_handler (C++   |     [\[9\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|  class)](api/languages/cpp_api.ht | roduct_opplERK15scalar_operator), |
| ml#_CPPv4N5cudaq13boson_handlerE) |     [\[10\]](api/languages/       |
| -   [cudaq::boson\_op (C++        | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     type)](api/languages/cpp_     | duct_opplERK6sum_opI9HandlerTyE), |
| api.html#_CPPv4N5cudaq8boson_opE) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq::boson\_op\_term (C++  | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|   type)](api/languages/cpp_api.ht |     [\[12\]](api/language         |
| ml#_CPPv4N5cudaq13boson_op_termE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -                                 | roduct_opplERR15scalar_operator), |
|    [cudaq::commutation\_relations |     [\[13\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     struct)]                      | duct_opplERR6sum_opI9HandlerTyE), |
| (api/languages/cpp_api.html#_CPPv |     [\[                           |
| 4N5cudaq21commutation_relationsE) | 14\]](api/languages/cpp_api.html# |
| -   [cudaq::complex (C++          | _CPPv4NKR5cudaq10product_opplEv), |
|     type)](api/languages/cpp      |     [\[15\]](api/languages/cpp_   |
| _api.html#_CPPv4N5cudaq7complexE) | api.html#_CPPv4NO5cudaq10product_ |
| -   [cudaq::complex\_matrix (C++  | opplERK10product_opI9HandlerTyE), |
|                                   |     [\[16\]](api/languag          |
| class)](api/languages/cpp_api.htm | es/cpp_api.html#_CPPv4NO5cudaq10p |
| l#_CPPv4N5cudaq14complex_matrixE) | roduct_opplERK15scalar_operator), |
| -                                 |     [\[17\]](api/languages        |
|  [cudaq::complex\_matrix::adjoint | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opplERK6sum_opI9HandlerTyE), |
|     function)](a                  |     [\[18\]](api/languages/cpp_   |
| pi/languages/cpp_api.html#_CPPv4N | api.html#_CPPv4NO5cudaq10product_ |
| 5cudaq14complex_matrix7adjointEv) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::co                    |     [\[19\]](api/languag          |
| mplex\_matrix::diagonal\_elements | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opplERR15scalar_operator), |
|     function)](api/languages      |     [\[20\]](api/languages        |
| /cpp_api.html#_CPPv4NK5cudaq14com | /cpp_api.html#_CPPv4NO5cudaq10pro |
| plex_matrix17diagonal_elementsEi) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cudaq::complex\_matrix::dump |     [                             |
|     (C++                          | \[21\]](api/languages/cpp_api.htm |
|     function)](api/language       | l#_CPPv4NO5cudaq10product_opplEv) |
| s/cpp_api.html#_CPPv4NK5cudaq14co | -                                 |
| mplex_matrix4dumpERNSt7ostreamE), |    [cudaq::product\_op::operator- |
|     [\[1\]]                       |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/langu          |
| 4NK5cudaq14complex_matrix4dumpEv) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cu                           | q10product_opmiE6sum_opI1TERK15sc |
| daq::complex\_matrix::eigenvalues | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     function)](api/lan            | languages/cpp_api.html#_CPPv4I0EN |
| guages/cpp_api.html#_CPPv4NK5cuda | 5cudaq10product_opmiE6sum_opI1TER |
| q14complex_matrix11eigenvaluesEv) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cud                          |     [\[2\]](api/langu             |
| aq::complex\_matrix::eigenvectors | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERK15sc |
|     function)](api/lang           | alar_operatorRR10product_opI1TE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[3\]](api/                  |
| 14complex_matrix12eigenvectorsEv) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cu                           | 5cudaq10product_opmiE6sum_opI1TER |
| daq::complex\_matrix::exponential | K15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[4\]](api/langu             |
|     function)](api/la             | ages/cpp_api.html#_CPPv4I0EN5cuda |
| nguages/cpp_api.html#_CPPv4N5cuda | q10product_opmiE6sum_opI1TERR15sc |
| q14complex_matrix11exponentialEv) | alar_operatorRK10product_opI1TE), |
| -                                 |     [\[5\]](api/                  |
| [cudaq::complex\_matrix::identity | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/languages      | R15scalar_operatorRK6sum_opI1TE), |
| /cpp_api.html#_CPPv4N5cudaq14comp |     [\[6\]](api/langu             |
| lex_matrix8identityEKNSt6size_tE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [                             | q10product_opmiE6sum_opI1TERR15sc |
| cudaq::complex\_matrix::kronecker | alar_operatorRR10product_opI1TE), |
|     (C++                          |     [\[7\]](api/                  |
|     function)](api/lang           | languages/cpp_api.html#_CPPv4I0EN |
| uages/cpp_api.html#_CPPv4I00EN5cu | 5cudaq10product_opmiE6sum_opI1TER |
| daq14complex_matrix9kroneckerE14c | R15scalar_operatorRR6sum_opI1TE), |
| omplex_matrix8Iterable8Iterable), |     [\[8\]](api/languages/cpp_a   |
|     [\[1\]](api/l                 | pi.html#_CPPv4NKR5cudaq10product_ |
| anguages/cpp_api.html#_CPPv4N5cud | opmiERK10product_opI9HandlerTyE), |
| aq14complex_matrix9kroneckerERK14 |     [\[9\]](api/language          |
| complex_matrixRK14complex_matrix) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::com                   | roduct_opmiERK15scalar_operator), |
| plex\_matrix::minimal\_eigenvalue |     [\[10\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     function)](api/languages/     | duct_opmiERK6sum_opI9HandlerTyE), |
| cpp_api.html#_CPPv4NK5cudaq14comp |     [\[11\]](api/languages/cpp_a  |
| lex_matrix18minimal_eigenvalueEv) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [c                            | opmiERR10product_opI9HandlerTyE), |
| udaq::complex\_matrix::operator() |     [\[12\]](api/language         |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/languages/cpp  | roduct_opmiERR15scalar_operator), |
| _api.html#_CPPv4N5cudaq14complex_ |     [\[13\]](api/languages/       |
| matrixclENSt6size_tENSt6size_tE), | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     [\[1\]](api/languages/cpp     | duct_opmiERR6sum_opI9HandlerTyE), |
| _api.html#_CPPv4NK5cudaq14complex |     [\[                           |
| _matrixclENSt6size_tENSt6size_tE) | 14\]](api/languages/cpp_api.html# |
| -   [c                            | _CPPv4NKR5cudaq10product_opmiEv), |
| udaq::complex\_matrix::operator\* |     [\[15\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     function)](api/langua         | opmiERK10product_opI9HandlerTyE), |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     [\[16\]](api/languag          |
| omplex_matrixmlEN14complex_matrix | es/cpp_api.html#_CPPv4NO5cudaq10p |
| 10value_typeERK14complex_matrix), | roduct_opmiERK15scalar_operator), |
|     [\[1\]                        |     [\[17\]](api/languages        |
| ](api/languages/cpp_api.html#_CPP | /cpp_api.html#_CPPv4NO5cudaq10pro |
| v4N5cudaq14complex_matrixmlERK14c | duct_opmiERK6sum_opI9HandlerTyE), |
| omplex_matrixRK14complex_matrix), |     [\[18\]](api/languages/cpp_   |
|                                   | api.html#_CPPv4NO5cudaq10product_ |
|  [\[2\]](api/languages/cpp_api.ht | opmiERR10product_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq14complex_matrixm |     [\[19\]](api/languag          |
| lERK14complex_matrixRKNSt6vectorI | es/cpp_api.html#_CPPv4NO5cudaq10p |
| N14complex_matrix10value_typeEEE) | roduct_opmiERR15scalar_operator), |
| -   [                             |     [\[20\]](api/languages        |
| cudaq::complex\_matrix::operator+ | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERR6sum_opI9HandlerTyE), |
|     function                      |     [                             |
| )](api/languages/cpp_api.html#_CP | \[21\]](api/languages/cpp_api.htm |
| Pv4N5cudaq14complex_matrixplERK14 | l#_CPPv4NO5cudaq10product_opmiEv) |
| complex_matrixRK14complex_matrix) | -                                 |
| -   [                             |    [cudaq::product\_op::operator/ |
| cudaq::complex\_matrix::operator- |     (C++                          |
|     (C++                          |     function)](api/language       |
|     function                      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| )](api/languages/cpp_api.html#_CP | roduct_opdvERK15scalar_operator), |
| Pv4N5cudaq14complex_matrixmiERK14 |     [\[1\]](api/language          |
| complex_matrixRK14complex_matrix) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cud                          | roduct_opdvERR15scalar_operator), |
| aq::complex\_matrix::operator\[\] |     [\[2\]](api/languag           |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|                                   | roduct_opdvERK15scalar_operator), |
|  function)](api/languages/cpp_api |     [\[3\]](api/langua            |
| .html#_CPPv4N5cudaq14complex_matr | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| ixixERKNSt6vectorINSt6size_tEEE), | product_opdvERR15scalar_operator) |
|     [\[1\]](api/languages/cpp_api | -                                 |
| .html#_CPPv4NK5cudaq14complex_mat |   [cudaq::product\_op::operator/= |
| rixixERKNSt6vectorINSt6size_tEEE) |     (C++                          |
| -                                 |     function)](api/langu          |
|    [cudaq::complex\_matrix::power | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     (C++                          | product_opdVERK15scalar_operator) |
|     function)]                    | -                                 |
| (api/languages/cpp_api.html#_CPPv |    [cudaq::product\_op::operator= |
| 4N5cudaq14complex_matrix5powerEi) |     (C++                          |
| -   [                             |     function)](api/la             |
| cudaq::complex\_matrix::set\_zero | nguages/cpp_api.html#_CPPv4I0_NSt |
|     (C++                          | 11enable_if_tIXaantNSt7is_sameI1T |
|     function)](ap                 | 9HandlerTyE5valueENSt16is_constru |
| i/languages/cpp_api.html#_CPPv4N5 | ctibleI9HandlerTy1TE5valueEEbEEEN |
| cudaq14complex_matrix8set_zeroEv) | 5cudaq10product_opaSER10product_o |
| -   [c                            | pI9HandlerTyERK10product_opI1TE), |
| udaq::complex\_matrix::to\_string |     [\[1\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4N5cudaq10product_ |
|     function)](api/               | opaSERK10product_opI9HandlerTyE), |
| languages/cpp_api.html#_CPPv4NK5c |     [\[2\]](api/languages/cp      |
| udaq14complex_matrix9to_stringEv) | p_api.html#_CPPv4N5cudaq10product |
| -   [cu                           | _opaSERR10product_opI9HandlerTyE) |
| daq::complex\_matrix::value\_type | -                                 |
|     (C++                          |   [cudaq::product\_op::operator== |
|     type)](api/                   |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/languages/cpp  |
| daq14complex_matrix10value_typeE) | _api.html#_CPPv4NK5cudaq10product |
| -   [cudaq::contrib (C++          | _opeqERK10product_opI9HandlerTyE) |
|     type)](api/languages/cpp      | -                                 |
| _api.html#_CPPv4N5cudaq7contribE) | [cudaq::product\_op::operator\[\] |
| -   [cudaq::contrib::draw (C++    |     (C++                          |
|     function)]                    |     function)](ap                 |
| (api/languages/cpp_api.html#_CPPv | i/languages/cpp_api.html#_CPPv4NK |
| 4I0Dp0EN5cudaq7contrib4drawENSt6s | 5cudaq10product_opixENSt6size_tE) |
| tringERR13QuantumKernelDpRR4Args) | -                                 |
| -   [c                            |  [cudaq::product\_op::product\_op |
| udaq::contrib::get\_unitary\_cmat |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/languages/cp   | pp_api.html#_CPPv4I0_NSt11enable_ |
| p_api.html#_CPPv4I0DpEN5cudaq7con | if_tIXaaNSt7is_sameI9HandlerTy14m |
| trib16get_unitary_cmatE14complex_ | atrix_handlerE5valueEaantNSt7is_s |
| matrixRR13QuantumKernelDpRR4Args) | ameI1T9HandlerTyE5valueENSt16is_c |
| -   [cudaq::CusvState (C++        | onstructibleI9HandlerTy1TE5valueE |
|                                   | EbEEEN5cudaq10product_op10product |
|    class)](api/languages/cpp_api. | _opERK10product_opI1TERKN14matrix |
| html#_CPPv4I0EN5cudaq9CusvStateE) | _handler20commutation_behaviorE), |
| -   [cudaq::depolarization1 (C++  |                                   |
|     c                             |  [\[1\]](api/languages/cpp_api.ht |
| lass)](api/languages/cpp_api.html | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| #_CPPv4N5cudaq15depolarization1E) | tNSt7is_sameI1T9HandlerTyE5valueE |
| -   [cudaq::depolarization2 (C++  | NSt16is_constructibleI9HandlerTy1 |
|     c                             | TE5valueEEbEEEN5cudaq10product_op |
| lass)](api/languages/cpp_api.html | 10product_opERK10product_opI1TE), |
| #_CPPv4N5cudaq15depolarization2E) |                                   |
| -   [cudaq:                       |   [\[2\]](api/languages/cpp_api.h |
| :depolarization2::depolarization2 | tml#_CPPv4N5cudaq10product_op10pr |
|     (C++                          | oduct_opENSt6size_tENSt6size_tE), |
|     function)](api/languages/cp   |     [\[3\]](api/languages/cp      |
| p_api.html#_CPPv4N5cudaq15depolar | p_api.html#_CPPv4N5cudaq10product |
| ization215depolarization2EK4real) | _op10product_opENSt7complexIdEE), |
| -   [cudaq:                       |     [\[4\]](api/l                 |
| :depolarization2::num\_parameters | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq10product_op10product_opERK10pr |
|     member)](api/langu            | oduct_opI9HandlerTyENSt6size_tE), |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     [\[5\]](api/l                 |
| depolarization214num_parametersE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cud                          | aq10product_op10product_opERR10pr |
| aq::depolarization2::num\_targets | oduct_opI9HandlerTyENSt6size_tE), |
|     (C++                          |     [\[6\]](api/languages         |
|     member)](api/la               | /cpp_api.html#_CPPv4N5cudaq10prod |
| nguages/cpp_api.html#_CPPv4N5cuda | uct_op10product_opERR9HandlerTy), |
| q15depolarization211num_targetsE) |     [\[7\]](ap                    |
| -                                 | i/languages/cpp_api.html#_CPPv4N5 |
|   [cudaq::depolarization\_channel | cudaq10product_op10product_opEd), |
|     (C++                          |     [\[8\]](a                     |
|     class)](                      | pi/languages/cpp_api.html#_CPPv4N |
| api/languages/cpp_api.html#_CPPv4 | 5cudaq10product_op10product_opEv) |
| N5cudaq22depolarization_channelE) | -   [cudaq::                      |
| -   [cudaq::depolar               | product\_op::to\_diagonal\_matrix |
| ization\_channel::num\_parameters |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](api/languages/cp     | languages/cpp_api.html#_CPPv4NK5c |
| p_api.html#_CPPv4N5cudaq22depolar | udaq10product_op18to_diagonal_mat |
| ization_channel14num_parametersE) | rixENSt13unordered_mapINSt6size_t |
| -   [cudaq::depo                  | ENSt7int64_tEEERKNSt13unordered_m |
| larization\_channel::num\_targets | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -                                 |
|     member)](api/languages        |   [cudaq::product\_op::to\_matrix |
| /cpp_api.html#_CPPv4N5cudaq22depo |     (C++                          |
| larization_channel11num_targetsE) |     funct                         |
| -   [cudaq::details (C++          | ion)](api/languages/cpp_api.html# |
|     type)](api/languages/cpp      | _CPPv4NK5cudaq10product_op9to_mat |
| _api.html#_CPPv4N5cudaq7detailsE) | rixENSt13unordered_mapINSt6size_t |
| -   [cudaq::details::future (C++  | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|  class)](api/languages/cpp_api.ht | -   [cudaq                        |
| ml#_CPPv4N5cudaq7details6futureE) | ::product\_op::to\_sparse\_matrix |
| -                                 |     (C++                          |
|   [cudaq::details::future::future |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     functio                       | 5cudaq10product_op16to_sparse_mat |
| n)](api/languages/cpp_api.html#_C | rixENSt13unordered_mapINSt6size_t |
| PPv4N5cudaq7details6future6future | ENSt7int64_tEEERKNSt13unordered_m |
| ERNSt6vectorI3JobEERNSt6stringERN | apINSt6stringENSt7complexIdEEEEb) |
| St3mapINSt6stringENSt6stringEEE), | -                                 |
|     [\[1\]](api/lang              |   [cudaq::product\_op::to\_string |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     (C++                          |
| details6future6futureERR6future), |     function)](                   |
|     [\[2\]]                       | api/languages/cpp_api.html#_CPPv4 |
| (api/languages/cpp_api.html#_CPPv | NK5cudaq10product_op9to_stringEv) |
| 4N5cudaq7details6future6futureEv) | -   [                             |
| -   [cuda                         | cudaq::product\_op::\~product\_op |
| q::details::kernel\_builder\_base |     (C++                          |
|     (C++                          |     fu                            |
|     class)](api/l                 | nction)](api/languages/cpp_api.ht |
| anguages/cpp_api.html#_CPPv4N5cud | ml#_CPPv4N5cudaq10product_opD0Ev) |
| aq7details19kernel_builder_baseE) | -   [cudaq::QPU (C++              |
| -   [cudaq::details::ke           |     class)](api/languages         |
| rnel\_builder\_base::operator\<\< | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     (C++                          | -   [cudaq::QPU::enqueue (C++     |
|     function)](api/langua         |     function)](ap                 |
| ges/cpp_api.html#_CPPv4N5cudaq7de | i/languages/cpp_api.html#_CPPv4N5 |
| tails19kernel_builder_baselsERNSt | cudaq3QPU7enqueueER11QuantumTask) |
| 7ostreamERK19kernel_builder_base) | -   [cudaq::QPU::getConnectivity  |
| -   [                             |     (C++                          |
| cudaq::details::KernelBuilderType |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     class)](api                   | v4N5cudaq3QPU15getConnectivityEv) |
| /languages/cpp_api.html#_CPPv4N5c | -                                 |
| udaq7details17KernelBuilderTypeE) | [cudaq::QPU::getExecutionThreadId |
| -   [cudaq::d                     |     (C++                          |
| etails::KernelBuilderType::create |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4NK5c |
|     function)                     | udaq3QPU20getExecutionThreadIdEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::QPU::getNumQubits     |
| v4N5cudaq7details17KernelBuilderT |     (C++                          |
| ype6createEPN4mlir11MLIRContextE) |     functi                        |
| -   [cudaq::details::Ker          | on)](api/languages/cpp_api.html#_ |
| nelBuilderType::KernelBuilderType | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     (C++                          | -   [                             |
|     function)](api/lang           | cudaq::QPU::getRemoteCapabilities |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     (C++                          |
| details17KernelBuilderType17Kerne |     function)](api/l              |
| lBuilderTypeERRNSt8functionIFN4ml | anguages/cpp_api.html#_CPPv4NK5cu |
| ir4TypeEPN4mlir11MLIRContextEEEE) | daq3QPU21getRemoteCapabilitiesEv) |
| -                                 | -   [cudaq::QPU::isEmulated (C++  |
|    [cudaq::diag\_matrix\_callback |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     class)                        | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::QPU::isSimulator (C++ |
| v4N5cudaq20diag_matrix_callbackE) |     funct                         |
| -   [cudaq::dyn (C++              | ion)](api/languages/cpp_api.html# |
|     member)](api/languages        | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | -   [cudaq::QPU::launchKernel     |
| -   [cudaq::ExecutionContext (C++ |     (C++                          |
|     cl                            |     function)](api/               |
| ass)](api/languages/cpp_api.html# | languages/cpp_api.html#_CPPv4N5cu |
| _CPPv4N5cudaq16ExecutionContextE) | daq3QPU12launchKernelERKNSt6strin |
| -   [cudaq                        | gE15KernelThunkTypePvNSt8uint64_t |
| ::ExecutionContext::amplitudeMaps | ENSt8uint64_tERKNSt6vectorIPvEE), |
|     (C++                          |                                   |
|     member)](api/langu            |  [\[1\]](api/languages/cpp_api.ht |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ml#_CPPv4N5cudaq3QPU12launchKerne |
| ExecutionContext13amplitudeMapsE) | lERKNSt6stringERKNSt6vectorIPvEE) |
| -   [c                            | -   [cudaq::Q                     |
| udaq::ExecutionContext::asyncExec | PU::launchSerializedCodeExecution |
|     (C++                          |     (C++                          |
|     member)](api/                 |     function)]                    |
| languages/cpp_api.html#_CPPv4N5cu | (api/languages/cpp_api.html#_CPPv |
| daq16ExecutionContext9asyncExecE) | 4N5cudaq3QPU29launchSerializedCod |
| -   [cud                          | eExecutionERKNSt6stringERN5cudaq3 |
| aq::ExecutionContext::asyncResult | 0SerializedCodeExecutionContextE) |
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
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [                             |
| cutionContext16canHandleObserveE) | cudaq::QPU::resetExecutionContext |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::ExecutionContext |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     func                          | daq3QPU21resetExecutionContextEv) |
| tion)](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq16ExecutionContext1 |  [cudaq::QPU::setExecutionContext |
| 6ExecutionContextERKNSt6stringE), |     (C++                          |
|     [\[1\]](api                   |                                   |
| /languages/cpp_api.html#_CPPv4N5c |   function)](api/languages/cpp_ap |
| udaq16ExecutionContext16Execution | i.html#_CPPv4N5cudaq3QPU19setExec |
| ContextERKNSt6stringENSt6size_tE) | utionContextEP16ExecutionContext) |
| -   [cudaq::E                     | -   [cudaq::QPU::setId (C++       |
| xecutionContext::expectationValue |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)](api/language         | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::QPU::setShots (C++    |
| cutionContext16expectationValueE) |     f                             |
| -   [cudaq::Execu                 | unction)](api/languages/cpp_api.h |
| tionContext::explicitMeasurements | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
|     (C++                          | -   [cudaq:                       |
|     member)](api/languages/cp     | :QPU::supportsConditionalFeedback |
| p_api.html#_CPPv4N5cudaq16Executi |     (C++                          |
| onContext20explicitMeasurementsE) |     function)](api/langua         |
| -   [cuda                         | ges/cpp_api.html#_CPPv4N5cudaq3QP |
| q::ExecutionContext::futureResult | U27supportsConditionalFeedbackEv) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/lang             | QPU::supportsExplicitMeasurements |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     (C++                          |
| 6ExecutionContext12futureResultE) |     function)](api/languag        |
| -   [cudaq::ExecutionContext      | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| ::hasConditionalsOnMeasureResults | 28supportsExplicitMeasurementsEv) |
|     (C++                          | -   [cudaq::QPU::\~QPU (C++       |
|     mem                           |     function)](api/languages/cp   |
| ber)](api/languages/cpp_api.html# | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| _CPPv4N5cudaq16ExecutionContext31 | -   [cudaq::QPUState (C++         |
| hasConditionalsOnMeasureResultsE) |     class)](api/languages/cpp_    |
| -   [cudaq::Executi               | api.html#_CPPv4N5cudaq8QPUStateE) |
| onContext::invocationResultBuffer | -   [cudaq::qreg (C++             |
|     (C++                          |     class)](api/lang              |
|     member)](api/languages/cpp_   | uages/cpp_api.html#_CPPv4I_NSt6si |
| api.html#_CPPv4N5cudaq16Execution | ze_tE_NSt6size_tE0EN5cudaq4qregE) |
| Context22invocationResultBufferE) | -   [cudaq::qreg::back (C++       |
| -   [cu                           |     function)                     |
| daq::ExecutionContext::kernelName | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4backENSt6size_tE), |
|     member)](api/la               |     [\[1\]](api/languages/cpp_ap  |
| nguages/cpp_api.html#_CPPv4N5cuda | i.html#_CPPv4N5cudaq4qreg4backEv) |
| q16ExecutionContext10kernelNameE) | -   [cudaq::qreg::begin (C++      |
| -   [cud                          |                                   |
| aq::ExecutionContext::kernelTrace |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     member)](api/lan              | -   [cudaq::qreg::clear (C++      |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 16ExecutionContext11kernelTraceE) |  function)](api/languages/cpp_api |
| -   [cudaq::                      | .html#_CPPv4N5cudaq4qreg5clearEv) |
| ExecutionContext::msm\_dimensions | -   [cudaq::qreg::front (C++      |
|     (C++                          |     function)]                    |
|     member)](api/langua           | (api/languages/cpp_api.html#_CPPv |
| ges/cpp_api.html#_CPPv4N5cudaq16E | 4N5cudaq4qreg5frontENSt6size_tE), |
| xecutionContext14msm_dimensionsE) |     [\[1\]](api/languages/cpp_api |
| -   [cudaq::Exe                   | .html#_CPPv4N5cudaq4qreg5frontEv) |
| cutionContext::msm\_prob\_err\_id | -   [cudaq::qreg::operator\[\]    |
|     (C++                          |     (C++                          |
|     member)](api/languag          |     functi                        |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | on)](api/languages/cpp_api.html#_ |
| ecutionContext15msm_prob_err_idE) | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| -   [cudaq::Exe                   | -   [cudaq::qreg::size (C++       |
| cutionContext::msm\_probabilities |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     member)](api/languages        | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| /cpp_api.html#_CPPv4N5cudaq16Exec | -   [cudaq::qreg::slice (C++      |
| utionContext17msm_probabilitiesE) |     function)](api/langu          |
| -                                 | ages/cpp_api.html#_CPPv4N5cudaq4q |
|    [cudaq::ExecutionContext::name | reg5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qreg::value\_type     |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |                                   |
| 4N5cudaq16ExecutionContext4nameE) | type)](api/languages/cpp_api.html |
| -   [cu                           | #_CPPv4N5cudaq4qreg10value_typeE) |
| daq::ExecutionContext::noiseModel | -   [cudaq::qspan (C++            |
|     (C++                          |     class)](api/lang              |
|     member)](api/la               | uages/cpp_api.html#_CPPv4I_NSt6si |
| nguages/cpp_api.html#_CPPv4N5cuda | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| q16ExecutionContext10noiseModelE) | -   [cudaq::QuakeValue (C++       |
| -   [cudaq::Exe                   |     class)](api/languages/cpp_api |
| cutionContext::numberTrajectories | .html#_CPPv4N5cudaq10QuakeValueE) |
|     (C++                          | -   [cudaq::Q                     |
|     member)](api/languages/       | uakeValue::canValidateNumElements |
| cpp_api.html#_CPPv4N5cudaq16Execu |     (C++                          |
| tionContext18numberTrajectoriesE) |     function)](api/languages      |
| -   [c                            | /cpp_api.html#_CPPv4N5cudaq10Quak |
| udaq::ExecutionContext::optResult | eValue22canValidateNumElementsEv) |
|     (C++                          | -                                 |
|     member)](api/                 |  [cudaq::QuakeValue::constantSize |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9optResultE) |     function)](api                |
| -   [cudaq::Execu                 | /languages/cpp_api.html#_CPPv4N5c |
| tionContext::overlapComputeStates | udaq10QuakeValue12constantSizeEv) |
|     (C++                          | -   [cudaq::QuakeValue::dump (C++ |
|     member)](api/languages/cp     |     function)](api/lan            |
| p_api.html#_CPPv4N5cudaq16Executi | guages/cpp_api.html#_CPPv4N5cudaq |
| onContext20overlapComputeStatesE) | 10QuakeValue4dumpERNSt7ostreamE), |
| -   [cudaq                        |     [\                            |
| ::ExecutionContext::overlapResult | [1\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq10QuakeValue4dumpEv) |
|     member)](api/langu            | -   [cudaq                        |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ::QuakeValue::getRequiredElements |
| ExecutionContext13overlapResultE) |     (C++                          |
| -   [cudaq                        |     function)](api/langua         |
| ::ExecutionContext::registerNames | ges/cpp_api.html#_CPPv4N5cudaq10Q |
|     (C++                          | uakeValue19getRequiredElementsEv) |
|     member)](api/langu            | -   [cudaq::QuakeValue::getValue  |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13registerNamesE) |     function)]                    |
| -   [cu                           | (api/languages/cpp_api.html#_CPPv |
| daq::ExecutionContext::reorderIdx | 4NK5cudaq10QuakeValue8getValueEv) |
|     (C++                          | -   [cudaq::QuakeValue::inverse   |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)                     |
| q16ExecutionContext10reorderIdxE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4NK5cudaq10QuakeValue7inverseEv) |
|  [cudaq::ExecutionContext::result | -   [cudaq::QuakeValue::isStdVec  |
|     (C++                          |     (C++                          |
|     member)](a                    |     function)                     |
| pi/languages/cpp_api.html#_CPPv4N | ](api/languages/cpp_api.html#_CPP |
| 5cudaq16ExecutionContext6resultE) | v4N5cudaq10QuakeValue8isStdVecEv) |
| -                                 | -                                 |
|   [cudaq::ExecutionContext::shots |    [cudaq::QuakeValue::operator\* |
|     (C++                          |     (C++                          |
|     member)](                     |     function)](api                |
| api/languages/cpp_api.html#_CPPv4 | /languages/cpp_api.html#_CPPv4N5c |
| N5cudaq16ExecutionContext5shotsE) | udaq10QuakeValuemlE10QuakeValue), |
| -   [cudaq::                      |                                   |
| ExecutionContext::simulationState | [\[1\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|     member)](api/languag          | -   [cudaq::QuakeValue::operator+ |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15simulationStateE) |     function)](api                |
| -                                 | /languages/cpp_api.html#_CPPv4N5c |
|    [cudaq::ExecutionContext::spin | udaq10QuakeValueplE10QuakeValue), |
|     (C++                          |     [                             |
|     member)]                      | \[1\]](api/languages/cpp_api.html |
| (api/languages/cpp_api.html#_CPPv | #_CPPv4N5cudaq10QuakeValueplEKd), |
| 4N5cudaq16ExecutionContext4spinE) |                                   |
| -   [cudaq::                      | [\[2\]](api/languages/cpp_api.htm |
| ExecutionContext::totalIterations | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|     (C++                          | -   [cudaq::QuakeValue::operator- |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api                |
| ecutionContext15totalIterationsE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::ExecutionResult (C++  | udaq10QuakeValuemiE10QuakeValue), |
|     st                            |     [                             |
| ruct)](api/languages/cpp_api.html | \[1\]](api/languages/cpp_api.html |
| #_CPPv4N5cudaq15ExecutionResultE) | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| -   [cud                          |     [                             |
| aq::ExecutionResult::appendResult | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     functio                       |                                   |
| n)](api/languages/cpp_api.html#_C | [\[3\]](api/languages/cpp_api.htm |
| PPv4N5cudaq15ExecutionResult12app | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| endResultENSt6stringENSt6size_tE) | -   [cudaq::QuakeValue::operator/ |
| -   [cu                           |     (C++                          |
| daq::ExecutionResult::deserialize |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)                     | udaq10QuakeValuedvE10QuakeValue), |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4N5cudaq15ExecutionResult11deser | [\[1\]](api/languages/cpp_api.htm |
| ializeERNSt6vectorINSt6size_tEEE) | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| -   [cudaq:                       | -                                 |
| :ExecutionResult::ExecutionResult |  [cudaq::QuakeValue::operator\[\] |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api                |
| n)](api/languages/cpp_api.html#_C | /languages/cpp_api.html#_CPPv4N5c |
| PPv4N5cudaq15ExecutionResult15Exe | udaq10QuakeValueixEKNSt6size_tE), |
| cutionResultE16CountsDictionary), |     [\[1\]](api/                  |
|     [\[1\]](api/lan               | languages/cpp_api.html#_CPPv4N5cu |
| guages/cpp_api.html#_CPPv4N5cudaq | daq10QuakeValueixERK10QuakeValue) |
| 15ExecutionResult15ExecutionResul | -                                 |
| tE16CountsDictionaryNSt6stringE), |    [cudaq::QuakeValue::QuakeValue |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     function)](api/languag        |
| Pv4N5cudaq15ExecutionResult15Exec | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| utionResultE16CountsDictionaryd), | akeValue10QuakeValueERN4mlir20Imp |
|                                   | licitLocOpBuilderEN4mlir5ValueE), |
|    [\[3\]](api/languages/cpp_api. |     [\[1\]                        |
| html#_CPPv4N5cudaq15ExecutionResu | ](api/languages/cpp_api.html#_CPP |
| lt15ExecutionResultENSt6stringE), | v4N5cudaq10QuakeValue10QuakeValue |
|     [\[4\                         | ERN4mlir20ImplicitLocOpBuilderEd) |
| ]](api/languages/cpp_api.html#_CP | -   [cudaq::QuakeValue::size (C++ |
| Pv4N5cudaq15ExecutionResult15Exec |     funct                         |
| utionResultERK15ExecutionResult), | ion)](api/languages/cpp_api.html# |
|     [\[5\]](api/language          | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | -   [cudaq::QuakeValue::slice     |
| cutionResult15ExecutionResultEd), |     (C++                          |
|     [\[6\]](api/languag           |     function)](api/languages/cpp_ |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | api.html#_CPPv4N5cudaq10QuakeValu |
| ecutionResult15ExecutionResultEv) | e5sliceEKNSt6size_tEKNSt6size_tE) |
| -   [                             | -   [cudaq::quantum\_platform     |
| cudaq::ExecutionResult::operator= |     (C++                          |
|     (C++                          |     cl                            |
|     function)](api/languages/     | ass)](api/languages/cpp_api.html# |
| cpp_api.html#_CPPv4N5cudaq15Execu | _CPPv4N5cudaq16quantum_platformE) |
| tionResultaSERK15ExecutionResult) | -   [cudaq                        |
| -   [c                            | ::quantum\_platform::clear\_shots |
| udaq::ExecutionResult::operator== |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     function)](api/languages/c    | uages/cpp_api.html#_CPPv4N5cudaq1 |
| pp_api.html#_CPPv4NK5cudaq15Execu | 6quantum_platform11clear_shotsEv) |
| tionResulteqERK15ExecutionResult) | -   [cudaq                        |
| -   [cud                          | ::quantum\_platform::connectivity |
| aq::ExecutionResult::registerName |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     member)](api/lan              | ages/cpp_api.html#_CPPv4N5cudaq16 |
| guages/cpp_api.html#_CPPv4N5cudaq | quantum_platform12connectivityEv) |
| 15ExecutionResult12registerNameE) | -   [cudaq::qu                    |
| -   [cudaq                        | antum\_platform::enqueueAsyncTask |
| ::ExecutionResult::sequentialData |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     member)](api/langu            | cpp_api.html#_CPPv4N5cudaq16quant |
| ages/cpp_api.html#_CPPv4N5cudaq15 | um_platform16enqueueAsyncTaskEKNS |
| ExecutionResult14sequentialDataE) | t6size_tER19KernelExecutionTask), |
| -   [                             |     [\[1\]](api/languag           |
| cudaq::ExecutionResult::serialize | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform16enqueueAsyncTaskE |
|     function)](api/l              | KNSt6size_tERNSt8functionIFvvEEE) |
| anguages/cpp_api.html#_CPPv4NK5cu | -   [cudaq::qua                   |
| daq15ExecutionResult9serializeEv) | ntum\_platform::get\_current\_qpu |
| -   [cudaq::fermion\_handler (C++ |     (C++                          |
|     c                             |     function)](api/language       |
| lass)](api/languages/cpp_api.html | s/cpp_api.html#_CPPv4N5cudaq16qua |
| #_CPPv4N5cudaq15fermion_handlerE) | ntum_platform15get_current_qpuEv) |
| -   [cudaq::fermion\_op (C++      | -   [cudaq::                      |
|     type)](api/languages/cpp_api  | quantum\_platform::get\_exec\_ctx |
| .html#_CPPv4N5cudaq10fermion_opE) |     (C++                          |
| -   [cudaq::fermion\_op\_term     |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|                                   | quantum_platform12get_exec_ctxEv) |
| type)](api/languages/cpp_api.html | -   [cud                          |
| #_CPPv4N5cudaq15fermion_op_termE) | aq::quantum\_platform::get\_noise |
| -   [cudaq::FermioniqBaseQPU (C++ |     (C++                          |
|     cl                            |     function)](api/l              |
| ass)](api/languages/cpp_api.html# | anguages/cpp_api.html#_CPPv4N5cud |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | aq16quantum_platform9get_noiseEv) |
| -   [cudaq::get\_state (C++       | -   [cudaq::qu                    |
|                                   | antum\_platform::get\_num\_qubits |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |                                   |
| ateEDaRR13QuantumKernelDpRR4Args) | function)](api/languages/cpp_api. |
| -   [cudaq::gradient (C++         | html#_CPPv4N5cudaq16quantum_platf |
|     class)](api/languages/cpp_    | orm14get_num_qubitsENSt6size_tE), |
| api.html#_CPPv4N5cudaq8gradientE) |     [\[1\]](api/languag           |
| -   [cudaq::gradient::clone (C++  | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     fun                           | antum_platform14get_num_qubitsEv) |
| ction)](api/languages/cpp_api.htm | -   [cudaq::quantum\_pl           |
| l#_CPPv4N5cudaq8gradient5cloneEv) | atform::get\_remote\_capabilities |
| -   [cudaq::gradient::compute     |     (C++                          |
|     (C++                          |     function)]                    |
|     function)](api/language       | (api/languages/cpp_api.html#_CPPv |
| s/cpp_api.html#_CPPv4N5cudaq8grad | 4NK5cudaq16quantum_platform23get_ |
| ient7computeERKNSt6vectorIdEERKNS | remote_capabilitiesEKNSt6size_tE) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cud                          |
|     [\[1\]](ap                    | aq::quantum\_platform::get\_shots |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq8gradient7computeERKNSt6vect |     function)](api/l              |
| orIdEERNSt6vectorIdEERK7spin_opd) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::gradient::gradient    | aq16quantum_platform9get_shotsEv) |
|     (C++                          | -   [cudaq                        |
|     function)](api/lang           | ::quantum\_platform::getLogStream |
| uages/cpp_api.html#_CPPv4I00EN5cu |     (C++                          |
| daq8gradient8gradientER7KernelT), |     function)](api/langu          |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq16 |
|    [\[1\]](api/languages/cpp_api. | quantum_platform12getLogStreamEv) |
| html#_CPPv4I00EN5cudaq8gradient8g | -   [cudaq                        |
| radientER7KernelTRR10ArgsMapper), | ::quantum\_platform::is\_emulated |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |                                   |
| Pv4I00EN5cudaq8gradient8gradientE |   function)](api/languages/cpp_ap |
| RR13QuantumKernelRR10ArgsMapper), | i.html#_CPPv4NK5cudaq16quantum_pl |
|     [\[3                          | atform11is_emulatedEKNSt6size_tE) |
| \]](api/languages/cpp_api.html#_C | -   [cud                          |
| PPv4N5cudaq8gradient8gradientERRN | aq::quantum\_platform::is\_remote |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[                           |     function)](api/languages/cp   |
| 4\]](api/languages/cpp_api.html#_ | p_api.html#_CPPv4N5cudaq16quantum |
| CPPv4N5cudaq8gradient8gradientEv) | _platform9is_remoteEKNSt6size_tE) |
| -   [cudaq::gradient::setArgs     | -   [cudaq:                       |
|     (C++                          | :quantum\_platform::is\_simulator |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |                                   |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |  function)](api/languages/cpp_api |
| tArgsEvR13QuantumKernelDpRR4Args) | .html#_CPPv4NK5cudaq16quantum_pla |
| -   [cudaq::gradient::setKernel   | tform12is_simulatorEKNSt6size_tE) |
|     (C++                          | -   [cu                           |
|     function)](api/languages/c    | daq::quantum\_platform::launchVQE |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     (C++                          |
| ient9setKernelEvR13QuantumKernel) |                                   |
| -   [cuda                         | function)](api/languages/cpp_api. |
| q::gradients::central\_difference | html#_CPPv4N5cudaq16quantum_platf |
|     (C++                          | orm9launchVQEEKNSt6stringEPKvPN5c |
|     class)](api/la                | udaq8gradientERKN5cudaq7spin_opER |
| nguages/cpp_api.html#_CPPv4N5cuda | N5cudaq9optimizerEKiKNSt6size_tE) |
| q9gradients18central_differenceE) | -   [cudaq::q                     |
| -   [cudaq::grad                  | uantum\_platform::list\_platforms |
| ients::central\_difference::clone |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/languages      | es/cpp_api.html#_CPPv4N5cudaq16qu |
| /cpp_api.html#_CPPv4N5cudaq9gradi | antum_platform14list_platformsEv) |
| ents18central_difference5cloneEv) | -                                 |
| -   [cudaq::gradie                |   [cudaq::quantum\_platform::name |
| nts::central\_difference::compute |     (C++                          |
|     (C++                          |     function)](a                  |
|     function)](                   | pi/languages/cpp_api.html#_CPPv4N |
| api/languages/cpp_api.html#_CPPv4 | K5cudaq16quantum_platform4nameEv) |
| N5cudaq9gradients18central_differ | -   [cu                           |
| ence7computeERKNSt6vectorIdEERKNS | daq::quantum\_platform::num\_qpus |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|                                   |     function)](api/l              |
|   [\[1\]](api/languages/cpp_api.h | anguages/cpp_api.html#_CPPv4NK5cu |
| tml#_CPPv4N5cudaq9gradients18cent | daq16quantum_platform8num_qpusEv) |
| ral_difference7computeERKNSt6vect | -   [cudaq::q                     |
| orIdEERNSt6vectorIdEERK7spin_opd) | uantum\_platform::onRandomSeedSet |
| -   [cudaq::gradien               |     (C++                          |
| ts::central\_difference::gradient |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     functio                       | html#_CPPv4N5cudaq16quantum_platf |
| n)](api/languages/cpp_api.html#_C | orm15onRandomSeedSetENSt6size_tE) |
| PPv4I00EN5cudaq9gradients18centra | -   [cudaq::qu                    |
| l_difference8gradientER7KernelT), | antum\_platform::reset\_exec\_ctx |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4I00EN5cuda |                                   |
| q9gradients18central_difference8g |  function)](api/languages/cpp_api |
| radientER7KernelTRR10ArgsMapper), | .html#_CPPv4N5cudaq16quantum_plat |
|     [\[2\]](api/languages/cpp_    | form14reset_exec_ctxENSt6size_tE) |
| api.html#_CPPv4I00EN5cudaq9gradie | -   [cudaq                        |
| nts18central_difference8gradientE | ::quantum\_platform::reset\_noise |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3\]](api/languages/cpp     |     function)](api/lang           |
| _api.html#_CPPv4N5cudaq9gradients | uages/cpp_api.html#_CPPv4N5cudaq1 |
| 18central_difference8gradientERRN | 6quantum_platform11reset_noiseEv) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::                      |
|     [\[4\]](api/languages/cp      | quantum\_platform::resetLogStream |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18central_difference8gradientEv) |     function)](api/languag        |
| -   [cuda                         | es/cpp_api.html#_CPPv4N5cudaq16qu |
| q::gradients::forward\_difference | antum_platform14resetLogStreamEv) |
|     (C++                          | -   [cudaq::qua                   |
|     class)](api/la                | ntum\_platform::set\_current\_qpu |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9gradients18forward_differenceE) |     f                             |
| -   [cudaq::grad                  | unction)](api/languages/cpp_api.h |
| ients::forward\_difference::clone | tml#_CPPv4N5cudaq16quantum_platfo |
|     (C++                          | rm15set_current_qpuEKNSt6size_tE) |
|     function)](api/languages      | -   [cudaq::                      |
| /cpp_api.html#_CPPv4N5cudaq9gradi | quantum\_platform::set\_exec\_ctx |
| ents18forward_difference5cloneEv) |     (C++                          |
| -   [cudaq::gradie                |     function)](api/languages      |
| nts::forward\_difference::compute | /cpp_api.html#_CPPv4N5cudaq16quan |
|     (C++                          | tum_platform12set_exec_ctxEPN5cud |
|     function)](                   | aq16ExecutionContextENSt6size_tE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cud                          |
| N5cudaq9gradients18forward_differ | aq::quantum\_platform::set\_noise |
| ence7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |                                   |
|                                   |    function)](api/languages/cpp_a |
|   [\[1\]](api/languages/cpp_api.h | pi.html#_CPPv4N5cudaq16quantum_pl |
| tml#_CPPv4N5cudaq9gradients18forw | atform9set_noiseEPK11noise_model) |
| ard_difference7computeERKNSt6vect | -   [cud                          |
| orIdEERNSt6vectorIdEERK7spin_opd) | aq::quantum\_platform::set\_shots |
| -   [cudaq::gradien               |     (C++                          |
| ts::forward\_difference::gradient |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     functio                       | aq16quantum_platform9set_shotsEi) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq                        |
| PPv4I00EN5cudaq9gradients18forwar | ::quantum\_platform::setLogStream |
| d_difference8gradientER7KernelT), |     (C++                          |
|     [\[1\]](api/langua            |                                   |
| ges/cpp_api.html#_CPPv4I00EN5cuda |  function)](api/languages/cpp_api |
| q9gradients18forward_difference8g | .html#_CPPv4N5cudaq16quantum_plat |
| radientER7KernelTRR10ArgsMapper), | form12setLogStreamERNSt7ostreamE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::qu                    |
| api.html#_CPPv4I00EN5cudaq9gradie | antum\_platform::setTargetBackend |
| nts18forward_difference8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     fun                           |
|     [\[3\]](api/languages/cpp     | ction)](api/languages/cpp_api.htm |
| _api.html#_CPPv4N5cudaq9gradients | l#_CPPv4N5cudaq16quantum_platform |
| 18forward_difference8gradientERRN | 16setTargetBackendERKNSt6stringE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::quantum\_platform     |
|     [\[4\]](api/languages/cp      | ::supports\_conditional\_feedback |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18forward_difference8gradientEv) |     function)](api/l              |
| -   [c                            | anguages/cpp_api.html#_CPPv4NK5cu |
| udaq::gradients::parameter\_shift | daq16quantum_platform29supports_c |
|     (C++                          | onditional_feedbackEKNSt6size_tE) |
|     class)](api                   | -   [cudaq::quantum\_platform:    |
| /languages/cpp_api.html#_CPPv4N5c | :supports\_explicit\_measurements |
| udaq9gradients15parameter_shiftE) |     (C++                          |
| -   [cudaq::g                     |     function)](api/la             |
| radients::parameter\_shift::clone | nguages/cpp_api.html#_CPPv4NK5cud |
|     (C++                          | aq16quantum_platform30supports_ex |
|     function)](api/langua         | plicit_measurementsEKNSt6size_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | -   [cudaq::quantum\_platf        |
| adients15parameter_shift5cloneEv) | orm::supports\_task\_distribution |
| -   [cudaq::gra                   |     (C++                          |
| dients::parameter\_shift::compute |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function                      | ml#_CPPv4NK5cudaq16quantum_platfo |
| )](api/languages/cpp_api.html#_CP | rm26supports_task_distributionEv) |
| Pv4N5cudaq9gradients15parameter_s | -   [cudaq::QuantumTask (C++      |
| hift7computeERKNSt6vectorIdEERKNS |     type)](api/languages/cpp_api. |
| t8functionIFdNSt6vectorIdEEEEEd), | html#_CPPv4N5cudaq11QuantumTaskE) |
|     [\[1\]](api/languages/cpp_ap  | -   [cudaq::qubit (C++            |
| i.html#_CPPv4N5cudaq9gradients15p |     type)](api/languages/c        |
| arameter_shift7computeERKNSt6vect | pp_api.html#_CPPv4N5cudaq5qubitE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QubitConnectivity     |
| -   [cudaq::grad                  |     (C++                          |
| ients::parameter\_shift::gradient |     ty                            |
|     (C++                          | pe)](api/languages/cpp_api.html#_ |
|     func                          | CPPv4N5cudaq17QubitConnectivityE) |
| tion)](api/languages/cpp_api.html | -   [cudaq::QubitEdge (C++        |
| #_CPPv4I00EN5cudaq9gradients15par |     type)](api/languages/cpp_a    |
| ameter_shift8gradientER7KernelT), | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|     [\[1\]](api/lan               | -   [cudaq::qudit (C++            |
| guages/cpp_api.html#_CPPv4I00EN5c |     clas                          |
| udaq9gradients15parameter_shift8g | s)](api/languages/cpp_api.html#_C |
| radientER7KernelTRR10ArgsMapper), | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     [\[2\]](api/languages/c       | -   [cudaq::qudit::qudit (C++     |
| pp_api.html#_CPPv4I00EN5cudaq9gra |                                   |
| dients15parameter_shift8gradientE | function)](api/languages/cpp_api. |
| RR13QuantumKernelRR10ArgsMapper), | html#_CPPv4N5cudaq5qudit5quditEv) |
|     [\[3\]](api/languages/        | -   [cudaq::qvector (C++          |
| cpp_api.html#_CPPv4N5cudaq9gradie |     class)                        |
| nts15parameter_shift8gradientERRN | ](api/languages/cpp_api.html#_CPP |
| St8functionIFvNSt6vectorIdEEEEE), | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     [\[4\]](api/languages         | -   [cudaq::qvector::back (C++    |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)](a                  |
| ents15parameter_shift8gradientEv) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::kernel\_builder (C++  | 5cudaq7qvector4backENSt6size_tE), |
|     clas                          |                                   |
| s)](api/languages/cpp_api.html#_C |   [\[1\]](api/languages/cpp_api.h |
| PPv4IDpEN5cudaq14kernel_builderE) | tml#_CPPv4N5cudaq7qvector4backEv) |
| -   [cu                           | -   [cudaq::qvector::begin (C++   |
| daq::kernel\_builder::constantVal |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     function)](api/la             | ml#_CPPv4N5cudaq7qvector5beginEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::qvector::clear (C++   |
| q14kernel_builder11constantValEd) |     fu                            |
| -   [cud                          | nction)](api/languages/cpp_api.ht |
| aq::kernel\_builder::getArguments | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     (C++                          | -   [cudaq::qvector::end (C++     |
|     function)](api/lan            |                                   |
| guages/cpp_api.html#_CPPv4N5cudaq | function)](api/languages/cpp_api. |
| 14kernel_builder12getArgumentsEv) | html#_CPPv4N5cudaq7qvector3endEv) |
| -   [cud                          | -   [cudaq::qvector::front (C++   |
| aq::kernel\_builder::getNumParams |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/lan            | cudaq7qvector5frontENSt6size_tE), |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 14kernel_builder12getNumParamsEv) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cu                           | ml#_CPPv4N5cudaq7qvector5frontEv) |
| daq::kernel\_builder::isArgStdVec | -   [cudaq::qvector::operator=    |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     functio                       |
| p_api.html#_CPPv4N5cudaq14kernel_ | n)](api/languages/cpp_api.html#_C |
| builder11isArgStdVecENSt6size_tE) | PPv4N5cudaq7qvectoraSERK7qvector) |
| -   [cudaq:                       | -   [cudaq::qvector::operator\[\] |
| :kernel\_builder::kernel\_builder |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/languages/cpp_ | ](api/languages/cpp_api.html#_CPP |
| api.html#_CPPv4N5cudaq14kernel_bu | v4N5cudaq7qvectorixEKNSt6size_tE) |
| ilder14kernel_builderERNSt6vector | -   [cudaq::qvector::qvector (C++ |
| IN7details17KernelBuilderTypeEEE) |     function)](api/               |
| -   [cudaq::kernel\_builder::name | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq7qvector7qvectorENSt6size_tE), |
|     function)                     |     [\[1\]](a                     |
| ](api/languages/cpp_api.html#_CPP | pi/languages/cpp_api.html#_CPPv4N |
| v4N5cudaq14kernel_builder4nameEv) | 5cudaq7qvector7qvectorERK5state), |
| -                                 |     [\[2\]](api                   |
|   [cudaq::kernel\_builder::qalloc | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq7qvector7qvectorERK7qvector), |
|     function)](api/language       |     [\[3\]](api/languages/cpp     |
| s/cpp_api.html#_CPPv4N5cudaq14ker | _api.html#_CPPv4N5cudaq7qvector7q |
| nel_builder6qallocE10QuakeValue), | vectorERKNSt6vectorI7complexEEb), |
|     [\[1\]](api/language          |     [\[4\]](ap                    |
| s/cpp_api.html#_CPPv4N5cudaq14ker | i/languages/cpp_api.html#_CPPv4N5 |
| nel_builder6qallocEKNSt6size_tE), | cudaq7qvector7qvectorERR7qvector) |
|     [\[2                          | -   [cudaq::qvector::size (C++    |
| \]](api/languages/cpp_api.html#_C |     fu                            |
| PPv4N5cudaq14kernel_builder6qallo | nction)](api/languages/cpp_api.ht |
| cERNSt6vectorINSt7complexIdEEEE), | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     [\[3\]](                      | -   [cudaq::qvector::slice (C++   |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/language       |
| N5cudaq14kernel_builder6qallocEv) | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| -   [cudaq::kernel\_builder::swap | tor5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qvector::value\_type  |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     typ                           |
| 4kernel_builder4swapEvRK10QuakeVa | e)](api/languages/cpp_api.html#_C |
| lueRK10QuakeValueRK10QuakeValue), | PPv4N5cudaq7qvector10value_typeE) |
|                                   | -   [cudaq::qview (C++            |
| [\[1\]](api/languages/cpp_api.htm |     clas                          |
| l#_CPPv4I00EN5cudaq14kernel_build | s)](api/languages/cpp_api.html#_C |
| er4swapEvRKNSt6vectorI10QuakeValu | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| eEERK10QuakeValueRK10QuakeValue), | -   [cudaq::qview::value\_type    |
|                                   |     (C++                          |
| [\[2\]](api/languages/cpp_api.htm |     t                             |
| l#_CPPv4N5cudaq14kernel_builder4s | ype)](api/languages/cpp_api.html# |
| wapERK10QuakeValueRK10QuakeValue) | _CPPv4N5cudaq5qview10value_typeE) |
| -   [cudaq::KernelExecutionTask   | -   [cudaq::range (C++            |
|     (C++                          |     func                          |
|     type                          | tion)](api/languages/cpp_api.html |
| )](api/languages/cpp_api.html#_CP | #_CPPv4I00EN5cudaq5rangeENSt6vect |
| Pv4N5cudaq19KernelExecutionTaskE) | orI11ElementTypeEE11ElementType), |
| -   [cudaq::KernelThunkResultType |     [\[1\]](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4I00EN5cudaq5rangeEN |
|     struct)]                      | St6vectorI11ElementTypeEE11Elemen |
| (api/languages/cpp_api.html#_CPPv | tType11ElementType11ElementType), |
| 4N5cudaq21KernelThunkResultTypeE) |     [                             |
| -   [cudaq::KernelThunkType (C++  | \[2\]](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| type)](api/languages/cpp_api.html | -   [cudaq::real (C++             |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     type)](api/languages/         |
| -   [cudaq::kraus\_channel (C++   | cpp_api.html#_CPPv4N5cudaq4realE) |
|                                   | -   [cudaq::registry (C++         |
|  class)](api/languages/cpp_api.ht |     type)](api/languages/cpp_     |
| ml#_CPPv4N5cudaq13kraus_channelE) | api.html#_CPPv4N5cudaq8registryE) |
| -   [cudaq::kraus\_channel::empty | -                                 |
|     (C++                          |  [cudaq::registry::RegisteredType |
|     function)]                    |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     class)](api/                  |
| 4NK5cudaq13kraus_channel5emptyEv) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::kraus\_c              | 5cudaq8registry14RegisteredTypeE) |
| hannel::generateUnitaryParameters | -   [cudaq::RemoteCapabilities    |
|     (C++                          |     (C++                          |
|                                   |     struc                         |
|    function)](api/languages/cpp_a | t)](api/languages/cpp_api.html#_C |
| pi.html#_CPPv4N5cudaq13kraus_chan | PPv4N5cudaq18RemoteCapabilitiesE) |
| nel25generateUnitaryParametersEv) | -   [cudaq::Remo                  |
| -                                 | teCapabilities::isRemoteSimulator |
|  [cudaq::kraus\_channel::get\_ops |     (C++                          |
|     (C++                          |     member)](api/languages/c      |
|     function)](a                  | pp_api.html#_CPPv4N5cudaq18Remote |
| pi/languages/cpp_api.html#_CPPv4N | Capabilities17isRemoteSimulatorE) |
| K5cudaq13kraus_channel7get_opsEv) | -   [cudaq::Remot                 |
| -   [cudaq::kra                   | eCapabilities::RemoteCapabilities |
| us\_channel::is\_unitary\_mixture |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api/languages      | _api.html#_CPPv4N5cudaq18RemoteCa |
| /cpp_api.html#_CPPv4NK5cudaq13kra | pabilities18RemoteCapabilitiesEb) |
| us_channel18is_unitary_mixtureEv) | -   [cudaq::Remot                 |
| -   [cuda                         | eCapabilities::serializedCodeExec |
| q::kraus\_channel::kraus\_channel |     (C++                          |
|     (C++                          |     member)](api/languages/cp     |
|     function)](api/lang           | p_api.html#_CPPv4N5cudaq18RemoteC |
| uages/cpp_api.html#_CPPv4IDpEN5cu | apabilities18serializedCodeExecE) |
| daq13kraus_channel13kraus_channel | -   [cudaq:                       |
| EDpRRNSt16initializer_listI1TEE), | :RemoteCapabilities::stateOverlap |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     member)](api/langua           |
| ml#_CPPv4N5cudaq13kraus_channel13 | ges/cpp_api.html#_CPPv4N5cudaq18R |
| kraus_channelERK13kraus_channel), | emoteCapabilities12stateOverlapE) |
|     [\[2\]                        | -                                 |
| ](api/languages/cpp_api.html#_CPP |   [cudaq::RemoteCapabilities::vqe |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERKNSt6vectorI8kraus_opEE), |     member)](                     |
|     [\[3\]                        | api/languages/cpp_api.html#_CPPv4 |
| ](api/languages/cpp_api.html#_CPP | N5cudaq18RemoteCapabilities3vqeE) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::RemoteSimulationState |
| hannelERRNSt6vectorI8kraus_opEE), |     (C++                          |
|     [\[4\]](api/lan               |     class)]                       |
| guages/cpp_api.html#_CPPv4N5cudaq | (api/languages/cpp_api.html#_CPPv |
| 13kraus_channel13kraus_channelEv) | 4N5cudaq21RemoteSimulationStateE) |
| -   [c                            | -   [cudaq::Resources (C++        |
| udaq::kraus\_channel::noise\_type |     class)](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     member)](api                  | -   [cudaq::run (C++              |
| /languages/cpp_api.html#_CPPv4N5c |     function)]                    |
| udaq13kraus_channel10noise_typeE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| [cudaq::kraus\_channel::operator= | 5invoke_result_tINSt7decay_tI13Qu |
|     (C++                          | antumKernelEEDpNSt7decay_tI4ARGSE |
|     function)](api/langua         | EEEEENSt6size_tERN5cudaq11noise_m |
| ges/cpp_api.html#_CPPv4N5cudaq13k | odelERR13QuantumKernelDpRR4ARGS), |
| raus_channelaSERK13kraus_channel) |     [\[1\]](api/langu             |
| -   [cu                           | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| daq::kraus\_channel::operator\[\] | daq3runENSt6vectorINSt15invoke_re |
|     (C++                          | sult_tINSt7decay_tI13QuantumKerne |
|     function)](api/l              | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| anguages/cpp_api.html#_CPPv4N5cud | ize_tERR13QuantumKernelDpRR4ARGS) |
| aq13kraus_channelixEKNSt6size_tE) | -   [cudaq::run\_async (C++       |
| -   [                             |     functio                       |
| cudaq::kraus\_channel::parameters | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     member)](api                  | tureINSt6vectorINSt15invoke_resul |
| /languages/cpp_api.html#_CPPv4N5c | t_tINSt7decay_tI13QuantumKernelEE |
| udaq13kraus_channel10parametersE) | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| -   [cud                          | ze_tENSt6size_tERN5cudaq11noise_m |
| aq::kraus\_channel::probabilities | odelERR13QuantumKernelDpRR4ARGS), |
|     (C++                          |     [\[1\]](api/la                |
|     member)](api/la               | nguages/cpp_api.html#_CPPv4I0DpEN |
| nguages/cpp_api.html#_CPPv4N5cuda | 5cudaq9run_asyncENSt6futureINSt6v |
| q13kraus_channel13probabilitiesE) | ectorINSt15invoke_result_tINSt7de |
| -   [                             | cay_tI13QuantumKernelEEDpNSt7deca |
| cudaq::kraus\_channel::push\_back | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4ARGS) |
|     function)](api/langua         | -   [cudaq::sample (C++           |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     function)](api/languages/cp   |
| raus_channel9push_backE8kraus_op) | p_api.html#_CPPv4I0Dp0EN5cudaq6sa |
| -   [cudaq::kraus\_channel::size  | mpleE13sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     function)                     |     [\[1\]                        |
| ](api/languages/cpp_api.html#_CPP | ](api/languages/cpp_api.html#_CPP |
| v4NK5cudaq13kraus_channel4sizeEv) | v4I0Dp0EN5cudaq6sampleE13sample_r |
| -   [cu                           | esultRR13QuantumKernelDpRR4Args), |
| daq::kraus\_channel::unitary\_ops |     [\[                           |
|     (C++                          | 2\]](api/languages/cpp_api.html#_ |
|     member)](api/                 | CPPv4I0Dp0EN5cudaq6sampleEDaNSt6s |
| languages/cpp_api.html#_CPPv4N5cu | ize_tERR13QuantumKernelDpRR4Args) |
| daq13kraus_channel11unitary_opsE) | -   [cudaq::sample\_options (C++  |
| -   [cudaq::kraus\_op (C++        |     s                             |
|     struct)](api/languages/cpp_   | truct)](api/languages/cpp_api.htm |
| api.html#_CPPv4N5cudaq8kraus_opE) | l#_CPPv4N5cudaq14sample_optionsE) |
| -   [cudaq::kraus\_op::adjoint    | -   [cudaq::sample\_result (C++   |
|     (C++                          |                                   |
|     functi                        |  class)](api/languages/cpp_api.ht |
| on)](api/languages/cpp_api.html#_ | ml#_CPPv4N5cudaq13sample_resultE) |
| CPPv4NK5cudaq8kraus_op7adjointEv) | -                                 |
| -   [cudaq::kraus\_op::data (C++  |    [cudaq::sample\_result::append |
|                                   |     (C++                          |
|  member)](api/languages/cpp_api.h |     function)](api/languages/cpp  |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | _api.html#_CPPv4N5cudaq13sample_r |
| -   [cudaq::kraus\_op::kraus\_op  | esult6appendER15ExecutionResultb) |
|     (C++                          | -   [cudaq::sample\_result::begin |
|     func                          |     (C++                          |
| tion)](api/languages/cpp_api.html |     function)]                    |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | (api/languages/cpp_api.html#_CPPv |
| opERRNSt16initializer_listI1TEE), | 4N5cudaq13sample_result5beginEv), |
|                                   |     [\[1\]]                       |
|  [\[1\]](api/languages/cpp_api.ht | (api/languages/cpp_api.html#_CPPv |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | 4NK5cudaq13sample_result5beginEv) |
| pENSt6vectorIN5cudaq7complexEEE), | -                                 |
|     [\[2\]](api/l                 |    [cudaq::sample\_result::cbegin |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq8kraus_op8kraus_opERK8kraus_op) |     function)](                   |
| -   [cudaq::kraus\_op::nCols (C++ | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq13sample_result6cbeginEv) |
| member)](api/languages/cpp_api.ht | -   [cudaq::sample\_result::cend  |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     (C++                          |
| -   [cudaq::kraus\_op::nRows (C++ |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
| member)](api/languages/cpp_api.ht | v4NK5cudaq13sample_result4cendEv) |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | -   [cudaq::sample\_result::clear |
| -   [cudaq::kraus\_op::operator=  |     (C++                          |
|     (C++                          |     function)                     |
|     function)                     | ](api/languages/cpp_api.html#_CPP |
| ](api/languages/cpp_api.html#_CPP | v4N5cudaq13sample_result5clearEv) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cudaq::sample\_result::count |
| -   [cudaq::kraus\_op::precision  |     (C++                          |
|     (C++                          |     function)](                   |
|     memb                          | api/languages/cpp_api.html#_CPPv4 |
| er)](api/languages/cpp_api.html#_ | NK5cudaq13sample_result5countENSt |
| CPPv4N5cudaq8kraus_op9precisionE) | 11string_viewEKNSt11string_viewE) |
| -   [cudaq::matrix\_callback (C++ | -   [c                            |
|     c                             | udaq::sample\_result::deserialize |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15matrix_callbackE) |     functio                       |
| -   [cudaq::matrix\_handler (C++  | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq13sample_result11deser |
| class)](api/languages/cpp_api.htm | ializeERNSt6vectorINSt6size_tEEE) |
| l#_CPPv4N5cudaq14matrix_handlerE) | -   [cudaq::sample\_result::dump  |
| -   [cudaq::matri                 |     (C++                          |
| x\_handler::commutation\_behavior |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     struct)](api/languages/       | ample_result4dumpERNSt7ostreamE), |
| cpp_api.html#_CPPv4N5cudaq14matri |     [\[1\]                        |
| x_handler20commutation_behaviorE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4NK5cudaq13sample_result4dumpEv) |
|   [cudaq::matrix\_handler::define | -   [cudaq::sample\_result::end   |
|     (C++                          |     (C++                          |
|     function)](a                  |     function                      |
| pi/languages/cpp_api.html#_CPPv4N | )](api/languages/cpp_api.html#_CP |
| 5cudaq14matrix_handler6defineENSt | Pv4N5cudaq13sample_result3endEv), |
| 6stringENSt6vectorINSt7int64_tEEE |     [\[1\                         |
| RR15matrix_callbackRKNSt13unorder | ]](api/languages/cpp_api.html#_CP |
| ed_mapINSt6stringENSt6stringEEE), | Pv4NK5cudaq13sample_result3endEv) |
|                                   | -   [c                            |
| [\[1\]](api/languages/cpp_api.htm | udaq::sample\_result::expectation |
| l#_CPPv4N5cudaq14matrix_handler6d |     (C++                          |
| efineENSt6stringENSt6vectorINSt7i |     f                             |
| nt64_tEEERR15matrix_callbackRR20d | unction)](api/languages/cpp_api.h |
| iag_matrix_callbackRKNSt13unorder | tml#_CPPv4NK5cudaq13sample_result |
| ed_mapINSt6stringENSt6stringEEE), | 11expectationEKNSt11string_viewE) |
|     [\[2\]](                      | -   [cud                          |
| api/languages/cpp_api.html#_CPPv4 | aq::sample\_result::get\_marginal |
| N5cudaq14matrix_handler6defineENS |     (C++                          |
| t6stringENSt6vectorINSt7int64_tEE |     function)](api/languages/cpp_ |
| ERR15matrix_callbackRRNSt13unorde | api.html#_CPPv4NK5cudaq13sample_r |
| red_mapINSt6stringENSt6stringEEE) | esult12get_marginalERKNSt6vectorI |
| -                                 | NSt6size_tEEEKNSt11string_viewE), |
|  [cudaq::matrix\_handler::degrees |     [\[1\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4NK5cudaq13sample_r |
|     function)](ap                 | esult12get_marginalERRKNSt6vector |
| i/languages/cpp_api.html#_CPPv4NK | INSt6size_tEEEKNSt11string_viewE) |
| 5cudaq14matrix_handler7degreesEv) | -   [cudaq::                      |
| -                                 | sample\_result::get\_total\_shots |
| [cudaq::matrix\_handler::displace |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/language       | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| s/cpp_api.html#_CPPv4N5cudaq14mat | sample_result15get_total_shotsEv) |
| rix_handler8displaceENSt6size_tE) | -   [cudaq::                      |
| -   [cudaq::matrix\_h             | sample\_result::has\_even\_parity |
| andler::get\_expected\_dimensions |     (C++                          |
|     (C++                          |     fun                           |
|                                   | ction)](api/languages/cpp_api.htm |
|    function)](api/languages/cpp_a | l#_CPPv4N5cudaq13sample_result15h |
| pi.html#_CPPv4NK5cudaq14matrix_ha | as_even_parityENSt11string_viewE) |
| ndler23get_expected_dimensionsEv) | -   [cudaq:                       |
| -   [cudaq::matrix\_hand          | :sample\_result::has\_expectation |
| ler::get\_parameter\_descriptions |     (C++                          |
|     (C++                          |     funct                         |
|                                   | ion)](api/languages/cpp_api.html# |
| function)](api/languages/cpp_api. | _CPPv4NK5cudaq13sample_result15ha |
| html#_CPPv4NK5cudaq14matrix_handl | s_expectationEKNSt11string_viewE) |
| er26get_parameter_descriptionsEv) | -   [cuda                         |
| -   [cu                           | q::sample\_result::most\_probable |
| daq::matrix\_handler::instantiate |     (C++                          |
|     (C++                          |     fun                           |
|     function)](a                  | ction)](api/languages/cpp_api.htm |
| pi/languages/cpp_api.html#_CPPv4N | l#_CPPv4NK5cudaq13sample_result13 |
| 5cudaq14matrix_handler11instantia | most_probableEKNSt11string_viewE) |
| teENSt6stringERKNSt6vectorINSt6si | -   [                             |
| ze_tEEERK20commutation_behavior), | cudaq::sample\_result::operator+= |
|     [\[1\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/langua         |
| N5cudaq14matrix_handler11instanti | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ateENSt6stringERRNSt6vectorINSt6s | ample_resultpLERK13sample_result) |
| ize_tEEERK20commutation_behavior) | -                                 |
| -   [cudaq:                       | [cudaq::sample\_result::operator= |
| :matrix\_handler::matrix\_handler |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/languag        | ges/cpp_api.html#_CPPv4N5cudaq13s |
| es/cpp_api.html#_CPPv4I0_NSt11ena | ample_resultaSER13sample_result), |
| ble_if_tINSt12is_base_of_vI16oper |     [\[1\]](api/langua            |
| ator_handler1TEEbEEEN5cudaq14matr | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ix_handler14matrix_handlerERK1T), | ample_resultaSERR13sample_result) |
|     [\[1\]](ap                    | -   [                             |
| i/languages/cpp_api.html#_CPPv4I0 | cudaq::sample\_result::operator== |
| _NSt11enable_if_tINSt12is_base_of |     (C++                          |
| _vI16operator_handler1TEEbEEEN5cu |     function)](api/languag        |
| daq14matrix_handler14matrix_handl | es/cpp_api.html#_CPPv4NK5cudaq13s |
| erERK1TRK20commutation_behavior), | ample_resulteqERK13sample_result) |
|     [\[2\]](api/languages/cpp_ap  | -   [c                            |
| i.html#_CPPv4N5cudaq14matrix_hand | udaq::sample\_result::probability |
| ler14matrix_handlerENSt6size_tE), |     (C++                          |
|     [\[3\]](api/                  |     function)](api/lan            |
| languages/cpp_api.html#_CPPv4N5cu | guages/cpp_api.html#_CPPv4NK5cuda |
| daq14matrix_handler14matrix_handl | q13sample_result11probabilityENSt |
| erENSt6stringERKNSt6vectorINSt6si | 11string_viewEKNSt11string_viewE) |
| ze_tEEERK20commutation_behavior), | -   [cudaq                        |
|     [\[4\]](api/                  | ::sample\_result::register\_names |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     function)](api/langu          |
| erENSt6stringERRNSt6vectorINSt6si | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| ze_tEEERK20commutation_behavior), | 3sample_result14register_namesEv) |
|     [\                            | -                                 |
| [5\]](api/languages/cpp_api.html# |   [cudaq::sample\_result::reorder |
| _CPPv4N5cudaq14matrix_handler14ma |     (C++                          |
| trix_handlerERK14matrix_handler), |     function)](api/langua         |
|     [                             | ges/cpp_api.html#_CPPv4N5cudaq13s |
| \[6\]](api/languages/cpp_api.html | ample_result7reorderERKNSt6vector |
| #_CPPv4N5cudaq14matrix_handler14m | INSt6size_tEEEKNSt11string_viewE) |
| atrix_handlerERR14matrix_handler) | -   [cuda                         |
| -                                 | q::sample\_result::sample\_result |
| [cudaq::matrix\_handler::momentum |     (C++                          |
|     (C++                          |     fun                           |
|     function)](api/language       | ction)](api/languages/cpp_api.htm |
| s/cpp_api.html#_CPPv4N5cudaq14mat | l#_CPPv4N5cudaq13sample_result13s |
| rix_handler8momentumENSt6size_tE) | ample_resultER15ExecutionResult), |
| -                                 |                                   |
|   [cudaq::matrix\_handler::number |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq13sample_result13 |
|     function)](api/langua         | sample_resultERK13sample_result), |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     [\[2\]](api/l                 |
| atrix_handler6numberENSt6size_tE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [                             | aq13sample_result13sample_resultE |
| cudaq::matrix\_handler::operator= | RNSt6vectorI15ExecutionResultEE), |
|     (C++                          |                                   |
|     fun                           |  [\[3\]](api/languages/cpp_api.ht |
| ction)](api/languages/cpp_api.htm | ml#_CPPv4N5cudaq13sample_result13 |
| l#_CPPv4I0_NSt11enable_if_tIXaant | sample_resultERR13sample_result), |
| NSt7is_sameI1T14matrix_handlerE5v |     [                             |
| alueENSt12is_base_of_vI16operator | \[4\]](api/languages/cpp_api.html |
| _handler1TEEEbEEEN5cudaq14matrix_ | #_CPPv4N5cudaq13sample_result13sa |
| handleraSER14matrix_handlerRK1T), | mple_resultERR15ExecutionResult), |
|     [\[1\]](api/languages         |     [\[5\]](api/la                |
| /cpp_api.html#_CPPv4N5cudaq14matr | nguages/cpp_api.html#_CPPv4N5cuda |
| ix_handleraSERK14matrix_handler), | q13sample_result13sample_resultEd |
|     [\[2\]](api/language          | RNSt6vectorI15ExecutionResultEE), |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     [\[6\]](api/lan               |
| rix_handleraSERR14matrix_handler) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [c                            | 13sample_result13sample_resultEv) |
| udaq::matrix\_handler::operator== | -                                 |
|     (C++                          | [cudaq::sample\_result::serialize |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4NK5cudaq14mat |     function)](api                |
| rix_handlereqERK14matrix_handler) | /languages/cpp_api.html#_CPPv4NK5 |
| -                                 | cudaq13sample_result9serializeEv) |
|   [cudaq::matrix\_handler::parity | -   [cudaq::sample\_result::size  |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     function)](api/languages/c    |
| ges/cpp_api.html#_CPPv4N5cudaq14m | pp_api.html#_CPPv4NK5cudaq13sampl |
| atrix_handler6parityENSt6size_tE) | e_result4sizeEKNSt11string_viewE) |
| -                                 | -                                 |
| [cudaq::matrix\_handler::position |   [cudaq::sample\_result::to\_map |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api/languages/cpp  |
| s/cpp_api.html#_CPPv4N5cudaq14mat | _api.html#_CPPv4NK5cudaq13sample_ |
| rix_handler8positionENSt6size_tE) | result6to_mapEKNSt11string_viewE) |
| -   [cudaq::ma                    | -   [cudaq:                       |
| trix\_handler::remove\_definition | :sample\_result::\~sample\_result |
|     (C++                          |     (C++                          |
|     fu                            |     funct                         |
| nction)](api/languages/cpp_api.ht | ion)](api/languages/cpp_api.html# |
| ml#_CPPv4N5cudaq14matrix_handler1 | _CPPv4N5cudaq13sample_resultD0Ev) |
| 7remove_definitionERKNSt6stringE) | -   [cudaq::scalar\_callback (C++ |
| -                                 |     c                             |
|  [cudaq::matrix\_handler::squeeze | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_callbackE) |
|     function)](api/languag        | -   [cu                           |
| es/cpp_api.html#_CPPv4N5cudaq14ma | daq::scalar\_callback::operator() |
| trix_handler7squeezeENSt6size_tE) |     (C++                          |
| -   [cudaq::matr                  |     function)](api/language       |
| ix\_handler::to\_diagonal\_matrix | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|     (C++                          | alar_callbackclERKNSt13unordered_ |
|     function)](api/lang           | mapINSt6stringENSt7complexIdEEEE) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [c                            |
| 14matrix_handler18to_diagonal_mat | udaq::scalar\_callback::operator= |
| rixERNSt13unordered_mapINSt6size_ |     (C++                          |
| tENSt7int64_tEEERKNSt13unordered_ |     function)](api/languages/c    |
| mapINSt6stringENSt7complexIdEEEE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [c                            | _callbackaSERK15scalar_callback), |
| udaq::matrix\_handler::to\_matrix |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|     function)                     | r_callbackaSERR15scalar_callback) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::s                     |
| v4NK5cudaq14matrix_handler9to_mat | calar\_callback::scalar\_callback |
| rixERNSt13unordered_mapINSt6size_ |     (C++                          |
| tENSt7int64_tEEERKNSt13unordered_ |     function)](api/languag        |
| mapINSt6stringENSt7complexIdEEEE) | es/cpp_api.html#_CPPv4I0_NSt11ena |
| -   [c                            | ble_if_tINSt16is_invocable_r_vINS |
| udaq::matrix\_handler::to\_string | t7complexIdEE8CallableRKNSt13unor |
|     (C++                          | dered_mapINSt6stringENSt7complexI |
|     function)](api/               | dEEEEEEbEEEN5cudaq15scalar_callba |
| languages/cpp_api.html#_CPPv4NK5c | ck15scalar_callbackERR8Callable), |
| udaq14matrix_handler9to_stringEb) |     [\[1\                         |
| -   [c                            | ]](api/languages/cpp_api.html#_CP |
| udaq::matrix\_handler::unique\_id | Pv4N5cudaq15scalar_callback15scal |
|     (C++                          | ar_callbackERK15scalar_callback), |
|     function)](api/               |     [\[2                          |
| languages/cpp_api.html#_CPPv4NK5c | \]](api/languages/cpp_api.html#_C |
| udaq14matrix_handler9unique_idEv) | PPv4N5cudaq15scalar_callback15sca |
| -   [cudaq::m                     | lar_callbackERR15scalar_callback) |
| atrix\_handler::\~matrix\_handler | -   [cudaq::scalar\_operator (C++ |
|     (C++                          |     c                             |
|     functi                        | lass)](api/languages/cpp_api.html |
| on)](api/languages/cpp_api.html#_ | #_CPPv4N5cudaq15scalar_operatorE) |
| CPPv4N5cudaq14matrix_handlerD0Ev) | -   [                             |
| -   [cudaq::matrix\_op (C++       | cudaq::scalar\_operator::evaluate |
|     type)](api/languages/cpp_a    |     (C++                          |
| pi.html#_CPPv4N5cudaq9matrix_opE) |                                   |
| -   [cudaq::matrix\_op\_term (C++ |    function)](api/languages/cpp_a |
|                                   | pi.html#_CPPv4NK5cudaq15scalar_op |
|  type)](api/languages/cpp_api.htm | erator8evaluateERKNSt13unordered_ |
| l#_CPPv4N5cudaq14matrix_op_termE) | mapINSt6stringENSt7complexIdEEEE) |
| -                                 | -   [cudaq::scalar\_opera         |
|  [cudaq::mdiag\_operator\_handler | tor::get\_parameter\_descriptions |
|     (C++                          |     (C++                          |
|     class)](                      |     f                             |
| api/languages/cpp_api.html#_CPPv4 | unction)](api/languages/cpp_api.h |
| N5cudaq22mdiag_operator_handlerE) | tml#_CPPv4NK5cudaq15scalar_operat |
| -   [cudaq::mpi (C++              | or26get_parameter_descriptionsEv) |
|     type)](api/languages          | -   [cuda                         |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | q::scalar\_operator::is\_constant |
| -   [cudaq::mpi::all\_gather (C++ |     (C++                          |
|     fu                            |     function)](api/lang           |
| nction)](api/languages/cpp_api.ht | uages/cpp_api.html#_CPPv4NK5cudaq |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | 15scalar_operator11is_constantEv) |
| RNSt6vectorIdEERKNSt6vectorIdEE), | -   [cu                           |
|                                   | daq::scalar\_operator::operator\* |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi10all_gather |     function                      |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::mpi::all\_reduce (C++ | Pv4N5cudaq15scalar_operatormlENSt |
|                                   | 7complexIdEERK15scalar_operator), |
|  function)](api/languages/cpp_api |     [\[1\                         |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | ]](api/languages/cpp_api.html#_CP |
| reduceE1TRK1TRK14BinaryFunction), | Pv4N5cudaq15scalar_operatormlENSt |
|     [\[1\]](api/langu             | 7complexIdEERR15scalar_operator), |
| ages/cpp_api.html#_CPPv4I00EN5cud |     [\[2\]](api/languages/cp      |
| aq3mpi10all_reduceE1TRK1TRK4Func) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::mpi::broadcast (C++   | operatormlEdRK15scalar_operator), |
|     function)](api/               |     [\[3\]](api/languages/cp      |
| languages/cpp_api.html#_CPPv4N5cu | p_api.html#_CPPv4N5cudaq15scalar_ |
| daq3mpi9broadcastERNSt6stringEi), | operatormlEdRR15scalar_operator), |
|     [\[1\]](api/la                |     [\[4\]](api/languages         |
| nguages/cpp_api.html#_CPPv4N5cuda | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| q3mpi9broadcastERNSt6vectorIdEEi) | alar_operatormlENSt7complexIdEE), |
| -   [cudaq::mpi::finalize (C++    |     [\[5\]](api/languages/cpp     |
|     f                             | _api.html#_CPPv4NKR5cudaq15scalar |
| unction)](api/languages/cpp_api.h | _operatormlERK15scalar_operator), |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     [\[6\]]                       |
| -   [cudaq::mpi::initialize (C++  | (api/languages/cpp_api.html#_CPPv |
|     function                      | 4NKR5cudaq15scalar_operatormlEd), |
| )](api/languages/cpp_api.html#_CP |     [\[7\]](api/language          |
| Pv4N5cudaq3mpi10initializeEiPPc), | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     [                             | alar_operatormlENSt7complexIdEE), |
| \[1\]](api/languages/cpp_api.html |     [\[8\]](api/languages/cp      |
| #_CPPv4N5cudaq3mpi10initializeEv) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::mpi::is\_initialized  | _operatormlERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     function                      | ]](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4NO5cudaq15scalar_operatormlEd) |
| Pv4N5cudaq3mpi14is_initializedEv) | -   [cud                          |
| -   [cudaq::mpi::num\_ranks (C++  | aq::scalar\_operator::operator\*= |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/languag        |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::mpi::rank (C++        | alar_operatormLENSt7complexIdEE), |
|                                   |     [\[1\]](api/languages/c       |
|    function)](api/languages/cpp_a | pp_api.html#_CPPv4N5cudaq15scalar |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | _operatormLERK15scalar_operator), |
| -   [cudaq::noise\_model (C++     |     [\[2                          |
|                                   | \]](api/languages/cpp_api.html#_C |
|    class)](api/languages/cpp_api. | PPv4N5cudaq15scalar_operatormLEd) |
| html#_CPPv4N5cudaq11noise_modelE) | -   [c                            |
| -   [cudaq::noise                 | udaq::scalar\_operator::operator+ |
| \_model::add\_all\_qubit\_channel |     (C++                          |
|     (C++                          |     function                      |
|     function)](api                | )](api/languages/cpp_api.html#_CP |
| /languages/cpp_api.html#_CPPv4IDp | Pv4N5cudaq15scalar_operatorplENSt |
| EN5cudaq11noise_model21add_all_qu | 7complexIdEERK15scalar_operator), |
| bit_channelEvRK13kraus_channeli), |     [\[1\                         |
|     [\[1\]](api/langua            | ]](api/languages/cpp_api.html#_CP |
| ges/cpp_api.html#_CPPv4N5cudaq11n | Pv4N5cudaq15scalar_operatorplENSt |
| oise_model21add_all_qubit_channel | 7complexIdEERR15scalar_operator), |
| ERKNSt6stringERK13kraus_channeli) |     [\[2\]](api/languages/cp      |
| -   [                             | p_api.html#_CPPv4N5cudaq15scalar_ |
| cudaq::noise\_model::add\_channel | operatorplEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     funct                         | p_api.html#_CPPv4N5cudaq15scalar_ |
| ion)](api/languages/cpp_api.html# | operatorplEdRR15scalar_operator), |
| _CPPv4IDpEN5cudaq11noise_model11a |     [\[4\]](api/languages         |
| dd_channelEvRK15PredicateFuncTy), | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     [\[1\]](api/languages/cpp_    | alar_operatorplENSt7complexIdEE), |
| api.html#_CPPv4IDpEN5cudaq11noise |     [\[5\]](api/languages/cpp     |
| _model11add_channelEvRKNSt6vector | _api.html#_CPPv4NKR5cudaq15scalar |
| INSt6size_tEEERK13kraus_channel), | _operatorplERK15scalar_operator), |
|     [\[2\]](ap                    |     [\[6\]]                       |
| i/languages/cpp_api.html#_CPPv4N5 | (api/languages/cpp_api.html#_CPPv |
| cudaq11noise_model11add_channelER | 4NKR5cudaq15scalar_operatorplEd), |
| KNSt6stringERK15PredicateFuncTy), |     [\[7\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
| [\[3\]](api/languages/cpp_api.htm | 4NKR5cudaq15scalar_operatorplEv), |
| l#_CPPv4N5cudaq11noise_model11add |     [\[8\]](api/language          |
| _channelERKNSt6stringERKNSt6vecto | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| rINSt6size_tEEERK13kraus_channel) | alar_operatorplENSt7complexIdEE), |
| -   [cudaq::noise\_model::empty   |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     function                      | _operatorplERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[10\]                       |
| Pv4NK5cudaq11noise_model5emptyEv) | ](api/languages/cpp_api.html#_CPP |
| -   [c                            | v4NO5cudaq15scalar_operatorplEd), |
| udaq::noise\_model::get\_channels |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/l              | Pv4NO5cudaq15scalar_operatorplEv) |
| anguages/cpp_api.html#_CPPv4I0ENK | -   [cu                           |
| 5cudaq11noise_model12get_channels | daq::scalar\_operator::operator+= |
| ENSt6vectorI13kraus_channelEERKNS |     (C++                          |
| t6vectorINSt6size_tEEERKNSt6vecto |     function)](api/languag        |
| rINSt6size_tEEERKNSt6vectorIdEE), | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     [\[1\]](api/languages/cpp_a   | alar_operatorpLENSt7complexIdEE), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[1\]](api/languages/c       |
| el12get_channelsERKNSt6stringERKN | pp_api.html#_CPPv4N5cudaq15scalar |
| St6vectorINSt6size_tEEERKNSt6vect | _operatorpLERK15scalar_operator), |
| orINSt6size_tEEERKNSt6vectorIdEE) |     [\[2                          |
| -   [                             | \]](api/languages/cpp_api.html#_C |
| cudaq::noise\_model::noise\_model | PPv4N5cudaq15scalar_operatorpLEd) |
|     (C++                          | -   [c                            |
|     function)](api                | udaq::scalar\_operator::operator- |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq11noise_model11noise_modelEv) |     function                      |
| -   [cud                          | )](api/languages/cpp_api.html#_CP |
| aq::noise\_model::PredicateFuncTy | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     type)](api/la                 |     [\[1\                         |
| nguages/cpp_api.html#_CPPv4N5cuda | ]](api/languages/cpp_api.html#_CP |
| q11noise_model15PredicateFuncTyE) | Pv4N5cudaq15scalar_operatormiENSt |
| -   [cudaq                        | 7complexIdEERR15scalar_operator), |
| ::noise\_model::register\_channel |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     function)](api/languages      | operatormiEdRK15scalar_operator), |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     [\[3\]](api/languages/cp      |
| noise_model16register_channelEvv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::no                    | operatormiEdRR15scalar_operator), |
| ise\_model::requires\_constructor |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     type)](api/languages/cp       | alar_operatormiENSt7complexIdEE), |
| p_api.html#_CPPv4I0DpEN5cudaq11no |     [\[5\]](api/languages/cpp     |
| ise_model20requires_constructorE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::noise\_model\_type    | _operatormiERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     e                             | (api/languages/cpp_api.html#_CPPv |
| num)](api/languages/cpp_api.html# | 4NKR5cudaq15scalar_operatormiEd), |
| _CPPv4N5cudaq16noise_model_typeE) |     [\[7\]]                       |
| -   [cudaq::noise                 | (api/languages/cpp_api.html#_CPPv |
| \_model\_type::amplitude\_damping | 4NKR5cudaq15scalar_operatormiEv), |
|     (C++                          |     [\[8\]](api/language          |
|     enumerator)](api/languages    | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| /cpp_api.html#_CPPv4N5cudaq16nois | alar_operatormiENSt7complexIdEE), |
| e_model_type17amplitude_dampingE) |     [\[9\]](api/languages/cp      |
| -   [cudaq::noise\_model\_        | p_api.html#_CPPv4NO5cudaq15scalar |
| type::amplitude\_damping\_channel | _operatormiERK15scalar_operator), |
|     (C++                          |     [\[10\]                       |
|     e                             | ](api/languages/cpp_api.html#_CPP |
| numerator)](api/languages/cpp_api | v4NO5cudaq15scalar_operatormiEd), |
| .html#_CPPv4N5cudaq16noise_model_ |     [\[11\                        |
| type25amplitude_damping_channelE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise                 | Pv4NO5cudaq15scalar_operatormiEv) |
| \_model\_type::bit\_flip\_channel | -   [cu                           |
|     (C++                          | daq::scalar\_operator::operator-= |
|     enumerator)](api/language     |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     function)](api/languag        |
| se_model_type16bit_flip_channelE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::no                    | alar_operatormIENSt7complexIdEE), |
| ise\_model\_type::depolarization1 |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](api/languag      | _operatormIERK15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[2                          |
| ise_model_type15depolarization1E) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::no                    | PPv4N5cudaq15scalar_operatormIEd) |
| ise\_model\_type::depolarization2 | -   [c                            |
|     (C++                          | udaq::scalar\_operator::operator/ |
|     enumerator)](api/languag      |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16no |     function                      |
| ise_model_type15depolarization2E) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise\_mod            | Pv4N5cudaq15scalar_operatordvENSt |
| el\_type::depolarization\_channel | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
|   enumerator)](api/languages/cpp_ | Pv4N5cudaq15scalar_operatordvENSt |
| api.html#_CPPv4N5cudaq16noise_mod | 7complexIdEERR15scalar_operator), |
| el_type22depolarization_channelE) |     [\[2\]](api/languages/cp      |
| -   [                             | p_api.html#_CPPv4N5cudaq15scalar_ |
| cudaq::noise\_model\_type::pauli1 | operatordvEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     enumerator)](a                | p_api.html#_CPPv4N5cudaq15scalar_ |
| pi/languages/cpp_api.html#_CPPv4N | operatordvEdRR15scalar_operator), |
| 5cudaq16noise_model_type6pauli1E) |     [\[4\]](api/languages         |
| -   [                             | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| cudaq::noise\_model\_type::pauli2 | alar_operatordvENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     enumerator)](a                | _api.html#_CPPv4NKR5cudaq15scalar |
| pi/languages/cpp_api.html#_CPPv4N | _operatordvERK15scalar_operator), |
| 5cudaq16noise_model_type6pauli2E) |     [\[6\]]                       |
| -   [cudaq::n                     | (api/languages/cpp_api.html#_CPPv |
| oise\_model\_type::phase\_damping | 4NKR5cudaq15scalar_operatordvEd), |
|     (C++                          |     [\[7\]](api/language          |
|     enumerator)](api/langu        | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| ages/cpp_api.html#_CPPv4N5cudaq16 | alar_operatordvENSt7complexIdEE), |
| noise_model_type13phase_dampingE) |     [\[8\]](api/languages/cp      |
| -   [cudaq::noise\_               | p_api.html#_CPPv4NO5cudaq15scalar |
| model\_type::phase\_flip\_channel | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     enumerator)](api/languages/   | ]](api/languages/cpp_api.html#_CP |
| cpp_api.html#_CPPv4N5cudaq16noise | Pv4NO5cudaq15scalar_operatordvEd) |
| _model_type18phase_flip_channelE) | -   [cu                           |
| -   [c                            | daq::scalar\_operator::operator/= |
| udaq::noise\_model\_type::unknown |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](ap               | es/cpp_api.html#_CPPv4N5cudaq15sc |
| i/languages/cpp_api.html#_CPPv4N5 | alar_operatordVENSt7complexIdEE), |
| cudaq16noise_model_type7unknownE) |     [\[1\]](api/languages/c       |
| -   [cu                           | pp_api.html#_CPPv4N5cudaq15scalar |
| daq::noise\_model\_type::x\_error | _operatordVERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     enumerator)](ap               | \]](api/languages/cpp_api.html#_C |
| i/languages/cpp_api.html#_CPPv4N5 | PPv4N5cudaq15scalar_operatordVEd) |
| cudaq16noise_model_type7x_errorE) | -   [c                            |
| -   [cu                           | udaq::scalar\_operator::operator= |
| daq::noise\_model\_type::y\_error |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     enumerator)](ap               | pp_api.html#_CPPv4N5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatoraSERK15scalar_operator), |
| cudaq16noise_model_type7y_errorE) |     [\[1\]](api/languages/        |
| -   [cu                           | cpp_api.html#_CPPv4N5cudaq15scala |
| daq::noise\_model\_type::z\_error | r_operatoraSERR15scalar_operator) |
|     (C++                          | -   [cu                           |
|     enumerator)](ap               | daq::scalar\_operator::operator== |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7z_errorE) |     function)](api/languages/c    |
| -   [cudaq::num\_available\_gpus  | pp_api.html#_CPPv4NK5cudaq15scala |
|     (C++                          | r_operatoreqERK15scalar_operator) |
|     function                      | -   [cudaq::s                     |
| )](api/languages/cpp_api.html#_CP | calar\_operator::scalar\_operator |
| Pv4N5cudaq18num_available_gpusEv) |     (C++                          |
| -   [cudaq::observe (C++          |     func                          |
|     function)](                   | tion)](api/languages/cpp_api.html |
| api/languages/cpp_api.html#_CPPv4 | #_CPPv4N5cudaq15scalar_operator15 |
| I00Dp0EN5cudaq7observeENSt6vector | scalar_operatorENSt7complexIdEE), |
| I14observe_resultEERR13QuantumKer |     [\[1\]](api/langu             |
| nelRK15SpinOpContainerDpRR4Args), | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     [\[1\]](api/languages/cpp_api | scalar_operator15scalar_operatorE |
| .html#_CPPv4I0Dp0EN5cudaq7observe | RK15scalar_callbackRRNSt13unorder |
| E14observe_resultNSt6size_tERR13Q | ed_mapINSt6stringENSt6stringEEE), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[2\                         |
|     [\[2                          | ]](api/languages/cpp_api.html#_CP |
| \]](api/languages/cpp_api.html#_C | Pv4N5cudaq15scalar_operator15scal |
| PPv4I0Dp0EN5cudaq7observeE14obser | ar_operatorERK15scalar_operator), |
| ve_resultRK15observe_optionsRR13Q |     [\[3\]](api/langu             |
| uantumKernelRK7spin_opDpRR4Args), | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     [\[3\]](api/langu             | scalar_operator15scalar_operatorE |
| ages/cpp_api.html#_CPPv4I0Dp0EN5c | RR15scalar_callbackRRNSt13unorder |
| udaq7observeE14observe_resultRR13 | ed_mapINSt6stringENSt6stringEEE), |
| QuantumKernelRK7spin_opDpRR4Args) |     [\[4\                         |
| -   [cudaq::observe\_options (C++ | ]](api/languages/cpp_api.html#_CP |
|     st                            | Pv4N5cudaq15scalar_operator15scal |
| ruct)](api/languages/cpp_api.html | ar_operatorERR15scalar_operator), |
| #_CPPv4N5cudaq15observe_optionsE) |     [\[5\]](api/language          |
| -   [cudaq::observe\_result (C++  | s/cpp_api.html#_CPPv4N5cudaq15sca |
|                                   | lar_operator15scalar_operatorEd), |
| class)](api/languages/cpp_api.htm |     [\[6\]](api/languag           |
| l#_CPPv4N5cudaq14observe_resultE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -                                 | alar_operator15scalar_operatorEv) |
|   [cudaq::observe\_result::counts | -   [cu                           |
|     (C++                          | daq::scalar\_operator::to\_matrix |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4N5cudaq14observ |                                   |
| e_result6countsERK12spin_op_term) |   function)](api/languages/cpp_ap |
| -   [cudaq::observe\_result::dump | i.html#_CPPv4NK5cudaq15scalar_ope |
|     (C++                          | rator9to_matrixERKNSt13unordered_ |
|     function)                     | mapINSt6stringENSt7complexIdEEEE) |
| ](api/languages/cpp_api.html#_CPP | -   [cu                           |
| v4N5cudaq14observe_result4dumpEv) | daq::scalar\_operator::to\_string |
| -   [cu                           |     (C++                          |
| daq::observe\_result::expectation |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|                                   | daq15scalar_operator9to_stringEv) |
| function)](api/languages/cpp_api. | -   [cudaq::sca                   |
| html#_CPPv4N5cudaq14observe_resul | lar\_operator::\~scalar\_operator |
| t11expectationERK12spin_op_term), |     (C++                          |
|     [\[1\]](api/la                |     functio                       |
| nguages/cpp_api.html#_CPPv4N5cuda | n)](api/languages/cpp_api.html#_C |
| q14observe_result11expectationEv) | PPv4N5cudaq15scalar_operatorD0Ev) |
| -   [cudaq:                       | -   [cuda                         |
| :observe\_result::id\_coefficient | q::SerializedCodeExecutionContext |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     class)](api/lang              |
| ages/cpp_api.html#_CPPv4N5cudaq14 | uages/cpp_api.html#_CPPv4N5cudaq3 |
| observe_result14id_coefficientEv) | 0SerializedCodeExecutionContextE) |
| -   [cudaq:                       | -   [cudaq::set\_noise (C++       |
| :observe\_result::observe\_result |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq9s |
|                                   | et_noiseERKN5cudaq11noise_modelE) |
|   function)](api/languages/cpp_ap | -   [cudaq::set\_random\_seed     |
| i.html#_CPPv4N5cudaq14observe_res |     (C++                          |
| ult14observe_resultEdRK7spin_op), |     function)](api/               |
|     [\[1\]](a                     | languages/cpp_api.html#_CPPv4N5cu |
| pi/languages/cpp_api.html#_CPPv4N | daq15set_random_seedENSt6size_tE) |
| 5cudaq14observe_result14observe_r | -   [cudaq::simulation\_precision |
| esultEdRK7spin_op13sample_result) |     (C++                          |
| -                                 |     enum)                         |
| [cudaq::observe\_result::operator | ](api/languages/cpp_api.html#_CPP |
|     double (C++                   | v4N5cudaq20simulation_precisionE) |
|     functio                       | -   [c                            |
| n)](api/languages/cpp_api.html#_C | udaq::simulation\_precision::fp32 |
| PPv4N5cudaq14observe_resultcvdEv) |     (C++                          |
| -   [                             |     enumerator)](api              |
| cudaq::observe\_result::raw\_data | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq20simulation_precision4fp32E) |
|     function)](ap                 | -   [c                            |
| i/languages/cpp_api.html#_CPPv4N5 | udaq::simulation\_precision::fp64 |
| cudaq14observe_result8raw_dataEv) |     (C++                          |
| -   [cudaq::operator\_handler     |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     cl                            | udaq20simulation_precision4fp64E) |
| ass)](api/languages/cpp_api.html# | -   [cudaq::SimulationState (C++  |
| _CPPv4N5cudaq16operator_handlerE) |     c                             |
| -   [cudaq::optimizable\_function | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15SimulationStateE) |
|     class)                        | -   [                             |
| ](api/languages/cpp_api.html#_CPP | cudaq::SimulationState::precision |
| v4N5cudaq20optimizable_functionE) |     (C++                          |
| -   [cudaq::optimization\_result  |     enum)](api                    |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     type                          | udaq15SimulationState9precisionE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq:                       |
| Pv4N5cudaq19optimization_resultE) | :SimulationState::precision::fp32 |
| -   [cudaq::optimizer (C++        |     (C++                          |
|     class)](api/languages/cpp_a   |     enumerator)](api/lang         |
| pi.html#_CPPv4N5cudaq9optimizerE) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [cudaq::optimizer::optimize   | 5SimulationState9precision4fp32E) |
|     (C++                          | -   [cudaq:                       |
|                                   | :SimulationState::precision::fp64 |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4N5cudaq9optimizer8opt |     enumerator)](api/lang         |
| imizeEKiRR20optimizable_function) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [cu                           | 5SimulationState9precision4fp64E) |
| daq::optimizer::requiresGradients | -                                 |
|     (C++                          |   [cudaq::SimulationState::Tensor |
|     function)](api/la             |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     struct)](                     |
| q9optimizer17requiresGradientsEv) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::orca (C++             | N5cudaq15SimulationState6TensorE) |
|     type)](api/languages/         | -   [cudaq::spin\_handler (C++    |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |                                   |
| -   [cudaq::orca::sample (C++     |   class)](api/languages/cpp_api.h |
|     function)](api/languages/c    | tml#_CPPv4N5cudaq12spin_handlerE) |
| pp_api.html#_CPPv4N5cudaq4orca6sa | -   [cudaq::sp                    |
| mpleERNSt6vectorINSt6size_tEEERNS | in\_handler::to\_diagonal\_matrix |
| t6vectorINSt6size_tEEERNSt6vector |     (C++                          |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     function)](api/la             |
|     [\[1\]]                       | nguages/cpp_api.html#_CPPv4NK5cud |
| (api/languages/cpp_api.html#_CPPv | aq12spin_handler18to_diagonal_mat |
| 4N5cudaq4orca6sampleERNSt6vectorI | rixERNSt13unordered_mapINSt6size_ |
| NSt6size_tEEERNSt6vectorINSt6size | tENSt7int64_tEEERKNSt13unordered_ |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::orca::sample\_async   | -                                 |
|     (C++                          | [cudaq::spin\_handler::to\_matrix |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     function                      |
| html#_CPPv4N5cudaq4orca12sample_a | )](api/languages/cpp_api.html#_CP |
| syncERNSt6vectorINSt6size_tEEERNS | Pv4N5cudaq12spin_handler9to_matri |
| t6vectorINSt6size_tEEERNSt6vector | xERKNSt6stringENSt7complexIdEEb), |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     [\[1                          |
|     [\[1\]](api/la                | \]](api/languages/cpp_api.html#_C |
| nguages/cpp_api.html#_CPPv4N5cuda | PPv4NK5cudaq12spin_handler9to_mat |
| q4orca12sample_asyncERNSt6vectorI | rixERNSt13unordered_mapINSt6size_ |
| NSt6size_tEEERNSt6vectorINSt6size | tENSt7int64_tEEERKNSt13unordered_ |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::OrcaRemoteRESTQPU     | -   [cudaq::                      |
|     (C++                          | spin\_handler::to\_sparse\_matrix |
|     cla                           |     (C++                          |
| ss)](api/languages/cpp_api.html#_ |     function)](api/               |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | languages/cpp_api.html#_CPPv4N5cu |
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
| -   [define() (cudaq.operators    | -   [deserialize()                |
|     method)](api/languages/python |     (cudaq.SampleResult           |
| _api.html#cudaq.operators.define) |     meth                          |
|     -   [(cuda                    | od)](api/languages/python_api.htm |
| q.operators.MatrixOperatorElement | l#cudaq.SampleResult.deserialize) |
|         class                     | -   [displace() (in module        |
|         method)](api/langu        |     cudaq.operators.custo         |
| ages/python_api.html#cudaq.operat | m)](api/languages/python_api.html |
| ors.MatrixOperatorElement.define) | #cudaq.operators.custom.displace) |
|     -   [(in module               | -   [distribute\_terms()          |
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
| -   [Depolarization1 (class in    | perators.MatrixOperatorTerm.dump) |
|     cudaq)](api/languages/pytho   |     -   [(                        |
| n_api.html#cudaq.Depolarization1) | cudaq.operators.spin.SpinOperator |
| -   [Depolarization2 (class in    |         method)](api              |
|     cudaq)](api/languages/pytho   | /languages/python_api.html#cudaq. |
| n_api.html#cudaq.Depolarization2) | operators.spin.SpinOperator.dump) |
| -   [DepolarizationChannel (class |     -   [(cuda                    |
|     in                            | q.operators.spin.SpinOperatorTerm |
|                                   |         method)](api/lan          |
| cudaq)](api/languages/python_api. | guages/python_api.html#cudaq.oper |
| html#cudaq.DepolarizationChannel) | ators.spin.SpinOperatorTerm.dump) |
| -   [description (cudaq.Target    |     -   [(cudaq.SampleResult      |
|                                   |                                   |
| property)](api/languages/python_a |    method)](api/languages/python_ |
| pi.html#cudaq.Target.description) | api.html#cudaq.SampleResult.dump) |
|                                   |     -   [(cudaq.State             |
|                                   |         method)](api/languages/   |
|                                   | python_api.html#cudaq.State.dump) |
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
| -   [evaluate()                   | pi.html#cudaq.operators.MatrixOpe |
|                                   | ratorElement.expected_dimensions) |
|   (cudaq.operators.ScalarOperator | -                                 |
|     method)](api/                 |  [extract\_c\_function\_pointer() |
| languages/python_api.html#cudaq.o |     (cudaq.PyKernelDecorator      |
| perators.ScalarOperator.evaluate) |     method)](api/languages/p      |
|                                   | ython_api.html#cudaq.PyKernelDeco |
|                                   | rator.extract_c_function_pointer) |
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
| -   [get()                        | -   [get\_register\_counts()      |
|     (cudaq.AsyncEvolveResult      |     (cudaq.SampleResult           |
|     m                             |     method)](api                  |
| ethod)](api/languages/python_api. | /languages/python_api.html#cudaq. |
| html#cudaq.AsyncEvolveResult.get) | SampleResult.get_register_counts) |
|                                   | -   [get\_sequential\_data()      |
|    -   [(cudaq.AsyncObserveResult |     (cudaq.SampleResult           |
|         me                        |     method)](api                  |
| thod)](api/languages/python_api.h | /languages/python_api.html#cudaq. |
| tml#cudaq.AsyncObserveResult.get) | SampleResult.get_sequential_data) |
|     -   [(cudaq.AsyncSampleResult | -   [get\_spin()                  |
|         m                         |     (cudaq.ObserveResult          |
| ethod)](api/languages/python_api. |     me                            |
| html#cudaq.AsyncSampleResult.get) | thod)](api/languages/python_api.h |
|     -   [(cudaq.AsyncStateResult  | tml#cudaq.ObserveResult.get_spin) |
|                                   | -   [get\_state() (in module      |
| method)](api/languages/python_api |     cudaq)](api/languages         |
| .html#cudaq.AsyncStateResult.get) | /python_api.html#cudaq.get_state) |
| -                                 | -   [get\_state\_async() (in      |
|  [get\_binary\_symplectic\_form() |     module                        |
|     (cuda                         |     cudaq)](api/languages/pytho   |
| q.operators.spin.SpinOperatorTerm | n_api.html#cudaq.get_state_async) |
|     metho                         | -   [get\_target() (in module     |
| d)](api/languages/python_api.html |     cudaq)](api/languages/        |
| #cudaq.operators.spin.SpinOperato | python_api.html#cudaq.get_target) |
| rTerm.get_binary_symplectic_form) | -   [get\_targets() (in module    |
| -   [get\_channels()              |     cudaq)](api/languages/p       |
|     (cudaq.NoiseModel             | ython_api.html#cudaq.get_targets) |
|     met                           | -   [get\_term\_count()           |
| hod)](api/languages/python_api.ht |     (                             |
| ml#cudaq.NoiseModel.get_channels) | cudaq.operators.spin.SpinOperator |
| -   [get\_coefficient()           |     method)](api/languages        |
|     (                             | /python_api.html#cudaq.operators. |
| cudaq.operators.spin.SpinOperator | spin.SpinOperator.get_term_count) |
|     method)](api/languages/       | -   [get\_total\_shots()          |
| python_api.html#cudaq.operators.s |     (cudaq.SampleResult           |
| pin.SpinOperator.get_coefficient) |     method)]                      |
|     -   [(cuda                    | (api/languages/python_api.html#cu |
| q.operators.spin.SpinOperatorTerm | daq.SampleResult.get_total_shots) |
|                                   | -   [getTensor() (cudaq.State     |
|       method)](api/languages/pyth |     method)](api/languages/pytho  |
| on_api.html#cudaq.operators.spin. | n_api.html#cudaq.State.getTensor) |
| SpinOperatorTerm.get_coefficient) | -   [getTensors() (cudaq.State    |
| -   [get\_marginal\_counts()      |     method)](api/languages/python |
|     (cudaq.SampleResult           | _api.html#cudaq.State.getTensors) |
|     method)](api                  | -   [gradient (class in           |
| /languages/python_api.html#cudaq. |     cudaq.g                       |
| SampleResult.get_marginal_counts) | radients)](api/languages/python_a |
| -   [get\_pauli\_word()           | pi.html#cudaq.gradients.gradient) |
|     (cuda                         | -   [GradientDescent (class in    |
| q.operators.spin.SpinOperatorTerm |     cudaq.optimizers              |
|     method)](api/languages/pyt    | )](api/languages/python_api.html# |
| hon_api.html#cudaq.operators.spin | cudaq.optimizers.GradientDescent) |
| .SpinOperatorTerm.get_pauli_word) |                                   |
| -   [get\_precision()             |                                   |
|     (cudaq.Target                 |                                   |
|                                   |                                   |
| method)](api/languages/python_api |                                   |
| .html#cudaq.Target.get_precision) |                                   |
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
| -   [get\_raw\_data()             |                                   |
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
|     static                        | -   [right\_multiply()            |
|     method)](api/l                |     (cudaq.SuperOperator static   |
| anguages/python_api.html#cudaq.op |     method)]                      |
| erators.spin.SpinOperator.random) | (api/languages/python_api.html#cu |
| -   [rank() (in module            | daq.SuperOperator.right_multiply) |
|     cudaq.mpi)](api/language      | -   [row\_count                   |
| s/python_api.html#cudaq.mpi.rank) |     (cudaq.KrausOperator          |
| -   [register\_names              |     prope                         |
|     (cudaq.SampleResult           | rty)](api/languages/python_api.ht |
|     attribute)                    | ml#cudaq.KrausOperator.row_count) |
| ](api/languages/python_api.html#c | -   [run() (in module             |
| udaq.SampleResult.register_names) |     cudaq)](api/lan               |
| -   [requires\_gradients()        | guages/python_api.html#cudaq.run) |
|     (cudaq.optimizers.COBYLA      | -   [run\_async() (in module      |
|     method)](api/lan              |     cudaq)](api/languages         |
| guages/python_api.html#cudaq.opti | /python_api.html#cudaq.run_async) |
| mizers.COBYLA.requires_gradients) | -   [RydbergHamiltonian (class in |
|     -   [                         |     cudaq.operators)]             |
| (cudaq.optimizers.GradientDescent | (api/languages/python_api.html#cu |
|         method)](api/languages/py | daq.operators.RydbergHamiltonian) |
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
