::: {.wy-grid-for-nav}
::: {.wy-side-scroll}
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../index.html){.icon .icon-home}

::: {.version}
pr-3308
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](quick_start.html){.reference .internal}
    -   [Install CUDA-Q](quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q Academic](quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q Kernel?](basics/kernel_intro.html){.reference
        .internal}
    -   [Building your first CUDA-Q
        Program](basics/build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q
        Program](basics/run_kernel.html){.reference .internal}
        -   [Sample](basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](basics/run_kernel.html#run){.reference .internal}
        -   [Observe](basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
-   [Examples](examples/examples.html){.reference .internal}
    -   [Introduction](examples/introduction.html){.reference .internal}
    -   [Building Kernels](examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring Kernels](examples/measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](../examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](../examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](../examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing Kernels](examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](examples/expectation_values.html){.reference .internal}
        -   [Parallelizing across Multiple
            Processors](examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs (`mgpu`{.code .docutils
            .literal
            .notranslate})](examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs (`mqpu`{.code
            .docutils .literal
            .notranslate})](examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends (`remote-mqpu`{.code .docutils
            .literal
            .notranslate})](examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](../examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [Built in CUDA-Q Optimizers and
            Gradients](../examples/python/optimizers_gradients.html#Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
        -   [Third-Party
            Optimizers](../examples/python/optimizers_gradients.html#Third-Party-Optimizers){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../examples/python/optimizers_gradients.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](../examples/python/noisy_simulations.html){.reference
        .internal}
    -   [Constructing Operators](examples/operators.html){.reference
        .internal}
        -   [Constructing Spin
            Operators](examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Infleqtion](examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [QuEra
            Computing](examples/hardware_providers.html#quera-computing){.reference
            .internal}
    -   [Dynamics Examples](examples/dynamics_examples.html){.reference
        .internal}
        -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
            Model)](../examples/python/dynamics/dynamics_intro_1.html){.reference
            .internal}
            -   [Why dynamics simulations vs. circuit
                simulations?](../examples/python/dynamics/dynamics_intro_1.html#Why-dynamics-simulations-vs.-circuit-simulations?){.reference
                .internal}
            -   [Functionality](../examples/python/dynamics/dynamics_intro_1.html#Functionality){.reference
                .internal}
            -   [Performance](../examples/python/dynamics/dynamics_intro_1.html#Performance){.reference
                .internal}
            -   [Section 1 - Simulating the Jaynes-Cummings
                Hamiltonian](../examples/python/dynamics/dynamics_intro_1.html#Section-1---Simulating-the-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Exercise 1 - Simulating a many-photon Jaynes-Cummings
                Hamiltonian](../examples/python/dynamics/dynamics_intro_1.html#Exercise-1---Simulating-a-many-photon-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Section 2 - Simulating open quantum systems with the
                `collapse_operators`{.docutils .literal
                .notranslate}](../examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L\_i\\)]{.math .notranslate
                .nohighlight}](../examples/python/dynamics/dynamics_intro_1.html#Exercise-2---Adding-additional-jump-operators-L_i){.reference
                .internal}
            -   [Section 3 - Many qubits coupled to the
                resonator](../examples/python/dynamics/dynamics_intro_1.html#Section-3---Many-qubits-coupled-to-the-resonator){.reference
                .internal}
        -   [Introduction to CUDA-Q Dynamics (Time Dependent
            Hamiltonians)](../examples/python/dynamics/dynamics_intro_2.html){.reference
            .internal}
            -   [The Landau-Zener
                model](../examples/python/dynamics/dynamics_intro_2.html#The-Landau-Zener-model){.reference
                .internal}
            -   [Section 1 - Implementing time dependent
                terms](../examples/python/dynamics/dynamics_intro_2.html#Section-1---Implementing-time-dependent-terms){.reference
                .internal}
            -   [Section 2 - Implementing custom
                operators](../examples/python/dynamics/dynamics_intro_2.html#Section-2---Implementing-custom-operators){.reference
                .internal}
            -   [Section 3 - Heisenberg Model with a time-varying
                magnetic
                field](../examples/python/dynamics/dynamics_intro_2.html#Section-3---Heisenberg-Model-with-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 1 - Define a time-varying magnetic
                field](../examples/python/dynamics/dynamics_intro_2.html#Exercise-1---Define-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 2
                (Optional)](../examples/python/dynamics/dynamics_intro_2.html#Exercise-2-(Optional)){.reference
                .internal}
        -   [Superconducting
            Qubits](../examples/python/dynamics/superconducting.html){.reference
            .internal}
            -   [Cavity
                QED](../examples/python/dynamics/superconducting.html#Cavity-QED){.reference
                .internal}
            -   [Cross
                Resonance](../examples/python/dynamics/superconducting.html#Cross-Resonance){.reference
                .internal}
            -   [Transmon
                Resonator](../examples/python/dynamics/superconducting.html#Transmon-Resonator){.reference
                .internal}
        -   [Spin
            Qubits](../examples/python/dynamics/spinqubits.html){.reference
            .internal}
            -   [Silicon Spin
                Qubit](../examples/python/dynamics/spinqubits.html#Silicon-Spin-Qubit){.reference
                .internal}
            -   [Heisenberg
                Model](../examples/python/dynamics/spinqubits.html#Heisenberg-Model){.reference
                .internal}
        -   [Trapped Ion
            Qubits](../examples/python/dynamics/iontrap.html){.reference
            .internal}
            -   [GHZ
                state](../examples/python/dynamics/iontrap.html#GHZ-state){.reference
                .internal}
        -   [Control](../examples/python/dynamics/control.html){.reference
            .internal}
            -   [Gate
                Calibration](../examples/python/dynamics/control.html#Gate-Calibration){.reference
                .internal}
            -   [Pulse](../examples/python/dynamics/control.html#Pulse){.reference
                .internal}
            -   [Qubit
                Control](../examples/python/dynamics/control.html#Qubit-Control){.reference
                .internal}
            -   [Qubit
                Dynamics](../examples/python/dynamics/control.html#Qubit-Dynamics){.reference
                .internal}
            -   [Landau-Zenner](../examples/python/dynamics/control.html#Landau-Zenner){.reference
                .internal}
-   [Applications](applications.html){.reference .internal}
    -   [Max-Cut with QAOA](../applications/python/qaoa.html){.reference
        .internal}
    -   [Molecular docking via
        DC-QAOA](../applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](../applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](../applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H\_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](../applications/python/krylov.html){.reference
        .internal}
        -   [Setup](../applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](../applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](../applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Bernstein-Vazirani
        Algorithm](../applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](../applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](../applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](../applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](../applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](../applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](../applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](../applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](../applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](../applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](../applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](../applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](../applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](../applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](../applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](../applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](../applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](../applications/python/hybrid_quantum_neural_networks.html){.reference
        .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](../applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](../applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using `Sample`{.docutils .literal .notranslate} to perform
            the Hadamard
            test](../applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](../applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](../applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](../applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](../applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the `[[4,2,2]]`{.docutils .literal
            .notranslate}
            encoding](../applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](../applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](../applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](../applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
            .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](../applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](../applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](../applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](../applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](../applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](../applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](../applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](../applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. `Compute overlap`{.docutils .literal
                .notranslate}](../applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](../applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](../applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](../applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](../applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](../applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](../applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](../applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](../applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](../applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum Fourier
        Transform](../applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](../applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](../applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](../applications/python/quantum_teleportation.html#Teleportation-explained){.reference
            .internal}
    -   [Quantum
        Volume](../applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](../applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](../applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Compiling Unitaries Using Diffusion
        Models](../applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](../applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](../applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](../applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](../applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](../applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](../applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](../applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](../applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](../applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](../applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](../applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](../applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](../applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](../applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](../applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](../applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](../applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](../applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
            .internal}
    -   [Quantum
        Transformer](../applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](../applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](../applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](../applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](../applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](../applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](../applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](../applications/python/quantum_transformer.html#Attention-Maps){.reference
                .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](../applications/python/afqmc.html){.reference .internal}
        -   [Hamiltonian preparation for
            VQE](../applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](../applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](../applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](../applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](../applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](../applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [ADAPT-QAOA
        algorithm](../applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](../applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H\_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](../applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A\_j\\)]{.math .notranslate
            .nohighlight}:](../applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H\_C,A\_j\]\\)]{.math .notranslate
            .nohighlight}:](../applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](../applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](../applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](../applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](../applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](../applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](../applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](../applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A\_i\\)]{.math .notranslate
            .nohighlight}\]](../applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](../applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](../applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](../applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](../applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](../applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](../applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](../applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](../applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](../applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](../applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](../applications/python/edge_detection.html#Post-processing){.reference
                .internal}
    -   [Factoring Integers With Shor's
        Algorithm](../applications/python/shors.html){.reference
        .internal}
        -   [Shor's
            algorithm](../applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](../applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](../applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](../applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](../applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](../applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](../applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](../applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
        -   [(a) Generate the molecular Hamiltonian using Hartree Fock
            molecular
            orbitals](../applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Hartree-Fock-molecular-orbitals){.reference
            .internal}
            -   [Active space
                Hamiltonian:](../applications/python/generate_fermionic_ham.html#Active-space-Hamiltonian:){.reference
                .internal}
        -   [(b) Generate the active space hamiltonian using HF
            molecular
            orbitals.](../applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-hamiltonian-using-HF-molecular-orbitals.){.reference
            .internal}
        -   [(c) Generate the active space Hamiltonian using the natural
            orbitals computed from MP2
            simulation](../applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
            .internal}
        -   [(d) Generate the active space Hamiltonian computed from the
            CASSCF molecular
            orbitals](../applications/python/generate_fermionic_ham.html#(d)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
            .internal}
            -   [Generate the electronic Hamiltonian using
                ROHF](../applications/python/generate_fermionic_ham.html#Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
    -   [Grover's
        Algorithm](../applications/python/grovers.html){.reference
        .internal}
        -   [Overview](../applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](../applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](../applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](../applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](../applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](../applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](../applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](../applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](../applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](../applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](../applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](../applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
            .internal}
    -   [The UCCSD Wavefunction
        ansatz](../applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](../applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](../applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](../applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](../applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](../applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](../applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](../applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](../applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](../applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](../applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](../applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](../applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](../applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](../applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](../applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](../applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](../applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](../applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
                .internal}
-   [Backends](backends/backends.html){.reference .internal}
    -   [Circuit Simulation](backends/simulators.html){.reference
        .internal}
        -   [State Vector
            Simulators](backends/sims/svsims.html){.reference .internal}
            -   [CPU](backends/sims/svsims.html#cpu){.reference
                .internal}
            -   [Single-GPU](backends/sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](backends/sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network
            Simulators](backends/sims/tnsims.html){.reference .internal}
            -   [Multi-GPU
                multi-node](backends/sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](backends/sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](backends/sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU
            Simulators](backends/sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](backends/sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](backends/sims/mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy Simulators](backends/sims/noisy.html){.reference
            .internal}
            -   [Trajectory Noisy
                Simulation](backends/sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](backends/sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](backends/sims/noisy.html#stim){.reference
                .internal}
        -   [Photonics
            Simulators](backends/sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](backends/sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware (QPUs)](backends/hardware.html){.reference
        .internal}
        -   [Ion Trap QPUs](backends/hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](backends/hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](backends/hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](backends/hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](backends/hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](backends/hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](backends/hardware/superconducting.html#oqc){.reference
                .internal}
        -   [Neutral Atom
            QPUs](backends/hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](backends/hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](backends/hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](backends/hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic QPUs](backends/hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](backends/hardware/photonic.html#orca-computing){.reference
                .internal}
    -   [Dynamics
        Simulation](backends/dynamics_backends.html){.reference
        .internal}
    -   [Cloud](backends/cloud.html){.reference .internal}
        -   [Amazon Braket
            (braket)](backends/cloud/braket.html){.reference .internal}
            -   [Setting
                Credentials](backends/cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submission from
                C++](backends/cloud/braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](backends/cloud/braket.html#submission-from-python){.reference
                .internal}
        -   [NVIDIA Quantum Cloud
            (nvqc)](backends/cloud/nvqc.html){.reference .internal}
            -   [Quick
                Start](backends/cloud/nvqc.html#quick-start){.reference
                .internal}
            -   [Simulator Backend
                Selection](backends/cloud/nvqc.html#simulator-backend-selection){.reference
                .internal}
            -   [Multiple
                GPUs](backends/cloud/nvqc.html#multiple-gpus){.reference
                .internal}
            -   [Multiple QPUs Asynchronous
                Execution](backends/cloud/nvqc.html#multiple-qpus-asynchronous-execution){.reference
                .internal}
            -   [FAQ](backends/cloud/nvqc.html#faq){.reference
                .internal}
-   [Dynamics](#){.current .reference .internal}
    -   [Quick Start](#quick-start){.reference .internal}
    -   [Operator](#operator){.reference .internal}
    -   [Time-Dependent Dynamics](#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](#super-operator-representation){.reference
        .internal}
    -   [Numerical Integrators](#numerical-integrators){.reference
        .internal}
    -   [Batch simulation](#batch-simulation){.reference .internal}
    -   [Multi-GPU Multi-Node
        Execution](#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](#examples){.reference .internal}
-   [CUDA-QX](cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q Solvers](cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](cudaqx/cudaqx.html#cuda-q-qec){.reference
        .internal}
-   [Installation](install/install.html){.reference .internal}
    -   [Local Installation](install/local_installation.html){.reference
        .internal}
        -   [Introduction](install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next
            Steps](install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](integration/integration.html){.reference .internal}
    -   [Downstream CMake
        Integration](integration/cmake_app.html){.reference .internal}
    -   [Combining CUDA with
        CUDA-Q](integration/cuda_gpu.html){.reference .internal}
    -   [Integrating with Third-Party
        Libraries](integration/libraries.html){.reference .internal}
        -   [Calling a CUDA-Q library from
            C++](integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](extending/extending.html){.reference .internal}
    -   [Add a new Hardware Backend](extending/backend.html){.reference
        .internal}
        -   [Overview](extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](extending/backend.html#server-helper-class){.reference
                .internal}
            -   [`CMakeLists.txt`{.docutils .literal
                .notranslate}](extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent `CMakeLists.txt`{.docutils .literal
                .notranslate}](extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](extending/backend.html#example-usage){.reference
            .internal}
        -   [Code Review](extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](extending/nvqir_simulator.html){.reference .internal}
        -   [`CircuitSimulator`{.code .docutils .literal
            .notranslate}](extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q IR](extending/cudaq_ir.html){.reference
        .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](extending/mlir_pass.html){.reference .internal}
-   [Specifications](../specification/index.html){.reference .internal}
    -   [Language Specification](../specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](../specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](../specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](../specification/cudaq/types.html){.reference
            .internal}
            -   [3.1. `cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}](../specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. `cudaq::qubit`{.code .docutils .literal
                .notranslate}](../specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](../specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](../specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. `cudaq::spin_op`{.code .docutils .literal
                .notranslate}](../specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](../specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on `cudaq::qubit`{.code .docutils
                .literal
                .notranslate}](../specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](../specification/cudaq/kernels.html){.reference
            .internal}
        -   [7. Sub-circuit
            Synthesis](../specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](../specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](../specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](../specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](../specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11.
            Platform](../specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](../specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. `cudaq::sample`{.code .docutils .literal
                .notranslate}](../specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. `cudaq::run`{.code .docutils .literal
                .notranslate}](../specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. `cudaq::observe`{.code .docutils .literal
                .notranslate}](../specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. `cudaq::optimizer`{.code .docutils .literal
                .notranslate} (deprecated, functionality moved to CUDA-Q
                libraries)](../specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. `cudaq::gradient`{.code .docutils .literal
                .notranslate} (deprecated, functionality moved to CUDA-Q
                libraries)](../specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](../specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](../specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](../specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](../specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](../specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](../specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](../specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](../specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](../specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](../specification/quake-dialect.html#motivation){.reference
            .internal}
-   [API Reference](../api/api.html){.reference .internal}
    -   [C++ API](../api/languages/cpp_api.html){.reference .internal}
        -   [Operators](../api/languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](../api/languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](../api/languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](../api/languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](../api/languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](../api/languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Platform](../api/languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](../api/languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](../api/languages/cpp_api.html#namespaces){.reference
            .internal}
    -   [Python API](../api/languages/python_api.html){.reference
        .internal}
        -   [Program
            Construction](../api/languages/python_api.html#program-construction){.reference
            .internal}
            -   [`make_kernel()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [`PyKernel`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [`Kernel`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [`PyKernelDecorator`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [`kernel()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](../api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [`sample()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [`sample_async()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [`run()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [`run_async()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [`observe()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [`observe_async()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [`get_state()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [`get_state_async()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [`vqe()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [`draw()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [`translate()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.translate){.reference
                .internal}
        -   [Backend
            Configuration](../api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [`has_target()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [`get_target()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [`get_targets()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [`set_target()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [`reset_target()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [`set_noise()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [`unset_noise()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [`cudaq.apply_noise()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [`initialize_cudaq()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [`num_available_gpus()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [`set_random_seed()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](../api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [`evolve()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [`evolve_async()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [`Schedule`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [`BaseIntegrator`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [`InitialState`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [`InitialStateType`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [`IntermediateResultSave`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](../api/languages/python_api.html#operators){.reference
            .internal}
            -   [`OperatorSum`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [`ProductOperator`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [`ElementaryOperator`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [`ScalarOperator`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [`RydbergHamiltonian`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [`SuperOperator`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [`operators.define()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [`operators.instantiate()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](../api/languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](../api/languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](../api/languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](../api/languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](../api/languages/python_api.html#data-types){.reference
            .internal}
            -   [`SimulationPrecision`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [`Target`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [`State`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [`Tensor`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [`QuakeValue`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [`qubit`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [`qreg`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [`qvector`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [`ComplexMatrix`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [`SampleResult`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [`AsyncSampleResult`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [`ObserveResult`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [`AsyncObserveResult`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [`AsyncStateResult`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [`OptimizationResult`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [`EvolveResult`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [`AsyncEvolveResult`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [Optimizers](../api/languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](../api/languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](../api/languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](../api/languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [`initialize()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [`rank()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [`num_ranks()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [`all_gather()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [`broadcast()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [`is_initialized()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [`finalize()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](../api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [`sample()`{.docutils .literal
                .notranslate}](../api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
    -   [Quantum Operations](../api/default_ops.html){.reference
        .internal}
        -   [Unitary Operations on
            Qubits](../api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [`x`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#x){.reference
                .internal}
            -   [`y`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#y){.reference
                .internal}
            -   [`z`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#z){.reference
                .internal}
            -   [`h`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#h){.reference
                .internal}
            -   [`r1`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#r1){.reference
                .internal}
            -   [`rx`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#rx){.reference
                .internal}
            -   [`ry`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#ry){.reference
                .internal}
            -   [`rz`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#rz){.reference
                .internal}
            -   [`s`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#s){.reference
                .internal}
            -   [`t`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#t){.reference
                .internal}
            -   [`swap`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#swap){.reference
                .internal}
            -   [`u3`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](../api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](../api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [`mz`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#mz){.reference
                .internal}
            -   [`mx`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#mx){.reference
                .internal}
            -   [`my`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](../api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](../api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [`create`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#create){.reference
                .internal}
            -   [`annihilate`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#annihilate){.reference
                .internal}
            -   [`phase_shift`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#phase-shift){.reference
                .internal}
            -   [`beam_splitter`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [`mz`{.code .docutils .literal
                .notranslate}](../api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](../versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](../index.html)

::: {.wy-nav-content}
::: {.rst-content}
::: {role="navigation" aria-label="Page navigation"}
-   [](../index.html){.icon .icon-home}
-   Dynamics Simulation
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](backends/cloud/nvqc.html "NVIDIA Quantum Cloud"){.btn
.btn-neutral .float-left} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](cudaqx/cudaqx.html "CUDA-QX"){.btn .btn-neutral
.float-right}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#dynamics-simulation .section}
Dynamics Simulation[¶](#dynamics-simulation "Permalink to this heading"){.headerlink}
=====================================================================================

CUDA-Q enables the design, simulation and execution of quantum dynamics
via the `evolve`{.docutils .literal .notranslate} API. Specifically,
this API allows us to solve the time evolution of quantum systems or
models. In the simulation mode, CUDA-Q provides the `dynamics`{.docutils
.literal .notranslate} backend target, which is based on the cuQuantum
library, optimized for performance and scale on NVIDIA GPU.

::: {#quick-start .section}
Quick Start[¶](#quick-start "Permalink to this heading"){.headerlink}
---------------------------------------------------------------------

In the example below, we demonstrate a simple time evolution simulation
workflow comprising of the following steps:

1.  Define a quantum system model

A quantum system model is defined by a Hamiltonian. For example, a
superconducting
[transmon](https://en.wikipedia.org/wiki/Transmon){.reference .external}
qubit can be modeled by the following Hamiltonian

::: {.math .notranslate .nohighlight}
\\\[H = \\frac{\\omega\_z}{2} \\sigma\_z + \\omega\_x \\cos(\\omega\_d
t)\\sigma\_x,\\\]
:::

where [\\(\\sigma\_z\\)]{.math .notranslate .nohighlight} and
[\\(\\sigma\_x\\)]{.math .notranslate .nohighlight} are Pauli Z and X
operators, respectively.

Using CUDA-Q `operator`{.code .docutils .literal .notranslate}, the
above time-dependent Hamiltonian can be set up as follows.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    omega_z = 6.5
    omega_x = 4.0
    omega_d = 0.5

    import numpy as np
    from cudaq import spin, ScalarOperator

    # Qubit Hamiltonian
    hamiltonian = 0.5 * omega_z * spin.z(0)
    # Add modulated driving term to the Hamiltonian
    hamiltonian += omega_x * ScalarOperator(lambda t: np.cos(omega_d * t)) * spin.x(
        0)
:::
:::

In particular, `ScalarOperator`{.code .docutils .literal .notranslate}
provides an easy way to model arbitrary time-dependent control signals.
Details about CUDA-Q `operator`{.code .docutils .literal .notranslate},
including builtin operators that it supports can be found [[here]{.std
.std-ref}](#operators){.reference .internal}.
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
        // Parameters
        double omega_z = 6.5;
        double omega_x = 4.0;
        double omega_d = 0.5;

        // Qubit Hamiltonian
        auto hamiltonian = spin_op(0.5 * omega_z * spin_op::z(0));

        // Time dependent modulation
        auto mod_func =
            [omega_d](const parameter_map &params) -> std::complex<double> {
          auto it = params.find("t");
          if (it != params.end()) {
            double t = it->second.real();
            const auto result = std::cos(omega_d * t);
            return result;
          }
          throw std::runtime_error("Cannot find the time parameter.");
        };

        hamiltonian += mod_func * spin_op::x(0) * omega_x;
:::
:::

Details about CUDA-Q `operator`{.code .docutils .literal .notranslate},
including builtin operators that it supports can be found [[here]{.std
.std-ref}](#operators){.reference .internal}.
:::
:::

2.  Setup the evolution simulation

The below code snippet shows how to simulate the time-evolution of the
above system with `cudaq.evolve`{.code .docutils .literal .notranslate}.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    import cudaq
    import cupy as cp
    from cudaq.dynamics import Schedule

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Dimensions of sub-systems: a single two-level system.
    dimensions = {0: 2}

    # Initial state of the system (ground state).
    rho0 = cudaq.State.from_data(
        cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

    # Schedule of time steps.
    steps = np.linspace(0, t_final, n_steps)
    schedule = Schedule(steps, ["t"])

    # Run the simulation.
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.x(0), spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.ALL)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
        double t_final = 1.0;
        int n_steps = 100;

        // Define dimensions of subsystem (single two-level system)
        cudaq::dimension_map dimensions = {{0, 2}};

        // Initial state (ground state)
        auto psi0 =
            cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

        // Schedule of time steps
        std::vector<double> steps(n_steps);
        for (int i = 0; i < n_steps; i++)
          steps[i] = i * t_final / (n_steps - 1);

        schedule schedule(steps, {"t"});

        // Numerical integrator
        // Here we choose a Runge-`Kutta` method for time evolution.
        cudaq::integrators::runge_kutta integrator(4, 0.01);

        // Observables to track
        auto observables = {cudaq::spin_op::x(0), cudaq::spin_op::y(0),
                            cudaq::spin_op::z(0)};

        // Run simulation
        // We evolve the system under the defined Hamiltonian. No collapsed
        // operators are provided (closed system evolution). The evolution returns
        // expectation values for all defined observables at each time step.
        auto evolution_result =
            cudaq::evolve(hamiltonian, dimensions, schedule, psi0, integrator, {},
                          observables, cudaq::IntermediateResultSave::All);
:::
:::
:::
:::

Specifically, we need to set up the simulation by providing:

-   The system model in terms of a Hamiltonian as well as any
    decoherence terms, so-called `collapse_operators`{.code .docutils
    .literal .notranslate}.

-   The dimensionality of component systems in the model. CUDA-Q
    `evolve`{.code .docutils .literal .notranslate} allows users to
    model arbitrary multi-level systems, such as photonic Fock space.

-   The initial quantum state.

-   The time schedule, aka time steps, of the evolution.

-   Any 'observable' operator that we want to measure the expectation
    value with respect to the evolving state.

::: {.admonition .note}
Note

By default, `evolve`{.code .docutils .literal .notranslate} will only
return the final state and expectation values. To save intermediate
results (at each time step specified in the schedule), the
`store_intermediate_results`{.code .docutils .literal .notranslate} flag
must be set to `True`{.code .docutils .literal .notranslate}.
:::

3.  Retrieve and plot the results

Once the simulation is complete, we can retrieve the final state and the
expectation values as well as intermediate values at each time step
(with
`store_intermediate_results=cudaq.IntermediateResultSave.ALL`{.code
.docutils .literal .notranslate}).

::: {.admonition .note}
Note

Storing intermediate states can be memory-intensive, especially for
large systems. If you only need the intermediate expectation values, you
can set `store_intermediate_results`{.code .docutils .literal
.notranslate} to `cudaq.IntermediateResultSave.EXPECTATION_VALUES`{.code
.docutils .literal .notranslate} (Python) /
`cudaq::IntermediateResultSave::ExpectationValue`{.code .docutils
.literal .notranslate} (C++) instead.
:::

For example, we can plot the Pauli expectation value for the above
simulation as follows.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]

    import matplotlib.pyplot as plt

    plt.plot(steps, get_result(0, evolution_result))
    plt.plot(steps, get_result(1, evolution_result))
    plt.plot(steps, get_result(2, evolution_result))
    plt.ylabel("Expectation value")
    plt.xlabel("Time")
    plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
:::
:::

In particular, for each time step, `evolve`{.code .docutils .literal
.notranslate} captures an array of expectation values, one for each
observable. Hence, we convert them into sequences for plotting purposes.
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
        // Extract and print results
        for (size_t i = 0; i < steps.size(); i++) {
          double ex =
              evolution_result.expectation_values.value()[0][i].expectation();
          double ey =
              evolution_result.expectation_values.value()[1][i].expectation();
          double ez =
              evolution_result.expectation_values.value()[2][i].expectation();
          std::cout << steps[i] << " " << ex << " " << ey << " " << ez << "\n";
        }
:::
:::
:::
:::

Examples that illustrate how to use the `dynamics`{.docutils .literal
.notranslate} target are available in the [CUDA-Q
repository](https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/python/dynamics){.reference
.external}.
:::

::: {#operator .section}
Operator[¶](#operator "Permalink to this heading"){.headerlink}
---------------------------------------------------------------

CUDA-Q provides builtin definitions for commonly-used operators, such as
the ladder operators ([\\(a\\)]{.math .notranslate .nohighlight} and
[\\(a\^\\dagger\\)]{.math .notranslate .nohighlight}) of a harmonic
oscillator, the Pauli spin operators for a two-level system, etc.

Here is a list of those operators.

+-------------------+--------------------------------------------------+
| Name              | Description                                      |
+===================+==================================================+
| `identity`{.code  | Identity operator                                |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `zero`{.code      | Zero or null operator                            |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `                 | Bosonic annihilation operator ([\\(a\\)]{.math   |
| annihilate`{.code | .notranslate .nohighlight})                      |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `create`{.code    | Bosonic creation operator                        |
| .docutils         | ([\\(a\^\\dagger\\)]{.math .notranslate          |
| .literal          | .nohighlight})                                   |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `number`{.code    | Number operator of a bosonic mode (equivalent to |
| .docutils         | [\\(a\^\\dagger a\\)]{.math .notranslate         |
| .literal          | .nohighlight})                                   |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `parity`{.code    | Parity operator of a bosonic mode (defined as    |
| .docutils         | [\\(e\^{i\\pi a\^\\dagger a}\\)]{.math           |
| .literal          | .notranslate .nohighlight})                      |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `displace`{.code  | Displacement operator of complex amplitude       |
| .docutils         | [\\(\\alpha\\)]{.math .notranslate .nohighlight} |
| .literal          | (`displacement`{.code .docutils .literal         |
| .notranslate}     | .notranslate}). It is defined as [\\(e\^{\\alpha |
|                   | a\^\\dagger - \\alpha\^\* a}\\)]{.math           |
|                   | .notranslate .nohighlight}.                      |
+-------------------+--------------------------------------------------+
| `squeeze`{.code   | Squeezing operator of complex squeezing          |
| .docutils         | amplitude [\\(z\\)]{.math .notranslate           |
| .literal          | .nohighlight} (`squeezing`{.code .docutils       |
| .notranslate}     | .literal .notranslate}). It is defined as        |
|                   | [\\(\\exp(\\frac{1}{2}(z\^\*a\^2 - z             |
|                   | a\^{\\dagger 2}))\\)]{.math .notranslate         |
|                   | .nohighlight}.                                   |
+-------------------+--------------------------------------------------+
| `position`{.code  | Position operator (equivalent to                 |
| .docutils         | [\\((a\^\\dagger + a)/2\\)]{.math .notranslate   |
| .literal          | .nohighlight})                                   |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `momentum`{.code  | Momentum operator (equivalent to                 |
| .docutils         | [\\(i(a\^\\dagger - a)/2\\)]{.math .notranslate  |
| .literal          | .nohighlight})                                   |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `spin.x`{.code    | Pauli [\\(\\sigma\_x\\)]{.math .notranslate      |
| .docutils         | .nohighlight} operator                           |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `spin.y`{.code    | Pauli [\\(\\sigma\_y\\)]{.math .notranslate      |
| .docutils         | .nohighlight} operator                           |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `spin.z`{.code    | Pauli [\\(\\sigma\_z\\)]{.math .notranslate      |
| .docutils         | .nohighlight} operator                           |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `spin.plus`{.code | Pauli raising ([\\(\\sigma\_+\\)]{.math          |
| .docutils         | .notranslate .nohighlight}) operator             |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `                 | Pauli lowering ([\\(\\sigma\_-\\)]{.math         |
| spin.minus`{.code | .notranslate .nohighlight}) operator             |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+

: [Builtin
Operators]{.caption-text}[¶](#id1 "Permalink to this table"){.headerlink}

As an example, let's look at the Jaynes-Cummings model, which describes
the interaction between a two-level atom and a light (Boson) field.

Mathematically, the Hamiltonian can be expressed as

::: {.math .notranslate .nohighlight}
\\\[H = \\omega\_c a\^\\dagger a + \\omega\_a \\frac{\\sigma\_z}{2} +
\\frac{\\Omega}{2}(a\\sigma\_+ + a\^\\dagger \\sigma\_-).\\\]
:::

This Hamiltonian can be converted to CUDA-Q `Operator`{.code .docutils
.literal .notranslate} representation with

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    from cudaq import operators

    hamiltonian = omega_c * operators.create(1) * operators.annihilate(1) \
                    + (omega_a / 2) * spin.z(0) \
                    + (Omega / 2) * (operators.annihilate(1) * spin.plus(0) + operators.create(1) * spin.minus(0))
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
        // Jaynes-Cummings Hamiltonian
        auto jc_hamiltonian =
            omega_c * boson_op::create(1) * boson_op::annihilate(1) +
            (omega_a / 2.0) * spin_op::z(0) +
            (Omega / 2.0) * (boson_op::annihilate(1) * spin_op::plus(0) +
                             boson_op::create(1) * spin_op::minus(0));
:::
:::
:::
:::

In the above code snippet, we map the cavity light field to degree index
1 and the two-level atom to degree index 0. The description of composite
quantum system dynamics is independent from the Hilbert space of the
system components. The latter is specified by the dimension map that is
provided to the `cudaq.evolve`{.code .docutils .literal .notranslate}
call.

Builtin operators support both dense and multi-diagonal sparse formats.
Depending on the sparsity of operator matrix and/or the sub-system
dimension, CUDA-Q will either use the dense or multi-diagonal data
formats for optimal performance.

Specifically, the following environment variable options are applicable
to the `dynamics`{.code .docutils .literal .notranslate} target. Any
environment variables must be set prior to setting the target or running
"`import cudaq`{.code .docutils .literal .notranslate}".

+-------------+--------------------+-----------------------------------+
| Option      | Value              | Description                       |
+-------------+--------------------+-----------------------------------+
| `CUDAQ_DYN  | Non-negative       | The minimum sub-system dimension  |
| AMICS_MIN_M | number             | on which the operator acts to     |
| ULTIDIAGONA |                    | activate multi-diagonal data      |
| L_DIMENSION |                    | format. For example, if a minimum |
| `{.docutils |                    | dimension configuration of        |
| .literal    |                    | `N`{.code .docutils .literal      |
| .n          |                    | .notranslate} is set, all         |
| otranslate} |                    | operators acting on degrees of    |
|             |                    | freedom (sub-system) whose        |
|             |                    | dimension is less than or equal   |
|             |                    | to `N`{.code .docutils .literal   |
|             |                    | .notranslate} would always use    |
|             |                    | the dense format. The final data  |
|             |                    | format to be used depends on the  |
|             |                    | next configuration. The default   |
|             |                    | is 4.                             |
+-------------+--------------------+-----------------------------------+
| `CUDAQ_D    | Non-negative       | The maximum number of diagonals   |
| YNAMICS_MAX | number             | for multi-diagonal                |
| _DIAGONAL_C |                    | representation. If the operator   |
| OUNT_FOR_MU |                    | matrix has more diagonals than    |
| LTIDIAGONAL |                    | this value, the dense format will |
| `{.docutils |                    | be used. Default is 1, i.e.,      |
| .literal    |                    | operators with only one diagonal  |
| .n          |                    | line (center, lower, or upper)    |
| otranslate} |                    | will use the multi-diagonal       |
|             |                    | sparse storage.                   |
+-------------+--------------------+-----------------------------------+

: [**Additional environment variable options for the \`dynamics\`
target**]{.caption-text}[¶](#id2 "Permalink to this table"){.headerlink}
:::

::: {#time-dependent-dynamics .section}
Time-Dependent Dynamics[¶](#time-dependent-dynamics "Permalink to this heading"){.headerlink}
---------------------------------------------------------------------------------------------

In the previous examples of operator construction, we assumed that the
systems under consideration were described by time-independent
Hamiltonian. However, we may want to simulate systems whose Hamiltonian
operators have explicit time dependence.

CUDA-Q provides multiple ways to construct time-dependent operators.

1.  Time-dependent coefficient

CUDA-Q `ScalarOperator`{.code .docutils .literal .notranslate} can be
used to wrap a Python/C++ function that returns the coefficient value at
a specific time.

As an example, we will look at a time-dependent Hamiltonian of the form
[\\(H = H\_0 + f(t)H\_1\\)]{.math .notranslate .nohighlight}, where
[\\(f(t)\\)]{.math .notranslate .nohighlight} is the time-dependent
driving strength given as [\\(cos(\\omega t)\\)]{.math .notranslate
.nohighlight}.

The following code sets up the problem

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    # Define the static (drift) and control terms
    H0 = spin.z(0)
    H1 = spin.x(0)
    H = H0 + ScalarOperator(lambda t: np.cos(omega * t)) * H1
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
        // Hamiltonian with driving frequency
        double omega = M_PI;
        auto H0 = spin_op::z(0);
        auto H1 = spin_op::x(0);
        auto mod_func =
            [omega](const std::unordered_map<std::string, std::complex<double>>
                        &parameters) {
              auto entry = parameters.find("t");
              if (entry == parameters.end())
                throw std::runtime_error("Cannot find value of expected parameter");
              const auto t = entry->second.real();
              return std::cos(omega * t);
            };
        auto driven_hamiltonian = H0 + mod_func * H1;
:::
:::
:::
:::

2.  Time-dependent operator

We can also construct a time-dependent operator from a function that
returns a complex matrix representing the time dynamics of that
operator.

As an example, let's looks at the [displacement
operator](https://en.wikipedia.org/wiki/Displacement_operator){.reference
.external}. It can be defined as follows:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    import numpy
    import scipy
    from cudaq import operators, NumericType
    from numpy.typing import NDArray


    def displacement_matrix(
            dimension: int,
            displacement: NumericType) -> NDArray[numpy.complexfloating]:
        """
        Returns the displacement operator matrix.
        Args:
            displacement: Amplitude of the displacement operator.
                See also https://en.wikipedia.org/wiki/Displacement_operator.
        """
        displacement = complex(displacement)
        term1 = displacement * operators.create(0).to_matrix({0: dimension})
        term2 = numpy.conjugate(displacement) * operators.annihilate(0).to_matrix(
            {0: dimension})
        return scipy.linalg.expm(term1 - term2)


    # The second argument here indicates the defined operator
    # acts on a single degree of freedom, which can have any dimension.
    # An argument [2], for example, would indicate that it can only
    # act on a single degree of freedom with dimension two.
    operators.define("displace", [0], displacement_matrix)


    def displacement(degree: int) -> operators.MatrixOperatorElement:
        """
        Instantiates a displacement operator acting on the given degree of freedom.
        """
        return operators.instantiate("displace", [degree])
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
        auto displacement_matrix =
            [](const std::vector<int64_t> &dimensions,
               const std::unordered_map<std::string, std::complex<double>>
                   &parameters) -> cudaq::complex_matrix {
          // Returns the displacement operator matrix.
          //  Args:
          //   - displacement: Amplitude of the displacement operator.
          // See also https://en.wikipedia.org/wiki/Displacement_operator.
          std::size_t dimension = dimensions[0];
          auto entry = parameters.find("displacement");
          if (entry == parameters.end())
            throw std::runtime_error("missing value for parameter 'displacement'");
          auto displacement_amplitude = entry->second;
          auto create = cudaq::complex_matrix(dimension, dimension);
          auto annihilate = cudaq::complex_matrix(dimension, dimension);
          for (std::size_t i = 0; i + 1 < dimension; i++) {
            create[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1));
            annihilate[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1));
          }
          auto term1 = displacement_amplitude * create;
          auto term2 = std::conj(displacement_amplitude) * annihilate;
          return (term1 - term2).exponential();
        };

        cudaq::matrix_handler::define("displace_op", {-1}, displacement_matrix);

        // Instantiate a displacement operator acting on the given degree of
        // freedom.
        auto displacement = [](std::size_t degree) {
          return cudaq::matrix_handler::instantiate("displace_op", {degree});
        };
:::
:::
:::
:::

The defined operator is parameterized by the `displacement`{.code
.docutils .literal .notranslate} amplitude. To create simulate the
evolution of an operator under a time dependent displacement amplitude,
we can define how the amplitude changes in time:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    import cudaq

    # Define a system consisting of a single degree of freedom (0) with dimension 3.
    system_dimensions = {0: 3}
    system_operator = displacement(0)

    # Define the time dependency of the system operator as a schedule that linearly
    # increases the displacement parameter from 0 to 1.
    time_dependence = Schedule(numpy.linspace(0, 1, 100), ['displacement'])
    initial_state = cudaq.State.from_data(
        numpy.ones(3, dtype=numpy.complex128) / numpy.sqrt(3))

    # Simulate the evolution of the system under this time dependent operator.
    cudaq.evolve(system_operator, system_dimensions, time_dependence, initial_state)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
        // Define a system consisting of a single degree of freedom (0) with
        // dimension 3.
        cudaq::dimension_map system_dimensions{{0, 3}};
        auto system_operator = displacement(0);

        // Define the time dependency of the system operator as a schedule that
        // linearly increases the displacement parameter from 0 to 1.
        cudaq::schedule time_dependence(cudaq::linspace(0, 1, 100),
                                        {"displacement"});
        const std::vector<std::complex<double>> state_vec(3, 1.0 / std::sqrt(3.0));
        auto initial_state = cudaq::state::from_data(state_vec);
        cudaq::integrators::runge_kutta integrator(4, 0.01);
        // Simulate the evolution of the system under this time dependent operator.
        cudaq::evolve(system_operator, system_dimensions, time_dependence,
                      initial_state, integrator);
:::
:::
:::
:::

Let's say we want to add a squeezing term to the system operator. We can
independently vary the squeezing amplitude and the displacement
amplitude by instantiating a schedule with a custom function that
returns the desired value for each parameter:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    system_operator = displacement(0) + operators.squeeze(0)


    # Define a schedule such that displacement amplitude increases linearly in time
    # but the squeezing amplitude decreases, that is follows the inverse schedule.
    def parameter_values(time_steps):

        def compute_value(param_name, step_idx):
            match param_name:
                case 'displacement':
                    return time_steps[int(step_idx)]
                case 'squeezing':
                    return time_steps[-int(step_idx + 1)]
                case _:
                    raise ValueError(f"value for parameter {param_name} undefined")

        return Schedule(range(len(time_steps)), system_operator.parameters.keys(),
                        compute_value)


    time_dependence = parameter_values(numpy.linspace(0, 1, 100))
    cudaq.evolve(system_operator, system_dimensions, time_dependence, initial_state)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
        auto hamiltonian = displacement(0) + cudaq::matrix_op::squeeze(0);

        // Define a schedule such that displacement amplitude increases linearly in
        // time but the squeezing amplitude decreases, that is follows the inverse
        // schedule. def parameter_values(time_steps):
        auto parameter_values = [](const std::vector<double> &time_steps) {
          auto compute_value = [time_steps](const std::string &param_name,
                                            const std::complex<double> &step) {
            int step_idx = (int)step.real();
            if (param_name == "displacement")
              return time_steps[step_idx];
            if (param_name == "squeezing")
              return time_steps[time_steps.size() - (step_idx + 1)];

            throw std::runtime_error("value for parameter " + param_name +
                                     " undefined");
          };

          std::vector<std::complex<double>> steps;
          for (int i = 0; i < time_steps.size(); ++i)
            steps.emplace_back(i);
          return cudaq::schedule(steps, {"displacement", "squeezing"},
                                 compute_value);
        };

        auto time_dependence_param = parameter_values(cudaq::linspace(0, 1, 100));
        cudaq::evolve(hamiltonian, system_dimensions, time_dependence_param,
                      initial_state, integrator);
:::
:::
:::
:::

Compile and Run C++ program

::: {.tab-set .docutils}
C++

::: {.tab-content .docutils}
::: {.highlight-bash .notranslate}
::: {.highlight}
    nvq++ --target dynamics dynamics.cpp -o dynamics && ./dynamics
:::
:::
:::
:::
:::

::: {#super-operator-representation .section}
Super-operator Representation[¶](#super-operator-representation "Permalink to this heading"){.headerlink}
---------------------------------------------------------------------------------------------------------

In the previous examples, we assumed that the system dynamics is driven
by a `Lindblad`{.code .docutils .literal .notranslate} master equation,
which is specified by the Hamiltonian operator and the collapse
operators.

However, we may want to simulate an arbitrary state evolution equation,
whereby the right-hand-side of the differential equation is provided as
a generic super-operator.

CUDA-Q provides a `SuperOperator`{.code .docutils .literal .notranslate}
(Python) / `super_op`{.code .docutils .literal .notranslate} (C++) class
that can be used to represent the right-hand-side of the evolution
equation. A super-operator can be constructed as a linear combination
(sum) of left and/or right multiplication actions of `Operator`{.code
.docutils .literal .notranslate} instances.

As an example, we will look at specifying the Schrodinger's equation
[\\(\\frac{d\|\\Psi\\rangle}{dt} = -i H \|\\Psi\\rangle\\)]{.math
.notranslate .nohighlight} as a super-operator.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, Schedule, RungeKuttaIntegrator
    import numpy as np

    hamiltonian = 2.0 * np.pi * 0.1 * spin.x(0)
    steps = np.linspace(0, 1, 10)
    schedule = Schedule(steps, ["t"])
    dimensions = {0: 2}
    # initial state
    psi0 = cudaq.dynamics.InitialState.ZERO
    # Create a super-operator that represents the evolution of the system
    # under the Hamiltonian `-iH|psi>`, where `H` is the Hamiltonian.
    se_super_op = cudaq.SuperOperator()
    # Apply `-iH|psi>` super-operator
    se_super_op += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
    evolution_result = cudaq.evolve(se_super_op,
                                    dimensions,
                                    schedule,
                                    psi0,
                                    observables=[spin.z(0)],
                                    store_intermediate_results=True,
                                    integrator=RungeKuttaIntegrator())
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
        const cudaq::dimension_map dims = {{0, 2}};
        cudaq::product_op<cudaq::matrix_handler> ham_ =
            2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
        cudaq::sum_op<cudaq::matrix_handler> ham(ham_);
        constexpr int numSteps = 10;
        cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
        auto initialState =
            cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
        cudaq::integrators::runge_kutta integrator(1, 0.001);
        // Create a super-operator to evolve the system under the Schrödinger
        // equation `-iH * |psi>`, where `H` is the Hamiltonian.
        cudaq::super_op sup;
        // Apply `-iH * |psi>` superop
        sup +=
            cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham);
        auto result = cudaq::evolve(sup, dims, schedule, initialState, integrator,
                                    {cudaq::spin_op::z(0)},
                                    cudaq::IntermediateResultSave::All);
:::
:::
:::
:::

The super-operator, once constructed, can be used in the `evolve`{.code
.docutils .literal .notranslate} API instead of the Hamiltonian and
collapse operators as shown in the above examples.
:::

::: {#numerical-integrators .section}
Numerical Integrators[¶](#numerical-integrators "Permalink to this heading"){.headerlink}
-----------------------------------------------------------------------------------------

For Python, CUDA-Q provides a set of numerical integrators, to be used
with the `dynamics`{.docutils .literal .notranslate} backend target.

+-------------------+--------------------------------------------------+
| Name              | Description                                      |
+===================+==================================================+
| `RungeKutta       | Explicit 4th-order Runge-Kutta method (default   |
| Integrator`{.code | integrator)                                      |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `ScipyZvode       | Complex-valued variable-coefficient ordinary     |
| Integrator`{.code | differential equation solver (provided by SciPy) |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `CUDA             | Runge-Kutta of order 5 of                        |
| TorchDiffEqDopri5 | Dormand-Prince-Shampine (provided by             |
| Integrator`{.code | `torchdiffeq`{.code .docutils .literal           |
| .docutils         | .notranslate})                                   |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `CUDATorchD       | Runge-Kutta of order 2 (provided by              |
| iffEqAdaptiveHeun | `torchdiffeq`{.code .docutils .literal           |
| Integrator`{.code | .notranslate})                                   |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `CUD              | Runge-Kutta of order 3 of Bogacki-Shampine       |
| ATorchDiffEqBosh3 | (provided by `torchdiffeq`{.code .docutils       |
| Integrator`{.code | .literal .notranslate})                          |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `CUDA             | Runge-Kutta of order 8 of                        |
| TorchDiffEqDopri8 | Dormand-Prince-Shampine (provided by             |
| Integrator`{.code | `torchdiffeq`{.code .docutils .literal           |
| .docutils         | .notranslate})                                   |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `CUD              | Euler method (provided by `torchdiffeq`{.code    |
| ATorchDiffEqEuler | .docutils .literal .notranslate})                |
| Integrator`{.code |                                                  |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `CUDATorchDi      | Explicit Adams-Bashforth method (provided by     |
| ffEqExplicitAdams | `torchdiffeq`{.code .docutils .literal           |
| Integrator`{.code | .notranslate})                                   |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `CUDATorchDi      | Implicit Adams-Bashforth-Moulton method          |
| ffEqImplicitAdams | (provided by `torchdiffeq`{.code .docutils       |
| Integrator`{.code | .literal .notranslate})                          |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `CUDATo           | Midpoint method (provided by `torchdiffeq`{.code |
| rchDiffEqMidpoint | .docutils .literal .notranslate})                |
| Integrator`{.code |                                                  |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+
| `C                | Fourth-order Runge-Kutta with 3/8 rule (provided |
| UDATorchDiffEqRK4 | by `torchdiffeq`{.code .docutils .literal        |
| Integrator`{.code | .notranslate})                                   |
| .docutils         |                                                  |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+

: [Numerical
Integrators]{.caption-text}[¶](#id3 "Permalink to this table"){.headerlink}

::: {.admonition .note}
Note

To use Torch-based integrators, users need to install
`torchdiffeq`{.code .docutils .literal .notranslate} (e.g., with
`pip install torchdiffeq`{.code .docutils .literal .notranslate}). This
is an optional dependency of CUDA-Q, thus will not be installed by
default.
:::

::: {.admonition .note}
Note

If you are using CUDA 12.8 on Blackwell, you may need to install nightly
torch.

See [[Blackwell Torch Dependencies]{.std
.std-ref}](install/local_installation.html#blackwell-torch-dependences){.reference
.internal} for more information.
:::

::: {.admonition .warning}
Warning

Torch-based integrators require a CUDA-enabled Torch installation.
Depending on your platform (e.g., `aarch64`{.code .docutils .literal
.notranslate}), the default Torch pip package may not have CUDA support.

The below command can be used to verify your installation:

::: {.highlight-bash .notranslate}
::: {.highlight}
    python3 -c "import torch; print(torch.version.cuda)"
:::
:::

If the output is a '`None`{.code .docutils .literal .notranslate}'
string, it indicates that your Torch installation does not support CUDA.
In this case, you need to install a CUDA-enabled Torch package via other
mechanisms, e.g., building Torch from source or using their Docker
images.
:::

For C++, CUDA-Q provides Runge-Kutta integrator, to be used with the
`dynamics`{.docutils .literal .notranslate} backend target.

+-------------------+--------------------------------------------------+
| Name              | Description                                      |
+===================+==================================================+
| `r                | 1st-order (Euler method), 2nd-order (Midpoint    |
| unge_kutta`{.code | method), and 4th-order (classical Runge-Kutta    |
| .docutils         | method).                                         |
| .literal          |                                                  |
| .notranslate}     |                                                  |
+-------------------+--------------------------------------------------+

: [Numerical
Integrators]{.caption-text}[¶](#id4 "Permalink to this table"){.headerlink}
:::

::: {#batch-simulation .section}
Batch simulation[¶](#batch-simulation "Permalink to this heading"){.headerlink}
-------------------------------------------------------------------------------

CUDA-Q `dynamics`{.docutils .literal .notranslate} target supports batch
simulation, which allows users to run multiple simulations
simultaneously. This batching capability applies to (1) multiple initial
states and/or (2) multiple Hamiltonians.

Batching can significantly improve performance when simulating many
small identical system dynamics, e.g., parameter sweeping or tomography.

For example, we can simulate the time evolution of multiple initial
states with the same Hamiltonian as follows:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    import cudaq
    import cupy as cp
    import numpy as np
    from cudaq import spin, Schedule, RungeKuttaIntegrator
    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

    # Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
    dimensions = {0: 2}

    # Initial states in the `SIC-POVM` set: https://en.wikipedia.org/wiki/SIC-POVM
    psi_1 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))
    psi_2 = cudaq.State.from_data(
        cp.array([1.0 / np.sqrt(3.0), np.sqrt(2.0 / 3.0)], dtype=cp.complex128))
    psi_3 = cudaq.State.from_data(
        cp.array([
            1.0 / np.sqrt(3.0),
            np.sqrt(2.0 / 3.0) * np.exp(1j * 2.0 * np.pi / 3.0)
        ],
                 dtype=cp.complex128))
    psi_4 = cudaq.State.from_data(
        cp.array([
            1.0 / np.sqrt(3.0),
            np.sqrt(2.0 / 3.0) * np.exp(1j * 4.0 * np.pi / 3.0)
        ],
                 dtype=cp.complex128))

    # We run the evolution for all the SIC state to determine the process tomography.
    sic_states = [psi_1, psi_2, psi_3, psi_4]
    # Schedule of time steps.
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["time"])

    # Run the batch simulation.
    evolution_results = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        sic_states,
        observables=[spin.x(0), spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=RungeKuttaIntegrator())
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
      // Qubit Hamiltonian
      auto hamiltonian = 2 * M_PI * 0.1 * cudaq::spin_op::x(0);

      // A single qubit with dimension 2.
      cudaq::dimension_map dimensions = {{0, 2}};

      // Initial states in the `SIC-POVM` set:
      // https://en.wikipedia.org/wiki/SIC-POVM
      auto psi_1 =
          cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
      auto psi_2 = cudaq::state::from_data(std::vector<std::complex<double>>{
          1.0 / std::sqrt(3.0), std::sqrt(2.0 / 3.0)});
      auto psi_3 = cudaq::state::from_data(std::vector<std::complex<double>>{
          1.0 / std::sqrt(3.0),
          std::sqrt(2.0 / 3.0) *
              std::exp(std::complex<double>{0.0, 1.0} * 2.0 * M_PI / 3.0)});
      auto psi_4 = cudaq::state::from_data(std::vector<std::complex<double>>{
          1.0 / std::sqrt(3.0),
          std::sqrt(2.0 / 3.0) *
              std::exp(std::complex<double>{0.0, 1.0} * 4.0 * M_PI / 3.0)});
      // We run the evolution for all the SIC state to determine the process
      // tomography.
      std::vector<cudaq::state> sic_states = {psi_1, psi_2, psi_3, psi_4};

      // Schedule of time steps.
      std::vector<double> steps = cudaq::linspace(0.0, 10.0, 101);
      cudaq::schedule schedule(steps);

      // A default Runge-`Kutta` integrator
      cudaq::integrators::runge_kutta integrator;

      // Run the batch simulation.
      auto evolve_results = cudaq::evolve(
          hamiltonian, dimensions, schedule, sic_states, integrator, {},
          {cudaq::spin_op::x(0), cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
          cudaq::IntermediateResultSave::ExpectationValue);
:::
:::
:::
:::

Similarly, we can also batch simulate the time evolution of multiple
Hamiltonians as follows:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    import cudaq
    import cupy as cp
    import numpy as np
    from cudaq import spin, Schedule, ScalarOperator, RungeKuttaIntegrator
    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Dimensions of sub-system.
    dimensions = {0: 2}

    # Qubit resonant frequency
    omega_z = 10.0 * 2 * np.pi

    # Transverse term
    omega_x = 2 * np.pi

    # Harmonic driving frequency (sweeping in the +/- 10% range around the resonant frequency).
    omega_drive = np.linspace(0.1 * omega_z, 1.1 * omega_z, 16)

    # Initial state of the system (ground state).
    psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

    # Batch the Hamiltonian operator together
    hamiltonians = [
        0.5 * omega_z * spin.z(0) + omega_x *
        ScalarOperator(lambda t, omega=omega: np.cos(omega * t)) * spin.x(0)
        for omega in omega_drive
    ]

    # Initial states for each Hamiltonian in the batch.
    # Here, we use the ground state for all Hamiltonians.
    initial_states = [psi0] * len(hamiltonians)

    # Schedule of time steps.
    steps = np.linspace(0, 0.5, 5000)
    schedule = Schedule(steps, ["t"])

    # Run the batch simulation.
    evolution_results = cudaq.evolve(
        hamiltonians,
        dimensions,
        schedule,
        initial_states,
        observables=[spin.x(0), spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=RungeKuttaIntegrator())
:::
:::

In this example, we show the most generic batching capability, where
each Hamiltonian in the batch corresponds to a specific initial state.
In other words, the vector of Hamiltonians and the vector of initial
states are of the same length. If only one initial state is provided, it
will be used for all Hamiltonians in the batch.
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
      // Dimensions of sub-system
      cudaq::dimension_map dimensions = {{0, 2}};
      // Qubit resonant frequency
      const double omega_z = 10.0 * 2 * M_PI;
      // Transverse driving term
      const double omega_x = 2 * M_PI;
      // Harmonic driving frequency (sweeping in the +/- 10% range around the
      // resonant frequency).
      const auto omega_drive = cudaq::linspace(0.9 * omega_z, 1.1 * omega_z, 16);
      const auto zero_state =
          cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
      // List of Hamiltonians to be batched together
      std::vector<cudaq::spin_op> hamiltonians;

      for (const auto &omega : omega_drive) {
        auto mod_func =
            [omega](const cudaq::parameter_map &params) -> std::complex<double> {
          auto it = params.find("t");
          if (it != params.end()) {
            double t = it->second.real();
            return std::cos(omega * t);
          }
          throw std::runtime_error("Cannot find the time parameter.");
        };

        // Add the Hamiltonian for each drive frequency to the batch.
        hamiltonians.emplace_back(0.5 * omega_z * cudaq::spin_op::z(0) +
                                  mod_func * cudaq::spin_op::x(0) * omega_x);
      }

      // The qubit starts in the |0> state for all operators in the batch.
      std::vector<cudaq::state> initial_states(hamiltonians.size(), zero_state);
      // Schedule of time steps
      const std::vector<double> steps = cudaq::linspace(0.0, 0.5, 5000);
      // The schedule carries the time parameter `labelled` `t`, which is used by
      // the callback.
      cudaq::schedule schedule(steps, {"t"});

      // A default Runge-`Kutta` integrator (4`th` order) with time step `dt`
      // depending on the schedule.
      cudaq::integrators::runge_kutta integrator;

      // Run the batch simulation.
      auto evolve_results = cudaq::evolve(
          hamiltonians, dimensions, schedule, initial_states, integrator, {},
          {cudaq::spin_op::x(0), cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
          cudaq::IntermediateResultSave::ExpectationValue);
:::
:::
:::
:::

The results of the batch simulation will be returned as a list of evolve
result objects, one for each Hamiltonian in the batch. For example, we
can extract the time evolution results of the expectation values for
each Hamiltonian in the batch as follows:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    # The results of the batched evolution is an array of evolution results,
    # one for each Hamiltonian operator in the batch.

    # For example, we can split the results into separate arrays for each observable.
    all_exp_val_x = []
    all_exp_val_y = []
    all_exp_val_z = []
    # Iterate over the evolution results in the batch:
    for evolution_result in evolution_results:
        # Extract the expectation values for each observable at the respective Hamiltonian operator in the batch.
        exp_val_x = [
            exp_vals[0].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]
        exp_val_y = [
            exp_vals[1].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]
        exp_val_z = [
            exp_vals[2].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]

        # Append the results to the respective lists.
        # These will be nested lists, where each inner list corresponds to the results for a specific Hamiltonian operator in the batch.
        all_exp_val_x.append(exp_val_x)
        all_exp_val_y.append(exp_val_y)
        all_exp_val_z.append(exp_val_z)
:::
:::

The expectation values are returned as a list of lists, where each inner
list corresponds to the expectation values for the observables at each
time step for the respective Hamiltonian in the batch.

::: {.highlight-bash .notranslate}
::: {.highlight}
    all_exp_val_x = [[0.0, ...], [0.0, ...], ..., [0.0, ...]]
    all_exp_val_y = [[0.0, ...], [0.0, ...], ..., [0.0, ...]]
    all_exp_val_z = [[1.0, ...], [1.0, ...], ..., [1.0, ...]]
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
      // The results of the batched evolution is an array of evolution results, one
      // for each Hamiltonian operator in the batch.

      // For example, we can split the results into separate arrays for each
      // observable. These will be nested lists, where each inner list corresponds
      // to the results for a specific Hamiltonian operator in the batch.
      std::vector<std::vector<double>> all_exp_val_x;
      std::vector<std::vector<double>> all_exp_val_y;
      std::vector<std::vector<double>> all_exp_val_z;
      // Iterate over the evolution results in the batch:
      for (auto &evolution_result : evolve_results) {
        // Extract the expectation values for each observable at the respective
        // Hamiltonian operator in the batch.
        std::vector<double> exp_val_x, exp_val_y, exp_val_z;
        for (auto &exp_vals : evolution_result.expectation_values.value()) {
          exp_val_x.push_back(exp_vals[0].expectation());
          exp_val_y.push_back(exp_vals[1].expectation());
          exp_val_z.push_back(exp_vals[2].expectation());
        }

        // Append the results to the respective lists.
        all_exp_val_x.push_back(exp_val_x);
        all_exp_val_y.push_back(exp_val_y);
        all_exp_val_z.push_back(exp_val_z);
      }
:::
:::

The expectation values are returned as a list of lists, where each inner
list corresponds to the expectation values for the observables at each
time step for the respective Hamiltonian in the batch.

::: {.highlight-bash .notranslate}
::: {.highlight}
    all_exp_val_x = [[0.0, ...], [0.0, ...], ..., [0.0, ...]]
    all_exp_val_y = [[0.0, ...], [0.0, ...], ..., [0.0, ...]]
    all_exp_val_z = [[1.0, ...], [1.0, ...], ..., [1.0, ...]]
:::
:::
:::
:::

Collapse operators and super-operators can also be batched in a similar
manner. Specifically, if the `collapse_operators`{.code .docutils
.literal .notranslate} parameter is a nested list of operators
([\\(\\{\\{L\\}\_1, \\{\\{L\\}\_2, \...\\}\\)]{.math .notranslate
.nohighlight}), then each set of collapsed operators in the list will be
applied to the corresponding Hamiltonian in the batch.

In order for all Hamiltonians to be batched, they must have the same
structure, i.e., same number of product terms and those terms must act
on the same degrees of freedom. The order of the terms in the
Hamiltonian does not matter, nor do the coefficient values/callback
functions and the specific operators on those product terms. Here are a
couple of examples of Hamiltonians that can or cannot be batched:

+----------------------+----------------------+----------------------+
| First Hamiltonian    | Second Hamiltonian   | Batchable?           |
+======================+======================+======================+
| [\\(H\_1 =           | [\\(H\_2 =           | Yes (different       |
| \\omega\_1           | \\omega\_2           | coefficients, same   |
| \\s                  | \\s                  | operator)            |
| igma\_z(0)\\)]{.math | igma\_z(0)\\)]{.math |                      |
| .notranslate         | .notranslate         |                      |
| .nohighlight}        | .nohighlight}        |                      |
+----------------------+----------------------+----------------------+
| [\\(H\_1 =           | [\\(H\_2 =           | Yes (same structure, |
| \\omega\_z           | \\omega\_z           | different callback   |
| \\sigma\_z(0) +      | \\sigma\_z(0) +      | coefficients)        |
| \\cos(\\omega\_xt)   | \\sin(\\omega\_xt)   |                      |
| \\s                  | \\s                  |                      |
| igma\_x(1)\\)]{.math | igma\_x(1)\\)]{.math |                      |
| .notranslate         | .notranslate         |                      |
| .nohighlight}        | .nohighlight}        |                      |
+----------------------+----------------------+----------------------+
| [\\(H\_1 =           | [\\(H\_2 =           | Yes (different       |
| \\omega\_z           | \\omega\_z           | operators on the     |
| \\sigma\_z(0) +      | \\sigma\_z(0) +      | same degree of       |
| \\cos(\\omega\_xt)   | \\cos(\\omega\_xt)   | freedom)             |
| \\s                  | \\s                  |                      |
| igma\_x(1)\\)]{.math | igma\_y(1)\\)]{.math |                      |
| .notranslate         | .notranslate         |                      |
| .nohighlight}        | .nohighlight}        |                      |
+----------------------+----------------------+----------------------+
| [\\(H\_1 =           | [\\(H\_2 =           | No (different number |
| \\omega\_z           | \\omega\_z           | of product terms)    |
| \\sigma\_z(0) +      | \\sigma\_z(0) +      |                      |
| \\cos(\\omega\_xt)   | \\cos(\\omega\_xt)   |                      |
| \\s                  | \\sigma\_x(1) +      |                      |
| igma\_x(1)\\)]{.math | \\cos(\\omega\_yt)   |                      |
| .notranslate         | \\s                  |                      |
| .nohighlight}        | igma\_y(1)\\)]{.math |                      |
|                      | .notranslate         |                      |
|                      | .nohighlight}        |                      |
+----------------------+----------------------+----------------------+
| [\\(H\_1 =           | [\\(H\_2 =           | No (different        |
| \\omega\_z           | \\omega\_z           | structures, two-body |
| \\sigma\_z(0) +      | \\sigma\_z(0) +      | operators vs. tensor |
| \\cos(\\omega\_xt)   | \\cos(\\omega\_xt)   | product of           |
| \\sigma\_{xx}(0,     | \\sigma\_x(0)\\s     | single-body          |
| 1)\\)]{.math         | igma\_x(1)\\)]{.math | operators)           |
| .notranslate         | .notranslate         |                      |
| .nohighlight}        | .nohighlight}        |                      |
+----------------------+----------------------+----------------------+

When the Hamiltonians are **not** `batchable`{.code .docutils .literal
.notranslate}, CUDA-Q will still run the simulations, but each
Hamiltonian will be simulated separately in a sequential manner. CUDA-Q
will log a warning "The input Hamiltonian and collapse operators are not
compatible for batching. Running the simulation in non-batched mode.",
when that happens.

::: {.admonition .note}
Note

Depending on the number of Hamiltonian operators together with factors
such as the integrator, schedule step size, and whether intermediate
results are stored, the batch simulation can be memory-intensive. If you
encounter out-of-memory issues, the `max_batch_size`{.code .docutils
.literal .notranslate} parameter can be used to limit the number of
Hamiltonians that are batched together in one run. For example, if you
set `max_batch_size=2`{.code .docutils .literal .notranslate}, then we
will run the simulations in batches of 2 Hamiltonians at a time, i.e.,
the first two Hamiltonians will be simulated together, then the next
two, and so on.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: {.highlight}
    # Run the batch simulation with a maximum batch size of 2.
    # This means that the evolution will be performed in batches of 2 Hamiltonian operators at a time, which can be useful for memory management or
    # performance tuning.
    results = cudaq.evolve(
        hamiltonians,
        dimensions,
        schedule,
        initial_states,
        observables=[spin.x(0), spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=RungeKuttaIntegrator(),
        max_batch_size=2)  # Set the maximum batch size to 2
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: {.highlight}
      // Run the batch simulation with a maximum batch size of 2.
      // This means that the evolution will be performed in batches of 2 Hamiltonian
      // operators at a time, which can be useful for memory management or
      // performance tuning.
      auto results = cudaq::evolve(
          hamiltonians, dimensions, schedule, initial_states, integrator, {},
          {cudaq::spin_op::x(0), cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
          cudaq::IntermediateResultSave::ExpectationValue, /*max_batch_size=*/2);
:::
:::
:::
:::
:::
:::

::: {#multi-gpu-multi-node-execution .section}
Multi-GPU Multi-Node Execution[¶](#multi-gpu-multi-node-execution "Permalink to this heading"){.headerlink}
-----------------------------------------------------------------------------------------------------------

CUDA-Q `dynamics`{.docutils .literal .notranslate} target supports
parallel execution on multiple GPUs. To enable parallel execution, the
application must initialize MPI as follows.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
> <div>
>
> ::: {.highlight-python .notranslate}
> ::: {.highlight}
>     cudaq.mpi.initialize()
>
>     # Set the target to our dynamics simulator
>     cudaq.set_target("dynamics")
>
>     # Initial state (expressed as an enum)
>     psi0 = cudaq.dynamics.InitialState.ZERO
>
>     # Run the simulation
>     evolution_result = cudaq.evolve(
>         H,
>         dimensions,
>         schedule,
>         psi0,
>         observables=[],
>         collapse_operators=[],
>         store_intermediate_results=cudaq.IntermediateResultSave.NONE,
>         integrator=RungeKuttaIntegrator())
>
>     cudaq.mpi.finalize()
> :::
> :::
>
> ::: {.highlight-bash .notranslate}
> ::: {.highlight}
>     mpiexec -np <N> python3 program.py
> :::
> :::
>
> </div>

where `N`{.docutils .literal .notranslate} is the number of processes.
:::

C++

::: {.tab-content .docutils}
> <div>
>
> ::: {.highlight-cpp .notranslate}
> ::: {.highlight}
>         cudaq::mpi::initialize();
>         // Initial state (expressed as an enum)
>         auto psi0 = cudaq::InitialState::ZERO;
>
>         // Run the simulation
>         auto evolution_result =
>             cudaq::evolve(H, dimensions, schedule, psi0, integrator);
>
>         cudaq::mpi::finalize();
> :::
> :::
>
> ::: {.highlight-bash .notranslate}
> ::: {.highlight}
>     nvq++ --target dynamics example.cpp -o a.out
>     mpiexec -np <N> a.out
> :::
> :::
>
> </div>

where `N`{.docutils .literal .notranslate} is the number of processes.
:::
:::

By initializing the MPI execution environment (via
`cudaq.mpi.initialize()`{.code .docutils .literal .notranslate}) in the
application code and invoking it via an MPI launcher, we have activated
the multi-node multi-GPU feature of the `dynamics`{.docutils .literal
.notranslate} target. Specifically, it will detect the number of
processes (GPUs) and distribute the computation across all available
GPUs.

::: {.admonition .note}
Note

The number of MPI processes must be a power of 2, one GPU per process.
:::

::: {.admonition .note}
Note

Not all integrators are capable of handling distributed state. Errors
will be raised if parallel execution is activated but the selected
integrator does not support distributed state.
:::
:::

::: {#examples .section}
Examples[¶](#examples "Permalink to this heading"){.headerlink}
---------------------------------------------------------------

The [[Dynamics Examples]{.std
.std-ref}](examples/dynamics_examples.html#dynamics-examples){.reference
.internal} section of the docs contains a number of excellent dynamics
examples demonstrating how to simulate basic physics models, specific
qubit modalities, and utilize multi-GPU multi-Node capabilities.
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](backends/cloud/nvqc.html "NVIDIA Quantum Cloud"){.btn
.btn-neutral .float-left} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](cudaqx/cudaqx.html "CUDA-QX"){.btn .btn-neutral
.float-right}
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
