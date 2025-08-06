::: {.wy-grid-for-nav}
::: {.wy-side-scroll}
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../index.html){.icon .icon-home}

::: {.version}
pr-3308
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](../quick_start.html){.reference .internal}
    -   [Install CUDA-Q](../quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](../quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q
        Academic](../quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](../basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q
        Kernel?](../basics/kernel_intro.html){.reference .internal}
    -   [Building your first CUDA-Q
        Program](../basics/build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q
        Program](../basics/run_kernel.html){.reference .internal}
        -   [Sample](../basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](../basics/run_kernel.html#run){.reference .internal}
        -   [Observe](../basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](../basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](../basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](../basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
-   [Examples](examples.html){.reference .internal}
    -   [Introduction](introduction.html){.reference .internal}
    -   [Building Kernels](building_kernels.html){.reference .internal}
        -   [Defining
            Kernels](building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum Operations](quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring Kernels](measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](../../examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](../../examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](../../examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing Kernels](executing_kernels.html){.reference
        .internal}
        -   [Sample](executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](executing_kernels.html#run){.reference .internal}
            -   [Return Custom Data
                Types](executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get State](executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](expectation_values.html){.reference .internal}
        -   [Parallelizing across Multiple
            Processors](expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU Workflows](multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs (`mgpu`{.code .docutils
            .literal
            .notranslate})](multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs (`mqpu`{.code
            .docutils .literal
            .notranslate})](multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends (`remote-mqpu`{.code .docutils
            .literal
            .notranslate})](multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](../../examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [Built in CUDA-Q Optimizers and
            Gradients](../../examples/python/optimizers_gradients.html#Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
        -   [Third-Party
            Optimizers](../../examples/python/optimizers_gradients.html#Third-Party-Optimizers){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../../examples/python/optimizers_gradients.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](../../examples/python/noisy_simulations.html){.reference
        .internal}
    -   [Constructing Operators](operators.html){.reference .internal}
        -   [Constructing Spin
            Operators](operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](hardware_providers.html){.reference .internal}
        -   [Amazon
            Braket](hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Infleqtion](hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](hardware_providers.html#ionq){.reference .internal}
        -   [IQM](hardware_providers.html#iqm){.reference .internal}
        -   [OQC](hardware_providers.html#oqc){.reference .internal}
        -   [ORCA
            Computing](hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](hardware_providers.html#quantinuum){.reference
            .internal}
        -   [QuEra
            Computing](hardware_providers.html#quera-computing){.reference
            .internal}
    -   [Dynamics Examples](#){.current .reference .internal}
        -   [Cavity QED](#cavity-qed){.reference .internal}
        -   [Cross Resonance](#cross-resonance){.reference .internal}
        -   [Gate Calibration](#gate-calibration){.reference .internal}
        -   [Heisenberg Model](#heisenberg-model){.reference .internal}
        -   [Ion Trap](#ion-trap){.reference .internal}
        -   [Landau Zener](#landau-zener){.reference .internal}
        -   [Pulse](#pulse){.reference .internal}
        -   [Qubit Control](#qubit-control){.reference .internal}
        -   [Qubit Dynamics](#qubit-dynamics){.reference .internal}
        -   [Silicon Spin Qubit](#silicon-spin-qubit){.reference
            .internal}
        -   [Tensor Callback](#tensor-callback){.reference .internal}
        -   [Transmon Resonator](#transmon-resonator){.reference
            .internal}
        -   [Initial State (Multi-GPU
            Multi-Node)](#initial-state-multi-gpu-multi-node){.reference
            .internal}
        -   [Heisenberg Model (Multi-GPU
            Multi-Node)](#heisenberg-model-multi-gpu-multi-node){.reference
            .internal}
-   [Applications](../applications.html){.reference .internal}
    -   [Max-Cut with
        QAOA](../../applications/python/qaoa.html){.reference .internal}
    -   [Molecular docking via
        DC-QAOA](../../applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](../../applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](../../applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H\_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](../../applications/python/krylov.html){.reference
        .internal}
        -   [Setup](../../applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](../../applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](../../applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Bernstein-Vazirani
        Algorithm](../../applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](../../applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](../../applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](../../applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](../../applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](../../applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](../../applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](../../applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](../../applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](../../applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](../../applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](../../applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](../../applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](../../applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](../../applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](../../applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](../../applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](../../applications/python/hybrid_quantum_neural_networks.html){.reference
        .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](../../applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](../../applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using `Sample`{.docutils .literal .notranslate} to perform
            the Hadamard
            test](../../applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](../../applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](../../applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](../../applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](../../applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the `[[4,2,2]]`{.docutils .literal
            .notranslate}
            encoding](../../applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](../../applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](../../applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](../../applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
            .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](../../applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](../../applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](../../applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](../../applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](../../applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](../../applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](../../applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](../../applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. `Compute overlap`{.docutils .literal
                .notranslate}](../../applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](../../applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](../../applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](../../applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](../../applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](../../applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](../../applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](../../applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](../../applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](../../applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum Fourier
        Transform](../../applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](../../applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](../../applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](../../applications/python/quantum_teleportation.html#Teleportation-explained){.reference
            .internal}
    -   [Quantum
        Volume](../../applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](../../applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](../../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](../../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](../../applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Compiling Unitaries Using Diffusion
        Models](../../applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](../../applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](../../applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](../../applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](../../applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](../../applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](../../applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](../../applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](../../applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](../../applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](../../applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](../../applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](../../applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](../../applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](../../applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](../../applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](../../applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../../applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](../../applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](../../applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
            .internal}
    -   [Quantum
        Transformer](../../applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](../../applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](../../applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](../../applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](../../applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](../../applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](../../applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](../../applications/python/quantum_transformer.html#Attention-Maps){.reference
                .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](../../applications/python/afqmc.html){.reference
        .internal}
        -   [Hamiltonian preparation for
            VQE](../../applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](../../applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](../../applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](../../applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](../../applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](../../applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [ADAPT-QAOA
        algorithm](../../applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](../../applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H\_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](../../applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A\_j\\)]{.math .notranslate
            .nohighlight}:](../../applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H\_C,A\_j\]\\)]{.math .notranslate
            .nohighlight}:](../../applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](../../applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](../../applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](../../applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](../../applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](../../applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](../../applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](../../applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A\_i\\)]{.math .notranslate
            .nohighlight}\]](../../applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](../../applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](../../applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](../../applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](../../applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](../../applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](../../applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](../../applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](../../applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](../../applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](../../applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](../../applications/python/edge_detection.html#Post-processing){.reference
                .internal}
    -   [Factoring Integers With Shor's
        Algorithm](../../applications/python/shors.html){.reference
        .internal}
        -   [Shor's
            algorithm](../../applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](../../applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](../../applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](../../applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](../../applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](../../applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](../../applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](../../applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
        -   [(a) Generate the molecular Hamiltonian using Hartree Fock
            molecular
            orbitals](../../applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Hartree-Fock-molecular-orbitals){.reference
            .internal}
            -   [Active space
                Hamiltonian:](../../applications/python/generate_fermionic_ham.html#Active-space-Hamiltonian:){.reference
                .internal}
        -   [(b) Generate the active space hamiltonian using HF
            molecular
            orbitals.](../../applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-hamiltonian-using-HF-molecular-orbitals.){.reference
            .internal}
        -   [(c) Generate the active space Hamiltonian using the natural
            orbitals computed from MP2
            simulation](../../applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
            .internal}
        -   [(d) Generate the active space Hamiltonian computed from the
            CASSCF molecular
            orbitals](../../applications/python/generate_fermionic_ham.html#(d)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
            .internal}
            -   [Generate the electronic Hamiltonian using
                ROHF](../../applications/python/generate_fermionic_ham.html#Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
    -   [Grover's
        Algorithm](../../applications/python/grovers.html){.reference
        .internal}
        -   [Overview](../../applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](../../applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](../../applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](../../applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](../../applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](../../applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](../../applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](../../applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](../../applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](../../applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](../../applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](../../applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
            .internal}
    -   [The UCCSD Wavefunction
        ansatz](../../applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](../../applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](../../applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](../../applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](../../applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](../../applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](../../applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](../../applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](../../applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](../../applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](../../applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](../../applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](../../applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](../../applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](../../applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](../../applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](../../applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](../../applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](../../applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
                .internal}
-   [Backends](../backends/backends.html){.reference .internal}
    -   [Circuit Simulation](../backends/simulators.html){.reference
        .internal}
        -   [State Vector
            Simulators](../backends/sims/svsims.html){.reference
            .internal}
            -   [CPU](../backends/sims/svsims.html#cpu){.reference
                .internal}
            -   [Single-GPU](../backends/sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](../backends/sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network
            Simulators](../backends/sims/tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](../backends/sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](../backends/sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](../backends/sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU
            Simulators](../backends/sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](../backends/sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](../backends/sims/mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy Simulators](../backends/sims/noisy.html){.reference
            .internal}
            -   [Trajectory Noisy
                Simulation](../backends/sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](../backends/sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](../backends/sims/noisy.html#stim){.reference
                .internal}
        -   [Photonics
            Simulators](../backends/sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](../backends/sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware (QPUs)](../backends/hardware.html){.reference
        .internal}
        -   [Ion Trap
            QPUs](../backends/hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](../backends/hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](../backends/hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](../backends/hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](../backends/hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](../backends/hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](../backends/hardware/superconducting.html#oqc){.reference
                .internal}
        -   [Neutral Atom
            QPUs](../backends/hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](../backends/hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](../backends/hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](../backends/hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic
            QPUs](../backends/hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](../backends/hardware/photonic.html#orca-computing){.reference
                .internal}
    -   [Dynamics
        Simulation](../backends/dynamics_backends.html){.reference
        .internal}
    -   [Cloud](../backends/cloud.html){.reference .internal}
        -   [Amazon Braket
            (braket)](../backends/cloud/braket.html){.reference
            .internal}
            -   [Setting
                Credentials](../backends/cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submission from
                C++](../backends/cloud/braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](../backends/cloud/braket.html#submission-from-python){.reference
                .internal}
        -   [NVIDIA Quantum Cloud
            (nvqc)](../backends/cloud/nvqc.html){.reference .internal}
            -   [Quick
                Start](../backends/cloud/nvqc.html#quick-start){.reference
                .internal}
            -   [Simulator Backend
                Selection](../backends/cloud/nvqc.html#simulator-backend-selection){.reference
                .internal}
            -   [Multiple
                GPUs](../backends/cloud/nvqc.html#multiple-gpus){.reference
                .internal}
            -   [Multiple QPUs Asynchronous
                Execution](../backends/cloud/nvqc.html#multiple-qpus-asynchronous-execution){.reference
                .internal}
            -   [FAQ](../backends/cloud/nvqc.html#faq){.reference
                .internal}
-   [Dynamics](../dynamics.html){.reference .internal}
    -   [Quick Start](../dynamics.html#quick-start){.reference
        .internal}
    -   [Operator](../dynamics.html#operator){.reference .internal}
    -   [Time-Dependent
        Dynamics](../dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](../dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](../dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch simulation](../dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](../dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](../dynamics.html#examples){.reference .internal}
-   [CUDA-QX](../cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q
        Solvers](../cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](../cudaqx/cudaqx.html#cuda-q-qec){.reference
        .internal}
-   [Installation](../install/install.html){.reference .internal}
    -   [Local
        Installation](../install/local_installation.html){.reference
        .internal}
        -   [Introduction](../install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](../install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](../install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](../install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](../install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](../install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](../install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](../install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](../install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](../install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](../install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](../install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](../install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](../install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](../install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](../install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](../install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](../install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](../install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](../install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](../install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](../install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](../install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next
            Steps](../install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](../install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](../install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](../install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](../install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](../install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](../install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](../install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](../install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](../install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](../install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](../install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](../integration/integration.html){.reference .internal}
    -   [Downstream CMake
        Integration](../integration/cmake_app.html){.reference
        .internal}
    -   [Combining CUDA with
        CUDA-Q](../integration/cuda_gpu.html){.reference .internal}
    -   [Integrating with Third-Party
        Libraries](../integration/libraries.html){.reference .internal}
        -   [Calling a CUDA-Q library from
            C++](../integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](../integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](../integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](../extending/extending.html){.reference .internal}
    -   [Add a new Hardware
        Backend](../extending/backend.html){.reference .internal}
        -   [Overview](../extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](../extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](../extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](../extending/backend.html#server-helper-class){.reference
                .internal}
            -   [`CMakeLists.txt`{.docutils .literal
                .notranslate}](../extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](../extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent `CMakeLists.txt`{.docutils .literal
                .notranslate}](../extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](../extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](../extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](../extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](../extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](../extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](../extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](../extending/backend.html#example-usage){.reference
            .internal}
        -   [Code
            Review](../extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](../extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](../extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](../extending/nvqir_simulator.html){.reference
        .internal}
        -   [`CircuitSimulator`{.code .docutils .literal
            .notranslate}](../extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](../extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q IR](../extending/cudaq_ir.html){.reference
        .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](../extending/mlir_pass.html){.reference .internal}
-   [Specifications](../../specification/index.html){.reference
    .internal}
    -   [Language
        Specification](../../specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](../../specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](../../specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](../../specification/cudaq/types.html){.reference
            .internal}
            -   [3.1. `cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. `cudaq::qubit`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](../../specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](../../specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. `cudaq::spin_op`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](../../specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on `cudaq::qubit`{.code .docutils
                .literal
                .notranslate}](../../specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](../../specification/cudaq/kernels.html){.reference
            .internal}
        -   [7. Sub-circuit
            Synthesis](../../specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](../../specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](../../specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](../../specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](../../specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11.
            Platform](../../specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](../../specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. `cudaq::sample`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. `cudaq::run`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. `cudaq::observe`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. `cudaq::optimizer`{.code .docutils .literal
                .notranslate} (deprecated, functionality moved to CUDA-Q
                libraries)](../../specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. `cudaq::gradient`{.code .docutils .literal
                .notranslate} (deprecated, functionality moved to CUDA-Q
                libraries)](../../specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](../../specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](../../specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](../../specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](../../specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](../../specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](../../specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](../../specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](../../specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](../../specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](../../specification/quake-dialect.html#motivation){.reference
            .internal}
-   [API Reference](../../api/api.html){.reference .internal}
    -   [C++ API](../../api/languages/cpp_api.html){.reference
        .internal}
        -   [Operators](../../api/languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](../../api/languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](../../api/languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](../../api/languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](../../api/languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](../../api/languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Platform](../../api/languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](../../api/languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](../../api/languages/cpp_api.html#namespaces){.reference
            .internal}
    -   [Python API](../../api/languages/python_api.html){.reference
        .internal}
        -   [Program
            Construction](../../api/languages/python_api.html#program-construction){.reference
            .internal}
            -   [`make_kernel()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [`PyKernel`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [`Kernel`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [`PyKernelDecorator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [`kernel()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](../../api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [`sample()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [`sample_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [`run()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [`run_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [`observe()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [`observe_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [`get_state()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [`get_state_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [`vqe()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [`draw()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [`translate()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.translate){.reference
                .internal}
        -   [Backend
            Configuration](../../api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [`has_target()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [`get_target()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [`get_targets()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [`set_target()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [`reset_target()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [`set_noise()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [`unset_noise()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [`cudaq.apply_noise()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [`initialize_cudaq()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [`num_available_gpus()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [`set_random_seed()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](../../api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [`evolve()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [`evolve_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [`Schedule`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [`BaseIntegrator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [`InitialState`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [`InitialStateType`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [`IntermediateResultSave`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](../../api/languages/python_api.html#operators){.reference
            .internal}
            -   [`OperatorSum`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [`ProductOperator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [`ElementaryOperator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [`ScalarOperator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [`RydbergHamiltonian`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [`SuperOperator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [`operators.define()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [`operators.instantiate()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](../../api/languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](../../api/languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](../../api/languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](../../api/languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](../../api/languages/python_api.html#data-types){.reference
            .internal}
            -   [`SimulationPrecision`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [`Target`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [`State`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [`Tensor`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [`QuakeValue`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [`qubit`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [`qreg`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [`qvector`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [`ComplexMatrix`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [`SampleResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [`AsyncSampleResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [`ObserveResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [`AsyncObserveResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [`AsyncStateResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [`OptimizationResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [`EvolveResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [`AsyncEvolveResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [Optimizers](../../api/languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](../../api/languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](../../api/languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](../../api/languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [`initialize()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [`rank()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [`num_ranks()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [`all_gather()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [`broadcast()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [`is_initialized()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [`finalize()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](../../api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [`sample()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
    -   [Quantum Operations](../../api/default_ops.html){.reference
        .internal}
        -   [Unitary Operations on
            Qubits](../../api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [`x`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#x){.reference
                .internal}
            -   [`y`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#y){.reference
                .internal}
            -   [`z`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#z){.reference
                .internal}
            -   [`h`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#h){.reference
                .internal}
            -   [`r1`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#r1){.reference
                .internal}
            -   [`rx`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#rx){.reference
                .internal}
            -   [`ry`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#ry){.reference
                .internal}
            -   [`rz`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#rz){.reference
                .internal}
            -   [`s`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#s){.reference
                .internal}
            -   [`t`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#t){.reference
                .internal}
            -   [`swap`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#swap){.reference
                .internal}
            -   [`u3`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](../../api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](../../api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [`mz`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#mz){.reference
                .internal}
            -   [`mx`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#mx){.reference
                .internal}
            -   [`my`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](../../api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](../../api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [`create`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#create){.reference
                .internal}
            -   [`annihilate`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#annihilate){.reference
                .internal}
            -   [`phase_shift`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#phase-shift){.reference
                .internal}
            -   [`beam_splitter`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [`mz`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](../../versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](../../index.html)

::: {.wy-nav-content}
::: {.rst-content}
::: {role="navigation" aria-label="Page navigation"}
-   [](../../index.html){.icon .icon-home}
-   [CUDA-Q by Example](examples.html)
-   CUDA-Q Dynamics
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](hardware_providers.html "Using Quantum Hardware Providers"){.btn
.btn-neutral .float-left} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](../applications.html "CUDA-Q Applications"){.btn
.btn-neutral .float-right}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#cuda-q-dynamics .section}
CUDA-Q Dynamics[¶](#cuda-q-dynamics "Permalink to this heading"){.headerlink}
=============================================================================

This page contains a number of examples that use CUDA-Q dynamics to
simulate a range of fundamental physical systems and specific qubit
modalities. All example problems simulate systems of very low dimension
so that the code can be run quickly on any device. For small problems,
the GPU will not provide a significant performance advantage over the
CPU. The GPU will start to outperform the CPU for cases where the total
dimension of all subsystems is O(1000).

::: {#cavity-qed .section}
Cavity QED[¶](#cavity-qed "Permalink to this heading"){.headerlink}
-------------------------------------------------------------------

::: {#id1 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import boson, Schedule, ScipyZvodeIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # This example demonstrate a simulation of cavity quantum electrodynamics (interaction between light confined in a reflective cavity and atoms)

    # System dimensions: atom (2-level system) and cavity (10-level system)
    dimensions = {0: 2, 1: 10}

    # Alias for commonly used operators
    # Cavity operators
    a = boson.annihilate(1)
    a_dag = boson.create(1)

    # Atom operators
    sm = boson.annihilate(0)
    sm_dag = boson.create(0)

    # Defining the Hamiltonian for the system: self-energy terms and cavity-atom interaction term.
    # This is the so-called Jaynes-Cummings model:
    # https://en.wikipedia.org/wiki/Jaynes%E2%80%93Cummings_model
    hamiltonian = 2 * np.pi * boson.number(1) + 2 * np.pi * boson.number(
        0) + 2 * np.pi * 0.25 * (sm * a_dag + sm_dag * a)

    # Initial state of the system
    # Atom in ground state
    qubit_state = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128)

    # Cavity in a state which has 5 photons initially
    cavity_state = cp.zeros((10, 10), dtype=cp.complex128)
    cavity_state[5][5] = 1.0
    rho0 = cudaq.State.from_data(cp.kron(cavity_state, qubit_state))

    steps = np.linspace(0, 10, 201)
    schedule = Schedule(steps, ["time"])

    # First, evolve the system without any collapse operators (ideal).
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[boson.number(1), boson.number(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    # Then, evolve the system with a collapse operator modeling cavity decay (leaking photons)
    evolution_result_decay = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[boson.number(1), boson.number(0)],
        collapse_operators=[np.sqrt(0.1) * a],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]
    ideal_results = [
        get_result(0, evolution_result),
        get_result(1, evolution_result)
    ]
    decay_results = [
        get_result(0, evolution_result_decay),
        get_result(1, evolution_result_decay)
    ]

    fig = plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, ideal_results[0])
    plt.plot(steps, ideal_results[1])
    plt.ylabel("Expectation value")
    plt.xlabel("Time")
    plt.legend(("Cavity Photon Number", "Atom Excitation Probability"))
    plt.title("No decay")

    plt.subplot(1, 2, 2)
    plt.plot(steps, decay_results[0])
    plt.plot(steps, decay_results[1])
    plt.ylabel("Expectation value")
    plt.xlabel("Time")
    plt.legend(("Cavity Photon Number", "Atom Excitation Probability"))
    plt.title("With decay")

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    fig.savefig('cavity_qed.png', dpi=fig.dpi)
:::
:::
:::

::: {#cross-resonance .section}
Cross Resonance[¶](#cross-resonance "Permalink to this heading"){.headerlink}
-----------------------------------------------------------------------------

::: {#id2 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, Schedule, ScipyZvodeIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # This example simulates cross-resonance interactions between superconducting qubits.
    # Cross-resonance interaction is key to implementing two-qubit conditional gates, e.g., the CNOT gate.
    # Ref: A simple all-microwave entangling gate for fixed-frequency superconducting qubits (Physical Review Letters 107, 080502)
    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Device parameters
    # Detuning between two qubits
    delta = 100 * 2 * np.pi
    # Static coupling between qubits
    J = 7 * 2 * np.pi
    # spurious electromagnetic `crosstalk` due to stray electromagnetic coupling in the device circuit and package
    # see (Physical Review Letters 107, 080502)
    m_12 = 0.2
    # Drive strength
    Omega = 20 * 2 * np.pi

    # Qubit Hamiltonian (in the rotating frame w.r.t. the target qubit)
    hamiltonian = delta / 2 * spin.z(0) + J * (
        spin.minus(1) * spin.plus(0) +
        spin.plus(1) * spin.minus(0)) + Omega * spin.x(0) + m_12 * Omega * spin.x(1)

    # Dimensions of sub-system
    dimensions = {0: 2, 1: 2}

    # Two initial states: |00> and |10>.
    # We show the 'conditional' evolution when controlled qubit is in |1> state.
    psi_00 = cudaq.State.from_data(
        cp.array([1.0, 0.0, 0.0, 0.0], dtype=cp.complex128))
    psi_10 = cudaq.State.from_data(
        cp.array([0.0, 1.0, 0.0, 0.0], dtype=cp.complex128))

    # Schedule of time steps.
    steps = np.linspace(0.0, 1.0, 1001)
    schedule = Schedule(steps, ["time"])

    # Run the simulations (batched).
    evolution_results = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule, [psi_00, psi_10],
        observables=[
            spin.x(0),
            spin.y(0),
            spin.z(0),
            spin.x(1),
            spin.y(1),
            spin.z(1)
        ],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]
    results_00 = [get_result(i, evolution_results[0]) for i in range(6)]
    results_10 = [get_result(i, evolution_results[1]) for i in range(6)]

    # The changes in recession frequencies of the target qubit when control qubit is in |1> state allow us to implement two-qubit conditional gates.
    fig = plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, results_00[5])
    plt.plot(steps, results_10[5])
    plt.ylabel(r"$\langle Z_2\rangle$")
    plt.xlabel("Time")
    plt.legend((r"$|\psi_0\rangle=|00\rangle$", r"$|\psi_0\rangle=|10\rangle$"))

    plt.subplot(1, 2, 2)
    plt.plot(steps, results_00[4])
    plt.plot(steps, results_10[4])
    plt.ylabel(r"$\langle Y_2\rangle$")
    plt.xlabel("Time")
    plt.legend((r"$|\psi_0\rangle=|00\rangle$", r"$|\psi_0\rangle=|10\rangle$"))

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    fig.savefig('cross_resonance.png', dpi=fig.dpi)
:::
:::
:::

::: {#gate-calibration .section}
Gate Calibration[¶](#gate-calibration "Permalink to this heading"){.headerlink}
-------------------------------------------------------------------------------

::: {#id3 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import boson, Schedule, ScalarOperator, ScipyZvodeIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # This example demonstrates the use of the dynamics simulator and optimizer to optimize a pulse.
    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Sample device parameters
    # Assuming a simple transmon device Hamiltonian in rotating frame.
    detuning = 0.0  # Detuning of the drive; assuming resonant drive
    anharmonicity = -340.0  # Anharmonicity
    sigma = 0.01  # sigma of the Gaussian pulse
    cutoff = 4.0 * sigma  # total length of drive pulse

    # Dimensions of sub-system
    # We model `transmon` as a 3-level system to account for leakage.
    dimensions = {0: 3}

    # Initial state of the system (ground state).
    psi0 = cudaq.State.from_data(cp.array([1.0, 0.0, 0.0], dtype=cp.complex128))


    def gaussian(t):
        """
        Gaussian shape with cutoff. Starts at t = 0, amplitude normalized to one
        """
        val = (np.exp(-((t-cutoff/2)/sigma)**2/2)-np.exp(-(cutoff/sigma)**2/8)) \
               / (1-np.exp(-(cutoff/sigma)**2/8))
        return val


    def dgaussian(t):
        """
        Derivative of Gaussian. Starts at t = 0, amplitude normalized to one
        """
        return -(t - cutoff / 2) / sigma * np.exp(-(
            (t - cutoff / 2) / sigma)**2 / 2 + 0.5)


    # Schedule of time steps.
    steps = np.linspace(0.0, cutoff, 201)
    schedule = Schedule(steps, ["t"])

    # We optimize for a X(pi/2) rotation
    target_state = np.array([1.0 / np.sqrt(2), -1j / np.sqrt(2), 0.0],
                            dtype=cp.complex128)


    # Optimize the amplitude of the drive pulse (DRAG - Derivative Removal by Adiabatic Gate)
    def cost_function(amps):
        amplitude = 100 * amps[0]
        drag_amp = 100 * amps[1]
        # Qubit Hamiltonian
        hamiltonian = detuning * boson.number(0) + (
            anharmonicity / 2) * boson.create(0) * boson.create(
                0) * boson.annihilate(0) * boson.annihilate(0)
        # Drive term
        hamiltonian += amplitude * ScalarOperator(gaussian) * (boson.create(0) +
                                                               boson.annihilate(0))

        # Drag term (leakage reduction)
        hamiltonian += 1j * drag_amp * ScalarOperator(dgaussian) * (
            boson.annihilate(0) - boson.create(0))

        # We optimize for a X(pi/2) rotation
        evolution_result = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi0,
            observables=[],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.NONE,
            integrator=ScipyZvodeIntegrator())
        final_state = evolution_result.final_state()

        overlap = np.abs(final_state.overlap(target_state))
        print(
            f"Gaussian amplitude = {amplitude}, derivative amplitude = {drag_amp}, Overlap: {overlap}"
        )
        return 1.0 - overlap


    # Specify the optimizer
    optimizer = cudaq.optimizers.NelderMead()
    optimal_error, optimal_parameters = optimizer.optimize(dimensions=2,
                                                           function=cost_function)

    print("optimal overlap =", 1.0 - optimal_error)
    print("optimal parameters =", optimal_parameters)
:::
:::
:::

::: {#heisenberg-model .section}
Heisenberg Model[¶](#heisenberg-model "Permalink to this heading"){.headerlink}
-------------------------------------------------------------------------------

::: {#id4 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, Schedule, ScipyZvodeIntegrator

    import numpy as np
    import cupy as cp
    import matplotlib.pyplot as plt
    import os

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # In this example, we solve the Quantum Heisenberg model (https://en.wikipedia.org/wiki/Quantum_Heisenberg_model),
    # which exhibits the so-called quantum quench effect.
    # e.g., see `Quantum quenches in the anisotropic spin-1/2 Heisenberg chain: different approaches to many-body dynamics far from equilibrium`
    # (New J. Phys. 12 055017)

    # Specifically, we demonstrate the use of batched Hamiltonian operators to simulate the Heisenberg model
    # with different coupling strengths.
    # These batched Hamiltonian operators allow us to efficiently compute the dynamics of multiple systems in a single simulation run.
    # Number of spins
    N = 9
    dimensions = {}
    for i in range(N):
        dimensions[i] = 2

    # Initial state: alternating spin up and down
    spin_state = ''
    for i in range(N):
        spin_state += str(int(i % 2))

    # Observable is the staggered magnetization operator
    staggered_magnetization_op = spin.empty()
    for i in range(N):
        if i % 2 == 0:
            staggered_magnetization_op += spin.z(i)
        else:
            staggered_magnetization_op -= spin.z(i)

    staggered_magnetization_op /= N

    observe_results = []
    batched_hamiltonian = []
    anisotropy_parameters = [0.0, 0.25, 4.0]
    for g in anisotropy_parameters:
        # Heisenberg model spin coupling strength
        Jx = 1.0
        Jy = 1.0
        Jz = g

        # Construct the Hamiltonian
        H = spin.empty()

        for i in range(N - 1):
            H += Jx * spin.x(i) * spin.x(i + 1)
            H += Jy * spin.y(i) * spin.y(i + 1)
            H += Jz * spin.z(i) * spin.z(i + 1)
        # Append the Hamiltonian to the batched list
        batched_hamiltonian.append(H)

    steps = np.linspace(0.0, 5, 1000)
    schedule = Schedule(steps, ["time"])

    # Prepare the initial state vector
    psi0_ = cp.zeros(2**N, dtype=cp.complex128)
    psi0_[int(spin_state, 2)] = 1.0
    psi0 = cudaq.State.from_data(psi0_)

    # Run the simulation in batched mode
    evolution_results = cudaq.evolve(
        batched_hamiltonian,
        dimensions,
        schedule,
        psi0,  # Same initial state for all Hamiltonian operators
        observables=[staggered_magnetization_op],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    for g, evolution_result in zip(anisotropy_parameters, evolution_results):
        exp_val = [
            exp_vals[0].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]

        observe_results.append((g, exp_val))

    # Plot the results
    fig = plt.figure(figsize=(12, 6))
    for g, exp_val in observe_results:
        plt.plot(steps, exp_val, label=f'$ g = {g}$')
    plt.legend(fontsize=16)
    plt.ylabel("Staggered Magnetization")
    plt.xlabel("Time")
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    fig.savefig("heisenberg_model.png", dpi=fig.dpi)
:::
:::
:::

::: {#ion-trap .section}
Ion Trap[¶](#ion-trap "Permalink to this heading"){.headerlink}
---------------------------------------------------------------

::: {#id5 .highlight-python .notranslate}
::: {.highlight}
    """
    Tutorial: Creating a GHZ State with Trapped Ions

    This example shows how to prepare a GHZ (`Greenberger-Horne-Zeilinger`) state 
    using trapped ions and CUDA-Q. We'll use the effective spin model from the 
    famous `Sørensen-Mølmer` paper.

    What we're doing:
    - Start with N ions all in ground state `|gg...g⟩`
    - Apply an effective Hamiltonian H = 4χJ_x²  
    - Watch the system evolve into a GHZ superposition: `(|gg...g⟩ + |ee...e⟩)/√2`

    This tutorial uses the simplified effective model (`Eq. 2` from the paper).

    Reference:
    Mølmer, Klaus, and Anders `Sørensen`. "`Multiparticle` entanglement of hot trapped ions." Physical Review Letters 82.9 (1999): 1835.
    """

    import cudaq
    from cudaq import spin, Schedule, RungeKuttaIntegrator
    import numpy as np
    import cupy as cp
    import matplotlib.pyplot as plt

    cudaq.set_target("dynamics")

    # Physical parameters, these come from the `Sørensen-Mølmer` paper
    nu = 1.0  # Trap frequency (our reference)
    delta = 0.9 * nu  # Laser detuning
    Omega = 0.1 * nu  # Laser strength
    eta = 0.1  # How strongly lasers couple to motion

    # The effective coupling strength (this is the key parameter!)
    chi = (eta**2 * Omega**2 * nu) / (2 * (nu**2 - delta**2))

    N = 8  # Number of ions (start small!)
    evolution_time = 1.0 / chi  # How long to evolve
    num_steps = 100  # Time resolution

    dimensions = {i: 2 for i in range(N)}  # Each ion is a 2-level system

    # collective spin operator J_x
    J_x = spin.empty()
    for i in range(N):
        J_x += spin.x(i)
    J_x /= 2  # Normalize

    hamiltonian = 4 * chi * J_x * J_x

    # Set up initial state and time evolution
    # Start with all ions in ground state `|gg...g⟩`
    initial_state_vector = cp.zeros(2**N, dtype=cp.complex128)
    initial_state_vector[0] = 1.0  # |00...0⟩ = `|gg...g⟩`
    initial_state = cudaq.State.from_data(initial_state_vector)

    # Set up time points for evolution
    times = np.linspace(0, evolution_time, num_steps)
    chi_times = chi * times  # Dimensionless time χt
    schedule = Schedule(times, ["t"])

    # Create observables to track the populations
    # We want to measure the probability of being in `|gg...g⟩` and `|ee...e⟩`

    # Projector onto `|gg...g⟩ = |00...0⟩`
    P_ground = spin.empty()
    for i in range(N):
        if i == 0:
            P_ground = (spin.identity(i) - spin.z(i)) / 2  # `|0⟩⟨0|`
        else:
            P_ground = P_ground * (spin.identity(i) - spin.z(i)) / 2

    # Projector onto `|ee...e⟩ = |11...1⟩`
    P_excited = spin.empty()
    for i in range(N):
        if i == 0:
            P_excited = (spin.identity(i) + spin.z(i)) / 2  # `|1⟩⟨1|`
        else:
            P_excited = P_excited * (spin.identity(i) + spin.z(i)) / 2

    observables = [P_ground, P_excited]

    # Run the simulation!
    print("Running time evolution...")
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        initial_state,
        observables=observables,
        collapse_operators=[],  # No decoherence for this tutorial
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE,  # Save expectation values
        integrator=RungeKuttaIntegrator())

    # Extract the results
    exp_vals = evolution_result.expectation_values()
    pop_ground = [exp_vals[i][0].expectation() for i in range(len(times))]
    pop_excited = [exp_vals[i][1].expectation() for i in range(len(times))]

    # The GHZ state appears at a special time: χt = π/8
    ghz_chi_t = np.pi / 8
    ghz_time_idx = np.argmin(np.abs(chi_times - ghz_chi_t))
    ghz_pop_ground = pop_ground[ghz_time_idx]
    ghz_pop_excited = pop_excited[ghz_time_idx]

    print(f"\nResults at GHZ time (χt = {ghz_chi_t:.3f}):")
    print(f"P(|{'g'*N}⟩) = {ghz_pop_ground:.3f}")
    print(f"P(|{'e'*N}⟩) = {ghz_pop_excited:.3f}")
    print(f"Total in extremes: {ghz_pop_ground + ghz_pop_excited:.3f}")

    # Check GHZ state quality
    # For perfect GHZ state: `P(gg) = P(ee) = 0.5`
    ghz_quality = 1 - 2 * abs(
        ghz_pop_ground - 0.5)  # Distance from ideal probability
    print(f"GHZ state quality: {ghz_quality:.3f} (1.0 = perfect)")
    print(f"Perfect GHZ would have P(gg) = P(ee) = 0.5")

    plt.figure(figsize=(10, 6))
    plt.plot(chi_times,
             pop_ground,
             'b-',
             linewidth=2,
             label=f"|{'g'*N}⟩ (all ground)")
    plt.plot(chi_times,
             pop_excited,
             'r-',
             linewidth=2,
             label=f"|{'e'*N}⟩ (all excited)")
    plt.axvline(ghz_chi_t,
                color='gray',
                linestyle='--',
                alpha=0.7,
                label="GHZ time")
    plt.axhline(0.5, color='black', linestyle=':', alpha=0.5, label="Perfect GHZ")

    plt.xlabel("Dimensionless time (χt)")
    plt.ylabel("Population")
    plt.title(f"Evolution to GHZ State ({N} trapped ions)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("ghz_trapped_ions.png")
    plt.show()
:::
:::
:::

::: {#landau-zener .section}
Landau Zener[¶](#landau-zener "Permalink to this heading"){.headerlink}
-----------------------------------------------------------------------

::: {#id6 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, boson, ScalarOperator, Schedule, ScipyZvodeIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # This example simulates the so-called Landau–Zener transition: given a time-dependent Hamiltonian such that the energy separation
    # of the two states is a linear function of time, an analytical formula exists to calculate
    # the probability of finding the system in the excited state after the transition.

    # References:
    # - https://en.wikipedia.org/wiki/Landau%E2%80%93Zener_formula
    # - `The Landau-Zener formula made simple`, `Eric P Glasbrenner and Wolfgang P Schleich 2023 J. Phys. B: At. Mol. Opt. Phys. 56 104001`
    # - QuTiP notebook: https://github.com/qutip/qutip-notebooks/blob/master/examples/landau-zener.ipynb

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Define some shorthand operators
    sx = spin.x(0)
    sz = spin.z(0)
    sm = boson.annihilate(0)
    sm_dag = boson.create(0)

    # Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
    dimensions = {0: 2}

    # Landau–Zener Hamiltonian:
    # `[[-alpha*t, g], [g, alpha*t]] = g * pauli_x - alpha * t * pauli_z`
    g = 2 * np.pi
    # Analytical equation:
    # `P(0) = exp(-pi * g ^ 2/ alpha)`
    # The target ground state probability that we want to achieve
    target_p0 = 0.75
    # Compute `alpha` parameter:
    alpha = (-np.pi * g**2) / np.log(target_p0)

    # Hamiltonian
    hamiltonian = g * sx - alpha * ScalarOperator(lambda t: t) * sz

    # Initial state of the system (ground state)
    psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

    # Schedule of time steps (simulating a long time range)
    steps = np.linspace(-2.0, 2.0, 5000)
    schedule = Schedule(steps, ["t"])

    # Run the simulation.
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[boson.number(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    prob1 = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]

    prob0 = [1 - val for val in prob1]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(steps, prob1, 'b', steps, prob0, 'r')
    ax.plot(steps, (1.0 - target_p0) * np.ones(np.shape(steps)), 'k')
    ax.plot(steps, target_p0 * np.ones(np.shape(steps)), 'm')
    ax.set_xlabel("Time")
    ax.set_ylabel("Occupation probability")
    ax.set_title("Landau-Zener transition")
    ax.legend(("Excited state", "Ground state", "LZ formula (Excited state)",
               "LZ formula (Ground state)"),
              loc=0)

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    plt.savefig('landau_zener.png', dpi=fig.dpi)
:::
:::
:::

::: {#pulse .section}
Pulse[¶](#pulse "Permalink to this heading"){.headerlink}
---------------------------------------------------------

::: {#id7 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, boson, ScalarOperator, Schedule, ScipyZvodeIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # This example simulates time evolution of a qubit (`transmon`) being driven by a pulse.
    # The pulse is a modulated signal with a Gaussian envelop.
    # The simulation is performed in the 'lab' frame.

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Device parameters
    # Strength of the Rabi-rate in GHz.
    rabi_rate = 0.1

    # Frequency of the qubit transition in GHz.
    omega = 5.0 * 2 * np.pi

    # Define Gaussian envelope function to approximately implement a `rx(pi/2)` gate.
    amplitude = 1. / 2.0  # Pi/2 rotation
    sigma = 1.0 / rabi_rate / amplitude
    pulse_duration = 6 * sigma


    def gaussian(t, duration, amplitude, sigma):
        # Gaussian envelope function
        return amplitude * np.exp(-0.5 * (t - duration / 2)**2 / (sigma)**2)


    def signal(t):
        # Modulated signal
        return np.cos(omega * t) * gaussian(t, pulse_duration, amplitude, sigma)


    # Qubit Hamiltonian
    hamiltonian = omega * spin.z(0) / 2
    # Add modulated driving term to the Hamiltonian
    hamiltonian += np.pi * rabi_rate * ScalarOperator(signal) * spin.x(0)

    # Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
    dimensions = {0: 2}

    # Initial state of the system (ground state).
    psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

    # Schedule of time steps.
    # Since this is a lab-frame simulation, the time step must be small to accurately capture the modulated signal.
    dt = 1 / omega / 20
    n_steps = int(np.ceil(pulse_duration / dt)) + 1
    steps = np.linspace(0, pulse_duration, n_steps)
    schedule = Schedule(steps, ["t"])

    # Run the simulation.
    # First, we run the simulation without any collapse operators (no decoherence).
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[boson.number(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    pop1 = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]
    pop0 = [1.0 - x for x in pop1]
    fig = plt.figure(figsize=(6, 16))
    envelop = [gaussian(t, pulse_duration, amplitude, sigma) for t in steps]

    plt.subplot(3, 1, 1)
    plt.plot(steps, envelop)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title("Envelope")

    modulated_signal = [
        np.cos(omega * t) * gaussian(t, pulse_duration, amplitude, sigma)
        for t in steps
    ]
    plt.subplot(3, 1, 2)
    plt.plot(steps, modulated_signal)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title("Signal")

    plt.subplot(3, 1, 3)
    plt.plot(steps, pop0)
    plt.plot(steps, pop1)
    plt.ylabel("Population")
    plt.xlabel("Time")
    plt.legend(("Population in |0>", "Population in |1>"))
    plt.title("Qubit State")

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    fig.savefig("pulse.png", dpi=fig.dpi)
:::
:::
:::

::: {#qubit-control .section}
Qubit Control[¶](#qubit-control "Permalink to this heading"){.headerlink}
-------------------------------------------------------------------------

::: {#id8 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, ScalarOperator, Schedule, ScipyZvodeIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # This example simulates time evolution of a qubit (`transmon`) being driven close to resonance in the presence of noise (decoherence).
    # Thus, it exhibits Rabi oscillations.
    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Qubit Hamiltonian reference: https://qiskit-community.github.io/qiskit-dynamics/tutorials/Rabi_oscillations.html
    # Device parameters
    # Qubit resonant frequency
    omega_z = 10.0 * 2 * np.pi
    # Transverse term
    omega_x = 2 * np.pi
    # Harmonic driving frequency
    # Note: we chose a frequency slightly different from the resonant frequency to demonstrate the off-resonance effect.
    omega_drive = 0.99 * omega_z

    # Qubit Hamiltonian
    hamiltonian = 0.5 * omega_z * spin.z(0)
    # Add modulated driving term to the Hamiltonian
    hamiltonian += omega_x * ScalarOperator(
        lambda t: np.cos(omega_drive * t)) * spin.x(0)

    # Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
    dimensions = {0: 2}

    # Initial state of the system (ground state).
    rho0 = cudaq.State.from_data(
        cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

    # Schedule of time steps.
    t_final = np.pi / omega_x
    dt = 2.0 * np.pi / omega_drive / 100
    n_steps = int(np.ceil(t_final / dt)) + 1
    steps = np.linspace(0, t_final, n_steps)
    schedule = Schedule(steps, ["t"])

    # Run the simulation.
    # First, we run the simulation without any collapse operators (no decoherence).
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.x(0), spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    # Now, run the simulation with qubit decoherence
    gamma_sm = 4.0
    gamma_sz = 1.0
    evolution_result_decay = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.x(0), spin.y(0), spin.z(0)],
        collapse_operators=[
            np.sqrt(gamma_sm) * spin.plus(0),
            np.sqrt(gamma_sz) * spin.z(0)
        ],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]
    ideal_results = [
        get_result(0, evolution_result),
        get_result(1, evolution_result),
        get_result(2, evolution_result)
    ]
    decoherence_results = [
        get_result(0, evolution_result_decay),
        get_result(1, evolution_result_decay),
        get_result(2, evolution_result_decay)
    ]

    fig = plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, ideal_results[0])
    plt.plot(steps, ideal_results[1])
    plt.plot(steps, ideal_results[2])
    plt.ylabel("Expectation value")
    plt.xlabel("Time")
    plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
    plt.title("No decoherence")

    plt.subplot(1, 2, 2)
    plt.plot(steps, decoherence_results[0])
    plt.plot(steps, decoherence_results[1])
    plt.plot(steps, decoherence_results[2])
    plt.ylabel("Expectation value")
    plt.xlabel("Time")
    plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
    plt.title("With decoherence")

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    fig.savefig('qubit_control.png', dpi=fig.dpi)
:::
:::
:::

::: {#qubit-dynamics .section}
Qubit Dynamics[¶](#qubit-dynamics "Permalink to this heading"){.headerlink}
---------------------------------------------------------------------------

::: {#id9 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, Schedule, RungeKuttaIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

    # Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
    dimensions = {0: 2}

    # Initial state of the system (ground state).
    rho0 = cudaq.State.from_data(
        cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

    # Schedule of time steps.
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["time"])

    # Run the simulation.
    # First, we run the simulation without any collapse operators (ideal).
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=RungeKuttaIntegrator())

    # Now, run the simulation with qubit decaying due to the presence of a collapse operator.
    evolution_result_decay = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[np.sqrt(0.05) * spin.x(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=RungeKuttaIntegrator())

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]
    ideal_results = [
        get_result(0, evolution_result),
        get_result(1, evolution_result)
    ]
    decay_results = [
        get_result(0, evolution_result_decay),
        get_result(1, evolution_result_decay)
    ]

    fig = plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, ideal_results[0])
    plt.plot(steps, ideal_results[1])
    plt.ylabel("Expectation value")
    plt.xlabel("Time")
    plt.legend(("Sigma-Y", "Sigma-Z"))
    plt.title("No decay")

    plt.subplot(1, 2, 2)
    plt.plot(steps, decay_results[0])
    plt.plot(steps, decay_results[1])
    plt.ylabel("Expectation value")
    plt.xlabel("Time")
    plt.legend(("Sigma-Y", "Sigma-Z"))
    plt.title("With decay")

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    fig.savefig('qubit_dynamics.png', dpi=fig.dpi)
:::
:::
:::

::: {#silicon-spin-qubit .section}
Silicon Spin Qubit[¶](#silicon-spin-qubit "Permalink to this heading"){.headerlink}
-----------------------------------------------------------------------------------

::: {#id10 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, boson, Schedule, ScalarOperator, ScipyZvodeIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # This example demonstrates simulation of an electrically-driven silicon spin qubit.
    # The system dynamics is taken from https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.19.044078

    dimensions = {0: 2}
    resonance_frequency = 2 * np.pi * 10  # 10 Ghz

    # Run the simulation:

    # Sweep the amplitude
    amplitudes = np.linspace(0.0, 0.5, 20)
    # Construct a list of Hamiltonian operator for each amplitude so that we can batch them all together
    batched_hamiltonian = []
    for amplitude in amplitudes:
        # Electric dipole spin resonance (`EDSR`) Hamiltonian
        H = 0.5 * resonance_frequency * spin.z(0) + amplitude * ScalarOperator(
            lambda t: 0.5 * np.sin(resonance_frequency * t)) * spin.x(0)
        # Append the Hamiltonian to the batched list
        # This allows us to compute the dynamics for all amplitudes in a single simulation run
        batched_hamiltonian.append(H)

    # Initial state is the ground state of the spin qubit
    # We run all simulations for the same initial state, but with different Hamiltonian operators.
    psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

    # Simulation schedule
    t_final = 100
    dt = 0.005
    n_steps = int(np.ceil(t_final / dt)) + 1
    steps = np.linspace(0, t_final, n_steps)
    schedule = Schedule(steps, ["t"])

    results = cudaq.evolve(
        batched_hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[boson.number(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]

    evolution_results = []
    for result in results:
        evolution_results.append(get_result(0, result))

    fig, ax = plt.subplots()
    im = ax.contourf(steps, amplitudes, evolution_results)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(f"Amplitude (a.u.)")
    fig.suptitle(f"Excited state probability")
    fig.colorbar(im)
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    # For reference, see figure 5 in https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.19.044078
    fig.savefig("spin_qubit_edsr.png", dpi=fig.dpi)
:::
:::
:::

::: {#tensor-callback .section}
Tensor Callback[¶](#tensor-callback "Permalink to this heading"){.headerlink}
-----------------------------------------------------------------------------

::: {#id11 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import MatrixOperatorElement, operators, boson, Schedule, ScipyZvodeIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # This example demonstrates the use of callback functions to define time-dependent operators.

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Consider a simple 2-level system Hamiltonian exhibits the Landau–Zener transition:
    # `[[-alpha*t, g], [g, alpha*t]]
    # This can be defined as a callback tensor:
    g = 2.0 * np.pi
    alpha = 10.0 * 2 * np.pi


    def callback_tensor(t):
        return np.array([[-alpha * t, g], [g, alpha * t]], dtype=np.complex128)


    # Analytical formula
    lz_formula_p0 = np.exp(-np.pi * g**2 / (alpha))
    lz_formula_p1 = 1.0 - lz_formula_p0

    # Let's define the control term as a callback tensor that acts on 2-level systems
    operators.define("lz_op", [2], callback_tensor)

    # Hamiltonian
    hamiltonian = operators.instantiate("lz_op", [0])

    # Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
    dimensions = {0: 2}

    # Initial state of the system (ground state)
    psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

    # Schedule of time steps (simulating a long time range)
    steps = np.linspace(-4.0, 4.0, 10000)
    schedule = Schedule(steps, ["t"])

    # Run the simulation.
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[boson.number(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    prob1 = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]

    prob0 = [1 - val for val in prob1]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(steps, prob1, 'b', steps, prob0, 'r')
    ax.plot(steps, lz_formula_p1 * np.ones(np.shape(steps)), 'k')
    ax.plot(steps, lz_formula_p0 * np.ones(np.shape(steps)), 'm')
    ax.set_xlabel("Time")
    ax.set_ylabel("Occupation probability")
    ax.set_title("Landau-Zener transition")
    ax.legend(("Excited state", "Ground state", "LZ formula (Excited state)",
               "LZ formula (Ground state)"),
              loc=0)

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    fig.savefig('tensor_callback.png', dpi=fig.dpi)
:::
:::
:::

::: {#transmon-resonator .section}
Transmon Resonator[¶](#transmon-resonator "Permalink to this heading"){.headerlink}
-----------------------------------------------------------------------------------

::: {#id12 .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import operators, spin, operators, Schedule, ScipyZvodeIntegrator
    import numpy as np
    import cupy as cp
    import os
    import matplotlib.pyplot as plt

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # This example demonstrates a simulation of a superconducting transmon qubit coupled to a resonator (i.e., cavity).
    # References:
    # - "Charge-insensitive qubit design derived from the Cooper pair box", PRA 76, 042319
    # - QuTiP lecture: https://github.com/jrjohansson/qutip-lectures/blob/master/Lecture-10-cQED-dispersive-regime.ipynb

    # Number of cavity photons
    N = 20

    # System dimensions: transmon + cavity
    dimensions = {0: 2, 1: N}

    # See III.B of PRA 76, 042319
    # System parameters
    # Unit: GHz
    omega_01 = 3.0 * 2 * np.pi  # transmon qubit frequency
    omega_r = 2.0 * 2 * np.pi  # resonator frequency
    # Dispersive shift
    chi_01 = 0.025 * 2 * np.pi
    chi_12 = 0.0

    omega_01_prime = omega_01 + chi_01
    omega_r_prime = omega_r - chi_12 / 2.0
    chi = chi_01 - chi_12 / 2.0

    # System Hamiltonian
    hamiltonian = 0.5 * omega_01_prime * spin.z(0) + (
        omega_r_prime + chi * spin.z(0)) * operators.number(1)

    # Initial state of the system
    # Transmon in a superposition state
    transmon_state = cp.array([1. / np.sqrt(2.), 1. / np.sqrt(2.)],
                              dtype=cp.complex128)


    # Helper to create a coherent state in Fock basis truncated at `num_levels`.
    # Note: There are a couple of ways of generating a coherent state,
    # e.g., see https://qutip.readthedocs.io/en/v5.0.3/apidoc/functions.html#qutip.core.states.coherent
    # or https://en.wikipedia.org/wiki/Coherent_state
    # Here, in this example, we use a the formula: `|alpha> = D(alpha)|0>`,
    # i.e., apply the displacement operator on a zero (or vacuum) state to compute the corresponding coherent state.
    def coherent_state(num_levels, amplitude):
        displace_mat = operators.displace(0).to_matrix({0: num_levels},
                                                       displacement=amplitude)
        # `D(alpha)|0>` is the first column of `D(alpha)` matrix
        return cp.array(np.transpose(displace_mat)[0])


    # Cavity in a coherent state
    cavity_state = coherent_state(N, 2.0)
    psi0 = cudaq.State.from_data(cp.kron(transmon_state, cavity_state))

    steps = np.linspace(0, 250, 1000)
    schedule = Schedule(steps, ["time"])

    # Evolve the system
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[
            operators.number(1),
            operators.number(0),
            operators.position(1),
            operators.position(0)
        ],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator())

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]
    count_results = [
        get_result(0, evolution_result),
        get_result(1, evolution_result)
    ]

    quadrature_results = [
        get_result(2, evolution_result),
        get_result(3, evolution_result)
    ]

    fig = plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, count_results[0])
    plt.plot(steps, count_results[1])
    plt.ylabel("n")
    plt.xlabel("Time [ns]")
    plt.legend(("Cavity Photon Number", "Transmon Excitation Probability"))
    plt.title("Excitation Numbers")

    plt.subplot(1, 2, 2)
    plt.plot(steps, quadrature_results[0])
    plt.ylabel("x")
    plt.xlabel("Time [ns]")
    plt.legend(("cavity"))
    plt.title("Resonator Quadrature")

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    fig.savefig('transmon_resonator.png', dpi=fig.dpi)
:::
:::
:::

::: {#initial-state-multi-gpu-multi-node .section}
Initial State (Multi-GPU Multi-Node)[¶](#initial-state-multi-gpu-multi-node "Permalink to this heading"){.headerlink}
---------------------------------------------------------------------------------------------------------------------

::: {#initial-state-mgmn .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, Schedule, RungeKuttaIntegrator

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # On a system with multiple GPUs, `mpiexec` can be used as follows:
    # `mpiexec -np <N> python3 multi_gpu.py `
    cudaq.mpi.initialize()

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # Large number of spins
    N = 20
    dimensions = {}
    for i in range(N):
        dimensions[i] = 2

    # Observable is the average magnetization operator
    avg_magnetization_op = spin.empty()
    for i in range(N):
        avg_magnetization_op += (spin.z(i) / N)

    # Arbitrary coupling constant
    g = 1.0
    # Construct the Hamiltonian
    H = spin.empty()
    for i in range(N):
        H += 2 * np.pi * spin.x(i)
        H += 2 * np.pi * spin.y(i)
    for i in range(N - 1):
        H += 2 * np.pi * g * spin.x(i) * spin.x(i + 1)
        H += 2 * np.pi * g * spin.y(i) * spin.z(i + 1)

    steps = np.linspace(0.0, 1, 200)
    schedule = Schedule(steps, ["time"])

    # Initial state (expressed as an enum)
    psi0 = cudaq.dynamics.InitialState.ZERO
    # This can also be used to initialize a uniformly-distributed wave-function instead.
    # `psi0 = cudaq.dynamics.InitialState.UNIFORM`

    # Run the simulation
    evolution_result = cudaq.evolve(
        H,
        dimensions,
        schedule,
        psi0,
        observables=[avg_magnetization_op],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=RungeKuttaIntegrator())

    exp_val = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]

    if cudaq.mpi.rank() == 0:
        # Plot the results
        fig = plt.figure(figsize=(12, 6))
        plt.plot(steps, exp_val)
        plt.ylabel("Average Magnetization")
        plt.xlabel("Time")
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        fig.savefig("spin_model.png", dpi=fig.dpi)

    cudaq.mpi.finalize()
:::
:::
:::

::: {#heisenberg-model-multi-gpu-multi-node .section}
Heisenberg Model (Multi-GPU Multi-Node)[¶](#heisenberg-model-multi-gpu-multi-node "Permalink to this heading"){.headerlink}
---------------------------------------------------------------------------------------------------------------------------

::: {#heisenberg-model-mgmn .highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin, Schedule, RungeKuttaIntegrator

    import numpy as np
    import cupy as cp
    import matplotlib.pyplot as plt
    import os

    # On a system with multiple GPUs, `mpiexec` can be used as follows:
    # `mpiexec -np <N> python3 multi_gpu.py `
    cudaq.mpi.initialize()

    # Set the target to our dynamics simulator
    cudaq.set_target("dynamics")

    # In this example, we solve the Quantum Heisenberg model (https://en.wikipedia.org/wiki/Quantum_Heisenberg_model),
    # which exhibits the so-called quantum quench effect.
    # e.g., see `Quantum quenches in the anisotropic spin-1/2 Heisenberg chain: different approaches to many-body dynamics far from equilibrium`
    # (New J. Phys. 12 055017)
    # Large number of spins
    N = 21
    dimensions = {}
    for i in range(N):
        dimensions[i] = 2

    # Initial state: alternating spin up and down
    spin_state = ''
    for i in range(N):
        spin_state += str(int(i % 2))

    # Observable is the staggered magnetization operator
    staggered_magnetization_op = spin.empty()
    for i in range(N):
        if i % 2 == 0:
            staggered_magnetization_op += spin.z(i)
        else:
            staggered_magnetization_op -= spin.z(i)

    staggered_magnetization_op /= N

    observe_results = []
    batched_hamiltonian = []
    anisotropy_parameters = [0.25, 4.0]
    for g in anisotropy_parameters:
        # Heisenberg model spin coupling strength
        Jx = 1.0
        Jy = 1.0
        Jz = g

        # Construct the Hamiltonian
        H = spin.empty()

        for i in range(N - 1):
            H += Jx * spin.x(i) * spin.x(i + 1)
            H += Jy * spin.y(i) * spin.y(i + 1)
            H += Jz * spin.z(i) * spin.z(i + 1)

        # Append the Hamiltonian to the batched list
        batched_hamiltonian.append(H)

    steps = np.linspace(0.0, 5, 500)
    schedule = Schedule(steps, ["time"])

    # Prepare the initial state vector
    psi0_ = cp.zeros(2**N, dtype=cp.complex128)
    psi0_[int(spin_state, 2)] = 1.0
    psi0 = cudaq.State.from_data(psi0_)

    # Run the simulation in batched mode
    # This allows us to compute the dynamics for all Hamiltonian operators in a single simulation run
    evolution_results = cudaq.evolve(
        batched_hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[staggered_magnetization_op],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=RungeKuttaIntegrator())

    for g, evolution_result in zip(anisotropy_parameters, evolution_results):
        exp_val = [
            exp_vals[0].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]

        observe_results.append((g, exp_val))

    if cudaq.mpi.rank() == 0:
        # Plot the results
        fig = plt.figure(figsize=(12, 6))
        for g, exp_val in observe_results:
            plt.plot(steps, exp_val, label=f'$ g = {g}$')
        plt.legend(fontsize=16)
        plt.ylabel("Staggered Magnetization")
        plt.xlabel("Time")
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        fig.savefig("heisenberg_model_mgpu.png", dpi=fig.dpi)

    cudaq.mpi.finalize()
:::
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](hardware_providers.html "Using Quantum Hardware Providers"){.btn
.btn-neutral .float-left} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](../applications.html "CUDA-Q Applications"){.btn
.btn-neutral .float-right}
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
