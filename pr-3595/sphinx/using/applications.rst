CUDA-Q Applications
====================

This page contains a number of different applications implemented using CUDA-Q. All notebooks can be found `here. <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/applications/python>`_


.. when adding applications
.. 1. Add notebook to the hidden TOC list directly below
.. 2. Add an html block along with any tags
.. 3. Add a preview image in the _static folder

.. |:spellcheck-disable:| replace:: \

.. toctree::
   :maxdepth: 1
   :hidden:

   /applications/python/qaoa
   /applications/python/digitized_counterdiabatic_qaoa
   /applications/python/krylov
   /applications/python/qsci
   /applications/python/bernstein_vazirani
   /applications/python/cost_minimization
   /applications/python/deutsch_algorithm
   /applications/python/divisive_clustering_coresets
   /applications/python/hybrid_quantum_neural_networks
   /applications/python/hadamard_test
   /applications/python/logical_aim_sqale
   /applications/python/hamiltonian_simulation
   /applications/python/quantum_fourier_transform
   /applications/python/quantum_teleportation
   /applications/python/quantum_volume
   /applications/python/readout_error_mitigation
   /applications/python/unitary_compilation_diffusion_models
   /applications/python/vqe_advanced
   /applications/python/quantum_transformer
   /applications/python/afqmc
   /applications/python/adapt_qaoa
   /applications/python/adapt_vqe
   /applications/python/edge_detection
   /applications/python/shors
   /applications/python/generate_fermionic_ham
   /applications/python/grovers
   /applications/python/quantum_pagerank
   /applications/python/uccsd_wf_ansatz
   /applications/python/mps_encoding
   /applications/python/qm_mm_pe
   /applications/python/skqd
   /applications/python/entanglement_acc_hamiltonian_simulation

.. |:spellcheck-enable:| replace:: \

.. raw:: html

    <div class="filter-groups" style="margin-bottom: 60px;">
        <div class="tag-filters">
            <h3>Filter by Domain:</h3>
            <button class="tag-button active" data-group="domain" data-tag="all">All</button>
            <button class="tag-button" data-group="domain" data-tag="optimization">Optimization</button>
            <button class="tag-button" data-group="domain" data-tag="chemistry">Chemistry</button>
            <button class="tag-button" data-group="domain" data-tag="fundamental">Fundamental Algorithms</button>
            <button class="tag-button" data-group="domain" data-tag="aiforq">AI for Quantum</button>
            <button class="tag-button" data-group="domain" data-tag="qforai">Quantum for AI</button>
            <button class="tag-button" data-group="domain" data-tag="dynamics">Dynamics</button>
            <button class="tag-button" data-group="domain" data-tag="collab">Community</button>
        </div>
        

        <div class="tag-filters" style="margin-top: 20px;">
    <h3>Filter by Backend:</h3>
    <div class="backend-group">
        <button class="tag-button backend-toggle" data-tag="noiseless">Noiseless Simulator</button>
        <div class="backend-options">
            <button class="tag-button sub-option" data-tag="cpu">CPU</button>
            <button class="tag-button sub-option" data-tag="gpu">Single GPU</button>
            <button class="tag-button sub-option" data-tag="mgpu">Multi-GPU</button>
            <button class="tag-button sub-option" data-tag="mqpu">Multi-QPU</button>
        </div>
    </div>
    <div class="backend-group">
        <button class="tag-button backend-toggle" data-tag="noisy">Noisy Simulator</button>
        <div class="backend-options">
            <button class="tag-button sub-option" data-tag="density">Density Matrix</button>
        </div>
    </div>
    <div class="backend-group">
        <button class="tag-button backend-toggle" data-tag="qpu">QPUs</button>
        <div class="backend-options">
            <button class="tag-button sub-option" data-tag="neutral">Neutral Atom</button>
        </div>
    </div>
   </div>
        
.. raw:: html

    <div class="notebook-entry" data-tags="optimization,noiseless,gpu">
        <a href="../applications/python/qaoa.html" class="notebook-title">QAOA for Max Cut Problem</a>
        <div class="notebook-content">
            Learn the theory behind the Quantum Approximate Optimization Algorithm (QAOA) and how it can be used to solve the Max Cut problem.
        </div>
        <img src="../_static/app_title_images/qaoa_preview.png" alt="QAOA Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,optimization,noiseless,gpu">
        <a href="../applications/python/digitized_counterdiabatic_qaoa.html" class="notebook-title">Digitized Counterdiabatic QAOA</a>
        <div class="notebook-content">
            Learn how the DC-QAOA algorithm is used to predict molecules that might be good candidates for drugs based on their interactions with proteins.
        </div>
        <img src="../_static/app_title_images/dcqaoa_preview.png" alt="DC-QAOA Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,noiseless,mqpu">
        <a href="../applications/python/krylov.html" class="notebook-title">Krylov Subspace Methods</a>
        <div class="notebook-content">
            Learn how the Krylov method uses the Hadamard test to predict the ground state energy of molecules. Also learn how to implement the same approach with the <code>mqpu</code> backend and simulate execution on multiple QPUs in parallel.
        </div>
        <img src="../_static/app_title_images/krylov_preview.png" alt="Krylov Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,noiseless,gpu">
        <a href="../applications/python/qsci.html" class="notebook-title">Quantum-Selected Configuration Interaction</a>
        <div class="notebook-content">
            Learn how the QSCI method uses the observe and sample primitives.
        </div>
        <img src="../_static/app_title_images/qsci_preview.png" alt="QSCI Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,noiseless">
        <a href="../applications/python/skqd.html" class="notebook-title">Sample-Based Krylov Quantum Diagonalization (SKQD)</a>
        <div class="notebook-content">
            Learn how to implement the Sample-Based Krylov Quantum Diagonalization (SKQD) algorithm to predict the ground state energy of molecules.
        </div>
    </div>

    <div class="notebook-entry" data-tags="cpu,fundamental,noiseless,gpu">
        <a href="../applications/python/bernstein_vazirani.html" class="notebook-title">The Bernstein-Vazirani Algorithm</a>
        <div class="notebook-content">
            Learn a famous quantum algorithm that provides intuition for why exponential speedups can be achieved with quantum computers.
        </div>
        <img src="../_static/app_title_images/bv_preview.png" alt="BV Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="cpu,fundamental,noiseless,gpu">
        <a href="../applications/python/deutsch_algorithm.html" class="notebook-title">Deutsch's Algorithm</a>
        <div class="notebook-content">
            Learn how quantum computers can provide an exponential speedup for identifying if a Boolean function is constant or balanced.
        </div>
        <img src="../_static/app_title_images/dj_preview.png" alt="DJ Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="qforai,noiseless,mgpu,collab">
        <a href="../applications/python/divisive_clustering_coresets.html" class="notebook-title">Divisive Clustering with Coresets</a>
        <div class="notebook-content">
            Explore an implementation of the work in this paper (https://arxiv.org/abs/2402.01529) which looks at ways to cluster large data sets on quantum computers using a data reduction technique called coresets. This notebook includes the full workflow, a QAOA implementation, and an example of using the <code>mgpu</code> backend to scale the problem to greater qubit numbers.
        </div>
        <img src="../_static/app_title_images/clustering_preview.png" alt="Clustering Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="qforai,noiseless,gpu">
        <a href="../applications/python/hybrid_quantum_neural_networks.html" class="notebook-title">Hybrid Quantum Neural Networks</a>
        <div class="notebook-content">
            Learn how to implement Neural Network composed of a traditional PyTorch layer and a quantum layer added with CUDA-Q
        </div>
        <img src="../_static/app_title_images/hqnn_preview.png" alt="HQNN Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="fundamental,noiseless,gpu,mqpu">
        <a href="../applications/python/hadamard_test.html" class="notebook-title">The Hadamard Test</a>
        <div class="notebook-content">
            Learn about the Hadamard test and how it can be used to estimate expectation values. This notebook also explores how the Hadamard test can be used for Krylov subspace method and accelerated with the <code>mqpu</code> backend to evaluate execution on multiple simulated QPUs in parallel.
        </div>
        <img src="../_static/app_title_images/hadamard_preview.png" alt="Hadamard Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,qec,qpu,neutral,collab">
        <a href="../applications/python/logical_aim_sqale.html" class="notebook-title">The Anderson Impurity Model With Logical Qubits</a>
        <div class="notebook-content">
            A collaboration between NVIDIA and Infleqtion demonstrated a logical qubit workflow built in CUDA-Q and executed on the Infleqtion's neutral atom QPU. (https://arxiv.org/abs/2412.07670)
        </div>
        <img src="../_static/app_title_images/aim_preview.png" alt="AIM Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="fundamental,noiseless,gpu">
        <a href="../applications/python/hamiltonian_simulation.html" class="notebook-title">Trotterized Hamiltonian Simulation</a>
        <div class="notebook-content">
            Trotterization is an approximation to enable simulation of a Hamiltonian. Learn how this technique works and simulate the dynamics of the Heisenberg model.
        </div>
        <img src="../_static/app_title_images/trotter_preview.png" alt="Trotter Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="cpu,fundamental,noiseless,gpu">
        <a href="../applications/python/quantum_fourier_transform.html" class="notebook-title">The Quantum Fourier Transform</a>
        <div class="notebook-content">
            The Quantum Fourier transform (QFT) is a fundamental quantum algorithm that is also an important subroutine of quantum phase estimation, Shor's, and other quantum algorithms. Learn the basics of the QFT and how to implement it in CUDA-Q.
        </div>
        <img src="../_static/app_title_images/qft_preview.png" alt="QFT Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="cpu,fundamental,noiseless,gpu">
        <a href="../applications/python/quantum_teleportation.html" class="notebook-title">Quantum Teleportation</a>
        <div class="notebook-content">
            Quantum teleportation is one of the strange phenomena that makes quantum computing so interesting. Learn how teleportation works and how it is implemented in CUDA-Q.
        </div>
        <img src="../_static/app_title_images/teleport_preview.png" alt="Quantum Teleportation Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="fundamental,noisy,density">
        <a href="../applications/python/quantum_volume.html" class="notebook-title">Quantum Volume</a>
        <div class="notebook-content">
            Benchmarking the performance of quantum computers, especially between different qubit modalities, is challenging. One method is to experimentally perform the quantum volume test. Learn how this test is performed and how it is implemented in CUDA-Q.
        </div>
        <img src="../_static/app_title_images/qv_preview.png" alt="Quantum Volume Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="noisy,density,qec">
        <a href="../applications/python/readout_error_mitigation.html" class="notebook-title">Readout Error Mitigation</a>
        <div class="notebook-content">
            Quantum computers are limited by their noise, which corrupts the outcome of applications. Error mitigation is a technique used to compensate for such errors via postprocessing. Learn how to combat noise in this CUDA-Q readout error mitigation tutorial.
        </div>
        <img src="../_static/app_title_images/readout_preview.png" alt="Readout Error Mitigation Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="aiforq,noiseless,gpu">
        <a href="../applications/python/unitary_compilation_diffusion_models.html" class="notebook-title">Compiling Unitaries with Diffusion Models</a>
        <div class="notebook-content">
            Implementing quantum circuits to apply arbitrary unitary operations is a complex task. This tutorial explores an AI for quantum application where a diffusion model can be used to compile unitaries.
        </div>
        <img src="../_static/app_title_images/diffusion_preview.png" alt="Diffusion Model Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,noiseless,mqpu">
        <a href="../applications/python/vqe_advanced.html" class="notebook-title">The Variational Quantum Eigensolver</a>
        <div class="notebook-content">
            The variational quantum eigensolver is a hybrid quantum classical algorithm for predicting the ground state of a Hamiltonian. Learn how to predict molecular energies with the VQE in CUDA-Q using active spaces, how to parallelize gradient evaluation, and how to use performance optimizations like gate fusion.
        </div>
        <img src="../_static/app_title_images/vqe_preview.png" alt="VQE Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,collab,qforai,noiseless,gpu">
        <a href="../applications/python/quantum_transformer.html" class="notebook-title">Quantum Transformer Model for Generating Molecules</a>
        <div class="notebook-content">
            Learn how to implement a hybrid quantum transformer model for generating molecules. The tutorial is based off a collaboration between NVIDIA and Yale. (https://arxiv.org/pdf/2502.19214)
        </div>
        <img src="../_static/app_title_images/quantum_transformer_preview.png" alt="Transformer Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,noiseless,gpu,collab">
        <a href="../applications/python/afqmc.html" class="notebook-title">Quantum Enhanced Auxiliary Field Quantum Monte Carlo</a>
        <div class="notebook-content">
            Quantum Enhanced Auxiliary Field Quantum Monte Carlo is an advanced variational technique for simulating molecular energies. Learn how NVIDIA and BASF collaborated to implement this technique.
        </div>
        <img src="../_static/app_title_images/afmqc_preview.png" alt="AFQMC Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="optimization,noiseless,gpu">
        <a href="../applications/python/adapt_qaoa.html" class="notebook-title">ADAPT QAOA</a>
        <div class="notebook-content">
            Learn how to implement the Adaptive Derivative-Assembled Pseudo-Trotter (ADAPT) ansatz QAOA using CUDA-Q. The method iteratively builds an ansatz to more efficiently converge to the ground state of a problem Hamiltonian.
        </div>
        <img src="../_static/app_title_images/adapt_qaoa_preview.png" alt="ADAPT Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,noiseless,gpu">
        <a href="../applications/python/adapt_vqe.html" class="notebook-title">ADAPT VQE</a>
        <div class="notebook-content">
            Learn how to implement the Adaptive Derivative-Assembled Pseudo-Trotter (ADAPT) to predict molecular ground state energies. The method iteratively builds an ansatz to more efficiently converge compared to traditional VQE.
        </div>
        <img src="../_static/app_title_images/adaptvqe_preview.png" alt="ADAPT VQE  Preview" class="notebook-image">
    </div>


    <div class="notebook-entry" data-tags="qforai,noiseless,gpu">
        <a href="../applications/python/edge_detection.html" class="notebook-title">Quantum Edge Detection</a>
        <div class="notebook-content">
            Learn how to encode image data with a quantum circuit and use a quantum algorithm to identify object boundaries in an image.
        </div>
        <img src="../_static/app_title_images/edgedetection_preview.png" alt="Edge Detection Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="fundamental,noiseless,gpu,cpu">
        <a href="../applications/python/shors.html" class="notebook-title">Shor's Algorithm</a>
        <div class="notebook-content">
            Learn how to code the famous Shor's algorithm to factor a product of primes using CUDA-Q.
        </div>
        <img src="../_static/app_title_images/shors_preview.png" alt="Shors Algorithm" class="notebook-image">
    </div>    

    <div class="notebook-entry" data-tags="chemistry,noiseless">
        <a href="../applications/python/generate_fermionic_ham.html" class="notebook-title">Generating the Electronic Hamiltonian</a>
        <div class="notebook-content">
            Learn how to generate the electronic hamiltonian and convert it to qubit hamiltonian using CUDA-Q.
        </div>
        <img src="../_static/app_title_images/electronic-ham.png" alt="Electronic Hamiltonian" class="notebook-image">
    </div> 
    <div class="notebook-entry" data-tags="cpu,fundamental,noiseless,gpu">
        <a href="../applications/python/grovers.html" class="notebook-title">Grover's Algorithm</a>
        <div class="notebook-content">
            Learn how quantum computers can quadratically speed up searching through an unstructured database by amplifying the probability of finding the desired item.
        </div>
        <img src="../_static/app_title_images/grovers_preview.png" alt="DJ Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="dynamics,noiseless,gpu">
        <a href="../applications/python/quantum_pagerank.html" class="notebook-title">Quantum Pagerank</a>
        <div class="notebook-content">
            Quantum stochastic walk using dynamic simulation demonstrated by the Quantum PageRank algorithm for social network analysis.
        </div>
        <img src="../_static/app_title_images/quantum_pagerank_preview.png" alt="Quantum Pagerank Preview" class="notebook-image">
    </div> 

    <div class="notebook-entry" data-tags="chemistry,noisless">
        <a href="../applications/python/uccsd_wf_ansatz.html" class="notebook-title">UCCSD Wavefunction Ansatz</a>
        <div class="notebook-content">
            Learn how to implement the UCCSD wavefunction ansatz using CUDA-Q.
        </div>
        <img src="../_static/app_title_images/uccsd.png" alt="UCCSD Wavefunction Ansatz" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="cpu,fundamental">
        <a href="../applications/python/mps_encoding.html" class="notebook-title">MPS Sequential Encoding</a>
        <div class="notebook-content">
            Learn how to approximately prepare quantum states via MPS using CUDA-Q.
        </div>
        <img src="../_static/app_title_images/mps_encoding.png" alt="MPS Encoding" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="chemistry,noisless">
        <a href="../applications/python/qm_mm_pe.html" class="notebook-title">QM/MM simulation: VQE within a Polarizable Embedded Framework.</a>
        <div class="notebook-content">
            Learn how to implement QM/MM with PE framework using CUDA-Q.
        </div>
        <img src="../_static/app_title_images/qmmm-pe.png" alt="QM/MM partitioning in the PE model." class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="gpu, dynamics, noiseless">
        <a href="../applications/python/entanglement_acc_hamiltonian_simulation" class="notebook-title">Entanglement Accelerates Quantum Simulation</a>
        <div class="notebook-content">
            Learn how entanglement growth can *reduce* the Trotter error of the first-order product formula (PF1), recovering the result from the [paper](https://www.nature.com/articles/s41567-025-02945-2) using NVIDIA CUDA-Q. 
        </div>
        <img src="../_static/app_title_images/entanglement_acc_hamiltonian_simulation_preview.png" alt="PF1 error decreases as entanglement spreads" class="notebook-image">
    </div>

    <script>
    document.addEventListener("DOMContentLoaded", function() {
        document.querySelectorAll('.notebook-entry').forEach(entry => {
            const tags = entry.dataset.tags.split(',');
            const contentDiv = entry.querySelector('.notebook-content');
            const tagsContainer = document.createElement('div');
            tagsContainer.className = 'data-tags'

            tags.forEach(tag => {
                const tagElem = document.createElement('span');
                tagElem.textContent = '#' + tag.trim() + ' ';
                tagsContainer.appendChild(tagElem);
            });
            contentDiv.appendChild(tagsContainer);
        });
    });
    </script>
