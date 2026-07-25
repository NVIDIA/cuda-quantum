CUDA-Q Applications
====================

This page contains a number of different applications implemented using CUDA-Q. All notebooks can be found `here. <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/applications/python>`_
To run these applications without a local installation, run the `CUDA-Q application Hub launchable <https://www.nvidia.com/cudaq-apps/>`_.


.. when adding applications
.. 1. Add notebook to the hidden TOC list directly below
.. 2. Add an html block along with any tags
.. 3. Add a preview image in the _static folder

.. |:spellcheck-disable:| replace:: \

.. toctree::
   :maxdepth: 1
   :hidden:

   /applications/python/krylov
   /applications/python/qsci
   /applications/python/hadamard_test
   /applications/python/hamiltonian_simulation
   /applications/python/quantum_volume
   /applications/python/readout_error_mitigation
   /applications/python/afqmc
   /applications/python/shors
   /applications/python/generate_fermionic_ham
   /applications/python/uccsd_wf_ansatz
   /applications/python/mps_encoding
   /applications/python/skqd
   /applications/python/entanglement_acc_hamiltonian_simulation
   /applications/python/ptsbe

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

    <div class="notebook-entry" data-tags="fundamental,noiseless,gpu,mqpu">
        <a href="../applications/python/hadamard_test.html" class="notebook-title">The Hadamard Test</a>
        <div class="notebook-content">
            Learn about the Hadamard test and how it can be used to estimate expectation values. This notebook also explores how the Hadamard test can be used for Krylov subspace method and accelerated with the <code>mqpu</code> backend to evaluate execution on multiple simulated QPUs in parallel.
        </div>
        <img src="../_static/app_title_images/hadamard_preview.png" alt="Hadamard Preview" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="fundamental,noiseless,gpu">
        <a href="../applications/python/hamiltonian_simulation.html" class="notebook-title">Trotterized Hamiltonian Simulation</a>
        <div class="notebook-content">
            Trotterization is an approximation to enable simulation of a Hamiltonian. Learn how this technique works and simulate the dynamics of the Heisenberg model.
        </div>
        <img src="../_static/app_title_images/trotter_preview.png" alt="Trotter Preview" class="notebook-image">
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

    <div class="notebook-entry" data-tags="chemistry,noiseless,gpu,collab">
        <a href="../applications/python/afqmc.html" class="notebook-title">Quantum Enhanced Auxiliary Field Quantum Monte Carlo</a>
        <div class="notebook-content">
            Quantum Enhanced Auxiliary Field Quantum Monte Carlo is an advanced variational technique for simulating molecular energies. Learn how NVIDIA and BASF collaborated to implement this technique.
        </div>
        <img src="../_static/app_title_images/afmqc_preview.png" alt="AFQMC Preview" class="notebook-image">
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

    <div class="notebook-entry" data-tags="chemistry,noiseless">
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

    <div class="notebook-entry" data-tags="gpu, dynamics, noiseless">
        <a href="../applications/python/entanglement_acc_hamiltonian_simulation" class="notebook-title">Entanglement Accelerates Quantum Simulation</a>
        <div class="notebook-content">
            Learn how entanglement growth can *reduce* the Trotter error of the first-order product formula (PF1), recovering the result from the [paper](https://www.nature.com/articles/s41567-025-02945-2) using NVIDIA CUDA-Q.
        </div>
        <img src="../_static/app_title_images/entanglement_acc_hamiltonian_simulation_preview.png" alt="PF1 error decreases as entanglement spreads" class="notebook-image">
    </div>

    <div class="notebook-entry" data-tags="fundamental,noisy,gpu">
        <a href="../applications/python/ptsbe.html" class="notebook-title">Noisy Circuit Simulation with PTSBE</a>
        <div class="notebook-content">
            Pre-Trajectory Sampling with Batch Execution (PTSBE) is an efficient method for sampling from noisy quantum circuits. Rather than simulating the full density matrix, PTSBE pre-samples unique noise trajectories and batches many shots across them, yielding orders-of-magnitude speedups for large shot counts. Based on the SC25 paper by Patti et al. (https://arxiv.org/abs/2504.16297).
        </div>
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
