Applications
===========

.. raw:: html

    <div class="filter-groups" style="margin-bottom: 60px;">
        <div class="tag-filters">
            <h3>Filter by Domain:</h3>
            <button class="tag-button active" data-group="domain" data-tag="all">All</button>
            <button class="tag-button" data-group="domain" data-tag="optimization">Optimization</button>
            <button class="tag-button" data-group="domain" data-tag="chemistry">Chemistry</button>
            <button class="tag-button" data-group="domain" data-tag="fundamental">Fundamental Algorithms</button>
        </div>
        
        <div class="tag-filters" style="margin-top: 20px;">
            <h3>Specialized Backends:</h3>
            <button class="tag-button active" data-group="backend" data-tag="all">All</button>
            <button class="tag-button" data-group="backend" data-tag="mqpu">MQPU</button>
            <button class="tag-button" data-group="backend" data-tag="mgpu">MGPU</button>
            <button class="tag-button" data-group="backend" data-tag="noise">Noisy</button>
            <button class="tag-button" data-group="backend" data-tag="qpu">QPU</button>
        </div>

        <div class="tag-filters" style="margin-top: 20px;">
            <h3>CUDA-X Library:</h3>
            <button class="tag-button" data-group="library" data-tag="solvers">Solvers</button>
        </div>

        <div class="tag-filters" style="margin-top: 20px;">
            <h3>Filter by Occasion:</h3>
            <button class="tag-button active" data-group="occasion" data-tag="collab">Collaboration</button>
            <button class="tag-button" data-group="occasion" data-tag="blog">Blog</button>
        </div>    
      </div>

.. raw:: html


    <div class="notebook-entry" data-tags="optimization" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**QAOA for Max Cut Problem**

.. figure:: /applications/app_title_images/qaoa_preview.png
    :align: right
    :width: 300px
    :alt: QAOA Preview
    :class: notebook-image

This notebook explains the theory behind the Quantum Approximate Optimization Algorithm (QAOA) and explains how it can be used to solve the Max Cut problem. 

.. toctree::
    :maxdepth: 1

    /applications/python/qaoa

.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="chemistry,optimization" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Digitized Counterdiabatic QAOA**

.. figure:: /applications/app_title_images/dcqaoa_preview.png
    :align: right
    :width: 300px
    :alt: DC-QAOA Preview
    :class: notebook-image

This notebook explores application of QAOA to predict molecules that might be good candidates for drugs based on their interactions with proteins.

.. toctree::
    :maxdepth: 1

    /applications/python/digitized_counterdiabatic_qaoa



 

.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="chemistry,mqpu" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Krylov Subspace Methods**

.. figure:: /applications/app_title_images/krylov_preview.png
    :align: right
    :width: 300px
    :alt: Krylov Preview
    :class: notebook-image

Learn how the Krylov method uses the Hadamard test to predict the ground state energy of molecules.  Also learn how to implement the same approach with the :code:`mqpu` backend and simulate execution on multiple QPUs in parallel.

.. toctree::
    :maxdepth: 1

    /applications/python/krylov

.. raw:: html



.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="fundamental" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Krylov Subspace Methods**

.. figure:: /applications/app_title_images/bv_preview.png
    :align: right
    :width: 300px
    :alt: Krylov Preview
    :class: notebook-image

Learn a famous quantum algorithm that provides intuition for why exponential speedups can be acheived with quantum computers.

.. toctree::
    :maxdepth: 1

      /applications/python/bernstein_vazirani

.. raw:: html





.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="fundamental" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**The Bernstein-Vazirani Algorithm**

.. figure:: /applications/app_title_images/bv_preview.png
    :align: right
    :width: 300px
    :alt: Krylov Preview
    :class: notebook-image

Learn a famous quantum algorithm that provides intuition for why exponential speedups can be acheived with quantum computers.

.. toctree::
    :maxdepth: 1

      /applications/python/bernstein_vazirani

.. raw:: html




.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="fundamental" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Cost Minimization**

.. figure:: /applications/app_title_images/cost_preview.png
    :align: right
    :width: 300px
    :alt: Krylov Preview
    :class: notebook-image

Explore a hello-world example for variational quantum algorithms.  Learn how to build a parameterized quantum circuit using a single qubit and minimize and expectation value. 

.. toctree::
    :maxdepth: 1

      /applications/python/cost_minimization

.. raw:: html




.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="fundamental" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**The Deutsch-Jozsa Algorithm**

.. figure:: /applications/app_title_images/dj_preview.png
    :align: right
    :width: 300px
    :alt: Krylov Preview
    :class: notebook-image

Learn how quantum computers can provide an exponential speedup for identifying if a Boolean function is constant or balanced.

.. toctree::
    :maxdepth: 1

      /applications/python/deutsch_jozsa

.. raw:: html



.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="qforai,mgpu" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Divisive Clustering with Coresets**

.. figure:: /applications/app_title_images/cluster_preview.png
    :align: right
    :width: 300px
    :alt: Krylov Preview
    :class: notebook-image

Explore an implementation of the work in this paper (https://arxiv.org/abs/2402.01529) which looks at ways to cluster large data sets on quantum computers using a data reduction techniqe called coresets. This notebook includes the full workflow, a QAOA implementation, and an example of using the :code:`mgpu` backend to scale the problem to greater qubit numbers.

.. toctree::
    :maxdepth: 1

      /applications/python/divisive_clustering_coresets

.. raw:: html





.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="fundamental,mqpu" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**The Hadamard Test**

.. figure:: /applications/app_title_images/hadamard_preview.png
    :align: right
    :width: 300px
    :alt: Krylov Preview
    :class: notebook-image

Learn about the Hadamard test and how it can be used to estimate expectation values.  This notebook also explores how the Hadamard test can be used for Krylov subspace method and accelerated with the :code:`mqpu` backend to evaluate execution on multiple simulated QPUs in parallel.

.. toctree::
    :maxdepth: 1

      /applications/python/hadamard_test

.. raw:: html




.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="chemistry,qec,qpu,solvers" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**The Anderson Impurity Model With Logical Qubits**

.. figure:: /applications/app_title_images/logical_aim_preview.png
    :align: right
    :width: 300px
    :alt: Krylov Preview
    :class: notebook-image

A recent collaboration between NVIDIA and Infleqtion demonstrated a logical qubit workflow built in CUDA-Q and executed on the Infleqtion's neutral atom QPU.  TO learn more, read the `paper <https://arxiv.org/abs/2412.07670>`_
  and corresponding `blog <https://blogs.nvidia.com/blog/logical-qubits-cuda-q-demo/>`_.
.. toctree::
    :maxdepth: 1

      /applications/python/logical_aim_sqale

.. raw:: html






.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="fundamental" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Trotterized Hamiltonian Simulation**

.. figure:: /applications/app_title_images/trotter_preview.png
    :align: right
    :width: 300px
    :alt: Krylov Preview
    :class: notebook-image

Trotterization is an approximation to enable simulation of a Hamiltonian.  Learn how this technique works and simulate the dynamics of the Heisenberg model.

.. toctree::
    :maxdepth: 1

      /applications/python/hamiltonian_simulation

.. raw:: html




.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="fundamental" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**The Quantum Fourier Transform**

.. figure:: /applications/app_title_images/qft_preview.png
    :align: right
    :width: 300px
    :alt: QFT Preview
    :class: notebook-image

The Quantum Fourier transform (QFT) is a fundamental quantum algoithm that is also and important subroutine of quantum phase estimation, Shors's, and other quantum algorithms.  Learn the basics of the QFT and how to implement it in CUDA-Q
.. toctree::
    :maxdepth: 1

      /applications/python/quantum_fourier_transform

.. raw:: html




.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="fundamental" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**The Quantum Fourier Transform**

.. figure:: /applications/app_title_images/qft_preview.png
    :align: right
    :width: 300px
    :alt: QFT Preview
    :class: notebook-image

The Quantum Fourier transform (QFT) is a fundamental quantum algoithm that is also and important subroutine of quantum phase estimation, Shors's, and other quantum algorithms.  Learn the basics of the QFT and how to implement it in CUDA-Q
.. toctree::
    :maxdepth: 1

      /applications/python/quantum_fourier_transform

.. raw:: html






.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="fundamental" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Quantum Teleportation**

.. figure:: /applications/app_title_images/teleport_preview.png
    :align: right
    :width: 300px
    :alt: Quantum Teleportation Preview
    :class: notebook-image

Quantum teleportation of one of the strange phenomena that makes quantum computing so interresting.  Learn how teleportation works and how it can be implemented in CUDA-Q
.. toctree::
    :maxdepth: 1

      /applications/python/quantum_teleportation

.. raw:: html






.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="aiforq,fundamental,noise" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Quantum Volume**

.. figure:: /applications/app_title_images/qv_preview.png
    :align: right
    :width: 300px
    :alt: Quantum Volume Preview
    :class: notebook-image


Benahmarking the performance of quantum computers, especially between different qubit modalities, is challenging.  One method is to experimentally perform the quantum volume test.  Learn how thsi test is performend and how it can be implemented in CUDA-Q.
.. toctree::
    :maxdepth: 1

      /applications/python/quantum_volume

.. raw:: html





.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="aiforq,noise" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Readout Error Mitigation**

.. figure:: /applications/app_title_images/readout_preview.png
    :align: right
    :width: 300px
    :alt: Readout Error Mitigation Preview
    :class: notebook-image


Quantum computers are limited by their noise, which corrupts the outcome of applications.  Error mitigation is a techniqe used to compensate for such errors via postprocessing.  Learn how to combat noise in this CUDA-Q readout error mitigation tutorial.
.. toctree::
    :maxdepth: 1

      /applications/python/readout_error_mitigation

.. raw:: html



   
.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="aiforq" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Compiling Unitaries with Diffusion Models**

.. figure:: /applications/app_title_images/diffusion_preview.png
    :align: right
    :width: 300px
    :alt: Readout Diffusion Preview
    :class: notebook-image

Implementing quantum circuits to apply arbitrary unitary operations is a complex task.  This tutorial explores an AI for quantum application where a diffusion modelc an be used to compile unitaries.
.. toctree::
    :maxdepth: 1

      /applications/python/unitary_compilation_diffusion_models

.. raw:: html


.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="chemistry, mqpu" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**The Variational Quantum Eigensolver**

.. figure:: /applications/app_title_images/vqe_preview.png
    :align: right
    :width: 300px
    :alt: VQE Preview
    :class: notebook-image

The variational quantum eigensolver is a hybrid quantum classical algorithm for predicting the ground state of a Hamiltonian.  Learn how to predict molecular energies with the VQE in CUDA-Q using active spaces, how to parallelize gradient evaluation, and how to use performance optimizations like gate fusion.
.. toctree::
    :maxdepth: 1

      /applications/python/vqe_advanced

.. raw:: html




.. raw:: html

    </div>
    <div class="notebook-entry" data-tags="chemistry,collab" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**Quantum Enhanced Auxiliary Field Quantum Monte Carlo**

.. figure:: /applications/app_title_images/afmqc_preview.png
    :align: right
    :width: 300px
    :alt: AFMQC Preview
    :class: notebook-image

Quantum Enhanced Auxiliary Field Quantum Monte Carlo is and advanced variational technique for simulating molecular energies.  Learn how NVIDIA and BASF collaborated to implement this technique.
.. toctree::
    :maxdepth: 1

      /applications/python/afqmc

.. raw:: html







    </div>
    </div>
