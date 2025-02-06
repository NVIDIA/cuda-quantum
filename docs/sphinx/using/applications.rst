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
            <h3>Filter by Backend:</h3>
            <button class="tag-button active" data-group="backend" data-tag="all">All</button>
            <button class="tag-button" data-group="backend" data-tag="mqpu">MQPU</button>
            <button class="tag-button" data-group="backend" data-tag="mgpu">MGPU</button>
        </div>

        <div class="tag-filters" style="margin-top: 20px;">
            <h3>Filter by Library:</h3>
            <button class="tag-button active" data-group="library" data-tag="all">All</button>
            <button class="tag-button" data-group="library" data-tag="solvers">Solvers</button>
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
    <div class="notebook-entry" data-tags="fundamental,mqpu" style="margin-bottom: 50px;">

.. rst-class:: notebook-title

**The Anderson Impurity Model With Logical Qubits**

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







    </div>
    </div>
