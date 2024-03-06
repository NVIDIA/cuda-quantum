Python
-------

If you want to develop CUDA Quantum applications using Python, install the
latest stable release of the CUDA Quantum Python API:  

.. code-block:: bash

    pip install cuda-quantum

For further information, see the `CUDA Quantum project <https://pypi.org/project/cuda-quantum/>`_ on PyPI.

We also offer a fully featured CUDA Quantum installation, including all C++ and Python tools, via our
`Docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda-quantum>`_. For further
installation methods and resources, see our :ref:`Installation Guide <install-cuda-quantum>`.

You should now be able to import CUDA Quantum and start building quantum programs in Python!

To test your installation, create a file titled `first_program.py`, containing the following code:

.. literalinclude:: /snippets/python/quick_start.py
    :language: python
    :start-after: [Begin Documentation]

You may now execute this file as you do any other Python program. For example, from the command line:

.. code-block:: bash

    python3 first_program.py

The Hadamard gate places the qubit in a superposition state, giving a roughly 50/50 mixture
of measurements in the `|0>` and `|1>` states.

If you have a local GPU, the following dependencies must be installed before using any GPU accelerated simulation target.

.. code-block:: bash
    
    conda create -y -n cuda-quantum python==3.10 pip
    conda install -y -n cuda-quantum -c "nvidia/label/cuda-11.8.0" cuda
    conda install -y -n cuda-quantum -c conda-forge mpi4py openmpi cxx-compiler cuquantum
    conda env config vars set -n cuda-quantum LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_PREFIX/envs/cuda-quantum/lib"
    conda env config vars set -n cuda-quantum MPI_PATH=$CONDA_PREFIX/envs/cuda-quantum
    conda run -n cuda-quantum pip install cuda-quantum
    conda activate cuda-quantum
    source $CONDA_PREFIX/lib/python3.10/site-packages/distributed_interfaces/activate_custom_mpi.sh

When a GPU is detected, the simulation target defaults to `nvidia`. Otherwise, the target will default to the
`qpp-cpu` simulator. To confirm the detection of your GPU, you may once again run the example

.. code-block:: bash

    python3 first_program.py

and confirm that `Simulation Target = nvidia` is printed to the console.

For further information on available targets, see :ref:`Backends <backends-landing-page>`.

Now that you have successfully run your first program, you are ready to move on to our :ref:`Basics Section <cudaq-basics-landing-page>`.