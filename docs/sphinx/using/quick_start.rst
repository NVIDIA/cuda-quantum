Quick Start
===================

**NVIDIA CUDA-Q**

CUDA-Q streamlines hybrid application development and promotes productivity and scalability
in quantum computing. It offers a unified programming model designed for a hybrid
setting |---| that is, CPUs, GPUs, and QPUs working together. CUDA-Q contains support for 
programming in Python and in C++. Learn more about the `key benefits of CUDA-Q <https://developer.nvidia.com/cuda-q>`_.

This Quick Start page guides you through installing CUDA-Q and running your first program.
If you have already installed and configured CUDA-Q, or if you are using our 
`Docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum>`_, you can move directly to our
:doc:`Basics Section <basics/basics>`. More information about working with containers and Docker alternatives can be 
found in our complete :doc:`Installation Guide <install/install>`.

.. note::

    CUDA-Q is currently supported on Linux only. On Windows, you can use `Windows Subsystem for Linux (WSL) <https://learn.microsoft.com/en-us/windows/wsl/>`_ to install CUDA-Q.

Install CUDA-Q
----------------------------

.. tab:: Python

   To develop CUDA-Q applications using Python, 
   please follow the instructions for `installing CUDA-Q <https://pypi.org/project/cudaq/>`_ from PyPI. 
   If you have an NVIDIA GPU, make sure to also follow the instructions for enabling GPU-acceleration.

   .. include:: ../../../python/README.md
      :parser: myst_parser.sphinx_
      :start-after: (Begin complete install)
      :end-before: (End complete install)

.. note::

    If you are using CUDA 12.8 on Blackwell, you may need to install additional
    dependencies to use the python wheels.
    See :ref:`Blackwell CUDA Dependencies <blackwell-cuda-dependencies>` for more information.

Once you completed the installation, please follow the instructions
   :ref:`below <validate-installation>` to run your first CUDA-Q program!

.. tab:: C++

   To develop CUDA-Q applications using C++, please make sure you have a C++ toolchain installed
   that supports C++20, for example `g++` version 11 or newer.
   Download the `install_cuda_quantum` file for your processor architecture and CUDA version (`_cu12` suffix for CUDA 12 and `_cu13` suffix for CUDA 13) 
   from the assets of the respective `GitHub release <https://github.com/NVIDIA/cuda-quantum/releases>`__; 
   that is, the file with the `aarch64` extension for ARM processors, and the one with `x86_64` for, e.g., Intel and AMD processors.

   To install CUDA-Q, execute the commands

   .. code-block:: bash

      sudo -E bash install_cuda_quantum*.$(uname -m) --accept 
      . /etc/profile

   If you have an NVIDIA GPU, please also install the `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`__ to enable GPU-acceleration within CUDA-Q.

   Please see the complete :ref:`installation guide <install-prebuilt-binaries>` for more details, including

   - a list of :ref:`supported operating systems <dependencies-and-compatibility>`, 
   - instructions on how to :ref:`enable MPI parallelization <distributed-computing-with-mpi>` within CUDA-Q, and
   - information about :ref:`updating CUDA-Q <updating-cuda-quantum>`.

   Once you completed the installation, please follow the instructions
   :ref:`below <validate-installation>` to run your first CUDA-Q program!   

.. |---|   unicode:: U+2014 .. EM DASH
   :trim:

.. _validate-installation:

Validate your Installation
----------------------------

Let's run a simple program to validate your installation.
The quantum kernel in the following program creates and measures the state 
:math:`(|00\rangle + |11\rangle) / \sqrt{2}`. That means each kernel execution should 
either yield `00` or `11`. The program samples, meaning it executes, the kernel 1000 times
and prints how many times each output was measured. On average, the values `00` and `11`
should be observed around 500 times each.

.. tab:: Python

   Create a file titled `program.py`, containing the following code:

   .. literalinclude:: /snippets/python/quick_start.py
      :language: python
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

   Run this program as you do any other Python program, for example:

   .. code-block:: console

      python3 program.py

.. tab:: C++

   Create a file titled `program.cpp`, containing the following code:

   .. literalinclude:: /snippets/cpp/quick_start.cpp
      :language: cpp
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

   Compile the program using the `nvq++` compiler and run the built application with the following command:

   .. code-block:: console

      nvq++ program.cpp -o program.x && ./program.x

If you have an NVIDIA GPU the program uses GPU acceleration by default.
To confirm that this works as expected and to see the effects of GPU acceleration, you can 
increase the numbers of qubits the program uses to 28 and
compare the time to execute the program on the 
`nvidia` target (:ref:`GPU-accelerated statevector simulator <cuQuantum single-GPU>`) to the time when setting the target to `qpp-cpu` (:ref:`OpenMP parallelized CPU-only statevector simulator <OpenMP CPU-only>`):

.. tab:: Python

   .. code-block:: console

      python3 program.py 28 --target nvidia

.. tab:: C++

   .. code-block:: console

      nvq++ program.cpp -o program.x --target nvidia && ./program.x 28

When you change the target to `qpp-cpu`, the program simply seems to hang; that is because it takes a long time for the CPU-only backend to simulate 28+ qubits! Cancel the execution with `Ctrl+C`.

For more information about enabling GPU-acceleration, please see
our complete :ref:`Installation Guide <additional-cuda-tools>`.
For further information on available targets, see :doc:`Backends <backends/backends>`.

You are now all set to start developing quantum applications using CUDA-Q!
Please proceed to :doc:`Basics <basics/basics>` for an introduction
to the fundamental features of CUDA-Q.


CUDA-Q Academic
------------------
`CUDA-Q Academic <https://github.com/NVIDIA/cuda-q-academic>`_  is a collection of courses designed for academic use based on the CUDA-Q platform.  Courses range in complexity from the basics of quantum computing up to implementations of algorithms from the research literature.  Users will learn quantum computing, CUDA-Q, and how to run hybrid algorithms that make the most of QPUs and AI supercomputers through coding exercises and interactive lessons.  

.. figure:: cudaq_academic/widget-recording.gif
   :width: 500px
   :align: center

   Example of the interactive material within CUDA-Q Academic.

The courses can be run using a local installation of CUDA-Q or via the cloud using `qBraid`, `CoCalc`, or `Google Colab`. 

Users just learning quantum computing are encouraged to begin with the `Quick Start to Quantum <https://github.com/NVIDIA/cuda-q-academic/blob/main/quick-start-to-quantum/README.md>`_ course which provides an introduction to the basics of quantum computing and helps you implement your first algorithm. 

Advanced users can choose from a number of learning paths spanning topics ranging from QEC to hybrid algorithms `here  <https://github.com/NVIDIA/cuda-q-academic>`_ .



