CUDA Quantum Open Beta Installation
*******************************************

Docker Image
--------------------

Install the Public Beta Docker Image
++++++++++++++++++++++++++++++++++++
This public beta release of CUDA Quantum is being deployed via 
a provided Docker image. The name of the image is :code:`nvcr.io/nvidia/cuda-quantum:0.3.0`,
and it has been built for :code:`x86_64,amd64` platforms. 

.. code-block:: console

    docker pull nvcr.io/nvidia/cuda-quantum:0.3.0

Use CUDA Quantum in a Terminal
+++++++++++++++++++++++++++++++++++++

The container can be run using the following command

.. code-block:: console

    docker run -it --name cuda-quantum nvcr.io/nvidia/cuda-quantum:0.3.0

This will give you terminal access to the created container, but you are free to attach 
an existing VSCode IDE to it.

(what you'll see) 

.. code-block:: console 

    user@host:~$ docker run -it --name cuda-quantum nvcr.io/nvidia/cuda-quantum:0.3.0
    To run a command as administrator (user "root"), use "sudo <command>".
    See "man sudo_root" for details.

    =========================
    NVIDIA CUDA Quantum
    =========================

    CUDA Quantum Version 0.3.0

    Copyright (c) 2023 NVIDIA Corporation & Affiliates
    All rights reserved.

    cudaq@container:~$ ls
    README.md  examples
    cudaq@container:~$ ls examples/
    cpp  python

.. note:: 

    If you have NVIDIA GPUs available and NVIDIA Docker correctly configured, 
    you can add :code:`--gpus all` to the :code:`docker run` command to expose all available GPUs 
    to the container, or :code:`--gpus '"device=1"'` to select a specific GPU device.
    Unless you specify this flag, you will not be able to compile to the :code:`--qpu cuquantum`
    target. 

.. note:: 

    If you would like a temporary container, pass :code:`--rm`. This will delete your 
    container upon exit. 

.. note:: 

    If you leave the container and did not specify :code:`--rm`, you
    can always get back in with :code:`docker exec -it cuda-quantum bash`.

You can stop and exit the container by typing the command :code:`exit`. If you did not specify
:code:`--rm`, the container and any changes you made in it still exist. You can get back to it using
the command :code:`docker start -i cuda-quantum`. 

.. note::
    You can delete an existing container and any changes you made using :code:`docker rm -v cuda-quantum`. 

Use CUDA Quantum in VS Code
+++++++++++++++++++++++++++++++++++++

If you have `VS Code`_ installed, you can use it to work inside your container.
To do so, install the `Dev Containers extension`_:

.. image:: _static/devContainersExtension.png 

Follow the steps :ref:`above<Use CUDA Quantum in a Terminal>` to start the container. 
Open VS Code and navigate to the Remote Explorer. You should see the running cuda-quantum dev container listed there.

.. image:: _static/attachToDevContainer.png 

Click on :code:`Attach to Container`. A new VS Code instance will open in that container. To open a terminal, 
go to the Terminal menu and select :code:`New Terminal`. 

.. image:: _static/openTerminal.png 

You can now get to work compiling the example 
codes with the :code:`nvq++` compiler, which is installed in your :code:`PATH`. 

.. image:: _static/getToWork.png 

.. _VS Code: https://code.visualstudio.com/download
.. _Dev Containers extension: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
.. _command palette: https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette

Build CUDA Quantum from Source
------------------------------

Here we will assume a Ubuntu 22.04 system. Adjust the package manager calls
for your distribution. Make sure that recent versions `cmake` and `ninja` installed.
The build also requires a recent version of `clang/clang++` or `gcc/g++`
(must have C++20 support).

Get the basic compilers you'll need via apt-get
+++++++++++++++++++++++++++++++++++++++++++++++
.. code:: bash
  
    apt-get update && apt-get install -y --no-install-recommends gcc g++ 

On Ubuntu 22.04 this will get you GCC 11. 

Get cuQuantum (optional)
++++++++++++++++++++++++

.. code:: bash 
    
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update && apt-get -y install cuquantum cuquantum-dev 

Get LLVM / Clang / MLIR
++++++++++++++++++++++++

You will need the same version of LLVM as our submodule in `tpls/llvm`.

.. code:: bash 

    mkdir llvm-project && cd llvm-project
    git init 
    git remote add origin https://github.com/llvm/llvm-project 
    # note this will change as the project evolves, 
    # Must be == to the hash we use for the tpls/llvm submodule.
    git fetch origin --depth=1 c0b45fef155fbe3f17f9a6f99074682c69545488
    git reset --hard FETCH_HEAD
    mkdir build && cd build
    cmake .. -G Ninja  
                -DLLVM_TARGETS_TO_BUILD="host" \
                -DCMAKE_INSTALL_PREFIX=/opt/llvm/
                -DLLVM_ENABLE_PROJECTS="clang;mlir" 
                -DCMAKE_BUILD_TYPE=Release 
                -DLLVM_ENABLE_ASSERTIONS=ON 
                -DLLVM_INSTALL_UTILS=TRUE 
    ninja install
    # This is needed for FileCheck tests.
    cp bin/llvm-lit /opt/llvm/bin/

Build CUDA Quantum
++++++++++++++++++
You must use the same compiler that you compiled LLVM with to compile CUDA Quantum.

.. code:: bash
    
    git clone https://github.com/NVIDIA/cuda-quantum && cd cuda-quantum
    mkdir build && cd build
    cmake .. -G -DCMAKE_INSTALL_PREFIX=$HOME/.cudaq 
                -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm 
                -DCUDAQ_ENABLE_PYTHON=TRUE
                \# (optional, if cuquantum is installed)
                -DCUSTATEVEC=/opt/nvidia/cuquantum
    ninja install
    ctest 

Next Steps
----------
With the CUDA Quantum Docker image installed and a container up and running, check out the
Using CUDA Quantum page_. To run the examples codes in the container, checkout the Compiling
and Executing section here_. 

Once in the VSCode IDE or in the terminal for the container in headless mode, you'll 
notice there is an :code:`examples/` folder. These examples are provided to 
get you started with CUDA Quantum and understanding the programming and execution model. 
Start of by trying to compile a simple one, like :code:`examples/cpp/basics/static_kernel.cpp`

.. code-block:: console 

    nvq++ examples/cpp/basics/static_kernel.cpp 
    ./a.out

If you have GPU support (e.g. you successfully provided :code:`--gpus` to your docker 
run command), try out the 30 qubit version of this example.

.. code-block:: console 

    nvq++ examples/cpp/basics/cuquantum_backends.cpp --qpu cuquantum 
    ./a.out 

.. _page: using/cudaq.html
.. _here: using/cudaq/compiling.html
