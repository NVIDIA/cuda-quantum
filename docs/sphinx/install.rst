Getting Started
*******************************************

This guide walks through how to :ref:`install CUDA Quantum <install-cuda-quantum>` on your system.

The section on :ref:`connecting to a remote host <connect_to_remote>` contains some
guidance for application development on a remote host where CUDA Quantum is installed.

.. _install-cuda-quantum:

Local Installation
------------------------------------

A fully featured CUDA Quantum installation including all C++ and Python tools is available as a 
`Docker image`_. A `Singularity`_ container can easily be created based on these images. Additionally, we distribute pre-built `Python wheels`_ via PyPI.
If you would like to build CUDA Quantum from source instead, please follow the instructions on the `CUDA Quantum GitHub repository`_.

.. _CUDA Quantum GitHub repository: https://github.com/NVIDIA/cuda-quantum/blob/main/Building.md

If you are unsure which option suits you best, we recommend using our `Docker image`_ to develop your applications in a controlled environment that does not depend on, or interfere with, other software
that is installed on your system.

Docker image
++++++++++++++++++++++++++++++++++++

To download and use our Docker images, you will need to install and launch the Docker engine. 
If you do not already have Docker installed on your system, you can get it by downloading and installing `Docker Desktop <https://docs.docker.com/get-docker/>`_. 
If you do not have the necessary administrator permissions to install software on your machine, 
take a look at the section below on how to use `Singularity`_ instead,
or consider using `Podman <https://github.com/containers/podman>`__.

Docker images for all CUDA Quantum releases are available on the `NGC Container Registry`_.
In addition to publishing stable releases, we also publish docker images whenever we update the main branch, or release branches, on our `GitHub repository <https://github.com/NVIDIA/cuda-quantum>`_.
These images are published in a separate location `nvidia/nightly` on NGC, as well as on GitHub.
To download the latest version on the main branch of our GitHub repository, for example, use the command

.. code-block:: console

    docker pull nvcr.io/nvidia/nightly/cuda-quantum:latest

.. note:: 
  
  Downloading images from `nvidia/nightly` images from NGC currently requires login (see instructions below). Alternatively, you can download the images from GitHub by pulling `ghcr.io/nvidia/cuda-quantum:latest`.

To login to NGC, please follow these steps if you have not done so already:

- `Create an account <https://ngc.nvidia.com/signin>`__
- `Sign in <https://ngc.nvidia.com/signin>`__ to access your account and go to `Setup <https://ngc.nvidia.com/setup>`__.
- Click on `Get API Key` and generate a new key (this will invalidate any existing keys).
- Follow the instructions that appear to use that key to log in to the NGC registry using Docker.

If you run `docker login nvcr.io`, you should see a message "Login Succeeded".

.. _NGC Container Registry: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda-quantum


Once you have downloaded an image, the container can be run using the command

.. code-block:: console

    docker run -it --name cuda-quantum <image_name>

replacing :code:`<image_name>` with the name and tag of the image you downloaded.
This will give you terminal access to the created container. To enable support 
for GPU-accelerated backends, you will need to pass the :code:`--gpus` flag when launching
the container, for example:

.. code-block:: console

    docker run -it --gpus all --name cuda-quantum nvcr.io/nvidia/nightly/cuda-quantum:latest

.. note:: 

  This command will fail if you do not have a suitable NVIDIA GPU available, or if your driver 
  version is insufficient. To improve compatibility with older drivers, you may need to install the 
  `NVIDIA container toolkit`_.

.. _NVIDIA container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

You can stop and exit the container by typing the command :code:`exit`. If you did not specify
:code:`--rm` flag when launching the container, the container still exists after exiting, as well as any 
changes you made in it. You can get back to it using
the command :code:`docker start -i cuda-quantum`. 
You can delete an existing container and any changes you made using :code:`docker rm -v cuda-quantum`.

When working with Docker images, the files inside the container are not visible outside the container environment. We recommend connecting VS Code to the running container to facilitate development.

Alternatively, it is possible, but not recommended, to launch an SSH server inside the container environment and connect to the container using SSH. To do so, make sure you have generated a suitable RSA key pair; if your `~/.ssh/` folder does not already contain the files `id_rsa.pub` and `id.rsa`,
follow the instructions for generating a new SSH key on `this page <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`__.
You can then launch the container and connect to it via SSH by executing the following commands:

.. code-block:: console

  docker run -itd --name cuda-quantum -p 2222:22 ghcr.io/nvidia/cuda-quantum:preview
  docker exec cuda-quantum bash -c "sudo apt-get install -y --no-install-recommends openssh-server"
  docker cp ~/.ssh/id_rsa.pub cuda-quantum:/home/cudaq/.ssh/authorized_keys
  docker exec -d cuda-quantum bash -c "sudo -E /usr/sbin/sshd -D"
  ssh cudaq@localhost -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null


Singularity
++++++++++++++++++++++++++++++++++++

You can use `Singularity <https://github.com/sylabs/singularity>`__ to run a CUDA Quantum container in a folder without needing administrator permissions.
If you do not already have Singularity installed, you can build a relocatable installation from source. 
To do so on Linux or WSL, make sure you have the `necessary prerequisites <https://docs.sylabs.io/guides/4.0/user-guide/quick_start.html#prerequisites>`__ installed, download a suitable version of the `go toolchain <https://docs.sylabs.io/guides/4.0/user-guide/quick_start.html#install-go>`__, and make sure the `go` binaries are on your :code:`PATH`. You can then build Singularity with the commands

.. code-block:: console

    wget https://github.com/sylabs/singularity/releases/download/v4.0.1/singularity-ce-4.0.1.tar.gz
    tar -xzf singularity-ce-4.0.1.tar.gz singularity-ce-4.0.1/ && rm singularity-ce-4.0.1.tar.gz && cd singularity-ce-4.0.1/
    ./mconfig --without-suid --prefix="$HOME/.local/singularity"
    make -C ./builddir && make -C ./builddir install && cd .. && rm -rf singularity-ce-4.0.1/
    echo 'PATH="$PATH:$HOME/.local/singularity/bin/"' >> ~/.profile && source ~/.profile

For more information about using Singularity on other systems, take a look at the `admin guide <https://docs.sylabs.io/guides/4.0/admin-guide/installation.html#installation-on-windows-or-mac>`__.

Once you have singularity installed, create a file `cuda-quantum.def` with the following content:

.. code-block:: console

    Bootstrap: docker
    From: ghcr.io/nvidia/cuda-quantum:latest

    %runscript
        mount devpts /dev/pts -t devpts
        cp -r /home/cudaq/*
        bash

You can then create a CUDA Quantum container by running the following commands:

.. code-block:: console

    singularity build --fakeroot cuda-quantum.sif cuda-quantum.def
    singularity run --writable --fakeroot cuda-quantum.sif

In addition to the files in your current folder, you should now
see a README as well as examples and tutorials.
To enable support for GPU-accelerated backends, you will need to pass the
the :code:`--nv` flag when running the container:

.. code-block:: console

    singularity run --writable --fakeroot --nv cuda-quantum.sif
    nvidia-smi

The output of the command above lists the GPUs that are visible and accessible in the container environment.

.. note:: 

  If you do not see any GPUs listed in the output of `nvidia-smi`, 
  it means the container environment is unable to access a suitable NVIDIA GPU. 
  This can happen if your driver version is insufficient, or if you are 
  working on WSL. To improve compatibility with older drivers, or to enable GPU support
  on WSL, please install the `NVIDIA container toolkit`_, and update the singularity configuration 
  to set `use nvidia-container-cli` to `yes` and configure the correct `nvidia-container-cli path`. 
  The two commands below use `sed` to do that:

  .. code-block:: console

    sed -i 's/use nvidia-container-cli = no/use nvidia-container-cli = yes/' "$HOME/.local/singularity/etc/singularity/singularity.conf"
    sed -i 's/# nvidia-container-cli path =/nvidia-container-cli path = \/usr\/bin\/nvidia-container-cli/' "$HOME/.local/singularity/etc/singularity/singularity.conf"

You can exit the container environment by typing the command :code:`exit`.
Any changes you made will still be visible after you exit the container, and you can re-enable the 
container environment at any time using the `run` command above.


.. _install-python-wheels:

Python wheels
++++++++++++++++++++++++++++++++++++

CUDA Quantum Python wheels are available on `PyPI.org <https://pypi.org/project/cuda-quantum>`__. Installation instructions can be found in the `project description <https://pypi.org/project/cuda-quantum/#description>`__.
For more information about available versions and documentation,
see :doc:`versions`.

At this time, wheels are distributed for Linux operating systems only. 

There are currently no source distributions available on PyPI, but you can download the source code for the latest version of the CUDA Quantum Python wheels from our `GitHub repository <https://github.com/NVIDIA/cuda-quantum>`__. The source code for previous versions can be downloaded from the respective `GitHub Release <https://github.com/NVIDIA/cuda-quantum/releases>`__.

To build the CUDA Quantum Python API from source using pip, run the following commands:

.. code-block:: console

    git clone https://github.com/NVIDIA/cuda-quantum.git
    cd cuda-quantum && ./scripts/install_prerequisites.sh
    pip install .

For more information about building the entire C++ and Python API from source, we refer to the `CUDA Quantum GitHub repository`_.


.. _connect_to_remote:

Connecting to a Remote Host
------------------------------------

VS Code using SSH
++++++++++++++++++++++++++++++++++++

You can connect to a remote host that has CUDA Quantum installed via SSH, 
and use a local installation of `VS Code <https://code.visualstudio.com/>`_ for development.

.. note:: 

  For the best user experience, we recommend to launch a CUDA Quantum container on the remote host, 
  and then connect `VS Code using tunnels`_, rather than connecting VS Code directly via SSH.
  If that is not possible, this section describes a good alternative using SSH.

To do so, make sure you have `Remote - SSH <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh>`__ extension installed.
Open the Command Pallette with `Ctrl+Shift+P`, select "Remote-SSH: Add new
SSH Host", and enter the SSH command to connect to your account on the remote host.
You can then connect to the host by opening the Command Pallette, selecting
"Remote SSH: Connect Current Window to Host", and choosing the newly created host.

When prompted, choose Linux as the operating system, and enter your
password. After the window reloaded, select "File: Open Folder" in the 
Command Pallette to open the desired folder. Our GitHub repository contains
a folder with VS Code configurations including a list of recommended extensions for
working with CUDA Quantum; you can copy `these configurations <https://github.com/NVIDIA/cuda-quantum/tree/main/docker/release/config/.vscode>`__ into the a folder named `.vscode` in your workspace to use them.

... connect to a running docker container on the remote host

... launch a singularity container on the remote host

Add to settings.json:

    "remote.SSH.enableRemoteCommand": true,
    "remote.SSH.useLocalServer": true,
    // needed only due to https://github.com/microsoft/vscode-remote-release/issues/6086
    "remote.SSH.remoteServerListenOnSocket": false,
    "remote.SSH.connectTimeout": 120,

Add to .ssh/config:

  Host cuda-quantum-sif~*
    RemoteCommand /home/bheim/.local/singularity/bin/singularity shell /home/bheim/cuda-quantum.sif
    RequestTTY yes

  Host 127.0.0.1 cuda-quantum-sif~127.0.0.1
    HostName 127.0.0.1
    User bheim

VS Code using tunnels
++++++++++++++++++++++++++++++++++++

The CUDA Quantum docker image contains an installation of the 
`VS Code CLI <https://code.visualstudio.com/docs/editor/command-line>`__ to enable
`remote access via tunnel <https://code.visualstudio.com/blogs/2022/12/07/remote-even-better>`__.
This allows to connect either a local installation of `VS Code <https://code.visualstudio.com/>`_, 
or the `VS Code Web UI <https://vscode.dev/>`__, to a running CUDA Quantum container on the same or a different machine. 

Creating a secure connection requires authenticating with the same Github or Microsoft account on each end.
Once authenticated, an SSH connection over the tunnel provides end-to-end encryption. To create a tunnel, 
execute the command 

.. code-block:: console

    code tunnel --name cuda-quantum-remote --accept-server-license-terms

in the running CUDA Quantum container, and follow the instructions to authenticate.
You can then either `open VS Code in a web browser <https://vscode.dev/tunnel/cuda-quantum-remote/home/cudaq/>`__, or connect a local installation of VS Code.
To connect a local installation of VS Code, make sure you have the `Remote - Tunnels <https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-server>`__ extension installed, then open the Command Pallette with `Ctrl+Shift+P`, select "Remote Tunnels: Connect to Tunnel", and enter `cuda-quantum-remote`. After the window reloaded, select "File: Open Folder" in the Command Pallette to open the `/home/cudaq/` folder.

You should see a pop up asking if you want to install the recommended extensions. Selecting to install them will
configure VS Code with extensions for working with C++, Python, and Jupyter.
You can always see the list of recommended extensions that aren't installed yet by clicking on the "Extensions" icon in the sidebar and navigating to the "Recommended" tab.


Use DGX Cloud
++++++++++++++++++++++++++++++++++++

Additional CUDA Tools
------------------------------------

CUDA Quantum makes use of CUDA tools in certain backends and components. 
If you install CUDA Quantum via `PyPI <https://pypi.org/project/cuda-quantum>`__, please follow the installation instructions there to install the necessary CUDA dependencies.
If you are using the CUDA Quantum container image, the image already contains all necessary runtime libraries to use all CUDA Quantum components. It does not, 
however, contain all development dependencies for CUDA, such as, for example the `nvcc` compiler. You can install all CUDA development dependencies by running the command 

.. code-block:: console

    sudo apt-get install cuda-toolkit-11.8

inside the container. Note that most Python packages that use GPU-acceleration, such as for example `CuPy <https://cupy.dev>`__, require an existing CUDA installation. After installing the `cuda-toolkit-11.8` you can install CuPy with the command

.. code-block:: console

    python3 -m pip install cupy-cuda11x


.. **************************


Docker Image
--------------------


Use CUDA Quantum in VS Code
+++++++++++++++++++++++++++++++++++++

If you have `VS Code`_ installed, you can use it to work inside your container.
To do so, install the `Dev Containers extension`_:

.. image:: _static/devContainersExtension.png 

Follow the steps :ref:`above<use-cuda-quantum-in-terminal>` to start the container. 
Open VS Code and navigate to the Remote Explorer. You should see the running cuda-quantum development container listed there.

.. image:: _static/attachToDevContainer.png 

Click on :code:`Attach to Container`. A new VS Code instance will open in that container. Open the `/home/cudaq`
folder to see the `README.md` file and the CUDA Quantum examples that are included in the container. To run the examples, 
open a terminal by going to the Terminal menu and select :code:`New Terminal`. 

.. image:: _static/openTerminal.png 

You can now compile and run the C++ examples using the :code:`nvq++` compiler,
or run the Python examples using the Python interpreter.

.. image:: _static/getToWork.png 

.. _VS Code: https://code.visualstudio.com/download
.. _Dev Containers extension: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
.. _command palette: https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette

.. note:: 

  VS Code extensions that you have installed locally, such as e.g. an extension for Jupyter notebooks, 
  may not be automatically active in the container environment. You may need to install your preferred 
  extension in the container environment for all of your development tools to be available.

.. **************************


.. _dependencies-and-compatibility:

Dependencies and Compatibility
--------------------------------

CUDA Quantum can be used to compile and run quantum programs on a CPU-only system, but a GPU is highly recommended and necessary to use the GPU-based simulators, see also :doc:`using/simulators`.

The supported CPUs include x86_64 (x86-64-v3 architecture and newer) and ARM64 architectures.

.. note:: 

  Some of the components included in the CUDA Quantum Python wheels depend on an existing CUDA installation on your system. For more information about installing the CUDA Quantum Python wheels, take a look at :ref:`this section <install-python-wheels>`.

The following table summarizes the required components.

.. list-table:: Supported Systems
    :widths: 30 50
    :header-rows: 0

    * - CPU architectures
      - x86_64, ARM64
    * - Operating System
      - Linux
    * - Tested Distributions
      - CentOS 8; Debian 11, 12; Fedora 38; OpenSUSE/SELD/SLES 15.5; RHEL 8, 9; Rocky 8, 9; Ubuntu 22.04
    * - Python versions
      - 3.8+

.. list-table:: Requirements for GPU Simulation
    :widths: 30 50
    :header-rows: 0

    * - GPU Architectures
      - Volta, Turing, Ampere, Ada, Hopper
    * - NVIDIA GPU with Compute Capability
      - 7.0+
    * - CUDA
      - 11.x (Driver 470.57.02+), 12.x (Driver 525.60.13+)

Detailed information about supported drivers for different CUDA versions and be found `here <https://docs.nvidia.com/deploy/cuda-compatibility/>`__.


Next Steps
----------

The Docker image contains a folder with example in the :code:`/home/cudaq` directory. These examples are provided to 
get you started with CUDA Quantum and understanding the programming and execution model. 
If you are not using the Docker image, you can find these examples on our `GitHub repository <https://github.com/NVIDIA/cuda-quantum>`__.

Start of by trying to compile a simple one, like :code:`examples/cpp/basics/static_kernel.cpp`:

.. code-block:: console 

    nvq++ examples/cpp/basics/static_kernel.cpp 
    ./a.out

If you have GPU support (e.g. you successfully provided :code:`--gpus` to your docker 
run command), try out the 30 qubit version of this example.

.. code-block:: console 

    nvq++ examples/cpp/basics/cuquantum_backends.cpp --target nvidia 
    ./a.out 

For more information about developing and running CUDA Quantum code, take a look at the page :doc:`Using CUDA Quantum <using/cudaq>`. 
