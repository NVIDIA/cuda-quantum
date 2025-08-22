Superconducting
=================

Anyon Technologies/Anyon Computing
+++++++++++++++++++++++++++++++++++

.. _anyon-backend:

Setting Credentials
```````````````````

Programmers of CUDA-Q may access the Anyon API from either
C++ or Python. Anyon requires a credential configuration file with username and password. 
The configuration file can be generated as follows, replacing
the ``<username>`` and ``<password>`` in the first line with your Anyon Technologies
account details. The credential in the file will be used by CUDA-Q to login to Anyon quantum services 
and will be updated by CUDA-Q with an obtained API token and refresh token. 
Note, the credential line will be deleted in the updated configuration file. 

.. code:: bash
    
    echo 'credentials: {"username":"<username>","password":"<password>"}' > $HOME/.anyon_config

Users can also login and get the keys manually using the following commands:

.. code:: bash

    # You may need to run: `apt-get update && apt-get install curl jq`
    curl -X POST --user "<username>:<password>"  -H "Content-Type: application/json" \
    https://api.anyon.cloud:5000/login > credentials.json
    id_token=`cat credentials.json | jq -r '."id_token"'`
    refresh_token=`cat credentials.json | jq -r '."refresh_token"'`
    echo "key: $id_token" > ~/.anyon_config
    echo "refresh: $refresh_token" >> ~/.anyon_config

The path to the configuration can be specified as an environment variable:

.. code:: bash

    export CUDAQ_ANYON_CREDENTIALS=$HOME/.anyon_config

Submitting
```````````````````

.. tab:: Python


        The target to which quantum kernels are submitted 
        can be controlled with the ``cudaq.set_target()`` function.

        To execute your kernels using Anyon Technologies backends, specify which machine to submit quantum kernels to
        by setting the :code:`machine` parameter of the target. 
        If :code:`machine` is not specified, the default machine will be ``telegraph-8q``.

        .. code:: python

            cudaq.set_target('anyon', machine='telegraph-8q')

        As shown above, ``telegraph-8q`` is an example of a physical QPU.

        To emulate the Anyon Technologies machine locally, without submitting through the cloud,
        you can also set the ``emulate`` flag to ``True``. This will emit any target 
        specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: python

            cudaq.set_target('anyon', emulate=True)

        The number of shots for a kernel execution can be set through
        the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default,
        the ``shots_count`` is set to 1000.

        .. code:: python 

            cudaq.sample(kernel, shots_count=10000)


.. tab:: C++


        To target quantum kernel code for execution in the Anyon Technologies backends,
        pass the flag ``--target anyon`` to the ``nvq++`` compiler. CUDA-Q will 
        authenticate via the Anyon Technologies REST API using the credential in your configuration file.

        .. code:: bash

            nvq++ --target anyon --<backend-type> <machine> src.cpp ...

        To execute your kernels using Anyon Technologies backends, pass the ``--anyon-machine`` flag to the ``nvq++`` compiler
        as the ``--<backend-type>`` to specify which machine to submit quantum kernels to:

        .. code:: bash

            nvq++ --target anyon --anyon-machine telegraph-8q src.cpp ...

        where ``telegraph-8q`` is an example of a physical QPU (Architecture: Telegraph, Qubit Count: 8).

        Currently, ``telegraph-8q`` and ``berkeley-25q`` are available for access over CUDA-Q.

        To emulate the Anyon Technologies machine locally, without submitting through the cloud,
        you can also pass the ``--emulate`` flag as the ``--<backend-type>`` to ``nvq++``. This will emit any target 
        specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: bash

            nvq++ --target anyon --emulate src.cpp

To see a complete example, take a look at :ref:`Anyon examples <anyon-examples>`.


IQM
+++++++++

.. _iqm-backend:

Support for submissions to IQM is currently under development. 
In particular, two-qubit gates can only be performed on adjacent qubits. For more information, we refer to the respective hardware documentation.
Support for automatically injecting the necessary operations during compilation to execute arbitrary multi-qubit gates will be added in future versions.

Setting Credentials
`````````````````````````

Programmers of CUDA-Q may access the IQM Server from either C++ or Python. Following the `quick start guide <https://iqm-finland.github.io/cortex-cli/readme.html#using-cortex-cli>`__, install `iqm-cortex-cli` and login to initialize the tokens file.
The path to the tokens file can either be passed explicitly via an environment variable or it will be loaded automatically if located in
the default location :code:`~/.cache/iqm-cortex-cli/tokens.json`.

.. code:: bash

    export IQM_TOKENS_FILE="path/to/tokens.json"


    
Submitting
`````````````````````````
    
.. tab:: Python 
    
        The target to which quantum kernels are submitted
        can be controlled with the ``cudaq.set_target()`` function.

        .. code:: python

            cudaq.set_target("iqm", url="https://<IQM Server>/cocos",**{"qpu-architecture": "Crystal_5"})

        To emulate the IQM Server locally, without submitting to the IQM Server,
        you can also set the ``emulate`` flag to ``True``. This will emit any target
        specific compiler diagnostics, before running a noise free emulation.

        .. code:: python

            cudaq.set_target('iqm', emulate=True)

        The number of shots for a kernel execution can be set through
        the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default,
        the ``shots_count`` is set to 1000.

        .. code:: python

            cudaq.sample(kernel, shots_count=10000)
  
.. tab:: C++
    
        To target quantum kernel code for execution on an IQM Server,
        pass the ``--target iqm`` flag to the ``nvq++`` compiler, along with a specified ``--iqm-machine``.

        .. note::
            The ``--iqm-machine`` is  a mandatory argument. This provided architecture must match
            the device architecture that the program has been compiled against. The hardware architecture for a
            specific IQM Server may be checked  via `https://<IQM server>/cocos/quantum-architecture`.

        .. code:: bash

            nvq++ --target iqm --iqm-machine Crystal_5 src.cpp

        Once the binary for a specific IQM QPU architecture is compiled, it can be executed against any IQM Server with the same QPU architecture:

        .. code:: bash

            nvq++ --target iqm --iqm-machine Crystal_5 src.cpp -o program
            IQM_SERVER_URL="https://demo.qc.iqm.fi/cocos" ./program

            # Executing the same program against an IQM Server with a different underlying QPU
            # architecture will result in an error.
            IQM_SERVER_URL="https://<Crystal_20 IQM Server>/cocos" ./program

        To emulate the IQM machine locally, without submitting to the IQM Server,
        you can also pass the ``--emulate`` flag to ``nvq++``. This will emit any target
        specific compiler diagnostics, before running a noise free emulation.

        .. code:: bash

            nvq++ --emulate --target iqm --iqm-machine Crystal_5 src.cpp

To see a complete example, take a look at :ref:`IQM examples <iqm-examples>`.


OQC
++++

.. _oqc-backend:


`Oxford Quantum Circuits <https://oxfordquantumcircuits.com/>`__ (OQC) is currently providing CUDA-Q integration for multiple Quantum Processing Unit types.
The 8 qubit ring topology Lucy device and the 32 qubit Kagome lattice topology Toshiko device are both supported via machine options described below.

Setting Credentials
`````````````````````````

In order to use the OQC devices you will need to register.
Registration is achieved by contacting `oqc_qcaas_support@oxfordquantumcircuits.com`.

Once registered you will be able to authenticate with your ``email`` and ``password``

There are three environment variables that the OQC target will look for during configuration:

1. ``OQC_URL``
2. ``OQC_EMAIL``
3. ``OQC_PASSWORD`` - is mandatory


Submitting
`````````````````````````

.. tab:: Python

        To set which OQC URL, set the :code:`url` parameter.
        To set which OQC email, set the :code:`email` parameter.
        To set which OQC machine, set the :code:`machine` parameter.

        .. code:: python

            import os
            import cudaq
            os.environ['OQC_PASSWORD'] = password
            cudaq.set_target("oqc", url=url, machine="lucy")

        You can then execute a kernel against the platform using the OQC Lucy device

        To emulate the OQC device locally, without submitting through the OQC QCaaS services, you can set the ``emulate`` flag to ``True``.
        This will emit any target specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: python

            cudaq.set_target("oqc", emulate=True)


.. tab:: C++


        To target quantum kernel code for execution on the OQC platform, provide the flag ``--target oqc`` to the ``nvq++`` compiler.

        Users may provide their :code:`email` and :code:`url` as extra arguments

        .. code:: bash

            nvq++ --target oqc --oqc-email <email> --oqc-url <url> src.cpp -o executable

        Where both environment variables and extra arguments are supplied, precedent is given to the extra arguments.
        To run the output, provide the runtime loaded variables and invoke the pre-built executable

        .. code:: bash

           OQC_PASSWORD=<password> ./executable

        To emulate the OQC device locally, without submitting through the OQC QCaaS services, you can pass the ``--emulate`` flag to ``nvq++``.
        This will emit any target specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: bash

            nvq++ --emulate --target oqc src.cpp -o executable

        .. note::

            The oqc target supports a ``--oqc-machine`` option.
            The default is the 8 qubit Lucy device.
            You can set this to be either ``toshiko`` or ``lucy`` via this flag.

.. note::

    The OQC quantum assembly toolchain (qat) which is used to compile and execute instructions can be found on github as `oqc-community/qat <https://github.com/oqc-community/qat>`__

To see a complete example, take a look at :ref:`OQC examples <oqc-examples>`.


Quantum Circuits, Inc.
+++++++++++++++++++++++

.. _qci-backend:

As part of its first phase of integration with CUDA-Q,
`Quantum Circuits <https://quantumcircuits.com/>`__ offers users the ability to simulate CUDA-Q
programs using its simulator, AquSim. AquSim is the first simulator that models error detection and
real-time control of Quantum Circuits' Dual-Rail Cavity Qubit systems, and uses a Monte Carlo
approach to do so on a shot-by-shot basis.

In this first phase, the supported features include all of the single and two-qubit gates offered by
CUDA-Q, together with real-time conditional logic enabled by feed-forward capability. AquSim is
currently wired to support ideal simulations only and noise models will be added in future
iterations.

With C++ and Python programming supported, users are able to prototype, test and explore quantum
applications in CUDA-Q in preparation for upcoming releases targeting Quantum Circuits QPUs.
Examples are provided to get started.

Users who wish to get started with running CUDA-Q on AquSim should visit our
`Explore <https://quantumcircuits.com/explore/>`__ page to learn more about the Quantum Circuits
Strategic Quantum Release Program.

Submitting
`````````````````````````

Until CUDA-Q release 0.13.0 is available, the integration with Quantum Circuits will be supported
through the nightly build Docker images.

Instructions on how to install and get started with CUDA-Q using Docker can be found :ref:`here <install-docker-image>`.

You may present your user token to Quantum Circuits via CUDA-Q by setting an environment variable
named :code:`QCI_AUTH_TOKEN` before running your CUDA-Q program.

.. code:: bash

    export QCI_AUTH_TOKEN="example-token"


.. tab:: Python

        To set the target to Quantum Circuits, add the following to your Python
        program:

        .. code:: python

            cudaq.set_target('qci')
            [... your Python here]

        To run on AquSim, simply execute the script using your Python interpreter.

.. tab:: C++

        When executing programs in C++, they must first be compiled using the
        CUDA-Q nvq++ compiler, and then submitted to run on AquSim.

        Note that your token is fetched from your environment at run time, not at compile time.

        In the example below, the compilation step shows two flags being passed to the nvq++
        compiler: the Quantum Circuits target :code:`--target qci`, and the output file
        :code:`-o example.x`.  The second line executes the program against AquSim. Here are the
        shell commands in full:

        .. code:: bash

            nvq++ example.cpp --target qci -o example.x
            ./example.x

To see a complete example of using Quantum Circuits' backends, please take a look at the
:ref:`Quantum Circuits examples <quantum-circuits-examples>`.
