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

The `IQM Resonance <https://meetiqm.com/products/iqm-resonance/>`__ portal offers access to various different IQM quantum computers.
The machines available will be constantly extended as development progresses.
Programmers of CUDA-Q may access the IQM Server from either C++ or Python.

With this version it is no longer necessary to define the target QPU architecture in the code or at compile-time.
The IQM backend integration now contacts at runtime the configured IQM server and fetches the active dynamic quantum architecture of the QPU.
This is then used as input to transpile the quantum kernel code just-in-time for the target QPU topology.
By setting the environment variable ``IQM_SERVER_URL`` the target server can be selected just when executing the program.
As result the python script or the compiled C++ program can be executed on different QPUs without any changes to the code.

For IQM architectures two-qubit gates can only be performed on adjacent qubits. For more information, we refer to the respective hardware documentation.
Support for automatically injecting the necessary operations during compilation to execute arbitrary multi-qubit gates will be added in future versions.

Setting Credentials
`````````````````````````

Create a free account on the `IQM Resonance <https://meetiqm.com/products/iqm-resonance/>`__ portal and log-in.
Navigate to the account profile (top right). There generate an "API Token" and copy the generated token-string.
Set the environment variable ``IQM_TOKEN`` to contain the value of the token-string.
The IQM backend integration will use this as authorization token at the IQM server.

Note: The previously used ``IQM_TOKENS_FILE`` environment variable can still be used to point to a tokens file but will be ignored if the ``IQM_TOKEN`` variable is set.
The tokens file cannot be generated by the ``iqmclient`` tool anymore but needs to be created manually using the token obtained from the Resonance profile page.
A token file can be created and the environment variable set like this:

.. code:: bash

    echo '{ "access_token": "<put-your-token-here>" }' > resonance-token.json
    export IQM_TOKENS_FILE="path/to/resonance-tokens.json"

Please find more documentation after logging in to the IQM Resonance portal.

Submitting
`````````````````````````

.. tab:: Python

        The target to which quantum kernels are submitted
        can be controlled with the ``cudaq.set_target()`` function.

        .. code:: python

            cudaq.set_target("iqm", url="https://<IQM Server>/")

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

        To see a complete example for using IQM server backends, take a look at our :doc:`Python examples<../../examples/examples>`.


.. tab:: C++

        To target quantum kernel code for execution on an IQM Server,
        pass the ``--target iqm`` flag to the ``nvq++`` compiler.

        .. code:: bash

            nvq++ --target iqm src.cpp

        Once the binary for an IQM QPU is compiled, it can be executed against any IQM Server:

        .. code:: bash

            nvq++ --target iqm src.cpp -o program
            IQM_SERVER_URL="https://demo.qc.iqm.fi/" ./program

        To emulate the IQM machine locally, without submitting to the IQM Server,
        you can also pass the ``--emulate`` flag to ``nvq++``. This will emit any target
        specific compiler diagnostics, before running a noise free emulation.

        .. code:: bash

            nvq++ --emulate --target iqm src.cpp

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
