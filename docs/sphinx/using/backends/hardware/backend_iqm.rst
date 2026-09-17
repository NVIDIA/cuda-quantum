IQM Backend Advanced Use Cases
==============================

This page describes advanced uses cases supported by the IQM backend integration.


Configuring the backend
+++++++++++++++++++++++

The IQM backend can be configured either in the code (Python), at compile time (C++), or through the environment in which the process runs.

The following settings can be configured:

- To which IQM quantum computer a job is sent by setting `IQM Server URL` plus `IQM Quantum Computer`.
- The API token for authorization at the IQM server.
- Different QPU architectures for testing.
- The use of emulation mode.

+-----------------------+----------------------+-----------------------------------+----------------------------+
| Setting               | Environment          | Python                            | C++                        |
|                       | (variable name)      | (parameter to cudaq.set_target()) | (option to nvq++)          |
+=======================+======================+===================================+============================+
| IQM Server URL        | IQM_SERVER_URL       | ``url``                           | ``--iqm-server-url``       |
+-----------------------+----------------------+-----------------------------------+----------------------------+
| IQM Quantum Computer  | IQM_QC               | ``qc``                            | ``--iqm-quantum-computer`` |
+-----------------------+----------------------+-----------------------------------+----------------------------+
| API token             | IQM_TOKEN            |                                   |                            |
+-----------------------+----------------------+-----------------------------------+----------------------------+
| Token file            | IQM_TOKENS_FILE      |                                   |                            |
| (deprecated)          |                      |                                   |                            |
+-----------------------+----------------------+-----------------------------------+----------------------------+
| load QPU architecture | IQM_QPU_QA           | ``mapping_file``                  |                            |
+-----------------------+----------------------+-----------------------------------+----------------------------+
| save QPU architecture | IQM_SAVE_QPU_QA      |                                   |                            |
+-----------------------+----------------------+-----------------------------------+----------------------------+
| Emulation mode        |                      | ``emulate``                       | ``--emulate``              |
+-----------------------+----------------------+-----------------------------------+----------------------------+

Please note that any value in an environment variable takes precedence over any value for the same setting in the code or at compile time.

Examples:
"""""""""

    .. tab:: Environment

        .. code:: bash

            IQM_TOKEN="your personal API token" IQM_SERVER_URL="https://resonance.iqm.tech/" IQM_QC="garnet" python3 program.py

        .. code:: bash

            export IQM_TOKEN="your personal API token"
            export IQM_SERVER_URL="https://resonance.iqm.tech/"
            export IQM_QC="garnet"
            python3 program.py

    .. tab:: Python

        .. code:: python

            cudaq.set_target('iqm', url="https://resonance.iqm.tech/", qc="garnet")

        .. code:: python

            cudaq.set_target('iqm', mapping_file="<path+filename>")

    .. tab:: C++

        .. code:: bash

            nvq++ --target iqm --iqm-server-url="https://resonance.iqm.tech" --iqm-quantum-computer="garnet" src.cpp -o program


Emulation Mode
++++++++++++++

.. tab:: Python

    To emulate the IQM Server locally, without submitting to the IQM Server, you can set the ``emulate`` flag to ``True``.
    This will emit any target specific compiler diagnostics, before running a noise free emulation.

    .. code:: python

        cudaq.set_target('iqm', emulate=True, url="https://<IQM Server>/", qc="<quantum computer>")

    Emulation mode will still contact the configured IQM Server to retrieve the dynamic quantum architecture resulting from the active calibration unless a QPU architecture file is explicitly specified.
    This can be done by setting `mapping_file` to point to a file describing the QPU architecture which should be emulated.
    If an architecture is specified no server URL is needed anymore.

    .. code:: python

        cudaq.set_target('iqm', emulate=True, mapping_file="<path+filename>")

    The QPU quantum architecture of a test with a real life IQM QPU can be saved for later use in emulation runs.
    To do so the environment variable ``IQM_SAVE_QPU_QA`` must be set to point to a filename in addition to setting the URL of a Resonance server.
    The test can even run as emulation as long as a server URL is given to retrieve the current dynamic quantum architecture from.

    .. code:: bash

        IQM_SERVER_URL="https://resonance.iqm.tech/" IQM_QC="<quantum computer>" IQM_SAVE_QPU_QA="<path+filename for QPU architecture file>" python3 program.py


    The file will be created with the given name. If the file already exists the execution is aborted with an error.


.. tab:: C++

    To emulate the IQM machine locally, without submitting to the IQM Server, you can pass the ``--emulate`` option to ``nvq++``.
    This will emit any target specific compiler diagnostics, before running a noise free emulation.

    .. code:: bash

        nvq++ --target iqm --emulate src.cpp -o program
        IQM_SERVER_URL="https://resonance.iqm.tech/" IQM_QC="<quantum computer>" ./program

    Emulation mode will still contact the configured IQM Server to retrieve the dynamic quantum architecture resulting from the active calibration unless a QPU architecture file is explicitly specified.
    This can be done by specifying a file with the architecture either at compile time or in an variable in the environment executing the binary.
    If an architecture is specified no server URL is needed anymore.

    .. code:: bash

        # With this binary multiple QPU architectures can be tested without recompilation.
        nvq++ --target iqm --emulate src.cpp -o program
        IQM_QPU_QA="<path+filename of QPU architecture file>" ./program

    .. code:: bash

        # This binary will use the given QPU architecture file until overwritten by environment variable "IQM_QPU_QA".
        nvq++ --target iqm --emulate --mapping-file <path+filename of QPU architecture file> src.cpp -o program
        ./program

    The QPU architecture of a test with an IQM server can be saved for later use in emulation runs.
    To do so the environment variable ``IQM_SAVE_QPU_QA`` must be set to point to a filename in addition to setting the URL of the Resonance server.
    The test can even run as emulation as long as a server URL is given to retrieve the current dynamic quantum architecture from.

    .. code:: bash

        nvq++ --target iqm --emulate src.cpp -o program
        IQM_SERVER_URL="https://resonance.iqm.tech/" IQM_QC="<quantum computer>" IQM_SAVE_QPU_QA="<path+filename for QPU architecture file>" ./program


The folder ``targettests/Target/IQM/`` contains sample QPU architecture files.
Find there files for the IQM Crystal architecture as well as files from real life QPUs which can be found on the IQM Resonance portal.

When no QPU architecture file is specified and the query to the configured IQM Server fails
(for example due to missing authentication or no network access), CUDA-Q logs a warning and leaves the compilation pipeline unresolved.
Kernel launches that do not require qubit mapping, such as ``dem_from_kernel``, will still run successfully with an unresolved compilation pipeline.
On the other hand, kernel launches that require full kernel compilation, such as ``sample`` or ``observe``, will fail at launch.

To see a complete example, take a look at :ref:`IQM examples <iqm-examples>`.


Setting the Number of Shots
+++++++++++++++++++++++++++

.. tab:: Python

        The number of shots for a kernel execution can be set through
        the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default,
        the ``shots_count`` is set to 1000.

        .. code:: python

            cudaq.sample(kernel, shots_count=10000)


Using Credentials Saved in a File
+++++++++++++++++++++++++++++++++

This way of providing the "API Token" is deprecated.
The preferred way to pass the "API Token" to the IQM backend is through the environment variable ``IQM_TOKEN``.
For backward compatibility the earlier used storage of the "API Token" in a file can still be used as follows:

The previously used ``IQM_TOKENS_FILE`` environment variable can still be used to point to a tokens file but will be ignored if the ``IQM_TOKEN`` variable is set.
The tokens file cannot be generated by the ``iqmclient`` tool anymore but can be created manually using the "API Token" obtained from the Resonance profile page.
A tokens file can be created and the environment variable set like this:

.. code:: bash

    echo '{ "access_token": "<put-your-token-here>" }' > resonance-token.json
    export IQM_TOKENS_FILE="path/to/resonance-token.json"

When storing the "API Token" in a file please make sure to restrict access to this file to only the account running tests.
No other user or group on the computer must have any access to this file
