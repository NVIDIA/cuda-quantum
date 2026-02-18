Ion Trap
============

IonQ
+++++++

.. _ionq-backend:

Setting Credentials
`````````````````````````

Programmers of CUDA-Q may access the `IonQ Quantum Cloud
<https://cloud.ionq.com/>`__ from either C++ or Python. Generate
an API key from your `IonQ account <https://cloud.ionq.com/>`__ and export
it as an environment variable:

.. code:: bash

  export IONQ_API_KEY="ionq_generated_api_key"


Submitting
`````````````````````````
.. tab:: Python

    First, set the :code:`ionq` backend.

    .. code:: python

        cudaq.set_target('ionq')

    By default, quantum kernel code will be submitted to the IonQ simulator.

    .. note:: 

       A "target" in :code:`cudaq` refers to a quantum compute provider, such as :code:`ionq`.
       However, IonQ's documentation uses the term "target" to refer to specific QPU's themselves.

    To specify which IonQ QPU to use, set the :code:`qpu` parameter.

    .. code:: python

       cudaq.set_target("ionq", qpu="qpu.aria-1")

    where ``qpu.aria-1`` is an example of a physical QPU.

   A list of available QPUs can be found `in the API documentation <https://docs.ionq.com/#tag/jobs>`__. To see which backends are available with your subscription login to your `IonQ account <https://cloud.ionq.com/jobs>`__.

   To emulate the IonQ machine locally, without submitting through the cloud, you can also set the ``emulate`` flag to ``True``. This will emit any target specific compiler diagnostics, before running a noise free emulation.

   .. code:: python

       cudaq.set_target('ionq', emulate=True)

   The number of shots for a kernel execution can be set through the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default, the ``shots_count`` is set to 1000.

   .. code:: python

       cudaq.sample(kernel, shots_count=10000)


.. tab:: C++

        To target quantum kernel code for execution in the IonQ Cloud,
        pass the flag ``--target ionq`` to the ``nvq++`` compiler.

        .. code:: bash

            nvq++ --target ionq src.cpp

        This will take the API key and handle all authentication with, and submission to, the IonQ QPU(s). By default, quantum kernel code will be submitted to the IonQsimulator.

        .. note:: 

                A "target" in :code:`cudaq` refers to a quantum compute provider, such as :code:`ionq`.
                However, IonQ's documentation uses the term "target" to refer to specific QPU's themselves.

        To execute your kernels on a QPU, pass the ``--ionq-machine`` flag to the ``nvq++`` compiler to specify which machine to submit quantum kernels to:

        .. code:: bash

                nvq++ --target ionq --ionq-machine qpu.aria-1 src.cpp ...

        where ``qpu.aria-1`` is an example of a physical QPU.

        A list of available QPUs can be found `in the API documentation <https://docs.ionq.com/#tag/jobs>`__. To see which backends are available  with your subscription login to your `IonQ account <https://cloud.ionq.com/jobs>`__.

        To emulate the IonQ machine locally, without submitting through the cloud, you can also pass the ``--emulate`` flag to ``nvq++``. This will emit any target  specific compiler diagnostics, before running a noise free emulation.

        .. code:: bash

                nvq++ --emulate --target ionq src.cpp

To see a complete example, take a look at :ref:`IonQ examples <ionq-examples>`.

Quantinuum
+++++++++++

.. _quantinuum-backend:

Quantinuum Nexus is a cloud-based platform that enables users to seamlessly run, review, and collaborate on quantum computing projects.
Access to the Quantinuum Nexus is available through `this website <https://nexus.quantinuum.com/>`__ and documentation can be found `here <https://docs.quantinuum.com/nexus/>`__.

Setting Credentials
```````````````````

Programmers of CUDA-Q may access the Quantinuum API from either
C++ or Python. Quantinuum requires a credential configuration file. 
The configuration file can be generated as follows, replacing
the ``email`` and ``credentials`` in the first line with your Quantinuum
account details.

.. code:: bash

    # You may need to run: `apt-get update && apt-get install curl`
    curl -c $HOME/.quantinuum_cookies.txt -X POST https://nexus.quantinuum.com/auth/login \
    -H "Content-Type: application/json" -d '{ "email":"<your_alias>@email.com","password":"<your_password>" }' >/dev/null
    awk '$6 == "myqos_oat" {refresh=$7} $6 == "myqos_id" {key=$7} END {print "key: " key "\nrefresh: " refresh}' $HOME/.quantinuum_cookies.txt > $HOME/.quantinuum_config
    rm $HOME/.quantinuum_cookies.txt

The path to the configuration can be specified as an environment variable:

.. code:: bash

    export CUDAQ_QUANTINUUM_CREDENTIALS=$HOME/.quantinuum_config


Submitting
`````````````````````````

Each job submitted to the Quantinuum Nexus is associated with a `project <https://docs.quantinuum.com/nexus/user_guide/concepts/projects.html>`__.
Create a project in the Nexus portal. You can find the project ID in the URL of the project page, or you may specify project with its name.


.. tab:: Python

       
        The backend to which quantum kernels are submitted 
        can be controlled with the ``cudaq.set_target()`` function.

        .. code:: python

            cudaq.set_target('quantinuum', project='nexus_project_name')
            # or
            cudaq.set_target('quantinuum', project='nexus_project_id')

        By default, quantum kernel code will be submitted to the Quantinuum syntax checker.
        Submission to the syntax checker merely validates the program; the kernels are not executed.

        To execute your kernels, specify which machine to submit quantum kernels to
        by setting the :code:`machine` parameter of the target.

        .. code:: python

            cudaq.set_target('quantinuum', machine='H2-2')

        where ``H2-2`` is an example of a physical QPU. Hardware specific
        emulators may be accessed by appending an ``E`` to the end (e.g, ``H2-2E``). For 
        access to the syntax checker for the provided machine, you may append an ``SC`` 
        to the end (e.g, ``H2-1SC``).

        For a comprehensive list of available machines, login to your `Quantinuum Nexus user account <https://nexus.quantinuum.com/>`__ 
        and navigate to the "Profile" tab, where you should find a table titled "Quantinuum Systems Access".

        To emulate the Quantinuum machine locally, without submitting through the cloud,
        you can set the ``emulate`` flag to ``True``. This will emit any target 
        specific compiler warnings and diagnostics, before running a noise free emulation.
        You do not need to specify project or machine when emulating.

        .. code:: python

            cudaq.set_target('quantinuum', emulate=True)

        The number of shots for a kernel execution can be set through
        the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default,
        the ``shots_count`` is set to 1000.

        .. code:: python 

            cudaq.sample(kernel, shots_count=10000)


.. tab:: C++

        To target quantum kernel code for execution in the Quantinuum backends,
        pass the flag ``--target quantinuum`` to the ``nvq++`` compiler. CUDA-Q will 
        authenticate via the Quantinuum REST API using the credential in your configuration file.
        By default, quantum kernel code will be submitted to the Quantinuum syntax checker.
        Submission to the syntax checker merely validates the program; the kernels are not executed.

        .. code:: bash

            nvq++ --target quantinuum src.cpp --quantinuum-project nexus_project_name ...
            # or
            nvq++ --target quantinuum src.cpp --quantinuum-project nexus_project_id ...

        To execute your kernels, pass the ``--quantinuum-machine`` flag to the ``nvq++`` compiler
        to specify which machine to submit quantum kernels to:

        .. code:: bash

            nvq++ --target quantinuum --quantinuum-machine H2-2 src.cpp ...

        where ``H2-2`` is an example of a physical QPU. Hardware specific
        emulators may be accessed by appending an ``E`` to the end (e.g, ``H2-2E``). For 
        access to the syntax checker for the provided machine, you may append an ``SC`` 
        to the end (e.g, ``H2-1SC``).

        For a comprehensive list of available machines, login to your `Quantinuum Nexus user account <https://nexus.quantinuum.com/>`__ 
        and navigate to the "Profile" tab, where you should find a table titled "Quantinuum Systems Access".

        To emulate the Quantinuum machine locally, without submitting through the cloud,
        you can pass the ``--emulate`` flag to ``nvq++``. This will emit any target 
        specific compiler warnings and diagnostics, before running a noise free emulation.
        You do not need to specify project or machine when emulating.

        .. code:: bash

            nvq++ --emulate --target quantinuum src.cpp

.. note:: 

       Quantinuum's syntax checker for Helios (e.g., ``Helios-1SC``) only performs QIR code validation and does not return any results.
       Thus, it always returns an empty result set. This is different from other Quantinuum backends (e.g., ``H2-1SC``) where the syntax checker returns dummy results.
       As a result, when using the Helios syntax checker, we may receive this warning message:

        .. code:: text
    
                WARNING: this kernel invocation produced 0 shots worth of results when executed. 

        It means that the kernel was successfully validated, but no execution results are available.
        To get results, please submit to the Helios emulator (e.g., ``Helios-1E``) or the actual quantum device (e.g., ``Helios-1``).



To see a complete example, take a look at :ref:`Quantinuum examples <quantinuum-examples>`.

.. note:: 

        In local emulation mode (``emulate`` flag set to ``True``), the program will be executed on the :ref:`default simulator <default-simulator>`.
        The environment variable ``CUDAQ_DEFAULT_SIMULATOR`` can be used to change the emulation simulator. 
        
        For example, the simulation floating point accuracy and/or the simulation capabilities (e.g., maximum number of qubits, supported quantum gates),
        depend on the selected simulator.  
        
        Any environment variables must be set prior to setting the target or running "`import cudaq`".
